/*
 * Copyright © 2017 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "common/gen_decoder.h"
#include "gen_disasm.h"
#include "util/macros.h"

#include <string.h>

void
gen_batch_decode_ctx_init(struct gen_batch_decode_ctx *ctx,
                          const struct gen_device_info *devinfo,
                          FILE *fp, enum gen_batch_decode_flags flags,
                          const char *xml_path,
                          struct gen_batch_decode_bo (*get_bo)(void *,
                                                               uint64_t),
                          unsigned (*get_state_size)(void *, uint32_t),
                          void *user_data)
{
   memset(ctx, 0, sizeof(*ctx));

   ctx->get_bo = get_bo;
   ctx->get_state_size = get_state_size;
   ctx->user_data = user_data;
   ctx->fp = fp;
   ctx->flags = flags;
   ctx->max_vbo_decoded_lines = -1; /* No limit! */

   if (xml_path == NULL)
      ctx->spec = gen_spec_load(devinfo);
   else
      ctx->spec = gen_spec_load_from_path(devinfo, xml_path);
   ctx->disasm = gen_disasm_create(devinfo);
}

void
gen_batch_decode_ctx_finish(struct gen_batch_decode_ctx *ctx)
{
   gen_spec_destroy(ctx->spec);
   gen_disasm_destroy(ctx->disasm);
}

#define CSI "\e["
#define RED_COLOR    CSI "31m"
#define BLUE_HEADER  CSI "0;44m"
#define GREEN_HEADER CSI "1;42m"
#define NORMAL       CSI "0m"

static void
ctx_print_group(struct gen_batch_decode_ctx *ctx,
                struct gen_group *group,
                uint64_t address, const void *map)
{
   gen_print_group(ctx->fp, group, address, map, 0,
                   (ctx->flags & GEN_BATCH_DECODE_IN_COLOR) != 0);
}

static struct gen_batch_decode_bo
ctx_get_bo(struct gen_batch_decode_ctx *ctx, uint64_t addr)
{
   if (gen_spec_get_gen(ctx->spec) >= gen_make_gen(8,0)) {
      /* On Broadwell and above, we have 48-bit addresses which consume two
       * dwords.  Some packets require that these get stored in a "canonical
       * form" which means that bit 47 is sign-extended through the upper
       * bits. In order to correctly handle those aub dumps, we need to mask
       * off the top 16 bits.
       */
      addr &= (~0ull >> 16);
   }

   struct gen_batch_decode_bo bo = ctx->get_bo(ctx->user_data, addr);

   if (gen_spec_get_gen(ctx->spec) >= gen_make_gen(8,0))
      bo.addr &= (~0ull >> 16);

   /* We may actually have an offset into the bo */
   if (bo.map != NULL) {
      assert(bo.addr <= addr);
      uint64_t offset = addr - bo.addr;
      bo.map += offset;
      bo.addr += offset;
      bo.size -= offset;
   }

   return bo;
}

static int
update_count(struct gen_batch_decode_ctx *ctx,
             uint32_t offset_from_dsba,
             unsigned element_dwords,
             unsigned guess)
{
   unsigned size = 0;

   if (ctx->get_state_size)
      size = ctx->get_state_size(ctx->user_data, offset_from_dsba);

   if (size > 0)
      return size / (sizeof(uint32_t) * element_dwords);

   /* In the absence of any information, just guess arbitrarily. */
   return guess;
}

static void
ctx_disassemble_program(struct gen_batch_decode_ctx *ctx,
                        uint32_t ksp, const char *type)
{
   uint64_t addr = ctx->instruction_base + ksp;
   struct gen_batch_decode_bo bo = ctx_get_bo(ctx, addr);
   if (!bo.map)
      return;

   fprintf(ctx->fp, "\nReferenced %s:\n", type);
   gen_disasm_disassemble(ctx->disasm, bo.map, 0, ctx->fp);
}

/* Heuristic to determine whether a uint32_t is probably actually a float
 * (http://stackoverflow.com/a/2953466)
 */

static bool
probably_float(uint32_t bits)
{
   int exp = ((bits & 0x7f800000U) >> 23) - 127;
   uint32_t mant = bits & 0x007fffff;

   /* +- 0.0 */
   if (exp == -127 && mant == 0)
      return true;

   /* +- 1 billionth to 1 billion */
   if (-30 <= exp && exp <= 30)
      return true;

   /* some value with only a few binary digits */
   if ((mant & 0x0000ffff) == 0)
      return true;

   return false;
}

static void
ctx_print_buffer(struct gen_batch_decode_ctx *ctx,
                 struct gen_batch_decode_bo bo,
                 uint32_t read_length,
                 uint32_t pitch,
                 int max_lines)
{
   const uint32_t *dw_end = bo.map + MIN2(bo.size, read_length);

   int column_count = 0, line_count = -1;
   for (const uint32_t *dw = bo.map; dw < dw_end; dw++) {
      if (column_count * 4 == pitch || column_count == 8) {
         fprintf(ctx->fp, "\n");
         column_count = 0;
         line_count++;

         if (max_lines >= 0 && line_count >= max_lines)
            break;
      }
      fprintf(ctx->fp, column_count == 0 ? "  " : " ");

      if ((ctx->flags & GEN_BATCH_DECODE_FLOATS) && probably_float(*dw))
         fprintf(ctx->fp, "  %8.2f", *(float *) dw);
      else
         fprintf(ctx->fp, "  0x%08x", *dw);

      column_count++;
   }
   fprintf(ctx->fp, "\n");
}

static void
handle_state_base_address(struct gen_batch_decode_ctx *ctx, const uint32_t *p)
{
   struct gen_group *inst = gen_spec_find_instruction(ctx->spec, p);

   struct gen_field_iterator iter;
   gen_field_iterator_init(&iter, inst, p, 0, false);

   uint64_t surface_base = 0, dynamic_base = 0, instruction_base = 0;
   bool surface_modify = 0, dynamic_modify = 0, instruction_modify = 0;

   while (gen_field_iterator_next(&iter)) {
      if (strcmp(iter.name, "Surface State Base Address") == 0) {
         surface_base = iter.raw_value;
      } else if (strcmp(iter.name, "Dynamic State Base Address") == 0) {
         dynamic_base = iter.raw_value;
      } else if (strcmp(iter.name, "Instruction Base Address") == 0) {
         instruction_base = iter.raw_value;
      } else if (strcmp(iter.name, "Surface State Base Address Modify Enable") == 0) {
         surface_modify = iter.raw_value;
      } else if (strcmp(iter.name, "Dynamic State Base Address Modify Enable") == 0) {
         dynamic_modify = iter.raw_value;
      } else if (strcmp(iter.name, "Instruction Base Address Modify Enable") == 0) {
         instruction_modify = iter.raw_value;
      }
   }

   if (dynamic_modify)
      ctx->dynamic_base = dynamic_base;

   if (surface_modify)
      ctx->surface_base = surface_base;

   if (instruction_modify)
      ctx->instruction_base = instruction_base;
}

static void
dump_binding_table(struct gen_batch_decode_ctx *ctx, uint32_t offset, int count)
{
   struct gen_group *strct =
      gen_spec_find_struct(ctx->spec, "RENDER_SURFACE_STATE");
   if (strct == NULL) {
      fprintf(ctx->fp, "did not find RENDER_SURFACE_STATE info\n");
      return;
   }

   if (count < 0)
      count = update_count(ctx, offset, 1, 8);

   if (offset % 32 != 0 || offset >= UINT16_MAX) {
      fprintf(ctx->fp, "  invalid binding table pointer\n");
      return;
   }

   struct gen_batch_decode_bo bind_bo =
      ctx_get_bo(ctx, ctx->surface_base + offset);

   if (bind_bo.map == NULL) {
      fprintf(ctx->fp, "  binding table unavailable\n");
      return;
   }

   const uint32_t *pointers = bind_bo.map;
   for (int i = 0; i < count; i++) {
      if (pointers[i] == 0)
         continue;

      uint64_t addr = ctx->surface_base + pointers[i];
      struct gen_batch_decode_bo bo = ctx_get_bo(ctx, addr);
      uint32_t size = strct->dw_length * 4;

      if (pointers[i] % 32 != 0 ||
          addr < bo.addr || addr + size >= bo.addr + bo.size) {
         fprintf(ctx->fp, "pointer %u: 0x%08x <not valid>\n", i, pointers[i]);
         continue;
      }

      fprintf(ctx->fp, "pointer %u: 0x%08x\n", i, pointers[i]);
      ctx_print_group(ctx, strct, addr, bo.map + (addr - bo.addr));
   }
}

static void
dump_samplers(struct gen_batch_decode_ctx *ctx, uint32_t offset, int count)
{
   struct gen_group *strct = gen_spec_find_struct(ctx->spec, "SAMPLER_STATE");

   if (count < 0)
      count = update_count(ctx, offset, strct->dw_length, 4);

   uint64_t state_addr = ctx->dynamic_base + offset;
   struct gen_batch_decode_bo bo = ctx_get_bo(ctx, state_addr);
   const void *state_map = bo.map;

   if (state_map == NULL) {
      fprintf(ctx->fp, "  samplers unavailable\n");
      return;
   }

   if (offset % 32 != 0 || state_addr - bo.addr >= bo.size) {
      fprintf(ctx->fp, "  invalid sampler state pointer\n");
      return;
   }

   for (int i = 0; i < count; i++) {
      fprintf(ctx->fp, "sampler state %d\n", i);
      ctx_print_group(ctx, strct, state_addr, state_map);
      state_addr += 16;
      state_map += 16;
   }
}

static void
handle_media_interface_descriptor_load(struct gen_batch_decode_ctx *ctx,
                                       const uint32_t *p)
{
   struct gen_group *inst = gen_spec_find_instruction(ctx->spec, p);
   struct gen_group *desc =
      gen_spec_find_struct(ctx->spec, "INTERFACE_DESCRIPTOR_DATA");

   struct gen_field_iterator iter;
   gen_field_iterator_init(&iter, inst, p, 0, false);
   uint32_t descriptor_offset = 0;
   int descriptor_count = 0;
   while (gen_field_iterator_next(&iter)) {
      if (strcmp(iter.name, "Interface Descriptor Data Start Address") == 0) {
         descriptor_offset = strtol(iter.value, NULL, 16);
      } else if (strcmp(iter.name, "Interface Descriptor Total Length") == 0) {
         descriptor_count =
            strtol(iter.value, NULL, 16) / (desc->dw_length * 4);
      }
   }

   uint64_t desc_addr = ctx->dynamic_base + descriptor_offset;
   struct gen_batch_decode_bo bo = ctx_get_bo(ctx, desc_addr);
   const void *desc_map = bo.map;

   if (desc_map == NULL) {
      fprintf(ctx->fp, "  interface descriptors unavailable\n");
      return;
   }

   for (int i = 0; i < descriptor_count; i++) {
      fprintf(ctx->fp, "descriptor %d: %08x\n", i, descriptor_offset);

      ctx_print_group(ctx, desc, desc_addr, desc_map);

      gen_field_iterator_init(&iter, desc, desc_map, 0, false);
      uint64_t ksp = 0;
      uint32_t sampler_offset = 0, sampler_count = 0;
      uint32_t binding_table_offset = 0, binding_entry_count = 0;
      while (gen_field_iterator_next(&iter)) {
         if (strcmp(iter.name, "Kernel Start Pointer") == 0) {
            ksp = strtoll(iter.value, NULL, 16);
         } else if (strcmp(iter.name, "Sampler State Pointer") == 0) {
            sampler_offset = strtol(iter.value, NULL, 16);
         } else if (strcmp(iter.name, "Sampler Count") == 0) {
            sampler_count = strtol(iter.value, NULL, 10);
         } else if (strcmp(iter.name, "Binding Table Pointer") == 0) {
            binding_table_offset = strtol(iter.value, NULL, 16);
         } else if (strcmp(iter.name, "Binding Table Entry Count") == 0) {
            binding_entry_count = strtol(iter.value, NULL, 10);
         }
      }

      ctx_disassemble_program(ctx, ksp, "compute shader");
      printf("\n");

      dump_samplers(ctx, sampler_offset, sampler_count);
      dump_binding_table(ctx, binding_table_offset, binding_entry_count);

      desc_map += desc->dw_length;
      desc_addr += desc->dw_length * 4;
   }
}

static void
handle_3dstate_vertex_buffers(struct gen_batch_decode_ctx *ctx,
                              const uint32_t *p)
{
   struct gen_group *inst = gen_spec_find_instruction(ctx->spec, p);
   struct gen_group *vbs = gen_spec_find_struct(ctx->spec, "VERTEX_BUFFER_STATE");

   struct gen_batch_decode_bo vb = {};
   uint32_t vb_size = 0;
   int index = -1;
   int pitch = -1;
   bool ready = false;

   struct gen_field_iterator iter;
   gen_field_iterator_init(&iter, inst, p, 0, false);
   while (gen_field_iterator_next(&iter)) {
      if (iter.struct_desc != vbs)
         continue;

      struct gen_field_iterator vbs_iter;
      gen_field_iterator_init(&vbs_iter, vbs, &iter.p[iter.start_bit / 32], 0, false);
      while (gen_field_iterator_next(&vbs_iter)) {
         if (strcmp(vbs_iter.name, "Vertex Buffer Index") == 0) {
            index = vbs_iter.raw_value;
         } else if (strcmp(vbs_iter.name, "Buffer Pitch") == 0) {
            pitch = vbs_iter.raw_value;
         } else if (strcmp(vbs_iter.name, "Buffer Starting Address") == 0) {
            vb = ctx_get_bo(ctx, vbs_iter.raw_value);
         } else if (strcmp(vbs_iter.name, "Buffer Size") == 0) {
            vb_size = vbs_iter.raw_value;
            ready = true;
         } else if (strcmp(vbs_iter.name, "End Address") == 0) {
            if (vb.map && vbs_iter.raw_value >= vb.addr)
               vb_size = vbs_iter.raw_value - vb.addr;
            else
               vb_size = 0;
            ready = true;
         }

         if (!ready)
            continue;

         fprintf(ctx->fp, "vertex buffer %d, size %d\n", index, vb_size);

         if (vb.map == NULL) {
            fprintf(ctx->fp, "  buffer contents unavailable\n");
            continue;
         }

         if (vb.map == 0 || vb_size == 0)
            continue;

         ctx_print_buffer(ctx, vb, vb_size, pitch, ctx->max_vbo_decoded_lines);

         vb.map = NULL;
         vb_size = 0;
         index = -1;
         pitch = -1;
         ready = false;
      }
   }
}

static void
handle_3dstate_index_buffer(struct gen_batch_decode_ctx *ctx,
                            const uint32_t *p)
{
   struct gen_group *inst = gen_spec_find_instruction(ctx->spec, p);

   struct gen_batch_decode_bo ib = {};
   uint32_t ib_size = 0;
   uint32_t format = 0;

   struct gen_field_iterator iter;
   gen_field_iterator_init(&iter, inst, p, 0, false);
   while (gen_field_iterator_next(&iter)) {
      if (strcmp(iter.name, "Index Format") == 0) {
         format = iter.raw_value;
      } else if (strcmp(iter.name, "Buffer Starting Address") == 0) {
         ib = ctx_get_bo(ctx, iter.raw_value);
      } else if (strcmp(iter.name, "Buffer Size") == 0) {
         ib_size = iter.raw_value;
      }
   }

   if (ib.map == NULL) {
      fprintf(ctx->fp, "  buffer contents unavailable\n");
      return;
   }

   const void *m = ib.map;
   const void *ib_end = ib.map + MIN2(ib.size, ib_size);
   for (int i = 0; m < ib_end && i < 10; i++) {
      switch (format) {
      case 0:
         fprintf(ctx->fp, "%3d ", *(uint8_t *)m);
         m += 1;
         break;
      case 1:
         fprintf(ctx->fp, "%3d ", *(uint16_t *)m);
         m += 2;
         break;
      case 2:
         fprintf(ctx->fp, "%3d ", *(uint32_t *)m);
         m += 4;
         break;
      }
   }

   if (m < ib_end)
      fprintf(ctx->fp, "...");
   fprintf(ctx->fp, "\n");
}

static void
decode_single_ksp(struct gen_batch_decode_ctx *ctx, const uint32_t *p)
{
   struct gen_group *inst = gen_spec_find_instruction(ctx->spec, p);

   uint64_t ksp = 0;
   bool is_simd8 = false; /* vertex shaders on Gen8+ only */
   bool is_enabled = true;

   struct gen_field_iterator iter;
   gen_field_iterator_init(&iter, inst, p, 0, false);
   while (gen_field_iterator_next(&iter)) {
      if (strcmp(iter.name, "Kernel Start Pointer") == 0) {
         ksp = iter.raw_value;
      } else if (strcmp(iter.name, "SIMD8 Dispatch Enable") == 0) {
         is_simd8 = iter.raw_value;
      } else if (strcmp(iter.name, "Dispatch Mode") == 0) {
         is_simd8 = strcmp(iter.value, "SIMD8") == 0;
      } else if (strcmp(iter.name, "Dispatch Enable") == 0) {
         is_simd8 = strcmp(iter.value, "SIMD8") == 0;
      } else if (strcmp(iter.name, "Enable") == 0) {
         is_enabled = iter.raw_value;
      }
   }

   const char *type =
      strcmp(inst->name,   "VS_STATE") == 0 ? "vertex shader" :
      strcmp(inst->name,   "GS_STATE") == 0 ? "geometry shader" :
      strcmp(inst->name,   "SF_STATE") == 0 ? "strips and fans shader" :
      strcmp(inst->name, "CLIP_STATE") == 0 ? "clip shader" :
      strcmp(inst->name, "3DSTATE_DS") == 0 ? "tessellation evaluation shader" :
      strcmp(inst->name, "3DSTATE_HS") == 0 ? "tessellation control shader" :
      strcmp(inst->name, "3DSTATE_VS") == 0 ? (is_simd8 ? "SIMD8 vertex shader" : "vec4 vertex shader") :
      strcmp(inst->name, "3DSTATE_GS") == 0 ? (is_simd8 ? "SIMD8 geometry shader" : "vec4 geometry shader") :
      NULL;

   if (is_enabled) {
      ctx_disassemble_program(ctx, ksp, type);
      printf("\n");
   }
}

static void
decode_ps_kernels(struct gen_batch_decode_ctx *ctx, const uint32_t *p)
{
   struct gen_group *inst = gen_spec_find_instruction(ctx->spec, p);

   uint64_t ksp[3] = {0, 0, 0};
   bool enabled[3] = {false, false, false};

   struct gen_field_iterator iter;
   gen_field_iterator_init(&iter, inst, p, 0, false);
   while (gen_field_iterator_next(&iter)) {
      if (strncmp(iter.name, "Kernel Start Pointer ",
                  strlen("Kernel Start Pointer ")) == 0) {
         int idx = iter.name[strlen("Kernel Start Pointer ")] - '0';
         ksp[idx] = strtol(iter.value, NULL, 16);
      } else if (strcmp(iter.name, "8 Pixel Dispatch Enable") == 0) {
         enabled[0] = strcmp(iter.value, "true") == 0;
      } else if (strcmp(iter.name, "16 Pixel Dispatch Enable") == 0) {
         enabled[1] = strcmp(iter.value, "true") == 0;
      } else if (strcmp(iter.name, "32 Pixel Dispatch Enable") == 0) {
         enabled[2] = strcmp(iter.value, "true") == 0;
      }
   }

   /* Reorder KSPs to be [8, 16, 32] instead of the hardware order. */
   if (enabled[0] + enabled[1] + enabled[2] == 1) {
      if (enabled[1]) {
         ksp[1] = ksp[0];
         ksp[0] = 0;
      } else if (enabled[2]) {
         ksp[2] = ksp[0];
         ksp[0] = 0;
      }
   } else {
      uint64_t tmp = ksp[1];
      ksp[1] = ksp[2];
      ksp[2] = tmp;
   }

   if (enabled[0])
      ctx_disassemble_program(ctx, ksp[0], "SIMD8 fragment shader");
   if (enabled[1])
      ctx_disassemble_program(ctx, ksp[1], "SIMD16 fragment shader");
   if (enabled[2])
      ctx_disassemble_program(ctx, ksp[2], "SIMD32 fragment shader");
   fprintf(ctx->fp, "\n");
}

static void
decode_3dstate_constant(struct gen_batch_decode_ctx *ctx, const uint32_t *p)
{
   struct gen_group *inst = gen_spec_find_instruction(ctx->spec, p);
   struct gen_group *body =
      gen_spec_find_struct(ctx->spec, "3DSTATE_CONSTANT_BODY");

   uint32_t read_length[4] = {0};
   uint64_t read_addr[4];

   struct gen_field_iterator outer;
   gen_field_iterator_init(&outer, inst, p, 0, false);
   while (gen_field_iterator_next(&outer)) {
      if (outer.struct_desc != body)
         continue;

      struct gen_field_iterator iter;
      gen_field_iterator_init(&iter, body, &outer.p[outer.start_bit / 32],
                              0, false);

      while (gen_field_iterator_next(&iter)) {
         int idx;
         if (sscanf(iter.name, "Read Length[%d]", &idx) == 1) {
            read_length[idx] = iter.raw_value;
         } else if (sscanf(iter.name, "Buffer[%d]", &idx) == 1) {
            read_addr[idx] = iter.raw_value;
         }
      }

      for (int i = 0; i < 4; i++) {
         if (read_length[i] == 0)
            continue;

         struct gen_batch_decode_bo buffer = ctx_get_bo(ctx, read_addr[i]);
         if (!buffer.map) {
            fprintf(ctx->fp, "constant buffer %d unavailable\n", i);
            continue;
         }

         unsigned size = read_length[i] * 32;
         fprintf(ctx->fp, "constant buffer %d, size %u\n", i, size);

         ctx_print_buffer(ctx, buffer, size, 0, -1);
      }
   }
}

static void
decode_3dstate_binding_table_pointers(struct gen_batch_decode_ctx *ctx,
                                      const uint32_t *p)
{
   dump_binding_table(ctx, p[1], -1);
}

static void
decode_3dstate_sampler_state_pointers(struct gen_batch_decode_ctx *ctx,
                                      const uint32_t *p)
{
   dump_samplers(ctx, p[1], -1);
}

static void
decode_3dstate_sampler_state_pointers_gen6(struct gen_batch_decode_ctx *ctx,
                                           const uint32_t *p)
{
   dump_samplers(ctx, p[1], -1);
   dump_samplers(ctx, p[2], -1);
   dump_samplers(ctx, p[3], -1);
}

static bool
str_ends_with(const char *str, const char *end)
{
   int offset = strlen(str) - strlen(end);
   if (offset < 0)
      return false;

   return strcmp(str + offset, end) == 0;
}

static void
decode_dynamic_state_pointers(struct gen_batch_decode_ctx *ctx,
                              const char *struct_type, const uint32_t *p,
                              int count)
{
   struct gen_group *inst = gen_spec_find_instruction(ctx->spec, p);

   uint32_t state_offset = 0;

   struct gen_field_iterator iter;
   gen_field_iterator_init(&iter, inst, p, 0, false);
   while (gen_field_iterator_next(&iter)) {
      if (str_ends_with(iter.name, "Pointer")) {
         state_offset = iter.raw_value;
         break;
      }
   }

   uint64_t state_addr = ctx->dynamic_base + state_offset;
   struct gen_batch_decode_bo bo = ctx_get_bo(ctx, state_addr);
   const void *state_map = bo.map;

   if (state_map == NULL) {
      fprintf(ctx->fp, "  dynamic %s state unavailable\n", struct_type);
      return;
   }

   struct gen_group *state = gen_spec_find_struct(ctx->spec, struct_type);
   if (strcmp(struct_type, "BLEND_STATE") == 0) {
      /* Blend states are different from the others because they have a header
       * struct called BLEND_STATE which is followed by a variable number of
       * BLEND_STATE_ENTRY structs.
       */
      fprintf(ctx->fp, "%s\n", struct_type);
      ctx_print_group(ctx, state, state_addr, state_map);

      state_addr += state->dw_length * 4;
      state_map += state->dw_length * 4;

      struct_type = "BLEND_STATE_ENTRY";
      state = gen_spec_find_struct(ctx->spec, struct_type);
   }

   for (int i = 0; i < count; i++) {
      fprintf(ctx->fp, "%s %d\n", struct_type, i);
      ctx_print_group(ctx, state, state_addr, state_map);

      state_addr += state->dw_length * 4;
      state_map += state->dw_length * 4;
   }
}

static void
decode_3dstate_viewport_state_pointers_cc(struct gen_batch_decode_ctx *ctx,
                                          const uint32_t *p)
{
   decode_dynamic_state_pointers(ctx, "CC_VIEWPORT", p, 4);
}

static void
decode_3dstate_viewport_state_pointers_sf_clip(struct gen_batch_decode_ctx *ctx,
                                               const uint32_t *p)
{
   decode_dynamic_state_pointers(ctx, "SF_CLIP_VIEWPORT", p, 4);
}

static void
decode_3dstate_blend_state_pointers(struct gen_batch_decode_ctx *ctx,
                                    const uint32_t *p)
{
   decode_dynamic_state_pointers(ctx, "BLEND_STATE", p, 1);
}

static void
decode_3dstate_cc_state_pointers(struct gen_batch_decode_ctx *ctx,
                                 const uint32_t *p)
{
   decode_dynamic_state_pointers(ctx, "COLOR_CALC_STATE", p, 1);
}

static void
decode_3dstate_scissor_state_pointers(struct gen_batch_decode_ctx *ctx,
                                      const uint32_t *p)
{
   decode_dynamic_state_pointers(ctx, "SCISSOR_RECT", p, 1);
}

static void
decode_load_register_imm(struct gen_batch_decode_ctx *ctx, const uint32_t *p)
{
   struct gen_group *reg = gen_spec_find_register(ctx->spec, p[1]);

   if (reg != NULL) {
      fprintf(ctx->fp, "register %s (0x%x): 0x%x\n",
              reg->name, reg->register_offset, p[2]);
      ctx_print_group(ctx, reg, reg->register_offset, &p[2]);
   }
}

struct custom_decoder {
   const char *cmd_name;
   void (*decode)(struct gen_batch_decode_ctx *ctx, const uint32_t *p);
} custom_decoders[] = {
   { "STATE_BASE_ADDRESS", handle_state_base_address },
   { "MEDIA_INTERFACE_DESCRIPTOR_LOAD", handle_media_interface_descriptor_load },
   { "3DSTATE_VERTEX_BUFFERS", handle_3dstate_vertex_buffers },
   { "3DSTATE_INDEX_BUFFER", handle_3dstate_index_buffer },
   { "3DSTATE_VS", decode_single_ksp },
   { "3DSTATE_GS", decode_single_ksp },
   { "3DSTATE_DS", decode_single_ksp },
   { "3DSTATE_HS", decode_single_ksp },
   { "3DSTATE_PS", decode_ps_kernels },
   { "3DSTATE_CONSTANT_VS", decode_3dstate_constant },
   { "3DSTATE_CONSTANT_GS", decode_3dstate_constant },
   { "3DSTATE_CONSTANT_PS", decode_3dstate_constant },
   { "3DSTATE_CONSTANT_HS", decode_3dstate_constant },
   { "3DSTATE_CONSTANT_DS", decode_3dstate_constant },

   { "3DSTATE_BINDING_TABLE_POINTERS_VS", decode_3dstate_binding_table_pointers },
   { "3DSTATE_BINDING_TABLE_POINTERS_HS", decode_3dstate_binding_table_pointers },
   { "3DSTATE_BINDING_TABLE_POINTERS_DS", decode_3dstate_binding_table_pointers },
   { "3DSTATE_BINDING_TABLE_POINTERS_GS", decode_3dstate_binding_table_pointers },
   { "3DSTATE_BINDING_TABLE_POINTERS_PS", decode_3dstate_binding_table_pointers },

   { "3DSTATE_SAMPLER_STATE_POINTERS_VS", decode_3dstate_sampler_state_pointers },
   { "3DSTATE_SAMPLER_STATE_POINTERS_HS", decode_3dstate_sampler_state_pointers },
   { "3DSTATE_SAMPLER_STATE_POINTERS_DS", decode_3dstate_sampler_state_pointers },
   { "3DSTATE_SAMPLER_STATE_POINTERS_GS", decode_3dstate_sampler_state_pointers },
   { "3DSTATE_SAMPLER_STATE_POINTERS_PS", decode_3dstate_sampler_state_pointers },
   { "3DSTATE_SAMPLER_STATE_POINTERS", decode_3dstate_sampler_state_pointers_gen6 },

   { "3DSTATE_VIEWPORT_STATE_POINTERS_CC", decode_3dstate_viewport_state_pointers_cc },
   { "3DSTATE_VIEWPORT_STATE_POINTERS_SF_CLIP", decode_3dstate_viewport_state_pointers_sf_clip },
   { "3DSTATE_BLEND_STATE_POINTERS", decode_3dstate_blend_state_pointers },
   { "3DSTATE_CC_STATE_POINTERS", decode_3dstate_cc_state_pointers },
   { "3DSTATE_SCISSOR_STATE_POINTERS", decode_3dstate_scissor_state_pointers },
   { "MI_LOAD_REGISTER_IMM", decode_load_register_imm }
};

void
gen_print_batch(struct gen_batch_decode_ctx *ctx,
                const uint32_t *batch, uint32_t batch_size,
                uint64_t batch_addr)
{
   const uint32_t *p, *end = batch + batch_size / sizeof(uint32_t);
   int length;
   struct gen_group *inst;

   for (p = batch; p < end; p += length) {
      inst = gen_spec_find_instruction(ctx->spec, p);
      length = gen_group_get_length(inst, p);
      assert(inst == NULL || length > 0);
      length = MAX2(1, length);

      const char *reset_color = ctx->flags & GEN_BATCH_DECODE_IN_COLOR ? NORMAL : "";

      uint64_t offset;
      if (ctx->flags & GEN_BATCH_DECODE_OFFSETS)
         offset = batch_addr + ((char *)p - (char *)batch);
      else
         offset = 0;

      if (inst == NULL) {
         fprintf(ctx->fp, "%s0x%08"PRIx64": unknown instruction %08x%s\n",
                 (ctx->flags & GEN_BATCH_DECODE_IN_COLOR) ? RED_COLOR : "",
                 offset, p[0], reset_color);
         continue;
      }

      const char *color;
      const char *inst_name = gen_group_get_name(inst);
      if (ctx->flags & GEN_BATCH_DECODE_IN_COLOR) {
         reset_color = NORMAL;
         if (ctx->flags & GEN_BATCH_DECODE_FULL) {
            if (strcmp(inst_name, "MI_BATCH_BUFFER_START") == 0 ||
                strcmp(inst_name, "MI_BATCH_BUFFER_END") == 0)
               color = GREEN_HEADER;
            else
               color = BLUE_HEADER;
         } else {
            color = NORMAL;
         }
      } else {
         color = "";
         reset_color = "";
      }

      fprintf(ctx->fp, "%s0x%08"PRIx64":  0x%08x:  %-80s%s\n",
              color, offset, p[0], inst_name, reset_color);

      if (ctx->flags & GEN_BATCH_DECODE_FULL) {
         ctx_print_group(ctx, inst, offset, p);

         for (int i = 0; i < ARRAY_SIZE(custom_decoders); i++) {
            if (strcmp(inst_name, custom_decoders[i].cmd_name) == 0) {
               custom_decoders[i].decode(ctx, p);
               break;
            }
         }
      }

      if (strcmp(inst_name, "MI_BATCH_BUFFER_START") == 0) {
         struct gen_batch_decode_bo next_batch = {};
         bool second_level;
         struct gen_field_iterator iter;
         gen_field_iterator_init(&iter, inst, p, 0, false);
         while (gen_field_iterator_next(&iter)) {
            if (strcmp(iter.name, "Batch Buffer Start Address") == 0) {
               next_batch = ctx_get_bo(ctx, iter.raw_value);
            } else if (strcmp(iter.name, "Second Level Batch Buffer") == 0) {
               second_level = iter.raw_value;
            }
         }

         if (next_batch.map == NULL) {
            fprintf(ctx->fp, "Secondary batch at 0x%08"PRIx64" unavailable\n",
                    next_batch.addr);
         } else {
            gen_print_batch(ctx, next_batch.map, next_batch.size,
                            next_batch.addr);
         }
         if (second_level) {
            /* MI_BATCH_BUFFER_START with "2nd Level Batch Buffer" set acts
             * like a subroutine call.  Commands that come afterwards get
             * processed once the 2nd level batch buffer returns with
             * MI_BATCH_BUFFER_END.
             */
            continue;
         } else {
            /* MI_BATCH_BUFFER_START with "2nd Level Batch Buffer" unset acts
             * like a goto.  Nothing after it will ever get processed.  In
             * order to prevent the recursion from growing, we just reset the
             * loop and continue;
             */
            break;
         }
      } else if (strcmp(inst_name, "MI_BATCH_BUFFER_END") == 0) {
         break;
      }
   }
}
