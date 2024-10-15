/*
 * Copyright © 2013 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \file brw_vec4_tcs.cpp
 *
 * Tessellaton control shader specific code derived from the vec4_visitor class.
 */

#include "brw_nir.h"
#include "brw_vec4_tcs.h"
#include "brw_fs.h"
#include "common/gen_debug.h"

namespace brw {

vec4_tcs_visitor::vec4_tcs_visitor(const struct brw_compiler *compiler,
                                   void *log_data,
                                   const struct brw_tcs_prog_key *key,
                                   struct brw_tcs_prog_data *prog_data,
                                   const nir_shader *nir,
                                   void *mem_ctx,
                                   int shader_time_index,
                                   const struct brw_vue_map *input_vue_map)
   : vec4_visitor(compiler, log_data, &key->tex, &prog_data->base,
                  nir, mem_ctx, false, shader_time_index),
     input_vue_map(input_vue_map), key(key)
{
}


void
vec4_tcs_visitor::setup_payload()
{
   int reg = 0;

   /* The payload always contains important data in r0, which contains
    * the URB handles that are passed on to the URB write at the end
    * of the thread.
    */
   reg++;

   /* r1.0 - r4.7 may contain the input control point URB handles,
    * which we use to pull vertex data.
    */
   reg += 4;

   /* Push constants may start at r5.0 */
   reg = setup_uniforms(reg);

   this->first_non_payload_grf = reg;
}


void
vec4_tcs_visitor::emit_prolog()
{
   invocation_id = src_reg(this, glsl_type::uint_type);
   emit(TCS_OPCODE_GET_INSTANCE_ID, dst_reg(invocation_id));

   /* HS threads are dispatched with the dispatch mask set to 0xFF.
    * If there are an odd number of output vertices, then the final
    * HS instance dispatched will only have its bottom half doing real
    * work, and so we need to disable the upper half:
    */
   if (nir->info.tess.tcs_vertices_out % 2) {
      emit(CMP(dst_null_d(), invocation_id,
               brw_imm_ud(nir->info.tess.tcs_vertices_out),
               BRW_CONDITIONAL_L));

      /* Matching ENDIF is in emit_thread_end() */
      emit(IF(BRW_PREDICATE_NORMAL));
   }
}


void
vec4_tcs_visitor::emit_thread_end()
{
   vec4_instruction *inst;
   current_annotation = "thread end";

   if (nir->info.tess.tcs_vertices_out % 2) {
      emit(BRW_OPCODE_ENDIF);
   }

   if (devinfo->gen == 7) {
      struct brw_tcs_prog_data *tcs_prog_data =
         (struct brw_tcs_prog_data *) prog_data;

      current_annotation = "release input vertices";

      /* Synchronize all threads, so we know that no one is still
       * using the input URB handles.
       */
      if (tcs_prog_data->instances > 1) {
         dst_reg header = dst_reg(this, glsl_type::uvec4_type);
         emit(TCS_OPCODE_CREATE_BARRIER_HEADER, header);
         emit(SHADER_OPCODE_BARRIER, dst_null_ud(), src_reg(header));
      }

      /* Make thread 0 (invocations <1, 0>) release pairs of ICP handles.
       * We want to compare the bottom half of invocation_id with 0, but
       * use that truth value for the top half as well.  Unfortunately,
       * we don't have stride in the vec4 world, nor UV immediates in
       * align16, so we need an opcode to get invocation_id<0,4,0>.
       */
      set_condmod(BRW_CONDITIONAL_Z,
                  emit(TCS_OPCODE_SRC0_010_IS_ZERO, dst_null_d(),
                       invocation_id));
      emit(IF(BRW_PREDICATE_NORMAL));
      for (unsigned i = 0; i < key->input_vertices; i += 2) {
         /* If we have an odd number of input vertices, the last will be
          * unpaired.  We don't want to use an interleaved URB write in
          * that case.
          */
         const bool is_unpaired = i == key->input_vertices - 1;

         dst_reg header(this, glsl_type::uvec4_type);
         emit(TCS_OPCODE_RELEASE_INPUT, header, brw_imm_ud(i),
              brw_imm_ud(is_unpaired));
      }
      emit(BRW_OPCODE_ENDIF);
   }

   if (unlikely(INTEL_DEBUG & DEBUG_SHADER_TIME))
      emit_shader_time_end();

   inst = emit(TCS_OPCODE_THREAD_END);
   inst->base_mrf = 14;
   inst->mlen = 2;
}


void
vec4_tcs_visitor::emit_input_urb_read(const dst_reg &dst,
                                      const src_reg &vertex_index,
                                      unsigned base_offset,
                                      unsigned first_component,
                                      const src_reg &indirect_offset)
{
   vec4_instruction *inst;
   dst_reg temp(this, glsl_type::ivec4_type);
   temp.type = dst.type;

   /* Set up the message header to reference the proper parts of the URB */
   dst_reg header = dst_reg(this, glsl_type::uvec4_type);
   inst = emit(TCS_OPCODE_SET_INPUT_URB_OFFSETS, header, vertex_index,
               indirect_offset);
   inst->force_writemask_all = true;

   /* Read into a temporary, ignoring writemasking. */
   inst = emit(VEC4_OPCODE_URB_READ, temp, src_reg(header));
   inst->offset = base_offset;
   inst->mlen = 1;
   inst->base_mrf = -1;

   /* Copy the temporary to the destination to deal with writemasking.
    *
    * Also attempt to deal with gl_PointSize being in the .w component.
    */
   if (inst->offset == 0 && indirect_offset.file == BAD_FILE) {
      emit(MOV(dst, swizzle(src_reg(temp), BRW_SWIZZLE_WWWW)));
   } else {
      src_reg src = src_reg(temp);
      src.swizzle = BRW_SWZ_COMP_INPUT(first_component);
      emit(MOV(dst, src));
   }
}

void
vec4_tcs_visitor::emit_output_urb_read(const dst_reg &dst,
                                       unsigned base_offset,
                                       unsigned first_component,
                                       const src_reg &indirect_offset)
{
   vec4_instruction *inst;

   /* Set up the message header to reference the proper parts of the URB */
   dst_reg header = dst_reg(this, glsl_type::uvec4_type);
   inst = emit(TCS_OPCODE_SET_OUTPUT_URB_OFFSETS, header,
               brw_imm_ud(dst.writemask << first_component), indirect_offset);
   inst->force_writemask_all = true;

   vec4_instruction *read = emit(VEC4_OPCODE_URB_READ, dst, src_reg(header));
   read->offset = base_offset;
   read->mlen = 1;
   read->base_mrf = -1;

   if (first_component) {
      /* Read into a temporary and copy with a swizzle and writemask. */
      read->dst = retype(dst_reg(this, glsl_type::ivec4_type), dst.type);
      emit(MOV(dst, swizzle(src_reg(read->dst),
                            BRW_SWZ_COMP_INPUT(first_component))));
   }
}

void
vec4_tcs_visitor::emit_urb_write(const src_reg &value,
                                 unsigned writemask,
                                 unsigned base_offset,
                                 const src_reg &indirect_offset)
{
   if (writemask == 0)
      return;

   src_reg message(this, glsl_type::uvec4_type, 2);
   vec4_instruction *inst;

   inst = emit(TCS_OPCODE_SET_OUTPUT_URB_OFFSETS, dst_reg(message),
               brw_imm_ud(writemask), indirect_offset);
   inst->force_writemask_all = true;
   inst = emit(MOV(byte_offset(dst_reg(retype(message, value.type)), REG_SIZE),
                   value));
   inst->force_writemask_all = true;

   inst = emit(TCS_OPCODE_URB_WRITE, dst_null_f(), message);
   inst->offset = base_offset;
   inst->mlen = 2;
   inst->base_mrf = -1;
}

void
vec4_tcs_visitor::nir_emit_intrinsic(nir_intrinsic_instr *instr)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_load_invocation_id:
      emit(MOV(get_nir_dest(instr->dest, BRW_REGISTER_TYPE_UD),
               invocation_id));
      break;
   case nir_intrinsic_load_primitive_id:
      emit(TCS_OPCODE_GET_PRIMITIVE_ID,
           get_nir_dest(instr->dest, BRW_REGISTER_TYPE_UD));
      break;
   case nir_intrinsic_load_patch_vertices_in:
      emit(MOV(get_nir_dest(instr->dest, BRW_REGISTER_TYPE_D),
               brw_imm_d(key->input_vertices)));
      break;
   case nir_intrinsic_load_per_vertex_input: {
      src_reg indirect_offset = get_indirect_offset(instr);
      unsigned imm_offset = instr->const_index[0];

      nir_const_value *vertex_const = nir_src_as_const_value(instr->src[0]);
      src_reg vertex_index =
         vertex_const ? src_reg(brw_imm_ud(vertex_const->u32[0]))
                      : get_nir_src(instr->src[0], BRW_REGISTER_TYPE_UD, 1);

      unsigned first_component = nir_intrinsic_component(instr);
      if (nir_dest_bit_size(instr->dest) == 64) {
         /* We need to emit up to two 32-bit URB reads, then shuffle
          * the result into a temporary, then move to the destination
          * honoring the writemask
          *
          * We don't need to divide first_component by 2 because
          * emit_input_urb_read takes a 32-bit type.
          */
         dst_reg tmp = dst_reg(this, glsl_type::dvec4_type);
         dst_reg tmp_d = retype(tmp, BRW_REGISTER_TYPE_D);
         emit_input_urb_read(tmp_d, vertex_index, imm_offset,
                             first_component, indirect_offset);
         if (instr->num_components > 2) {
            emit_input_urb_read(byte_offset(tmp_d, REG_SIZE), vertex_index,
                                imm_offset + 1, 0, indirect_offset);
         }

         src_reg tmp_src = retype(src_reg(tmp_d), BRW_REGISTER_TYPE_DF);
         dst_reg shuffled = dst_reg(this, glsl_type::dvec4_type);
         shuffle_64bit_data(shuffled, tmp_src, false);

         dst_reg dst = get_nir_dest(instr->dest, BRW_REGISTER_TYPE_DF);
         dst.writemask = brw_writemask_for_size(instr->num_components);
         emit(MOV(dst, src_reg(shuffled)));
      } else {
         dst_reg dst = get_nir_dest(instr->dest, BRW_REGISTER_TYPE_D);
         dst.writemask = brw_writemask_for_size(instr->num_components);
         emit_input_urb_read(dst, vertex_index, imm_offset,
                             first_component, indirect_offset);
      }
      break;
   }
   case nir_intrinsic_load_input:
      unreachable("nir_lower_io should use load_per_vertex_input intrinsics");
      break;
   case nir_intrinsic_load_output:
   case nir_intrinsic_load_per_vertex_output: {
      src_reg indirect_offset = get_indirect_offset(instr);
      unsigned imm_offset = instr->const_index[0];

      dst_reg dst = get_nir_dest(instr->dest, BRW_REGISTER_TYPE_D);
      dst.writemask = brw_writemask_for_size(instr->num_components);

      emit_output_urb_read(dst, imm_offset, nir_intrinsic_component(instr),
                           indirect_offset);
      break;
   }
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output: {
      src_reg value = get_nir_src(instr->src[0]);
      unsigned mask = instr->const_index[1];
      unsigned swiz = BRW_SWIZZLE_XYZW;

      src_reg indirect_offset = get_indirect_offset(instr);
      unsigned imm_offset = instr->const_index[0];

      unsigned first_component = nir_intrinsic_component(instr);
      if (first_component) {
         if (nir_src_bit_size(instr->src[0]) == 64)
            first_component /= 2;
         assert(swiz == BRW_SWIZZLE_XYZW);
         swiz = BRW_SWZ_COMP_OUTPUT(first_component);
         mask = mask << first_component;
      }

      if (nir_src_bit_size(instr->src[0]) == 64) {
         /* For 64-bit data we need to shuffle the data before we write and
          * emit two messages. Also, since each channel is twice as large we
          * need to fix the writemask in each 32-bit message to account for it.
          */
         value = swizzle(retype(value, BRW_REGISTER_TYPE_DF), swiz);
         dst_reg shuffled = dst_reg(this, glsl_type::dvec4_type);
         shuffle_64bit_data(shuffled, value, true);
         src_reg shuffled_float = src_reg(retype(shuffled, BRW_REGISTER_TYPE_F));

         for (int n = 0; n < 2; n++) {
            unsigned fixed_mask = 0;
            if (mask & WRITEMASK_X)
               fixed_mask |= WRITEMASK_XY;
            if (mask & WRITEMASK_Y)
               fixed_mask |= WRITEMASK_ZW;
            emit_urb_write(shuffled_float, fixed_mask,
                           imm_offset, indirect_offset);

            shuffled_float = byte_offset(shuffled_float, REG_SIZE);
            mask >>= 2;
            imm_offset++;
         }
      } else {
         emit_urb_write(swizzle(value, swiz), mask,
                        imm_offset, indirect_offset);
      }
      break;
   }

   case nir_intrinsic_barrier: {
      dst_reg header = dst_reg(this, glsl_type::uvec4_type);
      emit(TCS_OPCODE_CREATE_BARRIER_HEADER, header);
      emit(SHADER_OPCODE_BARRIER, dst_null_ud(), src_reg(header));
      break;
   }

   default:
      vec4_visitor::nir_emit_intrinsic(instr);
   }
}


extern "C" const unsigned *
brw_compile_tcs(const struct brw_compiler *compiler,
                void *log_data,
                void *mem_ctx,
                const struct brw_tcs_prog_key *key,
                struct brw_tcs_prog_data *prog_data,
                const nir_shader *src_shader,
                int shader_time_index,
                char **error_str)
{
   const struct gen_device_info *devinfo = compiler->devinfo;
   struct brw_vue_prog_data *vue_prog_data = &prog_data->base;
   const bool is_scalar = compiler->scalar_stage[MESA_SHADER_TESS_CTRL];
   const unsigned *assembly;

   nir_shader *nir = nir_shader_clone(mem_ctx, src_shader);
   nir->info.outputs_written = key->outputs_written;
   nir->info.patch_outputs_written = key->patch_outputs_written;

   struct brw_vue_map input_vue_map;
   brw_compute_vue_map(devinfo, &input_vue_map, nir->info.inputs_read,
                       nir->info.separate_shader);
   brw_compute_tess_vue_map(&vue_prog_data->vue_map,
                            nir->info.outputs_written,
                            nir->info.patch_outputs_written);

   nir = brw_nir_apply_sampler_key(nir, compiler, &key->tex, is_scalar);
   brw_nir_lower_vue_inputs(nir, &input_vue_map);
   brw_nir_lower_tcs_outputs(nir, &vue_prog_data->vue_map,
                             key->tes_primitive_mode);
   if (key->quads_workaround)
      brw_nir_apply_tcs_quads_workaround(nir);

   nir = brw_postprocess_nir(nir, compiler, is_scalar);

   if (is_scalar)
      prog_data->instances = DIV_ROUND_UP(nir->info.tess.tcs_vertices_out, 8);
   else
      prog_data->instances = DIV_ROUND_UP(nir->info.tess.tcs_vertices_out, 2);

   /* Compute URB entry size.  The maximum allowed URB entry size is 32k.
    * That divides up as follows:
    *
    *     32 bytes for the patch header (tessellation factors)
    *    480 bytes for per-patch varyings (a varying component is 4 bytes and
    *              gl_MaxTessPatchComponents = 120)
    *  16384 bytes for per-vertex varyings (a varying component is 4 bytes,
    *              gl_MaxPatchVertices = 32 and
    *              gl_MaxTessControlOutputComponents = 128)
    *
    *  15808 bytes left for varying packing overhead
    */
   const int num_per_patch_slots = vue_prog_data->vue_map.num_per_patch_slots;
   const int num_per_vertex_slots = vue_prog_data->vue_map.num_per_vertex_slots;
   unsigned output_size_bytes = 0;
   /* Note that the patch header is counted in num_per_patch_slots. */
   output_size_bytes += num_per_patch_slots * 16;
   output_size_bytes += nir->info.tess.tcs_vertices_out *
                        num_per_vertex_slots * 16;

   assert(output_size_bytes >= 1);
   if (output_size_bytes > GEN7_MAX_HS_URB_ENTRY_SIZE_BYTES)
      return NULL;

   /* URB entry sizes are stored as a multiple of 64 bytes. */
   vue_prog_data->urb_entry_size = ALIGN(output_size_bytes, 64) / 64;

   /* On Cannonlake software shall not program an allocation size that
    * specifies a size that is a multiple of 3 64B (512-bit) cachelines.
    */
   if (devinfo->gen == 10 &&
       vue_prog_data->urb_entry_size % 3 == 0)
      vue_prog_data->urb_entry_size++;

   /* HS does not use the usual payload pushing from URB to GRFs,
    * because we don't have enough registers for a full-size payload, and
    * the hardware is broken on Haswell anyway.
    */
   vue_prog_data->urb_read_length = 0;

   if (unlikely(INTEL_DEBUG & DEBUG_TCS)) {
      fprintf(stderr, "TCS Input ");
      brw_print_vue_map(stderr, &input_vue_map);
      fprintf(stderr, "TCS Output ");
      brw_print_vue_map(stderr, &vue_prog_data->vue_map);
   }

   if (is_scalar) {
      fs_visitor v(compiler, log_data, mem_ctx, (void *) key,
                   &prog_data->base.base, NULL, nir, 8,
                   shader_time_index, &input_vue_map);
      if (!v.run_tcs_single_patch()) {
         if (error_str)
            *error_str = ralloc_strdup(mem_ctx, v.fail_msg);
         return NULL;
      }

      prog_data->base.base.dispatch_grf_start_reg = v.payload.num_regs;
      prog_data->base.dispatch_mode = DISPATCH_MODE_SIMD8;

      fs_generator g(compiler, log_data, mem_ctx,
                     &prog_data->base.base, v.promoted_constants, false,
                     MESA_SHADER_TESS_CTRL);
      if (unlikely(INTEL_DEBUG & DEBUG_TCS)) {
         g.enable_debug(ralloc_asprintf(mem_ctx,
                                        "%s tessellation control shader %s",
                                        nir->info.label ? nir->info.label
                                                        : "unnamed",
                                        nir->info.name));
      }

      g.generate_code(v.cfg, 8);

      assembly = g.get_assembly();
   } else {
      vec4_tcs_visitor v(compiler, log_data, key, prog_data,
                         nir, mem_ctx, shader_time_index, &input_vue_map);
      if (!v.run()) {
         if (error_str)
            *error_str = ralloc_strdup(mem_ctx, v.fail_msg);
         return NULL;
      }

      if (unlikely(INTEL_DEBUG & DEBUG_TCS))
         v.dump_instructions();


      assembly = brw_vec4_generate_assembly(compiler, log_data, mem_ctx, nir,
                                            &prog_data->base, v.cfg);
   }

   return assembly;
}


} /* namespace brw */
