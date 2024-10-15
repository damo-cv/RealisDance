/*
 * Copyright © 2015 Intel Corporation
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

#include "anv_nir.h"
#include "program/prog_parameter.h"
#include "nir/nir_builder.h"
#include "compiler/brw_nir.h"

struct apply_pipeline_layout_state {
   nir_shader *shader;
   nir_builder builder;

   struct anv_pipeline_layout *layout;
   bool add_bounds_checks;

   unsigned first_image_uniform;

   bool uses_constants;
   uint8_t constants_offset;
   struct {
      BITSET_WORD *used;
      uint8_t *surface_offsets;
      uint8_t *sampler_offsets;
      uint8_t *image_offsets;
   } set[MAX_SETS];
};

static void
add_binding(struct apply_pipeline_layout_state *state,
            uint32_t set, uint32_t binding)
{
   BITSET_SET(state->set[set].used, binding);
}

static void
add_var_binding(struct apply_pipeline_layout_state *state, nir_variable *var)
{
   add_binding(state, var->data.descriptor_set, var->data.binding);
}

static void
add_deref_src_binding(struct apply_pipeline_layout_state *state, nir_src src)
{
   nir_deref_instr *deref = nir_src_as_deref(src);
   add_var_binding(state, nir_deref_instr_get_variable(deref));
}

static void
add_tex_src_binding(struct apply_pipeline_layout_state *state,
                    nir_tex_instr *tex, nir_tex_src_type deref_src_type)
{
   int deref_src_idx = nir_tex_instr_src_index(tex, deref_src_type);
   if (deref_src_idx < 0)
      return;

   add_deref_src_binding(state, tex->src[deref_src_idx].src);
}

static void
get_used_bindings_block(nir_block *block,
                        struct apply_pipeline_layout_state *state)
{
   nir_foreach_instr_safe(instr, block) {
      switch (instr->type) {
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         switch (intrin->intrinsic) {
         case nir_intrinsic_vulkan_resource_index:
            add_binding(state, nir_intrinsic_desc_set(intrin),
                        nir_intrinsic_binding(intrin));
            break;

         case nir_intrinsic_image_deref_load:
         case nir_intrinsic_image_deref_store:
         case nir_intrinsic_image_deref_atomic_add:
         case nir_intrinsic_image_deref_atomic_min:
         case nir_intrinsic_image_deref_atomic_max:
         case nir_intrinsic_image_deref_atomic_and:
         case nir_intrinsic_image_deref_atomic_or:
         case nir_intrinsic_image_deref_atomic_xor:
         case nir_intrinsic_image_deref_atomic_exchange:
         case nir_intrinsic_image_deref_atomic_comp_swap:
         case nir_intrinsic_image_deref_size:
         case nir_intrinsic_image_deref_samples:
         case nir_intrinsic_image_deref_load_param_intel:
         case nir_intrinsic_image_deref_load_raw_intel:
         case nir_intrinsic_image_deref_store_raw_intel:
            add_deref_src_binding(state, intrin->src[0]);
            break;

         case nir_intrinsic_load_constant:
            state->uses_constants = true;
            break;

         default:
            break;
         }
         break;
      }
      case nir_instr_type_tex: {
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         add_tex_src_binding(state, tex, nir_tex_src_texture_deref);
         add_tex_src_binding(state, tex, nir_tex_src_sampler_deref);
         break;
      }
      default:
         continue;
      }
   }
}

static void
lower_res_index_intrinsic(nir_intrinsic_instr *intrin,
                          struct apply_pipeline_layout_state *state)
{
   nir_builder *b = &state->builder;

   b->cursor = nir_before_instr(&intrin->instr);

   uint32_t set = nir_intrinsic_desc_set(intrin);
   uint32_t binding = nir_intrinsic_binding(intrin);

   uint32_t surface_index = state->set[set].surface_offsets[binding];
   uint32_t array_size =
      state->layout->set[set].layout->binding[binding].array_size;

   nir_const_value *const_array_index = nir_src_as_const_value(intrin->src[0]);

   nir_ssa_def *block_index;
   if (const_array_index) {
      unsigned array_index = const_array_index->u32[0];
      array_index = MIN2(array_index, array_size - 1);
      block_index = nir_imm_int(b, surface_index + array_index);
   } else {
      block_index = nir_ssa_for_src(b, intrin->src[0], 1);

      if (state->add_bounds_checks)
         block_index = nir_umin(b, block_index, nir_imm_int(b, array_size - 1));

      block_index = nir_iadd(b, nir_imm_int(b, surface_index), block_index);
   }

   assert(intrin->dest.is_ssa);
   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, nir_src_for_ssa(block_index));
   nir_instr_remove(&intrin->instr);
}

static void
lower_res_reindex_intrinsic(nir_intrinsic_instr *intrin,
                            struct apply_pipeline_layout_state *state)
{
   nir_builder *b = &state->builder;

   b->cursor = nir_before_instr(&intrin->instr);

   /* For us, the resource indices are just indices into the binding table and
    * array elements are sequential.  A resource_reindex just turns into an
    * add of the two indices.
    */
   assert(intrin->src[0].is_ssa && intrin->src[1].is_ssa);
   nir_ssa_def *new_index = nir_iadd(b, intrin->src[0].ssa,
                                        intrin->src[1].ssa);

   assert(intrin->dest.is_ssa);
   nir_ssa_def_rewrite_uses(&intrin->dest.ssa, nir_src_for_ssa(new_index));
   nir_instr_remove(&intrin->instr);
}

static void
lower_image_intrinsic(nir_intrinsic_instr *intrin,
                      struct apply_pipeline_layout_state *state)
{
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   nir_variable *var = nir_deref_instr_get_variable(deref);

   unsigned set = var->data.descriptor_set;
   unsigned binding = var->data.binding;
   unsigned array_size =
      state->layout->set[set].layout->binding[binding].array_size;

   nir_builder *b = &state->builder;
   b->cursor = nir_before_instr(&intrin->instr);

   nir_ssa_def *index = NULL;
   if (deref->deref_type != nir_deref_type_var) {
      assert(deref->deref_type == nir_deref_type_array);
      index = nir_ssa_for_src(b, deref->arr.index, 1);
      if (state->add_bounds_checks)
         index = nir_umin(b, index, nir_imm_int(b, array_size - 1));
   } else {
      index = nir_imm_int(b, 0);
   }

   if (intrin->intrinsic == nir_intrinsic_image_deref_load_param_intel) {
      b->cursor = nir_instr_remove(&intrin->instr);

      nir_intrinsic_instr *load =
         nir_intrinsic_instr_create(b->shader, nir_intrinsic_load_uniform);

      nir_intrinsic_set_base(load, state->first_image_uniform +
                                   state->set[set].image_offsets[binding] *
                                   BRW_IMAGE_PARAM_SIZE * 4);
      nir_intrinsic_set_range(load, array_size * BRW_IMAGE_PARAM_SIZE * 4);

      const unsigned param = nir_intrinsic_base(intrin);
      nir_ssa_def *offset =
         nir_imul(b, index, nir_imm_int(b, BRW_IMAGE_PARAM_SIZE * 4));
      offset = nir_iadd(b, offset, nir_imm_int(b, param * 16));
      load->src[0] = nir_src_for_ssa(offset);

      load->num_components = intrin->dest.ssa.num_components;
      nir_ssa_dest_init(&load->instr, &load->dest,
                        intrin->dest.ssa.num_components,
                        intrin->dest.ssa.bit_size, NULL);
      nir_builder_instr_insert(b, &load->instr);

      nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                               nir_src_for_ssa(&load->dest.ssa));
   } else {
      unsigned binding_offset = state->set[set].surface_offsets[binding];
      index = nir_iadd(b, index, nir_imm_int(b, binding_offset));
      brw_nir_rewrite_image_intrinsic(intrin, index);
   }
}

static void
lower_load_constant(nir_intrinsic_instr *intrin,
                    struct apply_pipeline_layout_state *state)
{
   nir_builder *b = &state->builder;

   b->cursor = nir_before_instr(&intrin->instr);

   nir_ssa_def *index = nir_imm_int(b, state->constants_offset);
   nir_ssa_def *offset = nir_iadd(b, nir_ssa_for_src(b, intrin->src[0], 1),
                                  nir_imm_int(b, nir_intrinsic_base(intrin)));

   nir_intrinsic_instr *load_ubo =
      nir_intrinsic_instr_create(b->shader, nir_intrinsic_load_ubo);
   load_ubo->num_components = intrin->num_components;
   load_ubo->src[0] = nir_src_for_ssa(index);
   load_ubo->src[1] = nir_src_for_ssa(offset);
   nir_ssa_dest_init(&load_ubo->instr, &load_ubo->dest,
                     intrin->dest.ssa.num_components,
                     intrin->dest.ssa.bit_size, NULL);
   nir_builder_instr_insert(b, &load_ubo->instr);

   nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                            nir_src_for_ssa(&load_ubo->dest.ssa));
   nir_instr_remove(&intrin->instr);
}

static void
lower_tex_deref(nir_tex_instr *tex, nir_tex_src_type deref_src_type,
                unsigned *base_index,
                struct apply_pipeline_layout_state *state)
{
   int deref_src_idx = nir_tex_instr_src_index(tex, deref_src_type);
   if (deref_src_idx < 0)
      return;

   nir_deref_instr *deref = nir_src_as_deref(tex->src[deref_src_idx].src);
   nir_variable *var = nir_deref_instr_get_variable(deref);

   unsigned set = var->data.descriptor_set;
   unsigned binding = var->data.binding;
   unsigned array_size =
      state->layout->set[set].layout->binding[binding].array_size;

   nir_tex_src_type offset_src_type;
   if (deref_src_type == nir_tex_src_texture_deref) {
      offset_src_type = nir_tex_src_texture_offset;
      *base_index = state->set[set].surface_offsets[binding];
   } else {
      assert(deref_src_type == nir_tex_src_sampler_deref);
      offset_src_type = nir_tex_src_sampler_offset;
      *base_index = state->set[set].sampler_offsets[binding];
   }

   nir_ssa_def *index = NULL;
   if (deref->deref_type != nir_deref_type_var) {
      assert(deref->deref_type == nir_deref_type_array);

      nir_const_value *const_index = nir_src_as_const_value(deref->arr.index);
      if (const_index) {
         *base_index += MIN2(const_index->u32[0], array_size - 1);
      } else {
         nir_builder *b = &state->builder;

         /* From VK_KHR_sampler_ycbcr_conversion:
          *
          * If sampler Y’CBCR conversion is enabled, the combined image
          * sampler must be indexed only by constant integral expressions when
          * aggregated into arrays in shader code, irrespective of the
          * shaderSampledImageArrayDynamicIndexing feature.
          */
         assert(nir_tex_instr_src_index(tex, nir_tex_src_plane) == -1);

         index = nir_ssa_for_src(b, deref->arr.index, 1);

         if (state->add_bounds_checks)
            index = nir_umin(b, index, nir_imm_int(b, array_size - 1));
      }
   }

   if (index) {
      nir_instr_rewrite_src(&tex->instr, &tex->src[deref_src_idx].src,
                            nir_src_for_ssa(index));
      tex->src[deref_src_idx].src_type = offset_src_type;
   } else {
      nir_tex_instr_remove_src(tex, deref_src_idx);
   }
}

static uint32_t
tex_instr_get_and_remove_plane_src(nir_tex_instr *tex)
{
   int plane_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_plane);
   if (plane_src_idx < 0)
      return 0;

   unsigned plane =
      nir_src_as_const_value(tex->src[plane_src_idx].src)->u32[0];

   nir_tex_instr_remove_src(tex, plane_src_idx);

   return plane;
}

static void
lower_tex(nir_tex_instr *tex, struct apply_pipeline_layout_state *state)
{
   state->builder.cursor = nir_before_instr(&tex->instr);

   unsigned plane = tex_instr_get_and_remove_plane_src(tex);

   lower_tex_deref(tex, nir_tex_src_texture_deref,
                   &tex->texture_index, state);
   tex->texture_index += plane;

   lower_tex_deref(tex, nir_tex_src_sampler_deref,
                   &tex->sampler_index, state);
   tex->sampler_index += plane;

   /* The backend only ever uses this to mark used surfaces.  We don't care
    * about that little optimization so it just needs to be non-zero.
    */
   tex->texture_array_size = 1;
}

static void
apply_pipeline_layout_block(nir_block *block,
                            struct apply_pipeline_layout_state *state)
{
   nir_foreach_instr_safe(instr, block) {
      switch (instr->type) {
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         switch (intrin->intrinsic) {
         case nir_intrinsic_vulkan_resource_index:
            lower_res_index_intrinsic(intrin, state);
            break;
         case nir_intrinsic_vulkan_resource_reindex:
            lower_res_reindex_intrinsic(intrin, state);
            break;
         case nir_intrinsic_image_deref_load:
         case nir_intrinsic_image_deref_store:
         case nir_intrinsic_image_deref_atomic_add:
         case nir_intrinsic_image_deref_atomic_min:
         case nir_intrinsic_image_deref_atomic_max:
         case nir_intrinsic_image_deref_atomic_and:
         case nir_intrinsic_image_deref_atomic_or:
         case nir_intrinsic_image_deref_atomic_xor:
         case nir_intrinsic_image_deref_atomic_exchange:
         case nir_intrinsic_image_deref_atomic_comp_swap:
         case nir_intrinsic_image_deref_size:
         case nir_intrinsic_image_deref_samples:
         case nir_intrinsic_image_deref_load_param_intel:
         case nir_intrinsic_image_deref_load_raw_intel:
         case nir_intrinsic_image_deref_store_raw_intel:
            lower_image_intrinsic(intrin, state);
            break;
         case nir_intrinsic_load_constant:
            lower_load_constant(intrin, state);
            break;
         default:
            break;
         }
         break;
      }
      case nir_instr_type_tex:
         lower_tex(nir_instr_as_tex(instr), state);
         break;
      default:
         continue;
      }
   }
}

static void
setup_vec4_uniform_value(uint32_t *params, uint32_t offset, unsigned n)
{
   for (unsigned i = 0; i < n; ++i)
      params[i] = ANV_PARAM_PUSH(offset + i * sizeof(uint32_t));

   for (unsigned i = n; i < 4; ++i)
      params[i] = BRW_PARAM_BUILTIN_ZERO;
}

void
anv_nir_apply_pipeline_layout(const struct anv_physical_device *pdevice,
                              bool robust_buffer_access,
                              struct anv_pipeline_layout *layout,
                              nir_shader *shader,
                              struct brw_stage_prog_data *prog_data,
                              struct anv_pipeline_bind_map *map)
{
   gl_shader_stage stage = shader->info.stage;

   struct apply_pipeline_layout_state state = {
      .shader = shader,
      .layout = layout,
      .add_bounds_checks = robust_buffer_access,
   };

   void *mem_ctx = ralloc_context(NULL);

   for (unsigned s = 0; s < layout->num_sets; s++) {
      const unsigned count = layout->set[s].layout->binding_count;
      const unsigned words = BITSET_WORDS(count);
      state.set[s].used = rzalloc_array(mem_ctx, BITSET_WORD, words);
      state.set[s].surface_offsets = rzalloc_array(mem_ctx, uint8_t, count);
      state.set[s].sampler_offsets = rzalloc_array(mem_ctx, uint8_t, count);
      state.set[s].image_offsets = rzalloc_array(mem_ctx, uint8_t, count);
   }

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_foreach_block(block, function->impl)
         get_used_bindings_block(block, &state);
   }

   if (state.uses_constants) {
      state.constants_offset = map->surface_count;
      map->surface_to_descriptor[map->surface_count].set =
         ANV_DESCRIPTOR_SET_SHADER_CONSTANTS;
      map->surface_count++;
   }

   for (uint32_t set = 0; set < layout->num_sets; set++) {
      struct anv_descriptor_set_layout *set_layout = layout->set[set].layout;

      BITSET_WORD b, _tmp;
      BITSET_FOREACH_SET(b, _tmp, state.set[set].used,
                         set_layout->binding_count) {
         struct anv_descriptor_set_binding_layout *binding =
            &set_layout->binding[b];

         if (binding->stage[stage].surface_index >= 0) {
            state.set[set].surface_offsets[b] = map->surface_count;
            struct anv_sampler **samplers = binding->immutable_samplers;
            for (unsigned i = 0; i < binding->array_size; i++) {
               uint8_t planes = samplers ? samplers[i]->n_planes : 1;
               for (uint8_t p = 0; p < planes; p++) {
                  map->surface_to_descriptor[map->surface_count++] =
                     (struct anv_pipeline_binding) {
                        .set = set,
                        .binding = b,
                        .index = i,
                        .plane = p,
                     };
               }
            }
         }

         if (binding->stage[stage].sampler_index >= 0) {
            state.set[set].sampler_offsets[b] = map->sampler_count;
            struct anv_sampler **samplers = binding->immutable_samplers;
            for (unsigned i = 0; i < binding->array_size; i++) {
               uint8_t planes = samplers ? samplers[i]->n_planes : 1;
               for (uint8_t p = 0; p < planes; p++) {
                  map->sampler_to_descriptor[map->sampler_count++] =
                     (struct anv_pipeline_binding) {
                        .set = set,
                        .binding = b,
                        .index = i,
                        .plane = p,
                     };
               }
            }
         }

         if (binding->stage[stage].image_index >= 0) {
            state.set[set].image_offsets[b] = map->image_count;
            map->image_count += binding->array_size;
         }
      }
   }

   if (map->image_count > 0 && pdevice->compiler->devinfo->gen < 9) {
      assert(map->image_count <= MAX_GEN8_IMAGES);
      assert(shader->num_uniforms == prog_data->nr_params * 4);
      state.first_image_uniform = shader->num_uniforms;
      uint32_t *param = brw_stage_prog_data_add_params(prog_data,
                                                       map->image_count *
                                                       BRW_IMAGE_PARAM_SIZE);
      struct anv_push_constants *null_data = NULL;
      const struct brw_image_param *image_param = null_data->images;
      for (uint32_t i = 0; i < map->image_count; i++) {
         setup_vec4_uniform_value(param + BRW_IMAGE_PARAM_OFFSET_OFFSET,
                                  (uintptr_t)image_param->offset, 2);
         setup_vec4_uniform_value(param + BRW_IMAGE_PARAM_SIZE_OFFSET,
                                  (uintptr_t)image_param->size, 3);
         setup_vec4_uniform_value(param + BRW_IMAGE_PARAM_STRIDE_OFFSET,
                                  (uintptr_t)image_param->stride, 4);
         setup_vec4_uniform_value(param + BRW_IMAGE_PARAM_TILING_OFFSET,
                                  (uintptr_t)image_param->tiling, 3);
         setup_vec4_uniform_value(param + BRW_IMAGE_PARAM_SWIZZLING_OFFSET,
                                  (uintptr_t)image_param->swizzling, 2);

         param += BRW_IMAGE_PARAM_SIZE;
         image_param ++;
      }
      assert(param == prog_data->param + prog_data->nr_params);

      shader->num_uniforms += map->image_count * BRW_IMAGE_PARAM_SIZE * 4;
      assert(shader->num_uniforms == prog_data->nr_params * 4);
   }

   nir_foreach_variable(var, &shader->uniforms) {
      const struct glsl_type *glsl_type = glsl_without_array(var->type);

      if (!glsl_type_is_image(glsl_type))
         continue;

      enum glsl_sampler_dim dim = glsl_get_sampler_dim(glsl_type);

      const uint32_t set = var->data.descriptor_set;
      const uint32_t binding = var->data.binding;
      const uint32_t array_size =
         layout->set[set].layout->binding[binding].array_size;

      if (!BITSET_TEST(state.set[set].used, binding))
         continue;

      struct anv_pipeline_binding *pipe_binding =
         &map->surface_to_descriptor[state.set[set].surface_offsets[binding]];
      for (unsigned i = 0; i < array_size; i++) {
         assert(pipe_binding[i].set == set);
         assert(pipe_binding[i].binding == binding);
         assert(pipe_binding[i].index == i);

         if (dim == GLSL_SAMPLER_DIM_SUBPASS ||
             dim == GLSL_SAMPLER_DIM_SUBPASS_MS)
            pipe_binding[i].input_attachment_index = var->data.index + i;

         pipe_binding[i].write_only =
            (var->data.image.access & ACCESS_NON_READABLE) != 0;
      }
   }

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      nir_builder_init(&state.builder, function->impl);
      nir_foreach_block(block, function->impl)
         apply_pipeline_layout_block(block, &state);
      nir_metadata_preserve(function->impl, nir_metadata_block_index |
                                            nir_metadata_dominance);
   }

   ralloc_free(mem_ctx);
}
