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

#include "anv_nir.h"
#include "anv_private.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"

struct ycbcr_state {
   nir_builder *builder;
   nir_ssa_def *image_size;
   nir_tex_instr *origin_tex;
   nir_deref_instr *tex_deref;
   struct anv_ycbcr_conversion *conversion;
};

static nir_ssa_def *
y_range(nir_builder *b,
        nir_ssa_def *y_channel,
        int bpc,
        VkSamplerYcbcrRangeKHR range)
{
   switch (range) {
   case VK_SAMPLER_YCBCR_RANGE_ITU_FULL:
      return y_channel;
   case VK_SAMPLER_YCBCR_RANGE_ITU_NARROW:
      return nir_fmul(b,
                      nir_fadd(b,
                               nir_fmul(b, y_channel,
                                        nir_imm_float(b, pow(2, bpc) - 1)),
                               nir_imm_float(b, -16.0f * pow(2, bpc - 8))),
                      nir_frcp(b, nir_imm_float(b, 219.0f * pow(2, bpc - 8))));
   default:
      unreachable("missing Ycbcr range");
      return NULL;
   }
}

static nir_ssa_def *
chroma_range(nir_builder *b,
             nir_ssa_def *chroma_channel,
             int bpc,
             VkSamplerYcbcrRangeKHR range)
{
   switch (range) {
   case VK_SAMPLER_YCBCR_RANGE_ITU_FULL:
      return nir_fadd(b, chroma_channel,
                      nir_imm_float(b, -pow(2, bpc - 1) / (pow(2, bpc) - 1.0f)));
   case VK_SAMPLER_YCBCR_RANGE_ITU_NARROW:
      return nir_fmul(b,
                      nir_fadd(b,
                               nir_fmul(b, chroma_channel,
                                        nir_imm_float(b, pow(2, bpc) - 1)),
                               nir_imm_float(b, -128.0f * pow(2, bpc - 8))),
                      nir_frcp(b, nir_imm_float(b, 224.0f * pow(2, bpc - 8))));
   default:
      unreachable("missing Ycbcr range");
      return NULL;
   }
}

static const nir_const_value *
ycbcr_model_to_rgb_matrix(VkSamplerYcbcrModelConversionKHR model)
{
   switch (model) {
   case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601: {
      static const nir_const_value bt601[3] = {
         { .f32 = {  1.402f,             1.0f,  0.0f,               0.0f } },
         { .f32 = { -0.714136286201022f, 1.0f, -0.344136286201022f, 0.0f } },
         { .f32 = {  0.0f,               1.0f,  1.772f,             0.0f } }
      };

      return bt601;
   }
   case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709: {
      static const nir_const_value bt709[3] = {
         { .f32 = {  1.5748031496063f,   1.0f,  0.0,                0.0f } },
         { .f32 = { -0.468125209181067f, 1.0f, -0.187327487470334f, 0.0f } },
         { .f32 = {  0.0f,               1.0f,  1.85563184264242f,  0.0f } }
      };

      return bt709;
   }
   case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020: {
      static const nir_const_value bt2020[3] = {
         { .f32 = { 1.4746f,             1.0f,  0.0f,               0.0f } },
         { .f32 = { -0.571353126843658f, 1.0f, -0.164553126843658f, 0.0f } },
         { .f32 = { 0.0f,                1.0f,  1.8814f,            0.0f } }
      };

      return bt2020;
   }
   default:
      unreachable("missing Ycbcr model");
      return NULL;
   }
}

static nir_ssa_def *
convert_ycbcr(struct ycbcr_state *state,
              nir_ssa_def *raw_channels,
              uint32_t *bpcs)
{
   nir_builder *b = state->builder;
   struct anv_ycbcr_conversion *conversion = state->conversion;

   nir_ssa_def *expanded_channels =
      nir_vec4(b,
               chroma_range(b, nir_channel(b, raw_channels, 0),
                            bpcs[0], conversion->ycbcr_range),
               y_range(b, nir_channel(b, raw_channels, 1),
                       bpcs[1], conversion->ycbcr_range),
               chroma_range(b, nir_channel(b, raw_channels, 2),
                            bpcs[2], conversion->ycbcr_range),
               nir_imm_float(b, 1.0f));

   if (conversion->ycbcr_model == VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY)
      return expanded_channels;

   const nir_const_value *conversion_matrix =
      ycbcr_model_to_rgb_matrix(conversion->ycbcr_model);

   nir_ssa_def *converted_channels[] = {
      nir_fdot4(b, expanded_channels, nir_build_imm(b, 4, 32, conversion_matrix[0])),
      nir_fdot4(b, expanded_channels, nir_build_imm(b, 4, 32, conversion_matrix[1])),
      nir_fdot4(b, expanded_channels, nir_build_imm(b, 4, 32, conversion_matrix[2]))
   };

   return nir_vec4(b,
                   converted_channels[0], converted_channels[1],
                   converted_channels[2], nir_imm_float(b, 1.0f));
}

/* TODO: we should probably replace this with a push constant/uniform. */
static nir_ssa_def *
get_texture_size(struct ycbcr_state *state, nir_deref_instr *texture)
{
   if (state->image_size)
      return state->image_size;

   nir_builder *b = state->builder;
   const struct glsl_type *type = texture->type;
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 1);

   tex->op = nir_texop_txs;
   tex->sampler_dim = glsl_get_sampler_dim(type);
   tex->is_array = glsl_sampler_type_is_array(type);
   tex->is_shadow = glsl_sampler_type_is_shadow(type);
   tex->dest_type = nir_type_int;

   tex->src[0].src_type = nir_tex_src_texture_deref;
   tex->src[0].src = nir_src_for_ssa(&texture->dest.ssa);

   nir_ssa_dest_init(&tex->instr, &tex->dest,
                     nir_tex_instr_dest_size(tex), 32, NULL);
   nir_builder_instr_insert(b, &tex->instr);

   state->image_size = nir_i2f32(b, &tex->dest.ssa);

   return state->image_size;
}

static nir_ssa_def *
implicit_downsampled_coord(nir_builder *b,
                           nir_ssa_def *value,
                           nir_ssa_def *max_value,
                           int div_scale)
{
   return nir_fadd(b,
                   value,
                   nir_fdiv(b,
                            nir_imm_float(b, 1.0f),
                            nir_fmul(b,
                                     nir_imm_float(b, div_scale),
                                     max_value)));
}

static nir_ssa_def *
implicit_downsampled_coords(struct ycbcr_state *state,
                            nir_ssa_def *old_coords,
                            const struct anv_format_plane *plane_format)
{
   nir_builder *b = state->builder;
   struct anv_ycbcr_conversion *conversion = state->conversion;
   nir_ssa_def *image_size = get_texture_size(state, state->tex_deref);
   nir_ssa_def *comp[4] = { NULL, };
   int c;

   for (c = 0; c < ARRAY_SIZE(conversion->chroma_offsets); c++) {
      if (plane_format->denominator_scales[c] > 1 &&
          conversion->chroma_offsets[c] == VK_CHROMA_LOCATION_COSITED_EVEN) {
         comp[c] = implicit_downsampled_coord(b,
                                              nir_channel(b, old_coords, c),
                                              nir_channel(b, image_size, c),
                                              plane_format->denominator_scales[c]);
      } else {
         comp[c] = nir_channel(b, old_coords, c);
      }
   }

   /* Leave other coordinates untouched */
   for (; c < old_coords->num_components; c++)
      comp[c] = nir_channel(b, old_coords, c);

   return nir_vec(b, comp, old_coords->num_components);
}

static nir_ssa_def *
create_plane_tex_instr_implicit(struct ycbcr_state *state,
                                uint32_t plane)
{
   nir_builder *b = state->builder;
   struct anv_ycbcr_conversion *conversion = state->conversion;
   const struct anv_format_plane *plane_format =
      &conversion->format->planes[plane];
   nir_tex_instr *old_tex = state->origin_tex;
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, old_tex->num_srcs + 1);

   for (uint32_t i = 0; i < old_tex->num_srcs; i++) {
      tex->src[i].src_type = old_tex->src[i].src_type;

      switch (old_tex->src[i].src_type) {
      case nir_tex_src_coord:
         if (plane_format->has_chroma && conversion->chroma_reconstruction) {
            assert(old_tex->src[i].src.is_ssa);
            tex->src[i].src =
               nir_src_for_ssa(implicit_downsampled_coords(state,
                                                           old_tex->src[i].src.ssa,
                                                           plane_format));
            break;
         }
         /* fall through */
      default:
         nir_src_copy(&tex->src[i].src, &old_tex->src[i].src, tex);
         break;
      }
   }
   tex->src[tex->num_srcs - 1].src = nir_src_for_ssa(nir_imm_int(b, plane));
   tex->src[tex->num_srcs - 1].src_type = nir_tex_src_plane;

   tex->sampler_dim = old_tex->sampler_dim;
   tex->dest_type = old_tex->dest_type;

   tex->op = old_tex->op;
   tex->coord_components = old_tex->coord_components;
   tex->is_new_style_shadow = old_tex->is_new_style_shadow;
   tex->component = old_tex->component;

   tex->texture_index = old_tex->texture_index;
   tex->texture_array_size = old_tex->texture_array_size;
   tex->sampler_index = old_tex->sampler_index;

   nir_ssa_dest_init(&tex->instr, &tex->dest,
                     old_tex->dest.ssa.num_components,
                     nir_dest_bit_size(old_tex->dest), NULL);
   nir_builder_instr_insert(b, &tex->instr);

   return &tex->dest.ssa;
}

static unsigned
channel_to_component(enum isl_channel_select channel)
{
   switch (channel) {
   case ISL_CHANNEL_SELECT_RED:
      return 0;
   case ISL_CHANNEL_SELECT_GREEN:
      return 1;
   case ISL_CHANNEL_SELECT_BLUE:
      return 2;
   case ISL_CHANNEL_SELECT_ALPHA:
      return 3;
   default:
      unreachable("invalid channel");
      return 0;
   }
}

static enum isl_channel_select
swizzle_channel(struct isl_swizzle swizzle, unsigned channel)
{
   switch (channel) {
   case 0:
      return swizzle.r;
   case 1:
      return swizzle.g;
   case 2:
      return swizzle.b;
   case 3:
      return swizzle.a;
   default:
      unreachable("invalid channel");
      return 0;
   }
}

static bool
try_lower_tex_ycbcr(struct anv_pipeline_layout *layout,
                    nir_builder *builder,
                    nir_tex_instr *tex)
{
   int deref_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   assert(deref_src_idx >= 0);
   nir_deref_instr *deref = nir_src_as_deref(tex->src[deref_src_idx].src);

   nir_variable *var = nir_deref_instr_get_variable(deref);
   const struct anv_descriptor_set_layout *set_layout =
      layout->set[var->data.descriptor_set].layout;
   const struct anv_descriptor_set_binding_layout *binding =
      &set_layout->binding[var->data.binding];

   /* For the following instructions, we don't apply any change and let the
    * instruction apply to the first plane.
    */
   if (tex->op == nir_texop_txs ||
       tex->op == nir_texop_query_levels ||
       tex->op == nir_texop_lod)
      return false;

   if (binding->immutable_samplers == NULL)
      return false;

   assert(tex->texture_index == 0);
   unsigned array_index = 0;
   if (deref->deref_type != nir_deref_type_var) {
      assert(deref->deref_type == nir_deref_type_array);
      nir_const_value *const_index = nir_src_as_const_value(deref->arr.index);
      if (!const_index)
         return false;
      array_index = MIN2(const_index->u32[0], binding->array_size - 1);
   }
   const struct anv_sampler *sampler = binding->immutable_samplers[array_index];

   if (sampler->conversion == NULL)
      return false;

   struct ycbcr_state state = {
      .builder = builder,
      .origin_tex = tex,
      .tex_deref = deref,
      .conversion = sampler->conversion,
   };

   builder->cursor = nir_before_instr(&tex->instr);

   const struct anv_format *format = state.conversion->format;
   const struct isl_format_layout *y_isl_layout = NULL;
   for (uint32_t p = 0; p < format->n_planes; p++) {
      if (!format->planes[p].has_chroma)
         y_isl_layout = isl_format_get_layout(format->planes[p].isl_format);
   }
   assert(y_isl_layout != NULL);
   uint8_t y_bpc = y_isl_layout->channels_array[0].bits;

   /* |ycbcr_comp| holds components in the order : Cr-Y-Cb */
   nir_ssa_def *ycbcr_comp[5] = { NULL, NULL, NULL,
                                  /* Use extra 2 channels for following swizzle */
                                  nir_imm_float(builder, 1.0f),
                                  nir_imm_float(builder, 0.0f),
   };
   uint8_t ycbcr_bpcs[5];
   memset(ycbcr_bpcs, y_bpc, sizeof(ycbcr_bpcs));

   /* Go through all the planes and gather the samples into a |ycbcr_comp|
    * while applying a swizzle required by the spec:
    *
    *    R, G, B should respectively map to Cr, Y, Cb
    */
   for (uint32_t p = 0; p < format->n_planes; p++) {
      const struct anv_format_plane *plane_format = &format->planes[p];
      nir_ssa_def *plane_sample = create_plane_tex_instr_implicit(&state, p);

      for (uint32_t pc = 0; pc < 4; pc++) {
         enum isl_channel_select ycbcr_swizzle =
            swizzle_channel(plane_format->ycbcr_swizzle, pc);
         if (ycbcr_swizzle == ISL_CHANNEL_SELECT_ZERO)
            continue;

         unsigned ycbcr_component = channel_to_component(ycbcr_swizzle);
         ycbcr_comp[ycbcr_component] = nir_channel(builder, plane_sample, pc);

         /* Also compute the number of bits for each component. */
         const struct isl_format_layout *isl_layout =
            isl_format_get_layout(plane_format->isl_format);
         ycbcr_bpcs[ycbcr_component] = isl_layout->channels_array[pc].bits;
      }
   }

   /* Now remaps components to the order specified by the conversion. */
   nir_ssa_def *swizzled_comp[4] = { NULL, };
   uint32_t swizzled_bpcs[4] = { 0, };

   for (uint32_t i = 0; i < ARRAY_SIZE(state.conversion->mapping); i++) {
      /* Maps to components in |ycbcr_comp| */
      static const uint32_t swizzle_mapping[] = {
         [VK_COMPONENT_SWIZZLE_ZERO] = 4,
         [VK_COMPONENT_SWIZZLE_ONE]  = 3,
         [VK_COMPONENT_SWIZZLE_R]    = 0,
         [VK_COMPONENT_SWIZZLE_G]    = 1,
         [VK_COMPONENT_SWIZZLE_B]    = 2,
         [VK_COMPONENT_SWIZZLE_A]    = 3,
      };
      const VkComponentSwizzle m = state.conversion->mapping[i];

      if (m == VK_COMPONENT_SWIZZLE_IDENTITY) {
         swizzled_comp[i] = ycbcr_comp[i];
         swizzled_bpcs[i] = ycbcr_bpcs[i];
      } else {
         swizzled_comp[i] = ycbcr_comp[swizzle_mapping[m]];
         swizzled_bpcs[i] = ycbcr_bpcs[swizzle_mapping[m]];
      }
   }

   nir_ssa_def *result = nir_vec(builder, swizzled_comp, 4);
   if (state.conversion->ycbcr_model != VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY)
      result = convert_ycbcr(&state, result, swizzled_bpcs);

   nir_ssa_def_rewrite_uses(&tex->dest.ssa, nir_src_for_ssa(result));
   nir_instr_remove(&tex->instr);

   return true;
}

bool
anv_nir_lower_ycbcr_textures(nir_shader *shader,
                             struct anv_pipeline_layout *layout)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;

      bool function_progress = false;
      nir_builder builder;
      nir_builder_init(&builder, function->impl);

      nir_foreach_block(block, function->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);
            function_progress |= try_lower_tex_ycbcr(layout, &builder, tex);
         }
      }

      if (function_progress) {
         nir_metadata_preserve(function->impl,
                               nir_metadata_block_index |
                               nir_metadata_dominance);
      }

      progress |= function_progress;
   }

   return progress;
}
