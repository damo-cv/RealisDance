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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "util/ralloc.h"

#include "main/macros.h" /* Needed for MAX3 and MAX2 for format_rgb9e5 */
#include "util/format_rgb9e5.h"
#include "util/format_srgb.h"

#include "blorp_priv.h"
#include "compiler/brw_eu_defines.h"

#include "blorp_nir_builder.h"

#define FILE_DEBUG_FLAG DEBUG_BLORP

struct brw_blorp_const_color_prog_key
{
   enum blorp_shader_type shader_type; /* Must be BLORP_SHADER_TYPE_CLEAR */
   bool use_simd16_replicated_data;
   bool clear_rgb_as_red;
   bool pad[3];
};

static bool
blorp_params_get_clear_kernel(struct blorp_context *blorp,
                              struct blorp_params *params,
                              bool use_replicated_data,
                              bool clear_rgb_as_red)
{
   const struct brw_blorp_const_color_prog_key blorp_key = {
      .shader_type = BLORP_SHADER_TYPE_CLEAR,
      .use_simd16_replicated_data = use_replicated_data,
      .clear_rgb_as_red = clear_rgb_as_red,
   };

   if (blorp->lookup_shader(blorp, &blorp_key, sizeof(blorp_key),
                            &params->wm_prog_kernel, &params->wm_prog_data))
      return true;

   void *mem_ctx = ralloc_context(NULL);

   nir_builder b;
   nir_builder_init_simple_shader(&b, mem_ctx, MESA_SHADER_FRAGMENT, NULL);
   b.shader->info.name = ralloc_strdup(b.shader, "BLORP-clear");

   nir_variable *v_color =
      BLORP_CREATE_NIR_INPUT(b.shader, clear_color, glsl_vec4_type());
   nir_ssa_def *color = nir_load_var(&b, v_color);

   if (clear_rgb_as_red) {
      nir_variable *frag_coord =
         nir_variable_create(b.shader, nir_var_shader_in,
                             glsl_vec4_type(), "gl_FragCoord");
      frag_coord->data.location = VARYING_SLOT_POS;
      frag_coord->data.origin_upper_left = true;

      nir_ssa_def *pos = nir_f2i32(&b, nir_load_var(&b, frag_coord));
      nir_ssa_def *comp = nir_umod(&b, nir_channel(&b, pos, 0),
                                       nir_imm_int(&b, 3));
      nir_ssa_def *color_component =
         nir_bcsel(&b, nir_ieq(&b, comp, nir_imm_int(&b, 0)),
                       nir_channel(&b, color, 0),
                       nir_bcsel(&b, nir_ieq(&b, comp, nir_imm_int(&b, 1)),
                                     nir_channel(&b, color, 1),
                                     nir_channel(&b, color, 2)));

      nir_ssa_def *u = nir_ssa_undef(&b, 1, 32);
      color = nir_vec4(&b, color_component, u, u, u);
   }

   nir_variable *frag_color = nir_variable_create(b.shader, nir_var_shader_out,
                                                  glsl_vec4_type(),
                                                  "gl_FragColor");
   frag_color->data.location = FRAG_RESULT_COLOR;
   nir_store_var(&b, frag_color, color, 0xf);

   struct brw_wm_prog_key wm_key;
   brw_blorp_init_wm_prog_key(&wm_key);

   struct brw_wm_prog_data prog_data;
   const unsigned *program =
      blorp_compile_fs(blorp, mem_ctx, b.shader, &wm_key, use_replicated_data,
                       &prog_data);

   bool result =
      blorp->upload_shader(blorp, &blorp_key, sizeof(blorp_key),
                           program, prog_data.base.program_size,
                           &prog_data.base, sizeof(prog_data),
                           &params->wm_prog_kernel, &params->wm_prog_data);

   ralloc_free(mem_ctx);
   return result;
}

struct layer_offset_vs_key {
   enum blorp_shader_type shader_type;
   unsigned num_inputs;
};

/* In the case of doing attachment clears, we are using a surface state that
 * is handed to us so we can't set (and don't even know) the base array layer.
 * In order to do a layered clear in this scenario, we need some way of adding
 * the base array layer to the instance id.  Unfortunately, our hardware has
 * no real concept of "base instance", so we have to do it manually in a
 * vertex shader.
 */
static bool
blorp_params_get_layer_offset_vs(struct blorp_context *blorp,
                                 struct blorp_params *params)
{
   struct layer_offset_vs_key blorp_key = {
      .shader_type = BLORP_SHADER_TYPE_LAYER_OFFSET_VS,
   };

   if (params->wm_prog_data)
      blorp_key.num_inputs = params->wm_prog_data->num_varying_inputs;

   if (blorp->lookup_shader(blorp, &blorp_key, sizeof(blorp_key),
                            &params->vs_prog_kernel, &params->vs_prog_data))
      return true;

   void *mem_ctx = ralloc_context(NULL);

   nir_builder b;
   nir_builder_init_simple_shader(&b, mem_ctx, MESA_SHADER_VERTEX, NULL);
   b.shader->info.name = ralloc_strdup(b.shader, "BLORP-layer-offset-vs");

   const struct glsl_type *uvec4_type = glsl_vector_type(GLSL_TYPE_UINT, 4);

   /* First we deal with the header which has instance and base instance */
   nir_variable *a_header = nir_variable_create(b.shader, nir_var_shader_in,
                                                uvec4_type, "header");
   a_header->data.location = VERT_ATTRIB_GENERIC0;

   nir_variable *v_layer = nir_variable_create(b.shader, nir_var_shader_out,
                                               glsl_int_type(), "layer_id");
   v_layer->data.location = VARYING_SLOT_LAYER;

   /* Compute the layer id */
   nir_ssa_def *header = nir_load_var(&b, a_header);
   nir_ssa_def *base_layer = nir_channel(&b, header, 0);
   nir_ssa_def *instance = nir_channel(&b, header, 1);
   nir_store_var(&b, v_layer, nir_iadd(&b, instance, base_layer), 0x1);

   /* Then we copy the vertex from the next slot to VARYING_SLOT_POS */
   nir_variable *a_vertex = nir_variable_create(b.shader, nir_var_shader_in,
                                                glsl_vec4_type(), "a_vertex");
   a_vertex->data.location = VERT_ATTRIB_GENERIC1;

   nir_variable *v_pos = nir_variable_create(b.shader, nir_var_shader_out,
                                             glsl_vec4_type(), "v_pos");
   v_pos->data.location = VARYING_SLOT_POS;

   nir_copy_var(&b, v_pos, a_vertex);

   /* Then we copy everything else */
   for (unsigned i = 0; i < blorp_key.num_inputs; i++) {
      nir_variable *a_in = nir_variable_create(b.shader, nir_var_shader_in,
                                               uvec4_type, "input");
      a_in->data.location = VERT_ATTRIB_GENERIC2 + i;

      nir_variable *v_out = nir_variable_create(b.shader, nir_var_shader_out,
                                                uvec4_type, "output");
      v_out->data.location = VARYING_SLOT_VAR0 + i;

      nir_copy_var(&b, v_out, a_in);
   }

   struct brw_vs_prog_data vs_prog_data;
   memset(&vs_prog_data, 0, sizeof(vs_prog_data));

   const unsigned *program =
      blorp_compile_vs(blorp, mem_ctx, b.shader, &vs_prog_data);

   bool result =
      blorp->upload_shader(blorp, &blorp_key, sizeof(blorp_key),
                           program, vs_prog_data.base.base.program_size,
                           &vs_prog_data.base.base, sizeof(vs_prog_data),
                           &params->vs_prog_kernel, &params->vs_prog_data);

   ralloc_free(mem_ctx);
   return result;
}

/* The x0, y0, x1, and y1 parameters must already be populated with the render
 * area of the framebuffer to be cleared.
 */
static void
get_fast_clear_rect(const struct isl_device *dev,
                    const struct isl_surf *aux_surf,
                    unsigned *x0, unsigned *y0,
                    unsigned *x1, unsigned *y1)
{
   unsigned int x_align, y_align;
   unsigned int x_scaledown, y_scaledown;

   /* Only single sampled surfaces need to (and actually can) be resolved. */
   if (aux_surf->usage == ISL_SURF_USAGE_CCS_BIT) {
      /* From the Ivy Bridge PRM, Vol2 Part1 11.7 "MCS Buffer for Render
       * Target(s)", beneath the "Fast Color Clear" bullet (p327):
       *
       *     Clear pass must have a clear rectangle that must follow
       *     alignment rules in terms of pixels and lines as shown in the
       *     table below. Further, the clear-rectangle height and width
       *     must be multiple of the following dimensions. If the height
       *     and width of the render target being cleared do not meet these
       *     requirements, an MCS buffer can be created such that it
       *     follows the requirement and covers the RT.
       *
       * The alignment size in the table that follows is related to the
       * alignment size that is baked into the CCS surface format but with X
       * alignment multiplied by 16 and Y alignment multiplied by 32.
       */
      x_align = isl_format_get_layout(aux_surf->format)->bw;
      y_align = isl_format_get_layout(aux_surf->format)->bh;

      x_align *= 16;

      /* SKL+ line alignment requirement for Y-tiled are half those of the prior
       * generations.
       */
      if (dev->info->gen >= 9)
         y_align *= 16;
      else
         y_align *= 32;

      /* From the Ivy Bridge PRM, Vol2 Part1 11.7 "MCS Buffer for Render
       * Target(s)", beneath the "Fast Color Clear" bullet (p327):
       *
       *     In order to optimize the performance MCS buffer (when bound to
       *     1X RT) clear similarly to MCS buffer clear for MSRT case,
       *     clear rect is required to be scaled by the following factors
       *     in the horizontal and vertical directions:
       *
       * The X and Y scale down factors in the table that follows are each
       * equal to half the alignment value computed above.
       */
      x_scaledown = x_align / 2;
      y_scaledown = y_align / 2;

      /* From BSpec: 3D-Media-GPGPU Engine > 3D Pipeline > Pixel > Pixel
       * Backend > MCS Buffer for Render Target(s) [DevIVB+] > Table "Color
       * Clear of Non-MultiSampled Render Target Restrictions":
       *
       *   Clear rectangle must be aligned to two times the number of
       *   pixels in the table shown below due to 16x16 hashing across the
       *   slice.
       */
      x_align *= 2;
      y_align *= 2;
   } else {
      assert(aux_surf->usage == ISL_SURF_USAGE_MCS_BIT);

      /* From the Ivy Bridge PRM, Vol2 Part1 11.7 "MCS Buffer for Render
       * Target(s)", beneath the "MSAA Compression" bullet (p326):
       *
       *     Clear pass for this case requires that scaled down primitive
       *     is sent down with upper left co-ordinate to coincide with
       *     actual rectangle being cleared. For MSAA, clear rectangle’s
       *     height and width need to as show in the following table in
       *     terms of (width,height) of the RT.
       *
       *     MSAA  Width of Clear Rect  Height of Clear Rect
       *      2X     Ceil(1/8*width)      Ceil(1/2*height)
       *      4X     Ceil(1/8*width)      Ceil(1/2*height)
       *      8X     Ceil(1/2*width)      Ceil(1/2*height)
       *     16X         width            Ceil(1/2*height)
       *
       * The text "with upper left co-ordinate to coincide with actual
       * rectangle being cleared" is a little confusing--it seems to imply
       * that to clear a rectangle from (x,y) to (x+w,y+h), one needs to
       * feed the pipeline using the rectangle (x,y) to
       * (x+Ceil(w/N),y+Ceil(h/2)), where N is either 2 or 8 depending on
       * the number of samples.  Experiments indicate that this is not
       * quite correct; actually, what the hardware appears to do is to
       * align whatever rectangle is sent down the pipeline to the nearest
       * multiple of 2x2 blocks, and then scale it up by a factor of N
       * horizontally and 2 vertically.  So the resulting alignment is 4
       * vertically and either 4 or 16 horizontally, and the scaledown
       * factor is 2 vertically and either 2 or 8 horizontally.
       */
      switch (aux_surf->format) {
      case ISL_FORMAT_MCS_2X:
      case ISL_FORMAT_MCS_4X:
         x_scaledown = 8;
         break;
      case ISL_FORMAT_MCS_8X:
         x_scaledown = 2;
         break;
      case ISL_FORMAT_MCS_16X:
         x_scaledown = 1;
         break;
      default:
         unreachable("Unexpected MCS format for fast clear");
      }
      y_scaledown = 2;
      x_align = x_scaledown * 2;
      y_align = y_scaledown * 2;
   }

   *x0 = ROUND_DOWN_TO(*x0,  x_align) / x_scaledown;
   *y0 = ROUND_DOWN_TO(*y0, y_align) / y_scaledown;
   *x1 = ALIGN(*x1, x_align) / x_scaledown;
   *y1 = ALIGN(*y1, y_align) / y_scaledown;
}

void
blorp_fast_clear(struct blorp_batch *batch,
                 const struct blorp_surf *surf, enum isl_format format,
                 uint32_t level, uint32_t start_layer, uint32_t num_layers,
                 uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
{
   /* Ensure that all layers undergoing the clear have an auxiliary buffer. */
   assert(start_layer + num_layers <=
          MAX2(surf->aux_surf->logical_level0_px.depth >> level,
               surf->aux_surf->logical_level0_px.array_len));

   struct blorp_params params;
   blorp_params_init(&params);
   params.num_layers = num_layers;

   params.x0 = x0;
   params.y0 = y0;
   params.x1 = x1;
   params.y1 = y1;

   memset(&params.wm_inputs.clear_color, 0xff, 4*sizeof(float));
   params.fast_clear_op = ISL_AUX_OP_FAST_CLEAR;

   get_fast_clear_rect(batch->blorp->isl_dev, surf->aux_surf,
                       &params.x0, &params.y0, &params.x1, &params.y1);

   if (!blorp_params_get_clear_kernel(batch->blorp, &params, true, false))
      return;

   brw_blorp_surface_info_init(batch->blorp, &params.dst, surf, level,
                               start_layer, format, true);
   params.num_samples = params.dst.surf.samples;

   batch->blorp->exec(batch, &params);
}

static union isl_color_value
swizzle_color_value(union isl_color_value src, struct isl_swizzle swizzle)
{
   union isl_color_value dst = { .u32 = { 0, } };

   /* We assign colors in ABGR order so that the first one will be taken in
    * RGBA precedence order.  According to the PRM docs for shader channel
    * select, this matches Haswell hardware behavior.
    */
   if ((unsigned)(swizzle.a - ISL_CHANNEL_SELECT_RED) < 4)
      dst.u32[swizzle.a - ISL_CHANNEL_SELECT_RED] = src.u32[3];
   if ((unsigned)(swizzle.b - ISL_CHANNEL_SELECT_RED) < 4)
      dst.u32[swizzle.b - ISL_CHANNEL_SELECT_RED] = src.u32[2];
   if ((unsigned)(swizzle.g - ISL_CHANNEL_SELECT_RED) < 4)
      dst.u32[swizzle.g - ISL_CHANNEL_SELECT_RED] = src.u32[1];
   if ((unsigned)(swizzle.r - ISL_CHANNEL_SELECT_RED) < 4)
      dst.u32[swizzle.r - ISL_CHANNEL_SELECT_RED] = src.u32[0];

   return dst;
}

void
blorp_clear(struct blorp_batch *batch,
            const struct blorp_surf *surf,
            enum isl_format format, struct isl_swizzle swizzle,
            uint32_t level, uint32_t start_layer, uint32_t num_layers,
            uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
            union isl_color_value clear_color,
            const bool color_write_disable[4])
{
   struct blorp_params params;
   blorp_params_init(&params);

   /* Manually apply the clear destination swizzle.  This way swizzled clears
    * will work for swizzles which we can't normally use for rendering and it
    * also ensures that they work on pre-Haswell hardware which can't swizlle
    * at all.
    */
   clear_color = swizzle_color_value(clear_color, swizzle);
   swizzle = ISL_SWIZZLE_IDENTITY;

   bool clear_rgb_as_red = false;
   if (format == ISL_FORMAT_R9G9B9E5_SHAREDEXP) {
      clear_color.u32[0] = float3_to_rgb9e5(clear_color.f32);
      format = ISL_FORMAT_R32_UINT;
   } else if (format == ISL_FORMAT_L8_UNORM_SRGB) {
      clear_color.f32[0] = util_format_linear_to_srgb_float(clear_color.f32[0]);
      format = ISL_FORMAT_R8_UNORM;
   } else if (format == ISL_FORMAT_A4B4G4R4_UNORM) {
      /* Broadwell and earlier cannot render to this format so we need to work
       * around it by swapping the colors around and using B4G4R4A4 instead.
       */
      const struct isl_swizzle ARGB = ISL_SWIZZLE(ALPHA, RED, GREEN, BLUE);
      clear_color = swizzle_color_value(clear_color, ARGB);
      format = ISL_FORMAT_B4G4R4A4_UNORM;
   } else if (isl_format_get_layout(format)->bpb % 3 == 0) {
      clear_rgb_as_red = true;
      if (format == ISL_FORMAT_R8G8B8_UNORM_SRGB) {
         clear_color.f32[0] = util_format_linear_to_srgb_float(clear_color.f32[0]);
         clear_color.f32[1] = util_format_linear_to_srgb_float(clear_color.f32[1]);
         clear_color.f32[2] = util_format_linear_to_srgb_float(clear_color.f32[2]);
      }
   }

   memcpy(&params.wm_inputs.clear_color, clear_color.f32, sizeof(float) * 4);

   bool use_simd16_replicated_data = true;

   /* From the SNB PRM (Vol4_Part1):
    *
    *     "Replicated data (Message Type = 111) is only supported when
    *      accessing tiled memory.  Using this Message Type to access linear
    *      (untiled) memory is UNDEFINED."
    */
   if (surf->surf->tiling == ISL_TILING_LINEAR)
      use_simd16_replicated_data = false;

   /* Replicated clears don't work yet before gen6 */
   if (batch->blorp->isl_dev->info->gen < 6)
      use_simd16_replicated_data = false;

   /* Constant color writes ignore everyting in blend and color calculator
    * state.  This is not documented.
    */
   if (color_write_disable) {
      for (unsigned i = 0; i < 4; i++) {
         params.color_write_disable[i] = color_write_disable[i];
         if (color_write_disable[i])
            use_simd16_replicated_data = false;
      }
   }

   if (!blorp_params_get_clear_kernel(batch->blorp, &params,
                                      use_simd16_replicated_data,
                                      clear_rgb_as_red))
      return;

   if (!blorp_ensure_sf_program(batch->blorp, &params))
      return;

   while (num_layers > 0) {
      brw_blorp_surface_info_init(batch->blorp, &params.dst, surf, level,
                                  start_layer, format, true);
      params.dst.view.swizzle = swizzle;

      params.x0 = x0;
      params.y0 = y0;
      params.x1 = x1;
      params.y1 = y1;

      if (params.dst.tile_x_sa || params.dst.tile_y_sa) {
         assert(params.dst.surf.samples == 1);
         assert(num_layers == 1);
         params.x0 += params.dst.tile_x_sa;
         params.y0 += params.dst.tile_y_sa;
         params.x1 += params.dst.tile_x_sa;
         params.y1 += params.dst.tile_y_sa;
      }

      /* The MinLOD and MinimumArrayElement don't work properly for cube maps.
       * Convert them to a single slice on gen4.
       */
      if (batch->blorp->isl_dev->info->gen == 4 &&
          (params.dst.surf.usage & ISL_SURF_USAGE_CUBE_BIT)) {
         blorp_surf_convert_to_single_slice(batch->blorp->isl_dev, &params.dst);
      }

      if (clear_rgb_as_red) {
         surf_fake_rgb_with_red(batch->blorp->isl_dev, &params.dst);
         params.x0 *= 3;
         params.x1 *= 3;
      }

      if (isl_format_is_compressed(params.dst.surf.format)) {
         blorp_surf_convert_to_uncompressed(batch->blorp->isl_dev, &params.dst,
                                            NULL, NULL, NULL, NULL);
                                            //&dst_x, &dst_y, &dst_w, &dst_h);
      }

      if (params.dst.tile_x_sa || params.dst.tile_y_sa) {
         /* Either we're on gen4 where there is no multisampling or the
          * surface is compressed which also implies no multisampling.
          * Therefore, sa == px and we don't need to do a conversion.
          */
         assert(params.dst.surf.samples == 1);
         params.x0 += params.dst.tile_x_sa;
         params.y0 += params.dst.tile_y_sa;
         params.x1 += params.dst.tile_x_sa;
         params.y1 += params.dst.tile_y_sa;
      }

      params.num_samples = params.dst.surf.samples;

      /* We may be restricted on the number of layers we can bind at any one
       * time.  In particular, Sandy Bridge has a maximum number of layers of
       * 512 but a maximum 3D texture size is much larger.
       */
      params.num_layers = MIN2(params.dst.view.array_len, num_layers);

      const unsigned max_image_width = 16 * 1024;
      if (params.dst.surf.logical_level0_px.width > max_image_width) {
         /* Clearing an RGB image as red multiplies the surface width by 3
          * so it may now be too wide for the hardware surface limits.  We
          * have to break the clear up into pieces in order to clear wide
          * images.
          */
         assert(clear_rgb_as_red);
         assert(params.dst.surf.dim == ISL_SURF_DIM_2D);
         assert(params.dst.surf.tiling == ISL_TILING_LINEAR);
         assert(params.dst.surf.logical_level0_px.depth == 1);
         assert(params.dst.surf.logical_level0_px.array_len == 1);
         assert(params.dst.surf.levels == 1);
         assert(params.dst.surf.samples == 1);
         assert(params.dst.tile_x_sa == 0 || params.dst.tile_y_sa == 0);
         assert(params.dst.aux_usage == ISL_AUX_USAGE_NONE);

         /* max_image_width rounded down to a multiple of 3 */
         const unsigned max_fake_rgb_width = (max_image_width / 3) * 3;
         const unsigned cpp =
            isl_format_get_layout(params.dst.surf.format)->bpb / 8;

         params.dst.surf.logical_level0_px.width = max_fake_rgb_width;
         params.dst.surf.phys_level0_sa.width = max_fake_rgb_width;

         uint32_t orig_x0 = params.x0, orig_x1 = params.x1;
         uint64_t orig_offset = params.dst.addr.offset;
         for (uint32_t x = orig_x0; x < orig_x1; x += max_fake_rgb_width) {
            /* Offset to the surface.  It's easy because we're linear */
            params.dst.addr.offset = orig_offset + x * cpp;

            params.x0 = 0;
            params.x1 = MIN2(orig_x1 - x, max_image_width);

            batch->blorp->exec(batch, &params);
         }
      } else {
         batch->blorp->exec(batch, &params);
      }

      start_layer += params.num_layers;
      num_layers -= params.num_layers;
   }
}

void
blorp_clear_depth_stencil(struct blorp_batch *batch,
                          const struct blorp_surf *depth,
                          const struct blorp_surf *stencil,
                          uint32_t level, uint32_t start_layer,
                          uint32_t num_layers,
                          uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                          bool clear_depth, float depth_value,
                          uint8_t stencil_mask, uint8_t stencil_value)
{
   struct blorp_params params;
   blorp_params_init(&params);

   params.x0 = x0;
   params.y0 = y0;
   params.x1 = x1;
   params.y1 = y1;

   if (ISL_DEV_GEN(batch->blorp->isl_dev) == 6) {
      /* For some reason, Sandy Bridge gets occlusion queries wrong if we
       * don't have a shader.  In particular, it records samples even though
       * we disable statistics in 3DSTATE_WM.  Give it the usual clear shader
       * to work around the issue.
       */
      if (!blorp_params_get_clear_kernel(batch->blorp, &params, false, false))
         return;
   }

   while (num_layers > 0) {
      params.num_layers = num_layers;

      if (stencil_mask) {
         brw_blorp_surface_info_init(batch->blorp, &params.stencil, stencil,
                                     level, start_layer,
                                     ISL_FORMAT_UNSUPPORTED, true);
         params.stencil_mask = stencil_mask;
         params.stencil_ref = stencil_value;

         params.dst.surf.samples = params.stencil.surf.samples;
         params.dst.surf.logical_level0_px =
            params.stencil.surf.logical_level0_px;
         params.dst.view = params.depth.view;

         params.num_samples = params.stencil.surf.samples;

         /* We may be restricted on the number of layers we can bind at any
          * one time.  In particular, Sandy Bridge has a maximum number of
          * layers of 512 but a maximum 3D texture size is much larger.
          */
         if (params.stencil.view.array_len < params.num_layers)
            params.num_layers = params.stencil.view.array_len;
      }

      if (clear_depth) {
         brw_blorp_surface_info_init(batch->blorp, &params.depth, depth,
                                     level, start_layer,
                                     ISL_FORMAT_UNSUPPORTED, true);
         params.z = depth_value;
         params.depth_format =
            isl_format_get_depth_format(depth->surf->format, false);

         params.dst.surf.samples = params.depth.surf.samples;
         params.dst.surf.logical_level0_px =
            params.depth.surf.logical_level0_px;
         params.dst.view = params.depth.view;

         params.num_samples = params.depth.surf.samples;

         /* We may be restricted on the number of layers we can bind at any
          * one time.  In particular, Sandy Bridge has a maximum number of
          * layers of 512 but a maximum 3D texture size is much larger.
          */
         if (params.depth.view.array_len < params.num_layers)
            params.num_layers = params.depth.view.array_len;
      }

      batch->blorp->exec(batch, &params);

      start_layer += params.num_layers;
      num_layers -= params.num_layers;
   }
}

bool
blorp_can_hiz_clear_depth(uint8_t gen, enum isl_format format,
                          uint32_t num_samples,
                          uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
{
   /* This function currently doesn't support any gen prior to gen8 */
   assert(gen >= 8);

   if (gen == 8 && format == ISL_FORMAT_R16_UNORM) {
      /* Apply the D16 alignment restrictions. On BDW, HiZ has an 8x4 sample
       * block with the following property: as the number of samples increases,
       * the number of pixels representable by this block decreases by a factor
       * of the sample dimensions. Sample dimensions scale following the MSAA
       * interleaved pattern.
       *
       * Sample|Sample|Pixel
       * Count |Dim   |Dim
       * ===================
       *    1  | 1x1  | 8x4
       *    2  | 2x1  | 4x4
       *    4  | 2x2  | 4x2
       *    8  | 4x2  | 2x2
       *   16  | 4x4  | 2x1
       *
       * Table: Pixel Dimensions in a HiZ Sample Block Pre-SKL
       */
      const struct isl_extent2d sa_block_dim =
         isl_get_interleaved_msaa_px_size_sa(num_samples);
      const uint8_t align_px_w = 8 / sa_block_dim.w;
      const uint8_t align_px_h = 4 / sa_block_dim.h;

      /* Fast depth clears clear an entire sample block at a time. As a result,
       * the rectangle must be aligned to the dimensions of the encompassing
       * pixel block for a successful operation.
       *
       * Fast clears can still work if the upper-left corner is aligned and the
       * bottom-rigtht corner touches the edge of a depth buffer whose extent
       * is unaligned. This is because each miplevel in the depth buffer is
       * padded by the Pixel Dim (similar to a standard compressed texture).
       * In this case, the clear rectangle could be padded by to match the full
       * depth buffer extent but to support multiple clearing techniques, we
       * chose to be unaware of the depth buffer's extent and thus don't handle
       * this case.
       */
      if (x0 % align_px_w || y0 % align_px_h ||
          x1 % align_px_w || y1 % align_px_h)
         return false;
   }
   return true;
}

void
blorp_hiz_clear_depth_stencil(struct blorp_batch *batch,
                              const struct blorp_surf *depth,
                              const struct blorp_surf *stencil,
                              uint32_t level,
                              uint32_t start_layer, uint32_t num_layers,
                              uint32_t x0, uint32_t y0,
                              uint32_t x1, uint32_t y1,
                              bool clear_depth, float depth_value,
                              bool clear_stencil, uint8_t stencil_value)
{
   struct blorp_params params;
   blorp_params_init(&params);

   /* This requires WM_HZ_OP which only exists on gen8+ */
   assert(ISL_DEV_GEN(batch->blorp->isl_dev) >= 8);

   params.hiz_op = ISL_AUX_OP_FAST_CLEAR;
   params.num_layers = 1;

   params.x0 = x0;
   params.y0 = y0;
   params.x1 = x1;
   params.y1 = y1;

   for (uint32_t l = 0; l < num_layers; l++) {
      const uint32_t layer = start_layer + l;
      if (clear_stencil) {
         brw_blorp_surface_info_init(batch->blorp, &params.stencil, stencil,
                                     level, layer,
                                     ISL_FORMAT_UNSUPPORTED, true);
         params.stencil_mask = 0xff;
         params.stencil_ref = stencil_value;
         params.num_samples = params.stencil.surf.samples;
      }

      if (clear_depth) {
         /* If we're clearing depth, we must have HiZ */
         assert(depth && depth->aux_usage == ISL_AUX_USAGE_HIZ);

         brw_blorp_surface_info_init(batch->blorp, &params.depth, depth,
                                     level, layer,
                                     ISL_FORMAT_UNSUPPORTED, true);
         params.depth.clear_color.f32[0] = depth_value;
         params.depth_format =
            isl_format_get_depth_format(depth->surf->format, false);
         params.num_samples = params.depth.surf.samples;
      }

      batch->blorp->exec(batch, &params);
   }
}

/* Given a depth stencil attachment, this function performs a fast depth clear
 * on a depth portion and a regular clear on the stencil portion. When
 * performing a fast depth clear on the depth portion, the HiZ buffer is simply
 * tagged as cleared so the depth clear value is not actually needed.
 */
void
blorp_gen8_hiz_clear_attachments(struct blorp_batch *batch,
                                 uint32_t num_samples,
                                 uint32_t x0, uint32_t y0,
                                 uint32_t x1, uint32_t y1,
                                 bool clear_depth, bool clear_stencil,
                                 uint8_t stencil_value)
{
   assert(batch->flags & BLORP_BATCH_NO_EMIT_DEPTH_STENCIL);

   struct blorp_params params;
   blorp_params_init(&params);
   params.num_layers = 1;
   params.hiz_op = ISL_AUX_OP_FAST_CLEAR;
   params.x0 = x0;
   params.y0 = y0;
   params.x1 = x1;
   params.y1 = y1;
   params.num_samples = num_samples;
   params.depth.enabled = clear_depth;
   params.stencil.enabled = clear_stencil;
   params.stencil_ref = stencil_value;
   batch->blorp->exec(batch, &params);
}

/** Clear active color/depth/stencili attachments
 *
 * This function performs a clear operation on the currently bound
 * color/depth/stencil attachments.  It is assumed that any information passed
 * in here is valid, consistent, and in-bounds relative to the currently
 * attached depth/stencil.  The binding_table_offset parameter is the 32-bit
 * offset relative to surface state base address where pre-baked binding table
 * that we are to use lives.  If clear_color is false, binding_table_offset
 * must point to a binding table with one entry which is a valid null surface
 * that matches the currently bound depth and stencil.
 */
void
blorp_clear_attachments(struct blorp_batch *batch,
                        uint32_t binding_table_offset,
                        enum isl_format depth_format,
                        uint32_t num_samples,
                        uint32_t start_layer, uint32_t num_layers,
                        uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                        bool clear_color, union isl_color_value color_value,
                        bool clear_depth, float depth_value,
                        uint8_t stencil_mask, uint8_t stencil_value)
{
   struct blorp_params params;
   blorp_params_init(&params);

   assert(batch->flags & BLORP_BATCH_NO_EMIT_DEPTH_STENCIL);

   params.x0 = x0;
   params.y0 = y0;
   params.x1 = x1;
   params.y1 = y1;

   params.use_pre_baked_binding_table = true;
   params.pre_baked_binding_table_offset = binding_table_offset;

   params.num_layers = num_layers;
   params.num_samples = num_samples;

   if (clear_color) {
      params.dst.enabled = true;

      memcpy(&params.wm_inputs.clear_color, color_value.f32, sizeof(float) * 4);

      /* Unfortunately, without knowing whether or not our destination surface
       * is tiled or not, we have to assume it may be linear.  This means no
       * SIMD16_REPDATA for us. :-(
       */
      if (!blorp_params_get_clear_kernel(batch->blorp, &params, false, false))
         return;
   }

   if (clear_depth) {
      params.depth.enabled = true;

      params.z = depth_value;
      params.depth_format = isl_format_get_depth_format(depth_format, false);
   }

   if (stencil_mask) {
      params.stencil.enabled = true;

      params.stencil_mask = stencil_mask;
      params.stencil_ref = stencil_value;
   }

   if (!blorp_params_get_layer_offset_vs(batch->blorp, &params))
      return;

   params.vs_inputs.base_layer = start_layer;

   batch->blorp->exec(batch, &params);
}

void
blorp_ccs_resolve(struct blorp_batch *batch,
                  struct blorp_surf *surf, uint32_t level,
                  uint32_t start_layer, uint32_t num_layers,
                  enum isl_format format,
                  enum isl_aux_op resolve_op)
{
   struct blorp_params params;

   blorp_params_init(&params);
   brw_blorp_surface_info_init(batch->blorp, &params.dst, surf,
                               level, start_layer, format, true);

   /* From the Ivy Bridge PRM, Vol2 Part1 11.9 "Render Target Resolve":
    *
    *     A rectangle primitive must be scaled down by the following factors
    *     with respect to render target being resolved.
    *
    * The scaledown factors in the table that follows are related to the block
    * size of the CCS format.  For IVB and HSW, we divide by two, for BDW we
    * multiply by 8 and 16. On Sky Lake, we multiply by 8.
    */
   const struct isl_format_layout *aux_fmtl =
      isl_format_get_layout(params.dst.aux_surf.format);
   assert(aux_fmtl->txc == ISL_TXC_CCS);

   unsigned x_scaledown, y_scaledown;
   if (ISL_DEV_GEN(batch->blorp->isl_dev) >= 9) {
      x_scaledown = aux_fmtl->bw * 8;
      y_scaledown = aux_fmtl->bh * 8;
   } else if (ISL_DEV_GEN(batch->blorp->isl_dev) >= 8) {
      x_scaledown = aux_fmtl->bw * 8;
      y_scaledown = aux_fmtl->bh * 16;
   } else {
      x_scaledown = aux_fmtl->bw / 2;
      y_scaledown = aux_fmtl->bh / 2;
   }
   params.x0 = params.y0 = 0;
   params.x1 = minify(params.dst.aux_surf.logical_level0_px.width, level);
   params.y1 = minify(params.dst.aux_surf.logical_level0_px.height, level);
   params.x1 = ALIGN(params.x1, x_scaledown) / x_scaledown;
   params.y1 = ALIGN(params.y1, y_scaledown) / y_scaledown;

   if (batch->blorp->isl_dev->info->gen >= 9) {
      assert(resolve_op == ISL_AUX_OP_FULL_RESOLVE ||
             resolve_op == ISL_AUX_OP_PARTIAL_RESOLVE);
   } else {
      /* Broadwell and earlier do not have a partial resolve */
      assert(resolve_op == ISL_AUX_OP_FULL_RESOLVE);
   }
   params.fast_clear_op = resolve_op;
   params.num_layers = num_layers;

   /* Note: there is no need to initialize push constants because it doesn't
    * matter what data gets dispatched to the render target.  However, we must
    * ensure that the fragment shader delivers the data using the "replicated
    * color" message.
    */

   if (!blorp_params_get_clear_kernel(batch->blorp, &params, true, false))
      return;

   batch->blorp->exec(batch, &params);
}

static nir_ssa_def *
blorp_nir_bit(nir_builder *b, nir_ssa_def *src, unsigned bit)
{
   return nir_iand(b, nir_ushr(b, src, nir_imm_int(b, bit)),
                      nir_imm_int(b, 1));
}

struct blorp_mcs_partial_resolve_key
{
   enum blorp_shader_type shader_type;
   bool indirect_clear_color;
   bool int_format;
   uint32_t num_samples;
};

static bool
blorp_params_get_mcs_partial_resolve_kernel(struct blorp_context *blorp,
                                            struct blorp_params *params)
{
   const struct blorp_mcs_partial_resolve_key blorp_key = {
      .shader_type = BLORP_SHADER_TYPE_MCS_PARTIAL_RESOLVE,
      .indirect_clear_color = params->dst.clear_color_addr.buffer != NULL,
      .int_format = isl_format_has_int_channel(params->dst.view.format),
      .num_samples = params->num_samples,
   };

   if (blorp->lookup_shader(blorp, &blorp_key, sizeof(blorp_key),
                            &params->wm_prog_kernel, &params->wm_prog_data))
      return true;

   void *mem_ctx = ralloc_context(NULL);

   nir_builder b;
   nir_builder_init_simple_shader(&b, mem_ctx, MESA_SHADER_FRAGMENT, NULL);
   b.shader->info.name = ralloc_strdup(b.shader, "BLORP-mcs-partial-resolve");

   nir_variable *v_color =
      BLORP_CREATE_NIR_INPUT(b.shader, clear_color, glsl_vec4_type());

   nir_variable *frag_color =
      nir_variable_create(b.shader, nir_var_shader_out,
                          glsl_vec4_type(), "gl_FragColor");
   frag_color->data.location = FRAG_RESULT_COLOR;

   /* Do an MCS fetch and check if it is equal to the magic clear value */
   nir_ssa_def *mcs =
      blorp_nir_txf_ms_mcs(&b, nir_f2i32(&b, blorp_nir_frag_coord(&b)),
                               nir_load_layer_id(&b));
   nir_ssa_def *is_clear =
      blorp_nir_mcs_is_clear_color(&b, mcs, blorp_key.num_samples);

   /* If we aren't the clear value, discard. */
   nir_intrinsic_instr *discard =
      nir_intrinsic_instr_create(b.shader, nir_intrinsic_discard_if);
   discard->src[0] = nir_src_for_ssa(nir_inot(&b, is_clear));
   nir_builder_instr_insert(&b, &discard->instr);

   nir_ssa_def *clear_color = nir_load_var(&b, v_color);
   if (blorp_key.indirect_clear_color && blorp->isl_dev->info->gen <= 8) {
      /* Gen7-8 clear colors are stored as single 0/1 bits */
      clear_color = nir_vec4(&b, blorp_nir_bit(&b, clear_color, 31),
                                 blorp_nir_bit(&b, clear_color, 30),
                                 blorp_nir_bit(&b, clear_color, 29),
                                 blorp_nir_bit(&b, clear_color, 28));

      if (!blorp_key.int_format)
         clear_color = nir_i2f32(&b, clear_color);
   }
   nir_store_var(&b, frag_color, clear_color, 0xf);

   struct brw_wm_prog_key wm_key;
   brw_blorp_init_wm_prog_key(&wm_key);
   wm_key.tex.compressed_multisample_layout_mask = 1;
   wm_key.tex.msaa_16 = blorp_key.num_samples == 16;
   wm_key.multisample_fbo = true;

   struct brw_wm_prog_data prog_data;
   const unsigned *program =
      blorp_compile_fs(blorp, mem_ctx, b.shader, &wm_key, false,
                       &prog_data);

   bool result =
      blorp->upload_shader(blorp, &blorp_key, sizeof(blorp_key),
                           program, prog_data.base.program_size,
                           &prog_data.base, sizeof(prog_data),
                           &params->wm_prog_kernel, &params->wm_prog_data);

   ralloc_free(mem_ctx);
   return result;
}

void
blorp_mcs_partial_resolve(struct blorp_batch *batch,
                          struct blorp_surf *surf,
                          enum isl_format format,
                          uint32_t start_layer, uint32_t num_layers)
{
   struct blorp_params params;
   blorp_params_init(&params);

   assert(batch->blorp->isl_dev->info->gen >= 7);

   params.x0 = 0;
   params.y0 = 0;
   params.x1 = surf->surf->logical_level0_px.width;
   params.y1 = surf->surf->logical_level0_px.height;

   brw_blorp_surface_info_init(batch->blorp, &params.src, surf, 0,
                               start_layer, format, false);
   brw_blorp_surface_info_init(batch->blorp, &params.dst, surf, 0,
                               start_layer, format, true);

   params.num_samples = params.dst.surf.samples;
   params.num_layers = num_layers;
   params.dst_clear_color_as_input = surf->clear_color_addr.buffer != NULL;

   memcpy(&params.wm_inputs.clear_color,
          surf->clear_color.f32, sizeof(float) * 4);

   if (!blorp_params_get_mcs_partial_resolve_kernel(batch->blorp, &params))
      return;

   batch->blorp->exec(batch, &params);
}

/** Clear a CCS to the "uncompressed" state
 *
 * This pass is the CCS equivalent of a "HiZ resolve".  It sets the CCS values
 * for a given layer/level of a surface to 0x0 which is the "uncompressed"
 * state which tells the sampler to go look at the main surface.
 */
void
blorp_ccs_ambiguate(struct blorp_batch *batch,
                    struct blorp_surf *surf,
                    uint32_t level, uint32_t layer)
{
   struct blorp_params params;
   blorp_params_init(&params);

   assert(ISL_DEV_GEN(batch->blorp->isl_dev) >= 7);

   const struct isl_format_layout *aux_fmtl =
      isl_format_get_layout(surf->aux_surf->format);
   assert(aux_fmtl->txc == ISL_TXC_CCS);

   params.dst = (struct brw_blorp_surface_info) {
      .enabled = true,
      .addr = surf->aux_addr,
      .view = {
         .usage = ISL_SURF_USAGE_RENDER_TARGET_BIT,
         .format = ISL_FORMAT_R32G32B32A32_UINT,
         .base_level = 0,
         .base_array_layer = 0,
         .levels = 1,
         .array_len = 1,
         .swizzle = ISL_SWIZZLE_IDENTITY,
      },
   };

   uint32_t z = 0;
   if (surf->surf->dim == ISL_SURF_DIM_3D) {
      z = layer;
      layer = 0;
   }

   uint32_t offset_B, x_offset_el, y_offset_el;
   isl_surf_get_image_offset_el(surf->aux_surf, level, layer, z,
                                &x_offset_el, &y_offset_el);
   isl_tiling_get_intratile_offset_el(surf->aux_surf->tiling, aux_fmtl->bpb,
                                      surf->aux_surf->row_pitch_B,
                                      x_offset_el, y_offset_el,
                                      &offset_B, &x_offset_el, &y_offset_el);
   params.dst.addr.offset += offset_B;

   const uint32_t width_px =
      minify(surf->aux_surf->logical_level0_px.width, level);
   const uint32_t height_px =
      minify(surf->aux_surf->logical_level0_px.height, level);
   const uint32_t width_el = DIV_ROUND_UP(width_px, aux_fmtl->bw);
   const uint32_t height_el = DIV_ROUND_UP(height_px, aux_fmtl->bh);

   struct isl_tile_info ccs_tile_info;
   isl_surf_get_tile_info(surf->aux_surf, &ccs_tile_info);

   /* We're going to map it as a regular RGBA32_UINT surface.  We need to
    * downscale a good deal.  We start by computing the area on the CCS to
    * clear in units of Y-tiled cache lines.
    */
   uint32_t x_offset_cl, y_offset_cl, width_cl, height_cl;
   if (ISL_DEV_GEN(batch->blorp->isl_dev) >= 8) {
      /* From the Sky Lake PRM Vol. 12 in the section on planes:
       *
       *    "The Color Control Surface (CCS) contains the compression status
       *    of the cache-line pairs. The compression state of the cache-line
       *    pair is specified by 2 bits in the CCS.  Each CCS cache-line
       *    represents an area on the main surface of 16x16 sets of 128 byte
       *    Y-tiled cache-line-pairs. CCS is always Y tiled."
       *
       * Each 2-bit surface element in the CCS corresponds to a single
       * cache-line pair in the main surface.  This means that 16x16 el block
       * in the CCS maps to a Y-tiled cache line.  Fortunately, CCS layouts
       * are calculated with a very large alignment so we can round up to a
       * whole cache line without worrying about overdraw.
       */

      /* On Broadwell and above, a CCS tile is the same as a Y tile when
       * viewed at the cache-line granularity.  Fortunately, the horizontal
       * and vertical alignment requirements of the CCS are such that we can
       * align to an entire cache line without worrying about crossing over
       * from one LOD to another.
       */
      const uint32_t x_el_per_cl = ccs_tile_info.logical_extent_el.w / 8;
      const uint32_t y_el_per_cl = ccs_tile_info.logical_extent_el.h / 8;
      assert(surf->aux_surf->image_alignment_el.w % x_el_per_cl == 0);
      assert(surf->aux_surf->image_alignment_el.h % y_el_per_cl == 0);

      assert(x_offset_el % x_el_per_cl == 0);
      assert(y_offset_el % y_el_per_cl == 0);
      x_offset_cl = x_offset_el / x_el_per_cl;
      y_offset_cl = y_offset_el / y_el_per_cl;
      width_cl = DIV_ROUND_UP(width_el, x_el_per_cl);
      height_cl = DIV_ROUND_UP(height_el, y_el_per_cl);
   } else {
      /* On gen7, the CCS tiling is not so nice.  However, there we are
       * guaranteed that we only have a single level and slice so we don't
       * have to worry about it and can just align to a whole tile.
       */
      assert(surf->aux_surf->logical_level0_px.depth == 1);
      assert(surf->aux_surf->logical_level0_px.array_len == 1);
      assert(x_offset_el == 0 && y_offset_el == 0);
      const uint32_t width_tl =
         DIV_ROUND_UP(width_el, ccs_tile_info.logical_extent_el.w);
      const uint32_t height_tl =
         DIV_ROUND_UP(height_el, ccs_tile_info.logical_extent_el.h);
      x_offset_cl = 0;
      y_offset_cl = 0;
      width_cl = width_tl * 8;
      height_cl = height_tl * 8;
   }

   /* We're going to use a RGBA32 format so as to write data as quickly as
    * possible.  A y-tiled cache line will then be 1x4 px.
    */
   const uint32_t x_offset_rgba_px = x_offset_cl;
   const uint32_t y_offset_rgba_px = y_offset_cl * 4;
   const uint32_t width_rgba_px = width_cl;
   const uint32_t height_rgba_px = height_cl * 4;

   MAYBE_UNUSED bool ok =
      isl_surf_init(batch->blorp->isl_dev, &params.dst.surf,
                    .dim = ISL_SURF_DIM_2D,
                    .format = ISL_FORMAT_R32G32B32A32_UINT,
                    .width = width_rgba_px + x_offset_rgba_px,
                    .height = height_rgba_px + y_offset_rgba_px,
                    .depth = 1,
                    .levels = 1,
                    .array_len = 1,
                    .samples = 1,
                    .row_pitch_B = surf->aux_surf->row_pitch_B,
                    .usage = ISL_SURF_USAGE_RENDER_TARGET_BIT,
                    .tiling_flags = ISL_TILING_Y0_BIT);
   assert(ok);

   params.x0 = x_offset_rgba_px;
   params.y0 = y_offset_rgba_px;
   params.x1 = x_offset_rgba_px + width_rgba_px;
   params.y1 = y_offset_rgba_px + height_rgba_px;

   /* A CCS value of 0 means "uncompressed." */
   memset(&params.wm_inputs.clear_color, 0,
          sizeof(params.wm_inputs.clear_color));

   if (!blorp_params_get_clear_kernel(batch->blorp, &params, true, false))
      return;

   batch->blorp->exec(batch, &params);
}
