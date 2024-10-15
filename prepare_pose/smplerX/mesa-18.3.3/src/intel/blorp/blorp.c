/*
 * Copyright © 2012 Intel Corporation
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

#include <errno.h>

#include "program/prog_instruction.h"

#include "blorp_priv.h"
#include "compiler/brw_compiler.h"
#include "compiler/brw_nir.h"

void
blorp_init(struct blorp_context *blorp, void *driver_ctx,
           struct isl_device *isl_dev)
{
   blorp->driver_ctx = driver_ctx;
   blorp->isl_dev = isl_dev;
}

void
blorp_finish(struct blorp_context *blorp)
{
   blorp->driver_ctx = NULL;
}

void
blorp_batch_init(struct blorp_context *blorp,
                 struct blorp_batch *batch, void *driver_batch,
                 enum blorp_batch_flags flags)
{
   batch->blorp = blorp;
   batch->driver_batch = driver_batch;
   batch->flags = flags;
}

void
blorp_batch_finish(struct blorp_batch *batch)
{
   batch->blorp = NULL;
}

void
brw_blorp_surface_info_init(struct blorp_context *blorp,
                            struct brw_blorp_surface_info *info,
                            const struct blorp_surf *surf,
                            unsigned int level, unsigned int layer,
                            enum isl_format format, bool is_render_target)
{
   assert(level < surf->surf->levels);
   assert(layer < MAX2(surf->surf->logical_level0_px.depth >> level,
                       surf->surf->logical_level0_px.array_len));

   info->enabled = true;

   if (format == ISL_FORMAT_UNSUPPORTED)
      format = surf->surf->format;

   info->surf = *surf->surf;
   info->addr = surf->addr;

   info->aux_usage = surf->aux_usage;
   if (info->aux_usage != ISL_AUX_USAGE_NONE) {
      info->aux_surf = *surf->aux_surf;
      info->aux_addr = surf->aux_addr;
      assert(level < info->aux_surf.levels);
      assert(layer < MAX2(info->aux_surf.logical_level0_px.depth >> level,
                          info->aux_surf.logical_level0_px.array_len));
   }

   info->clear_color = surf->clear_color;
   info->clear_color_addr = surf->clear_color_addr;

   info->view = (struct isl_view) {
      .usage = is_render_target ? ISL_SURF_USAGE_RENDER_TARGET_BIT :
                                  ISL_SURF_USAGE_TEXTURE_BIT,
      .format = format,
      .base_level = level,
      .levels = 1,
      .swizzle = ISL_SWIZZLE_IDENTITY,
   };

   info->view.array_len = MAX2(info->surf.logical_level0_px.depth,
                               info->surf.logical_level0_px.array_len);

   if (!is_render_target &&
       (info->surf.dim == ISL_SURF_DIM_3D ||
        info->surf.msaa_layout == ISL_MSAA_LAYOUT_ARRAY)) {
      /* 3-D textures don't support base_array layer and neither do 2-D
       * multisampled textures on IVB so we need to pass it through the
       * sampler in those cases.  These are also two cases where we are
       * guaranteed that we won't be doing any funny surface hacks.
       */
      info->view.base_array_layer = 0;
      info->z_offset = layer;
   } else {
      info->view.base_array_layer = layer;

      assert(info->view.array_len >= info->view.base_array_layer);
      info->view.array_len -= info->view.base_array_layer;
      info->z_offset = 0;
   }

   /* Sandy Bridge and earlier have a limit of a maximum of 512 layers for
    * layered rendering.
    */
   if (is_render_target && blorp->isl_dev->info->gen <= 6)
      info->view.array_len = MIN2(info->view.array_len, 512);

   if (surf->tile_x_sa || surf->tile_y_sa) {
      /* This is only allowed on simple 2D surfaces without MSAA */
      assert(info->surf.dim == ISL_SURF_DIM_2D);
      assert(info->surf.samples == 1);
      assert(info->surf.levels == 1);
      assert(info->surf.logical_level0_px.array_len == 1);
      assert(info->aux_usage == ISL_AUX_USAGE_NONE);

      info->tile_x_sa = surf->tile_x_sa;
      info->tile_y_sa = surf->tile_y_sa;

      /* Instead of using the X/Y Offset fields in RENDER_SURFACE_STATE, we
       * place the image at the tile boundary and offset our sampling or
       * rendering.  For this reason, we need to grow the image by the offset
       * to ensure that the hardware doesn't think we've gone past the edge.
       */
      info->surf.logical_level0_px.w += surf->tile_x_sa;
      info->surf.logical_level0_px.h += surf->tile_y_sa;
      info->surf.phys_level0_sa.w += surf->tile_x_sa;
      info->surf.phys_level0_sa.h += surf->tile_y_sa;
   }
}


void
blorp_params_init(struct blorp_params *params)
{
   memset(params, 0, sizeof(*params));
   params->num_samples = 1;
   params->num_draw_buffers = 1;
   params->num_layers = 1;
}

void
brw_blorp_init_wm_prog_key(struct brw_wm_prog_key *wm_key)
{
   memset(wm_key, 0, sizeof(*wm_key));
   wm_key->nr_color_regions = 1;
   for (int i = 0; i < MAX_SAMPLERS; i++)
      wm_key->tex.swizzles[i] = SWIZZLE_XYZW;
}

const unsigned *
blorp_compile_fs(struct blorp_context *blorp, void *mem_ctx,
                 struct nir_shader *nir,
                 struct brw_wm_prog_key *wm_key,
                 bool use_repclear,
                 struct brw_wm_prog_data *wm_prog_data)
{
   const struct brw_compiler *compiler = blorp->compiler;

   nir->options =
      compiler->glsl_compiler_options[MESA_SHADER_FRAGMENT].NirOptions;

   memset(wm_prog_data, 0, sizeof(*wm_prog_data));

   assert(exec_list_is_empty(&nir->uniforms));
   wm_prog_data->base.nr_params = 0;
   wm_prog_data->base.param = NULL;

   /* BLORP always uses the first two binding table entries:
    * - Surface 0 is the render target (which always start from 0)
    * - Surface 1 is the source texture
    */
   wm_prog_data->base.binding_table.texture_start = BLORP_TEXTURE_BT_INDEX;

   nir = brw_preprocess_nir(compiler, nir);
   nir_remove_dead_variables(nir, nir_var_shader_in);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   if (blorp->compiler->devinfo->gen < 6) {
      if (nir->info.fs.uses_discard)
         wm_key->iz_lookup |= BRW_WM_IZ_PS_KILL_ALPHATEST_BIT;

      wm_key->input_slots_valid = nir->info.inputs_read | VARYING_BIT_POS;
   }

   const unsigned *program =
      brw_compile_fs(compiler, blorp->driver_ctx, mem_ctx, wm_key,
                     wm_prog_data, nir, NULL, -1, -1, -1, false, use_repclear,
                     NULL, NULL);

   return program;
}

const unsigned *
blorp_compile_vs(struct blorp_context *blorp, void *mem_ctx,
                 struct nir_shader *nir,
                 struct brw_vs_prog_data *vs_prog_data)
{
   const struct brw_compiler *compiler = blorp->compiler;

   nir->options =
      compiler->glsl_compiler_options[MESA_SHADER_VERTEX].NirOptions;

   nir = brw_preprocess_nir(compiler, nir);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   vs_prog_data->inputs_read = nir->info.inputs_read;

   brw_compute_vue_map(compiler->devinfo,
                       &vs_prog_data->base.vue_map,
                       nir->info.outputs_written,
                       nir->info.separate_shader);

   struct brw_vs_prog_key vs_key = { 0, };

   const unsigned *program =
      brw_compile_vs(compiler, blorp->driver_ctx, mem_ctx,
                     &vs_key, vs_prog_data, nir, -1, NULL);

   return program;
}

struct blorp_sf_key {
   enum blorp_shader_type shader_type; /* Must be BLORP_SHADER_TYPE_GEN4_SF */

   struct brw_sf_prog_key key;
};

bool
blorp_ensure_sf_program(struct blorp_context *blorp,
                        struct blorp_params *params)
{
   const struct brw_wm_prog_data *wm_prog_data = params->wm_prog_data;
   assert(params->wm_prog_data);

   /* Gen6+ doesn't need a strips and fans program */
   if (blorp->compiler->devinfo->gen >= 6)
      return true;

   struct blorp_sf_key key = {
      .shader_type = BLORP_SHADER_TYPE_GEN4_SF,
   };

   /* Everything gets compacted in vertex setup, so we just need a
    * pass-through for the correct number of input varyings.
    */
   const uint64_t slots_valid = VARYING_BIT_POS |
      ((1ull << wm_prog_data->num_varying_inputs) - 1) << VARYING_SLOT_VAR0;

   key.key.attrs = slots_valid;
   key.key.primitive = BRW_SF_PRIM_TRIANGLES;
   key.key.contains_flat_varying = wm_prog_data->contains_flat_varying;

   STATIC_ASSERT(sizeof(key.key.interp_mode) ==
                 sizeof(wm_prog_data->interp_mode));
   memcpy(key.key.interp_mode, wm_prog_data->interp_mode,
          sizeof(key.key.interp_mode));

   if (blorp->lookup_shader(blorp, &key, sizeof(key),
                            &params->sf_prog_kernel, &params->sf_prog_data))
      return true;

   void *mem_ctx = ralloc_context(NULL);

   const unsigned *program;
   unsigned program_size;

   struct brw_vue_map vue_map;
   brw_compute_vue_map(blorp->compiler->devinfo, &vue_map, slots_valid, false);

   struct brw_sf_prog_data prog_data_tmp;
   program = brw_compile_sf(blorp->compiler, mem_ctx, &key.key,
                            &prog_data_tmp, &vue_map, &program_size);

   bool result =
      blorp->upload_shader(blorp, &key, sizeof(key), program, program_size,
                           (void *)&prog_data_tmp, sizeof(prog_data_tmp),
                           &params->sf_prog_kernel, &params->sf_prog_data);

   ralloc_free(mem_ctx);

   return result;
}

void
blorp_hiz_op(struct blorp_batch *batch, struct blorp_surf *surf,
             uint32_t level, uint32_t start_layer, uint32_t num_layers,
             enum isl_aux_op op)
{
   struct blorp_params params;
   blorp_params_init(&params);

   params.hiz_op = op;
   params.full_surface_hiz_op = true;

   for (uint32_t a = 0; a < num_layers; a++) {
      const uint32_t layer = start_layer + a;

      brw_blorp_surface_info_init(batch->blorp, &params.depth, surf, level,
                                  layer, surf->surf->format, true);

      /* Align the rectangle primitive to 8x4 pixels.
       *
       * During fast depth clears, the emitted rectangle primitive  must be
       * aligned to 8x4 pixels.  From the Ivybridge PRM, Vol 2 Part 1 Section
       * 11.5.3.1 Depth Buffer Clear (and the matching section in the
       * Sandybridge PRM):
       *
       *     If Number of Multisamples is NUMSAMPLES_1, the rectangle must be
       *     aligned to an 8x4 pixel block relative to the upper left corner
       *     of the depth buffer [...]
       *
       * For hiz resolves, the rectangle must also be 8x4 aligned. Item
       * WaHizAmbiguate8x4Aligned from the Haswell workarounds page and the
       * Ivybridge simulator require the alignment.
       *
       * To be safe, let's just align the rect for all hiz operations and all
       * hardware generations.
       *
       * However, for some miptree slices of a Z24 texture, emitting an 8x4
       * aligned rectangle that covers the slice may clobber adjacent slices
       * if we strictly adhered to the texture alignments specified in the
       * PRM.  The Ivybridge PRM, Section "Alignment Unit Size", states that
       * SURFACE_STATE.Surface_Horizontal_Alignment should be 4 for Z24
       * surfaces, not 8. But commit 1f112cc increased the alignment from 4 to
       * 8, which prevents the clobbering.
       */
      params.x1 = minify(params.depth.surf.logical_level0_px.width,
                         params.depth.view.base_level);
      params.y1 = minify(params.depth.surf.logical_level0_px.height,
                         params.depth.view.base_level);
      params.x1 = ALIGN(params.x1, 8);
      params.y1 = ALIGN(params.y1, 4);

      if (params.depth.view.base_level == 0) {
         /* TODO: What about MSAA? */
         params.depth.surf.logical_level0_px.width = params.x1;
         params.depth.surf.logical_level0_px.height = params.y1;
      }

      params.dst.surf.samples = params.depth.surf.samples;
      params.dst.surf.logical_level0_px = params.depth.surf.logical_level0_px;
      params.depth_format =
         isl_format_get_depth_format(surf->surf->format, false);
      params.num_samples = params.depth.surf.samples;

      batch->blorp->exec(batch, &params);
   }
}
