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

#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include "anv_private.h"

#include "genxml/gen_macros.h"
#include "genxml/genX_pack.h"

#if GEN_GEN == 8
void
gen8_cmd_buffer_emit_viewport(struct anv_cmd_buffer *cmd_buffer)
{
   uint32_t count = cmd_buffer->state.gfx.dynamic.viewport.count;
   const VkViewport *viewports =
      cmd_buffer->state.gfx.dynamic.viewport.viewports;
   struct anv_state sf_clip_state =
      anv_cmd_buffer_alloc_dynamic_state(cmd_buffer, count * 64, 64);

   for (uint32_t i = 0; i < count; i++) {
      const VkViewport *vp = &viewports[i];

      /* The gen7 state struct has just the matrix and guardband fields, the
       * gen8 struct adds the min/max viewport fields. */
      struct GENX(SF_CLIP_VIEWPORT) sf_clip_viewport = {
         .ViewportMatrixElementm00 = vp->width / 2,
         .ViewportMatrixElementm11 = vp->height / 2,
         .ViewportMatrixElementm22 = vp->maxDepth - vp->minDepth,
         .ViewportMatrixElementm30 = vp->x + vp->width / 2,
         .ViewportMatrixElementm31 = vp->y + vp->height / 2,
         .ViewportMatrixElementm32 = vp->minDepth,
         .XMinClipGuardband = -1.0f,
         .XMaxClipGuardband = 1.0f,
         .YMinClipGuardband = -1.0f,
         .YMaxClipGuardband = 1.0f,
         .XMinViewPort = vp->x,
         .XMaxViewPort = vp->x + vp->width - 1,
         .YMinViewPort = MIN2(vp->y, vp->y + vp->height),
         .YMaxViewPort = MAX2(vp->y, vp->y + vp->height) - 1,
      };

      GENX(SF_CLIP_VIEWPORT_pack)(NULL, sf_clip_state.map + i * 64,
                                 &sf_clip_viewport);
   }

   anv_state_flush(cmd_buffer->device, sf_clip_state);

   anv_batch_emit(&cmd_buffer->batch,
                  GENX(3DSTATE_VIEWPORT_STATE_POINTERS_SF_CLIP), clip) {
      clip.SFClipViewportPointer = sf_clip_state.offset;
   }
}

void
gen8_cmd_buffer_emit_depth_viewport(struct anv_cmd_buffer *cmd_buffer,
                                    bool depth_clamp_enable)
{
   uint32_t count = cmd_buffer->state.gfx.dynamic.viewport.count;
   const VkViewport *viewports =
      cmd_buffer->state.gfx.dynamic.viewport.viewports;
   struct anv_state cc_state =
      anv_cmd_buffer_alloc_dynamic_state(cmd_buffer, count * 8, 32);

   for (uint32_t i = 0; i < count; i++) {
      const VkViewport *vp = &viewports[i];

      struct GENX(CC_VIEWPORT) cc_viewport = {
         .MinimumDepth = depth_clamp_enable ? vp->minDepth : 0.0f,
         .MaximumDepth = depth_clamp_enable ? vp->maxDepth : 1.0f,
      };

      GENX(CC_VIEWPORT_pack)(NULL, cc_state.map + i * 8, &cc_viewport);
   }

   anv_state_flush(cmd_buffer->device, cc_state);

   anv_batch_emit(&cmd_buffer->batch,
                  GENX(3DSTATE_VIEWPORT_STATE_POINTERS_CC), cc) {
      cc.CCViewportPointer = cc_state.offset;
   }
}
#endif

void
genX(cmd_buffer_enable_pma_fix)(struct anv_cmd_buffer *cmd_buffer, bool enable)
{
   if (cmd_buffer->state.pma_fix_enabled == enable)
      return;

   cmd_buffer->state.pma_fix_enabled = enable;

   /* According to the Broadwell PIPE_CONTROL documentation, software should
    * emit a PIPE_CONTROL with the CS Stall and Depth Cache Flush bits set
    * prior to the LRI.  If stencil buffer writes are enabled, then a Render
    * Cache Flush is also necessary.
    *
    * The Skylake docs say to use a depth stall rather than a command
    * streamer stall.  However, the hardware seems to violently disagree.
    * A full command streamer stall seems to be needed in both cases.
    */
   anv_batch_emit(&cmd_buffer->batch, GENX(PIPE_CONTROL), pc) {
      pc.DepthCacheFlushEnable = true;
      pc.CommandStreamerStallEnable = true;
      pc.RenderTargetCacheFlushEnable = true;
   }

#if GEN_GEN == 9

   uint32_t cache_mode;
   anv_pack_struct(&cache_mode, GENX(CACHE_MODE_0),
                   .STCPMAOptimizationEnable = enable,
                   .STCPMAOptimizationEnableMask = true);
   anv_batch_emit(&cmd_buffer->batch, GENX(MI_LOAD_REGISTER_IMM), lri) {
      lri.RegisterOffset   = GENX(CACHE_MODE_0_num);
      lri.DataDWord        = cache_mode;
   }

#elif GEN_GEN == 8

   uint32_t cache_mode;
   anv_pack_struct(&cache_mode, GENX(CACHE_MODE_1),
                   .NPPMAFixEnable = enable,
                   .NPEarlyZFailsDisable = enable,
                   .NPPMAFixEnableMask = true,
                   .NPEarlyZFailsDisableMask = true);
   anv_batch_emit(&cmd_buffer->batch, GENX(MI_LOAD_REGISTER_IMM), lri) {
      lri.RegisterOffset   = GENX(CACHE_MODE_1_num);
      lri.DataDWord        = cache_mode;
   }

#endif /* GEN_GEN == 8 */

   /* After the LRI, a PIPE_CONTROL with both the Depth Stall and Depth Cache
    * Flush bits is often necessary.  We do it regardless because it's easier.
    * The render cache flush is also necessary if stencil writes are enabled.
    *
    * Again, the Skylake docs give a different set of flushes but the BDW
    * flushes seem to work just as well.
    */
   anv_batch_emit(&cmd_buffer->batch, GENX(PIPE_CONTROL), pc) {
      pc.DepthStallEnable = true;
      pc.DepthCacheFlushEnable = true;
      pc.RenderTargetCacheFlushEnable = true;
   }
}

UNUSED static bool
want_depth_pma_fix(struct anv_cmd_buffer *cmd_buffer)
{
   assert(GEN_GEN == 8);

   /* From the Broadwell PRM Vol. 2c CACHE_MODE_1::NP_PMA_FIX_ENABLE:
    *
    *    SW must set this bit in order to enable this fix when following
    *    expression is TRUE.
    *
    *    3DSTATE_WM::ForceThreadDispatch != 1 &&
    *    !(3DSTATE_RASTER::ForceSampleCount != NUMRASTSAMPLES_0) &&
    *    (3DSTATE_DEPTH_BUFFER::SURFACE_TYPE != NULL) &&
    *    (3DSTATE_DEPTH_BUFFER::HIZ Enable) &&
    *    !(3DSTATE_WM::EDSC_Mode == EDSC_PREPS) &&
    *    (3DSTATE_PS_EXTRA::PixelShaderValid) &&
    *    !(3DSTATE_WM_HZ_OP::DepthBufferClear ||
    *      3DSTATE_WM_HZ_OP::DepthBufferResolve ||
    *      3DSTATE_WM_HZ_OP::Hierarchical Depth Buffer Resolve Enable ||
    *      3DSTATE_WM_HZ_OP::StencilBufferClear) &&
    *    (3DSTATE_WM_DEPTH_STENCIL::DepthTestEnable) &&
    *    (((3DSTATE_PS_EXTRA::PixelShaderKillsPixels ||
    *       3DSTATE_PS_EXTRA::oMask Present to RenderTarget ||
    *       3DSTATE_PS_BLEND::AlphaToCoverageEnable ||
    *       3DSTATE_PS_BLEND::AlphaTestEnable ||
    *       3DSTATE_WM_CHROMAKEY::ChromaKeyKillEnable) &&
    *      3DSTATE_WM::ForceKillPix != ForceOff &&
    *      ((3DSTATE_WM_DEPTH_STENCIL::DepthWriteEnable &&
    *        3DSTATE_DEPTH_BUFFER::DEPTH_WRITE_ENABLE) ||
    *       (3DSTATE_WM_DEPTH_STENCIL::Stencil Buffer Write Enable &&
    *        3DSTATE_DEPTH_BUFFER::STENCIL_WRITE_ENABLE &&
    *        3DSTATE_STENCIL_BUFFER::STENCIL_BUFFER_ENABLE))) ||
    *     (3DSTATE_PS_EXTRA:: Pixel Shader Computed Depth mode != PSCDEPTH_OFF))
    */

   /* These are always true:
    *    3DSTATE_WM::ForceThreadDispatch != 1 &&
    *    !(3DSTATE_RASTER::ForceSampleCount != NUMRASTSAMPLES_0)
    */

   /* We only enable the PMA fix if we know for certain that HiZ is enabled.
    * If we don't know whether HiZ is enabled or not, we disable the PMA fix
    * and there is no harm.
    *
    * (3DSTATE_DEPTH_BUFFER::SURFACE_TYPE != NULL) &&
    * 3DSTATE_DEPTH_BUFFER::HIZ Enable
    */
   if (!cmd_buffer->state.hiz_enabled)
      return false;

   /* 3DSTATE_PS_EXTRA::PixelShaderValid */
   struct anv_pipeline *pipeline = cmd_buffer->state.gfx.base.pipeline;
   if (!anv_pipeline_has_stage(pipeline, MESA_SHADER_FRAGMENT))
      return false;

   /* !(3DSTATE_WM::EDSC_Mode == EDSC_PREPS) */
   const struct brw_wm_prog_data *wm_prog_data = get_wm_prog_data(pipeline);
   if (wm_prog_data->early_fragment_tests)
      return false;

   /* We never use anv_pipeline for HiZ ops so this is trivially true:
    *    !(3DSTATE_WM_HZ_OP::DepthBufferClear ||
    *      3DSTATE_WM_HZ_OP::DepthBufferResolve ||
    *      3DSTATE_WM_HZ_OP::Hierarchical Depth Buffer Resolve Enable ||
    *      3DSTATE_WM_HZ_OP::StencilBufferClear)
    */

   /* 3DSTATE_WM_DEPTH_STENCIL::DepthTestEnable */
   if (!pipeline->depth_test_enable)
      return false;

   /* (((3DSTATE_PS_EXTRA::PixelShaderKillsPixels ||
    *    3DSTATE_PS_EXTRA::oMask Present to RenderTarget ||
    *    3DSTATE_PS_BLEND::AlphaToCoverageEnable ||
    *    3DSTATE_PS_BLEND::AlphaTestEnable ||
    *    3DSTATE_WM_CHROMAKEY::ChromaKeyKillEnable) &&
    *   3DSTATE_WM::ForceKillPix != ForceOff &&
    *   ((3DSTATE_WM_DEPTH_STENCIL::DepthWriteEnable &&
    *     3DSTATE_DEPTH_BUFFER::DEPTH_WRITE_ENABLE) ||
    *    (3DSTATE_WM_DEPTH_STENCIL::Stencil Buffer Write Enable &&
    *     3DSTATE_DEPTH_BUFFER::STENCIL_WRITE_ENABLE &&
    *     3DSTATE_STENCIL_BUFFER::STENCIL_BUFFER_ENABLE))) ||
    *  (3DSTATE_PS_EXTRA:: Pixel Shader Computed Depth mode != PSCDEPTH_OFF))
    */
   return (pipeline->kill_pixel && (pipeline->writes_depth ||
                                    pipeline->writes_stencil)) ||
          wm_prog_data->computed_depth_mode != PSCDEPTH_OFF;
}

UNUSED static bool
want_stencil_pma_fix(struct anv_cmd_buffer *cmd_buffer)
{
   if (GEN_GEN > 9)
      return false;
   assert(GEN_GEN == 9);

   /* From the Skylake PRM Vol. 2c CACHE_MODE_1::STC PMA Optimization Enable:
    *
    *    Clearing this bit will force the STC cache to wait for pending
    *    retirement of pixels at the HZ-read stage and do the STC-test for
    *    Non-promoted, R-computed and Computed depth modes instead of
    *    postponing the STC-test to RCPFE.
    *
    *    STC_TEST_EN = 3DSTATE_STENCIL_BUFFER::STENCIL_BUFFER_ENABLE &&
    *                  3DSTATE_WM_DEPTH_STENCIL::StencilTestEnable
    *
    *    STC_WRITE_EN = 3DSTATE_STENCIL_BUFFER::STENCIL_BUFFER_ENABLE &&
    *                   (3DSTATE_WM_DEPTH_STENCIL::Stencil Buffer Write Enable &&
    *                    3DSTATE_DEPTH_BUFFER::STENCIL_WRITE_ENABLE)
    *
    *    COMP_STC_EN = STC_TEST_EN &&
    *                  3DSTATE_PS_EXTRA::PixelShaderComputesStencil
    *
    *    SW parses the pipeline states to generate the following logical
    *    signal indicating if PMA FIX can be enabled.
    *
    *    STC_PMA_OPT =
    *       3DSTATE_WM::ForceThreadDispatch != 1 &&
    *       !(3DSTATE_RASTER::ForceSampleCount != NUMRASTSAMPLES_0) &&
    *       3DSTATE_DEPTH_BUFFER::SURFACE_TYPE != NULL &&
    *       3DSTATE_DEPTH_BUFFER::HIZ Enable &&
    *       !(3DSTATE_WM::EDSC_Mode == 2) &&
    *       3DSTATE_PS_EXTRA::PixelShaderValid &&
    *       !(3DSTATE_WM_HZ_OP::DepthBufferClear ||
    *         3DSTATE_WM_HZ_OP::DepthBufferResolve ||
    *         3DSTATE_WM_HZ_OP::Hierarchical Depth Buffer Resolve Enable ||
    *         3DSTATE_WM_HZ_OP::StencilBufferClear) &&
    *       (COMP_STC_EN || STC_WRITE_EN) &&
    *       ((3DSTATE_PS_EXTRA::PixelShaderKillsPixels ||
    *         3DSTATE_WM::ForceKillPix == ON ||
    *         3DSTATE_PS_EXTRA::oMask Present to RenderTarget ||
    *         3DSTATE_PS_BLEND::AlphaToCoverageEnable ||
    *         3DSTATE_PS_BLEND::AlphaTestEnable ||
    *         3DSTATE_WM_CHROMAKEY::ChromaKeyKillEnable) ||
    *        (3DSTATE_PS_EXTRA::Pixel Shader Computed Depth mode != PSCDEPTH_OFF))
    */

   /* These are always true:
    *    3DSTATE_WM::ForceThreadDispatch != 1 &&
    *    !(3DSTATE_RASTER::ForceSampleCount != NUMRASTSAMPLES_0)
    */

   /* We only enable the PMA fix if we know for certain that HiZ is enabled.
    * If we don't know whether HiZ is enabled or not, we disable the PMA fix
    * and there is no harm.
    *
    * (3DSTATE_DEPTH_BUFFER::SURFACE_TYPE != NULL) &&
    * 3DSTATE_DEPTH_BUFFER::HIZ Enable
    */
   if (!cmd_buffer->state.hiz_enabled)
      return false;

   /* We can't possibly know if HiZ is enabled without the framebuffer */
   assert(cmd_buffer->state.framebuffer);

   /* HiZ is enabled so we had better have a depth buffer with HiZ */
   const struct anv_image_view *ds_iview =
      anv_cmd_buffer_get_depth_stencil_view(cmd_buffer);
   assert(ds_iview && ds_iview->image->planes[0].aux_usage == ISL_AUX_USAGE_HIZ);

   /* 3DSTATE_PS_EXTRA::PixelShaderValid */
   struct anv_pipeline *pipeline = cmd_buffer->state.gfx.base.pipeline;
   if (!anv_pipeline_has_stage(pipeline, MESA_SHADER_FRAGMENT))
      return false;

   /* !(3DSTATE_WM::EDSC_Mode == 2) */
   const struct brw_wm_prog_data *wm_prog_data = get_wm_prog_data(pipeline);
   if (wm_prog_data->early_fragment_tests)
      return false;

   /* We never use anv_pipeline for HiZ ops so this is trivially true:
   *    !(3DSTATE_WM_HZ_OP::DepthBufferClear ||
    *      3DSTATE_WM_HZ_OP::DepthBufferResolve ||
    *      3DSTATE_WM_HZ_OP::Hierarchical Depth Buffer Resolve Enable ||
    *      3DSTATE_WM_HZ_OP::StencilBufferClear)
    */

   /* 3DSTATE_STENCIL_BUFFER::STENCIL_BUFFER_ENABLE &&
    * 3DSTATE_WM_DEPTH_STENCIL::StencilTestEnable
    */
   const bool stc_test_en =
      (ds_iview->image->aspects & VK_IMAGE_ASPECT_STENCIL_BIT) &&
      pipeline->stencil_test_enable;

   /* 3DSTATE_STENCIL_BUFFER::STENCIL_BUFFER_ENABLE &&
    * (3DSTATE_WM_DEPTH_STENCIL::Stencil Buffer Write Enable &&
    *  3DSTATE_DEPTH_BUFFER::STENCIL_WRITE_ENABLE)
    */
   const bool stc_write_en =
      (ds_iview->image->aspects & VK_IMAGE_ASPECT_STENCIL_BIT) &&
      pipeline->writes_stencil;

   /* STC_TEST_EN && 3DSTATE_PS_EXTRA::PixelShaderComputesStencil */
   const bool comp_stc_en = stc_test_en && wm_prog_data->computed_stencil;

   /* COMP_STC_EN || STC_WRITE_EN */
   if (!(comp_stc_en || stc_write_en))
      return false;

   /* (3DSTATE_PS_EXTRA::PixelShaderKillsPixels ||
    *  3DSTATE_WM::ForceKillPix == ON ||
    *  3DSTATE_PS_EXTRA::oMask Present to RenderTarget ||
    *  3DSTATE_PS_BLEND::AlphaToCoverageEnable ||
    *  3DSTATE_PS_BLEND::AlphaTestEnable ||
    *  3DSTATE_WM_CHROMAKEY::ChromaKeyKillEnable) ||
    * (3DSTATE_PS_EXTRA::Pixel Shader Computed Depth mode != PSCDEPTH_OFF)
    */
   return pipeline->kill_pixel ||
          wm_prog_data->computed_depth_mode != PSCDEPTH_OFF;
}

void
genX(cmd_buffer_flush_dynamic_state)(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_pipeline *pipeline = cmd_buffer->state.gfx.base.pipeline;
   struct anv_dynamic_state *d = &cmd_buffer->state.gfx.dynamic;

   if (cmd_buffer->state.gfx.dirty & (ANV_CMD_DIRTY_PIPELINE |
                                      ANV_CMD_DIRTY_DYNAMIC_LINE_WIDTH)) {
      uint32_t sf_dw[GENX(3DSTATE_SF_length)];
      struct GENX(3DSTATE_SF) sf = {
         GENX(3DSTATE_SF_header),
      };
#if GEN_GEN == 8
      if (cmd_buffer->device->info.is_cherryview) {
         sf.CHVLineWidth = d->line_width;
      } else {
         sf.LineWidth = d->line_width;
      }
#else
      sf.LineWidth = d->line_width,
#endif
      GENX(3DSTATE_SF_pack)(NULL, sf_dw, &sf);
      anv_batch_emit_merge(&cmd_buffer->batch, sf_dw, pipeline->gen8.sf);
   }

   if (cmd_buffer->state.gfx.dirty & (ANV_CMD_DIRTY_PIPELINE |
                                      ANV_CMD_DIRTY_DYNAMIC_DEPTH_BIAS)){
      uint32_t raster_dw[GENX(3DSTATE_RASTER_length)];
      struct GENX(3DSTATE_RASTER) raster = {
         GENX(3DSTATE_RASTER_header),
         .GlobalDepthOffsetConstant = d->depth_bias.bias,
         .GlobalDepthOffsetScale = d->depth_bias.slope,
         .GlobalDepthOffsetClamp = d->depth_bias.clamp
      };
      GENX(3DSTATE_RASTER_pack)(NULL, raster_dw, &raster);
      anv_batch_emit_merge(&cmd_buffer->batch, raster_dw,
                           pipeline->gen8.raster);
   }

   /* Stencil reference values moved from COLOR_CALC_STATE in gen8 to
    * 3DSTATE_WM_DEPTH_STENCIL in gen9. That means the dirty bits gets split
    * across different state packets for gen8 and gen9. We handle that by
    * using a big old #if switch here.
    */
#if GEN_GEN == 8
   if (cmd_buffer->state.gfx.dirty & (ANV_CMD_DIRTY_DYNAMIC_BLEND_CONSTANTS |
                                      ANV_CMD_DIRTY_DYNAMIC_STENCIL_REFERENCE)) {
      struct anv_state cc_state =
         anv_cmd_buffer_alloc_dynamic_state(cmd_buffer,
                                            GENX(COLOR_CALC_STATE_length) * 4,
                                            64);
      struct GENX(COLOR_CALC_STATE) cc = {
         .BlendConstantColorRed = d->blend_constants[0],
         .BlendConstantColorGreen = d->blend_constants[1],
         .BlendConstantColorBlue = d->blend_constants[2],
         .BlendConstantColorAlpha = d->blend_constants[3],
         .StencilReferenceValue = d->stencil_reference.front & 0xff,
         .BackfaceStencilReferenceValue = d->stencil_reference.back & 0xff,
      };
      GENX(COLOR_CALC_STATE_pack)(NULL, cc_state.map, &cc);

      anv_state_flush(cmd_buffer->device, cc_state);

      anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_CC_STATE_POINTERS), ccp) {
         ccp.ColorCalcStatePointer        = cc_state.offset;
         ccp.ColorCalcStatePointerValid   = true;
      }
   }

   if (cmd_buffer->state.gfx.dirty & (ANV_CMD_DIRTY_PIPELINE |
                                      ANV_CMD_DIRTY_RENDER_TARGETS |
                                      ANV_CMD_DIRTY_DYNAMIC_STENCIL_COMPARE_MASK |
                                      ANV_CMD_DIRTY_DYNAMIC_STENCIL_WRITE_MASK)) {
      uint32_t wm_depth_stencil_dw[GENX(3DSTATE_WM_DEPTH_STENCIL_length)];

      struct GENX(3DSTATE_WM_DEPTH_STENCIL wm_depth_stencil) = {
         GENX(3DSTATE_WM_DEPTH_STENCIL_header),

         .StencilTestMask = d->stencil_compare_mask.front & 0xff,
         .StencilWriteMask = d->stencil_write_mask.front & 0xff,

         .BackfaceStencilTestMask = d->stencil_compare_mask.back & 0xff,
         .BackfaceStencilWriteMask = d->stencil_write_mask.back & 0xff,

         .StencilBufferWriteEnable =
            (d->stencil_write_mask.front || d->stencil_write_mask.back) &&
            pipeline->writes_stencil,
      };
      GENX(3DSTATE_WM_DEPTH_STENCIL_pack)(NULL, wm_depth_stencil_dw,
                                          &wm_depth_stencil);

      anv_batch_emit_merge(&cmd_buffer->batch, wm_depth_stencil_dw,
                           pipeline->gen8.wm_depth_stencil);

      genX(cmd_buffer_enable_pma_fix)(cmd_buffer,
                                      want_depth_pma_fix(cmd_buffer));
   }
#else
   if (cmd_buffer->state.gfx.dirty & ANV_CMD_DIRTY_DYNAMIC_BLEND_CONSTANTS) {
      struct anv_state cc_state =
         anv_cmd_buffer_alloc_dynamic_state(cmd_buffer,
                                            GENX(COLOR_CALC_STATE_length) * 4,
                                            64);
      struct GENX(COLOR_CALC_STATE) cc = {
         .BlendConstantColorRed = d->blend_constants[0],
         .BlendConstantColorGreen = d->blend_constants[1],
         .BlendConstantColorBlue = d->blend_constants[2],
         .BlendConstantColorAlpha = d->blend_constants[3],
      };
      GENX(COLOR_CALC_STATE_pack)(NULL, cc_state.map, &cc);

      anv_state_flush(cmd_buffer->device, cc_state);

      anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_CC_STATE_POINTERS), ccp) {
         ccp.ColorCalcStatePointer = cc_state.offset;
         ccp.ColorCalcStatePointerValid = true;
      }
   }

   if (cmd_buffer->state.gfx.dirty & (ANV_CMD_DIRTY_PIPELINE |
                                      ANV_CMD_DIRTY_RENDER_TARGETS |
                                      ANV_CMD_DIRTY_DYNAMIC_STENCIL_COMPARE_MASK |
                                      ANV_CMD_DIRTY_DYNAMIC_STENCIL_WRITE_MASK |
                                      ANV_CMD_DIRTY_DYNAMIC_STENCIL_REFERENCE)) {
      uint32_t dwords[GENX(3DSTATE_WM_DEPTH_STENCIL_length)];
      struct GENX(3DSTATE_WM_DEPTH_STENCIL) wm_depth_stencil = {
         GENX(3DSTATE_WM_DEPTH_STENCIL_header),

         .StencilTestMask = d->stencil_compare_mask.front & 0xff,
         .StencilWriteMask = d->stencil_write_mask.front & 0xff,

         .BackfaceStencilTestMask = d->stencil_compare_mask.back & 0xff,
         .BackfaceStencilWriteMask = d->stencil_write_mask.back & 0xff,

         .StencilReferenceValue = d->stencil_reference.front & 0xff,
         .BackfaceStencilReferenceValue = d->stencil_reference.back & 0xff,

         .StencilBufferWriteEnable =
            (d->stencil_write_mask.front || d->stencil_write_mask.back) &&
            pipeline->writes_stencil,
      };
      GENX(3DSTATE_WM_DEPTH_STENCIL_pack)(NULL, dwords, &wm_depth_stencil);

      anv_batch_emit_merge(&cmd_buffer->batch, dwords,
                           pipeline->gen9.wm_depth_stencil);

      genX(cmd_buffer_enable_pma_fix)(cmd_buffer,
                                      want_stencil_pma_fix(cmd_buffer));
   }
#endif

   if (cmd_buffer->state.gfx.dirty & (ANV_CMD_DIRTY_PIPELINE |
                                      ANV_CMD_DIRTY_INDEX_BUFFER)) {
      anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_VF), vf) {
         vf.IndexedDrawCutIndexEnable  = pipeline->primitive_restart;
         vf.CutIndex                   = cmd_buffer->state.restart_index;
      }
   }

   cmd_buffer->state.gfx.dirty = 0;
}

void genX(CmdBindIndexBuffer)(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    _buffer,
    VkDeviceSize                                offset,
    VkIndexType                                 indexType)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_buffer, buffer, _buffer);

   static const uint32_t vk_to_gen_index_type[] = {
      [VK_INDEX_TYPE_UINT16]                    = INDEX_WORD,
      [VK_INDEX_TYPE_UINT32]                    = INDEX_DWORD,
   };

   static const uint32_t restart_index_for_type[] = {
      [VK_INDEX_TYPE_UINT16]                    = UINT16_MAX,
      [VK_INDEX_TYPE_UINT32]                    = UINT32_MAX,
   };

   cmd_buffer->state.restart_index = restart_index_for_type[indexType];

   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_INDEX_BUFFER), ib) {
      ib.IndexFormat                = vk_to_gen_index_type[indexType];
      ib.IndexBufferMOCS            = anv_mocs_for_bo(cmd_buffer->device,
                                                      buffer->address.bo);
      ib.BufferStartingAddress      = anv_address_add(buffer->address, offset);
      ib.BufferSize                 = buffer->size - offset;
   }

   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_INDEX_BUFFER;
}

/* Set of stage bits for which are pipelined, i.e. they get queued by the
 * command streamer for later execution.
 */
#define ANV_PIPELINE_STAGE_PIPELINED_BITS \
   (VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | \
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | \
    VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | \
    VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | \
    VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | \
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | \
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | \
    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | \
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | \
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | \
    VK_PIPELINE_STAGE_TRANSFER_BIT | \
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT | \
    VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | \
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)

void genX(CmdSetEvent)(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     _event,
    VkPipelineStageFlags                        stageMask)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_event, event, _event);

   anv_batch_emit(&cmd_buffer->batch, GENX(PIPE_CONTROL), pc) {
      if (stageMask & ANV_PIPELINE_STAGE_PIPELINED_BITS) {
         pc.StallAtPixelScoreboard = true;
         pc.CommandStreamerStallEnable = true;
      }

      pc.DestinationAddressType  = DAT_PPGTT,
      pc.PostSyncOperation       = WriteImmediateData,
      pc.Address = (struct anv_address) {
         &cmd_buffer->device->dynamic_state_pool.block_pool.bo,
         event->state.offset
      };
      pc.ImmediateData           = VK_EVENT_SET;
   }
}

void genX(CmdResetEvent)(
    VkCommandBuffer                             commandBuffer,
    VkEvent                                     _event,
    VkPipelineStageFlags                        stageMask)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_event, event, _event);

   anv_batch_emit(&cmd_buffer->batch, GENX(PIPE_CONTROL), pc) {
      if (stageMask & ANV_PIPELINE_STAGE_PIPELINED_BITS) {
         pc.StallAtPixelScoreboard = true;
         pc.CommandStreamerStallEnable = true;
      }

      pc.DestinationAddressType  = DAT_PPGTT;
      pc.PostSyncOperation       = WriteImmediateData;
      pc.Address = (struct anv_address) {
         &cmd_buffer->device->dynamic_state_pool.block_pool.bo,
         event->state.offset
      };
      pc.ImmediateData           = VK_EVENT_RESET;
   }
}

void genX(CmdWaitEvents)(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    eventCount,
    const VkEvent*                              pEvents,
    VkPipelineStageFlags                        srcStageMask,
    VkPipelineStageFlags                        destStageMask,
    uint32_t                                    memoryBarrierCount,
    const VkMemoryBarrier*                      pMemoryBarriers,
    uint32_t                                    bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier*                pBufferMemoryBarriers,
    uint32_t                                    imageMemoryBarrierCount,
    const VkImageMemoryBarrier*                 pImageMemoryBarriers)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   for (uint32_t i = 0; i < eventCount; i++) {
      ANV_FROM_HANDLE(anv_event, event, pEvents[i]);

      anv_batch_emit(&cmd_buffer->batch, GENX(MI_SEMAPHORE_WAIT), sem) {
         sem.WaitMode            = PollingMode,
         sem.CompareOperation    = COMPARE_SAD_EQUAL_SDD,
         sem.SemaphoreDataDword  = VK_EVENT_SET,
         sem.SemaphoreAddress = (struct anv_address) {
            &cmd_buffer->device->dynamic_state_pool.block_pool.bo,
            event->state.offset
         };
      }
   }

   genX(CmdPipelineBarrier)(commandBuffer, srcStageMask, destStageMask,
                            false, /* byRegion */
                            memoryBarrierCount, pMemoryBarriers,
                            bufferMemoryBarrierCount, pBufferMemoryBarriers,
                            imageMemoryBarrierCount, pImageMemoryBarriers);
}
