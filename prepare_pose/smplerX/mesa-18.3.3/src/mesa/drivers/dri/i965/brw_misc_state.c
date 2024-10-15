/*
 Copyright (C) Intel Corp.  2006.  All Rights Reserved.
 Intel funded Tungsten Graphics to
 develop this 3D driver.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice (including the
 next paragraph) shall be included in all copies or substantial
 portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE COPYRIGHT OWNER(S) AND/OR ITS SUPPLIERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 **********************************************************************/
 /*
  * Authors:
  *   Keith Whitwell <keithw@vmware.com>
  */



#include "intel_batchbuffer.h"
#include "intel_fbo.h"
#include "intel_mipmap_tree.h"

#include "brw_context.h"
#include "brw_state.h"
#include "brw_defines.h"
#include "compiler/brw_eu_defines.h"

#include "main/framebuffer.h"
#include "main/fbobject.h"
#include "main/format_utils.h"
#include "main/glformats.h"

/**
 * Upload pointers to the per-stage state.
 *
 * The state pointers in this packet are all relative to the general state
 * base address set by CMD_STATE_BASE_ADDRESS, which is 0.
 */
static void
upload_pipelined_state_pointers(struct brw_context *brw)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (devinfo->gen == 5) {
      /* Need to flush before changing clip max threads for errata. */
      BEGIN_BATCH(1);
      OUT_BATCH(MI_FLUSH);
      ADVANCE_BATCH();
   }

   BEGIN_BATCH(7);
   OUT_BATCH(_3DSTATE_PIPELINED_POINTERS << 16 | (7 - 2));
   OUT_RELOC(brw->batch.state.bo, 0, brw->vs.base.state_offset);
   if (brw->ff_gs.prog_active)
      OUT_RELOC(brw->batch.state.bo, 0, brw->ff_gs.state_offset | 1);
   else
      OUT_BATCH(0);
   OUT_RELOC(brw->batch.state.bo, 0, brw->clip.state_offset | 1);
   OUT_RELOC(brw->batch.state.bo, 0, brw->sf.state_offset);
   OUT_RELOC(brw->batch.state.bo, 0, brw->wm.base.state_offset);
   OUT_RELOC(brw->batch.state.bo, 0, brw->cc.state_offset);
   ADVANCE_BATCH();

   brw->ctx.NewDriverState |= BRW_NEW_PSP;
}

static void
upload_psp_urb_cbs(struct brw_context *brw)
{
   upload_pipelined_state_pointers(brw);
   brw_upload_urb_fence(brw);
   brw_upload_cs_urb_state(brw);
}

const struct brw_tracked_state brw_psp_urb_cbs = {
   .dirty = {
      .mesa = 0,
      .brw = BRW_NEW_BATCH |
             BRW_NEW_BLORP |
             BRW_NEW_FF_GS_PROG_DATA |
             BRW_NEW_GEN4_UNIT_STATE |
             BRW_NEW_STATE_BASE_ADDRESS |
             BRW_NEW_URB_FENCE,
   },
   .emit = upload_psp_urb_cbs,
};

uint32_t
brw_depthbuffer_format(struct brw_context *brw)
{
   struct gl_context *ctx = &brw->ctx;
   struct gl_framebuffer *fb = ctx->DrawBuffer;
   struct intel_renderbuffer *drb = intel_get_renderbuffer(fb, BUFFER_DEPTH);
   struct intel_renderbuffer *srb;

   if (!drb &&
       (srb = intel_get_renderbuffer(fb, BUFFER_STENCIL)) &&
       !srb->mt->stencil_mt &&
       (intel_rb_format(srb) == MESA_FORMAT_Z24_UNORM_S8_UINT ||
	intel_rb_format(srb) == MESA_FORMAT_Z32_FLOAT_S8X24_UINT)) {
      drb = srb;
   }

   if (!drb)
      return BRW_DEPTHFORMAT_D32_FLOAT;

   return brw_depth_format(brw, drb->mt->format);
}

static struct intel_mipmap_tree *
get_stencil_miptree(struct intel_renderbuffer *irb)
{
   if (!irb)
      return NULL;
   if (irb->mt->stencil_mt)
      return irb->mt->stencil_mt;
   return intel_renderbuffer_get_mt(irb);
}

static bool
rebase_depth_stencil(struct brw_context *brw, struct intel_renderbuffer *irb,
                     bool invalidate)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   struct gl_context *ctx = &brw->ctx;
   uint32_t tile_mask_x = 0, tile_mask_y = 0;

   intel_get_tile_masks(irb->mt->surf.tiling, irb->mt->cpp,
                        &tile_mask_x, &tile_mask_y);
   assert(!intel_miptree_level_has_hiz(irb->mt, irb->mt_level));

   uint32_t tile_x = irb->draw_x & tile_mask_x;
   uint32_t tile_y = irb->draw_y & tile_mask_y;

   /* According to the Sandy Bridge PRM, volume 2 part 1, pp326-327
    * (3DSTATE_DEPTH_BUFFER dw5), in the documentation for "Depth
    * Coordinate Offset X/Y":
    *
    *   "The 3 LSBs of both offsets must be zero to ensure correct
    *   alignment"
    */
   bool rebase = tile_x & 7 || tile_y & 7;

   /* We didn't even have intra-tile offsets before g45. */
   rebase |= (!devinfo->has_surface_tile_offset && (tile_x || tile_y));

   if (rebase) {
      perf_debug("HW workaround: blitting depth level %d to a temporary "
                 "to fix alignment (depth tile offset %d,%d)\n",
                 irb->mt_level, tile_x, tile_y);
      intel_renderbuffer_move_to_temp(brw, irb, invalidate);

      /* There is now only single slice miptree. */
      brw->depthstencil.tile_x = 0;
      brw->depthstencil.tile_y = 0;
      brw->depthstencil.depth_offset = 0;
      return true;
   }

   /* While we just tried to get everything aligned, we may have failed to do
    * so in the case of rendering to array or 3D textures, where nonzero faces
    * will still have an offset post-rebase.  At least give an informative
    * warning.
    */
   WARN_ONCE((tile_x & 7) || (tile_y & 7),
             "Depth/stencil buffer needs alignment to 8-pixel boundaries.\n"
             "Truncating offset (%u:%u), bad rendering may occur.\n",
             tile_x, tile_y);
   tile_x &= ~7;
   tile_y &= ~7;

   brw->depthstencil.tile_x = tile_x;
   brw->depthstencil.tile_y = tile_y;
   brw->depthstencil.depth_offset = intel_miptree_get_aligned_offset(
                                       irb->mt,
                                       irb->draw_x & ~tile_mask_x,
                                       irb->draw_y & ~tile_mask_y);

   return false;
}

void
brw_workaround_depthstencil_alignment(struct brw_context *brw,
                                      GLbitfield clear_mask)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   struct gl_context *ctx = &brw->ctx;
   struct gl_framebuffer *fb = ctx->DrawBuffer;
   struct intel_renderbuffer *depth_irb = intel_get_renderbuffer(fb, BUFFER_DEPTH);
   struct intel_renderbuffer *stencil_irb = intel_get_renderbuffer(fb, BUFFER_STENCIL);
   struct intel_mipmap_tree *depth_mt = NULL;
   bool invalidate_depth = clear_mask & BUFFER_BIT_DEPTH;
   bool invalidate_stencil = clear_mask & BUFFER_BIT_STENCIL;

   if (depth_irb)
      depth_mt = depth_irb->mt;

   /* Initialize brw->depthstencil to 'nop' workaround state.
    */
   brw->depthstencil.tile_x = 0;
   brw->depthstencil.tile_y = 0;
   brw->depthstencil.depth_offset = 0;

   /* Gen6+ doesn't require the workarounds, since we always program the
    * surface state at the start of the whole surface.
    */
   if (devinfo->gen >= 6)
      return;

   /* Check if depth buffer is in depth/stencil format.  If so, then it's only
    * safe to invalidate it if we're also clearing stencil.
    */
   if (depth_irb && invalidate_depth &&
      _mesa_get_format_base_format(depth_mt->format) == GL_DEPTH_STENCIL)
      invalidate_depth = invalidate_stencil && stencil_irb;

   if (depth_irb) {
      if (rebase_depth_stencil(brw, depth_irb, invalidate_depth)) {
         /* In the case of stencil_irb being the same packed depth/stencil
          * texture but not the same rb, make it point at our rebased mt, too.
          */
         if (stencil_irb &&
             stencil_irb != depth_irb &&
             stencil_irb->mt == depth_mt) {
            intel_miptree_reference(&stencil_irb->mt, depth_irb->mt);
            intel_renderbuffer_set_draw_offset(stencil_irb);
         }
      }

      if (stencil_irb) {
         assert(stencil_irb->mt == depth_irb->mt);
         assert(stencil_irb->mt_level == depth_irb->mt_level);
         assert(stencil_irb->mt_layer == depth_irb->mt_layer);
      }
   }

   /* If there is no depth attachment, consider if stencil needs rebase. */
   if (!depth_irb && stencil_irb)
       rebase_depth_stencil(brw, stencil_irb, invalidate_stencil);
}

static void
brw_emit_depth_stencil_hiz(struct brw_context *brw,
                           struct intel_renderbuffer *depth_irb,
                           struct intel_mipmap_tree *depth_mt,
                           struct intel_renderbuffer *stencil_irb,
                           struct intel_mipmap_tree *stencil_mt)
{
   uint32_t tile_x = brw->depthstencil.tile_x;
   uint32_t tile_y = brw->depthstencil.tile_y;
   uint32_t depth_surface_type = BRW_SURFACE_NULL;
   uint32_t depthbuffer_format = BRW_DEPTHFORMAT_D32_FLOAT;
   uint32_t depth_offset = 0;
   uint32_t width = 1, height = 1;
   bool tiled_surface = true;

   /* If there's a packed depth/stencil bound to stencil only, we need to
    * emit the packed depth/stencil buffer packet.
    */
   if (!depth_irb && stencil_irb) {
      depth_irb = stencil_irb;
      depth_mt = stencil_mt;
   }

   if (depth_irb && depth_mt) {
      depthbuffer_format = brw_depthbuffer_format(brw);
      depth_surface_type = BRW_SURFACE_2D;
      depth_offset = brw->depthstencil.depth_offset;
      width = depth_irb->Base.Base.Width;
      height = depth_irb->Base.Base.Height;
      tiled_surface = depth_mt->surf.tiling != ISL_TILING_LINEAR;
   }

   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   const unsigned len = (devinfo->is_g4x || devinfo->gen == 5) ? 6 : 5;

   BEGIN_BATCH(len);
   OUT_BATCH(_3DSTATE_DEPTH_BUFFER << 16 | (len - 2));
   OUT_BATCH((depth_mt ? depth_mt->surf.row_pitch_B - 1 : 0) |
             (depthbuffer_format << 18) |
             (BRW_TILEWALK_YMAJOR << 26) |
             (tiled_surface << 27) |
             (depth_surface_type << 29));

   if (depth_mt) {
      OUT_RELOC(depth_mt->bo, RELOC_WRITE, depth_offset);
   } else {
      OUT_BATCH(0);
   }

   OUT_BATCH(((width + tile_x - 1) << 6) |
             ((height + tile_y - 1) << 19));
   OUT_BATCH(0);

   if (devinfo->is_g4x || devinfo->gen >= 5)
      OUT_BATCH(tile_x | (tile_y << 16));
   else
      assert(tile_x == 0 && tile_y == 0);

   if (devinfo->gen >= 6)
      OUT_BATCH(0);

   ADVANCE_BATCH();
}

void
brw_emit_depthbuffer(struct brw_context *brw)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   struct gl_context *ctx = &brw->ctx;
   struct gl_framebuffer *fb = ctx->DrawBuffer;
   /* _NEW_BUFFERS */
   struct intel_renderbuffer *depth_irb = intel_get_renderbuffer(fb, BUFFER_DEPTH);
   struct intel_renderbuffer *stencil_irb = intel_get_renderbuffer(fb, BUFFER_STENCIL);
   struct intel_mipmap_tree *depth_mt = intel_renderbuffer_get_mt(depth_irb);
   struct intel_mipmap_tree *stencil_mt = get_stencil_miptree(stencil_irb);

   if (depth_mt)
      brw_cache_flush_for_depth(brw, depth_mt->bo);
   if (stencil_mt)
      brw_cache_flush_for_depth(brw, stencil_mt->bo);

   if (devinfo->gen < 6) {
      brw_emit_depth_stencil_hiz(brw, depth_irb, depth_mt,
                                 stencil_irb, stencil_mt);
      return;
   }

   /* Skip repeated NULL depth/stencil emits (think 2D rendering). */
   if (!depth_mt && !stencil_mt && brw->no_depth_or_stencil) {
      assert(brw->hw_ctx);
      return;
   }

   brw_emit_depth_stall_flushes(brw);

   const unsigned ds_dwords = brw->isl_dev.ds.size / 4;
   intel_batchbuffer_begin(brw, ds_dwords);
   uint32_t *ds_map = brw->batch.map_next;
   const uint32_t ds_offset = (char *)ds_map - (char *)brw->batch.batch.map;

   struct isl_view view = {
      /* Some nice defaults */
      .base_level = 0,
      .levels = 1,
      .base_array_layer = 0,
      .array_len = 1,
      .swizzle = ISL_SWIZZLE_IDENTITY,
   };

   struct isl_depth_stencil_hiz_emit_info info = {
      .view = &view,
   };

   if (depth_mt) {
      view.usage |= ISL_SURF_USAGE_DEPTH_BIT;
      info.depth_surf = &depth_mt->surf;

      info.depth_address =
         brw_batch_reloc(&brw->batch,
                         ds_offset + brw->isl_dev.ds.depth_offset,
                         depth_mt->bo, depth_mt->offset, RELOC_WRITE);

      info.mocs = brw_get_bo_mocs(devinfo, depth_mt->bo);
      view.base_level = depth_irb->mt_level - depth_irb->mt->first_level;
      view.base_array_layer = depth_irb->mt_layer;
      view.array_len = MAX2(depth_irb->layer_count, 1);
      view.format = depth_mt->surf.format;

      info.hiz_usage = depth_mt->aux_usage;
      if (!intel_renderbuffer_has_hiz(depth_irb)) {
         /* Just because a miptree has ISL_AUX_USAGE_HIZ does not mean that
          * all miplevels of that miptree are guaranteed to support HiZ.  See
          * intel_miptree_level_enable_hiz for details.
          */
         info.hiz_usage = ISL_AUX_USAGE_NONE;
      }

      if (info.hiz_usage == ISL_AUX_USAGE_HIZ) {
         info.hiz_surf = &depth_mt->aux_buf->surf;

         uint32_t hiz_offset = 0;
         if (devinfo->gen == 6) {
            /* HiZ surfaces on Sandy Bridge technically don't support
             * mip-mapping.  However, we can fake it by offsetting to the
             * first slice of LOD0 in the HiZ surface.
             */
            isl_surf_get_image_offset_B_tile_sa(&depth_mt->aux_buf->surf,
                                                view.base_level, 0, 0,
                                                &hiz_offset, NULL, NULL);
         }

         info.hiz_address =
            brw_batch_reloc(&brw->batch,
                            ds_offset + brw->isl_dev.ds.hiz_offset,
                            depth_mt->aux_buf->bo,
                            depth_mt->aux_buf->offset + hiz_offset,
                            RELOC_WRITE);
      }

      info.depth_clear_value = depth_mt->fast_clear_color.f32[0];
   }

   if (stencil_mt) {
      view.usage |= ISL_SURF_USAGE_STENCIL_BIT;
      info.stencil_surf = &stencil_mt->surf;

      if (!depth_mt) {
         info.mocs = brw_get_bo_mocs(devinfo, stencil_mt->bo);
         view.base_level = stencil_irb->mt_level - stencil_irb->mt->first_level;
         view.base_array_layer = stencil_irb->mt_layer;
         view.array_len = MAX2(stencil_irb->layer_count, 1);
         view.format = stencil_mt->surf.format;
      }

      uint32_t stencil_offset = 0;
      if (devinfo->gen == 6) {
         /* Stencil surfaces on Sandy Bridge technically don't support
          * mip-mapping.  However, we can fake it by offsetting to the
          * first slice of LOD0 in the stencil surface.
          */
         isl_surf_get_image_offset_B_tile_sa(&stencil_mt->surf,
                                             view.base_level, 0, 0,
                                             &stencil_offset, NULL, NULL);
      }

      info.stencil_address =
         brw_batch_reloc(&brw->batch,
                         ds_offset + brw->isl_dev.ds.stencil_offset,
                         stencil_mt->bo,
                         stencil_mt->offset + stencil_offset,
                         RELOC_WRITE);
   }

   isl_emit_depth_stencil_hiz_s(&brw->isl_dev, ds_map, &info);

   brw->batch.map_next += ds_dwords;
   intel_batchbuffer_advance(brw);

   brw->no_depth_or_stencil = !depth_mt && !stencil_mt;
}

const struct brw_tracked_state brw_depthbuffer = {
   .dirty = {
      .mesa = _NEW_BUFFERS,
      .brw = BRW_NEW_AUX_STATE |
             BRW_NEW_BATCH |
             BRW_NEW_BLORP,
   },
   .emit = brw_emit_depthbuffer,
};

void
brw_emit_select_pipeline(struct brw_context *brw, enum brw_pipeline pipeline)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   const bool is_965 = devinfo->gen == 4 && !devinfo->is_g4x;
   const uint32_t _3DSTATE_PIPELINE_SELECT =
      is_965 ? CMD_PIPELINE_SELECT_965 : CMD_PIPELINE_SELECT_GM45;

   if (devinfo->gen >= 8 && devinfo->gen < 10) {
      /* From the Broadwell PRM, Volume 2a: Instructions, PIPELINE_SELECT:
       *
       *   Software must clear the COLOR_CALC_STATE Valid field in
       *   3DSTATE_CC_STATE_POINTERS command prior to send a PIPELINE_SELECT
       *   with Pipeline Select set to GPGPU.
       *
       * The internal hardware docs recommend the same workaround for Gen9
       * hardware too.
       */
      if (pipeline == BRW_COMPUTE_PIPELINE) {
         BEGIN_BATCH(2);
         OUT_BATCH(_3DSTATE_CC_STATE_POINTERS << 16 | (2 - 2));
         OUT_BATCH(0);
         ADVANCE_BATCH();

         brw->ctx.NewDriverState |= BRW_NEW_CC_STATE;
      }
   }

   if (devinfo->gen >= 6) {
      /* From "BXML » GT » MI » vol1a GPU Overview » [Instruction]
       * PIPELINE_SELECT [DevBWR+]":
       *
       *   Project: DEVSNB+
       *
       *   Software must ensure all the write caches are flushed through a
       *   stalling PIPE_CONTROL command followed by another PIPE_CONTROL
       *   command to invalidate read only caches prior to programming
       *   MI_PIPELINE_SELECT command to change the Pipeline Select Mode.
       */
      const unsigned dc_flush =
         devinfo->gen >= 7 ? PIPE_CONTROL_DATA_CACHE_FLUSH : 0;

      brw_emit_pipe_control_flush(brw,
                                  PIPE_CONTROL_RENDER_TARGET_FLUSH |
                                  PIPE_CONTROL_DEPTH_CACHE_FLUSH |
                                  dc_flush |
                                  PIPE_CONTROL_CS_STALL);

      brw_emit_pipe_control_flush(brw,
                                  PIPE_CONTROL_TEXTURE_CACHE_INVALIDATE |
                                  PIPE_CONTROL_CONST_CACHE_INVALIDATE |
                                  PIPE_CONTROL_STATE_CACHE_INVALIDATE |
                                  PIPE_CONTROL_INSTRUCTION_INVALIDATE);

   } else {
      /* From "BXML » GT » MI » vol1a GPU Overview » [Instruction]
       * PIPELINE_SELECT [DevBWR+]":
       *
       *   Project: PRE-DEVSNB
       *
       *   Software must ensure the current pipeline is flushed via an
       *   MI_FLUSH or PIPE_CONTROL prior to the execution of PIPELINE_SELECT.
       */
      BEGIN_BATCH(1);
      OUT_BATCH(MI_FLUSH);
      ADVANCE_BATCH();
   }

   /* Select the pipeline */
   BEGIN_BATCH(1);
   OUT_BATCH(_3DSTATE_PIPELINE_SELECT << 16 |
             (devinfo->gen >= 9 ? (3 << 8) : 0) |
             (pipeline == BRW_COMPUTE_PIPELINE ? 2 : 0));
   ADVANCE_BATCH();

   if (devinfo->gen == 7 && !devinfo->is_haswell &&
       pipeline == BRW_RENDER_PIPELINE) {
      /* From "BXML » GT » MI » vol1a GPU Overview » [Instruction]
       * PIPELINE_SELECT [DevBWR+]":
       *
       *   Project: DEVIVB, DEVHSW:GT3:A0
       *
       *   Software must send a pipe_control with a CS stall and a post sync
       *   operation and then a dummy DRAW after every MI_SET_CONTEXT and
       *   after any PIPELINE_SELECT that is enabling 3D mode.
       */
      gen7_emit_cs_stall_flush(brw);

      BEGIN_BATCH(7);
      OUT_BATCH(CMD_3D_PRIM << 16 | (7 - 2));
      OUT_BATCH(_3DPRIM_POINTLIST);
      OUT_BATCH(0);
      OUT_BATCH(0);
      OUT_BATCH(0);
      OUT_BATCH(0);
      OUT_BATCH(0);
      ADVANCE_BATCH();
   }

   if (devinfo->is_geminilake) {
      /* Project: DevGLK
       *
       * "This chicken bit works around a hardware issue with barrier logic
       *  encountered when switching between GPGPU and 3D pipelines.  To
       *  workaround the issue, this mode bit should be set after a pipeline
       *  is selected."
       */
      const unsigned barrier_mode =
         pipeline == BRW_RENDER_PIPELINE ? GLK_SCEC_BARRIER_MODE_3D_HULL
                                         : GLK_SCEC_BARRIER_MODE_GPGPU;
      brw_load_register_imm32(brw, SLICE_COMMON_ECO_CHICKEN1,
                              barrier_mode | GLK_SCEC_BARRIER_MODE_MASK);
   }
}

/**
 * Misc invariant state packets
 */
void
brw_upload_invariant_state(struct brw_context *brw)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   const bool is_965 = devinfo->gen == 4 && !devinfo->is_g4x;

   brw_emit_select_pipeline(brw, BRW_RENDER_PIPELINE);
   brw->last_pipeline = BRW_RENDER_PIPELINE;

   if (devinfo->gen >= 8) {
      BEGIN_BATCH(3);
      OUT_BATCH(CMD_STATE_SIP << 16 | (3 - 2));
      OUT_BATCH(0);
      OUT_BATCH(0);
      ADVANCE_BATCH();
   } else {
      BEGIN_BATCH(2);
      OUT_BATCH(CMD_STATE_SIP << 16 | (2 - 2));
      OUT_BATCH(0);
      ADVANCE_BATCH();
   }

   /* Original Gen4 doesn't have 3DSTATE_AA_LINE_PARAMETERS. */
   if (!is_965) {
      BEGIN_BATCH(3);
      OUT_BATCH(_3DSTATE_AA_LINE_PARAMETERS << 16 | (3 - 2));
      /* use legacy aa line coverage computation */
      OUT_BATCH(0);
      OUT_BATCH(0);
      ADVANCE_BATCH();
   }

   const uint32_t _3DSTATE_VF_STATISTICS =
      is_965 ? GEN4_3DSTATE_VF_STATISTICS : GM45_3DSTATE_VF_STATISTICS;
   BEGIN_BATCH(1);
   OUT_BATCH(_3DSTATE_VF_STATISTICS << 16 | 1);
   ADVANCE_BATCH();
}

/**
 * Define the base addresses which some state is referenced from.
 *
 * This allows us to avoid having to emit relocations for the objects,
 * and is actually required for binding table pointers on gen6.
 *
 * Surface state base address covers binding table pointers and
 * surface state objects, but not the surfaces that the surface state
 * objects point to.
 */
void
brw_upload_state_base_address(struct brw_context *brw)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (brw->batch.state_base_address_emitted)
      return;

   /* FINISHME: According to section 3.6.1 "STATE_BASE_ADDRESS" of
    * vol1a of the G45 PRM, MI_FLUSH with the ISC invalidate should be
    * programmed prior to STATE_BASE_ADDRESS.
    *
    * However, given that the instruction SBA (general state base
    * address) on this chipset is always set to 0 across X and GL,
    * maybe this isn't required for us in particular.
    */

   if (devinfo->gen >= 6) {
      const unsigned dc_flush =
         devinfo->gen >= 7 ? PIPE_CONTROL_DATA_CACHE_FLUSH : 0;

      /* Emit a render target cache flush.
       *
       * This isn't documented anywhere in the PRM.  However, it seems to be
       * necessary prior to changing the surface state base adress.  We've
       * seen issues in Vulkan where we get GPU hangs when using multi-level
       * command buffers which clear depth, reset state base address, and then
       * go render stuff.
       *
       * Normally, in GL, we would trust the kernel to do sufficient stalls
       * and flushes prior to executing our batch.  However, it doesn't seem
       * as if the kernel's flushing is always sufficient and we don't want to
       * rely on it.
       *
       * We make this an end-of-pipe sync instead of a normal flush because we
       * do not know the current status of the GPU.  On Haswell at least,
       * having a fast-clear operation in flight at the same time as a normal
       * rendering operation can cause hangs.  Since the kernel's flushing is
       * insufficient, we need to ensure that any rendering operations from
       * other processes are definitely complete before we try to do our own
       * rendering.  It's a bit of a big hammer but it appears to work.
       */
      brw_emit_end_of_pipe_sync(brw,
                                PIPE_CONTROL_RENDER_TARGET_FLUSH |
                                PIPE_CONTROL_DEPTH_CACHE_FLUSH |
                                dc_flush);
   }

   if (devinfo->gen >= 8) {
      /* STATE_BASE_ADDRESS has issues with 48-bit address spaces.  If the
       * address + size as seen by STATE_BASE_ADDRESS overflows 48 bits,
       * the GPU appears to treat all accesses to the buffer as being out
       * of bounds and returns zero.  To work around this, we pin all SBAs
       * to the bottom 4GB.
       */
      uint32_t mocs_wb = devinfo->gen >= 9 ? SKL_MOCS_WB : BDW_MOCS_WB;
      int pkt_len = devinfo->gen >= 10 ? 22 : (devinfo->gen >= 9 ? 19 : 16);

      BEGIN_BATCH(pkt_len);
      OUT_BATCH(CMD_STATE_BASE_ADDRESS << 16 | (pkt_len - 2));
      /* General state base address: stateless DP read/write requests */
      OUT_BATCH(mocs_wb << 4 | 1);
      OUT_BATCH(0);
      OUT_BATCH(mocs_wb << 16);
      /* Surface state base address: */
      OUT_RELOC64(brw->batch.state.bo, RELOC_32BIT, mocs_wb << 4 | 1);
      /* Dynamic state base address: */
      OUT_RELOC64(brw->batch.state.bo, RELOC_32BIT, mocs_wb << 4 | 1);
      /* Indirect object base address: MEDIA_OBJECT data */
      OUT_BATCH(mocs_wb << 4 | 1);
      OUT_BATCH(0);
      /* Instruction base address: shader kernels (incl. SIP) */
      OUT_RELOC64(brw->cache.bo, RELOC_32BIT, mocs_wb << 4 | 1);
      /* General state buffer size */
      OUT_BATCH(0xfffff001);
      /* Dynamic state buffer size */
      OUT_BATCH(ALIGN(MAX_STATE_SIZE, 4096) | 1);
      /* Indirect object upper bound */
      OUT_BATCH(0xfffff001);
      /* Instruction access upper bound */
      OUT_BATCH(ALIGN(brw->cache.bo->size, 4096) | 1);
      if (devinfo->gen >= 9) {
         OUT_BATCH(1);
         OUT_BATCH(0);
         OUT_BATCH(0);
      }
      if (devinfo->gen >= 10) {
         OUT_BATCH(1);
         OUT_BATCH(0);
         OUT_BATCH(0);
      }
      ADVANCE_BATCH();
   } else if (devinfo->gen >= 6) {
      uint8_t mocs = devinfo->gen == 7 ? GEN7_MOCS_L3 : 0;

       BEGIN_BATCH(10);
       OUT_BATCH(CMD_STATE_BASE_ADDRESS << 16 | (10 - 2));
       OUT_BATCH(mocs << 8 | /* General State Memory Object Control State */
                 mocs << 4 | /* Stateless Data Port Access Memory Object Control State */
                 1); /* General State Base Address Modify Enable */
       /* Surface state base address:
	* BINDING_TABLE_STATE
	* SURFACE_STATE
	*/
       OUT_RELOC(brw->batch.state.bo, 0, 1);
        /* Dynamic state base address:
	 * SAMPLER_STATE
	 * SAMPLER_BORDER_COLOR_STATE
	 * CLIP, SF, WM/CC viewport state
	 * COLOR_CALC_STATE
	 * DEPTH_STENCIL_STATE
	 * BLEND_STATE
	 * Push constants (when INSTPM: CONSTANT_BUFFER Address Offset
	 * Disable is clear, which we rely on)
	 */
       OUT_RELOC(brw->batch.state.bo, 0, 1);

       OUT_BATCH(1); /* Indirect object base address: MEDIA_OBJECT data */

       /* Instruction base address: shader kernels (incl. SIP) */
       OUT_RELOC(brw->cache.bo, 0, 1);

       OUT_BATCH(1); /* General state upper bound */
       /* Dynamic state upper bound.  Although the documentation says that
	* programming it to zero will cause it to be ignored, that is a lie.
	* If this isn't programmed to a real bound, the sampler border color
	* pointer is rejected, causing border color to mysteriously fail.
	*/
       OUT_BATCH(0xfffff001);
       OUT_BATCH(1); /* Indirect object upper bound */
       OUT_BATCH(1); /* Instruction access upper bound */
       ADVANCE_BATCH();
   } else if (devinfo->gen == 5) {
       BEGIN_BATCH(8);
       OUT_BATCH(CMD_STATE_BASE_ADDRESS << 16 | (8 - 2));
       OUT_BATCH(1); /* General state base address */
       OUT_RELOC(brw->batch.state.bo, 0, 1); /* Surface state base address */
       OUT_BATCH(1); /* Indirect object base address */
       OUT_RELOC(brw->cache.bo, 0, 1); /* Instruction base address */
       OUT_BATCH(0xfffff001); /* General state upper bound */
       OUT_BATCH(1); /* Indirect object upper bound */
       OUT_BATCH(1); /* Instruction access upper bound */
       ADVANCE_BATCH();
   } else {
       BEGIN_BATCH(6);
       OUT_BATCH(CMD_STATE_BASE_ADDRESS << 16 | (6 - 2));
       OUT_BATCH(1); /* General state base address */
       OUT_RELOC(brw->batch.state.bo, 0, 1); /* Surface state base address */
       OUT_BATCH(1); /* Indirect object base address */
       OUT_BATCH(1); /* General state upper bound */
       OUT_BATCH(1); /* Indirect object upper bound */
       ADVANCE_BATCH();
   }

   if (devinfo->gen >= 6) {
      brw_emit_pipe_control_flush(brw,
                                  PIPE_CONTROL_INSTRUCTION_INVALIDATE |
                                  PIPE_CONTROL_STATE_CACHE_INVALIDATE |
                                  PIPE_CONTROL_TEXTURE_CACHE_INVALIDATE);
   }

   /* According to section 3.6.1 of VOL1 of the 965 PRM,
    * STATE_BASE_ADDRESS updates require a reissue of:
    *
    * 3DSTATE_PIPELINE_POINTERS
    * 3DSTATE_BINDING_TABLE_POINTERS
    * MEDIA_STATE_POINTERS
    *
    * and this continues through Ironlake.  The Sandy Bridge PRM, vol
    * 1 part 1 says that the folowing packets must be reissued:
    *
    * 3DSTATE_CC_POINTERS
    * 3DSTATE_BINDING_TABLE_POINTERS
    * 3DSTATE_SAMPLER_STATE_POINTERS
    * 3DSTATE_VIEWPORT_STATE_POINTERS
    * MEDIA_STATE_POINTERS
    *
    * Those are always reissued following SBA updates anyway (new
    * batch time), except in the case of the program cache BO
    * changing.  Having a separate state flag makes the sequence more
    * obvious.
    */

   brw->ctx.NewDriverState |= BRW_NEW_STATE_BASE_ADDRESS;
   brw->batch.state_base_address_emitted = true;
}
