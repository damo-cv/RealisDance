/*
 * Copyright © 2011 Intel Corporation
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

#include "intel_batchbuffer.h"
#include "intel_mipmap_tree.h"
#include "intel_fbo.h"

#include "brw_context.h"
#include "brw_state.h"

#include "blorp/blorp_genX_exec.h"

#if GEN_GEN <= 5
#include "gen4_blorp_exec.h"
#endif

#include "brw_blorp.h"

static void *
blorp_emit_dwords(struct blorp_batch *batch, unsigned n)
{
   assert(batch->blorp->driver_ctx == batch->driver_batch);
   struct brw_context *brw = batch->driver_batch;

   intel_batchbuffer_begin(brw, n);
   uint32_t *map = brw->batch.map_next;
   brw->batch.map_next += n;
   intel_batchbuffer_advance(brw);
   return map;
}

static uint64_t
blorp_emit_reloc(struct blorp_batch *batch,
                 void *location, struct blorp_address address, uint32_t delta)
{
   assert(batch->blorp->driver_ctx == batch->driver_batch);
   struct brw_context *brw = batch->driver_batch;
   uint32_t offset;

   if (GEN_GEN < 6 && brw_ptr_in_state_buffer(&brw->batch, location)) {
      offset = (char *)location - (char *)brw->batch.state.map;
      return brw_state_reloc(&brw->batch, offset,
                             address.buffer, address.offset + delta,
                             address.reloc_flags);
   }

   assert(!brw_ptr_in_state_buffer(&brw->batch, location));

   offset = (char *)location - (char *)brw->batch.batch.map;
   return brw_batch_reloc(&brw->batch, offset,
                          address.buffer, address.offset + delta,
                          address.reloc_flags);
}

static void
blorp_surface_reloc(struct blorp_batch *batch, uint32_t ss_offset,
                    struct blorp_address address, uint32_t delta)
{
   assert(batch->blorp->driver_ctx == batch->driver_batch);
   struct brw_context *brw = batch->driver_batch;
   struct brw_bo *bo = address.buffer;

   uint64_t reloc_val =
      brw_state_reloc(&brw->batch, ss_offset, bo, address.offset + delta,
                      address.reloc_flags);

   void *reloc_ptr = (void *)brw->batch.state.map + ss_offset;
#if GEN_GEN >= 8
   *(uint64_t *)reloc_ptr = reloc_val;
#else
   *(uint32_t *)reloc_ptr = reloc_val;
#endif
}

#if GEN_GEN >= 7 && GEN_GEN < 10
static struct blorp_address
blorp_get_surface_base_address(struct blorp_batch *batch)
{
   assert(batch->blorp->driver_ctx == batch->driver_batch);
   struct brw_context *brw = batch->driver_batch;
   return (struct blorp_address) {
      .buffer = brw->batch.state.bo,
      .offset = 0,
   };
}
#endif

static void *
blorp_alloc_dynamic_state(struct blorp_batch *batch,
                          uint32_t size,
                          uint32_t alignment,
                          uint32_t *offset)
{
   assert(batch->blorp->driver_ctx == batch->driver_batch);
   struct brw_context *brw = batch->driver_batch;

   return brw_state_batch(brw, size, alignment, offset);
}

static void
blorp_alloc_binding_table(struct blorp_batch *batch, unsigned num_entries,
                          unsigned state_size, unsigned state_alignment,
                          uint32_t *bt_offset, uint32_t *surface_offsets,
                          void **surface_maps)
{
   assert(batch->blorp->driver_ctx == batch->driver_batch);
   struct brw_context *brw = batch->driver_batch;

   uint32_t *bt_map = brw_state_batch(brw,
                                      num_entries * sizeof(uint32_t), 32,
                                      bt_offset);

   for (unsigned i = 0; i < num_entries; i++) {
      surface_maps[i] = brw_state_batch(brw,
                                        state_size, state_alignment,
                                        &(surface_offsets)[i]);
      bt_map[i] = surface_offsets[i];
   }
}

static void *
blorp_alloc_vertex_buffer(struct blorp_batch *batch, uint32_t size,
                          struct blorp_address *addr)
{
   assert(batch->blorp->driver_ctx == batch->driver_batch);
   struct brw_context *brw = batch->driver_batch;

   /* From the Skylake PRM, 3DSTATE_VERTEX_BUFFERS:
    *
    *    "The VF cache needs to be invalidated before binding and then using
    *    Vertex Buffers that overlap with any previously bound Vertex Buffer
    *    (at a 64B granularity) since the last invalidation.  A VF cache
    *    invalidate is performed by setting the "VF Cache Invalidation Enable"
    *    bit in PIPE_CONTROL."
    *
    * This restriction first appears in the Skylake PRM but the internal docs
    * also list it as being an issue on Broadwell.  In order to avoid this
    * problem, we align all vertex buffer allocations to 64 bytes.
    */
   uint32_t offset;
   void *data = brw_state_batch(brw, size, 64, &offset);

   *addr = (struct blorp_address) {
      .buffer = brw->batch.state.bo,
      .offset = offset,

      /* The VF cache designers apparently cut corners, and made the cache
       * only consider the bottom 32 bits of memory addresses.  If you happen
       * to have two vertex buffers which get placed exactly 4 GiB apart and
       * use them in back-to-back draw calls, you can get collisions.  To work
       * around this problem, we restrict vertex buffers to the low 32 bits of
       * the address space.
       */
      .reloc_flags = RELOC_32BIT,

#if GEN_GEN == 10
      .mocs = CNL_MOCS_WB,
#elif GEN_GEN == 9
      .mocs = SKL_MOCS_WB,
#elif GEN_GEN == 8
      .mocs = BDW_MOCS_WB,
#elif GEN_GEN == 7
      .mocs = GEN7_MOCS_L3,
#endif
   };

   return data;
}

/**
 * See vf_invalidate_for_vb_48b_transitions in genX_state_upload.c.
 */
static void
blorp_vf_invalidate_for_vb_48b_transitions(struct blorp_batch *batch,
                                           const struct blorp_address *addrs,
                                           unsigned num_vbs)
{
#if GEN_GEN >= 8
   struct brw_context *brw = batch->driver_batch;
   bool need_invalidate = false;

   for (unsigned i = 0; i < num_vbs; i++) {
      struct brw_bo *bo = addrs[i].buffer;
      uint16_t high_bits =
         bo && (bo->kflags & EXEC_OBJECT_PINNED) ? bo->gtt_offset >> 32u : 0;

      if (high_bits != brw->vb.last_bo_high_bits[i]) {
         need_invalidate = true;
         brw->vb.last_bo_high_bits[i] = high_bits;
      }
   }

   if (need_invalidate) {
      brw_emit_pipe_control_flush(brw, PIPE_CONTROL_VF_CACHE_INVALIDATE | PIPE_CONTROL_CS_STALL);
   }
#endif
}

#if GEN_GEN >= 8
static struct blorp_address
blorp_get_workaround_page(struct blorp_batch *batch)
{
   assert(batch->blorp->driver_ctx == batch->driver_batch);
   struct brw_context *brw = batch->driver_batch;

   return (struct blorp_address) {
      .buffer = brw->workaround_bo,
   };
}
#endif

static void
blorp_flush_range(UNUSED struct blorp_batch *batch, UNUSED void *start,
                  UNUSED size_t size)
{
   /* All allocated states come from the batch which we will flush before we
    * submit it.  There's nothing for us to do here.
    */
}

static void
blorp_emit_urb_config(struct blorp_batch *batch,
                      unsigned vs_entry_size,
                      MAYBE_UNUSED unsigned sf_entry_size)
{
   assert(batch->blorp->driver_ctx == batch->driver_batch);
   struct brw_context *brw = batch->driver_batch;

#if GEN_GEN >= 7
   if (brw->urb.vsize >= vs_entry_size)
      return;

   gen7_upload_urb(brw, vs_entry_size, false, false);
#elif GEN_GEN == 6
   gen6_upload_urb(brw, vs_entry_size, false, 0);
#else
   /* We calculate it now and emit later. */
   brw_calculate_urb_fence(brw, 0, vs_entry_size, sf_entry_size);
#endif
}

void
genX(blorp_exec)(struct blorp_batch *batch,
                 const struct blorp_params *params)
{
   assert(batch->blorp->driver_ctx == batch->driver_batch);
   struct brw_context *brw = batch->driver_batch;
   struct gl_context *ctx = &brw->ctx;
   bool check_aperture_failed_once = false;

#if GEN_GEN >= 11
   /* The PIPE_CONTROL command description says:
    *
    * "Whenever a Binding Table Index (BTI) used by a Render Taget Message
    *  points to a different RENDER_SURFACE_STATE, SW must issue a Render
    *  Target Cache Flush by enabling this bit. When render target flush
    *  is set due to new association of BTI, PS Scoreboard Stall bit must
    *  be set in this packet."
   */
   brw_emit_pipe_control_flush(brw,
                               PIPE_CONTROL_RENDER_TARGET_FLUSH |
                               PIPE_CONTROL_STALL_AT_SCOREBOARD);
#endif

   /* Flush the sampler and render caches.  We definitely need to flush the
    * sampler cache so that we get updated contents from the render cache for
    * the glBlitFramebuffer() source.  Also, we are sometimes warned in the
    * docs to flush the cache between reinterpretations of the same surface
    * data with different formats, which blorp does for stencil and depth
    * data.
    */
   if (params->src.enabled)
      brw_cache_flush_for_read(brw, params->src.addr.buffer);
   if (params->dst.enabled) {
      brw_cache_flush_for_render(brw, params->dst.addr.buffer,
                                 params->dst.view.format,
                                 params->dst.aux_usage);
   }
   if (params->depth.enabled)
      brw_cache_flush_for_depth(brw, params->depth.addr.buffer);
   if (params->stencil.enabled)
      brw_cache_flush_for_depth(brw, params->stencil.addr.buffer);

   brw_select_pipeline(brw, BRW_RENDER_PIPELINE);

retry:
   intel_batchbuffer_require_space(brw, 1400);
   brw_require_statebuffer_space(brw, 600);
   intel_batchbuffer_save_state(brw);
   check_aperture_failed_once |= intel_batchbuffer_saved_state_is_empty(brw);
   brw->batch.no_wrap = true;

#if GEN_GEN == 6
   /* Emit workaround flushes when we switch from drawing to blorping. */
   brw_emit_post_sync_nonzero_flush(brw);
#endif

   brw_upload_state_base_address(brw);

#if GEN_GEN >= 8
   gen7_l3_state.emit(brw);
#endif

#if GEN_GEN >= 6
   brw_emit_depth_stall_flushes(brw);
#endif

#if GEN_GEN == 8
   gen8_write_pma_stall_bits(brw, 0);
#endif

   blorp_emit(batch, GENX(3DSTATE_DRAWING_RECTANGLE), rect) {
      rect.ClippedDrawingRectangleXMax = MAX2(params->x1, params->x0) - 1;
      rect.ClippedDrawingRectangleYMax = MAX2(params->y1, params->y0) - 1;
   }

   blorp_exec(batch, params);

   brw->batch.no_wrap = false;

   /* Check if the blorp op we just did would make our batch likely to fail to
    * map all the BOs into the GPU at batch exec time later.  If so, flush the
    * batch and try again with nothing else in the batch.
    */
   if (!brw_batch_has_aperture_space(brw, 0)) {
      if (!check_aperture_failed_once) {
         check_aperture_failed_once = true;
         intel_batchbuffer_reset_to_saved(brw);
         intel_batchbuffer_flush(brw);
         goto retry;
      } else {
         int ret = intel_batchbuffer_flush(brw);
         WARN_ONCE(ret == -ENOSPC,
                   "i965: blorp emit exceeded available aperture space\n");
      }
   }

   if (unlikely(brw->always_flush_batch))
      intel_batchbuffer_flush(brw);

   /* We've smashed all state compared to what the normal 3D pipeline
    * rendering tracks for GL.
    */
   brw->ctx.NewDriverState |= BRW_NEW_BLORP;
   brw->no_depth_or_stencil = !params->depth.enabled &&
                              !params->stencil.enabled;
   brw->ib.index_size = -1;

   if (params->dst.enabled) {
      brw_render_cache_add_bo(brw, params->dst.addr.buffer,
                              params->dst.view.format,
                              params->dst.aux_usage);
   }
   if (params->depth.enabled)
      brw_depth_cache_add_bo(brw, params->depth.addr.buffer);
   if (params->stencil.enabled)
      brw_depth_cache_add_bo(brw, params->stencil.addr.buffer);
}
