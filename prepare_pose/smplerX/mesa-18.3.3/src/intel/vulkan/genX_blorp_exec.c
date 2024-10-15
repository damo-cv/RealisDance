/*
 * Copyright © 2016 Intel Corporation
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

#include "anv_private.h"

/* These are defined in anv_private.h and blorp_genX_exec.h */
#undef __gen_address_type
#undef __gen_user_data
#undef __gen_combine_address

#include "common/gen_l3_config.h"
#include "common/gen_sample_positions.h"
#include "blorp/blorp_genX_exec.h"

static void *
blorp_emit_dwords(struct blorp_batch *batch, unsigned n)
{
   struct anv_cmd_buffer *cmd_buffer = batch->driver_batch;
   return anv_batch_emit_dwords(&cmd_buffer->batch, n);
}

static uint64_t
blorp_emit_reloc(struct blorp_batch *batch,
                 void *location, struct blorp_address address, uint32_t delta)
{
   struct anv_cmd_buffer *cmd_buffer = batch->driver_batch;
   assert(cmd_buffer->batch.start <= location &&
          location < cmd_buffer->batch.end);
   return anv_batch_emit_reloc(&cmd_buffer->batch, location,
                               address.buffer, address.offset + delta);
}

static void
blorp_surface_reloc(struct blorp_batch *batch, uint32_t ss_offset,
                    struct blorp_address address, uint32_t delta)
{
   struct anv_cmd_buffer *cmd_buffer = batch->driver_batch;
   VkResult result =
      anv_reloc_list_add(&cmd_buffer->surface_relocs, &cmd_buffer->pool->alloc,
                         ss_offset, address.buffer, address.offset + delta);
   if (result != VK_SUCCESS)
      anv_batch_set_error(&cmd_buffer->batch, result);

   void *dest = cmd_buffer->device->surface_state_pool.block_pool.map +
      ss_offset;
   uint64_t val = ((struct anv_bo*)address.buffer)->offset + address.offset +
      delta;
   write_reloc(cmd_buffer->device, dest, val, false);
}

#if GEN_GEN >= 7 && GEN_GEN < 10
static struct blorp_address
blorp_get_surface_base_address(struct blorp_batch *batch)
{
   struct anv_cmd_buffer *cmd_buffer = batch->driver_batch;
   return (struct blorp_address) {
      .buffer = &cmd_buffer->device->surface_state_pool.block_pool.bo,
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
   struct anv_cmd_buffer *cmd_buffer = batch->driver_batch;

   struct anv_state state =
      anv_cmd_buffer_alloc_dynamic_state(cmd_buffer, size, alignment);

   *offset = state.offset;
   return state.map;
}

static void
blorp_alloc_binding_table(struct blorp_batch *batch, unsigned num_entries,
                          unsigned state_size, unsigned state_alignment,
                          uint32_t *bt_offset,
                          uint32_t *surface_offsets, void **surface_maps)
{
   struct anv_cmd_buffer *cmd_buffer = batch->driver_batch;

   uint32_t state_offset;
   struct anv_state bt_state;

   VkResult result =
      anv_cmd_buffer_alloc_blorp_binding_table(cmd_buffer, num_entries,
                                               &state_offset, &bt_state);
   if (result != VK_SUCCESS)
      return;

   uint32_t *bt_map = bt_state.map;
   *bt_offset = bt_state.offset;

   for (unsigned i = 0; i < num_entries; i++) {
      struct anv_state surface_state =
         anv_cmd_buffer_alloc_surface_state(cmd_buffer);
      bt_map[i] = surface_state.offset + state_offset;
      surface_offsets[i] = surface_state.offset;
      surface_maps[i] = surface_state.map;
   }

   anv_state_flush(cmd_buffer->device, bt_state);
}

static void *
blorp_alloc_vertex_buffer(struct blorp_batch *batch, uint32_t size,
                          struct blorp_address *addr)
{
   struct anv_cmd_buffer *cmd_buffer = batch->driver_batch;

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
   struct anv_state vb_state =
      anv_cmd_buffer_alloc_dynamic_state(cmd_buffer, size, 64);

   *addr = (struct blorp_address) {
      .buffer = &cmd_buffer->device->dynamic_state_pool.block_pool.bo,
      .offset = vb_state.offset,
      .mocs = cmd_buffer->device->default_mocs,
   };

   return vb_state.map;
}

static void
blorp_vf_invalidate_for_vb_48b_transitions(struct blorp_batch *batch,
                                           const struct blorp_address *addrs,
                                           unsigned num_vbs)
{
   /* anv forces all vertex buffers into the low 4GB so there are never any
    * transitions that require a VF invalidation.
    */
}

#if GEN_GEN >= 8
static struct blorp_address
blorp_get_workaround_page(struct blorp_batch *batch)
{
   struct anv_cmd_buffer *cmd_buffer = batch->driver_batch;

   return (struct blorp_address) {
      .buffer = &cmd_buffer->device->workaround_bo,
   };
}
#endif

static void
blorp_flush_range(struct blorp_batch *batch, void *start, size_t size)
{
   struct anv_device *device = batch->blorp->driver_ctx;
   if (!device->info.has_llc)
      gen_flush_range(start, size);
}

static void
blorp_emit_urb_config(struct blorp_batch *batch,
                      unsigned vs_entry_size, unsigned sf_entry_size)
{
   struct anv_device *device = batch->blorp->driver_ctx;
   struct anv_cmd_buffer *cmd_buffer = batch->driver_batch;

   assert(sf_entry_size == 0);

   const unsigned entry_size[4] = { vs_entry_size, 1, 1, 1 };

   genX(emit_urb_setup)(device, &cmd_buffer->batch,
                        cmd_buffer->state.current_l3_config,
                        VK_SHADER_STAGE_VERTEX_BIT |
                        VK_SHADER_STAGE_FRAGMENT_BIT,
                        entry_size);
}

void
genX(blorp_exec)(struct blorp_batch *batch,
                 const struct blorp_params *params)
{
   struct anv_cmd_buffer *cmd_buffer = batch->driver_batch;

   if (!cmd_buffer->state.current_l3_config) {
      const struct gen_l3_config *cfg =
         gen_get_default_l3_config(&cmd_buffer->device->info);
      genX(cmd_buffer_config_l3)(cmd_buffer, cfg);
   }

#if GEN_GEN >= 11
   /* The PIPE_CONTROL command description says:
    *
    *    "Whenever a Binding Table Index (BTI) used by a Render Taget Message
    *     points to a different RENDER_SURFACE_STATE, SW must issue a Render
    *     Target Cache Flush by enabling this bit. When render target flush
    *     is set due to new association of BTI, PS Scoreboard Stall bit must
    *     be set in this packet."
    */
   cmd_buffer->state.pending_pipe_bits |=
      ANV_PIPE_RENDER_TARGET_CACHE_FLUSH_BIT |
      ANV_PIPE_STALL_AT_SCOREBOARD_BIT;
#endif

#if GEN_GEN == 7
   /* The MI_LOAD/STORE_REGISTER_MEM commands which BLORP uses to implement
    * indirect fast-clear colors can cause GPU hangs if we don't stall first.
    * See genX(cmd_buffer_mi_memcpy) for more details.
    */
   if (params->src.clear_color_addr.buffer ||
       params->dst.clear_color_addr.buffer)
      cmd_buffer->state.pending_pipe_bits |= ANV_PIPE_CS_STALL_BIT;
#endif

   genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);

   genX(flush_pipeline_select_3d)(cmd_buffer);

   genX(cmd_buffer_emit_gen7_depth_flush)(cmd_buffer);

   /* BLORP doesn't do anything fancy with depth such as discards, so we want
    * the PMA fix off.  Also, off is always the safe option.
    */
   genX(cmd_buffer_enable_pma_fix)(cmd_buffer, false);

   /* Disable VF statistics */
   blorp_emit(batch, GENX(3DSTATE_VF_STATISTICS), vf) {
      vf.StatisticsEnable = false;
   }

   blorp_exec(batch, params);

   cmd_buffer->state.gfx.vb_dirty = ~0;
   cmd_buffer->state.gfx.dirty = ~0;
   cmd_buffer->state.push_constants_dirty = ~0;
   cmd_buffer->state.pending_pipe_bits |= ANV_PIPE_RENDER_TARGET_WRITES;
}
