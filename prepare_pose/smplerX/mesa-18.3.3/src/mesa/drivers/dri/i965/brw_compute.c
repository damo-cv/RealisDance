/*
 * Copyright © 2014 Intel Corporation
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

#include <sys/errno.h>

#include "main/condrender.h"
#include "main/mtypes.h"
#include "main/state.h"
#include "brw_context.h"
#include "brw_draw.h"
#include "brw_state.h"
#include "intel_batchbuffer.h"
#include "intel_buffer_objects.h"
#include "brw_defines.h"


static void
prepare_indirect_gpgpu_walker(struct brw_context *brw)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   GLintptr indirect_offset = brw->compute.num_work_groups_offset;
   struct brw_bo *bo = brw->compute.num_work_groups_bo;

   brw_load_register_mem(brw, GEN7_GPGPU_DISPATCHDIMX, bo, indirect_offset + 0);
   brw_load_register_mem(brw, GEN7_GPGPU_DISPATCHDIMY, bo, indirect_offset + 4);
   brw_load_register_mem(brw, GEN7_GPGPU_DISPATCHDIMZ, bo, indirect_offset + 8);

   if (devinfo->gen > 7)
      return;

   /* Clear upper 32-bits of SRC0 and all 64-bits of SRC1 */
   BEGIN_BATCH(7);
   OUT_BATCH(MI_LOAD_REGISTER_IMM | (7 - 2));
   OUT_BATCH(MI_PREDICATE_SRC0 + 4);
   OUT_BATCH(0u);
   OUT_BATCH(MI_PREDICATE_SRC1 + 0);
   OUT_BATCH(0u);
   OUT_BATCH(MI_PREDICATE_SRC1 + 4);
   OUT_BATCH(0u);
   ADVANCE_BATCH();

   /* Load compute_dispatch_indirect_x_size into SRC0 */
   brw_load_register_mem(brw, MI_PREDICATE_SRC0, bo, indirect_offset + 0);

   /* predicate = (compute_dispatch_indirect_x_size == 0); */
   BEGIN_BATCH(1);
   OUT_BATCH(GEN7_MI_PREDICATE |
             MI_PREDICATE_LOADOP_LOAD |
             MI_PREDICATE_COMBINEOP_SET |
             MI_PREDICATE_COMPAREOP_SRCS_EQUAL);
   ADVANCE_BATCH();

   /* Load compute_dispatch_indirect_y_size into SRC0 */
   brw_load_register_mem(brw, MI_PREDICATE_SRC0, bo, indirect_offset + 4);

   /* predicate |= (compute_dispatch_indirect_y_size == 0); */
   BEGIN_BATCH(1);
   OUT_BATCH(GEN7_MI_PREDICATE |
             MI_PREDICATE_LOADOP_LOAD |
             MI_PREDICATE_COMBINEOP_OR |
             MI_PREDICATE_COMPAREOP_SRCS_EQUAL);
   ADVANCE_BATCH();

   /* Load compute_dispatch_indirect_z_size into SRC0 */
   brw_load_register_mem(brw, MI_PREDICATE_SRC0, bo, indirect_offset + 8);

   /* predicate |= (compute_dispatch_indirect_z_size == 0); */
   BEGIN_BATCH(1);
   OUT_BATCH(GEN7_MI_PREDICATE |
             MI_PREDICATE_LOADOP_LOAD |
             MI_PREDICATE_COMBINEOP_OR |
             MI_PREDICATE_COMPAREOP_SRCS_EQUAL);
   ADVANCE_BATCH();

   /* predicate = !predicate; */
   BEGIN_BATCH(1);
   OUT_BATCH(GEN7_MI_PREDICATE |
             MI_PREDICATE_LOADOP_LOADINV |
             MI_PREDICATE_COMBINEOP_OR |
             MI_PREDICATE_COMPAREOP_FALSE);
   ADVANCE_BATCH();
}

static void
brw_emit_gpgpu_walker(struct brw_context *brw)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   const struct brw_cs_prog_data *prog_data =
      brw_cs_prog_data(brw->cs.base.prog_data);

   const GLuint *num_groups = brw->compute.num_work_groups;
   uint32_t indirect_flag;

   if (brw->compute.num_work_groups_bo == NULL) {
      indirect_flag = 0;
   } else {
      indirect_flag =
         GEN7_GPGPU_INDIRECT_PARAMETER_ENABLE |
         (devinfo->gen == 7 ? GEN7_GPGPU_PREDICATE_ENABLE : 0);
      prepare_indirect_gpgpu_walker(brw);
   }

   const unsigned simd_size = prog_data->simd_size;
   unsigned group_size = prog_data->local_size[0] *
      prog_data->local_size[1] * prog_data->local_size[2];
   unsigned thread_width_max =
      (group_size + simd_size - 1) / simd_size;

   uint32_t right_mask = 0xffffffffu >> (32 - simd_size);
   const unsigned right_non_aligned = group_size & (simd_size - 1);
   if (right_non_aligned != 0)
      right_mask >>= (simd_size - right_non_aligned);

   uint32_t dwords = devinfo->gen < 8 ? 11 : 15;
   BEGIN_BATCH(dwords);
   OUT_BATCH(GPGPU_WALKER << 16 | (dwords - 2) | indirect_flag);
   OUT_BATCH(0);
   if (devinfo->gen >= 8) {
      OUT_BATCH(0);                     /* Indirect Data Length */
      OUT_BATCH(0);                     /* Indirect Data Start Address */
   }
   assert(thread_width_max <= brw->screen->devinfo.max_cs_threads);
   OUT_BATCH(SET_FIELD(simd_size / 16, GPGPU_WALKER_SIMD_SIZE) |
             SET_FIELD(thread_width_max - 1, GPGPU_WALKER_THREAD_WIDTH_MAX));
   OUT_BATCH(0);                        /* Thread Group ID Starting X */
   if (devinfo->gen >= 8)
      OUT_BATCH(0);                     /* MBZ */
   OUT_BATCH(num_groups[0]);            /* Thread Group ID X Dimension */
   OUT_BATCH(0);                        /* Thread Group ID Starting Y */
   if (devinfo->gen >= 8)
      OUT_BATCH(0);                     /* MBZ */
   OUT_BATCH(num_groups[1]);            /* Thread Group ID Y Dimension */
   OUT_BATCH(0);                        /* Thread Group ID Starting/Resume Z */
   OUT_BATCH(num_groups[2]);            /* Thread Group ID Z Dimension */
   OUT_BATCH(right_mask);               /* Right Execution Mask */
   OUT_BATCH(0xffffffff);               /* Bottom Execution Mask */
   ADVANCE_BATCH();

   BEGIN_BATCH(2);
   OUT_BATCH(MEDIA_STATE_FLUSH << 16 | (2 - 2));
   OUT_BATCH(0);
   ADVANCE_BATCH();
}


static void
brw_dispatch_compute_common(struct gl_context *ctx)
{
   struct brw_context *brw = brw_context(ctx);
   bool fail_next;

   if (!_mesa_check_conditional_render(ctx))
      return;

   if (ctx->NewState)
      _mesa_update_state(ctx);

   brw_validate_textures(brw);

   brw_predraw_resolve_inputs(brw, false, NULL);

   /* Flush the batch if the batch/state buffers are nearly full.  We can
    * grow them if needed, but this is not free, so we'd like to avoid it.
    */
   intel_batchbuffer_require_space(brw, 600);
   brw_require_statebuffer_space(brw, 2500);
   intel_batchbuffer_save_state(brw);
   fail_next = intel_batchbuffer_saved_state_is_empty(brw);

 retry:
   brw->batch.no_wrap = true;
   brw_upload_compute_state(brw);

   brw_emit_gpgpu_walker(brw);

   brw->batch.no_wrap = false;

   if (!brw_batch_has_aperture_space(brw, 0)) {
      if (!fail_next) {
         intel_batchbuffer_reset_to_saved(brw);
         intel_batchbuffer_flush(brw);
         fail_next = true;
         goto retry;
      } else {
         int ret = intel_batchbuffer_flush(brw);
         WARN_ONCE(ret == -ENOSPC,
                   "i965: Single compute shader dispatch "
                   "exceeded available aperture space\n");
      }
   }

   /* Now that we know we haven't run out of aperture space, we can safely
    * reset the dirty bits.
    */
   brw_compute_state_finished(brw);

   if (brw->always_flush_batch)
      intel_batchbuffer_flush(brw);

   brw_program_cache_check_size(brw);

   /* Note: since compute shaders can't write to framebuffers, there's no need
    * to call brw_postdraw_set_buffers_need_resolve().
    */
}

static void
brw_dispatch_compute(struct gl_context *ctx, const GLuint *num_groups) {
   struct brw_context *brw = brw_context(ctx);

   brw->compute.num_work_groups_bo = NULL;
   brw->compute.num_work_groups = num_groups;
   ctx->NewDriverState |= BRW_NEW_CS_WORK_GROUPS;

   brw_dispatch_compute_common(ctx);
}

static void
brw_dispatch_compute_indirect(struct gl_context *ctx, GLintptr indirect)
{
   struct brw_context *brw = brw_context(ctx);
   static const GLuint indirect_group_counts[3] = { 0, 0, 0 };
   struct gl_buffer_object *indirect_buffer = ctx->DispatchIndirectBuffer;
   struct brw_bo *bo =
      intel_bufferobj_buffer(brw,
                             intel_buffer_object(indirect_buffer),
                             indirect, 3 * sizeof(GLuint), false);

   brw->compute.num_work_groups_bo = bo;
   brw->compute.num_work_groups_offset = indirect;
   brw->compute.num_work_groups = indirect_group_counts;
   ctx->NewDriverState |= BRW_NEW_CS_WORK_GROUPS;

   brw_dispatch_compute_common(ctx);
}

void
brw_init_compute_functions(struct dd_function_table *functions)
{
   functions->DispatchCompute = brw_dispatch_compute;
   functions->DispatchComputeIndirect = brw_dispatch_compute_indirect;
}
