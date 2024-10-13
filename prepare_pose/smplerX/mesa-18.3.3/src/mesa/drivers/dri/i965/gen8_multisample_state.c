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

#include "intel_batchbuffer.h"

#include "brw_context.h"
#include "brw_defines.h"
#include "brw_multisample_state.h"

/**
 * From Gen10 Workarounds page in h/w specs:
 * WaSampleOffsetIZ:
 * Prior to the 3DSTATE_SAMPLE_PATTERN driver must ensure there are no
 * markers in the pipeline by programming a PIPE_CONTROL with stall.
 */
static void
gen10_emit_wa_cs_stall_flush(struct brw_context *brw)
{
   UNUSED const struct gen_device_info *devinfo = &brw->screen->devinfo;
   assert(devinfo->gen == 10);
   brw_emit_pipe_control_flush(brw,
                               PIPE_CONTROL_CS_STALL |
                               PIPE_CONTROL_STALL_AT_SCOREBOARD);
}

/**
 * From Gen10 Workarounds page in h/w specs:
 * WaSampleOffsetIZ:
 * When 3DSTATE_SAMPLE_PATTERN is programmed, driver must then issue an
 * MI_LOAD_REGISTER_IMM command to an offset between 0x7000 and 0x7FFF(SVL)
 * after the command to ensure the state has been delivered prior to any
 * command causing a marker in the pipeline.
 */
static void
gen10_emit_wa_lri_to_cache_mode_zero(struct brw_context *brw)
{
   UNUSED const struct gen_device_info *devinfo = &brw->screen->devinfo;
   assert(devinfo->gen == 10);

   /* Write to CACHE_MODE_0 (0x7000) */
   brw_load_register_imm32(brw, GEN7_CACHE_MODE_0, 0);

   /* Before changing the value of CACHE_MODE_0 register, GFX pipeline must
    * be idle; i.e., full flush is required.
    */
   brw_emit_pipe_control_flush(brw,
                               PIPE_CONTROL_CACHE_FLUSH_BITS |
                               PIPE_CONTROL_CACHE_INVALIDATE_BITS);
}

/**
 * 3DSTATE_SAMPLE_PATTERN
 */
void
gen8_emit_3dstate_sample_pattern(struct brw_context *brw)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (devinfo->gen == 10)
      gen10_emit_wa_cs_stall_flush(brw);

   BEGIN_BATCH(9);
   OUT_BATCH(_3DSTATE_SAMPLE_PATTERN << 16 | (9 - 2));

   /* 16x MSAA */
   OUT_BATCH(brw_multisample_positions_16x[0]); /* positions  3,  2,  1,  0 */
   OUT_BATCH(brw_multisample_positions_16x[1]); /* positions  7,  6,  5,  4 */
   OUT_BATCH(brw_multisample_positions_16x[2]); /* positions 11, 10,  9,  8 */
   OUT_BATCH(brw_multisample_positions_16x[3]); /* positions 15, 14, 13, 12 */

   /* 8x MSAA */
   OUT_BATCH(brw_multisample_positions_8x[1]); /* sample positions 7654 */
   OUT_BATCH(brw_multisample_positions_8x[0]); /* sample positions 3210 */

   /* 4x MSAA */
   OUT_BATCH(brw_multisample_positions_4x);

   /* 1x and 2x MSAA */
   OUT_BATCH(brw_multisample_positions_1x_2x);
   ADVANCE_BATCH();

   if (devinfo->gen == 10)
      gen10_emit_wa_lri_to_cache_mode_zero(brw);
}
