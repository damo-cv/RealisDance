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

#ifndef BRW_PIPE_CONTROL_DOT_H
#define BRW_PIPE_CONTROL_DOT_H

struct brw_context;
struct gen_device_info;
struct brw_bo;

/** @{
 *
 * PIPE_CONTROL operation, a combination MI_FLUSH and register write with
 * additional flushing control.
 */
#define _3DSTATE_PIPE_CONTROL		(CMD_3D | (3 << 27) | (2 << 24))
#define PIPE_CONTROL_LRI_WRITE_IMMEDIATE (1 << 23) /* Gen7+ */
#define PIPE_CONTROL_CS_STALL		(1 << 20)
#define PIPE_CONTROL_GLOBAL_SNAPSHOT_COUNT_RESET	(1 << 19)
#define PIPE_CONTROL_TLB_INVALIDATE	(1 << 18)
#define PIPE_CONTROL_SYNC_GFDT		(1 << 17)
#define PIPE_CONTROL_MEDIA_STATE_CLEAR	(1 << 16)
#define PIPE_CONTROL_NO_WRITE		(0 << 14)
#define PIPE_CONTROL_WRITE_IMMEDIATE	(1 << 14)
#define PIPE_CONTROL_WRITE_DEPTH_COUNT	(2 << 14)
#define PIPE_CONTROL_WRITE_TIMESTAMP	(3 << 14)
#define PIPE_CONTROL_DEPTH_STALL	(1 << 13)
#define PIPE_CONTROL_RENDER_TARGET_FLUSH (1 << 12)
#define PIPE_CONTROL_INSTRUCTION_INVALIDATE (1 << 11)
#define PIPE_CONTROL_TEXTURE_CACHE_INVALIDATE	(1 << 10) /* GM45+ only */
#define PIPE_CONTROL_ISP_DIS		(1 << 9)
#define PIPE_CONTROL_INTERRUPT_ENABLE	(1 << 8)
#define PIPE_CONTROL_FLUSH_ENABLE	(1 << 7) /* Gen7+ only */
/* GT */
#define PIPE_CONTROL_DATA_CACHE_FLUSH   	(1 << 5)
#define PIPE_CONTROL_VF_CACHE_INVALIDATE	(1 << 4)
#define PIPE_CONTROL_CONST_CACHE_INVALIDATE	(1 << 3)
#define PIPE_CONTROL_STATE_CACHE_INVALIDATE	(1 << 2)
#define PIPE_CONTROL_STALL_AT_SCOREBOARD	(1 << 1)
#define PIPE_CONTROL_DEPTH_CACHE_FLUSH		(1 << 0)
#define PIPE_CONTROL_PPGTT_WRITE	(0 << 2)
#define PIPE_CONTROL_GLOBAL_GTT_WRITE	(1 << 2)

#define PIPE_CONTROL_CACHE_FLUSH_BITS \
   (PIPE_CONTROL_DEPTH_CACHE_FLUSH | PIPE_CONTROL_DATA_CACHE_FLUSH | \
    PIPE_CONTROL_RENDER_TARGET_FLUSH)

#define PIPE_CONTROL_CACHE_INVALIDATE_BITS \
   (PIPE_CONTROL_STATE_CACHE_INVALIDATE | PIPE_CONTROL_CONST_CACHE_INVALIDATE | \
    PIPE_CONTROL_VF_CACHE_INVALIDATE | PIPE_CONTROL_TEXTURE_CACHE_INVALIDATE | \
    PIPE_CONTROL_INSTRUCTION_INVALIDATE)

/** @} */

int brw_init_pipe_control(struct brw_context *brw,
			  const struct gen_device_info *info);
void brw_fini_pipe_control(struct brw_context *brw);

void brw_emit_pipe_control_flush(struct brw_context *brw, uint32_t flags);
void brw_emit_pipe_control_write(struct brw_context *brw, uint32_t flags,
                                 struct brw_bo *bo, uint32_t offset,
                                 uint64_t imm);
void brw_emit_end_of_pipe_sync(struct brw_context *brw, uint32_t flags);
void brw_emit_mi_flush(struct brw_context *brw);
void brw_emit_post_sync_nonzero_flush(struct brw_context *brw);
void brw_emit_depth_stall_flushes(struct brw_context *brw);
void gen7_emit_vs_workaround_flush(struct brw_context *brw);
void gen7_emit_cs_stall_flush(struct brw_context *brw);
void gen10_emit_isp_disable(struct brw_context *brw);

#endif
