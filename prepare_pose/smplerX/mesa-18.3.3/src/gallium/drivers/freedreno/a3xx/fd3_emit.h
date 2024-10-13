/*
 * Copyright (C) 2013 Rob Clark <robclark@freedesktop.org>
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#ifndef FD3_EMIT_H
#define FD3_EMIT_H

#include "pipe/p_context.h"

#include "freedreno_context.h"
#include "fd3_format.h"
#include "fd3_program.h"
#include "ir3_shader.h"

struct fd_ringbuffer;

void fd3_emit_gmem_restore_tex(struct fd_ringbuffer *ring,
		struct pipe_surface **psurf, int bufs);

/* grouped together emit-state for prog/vertex/state emit: */
struct fd3_emit {
	struct pipe_debug_callback *debug;
	const struct fd_vertex_state *vtx;
	const struct fd_program_stateobj *prog;
	const struct pipe_draw_info *info;
	bool binning_pass;
	struct ir3_shader_key key;
	enum fd_dirty_3d_state dirty;

	uint32_t sprite_coord_enable;
	bool sprite_coord_mode;
	bool rasterflat;

	/* cached to avoid repeated lookups of same variants: */
	const struct ir3_shader_variant *vp, *fp;
};

static inline const struct ir3_shader_variant *
fd3_emit_get_vp(struct fd3_emit *emit)
{
	if (!emit->vp) {
		struct ir3_shader *shader = emit->prog->vp;
		emit->vp = ir3_shader_variant(shader, emit->key,
				emit->binning_pass, emit->debug);
	}
	return emit->vp;
}

static inline const struct ir3_shader_variant *
fd3_emit_get_fp(struct fd3_emit *emit)
{
	if (!emit->fp) {
		if (emit->binning_pass) {
			/* use dummy stateobj to simplify binning vs non-binning: */
			static const struct ir3_shader_variant binning_fp = {};
			emit->fp = &binning_fp;
		} else {
			struct ir3_shader *shader = emit->prog->fp;
			emit->fp = ir3_shader_variant(shader, emit->key,
					false, emit->debug);
		}
	}
	return emit->fp;
}

void fd3_emit_vertex_bufs(struct fd_ringbuffer *ring, struct fd3_emit *emit);

void fd3_emit_state(struct fd_context *ctx, struct fd_ringbuffer *ring,
		struct fd3_emit *emit);

void fd3_emit_restore(struct fd_batch *batch, struct fd_ringbuffer *ring);

void fd3_emit_init(struct pipe_context *pctx);

static inline void
fd3_emit_cache_flush(struct fd_batch *batch, struct fd_ringbuffer *ring)
{
	fd_wfi(batch, ring);
	OUT_PKT0(ring, REG_A3XX_UCHE_CACHE_INVALIDATE0_REG, 2);
	OUT_RING(ring, A3XX_UCHE_CACHE_INVALIDATE0_REG_ADDR(0));
	OUT_RING(ring, A3XX_UCHE_CACHE_INVALIDATE1_REG_ADDR(0) |
			A3XX_UCHE_CACHE_INVALIDATE1_REG_OPCODE(INVALIDATE) |
			A3XX_UCHE_CACHE_INVALIDATE1_REG_ENTIRE_CACHE);
}

#endif /* FD3_EMIT_H */
