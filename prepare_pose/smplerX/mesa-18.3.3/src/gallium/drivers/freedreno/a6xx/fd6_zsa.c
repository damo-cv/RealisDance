/*
 * Copyright (C) 2016 Rob Clark <robclark@freedesktop.org>
 * Copyright © 2018 Google, Inc.
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


#include "pipe/p_state.h"
#include "util/u_string.h"
#include "util/u_memory.h"

#include "fd6_zsa.h"
#include "fd6_context.h"
#include "fd6_format.h"

void *
fd6_zsa_state_create(struct pipe_context *pctx,
		const struct pipe_depth_stencil_alpha_state *cso)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd6_zsa_stateobj *so;

	so = CALLOC_STRUCT(fd6_zsa_stateobj);
	if (!so)
		return NULL;

	so->base = *cso;

	switch (cso->depth.func) {
	case PIPE_FUNC_LESS:
	case PIPE_FUNC_LEQUAL:
		so->gras_lrz_cntl = A6XX_GRAS_LRZ_CNTL_ENABLE;
		so->rb_lrz_cntl = A6XX_RB_LRZ_CNTL_ENABLE;
		break;

	case PIPE_FUNC_GREATER:
	case PIPE_FUNC_GEQUAL:
		so->gras_lrz_cntl = A6XX_GRAS_LRZ_CNTL_ENABLE | A6XX_GRAS_LRZ_CNTL_GREATER;
		so->rb_lrz_cntl = A6XX_RB_LRZ_CNTL_ENABLE;
		break;

	default:
		/* LRZ not enabled */
		so->gras_lrz_cntl = 0;
		break;
	}

	if (cso->depth.writemask) {
		if (cso->depth.enabled)
			so->gras_lrz_cntl |= A6XX_GRAS_LRZ_CNTL_UNK4;
		so->lrz_write = true;
	}

	so->rb_depth_cntl |=
		A6XX_RB_DEPTH_CNTL_ZFUNC(cso->depth.func); /* maps 1:1 */

	if (cso->depth.enabled)
		so->rb_depth_cntl |=
			A6XX_RB_DEPTH_CNTL_Z_ENABLE |
			A6XX_RB_DEPTH_CNTL_Z_TEST_ENABLE;

	if (cso->depth.writemask)
		so->rb_depth_cntl |= A6XX_RB_DEPTH_CNTL_Z_WRITE_ENABLE;

	if (cso->stencil[0].enabled) {
		const struct pipe_stencil_state *s = &cso->stencil[0];

		so->rb_stencil_control |=
			A6XX_RB_STENCIL_CONTROL_STENCIL_READ |
			A6XX_RB_STENCIL_CONTROL_STENCIL_ENABLE |
			A6XX_RB_STENCIL_CONTROL_FUNC(s->func) | /* maps 1:1 */
			A6XX_RB_STENCIL_CONTROL_FAIL(fd_stencil_op(s->fail_op)) |
			A6XX_RB_STENCIL_CONTROL_ZPASS(fd_stencil_op(s->zpass_op)) |
			A6XX_RB_STENCIL_CONTROL_ZFAIL(fd_stencil_op(s->zfail_op));

		so->rb_stencilmask = A6XX_RB_STENCILMASK_MASK(s->valuemask);
		so->rb_stencilwrmask = A6XX_RB_STENCILWRMASK_WRMASK(s->writemask);

		if (cso->stencil[1].enabled) {
			const struct pipe_stencil_state *bs = &cso->stencil[1];

			so->rb_stencil_control |=
				A6XX_RB_STENCIL_CONTROL_STENCIL_ENABLE_BF |
				A6XX_RB_STENCIL_CONTROL_FUNC_BF(bs->func) | /* maps 1:1 */
				A6XX_RB_STENCIL_CONTROL_FAIL_BF(fd_stencil_op(bs->fail_op)) |
				A6XX_RB_STENCIL_CONTROL_ZPASS_BF(fd_stencil_op(bs->zpass_op)) |
				A6XX_RB_STENCIL_CONTROL_ZFAIL_BF(fd_stencil_op(bs->zfail_op));

			so->rb_stencilmask |= A6XX_RB_STENCILMASK_BFMASK(bs->valuemask);
			so->rb_stencilwrmask |= A6XX_RB_STENCILWRMASK_BFWRMASK(bs->writemask);
		}
	}

	if (cso->alpha.enabled) {
		uint32_t ref = cso->alpha.ref_value * 255.0;
		so->rb_alpha_control =
			A6XX_RB_ALPHA_CONTROL_ALPHA_TEST |
			A6XX_RB_ALPHA_CONTROL_ALPHA_REF(ref) |
			A6XX_RB_ALPHA_CONTROL_ALPHA_TEST_FUNC(cso->alpha.func);
//		so->rb_depth_control |=
//			A6XX_RB_DEPTH_CONTROL_EARLY_Z_DISABLE;
	}

	so->stateobj = fd_ringbuffer_new_object(ctx->pipe, 9 * 4);
	struct fd_ringbuffer *ring = so->stateobj;

	OUT_PKT4(ring, REG_A6XX_RB_ALPHA_CONTROL, 1);
	OUT_RING(ring, so->rb_alpha_control);

	OUT_PKT4(ring, REG_A6XX_RB_STENCIL_CONTROL, 1);
	OUT_RING(ring, so->rb_stencil_control);

	OUT_PKT4(ring, REG_A6XX_RB_DEPTH_CNTL, 1);
	OUT_RING(ring, so->rb_depth_cntl);

	OUT_PKT4(ring, REG_A6XX_RB_STENCILMASK, 2);
	OUT_RING(ring, so->rb_stencilmask);
	OUT_RING(ring, so->rb_stencilwrmask);

	so->stateobj_no_alpha = fd_ringbuffer_new_object(ctx->pipe, 9 * 4);
	ring = so->stateobj_no_alpha;

	OUT_PKT4(ring, REG_A6XX_RB_ALPHA_CONTROL, 1);
	OUT_RING(ring, so->rb_alpha_control & ~A6XX_RB_ALPHA_CONTROL_ALPHA_TEST);

	OUT_PKT4(ring, REG_A6XX_RB_STENCIL_CONTROL, 1);
	OUT_RING(ring, so->rb_stencil_control);

	OUT_PKT4(ring, REG_A6XX_RB_DEPTH_CNTL, 1);
	OUT_RING(ring, so->rb_depth_cntl);

	OUT_PKT4(ring, REG_A6XX_RB_STENCILMASK, 2);
	OUT_RING(ring, so->rb_stencilmask);
	OUT_RING(ring, so->rb_stencilwrmask);

	return so;
}

void
fd6_depth_stencil_alpha_state_delete(struct pipe_context *pctx, void *hwcso)
{
	struct fd6_zsa_stateobj *so = hwcso;

	fd_ringbuffer_del(so->stateobj);
	fd_ringbuffer_del(so->stateobj_no_alpha);
	FREE(hwcso);
}
