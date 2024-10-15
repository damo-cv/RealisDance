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

#include "fd6_rasterizer.h"
#include "fd6_context.h"
#include "fd6_format.h"

void *
fd6_rasterizer_state_create(struct pipe_context *pctx,
		const struct pipe_rasterizer_state *cso)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd6_rasterizer_stateobj *so;
	float psize_min, psize_max;

	so = CALLOC_STRUCT(fd6_rasterizer_stateobj);
	if (!so)
		return NULL;

	so->base = *cso;

	if (cso->point_size_per_vertex) {
		psize_min = util_get_min_point_size(cso);
		psize_max = 4092;
	} else {
		/* Force the point size to be as if the vertex output was disabled. */
		psize_min = cso->point_size;
		psize_max = cso->point_size;
	}

	so->gras_su_point_minmax =
			A6XX_GRAS_SU_POINT_MINMAX_MIN(psize_min) |
			A6XX_GRAS_SU_POINT_MINMAX_MAX(psize_max);
	so->gras_su_point_size   = A6XX_GRAS_SU_POINT_SIZE(cso->point_size);
	so->gras_su_poly_offset_scale =
			A6XX_GRAS_SU_POLY_OFFSET_SCALE(cso->offset_scale);
	so->gras_su_poly_offset_offset =
			A6XX_GRAS_SU_POLY_OFFSET_OFFSET(cso->offset_units);
	so->gras_su_poly_offset_clamp =
			A6XX_GRAS_SU_POLY_OFFSET_OFFSET_CLAMP(cso->offset_clamp);

	so->gras_su_cntl =
			A6XX_GRAS_SU_CNTL_LINEHALFWIDTH(cso->line_width/2.0);
#if 0
	so->pc_raster_cntl =
		A6XX_PC_RASTER_CNTL_POLYMODE_FRONT_PTYPE(fd_polygon_mode(cso->fill_front)) |
		A6XX_PC_RASTER_CNTL_POLYMODE_BACK_PTYPE(fd_polygon_mode(cso->fill_back));
#endif

#if 0
	if (cso->fill_front != PIPE_POLYGON_MODE_FILL ||
		cso->fill_back != PIPE_POLYGON_MODE_FILL)
		so->pc_raster_cntl |= A6XX_PC_RASTER_CNTL_POLYMODE_ENABLE;
#endif

	if (cso->cull_face & PIPE_FACE_FRONT)
		so->gras_su_cntl |= A6XX_GRAS_SU_CNTL_CULL_FRONT;
	if (cso->cull_face & PIPE_FACE_BACK)
		so->gras_su_cntl |= A6XX_GRAS_SU_CNTL_CULL_BACK;
	if (!cso->front_ccw)
		so->gras_su_cntl |= A6XX_GRAS_SU_CNTL_FRONT_CW;
	if (cso->offset_tri)
		so->gras_su_cntl |= A6XX_GRAS_SU_CNTL_POLY_OFFSET;

	if (!cso->flatshade_first)
		so->pc_primitive_cntl |= A6XX_PC_PRIMITIVE_CNTL_0_PROVOKING_VTX_LAST;

//	if (!cso->depth_clip)
//		so->gras_cl_clip_cntl |= A6XX_GRAS_CL_CLIP_CNTL_ZNEAR_CLIP_DISABLE |
//			A6XX_GRAS_CL_CLIP_CNTL_ZFAR_CLIP_DISABLE;
#if 0
	if (cso->clip_halfz)
		so->gras_cl_clip_cntl |= A6XX_GRAS_CL_CNTL_ZERO_GB_SCALE_Z;
#endif

	so->stateobj = fd_ringbuffer_new_object(ctx->pipe, 15 * 4);
	struct fd_ringbuffer *ring = so->stateobj;

	OUT_PKT4(ring, REG_A6XX_GRAS_UNKNOWN_8000, 1);
	OUT_RING(ring, 0x80);
	OUT_PKT4(ring, REG_A6XX_GRAS_UNKNOWN_8001, 1);
	OUT_RING(ring, 0x0);
	OUT_PKT4(ring, REG_A6XX_GRAS_UNKNOWN_8004, 1);
	OUT_RING(ring, 0x0);

	OUT_PKT4(ring, REG_A6XX_GRAS_SU_CNTL, 1);
	OUT_RING(ring, so->gras_su_cntl);

	OUT_PKT4(ring, REG_A6XX_GRAS_SU_POINT_MINMAX, 2);
	OUT_RING(ring, so->gras_su_point_minmax);
	OUT_RING(ring, so->gras_su_point_size);

	OUT_PKT4(ring, REG_A6XX_GRAS_SU_POLY_OFFSET_SCALE, 3);
	OUT_RING(ring, so->gras_su_poly_offset_scale);
	OUT_RING(ring, so->gras_su_poly_offset_offset);
	OUT_RING(ring, so->gras_su_poly_offset_clamp);

#if 0
	OUT_PKT4(ring, REG_A6XX_PC_RASTER_CNTL, 1);
	OUT_RING(ring, so->pc_raster_cntl);

	OUT_PKT4(ring, REG_A6XX_GRAS_CL_CNTL, 1);
	OUT_RING(ring, so->gras_cl_clip_cntl);
#endif

	return so;
}

void
fd6_rasterizer_state_delete(struct pipe_context *pctx, void *hwcso)
{
	struct fd6_rasterizer_stateobj *so = hwcso;

	fd_ringbuffer_del(so->stateobj);
	FREE(hwcso);
}

