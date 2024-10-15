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

#include <stdio.h>

#include "pipe/p_state.h"
#include "util/u_string.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "util/u_format.h"

#include "freedreno_draw.h"
#include "freedreno_state.h"
#include "freedreno_resource.h"

#include "fd6_gmem.h"
#include "fd6_context.h"
#include "fd6_draw.h"
#include "fd6_emit.h"
#include "fd6_program.h"
#include "fd6_format.h"
#include "fd6_zsa.h"

/* some bits in common w/ a4xx: */
#include "a4xx/fd4_draw.h"

static void
emit_mrt(struct fd_ringbuffer *ring, struct pipe_framebuffer_state *pfb,
		struct fd_gmem_stateobj *gmem)
{
	unsigned char mrt_comp[A6XX_MAX_RENDER_TARGETS] = {0};
	unsigned srgb_cntl = 0;
	unsigned i;

	for (i = 0; i < pfb->nr_cbufs; i++) {
		enum a6xx_color_fmt format = 0;
		enum a3xx_color_swap swap = WZYX;
		bool sint = false, uint = false;
		struct fd_resource *rsc = NULL;
		struct fd_resource_slice *slice = NULL;
		uint32_t stride = 0;
		uint32_t offset = 0;

		if (!pfb->cbufs[i])
			continue;

		mrt_comp[i] = 0xf;

		struct pipe_surface *psurf = pfb->cbufs[i];
		enum pipe_format pformat = psurf->format;
		rsc = fd_resource(psurf->texture);
		if (!rsc->bo)
			continue;
				
		uint32_t base = gmem ? gmem->cbuf_base[i] : 0;
		slice = fd_resource_slice(rsc, psurf->u.tex.level);
		format = fd6_pipe2color(pformat);
		swap = fd6_pipe2swap(pformat);
		sint = util_format_is_pure_sint(pformat);
		uint = util_format_is_pure_uint(pformat);

		if (util_format_is_srgb(pformat))
			srgb_cntl |= (1 << i);

		offset = fd_resource_offset(rsc, psurf->u.tex.level,
									psurf->u.tex.first_layer);

		stride = slice->pitch * rsc->cpp;

		debug_assert(psurf->u.tex.first_layer == psurf->u.tex.last_layer);
		debug_assert((offset + slice->size0) <= fd_bo_size(rsc->bo));

		OUT_PKT4(ring, REG_A6XX_RB_MRT_BUF_INFO(i), 6);
		OUT_RING(ring, A6XX_RB_MRT_BUF_INFO_COLOR_FORMAT(format) |
				A6XX_RB_MRT_BUF_INFO_COLOR_TILE_MODE(rsc->tile_mode) |
				A6XX_RB_MRT_BUF_INFO_COLOR_SWAP(swap));
		OUT_RING(ring, A6XX_RB_MRT_PITCH(stride));
		OUT_RING(ring, A6XX_RB_MRT_ARRAY_PITCH(slice->size0));
		OUT_RELOCW(ring, rsc->bo, offset, 0, 0);	/* BASE_LO/HI */
		OUT_RING(ring, base);			/* RB_MRT[i].BASE_GMEM */
		OUT_PKT4(ring, REG_A6XX_SP_FS_MRT_REG(i), 1);
		OUT_RING(ring, A6XX_SP_FS_MRT_REG_COLOR_FORMAT(format) |
				COND(sint, A6XX_SP_FS_MRT_REG_COLOR_SINT) |
				COND(uint, A6XX_SP_FS_MRT_REG_COLOR_UINT));

#if 0
		/* when we support UBWC, these would be the system memory
		 * addr/pitch/etc:
		 */
		OUT_PKT4(ring, REG_A6XX_RB_MRT_FLAG_BUFFER(i), 4);
		OUT_RING(ring, 0x00000000);    /* RB_MRT_FLAG_BUFFER[i].ADDR_LO */
		OUT_RING(ring, 0x00000000);    /* RB_MRT_FLAG_BUFFER[i].ADDR_HI */
		OUT_RING(ring, A6XX_RB_MRT_FLAG_BUFFER_PITCH(0));
		OUT_RING(ring, A6XX_RB_MRT_FLAG_BUFFER_ARRAY_PITCH(0));
#endif
	}

	OUT_PKT4(ring, REG_A6XX_RB_SRGB_CNTL, 1);
	OUT_RING(ring, srgb_cntl);

	OUT_PKT4(ring, REG_A6XX_SP_SRGB_CNTL, 1);
	OUT_RING(ring, srgb_cntl);

	OUT_PKT4(ring, REG_A6XX_RB_RENDER_COMPONENTS, 1);
	OUT_RING(ring, A6XX_RB_RENDER_COMPONENTS_RT0(mrt_comp[0]) |
			A6XX_RB_RENDER_COMPONENTS_RT1(mrt_comp[1]) |
			A6XX_RB_RENDER_COMPONENTS_RT2(mrt_comp[2]) |
			A6XX_RB_RENDER_COMPONENTS_RT3(mrt_comp[3]) |
			A6XX_RB_RENDER_COMPONENTS_RT4(mrt_comp[4]) |
			A6XX_RB_RENDER_COMPONENTS_RT5(mrt_comp[5]) |
			A6XX_RB_RENDER_COMPONENTS_RT6(mrt_comp[6]) |
			A6XX_RB_RENDER_COMPONENTS_RT7(mrt_comp[7]));

	OUT_PKT4(ring, REG_A6XX_SP_FS_RENDER_COMPONENTS, 1);
	OUT_RING(ring,
			A6XX_SP_FS_RENDER_COMPONENTS_RT0(mrt_comp[0]) |
			A6XX_SP_FS_RENDER_COMPONENTS_RT1(mrt_comp[1]) |
			A6XX_SP_FS_RENDER_COMPONENTS_RT2(mrt_comp[2]) |
			A6XX_SP_FS_RENDER_COMPONENTS_RT3(mrt_comp[3]) |
			A6XX_SP_FS_RENDER_COMPONENTS_RT4(mrt_comp[4]) |
			A6XX_SP_FS_RENDER_COMPONENTS_RT5(mrt_comp[5]) |
			A6XX_SP_FS_RENDER_COMPONENTS_RT6(mrt_comp[6]) |
			A6XX_SP_FS_RENDER_COMPONENTS_RT7(mrt_comp[7]));
}

static void
emit_zs(struct fd_ringbuffer *ring, struct pipe_surface *zsbuf,
		struct fd_gmem_stateobj *gmem)
{
	if (zsbuf) {
		struct fd_resource *rsc = fd_resource(zsbuf->texture);
		enum a6xx_depth_format fmt = fd6_pipe2depth(zsbuf->format);
		struct fd_resource_slice *slice = fd_resource_slice(rsc, 0);
		uint32_t stride = slice->pitch * rsc->cpp;
		uint32_t size = slice->size0;
		uint32_t base = gmem ? gmem->zsbuf_base[0] : 0;

		OUT_PKT4(ring, REG_A6XX_RB_DEPTH_BUFFER_INFO, 6);
		OUT_RING(ring, A6XX_RB_DEPTH_BUFFER_INFO_DEPTH_FORMAT(fmt));
		OUT_RING(ring, A6XX_RB_DEPTH_BUFFER_PITCH(stride));
		OUT_RING(ring, A6XX_RB_DEPTH_BUFFER_ARRAY_PITCH(size));
		OUT_RELOCW(ring, rsc->bo, 0, 0, 0);  /* RB_DEPTH_BUFFER_BASE_LO/HI */
		OUT_RING(ring, base); /* RB_DEPTH_BUFFER_BASE_GMEM */

		OUT_PKT4(ring, REG_A6XX_GRAS_SU_DEPTH_BUFFER_INFO, 1);
		OUT_RING(ring, A6XX_GRAS_SU_DEPTH_BUFFER_INFO_DEPTH_FORMAT(fmt));

		OUT_PKT4(ring, REG_A6XX_RB_DEPTH_FLAG_BUFFER_BASE_LO, 3);
		OUT_RING(ring, 0x00000000);    /* RB_DEPTH_FLAG_BUFFER_BASE_LO */
		OUT_RING(ring, 0x00000000);    /* RB_DEPTH_FLAG_BUFFER_BASE_HI */
		OUT_RING(ring, 0x00000000);    /* RB_DEPTH_FLAG_BUFFER_PITCH */

		if (rsc->lrz) {
			OUT_PKT4(ring, REG_A6XX_GRAS_LRZ_BUFFER_BASE_LO, 5);
			OUT_RELOCW(ring, rsc->lrz, 0, 0, 0);
			OUT_RING(ring, A6XX_GRAS_LRZ_BUFFER_PITCH_PITCH(rsc->lrz_pitch));
			//OUT_RELOCW(ring, rsc->lrz, 0, 0, 0); /* GRAS_LRZ_FAST_CLEAR_BUFFER_BASE_LO/HI */
			// XXX a6xx seems to use a different buffer here.. not sure what for..
			OUT_RING(ring, 0x00000000);
			OUT_RING(ring, 0x00000000);
		} else {
			OUT_PKT4(ring, REG_A6XX_GRAS_LRZ_BUFFER_BASE_LO, 5);
			OUT_RING(ring, 0x00000000);
			OUT_RING(ring, 0x00000000);
			OUT_RING(ring, 0x00000000);     /* GRAS_LRZ_BUFFER_PITCH */
			OUT_RING(ring, 0x00000000);     /* GRAS_LRZ_FAST_CLEAR_BUFFER_BASE_LO */
			OUT_RING(ring, 0x00000000);
		}

		if (rsc->stencil) {
			struct fd_resource_slice *slice = fd_resource_slice(rsc->stencil, 0);
			stride = slice->pitch * rsc->cpp;
			size = slice->size0;
			uint32_t base = gmem ? gmem->zsbuf_base[1] : 0;

			OUT_PKT4(ring, REG_A6XX_RB_STENCIL_INFO, 6);
			OUT_RING(ring, A6XX_RB_STENCIL_INFO_SEPARATE_STENCIL);
			OUT_RING(ring, A6XX_RB_STENCIL_BUFFER_PITCH(stride));
			OUT_RING(ring, A6XX_RB_STENCIL_BUFFER_ARRAY_PITCH(size));
			OUT_RELOCW(ring, rsc->stencil->bo, 0, 0, 0);  /* RB_STENCIL_BASE_LO/HI */
			OUT_RING(ring, base);  /* RB_STENCIL_BASE_LO */
		} else {
			OUT_PKT4(ring, REG_A6XX_RB_STENCIL_INFO, 1);
			OUT_RING(ring, 0x00000000);     /* RB_STENCIL_INFO */
		}
	} else {
		OUT_PKT4(ring, REG_A6XX_RB_DEPTH_BUFFER_INFO, 6);
		OUT_RING(ring, A6XX_RB_DEPTH_BUFFER_INFO_DEPTH_FORMAT(DEPTH6_NONE));
		OUT_RING(ring, 0x00000000);    /* RB_DEPTH_BUFFER_PITCH */
		OUT_RING(ring, 0x00000000);    /* RB_DEPTH_BUFFER_ARRAY_PITCH */
		OUT_RING(ring, 0x00000000);    /* RB_DEPTH_BUFFER_BASE_LO */
		OUT_RING(ring, 0x00000000);    /* RB_DEPTH_BUFFER_BASE_HI */
		OUT_RING(ring, 0x00000000);    /* RB_DEPTH_BUFFER_BASE_GMEM */

		OUT_PKT4(ring, REG_A6XX_GRAS_SU_DEPTH_BUFFER_INFO, 1);
		OUT_RING(ring, A6XX_GRAS_SU_DEPTH_BUFFER_INFO_DEPTH_FORMAT(DEPTH6_NONE));

		OUT_PKT4(ring, REG_A6XX_GRAS_LRZ_BUFFER_BASE_LO, 5);
		OUT_RING(ring, 0x00000000);    /* RB_DEPTH_FLAG_BUFFER_BASE_LO */
		OUT_RING(ring, 0x00000000);    /* RB_DEPTH_FLAG_BUFFER_BASE_HI */
		OUT_RING(ring, 0x00000000);    /* GRAS_LRZ_BUFFER_PITCH */
		OUT_RING(ring, 0x00000000);    /* GRAS_LRZ_FAST_CLEAR_BUFFER_BASE_LO */
		OUT_RING(ring, 0x00000000);    /* GRAS_LRZ_FAST_CLEAR_BUFFER_BASE_HI */

		OUT_PKT4(ring, REG_A6XX_RB_STENCIL_INFO, 1);
		OUT_RING(ring, 0x00000000);     /* RB_STENCIL_INFO */
	}
}

static bool
use_hw_binning(struct fd_batch *batch)
{
	struct fd_gmem_stateobj *gmem = &batch->ctx->gmem;

	// TODO figure out hw limits for binning

	return fd_binning_enabled && ((gmem->nbins_x * gmem->nbins_y) > 2) &&
			(batch->num_draws > 0);
}

static void
patch_draws(struct fd_batch *batch, enum pc_di_vis_cull_mode vismode)
{
	unsigned i;
	for (i = 0; i < fd_patch_num_elements(&batch->draw_patches); i++) {
		struct fd_cs_patch *patch = fd_patch_element(&batch->draw_patches, i);
		*patch->cs = patch->val | DRAW4(0, 0, 0, vismode);
	}
	util_dynarray_resize(&batch->draw_patches, 0);
}

static void
patch_gmem_bases(struct fd_batch *batch)
{
	struct fd_gmem_stateobj *gmem = &batch->ctx->gmem;
	unsigned i;

	for (i = 0; i < fd_patch_num_elements(&batch->gmem_patches); i++) {
		struct fd_cs_patch *patch = fd_patch_element(&batch->gmem_patches, i);
		if (patch->val < MAX_RENDER_TARGETS)
			*patch->cs = gmem->cbuf_base[patch->val];
		else
			*patch->cs = gmem->zsbuf_base[0];
	}
	util_dynarray_resize(&batch->gmem_patches, 0);
}

static void
update_render_cntl(struct fd_batch *batch, bool binning)
{
	struct fd_ringbuffer *ring = batch->gmem;
	uint32_t cntl = 0;

	cntl |= A6XX_RB_RENDER_CNTL_UNK4;
	if (binning)
		cntl |= A6XX_RB_RENDER_CNTL_BINNING;

	OUT_PKT7(ring, CP_REG_WRITE, 3);
	OUT_RING(ring, 0x2);
	OUT_RING(ring, REG_A6XX_RB_RENDER_CNTL);
	OUT_RING(ring, cntl);
}

static void
update_vsc_pipe(struct fd_batch *batch)
{
	struct fd_context *ctx = batch->ctx;
	struct fd6_context *fd6_ctx = fd6_context(ctx);
	struct fd_gmem_stateobj *gmem = &ctx->gmem;
	struct fd_ringbuffer *ring = batch->gmem;
	unsigned n = gmem->nbins_x * gmem->nbins_y;
	int i;

	OUT_PKT4(ring, REG_A6XX_VSC_BIN_SIZE, 3);
	OUT_RING(ring, A6XX_VSC_BIN_SIZE_WIDTH(gmem->bin_w) |
			A6XX_VSC_BIN_SIZE_HEIGHT(gmem->bin_h));
	OUT_RELOCW(ring, fd6_ctx->vsc_data,
			n * A6XX_VSC_DATA_PITCH, 0, 0); /* VSC_SIZE_ADDRESS_LO/HI */

	OUT_PKT4(ring, REG_A6XX_VSC_BIN_COUNT, 1);
	OUT_RING(ring, A6XX_VSC_BIN_COUNT_NX(gmem->nbins_x) |
			A6XX_VSC_BIN_COUNT_NY(gmem->nbins_y));

	OUT_PKT4(ring, REG_A6XX_VSC_PIPE_CONFIG_REG(0), 32);
	for (i = 0; i < 32; i++) {
		struct fd_vsc_pipe *pipe = &ctx->vsc_pipe[i];
		OUT_RING(ring, A6XX_VSC_PIPE_CONFIG_REG_X(pipe->x) |
				A6XX_VSC_PIPE_CONFIG_REG_Y(pipe->y) |
				A6XX_VSC_PIPE_CONFIG_REG_W(pipe->w) |
				A6XX_VSC_PIPE_CONFIG_REG_H(pipe->h));
	}

	OUT_PKT4(ring, REG_A6XX_VSC_PIPE_DATA2_ADDRESS_LO, 4);
	OUT_RELOCW(ring, fd6_ctx->vsc_data2, 0, 0, 0);
	OUT_RING(ring, A6XX_VSC_DATA2_PITCH);
	OUT_RING(ring, fd_bo_size(fd6_ctx->vsc_data2));

	OUT_PKT4(ring, REG_A6XX_VSC_PIPE_DATA_ADDRESS_LO, 4);
	OUT_RELOCW(ring, fd6_ctx->vsc_data, 0, 0, 0);
	OUT_RING(ring, A6XX_VSC_DATA_PITCH);
	OUT_RING(ring, fd_bo_size(fd6_ctx->vsc_data));
}

static void
set_scissor(struct fd_ringbuffer *ring, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2)
{
	OUT_PKT4(ring, REG_A6XX_GRAS_SC_WINDOW_SCISSOR_TL, 2);
	OUT_RING(ring, A6XX_GRAS_SC_WINDOW_SCISSOR_TL_X(x1) |
			 A6XX_GRAS_SC_WINDOW_SCISSOR_TL_Y(y1));
	OUT_RING(ring, A6XX_GRAS_SC_WINDOW_SCISSOR_BR_X(x2) |
			 A6XX_GRAS_SC_WINDOW_SCISSOR_BR_Y(y2));

	OUT_PKT4(ring, REG_A6XX_GRAS_RESOLVE_CNTL_1, 2);
	OUT_RING(ring, A6XX_GRAS_RESOLVE_CNTL_1_X(x1) |
			 A6XX_GRAS_RESOLVE_CNTL_1_Y(y1));
	OUT_RING(ring, A6XX_GRAS_RESOLVE_CNTL_2_X(x2) |
			 A6XX_GRAS_RESOLVE_CNTL_2_Y(y2));
}

static void
set_bin_size(struct fd_ringbuffer *ring, uint32_t w, uint32_t h, uint32_t flag)
{
	OUT_PKT4(ring, REG_A6XX_GRAS_BIN_CONTROL, 1);
	OUT_RING(ring, A6XX_GRAS_BIN_CONTROL_BINW(w) |
			 A6XX_GRAS_BIN_CONTROL_BINH(h) | flag);

	OUT_PKT4(ring, REG_A6XX_RB_BIN_CONTROL, 1);
	OUT_RING(ring, A6XX_RB_BIN_CONTROL_BINW(w) |
			 A6XX_RB_BIN_CONTROL_BINH(h) | flag);

	/* no flag for RB_BIN_CONTROL2... */
	OUT_PKT4(ring, REG_A6XX_RB_BIN_CONTROL2, 1);
	OUT_RING(ring, A6XX_RB_BIN_CONTROL2_BINW(w) |
			 A6XX_RB_BIN_CONTROL2_BINH(h));
}

static void
emit_binning_pass(struct fd_batch *batch)
{
	struct fd_context *ctx = batch->ctx;
	struct fd_ringbuffer *ring = batch->gmem;
	struct fd_gmem_stateobj *gmem = &batch->ctx->gmem;

	uint32_t x1 = gmem->minx;
	uint32_t y1 = gmem->miny;
	uint32_t x2 = gmem->minx + gmem->width - 1;
	uint32_t y2 = gmem->miny + gmem->height - 1;

	set_scissor(ring, x1, y1, x2, y2);

	emit_marker6(ring, 7);
	OUT_PKT7(ring, CP_SET_MARKER, 1);
	OUT_RING(ring, A2XX_CP_SET_MARKER_0_MODE(RM6_BINNING));
	emit_marker6(ring, 7);

	OUT_PKT7(ring, CP_SET_VISIBILITY_OVERRIDE, 1);
	OUT_RING(ring, 0x1);

	OUT_PKT7(ring, CP_SET_MODE, 1);
	OUT_RING(ring, 0x1);

	OUT_WFI5(ring);

	OUT_PKT4(ring, REG_A6XX_VFD_MODE_CNTL, 1);
	OUT_RING(ring, A6XX_VFD_MODE_CNTL_BINNING_PASS);

	update_vsc_pipe(batch);

	OUT_PKT4(ring, REG_A6XX_PC_UNKNOWN_9805, 1);
	OUT_RING(ring, 0x1);

	OUT_PKT4(ring, REG_A6XX_SP_UNKNOWN_A0F8, 1);
	OUT_RING(ring, 0x1);

	OUT_PKT7(ring, CP_EVENT_WRITE, 1);
	OUT_RING(ring, UNK_2C);

	OUT_PKT4(ring, REG_A6XX_RB_WINDOW_OFFSET, 1);
	OUT_RING(ring, A6XX_RB_WINDOW_OFFSET_X(0) |
			A6XX_RB_WINDOW_OFFSET_Y(0));

	OUT_PKT4(ring, REG_A6XX_SP_TP_WINDOW_OFFSET, 1);
	OUT_RING(ring, A6XX_SP_TP_WINDOW_OFFSET_X(0) |
			A6XX_SP_TP_WINDOW_OFFSET_Y(0));

	/* emit IB to binning drawcmds: */
	fd6_emit_ib(ring, batch->draw);

	fd_reset_wfi(batch);

	OUT_PKT7(ring, CP_SET_DRAW_STATE, 3);
	OUT_RING(ring, CP_SET_DRAW_STATE__0_COUNT(0) |
			CP_SET_DRAW_STATE__0_DISABLE_ALL_GROUPS |
			CP_SET_DRAW_STATE__0_GROUP_ID(0));
	OUT_RING(ring, CP_SET_DRAW_STATE__1_ADDR_LO(0));
	OUT_RING(ring, CP_SET_DRAW_STATE__2_ADDR_HI(0));

	OUT_PKT7(ring, CP_EVENT_WRITE, 1);
	OUT_RING(ring, UNK_2D);

	OUT_PKT7(ring, CP_EVENT_WRITE, 4);
	OUT_RING(ring, CACHE_FLUSH_TS);
	OUT_RELOCW(ring, fd6_context(ctx)->blit_mem, 0, 0, 0);  /* ADDR_LO/HI */
	OUT_RING(ring, 0x00000000);

	fd_wfi(batch, ring);
}

static void
disable_msaa(struct fd_ringbuffer *ring)
{
	// TODO MSAA
	OUT_PKT4(ring, REG_A6XX_SP_TP_RAS_MSAA_CNTL, 2);
	OUT_RING(ring, A6XX_SP_TP_RAS_MSAA_CNTL_SAMPLES(MSAA_ONE));
	OUT_RING(ring, A6XX_SP_TP_DEST_MSAA_CNTL_SAMPLES(MSAA_ONE) |
			 A6XX_SP_TP_DEST_MSAA_CNTL_MSAA_DISABLE);

	OUT_PKT4(ring, REG_A6XX_GRAS_RAS_MSAA_CNTL, 2);
	OUT_RING(ring, A6XX_GRAS_RAS_MSAA_CNTL_SAMPLES(MSAA_ONE));
	OUT_RING(ring, A6XX_GRAS_DEST_MSAA_CNTL_SAMPLES(MSAA_ONE) |
			 A6XX_GRAS_DEST_MSAA_CNTL_MSAA_DISABLE);

	OUT_PKT4(ring, REG_A6XX_RB_RAS_MSAA_CNTL, 2);
	OUT_RING(ring, A6XX_RB_RAS_MSAA_CNTL_SAMPLES(MSAA_ONE));
	OUT_RING(ring, A6XX_RB_DEST_MSAA_CNTL_SAMPLES(MSAA_ONE) |
			 A6XX_RB_DEST_MSAA_CNTL_MSAA_DISABLE);
}

/* before first tile */
static void
fd6_emit_tile_init(struct fd_batch *batch)
{
	struct fd_context *ctx = batch->ctx;
	struct fd_ringbuffer *ring = batch->gmem;
	struct pipe_framebuffer_state *pfb = &batch->framebuffer;
	struct fd_gmem_stateobj *gmem = &batch->ctx->gmem;

	fd6_emit_restore(batch, ring);

	fd6_emit_lrz_flush(ring);

	if (batch->lrz_clear)
		fd6_emit_ib(ring, batch->lrz_clear);

	fd6_cache_flush(batch, ring);

	OUT_PKT7(ring, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
	OUT_RING(ring, 0x0);

	/* 0x10000000 for BYPASS.. 0x7c13c080 for GMEM: */
	fd_wfi(batch, ring);
	OUT_PKT4(ring, REG_A6XX_RB_CCU_CNTL, 1);
	OUT_RING(ring, 0x7c400004);   /* RB_CCU_CNTL */

	emit_zs(ring, pfb->zsbuf, &ctx->gmem);
	emit_mrt(ring, pfb, &ctx->gmem);

	patch_gmem_bases(batch);

	disable_msaa(ring);

	if (use_hw_binning(batch)) {
		set_bin_size(ring, gmem->bin_w, gmem->bin_h,
				A6XX_RB_BIN_CONTROL_BINNING_PASS | 0x6000000);
		update_render_cntl(batch, true);
		emit_binning_pass(batch);
		patch_draws(batch, USE_VISIBILITY);

		set_bin_size(ring, gmem->bin_w, gmem->bin_h,
				A6XX_RB_BIN_CONTROL_USE_VIZ | 0x6000000);

		OUT_PKT4(ring, REG_A6XX_VFD_MODE_CNTL, 1);
		OUT_RING(ring, 0x0);
	} else {
		set_bin_size(ring, gmem->bin_w, gmem->bin_h, 0x6000000);
		patch_draws(batch, IGNORE_VISIBILITY);
	}

	update_render_cntl(batch, false);
}

static void
set_window_offset(struct fd_ringbuffer *ring, uint32_t x1, uint32_t y1)
{
	OUT_PKT4(ring, REG_A6XX_RB_WINDOW_OFFSET, 1);
	OUT_RING(ring, A6XX_RB_WINDOW_OFFSET_X(x1) |
			A6XX_RB_WINDOW_OFFSET_Y(y1));

	OUT_PKT4(ring, REG_A6XX_RB_WINDOW_OFFSET2, 1);
	OUT_RING(ring, A6XX_RB_WINDOW_OFFSET2_X(x1) |
			A6XX_RB_WINDOW_OFFSET2_Y(y1));

	OUT_PKT4(ring, REG_A6XX_SP_WINDOW_OFFSET, 1);
	OUT_RING(ring, A6XX_SP_WINDOW_OFFSET_X(x1) |
			A6XX_SP_WINDOW_OFFSET_Y(y1));

	OUT_PKT4(ring, REG_A6XX_SP_TP_WINDOW_OFFSET, 1);
	OUT_RING(ring, A6XX_SP_TP_WINDOW_OFFSET_X(x1) |
			A6XX_SP_TP_WINDOW_OFFSET_Y(y1));
}

/* before mem2gmem */
static void
fd6_emit_tile_prep(struct fd_batch *batch, struct fd_tile *tile)
{
	struct fd_context *ctx = batch->ctx;
	struct fd6_context *fd6_ctx = fd6_context(ctx);
	struct fd_ringbuffer *ring = batch->gmem;

	OUT_PKT7(ring, CP_SET_MARKER, 1);
	OUT_RING(ring, A2XX_CP_SET_MARKER_0_MODE(0x7));

	emit_marker6(ring, 7);
	OUT_PKT7(ring, CP_SET_MARKER, 1);
	OUT_RING(ring, A2XX_CP_SET_MARKER_0_MODE(RM6_GMEM) | 0x10);
	emit_marker6(ring, 7);

	uint32_t x1 = tile->xoff;
	uint32_t y1 = tile->yoff;
	uint32_t x2 = tile->xoff + tile->bin_w - 1;
	uint32_t y2 = tile->yoff + tile->bin_h - 1;

	set_scissor(ring, x1, y1, x2, y2);

	set_window_offset(ring, x1, y1);

	OUT_PKT4(ring, REG_A6XX_VPC_SO_OVERRIDE, 1);
	OUT_RING(ring, A6XX_VPC_SO_OVERRIDE_SO_DISABLE);

	if (use_hw_binning(batch)) {
		struct fd_gmem_stateobj *gmem = &ctx->gmem;
		struct fd_vsc_pipe *pipe = &ctx->vsc_pipe[tile->p];
		unsigned n = gmem->nbins_x * gmem->nbins_y;

		OUT_PKT7(ring, CP_WAIT_FOR_ME, 0);

		OUT_PKT7(ring, CP_SET_VISIBILITY_OVERRIDE, 1);
		OUT_RING(ring, 0x0);

		OUT_PKT7(ring, CP_SET_MODE, 1);
		OUT_RING(ring, 0x0);

		OUT_PKT7(ring, CP_SET_BIN_DATA5, 7);
		OUT_RING(ring, CP_SET_BIN_DATA5_0_VSC_SIZE(pipe->w * pipe->h) |
				CP_SET_BIN_DATA5_0_VSC_N(tile->n));
		OUT_RELOC(ring, fd6_ctx->vsc_data,       /* VSC_PIPE[p].DATA_ADDRESS */
				(tile->p * A6XX_VSC_DATA_PITCH), 0, 0);
		OUT_RELOC(ring, fd6_ctx->vsc_data,       /* VSC_SIZE_ADDRESS + (p * 4) */
				(tile->p * 4) + (n * A6XX_VSC_DATA_PITCH), 0, 0);
		OUT_RELOC(ring, fd6_ctx->vsc_data2,
				(tile->p * A6XX_VSC_DATA2_PITCH), 0, 0);
	} else {
		OUT_PKT7(ring, CP_SET_VISIBILITY_OVERRIDE, 1);
		OUT_RING(ring, 0x1);

		OUT_PKT7(ring, CP_SET_MODE, 1);
		OUT_RING(ring, 0x0);
	}
}

static void
set_blit_scissor(struct fd_batch *batch)
{
	struct fd_ringbuffer *ring = batch->gmem;
	struct pipe_scissor_state blit_scissor;
	struct pipe_framebuffer_state *pfb = &batch->framebuffer;

	blit_scissor.minx = batch->max_scissor.minx;
	blit_scissor.miny = batch->max_scissor.miny;
	blit_scissor.maxx = MIN2(pfb->width, batch->max_scissor.maxx);
	blit_scissor.maxy = MIN2(pfb->height, batch->max_scissor.maxy);

	OUT_PKT4(ring, REG_A6XX_RB_BLIT_SCISSOR_TL, 2);
	OUT_RING(ring,
			 A6XX_RB_BLIT_SCISSOR_TL_X(blit_scissor.minx) |
			 A6XX_RB_BLIT_SCISSOR_TL_Y(blit_scissor.miny));
	OUT_RING(ring,
			 A6XX_RB_BLIT_SCISSOR_BR_X(blit_scissor.maxx - 1) |
			 A6XX_RB_BLIT_SCISSOR_BR_Y(blit_scissor.maxy - 1));
}

static void
emit_blit(struct fd_batch *batch, uint32_t base,
		  struct pipe_surface *psurf,
		  struct fd_resource *rsc)
{
	struct fd_ringbuffer *ring = batch->gmem;
	struct fd_resource_slice *slice;
	uint32_t offset;

	slice = fd_resource_slice(rsc, psurf->u.tex.level);
	offset = fd_resource_offset(rsc, psurf->u.tex.level,
			psurf->u.tex.first_layer);

	debug_assert(psurf->u.tex.first_layer == psurf->u.tex.last_layer);

	enum pipe_format pfmt = psurf->format;
	enum a6xx_color_fmt format = fd6_pipe2color(pfmt);
	uint32_t stride = slice->pitch * rsc->cpp;
	uint32_t size = slice->size0;
	enum a3xx_color_swap swap = fd6_pipe2swap(pfmt);

	// TODO: tile mode
	// bool tiled;
	// tiled = rsc->tile_mode &&
	//   !fd_resource_level_linear(psurf->texture, psurf->u.tex.level);

	OUT_PKT4(ring, REG_A6XX_RB_BLIT_DST_INFO, 5);
	OUT_RING(ring,
			 A6XX_RB_BLIT_DST_INFO_TILE_MODE(TILE6_LINEAR) |
			 A6XX_RB_BLIT_DST_INFO_COLOR_FORMAT(format) |
			 A6XX_RB_BLIT_DST_INFO_COLOR_SWAP(swap));
	OUT_RELOCW(ring, rsc->bo, offset, 0, 0);  /* RB_BLIT_DST_LO/HI */
	OUT_RING(ring, A6XX_RB_BLIT_DST_PITCH(stride));
	OUT_RING(ring, A6XX_RB_BLIT_DST_ARRAY_PITCH(size));

	OUT_PKT4(ring, REG_A6XX_RB_BLIT_BASE_GMEM, 1);
	OUT_RING(ring, base);

	fd6_emit_blit(batch, ring);
}

static void
emit_restore_blit(struct fd_batch *batch, uint32_t base,
				  struct pipe_surface *psurf,
				  struct fd_resource *rsc,
				  unsigned buffer)
{
	struct fd_ringbuffer *ring = batch->gmem;
	uint32_t info = 0;

	switch (buffer) {
	case FD_BUFFER_COLOR:
		info |= A6XX_RB_BLIT_INFO_UNK0;
		break;
	case FD_BUFFER_STENCIL:
		info |= A6XX_RB_BLIT_INFO_UNK0;
		break;
	case FD_BUFFER_DEPTH:
		info |= A6XX_RB_BLIT_INFO_DEPTH | A6XX_RB_BLIT_INFO_UNK0;
		break;
	}

	if (util_format_is_pure_integer(psurf->format))
		info |= A6XX_RB_BLIT_INFO_INTEGER;

	OUT_PKT4(ring, REG_A6XX_RB_BLIT_INFO, 1);
	OUT_RING(ring, info | A6XX_RB_BLIT_INFO_GMEM);

	emit_blit(batch, base, psurf, rsc);
}

/*
 * transfer from system memory to gmem
 */
static void
fd6_emit_tile_mem2gmem(struct fd_batch *batch, struct fd_tile *tile)
{
	struct fd_context *ctx = batch->ctx;
	struct fd_gmem_stateobj *gmem = &ctx->gmem;
	struct pipe_framebuffer_state *pfb = &batch->framebuffer;

	set_blit_scissor(batch);

	if (fd_gmem_needs_restore(batch, tile, FD_BUFFER_COLOR)) {
		unsigned i;
		for (i = 0; i < pfb->nr_cbufs; i++) {
			if (!pfb->cbufs[i])
				continue;
			if (!(batch->restore & (PIPE_CLEAR_COLOR0 << i)))
				continue;
			emit_restore_blit(batch, gmem->cbuf_base[i], pfb->cbufs[i],
							  fd_resource(pfb->cbufs[i]->texture),
							  FD_BUFFER_COLOR);
		}
	}

	if (fd_gmem_needs_restore(batch, tile, FD_BUFFER_DEPTH | FD_BUFFER_STENCIL)) {
		struct fd_resource *rsc = fd_resource(pfb->zsbuf->texture);

		if (!rsc->stencil || fd_gmem_needs_restore(batch, tile, FD_BUFFER_DEPTH)) {
			emit_restore_blit(batch, gmem->zsbuf_base[0], pfb->zsbuf, rsc,
							  FD_BUFFER_DEPTH);
		}
		if (rsc->stencil && fd_gmem_needs_restore(batch, tile, FD_BUFFER_STENCIL)) {
			emit_restore_blit(batch, gmem->zsbuf_base[1], pfb->zsbuf, rsc->stencil,
							  FD_BUFFER_STENCIL);
		}
	}
}

/* before IB to rendering cmds: */
static void
fd6_emit_tile_renderprep(struct fd_batch *batch, struct fd_tile *tile)
{
}

static void
emit_resolve_blit(struct fd_batch *batch, uint32_t base,
				  struct pipe_surface *psurf,
				  struct fd_resource *rsc,
				  unsigned buffer)
{
	struct fd_ringbuffer *ring = batch->gmem;
	uint32_t info = 0;

	if (!rsc->valid)
		return;

	switch (buffer) {
	case FD_BUFFER_COLOR:
		break;
	case FD_BUFFER_STENCIL:
		info |= A6XX_RB_BLIT_INFO_UNK0;
		break;
	case FD_BUFFER_DEPTH:
		info |= A6XX_RB_BLIT_INFO_DEPTH;
		break;
	}

	if (util_format_is_pure_integer(psurf->format))
		info |= A6XX_RB_BLIT_INFO_INTEGER;

	OUT_PKT4(ring, REG_A6XX_RB_BLIT_INFO, 1);
	OUT_RING(ring, info);

	emit_blit(batch, base, psurf, rsc);
}

/*
 * transfer from gmem to system memory (ie. normal RAM)
 */

static void
fd6_emit_tile_gmem2mem(struct fd_batch *batch, struct fd_tile *tile)
{
	struct fd_context *ctx = batch->ctx;
	struct fd_gmem_stateobj *gmem = &ctx->gmem;
	struct pipe_framebuffer_state *pfb = &batch->framebuffer;
	struct fd_ringbuffer *ring = batch->gmem;

	if (use_hw_binning(batch)) {
		OUT_PKT7(ring, CP_SET_MARKER, 1);
		OUT_RING(ring, A2XX_CP_SET_MARKER_0_MODE(0x5) | 0x10);
	}

	OUT_PKT7(ring, CP_SET_DRAW_STATE, 3);
	OUT_RING(ring, CP_SET_DRAW_STATE__0_COUNT(0) |
			CP_SET_DRAW_STATE__0_DISABLE_ALL_GROUPS |
			CP_SET_DRAW_STATE__0_GROUP_ID(0));
	OUT_RING(ring, CP_SET_DRAW_STATE__1_ADDR_LO(0));
	OUT_RING(ring, CP_SET_DRAW_STATE__2_ADDR_HI(0));

	OUT_PKT7(ring, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
	OUT_RING(ring, 0x0);

	emit_marker6(ring, 7);
	OUT_PKT7(ring, CP_SET_MARKER, 1);
	OUT_RING(ring, A2XX_CP_SET_MARKER_0_MODE(RM6_RESOLVE) | 0x10);
	emit_marker6(ring, 7);

	set_blit_scissor(batch);

	if (batch->resolve & (FD_BUFFER_DEPTH | FD_BUFFER_STENCIL)) {
		struct fd_resource *rsc = fd_resource(pfb->zsbuf->texture);

		if (!rsc->stencil || (batch->resolve & FD_BUFFER_DEPTH)) {
			emit_resolve_blit(batch, gmem->zsbuf_base[0], pfb->zsbuf, rsc,
							  FD_BUFFER_DEPTH);
		}
		if (rsc->stencil && (batch->resolve & FD_BUFFER_STENCIL)) {
			emit_resolve_blit(batch, gmem->zsbuf_base[1], pfb->zsbuf, rsc->stencil,
							  FD_BUFFER_STENCIL);
		}
	}

	if (batch->resolve & FD_BUFFER_COLOR) {
		unsigned i;
		for (i = 0; i < pfb->nr_cbufs; i++) {
			if (!pfb->cbufs[i])
				continue;
			if (!(batch->resolve & (PIPE_CLEAR_COLOR0 << i)))
				continue;
			emit_resolve_blit(batch, gmem->cbuf_base[i], pfb->cbufs[i],
							  fd_resource(pfb->cbufs[i]->texture),
							  FD_BUFFER_COLOR);
		}
	}
}

static void
fd6_emit_tile_fini(struct fd_batch *batch)
{
	struct fd_ringbuffer *ring = batch->gmem;

	OUT_PKT4(ring, REG_A6XX_GRAS_LRZ_CNTL, 1);
	OUT_RING(ring, A6XX_GRAS_LRZ_CNTL_ENABLE | A6XX_GRAS_LRZ_CNTL_UNK3);

	fd6_emit_lrz_flush(ring);

	fd6_event_write(batch, ring, CACHE_FLUSH_TS, true);
}

static void
fd6_emit_sysmem_prep(struct fd_batch *batch)
{
	struct pipe_framebuffer_state *pfb = &batch->framebuffer;
	struct fd_ringbuffer *ring = batch->gmem;

	fd6_emit_restore(batch, ring);

	fd6_emit_lrz_flush(ring);

	emit_marker6(ring, 7);
	OUT_PKT7(ring, CP_SET_MARKER, 1);
	OUT_RING(ring, A2XX_CP_SET_MARKER_0_MODE(RM6_BYPASS) | 0x10); /* | 0x10 ? */
	emit_marker6(ring, 7);

	OUT_PKT7(ring, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
	OUT_RING(ring, 0x0);

	fd6_event_write(batch, ring, PC_CCU_INVALIDATE_COLOR, false);
	fd6_cache_flush(batch, ring);

#if 0
	OUT_PKT4(ring, REG_A6XX_PC_POWER_CNTL, 1);
	OUT_RING(ring, 0x00000003);   /* PC_POWER_CNTL */
#endif

#if 0
	OUT_PKT4(ring, REG_A6XX_VFD_POWER_CNTL, 1);
	OUT_RING(ring, 0x00000003);   /* VFD_POWER_CNTL */
#endif

	/* 0x10000000 for BYPASS.. 0x7c13c080 for GMEM: */
	fd_wfi(batch, ring);
	OUT_PKT4(ring, REG_A6XX_RB_CCU_CNTL, 1);
	OUT_RING(ring, 0x10000000);   /* RB_CCU_CNTL */

	set_scissor(ring, 0, 0, pfb->width - 1, pfb->height - 1);

	set_window_offset(ring, 0, 0);

	set_bin_size(ring, 0, 0, 0xc00000); /* 0xc00000 = BYPASS? */

	OUT_PKT7(ring, CP_SET_VISIBILITY_OVERRIDE, 1);
	OUT_RING(ring, 0x1);

	patch_draws(batch, IGNORE_VISIBILITY);

	emit_zs(ring, pfb->zsbuf, NULL);
	emit_mrt(ring, pfb, NULL);

	disable_msaa(ring);
}

static void
fd6_emit_sysmem_fini(struct fd_batch *batch)
{
	struct fd_ringbuffer *ring = batch->gmem;

	OUT_PKT7(ring, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
	OUT_RING(ring, 0x0);

	fd6_emit_lrz_flush(ring);

	fd6_event_write(batch, ring, UNK_1D, true);
}

void
fd6_gmem_init(struct pipe_context *pctx)
{
	struct fd_context *ctx = fd_context(pctx);

	ctx->emit_tile_init = fd6_emit_tile_init;
	ctx->emit_tile_prep = fd6_emit_tile_prep;
	ctx->emit_tile_mem2gmem = fd6_emit_tile_mem2gmem;
	ctx->emit_tile_renderprep = fd6_emit_tile_renderprep;
	ctx->emit_tile_gmem2mem = fd6_emit_tile_gmem2mem;
	ctx->emit_tile_fini = fd6_emit_tile_fini;
	ctx->emit_sysmem_prep = fd6_emit_sysmem_prep;
	ctx->emit_sysmem_fini = fd6_emit_sysmem_fini;
}
