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
#include "util/u_prim.h"

#include "freedreno_state.h"
#include "freedreno_resource.h"

#include "fd6_draw.h"
#include "fd6_context.h"
#include "fd6_emit.h"
#include "fd6_program.h"
#include "fd6_format.h"
#include "fd6_zsa.h"

/* some bits in common w/ a4xx: */
#include "a4xx/fd4_draw.h"

static void
draw_emit_indirect(struct fd_batch *batch, struct fd_ringbuffer *ring,
				   enum pc_di_primtype primtype,
				   const struct pipe_draw_info *info,
				   unsigned index_offset)
{
	struct fd_resource *ind = fd_resource(info->indirect->buffer);

	if (info->index_size) {
		struct pipe_resource *idx = info->index.resource;
		unsigned max_indicies = (idx->width0 - info->indirect->offset) /
			info->index_size;

		OUT_PKT7(ring, CP_DRAW_INDX_INDIRECT, 6);
		OUT_RINGP(ring, DRAW4(primtype, DI_SRC_SEL_DMA,
							  fd4_size2indextype(info->index_size), 0),
				  &batch->draw_patches);
		OUT_RELOC(ring, fd_resource(idx)->bo,
				  index_offset, 0, 0);
		// XXX: Check A5xx vs A6xx
		OUT_RING(ring, A5XX_CP_DRAW_INDX_INDIRECT_3_MAX_INDICES(max_indicies));
		OUT_RELOC(ring, ind->bo, info->indirect->offset, 0, 0);
	} else {
		OUT_PKT7(ring, CP_DRAW_INDIRECT, 3);
		OUT_RINGP(ring, DRAW4(primtype, DI_SRC_SEL_AUTO_INDEX, 0, 0),
				  &batch->draw_patches);
		OUT_RELOC(ring, ind->bo, info->indirect->offset, 0, 0);
	}
}

static void
draw_emit(struct fd_batch *batch, struct fd_ringbuffer *ring,
		  enum pc_di_primtype primtype,
		  const struct pipe_draw_info *info,
		  unsigned index_offset)
{
	if (info->index_size) {
		assert(!info->has_user_indices);

		struct pipe_resource *idx_buffer = info->index.resource;
		uint32_t idx_size = info->index_size * info->count;
		uint32_t idx_offset = index_offset + info->start * info->index_size;

		/* leave vis mode blank for now, it will be patched up when
		 * we know if we are binning or not
		 */
		uint32_t draw = CP_DRAW_INDX_OFFSET_0_PRIM_TYPE(primtype) |
			CP_DRAW_INDX_OFFSET_0_SOURCE_SELECT(DI_SRC_SEL_DMA) |
			CP_DRAW_INDX_OFFSET_0_INDEX_SIZE(fd4_size2indextype(info->index_size)) |
			0x2000;

		OUT_PKT7(ring, CP_DRAW_INDX_OFFSET, 7);
		OUT_RINGP(ring, draw, &batch->draw_patches);
		OUT_RING(ring, info->instance_count);    /* NumInstances */
		OUT_RING(ring, info->count);             /* NumIndices */
		OUT_RING(ring, 0x0);           /* XXX */
		OUT_RELOC(ring, fd_resource(idx_buffer)->bo, idx_offset, 0, 0);
		OUT_RING (ring, idx_size);
	} else {
		/* leave vis mode blank for now, it will be patched up when
		 * we know if we are binning or not
		 */
		uint32_t draw = CP_DRAW_INDX_OFFSET_0_PRIM_TYPE(primtype) |
			CP_DRAW_INDX_OFFSET_0_SOURCE_SELECT(DI_SRC_SEL_AUTO_INDEX) |
			0x2000;

		OUT_PKT7(ring, CP_DRAW_INDX_OFFSET, 3);
		OUT_RINGP(ring, draw, &batch->draw_patches);
		OUT_RING(ring, info->instance_count);    /* NumInstances */
		OUT_RING(ring, info->count);             /* NumIndices */
	}
}

/* fixup dirty shader state in case some "unrelated" (from the state-
 * tracker's perspective) state change causes us to switch to a
 * different variant.
 */
static void
fixup_shader_state(struct fd_context *ctx, struct ir3_shader_key *key)
{
	struct fd6_context *fd6_ctx = fd6_context(ctx);
	struct ir3_shader_key *last_key = &fd6_ctx->last_key;

	if (!ir3_shader_key_equal(last_key, key)) {
		if (ir3_shader_key_changes_fs(last_key, key)) {
			ctx->dirty_shader[PIPE_SHADER_FRAGMENT] |= FD_DIRTY_SHADER_PROG;
			ctx->dirty |= FD_DIRTY_PROG;
		}

		if (ir3_shader_key_changes_vs(last_key, key)) {
			ctx->dirty_shader[PIPE_SHADER_VERTEX] |= FD_DIRTY_SHADER_PROG;
			ctx->dirty |= FD_DIRTY_PROG;
		}

		fd6_ctx->last_key = *key;
	}
}

static bool
fd6_draw_vbo(struct fd_context *ctx, const struct pipe_draw_info *info,
             unsigned index_offset)
{
	struct fd6_context *fd6_ctx = fd6_context(ctx);
	struct fd6_emit emit = {
		.ctx = ctx,
		.vtx  = &ctx->vtx,
		.info = info,
		.key = {
			.vs = ctx->prog.vp,
			.fs = ctx->prog.fp,
			.key = {
				.color_two_side = ctx->rasterizer->light_twoside,
				.vclamp_color = ctx->rasterizer->clamp_vertex_color,
				.fclamp_color = ctx->rasterizer->clamp_fragment_color,
				.rasterflat = ctx->rasterizer->flatshade,
				.ucp_enables = ctx->rasterizer->clip_plane_enable,
				.has_per_samp = (fd6_ctx->fsaturate || fd6_ctx->vsaturate ||
						fd6_ctx->fastc_srgb || fd6_ctx->vastc_srgb),
				.vsaturate_s = fd6_ctx->vsaturate_s,
				.vsaturate_t = fd6_ctx->vsaturate_t,
				.vsaturate_r = fd6_ctx->vsaturate_r,
				.fsaturate_s = fd6_ctx->fsaturate_s,
				.fsaturate_t = fd6_ctx->fsaturate_t,
				.fsaturate_r = fd6_ctx->fsaturate_r,
				.vastc_srgb = fd6_ctx->vastc_srgb,
				.fastc_srgb = fd6_ctx->fastc_srgb,
				.vsamples = ctx->tex[PIPE_SHADER_VERTEX].samples,
				.fsamples = ctx->tex[PIPE_SHADER_FRAGMENT].samples,
			}
		},
		.rasterflat = ctx->rasterizer->flatshade,
		.sprite_coord_enable = ctx->rasterizer->sprite_coord_enable,
		.sprite_coord_mode = ctx->rasterizer->sprite_coord_mode,
	};

	fixup_shader_state(ctx, &emit.key.key);

	if (!(ctx->dirty & FD_DIRTY_PROG)) {
		emit.prog = fd6_ctx->prog;
	} else {
		fd6_ctx->prog = fd6_emit_get_prog(&emit);
	}

	emit.dirty = ctx->dirty;      /* *after* fixup_shader_state() */
	emit.bs = fd6_emit_get_prog(&emit)->bs;
	emit.vs = fd6_emit_get_prog(&emit)->vs;
	emit.fs = fd6_emit_get_prog(&emit)->fs;

	const struct ir3_shader_variant *vp = emit.vs;
	const struct ir3_shader_variant *fp = emit.fs;

	/* do regular pass first, since that is more likely to fail compiling: */

	if (!vp || !fp)
		return false;

	ctx->stats.vs_regs += ir3_shader_halfregs(vp);
	ctx->stats.fs_regs += ir3_shader_halfregs(fp);

	/* figure out whether we need to disable LRZ write for binning
	 * pass using draw pass's fp:
	 */
	emit.no_lrz_write = fp->writes_pos || fp->has_kill;

	struct fd_ringbuffer *ring = ctx->batch->draw;
	enum pc_di_primtype primtype = ctx->primtypes[info->mode];

	fd6_emit_state(ring, &emit);

	OUT_PKT4(ring, REG_A6XX_VFD_INDEX_OFFSET, 2);
	OUT_RING(ring, info->index_size ? info->index_bias : info->start); /* VFD_INDEX_OFFSET */
	OUT_RING(ring, info->start_instance);   /* VFD_INSTANCE_START_OFFSET */

	OUT_PKT4(ring, REG_A6XX_PC_RESTART_INDEX, 1);
	OUT_RING(ring, info->primitive_restart ? /* PC_RESTART_INDEX */
			info->restart_index : 0xffffffff);

	/* for debug after a lock up, write a unique counter value
	 * to scratch7 for each draw, to make it easier to match up
	 * register dumps to cmdstream.  The combination of IB
	 * (scratch6) and DRAW is enough to "triangulate" the
	 * particular draw that caused lockup.
	 */
	emit_marker6(ring, 7);

	if (info->indirect) {
		draw_emit_indirect(ctx->batch, ring, primtype,
						   info, index_offset);
	} else {
		draw_emit(ctx->batch, ring, primtype,
				  info, index_offset);
	}

	emit_marker6(ring, 7);
	fd_reset_wfi(ctx->batch);

	if (emit.streamout_mask) {
		struct fd_ringbuffer *ring = ctx->batch->draw;

		for (unsigned i = 0; i < PIPE_MAX_SO_BUFFERS; i++) {
			if (emit.streamout_mask & (1 << i)) {
				fd6_event_write(ctx->batch, ring, FLUSH_SO_0 + i, false);
			}
		}
	}

	fd_context_all_clean(ctx);

	return true;
}

static bool is_z32(enum pipe_format format)
{
	switch (format) {
	case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
	case PIPE_FORMAT_Z32_UNORM:
	case PIPE_FORMAT_Z32_FLOAT:
		return true;
	default:
		return false;
	}
}

static void
fd6_clear_lrz(struct fd_batch *batch, struct fd_resource *zsbuf, double depth)
{
	struct fd_ringbuffer *ring;

	// TODO mid-frame clears (ie. app doing crazy stuff)??  Maybe worth
	// splitting both clear and lrz clear out into their own rb's.  And
	// just throw away any draws prior to clear.  (Anything not fullscreen
	// clear, just fallback to generic path that treats it as a normal
	// draw

	if (!batch->lrz_clear) {
		batch->lrz_clear = fd_submit_new_ringbuffer(batch->submit, 0x1000, 0);
	}

	ring = batch->lrz_clear;

	emit_marker6(ring, 7);
	OUT_PKT7(ring, CP_SET_MARKER, 1);
	OUT_RING(ring, A2XX_CP_SET_MARKER_0_MODE(RM6_BYPASS));
	emit_marker6(ring, 7);

	OUT_PKT4(ring, REG_A6XX_RB_CCU_CNTL, 1);
	OUT_RING(ring, 0x10000000);

	OUT_PKT4(ring, REG_A6XX_HLSQ_UPDATE_CNTL, 1);
	OUT_RING(ring, 0x7ffff);

	emit_marker6(ring, 7);
	OUT_PKT7(ring, CP_SET_MARKER, 1);
	OUT_RING(ring, A2XX_CP_SET_MARKER_0_MODE(0xc));
	emit_marker6(ring, 7);

	OUT_PKT4(ring, REG_A6XX_RB_UNKNOWN_8C01, 1);
	OUT_RING(ring, 0x0);

	OUT_PKT4(ring, REG_A6XX_SP_PS_2D_SRC_INFO, 13);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);

	OUT_PKT4(ring, REG_A6XX_SP_UNKNOWN_ACC0, 1);
	OUT_RING(ring, 0x0000f410);

	OUT_PKT4(ring, REG_A6XX_GRAS_2D_BLIT_CNTL, 1);
	OUT_RING(ring, A6XX_GRAS_2D_BLIT_CNTL_COLOR_FORMAT(RB6_R16_UNORM) |
			0x4f00080);

	OUT_PKT4(ring, REG_A6XX_RB_2D_BLIT_CNTL, 1);
	OUT_RING(ring, A6XX_RB_2D_BLIT_CNTL_COLOR_FORMAT(RB6_R16_UNORM) |
			0x4f00080);

	fd6_event_write(batch, ring, UNK_1D, true);
	fd6_event_write(batch, ring, PC_CCU_INVALIDATE_COLOR, false);

	OUT_PKT4(ring, REG_A6XX_RB_2D_SRC_SOLID_C0, 4);
	OUT_RING(ring, fui(depth));
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);

	OUT_PKT4(ring, REG_A6XX_RB_2D_DST_INFO, 9);
	OUT_RING(ring, A6XX_RB_2D_DST_INFO_COLOR_FORMAT(RB6_R16_UNORM) |
			A6XX_RB_2D_DST_INFO_TILE_MODE(TILE6_LINEAR) |
			A6XX_RB_2D_DST_INFO_COLOR_SWAP(WZYX));
	OUT_RELOCW(ring, zsbuf->lrz, 0, 0, 0);
	OUT_RING(ring, A6XX_RB_2D_DST_SIZE_PITCH(zsbuf->lrz_pitch * 2));
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);

	OUT_PKT4(ring, REG_A6XX_GRAS_2D_SRC_TL_X, 4);
	OUT_RING(ring, A6XX_GRAS_2D_SRC_TL_X_X(0));
	OUT_RING(ring, A6XX_GRAS_2D_SRC_BR_X_X(0));
	OUT_RING(ring, A6XX_GRAS_2D_SRC_TL_Y_Y(0));
	OUT_RING(ring, A6XX_GRAS_2D_SRC_BR_Y_Y(0));

	OUT_PKT4(ring, REG_A6XX_GRAS_2D_DST_TL, 2);
	OUT_RING(ring, A6XX_GRAS_2D_DST_TL_X(0) |
			A6XX_GRAS_2D_DST_TL_Y(0));
	OUT_RING(ring, A6XX_GRAS_2D_DST_BR_X(zsbuf->lrz_width - 1) |
			A6XX_GRAS_2D_DST_BR_Y(zsbuf->lrz_height - 1));

	fd6_event_write(batch, ring, 0x3f, false);

	OUT_WFI5(ring);

	OUT_PKT4(ring, REG_A6XX_RB_UNKNOWN_8E04, 1);
	OUT_RING(ring, 0x1000000);

	OUT_PKT7(ring, CP_BLIT, 1);
	OUT_RING(ring, CP_BLIT_0_OP(BLIT_OP_SCALE));

	OUT_WFI5(ring);

	OUT_PKT4(ring, REG_A6XX_RB_UNKNOWN_8E04, 1);
	OUT_RING(ring, 0x0);

	fd6_event_write(batch, ring, UNK_1D, true);
	fd6_event_write(batch, ring, FACENESS_FLUSH, true);
	fd6_event_write(batch, ring, CACHE_FLUSH_TS, true);

	fd6_cache_flush(batch, ring);
}

static bool
fd6_clear(struct fd_context *ctx, unsigned buffers,
		const union pipe_color_union *color, double depth, unsigned stencil)
{
	struct pipe_framebuffer_state *pfb = &ctx->batch->framebuffer;
	struct pipe_scissor_state *scissor = fd_context_get_scissor(ctx);
	struct fd_ringbuffer *ring = ctx->batch->draw;

	if ((buffers & (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL)) &&
			is_z32(pfb->zsbuf->format))
		return false;

	OUT_PKT4(ring, REG_A6XX_RB_BLIT_SCISSOR_TL, 2);
	OUT_RING(ring, A6XX_RB_BLIT_SCISSOR_TL_X(scissor->minx) |
			 A6XX_RB_BLIT_SCISSOR_TL_Y(scissor->miny));
	OUT_RING(ring, A6XX_RB_BLIT_SCISSOR_BR_X(scissor->maxx - 1) |
			 A6XX_RB_BLIT_SCISSOR_BR_Y(scissor->maxy - 1));

	if (buffers & PIPE_CLEAR_COLOR) {
		for (int i = 0; i < pfb->nr_cbufs; i++) {
			union util_color uc = {0};

			if (!pfb->cbufs[i])
				continue;

			if (!(buffers & (PIPE_CLEAR_COLOR0 << i)))
				continue;

			enum pipe_format pfmt = pfb->cbufs[i]->format;

			// XXX I think RB_CLEAR_COLOR_DWn wants to take into account SWAP??
			union pipe_color_union swapped;
			switch (fd6_pipe2swap(pfmt)) {
			case WZYX:
				swapped.ui[0] = color->ui[0];
				swapped.ui[1] = color->ui[1];
				swapped.ui[2] = color->ui[2];
				swapped.ui[3] = color->ui[3];
				break;
			case WXYZ:
				swapped.ui[2] = color->ui[0];
				swapped.ui[1] = color->ui[1];
				swapped.ui[0] = color->ui[2];
				swapped.ui[3] = color->ui[3];
				break;
			case ZYXW:
				swapped.ui[3] = color->ui[0];
				swapped.ui[0] = color->ui[1];
				swapped.ui[1] = color->ui[2];
				swapped.ui[2] = color->ui[3];
				break;
			case XYZW:
				swapped.ui[3] = color->ui[0];
				swapped.ui[2] = color->ui[1];
				swapped.ui[1] = color->ui[2];
				swapped.ui[0] = color->ui[3];
				break;
			}

			if (util_format_is_pure_uint(pfmt)) {
				util_format_write_4ui(pfmt, swapped.ui, 0, &uc, 0, 0, 0, 1, 1);
			} else if (util_format_is_pure_sint(pfmt)) {
				util_format_write_4i(pfmt, swapped.i, 0, &uc, 0, 0, 0, 1, 1);
			} else {
				util_pack_color(swapped.f, pfmt, &uc);
			}

			OUT_PKT4(ring, REG_A6XX_RB_BLIT_DST_INFO, 1);
			OUT_RING(ring, A6XX_RB_BLIT_DST_INFO_TILE_MODE(TILE6_LINEAR) |
				A6XX_RB_BLIT_DST_INFO_COLOR_FORMAT(fd6_pipe2color(pfmt)));

			OUT_PKT4(ring, REG_A6XX_RB_BLIT_INFO, 1);
			OUT_RING(ring, A6XX_RB_BLIT_INFO_GMEM |
				A6XX_RB_BLIT_INFO_CLEAR_MASK(0xf));

			OUT_PKT4(ring, REG_A6XX_RB_BLIT_BASE_GMEM, 1);
			OUT_RINGP(ring, i, &ctx->batch->gmem_patches);

			OUT_PKT4(ring, REG_A6XX_RB_UNKNOWN_88D0, 1);
			OUT_RING(ring, 0);

			OUT_PKT4(ring, REG_A6XX_RB_BLIT_CLEAR_COLOR_DW0, 4);
			OUT_RING(ring, uc.ui[0]);
			OUT_RING(ring, uc.ui[1]);
			OUT_RING(ring, uc.ui[2]);
			OUT_RING(ring, uc.ui[3]);

			fd6_emit_blit(ctx->batch, ring);
		}
	}

	if (pfb->zsbuf && (buffers & (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL))) {
		enum pipe_format pfmt = pfb->zsbuf->format;
		uint32_t clear = util_pack_z_stencil(pfmt, depth, stencil);
		uint32_t mask = 0;

		if (buffers & PIPE_CLEAR_DEPTH)
			mask |= 0x1;

		if (buffers & PIPE_CLEAR_STENCIL)
			mask |= 0x2;

		OUT_PKT4(ring, REG_A6XX_RB_BLIT_DST_INFO, 1);
		OUT_RING(ring, A6XX_RB_BLIT_DST_INFO_TILE_MODE(TILE6_LINEAR) |
			A6XX_RB_BLIT_DST_INFO_COLOR_FORMAT(fd6_pipe2color(pfmt)));

		OUT_PKT4(ring, REG_A6XX_RB_BLIT_INFO, 1);
		OUT_RING(ring, A6XX_RB_BLIT_INFO_GMEM |
			// XXX UNK0 for separate stencil ??
			A6XX_RB_BLIT_INFO_DEPTH |
			A6XX_RB_BLIT_INFO_CLEAR_MASK(mask));

		OUT_PKT4(ring, REG_A6XX_RB_BLIT_BASE_GMEM, 1);
		OUT_RINGP(ring, MAX_RENDER_TARGETS, &ctx->batch->gmem_patches);

		OUT_PKT4(ring, REG_A6XX_RB_UNKNOWN_88D0, 1);
		OUT_RING(ring, 0);

		OUT_PKT4(ring, REG_A6XX_RB_BLIT_CLEAR_COLOR_DW0, 1);
		OUT_RING(ring, clear);

		fd6_emit_blit(ctx->batch, ring);

		if (pfb->zsbuf && (buffers & PIPE_CLEAR_DEPTH)) {
			struct fd_resource *zsbuf = fd_resource(pfb->zsbuf->texture);
			if (zsbuf->lrz) {
				zsbuf->lrz_valid = true;
				fd6_clear_lrz(ctx->batch, zsbuf, depth);
			}
		}
	}

	return true;
}

void
fd6_draw_init(struct pipe_context *pctx)
{
	struct fd_context *ctx = fd_context(pctx);
	ctx->draw_vbo = fd6_draw_vbo;
	ctx->clear = fd6_clear;
}
