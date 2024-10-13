/*
 * Copyright (C) 2017 Rob Clark <robclark@freedesktop.org>
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

#include "util/u_dump.h"

#include "freedreno_blitter.h"
#include "freedreno_fence.h"
#include "freedreno_resource.h"

#include "fd6_blitter.h"
#include "fd6_format.h"
#include "fd6_emit.h"

/* Make sure none of the requested dimensions extend beyond the size of the
 * resource.  Not entirely sure why this happens, but sometimes it does, and
 * w/ 2d blt doesn't have wrap modes like a sampler, so force those cases
 * back to u_blitter
 */
static bool
ok_dims(const struct pipe_resource *r, const struct pipe_box *b, int lvl)
{
	int last_layer =
		r->target == PIPE_TEXTURE_3D ? u_minify(r->depth0, lvl)
		: r->array_size;


	return (b->x >= 0) && (b->x + b->width <= u_minify(r->width0, lvl)) &&
		(b->y >= 0) && (b->y + b->height <= u_minify(r->height0, lvl)) &&
		(b->z >= 0) && (b->z + b->depth <= last_layer);
}

#define DEBUG_BLIT_FALLBACK 0
#define fail_if(cond)													\
	do {																\
		if (cond) {														\
			if (DEBUG_BLIT_FALLBACK) {									\
				fprintf(stderr, "falling back: %s for blit:\n", #cond);	\
				util_dump_blit_info(stderr, info);						\
				fprintf(stderr, "\nsrc: ");								\
				util_dump_resource(stderr, info->src.resource);			\
				fprintf(stderr, "\ndst: ");								\
				util_dump_resource(stderr, info->dst.resource);			\
				fprintf(stderr, "\n");									\
			}															\
			return false;												\
		}																\
	} while (0)

static bool
can_do_blit(const struct pipe_blit_info *info)
{
	/* I think we can do scaling, but not in z dimension since that would
	 * require blending..
	 */
	fail_if(info->dst.box.depth != info->src.box.depth);

	/* We can blit if both or neither formats are compressed formats... */
	fail_if(util_format_is_compressed(info->src.format) !=
			util_format_is_compressed(info->src.format));

	/* ... but only if they're the same compression format. */
	fail_if(util_format_is_compressed(info->src.format) &&
			info->src.format != info->dst.format);

	/* hw ignores {SRC,DST}_INFO.COLOR_SWAP if {SRC,DST}_INFO.TILE_MODE
	 * is set (not linear).  We can kind of get around that when tiling/
	 * untiling by setting both src and dst COLOR_SWAP=WZYX, but that
	 * means the formats must match:
	 */
	fail_if((fd_resource(info->dst.resource)->tile_mode ||
			 fd_resource(info->src.resource)->tile_mode) &&
			info->dst.format != info->src.format);

	/* src box can be inverted, which we don't support.. dst box cannot: */
	fail_if((info->src.box.width < 0) || (info->src.box.height < 0));

	fail_if(!ok_dims(info->src.resource, &info->src.box, info->src.level));

	fail_if(!ok_dims(info->dst.resource, &info->dst.box, info->dst.level));

	debug_assert(info->dst.box.width >= 0);
	debug_assert(info->dst.box.height >= 0);
	debug_assert(info->dst.box.depth >= 0);

	fail_if(info->dst.resource->nr_samples + info->src.resource->nr_samples > 2);

	fail_if(info->window_rectangle_include);

	fail_if(info->render_condition_enable);

	fail_if(info->alpha_blend);

	fail_if(info->mask != util_format_get_mask(info->src.format));

	fail_if(info->mask != util_format_get_mask(info->dst.format));

	return true;
}

static void
emit_setup(struct fd_ringbuffer *ring)
{
	OUT_PKT7(ring, CP_EVENT_WRITE, 1);
	OUT_RING(ring, PC_CCU_INVALIDATE_COLOR);

	OUT_PKT7(ring, CP_EVENT_WRITE, 1);
	OUT_RING(ring, LRZ_FLUSH);

	OUT_PKT7(ring, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
	OUT_RING(ring, 0x0);

	OUT_WFI5(ring);

	OUT_PKT4(ring, REG_A6XX_RB_CCU_CNTL, 1);
	OUT_RING(ring, 0x10000000);
}

/* buffers need to be handled specially since x/width can exceed the bounds
 * supported by hw.. if necessary decompose into (potentially) two 2D blits
 */
static void
emit_blit_buffer(struct fd_ringbuffer *ring, const struct pipe_blit_info *info)
{
	const struct pipe_box *sbox = &info->src.box;
	const struct pipe_box *dbox = &info->dst.box;
	struct fd_resource *src, *dst;
	unsigned sshift, dshift;

	if (DEBUG_BLIT_FALLBACK) {
		fprintf(stderr, "buffer blit: ");
		util_dump_blit_info(stderr, info);
		fprintf(stderr, "\ndst resource: ");
		util_dump_resource(stderr, info->dst.resource);
		fprintf(stderr, "\nsrc resource: ");
		util_dump_resource(stderr, info->src.resource);
		fprintf(stderr, "\n");
	}

	src = fd_resource(info->src.resource);
	dst = fd_resource(info->dst.resource);

	debug_assert(src->cpp == 1);
	debug_assert(dst->cpp == 1);
	debug_assert(info->src.resource->format == info->dst.resource->format);
	debug_assert((sbox->y == 0) && (sbox->height == 1));
	debug_assert((dbox->y == 0) && (dbox->height == 1));
	debug_assert((sbox->z == 0) && (sbox->depth == 1));
	debug_assert((dbox->z == 0) && (dbox->depth == 1));
	debug_assert(sbox->width == dbox->width);
	debug_assert(info->src.level == 0);
	debug_assert(info->dst.level == 0);

	/*
	 * Buffers can have dimensions bigger than max width, remap into
	 * multiple 1d blits to fit within max dimension
	 *
	 * Note that blob uses .ARRAY_PITCH=128 for blitting buffers, which
	 * seems to prevent overfetch related faults.  Not quite sure what
	 * the deal is there.
	 *
	 * Low 6 bits of SRC/DST addresses need to be zero (ie. address
	 * aligned to 64) so we need to shift src/dst x1/x2 to make up the
	 * difference.  On top of already splitting up the blit so width
	 * isn't > 16k.
	 *
	 * We perhaps could do a bit better, if src and dst are aligned but
	 * in the worst case this means we have to split the copy up into
	 * 16k (0x4000) minus 64 (0x40).
	 */

	sshift = sbox->x & 0x3f;
	dshift = dbox->x & 0x3f;

	OUT_PKT7(ring, CP_SET_MARKER, 1);
	OUT_RING(ring, A2XX_CP_SET_MARKER_0_MODE(RM6_BLIT2DSCALE));

	uint32_t blit_cntl = A6XX_RB_2D_BLIT_CNTL_COLOR_FORMAT(RB6_R8_UNORM) | 0x20f00000;
	OUT_PKT4(ring, REG_A6XX_RB_2D_BLIT_CNTL, 1);
	OUT_RING(ring, blit_cntl);

	OUT_PKT4(ring, REG_A6XX_GRAS_2D_BLIT_CNTL, 1);
	OUT_RING(ring, blit_cntl);

	for (unsigned off = 0; off < sbox->width; off += (0x4000 - 0x40)) {
		unsigned soff, doff, w, p;

		soff = (sbox->x + off) & ~0x3f;
		doff = (dbox->x + off) & ~0x3f;

		w = MIN2(sbox->width - off, (0x4000 - 0x40));
		p = align(w, 64);

		debug_assert((soff + w) <= fd_bo_size(src->bo));
		debug_assert((doff + w) <= fd_bo_size(dst->bo));

		/*
		 * Emit source:
		 */
		OUT_PKT4(ring, REG_A6XX_SP_PS_2D_SRC_INFO, 13);
		OUT_RING(ring, A6XX_SP_PS_2D_SRC_INFO_COLOR_FORMAT(RB6_R8_UNORM) |
				A6XX_SP_PS_2D_SRC_INFO_TILE_MODE(TILE6_LINEAR) |
				 A6XX_SP_PS_2D_SRC_INFO_COLOR_SWAP(WZYX) | 0x500000);
		OUT_RING(ring, A6XX_SP_PS_2D_SRC_SIZE_WIDTH(sshift + w) |
				 A6XX_SP_PS_2D_SRC_SIZE_HEIGHT(1)); /* SP_PS_2D_SRC_SIZE */
		OUT_RELOC(ring, src->bo, soff, 0, 0);    /* SP_PS_2D_SRC_LO/HI */
		OUT_RING(ring, A6XX_SP_PS_2D_SRC_PITCH_PITCH(p));

		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);

		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);

		/*
		 * Emit destination:
		 */
		OUT_PKT4(ring, REG_A6XX_RB_2D_DST_INFO, 9);
		OUT_RING(ring, A6XX_RB_2D_DST_INFO_COLOR_FORMAT(RB6_R8_UNORM) |
				 A6XX_RB_2D_DST_INFO_TILE_MODE(TILE6_LINEAR) |
				 A6XX_RB_2D_DST_INFO_COLOR_SWAP(WZYX));
		OUT_RELOC(ring, dst->bo, doff, 0, 0);    /* RB_2D_DST_LO/HI */
		OUT_RING(ring, A6XX_RB_2D_DST_SIZE_PITCH(p));
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);

		/*
		 * Blit command:
		 */
		OUT_PKT4(ring, REG_A6XX_GRAS_2D_SRC_TL_X, 4);
		OUT_RING(ring, A6XX_GRAS_2D_SRC_TL_X_X(sshift));
		OUT_RING(ring, A6XX_GRAS_2D_SRC_BR_X_X(sshift + w - 1));
		OUT_RING(ring, A6XX_GRAS_2D_SRC_TL_Y_Y(0));
		OUT_RING(ring, A6XX_GRAS_2D_SRC_BR_Y_Y(0));

		OUT_PKT4(ring, REG_A6XX_GRAS_2D_DST_TL, 2);
		OUT_RING(ring, A6XX_GRAS_2D_DST_TL_X(dshift) | A6XX_GRAS_2D_DST_TL_Y(0));
		OUT_RING(ring, A6XX_GRAS_2D_DST_BR_X(dshift + w - 1) | A6XX_GRAS_2D_DST_BR_Y(0));

		OUT_PKT7(ring, CP_EVENT_WRITE, 1);
		OUT_RING(ring, 0x3f);
		OUT_WFI5(ring);

		OUT_PKT4(ring, 0x8c01, 1);
		OUT_RING(ring, 0);

		OUT_PKT4(ring, 0xacc0, 1);
		OUT_RING(ring, 0xf180);

		OUT_PKT4(ring, 0x8e04, 1);
		OUT_RING(ring, 0x01000000);

		OUT_PKT7(ring, CP_BLIT, 1);
		OUT_RING(ring, CP_BLIT_0_OP(BLIT_OP_SCALE));

		OUT_WFI5(ring);

		OUT_PKT4(ring, 0x8e04, 1);
		OUT_RING(ring, 0);
	}
}

static void
emit_blit_texture(struct fd_ringbuffer *ring, const struct pipe_blit_info *info)
{
	const struct pipe_box *sbox = &info->src.box;
	const struct pipe_box *dbox = &info->dst.box;
	struct fd_resource *src, *dst;
	struct fd_resource_slice *sslice, *dslice;
	enum a6xx_color_fmt sfmt, dfmt;
	enum a6xx_tile_mode stile, dtile;
	enum a3xx_color_swap sswap, dswap;
	unsigned spitch, dpitch;
	unsigned sx1, sy1, sx2, sy2;
	unsigned dx1, dy1, dx2, dy2;

	if (DEBUG_BLIT_FALLBACK) {
		fprintf(stderr, "texture blit: ");
		util_dump_blit_info(stderr, info);
		fprintf(stderr, "\ndst resource: ");
		util_dump_resource(stderr, info->dst.resource);
		fprintf(stderr, "\nsrc resource: ");
		util_dump_resource(stderr, info->src.resource);
		fprintf(stderr, "\n");
	}

	src = fd_resource(info->src.resource);
	dst = fd_resource(info->dst.resource);

	sslice = fd_resource_slice(src, info->src.level);
	dslice = fd_resource_slice(dst, info->dst.level);

	sfmt = fd6_pipe2color(info->src.format);
	dfmt = fd6_pipe2color(info->dst.format);

	int blocksize = util_format_get_blocksize(info->src.format);
	int blockwidth = util_format_get_blockwidth(info->src.format);
	int blockheight = util_format_get_blockheight(info->src.format);
	int nelements;

	stile = fd_resource_level_linear(info->src.resource, info->src.level) ?
			TILE6_LINEAR : src->tile_mode;
	dtile = fd_resource_level_linear(info->dst.resource, info->dst.level) ?
			TILE6_LINEAR : dst->tile_mode;

	sswap = fd6_pipe2swap(info->src.format);
	dswap = fd6_pipe2swap(info->dst.format);

	if (util_format_is_compressed(info->src.format)) {
		debug_assert(info->src.format == info->dst.format);
		sfmt = dfmt = RB6_R8_UNORM;
		nelements = blocksize;
	} else {
		debug_assert(!util_format_is_compressed(info->dst.format));
		nelements = 1;
	}

	spitch = DIV_ROUND_UP(sslice->pitch, blockwidth) * src->cpp;
	dpitch = DIV_ROUND_UP(dslice->pitch, blockwidth) * dst->cpp;

	sx1 = sbox->x / blockwidth * nelements;
	sy1 = sbox->y / blockheight;
	sx2 = DIV_ROUND_UP(sbox->x + sbox->width, blockwidth) * nelements - 1;
	sy2 = DIV_ROUND_UP(sbox->y + sbox->height, blockheight) - 1;

	dx1 = dbox->x / blockwidth * nelements;
	dy1 = dbox->y / blockheight;
	dx2 = DIV_ROUND_UP(dbox->x + dbox->width, blockwidth) * nelements - 1;
	dy2 = DIV_ROUND_UP(dbox->y + dbox->height, blockheight) - 1;

	uint32_t width = DIV_ROUND_UP(u_minify(src->base.width0, info->src.level), blockwidth) * nelements;
	uint32_t height = DIV_ROUND_UP(u_minify(src->base.height0, info->src.level), blockheight);

	/* if dtile, then dswap ignored by hw, and likewise if stile then sswap
	 * ignored by hw.. but in this case we have already rejected the blit
	 * if src and dst formats differ, so juse use WZYX for both src and
	 * dst swap mode (so we don't change component order)
	 */
	if (stile || dtile) {
		debug_assert(info->src.format == info->dst.format);
		sswap = dswap = WZYX;
	}

	OUT_PKT7(ring, CP_SET_MARKER, 1);
	OUT_RING(ring, A2XX_CP_SET_MARKER_0_MODE(RM6_BLIT2DSCALE));

	uint32_t blit_cntl = A6XX_RB_2D_BLIT_CNTL_COLOR_FORMAT(dfmt) | 0xf00000;

	if (dtile != stile)
		blit_cntl |= 0x20000000;

	if (info->scissor_enable) {
		OUT_PKT4(ring, REG_A6XX_GRAS_RESOLVE_CNTL_1, 2);
		OUT_RING(ring, A6XX_GRAS_RESOLVE_CNTL_1_X(info->scissor.minx) |
				 A6XX_GRAS_RESOLVE_CNTL_1_Y(info->scissor.miny));
		OUT_RING(ring, A6XX_GRAS_RESOLVE_CNTL_1_X(info->scissor.maxx - 1) |
				 A6XX_GRAS_RESOLVE_CNTL_1_Y(info->scissor.maxy - 1));
		blit_cntl |= A6XX_RB_2D_BLIT_CNTL_SCISSOR;
	}

	OUT_PKT4(ring, REG_A6XX_RB_2D_BLIT_CNTL, 1);
	OUT_RING(ring, blit_cntl);

	OUT_PKT4(ring, REG_A6XX_GRAS_2D_BLIT_CNTL, 1);
	OUT_RING(ring, blit_cntl);

	for (unsigned i = 0; i < info->dst.box.depth; i++) {
		unsigned soff = fd_resource_offset(src, info->src.level, sbox->z + i);
		unsigned doff = fd_resource_offset(dst, info->dst.level, dbox->z + i);

		/*
		 * Emit source:
		 */
		uint32_t filter = 0;
		if (info->filter == PIPE_TEX_FILTER_LINEAR)
			filter = A6XX_SP_PS_2D_SRC_INFO_FILTER;

		OUT_PKT4(ring, REG_A6XX_SP_PS_2D_SRC_INFO, 13);
		OUT_RING(ring, A6XX_SP_PS_2D_SRC_INFO_COLOR_FORMAT(sfmt) |
				A6XX_SP_PS_2D_SRC_INFO_TILE_MODE(stile) |
				A6XX_SP_PS_2D_SRC_INFO_COLOR_SWAP(sswap) | 0x500000 | filter);
		OUT_RING(ring, A6XX_SP_PS_2D_SRC_SIZE_WIDTH(width) |
				 A6XX_SP_PS_2D_SRC_SIZE_HEIGHT(height)); /* SP_PS_2D_SRC_SIZE */
		OUT_RELOC(ring, src->bo, soff, 0, 0);    /* SP_PS_2D_SRC_LO/HI */
		OUT_RING(ring, A6XX_SP_PS_2D_SRC_PITCH_PITCH(spitch));
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);

		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);

		/*
		 * Emit destination:
		 */
		OUT_PKT4(ring, REG_A6XX_RB_2D_DST_INFO, 9);
		OUT_RING(ring, A6XX_RB_2D_DST_INFO_COLOR_FORMAT(dfmt) |
				 A6XX_RB_2D_DST_INFO_TILE_MODE(dtile) |
				 A6XX_RB_2D_DST_INFO_COLOR_SWAP(dswap));
		OUT_RELOC(ring, dst->bo, doff, 0, 0);    /* RB_2D_DST_LO/HI */
		OUT_RING(ring, A6XX_RB_2D_DST_SIZE_PITCH(dpitch));
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);

		/*
		 * Blit command:
		 */
		OUT_PKT4(ring, REG_A6XX_GRAS_2D_SRC_TL_X, 4);
		OUT_RING(ring, A6XX_GRAS_2D_SRC_TL_X_X(sx1));
		OUT_RING(ring, A6XX_GRAS_2D_SRC_BR_X_X(sx2));
		OUT_RING(ring, A6XX_GRAS_2D_SRC_TL_Y_Y(sy1));
		OUT_RING(ring, A6XX_GRAS_2D_SRC_BR_Y_Y(sy2));

		OUT_PKT4(ring, REG_A6XX_GRAS_2D_DST_TL, 2);
		OUT_RING(ring, A6XX_GRAS_2D_DST_TL_X(dx1) | A6XX_GRAS_2D_DST_TL_Y(dy1));
		OUT_RING(ring, A6XX_GRAS_2D_DST_BR_X(dx2) | A6XX_GRAS_2D_DST_BR_Y(dy2));

		OUT_PKT7(ring, CP_EVENT_WRITE, 1);
		OUT_RING(ring, 0x3f);
		OUT_WFI5(ring);

		OUT_PKT4(ring, 0x8c01, 1);
		OUT_RING(ring, 0);

		OUT_PKT4(ring, 0xacc0, 1);
		OUT_RING(ring, 0xf180);

		OUT_PKT4(ring, 0x8e04, 1);
		OUT_RING(ring, 0x01000000);

		OUT_PKT7(ring, CP_BLIT, 1);
		OUT_RING(ring, CP_BLIT_0_OP(BLIT_OP_SCALE));

		OUT_WFI5(ring);

		OUT_PKT4(ring, 0x8e04, 1);
		OUT_RING(ring, 0);
	}
}

static void
fd6_blit(struct pipe_context *pctx, const struct pipe_blit_info *info)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_batch *batch;

	if (!can_do_blit(info)) {
		fd_blitter_pipe_begin(ctx, info->render_condition_enable, false, FD_STAGE_BLIT);
		fd_blitter_blit(ctx, info);
		fd_blitter_pipe_end(ctx);
		return;
	}

	fd_fence_ref(pctx->screen, &ctx->last_fence, NULL);

	batch = fd_bc_alloc_batch(&ctx->screen->batch_cache, ctx, true);

	fd6_emit_restore(batch, batch->draw);
	fd6_emit_lrz_flush(batch->draw);

	mtx_lock(&ctx->screen->lock);

	fd_batch_resource_used(batch, fd_resource(info->src.resource), false);
	fd_batch_resource_used(batch, fd_resource(info->dst.resource), true);

	mtx_unlock(&ctx->screen->lock);

	emit_setup(batch->draw);

	if ((info->src.resource->target == PIPE_BUFFER) &&
			(info->dst.resource->target == PIPE_BUFFER)) {
		assert(fd_resource(info->src.resource)->tile_mode == TILE6_LINEAR);
		assert(fd_resource(info->dst.resource)->tile_mode == TILE6_LINEAR);
		emit_blit_buffer(batch->draw, info);
	} else {
		/* I don't *think* we need to handle blits between buffer <-> !buffer */
		debug_assert(info->src.resource->target != PIPE_BUFFER);
		debug_assert(info->dst.resource->target != PIPE_BUFFER);
		emit_blit_texture(batch->draw, info);
	}

	fd6_event_write(batch, batch->draw, 0x1d, true);
	fd6_event_write(batch, batch->draw, FACENESS_FLUSH, true);
	fd6_event_write(batch, batch->draw, CACHE_FLUSH_TS, true);

	fd_resource(info->dst.resource)->valid = true;
	batch->needs_flush = true;

	fd_batch_flush(batch, false, false);
	fd_batch_reference(&batch, NULL);
}

static void
fd6_resource_copy_region(struct pipe_context *pctx,
		struct pipe_resource *dst,
		unsigned dst_level,
		unsigned dstx, unsigned dsty, unsigned dstz,
		struct pipe_resource *src,
		unsigned src_level,
		const struct pipe_box *src_box)
{
	struct pipe_blit_info info;

	debug_assert(src->format == dst->format);

	memset(&info, 0, sizeof info);
	info.dst.resource = dst;
	info.dst.level = dst_level;
	info.dst.box.x = dstx;
	info.dst.box.y = dsty;
	info.dst.box.z = dstz;
	info.dst.box.width = src_box->width;
	info.dst.box.height = src_box->height;
	assert(info.dst.box.width >= 0);
	assert(info.dst.box.height >= 0);
	info.dst.box.depth = 1;
	info.dst.format = dst->format;
	info.src.resource = src;
	info.src.level = src_level;
	info.src.box = *src_box;
	info.src.format = src->format;
	info.mask = util_format_get_mask(src->format);
	info.filter = PIPE_TEX_FILTER_NEAREST;
	info.scissor_enable = 0;

	fd6_blit(pctx, &info);
}

void
fd6_blitter_init(struct pipe_context *pctx)
{
	if (fd_mesa_debug & FD_DBG_NOBLIT)
		return;

	pctx->resource_copy_region = fd6_resource_copy_region;
	pctx->blit = fd6_blit;
}

unsigned
fd6_tile_mode(const struct pipe_resource *tmpl)
{
	/* basically just has to be a format we can blit, so uploads/downloads
	 * via linear staging buffer works:
	 */
	return TILE6_3;
}
