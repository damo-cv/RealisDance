/*
 * Copyright 2010 Jerome Glisse <glisse@freedesktop.org>
 * Copyright 2018 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "sid.h"
#include "si_pipe.h"

#include "util/u_format.h"

static void si_dma_copy_buffer(struct si_context *ctx,
				struct pipe_resource *dst,
				struct pipe_resource *src,
				uint64_t dst_offset,
				uint64_t src_offset,
				uint64_t size)
{
	struct radeon_cmdbuf *cs = ctx->dma_cs;
	unsigned i, ncopy, count, max_size, sub_cmd, shift;
	struct r600_resource *rdst = r600_resource(dst);
	struct r600_resource *rsrc = r600_resource(src);

	/* Mark the buffer range of destination as valid (initialized),
	 * so that transfer_map knows it should wait for the GPU when mapping
	 * that range. */
	util_range_add(&rdst->valid_buffer_range, dst_offset,
		       dst_offset + size);

	dst_offset += rdst->gpu_address;
	src_offset += rsrc->gpu_address;

	/* see whether we should use the dword-aligned or byte-aligned copy */
	if (!(dst_offset % 4) && !(src_offset % 4) && !(size % 4)) {
		sub_cmd = SI_DMA_COPY_DWORD_ALIGNED;
		shift = 2;
		max_size = SI_DMA_COPY_MAX_DWORD_ALIGNED_SIZE;
	} else {
		sub_cmd = SI_DMA_COPY_BYTE_ALIGNED;
		shift = 0;
		max_size = SI_DMA_COPY_MAX_BYTE_ALIGNED_SIZE;
	}

	ncopy = DIV_ROUND_UP(size, max_size);
	si_need_dma_space(ctx, ncopy * 5, rdst, rsrc);

	for (i = 0; i < ncopy; i++) {
		count = MIN2(size, max_size);
		radeon_emit(cs, SI_DMA_PACKET(SI_DMA_PACKET_COPY, sub_cmd,
					      count >> shift));
		radeon_emit(cs, dst_offset);
		radeon_emit(cs, src_offset);
		radeon_emit(cs, (dst_offset >> 32UL) & 0xff);
		radeon_emit(cs, (src_offset >> 32UL) & 0xff);
		dst_offset += count;
		src_offset += count;
		size -= count;
	}
}

static void si_dma_copy_tile(struct si_context *ctx,
			     struct pipe_resource *dst,
			     unsigned dst_level,
			     unsigned dst_x,
			     unsigned dst_y,
			     unsigned dst_z,
			     struct pipe_resource *src,
			     unsigned src_level,
			     unsigned src_x,
			     unsigned src_y,
			     unsigned src_z,
			     unsigned copy_height,
			     unsigned pitch,
			     unsigned bpp)
{
	struct radeon_cmdbuf *cs = ctx->dma_cs;
	struct si_texture *ssrc = (struct si_texture*)src;
	struct si_texture *sdst = (struct si_texture*)dst;
	unsigned dst_mode = sdst->surface.u.legacy.level[dst_level].mode;
	bool detile = dst_mode == RADEON_SURF_MODE_LINEAR_ALIGNED;
	struct si_texture *linear = detile ? sdst : ssrc;
	struct si_texture *tiled = detile ? ssrc : sdst;
	unsigned linear_lvl = detile ? dst_level : src_level;
	unsigned tiled_lvl = detile ? src_level : dst_level;
	struct radeon_info *info = &ctx->screen->info;
	unsigned index = tiled->surface.u.legacy.tiling_index[tiled_lvl];
	unsigned tile_mode = info->si_tile_mode_array[index];
	unsigned array_mode, lbpp, pitch_tile_max, slice_tile_max, size;
	unsigned ncopy, height, cheight, i;
	unsigned linear_x, linear_y, linear_z,  tiled_x, tiled_y, tiled_z;
	unsigned sub_cmd, bank_h, bank_w, mt_aspect, nbanks, tile_split, mt;
	uint64_t base, addr;
	unsigned pipe_config;

	assert(dst_mode != ssrc->surface.u.legacy.level[src_level].mode);

	sub_cmd = SI_DMA_COPY_TILED;
	lbpp = util_logbase2(bpp);
	pitch_tile_max = ((pitch / bpp) / 8) - 1;

	linear_x = detile ? dst_x : src_x;
	linear_y = detile ? dst_y : src_y;
	linear_z = detile ? dst_z : src_z;
	tiled_x = detile ? src_x : dst_x;
	tiled_y = detile ? src_y : dst_y;
	tiled_z = detile ? src_z : dst_z;

	assert(!util_format_is_depth_and_stencil(tiled->buffer.b.b.format));

	array_mode = G_009910_ARRAY_MODE(tile_mode);
	slice_tile_max = (tiled->surface.u.legacy.level[tiled_lvl].nblk_x *
			  tiled->surface.u.legacy.level[tiled_lvl].nblk_y) / (8*8) - 1;
	/* linear height must be the same as the slice tile max height, it's ok even
	 * if the linear destination/source have smaller heigh as the size of the
	 * dma packet will be using the copy_height which is always smaller or equal
	 * to the linear height
	 */
	height = tiled->surface.u.legacy.level[tiled_lvl].nblk_y;
	base = tiled->surface.u.legacy.level[tiled_lvl].offset;
	addr = linear->surface.u.legacy.level[linear_lvl].offset;
	addr += (uint64_t)linear->surface.u.legacy.level[linear_lvl].slice_size_dw * 4 * linear_z;
	addr += linear_y * pitch + linear_x * bpp;
	bank_h = G_009910_BANK_HEIGHT(tile_mode);
	bank_w = G_009910_BANK_WIDTH(tile_mode);
	mt_aspect = G_009910_MACRO_TILE_ASPECT(tile_mode);
	/* Non-depth modes don't have TILE_SPLIT set. */
	tile_split = util_logbase2(tiled->surface.u.legacy.tile_split >> 6);
	nbanks = G_009910_NUM_BANKS(tile_mode);
	base += tiled->buffer.gpu_address;
	addr += linear->buffer.gpu_address;

	pipe_config = G_009910_PIPE_CONFIG(tile_mode);
	mt = G_009910_MICRO_TILE_MODE(tile_mode);
	size = copy_height * pitch;
	ncopy = DIV_ROUND_UP(size, SI_DMA_COPY_MAX_DWORD_ALIGNED_SIZE);
	si_need_dma_space(ctx, ncopy * 9, &sdst->buffer, &ssrc->buffer);

	for (i = 0; i < ncopy; i++) {
		cheight = copy_height;
		if (cheight * pitch > SI_DMA_COPY_MAX_DWORD_ALIGNED_SIZE) {
			cheight = SI_DMA_COPY_MAX_DWORD_ALIGNED_SIZE / pitch;
		}
		size = cheight * pitch;
		radeon_emit(cs, SI_DMA_PACKET(SI_DMA_PACKET_COPY, sub_cmd, size / 4));
		radeon_emit(cs, base >> 8);
		radeon_emit(cs, (detile << 31) | (array_mode << 27) |
				(lbpp << 24) | (bank_h << 21) |
				(bank_w << 18) | (mt_aspect << 16));
		radeon_emit(cs, (pitch_tile_max << 0) | ((height - 1) << 16));
		radeon_emit(cs, (slice_tile_max << 0) | (pipe_config << 26));
		radeon_emit(cs, (tiled_x << 0) | (tiled_z << 18));
		radeon_emit(cs, (tiled_y << 0) | (tile_split << 21) | (nbanks << 25) | (mt << 27));
		radeon_emit(cs, addr & 0xfffffffc);
		radeon_emit(cs, (addr >> 32UL) & 0xff);
		copy_height -= cheight;
		addr += cheight * pitch;
		tiled_y += cheight;
	}
}

static void si_dma_copy(struct pipe_context *ctx,
			struct pipe_resource *dst,
			unsigned dst_level,
			unsigned dstx, unsigned dsty, unsigned dstz,
			struct pipe_resource *src,
			unsigned src_level,
			const struct pipe_box *src_box)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_texture *ssrc = (struct si_texture*)src;
	struct si_texture *sdst = (struct si_texture*)dst;
	unsigned dst_pitch, src_pitch, bpp, dst_mode, src_mode;
	unsigned src_w, dst_w;
	unsigned src_x, src_y;
	unsigned dst_x = dstx, dst_y = dsty, dst_z = dstz;

	if (sctx->dma_cs == NULL ||
	    src->flags & PIPE_RESOURCE_FLAG_SPARSE ||
	    dst->flags & PIPE_RESOURCE_FLAG_SPARSE) {
		goto fallback;
	}

	if (dst->target == PIPE_BUFFER && src->target == PIPE_BUFFER) {
		si_dma_copy_buffer(sctx, dst, src, dst_x, src_box->x, src_box->width);
		return;
	}

	/* XXX: Using the asynchronous DMA engine for multi-dimensional
	 * operations seems to cause random GPU lockups for various people.
	 * While the root cause for this might need to be fixed in the kernel,
	 * let's disable it for now.
	 *
	 * Before re-enabling this, please make sure you can hit all newly
	 * enabled paths in your testing, preferably with both piglit and real
	 * world apps, and get in touch with people on the bug reports below
	 * for stability testing.
	 *
	 * https://bugs.freedesktop.org/show_bug.cgi?id=85647
	 * https://bugs.freedesktop.org/show_bug.cgi?id=83500
	 */
	goto fallback;

	if (src_box->depth > 1 ||
	    !si_prepare_for_dma_blit(sctx, sdst, dst_level, dstx, dsty,
				     dstz, ssrc, src_level, src_box))
		goto fallback;

	src_x = util_format_get_nblocksx(src->format, src_box->x);
	dst_x = util_format_get_nblocksx(src->format, dst_x);
	src_y = util_format_get_nblocksy(src->format, src_box->y);
	dst_y = util_format_get_nblocksy(src->format, dst_y);

	bpp = sdst->surface.bpe;
	dst_pitch = sdst->surface.u.legacy.level[dst_level].nblk_x * sdst->surface.bpe;
	src_pitch = ssrc->surface.u.legacy.level[src_level].nblk_x * ssrc->surface.bpe;
	src_w = u_minify(ssrc->buffer.b.b.width0, src_level);
	dst_w = u_minify(sdst->buffer.b.b.width0, dst_level);

	dst_mode = sdst->surface.u.legacy.level[dst_level].mode;
	src_mode = ssrc->surface.u.legacy.level[src_level].mode;

	if (src_pitch != dst_pitch || src_box->x || dst_x || src_w != dst_w ||
	    src_box->width != src_w ||
	    src_box->height != u_minify(ssrc->buffer.b.b.height0, src_level) ||
	    src_box->height != u_minify(sdst->buffer.b.b.height0, dst_level) ||
	    ssrc->surface.u.legacy.level[src_level].nblk_y !=
	    sdst->surface.u.legacy.level[dst_level].nblk_y) {
		/* FIXME si can do partial blit */
		goto fallback;
	}
	/* the x test here are currently useless (because we don't support partial blit)
	 * but keep them around so we don't forget about those
	 */
	if ((src_pitch % 8) || (src_box->x % 8) || (dst_x % 8) ||
	    (src_box->y % 8) || (dst_y % 8) || (src_box->height % 8)) {
		goto fallback;
	}

	if (src_mode == dst_mode) {
		uint64_t dst_offset, src_offset;
		/* simple dma blit would do NOTE code here assume :
		 *   src_box.x/y == 0
		 *   dst_x/y == 0
		 *   dst_pitch == src_pitch
		 */
		src_offset= ssrc->surface.u.legacy.level[src_level].offset;
		src_offset += (uint64_t)ssrc->surface.u.legacy.level[src_level].slice_size_dw * 4 * src_box->z;
		src_offset += src_y * src_pitch + src_x * bpp;
		dst_offset = sdst->surface.u.legacy.level[dst_level].offset;
		dst_offset += (uint64_t)sdst->surface.u.legacy.level[dst_level].slice_size_dw * 4 * dst_z;
		dst_offset += dst_y * dst_pitch + dst_x * bpp;
		si_dma_copy_buffer(sctx, dst, src, dst_offset, src_offset,
				   (uint64_t)ssrc->surface.u.legacy.level[src_level].slice_size_dw * 4);
	} else {
		si_dma_copy_tile(sctx, dst, dst_level, dst_x, dst_y, dst_z,
				 src, src_level, src_x, src_y, src_box->z,
				 src_box->height / ssrc->surface.blk_h,
				 dst_pitch, bpp);
	}
	return;

fallback:
	si_resource_copy_region(ctx, dst, dst_level, dstx, dsty, dstz,
				src, src_level, src_box);
}

void si_init_dma_functions(struct si_context *sctx)
{
	sctx->dma_copy = si_dma_copy;
}
