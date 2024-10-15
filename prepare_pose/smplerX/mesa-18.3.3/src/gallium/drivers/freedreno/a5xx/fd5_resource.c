/*
 * Copyright (C) 2018 Rob Clark <robclark@freedesktop.org>
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

#include "fd5_resource.h"

/* indexed by cpp: */
static const struct {
	unsigned pitchalign;
	unsigned heightalign;
} tile_alignment[] = {
	[1]  = { 128, 32 },
	[2]  = { 128, 16 },
	[3]  = { 128, 16 },
	[4]  = {  64, 16 },
	[8]  = {  64, 16 },
	[12] = {  64, 16 },
	[16] = {  64, 16 },
};

/* NOTE: good way to test this is:  (for example)
 *  piglit/bin/texelFetch fs sampler2D 100x100x1-100x300x1
 */
static uint32_t
setup_slices(struct fd_resource *rsc, uint32_t alignment, enum pipe_format format)
{
	struct pipe_resource *prsc = &rsc->base;
	struct fd_screen *screen = fd_screen(prsc->screen);
	enum util_format_layout layout = util_format_description(format)->layout;
	uint32_t pitchalign = screen->gmem_alignw;
	uint32_t heightalign;
	uint32_t level, size = 0;
	uint32_t width = prsc->width0;
	uint32_t height = prsc->height0;
	uint32_t depth = prsc->depth0;
	/* in layer_first layout, the level (slice) contains just one
	 * layer (since in fact the layer contains the slices)
	 */
	uint32_t layers_in_level = rsc->layer_first ? 1 : prsc->array_size;

	heightalign = tile_alignment[rsc->cpp].heightalign;

	for (level = 0; level <= prsc->last_level; level++) {
		struct fd_resource_slice *slice = fd_resource_slice(rsc, level);
		bool linear_level = fd_resource_level_linear(prsc, level);
		uint32_t aligned_height = height;
		uint32_t blocks;

		if (rsc->tile_mode && !linear_level) {
			pitchalign = tile_alignment[rsc->cpp].pitchalign;
			aligned_height = align(aligned_height, heightalign);
		} else {
			pitchalign = 64;

			/* The blits used for mem<->gmem work at a granularity of
			 * 32x32, which can cause faults due to over-fetch on the
			 * last level.  The simple solution is to over-allocate a
			 * bit the last level to ensure any over-fetch is harmless.
			 * The pitch is already sufficiently aligned, but height
			 * may not be:
			 */
			if ((level == prsc->last_level) && (prsc->target != PIPE_BUFFER))
				aligned_height = align(aligned_height, 32);
		}

		if (layout == UTIL_FORMAT_LAYOUT_ASTC)
			slice->pitch =
				util_align_npot(width, pitchalign * util_format_get_blockwidth(format));
		else
			slice->pitch = align(width, pitchalign);

		slice->offset = size;
		blocks = util_format_get_nblocks(format, slice->pitch, aligned_height);

		/* 1d array and 2d array textures must all have the same layer size
		 * for each miplevel on a3xx. 3d textures can have different layer
		 * sizes for high levels, but the hw auto-sizer is buggy (or at least
		 * different than what this code does), so as soon as the layer size
		 * range gets into range, we stop reducing it.
		 */
		if (prsc->target == PIPE_TEXTURE_3D && (
					level == 1 ||
					(level > 1 && rsc->slices[level - 1].size0 > 0xf000)))
			slice->size0 = align(blocks * rsc->cpp, alignment);
		else if (level == 0 || rsc->layer_first || alignment == 1)
			slice->size0 = align(blocks * rsc->cpp, alignment);
		else
			slice->size0 = rsc->slices[level - 1].size0;

#if 0
		debug_printf("%s: %ux%ux%u@%u: %2u: stride=%4u, size=%7u, aligned_height=%3u\n",
				util_format_name(prsc->format),
				prsc->width0, prsc->height0, prsc->depth0, rsc->cpp,
				level, slice->pitch * rsc->cpp,
				slice->size0 * depth * layers_in_level,
				aligned_height);
#endif

		size += slice->size0 * depth * layers_in_level;

		width = u_minify(width, 1);
		height = u_minify(height, 1);
		depth = u_minify(depth, 1);
	}

	return size;
}

uint32_t
fd5_setup_slices(struct fd_resource *rsc)
{
	uint32_t alignment;

	switch (rsc->base.target) {
	case PIPE_TEXTURE_3D:
		rsc->layer_first = false;
		alignment = 4096;
		break;
	default:
		rsc->layer_first = true;
		alignment = 1;
		break;
	}

	return setup_slices(rsc, alignment, rsc->base.format);
}
