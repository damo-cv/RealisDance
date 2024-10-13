/*
 * Copyright (C) 2012 Rob Clark <robclark@freedesktop.org>
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

#ifndef FREEDRENO_RESOURCE_H_
#define FREEDRENO_RESOURCE_H_

#include "util/list.h"
#include "util/u_range.h"
#include "util/u_transfer_helper.h"

#include "freedreno_batch.h"
#include "freedreno_util.h"

/* Texture Layout on a3xx:
 *
 * Each mipmap-level contains all of it's layers (ie. all cubmap
 * faces, all 1d/2d array elements, etc).  The texture sampler is
 * programmed with the start address of each mipmap level, and hw
 * derives the layer offset within the level.
 *
 * Texture Layout on a4xx:
 *
 * For cubemap and 2d array, each layer contains all of it's mipmap
 * levels (layer_first layout).
 *
 * 3d textures are layed out as on a3xx, but unknown about 3d-array
 * textures.
 *
 * In either case, the slice represents the per-miplevel information,
 * but in layer_first layout it only includes the first layer, and
 * an additional offset of (rsc->layer_size * layer) must be added.
 */
struct fd_resource_slice {
	uint32_t offset;         /* offset of first layer in slice */
	uint32_t pitch;
	uint32_t size0;          /* size of first layer in slice */
};

struct set;

struct fd_resource {
	struct pipe_resource base;
	struct fd_bo *bo;
	uint32_t cpp;
	enum pipe_format internal_format;
	bool layer_first;        /* see above description */
	uint32_t layer_size;
	struct fd_resource_slice slices[MAX_MIP_LEVELS];
	/* buffer range that has been initialized */
	struct util_range valid_buffer_range;
	bool valid;

	/* reference to the resource holding stencil data for a z32_s8 texture */
	/* TODO rename to secondary or auxiliary? */
	struct fd_resource *stencil;

	/* bitmask of in-flight batches which reference this resource.  Note
	 * that the batch doesn't hold reference to resources (but instead
	 * the fd_ringbuffer holds refs to the underlying fd_bo), but in case
	 * the resource is destroyed we need to clean up the batch's weak
	 * references to us.
	 */
	uint32_t batch_mask;

	/* reference to batch that writes this resource: */
	struct fd_batch *write_batch;

	/* Set of batches whose batch-cache key references this resource.
	 * We need to track this to know which batch-cache entries to
	 * invalidate if, for example, the resource is invalidated or
	 * shadowed.
	 */
	uint32_t bc_batch_mask;

	/* Sequence # incremented each time bo changes: */
	uint16_t seqno;

	unsigned tile_mode : 2;
	unsigned preferred_tile_mode : 2;

	/*
	 * LRZ
	 */
	bool lrz_valid : 1;
	uint16_t lrz_width;  // for lrz clear, does this differ from lrz_pitch?
	uint16_t lrz_height;
	uint16_t lrz_pitch;
	struct fd_bo *lrz;
};

static inline struct fd_resource *
fd_resource(struct pipe_resource *ptex)
{
	return (struct fd_resource *)ptex;
}

static inline bool
pending(struct fd_resource *rsc, bool write)
{
	/* if we have a pending GPU write, we are busy in any case: */
	if (rsc->write_batch)
		return true;

	/* if CPU wants to write, but we are pending a GPU read, we are busy: */
	if (write && rsc->batch_mask)
		return true;

	if (rsc->stencil && pending(rsc->stencil, write))
		return true;

	return false;
}

struct fd_transfer {
	struct pipe_transfer base;
	struct pipe_resource *staging_prsc;
	struct pipe_box staging_box;
};

static inline struct fd_transfer *
fd_transfer(struct pipe_transfer *ptrans)
{
	return (struct fd_transfer *)ptrans;
}

static inline struct fd_resource_slice *
fd_resource_slice(struct fd_resource *rsc, unsigned level)
{
	assert(level <= rsc->base.last_level);
	return &rsc->slices[level];
}

/* get offset for specified mipmap level and texture/array layer */
static inline uint32_t
fd_resource_offset(struct fd_resource *rsc, unsigned level, unsigned layer)
{
	struct fd_resource_slice *slice = fd_resource_slice(rsc, level);
	unsigned offset;
	if (rsc->layer_first) {
		offset = slice->offset + (rsc->layer_size * layer);
	} else {
		offset = slice->offset + (slice->size0 * layer);
	}
	debug_assert(offset < fd_bo_size(rsc->bo));
	return offset;
}

/* This might be a5xx specific, but higher mipmap levels are always linear: */
static inline bool
fd_resource_level_linear(struct pipe_resource *prsc, int level)
{
	unsigned w = u_minify(prsc->width0, level);
	if (w < 16)
		return true;
	return false;
}

void fd_blitter_pipe_begin(struct fd_context *ctx, bool render_cond, bool discard,
		enum fd_render_stage stage);
void fd_blitter_pipe_end(struct fd_context *ctx);

void fd_resource_screen_init(struct pipe_screen *pscreen);
void fd_resource_context_init(struct pipe_context *pctx);

uint32_t fd_setup_slices(struct fd_resource *rsc);
void fd_resource_resize(struct pipe_resource *prsc, uint32_t sz);

bool fd_render_condition_check(struct pipe_context *pctx);

#endif /* FREEDRENO_RESOURCE_H_ */
