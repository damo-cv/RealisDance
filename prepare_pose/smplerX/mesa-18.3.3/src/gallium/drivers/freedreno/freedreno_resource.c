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

#include "util/u_format.h"
#include "util/u_format_rgtc.h"
#include "util/u_format_zs.h"
#include "util/u_inlines.h"
#include "util/u_transfer.h"
#include "util/u_string.h"
#include "util/u_surface.h"
#include "util/set.h"

#include "freedreno_resource.h"
#include "freedreno_batch_cache.h"
#include "freedreno_fence.h"
#include "freedreno_screen.h"
#include "freedreno_surface.h"
#include "freedreno_context.h"
#include "freedreno_query_hw.h"
#include "freedreno_util.h"

#include <errno.h>

/* XXX this should go away, needed for 'struct winsys_handle' */
#include "state_tracker/drm_driver.h"

/**
 * Go through the entire state and see if the resource is bound
 * anywhere. If it is, mark the relevant state as dirty. This is
 * called on realloc_bo to ensure the neccessary state is re-
 * emitted so the GPU looks at the new backing bo.
 */
static void
rebind_resource(struct fd_context *ctx, struct pipe_resource *prsc)
{
	/* VBOs */
	for (unsigned i = 0; i < ctx->vtx.vertexbuf.count && !(ctx->dirty & FD_DIRTY_VTXBUF); i++) {
		if (ctx->vtx.vertexbuf.vb[i].buffer.resource == prsc)
			ctx->dirty |= FD_DIRTY_VTXBUF;
	}

	/* per-shader-stage resources: */
	for (unsigned stage = 0; stage < PIPE_SHADER_TYPES; stage++) {
		/* Constbufs.. note that constbuf[0] is normal uniforms emitted in
		 * cmdstream rather than by pointer..
		 */
		const unsigned num_ubos = util_last_bit(ctx->constbuf[stage].enabled_mask);
		for (unsigned i = 1; i < num_ubos; i++) {
			if (ctx->dirty_shader[stage] & FD_DIRTY_SHADER_CONST)
				break;
			if (ctx->constbuf[stage].cb[i].buffer == prsc)
				ctx->dirty_shader[stage] |= FD_DIRTY_SHADER_CONST;
		}

		/* Textures */
		for (unsigned i = 0; i < ctx->tex[stage].num_textures; i++) {
			if (ctx->dirty_shader[stage] & FD_DIRTY_SHADER_TEX)
				break;
			if (ctx->tex[stage].textures[i] && (ctx->tex[stage].textures[i]->texture == prsc))
				ctx->dirty_shader[stage] |= FD_DIRTY_SHADER_TEX;
		}

		/* SSBOs */
		const unsigned num_ssbos = util_last_bit(ctx->shaderbuf[stage].enabled_mask);
		for (unsigned i = 0; i < num_ssbos; i++) {
			if (ctx->dirty_shader[stage] & FD_DIRTY_SHADER_SSBO)
				break;
			if (ctx->shaderbuf[stage].sb[i].buffer == prsc)
				ctx->dirty_shader[stage] |= FD_DIRTY_SHADER_SSBO;
		}
	}
}

static void
realloc_bo(struct fd_resource *rsc, uint32_t size)
{
	struct fd_screen *screen = fd_screen(rsc->base.screen);
	uint32_t flags = DRM_FREEDRENO_GEM_CACHE_WCOMBINE |
			DRM_FREEDRENO_GEM_TYPE_KMEM; /* TODO */

	/* if we start using things other than write-combine,
	 * be sure to check for PIPE_RESOURCE_FLAG_MAP_COHERENT
	 */

	if (rsc->bo)
		fd_bo_del(rsc->bo);

	rsc->bo = fd_bo_new(screen->dev, size, flags);
	rsc->seqno = p_atomic_inc_return(&screen->rsc_seqno);
	util_range_set_empty(&rsc->valid_buffer_range);
	fd_bc_invalidate_resource(rsc, true);
}

static void
do_blit(struct fd_context *ctx, const struct pipe_blit_info *blit, bool fallback)
{
	/* TODO size threshold too?? */
	if (!fallback) {
		/* do blit on gpu: */
		fd_blitter_pipe_begin(ctx, false, true, FD_STAGE_BLIT);
		ctx->blit(ctx, blit);
		fd_blitter_pipe_end(ctx);
	} else {
		/* do blit on cpu: */
		util_resource_copy_region(&ctx->base,
				blit->dst.resource, blit->dst.level, blit->dst.box.x,
				blit->dst.box.y, blit->dst.box.z,
				blit->src.resource, blit->src.level, &blit->src.box);
	}
}

static bool
fd_try_shadow_resource(struct fd_context *ctx, struct fd_resource *rsc,
		unsigned level, const struct pipe_box *box)
{
	struct pipe_context *pctx = &ctx->base;
	struct pipe_resource *prsc = &rsc->base;
	bool fallback = false;

	if (prsc->next)
		return false;

	/* TODO: somehow munge dimensions and format to copy unsupported
	 * render target format to something that is supported?
	 */
	if (!pctx->screen->is_format_supported(pctx->screen,
			prsc->format, prsc->target, prsc->nr_samples,
			prsc->nr_storage_samples,
			PIPE_BIND_RENDER_TARGET))
		fallback = true;

	/* do shadowing back-blits on the cpu for buffers: */
	if (prsc->target == PIPE_BUFFER)
		fallback = true;

	bool whole_level = util_texrange_covers_whole_level(prsc, level,
		box->x, box->y, box->z, box->width, box->height, box->depth);

	/* TODO need to be more clever about current level */
	if ((prsc->target >= PIPE_TEXTURE_2D) && !whole_level)
		return false;

	struct pipe_resource *pshadow =
		pctx->screen->resource_create(pctx->screen, prsc);

	if (!pshadow)
		return false;

	assert(!ctx->in_shadow);
	ctx->in_shadow = true;

	/* get rid of any references that batch-cache might have to us (which
	 * should empty/destroy rsc->batches hashset)
	 */
	fd_bc_invalidate_resource(rsc, false);

	mtx_lock(&ctx->screen->lock);

	/* Swap the backing bo's, so shadow becomes the old buffer,
	 * blit from shadow to new buffer.  From here on out, we
	 * cannot fail.
	 *
	 * Note that we need to do it in this order, otherwise if
	 * we go down cpu blit path, the recursive transfer_map()
	 * sees the wrong status..
	 */
	struct fd_resource *shadow = fd_resource(pshadow);

	DBG("shadow: %p (%d) -> %p (%d)\n", rsc, rsc->base.reference.count,
			shadow, shadow->base.reference.count);

	/* TODO valid_buffer_range?? */
	swap(rsc->bo,        shadow->bo);
	swap(rsc->write_batch,   shadow->write_batch);
	rsc->seqno = p_atomic_inc_return(&ctx->screen->rsc_seqno);

	/* at this point, the newly created shadow buffer is not referenced
	 * by any batches, but the existing rsc (probably) is.  We need to
	 * transfer those references over:
	 */
	debug_assert(shadow->batch_mask == 0);
	struct fd_batch *batch;
	foreach_batch(batch, &ctx->screen->batch_cache, rsc->batch_mask) {
		struct set_entry *entry = _mesa_set_search(batch->resources, rsc);
		_mesa_set_remove(batch->resources, entry);
		_mesa_set_add(batch->resources, shadow);
	}
	swap(rsc->batch_mask, shadow->batch_mask);

	mtx_unlock(&ctx->screen->lock);

	struct pipe_blit_info blit = {};
	blit.dst.resource = prsc;
	blit.dst.format   = prsc->format;
	blit.src.resource = pshadow;
	blit.src.format   = pshadow->format;
	blit.mask = util_format_get_mask(prsc->format);
	blit.filter = PIPE_TEX_FILTER_NEAREST;

#define set_box(field, val) do {     \
		blit.dst.field = (val);      \
		blit.src.field = (val);      \
	} while (0)

	/* blit the other levels in their entirety: */
	for (unsigned l = 0; l <= prsc->last_level; l++) {
		if (l == level)
			continue;

		/* just blit whole level: */
		set_box(level, l);
		set_box(box.width,  u_minify(prsc->width0, l));
		set_box(box.height, u_minify(prsc->height0, l));
		set_box(box.depth,  u_minify(prsc->depth0, l));

		do_blit(ctx, &blit, fallback);
	}

	/* deal w/ current level specially, since we might need to split
	 * it up into a couple blits:
	 */
	if (!whole_level) {
		set_box(level, level);

		switch (prsc->target) {
		case PIPE_BUFFER:
		case PIPE_TEXTURE_1D:
			set_box(box.y, 0);
			set_box(box.z, 0);
			set_box(box.height, 1);
			set_box(box.depth, 1);

			if (box->x > 0) {
				set_box(box.x, 0);
				set_box(box.width, box->x);

				do_blit(ctx, &blit, fallback);
			}
			if ((box->x + box->width) < u_minify(prsc->width0, level)) {
				set_box(box.x, box->x + box->width);
				set_box(box.width, u_minify(prsc->width0, level) - (box->x + box->width));

				do_blit(ctx, &blit, fallback);
			}
			break;
		case PIPE_TEXTURE_2D:
			/* TODO */
		default:
			unreachable("TODO");
		}
	}

	ctx->in_shadow = false;

	pipe_resource_reference(&pshadow, NULL);

	return true;
}

static struct fd_resource *
fd_alloc_staging(struct fd_context *ctx, struct fd_resource *rsc,
		unsigned level, const struct pipe_box *box)
{
	struct pipe_context *pctx = &ctx->base;
	struct pipe_resource tmpl = rsc->base;

	tmpl.width0  = box->width;
	tmpl.height0 = box->height;
	tmpl.depth0  = box->depth;
	tmpl.array_size = 1;
	tmpl.last_level = 0;
	tmpl.bind |= PIPE_BIND_LINEAR;

	struct pipe_resource *pstaging =
		pctx->screen->resource_create(pctx->screen, &tmpl);
	if (!pstaging)
		return NULL;

	return fd_resource(pstaging);
}

static void
fd_blit_from_staging(struct fd_context *ctx, struct fd_transfer *trans)
{
	struct pipe_resource *dst = trans->base.resource;
	struct pipe_blit_info blit = {};

	blit.dst.resource = dst;
	blit.dst.format   = dst->format;
	blit.dst.level    = trans->base.level;
	blit.dst.box      = trans->base.box;
	blit.src.resource = trans->staging_prsc;
	blit.src.format   = trans->staging_prsc->format;
	blit.src.level    = 0;
	blit.src.box      = trans->staging_box;
	blit.mask = util_format_get_mask(trans->staging_prsc->format);
	blit.filter = PIPE_TEX_FILTER_NEAREST;

	do_blit(ctx, &blit, false);
}

static void
fd_blit_to_staging(struct fd_context *ctx, struct fd_transfer *trans)
{
	struct pipe_resource *src = trans->base.resource;
	struct pipe_blit_info blit = {};

	blit.src.resource = src;
	blit.src.format   = src->format;
	blit.src.level    = trans->base.level;
	blit.src.box      = trans->base.box;
	blit.dst.resource = trans->staging_prsc;
	blit.dst.format   = trans->staging_prsc->format;
	blit.dst.level    = 0;
	blit.dst.box      = trans->staging_box;
	blit.mask = util_format_get_mask(trans->staging_prsc->format);
	blit.filter = PIPE_TEX_FILTER_NEAREST;

	do_blit(ctx, &blit, false);
}

static unsigned
fd_resource_layer_offset(struct fd_resource *rsc,
						 struct fd_resource_slice *slice,
						 unsigned layer)
{
	if (rsc->layer_first)
		return layer * rsc->layer_size;
	else
		return layer * slice->size0;
}

static void fd_resource_transfer_flush_region(struct pipe_context *pctx,
		struct pipe_transfer *ptrans,
		const struct pipe_box *box)
{
	struct fd_resource *rsc = fd_resource(ptrans->resource);

	if (ptrans->resource->target == PIPE_BUFFER)
		util_range_add(&rsc->valid_buffer_range,
					   ptrans->box.x + box->x,
					   ptrans->box.x + box->x + box->width);
}

static void
flush_resource(struct fd_context *ctx, struct fd_resource *rsc, unsigned usage)
{
	struct fd_batch *write_batch = NULL;

	fd_batch_reference(&write_batch, rsc->write_batch);

	if (usage & PIPE_TRANSFER_WRITE) {
		struct fd_batch *batch, *batches[32] = {};
		uint32_t batch_mask;

		/* This is a bit awkward, probably a fd_batch_flush_locked()
		 * would make things simpler.. but we need to hold the lock
		 * to iterate the batches which reference this resource.  So
		 * we must first grab references under a lock, then flush.
		 */
		mtx_lock(&ctx->screen->lock);
		batch_mask = rsc->batch_mask;
		foreach_batch(batch, &ctx->screen->batch_cache, batch_mask)
			fd_batch_reference(&batches[batch->idx], batch);
		mtx_unlock(&ctx->screen->lock);

		foreach_batch(batch, &ctx->screen->batch_cache, batch_mask)
			fd_batch_flush(batch, false, false);

		foreach_batch(batch, &ctx->screen->batch_cache, batch_mask) {
			fd_batch_sync(batch);
			fd_batch_reference(&batches[batch->idx], NULL);
		}
		assert(rsc->batch_mask == 0);
	} else if (write_batch) {
		fd_batch_flush(write_batch, true, false);
	}

	fd_batch_reference(&write_batch, NULL);

	assert(!rsc->write_batch);
}

static void
fd_flush_resource(struct pipe_context *pctx, struct pipe_resource *prsc)
{
	flush_resource(fd_context(pctx), fd_resource(prsc), PIPE_TRANSFER_READ);
}

static void
fd_resource_transfer_unmap(struct pipe_context *pctx,
		struct pipe_transfer *ptrans)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_resource *rsc = fd_resource(ptrans->resource);
	struct fd_transfer *trans = fd_transfer(ptrans);

	if (trans->staging_prsc) {
		if (ptrans->usage & PIPE_TRANSFER_WRITE)
			fd_blit_from_staging(ctx, trans);
		pipe_resource_reference(&trans->staging_prsc, NULL);
	}

	if (!(ptrans->usage & PIPE_TRANSFER_UNSYNCHRONIZED)) {
		fd_bo_cpu_fini(rsc->bo);
	}

	util_range_add(&rsc->valid_buffer_range,
				   ptrans->box.x,
				   ptrans->box.x + ptrans->box.width);

	pipe_resource_reference(&ptrans->resource, NULL);
	slab_free(&ctx->transfer_pool, ptrans);
}

static void *
fd_resource_transfer_map(struct pipe_context *pctx,
		struct pipe_resource *prsc,
		unsigned level, unsigned usage,
		const struct pipe_box *box,
		struct pipe_transfer **pptrans)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_resource *rsc = fd_resource(prsc);
	struct fd_resource_slice *slice = fd_resource_slice(rsc, level);
	struct fd_transfer *trans;
	struct pipe_transfer *ptrans;
	enum pipe_format format = prsc->format;
	uint32_t op = 0;
	uint32_t offset;
	char *buf;
	int ret = 0;

	DBG("prsc=%p, level=%u, usage=%x, box=%dx%d+%d,%d", prsc, level, usage,
		box->width, box->height, box->x, box->y);

	ptrans = slab_alloc(&ctx->transfer_pool);
	if (!ptrans)
		return NULL;

	/* slab_alloc_st() doesn't zero: */
	trans = fd_transfer(ptrans);
	memset(trans, 0, sizeof(*trans));

	pipe_resource_reference(&ptrans->resource, prsc);
	ptrans->level = level;
	ptrans->usage = usage;
	ptrans->box = *box;
	ptrans->stride = util_format_get_nblocksx(format, slice->pitch) * rsc->cpp;
	ptrans->layer_stride = rsc->layer_first ? rsc->layer_size : slice->size0;

	/* we always need a staging texture for tiled buffers:
	 *
	 * TODO we might sometimes want to *also* shadow the resource to avoid
	 * splitting a batch.. for ex, mid-frame texture uploads to a tiled
	 * texture.
	 */
	if (rsc->tile_mode) {
		struct fd_resource *staging_rsc;

		staging_rsc = fd_alloc_staging(ctx, rsc, level, box);
		if (staging_rsc) {
			// TODO for PIPE_TRANSFER_READ, need to do untiling blit..
			trans->staging_prsc = &staging_rsc->base;
			trans->base.stride = util_format_get_nblocksx(format,
				staging_rsc->slices[0].pitch) * staging_rsc->cpp;
			trans->base.layer_stride = staging_rsc->layer_first ?
				staging_rsc->layer_size : staging_rsc->slices[0].size0;
			trans->staging_box = *box;
			trans->staging_box.x = 0;
			trans->staging_box.y = 0;
			trans->staging_box.z = 0;

			if (usage & PIPE_TRANSFER_READ) {
				fd_blit_to_staging(ctx, trans);
				fd_bo_cpu_prep(rsc->bo, ctx->pipe, DRM_FREEDRENO_PREP_READ);
			}

			buf = fd_bo_map(staging_rsc->bo);
			offset = 0;

			*pptrans = ptrans;

			ctx->stats.staging_uploads++;

			return buf;
		}
	}

	if (ctx->in_shadow && !(usage & PIPE_TRANSFER_READ))
		usage |= PIPE_TRANSFER_UNSYNCHRONIZED;

	if (usage & PIPE_TRANSFER_READ)
		op |= DRM_FREEDRENO_PREP_READ;

	if (usage & PIPE_TRANSFER_WRITE)
		op |= DRM_FREEDRENO_PREP_WRITE;

	if (usage & PIPE_TRANSFER_DISCARD_WHOLE_RESOURCE) {
		realloc_bo(rsc, fd_bo_size(rsc->bo));
		rebind_resource(ctx, prsc);
	} else if ((usage & PIPE_TRANSFER_WRITE) &&
			   prsc->target == PIPE_BUFFER &&
			   !util_ranges_intersect(&rsc->valid_buffer_range,
									  box->x, box->x + box->width)) {
		/* We are trying to write to a previously uninitialized range. No need
		 * to wait.
		 */
	} else if (!(usage & PIPE_TRANSFER_UNSYNCHRONIZED)) {
		struct fd_batch *write_batch = NULL;

		/* hold a reference, so it doesn't disappear under us: */
		fd_batch_reference(&write_batch, rsc->write_batch);

		if ((usage & PIPE_TRANSFER_WRITE) && write_batch &&
				write_batch->back_blit) {
			/* if only thing pending is a back-blit, we can discard it: */
			fd_batch_reset(write_batch);
		}

		/* If the GPU is writing to the resource, or if it is reading from the
		 * resource and we're trying to write to it, flush the renders.
		 */
		bool needs_flush = pending(rsc, !!(usage & PIPE_TRANSFER_WRITE));
		bool busy = needs_flush || (0 != fd_bo_cpu_prep(rsc->bo,
				ctx->pipe, op | DRM_FREEDRENO_PREP_NOSYNC));

		/* if we need to flush/stall, see if we can make a shadow buffer
		 * to avoid this:
		 *
		 * TODO we could go down this path !reorder && !busy_for_read
		 * ie. we only *don't* want to go down this path if the blit
		 * will trigger a flush!
		 */
		if (ctx->screen->reorder && busy && !(usage & PIPE_TRANSFER_READ) &&
				(usage & PIPE_TRANSFER_DISCARD_RANGE)) {
			/* try shadowing only if it avoids a flush, otherwise staging would
			 * be better:
			 */
			if (needs_flush && fd_try_shadow_resource(ctx, rsc, level, box)) {
				needs_flush = busy = false;
				rebind_resource(ctx, prsc);
				ctx->stats.shadow_uploads++;
			} else {
				struct fd_resource *staging_rsc;

				if (needs_flush) {
					flush_resource(ctx, rsc, usage);
					needs_flush = false;
				}

				/* in this case, we don't need to shadow the whole resource,
				 * since any draw that references the previous contents has
				 * already had rendering flushed for all tiles.  So we can
				 * use a staging buffer to do the upload.
				 */
				staging_rsc = fd_alloc_staging(ctx, rsc, level, box);
				if (staging_rsc) {
					trans->staging_prsc = &staging_rsc->base;
					trans->base.stride = util_format_get_nblocksx(format,
						staging_rsc->slices[0].pitch) * staging_rsc->cpp;
					trans->base.layer_stride = staging_rsc->layer_first ?
						staging_rsc->layer_size : staging_rsc->slices[0].size0;
					trans->staging_box = *box;
					trans->staging_box.x = 0;
					trans->staging_box.y = 0;
					trans->staging_box.z = 0;
					buf = fd_bo_map(staging_rsc->bo);
					offset = 0;

					*pptrans = ptrans;

					fd_batch_reference(&write_batch, NULL);

					ctx->stats.staging_uploads++;

					return buf;
				}
			}
		}

		if (needs_flush) {
			flush_resource(ctx, rsc, usage);
			needs_flush = false;
		}

		fd_batch_reference(&write_batch, NULL);

		/* The GPU keeps track of how the various bo's are being used, and
		 * will wait if necessary for the proper operation to have
		 * completed.
		 */
		if (busy) {
			ret = fd_bo_cpu_prep(rsc->bo, ctx->pipe, op);
			if (ret)
				goto fail;
		}
	}

	buf = fd_bo_map(rsc->bo);
	offset = slice->offset +
		box->y / util_format_get_blockheight(format) * ptrans->stride +
		box->x / util_format_get_blockwidth(format) * rsc->cpp +
		fd_resource_layer_offset(rsc, slice, box->z);

	if (usage & PIPE_TRANSFER_WRITE)
		rsc->valid = true;

	*pptrans = ptrans;

	return buf + offset;

fail:
	fd_resource_transfer_unmap(pctx, ptrans);
	return NULL;
}

static void
fd_resource_destroy(struct pipe_screen *pscreen,
		struct pipe_resource *prsc)
{
	struct fd_resource *rsc = fd_resource(prsc);
	fd_bc_invalidate_resource(rsc, true);
	if (rsc->bo)
		fd_bo_del(rsc->bo);
	util_range_destroy(&rsc->valid_buffer_range);
	FREE(rsc);
}

static boolean
fd_resource_get_handle(struct pipe_screen *pscreen,
		struct pipe_context *pctx,
		struct pipe_resource *prsc,
		struct winsys_handle *handle,
		unsigned usage)
{
	struct fd_resource *rsc = fd_resource(prsc);

	return fd_screen_bo_get_handle(pscreen, rsc->bo,
			rsc->slices[0].pitch * rsc->cpp, handle);
}

static uint32_t
setup_slices(struct fd_resource *rsc, uint32_t alignment, enum pipe_format format)
{
	struct pipe_resource *prsc = &rsc->base;
	struct fd_screen *screen = fd_screen(prsc->screen);
	enum util_format_layout layout = util_format_description(format)->layout;
	uint32_t pitchalign = screen->gmem_alignw;
	uint32_t level, size = 0;
	uint32_t width = prsc->width0;
	uint32_t height = prsc->height0;
	uint32_t depth = prsc->depth0;
	/* in layer_first layout, the level (slice) contains just one
	 * layer (since in fact the layer contains the slices)
	 */
	uint32_t layers_in_level = rsc->layer_first ? 1 : prsc->array_size;

	for (level = 0; level <= prsc->last_level; level++) {
		struct fd_resource_slice *slice = fd_resource_slice(rsc, level);
		uint32_t blocks;

		if (layout == UTIL_FORMAT_LAYOUT_ASTC)
			slice->pitch = width =
				util_align_npot(width, pitchalign * util_format_get_blockwidth(format));
		else
			slice->pitch = width = align(width, pitchalign);
		slice->offset = size;
		blocks = util_format_get_nblocks(format, width, height);
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

		size += slice->size0 * depth * layers_in_level;

		width = u_minify(width, 1);
		height = u_minify(height, 1);
		depth = u_minify(depth, 1);
	}

	return size;
}

static uint32_t
slice_alignment(enum pipe_texture_target target)
{
	/* on a3xx, 2d array and 3d textures seem to want their
	 * layers aligned to page boundaries:
	 */
	switch (target) {
	case PIPE_TEXTURE_3D:
	case PIPE_TEXTURE_1D_ARRAY:
	case PIPE_TEXTURE_2D_ARRAY:
		return 4096;
	default:
		return 1;
	}
}

/* cross generation texture layout to plug in to screen->setup_slices()..
 * replace with generation specific one as-needed.
 *
 * TODO for a4xx probably can extract out the a4xx specific logic int
 * a small fd4_setup_slices() wrapper that sets up layer_first, and then
 * calls this.
 */
uint32_t
fd_setup_slices(struct fd_resource *rsc)
{
	uint32_t alignment;

	alignment = slice_alignment(rsc->base.target);

	struct fd_screen *screen = fd_screen(rsc->base.screen);
	if (is_a4xx(screen)) {
		switch (rsc->base.target) {
		case PIPE_TEXTURE_3D:
			rsc->layer_first = false;
			break;
		default:
			rsc->layer_first = true;
			alignment = 1;
			break;
		}
	}

	return setup_slices(rsc, alignment, rsc->base.format);
}

/* special case to resize query buf after allocated.. */
void
fd_resource_resize(struct pipe_resource *prsc, uint32_t sz)
{
	struct fd_resource *rsc = fd_resource(prsc);

	debug_assert(prsc->width0 == 0);
	debug_assert(prsc->target == PIPE_BUFFER);
	debug_assert(prsc->bind == PIPE_BIND_QUERY_BUFFER);

	prsc->width0 = sz;
	realloc_bo(rsc, fd_screen(prsc->screen)->setup_slices(rsc));
}

// TODO common helper?
static bool
has_depth(enum pipe_format format)
{
	switch (format) {
	case PIPE_FORMAT_Z16_UNORM:
	case PIPE_FORMAT_Z32_UNORM:
	case PIPE_FORMAT_Z32_FLOAT:
	case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
	case PIPE_FORMAT_Z24_UNORM_S8_UINT:
	case PIPE_FORMAT_S8_UINT_Z24_UNORM:
	case PIPE_FORMAT_Z24X8_UNORM:
	case PIPE_FORMAT_X8Z24_UNORM:
		return true;
	default:
		return false;
	}
}

/**
 * Create a new texture object, using the given template info.
 */
static struct pipe_resource *
fd_resource_create(struct pipe_screen *pscreen,
		const struct pipe_resource *tmpl)
{
	struct fd_screen *screen = fd_screen(pscreen);
	struct fd_resource *rsc = CALLOC_STRUCT(fd_resource);
	struct pipe_resource *prsc = &rsc->base;
	enum pipe_format format = tmpl->format;
	uint32_t size;

	DBG("%p: target=%d, format=%s, %ux%ux%u, array_size=%u, last_level=%u, "
			"nr_samples=%u, usage=%u, bind=%x, flags=%x", prsc,
			tmpl->target, util_format_name(format),
			tmpl->width0, tmpl->height0, tmpl->depth0,
			tmpl->array_size, tmpl->last_level, tmpl->nr_samples,
			tmpl->usage, tmpl->bind, tmpl->flags);

	if (!rsc)
		return NULL;

	*prsc = *tmpl;

#define LINEAR \
	(PIPE_BIND_SCANOUT | \
	 PIPE_BIND_LINEAR  | \
	 PIPE_BIND_DISPLAY_TARGET)

	if (screen->tile_mode &&
			(tmpl->target != PIPE_BUFFER) &&
			(tmpl->bind & PIPE_BIND_SAMPLER_VIEW) &&
			!(tmpl->bind & LINEAR)) {
		rsc->tile_mode = screen->tile_mode(tmpl);
	}

	pipe_reference_init(&prsc->reference, 1);

	prsc->screen = pscreen;

	util_range_init(&rsc->valid_buffer_range);

	rsc->internal_format = format;
	rsc->cpp = util_format_get_blocksize(format);
	prsc->nr_samples = MAX2(1, prsc->nr_samples);
	rsc->cpp *= prsc->nr_samples;

	assert(rsc->cpp);

	// XXX probably need some extra work if we hit rsc shadowing path w/ lrz..
	if ((is_a5xx(screen) || is_a6xx(screen)) &&
		 (fd_mesa_debug & FD_DBG_LRZ) && has_depth(format)) {
		const uint32_t flags = DRM_FREEDRENO_GEM_CACHE_WCOMBINE |
				DRM_FREEDRENO_GEM_TYPE_KMEM; /* TODO */
		unsigned lrz_pitch  = align(DIV_ROUND_UP(tmpl->width0, 8), 64);
		unsigned lrz_height = DIV_ROUND_UP(tmpl->height0, 8);
		unsigned size = lrz_pitch * lrz_height * 2;

		size += 0x1000; /* for GRAS_LRZ_FAST_CLEAR_BUFFER */

		rsc->lrz_height = lrz_height;
		rsc->lrz_width = lrz_pitch;
		rsc->lrz_pitch = lrz_pitch;
		rsc->lrz = fd_bo_new(screen->dev, size, flags);
	}

	size = screen->setup_slices(rsc);

	/* special case for hw-query buffer, which we need to allocate before we
	 * know the size:
	 */
	if (size == 0) {
		/* note, semi-intention == instead of & */
		debug_assert(prsc->bind == PIPE_BIND_QUERY_BUFFER);
		return prsc;
	}

	if (rsc->layer_first) {
		rsc->layer_size = align(size, 4096);
		size = rsc->layer_size * prsc->array_size;
	}

	realloc_bo(rsc, size);
	if (!rsc->bo)
		goto fail;

	return prsc;
fail:
	fd_resource_destroy(pscreen, prsc);
	return NULL;
}

/**
 * Create a texture from a winsys_handle. The handle is often created in
 * another process by first creating a pipe texture and then calling
 * resource_get_handle.
 */
static struct pipe_resource *
fd_resource_from_handle(struct pipe_screen *pscreen,
		const struct pipe_resource *tmpl,
		struct winsys_handle *handle, unsigned usage)
{
	struct fd_resource *rsc = CALLOC_STRUCT(fd_resource);
	struct fd_resource_slice *slice = &rsc->slices[0];
	struct pipe_resource *prsc = &rsc->base;
	uint32_t pitchalign = fd_screen(pscreen)->gmem_alignw;

	DBG("target=%d, format=%s, %ux%ux%u, array_size=%u, last_level=%u, "
			"nr_samples=%u, usage=%u, bind=%x, flags=%x",
			tmpl->target, util_format_name(tmpl->format),
			tmpl->width0, tmpl->height0, tmpl->depth0,
			tmpl->array_size, tmpl->last_level, tmpl->nr_samples,
			tmpl->usage, tmpl->bind, tmpl->flags);

	if (!rsc)
		return NULL;

	*prsc = *tmpl;

	pipe_reference_init(&prsc->reference, 1);

	prsc->screen = pscreen;

	util_range_init(&rsc->valid_buffer_range);

	rsc->bo = fd_screen_bo_from_handle(pscreen, handle);
	if (!rsc->bo)
		goto fail;

	prsc->nr_samples = MAX2(1, prsc->nr_samples);
	rsc->internal_format = tmpl->format;
	rsc->cpp = prsc->nr_samples * util_format_get_blocksize(tmpl->format);
	slice->pitch = handle->stride / rsc->cpp;
	slice->offset = handle->offset;
	slice->size0 = handle->stride * prsc->height0;

	if ((slice->pitch < align(prsc->width0, pitchalign)) ||
			(slice->pitch & (pitchalign - 1)))
		goto fail;

	assert(rsc->cpp);

	return prsc;

fail:
	fd_resource_destroy(pscreen, prsc);
	return NULL;
}

/**
 * _copy_region using pipe (3d engine)
 */
static bool
fd_blitter_pipe_copy_region(struct fd_context *ctx,
		struct pipe_resource *dst,
		unsigned dst_level,
		unsigned dstx, unsigned dsty, unsigned dstz,
		struct pipe_resource *src,
		unsigned src_level,
		const struct pipe_box *src_box)
{
	/* not until we allow rendertargets to be buffers */
	if (dst->target == PIPE_BUFFER || src->target == PIPE_BUFFER)
		return false;

	if (!util_blitter_is_copy_supported(ctx->blitter, dst, src))
		return false;

	/* TODO we could discard if dst box covers dst level fully.. */
	fd_blitter_pipe_begin(ctx, false, false, FD_STAGE_BLIT);
	util_blitter_copy_texture(ctx->blitter,
			dst, dst_level, dstx, dsty, dstz,
			src, src_level, src_box);
	fd_blitter_pipe_end(ctx);

	return true;
}

/**
 * Copy a block of pixels from one resource to another.
 * The resource must be of the same format.
 * Resources with nr_samples > 1 are not allowed.
 */
static void
fd_resource_copy_region(struct pipe_context *pctx,
		struct pipe_resource *dst,
		unsigned dst_level,
		unsigned dstx, unsigned dsty, unsigned dstz,
		struct pipe_resource *src,
		unsigned src_level,
		const struct pipe_box *src_box)
{
	struct fd_context *ctx = fd_context(pctx);

	/* TODO if we have 2d core, or other DMA engine that could be used
	 * for simple copies and reasonably easily synchronized with the 3d
	 * core, this is where we'd plug it in..
	 */

	/* try blit on 3d pipe: */
	if (fd_blitter_pipe_copy_region(ctx,
			dst, dst_level, dstx, dsty, dstz,
			src, src_level, src_box))
		return;

	/* else fallback to pure sw: */
	util_resource_copy_region(pctx,
			dst, dst_level, dstx, dsty, dstz,
			src, src_level, src_box);
}

bool
fd_render_condition_check(struct pipe_context *pctx)
{
	struct fd_context *ctx = fd_context(pctx);

	if (!ctx->cond_query)
		return true;

	union pipe_query_result res = { 0 };
	bool wait =
		ctx->cond_mode != PIPE_RENDER_COND_NO_WAIT &&
		ctx->cond_mode != PIPE_RENDER_COND_BY_REGION_NO_WAIT;

	if (pctx->get_query_result(pctx, ctx->cond_query, wait, &res))
			return (bool)res.u64 != ctx->cond_cond;

	return true;
}

/**
 * Optimal hardware path for blitting pixels.
 * Scaling, format conversion, up- and downsampling (resolve) are allowed.
 */
static void
fd_blit(struct pipe_context *pctx, const struct pipe_blit_info *blit_info)
{
	struct fd_context *ctx = fd_context(pctx);
	struct pipe_blit_info info = *blit_info;
	bool discard = false;

	if (info.render_condition_enable && !fd_render_condition_check(pctx))
		return;

	if (!info.scissor_enable && !info.alpha_blend) {
		discard = util_texrange_covers_whole_level(info.dst.resource,
				info.dst.level, info.dst.box.x, info.dst.box.y,
				info.dst.box.z, info.dst.box.width,
				info.dst.box.height, info.dst.box.depth);
	}

	if (util_try_blit_via_copy_region(pctx, &info)) {
		return; /* done */
	}

	if (info.mask & PIPE_MASK_S) {
		DBG("cannot blit stencil, skipping");
		info.mask &= ~PIPE_MASK_S;
	}

	if (!util_blitter_is_blit_supported(ctx->blitter, &info)) {
		DBG("blit unsupported %s -> %s",
				util_format_short_name(info.src.resource->format),
				util_format_short_name(info.dst.resource->format));
		return;
	}

	fd_blitter_pipe_begin(ctx, info.render_condition_enable, discard, FD_STAGE_BLIT);
	ctx->blit(ctx, &info);
	fd_blitter_pipe_end(ctx);
}

void
fd_blitter_pipe_begin(struct fd_context *ctx, bool render_cond, bool discard,
		enum fd_render_stage stage)
{
	fd_fence_ref(ctx->base.screen, &ctx->last_fence, NULL);

	util_blitter_save_fragment_constant_buffer_slot(ctx->blitter,
			ctx->constbuf[PIPE_SHADER_FRAGMENT].cb);
	util_blitter_save_vertex_buffer_slot(ctx->blitter, ctx->vtx.vertexbuf.vb);
	util_blitter_save_vertex_elements(ctx->blitter, ctx->vtx.vtx);
	util_blitter_save_vertex_shader(ctx->blitter, ctx->prog.vp);
	util_blitter_save_so_targets(ctx->blitter, ctx->streamout.num_targets,
			ctx->streamout.targets);
	util_blitter_save_rasterizer(ctx->blitter, ctx->rasterizer);
	util_blitter_save_viewport(ctx->blitter, &ctx->viewport);
	util_blitter_save_scissor(ctx->blitter, &ctx->scissor);
	util_blitter_save_fragment_shader(ctx->blitter, ctx->prog.fp);
	util_blitter_save_blend(ctx->blitter, ctx->blend);
	util_blitter_save_depth_stencil_alpha(ctx->blitter, ctx->zsa);
	util_blitter_save_stencil_ref(ctx->blitter, &ctx->stencil_ref);
	util_blitter_save_sample_mask(ctx->blitter, ctx->sample_mask);
	util_blitter_save_framebuffer(ctx->blitter, &ctx->framebuffer);
	util_blitter_save_fragment_sampler_states(ctx->blitter,
			ctx->tex[PIPE_SHADER_FRAGMENT].num_samplers,
			(void **)ctx->tex[PIPE_SHADER_FRAGMENT].samplers);
	util_blitter_save_fragment_sampler_views(ctx->blitter,
			ctx->tex[PIPE_SHADER_FRAGMENT].num_textures,
			ctx->tex[PIPE_SHADER_FRAGMENT].textures);
	if (!render_cond)
		util_blitter_save_render_condition(ctx->blitter,
			ctx->cond_query, ctx->cond_cond, ctx->cond_mode);

	if (ctx->batch)
		fd_batch_set_stage(ctx->batch, stage);

	ctx->in_blit = discard;
}

void
fd_blitter_pipe_end(struct fd_context *ctx)
{
	if (ctx->batch)
		fd_batch_set_stage(ctx->batch, FD_STAGE_NULL);
	ctx->in_blit = false;
}

static void
fd_invalidate_resource(struct pipe_context *pctx, struct pipe_resource *prsc)
{
	struct fd_resource *rsc = fd_resource(prsc);

	/*
	 * TODO I guess we could track that the resource is invalidated and
	 * use that as a hint to realloc rather than stall in _transfer_map(),
	 * even in the non-DISCARD_WHOLE_RESOURCE case?
	 */

	if (rsc->write_batch) {
		struct fd_batch *batch = rsc->write_batch;
		struct pipe_framebuffer_state *pfb = &batch->framebuffer;

		if (pfb->zsbuf && pfb->zsbuf->texture == prsc)
			batch->resolve &= ~(FD_BUFFER_DEPTH | FD_BUFFER_STENCIL);

		for (unsigned i = 0; i < pfb->nr_cbufs; i++) {
			if (pfb->cbufs[i] && pfb->cbufs[i]->texture == prsc) {
				batch->resolve &= ~(PIPE_CLEAR_COLOR0 << i);
			}
		}
	}

	rsc->valid = false;
}

static enum pipe_format
fd_resource_get_internal_format(struct pipe_resource *prsc)
{
	return fd_resource(prsc)->internal_format;
}

static void
fd_resource_set_stencil(struct pipe_resource *prsc,
		struct pipe_resource *stencil)
{
	fd_resource(prsc)->stencil = fd_resource(stencil);
}

static struct pipe_resource *
fd_resource_get_stencil(struct pipe_resource *prsc)
{
	struct fd_resource *rsc = fd_resource(prsc);
	if (rsc->stencil)
		return &rsc->stencil->base;
	return NULL;
}

static const struct u_transfer_vtbl transfer_vtbl = {
		.resource_create          = fd_resource_create,
		.resource_destroy         = fd_resource_destroy,
		.transfer_map             = fd_resource_transfer_map,
		.transfer_flush_region    = fd_resource_transfer_flush_region,
		.transfer_unmap           = fd_resource_transfer_unmap,
		.get_internal_format      = fd_resource_get_internal_format,
		.set_stencil              = fd_resource_set_stencil,
		.get_stencil              = fd_resource_get_stencil,
};

void
fd_resource_screen_init(struct pipe_screen *pscreen)
{
	struct fd_screen *screen = fd_screen(pscreen);
	bool fake_rgtc = screen->gpu_id < 400;

	pscreen->resource_create = u_transfer_helper_resource_create;
	pscreen->resource_from_handle = fd_resource_from_handle;
	pscreen->resource_get_handle = fd_resource_get_handle;
	pscreen->resource_destroy = u_transfer_helper_resource_destroy;

	pscreen->transfer_helper = u_transfer_helper_create(&transfer_vtbl,
			true, false, fake_rgtc, true);

	if (!screen->setup_slices)
		screen->setup_slices = fd_setup_slices;
}

void
fd_resource_context_init(struct pipe_context *pctx)
{
	pctx->transfer_map = u_transfer_helper_transfer_map;
	pctx->transfer_flush_region = u_transfer_helper_transfer_flush_region;
	pctx->transfer_unmap = u_transfer_helper_transfer_unmap;
	pctx->buffer_subdata = u_default_buffer_subdata;
	pctx->texture_subdata = u_default_texture_subdata;
	pctx->create_surface = fd_create_surface;
	pctx->surface_destroy = fd_surface_destroy;
	pctx->resource_copy_region = fd_resource_copy_region;
	pctx->blit = fd_blit;
	pctx->flush_resource = fd_flush_resource;
	pctx->invalidate_resource = fd_invalidate_resource;
}
