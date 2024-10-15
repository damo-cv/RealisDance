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

#include "pipe/p_state.h"
#include "util/u_draw.h"
#include "util/u_string.h"
#include "util/u_memory.h"
#include "util/u_prim.h"
#include "util/u_format.h"
#include "util/u_helpers.h"

#include "freedreno_draw.h"
#include "freedreno_context.h"
#include "freedreno_fence.h"
#include "freedreno_state.h"
#include "freedreno_resource.h"
#include "freedreno_query_acc.h"
#include "freedreno_query_hw.h"
#include "freedreno_util.h"

static void
resource_read(struct fd_batch *batch, struct pipe_resource *prsc)
{
	if (!prsc)
		return;
	fd_batch_resource_used(batch, fd_resource(prsc), false);
}

static void
resource_written(struct fd_batch *batch, struct pipe_resource *prsc)
{
	if (!prsc)
		return;
	fd_batch_resource_used(batch, fd_resource(prsc), true);
}

static void
fd_draw_vbo(struct pipe_context *pctx, const struct pipe_draw_info *info)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_batch *batch = fd_context_batch(ctx);
	struct pipe_framebuffer_state *pfb = &batch->framebuffer;
	struct pipe_scissor_state *scissor = fd_context_get_scissor(ctx);
	unsigned i, prims, buffers = 0, restore_buffers = 0;

	/* for debugging problems with indirect draw, it is convenient
	 * to be able to emulate it, to determine if game is feeding us
	 * bogus data:
	 */
	if (info->indirect && (fd_mesa_debug & FD_DBG_NOINDR)) {
		util_draw_indirect(pctx, info);
		return;
	}

	if (!info->count_from_stream_output && !info->indirect &&
	    !info->primitive_restart &&
	    !u_trim_pipe_prim(info->mode, (unsigned*)&info->count))
		return;

	/* if we supported transform feedback, we'd have to disable this: */
	if (((scissor->maxx - scissor->minx) *
			(scissor->maxy - scissor->miny)) == 0) {
		return;
	}

	/* TODO: push down the region versions into the tiles */
	if (!fd_render_condition_check(pctx))
		return;

	/* emulate unsupported primitives: */
	if (!fd_supported_prim(ctx, info->mode)) {
		if (ctx->streamout.num_targets > 0)
			debug_error("stream-out with emulated prims");
		util_primconvert_save_rasterizer_state(ctx->primconvert, ctx->rasterizer);
		util_primconvert_draw_vbo(ctx->primconvert, info);
		return;
	}

	fd_fence_ref(pctx->screen, &ctx->last_fence, NULL);

	/* Upload a user index buffer. */
	struct pipe_resource *indexbuf = NULL;
	unsigned index_offset = 0;
	struct pipe_draw_info new_info;
	if (info->index_size) {
		if (info->has_user_indices) {
			if (!util_upload_index_buffer(pctx, info, &indexbuf, &index_offset))
				return;
			new_info = *info;
			new_info.index.resource = indexbuf;
			new_info.has_user_indices = false;
			info = &new_info;
		} else {
			indexbuf = info->index.resource;
		}
	}

	if (ctx->in_blit) {
		fd_batch_reset(batch);
		fd_context_all_dirty(ctx);
	}

	batch->blit = ctx->in_blit;
	batch->back_blit = ctx->in_shadow;

	/* NOTE: needs to be before resource_written(batch->query_buf), otherwise
	 * query_buf may not be created yet.
	 */
	fd_batch_set_stage(batch, FD_STAGE_DRAW);

	/*
	 * Figure out the buffers/features we need:
	 */

	mtx_lock(&ctx->screen->lock);

	if (ctx->dirty & (FD_DIRTY_FRAMEBUFFER | FD_DIRTY_ZSA)) {
		if (fd_depth_enabled(ctx)) {
			if (fd_resource(pfb->zsbuf->texture)->valid) {
				restore_buffers |= FD_BUFFER_DEPTH;
			} else {
				batch->invalidated |= FD_BUFFER_DEPTH;
			}
			buffers |= FD_BUFFER_DEPTH;
			resource_written(batch, pfb->zsbuf->texture);
			batch->gmem_reason |= FD_GMEM_DEPTH_ENABLED;
		}

		if (fd_stencil_enabled(ctx)) {
			if (fd_resource(pfb->zsbuf->texture)->valid) {
				restore_buffers |= FD_BUFFER_STENCIL;
			} else {
				batch->invalidated |= FD_BUFFER_STENCIL;
			}
			buffers |= FD_BUFFER_STENCIL;
			resource_written(batch, pfb->zsbuf->texture);
			batch->gmem_reason |= FD_GMEM_STENCIL_ENABLED;
		}
	}

	if (ctx->dirty & FD_DIRTY_FRAMEBUFFER) {
		for (i = 0; i < pfb->nr_cbufs; i++) {
			if (!pfb->cbufs[i])
				continue;

			resource_written(batch, pfb->cbufs[i]->texture);
		}
	}

	if (fd_logicop_enabled(ctx))
		batch->gmem_reason |= FD_GMEM_LOGICOP_ENABLED;

	for (i = 0; i < pfb->nr_cbufs; i++) {
		struct pipe_resource *surf;

		if (!pfb->cbufs[i])
			continue;

		surf = pfb->cbufs[i]->texture;

		if (fd_resource(surf)->valid) {
			restore_buffers |= PIPE_CLEAR_COLOR0 << i;
		} else {
			batch->invalidated |= PIPE_CLEAR_COLOR0 << i;
		}

		buffers |= PIPE_CLEAR_COLOR0 << i;

		if (fd_blend_enabled(ctx, i))
			batch->gmem_reason |= FD_GMEM_BLEND_ENABLED;
	}

	/* Mark SSBOs as being written.. we don't actually know which ones are
	 * read vs written, so just assume the worst
	 */
	if (ctx->dirty_shader[PIPE_SHADER_FRAGMENT] & FD_DIRTY_SHADER_SSBO) {
		foreach_bit(i, ctx->shaderbuf[PIPE_SHADER_FRAGMENT].enabled_mask)
				resource_written(batch, ctx->shaderbuf[PIPE_SHADER_FRAGMENT].sb[i].buffer);
	}

	if (ctx->dirty_shader[PIPE_SHADER_FRAGMENT] & FD_DIRTY_SHADER_IMAGE) {
		foreach_bit(i, ctx->shaderimg[PIPE_SHADER_FRAGMENT].enabled_mask) {
			struct pipe_image_view *img =
					&ctx->shaderimg[PIPE_SHADER_FRAGMENT].si[i];
			if (img->access & PIPE_IMAGE_ACCESS_WRITE)
				resource_written(batch, img->resource);
			else
				resource_read(batch, img->resource);
		}
	}

	if (ctx->dirty_shader[PIPE_SHADER_VERTEX] & FD_DIRTY_SHADER_CONST) {
		foreach_bit(i, ctx->constbuf[PIPE_SHADER_VERTEX].enabled_mask)
			resource_read(batch, ctx->constbuf[PIPE_SHADER_VERTEX].cb[i].buffer);
	}

	if (ctx->dirty_shader[PIPE_SHADER_FRAGMENT] & FD_DIRTY_SHADER_CONST) {
		foreach_bit(i, ctx->constbuf[PIPE_SHADER_FRAGMENT].enabled_mask)
			resource_read(batch, ctx->constbuf[PIPE_SHADER_FRAGMENT].cb[i].buffer);
	}

	/* Mark VBOs as being read */
	if (ctx->dirty & FD_DIRTY_VTXBUF) {
		foreach_bit(i, ctx->vtx.vertexbuf.enabled_mask) {
			assert(!ctx->vtx.vertexbuf.vb[i].is_user_buffer);
			resource_read(batch, ctx->vtx.vertexbuf.vb[i].buffer.resource);
		}
	}

	/* Mark index buffer as being read */
	resource_read(batch, indexbuf);

	/* Mark indirect draw buffer as being read */
	if (info->indirect)
		resource_read(batch, info->indirect->buffer);

	/* Mark textures as being read */
	if (ctx->dirty_shader[PIPE_SHADER_VERTEX] & FD_DIRTY_SHADER_TEX) {
		foreach_bit(i, ctx->tex[PIPE_SHADER_VERTEX].valid_textures)
			resource_read(batch, ctx->tex[PIPE_SHADER_VERTEX].textures[i]->texture);
	}

	if (ctx->dirty_shader[PIPE_SHADER_FRAGMENT] & FD_DIRTY_SHADER_TEX) {
		foreach_bit(i, ctx->tex[PIPE_SHADER_FRAGMENT].valid_textures)
			resource_read(batch, ctx->tex[PIPE_SHADER_FRAGMENT].textures[i]->texture);
	}

	/* Mark streamout buffers as being written.. */
	if (ctx->dirty & FD_DIRTY_STREAMOUT) {
		for (i = 0; i < ctx->streamout.num_targets; i++)
			if (ctx->streamout.targets[i])
				resource_written(batch, ctx->streamout.targets[i]->buffer);
	}

	resource_written(batch, batch->query_buf);

	list_for_each_entry(struct fd_acc_query, aq, &ctx->acc_active_queries, node)
		resource_written(batch, aq->prsc);

	mtx_unlock(&ctx->screen->lock);

	batch->num_draws++;

	prims = u_reduced_prims_for_vertices(info->mode, info->count);

	ctx->stats.draw_calls++;

	/* TODO prims_emitted should be clipped when the stream-out buffer is
	 * not large enough.  See max_tf_vtx().. probably need to move that
	 * into common code.  Although a bit more annoying since a2xx doesn't
	 * use ir3 so no common way to get at the pipe_stream_output_info
	 * which is needed for this calculation.
	 */
	if (ctx->streamout.num_targets > 0)
		ctx->stats.prims_emitted += prims;
	ctx->stats.prims_generated += prims;

	/* any buffers that haven't been cleared yet, we need to restore: */
	batch->restore |= restore_buffers & (FD_BUFFER_ALL & ~batch->invalidated);
	/* and any buffers used, need to be resolved: */
	batch->resolve |= buffers;

	DBG("%p: %x %ux%u num_draws=%u (%s/%s)", batch, buffers,
		pfb->width, pfb->height, batch->num_draws,
		util_format_short_name(pipe_surface_format(pfb->cbufs[0])),
		util_format_short_name(pipe_surface_format(pfb->zsbuf)));

	if (ctx->draw_vbo(ctx, info, index_offset))
		batch->needs_flush = true;

	for (i = 0; i < ctx->streamout.num_targets; i++)
		ctx->streamout.offsets[i] += info->count;

	if (fd_mesa_debug & FD_DBG_DDRAW)
		fd_context_all_dirty(ctx);

	fd_batch_check_size(batch);

	if (info == &new_info)
		pipe_resource_reference(&indexbuf, NULL);
}

/* Generic clear implementation (partially) using u_blitter: */
static void
fd_blitter_clear(struct pipe_context *pctx, unsigned buffers,
		const union pipe_color_union *color, double depth, unsigned stencil)
{
	struct fd_context *ctx = fd_context(pctx);
	struct pipe_framebuffer_state *pfb = &ctx->batch->framebuffer;
	struct blitter_context *blitter = ctx->blitter;

	fd_blitter_pipe_begin(ctx, false, true, FD_STAGE_CLEAR);

	util_blitter_common_clear_setup(blitter, pfb->width, pfb->height,
			buffers, NULL, NULL);

	struct pipe_stencil_ref sr = {
		.ref_value = { stencil & 0xff }
	};
	pctx->set_stencil_ref(pctx, &sr);

	struct pipe_constant_buffer cb = {
		.buffer_size = 16,
		.user_buffer = &color->ui,
	};
	pctx->set_constant_buffer(pctx, PIPE_SHADER_FRAGMENT, 0, &cb);

	if (!ctx->clear_rs_state) {
		const struct pipe_rasterizer_state tmpl = {
			.cull_face = PIPE_FACE_NONE,
			.half_pixel_center = 1,
			.bottom_edge_rule = 1,
			.flatshade = 1,
			.depth_clip_near = 1,
			.depth_clip_far = 1,
		};
		ctx->clear_rs_state = pctx->create_rasterizer_state(pctx, &tmpl);
	}
	pctx->bind_rasterizer_state(pctx, ctx->clear_rs_state);

	struct pipe_viewport_state vp = {
		.scale     = { 0.5f * pfb->width, -0.5f * pfb->height, depth },
		.translate = { 0.5f * pfb->width,  0.5f * pfb->height, 0.0f },
	};
	pctx->set_viewport_states(pctx, 0, 1, &vp);

	pctx->bind_vertex_elements_state(pctx, ctx->solid_vbuf_state.vtx);
	pctx->set_vertex_buffers(pctx, blitter->vb_slot, 1,
			&ctx->solid_vbuf_state.vertexbuf.vb[0]);
	pctx->set_stream_output_targets(pctx, 0, NULL, NULL);
	pctx->bind_vs_state(pctx, ctx->solid_prog.vp);
	pctx->bind_fs_state(pctx, ctx->solid_prog.fp);

	struct pipe_draw_info info = {
		.mode = PIPE_PRIM_MAX,    /* maps to DI_PT_RECTLIST */
		.count = 2,
		.max_index = 1,
		.instance_count = 1,
	};
	ctx->draw_vbo(ctx, &info, 0);

	util_blitter_restore_constant_buffer_state(blitter);
	util_blitter_restore_vertex_states(blitter);
	util_blitter_restore_fragment_states(blitter);
	util_blitter_restore_textures(blitter);
	util_blitter_restore_fb_state(blitter);
	util_blitter_restore_render_cond(blitter);
	util_blitter_unset_running_flag(blitter);

	fd_blitter_pipe_end(ctx);
}

static void
fd_clear(struct pipe_context *pctx, unsigned buffers,
		const union pipe_color_union *color, double depth, unsigned stencil)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_batch *batch = fd_context_batch(ctx);
	struct pipe_framebuffer_state *pfb = &batch->framebuffer;
	unsigned cleared_buffers;
	int i;

	/* TODO: push down the region versions into the tiles */
	if (!fd_render_condition_check(pctx))
		return;

	fd_fence_ref(pctx->screen, &ctx->last_fence, NULL);

	if (ctx->in_blit) {
		fd_batch_reset(batch);
		fd_context_all_dirty(ctx);
	}

	/* pctx->clear() is only for full-surface clears, so scissor is
	 * equivalent to having GL_SCISSOR_TEST disabled:
	 */
	batch->max_scissor.minx = 0;
	batch->max_scissor.miny = 0;
	batch->max_scissor.maxx = pfb->width;
	batch->max_scissor.maxy = pfb->height;

	/* for bookkeeping about which buffers have been cleared (and thus
	 * can fully or partially skip mem2gmem) we need to ignore buffers
	 * that have already had a draw, in case apps do silly things like
	 * clear after draw (ie. if you only clear the color buffer, but
	 * something like alpha-test causes side effects from the draw in
	 * the depth buffer, etc)
	 */
	cleared_buffers = buffers & (FD_BUFFER_ALL & ~batch->restore);
	batch->cleared |= cleared_buffers;
	batch->invalidated |= cleared_buffers;

	batch->resolve |= buffers;
	batch->needs_flush = true;

	mtx_lock(&ctx->screen->lock);

	if (buffers & PIPE_CLEAR_COLOR)
		for (i = 0; i < pfb->nr_cbufs; i++)
			if (buffers & (PIPE_CLEAR_COLOR0 << i))
				resource_written(batch, pfb->cbufs[i]->texture);

	if (buffers & (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL)) {
		resource_written(batch, pfb->zsbuf->texture);
		batch->gmem_reason |= FD_GMEM_CLEARS_DEPTH_STENCIL;
	}

	resource_written(batch, batch->query_buf);

	list_for_each_entry(struct fd_acc_query, aq, &ctx->acc_active_queries, node)
		resource_written(batch, aq->prsc);

	mtx_unlock(&ctx->screen->lock);

	DBG("%p: %x %ux%u depth=%f, stencil=%u (%s/%s)", batch, buffers,
		pfb->width, pfb->height, depth, stencil,
		util_format_short_name(pipe_surface_format(pfb->cbufs[0])),
		util_format_short_name(pipe_surface_format(pfb->zsbuf)));

	/* if per-gen backend doesn't implement ctx->clear() generic
	 * blitter clear:
	 */
	bool fallback = true;

	if (ctx->clear) {
		fd_batch_set_stage(batch, FD_STAGE_CLEAR);

		if (ctx->clear(ctx, buffers, color, depth, stencil)) {
			if (fd_mesa_debug & FD_DBG_DCLEAR)
				fd_context_all_dirty(ctx);

			fallback = false;
		}
	}

	if (fallback) {
		fd_blitter_clear(pctx, buffers, color, depth, stencil);
	}

	fd_batch_check_size(batch);
}

static void
fd_clear_render_target(struct pipe_context *pctx, struct pipe_surface *ps,
		const union pipe_color_union *color,
		unsigned x, unsigned y, unsigned w, unsigned h,
		bool render_condition_enabled)
{
	DBG("TODO: x=%u, y=%u, w=%u, h=%u", x, y, w, h);
}

static void
fd_clear_depth_stencil(struct pipe_context *pctx, struct pipe_surface *ps,
		unsigned buffers, double depth, unsigned stencil,
		unsigned x, unsigned y, unsigned w, unsigned h,
		bool render_condition_enabled)
{
	DBG("TODO: buffers=%u, depth=%f, stencil=%u, x=%u, y=%u, w=%u, h=%u",
			buffers, depth, stencil, x, y, w, h);
}

static void
fd_launch_grid(struct pipe_context *pctx, const struct pipe_grid_info *info)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_batch *batch, *save_batch = NULL;
	unsigned i;

	batch = fd_bc_alloc_batch(&ctx->screen->batch_cache, ctx, true);
	fd_batch_reference(&save_batch, ctx->batch);
	fd_batch_reference(&ctx->batch, batch);
	fd_context_all_dirty(ctx);

	mtx_lock(&ctx->screen->lock);

	/* Mark SSBOs as being written.. we don't actually know which ones are
	 * read vs written, so just assume the worst
	 */
	foreach_bit(i, ctx->shaderbuf[PIPE_SHADER_COMPUTE].enabled_mask)
		resource_written(batch, ctx->shaderbuf[PIPE_SHADER_COMPUTE].sb[i].buffer);

	foreach_bit(i, ctx->shaderimg[PIPE_SHADER_COMPUTE].enabled_mask) {
		struct pipe_image_view *img =
			&ctx->shaderimg[PIPE_SHADER_COMPUTE].si[i];
		if (img->access & PIPE_IMAGE_ACCESS_WRITE)
			resource_written(batch, img->resource);
		else
			resource_read(batch, img->resource);
	}

	/* UBO's are read */
	foreach_bit(i, ctx->constbuf[PIPE_SHADER_COMPUTE].enabled_mask)
		resource_read(batch, ctx->constbuf[PIPE_SHADER_COMPUTE].cb[i].buffer);

	/* Mark textures as being read */
	foreach_bit(i, ctx->tex[PIPE_SHADER_COMPUTE].valid_textures)
		resource_read(batch, ctx->tex[PIPE_SHADER_COMPUTE].textures[i]->texture);

	/* For global buffers, we don't really know if read or written, so assume
	 * the worst:
	 */
	foreach_bit(i, ctx->global_bindings.enabled_mask)
		resource_written(batch, ctx->global_bindings.buf[i]);

	if (info->indirect)
		resource_read(batch, info->indirect);

	mtx_unlock(&ctx->screen->lock);

	batch->needs_flush = true;
	ctx->launch_grid(ctx, info);

	fd_batch_flush(batch, false, false);

	fd_batch_reference(&ctx->batch, save_batch);
	fd_context_all_dirty(ctx);
	fd_batch_reference(&save_batch, NULL);
	fd_batch_reference(&batch, NULL);
}

void
fd_draw_init(struct pipe_context *pctx)
{
	pctx->draw_vbo = fd_draw_vbo;
	pctx->clear = fd_clear;
	pctx->clear_render_target = fd_clear_render_target;
	pctx->clear_depth_stencil = fd_clear_depth_stencil;

	if (has_compute(fd_screen(pctx->screen))) {
		pctx->launch_grid = fd_launch_grid;
	}
}
