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
#include "util/u_dual_blend.h"
#include "util/u_string.h"
#include "util/u_memory.h"
#include "util/u_helpers.h"

#include "freedreno_state.h"
#include "freedreno_context.h"
#include "freedreno_resource.h"
#include "freedreno_texture.h"
#include "freedreno_gmem.h"
#include "freedreno_query_hw.h"
#include "freedreno_util.h"

/* All the generic state handling.. In case of CSO's that are specific
 * to the GPU version, when the bind and the delete are common they can
 * go in here.
 */

static void
fd_set_blend_color(struct pipe_context *pctx,
		const struct pipe_blend_color *blend_color)
{
	struct fd_context *ctx = fd_context(pctx);
	ctx->blend_color = *blend_color;
	ctx->dirty |= FD_DIRTY_BLEND_COLOR;
}

static void
fd_set_stencil_ref(struct pipe_context *pctx,
		const struct pipe_stencil_ref *stencil_ref)
{
	struct fd_context *ctx = fd_context(pctx);
	ctx->stencil_ref =* stencil_ref;
	ctx->dirty |= FD_DIRTY_STENCIL_REF;
}

static void
fd_set_clip_state(struct pipe_context *pctx,
		const struct pipe_clip_state *clip)
{
	struct fd_context *ctx = fd_context(pctx);
	ctx->ucp = *clip;
	ctx->dirty |= FD_DIRTY_UCP;
}

static void
fd_set_sample_mask(struct pipe_context *pctx, unsigned sample_mask)
{
	struct fd_context *ctx = fd_context(pctx);
	ctx->sample_mask = (uint16_t)sample_mask;
	ctx->dirty |= FD_DIRTY_SAMPLE_MASK;
}

/* notes from calim on #dri-devel:
 * index==0 will be non-UBO (ie. glUniformXYZ()) all packed together padded
 * out to vec4's
 * I should be able to consider that I own the user_ptr until the next
 * set_constant_buffer() call, at which point I don't really care about the
 * previous values.
 * index>0 will be UBO's.. well, I'll worry about that later
 */
static void
fd_set_constant_buffer(struct pipe_context *pctx,
		enum pipe_shader_type shader, uint index,
		const struct pipe_constant_buffer *cb)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_constbuf_stateobj *so = &ctx->constbuf[shader];

	util_copy_constant_buffer(&so->cb[index], cb);

	/* Note that the state tracker can unbind constant buffers by
	 * passing NULL here.
	 */
	if (unlikely(!cb)) {
		so->enabled_mask &= ~(1 << index);
		return;
	}

	so->enabled_mask |= 1 << index;
	ctx->dirty_shader[shader] |= FD_DIRTY_SHADER_CONST;
	ctx->dirty |= FD_DIRTY_CONST;
}

static void
fd_set_shader_buffers(struct pipe_context *pctx,
		enum pipe_shader_type shader,
		unsigned start, unsigned count,
		const struct pipe_shader_buffer *buffers)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_shaderbuf_stateobj *so = &ctx->shaderbuf[shader];
	unsigned mask = 0;

	if (buffers) {
		for (unsigned i = 0; i < count; i++) {
			unsigned n = i + start;
			struct pipe_shader_buffer *buf = &so->sb[n];

			if ((buf->buffer == buffers[i].buffer) &&
					(buf->buffer_offset == buffers[i].buffer_offset) &&
					(buf->buffer_size == buffers[i].buffer_size))
				continue;

			mask |= BIT(n);

			buf->buffer_offset = buffers[i].buffer_offset;
			buf->buffer_size = buffers[i].buffer_size;
			pipe_resource_reference(&buf->buffer, buffers[i].buffer);

			if (buf->buffer)
				so->enabled_mask |= BIT(n);
			else
				so->enabled_mask &= ~BIT(n);
		}
	} else {
		mask = (BIT(count) - 1) << start;

		for (unsigned i = 0; i < count; i++) {
			unsigned n = i + start;
			struct pipe_shader_buffer *buf = &so->sb[n];

			pipe_resource_reference(&buf->buffer, NULL);
		}

		so->enabled_mask &= ~mask;
	}

	ctx->dirty_shader[shader] |= FD_DIRTY_SHADER_SSBO;
}

static void
fd_set_shader_images(struct pipe_context *pctx,
		enum pipe_shader_type shader,
		unsigned start, unsigned count,
		const struct pipe_image_view *images)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_shaderimg_stateobj *so = &ctx->shaderimg[shader];

	unsigned mask = 0;

	if (images) {
		for (unsigned i = 0; i < count; i++) {
			unsigned n = i + start;
			struct pipe_image_view *buf = &so->si[n];

			if ((buf->resource == images[i].resource) &&
					(buf->format == images[i].format) &&
					(buf->access == images[i].access) &&
					!memcmp(&buf->u, &images[i].u, sizeof(buf->u)))
				continue;

			mask |= BIT(n);
			util_copy_image_view(buf, &images[i]);

			if (buf->resource)
				so->enabled_mask |= BIT(n);
			else
				so->enabled_mask &= ~BIT(n);
		}
	} else {
		mask = (BIT(count) - 1) << start;

		for (unsigned i = 0; i < count; i++) {
			unsigned n = i + start;
			struct pipe_image_view *img = &so->si[n];

			pipe_resource_reference(&img->resource, NULL);
		}

		so->enabled_mask &= ~mask;
	}

	ctx->dirty_shader[shader] |= FD_DIRTY_SHADER_IMAGE;
}

static void
fd_set_framebuffer_state(struct pipe_context *pctx,
		const struct pipe_framebuffer_state *framebuffer)
{
	struct fd_context *ctx = fd_context(pctx);
	struct pipe_framebuffer_state *cso;

	DBG("%ux%u, %u layers, %u samples",
		framebuffer->width, framebuffer->height,
		framebuffer->layers, framebuffer->samples);

	cso = &ctx->framebuffer;

	if (util_framebuffer_state_equal(cso, framebuffer))
		return;

	util_copy_framebuffer_state(cso, framebuffer);

	cso->samples = util_framebuffer_get_num_samples(cso);

	if (ctx->screen->reorder) {
		struct fd_batch *old_batch = NULL;

		fd_batch_reference(&old_batch, ctx->batch);

		if (likely(old_batch))
			fd_batch_set_stage(old_batch, FD_STAGE_NULL);

		fd_batch_reference(&ctx->batch, NULL);
		fd_context_all_dirty(ctx);

		if (old_batch && old_batch->blit && !old_batch->back_blit) {
			/* for blits, there is not really much point in hanging on
			 * to the uncommitted batch (ie. you probably don't blit
			 * multiple times to the same surface), so we might as
			 * well go ahead and flush this one:
			 */
			fd_batch_flush(old_batch, false, false);
		}

		fd_batch_reference(&old_batch, NULL);
	} else {
		DBG("%d: cbufs[0]=%p, zsbuf=%p", ctx->batch->needs_flush,
				framebuffer->cbufs[0], framebuffer->zsbuf);
		fd_batch_flush(ctx->batch, false, false);
		util_copy_framebuffer_state(&ctx->batch->framebuffer, cso);
	}

	ctx->dirty |= FD_DIRTY_FRAMEBUFFER;

	ctx->disabled_scissor.minx = 0;
	ctx->disabled_scissor.miny = 0;
	ctx->disabled_scissor.maxx = cso->width;
	ctx->disabled_scissor.maxy = cso->height;

	ctx->dirty |= FD_DIRTY_SCISSOR;
}

static void
fd_set_polygon_stipple(struct pipe_context *pctx,
		const struct pipe_poly_stipple *stipple)
{
	struct fd_context *ctx = fd_context(pctx);
	ctx->stipple = *stipple;
	ctx->dirty |= FD_DIRTY_STIPPLE;
}

static void
fd_set_scissor_states(struct pipe_context *pctx,
		unsigned start_slot,
		unsigned num_scissors,
		const struct pipe_scissor_state *scissor)
{
	struct fd_context *ctx = fd_context(pctx);

	ctx->scissor = *scissor;
	ctx->dirty |= FD_DIRTY_SCISSOR;
}

static void
fd_set_viewport_states(struct pipe_context *pctx,
		unsigned start_slot,
		unsigned num_viewports,
		const struct pipe_viewport_state *viewport)
{
	struct fd_context *ctx = fd_context(pctx);
	ctx->viewport = *viewport;
	ctx->dirty |= FD_DIRTY_VIEWPORT;
}

static void
fd_set_vertex_buffers(struct pipe_context *pctx,
		unsigned start_slot, unsigned count,
		const struct pipe_vertex_buffer *vb)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_vertexbuf_stateobj *so = &ctx->vtx.vertexbuf;
	int i;

	/* on a2xx, pitch is encoded in the vtx fetch instruction, so
	 * we need to mark VTXSTATE as dirty as well to trigger patching
	 * and re-emitting the vtx shader:
	 */
	if (ctx->screen->gpu_id < 300) {
		for (i = 0; i < count; i++) {
			bool new_enabled = vb && vb[i].buffer.resource;
			bool old_enabled = so->vb[i].buffer.resource != NULL;
			uint32_t new_stride = vb ? vb[i].stride : 0;
			uint32_t old_stride = so->vb[i].stride;
			if ((new_enabled != old_enabled) || (new_stride != old_stride)) {
				ctx->dirty |= FD_DIRTY_VTXSTATE;
				break;
			}
		}
	}

	util_set_vertex_buffers_mask(so->vb, &so->enabled_mask, vb, start_slot, count);
	so->count = util_last_bit(so->enabled_mask);

	ctx->dirty |= FD_DIRTY_VTXBUF;
}

static void
fd_blend_state_bind(struct pipe_context *pctx, void *hwcso)
{
	struct fd_context *ctx = fd_context(pctx);
	struct pipe_blend_state *cso = hwcso;
	bool old_is_dual = ctx->blend ?
		ctx->blend->rt[0].blend_enable && util_blend_state_is_dual(ctx->blend, 0) :
		false;
	bool new_is_dual = cso ?
		cso->rt[0].blend_enable && util_blend_state_is_dual(cso, 0) :
		false;
	ctx->blend = hwcso;
	ctx->dirty |= FD_DIRTY_BLEND;
	if (old_is_dual != new_is_dual)
		ctx->dirty |= FD_DIRTY_BLEND_DUAL;
}

static void
fd_blend_state_delete(struct pipe_context *pctx, void *hwcso)
{
	FREE(hwcso);
}

static void
fd_rasterizer_state_bind(struct pipe_context *pctx, void *hwcso)
{
	struct fd_context *ctx = fd_context(pctx);
	struct pipe_scissor_state *old_scissor = fd_context_get_scissor(ctx);

	ctx->rasterizer = hwcso;
	ctx->dirty |= FD_DIRTY_RASTERIZER;

	/* if scissor enable bit changed we need to mark scissor
	 * state as dirty as well:
	 * NOTE: we can do a shallow compare, since we only care
	 * if it changed to/from &ctx->disable_scissor
	 */
	if (old_scissor != fd_context_get_scissor(ctx))
		ctx->dirty |= FD_DIRTY_SCISSOR;
}

static void
fd_rasterizer_state_delete(struct pipe_context *pctx, void *hwcso)
{
	FREE(hwcso);
}

static void
fd_zsa_state_bind(struct pipe_context *pctx, void *hwcso)
{
	struct fd_context *ctx = fd_context(pctx);
	ctx->zsa = hwcso;
	ctx->dirty |= FD_DIRTY_ZSA;
}

static void
fd_zsa_state_delete(struct pipe_context *pctx, void *hwcso)
{
	FREE(hwcso);
}

static void *
fd_vertex_state_create(struct pipe_context *pctx, unsigned num_elements,
		const struct pipe_vertex_element *elements)
{
	struct fd_vertex_stateobj *so = CALLOC_STRUCT(fd_vertex_stateobj);

	if (!so)
		return NULL;

	memcpy(so->pipe, elements, sizeof(*elements) * num_elements);
	so->num_elements = num_elements;

	return so;
}

static void
fd_vertex_state_delete(struct pipe_context *pctx, void *hwcso)
{
	FREE(hwcso);
}

static void
fd_vertex_state_bind(struct pipe_context *pctx, void *hwcso)
{
	struct fd_context *ctx = fd_context(pctx);
	ctx->vtx.vtx = hwcso;
	ctx->dirty |= FD_DIRTY_VTXSTATE;
}

static struct pipe_stream_output_target *
fd_create_stream_output_target(struct pipe_context *pctx,
		struct pipe_resource *prsc, unsigned buffer_offset,
		unsigned buffer_size)
{
	struct pipe_stream_output_target *target;
	struct fd_resource *rsc = fd_resource(prsc);

	target = CALLOC_STRUCT(pipe_stream_output_target);
	if (!target)
		return NULL;

	pipe_reference_init(&target->reference, 1);
	pipe_resource_reference(&target->buffer, prsc);

	target->context = pctx;
	target->buffer_offset = buffer_offset;
	target->buffer_size = buffer_size;

	assert(rsc->base.target == PIPE_BUFFER);
	util_range_add(&rsc->valid_buffer_range,
		buffer_offset, buffer_offset + buffer_size);

	return target;
}

static void
fd_stream_output_target_destroy(struct pipe_context *pctx,
		struct pipe_stream_output_target *target)
{
	pipe_resource_reference(&target->buffer, NULL);
	FREE(target);
}

static void
fd_set_stream_output_targets(struct pipe_context *pctx,
		unsigned num_targets, struct pipe_stream_output_target **targets,
		const unsigned *offsets)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_streamout_stateobj *so = &ctx->streamout;
	unsigned i;

	debug_assert(num_targets <= ARRAY_SIZE(so->targets));

	for (i = 0; i < num_targets; i++) {
		boolean changed = targets[i] != so->targets[i];
		boolean append = (offsets[i] == (unsigned)-1);

		if (!changed && append)
			continue;

		if (!append)
			so->offsets[i] = offsets[i];

		pipe_so_target_reference(&so->targets[i], targets[i]);
	}

	for (; i < so->num_targets; i++) {
		pipe_so_target_reference(&so->targets[i], NULL);
	}

	so->num_targets = num_targets;

	ctx->dirty |= FD_DIRTY_STREAMOUT;
}

static void
fd_bind_compute_state(struct pipe_context *pctx, void *state)
{
	struct fd_context *ctx = fd_context(pctx);
	ctx->compute = state;
	ctx->dirty_shader[PIPE_SHADER_COMPUTE] |= FD_DIRTY_SHADER_PROG;
}

static void
fd_set_compute_resources(struct pipe_context *pctx,
		unsigned start, unsigned count, struct pipe_surface **prscs)
{
	// TODO
}

/* used by clover to bind global objects, returning the bo address
 * via handles[n]
 */
static void
fd_set_global_binding(struct pipe_context *pctx,
		unsigned first, unsigned count, struct pipe_resource **prscs,
		uint32_t **handles)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_global_bindings_stateobj *so = &ctx->global_bindings;
	unsigned mask = 0;

	if (prscs) {
		for (unsigned i = 0; i < count; i++) {
			unsigned n = i + first;

			mask |= BIT(n);

			pipe_resource_reference(&so->buf[n], prscs[i]);

			if (so->buf[n]) {
				struct fd_resource *rsc = fd_resource(so->buf[n]);
				uint64_t iova = fd_bo_get_iova(rsc->bo);
				// TODO need to scream if iova > 32b or fix gallium API..
				*handles[i] += iova;
			}

			if (prscs[i])
				so->enabled_mask |= BIT(n);
			else
				so->enabled_mask &= ~BIT(n);
		}
	} else {
		mask = (BIT(count) - 1) << first;

		for (unsigned i = 0; i < count; i++) {
			unsigned n = i + first;
			if (so->buf[n]) {
				struct fd_resource *rsc = fd_resource(so->buf[n]);
				fd_bo_put_iova(rsc->bo);
			}
			pipe_resource_reference(&so->buf[n], NULL);
		}

		so->enabled_mask &= ~mask;
	}

}

void
fd_state_init(struct pipe_context *pctx)
{
	pctx->set_blend_color = fd_set_blend_color;
	pctx->set_stencil_ref = fd_set_stencil_ref;
	pctx->set_clip_state = fd_set_clip_state;
	pctx->set_sample_mask = fd_set_sample_mask;
	pctx->set_constant_buffer = fd_set_constant_buffer;
	pctx->set_shader_buffers = fd_set_shader_buffers;
	pctx->set_shader_images = fd_set_shader_images;
	pctx->set_framebuffer_state = fd_set_framebuffer_state;
	pctx->set_polygon_stipple = fd_set_polygon_stipple;
	pctx->set_scissor_states = fd_set_scissor_states;
	pctx->set_viewport_states = fd_set_viewport_states;

	pctx->set_vertex_buffers = fd_set_vertex_buffers;

	pctx->bind_blend_state = fd_blend_state_bind;
	pctx->delete_blend_state = fd_blend_state_delete;

	pctx->bind_rasterizer_state = fd_rasterizer_state_bind;
	pctx->delete_rasterizer_state = fd_rasterizer_state_delete;

	pctx->bind_depth_stencil_alpha_state = fd_zsa_state_bind;
	pctx->delete_depth_stencil_alpha_state = fd_zsa_state_delete;

	pctx->create_vertex_elements_state = fd_vertex_state_create;
	pctx->delete_vertex_elements_state = fd_vertex_state_delete;
	pctx->bind_vertex_elements_state = fd_vertex_state_bind;

	pctx->create_stream_output_target = fd_create_stream_output_target;
	pctx->stream_output_target_destroy = fd_stream_output_target_destroy;
	pctx->set_stream_output_targets = fd_set_stream_output_targets;

	if (has_compute(fd_screen(pctx->screen))) {
		pctx->bind_compute_state = fd_bind_compute_state;
		pctx->set_compute_resources = fd_set_compute_resources;
		pctx->set_global_binding = fd_set_global_binding;
	}
}
