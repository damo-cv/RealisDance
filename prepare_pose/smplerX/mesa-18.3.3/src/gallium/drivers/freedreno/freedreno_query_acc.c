/*
 * Copyright (C) 2017 Rob Clark <robclark@freedesktop.org>
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

#include "util/u_memory.h"
#include "util/u_inlines.h"

#include "freedreno_query_acc.h"
#include "freedreno_context.h"
#include "freedreno_resource.h"
#include "freedreno_util.h"


static bool
is_active(struct fd_acc_query *aq, enum fd_render_stage stage)
{
	return !!(aq->provider->active & stage);
}

static void
fd_acc_destroy_query(struct fd_context *ctx, struct fd_query *q)
{
	struct fd_acc_query *aq = fd_acc_query(q);

	DBG("%p: active=%d", q, q->active);

	pipe_resource_reference(&aq->prsc, NULL);
	list_del(&aq->node);

	free(aq->query_data);
	free(aq);
}

static void
realloc_query_bo(struct fd_context *ctx, struct fd_acc_query *aq)
{
	struct fd_resource *rsc;
	void *map;

	pipe_resource_reference(&aq->prsc, NULL);

	aq->prsc = pipe_buffer_create(&ctx->screen->base,
			PIPE_BIND_QUERY_BUFFER, 0, 0x1000);

	/* don't assume the buffer is zero-initialized: */
	rsc = fd_resource(aq->prsc);

	fd_bo_cpu_prep(rsc->bo, ctx->pipe, DRM_FREEDRENO_PREP_WRITE);

	map = fd_bo_map(rsc->bo);
	memset(map, 0, aq->size);
	fd_bo_cpu_fini(rsc->bo);
}

static boolean
fd_acc_begin_query(struct fd_context *ctx, struct fd_query *q)
{
	struct fd_batch *batch = fd_context_batch(ctx);
	struct fd_acc_query *aq = fd_acc_query(q);
	const struct fd_acc_sample_provider *p = aq->provider;

	DBG("%p: active=%d", q, q->active);

	/* ->begin_query() discards previous results, so realloc bo: */
	realloc_query_bo(ctx, aq);

	/* then resume query if needed to collect first sample: */
	if (batch && is_active(aq, batch->stage))
		p->resume(aq, batch);

	/* add to active list: */
	assert(list_empty(&aq->node));
	list_addtail(&aq->node, &ctx->acc_active_queries);

	return true;
}

static void
fd_acc_end_query(struct fd_context *ctx, struct fd_query *q)
{
	struct fd_batch *batch = fd_context_batch(ctx);
	struct fd_acc_query *aq = fd_acc_query(q);
	const struct fd_acc_sample_provider *p = aq->provider;

	DBG("%p: active=%d", q, q->active);

	if (batch && is_active(aq, batch->stage))
		p->pause(aq, batch);

	/* remove from active list: */
	list_delinit(&aq->node);
}

static boolean
fd_acc_get_query_result(struct fd_context *ctx, struct fd_query *q,
		boolean wait, union pipe_query_result *result)
{
	struct fd_acc_query *aq = fd_acc_query(q);
	const struct fd_acc_sample_provider *p = aq->provider;
	struct fd_resource *rsc = fd_resource(aq->prsc);

	DBG("%p: wait=%d, active=%d", q, wait, q->active);

	assert(LIST_IS_EMPTY(&aq->node));

	/* if !wait, then check the last sample (the one most likely to
	 * not be ready yet) and bail if it is not ready:
	 */
	if (!wait) {
		int ret;

		if (pending(rsc, false)) {
			/* piglit spec@arb_occlusion_query@occlusion_query_conform
			 * test, and silly apps perhaps, get stuck in a loop trying
			 * to get  query result forever with wait==false..  we don't
			 * wait to flush unnecessarily but we also don't want to
			 * spin forever:
			 */
			if (aq->no_wait_cnt++ > 5)
				fd_batch_flush(rsc->write_batch, false, false);
			return false;
		}

		ret = fd_bo_cpu_prep(rsc->bo, ctx->pipe,
				DRM_FREEDRENO_PREP_READ | DRM_FREEDRENO_PREP_NOSYNC);
		if (ret)
			return false;

		fd_bo_cpu_fini(rsc->bo);
	}

	if (rsc->write_batch)
		fd_batch_flush(rsc->write_batch, true, false);

	/* get the result: */
	fd_bo_cpu_prep(rsc->bo, ctx->pipe, DRM_FREEDRENO_PREP_READ);

	void *ptr = fd_bo_map(rsc->bo);
	p->result(aq, ptr, result);
	fd_bo_cpu_fini(rsc->bo);

	return true;
}

static const struct fd_query_funcs acc_query_funcs = {
		.destroy_query    = fd_acc_destroy_query,
		.begin_query      = fd_acc_begin_query,
		.end_query        = fd_acc_end_query,
		.get_query_result = fd_acc_get_query_result,
};

struct fd_query *
fd_acc_create_query2(struct fd_context *ctx, unsigned query_type,
		const struct fd_acc_sample_provider *provider)
{
	struct fd_acc_query *aq;
	struct fd_query *q;

	aq = CALLOC_STRUCT(fd_acc_query);
	if (!aq)
		return NULL;

	DBG("%p: query_type=%u", aq, query_type);

	aq->provider = provider;
	aq->size = provider->size;

	list_inithead(&aq->node);

	q = &aq->base;
	q->funcs = &acc_query_funcs;
	q->type = query_type;

	return q;
}

struct fd_query *
fd_acc_create_query(struct fd_context *ctx, unsigned query_type)
{
	int idx = pidx(query_type);

	if ((idx < 0) || !ctx->acc_sample_providers[idx])
		return NULL;

	return fd_acc_create_query2(ctx, query_type,
			ctx->acc_sample_providers[idx]);
}

void
fd_acc_query_set_stage(struct fd_batch *batch, enum fd_render_stage stage)
{
	if (stage != batch->stage) {
		struct fd_acc_query *aq;
		LIST_FOR_EACH_ENTRY(aq, &batch->ctx->acc_active_queries, node) {
			const struct fd_acc_sample_provider *p = aq->provider;

			bool was_active = is_active(aq, batch->stage);
			bool now_active = is_active(aq, stage);

			if (now_active && !was_active)
				p->resume(aq, batch);
			else if (was_active && !now_active)
				p->pause(aq, batch);
		}
	}
}

void
fd_acc_query_register_provider(struct pipe_context *pctx,
		const struct fd_acc_sample_provider *provider)
{
	struct fd_context *ctx = fd_context(pctx);
	int idx = pidx(provider->query_type);

	assert((0 <= idx) && (idx < MAX_HW_SAMPLE_PROVIDERS));
	assert(!ctx->acc_sample_providers[idx]);

	ctx->acc_sample_providers[idx] = provider;
}
