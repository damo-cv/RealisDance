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

/* NOTE: see https://github.com/freedreno/freedreno/wiki/A5xx-Queries */

#include "freedreno_query_acc.h"
#include "freedreno_resource.h"

#include "fd6_context.h"
#include "fd6_emit.h"
#include "fd6_format.h"
#include "fd6_query.h"

struct PACKED fd6_query_sample {
	uint64_t start;
	uint64_t result;
	uint64_t stop;
};

#define query_sample(aq, field)                 \
	fd_resource((aq)->prsc)->bo,                \
	offsetof(struct fd6_query_sample, field),   \
	0, 0

/*
 * Occlusion Query:
 *
 * OCCLUSION_COUNTER and OCCLUSION_PREDICATE differ only in how they
 * interpret results
 */

static void
occlusion_resume(struct fd_acc_query *aq, struct fd_batch *batch)
{
	struct fd_ringbuffer *ring = batch->draw;

	OUT_PKT4(ring, REG_A6XX_RB_SAMPLE_COUNT_CONTROL, 1);
	OUT_RING(ring, A6XX_RB_SAMPLE_COUNT_CONTROL_COPY);

	OUT_PKT4(ring, REG_A6XX_RB_SAMPLE_COUNT_ADDR_LO, 2);
	OUT_RELOCW(ring, query_sample(aq, start));

	fd6_event_write(batch, ring, ZPASS_DONE, false);

	fd6_context(batch->ctx)->samples_passed_queries++;
}

static void
occlusion_pause(struct fd_acc_query *aq, struct fd_batch *batch)
{
	struct fd_ringbuffer *ring = batch->draw;

	OUT_PKT7(ring, CP_MEM_WRITE, 4);
	OUT_RELOCW(ring, query_sample(aq, stop));
	OUT_RING(ring, 0xffffffff);
	OUT_RING(ring, 0xffffffff);

	OUT_PKT7(ring, CP_WAIT_MEM_WRITES, 0);

	OUT_PKT4(ring, REG_A6XX_RB_SAMPLE_COUNT_CONTROL, 1);
	OUT_RING(ring, A6XX_RB_SAMPLE_COUNT_CONTROL_COPY);

	OUT_PKT4(ring, REG_A6XX_RB_SAMPLE_COUNT_ADDR_LO, 2);
	OUT_RELOCW(ring, query_sample(aq, stop));

	OUT_PKT7(ring, CP_EVENT_WRITE, 1);
	OUT_RING(ring, ZPASS_DONE);
	fd_reset_wfi(batch);

	OUT_PKT7(ring, CP_WAIT_REG_MEM, 6);
	OUT_RING(ring, 0x00000014);   // XXX
	OUT_RELOC(ring, query_sample(aq, stop));
	OUT_RING(ring, 0xffffffff);
	OUT_RING(ring, 0xffffffff);
	OUT_RING(ring, 0x00000010);   // XXX

	/* result += stop - start: */
	OUT_PKT7(ring, CP_MEM_TO_MEM, 9);
	OUT_RING(ring, CP_MEM_TO_MEM_0_DOUBLE |
			CP_MEM_TO_MEM_0_NEG_C);
	OUT_RELOCW(ring, query_sample(aq, result));     /* dst */
	OUT_RELOC(ring, query_sample(aq, result));      /* srcA */
	OUT_RELOC(ring, query_sample(aq, stop));        /* srcB */
	OUT_RELOC(ring, query_sample(aq, start));       /* srcC */

	fd6_context(batch->ctx)->samples_passed_queries--;
}

static void
occlusion_counter_result(struct fd_acc_query *aq, void *buf,
		union pipe_query_result *result)
{
	struct fd6_query_sample *sp = buf;
	result->u64 = sp->result;
}

static void
occlusion_predicate_result(struct fd_acc_query *aq, void *buf,
		union pipe_query_result *result)
{
	struct fd6_query_sample *sp = buf;
	result->b = !!sp->result;
}

static const struct fd_acc_sample_provider occlusion_counter = {
		.query_type = PIPE_QUERY_OCCLUSION_COUNTER,
		.active = FD_STAGE_DRAW,
		.size = sizeof(struct fd6_query_sample),
		.resume = occlusion_resume,
		.pause = occlusion_pause,
		.result = occlusion_counter_result,
};

static const struct fd_acc_sample_provider occlusion_predicate = {
		.query_type = PIPE_QUERY_OCCLUSION_PREDICATE,
		.active = FD_STAGE_DRAW,
		.size = sizeof(struct fd6_query_sample),
		.resume = occlusion_resume,
		.pause = occlusion_pause,
		.result = occlusion_predicate_result,
};

static const struct fd_acc_sample_provider occlusion_predicate_conservative = {
		.query_type = PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE,
		.active = FD_STAGE_DRAW,
		.size = sizeof(struct fd6_query_sample),
		.resume = occlusion_resume,
		.pause = occlusion_pause,
		.result = occlusion_predicate_result,
};

/*
 * Timestamp Queries:
 */

static void
timestamp_resume(struct fd_acc_query *aq, struct fd_batch *batch)
{
	struct fd_ringbuffer *ring = batch->draw;

	OUT_PKT7(ring, CP_EVENT_WRITE, 4);
	OUT_RING(ring, CP_EVENT_WRITE_0_EVENT(CACHE_FLUSH_AND_INV_EVENT) |
			CP_EVENT_WRITE_0_TIMESTAMP);
	OUT_RELOCW(ring, query_sample(aq, start));
	OUT_RING(ring, 0x00000000);

	fd_reset_wfi(batch);
}

static void
timestamp_pause(struct fd_acc_query *aq, struct fd_batch *batch)
{
	struct fd_ringbuffer *ring = batch->draw;

	OUT_PKT7(ring, CP_EVENT_WRITE, 4);
	OUT_RING(ring, CP_EVENT_WRITE_0_EVENT(CACHE_FLUSH_AND_INV_EVENT) |
			CP_EVENT_WRITE_0_TIMESTAMP);
	OUT_RELOCW(ring, query_sample(aq, stop));
	OUT_RING(ring, 0x00000000);

	fd_reset_wfi(batch);
	fd_wfi(batch, ring);

	/* result += stop - start: */
	OUT_PKT7(ring, CP_MEM_TO_MEM, 9);
	OUT_RING(ring, CP_MEM_TO_MEM_0_DOUBLE |
			CP_MEM_TO_MEM_0_NEG_C);
	OUT_RELOCW(ring, query_sample(aq, result));     /* dst */
	OUT_RELOC(ring, query_sample(aq, result));      /* srcA */
	OUT_RELOC(ring, query_sample(aq, stop));        /* srcB */
	OUT_RELOC(ring, query_sample(aq, start));       /* srcC */
}

static uint64_t
ticks_to_ns(uint32_t ts)
{
	/* This is based on the 19.2MHz always-on rbbm timer.
	 *
	 * TODO we should probably query this value from kernel..
	 */
	return ts * (1000000000 / 19200000);
}

static void
time_elapsed_accumulate_result(struct fd_acc_query *aq, void *buf,
		union pipe_query_result *result)
{
	struct fd6_query_sample *sp = buf;
	result->u64 = ticks_to_ns(sp->result);
}

static void
timestamp_accumulate_result(struct fd_acc_query *aq, void *buf,
		union pipe_query_result *result)
{
	struct fd6_query_sample *sp = buf;
	result->u64 = ticks_to_ns(sp->result);
}

static const struct fd_acc_sample_provider time_elapsed = {
		.query_type = PIPE_QUERY_TIME_ELAPSED,
		.active = FD_STAGE_DRAW | FD_STAGE_CLEAR,
		.size = sizeof(struct fd6_query_sample),
		.resume = timestamp_resume,
		.pause = timestamp_pause,
		.result = time_elapsed_accumulate_result,
};

/* NOTE: timestamp query isn't going to give terribly sensible results
 * on a tiler.  But it is needed by qapitrace profile heatmap.  If you
 * add in a binning pass, the results get even more non-sensical.  So
 * we just return the timestamp on the first tile and hope that is
 * kind of good enough.
 */

static const struct fd_acc_sample_provider timestamp = {
		.query_type = PIPE_QUERY_TIMESTAMP,
		.active = FD_STAGE_ALL,
		.size = sizeof(struct fd6_query_sample),
		.resume = timestamp_resume,
		.pause = timestamp_pause,
		.result = timestamp_accumulate_result,
};

void
fd6_query_context_init(struct pipe_context *pctx)
{
	struct fd_context *ctx = fd_context(pctx);

	ctx->create_query = fd_acc_create_query;
	ctx->query_set_stage = fd_acc_query_set_stage;

	fd_acc_query_register_provider(pctx, &occlusion_counter);
	fd_acc_query_register_provider(pctx, &occlusion_predicate);
	fd_acc_query_register_provider(pctx, &occlusion_predicate_conservative);

	fd_acc_query_register_provider(pctx, &time_elapsed);
	fd_acc_query_register_provider(pctx, &timestamp);
}
