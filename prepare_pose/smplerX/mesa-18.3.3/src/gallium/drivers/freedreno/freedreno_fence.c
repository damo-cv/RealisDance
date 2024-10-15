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

#include <libsync.h>

#include "util/u_inlines.h"

#include "freedreno_fence.h"
#include "freedreno_context.h"
#include "freedreno_util.h"

struct pipe_fence_handle {
	struct pipe_reference reference;
	/* fence holds a weak reference to the batch until the batch is flushed,
	 * at which point fd_fence_populate() is called and timestamp and possibly
	 * fence_fd become valid and the week reference is dropped.
	 */
	struct fd_batch *batch;
	struct fd_pipe *pipe;
	struct fd_screen *screen;
	int fence_fd;
	uint32_t timestamp;
};

static void fence_flush(struct pipe_fence_handle *fence)
{
	if (fence->batch)
		fd_batch_flush(fence->batch, true, true);
	debug_assert(!fence->batch);
}

void fd_fence_populate(struct pipe_fence_handle *fence,
		uint32_t timestamp, int fence_fd)
{
	if (!fence->batch)
		return;
	fence->timestamp = timestamp;
	fence->fence_fd = fence_fd;
	fence->batch = NULL;
}

static void fd_fence_destroy(struct pipe_fence_handle *fence)
{
	if (fence->fence_fd != -1)
		close(fence->fence_fd);
	fd_pipe_del(fence->pipe);
	FREE(fence);
}

void fd_fence_ref(struct pipe_screen *pscreen,
		struct pipe_fence_handle **ptr,
		struct pipe_fence_handle *pfence)
{
	if (pipe_reference(&(*ptr)->reference, &pfence->reference))
		fd_fence_destroy(*ptr);

	*ptr = pfence;
}

boolean fd_fence_finish(struct pipe_screen *pscreen,
		struct pipe_context *ctx,
		struct pipe_fence_handle *fence,
		uint64_t timeout)
{
	fence_flush(fence);

	if (fence->fence_fd != -1) {
		int ret = sync_wait(fence->fence_fd, timeout / 1000000);
		return ret == 0;
	}

	if (fd_pipe_wait_timeout(fence->pipe, fence->timestamp, timeout))
		return false;

	return true;
}

static struct pipe_fence_handle * fence_create(struct fd_context *ctx,
		struct fd_batch *batch, uint32_t timestamp, int fence_fd)
{
	struct pipe_fence_handle *fence;

	fence = CALLOC_STRUCT(pipe_fence_handle);
	if (!fence)
		return NULL;

	pipe_reference_init(&fence->reference, 1);

	fence->batch = batch;
	fence->pipe = fd_pipe_ref(ctx->pipe);
	fence->screen = ctx->screen;
	fence->timestamp = timestamp;
	fence->fence_fd = fence_fd;

	return fence;
}

void fd_create_fence_fd(struct pipe_context *pctx,
		struct pipe_fence_handle **pfence, int fd,
		enum pipe_fd_type type)
{
	assert(type == PIPE_FD_TYPE_NATIVE_SYNC);
	*pfence = fence_create(fd_context(pctx), NULL, 0, dup(fd));
}

void fd_fence_server_sync(struct pipe_context *pctx,
		struct pipe_fence_handle *fence)
{
	struct fd_context *ctx = fd_context(pctx);
	struct fd_batch *batch = fd_context_batch(ctx);

	fence_flush(fence);

	/* if not an external fence, then nothing more to do without preemption: */
	if (fence->fence_fd == -1)
		return;

	if (sync_accumulate("freedreno", &batch->in_fence_fd, fence->fence_fd)) {
		/* error */
	}
}

int fd_fence_get_fd(struct pipe_screen *pscreen,
		struct pipe_fence_handle *fence)
{
	fence_flush(fence);
	return dup(fence->fence_fd);
}

struct pipe_fence_handle * fd_fence_create(struct fd_batch *batch)
{
	return fence_create(batch->ctx, batch, 0, -1);
}
