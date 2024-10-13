/**************************************************************************
 *
 * Copyright 2013 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include <unistd.h>

#include "util/u_memory.h"
#include "util/u_video.h"

#include "vl/vl_defines.h"
#include "vl/vl_video_buffer.h"

#include "radeonsi/si_pipe.h"
#include "radeon_video.h"
#include "radeon_vce.h"

/* generate an stream handle */
unsigned si_vid_alloc_stream_handle()
{
	static unsigned counter = 0;
	unsigned stream_handle = 0;
	unsigned pid = getpid();
	int i;

	for (i = 0; i < 32; ++i)
		stream_handle |= ((pid >> i) & 1) << (31 - i);

	stream_handle ^= ++counter;
	return stream_handle;
}

/* create a buffer in the winsys */
bool si_vid_create_buffer(struct pipe_screen *screen, struct rvid_buffer *buffer,
			  unsigned size, unsigned usage)
{
	memset(buffer, 0, sizeof(*buffer));
	buffer->usage = usage;

	/* Hardware buffer placement restrictions require the kernel to be
	 * able to move buffers around individually, so request a
	 * non-sub-allocated buffer.
	 */
	buffer->res = r600_resource(pipe_buffer_create(screen, PIPE_BIND_SHARED,
						       usage, size));

	return buffer->res != NULL;
}

/* destroy a buffer */
void si_vid_destroy_buffer(struct rvid_buffer *buffer)
{
	r600_resource_reference(&buffer->res, NULL);
}

/* reallocate a buffer, preserving its content */
bool si_vid_resize_buffer(struct pipe_screen *screen, struct radeon_cmdbuf *cs,
			  struct rvid_buffer *new_buf, unsigned new_size)
{
	struct si_screen *sscreen = (struct si_screen *)screen;
	struct radeon_winsys* ws = sscreen->ws;
	unsigned bytes = MIN2(new_buf->res->buf->size, new_size);
	struct rvid_buffer old_buf = *new_buf;
	void *src = NULL, *dst = NULL;

	if (!si_vid_create_buffer(screen, new_buf, new_size, new_buf->usage))
		goto error;

	src = ws->buffer_map(old_buf.res->buf, cs, PIPE_TRANSFER_READ);
	if (!src)
		goto error;

	dst = ws->buffer_map(new_buf->res->buf, cs, PIPE_TRANSFER_WRITE);
	if (!dst)
		goto error;

	memcpy(dst, src, bytes);
	if (new_size > bytes) {
		new_size -= bytes;
		dst += bytes;
		memset(dst, 0, new_size);
	}
	ws->buffer_unmap(new_buf->res->buf);
	ws->buffer_unmap(old_buf.res->buf);
	si_vid_destroy_buffer(&old_buf);
	return true;

error:
	if (src)
		ws->buffer_unmap(old_buf.res->buf);
	si_vid_destroy_buffer(new_buf);
	*new_buf = old_buf;
	return false;
}

/* clear the buffer with zeros */
void si_vid_clear_buffer(struct pipe_context *context, struct rvid_buffer* buffer)
{
	struct si_context *sctx = (struct si_context*)context;

	si_sdma_clear_buffer(sctx, &buffer->res->b.b, 0, buffer->res->buf->size, 0);
	context->flush(context, NULL, 0);
}

/**
 * join surfaces into the same buffer with identical tiling params
 * sumup their sizes and replace the backend buffers with a single bo
 */
void si_vid_join_surfaces(struct si_context *sctx,
			  struct pb_buffer** buffers[VL_NUM_COMPONENTS],
			  struct radeon_surf *surfaces[VL_NUM_COMPONENTS])
{
	struct radeon_winsys *ws = sctx->ws;;
	unsigned best_tiling, best_wh, off;
	unsigned size, alignment;
	struct pb_buffer *pb;
	unsigned i, j;

	for (i = 0, best_tiling = 0, best_wh = ~0; i < VL_NUM_COMPONENTS; ++i) {
		unsigned wh;

		if (!surfaces[i])
			continue;

		if (sctx->chip_class < GFX9) {
			/* choose the smallest bank w/h for now */
			wh = surfaces[i]->u.legacy.bankw * surfaces[i]->u.legacy.bankh;
			if (wh < best_wh) {
				best_wh = wh;
				best_tiling = i;
			}
		}
	}

	for (i = 0, off = 0; i < VL_NUM_COMPONENTS; ++i) {
		if (!surfaces[i])
			continue;

		/* adjust the texture layer offsets */
		off = align(off, surfaces[i]->surf_alignment);

		if (sctx->chip_class < GFX9) {
			/* copy the tiling parameters */
			surfaces[i]->u.legacy.bankw = surfaces[best_tiling]->u.legacy.bankw;
			surfaces[i]->u.legacy.bankh = surfaces[best_tiling]->u.legacy.bankh;
			surfaces[i]->u.legacy.mtilea = surfaces[best_tiling]->u.legacy.mtilea;
			surfaces[i]->u.legacy.tile_split = surfaces[best_tiling]->u.legacy.tile_split;

			for (j = 0; j < ARRAY_SIZE(surfaces[i]->u.legacy.level); ++j)
				surfaces[i]->u.legacy.level[j].offset += off;
		} else {
			surfaces[i]->u.gfx9.surf_offset += off;
			for (j = 0; j < ARRAY_SIZE(surfaces[i]->u.gfx9.offset); ++j)
				surfaces[i]->u.gfx9.offset[j] += off;
		}

		off += surfaces[i]->surf_size;
	}

	for (i = 0, size = 0, alignment = 0; i < VL_NUM_COMPONENTS; ++i) {
		if (!buffers[i] || !*buffers[i])
			continue;

		size = align(size, (*buffers[i])->alignment);
		size += (*buffers[i])->size;
		alignment = MAX2(alignment, (*buffers[i])->alignment * 1);
	}

	if (!size)
		return;

	/* TODO: 2D tiling workaround */
	alignment *= 2;

	pb = ws->buffer_create(ws, size, alignment, RADEON_DOMAIN_VRAM,
			       RADEON_FLAG_GTT_WC);
	if (!pb)
		return;

	for (i = 0; i < VL_NUM_COMPONENTS; ++i) {
		if (!buffers[i] || !*buffers[i])
			continue;

		pb_reference(buffers[i], pb);
	}

	pb_reference(&pb, NULL);
}
