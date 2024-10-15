/*
 * Copyright (C) 2016 Rob Clark <robclark@freedesktop.org>
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

#ifndef FD6_TEXTURE_H_
#define FD6_TEXTURE_H_

#include "pipe/p_context.h"

#include "freedreno_texture.h"
#include "freedreno_resource.h"

#include "fd6_context.h"
#include "fd6_format.h"

struct fd6_sampler_stateobj {
	struct pipe_sampler_state base;
	uint32_t texsamp0, texsamp1, texsamp2, texsamp3;
	bool saturate_s, saturate_t, saturate_r;
	bool needs_border;
	uint16_t seqno;
};

static inline struct fd6_sampler_stateobj *
fd6_sampler_stateobj(struct pipe_sampler_state *samp)
{
	return (struct fd6_sampler_stateobj *)samp;
}

struct fd6_pipe_sampler_view {
	struct pipe_sampler_view base;
	uint32_t texconst0, texconst1, texconst2, texconst3, texconst5;
	uint32_t texconst6, texconst7, texconst8, texconst9, texconst10, texconst11;
	uint32_t offset;
	bool astc_srgb;
	uint16_t seqno;
};

static inline struct fd6_pipe_sampler_view *
fd6_pipe_sampler_view(struct pipe_sampler_view *pview)
{
	return (struct fd6_pipe_sampler_view *)pview;
}

void fd6_texture_init(struct pipe_context *pctx);
void fd6_texture_fini(struct pipe_context *pctx);

static inline enum a6xx_tex_type
fd6_tex_type(unsigned target)
{
	switch (target) {
	default:
		assert(0);
	case PIPE_BUFFER:
	case PIPE_TEXTURE_1D:
	case PIPE_TEXTURE_1D_ARRAY:
		return A6XX_TEX_1D;
	case PIPE_TEXTURE_RECT:
	case PIPE_TEXTURE_2D:
	case PIPE_TEXTURE_2D_ARRAY:
		return A6XX_TEX_2D;
	case PIPE_TEXTURE_3D:
		return A6XX_TEX_3D;
	case PIPE_TEXTURE_CUBE:
	case PIPE_TEXTURE_CUBE_ARRAY:
		return A6XX_TEX_CUBE;
	}
}

/*
 * Texture stateobj:
 *
 * The sampler and sampler-view state is mapped to a single hardware
 * stateobj which can be emit'd as a pointer in a CP_SET_DRAW_STATE
 * packet, to avoid the overhead of re-generating the entire cmdstream
 * when application toggles thru multiple different texture states.
 */

struct fd6_texture_key {
	struct {
		/* We need to track the seqno of the rsc as well as of the
		 * sampler view, because resource shadowing/etc can result
		 * that the underlying bo changes (which means the previous
		 * state was no longer valid.
		 */
		uint16_t rsc_seqno;
		uint16_t seqno;
	} view[16];
	struct {
		uint16_t seqno;
	} samp[16];
	uint8_t bcolor_offset;
};

struct fd6_texture_state {
	struct fd6_texture_key key;
	struct fd_ringbuffer *stateobj;
	bool needs_border;
};

struct fd6_texture_state * fd6_texture_state(struct fd_context *ctx,
		enum a6xx_state_block sb, struct fd_texture_stateobj *tex);

#endif /* FD6_TEXTURE_H_ */
