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

#ifndef FD6_PROGRAM_H_
#define FD6_PROGRAM_H_

#include "pipe/p_context.h"
#include "freedreno_context.h"
#include "ir3_shader.h"
#include "ir3_cache.h"

struct fd6_streamout_state {
	uint32_t ncomp[PIPE_MAX_SO_BUFFERS];
	uint32_t prog[256/2];
	uint32_t prog_count;
	uint32_t vpc_so_buf_cntl;
};

struct fd6_emit;

struct fd6_program_state {
	struct ir3_program_state base;
	struct ir3_shader_variant *bs;     /* binning pass vs */
	struct ir3_shader_variant *vs;
	struct ir3_shader_variant *fs;
	struct fd_ringbuffer *binning_stateobj;
	struct fd_ringbuffer *stateobj;

	/* cached state about current emitted shader program (3d): */
	struct fd6_streamout_state tf;

	/* index and # of varyings: */
	uint8_t fs_inputs[16];
	uint8_t fs_inputs_count;

	uint32_t vinterp[8];
};

static inline struct fd6_program_state *
fd6_program_state(struct ir3_program_state *state)
{
	return (struct fd6_program_state *)state;
}

void fd6_emit_shader(struct fd_ringbuffer *ring, const struct ir3_shader_variant *so);

void fd6_program_emit(struct fd_ringbuffer *ring, struct fd6_emit *emit);

void fd6_prog_init(struct pipe_context *pctx);

#endif /* FD6_PROGRAM_H_ */
