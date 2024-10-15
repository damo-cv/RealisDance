/*
 * Copyright 2017 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef SI_COMPUTE_H
#define SI_COMPUTE_H

#include "util/u_inlines.h"

#include "si_shader.h"

#define MAX_GLOBAL_BUFFERS 32

struct si_compute {
	struct pipe_reference reference;
	struct si_screen *screen;
	union {
		struct tgsi_token *tgsi;
		struct nir_shader *nir;
	} ir;
	struct util_queue_fence ready;
	struct si_compiler_ctx_state compiler_ctx_state;

	/* bitmasks of used descriptor slots */
	uint32_t active_const_and_shader_buffers;
	uint64_t active_samplers_and_images;

	unsigned ir_type;
	unsigned local_size;
	unsigned private_size;
	unsigned input_size;
	struct si_shader shader;

	struct pipe_resource *global_buffers[MAX_GLOBAL_BUFFERS];
	unsigned use_code_object_v2 : 1;
	unsigned uses_grid_size:1;
	unsigned uses_bindless_samplers:1;
	unsigned uses_bindless_images:1;
	bool reads_variable_block_size;
	unsigned num_cs_user_data_dwords;
};

void si_destroy_compute(struct si_compute *program);

static inline void
si_compute_reference(struct si_compute **dst, struct si_compute *src)
{
	if (pipe_reference(&(*dst)->reference, &src->reference))
		si_destroy_compute(*dst);

	*dst = src;
}

#endif /* SI_COMPUTE_H */
