/*
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
 *
 * Authors:
 *      Adam Rak <adam.rak@streamnovation.com>
 */
#ifndef EVERGREEN_COMPUTE_INTERNAL_H
#define EVERGREEN_COMPUTE_INTERNAL_H

#include "ac_binary.h"
#include "r600_asm.h"
#ifdef HAVE_OPENCL
#include <llvm-c/Core.h>
#endif

struct r600_pipe_compute {
	struct r600_context *ctx;

	struct ac_shader_binary binary;

	enum pipe_shader_ir ir_type;

	/* tgsi selector */
	struct r600_pipe_shader_selector *sel;

	struct r600_resource *code_bo;
	struct r600_bytecode bc;

	unsigned local_size;
	unsigned private_size;
	unsigned input_size;
	struct r600_resource *kernel_param;

#ifdef HAVE_OPENCL
	LLVMContextRef llvm_ctx;
#endif
};

struct r600_resource* r600_compute_buffer_alloc_vram(struct r600_screen *screen, unsigned size);

#endif
