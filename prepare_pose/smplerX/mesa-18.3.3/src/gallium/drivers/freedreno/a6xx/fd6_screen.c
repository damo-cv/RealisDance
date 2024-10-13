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

#include "pipe/p_screen.h"
#include "util/u_format.h"

#include "fd6_screen.h"
#include "fd6_context.h"
#include "fd6_format.h"
#include "fd6_resource.h"

#include "ir3_compiler.h"

static boolean
fd6_screen_is_format_supported(struct pipe_screen *pscreen,
		enum pipe_format format,
		enum pipe_texture_target target,
		unsigned sample_count,
		unsigned storage_sample_count,
		unsigned usage)
{
	unsigned retval = 0;

	if ((target >= PIPE_MAX_TEXTURE_TYPES) ||
			(sample_count > 1)) { /* TODO add MSAA */
		DBG("not supported: format=%s, target=%d, sample_count=%d, usage=%x",
				util_format_name(format), target, sample_count, usage);
		return FALSE;
	}

	if (MAX2(1, sample_count) != MAX2(1, storage_sample_count))
		return false;

	if ((usage & PIPE_BIND_VERTEX_BUFFER) &&
			(fd6_pipe2vtx(format) != (enum a6xx_vtx_fmt)~0)) {
		retval |= PIPE_BIND_VERTEX_BUFFER;
	}

	if ((usage & PIPE_BIND_SAMPLER_VIEW) &&
			(target == PIPE_BUFFER ||
			 util_format_get_blocksize(format) != 12) &&
			(fd6_pipe2tex(format) != (enum a6xx_tex_fmt)~0)) {
		retval |= PIPE_BIND_SAMPLER_VIEW;
	}

	if ((usage & (PIPE_BIND_RENDER_TARGET |
				PIPE_BIND_DISPLAY_TARGET |
				PIPE_BIND_SCANOUT |
				PIPE_BIND_SHARED |
				PIPE_BIND_COMPUTE_RESOURCE)) &&
			(fd6_pipe2color(format) != (enum a6xx_color_fmt)~0) &&
			(fd6_pipe2tex(format) != (enum a6xx_tex_fmt)~0)) {
		retval |= usage & (PIPE_BIND_RENDER_TARGET |
				PIPE_BIND_DISPLAY_TARGET |
				PIPE_BIND_SCANOUT |
				PIPE_BIND_SHARED |
				PIPE_BIND_COMPUTE_RESOURCE);
	}

	/* For ARB_framebuffer_no_attachments: */
	if ((usage & PIPE_BIND_RENDER_TARGET) && (format == PIPE_FORMAT_NONE)) {
		retval |= usage & PIPE_BIND_RENDER_TARGET;
	}

	if ((usage & PIPE_BIND_DEPTH_STENCIL) &&
			(fd6_pipe2depth(format) != (enum a6xx_depth_format)~0) &&
			(fd6_pipe2tex(format) != (enum a6xx_tex_fmt)~0)) {
		retval |= PIPE_BIND_DEPTH_STENCIL;
	}

	if ((usage & PIPE_BIND_INDEX_BUFFER) &&
			(fd_pipe2index(format) != (enum pc_di_index_size)~0)) {
		retval |= PIPE_BIND_INDEX_BUFFER;
	}

	if (retval != usage) {
		DBG("not supported: format=%s, target=%d, sample_count=%d, "
				"usage=%x, retval=%x", util_format_name(format),
				target, sample_count, usage, retval);
	}

	return retval == usage;
}

void
fd6_screen_init(struct pipe_screen *pscreen)
{
	struct fd_screen *screen = fd_screen(pscreen);
	screen->max_rts = A6XX_MAX_RENDER_TARGETS;
	screen->compiler = ir3_compiler_create(screen->dev, screen->gpu_id);
	pscreen->context_create = fd6_context_create;
	pscreen->is_format_supported = fd6_screen_is_format_supported;

	screen->setup_slices = fd6_setup_slices;
}
