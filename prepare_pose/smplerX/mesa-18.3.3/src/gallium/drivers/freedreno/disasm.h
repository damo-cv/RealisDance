/*
 * Copyright © 2012 Rob Clark <robclark@freedesktop.org>
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
 */

#ifndef DISASM_H_
#define DISASM_H_

#include <stdio.h>
#include <stdbool.h>

#include "util/u_debug.h"

enum fd_shader_debug {
	FD_DBG_SHADER_VS = 0x01,
	FD_DBG_SHADER_FS = 0x02,
	FD_DBG_SHADER_CS = 0x04,
};

extern enum fd_shader_debug fd_shader_debug;

enum shader_t {
	SHADER_VERTEX,
	SHADER_TCS,
	SHADER_TES,
	SHADER_GEOM,
	SHADER_FRAGMENT,
	SHADER_COMPUTE,
	SHADER_MAX,
};

static inline bool
shader_debug_enabled(enum shader_t type)
{
	switch (type) {
	case SHADER_VERTEX:      return !!(fd_shader_debug & FD_DBG_SHADER_VS);
	case SHADER_FRAGMENT:    return !!(fd_shader_debug & FD_DBG_SHADER_FS);
	case SHADER_COMPUTE:     return !!(fd_shader_debug & FD_DBG_SHADER_CS);
	default:
		debug_assert(0);
		return false;
	}
}

static inline const char *
shader_stage_name(enum shader_t type)
{
	/* NOTE these names are chosen to match the INTEL_DEBUG output
	 * which frameretrace parses.  Hurray accidental ABI!
	 */
	switch (type) {
	case SHADER_VERTEX:      return "vertex";
	case SHADER_TCS:         return "tessellation control";
	case SHADER_TES:         return "tessellation evaluation";
	case SHADER_GEOM:        return "geometry";
	case SHADER_FRAGMENT:    return "fragment";
	case SHADER_COMPUTE:     return "compute";
	default:
		debug_assert(0);
		return NULL;
	}
}

/* bitmask of debug flags */
enum debug_t {
	PRINT_RAW      = 0x1,    /* dump raw hexdump */
	PRINT_VERBOSE  = 0x2,
};

int disasm_a2xx(uint32_t *dwords, int sizedwords, int level, enum shader_t type);
int disasm_a3xx(uint32_t *dwords, int sizedwords, int level, FILE *out);
void disasm_set_debug(enum debug_t debug);

#endif /* DISASM_H_ */
