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

#ifndef FD6_CONTEXT_H_
#define FD6_CONTEXT_H_

#include "util/u_upload_mgr.h"

#include "freedreno_context.h"

#include "ir3_shader.h"

#include "a6xx.xml.h"

struct fd6_context {
	struct fd_context base;

	struct fd_bo *vs_pvt_mem, *fs_pvt_mem;

	/* Two buffers related to hw binning / visibility stream (VSC).
	 * Compared to previous generations
	 *   (1) we cannot specify individual buffers per VSC, instead
	 *       just a pitch and base address
	 *   (2) there is a second smaller buffer, for something.. we
	 *       also stash VSC_BIN_SIZE at end of 2nd buffer.
	 */
	struct fd_bo *vsc_data, *vsc_data2;

// TODO annoyingly large sizes to prevent hangs with larger amounts
// of geometry, like aquarium with max # of fish.  Need to figure
// out how to calculate the required size.
#define A6XX_VSC_DATA_PITCH  0x4400
#define A6XX_VSC_DATA2_PITCH 0x10400

	/* TODO not sure what this is for.. probably similar to
	 * CACHE_FLUSH_TS on kernel side, where value gets written
	 * to this address synchronized w/ 3d (ie. a way to
	 * synchronize when the CP is running far ahead)
	 */
	struct fd_bo *blit_mem;
	uint32_t seqno;

	struct u_upload_mgr *border_color_uploader;
	struct pipe_resource *border_color_buf;

	/* if *any* of bits are set in {v,f}saturate_{s,t,r} */
	bool vsaturate, fsaturate;

	/* bitmask of sampler which needs coords clamped for vertex
	 * shader:
	 */
	uint16_t vsaturate_s, vsaturate_t, vsaturate_r;

	/* bitmask of sampler which needs coords clamped for frag
	 * shader:
	 */
	uint16_t fsaturate_s, fsaturate_t, fsaturate_r;

	/* bitmask of samplers which need astc srgb workaround: */
	uint16_t vastc_srgb, fastc_srgb;

	/* some state changes require a different shader variant.  Keep
	 * track of this so we know when we need to re-emit shader state
	 * due to variant change.  See fixup_shader_state()
	 */
	struct ir3_shader_key last_key;

	/* number of active samples-passed queries: */
	int samples_passed_queries;

	/* maps per-shader-stage state plus variant key to hw
	 * program stateobj:
	 */
	struct ir3_cache *shader_cache;

	/* cached stateobjs to avoid hashtable lookup when not dirty: */
	const struct fd6_program_state *prog;

	uint16_t tex_seqno;
	struct hash_table *tex_cache;
};

static inline struct fd6_context *
fd6_context(struct fd_context *ctx)
{
	return (struct fd6_context *)ctx;
}

struct pipe_context *
fd6_context_create(struct pipe_screen *pscreen, void *priv, unsigned flags);

static inline void
emit_marker6(struct fd_ringbuffer *ring, int scratch_idx)
{
	extern unsigned marker_cnt;
	unsigned reg = REG_A6XX_CP_SCRATCH_REG(scratch_idx);
	OUT_PKT4(ring, reg, 1);
	OUT_RING(ring, ++marker_cnt);
}

#endif /* FD6_CONTEXT_H_ */
