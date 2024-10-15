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

#ifndef FD6_EMIT_H
#define FD6_EMIT_H

#include "pipe/p_context.h"

#include "freedreno_context.h"
#include "fd6_context.h"
#include "fd6_format.h"
#include "fd6_program.h"
#include "ir3_shader.h"

struct fd_ringbuffer;

/* To collect all the state objects to emit in a single CP_SET_DRAW_STATE
 * packet, the emit tracks a collection of however many state_group's that
 * need to be emit'd.
 */
enum fd6_state_id {
	FD6_GROUP_PROG,
	FD6_GROUP_PROG_BINNING,
	FD6_GROUP_LRZ,
	FD6_GROUP_LRZ_BINNING,
	FD6_GROUP_VBO,
	FD6_GROUP_VBO_BINNING,
	FD6_GROUP_VS_CONST,
	FD6_GROUP_FS_CONST,
	FD6_GROUP_VS_TEX,
	FD6_GROUP_FS_TEX,
	FD6_GROUP_RASTERIZER,
	FD6_GROUP_ZSA,
};

struct fd6_state_group {
	struct fd_ringbuffer *stateobj;
	enum fd6_state_id group_id;
	uint8_t enable_mask;
};

/* grouped together emit-state for prog/vertex/state emit: */
struct fd6_emit {
	struct fd_context *ctx;
	const struct fd_vertex_state *vtx;
	const struct pipe_draw_info *info;
	struct ir3_cache_key key;
	enum fd_dirty_3d_state dirty;

	uint32_t sprite_coord_enable;  /* bitmask */
	bool sprite_coord_mode;
	bool rasterflat;
	bool no_decode_srgb;

	/* in binning pass, we don't have real frag shader, so we
	 * don't know if real draw disqualifies lrz write.  So just
	 * figure that out up-front and stash it in the emit.
	 */
	bool no_lrz_write;

	/* cached to avoid repeated lookups: */
	const struct fd6_program_state *prog;

	struct ir3_shader_variant *bs;
	struct ir3_shader_variant *vs;
	struct ir3_shader_variant *fs;

	unsigned streamout_mask;

	struct fd6_state_group groups[32];
	unsigned num_groups;
};

static inline const struct fd6_program_state *
fd6_emit_get_prog(struct fd6_emit *emit)
{
	if (!emit->prog) {
		struct fd6_context *fd6_ctx = fd6_context(emit->ctx);
		struct ir3_program_state *s =
				ir3_cache_lookup(fd6_ctx->shader_cache, &emit->key, &emit->ctx->debug);
		emit->prog = fd6_program_state(s);
	}
	return emit->prog;
}

static inline void
fd6_emit_add_group(struct fd6_emit *emit, struct fd_ringbuffer *stateobj,
		enum fd6_state_id group_id, unsigned enable_mask)
{
	debug_assert(emit->num_groups < ARRAY_SIZE(emit->groups));
	struct fd6_state_group *g = &emit->groups[emit->num_groups++];
	g->stateobj = fd_ringbuffer_ref(stateobj);
	g->group_id = group_id;
	g->enable_mask = enable_mask;
}

static inline void
fd6_event_write(struct fd_batch *batch, struct fd_ringbuffer *ring,
		enum vgt_event_type evt, bool timestamp)
{
	fd_reset_wfi(batch);

	OUT_PKT7(ring, CP_EVENT_WRITE, timestamp ? 4 : 1);
	OUT_RING(ring, CP_EVENT_WRITE_0_EVENT(evt));
	if (timestamp) {
		struct fd6_context *fd6_ctx = fd6_context(batch->ctx);
		OUT_RELOCW(ring, fd6_ctx->blit_mem, 0, 0, 0);  /* ADDR_LO/HI */
		OUT_RING(ring, ++fd6_ctx->seqno);
	}
}

static inline void
fd6_cache_flush(struct fd_batch *batch, struct fd_ringbuffer *ring)
{
	fd6_event_write(batch, ring, 0x31, false);
}

static inline void
fd6_emit_blit(struct fd_batch *batch, struct fd_ringbuffer *ring)
{
	emit_marker6(ring, 7);
	fd6_event_write(batch, ring, BLIT, false);
	emit_marker6(ring, 7);
}

static inline void
fd6_emit_lrz_flush(struct fd_ringbuffer *ring)
{
	OUT_PKT7(ring, CP_EVENT_WRITE, 1);
	OUT_RING(ring, LRZ_FLUSH);
}

static inline enum a6xx_state_block
fd6_stage2shadersb(enum shader_t type)
{
	switch (type) {
	case SHADER_VERTEX:
		return SB6_VS_SHADER;
	case SHADER_FRAGMENT:
		return SB6_FS_SHADER;
	case SHADER_COMPUTE:
		return SB6_CS_SHADER;
	default:
		unreachable("bad shader type");
		return ~0;
	}
}

bool fd6_emit_textures(struct fd_pipe *pipe, struct fd_ringbuffer *ring,
		enum a6xx_state_block sb, struct fd_texture_stateobj *tex,
		unsigned bcolor_offset);

void fd6_emit_state(struct fd_ringbuffer *ring, struct fd6_emit *emit);

void fd6_emit_cs_state(struct fd_context *ctx, struct fd_ringbuffer *ring,
		struct ir3_shader_variant *cp);

void fd6_emit_restore(struct fd_batch *batch, struct fd_ringbuffer *ring);

void fd6_emit_init(struct pipe_context *pctx);

static inline void
fd6_emit_ib(struct fd_ringbuffer *ring, struct fd_ringbuffer *target)
{
	emit_marker6(ring, 6);
	__OUT_IB5(ring, target);
	emit_marker6(ring, 6);
}

#define WRITE(reg, val) do {					\
		OUT_PKT4(ring, reg, 1);					\
		OUT_RING(ring, val);					\
	} while (0)


#endif /* FD6_EMIT_H */
