/*
 * Copyright 2018 Advanced Micro Devices, Inc.
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
 *
 */

#include "si_pipe.h"

/* Note: Compute shaders always use SI_COMPUTE_DST_CACHE_POLICY for dst
 * and L2_STREAM for src.
 */
static enum si_cache_policy get_cache_policy(struct si_context *sctx,
					     enum si_coherency coher,
					     uint64_t size)
{
	if ((sctx->chip_class >= GFX9 && (coher == SI_COHERENCY_CB_META ||
					  coher == SI_COHERENCY_CP)) ||
	    (sctx->chip_class >= CIK && coher == SI_COHERENCY_SHADER))
		return size <= 256 * 1024 ? L2_LRU : L2_STREAM;

	return L2_BYPASS;
}

unsigned si_get_flush_flags(struct si_context *sctx, enum si_coherency coher,
			    enum si_cache_policy cache_policy)
{
	switch (coher) {
	default:
	case SI_COHERENCY_NONE:
	case SI_COHERENCY_CP:
		return 0;
	case SI_COHERENCY_SHADER:
		return SI_CONTEXT_INV_SMEM_L1 |
		       SI_CONTEXT_INV_VMEM_L1 |
		       (cache_policy == L2_BYPASS ? SI_CONTEXT_INV_GLOBAL_L2 : 0);
	case SI_COHERENCY_CB_META:
		return SI_CONTEXT_FLUSH_AND_INV_CB;
	}
}

static void si_compute_do_clear_or_copy(struct si_context *sctx,
					struct pipe_resource *dst,
					unsigned dst_offset,
					struct pipe_resource *src,
					unsigned src_offset,
					unsigned size,
					const uint32_t *clear_value,
					unsigned clear_value_size,
					enum si_coherency coher)
{
	struct pipe_context *ctx = &sctx->b;

	assert(src_offset % 4 == 0);
	assert(dst_offset % 4 == 0);
	assert(size % 4 == 0);

	assert(dst->target != PIPE_BUFFER || dst_offset + size <= dst->width0);
	assert(!src || src_offset + size <= src->width0);

	sctx->flags |= SI_CONTEXT_PS_PARTIAL_FLUSH |
		       SI_CONTEXT_CS_PARTIAL_FLUSH |
		       si_get_flush_flags(sctx, coher, SI_COMPUTE_DST_CACHE_POLICY);
	si_emit_cache_flush(sctx);

	/* Save states. */
	void *saved_cs = sctx->cs_shader_state.program;
	struct pipe_shader_buffer saved_sb[2] = {};
	si_get_shader_buffers(sctx, PIPE_SHADER_COMPUTE, 0, src ? 2 : 1, saved_sb);

	/* The memory accesses are coalesced, meaning that the 1st instruction writes
	 * the 1st contiguous block of data for the whole wave, the 2nd instruction
	 * writes the 2nd contiguous block of data, etc.
	 */
	unsigned dwords_per_thread = src ? SI_COMPUTE_COPY_DW_PER_THREAD :
					   SI_COMPUTE_CLEAR_DW_PER_THREAD;
	unsigned instructions_per_thread = MAX2(1, dwords_per_thread / 4);
	unsigned dwords_per_instruction = dwords_per_thread / instructions_per_thread;
	unsigned dwords_per_wave = dwords_per_thread * 64;

	unsigned num_dwords = size / 4;
	unsigned num_instructions = DIV_ROUND_UP(num_dwords, dwords_per_instruction);

	struct pipe_grid_info info = {};
	info.block[0] = MIN2(64, num_instructions);
	info.block[1] = 1;
	info.block[2] = 1;
	info.grid[0] = DIV_ROUND_UP(num_dwords, dwords_per_wave);
	info.grid[1] = 1;
	info.grid[2] = 1;

	struct pipe_shader_buffer sb[2] = {};
	sb[0].buffer = dst;
	sb[0].buffer_offset = dst_offset;
	sb[0].buffer_size = size;

	if (src) {
		sb[1].buffer = src;
		sb[1].buffer_offset = src_offset;
		sb[1].buffer_size = size;

		ctx->set_shader_buffers(ctx, PIPE_SHADER_COMPUTE, 0, 2, sb);
		ctx->bind_compute_state(ctx, sctx->cs_copy_buffer);
	} else {
		assert(clear_value_size >= 4 &&
		       clear_value_size <= 16 &&
		       util_is_power_of_two_or_zero(clear_value_size));

		for (unsigned i = 0; i < 4; i++)
			sctx->cs_user_data[i] = clear_value[i % (clear_value_size / 4)];

		ctx->set_shader_buffers(ctx, PIPE_SHADER_COMPUTE, 0, 1, sb);
		ctx->bind_compute_state(ctx, sctx->cs_clear_buffer);
	}

	ctx->launch_grid(ctx, &info);

	enum si_cache_policy cache_policy = get_cache_policy(sctx, coher, size);
	sctx->flags |= SI_CONTEXT_CS_PARTIAL_FLUSH |
		       (cache_policy == L2_BYPASS ? SI_CONTEXT_WRITEBACK_GLOBAL_L2 : 0);

	if (cache_policy != L2_BYPASS)
		r600_resource(dst)->TC_L2_dirty = true;

	/* Restore states. */
	ctx->bind_compute_state(ctx, saved_cs);
	ctx->set_shader_buffers(ctx, PIPE_SHADER_COMPUTE, 0, src ? 2 : 1, saved_sb);
}

void si_clear_buffer(struct si_context *sctx, struct pipe_resource *dst,
		     uint64_t offset, uint64_t size, uint32_t *clear_value,
		     uint32_t clear_value_size, enum si_coherency coher)
{
	if (!size)
		return;

	unsigned clear_alignment = MIN2(clear_value_size, 4);

	assert(clear_value_size != 3 && clear_value_size != 6); /* 12 is allowed. */
	assert(offset % clear_alignment == 0);
	assert(size % clear_alignment == 0);
	assert(size < (UINT_MAX & ~0xf)); /* TODO: test 64-bit sizes in all codepaths */

	/* Reduce a large clear value size if possible. */
	if (clear_value_size > 4) {
		bool clear_dword_duplicated = true;

		/* See if we can lower large fills to dword fills. */
		for (unsigned i = 1; i < clear_value_size / 4; i++) {
			if (clear_value[0] != clear_value[i]) {
				clear_dword_duplicated = false;
				break;
			}
		}
		if (clear_dword_duplicated)
			clear_value_size = 4;
	}

	/* Expand a small clear value size. */
	uint32_t tmp_clear_value;
	if (clear_value_size <= 2) {
		if (clear_value_size == 1) {
			tmp_clear_value = *(uint8_t*)clear_value;
			tmp_clear_value |= (tmp_clear_value << 8) |
					   (tmp_clear_value << 16) |
					   (tmp_clear_value << 24);
		} else {
			tmp_clear_value = *(uint16_t*)clear_value;
			tmp_clear_value |= tmp_clear_value << 16;
		}
		clear_value = &tmp_clear_value;
		clear_value_size = 4;
	}

	/* Use transform feedback for 12-byte clears. */
	/* TODO: Use compute. */
	if (clear_value_size == 12) {
		union pipe_color_union streamout_clear_value;

		memcpy(&streamout_clear_value, clear_value, clear_value_size);
		si_blitter_begin(sctx, SI_DISABLE_RENDER_COND);
		util_blitter_clear_buffer(sctx->blitter, dst, offset,
					  size, clear_value_size / 4,
					  &streamout_clear_value);
		si_blitter_end(sctx);
		return;
	}

	uint64_t aligned_size = size & ~3ull;
	if (aligned_size >= 4) {
		/* Before GFX9, CP DMA was very slow when clearing GTT, so never
		 * use CP DMA clears on those chips, because we can't be certain
		 * about buffer placements.
		 */
		if (clear_value_size > 4 ||
		    (clear_value_size == 4 &&
		     offset % 4 == 0 &&
		     (size > 32*1024 || sctx->chip_class <= VI))) {
			si_compute_do_clear_or_copy(sctx, dst, offset, NULL, 0,
						    aligned_size, clear_value,
						    clear_value_size, coher);
		} else {
			assert(clear_value_size == 4);
			si_cp_dma_clear_buffer(sctx, dst, offset,
					       aligned_size, *clear_value, coher,
					       get_cache_policy(sctx, coher, size));
		}

		offset += aligned_size;
		size -= aligned_size;
	}

	/* Handle non-dword alignment. */
	if (size) {
		assert(dst);
		assert(dst->target == PIPE_BUFFER);
		assert(size < 4);

		pipe_buffer_write(&sctx->b, dst, offset, size, clear_value);
	}
}

static void si_pipe_clear_buffer(struct pipe_context *ctx,
				 struct pipe_resource *dst,
				 unsigned offset, unsigned size,
				 const void *clear_value,
				 int clear_value_size)
{
	enum si_coherency coher;

	if (dst->flags & SI_RESOURCE_FLAG_SO_FILLED_SIZE)
		coher = SI_COHERENCY_CP;
	else
		coher = SI_COHERENCY_SHADER;

	si_clear_buffer((struct si_context*)ctx, dst, offset, size, (uint32_t*)clear_value,
			clear_value_size, coher);
}

void si_copy_buffer(struct si_context *sctx,
		    struct pipe_resource *dst, struct pipe_resource *src,
		    uint64_t dst_offset, uint64_t src_offset, unsigned size)
{
	if (!size)
		return;

	enum si_coherency coher = SI_COHERENCY_SHADER;
	enum si_cache_policy cache_policy = get_cache_policy(sctx, coher, size);

	/* Only use compute for VRAM copies on dGPUs. */
	if (sctx->screen->info.has_dedicated_vram &&
	    r600_resource(dst)->domains & RADEON_DOMAIN_VRAM &&
	    r600_resource(src)->domains & RADEON_DOMAIN_VRAM &&
	    size > 32 * 1024 &&
	    dst_offset % 4 == 0 && src_offset % 4 == 0 && size % 4 == 0) {
		si_compute_do_clear_or_copy(sctx, dst, dst_offset, src, src_offset,
					    size, NULL, 0, coher);
	} else {
		si_cp_dma_copy_buffer(sctx, dst, src, dst_offset, src_offset, size,
				      0, coher, cache_policy);
	}
}

void si_init_compute_blit_functions(struct si_context *sctx)
{
	sctx->b.clear_buffer = si_pipe_clear_buffer;
}
