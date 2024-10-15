/*
 * Copyright (C) 2017 Rob Clark <robclark@freedesktop.org>
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

#include "pipe/p_state.h"

#include "freedreno_resource.h"
#include "fd6_image.h"
#include "fd6_format.h"
#include "fd6_texture.h"

static enum a6xx_state_block texsb[] = {
	[PIPE_SHADER_COMPUTE] = SB6_CS_TEX,
	[PIPE_SHADER_FRAGMENT] = SB6_FS_TEX,
};

static enum a6xx_state_block imgsb[] = {
	[PIPE_SHADER_COMPUTE] = SB6_CS_SSBO,
	[PIPE_SHADER_FRAGMENT] = SB6_SSBO,
};

struct fd6_image {
	enum pipe_format pfmt;
	enum a6xx_tex_fmt fmt;
	enum a6xx_tex_fetchsize fetchsize;
	enum a6xx_tex_type type;
	bool srgb;
	uint32_t cpp;
	uint32_t width;
	uint32_t height;
	uint32_t depth;
	uint32_t pitch;
	uint32_t array_pitch;
	struct fd_bo *bo;
	uint32_t offset;
};

static void translate_image(struct fd6_image *img, struct pipe_image_view *pimg)
{
	enum pipe_format format = pimg->format;
	struct pipe_resource *prsc = pimg->resource;
	struct fd_resource *rsc = fd_resource(prsc);
	unsigned lvl;

	if (!pimg->resource) {
		memset(img, 0, sizeof(*img));
		return;
	}

	img->pfmt      = format;
	img->fmt       = fd6_pipe2tex(format);
	img->fetchsize = fd6_pipe2fetchsize(format);
	img->type      = fd6_tex_type(prsc->target);
	img->srgb      = util_format_is_srgb(format);
	img->cpp       = rsc->cpp;
	img->bo        = rsc->bo;

	if (prsc->target == PIPE_BUFFER) {
		lvl = 0;
		img->offset = pimg->u.buf.offset;
		img->pitch  = pimg->u.buf.size;
		img->array_pitch = 0;
	} else {
		lvl = pimg->u.tex.level;
		img->offset = rsc->slices[lvl].offset;
		img->pitch  = rsc->slices[lvl].pitch * rsc->cpp;
		img->array_pitch = rsc->layer_size;
	}

	img->width     = u_minify(prsc->width0, lvl);
	img->height    = u_minify(prsc->height0, lvl);
	img->depth     = u_minify(prsc->depth0, lvl);
}

static void emit_image_tex(struct fd_ringbuffer *ring, unsigned slot,
		struct fd6_image *img, enum pipe_shader_type shader)
{
	unsigned opcode = CP_LOAD_STATE6_FRAG;

	assert(shader == PIPE_SHADER_COMPUTE || shader == PIPE_SHADER_FRAGMENT);

	OUT_PKT7(ring, opcode, 3 + 12);
	OUT_RING(ring, CP_LOAD_STATE6_0_DST_OFF(slot) |
		CP_LOAD_STATE6_0_STATE_TYPE(ST6_CONSTANTS) |
		CP_LOAD_STATE6_0_STATE_SRC(SS6_DIRECT) |
		CP_LOAD_STATE6_0_STATE_BLOCK(texsb[shader]) |
		CP_LOAD_STATE6_0_NUM_UNIT(1));
	OUT_RING(ring, CP_LOAD_STATE6_1_EXT_SRC_ADDR(0));
	OUT_RING(ring, CP_LOAD_STATE6_2_EXT_SRC_ADDR_HI(0));

	OUT_RING(ring, A6XX_TEX_CONST_0_FMT(img->fmt) |
		fd6_tex_swiz(img->pfmt, PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y,
			PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W) |
		COND(img->srgb, A6XX_TEX_CONST_0_SRGB));
	OUT_RING(ring, A6XX_TEX_CONST_1_WIDTH(img->width) |
		A6XX_TEX_CONST_1_HEIGHT(img->height));
	OUT_RING(ring, A6XX_TEX_CONST_2_FETCHSIZE(img->fetchsize) |
		A6XX_TEX_CONST_2_TYPE(img->type) |
		A6XX_TEX_CONST_2_PITCH(img->pitch));
	OUT_RING(ring, A6XX_TEX_CONST_3_ARRAY_PITCH(img->array_pitch));
	if (img->bo) {
		OUT_RELOC(ring, img->bo, img->offset,
				(uint64_t)A6XX_TEX_CONST_5_DEPTH(img->depth) << 32, 0);
	} else {
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, A6XX_TEX_CONST_5_DEPTH(img->depth));
	}
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
	OUT_RING(ring, 0x00000000);
}

static void emit_image_ssbo(struct fd_ringbuffer *ring, unsigned slot,
		struct fd6_image *img, enum pipe_shader_type shader)
{
	unsigned opcode = CP_LOAD_STATE6_FRAG;

	assert(shader == PIPE_SHADER_COMPUTE || shader == PIPE_SHADER_FRAGMENT);

#if 0
	OUT_PKT7(ring, opcode, 3 + 4);
	OUT_RING(ring, CP_LOAD_STATE6_0_DST_OFF(slot) |
		CP_LOAD_STATE6_0_STATE_TYPE(0) |
		CP_LOAD_STATE6_0_STATE_SRC(SS6_DIRECT) |
		CP_LOAD_STATE6_0_STATE_BLOCK(imgsb[shader]) |
		CP_LOAD_STATE6_0_NUM_UNIT(1));
	OUT_RING(ring, CP_LOAD_STATE6_1_EXT_SRC_ADDR(0));
	OUT_RING(ring, CP_LOAD_STATE6_2_EXT_SRC_ADDR_HI(0));
	OUT_RING(ring, A6XX_SSBO_0_0_BASE_LO(0));
	OUT_RING(ring, A6XX_SSBO_0_1_PITCH(img->pitch));
	OUT_RING(ring, A6XX_SSBO_0_2_ARRAY_PITCH(img->array_pitch));
	OUT_RING(ring, A6XX_SSBO_0_3_CPP(img->cpp));
#endif

#if 0
	OUT_PKT7(ring, opcode, 3 + 2);
	OUT_RING(ring, CP_LOAD_STATE6_0_DST_OFF(slot) |
		CP_LOAD_STATE6_0_STATE_TYPE(1) |
		CP_LOAD_STATE6_0_STATE_SRC(SS6_DIRECT) |
		CP_LOAD_STATE6_0_STATE_BLOCK(imgsb[shader]) |
		CP_LOAD_STATE6_0_NUM_UNIT(1));
	OUT_RING(ring, CP_LOAD_STATE6_1_EXT_SRC_ADDR(0));
	OUT_RING(ring, CP_LOAD_STATE6_2_EXT_SRC_ADDR_HI(0));
	OUT_RING(ring, A6XX_SSBO_1_0_FMT(img->fmt) |
		A6XX_SSBO_1_0_WIDTH(img->width));
	OUT_RING(ring, A6XX_SSBO_1_1_HEIGHT(img->height) |
		A6XX_SSBO_1_1_DEPTH(img->depth));
#endif

	OUT_PKT7(ring, opcode, 3 + 2);
	OUT_RING(ring, CP_LOAD_STATE6_0_DST_OFF(slot) |
		CP_LOAD_STATE6_0_STATE_TYPE(2) |
		CP_LOAD_STATE6_0_STATE_SRC(SS6_DIRECT) |
		CP_LOAD_STATE6_0_STATE_BLOCK(imgsb[shader]) |
		CP_LOAD_STATE6_0_NUM_UNIT(1));
	OUT_RING(ring, CP_LOAD_STATE6_1_EXT_SRC_ADDR(0));
	OUT_RING(ring, CP_LOAD_STATE6_2_EXT_SRC_ADDR_HI(0));
	if (img->bo) {
		OUT_RELOCW(ring, img->bo, img->offset, 0, 0);
	} else {
		OUT_RING(ring, 0x00000000);
		OUT_RING(ring, 0x00000000);
	}
}

/* Note that to avoid conflicts with textures and non-image "SSBO"s, images
 * are placedd, in reverse order, at the end of the state block, so for
 * example the sampler state:
 *
 *   0:   first texture
 *   1:   second texture
 *   ....
 *   N-1: second image
 *   N:   first image
 */
static unsigned
get_image_slot(unsigned index)
{
	/* TODO figure out real limit per generation, and don't hardcode.
	 * This needs to match get_image_slot() in ir3_compiler_nir.
	 * Possibly should be factored out into shared helper?
	 */
	const unsigned max_samplers = 16;
	return max_samplers - index - 1;
}

/* Emit required "SSBO" and sampler state.  The sampler state is used by the
 * hw for imageLoad(), and "SSBO" state for imageStore().  Returns max sampler
 * used.
 */
void
fd6_emit_images(struct fd_context *ctx, struct fd_ringbuffer *ring,
		enum pipe_shader_type shader)
{
	struct fd_shaderimg_stateobj *so = &ctx->shaderimg[shader];
	unsigned enabled_mask = so->enabled_mask;

	while (enabled_mask) {
		unsigned index = u_bit_scan(&enabled_mask);
		unsigned slot = get_image_slot(index);
		struct fd6_image img;

		translate_image(&img, &so->si[index]);

		emit_image_tex(ring, slot, &img, shader);
		emit_image_ssbo(ring, slot, &img, shader);
	}
}
