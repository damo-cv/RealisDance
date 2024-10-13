/*
 * Copyright 2012 Advanced Micro Devices, Inc.
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

#include "si_build_pm4.h"
#include "util/u_viewport.h"
#include "tgsi/tgsi_scan.h"

#define SI_MAX_SCISSOR 16384

static void si_set_scissor_states(struct pipe_context *pctx,
				  unsigned start_slot,
				  unsigned num_scissors,
				  const struct pipe_scissor_state *state)
{
	struct si_context *ctx = (struct si_context *)pctx;
	int i;

	for (i = 0; i < num_scissors; i++)
		ctx->scissors.states[start_slot + i] = state[i];

	if (!ctx->queued.named.rasterizer ||
	    !ctx->queued.named.rasterizer->scissor_enable)
		return;

	ctx->scissors.dirty_mask |= ((1 << num_scissors) - 1) << start_slot;
	si_mark_atom_dirty(ctx, &ctx->atoms.s.scissors);
}

/* Since the guard band disables clipping, we have to clip per-pixel
 * using a scissor.
 */
static void si_get_scissor_from_viewport(struct si_context *ctx,
					 const struct pipe_viewport_state *vp,
					 struct si_signed_scissor *scissor)
{
	float tmp, minx, miny, maxx, maxy;

	/* Convert (-1, -1) and (1, 1) from clip space into window space. */
	minx = -vp->scale[0] + vp->translate[0];
	miny = -vp->scale[1] + vp->translate[1];
	maxx = vp->scale[0] + vp->translate[0];
	maxy = vp->scale[1] + vp->translate[1];

	/* Handle inverted viewports. */
	if (minx > maxx) {
		tmp = minx;
		minx = maxx;
		maxx = tmp;
	}
	if (miny > maxy) {
		tmp = miny;
		miny = maxy;
		maxy = tmp;
	}

	/* Convert to integer and round up the max bounds. */
	scissor->minx = minx;
	scissor->miny = miny;
	scissor->maxx = ceilf(maxx);
	scissor->maxy = ceilf(maxy);
}

static void si_clamp_scissor(struct si_context *ctx,
			     struct pipe_scissor_state *out,
			     struct si_signed_scissor *scissor)
{
	out->minx = CLAMP(scissor->minx, 0, SI_MAX_SCISSOR);
	out->miny = CLAMP(scissor->miny, 0, SI_MAX_SCISSOR);
	out->maxx = CLAMP(scissor->maxx, 0, SI_MAX_SCISSOR);
	out->maxy = CLAMP(scissor->maxy, 0, SI_MAX_SCISSOR);
}

static void si_clip_scissor(struct pipe_scissor_state *out,
			    struct pipe_scissor_state *clip)
{
	out->minx = MAX2(out->minx, clip->minx);
	out->miny = MAX2(out->miny, clip->miny);
	out->maxx = MIN2(out->maxx, clip->maxx);
	out->maxy = MIN2(out->maxy, clip->maxy);
}

static void si_scissor_make_union(struct si_signed_scissor *out,
				  struct si_signed_scissor *in)
{
	out->minx = MIN2(out->minx, in->minx);
	out->miny = MIN2(out->miny, in->miny);
	out->maxx = MAX2(out->maxx, in->maxx);
	out->maxy = MAX2(out->maxy, in->maxy);
	out->quant_mode = MIN2(out->quant_mode, in->quant_mode);
}

static void si_emit_one_scissor(struct si_context *ctx,
				struct radeon_cmdbuf *cs,
				struct si_signed_scissor *vp_scissor,
				struct pipe_scissor_state *scissor)
{
	struct pipe_scissor_state final;

	if (ctx->vs_disables_clipping_viewport) {
		final.minx = final.miny = 0;
		final.maxx = final.maxy = SI_MAX_SCISSOR;
	} else {
		si_clamp_scissor(ctx, &final, vp_scissor);
	}

	if (scissor)
		si_clip_scissor(&final, scissor);

	/* Workaround for a hw bug on SI that occurs when PA_SU_HARDWARE_-
	 * SCREEN_OFFSET != 0 and any_scissor.BR_X/Y <= 0.
	 */
	if (ctx->chip_class == SI && (final.maxx == 0 || final.maxy == 0)) {
		radeon_emit(cs, S_028250_TL_X(1) |
				S_028250_TL_Y(1) |
				S_028250_WINDOW_OFFSET_DISABLE(1));
		radeon_emit(cs, S_028254_BR_X(1) |
				S_028254_BR_Y(1));
		return;
	}

	radeon_emit(cs, S_028250_TL_X(final.minx) |
			S_028250_TL_Y(final.miny) |
			S_028250_WINDOW_OFFSET_DISABLE(1));
	radeon_emit(cs, S_028254_BR_X(final.maxx) |
			S_028254_BR_Y(final.maxy));
}

#define MAX_PA_SU_HARDWARE_SCREEN_OFFSET 8176

static void si_emit_guardband(struct si_context *ctx)
{
	const struct si_state_rasterizer *rs = ctx->queued.named.rasterizer;
	struct si_signed_scissor vp_as_scissor;
	struct pipe_viewport_state vp;
	float left, top, right, bottom, max_range, guardband_x, guardband_y;
	float discard_x, discard_y;

	if (ctx->vs_writes_viewport_index) {
		/* Shaders can draw to any viewport. Make a union of all
		 * viewports. */
		vp_as_scissor = ctx->viewports.as_scissor[0];
		for (unsigned i = 1; i < SI_MAX_VIEWPORTS; i++) {
			si_scissor_make_union(&vp_as_scissor,
					      &ctx->viewports.as_scissor[i]);
		}
	} else {
		vp_as_scissor = ctx->viewports.as_scissor[0];
	}

	/* Blits don't set the viewport state. The vertex shader determines
	 * the viewport size by scaling the coordinates, so we don't know
	 * how large the viewport is. Assume the worst case.
	 */
	if (ctx->vs_disables_clipping_viewport)
		vp_as_scissor.quant_mode = SI_QUANT_MODE_16_8_FIXED_POINT_1_256TH;

	/* Determine the optimal hardware screen offset to center the viewport
	 * within the viewport range in order to maximize the guardband size.
	 */
	int hw_screen_offset_x = (vp_as_scissor.maxx + vp_as_scissor.minx) / 2;
	int hw_screen_offset_y = (vp_as_scissor.maxy + vp_as_scissor.miny) / 2;

	/* SI-CI need to align the offset to an ubertile consisting of all SEs. */
	const unsigned hw_screen_offset_alignment =
		ctx->chip_class >= VI ? 16 : MAX2(ctx->screen->se_tile_repeat, 16);

	hw_screen_offset_x = CLAMP(hw_screen_offset_x, 0, MAX_PA_SU_HARDWARE_SCREEN_OFFSET);
	hw_screen_offset_y = CLAMP(hw_screen_offset_y, 0, MAX_PA_SU_HARDWARE_SCREEN_OFFSET);

	/* Align the screen offset by dropping the low bits. */
	hw_screen_offset_x &= ~(hw_screen_offset_alignment - 1);
	hw_screen_offset_y &= ~(hw_screen_offset_alignment - 1);

	/* Apply the offset to center the viewport and maximize the guardband. */
	vp_as_scissor.minx -= hw_screen_offset_x;
	vp_as_scissor.maxx -= hw_screen_offset_x;
	vp_as_scissor.miny -= hw_screen_offset_y;
	vp_as_scissor.maxy -= hw_screen_offset_y;

	/* Reconstruct the viewport transformation from the scissor. */
	vp.translate[0] = (vp_as_scissor.minx + vp_as_scissor.maxx) / 2.0;
	vp.translate[1] = (vp_as_scissor.miny + vp_as_scissor.maxy) / 2.0;
	vp.scale[0] = vp_as_scissor.maxx - vp.translate[0];
	vp.scale[1] = vp_as_scissor.maxy - vp.translate[1];

	/* Treat a 0x0 viewport as 1x1 to prevent division by zero. */
	if (vp_as_scissor.minx == vp_as_scissor.maxx)
		vp.scale[0] = 0.5;
	if (vp_as_scissor.miny == vp_as_scissor.maxy)
		vp.scale[1] = 0.5;

	/* Find the biggest guard band that is inside the supported viewport
	 * range. The guard band is specified as a horizontal and vertical
	 * distance from (0,0) in clip space.
	 *
	 * This is done by applying the inverse viewport transformation
	 * on the viewport limits to get those limits in clip space.
	 *
	 * The viewport range is [-max_viewport_size/2, max_viewport_size/2].
	 */
	static unsigned max_viewport_size[] = {65535, 16383, 4095};
	assert(vp_as_scissor.quant_mode < ARRAY_SIZE(max_viewport_size));
	max_range = max_viewport_size[vp_as_scissor.quant_mode] / 2;
	left   = (-max_range - vp.translate[0]) / vp.scale[0];
	right  = ( max_range - vp.translate[0]) / vp.scale[0];
	top    = (-max_range - vp.translate[1]) / vp.scale[1];
	bottom = ( max_range - vp.translate[1]) / vp.scale[1];

	assert(left <= -1 && top <= -1 && right >= 1 && bottom >= 1);

	guardband_x = MIN2(-left, right);
	guardband_y = MIN2(-top, bottom);

	discard_x = 1.0;
	discard_y = 1.0;

	if (unlikely(util_prim_is_points_or_lines(ctx->current_rast_prim))) {
		/* When rendering wide points or lines, we need to be more
		 * conservative about when to discard them entirely. */
		float pixels;

		if (ctx->current_rast_prim == PIPE_PRIM_POINTS)
			pixels = rs->max_point_size;
		else
			pixels = rs->line_width;

		/* Add half the point size / line width */
		discard_x += pixels / (2.0 * vp.scale[0]);
		discard_y += pixels / (2.0 * vp.scale[1]);

		/* Discard primitives that would lie entirely outside the clip
		 * region. */
		discard_x = MIN2(discard_x, guardband_x);
		discard_y = MIN2(discard_y, guardband_y);
	}

	/* If any of the GB registers is updated, all of them must be updated.
	 * R_028BE8_PA_CL_GB_VERT_CLIP_ADJ, R_028BEC_PA_CL_GB_VERT_DISC_ADJ
	 * R_028BF0_PA_CL_GB_HORZ_CLIP_ADJ, R_028BF4_PA_CL_GB_HORZ_DISC_ADJ
	 */
	unsigned initial_cdw = ctx->gfx_cs->current.cdw;
	radeon_opt_set_context_reg4(ctx, R_028BE8_PA_CL_GB_VERT_CLIP_ADJ,
				    SI_TRACKED_PA_CL_GB_VERT_CLIP_ADJ,
				    fui(guardband_y), fui(discard_y),
				    fui(guardband_x), fui(discard_x));
	radeon_opt_set_context_reg(ctx, R_028234_PA_SU_HARDWARE_SCREEN_OFFSET,
				   SI_TRACKED_PA_SU_HARDWARE_SCREEN_OFFSET,
				   S_028234_HW_SCREEN_OFFSET_X(hw_screen_offset_x >> 4) |
				   S_028234_HW_SCREEN_OFFSET_Y(hw_screen_offset_y >> 4));
	radeon_opt_set_context_reg(ctx, R_028BE4_PA_SU_VTX_CNTL,
				   SI_TRACKED_PA_SU_VTX_CNTL,
				   S_028BE4_PIX_CENTER(rs->half_pixel_center) |
				   S_028BE4_QUANT_MODE(V_028BE4_X_16_8_FIXED_POINT_1_256TH +
						       vp_as_scissor.quant_mode));
	if (initial_cdw != ctx->gfx_cs->current.cdw)
		ctx->context_roll_counter++;
}

static void si_emit_scissors(struct si_context *ctx)
{
	struct radeon_cmdbuf *cs = ctx->gfx_cs;
	struct pipe_scissor_state *states = ctx->scissors.states;
	unsigned mask = ctx->scissors.dirty_mask;
	bool scissor_enabled = ctx->queued.named.rasterizer->scissor_enable;

	/* The simple case: Only 1 viewport is active. */
	if (!ctx->vs_writes_viewport_index) {
		struct si_signed_scissor *vp = &ctx->viewports.as_scissor[0];

		if (!(mask & 1))
			return;

		radeon_set_context_reg_seq(cs, R_028250_PA_SC_VPORT_SCISSOR_0_TL, 2);
		si_emit_one_scissor(ctx, cs, vp, scissor_enabled ? &states[0] : NULL);
		ctx->scissors.dirty_mask &= ~1; /* clear one bit */
		return;
	}

	while (mask) {
		int start, count, i;

		u_bit_scan_consecutive_range(&mask, &start, &count);

		radeon_set_context_reg_seq(cs, R_028250_PA_SC_VPORT_SCISSOR_0_TL +
					       start * 4 * 2, count * 2);
		for (i = start; i < start+count; i++) {
			si_emit_one_scissor(ctx, cs, &ctx->viewports.as_scissor[i],
					    scissor_enabled ? &states[i] : NULL);
		}
	}
	ctx->scissors.dirty_mask = 0;
}

static void si_set_viewport_states(struct pipe_context *pctx,
				   unsigned start_slot,
				   unsigned num_viewports,
				   const struct pipe_viewport_state *state)
{
	struct si_context *ctx = (struct si_context *)pctx;
	unsigned mask;
	int i;

	for (i = 0; i < num_viewports; i++) {
		unsigned index = start_slot + i;
		struct si_signed_scissor *scissor = &ctx->viewports.as_scissor[index];

		ctx->viewports.states[index] = state[i];

		si_get_scissor_from_viewport(ctx, &state[i], scissor);

		unsigned w = scissor->maxx - scissor->minx;
		unsigned h = scissor->maxy - scissor->miny;
		unsigned max_extent = MAX2(w, h);

		unsigned center_x = (scissor->maxx + scissor->minx) / 2;
		unsigned center_y = (scissor->maxy + scissor->miny) / 2;
		unsigned max_center = MAX2(center_x, center_y);

		/* PA_SU_HARDWARE_SCREEN_OFFSET can't center viewports whose
		 * center start farther than MAX_PA_SU_HARDWARE_SCREEN_OFFSET.
		 * (for example, a 1x1 viewport in the lower right corner of
		 * 16Kx16K) Such viewports need a greater guardband, so they
		 * have to use a worse quantization mode.
		 */
		unsigned distance_off_center =
			MAX2(0, (int)max_center - MAX_PA_SU_HARDWARE_SCREEN_OFFSET);
		max_extent += distance_off_center;

		/* Determine the best quantization mode (subpixel precision),
		 * but also leave enough space for the guardband.
		 *
		 * Note that primitive binning requires QUANT_MODE == 16_8 on Vega10
		 * and Raven1. What we do depends on the chip:
		 * - Vega10: Never use primitive binning.
		 * - Raven1: Always use QUANT_MODE == 16_8.
		 */
		if (ctx->family == CHIP_RAVEN)
			max_extent = 16384; /* Use QUANT_MODE == 16_8. */

		if (max_extent <= 1024) /* 4K scanline area for guardband */
			scissor->quant_mode = SI_QUANT_MODE_12_12_FIXED_POINT_1_4096TH;
		else if (max_extent <= 4096) /* 16K scanline area for guardband */
			scissor->quant_mode = SI_QUANT_MODE_14_10_FIXED_POINT_1_1024TH;
		else /* 64K scanline area for guardband */
			scissor->quant_mode = SI_QUANT_MODE_16_8_FIXED_POINT_1_256TH;
	}

	mask = ((1 << num_viewports) - 1) << start_slot;
	ctx->viewports.dirty_mask |= mask;
	ctx->viewports.depth_range_dirty_mask |= mask;
	ctx->scissors.dirty_mask |= mask;
	si_mark_atom_dirty(ctx, &ctx->atoms.s.viewports);
	si_mark_atom_dirty(ctx, &ctx->atoms.s.guardband);
	si_mark_atom_dirty(ctx, &ctx->atoms.s.scissors);
}

static void si_emit_one_viewport(struct si_context *ctx,
				 struct pipe_viewport_state *state)
{
	struct radeon_cmdbuf *cs = ctx->gfx_cs;

	radeon_emit(cs, fui(state->scale[0]));
	radeon_emit(cs, fui(state->translate[0]));
	radeon_emit(cs, fui(state->scale[1]));
	radeon_emit(cs, fui(state->translate[1]));
	radeon_emit(cs, fui(state->scale[2]));
	radeon_emit(cs, fui(state->translate[2]));
}

static void si_emit_viewports(struct si_context *ctx)
{
	struct radeon_cmdbuf *cs = ctx->gfx_cs;
	struct pipe_viewport_state *states = ctx->viewports.states;
	unsigned mask = ctx->viewports.dirty_mask;

	/* The simple case: Only 1 viewport is active. */
	if (!ctx->vs_writes_viewport_index) {
		if (!(mask & 1))
			return;

		radeon_set_context_reg_seq(cs, R_02843C_PA_CL_VPORT_XSCALE, 6);
		si_emit_one_viewport(ctx, &states[0]);
		ctx->viewports.dirty_mask &= ~1; /* clear one bit */
		return;
	}

	while (mask) {
		int start, count, i;

		u_bit_scan_consecutive_range(&mask, &start, &count);

		radeon_set_context_reg_seq(cs, R_02843C_PA_CL_VPORT_XSCALE +
					       start * 4 * 6, count * 6);
		for (i = start; i < start+count; i++)
			si_emit_one_viewport(ctx, &states[i]);
	}
	ctx->viewports.dirty_mask = 0;
}

static inline void
si_viewport_zmin_zmax(const struct pipe_viewport_state *vp, bool halfz,
		      bool window_space_position, float *zmin, float *zmax)
{
	if (window_space_position) {
		*zmin = 0;
		*zmax = 1;
		return;
	}
	util_viewport_zmin_zmax(vp, halfz, zmin, zmax);
}

static void si_emit_depth_ranges(struct si_context *ctx)
{
	struct radeon_cmdbuf *cs = ctx->gfx_cs;
	struct pipe_viewport_state *states = ctx->viewports.states;
	unsigned mask = ctx->viewports.depth_range_dirty_mask;
	bool clip_halfz = ctx->queued.named.rasterizer->clip_halfz;
	bool window_space = ctx->vs_disables_clipping_viewport;
	float zmin, zmax;

	/* The simple case: Only 1 viewport is active. */
	if (!ctx->vs_writes_viewport_index) {
		if (!(mask & 1))
			return;

		si_viewport_zmin_zmax(&states[0], clip_halfz, window_space,
				      &zmin, &zmax);

		radeon_set_context_reg_seq(cs, R_0282D0_PA_SC_VPORT_ZMIN_0, 2);
		radeon_emit(cs, fui(zmin));
		radeon_emit(cs, fui(zmax));
		ctx->viewports.depth_range_dirty_mask &= ~1; /* clear one bit */
		return;
	}

	while (mask) {
		int start, count, i;

		u_bit_scan_consecutive_range(&mask, &start, &count);

		radeon_set_context_reg_seq(cs, R_0282D0_PA_SC_VPORT_ZMIN_0 +
					   start * 4 * 2, count * 2);
		for (i = start; i < start+count; i++) {
			si_viewport_zmin_zmax(&states[i], clip_halfz, window_space,
					      &zmin, &zmax);
			radeon_emit(cs, fui(zmin));
			radeon_emit(cs, fui(zmax));
		}
	}
	ctx->viewports.depth_range_dirty_mask = 0;
}

static void si_emit_viewport_states(struct si_context *ctx)
{
	si_emit_viewports(ctx);
	si_emit_depth_ranges(ctx);
}

/**
 * This reacts to 2 state changes:
 * - VS.writes_viewport_index
 * - VS output position in window space (enable/disable)
 *
 * Normally, we only emit 1 viewport and 1 scissor if no shader is using
 * the VIEWPORT_INDEX output, and emitting the other viewports and scissors
 * is delayed. When a shader with VIEWPORT_INDEX appears, this should be
 * called to emit the rest.
 */
void si_update_vs_viewport_state(struct si_context *ctx)
{
	struct tgsi_shader_info *info = si_get_vs_info(ctx);
	bool vs_window_space;

	if (!info)
		return;

	/* When the VS disables clipping and viewport transformation. */
	vs_window_space =
		info->properties[TGSI_PROPERTY_VS_WINDOW_SPACE_POSITION];

	if (ctx->vs_disables_clipping_viewport != vs_window_space) {
		ctx->vs_disables_clipping_viewport = vs_window_space;
		ctx->scissors.dirty_mask = (1 << SI_MAX_VIEWPORTS) - 1;
		ctx->viewports.depth_range_dirty_mask = (1 << SI_MAX_VIEWPORTS) - 1;
		si_mark_atom_dirty(ctx, &ctx->atoms.s.scissors);
		si_mark_atom_dirty(ctx, &ctx->atoms.s.viewports);
	}

	/* Viewport index handling. */
	if (ctx->vs_writes_viewport_index == info->writes_viewport_index)
		return;

	/* This changes how the guardband is computed. */
	ctx->vs_writes_viewport_index = info->writes_viewport_index;
	si_mark_atom_dirty(ctx, &ctx->atoms.s.guardband);

	if (!ctx->vs_writes_viewport_index)
		return;

	if (ctx->scissors.dirty_mask)
	    si_mark_atom_dirty(ctx, &ctx->atoms.s.scissors);

	if (ctx->viewports.dirty_mask ||
	    ctx->viewports.depth_range_dirty_mask)
	    si_mark_atom_dirty(ctx, &ctx->atoms.s.viewports);
}

static void si_emit_window_rectangles(struct si_context *sctx)
{
	/* There are four clipping rectangles. Their corner coordinates are inclusive.
	 * Every pixel is assigned a number from 0 and 15 by setting bits 0-3 depending
	 * on whether the pixel is inside cliprects 0-3, respectively. For example,
	 * if a pixel is inside cliprects 0 and 1, but outside 2 and 3, it is assigned
	 * the number 3 (binary 0011).
	 *
	 * If CLIPRECT_RULE & (1 << number), the pixel is rasterized.
	 */
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	static const unsigned outside[4] = {
		/* outside rectangle 0 */
		V_02820C_OUT |
		V_02820C_IN_1 |
		V_02820C_IN_2 |
		V_02820C_IN_21 |
		V_02820C_IN_3 |
		V_02820C_IN_31 |
		V_02820C_IN_32 |
		V_02820C_IN_321,
		/* outside rectangles 0, 1 */
		V_02820C_OUT |
		V_02820C_IN_2 |
		V_02820C_IN_3 |
		V_02820C_IN_32,
		/* outside rectangles 0, 1, 2 */
		V_02820C_OUT |
		V_02820C_IN_3,
		/* outside rectangles 0, 1, 2, 3 */
		V_02820C_OUT,
	};
	const unsigned disabled = 0xffff; /* all inside and outside cases */
	unsigned num_rectangles = sctx->num_window_rectangles;
	struct pipe_scissor_state *rects = sctx->window_rectangles;
	unsigned rule;

	assert(num_rectangles <= 4);

	if (num_rectangles == 0)
		rule = disabled;
	else if (sctx->window_rectangles_include)
		rule = ~outside[num_rectangles - 1];
	else
		rule = outside[num_rectangles - 1];

	radeon_opt_set_context_reg(sctx, R_02820C_PA_SC_CLIPRECT_RULE,
				   SI_TRACKED_PA_SC_CLIPRECT_RULE, rule);
	if (num_rectangles == 0)
		return;

	radeon_set_context_reg_seq(cs, R_028210_PA_SC_CLIPRECT_0_TL,
				   num_rectangles * 2);
	for (unsigned i = 0; i < num_rectangles; i++) {
		radeon_emit(cs, S_028210_TL_X(rects[i].minx) |
				S_028210_TL_Y(rects[i].miny));
		radeon_emit(cs, S_028214_BR_X(rects[i].maxx) |
				S_028214_BR_Y(rects[i].maxy));
	}
}

static void si_set_window_rectangles(struct pipe_context *ctx,
				     boolean include,
				     unsigned num_rectangles,
				     const struct pipe_scissor_state *rects)
{
	struct si_context *sctx = (struct si_context *)ctx;

	sctx->num_window_rectangles = num_rectangles;
	sctx->window_rectangles_include = include;
	if (num_rectangles) {
		memcpy(sctx->window_rectangles, rects,
		       sizeof(*rects) * num_rectangles);
	}

	si_mark_atom_dirty(sctx, &sctx->atoms.s.window_rectangles);
}

void si_init_viewport_functions(struct si_context *ctx)
{
	ctx->atoms.s.guardband.emit = si_emit_guardband;
	ctx->atoms.s.scissors.emit = si_emit_scissors;
	ctx->atoms.s.viewports.emit = si_emit_viewport_states;
	ctx->atoms.s.window_rectangles.emit = si_emit_window_rectangles;

	ctx->b.set_scissor_states = si_set_scissor_states;
	ctx->b.set_viewport_states = si_set_viewport_states;
	ctx->b.set_window_rectangles = si_set_window_rectangles;

	for (unsigned i = 0; i < 16; i++)
		ctx->viewports.as_scissor[i].quant_mode = SI_QUANT_MODE_16_8_FIXED_POINT_1_256TH;
}
