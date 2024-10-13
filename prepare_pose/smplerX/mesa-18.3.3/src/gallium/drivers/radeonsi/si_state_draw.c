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
#include "gfx9d.h"

#include "util/u_index_modify.h"
#include "util/u_log.h"
#include "util/u_upload_mgr.h"
#include "util/u_prim.h"

#include "ac_debug.h"

/* special primitive types */
#define SI_PRIM_RECTANGLE_LIST	PIPE_PRIM_MAX

static unsigned si_conv_pipe_prim(unsigned mode)
{
        static const unsigned prim_conv[] = {
		[PIPE_PRIM_POINTS]			= V_008958_DI_PT_POINTLIST,
		[PIPE_PRIM_LINES]			= V_008958_DI_PT_LINELIST,
		[PIPE_PRIM_LINE_LOOP]			= V_008958_DI_PT_LINELOOP,
		[PIPE_PRIM_LINE_STRIP]			= V_008958_DI_PT_LINESTRIP,
		[PIPE_PRIM_TRIANGLES]			= V_008958_DI_PT_TRILIST,
		[PIPE_PRIM_TRIANGLE_STRIP]		= V_008958_DI_PT_TRISTRIP,
		[PIPE_PRIM_TRIANGLE_FAN]		= V_008958_DI_PT_TRIFAN,
		[PIPE_PRIM_QUADS]			= V_008958_DI_PT_QUADLIST,
		[PIPE_PRIM_QUAD_STRIP]			= V_008958_DI_PT_QUADSTRIP,
		[PIPE_PRIM_POLYGON]			= V_008958_DI_PT_POLYGON,
		[PIPE_PRIM_LINES_ADJACENCY]		= V_008958_DI_PT_LINELIST_ADJ,
		[PIPE_PRIM_LINE_STRIP_ADJACENCY]	= V_008958_DI_PT_LINESTRIP_ADJ,
		[PIPE_PRIM_TRIANGLES_ADJACENCY]		= V_008958_DI_PT_TRILIST_ADJ,
		[PIPE_PRIM_TRIANGLE_STRIP_ADJACENCY]	= V_008958_DI_PT_TRISTRIP_ADJ,
		[PIPE_PRIM_PATCHES]			= V_008958_DI_PT_PATCH,
		[SI_PRIM_RECTANGLE_LIST]		= V_008958_DI_PT_RECTLIST
        };
	assert(mode < ARRAY_SIZE(prim_conv));
	return prim_conv[mode];
}

/**
 * This calculates the LDS size for tessellation shaders (VS, TCS, TES).
 * LS.LDS_SIZE is shared by all 3 shader stages.
 *
 * The information about LDS and other non-compile-time parameters is then
 * written to userdata SGPRs.
 */
static bool si_emit_derived_tess_state(struct si_context *sctx,
				       const struct pipe_draw_info *info,
				       unsigned *num_patches)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	struct si_shader *ls_current;
	struct si_shader_selector *ls;
	/* The TES pointer will only be used for sctx->last_tcs.
	 * It would be wrong to think that TCS = TES. */
	struct si_shader_selector *tcs =
		sctx->tcs_shader.cso ? sctx->tcs_shader.cso : sctx->tes_shader.cso;
	unsigned tess_uses_primid = sctx->ia_multi_vgt_param_key.u.tess_uses_prim_id;
	bool has_primid_instancing_bug = sctx->chip_class == SI &&
					 sctx->screen->info.max_se == 1;
	unsigned tes_sh_base = sctx->shader_pointers.sh_base[PIPE_SHADER_TESS_EVAL];
	unsigned num_tcs_input_cp = info->vertices_per_patch;
	unsigned num_tcs_output_cp, num_tcs_inputs, num_tcs_outputs;
	unsigned num_tcs_patch_outputs;
	unsigned input_vertex_size, output_vertex_size, pervertex_output_patch_size;
	unsigned input_patch_size, output_patch_size, output_patch0_offset;
	unsigned perpatch_output_offset, lds_size;
	unsigned tcs_in_layout, tcs_out_layout, tcs_out_offsets;
	unsigned offchip_layout, hardware_lds_size, ls_hs_config;

	/* Since GFX9 has merged LS-HS in the TCS state, set LS = TCS. */
	if (sctx->chip_class >= GFX9) {
		if (sctx->tcs_shader.cso)
			ls_current = sctx->tcs_shader.current;
		else
			ls_current = sctx->fixed_func_tcs_shader.current;

		ls = ls_current->key.part.tcs.ls;
	} else {
		ls_current = sctx->vs_shader.current;
		ls = sctx->vs_shader.cso;
	}

	if (sctx->last_ls == ls_current &&
	    sctx->last_tcs == tcs &&
	    sctx->last_tes_sh_base == tes_sh_base &&
	    sctx->last_num_tcs_input_cp == num_tcs_input_cp &&
	    (!has_primid_instancing_bug ||
	     (sctx->last_tess_uses_primid == tess_uses_primid))) {
		*num_patches = sctx->last_num_patches;
		return false;
	}

	sctx->last_ls = ls_current;
	sctx->last_tcs = tcs;
	sctx->last_tes_sh_base = tes_sh_base;
	sctx->last_num_tcs_input_cp = num_tcs_input_cp;
	sctx->last_tess_uses_primid = tess_uses_primid;

	/* This calculates how shader inputs and outputs among VS, TCS, and TES
	 * are laid out in LDS. */
	num_tcs_inputs = util_last_bit64(ls->outputs_written);

	if (sctx->tcs_shader.cso) {
		num_tcs_outputs = util_last_bit64(tcs->outputs_written);
		num_tcs_output_cp = tcs->info.properties[TGSI_PROPERTY_TCS_VERTICES_OUT];
		num_tcs_patch_outputs = util_last_bit64(tcs->patch_outputs_written);
	} else {
		/* No TCS. Route varyings from LS to TES. */
		num_tcs_outputs = num_tcs_inputs;
		num_tcs_output_cp = num_tcs_input_cp;
		num_tcs_patch_outputs = 2; /* TESSINNER + TESSOUTER */
	}

	input_vertex_size = ls->lshs_vertex_stride;
	output_vertex_size = num_tcs_outputs * 16;

	input_patch_size = num_tcs_input_cp * input_vertex_size;

	pervertex_output_patch_size = num_tcs_output_cp * output_vertex_size;
	output_patch_size = pervertex_output_patch_size + num_tcs_patch_outputs * 16;

	/* Ensure that we only need one wave per SIMD so we don't need to check
	 * resource usage. Also ensures that the number of tcs in and out
	 * vertices per threadgroup are at most 256.
	 */
	unsigned max_verts_per_patch = MAX2(num_tcs_input_cp, num_tcs_output_cp);
	*num_patches = 256 / max_verts_per_patch;

	/* Make sure that the data fits in LDS. This assumes the shaders only
	 * use LDS for the inputs and outputs.
	 *
	 * While CIK can use 64K per threadgroup, there is a hang on Stoney
	 * with 2 CUs if we use more than 32K. The closed Vulkan driver also
	 * uses 32K at most on all GCN chips.
	 */
	hardware_lds_size = 32768;
	*num_patches = MIN2(*num_patches, hardware_lds_size / (input_patch_size +
	                                                       output_patch_size));

	/* Make sure the output data fits in the offchip buffer */
	*num_patches = MIN2(*num_patches,
			    (sctx->screen->tess_offchip_block_dw_size * 4) /
			    output_patch_size);

	/* Not necessary for correctness, but improves performance.
	 * The hardware can do more, but the radeonsi shader constant is
	 * limited to 6 bits.
	 */
	*num_patches = MIN2(*num_patches, 63); /* triangles: 3 full waves except 3 lanes */

	/* When distributed tessellation is unsupported, switch between SEs
	 * at a higher frequency to compensate for it.
	 */
	if (!sctx->screen->has_distributed_tess && sctx->screen->info.max_se > 1)
		*num_patches = MIN2(*num_patches, 16); /* recommended */

	/* Make sure that vector lanes are reasonably occupied. It probably
	 * doesn't matter much because this is LS-HS, and TES is likely to
	 * occupy significantly more CUs.
	 */
	unsigned temp_verts_per_tg = *num_patches * max_verts_per_patch;
	if (temp_verts_per_tg > 64 && temp_verts_per_tg % 64 < 48)
		*num_patches = (temp_verts_per_tg & ~63) / max_verts_per_patch;

	if (sctx->chip_class == SI) {
		/* SI bug workaround, related to power management. Limit LS-HS
		 * threadgroups to only one wave.
		 */
		unsigned one_wave = 64 / max_verts_per_patch;
		*num_patches = MIN2(*num_patches, one_wave);
	}

	/* The VGT HS block increments the patch ID unconditionally
	 * within a single threadgroup. This results in incorrect
	 * patch IDs when instanced draws are used.
	 *
	 * The intended solution is to restrict threadgroups to
	 * a single instance by setting SWITCH_ON_EOI, which
	 * should cause IA to split instances up. However, this
	 * doesn't work correctly on SI when there is no other
	 * SE to switch to.
	 */
	if (has_primid_instancing_bug && tess_uses_primid)
		*num_patches = 1;

	sctx->last_num_patches = *num_patches;

	output_patch0_offset = input_patch_size * *num_patches;
	perpatch_output_offset = output_patch0_offset + pervertex_output_patch_size;

	/* Compute userdata SGPRs. */
	assert(((input_vertex_size / 4) & ~0xff) == 0);
	assert(((output_vertex_size / 4) & ~0xff) == 0);
	assert(((input_patch_size / 4) & ~0x1fff) == 0);
	assert(((output_patch_size / 4) & ~0x1fff) == 0);
	assert(((output_patch0_offset / 16) & ~0xffff) == 0);
	assert(((perpatch_output_offset / 16) & ~0xffff) == 0);
	assert(num_tcs_input_cp <= 32);
	assert(num_tcs_output_cp <= 32);

	uint64_t ring_va = r600_resource(sctx->tess_rings)->gpu_address;
	assert((ring_va & u_bit_consecutive(0, 19)) == 0);

	tcs_in_layout = S_VS_STATE_LS_OUT_PATCH_SIZE(input_patch_size / 4) |
			S_VS_STATE_LS_OUT_VERTEX_SIZE(input_vertex_size / 4);
	tcs_out_layout = (output_patch_size / 4) |
			 (num_tcs_input_cp << 13) |
			 ring_va;
	tcs_out_offsets = (output_patch0_offset / 16) |
			  ((perpatch_output_offset / 16) << 16);
	offchip_layout = *num_patches |
			 (num_tcs_output_cp << 6) |
			 (pervertex_output_patch_size * *num_patches << 12);

	/* Compute the LDS size. */
	lds_size = output_patch0_offset + output_patch_size * *num_patches;

	if (sctx->chip_class >= CIK) {
		assert(lds_size <= 65536);
		lds_size = align(lds_size, 512) / 512;
	} else {
		assert(lds_size <= 32768);
		lds_size = align(lds_size, 256) / 256;
	}

	/* Set SI_SGPR_VS_STATE_BITS. */
	sctx->current_vs_state &= C_VS_STATE_LS_OUT_PATCH_SIZE &
				  C_VS_STATE_LS_OUT_VERTEX_SIZE;
	sctx->current_vs_state |= tcs_in_layout;

	if (sctx->chip_class >= GFX9) {
		unsigned hs_rsrc2 = ls_current->config.rsrc2 |
				    S_00B42C_LDS_SIZE(lds_size);

		radeon_set_sh_reg(cs, R_00B42C_SPI_SHADER_PGM_RSRC2_HS, hs_rsrc2);

		/* Set userdata SGPRs for merged LS-HS. */
		radeon_set_sh_reg_seq(cs,
				      R_00B430_SPI_SHADER_USER_DATA_LS_0 +
				      GFX9_SGPR_TCS_OFFCHIP_LAYOUT * 4, 3);
		radeon_emit(cs, offchip_layout);
		radeon_emit(cs, tcs_out_offsets);
		radeon_emit(cs, tcs_out_layout);
	} else {
		unsigned ls_rsrc2 = ls_current->config.rsrc2;

		si_multiwave_lds_size_workaround(sctx->screen, &lds_size);
		ls_rsrc2 |= S_00B52C_LDS_SIZE(lds_size);

		/* Due to a hw bug, RSRC2_LS must be written twice with another
		 * LS register written in between. */
		if (sctx->chip_class == CIK && sctx->family != CHIP_HAWAII)
			radeon_set_sh_reg(cs, R_00B52C_SPI_SHADER_PGM_RSRC2_LS, ls_rsrc2);
		radeon_set_sh_reg_seq(cs, R_00B528_SPI_SHADER_PGM_RSRC1_LS, 2);
		radeon_emit(cs, ls_current->config.rsrc1);
		radeon_emit(cs, ls_rsrc2);

		/* Set userdata SGPRs for TCS. */
		radeon_set_sh_reg_seq(cs,
			R_00B430_SPI_SHADER_USER_DATA_HS_0 + GFX6_SGPR_TCS_OFFCHIP_LAYOUT * 4, 4);
		radeon_emit(cs, offchip_layout);
		radeon_emit(cs, tcs_out_offsets);
		radeon_emit(cs, tcs_out_layout);
		radeon_emit(cs, tcs_in_layout);
	}

	/* Set userdata SGPRs for TES. */
	radeon_set_sh_reg_seq(cs, tes_sh_base + SI_SGPR_TES_OFFCHIP_LAYOUT * 4, 2);
	radeon_emit(cs, offchip_layout);
	radeon_emit(cs, ring_va);

	ls_hs_config = S_028B58_NUM_PATCHES(*num_patches) |
		       S_028B58_HS_NUM_INPUT_CP(num_tcs_input_cp) |
		       S_028B58_HS_NUM_OUTPUT_CP(num_tcs_output_cp);

	if (sctx->last_ls_hs_config != ls_hs_config) {
		if (sctx->chip_class >= CIK) {
			radeon_set_context_reg_idx(cs, R_028B58_VGT_LS_HS_CONFIG, 2,
						   ls_hs_config);
		} else {
			radeon_set_context_reg(cs, R_028B58_VGT_LS_HS_CONFIG,
					       ls_hs_config);
		}
		sctx->last_ls_hs_config = ls_hs_config;
		return true; /* true if the context rolls */
	}
	return false;
}

static unsigned si_num_prims_for_vertices(const struct pipe_draw_info *info)
{
	switch (info->mode) {
	case PIPE_PRIM_PATCHES:
		return info->count / info->vertices_per_patch;
	case SI_PRIM_RECTANGLE_LIST:
		return info->count / 3;
	default:
		return u_prims_for_vertices(info->mode, info->count);
	}
}

static unsigned
si_get_init_multi_vgt_param(struct si_screen *sscreen,
			    union si_vgt_param_key *key)
{
	STATIC_ASSERT(sizeof(union si_vgt_param_key) == 4);
	unsigned max_primgroup_in_wave = 2;

	/* SWITCH_ON_EOP(0) is always preferable. */
	bool wd_switch_on_eop = false;
	bool ia_switch_on_eop = false;
	bool ia_switch_on_eoi = false;
	bool partial_vs_wave = false;
	bool partial_es_wave = false;

	if (key->u.uses_tess) {
		/* SWITCH_ON_EOI must be set if PrimID is used. */
		if (key->u.tess_uses_prim_id)
			ia_switch_on_eoi = true;

		/* Bug with tessellation and GS on Bonaire and older 2 SE chips. */
		if ((sscreen->info.family == CHIP_TAHITI ||
		     sscreen->info.family == CHIP_PITCAIRN ||
		     sscreen->info.family == CHIP_BONAIRE) &&
		    key->u.uses_gs)
			partial_vs_wave = true;

		/* Needed for 028B6C_DISTRIBUTION_MODE != 0. (implies >= VI) */
		if (sscreen->has_distributed_tess) {
			if (key->u.uses_gs) {
				if (sscreen->info.chip_class == VI)
					partial_es_wave = true;
			} else {
				partial_vs_wave = true;
			}
		}
	}

	/* This is a hardware requirement. */
	if (key->u.line_stipple_enabled ||
	    (sscreen->debug_flags & DBG(SWITCH_ON_EOP))) {
		ia_switch_on_eop = true;
		wd_switch_on_eop = true;
	}

	if (sscreen->info.chip_class >= CIK) {
		/* WD_SWITCH_ON_EOP has no effect on GPUs with less than
		 * 4 shader engines. Set 1 to pass the assertion below.
		 * The other cases are hardware requirements.
		 *
		 * Polaris supports primitive restart with WD_SWITCH_ON_EOP=0
		 * for points, line strips, and tri strips.
		 */
		if (sscreen->info.max_se <= 2 ||
		    key->u.prim == PIPE_PRIM_POLYGON ||
		    key->u.prim == PIPE_PRIM_LINE_LOOP ||
		    key->u.prim == PIPE_PRIM_TRIANGLE_FAN ||
		    key->u.prim == PIPE_PRIM_TRIANGLE_STRIP_ADJACENCY ||
		    (key->u.primitive_restart &&
		     (sscreen->info.family < CHIP_POLARIS10 ||
		      (key->u.prim != PIPE_PRIM_POINTS &&
		       key->u.prim != PIPE_PRIM_LINE_STRIP &&
		       key->u.prim != PIPE_PRIM_TRIANGLE_STRIP))) ||
		    key->u.count_from_stream_output)
			wd_switch_on_eop = true;

		/* Hawaii hangs if instancing is enabled and WD_SWITCH_ON_EOP is 0.
		 * We don't know that for indirect drawing, so treat it as
		 * always problematic. */
		if (sscreen->info.family == CHIP_HAWAII &&
		    key->u.uses_instancing)
			wd_switch_on_eop = true;

		/* Performance recommendation for 4 SE Gfx7-8 parts if
		 * instances are smaller than a primgroup.
		 * Assume indirect draws always use small instances.
		 * This is needed for good VS wave utilization.
		 */
		if (sscreen->info.chip_class <= VI &&
		    sscreen->info.max_se == 4 &&
		    key->u.multi_instances_smaller_than_primgroup)
			wd_switch_on_eop = true;

		/* Required on CIK and later. */
		if (sscreen->info.max_se == 4 && !wd_switch_on_eop)
			ia_switch_on_eoi = true;

		/* HW engineers suggested that PARTIAL_VS_WAVE_ON should be set
		 * to work around a GS hang.
		 */
		if (key->u.uses_gs &&
		    (sscreen->info.family == CHIP_TONGA ||
		     sscreen->info.family == CHIP_FIJI ||
		     sscreen->info.family == CHIP_POLARIS10 ||
		     sscreen->info.family == CHIP_POLARIS11 ||
		     sscreen->info.family == CHIP_POLARIS12 ||
		     sscreen->info.family == CHIP_VEGAM))
			partial_vs_wave = true;

		/* Required by Hawaii and, for some special cases, by VI. */
		if (ia_switch_on_eoi &&
		    (sscreen->info.family == CHIP_HAWAII ||
		     (sscreen->info.chip_class == VI &&
		      (key->u.uses_gs || max_primgroup_in_wave != 2))))
			partial_vs_wave = true;

		/* Instancing bug on Bonaire. */
		if (sscreen->info.family == CHIP_BONAIRE && ia_switch_on_eoi &&
		    key->u.uses_instancing)
			partial_vs_wave = true;

		/* This only applies to Polaris10 and later 4 SE chips.
		 * wd_switch_on_eop is already true on all other chips.
		 */
		if (!wd_switch_on_eop && key->u.primitive_restart)
			partial_vs_wave = true;

		/* If the WD switch is false, the IA switch must be false too. */
		assert(wd_switch_on_eop || !ia_switch_on_eop);
	}

	/* If SWITCH_ON_EOI is set, PARTIAL_ES_WAVE must be set too. */
	if (sscreen->info.chip_class <= VI && ia_switch_on_eoi)
		partial_es_wave = true;

	return S_028AA8_SWITCH_ON_EOP(ia_switch_on_eop) |
		S_028AA8_SWITCH_ON_EOI(ia_switch_on_eoi) |
		S_028AA8_PARTIAL_VS_WAVE_ON(partial_vs_wave) |
		S_028AA8_PARTIAL_ES_WAVE_ON(partial_es_wave) |
		S_028AA8_WD_SWITCH_ON_EOP(sscreen->info.chip_class >= CIK ? wd_switch_on_eop : 0) |
		/* The following field was moved to VGT_SHADER_STAGES_EN in GFX9. */
		S_028AA8_MAX_PRIMGRP_IN_WAVE(sscreen->info.chip_class == VI ?
					     max_primgroup_in_wave : 0) |
		S_030960_EN_INST_OPT_BASIC(sscreen->info.chip_class >= GFX9) |
		S_030960_EN_INST_OPT_ADV(sscreen->info.chip_class >= GFX9);
}

void si_init_ia_multi_vgt_param_table(struct si_context *sctx)
{
	for (int prim = 0; prim <= SI_PRIM_RECTANGLE_LIST; prim++)
	for (int uses_instancing = 0; uses_instancing < 2; uses_instancing++)
	for (int multi_instances = 0; multi_instances < 2; multi_instances++)
	for (int primitive_restart = 0; primitive_restart < 2; primitive_restart++)
	for (int count_from_so = 0; count_from_so < 2; count_from_so++)
	for (int line_stipple = 0; line_stipple < 2; line_stipple++)
	for (int uses_tess = 0; uses_tess < 2; uses_tess++)
	for (int tess_uses_primid = 0; tess_uses_primid < 2; tess_uses_primid++)
	for (int uses_gs = 0; uses_gs < 2; uses_gs++) {
		union si_vgt_param_key key;

		key.index = 0;
		key.u.prim = prim;
		key.u.uses_instancing = uses_instancing;
		key.u.multi_instances_smaller_than_primgroup = multi_instances;
		key.u.primitive_restart = primitive_restart;
		key.u.count_from_stream_output = count_from_so;
		key.u.line_stipple_enabled = line_stipple;
		key.u.uses_tess = uses_tess;
		key.u.tess_uses_prim_id = tess_uses_primid;
		key.u.uses_gs = uses_gs;

		sctx->ia_multi_vgt_param[key.index] =
			si_get_init_multi_vgt_param(sctx->screen, &key);
	}
}

static unsigned si_get_ia_multi_vgt_param(struct si_context *sctx,
					  const struct pipe_draw_info *info,
					  unsigned num_patches)
{
	union si_vgt_param_key key = sctx->ia_multi_vgt_param_key;
	unsigned primgroup_size;
	unsigned ia_multi_vgt_param;

	if (sctx->tes_shader.cso) {
		primgroup_size = num_patches; /* must be a multiple of NUM_PATCHES */
	} else if (sctx->gs_shader.cso) {
		primgroup_size = 64; /* recommended with a GS */
	} else {
		primgroup_size = 128; /* recommended without a GS and tess */
	}

	key.u.prim = info->mode;
	key.u.uses_instancing = info->indirect || info->instance_count > 1;
	key.u.multi_instances_smaller_than_primgroup =
		info->indirect ||
		(info->instance_count > 1 &&
		 (info->count_from_stream_output ||
		  si_num_prims_for_vertices(info) < primgroup_size));
	key.u.primitive_restart = info->primitive_restart;
	key.u.count_from_stream_output = info->count_from_stream_output != NULL;

	ia_multi_vgt_param = sctx->ia_multi_vgt_param[key.index] |
			     S_028AA8_PRIMGROUP_SIZE(primgroup_size - 1);

	if (sctx->gs_shader.cso) {
		/* GS requirement. */
		if (sctx->chip_class <= VI &&
		    SI_GS_PER_ES / primgroup_size >= sctx->screen->gs_table_depth - 3)
			ia_multi_vgt_param |= S_028AA8_PARTIAL_ES_WAVE_ON(1);

		/* GS hw bug with single-primitive instances and SWITCH_ON_EOI.
		 * The hw doc says all multi-SE chips are affected, but Vulkan
		 * only applies it to Hawaii. Do what Vulkan does.
		 */
		if (sctx->family == CHIP_HAWAII &&
		    G_028AA8_SWITCH_ON_EOI(ia_multi_vgt_param) &&
		    (info->indirect ||
		     (info->instance_count > 1 &&
		      (info->count_from_stream_output ||
		       si_num_prims_for_vertices(info) <= 1))))
			sctx->flags |= SI_CONTEXT_VGT_FLUSH;
	}

	return ia_multi_vgt_param;
}

/* rast_prim is the primitive type after GS. */
static bool si_emit_rasterizer_prim_state(struct si_context *sctx)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	enum pipe_prim_type rast_prim = sctx->current_rast_prim;
	struct si_state_rasterizer *rs = sctx->queued.named.rasterizer;

	/* Skip this if not rendering lines. */
	if (!util_prim_is_lines(rast_prim))
		return false;

	if (rast_prim == sctx->last_rast_prim &&
	    rs->pa_sc_line_stipple == sctx->last_sc_line_stipple)
		return false;

	/* For lines, reset the stipple pattern at each primitive. Otherwise,
	 * reset the stipple pattern at each packet (line strips, line loops).
	 */
	radeon_set_context_reg(cs, R_028A0C_PA_SC_LINE_STIPPLE,
		rs->pa_sc_line_stipple |
		S_028A0C_AUTO_RESET_CNTL(rast_prim == PIPE_PRIM_LINES ? 1 : 2));

	sctx->last_rast_prim = rast_prim;
	sctx->last_sc_line_stipple = rs->pa_sc_line_stipple;
	return true; /* true if the context rolls */
}

static void si_emit_vs_state(struct si_context *sctx,
			     const struct pipe_draw_info *info)
{
	sctx->current_vs_state &= C_VS_STATE_INDEXED;
	sctx->current_vs_state |= S_VS_STATE_INDEXED(!!info->index_size);

	if (sctx->num_vs_blit_sgprs) {
		/* Re-emit the state after we leave u_blitter. */
		sctx->last_vs_state = ~0;
		return;
	}

	if (sctx->current_vs_state != sctx->last_vs_state) {
		struct radeon_cmdbuf *cs = sctx->gfx_cs;

		/* For the API vertex shader (VS_STATE_INDEXED). */
		radeon_set_sh_reg(cs,
			sctx->shader_pointers.sh_base[PIPE_SHADER_VERTEX] +
			SI_SGPR_VS_STATE_BITS * 4,
			sctx->current_vs_state);

		/* For vertex color clamping, which is done in the last stage
		 * before the rasterizer. */
		if (sctx->gs_shader.cso || sctx->tes_shader.cso) {
			/* GS copy shader or TES if GS is missing. */
			radeon_set_sh_reg(cs,
				R_00B130_SPI_SHADER_USER_DATA_VS_0 +
				SI_SGPR_VS_STATE_BITS * 4,
				sctx->current_vs_state);
		}

		sctx->last_vs_state = sctx->current_vs_state;
	}
}

static inline bool si_prim_restart_index_changed(struct si_context *sctx,
						 const struct pipe_draw_info *info)
{
	return info->primitive_restart &&
	       (info->restart_index != sctx->last_restart_index ||
		sctx->last_restart_index == SI_RESTART_INDEX_UNKNOWN);
}

static void si_emit_draw_registers(struct si_context *sctx,
				   const struct pipe_draw_info *info,
				   unsigned num_patches)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	unsigned prim = si_conv_pipe_prim(info->mode);
	unsigned ia_multi_vgt_param;

	ia_multi_vgt_param = si_get_ia_multi_vgt_param(sctx, info, num_patches);

	/* Draw state. */
	if (ia_multi_vgt_param != sctx->last_multi_vgt_param) {
		if (sctx->chip_class >= GFX9)
			radeon_set_uconfig_reg_idx(cs, R_030960_IA_MULTI_VGT_PARAM, 4, ia_multi_vgt_param);
		else if (sctx->chip_class >= CIK)
			radeon_set_context_reg_idx(cs, R_028AA8_IA_MULTI_VGT_PARAM, 1, ia_multi_vgt_param);
		else
			radeon_set_context_reg(cs, R_028AA8_IA_MULTI_VGT_PARAM, ia_multi_vgt_param);

		sctx->last_multi_vgt_param = ia_multi_vgt_param;
	}
	if (prim != sctx->last_prim) {
		if (sctx->chip_class >= CIK)
			radeon_set_uconfig_reg_idx(cs, R_030908_VGT_PRIMITIVE_TYPE, 1, prim);
		else
			radeon_set_config_reg(cs, R_008958_VGT_PRIMITIVE_TYPE, prim);

		sctx->last_prim = prim;
	}

	/* Primitive restart. */
	if (info->primitive_restart != sctx->last_primitive_restart_en) {
		if (sctx->chip_class >= GFX9)
			radeon_set_uconfig_reg(cs, R_03092C_VGT_MULTI_PRIM_IB_RESET_EN,
					       info->primitive_restart);
		else
			radeon_set_context_reg(cs, R_028A94_VGT_MULTI_PRIM_IB_RESET_EN,
					       info->primitive_restart);

		sctx->last_primitive_restart_en = info->primitive_restart;

	}
	if (si_prim_restart_index_changed(sctx, info)) {
		radeon_set_context_reg(cs, R_02840C_VGT_MULTI_PRIM_IB_RESET_INDX,
				       info->restart_index);
		sctx->last_restart_index = info->restart_index;
	}
}

static void si_emit_draw_packets(struct si_context *sctx,
				 const struct pipe_draw_info *info,
				 struct pipe_resource *indexbuf,
				 unsigned index_size,
				 unsigned index_offset)
{
	struct pipe_draw_indirect_info *indirect = info->indirect;
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	unsigned sh_base_reg = sctx->shader_pointers.sh_base[PIPE_SHADER_VERTEX];
	bool render_cond_bit = sctx->render_cond && !sctx->render_cond_force_off;
	uint32_t index_max_size = 0;
	uint64_t index_va = 0;

	if (info->count_from_stream_output) {
		struct si_streamout_target *t =
			(struct si_streamout_target*)info->count_from_stream_output;
		uint64_t va = t->buf_filled_size->gpu_address +
			      t->buf_filled_size_offset;

		radeon_set_context_reg(cs, R_028B30_VGT_STRMOUT_DRAW_OPAQUE_VERTEX_STRIDE,
				       t->stride_in_dw);

		radeon_emit(cs, PKT3(PKT3_COPY_DATA, 4, 0));
		radeon_emit(cs, COPY_DATA_SRC_SEL(COPY_DATA_SRC_MEM) |
			    COPY_DATA_DST_SEL(COPY_DATA_REG) |
			    COPY_DATA_WR_CONFIRM);
		radeon_emit(cs, va);     /* src address lo */
		radeon_emit(cs, va >> 32); /* src address hi */
		radeon_emit(cs, R_028B2C_VGT_STRMOUT_DRAW_OPAQUE_BUFFER_FILLED_SIZE >> 2);
		radeon_emit(cs, 0); /* unused */

		radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
				      t->buf_filled_size, RADEON_USAGE_READ,
				      RADEON_PRIO_SO_FILLED_SIZE);
	}

	/* draw packet */
	if (index_size) {
		if (index_size != sctx->last_index_size) {
			unsigned index_type;

			/* index type */
			switch (index_size) {
			case 1:
				index_type = V_028A7C_VGT_INDEX_8;
				break;
			case 2:
				index_type = V_028A7C_VGT_INDEX_16 |
					     (SI_BIG_ENDIAN && sctx->chip_class <= CIK ?
						      V_028A7C_VGT_DMA_SWAP_16_BIT : 0);
				break;
			case 4:
				index_type = V_028A7C_VGT_INDEX_32 |
					     (SI_BIG_ENDIAN && sctx->chip_class <= CIK ?
						      V_028A7C_VGT_DMA_SWAP_32_BIT : 0);
				break;
			default:
				assert(!"unreachable");
				return;
			}

			if (sctx->chip_class >= GFX9) {
				radeon_set_uconfig_reg_idx(cs, R_03090C_VGT_INDEX_TYPE,
							   2, index_type);
			} else {
				radeon_emit(cs, PKT3(PKT3_INDEX_TYPE, 0, 0));
				radeon_emit(cs, index_type);
			}

			sctx->last_index_size = index_size;
		}

		index_max_size = (indexbuf->width0 - index_offset) /
				  index_size;
		index_va = r600_resource(indexbuf)->gpu_address + index_offset;

		radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
				      r600_resource(indexbuf),
				      RADEON_USAGE_READ, RADEON_PRIO_INDEX_BUFFER);
	} else {
		/* On CI and later, non-indexed draws overwrite VGT_INDEX_TYPE,
		 * so the state must be re-emitted before the next indexed draw.
		 */
		if (sctx->chip_class >= CIK)
			sctx->last_index_size = -1;
	}

	if (indirect) {
		uint64_t indirect_va = r600_resource(indirect->buffer)->gpu_address;

		assert(indirect_va % 8 == 0);

		si_invalidate_draw_sh_constants(sctx);

		radeon_emit(cs, PKT3(PKT3_SET_BASE, 2, 0));
		radeon_emit(cs, 1);
		radeon_emit(cs, indirect_va);
		radeon_emit(cs, indirect_va >> 32);

		radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
				      r600_resource(indirect->buffer),
				      RADEON_USAGE_READ, RADEON_PRIO_DRAW_INDIRECT);

		unsigned di_src_sel = index_size ? V_0287F0_DI_SRC_SEL_DMA
						    : V_0287F0_DI_SRC_SEL_AUTO_INDEX;

		assert(indirect->offset % 4 == 0);

		if (index_size) {
			radeon_emit(cs, PKT3(PKT3_INDEX_BASE, 1, 0));
			radeon_emit(cs, index_va);
			radeon_emit(cs, index_va >> 32);

			radeon_emit(cs, PKT3(PKT3_INDEX_BUFFER_SIZE, 0, 0));
			radeon_emit(cs, index_max_size);
		}

		if (!sctx->screen->has_draw_indirect_multi) {
			radeon_emit(cs, PKT3(index_size ? PKT3_DRAW_INDEX_INDIRECT
							   : PKT3_DRAW_INDIRECT,
					     3, render_cond_bit));
			radeon_emit(cs, indirect->offset);
			radeon_emit(cs, (sh_base_reg + SI_SGPR_BASE_VERTEX * 4 - SI_SH_REG_OFFSET) >> 2);
			radeon_emit(cs, (sh_base_reg + SI_SGPR_START_INSTANCE * 4 - SI_SH_REG_OFFSET) >> 2);
			radeon_emit(cs, di_src_sel);
		} else {
			uint64_t count_va = 0;

			if (indirect->indirect_draw_count) {
				struct r600_resource *params_buf =
					r600_resource(indirect->indirect_draw_count);

				radeon_add_to_buffer_list(
					sctx, sctx->gfx_cs, params_buf,
					RADEON_USAGE_READ, RADEON_PRIO_DRAW_INDIRECT);

				count_va = params_buf->gpu_address + indirect->indirect_draw_count_offset;
			}

			radeon_emit(cs, PKT3(index_size ? PKT3_DRAW_INDEX_INDIRECT_MULTI :
							     PKT3_DRAW_INDIRECT_MULTI,
					     8, render_cond_bit));
			radeon_emit(cs, indirect->offset);
			radeon_emit(cs, (sh_base_reg + SI_SGPR_BASE_VERTEX * 4 - SI_SH_REG_OFFSET) >> 2);
			radeon_emit(cs, (sh_base_reg + SI_SGPR_START_INSTANCE * 4 - SI_SH_REG_OFFSET) >> 2);
			radeon_emit(cs, ((sh_base_reg + SI_SGPR_DRAWID * 4 - SI_SH_REG_OFFSET) >> 2) |
					S_2C3_DRAW_INDEX_ENABLE(1) |
					S_2C3_COUNT_INDIRECT_ENABLE(!!indirect->indirect_draw_count));
			radeon_emit(cs, indirect->draw_count);
			radeon_emit(cs, count_va);
			radeon_emit(cs, count_va >> 32);
			radeon_emit(cs, indirect->stride);
			radeon_emit(cs, di_src_sel);
		}
	} else {
		int base_vertex;

		radeon_emit(cs, PKT3(PKT3_NUM_INSTANCES, 0, 0));
		radeon_emit(cs, info->instance_count);

		/* Base vertex and start instance. */
		base_vertex = index_size ? info->index_bias : info->start;

		if (sctx->num_vs_blit_sgprs) {
			/* Re-emit draw constants after we leave u_blitter. */
			si_invalidate_draw_sh_constants(sctx);

			/* Blit VS doesn't use BASE_VERTEX, START_INSTANCE, and DRAWID. */
			radeon_set_sh_reg_seq(cs, sh_base_reg + SI_SGPR_VS_BLIT_DATA * 4,
					      sctx->num_vs_blit_sgprs);
			radeon_emit_array(cs, sctx->vs_blit_sh_data,
					  sctx->num_vs_blit_sgprs);
		} else if (base_vertex != sctx->last_base_vertex ||
			   sctx->last_base_vertex == SI_BASE_VERTEX_UNKNOWN ||
			   info->start_instance != sctx->last_start_instance ||
			   info->drawid != sctx->last_drawid ||
			   sh_base_reg != sctx->last_sh_base_reg) {
			radeon_set_sh_reg_seq(cs, sh_base_reg + SI_SGPR_BASE_VERTEX * 4, 3);
			radeon_emit(cs, base_vertex);
			radeon_emit(cs, info->start_instance);
			radeon_emit(cs, info->drawid);

			sctx->last_base_vertex = base_vertex;
			sctx->last_start_instance = info->start_instance;
			sctx->last_drawid = info->drawid;
			sctx->last_sh_base_reg = sh_base_reg;
		}

		if (index_size) {
			index_va += info->start * index_size;

			radeon_emit(cs, PKT3(PKT3_DRAW_INDEX_2, 4, render_cond_bit));
			radeon_emit(cs, index_max_size);
			radeon_emit(cs, index_va);
			radeon_emit(cs, index_va >> 32);
			radeon_emit(cs, info->count);
			radeon_emit(cs, V_0287F0_DI_SRC_SEL_DMA);
		} else {
			radeon_emit(cs, PKT3(PKT3_DRAW_INDEX_AUTO, 1, render_cond_bit));
			radeon_emit(cs, info->count);
			radeon_emit(cs, V_0287F0_DI_SRC_SEL_AUTO_INDEX |
				        S_0287F0_USE_OPAQUE(!!info->count_from_stream_output));
		}
	}
}

static void si_emit_surface_sync(struct si_context *sctx,
				 unsigned cp_coher_cntl)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;

	if (sctx->chip_class >= GFX9) {
		/* Flush caches and wait for the caches to assert idle. */
		radeon_emit(cs, PKT3(PKT3_ACQUIRE_MEM, 5, 0));
		radeon_emit(cs, cp_coher_cntl);	/* CP_COHER_CNTL */
		radeon_emit(cs, 0xffffffff);	/* CP_COHER_SIZE */
		radeon_emit(cs, 0xffffff);	/* CP_COHER_SIZE_HI */
		radeon_emit(cs, 0);		/* CP_COHER_BASE */
		radeon_emit(cs, 0);		/* CP_COHER_BASE_HI */
		radeon_emit(cs, 0x0000000A);	/* POLL_INTERVAL */
	} else {
		/* ACQUIRE_MEM is only required on a compute ring. */
		radeon_emit(cs, PKT3(PKT3_SURFACE_SYNC, 3, 0));
		radeon_emit(cs, cp_coher_cntl);   /* CP_COHER_CNTL */
		radeon_emit(cs, 0xffffffff);      /* CP_COHER_SIZE */
		radeon_emit(cs, 0);               /* CP_COHER_BASE */
		radeon_emit(cs, 0x0000000A);      /* POLL_INTERVAL */
	}
}

void si_emit_cache_flush(struct si_context *sctx)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	uint32_t flags = sctx->flags;
	uint32_t cp_coher_cntl = 0;
	uint32_t flush_cb_db = flags & (SI_CONTEXT_FLUSH_AND_INV_CB |
					SI_CONTEXT_FLUSH_AND_INV_DB);

	if (flags & SI_CONTEXT_FLUSH_AND_INV_CB)
		sctx->num_cb_cache_flushes++;
	if (flags & SI_CONTEXT_FLUSH_AND_INV_DB)
		sctx->num_db_cache_flushes++;

	/* SI has a bug that it always flushes ICACHE and KCACHE if either
	 * bit is set. An alternative way is to write SQC_CACHES, but that
	 * doesn't seem to work reliably. Since the bug doesn't affect
	 * correctness (it only does more work than necessary) and
	 * the performance impact is likely negligible, there is no plan
	 * to add a workaround for it.
	 */

	if (flags & SI_CONTEXT_INV_ICACHE)
		cp_coher_cntl |= S_0085F0_SH_ICACHE_ACTION_ENA(1);
	if (flags & SI_CONTEXT_INV_SMEM_L1)
		cp_coher_cntl |= S_0085F0_SH_KCACHE_ACTION_ENA(1);

	if (sctx->chip_class <= VI) {
		if (flags & SI_CONTEXT_FLUSH_AND_INV_CB) {
			cp_coher_cntl |= S_0085F0_CB_ACTION_ENA(1) |
					 S_0085F0_CB0_DEST_BASE_ENA(1) |
					 S_0085F0_CB1_DEST_BASE_ENA(1) |
					 S_0085F0_CB2_DEST_BASE_ENA(1) |
					 S_0085F0_CB3_DEST_BASE_ENA(1) |
					 S_0085F0_CB4_DEST_BASE_ENA(1) |
					 S_0085F0_CB5_DEST_BASE_ENA(1) |
					 S_0085F0_CB6_DEST_BASE_ENA(1) |
					 S_0085F0_CB7_DEST_BASE_ENA(1);

			/* Necessary for DCC */
			if (sctx->chip_class == VI)
				si_cp_release_mem(sctx,
						  V_028A90_FLUSH_AND_INV_CB_DATA_TS,
						  0, EOP_DST_SEL_MEM, EOP_INT_SEL_NONE,
						  EOP_DATA_SEL_DISCARD, NULL,
						  0, 0, SI_NOT_QUERY);
		}
		if (flags & SI_CONTEXT_FLUSH_AND_INV_DB)
			cp_coher_cntl |= S_0085F0_DB_ACTION_ENA(1) |
					 S_0085F0_DB_DEST_BASE_ENA(1);
	}

	if (flags & SI_CONTEXT_FLUSH_AND_INV_CB) {
		/* Flush CMASK/FMASK/DCC. SURFACE_SYNC will wait for idle. */
		radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_FLUSH_AND_INV_CB_META) | EVENT_INDEX(0));
	}
	if (flags & (SI_CONTEXT_FLUSH_AND_INV_DB |
		     SI_CONTEXT_FLUSH_AND_INV_DB_META)) {
		/* Flush HTILE. SURFACE_SYNC will wait for idle. */
		radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_FLUSH_AND_INV_DB_META) | EVENT_INDEX(0));
	}

	/* Wait for shader engines to go idle.
	 * VS and PS waits are unnecessary if SURFACE_SYNC is going to wait
	 * for everything including CB/DB cache flushes.
	 */
	if (!flush_cb_db) {
		if (flags & SI_CONTEXT_PS_PARTIAL_FLUSH) {
			radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
			radeon_emit(cs, EVENT_TYPE(V_028A90_PS_PARTIAL_FLUSH) | EVENT_INDEX(4));
			/* Only count explicit shader flushes, not implicit ones
			 * done by SURFACE_SYNC.
			 */
			sctx->num_vs_flushes++;
			sctx->num_ps_flushes++;
		} else if (flags & SI_CONTEXT_VS_PARTIAL_FLUSH) {
			radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
			radeon_emit(cs, EVENT_TYPE(V_028A90_VS_PARTIAL_FLUSH) | EVENT_INDEX(4));
			sctx->num_vs_flushes++;
		}
	}

	if (flags & SI_CONTEXT_CS_PARTIAL_FLUSH &&
	    sctx->compute_is_busy) {
		radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_CS_PARTIAL_FLUSH) | EVENT_INDEX(4));
		sctx->num_cs_flushes++;
		sctx->compute_is_busy = false;
	}

	/* VGT state synchronization. */
	if (flags & SI_CONTEXT_VGT_FLUSH) {
		radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_VGT_FLUSH) | EVENT_INDEX(0));
	}
	if (flags & SI_CONTEXT_VGT_STREAMOUT_SYNC) {
		radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_VGT_STREAMOUT_SYNC) | EVENT_INDEX(0));
	}

	/* GFX9: Wait for idle if we're flushing CB or DB. ACQUIRE_MEM doesn't
	 * wait for idle on GFX9. We have to use a TS event.
	 */
	if (sctx->chip_class >= GFX9 && flush_cb_db) {
		uint64_t va;
		unsigned tc_flags, cb_db_event;

		/* Set the CB/DB flush event. */
		switch (flush_cb_db) {
		case SI_CONTEXT_FLUSH_AND_INV_CB:
			cb_db_event = V_028A90_FLUSH_AND_INV_CB_DATA_TS;
			break;
		case SI_CONTEXT_FLUSH_AND_INV_DB:
			cb_db_event = V_028A90_FLUSH_AND_INV_DB_DATA_TS;
			break;
		default:
			/* both CB & DB */
			cb_db_event = V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT;
		}

		/* These are the only allowed combinations. If you need to
		 * do multiple operations at once, do them separately.
		 * All operations that invalidate L2 also seem to invalidate
		 * metadata. Volatile (VOL) and WC flushes are not listed here.
		 *
		 * TC    | TC_WB         = writeback & invalidate L2 & L1
		 * TC    | TC_WB | TC_NC = writeback & invalidate L2 for MTYPE == NC
		 *         TC_WB | TC_NC = writeback L2 for MTYPE == NC
		 * TC            | TC_NC = invalidate L2 for MTYPE == NC
		 * TC    | TC_MD         = writeback & invalidate L2 metadata (DCC, etc.)
		 * TCL1                  = invalidate L1
		 */
		tc_flags = 0;

		if (flags & SI_CONTEXT_INV_L2_METADATA) {
			tc_flags = EVENT_TC_ACTION_ENA |
				   EVENT_TC_MD_ACTION_ENA;
		}

		/* Ideally flush TC together with CB/DB. */
		if (flags & SI_CONTEXT_INV_GLOBAL_L2) {
			/* Writeback and invalidate everything in L2 & L1. */
			tc_flags = EVENT_TC_ACTION_ENA |
				   EVENT_TC_WB_ACTION_ENA;

			/* Clear the flags. */
			flags &= ~(SI_CONTEXT_INV_GLOBAL_L2 |
				   SI_CONTEXT_WRITEBACK_GLOBAL_L2 |
				   SI_CONTEXT_INV_VMEM_L1);
			sctx->num_L2_invalidates++;
		}

		/* Do the flush (enqueue the event and wait for it). */
		va = sctx->wait_mem_scratch->gpu_address;
		sctx->wait_mem_number++;

		si_cp_release_mem(sctx, cb_db_event, tc_flags,
				  EOP_DST_SEL_MEM,
				  EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
				  EOP_DATA_SEL_VALUE_32BIT,
				  sctx->wait_mem_scratch, va,
				  sctx->wait_mem_number, SI_NOT_QUERY);
		si_cp_wait_mem(sctx, va, sctx->wait_mem_number, 0xffffffff, 0);
	}

	/* Make sure ME is idle (it executes most packets) before continuing.
	 * This prevents read-after-write hazards between PFP and ME.
	 */
	if (cp_coher_cntl ||
	    (flags & (SI_CONTEXT_CS_PARTIAL_FLUSH |
			    SI_CONTEXT_INV_VMEM_L1 |
			    SI_CONTEXT_INV_GLOBAL_L2 |
			    SI_CONTEXT_WRITEBACK_GLOBAL_L2))) {
		radeon_emit(cs, PKT3(PKT3_PFP_SYNC_ME, 0, 0));
		radeon_emit(cs, 0);
	}

	/* SI-CI-VI only:
	 *   When one of the CP_COHER_CNTL.DEST_BASE flags is set, SURFACE_SYNC
	 *   waits for idle, so it should be last. SURFACE_SYNC is done in PFP.
	 *
	 * cp_coher_cntl should contain all necessary flags except TC flags
	 * at this point.
	 *
	 * SI-CIK don't support L2 write-back.
	 */
	if (flags & SI_CONTEXT_INV_GLOBAL_L2 ||
	    (sctx->chip_class <= CIK &&
	     (flags & SI_CONTEXT_WRITEBACK_GLOBAL_L2))) {
		/* Invalidate L1 & L2. (L1 is always invalidated on SI)
		 * WB must be set on VI+ when TC_ACTION is set.
		 */
		si_emit_surface_sync(sctx, cp_coher_cntl |
				     S_0085F0_TC_ACTION_ENA(1) |
				     S_0085F0_TCL1_ACTION_ENA(1) |
				     S_0301F0_TC_WB_ACTION_ENA(sctx->chip_class >= VI));
		cp_coher_cntl = 0;
		sctx->num_L2_invalidates++;
	} else {
		/* L1 invalidation and L2 writeback must be done separately,
		 * because both operations can't be done together.
		 */
		if (flags & SI_CONTEXT_WRITEBACK_GLOBAL_L2) {
			/* WB = write-back
			 * NC = apply to non-coherent MTYPEs
			 *      (i.e. MTYPE <= 1, which is what we use everywhere)
			 *
			 * WB doesn't work without NC.
			 */
			si_emit_surface_sync(sctx, cp_coher_cntl |
					     S_0301F0_TC_WB_ACTION_ENA(1) |
					     S_0301F0_TC_NC_ACTION_ENA(1));
			cp_coher_cntl = 0;
			sctx->num_L2_writebacks++;
		}
		if (flags & SI_CONTEXT_INV_VMEM_L1) {
			/* Invalidate per-CU VMEM L1. */
			si_emit_surface_sync(sctx, cp_coher_cntl |
					     S_0085F0_TCL1_ACTION_ENA(1));
			cp_coher_cntl = 0;
		}
	}

	/* If TC flushes haven't cleared this... */
	if (cp_coher_cntl)
		si_emit_surface_sync(sctx, cp_coher_cntl);

	if (flags & SI_CONTEXT_START_PIPELINE_STATS) {
		radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_PIPELINESTAT_START) |
			        EVENT_INDEX(0));
	} else if (flags & SI_CONTEXT_STOP_PIPELINE_STATS) {
		radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
		radeon_emit(cs, EVENT_TYPE(V_028A90_PIPELINESTAT_STOP) |
			        EVENT_INDEX(0));
	}

	sctx->flags = 0;
}

static void si_get_draw_start_count(struct si_context *sctx,
				    const struct pipe_draw_info *info,
				    unsigned *start, unsigned *count)
{
	struct pipe_draw_indirect_info *indirect = info->indirect;

	if (indirect) {
		unsigned indirect_count;
		struct pipe_transfer *transfer;
		unsigned begin, end;
		unsigned map_size;
		unsigned *data;

		if (indirect->indirect_draw_count) {
			data = pipe_buffer_map_range(&sctx->b,
					indirect->indirect_draw_count,
					indirect->indirect_draw_count_offset,
					sizeof(unsigned),
					PIPE_TRANSFER_READ, &transfer);

			indirect_count = *data;

			pipe_buffer_unmap(&sctx->b, transfer);
		} else {
			indirect_count = indirect->draw_count;
		}

		if (!indirect_count) {
			*start = *count = 0;
			return;
		}

		map_size = (indirect_count - 1) * indirect->stride + 3 * sizeof(unsigned);
		data = pipe_buffer_map_range(&sctx->b, indirect->buffer,
					     indirect->offset, map_size,
					     PIPE_TRANSFER_READ, &transfer);

		begin = UINT_MAX;
		end = 0;

		for (unsigned i = 0; i < indirect_count; ++i) {
			unsigned count = data[0];
			unsigned start = data[2];

			if (count > 0) {
				begin = MIN2(begin, start);
				end = MAX2(end, start + count);
			}

			data += indirect->stride / sizeof(unsigned);
		}

		pipe_buffer_unmap(&sctx->b, transfer);

		if (begin < end) {
			*start = begin;
			*count = end - begin;
		} else {
			*start = *count = 0;
		}
	} else {
		*start = info->start;
		*count = info->count;
	}
}

static void si_emit_all_states(struct si_context *sctx, const struct pipe_draw_info *info,
			       unsigned skip_atom_mask)
{
	unsigned num_patches = 0;
	/* Vega10/Raven scissor bug workaround. When any context register is
	 * written (i.e. the GPU rolls the context), PA_SC_VPORT_SCISSOR
	 * registers must be written too.
	 */
	bool handle_scissor_bug = (sctx->family == CHIP_VEGA10 || sctx->family == CHIP_RAVEN) &&
				  !si_is_atom_dirty(sctx, &sctx->atoms.s.scissors);
	bool context_roll = false; /* set correctly for GFX9 only */

	context_roll |= si_emit_rasterizer_prim_state(sctx);
	if (sctx->tes_shader.cso)
		context_roll |= si_emit_derived_tess_state(sctx, info, &num_patches);

	if (handle_scissor_bug &&
	    (info->count_from_stream_output ||
	     sctx->dirty_atoms & si_atoms_that_always_roll_context() ||
	     sctx->dirty_states & si_states_that_always_roll_context() ||
	     si_prim_restart_index_changed(sctx, info)))
		context_roll = true;

	sctx->context_roll_counter = 0;

	/* Emit state atoms. */
	unsigned mask = sctx->dirty_atoms & ~skip_atom_mask;
	while (mask)
		sctx->atoms.array[u_bit_scan(&mask)].emit(sctx);

	sctx->dirty_atoms &= skip_atom_mask;

	/* Emit states. */
	mask = sctx->dirty_states;
	while (mask) {
		unsigned i = u_bit_scan(&mask);
		struct si_pm4_state *state = sctx->queued.array[i];

		if (!state || sctx->emitted.array[i] == state)
			continue;

		si_pm4_emit(sctx, state);
		sctx->emitted.array[i] = state;
	}
	sctx->dirty_states = 0;

	if (handle_scissor_bug &&
	    (context_roll || sctx->context_roll_counter)) {
		sctx->scissors.dirty_mask = (1 << SI_MAX_VIEWPORTS) - 1;
		sctx->atoms.s.scissors.emit(sctx);
	}

	/* Emit draw states. */
	si_emit_vs_state(sctx, info);
	si_emit_draw_registers(sctx, info, num_patches);
}

void si_draw_vbo(struct pipe_context *ctx, const struct pipe_draw_info *info)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_state_rasterizer *rs = sctx->queued.named.rasterizer;
	struct pipe_resource *indexbuf = info->index.resource;
	unsigned dirty_tex_counter;
	enum pipe_prim_type rast_prim;
	unsigned index_size = info->index_size;
	unsigned index_offset = info->indirect ? info->start * index_size : 0;

	if (likely(!info->indirect)) {
		/* SI-CI treat instance_count==0 as instance_count==1. There is
		 * no workaround for indirect draws, but we can at least skip
		 * direct draws.
		 */
		if (unlikely(!info->instance_count))
			return;

		/* Handle count == 0. */
		if (unlikely(!info->count &&
			     (index_size || !info->count_from_stream_output)))
			return;
	}

	if (unlikely(!sctx->vs_shader.cso ||
		     !rs ||
		     (!sctx->ps_shader.cso && !rs->rasterizer_discard) ||
		     (!!sctx->tes_shader.cso != (info->mode == PIPE_PRIM_PATCHES)))) {
		assert(0);
		return;
	}

	/* Recompute and re-emit the texture resource states if needed. */
	dirty_tex_counter = p_atomic_read(&sctx->screen->dirty_tex_counter);
	if (unlikely(dirty_tex_counter != sctx->last_dirty_tex_counter)) {
		sctx->last_dirty_tex_counter = dirty_tex_counter;
		sctx->framebuffer.dirty_cbufs |=
			((1 << sctx->framebuffer.state.nr_cbufs) - 1);
		sctx->framebuffer.dirty_zsbuf = true;
		si_mark_atom_dirty(sctx, &sctx->atoms.s.framebuffer);
		si_update_all_texture_descriptors(sctx);
	}

	si_decompress_textures(sctx, u_bit_consecutive(0, SI_NUM_GRAPHICS_SHADERS));

	/* Set the rasterization primitive type.
	 *
	 * This must be done after si_decompress_textures, which can call
	 * draw_vbo recursively, and before si_update_shaders, which uses
	 * current_rast_prim for this draw_vbo call. */
	if (sctx->gs_shader.cso)
		rast_prim = sctx->gs_shader.cso->gs_output_prim;
	else if (sctx->tes_shader.cso) {
		if (sctx->tes_shader.cso->info.properties[TGSI_PROPERTY_TES_POINT_MODE])
			rast_prim = PIPE_PRIM_POINTS;
		else
			rast_prim = sctx->tes_shader.cso->info.properties[TGSI_PROPERTY_TES_PRIM_MODE];
	} else
		rast_prim = info->mode;

	if (rast_prim != sctx->current_rast_prim) {
		if (util_prim_is_points_or_lines(sctx->current_rast_prim) !=
		    util_prim_is_points_or_lines(rast_prim))
			si_mark_atom_dirty(sctx, &sctx->atoms.s.guardband);

		sctx->current_rast_prim = rast_prim;
		sctx->do_update_shaders = true;
	}

	if (sctx->tes_shader.cso &&
	    sctx->screen->has_ls_vgpr_init_bug) {
		/* Determine whether the LS VGPR fix should be applied.
		 *
		 * It is only required when num input CPs > num output CPs,
		 * which cannot happen with the fixed function TCS. We should
		 * also update this bit when switching from TCS to fixed
		 * function TCS.
		 */
		struct si_shader_selector *tcs = sctx->tcs_shader.cso;
		bool ls_vgpr_fix =
			tcs &&
			info->vertices_per_patch >
			tcs->info.properties[TGSI_PROPERTY_TCS_VERTICES_OUT];

		if (ls_vgpr_fix != sctx->ls_vgpr_fix) {
			sctx->ls_vgpr_fix = ls_vgpr_fix;
			sctx->do_update_shaders = true;
		}
	}

	if (sctx->gs_shader.cso) {
		/* Determine whether the GS triangle strip adjacency fix should
		 * be applied. Rotate every other triangle if
		 * - triangle strips with adjacency are fed to the GS and
		 * - primitive restart is disabled (the rotation doesn't help
		 *   when the restart occurs after an odd number of triangles).
		 */
		bool gs_tri_strip_adj_fix =
			!sctx->tes_shader.cso &&
			info->mode == PIPE_PRIM_TRIANGLE_STRIP_ADJACENCY &&
			!info->primitive_restart;

		if (gs_tri_strip_adj_fix != sctx->gs_tri_strip_adj_fix) {
			sctx->gs_tri_strip_adj_fix = gs_tri_strip_adj_fix;
			sctx->do_update_shaders = true;
		}
	}

	if (sctx->do_update_shaders && !si_update_shaders(sctx))
		return;

	if (index_size) {
		/* Translate or upload, if needed. */
		/* 8-bit indices are supported on VI. */
		if (sctx->chip_class <= CIK && index_size == 1) {
			unsigned start, count, start_offset, size, offset;
			void *ptr;

			si_get_draw_start_count(sctx, info, &start, &count);
			start_offset = start * 2;
			size = count * 2;

			indexbuf = NULL;
			u_upload_alloc(ctx->stream_uploader, start_offset,
				       size,
				       si_optimal_tcc_alignment(sctx, size),
				       &offset, &indexbuf, &ptr);
			if (!indexbuf)
				return;

			util_shorten_ubyte_elts_to_userptr(&sctx->b, info, 0, 0,
							   index_offset + start,
							   count, ptr);

			/* info->start will be added by the drawing code */
			index_offset = offset - start_offset;
			index_size = 2;
		} else if (info->has_user_indices) {
			unsigned start_offset;

			assert(!info->indirect);
			start_offset = info->start * index_size;

			indexbuf = NULL;
			u_upload_data(ctx->stream_uploader, start_offset,
				      info->count * index_size,
				      sctx->screen->info.tcc_cache_line_size,
				      (char*)info->index.user + start_offset,
				      &index_offset, &indexbuf);
			if (!indexbuf)
				return;

			/* info->start will be added by the drawing code */
			index_offset -= start_offset;
		} else if (sctx->chip_class <= CIK &&
			   r600_resource(indexbuf)->TC_L2_dirty) {
			/* VI reads index buffers through TC L2, so it doesn't
			 * need this. */
			sctx->flags |= SI_CONTEXT_WRITEBACK_GLOBAL_L2;
			r600_resource(indexbuf)->TC_L2_dirty = false;
		}
	}

	if (info->indirect) {
		struct pipe_draw_indirect_info *indirect = info->indirect;

		/* Add the buffer size for memory checking in need_cs_space. */
		si_context_add_resource_size(sctx, indirect->buffer);

		/* Indirect buffers use TC L2 on GFX9, but not older hw. */
		if (sctx->chip_class <= VI) {
			if (r600_resource(indirect->buffer)->TC_L2_dirty) {
				sctx->flags |= SI_CONTEXT_WRITEBACK_GLOBAL_L2;
				r600_resource(indirect->buffer)->TC_L2_dirty = false;
			}

			if (indirect->indirect_draw_count &&
			    r600_resource(indirect->indirect_draw_count)->TC_L2_dirty) {
				sctx->flags |= SI_CONTEXT_WRITEBACK_GLOBAL_L2;
				r600_resource(indirect->indirect_draw_count)->TC_L2_dirty = false;
			}
		}
	}

	si_need_gfx_cs_space(sctx);

	/* Since we've called si_context_add_resource_size for vertex buffers,
	 * this must be called after si_need_cs_space, because we must let
	 * need_cs_space flush before we add buffers to the buffer list.
	 */
	if (!si_upload_vertex_buffer_descriptors(sctx))
		return;

	/* Use optimal packet order based on whether we need to sync the pipeline. */
	if (unlikely(sctx->flags & (SI_CONTEXT_FLUSH_AND_INV_CB |
				      SI_CONTEXT_FLUSH_AND_INV_DB |
				      SI_CONTEXT_PS_PARTIAL_FLUSH |
				      SI_CONTEXT_CS_PARTIAL_FLUSH))) {
		/* If we have to wait for idle, set all states first, so that all
		 * SET packets are processed in parallel with previous draw calls.
		 * Then draw and prefetch at the end. This ensures that the time
		 * the CUs are idle is very short.
		 */
		unsigned masked_atoms = 0;

		if (unlikely(sctx->flags & SI_CONTEXT_FLUSH_FOR_RENDER_COND))
			masked_atoms |= si_get_atom_bit(sctx, &sctx->atoms.s.render_cond);

		if (!si_upload_graphics_shader_descriptors(sctx))
			return;

		/* Emit all states except possibly render condition. */
		si_emit_all_states(sctx, info, masked_atoms);
		si_emit_cache_flush(sctx);
		/* <-- CUs are idle here. */

		if (si_is_atom_dirty(sctx, &sctx->atoms.s.render_cond))
			sctx->atoms.s.render_cond.emit(sctx);
		sctx->dirty_atoms = 0;

		si_emit_draw_packets(sctx, info, indexbuf, index_size, index_offset);
		/* <-- CUs are busy here. */

		/* Start prefetches after the draw has been started. Both will run
		 * in parallel, but starting the draw first is more important.
		 */
		if (sctx->chip_class >= CIK && sctx->prefetch_L2_mask)
			cik_emit_prefetch_L2(sctx, false);
	} else {
		/* If we don't wait for idle, start prefetches first, then set
		 * states, and draw at the end.
		 */
		if (sctx->flags)
			si_emit_cache_flush(sctx);

		/* Only prefetch the API VS and VBO descriptors. */
		if (sctx->chip_class >= CIK && sctx->prefetch_L2_mask)
			cik_emit_prefetch_L2(sctx, true);

		if (!si_upload_graphics_shader_descriptors(sctx))
			return;

		si_emit_all_states(sctx, info, 0);
		si_emit_draw_packets(sctx, info, indexbuf, index_size, index_offset);

		/* Prefetch the remaining shaders after the draw has been
		 * started. */
		if (sctx->chip_class >= CIK && sctx->prefetch_L2_mask)
			cik_emit_prefetch_L2(sctx, false);
	}

	if (unlikely(sctx->current_saved_cs)) {
		si_trace_emit(sctx);
		si_log_draw_state(sctx, sctx->log);
	}

	/* Workaround for a VGT hang when streamout is enabled.
	 * It must be done after drawing. */
	if ((sctx->family == CHIP_HAWAII ||
	     sctx->family == CHIP_TONGA ||
	     sctx->family == CHIP_FIJI) &&
	    si_get_strmout_en(sctx)) {
		sctx->flags |= SI_CONTEXT_VGT_STREAMOUT_SYNC;
	}

	if (unlikely(sctx->decompression_enabled)) {
		sctx->num_decompress_calls++;
	} else {
		sctx->num_draw_calls++;
		if (sctx->framebuffer.state.nr_cbufs > 1)
			sctx->num_mrt_draw_calls++;
		if (info->primitive_restart)
			sctx->num_prim_restart_calls++;
		if (G_0286E8_WAVESIZE(sctx->spi_tmpring_size))
			sctx->num_spill_draw_calls++;
	}
	if (index_size && indexbuf != info->index.resource)
		pipe_resource_reference(&indexbuf, NULL);
}

void si_draw_rectangle(struct blitter_context *blitter,
		       void *vertex_elements_cso,
		       blitter_get_vs_func get_vs,
		       int x1, int y1, int x2, int y2,
		       float depth, unsigned num_instances,
		       enum blitter_attrib_type type,
		       const union blitter_attrib *attrib)
{
	struct pipe_context *pipe = util_blitter_get_pipe(blitter);
	struct si_context *sctx = (struct si_context*)pipe;

	/* Pack position coordinates as signed int16. */
	sctx->vs_blit_sh_data[0] = (uint32_t)(x1 & 0xffff) |
				   ((uint32_t)(y1 & 0xffff) << 16);
	sctx->vs_blit_sh_data[1] = (uint32_t)(x2 & 0xffff) |
				   ((uint32_t)(y2 & 0xffff) << 16);
	sctx->vs_blit_sh_data[2] = fui(depth);

	switch (type) {
	case UTIL_BLITTER_ATTRIB_COLOR:
		memcpy(&sctx->vs_blit_sh_data[3], attrib->color,
		       sizeof(float)*4);
		break;
	case UTIL_BLITTER_ATTRIB_TEXCOORD_XY:
	case UTIL_BLITTER_ATTRIB_TEXCOORD_XYZW:
		memcpy(&sctx->vs_blit_sh_data[3], &attrib->texcoord,
		       sizeof(attrib->texcoord));
		break;
	case UTIL_BLITTER_ATTRIB_NONE:;
	}

	pipe->bind_vs_state(pipe, si_get_blitter_vs(sctx, type, num_instances));

	struct pipe_draw_info info = {};
	info.mode = SI_PRIM_RECTANGLE_LIST;
	info.count = 3;
	info.instance_count = num_instances;

	/* Don't set per-stage shader pointers for VS. */
	sctx->shader_pointers_dirty &= ~SI_DESCS_SHADER_MASK(VERTEX);
	sctx->vertex_buffer_pointer_dirty = false;

	si_draw_vbo(pipe, &info);
}

void si_trace_emit(struct si_context *sctx)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	uint64_t va = sctx->current_saved_cs->trace_buf->gpu_address;
	uint32_t trace_id = ++sctx->current_saved_cs->trace_id;

	radeon_emit(cs, PKT3(PKT3_WRITE_DATA, 3, 0));
	radeon_emit(cs, S_370_DST_SEL(V_370_MEMORY_SYNC) |
		    S_370_WR_CONFIRM(1) |
		    S_370_ENGINE_SEL(V_370_ME));
	radeon_emit(cs, va);
	radeon_emit(cs, va >> 32);
	radeon_emit(cs, trace_id);
	radeon_emit(cs, PKT3(PKT3_NOP, 0, 0));
	radeon_emit(cs, AC_ENCODE_TRACE_POINT(trace_id));

	if (sctx->log)
		u_log_flush(sctx->log);
}
