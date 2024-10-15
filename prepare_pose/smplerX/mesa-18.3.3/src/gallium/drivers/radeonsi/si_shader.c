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

#include "util/u_memory.h"
#include "util/u_string.h"
#include "tgsi/tgsi_build.h"
#include "tgsi/tgsi_util.h"
#include "tgsi/tgsi_dump.h"

#include "ac_exp_param.h"
#include "ac_shader_util.h"
#include "ac_llvm_util.h"
#include "si_shader_internal.h"
#include "si_pipe.h"
#include "sid.h"

#include "compiler/nir/nir.h"

static const char *scratch_rsrc_dword0_symbol =
	"SCRATCH_RSRC_DWORD0";

static const char *scratch_rsrc_dword1_symbol =
	"SCRATCH_RSRC_DWORD1";

struct si_shader_output_values
{
	LLVMValueRef values[4];
	unsigned semantic_name;
	unsigned semantic_index;
	ubyte vertex_stream[4];
};

/**
 * Used to collect types and other info about arguments of the LLVM function
 * before the function is created.
 */
struct si_function_info {
	LLVMTypeRef types[100];
	LLVMValueRef *assign[100];
	unsigned num_sgpr_params;
	unsigned num_params;
};

enum si_arg_regfile {
	ARG_SGPR,
	ARG_VGPR
};

static void si_init_shader_ctx(struct si_shader_context *ctx,
			       struct si_screen *sscreen,
			       struct ac_llvm_compiler *compiler);

static void si_llvm_emit_barrier(const struct lp_build_tgsi_action *action,
				 struct lp_build_tgsi_context *bld_base,
				 struct lp_build_emit_data *emit_data);

static void si_dump_shader_key(unsigned processor, const struct si_shader *shader,
			       FILE *f);

static void si_build_vs_prolog_function(struct si_shader_context *ctx,
					union si_shader_part_key *key);
static void si_build_tcs_epilog_function(struct si_shader_context *ctx,
					 union si_shader_part_key *key);
static void si_build_ps_prolog_function(struct si_shader_context *ctx,
					union si_shader_part_key *key);
static void si_build_ps_epilog_function(struct si_shader_context *ctx,
					union si_shader_part_key *key);

/* Ideally pass the sample mask input to the PS epilog as v14, which
 * is its usual location, so that the shader doesn't have to add v_mov.
 */
#define PS_EPILOG_SAMPLEMASK_MIN_LOC 14

static bool llvm_type_is_64bit(struct si_shader_context *ctx,
			       LLVMTypeRef type)
{
	if (type == ctx->ac.i64 || type == ctx->ac.f64)
		return true;

	return false;
}

static bool is_merged_shader(struct si_shader_context *ctx)
{
	if (ctx->screen->info.chip_class <= VI)
		return false;

	return ctx->shader->key.as_ls ||
	       ctx->shader->key.as_es ||
	       ctx->type == PIPE_SHADER_TESS_CTRL ||
	       ctx->type == PIPE_SHADER_GEOMETRY;
}

static void si_init_function_info(struct si_function_info *fninfo)
{
	fninfo->num_params = 0;
	fninfo->num_sgpr_params = 0;
}

static unsigned add_arg_assign(struct si_function_info *fninfo,
			enum si_arg_regfile regfile, LLVMTypeRef type,
			LLVMValueRef *assign)
{
	assert(regfile != ARG_SGPR || fninfo->num_sgpr_params == fninfo->num_params);

	unsigned idx = fninfo->num_params++;
	assert(idx < ARRAY_SIZE(fninfo->types));

	if (regfile == ARG_SGPR)
		fninfo->num_sgpr_params = fninfo->num_params;

	fninfo->types[idx] = type;
	fninfo->assign[idx] = assign;
	return idx;
}

static unsigned add_arg(struct si_function_info *fninfo,
			enum si_arg_regfile regfile, LLVMTypeRef type)
{
	return add_arg_assign(fninfo, regfile, type, NULL);
}

static void add_arg_assign_checked(struct si_function_info *fninfo,
				   enum si_arg_regfile regfile, LLVMTypeRef type,
				   LLVMValueRef *assign, unsigned idx)
{
	MAYBE_UNUSED unsigned actual = add_arg_assign(fninfo, regfile, type, assign);
	assert(actual == idx);
}

static void add_arg_checked(struct si_function_info *fninfo,
			    enum si_arg_regfile regfile, LLVMTypeRef type,
			    unsigned idx)
{
	add_arg_assign_checked(fninfo, regfile, type, NULL, idx);
}

/**
 * Returns a unique index for a per-patch semantic name and index. The index
 * must be less than 32, so that a 32-bit bitmask of used inputs or outputs
 * can be calculated.
 */
unsigned si_shader_io_get_unique_index_patch(unsigned semantic_name, unsigned index)
{
	switch (semantic_name) {
	case TGSI_SEMANTIC_TESSOUTER:
		return 0;
	case TGSI_SEMANTIC_TESSINNER:
		return 1;
	case TGSI_SEMANTIC_PATCH:
		assert(index < 30);
		return 2 + index;

	default:
		assert(!"invalid semantic name");
		return 0;
	}
}

/**
 * Returns a unique index for a semantic name and index. The index must be
 * less than 64, so that a 64-bit bitmask of used inputs or outputs can be
 * calculated.
 */
unsigned si_shader_io_get_unique_index(unsigned semantic_name, unsigned index,
				       unsigned is_varying)
{
	switch (semantic_name) {
	case TGSI_SEMANTIC_POSITION:
		return 0;
	case TGSI_SEMANTIC_GENERIC:
		/* Since some shader stages use the the highest used IO index
		 * to determine the size to allocate for inputs/outputs
		 * (in LDS, tess and GS rings). GENERIC should be placed right
		 * after POSITION to make that size as small as possible.
		 */
		if (index < SI_MAX_IO_GENERIC)
			return 1 + index;

		assert(!"invalid generic index");
		return 0;
	case TGSI_SEMANTIC_PSIZE:
		return SI_MAX_IO_GENERIC + 1;
	case TGSI_SEMANTIC_CLIPDIST:
		assert(index <= 1);
		return SI_MAX_IO_GENERIC + 2 + index;
	case TGSI_SEMANTIC_FOG:
		return SI_MAX_IO_GENERIC + 4;
	case TGSI_SEMANTIC_LAYER:
		return SI_MAX_IO_GENERIC + 5;
	case TGSI_SEMANTIC_VIEWPORT_INDEX:
		return SI_MAX_IO_GENERIC + 6;
	case TGSI_SEMANTIC_PRIMID:
		return SI_MAX_IO_GENERIC + 7;
	case TGSI_SEMANTIC_COLOR:
		assert(index < 2);
		return SI_MAX_IO_GENERIC + 8 + index;
	case TGSI_SEMANTIC_BCOLOR:
		assert(index < 2);
		/* If it's a varying, COLOR and BCOLOR alias. */
		if (is_varying)
			return SI_MAX_IO_GENERIC + 8 + index;
		else
			return SI_MAX_IO_GENERIC + 10 + index;
	case TGSI_SEMANTIC_TEXCOORD:
		assert(index < 8);
		STATIC_ASSERT(SI_MAX_IO_GENERIC + 12 + 8 <= 63);
		return SI_MAX_IO_GENERIC + 12 + index;
	case TGSI_SEMANTIC_CLIPVERTEX:
		return 63;
	default:
		fprintf(stderr, "invalid semantic name = %u\n", semantic_name);
		assert(!"invalid semantic name");
		return 0;
	}
}

/**
 * Get the value of a shader input parameter and extract a bitfield.
 */
static LLVMValueRef unpack_llvm_param(struct si_shader_context *ctx,
				      LLVMValueRef value, unsigned rshift,
				      unsigned bitwidth)
{
	if (LLVMGetTypeKind(LLVMTypeOf(value)) == LLVMFloatTypeKind)
		value = ac_to_integer(&ctx->ac, value);

	if (rshift)
		value = LLVMBuildLShr(ctx->ac.builder, value,
				      LLVMConstInt(ctx->i32, rshift, 0), "");

	if (rshift + bitwidth < 32) {
		unsigned mask = (1 << bitwidth) - 1;
		value = LLVMBuildAnd(ctx->ac.builder, value,
				     LLVMConstInt(ctx->i32, mask, 0), "");
	}

	return value;
}

LLVMValueRef si_unpack_param(struct si_shader_context *ctx,
			     unsigned param, unsigned rshift,
			     unsigned bitwidth)
{
	LLVMValueRef value = LLVMGetParam(ctx->main_fn, param);

	return unpack_llvm_param(ctx, value, rshift, bitwidth);
}

static LLVMValueRef get_rel_patch_id(struct si_shader_context *ctx)
{
	switch (ctx->type) {
	case PIPE_SHADER_TESS_CTRL:
		return unpack_llvm_param(ctx, ctx->abi.tcs_rel_ids, 0, 8);

	case PIPE_SHADER_TESS_EVAL:
		return LLVMGetParam(ctx->main_fn,
				    ctx->param_tes_rel_patch_id);

	default:
		assert(0);
		return NULL;
	}
}

/* Tessellation shaders pass outputs to the next shader using LDS.
 *
 * LS outputs = TCS inputs
 * TCS outputs = TES inputs
 *
 * The LDS layout is:
 * - TCS inputs for patch 0
 * - TCS inputs for patch 1
 * - TCS inputs for patch 2		= get_tcs_in_current_patch_offset (if RelPatchID==2)
 * - ...
 * - TCS outputs for patch 0            = get_tcs_out_patch0_offset
 * - Per-patch TCS outputs for patch 0  = get_tcs_out_patch0_patch_data_offset
 * - TCS outputs for patch 1
 * - Per-patch TCS outputs for patch 1
 * - TCS outputs for patch 2            = get_tcs_out_current_patch_offset (if RelPatchID==2)
 * - Per-patch TCS outputs for patch 2  = get_tcs_out_current_patch_data_offset (if RelPatchID==2)
 * - ...
 *
 * All three shaders VS(LS), TCS, TES share the same LDS space.
 */

static LLVMValueRef
get_tcs_in_patch_stride(struct si_shader_context *ctx)
{
	return si_unpack_param(ctx, ctx->param_vs_state_bits, 8, 13);
}

static unsigned get_tcs_out_vertex_dw_stride_constant(struct si_shader_context *ctx)
{
	assert(ctx->type == PIPE_SHADER_TESS_CTRL);

	if (ctx->shader->key.mono.u.ff_tcs_inputs_to_copy)
		return util_last_bit64(ctx->shader->key.mono.u.ff_tcs_inputs_to_copy) * 4;

	return util_last_bit64(ctx->shader->selector->outputs_written) * 4;
}

static LLVMValueRef get_tcs_out_vertex_dw_stride(struct si_shader_context *ctx)
{
	unsigned stride = get_tcs_out_vertex_dw_stride_constant(ctx);

	return LLVMConstInt(ctx->i32, stride, 0);
}

static LLVMValueRef get_tcs_out_patch_stride(struct si_shader_context *ctx)
{
	if (ctx->shader->key.mono.u.ff_tcs_inputs_to_copy)
		return si_unpack_param(ctx, ctx->param_tcs_out_lds_layout, 0, 13);

	const struct tgsi_shader_info *info = &ctx->shader->selector->info;
	unsigned tcs_out_vertices = info->properties[TGSI_PROPERTY_TCS_VERTICES_OUT];
	unsigned vertex_dw_stride = get_tcs_out_vertex_dw_stride_constant(ctx);
	unsigned num_patch_outputs = util_last_bit64(ctx->shader->selector->patch_outputs_written);
	unsigned patch_dw_stride = tcs_out_vertices * vertex_dw_stride +
				   num_patch_outputs * 4;
	return LLVMConstInt(ctx->i32, patch_dw_stride, 0);
}

static LLVMValueRef
get_tcs_out_patch0_offset(struct si_shader_context *ctx)
{
	return LLVMBuildMul(ctx->ac.builder,
			    si_unpack_param(ctx,
					    ctx->param_tcs_out_lds_offsets,
					    0, 16),
			    LLVMConstInt(ctx->i32, 4, 0), "");
}

static LLVMValueRef
get_tcs_out_patch0_patch_data_offset(struct si_shader_context *ctx)
{
	return LLVMBuildMul(ctx->ac.builder,
			    si_unpack_param(ctx,
					    ctx->param_tcs_out_lds_offsets,
					    16, 16),
			    LLVMConstInt(ctx->i32, 4, 0), "");
}

static LLVMValueRef
get_tcs_in_current_patch_offset(struct si_shader_context *ctx)
{
	LLVMValueRef patch_stride = get_tcs_in_patch_stride(ctx);
	LLVMValueRef rel_patch_id = get_rel_patch_id(ctx);

	return LLVMBuildMul(ctx->ac.builder, patch_stride, rel_patch_id, "");
}

static LLVMValueRef
get_tcs_out_current_patch_offset(struct si_shader_context *ctx)
{
	LLVMValueRef patch0_offset = get_tcs_out_patch0_offset(ctx);
	LLVMValueRef patch_stride = get_tcs_out_patch_stride(ctx);
	LLVMValueRef rel_patch_id = get_rel_patch_id(ctx);

	return ac_build_imad(&ctx->ac, patch_stride, rel_patch_id, patch0_offset);
}

static LLVMValueRef
get_tcs_out_current_patch_data_offset(struct si_shader_context *ctx)
{
	LLVMValueRef patch0_patch_data_offset =
		get_tcs_out_patch0_patch_data_offset(ctx);
	LLVMValueRef patch_stride = get_tcs_out_patch_stride(ctx);
	LLVMValueRef rel_patch_id = get_rel_patch_id(ctx);

	return ac_build_imad(&ctx->ac, patch_stride, rel_patch_id, patch0_patch_data_offset);
}

static LLVMValueRef get_num_tcs_out_vertices(struct si_shader_context *ctx)
{
	unsigned tcs_out_vertices =
		ctx->shader->selector ?
		ctx->shader->selector->info.properties[TGSI_PROPERTY_TCS_VERTICES_OUT] : 0;

	/* If !tcs_out_vertices, it's either the fixed-func TCS or the TCS epilog. */
	if (ctx->type == PIPE_SHADER_TESS_CTRL && tcs_out_vertices)
		return LLVMConstInt(ctx->i32, tcs_out_vertices, 0);

	return si_unpack_param(ctx, ctx->param_tcs_offchip_layout, 6, 6);
}

static LLVMValueRef get_tcs_in_vertex_dw_stride(struct si_shader_context *ctx)
{
	unsigned stride;

	switch (ctx->type) {
	case PIPE_SHADER_VERTEX:
		stride = ctx->shader->selector->lshs_vertex_stride / 4;
		return LLVMConstInt(ctx->i32, stride, 0);

	case PIPE_SHADER_TESS_CTRL:
		if (ctx->screen->info.chip_class >= GFX9 &&
		    ctx->shader->is_monolithic) {
			stride = ctx->shader->key.part.tcs.ls->lshs_vertex_stride / 4;
			return LLVMConstInt(ctx->i32, stride, 0);
		}
		return si_unpack_param(ctx, ctx->param_vs_state_bits, 24, 8);

	default:
		assert(0);
		return NULL;
	}
}

/* Bitcast <4 x float> to <2 x double>, extract the component, and convert
 * to float. */
static LLVMValueRef extract_double_to_float(struct si_shader_context *ctx,
					    LLVMValueRef vec4,
					    unsigned double_index)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMTypeRef f64 = LLVMDoubleTypeInContext(ctx->ac.context);
	LLVMValueRef dvec2 = LLVMBuildBitCast(builder, vec4,
					      LLVMVectorType(f64, 2), "");
	LLVMValueRef index = LLVMConstInt(ctx->i32, double_index, 0);
	LLVMValueRef value = LLVMBuildExtractElement(builder, dvec2, index, "");
	return LLVMBuildFPTrunc(builder, value, ctx->f32, "");
}

static LLVMValueRef unpack_sint16(struct si_shader_context *ctx,
				 LLVMValueRef i32, unsigned index)
{
	assert(index <= 1);

	if (index == 1)
		return LLVMBuildAShr(ctx->ac.builder, i32,
				     LLVMConstInt(ctx->i32, 16, 0), "");

	return LLVMBuildSExt(ctx->ac.builder,
			     LLVMBuildTrunc(ctx->ac.builder, i32,
					    ctx->ac.i16, ""),
			     ctx->i32, "");
}

void si_llvm_load_input_vs(
	struct si_shader_context *ctx,
	unsigned input_index,
	LLVMValueRef out[4])
{
	const struct tgsi_shader_info *info = &ctx->shader->selector->info;
	unsigned vs_blit_property = info->properties[TGSI_PROPERTY_VS_BLIT_SGPRS];

	if (vs_blit_property) {
		LLVMValueRef vertex_id = ctx->abi.vertex_id;
		LLVMValueRef sel_x1 = LLVMBuildICmp(ctx->ac.builder,
						    LLVMIntULE, vertex_id,
						    ctx->i32_1, "");
		/* Use LLVMIntNE, because we have 3 vertices and only
		 * the middle one should use y2.
		 */
		LLVMValueRef sel_y1 = LLVMBuildICmp(ctx->ac.builder,
						    LLVMIntNE, vertex_id,
						    ctx->i32_1, "");

		if (input_index == 0) {
			/* Position: */
			LLVMValueRef x1y1 = LLVMGetParam(ctx->main_fn,
							 ctx->param_vs_blit_inputs);
			LLVMValueRef x2y2 = LLVMGetParam(ctx->main_fn,
							 ctx->param_vs_blit_inputs + 1);

			LLVMValueRef x1 = unpack_sint16(ctx, x1y1, 0);
			LLVMValueRef y1 = unpack_sint16(ctx, x1y1, 1);
			LLVMValueRef x2 = unpack_sint16(ctx, x2y2, 0);
			LLVMValueRef y2 = unpack_sint16(ctx, x2y2, 1);

			LLVMValueRef x = LLVMBuildSelect(ctx->ac.builder, sel_x1,
							 x1, x2, "");
			LLVMValueRef y = LLVMBuildSelect(ctx->ac.builder, sel_y1,
							 y1, y2, "");

			out[0] = LLVMBuildSIToFP(ctx->ac.builder, x, ctx->f32, "");
			out[1] = LLVMBuildSIToFP(ctx->ac.builder, y, ctx->f32, "");
			out[2] = LLVMGetParam(ctx->main_fn,
					      ctx->param_vs_blit_inputs + 2);
			out[3] = ctx->ac.f32_1;
			return;
		}

		/* Color or texture coordinates: */
		assert(input_index == 1);

		if (vs_blit_property == SI_VS_BLIT_SGPRS_POS_COLOR) {
			for (int i = 0; i < 4; i++) {
				out[i] = LLVMGetParam(ctx->main_fn,
						      ctx->param_vs_blit_inputs + 3 + i);
			}
		} else {
			assert(vs_blit_property == SI_VS_BLIT_SGPRS_POS_TEXCOORD);
			LLVMValueRef x1 = LLVMGetParam(ctx->main_fn,
						       ctx->param_vs_blit_inputs + 3);
			LLVMValueRef y1 = LLVMGetParam(ctx->main_fn,
						       ctx->param_vs_blit_inputs + 4);
			LLVMValueRef x2 = LLVMGetParam(ctx->main_fn,
						       ctx->param_vs_blit_inputs + 5);
			LLVMValueRef y2 = LLVMGetParam(ctx->main_fn,
						       ctx->param_vs_blit_inputs + 6);

			out[0] = LLVMBuildSelect(ctx->ac.builder, sel_x1,
						 x1, x2, "");
			out[1] = LLVMBuildSelect(ctx->ac.builder, sel_y1,
						 y1, y2, "");
			out[2] = LLVMGetParam(ctx->main_fn,
					      ctx->param_vs_blit_inputs + 7);
			out[3] = LLVMGetParam(ctx->main_fn,
					      ctx->param_vs_blit_inputs + 8);
		}
		return;
	}

	unsigned chan;
	unsigned fix_fetch;
	unsigned num_fetches;
	unsigned fetch_stride;
	unsigned num_channels;

	LLVMValueRef t_list_ptr;
	LLVMValueRef t_offset;
	LLVMValueRef t_list;
	LLVMValueRef vertex_index;
	LLVMValueRef input[3];

	/* Load the T list */
	t_list_ptr = LLVMGetParam(ctx->main_fn, ctx->param_vertex_buffers);

	t_offset = LLVMConstInt(ctx->i32, input_index, 0);

	t_list = ac_build_load_to_sgpr(&ctx->ac, t_list_ptr, t_offset);

	vertex_index = LLVMGetParam(ctx->main_fn,
				    ctx->param_vertex_index0 +
				    input_index);

	fix_fetch = ctx->shader->key.mono.vs_fix_fetch[input_index];

	/* Do multiple loads for special formats. */
	switch (fix_fetch) {
	case SI_FIX_FETCH_RG_64_FLOAT:
		num_fetches = 1; /* 1 2-dword or 4-dword load */
		fetch_stride = 0;
		if (util_last_bit(info->input_usage_mask[input_index]) >= 2)
			num_channels = 4; /* 2 doubles in 4 dwords */
		else
			num_channels = 2; /* 1 double in 2 dwords */
		break;
	case SI_FIX_FETCH_RGB_64_FLOAT:
		num_fetches = 3; /* 3 2-dword loads */
		fetch_stride = 8;
		num_channels = 2;
		break;
	case SI_FIX_FETCH_RGBA_64_FLOAT:
		num_fetches = 2; /* 2 4-dword loads */
		fetch_stride = 16;
		num_channels = 4;
		break;
	case SI_FIX_FETCH_RGB_8:
	case SI_FIX_FETCH_RGB_8_INT:
		num_fetches = 3;
		fetch_stride = 1;
		num_channels = 1;
		break;
	case SI_FIX_FETCH_RGB_16:
	case SI_FIX_FETCH_RGB_16_INT:
		num_fetches = 3;
		fetch_stride = 2;
		num_channels = 1;
		break;
	default:
		num_fetches = 1;
		fetch_stride = 0;
		num_channels = util_last_bit(info->input_usage_mask[input_index]);
	}

	for (unsigned i = 0; i < num_fetches; i++) {
		LLVMValueRef voffset = LLVMConstInt(ctx->i32, fetch_stride * i, 0);

		input[i] = ac_build_buffer_load_format(&ctx->ac, t_list,
						       vertex_index, voffset,
						       num_channels, false, true);
		input[i] = ac_build_expand_to_vec4(&ctx->ac, input[i], num_channels);
	}

	/* Break up the vec4 into individual components */
	for (chan = 0; chan < 4; chan++) {
		LLVMValueRef llvm_chan = LLVMConstInt(ctx->i32, chan, 0);
		out[chan] = LLVMBuildExtractElement(ctx->ac.builder,
						    input[0], llvm_chan, "");
	}

	switch (fix_fetch) {
	case SI_FIX_FETCH_A2_SNORM:
	case SI_FIX_FETCH_A2_SSCALED:
	case SI_FIX_FETCH_A2_SINT: {
		/* The hardware returns an unsigned value; convert it to a
		 * signed one.
		 */
		LLVMValueRef tmp = out[3];
		LLVMValueRef c30 = LLVMConstInt(ctx->i32, 30, 0);

		/* First, recover the sign-extended signed integer value. */
		if (fix_fetch == SI_FIX_FETCH_A2_SSCALED)
			tmp = LLVMBuildFPToUI(ctx->ac.builder, tmp, ctx->i32, "");
		else
			tmp = ac_to_integer(&ctx->ac, tmp);

		/* For the integer-like cases, do a natural sign extension.
		 *
		 * For the SNORM case, the values are 0.0, 0.333, 0.666, 1.0
		 * and happen to contain 0, 1, 2, 3 as the two LSBs of the
		 * exponent.
		 */
		tmp = LLVMBuildShl(ctx->ac.builder, tmp,
				   fix_fetch == SI_FIX_FETCH_A2_SNORM ?
				   LLVMConstInt(ctx->i32, 7, 0) : c30, "");
		tmp = LLVMBuildAShr(ctx->ac.builder, tmp, c30, "");

		/* Convert back to the right type. */
		if (fix_fetch == SI_FIX_FETCH_A2_SNORM) {
			LLVMValueRef clamp;
			LLVMValueRef neg_one = LLVMConstReal(ctx->f32, -1.0);
			tmp = LLVMBuildSIToFP(ctx->ac.builder, tmp, ctx->f32, "");
			clamp = LLVMBuildFCmp(ctx->ac.builder, LLVMRealULT, tmp, neg_one, "");
			tmp = LLVMBuildSelect(ctx->ac.builder, clamp, neg_one, tmp, "");
		} else if (fix_fetch == SI_FIX_FETCH_A2_SSCALED) {
			tmp = LLVMBuildSIToFP(ctx->ac.builder, tmp, ctx->f32, "");
		}

		out[3] = tmp;
		break;
	}
	case SI_FIX_FETCH_RGBA_32_UNORM:
	case SI_FIX_FETCH_RGBX_32_UNORM:
		for (chan = 0; chan < 4; chan++) {
			out[chan] = ac_to_integer(&ctx->ac, out[chan]);
			out[chan] = LLVMBuildUIToFP(ctx->ac.builder,
						    out[chan], ctx->f32, "");
			out[chan] = LLVMBuildFMul(ctx->ac.builder, out[chan],
						  LLVMConstReal(ctx->f32, 1.0 / UINT_MAX), "");
		}
		/* RGBX UINT returns 1 in alpha, which would be rounded to 0 by normalizing. */
		if (fix_fetch == SI_FIX_FETCH_RGBX_32_UNORM)
			out[3] = LLVMConstReal(ctx->f32, 1);
		break;
	case SI_FIX_FETCH_RGBA_32_SNORM:
	case SI_FIX_FETCH_RGBX_32_SNORM:
	case SI_FIX_FETCH_RGBA_32_FIXED:
	case SI_FIX_FETCH_RGBX_32_FIXED: {
		double scale;
		if (fix_fetch >= SI_FIX_FETCH_RGBA_32_FIXED)
			scale = 1.0 / 0x10000;
		else
			scale = 1.0 / INT_MAX;

		for (chan = 0; chan < 4; chan++) {
			out[chan] = ac_to_integer(&ctx->ac, out[chan]);
			out[chan] = LLVMBuildSIToFP(ctx->ac.builder,
						    out[chan], ctx->f32, "");
			out[chan] = LLVMBuildFMul(ctx->ac.builder, out[chan],
						  LLVMConstReal(ctx->f32, scale), "");
		}
		/* RGBX SINT returns 1 in alpha, which would be rounded to 0 by normalizing. */
		if (fix_fetch == SI_FIX_FETCH_RGBX_32_SNORM ||
		    fix_fetch == SI_FIX_FETCH_RGBX_32_FIXED)
			out[3] = LLVMConstReal(ctx->f32, 1);
		break;
	}
	case SI_FIX_FETCH_RGBA_32_USCALED:
		for (chan = 0; chan < 4; chan++) {
			out[chan] = ac_to_integer(&ctx->ac, out[chan]);
			out[chan] = LLVMBuildUIToFP(ctx->ac.builder,
						    out[chan], ctx->f32, "");
		}
		break;
	case SI_FIX_FETCH_RGBA_32_SSCALED:
		for (chan = 0; chan < 4; chan++) {
			out[chan] = ac_to_integer(&ctx->ac, out[chan]);
			out[chan] = LLVMBuildSIToFP(ctx->ac.builder,
						    out[chan], ctx->f32, "");
		}
		break;
	case SI_FIX_FETCH_RG_64_FLOAT:
		for (chan = 0; chan < 2; chan++)
			out[chan] = extract_double_to_float(ctx, input[0], chan);

		out[2] = LLVMConstReal(ctx->f32, 0);
		out[3] = LLVMConstReal(ctx->f32, 1);
		break;
	case SI_FIX_FETCH_RGB_64_FLOAT:
		for (chan = 0; chan < 3; chan++)
			out[chan] = extract_double_to_float(ctx, input[chan], 0);

		out[3] = LLVMConstReal(ctx->f32, 1);
		break;
	case SI_FIX_FETCH_RGBA_64_FLOAT:
		for (chan = 0; chan < 4; chan++) {
			out[chan] = extract_double_to_float(ctx, input[chan / 2],
							    chan % 2);
		}
		break;
	case SI_FIX_FETCH_RGB_8:
	case SI_FIX_FETCH_RGB_8_INT:
	case SI_FIX_FETCH_RGB_16:
	case SI_FIX_FETCH_RGB_16_INT:
		for (chan = 0; chan < 3; chan++) {
			out[chan] = LLVMBuildExtractElement(ctx->ac.builder,
							    input[chan],
							    ctx->i32_0, "");
		}
		if (fix_fetch == SI_FIX_FETCH_RGB_8 ||
		    fix_fetch == SI_FIX_FETCH_RGB_16) {
			out[3] = LLVMConstReal(ctx->f32, 1);
		} else {
			out[3] = ac_to_float(&ctx->ac, ctx->i32_1);
		}
		break;
	}
}

static void declare_input_vs(
	struct si_shader_context *ctx,
	unsigned input_index,
	const struct tgsi_full_declaration *decl,
	LLVMValueRef out[4])
{
	si_llvm_load_input_vs(ctx, input_index, out);
}

static LLVMValueRef get_primitive_id(struct si_shader_context *ctx,
				     unsigned swizzle)
{
	if (swizzle > 0)
		return ctx->i32_0;

	switch (ctx->type) {
	case PIPE_SHADER_VERTEX:
		return LLVMGetParam(ctx->main_fn,
				    ctx->param_vs_prim_id);
	case PIPE_SHADER_TESS_CTRL:
		return ctx->abi.tcs_patch_id;
	case PIPE_SHADER_TESS_EVAL:
		return ctx->abi.tes_patch_id;
	case PIPE_SHADER_GEOMETRY:
		return ctx->abi.gs_prim_id;
	default:
		assert(0);
		return ctx->i32_0;
	}
}

/**
 * Return the value of tgsi_ind_register for indexing.
 * This is the indirect index with the constant offset added to it.
 */
LLVMValueRef si_get_indirect_index(struct si_shader_context *ctx,
				   const struct tgsi_ind_register *ind,
				   unsigned addr_mul,
				   int rel_index)
{
	LLVMValueRef result;

	if (ind->File == TGSI_FILE_ADDRESS) {
		result = ctx->addrs[ind->Index][ind->Swizzle];
		result = LLVMBuildLoad(ctx->ac.builder, result, "");
	} else {
		struct tgsi_full_src_register src = {};

		src.Register.File = ind->File;
		src.Register.Index = ind->Index;

		/* Set the second index to 0 for constants. */
		if (ind->File == TGSI_FILE_CONSTANT)
			src.Register.Dimension = 1;

		result = ctx->bld_base.emit_fetch_funcs[ind->File](&ctx->bld_base, &src,
								   TGSI_TYPE_SIGNED,
								   ind->Swizzle);
		result = ac_to_integer(&ctx->ac, result);
	}

	return ac_build_imad(&ctx->ac, result, LLVMConstInt(ctx->i32, addr_mul, 0),
			     LLVMConstInt(ctx->i32, rel_index, 0));
}

/**
 * Like si_get_indirect_index, but restricts the return value to a (possibly
 * undefined) value inside [0..num).
 */
LLVMValueRef si_get_bounded_indirect_index(struct si_shader_context *ctx,
					   const struct tgsi_ind_register *ind,
					   int rel_index, unsigned num)
{
	LLVMValueRef result = si_get_indirect_index(ctx, ind, 1, rel_index);

	return si_llvm_bound_index(ctx, result, num);
}

static LLVMValueRef get_dw_address_from_generic_indices(struct si_shader_context *ctx,
							LLVMValueRef vertex_dw_stride,
							LLVMValueRef base_addr,
							LLVMValueRef vertex_index,
							LLVMValueRef param_index,
							unsigned input_index,
							ubyte *name,
							ubyte *index,
							bool is_patch)
{
	if (vertex_dw_stride) {
		base_addr = ac_build_imad(&ctx->ac, vertex_index,
					  vertex_dw_stride, base_addr);
	}

	if (param_index) {
		base_addr = ac_build_imad(&ctx->ac, param_index,
					  LLVMConstInt(ctx->i32, 4, 0), base_addr);
	}

	int param = is_patch ?
		si_shader_io_get_unique_index_patch(name[input_index],
						    index[input_index]) :
		si_shader_io_get_unique_index(name[input_index],
					      index[input_index], false);

	/* Add the base address of the element. */
	return LLVMBuildAdd(ctx->ac.builder, base_addr,
			    LLVMConstInt(ctx->i32, param * 4, 0), "");
}

/**
 * Calculate a dword address given an input or output register and a stride.
 */
static LLVMValueRef get_dw_address(struct si_shader_context *ctx,
				   const struct tgsi_full_dst_register *dst,
				   const struct tgsi_full_src_register *src,
				   LLVMValueRef vertex_dw_stride,
				   LLVMValueRef base_addr)
{
	struct tgsi_shader_info *info = &ctx->shader->selector->info;
	ubyte *name, *index, *array_first;
	int input_index;
	struct tgsi_full_dst_register reg;
	LLVMValueRef vertex_index = NULL;
	LLVMValueRef ind_index = NULL;

	/* Set the register description. The address computation is the same
	 * for sources and destinations. */
	if (src) {
		reg.Register.File = src->Register.File;
		reg.Register.Index = src->Register.Index;
		reg.Register.Indirect = src->Register.Indirect;
		reg.Register.Dimension = src->Register.Dimension;
		reg.Indirect = src->Indirect;
		reg.Dimension = src->Dimension;
		reg.DimIndirect = src->DimIndirect;
	} else
		reg = *dst;

	/* If the register is 2-dimensional (e.g. an array of vertices
	 * in a primitive), calculate the base address of the vertex. */
	if (reg.Register.Dimension) {
		if (reg.Dimension.Indirect)
			vertex_index = si_get_indirect_index(ctx, &reg.DimIndirect,
						      1, reg.Dimension.Index);
		else
			vertex_index = LLVMConstInt(ctx->i32, reg.Dimension.Index, 0);
	}

	/* Get information about the register. */
	if (reg.Register.File == TGSI_FILE_INPUT) {
		name = info->input_semantic_name;
		index = info->input_semantic_index;
		array_first = info->input_array_first;
	} else if (reg.Register.File == TGSI_FILE_OUTPUT) {
		name = info->output_semantic_name;
		index = info->output_semantic_index;
		array_first = info->output_array_first;
	} else {
		assert(0);
		return NULL;
	}

	if (reg.Register.Indirect) {
		/* Add the relative address of the element. */
		if (reg.Indirect.ArrayID)
			input_index = array_first[reg.Indirect.ArrayID];
		else
			input_index = reg.Register.Index;

		ind_index = si_get_indirect_index(ctx, &reg.Indirect,
						  1, reg.Register.Index - input_index);
	} else {
		input_index = reg.Register.Index;
	}

	return get_dw_address_from_generic_indices(ctx, vertex_dw_stride,
						   base_addr, vertex_index,
						   ind_index, input_index,
						   name, index,
						   !reg.Register.Dimension);
}

/* The offchip buffer layout for TCS->TES is
 *
 * - attribute 0 of patch 0 vertex 0
 * - attribute 0 of patch 0 vertex 1
 * - attribute 0 of patch 0 vertex 2
 *   ...
 * - attribute 0 of patch 1 vertex 0
 * - attribute 0 of patch 1 vertex 1
 *   ...
 * - attribute 1 of patch 0 vertex 0
 * - attribute 1 of patch 0 vertex 1
 *   ...
 * - per patch attribute 0 of patch 0
 * - per patch attribute 0 of patch 1
 *   ...
 *
 * Note that every attribute has 4 components.
 */
static LLVMValueRef get_tcs_tes_buffer_address(struct si_shader_context *ctx,
					       LLVMValueRef rel_patch_id,
                                               LLVMValueRef vertex_index,
                                               LLVMValueRef param_index)
{
	LLVMValueRef base_addr, vertices_per_patch, num_patches, total_vertices;
	LLVMValueRef param_stride, constant16;

	vertices_per_patch = get_num_tcs_out_vertices(ctx);
	num_patches = si_unpack_param(ctx, ctx->param_tcs_offchip_layout, 0, 6);
	total_vertices = LLVMBuildMul(ctx->ac.builder, vertices_per_patch,
	                              num_patches, "");

	constant16 = LLVMConstInt(ctx->i32, 16, 0);
	if (vertex_index) {
		base_addr = ac_build_imad(&ctx->ac, rel_patch_id,
					  vertices_per_patch, vertex_index);
		param_stride = total_vertices;
	} else {
		base_addr = rel_patch_id;
		param_stride = num_patches;
	}

	base_addr = ac_build_imad(&ctx->ac, param_index, param_stride, base_addr);
	base_addr = LLVMBuildMul(ctx->ac.builder, base_addr, constant16, "");

	if (!vertex_index) {
		LLVMValueRef patch_data_offset =
		           si_unpack_param(ctx, ctx->param_tcs_offchip_layout, 12, 20);

		base_addr = LLVMBuildAdd(ctx->ac.builder, base_addr,
		                         patch_data_offset, "");
	}
	return base_addr;
}

/* This is a generic helper that can be shared by the NIR and TGSI backends */
static LLVMValueRef get_tcs_tes_buffer_address_from_generic_indices(
					struct si_shader_context *ctx,
					LLVMValueRef vertex_index,
					LLVMValueRef param_index,
					unsigned param_base,
					ubyte *name,
					ubyte *index,
					bool is_patch)
{
	unsigned param_index_base;

	param_index_base = is_patch ?
		si_shader_io_get_unique_index_patch(name[param_base], index[param_base]) :
		si_shader_io_get_unique_index(name[param_base], index[param_base], false);

	if (param_index) {
		param_index = LLVMBuildAdd(ctx->ac.builder, param_index,
					   LLVMConstInt(ctx->i32, param_index_base, 0),
					   "");
	} else {
		param_index = LLVMConstInt(ctx->i32, param_index_base, 0);
	}

	return get_tcs_tes_buffer_address(ctx, get_rel_patch_id(ctx),
					  vertex_index, param_index);
}

static LLVMValueRef get_tcs_tes_buffer_address_from_reg(
                                       struct si_shader_context *ctx,
                                       const struct tgsi_full_dst_register *dst,
                                       const struct tgsi_full_src_register *src)
{
	struct tgsi_shader_info *info = &ctx->shader->selector->info;
	ubyte *name, *index, *array_first;
	struct tgsi_full_src_register reg;
	LLVMValueRef vertex_index = NULL;
	LLVMValueRef param_index = NULL;
	unsigned param_base;

	reg = src ? *src : tgsi_full_src_register_from_dst(dst);

	if (reg.Register.Dimension) {

		if (reg.Dimension.Indirect)
			vertex_index = si_get_indirect_index(ctx, &reg.DimIndirect,
							     1, reg.Dimension.Index);
		else
			vertex_index = LLVMConstInt(ctx->i32, reg.Dimension.Index, 0);
	}

	/* Get information about the register. */
	if (reg.Register.File == TGSI_FILE_INPUT) {
		name = info->input_semantic_name;
		index = info->input_semantic_index;
		array_first = info->input_array_first;
	} else if (reg.Register.File == TGSI_FILE_OUTPUT) {
		name = info->output_semantic_name;
		index = info->output_semantic_index;
		array_first = info->output_array_first;
	} else {
		assert(0);
		return NULL;
	}

	if (reg.Register.Indirect) {
		if (reg.Indirect.ArrayID)
			param_base = array_first[reg.Indirect.ArrayID];
		else
			param_base = reg.Register.Index;

		param_index = si_get_indirect_index(ctx, &reg.Indirect,
						    1, reg.Register.Index - param_base);

	} else {
		param_base = reg.Register.Index;
	}

	return get_tcs_tes_buffer_address_from_generic_indices(ctx, vertex_index,
							       param_index, param_base,
							       name, index, !reg.Register.Dimension);
}

static LLVMValueRef buffer_load(struct lp_build_tgsi_context *bld_base,
                                LLVMTypeRef type, unsigned swizzle,
                                LLVMValueRef buffer, LLVMValueRef offset,
                                LLVMValueRef base, bool can_speculate)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef value, value2;
	LLVMTypeRef vec_type = LLVMVectorType(type, 4);

	if (swizzle == ~0) {
		value = ac_build_buffer_load(&ctx->ac, buffer, 4, NULL, base, offset,
					     0, 1, 0, can_speculate, false);

		return LLVMBuildBitCast(ctx->ac.builder, value, vec_type, "");
	}

	if (!llvm_type_is_64bit(ctx, type)) {
		value = ac_build_buffer_load(&ctx->ac, buffer, 4, NULL, base, offset,
					     0, 1, 0, can_speculate, false);

		value = LLVMBuildBitCast(ctx->ac.builder, value, vec_type, "");
		return LLVMBuildExtractElement(ctx->ac.builder, value,
		                    LLVMConstInt(ctx->i32, swizzle, 0), "");
	}

	value = ac_build_buffer_load(&ctx->ac, buffer, 1, NULL, base, offset,
	                          swizzle * 4, 1, 0, can_speculate, false);

	value2 = ac_build_buffer_load(&ctx->ac, buffer, 1, NULL, base, offset,
	                           swizzle * 4 + 4, 1, 0, can_speculate, false);

	return si_llvm_emit_fetch_64bit(bld_base, type, value, value2);
}

/**
 * Load from LDS.
 *
 * \param type		output value type
 * \param swizzle	offset (typically 0..3); it can be ~0, which loads a vec4
 * \param dw_addr	address in dwords
 */
static LLVMValueRef lds_load(struct lp_build_tgsi_context *bld_base,
			     LLVMTypeRef type, unsigned swizzle,
			     LLVMValueRef dw_addr)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef value;

	if (swizzle == ~0) {
		LLVMValueRef values[TGSI_NUM_CHANNELS];

		for (unsigned chan = 0; chan < TGSI_NUM_CHANNELS; chan++)
			values[chan] = lds_load(bld_base, type, chan, dw_addr);

		return ac_build_gather_values(&ctx->ac, values,
					      TGSI_NUM_CHANNELS);
	}

	/* Split 64-bit loads. */
	if (llvm_type_is_64bit(ctx, type)) {
		LLVMValueRef lo, hi;

		lo = lds_load(bld_base, ctx->i32, swizzle, dw_addr);
		hi = lds_load(bld_base, ctx->i32, swizzle + 1, dw_addr);
		return si_llvm_emit_fetch_64bit(bld_base, type, lo, hi);
	}

	dw_addr = LLVMBuildAdd(ctx->ac.builder, dw_addr,
			       LLVMConstInt(ctx->i32, swizzle, 0), "");

	value = ac_lds_load(&ctx->ac, dw_addr);

	return LLVMBuildBitCast(ctx->ac.builder, value, type, "");
}

/**
 * Store to LDS.
 *
 * \param swizzle	offset (typically 0..3)
 * \param dw_addr	address in dwords
 * \param value		value to store
 */
static void lds_store(struct si_shader_context *ctx,
		      unsigned dw_offset_imm, LLVMValueRef dw_addr,
		      LLVMValueRef value)
{
	dw_addr = LLVMBuildAdd(ctx->ac.builder, dw_addr,
			       LLVMConstInt(ctx->i32, dw_offset_imm, 0), "");

	ac_lds_store(&ctx->ac, dw_addr, value);
}

enum si_tess_ring {
	TCS_FACTOR_RING,
	TESS_OFFCHIP_RING_TCS,
	TESS_OFFCHIP_RING_TES,
};

static LLVMValueRef get_tess_ring_descriptor(struct si_shader_context *ctx,
					     enum si_tess_ring ring)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	unsigned param = ring == TESS_OFFCHIP_RING_TES ? ctx->param_tes_offchip_addr :
							 ctx->param_tcs_out_lds_layout;
	LLVMValueRef addr = LLVMGetParam(ctx->main_fn, param);

	/* TCS only receives high 13 bits of the address. */
	if (ring == TESS_OFFCHIP_RING_TCS || ring == TCS_FACTOR_RING) {
		addr = LLVMBuildAnd(builder, addr,
				    LLVMConstInt(ctx->i32, 0xfff80000, 0), "");
	}

	if (ring == TCS_FACTOR_RING) {
		unsigned tf_offset = ctx->screen->tess_offchip_ring_size;
		addr = LLVMBuildAdd(builder, addr,
				    LLVMConstInt(ctx->i32, tf_offset, 0), "");
	}

	LLVMValueRef desc[4];
	desc[0] = addr;
	desc[1] = LLVMConstInt(ctx->i32,
			       S_008F04_BASE_ADDRESS_HI(ctx->screen->info.address32_hi), 0);
	desc[2] = LLVMConstInt(ctx->i32, 0xffffffff, 0);
	desc[3] = LLVMConstInt(ctx->i32,
			       S_008F0C_DST_SEL_X(V_008F0C_SQ_SEL_X) |
			       S_008F0C_DST_SEL_Y(V_008F0C_SQ_SEL_Y) |
			       S_008F0C_DST_SEL_Z(V_008F0C_SQ_SEL_Z) |
			       S_008F0C_DST_SEL_W(V_008F0C_SQ_SEL_W) |
			       S_008F0C_NUM_FORMAT(V_008F0C_BUF_NUM_FORMAT_FLOAT) |
			       S_008F0C_DATA_FORMAT(V_008F0C_BUF_DATA_FORMAT_32), 0);

	return ac_build_gather_values(&ctx->ac, desc, 4);
}

static LLVMValueRef fetch_input_tcs(
	struct lp_build_tgsi_context *bld_base,
	const struct tgsi_full_src_register *reg,
	enum tgsi_opcode_type type, unsigned swizzle_in)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef dw_addr, stride;
	unsigned swizzle = swizzle_in & 0xffff;
	stride = get_tcs_in_vertex_dw_stride(ctx);
	dw_addr = get_tcs_in_current_patch_offset(ctx);
	dw_addr = get_dw_address(ctx, NULL, reg, stride, dw_addr);

	return lds_load(bld_base, tgsi2llvmtype(bld_base, type), swizzle, dw_addr);
}

static LLVMValueRef si_nir_load_tcs_varyings(struct ac_shader_abi *abi,
					     LLVMTypeRef type,
					     LLVMValueRef vertex_index,
					     LLVMValueRef param_index,
					     unsigned const_index,
					     unsigned location,
					     unsigned driver_location,
					     unsigned component,
					     unsigned num_components,
					     bool is_patch,
					     bool is_compact,
					     bool load_input)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct tgsi_shader_info *info = &ctx->shader->selector->info;
	struct lp_build_tgsi_context *bld_base = &ctx->bld_base;
	LLVMValueRef dw_addr, stride;

	driver_location = driver_location / 4;

	if (load_input) {
		stride = get_tcs_in_vertex_dw_stride(ctx);
		dw_addr = get_tcs_in_current_patch_offset(ctx);
	} else {
		if (is_patch) {
			stride = NULL;
			dw_addr = get_tcs_out_current_patch_data_offset(ctx);
		} else {
			stride = get_tcs_out_vertex_dw_stride(ctx);
			dw_addr = get_tcs_out_current_patch_offset(ctx);
		}
	}

	if (param_index) {
		/* Add the constant index to the indirect index */
		param_index = LLVMBuildAdd(ctx->ac.builder, param_index,
					   LLVMConstInt(ctx->i32, const_index, 0), "");
	} else {
		param_index = LLVMConstInt(ctx->i32, const_index, 0);
	}

	ubyte *names;
	ubyte *indices;
	if (load_input) {
		names = info->input_semantic_name;
		indices = info->input_semantic_index;
	} else {
		names = info->output_semantic_name;
		indices = info->output_semantic_index;
	}

	dw_addr = get_dw_address_from_generic_indices(ctx, stride, dw_addr,
						      vertex_index, param_index,
						      driver_location,
						      names, indices,
						      is_patch);

	LLVMValueRef value[4];
	for (unsigned i = 0; i < num_components; i++) {
		unsigned offset = i;
		if (llvm_type_is_64bit(ctx, type))
			offset *= 2;

		offset += component;
		value[i + component] = lds_load(bld_base, type, offset, dw_addr);
	}

	return ac_build_varying_gather_values(&ctx->ac, value, num_components, component);
}

static LLVMValueRef fetch_output_tcs(
		struct lp_build_tgsi_context *bld_base,
		const struct tgsi_full_src_register *reg,
		enum tgsi_opcode_type type, unsigned swizzle_in)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef dw_addr, stride;
	unsigned swizzle = (swizzle_in & 0xffff);

	if (reg->Register.Dimension) {
		stride = get_tcs_out_vertex_dw_stride(ctx);
		dw_addr = get_tcs_out_current_patch_offset(ctx);
		dw_addr = get_dw_address(ctx, NULL, reg, stride, dw_addr);
	} else {
		dw_addr = get_tcs_out_current_patch_data_offset(ctx);
		dw_addr = get_dw_address(ctx, NULL, reg, NULL, dw_addr);
	}

	return lds_load(bld_base, tgsi2llvmtype(bld_base, type), swizzle, dw_addr);
}

static LLVMValueRef fetch_input_tes(
	struct lp_build_tgsi_context *bld_base,
	const struct tgsi_full_src_register *reg,
	enum tgsi_opcode_type type, unsigned swizzle_in)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef base, addr;
	unsigned swizzle = (swizzle_in & 0xffff);

	base = LLVMGetParam(ctx->main_fn, ctx->param_tcs_offchip_offset);
	addr = get_tcs_tes_buffer_address_from_reg(ctx, NULL, reg);

	return buffer_load(bld_base, tgsi2llvmtype(bld_base, type), swizzle,
			   ctx->tess_offchip_ring, base, addr, true);
}

LLVMValueRef si_nir_load_input_tes(struct ac_shader_abi *abi,
				   LLVMTypeRef type,
				   LLVMValueRef vertex_index,
				   LLVMValueRef param_index,
				   unsigned const_index,
				   unsigned location,
				   unsigned driver_location,
				   unsigned component,
				   unsigned num_components,
				   bool is_patch,
				   bool is_compact,
				   bool load_input)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct tgsi_shader_info *info = &ctx->shader->selector->info;
	LLVMValueRef base, addr;

	driver_location = driver_location / 4;

	base = LLVMGetParam(ctx->main_fn, ctx->param_tcs_offchip_offset);

	if (param_index) {
		/* Add the constant index to the indirect index */
		param_index = LLVMBuildAdd(ctx->ac.builder, param_index,
					   LLVMConstInt(ctx->i32, const_index, 0), "");
	} else {
		param_index = LLVMConstInt(ctx->i32, const_index, 0);
	}

	addr = get_tcs_tes_buffer_address_from_generic_indices(ctx, vertex_index,
							       param_index, driver_location,
							       info->input_semantic_name,
							       info->input_semantic_index,
							       is_patch);

	/* TODO: This will generate rather ordinary llvm code, although it
	 * should be easy for the optimiser to fix up. In future we might want
	 * to refactor buffer_load(), but for now this maximises code sharing
	 * between the NIR and TGSI backends.
	 */
	LLVMValueRef value[4];
	for (unsigned i = 0; i < num_components; i++) {
		unsigned offset = i;
		if (llvm_type_is_64bit(ctx, type))
			offset *= 2;

		offset += component;
		value[i + component] = buffer_load(&ctx->bld_base, type, offset,
						   ctx->tess_offchip_ring, base, addr, true);
	}

	return ac_build_varying_gather_values(&ctx->ac, value, num_components, component);
}

static void store_output_tcs(struct lp_build_tgsi_context *bld_base,
			     const struct tgsi_full_instruction *inst,
			     const struct tgsi_opcode_info *info,
			     unsigned index,
			     LLVMValueRef dst[4])
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	const struct tgsi_full_dst_register *reg = &inst->Dst[index];
	const struct tgsi_shader_info *sh_info = &ctx->shader->selector->info;
	unsigned chan_index;
	LLVMValueRef dw_addr, stride;
	LLVMValueRef buffer, base, buf_addr;
	LLVMValueRef values[4];
	bool skip_lds_store;
	bool is_tess_factor = false, is_tess_inner = false;

	/* Only handle per-patch and per-vertex outputs here.
	 * Vectors will be lowered to scalars and this function will be called again.
	 */
	if (reg->Register.File != TGSI_FILE_OUTPUT ||
	    (dst[0] && LLVMGetTypeKind(LLVMTypeOf(dst[0])) == LLVMVectorTypeKind)) {
		si_llvm_emit_store(bld_base, inst, info, index, dst);
		return;
	}

	if (reg->Register.Dimension) {
		stride = get_tcs_out_vertex_dw_stride(ctx);
		dw_addr = get_tcs_out_current_patch_offset(ctx);
		dw_addr = get_dw_address(ctx, reg, NULL, stride, dw_addr);
		skip_lds_store = !sh_info->reads_pervertex_outputs;
	} else {
		dw_addr = get_tcs_out_current_patch_data_offset(ctx);
		dw_addr = get_dw_address(ctx, reg, NULL, NULL, dw_addr);
		skip_lds_store = !sh_info->reads_perpatch_outputs;

		if (!reg->Register.Indirect) {
			int name = sh_info->output_semantic_name[reg->Register.Index];

			/* Always write tess factors into LDS for the TCS epilog. */
			if (name == TGSI_SEMANTIC_TESSINNER ||
			    name == TGSI_SEMANTIC_TESSOUTER) {
				/* The epilog doesn't read LDS if invocation 0 defines tess factors. */
				skip_lds_store = !sh_info->reads_tessfactor_outputs &&
						 ctx->shader->selector->tcs_info.tessfactors_are_def_in_all_invocs;
				is_tess_factor = true;
				is_tess_inner = name == TGSI_SEMANTIC_TESSINNER;
			}
		}
	}

	buffer = get_tess_ring_descriptor(ctx, TESS_OFFCHIP_RING_TCS);

	base = LLVMGetParam(ctx->main_fn, ctx->param_tcs_offchip_offset);
	buf_addr = get_tcs_tes_buffer_address_from_reg(ctx, reg, NULL);

	uint32_t writemask = reg->Register.WriteMask;
	while (writemask) {
		chan_index = u_bit_scan(&writemask);
		LLVMValueRef value = dst[chan_index];

		if (inst->Instruction.Saturate)
			value = ac_build_clamp(&ctx->ac, value);

		/* Skip LDS stores if there is no LDS read of this output. */
		if (!skip_lds_store)
			lds_store(ctx, chan_index, dw_addr, value);

		value = ac_to_integer(&ctx->ac, value);
		values[chan_index] = value;

		if (reg->Register.WriteMask != 0xF && !is_tess_factor) {
			ac_build_buffer_store_dword(&ctx->ac, buffer, value, 1,
						    buf_addr, base,
						    4 * chan_index, 1, 0, true, false);
		}

		/* Write tess factors into VGPRs for the epilog. */
		if (is_tess_factor &&
		    ctx->shader->selector->tcs_info.tessfactors_are_def_in_all_invocs) {
			if (!is_tess_inner) {
				LLVMBuildStore(ctx->ac.builder, value, /* outer */
					       ctx->invoc0_tess_factors[chan_index]);
			} else if (chan_index < 2) {
				LLVMBuildStore(ctx->ac.builder, value, /* inner */
					       ctx->invoc0_tess_factors[4 + chan_index]);
			}
		}
	}

	if (reg->Register.WriteMask == 0xF && !is_tess_factor) {
		LLVMValueRef value = ac_build_gather_values(&ctx->ac,
		                                            values, 4);
		ac_build_buffer_store_dword(&ctx->ac, buffer, value, 4, buf_addr,
					    base, 0, 1, 0, true, false);
	}
}

static void si_nir_store_output_tcs(struct ac_shader_abi *abi,
				    const struct nir_variable *var,
				    LLVMValueRef vertex_index,
				    LLVMValueRef param_index,
				    unsigned const_index,
				    LLVMValueRef src,
				    unsigned writemask)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct tgsi_shader_info *info = &ctx->shader->selector->info;
	const unsigned component = var->data.location_frac;
	const bool is_patch = var->data.patch;
	unsigned driver_location = var->data.driver_location;
	LLVMValueRef dw_addr, stride;
	LLVMValueRef buffer, base, addr;
	LLVMValueRef values[4];
	bool skip_lds_store;
	bool is_tess_factor = false, is_tess_inner = false;

	driver_location = driver_location / 4;

	if (param_index) {
		/* Add the constant index to the indirect index */
		param_index = LLVMBuildAdd(ctx->ac.builder, param_index,
					   LLVMConstInt(ctx->i32, const_index, 0), "");
	} else {
		if (const_index != 0)
			param_index = LLVMConstInt(ctx->i32, const_index, 0);
	}

	if (!is_patch) {
		stride = get_tcs_out_vertex_dw_stride(ctx);
		dw_addr = get_tcs_out_current_patch_offset(ctx);
		dw_addr = get_dw_address_from_generic_indices(ctx, stride, dw_addr,
							      vertex_index, param_index,
							      driver_location,
							      info->output_semantic_name,
							      info->output_semantic_index,
							      is_patch);

		skip_lds_store = !info->reads_pervertex_outputs;
	} else {
		dw_addr = get_tcs_out_current_patch_data_offset(ctx);
		dw_addr = get_dw_address_from_generic_indices(ctx, NULL, dw_addr,
							      vertex_index, param_index,
							      driver_location,
							      info->output_semantic_name,
							      info->output_semantic_index,
							      is_patch);

		skip_lds_store = !info->reads_perpatch_outputs;

		if (!param_index) {
			int name = info->output_semantic_name[driver_location];

			/* Always write tess factors into LDS for the TCS epilog. */
			if (name == TGSI_SEMANTIC_TESSINNER ||
			    name == TGSI_SEMANTIC_TESSOUTER) {
				/* The epilog doesn't read LDS if invocation 0 defines tess factors. */
				skip_lds_store = !info->reads_tessfactor_outputs &&
						 ctx->shader->selector->tcs_info.tessfactors_are_def_in_all_invocs;
				is_tess_factor = true;
				is_tess_inner = name == TGSI_SEMANTIC_TESSINNER;
			}
		}
	}

	buffer = get_tess_ring_descriptor(ctx, TESS_OFFCHIP_RING_TCS);

	base = LLVMGetParam(ctx->main_fn, ctx->param_tcs_offchip_offset);

	addr = get_tcs_tes_buffer_address_from_generic_indices(ctx, vertex_index,
							       param_index, driver_location,
							       info->output_semantic_name,
							       info->output_semantic_index,
							       is_patch);

	for (unsigned chan = 0; chan < 4; chan++) {
		if (!(writemask & (1 << chan)))
			continue;
		LLVMValueRef value = ac_llvm_extract_elem(&ctx->ac, src, chan - component);

		/* Skip LDS stores if there is no LDS read of this output. */
		if (!skip_lds_store)
			lds_store(ctx, chan, dw_addr, value);

		value = ac_to_integer(&ctx->ac, value);
		values[chan] = value;

		if (writemask != 0xF && !is_tess_factor) {
			ac_build_buffer_store_dword(&ctx->ac, buffer, value, 1,
						    addr, base,
						    4 * chan, 1, 0, true, false);
		}

		/* Write tess factors into VGPRs for the epilog. */
		if (is_tess_factor &&
		    ctx->shader->selector->tcs_info.tessfactors_are_def_in_all_invocs) {
			if (!is_tess_inner) {
				LLVMBuildStore(ctx->ac.builder, value, /* outer */
					       ctx->invoc0_tess_factors[chan]);
			} else if (chan < 2) {
				LLVMBuildStore(ctx->ac.builder, value, /* inner */
					       ctx->invoc0_tess_factors[4 + chan]);
			}
		}
	}

	if (writemask == 0xF && !is_tess_factor) {
		LLVMValueRef value = ac_build_gather_values(&ctx->ac,
		                                            values, 4);
		ac_build_buffer_store_dword(&ctx->ac, buffer, value, 4, addr,
					    base, 0, 1, 0, true, false);
	}
}

LLVMValueRef si_llvm_load_input_gs(struct ac_shader_abi *abi,
				   unsigned input_index,
				   unsigned vtx_offset_param,
				   LLVMTypeRef type,
				   unsigned swizzle)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct lp_build_tgsi_context *bld_base = &ctx->bld_base;
	struct si_shader *shader = ctx->shader;
	LLVMValueRef vtx_offset, soffset;
	struct tgsi_shader_info *info = &shader->selector->info;
	unsigned semantic_name = info->input_semantic_name[input_index];
	unsigned semantic_index = info->input_semantic_index[input_index];
	unsigned param;
	LLVMValueRef value;

	param = si_shader_io_get_unique_index(semantic_name, semantic_index, false);

	/* GFX9 has the ESGS ring in LDS. */
	if (ctx->screen->info.chip_class >= GFX9) {
		unsigned index = vtx_offset_param;

		switch (index / 2) {
		case 0:
			vtx_offset = si_unpack_param(ctx, ctx->param_gs_vtx01_offset,
						  index % 2 ? 16 : 0, 16);
			break;
		case 1:
			vtx_offset = si_unpack_param(ctx, ctx->param_gs_vtx23_offset,
						  index % 2 ? 16 : 0, 16);
			break;
		case 2:
			vtx_offset = si_unpack_param(ctx, ctx->param_gs_vtx45_offset,
						  index % 2 ? 16 : 0, 16);
			break;
		default:
			assert(0);
			return NULL;
		}

		vtx_offset = LLVMBuildAdd(ctx->ac.builder, vtx_offset,
					  LLVMConstInt(ctx->i32, param * 4, 0), "");
		return lds_load(bld_base, type, swizzle, vtx_offset);
	}

	/* GFX6: input load from the ESGS ring in memory. */
	if (swizzle == ~0) {
		LLVMValueRef values[TGSI_NUM_CHANNELS];
		unsigned chan;
		for (chan = 0; chan < TGSI_NUM_CHANNELS; chan++) {
			values[chan] = si_llvm_load_input_gs(abi, input_index, vtx_offset_param,
							     type, chan);
		}
		return ac_build_gather_values(&ctx->ac, values,
					      TGSI_NUM_CHANNELS);
	}

	/* Get the vertex offset parameter on GFX6. */
	LLVMValueRef gs_vtx_offset = ctx->gs_vtx_offset[vtx_offset_param];

	vtx_offset = LLVMBuildMul(ctx->ac.builder, gs_vtx_offset,
				  LLVMConstInt(ctx->i32, 4, 0), "");

	soffset = LLVMConstInt(ctx->i32, (param * 4 + swizzle) * 256, 0);

	value = ac_build_buffer_load(&ctx->ac, ctx->esgs_ring, 1, ctx->i32_0,
				     vtx_offset, soffset, 0, 1, 0, true, false);
	if (llvm_type_is_64bit(ctx, type)) {
		LLVMValueRef value2;
		soffset = LLVMConstInt(ctx->i32, (param * 4 + swizzle + 1) * 256, 0);

		value2 = ac_build_buffer_load(&ctx->ac, ctx->esgs_ring, 1,
					      ctx->i32_0, vtx_offset, soffset,
					      0, 1, 0, true, false);
		return si_llvm_emit_fetch_64bit(bld_base, type, value, value2);
	}
	return LLVMBuildBitCast(ctx->ac.builder, value, type, "");
}

static LLVMValueRef si_nir_load_input_gs(struct ac_shader_abi *abi,
					 unsigned location,
					 unsigned driver_location,
					 unsigned component,
					 unsigned num_components,
					 unsigned vertex_index,
					 unsigned const_index,
					 LLVMTypeRef type)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);

	LLVMValueRef value[4];
	for (unsigned i = 0; i < num_components; i++) {
		unsigned offset = i;
		if (llvm_type_is_64bit(ctx, type))
			offset *= 2;

		offset += component;
		value[i + component] = si_llvm_load_input_gs(&ctx->abi, driver_location  / 4,
							     vertex_index, type, offset);
	}

	return ac_build_varying_gather_values(&ctx->ac, value, num_components, component);
}

static LLVMValueRef fetch_input_gs(
	struct lp_build_tgsi_context *bld_base,
	const struct tgsi_full_src_register *reg,
	enum tgsi_opcode_type type,
	unsigned swizzle_in)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	struct tgsi_shader_info *info = &ctx->shader->selector->info;
	unsigned swizzle = swizzle_in & 0xffff;

	unsigned semantic_name = info->input_semantic_name[reg->Register.Index];
	if (swizzle != ~0 && semantic_name == TGSI_SEMANTIC_PRIMID)
		return get_primitive_id(ctx, swizzle);

	if (!reg->Register.Dimension)
		return NULL;

	return si_llvm_load_input_gs(&ctx->abi, reg->Register.Index,
				     reg->Dimension.Index,
				     tgsi2llvmtype(bld_base, type),
				     swizzle);
}

static int lookup_interp_param_index(unsigned interpolate, unsigned location)
{
	switch (interpolate) {
	case TGSI_INTERPOLATE_CONSTANT:
		return 0;

	case TGSI_INTERPOLATE_LINEAR:
		if (location == TGSI_INTERPOLATE_LOC_SAMPLE)
			return SI_PARAM_LINEAR_SAMPLE;
		else if (location == TGSI_INTERPOLATE_LOC_CENTROID)
			return SI_PARAM_LINEAR_CENTROID;
		else
			return SI_PARAM_LINEAR_CENTER;
		break;
	case TGSI_INTERPOLATE_COLOR:
	case TGSI_INTERPOLATE_PERSPECTIVE:
		if (location == TGSI_INTERPOLATE_LOC_SAMPLE)
			return SI_PARAM_PERSP_SAMPLE;
		else if (location == TGSI_INTERPOLATE_LOC_CENTROID)
			return SI_PARAM_PERSP_CENTROID;
		else
			return SI_PARAM_PERSP_CENTER;
		break;
	default:
		fprintf(stderr, "Warning: Unhandled interpolation mode.\n");
		return -1;
	}
}

static LLVMValueRef si_build_fs_interp(struct si_shader_context *ctx,
				       unsigned attr_index, unsigned chan,
				       LLVMValueRef prim_mask,
				       LLVMValueRef i, LLVMValueRef j)
{
	if (i || j) {
		return ac_build_fs_interp(&ctx->ac,
					  LLVMConstInt(ctx->i32, chan, 0),
					  LLVMConstInt(ctx->i32, attr_index, 0),
					  prim_mask, i, j);
	}
	return ac_build_fs_interp_mov(&ctx->ac,
				      LLVMConstInt(ctx->i32, 2, 0), /* P0 */
				      LLVMConstInt(ctx->i32, chan, 0),
				      LLVMConstInt(ctx->i32, attr_index, 0),
				      prim_mask);
}

/**
 * Interpolate a fragment shader input.
 *
 * @param ctx		context
 * @param input_index		index of the input in hardware
 * @param semantic_name		TGSI_SEMANTIC_*
 * @param semantic_index	semantic index
 * @param num_interp_inputs	number of all interpolated inputs (= BCOLOR offset)
 * @param colors_read_mask	color components read (4 bits for each color, 8 bits in total)
 * @param interp_param		interpolation weights (i,j)
 * @param prim_mask		SI_PARAM_PRIM_MASK
 * @param face			SI_PARAM_FRONT_FACE
 * @param result		the return value (4 components)
 */
static void interp_fs_input(struct si_shader_context *ctx,
			    unsigned input_index,
			    unsigned semantic_name,
			    unsigned semantic_index,
			    unsigned num_interp_inputs,
			    unsigned colors_read_mask,
			    LLVMValueRef interp_param,
			    LLVMValueRef prim_mask,
			    LLVMValueRef face,
			    LLVMValueRef result[4])
{
	LLVMValueRef i = NULL, j = NULL;
	unsigned chan;

	/* fs.constant returns the param from the middle vertex, so it's not
	 * really useful for flat shading. It's meant to be used for custom
	 * interpolation (but the intrinsic can't fetch from the other two
	 * vertices).
	 *
	 * Luckily, it doesn't matter, because we rely on the FLAT_SHADE state
	 * to do the right thing. The only reason we use fs.constant is that
	 * fs.interp cannot be used on integers, because they can be equal
	 * to NaN.
	 *
	 * When interp is false we will use fs.constant or for newer llvm,
         * amdgcn.interp.mov.
	 */
	bool interp = interp_param != NULL;

	if (interp) {
		interp_param = LLVMBuildBitCast(ctx->ac.builder, interp_param,
						LLVMVectorType(ctx->f32, 2), "");

		i = LLVMBuildExtractElement(ctx->ac.builder, interp_param,
						ctx->i32_0, "");
		j = LLVMBuildExtractElement(ctx->ac.builder, interp_param,
						ctx->i32_1, "");
	}

	if (semantic_name == TGSI_SEMANTIC_COLOR &&
	    ctx->shader->key.part.ps.prolog.color_two_side) {
		LLVMValueRef is_face_positive;

		/* If BCOLOR0 is used, BCOLOR1 is at offset "num_inputs + 1",
		 * otherwise it's at offset "num_inputs".
		 */
		unsigned back_attr_offset = num_interp_inputs;
		if (semantic_index == 1 && colors_read_mask & 0xf)
			back_attr_offset += 1;

		is_face_positive = LLVMBuildICmp(ctx->ac.builder, LLVMIntNE,
						 face, ctx->i32_0, "");

		for (chan = 0; chan < TGSI_NUM_CHANNELS; chan++) {
			LLVMValueRef front, back;

			front = si_build_fs_interp(ctx,
						   input_index, chan,
						   prim_mask, i, j);
			back = si_build_fs_interp(ctx,
						  back_attr_offset, chan,
						  prim_mask, i, j);

			result[chan] = LLVMBuildSelect(ctx->ac.builder,
						is_face_positive,
						front,
						back,
						"");
		}
	} else if (semantic_name == TGSI_SEMANTIC_FOG) {
		result[0] = si_build_fs_interp(ctx, input_index,
					       0, prim_mask, i, j);
		result[1] =
		result[2] = LLVMConstReal(ctx->f32, 0.0f);
		result[3] = LLVMConstReal(ctx->f32, 1.0f);
	} else {
		for (chan = 0; chan < TGSI_NUM_CHANNELS; chan++) {
			result[chan] = si_build_fs_interp(ctx,
							  input_index, chan,
							  prim_mask, i, j);
		}
	}
}

void si_llvm_load_input_fs(
	struct si_shader_context *ctx,
	unsigned input_index,
	LLVMValueRef out[4])
{
	struct si_shader *shader = ctx->shader;
	struct tgsi_shader_info *info = &shader->selector->info;
	LLVMValueRef main_fn = ctx->main_fn;
	LLVMValueRef interp_param = NULL;
	int interp_param_idx;
	enum tgsi_semantic semantic_name = info->input_semantic_name[input_index];
	unsigned semantic_index = info->input_semantic_index[input_index];
	enum tgsi_interpolate_mode interp_mode = info->input_interpolate[input_index];
	enum tgsi_interpolate_loc interp_loc = info->input_interpolate_loc[input_index];

	/* Get colors from input VGPRs (set by the prolog). */
	if (semantic_name == TGSI_SEMANTIC_COLOR) {
		unsigned colors_read = shader->selector->info.colors_read;
		unsigned mask = colors_read >> (semantic_index * 4);
		unsigned offset = SI_PARAM_POS_FIXED_PT + 1 +
				  (semantic_index ? util_bitcount(colors_read & 0xf) : 0);
		LLVMValueRef undef = LLVMGetUndef(ctx->f32);

		out[0] = mask & 0x1 ? LLVMGetParam(main_fn, offset++) : undef;
		out[1] = mask & 0x2 ? LLVMGetParam(main_fn, offset++) : undef;
		out[2] = mask & 0x4 ? LLVMGetParam(main_fn, offset++) : undef;
		out[3] = mask & 0x8 ? LLVMGetParam(main_fn, offset++) : undef;
		return;
	}

	interp_param_idx = lookup_interp_param_index(interp_mode, interp_loc);
	if (interp_param_idx == -1)
		return;
	else if (interp_param_idx) {
		interp_param = LLVMGetParam(ctx->main_fn, interp_param_idx);
	}

	interp_fs_input(ctx, input_index, semantic_name,
			semantic_index, 0, /* this param is unused */
			shader->selector->info.colors_read, interp_param,
			ctx->abi.prim_mask,
			LLVMGetParam(main_fn, SI_PARAM_FRONT_FACE),
			&out[0]);
}

static void declare_input_fs(
	struct si_shader_context *ctx,
	unsigned input_index,
	const struct tgsi_full_declaration *decl,
	LLVMValueRef out[4])
{
	si_llvm_load_input_fs(ctx, input_index, out);
}

LLVMValueRef si_get_sample_id(struct si_shader_context *ctx)
{
	return si_unpack_param(ctx, SI_PARAM_ANCILLARY, 8, 4);
}

static LLVMValueRef get_base_vertex(struct ac_shader_abi *abi)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);

	/* For non-indexed draws, the base vertex set by the driver
	 * (for direct draws) or the CP (for indirect draws) is the
	 * first vertex ID, but GLSL expects 0 to be returned.
	 */
	LLVMValueRef vs_state = LLVMGetParam(ctx->main_fn,
					     ctx->param_vs_state_bits);
	LLVMValueRef indexed;

	indexed = LLVMBuildLShr(ctx->ac.builder, vs_state, ctx->i32_1, "");
	indexed = LLVMBuildTrunc(ctx->ac.builder, indexed, ctx->i1, "");

	return LLVMBuildSelect(ctx->ac.builder, indexed, ctx->abi.base_vertex,
			       ctx->i32_0, "");
}

static LLVMValueRef get_block_size(struct ac_shader_abi *abi)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);

	LLVMValueRef values[3];
	LLVMValueRef result;
	unsigned i;
	unsigned *properties = ctx->shader->selector->info.properties;

	if (properties[TGSI_PROPERTY_CS_FIXED_BLOCK_WIDTH] != 0) {
		unsigned sizes[3] = {
			properties[TGSI_PROPERTY_CS_FIXED_BLOCK_WIDTH],
			properties[TGSI_PROPERTY_CS_FIXED_BLOCK_HEIGHT],
			properties[TGSI_PROPERTY_CS_FIXED_BLOCK_DEPTH]
		};

		for (i = 0; i < 3; ++i)
			values[i] = LLVMConstInt(ctx->i32, sizes[i], 0);

		result = ac_build_gather_values(&ctx->ac, values, 3);
	} else {
		result = LLVMGetParam(ctx->main_fn, ctx->param_block_size);
	}

	return result;
}

/**
 * Load a dword from a constant buffer.
 */
static LLVMValueRef buffer_load_const(struct si_shader_context *ctx,
				      LLVMValueRef resource,
				      LLVMValueRef offset)
{
	return ac_build_buffer_load(&ctx->ac, resource, 1, NULL, offset, NULL,
				    0, 0, 0, true, true);
}

static LLVMValueRef load_sample_position(struct ac_shader_abi *abi, LLVMValueRef sample_id)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	LLVMValueRef desc = LLVMGetParam(ctx->main_fn, ctx->param_rw_buffers);
	LLVMValueRef buf_index = LLVMConstInt(ctx->i32, SI_PS_CONST_SAMPLE_POSITIONS, 0);
	LLVMValueRef resource = ac_build_load_to_sgpr(&ctx->ac, desc, buf_index);

	/* offset = sample_id * 8  (8 = 2 floats containing samplepos.xy) */
	LLVMValueRef offset0 = LLVMBuildMul(ctx->ac.builder, sample_id, LLVMConstInt(ctx->i32, 8, 0), "");
	LLVMValueRef offset1 = LLVMBuildAdd(ctx->ac.builder, offset0, LLVMConstInt(ctx->i32, 4, 0), "");

	LLVMValueRef pos[4] = {
		buffer_load_const(ctx, resource, offset0),
		buffer_load_const(ctx, resource, offset1),
		LLVMConstReal(ctx->f32, 0),
		LLVMConstReal(ctx->f32, 0)
	};

	return ac_build_gather_values(&ctx->ac, pos, 4);
}

static LLVMValueRef load_sample_mask_in(struct ac_shader_abi *abi)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	return ac_to_integer(&ctx->ac, abi->sample_coverage);
}

static LLVMValueRef si_load_tess_coord(struct ac_shader_abi *abi)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	LLVMValueRef coord[4] = {
		LLVMGetParam(ctx->main_fn, ctx->param_tes_u),
		LLVMGetParam(ctx->main_fn, ctx->param_tes_v),
		ctx->ac.f32_0,
		ctx->ac.f32_0
	};

	/* For triangles, the vector should be (u, v, 1-u-v). */
	if (ctx->shader->selector->info.properties[TGSI_PROPERTY_TES_PRIM_MODE] ==
	    PIPE_PRIM_TRIANGLES) {
		coord[2] = LLVMBuildFSub(ctx->ac.builder, ctx->ac.f32_1,
					 LLVMBuildFAdd(ctx->ac.builder,
						       coord[0], coord[1], ""), "");
	}
	return ac_build_gather_values(&ctx->ac, coord, 4);
}

static LLVMValueRef load_tess_level(struct si_shader_context *ctx,
				    unsigned semantic_name)
{
	LLVMValueRef base, addr;

	int param = si_shader_io_get_unique_index_patch(semantic_name, 0);

	base = LLVMGetParam(ctx->main_fn, ctx->param_tcs_offchip_offset);
	addr = get_tcs_tes_buffer_address(ctx, get_rel_patch_id(ctx), NULL,
					  LLVMConstInt(ctx->i32, param, 0));

	return buffer_load(&ctx->bld_base, ctx->f32,
			   ~0, ctx->tess_offchip_ring, base, addr, true);

}

static LLVMValueRef si_load_tess_level(struct ac_shader_abi *abi,
				       unsigned varying_id)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	unsigned semantic_name;

	switch (varying_id) {
	case VARYING_SLOT_TESS_LEVEL_INNER:
		semantic_name = TGSI_SEMANTIC_TESSINNER;
		break;
	case VARYING_SLOT_TESS_LEVEL_OUTER:
		semantic_name = TGSI_SEMANTIC_TESSOUTER;
		break;
	default:
		unreachable("unknown tess level");
	}

	return load_tess_level(ctx, semantic_name);

}

static LLVMValueRef si_load_patch_vertices_in(struct ac_shader_abi *abi)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	if (ctx->type == PIPE_SHADER_TESS_CTRL)
		return si_unpack_param(ctx, ctx->param_tcs_out_lds_layout, 13, 6);
	else if (ctx->type == PIPE_SHADER_TESS_EVAL)
		return get_num_tcs_out_vertices(ctx);
	else
		unreachable("invalid shader stage for TGSI_SEMANTIC_VERTICESIN");
}

void si_load_system_value(struct si_shader_context *ctx,
			  unsigned index,
			  const struct tgsi_full_declaration *decl)
{
	LLVMValueRef value = 0;

	assert(index < RADEON_LLVM_MAX_SYSTEM_VALUES);

	switch (decl->Semantic.Name) {
	case TGSI_SEMANTIC_INSTANCEID:
		value = ctx->abi.instance_id;
		break;

	case TGSI_SEMANTIC_VERTEXID:
		value = LLVMBuildAdd(ctx->ac.builder,
				     ctx->abi.vertex_id,
				     ctx->abi.base_vertex, "");
		break;

	case TGSI_SEMANTIC_VERTEXID_NOBASE:
		/* Unused. Clarify the meaning in indexed vs. non-indexed
		 * draws if this is ever used again. */
		assert(false);
		break;

	case TGSI_SEMANTIC_BASEVERTEX:
		value = get_base_vertex(&ctx->abi);
		break;

	case TGSI_SEMANTIC_BASEINSTANCE:
		value = ctx->abi.start_instance;
		break;

	case TGSI_SEMANTIC_DRAWID:
		value = ctx->abi.draw_id;
		break;

	case TGSI_SEMANTIC_INVOCATIONID:
		if (ctx->type == PIPE_SHADER_TESS_CTRL)
			value = unpack_llvm_param(ctx, ctx->abi.tcs_rel_ids, 8, 5);
		else if (ctx->type == PIPE_SHADER_GEOMETRY)
			value = ctx->abi.gs_invocation_id;
		else
			assert(!"INVOCATIONID not implemented");
		break;

	case TGSI_SEMANTIC_POSITION:
	{
		LLVMValueRef pos[4] = {
			LLVMGetParam(ctx->main_fn, SI_PARAM_POS_X_FLOAT),
			LLVMGetParam(ctx->main_fn, SI_PARAM_POS_Y_FLOAT),
			LLVMGetParam(ctx->main_fn, SI_PARAM_POS_Z_FLOAT),
			ac_build_fdiv(&ctx->ac, ctx->ac.f32_1,
				      LLVMGetParam(ctx->main_fn, SI_PARAM_POS_W_FLOAT)),
		};
		value = ac_build_gather_values(&ctx->ac, pos, 4);
		break;
	}

	case TGSI_SEMANTIC_FACE:
		value = ctx->abi.front_face;
		break;

	case TGSI_SEMANTIC_SAMPLEID:
		value = si_get_sample_id(ctx);
		break;

	case TGSI_SEMANTIC_SAMPLEPOS: {
		LLVMValueRef pos[4] = {
			LLVMGetParam(ctx->main_fn, SI_PARAM_POS_X_FLOAT),
			LLVMGetParam(ctx->main_fn, SI_PARAM_POS_Y_FLOAT),
			LLVMConstReal(ctx->f32, 0),
			LLVMConstReal(ctx->f32, 0)
		};
		pos[0] = ac_build_fract(&ctx->ac, pos[0], 32);
		pos[1] = ac_build_fract(&ctx->ac, pos[1], 32);
		value = ac_build_gather_values(&ctx->ac, pos, 4);
		break;
	}

	case TGSI_SEMANTIC_SAMPLEMASK:
		/* This can only occur with the OpenGL Core profile, which
		 * doesn't support smoothing.
		 */
		value = LLVMGetParam(ctx->main_fn, SI_PARAM_SAMPLE_COVERAGE);
		break;

	case TGSI_SEMANTIC_TESSCOORD:
		value = si_load_tess_coord(&ctx->abi);
		break;

	case TGSI_SEMANTIC_VERTICESIN:
		value = si_load_patch_vertices_in(&ctx->abi);
		break;

	case TGSI_SEMANTIC_TESSINNER:
	case TGSI_SEMANTIC_TESSOUTER:
		value = load_tess_level(ctx, decl->Semantic.Name);
		break;

	case TGSI_SEMANTIC_DEFAULT_TESSOUTER_SI:
	case TGSI_SEMANTIC_DEFAULT_TESSINNER_SI:
	{
		LLVMValueRef buf, slot, val[4];
		int i, offset;

		slot = LLVMConstInt(ctx->i32, SI_HS_CONST_DEFAULT_TESS_LEVELS, 0);
		buf = LLVMGetParam(ctx->main_fn, ctx->param_rw_buffers);
		buf = ac_build_load_to_sgpr(&ctx->ac, buf, slot);
		offset = decl->Semantic.Name == TGSI_SEMANTIC_DEFAULT_TESSINNER_SI ? 4 : 0;

		for (i = 0; i < 4; i++)
			val[i] = buffer_load_const(ctx, buf,
						   LLVMConstInt(ctx->i32, (offset + i) * 4, 0));
		value = ac_build_gather_values(&ctx->ac, val, 4);
		break;
	}

	case TGSI_SEMANTIC_PRIMID:
		value = get_primitive_id(ctx, 0);
		break;

	case TGSI_SEMANTIC_GRID_SIZE:
		value = ctx->abi.num_work_groups;
		break;

	case TGSI_SEMANTIC_BLOCK_SIZE:
		value = get_block_size(&ctx->abi);
		break;

	case TGSI_SEMANTIC_BLOCK_ID:
	{
		LLVMValueRef values[3];

		for (int i = 0; i < 3; i++) {
			values[i] = ctx->i32_0;
			if (ctx->abi.workgroup_ids[i]) {
				values[i] = ctx->abi.workgroup_ids[i];
			}
		}
		value = ac_build_gather_values(&ctx->ac, values, 3);
		break;
	}

	case TGSI_SEMANTIC_THREAD_ID:
		value = ctx->abi.local_invocation_ids;
		break;

	case TGSI_SEMANTIC_HELPER_INVOCATION:
		value = ac_build_intrinsic(&ctx->ac,
					   "llvm.amdgcn.ps.live",
					   ctx->i1, NULL, 0,
					   AC_FUNC_ATTR_READNONE);
		value = LLVMBuildNot(ctx->ac.builder, value, "");
		value = LLVMBuildSExt(ctx->ac.builder, value, ctx->i32, "");
		break;

	case TGSI_SEMANTIC_SUBGROUP_SIZE:
		value = LLVMConstInt(ctx->i32, 64, 0);
		break;

	case TGSI_SEMANTIC_SUBGROUP_INVOCATION:
		value = ac_get_thread_id(&ctx->ac);
		break;

	case TGSI_SEMANTIC_SUBGROUP_EQ_MASK:
	{
		LLVMValueRef id = ac_get_thread_id(&ctx->ac);
		id = LLVMBuildZExt(ctx->ac.builder, id, ctx->i64, "");
		value = LLVMBuildShl(ctx->ac.builder, LLVMConstInt(ctx->i64, 1, 0), id, "");
		value = LLVMBuildBitCast(ctx->ac.builder, value, ctx->v2i32, "");
		break;
	}

	case TGSI_SEMANTIC_SUBGROUP_GE_MASK:
	case TGSI_SEMANTIC_SUBGROUP_GT_MASK:
	case TGSI_SEMANTIC_SUBGROUP_LE_MASK:
	case TGSI_SEMANTIC_SUBGROUP_LT_MASK:
	{
		LLVMValueRef id = ac_get_thread_id(&ctx->ac);
		if (decl->Semantic.Name == TGSI_SEMANTIC_SUBGROUP_GT_MASK ||
		    decl->Semantic.Name == TGSI_SEMANTIC_SUBGROUP_LE_MASK) {
			/* All bits set except LSB */
			value = LLVMConstInt(ctx->i64, -2, 0);
		} else {
			/* All bits set */
			value = LLVMConstInt(ctx->i64, -1, 0);
		}
		id = LLVMBuildZExt(ctx->ac.builder, id, ctx->i64, "");
		value = LLVMBuildShl(ctx->ac.builder, value, id, "");
		if (decl->Semantic.Name == TGSI_SEMANTIC_SUBGROUP_LE_MASK ||
		    decl->Semantic.Name == TGSI_SEMANTIC_SUBGROUP_LT_MASK)
			value = LLVMBuildNot(ctx->ac.builder, value, "");
		value = LLVMBuildBitCast(ctx->ac.builder, value, ctx->v2i32, "");
		break;
	}

	case TGSI_SEMANTIC_CS_USER_DATA:
		value = LLVMGetParam(ctx->main_fn, ctx->param_cs_user_data);
		break;

	default:
		assert(!"unknown system value");
		return;
	}

	ctx->system_values[index] = value;
}

void si_declare_compute_memory(struct si_shader_context *ctx)
{
	struct si_shader_selector *sel = ctx->shader->selector;
	unsigned lds_size = sel->info.properties[TGSI_PROPERTY_CS_LOCAL_SIZE];

	LLVMTypeRef i8p = LLVMPointerType(ctx->i8, AC_ADDR_SPACE_LDS);
	LLVMValueRef var;

	assert(!ctx->ac.lds);

	var = LLVMAddGlobalInAddressSpace(ctx->ac.module,
	                                  LLVMArrayType(ctx->i8, lds_size),
	                                  "compute_lds",
	                                  AC_ADDR_SPACE_LDS);
	LLVMSetAlignment(var, 4);

	ctx->ac.lds = LLVMBuildBitCast(ctx->ac.builder, var, i8p, "");
}

void si_tgsi_declare_compute_memory(struct si_shader_context *ctx,
				    const struct tgsi_full_declaration *decl)
{
	assert(decl->Declaration.MemType == TGSI_MEMORY_TYPE_SHARED);
	assert(decl->Range.First == decl->Range.Last);

	si_declare_compute_memory(ctx);
}

static LLVMValueRef load_const_buffer_desc_fast_path(struct si_shader_context *ctx)
{
	LLVMValueRef ptr =
		LLVMGetParam(ctx->main_fn, ctx->param_const_and_shader_buffers);
	struct si_shader_selector *sel = ctx->shader->selector;

	/* Do the bounds checking with a descriptor, because
	 * doing computation and manual bounds checking of 64-bit
	 * addresses generates horrible VALU code with very high
	 * VGPR usage and very low SIMD occupancy.
	 */
	ptr = LLVMBuildPtrToInt(ctx->ac.builder, ptr, ctx->ac.intptr, "");

	LLVMValueRef desc0, desc1;
	if (HAVE_32BIT_POINTERS) {
		desc0 = ptr;
		desc1 = LLVMConstInt(ctx->i32,
				     S_008F04_BASE_ADDRESS_HI(ctx->screen->info.address32_hi), 0);
	} else {
		ptr = LLVMBuildBitCast(ctx->ac.builder, ptr, ctx->v2i32, "");
		desc0 = LLVMBuildExtractElement(ctx->ac.builder, ptr, ctx->i32_0, "");
		desc1 = LLVMBuildExtractElement(ctx->ac.builder, ptr, ctx->i32_1, "");
		/* Mask out all bits except BASE_ADDRESS_HI. */
		desc1 = LLVMBuildAnd(ctx->ac.builder, desc1,
				     LLVMConstInt(ctx->i32, ~C_008F04_BASE_ADDRESS_HI, 0), "");
	}

	LLVMValueRef desc_elems[] = {
		desc0,
		desc1,
		LLVMConstInt(ctx->i32, (sel->info.const_file_max[0] + 1) * 16, 0),
		LLVMConstInt(ctx->i32,
			S_008F0C_DST_SEL_X(V_008F0C_SQ_SEL_X) |
			S_008F0C_DST_SEL_Y(V_008F0C_SQ_SEL_Y) |
			S_008F0C_DST_SEL_Z(V_008F0C_SQ_SEL_Z) |
			S_008F0C_DST_SEL_W(V_008F0C_SQ_SEL_W) |
			S_008F0C_NUM_FORMAT(V_008F0C_BUF_NUM_FORMAT_FLOAT) |
			S_008F0C_DATA_FORMAT(V_008F0C_BUF_DATA_FORMAT_32), 0)
	};

	return ac_build_gather_values(&ctx->ac, desc_elems, 4);
}

static LLVMValueRef load_const_buffer_desc(struct si_shader_context *ctx, int i)
{
	LLVMValueRef list_ptr = LLVMGetParam(ctx->main_fn,
					     ctx->param_const_and_shader_buffers);

	return ac_build_load_to_sgpr(&ctx->ac, list_ptr,
				     LLVMConstInt(ctx->i32, si_get_constbuf_slot(i), 0));
}

static LLVMValueRef load_ubo(struct ac_shader_abi *abi, LLVMValueRef index)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct si_shader_selector *sel = ctx->shader->selector;

	LLVMValueRef ptr = LLVMGetParam(ctx->main_fn, ctx->param_const_and_shader_buffers);

	if (sel->info.const_buffers_declared == 1 &&
	    sel->info.shader_buffers_declared == 0) {
		return load_const_buffer_desc_fast_path(ctx);
	}

	index = si_llvm_bound_index(ctx, index, ctx->num_const_buffers);
	index = LLVMBuildAdd(ctx->ac.builder, index,
			     LLVMConstInt(ctx->i32, SI_NUM_SHADER_BUFFERS, 0), "");

	return ac_build_load_to_sgpr(&ctx->ac, ptr, index);
}

static LLVMValueRef
load_ssbo(struct ac_shader_abi *abi, LLVMValueRef index, bool write)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	LLVMValueRef rsrc_ptr = LLVMGetParam(ctx->main_fn,
					     ctx->param_const_and_shader_buffers);

	index = si_llvm_bound_index(ctx, index, ctx->num_shader_buffers);
	index = LLVMBuildSub(ctx->ac.builder,
			     LLVMConstInt(ctx->i32, SI_NUM_SHADER_BUFFERS - 1, 0),
			     index, "");

	return ac_build_load_to_sgpr(&ctx->ac, rsrc_ptr, index);
}

static LLVMValueRef fetch_constant(
	struct lp_build_tgsi_context *bld_base,
	const struct tgsi_full_src_register *reg,
	enum tgsi_opcode_type type,
	unsigned swizzle_in)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	struct si_shader_selector *sel = ctx->shader->selector;
	const struct tgsi_ind_register *ireg = &reg->Indirect;
	unsigned buf, idx;
	unsigned swizzle = swizzle_in & 0xffff;

	LLVMValueRef addr, bufp;

	if (swizzle_in == LP_CHAN_ALL) {
		unsigned chan;
		LLVMValueRef values[4];
		for (chan = 0; chan < TGSI_NUM_CHANNELS; ++chan)
			values[chan] = fetch_constant(bld_base, reg, type, chan);

		return ac_build_gather_values(&ctx->ac, values, 4);
	}

	/* Split 64-bit loads. */
	if (tgsi_type_is_64bit(type)) {
		LLVMValueRef lo, hi;

		lo = fetch_constant(bld_base, reg, TGSI_TYPE_UNSIGNED, swizzle);
		hi = fetch_constant(bld_base, reg, TGSI_TYPE_UNSIGNED, (swizzle_in >> 16));
		return si_llvm_emit_fetch_64bit(bld_base, tgsi2llvmtype(bld_base, type),
						lo, hi);
	}

	idx = reg->Register.Index * 4 + swizzle;
	if (reg->Register.Indirect) {
		addr = si_get_indirect_index(ctx, ireg, 16, idx * 4);
	} else {
		addr = LLVMConstInt(ctx->i32, idx * 4, 0);
	}

	/* Fast path when user data SGPRs point to constant buffer 0 directly. */
	if (sel->info.const_buffers_declared == 1 &&
	    sel->info.shader_buffers_declared == 0) {
		LLVMValueRef desc = load_const_buffer_desc_fast_path(ctx);
		LLVMValueRef result = buffer_load_const(ctx, desc, addr);
		return bitcast(bld_base, type, result);
	}

	assert(reg->Register.Dimension);
	buf = reg->Dimension.Index;

	if (reg->Dimension.Indirect) {
		LLVMValueRef ptr = LLVMGetParam(ctx->main_fn, ctx->param_const_and_shader_buffers);
		LLVMValueRef index;
		index = si_get_bounded_indirect_index(ctx, &reg->DimIndirect,
						      reg->Dimension.Index,
						      ctx->num_const_buffers);
		index = LLVMBuildAdd(ctx->ac.builder, index,
				     LLVMConstInt(ctx->i32, SI_NUM_SHADER_BUFFERS, 0), "");
		bufp = ac_build_load_to_sgpr(&ctx->ac, ptr, index);
	} else
		bufp = load_const_buffer_desc(ctx, buf);

	return bitcast(bld_base, type, buffer_load_const(ctx, bufp, addr));
}

/* Initialize arguments for the shader export intrinsic */
static void si_llvm_init_export_args(struct si_shader_context *ctx,
				     LLVMValueRef *values,
				     unsigned target,
				     struct ac_export_args *args)
{
	LLVMValueRef f32undef = LLVMGetUndef(ctx->ac.f32);
	unsigned spi_shader_col_format = V_028714_SPI_SHADER_32_ABGR;
	unsigned chan;
	bool is_int8, is_int10;

	/* Default is 0xf. Adjusted below depending on the format. */
	args->enabled_channels = 0xf; /* writemask */

	/* Specify whether the EXEC mask represents the valid mask */
	args->valid_mask = 0;

	/* Specify whether this is the last export */
	args->done = 0;

	/* Specify the target we are exporting */
	args->target = target;

	if (ctx->type == PIPE_SHADER_FRAGMENT) {
		const struct si_shader_key *key = &ctx->shader->key;
		unsigned col_formats = key->part.ps.epilog.spi_shader_col_format;
		int cbuf = target - V_008DFC_SQ_EXP_MRT;

		assert(cbuf >= 0 && cbuf < 8);
		spi_shader_col_format = (col_formats >> (cbuf * 4)) & 0xf;
		is_int8 = (key->part.ps.epilog.color_is_int8 >> cbuf) & 0x1;
		is_int10 = (key->part.ps.epilog.color_is_int10 >> cbuf) & 0x1;
	}

	args->compr = false;
	args->out[0] = f32undef;
	args->out[1] = f32undef;
	args->out[2] = f32undef;
	args->out[3] = f32undef;

	LLVMValueRef (*packf)(struct ac_llvm_context *ctx, LLVMValueRef args[2]) = NULL;
	LLVMValueRef (*packi)(struct ac_llvm_context *ctx, LLVMValueRef args[2],
			      unsigned bits, bool hi) = NULL;

	switch (spi_shader_col_format) {
	case V_028714_SPI_SHADER_ZERO:
		args->enabled_channels = 0; /* writemask */
		args->target = V_008DFC_SQ_EXP_NULL;
		break;

	case V_028714_SPI_SHADER_32_R:
		args->enabled_channels = 1; /* writemask */
		args->out[0] = values[0];
		break;

	case V_028714_SPI_SHADER_32_GR:
		args->enabled_channels = 0x3; /* writemask */
		args->out[0] = values[0];
		args->out[1] = values[1];
		break;

	case V_028714_SPI_SHADER_32_AR:
		args->enabled_channels = 0x9; /* writemask */
		args->out[0] = values[0];
		args->out[3] = values[3];
		break;

	case V_028714_SPI_SHADER_FP16_ABGR:
		packf = ac_build_cvt_pkrtz_f16;
		break;

	case V_028714_SPI_SHADER_UNORM16_ABGR:
		packf = ac_build_cvt_pknorm_u16;
		break;

	case V_028714_SPI_SHADER_SNORM16_ABGR:
		packf = ac_build_cvt_pknorm_i16;
		break;

	case V_028714_SPI_SHADER_UINT16_ABGR:
		packi = ac_build_cvt_pk_u16;
		break;

	case V_028714_SPI_SHADER_SINT16_ABGR:
		packi = ac_build_cvt_pk_i16;
		break;

	case V_028714_SPI_SHADER_32_ABGR:
		memcpy(&args->out[0], values, sizeof(values[0]) * 4);
		break;
	}

	/* Pack f16 or norm_i16/u16. */
	if (packf) {
		for (chan = 0; chan < 2; chan++) {
			LLVMValueRef pack_args[2] = {
				values[2 * chan],
				values[2 * chan + 1]
			};
			LLVMValueRef packed;

			packed = packf(&ctx->ac, pack_args);
			args->out[chan] = ac_to_float(&ctx->ac, packed);
		}
		args->compr = 1; /* COMPR flag */
	}
	/* Pack i16/u16. */
	if (packi) {
		for (chan = 0; chan < 2; chan++) {
			LLVMValueRef pack_args[2] = {
				ac_to_integer(&ctx->ac, values[2 * chan]),
				ac_to_integer(&ctx->ac, values[2 * chan + 1])
			};
			LLVMValueRef packed;

			packed = packi(&ctx->ac, pack_args,
				       is_int8 ? 8 : is_int10 ? 10 : 16,
				       chan == 1);
			args->out[chan] = ac_to_float(&ctx->ac, packed);
		}
		args->compr = 1; /* COMPR flag */
	}
}

static void si_alpha_test(struct lp_build_tgsi_context *bld_base,
			  LLVMValueRef alpha)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);

	if (ctx->shader->key.part.ps.epilog.alpha_func != PIPE_FUNC_NEVER) {
		static LLVMRealPredicate cond_map[PIPE_FUNC_ALWAYS + 1] = {
			[PIPE_FUNC_LESS] = LLVMRealOLT,
			[PIPE_FUNC_EQUAL] = LLVMRealOEQ,
			[PIPE_FUNC_LEQUAL] = LLVMRealOLE,
			[PIPE_FUNC_GREATER] = LLVMRealOGT,
			[PIPE_FUNC_NOTEQUAL] = LLVMRealONE,
			[PIPE_FUNC_GEQUAL] = LLVMRealOGE,
		};
		LLVMRealPredicate cond = cond_map[ctx->shader->key.part.ps.epilog.alpha_func];
		assert(cond);

		LLVMValueRef alpha_ref = LLVMGetParam(ctx->main_fn,
				SI_PARAM_ALPHA_REF);
		LLVMValueRef alpha_pass =
			LLVMBuildFCmp(ctx->ac.builder, cond, alpha, alpha_ref, "");
		ac_build_kill_if_false(&ctx->ac, alpha_pass);
	} else {
		ac_build_kill_if_false(&ctx->ac, ctx->i1false);
	}
}

static LLVMValueRef si_scale_alpha_by_sample_mask(struct lp_build_tgsi_context *bld_base,
						  LLVMValueRef alpha,
						  unsigned samplemask_param)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef coverage;

	/* alpha = alpha * popcount(coverage) / SI_NUM_SMOOTH_AA_SAMPLES */
	coverage = LLVMGetParam(ctx->main_fn,
				samplemask_param);
	coverage = ac_to_integer(&ctx->ac, coverage);

	coverage = ac_build_intrinsic(&ctx->ac, "llvm.ctpop.i32",
				   ctx->i32,
				   &coverage, 1, AC_FUNC_ATTR_READNONE);

	coverage = LLVMBuildUIToFP(ctx->ac.builder, coverage,
				   ctx->f32, "");

	coverage = LLVMBuildFMul(ctx->ac.builder, coverage,
				 LLVMConstReal(ctx->f32,
					1.0 / SI_NUM_SMOOTH_AA_SAMPLES), "");

	return LLVMBuildFMul(ctx->ac.builder, alpha, coverage, "");
}

static void si_llvm_emit_clipvertex(struct si_shader_context *ctx,
				    struct ac_export_args *pos, LLVMValueRef *out_elts)
{
	unsigned reg_index;
	unsigned chan;
	unsigned const_chan;
	LLVMValueRef base_elt;
	LLVMValueRef ptr = LLVMGetParam(ctx->main_fn, ctx->param_rw_buffers);
	LLVMValueRef constbuf_index = LLVMConstInt(ctx->i32,
						   SI_VS_CONST_CLIP_PLANES, 0);
	LLVMValueRef const_resource = ac_build_load_to_sgpr(&ctx->ac, ptr, constbuf_index);

	for (reg_index = 0; reg_index < 2; reg_index ++) {
		struct ac_export_args *args = &pos[2 + reg_index];

		args->out[0] =
		args->out[1] =
		args->out[2] =
		args->out[3] = LLVMConstReal(ctx->f32, 0.0f);

		/* Compute dot products of position and user clip plane vectors */
		for (chan = 0; chan < TGSI_NUM_CHANNELS; chan++) {
			for (const_chan = 0; const_chan < TGSI_NUM_CHANNELS; const_chan++) {
				LLVMValueRef addr =
					LLVMConstInt(ctx->i32, ((reg_index * 4 + chan) * 4 +
								const_chan) * 4, 0);
				base_elt = buffer_load_const(ctx, const_resource,
							     addr);
				args->out[chan] = ac_build_fmad(&ctx->ac, base_elt,
								out_elts[const_chan], args->out[chan]);
			}
		}

		args->enabled_channels = 0xf;
		args->valid_mask = 0;
		args->done = 0;
		args->target = V_008DFC_SQ_EXP_POS + 2 + reg_index;
		args->compr = 0;
	}
}

static void si_dump_streamout(struct pipe_stream_output_info *so)
{
	unsigned i;

	if (so->num_outputs)
		fprintf(stderr, "STREAMOUT\n");

	for (i = 0; i < so->num_outputs; i++) {
		unsigned mask = ((1 << so->output[i].num_components) - 1) <<
				so->output[i].start_component;
		fprintf(stderr, "  %i: BUF%i[%i..%i] <- OUT[%i].%s%s%s%s\n",
			i, so->output[i].output_buffer,
			so->output[i].dst_offset, so->output[i].dst_offset + so->output[i].num_components - 1,
			so->output[i].register_index,
			mask & 1 ? "x" : "",
		        mask & 2 ? "y" : "",
		        mask & 4 ? "z" : "",
		        mask & 8 ? "w" : "");
	}
}

static void emit_streamout_output(struct si_shader_context *ctx,
				  LLVMValueRef const *so_buffers,
				  LLVMValueRef const *so_write_offsets,
				  struct pipe_stream_output *stream_out,
				  struct si_shader_output_values *shader_out)
{
	unsigned buf_idx = stream_out->output_buffer;
	unsigned start = stream_out->start_component;
	unsigned num_comps = stream_out->num_components;
	LLVMValueRef out[4];

	assert(num_comps && num_comps <= 4);
	if (!num_comps || num_comps > 4)
		return;

	/* Load the output as int. */
	for (int j = 0; j < num_comps; j++) {
		assert(stream_out->stream == shader_out->vertex_stream[start + j]);

		out[j] = ac_to_integer(&ctx->ac, shader_out->values[start + j]);
	}

	/* Pack the output. */
	LLVMValueRef vdata = NULL;

	switch (num_comps) {
	case 1: /* as i32 */
		vdata = out[0];
		break;
	case 2: /* as v2i32 */
	case 3: /* as v4i32 (aligned to 4) */
		out[3] = LLVMGetUndef(ctx->i32);
		/* fall through */
	case 4: /* as v4i32 */
		vdata = ac_build_gather_values(&ctx->ac, out, util_next_power_of_two(num_comps));
		break;
	}

	ac_build_buffer_store_dword(&ctx->ac, so_buffers[buf_idx],
				    vdata, num_comps,
				    so_write_offsets[buf_idx],
				    ctx->i32_0,
				    stream_out->dst_offset * 4, 1, 1, true, false);
}

/**
 * Write streamout data to buffers for vertex stream @p stream (different
 * vertex streams can occur for GS copy shaders).
 */
static void si_llvm_emit_streamout(struct si_shader_context *ctx,
				   struct si_shader_output_values *outputs,
				   unsigned noutput, unsigned stream)
{
	struct si_shader_selector *sel = ctx->shader->selector;
	struct pipe_stream_output_info *so = &sel->so;
	LLVMBuilderRef builder = ctx->ac.builder;
	int i;
	struct lp_build_if_state if_ctx;

	/* Get bits [22:16], i.e. (so_param >> 16) & 127; */
	LLVMValueRef so_vtx_count =
		si_unpack_param(ctx, ctx->param_streamout_config, 16, 7);

	LLVMValueRef tid = ac_get_thread_id(&ctx->ac);

	/* can_emit = tid < so_vtx_count; */
	LLVMValueRef can_emit =
		LLVMBuildICmp(builder, LLVMIntULT, tid, so_vtx_count, "");

	/* Emit the streamout code conditionally. This actually avoids
	 * out-of-bounds buffer access. The hw tells us via the SGPR
	 * (so_vtx_count) which threads are allowed to emit streamout data. */
	lp_build_if(&if_ctx, &ctx->gallivm, can_emit);
	{
		/* The buffer offset is computed as follows:
		 *   ByteOffset = streamout_offset[buffer_id]*4 +
		 *                (streamout_write_index + thread_id)*stride[buffer_id] +
		 *                attrib_offset
                 */

		LLVMValueRef so_write_index =
			LLVMGetParam(ctx->main_fn,
				     ctx->param_streamout_write_index);

		/* Compute (streamout_write_index + thread_id). */
		so_write_index = LLVMBuildAdd(builder, so_write_index, tid, "");

		/* Load the descriptor and compute the write offset for each
		 * enabled buffer. */
		LLVMValueRef so_write_offset[4] = {};
		LLVMValueRef so_buffers[4];
		LLVMValueRef buf_ptr = LLVMGetParam(ctx->main_fn,
						    ctx->param_rw_buffers);

		for (i = 0; i < 4; i++) {
			if (!so->stride[i])
				continue;

			LLVMValueRef offset = LLVMConstInt(ctx->i32,
							   SI_VS_STREAMOUT_BUF0 + i, 0);

			so_buffers[i] = ac_build_load_to_sgpr(&ctx->ac, buf_ptr, offset);

			LLVMValueRef so_offset = LLVMGetParam(ctx->main_fn,
							      ctx->param_streamout_offset[i]);
			so_offset = LLVMBuildMul(builder, so_offset, LLVMConstInt(ctx->i32, 4, 0), "");

			so_write_offset[i] = ac_build_imad(&ctx->ac, so_write_index,
							   LLVMConstInt(ctx->i32, so->stride[i]*4, 0),
							   so_offset);
		}

		/* Write streamout data. */
		for (i = 0; i < so->num_outputs; i++) {
			unsigned reg = so->output[i].register_index;

			if (reg >= noutput)
				continue;

			if (stream != so->output[i].stream)
				continue;

			emit_streamout_output(ctx, so_buffers, so_write_offset,
					      &so->output[i], &outputs[reg]);
		}
	}
	lp_build_endif(&if_ctx);
}

static void si_export_param(struct si_shader_context *ctx, unsigned index,
			    LLVMValueRef *values)
{
	struct ac_export_args args;

	si_llvm_init_export_args(ctx, values,
				 V_008DFC_SQ_EXP_PARAM + index, &args);
	ac_build_export(&ctx->ac, &args);
}

static void si_build_param_exports(struct si_shader_context *ctx,
				   struct si_shader_output_values *outputs,
			           unsigned noutput)
{
	struct si_shader *shader = ctx->shader;
	unsigned param_count = 0;

	for (unsigned i = 0; i < noutput; i++) {
		unsigned semantic_name = outputs[i].semantic_name;
		unsigned semantic_index = outputs[i].semantic_index;

		if (outputs[i].vertex_stream[0] != 0 &&
		    outputs[i].vertex_stream[1] != 0 &&
		    outputs[i].vertex_stream[2] != 0 &&
		    outputs[i].vertex_stream[3] != 0)
			continue;

		switch (semantic_name) {
		case TGSI_SEMANTIC_LAYER:
		case TGSI_SEMANTIC_VIEWPORT_INDEX:
		case TGSI_SEMANTIC_CLIPDIST:
		case TGSI_SEMANTIC_COLOR:
		case TGSI_SEMANTIC_BCOLOR:
		case TGSI_SEMANTIC_PRIMID:
		case TGSI_SEMANTIC_FOG:
		case TGSI_SEMANTIC_TEXCOORD:
		case TGSI_SEMANTIC_GENERIC:
			break;
		default:
			continue;
		}

		if ((semantic_name != TGSI_SEMANTIC_GENERIC ||
		     semantic_index < SI_MAX_IO_GENERIC) &&
		    shader->key.opt.kill_outputs &
		    (1ull << si_shader_io_get_unique_index(semantic_name,
							   semantic_index, true)))
			continue;

		si_export_param(ctx, param_count, outputs[i].values);

		assert(i < ARRAY_SIZE(shader->info.vs_output_param_offset));
		shader->info.vs_output_param_offset[i] = param_count++;
	}

	shader->info.nr_param_exports = param_count;
}

/* Generate export instructions for hardware VS shader stage */
static void si_llvm_export_vs(struct si_shader_context *ctx,
			      struct si_shader_output_values *outputs,
			      unsigned noutput)
{
	struct si_shader *shader = ctx->shader;
	struct ac_export_args pos_args[4] = {};
	LLVMValueRef psize_value = NULL, edgeflag_value = NULL, layer_value = NULL, viewport_index_value = NULL;
	unsigned pos_idx;
	int i;

	/* Build position exports. */
	for (i = 0; i < noutput; i++) {
		switch (outputs[i].semantic_name) {
		case TGSI_SEMANTIC_POSITION:
			si_llvm_init_export_args(ctx, outputs[i].values,
						 V_008DFC_SQ_EXP_POS, &pos_args[0]);
			break;
		case TGSI_SEMANTIC_PSIZE:
			psize_value = outputs[i].values[0];
			break;
		case TGSI_SEMANTIC_LAYER:
			layer_value = outputs[i].values[0];
			break;
		case TGSI_SEMANTIC_VIEWPORT_INDEX:
			viewport_index_value = outputs[i].values[0];
			break;
		case TGSI_SEMANTIC_EDGEFLAG:
			edgeflag_value = outputs[i].values[0];
			break;
		case TGSI_SEMANTIC_CLIPDIST:
			if (!shader->key.opt.clip_disable) {
				unsigned index = 2 + outputs[i].semantic_index;
				si_llvm_init_export_args(ctx, outputs[i].values,
							 V_008DFC_SQ_EXP_POS + index,
							 &pos_args[index]);
			}
			break;
		case TGSI_SEMANTIC_CLIPVERTEX:
			if (!shader->key.opt.clip_disable) {
				si_llvm_emit_clipvertex(ctx, pos_args,
							outputs[i].values);
			}
			break;
		}
	}

	/* We need to add the position output manually if it's missing. */
	if (!pos_args[0].out[0]) {
		pos_args[0].enabled_channels = 0xf; /* writemask */
		pos_args[0].valid_mask = 0; /* EXEC mask */
		pos_args[0].done = 0; /* last export? */
		pos_args[0].target = V_008DFC_SQ_EXP_POS;
		pos_args[0].compr = 0; /* COMPR flag */
		pos_args[0].out[0] = ctx->ac.f32_0; /* X */
		pos_args[0].out[1] = ctx->ac.f32_0; /* Y */
		pos_args[0].out[2] = ctx->ac.f32_0; /* Z */
		pos_args[0].out[3] = ctx->ac.f32_1;  /* W */
	}

	/* Write the misc vector (point size, edgeflag, layer, viewport). */
	if (shader->selector->info.writes_psize ||
	    shader->selector->info.writes_edgeflag ||
	    shader->selector->info.writes_viewport_index ||
	    shader->selector->info.writes_layer) {
		pos_args[1].enabled_channels = shader->selector->info.writes_psize |
					       (shader->selector->info.writes_edgeflag << 1) |
					       (shader->selector->info.writes_layer << 2);

		pos_args[1].valid_mask = 0; /* EXEC mask */
		pos_args[1].done = 0; /* last export? */
		pos_args[1].target = V_008DFC_SQ_EXP_POS + 1;
		pos_args[1].compr = 0; /* COMPR flag */
		pos_args[1].out[0] = ctx->ac.f32_0; /* X */
		pos_args[1].out[1] = ctx->ac.f32_0; /* Y */
		pos_args[1].out[2] = ctx->ac.f32_0; /* Z */
		pos_args[1].out[3] = ctx->ac.f32_0; /* W */

		if (shader->selector->info.writes_psize)
			pos_args[1].out[0] = psize_value;

		if (shader->selector->info.writes_edgeflag) {
			/* The output is a float, but the hw expects an integer
			 * with the first bit containing the edge flag. */
			edgeflag_value = LLVMBuildFPToUI(ctx->ac.builder,
							 edgeflag_value,
							 ctx->i32, "");
			edgeflag_value = ac_build_umin(&ctx->ac,
						      edgeflag_value,
						      ctx->i32_1);

			/* The LLVM intrinsic expects a float. */
			pos_args[1].out[1] = ac_to_float(&ctx->ac, edgeflag_value);
		}

		if (ctx->screen->info.chip_class >= GFX9) {
			/* GFX9 has the layer in out.z[10:0] and the viewport
			 * index in out.z[19:16].
			 */
			if (shader->selector->info.writes_layer)
				pos_args[1].out[2] = layer_value;

			if (shader->selector->info.writes_viewport_index) {
				LLVMValueRef v = viewport_index_value;

				v = ac_to_integer(&ctx->ac, v);
				v = LLVMBuildShl(ctx->ac.builder, v,
						 LLVMConstInt(ctx->i32, 16, 0), "");
				v = LLVMBuildOr(ctx->ac.builder, v,
						ac_to_integer(&ctx->ac,  pos_args[1].out[2]), "");
				pos_args[1].out[2] = ac_to_float(&ctx->ac, v);
				pos_args[1].enabled_channels |= 1 << 2;
			}
		} else {
			if (shader->selector->info.writes_layer)
				pos_args[1].out[2] = layer_value;

			if (shader->selector->info.writes_viewport_index) {
				pos_args[1].out[3] = viewport_index_value;
				pos_args[1].enabled_channels |= 1 << 3;
			}
		}
	}

	for (i = 0; i < 4; i++)
		if (pos_args[i].out[0])
			shader->info.nr_pos_exports++;

	pos_idx = 0;
	for (i = 0; i < 4; i++) {
		if (!pos_args[i].out[0])
			continue;

		/* Specify the target we are exporting */
		pos_args[i].target = V_008DFC_SQ_EXP_POS + pos_idx++;

		if (pos_idx == shader->info.nr_pos_exports)
			/* Specify that this is the last export */
			pos_args[i].done = 1;

		ac_build_export(&ctx->ac, &pos_args[i]);
	}

	/* Build parameter exports. */
	si_build_param_exports(ctx, outputs, noutput);
}

/**
 * Forward all outputs from the vertex shader to the TES. This is only used
 * for the fixed function TCS.
 */
static void si_copy_tcs_inputs(struct lp_build_tgsi_context *bld_base)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef invocation_id, buffer, buffer_offset;
	LLVMValueRef lds_vertex_stride, lds_base;
	uint64_t inputs;

	invocation_id = unpack_llvm_param(ctx, ctx->abi.tcs_rel_ids, 8, 5);
	buffer = get_tess_ring_descriptor(ctx, TESS_OFFCHIP_RING_TCS);
	buffer_offset = LLVMGetParam(ctx->main_fn, ctx->param_tcs_offchip_offset);

	lds_vertex_stride = get_tcs_in_vertex_dw_stride(ctx);
	lds_base = get_tcs_in_current_patch_offset(ctx);
	lds_base = ac_build_imad(&ctx->ac, invocation_id, lds_vertex_stride,
				 lds_base);

	inputs = ctx->shader->key.mono.u.ff_tcs_inputs_to_copy;
	while (inputs) {
		unsigned i = u_bit_scan64(&inputs);

		LLVMValueRef lds_ptr = LLVMBuildAdd(ctx->ac.builder, lds_base,
		                            LLVMConstInt(ctx->i32, 4 * i, 0),
		                             "");

		LLVMValueRef buffer_addr = get_tcs_tes_buffer_address(ctx,
					      get_rel_patch_id(ctx),
		                              invocation_id,
		                              LLVMConstInt(ctx->i32, i, 0));

		LLVMValueRef value = lds_load(bld_base, ctx->ac.i32, ~0,
		                              lds_ptr);

		ac_build_buffer_store_dword(&ctx->ac, buffer, value, 4, buffer_addr,
					    buffer_offset, 0, 1, 0, true, false);
	}
}

static void si_write_tess_factors(struct lp_build_tgsi_context *bld_base,
				  LLVMValueRef rel_patch_id,
				  LLVMValueRef invocation_id,
				  LLVMValueRef tcs_out_current_patch_data_offset,
				  LLVMValueRef invoc0_tf_outer[4],
				  LLVMValueRef invoc0_tf_inner[2])
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	struct si_shader *shader = ctx->shader;
	unsigned tess_inner_index, tess_outer_index;
	LLVMValueRef lds_base, lds_inner, lds_outer, byteoffset, buffer;
	LLVMValueRef out[6], vec0, vec1, tf_base, inner[4], outer[4];
	unsigned stride, outer_comps, inner_comps, i, offset;
	struct lp_build_if_state if_ctx, inner_if_ctx;

	/* Add a barrier before loading tess factors from LDS. */
	if (!shader->key.part.tcs.epilog.invoc0_tess_factors_are_def)
		si_llvm_emit_barrier(NULL, bld_base, NULL);

	/* Do this only for invocation 0, because the tess levels are per-patch,
	 * not per-vertex.
	 *
	 * This can't jump, because invocation 0 executes this. It should
	 * at least mask out the loads and stores for other invocations.
	 */
	lp_build_if(&if_ctx, &ctx->gallivm,
		    LLVMBuildICmp(ctx->ac.builder, LLVMIntEQ,
				  invocation_id, ctx->i32_0, ""));

	/* Determine the layout of one tess factor element in the buffer. */
	switch (shader->key.part.tcs.epilog.prim_mode) {
	case PIPE_PRIM_LINES:
		stride = 2; /* 2 dwords, 1 vec2 store */
		outer_comps = 2;
		inner_comps = 0;
		break;
	case PIPE_PRIM_TRIANGLES:
		stride = 4; /* 4 dwords, 1 vec4 store */
		outer_comps = 3;
		inner_comps = 1;
		break;
	case PIPE_PRIM_QUADS:
		stride = 6; /* 6 dwords, 2 stores (vec4 + vec2) */
		outer_comps = 4;
		inner_comps = 2;
		break;
	default:
		assert(0);
		return;
	}

	for (i = 0; i < 4; i++) {
		inner[i] = LLVMGetUndef(ctx->i32);
		outer[i] = LLVMGetUndef(ctx->i32);
	}

	if (shader->key.part.tcs.epilog.invoc0_tess_factors_are_def) {
		/* Tess factors are in VGPRs. */
		for (i = 0; i < outer_comps; i++)
			outer[i] = out[i] = invoc0_tf_outer[i];
		for (i = 0; i < inner_comps; i++)
			inner[i] = out[outer_comps+i] = invoc0_tf_inner[i];
	} else {
		/* Load tess_inner and tess_outer from LDS.
		 * Any invocation can write them, so we can't get them from a temporary.
		 */
		tess_inner_index = si_shader_io_get_unique_index_patch(TGSI_SEMANTIC_TESSINNER, 0);
		tess_outer_index = si_shader_io_get_unique_index_patch(TGSI_SEMANTIC_TESSOUTER, 0);

		lds_base = tcs_out_current_patch_data_offset;
		lds_inner = LLVMBuildAdd(ctx->ac.builder, lds_base,
					 LLVMConstInt(ctx->i32,
						      tess_inner_index * 4, 0), "");
		lds_outer = LLVMBuildAdd(ctx->ac.builder, lds_base,
					 LLVMConstInt(ctx->i32,
						      tess_outer_index * 4, 0), "");

		for (i = 0; i < outer_comps; i++) {
			outer[i] = out[i] =
				lds_load(bld_base, ctx->ac.i32, i, lds_outer);
		}
		for (i = 0; i < inner_comps; i++) {
			inner[i] = out[outer_comps+i] =
				lds_load(bld_base, ctx->ac.i32, i, lds_inner);
		}
	}

	if (shader->key.part.tcs.epilog.prim_mode == PIPE_PRIM_LINES) {
		/* For isolines, the hardware expects tess factors in the
		 * reverse order from what GLSL / TGSI specify.
		 */
		LLVMValueRef tmp = out[0];
		out[0] = out[1];
		out[1] = tmp;
	}

	/* Convert the outputs to vectors for stores. */
	vec0 = ac_build_gather_values(&ctx->ac, out, MIN2(stride, 4));
	vec1 = NULL;

	if (stride > 4)
		vec1 = ac_build_gather_values(&ctx->ac, out+4, stride - 4);

	/* Get the buffer. */
	buffer = get_tess_ring_descriptor(ctx, TCS_FACTOR_RING);

	/* Get the offset. */
	tf_base = LLVMGetParam(ctx->main_fn,
			       ctx->param_tcs_factor_offset);
	byteoffset = LLVMBuildMul(ctx->ac.builder, rel_patch_id,
				  LLVMConstInt(ctx->i32, 4 * stride, 0), "");

	lp_build_if(&inner_if_ctx, &ctx->gallivm,
		    LLVMBuildICmp(ctx->ac.builder, LLVMIntEQ,
				  rel_patch_id, ctx->i32_0, ""));

	/* Store the dynamic HS control word. */
	offset = 0;
	if (ctx->screen->info.chip_class <= VI) {
		ac_build_buffer_store_dword(&ctx->ac, buffer,
					    LLVMConstInt(ctx->i32, 0x80000000, 0),
					    1, ctx->i32_0, tf_base,
					    offset, 1, 0, true, false);
		offset += 4;
	}

	lp_build_endif(&inner_if_ctx);

	/* Store the tessellation factors. */
	ac_build_buffer_store_dword(&ctx->ac, buffer, vec0,
				    MIN2(stride, 4), byteoffset, tf_base,
				    offset, 1, 0, true, false);
	offset += 16;
	if (vec1)
		ac_build_buffer_store_dword(&ctx->ac, buffer, vec1,
					    stride - 4, byteoffset, tf_base,
					    offset, 1, 0, true, false);

	/* Store the tess factors into the offchip buffer if TES reads them. */
	if (shader->key.part.tcs.epilog.tes_reads_tess_factors) {
		LLVMValueRef buf, base, inner_vec, outer_vec, tf_outer_offset;
		LLVMValueRef tf_inner_offset;
		unsigned param_outer, param_inner;

		buf = get_tess_ring_descriptor(ctx, TESS_OFFCHIP_RING_TCS);
		base = LLVMGetParam(ctx->main_fn, ctx->param_tcs_offchip_offset);

		param_outer = si_shader_io_get_unique_index_patch(
				      TGSI_SEMANTIC_TESSOUTER, 0);
		tf_outer_offset = get_tcs_tes_buffer_address(ctx, rel_patch_id, NULL,
					LLVMConstInt(ctx->i32, param_outer, 0));

		outer_vec = ac_build_gather_values(&ctx->ac, outer,
						   util_next_power_of_two(outer_comps));

		ac_build_buffer_store_dword(&ctx->ac, buf, outer_vec,
					    outer_comps, tf_outer_offset,
					    base, 0, 1, 0, true, false);
		if (inner_comps) {
			param_inner = si_shader_io_get_unique_index_patch(
					      TGSI_SEMANTIC_TESSINNER, 0);
			tf_inner_offset = get_tcs_tes_buffer_address(ctx, rel_patch_id, NULL,
					LLVMConstInt(ctx->i32, param_inner, 0));

			inner_vec = inner_comps == 1 ? inner[0] :
				    ac_build_gather_values(&ctx->ac, inner, inner_comps);
			ac_build_buffer_store_dword(&ctx->ac, buf, inner_vec,
						    inner_comps, tf_inner_offset,
						    base, 0, 1, 0, true, false);
		}
	}

	lp_build_endif(&if_ctx);
}

static LLVMValueRef
si_insert_input_ret(struct si_shader_context *ctx, LLVMValueRef ret,
		    unsigned param, unsigned return_index)
{
	return LLVMBuildInsertValue(ctx->ac.builder, ret,
				    LLVMGetParam(ctx->main_fn, param),
				    return_index, "");
}

static LLVMValueRef
si_insert_input_ret_float(struct si_shader_context *ctx, LLVMValueRef ret,
			  unsigned param, unsigned return_index)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef p = LLVMGetParam(ctx->main_fn, param);

	return LLVMBuildInsertValue(builder, ret,
				    ac_to_float(&ctx->ac, p),
				    return_index, "");
}

static LLVMValueRef
si_insert_input_ptr(struct si_shader_context *ctx, LLVMValueRef ret,
		    unsigned param, unsigned return_index)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef ptr, lo, hi;

	if (HAVE_32BIT_POINTERS) {
		ptr = LLVMGetParam(ctx->main_fn, param);
		ptr = LLVMBuildPtrToInt(builder, ptr, ctx->i32, "");
		return LLVMBuildInsertValue(builder, ret, ptr, return_index, "");
	}

	ptr = LLVMGetParam(ctx->main_fn, param);
	ptr = LLVMBuildPtrToInt(builder, ptr, ctx->i64, "");
	ptr = LLVMBuildBitCast(builder, ptr, ctx->v2i32, "");
	lo = LLVMBuildExtractElement(builder, ptr, ctx->i32_0, "");
	hi = LLVMBuildExtractElement(builder, ptr, ctx->i32_1, "");
	ret = LLVMBuildInsertValue(builder, ret, lo, return_index, "");
	return LLVMBuildInsertValue(builder, ret, hi, return_index + 1, "");
}

/* This only writes the tessellation factor levels. */
static void si_llvm_emit_tcs_epilogue(struct ac_shader_abi *abi,
				      unsigned max_outputs,
				      LLVMValueRef *addrs)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct lp_build_tgsi_context *bld_base = &ctx->bld_base;
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef rel_patch_id, invocation_id, tf_lds_offset;

	si_copy_tcs_inputs(bld_base);

	rel_patch_id = get_rel_patch_id(ctx);
	invocation_id = unpack_llvm_param(ctx, ctx->abi.tcs_rel_ids, 8, 5);
	tf_lds_offset = get_tcs_out_current_patch_data_offset(ctx);

	if (ctx->screen->info.chip_class >= GFX9) {
		LLVMBasicBlockRef blocks[2] = {
			LLVMGetInsertBlock(builder),
			ctx->merged_wrap_if_state.entry_block
		};
		LLVMValueRef values[2];

		lp_build_endif(&ctx->merged_wrap_if_state);

		values[0] = rel_patch_id;
		values[1] = LLVMGetUndef(ctx->i32);
		rel_patch_id = ac_build_phi(&ctx->ac, ctx->i32, 2, values, blocks);

		values[0] = tf_lds_offset;
		values[1] = LLVMGetUndef(ctx->i32);
		tf_lds_offset = ac_build_phi(&ctx->ac, ctx->i32, 2, values, blocks);

		values[0] = invocation_id;
		values[1] = ctx->i32_1; /* cause the epilog to skip threads */
		invocation_id = ac_build_phi(&ctx->ac, ctx->i32, 2, values, blocks);
	}

	/* Return epilog parameters from this function. */
	LLVMValueRef ret = ctx->return_value;
	unsigned vgpr;

	if (ctx->screen->info.chip_class >= GFX9) {
		ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_offchip_layout,
					  8 + GFX9_SGPR_TCS_OFFCHIP_LAYOUT);
		ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_out_lds_layout,
					  8 + GFX9_SGPR_TCS_OUT_LAYOUT);
		/* Tess offchip and tess factor offsets are at the beginning. */
		ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_offchip_offset, 2);
		ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_factor_offset, 4);
		vgpr = 8 + GFX9_SGPR_TCS_OUT_LAYOUT + 1;
	} else {
		ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_offchip_layout,
					  GFX6_SGPR_TCS_OFFCHIP_LAYOUT);
		ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_out_lds_layout,
					  GFX6_SGPR_TCS_OUT_LAYOUT);
		/* Tess offchip and tess factor offsets are after user SGPRs. */
		ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_offchip_offset,
					  GFX6_TCS_NUM_USER_SGPR);
		ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_factor_offset,
					  GFX6_TCS_NUM_USER_SGPR + 1);
		vgpr = GFX6_TCS_NUM_USER_SGPR + 2;
	}

	/* VGPRs */
	rel_patch_id = ac_to_float(&ctx->ac, rel_patch_id);
	invocation_id = ac_to_float(&ctx->ac, invocation_id);
	tf_lds_offset = ac_to_float(&ctx->ac, tf_lds_offset);

	/* Leave a hole corresponding to the two input VGPRs. This ensures that
	 * the invocation_id output does not alias the tcs_rel_ids input,
	 * which saves a V_MOV on gfx9.
	 */
	vgpr += 2;

	ret = LLVMBuildInsertValue(builder, ret, rel_patch_id, vgpr++, "");
	ret = LLVMBuildInsertValue(builder, ret, invocation_id, vgpr++, "");

	if (ctx->shader->selector->tcs_info.tessfactors_are_def_in_all_invocs) {
		vgpr++; /* skip the tess factor LDS offset */
		for (unsigned i = 0; i < 6; i++) {
			LLVMValueRef value =
				LLVMBuildLoad(builder, ctx->invoc0_tess_factors[i], "");
			value = ac_to_float(&ctx->ac, value);
			ret = LLVMBuildInsertValue(builder, ret, value, vgpr++, "");
		}
	} else {
		ret = LLVMBuildInsertValue(builder, ret, tf_lds_offset, vgpr++, "");
	}
	ctx->return_value = ret;
}

/* Pass TCS inputs from LS to TCS on GFX9. */
static void si_set_ls_return_value_for_tcs(struct si_shader_context *ctx)
{
	LLVMValueRef ret = ctx->return_value;

	ret = si_insert_input_ptr(ctx, ret, 0, 0);
	if (HAVE_32BIT_POINTERS)
		ret = si_insert_input_ptr(ctx, ret, 1, 1);
	ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_offchip_offset, 2);
	ret = si_insert_input_ret(ctx, ret, ctx->param_merged_wave_info, 3);
	ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_factor_offset, 4);
	ret = si_insert_input_ret(ctx, ret, ctx->param_merged_scratch_offset, 5);

	ret = si_insert_input_ptr(ctx, ret, ctx->param_rw_buffers,
				  8 + SI_SGPR_RW_BUFFERS);
	ret = si_insert_input_ptr(ctx, ret,
				  ctx->param_bindless_samplers_and_images,
				  8 + SI_SGPR_BINDLESS_SAMPLERS_AND_IMAGES);

	ret = si_insert_input_ret(ctx, ret, ctx->param_vs_state_bits,
				  8 + SI_SGPR_VS_STATE_BITS);

#if !HAVE_32BIT_POINTERS
	ret = si_insert_input_ptr(ctx, ret, ctx->param_vs_state_bits + 4,
				  8 + GFX9_SGPR_2ND_SAMPLERS_AND_IMAGES);
#endif

	ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_offchip_layout,
				  8 + GFX9_SGPR_TCS_OFFCHIP_LAYOUT);
	ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_out_lds_offsets,
				  8 + GFX9_SGPR_TCS_OUT_OFFSETS);
	ret = si_insert_input_ret(ctx, ret, ctx->param_tcs_out_lds_layout,
				  8 + GFX9_SGPR_TCS_OUT_LAYOUT);

	unsigned vgpr = 8 + GFX9_TCS_NUM_USER_SGPR;
	ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
				   ac_to_float(&ctx->ac, ctx->abi.tcs_patch_id),
				   vgpr++, "");
	ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
				   ac_to_float(&ctx->ac, ctx->abi.tcs_rel_ids),
				   vgpr++, "");
	ctx->return_value = ret;
}

/* Pass GS inputs from ES to GS on GFX9. */
static void si_set_es_return_value_for_gs(struct si_shader_context *ctx)
{
	LLVMValueRef ret = ctx->return_value;

	ret = si_insert_input_ptr(ctx, ret, 0, 0);
	if (HAVE_32BIT_POINTERS)
		ret = si_insert_input_ptr(ctx, ret, 1, 1);
	ret = si_insert_input_ret(ctx, ret, ctx->param_gs2vs_offset, 2);
	ret = si_insert_input_ret(ctx, ret, ctx->param_merged_wave_info, 3);
	ret = si_insert_input_ret(ctx, ret, ctx->param_merged_scratch_offset, 5);

	ret = si_insert_input_ptr(ctx, ret, ctx->param_rw_buffers,
				  8 + SI_SGPR_RW_BUFFERS);
	ret = si_insert_input_ptr(ctx, ret,
				  ctx->param_bindless_samplers_and_images,
				  8 + SI_SGPR_BINDLESS_SAMPLERS_AND_IMAGES);

#if !HAVE_32BIT_POINTERS
	ret = si_insert_input_ptr(ctx, ret, ctx->param_vs_state_bits + 4,
				  8 + GFX9_SGPR_2ND_SAMPLERS_AND_IMAGES);
#endif

	unsigned vgpr;
	if (ctx->type == PIPE_SHADER_VERTEX)
		vgpr = 8 + GFX9_VSGS_NUM_USER_SGPR;
	else
		vgpr = 8 + GFX9_TESGS_NUM_USER_SGPR;

	for (unsigned i = 0; i < 5; i++) {
		unsigned param = ctx->param_gs_vtx01_offset + i;
		ret = si_insert_input_ret_float(ctx, ret, param, vgpr++);
	}
	ctx->return_value = ret;
}

static void si_llvm_emit_ls_epilogue(struct ac_shader_abi *abi,
				     unsigned max_outputs,
				     LLVMValueRef *addrs)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct si_shader *shader = ctx->shader;
	struct tgsi_shader_info *info = &shader->selector->info;
	unsigned i, chan;
	LLVMValueRef vertex_id = LLVMGetParam(ctx->main_fn,
					      ctx->param_rel_auto_id);
	LLVMValueRef vertex_dw_stride = get_tcs_in_vertex_dw_stride(ctx);
	LLVMValueRef base_dw_addr = LLVMBuildMul(ctx->ac.builder, vertex_id,
						 vertex_dw_stride, "");

	/* Write outputs to LDS. The next shader (TCS aka HS) will read
	 * its inputs from it. */
	for (i = 0; i < info->num_outputs; i++) {
		unsigned name = info->output_semantic_name[i];
		unsigned index = info->output_semantic_index[i];

		/* The ARB_shader_viewport_layer_array spec contains the
		 * following issue:
		 *
		 *    2) What happens if gl_ViewportIndex or gl_Layer is
		 *    written in the vertex shader and a geometry shader is
		 *    present?
		 *
		 *    RESOLVED: The value written by the last vertex processing
		 *    stage is used. If the last vertex processing stage
		 *    (vertex, tessellation evaluation or geometry) does not
		 *    statically assign to gl_ViewportIndex or gl_Layer, index
		 *    or layer zero is assumed.
		 *
		 * So writes to those outputs in VS-as-LS are simply ignored.
		 */
		if (name == TGSI_SEMANTIC_LAYER ||
		    name == TGSI_SEMANTIC_VIEWPORT_INDEX)
			continue;

		int param = si_shader_io_get_unique_index(name, index, false);
		LLVMValueRef dw_addr = LLVMBuildAdd(ctx->ac.builder, base_dw_addr,
					LLVMConstInt(ctx->i32, param * 4, 0), "");

		for (chan = 0; chan < 4; chan++) {
			if (!(info->output_usagemask[i] & (1 << chan)))
				continue;

			lds_store(ctx, chan, dw_addr,
				  LLVMBuildLoad(ctx->ac.builder, addrs[4 * i + chan], ""));
		}
	}

	if (ctx->screen->info.chip_class >= GFX9)
		si_set_ls_return_value_for_tcs(ctx);
}

static void si_llvm_emit_es_epilogue(struct ac_shader_abi *abi,
				     unsigned max_outputs,
				     LLVMValueRef *addrs)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct si_shader *es = ctx->shader;
	struct tgsi_shader_info *info = &es->selector->info;
	LLVMValueRef soffset = LLVMGetParam(ctx->main_fn,
					    ctx->param_es2gs_offset);
	LLVMValueRef lds_base = NULL;
	unsigned chan;
	int i;

	if (ctx->screen->info.chip_class >= GFX9 && info->num_outputs) {
		unsigned itemsize_dw = es->selector->esgs_itemsize / 4;
		LLVMValueRef vertex_idx = ac_get_thread_id(&ctx->ac);
		LLVMValueRef wave_idx = si_unpack_param(ctx, ctx->param_merged_wave_info, 24, 4);
		vertex_idx = LLVMBuildOr(ctx->ac.builder, vertex_idx,
					 LLVMBuildMul(ctx->ac.builder, wave_idx,
						      LLVMConstInt(ctx->i32, 64, false), ""), "");
		lds_base = LLVMBuildMul(ctx->ac.builder, vertex_idx,
					LLVMConstInt(ctx->i32, itemsize_dw, 0), "");
	}

	for (i = 0; i < info->num_outputs; i++) {
		int param;

		if (info->output_semantic_name[i] == TGSI_SEMANTIC_VIEWPORT_INDEX ||
		    info->output_semantic_name[i] == TGSI_SEMANTIC_LAYER)
			continue;

		param = si_shader_io_get_unique_index(info->output_semantic_name[i],
						      info->output_semantic_index[i], false);

		for (chan = 0; chan < 4; chan++) {
			if (!(info->output_usagemask[i] & (1 << chan)))
				continue;

			LLVMValueRef out_val = LLVMBuildLoad(ctx->ac.builder, addrs[4 * i + chan], "");
			out_val = ac_to_integer(&ctx->ac, out_val);

			/* GFX9 has the ESGS ring in LDS. */
			if (ctx->screen->info.chip_class >= GFX9) {
				lds_store(ctx, param * 4 + chan, lds_base, out_val);
				continue;
			}

			ac_build_buffer_store_dword(&ctx->ac,
						    ctx->esgs_ring,
						    out_val, 1, NULL, soffset,
						    (4 * param + chan) * 4,
						    1, 1, true, true);
		}
	}

	if (ctx->screen->info.chip_class >= GFX9)
		si_set_es_return_value_for_gs(ctx);
}

static LLVMValueRef si_get_gs_wave_id(struct si_shader_context *ctx)
{
	if (ctx->screen->info.chip_class >= GFX9)
		return si_unpack_param(ctx, ctx->param_merged_wave_info, 16, 8);
	else
		return LLVMGetParam(ctx->main_fn, ctx->param_gs_wave_id);
}

static void emit_gs_epilogue(struct si_shader_context *ctx)
{
	ac_build_sendmsg(&ctx->ac, AC_SENDMSG_GS_OP_NOP | AC_SENDMSG_GS_DONE,
			 si_get_gs_wave_id(ctx));

	if (ctx->screen->info.chip_class >= GFX9)
		lp_build_endif(&ctx->merged_wrap_if_state);
}

static void si_llvm_emit_gs_epilogue(struct ac_shader_abi *abi,
				     unsigned max_outputs,
				     LLVMValueRef *addrs)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct tgsi_shader_info UNUSED *info = &ctx->shader->selector->info;

	assert(info->num_outputs <= max_outputs);

	emit_gs_epilogue(ctx);
}

static void si_tgsi_emit_gs_epilogue(struct lp_build_tgsi_context *bld_base)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	emit_gs_epilogue(ctx);
}

static void si_llvm_emit_vs_epilogue(struct ac_shader_abi *abi,
				     unsigned max_outputs,
				     LLVMValueRef *addrs)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct tgsi_shader_info *info = &ctx->shader->selector->info;
	struct si_shader_output_values *outputs = NULL;
	int i,j;

	assert(!ctx->shader->is_gs_copy_shader);
	assert(info->num_outputs <= max_outputs);

	outputs = MALLOC((info->num_outputs + 1) * sizeof(outputs[0]));

	/* Vertex color clamping.
	 *
	 * This uses a state constant loaded in a user data SGPR and
	 * an IF statement is added that clamps all colors if the constant
	 * is true.
	 */
	struct lp_build_if_state if_ctx;
	LLVMValueRef cond = NULL;
	LLVMValueRef addr, val;

	for (i = 0; i < info->num_outputs; i++) {
		if (info->output_semantic_name[i] != TGSI_SEMANTIC_COLOR &&
		    info->output_semantic_name[i] != TGSI_SEMANTIC_BCOLOR)
			continue;

		/* We've found a color. */
		if (!cond) {
			/* The state is in the first bit of the user SGPR. */
			cond = LLVMGetParam(ctx->main_fn,
					    ctx->param_vs_state_bits);
			cond = LLVMBuildTrunc(ctx->ac.builder, cond,
					      ctx->i1, "");
			lp_build_if(&if_ctx, &ctx->gallivm, cond);
		}

		for (j = 0; j < 4; j++) {
			addr = addrs[4 * i + j];
			val = LLVMBuildLoad(ctx->ac.builder, addr, "");
			val = ac_build_clamp(&ctx->ac, val);
			LLVMBuildStore(ctx->ac.builder, val, addr);
		}
	}

	if (cond)
		lp_build_endif(&if_ctx);

	for (i = 0; i < info->num_outputs; i++) {
		outputs[i].semantic_name = info->output_semantic_name[i];
		outputs[i].semantic_index = info->output_semantic_index[i];

		for (j = 0; j < 4; j++) {
			outputs[i].values[j] =
				LLVMBuildLoad(ctx->ac.builder,
					      addrs[4 * i + j],
					      "");
			outputs[i].vertex_stream[j] =
				(info->output_streams[i] >> (2 * j)) & 3;
		}
	}

	if (ctx->shader->selector->so.num_outputs)
		si_llvm_emit_streamout(ctx, outputs, i, 0);

	/* Export PrimitiveID. */
	if (ctx->shader->key.mono.u.vs_export_prim_id) {
		outputs[i].semantic_name = TGSI_SEMANTIC_PRIMID;
		outputs[i].semantic_index = 0;
		outputs[i].values[0] = ac_to_float(&ctx->ac, get_primitive_id(ctx, 0));
		for (j = 1; j < 4; j++)
			outputs[i].values[j] = LLVMConstReal(ctx->f32, 0);

		memset(outputs[i].vertex_stream, 0,
		       sizeof(outputs[i].vertex_stream));
		i++;
	}

	si_llvm_export_vs(ctx, outputs, i);
	FREE(outputs);
}

static void si_tgsi_emit_epilogue(struct lp_build_tgsi_context *bld_base)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);

	ctx->abi.emit_outputs(&ctx->abi, RADEON_LLVM_MAX_OUTPUTS,
			      &ctx->outputs[0][0]);
}

struct si_ps_exports {
	unsigned num;
	struct ac_export_args args[10];
};

static void si_export_mrt_z(struct lp_build_tgsi_context *bld_base,
			    LLVMValueRef depth, LLVMValueRef stencil,
			    LLVMValueRef samplemask, struct si_ps_exports *exp)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	struct ac_export_args args;

	ac_export_mrt_z(&ctx->ac, depth, stencil, samplemask, &args);

	memcpy(&exp->args[exp->num++], &args, sizeof(args));
}

static void si_export_mrt_color(struct lp_build_tgsi_context *bld_base,
				LLVMValueRef *color, unsigned index,
				unsigned samplemask_param,
				bool is_last, struct si_ps_exports *exp)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	int i;

	/* Clamp color */
	if (ctx->shader->key.part.ps.epilog.clamp_color)
		for (i = 0; i < 4; i++)
			color[i] = ac_build_clamp(&ctx->ac, color[i]);

	/* Alpha to one */
	if (ctx->shader->key.part.ps.epilog.alpha_to_one)
		color[3] = ctx->ac.f32_1;

	/* Alpha test */
	if (index == 0 &&
	    ctx->shader->key.part.ps.epilog.alpha_func != PIPE_FUNC_ALWAYS)
		si_alpha_test(bld_base, color[3]);

	/* Line & polygon smoothing */
	if (ctx->shader->key.part.ps.epilog.poly_line_smoothing)
		color[3] = si_scale_alpha_by_sample_mask(bld_base, color[3],
							 samplemask_param);

	/* If last_cbuf > 0, FS_COLOR0_WRITES_ALL_CBUFS is true. */
	if (ctx->shader->key.part.ps.epilog.last_cbuf > 0) {
		struct ac_export_args args[8];
		int c, last = -1;

		/* Get the export arguments, also find out what the last one is. */
		for (c = 0; c <= ctx->shader->key.part.ps.epilog.last_cbuf; c++) {
			si_llvm_init_export_args(ctx, color,
						 V_008DFC_SQ_EXP_MRT + c, &args[c]);
			if (args[c].enabled_channels)
				last = c;
		}

		/* Emit all exports. */
		for (c = 0; c <= ctx->shader->key.part.ps.epilog.last_cbuf; c++) {
			if (is_last && last == c) {
				args[c].valid_mask = 1; /* whether the EXEC mask is valid */
				args[c].done = 1; /* DONE bit */
			} else if (!args[c].enabled_channels)
				continue; /* unnecessary NULL export */

			memcpy(&exp->args[exp->num++], &args[c], sizeof(args[c]));
		}
	} else {
		struct ac_export_args args;

		/* Export */
		si_llvm_init_export_args(ctx, color, V_008DFC_SQ_EXP_MRT + index,
					 &args);
		if (is_last) {
			args.valid_mask = 1; /* whether the EXEC mask is valid */
			args.done = 1; /* DONE bit */
		} else if (!args.enabled_channels)
			return; /* unnecessary NULL export */

		memcpy(&exp->args[exp->num++], &args, sizeof(args));
	}
}

static void si_emit_ps_exports(struct si_shader_context *ctx,
			       struct si_ps_exports *exp)
{
	for (unsigned i = 0; i < exp->num; i++)
		ac_build_export(&ctx->ac, &exp->args[i]);
}

/**
 * Return PS outputs in this order:
 *
 * v[0:3] = color0.xyzw
 * v[4:7] = color1.xyzw
 * ...
 * vN+0 = Depth
 * vN+1 = Stencil
 * vN+2 = SampleMask
 * vN+3 = SampleMaskIn (used for OpenGL smoothing)
 *
 * The alpha-ref SGPR is returned via its original location.
 */
static void si_llvm_return_fs_outputs(struct ac_shader_abi *abi,
				      unsigned max_outputs,
				      LLVMValueRef *addrs)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct si_shader *shader = ctx->shader;
	struct tgsi_shader_info *info = &shader->selector->info;
	LLVMBuilderRef builder = ctx->ac.builder;
	unsigned i, j, first_vgpr, vgpr;

	LLVMValueRef color[8][4] = {};
	LLVMValueRef depth = NULL, stencil = NULL, samplemask = NULL;
	LLVMValueRef ret;

	if (ctx->postponed_kill)
		ac_build_kill_if_false(&ctx->ac, LLVMBuildLoad(builder, ctx->postponed_kill, ""));

	/* Read the output values. */
	for (i = 0; i < info->num_outputs; i++) {
		unsigned semantic_name = info->output_semantic_name[i];
		unsigned semantic_index = info->output_semantic_index[i];

		switch (semantic_name) {
		case TGSI_SEMANTIC_COLOR:
			assert(semantic_index < 8);
			for (j = 0; j < 4; j++) {
				LLVMValueRef ptr = addrs[4 * i + j];
				LLVMValueRef result = LLVMBuildLoad(builder, ptr, "");
				color[semantic_index][j] = result;
			}
			break;
		case TGSI_SEMANTIC_POSITION:
			depth = LLVMBuildLoad(builder,
					      addrs[4 * i + 2], "");
			break;
		case TGSI_SEMANTIC_STENCIL:
			stencil = LLVMBuildLoad(builder,
						addrs[4 * i + 1], "");
			break;
		case TGSI_SEMANTIC_SAMPLEMASK:
			samplemask = LLVMBuildLoad(builder,
						   addrs[4 * i + 0], "");
			break;
		default:
			fprintf(stderr, "Warning: SI unhandled fs output type:%d\n",
				semantic_name);
		}
	}

	/* Fill the return structure. */
	ret = ctx->return_value;

	/* Set SGPRs. */
	ret = LLVMBuildInsertValue(builder, ret,
				   ac_to_integer(&ctx->ac,
                                                 LLVMGetParam(ctx->main_fn,
                                                              SI_PARAM_ALPHA_REF)),
				   SI_SGPR_ALPHA_REF, "");

	/* Set VGPRs */
	first_vgpr = vgpr = SI_SGPR_ALPHA_REF + 1;
	for (i = 0; i < ARRAY_SIZE(color); i++) {
		if (!color[i][0])
			continue;

		for (j = 0; j < 4; j++)
			ret = LLVMBuildInsertValue(builder, ret, color[i][j], vgpr++, "");
	}
	if (depth)
		ret = LLVMBuildInsertValue(builder, ret, depth, vgpr++, "");
	if (stencil)
		ret = LLVMBuildInsertValue(builder, ret, stencil, vgpr++, "");
	if (samplemask)
		ret = LLVMBuildInsertValue(builder, ret, samplemask, vgpr++, "");

	/* Add the input sample mask for smoothing at the end. */
	if (vgpr < first_vgpr + PS_EPILOG_SAMPLEMASK_MIN_LOC)
		vgpr = first_vgpr + PS_EPILOG_SAMPLEMASK_MIN_LOC;
	ret = LLVMBuildInsertValue(builder, ret,
				   LLVMGetParam(ctx->main_fn,
						SI_PARAM_SAMPLE_COVERAGE), vgpr++, "");

	ctx->return_value = ret;
}

static void membar_emit(
		const struct lp_build_tgsi_action *action,
		struct lp_build_tgsi_context *bld_base,
		struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef src0 = lp_build_emit_fetch(bld_base, emit_data->inst, 0, 0);
	unsigned flags = LLVMConstIntGetZExtValue(src0);
	unsigned waitcnt = NOOP_WAITCNT;

	if (flags & TGSI_MEMBAR_THREAD_GROUP)
		waitcnt &= VM_CNT & LGKM_CNT;

	if (flags & (TGSI_MEMBAR_ATOMIC_BUFFER |
		     TGSI_MEMBAR_SHADER_BUFFER |
		     TGSI_MEMBAR_SHADER_IMAGE))
		waitcnt &= VM_CNT;

	if (flags & TGSI_MEMBAR_SHARED)
		waitcnt &= LGKM_CNT;

	if (waitcnt != NOOP_WAITCNT)
		ac_build_waitcnt(&ctx->ac, waitcnt);
}

static void clock_emit(
		const struct lp_build_tgsi_action *action,
		struct lp_build_tgsi_context *bld_base,
		struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef tmp = ac_build_shader_clock(&ctx->ac);

	emit_data->output[0] =
		LLVMBuildExtractElement(ctx->ac.builder, tmp, ctx->i32_0, "");
	emit_data->output[1] =
		LLVMBuildExtractElement(ctx->ac.builder, tmp, ctx->i32_1, "");
}

static void si_llvm_emit_ddxy(
	const struct lp_build_tgsi_action *action,
	struct lp_build_tgsi_context *bld_base,
	struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	unsigned opcode = emit_data->info->opcode;
	LLVMValueRef val;
	int idx;
	unsigned mask;

	if (opcode == TGSI_OPCODE_DDX_FINE)
		mask = AC_TID_MASK_LEFT;
	else if (opcode == TGSI_OPCODE_DDY_FINE)
		mask = AC_TID_MASK_TOP;
	else
		mask = AC_TID_MASK_TOP_LEFT;

	/* for DDX we want to next X pixel, DDY next Y pixel. */
	idx = (opcode == TGSI_OPCODE_DDX || opcode == TGSI_OPCODE_DDX_FINE) ? 1 : 2;

	val = ac_to_integer(&ctx->ac, emit_data->args[0]);
	val = ac_build_ddxy(&ctx->ac, mask, idx, val);
	emit_data->output[emit_data->chan] = val;
}

/*
 * this takes an I,J coordinate pair,
 * and works out the X and Y derivatives.
 * it returns DDX(I), DDX(J), DDY(I), DDY(J).
 */
static LLVMValueRef si_llvm_emit_ddxy_interp(
	struct lp_build_tgsi_context *bld_base,
	LLVMValueRef interp_ij)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef result[4], a;
	unsigned i;

	for (i = 0; i < 2; i++) {
		a = LLVMBuildExtractElement(ctx->ac.builder, interp_ij,
					    LLVMConstInt(ctx->i32, i, 0), "");
		result[i] = ac_build_ddxy(&ctx->ac, AC_TID_MASK_TOP_LEFT, 1,
					  ac_to_integer(&ctx->ac, a)); /* DDX */
		result[2+i] = ac_build_ddxy(&ctx->ac, AC_TID_MASK_TOP_LEFT, 2,
					    ac_to_integer(&ctx->ac, a)); /* DDY */
	}

	return ac_build_gather_values(&ctx->ac, result, 4);
}

static void build_interp_intrinsic(const struct lp_build_tgsi_action *action,
				struct lp_build_tgsi_context *bld_base,
				struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	struct si_shader *shader = ctx->shader;
	const struct tgsi_shader_info *info = &shader->selector->info;
	LLVMValueRef interp_param;
	const struct tgsi_full_instruction *inst = emit_data->inst;
	const struct tgsi_full_src_register *input = &inst->Src[0];
	int input_base, input_array_size;
	int chan;
	int i;
	LLVMValueRef prim_mask = ctx->abi.prim_mask;
	LLVMValueRef array_idx, offset_x = NULL, offset_y = NULL;
	int interp_param_idx;
	unsigned interp;
	unsigned location;

	if (inst->Instruction.Opcode == TGSI_OPCODE_INTERP_OFFSET) {
		/* offset is in second src, first two channels */
		offset_x = lp_build_emit_fetch(bld_base, emit_data->inst, 1,
					       TGSI_CHAN_X);
		offset_y = lp_build_emit_fetch(bld_base, emit_data->inst, 1,
					       TGSI_CHAN_Y);
	} else if (inst->Instruction.Opcode == TGSI_OPCODE_INTERP_SAMPLE) {
		LLVMValueRef sample_position;
		LLVMValueRef sample_id;
		LLVMValueRef halfval = LLVMConstReal(ctx->f32, 0.5f);

		/* fetch sample ID, then fetch its sample position,
		 * and place into first two channels.
		 */
		sample_id = lp_build_emit_fetch(bld_base,
						emit_data->inst, 1, TGSI_CHAN_X);
		sample_id = ac_to_integer(&ctx->ac, sample_id);

		/* Section 8.13.2 (Interpolation Functions) of the OpenGL Shading
		 * Language 4.50 spec says about interpolateAtSample:
		 *
		 *    "Returns the value of the input interpolant variable at
		 *     the location of sample number sample. If multisample
		 *     buffers are not available, the input variable will be
		 *     evaluated at the center of the pixel. If sample sample
		 *     does not exist, the position used to interpolate the
		 *     input variable is undefined."
		 *
		 * This means that sample_id values outside of the valid are
		 * in fact valid input, and the usual mechanism for loading the
		 * sample position doesn't work.
		 */
		if (ctx->shader->key.mono.u.ps.interpolate_at_sample_force_center) {
			LLVMValueRef center[4] = {
				LLVMConstReal(ctx->f32, 0.5),
				LLVMConstReal(ctx->f32, 0.5),
				ctx->ac.f32_0,
				ctx->ac.f32_0,
			};

			sample_position = ac_build_gather_values(&ctx->ac, center, 4);
		} else {
			sample_position = load_sample_position(&ctx->abi, sample_id);
		}

		offset_x = LLVMBuildExtractElement(ctx->ac.builder, sample_position,
						   ctx->i32_0, "");

		offset_x = LLVMBuildFSub(ctx->ac.builder, offset_x, halfval, "");
		offset_y = LLVMBuildExtractElement(ctx->ac.builder, sample_position,
						   ctx->i32_1, "");
		offset_y = LLVMBuildFSub(ctx->ac.builder, offset_y, halfval, "");
	}

	assert(input->Register.File == TGSI_FILE_INPUT);

	if (input->Register.Indirect) {
		unsigned array_id = input->Indirect.ArrayID;

		if (array_id) {
			input_base = info->input_array_first[array_id];
			input_array_size = info->input_array_last[array_id] - input_base + 1;
		} else {
			input_base = inst->Src[0].Register.Index;
			input_array_size = info->num_inputs - input_base;
		}

		array_idx = si_get_indirect_index(ctx, &input->Indirect,
						  1, input->Register.Index - input_base);
	} else {
		input_base = inst->Src[0].Register.Index;
		input_array_size = 1;
		array_idx = ctx->i32_0;
	}

	interp = shader->selector->info.input_interpolate[input_base];

	if (inst->Instruction.Opcode == TGSI_OPCODE_INTERP_OFFSET ||
	    inst->Instruction.Opcode == TGSI_OPCODE_INTERP_SAMPLE)
		location = TGSI_INTERPOLATE_LOC_CENTER;
	else
		location = TGSI_INTERPOLATE_LOC_CENTROID;

	interp_param_idx = lookup_interp_param_index(interp, location);
	if (interp_param_idx == -1)
		return;
	else if (interp_param_idx)
		interp_param = LLVMGetParam(ctx->main_fn, interp_param_idx);
	else
		interp_param = NULL;

	if (inst->Instruction.Opcode == TGSI_OPCODE_INTERP_OFFSET ||
	    inst->Instruction.Opcode == TGSI_OPCODE_INTERP_SAMPLE) {
		LLVMValueRef ij_out[2];
		LLVMValueRef ddxy_out = si_llvm_emit_ddxy_interp(bld_base, interp_param);

		/*
		 * take the I then J parameters, and the DDX/Y for it, and
		 * calculate the IJ inputs for the interpolator.
		 * temp1 = ddx * offset/sample.x + I;
		 * interp_param.I = ddy * offset/sample.y + temp1;
		 * temp1 = ddx * offset/sample.x + J;
		 * interp_param.J = ddy * offset/sample.y + temp1;
		 */
		for (i = 0; i < 2; i++) {
			LLVMValueRef ix_ll = LLVMConstInt(ctx->i32, i, 0);
			LLVMValueRef iy_ll = LLVMConstInt(ctx->i32, i + 2, 0);
			LLVMValueRef ddx_el = LLVMBuildExtractElement(ctx->ac.builder,
								      ddxy_out, ix_ll, "");
			LLVMValueRef ddy_el = LLVMBuildExtractElement(ctx->ac.builder,
								      ddxy_out, iy_ll, "");
			LLVMValueRef interp_el = LLVMBuildExtractElement(ctx->ac.builder,
									 interp_param, ix_ll, "");
			LLVMValueRef temp;

			interp_el = ac_to_float(&ctx->ac, interp_el);

			temp = ac_build_fmad(&ctx->ac, ddx_el, offset_x, interp_el);
			ij_out[i] = ac_build_fmad(&ctx->ac, ddy_el, offset_y, temp);
		}
		interp_param = ac_build_gather_values(&ctx->ac, ij_out, 2);
	}

	if (interp_param)
		interp_param = ac_to_float(&ctx->ac, interp_param);

	for (chan = 0; chan < 4; chan++) {
		LLVMValueRef gather = LLVMGetUndef(LLVMVectorType(ctx->f32, input_array_size));
		unsigned schan = tgsi_util_get_full_src_register_swizzle(&inst->Src[0], chan);

		for (unsigned idx = 0; idx < input_array_size; ++idx) {
			LLVMValueRef v, i = NULL, j = NULL;

			if (interp_param) {
				i = LLVMBuildExtractElement(
					ctx->ac.builder, interp_param, ctx->i32_0, "");
				j = LLVMBuildExtractElement(
					ctx->ac.builder, interp_param, ctx->i32_1, "");
			}
			v = si_build_fs_interp(ctx, input_base + idx, schan,
					       prim_mask, i, j);

			gather = LLVMBuildInsertElement(ctx->ac.builder,
				gather, v, LLVMConstInt(ctx->i32, idx, false), "");
		}

		emit_data->output[chan] = LLVMBuildExtractElement(
			ctx->ac.builder, gather, array_idx, "");
	}
}

static void vote_all_emit(
	const struct lp_build_tgsi_action *action,
	struct lp_build_tgsi_context *bld_base,
	struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);

        LLVMValueRef tmp = ac_build_vote_all(&ctx->ac, emit_data->args[0]);
	emit_data->output[emit_data->chan] =
		LLVMBuildSExt(ctx->ac.builder, tmp, ctx->i32, "");
}

static void vote_any_emit(
	const struct lp_build_tgsi_action *action,
	struct lp_build_tgsi_context *bld_base,
	struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);

        LLVMValueRef tmp = ac_build_vote_any(&ctx->ac, emit_data->args[0]);
	emit_data->output[emit_data->chan] =
		LLVMBuildSExt(ctx->ac.builder, tmp, ctx->i32, "");
}

static void vote_eq_emit(
	const struct lp_build_tgsi_action *action,
	struct lp_build_tgsi_context *bld_base,
	struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);

        LLVMValueRef tmp = ac_build_vote_eq(&ctx->ac, emit_data->args[0]);
	emit_data->output[emit_data->chan] =
		LLVMBuildSExt(ctx->ac.builder, tmp, ctx->i32, "");
}

static void ballot_emit(
	const struct lp_build_tgsi_action *action,
	struct lp_build_tgsi_context *bld_base,
	struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef tmp;

	tmp = lp_build_emit_fetch(bld_base, emit_data->inst, 0, TGSI_CHAN_X);
	tmp = ac_build_ballot(&ctx->ac, tmp);
	tmp = LLVMBuildBitCast(builder, tmp, ctx->v2i32, "");

	emit_data->output[0] = LLVMBuildExtractElement(builder, tmp, ctx->i32_0, "");
	emit_data->output[1] = LLVMBuildExtractElement(builder, tmp, ctx->i32_1, "");
}

static void read_lane_emit(
	const struct lp_build_tgsi_action *action,
	struct lp_build_tgsi_context *bld_base,
	struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);

	if (emit_data->inst->Instruction.Opcode == TGSI_OPCODE_READ_INVOC) {
		emit_data->args[0] = lp_build_emit_fetch(bld_base, emit_data->inst,
							 0, emit_data->src_chan);

		/* Always read the source invocation (= lane) from the X channel. */
		emit_data->args[1] = lp_build_emit_fetch(bld_base, emit_data->inst,
							 1, TGSI_CHAN_X);
		emit_data->arg_count = 2;
	}

	/* We currently have no other way to prevent LLVM from lifting the icmp
	 * calls to a dominating basic block.
	 */
	ac_build_optimization_barrier(&ctx->ac, &emit_data->args[0]);

	for (unsigned i = 0; i < emit_data->arg_count; ++i)
		emit_data->args[i] = ac_to_integer(&ctx->ac, emit_data->args[i]);

	emit_data->output[emit_data->chan] =
		ac_build_intrinsic(&ctx->ac, action->intr_name,
				   ctx->i32, emit_data->args, emit_data->arg_count,
				   AC_FUNC_ATTR_READNONE |
				   AC_FUNC_ATTR_CONVERGENT);
}

static unsigned si_llvm_get_stream(struct lp_build_tgsi_context *bld_base,
				       struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	struct tgsi_src_register src0 = emit_data->inst->Src[0].Register;
	LLVMValueRef imm;
	unsigned stream;

	assert(src0.File == TGSI_FILE_IMMEDIATE);

	imm = ctx->imms[src0.Index * TGSI_NUM_CHANNELS + src0.SwizzleX];
	stream = LLVMConstIntGetZExtValue(imm) & 0x3;
	return stream;
}

/* Emit one vertex from the geometry shader */
static void si_llvm_emit_vertex(struct ac_shader_abi *abi,
				unsigned stream,
				LLVMValueRef *addrs)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	struct tgsi_shader_info *info = &ctx->shader->selector->info;
	struct si_shader *shader = ctx->shader;
	struct lp_build_if_state if_state;
	LLVMValueRef soffset = LLVMGetParam(ctx->main_fn,
					    ctx->param_gs2vs_offset);
	LLVMValueRef gs_next_vertex;
	LLVMValueRef can_emit;
	unsigned chan, offset;
	int i;

	/* Write vertex attribute values to GSVS ring */
	gs_next_vertex = LLVMBuildLoad(ctx->ac.builder,
				       ctx->gs_next_vertex[stream],
				       "");

	/* If this thread has already emitted the declared maximum number of
	 * vertices, skip the write: excessive vertex emissions are not
	 * supposed to have any effect.
	 *
	 * If the shader has no writes to memory, kill it instead. This skips
	 * further memory loads and may allow LLVM to skip to the end
	 * altogether.
	 */
	can_emit = LLVMBuildICmp(ctx->ac.builder, LLVMIntULT, gs_next_vertex,
				 LLVMConstInt(ctx->i32,
					      shader->selector->gs_max_out_vertices, 0), "");

	bool use_kill = !info->writes_memory;
	if (use_kill) {
		ac_build_kill_if_false(&ctx->ac, can_emit);
	} else {
		lp_build_if(&if_state, &ctx->gallivm, can_emit);
	}

	offset = 0;
	for (i = 0; i < info->num_outputs; i++) {
		for (chan = 0; chan < 4; chan++) {
			if (!(info->output_usagemask[i] & (1 << chan)) ||
			    ((info->output_streams[i] >> (2 * chan)) & 3) != stream)
				continue;

			LLVMValueRef out_val = LLVMBuildLoad(ctx->ac.builder, addrs[4 * i + chan], "");
			LLVMValueRef voffset =
				LLVMConstInt(ctx->i32, offset *
					     shader->selector->gs_max_out_vertices, 0);
			offset++;

			voffset = LLVMBuildAdd(ctx->ac.builder, voffset, gs_next_vertex, "");
			voffset = LLVMBuildMul(ctx->ac.builder, voffset,
					       LLVMConstInt(ctx->i32, 4, 0), "");

			out_val = ac_to_integer(&ctx->ac, out_val);

			ac_build_buffer_store_dword(&ctx->ac,
						    ctx->gsvs_ring[stream],
						    out_val, 1,
						    voffset, soffset, 0,
						    1, 1, true, true);
		}
	}

	gs_next_vertex = LLVMBuildAdd(ctx->ac.builder, gs_next_vertex, ctx->i32_1, "");
	LLVMBuildStore(ctx->ac.builder, gs_next_vertex, ctx->gs_next_vertex[stream]);

	/* Signal vertex emission if vertex data was written. */
	if (offset) {
		ac_build_sendmsg(&ctx->ac, AC_SENDMSG_GS_OP_EMIT | AC_SENDMSG_GS | (stream << 8),
				 si_get_gs_wave_id(ctx));
	}

	if (!use_kill)
		lp_build_endif(&if_state);
}

/* Emit one vertex from the geometry shader */
static void si_tgsi_emit_vertex(
	const struct lp_build_tgsi_action *action,
	struct lp_build_tgsi_context *bld_base,
	struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	unsigned stream = si_llvm_get_stream(bld_base, emit_data);

	si_llvm_emit_vertex(&ctx->abi, stream, ctx->outputs[0]);
}

/* Cut one primitive from the geometry shader */
static void si_llvm_emit_primitive(struct ac_shader_abi *abi,
				   unsigned stream)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);

	/* Signal primitive cut */
	ac_build_sendmsg(&ctx->ac, AC_SENDMSG_GS_OP_CUT | AC_SENDMSG_GS | (stream << 8),
			 si_get_gs_wave_id(ctx));
}

/* Cut one primitive from the geometry shader */
static void si_tgsi_emit_primitive(
	const struct lp_build_tgsi_action *action,
	struct lp_build_tgsi_context *bld_base,
	struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);

	si_llvm_emit_primitive(&ctx->abi, si_llvm_get_stream(bld_base, emit_data));
}

static void si_llvm_emit_barrier(const struct lp_build_tgsi_action *action,
				 struct lp_build_tgsi_context *bld_base,
				 struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);

	/* SI only (thanks to a hw bug workaround):
	 * The real barrier instruction isn’t needed, because an entire patch
	 * always fits into a single wave.
	 */
	if (ctx->screen->info.chip_class == SI &&
	    ctx->type == PIPE_SHADER_TESS_CTRL) {
		ac_build_waitcnt(&ctx->ac, LGKM_CNT & VM_CNT);
		return;
	}

	ac_build_s_barrier(&ctx->ac);
}

static void si_create_function(struct si_shader_context *ctx,
			       const char *name,
			       LLVMTypeRef *returns, unsigned num_returns,
			       struct si_function_info *fninfo,
			       unsigned max_workgroup_size)
{
	int i;

	si_llvm_create_func(ctx, name, returns, num_returns,
			    fninfo->types, fninfo->num_params);
	ctx->return_value = LLVMGetUndef(ctx->return_type);

	for (i = 0; i < fninfo->num_sgpr_params; ++i) {
		LLVMValueRef P = LLVMGetParam(ctx->main_fn, i);

		/* The combination of:
		 * - noalias
		 * - dereferenceable
		 * - invariant.load
		 * allows the optimization passes to move loads and reduces
		 * SGPR spilling significantly.
		 */
		ac_add_function_attr(ctx->ac.context, ctx->main_fn, i + 1,
				     AC_FUNC_ATTR_INREG);

		if (LLVMGetTypeKind(LLVMTypeOf(P)) == LLVMPointerTypeKind) {
			ac_add_function_attr(ctx->ac.context, ctx->main_fn, i + 1,
					     AC_FUNC_ATTR_NOALIAS);
			ac_add_attr_dereferenceable(P, UINT64_MAX);
		}
	}

	for (i = 0; i < fninfo->num_params; ++i) {
		if (fninfo->assign[i])
			*fninfo->assign[i] = LLVMGetParam(ctx->main_fn, i);
	}

	if (ctx->screen->info.address32_hi) {
		ac_llvm_add_target_dep_function_attr(ctx->main_fn,
						     "amdgpu-32bit-address-high-bits",
						     ctx->screen->info.address32_hi);
	}

	if (max_workgroup_size) {
		ac_llvm_add_target_dep_function_attr(ctx->main_fn,
						     "amdgpu-max-work-group-size",
						     max_workgroup_size);
	}
	LLVMAddTargetDependentFunctionAttr(ctx->main_fn,
					   "no-signed-zeros-fp-math",
					   "true");

	if (ctx->screen->debug_flags & DBG(UNSAFE_MATH)) {
		/* These were copied from some LLVM test. */
		LLVMAddTargetDependentFunctionAttr(ctx->main_fn,
						   "less-precise-fpmad",
						   "true");
		LLVMAddTargetDependentFunctionAttr(ctx->main_fn,
						   "no-infs-fp-math",
						   "true");
		LLVMAddTargetDependentFunctionAttr(ctx->main_fn,
						   "no-nans-fp-math",
						   "true");
		LLVMAddTargetDependentFunctionAttr(ctx->main_fn,
						   "unsafe-fp-math",
						   "true");
	}
}

static void declare_streamout_params(struct si_shader_context *ctx,
				     struct pipe_stream_output_info *so,
				     struct si_function_info *fninfo)
{
	int i;

	/* Streamout SGPRs. */
	if (so->num_outputs) {
		if (ctx->type != PIPE_SHADER_TESS_EVAL)
			ctx->param_streamout_config = add_arg(fninfo, ARG_SGPR, ctx->ac.i32);
		else
			ctx->param_streamout_config = fninfo->num_params - 1;

		ctx->param_streamout_write_index = add_arg(fninfo, ARG_SGPR, ctx->ac.i32);
	}
	/* A streamout buffer offset is loaded if the stride is non-zero. */
	for (i = 0; i < 4; i++) {
		if (!so->stride[i])
			continue;

		ctx->param_streamout_offset[i] = add_arg(fninfo, ARG_SGPR, ctx->ac.i32);
	}
}

static unsigned si_get_max_workgroup_size(const struct si_shader *shader)
{
	switch (shader->selector->type) {
	case PIPE_SHADER_TESS_CTRL:
		/* Return this so that LLVM doesn't remove s_barrier
		 * instructions on chips where we use s_barrier. */
		return shader->selector->screen->info.chip_class >= CIK ? 128 : 64;

	case PIPE_SHADER_GEOMETRY:
		return shader->selector->screen->info.chip_class >= GFX9 ? 128 : 64;

	case PIPE_SHADER_COMPUTE:
		break; /* see below */

	default:
		return 0;
	}

	const unsigned *properties = shader->selector->info.properties;
	unsigned max_work_group_size =
	               properties[TGSI_PROPERTY_CS_FIXED_BLOCK_WIDTH] *
	               properties[TGSI_PROPERTY_CS_FIXED_BLOCK_HEIGHT] *
	               properties[TGSI_PROPERTY_CS_FIXED_BLOCK_DEPTH];

	if (!max_work_group_size) {
		/* This is a variable group size compute shader,
		 * compile it for the maximum possible group size.
		 */
		max_work_group_size = SI_MAX_VARIABLE_THREADS_PER_BLOCK;
	}
	return max_work_group_size;
}

static void declare_const_and_shader_buffers(struct si_shader_context *ctx,
					     struct si_function_info *fninfo,
					     bool assign_params)
{
	LLVMTypeRef const_shader_buf_type;

	if (ctx->shader->selector->info.const_buffers_declared == 1 &&
	    ctx->shader->selector->info.shader_buffers_declared == 0)
		const_shader_buf_type = ctx->f32;
	else
		const_shader_buf_type = ctx->v4i32;

	unsigned const_and_shader_buffers =
		add_arg(fninfo, ARG_SGPR,
			ac_array_in_const32_addr_space(const_shader_buf_type));

	if (assign_params)
		ctx->param_const_and_shader_buffers = const_and_shader_buffers;
}

static void declare_samplers_and_images(struct si_shader_context *ctx,
					struct si_function_info *fninfo,
					bool assign_params)
{
	unsigned samplers_and_images =
		add_arg(fninfo, ARG_SGPR,
			ac_array_in_const32_addr_space(ctx->v8i32));

	if (assign_params)
		ctx->param_samplers_and_images = samplers_and_images;
}

static void declare_per_stage_desc_pointers(struct si_shader_context *ctx,
					    struct si_function_info *fninfo,
					    bool assign_params)
{
	declare_const_and_shader_buffers(ctx, fninfo, assign_params);
	declare_samplers_and_images(ctx, fninfo, assign_params);
}

static void declare_global_desc_pointers(struct si_shader_context *ctx,
					 struct si_function_info *fninfo)
{
	ctx->param_rw_buffers = add_arg(fninfo, ARG_SGPR,
		ac_array_in_const32_addr_space(ctx->v4i32));
	ctx->param_bindless_samplers_and_images = add_arg(fninfo, ARG_SGPR,
		ac_array_in_const32_addr_space(ctx->v8i32));
}

static void declare_vs_specific_input_sgprs(struct si_shader_context *ctx,
					    struct si_function_info *fninfo)
{
	ctx->param_vs_state_bits = add_arg(fninfo, ARG_SGPR, ctx->i32);
	add_arg_assign(fninfo, ARG_SGPR, ctx->i32, &ctx->abi.base_vertex);
	add_arg_assign(fninfo, ARG_SGPR, ctx->i32, &ctx->abi.start_instance);
	add_arg_assign(fninfo, ARG_SGPR, ctx->i32, &ctx->abi.draw_id);
}

static void declare_vs_input_vgprs(struct si_shader_context *ctx,
				   struct si_function_info *fninfo,
				   unsigned *num_prolog_vgprs)
{
	struct si_shader *shader = ctx->shader;

	add_arg_assign(fninfo, ARG_VGPR, ctx->i32, &ctx->abi.vertex_id);
	if (shader->key.as_ls) {
		ctx->param_rel_auto_id = add_arg(fninfo, ARG_VGPR, ctx->i32);
		add_arg_assign(fninfo, ARG_VGPR, ctx->i32, &ctx->abi.instance_id);
	} else {
		add_arg_assign(fninfo, ARG_VGPR, ctx->i32, &ctx->abi.instance_id);
		ctx->param_vs_prim_id = add_arg(fninfo, ARG_VGPR, ctx->i32);
	}
	add_arg(fninfo, ARG_VGPR, ctx->i32); /* unused */

	if (!shader->is_gs_copy_shader) {
		/* Vertex load indices. */
		ctx->param_vertex_index0 = fninfo->num_params;
		for (unsigned i = 0; i < shader->selector->info.num_inputs; i++)
			add_arg(fninfo, ARG_VGPR, ctx->i32);
		*num_prolog_vgprs += shader->selector->info.num_inputs;
	}
}

static void declare_tes_input_vgprs(struct si_shader_context *ctx,
				    struct si_function_info *fninfo)
{
	ctx->param_tes_u = add_arg(fninfo, ARG_VGPR, ctx->f32);
	ctx->param_tes_v = add_arg(fninfo, ARG_VGPR, ctx->f32);
	ctx->param_tes_rel_patch_id = add_arg(fninfo, ARG_VGPR, ctx->i32);
	add_arg_assign(fninfo, ARG_VGPR, ctx->i32, &ctx->abi.tes_patch_id);
}

enum {
	/* Convenient merged shader definitions. */
	SI_SHADER_MERGED_VERTEX_TESSCTRL = PIPE_SHADER_TYPES,
	SI_SHADER_MERGED_VERTEX_OR_TESSEVAL_GEOMETRY,
};

static void create_function(struct si_shader_context *ctx)
{
	struct si_shader *shader = ctx->shader;
	struct si_function_info fninfo;
	LLVMTypeRef returns[16+32*4];
	unsigned i, num_return_sgprs;
	unsigned num_returns = 0;
	unsigned num_prolog_vgprs = 0;
	unsigned type = ctx->type;
	unsigned vs_blit_property =
		shader->selector->info.properties[TGSI_PROPERTY_VS_BLIT_SGPRS];

	si_init_function_info(&fninfo);

	/* Set MERGED shaders. */
	if (ctx->screen->info.chip_class >= GFX9) {
		if (shader->key.as_ls || type == PIPE_SHADER_TESS_CTRL)
			type = SI_SHADER_MERGED_VERTEX_TESSCTRL; /* LS or HS */
		else if (shader->key.as_es || type == PIPE_SHADER_GEOMETRY)
			type = SI_SHADER_MERGED_VERTEX_OR_TESSEVAL_GEOMETRY;
	}

	LLVMTypeRef v3i32 = LLVMVectorType(ctx->i32, 3);

	switch (type) {
	case PIPE_SHADER_VERTEX:
		declare_global_desc_pointers(ctx, &fninfo);

		if (vs_blit_property) {
			ctx->param_vs_blit_inputs = fninfo.num_params;
			add_arg(&fninfo, ARG_SGPR, ctx->i32); /* i16 x1, y1 */
			add_arg(&fninfo, ARG_SGPR, ctx->i32); /* i16 x2, y2 */
			add_arg(&fninfo, ARG_SGPR, ctx->f32); /* depth */

			if (vs_blit_property == SI_VS_BLIT_SGPRS_POS_COLOR) {
				add_arg(&fninfo, ARG_SGPR, ctx->f32); /* color0 */
				add_arg(&fninfo, ARG_SGPR, ctx->f32); /* color1 */
				add_arg(&fninfo, ARG_SGPR, ctx->f32); /* color2 */
				add_arg(&fninfo, ARG_SGPR, ctx->f32); /* color3 */
			} else if (vs_blit_property == SI_VS_BLIT_SGPRS_POS_TEXCOORD) {
				add_arg(&fninfo, ARG_SGPR, ctx->f32); /* texcoord.x1 */
				add_arg(&fninfo, ARG_SGPR, ctx->f32); /* texcoord.y1 */
				add_arg(&fninfo, ARG_SGPR, ctx->f32); /* texcoord.x2 */
				add_arg(&fninfo, ARG_SGPR, ctx->f32); /* texcoord.y2 */
				add_arg(&fninfo, ARG_SGPR, ctx->f32); /* texcoord.z */
				add_arg(&fninfo, ARG_SGPR, ctx->f32); /* texcoord.w */
			}

			/* VGPRs */
			declare_vs_input_vgprs(ctx, &fninfo, &num_prolog_vgprs);
			break;
		}

		declare_per_stage_desc_pointers(ctx, &fninfo, true);
		declare_vs_specific_input_sgprs(ctx, &fninfo);
		ctx->param_vertex_buffers = add_arg(&fninfo, ARG_SGPR,
			ac_array_in_const32_addr_space(ctx->v4i32));

		if (shader->key.as_es) {
			ctx->param_es2gs_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		} else if (shader->key.as_ls) {
			/* no extra parameters */
		} else {
			if (shader->is_gs_copy_shader) {
				fninfo.num_params = ctx->param_vs_state_bits + 1;
				fninfo.num_sgpr_params = fninfo.num_params;
			}

			/* The locations of the other parameters are assigned dynamically. */
			declare_streamout_params(ctx, &shader->selector->so,
						 &fninfo);
		}

		/* VGPRs */
		declare_vs_input_vgprs(ctx, &fninfo, &num_prolog_vgprs);
		break;

	case PIPE_SHADER_TESS_CTRL: /* SI-CI-VI */
		declare_global_desc_pointers(ctx, &fninfo);
		declare_per_stage_desc_pointers(ctx, &fninfo, true);
		ctx->param_tcs_offchip_layout = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_out_lds_offsets = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_out_lds_layout = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_vs_state_bits = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_offchip_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_factor_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);

		/* VGPRs */
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->abi.tcs_patch_id);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->abi.tcs_rel_ids);

		/* param_tcs_offchip_offset and param_tcs_factor_offset are
		 * placed after the user SGPRs.
		 */
		for (i = 0; i < GFX6_TCS_NUM_USER_SGPR + 2; i++)
			returns[num_returns++] = ctx->i32; /* SGPRs */
		for (i = 0; i < 11; i++)
			returns[num_returns++] = ctx->f32; /* VGPRs */
		break;

	case SI_SHADER_MERGED_VERTEX_TESSCTRL:
		/* Merged stages have 8 system SGPRs at the beginning. */
		/* SPI_SHADER_USER_DATA_ADDR_LO/HI_HS */
		if (HAVE_32BIT_POINTERS) {
			declare_per_stage_desc_pointers(ctx, &fninfo,
							ctx->type == PIPE_SHADER_TESS_CTRL);
		} else {
			declare_const_and_shader_buffers(ctx, &fninfo,
							 ctx->type == PIPE_SHADER_TESS_CTRL);
		}
		ctx->param_tcs_offchip_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_merged_wave_info = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_factor_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_merged_scratch_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32); /* unused */
		add_arg(&fninfo, ARG_SGPR, ctx->i32); /* unused */

		declare_global_desc_pointers(ctx, &fninfo);
		declare_per_stage_desc_pointers(ctx, &fninfo,
						ctx->type == PIPE_SHADER_VERTEX);
		declare_vs_specific_input_sgprs(ctx, &fninfo);

		if (!HAVE_32BIT_POINTERS) {
			declare_samplers_and_images(ctx, &fninfo,
						    ctx->type == PIPE_SHADER_TESS_CTRL);
		}
		ctx->param_tcs_offchip_layout = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_out_lds_offsets = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_out_lds_layout = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		if (!HAVE_32BIT_POINTERS) /* Align to 2 dwords. */
			add_arg(&fninfo, ARG_SGPR, ctx->i32); /* unused */
		ctx->param_vertex_buffers = add_arg(&fninfo, ARG_SGPR,
			ac_array_in_const32_addr_space(ctx->v4i32));

		/* VGPRs (first TCS, then VS) */
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->abi.tcs_patch_id);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->abi.tcs_rel_ids);

		if (ctx->type == PIPE_SHADER_VERTEX) {
			declare_vs_input_vgprs(ctx, &fninfo,
					       &num_prolog_vgprs);

			/* LS return values are inputs to the TCS main shader part. */
			for (i = 0; i < 8 + GFX9_TCS_NUM_USER_SGPR; i++)
				returns[num_returns++] = ctx->i32; /* SGPRs */
			for (i = 0; i < 2; i++)
				returns[num_returns++] = ctx->f32; /* VGPRs */
		} else {
			/* TCS return values are inputs to the TCS epilog.
			 *
			 * param_tcs_offchip_offset, param_tcs_factor_offset,
			 * param_tcs_offchip_layout, and param_rw_buffers
			 * should be passed to the epilog.
			 */
			for (i = 0; i <= 8 + GFX9_SGPR_TCS_OUT_LAYOUT; i++)
				returns[num_returns++] = ctx->i32; /* SGPRs */
			for (i = 0; i < 11; i++)
				returns[num_returns++] = ctx->f32; /* VGPRs */
		}
		break;

	case SI_SHADER_MERGED_VERTEX_OR_TESSEVAL_GEOMETRY:
		/* Merged stages have 8 system SGPRs at the beginning. */
		/* SPI_SHADER_USER_DATA_ADDR_LO/HI_GS */
		if (HAVE_32BIT_POINTERS) {
			declare_per_stage_desc_pointers(ctx, &fninfo,
							ctx->type == PIPE_SHADER_GEOMETRY);
		} else {
			declare_const_and_shader_buffers(ctx, &fninfo,
							 ctx->type == PIPE_SHADER_GEOMETRY);
		}
		ctx->param_gs2vs_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_merged_wave_info = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_offchip_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_merged_scratch_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32); /* unused (SPI_SHADER_PGM_LO/HI_GS << 8) */
		add_arg(&fninfo, ARG_SGPR, ctx->i32); /* unused (SPI_SHADER_PGM_LO/HI_GS >> 24) */

		declare_global_desc_pointers(ctx, &fninfo);
		declare_per_stage_desc_pointers(ctx, &fninfo,
						(ctx->type == PIPE_SHADER_VERTEX ||
						 ctx->type == PIPE_SHADER_TESS_EVAL));
		if (ctx->type == PIPE_SHADER_VERTEX) {
			declare_vs_specific_input_sgprs(ctx, &fninfo);
		} else {
			ctx->param_vs_state_bits = add_arg(&fninfo, ARG_SGPR, ctx->i32);
			ctx->param_tcs_offchip_layout = add_arg(&fninfo, ARG_SGPR, ctx->i32);
			ctx->param_tes_offchip_addr = add_arg(&fninfo, ARG_SGPR, ctx->i32);
			/* Declare as many input SGPRs as the VS has. */
			if (!HAVE_32BIT_POINTERS)
				add_arg(&fninfo, ARG_SGPR, ctx->i32); /* unused */
		}

		if (!HAVE_32BIT_POINTERS) {
			declare_samplers_and_images(ctx, &fninfo,
						    ctx->type == PIPE_SHADER_GEOMETRY);
		}
		if (ctx->type == PIPE_SHADER_VERTEX) {
			ctx->param_vertex_buffers = add_arg(&fninfo, ARG_SGPR,
				ac_array_in_const32_addr_space(ctx->v4i32));
		}

		/* VGPRs (first GS, then VS/TES) */
		ctx->param_gs_vtx01_offset = add_arg(&fninfo, ARG_VGPR, ctx->i32);
		ctx->param_gs_vtx23_offset = add_arg(&fninfo, ARG_VGPR, ctx->i32);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->abi.gs_prim_id);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->abi.gs_invocation_id);
		ctx->param_gs_vtx45_offset = add_arg(&fninfo, ARG_VGPR, ctx->i32);

		if (ctx->type == PIPE_SHADER_VERTEX) {
			declare_vs_input_vgprs(ctx, &fninfo,
					       &num_prolog_vgprs);
		} else if (ctx->type == PIPE_SHADER_TESS_EVAL) {
			declare_tes_input_vgprs(ctx, &fninfo);
		}

		if (ctx->type == PIPE_SHADER_VERTEX ||
		    ctx->type == PIPE_SHADER_TESS_EVAL) {
			unsigned num_user_sgprs;

			if (ctx->type == PIPE_SHADER_VERTEX)
				num_user_sgprs = GFX9_VSGS_NUM_USER_SGPR;
			else
				num_user_sgprs = GFX9_TESGS_NUM_USER_SGPR;

			/* ES return values are inputs to GS. */
			for (i = 0; i < 8 + num_user_sgprs; i++)
				returns[num_returns++] = ctx->i32; /* SGPRs */
			for (i = 0; i < 5; i++)
				returns[num_returns++] = ctx->f32; /* VGPRs */
		}
		break;

	case PIPE_SHADER_TESS_EVAL:
		declare_global_desc_pointers(ctx, &fninfo);
		declare_per_stage_desc_pointers(ctx, &fninfo, true);
		ctx->param_vs_state_bits = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_offchip_layout = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tes_offchip_addr = add_arg(&fninfo, ARG_SGPR, ctx->i32);

		if (shader->key.as_es) {
			ctx->param_tcs_offchip_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
			add_arg(&fninfo, ARG_SGPR, ctx->i32);
			ctx->param_es2gs_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		} else {
			add_arg(&fninfo, ARG_SGPR, ctx->i32);
			declare_streamout_params(ctx, &shader->selector->so,
						 &fninfo);
			ctx->param_tcs_offchip_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		}

		/* VGPRs */
		declare_tes_input_vgprs(ctx, &fninfo);
		break;

	case PIPE_SHADER_GEOMETRY:
		declare_global_desc_pointers(ctx, &fninfo);
		declare_per_stage_desc_pointers(ctx, &fninfo, true);
		ctx->param_gs2vs_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_gs_wave_id = add_arg(&fninfo, ARG_SGPR, ctx->i32);

		/* VGPRs */
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->gs_vtx_offset[0]);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->gs_vtx_offset[1]);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->abi.gs_prim_id);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->gs_vtx_offset[2]);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->gs_vtx_offset[3]);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->gs_vtx_offset[4]);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->gs_vtx_offset[5]);
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &ctx->abi.gs_invocation_id);
		break;

	case PIPE_SHADER_FRAGMENT:
		declare_global_desc_pointers(ctx, &fninfo);
		declare_per_stage_desc_pointers(ctx, &fninfo, true);
		add_arg_checked(&fninfo, ARG_SGPR, ctx->f32, SI_PARAM_ALPHA_REF);
		add_arg_assign_checked(&fninfo, ARG_SGPR, ctx->i32,
				       &ctx->abi.prim_mask, SI_PARAM_PRIM_MASK);

		add_arg_checked(&fninfo, ARG_VGPR, ctx->v2i32, SI_PARAM_PERSP_SAMPLE);
		add_arg_checked(&fninfo, ARG_VGPR, ctx->v2i32, SI_PARAM_PERSP_CENTER);
		add_arg_checked(&fninfo, ARG_VGPR, ctx->v2i32, SI_PARAM_PERSP_CENTROID);
		add_arg_checked(&fninfo, ARG_VGPR, v3i32, SI_PARAM_PERSP_PULL_MODEL);
		add_arg_checked(&fninfo, ARG_VGPR, ctx->v2i32, SI_PARAM_LINEAR_SAMPLE);
		add_arg_checked(&fninfo, ARG_VGPR, ctx->v2i32, SI_PARAM_LINEAR_CENTER);
		add_arg_checked(&fninfo, ARG_VGPR, ctx->v2i32, SI_PARAM_LINEAR_CENTROID);
		add_arg_checked(&fninfo, ARG_VGPR, ctx->f32, SI_PARAM_LINE_STIPPLE_TEX);
		add_arg_assign_checked(&fninfo, ARG_VGPR, ctx->f32,
				       &ctx->abi.frag_pos[0], SI_PARAM_POS_X_FLOAT);
		add_arg_assign_checked(&fninfo, ARG_VGPR, ctx->f32,
				       &ctx->abi.frag_pos[1], SI_PARAM_POS_Y_FLOAT);
		add_arg_assign_checked(&fninfo, ARG_VGPR, ctx->f32,
				       &ctx->abi.frag_pos[2], SI_PARAM_POS_Z_FLOAT);
		add_arg_assign_checked(&fninfo, ARG_VGPR, ctx->f32,
				       &ctx->abi.frag_pos[3], SI_PARAM_POS_W_FLOAT);
		add_arg_assign_checked(&fninfo, ARG_VGPR, ctx->i32,
				       &ctx->abi.front_face, SI_PARAM_FRONT_FACE);
		shader->info.face_vgpr_index = 20;
		add_arg_assign_checked(&fninfo, ARG_VGPR, ctx->i32,
				       &ctx->abi.ancillary, SI_PARAM_ANCILLARY);
		shader->info.ancillary_vgpr_index = 21;
		add_arg_assign_checked(&fninfo, ARG_VGPR, ctx->f32,
				       &ctx->abi.sample_coverage, SI_PARAM_SAMPLE_COVERAGE);
		add_arg_checked(&fninfo, ARG_VGPR, ctx->i32, SI_PARAM_POS_FIXED_PT);

		/* Color inputs from the prolog. */
		if (shader->selector->info.colors_read) {
			unsigned num_color_elements =
				util_bitcount(shader->selector->info.colors_read);

			assert(fninfo.num_params + num_color_elements <= ARRAY_SIZE(fninfo.types));
			for (i = 0; i < num_color_elements; i++)
				add_arg(&fninfo, ARG_VGPR, ctx->f32);

			num_prolog_vgprs += num_color_elements;
		}

		/* Outputs for the epilog. */
		num_return_sgprs = SI_SGPR_ALPHA_REF + 1;
		num_returns =
			num_return_sgprs +
			util_bitcount(shader->selector->info.colors_written) * 4 +
			shader->selector->info.writes_z +
			shader->selector->info.writes_stencil +
			shader->selector->info.writes_samplemask +
			1 /* SampleMaskIn */;

		num_returns = MAX2(num_returns,
				   num_return_sgprs +
				   PS_EPILOG_SAMPLEMASK_MIN_LOC + 1);

		for (i = 0; i < num_return_sgprs; i++)
			returns[i] = ctx->i32;
		for (; i < num_returns; i++)
			returns[i] = ctx->f32;
		break;

	case PIPE_SHADER_COMPUTE:
		declare_global_desc_pointers(ctx, &fninfo);
		declare_per_stage_desc_pointers(ctx, &fninfo, true);
		if (shader->selector->info.uses_grid_size)
			add_arg_assign(&fninfo, ARG_SGPR, v3i32, &ctx->abi.num_work_groups);
		if (shader->selector->info.uses_block_size &&
		    shader->selector->info.properties[TGSI_PROPERTY_CS_FIXED_BLOCK_WIDTH] == 0)
			ctx->param_block_size = add_arg(&fninfo, ARG_SGPR, v3i32);

		unsigned cs_user_data_dwords =
			shader->selector->info.properties[TGSI_PROPERTY_CS_USER_DATA_DWORDS];
		if (cs_user_data_dwords) {
			ctx->param_cs_user_data = add_arg(&fninfo, ARG_SGPR,
							  LLVMVectorType(ctx->i32, cs_user_data_dwords));
		}

		for (i = 0; i < 3; i++) {
			ctx->abi.workgroup_ids[i] = NULL;
			if (shader->selector->info.uses_block_id[i])
				add_arg_assign(&fninfo, ARG_SGPR, ctx->i32, &ctx->abi.workgroup_ids[i]);
		}

		add_arg_assign(&fninfo, ARG_VGPR, v3i32, &ctx->abi.local_invocation_ids);
		break;
	default:
		assert(0 && "unimplemented shader");
		return;
	}

	si_create_function(ctx, "main", returns, num_returns, &fninfo,
			   si_get_max_workgroup_size(shader));

	/* Reserve register locations for VGPR inputs the PS prolog may need. */
	if (ctx->type == PIPE_SHADER_FRAGMENT && !ctx->shader->is_monolithic) {
		ac_llvm_add_target_dep_function_attr(ctx->main_fn,
						     "InitialPSInputAddr",
						     S_0286D0_PERSP_SAMPLE_ENA(1) |
						     S_0286D0_PERSP_CENTER_ENA(1) |
						     S_0286D0_PERSP_CENTROID_ENA(1) |
						     S_0286D0_LINEAR_SAMPLE_ENA(1) |
						     S_0286D0_LINEAR_CENTER_ENA(1) |
						     S_0286D0_LINEAR_CENTROID_ENA(1) |
						     S_0286D0_FRONT_FACE_ENA(1) |
						     S_0286D0_ANCILLARY_ENA(1) |
						     S_0286D0_POS_FIXED_PT_ENA(1));
	}

	shader->info.num_input_sgprs = 0;
	shader->info.num_input_vgprs = 0;

	for (i = 0; i < fninfo.num_sgpr_params; ++i)
		shader->info.num_input_sgprs += ac_get_type_size(fninfo.types[i]) / 4;

	for (; i < fninfo.num_params; ++i)
		shader->info.num_input_vgprs += ac_get_type_size(fninfo.types[i]) / 4;

	assert(shader->info.num_input_vgprs >= num_prolog_vgprs);
	shader->info.num_input_vgprs -= num_prolog_vgprs;

	if (shader->key.as_ls ||
	    ctx->type == PIPE_SHADER_TESS_CTRL ||
	    /* GFX9 has the ESGS ring buffer in LDS. */
	    type == SI_SHADER_MERGED_VERTEX_OR_TESSEVAL_GEOMETRY)
		ac_declare_lds_as_pointer(&ctx->ac);
}

/**
 * Load ESGS and GSVS ring buffer resource descriptors and save the variables
 * for later use.
 */
static void preload_ring_buffers(struct si_shader_context *ctx)
{
	LLVMBuilderRef builder = ctx->ac.builder;

	LLVMValueRef buf_ptr = LLVMGetParam(ctx->main_fn,
					    ctx->param_rw_buffers);

	if (ctx->screen->info.chip_class <= VI &&
	    (ctx->shader->key.as_es || ctx->type == PIPE_SHADER_GEOMETRY)) {
		unsigned ring =
			ctx->type == PIPE_SHADER_GEOMETRY ? SI_GS_RING_ESGS
							     : SI_ES_RING_ESGS;
		LLVMValueRef offset = LLVMConstInt(ctx->i32, ring, 0);

		ctx->esgs_ring =
			ac_build_load_to_sgpr(&ctx->ac, buf_ptr, offset);
	}

	if (ctx->shader->is_gs_copy_shader) {
		LLVMValueRef offset = LLVMConstInt(ctx->i32, SI_RING_GSVS, 0);

		ctx->gsvs_ring[0] =
			ac_build_load_to_sgpr(&ctx->ac, buf_ptr, offset);
	} else if (ctx->type == PIPE_SHADER_GEOMETRY) {
		const struct si_shader_selector *sel = ctx->shader->selector;
		LLVMValueRef offset = LLVMConstInt(ctx->i32, SI_RING_GSVS, 0);
		LLVMValueRef base_ring;

		base_ring = ac_build_load_to_sgpr(&ctx->ac, buf_ptr, offset);

		/* The conceptual layout of the GSVS ring is
		 *   v0c0 .. vLv0 v0c1 .. vLc1 ..
		 * but the real memory layout is swizzled across
		 * threads:
		 *   t0v0c0 .. t15v0c0 t0v1c0 .. t15v1c0 ... t15vLcL
		 *   t16v0c0 ..
		 * Override the buffer descriptor accordingly.
		 */
		LLVMTypeRef v2i64 = LLVMVectorType(ctx->i64, 2);
		uint64_t stream_offset = 0;

		for (unsigned stream = 0; stream < 4; ++stream) {
			unsigned num_components;
			unsigned stride;
			unsigned num_records;
			LLVMValueRef ring, tmp;

			num_components = sel->info.num_stream_output_components[stream];
			if (!num_components)
				continue;

			stride = 4 * num_components * sel->gs_max_out_vertices;

			/* Limit on the stride field for <= CIK. */
			assert(stride < (1 << 14));

			num_records = 64;

			ring = LLVMBuildBitCast(builder, base_ring, v2i64, "");
			tmp = LLVMBuildExtractElement(builder, ring, ctx->i32_0, "");
			tmp = LLVMBuildAdd(builder, tmp,
					   LLVMConstInt(ctx->i64,
							stream_offset, 0), "");
			stream_offset += stride * 64;

			ring = LLVMBuildInsertElement(builder, ring, tmp, ctx->i32_0, "");
			ring = LLVMBuildBitCast(builder, ring, ctx->v4i32, "");
			tmp = LLVMBuildExtractElement(builder, ring, ctx->i32_1, "");
			tmp = LLVMBuildOr(builder, tmp,
				LLVMConstInt(ctx->i32,
					     S_008F04_STRIDE(stride) |
					     S_008F04_SWIZZLE_ENABLE(1), 0), "");
			ring = LLVMBuildInsertElement(builder, ring, tmp, ctx->i32_1, "");
			ring = LLVMBuildInsertElement(builder, ring,
					LLVMConstInt(ctx->i32, num_records, 0),
					LLVMConstInt(ctx->i32, 2, 0), "");
			ring = LLVMBuildInsertElement(builder, ring,
				LLVMConstInt(ctx->i32,
					     S_008F0C_DST_SEL_X(V_008F0C_SQ_SEL_X) |
					     S_008F0C_DST_SEL_Y(V_008F0C_SQ_SEL_Y) |
					     S_008F0C_DST_SEL_Z(V_008F0C_SQ_SEL_Z) |
					     S_008F0C_DST_SEL_W(V_008F0C_SQ_SEL_W) |
					     S_008F0C_NUM_FORMAT(V_008F0C_BUF_NUM_FORMAT_FLOAT) |
					     S_008F0C_DATA_FORMAT(V_008F0C_BUF_DATA_FORMAT_32) |
					     S_008F0C_ELEMENT_SIZE(1) | /* element_size = 4 (bytes) */
					     S_008F0C_INDEX_STRIDE(1) | /* index_stride = 16 (elements) */
					     S_008F0C_ADD_TID_ENABLE(1),
					     0),
				LLVMConstInt(ctx->i32, 3, 0), "");

			ctx->gsvs_ring[stream] = ring;
		}
	} else if (ctx->type == PIPE_SHADER_TESS_EVAL) {
		ctx->tess_offchip_ring = get_tess_ring_descriptor(ctx, TESS_OFFCHIP_RING_TES);
	}
}

static void si_llvm_emit_polygon_stipple(struct si_shader_context *ctx,
					 LLVMValueRef param_rw_buffers,
					 unsigned param_pos_fixed_pt)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef slot, desc, offset, row, bit, address[2];

	/* Use the fixed-point gl_FragCoord input.
	 * Since the stipple pattern is 32x32 and it repeats, just get 5 bits
	 * per coordinate to get the repeating effect.
	 */
	address[0] = si_unpack_param(ctx, param_pos_fixed_pt, 0, 5);
	address[1] = si_unpack_param(ctx, param_pos_fixed_pt, 16, 5);

	/* Load the buffer descriptor. */
	slot = LLVMConstInt(ctx->i32, SI_PS_CONST_POLY_STIPPLE, 0);
	desc = ac_build_load_to_sgpr(&ctx->ac, param_rw_buffers, slot);

	/* The stipple pattern is 32x32, each row has 32 bits. */
	offset = LLVMBuildMul(builder, address[1],
			      LLVMConstInt(ctx->i32, 4, 0), "");
	row = buffer_load_const(ctx, desc, offset);
	row = ac_to_integer(&ctx->ac, row);
	bit = LLVMBuildLShr(builder, row, address[0], "");
	bit = LLVMBuildTrunc(builder, bit, ctx->i1, "");
	ac_build_kill_if_false(&ctx->ac, bit);
}

void si_shader_binary_read_config(struct ac_shader_binary *binary,
				  struct si_shader_config *conf,
				  unsigned symbol_offset)
{
	unsigned i;
	const unsigned char *config =
		ac_shader_binary_config_start(binary, symbol_offset);
	bool really_needs_scratch = false;

	/* LLVM adds SGPR spills to the scratch size.
	 * Find out if we really need the scratch buffer.
	 */
	for (i = 0; i < binary->reloc_count; i++) {
		const struct ac_shader_reloc *reloc = &binary->relocs[i];

		if (!strcmp(scratch_rsrc_dword0_symbol, reloc->name) ||
		    !strcmp(scratch_rsrc_dword1_symbol, reloc->name)) {
			really_needs_scratch = true;
			break;
		}
	}

	/* XXX: We may be able to emit some of these values directly rather than
	 * extracting fields to be emitted later.
	 */

	for (i = 0; i < binary->config_size_per_symbol; i+= 8) {
		unsigned reg = util_le32_to_cpu(*(uint32_t*)(config + i));
		unsigned value = util_le32_to_cpu(*(uint32_t*)(config + i + 4));
		switch (reg) {
		case R_00B028_SPI_SHADER_PGM_RSRC1_PS:
		case R_00B128_SPI_SHADER_PGM_RSRC1_VS:
		case R_00B228_SPI_SHADER_PGM_RSRC1_GS:
		case R_00B428_SPI_SHADER_PGM_RSRC1_HS:
		case R_00B848_COMPUTE_PGM_RSRC1:
			conf->num_sgprs = MAX2(conf->num_sgprs, (G_00B028_SGPRS(value) + 1) * 8);
			conf->num_vgprs = MAX2(conf->num_vgprs, (G_00B028_VGPRS(value) + 1) * 4);
			conf->float_mode =  G_00B028_FLOAT_MODE(value);
			conf->rsrc1 = value;
			break;
		case R_00B02C_SPI_SHADER_PGM_RSRC2_PS:
			conf->lds_size = MAX2(conf->lds_size, G_00B02C_EXTRA_LDS_SIZE(value));
			break;
		case R_00B84C_COMPUTE_PGM_RSRC2:
			conf->lds_size = MAX2(conf->lds_size, G_00B84C_LDS_SIZE(value));
			conf->rsrc2 = value;
			break;
		case R_0286CC_SPI_PS_INPUT_ENA:
			conf->spi_ps_input_ena = value;
			break;
		case R_0286D0_SPI_PS_INPUT_ADDR:
			conf->spi_ps_input_addr = value;
			break;
		case R_0286E8_SPI_TMPRING_SIZE:
		case R_00B860_COMPUTE_TMPRING_SIZE:
			/* WAVESIZE is in units of 256 dwords. */
			if (really_needs_scratch)
				conf->scratch_bytes_per_wave =
					G_00B860_WAVESIZE(value) * 256 * 4;
			break;
		case 0x4: /* SPILLED_SGPRS */
			conf->spilled_sgprs = value;
			break;
		case 0x8: /* SPILLED_VGPRS */
			conf->spilled_vgprs = value;
			break;
		default:
			{
				static bool printed;

				if (!printed) {
					fprintf(stderr, "Warning: LLVM emitted unknown "
						"config register: 0x%x\n", reg);
					printed = true;
				}
			}
			break;
		}
	}

	if (!conf->spi_ps_input_addr)
		conf->spi_ps_input_addr = conf->spi_ps_input_ena;
}

void si_shader_apply_scratch_relocs(struct si_shader *shader,
				    uint64_t scratch_va)
{
	unsigned i;
	uint32_t scratch_rsrc_dword0 = scratch_va;
	uint32_t scratch_rsrc_dword1 =
		S_008F04_BASE_ADDRESS_HI(scratch_va >> 32);

	/* Enable scratch coalescing. */
	scratch_rsrc_dword1 |= S_008F04_SWIZZLE_ENABLE(1);

	for (i = 0 ; i < shader->binary.reloc_count; i++) {
		const struct ac_shader_reloc *reloc =
					&shader->binary.relocs[i];
		if (!strcmp(scratch_rsrc_dword0_symbol, reloc->name)) {
			util_memcpy_cpu_to_le32(shader->binary.code + reloc->offset,
			&scratch_rsrc_dword0, 4);
		} else if (!strcmp(scratch_rsrc_dword1_symbol, reloc->name)) {
			util_memcpy_cpu_to_le32(shader->binary.code + reloc->offset,
			&scratch_rsrc_dword1, 4);
		}
	}
}

/* For the UMR disassembler. */
#define DEBUGGER_END_OF_CODE_MARKER	0xbf9f0000 /* invalid instruction */
#define DEBUGGER_NUM_MARKERS		5

static unsigned si_get_shader_binary_size(const struct si_shader *shader)
{
	unsigned size = shader->binary.code_size;

	if (shader->prolog)
		size += shader->prolog->binary.code_size;
	if (shader->previous_stage)
		size += shader->previous_stage->binary.code_size;
	if (shader->prolog2)
		size += shader->prolog2->binary.code_size;
	if (shader->epilog)
		size += shader->epilog->binary.code_size;
	return size + DEBUGGER_NUM_MARKERS * 4;
}

int si_shader_binary_upload(struct si_screen *sscreen, struct si_shader *shader)
{
	const struct ac_shader_binary *prolog =
		shader->prolog ? &shader->prolog->binary : NULL;
	const struct ac_shader_binary *previous_stage =
		shader->previous_stage ? &shader->previous_stage->binary : NULL;
	const struct ac_shader_binary *prolog2 =
		shader->prolog2 ? &shader->prolog2->binary : NULL;
	const struct ac_shader_binary *epilog =
		shader->epilog ? &shader->epilog->binary : NULL;
	const struct ac_shader_binary *mainb = &shader->binary;
	unsigned bo_size = si_get_shader_binary_size(shader) +
			   (!epilog ? mainb->rodata_size : 0);
	unsigned char *ptr;

	assert(!prolog || !prolog->rodata_size);
	assert(!previous_stage || !previous_stage->rodata_size);
	assert(!prolog2 || !prolog2->rodata_size);
	assert((!prolog && !previous_stage && !prolog2 && !epilog) ||
	       !mainb->rodata_size);
	assert(!epilog || !epilog->rodata_size);

	r600_resource_reference(&shader->bo, NULL);
	shader->bo = si_aligned_buffer_create(&sscreen->b,
					      sscreen->cpdma_prefetch_writes_memory ?
						0 : SI_RESOURCE_FLAG_READ_ONLY,
                                              PIPE_USAGE_IMMUTABLE,
                                              align(bo_size, SI_CPDMA_ALIGNMENT),
                                              256);
	if (!shader->bo)
		return -ENOMEM;

	/* Upload. */
	ptr = sscreen->ws->buffer_map(shader->bo->buf, NULL,
					PIPE_TRANSFER_READ_WRITE |
					PIPE_TRANSFER_UNSYNCHRONIZED);

	/* Don't use util_memcpy_cpu_to_le32. LLVM binaries are
	 * endian-independent. */
	if (prolog) {
		memcpy(ptr, prolog->code, prolog->code_size);
		ptr += prolog->code_size;
	}
	if (previous_stage) {
		memcpy(ptr, previous_stage->code, previous_stage->code_size);
		ptr += previous_stage->code_size;
	}
	if (prolog2) {
		memcpy(ptr, prolog2->code, prolog2->code_size);
		ptr += prolog2->code_size;
	}

	memcpy(ptr, mainb->code, mainb->code_size);
	ptr += mainb->code_size;

	if (epilog) {
		memcpy(ptr, epilog->code, epilog->code_size);
		ptr += epilog->code_size;
	} else if (mainb->rodata_size > 0) {
		memcpy(ptr, mainb->rodata, mainb->rodata_size);
		ptr += mainb->rodata_size;
	}

	/* Add end-of-code markers for the UMR disassembler. */
	uint32_t *ptr32 = (uint32_t*)ptr;
	for (unsigned i = 0; i < DEBUGGER_NUM_MARKERS; i++)
		ptr32[i] = DEBUGGER_END_OF_CODE_MARKER;

	sscreen->ws->buffer_unmap(shader->bo->buf);
	return 0;
}

static void si_shader_dump_disassembly(const struct ac_shader_binary *binary,
				       struct pipe_debug_callback *debug,
				       const char *name, FILE *file)
{
	char *line, *p;
	unsigned i, count;

	if (binary->disasm_string) {
		fprintf(file, "Shader %s disassembly:\n", name);
		fprintf(file, "%s", binary->disasm_string);

		if (debug && debug->debug_message) {
			/* Very long debug messages are cut off, so send the
			 * disassembly one line at a time. This causes more
			 * overhead, but on the plus side it simplifies
			 * parsing of resulting logs.
			 */
			pipe_debug_message(debug, SHADER_INFO,
					   "Shader Disassembly Begin");

			line = binary->disasm_string;
			while (*line) {
				p = util_strchrnul(line, '\n');
				count = p - line;

				if (count) {
					pipe_debug_message(debug, SHADER_INFO,
							   "%.*s", count, line);
				}

				if (!*p)
					break;
				line = p + 1;
			}

			pipe_debug_message(debug, SHADER_INFO,
					   "Shader Disassembly End");
		}
	} else {
		fprintf(file, "Shader %s binary:\n", name);
		for (i = 0; i < binary->code_size; i += 4) {
			fprintf(file, "@0x%x: %02x%02x%02x%02x\n", i,
				binary->code[i + 3], binary->code[i + 2],
				binary->code[i + 1], binary->code[i]);
		}
	}
}

static void si_calculate_max_simd_waves(struct si_shader *shader)
{
	struct si_screen *sscreen = shader->selector->screen;
	struct si_shader_config *conf = &shader->config;
	unsigned num_inputs = shader->selector->info.num_inputs;
	unsigned lds_increment = sscreen->info.chip_class >= CIK ? 512 : 256;
	unsigned lds_per_wave = 0;
	unsigned max_simd_waves;

	max_simd_waves = ac_get_max_simd_waves(sscreen->info.family);

	/* Compute LDS usage for PS. */
	switch (shader->selector->type) {
	case PIPE_SHADER_FRAGMENT:
		/* The minimum usage per wave is (num_inputs * 48). The maximum
		 * usage is (num_inputs * 48 * 16).
		 * We can get anything in between and it varies between waves.
		 *
		 * The 48 bytes per input for a single primitive is equal to
		 * 4 bytes/component * 4 components/input * 3 points.
		 *
		 * Other stages don't know the size at compile time or don't
		 * allocate LDS per wave, but instead they do it per thread group.
		 */
		lds_per_wave = conf->lds_size * lds_increment +
			       align(num_inputs * 48, lds_increment);
		break;
	case PIPE_SHADER_COMPUTE:
		if (shader->selector) {
			unsigned max_workgroup_size =
				si_get_max_workgroup_size(shader);
			lds_per_wave = (conf->lds_size * lds_increment) /
				       DIV_ROUND_UP(max_workgroup_size, 64);
		}
		break;
	}

	/* Compute the per-SIMD wave counts. */
	if (conf->num_sgprs) {
		if (sscreen->info.chip_class >= VI)
			max_simd_waves = MIN2(max_simd_waves, 800 / conf->num_sgprs);
		else
			max_simd_waves = MIN2(max_simd_waves, 512 / conf->num_sgprs);
	}

	if (conf->num_vgprs)
		max_simd_waves = MIN2(max_simd_waves, 256 / conf->num_vgprs);

	/* LDS is 64KB per CU (4 SIMDs), which is 16KB per SIMD (usage above
	 * 16KB makes some SIMDs unoccupied). */
	if (lds_per_wave)
		max_simd_waves = MIN2(max_simd_waves, 16384 / lds_per_wave);

	conf->max_simd_waves = max_simd_waves;
}

void si_shader_dump_stats_for_shader_db(const struct si_shader *shader,
					struct pipe_debug_callback *debug)
{
	const struct si_shader_config *conf = &shader->config;

	pipe_debug_message(debug, SHADER_INFO,
			   "Shader Stats: SGPRS: %d VGPRS: %d Code Size: %d "
			   "LDS: %d Scratch: %d Max Waves: %d Spilled SGPRs: %d "
			   "Spilled VGPRs: %d PrivMem VGPRs: %d",
			   conf->num_sgprs, conf->num_vgprs,
			   si_get_shader_binary_size(shader),
			   conf->lds_size, conf->scratch_bytes_per_wave,
			   conf->max_simd_waves, conf->spilled_sgprs,
			   conf->spilled_vgprs, conf->private_mem_vgprs);
}

static void si_shader_dump_stats(struct si_screen *sscreen,
				 const struct si_shader *shader,
			         unsigned processor,
				 FILE *file,
				 bool check_debug_option)
{
	const struct si_shader_config *conf = &shader->config;

	if (!check_debug_option ||
	    si_can_dump_shader(sscreen, processor)) {
		if (processor == PIPE_SHADER_FRAGMENT) {
			fprintf(file, "*** SHADER CONFIG ***\n"
				"SPI_PS_INPUT_ADDR = 0x%04x\n"
				"SPI_PS_INPUT_ENA  = 0x%04x\n",
				conf->spi_ps_input_addr, conf->spi_ps_input_ena);
		}

		fprintf(file, "*** SHADER STATS ***\n"
			"SGPRS: %d\n"
			"VGPRS: %d\n"
		        "Spilled SGPRs: %d\n"
			"Spilled VGPRs: %d\n"
			"Private memory VGPRs: %d\n"
			"Code Size: %d bytes\n"
			"LDS: %d blocks\n"
			"Scratch: %d bytes per wave\n"
			"Max Waves: %d\n"
			"********************\n\n\n",
			conf->num_sgprs, conf->num_vgprs,
			conf->spilled_sgprs, conf->spilled_vgprs,
			conf->private_mem_vgprs,
			si_get_shader_binary_size(shader),
			conf->lds_size, conf->scratch_bytes_per_wave,
			conf->max_simd_waves);
	}
}

const char *si_get_shader_name(const struct si_shader *shader, unsigned processor)
{
	switch (processor) {
	case PIPE_SHADER_VERTEX:
		if (shader->key.as_es)
			return "Vertex Shader as ES";
		else if (shader->key.as_ls)
			return "Vertex Shader as LS";
		else
			return "Vertex Shader as VS";
	case PIPE_SHADER_TESS_CTRL:
		return "Tessellation Control Shader";
	case PIPE_SHADER_TESS_EVAL:
		if (shader->key.as_es)
			return "Tessellation Evaluation Shader as ES";
		else
			return "Tessellation Evaluation Shader as VS";
	case PIPE_SHADER_GEOMETRY:
		if (shader->is_gs_copy_shader)
			return "GS Copy Shader as VS";
		else
			return "Geometry Shader";
	case PIPE_SHADER_FRAGMENT:
		return "Pixel Shader";
	case PIPE_SHADER_COMPUTE:
		return "Compute Shader";
	default:
		return "Unknown Shader";
	}
}

void si_shader_dump(struct si_screen *sscreen, const struct si_shader *shader,
		    struct pipe_debug_callback *debug, unsigned processor,
		    FILE *file, bool check_debug_option)
{
	if (!check_debug_option ||
	    si_can_dump_shader(sscreen, processor))
		si_dump_shader_key(processor, shader, file);

	if (!check_debug_option && shader->binary.llvm_ir_string) {
		if (shader->previous_stage &&
		    shader->previous_stage->binary.llvm_ir_string) {
			fprintf(file, "\n%s - previous stage - LLVM IR:\n\n",
				si_get_shader_name(shader, processor));
			fprintf(file, "%s\n", shader->previous_stage->binary.llvm_ir_string);
		}

		fprintf(file, "\n%s - main shader part - LLVM IR:\n\n",
			si_get_shader_name(shader, processor));
		fprintf(file, "%s\n", shader->binary.llvm_ir_string);
	}

	if (!check_debug_option ||
	    (si_can_dump_shader(sscreen, processor) &&
	     !(sscreen->debug_flags & DBG(NO_ASM)))) {
		fprintf(file, "\n%s:\n", si_get_shader_name(shader, processor));

		if (shader->prolog)
			si_shader_dump_disassembly(&shader->prolog->binary,
						   debug, "prolog", file);
		if (shader->previous_stage)
			si_shader_dump_disassembly(&shader->previous_stage->binary,
						   debug, "previous stage", file);
		if (shader->prolog2)
			si_shader_dump_disassembly(&shader->prolog2->binary,
						   debug, "prolog2", file);

		si_shader_dump_disassembly(&shader->binary, debug, "main", file);

		if (shader->epilog)
			si_shader_dump_disassembly(&shader->epilog->binary,
						   debug, "epilog", file);
		fprintf(file, "\n");
	}

	si_shader_dump_stats(sscreen, shader, processor, file,
			     check_debug_option);
}

static int si_compile_llvm(struct si_screen *sscreen,
			   struct ac_shader_binary *binary,
			   struct si_shader_config *conf,
			   struct ac_llvm_compiler *compiler,
			   LLVMModuleRef mod,
			   struct pipe_debug_callback *debug,
			   unsigned processor,
			   const char *name,
			   bool less_optimized)
{
	int r = 0;
	unsigned count = p_atomic_inc_return(&sscreen->num_compilations);

	if (si_can_dump_shader(sscreen, processor)) {
		fprintf(stderr, "radeonsi: Compiling shader %d\n", count);

		if (!(sscreen->debug_flags & (DBG(NO_IR) | DBG(PREOPT_IR)))) {
			fprintf(stderr, "%s LLVM IR:\n\n", name);
			ac_dump_module(mod);
			fprintf(stderr, "\n");
		}
	}

	if (sscreen->record_llvm_ir) {
		char *ir = LLVMPrintModuleToString(mod);
		binary->llvm_ir_string = strdup(ir);
		LLVMDisposeMessage(ir);
	}

	if (!si_replace_shader(count, binary)) {
		r = si_llvm_compile(mod, binary, compiler, debug,
				    less_optimized);
		if (r)
			return r;
	}

	si_shader_binary_read_config(binary, conf, 0);

	/* Enable 64-bit and 16-bit denormals, because there is no performance
	 * cost.
	 *
	 * If denormals are enabled, all floating-point output modifiers are
	 * ignored.
	 *
	 * Don't enable denormals for 32-bit floats, because:
	 * - Floating-point output modifiers would be ignored by the hw.
	 * - Some opcodes don't support denormals, such as v_mad_f32. We would
	 *   have to stop using those.
	 * - SI & CI would be very slow.
	 */
	conf->float_mode |= V_00B028_FP_64_DENORMS;

	FREE(binary->config);
	FREE(binary->global_symbol_offsets);
	binary->config = NULL;
	binary->global_symbol_offsets = NULL;

	/* Some shaders can't have rodata because their binaries can be
	 * concatenated.
	 */
	if (binary->rodata_size &&
	    (processor == PIPE_SHADER_VERTEX ||
	     processor == PIPE_SHADER_TESS_CTRL ||
	     processor == PIPE_SHADER_TESS_EVAL ||
	     processor == PIPE_SHADER_FRAGMENT)) {
		fprintf(stderr, "radeonsi: The shader can't have rodata.");
		return -EINVAL;
	}

	return r;
}

static void si_llvm_build_ret(struct si_shader_context *ctx, LLVMValueRef ret)
{
	if (LLVMGetTypeKind(LLVMTypeOf(ret)) == LLVMVoidTypeKind)
		LLVMBuildRetVoid(ctx->ac.builder);
	else
		LLVMBuildRet(ctx->ac.builder, ret);
}

/* Generate code for the hardware VS shader stage to go with a geometry shader */
struct si_shader *
si_generate_gs_copy_shader(struct si_screen *sscreen,
			   struct ac_llvm_compiler *compiler,
			   struct si_shader_selector *gs_selector,
			   struct pipe_debug_callback *debug)
{
	struct si_shader_context ctx;
	struct si_shader *shader;
	LLVMBuilderRef builder;
	struct si_shader_output_values outputs[SI_MAX_VS_OUTPUTS];
	struct tgsi_shader_info *gsinfo = &gs_selector->info;
	int i, r;


	shader = CALLOC_STRUCT(si_shader);
	if (!shader)
		return NULL;

	/* We can leave the fence as permanently signaled because the GS copy
	 * shader only becomes visible globally after it has been compiled. */
	util_queue_fence_init(&shader->ready);

	shader->selector = gs_selector;
	shader->is_gs_copy_shader = true;

	si_init_shader_ctx(&ctx, sscreen, compiler);
	ctx.shader = shader;
	ctx.type = PIPE_SHADER_VERTEX;

	builder = ctx.ac.builder;

	create_function(&ctx);
	preload_ring_buffers(&ctx);

	LLVMValueRef voffset =
		LLVMBuildMul(ctx.ac.builder, ctx.abi.vertex_id,
			     LLVMConstInt(ctx.i32, 4, 0), "");

	/* Fetch the vertex stream ID.*/
	LLVMValueRef stream_id;

	if (gs_selector->so.num_outputs)
		stream_id = si_unpack_param(&ctx, ctx.param_streamout_config, 24, 2);
	else
		stream_id = ctx.i32_0;

	/* Fill in output information. */
	for (i = 0; i < gsinfo->num_outputs; ++i) {
		outputs[i].semantic_name = gsinfo->output_semantic_name[i];
		outputs[i].semantic_index = gsinfo->output_semantic_index[i];

		for (int chan = 0; chan < 4; chan++) {
			outputs[i].vertex_stream[chan] =
				(gsinfo->output_streams[i] >> (2 * chan)) & 3;
		}
	}

	LLVMBasicBlockRef end_bb;
	LLVMValueRef switch_inst;

	end_bb = LLVMAppendBasicBlockInContext(ctx.ac.context, ctx.main_fn, "end");
	switch_inst = LLVMBuildSwitch(builder, stream_id, end_bb, 4);

	for (int stream = 0; stream < 4; stream++) {
		LLVMBasicBlockRef bb;
		unsigned offset;

		if (!gsinfo->num_stream_output_components[stream])
			continue;

		if (stream > 0 && !gs_selector->so.num_outputs)
			continue;

		bb = LLVMInsertBasicBlockInContext(ctx.ac.context, end_bb, "out");
		LLVMAddCase(switch_inst, LLVMConstInt(ctx.i32, stream, 0), bb);
		LLVMPositionBuilderAtEnd(builder, bb);

		/* Fetch vertex data from GSVS ring */
		offset = 0;
		for (i = 0; i < gsinfo->num_outputs; ++i) {
			for (unsigned chan = 0; chan < 4; chan++) {
				if (!(gsinfo->output_usagemask[i] & (1 << chan)) ||
				    outputs[i].vertex_stream[chan] != stream) {
					outputs[i].values[chan] = LLVMGetUndef(ctx.f32);
					continue;
				}

				LLVMValueRef soffset = LLVMConstInt(ctx.i32,
					offset * gs_selector->gs_max_out_vertices * 16 * 4, 0);
				offset++;

				outputs[i].values[chan] =
					ac_build_buffer_load(&ctx.ac,
							     ctx.gsvs_ring[0], 1,
							     ctx.i32_0, voffset,
							     soffset, 0, 1, 1,
							     true, false);
			}
		}

		/* Streamout and exports. */
		if (gs_selector->so.num_outputs) {
			si_llvm_emit_streamout(&ctx, outputs,
					       gsinfo->num_outputs,
					       stream);
		}

		if (stream == 0) {
			/* Vertex color clamping.
			 *
			 * This uses a state constant loaded in a user data SGPR and
			 * an IF statement is added that clamps all colors if the constant
			 * is true.
			 */
			struct lp_build_if_state if_ctx;
			LLVMValueRef v[2], cond = NULL;
			LLVMBasicBlockRef blocks[2];

			for (unsigned i = 0; i < gsinfo->num_outputs; i++) {
				if (gsinfo->output_semantic_name[i] != TGSI_SEMANTIC_COLOR &&
				    gsinfo->output_semantic_name[i] != TGSI_SEMANTIC_BCOLOR)
					continue;

				/* We've found a color. */
				if (!cond) {
					/* The state is in the first bit of the user SGPR. */
					cond = LLVMGetParam(ctx.main_fn,
							    ctx.param_vs_state_bits);
					cond = LLVMBuildTrunc(ctx.ac.builder, cond,
							      ctx.i1, "");
					lp_build_if(&if_ctx, &ctx.gallivm, cond);
					/* Remember blocks for Phi. */
					blocks[0] = if_ctx.true_block;
					blocks[1] = if_ctx.entry_block;
				}

				for (unsigned j = 0; j < 4; j++) {
					/* Insert clamp into the true block. */
					v[0] = ac_build_clamp(&ctx.ac, outputs[i].values[j]);
					v[1] = outputs[i].values[j];

					/* Insert Phi into the endif block. */
					LLVMPositionBuilderAtEnd(ctx.ac.builder, if_ctx.merge_block);
					outputs[i].values[j] = ac_build_phi(&ctx.ac, ctx.f32, 2, v, blocks);
					LLVMPositionBuilderAtEnd(ctx.ac.builder, if_ctx.true_block);
				}
			}
			if (cond)
				lp_build_endif(&if_ctx);

			si_llvm_export_vs(&ctx, outputs, gsinfo->num_outputs);
		}

		LLVMBuildBr(builder, end_bb);
	}

	LLVMPositionBuilderAtEnd(builder, end_bb);

	LLVMBuildRetVoid(ctx.ac.builder);

	ctx.type = PIPE_SHADER_GEOMETRY; /* override for shader dumping */
	si_llvm_optimize_module(&ctx);

	r = si_compile_llvm(sscreen, &ctx.shader->binary,
			    &ctx.shader->config, ctx.compiler,
			    ctx.ac.module,
			    debug, PIPE_SHADER_GEOMETRY,
			    "GS Copy Shader", false);
	if (!r) {
		if (si_can_dump_shader(sscreen, PIPE_SHADER_GEOMETRY))
			fprintf(stderr, "GS Copy Shader:\n");
		si_shader_dump(sscreen, ctx.shader, debug,
			       PIPE_SHADER_GEOMETRY, stderr, true);
		r = si_shader_binary_upload(sscreen, ctx.shader);
	}

	si_llvm_dispose(&ctx);

	if (r != 0) {
		FREE(shader);
		shader = NULL;
	}
	return shader;
}

static void si_dump_shader_key_vs(const struct si_shader_key *key,
				  const struct si_vs_prolog_bits *prolog,
				  const char *prefix, FILE *f)
{
	fprintf(f, "  %s.instance_divisor_is_one = %u\n",
		prefix, prolog->instance_divisor_is_one);
	fprintf(f, "  %s.instance_divisor_is_fetched = %u\n",
		prefix, prolog->instance_divisor_is_fetched);
	fprintf(f, "  %s.ls_vgpr_fix = %u\n",
		prefix, prolog->ls_vgpr_fix);

	fprintf(f, "  mono.vs.fix_fetch = {");
	for (int i = 0; i < SI_MAX_ATTRIBS; i++)
		fprintf(f, !i ? "%u" : ", %u", key->mono.vs_fix_fetch[i]);
	fprintf(f, "}\n");
}

static void si_dump_shader_key(unsigned processor, const struct si_shader *shader,
			       FILE *f)
{
	const struct si_shader_key *key = &shader->key;

	fprintf(f, "SHADER KEY\n");

	switch (processor) {
	case PIPE_SHADER_VERTEX:
		si_dump_shader_key_vs(key, &key->part.vs.prolog,
				      "part.vs.prolog", f);
		fprintf(f, "  as_es = %u\n", key->as_es);
		fprintf(f, "  as_ls = %u\n", key->as_ls);
		fprintf(f, "  mono.u.vs_export_prim_id = %u\n",
			key->mono.u.vs_export_prim_id);
		break;

	case PIPE_SHADER_TESS_CTRL:
		if (shader->selector->screen->info.chip_class >= GFX9) {
			si_dump_shader_key_vs(key, &key->part.tcs.ls_prolog,
					      "part.tcs.ls_prolog", f);
		}
		fprintf(f, "  part.tcs.epilog.prim_mode = %u\n", key->part.tcs.epilog.prim_mode);
		fprintf(f, "  mono.u.ff_tcs_inputs_to_copy = 0x%"PRIx64"\n", key->mono.u.ff_tcs_inputs_to_copy);
		break;

	case PIPE_SHADER_TESS_EVAL:
		fprintf(f, "  as_es = %u\n", key->as_es);
		fprintf(f, "  mono.u.vs_export_prim_id = %u\n",
			key->mono.u.vs_export_prim_id);
		break;

	case PIPE_SHADER_GEOMETRY:
		if (shader->is_gs_copy_shader)
			break;

		if (shader->selector->screen->info.chip_class >= GFX9 &&
		    key->part.gs.es->type == PIPE_SHADER_VERTEX) {
			si_dump_shader_key_vs(key, &key->part.gs.vs_prolog,
					      "part.gs.vs_prolog", f);
		}
		fprintf(f, "  part.gs.prolog.tri_strip_adj_fix = %u\n", key->part.gs.prolog.tri_strip_adj_fix);
		break;

	case PIPE_SHADER_COMPUTE:
		break;

	case PIPE_SHADER_FRAGMENT:
		fprintf(f, "  part.ps.prolog.color_two_side = %u\n", key->part.ps.prolog.color_two_side);
		fprintf(f, "  part.ps.prolog.flatshade_colors = %u\n", key->part.ps.prolog.flatshade_colors);
		fprintf(f, "  part.ps.prolog.poly_stipple = %u\n", key->part.ps.prolog.poly_stipple);
		fprintf(f, "  part.ps.prolog.force_persp_sample_interp = %u\n", key->part.ps.prolog.force_persp_sample_interp);
		fprintf(f, "  part.ps.prolog.force_linear_sample_interp = %u\n", key->part.ps.prolog.force_linear_sample_interp);
		fprintf(f, "  part.ps.prolog.force_persp_center_interp = %u\n", key->part.ps.prolog.force_persp_center_interp);
		fprintf(f, "  part.ps.prolog.force_linear_center_interp = %u\n", key->part.ps.prolog.force_linear_center_interp);
		fprintf(f, "  part.ps.prolog.bc_optimize_for_persp = %u\n", key->part.ps.prolog.bc_optimize_for_persp);
		fprintf(f, "  part.ps.prolog.bc_optimize_for_linear = %u\n", key->part.ps.prolog.bc_optimize_for_linear);
		fprintf(f, "  part.ps.epilog.spi_shader_col_format = 0x%x\n", key->part.ps.epilog.spi_shader_col_format);
		fprintf(f, "  part.ps.epilog.color_is_int8 = 0x%X\n", key->part.ps.epilog.color_is_int8);
		fprintf(f, "  part.ps.epilog.color_is_int10 = 0x%X\n", key->part.ps.epilog.color_is_int10);
		fprintf(f, "  part.ps.epilog.last_cbuf = %u\n", key->part.ps.epilog.last_cbuf);
		fprintf(f, "  part.ps.epilog.alpha_func = %u\n", key->part.ps.epilog.alpha_func);
		fprintf(f, "  part.ps.epilog.alpha_to_one = %u\n", key->part.ps.epilog.alpha_to_one);
		fprintf(f, "  part.ps.epilog.poly_line_smoothing = %u\n", key->part.ps.epilog.poly_line_smoothing);
		fprintf(f, "  part.ps.epilog.clamp_color = %u\n", key->part.ps.epilog.clamp_color);
		break;

	default:
		assert(0);
	}

	if ((processor == PIPE_SHADER_GEOMETRY ||
	     processor == PIPE_SHADER_TESS_EVAL ||
	     processor == PIPE_SHADER_VERTEX) &&
	    !key->as_es && !key->as_ls) {
		fprintf(f, "  opt.kill_outputs = 0x%"PRIx64"\n", key->opt.kill_outputs);
		fprintf(f, "  opt.clip_disable = %u\n", key->opt.clip_disable);
	}
}

static void si_init_shader_ctx(struct si_shader_context *ctx,
			       struct si_screen *sscreen,
			       struct ac_llvm_compiler *compiler)
{
	struct lp_build_tgsi_context *bld_base;

	si_llvm_context_init(ctx, sscreen, compiler);

	bld_base = &ctx->bld_base;
	bld_base->emit_fetch_funcs[TGSI_FILE_CONSTANT] = fetch_constant;

	bld_base->op_actions[TGSI_OPCODE_INTERP_CENTROID].emit = build_interp_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_INTERP_SAMPLE].emit = build_interp_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_INTERP_OFFSET].emit = build_interp_intrinsic;

	bld_base->op_actions[TGSI_OPCODE_MEMBAR].emit = membar_emit;

	bld_base->op_actions[TGSI_OPCODE_CLOCK].emit = clock_emit;

	bld_base->op_actions[TGSI_OPCODE_DDX].emit = si_llvm_emit_ddxy;
	bld_base->op_actions[TGSI_OPCODE_DDY].emit = si_llvm_emit_ddxy;
	bld_base->op_actions[TGSI_OPCODE_DDX_FINE].emit = si_llvm_emit_ddxy;
	bld_base->op_actions[TGSI_OPCODE_DDY_FINE].emit = si_llvm_emit_ddxy;

	bld_base->op_actions[TGSI_OPCODE_VOTE_ALL].emit = vote_all_emit;
	bld_base->op_actions[TGSI_OPCODE_VOTE_ANY].emit = vote_any_emit;
	bld_base->op_actions[TGSI_OPCODE_VOTE_EQ].emit = vote_eq_emit;
	bld_base->op_actions[TGSI_OPCODE_BALLOT].emit = ballot_emit;
	bld_base->op_actions[TGSI_OPCODE_READ_FIRST].intr_name = "llvm.amdgcn.readfirstlane";
	bld_base->op_actions[TGSI_OPCODE_READ_FIRST].emit = read_lane_emit;
	bld_base->op_actions[TGSI_OPCODE_READ_INVOC].intr_name = "llvm.amdgcn.readlane";
	bld_base->op_actions[TGSI_OPCODE_READ_INVOC].emit = read_lane_emit;

	bld_base->op_actions[TGSI_OPCODE_EMIT].emit = si_tgsi_emit_vertex;
	bld_base->op_actions[TGSI_OPCODE_ENDPRIM].emit = si_tgsi_emit_primitive;
	bld_base->op_actions[TGSI_OPCODE_BARRIER].emit = si_llvm_emit_barrier;
}

static void si_optimize_vs_outputs(struct si_shader_context *ctx)
{
	struct si_shader *shader = ctx->shader;
	struct tgsi_shader_info *info = &shader->selector->info;

	if ((ctx->type != PIPE_SHADER_VERTEX &&
	     ctx->type != PIPE_SHADER_TESS_EVAL) ||
	    shader->key.as_ls ||
	    shader->key.as_es)
		return;

	ac_optimize_vs_outputs(&ctx->ac,
			       ctx->main_fn,
			       shader->info.vs_output_param_offset,
			       info->num_outputs,
			       &shader->info.nr_param_exports);
}

static void si_init_exec_from_input(struct si_shader_context *ctx,
				    unsigned param, unsigned bitoffset)
{
	LLVMValueRef args[] = {
		LLVMGetParam(ctx->main_fn, param),
		LLVMConstInt(ctx->i32, bitoffset, 0),
	};
	ac_build_intrinsic(&ctx->ac,
			   "llvm.amdgcn.init.exec.from.input",
			   ctx->voidt, args, 2, AC_FUNC_ATTR_CONVERGENT);
}

static bool si_vs_needs_prolog(const struct si_shader_selector *sel,
			       const struct si_vs_prolog_bits *key)
{
	/* VGPR initialization fixup for Vega10 and Raven is always done in the
	 * VS prolog. */
	return sel->vs_needs_prolog || key->ls_vgpr_fix;
}

static bool si_compile_tgsi_main(struct si_shader_context *ctx)
{
	struct si_shader *shader = ctx->shader;
	struct si_shader_selector *sel = shader->selector;
	struct lp_build_tgsi_context *bld_base = &ctx->bld_base;

	// TODO clean all this up!
	switch (ctx->type) {
	case PIPE_SHADER_VERTEX:
		ctx->load_input = declare_input_vs;
		if (shader->key.as_ls)
			ctx->abi.emit_outputs = si_llvm_emit_ls_epilogue;
		else if (shader->key.as_es)
			ctx->abi.emit_outputs = si_llvm_emit_es_epilogue;
		else
			ctx->abi.emit_outputs = si_llvm_emit_vs_epilogue;
		bld_base->emit_epilogue = si_tgsi_emit_epilogue;
		ctx->abi.load_base_vertex = get_base_vertex;
		break;
	case PIPE_SHADER_TESS_CTRL:
		bld_base->emit_fetch_funcs[TGSI_FILE_INPUT] = fetch_input_tcs;
		ctx->abi.load_tess_varyings = si_nir_load_tcs_varyings;
		bld_base->emit_fetch_funcs[TGSI_FILE_OUTPUT] = fetch_output_tcs;
		bld_base->emit_store = store_output_tcs;
		ctx->abi.store_tcs_outputs = si_nir_store_output_tcs;
		ctx->abi.emit_outputs = si_llvm_emit_tcs_epilogue;
		ctx->abi.load_patch_vertices_in = si_load_patch_vertices_in;
		bld_base->emit_epilogue = si_tgsi_emit_epilogue;
		break;
	case PIPE_SHADER_TESS_EVAL:
		bld_base->emit_fetch_funcs[TGSI_FILE_INPUT] = fetch_input_tes;
		ctx->abi.load_tess_varyings = si_nir_load_input_tes;
		ctx->abi.load_tess_coord = si_load_tess_coord;
		ctx->abi.load_tess_level = si_load_tess_level;
		ctx->abi.load_patch_vertices_in = si_load_patch_vertices_in;
		if (shader->key.as_es)
			ctx->abi.emit_outputs = si_llvm_emit_es_epilogue;
		else
			ctx->abi.emit_outputs = si_llvm_emit_vs_epilogue;
		bld_base->emit_epilogue = si_tgsi_emit_epilogue;
		break;
	case PIPE_SHADER_GEOMETRY:
		bld_base->emit_fetch_funcs[TGSI_FILE_INPUT] = fetch_input_gs;
		ctx->abi.load_inputs = si_nir_load_input_gs;
		ctx->abi.emit_vertex = si_llvm_emit_vertex;
		ctx->abi.emit_primitive = si_llvm_emit_primitive;
		ctx->abi.emit_outputs = si_llvm_emit_gs_epilogue;
		bld_base->emit_epilogue = si_tgsi_emit_gs_epilogue;
		break;
	case PIPE_SHADER_FRAGMENT:
		ctx->load_input = declare_input_fs;
		ctx->abi.emit_outputs = si_llvm_return_fs_outputs;
		bld_base->emit_epilogue = si_tgsi_emit_epilogue;
		ctx->abi.lookup_interp_param = si_nir_lookup_interp_param;
		ctx->abi.load_sample_position = load_sample_position;
		ctx->abi.load_sample_mask_in = load_sample_mask_in;
		ctx->abi.emit_kill = si_llvm_emit_kill;
		break;
	case PIPE_SHADER_COMPUTE:
		ctx->abi.load_local_group_size = get_block_size;
		break;
	default:
		assert(!"Unsupported shader type");
		return false;
	}

	ctx->abi.load_ubo = load_ubo;
	ctx->abi.load_ssbo = load_ssbo;

	create_function(ctx);
	preload_ring_buffers(ctx);

	/* For GFX9 merged shaders:
	 * - Set EXEC for the first shader. If the prolog is present, set
	 *   EXEC there instead.
	 * - Add a barrier before the second shader.
	 * - In the second shader, reset EXEC to ~0 and wrap the main part in
	 *   an if-statement. This is required for correctness in geometry
	 *   shaders, to ensure that empty GS waves do not send GS_EMIT and
	 *   GS_CUT messages.
	 *
	 * For monolithic merged shaders, the first shader is wrapped in an
	 * if-block together with its prolog in si_build_wrapper_function.
	 */
	if (ctx->screen->info.chip_class >= GFX9) {
		if (!shader->is_monolithic &&
		    sel->info.num_instructions > 1 && /* not empty shader */
		    (shader->key.as_es || shader->key.as_ls) &&
		    (ctx->type == PIPE_SHADER_TESS_EVAL ||
		     (ctx->type == PIPE_SHADER_VERTEX &&
		      !si_vs_needs_prolog(sel, &shader->key.part.vs.prolog)))) {
			si_init_exec_from_input(ctx,
						ctx->param_merged_wave_info, 0);
		} else if (ctx->type == PIPE_SHADER_TESS_CTRL ||
			   ctx->type == PIPE_SHADER_GEOMETRY) {
			if (!shader->is_monolithic)
				ac_init_exec_full_mask(&ctx->ac);

			LLVMValueRef num_threads = si_unpack_param(ctx, ctx->param_merged_wave_info, 8, 8);
			LLVMValueRef ena =
				LLVMBuildICmp(ctx->ac.builder, LLVMIntULT,
					    ac_get_thread_id(&ctx->ac), num_threads, "");
			lp_build_if(&ctx->merged_wrap_if_state, &ctx->gallivm, ena);

			/* The barrier must execute for all shaders in a
			 * threadgroup.
			 *
			 * Execute the barrier inside the conditional block,
			 * so that empty waves can jump directly to s_endpgm,
			 * which will also signal the barrier.
			 *
			 * If the shader is TCS and the TCS epilog is present
			 * and contains a barrier, it will wait there and then
			 * reach s_endpgm.
			 */
			si_llvm_emit_barrier(NULL, bld_base, NULL);
		}
	}

	if (ctx->type == PIPE_SHADER_TESS_CTRL &&
	    sel->tcs_info.tessfactors_are_def_in_all_invocs) {
		for (unsigned i = 0; i < 6; i++) {
			ctx->invoc0_tess_factors[i] =
				ac_build_alloca_undef(&ctx->ac, ctx->i32, "");
		}
	}

	if (ctx->type == PIPE_SHADER_GEOMETRY) {
		int i;
		for (i = 0; i < 4; i++) {
			ctx->gs_next_vertex[i] =
				ac_build_alloca(&ctx->ac, ctx->i32, "");
		}
	}

	if (sel->force_correct_derivs_after_kill) {
		ctx->postponed_kill = ac_build_alloca_undef(&ctx->ac, ctx->i1, "");
		/* true = don't kill. */
		LLVMBuildStore(ctx->ac.builder, ctx->i1true,
			       ctx->postponed_kill);
	}

	if (sel->tokens) {
		if (!lp_build_tgsi_llvm(bld_base, sel->tokens)) {
			fprintf(stderr, "Failed to translate shader from TGSI to LLVM\n");
			return false;
		}
	} else {
		if (!si_nir_build_llvm(ctx, sel->nir)) {
			fprintf(stderr, "Failed to translate shader from NIR to LLVM\n");
			return false;
		}
	}

	si_llvm_build_ret(ctx, ctx->return_value);
	return true;
}

/**
 * Compute the VS prolog key, which contains all the information needed to
 * build the VS prolog function, and set shader->info bits where needed.
 *
 * \param info             Shader info of the vertex shader.
 * \param num_input_sgprs  Number of input SGPRs for the vertex shader.
 * \param prolog_key       Key of the VS prolog
 * \param shader_out       The vertex shader, or the next shader if merging LS+HS or ES+GS.
 * \param key              Output shader part key.
 */
static void si_get_vs_prolog_key(const struct tgsi_shader_info *info,
				 unsigned num_input_sgprs,
				 const struct si_vs_prolog_bits *prolog_key,
				 struct si_shader *shader_out,
				 union si_shader_part_key *key)
{
	memset(key, 0, sizeof(*key));
	key->vs_prolog.states = *prolog_key;
	key->vs_prolog.num_input_sgprs = num_input_sgprs;
	key->vs_prolog.last_input = MAX2(1, info->num_inputs) - 1;
	key->vs_prolog.as_ls = shader_out->key.as_ls;
	key->vs_prolog.as_es = shader_out->key.as_es;

	if (shader_out->selector->type == PIPE_SHADER_TESS_CTRL) {
		key->vs_prolog.as_ls = 1;
		key->vs_prolog.num_merged_next_stage_vgprs = 2;
	} else if (shader_out->selector->type == PIPE_SHADER_GEOMETRY) {
		key->vs_prolog.as_es = 1;
		key->vs_prolog.num_merged_next_stage_vgprs = 5;
	}

	/* Enable loading the InstanceID VGPR. */
	uint16_t input_mask = u_bit_consecutive(0, info->num_inputs);

	if ((key->vs_prolog.states.instance_divisor_is_one |
	     key->vs_prolog.states.instance_divisor_is_fetched) & input_mask)
		shader_out->info.uses_instanceid = true;
}

/**
 * Compute the PS prolog key, which contains all the information needed to
 * build the PS prolog function, and set related bits in shader->config.
 */
static void si_get_ps_prolog_key(struct si_shader *shader,
				 union si_shader_part_key *key,
				 bool separate_prolog)
{
	struct tgsi_shader_info *info = &shader->selector->info;

	memset(key, 0, sizeof(*key));
	key->ps_prolog.states = shader->key.part.ps.prolog;
	key->ps_prolog.colors_read = info->colors_read;
	key->ps_prolog.num_input_sgprs = shader->info.num_input_sgprs;
	key->ps_prolog.num_input_vgprs = shader->info.num_input_vgprs;
	key->ps_prolog.wqm = info->uses_derivatives &&
		(key->ps_prolog.colors_read ||
		 key->ps_prolog.states.force_persp_sample_interp ||
		 key->ps_prolog.states.force_linear_sample_interp ||
		 key->ps_prolog.states.force_persp_center_interp ||
		 key->ps_prolog.states.force_linear_center_interp ||
		 key->ps_prolog.states.bc_optimize_for_persp ||
		 key->ps_prolog.states.bc_optimize_for_linear);
	key->ps_prolog.ancillary_vgpr_index = shader->info.ancillary_vgpr_index;

	if (info->colors_read) {
		unsigned *color = shader->selector->color_attr_index;

		if (shader->key.part.ps.prolog.color_two_side) {
			/* BCOLORs are stored after the last input. */
			key->ps_prolog.num_interp_inputs = info->num_inputs;
			key->ps_prolog.face_vgpr_index = shader->info.face_vgpr_index;
			shader->config.spi_ps_input_ena |= S_0286CC_FRONT_FACE_ENA(1);
		}

		for (unsigned i = 0; i < 2; i++) {
			unsigned interp = info->input_interpolate[color[i]];
			unsigned location = info->input_interpolate_loc[color[i]];

			if (!(info->colors_read & (0xf << i*4)))
				continue;

			key->ps_prolog.color_attr_index[i] = color[i];

			if (shader->key.part.ps.prolog.flatshade_colors &&
			    interp == TGSI_INTERPOLATE_COLOR)
				interp = TGSI_INTERPOLATE_CONSTANT;

			switch (interp) {
			case TGSI_INTERPOLATE_CONSTANT:
				key->ps_prolog.color_interp_vgpr_index[i] = -1;
				break;
			case TGSI_INTERPOLATE_PERSPECTIVE:
			case TGSI_INTERPOLATE_COLOR:
				/* Force the interpolation location for colors here. */
				if (shader->key.part.ps.prolog.force_persp_sample_interp)
					location = TGSI_INTERPOLATE_LOC_SAMPLE;
				if (shader->key.part.ps.prolog.force_persp_center_interp)
					location = TGSI_INTERPOLATE_LOC_CENTER;

				switch (location) {
				case TGSI_INTERPOLATE_LOC_SAMPLE:
					key->ps_prolog.color_interp_vgpr_index[i] = 0;
					shader->config.spi_ps_input_ena |=
						S_0286CC_PERSP_SAMPLE_ENA(1);
					break;
				case TGSI_INTERPOLATE_LOC_CENTER:
					key->ps_prolog.color_interp_vgpr_index[i] = 2;
					shader->config.spi_ps_input_ena |=
						S_0286CC_PERSP_CENTER_ENA(1);
					break;
				case TGSI_INTERPOLATE_LOC_CENTROID:
					key->ps_prolog.color_interp_vgpr_index[i] = 4;
					shader->config.spi_ps_input_ena |=
						S_0286CC_PERSP_CENTROID_ENA(1);
					break;
				default:
					assert(0);
				}
				break;
			case TGSI_INTERPOLATE_LINEAR:
				/* Force the interpolation location for colors here. */
				if (shader->key.part.ps.prolog.force_linear_sample_interp)
					location = TGSI_INTERPOLATE_LOC_SAMPLE;
				if (shader->key.part.ps.prolog.force_linear_center_interp)
					location = TGSI_INTERPOLATE_LOC_CENTER;

				/* The VGPR assignment for non-monolithic shaders
				 * works because InitialPSInputAddr is set on the
				 * main shader and PERSP_PULL_MODEL is never used.
				 */
				switch (location) {
				case TGSI_INTERPOLATE_LOC_SAMPLE:
					key->ps_prolog.color_interp_vgpr_index[i] =
						separate_prolog ? 6 : 9;
					shader->config.spi_ps_input_ena |=
						S_0286CC_LINEAR_SAMPLE_ENA(1);
					break;
				case TGSI_INTERPOLATE_LOC_CENTER:
					key->ps_prolog.color_interp_vgpr_index[i] =
						separate_prolog ? 8 : 11;
					shader->config.spi_ps_input_ena |=
						S_0286CC_LINEAR_CENTER_ENA(1);
					break;
				case TGSI_INTERPOLATE_LOC_CENTROID:
					key->ps_prolog.color_interp_vgpr_index[i] =
						separate_prolog ? 10 : 13;
					shader->config.spi_ps_input_ena |=
						S_0286CC_LINEAR_CENTROID_ENA(1);
					break;
				default:
					assert(0);
				}
				break;
			default:
				assert(0);
			}
		}
	}
}

/**
 * Check whether a PS prolog is required based on the key.
 */
static bool si_need_ps_prolog(const union si_shader_part_key *key)
{
	return key->ps_prolog.colors_read ||
	       key->ps_prolog.states.force_persp_sample_interp ||
	       key->ps_prolog.states.force_linear_sample_interp ||
	       key->ps_prolog.states.force_persp_center_interp ||
	       key->ps_prolog.states.force_linear_center_interp ||
	       key->ps_prolog.states.bc_optimize_for_persp ||
	       key->ps_prolog.states.bc_optimize_for_linear ||
	       key->ps_prolog.states.poly_stipple ||
	       key->ps_prolog.states.samplemask_log_ps_iter;
}

/**
 * Compute the PS epilog key, which contains all the information needed to
 * build the PS epilog function.
 */
static void si_get_ps_epilog_key(struct si_shader *shader,
				 union si_shader_part_key *key)
{
	struct tgsi_shader_info *info = &shader->selector->info;
	memset(key, 0, sizeof(*key));
	key->ps_epilog.colors_written = info->colors_written;
	key->ps_epilog.writes_z = info->writes_z;
	key->ps_epilog.writes_stencil = info->writes_stencil;
	key->ps_epilog.writes_samplemask = info->writes_samplemask;
	key->ps_epilog.states = shader->key.part.ps.epilog;
}

/**
 * Build the GS prolog function. Rotate the input vertices for triangle strips
 * with adjacency.
 */
static void si_build_gs_prolog_function(struct si_shader_context *ctx,
					union si_shader_part_key *key)
{
	unsigned num_sgprs, num_vgprs;
	struct si_function_info fninfo;
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMTypeRef returns[48];
	LLVMValueRef func, ret;

	si_init_function_info(&fninfo);

	if (ctx->screen->info.chip_class >= GFX9) {
		if (key->gs_prolog.states.gfx9_prev_is_vs)
			num_sgprs = 8 + GFX9_VSGS_NUM_USER_SGPR;
		else
			num_sgprs = 8 + GFX9_TESGS_NUM_USER_SGPR;
		num_vgprs = 5; /* ES inputs are not needed by GS */
	} else {
		num_sgprs = GFX6_GS_NUM_USER_SGPR + 2;
		num_vgprs = 8;
	}

	for (unsigned i = 0; i < num_sgprs; ++i) {
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		returns[i] = ctx->i32;
	}

	for (unsigned i = 0; i < num_vgprs; ++i) {
		add_arg(&fninfo, ARG_VGPR, ctx->i32);
		returns[num_sgprs + i] = ctx->f32;
	}

	/* Create the function. */
	si_create_function(ctx, "gs_prolog", returns, num_sgprs + num_vgprs,
			   &fninfo, 0);
	func = ctx->main_fn;

	/* Set the full EXEC mask for the prolog, because we are only fiddling
	 * with registers here. The main shader part will set the correct EXEC
	 * mask.
	 */
	if (ctx->screen->info.chip_class >= GFX9 && !key->gs_prolog.is_monolithic)
		ac_init_exec_full_mask(&ctx->ac);

	/* Copy inputs to outputs. This should be no-op, as the registers match,
	 * but it will prevent the compiler from overwriting them unintentionally.
	 */
	ret = ctx->return_value;
	for (unsigned i = 0; i < num_sgprs; i++) {
		LLVMValueRef p = LLVMGetParam(func, i);
		ret = LLVMBuildInsertValue(builder, ret, p, i, "");
	}
	for (unsigned i = 0; i < num_vgprs; i++) {
		LLVMValueRef p = LLVMGetParam(func, num_sgprs + i);
		p = ac_to_float(&ctx->ac, p);
		ret = LLVMBuildInsertValue(builder, ret, p, num_sgprs + i, "");
	}

	if (key->gs_prolog.states.tri_strip_adj_fix) {
		/* Remap the input vertices for every other primitive. */
		const unsigned gfx6_vtx_params[6] = {
			num_sgprs,
			num_sgprs + 1,
			num_sgprs + 3,
			num_sgprs + 4,
			num_sgprs + 5,
			num_sgprs + 6
		};
		const unsigned gfx9_vtx_params[3] = {
			num_sgprs,
			num_sgprs + 1,
			num_sgprs + 4,
		};
		LLVMValueRef vtx_in[6], vtx_out[6];
		LLVMValueRef prim_id, rotate;

		if (ctx->screen->info.chip_class >= GFX9) {
			for (unsigned i = 0; i < 3; i++) {
				vtx_in[i*2] = si_unpack_param(ctx, gfx9_vtx_params[i], 0, 16);
				vtx_in[i*2+1] = si_unpack_param(ctx, gfx9_vtx_params[i], 16, 16);
			}
		} else {
			for (unsigned i = 0; i < 6; i++)
				vtx_in[i] = LLVMGetParam(func, gfx6_vtx_params[i]);
		}

		prim_id = LLVMGetParam(func, num_sgprs + 2);
		rotate = LLVMBuildTrunc(builder, prim_id, ctx->i1, "");

		for (unsigned i = 0; i < 6; ++i) {
			LLVMValueRef base, rotated;
			base = vtx_in[i];
			rotated = vtx_in[(i + 4) % 6];
			vtx_out[i] = LLVMBuildSelect(builder, rotate, rotated, base, "");
		}

		if (ctx->screen->info.chip_class >= GFX9) {
			for (unsigned i = 0; i < 3; i++) {
				LLVMValueRef hi, out;

				hi = LLVMBuildShl(builder, vtx_out[i*2+1],
						  LLVMConstInt(ctx->i32, 16, 0), "");
				out = LLVMBuildOr(builder, vtx_out[i*2], hi, "");
				out = ac_to_float(&ctx->ac, out);
				ret = LLVMBuildInsertValue(builder, ret, out,
							   gfx9_vtx_params[i], "");
			}
		} else {
			for (unsigned i = 0; i < 6; i++) {
				LLVMValueRef out;

				out = ac_to_float(&ctx->ac, vtx_out[i]);
				ret = LLVMBuildInsertValue(builder, ret, out,
							   gfx6_vtx_params[i], "");
			}
		}
	}

	LLVMBuildRet(builder, ret);
}

/**
 * Given a list of shader part functions, build a wrapper function that
 * runs them in sequence to form a monolithic shader.
 */
static void si_build_wrapper_function(struct si_shader_context *ctx,
				      LLVMValueRef *parts,
				      unsigned num_parts,
				      unsigned main_part,
				      unsigned next_shader_first_part)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	/* PS epilog has one arg per color component; gfx9 merged shader
	 * prologs need to forward 32 user SGPRs.
	 */
	struct si_function_info fninfo;
	LLVMValueRef initial[64], out[64];
	LLVMTypeRef function_type;
	unsigned num_first_params;
	unsigned num_out, initial_num_out;
	MAYBE_UNUSED unsigned num_out_sgpr; /* used in debug checks */
	MAYBE_UNUSED unsigned initial_num_out_sgpr; /* used in debug checks */
	unsigned num_sgprs, num_vgprs;
	unsigned gprs;
	struct lp_build_if_state if_state;

	si_init_function_info(&fninfo);

	for (unsigned i = 0; i < num_parts; ++i) {
		ac_add_function_attr(ctx->ac.context, parts[i], -1,
				     AC_FUNC_ATTR_ALWAYSINLINE);
		LLVMSetLinkage(parts[i], LLVMPrivateLinkage);
	}

	/* The parameters of the wrapper function correspond to those of the
	 * first part in terms of SGPRs and VGPRs, but we use the types of the
	 * main part to get the right types. This is relevant for the
	 * dereferenceable attribute on descriptor table pointers.
	 */
	num_sgprs = 0;
	num_vgprs = 0;

	function_type = LLVMGetElementType(LLVMTypeOf(parts[0]));
	num_first_params = LLVMCountParamTypes(function_type);

	for (unsigned i = 0; i < num_first_params; ++i) {
		LLVMValueRef param = LLVMGetParam(parts[0], i);

		if (ac_is_sgpr_param(param)) {
			assert(num_vgprs == 0);
			num_sgprs += ac_get_type_size(LLVMTypeOf(param)) / 4;
		} else {
			num_vgprs += ac_get_type_size(LLVMTypeOf(param)) / 4;
		}
	}

	gprs = 0;
	while (gprs < num_sgprs + num_vgprs) {
		LLVMValueRef param = LLVMGetParam(parts[main_part], fninfo.num_params);
		LLVMTypeRef type = LLVMTypeOf(param);
		unsigned size = ac_get_type_size(type) / 4;

		add_arg(&fninfo, gprs < num_sgprs ? ARG_SGPR : ARG_VGPR, type);

		assert(ac_is_sgpr_param(param) == (gprs < num_sgprs));
		assert(gprs + size <= num_sgprs + num_vgprs &&
		       (gprs >= num_sgprs || gprs + size <= num_sgprs));

		gprs += size;
	}

	si_create_function(ctx, "wrapper", NULL, 0, &fninfo,
			   si_get_max_workgroup_size(ctx->shader));

	if (is_merged_shader(ctx))
		ac_init_exec_full_mask(&ctx->ac);

	/* Record the arguments of the function as if they were an output of
	 * a previous part.
	 */
	num_out = 0;
	num_out_sgpr = 0;

	for (unsigned i = 0; i < fninfo.num_params; ++i) {
		LLVMValueRef param = LLVMGetParam(ctx->main_fn, i);
		LLVMTypeRef param_type = LLVMTypeOf(param);
		LLVMTypeRef out_type = i < fninfo.num_sgpr_params ? ctx->i32 : ctx->f32;
		unsigned size = ac_get_type_size(param_type) / 4;

		if (size == 1) {
			if (LLVMGetTypeKind(param_type) == LLVMPointerTypeKind) {
				param = LLVMBuildPtrToInt(builder, param, ctx->i32, "");
				param_type = ctx->i32;
			}

			if (param_type != out_type)
				param = LLVMBuildBitCast(builder, param, out_type, "");
			out[num_out++] = param;
		} else {
			LLVMTypeRef vector_type = LLVMVectorType(out_type, size);

			if (LLVMGetTypeKind(param_type) == LLVMPointerTypeKind) {
				param = LLVMBuildPtrToInt(builder, param, ctx->i64, "");
				param_type = ctx->i64;
			}

			if (param_type != vector_type)
				param = LLVMBuildBitCast(builder, param, vector_type, "");

			for (unsigned j = 0; j < size; ++j)
				out[num_out++] = LLVMBuildExtractElement(
					builder, param, LLVMConstInt(ctx->i32, j, 0), "");
		}

		if (i < fninfo.num_sgpr_params)
			num_out_sgpr = num_out;
	}

	memcpy(initial, out, sizeof(out));
	initial_num_out = num_out;
	initial_num_out_sgpr = num_out_sgpr;

	/* Now chain the parts. */
	for (unsigned part = 0; part < num_parts; ++part) {
		LLVMValueRef in[48];
		LLVMValueRef ret;
		LLVMTypeRef ret_type;
		unsigned out_idx = 0;
		unsigned num_params = LLVMCountParams(parts[part]);

		/* Merged shaders are executed conditionally depending
		 * on the number of enabled threads passed in the input SGPRs. */
		if (is_merged_shader(ctx) && part == 0) {
			LLVMValueRef ena, count = initial[3];

			count = LLVMBuildAnd(builder, count,
					     LLVMConstInt(ctx->i32, 0x7f, 0), "");
			ena = LLVMBuildICmp(builder, LLVMIntULT,
					    ac_get_thread_id(&ctx->ac), count, "");
			lp_build_if(&if_state, &ctx->gallivm, ena);
		}

		/* Derive arguments for the next part from outputs of the
		 * previous one.
		 */
		for (unsigned param_idx = 0; param_idx < num_params; ++param_idx) {
			LLVMValueRef param;
			LLVMTypeRef param_type;
			bool is_sgpr;
			unsigned param_size;
			LLVMValueRef arg = NULL;

			param = LLVMGetParam(parts[part], param_idx);
			param_type = LLVMTypeOf(param);
			param_size = ac_get_type_size(param_type) / 4;
			is_sgpr = ac_is_sgpr_param(param);

			if (is_sgpr) {
				ac_add_function_attr(ctx->ac.context, parts[part],
						     param_idx + 1, AC_FUNC_ATTR_INREG);
			} else if (out_idx < num_out_sgpr) {
				/* Skip returned SGPRs the current part doesn't
				 * declare on the input. */
				out_idx = num_out_sgpr;
			}

			assert(out_idx + param_size <= (is_sgpr ? num_out_sgpr : num_out));

			if (param_size == 1)
				arg = out[out_idx];
			else
				arg = ac_build_gather_values(&ctx->ac, &out[out_idx], param_size);

			if (LLVMTypeOf(arg) != param_type) {
				if (LLVMGetTypeKind(param_type) == LLVMPointerTypeKind) {
					if (LLVMGetPointerAddressSpace(param_type) ==
					    AC_ADDR_SPACE_CONST_32BIT) {
						arg = LLVMBuildBitCast(builder, arg, ctx->i32, "");
						arg = LLVMBuildIntToPtr(builder, arg, param_type, "");
					} else {
						arg = LLVMBuildBitCast(builder, arg, ctx->i64, "");
						arg = LLVMBuildIntToPtr(builder, arg, param_type, "");
					}
				} else {
					arg = LLVMBuildBitCast(builder, arg, param_type, "");
				}
			}

			in[param_idx] = arg;
			out_idx += param_size;
		}

		ret = LLVMBuildCall(builder, parts[part], in, num_params, "");

		if (is_merged_shader(ctx) &&
		    part + 1 == next_shader_first_part) {
			lp_build_endif(&if_state);

			/* The second half of the merged shader should use
			 * the inputs from the toplevel (wrapper) function,
			 * not the return value from the last call.
			 *
			 * That's because the last call was executed condi-
			 * tionally, so we can't consume it in the main
			 * block.
			 */
			memcpy(out, initial, sizeof(initial));
			num_out = initial_num_out;
			num_out_sgpr = initial_num_out_sgpr;
			continue;
		}

		/* Extract the returned GPRs. */
		ret_type = LLVMTypeOf(ret);
		num_out = 0;
		num_out_sgpr = 0;

		if (LLVMGetTypeKind(ret_type) != LLVMVoidTypeKind) {
			assert(LLVMGetTypeKind(ret_type) == LLVMStructTypeKind);

			unsigned ret_size = LLVMCountStructElementTypes(ret_type);

			for (unsigned i = 0; i < ret_size; ++i) {
				LLVMValueRef val =
					LLVMBuildExtractValue(builder, ret, i, "");

				assert(num_out < ARRAY_SIZE(out));
				out[num_out++] = val;

				if (LLVMTypeOf(val) == ctx->i32) {
					assert(num_out_sgpr + 1 == num_out);
					num_out_sgpr = num_out;
				}
			}
		}
	}

	LLVMBuildRetVoid(builder);
}

static bool si_should_optimize_less(struct ac_llvm_compiler *compiler,
				    struct si_shader_selector *sel)
{
	if (!compiler->low_opt_passes)
		return false;

	/* Assume a slow CPU. */
	assert(!sel->screen->info.has_dedicated_vram &&
	       sel->screen->info.chip_class <= VI);

	/* For a crazy dEQP test containing 2597 memory opcodes, mostly
	 * buffer stores. */
	return sel->type == PIPE_SHADER_COMPUTE &&
	       sel->info.num_memory_instructions > 1000;
}

int si_compile_tgsi_shader(struct si_screen *sscreen,
			   struct ac_llvm_compiler *compiler,
			   struct si_shader *shader,
			   struct pipe_debug_callback *debug)
{
	struct si_shader_selector *sel = shader->selector;
	struct si_shader_context ctx;
	int r = -1;

	/* Dump TGSI code before doing TGSI->LLVM conversion in case the
	 * conversion fails. */
	if (si_can_dump_shader(sscreen, sel->info.processor) &&
	    !(sscreen->debug_flags & DBG(NO_TGSI))) {
		if (sel->tokens)
			tgsi_dump(sel->tokens, 0);
		else
			nir_print_shader(sel->nir, stderr);
		si_dump_streamout(&sel->so);
	}

	si_init_shader_ctx(&ctx, sscreen, compiler);
	si_llvm_context_set_tgsi(&ctx, shader);

	memset(shader->info.vs_output_param_offset, AC_EXP_PARAM_UNDEFINED,
	       sizeof(shader->info.vs_output_param_offset));

	shader->info.uses_instanceid = sel->info.uses_instanceid;

	if (!si_compile_tgsi_main(&ctx)) {
		si_llvm_dispose(&ctx);
		return -1;
	}

	if (shader->is_monolithic && ctx.type == PIPE_SHADER_VERTEX) {
		LLVMValueRef parts[2];
		bool need_prolog = sel->vs_needs_prolog;

		parts[1] = ctx.main_fn;

		if (need_prolog) {
			union si_shader_part_key prolog_key;
			si_get_vs_prolog_key(&sel->info,
					     shader->info.num_input_sgprs,
					     &shader->key.part.vs.prolog,
					     shader, &prolog_key);
			si_build_vs_prolog_function(&ctx, &prolog_key);
			parts[0] = ctx.main_fn;
		}

		si_build_wrapper_function(&ctx, parts + !need_prolog,
					  1 + need_prolog, need_prolog, 0);
	} else if (shader->is_monolithic && ctx.type == PIPE_SHADER_TESS_CTRL) {
		if (sscreen->info.chip_class >= GFX9) {
			struct si_shader_selector *ls = shader->key.part.tcs.ls;
			LLVMValueRef parts[4];
			bool vs_needs_prolog =
				si_vs_needs_prolog(ls, &shader->key.part.tcs.ls_prolog);

			/* TCS main part */
			parts[2] = ctx.main_fn;

			/* TCS epilog */
			union si_shader_part_key tcs_epilog_key;
			memset(&tcs_epilog_key, 0, sizeof(tcs_epilog_key));
			tcs_epilog_key.tcs_epilog.states = shader->key.part.tcs.epilog;
			si_build_tcs_epilog_function(&ctx, &tcs_epilog_key);
			parts[3] = ctx.main_fn;

			/* VS as LS main part */
			struct si_shader shader_ls = {};
			shader_ls.selector = ls;
			shader_ls.key.as_ls = 1;
			shader_ls.key.mono = shader->key.mono;
			shader_ls.key.opt = shader->key.opt;
			shader_ls.is_monolithic = true;
			si_llvm_context_set_tgsi(&ctx, &shader_ls);

			if (!si_compile_tgsi_main(&ctx)) {
				si_llvm_dispose(&ctx);
				return -1;
			}
			shader->info.uses_instanceid |= ls->info.uses_instanceid;
			parts[1] = ctx.main_fn;

			/* LS prolog */
			if (vs_needs_prolog) {
				union si_shader_part_key vs_prolog_key;
				si_get_vs_prolog_key(&ls->info,
						     shader_ls.info.num_input_sgprs,
						     &shader->key.part.tcs.ls_prolog,
						     shader, &vs_prolog_key);
				vs_prolog_key.vs_prolog.is_monolithic = true;
				si_build_vs_prolog_function(&ctx, &vs_prolog_key);
				parts[0] = ctx.main_fn;
			}

			/* Reset the shader context. */
			ctx.shader = shader;
			ctx.type = PIPE_SHADER_TESS_CTRL;

			si_build_wrapper_function(&ctx,
						  parts + !vs_needs_prolog,
						  4 - !vs_needs_prolog, vs_needs_prolog,
						  vs_needs_prolog ? 2 : 1);
		} else {
			LLVMValueRef parts[2];
			union si_shader_part_key epilog_key;

			parts[0] = ctx.main_fn;

			memset(&epilog_key, 0, sizeof(epilog_key));
			epilog_key.tcs_epilog.states = shader->key.part.tcs.epilog;
			si_build_tcs_epilog_function(&ctx, &epilog_key);
			parts[1] = ctx.main_fn;

			si_build_wrapper_function(&ctx, parts, 2, 0, 0);
		}
	} else if (shader->is_monolithic && ctx.type == PIPE_SHADER_GEOMETRY) {
		if (ctx.screen->info.chip_class >= GFX9) {
			struct si_shader_selector *es = shader->key.part.gs.es;
			LLVMValueRef es_prolog = NULL;
			LLVMValueRef es_main = NULL;
			LLVMValueRef gs_prolog = NULL;
			LLVMValueRef gs_main = ctx.main_fn;

			/* GS prolog */
			union si_shader_part_key gs_prolog_key;
			memset(&gs_prolog_key, 0, sizeof(gs_prolog_key));
			gs_prolog_key.gs_prolog.states = shader->key.part.gs.prolog;
			gs_prolog_key.gs_prolog.is_monolithic = true;
			si_build_gs_prolog_function(&ctx, &gs_prolog_key);
			gs_prolog = ctx.main_fn;

			/* ES main part */
			struct si_shader shader_es = {};
			shader_es.selector = es;
			shader_es.key.as_es = 1;
			shader_es.key.mono = shader->key.mono;
			shader_es.key.opt = shader->key.opt;
			shader_es.is_monolithic = true;
			si_llvm_context_set_tgsi(&ctx, &shader_es);

			if (!si_compile_tgsi_main(&ctx)) {
				si_llvm_dispose(&ctx);
				return -1;
			}
			shader->info.uses_instanceid |= es->info.uses_instanceid;
			es_main = ctx.main_fn;

			/* ES prolog */
			if (es->vs_needs_prolog) {
				union si_shader_part_key vs_prolog_key;
				si_get_vs_prolog_key(&es->info,
						     shader_es.info.num_input_sgprs,
						     &shader->key.part.gs.vs_prolog,
						     shader, &vs_prolog_key);
				vs_prolog_key.vs_prolog.is_monolithic = true;
				si_build_vs_prolog_function(&ctx, &vs_prolog_key);
				es_prolog = ctx.main_fn;
			}

			/* Reset the shader context. */
			ctx.shader = shader;
			ctx.type = PIPE_SHADER_GEOMETRY;

			/* Prepare the array of shader parts. */
			LLVMValueRef parts[4];
			unsigned num_parts = 0, main_part, next_first_part;

			if (es_prolog)
				parts[num_parts++] = es_prolog;

			parts[main_part = num_parts++] = es_main;
			parts[next_first_part = num_parts++] = gs_prolog;
			parts[num_parts++] = gs_main;

			si_build_wrapper_function(&ctx, parts, num_parts,
						  main_part, next_first_part);
		} else {
			LLVMValueRef parts[2];
			union si_shader_part_key prolog_key;

			parts[1] = ctx.main_fn;

			memset(&prolog_key, 0, sizeof(prolog_key));
			prolog_key.gs_prolog.states = shader->key.part.gs.prolog;
			si_build_gs_prolog_function(&ctx, &prolog_key);
			parts[0] = ctx.main_fn;

			si_build_wrapper_function(&ctx, parts, 2, 1, 0);
		}
	} else if (shader->is_monolithic && ctx.type == PIPE_SHADER_FRAGMENT) {
		LLVMValueRef parts[3];
		union si_shader_part_key prolog_key;
		union si_shader_part_key epilog_key;
		bool need_prolog;

		si_get_ps_prolog_key(shader, &prolog_key, false);
		need_prolog = si_need_ps_prolog(&prolog_key);

		parts[need_prolog ? 1 : 0] = ctx.main_fn;

		if (need_prolog) {
			si_build_ps_prolog_function(&ctx, &prolog_key);
			parts[0] = ctx.main_fn;
		}

		si_get_ps_epilog_key(shader, &epilog_key);
		si_build_ps_epilog_function(&ctx, &epilog_key);
		parts[need_prolog ? 2 : 1] = ctx.main_fn;

		si_build_wrapper_function(&ctx, parts, need_prolog ? 3 : 2,
					  need_prolog ? 1 : 0, 0);
	}

	si_llvm_optimize_module(&ctx);

	/* Post-optimization transformations and analysis. */
	si_optimize_vs_outputs(&ctx);

	if ((debug && debug->debug_message) ||
	    si_can_dump_shader(sscreen, ctx.type)) {
		ctx.shader->config.private_mem_vgprs =
			ac_count_scratch_private_memory(ctx.main_fn);
	}

	/* Make sure the input is a pointer and not integer followed by inttoptr. */
	assert(LLVMGetTypeKind(LLVMTypeOf(LLVMGetParam(ctx.main_fn, 0))) ==
	       LLVMPointerTypeKind);

	/* Compile to bytecode. */
	r = si_compile_llvm(sscreen, &shader->binary, &shader->config, compiler,
			    ctx.ac.module, debug, ctx.type,
			    si_get_shader_name(shader, ctx.type),
			    si_should_optimize_less(compiler, shader->selector));
	si_llvm_dispose(&ctx);
	if (r) {
		fprintf(stderr, "LLVM failed to compile shader\n");
		return r;
	}

	/* Validate SGPR and VGPR usage for compute to detect compiler bugs.
	 * LLVM 3.9svn has this bug.
	 */
	if (sel->type == PIPE_SHADER_COMPUTE) {
		unsigned wave_size = 64;
		unsigned max_vgprs = 256;
		unsigned max_sgprs = sscreen->info.chip_class >= VI ? 800 : 512;
		unsigned max_sgprs_per_wave = 128;
		unsigned max_block_threads = si_get_max_workgroup_size(shader);
		unsigned min_waves_per_cu = DIV_ROUND_UP(max_block_threads, wave_size);
		unsigned min_waves_per_simd = DIV_ROUND_UP(min_waves_per_cu, 4);

		max_vgprs = max_vgprs / min_waves_per_simd;
		max_sgprs = MIN2(max_sgprs / min_waves_per_simd, max_sgprs_per_wave);

		if (shader->config.num_sgprs > max_sgprs ||
		    shader->config.num_vgprs > max_vgprs) {
			fprintf(stderr, "LLVM failed to compile a shader correctly: "
				"SGPR:VGPR usage is %u:%u, but the hw limit is %u:%u\n",
				shader->config.num_sgprs, shader->config.num_vgprs,
				max_sgprs, max_vgprs);

			/* Just terminate the process, because dependent
			 * shaders can hang due to bad input data, but use
			 * the env var to allow shader-db to work.
			 */
			if (!debug_get_bool_option("SI_PASS_BAD_SHADERS", false))
				abort();
		}
	}

	/* Add the scratch offset to input SGPRs. */
	if (shader->config.scratch_bytes_per_wave && !is_merged_shader(&ctx))
		shader->info.num_input_sgprs += 1; /* scratch byte offset */

	/* Calculate the number of fragment input VGPRs. */
	if (ctx.type == PIPE_SHADER_FRAGMENT) {
		shader->info.num_input_vgprs = 0;
		shader->info.face_vgpr_index = -1;
		shader->info.ancillary_vgpr_index = -1;

		if (G_0286CC_PERSP_SAMPLE_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 2;
		if (G_0286CC_PERSP_CENTER_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 2;
		if (G_0286CC_PERSP_CENTROID_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 2;
		if (G_0286CC_PERSP_PULL_MODEL_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 3;
		if (G_0286CC_LINEAR_SAMPLE_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 2;
		if (G_0286CC_LINEAR_CENTER_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 2;
		if (G_0286CC_LINEAR_CENTROID_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 2;
		if (G_0286CC_LINE_STIPPLE_TEX_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 1;
		if (G_0286CC_POS_X_FLOAT_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 1;
		if (G_0286CC_POS_Y_FLOAT_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 1;
		if (G_0286CC_POS_Z_FLOAT_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 1;
		if (G_0286CC_POS_W_FLOAT_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 1;
		if (G_0286CC_FRONT_FACE_ENA(shader->config.spi_ps_input_addr)) {
			shader->info.face_vgpr_index = shader->info.num_input_vgprs;
			shader->info.num_input_vgprs += 1;
		}
		if (G_0286CC_ANCILLARY_ENA(shader->config.spi_ps_input_addr)) {
			shader->info.ancillary_vgpr_index = shader->info.num_input_vgprs;
			shader->info.num_input_vgprs += 1;
		}
		if (G_0286CC_SAMPLE_COVERAGE_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 1;
		if (G_0286CC_POS_FIXED_PT_ENA(shader->config.spi_ps_input_addr))
			shader->info.num_input_vgprs += 1;
	}

	si_calculate_max_simd_waves(shader);
	si_shader_dump_stats_for_shader_db(shader, debug);
	return 0;
}

/**
 * Create, compile and return a shader part (prolog or epilog).
 *
 * \param sscreen	screen
 * \param list		list of shader parts of the same category
 * \param type		shader type
 * \param key		shader part key
 * \param prolog	whether the part being requested is a prolog
 * \param tm		LLVM target machine
 * \param debug		debug callback
 * \param build		the callback responsible for building the main function
 * \return		non-NULL on success
 */
static struct si_shader_part *
si_get_shader_part(struct si_screen *sscreen,
		   struct si_shader_part **list,
		   enum pipe_shader_type type,
		   bool prolog,
		   union si_shader_part_key *key,
		   struct ac_llvm_compiler *compiler,
		   struct pipe_debug_callback *debug,
		   void (*build)(struct si_shader_context *,
				 union si_shader_part_key *),
		   const char *name)
{
	struct si_shader_part *result;

	mtx_lock(&sscreen->shader_parts_mutex);

	/* Find existing. */
	for (result = *list; result; result = result->next) {
		if (memcmp(&result->key, key, sizeof(*key)) == 0) {
			mtx_unlock(&sscreen->shader_parts_mutex);
			return result;
		}
	}

	/* Compile a new one. */
	result = CALLOC_STRUCT(si_shader_part);
	result->key = *key;

	struct si_shader shader = {};
	struct si_shader_context ctx;

	si_init_shader_ctx(&ctx, sscreen, compiler);
	ctx.shader = &shader;
	ctx.type = type;

	switch (type) {
	case PIPE_SHADER_VERTEX:
		shader.key.as_ls = key->vs_prolog.as_ls;
		shader.key.as_es = key->vs_prolog.as_es;
		break;
	case PIPE_SHADER_TESS_CTRL:
		assert(!prolog);
		shader.key.part.tcs.epilog = key->tcs_epilog.states;
		break;
	case PIPE_SHADER_GEOMETRY:
		assert(prolog);
		break;
	case PIPE_SHADER_FRAGMENT:
		if (prolog)
			shader.key.part.ps.prolog = key->ps_prolog.states;
		else
			shader.key.part.ps.epilog = key->ps_epilog.states;
		break;
	default:
		unreachable("bad shader part");
	}

	build(&ctx, key);

	/* Compile. */
	si_llvm_optimize_module(&ctx);

	if (si_compile_llvm(sscreen, &result->binary, &result->config, compiler,
			    ctx.ac.module, debug, ctx.type, name, false)) {
		FREE(result);
		result = NULL;
		goto out;
	}

	result->next = *list;
	*list = result;

out:
	si_llvm_dispose(&ctx);
	mtx_unlock(&sscreen->shader_parts_mutex);
	return result;
}

static LLVMValueRef si_prolog_get_rw_buffers(struct si_shader_context *ctx)
{
	LLVMValueRef ptr[2], list;
	bool merged_shader = is_merged_shader(ctx);

	if (HAVE_32BIT_POINTERS) {
		ptr[0] = LLVMGetParam(ctx->main_fn, (merged_shader ? 8 : 0) + SI_SGPR_RW_BUFFERS);
		list = LLVMBuildIntToPtr(ctx->ac.builder, ptr[0],
					 ac_array_in_const32_addr_space(ctx->v4i32), "");
		return list;
	}

	/* Get the pointer to rw buffers. */
	ptr[0] = LLVMGetParam(ctx->main_fn, (merged_shader ? 8 : 0) + SI_SGPR_RW_BUFFERS);
	ptr[1] = LLVMGetParam(ctx->main_fn, (merged_shader ? 8 : 0) + SI_SGPR_RW_BUFFERS + 1);
	list = ac_build_gather_values(&ctx->ac, ptr, 2);
	list = LLVMBuildBitCast(ctx->ac.builder, list, ctx->i64, "");
	list = LLVMBuildIntToPtr(ctx->ac.builder, list,
				 ac_array_in_const_addr_space(ctx->v4i32), "");
	return list;
}

/**
 * Build the vertex shader prolog function.
 *
 * The inputs are the same as VS (a lot of SGPRs and 4 VGPR system values).
 * All inputs are returned unmodified. The vertex load indices are
 * stored after them, which will be used by the API VS for fetching inputs.
 *
 * For example, the expected outputs for instance_divisors[] = {0, 1, 2} are:
 *   input_v0,
 *   input_v1,
 *   input_v2,
 *   input_v3,
 *   (VertexID + BaseVertex),
 *   (InstanceID + StartInstance),
 *   (InstanceID / 2 + StartInstance)
 */
static void si_build_vs_prolog_function(struct si_shader_context *ctx,
					union si_shader_part_key *key)
{
	struct si_function_info fninfo;
	LLVMTypeRef *returns;
	LLVMValueRef ret, func;
	int num_returns, i;
	unsigned first_vs_vgpr = key->vs_prolog.num_merged_next_stage_vgprs;
	unsigned num_input_vgprs = key->vs_prolog.num_merged_next_stage_vgprs + 4;
	LLVMValueRef input_vgprs[9];
	unsigned num_all_input_regs = key->vs_prolog.num_input_sgprs +
				      num_input_vgprs;
	unsigned user_sgpr_base = key->vs_prolog.num_merged_next_stage_vgprs ? 8 : 0;

	si_init_function_info(&fninfo);

	/* 4 preloaded VGPRs + vertex load indices as prolog outputs */
	returns = alloca((num_all_input_regs + key->vs_prolog.last_input + 1) *
			 sizeof(LLVMTypeRef));
	num_returns = 0;

	/* Declare input and output SGPRs. */
	for (i = 0; i < key->vs_prolog.num_input_sgprs; i++) {
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		returns[num_returns++] = ctx->i32;
	}

	/* Preloaded VGPRs (outputs must be floats) */
	for (i = 0; i < num_input_vgprs; i++) {
		add_arg_assign(&fninfo, ARG_VGPR, ctx->i32, &input_vgprs[i]);
		returns[num_returns++] = ctx->f32;
	}

	/* Vertex load indices. */
	for (i = 0; i <= key->vs_prolog.last_input; i++)
		returns[num_returns++] = ctx->f32;

	/* Create the function. */
	si_create_function(ctx, "vs_prolog", returns, num_returns, &fninfo, 0);
	func = ctx->main_fn;

	if (key->vs_prolog.num_merged_next_stage_vgprs) {
		if (!key->vs_prolog.is_monolithic)
			si_init_exec_from_input(ctx, 3, 0);

		if (key->vs_prolog.as_ls &&
		    ctx->screen->has_ls_vgpr_init_bug) {
			/* If there are no HS threads, SPI loads the LS VGPRs
			 * starting at VGPR 0. Shift them back to where they
			 * belong.
			 */
			LLVMValueRef has_hs_threads =
				LLVMBuildICmp(ctx->ac.builder, LLVMIntNE,
				    si_unpack_param(ctx, 3, 8, 8),
				    ctx->i32_0, "");

			for (i = 4; i > 0; --i) {
				input_vgprs[i + 1] =
					LLVMBuildSelect(ctx->ac.builder, has_hs_threads,
						        input_vgprs[i + 1],
						        input_vgprs[i - 1], "");
			}
		}
	}

	ctx->abi.vertex_id = input_vgprs[first_vs_vgpr];
	ctx->abi.instance_id = input_vgprs[first_vs_vgpr + (key->vs_prolog.as_ls ? 2 : 1)];

	/* Copy inputs to outputs. This should be no-op, as the registers match,
	 * but it will prevent the compiler from overwriting them unintentionally.
	 */
	ret = ctx->return_value;
	for (i = 0; i < key->vs_prolog.num_input_sgprs; i++) {
		LLVMValueRef p = LLVMGetParam(func, i);
		ret = LLVMBuildInsertValue(ctx->ac.builder, ret, p, i, "");
	}
	for (i = 0; i < num_input_vgprs; i++) {
		LLVMValueRef p = input_vgprs[i];
		p = ac_to_float(&ctx->ac, p);
		ret = LLVMBuildInsertValue(ctx->ac.builder, ret, p,
					   key->vs_prolog.num_input_sgprs + i, "");
	}

	/* Compute vertex load indices from instance divisors. */
	LLVMValueRef instance_divisor_constbuf = NULL;

	if (key->vs_prolog.states.instance_divisor_is_fetched) {
		LLVMValueRef list = si_prolog_get_rw_buffers(ctx);
		LLVMValueRef buf_index =
			LLVMConstInt(ctx->i32, SI_VS_CONST_INSTANCE_DIVISORS, 0);
		instance_divisor_constbuf =
			ac_build_load_to_sgpr(&ctx->ac, list, buf_index);
	}

	for (i = 0; i <= key->vs_prolog.last_input; i++) {
		bool divisor_is_one =
			key->vs_prolog.states.instance_divisor_is_one & (1u << i);
		bool divisor_is_fetched =
			key->vs_prolog.states.instance_divisor_is_fetched & (1u << i);
		LLVMValueRef index = NULL;

		if (divisor_is_one) {
			index = ctx->abi.instance_id;
		} else if (divisor_is_fetched) {
			LLVMValueRef udiv_factors[4];

			for (unsigned j = 0; j < 4; j++) {
				udiv_factors[j] =
					buffer_load_const(ctx, instance_divisor_constbuf,
							  LLVMConstInt(ctx->i32, i*16 + j*4, 0));
				udiv_factors[j] = ac_to_integer(&ctx->ac, udiv_factors[j]);
			}
			/* The faster NUW version doesn't work when InstanceID == UINT_MAX.
			 * Such InstanceID might not be achievable in a reasonable time though.
			 */
			index = ac_build_fast_udiv_nuw(&ctx->ac, ctx->abi.instance_id,
						       udiv_factors[0], udiv_factors[1],
						       udiv_factors[2], udiv_factors[3]);
		}

		if (divisor_is_one || divisor_is_fetched) {
			/* Add StartInstance. */
			index = LLVMBuildAdd(ctx->ac.builder, index,
					     LLVMGetParam(ctx->main_fn, user_sgpr_base +
							  SI_SGPR_START_INSTANCE), "");
		} else {
			/* VertexID + BaseVertex */
			index = LLVMBuildAdd(ctx->ac.builder,
					     ctx->abi.vertex_id,
					     LLVMGetParam(func, user_sgpr_base +
								SI_SGPR_BASE_VERTEX), "");
		}

		index = ac_to_float(&ctx->ac, index);
		ret = LLVMBuildInsertValue(ctx->ac.builder, ret, index,
					   fninfo.num_params + i, "");
	}

	si_llvm_build_ret(ctx, ret);
}

static bool si_get_vs_prolog(struct si_screen *sscreen,
			     struct ac_llvm_compiler *compiler,
			     struct si_shader *shader,
			     struct pipe_debug_callback *debug,
			     struct si_shader *main_part,
			     const struct si_vs_prolog_bits *key)
{
	struct si_shader_selector *vs = main_part->selector;

	if (!si_vs_needs_prolog(vs, key))
		return true;

	/* Get the prolog. */
	union si_shader_part_key prolog_key;
	si_get_vs_prolog_key(&vs->info, main_part->info.num_input_sgprs,
			     key, shader, &prolog_key);

	shader->prolog =
		si_get_shader_part(sscreen, &sscreen->vs_prologs,
				   PIPE_SHADER_VERTEX, true, &prolog_key, compiler,
				   debug, si_build_vs_prolog_function,
				   "Vertex Shader Prolog");
	return shader->prolog != NULL;
}

/**
 * Select and compile (or reuse) vertex shader parts (prolog & epilog).
 */
static bool si_shader_select_vs_parts(struct si_screen *sscreen,
				      struct ac_llvm_compiler *compiler,
				      struct si_shader *shader,
				      struct pipe_debug_callback *debug)
{
	return si_get_vs_prolog(sscreen, compiler, shader, debug, shader,
				&shader->key.part.vs.prolog);
}

/**
 * Compile the TCS epilog function. This writes tesselation factors to memory
 * based on the output primitive type of the tesselator (determined by TES).
 */
static void si_build_tcs_epilog_function(struct si_shader_context *ctx,
					 union si_shader_part_key *key)
{
	struct lp_build_tgsi_context *bld_base = &ctx->bld_base;
	struct si_function_info fninfo;
	LLVMValueRef func;

	si_init_function_info(&fninfo);

	if (ctx->screen->info.chip_class >= GFX9) {
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_offchip_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32); /* wave info */
		ctx->param_tcs_factor_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
		add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
		add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
		add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		if (!HAVE_32BIT_POINTERS)
			add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
		ctx->param_tcs_offchip_layout = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_out_lds_layout = add_arg(&fninfo, ARG_SGPR, ctx->i32);
	} else {
		add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
		add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
		add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
		add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
		ctx->param_tcs_offchip_layout = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_out_lds_layout = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_offchip_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
		ctx->param_tcs_factor_offset = add_arg(&fninfo, ARG_SGPR, ctx->i32);
	}

	add_arg(&fninfo, ARG_VGPR, ctx->i32); /* VGPR gap */
	add_arg(&fninfo, ARG_VGPR, ctx->i32); /* VGPR gap */
	unsigned tess_factors_idx =
		add_arg(&fninfo, ARG_VGPR, ctx->i32); /* patch index within the wave (REL_PATCH_ID) */
	add_arg(&fninfo, ARG_VGPR, ctx->i32); /* invocation ID within the patch */
	add_arg(&fninfo, ARG_VGPR, ctx->i32); /* LDS offset where tess factors should be loaded from */

	for (unsigned i = 0; i < 6; i++)
		add_arg(&fninfo, ARG_VGPR, ctx->i32); /* tess factors */

	/* Create the function. */
	si_create_function(ctx, "tcs_epilog", NULL, 0, &fninfo,
			   ctx->screen->info.chip_class >= CIK ? 128 : 64);
	ac_declare_lds_as_pointer(&ctx->ac);
	func = ctx->main_fn;

	LLVMValueRef invoc0_tess_factors[6];
	for (unsigned i = 0; i < 6; i++)
		invoc0_tess_factors[i] = LLVMGetParam(func, tess_factors_idx + 3 + i);

	si_write_tess_factors(bld_base,
			      LLVMGetParam(func, tess_factors_idx),
			      LLVMGetParam(func, tess_factors_idx + 1),
			      LLVMGetParam(func, tess_factors_idx + 2),
			      invoc0_tess_factors, invoc0_tess_factors + 4);

	LLVMBuildRetVoid(ctx->ac.builder);
}

/**
 * Select and compile (or reuse) TCS parts (epilog).
 */
static bool si_shader_select_tcs_parts(struct si_screen *sscreen,
				       struct ac_llvm_compiler *compiler,
				       struct si_shader *shader,
				       struct pipe_debug_callback *debug)
{
	if (sscreen->info.chip_class >= GFX9) {
		struct si_shader *ls_main_part =
			shader->key.part.tcs.ls->main_shader_part_ls;

		if (!si_get_vs_prolog(sscreen, compiler, shader, debug, ls_main_part,
				      &shader->key.part.tcs.ls_prolog))
			return false;

		shader->previous_stage = ls_main_part;
	}

	/* Get the epilog. */
	union si_shader_part_key epilog_key;
	memset(&epilog_key, 0, sizeof(epilog_key));
	epilog_key.tcs_epilog.states = shader->key.part.tcs.epilog;

	shader->epilog = si_get_shader_part(sscreen, &sscreen->tcs_epilogs,
					    PIPE_SHADER_TESS_CTRL, false,
					    &epilog_key, compiler, debug,
					    si_build_tcs_epilog_function,
					    "Tessellation Control Shader Epilog");
	return shader->epilog != NULL;
}

/**
 * Select and compile (or reuse) GS parts (prolog).
 */
static bool si_shader_select_gs_parts(struct si_screen *sscreen,
				      struct ac_llvm_compiler *compiler,
				      struct si_shader *shader,
				      struct pipe_debug_callback *debug)
{
	if (sscreen->info.chip_class >= GFX9) {
		struct si_shader *es_main_part =
			shader->key.part.gs.es->main_shader_part_es;

		if (shader->key.part.gs.es->type == PIPE_SHADER_VERTEX &&
		    !si_get_vs_prolog(sscreen, compiler, shader, debug, es_main_part,
				      &shader->key.part.gs.vs_prolog))
			return false;

		shader->previous_stage = es_main_part;
	}

	if (!shader->key.part.gs.prolog.tri_strip_adj_fix)
		return true;

	union si_shader_part_key prolog_key;
	memset(&prolog_key, 0, sizeof(prolog_key));
	prolog_key.gs_prolog.states = shader->key.part.gs.prolog;

	shader->prolog2 = si_get_shader_part(sscreen, &sscreen->gs_prologs,
					    PIPE_SHADER_GEOMETRY, true,
					    &prolog_key, compiler, debug,
					    si_build_gs_prolog_function,
					    "Geometry Shader Prolog");
	return shader->prolog2 != NULL;
}

/**
 * Build the pixel shader prolog function. This handles:
 * - two-side color selection and interpolation
 * - overriding interpolation parameters for the API PS
 * - polygon stippling
 *
 * All preloaded SGPRs and VGPRs are passed through unmodified unless they are
 * overriden by other states. (e.g. per-sample interpolation)
 * Interpolated colors are stored after the preloaded VGPRs.
 */
static void si_build_ps_prolog_function(struct si_shader_context *ctx,
					union si_shader_part_key *key)
{
	struct si_function_info fninfo;
	LLVMValueRef ret, func;
	int num_returns, i, num_color_channels;

	assert(si_need_ps_prolog(key));

	si_init_function_info(&fninfo);

	/* Declare inputs. */
	for (i = 0; i < key->ps_prolog.num_input_sgprs; i++)
		add_arg(&fninfo, ARG_SGPR, ctx->i32);

	for (i = 0; i < key->ps_prolog.num_input_vgprs; i++)
		add_arg(&fninfo, ARG_VGPR, ctx->f32);

	/* Declare outputs (same as inputs + add colors if needed) */
	num_returns = fninfo.num_params;
	num_color_channels = util_bitcount(key->ps_prolog.colors_read);
	for (i = 0; i < num_color_channels; i++)
		fninfo.types[num_returns++] = ctx->f32;

	/* Create the function. */
	si_create_function(ctx, "ps_prolog", fninfo.types, num_returns,
			   &fninfo, 0);
	func = ctx->main_fn;

	/* Copy inputs to outputs. This should be no-op, as the registers match,
	 * but it will prevent the compiler from overwriting them unintentionally.
	 */
	ret = ctx->return_value;
	for (i = 0; i < fninfo.num_params; i++) {
		LLVMValueRef p = LLVMGetParam(func, i);
		ret = LLVMBuildInsertValue(ctx->ac.builder, ret, p, i, "");
	}

	/* Polygon stippling. */
	if (key->ps_prolog.states.poly_stipple) {
		/* POS_FIXED_PT is always last. */
		unsigned pos = key->ps_prolog.num_input_sgprs +
			       key->ps_prolog.num_input_vgprs - 1;
		LLVMValueRef list = si_prolog_get_rw_buffers(ctx);

		si_llvm_emit_polygon_stipple(ctx, list, pos);
	}

	if (key->ps_prolog.states.bc_optimize_for_persp ||
	    key->ps_prolog.states.bc_optimize_for_linear) {
		unsigned i, base = key->ps_prolog.num_input_sgprs;
		LLVMValueRef center[2], centroid[2], tmp, bc_optimize;

		/* The shader should do: if (PRIM_MASK[31]) CENTROID = CENTER;
		 * The hw doesn't compute CENTROID if the whole wave only
		 * contains fully-covered quads.
		 *
		 * PRIM_MASK is after user SGPRs.
		 */
		bc_optimize = LLVMGetParam(func, SI_PS_NUM_USER_SGPR);
		bc_optimize = LLVMBuildLShr(ctx->ac.builder, bc_optimize,
					    LLVMConstInt(ctx->i32, 31, 0), "");
		bc_optimize = LLVMBuildTrunc(ctx->ac.builder, bc_optimize,
					     ctx->i1, "");

		if (key->ps_prolog.states.bc_optimize_for_persp) {
			/* Read PERSP_CENTER. */
			for (i = 0; i < 2; i++)
				center[i] = LLVMGetParam(func, base + 2 + i);
			/* Read PERSP_CENTROID. */
			for (i = 0; i < 2; i++)
				centroid[i] = LLVMGetParam(func, base + 4 + i);
			/* Select PERSP_CENTROID. */
			for (i = 0; i < 2; i++) {
				tmp = LLVMBuildSelect(ctx->ac.builder, bc_optimize,
						      center[i], centroid[i], "");
				ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
							   tmp, base + 4 + i, "");
			}
		}
		if (key->ps_prolog.states.bc_optimize_for_linear) {
			/* Read LINEAR_CENTER. */
			for (i = 0; i < 2; i++)
				center[i] = LLVMGetParam(func, base + 8 + i);
			/* Read LINEAR_CENTROID. */
			for (i = 0; i < 2; i++)
				centroid[i] = LLVMGetParam(func, base + 10 + i);
			/* Select LINEAR_CENTROID. */
			for (i = 0; i < 2; i++) {
				tmp = LLVMBuildSelect(ctx->ac.builder, bc_optimize,
						      center[i], centroid[i], "");
				ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
							   tmp, base + 10 + i, "");
			}
		}
	}

	/* Force per-sample interpolation. */
	if (key->ps_prolog.states.force_persp_sample_interp) {
		unsigned i, base = key->ps_prolog.num_input_sgprs;
		LLVMValueRef persp_sample[2];

		/* Read PERSP_SAMPLE. */
		for (i = 0; i < 2; i++)
			persp_sample[i] = LLVMGetParam(func, base + i);
		/* Overwrite PERSP_CENTER. */
		for (i = 0; i < 2; i++)
			ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
						   persp_sample[i], base + 2 + i, "");
		/* Overwrite PERSP_CENTROID. */
		for (i = 0; i < 2; i++)
			ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
						   persp_sample[i], base + 4 + i, "");
	}
	if (key->ps_prolog.states.force_linear_sample_interp) {
		unsigned i, base = key->ps_prolog.num_input_sgprs;
		LLVMValueRef linear_sample[2];

		/* Read LINEAR_SAMPLE. */
		for (i = 0; i < 2; i++)
			linear_sample[i] = LLVMGetParam(func, base + 6 + i);
		/* Overwrite LINEAR_CENTER. */
		for (i = 0; i < 2; i++)
			ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
						   linear_sample[i], base + 8 + i, "");
		/* Overwrite LINEAR_CENTROID. */
		for (i = 0; i < 2; i++)
			ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
						   linear_sample[i], base + 10 + i, "");
	}

	/* Force center interpolation. */
	if (key->ps_prolog.states.force_persp_center_interp) {
		unsigned i, base = key->ps_prolog.num_input_sgprs;
		LLVMValueRef persp_center[2];

		/* Read PERSP_CENTER. */
		for (i = 0; i < 2; i++)
			persp_center[i] = LLVMGetParam(func, base + 2 + i);
		/* Overwrite PERSP_SAMPLE. */
		for (i = 0; i < 2; i++)
			ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
						   persp_center[i], base + i, "");
		/* Overwrite PERSP_CENTROID. */
		for (i = 0; i < 2; i++)
			ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
						   persp_center[i], base + 4 + i, "");
	}
	if (key->ps_prolog.states.force_linear_center_interp) {
		unsigned i, base = key->ps_prolog.num_input_sgprs;
		LLVMValueRef linear_center[2];

		/* Read LINEAR_CENTER. */
		for (i = 0; i < 2; i++)
			linear_center[i] = LLVMGetParam(func, base + 8 + i);
		/* Overwrite LINEAR_SAMPLE. */
		for (i = 0; i < 2; i++)
			ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
						   linear_center[i], base + 6 + i, "");
		/* Overwrite LINEAR_CENTROID. */
		for (i = 0; i < 2; i++)
			ret = LLVMBuildInsertValue(ctx->ac.builder, ret,
						   linear_center[i], base + 10 + i, "");
	}

	/* Interpolate colors. */
	unsigned color_out_idx = 0;
	for (i = 0; i < 2; i++) {
		unsigned writemask = (key->ps_prolog.colors_read >> (i * 4)) & 0xf;
		unsigned face_vgpr = key->ps_prolog.num_input_sgprs +
				     key->ps_prolog.face_vgpr_index;
		LLVMValueRef interp[2], color[4];
		LLVMValueRef interp_ij = NULL, prim_mask = NULL, face = NULL;

		if (!writemask)
			continue;

		/* If the interpolation qualifier is not CONSTANT (-1). */
		if (key->ps_prolog.color_interp_vgpr_index[i] != -1) {
			unsigned interp_vgpr = key->ps_prolog.num_input_sgprs +
					       key->ps_prolog.color_interp_vgpr_index[i];

			/* Get the (i,j) updated by bc_optimize handling. */
			interp[0] = LLVMBuildExtractValue(ctx->ac.builder, ret,
							  interp_vgpr, "");
			interp[1] = LLVMBuildExtractValue(ctx->ac.builder, ret,
							  interp_vgpr + 1, "");
			interp_ij = ac_build_gather_values(&ctx->ac, interp, 2);
		}

		/* Use the absolute location of the input. */
		prim_mask = LLVMGetParam(func, SI_PS_NUM_USER_SGPR);

		if (key->ps_prolog.states.color_two_side) {
			face = LLVMGetParam(func, face_vgpr);
			face = ac_to_integer(&ctx->ac, face);
		}

		interp_fs_input(ctx,
				key->ps_prolog.color_attr_index[i],
				TGSI_SEMANTIC_COLOR, i,
				key->ps_prolog.num_interp_inputs,
				key->ps_prolog.colors_read, interp_ij,
				prim_mask, face, color);

		while (writemask) {
			unsigned chan = u_bit_scan(&writemask);
			ret = LLVMBuildInsertValue(ctx->ac.builder, ret, color[chan],
						   fninfo.num_params + color_out_idx++, "");
		}
	}

	/* Section 15.2.2 (Shader Inputs) of the OpenGL 4.5 (Core Profile) spec
	 * says:
	 *
	 *    "When per-sample shading is active due to the use of a fragment
	 *     input qualified by sample or due to the use of the gl_SampleID
	 *     or gl_SamplePosition variables, only the bit for the current
	 *     sample is set in gl_SampleMaskIn. When state specifies multiple
	 *     fragment shader invocations for a given fragment, the sample
	 *     mask for any single fragment shader invocation may specify a
	 *     subset of the covered samples for the fragment. In this case,
	 *     the bit corresponding to each covered sample will be set in
	 *     exactly one fragment shader invocation."
	 *
	 * The samplemask loaded by hardware is always the coverage of the
	 * entire pixel/fragment, so mask bits out based on the sample ID.
	 */
	if (key->ps_prolog.states.samplemask_log_ps_iter) {
		/* The bit pattern matches that used by fixed function fragment
		 * processing. */
		static const uint16_t ps_iter_masks[] = {
			0xffff, /* not used */
			0x5555,
			0x1111,
			0x0101,
			0x0001,
		};
		assert(key->ps_prolog.states.samplemask_log_ps_iter < ARRAY_SIZE(ps_iter_masks));

		uint32_t ps_iter_mask = ps_iter_masks[key->ps_prolog.states.samplemask_log_ps_iter];
		unsigned ancillary_vgpr = key->ps_prolog.num_input_sgprs +
					  key->ps_prolog.ancillary_vgpr_index;
		LLVMValueRef sampleid = si_unpack_param(ctx, ancillary_vgpr, 8, 4);
		LLVMValueRef samplemask = LLVMGetParam(func, ancillary_vgpr + 1);

		samplemask = ac_to_integer(&ctx->ac, samplemask);
		samplemask = LLVMBuildAnd(
			ctx->ac.builder,
			samplemask,
			LLVMBuildShl(ctx->ac.builder,
				     LLVMConstInt(ctx->i32, ps_iter_mask, false),
				     sampleid, ""),
			"");
		samplemask = ac_to_float(&ctx->ac, samplemask);

		ret = LLVMBuildInsertValue(ctx->ac.builder, ret, samplemask,
					   ancillary_vgpr + 1, "");
	}

	/* Tell LLVM to insert WQM instruction sequence when needed. */
	if (key->ps_prolog.wqm) {
		LLVMAddTargetDependentFunctionAttr(func,
						   "amdgpu-ps-wqm-outputs", "");
	}

	si_llvm_build_ret(ctx, ret);
}

/**
 * Build the pixel shader epilog function. This handles everything that must be
 * emulated for pixel shader exports. (alpha-test, format conversions, etc)
 */
static void si_build_ps_epilog_function(struct si_shader_context *ctx,
					union si_shader_part_key *key)
{
	struct lp_build_tgsi_context *bld_base = &ctx->bld_base;
	struct si_function_info fninfo;
	LLVMValueRef depth = NULL, stencil = NULL, samplemask = NULL;
	int i;
	struct si_ps_exports exp = {};

	si_init_function_info(&fninfo);

	/* Declare input SGPRs. */
	ctx->param_rw_buffers = add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
	ctx->param_bindless_samplers_and_images = add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
	ctx->param_const_and_shader_buffers = add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
	ctx->param_samplers_and_images = add_arg(&fninfo, ARG_SGPR, ctx->ac.intptr);
	add_arg_checked(&fninfo, ARG_SGPR, ctx->f32, SI_PARAM_ALPHA_REF);

	/* Declare input VGPRs. */
	unsigned required_num_params =
		     fninfo.num_sgpr_params +
		     util_bitcount(key->ps_epilog.colors_written) * 4 +
		     key->ps_epilog.writes_z +
		     key->ps_epilog.writes_stencil +
		     key->ps_epilog.writes_samplemask;

	required_num_params = MAX2(required_num_params,
				   fninfo.num_sgpr_params + PS_EPILOG_SAMPLEMASK_MIN_LOC + 1);

	while (fninfo.num_params < required_num_params)
		add_arg(&fninfo, ARG_VGPR, ctx->f32);

	/* Create the function. */
	si_create_function(ctx, "ps_epilog", NULL, 0, &fninfo, 0);
	/* Disable elimination of unused inputs. */
	ac_llvm_add_target_dep_function_attr(ctx->main_fn,
					     "InitialPSInputAddr", 0xffffff);

	/* Process colors. */
	unsigned vgpr = fninfo.num_sgpr_params;
	unsigned colors_written = key->ps_epilog.colors_written;
	int last_color_export = -1;

	/* Find the last color export. */
	if (!key->ps_epilog.writes_z &&
	    !key->ps_epilog.writes_stencil &&
	    !key->ps_epilog.writes_samplemask) {
		unsigned spi_format = key->ps_epilog.states.spi_shader_col_format;

		/* If last_cbuf > 0, FS_COLOR0_WRITES_ALL_CBUFS is true. */
		if (colors_written == 0x1 && key->ps_epilog.states.last_cbuf > 0) {
			/* Just set this if any of the colorbuffers are enabled. */
			if (spi_format &
			    ((1ull << (4 * (key->ps_epilog.states.last_cbuf + 1))) - 1))
				last_color_export = 0;
		} else {
			for (i = 0; i < 8; i++)
				if (colors_written & (1 << i) &&
				    (spi_format >> (i * 4)) & 0xf)
					last_color_export = i;
		}
	}

	while (colors_written) {
		LLVMValueRef color[4];
		int mrt = u_bit_scan(&colors_written);

		for (i = 0; i < 4; i++)
			color[i] = LLVMGetParam(ctx->main_fn, vgpr++);

		si_export_mrt_color(bld_base, color, mrt,
				    fninfo.num_params - 1,
				    mrt == last_color_export, &exp);
	}

	/* Process depth, stencil, samplemask. */
	if (key->ps_epilog.writes_z)
		depth = LLVMGetParam(ctx->main_fn, vgpr++);
	if (key->ps_epilog.writes_stencil)
		stencil = LLVMGetParam(ctx->main_fn, vgpr++);
	if (key->ps_epilog.writes_samplemask)
		samplemask = LLVMGetParam(ctx->main_fn, vgpr++);

	if (depth || stencil || samplemask)
		si_export_mrt_z(bld_base, depth, stencil, samplemask, &exp);
	else if (last_color_export == -1)
		ac_build_export_null(&ctx->ac);

	if (exp.num)
		si_emit_ps_exports(ctx, &exp);

	/* Compile. */
	LLVMBuildRetVoid(ctx->ac.builder);
}

/**
 * Select and compile (or reuse) pixel shader parts (prolog & epilog).
 */
static bool si_shader_select_ps_parts(struct si_screen *sscreen,
				      struct ac_llvm_compiler *compiler,
				      struct si_shader *shader,
				      struct pipe_debug_callback *debug)
{
	union si_shader_part_key prolog_key;
	union si_shader_part_key epilog_key;

	/* Get the prolog. */
	si_get_ps_prolog_key(shader, &prolog_key, true);

	/* The prolog is a no-op if these aren't set. */
	if (si_need_ps_prolog(&prolog_key)) {
		shader->prolog =
			si_get_shader_part(sscreen, &sscreen->ps_prologs,
					   PIPE_SHADER_FRAGMENT, true,
					   &prolog_key, compiler, debug,
					   si_build_ps_prolog_function,
					   "Fragment Shader Prolog");
		if (!shader->prolog)
			return false;
	}

	/* Get the epilog. */
	si_get_ps_epilog_key(shader, &epilog_key);

	shader->epilog =
		si_get_shader_part(sscreen, &sscreen->ps_epilogs,
				   PIPE_SHADER_FRAGMENT, false,
				   &epilog_key, compiler, debug,
				   si_build_ps_epilog_function,
				   "Fragment Shader Epilog");
	if (!shader->epilog)
		return false;

	/* Enable POS_FIXED_PT if polygon stippling is enabled. */
	if (shader->key.part.ps.prolog.poly_stipple) {
		shader->config.spi_ps_input_ena |= S_0286CC_POS_FIXED_PT_ENA(1);
		assert(G_0286CC_POS_FIXED_PT_ENA(shader->config.spi_ps_input_addr));
	}

	/* Set up the enable bits for per-sample shading if needed. */
	if (shader->key.part.ps.prolog.force_persp_sample_interp &&
	    (G_0286CC_PERSP_CENTER_ENA(shader->config.spi_ps_input_ena) ||
	     G_0286CC_PERSP_CENTROID_ENA(shader->config.spi_ps_input_ena))) {
		shader->config.spi_ps_input_ena &= C_0286CC_PERSP_CENTER_ENA;
		shader->config.spi_ps_input_ena &= C_0286CC_PERSP_CENTROID_ENA;
		shader->config.spi_ps_input_ena |= S_0286CC_PERSP_SAMPLE_ENA(1);
	}
	if (shader->key.part.ps.prolog.force_linear_sample_interp &&
	    (G_0286CC_LINEAR_CENTER_ENA(shader->config.spi_ps_input_ena) ||
	     G_0286CC_LINEAR_CENTROID_ENA(shader->config.spi_ps_input_ena))) {
		shader->config.spi_ps_input_ena &= C_0286CC_LINEAR_CENTER_ENA;
		shader->config.spi_ps_input_ena &= C_0286CC_LINEAR_CENTROID_ENA;
		shader->config.spi_ps_input_ena |= S_0286CC_LINEAR_SAMPLE_ENA(1);
	}
	if (shader->key.part.ps.prolog.force_persp_center_interp &&
	    (G_0286CC_PERSP_SAMPLE_ENA(shader->config.spi_ps_input_ena) ||
	     G_0286CC_PERSP_CENTROID_ENA(shader->config.spi_ps_input_ena))) {
		shader->config.spi_ps_input_ena &= C_0286CC_PERSP_SAMPLE_ENA;
		shader->config.spi_ps_input_ena &= C_0286CC_PERSP_CENTROID_ENA;
		shader->config.spi_ps_input_ena |= S_0286CC_PERSP_CENTER_ENA(1);
	}
	if (shader->key.part.ps.prolog.force_linear_center_interp &&
	    (G_0286CC_LINEAR_SAMPLE_ENA(shader->config.spi_ps_input_ena) ||
	     G_0286CC_LINEAR_CENTROID_ENA(shader->config.spi_ps_input_ena))) {
		shader->config.spi_ps_input_ena &= C_0286CC_LINEAR_SAMPLE_ENA;
		shader->config.spi_ps_input_ena &= C_0286CC_LINEAR_CENTROID_ENA;
		shader->config.spi_ps_input_ena |= S_0286CC_LINEAR_CENTER_ENA(1);
	}

	/* POW_W_FLOAT requires that one of the perspective weights is enabled. */
	if (G_0286CC_POS_W_FLOAT_ENA(shader->config.spi_ps_input_ena) &&
	    !(shader->config.spi_ps_input_ena & 0xf)) {
		shader->config.spi_ps_input_ena |= S_0286CC_PERSP_CENTER_ENA(1);
		assert(G_0286CC_PERSP_CENTER_ENA(shader->config.spi_ps_input_addr));
	}

	/* At least one pair of interpolation weights must be enabled. */
	if (!(shader->config.spi_ps_input_ena & 0x7f)) {
		shader->config.spi_ps_input_ena |= S_0286CC_LINEAR_CENTER_ENA(1);
		assert(G_0286CC_LINEAR_CENTER_ENA(shader->config.spi_ps_input_addr));
	}

	/* Samplemask fixup requires the sample ID. */
	if (shader->key.part.ps.prolog.samplemask_log_ps_iter) {
		shader->config.spi_ps_input_ena |= S_0286CC_ANCILLARY_ENA(1);
		assert(G_0286CC_ANCILLARY_ENA(shader->config.spi_ps_input_addr));
	}

	/* The sample mask input is always enabled, because the API shader always
	 * passes it through to the epilog. Disable it here if it's unused.
	 */
	if (!shader->key.part.ps.epilog.poly_line_smoothing &&
	    !shader->selector->info.reads_samplemask)
		shader->config.spi_ps_input_ena &= C_0286CC_SAMPLE_COVERAGE_ENA;

	return true;
}

void si_multiwave_lds_size_workaround(struct si_screen *sscreen,
				      unsigned *lds_size)
{
	/* If tessellation is all offchip and on-chip GS isn't used, this
	 * workaround is not needed.
	 */
	return;

	/* SPI barrier management bug:
	 *   Make sure we have at least 4k of LDS in use to avoid the bug.
	 *   It applies to workgroup sizes of more than one wavefront.
	 */
	if (sscreen->info.family == CHIP_BONAIRE ||
	    sscreen->info.family == CHIP_KABINI ||
	    sscreen->info.family == CHIP_MULLINS)
		*lds_size = MAX2(*lds_size, 8);
}

static void si_fix_resource_usage(struct si_screen *sscreen,
				  struct si_shader *shader)
{
	unsigned min_sgprs = shader->info.num_input_sgprs + 2; /* VCC */

	shader->config.num_sgprs = MAX2(shader->config.num_sgprs, min_sgprs);

	if (shader->selector->type == PIPE_SHADER_COMPUTE &&
	    si_get_max_workgroup_size(shader) > 64) {
		si_multiwave_lds_size_workaround(sscreen,
						 &shader->config.lds_size);
	}
}

int si_shader_create(struct si_screen *sscreen, struct ac_llvm_compiler *compiler,
		     struct si_shader *shader,
		     struct pipe_debug_callback *debug)
{
	struct si_shader_selector *sel = shader->selector;
	struct si_shader *mainp = *si_get_main_shader_part(sel, &shader->key);
	int r;

	/* LS, ES, VS are compiled on demand if the main part hasn't been
	 * compiled for that stage.
	 *
	 * Vertex shaders are compiled on demand when a vertex fetch
	 * workaround must be applied.
	 */
	if (shader->is_monolithic) {
		/* Monolithic shader (compiled as a whole, has many variants,
		 * may take a long time to compile).
		 */
		r = si_compile_tgsi_shader(sscreen, compiler, shader, debug);
		if (r)
			return r;
	} else {
		/* The shader consists of several parts:
		 *
		 * - the middle part is the user shader, it has 1 variant only
		 *   and it was compiled during the creation of the shader
		 *   selector
		 * - the prolog part is inserted at the beginning
		 * - the epilog part is inserted at the end
		 *
		 * The prolog and epilog have many (but simple) variants.
		 *
		 * Starting with gfx9, geometry and tessellation control
		 * shaders also contain the prolog and user shader parts of
		 * the previous shader stage.
		 */

		if (!mainp)
			return -1;

		/* Copy the compiled TGSI shader data over. */
		shader->is_binary_shared = true;
		shader->binary = mainp->binary;
		shader->config = mainp->config;
		shader->info.num_input_sgprs = mainp->info.num_input_sgprs;
		shader->info.num_input_vgprs = mainp->info.num_input_vgprs;
		shader->info.face_vgpr_index = mainp->info.face_vgpr_index;
		shader->info.ancillary_vgpr_index = mainp->info.ancillary_vgpr_index;
		memcpy(shader->info.vs_output_param_offset,
		       mainp->info.vs_output_param_offset,
		       sizeof(mainp->info.vs_output_param_offset));
		shader->info.uses_instanceid = mainp->info.uses_instanceid;
		shader->info.nr_pos_exports = mainp->info.nr_pos_exports;
		shader->info.nr_param_exports = mainp->info.nr_param_exports;

		/* Select prologs and/or epilogs. */
		switch (sel->type) {
		case PIPE_SHADER_VERTEX:
			if (!si_shader_select_vs_parts(sscreen, compiler, shader, debug))
				return -1;
			break;
		case PIPE_SHADER_TESS_CTRL:
			if (!si_shader_select_tcs_parts(sscreen, compiler, shader, debug))
				return -1;
			break;
		case PIPE_SHADER_TESS_EVAL:
			break;
		case PIPE_SHADER_GEOMETRY:
			if (!si_shader_select_gs_parts(sscreen, compiler, shader, debug))
				return -1;
			break;
		case PIPE_SHADER_FRAGMENT:
			if (!si_shader_select_ps_parts(sscreen, compiler, shader, debug))
				return -1;

			/* Make sure we have at least as many VGPRs as there
			 * are allocated inputs.
			 */
			shader->config.num_vgprs = MAX2(shader->config.num_vgprs,
							shader->info.num_input_vgprs);
			break;
		}

		/* Update SGPR and VGPR counts. */
		if (shader->prolog) {
			shader->config.num_sgprs = MAX2(shader->config.num_sgprs,
							shader->prolog->config.num_sgprs);
			shader->config.num_vgprs = MAX2(shader->config.num_vgprs,
							shader->prolog->config.num_vgprs);
		}
		if (shader->previous_stage) {
			shader->config.num_sgprs = MAX2(shader->config.num_sgprs,
							shader->previous_stage->config.num_sgprs);
			shader->config.num_vgprs = MAX2(shader->config.num_vgprs,
							shader->previous_stage->config.num_vgprs);
			shader->config.spilled_sgprs =
				MAX2(shader->config.spilled_sgprs,
				     shader->previous_stage->config.spilled_sgprs);
			shader->config.spilled_vgprs =
				MAX2(shader->config.spilled_vgprs,
				     shader->previous_stage->config.spilled_vgprs);
			shader->config.private_mem_vgprs =
				MAX2(shader->config.private_mem_vgprs,
				     shader->previous_stage->config.private_mem_vgprs);
			shader->config.scratch_bytes_per_wave =
				MAX2(shader->config.scratch_bytes_per_wave,
				     shader->previous_stage->config.scratch_bytes_per_wave);
			shader->info.uses_instanceid |=
				shader->previous_stage->info.uses_instanceid;
		}
		if (shader->prolog2) {
			shader->config.num_sgprs = MAX2(shader->config.num_sgprs,
							shader->prolog2->config.num_sgprs);
			shader->config.num_vgprs = MAX2(shader->config.num_vgprs,
							shader->prolog2->config.num_vgprs);
		}
		if (shader->epilog) {
			shader->config.num_sgprs = MAX2(shader->config.num_sgprs,
							shader->epilog->config.num_sgprs);
			shader->config.num_vgprs = MAX2(shader->config.num_vgprs,
							shader->epilog->config.num_vgprs);
		}
		si_calculate_max_simd_waves(shader);
	}

	si_fix_resource_usage(sscreen, shader);
	si_shader_dump(sscreen, shader, debug, sel->info.processor,
		       stderr, true);

	/* Upload. */
	r = si_shader_binary_upload(sscreen, shader);
	if (r) {
		fprintf(stderr, "LLVM failed to upload shader\n");
		return r;
	}

	return 0;
}

void si_shader_destroy(struct si_shader *shader)
{
	if (shader->scratch_bo)
		r600_resource_reference(&shader->scratch_bo, NULL);

	r600_resource_reference(&shader->bo, NULL);

	if (!shader->is_binary_shared)
		ac_shader_binary_clean(&shader->binary);

	free(shader->shader_log);
}
