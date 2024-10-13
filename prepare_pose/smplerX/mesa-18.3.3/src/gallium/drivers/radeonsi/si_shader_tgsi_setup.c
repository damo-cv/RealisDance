/*
 * Copyright 2016 Advanced Micro Devices, Inc.
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

#include "si_shader_internal.h"
#include "si_pipe.h"
#include "ac_llvm_util.h"
#include "util/u_memory.h"

enum si_llvm_calling_convention {
	RADEON_LLVM_AMDGPU_VS = 87,
	RADEON_LLVM_AMDGPU_GS = 88,
	RADEON_LLVM_AMDGPU_PS = 89,
	RADEON_LLVM_AMDGPU_CS = 90,
	RADEON_LLVM_AMDGPU_HS = 93,
};

struct si_llvm_diagnostics {
	struct pipe_debug_callback *debug;
	unsigned retval;
};

static void si_diagnostic_handler(LLVMDiagnosticInfoRef di, void *context)
{
	struct si_llvm_diagnostics *diag = (struct si_llvm_diagnostics *)context;
	LLVMDiagnosticSeverity severity = LLVMGetDiagInfoSeverity(di);
	char *description = LLVMGetDiagInfoDescription(di);
	const char *severity_str = NULL;

	switch (severity) {
	case LLVMDSError:
		severity_str = "error";
		break;
	case LLVMDSWarning:
		severity_str = "warning";
		break;
	case LLVMDSRemark:
		severity_str = "remark";
		break;
	case LLVMDSNote:
		severity_str = "note";
		break;
	default:
		severity_str = "unknown";
	}

	pipe_debug_message(diag->debug, SHADER_INFO,
			   "LLVM diagnostic (%s): %s", severity_str, description);

	if (severity == LLVMDSError) {
		diag->retval = 1;
		fprintf(stderr,"LLVM triggered Diagnostic Handler: %s\n", description);
	}

	LLVMDisposeMessage(description);
}

/**
 * Compile an LLVM module to machine code.
 *
 * @returns 0 for success, 1 for failure
 */
unsigned si_llvm_compile(LLVMModuleRef M, struct ac_shader_binary *binary,
			 struct ac_llvm_compiler *compiler,
			 struct pipe_debug_callback *debug,
			 bool less_optimized)
{
	struct ac_compiler_passes *passes =
		less_optimized && compiler->low_opt_passes ?
			compiler->low_opt_passes : compiler->passes;
	struct si_llvm_diagnostics diag;
	LLVMContextRef llvm_ctx;

	diag.debug = debug;
	diag.retval = 0;

	/* Setup Diagnostic Handler*/
	llvm_ctx = LLVMGetModuleContext(M);

	LLVMContextSetDiagnosticHandler(llvm_ctx, si_diagnostic_handler, &diag);

	/* Compile IR. */
	if (!ac_compile_module_to_binary(passes, M, binary))
		diag.retval = 1;

	if (diag.retval != 0)
		pipe_debug_message(debug, SHADER_INFO, "LLVM compile failed");
	return diag.retval;
}

LLVMTypeRef tgsi2llvmtype(struct lp_build_tgsi_context *bld_base,
			  enum tgsi_opcode_type type)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);

	switch (type) {
	case TGSI_TYPE_UNSIGNED:
	case TGSI_TYPE_SIGNED:
		return ctx->ac.i32;
	case TGSI_TYPE_UNSIGNED64:
	case TGSI_TYPE_SIGNED64:
		return ctx->ac.i64;
	case TGSI_TYPE_DOUBLE:
		return ctx->ac.f64;
	case TGSI_TYPE_UNTYPED:
	case TGSI_TYPE_FLOAT:
		return ctx->ac.f32;
	default: break;
	}
	return 0;
}

LLVMValueRef bitcast(struct lp_build_tgsi_context *bld_base,
		     enum tgsi_opcode_type type, LLVMValueRef value)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMTypeRef dst_type = tgsi2llvmtype(bld_base, type);

	if (dst_type)
		return LLVMBuildBitCast(ctx->ac.builder, value, dst_type, "");
	else
		return value;
}

/**
 * Return a value that is equal to the given i32 \p index if it lies in [0,num)
 * or an undefined value in the same interval otherwise.
 */
LLVMValueRef si_llvm_bound_index(struct si_shader_context *ctx,
				 LLVMValueRef index,
				 unsigned num)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef c_max = LLVMConstInt(ctx->i32, num - 1, 0);
	LLVMValueRef cc;

	if (util_is_power_of_two_or_zero(num)) {
		index = LLVMBuildAnd(builder, index, c_max, "");
	} else {
		/* In theory, this MAX pattern should result in code that is
		 * as good as the bit-wise AND above.
		 *
		 * In practice, LLVM generates worse code (at the time of
		 * writing), because its value tracking is not strong enough.
		 */
		cc = LLVMBuildICmp(builder, LLVMIntULE, index, c_max, "");
		index = LLVMBuildSelect(builder, cc, index, c_max, "");
	}

	return index;
}

static LLVMValueRef emit_swizzle(struct lp_build_tgsi_context *bld_base,
				 LLVMValueRef value,
				 unsigned swizzle_x,
				 unsigned swizzle_y,
				 unsigned swizzle_z,
				 unsigned swizzle_w)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef swizzles[4];

	swizzles[0] = LLVMConstInt(ctx->i32, swizzle_x, 0);
	swizzles[1] = LLVMConstInt(ctx->i32, swizzle_y, 0);
	swizzles[2] = LLVMConstInt(ctx->i32, swizzle_z, 0);
	swizzles[3] = LLVMConstInt(ctx->i32, swizzle_w, 0);

	return LLVMBuildShuffleVector(ctx->ac.builder,
				      value,
				      LLVMGetUndef(LLVMTypeOf(value)),
				      LLVMConstVector(swizzles, 4), "");
}

/**
 * Return the description of the array covering the given temporary register
 * index.
 */
static unsigned
get_temp_array_id(struct lp_build_tgsi_context *bld_base,
		  unsigned reg_index,
		  const struct tgsi_ind_register *reg)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	unsigned num_arrays = ctx->bld_base.info->array_max[TGSI_FILE_TEMPORARY];
	unsigned i;

	if (reg && reg->ArrayID > 0 && reg->ArrayID <= num_arrays)
		return reg->ArrayID;

	for (i = 0; i < num_arrays; i++) {
		const struct tgsi_array_info *array = &ctx->temp_arrays[i];

		if (reg_index >= array->range.First && reg_index <= array->range.Last)
			return i + 1;
	}

	return 0;
}

static struct tgsi_declaration_range
get_array_range(struct lp_build_tgsi_context *bld_base,
		unsigned File, unsigned reg_index,
		const struct tgsi_ind_register *reg)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	struct tgsi_declaration_range range;

	if (File == TGSI_FILE_TEMPORARY) {
		unsigned array_id = get_temp_array_id(bld_base, reg_index, reg);
		if (array_id)
			return ctx->temp_arrays[array_id - 1].range;
	}

	range.First = 0;
	range.Last = bld_base->info->file_max[File];
	return range;
}

/**
 * For indirect registers, construct a pointer directly to the requested
 * element using getelementptr if possible.
 *
 * Returns NULL if the insertelement/extractelement fallback for array access
 * must be used.
 */
static LLVMValueRef
get_pointer_into_array(struct si_shader_context *ctx,
		       unsigned file,
		       unsigned swizzle,
		       unsigned reg_index,
		       const struct tgsi_ind_register *reg_indirect)
{
	unsigned array_id;
	struct tgsi_array_info *array;
	LLVMValueRef idxs[2];
	LLVMValueRef index;
	LLVMValueRef alloca;

	if (file != TGSI_FILE_TEMPORARY)
		return NULL;

	array_id = get_temp_array_id(&ctx->bld_base, reg_index, reg_indirect);
	if (!array_id)
		return NULL;

	alloca = ctx->temp_array_allocas[array_id - 1];
	if (!alloca)
		return NULL;

	array = &ctx->temp_arrays[array_id - 1];

	if (!(array->writemask & (1 << swizzle)))
		return ctx->undef_alloca;

	index = si_get_indirect_index(ctx, reg_indirect, 1,
				      reg_index - ctx->temp_arrays[array_id - 1].range.First);

	/* Ensure that the index is within a valid range, to guard against
	 * VM faults and overwriting critical data (e.g. spilled resource
	 * descriptors).
	 *
	 * TODO It should be possible to avoid the additional instructions
	 * if LLVM is changed so that it guarantuees:
	 * 1. the scratch space descriptor isolates the current wave (this
	 *    could even save the scratch offset SGPR at the cost of an
	 *    additional SALU instruction)
	 * 2. the memory for allocas must be allocated at the _end_ of the
	 *    scratch space (after spilled registers)
	 */
	index = si_llvm_bound_index(ctx, index, array->range.Last - array->range.First + 1);

	index = ac_build_imad(&ctx->ac, index,
			      LLVMConstInt(ctx->i32, util_bitcount(array->writemask), 0),
			      LLVMConstInt(ctx->i32,
					   util_bitcount(array->writemask & ((1 << swizzle) - 1)), 0));
	idxs[0] = ctx->i32_0;
	idxs[1] = index;
	return LLVMBuildGEP(ctx->ac.builder, alloca, idxs, 2, "");
}

LLVMValueRef
si_llvm_emit_fetch_64bit(struct lp_build_tgsi_context *bld_base,
			 LLVMTypeRef type,
			 LLVMValueRef ptr,
			 LLVMValueRef ptr2)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef values[2] = {
		ac_to_integer(&ctx->ac, ptr),
		ac_to_integer(&ctx->ac, ptr2),
	};
	LLVMValueRef result = ac_build_gather_values(&ctx->ac, values, 2);
	return LLVMBuildBitCast(ctx->ac.builder, result, type, "");
}

static LLVMValueRef
emit_array_fetch(struct lp_build_tgsi_context *bld_base,
		 unsigned File, enum tgsi_opcode_type type,
		 struct tgsi_declaration_range range,
		 unsigned swizzle_in)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	unsigned i, size = range.Last - range.First + 1;
	LLVMTypeRef vec = LLVMVectorType(tgsi2llvmtype(bld_base, type), size);
	LLVMValueRef result = LLVMGetUndef(vec);
	unsigned swizzle = swizzle_in;
	struct tgsi_full_src_register tmp_reg = {};
	tmp_reg.Register.File = File;
	if (tgsi_type_is_64bit(type))
		swizzle |= (swizzle_in + 1) << 16;

	for (i = 0; i < size; ++i) {
		tmp_reg.Register.Index = i + range.First;

		LLVMValueRef temp = si_llvm_emit_fetch(bld_base, &tmp_reg, type, swizzle);
		result = LLVMBuildInsertElement(ctx->ac.builder, result, temp,
			LLVMConstInt(ctx->i32, i, 0), "array_vector");
	}
	return result;
}

static LLVMValueRef
load_value_from_array(struct lp_build_tgsi_context *bld_base,
		      unsigned file,
		      enum tgsi_opcode_type type,
		      unsigned swizzle,
		      unsigned reg_index,
		      const struct tgsi_ind_register *reg_indirect)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef ptr;

	ptr = get_pointer_into_array(ctx, file, swizzle, reg_index, reg_indirect);
	if (ptr) {
		LLVMValueRef val = LLVMBuildLoad(builder, ptr, "");
		if (tgsi_type_is_64bit(type)) {
			LLVMValueRef ptr_hi, val_hi;
			ptr_hi = LLVMBuildGEP(builder, ptr, &ctx->i32_1, 1, "");
			val_hi = LLVMBuildLoad(builder, ptr_hi, "");
			val = si_llvm_emit_fetch_64bit(bld_base, tgsi2llvmtype(bld_base, type),
						       val, val_hi);
		}

		return val;
	} else {
		struct tgsi_declaration_range range =
			get_array_range(bld_base, file, reg_index, reg_indirect);
		LLVMValueRef index =
			si_get_indirect_index(ctx, reg_indirect, 1, reg_index - range.First);
		LLVMValueRef array =
			emit_array_fetch(bld_base, file, type, range, swizzle);
		return LLVMBuildExtractElement(builder, array, index, "");
	}
}

static void
store_value_to_array(struct lp_build_tgsi_context *bld_base,
		     LLVMValueRef value,
		     unsigned file,
		     unsigned chan_index,
		     unsigned reg_index,
		     const struct tgsi_ind_register *reg_indirect)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef ptr;

	ptr = get_pointer_into_array(ctx, file, chan_index, reg_index, reg_indirect);
	if (ptr) {
		LLVMBuildStore(builder, value, ptr);
	} else {
		unsigned i, size;
		struct tgsi_declaration_range range = get_array_range(bld_base, file, reg_index, reg_indirect);
		LLVMValueRef index = si_get_indirect_index(ctx, reg_indirect, 1, reg_index - range.First);
		LLVMValueRef array =
			emit_array_fetch(bld_base, file, TGSI_TYPE_FLOAT, range, chan_index);
		LLVMValueRef temp_ptr;

		array = LLVMBuildInsertElement(builder, array, value, index, "");

		size = range.Last - range.First + 1;
		for (i = 0; i < size; ++i) {
			switch(file) {
			case TGSI_FILE_OUTPUT:
				temp_ptr = ctx->outputs[i + range.First][chan_index];
				break;

			case TGSI_FILE_TEMPORARY:
				if (range.First + i >= ctx->temps_count)
					continue;
				temp_ptr = ctx->temps[(i + range.First) * TGSI_NUM_CHANNELS + chan_index];
				break;

			default:
				continue;
			}
			value = LLVMBuildExtractElement(builder, array,
				LLVMConstInt(ctx->i32, i, 0), "");
			LLVMBuildStore(builder, value, temp_ptr);
		}
	}
}

/* If this is true, preload FS inputs at the beginning of shaders. Otherwise,
 * reload them at each use. This must be true if the shader is using
 * derivatives and KILL, because KILL can leave the WQM and then a lazy
 * input load isn't in the WQM anymore.
 */
static bool si_preload_fs_inputs(struct si_shader_context *ctx)
{
	struct si_shader_selector *sel = ctx->shader->selector;

	return sel->info.uses_derivatives &&
	       sel->info.uses_kill;
}

static LLVMValueRef
get_output_ptr(struct lp_build_tgsi_context *bld_base, unsigned index,
	       unsigned chan)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);

	assert(index <= ctx->bld_base.info->file_max[TGSI_FILE_OUTPUT]);
	return ctx->outputs[index][chan];
}

LLVMValueRef si_llvm_emit_fetch(struct lp_build_tgsi_context *bld_base,
				const struct tgsi_full_src_register *reg,
				enum tgsi_opcode_type type,
				unsigned swizzle_in)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef result = NULL, ptr, ptr2;
	unsigned swizzle = swizzle_in & 0xffff;

	if (swizzle_in == ~0) {
		LLVMValueRef values[TGSI_NUM_CHANNELS];
		unsigned chan;
		for (chan = 0; chan < TGSI_NUM_CHANNELS; chan++) {
			values[chan] = si_llvm_emit_fetch(bld_base, reg, type, chan);
		}
		return ac_build_gather_values(&ctx->ac, values,
					      TGSI_NUM_CHANNELS);
	}

	if (reg->Register.Indirect) {
		LLVMValueRef load = load_value_from_array(bld_base, reg->Register.File, type,
				swizzle, reg->Register.Index, &reg->Indirect);
		return bitcast(bld_base, type, load);
	}

	switch(reg->Register.File) {
	case TGSI_FILE_IMMEDIATE: {
		LLVMTypeRef ctype = tgsi2llvmtype(bld_base, type);
		if (tgsi_type_is_64bit(type)) {
			result = LLVMGetUndef(LLVMVectorType(ctx->i32, 2));
			result = LLVMConstInsertElement(result,
							ctx->imms[reg->Register.Index * TGSI_NUM_CHANNELS + swizzle],
							ctx->i32_0);
			result = LLVMConstInsertElement(result,
							ctx->imms[reg->Register.Index * TGSI_NUM_CHANNELS + (swizzle_in >> 16)],
							ctx->i32_1);
			return LLVMConstBitCast(result, ctype);
		} else {
			return LLVMConstBitCast(ctx->imms[reg->Register.Index * TGSI_NUM_CHANNELS + swizzle], ctype);
		}
	}

	case TGSI_FILE_INPUT: {
		unsigned index = reg->Register.Index;
		LLVMValueRef input[4];

		/* I don't think doing this for vertex shaders is beneficial.
		 * For those, we want to make sure the VMEM loads are executed
		 * only once. Fragment shaders don't care much, because
		 * v_interp instructions are much cheaper than VMEM loads.
		 */
		if (!si_preload_fs_inputs(ctx) &&
		    ctx->bld_base.info->processor == PIPE_SHADER_FRAGMENT)
			ctx->load_input(ctx, index, &ctx->input_decls[index], input);
		else
			memcpy(input, &ctx->inputs[index * 4], sizeof(input));

		result = input[swizzle];

		if (tgsi_type_is_64bit(type)) {
			ptr = result;
			ptr2 = input[swizzle_in >> 16];
			return si_llvm_emit_fetch_64bit(bld_base, tgsi2llvmtype(bld_base, type),
							ptr, ptr2);
		}
		break;
	}

	case TGSI_FILE_TEMPORARY:
		if (reg->Register.Index >= ctx->temps_count)
			return LLVMGetUndef(tgsi2llvmtype(bld_base, type));
		ptr = ctx->temps[reg->Register.Index * TGSI_NUM_CHANNELS + swizzle];
		if (tgsi_type_is_64bit(type)) {
			ptr2 = ctx->temps[reg->Register.Index * TGSI_NUM_CHANNELS + (swizzle_in >> 16)];
			return si_llvm_emit_fetch_64bit(bld_base, tgsi2llvmtype(bld_base, type),
							LLVMBuildLoad(builder, ptr, ""),
							LLVMBuildLoad(builder, ptr2, ""));
		}
		result = LLVMBuildLoad(builder, ptr, "");
		break;

	case TGSI_FILE_OUTPUT:
		ptr = get_output_ptr(bld_base, reg->Register.Index, swizzle);
		if (tgsi_type_is_64bit(type)) {
			ptr2 = get_output_ptr(bld_base, reg->Register.Index, (swizzle_in >> 16));
			return si_llvm_emit_fetch_64bit(bld_base, tgsi2llvmtype(bld_base, type),
							LLVMBuildLoad(builder, ptr, ""),
							LLVMBuildLoad(builder, ptr2, ""));
		}
		result = LLVMBuildLoad(builder, ptr, "");
		break;

	default:
		return LLVMGetUndef(tgsi2llvmtype(bld_base, type));
	}

	return bitcast(bld_base, type, result);
}

static LLVMValueRef fetch_system_value(struct lp_build_tgsi_context *bld_base,
				       const struct tgsi_full_src_register *reg,
				       enum tgsi_opcode_type type,
				       unsigned swizzle_in)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef cval = ctx->system_values[reg->Register.Index];
	unsigned swizzle = swizzle_in & 0xffff;

	if (tgsi_type_is_64bit(type)) {
		LLVMValueRef lo, hi;

		assert(swizzle == 0 || swizzle == 2);

		lo = LLVMBuildExtractElement(
			builder, cval, LLVMConstInt(ctx->i32, swizzle, 0), "");
		hi = LLVMBuildExtractElement(
			builder, cval, LLVMConstInt(ctx->i32, (swizzle_in >> 16), 0), "");

		return si_llvm_emit_fetch_64bit(bld_base, tgsi2llvmtype(bld_base, type),
						lo, hi);
	}

	if (LLVMGetTypeKind(LLVMTypeOf(cval)) == LLVMVectorTypeKind) {
		cval = LLVMBuildExtractElement(
			builder, cval, LLVMConstInt(ctx->i32, swizzle, 0), "");
	} else {
		assert(swizzle == 0);
	}

	return bitcast(bld_base, type, cval);
}

static void emit_declaration(struct lp_build_tgsi_context *bld_base,
			     const struct tgsi_full_declaration *decl)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMBuilderRef builder = ctx->ac.builder;
	unsigned first, last, i;
	switch(decl->Declaration.File) {
	case TGSI_FILE_ADDRESS:
	{
		 unsigned idx;
		for (idx = decl->Range.First; idx <= decl->Range.Last; idx++) {
			unsigned chan;
			for (chan = 0; chan < TGSI_NUM_CHANNELS; chan++) {
				 ctx->addrs[idx][chan] = ac_build_alloca_undef(
					&ctx->ac, ctx->i32, "");
			}
		}
		break;
	}

	case TGSI_FILE_TEMPORARY:
	{
		char name[18] = "";
		LLVMValueRef array_alloca = NULL;
		unsigned decl_size;
		unsigned writemask = decl->Declaration.UsageMask;
		first = decl->Range.First;
		last = decl->Range.Last;
		decl_size = 4 * ((last - first) + 1);

		if (decl->Declaration.Array) {
			unsigned id = decl->Array.ArrayID - 1;
			unsigned array_size;

			writemask &= ctx->temp_arrays[id].writemask;
			ctx->temp_arrays[id].writemask = writemask;
			array_size = ((last - first) + 1) * util_bitcount(writemask);

			/* If the array has more than 16 elements, store it
			 * in memory using an alloca that spans the entire
			 * array.
			 *
			 * Otherwise, store each array element individually.
			 * We will then generate vectors (per-channel, up to
			 * <16 x float> if the usagemask is a single bit) for
			 * indirect addressing.
			 *
			 * Note that 16 is the number of vector elements that
			 * LLVM will store in a register, so theoretically an
			 * array with up to 4 * 16 = 64 elements could be
			 * handled this way, but whether that's a good idea
			 * depends on VGPR register pressure elsewhere.
			 *
			 * FIXME: We shouldn't need to have the non-alloca
			 * code path for arrays. LLVM should be smart enough to
			 * promote allocas into registers when profitable.
			 */
			if (array_size > 16 ||
			    !ctx->screen->llvm_has_working_vgpr_indexing) {
				array_alloca = ac_build_alloca_undef(&ctx->ac,
					LLVMArrayType(ctx->f32,
						      array_size), "array");
				ctx->temp_array_allocas[id] = array_alloca;
			}
		}

		if (!ctx->temps_count) {
			ctx->temps_count = bld_base->info->file_max[TGSI_FILE_TEMPORARY] + 1;
			ctx->temps = MALLOC(TGSI_NUM_CHANNELS * ctx->temps_count * sizeof(LLVMValueRef));
		}
		if (!array_alloca) {
			for (i = 0; i < decl_size; ++i) {
#ifdef DEBUG
				snprintf(name, sizeof(name), "TEMP%d.%c",
					 first + i / 4, "xyzw"[i % 4]);
#endif
				ctx->temps[first * TGSI_NUM_CHANNELS + i] =
					ac_build_alloca_undef(&ctx->ac,
							      ctx->f32,
							      name);
			}
		} else {
			LLVMValueRef idxs[2] = {
				ctx->i32_0,
				NULL
			};
			unsigned j = 0;

			if (writemask != TGSI_WRITEMASK_XYZW &&
			    !ctx->undef_alloca) {
				/* Create a dummy alloca. We use it so that we
				 * have a pointer that is safe to load from if
				 * a shader ever reads from a channel that
				 * it never writes to.
				 */
				ctx->undef_alloca = ac_build_alloca_undef(
					&ctx->ac, ctx->f32, "undef");
			}

			for (i = 0; i < decl_size; ++i) {
				LLVMValueRef ptr;
				if (writemask & (1 << (i % 4))) {
#ifdef DEBUG
					snprintf(name, sizeof(name), "TEMP%d.%c",
						 first + i / 4, "xyzw"[i % 4]);
#endif
					idxs[1] = LLVMConstInt(ctx->i32, j, 0);
					ptr = LLVMBuildGEP(builder, array_alloca, idxs, 2, name);
					j++;
				} else {
					ptr = ctx->undef_alloca;
				}
				ctx->temps[first * TGSI_NUM_CHANNELS + i] = ptr;
			}
		}
		break;
	}
	case TGSI_FILE_INPUT:
	{
		unsigned idx;
		for (idx = decl->Range.First; idx <= decl->Range.Last; idx++) {
			if (ctx->load_input &&
			    ctx->input_decls[idx].Declaration.File != TGSI_FILE_INPUT) {
				ctx->input_decls[idx] = *decl;
				ctx->input_decls[idx].Range.First = idx;
				ctx->input_decls[idx].Range.Last = idx;
				ctx->input_decls[idx].Semantic.Index += idx - decl->Range.First;

				if (si_preload_fs_inputs(ctx) ||
				    bld_base->info->processor != PIPE_SHADER_FRAGMENT)
					ctx->load_input(ctx, idx, &ctx->input_decls[idx],
							&ctx->inputs[idx * 4]);
			}
		}
	}
	break;

	case TGSI_FILE_SYSTEM_VALUE:
	{
		unsigned idx;
		for (idx = decl->Range.First; idx <= decl->Range.Last; idx++) {
			si_load_system_value(ctx, idx, decl);
		}
	}
	break;

	case TGSI_FILE_OUTPUT:
	{
		char name[16] = "";
		unsigned idx;
		for (idx = decl->Range.First; idx <= decl->Range.Last; idx++) {
			unsigned chan;
			assert(idx < RADEON_LLVM_MAX_OUTPUTS);
			if (ctx->outputs[idx][0])
				continue;
			for (chan = 0; chan < TGSI_NUM_CHANNELS; chan++) {
#ifdef DEBUG
				snprintf(name, sizeof(name), "OUT%d.%c",
					 idx, "xyzw"[chan % 4]);
#endif
				ctx->outputs[idx][chan] = ac_build_alloca_undef(
					&ctx->ac, ctx->f32, name);
			}
		}
		break;
	}

	case TGSI_FILE_MEMORY:
		si_tgsi_declare_compute_memory(ctx, decl);
		break;

	default:
		break;
	}
}

void si_llvm_emit_store(struct lp_build_tgsi_context *bld_base,
			const struct tgsi_full_instruction *inst,
			const struct tgsi_opcode_info *info,
			unsigned index,
			LLVMValueRef dst[4])
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	const struct tgsi_full_dst_register *reg = &inst->Dst[index];
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef temp_ptr, temp_ptr2 = NULL;
	bool is_vec_store = false;
	enum tgsi_opcode_type dtype = tgsi_opcode_infer_dst_type(inst->Instruction.Opcode, index);

	if (dst[0]) {
		LLVMTypeKind k = LLVMGetTypeKind(LLVMTypeOf(dst[0]));
		is_vec_store = (k == LLVMVectorTypeKind);
	}

	if (is_vec_store) {
		LLVMValueRef values[4] = {};
		uint32_t writemask = reg->Register.WriteMask;
		while (writemask) {
			unsigned chan = u_bit_scan(&writemask);
			LLVMValueRef index = LLVMConstInt(ctx->i32, chan, 0);
			values[chan]  = LLVMBuildExtractElement(ctx->ac.builder,
							dst[0], index, "");
		}
		bld_base->emit_store(bld_base, inst, info, index, values);
		return;
	}

	uint32_t writemask = reg->Register.WriteMask;
	while (writemask) {
		unsigned chan_index = u_bit_scan(&writemask);
		LLVMValueRef value = dst[chan_index];

		if (tgsi_type_is_64bit(dtype) && (chan_index == 1 || chan_index == 3))
			continue;
		if (inst->Instruction.Saturate)
			value = ac_build_clamp(&ctx->ac, value);

		if (reg->Register.File == TGSI_FILE_ADDRESS) {
			temp_ptr = ctx->addrs[reg->Register.Index][chan_index];
			LLVMBuildStore(builder, value, temp_ptr);
			continue;
		}

		if (!tgsi_type_is_64bit(dtype))
			value = ac_to_float(&ctx->ac, value);

		if (reg->Register.Indirect) {
			unsigned file = reg->Register.File;
			unsigned reg_index = reg->Register.Index;
			store_value_to_array(bld_base, value, file, chan_index,
					     reg_index, &reg->Indirect);
		} else {
			switch(reg->Register.File) {
			case TGSI_FILE_OUTPUT:
				temp_ptr = ctx->outputs[reg->Register.Index][chan_index];
				if (tgsi_type_is_64bit(dtype))
					temp_ptr2 = ctx->outputs[reg->Register.Index][chan_index + 1];
				break;

			case TGSI_FILE_TEMPORARY:
			{
				if (reg->Register.Index >= ctx->temps_count)
					continue;

				temp_ptr = ctx->temps[ TGSI_NUM_CHANNELS * reg->Register.Index + chan_index];
				if (tgsi_type_is_64bit(dtype))
					temp_ptr2 = ctx->temps[ TGSI_NUM_CHANNELS * reg->Register.Index + chan_index + 1];

				break;
			}
			default:
				return;
			}
			if (!tgsi_type_is_64bit(dtype))
				LLVMBuildStore(builder, value, temp_ptr);
			else {
				LLVMValueRef ptr = LLVMBuildBitCast(builder, value,
								    LLVMVectorType(ctx->i32, 2), "");
				LLVMValueRef val2;
				value = LLVMBuildExtractElement(builder, ptr,
								ctx->i32_0, "");
				val2 = LLVMBuildExtractElement(builder, ptr,
							       ctx->i32_1, "");

				LLVMBuildStore(builder, ac_to_float(&ctx->ac, value), temp_ptr);
				LLVMBuildStore(builder, ac_to_float(&ctx->ac, val2), temp_ptr2);
			}
		}
	}
}

static int get_line(int pc)
{
	/* Subtract 1 so that the number shown is that of the corresponding
	 * opcode in the TGSI dump, e.g. an if block has the same suffix as
	 * the instruction number of the corresponding TGSI IF.
	 */
	return pc - 1;
}

static void bgnloop_emit(const struct lp_build_tgsi_action *action,
			 struct lp_build_tgsi_context *bld_base,
			 struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	ac_build_bgnloop(&ctx->ac, get_line(bld_base->pc));
}

static void brk_emit(const struct lp_build_tgsi_action *action,
		     struct lp_build_tgsi_context *bld_base,
		     struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	ac_build_break(&ctx->ac);
}

static void cont_emit(const struct lp_build_tgsi_action *action,
		      struct lp_build_tgsi_context *bld_base,
		      struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	ac_build_continue(&ctx->ac);
}

static void else_emit(const struct lp_build_tgsi_action *action,
		      struct lp_build_tgsi_context *bld_base,
		      struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	ac_build_else(&ctx->ac, get_line(bld_base->pc));
}

static void endif_emit(const struct lp_build_tgsi_action *action,
		       struct lp_build_tgsi_context *bld_base,
		       struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	ac_build_endif(&ctx->ac, get_line(bld_base->pc));
}

static void endloop_emit(const struct lp_build_tgsi_action *action,
			 struct lp_build_tgsi_context *bld_base,
			 struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	ac_build_endloop(&ctx->ac, get_line(bld_base->pc));
}

static void if_emit(const struct lp_build_tgsi_action *action,
		    struct lp_build_tgsi_context *bld_base,
		    struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	ac_build_if(&ctx->ac, emit_data->args[0], get_line(bld_base->pc));
}

static void uif_emit(const struct lp_build_tgsi_action *action,
		     struct lp_build_tgsi_context *bld_base,
		     struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	ac_build_uif(&ctx->ac, emit_data->args[0], get_line(bld_base->pc));
}

static void emit_immediate(struct lp_build_tgsi_context *bld_base,
			   const struct tgsi_full_immediate *imm)
{
	unsigned i;
	struct si_shader_context *ctx = si_shader_context(bld_base);

	for (i = 0; i < 4; ++i) {
		ctx->imms[ctx->imms_num * TGSI_NUM_CHANNELS + i] =
				LLVMConstInt(ctx->i32, imm->u[i].Uint, false   );
	}

	ctx->imms_num++;
}

void si_llvm_context_init(struct si_shader_context *ctx,
			  struct si_screen *sscreen,
			  struct ac_llvm_compiler *compiler)
{
	struct lp_type type;

	/* Initialize the gallivm object:
	 * We are only using the module, context, and builder fields of this struct.
	 * This should be enough for us to be able to pass our gallivm struct to the
	 * helper functions in the gallivm module.
	 */
	memset(ctx, 0, sizeof(*ctx));
	ctx->screen = sscreen;
	ctx->compiler = compiler;

	ac_llvm_context_init(&ctx->ac, sscreen->info.chip_class, sscreen->info.family);
	ctx->ac.module = ac_create_module(compiler->tm, ctx->ac.context);

	enum ac_float_mode float_mode =
		sscreen->debug_flags & DBG(UNSAFE_MATH) ?
			AC_FLOAT_MODE_UNSAFE_FP_MATH :
			AC_FLOAT_MODE_NO_SIGNED_ZEROS_FP_MATH;
	ctx->ac.builder = ac_create_builder(ctx->ac.context, float_mode);

	ctx->gallivm.context = ctx->ac.context;
	ctx->gallivm.module = ctx->ac.module;
	ctx->gallivm.builder = ctx->ac.builder;

	struct lp_build_tgsi_context *bld_base = &ctx->bld_base;

	type.floating = true;
	type.fixed = false;
	type.sign = true;
	type.norm = false;
	type.width = 32;
	type.length = 1;

	lp_build_context_init(&bld_base->base, &ctx->gallivm, type);
	lp_build_context_init(&ctx->bld_base.uint_bld, &ctx->gallivm, lp_uint_type(type));
	lp_build_context_init(&ctx->bld_base.int_bld, &ctx->gallivm, lp_int_type(type));
	type.width *= 2;
	lp_build_context_init(&ctx->bld_base.dbl_bld, &ctx->gallivm, type);
	lp_build_context_init(&ctx->bld_base.uint64_bld, &ctx->gallivm, lp_uint_type(type));
	lp_build_context_init(&ctx->bld_base.int64_bld, &ctx->gallivm, lp_int_type(type));

	bld_base->soa = 1;
	bld_base->emit_swizzle = emit_swizzle;
	bld_base->emit_declaration = emit_declaration;
	bld_base->emit_immediate = emit_immediate;

	bld_base->op_actions[TGSI_OPCODE_BGNLOOP].emit = bgnloop_emit;
	bld_base->op_actions[TGSI_OPCODE_BRK].emit = brk_emit;
	bld_base->op_actions[TGSI_OPCODE_CONT].emit = cont_emit;
	bld_base->op_actions[TGSI_OPCODE_IF].emit = if_emit;
	bld_base->op_actions[TGSI_OPCODE_UIF].emit = uif_emit;
	bld_base->op_actions[TGSI_OPCODE_ELSE].emit = else_emit;
	bld_base->op_actions[TGSI_OPCODE_ENDIF].emit = endif_emit;
	bld_base->op_actions[TGSI_OPCODE_ENDLOOP].emit = endloop_emit;

	si_shader_context_init_alu(&ctx->bld_base);
	si_shader_context_init_mem(ctx);

	ctx->voidt = LLVMVoidTypeInContext(ctx->ac.context);
	ctx->i1 = LLVMInt1TypeInContext(ctx->ac.context);
	ctx->i8 = LLVMInt8TypeInContext(ctx->ac.context);
	ctx->i32 = LLVMInt32TypeInContext(ctx->ac.context);
	ctx->i64 = LLVMInt64TypeInContext(ctx->ac.context);
	ctx->i128 = LLVMIntTypeInContext(ctx->ac.context, 128);
	ctx->f32 = LLVMFloatTypeInContext(ctx->ac.context);
	ctx->v2i32 = LLVMVectorType(ctx->i32, 2);
	ctx->v4i32 = LLVMVectorType(ctx->i32, 4);
	ctx->v4f32 = LLVMVectorType(ctx->f32, 4);
	ctx->v8i32 = LLVMVectorType(ctx->i32, 8);

	ctx->i32_0 = LLVMConstInt(ctx->i32, 0, 0);
	ctx->i32_1 = LLVMConstInt(ctx->i32, 1, 0);
	ctx->i1false = LLVMConstInt(ctx->i1, 0, 0);
	ctx->i1true = LLVMConstInt(ctx->i1, 1, 0);
}

/* Set the context to a certain TGSI shader. Can be called repeatedly
 * to change the shader. */
void si_llvm_context_set_tgsi(struct si_shader_context *ctx,
			      struct si_shader *shader)
{
	const struct tgsi_shader_info *info = NULL;
	const struct tgsi_token *tokens = NULL;

	if (shader && shader->selector) {
		info = &shader->selector->info;
		tokens = shader->selector->tokens;
	}

	ctx->shader = shader;
	ctx->type = info ? info->processor : -1;
	ctx->bld_base.info = info;

	/* Clean up the old contents. */
	FREE(ctx->temp_arrays);
	ctx->temp_arrays = NULL;
	FREE(ctx->temp_array_allocas);
	ctx->temp_array_allocas = NULL;

	FREE(ctx->imms);
	ctx->imms = NULL;
	ctx->imms_num = 0;

	FREE(ctx->temps);
	ctx->temps = NULL;
	ctx->temps_count = 0;

	if (!info)
		return;

	ctx->num_const_buffers = util_last_bit(info->const_buffers_declared);
	ctx->num_shader_buffers = util_last_bit(info->shader_buffers_declared);

	ctx->num_samplers = util_last_bit(info->samplers_declared);
	ctx->num_images = util_last_bit(info->images_declared);

	if (!tokens)
		return;

	if (info->array_max[TGSI_FILE_TEMPORARY] > 0) {
		int size = info->array_max[TGSI_FILE_TEMPORARY];

		ctx->temp_arrays = CALLOC(size, sizeof(ctx->temp_arrays[0]));
		ctx->temp_array_allocas = CALLOC(size, sizeof(ctx->temp_array_allocas[0]));

		tgsi_scan_arrays(tokens, TGSI_FILE_TEMPORARY, size,
				 ctx->temp_arrays);
	}
	if (info->file_max[TGSI_FILE_IMMEDIATE] >= 0) {
		int size = info->file_max[TGSI_FILE_IMMEDIATE] + 1;
		ctx->imms = MALLOC(size * TGSI_NUM_CHANNELS * sizeof(LLVMValueRef));
	}

	/* Re-set these to start with a clean slate. */
	ctx->bld_base.num_instructions = 0;
	ctx->bld_base.pc = 0;
	memset(ctx->outputs, 0, sizeof(ctx->outputs));

	ctx->bld_base.emit_store = si_llvm_emit_store;
	ctx->bld_base.emit_fetch_funcs[TGSI_FILE_IMMEDIATE] = si_llvm_emit_fetch;
	ctx->bld_base.emit_fetch_funcs[TGSI_FILE_INPUT] = si_llvm_emit_fetch;
	ctx->bld_base.emit_fetch_funcs[TGSI_FILE_TEMPORARY] = si_llvm_emit_fetch;
	ctx->bld_base.emit_fetch_funcs[TGSI_FILE_OUTPUT] = si_llvm_emit_fetch;
	ctx->bld_base.emit_fetch_funcs[TGSI_FILE_SYSTEM_VALUE] = fetch_system_value;
}

void si_llvm_create_func(struct si_shader_context *ctx,
			 const char *name,
			 LLVMTypeRef *return_types, unsigned num_return_elems,
			 LLVMTypeRef *ParamTypes, unsigned ParamCount)
{
	LLVMTypeRef main_fn_type, ret_type;
	LLVMBasicBlockRef main_fn_body;
	enum si_llvm_calling_convention call_conv;
	unsigned real_shader_type;

	if (num_return_elems)
		ret_type = LLVMStructTypeInContext(ctx->ac.context,
						   return_types,
						   num_return_elems, true);
	else
		ret_type = ctx->voidt;

	/* Setup the function */
	ctx->return_type = ret_type;
	main_fn_type = LLVMFunctionType(ret_type, ParamTypes, ParamCount, 0);
	ctx->main_fn = LLVMAddFunction(ctx->gallivm.module, name, main_fn_type);
	main_fn_body = LLVMAppendBasicBlockInContext(ctx->ac.context,
			ctx->main_fn, "main_body");
	LLVMPositionBuilderAtEnd(ctx->ac.builder, main_fn_body);

	real_shader_type = ctx->type;

	/* LS is merged into HS (TCS), and ES is merged into GS. */
	if (ctx->screen->info.chip_class >= GFX9) {
		if (ctx->shader->key.as_ls)
			real_shader_type = PIPE_SHADER_TESS_CTRL;
		else if (ctx->shader->key.as_es)
			real_shader_type = PIPE_SHADER_GEOMETRY;
	}

	switch (real_shader_type) {
	case PIPE_SHADER_VERTEX:
	case PIPE_SHADER_TESS_EVAL:
		call_conv = RADEON_LLVM_AMDGPU_VS;
		break;
	case PIPE_SHADER_TESS_CTRL:
		call_conv = RADEON_LLVM_AMDGPU_HS;
		break;
	case PIPE_SHADER_GEOMETRY:
		call_conv = RADEON_LLVM_AMDGPU_GS;
		break;
	case PIPE_SHADER_FRAGMENT:
		call_conv = RADEON_LLVM_AMDGPU_PS;
		break;
	case PIPE_SHADER_COMPUTE:
		call_conv = RADEON_LLVM_AMDGPU_CS;
		break;
	default:
		unreachable("Unhandle shader type");
	}

	LLVMSetFunctionCallConv(ctx->main_fn, call_conv);
}

void si_llvm_optimize_module(struct si_shader_context *ctx)
{
	/* Dump LLVM IR before any optimization passes */
	if (ctx->screen->debug_flags & DBG(PREOPT_IR) &&
	    si_can_dump_shader(ctx->screen, ctx->type))
		LLVMDumpModule(ctx->gallivm.module);

	/* Run the pass */
	LLVMRunPassManager(ctx->compiler->passmgr, ctx->gallivm.module);
	LLVMDisposeBuilder(ctx->ac.builder);
}

void si_llvm_dispose(struct si_shader_context *ctx)
{
	LLVMDisposeModule(ctx->gallivm.module);
	LLVMContextDispose(ctx->gallivm.context);
	FREE(ctx->temp_arrays);
	ctx->temp_arrays = NULL;
	FREE(ctx->temp_array_allocas);
	ctx->temp_array_allocas = NULL;
	FREE(ctx->temps);
	ctx->temps = NULL;
	ctx->temps_count = 0;
	FREE(ctx->imms);
	ctx->imms = NULL;
	ctx->imms_num = 0;
	ac_llvm_context_dispose(&ctx->ac);
}
