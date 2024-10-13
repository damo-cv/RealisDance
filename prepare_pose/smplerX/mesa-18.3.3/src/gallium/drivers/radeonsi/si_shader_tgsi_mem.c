/*
 * Copyright 2017 Advanced Micro Devices, Inc.
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
#include "sid.h"
#include "tgsi/tgsi_build.h"
#include "tgsi/tgsi_util.h"
#include "ac_llvm_util.h"

static void tex_fetch_ptrs(struct lp_build_tgsi_context *bld_base,
			   struct lp_build_emit_data *emit_data,
			   LLVMValueRef *res_ptr, LLVMValueRef *samp_ptr,
			   LLVMValueRef *fmask_ptr);

/**
 * Given a v8i32 resource descriptor for a buffer, extract the size of the
 * buffer in number of elements and return it as an i32.
 */
static LLVMValueRef get_buffer_size(
	struct lp_build_tgsi_context *bld_base,
	LLVMValueRef descriptor)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef size =
		LLVMBuildExtractElement(builder, descriptor,
					LLVMConstInt(ctx->i32, 2, 0), "");

	if (ctx->screen->info.chip_class == VI) {
		/* On VI, the descriptor contains the size in bytes,
		 * but TXQ must return the size in elements.
		 * The stride is always non-zero for resources using TXQ.
		 */
		LLVMValueRef stride =
			LLVMBuildExtractElement(builder, descriptor,
						ctx->i32_1, "");
		stride = LLVMBuildLShr(builder, stride,
				       LLVMConstInt(ctx->i32, 16, 0), "");
		stride = LLVMBuildAnd(builder, stride,
				      LLVMConstInt(ctx->i32, 0x3FFF, 0), "");

		size = LLVMBuildUDiv(builder, size, stride, "");
	}

	return size;
}

static LLVMValueRef
shader_buffer_fetch_rsrc(struct si_shader_context *ctx,
			 const struct tgsi_full_src_register *reg,
			 bool ubo)
{
	LLVMValueRef index;

	if (!reg->Register.Indirect) {
		index = LLVMConstInt(ctx->i32, reg->Register.Index, false);
	} else {
		index = si_get_indirect_index(ctx, &reg->Indirect,
					      1, reg->Register.Index);
	}

	if (ubo)
		return ctx->abi.load_ubo(&ctx->abi, index);
	else
		return ctx->abi.load_ssbo(&ctx->abi, index, false);
}

static enum ac_image_dim
ac_texture_dim_from_tgsi_target(struct si_screen *screen, enum tgsi_texture_type target)
{
	switch (target) {
	case TGSI_TEXTURE_1D:
	case TGSI_TEXTURE_SHADOW1D:
		if (screen->info.chip_class >= GFX9)
			return ac_image_2d;
		return ac_image_1d;
	case TGSI_TEXTURE_2D:
	case TGSI_TEXTURE_SHADOW2D:
	case TGSI_TEXTURE_RECT:
	case TGSI_TEXTURE_SHADOWRECT:
		return ac_image_2d;
	case TGSI_TEXTURE_3D:
		return ac_image_3d;
	case TGSI_TEXTURE_CUBE:
	case TGSI_TEXTURE_SHADOWCUBE:
	case TGSI_TEXTURE_CUBE_ARRAY:
	case TGSI_TEXTURE_SHADOWCUBE_ARRAY:
		return ac_image_cube;
	case TGSI_TEXTURE_1D_ARRAY:
	case TGSI_TEXTURE_SHADOW1D_ARRAY:
		if (screen->info.chip_class >= GFX9)
			return ac_image_2darray;
		return ac_image_1darray;
	case TGSI_TEXTURE_2D_ARRAY:
	case TGSI_TEXTURE_SHADOW2D_ARRAY:
		return ac_image_2darray;
	case TGSI_TEXTURE_2D_MSAA:
		return ac_image_2dmsaa;
	case TGSI_TEXTURE_2D_ARRAY_MSAA:
		return ac_image_2darraymsaa;
	default:
		unreachable("unhandled texture type");
	}
}

static enum ac_image_dim
ac_image_dim_from_tgsi_target(struct si_screen *screen, enum tgsi_texture_type target)
{
	enum ac_image_dim dim = ac_texture_dim_from_tgsi_target(screen, target);

	/* Match the resource type set in the descriptor. */
	if (dim == ac_image_cube ||
	    (screen->info.chip_class <= VI && dim == ac_image_3d))
		dim = ac_image_2darray;
	else if (target == TGSI_TEXTURE_2D && screen->info.chip_class >= GFX9) {
		/* When a single layer of a 3D texture is bound, the shader
		 * will refer to a 2D target, but the descriptor has a 3D type.
		 * Since the HW ignores BASE_ARRAY in this case, we need to
		 * send 3 coordinates. This doesn't hurt when the underlying
		 * texture is non-3D.
		 */
		dim = ac_image_3d;
	}

	return dim;
}

/**
 * Given a 256-bit resource descriptor, force the DCC enable bit to off.
 *
 * At least on Tonga, executing image stores on images with DCC enabled and
 * non-trivial can eventually lead to lockups. This can occur when an
 * application binds an image as read-only but then uses a shader that writes
 * to it. The OpenGL spec allows almost arbitrarily bad behavior (including
 * program termination) in this case, but it doesn't cost much to be a bit
 * nicer: disabling DCC in the shader still leads to undefined results but
 * avoids the lockup.
 */
static LLVMValueRef force_dcc_off(struct si_shader_context *ctx,
				  LLVMValueRef rsrc)
{
	if (ctx->screen->info.chip_class <= CIK) {
		return rsrc;
	} else {
		LLVMValueRef i32_6 = LLVMConstInt(ctx->i32, 6, 0);
		LLVMValueRef i32_C = LLVMConstInt(ctx->i32, C_008F28_COMPRESSION_EN, 0);
		LLVMValueRef tmp;

		tmp = LLVMBuildExtractElement(ctx->ac.builder, rsrc, i32_6, "");
		tmp = LLVMBuildAnd(ctx->ac.builder, tmp, i32_C, "");
		return LLVMBuildInsertElement(ctx->ac.builder, rsrc, tmp, i32_6, "");
	}
}

LLVMValueRef si_load_image_desc(struct si_shader_context *ctx,
				LLVMValueRef list, LLVMValueRef index,
				enum ac_descriptor_type desc_type, bool dcc_off,
				bool bindless)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef rsrc;

	if (desc_type == AC_DESC_BUFFER) {
		index = ac_build_imad(&ctx->ac, index, LLVMConstInt(ctx->i32, 2, 0),
				      ctx->i32_1);
		list = LLVMBuildPointerCast(builder, list,
					    ac_array_in_const32_addr_space(ctx->v4i32), "");
	} else {
		assert(desc_type == AC_DESC_IMAGE);
	}

	if (bindless)
		rsrc = ac_build_load_to_sgpr_uint_wraparound(&ctx->ac, list, index);
	else
		rsrc = ac_build_load_to_sgpr(&ctx->ac, list, index);

	if (desc_type == AC_DESC_IMAGE && dcc_off)
		rsrc = force_dcc_off(ctx, rsrc);
	return rsrc;
}

/**
 * Load the resource descriptor for \p image.
 */
static void
image_fetch_rsrc(
	struct lp_build_tgsi_context *bld_base,
	const struct tgsi_full_src_register *image,
	bool is_store, unsigned target,
	LLVMValueRef *rsrc)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef rsrc_ptr = LLVMGetParam(ctx->main_fn,
					     ctx->param_samplers_and_images);
	LLVMValueRef index;
	bool dcc_off = is_store;

	if (!image->Register.Indirect) {
		const struct tgsi_shader_info *info = bld_base->info;
		unsigned images_writemask = info->images_store |
					    info->images_atomic;

		index = LLVMConstInt(ctx->i32,
				     si_get_image_slot(image->Register.Index), 0);

		if (images_writemask & (1 << image->Register.Index))
			dcc_off = true;
	} else {
		/* From the GL_ARB_shader_image_load_store extension spec:
		 *
		 *    If a shader performs an image load, store, or atomic
		 *    operation using an image variable declared as an array,
		 *    and if the index used to select an individual element is
		 *    negative or greater than or equal to the size of the
		 *    array, the results of the operation are undefined but may
		 *    not lead to termination.
		 */
		index = si_get_bounded_indirect_index(ctx, &image->Indirect,
						      image->Register.Index,
						      ctx->num_images);
		index = LLVMBuildSub(ctx->ac.builder,
				     LLVMConstInt(ctx->i32, SI_NUM_IMAGES - 1, 0),
				     index, "");
	}

	bool bindless = false;

	if (image->Register.File != TGSI_FILE_IMAGE) {
		/* Bindless descriptors are accessible from a different pair of
		 * user SGPR indices.
		 */
		rsrc_ptr = LLVMGetParam(ctx->main_fn,
					ctx->param_bindless_samplers_and_images);
		index = lp_build_emit_fetch_src(bld_base, image,
						TGSI_TYPE_UNSIGNED, 0);

		/* For simplicity, bindless image descriptors use fixed
		 * 16-dword slots for now.
		 */
		index = LLVMBuildMul(ctx->ac.builder, index,
				     LLVMConstInt(ctx->i32, 2, 0), "");
		bindless = true;
	}

	*rsrc = si_load_image_desc(ctx, rsrc_ptr, index,
				   target == TGSI_TEXTURE_BUFFER ? AC_DESC_BUFFER : AC_DESC_IMAGE,
				   dcc_off, bindless);
}

static void image_fetch_coords(
		struct lp_build_tgsi_context *bld_base,
		const struct tgsi_full_instruction *inst,
		unsigned src, LLVMValueRef desc,
		LLVMValueRef *coords)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMBuilderRef builder = ctx->ac.builder;
	unsigned target = inst->Memory.Texture;
	unsigned num_coords = tgsi_util_get_texture_coord_dim(target);
	LLVMValueRef tmp;
	int chan;

	if (target == TGSI_TEXTURE_2D_MSAA ||
	    target == TGSI_TEXTURE_2D_ARRAY_MSAA) {
		/* Need the sample index as well. */
		num_coords++;
	}

	for (chan = 0; chan < num_coords; ++chan) {
		tmp = lp_build_emit_fetch(bld_base, inst, src, chan);
		tmp = ac_to_integer(&ctx->ac, tmp);
		coords[chan] = tmp;
	}

	if (ctx->screen->info.chip_class >= GFX9) {
		/* 1D textures are allocated and used as 2D on GFX9. */
		if (target == TGSI_TEXTURE_1D) {
			coords[1] = ctx->i32_0;
		} else if (target == TGSI_TEXTURE_1D_ARRAY) {
			coords[2] = coords[1];
			coords[1] = ctx->i32_0;
		} else if (target == TGSI_TEXTURE_2D) {
			/* The hw can't bind a slice of a 3D image as a 2D
			 * image, because it ignores BASE_ARRAY if the target
			 * is 3D. The workaround is to read BASE_ARRAY and set
			 * it as the 3rd address operand for all 2D images.
			 */
			LLVMValueRef first_layer, const5, mask;

			const5 = LLVMConstInt(ctx->i32, 5, 0);
			mask = LLVMConstInt(ctx->i32, S_008F24_BASE_ARRAY(~0), 0);
			first_layer = LLVMBuildExtractElement(builder, desc, const5, "");
			first_layer = LLVMBuildAnd(builder, first_layer, mask, "");

			coords[2] = first_layer;
		}
	}
}

static unsigned get_cache_policy(struct si_shader_context *ctx,
				 const struct tgsi_full_instruction *inst,
				 bool atomic, bool may_store_unaligned,
				 bool writeonly_memory)
{
	unsigned cache_policy = 0;

	if (!atomic &&
	    /* SI has a TC L1 bug causing corruption of 8bit/16bit stores.
	     * All store opcodes not aligned to a dword are affected.
	     * The only way to get unaligned stores in radeonsi is through
	     * shader images. */
	    ((may_store_unaligned && ctx->screen->info.chip_class == SI) ||
	     /* If this is write-only, don't keep data in L1 to prevent
	      * evicting L1 cache lines that may be needed by other
	      * instructions. */
	     writeonly_memory ||
	     inst->Memory.Qualifier & (TGSI_MEMORY_COHERENT | TGSI_MEMORY_VOLATILE)))
		cache_policy |= ac_glc;

	if (inst->Memory.Qualifier & TGSI_MEMORY_STREAM_CACHE_POLICY)
		cache_policy |= ac_slc;

	return cache_policy;
}

static LLVMValueRef get_memory_ptr(struct si_shader_context *ctx,
                                   const struct tgsi_full_instruction *inst,
                                   LLVMTypeRef type, int arg)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef offset, ptr;
	int addr_space;

	offset = lp_build_emit_fetch(&ctx->bld_base, inst, arg, 0);
	offset = ac_to_integer(&ctx->ac, offset);

	ptr = ctx->ac.lds;
	ptr = LLVMBuildGEP(builder, ptr, &offset, 1, "");
	addr_space = LLVMGetPointerAddressSpace(LLVMTypeOf(ptr));
	ptr = LLVMBuildBitCast(builder, ptr, LLVMPointerType(type, addr_space), "");

	return ptr;
}

static void load_emit_memory(
		struct si_shader_context *ctx,
		struct lp_build_emit_data *emit_data)
{
	const struct tgsi_full_instruction *inst = emit_data->inst;
	unsigned writemask = inst->Dst[0].Register.WriteMask;
	LLVMValueRef channels[4], ptr, derived_ptr, index;
	int chan;

	ptr = get_memory_ptr(ctx, inst, ctx->f32, 1);

	for (chan = 0; chan < 4; ++chan) {
		if (!(writemask & (1 << chan))) {
			channels[chan] = LLVMGetUndef(ctx->f32);
			continue;
		}

		index = LLVMConstInt(ctx->i32, chan, 0);
		derived_ptr = LLVMBuildGEP(ctx->ac.builder, ptr, &index, 1, "");
		channels[chan] = LLVMBuildLoad(ctx->ac.builder, derived_ptr, "");
	}
	emit_data->output[emit_data->chan] = ac_build_gather_values(&ctx->ac, channels, 4);
}

/**
 * Return true if the memory accessed by a LOAD or STORE instruction is
 * read-only or write-only, respectively.
 *
 * \param shader_buffers_reverse_access_mask
 *	For LOAD, set this to (store | atomic) slot usage in the shader.
 *	For STORE, set this to (load | atomic) slot usage in the shader.
 * \param images_reverse_access_mask  Same as above, but for images.
 */
static bool is_oneway_access_only(const struct tgsi_full_instruction *inst,
				  const struct tgsi_shader_info *info,
				  unsigned shader_buffers_reverse_access_mask,
				  unsigned images_reverse_access_mask)
{
	/* RESTRICT means NOALIAS.
	 * If there are no writes, we can assume the accessed memory is read-only.
	 * If there are no reads, we can assume the accessed memory is write-only.
	 */
	if (inst->Memory.Qualifier & TGSI_MEMORY_RESTRICT) {
		unsigned reverse_access_mask;

		if (inst->Src[0].Register.File == TGSI_FILE_BUFFER) {
			reverse_access_mask = shader_buffers_reverse_access_mask;
		} else if (inst->Memory.Texture == TGSI_TEXTURE_BUFFER) {
			reverse_access_mask = info->images_buffers &
					      images_reverse_access_mask;
		} else {
			reverse_access_mask = ~info->images_buffers &
					      images_reverse_access_mask;
		}

		if (inst->Src[0].Register.Indirect) {
			if (!reverse_access_mask)
				return true;
		} else {
			if (!(reverse_access_mask &
			      (1u << inst->Src[0].Register.Index)))
				return true;
		}
	}

	/* If there are no buffer writes (for both shader buffers & image
	 * buffers), it implies that buffer memory is read-only.
	 * If there are no buffer reads (for both shader buffers & image
	 * buffers), it implies that buffer memory is write-only.
	 *
	 * Same for the case when there are no writes/reads for non-buffer
	 * images.
	 */
	if (inst->Src[0].Register.File == TGSI_FILE_BUFFER ||
	    (inst->Memory.Texture == TGSI_TEXTURE_BUFFER &&
	     (inst->Src[0].Register.File == TGSI_FILE_IMAGE ||
	      tgsi_is_bindless_image_file(inst->Src[0].Register.File)))) {
		if (!shader_buffers_reverse_access_mask &&
		    !(info->images_buffers & images_reverse_access_mask))
			return true;
	} else {
		if (!(~info->images_buffers & images_reverse_access_mask))
			return true;
	}
	return false;
}

static void load_emit(
		const struct lp_build_tgsi_action *action,
		struct lp_build_tgsi_context *bld_base,
		struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	const struct tgsi_full_instruction * inst = emit_data->inst;
	const struct tgsi_shader_info *info = &ctx->shader->selector->info;
	bool can_speculate = false;
	LLVMValueRef vindex = ctx->i32_0;
	LLVMValueRef voffset = ctx->i32_0;
	struct ac_image_args args = {};

	if (inst->Src[0].Register.File == TGSI_FILE_MEMORY) {
		load_emit_memory(ctx, emit_data);
		return;
	}

	if (inst->Src[0].Register.File == TGSI_FILE_BUFFER ||
	    inst->Src[0].Register.File == TGSI_FILE_CONSTBUF) {
		bool ubo = inst->Src[0].Register.File == TGSI_FILE_CONSTBUF;
		args.resource = shader_buffer_fetch_rsrc(ctx, &inst->Src[0], ubo);
		voffset = ac_to_integer(&ctx->ac, lp_build_emit_fetch(bld_base, inst, 1, 0));
	} else if (inst->Src[0].Register.File == TGSI_FILE_IMAGE ||
		   tgsi_is_bindless_image_file(inst->Src[0].Register.File)) {
		unsigned target = inst->Memory.Texture;

		image_fetch_rsrc(bld_base, &inst->Src[0], false, target, &args.resource);
		image_fetch_coords(bld_base, inst, 1, args.resource, args.coords);
		vindex = args.coords[0]; /* for buffers only */
	}

	if (inst->Src[0].Register.File == TGSI_FILE_CONSTBUF) {
		emit_data->output[emit_data->chan] =
			ac_build_buffer_load(&ctx->ac, args.resource,
					     util_last_bit(inst->Dst[0].Register.WriteMask),
					     NULL, voffset, NULL, 0, 0, 0, true, true);
		return;
	}

	if (inst->Memory.Qualifier & TGSI_MEMORY_VOLATILE)
		ac_build_waitcnt(&ctx->ac, VM_CNT);

	can_speculate = !(inst->Memory.Qualifier & TGSI_MEMORY_VOLATILE) &&
			  is_oneway_access_only(inst, info,
						info->shader_buffers_store |
						info->shader_buffers_atomic,
						info->images_store |
						info->images_atomic);
	args.cache_policy = get_cache_policy(ctx, inst, false, false, false);

	if (inst->Src[0].Register.File == TGSI_FILE_BUFFER) {
		/* Don't use SMEM for shader buffer loads, because LLVM doesn't
		 * select SMEM for SI.load.const with a non-constant offset, and
		 * constant offsets practically don't exist with shader buffers.
		 *
		 * Also, SI.load.const doesn't use inst_offset when it's lowered
		 * to VMEM, so we just end up with more VALU instructions in the end
		 * and no benefit.
		 *
		 * TODO: Remove this line once LLVM can select SMEM with a non-constant
		 *       offset, and can derive inst_offset when VMEM is selected.
		 *       After that, si_memory_barrier should invalidate sL1 for shader
		 *       buffers.
		 */
		emit_data->output[emit_data->chan] =
			ac_build_buffer_load(&ctx->ac, args.resource,
					     util_last_bit(inst->Dst[0].Register.WriteMask),
					     NULL, voffset, NULL, 0,
					     !!(args.cache_policy & ac_glc),
					     !!(args.cache_policy & ac_slc),
					     can_speculate, false);
		return;
	}

	if (inst->Memory.Texture == TGSI_TEXTURE_BUFFER) {
		unsigned num_channels = util_last_bit(inst->Dst[0].Register.WriteMask);
		LLVMValueRef result =
			ac_build_buffer_load_format(&ctx->ac,
						    args.resource,
						    vindex,
						    ctx->i32_0,
						    num_channels,
						    !!(args.cache_policy & ac_glc),
						    can_speculate);
		emit_data->output[emit_data->chan] =
			ac_build_expand_to_vec4(&ctx->ac, result, num_channels);
	} else {
		args.opcode = ac_image_load;
		args.dim = ac_image_dim_from_tgsi_target(ctx->screen, inst->Memory.Texture);
		args.attributes = ac_get_load_intr_attribs(can_speculate);
		args.dmask = 0xf;

		emit_data->output[emit_data->chan] =
			ac_build_image_opcode(&ctx->ac, &args);
	}
}

static void store_emit_buffer(struct si_shader_context *ctx,
			      LLVMValueRef resource,
			      unsigned writemask,
			      LLVMValueRef value,
			      LLVMValueRef voffset,
			      unsigned cache_policy,
			      bool writeonly_memory)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef base_data = value;
	LLVMValueRef base_offset = voffset;

	while (writemask) {
		int start, count;
		const char *intrinsic_name;
		LLVMValueRef data, voff;

		u_bit_scan_consecutive_range(&writemask, &start, &count);

		/* Due to an LLVM limitation, split 3-element writes
		 * into a 2-element and a 1-element write. */
		if (count == 3) {
			writemask |= 1 << (start + 2);
			count = 2;
		}

		if (count == 4) {
			data = base_data;
			intrinsic_name = "llvm.amdgcn.buffer.store.v4f32";
		} else if (count == 2) {
			LLVMValueRef values[2] = {
				LLVMBuildExtractElement(builder, base_data,
							LLVMConstInt(ctx->i32, start, 0), ""),
				LLVMBuildExtractElement(builder, base_data,
							LLVMConstInt(ctx->i32, start + 1, 0), ""),
			};

			data = ac_build_gather_values(&ctx->ac, values, 2);
			intrinsic_name = "llvm.amdgcn.buffer.store.v2f32";
		} else {
			assert(count == 1);
			data = LLVMBuildExtractElement(
				builder, base_data,
				LLVMConstInt(ctx->i32, start, 0), "");
			intrinsic_name = "llvm.amdgcn.buffer.store.f32";
		}

		voff = base_offset;
		if (start != 0) {
			voff = LLVMBuildAdd(
				builder, voff,
				LLVMConstInt(ctx->i32, start * 4, 0), "");
		}

		LLVMValueRef args[] = {
			data,
			resource,
			ctx->i32_0, /* vindex */
			voff,
			LLVMConstInt(ctx->i1, !!(cache_policy & ac_glc), 0),
			LLVMConstInt(ctx->i1, !!(cache_policy & ac_slc), 0),
		};
		ac_build_intrinsic(&ctx->ac, intrinsic_name, ctx->voidt, args, 6,
				   ac_get_store_intr_attribs(writeonly_memory));
	}
}

static void store_emit_memory(
		struct si_shader_context *ctx,
		struct lp_build_emit_data *emit_data)
{
	const struct tgsi_full_instruction *inst = emit_data->inst;
	LLVMBuilderRef builder = ctx->ac.builder;
	unsigned writemask = inst->Dst[0].Register.WriteMask;
	LLVMValueRef ptr, derived_ptr, data, index;
	int chan;

	ptr = get_memory_ptr(ctx, inst, ctx->f32, 0);

	for (chan = 0; chan < 4; ++chan) {
		if (!(writemask & (1 << chan))) {
			continue;
		}
		data = lp_build_emit_fetch(&ctx->bld_base, inst, 1, chan);
		index = LLVMConstInt(ctx->i32, chan, 0);
		derived_ptr = LLVMBuildGEP(builder, ptr, &index, 1, "");
		LLVMBuildStore(builder, data, derived_ptr);
	}
}

static void store_emit(
		const struct lp_build_tgsi_action *action,
		struct lp_build_tgsi_context *bld_base,
		struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	const struct tgsi_full_instruction * inst = emit_data->inst;
	const struct tgsi_shader_info *info = &ctx->shader->selector->info;
	struct tgsi_full_src_register resource_reg =
		tgsi_full_src_register_from_dst(&inst->Dst[0]);
	unsigned target = inst->Memory.Texture;
	bool writeonly_memory = is_oneway_access_only(inst, info,
						      info->shader_buffers_load |
						      info->shader_buffers_atomic,
						      info->images_load |
						      info->images_atomic);
	bool is_image = inst->Dst[0].Register.File == TGSI_FILE_IMAGE ||
			tgsi_is_bindless_image_file(inst->Dst[0].Register.File);
	LLVMValueRef chans[4], value;
	LLVMValueRef vindex = ctx->i32_0;
	LLVMValueRef voffset = ctx->i32_0;
	struct ac_image_args args = {};

	if (inst->Dst[0].Register.File == TGSI_FILE_MEMORY) {
		store_emit_memory(ctx, emit_data);
		return;
	}

	for (unsigned chan = 0; chan < 4; ++chan)
		chans[chan] = lp_build_emit_fetch(bld_base, inst, 1, chan);

	value = ac_build_gather_values(&ctx->ac, chans, 4);

	if (inst->Dst[0].Register.File == TGSI_FILE_BUFFER) {
		args.resource = shader_buffer_fetch_rsrc(ctx, &resource_reg, false);
		voffset = ac_to_integer(&ctx->ac, lp_build_emit_fetch(bld_base, inst, 0, 0));
	} else if (is_image) {
		image_fetch_rsrc(bld_base, &resource_reg, true, target, &args.resource);
		image_fetch_coords(bld_base, inst, 0, args.resource, args.coords);
		vindex = args.coords[0]; /* for buffers only */
	} else {
		unreachable("unexpected register file");
	}

	if (inst->Memory.Qualifier & TGSI_MEMORY_VOLATILE)
		ac_build_waitcnt(&ctx->ac, VM_CNT);

	args.cache_policy = get_cache_policy(ctx, inst,
					     false, /* atomic */
					     is_image, /* may_store_unaligned */
					     writeonly_memory);

	if (inst->Dst[0].Register.File == TGSI_FILE_BUFFER) {
		store_emit_buffer(ctx, args.resource, inst->Dst[0].Register.WriteMask,
				  value, voffset, args.cache_policy, writeonly_memory);
		return;
	}

	if (target == TGSI_TEXTURE_BUFFER) {
		LLVMValueRef buf_args[] = {
			value,
			args.resource,
			vindex,
			ctx->i32_0, /* voffset */
			LLVMConstInt(ctx->i1, !!(args.cache_policy & ac_glc), 0),
			LLVMConstInt(ctx->i1, !!(args.cache_policy & ac_slc), 0),
		};

		emit_data->output[emit_data->chan] = ac_build_intrinsic(
			&ctx->ac, "llvm.amdgcn.buffer.store.format.v4f32",
			ctx->voidt, buf_args, 6,
			ac_get_store_intr_attribs(writeonly_memory));
	} else {
		args.opcode = ac_image_store;
		args.data[0] = value;
		args.dim = ac_image_dim_from_tgsi_target(ctx->screen, inst->Memory.Texture);
		args.attributes = ac_get_store_intr_attribs(writeonly_memory);
		args.dmask = 0xf;

		emit_data->output[emit_data->chan] =
			ac_build_image_opcode(&ctx->ac, &args);
	}
}

static void atomic_emit_memory(struct si_shader_context *ctx,
                               struct lp_build_emit_data *emit_data) {
	LLVMBuilderRef builder = ctx->ac.builder;
	const struct tgsi_full_instruction * inst = emit_data->inst;
	LLVMValueRef ptr, result, arg;

	ptr = get_memory_ptr(ctx, inst, ctx->i32, 1);

	arg = lp_build_emit_fetch(&ctx->bld_base, inst, 2, 0);
	arg = ac_to_integer(&ctx->ac, arg);

	if (inst->Instruction.Opcode == TGSI_OPCODE_ATOMCAS) {
		LLVMValueRef new_data;
		new_data = lp_build_emit_fetch(&ctx->bld_base,
		                               inst, 3, 0);

		new_data = ac_to_integer(&ctx->ac, new_data);

		result = LLVMBuildAtomicCmpXchg(builder, ptr, arg, new_data,
		                       LLVMAtomicOrderingSequentiallyConsistent,
		                       LLVMAtomicOrderingSequentiallyConsistent,
		                       false);

		result = LLVMBuildExtractValue(builder, result, 0, "");
	} else {
		LLVMAtomicRMWBinOp op;

		switch(inst->Instruction.Opcode) {
			case TGSI_OPCODE_ATOMUADD:
				op = LLVMAtomicRMWBinOpAdd;
				break;
			case TGSI_OPCODE_ATOMXCHG:
				op = LLVMAtomicRMWBinOpXchg;
				break;
			case TGSI_OPCODE_ATOMAND:
				op = LLVMAtomicRMWBinOpAnd;
				break;
			case TGSI_OPCODE_ATOMOR:
				op = LLVMAtomicRMWBinOpOr;
				break;
			case TGSI_OPCODE_ATOMXOR:
				op = LLVMAtomicRMWBinOpXor;
				break;
			case TGSI_OPCODE_ATOMUMIN:
				op = LLVMAtomicRMWBinOpUMin;
				break;
			case TGSI_OPCODE_ATOMUMAX:
				op = LLVMAtomicRMWBinOpUMax;
				break;
			case TGSI_OPCODE_ATOMIMIN:
				op = LLVMAtomicRMWBinOpMin;
				break;
			case TGSI_OPCODE_ATOMIMAX:
				op = LLVMAtomicRMWBinOpMax;
				break;
			default:
				unreachable("unknown atomic opcode");
		}

		result = LLVMBuildAtomicRMW(builder, op, ptr, arg,
		                       LLVMAtomicOrderingSequentiallyConsistent,
		                       false);
	}
	emit_data->output[emit_data->chan] =
		LLVMBuildBitCast(builder, result, ctx->f32, "");
}

static void atomic_emit(
		const struct lp_build_tgsi_action *action,
		struct lp_build_tgsi_context *bld_base,
		struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	const struct tgsi_full_instruction * inst = emit_data->inst;
	struct ac_image_args args = {};
	unsigned num_data = 0;
	LLVMValueRef vindex = ctx->i32_0;
	LLVMValueRef voffset = ctx->i32_0;

	if (inst->Src[0].Register.File == TGSI_FILE_MEMORY) {
		atomic_emit_memory(ctx, emit_data);
		return;
	}

	if (inst->Instruction.Opcode == TGSI_OPCODE_ATOMCAS) {
		/* llvm.amdgcn.image/buffer.atomic.cmpswap reflect the hardware order
		 * of arguments, which is reversed relative to TGSI (and GLSL)
		 */
		args.data[num_data++] =
			ac_to_integer(&ctx->ac, lp_build_emit_fetch(bld_base, inst, 3, 0));
	}

	args.data[num_data++] =
		ac_to_integer(&ctx->ac, lp_build_emit_fetch(bld_base, inst, 2, 0));
	args.cache_policy = get_cache_policy(ctx, inst, true, false, false);

	if (inst->Src[0].Register.File == TGSI_FILE_BUFFER) {
		args.resource = shader_buffer_fetch_rsrc(ctx, &inst->Src[0], false);
		voffset = ac_to_integer(&ctx->ac, lp_build_emit_fetch(bld_base, inst, 1, 0));
	} else if (inst->Src[0].Register.File == TGSI_FILE_IMAGE ||
		   tgsi_is_bindless_image_file(inst->Src[0].Register.File)) {
		image_fetch_rsrc(bld_base, &inst->Src[0], true,
				inst->Memory.Texture, &args.resource);
		image_fetch_coords(bld_base, inst, 1, args.resource, args.coords);
		vindex = args.coords[0]; /* for buffers only */
	}

	if (inst->Src[0].Register.File == TGSI_FILE_BUFFER ||
	    inst->Memory.Texture == TGSI_TEXTURE_BUFFER) {
		LLVMValueRef buf_args[7];
		unsigned num_args = 0;

		buf_args[num_args++] = args.data[0];
		if (inst->Instruction.Opcode == TGSI_OPCODE_ATOMCAS)
			buf_args[num_args++] = args.data[1];

		buf_args[num_args++] = args.resource;
		buf_args[num_args++] = vindex;
		buf_args[num_args++] = voffset;
		buf_args[num_args++] = args.cache_policy & ac_slc ? ctx->i1true : ctx->i1false;

		char intrinsic_name[40];
		snprintf(intrinsic_name, sizeof(intrinsic_name),
			 "llvm.amdgcn.buffer.atomic.%s", action->intr_name);
		emit_data->output[emit_data->chan] =
			ac_to_float(&ctx->ac,
				    ac_build_intrinsic(&ctx->ac, intrinsic_name,
						       ctx->i32, buf_args, num_args, 0));
	} else {
		if (inst->Instruction.Opcode == TGSI_OPCODE_ATOMCAS) {
			args.opcode = ac_image_atomic_cmpswap;
		} else {
			args.opcode = ac_image_atomic;
			switch (inst->Instruction.Opcode) {
			case TGSI_OPCODE_ATOMXCHG: args.atomic = ac_atomic_swap; break;
			case TGSI_OPCODE_ATOMUADD: args.atomic = ac_atomic_add; break;
			case TGSI_OPCODE_ATOMAND: args.atomic = ac_atomic_and; break;
			case TGSI_OPCODE_ATOMOR: args.atomic = ac_atomic_or; break;
			case TGSI_OPCODE_ATOMXOR: args.atomic = ac_atomic_xor; break;
			case TGSI_OPCODE_ATOMUMIN: args.atomic = ac_atomic_umin; break;
			case TGSI_OPCODE_ATOMUMAX: args.atomic = ac_atomic_umax; break;
			case TGSI_OPCODE_ATOMIMIN: args.atomic = ac_atomic_smin; break;
			case TGSI_OPCODE_ATOMIMAX: args.atomic = ac_atomic_smax; break;
			default: unreachable("unhandled image atomic");
			}
		}

		args.dim = ac_image_dim_from_tgsi_target(ctx->screen, inst->Memory.Texture);
		emit_data->output[emit_data->chan] =
			ac_to_float(&ctx->ac, ac_build_image_opcode(&ctx->ac, &args));
	}
}

static LLVMValueRef fix_resinfo(struct si_shader_context *ctx,
				unsigned target, LLVMValueRef out)
{
	LLVMBuilderRef builder = ctx->ac.builder;

	/* 1D textures are allocated and used as 2D on GFX9. */
        if (ctx->screen->info.chip_class >= GFX9 &&
	    (target == TGSI_TEXTURE_1D_ARRAY ||
	     target == TGSI_TEXTURE_SHADOW1D_ARRAY)) {
		LLVMValueRef layers =
			LLVMBuildExtractElement(builder, out,
						LLVMConstInt(ctx->i32, 2, 0), "");
		out = LLVMBuildInsertElement(builder, out, layers,
					     ctx->i32_1, "");
	}

	/* Divide the number of layers by 6 to get the number of cubes. */
	if (target == TGSI_TEXTURE_CUBE_ARRAY ||
	    target == TGSI_TEXTURE_SHADOWCUBE_ARRAY) {
		LLVMValueRef imm2 = LLVMConstInt(ctx->i32, 2, 0);

		LLVMValueRef z = LLVMBuildExtractElement(builder, out, imm2, "");
		z = LLVMBuildSDiv(builder, z, LLVMConstInt(ctx->i32, 6, 0), "");

		out = LLVMBuildInsertElement(builder, out, z, imm2, "");
	}
	return out;
}

static void resq_emit(
		const struct lp_build_tgsi_action *action,
		struct lp_build_tgsi_context *bld_base,
		struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMBuilderRef builder = ctx->ac.builder;
	const struct tgsi_full_instruction *inst = emit_data->inst;
	const struct tgsi_full_src_register *reg =
		&inst->Src[inst->Instruction.Opcode == TGSI_OPCODE_TXQ ? 1 : 0];

	if (reg->Register.File == TGSI_FILE_BUFFER) {
		LLVMValueRef rsrc = shader_buffer_fetch_rsrc(ctx, reg, false);

		emit_data->output[emit_data->chan] =
			LLVMBuildExtractElement(builder, rsrc,
						LLVMConstInt(ctx->i32, 2, 0), "");
		return;
	}

	if (inst->Instruction.Opcode == TGSI_OPCODE_TXQ &&
	    inst->Texture.Texture == TGSI_TEXTURE_BUFFER) {
		LLVMValueRef rsrc;

		tex_fetch_ptrs(bld_base, emit_data, &rsrc, NULL, NULL);
		/* Read the size from the buffer descriptor directly. */
		emit_data->output[emit_data->chan] =
			get_buffer_size(bld_base, rsrc);
		return;
	}

	if (inst->Instruction.Opcode == TGSI_OPCODE_RESQ &&
	    inst->Memory.Texture == TGSI_TEXTURE_BUFFER) {
		LLVMValueRef rsrc;

		image_fetch_rsrc(bld_base, reg, false, inst->Memory.Texture, &rsrc);
		emit_data->output[emit_data->chan] =
			get_buffer_size(bld_base, rsrc);
		return;
	}

	unsigned target;

	if (inst->Instruction.Opcode == TGSI_OPCODE_TXQ) {
		target = inst->Texture.Texture;
	} else {
		if (inst->Memory.Texture == TGSI_TEXTURE_3D)
			target = TGSI_TEXTURE_2D_ARRAY;
		else
			target = inst->Memory.Texture;
	}

	struct ac_image_args args = {};
	args.opcode = ac_image_get_resinfo;
	args.dim = ac_texture_dim_from_tgsi_target(ctx->screen, target);
	args.dmask = 0xf;

	if (inst->Instruction.Opcode == TGSI_OPCODE_TXQ) {
		tex_fetch_ptrs(bld_base, emit_data, &args.resource, NULL, NULL);
		args.lod = lp_build_emit_fetch(bld_base, inst, 0, TGSI_CHAN_X);
	} else {
		image_fetch_rsrc(bld_base, reg, false, target, &args.resource);
		args.lod = ctx->i32_0;
	}

	emit_data->output[emit_data->chan] =
		fix_resinfo(ctx, target, ac_build_image_opcode(&ctx->ac, &args));
}

/**
 * Load an image view, fmask view. or sampler state descriptor.
 */
LLVMValueRef si_load_sampler_desc(struct si_shader_context *ctx,
				  LLVMValueRef list, LLVMValueRef index,
				  enum ac_descriptor_type type)
{
	LLVMBuilderRef builder = ctx->ac.builder;

	switch (type) {
	case AC_DESC_IMAGE:
		/* The image is at [0:7]. */
		index = LLVMBuildMul(builder, index, LLVMConstInt(ctx->i32, 2, 0), "");
		break;
	case AC_DESC_BUFFER:
		/* The buffer is in [4:7]. */
		index = ac_build_imad(&ctx->ac, index, LLVMConstInt(ctx->i32, 4, 0),
				      ctx->i32_1);
		list = LLVMBuildPointerCast(builder, list,
					    ac_array_in_const32_addr_space(ctx->v4i32), "");
		break;
	case AC_DESC_FMASK:
		/* The FMASK is at [8:15]. */
		index = ac_build_imad(&ctx->ac, index, LLVMConstInt(ctx->i32, 2, 0),
				      ctx->i32_1);
		break;
	case AC_DESC_SAMPLER:
		/* The sampler state is at [12:15]. */
		index = ac_build_imad(&ctx->ac, index, LLVMConstInt(ctx->i32, 4, 0),
				      LLVMConstInt(ctx->i32, 3, 0));
		list = LLVMBuildPointerCast(builder, list,
					    ac_array_in_const32_addr_space(ctx->v4i32), "");
		break;
	}

	return ac_build_load_to_sgpr(&ctx->ac, list, index);
}

/* Disable anisotropic filtering if BASE_LEVEL == LAST_LEVEL.
 *
 * SI-CI:
 *   If BASE_LEVEL == LAST_LEVEL, the shader must disable anisotropic
 *   filtering manually. The driver sets img7 to a mask clearing
 *   MAX_ANISO_RATIO if BASE_LEVEL == LAST_LEVEL. The shader must do:
 *     s_and_b32 samp0, samp0, img7
 *
 * VI:
 *   The ANISO_OVERRIDE sampler field enables this fix in TA.
 */
static LLVMValueRef sici_fix_sampler_aniso(struct si_shader_context *ctx,
					   LLVMValueRef res, LLVMValueRef samp)
{
	LLVMValueRef img7, samp0;

	if (ctx->screen->info.chip_class >= VI)
		return samp;

	img7 = LLVMBuildExtractElement(ctx->ac.builder, res,
				       LLVMConstInt(ctx->i32, 7, 0), "");
	samp0 = LLVMBuildExtractElement(ctx->ac.builder, samp,
					ctx->i32_0, "");
	samp0 = LLVMBuildAnd(ctx->ac.builder, samp0, img7, "");
	return LLVMBuildInsertElement(ctx->ac.builder, samp, samp0,
				      ctx->i32_0, "");
}

static void tex_fetch_ptrs(struct lp_build_tgsi_context *bld_base,
			   struct lp_build_emit_data *emit_data,
			   LLVMValueRef *res_ptr, LLVMValueRef *samp_ptr,
			   LLVMValueRef *fmask_ptr)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef list = LLVMGetParam(ctx->main_fn, ctx->param_samplers_and_images);
	const struct tgsi_full_instruction *inst = emit_data->inst;
	const struct tgsi_full_src_register *reg;
	unsigned target = inst->Texture.Texture;
	unsigned sampler_src;
	LLVMValueRef index;

	sampler_src = emit_data->inst->Instruction.NumSrcRegs - 1;
	reg = &emit_data->inst->Src[sampler_src];

	if (reg->Register.Indirect) {
		index = si_get_bounded_indirect_index(ctx,
						      &reg->Indirect,
						      reg->Register.Index,
						      ctx->num_samplers);
		index = LLVMBuildAdd(ctx->ac.builder, index,
				     LLVMConstInt(ctx->i32, SI_NUM_IMAGES / 2, 0), "");
	} else {
		index = LLVMConstInt(ctx->i32,
				     si_get_sampler_slot(reg->Register.Index), 0);
	}

	if (reg->Register.File != TGSI_FILE_SAMPLER) {
		/* Bindless descriptors are accessible from a different pair of
		 * user SGPR indices.
		 */
		list = LLVMGetParam(ctx->main_fn,
				    ctx->param_bindless_samplers_and_images);
		index = lp_build_emit_fetch_src(bld_base, reg,
						TGSI_TYPE_UNSIGNED, 0);

		/* Since bindless handle arithmetic can contain an unsigned integer
		 * wraparound and si_load_sampler_desc assumes there isn't any,
		 * use GEP without "inbounds" (inside ac_build_pointer_add)
		 * to prevent incorrect code generation and hangs.
		 */
		index = LLVMBuildMul(ctx->ac.builder, index, LLVMConstInt(ctx->i32, 2, 0), "");
		list = ac_build_pointer_add(&ctx->ac, list, index);
		index = ctx->i32_0;
	}

	if (target == TGSI_TEXTURE_BUFFER)
		*res_ptr = si_load_sampler_desc(ctx, list, index, AC_DESC_BUFFER);
	else
		*res_ptr = si_load_sampler_desc(ctx, list, index, AC_DESC_IMAGE);

	if (samp_ptr)
		*samp_ptr = NULL;
	if (fmask_ptr)
		*fmask_ptr = NULL;

	if (target == TGSI_TEXTURE_2D_MSAA ||
	    target == TGSI_TEXTURE_2D_ARRAY_MSAA) {
		if (fmask_ptr)
			*fmask_ptr = si_load_sampler_desc(ctx, list, index,
						          AC_DESC_FMASK);
	} else if (target != TGSI_TEXTURE_BUFFER) {
		if (samp_ptr) {
			*samp_ptr = si_load_sampler_desc(ctx, list, index,
						         AC_DESC_SAMPLER);
			*samp_ptr = sici_fix_sampler_aniso(ctx, *res_ptr, *samp_ptr);
		}
	}
}

/* Gather4 should follow the same rules as bilinear filtering, but the hardware
 * incorrectly forces nearest filtering if the texture format is integer.
 * The only effect it has on Gather4, which always returns 4 texels for
 * bilinear filtering, is that the final coordinates are off by 0.5 of
 * the texel size.
 *
 * The workaround is to subtract 0.5 from the unnormalized coordinates,
 * or (0.5 / size) from the normalized coordinates.
 *
 * However, cube textures with 8_8_8_8 data formats require a different
 * workaround of overriding the num format to USCALED/SSCALED. This would lose
 * precision in 32-bit data formats, so it needs to be applied dynamically at
 * runtime. In this case, return an i1 value that indicates whether the
 * descriptor was overridden (and hence a fixup of the sampler result is needed).
 */
static LLVMValueRef
si_lower_gather4_integer(struct si_shader_context *ctx,
			 struct ac_image_args *args,
			 unsigned target,
			 enum tgsi_return_type return_type)
{
	LLVMBuilderRef builder = ctx->ac.builder;
	LLVMValueRef wa_8888 = NULL;
	LLVMValueRef half_texel[2];

	assert(return_type == TGSI_RETURN_TYPE_SINT ||
	       return_type == TGSI_RETURN_TYPE_UINT);

	if (target == TGSI_TEXTURE_CUBE ||
	    target == TGSI_TEXTURE_CUBE_ARRAY) {
		LLVMValueRef formats;
		LLVMValueRef data_format;
		LLVMValueRef wa_formats;

		formats = LLVMBuildExtractElement(builder, args->resource, ctx->i32_1, "");

		data_format = LLVMBuildLShr(builder, formats,
					    LLVMConstInt(ctx->i32, 20, false), "");
		data_format = LLVMBuildAnd(builder, data_format,
					   LLVMConstInt(ctx->i32, (1u << 6) - 1, false), "");
		wa_8888 = LLVMBuildICmp(
			builder, LLVMIntEQ, data_format,
			LLVMConstInt(ctx->i32, V_008F14_IMG_DATA_FORMAT_8_8_8_8, false),
			"");

		uint32_t wa_num_format =
			return_type == TGSI_RETURN_TYPE_UINT ?
			S_008F14_NUM_FORMAT_GFX6(V_008F14_IMG_NUM_FORMAT_USCALED) :
			S_008F14_NUM_FORMAT_GFX6(V_008F14_IMG_NUM_FORMAT_SSCALED);
		wa_formats = LLVMBuildAnd(builder, formats,
					  LLVMConstInt(ctx->i32, C_008F14_NUM_FORMAT_GFX6, false),
					  "");
		wa_formats = LLVMBuildOr(builder, wa_formats,
					LLVMConstInt(ctx->i32, wa_num_format, false), "");

		formats = LLVMBuildSelect(builder, wa_8888, wa_formats, formats, "");
		args->resource = LLVMBuildInsertElement(
			builder, args->resource, formats, ctx->i32_1, "");
	}

	if (target == TGSI_TEXTURE_RECT ||
	    target == TGSI_TEXTURE_SHADOWRECT) {
		assert(!wa_8888);
		half_texel[0] = half_texel[1] = LLVMConstReal(ctx->f32, -0.5);
	} else {
		struct ac_image_args resinfo = {};
		struct lp_build_if_state if_ctx;

		if (wa_8888) {
			/* Skip the texture size query entirely if we don't need it. */
			lp_build_if(&if_ctx, &ctx->gallivm, LLVMBuildNot(builder, wa_8888, ""));
		}

		/* Query the texture size. */
		resinfo.opcode = ac_image_get_resinfo;
		resinfo.dim = ac_texture_dim_from_tgsi_target(ctx->screen, target);
		resinfo.resource = args->resource;
		resinfo.sampler = args->sampler;
		resinfo.lod = ctx->ac.i32_0;
		resinfo.dmask = 0xf;

		LLVMValueRef texsize =
			fix_resinfo(ctx, target,
				    ac_build_image_opcode(&ctx->ac, &resinfo));

		/* Compute -0.5 / size. */
		for (unsigned c = 0; c < 2; c++) {
			half_texel[c] =
				LLVMBuildExtractElement(builder, texsize,
							LLVMConstInt(ctx->i32, c, 0), "");
			half_texel[c] = LLVMBuildUIToFP(builder, half_texel[c], ctx->f32, "");
			half_texel[c] = ac_build_fdiv(&ctx->ac, ctx->ac.f32_1, half_texel[c]);
			half_texel[c] = LLVMBuildFMul(builder, half_texel[c],
						      LLVMConstReal(ctx->f32, -0.5), "");
		}

		if (wa_8888) {
			lp_build_endif(&if_ctx);

			LLVMBasicBlockRef bb[2] = { if_ctx.true_block, if_ctx.entry_block };

			for (unsigned c = 0; c < 2; c++) {
				LLVMValueRef values[2] = { half_texel[c], ctx->ac.f32_0 };
				half_texel[c] = ac_build_phi(&ctx->ac, ctx->f32, 2,
							     values, bb);
			}
		}
	}

	for (unsigned c = 0; c < 2; c++) {
		LLVMValueRef tmp;
		tmp = ac_to_float(&ctx->ac, args->coords[c]);
		tmp = LLVMBuildFAdd(builder, tmp, half_texel[c], "");
		args->coords[c] = ac_to_integer(&ctx->ac, tmp);
	}

	return wa_8888;
}

/* The second half of the cube texture 8_8_8_8 integer workaround: adjust the
 * result after the gather operation.
 */
static LLVMValueRef
si_fix_gather4_integer_result(struct si_shader_context *ctx,
			   LLVMValueRef result,
			   enum tgsi_return_type return_type,
			   LLVMValueRef wa)
{
	LLVMBuilderRef builder = ctx->ac.builder;

	assert(return_type == TGSI_RETURN_TYPE_SINT ||
	       return_type == TGSI_RETURN_TYPE_UINT);

	for (unsigned chan = 0; chan < 4; ++chan) {
		LLVMValueRef chanv = LLVMConstInt(ctx->i32, chan, false);
		LLVMValueRef value;
		LLVMValueRef wa_value;

		value = LLVMBuildExtractElement(builder, result, chanv, "");

		if (return_type == TGSI_RETURN_TYPE_UINT)
			wa_value = LLVMBuildFPToUI(builder, value, ctx->i32, "");
		else
			wa_value = LLVMBuildFPToSI(builder, value, ctx->i32, "");
		wa_value = ac_to_float(&ctx->ac, wa_value);
		value = LLVMBuildSelect(builder, wa, wa_value, value, "");

		result = LLVMBuildInsertElement(builder, result, value, chanv, "");
	}

	return result;
}

static void build_tex_intrinsic(const struct lp_build_tgsi_action *action,
				struct lp_build_tgsi_context *bld_base,
				struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	const struct tgsi_full_instruction *inst = emit_data->inst;
	unsigned opcode = inst->Instruction.Opcode;
	unsigned target = inst->Texture.Texture;
	struct ac_image_args args = {};
	int ref_pos = tgsi_util_get_shadow_ref_src_index(target);
	unsigned chan;
	bool has_offset = inst->Texture.NumOffsets > 0;
	LLVMValueRef fmask_ptr = NULL;

	tex_fetch_ptrs(bld_base, emit_data, &args.resource, &args.sampler, &fmask_ptr);

	if (target == TGSI_TEXTURE_BUFFER) {
		LLVMValueRef vindex = lp_build_emit_fetch(bld_base, inst, 0, TGSI_CHAN_X);
		unsigned num_channels =
			util_last_bit(inst->Dst[0].Register.WriteMask);
		LLVMValueRef result =
			ac_build_buffer_load_format(&ctx->ac,
						    args.resource,
						    vindex,
						    ctx->i32_0,
						    num_channels, false, true);
		emit_data->output[emit_data->chan] =
			ac_build_expand_to_vec4(&ctx->ac, result, num_channels);
		return;
	}

	/* Fetch and project texture coordinates */
	args.coords[3] = lp_build_emit_fetch(bld_base, inst, 0, TGSI_CHAN_W);
	for (chan = 0; chan < 3; chan++) {
		args.coords[chan] = lp_build_emit_fetch(bld_base, inst, 0, chan);
		if (opcode == TGSI_OPCODE_TXP)
			args.coords[chan] = ac_build_fdiv(&ctx->ac,
				args.coords[chan], args.coords[3]);
	}

	if (opcode == TGSI_OPCODE_TXP)
		args.coords[3] = ctx->ac.f32_1;

	/* Pack offsets. */
	if (has_offset &&
	    opcode != TGSI_OPCODE_TXF &&
	    opcode != TGSI_OPCODE_TXF_LZ) {
		/* The offsets are six-bit signed integers packed like this:
		 *   X=[5:0], Y=[13:8], and Z=[21:16].
		 */
		LLVMValueRef offset[3], pack;

		assert(inst->Texture.NumOffsets == 1);

		for (chan = 0; chan < 3; chan++) {
			offset[chan] = lp_build_emit_fetch_texoffset(bld_base, inst, 0, chan);
			offset[chan] = LLVMBuildAnd(ctx->ac.builder, offset[chan],
						    LLVMConstInt(ctx->i32, 0x3f, 0), "");
			if (chan)
				offset[chan] = LLVMBuildShl(ctx->ac.builder, offset[chan],
							    LLVMConstInt(ctx->i32, chan*8, 0), "");
		}

		pack = LLVMBuildOr(ctx->ac.builder, offset[0], offset[1], "");
		pack = LLVMBuildOr(ctx->ac.builder, pack, offset[2], "");
		args.offset = pack;
	}

	/* Pack LOD bias value */
	if (opcode == TGSI_OPCODE_TXB)
		args.bias = args.coords[3];
	if (opcode == TGSI_OPCODE_TXB2)
		args.bias = lp_build_emit_fetch(bld_base, inst, 1, TGSI_CHAN_X);

	/* Pack depth comparison value */
	if (tgsi_is_shadow_target(target) && opcode != TGSI_OPCODE_LODQ) {
		LLVMValueRef z;

		if (target == TGSI_TEXTURE_SHADOWCUBE_ARRAY) {
			z = lp_build_emit_fetch(bld_base, inst, 1, TGSI_CHAN_X);
		} else {
			assert(ref_pos >= 0);
			z = args.coords[ref_pos];
		}

		/* Section 8.23.1 (Depth Texture Comparison Mode) of the
		 * OpenGL 4.5 spec says:
		 *
		 *    "If the texture’s internal format indicates a fixed-point
		 *     depth texture, then D_t and D_ref are clamped to the
		 *     range [0, 1]; otherwise no clamping is performed."
		 *
		 * TC-compatible HTILE promotes Z16 and Z24 to Z32_FLOAT,
		 * so the depth comparison value isn't clamped for Z16 and
		 * Z24 anymore. Do it manually here.
		 */
		if (ctx->screen->info.chip_class >= VI) {
			LLVMValueRef upgraded;
			LLVMValueRef clamped;
			upgraded = LLVMBuildExtractElement(ctx->ac.builder, args.sampler,
							   LLVMConstInt(ctx->i32, 3, false), "");
			upgraded = LLVMBuildLShr(ctx->ac.builder, upgraded,
						 LLVMConstInt(ctx->i32, 29, false), "");
			upgraded = LLVMBuildTrunc(ctx->ac.builder, upgraded, ctx->i1, "");
			clamped = ac_build_clamp(&ctx->ac, z);
			z = LLVMBuildSelect(ctx->ac.builder, upgraded, clamped, z, "");
		}

		args.compare = z;
	}

	/* Pack user derivatives */
	if (opcode == TGSI_OPCODE_TXD) {
		int param, num_src_deriv_channels, num_dst_deriv_channels;

		switch (target) {
		case TGSI_TEXTURE_3D:
			num_src_deriv_channels = 3;
			num_dst_deriv_channels = 3;
			break;
		case TGSI_TEXTURE_2D:
		case TGSI_TEXTURE_SHADOW2D:
		case TGSI_TEXTURE_RECT:
		case TGSI_TEXTURE_SHADOWRECT:
		case TGSI_TEXTURE_2D_ARRAY:
		case TGSI_TEXTURE_SHADOW2D_ARRAY:
			num_src_deriv_channels = 2;
			num_dst_deriv_channels = 2;
			break;
		case TGSI_TEXTURE_CUBE:
		case TGSI_TEXTURE_SHADOWCUBE:
		case TGSI_TEXTURE_CUBE_ARRAY:
		case TGSI_TEXTURE_SHADOWCUBE_ARRAY:
			/* Cube derivatives will be converted to 2D. */
			num_src_deriv_channels = 3;
			num_dst_deriv_channels = 3;
			break;
		case TGSI_TEXTURE_1D:
		case TGSI_TEXTURE_SHADOW1D:
		case TGSI_TEXTURE_1D_ARRAY:
		case TGSI_TEXTURE_SHADOW1D_ARRAY:
			num_src_deriv_channels = 1;

			/* 1D textures are allocated and used as 2D on GFX9. */
			if (ctx->screen->info.chip_class >= GFX9) {
				num_dst_deriv_channels = 2;
			} else {
				num_dst_deriv_channels = 1;
			}
			break;
		default:
			unreachable("invalid target");
		}

		for (param = 0; param < 2; param++) {
			for (chan = 0; chan < num_src_deriv_channels; chan++)
				args.derivs[param * num_dst_deriv_channels + chan] =
					lp_build_emit_fetch(bld_base, inst, param+1, chan);

			/* Fill in the rest with zeros. */
			for (chan = num_src_deriv_channels;
			     chan < num_dst_deriv_channels; chan++)
				args.derivs[param * num_dst_deriv_channels + chan] =
					ctx->ac.f32_0;
		}
	}

	if (target == TGSI_TEXTURE_CUBE ||
	    target == TGSI_TEXTURE_CUBE_ARRAY ||
	    target == TGSI_TEXTURE_SHADOWCUBE ||
	    target == TGSI_TEXTURE_SHADOWCUBE_ARRAY) {
		ac_prepare_cube_coords(&ctx->ac,
				       opcode == TGSI_OPCODE_TXD,
				       target == TGSI_TEXTURE_CUBE_ARRAY ||
				       target == TGSI_TEXTURE_SHADOWCUBE_ARRAY,
				       opcode == TGSI_OPCODE_LODQ,
				       args.coords, args.derivs);
	} else if (tgsi_is_array_sampler(target) &&
		   opcode != TGSI_OPCODE_TXF &&
		   opcode != TGSI_OPCODE_TXF_LZ &&
		   ctx->screen->info.chip_class <= VI) {
		unsigned array_coord = target == TGSI_TEXTURE_1D_ARRAY ? 1 : 2;
		args.coords[array_coord] = ac_build_round(&ctx->ac, args.coords[array_coord]);
	}

	/* 1D textures are allocated and used as 2D on GFX9. */
	if (ctx->screen->info.chip_class >= GFX9) {
		LLVMValueRef filler;

		/* Use 0.5, so that we don't sample the border color. */
		if (opcode == TGSI_OPCODE_TXF ||
		    opcode == TGSI_OPCODE_TXF_LZ)
			filler = ctx->i32_0;
		else
			filler = LLVMConstReal(ctx->f32, 0.5);

		if (target == TGSI_TEXTURE_1D ||
		    target == TGSI_TEXTURE_SHADOW1D) {
			args.coords[1] = filler;
		} else if (target == TGSI_TEXTURE_1D_ARRAY ||
			   target == TGSI_TEXTURE_SHADOW1D_ARRAY) {
			args.coords[2] = args.coords[1];
			args.coords[1] = filler;
		}
	}

	/* Pack LOD or sample index */
	if (opcode == TGSI_OPCODE_TXL)
		args.lod = args.coords[3];
	else if (opcode == TGSI_OPCODE_TXL2)
		args.lod = lp_build_emit_fetch(bld_base, inst, 1, TGSI_CHAN_X);
	else if (opcode == TGSI_OPCODE_TXF) {
		if (target == TGSI_TEXTURE_2D_MSAA) {
			/* No LOD, but move sample index into the right place. */
			args.coords[2] = args.coords[3];
		} else if (target != TGSI_TEXTURE_2D_ARRAY_MSAA) {
			args.lod = args.coords[3];
		}
	}

	if (target == TGSI_TEXTURE_2D_MSAA ||
	    target == TGSI_TEXTURE_2D_ARRAY_MSAA) {
		ac_apply_fmask_to_sample(&ctx->ac, fmask_ptr, args.coords,
					 target == TGSI_TEXTURE_2D_ARRAY_MSAA);
	}

	if (opcode == TGSI_OPCODE_TXF ||
	    opcode == TGSI_OPCODE_TXF_LZ) {
		/* add tex offsets */
		if (inst->Texture.NumOffsets) {
			const struct tgsi_texture_offset *off = inst->TexOffsets;

			assert(inst->Texture.NumOffsets == 1);

			switch (target) {
			case TGSI_TEXTURE_3D:
				args.coords[2] =
					LLVMBuildAdd(ctx->ac.builder, args.coords[2],
						ctx->imms[off->Index * TGSI_NUM_CHANNELS + off->SwizzleZ], "");
				/* fall through */
			case TGSI_TEXTURE_2D:
			case TGSI_TEXTURE_SHADOW2D:
			case TGSI_TEXTURE_RECT:
			case TGSI_TEXTURE_SHADOWRECT:
			case TGSI_TEXTURE_2D_ARRAY:
			case TGSI_TEXTURE_SHADOW2D_ARRAY:
				args.coords[1] =
					LLVMBuildAdd(ctx->ac.builder, args.coords[1],
						ctx->imms[off->Index * TGSI_NUM_CHANNELS + off->SwizzleY], "");
				/* fall through */
			case TGSI_TEXTURE_1D:
			case TGSI_TEXTURE_SHADOW1D:
			case TGSI_TEXTURE_1D_ARRAY:
			case TGSI_TEXTURE_SHADOW1D_ARRAY:
				args.coords[0] =
					LLVMBuildAdd(ctx->ac.builder, args.coords[0],
						ctx->imms[off->Index * TGSI_NUM_CHANNELS + off->SwizzleX], "");
				break;
				/* texture offsets do not apply to other texture targets */
			}
		}
	}

	if (opcode == TGSI_OPCODE_TG4) {
		unsigned gather_comp = 0;

		/* DMASK was repurposed for GATHER4. 4 components are always
		 * returned and DMASK works like a swizzle - it selects
		 * the component to fetch. The only valid DMASK values are
		 * 1=red, 2=green, 4=blue, 8=alpha. (e.g. 1 returns
		 * (red,red,red,red) etc.) The ISA document doesn't mention
		 * this.
		 */

		/* Get the component index from src1.x for Gather4. */
		if (!tgsi_is_shadow_target(target)) {
			LLVMValueRef comp_imm;
			struct tgsi_src_register src1 = inst->Src[1].Register;

			assert(src1.File == TGSI_FILE_IMMEDIATE);

			comp_imm = ctx->imms[src1.Index * TGSI_NUM_CHANNELS + src1.SwizzleX];
			gather_comp = LLVMConstIntGetZExtValue(comp_imm);
			gather_comp = CLAMP(gather_comp, 0, 3);
		}

		args.dmask = 1 << gather_comp;
	} else {
		args.dmask = 0xf;
	}

	args.dim = ac_texture_dim_from_tgsi_target(ctx->screen, target);
	args.unorm = target == TGSI_TEXTURE_RECT ||
		     target == TGSI_TEXTURE_SHADOWRECT;
	args.opcode = ac_image_sample;

	switch (opcode) {
	case TGSI_OPCODE_TXF:
	case TGSI_OPCODE_TXF_LZ:
		args.opcode = opcode == TGSI_OPCODE_TXF_LZ ||
			      target == TGSI_TEXTURE_2D_MSAA ||
			      target == TGSI_TEXTURE_2D_ARRAY_MSAA ?
				      ac_image_load : ac_image_load_mip;
		break;
	case TGSI_OPCODE_LODQ:
		args.opcode = ac_image_get_lod;
		break;
	case TGSI_OPCODE_TEX:
	case TGSI_OPCODE_TEX2:
	case TGSI_OPCODE_TXP:
		if (ctx->type != PIPE_SHADER_FRAGMENT)
			args.level_zero = true;
		break;
	case TGSI_OPCODE_TEX_LZ:
		args.level_zero = true;
		break;
	case TGSI_OPCODE_TXB:
	case TGSI_OPCODE_TXB2:
		assert(ctx->type == PIPE_SHADER_FRAGMENT);
		break;
	case TGSI_OPCODE_TXL:
	case TGSI_OPCODE_TXL2:
		break;
	case TGSI_OPCODE_TXD:
		break;
	case TGSI_OPCODE_TG4:
		args.opcode = ac_image_gather4;
		args.level_zero = true;
		break;
	default:
		assert(0);
		return;
	}

	/* The hardware needs special lowering for Gather4 with integer formats. */
	LLVMValueRef gather4_int_result_workaround = NULL;

	if (ctx->screen->info.chip_class <= VI &&
	    opcode == TGSI_OPCODE_TG4) {
		assert(inst->Texture.ReturnType != TGSI_RETURN_TYPE_UNKNOWN);

		if (inst->Texture.ReturnType == TGSI_RETURN_TYPE_SINT ||
		    inst->Texture.ReturnType == TGSI_RETURN_TYPE_UINT) {
			gather4_int_result_workaround =
				si_lower_gather4_integer(ctx, &args, target,
							 inst->Texture.ReturnType);
		}
	}

	args.attributes = AC_FUNC_ATTR_READNONE;
	LLVMValueRef result = ac_build_image_opcode(&ctx->ac, &args);

	if (gather4_int_result_workaround) {
		result = si_fix_gather4_integer_result(ctx, result,
						       inst->Texture.ReturnType,
						       gather4_int_result_workaround);
	}

	emit_data->output[emit_data->chan] = result;
}

static void si_llvm_emit_txqs(
	const struct lp_build_tgsi_action *action,
	struct lp_build_tgsi_context *bld_base,
	struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	LLVMValueRef res, samples;
	LLVMValueRef res_ptr, samp_ptr, fmask_ptr = NULL;

	tex_fetch_ptrs(bld_base, emit_data, &res_ptr, &samp_ptr, &fmask_ptr);

	/* Read the samples from the descriptor directly. */
	res = LLVMBuildBitCast(ctx->ac.builder, res_ptr, ctx->v8i32, "");
	samples = LLVMBuildExtractElement(ctx->ac.builder, res,
					  LLVMConstInt(ctx->i32, 3, 0), "");
	samples = LLVMBuildLShr(ctx->ac.builder, samples,
				LLVMConstInt(ctx->i32, 16, 0), "");
	samples = LLVMBuildAnd(ctx->ac.builder, samples,
			       LLVMConstInt(ctx->i32, 0xf, 0), "");
	samples = LLVMBuildShl(ctx->ac.builder, ctx->i32_1,
			       samples, "");

	emit_data->output[emit_data->chan] = samples;
}

static void si_llvm_emit_fbfetch(const struct lp_build_tgsi_action *action,
				 struct lp_build_tgsi_context *bld_base,
				 struct lp_build_emit_data *emit_data)
{
	struct si_shader_context *ctx = si_shader_context(bld_base);
	struct ac_image_args args = {};
	LLVMValueRef ptr, image, fmask;

	/* Ignore src0, because KHR_blend_func_extended disallows multiple render
	 * targets.
	 */

	/* Load the image descriptor. */
	STATIC_ASSERT(SI_PS_IMAGE_COLORBUF0 % 2 == 0);
	ptr = LLVMGetParam(ctx->main_fn, ctx->param_rw_buffers);
	ptr = LLVMBuildPointerCast(ctx->ac.builder, ptr,
				   ac_array_in_const32_addr_space(ctx->v8i32), "");
	image = ac_build_load_to_sgpr(&ctx->ac, ptr,
			LLVMConstInt(ctx->i32, SI_PS_IMAGE_COLORBUF0 / 2, 0));

	unsigned chan = 0;

	args.coords[chan++] = si_unpack_param(ctx, SI_PARAM_POS_FIXED_PT, 0, 16);

	if (!ctx->shader->key.mono.u.ps.fbfetch_is_1D)
		args.coords[chan++] = si_unpack_param(ctx, SI_PARAM_POS_FIXED_PT, 16, 16);

	/* Get the current render target layer index. */
	if (ctx->shader->key.mono.u.ps.fbfetch_layered)
		args.coords[chan++] = si_unpack_param(ctx, SI_PARAM_ANCILLARY, 16, 11);

	if (ctx->shader->key.mono.u.ps.fbfetch_msaa)
		args.coords[chan++] = si_get_sample_id(ctx);

	if (ctx->shader->key.mono.u.ps.fbfetch_msaa) {
		fmask = ac_build_load_to_sgpr(&ctx->ac, ptr,
			LLVMConstInt(ctx->i32, SI_PS_IMAGE_COLORBUF0_FMASK / 2, 0));

		ac_apply_fmask_to_sample(&ctx->ac, fmask, args.coords,
					 ctx->shader->key.mono.u.ps.fbfetch_layered);
	}

	args.opcode = ac_image_load;
	args.resource = image;
	args.dmask = 0xf;
	if (ctx->shader->key.mono.u.ps.fbfetch_msaa)
		args.dim = ctx->shader->key.mono.u.ps.fbfetch_layered ?
			ac_image_2darraymsaa : ac_image_2dmsaa;
	else if (ctx->shader->key.mono.u.ps.fbfetch_is_1D)
		args.dim = ctx->shader->key.mono.u.ps.fbfetch_layered ?
			ac_image_1darray : ac_image_1d;
	else
		args.dim = ctx->shader->key.mono.u.ps.fbfetch_layered ?
			ac_image_2darray : ac_image_2d;

	emit_data->output[emit_data->chan] =
		ac_build_image_opcode(&ctx->ac, &args);
}

/**
 * Setup actions for TGSI memory opcode, including texture opcodes.
 */
void si_shader_context_init_mem(struct si_shader_context *ctx)
{
	struct lp_build_tgsi_context *bld_base = &ctx->bld_base;

	bld_base->op_actions[TGSI_OPCODE_TEX].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TEX_LZ].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TEX2].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TXB].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TXB2].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TXD].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TXF].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TXF_LZ].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TXL].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TXL2].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TXP].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TXQ].emit = resq_emit;
	bld_base->op_actions[TGSI_OPCODE_TG4].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_LODQ].emit = build_tex_intrinsic;
	bld_base->op_actions[TGSI_OPCODE_TXQS].emit = si_llvm_emit_txqs;

	bld_base->op_actions[TGSI_OPCODE_FBFETCH].emit = si_llvm_emit_fbfetch;

	bld_base->op_actions[TGSI_OPCODE_LOAD].emit = load_emit;
	bld_base->op_actions[TGSI_OPCODE_STORE].emit = store_emit;
	bld_base->op_actions[TGSI_OPCODE_RESQ].emit = resq_emit;

	bld_base->op_actions[TGSI_OPCODE_ATOMUADD].emit = atomic_emit;
	bld_base->op_actions[TGSI_OPCODE_ATOMUADD].intr_name = "add";
	bld_base->op_actions[TGSI_OPCODE_ATOMXCHG].emit = atomic_emit;
	bld_base->op_actions[TGSI_OPCODE_ATOMXCHG].intr_name = "swap";
	bld_base->op_actions[TGSI_OPCODE_ATOMCAS].emit = atomic_emit;
	bld_base->op_actions[TGSI_OPCODE_ATOMCAS].intr_name = "cmpswap";
	bld_base->op_actions[TGSI_OPCODE_ATOMAND].emit = atomic_emit;
	bld_base->op_actions[TGSI_OPCODE_ATOMAND].intr_name = "and";
	bld_base->op_actions[TGSI_OPCODE_ATOMOR].emit = atomic_emit;
	bld_base->op_actions[TGSI_OPCODE_ATOMOR].intr_name = "or";
	bld_base->op_actions[TGSI_OPCODE_ATOMXOR].emit = atomic_emit;
	bld_base->op_actions[TGSI_OPCODE_ATOMXOR].intr_name = "xor";
	bld_base->op_actions[TGSI_OPCODE_ATOMUMIN].emit = atomic_emit;
	bld_base->op_actions[TGSI_OPCODE_ATOMUMIN].intr_name = "umin";
	bld_base->op_actions[TGSI_OPCODE_ATOMUMAX].emit = atomic_emit;
	bld_base->op_actions[TGSI_OPCODE_ATOMUMAX].intr_name = "umax";
	bld_base->op_actions[TGSI_OPCODE_ATOMIMIN].emit = atomic_emit;
	bld_base->op_actions[TGSI_OPCODE_ATOMIMIN].intr_name = "smin";
	bld_base->op_actions[TGSI_OPCODE_ATOMIMAX].emit = atomic_emit;
	bld_base->op_actions[TGSI_OPCODE_ATOMIMAX].intr_name = "smax";
}
