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

/* The compiler middle-end architecture: Explaining (non-)monolithic shaders
 * -------------------------------------------------------------------------
 *
 * Typically, there is one-to-one correspondence between API and HW shaders,
 * that is, for every API shader, there is exactly one shader binary in
 * the driver.
 *
 * The problem with that is that we also have to emulate some API states
 * (e.g. alpha-test, and many others) in shaders too. The two obvious ways
 * to deal with it are:
 * - each shader has multiple variants for each combination of emulated states,
 *   and the variants are compiled on demand, possibly relying on a shader
 *   cache for good performance
 * - patch shaders at the binary level
 *
 * This driver uses something completely different. The emulated states are
 * usually implemented at the beginning or end of shaders. Therefore, we can
 * split the shader into 3 parts:
 * - prolog part (shader code dependent on states)
 * - main part (the API shader)
 * - epilog part (shader code dependent on states)
 *
 * Each part is compiled as a separate shader and the final binaries are
 * concatenated. This type of shader is called non-monolithic, because it
 * consists of multiple independent binaries. Creating a new shader variant
 * is therefore only a concatenation of shader parts (binaries) and doesn't
 * involve any compilation. The main shader parts are the only parts that are
 * compiled when applications create shader objects. The prolog and epilog
 * parts are compiled on the first use and saved, so that their binaries can
 * be reused by many other shaders.
 *
 * One of the roles of the prolog part is to compute vertex buffer addresses
 * for vertex shaders. A few of the roles of the epilog part are color buffer
 * format conversions in pixel shaders that we have to do manually, and write
 * tessellation factors in tessellation control shaders. The prolog and epilog
 * have many other important responsibilities in various shader stages.
 * They don't just "emulate legacy stuff".
 *
 * Monolithic shaders are shaders where the parts are combined before LLVM
 * compilation, and the whole thing is compiled and optimized as one unit with
 * one binary on the output. The result is the same as the non-monolithic
 * shader, but the final code can be better, because LLVM can optimize across
 * all shader parts. Monolithic shaders aren't usually used except for these
 * special cases:
 *
 * 1) Some rarely-used states require modification of the main shader part
 *    itself, and in such cases, only the monolithic shader variant is
 *    compiled, and that's always done on the first use.
 *
 * 2) When we do cross-stage optimizations for separate shader objects and
 *    e.g. eliminate unused shader varyings, the resulting optimized shader
 *    variants are always compiled as monolithic shaders, and always
 *    asynchronously (i.e. not stalling ongoing rendering). We call them
 *    "optimized monolithic" shaders. The important property here is that
 *    the non-monolithic unoptimized shader variant is always available for use
 *    when the asynchronous compilation of the optimized shader is not done
 *    yet.
 *
 * Starting with GFX9 chips, some shader stages are merged, and the number of
 * shader parts per shader increased. The complete new list of shader parts is:
 * - 1st shader: prolog part
 * - 1st shader: main part
 * - 2nd shader: prolog part
 * - 2nd shader: main part
 * - 2nd shader: epilog part
 */

/* How linking shader inputs and outputs between vertex, tessellation, and
 * geometry shaders works.
 *
 * Inputs and outputs between shaders are stored in a buffer. This buffer
 * lives in LDS (typical case for tessellation), but it can also live
 * in memory (ESGS). Each input or output has a fixed location within a vertex.
 * The highest used input or output determines the stride between vertices.
 *
 * Since GS and tessellation are only possible in the OpenGL core profile,
 * only these semantics are valid for per-vertex data:
 *
 *   Name             Location
 *
 *   POSITION         0
 *   PSIZE            1
 *   CLIPDIST0..1     2..3
 *   CULLDIST0..1     (not implemented)
 *   GENERIC0..31     4..35
 *
 * For example, a shader only writing GENERIC0 has the output stride of 5.
 *
 * Only these semantics are valid for per-patch data:
 *
 *   Name             Location
 *
 *   TESSOUTER        0
 *   TESSINNER        1
 *   PATCH0..29       2..31
 *
 * That's how independent shaders agree on input and output locations.
 * The si_shader_io_get_unique_index function assigns the locations.
 *
 * For tessellation, other required information for calculating the input and
 * output addresses like the vertex stride, the patch stride, and the offsets
 * where per-vertex and per-patch data start, is passed to the shader via
 * user data SGPRs. The offsets and strides are calculated at draw time and
 * aren't available at compile time.
 */

#ifndef SI_SHADER_H
#define SI_SHADER_H

#include <llvm-c/Core.h> /* LLVMModuleRef */
#include <llvm-c/TargetMachine.h>
#include "tgsi/tgsi_scan.h"
#include "util/u_inlines.h"
#include "util/u_queue.h"

#include "ac_binary.h"
#include "ac_llvm_build.h"
#include "ac_llvm_util.h"

#include <stdio.h>

struct nir_shader;
struct si_shader;
struct si_context;

#define SI_MAX_ATTRIBS		16
#define SI_MAX_VS_OUTPUTS	40

/* Shader IO unique indices are supported for TGSI_SEMANTIC_GENERIC with an
 * index smaller than this.
 */
#define SI_MAX_IO_GENERIC       43

/* SGPR user data indices */
enum {
	SI_SGPR_RW_BUFFERS,  /* rings (& stream-out, VS only) */
#if !HAVE_32BIT_POINTERS
	SI_SGPR_RW_BUFFERS_HI,
#endif
	SI_SGPR_BINDLESS_SAMPLERS_AND_IMAGES,
#if !HAVE_32BIT_POINTERS
	SI_SGPR_BINDLESS_SAMPLERS_AND_IMAGES_HI,
#endif
	SI_SGPR_CONST_AND_SHADER_BUFFERS, /* or just a constant buffer 0 pointer */
#if !HAVE_32BIT_POINTERS
	SI_SGPR_CONST_AND_SHADER_BUFFERS_HI,
#endif
	SI_SGPR_SAMPLERS_AND_IMAGES,
#if !HAVE_32BIT_POINTERS
	SI_SGPR_SAMPLERS_AND_IMAGES_HI,
#endif
	SI_NUM_RESOURCE_SGPRS,

	/* API VS, TES without GS, GS copy shader */
	SI_SGPR_VS_STATE_BITS = SI_NUM_RESOURCE_SGPRS,
	SI_NUM_VS_STATE_RESOURCE_SGPRS,

	/* all VS variants */
	SI_SGPR_BASE_VERTEX = SI_NUM_VS_STATE_RESOURCE_SGPRS,
	SI_SGPR_START_INSTANCE,
	SI_SGPR_DRAWID,
	SI_VS_NUM_USER_SGPR,

	SI_SGPR_VS_BLIT_DATA = SI_SGPR_CONST_AND_SHADER_BUFFERS,

	/* TES */
	SI_SGPR_TES_OFFCHIP_LAYOUT = SI_NUM_VS_STATE_RESOURCE_SGPRS,
	SI_SGPR_TES_OFFCHIP_ADDR,
	SI_TES_NUM_USER_SGPR,

	/* GFX6-8: TCS only */
	GFX6_SGPR_TCS_OFFCHIP_LAYOUT = SI_NUM_RESOURCE_SGPRS,
	GFX6_SGPR_TCS_OUT_OFFSETS,
	GFX6_SGPR_TCS_OUT_LAYOUT,
	GFX6_SGPR_TCS_IN_LAYOUT,
	GFX6_TCS_NUM_USER_SGPR,

	/* GFX9: Merged shaders. */
#if HAVE_32BIT_POINTERS
	/* 2ND_CONST_AND_SHADER_BUFFERS is set in USER_DATA_ADDR_LO (SGPR0). */
	/* 2ND_SAMPLERS_AND_IMAGES is set in USER_DATA_ADDR_HI (SGPR1). */
	GFX9_MERGED_NUM_USER_SGPR = SI_VS_NUM_USER_SGPR,
#else
	/* 2ND_CONST_AND_SHADER_BUFFERS is set in USER_DATA_ADDR_LO/HI (SGPR[0:1]). */
	GFX9_SGPR_2ND_SAMPLERS_AND_IMAGES = SI_VS_NUM_USER_SGPR,
	GFX9_SGPR_2ND_SAMPLERS_AND_IMAGES_HI,
	GFX9_MERGED_NUM_USER_SGPR,
#endif

	/* GFX9: Merged LS-HS (VS-TCS) only. */
	GFX9_SGPR_TCS_OFFCHIP_LAYOUT = GFX9_MERGED_NUM_USER_SGPR,
	GFX9_SGPR_TCS_OUT_OFFSETS,
	GFX9_SGPR_TCS_OUT_LAYOUT,
#if !HAVE_32BIT_POINTERS
	GFX9_SGPR_align_for_vb_pointer,
#endif
	GFX9_TCS_NUM_USER_SGPR,

	/* GS limits */
	GFX6_GS_NUM_USER_SGPR = SI_NUM_RESOURCE_SGPRS,
#if HAVE_32BIT_POINTERS
	GFX9_VSGS_NUM_USER_SGPR = SI_VS_NUM_USER_SGPR,
	GFX9_TESGS_NUM_USER_SGPR = SI_TES_NUM_USER_SGPR,
#else
	GFX9_VSGS_NUM_USER_SGPR = GFX9_MERGED_NUM_USER_SGPR,
	GFX9_TESGS_NUM_USER_SGPR = GFX9_MERGED_NUM_USER_SGPR,
#endif
	SI_GSCOPY_NUM_USER_SGPR = SI_NUM_VS_STATE_RESOURCE_SGPRS,

	/* PS only */
	SI_SGPR_ALPHA_REF	= SI_NUM_RESOURCE_SGPRS,
	SI_PS_NUM_USER_SGPR,
};

/* LLVM function parameter indices */
enum {
	SI_NUM_RESOURCE_PARAMS = 4,

	/* PS only parameters */
	SI_PARAM_ALPHA_REF = SI_NUM_RESOURCE_PARAMS,
	SI_PARAM_PRIM_MASK,
	SI_PARAM_PERSP_SAMPLE,
	SI_PARAM_PERSP_CENTER,
	SI_PARAM_PERSP_CENTROID,
	SI_PARAM_PERSP_PULL_MODEL,
	SI_PARAM_LINEAR_SAMPLE,
	SI_PARAM_LINEAR_CENTER,
	SI_PARAM_LINEAR_CENTROID,
	SI_PARAM_LINE_STIPPLE_TEX,
	SI_PARAM_POS_X_FLOAT,
	SI_PARAM_POS_Y_FLOAT,
	SI_PARAM_POS_Z_FLOAT,
	SI_PARAM_POS_W_FLOAT,
	SI_PARAM_FRONT_FACE,
	SI_PARAM_ANCILLARY,
	SI_PARAM_SAMPLE_COVERAGE,
	SI_PARAM_POS_FIXED_PT,

	SI_NUM_PARAMS = SI_PARAM_POS_FIXED_PT + 9, /* +8 for COLOR[0..1] */
};

/* Fields of driver-defined VS state SGPR. */
/* Clamp vertex color output (only used in VS as VS). */
#define S_VS_STATE_CLAMP_VERTEX_COLOR(x)	(((unsigned)(x) & 0x1) << 0)
#define C_VS_STATE_CLAMP_VERTEX_COLOR		0xFFFFFFFE
#define S_VS_STATE_INDEXED(x)			(((unsigned)(x) & 0x1) << 1)
#define C_VS_STATE_INDEXED			0xFFFFFFFD
#define S_VS_STATE_LS_OUT_PATCH_SIZE(x)		(((unsigned)(x) & 0x1FFF) << 8)
#define C_VS_STATE_LS_OUT_PATCH_SIZE		0xFFE000FF
#define S_VS_STATE_LS_OUT_VERTEX_SIZE(x)	(((unsigned)(x) & 0xFF) << 24)
#define C_VS_STATE_LS_OUT_VERTEX_SIZE		0x00FFFFFF

/* SI-specific system values. */
enum {
	/* Values from set_tess_state. */
	TGSI_SEMANTIC_DEFAULT_TESSOUTER_SI = TGSI_SEMANTIC_COUNT,
	TGSI_SEMANTIC_DEFAULT_TESSINNER_SI,

	/* Up to 4 dwords in user SGPRs for compute shaders. */
	TGSI_SEMANTIC_CS_USER_DATA,
};

enum {
	/* Use a property enum that CS wouldn't use. */
	TGSI_PROPERTY_CS_LOCAL_SIZE = TGSI_PROPERTY_FS_COORD_ORIGIN,

	/* The number of used user data dwords in the range [1, 4]. */
	TGSI_PROPERTY_CS_USER_DATA_DWORDS = TGSI_PROPERTY_FS_COORD_PIXEL_CENTER,

	/* Use a property enum that VS wouldn't use. */
	TGSI_PROPERTY_VS_BLIT_SGPRS = TGSI_PROPERTY_FS_COORD_ORIGIN,

	/* These represent the number of SGPRs the shader uses. */
	SI_VS_BLIT_SGPRS_POS = 3,
	SI_VS_BLIT_SGPRS_POS_COLOR = 7,
	SI_VS_BLIT_SGPRS_POS_TEXCOORD = 9,
};

/* For VS shader key fix_fetch. */
enum {
	SI_FIX_FETCH_NONE = 0,
	SI_FIX_FETCH_A2_SNORM,
	SI_FIX_FETCH_A2_SSCALED,
	SI_FIX_FETCH_A2_SINT,
	SI_FIX_FETCH_RGBA_32_UNORM,
	SI_FIX_FETCH_RGBX_32_UNORM,
	SI_FIX_FETCH_RGBA_32_SNORM,
	SI_FIX_FETCH_RGBX_32_SNORM,
	SI_FIX_FETCH_RGBA_32_USCALED,
	SI_FIX_FETCH_RGBA_32_SSCALED,
	SI_FIX_FETCH_RGBA_32_FIXED,
	SI_FIX_FETCH_RGBX_32_FIXED,
	SI_FIX_FETCH_RG_64_FLOAT,
	SI_FIX_FETCH_RGB_64_FLOAT,
	SI_FIX_FETCH_RGBA_64_FLOAT,
	SI_FIX_FETCH_RGB_8,	/* A = 1.0 */
	SI_FIX_FETCH_RGB_8_INT,	/* A = 1 */
	SI_FIX_FETCH_RGB_16,
	SI_FIX_FETCH_RGB_16_INT,
};

struct si_shader;

/* State of the context creating the shader object. */
struct si_compiler_ctx_state {
	/* Should only be used by si_init_shader_selector_async and
	 * si_build_shader_variant if thread_index == -1 (non-threaded). */
	struct ac_llvm_compiler		*compiler;

	/* Used if thread_index == -1 or if debug.async is true. */
	struct pipe_debug_callback	debug;

	/* Used for creating the log string for gallium/ddebug. */
	bool				is_debug_context;
};

/* A shader selector is a gallium CSO and contains shader variants and
 * binaries for one TGSI program. This can be shared by multiple contexts.
 */
struct si_shader_selector {
	struct pipe_reference	reference;
	struct si_screen	*screen;
	struct util_queue_fence ready;
	struct si_compiler_ctx_state compiler_ctx_state;

	mtx_t		mutex;
	struct si_shader	*first_variant; /* immutable after the first variant */
	struct si_shader	*last_variant; /* mutable */

	/* The compiled TGSI shader expecting a prolog and/or epilog (not
	 * uploaded to a buffer).
	 */
	struct si_shader	*main_shader_part;
	struct si_shader	*main_shader_part_ls; /* as_ls is set in the key */
	struct si_shader	*main_shader_part_es; /* as_es is set in the key */

	struct si_shader	*gs_copy_shader;

	struct tgsi_token       *tokens;
	struct nir_shader       *nir;
	struct pipe_stream_output_info  so;
	struct tgsi_shader_info		info;
	struct tgsi_tessctrl_info	tcs_info;

	/* PIPE_SHADER_[VERTEX|FRAGMENT|...] */
	unsigned	type;
	bool		vs_needs_prolog;
	bool		force_correct_derivs_after_kill;
	unsigned	pa_cl_vs_out_cntl;
	ubyte		clipdist_mask;
	ubyte		culldist_mask;

	/* ES parameters. */
	unsigned	esgs_itemsize; /* vertex stride */
	unsigned	lshs_vertex_stride;

	/* GS parameters. */
	unsigned	gs_input_verts_per_prim;
	unsigned	gs_output_prim;
	unsigned	gs_max_out_vertices;
	unsigned	gs_num_invocations;
	unsigned	max_gs_stream; /* count - 1 */
	unsigned	gsvs_vertex_size;
	unsigned	max_gsvs_emit_size;
	unsigned	enabled_streamout_buffer_mask;

	/* PS parameters. */
	unsigned	color_attr_index[2];
	unsigned	db_shader_control;
	/* Set 0xf or 0x0 (4 bits) per each written output.
	 * ANDed with spi_shader_col_format.
	 */
	unsigned	colors_written_4bit;

	uint64_t	outputs_written_before_ps; /* "get_unique_index" bits */
	uint64_t	outputs_written;	/* "get_unique_index" bits */
	uint32_t	patch_outputs_written;	/* "get_unique_index_patch" bits */

	uint64_t	inputs_read;		/* "get_unique_index" bits */

	/* bitmasks of used descriptor slots */
	uint32_t	active_const_and_shader_buffers;
	uint64_t	active_samplers_and_images;
};

/* Valid shader configurations:
 *
 * API shaders       VS | TCS | TES | GS |pass| PS
 * are compiled as:     |     |     |    |thru|
 *                      |     |     |    |    |
 * Only VS & PS:     VS |     |     |    |    | PS
 * GFX6 - with GS:   ES |     |     | GS | VS | PS
 *      - with tess: LS | HS  | VS  |    |    | PS
 *      - with both: LS | HS  | ES  | GS | VS | PS
 * GFX9 - with GS:   -> |     |     | GS | VS | PS
 *      - with tess: -> | HS  | VS  |    |    | PS
 *      - with both: -> | HS  | ->  | GS | VS | PS
 *
 * -> = merged with the next stage
 */

/* Use the byte alignment for all following structure members for optimal
 * shader key memory footprint.
 */
#pragma pack(push, 1)

/* Common VS bits between the shader key and the prolog key. */
struct si_vs_prolog_bits {
	/* - If neither "is_one" nor "is_fetched" has a bit set, the instance
	 *   divisor is 0.
	 * - If "is_one" has a bit set, the instance divisor is 1.
	 * - If "is_fetched" has a bit set, the instance divisor will be loaded
	 *   from the constant buffer.
	 */
	uint16_t	instance_divisor_is_one;     /* bitmask of inputs */
	uint16_t	instance_divisor_is_fetched; /* bitmask of inputs */
	unsigned	ls_vgpr_fix:1;
};

/* Common TCS bits between the shader key and the epilog key. */
struct si_tcs_epilog_bits {
	unsigned	prim_mode:3;
	unsigned	invoc0_tess_factors_are_def:1;
	unsigned	tes_reads_tess_factors:1;
};

struct si_gs_prolog_bits {
	unsigned	tri_strip_adj_fix:1;
	unsigned	gfx9_prev_is_vs:1;
};

/* Common PS bits between the shader key and the prolog key. */
struct si_ps_prolog_bits {
	unsigned	color_two_side:1;
	unsigned	flatshade_colors:1;
	unsigned	poly_stipple:1;
	unsigned	force_persp_sample_interp:1;
	unsigned	force_linear_sample_interp:1;
	unsigned	force_persp_center_interp:1;
	unsigned	force_linear_center_interp:1;
	unsigned	bc_optimize_for_persp:1;
	unsigned	bc_optimize_for_linear:1;
	unsigned	samplemask_log_ps_iter:3;
};

/* Common PS bits between the shader key and the epilog key. */
struct si_ps_epilog_bits {
	unsigned	spi_shader_col_format;
	unsigned	color_is_int8:8;
	unsigned	color_is_int10:8;
	unsigned	last_cbuf:3;
	unsigned	alpha_func:3;
	unsigned	alpha_to_one:1;
	unsigned	poly_line_smoothing:1;
	unsigned	clamp_color:1;
};

union si_shader_part_key {
	struct {
		struct si_vs_prolog_bits states;
		unsigned	num_input_sgprs:6;
		/* For merged stages such as LS-HS, HS input VGPRs are first. */
		unsigned	num_merged_next_stage_vgprs:3;
		unsigned	last_input:4;
		unsigned	as_ls:1;
		unsigned	as_es:1;
		/* Prologs for monolithic shaders shouldn't set EXEC. */
		unsigned	is_monolithic:1;
	} vs_prolog;
	struct {
		struct si_tcs_epilog_bits states;
	} tcs_epilog;
	struct {
		struct si_gs_prolog_bits states;
		/* Prologs of monolithic shaders shouldn't set EXEC. */
		unsigned	is_monolithic:1;
	} gs_prolog;
	struct {
		struct si_ps_prolog_bits states;
		unsigned	num_input_sgprs:6;
		unsigned	num_input_vgprs:5;
		/* Color interpolation and two-side color selection. */
		unsigned	colors_read:8; /* color input components read */
		unsigned	num_interp_inputs:5; /* BCOLOR is at this location */
		unsigned	face_vgpr_index:5;
		unsigned	ancillary_vgpr_index:5;
		unsigned	wqm:1;
		char		color_attr_index[2];
		signed char	color_interp_vgpr_index[2]; /* -1 == constant */
	} ps_prolog;
	struct {
		struct si_ps_epilog_bits states;
		unsigned	colors_written:8;
		unsigned	writes_z:1;
		unsigned	writes_stencil:1;
		unsigned	writes_samplemask:1;
	} ps_epilog;
};

struct si_shader_key {
	/* Prolog and epilog flags. */
	union {
		struct {
			struct si_vs_prolog_bits prolog;
		} vs;
		struct {
			struct si_vs_prolog_bits ls_prolog; /* for merged LS-HS */
			struct si_shader_selector *ls;   /* for merged LS-HS */
			struct si_tcs_epilog_bits epilog;
		} tcs; /* tessellation control shader */
		struct {
			struct si_vs_prolog_bits vs_prolog; /* for merged ES-GS */
			struct si_shader_selector *es;   /* for merged ES-GS */
			struct si_gs_prolog_bits prolog;
		} gs;
		struct {
			struct si_ps_prolog_bits prolog;
			struct si_ps_epilog_bits epilog;
		} ps;
	} part;

	/* These two are initially set according to the NEXT_SHADER property,
	 * or guessed if the property doesn't seem correct.
	 */
	unsigned as_es:1; /* export shader, which precedes GS */
	unsigned as_ls:1; /* local shader, which precedes TCS */

	/* Flags for monolithic compilation only. */
	struct {
		/* One byte for every input: SI_FIX_FETCH_* enums. */
		uint8_t		vs_fix_fetch[SI_MAX_ATTRIBS];

		union {
			uint64_t	ff_tcs_inputs_to_copy; /* for fixed-func TCS */
			/* When PS needs PrimID and GS is disabled. */
			unsigned	vs_export_prim_id:1;
			struct {
				unsigned interpolate_at_sample_force_center:1;
				unsigned fbfetch_msaa;
				unsigned fbfetch_is_1D;
				unsigned fbfetch_layered;
			} ps;
		} u;
	} mono;

	/* Optimization flags for asynchronous compilation only. */
	struct {
		/* For HW VS (it can be VS, TES, GS) */
		uint64_t	kill_outputs; /* "get_unique_index" bits */
		unsigned	clip_disable:1;

		/* For shaders where monolithic variants have better code.
		 *
		 * This is a flag that has no effect on code generation,
		 * but forces monolithic shaders to be used as soon as
		 * possible, because it's in the "opt" group.
		 */
		unsigned	prefer_mono:1;
	} opt;
};

/* Restore the pack alignment to default. */
#pragma pack(pop)

struct si_shader_config {
	unsigned			num_sgprs;
	unsigned			num_vgprs;
	unsigned			spilled_sgprs;
	unsigned			spilled_vgprs;
	unsigned			private_mem_vgprs;
	unsigned			lds_size;
	unsigned			max_simd_waves;
	unsigned			spi_ps_input_ena;
	unsigned			spi_ps_input_addr;
	unsigned			float_mode;
	unsigned			scratch_bytes_per_wave;
	unsigned			rsrc1;
	unsigned			rsrc2;
};

/* GCN-specific shader info. */
struct si_shader_info {
	ubyte			vs_output_param_offset[SI_MAX_VS_OUTPUTS];
	ubyte			num_input_sgprs;
	ubyte			num_input_vgprs;
	signed char		face_vgpr_index;
	signed char		ancillary_vgpr_index;
	bool			uses_instanceid;
	ubyte			nr_pos_exports;
	ubyte			nr_param_exports;
};

struct si_shader {
	struct si_compiler_ctx_state	compiler_ctx_state;

	struct si_shader_selector	*selector;
	struct si_shader_selector	*previous_stage_sel; /* for refcounting */
	struct si_shader		*next_variant;

	struct si_shader_part		*prolog;
	struct si_shader		*previous_stage; /* for GFX9 */
	struct si_shader_part		*prolog2;
	struct si_shader_part		*epilog;

	struct si_pm4_state		*pm4;
	struct r600_resource		*bo;
	struct r600_resource		*scratch_bo;
	struct si_shader_key		key;
	struct util_queue_fence		ready;
	bool				compilation_failed;
	bool				is_monolithic;
	bool				is_optimized;
	bool				is_binary_shared;
	bool				is_gs_copy_shader;

	/* The following data is all that's needed for binary shaders. */
	struct ac_shader_binary	binary;
	struct si_shader_config		config;
	struct si_shader_info		info;

	/* Shader key + LLVM IR + disassembly + statistics.
	 * Generated for debug contexts only.
	 */
	char				*shader_log;
	size_t				shader_log_size;

	/* For save precompute context registers values. */
	union {
		struct {
			unsigned	vgt_gsvs_ring_offset_1;
			unsigned	vgt_gsvs_ring_offset_2;
			unsigned	vgt_gsvs_ring_offset_3;
			unsigned	vgt_gs_out_prim_type;
			unsigned	vgt_gsvs_ring_itemsize;
			unsigned	vgt_gs_max_vert_out;
			unsigned	vgt_gs_vert_itemsize;
			unsigned	vgt_gs_vert_itemsize_1;
			unsigned	vgt_gs_vert_itemsize_2;
			unsigned	vgt_gs_vert_itemsize_3;
			unsigned	vgt_gs_instance_cnt;
			unsigned	vgt_gs_onchip_cntl;
			unsigned	vgt_gs_max_prims_per_subgroup;
			unsigned	vgt_esgs_ring_itemsize;
		} gs;

		struct {
			unsigned	vgt_gs_mode;
			unsigned	vgt_primitiveid_en;
			unsigned	vgt_reuse_off;
			unsigned	spi_vs_out_config;
			unsigned	spi_shader_pos_format;
			unsigned	pa_cl_vte_cntl;
		} vs;

		struct {
			unsigned	spi_ps_input_ena;
			unsigned	spi_ps_input_addr;
			unsigned	spi_baryc_cntl;
			unsigned	spi_ps_in_control;
			unsigned	spi_shader_z_format;
			unsigned	spi_shader_col_format;
			unsigned	cb_shader_mask;
		} ps;
	} ctx_reg;

	/*For save precompute registers value */
	unsigned vgt_tf_param; /* VGT_TF_PARAM */
	unsigned vgt_vertex_reuse_block_cntl; /* VGT_VERTEX_REUSE_BLOCK_CNTL */
};

struct si_shader_part {
	struct si_shader_part *next;
	union si_shader_part_key key;
	struct ac_shader_binary binary;
	struct si_shader_config config;
};

/* si_shader.c */
struct si_shader *
si_generate_gs_copy_shader(struct si_screen *sscreen,
			   struct ac_llvm_compiler *compiler,
			   struct si_shader_selector *gs_selector,
			   struct pipe_debug_callback *debug);
int si_compile_tgsi_shader(struct si_screen *sscreen,
			   struct ac_llvm_compiler *compiler,
			   struct si_shader *shader,
			   struct pipe_debug_callback *debug);
int si_shader_create(struct si_screen *sscreen, struct ac_llvm_compiler *compiler,
		     struct si_shader *shader,
		     struct pipe_debug_callback *debug);
void si_shader_destroy(struct si_shader *shader);
unsigned si_shader_io_get_unique_index_patch(unsigned semantic_name, unsigned index);
unsigned si_shader_io_get_unique_index(unsigned semantic_name, unsigned index,
				       unsigned is_varying);
int si_shader_binary_upload(struct si_screen *sscreen, struct si_shader *shader);
void si_shader_dump(struct si_screen *sscreen, const struct si_shader *shader,
		    struct pipe_debug_callback *debug, unsigned processor,
		    FILE *f, bool check_debug_option);
void si_shader_dump_stats_for_shader_db(const struct si_shader *shader,
					struct pipe_debug_callback *debug);
void si_multiwave_lds_size_workaround(struct si_screen *sscreen,
				      unsigned *lds_size);
void si_shader_apply_scratch_relocs(struct si_shader *shader,
				    uint64_t scratch_va);
void si_shader_binary_read_config(struct ac_shader_binary *binary,
				  struct si_shader_config *conf,
				  unsigned symbol_offset);
const char *si_get_shader_name(const struct si_shader *shader, unsigned processor);

/* si_shader_nir.c */
void si_nir_scan_shader(const struct nir_shader *nir,
			struct tgsi_shader_info *info);
void si_nir_scan_tess_ctrl(const struct nir_shader *nir,
			   const struct tgsi_shader_info *info,
			   struct tgsi_tessctrl_info *out);
void si_lower_nir(struct si_shader_selector *sel);

/* Inline helpers. */

/* Return the pointer to the main shader part's pointer. */
static inline struct si_shader **
si_get_main_shader_part(struct si_shader_selector *sel,
			struct si_shader_key *key)
{
	if (key->as_ls)
		return &sel->main_shader_part_ls;
	if (key->as_es)
		return &sel->main_shader_part_es;
	return &sel->main_shader_part;
}

static inline bool
si_shader_uses_bindless_samplers(struct si_shader_selector *selector)
{
	return selector ? selector->info.uses_bindless_samplers : false;
}

static inline bool
si_shader_uses_bindless_images(struct si_shader_selector *selector)
{
	return selector ? selector->info.uses_bindless_images : false;
}

void si_destroy_shader_selector(struct si_context *sctx,
			        struct si_shader_selector *sel);

static inline void
si_shader_selector_reference(struct si_context *sctx,
			     struct si_shader_selector **dst,
			     struct si_shader_selector *src)
{
	if (pipe_reference(&(*dst)->reference, &src->reference))
		si_destroy_shader_selector(sctx, *dst);

	*dst = src;
}

#endif
