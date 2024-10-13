/*
 * Copyright (C) 2012 Rob Clark <robclark@freedesktop.org>
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


#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"

#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "util/u_format.h"
#include "util/u_format_s3tc.h"
#include "util/u_screen.h"
#include "util/u_string.h"
#include "util/u_debug.h"

#include "util/os_time.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/sysinfo.h>

#include "freedreno_screen.h"
#include "freedreno_resource.h"
#include "freedreno_fence.h"
#include "freedreno_query.h"
#include "freedreno_util.h"

#include "a2xx/fd2_screen.h"
#include "a3xx/fd3_screen.h"
#include "a4xx/fd4_screen.h"
#include "a5xx/fd5_screen.h"
#include "a6xx/fd6_screen.h"


#include "ir3/ir3_nir.h"

/* XXX this should go away */
#include "state_tracker/drm_driver.h"

static const struct debug_named_value debug_options[] = {
		{"msgs",      FD_DBG_MSGS,   "Print debug messages"},
		{"disasm",    FD_DBG_DISASM, "Dump TGSI and adreno shader disassembly"},
		{"dclear",    FD_DBG_DCLEAR, "Mark all state dirty after clear"},
		{"ddraw",     FD_DBG_DDRAW,  "Mark all state dirty after draw"},
		{"noscis",    FD_DBG_NOSCIS, "Disable scissor optimization"},
		{"direct",    FD_DBG_DIRECT, "Force inline (SS_DIRECT) state loads"},
		{"nobypass",  FD_DBG_NOBYPASS, "Disable GMEM bypass"},
		{"fraghalf",  FD_DBG_FRAGHALF, "Use half-precision in fragment shader"},
		{"nobin",     FD_DBG_NOBIN,  "Disable hw binning"},
		{"optmsgs",   FD_DBG_OPTMSGS,"Enable optimizer debug messages"},
		{"glsl120",   FD_DBG_GLSL120,"Temporary flag to force GLSL 1.20 (rather than 1.30) on a3xx+"},
		{"shaderdb",  FD_DBG_SHADERDB, "Enable shaderdb output"},
		{"flush",     FD_DBG_FLUSH,  "Force flush after every draw"},
		{"deqp",      FD_DBG_DEQP,   "Enable dEQP hacks"},
		{"inorder",   FD_DBG_INORDER,"Disable reordering for draws/blits"},
		{"bstat",     FD_DBG_BSTAT,  "Print batch stats at context destroy"},
		{"nogrow",    FD_DBG_NOGROW, "Disable \"growable\" cmdstream buffers, even if kernel supports it"},
		{"lrz",       FD_DBG_LRZ,    "Enable experimental LRZ support (a5xx+)"},
		{"noindirect",FD_DBG_NOINDR, "Disable hw indirect draws (emulate on CPU)"},
		{"noblit",    FD_DBG_NOBLIT, "Disable blitter (fallback to generic blit path)"},
		{"hiprio",    FD_DBG_HIPRIO, "Force high-priority context"},
		{"ttile",     FD_DBG_TTILE,  "Enable texture tiling (a5xx)"},
		{"perfcntrs", FD_DBG_PERFC,  "Expose performance counters"},
		{"softpin",   FD_DBG_SOFTPIN,"Enable softpin command submission (experimental)"},
		DEBUG_NAMED_VALUE_END
};

DEBUG_GET_ONCE_FLAGS_OPTION(fd_mesa_debug, "FD_MESA_DEBUG", debug_options, 0)

int fd_mesa_debug = 0;
bool fd_binning_enabled = true;
static bool glsl120 = false;

static const struct debug_named_value shader_debug_options[] = {
		{"vs", FD_DBG_SHADER_VS, "Print shader disasm for vertex shaders"},
		{"fs", FD_DBG_SHADER_FS, "Print shader disasm for fragment shaders"},
		{"cs", FD_DBG_SHADER_CS, "Print shader disasm for compute shaders"},
		DEBUG_NAMED_VALUE_END
};

DEBUG_GET_ONCE_FLAGS_OPTION(fd_shader_debug, "FD_SHADER_DEBUG", shader_debug_options, 0)

enum fd_shader_debug fd_shader_debug = 0;

static const char *
fd_screen_get_name(struct pipe_screen *pscreen)
{
	static char buffer[128];
	util_snprintf(buffer, sizeof(buffer), "FD%03d",
			fd_screen(pscreen)->device_id);
	return buffer;
}

static const char *
fd_screen_get_vendor(struct pipe_screen *pscreen)
{
	return "freedreno";
}

static const char *
fd_screen_get_device_vendor(struct pipe_screen *pscreen)
{
	return "Qualcomm";
}


static uint64_t
fd_screen_get_timestamp(struct pipe_screen *pscreen)
{
	struct fd_screen *screen = fd_screen(pscreen);

	if (screen->has_timestamp) {
		uint64_t n;
		fd_pipe_get_param(screen->pipe, FD_TIMESTAMP, &n);
		debug_assert(screen->max_freq > 0);
		return n * 1000000000 / screen->max_freq;
	} else {
		int64_t cpu_time = os_time_get() * 1000;
		return cpu_time + screen->cpu_gpu_time_delta;
	}

}

static void
fd_screen_destroy(struct pipe_screen *pscreen)
{
	struct fd_screen *screen = fd_screen(pscreen);

	if (screen->pipe)
		fd_pipe_del(screen->pipe);

	if (screen->dev)
		fd_device_del(screen->dev);

	fd_bc_fini(&screen->batch_cache);

	slab_destroy_parent(&screen->transfer_pool);

	mtx_destroy(&screen->lock);

	ralloc_free(screen->compiler);

	free(screen->perfcntr_queries);
	free(screen);
}

/*
TODO either move caps to a2xx/a3xx specific code, or maybe have some
tables for things that differ if the delta is not too much..
 */
static int
fd_screen_get_param(struct pipe_screen *pscreen, enum pipe_cap param)
{
	struct fd_screen *screen = fd_screen(pscreen);

	/* this is probably not totally correct.. but it's a start: */
	switch (param) {
	/* Supported features (boolean caps). */
	case PIPE_CAP_NPOT_TEXTURES:
	case PIPE_CAP_MIXED_FRAMEBUFFER_SIZES:
	case PIPE_CAP_ANISOTROPIC_FILTER:
	case PIPE_CAP_POINT_SPRITE:
	case PIPE_CAP_BLEND_EQUATION_SEPARATE:
	case PIPE_CAP_TEXTURE_SWIZZLE:
	case PIPE_CAP_MIXED_COLORBUFFER_FORMATS:
	case PIPE_CAP_TGSI_FS_COORD_ORIGIN_UPPER_LEFT:
	case PIPE_CAP_TGSI_FS_COORD_PIXEL_CENTER_INTEGER:
	case PIPE_CAP_SEAMLESS_CUBE_MAP:
	case PIPE_CAP_VERTEX_COLOR_UNCLAMPED:
	case PIPE_CAP_QUADS_FOLLOW_PROVOKING_VERTEX_CONVENTION:
	case PIPE_CAP_VERTEX_BUFFER_OFFSET_4BYTE_ALIGNED_ONLY:
	case PIPE_CAP_VERTEX_BUFFER_STRIDE_4BYTE_ALIGNED_ONLY:
	case PIPE_CAP_VERTEX_ELEMENT_SRC_OFFSET_4BYTE_ALIGNED_ONLY:
	case PIPE_CAP_BUFFER_MAP_PERSISTENT_COHERENT:
	case PIPE_CAP_STRING_MARKER:
	case PIPE_CAP_MIXED_COLOR_DEPTH_BITS:
	case PIPE_CAP_TEXTURE_BARRIER:
	case PIPE_CAP_INVALIDATE_BUFFER:
		return 1;

	case PIPE_CAP_VERTEXID_NOBASE:
		return is_a3xx(screen) || is_a4xx(screen);

	case PIPE_CAP_COMPUTE:
		return has_compute(screen);

	case PIPE_CAP_PREFER_BLIT_BASED_TEXTURE_TRANSFER:
	case PIPE_CAP_PCI_GROUP:
	case PIPE_CAP_PCI_BUS:
	case PIPE_CAP_PCI_DEVICE:
	case PIPE_CAP_PCI_FUNCTION:
	case PIPE_CAP_DEPTH_CLIP_DISABLE_SEPARATE:
		return 0;

	case PIPE_CAP_SM3:
	case PIPE_CAP_PRIMITIVE_RESTART:
	case PIPE_CAP_TGSI_INSTANCEID:
	case PIPE_CAP_VERTEX_ELEMENT_INSTANCE_DIVISOR:
	case PIPE_CAP_INDEP_BLEND_ENABLE:
	case PIPE_CAP_INDEP_BLEND_FUNC:
	case PIPE_CAP_TEXTURE_BUFFER_OBJECTS:
	case PIPE_CAP_TEXTURE_HALF_FLOAT_LINEAR:
	case PIPE_CAP_CONDITIONAL_RENDER:
	case PIPE_CAP_CONDITIONAL_RENDER_INVERTED:
	case PIPE_CAP_SEAMLESS_CUBE_MAP_PER_TEXTURE:
	case PIPE_CAP_CLIP_HALFZ:
		return is_a3xx(screen) || is_a4xx(screen) || is_a5xx(screen) || is_a6xx(screen);

	case PIPE_CAP_FAKE_SW_MSAA:
		return !fd_screen_get_param(pscreen, PIPE_CAP_TEXTURE_MULTISAMPLE);

	case PIPE_CAP_TEXTURE_MULTISAMPLE:
		return is_a5xx(screen) || is_a6xx(screen);

	case PIPE_CAP_DEPTH_CLIP_DISABLE:
		return is_a3xx(screen) || is_a4xx(screen);

	case PIPE_CAP_POLYGON_OFFSET_CLAMP:
		return is_a5xx(screen) || is_a6xx(screen);

	case PIPE_CAP_TEXTURE_BUFFER_OFFSET_ALIGNMENT:
		if (is_a3xx(screen)) return 16;
		if (is_a4xx(screen)) return 32;
		if (is_a5xx(screen)) return 32;
		if (is_a6xx(screen)) return 32;
		return 0;
	case PIPE_CAP_MAX_TEXTURE_BUFFER_SIZE:
		/* We could possibly emulate more by pretending 2d/rect textures and
		 * splitting high bits of index into 2nd dimension..
		 */
		if (is_a3xx(screen)) return 8192;
		if (is_a4xx(screen)) return 16384;
		if (is_a5xx(screen)) return 16384;
		if (is_a6xx(screen)) return 16384;
		return 0;

	case PIPE_CAP_TEXTURE_FLOAT_LINEAR:
	case PIPE_CAP_CUBE_MAP_ARRAY:
	case PIPE_CAP_SAMPLER_VIEW_TARGET:
	case PIPE_CAP_TEXTURE_QUERY_LOD:
		return is_a4xx(screen) || is_a5xx(screen) || is_a6xx(screen);

	case PIPE_CAP_START_INSTANCE:
		/* Note that a5xx can do this, it just can't (at least with
		 * current firmware) do draw_indirect with base_instance.
		 * Since draw_indirect is needed sooner (gles31 and gl40 vs
		 * gl42), hide base_instance on a5xx.  :-/
		 */
		return is_a4xx(screen);

	case PIPE_CAP_CONSTANT_BUFFER_OFFSET_ALIGNMENT:
		return 64;

	case PIPE_CAP_GLSL_FEATURE_LEVEL:
	case PIPE_CAP_GLSL_FEATURE_LEVEL_COMPATIBILITY:
		if (glsl120)
			return 120;
		return is_ir3(screen) ? 140 : 120;

	case PIPE_CAP_SHADER_BUFFER_OFFSET_ALIGNMENT:
		if (is_a5xx(screen) || is_a6xx(screen))
			return 4;
		return 0;

	case PIPE_CAP_MAX_TEXTURE_GATHER_COMPONENTS:
		if (is_a4xx(screen) || is_a5xx(screen) || is_a6xx(screen))
			return 4;
		return 0;

	/* TODO if we need this, do it in nir/ir3 backend to avoid breaking precompile: */
	case PIPE_CAP_FORCE_PERSAMPLE_INTERP:
		return 0;

	case PIPE_CAP_ALLOW_MAPPED_BUFFERS_DURING_EXECUTION:
		return 0;

	case PIPE_CAP_CONTEXT_PRIORITY_MASK:
		return screen->priority_mask;

	case PIPE_CAP_DRAW_INDIRECT:
		if (is_a4xx(screen) || is_a5xx(screen) || is_a6xx(screen))
			return 1;
		return 0;

	case PIPE_CAP_FRAMEBUFFER_NO_ATTACHMENT:
		if (is_a4xx(screen) || is_a5xx(screen) || is_a6xx(screen))
			return 1;
		return 0;

	case PIPE_CAP_LOAD_CONSTBUF:
		/* name is confusing, but this turns on std430 packing */
		if (is_ir3(screen))
			return 1;
		return 0;

	case PIPE_CAP_MAX_VIEWPORTS:
		return 1;

	case PIPE_CAP_SHAREABLE_SHADERS:
	case PIPE_CAP_GLSL_OPTIMIZE_CONSERVATIVELY:
	/* manage the variants for these ourself, to avoid breaking precompile: */
	case PIPE_CAP_FRAGMENT_COLOR_CLAMPED:
	case PIPE_CAP_VERTEX_COLOR_CLAMPED:
		if (is_ir3(screen))
			return 1;
		return 0;

	/* Stream output. */
	case PIPE_CAP_MAX_STREAM_OUTPUT_BUFFERS:
		if (is_ir3(screen))
			return PIPE_MAX_SO_BUFFERS;
		return 0;
	case PIPE_CAP_STREAM_OUTPUT_PAUSE_RESUME:
	case PIPE_CAP_STREAM_OUTPUT_INTERLEAVE_BUFFERS:
		if (is_ir3(screen))
			return 1;
		return 0;
	case PIPE_CAP_MAX_STREAM_OUTPUT_SEPARATE_COMPONENTS:
	case PIPE_CAP_MAX_STREAM_OUTPUT_INTERLEAVED_COMPONENTS:
		if (is_ir3(screen))
			return 16 * 4;   /* should only be shader out limit? */
		return 0;

	/* Texturing. */
	case PIPE_CAP_MAX_TEXTURE_2D_LEVELS:
	case PIPE_CAP_MAX_TEXTURE_CUBE_LEVELS:
		return MAX_MIP_LEVELS;
	case PIPE_CAP_MAX_TEXTURE_3D_LEVELS:
		return 11;

	case PIPE_CAP_MAX_TEXTURE_ARRAY_LAYERS:
		return (is_a3xx(screen) || is_a4xx(screen) || is_a5xx(screen) || is_a6xx(screen)) ? 256 : 0;

	/* Render targets. */
	case PIPE_CAP_MAX_RENDER_TARGETS:
		return screen->max_rts;
	case PIPE_CAP_MAX_DUAL_SOURCE_RENDER_TARGETS:
		return is_a3xx(screen) ? 1 : 0;

	/* Queries. */
	case PIPE_CAP_OCCLUSION_QUERY:
		return is_a3xx(screen) || is_a4xx(screen) || is_a5xx(screen) || is_a6xx(screen);
	case PIPE_CAP_QUERY_TIMESTAMP:
	case PIPE_CAP_QUERY_TIME_ELAPSED:
		/* only a4xx, requires new enough kernel so we know max_freq: */
		return (screen->max_freq > 0) && (is_a4xx(screen) || is_a5xx(screen) || is_a6xx(screen));

	case PIPE_CAP_VENDOR_ID:
		return 0x5143;
	case PIPE_CAP_DEVICE_ID:
		return 0xFFFFFFFF;
	case PIPE_CAP_ACCELERATED:
		return 1;
	case PIPE_CAP_VIDEO_MEMORY:
		DBG("FINISHME: The value returned is incorrect\n");
		return 10;
	case PIPE_CAP_UMA:
		return 1;
	case PIPE_CAP_NATIVE_FENCE_FD:
		return fd_device_version(screen->dev) >= FD_VERSION_FENCE_FD;
	default:
		return u_pipe_screen_get_param_defaults(pscreen, param);
	}
}

static float
fd_screen_get_paramf(struct pipe_screen *pscreen, enum pipe_capf param)
{
	switch (param) {
	case PIPE_CAPF_MAX_LINE_WIDTH:
	case PIPE_CAPF_MAX_LINE_WIDTH_AA:
		/* NOTE: actual value is 127.0f, but this is working around a deqp
		 * bug.. dEQP-GLES3.functional.rasterization.primitives.lines_wide
		 * uses too small of a render target size, and gets confused when
		 * the lines start going offscreen.
		 *
		 * See: https://code.google.com/p/android/issues/detail?id=206513
		 */
		if (fd_mesa_debug & FD_DBG_DEQP)
			return 48.0f;
		return 127.0f;
	case PIPE_CAPF_MAX_POINT_WIDTH:
	case PIPE_CAPF_MAX_POINT_WIDTH_AA:
		return 4092.0f;
	case PIPE_CAPF_MAX_TEXTURE_ANISOTROPY:
		return 16.0f;
	case PIPE_CAPF_MAX_TEXTURE_LOD_BIAS:
		return 15.0f;
	case PIPE_CAPF_MIN_CONSERVATIVE_RASTER_DILATE:
	case PIPE_CAPF_MAX_CONSERVATIVE_RASTER_DILATE:
	case PIPE_CAPF_CONSERVATIVE_RASTER_DILATE_GRANULARITY:
		return 0.0f;
	}
	debug_printf("unknown paramf %d\n", param);
	return 0;
}

static int
fd_screen_get_shader_param(struct pipe_screen *pscreen,
		enum pipe_shader_type shader,
		enum pipe_shader_cap param)
{
	struct fd_screen *screen = fd_screen(pscreen);

	switch(shader)
	{
	case PIPE_SHADER_FRAGMENT:
	case PIPE_SHADER_VERTEX:
		break;
	case PIPE_SHADER_COMPUTE:
		if (has_compute(screen))
			break;
		return 0;
	case PIPE_SHADER_GEOMETRY:
		/* maye we could emulate.. */
		return 0;
	default:
		DBG("unknown shader type %d", shader);
		return 0;
	}

	/* this is probably not totally correct.. but it's a start: */
	switch (param) {
	case PIPE_SHADER_CAP_MAX_INSTRUCTIONS:
	case PIPE_SHADER_CAP_MAX_ALU_INSTRUCTIONS:
	case PIPE_SHADER_CAP_MAX_TEX_INSTRUCTIONS:
	case PIPE_SHADER_CAP_MAX_TEX_INDIRECTIONS:
		return 16384;
	case PIPE_SHADER_CAP_MAX_CONTROL_FLOW_DEPTH:
		return 8; /* XXX */
	case PIPE_SHADER_CAP_MAX_INPUTS:
	case PIPE_SHADER_CAP_MAX_OUTPUTS:
		return 16;
	case PIPE_SHADER_CAP_MAX_TEMPS:
		return 64; /* Max native temporaries. */
	case PIPE_SHADER_CAP_MAX_CONST_BUFFER_SIZE:
		/* NOTE: seems to be limit for a3xx is actually 512 but
		 * split between VS and FS.  Use lower limit of 256 to
		 * avoid getting into impossible situations:
		 */
		return ((is_a3xx(screen) || is_a4xx(screen) || is_a5xx(screen) || is_a6xx(screen)) ? 4096 : 64) * sizeof(float[4]);
	case PIPE_SHADER_CAP_MAX_CONST_BUFFERS:
		return is_ir3(screen) ? 16 : 1;
	case PIPE_SHADER_CAP_TGSI_CONT_SUPPORTED:
		return 1;
	case PIPE_SHADER_CAP_INDIRECT_INPUT_ADDR:
	case PIPE_SHADER_CAP_INDIRECT_OUTPUT_ADDR:
		/* Technically this should be the same as for TEMP/CONST, since
		 * everything is just normal registers.  This is just temporary
		 * hack until load_input/store_output handle arrays in a similar
		 * way as load_var/store_var..
		 */
		return 0;
	case PIPE_SHADER_CAP_INDIRECT_TEMP_ADDR:
	case PIPE_SHADER_CAP_INDIRECT_CONST_ADDR:
		/* a2xx compiler doesn't handle indirect: */
		return is_ir3(screen) ? 1 : 0;
	case PIPE_SHADER_CAP_SUBROUTINES:
	case PIPE_SHADER_CAP_TGSI_DROUND_SUPPORTED:
	case PIPE_SHADER_CAP_TGSI_DFRACEXP_DLDEXP_SUPPORTED:
	case PIPE_SHADER_CAP_TGSI_LDEXP_SUPPORTED:
	case PIPE_SHADER_CAP_TGSI_FMA_SUPPORTED:
	case PIPE_SHADER_CAP_TGSI_ANY_INOUT_DECL_RANGE:
	case PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTERS:
	case PIPE_SHADER_CAP_LOWER_IF_THRESHOLD:
	case PIPE_SHADER_CAP_TGSI_SKIP_MERGE_REGISTERS:
	case PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTER_BUFFERS:
		return 0;
	case PIPE_SHADER_CAP_TGSI_SQRT_SUPPORTED:
		return 1;
	case PIPE_SHADER_CAP_INTEGERS:
		if (glsl120)
			return 0;
		return is_ir3(screen) ? 1 : 0;
	case PIPE_SHADER_CAP_INT64_ATOMICS:
		return 0;
	case PIPE_SHADER_CAP_FP16:
		return 0;
	case PIPE_SHADER_CAP_MAX_TEXTURE_SAMPLERS:
	case PIPE_SHADER_CAP_MAX_SAMPLER_VIEWS:
		return 16;
	case PIPE_SHADER_CAP_PREFERRED_IR:
		if (is_ir3(screen))
			return PIPE_SHADER_IR_NIR;
		return PIPE_SHADER_IR_TGSI;
	case PIPE_SHADER_CAP_SUPPORTED_IRS:
		if (is_ir3(screen)) {
			return (1 << PIPE_SHADER_IR_NIR) | (1 << PIPE_SHADER_IR_TGSI);
		} else {
			return (1 << PIPE_SHADER_IR_TGSI);
		}
		return 0;
	case PIPE_SHADER_CAP_MAX_UNROLL_ITERATIONS_HINT:
		return 32;
	case PIPE_SHADER_CAP_SCALAR_ISA:
		return is_ir3(screen) ? 1 : 0;
	case PIPE_SHADER_CAP_MAX_SHADER_BUFFERS:
	case PIPE_SHADER_CAP_MAX_SHADER_IMAGES:
		if (is_a5xx(screen) || is_a6xx(screen)) {
			/* a5xx (and a4xx for that matter) has one state-block
			 * for compute-shader SSBO's and another that is shared
			 * by VS/HS/DS/GS/FS..  so to simplify things for now
			 * just advertise SSBOs for FS and CS.  We could possibly
			 * do what blob does, and partition the space for
			 * VS/HS/DS/GS/FS.  The blob advertises:
			 *
			 *   GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS: 4
			 *   GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS: 4
			 *   GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS: 4
			 *   GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS: 4
			 *   GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS: 4
			 *   GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS: 24
			 *   GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS: 24
			 *
			 * I think that way we could avoid having to patch shaders
			 * for actual SSBO indexes by using a static partitioning.
			 *
			 * Note same state block is used for images and buffers,
			 * but images also need texture state for read access
			 * (isam/isam.3d)
			 */
			switch(shader)
			{
			case PIPE_SHADER_FRAGMENT:
			case PIPE_SHADER_COMPUTE:
				return 24;
			default:
				return 0;
			}
		}
		return 0;
	}
	debug_printf("unknown shader param %d\n", param);
	return 0;
}

/* TODO depending on how much the limits differ for a3xx/a4xx, maybe move this
 * into per-generation backend?
 */
static int
fd_get_compute_param(struct pipe_screen *pscreen, enum pipe_shader_ir ir_type,
		enum pipe_compute_cap param, void *ret)
{
	struct fd_screen *screen = fd_screen(pscreen);
	const char * const ir = "ir3";

	if (!has_compute(screen))
		return 0;

#define RET(x) do {                  \
   if (ret)                          \
      memcpy(ret, x, sizeof(x));     \
   return sizeof(x);                 \
} while (0)

	switch (param) {
	case PIPE_COMPUTE_CAP_ADDRESS_BITS:
// don't expose 64b pointer support yet, until ir3 supports 64b
// math, otherwise spir64 target is used and we get 64b pointer
// calculations that we can't do yet
//		if (is_a5xx(screen))
//			RET((uint32_t []){ 64 });
		RET((uint32_t []){ 32 });

	case PIPE_COMPUTE_CAP_IR_TARGET:
		if (ret)
			sprintf(ret, ir);
		return strlen(ir) * sizeof(char);

	case PIPE_COMPUTE_CAP_GRID_DIMENSION:
		RET((uint64_t []) { 3 });

	case PIPE_COMPUTE_CAP_MAX_GRID_SIZE:
		RET(((uint64_t []) { 65535, 65535, 65535 }));

	case PIPE_COMPUTE_CAP_MAX_BLOCK_SIZE:
		RET(((uint64_t []) { 1024, 1024, 64 }));

	case PIPE_COMPUTE_CAP_MAX_THREADS_PER_BLOCK:
		RET((uint64_t []) { 1024 });

	case PIPE_COMPUTE_CAP_MAX_GLOBAL_SIZE:
		RET((uint64_t []) { screen->ram_size });

	case PIPE_COMPUTE_CAP_MAX_LOCAL_SIZE:
		RET((uint64_t []) { 32768 });

	case PIPE_COMPUTE_CAP_MAX_PRIVATE_SIZE:
	case PIPE_COMPUTE_CAP_MAX_INPUT_SIZE:
		RET((uint64_t []) { 4096 });

	case PIPE_COMPUTE_CAP_MAX_MEM_ALLOC_SIZE:
		RET((uint64_t []) { screen->ram_size });

	case PIPE_COMPUTE_CAP_MAX_CLOCK_FREQUENCY:
		RET((uint32_t []) { screen->max_freq / 1000000 });

	case PIPE_COMPUTE_CAP_MAX_COMPUTE_UNITS:
		RET((uint32_t []) { 9999 });  // TODO

	case PIPE_COMPUTE_CAP_IMAGES_SUPPORTED:
		RET((uint32_t []) { 1 });

	case PIPE_COMPUTE_CAP_SUBGROUP_SIZE:
		RET((uint32_t []) { 32 });  // TODO

	case PIPE_COMPUTE_CAP_MAX_VARIABLE_THREADS_PER_BLOCK:
		RET((uint64_t []) { 1024 }); // TODO
	}

	return 0;
}

static const void *
fd_get_compiler_options(struct pipe_screen *pscreen,
		enum pipe_shader_ir ir, unsigned shader)
{
	struct fd_screen *screen = fd_screen(pscreen);

	if (is_ir3(screen))
		return ir3_get_compiler_options(screen->compiler);

	return NULL;
}

boolean
fd_screen_bo_get_handle(struct pipe_screen *pscreen,
		struct fd_bo *bo,
		unsigned stride,
		struct winsys_handle *whandle)
{
	whandle->stride = stride;

	if (whandle->type == WINSYS_HANDLE_TYPE_SHARED) {
		return fd_bo_get_name(bo, &whandle->handle) == 0;
	} else if (whandle->type == WINSYS_HANDLE_TYPE_KMS) {
		whandle->handle = fd_bo_handle(bo);
		return TRUE;
	} else if (whandle->type == WINSYS_HANDLE_TYPE_FD) {
		whandle->handle = fd_bo_dmabuf(bo);
		return TRUE;
	} else {
		return FALSE;
	}
}

struct fd_bo *
fd_screen_bo_from_handle(struct pipe_screen *pscreen,
		struct winsys_handle *whandle)
{
	struct fd_screen *screen = fd_screen(pscreen);
	struct fd_bo *bo;

	if (whandle->type == WINSYS_HANDLE_TYPE_SHARED) {
		bo = fd_bo_from_name(screen->dev, whandle->handle);
	} else if (whandle->type == WINSYS_HANDLE_TYPE_KMS) {
		bo = fd_bo_from_handle(screen->dev, whandle->handle, 0);
	} else if (whandle->type == WINSYS_HANDLE_TYPE_FD) {
		bo = fd_bo_from_dmabuf(screen->dev, whandle->handle);
	} else {
		DBG("Attempt to import unsupported handle type %d", whandle->type);
		return NULL;
	}

	if (!bo) {
		DBG("ref name 0x%08x failed", whandle->handle);
		return NULL;
	}

	return bo;
}

struct pipe_screen *
fd_screen_create(struct fd_device *dev)
{
	struct fd_screen *screen = CALLOC_STRUCT(fd_screen);
	struct pipe_screen *pscreen;
	uint64_t val;

	fd_mesa_debug = debug_get_option_fd_mesa_debug();
	fd_shader_debug = debug_get_option_fd_shader_debug();

	if (fd_mesa_debug & FD_DBG_NOBIN)
		fd_binning_enabled = false;

	glsl120 = !!(fd_mesa_debug & FD_DBG_GLSL120);

	if (!screen)
		return NULL;

	pscreen = &screen->base;

	screen->dev = dev;
	screen->refcnt = 1;

	// maybe this should be in context?
	screen->pipe = fd_pipe_new(screen->dev, FD_PIPE_3D);
	if (!screen->pipe) {
		DBG("could not create 3d pipe");
		goto fail;
	}

	if (fd_pipe_get_param(screen->pipe, FD_GMEM_SIZE, &val)) {
		DBG("could not get GMEM size");
		goto fail;
	}
	screen->gmemsize_bytes = val;

	if (fd_pipe_get_param(screen->pipe, FD_DEVICE_ID, &val)) {
		DBG("could not get device-id");
		goto fail;
	}
	screen->device_id = val;

	if (fd_pipe_get_param(screen->pipe, FD_MAX_FREQ, &val)) {
		DBG("could not get gpu freq");
		/* this limits what performance related queries are
		 * supported but is not fatal
		 */
		screen->max_freq = 0;
	} else {
		screen->max_freq = val;
		if (fd_pipe_get_param(screen->pipe, FD_TIMESTAMP, &val) == 0)
			screen->has_timestamp = true;
	}

	if (fd_pipe_get_param(screen->pipe, FD_GPU_ID, &val)) {
		DBG("could not get gpu-id");
		goto fail;
	}
	screen->gpu_id = val;

	if (fd_pipe_get_param(screen->pipe, FD_CHIP_ID, &val)) {
		DBG("could not get chip-id");
		/* older kernels may not have this property: */
		unsigned core  = screen->gpu_id / 100;
		unsigned major = (screen->gpu_id % 100) / 10;
		unsigned minor = screen->gpu_id % 10;
		unsigned patch = 0;  /* assume the worst */
		val = (patch & 0xff) | ((minor & 0xff) << 8) |
			((major & 0xff) << 16) | ((core & 0xff) << 24);
	}
	screen->chip_id = val;

	if (fd_pipe_get_param(screen->pipe, FD_NR_RINGS, &val)) {
		DBG("could not get # of rings");
		screen->priority_mask = 0;
	} else {
		/* # of rings equates to number of unique priority values: */
		screen->priority_mask = (1 << val) - 1;
	}

	struct sysinfo si;
	sysinfo(&si);
	screen->ram_size = si.totalram;

	DBG("Pipe Info:");
	DBG(" GPU-id:          %d", screen->gpu_id);
	DBG(" Chip-id:         0x%08x", screen->chip_id);
	DBG(" GMEM size:       0x%08x", screen->gmemsize_bytes);

	/* explicitly checking for GPU revisions that are known to work.  This
	 * may be overly conservative for a3xx, where spoofing the gpu_id with
	 * the blob driver seems to generate identical cmdstream dumps.  But
	 * on a2xx, there seem to be small differences between the GPU revs
	 * so it is probably better to actually test first on real hardware
	 * before enabling:
	 *
	 * If you have a different adreno version, feel free to add it to one
	 * of the cases below and see what happens.  And if it works, please
	 * send a patch ;-)
	 */
	switch (screen->gpu_id) {
	case 205:
	case 220:
		fd2_screen_init(pscreen);
		break;
	case 305:
	case 307:
	case 320:
	case 330:
		fd3_screen_init(pscreen);
		break;
	case 420:
	case 430:
		fd4_screen_init(pscreen);
		break;
	case 530:
		fd5_screen_init(pscreen);
		break;
	case 630:
		fd6_screen_init(pscreen);
		break;
	default:
		debug_printf("unsupported GPU: a%03d\n", screen->gpu_id);
		goto fail;
	}

	if (screen->gpu_id >= 600) {
		screen->gmem_alignw = 32;
		screen->gmem_alignh = 32;
		screen->num_vsc_pipes = 32;
	} else if (screen->gpu_id >= 500) {
		screen->gmem_alignw = 64;
		screen->gmem_alignh = 32;
		screen->num_vsc_pipes = 16;
	} else {
		screen->gmem_alignw = 32;
		screen->gmem_alignh = 32;
		screen->num_vsc_pipes = 8;
	}

	/* NOTE: don't enable reordering on a2xx, since completely untested.
	 * Also, don't enable if we have too old of a kernel to support
	 * growable cmdstream buffers, since memory requirement for cmdstream
	 * buffers would be too much otherwise.
	 */
	if ((screen->gpu_id >= 300) && (fd_device_version(dev) >= FD_VERSION_UNLIMITED_CMDS))
		screen->reorder = !(fd_mesa_debug & FD_DBG_INORDER);

	fd_bc_init(&screen->batch_cache);

	(void) mtx_init(&screen->lock, mtx_plain);

	pscreen->destroy = fd_screen_destroy;
	pscreen->get_param = fd_screen_get_param;
	pscreen->get_paramf = fd_screen_get_paramf;
	pscreen->get_shader_param = fd_screen_get_shader_param;
	pscreen->get_compute_param = fd_get_compute_param;
	pscreen->get_compiler_options = fd_get_compiler_options;

	fd_resource_screen_init(pscreen);
	fd_query_screen_init(pscreen);

	pscreen->get_name = fd_screen_get_name;
	pscreen->get_vendor = fd_screen_get_vendor;
	pscreen->get_device_vendor = fd_screen_get_device_vendor;

	pscreen->get_timestamp = fd_screen_get_timestamp;

	pscreen->fence_reference = fd_fence_ref;
	pscreen->fence_finish = fd_fence_finish;
	pscreen->fence_get_fd = fd_fence_get_fd;

	slab_create_parent(&screen->transfer_pool, sizeof(struct fd_transfer), 16);

	return pscreen;

fail:
	fd_screen_destroy(pscreen);
	return NULL;
}
