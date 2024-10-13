/*
 * Copyright (c) 2012-2015 Etnaviv Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Wladimir J. van der Laan <laanwj@gmail.com>
 *    Christian Gmeiner <christian.gmeiner@gmail.com>
 */

#include "etnaviv_screen.h"

#include "hw/common.xml.h"

#include "etnaviv_compiler.h"
#include "etnaviv_context.h"
#include "etnaviv_debug.h"
#include "etnaviv_fence.h"
#include "etnaviv_format.h"
#include "etnaviv_query.h"
#include "etnaviv_resource.h"
#include "etnaviv_translate.h"

#include "util/os_time.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_screen.h"
#include "util/u_string.h"

#include "state_tracker/drm_driver.h"

#include <drm_fourcc.h>

#define ETNA_DRM_VERSION(major, minor) ((major) << 16 | (minor))
#define ETNA_DRM_VERSION_FENCE_FD      ETNA_DRM_VERSION(1, 1)
#define ETNA_DRM_VERSION_PERFMON       ETNA_DRM_VERSION(1, 2)

static const struct debug_named_value debug_options[] = {
   {"dbg_msgs",       ETNA_DBG_MSGS, "Print debug messages"},
   {"frame_msgs",     ETNA_DBG_FRAME_MSGS, "Print frame messages"},
   {"resource_msgs",  ETNA_DBG_RESOURCE_MSGS, "Print resource messages"},
   {"compiler_msgs",  ETNA_DBG_COMPILER_MSGS, "Print compiler messages"},
   {"linker_msgs",    ETNA_DBG_LINKER_MSGS, "Print linker messages"},
   {"dump_shaders",   ETNA_DBG_DUMP_SHADERS, "Dump shaders"},
   {"no_ts",          ETNA_DBG_NO_TS, "Disable TS"},
   {"no_autodisable", ETNA_DBG_NO_AUTODISABLE, "Disable autodisable"},
   {"no_supertile",   ETNA_DBG_NO_SUPERTILE, "Disable supertiles"},
   {"no_early_z",     ETNA_DBG_NO_EARLY_Z, "Disable early z"},
   {"cflush_all",     ETNA_DBG_CFLUSH_ALL, "Flush every cash before state update"},
   {"msaa2x",         ETNA_DBG_MSAA_2X, "Force 2x msaa"},
   {"msaa4x",         ETNA_DBG_MSAA_4X, "Force 4x msaa"},
   {"flush_all",      ETNA_DBG_FLUSH_ALL, "Flush after every rendered primitive"},
   {"zero",           ETNA_DBG_ZERO, "Zero all resources after allocation"},
   {"draw_stall",     ETNA_DBG_DRAW_STALL, "Stall FE/PE after each rendered primitive"},
   {"shaderdb",       ETNA_DBG_SHADERDB, "Enable shaderdb output"},
   {"no_singlebuffer",ETNA_DBG_NO_SINGLEBUF, "Disable single buffer feature"},
   DEBUG_NAMED_VALUE_END
};

DEBUG_GET_ONCE_FLAGS_OPTION(etna_mesa_debug, "ETNA_MESA_DEBUG", debug_options, 0)
int etna_mesa_debug = 0;

static void
etna_screen_destroy(struct pipe_screen *pscreen)
{
   struct etna_screen *screen = etna_screen(pscreen);

   if (screen->perfmon)
      etna_perfmon_del(screen->perfmon);

   if (screen->pipe)
      etna_pipe_del(screen->pipe);

   if (screen->gpu)
      etna_gpu_del(screen->gpu);

   if (screen->ro)
      FREE(screen->ro);

   if (screen->dev)
      etna_device_del(screen->dev);

   FREE(screen);
}

static const char *
etna_screen_get_name(struct pipe_screen *pscreen)
{
   struct etna_screen *priv = etna_screen(pscreen);
   static char buffer[128];

   util_snprintf(buffer, sizeof(buffer), "Vivante GC%x rev %04x", priv->model,
                 priv->revision);

   return buffer;
}

static const char *
etna_screen_get_vendor(struct pipe_screen *pscreen)
{
   return "etnaviv";
}

static const char *
etna_screen_get_device_vendor(struct pipe_screen *pscreen)
{
   return "Vivante";
}

static int
etna_screen_get_param(struct pipe_screen *pscreen, enum pipe_cap param)
{
   struct etna_screen *screen = etna_screen(pscreen);

   switch (param) {
   /* Supported features (boolean caps). */
   case PIPE_CAP_ANISOTROPIC_FILTER:
   case PIPE_CAP_POINT_SPRITE:
   case PIPE_CAP_BLEND_EQUATION_SEPARATE:
   case PIPE_CAP_TGSI_FS_COORD_ORIGIN_UPPER_LEFT:
   case PIPE_CAP_TGSI_FS_COORD_PIXEL_CENTER_HALF_INTEGER:
   case PIPE_CAP_SM3:
   case PIPE_CAP_TEXTURE_BARRIER:
   case PIPE_CAP_QUADS_FOLLOW_PROVOKING_VERTEX_CONVENTION:
   case PIPE_CAP_VERTEX_BUFFER_OFFSET_4BYTE_ALIGNED_ONLY:
   case PIPE_CAP_VERTEX_BUFFER_STRIDE_4BYTE_ALIGNED_ONLY:
   case PIPE_CAP_VERTEX_ELEMENT_SRC_OFFSET_4BYTE_ALIGNED_ONLY:
   case PIPE_CAP_TGSI_TEXCOORD:
   case PIPE_CAP_VERTEX_COLOR_UNCLAMPED:
   case PIPE_CAP_MIXED_COLOR_DEPTH_BITS:
      return 1;
   case PIPE_CAP_NATIVE_FENCE_FD:
      return screen->drm_version >= ETNA_DRM_VERSION_FENCE_FD;

   /* Memory */
   case PIPE_CAP_CONSTANT_BUFFER_OFFSET_ALIGNMENT:
      return 256;
   case PIPE_CAP_MIN_MAP_BUFFER_ALIGNMENT:
      return 4; /* XXX could easily be supported */
   case PIPE_CAP_GLSL_FEATURE_LEVEL:
   case PIPE_CAP_GLSL_FEATURE_LEVEL_COMPATIBILITY:
      return 120;

   case PIPE_CAP_NPOT_TEXTURES:
      return true; /* VIV_FEATURE(priv->dev, chipMinorFeatures1,
                      NON_POWER_OF_TWO); */

   case PIPE_CAP_TEXTURE_SWIZZLE:
   case PIPE_CAP_PRIMITIVE_RESTART:
      return VIV_FEATURE(screen, chipMinorFeatures1, HALTI0);

   case PIPE_CAP_ENDIANNESS:
      return PIPE_ENDIAN_LITTLE; /* on most Viv hw this is configurable (feature
                                    ENDIANNESS_CONFIG) */

   /* Unsupported features. */
   case PIPE_CAP_SEAMLESS_CUBE_MAP:
   case PIPE_CAP_COMPUTE: /* XXX supported on gc2000 */
   case PIPE_CAP_MIXED_COLORBUFFER_FORMATS: /* only one colorbuffer supported, so mixing makes no sense */
   case PIPE_CAP_CONDITIONAL_RENDER: /* no occlusion queries */
   case PIPE_CAP_TGSI_INSTANCEID: /* no idea, really */
   case PIPE_CAP_START_INSTANCE: /* instancing not supported AFAIK */
   case PIPE_CAP_VERTEX_ELEMENT_INSTANCE_DIVISOR: /* instancing not supported AFAIK */
   case PIPE_CAP_SHADER_STENCIL_EXPORT: /* Fragment shader cannot export stencil value */
   case PIPE_CAP_MAX_DUAL_SOURCE_RENDER_TARGETS: /* no dual-source supported */
   case PIPE_CAP_TEXTURE_MULTISAMPLE: /* no texture multisample */
   case PIPE_CAP_TEXTURE_MIRROR_CLAMP: /* only mirrored repeat */
   case PIPE_CAP_TEXTURE_MIRROR_CLAMP_TO_EDGE: /* only mirrored repeat */
   case PIPE_CAP_INDEP_BLEND_ENABLE:
   case PIPE_CAP_INDEP_BLEND_FUNC:
   case PIPE_CAP_DEPTH_CLIP_DISABLE:
   case PIPE_CAP_DEPTH_CLIP_DISABLE_SEPARATE:
   case PIPE_CAP_SEAMLESS_CUBE_MAP_PER_TEXTURE:
   case PIPE_CAP_TGSI_FS_COORD_ORIGIN_LOWER_LEFT:
   case PIPE_CAP_TGSI_FS_COORD_PIXEL_CENTER_INTEGER:
   case PIPE_CAP_TGSI_CAN_COMPACT_CONSTANTS: /* Don't skip strict max uniform limit check */
   case PIPE_CAP_FRAGMENT_COLOR_CLAMPED:
   case PIPE_CAP_VERTEX_COLOR_CLAMPED:
   case PIPE_CAP_USER_VERTEX_BUFFERS:
   case PIPE_CAP_TEXTURE_BUFFER_OBJECTS:
   case PIPE_CAP_TEXTURE_BUFFER_OFFSET_ALIGNMENT:
   case PIPE_CAP_BUFFER_SAMPLER_VIEW_RGBA_ONLY:
   case PIPE_CAP_MIXED_FRAMEBUFFER_SIZES: /* TODO: test me out with piglit */
   case PIPE_CAP_TGSI_VS_LAYER_VIEWPORT:
   case PIPE_CAP_MAX_TEXTURE_GATHER_COMPONENTS:
   case PIPE_CAP_TEXTURE_GATHER_SM5:
   case PIPE_CAP_BUFFER_MAP_PERSISTENT_COHERENT:
   case PIPE_CAP_FAKE_SW_MSAA:
   case PIPE_CAP_TEXTURE_QUERY_LOD:
   case PIPE_CAP_SAMPLE_SHADING:
   case PIPE_CAP_TEXTURE_GATHER_OFFSETS:
   case PIPE_CAP_TGSI_VS_WINDOW_SPACE_POSITION:
   case PIPE_CAP_DRAW_INDIRECT:
   case PIPE_CAP_TGSI_FS_FINE_DERIVATIVE:
   case PIPE_CAP_CONDITIONAL_RENDER_INVERTED:
   case PIPE_CAP_SAMPLER_VIEW_TARGET:
   case PIPE_CAP_CLIP_HALFZ:
   case PIPE_CAP_VERTEXID_NOBASE:
   case PIPE_CAP_POLYGON_OFFSET_CLAMP:
   case PIPE_CAP_MULTISAMPLE_Z_RESOLVE:
   case PIPE_CAP_RESOURCE_FROM_USER_MEMORY:
   case PIPE_CAP_DEVICE_RESET_STATUS_QUERY:
   case PIPE_CAP_MAX_SHADER_PATCH_VARYINGS:
   case PIPE_CAP_TEXTURE_FLOAT_LINEAR:
   case PIPE_CAP_TEXTURE_HALF_FLOAT_LINEAR:
   case PIPE_CAP_DEPTH_BOUNDS_TEST:
   case PIPE_CAP_TGSI_TXQS:
   case PIPE_CAP_FORCE_PERSAMPLE_INTERP:
   case PIPE_CAP_SHAREABLE_SHADERS:
   case PIPE_CAP_COPY_BETWEEN_COMPRESSED_AND_PLAIN_FORMATS:
   case PIPE_CAP_CLEAR_TEXTURE:
   case PIPE_CAP_DRAW_PARAMETERS:
   case PIPE_CAP_TGSI_PACK_HALF_FLOAT:
   case PIPE_CAP_MULTI_DRAW_INDIRECT:
   case PIPE_CAP_MULTI_DRAW_INDIRECT_PARAMS:
   case PIPE_CAP_TGSI_FS_POSITION_IS_SYSVAL:
   case PIPE_CAP_TGSI_FS_FACE_IS_INTEGER_SYSVAL:
   case PIPE_CAP_SHADER_BUFFER_OFFSET_ALIGNMENT:
   case PIPE_CAP_INVALIDATE_BUFFER:
   case PIPE_CAP_GENERATE_MIPMAP:
   case PIPE_CAP_STRING_MARKER:
   case PIPE_CAP_SURFACE_REINTERPRET_BLOCKS:
   case PIPE_CAP_QUERY_BUFFER_OBJECT:
   case PIPE_CAP_QUERY_MEMORY_INFO:
   case PIPE_CAP_FRAMEBUFFER_NO_ATTACHMENT:
   case PIPE_CAP_ROBUST_BUFFER_ACCESS_BEHAVIOR:
   case PIPE_CAP_CULL_DISTANCE:
   case PIPE_CAP_PRIMITIVE_RESTART_FOR_PATCHES:
   case PIPE_CAP_TGSI_VOTE:
   case PIPE_CAP_MAX_WINDOW_RECTANGLES:
   case PIPE_CAP_POLYGON_OFFSET_UNITS_UNSCALED:
   case PIPE_CAP_VIEWPORT_SUBPIXEL_BITS:
   case PIPE_CAP_TGSI_ARRAY_COMPONENTS:
   case PIPE_CAP_STREAM_OUTPUT_INTERLEAVE_BUFFERS:
   case PIPE_CAP_TGSI_CAN_READ_OUTPUTS:
   case PIPE_CAP_GLSL_OPTIMIZE_CONSERVATIVELY:
   case PIPE_CAP_TGSI_FS_FBFETCH:
   case PIPE_CAP_TGSI_MUL_ZERO_WINS:
   case PIPE_CAP_DOUBLES:
   case PIPE_CAP_INT64:
   case PIPE_CAP_INT64_DIVMOD:
   case PIPE_CAP_TGSI_TEX_TXF_LZ:
   case PIPE_CAP_TGSI_CLOCK:
   case PIPE_CAP_POLYGON_MODE_FILL_RECTANGLE:
   case PIPE_CAP_SPARSE_BUFFER_PAGE_SIZE:
   case PIPE_CAP_TGSI_BALLOT:
   case PIPE_CAP_TGSI_TES_LAYER_VIEWPORT:
   case PIPE_CAP_CAN_BIND_CONST_BUFFER_AS_VERTEX:
   case PIPE_CAP_ALLOW_MAPPED_BUFFERS_DURING_EXECUTION:
   case PIPE_CAP_POST_DEPTH_COVERAGE:
   case PIPE_CAP_BINDLESS_TEXTURE:
   case PIPE_CAP_NIR_SAMPLERS_AS_DEREF:
   case PIPE_CAP_QUERY_SO_OVERFLOW:
   case PIPE_CAP_MEMOBJ:
   case PIPE_CAP_LOAD_CONSTBUF:
   case PIPE_CAP_TGSI_ANY_REG_AS_ADDRESS:
   case PIPE_CAP_TILE_RASTER_ORDER:
   case PIPE_CAP_MAX_COMBINED_SHADER_OUTPUT_RESOURCES:
   case PIPE_CAP_FRAMEBUFFER_MSAA_CONSTRAINTS:
   case PIPE_CAP_SIGNED_VERTEX_BUFFER_OFFSET:
   case PIPE_CAP_CONTEXT_PRIORITY_MASK:
   case PIPE_CAP_FENCE_SIGNAL:
   case PIPE_CAP_CONSTBUF0_FLAGS:
   case PIPE_CAP_CONSERVATIVE_RASTER_POST_SNAP_TRIANGLES:
   case PIPE_CAP_CONSERVATIVE_RASTER_POST_SNAP_POINTS_LINES:
   case PIPE_CAP_CONSERVATIVE_RASTER_PRE_SNAP_TRIANGLES:
   case PIPE_CAP_CONSERVATIVE_RASTER_PRE_SNAP_POINTS_LINES:
   case PIPE_CAP_CONSERVATIVE_RASTER_POST_DEPTH_COVERAGE:
   case PIPE_CAP_MAX_CONSERVATIVE_RASTER_SUBPIXEL_PRECISION_BIAS:
   case PIPE_CAP_PACKED_UNIFORMS:
   case PIPE_CAP_PROGRAMMABLE_SAMPLE_LOCATIONS:
   case PIPE_CAP_MAX_TEXTURE_UPLOAD_MEMORY_BUDGET:
      return 0;

   case PIPE_CAP_MAX_GS_INVOCATIONS:
      return 32;

   case PIPE_CAP_MAX_SHADER_BUFFER_SIZE:
      return 1 << 27;

   /* Stream output. */
   case PIPE_CAP_MAX_STREAM_OUTPUT_BUFFERS:
   case PIPE_CAP_STREAM_OUTPUT_PAUSE_RESUME:
   case PIPE_CAP_MAX_STREAM_OUTPUT_SEPARATE_COMPONENTS:
   case PIPE_CAP_MAX_STREAM_OUTPUT_INTERLEAVED_COMPONENTS:
      return 0;

   /* Geometry shader output, unsupported. */
   case PIPE_CAP_MAX_GEOMETRY_OUTPUT_VERTICES:
   case PIPE_CAP_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS:
   case PIPE_CAP_MAX_VERTEX_STREAMS:
      return 0;

   case PIPE_CAP_MAX_VERTEX_ATTRIB_STRIDE:
      return 128;
   case PIPE_CAP_MAX_VERTEX_ELEMENT_SRC_OFFSET:
      return 255;

   /* Texturing. */
   case PIPE_CAP_MAX_TEXTURE_2D_LEVELS:
   case PIPE_CAP_MAX_TEXTURE_CUBE_LEVELS:
   {
      int log2_max_tex_size = util_last_bit(screen->specs.max_texture_size);
      assert(log2_max_tex_size > 0);
      return log2_max_tex_size;
   }
   case PIPE_CAP_MAX_TEXTURE_3D_LEVELS: /* 3D textures not supported - fake it */
      return 5;
   case PIPE_CAP_MAX_TEXTURE_ARRAY_LAYERS:
      return 0;
   case PIPE_CAP_CUBE_MAP_ARRAY:
      return 0;
   case PIPE_CAP_MIN_TEXTURE_GATHER_OFFSET:
   case PIPE_CAP_MIN_TEXEL_OFFSET:
      return -8;
   case PIPE_CAP_MAX_TEXTURE_GATHER_OFFSET:
   case PIPE_CAP_MAX_TEXEL_OFFSET:
      return 7;
   case PIPE_CAP_TEXTURE_BORDER_COLOR_QUIRK:
      return 0;
   case PIPE_CAP_MAX_TEXTURE_BUFFER_SIZE:
      return 65536;

   /* Render targets. */
   case PIPE_CAP_MAX_RENDER_TARGETS:
      return 1;

   /* Viewports and scissors. */
   case PIPE_CAP_MAX_VIEWPORTS:
      return 1;

   /* Timer queries. */
   case PIPE_CAP_QUERY_TIME_ELAPSED:
      return 0;
   case PIPE_CAP_OCCLUSION_QUERY:
      return VIV_FEATURE(screen, chipMinorFeatures1, HALTI0);
   case PIPE_CAP_QUERY_TIMESTAMP:
      return 1;
   case PIPE_CAP_QUERY_PIPELINE_STATISTICS:
      return 0;

   /* Preferences */
   case PIPE_CAP_PREFER_BLIT_BASED_TEXTURE_TRANSFER:
      return 0;

   case PIPE_CAP_PCI_GROUP:
   case PIPE_CAP_PCI_BUS:
   case PIPE_CAP_PCI_DEVICE:
   case PIPE_CAP_PCI_FUNCTION:
      return 0;
   case PIPE_CAP_VENDOR_ID:
   case PIPE_CAP_DEVICE_ID:
      return 0xFFFFFFFF;
   case PIPE_CAP_ACCELERATED:
      return 1;
   case PIPE_CAP_VIDEO_MEMORY:
      return 0;
   case PIPE_CAP_UMA:
      return 1;
   default:
      return u_pipe_screen_get_param_defaults(pscreen, param);
   }
}

static float
etna_screen_get_paramf(struct pipe_screen *pscreen, enum pipe_capf param)
{
   struct etna_screen *screen = etna_screen(pscreen);

   switch (param) {
   case PIPE_CAPF_MAX_LINE_WIDTH:
   case PIPE_CAPF_MAX_LINE_WIDTH_AA:
   case PIPE_CAPF_MAX_POINT_WIDTH:
   case PIPE_CAPF_MAX_POINT_WIDTH_AA:
      return 8192.0f;
   case PIPE_CAPF_MAX_TEXTURE_ANISOTROPY:
      return 16.0f;
   case PIPE_CAPF_MAX_TEXTURE_LOD_BIAS:
      return util_last_bit(screen->specs.max_texture_size);
   case PIPE_CAPF_MIN_CONSERVATIVE_RASTER_DILATE:
   case PIPE_CAPF_MAX_CONSERVATIVE_RASTER_DILATE:
   case PIPE_CAPF_CONSERVATIVE_RASTER_DILATE_GRANULARITY:
      return 0.0f;
   }

   debug_printf("unknown paramf %d", param);
   return 0;
}

static int
etna_screen_get_shader_param(struct pipe_screen *pscreen,
                             enum pipe_shader_type shader,
                             enum pipe_shader_cap param)
{
   struct etna_screen *screen = etna_screen(pscreen);

   switch (shader) {
   case PIPE_SHADER_FRAGMENT:
   case PIPE_SHADER_VERTEX:
      break;
   case PIPE_SHADER_COMPUTE:
   case PIPE_SHADER_GEOMETRY:
   case PIPE_SHADER_TESS_CTRL:
   case PIPE_SHADER_TESS_EVAL:
      return 0;
   default:
      DBG("unknown shader type %d", shader);
      return 0;
   }

   switch (param) {
   case PIPE_SHADER_CAP_MAX_INSTRUCTIONS:
   case PIPE_SHADER_CAP_MAX_ALU_INSTRUCTIONS:
   case PIPE_SHADER_CAP_MAX_TEX_INSTRUCTIONS:
   case PIPE_SHADER_CAP_MAX_TEX_INDIRECTIONS:
      return ETNA_MAX_TOKENS;
   case PIPE_SHADER_CAP_MAX_CONTROL_FLOW_DEPTH:
      return ETNA_MAX_DEPTH; /* XXX */
   case PIPE_SHADER_CAP_MAX_INPUTS:
      /* Maximum number of inputs for the vertex shader is the number
       * of vertex elements - each element defines one vertex shader
       * input register.  For the fragment shader, this is the number
       * of varyings. */
      return shader == PIPE_SHADER_FRAGMENT ? screen->specs.max_varyings
                                            : screen->specs.vertex_max_elements;
   case PIPE_SHADER_CAP_MAX_OUTPUTS:
      return 16; /* see VIVS_VS_OUTPUT */
   case PIPE_SHADER_CAP_MAX_TEMPS:
      return 64; /* Max native temporaries. */
   case PIPE_SHADER_CAP_MAX_CONST_BUFFERS:
      return 1;
   case PIPE_SHADER_CAP_TGSI_CONT_SUPPORTED:
      return 1;
   case PIPE_SHADER_CAP_INDIRECT_INPUT_ADDR:
   case PIPE_SHADER_CAP_INDIRECT_OUTPUT_ADDR:
   case PIPE_SHADER_CAP_INDIRECT_TEMP_ADDR:
   case PIPE_SHADER_CAP_INDIRECT_CONST_ADDR:
      return 1;
   case PIPE_SHADER_CAP_SUBROUTINES:
      return 0;
   case PIPE_SHADER_CAP_TGSI_SQRT_SUPPORTED:
      return VIV_FEATURE(screen, chipMinorFeatures0, HAS_SQRT_TRIG);
   case PIPE_SHADER_CAP_INTEGERS:
   case PIPE_SHADER_CAP_INT64_ATOMICS:
   case PIPE_SHADER_CAP_FP16:
      return 0;
   case PIPE_SHADER_CAP_MAX_TEXTURE_SAMPLERS:
   case PIPE_SHADER_CAP_MAX_SAMPLER_VIEWS:
      return shader == PIPE_SHADER_FRAGMENT
                ? screen->specs.fragment_sampler_count
                : screen->specs.vertex_sampler_count;
   case PIPE_SHADER_CAP_PREFERRED_IR:
      return PIPE_SHADER_IR_TGSI;
   case PIPE_SHADER_CAP_MAX_CONST_BUFFER_SIZE:
      return 4096;
   case PIPE_SHADER_CAP_TGSI_DROUND_SUPPORTED:
   case PIPE_SHADER_CAP_TGSI_DFRACEXP_DLDEXP_SUPPORTED:
   case PIPE_SHADER_CAP_TGSI_LDEXP_SUPPORTED:
   case PIPE_SHADER_CAP_TGSI_FMA_SUPPORTED:
   case PIPE_SHADER_CAP_TGSI_ANY_INOUT_DECL_RANGE:
      return false;
   case PIPE_SHADER_CAP_SUPPORTED_IRS:
      return 0;
   case PIPE_SHADER_CAP_MAX_UNROLL_ITERATIONS_HINT:
      return 32;
   case PIPE_SHADER_CAP_MAX_SHADER_BUFFERS:
   case PIPE_SHADER_CAP_MAX_SHADER_IMAGES:
   case PIPE_SHADER_CAP_LOWER_IF_THRESHOLD:
   case PIPE_SHADER_CAP_TGSI_SKIP_MERGE_REGISTERS:
   case PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTERS:
   case PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTER_BUFFERS:
   case PIPE_SHADER_CAP_SCALAR_ISA:
      return 0;
   }

   debug_printf("unknown shader param %d", param);
   return 0;
}

static uint64_t
etna_screen_get_timestamp(struct pipe_screen *pscreen)
{
   return os_time_get_nano();
}

static bool
gpu_supports_texure_format(struct etna_screen *screen, uint32_t fmt,
                           enum pipe_format format)
{
   bool supported = true;

   if (fmt == TEXTURE_FORMAT_ETC1)
      supported = VIV_FEATURE(screen, chipFeatures, ETC1_TEXTURE_COMPRESSION);

   if (fmt >= TEXTURE_FORMAT_DXT1 && fmt <= TEXTURE_FORMAT_DXT4_DXT5)
      supported = VIV_FEATURE(screen, chipFeatures, DXT_TEXTURE_COMPRESSION);

   if (util_format_is_srgb(format))
      supported = VIV_FEATURE(screen, chipMinorFeatures1, HALTI0);

   if (fmt & EXT_FORMAT) {
      supported = VIV_FEATURE(screen, chipMinorFeatures1, HALTI0);

      /* ETC1 is checked above, as it has its own feature bit. ETC2 is
       * supported with HALTI0, however that implementation is buggy in hardware.
       * The blob driver does per-block patching to work around this. As this
       * is currently not implemented by etnaviv, enable it for HALTI1 (GC3000)
       * only.
       */
      if (util_format_is_etc(format))
         supported = VIV_FEATURE(screen, chipMinorFeatures2, HALTI1);
   }

   if (fmt & ASTC_FORMAT) {
      supported = screen->specs.tex_astc;
   }

   if (!supported)
      return false;

   if (texture_format_needs_swiz(format))
      return VIV_FEATURE(screen, chipMinorFeatures1, HALTI0);

   return true;
}

static boolean
etna_screen_is_format_supported(struct pipe_screen *pscreen,
                                enum pipe_format format,
                                enum pipe_texture_target target,
                                unsigned sample_count,
                                unsigned storage_sample_count,
                                unsigned usage)
{
   struct etna_screen *screen = etna_screen(pscreen);
   unsigned allowed = 0;

   if (target != PIPE_BUFFER &&
       target != PIPE_TEXTURE_1D &&
       target != PIPE_TEXTURE_2D &&
       target != PIPE_TEXTURE_3D &&
       target != PIPE_TEXTURE_CUBE &&
       target != PIPE_TEXTURE_RECT)
      return FALSE;

   if (MAX2(1, sample_count) != MAX2(1, storage_sample_count))
      return false;

   if (usage & PIPE_BIND_RENDER_TARGET) {
      /* if render target, must be RS-supported format */
      if (translate_rs_format(format) != ETNA_NO_MATCH) {
         /* Validate MSAA; number of samples must be allowed, and render target
          * must have MSAA'able format. */
         if (sample_count > 1) {
            if (translate_samples_to_xyscale(sample_count, NULL, NULL, NULL) &&
                translate_msaa_format(format) != ETNA_NO_MATCH) {
               allowed |= PIPE_BIND_RENDER_TARGET;
            }
         } else {
            allowed |= PIPE_BIND_RENDER_TARGET;
         }
      }
   }

   if (usage & PIPE_BIND_DEPTH_STENCIL) {
      if (translate_depth_format(format) != ETNA_NO_MATCH)
         allowed |= PIPE_BIND_DEPTH_STENCIL;
   }

   if (usage & PIPE_BIND_SAMPLER_VIEW) {
      uint32_t fmt = translate_texture_format(format);

      if (!gpu_supports_texure_format(screen, fmt, format))
         fmt = ETNA_NO_MATCH;

      if (sample_count < 2 && fmt != ETNA_NO_MATCH)
         allowed |= PIPE_BIND_SAMPLER_VIEW;
   }

   if (usage & PIPE_BIND_VERTEX_BUFFER) {
      if (translate_vertex_format_type(format) != ETNA_NO_MATCH)
         allowed |= PIPE_BIND_VERTEX_BUFFER;
   }

   if (usage & PIPE_BIND_INDEX_BUFFER) {
      /* must be supported index format */
      if (format == PIPE_FORMAT_I8_UINT || format == PIPE_FORMAT_I16_UINT ||
          (format == PIPE_FORMAT_I32_UINT &&
           VIV_FEATURE(screen, chipFeatures, 32_BIT_INDICES))) {
         allowed |= PIPE_BIND_INDEX_BUFFER;
      }
   }

   /* Always allowed */
   allowed |=
      usage & (PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_SCANOUT | PIPE_BIND_SHARED);

   if (usage != allowed) {
      DBG("not supported: format=%s, target=%d, sample_count=%d, "
          "usage=%x, allowed=%x",
          util_format_name(format), target, sample_count, usage, allowed);
   }

   return usage == allowed;
}

const uint64_t supported_modifiers[] = {
   DRM_FORMAT_MOD_LINEAR,
   DRM_FORMAT_MOD_VIVANTE_TILED,
   DRM_FORMAT_MOD_VIVANTE_SUPER_TILED,
   DRM_FORMAT_MOD_VIVANTE_SPLIT_TILED,
   DRM_FORMAT_MOD_VIVANTE_SPLIT_SUPER_TILED,
};

static void
etna_screen_query_dmabuf_modifiers(struct pipe_screen *pscreen,
                                   enum pipe_format format, int max,
                                   uint64_t *modifiers,
                                   unsigned int *external_only, int *count)
{
   struct etna_screen *screen = etna_screen(pscreen);
   int i, num_modifiers = 0;

   if (max > ARRAY_SIZE(supported_modifiers))
      max = ARRAY_SIZE(supported_modifiers);

   if (!max) {
      modifiers = NULL;
      max = ARRAY_SIZE(supported_modifiers);
   }

   for (i = 0; num_modifiers < max; i++) {
      /* don't advertise split tiled formats on single pipe/buffer GPUs */
      if ((screen->specs.pixel_pipes == 1 || screen->specs.single_buffer) &&
          i >= 3)
         break;

      if (modifiers)
         modifiers[num_modifiers] = supported_modifiers[i];
      if (external_only)
         external_only[num_modifiers] = util_format_is_yuv(format) ? 1 : 0;
      num_modifiers++;
   }

   *count = num_modifiers;
}

static boolean
etna_get_specs(struct etna_screen *screen)
{
   uint64_t val;
   uint32_t instruction_count;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_INSTRUCTION_COUNT, &val)) {
      DBG("could not get ETNA_GPU_INSTRUCTION_COUNT");
      goto fail;
   }
   instruction_count = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_VERTEX_OUTPUT_BUFFER_SIZE,
                          &val)) {
      DBG("could not get ETNA_GPU_VERTEX_OUTPUT_BUFFER_SIZE");
      goto fail;
   }
   screen->specs.vertex_output_buffer_size = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_VERTEX_CACHE_SIZE, &val)) {
      DBG("could not get ETNA_GPU_VERTEX_CACHE_SIZE");
      goto fail;
   }
   screen->specs.vertex_cache_size = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_SHADER_CORE_COUNT, &val)) {
      DBG("could not get ETNA_GPU_SHADER_CORE_COUNT");
      goto fail;
   }
   screen->specs.shader_core_count = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_STREAM_COUNT, &val)) {
      DBG("could not get ETNA_GPU_STREAM_COUNT");
      goto fail;
   }
   screen->specs.stream_count = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_REGISTER_MAX, &val)) {
      DBG("could not get ETNA_GPU_REGISTER_MAX");
      goto fail;
   }
   screen->specs.max_registers = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_PIXEL_PIPES, &val)) {
      DBG("could not get ETNA_GPU_PIXEL_PIPES");
      goto fail;
   }
   screen->specs.pixel_pipes = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_NUM_CONSTANTS, &val)) {
      DBG("could not get %s", "ETNA_GPU_NUM_CONSTANTS");
      goto fail;
   }
   if (val == 0) {
      fprintf(stderr, "Warning: zero num constants (update kernel?)\n");
      val = 168;
   }
   screen->specs.num_constants = val;

   /* Figure out gross GPU architecture. See rnndb/common.xml for a specific
    * description of the differences. */
   if (VIV_FEATURE(screen, chipMinorFeatures5, HALTI5))
      screen->specs.halti = 5; /* New GC7000/GC8x00  */
   else if (VIV_FEATURE(screen, chipMinorFeatures5, HALTI4))
      screen->specs.halti = 4; /* Old GC7000/GC7400 */
   else if (VIV_FEATURE(screen, chipMinorFeatures5, HALTI3))
      screen->specs.halti = 3; /* None? */
   else if (VIV_FEATURE(screen, chipMinorFeatures4, HALTI2))
      screen->specs.halti = 2; /* GC2500/GC3000/GC5000/GC6400 */
   else if (VIV_FEATURE(screen, chipMinorFeatures2, HALTI1))
      screen->specs.halti = 1; /* GC900/GC4000/GC7000UL */
   else if (VIV_FEATURE(screen, chipMinorFeatures1, HALTI0))
      screen->specs.halti = 0; /* GC880/GC2000/GC7000TM */
   else
      screen->specs.halti = -1; /* GC7000nanolite / pre-GC2000 except GC880 */
   if (screen->specs.halti >= 0)
      DBG("etnaviv: GPU arch: HALTI%d", screen->specs.halti);
   else
      DBG("etnaviv: GPU arch: pre-HALTI");

   screen->specs.can_supertile =
      VIV_FEATURE(screen, chipMinorFeatures0, SUPER_TILED);
   screen->specs.bits_per_tile =
      VIV_FEATURE(screen, chipMinorFeatures0, 2BITPERTILE) ? 2 : 4;
   screen->specs.ts_clear_value =
      VIV_FEATURE(screen, chipMinorFeatures0, 2BITPERTILE) ? 0x55555555
                                                           : 0x11111111;

   /* vertex and fragment samplers live in one address space */
   screen->specs.vertex_sampler_offset = 8;
   screen->specs.fragment_sampler_count = 8;
   screen->specs.vertex_sampler_count = 4;
   screen->specs.vs_need_z_div =
      screen->model < 0x1000 && screen->model != 0x880;
   screen->specs.has_sin_cos_sqrt =
      VIV_FEATURE(screen, chipMinorFeatures0, HAS_SQRT_TRIG);
   screen->specs.has_sign_floor_ceil =
      VIV_FEATURE(screen, chipMinorFeatures0, HAS_SIGN_FLOOR_CEIL);
   screen->specs.has_shader_range_registers =
      screen->model >= 0x1000 || screen->model == 0x880;
   screen->specs.npot_tex_any_wrap =
      VIV_FEATURE(screen, chipMinorFeatures1, NON_POWER_OF_TWO);
   screen->specs.has_new_transcendentals =
      VIV_FEATURE(screen, chipMinorFeatures3, HAS_FAST_TRANSCENDENTALS);
   screen->specs.has_halti2_instructions =
      VIV_FEATURE(screen, chipMinorFeatures4, HALTI2);

   if (screen->specs.halti >= 5) {
      /* GC7000 - this core must load shaders from memory. */
      screen->specs.vs_offset = 0;
      screen->specs.ps_offset = 0;
      screen->specs.max_instructions = 0; /* Do not program shaders manually */
      screen->specs.has_icache = true;
   } else if (VIV_FEATURE(screen, chipMinorFeatures3, INSTRUCTION_CACHE)) {
      /* GC3000 - this core is capable of loading shaders from
       * memory. It can also run shaders from registers, as a fallback, but
       * "max_instructions" does not have the correct value. It has place for
       * 2*256 instructions just like GC2000, but the offsets are slightly
       * different.
       */
      screen->specs.vs_offset = 0xC000;
      /* State 08000-0C000 mirrors 0C000-0E000, and the Vivante driver uses
       * this mirror for writing PS instructions, probably safest to do the
       * same.
       */
      screen->specs.ps_offset = 0x8000 + 0x1000;
      screen->specs.max_instructions = 256; /* maximum number instructions for non-icache use */
      screen->specs.has_icache = true;
   } else {
      if (instruction_count > 256) { /* unified instruction memory? */
         screen->specs.vs_offset = 0xC000;
         screen->specs.ps_offset = 0xD000; /* like vivante driver */
         screen->specs.max_instructions = 256;
      } else {
         screen->specs.vs_offset = 0x4000;
         screen->specs.ps_offset = 0x6000;
         screen->specs.max_instructions = instruction_count / 2;
      }
      screen->specs.has_icache = false;
   }

   if (VIV_FEATURE(screen, chipMinorFeatures1, HALTI0)) {
      screen->specs.max_varyings = 12;
      screen->specs.vertex_max_elements = 16;
   } else {
      screen->specs.max_varyings = 8;
      /* Etna_viv documentation seems confused over the correct value
       * here so choose the lower to be safe: HALTI0 says 16 i.s.o.
       * 10, but VERTEX_ELEMENT_CONFIG register says 16 i.s.o. 12. */
      screen->specs.vertex_max_elements = 10;
   }

   /* Etna_viv documentation does not indicate where varyings above 8 are
    * stored. Moreover, if we are passed more than 8 varyings, we will
    * walk off the end of some arrays. Limit the maximum number of varyings. */
   if (screen->specs.max_varyings > ETNA_NUM_VARYINGS)
      screen->specs.max_varyings = ETNA_NUM_VARYINGS;

   /* from QueryShaderCaps in kernel driver */
   if (screen->model < chipModel_GC4000) {
      screen->specs.max_vs_uniforms = 168;
      screen->specs.max_ps_uniforms = 64;
   } else {
      screen->specs.max_vs_uniforms = 256;
      screen->specs.max_ps_uniforms = 256;
   }

   if (screen->specs.halti >= 5) {
      screen->specs.has_unified_uniforms = true;
      screen->specs.vs_uniforms_offset = VIVS_SH_HALTI5_UNIFORMS_MIRROR(0);
      screen->specs.ps_uniforms_offset = VIVS_SH_HALTI5_UNIFORMS(screen->specs.max_vs_uniforms*4);
   } else if (screen->specs.halti >= 1) {
      /* unified uniform memory on GC3000 - HALTI1 feature bit is just a guess
      */
      screen->specs.has_unified_uniforms = true;
      screen->specs.vs_uniforms_offset = VIVS_SH_UNIFORMS(0);
      /* hardcode PS uniforms to start after end of VS uniforms -
       * for more flexibility this offset could be variable based on the
       * shader.
       */
      screen->specs.ps_uniforms_offset = VIVS_SH_UNIFORMS(screen->specs.max_vs_uniforms*4);
   } else {
      screen->specs.has_unified_uniforms = false;
      screen->specs.vs_uniforms_offset = VIVS_VS_UNIFORMS(0);
      screen->specs.ps_uniforms_offset = VIVS_PS_UNIFORMS(0);
   }

   screen->specs.max_texture_size =
      VIV_FEATURE(screen, chipMinorFeatures0, TEXTURE_8K) ? 8192 : 2048;
   screen->specs.max_rendertarget_size =
      VIV_FEATURE(screen, chipMinorFeatures0, RENDERTARGET_8K) ? 8192 : 2048;

   screen->specs.single_buffer = VIV_FEATURE(screen, chipMinorFeatures4, SINGLE_BUFFER);
   if (screen->specs.single_buffer)
      DBG("etnaviv: Single buffer mode enabled with %d pixel pipes", screen->specs.pixel_pipes);

   screen->specs.tex_astc = VIV_FEATURE(screen, chipMinorFeatures4, TEXTURE_ASTC);

   screen->specs.use_blt = VIV_FEATURE(screen, chipMinorFeatures5, BLT_ENGINE);

   return true;

fail:
   return false;
}

struct etna_bo *
etna_screen_bo_from_handle(struct pipe_screen *pscreen,
                           struct winsys_handle *whandle, unsigned *out_stride)
{
   struct etna_screen *screen = etna_screen(pscreen);
   struct etna_bo *bo;

   if (whandle->type == WINSYS_HANDLE_TYPE_SHARED) {
      bo = etna_bo_from_name(screen->dev, whandle->handle);
   } else if (whandle->type == WINSYS_HANDLE_TYPE_FD) {
      bo = etna_bo_from_dmabuf(screen->dev, whandle->handle);
   } else {
      DBG("Attempt to import unsupported handle type %d", whandle->type);
      return NULL;
   }

   if (!bo) {
      DBG("ref name 0x%08x failed", whandle->handle);
      return NULL;
   }

   *out_stride = whandle->stride;

   return bo;
}

struct pipe_screen *
etna_screen_create(struct etna_device *dev, struct etna_gpu *gpu,
                   struct renderonly *ro)
{
   struct etna_screen *screen = CALLOC_STRUCT(etna_screen);
   struct pipe_screen *pscreen;
   drmVersionPtr version;
   uint64_t val;

   if (!screen)
      return NULL;

   pscreen = &screen->base;
   screen->dev = dev;
   screen->gpu = gpu;
   screen->ro = renderonly_dup(ro);
   screen->refcnt = 1;

   if (!screen->ro) {
      DBG("could not create renderonly object");
      goto fail;
   }

   version = drmGetVersion(screen->ro->gpu_fd);
   screen->drm_version = ETNA_DRM_VERSION(version->version_major,
                                          version->version_minor);
   drmFreeVersion(version);

   etna_mesa_debug = debug_get_option_etna_mesa_debug();

   /* Disable autodisable for correct rendering with TS */
   etna_mesa_debug |= ETNA_DBG_NO_AUTODISABLE;

   screen->pipe = etna_pipe_new(gpu, ETNA_PIPE_3D);
   if (!screen->pipe) {
      DBG("could not create 3d pipe");
      goto fail;
   }

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_MODEL, &val)) {
      DBG("could not get ETNA_GPU_MODEL");
      goto fail;
   }
   screen->model = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_REVISION, &val)) {
      DBG("could not get ETNA_GPU_REVISION");
      goto fail;
   }
   screen->revision = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_FEATURES_0, &val)) {
      DBG("could not get ETNA_GPU_FEATURES_0");
      goto fail;
   }
   screen->features[0] = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_FEATURES_1, &val)) {
      DBG("could not get ETNA_GPU_FEATURES_1");
      goto fail;
   }
   screen->features[1] = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_FEATURES_2, &val)) {
      DBG("could not get ETNA_GPU_FEATURES_2");
      goto fail;
   }
   screen->features[2] = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_FEATURES_3, &val)) {
      DBG("could not get ETNA_GPU_FEATURES_3");
      goto fail;
   }
   screen->features[3] = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_FEATURES_4, &val)) {
      DBG("could not get ETNA_GPU_FEATURES_4");
      goto fail;
   }
   screen->features[4] = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_FEATURES_5, &val)) {
      DBG("could not get ETNA_GPU_FEATURES_5");
      goto fail;
   }
   screen->features[5] = val;

   if (etna_gpu_get_param(screen->gpu, ETNA_GPU_FEATURES_6, &val)) {
      DBG("could not get ETNA_GPU_FEATURES_6");
      goto fail;
   }
   screen->features[6] = val;

   if (!etna_get_specs(screen))
      goto fail;

   /* apply debug options that disable individual features */
   if (DBG_ENABLED(ETNA_DBG_NO_EARLY_Z))
      screen->features[viv_chipFeatures] |= chipFeatures_NO_EARLY_Z;
   if (DBG_ENABLED(ETNA_DBG_NO_TS))
         screen->features[viv_chipFeatures] &= ~chipFeatures_FAST_CLEAR;
   if (DBG_ENABLED(ETNA_DBG_NO_AUTODISABLE))
      screen->features[viv_chipMinorFeatures1] &= ~chipMinorFeatures1_AUTO_DISABLE;
   if (DBG_ENABLED(ETNA_DBG_NO_SUPERTILE))
      screen->specs.can_supertile = 0;
   if (DBG_ENABLED(ETNA_DBG_NO_SINGLEBUF))
      screen->specs.single_buffer = 0;

   pscreen->destroy = etna_screen_destroy;
   pscreen->get_param = etna_screen_get_param;
   pscreen->get_paramf = etna_screen_get_paramf;
   pscreen->get_shader_param = etna_screen_get_shader_param;

   pscreen->get_name = etna_screen_get_name;
   pscreen->get_vendor = etna_screen_get_vendor;
   pscreen->get_device_vendor = etna_screen_get_device_vendor;

   pscreen->get_timestamp = etna_screen_get_timestamp;
   pscreen->context_create = etna_context_create;
   pscreen->is_format_supported = etna_screen_is_format_supported;
   pscreen->query_dmabuf_modifiers = etna_screen_query_dmabuf_modifiers;

   etna_fence_screen_init(pscreen);
   etna_query_screen_init(pscreen);
   etna_resource_screen_init(pscreen);

   util_dynarray_init(&screen->supported_pm_queries, NULL);
   slab_create_parent(&screen->transfer_pool, sizeof(struct etna_transfer), 16);

   if (screen->drm_version >= ETNA_DRM_VERSION_PERFMON)
      etna_pm_query_setup(screen);

   return pscreen;

fail:
   etna_screen_destroy(pscreen);
   return NULL;
}
