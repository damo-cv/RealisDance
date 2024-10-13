/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


#include "util/u_memory.h"
#include "util/u_format.h"
#include "util/u_format_s3tc.h"
#include "util/u_screen.h"
#include "util/u_video.h"
#include "util/os_misc.h"
#include "util/os_time.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "draw/draw_context.h"

#include "state_tracker/sw_winsys.h"
#include "tgsi/tgsi_exec.h"

#include "sp_texture.h"
#include "sp_screen.h"
#include "sp_context.h"
#include "sp_fence.h"
#include "sp_public.h"

DEBUG_GET_ONCE_BOOL_OPTION(use_llvm, "SOFTPIPE_USE_LLVM", FALSE)

static const char *
softpipe_get_vendor(struct pipe_screen *screen)
{
   return "VMware, Inc.";
}


static const char *
softpipe_get_name(struct pipe_screen *screen)
{
   return "softpipe";
}


static int
softpipe_get_param(struct pipe_screen *screen, enum pipe_cap param)
{
   switch (param) {
   case PIPE_CAP_NPOT_TEXTURES:
   case PIPE_CAP_MIXED_FRAMEBUFFER_SIZES:
   case PIPE_CAP_MIXED_COLOR_DEPTH_BITS:
      return 1;
   case PIPE_CAP_SM3:
      return 1;
   case PIPE_CAP_ANISOTROPIC_FILTER:
      return 1;
   case PIPE_CAP_POINT_SPRITE:
      return 1;
   case PIPE_CAP_MAX_RENDER_TARGETS:
      return PIPE_MAX_COLOR_BUFS;
   case PIPE_CAP_MAX_DUAL_SOURCE_RENDER_TARGETS:
      return 1;
   case PIPE_CAP_OCCLUSION_QUERY:
      return 1;
   case PIPE_CAP_QUERY_TIME_ELAPSED:
      return 1;
   case PIPE_CAP_QUERY_PIPELINE_STATISTICS:
      return 1;
   case PIPE_CAP_TEXTURE_MIRROR_CLAMP:
   case PIPE_CAP_TEXTURE_MIRROR_CLAMP_TO_EDGE:
      return 1;
   case PIPE_CAP_TEXTURE_SWIZZLE:
      return 1;
   case PIPE_CAP_TEXTURE_BORDER_COLOR_QUIRK:
   case PIPE_CAP_MAX_TEXTURE_UPLOAD_MEMORY_BUDGET:
      return 0;
   case PIPE_CAP_MAX_TEXTURE_2D_LEVELS:
      return SP_MAX_TEXTURE_2D_LEVELS;
   case PIPE_CAP_MAX_TEXTURE_3D_LEVELS:
      return SP_MAX_TEXTURE_3D_LEVELS;
   case PIPE_CAP_MAX_TEXTURE_CUBE_LEVELS:
      return SP_MAX_TEXTURE_CUBE_LEVELS;
   case PIPE_CAP_BLEND_EQUATION_SEPARATE:
      return 1;
   case PIPE_CAP_INDEP_BLEND_ENABLE:
      return 1;
   case PIPE_CAP_INDEP_BLEND_FUNC:
      return 1;
   case PIPE_CAP_TGSI_FS_COORD_ORIGIN_UPPER_LEFT:
   case PIPE_CAP_TGSI_FS_COORD_ORIGIN_LOWER_LEFT:
   case PIPE_CAP_TGSI_FS_COORD_PIXEL_CENTER_HALF_INTEGER:
   case PIPE_CAP_TGSI_FS_COORD_PIXEL_CENTER_INTEGER:
      return 1;
   case PIPE_CAP_DEPTH_CLIP_DISABLE:
      return 1;
   case PIPE_CAP_MAX_STREAM_OUTPUT_BUFFERS:
      return PIPE_MAX_SO_BUFFERS;
   case PIPE_CAP_MAX_STREAM_OUTPUT_SEPARATE_COMPONENTS:
   case PIPE_CAP_MAX_STREAM_OUTPUT_INTERLEAVED_COMPONENTS:
      return 16*4;
   case PIPE_CAP_MAX_GEOMETRY_OUTPUT_VERTICES:
   case PIPE_CAP_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS:
      return 1024;
   case PIPE_CAP_MAX_VERTEX_STREAMS:
      return 1;
   case PIPE_CAP_MAX_VERTEX_ATTRIB_STRIDE:
      return 2048;
   case PIPE_CAP_PRIMITIVE_RESTART:
      return 1;
   case PIPE_CAP_SHADER_STENCIL_EXPORT:
      return 1;
   case PIPE_CAP_TGSI_INSTANCEID:
   case PIPE_CAP_VERTEX_ELEMENT_INSTANCE_DIVISOR:
   case PIPE_CAP_START_INSTANCE:
      return 1;
   case PIPE_CAP_SEAMLESS_CUBE_MAP:
   case PIPE_CAP_SEAMLESS_CUBE_MAP_PER_TEXTURE:
      return 1;
   case PIPE_CAP_MAX_TEXTURE_ARRAY_LAYERS:
      return 256; /* for GL3 */
   case PIPE_CAP_MIN_TEXEL_OFFSET:
      return -8;
   case PIPE_CAP_MAX_TEXEL_OFFSET:
      return 7;
   case PIPE_CAP_CONDITIONAL_RENDER:
      return 1;
   case PIPE_CAP_TEXTURE_BARRIER:
   case PIPE_CAP_DEPTH_CLIP_DISABLE_SEPARATE:
      return 0;
   case PIPE_CAP_FRAGMENT_COLOR_CLAMPED:
   case PIPE_CAP_VERTEX_COLOR_UNCLAMPED: /* draw module */
   case PIPE_CAP_VERTEX_COLOR_CLAMPED: /* draw module */
      return 1;
   case PIPE_CAP_MIXED_COLORBUFFER_FORMATS:
      return 0;
   case PIPE_CAP_GLSL_FEATURE_LEVEL:
      return 330;
   case PIPE_CAP_GLSL_FEATURE_LEVEL_COMPATIBILITY:
      return 140;
   case PIPE_CAP_QUADS_FOLLOW_PROVOKING_VERTEX_CONVENTION:
   case PIPE_CAP_TGSI_TEX_TXF_LZ:
      return 0;
   case PIPE_CAP_COMPUTE:
      return 1;
   case PIPE_CAP_USER_VERTEX_BUFFERS:
   case PIPE_CAP_STREAM_OUTPUT_PAUSE_RESUME:
   case PIPE_CAP_STREAM_OUTPUT_INTERLEAVE_BUFFERS:
   case PIPE_CAP_TGSI_VS_LAYER_VIEWPORT:
   case PIPE_CAP_DOUBLES:
   case PIPE_CAP_INT64:
   case PIPE_CAP_INT64_DIVMOD:
      return 1;
   case PIPE_CAP_CONSTANT_BUFFER_OFFSET_ALIGNMENT:
      return 16;
   case PIPE_CAP_TGSI_CAN_COMPACT_CONSTANTS:
   case PIPE_CAP_VERTEX_BUFFER_OFFSET_4BYTE_ALIGNED_ONLY:
   case PIPE_CAP_VERTEX_BUFFER_STRIDE_4BYTE_ALIGNED_ONLY:
   case PIPE_CAP_VERTEX_ELEMENT_SRC_OFFSET_4BYTE_ALIGNED_ONLY:
   case PIPE_CAP_TEXTURE_MULTISAMPLE:
      return 0;
   case PIPE_CAP_MIN_MAP_BUFFER_ALIGNMENT:
      return 64;
   case PIPE_CAP_QUERY_TIMESTAMP:
   case PIPE_CAP_CUBE_MAP_ARRAY:
      return 1;
   case PIPE_CAP_TEXTURE_BUFFER_OBJECTS:
      return 1;
   case PIPE_CAP_BUFFER_SAMPLER_VIEW_RGBA_ONLY:
      return 0;
   case PIPE_CAP_MAX_TEXTURE_BUFFER_SIZE:
      return 65536;
   case PIPE_CAP_TEXTURE_BUFFER_OFFSET_ALIGNMENT:
      return 0;
   case PIPE_CAP_TGSI_TEXCOORD:
   case PIPE_CAP_PREFER_BLIT_BASED_TEXTURE_TRANSFER:
      return 0;
   case PIPE_CAP_MAX_VIEWPORTS:
      return PIPE_MAX_VIEWPORTS;
   case PIPE_CAP_ENDIANNESS:
      return PIPE_ENDIAN_NATIVE;
   case PIPE_CAP_MAX_TEXTURE_GATHER_COMPONENTS:
      return 4;
   case PIPE_CAP_TEXTURE_GATHER_SM5:
   case PIPE_CAP_TEXTURE_QUERY_LOD:
      return 1;
   case PIPE_CAP_BUFFER_MAP_PERSISTENT_COHERENT:
   case PIPE_CAP_SAMPLE_SHADING:
   case PIPE_CAP_TEXTURE_GATHER_OFFSETS:
      return 0;
   case PIPE_CAP_TGSI_VS_WINDOW_SPACE_POSITION:
      return 1;
   case PIPE_CAP_TGSI_FS_FINE_DERIVATIVE:
      return 0;
   case PIPE_CAP_SAMPLER_VIEW_TARGET:
      return 1;
   case PIPE_CAP_FAKE_SW_MSAA:
      return 1;
   case PIPE_CAP_MIN_TEXTURE_GATHER_OFFSET:
      return -32;
   case PIPE_CAP_MAX_TEXTURE_GATHER_OFFSET:
      return 31;
   case PIPE_CAP_DRAW_INDIRECT:
      return 1;
   case PIPE_CAP_QUERY_SO_OVERFLOW:
      return 1;

   case PIPE_CAP_VENDOR_ID:
      return 0xFFFFFFFF;
   case PIPE_CAP_DEVICE_ID:
      return 0xFFFFFFFF;
   case PIPE_CAP_ACCELERATED:
      return 0;
   case PIPE_CAP_VIDEO_MEMORY: {
      /* XXX: Do we want to return the full amount fo system memory ? */
      uint64_t system_memory;

      if (!os_get_total_physical_memory(&system_memory))
         return 0;

      if (sizeof(void *) == 4)
         /* Cap to 2 GB on 32 bits system. We do this because softpipe does
          * eat application memory, which is quite limited on 32 bits. App
          * shouldn't expect too much available memory. */
         system_memory = MIN2(system_memory, 2048 << 20);

      return (int)(system_memory >> 20);
   }
   case PIPE_CAP_UMA:
      return 0;
   case PIPE_CAP_CONDITIONAL_RENDER_INVERTED:
      return 1;
   case PIPE_CAP_CLIP_HALFZ:
   case PIPE_CAP_TEXTURE_FLOAT_LINEAR:
   case PIPE_CAP_TEXTURE_HALF_FLOAT_LINEAR:
      return 1;
   case PIPE_CAP_FRAMEBUFFER_NO_ATTACHMENT:
   case PIPE_CAP_CULL_DISTANCE:
      return 1;
   case PIPE_CAP_VERTEXID_NOBASE:
      return 0;
   case PIPE_CAP_POLYGON_OFFSET_CLAMP:
      return 0;
   case PIPE_CAP_COPY_BETWEEN_COMPRESSED_AND_PLAIN_FORMATS:
   case PIPE_CAP_TGSI_ARRAY_COMPONENTS:
      return 1;
   case PIPE_CAP_CLEAR_TEXTURE:
      return 1;
   case PIPE_CAP_MULTISAMPLE_Z_RESOLVE:
   case PIPE_CAP_RESOURCE_FROM_USER_MEMORY:
   case PIPE_CAP_DEVICE_RESET_STATUS_QUERY:
   case PIPE_CAP_MAX_SHADER_PATCH_VARYINGS:
   case PIPE_CAP_DEPTH_BOUNDS_TEST:
   case PIPE_CAP_TGSI_TXQS:
   case PIPE_CAP_FORCE_PERSAMPLE_INTERP:
   case PIPE_CAP_SHAREABLE_SHADERS:
   case PIPE_CAP_DRAW_PARAMETERS:
   case PIPE_CAP_TGSI_PACK_HALF_FLOAT:
   case PIPE_CAP_MULTI_DRAW_INDIRECT:
   case PIPE_CAP_MULTI_DRAW_INDIRECT_PARAMS:
   case PIPE_CAP_TGSI_FS_POSITION_IS_SYSVAL:
   case PIPE_CAP_TGSI_FS_FACE_IS_INTEGER_SYSVAL:
   case PIPE_CAP_INVALIDATE_BUFFER:
   case PIPE_CAP_GENERATE_MIPMAP:
   case PIPE_CAP_STRING_MARKER:
   case PIPE_CAP_SURFACE_REINTERPRET_BLOCKS:
   case PIPE_CAP_QUERY_BUFFER_OBJECT:
   case PIPE_CAP_QUERY_MEMORY_INFO:
   case PIPE_CAP_PCI_GROUP:
   case PIPE_CAP_PCI_BUS:
   case PIPE_CAP_PCI_DEVICE:
   case PIPE_CAP_PCI_FUNCTION:
   case PIPE_CAP_ROBUST_BUFFER_ACCESS_BEHAVIOR:
   case PIPE_CAP_PRIMITIVE_RESTART_FOR_PATCHES:
   case PIPE_CAP_TGSI_VOTE:
   case PIPE_CAP_MAX_WINDOW_RECTANGLES:
   case PIPE_CAP_POLYGON_OFFSET_UNITS_UNSCALED:
   case PIPE_CAP_VIEWPORT_SUBPIXEL_BITS:
   case PIPE_CAP_TGSI_CAN_READ_OUTPUTS:
   case PIPE_CAP_NATIVE_FENCE_FD:
   case PIPE_CAP_GLSL_OPTIMIZE_CONSERVATIVELY:
   case PIPE_CAP_TGSI_FS_FBFETCH:
   case PIPE_CAP_TGSI_MUL_ZERO_WINS:
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
   case PIPE_CAP_PACKED_UNIFORMS:
   case PIPE_CAP_CONSERVATIVE_RASTER_POST_SNAP_TRIANGLES:
   case PIPE_CAP_CONSERVATIVE_RASTER_POST_SNAP_POINTS_LINES:
   case PIPE_CAP_CONSERVATIVE_RASTER_PRE_SNAP_TRIANGLES:
   case PIPE_CAP_CONSERVATIVE_RASTER_PRE_SNAP_POINTS_LINES:
   case PIPE_CAP_CONSERVATIVE_RASTER_POST_DEPTH_COVERAGE:
   case PIPE_CAP_MAX_CONSERVATIVE_RASTER_SUBPIXEL_PRECISION_BIAS:
   case PIPE_CAP_PROGRAMMABLE_SAMPLE_LOCATIONS:
      return 0;
   case PIPE_CAP_MAX_GS_INVOCATIONS:
      return 32;
   case PIPE_CAP_MAX_SHADER_BUFFER_SIZE:
      return 1 << 27;
   case PIPE_CAP_SHADER_BUFFER_OFFSET_ALIGNMENT:
      return 4;
   default:
      return u_pipe_screen_get_param_defaults(screen, param);
   }
}

static int
softpipe_get_shader_param(struct pipe_screen *screen,
                          enum pipe_shader_type shader,
                          enum pipe_shader_cap param)
{
   struct softpipe_screen *sp_screen = softpipe_screen(screen);
   switch(shader)
   {
   case PIPE_SHADER_FRAGMENT:
      return tgsi_exec_get_shader_param(param);
   case PIPE_SHADER_COMPUTE:
      return tgsi_exec_get_shader_param(param);
   case PIPE_SHADER_VERTEX:
   case PIPE_SHADER_GEOMETRY:
      if (sp_screen->use_llvm)
         return draw_get_shader_param(shader, param);
      else
         return draw_get_shader_param_no_llvm(shader, param);
   default:
      return 0;
   }
}

static float
softpipe_get_paramf(struct pipe_screen *screen, enum pipe_capf param)
{
   switch (param) {
   case PIPE_CAPF_MAX_LINE_WIDTH:
      /* fall-through */
   case PIPE_CAPF_MAX_LINE_WIDTH_AA:
      return 255.0; /* arbitrary */
   case PIPE_CAPF_MAX_POINT_WIDTH:
      /* fall-through */
   case PIPE_CAPF_MAX_POINT_WIDTH_AA:
      return 255.0; /* arbitrary */
   case PIPE_CAPF_MAX_TEXTURE_ANISOTROPY:
      return 16.0;
   case PIPE_CAPF_MAX_TEXTURE_LOD_BIAS:
      return 16.0; /* arbitrary */
   case PIPE_CAPF_MIN_CONSERVATIVE_RASTER_DILATE:
      return 0.0;
   case PIPE_CAPF_MAX_CONSERVATIVE_RASTER_DILATE:
      return 0.0;
   case PIPE_CAPF_CONSERVATIVE_RASTER_DILATE_GRANULARITY:
      return 0.0;
   }
   /* should only get here on unhandled cases */
   debug_printf("Unexpected PIPE_CAPF %d query\n", param);
   return 0.0;
}

/**
 * Query format support for creating a texture, drawing surface, etc.
 * \param format  the format to test
 * \param type  one of PIPE_TEXTURE, PIPE_SURFACE
 */
static boolean
softpipe_is_format_supported( struct pipe_screen *screen,
                              enum pipe_format format,
                              enum pipe_texture_target target,
                              unsigned sample_count,
                              unsigned storage_sample_count,
                              unsigned bind)
{
   struct sw_winsys *winsys = softpipe_screen(screen)->winsys;
   const struct util_format_description *format_desc;

   assert(target == PIPE_BUFFER ||
          target == PIPE_TEXTURE_1D ||
          target == PIPE_TEXTURE_1D_ARRAY ||
          target == PIPE_TEXTURE_2D ||
          target == PIPE_TEXTURE_2D_ARRAY ||
          target == PIPE_TEXTURE_RECT ||
          target == PIPE_TEXTURE_3D ||
          target == PIPE_TEXTURE_CUBE ||
          target == PIPE_TEXTURE_CUBE_ARRAY);

   if (MAX2(1, sample_count) != MAX2(1, storage_sample_count))
      return false;

   format_desc = util_format_description(format);
   if (!format_desc)
      return FALSE;

   if (sample_count > 1)
      return FALSE;

   if (bind & (PIPE_BIND_DISPLAY_TARGET |
               PIPE_BIND_SCANOUT |
               PIPE_BIND_SHARED)) {
      if(!winsys->is_displaytarget_format_supported(winsys, bind, format))
         return FALSE;
   }

   if (bind & PIPE_BIND_RENDER_TARGET) {
      if (format_desc->colorspace == UTIL_FORMAT_COLORSPACE_ZS)
         return FALSE;

      /*
       * Although possible, it is unnatural to render into compressed or YUV
       * surfaces. So disable these here to avoid going into weird paths
       * inside the state trackers.
       */
      if (format_desc->block.width != 1 ||
          format_desc->block.height != 1)
         return FALSE;
   }

   if (bind & PIPE_BIND_DEPTH_STENCIL) {
      if (format_desc->colorspace != UTIL_FORMAT_COLORSPACE_ZS)
         return FALSE;
   }

   if (format_desc->layout == UTIL_FORMAT_LAYOUT_ASTC) {
      /* Software decoding is not hooked up. */
      return FALSE;
   }

   if ((bind & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW)) &&
       ((bind & PIPE_BIND_DISPLAY_TARGET) == 0) &&
       target != PIPE_BUFFER) {
      const struct util_format_description *desc =
         util_format_description(format);
      if (desc->nr_channels == 3 && desc->is_array) {
         /* Don't support any 3-component formats for rendering/texturing
          * since we don't support the corresponding 8-bit 3 channel UNORM
          * formats.  This allows us to support GL_ARB_copy_image between
          * GL_RGB8 and GL_RGB8UI, for example.  Otherwise, we may be asked to
          * do a resource copy between PIPE_FORMAT_R8G8B8_UINT and
          * PIPE_FORMAT_R8G8B8X8_UNORM, for example, which will not work
          * (different bpp).
          */
         return FALSE;
      }
   }

   if (format_desc->layout == UTIL_FORMAT_LAYOUT_ETC &&
       format != PIPE_FORMAT_ETC1_RGB8)
      return FALSE;

   /*
    * All other operations (sampling, transfer, etc).
    */

   /*
    * Everything else should be supported by u_format.
    */
   return TRUE;
}


static void
softpipe_destroy_screen( struct pipe_screen *screen )
{
   struct softpipe_screen *sp_screen = softpipe_screen(screen);
   struct sw_winsys *winsys = sp_screen->winsys;

   if(winsys->destroy)
      winsys->destroy(winsys);

   FREE(screen);
}


/* This is often overriden by the co-state tracker.
 */
static void
softpipe_flush_frontbuffer(struct pipe_screen *_screen,
                           struct pipe_resource *resource,
                           unsigned level, unsigned layer,
                           void *context_private,
                           struct pipe_box *sub_box)
{
   struct softpipe_screen *screen = softpipe_screen(_screen);
   struct sw_winsys *winsys = screen->winsys;
   struct softpipe_resource *texture = softpipe_resource(resource);

   assert(texture->dt);
   if (texture->dt)
      winsys->displaytarget_display(winsys, texture->dt, context_private, sub_box);
}

static uint64_t
softpipe_get_timestamp(struct pipe_screen *_screen)
{
   return os_time_get_nano();
}

static int
softpipe_get_compute_param(struct pipe_screen *_screen,
                           enum pipe_shader_ir ir_type,
                           enum pipe_compute_cap param,
                           void *ret)
{
   switch (param) {
   case PIPE_COMPUTE_CAP_IR_TARGET:
      return 0;
   case PIPE_COMPUTE_CAP_MAX_GRID_SIZE:
      if (ret) {
         uint64_t *grid_size = ret;
         grid_size[0] = 65535;
         grid_size[1] = 65535;
         grid_size[2] = 65535;
      }
      return 3 * sizeof(uint64_t) ;
   case PIPE_COMPUTE_CAP_MAX_BLOCK_SIZE:
      if (ret) {
         uint64_t *block_size = ret;
         block_size[0] = 1024;
         block_size[1] = 1024;
         block_size[2] = 1024;
      }
      return 3 * sizeof(uint64_t);
   case PIPE_COMPUTE_CAP_MAX_THREADS_PER_BLOCK:
      if (ret) {
         uint64_t *max_threads_per_block = ret;
         *max_threads_per_block = 1024;
      }
      return sizeof(uint64_t);
   case PIPE_COMPUTE_CAP_MAX_LOCAL_SIZE:
      if (ret) {
         uint64_t *max_local_size = ret;
         *max_local_size = 32768;
      }
      return sizeof(uint64_t);
   case PIPE_COMPUTE_CAP_GRID_DIMENSION:
   case PIPE_COMPUTE_CAP_MAX_GLOBAL_SIZE:
   case PIPE_COMPUTE_CAP_MAX_PRIVATE_SIZE:
   case PIPE_COMPUTE_CAP_MAX_INPUT_SIZE:
   case PIPE_COMPUTE_CAP_MAX_MEM_ALLOC_SIZE:
   case PIPE_COMPUTE_CAP_MAX_CLOCK_FREQUENCY:
   case PIPE_COMPUTE_CAP_MAX_COMPUTE_UNITS:
   case PIPE_COMPUTE_CAP_IMAGES_SUPPORTED:
   case PIPE_COMPUTE_CAP_SUBGROUP_SIZE:
   case PIPE_COMPUTE_CAP_ADDRESS_BITS:
   case PIPE_COMPUTE_CAP_MAX_VARIABLE_THREADS_PER_BLOCK:
      break;
   }
   return 0;
}

/**
 * Create a new pipe_screen object
 * Note: we're not presently subclassing pipe_screen (no softpipe_screen).
 */
struct pipe_screen *
softpipe_create_screen(struct sw_winsys *winsys)
{
   struct softpipe_screen *screen = CALLOC_STRUCT(softpipe_screen);

   if (!screen)
      return NULL;

   screen->winsys = winsys;

   screen->base.destroy = softpipe_destroy_screen;

   screen->base.get_name = softpipe_get_name;
   screen->base.get_vendor = softpipe_get_vendor;
   screen->base.get_device_vendor = softpipe_get_vendor; // TODO should be the CPU vendor
   screen->base.get_param = softpipe_get_param;
   screen->base.get_shader_param = softpipe_get_shader_param;
   screen->base.get_paramf = softpipe_get_paramf;
   screen->base.get_timestamp = softpipe_get_timestamp;
   screen->base.is_format_supported = softpipe_is_format_supported;
   screen->base.context_create = softpipe_create_context;
   screen->base.flush_frontbuffer = softpipe_flush_frontbuffer;
   screen->base.get_compute_param = softpipe_get_compute_param;
   screen->use_llvm = debug_get_option_use_llvm();

   softpipe_init_screen_texture_funcs(&screen->base);
   softpipe_init_screen_fence_funcs(&screen->base);

   return &screen->base;
}
