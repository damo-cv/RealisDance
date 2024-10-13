/*
 * Copyright 2014, 2015 Red Hat.
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
#include "util/u_format.h"
#include "util/u_format_s3tc.h"
#include "util/u_screen.h"
#include "util/u_video.h"
#include "util/u_math.h"
#include "util/os_time.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"

#include "tgsi/tgsi_exec.h"

#include "virgl_screen.h"
#include "virgl_resource.h"
#include "virgl_public.h"
#include "virgl_context.h"

int virgl_debug = 0;
static const struct debug_named_value debug_options[] = {
   { "verbose", VIRGL_DEBUG_VERBOSE, NULL },
   { "tgsi", VIRGL_DEBUG_TGSI, NULL },
   DEBUG_NAMED_VALUE_END
};
DEBUG_GET_ONCE_FLAGS_OPTION(virgl_debug, "VIRGL_DEBUG", debug_options, 0)

static const char *
virgl_get_vendor(struct pipe_screen *screen)
{
   return "Red Hat";
}


static const char *
virgl_get_name(struct pipe_screen *screen)
{
   return "virgl";
}

static int
virgl_get_param(struct pipe_screen *screen, enum pipe_cap param)
{
   struct virgl_screen *vscreen = virgl_screen(screen);
   switch (param) {
   case PIPE_CAP_NPOT_TEXTURES:
      return 1;
   case PIPE_CAP_SM3:
      return 1;
   case PIPE_CAP_ANISOTROPIC_FILTER:
      return 1;
   case PIPE_CAP_POINT_SPRITE:
      return 1;
   case PIPE_CAP_MAX_RENDER_TARGETS:
      return vscreen->caps.caps.v1.max_render_targets;
   case PIPE_CAP_MAX_DUAL_SOURCE_RENDER_TARGETS:
      return vscreen->caps.caps.v1.max_dual_source_render_targets;
   case PIPE_CAP_OCCLUSION_QUERY:
      return vscreen->caps.caps.v1.bset.occlusion_query;
   case PIPE_CAP_TEXTURE_MIRROR_CLAMP:
   case PIPE_CAP_TEXTURE_MIRROR_CLAMP_TO_EDGE:
      return vscreen->caps.caps.v1.bset.mirror_clamp;
   case PIPE_CAP_TEXTURE_SWIZZLE:
      return 1;
   case PIPE_CAP_MAX_TEXTURE_2D_LEVELS:
      if (vscreen->caps.caps.v2.max_texture_2d_size)
         return 1 + util_logbase2(vscreen->caps.caps.v2.max_texture_2d_size);
      return 15; /* 16K x 16K */
   case PIPE_CAP_MAX_TEXTURE_3D_LEVELS:
      if (vscreen->caps.caps.v2.max_texture_3d_size)
         return 1 + util_logbase2(vscreen->caps.caps.v2.max_texture_3d_size);
      return 9; /* 256 x 256 x 256 */
   case PIPE_CAP_MAX_TEXTURE_CUBE_LEVELS:
      if (vscreen->caps.caps.v2.max_texture_cube_size)
         return 1 + util_logbase2(vscreen->caps.caps.v2.max_texture_cube_size);
      return 13; /* 4K x 4K */
   case PIPE_CAP_BLEND_EQUATION_SEPARATE:
      return 1;
   case PIPE_CAP_INDEP_BLEND_ENABLE:
      return vscreen->caps.caps.v1.bset.indep_blend_enable;
   case PIPE_CAP_INDEP_BLEND_FUNC:
      return vscreen->caps.caps.v1.bset.indep_blend_func;
   case PIPE_CAP_TGSI_FS_COORD_ORIGIN_UPPER_LEFT:
   case PIPE_CAP_TGSI_FS_COORD_PIXEL_CENTER_HALF_INTEGER:
   case PIPE_CAP_TGSI_FS_COORD_PIXEL_CENTER_INTEGER:
      return 1;
   case PIPE_CAP_TGSI_FS_COORD_ORIGIN_LOWER_LEFT:
      return vscreen->caps.caps.v1.bset.fragment_coord_conventions;
   case PIPE_CAP_DEPTH_CLIP_DISABLE:
      return vscreen->caps.caps.v1.bset.depth_clip_disable;
   case PIPE_CAP_MAX_STREAM_OUTPUT_BUFFERS:
      return vscreen->caps.caps.v1.max_streamout_buffers;
   case PIPE_CAP_MAX_STREAM_OUTPUT_SEPARATE_COMPONENTS:
   case PIPE_CAP_MAX_STREAM_OUTPUT_INTERLEAVED_COMPONENTS:
      return 16*4;
   case PIPE_CAP_PRIMITIVE_RESTART:
      return vscreen->caps.caps.v1.bset.primitive_restart;
   case PIPE_CAP_SHADER_STENCIL_EXPORT:
      return vscreen->caps.caps.v1.bset.shader_stencil_export;
   case PIPE_CAP_TGSI_INSTANCEID:
   case PIPE_CAP_VERTEX_ELEMENT_INSTANCE_DIVISOR:
      return 1;
   case PIPE_CAP_SEAMLESS_CUBE_MAP:
      return vscreen->caps.caps.v1.bset.seamless_cube_map;
   case PIPE_CAP_SEAMLESS_CUBE_MAP_PER_TEXTURE:
      return vscreen->caps.caps.v1.bset.seamless_cube_map_per_texture;
   case PIPE_CAP_MAX_TEXTURE_ARRAY_LAYERS:
      return vscreen->caps.caps.v1.max_texture_array_layers;
   case PIPE_CAP_MIN_TEXEL_OFFSET:
      return vscreen->caps.caps.v2.min_texel_offset;
   case PIPE_CAP_MIN_TEXTURE_GATHER_OFFSET:
      return vscreen->caps.caps.v2.min_texture_gather_offset;
   case PIPE_CAP_MAX_TEXEL_OFFSET:
      return vscreen->caps.caps.v2.max_texel_offset;
   case PIPE_CAP_MAX_TEXTURE_GATHER_OFFSET:
      return vscreen->caps.caps.v2.max_texture_gather_offset;
   case PIPE_CAP_CONDITIONAL_RENDER:
      return vscreen->caps.caps.v1.bset.conditional_render;
   case PIPE_CAP_TEXTURE_BARRIER:
      return vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_TEXTURE_BARRIER;
   case PIPE_CAP_VERTEX_COLOR_UNCLAMPED:
      return 1;
   case PIPE_CAP_FRAGMENT_COLOR_CLAMPED:
   case PIPE_CAP_VERTEX_COLOR_CLAMPED:
      return vscreen->caps.caps.v1.bset.color_clamping;
   case PIPE_CAP_MIXED_COLORBUFFER_FORMATS:
      return 1;
   case PIPE_CAP_GLSL_FEATURE_LEVEL:
      return vscreen->caps.caps.v1.glsl_level;
   case PIPE_CAP_GLSL_FEATURE_LEVEL_COMPATIBILITY:
      return MIN2(vscreen->caps.caps.v1.glsl_level, 140);
   case PIPE_CAP_QUADS_FOLLOW_PROVOKING_VERTEX_CONVENTION:
   case PIPE_CAP_DEPTH_CLIP_DISABLE_SEPARATE:
      return 0;
   case PIPE_CAP_COMPUTE:
      return vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_COMPUTE_SHADER;
   case PIPE_CAP_USER_VERTEX_BUFFERS:
      return 0;
   case PIPE_CAP_CONSTANT_BUFFER_OFFSET_ALIGNMENT:
      return vscreen->caps.caps.v2.uniform_buffer_offset_alignment;
   case PIPE_CAP_STREAM_OUTPUT_PAUSE_RESUME:
   case PIPE_CAP_STREAM_OUTPUT_INTERLEAVE_BUFFERS:
      return vscreen->caps.caps.v1.bset.streamout_pause_resume;
   case PIPE_CAP_START_INSTANCE:
      return vscreen->caps.caps.v1.bset.start_instance;
   case PIPE_CAP_TGSI_CAN_COMPACT_CONSTANTS:
   case PIPE_CAP_VERTEX_BUFFER_OFFSET_4BYTE_ALIGNED_ONLY:
   case PIPE_CAP_VERTEX_BUFFER_STRIDE_4BYTE_ALIGNED_ONLY:
   case PIPE_CAP_VERTEX_ELEMENT_SRC_OFFSET_4BYTE_ALIGNED_ONLY:
   case PIPE_CAP_PREFER_BLIT_BASED_TEXTURE_TRANSFER:
      return 0;
   case PIPE_CAP_QUERY_TIMESTAMP:
      return 1;
   case PIPE_CAP_QUERY_TIME_ELAPSED:
      return 0;
   case PIPE_CAP_TGSI_TEXCOORD:
      return 0;
   case PIPE_CAP_MIN_MAP_BUFFER_ALIGNMENT:
      return VIRGL_MAP_BUFFER_ALIGNMENT;
   case PIPE_CAP_TEXTURE_BUFFER_OBJECTS:
      return vscreen->caps.caps.v1.max_tbo_size > 0;
   case PIPE_CAP_TEXTURE_BUFFER_OFFSET_ALIGNMENT:
      return vscreen->caps.caps.v2.texture_buffer_offset_alignment;
   case PIPE_CAP_BUFFER_SAMPLER_VIEW_RGBA_ONLY:
      return 0;
   case PIPE_CAP_CUBE_MAP_ARRAY:
      return vscreen->caps.caps.v1.bset.cube_map_array;
   case PIPE_CAP_TEXTURE_MULTISAMPLE:
      return vscreen->caps.caps.v1.bset.texture_multisample;
   case PIPE_CAP_MAX_VIEWPORTS:
      return vscreen->caps.caps.v1.max_viewports;
   case PIPE_CAP_MAX_TEXTURE_BUFFER_SIZE:
      return vscreen->caps.caps.v1.max_tbo_size;
   case PIPE_CAP_TEXTURE_BORDER_COLOR_QUIRK:
   case PIPE_CAP_QUERY_PIPELINE_STATISTICS:
   case PIPE_CAP_ENDIANNESS:
      return 0;
   case PIPE_CAP_MIXED_FRAMEBUFFER_SIZES:
   case PIPE_CAP_MIXED_COLOR_DEPTH_BITS:
      return 1;
   case PIPE_CAP_TGSI_VS_LAYER_VIEWPORT:
      return 0;
   case PIPE_CAP_MAX_GEOMETRY_OUTPUT_VERTICES:
      return vscreen->caps.caps.v2.max_geom_output_vertices;
   case PIPE_CAP_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS:
      return vscreen->caps.caps.v2.max_geom_total_output_components;
   case PIPE_CAP_TEXTURE_QUERY_LOD:
      return vscreen->caps.caps.v1.bset.texture_query_lod;
   case PIPE_CAP_MAX_TEXTURE_GATHER_COMPONENTS:
      return vscreen->caps.caps.v1.max_texture_gather_components;
   case PIPE_CAP_DRAW_INDIRECT:
      return vscreen->caps.caps.v1.bset.has_indirect_draw;
   case PIPE_CAP_SAMPLE_SHADING:
   case PIPE_CAP_FORCE_PERSAMPLE_INTERP:
      return vscreen->caps.caps.v1.bset.has_sample_shading;
   case PIPE_CAP_CULL_DISTANCE:
      return vscreen->caps.caps.v1.bset.has_cull;
   case PIPE_CAP_MAX_VERTEX_STREAMS:
      return vscreen->caps.caps.v1.glsl_level >= 400 ? 4 : 1;
   case PIPE_CAP_CONDITIONAL_RENDER_INVERTED:
      return vscreen->caps.caps.v1.bset.conditional_render_inverted;
   case PIPE_CAP_TGSI_FS_FINE_DERIVATIVE:
      return vscreen->caps.caps.v1.bset.derivative_control;
   case PIPE_CAP_POLYGON_OFFSET_CLAMP:
      return vscreen->caps.caps.v1.bset.polygon_offset_clamp;
   case PIPE_CAP_QUERY_SO_OVERFLOW:
      return vscreen->caps.caps.v1.bset.transform_feedback_overflow_query;
   case PIPE_CAP_SHADER_BUFFER_OFFSET_ALIGNMENT:
      return vscreen->caps.caps.v2.shader_buffer_offset_alignment;
   case PIPE_CAP_DOUBLES:
      return vscreen->caps.caps.v1.bset.has_fp64;
   case PIPE_CAP_MAX_SHADER_PATCH_VARYINGS:
      return vscreen->caps.caps.v2.max_shader_patch_varyings;
   case PIPE_CAP_SAMPLER_VIEW_TARGET:
      return vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_TEXTURE_VIEW;
   case PIPE_CAP_MAX_VERTEX_ATTRIB_STRIDE:
      return vscreen->caps.caps.v2.max_vertex_attrib_stride;
   case PIPE_CAP_COPY_BETWEEN_COMPRESSED_AND_PLAIN_FORMATS:
      return vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_COPY_IMAGE;
   case PIPE_CAP_TGSI_TXQS:
      return vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_TXQS;
   case PIPE_CAP_FRAMEBUFFER_NO_ATTACHMENT:
      return vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_FB_NO_ATTACH;
   case PIPE_CAP_ROBUST_BUFFER_ACCESS_BEHAVIOR:
      return vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_ROBUST_BUFFER_ACCESS;
   case PIPE_CAP_TGSI_FS_FBFETCH:
      return vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_TGSI_FBFETCH;
   case PIPE_CAP_TGSI_CLOCK:
      return vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_SHADER_CLOCK;
   case PIPE_CAP_TGSI_ARRAY_COMPONENTS:
      return vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_TGSI_COMPONENTS;
   case PIPE_CAP_MAX_COMBINED_SHADER_BUFFERS:
      return vscreen->caps.caps.v2.max_combined_shader_buffers;
   case PIPE_CAP_MAX_COMBINED_HW_ATOMIC_COUNTERS:
      return vscreen->caps.caps.v2.max_combined_atomic_counters;
   case PIPE_CAP_MAX_COMBINED_HW_ATOMIC_COUNTER_BUFFERS:
      return vscreen->caps.caps.v2.max_combined_atomic_counter_buffers;
   case PIPE_CAP_TEXTURE_GATHER_SM5:
   case PIPE_CAP_BUFFER_MAP_PERSISTENT_COHERENT:
   case PIPE_CAP_FAKE_SW_MSAA:
   case PIPE_CAP_TEXTURE_GATHER_OFFSETS:
   case PIPE_CAP_TGSI_VS_WINDOW_SPACE_POSITION:
   case PIPE_CAP_MULTI_DRAW_INDIRECT:
   case PIPE_CAP_MULTI_DRAW_INDIRECT_PARAMS:
   case PIPE_CAP_CLIP_HALFZ:
   case PIPE_CAP_VERTEXID_NOBASE:
   case PIPE_CAP_MULTISAMPLE_Z_RESOLVE:
   case PIPE_CAP_RESOURCE_FROM_USER_MEMORY:
   case PIPE_CAP_DEVICE_RESET_STATUS_QUERY:
   case PIPE_CAP_TEXTURE_FLOAT_LINEAR:
   case PIPE_CAP_TEXTURE_HALF_FLOAT_LINEAR:
   case PIPE_CAP_DEPTH_BOUNDS_TEST:
   case PIPE_CAP_SHAREABLE_SHADERS:
   case PIPE_CAP_CLEAR_TEXTURE:
   case PIPE_CAP_DRAW_PARAMETERS:
   case PIPE_CAP_TGSI_PACK_HALF_FLOAT:
   case PIPE_CAP_TGSI_FS_POSITION_IS_SYSVAL:
   case PIPE_CAP_TGSI_FS_FACE_IS_INTEGER_SYSVAL:
   case PIPE_CAP_INVALIDATE_BUFFER:
   case PIPE_CAP_GENERATE_MIPMAP:
   case PIPE_CAP_SURFACE_REINTERPRET_BLOCKS:
   case PIPE_CAP_QUERY_BUFFER_OBJECT:
   case PIPE_CAP_STRING_MARKER:
   case PIPE_CAP_QUERY_MEMORY_INFO:
   case PIPE_CAP_PCI_GROUP:
   case PIPE_CAP_PCI_BUS:
   case PIPE_CAP_PCI_DEVICE:
   case PIPE_CAP_PCI_FUNCTION:
   case PIPE_CAP_PRIMITIVE_RESTART_FOR_PATCHES:
   case PIPE_CAP_TGSI_VOTE:
   case PIPE_CAP_MAX_WINDOW_RECTANGLES:
   case PIPE_CAP_POLYGON_OFFSET_UNITS_UNSCALED:
   case PIPE_CAP_VIEWPORT_SUBPIXEL_BITS:
   case PIPE_CAP_TGSI_CAN_READ_OUTPUTS:
   case PIPE_CAP_GLSL_OPTIMIZE_CONSERVATIVELY:
   case PIPE_CAP_TGSI_MUL_ZERO_WINS:
   case PIPE_CAP_INT64:
   case PIPE_CAP_INT64_DIVMOD:
   case PIPE_CAP_TGSI_TEX_TXF_LZ:
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
   case PIPE_CAP_MAX_TEXTURE_UPLOAD_MEMORY_BUDGET:
      return 0;
   case PIPE_CAP_MAX_GS_INVOCATIONS:
      return 32;
   case PIPE_CAP_MAX_SHADER_BUFFER_SIZE:
      return 1 << 27;
   case PIPE_CAP_VENDOR_ID:
      return 0x1af4;
   case PIPE_CAP_DEVICE_ID:
      return 0x1010;
   case PIPE_CAP_ACCELERATED:
      return 1;
   case PIPE_CAP_UMA:
   case PIPE_CAP_VIDEO_MEMORY:
      return 0;
   case PIPE_CAP_NATIVE_FENCE_FD:
      return 0;
   default:
      return u_pipe_screen_get_param_defaults(screen, param);
   }
}

static int
virgl_get_shader_param(struct pipe_screen *screen,
                       enum pipe_shader_type shader,
                       enum pipe_shader_cap param)
{
   struct virgl_screen *vscreen = virgl_screen(screen);

   if ((shader == PIPE_SHADER_TESS_CTRL || shader == PIPE_SHADER_TESS_EVAL) &&
       !vscreen->caps.caps.v1.bset.has_tessellation_shaders)
      return 0;

   if (shader == PIPE_SHADER_COMPUTE &&
       !(vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_COMPUTE_SHADER))
     return 0;

   switch(shader)
   {
   case PIPE_SHADER_FRAGMENT:
   case PIPE_SHADER_VERTEX:
   case PIPE_SHADER_GEOMETRY:
   case PIPE_SHADER_TESS_CTRL:
   case PIPE_SHADER_TESS_EVAL:
   case PIPE_SHADER_COMPUTE:
      switch (param) {
      case PIPE_SHADER_CAP_MAX_INSTRUCTIONS:
      case PIPE_SHADER_CAP_MAX_ALU_INSTRUCTIONS:
      case PIPE_SHADER_CAP_MAX_TEX_INSTRUCTIONS:
      case PIPE_SHADER_CAP_MAX_TEX_INDIRECTIONS:
         return INT_MAX;
      case PIPE_SHADER_CAP_INDIRECT_OUTPUT_ADDR:
      case PIPE_SHADER_CAP_INDIRECT_TEMP_ADDR:
      case PIPE_SHADER_CAP_INDIRECT_CONST_ADDR:
         return 1;
      case PIPE_SHADER_CAP_MAX_INPUTS:
         if (vscreen->caps.caps.v1.glsl_level < 150)
            return vscreen->caps.caps.v2.max_vertex_attribs;
         return (shader == PIPE_SHADER_VERTEX ||
                 shader == PIPE_SHADER_GEOMETRY) ? vscreen->caps.caps.v2.max_vertex_attribs : 32;
      case PIPE_SHADER_CAP_MAX_OUTPUTS:
         if (shader == PIPE_SHADER_FRAGMENT)
            return vscreen->caps.caps.v1.max_render_targets;
         return vscreen->caps.caps.v2.max_vertex_outputs;
     // case PIPE_SHADER_CAP_MAX_CONSTS:
     //    return 4096;
      case PIPE_SHADER_CAP_MAX_TEMPS:
         return 256;
      case PIPE_SHADER_CAP_MAX_CONST_BUFFERS:
         return vscreen->caps.caps.v1.max_uniform_blocks;
    //  case PIPE_SHADER_CAP_MAX_ADDRS:
     //    return 1;
      case PIPE_SHADER_CAP_SUBROUTINES:
         return 1;
      case PIPE_SHADER_CAP_MAX_TEXTURE_SAMPLERS:
            return 16;
      case PIPE_SHADER_CAP_INTEGERS:
         return vscreen->caps.caps.v1.glsl_level >= 130;
      case PIPE_SHADER_CAP_MAX_CONTROL_FLOW_DEPTH:
         return 32;
      case PIPE_SHADER_CAP_MAX_CONST_BUFFER_SIZE:
         return 4096 * sizeof(float[4]);
      case PIPE_SHADER_CAP_MAX_SHADER_BUFFERS:
         if (shader == PIPE_SHADER_FRAGMENT || shader == PIPE_SHADER_COMPUTE)
            return vscreen->caps.caps.v2.max_shader_buffer_frag_compute;
         else
            return vscreen->caps.caps.v2.max_shader_buffer_other_stages;
      case PIPE_SHADER_CAP_MAX_SHADER_IMAGES:
         if (shader == PIPE_SHADER_FRAGMENT || shader == PIPE_SHADER_COMPUTE)
            return vscreen->caps.caps.v2.max_shader_image_frag_compute;
         else
            return vscreen->caps.caps.v2.max_shader_image_other_stages;
      case PIPE_SHADER_CAP_SUPPORTED_IRS:
         return (1 << PIPE_SHADER_IR_TGSI);
      case PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTERS:
         return vscreen->caps.caps.v2.max_atomic_counters[shader];
      case PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTER_BUFFERS:
         return vscreen->caps.caps.v2.max_atomic_counter_buffers[shader];
      case PIPE_SHADER_CAP_LOWER_IF_THRESHOLD:
      case PIPE_SHADER_CAP_TGSI_SKIP_MERGE_REGISTERS:
      case PIPE_SHADER_CAP_INT64_ATOMICS:
      case PIPE_SHADER_CAP_FP16:
         return 0;
      case PIPE_SHADER_CAP_SCALAR_ISA:
         return 1;
      default:
         return 0;
      }
   default:
      return 0;
   }
}

static float
virgl_get_paramf(struct pipe_screen *screen, enum pipe_capf param)
{
   struct virgl_screen *vscreen = virgl_screen(screen);
   switch (param) {
   case PIPE_CAPF_MAX_LINE_WIDTH:
      return vscreen->caps.caps.v2.max_aliased_line_width;
   case PIPE_CAPF_MAX_LINE_WIDTH_AA:
      return vscreen->caps.caps.v2.max_smooth_line_width;
   case PIPE_CAPF_MAX_POINT_WIDTH:
      return vscreen->caps.caps.v2.max_aliased_point_size;
   case PIPE_CAPF_MAX_POINT_WIDTH_AA:
      return vscreen->caps.caps.v2.max_smooth_point_size;
   case PIPE_CAPF_MAX_TEXTURE_ANISOTROPY:
      return 16.0;
   case PIPE_CAPF_MAX_TEXTURE_LOD_BIAS:
      return vscreen->caps.caps.v2.max_texture_lod_bias;
   case PIPE_CAPF_MIN_CONSERVATIVE_RASTER_DILATE:
   case PIPE_CAPF_MAX_CONSERVATIVE_RASTER_DILATE:
   case PIPE_CAPF_CONSERVATIVE_RASTER_DILATE_GRANULARITY:
      return 0.0f;
   }
   /* should only get here on unhandled cases */
   debug_printf("Unexpected PIPE_CAPF %d query\n", param);
   return 0.0;
}

static int
virgl_get_compute_param(struct pipe_screen *screen,
                        enum pipe_shader_ir ir_type,
                        enum pipe_compute_cap param,
                        void *ret)
{
   struct virgl_screen *vscreen = virgl_screen(screen);
   if (!(vscreen->caps.caps.v2.capability_bits & VIRGL_CAP_COMPUTE_SHADER))
      return 0;
   switch (param) {
   case PIPE_COMPUTE_CAP_MAX_GRID_SIZE:
      if (ret) {
         uint64_t *grid_size = ret;
         grid_size[0] = vscreen->caps.caps.v2.max_compute_grid_size[0];
         grid_size[1] = vscreen->caps.caps.v2.max_compute_grid_size[1];
         grid_size[2] = vscreen->caps.caps.v2.max_compute_grid_size[2];
      }
      return 3 * sizeof(uint64_t) ;
   case PIPE_COMPUTE_CAP_MAX_BLOCK_SIZE:
      if (ret) {
         uint64_t *block_size = ret;
         block_size[0] = vscreen->caps.caps.v2.max_compute_block_size[0];
         block_size[1] = vscreen->caps.caps.v2.max_compute_block_size[1];
         block_size[2] = vscreen->caps.caps.v2.max_compute_block_size[2];
      }
      return 3 * sizeof(uint64_t);
   case PIPE_COMPUTE_CAP_MAX_THREADS_PER_BLOCK:
      if (ret) {
         uint64_t *max_threads_per_block = ret;
         *max_threads_per_block = vscreen->caps.caps.v2.max_compute_work_group_invocations;
      }
      return sizeof(uint64_t);
   case PIPE_COMPUTE_CAP_MAX_LOCAL_SIZE:
      if (ret) {
         uint64_t *max_local_size = ret;
         /* Value reported by the closed source driver. */
         *max_local_size = vscreen->caps.caps.v2.max_compute_shared_memory_size;
      }
      return sizeof(uint64_t);
   default:
      break;
   }
   return 0;
}

static boolean
virgl_is_vertex_format_supported(struct pipe_screen *screen,
                                 enum pipe_format format)
{
   struct virgl_screen *vscreen = virgl_screen(screen);
   const struct util_format_description *format_desc;
   int i;

   format_desc = util_format_description(format);
   if (!format_desc)
      return FALSE;

   if (format == PIPE_FORMAT_R11G11B10_FLOAT) {
      int vformat = VIRGL_FORMAT_R11G11B10_FLOAT;
      int big = vformat / 32;
      int small = vformat % 32;
      if (!(vscreen->caps.caps.v1.vertexbuffer.bitmask[big] & (1 << small)))
         return FALSE;
      return TRUE;
   }

   /* Find the first non-VOID channel. */
   for (i = 0; i < 4; i++) {
      if (format_desc->channel[i].type != UTIL_FORMAT_TYPE_VOID) {
         break;
      }
   }

   if (i == 4)
      return FALSE;

   if (format_desc->layout != UTIL_FORMAT_LAYOUT_PLAIN)
      return FALSE;

   if (format_desc->channel[i].type == UTIL_FORMAT_TYPE_FIXED)
      return FALSE;
   return TRUE;
}

/**
 * Query format support for creating a texture, drawing surface, etc.
 * \param format  the format to test
 * \param type  one of PIPE_TEXTURE, PIPE_SURFACE
 */
static boolean
virgl_is_format_supported( struct pipe_screen *screen,
                                 enum pipe_format format,
                                 enum pipe_texture_target target,
                                 unsigned sample_count,
                                 unsigned storage_sample_count,
                                 unsigned bind)
{
   struct virgl_screen *vscreen = virgl_screen(screen);
   const struct util_format_description *format_desc;
   int i;

   if (MAX2(1, sample_count) != MAX2(1, storage_sample_count))
      return false;

   assert(target == PIPE_BUFFER ||
          target == PIPE_TEXTURE_1D ||
          target == PIPE_TEXTURE_1D_ARRAY ||
          target == PIPE_TEXTURE_2D ||
          target == PIPE_TEXTURE_2D_ARRAY ||
          target == PIPE_TEXTURE_RECT ||
          target == PIPE_TEXTURE_3D ||
          target == PIPE_TEXTURE_CUBE ||
          target == PIPE_TEXTURE_CUBE_ARRAY);

   format_desc = util_format_description(format);
   if (!format_desc)
      return FALSE;

   if (util_format_is_intensity(format))
      return FALSE;

   if (sample_count > 1) {
      if (!vscreen->caps.caps.v1.bset.texture_multisample)
         return FALSE;

      if (bind & PIPE_BIND_SHADER_IMAGE) {
         if (sample_count > vscreen->caps.caps.v2.max_image_samples)
            return FALSE;
      }

      if (sample_count > vscreen->caps.caps.v1.max_samples)
         return FALSE;
   }

   if (bind & PIPE_BIND_VERTEX_BUFFER) {
      return virgl_is_vertex_format_supported(screen, format);
   }

   /* Allow 3-comp 32 bit textures only for TBOs (needed for ARB_tbo_rgb32) */
   if ((format == PIPE_FORMAT_R32G32B32_FLOAT ||
       format == PIPE_FORMAT_R32G32B32_SINT ||
       format == PIPE_FORMAT_R32G32B32_UINT) &&
       target != PIPE_BUFFER)
      return FALSE;

   if (bind & PIPE_BIND_RENDER_TARGET) {
      /* For ARB_framebuffer_no_attachments. */
      if (format == PIPE_FORMAT_NONE)
         return TRUE;

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

      {
         int big = format / 32;
         int small = format % 32;
         if (!(vscreen->caps.caps.v1.render.bitmask[big] & (1 << small)))
            return FALSE;
      }
   }

   if (bind & PIPE_BIND_DEPTH_STENCIL) {
      if (format_desc->colorspace != UTIL_FORMAT_COLORSPACE_ZS)
         return FALSE;
   }

   /*
    * All other operations (sampling, transfer, etc).
    */

   if (format_desc->layout == UTIL_FORMAT_LAYOUT_S3TC) {
      goto out_lookup;
   }
   if (format_desc->layout == UTIL_FORMAT_LAYOUT_RGTC) {
      goto out_lookup;
   }
   if (format_desc->layout == UTIL_FORMAT_LAYOUT_BPTC) {
      goto out_lookup;
   }

   if (format == PIPE_FORMAT_R11G11B10_FLOAT) {
      goto out_lookup;
   } else if (format == PIPE_FORMAT_R9G9B9E5_FLOAT) {
      goto out_lookup;
   }

   /* Find the first non-VOID channel. */
   for (i = 0; i < 4; i++) {
      if (format_desc->channel[i].type != UTIL_FORMAT_TYPE_VOID) {
         break;
      }
   }

   if (i == 4)
      return FALSE;

   /* no L4A4 */
   if (format_desc->nr_channels < 4 && format_desc->channel[i].size == 4)
      return FALSE;

 out_lookup:
   {
      int big = format / 32;
      int small = format % 32;
      if (!(vscreen->caps.caps.v1.sampler.bitmask[big] & (1 << small)))
         return FALSE;
   }
   /*
    * Everything else should be supported by u_format.
    */
   return TRUE;
}

static void virgl_flush_frontbuffer(struct pipe_screen *screen,
                                      struct pipe_resource *res,
                                      unsigned level, unsigned layer,
                                    void *winsys_drawable_handle, struct pipe_box *sub_box)
{
   struct virgl_screen *vscreen = virgl_screen(screen);
   struct virgl_winsys *vws = vscreen->vws;
   struct virgl_resource *vres = virgl_resource(res);

   if (vws->flush_frontbuffer)
      vws->flush_frontbuffer(vws, vres->hw_res, level, layer, winsys_drawable_handle,
                             sub_box);
}

static void virgl_fence_reference(struct pipe_screen *screen,
                                  struct pipe_fence_handle **ptr,
                                  struct pipe_fence_handle *fence)
{
   struct virgl_screen *vscreen = virgl_screen(screen);
   struct virgl_winsys *vws = vscreen->vws;

   vws->fence_reference(vws, ptr, fence);
}

static boolean virgl_fence_finish(struct pipe_screen *screen,
                                  struct pipe_context *ctx,
                                  struct pipe_fence_handle *fence,
                                  uint64_t timeout)
{
   struct virgl_screen *vscreen = virgl_screen(screen);
   struct virgl_winsys *vws = vscreen->vws;

   return vws->fence_wait(vws, fence, timeout);
}

static uint64_t
virgl_get_timestamp(struct pipe_screen *_screen)
{
   return os_time_get_nano();
}

static void
virgl_destroy_screen(struct pipe_screen *screen)
{
   struct virgl_screen *vscreen = virgl_screen(screen);
   struct virgl_winsys *vws = vscreen->vws;

   slab_destroy_parent(&vscreen->texture_transfer_pool);

   if (vws)
      vws->destroy(vws);
   FREE(vscreen);
}

struct pipe_screen *
virgl_create_screen(struct virgl_winsys *vws)
{
   struct virgl_screen *screen = CALLOC_STRUCT(virgl_screen);

   if (!screen)
      return NULL;

   virgl_debug = debug_get_option_virgl_debug();

   screen->vws = vws;
   screen->base.get_name = virgl_get_name;
   screen->base.get_vendor = virgl_get_vendor;
   screen->base.get_param = virgl_get_param;
   screen->base.get_shader_param = virgl_get_shader_param;
   screen->base.get_compute_param = virgl_get_compute_param;
   screen->base.get_paramf = virgl_get_paramf;
   screen->base.is_format_supported = virgl_is_format_supported;
   screen->base.destroy = virgl_destroy_screen;
   screen->base.context_create = virgl_context_create;
   screen->base.flush_frontbuffer = virgl_flush_frontbuffer;
   screen->base.get_timestamp = virgl_get_timestamp;
   screen->base.fence_reference = virgl_fence_reference;
   //screen->base.fence_signalled = virgl_fence_signalled;
   screen->base.fence_finish = virgl_fence_finish;

   virgl_init_screen_resource_functions(&screen->base);

   vws->get_caps(vws, &screen->caps);

   screen->refcnt = 1;

   slab_create_parent(&screen->texture_transfer_pool, sizeof(struct virgl_transfer), 16);

   return &screen->base;
}
