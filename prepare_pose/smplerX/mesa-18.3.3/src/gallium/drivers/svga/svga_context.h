/**********************************************************
 * Copyright 2008-2009 VMware, Inc.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **********************************************************/

#ifndef SVGA_CONTEXT_H
#define SVGA_CONTEXT_H


#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_state.h"

#include "util/os_time.h"

#include "util/u_blitter.h"
#include "util/list.h"

#include "tgsi/tgsi_scan.h"

#include "svga_screen.h"
#include "svga_state.h"
#include "svga_winsys.h"
#include "svga_hw_reg.h"
#include "svga3d_shaderdefs.h"


/** Non-GPU queries for gallium HUD */
enum svga_hud {
/* per-frame counters */
   SVGA_QUERY_NUM_DRAW_CALLS = PIPE_QUERY_DRIVER_SPECIFIC,
   SVGA_QUERY_NUM_FALLBACKS,
   SVGA_QUERY_NUM_FLUSHES,
   SVGA_QUERY_NUM_VALIDATIONS,
   SVGA_QUERY_MAP_BUFFER_TIME,
   SVGA_QUERY_NUM_BUFFERS_MAPPED,
   SVGA_QUERY_NUM_TEXTURES_MAPPED,
   SVGA_QUERY_NUM_BYTES_UPLOADED,
   SVGA_QUERY_COMMAND_BUFFER_SIZE,
   SVGA_QUERY_FLUSH_TIME,
   SVGA_QUERY_SURFACE_WRITE_FLUSHES,
   SVGA_QUERY_NUM_READBACKS,
   SVGA_QUERY_NUM_RESOURCE_UPDATES,
   SVGA_QUERY_NUM_BUFFER_UPLOADS,
   SVGA_QUERY_NUM_CONST_BUF_UPDATES,
   SVGA_QUERY_NUM_CONST_UPDATES,

/* running total counters */
   SVGA_QUERY_MEMORY_USED,
   SVGA_QUERY_NUM_SHADERS,
   SVGA_QUERY_NUM_RESOURCES,
   SVGA_QUERY_NUM_STATE_OBJECTS,
   SVGA_QUERY_NUM_SURFACE_VIEWS,
   SVGA_QUERY_NUM_GENERATE_MIPMAP,
   SVGA_QUERY_NUM_FAILED_ALLOCATIONS,
   SVGA_QUERY_NUM_COMMANDS_PER_DRAW,

/*SVGA_QUERY_MAX has to be last because it is size of an array*/
   SVGA_QUERY_MAX
};

/**
 * Maximum supported number of constant buffers per shader
 */
#define SVGA_MAX_CONST_BUFS 14

/**
 * Maximum constant buffer size that can be set in the
 * DXSetSingleConstantBuffer command is
 * DX10 constant buffer element count * 4 4-bytes components
 */
#define SVGA_MAX_CONST_BUF_SIZE (4096 * 4 * sizeof(int))

#define CONST0_UPLOAD_ALIGNMENT 256

struct draw_vertex_shader;
struct draw_fragment_shader;
struct svga_shader_variant;
struct SVGACmdMemory;
struct util_bitmask;


struct svga_cache_context;
struct svga_tracked_state;

struct svga_blend_state {
   unsigned need_white_fragments:1;
   unsigned independent_blend_enable:1;
   unsigned alpha_to_coverage:1;
   unsigned alpha_to_one:1;
   unsigned blend_color_alpha:1;  /**< set blend color to alpha value */

   /** Per-render target state */
   struct {
      uint8_t writemask;

      boolean blend_enable;
      uint8_t srcblend;
      uint8_t dstblend;
      uint8_t blendeq;

      boolean separate_alpha_blend_enable;
      uint8_t srcblend_alpha;
      uint8_t dstblend_alpha;
      uint8_t blendeq_alpha;
   } rt[PIPE_MAX_COLOR_BUFS];

   SVGA3dBlendStateId id;  /**< vgpu10 */
};

struct svga_depth_stencil_state {
   unsigned zfunc:8;
   unsigned zenable:1;
   unsigned zwriteenable:1;

   unsigned alphatestenable:1;
   unsigned alphafunc:8;

   struct {
      unsigned enabled:1;
      unsigned func:8;
      unsigned fail:8;
      unsigned zfail:8;
      unsigned pass:8;
   } stencil[2];

   /* SVGA3D has one ref/mask/writemask triple shared between front &
    * back face stencil.  We really need two:
    */
   unsigned stencil_mask:8;
   unsigned stencil_writemask:8;

   float    alpharef;

   SVGA3dDepthStencilStateId id;  /**< vgpu10 */
};

#define SVGA_UNFILLED_DISABLE 0
#define SVGA_UNFILLED_LINE    1
#define SVGA_UNFILLED_POINT   2

#define SVGA_PIPELINE_FLAG_POINTS   (1<<PIPE_PRIM_POINTS)
#define SVGA_PIPELINE_FLAG_LINES    (1<<PIPE_PRIM_LINES)
#define SVGA_PIPELINE_FLAG_TRIS     (1<<PIPE_PRIM_TRIANGLES)

struct svga_rasterizer_state {
   struct pipe_rasterizer_state templ; /* needed for draw module */

   unsigned shademode:8;
   unsigned cullmode:8;
   unsigned scissortestenable:1;
   unsigned multisampleantialias:1;
   unsigned antialiasedlineenable:1;
   unsigned lastpixel:1;
   unsigned pointsprite:1;

   unsigned linepattern;

   float slopescaledepthbias;
   float depthbias;
   float pointsize;
   float linewidth;

   unsigned hw_fillmode:2;         /* PIPE_POLYGON_MODE_x */

   /** Which prims do we need help for?  Bitmask of (1 << PIPE_PRIM_x) flags */
   unsigned need_pipeline:16;

   SVGA3dRasterizerStateId id;    /**< vgpu10 */

   /** For debugging: */
   const char* need_pipeline_tris_str;
   const char* need_pipeline_lines_str;
   const char* need_pipeline_points_str;
};

struct svga_sampler_state {
   unsigned mipfilter;
   unsigned magfilter;
   unsigned minfilter;
   unsigned aniso_level;
   float lod_bias;
   unsigned addressu;
   unsigned addressv;
   unsigned addressw;
   unsigned bordercolor;
   unsigned normalized_coords:1;
   unsigned compare_mode:1;
   unsigned compare_func:3;

   unsigned min_lod;
   unsigned view_min_lod;
   unsigned view_max_lod;

   SVGA3dSamplerId id[2];
};


struct svga_pipe_sampler_view
{
   struct pipe_sampler_view base;

   SVGA3dShaderResourceViewId id;
};


static inline struct svga_pipe_sampler_view *
svga_pipe_sampler_view(struct pipe_sampler_view *v)
{
   return (struct svga_pipe_sampler_view *) v;
}


struct svga_velems_state {
   unsigned count;
   struct pipe_vertex_element velem[PIPE_MAX_ATTRIBS];
   SVGA3dDeclType decl_type[PIPE_MAX_ATTRIBS]; /**< vertex attrib formats */

   /** Bitmasks indicating which attributes need format conversion */
   unsigned adjust_attrib_range;     /**< range adjustment */
   unsigned attrib_is_pure_int;      /**< pure int */
   unsigned adjust_attrib_w_1;       /**< set w = 1 */
   unsigned adjust_attrib_itof;      /**< int->float */
   unsigned adjust_attrib_utof;      /**< uint->float */
   unsigned attrib_is_bgra;          /**< R / B swizzling */
   unsigned attrib_puint_to_snorm;   /**< 10_10_10_2 packed uint -> snorm */
   unsigned attrib_puint_to_uscaled; /**< 10_10_10_2 packed uint -> uscaled */
   unsigned attrib_puint_to_sscaled; /**< 10_10_10_2 packed uint -> sscaled */

   boolean need_swvfetch;

   SVGA3dElementLayoutId id; /**< VGPU10 */
};


/* Use to calculate differences between state emitted to hardware and
 * current driver-calculated state.
 */
struct svga_state
{
   const struct svga_blend_state *blend;
   const struct svga_depth_stencil_state *depth;
   const struct svga_rasterizer_state *rast;
   const struct svga_sampler_state *sampler[PIPE_SHADER_TYPES][PIPE_MAX_SAMPLERS];
   const struct svga_velems_state *velems;

   struct pipe_sampler_view *sampler_views[PIPE_SHADER_TYPES][PIPE_MAX_SAMPLERS]; /* or texture ID's? */
   struct svga_fragment_shader *fs;
   struct svga_vertex_shader *vs;
   struct svga_geometry_shader *user_gs; /* user-specified GS */
   struct svga_geometry_shader *gs;      /* derived GS */

   struct pipe_vertex_buffer vb[PIPE_MAX_ATTRIBS];
   /** Constant buffers for each shader.
    * The size should probably always match with that of
    * svga_shader_emitter_v10.num_shader_consts.
    */
   struct pipe_constant_buffer constbufs[PIPE_SHADER_TYPES][SVGA_MAX_CONST_BUFS];

   struct pipe_framebuffer_state framebuffer;
   float depthscale;

   /* Hack to limit the number of different render targets between
    * flushes.  Helps avoid blowing out our surface cache in EXA.
    */
   int nr_fbs;

   struct pipe_poly_stipple poly_stipple;
   struct pipe_scissor_state scissor;
   struct pipe_blend_color blend_color;
   struct pipe_stencil_ref stencil_ref;
   struct pipe_clip_state clip;
   struct pipe_viewport_state viewport;

   unsigned num_samplers[PIPE_SHADER_TYPES];
   unsigned num_sampler_views[PIPE_SHADER_TYPES];
   unsigned num_vertex_buffers;
   enum pipe_prim_type reduced_prim;

   struct {
      unsigned flag_1d;
      unsigned flag_srgb;
   } tex_flags;

   unsigned sample_mask;
};

struct svga_prescale {
   float translate[4];
   float scale[4];
   boolean enabled;
};


/* Updated by calling svga_update_state( SVGA_STATE_HW_CLEAR )
 */
struct svga_hw_clear_state
{
   SVGA3dRect viewport;

   struct {
      float zmin, zmax;
   } depthrange;

   struct pipe_framebuffer_state framebuffer;
   struct svga_prescale prescale;

   /* VGPU10 state */
   unsigned num_rendertargets;
   struct pipe_surface *rtv[SVGA3D_MAX_RENDER_TARGETS];
   struct pipe_surface *dsv;
};

struct svga_hw_view_state
{
   struct pipe_resource *texture;
   struct svga_sampler_view *v;
   unsigned min_lod;
   unsigned max_lod;
   boolean dirty;
};

/* Updated by calling svga_update_state( SVGA_STATE_HW_DRAW )
 */
struct svga_hw_draw_state
{
   /** VGPU9 rasterization state */
   unsigned rs[SVGA3D_RS_MAX];
   /** VGPU9 texture sampler and bindings state */
   unsigned ts[SVGA3D_PIXEL_SAMPLERREG_MAX][SVGA3D_TS_MAX];

   /** VGPU9 texture views */
   unsigned num_views;
   unsigned num_backed_views; /* views with backing copy of texture */
   struct svga_hw_view_state views[PIPE_MAX_SAMPLERS];

   /** VGPU9 constant buffer values */
   float cb[PIPE_SHADER_TYPES][SVGA3D_CONSTREG_MAX][4];

   /** Currently bound shaders */
   struct svga_shader_variant *fs;
   struct svga_shader_variant *vs;
   struct svga_shader_variant *gs;

   /** Currently bound constant buffer, per shader stage */
   struct pipe_resource *constbuf[PIPE_SHADER_TYPES];

   /** Bitmask of enabled constant buffers */
   unsigned enabled_constbufs[PIPE_SHADER_TYPES];

   /**
    * These are used to reduce the number of times we call u_upload_unmap()
    * while updating the zero-th/default VGPU10 constant buffer.
    */
   struct pipe_resource *const0_buffer;
   struct svga_winsys_surface *const0_handle;

   /** VGPU10 HW state (used to prevent emitting redundant state) */
   SVGA3dDepthStencilStateId depth_stencil_id;
   unsigned stencil_ref;
   SVGA3dBlendStateId blend_id;
   float blend_factor[4];
   unsigned blend_sample_mask;
   SVGA3dRasterizerStateId rasterizer_id;
   SVGA3dElementLayoutId layout_id;
   SVGA3dPrimitiveType topology;

   /** Vertex buffer state */
   SVGA3dVertexBuffer vbuffer_attrs[PIPE_MAX_ATTRIBS];
   struct pipe_resource *vbuffers[PIPE_MAX_ATTRIBS];
   unsigned num_vbuffers;

   struct pipe_resource *ib;  /**< index buffer for drawing */
   SVGA3dSurfaceFormat ib_format;
   unsigned ib_offset;

   unsigned num_samplers[PIPE_SHADER_TYPES];
   SVGA3dSamplerId samplers[PIPE_SHADER_TYPES][PIPE_MAX_SAMPLERS];

   unsigned num_sampler_views[PIPE_SHADER_TYPES];
   struct pipe_sampler_view
      *sampler_views[PIPE_SHADER_TYPES][PIPE_MAX_SAMPLERS];

   /* used for rebinding */
   unsigned default_constbuf_size[PIPE_SHADER_TYPES];

   boolean rasterizer_discard; /* set if rasterization is disabled */
   boolean has_backed_views;   /* set if any of the rtv/dsv is a backed surface view */
};


/* Updated by calling svga_update_state( SVGA_STATE_NEED_SWTNL )
 */
struct svga_sw_state
{
   /* which parts we need */
   boolean need_swvfetch;
   boolean need_pipeline;
   boolean need_swtnl;

   /* Flag to make sure that need sw is on while
    * updating state within a swtnl call.
    */
   boolean in_swtnl_draw;
};


/* Queue some state updates (like rss) and submit them to hardware in
 * a single packet.
 */
struct svga_hw_queue;

struct svga_query;
struct svga_qmem_alloc_entry;

struct svga_context
{
   struct pipe_context pipe;
   struct svga_winsys_context *swc;
   struct blitter_context *blitter;
   struct u_upload_mgr *const0_upload;
   struct u_upload_mgr *tex_upload;

   struct {
      boolean no_swtnl;
      boolean force_swtnl;
      boolean use_min_mipmap;

      /* incremented for each shader */
      unsigned shader_id;

      boolean no_line_width;
      boolean force_hw_line_stipple;

      /** To report perf/conformance/etc issues to the state tracker */
      struct pipe_debug_callback callback;
   } debug;

   struct {
      struct draw_context *draw;
      struct vbuf_render *backend;
      unsigned hw_prim;
      boolean new_vbuf;
      boolean new_vdecl;
   } swtnl;

   /* Bitmask of blend state objects IDs */
   struct util_bitmask *blend_object_id_bm;

   /* Bitmask of depth/stencil state objects IDs */
   struct util_bitmask *ds_object_id_bm;

   /* Bitmaks of input element object IDs */
   struct util_bitmask *input_element_object_id_bm;

   /* Bitmask of rasterizer object IDs */
   struct util_bitmask *rast_object_id_bm;

   /* Bitmask of sampler state objects IDs */
   struct util_bitmask *sampler_object_id_bm;

   /* Bitmask of sampler view IDs */
   struct util_bitmask *sampler_view_id_bm;

   /* Bitmask of used shader IDs */
   struct util_bitmask *shader_id_bm;

   /* Bitmask of used surface view IDs */
   struct util_bitmask *surface_view_id_bm;

   /* Bitmask of used stream output IDs */
   struct util_bitmask *stream_output_id_bm;

   /* Bitmask of used query IDs */
   struct util_bitmask *query_id_bm;

   struct {
      unsigned dirty[SVGA_STATE_MAX];

      /** bitmasks of which const buffers are changed */
      unsigned dirty_constbufs[PIPE_SHADER_TYPES];

      unsigned texture_timestamp;

      struct svga_sw_state          sw;
      struct svga_hw_draw_state     hw_draw;
      struct svga_hw_clear_state    hw_clear;
   } state;

   struct svga_state curr;      /* state from the state tracker */
   unsigned dirty;              /* statechanges since last update_state() */

   union {
      struct {
         unsigned rendertargets:1;
         unsigned texture_samplers:1;
         unsigned constbufs:1;
         unsigned vs:1;
         unsigned fs:1;
         unsigned gs:1;
         unsigned query:1;
      } flags;
      unsigned val;
   } rebind;

   struct svga_hwtnl *hwtnl;

   /** Queries states */
   struct svga_winsys_gb_query *gb_query;     /**< gb query object, one per context */
   unsigned gb_query_len;                     /**< gb query object size */
   struct util_bitmask *gb_query_alloc_mask;  /**< gb query object allocation mask */
   struct svga_qmem_alloc_entry *gb_query_map[SVGA_QUERY_MAX];
                                              /**< query mem block mapping */
   struct svga_query *sq[SVGA_QUERY_MAX];     /**< queries currently in progress */

   /** List of buffers with queued transfers */
   struct list_head dirty_buffers;

   /** performance / info queries for HUD */
   struct {
      uint64_t num_draw_calls;          /**< SVGA_QUERY_DRAW_CALLS */
      uint64_t num_fallbacks;           /**< SVGA_QUERY_NUM_FALLBACKS */
      uint64_t num_flushes;             /**< SVGA_QUERY_NUM_FLUSHES */
      uint64_t num_validations;         /**< SVGA_QUERY_NUM_VALIDATIONS */
      uint64_t map_buffer_time;         /**< SVGA_QUERY_MAP_BUFFER_TIME */
      uint64_t num_buffers_mapped;      /**< SVGA_QUERY_NUM_BUFFERS_MAPPED */
      uint64_t num_textures_mapped;     /**< SVGA_QUERY_NUM_TEXTURES_MAPPED */
      uint64_t command_buffer_size;     /**< SVGA_QUERY_COMMAND_BUFFER_SIZE */
      uint64_t flush_time;              /**< SVGA_QUERY_FLUSH_TIME */
      uint64_t surface_write_flushes;   /**< SVGA_QUERY_SURFACE_WRITE_FLUSHES */
      uint64_t num_readbacks;           /**< SVGA_QUERY_NUM_READBACKS */
      uint64_t num_resource_updates;    /**< SVGA_QUERY_NUM_RESOURCE_UPDATES */
      uint64_t num_buffer_uploads;      /**< SVGA_QUERY_NUM_BUFFER_UPLOADS */
      uint64_t num_const_buf_updates;   /**< SVGA_QUERY_NUM_CONST_BUF_UPDATES */
      uint64_t num_const_updates;       /**< SVGA_QUERY_NUM_CONST_UPDATES */
      uint64_t num_shaders;             /**< SVGA_QUERY_NUM_SHADERS */

      /** The following are summed for SVGA_QUERY_NUM_STATE_OBJECTS */
      uint64_t num_blend_objects;
      uint64_t num_depthstencil_objects;
      uint64_t num_rasterizer_objects;
      uint64_t num_sampler_objects;
      uint64_t num_samplerview_objects;
      uint64_t num_vertexelement_objects;

      uint64_t num_surface_views;       /**< SVGA_QUERY_NUM_SURFACE_VIEWS */
      uint64_t num_bytes_uploaded;      /**< SVGA_QUERY_NUM_BYTES_UPLOADED */
      uint64_t num_generate_mipmap;     /**< SVGA_QUERY_NUM_GENERATE_MIPMAP */

      boolean uses_time;                /**< os_time_get() calls needed? */
   } hud;

   /** The currently bound stream output targets */
   unsigned num_so_targets;
   struct svga_winsys_surface *so_surfaces[SVGA3D_DX_MAX_SOTARGETS];
   struct pipe_stream_output_target *so_targets[SVGA3D_DX_MAX_SOTARGETS];
   struct svga_stream_output *current_so;

   /** A blend state with blending disabled, for falling back to when blending
    * is illegal (e.g. an integer texture is bound)
    */
   struct svga_blend_state *noop_blend;

   struct {
      struct pipe_resource *texture;
      struct svga_pipe_sampler_view *sampler_view;
      void *sampler;
   } polygon_stipple;

   /** Alternate rasterizer states created for point sprite */
   struct svga_rasterizer_state *rasterizer_no_cull[2];

   /** Depth stencil state created to disable depth stencil test */
   struct svga_depth_stencil_state *depthstencil_disable;

   /** Current conditional rendering predicate */
   struct {
      SVGA3dQueryId query_id;
      boolean cond;
   } pred;

   boolean render_condition;
   boolean disable_rasterizer; /* Set if to disable rasterization */
};

/* A flag for each state_tracker state object:
 */
#define SVGA_NEW_BLEND               0x1
#define SVGA_NEW_DEPTH_STENCIL_ALPHA 0x2
#define SVGA_NEW_RAST                0x4
#define SVGA_NEW_SAMPLER             0x8
#define SVGA_NEW_TEXTURE             0x10
#define SVGA_NEW_VBUFFER             0x20
#define SVGA_NEW_VELEMENT            0x40
#define SVGA_NEW_FS                  0x80
#define SVGA_NEW_VS                  0x100
#define SVGA_NEW_FS_CONST_BUFFER     0x200
#define SVGA_NEW_VS_CONST_BUFFER     0x400
#define SVGA_NEW_FRAME_BUFFER        0x800
#define SVGA_NEW_STIPPLE             0x1000
#define SVGA_NEW_SCISSOR             0x2000
#define SVGA_NEW_BLEND_COLOR         0x4000
#define SVGA_NEW_CLIP                0x8000
#define SVGA_NEW_VIEWPORT            0x10000
#define SVGA_NEW_PRESCALE            0x20000
#define SVGA_NEW_REDUCED_PRIMITIVE   0x40000
#define SVGA_NEW_TEXTURE_BINDING     0x80000
#define SVGA_NEW_NEED_PIPELINE       0x100000
#define SVGA_NEW_NEED_SWVFETCH       0x200000
#define SVGA_NEW_NEED_SWTNL          0x400000
#define SVGA_NEW_FS_VARIANT          0x800000
#define SVGA_NEW_VS_VARIANT          0x1000000
#define SVGA_NEW_TEXTURE_FLAGS       0x4000000
#define SVGA_NEW_STENCIL_REF         0x8000000
#define SVGA_NEW_GS                  0x10000000
#define SVGA_NEW_GS_CONST_BUFFER     0x20000000
#define SVGA_NEW_GS_VARIANT          0x40000000
#define SVGA_NEW_TEXTURE_CONSTS      0x80000000


void svga_init_state_functions( struct svga_context *svga );
void svga_init_flush_functions( struct svga_context *svga );
void svga_init_string_functions( struct svga_context *svga );
void svga_init_blit_functions(struct svga_context *svga);

void svga_init_blend_functions( struct svga_context *svga );
void svga_init_depth_stencil_functions( struct svga_context *svga );
void svga_init_misc_functions( struct svga_context *svga );
void svga_init_rasterizer_functions( struct svga_context *svga );
void svga_init_sampler_functions( struct svga_context *svga );
void svga_init_fs_functions( struct svga_context *svga );
void svga_init_vs_functions( struct svga_context *svga );
void svga_init_gs_functions( struct svga_context *svga );
void svga_init_vertex_functions( struct svga_context *svga );
void svga_init_constbuffer_functions( struct svga_context *svga );
void svga_init_draw_functions( struct svga_context *svga );
void svga_init_query_functions( struct svga_context *svga );
void svga_init_surface_functions(struct svga_context *svga);
void svga_init_stream_output_functions( struct svga_context *svga );
void svga_init_clear_functions( struct svga_context *svga );

void svga_cleanup_vertex_state( struct svga_context *svga );
void svga_cleanup_sampler_state( struct svga_context *svga );
void svga_cleanup_tss_binding( struct svga_context *svga );
void svga_cleanup_framebuffer( struct svga_context *svga );

void svga_context_flush( struct svga_context *svga,
                         struct pipe_fence_handle **pfence );

void svga_context_finish(struct svga_context *svga);

void svga_hwtnl_flush_retry( struct svga_context *svga );
void svga_hwtnl_flush_buffer( struct svga_context *svga,
                              struct pipe_resource *buffer );

void svga_surfaces_flush(struct svga_context *svga);

struct pipe_context *
svga_context_create(struct pipe_screen *screen,
                    void *priv, unsigned flags);

void svga_toggle_render_condition(struct svga_context *svga,
                                  boolean render_condition_enabled,
                                  boolean on);

/***********************************************************************
 * Inline conversion functions.  These are better-typed than the
 * macros used previously:
 */
static inline struct svga_context *
svga_context( struct pipe_context *pipe )
{
   return (struct svga_context *)pipe;
}

static inline struct svga_winsys_screen *
svga_sws(struct svga_context *svga)
{
   return svga_screen(svga->pipe.screen)->sws;
}

static inline boolean
svga_have_gb_objects(const struct svga_context *svga)
{
   return svga_screen(svga->pipe.screen)->sws->have_gb_objects;
}

static inline boolean
svga_have_gb_dma(const struct svga_context *svga)
{
   return svga_screen(svga->pipe.screen)->sws->have_gb_dma;
}

static inline boolean
svga_have_vgpu10(const struct svga_context *svga)
{
   return svga_screen(svga->pipe.screen)->sws->have_vgpu10;
}

static inline boolean
svga_have_sm4_1(const struct svga_context *svga)
{
   return svga_screen(svga->pipe.screen)->sws->have_sm4_1;
}

static inline boolean
svga_need_to_rebind_resources(const struct svga_context *svga)
{
   return svga_screen(svga->pipe.screen)->sws->need_to_rebind_resources;
}

static inline boolean
svga_rects_equal(const SVGA3dRect *r1, const SVGA3dRect *r2)
{
   return memcmp(r1, r2, sizeof(*r1)) == 0;
}

/**
 * If the Gallium HUD is enabled, this will return the current time.
 * Otherwise, just return zero.
 */
static inline int64_t
svga_get_time(struct svga_context *svga)
{
   return svga->hud.uses_time ? os_time_get() : 0;
}


#endif
