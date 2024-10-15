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

#include "pipe/p_shader_tokens.h"

#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_format.h"
#include "util/u_prim.h"
#include "util/u_transfer.h"
#include "util/u_helpers.h"
#include "util/slab.h"
#include "util/u_upload_mgr.h"
#include "util/u_blitter.h"
#include "tgsi/tgsi_text.h"
#include "indices/u_primconvert.h"

#include "pipebuffer/pb_buffer.h"

#include "virgl_encode.h"
#include "virgl_context.h"
#include "virgl_protocol.h"
#include "virgl_resource.h"
#include "virgl_screen.h"

struct virgl_vertex_elements_state {
   uint32_t handle;
   uint8_t binding_map[PIPE_MAX_ATTRIBS];
   uint8_t num_bindings;
};

static uint32_t next_handle;
uint32_t virgl_object_assign_handle(void)
{
   return ++next_handle;
}

static void virgl_buffer_flush(struct virgl_context *vctx,
                              struct virgl_buffer *vbuf)
{
   struct virgl_screen *rs = virgl_screen(vctx->base.screen);
   struct pipe_box box;

   assert(vbuf->on_list);

   box.height = 1;
   box.depth = 1;
   box.y = 0;
   box.z = 0;

   box.x = vbuf->valid_buffer_range.start;
   box.width = MIN2(vbuf->valid_buffer_range.end - vbuf->valid_buffer_range.start, vbuf->base.u.b.width0);

   vctx->num_transfers++;
   rs->vws->transfer_put(rs->vws, vbuf->base.hw_res,
                         &box, 0, 0, box.x, 0);

   util_range_set_empty(&vbuf->valid_buffer_range);
}

static void virgl_attach_res_framebuffer(struct virgl_context *vctx)
{
   struct virgl_winsys *vws = virgl_screen(vctx->base.screen)->vws;
   struct pipe_surface *surf;
   struct virgl_resource *res;
   unsigned i;

   surf = vctx->framebuffer.zsbuf;
   if (surf) {
      res = virgl_resource(surf->texture);
      if (res)
         vws->emit_res(vws, vctx->cbuf, res->hw_res, FALSE);
   }
   for (i = 0; i < vctx->framebuffer.nr_cbufs; i++) {
      surf = vctx->framebuffer.cbufs[i];
      if (surf) {
         res = virgl_resource(surf->texture);
         if (res)
            vws->emit_res(vws, vctx->cbuf, res->hw_res, FALSE);
      }
   }
}

static void virgl_attach_res_sampler_views(struct virgl_context *vctx,
                                           enum pipe_shader_type shader_type)
{
   struct virgl_winsys *vws = virgl_screen(vctx->base.screen)->vws;
   struct virgl_textures_info *tinfo = &vctx->samplers[shader_type];
   struct virgl_resource *res;
   uint32_t remaining_mask = tinfo->enabled_mask;
   unsigned i;
   while (remaining_mask) {
      i = u_bit_scan(&remaining_mask);
      assert(tinfo->views[i]);

      res = virgl_resource(tinfo->views[i]->base.texture);
      if (res)
         vws->emit_res(vws, vctx->cbuf, res->hw_res, FALSE);
   }
}

static void virgl_attach_res_vertex_buffers(struct virgl_context *vctx)
{
   struct virgl_winsys *vws = virgl_screen(vctx->base.screen)->vws;
   struct virgl_resource *res;
   unsigned i;

   for (i = 0; i < vctx->num_vertex_buffers; i++) {
      res = virgl_resource(vctx->vertex_buffer[i].buffer.resource);
      if (res)
         vws->emit_res(vws, vctx->cbuf, res->hw_res, FALSE);
   }
}

static void virgl_attach_res_index_buffer(struct virgl_context *vctx,
					  struct virgl_indexbuf *ib)
{
   struct virgl_winsys *vws = virgl_screen(vctx->base.screen)->vws;
   struct virgl_resource *res;

   res = virgl_resource(ib->buffer);
   if (res)
      vws->emit_res(vws, vctx->cbuf, res->hw_res, FALSE);
}

static void virgl_attach_res_so_targets(struct virgl_context *vctx)
{
   struct virgl_winsys *vws = virgl_screen(vctx->base.screen)->vws;
   struct virgl_resource *res;
   unsigned i;

   for (i = 0; i < vctx->num_so_targets; i++) {
      res = virgl_resource(vctx->so_targets[i].base.buffer);
      if (res)
         vws->emit_res(vws, vctx->cbuf, res->hw_res, FALSE);
   }
}

static void virgl_attach_res_uniform_buffers(struct virgl_context *vctx,
                                             enum pipe_shader_type shader_type)
{
   struct virgl_winsys *vws = virgl_screen(vctx->base.screen)->vws;
   struct virgl_resource *res;
   unsigned i;
   for (i = 0; i < PIPE_MAX_CONSTANT_BUFFERS; i++) {
      res = virgl_resource(vctx->ubos[shader_type][i]);
      if (res) {
         vws->emit_res(vws, vctx->cbuf, res->hw_res, FALSE);
      }
   }
}

static void virgl_attach_res_shader_buffers(struct virgl_context *vctx,
                                            enum pipe_shader_type shader_type)
{
   struct virgl_winsys *vws = virgl_screen(vctx->base.screen)->vws;
   struct virgl_resource *res;
   unsigned i;
   for (i = 0; i < PIPE_MAX_SHADER_BUFFERS; i++) {
      res = virgl_resource(vctx->ssbos[shader_type][i]);
      if (res) {
         vws->emit_res(vws, vctx->cbuf, res->hw_res, FALSE);
      }
   }
}

static void virgl_attach_res_shader_images(struct virgl_context *vctx,
                                           enum pipe_shader_type shader_type)
{
   struct virgl_winsys *vws = virgl_screen(vctx->base.screen)->vws;
   struct virgl_resource *res;
   unsigned i;
   for (i = 0; i < PIPE_MAX_SHADER_IMAGES; i++) {
      res = virgl_resource(vctx->images[shader_type][i]);
      if (res) {
         vws->emit_res(vws, vctx->cbuf, res->hw_res, FALSE);
      }
   }
}

static void virgl_attach_res_atomic_buffers(struct virgl_context *vctx)
{
   struct virgl_winsys *vws = virgl_screen(vctx->base.screen)->vws;
   struct virgl_resource *res;
   unsigned i;
   for (i = 0; i < PIPE_MAX_HW_ATOMIC_BUFFERS; i++) {
      res = virgl_resource(vctx->atomic_buffers[i]);
      if (res) {
         vws->emit_res(vws, vctx->cbuf, res->hw_res, FALSE);
      }
   }
}

/*
 * after flushing, the hw context still has a bunch of
 * resources bound, so we need to rebind those here.
 */
static void virgl_reemit_res(struct virgl_context *vctx)
{
   enum pipe_shader_type shader_type;

   /* reattach any flushed resources */
   /* framebuffer, sampler views, vertex/index/uniform/stream buffers */
   virgl_attach_res_framebuffer(vctx);

   for (shader_type = 0; shader_type < PIPE_SHADER_TYPES; shader_type++) {
      virgl_attach_res_sampler_views(vctx, shader_type);
      virgl_attach_res_uniform_buffers(vctx, shader_type);
      virgl_attach_res_shader_buffers(vctx, shader_type);
      virgl_attach_res_shader_images(vctx, shader_type);
   }
   virgl_attach_res_atomic_buffers(vctx);
   virgl_attach_res_vertex_buffers(vctx);
   virgl_attach_res_so_targets(vctx);
}

static struct pipe_surface *virgl_create_surface(struct pipe_context *ctx,
                                                struct pipe_resource *resource,
                                                const struct pipe_surface *templ)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_surface *surf;
   struct virgl_resource *res = virgl_resource(resource);
   uint32_t handle;

   surf = CALLOC_STRUCT(virgl_surface);
   if (!surf)
      return NULL;

   res->clean = FALSE;
   handle = virgl_object_assign_handle();
   pipe_reference_init(&surf->base.reference, 1);
   pipe_resource_reference(&surf->base.texture, resource);
   surf->base.context = ctx;
   surf->base.format = templ->format;
   if (resource->target != PIPE_BUFFER) {
      surf->base.width = u_minify(resource->width0, templ->u.tex.level);
      surf->base.height = u_minify(resource->height0, templ->u.tex.level);
      surf->base.u.tex.level = templ->u.tex.level;
      surf->base.u.tex.first_layer = templ->u.tex.first_layer;
      surf->base.u.tex.last_layer = templ->u.tex.last_layer;
   } else {
      surf->base.width = templ->u.buf.last_element - templ->u.buf.first_element + 1;
      surf->base.height = resource->height0;
      surf->base.u.buf.first_element = templ->u.buf.first_element;
      surf->base.u.buf.last_element = templ->u.buf.last_element;
   }
   virgl_encoder_create_surface(vctx, handle, res, &surf->base);
   surf->handle = handle;
   return &surf->base;
}

static void virgl_surface_destroy(struct pipe_context *ctx,
                                 struct pipe_surface *psurf)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_surface *surf = virgl_surface(psurf);

   pipe_resource_reference(&surf->base.texture, NULL);
   virgl_encode_delete_object(vctx, surf->handle, VIRGL_OBJECT_SURFACE);
   FREE(surf);
}

static void *virgl_create_blend_state(struct pipe_context *ctx,
                                              const struct pipe_blend_state *blend_state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle;
   handle = virgl_object_assign_handle();

   virgl_encode_blend_state(vctx, handle, blend_state);
   return (void *)(unsigned long)handle;

}

static void virgl_bind_blend_state(struct pipe_context *ctx,
                                           void *blend_state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle = (unsigned long)blend_state;
   virgl_encode_bind_object(vctx, handle, VIRGL_OBJECT_BLEND);
}

static void virgl_delete_blend_state(struct pipe_context *ctx,
                                     void *blend_state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle = (unsigned long)blend_state;
   virgl_encode_delete_object(vctx, handle, VIRGL_OBJECT_BLEND);
}

static void *virgl_create_depth_stencil_alpha_state(struct pipe_context *ctx,
                                                   const struct pipe_depth_stencil_alpha_state *blend_state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle;
   handle = virgl_object_assign_handle();

   virgl_encode_dsa_state(vctx, handle, blend_state);
   return (void *)(unsigned long)handle;
}

static void virgl_bind_depth_stencil_alpha_state(struct pipe_context *ctx,
                                                void *blend_state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle = (unsigned long)blend_state;
   virgl_encode_bind_object(vctx, handle, VIRGL_OBJECT_DSA);
}

static void virgl_delete_depth_stencil_alpha_state(struct pipe_context *ctx,
                                                  void *dsa_state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle = (unsigned long)dsa_state;
   virgl_encode_delete_object(vctx, handle, VIRGL_OBJECT_DSA);
}

static void *virgl_create_rasterizer_state(struct pipe_context *ctx,
                                                   const struct pipe_rasterizer_state *rs_state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle;
   handle = virgl_object_assign_handle();

   virgl_encode_rasterizer_state(vctx, handle, rs_state);
   return (void *)(unsigned long)handle;
}

static void virgl_bind_rasterizer_state(struct pipe_context *ctx,
                                                void *rs_state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle = (unsigned long)rs_state;

   virgl_encode_bind_object(vctx, handle, VIRGL_OBJECT_RASTERIZER);
}

static void virgl_delete_rasterizer_state(struct pipe_context *ctx,
                                         void *rs_state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle = (unsigned long)rs_state;
   virgl_encode_delete_object(vctx, handle, VIRGL_OBJECT_RASTERIZER);
}

static void virgl_set_framebuffer_state(struct pipe_context *ctx,
                                                const struct pipe_framebuffer_state *state)
{
   struct virgl_context *vctx = virgl_context(ctx);

   vctx->framebuffer = *state;
   virgl_encoder_set_framebuffer_state(vctx, state);
   virgl_attach_res_framebuffer(vctx);
}

static void virgl_set_viewport_states(struct pipe_context *ctx,
                                     unsigned start_slot,
                                     unsigned num_viewports,
                                     const struct pipe_viewport_state *state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   virgl_encoder_set_viewport_states(vctx, start_slot, num_viewports, state);
}

static void *virgl_create_vertex_elements_state(struct pipe_context *ctx,
                                                        unsigned num_elements,
                                                        const struct pipe_vertex_element *elements)
{
   struct pipe_vertex_element new_elements[PIPE_MAX_ATTRIBS];
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_vertex_elements_state *state =
      CALLOC_STRUCT(virgl_vertex_elements_state);

   for (int i = 0; i < num_elements; ++i) {
      if (elements[i].instance_divisor) {
	 /* Virglrenderer doesn't deal with instance_divisor correctly if
	  * there isn't a 1:1 relationship between elements and bindings.
	  * So let's make sure there is, by duplicating bindings.
	  */
	 for (int j = 0; j < num_elements; ++j) {
            new_elements[j] = elements[j];
            new_elements[j].vertex_buffer_index = j;
            state->binding_map[j] = elements[j].vertex_buffer_index;
	 }
	 elements = new_elements;
	 state->num_bindings = num_elements;
	 break;
      }
   }

   state->handle = virgl_object_assign_handle();
   virgl_encoder_create_vertex_elements(vctx, state->handle,
                                       num_elements, elements);
   return state;
}

static void virgl_delete_vertex_elements_state(struct pipe_context *ctx,
                                              void *ve)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_vertex_elements_state *state =
      (struct virgl_vertex_elements_state *)ve;
   virgl_encode_delete_object(vctx, state->handle, VIRGL_OBJECT_VERTEX_ELEMENTS);
   FREE(state);
}

static void virgl_bind_vertex_elements_state(struct pipe_context *ctx,
                                                     void *ve)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_vertex_elements_state *state =
      (struct virgl_vertex_elements_state *)ve;
   vctx->vertex_elements = state;
   virgl_encode_bind_object(vctx, state ? state->handle : 0,
                            VIRGL_OBJECT_VERTEX_ELEMENTS);
   vctx->vertex_array_dirty = TRUE;
}

static void virgl_set_vertex_buffers(struct pipe_context *ctx,
                                    unsigned start_slot,
                                    unsigned num_buffers,
                                    const struct pipe_vertex_buffer *buffers)
{
   struct virgl_context *vctx = virgl_context(ctx);

   util_set_vertex_buffers_count(vctx->vertex_buffer,
                                 &vctx->num_vertex_buffers,
                                 buffers, start_slot, num_buffers);

   vctx->vertex_array_dirty = TRUE;
}

static void virgl_hw_set_vertex_buffers(struct pipe_context *ctx)
{
   struct virgl_context *vctx = virgl_context(ctx);

   if (vctx->vertex_array_dirty) {
      struct virgl_vertex_elements_state *ve = vctx->vertex_elements;

      if (ve->num_bindings) {
         struct pipe_vertex_buffer vertex_buffers[PIPE_MAX_ATTRIBS];
         for (int i = 0; i < ve->num_bindings; ++i)
            vertex_buffers[i] = vctx->vertex_buffer[ve->binding_map[i]];

         virgl_encoder_set_vertex_buffers(vctx, ve->num_bindings, vertex_buffers);
      } else
         virgl_encoder_set_vertex_buffers(vctx, vctx->num_vertex_buffers, vctx->vertex_buffer);

      virgl_attach_res_vertex_buffers(vctx);
   }
}

static void virgl_set_stencil_ref(struct pipe_context *ctx,
                                 const struct pipe_stencil_ref *ref)
{
   struct virgl_context *vctx = virgl_context(ctx);
   virgl_encoder_set_stencil_ref(vctx, ref);
}

static void virgl_set_blend_color(struct pipe_context *ctx,
                                 const struct pipe_blend_color *color)
{
   struct virgl_context *vctx = virgl_context(ctx);
   virgl_encoder_set_blend_color(vctx, color);
}

static void virgl_hw_set_index_buffer(struct pipe_context *ctx,
                                     struct virgl_indexbuf *ib)
{
   struct virgl_context *vctx = virgl_context(ctx);
   virgl_encoder_set_index_buffer(vctx, ib);
   virgl_attach_res_index_buffer(vctx, ib);
}

static void virgl_set_constant_buffer(struct pipe_context *ctx,
                                     enum pipe_shader_type shader, uint index,
                                     const struct pipe_constant_buffer *buf)
{
   struct virgl_context *vctx = virgl_context(ctx);

   if (buf) {
      if (!buf->user_buffer){
         struct virgl_resource *res = virgl_resource(buf->buffer);
         virgl_encoder_set_uniform_buffer(vctx, shader, index, buf->buffer_offset,
                                          buf->buffer_size, res);
         pipe_resource_reference(&vctx->ubos[shader][index], buf->buffer);
         return;
      }
      pipe_resource_reference(&vctx->ubos[shader][index], NULL);
      virgl_encoder_write_constant_buffer(vctx, shader, index, buf->buffer_size / 4, buf->user_buffer);
   } else {
      virgl_encoder_write_constant_buffer(vctx, shader, index, 0, NULL);
      pipe_resource_reference(&vctx->ubos[shader][index], NULL);
   }
}

void virgl_transfer_inline_write(struct pipe_context *ctx,
                                struct pipe_resource *res,
                                unsigned level,
                                unsigned usage,
                                const struct pipe_box *box,
                                const void *data,
                                unsigned stride,
                                unsigned layer_stride)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *vs = virgl_screen(ctx->screen);
   struct virgl_resource *grres = virgl_resource(res);
   struct virgl_buffer *vbuf = virgl_buffer(res);

   grres->clean = FALSE;

   if (virgl_res_needs_flush_wait(vctx, &vbuf->base, usage)) {
      ctx->flush(ctx, NULL, 0);

      vs->vws->resource_wait(vs->vws, vbuf->base.hw_res);
   }

   virgl_encoder_inline_write(vctx, grres, level, usage,
                              box, data, stride, layer_stride);
}

static void *virgl_shader_encoder(struct pipe_context *ctx,
                                  const struct pipe_shader_state *shader,
                                  unsigned type)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle;
   struct tgsi_token *new_tokens;
   int ret;

   new_tokens = virgl_tgsi_transform(vctx, shader->tokens);
   if (!new_tokens)
      return NULL;

   handle = virgl_object_assign_handle();
   /* encode VS state */
   ret = virgl_encode_shader_state(vctx, handle, type,
                                   &shader->stream_output, 0,
                                   new_tokens);
   if (ret) {
      return NULL;
   }

   FREE(new_tokens);
   return (void *)(unsigned long)handle;

}
static void *virgl_create_vs_state(struct pipe_context *ctx,
                                   const struct pipe_shader_state *shader)
{
   return virgl_shader_encoder(ctx, shader, PIPE_SHADER_VERTEX);
}

static void *virgl_create_tcs_state(struct pipe_context *ctx,
                                   const struct pipe_shader_state *shader)
{
   return virgl_shader_encoder(ctx, shader, PIPE_SHADER_TESS_CTRL);
}

static void *virgl_create_tes_state(struct pipe_context *ctx,
                                   const struct pipe_shader_state *shader)
{
   return virgl_shader_encoder(ctx, shader, PIPE_SHADER_TESS_EVAL);
}

static void *virgl_create_gs_state(struct pipe_context *ctx,
                                   const struct pipe_shader_state *shader)
{
   return virgl_shader_encoder(ctx, shader, PIPE_SHADER_GEOMETRY);
}

static void *virgl_create_fs_state(struct pipe_context *ctx,
                                   const struct pipe_shader_state *shader)
{
   return virgl_shader_encoder(ctx, shader, PIPE_SHADER_FRAGMENT);
}

static void
virgl_delete_fs_state(struct pipe_context *ctx,
                     void *fs)
{
   uint32_t handle = (unsigned long)fs;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_delete_object(vctx, handle, VIRGL_OBJECT_SHADER);
}

static void
virgl_delete_gs_state(struct pipe_context *ctx,
                     void *gs)
{
   uint32_t handle = (unsigned long)gs;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_delete_object(vctx, handle, VIRGL_OBJECT_SHADER);
}

static void
virgl_delete_vs_state(struct pipe_context *ctx,
                     void *vs)
{
   uint32_t handle = (unsigned long)vs;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_delete_object(vctx, handle, VIRGL_OBJECT_SHADER);
}

static void
virgl_delete_tcs_state(struct pipe_context *ctx,
                       void *tcs)
{
   uint32_t handle = (unsigned long)tcs;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_delete_object(vctx, handle, VIRGL_OBJECT_SHADER);
}

static void
virgl_delete_tes_state(struct pipe_context *ctx,
                      void *tes)
{
   uint32_t handle = (unsigned long)tes;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_delete_object(vctx, handle, VIRGL_OBJECT_SHADER);
}

static void virgl_bind_vs_state(struct pipe_context *ctx,
                                        void *vss)
{
   uint32_t handle = (unsigned long)vss;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_bind_shader(vctx, handle, PIPE_SHADER_VERTEX);
}

static void virgl_bind_tcs_state(struct pipe_context *ctx,
                               void *vss)
{
   uint32_t handle = (unsigned long)vss;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_bind_shader(vctx, handle, PIPE_SHADER_TESS_CTRL);
}

static void virgl_bind_tes_state(struct pipe_context *ctx,
                               void *vss)
{
   uint32_t handle = (unsigned long)vss;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_bind_shader(vctx, handle, PIPE_SHADER_TESS_EVAL);
}

static void virgl_bind_gs_state(struct pipe_context *ctx,
                               void *vss)
{
   uint32_t handle = (unsigned long)vss;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_bind_shader(vctx, handle, PIPE_SHADER_GEOMETRY);
}


static void virgl_bind_fs_state(struct pipe_context *ctx,
                                        void *vss)
{
   uint32_t handle = (unsigned long)vss;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_bind_shader(vctx, handle, PIPE_SHADER_FRAGMENT);
}

static void virgl_clear(struct pipe_context *ctx,
                                unsigned buffers,
                                const union pipe_color_union *color,
                                double depth, unsigned stencil)
{
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_clear(vctx, buffers, color, depth, stencil);
}

static void virgl_draw_vbo(struct pipe_context *ctx,
                                   const struct pipe_draw_info *dinfo)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *rs = virgl_screen(ctx->screen);
   struct virgl_indexbuf ib = {};
   struct pipe_draw_info info = *dinfo;

   if (!dinfo->count_from_stream_output && !dinfo->indirect &&
       !dinfo->primitive_restart &&
       !u_trim_pipe_prim(dinfo->mode, (unsigned*)&dinfo->count))
      return;

   if (!(rs->caps.caps.v1.prim_mask & (1 << dinfo->mode))) {
      util_primconvert_draw_vbo(vctx->primconvert, dinfo);
      return;
   }
   if (info.index_size) {
           pipe_resource_reference(&ib.buffer, info.has_user_indices ? NULL : info.index.resource);
           ib.user_buffer = info.has_user_indices ? info.index.user : NULL;
           ib.index_size = dinfo->index_size;
           ib.offset = info.start * ib.index_size;

           if (ib.user_buffer) {
                   u_upload_data(vctx->uploader, 0, info.count * ib.index_size, 256,
                                 ib.user_buffer, &ib.offset, &ib.buffer);
                   ib.user_buffer = NULL;
           }
   }

   u_upload_unmap(vctx->uploader);

   vctx->num_draws++;
   virgl_hw_set_vertex_buffers(ctx);
   if (info.index_size)
      virgl_hw_set_index_buffer(ctx, &ib);

   virgl_encoder_draw_vbo(vctx, &info);

   pipe_resource_reference(&ib.buffer, NULL);

}

static void virgl_flush_eq(struct virgl_context *ctx, void *closure)
{
   struct virgl_screen *rs = virgl_screen(ctx->base.screen);

   /* send the buffer to the remote side for decoding */
   ctx->num_transfers = ctx->num_draws = 0;
   rs->vws->submit_cmd(rs->vws, ctx->cbuf);

   virgl_encoder_set_sub_ctx(ctx, ctx->hw_sub_ctx_id);

   /* add back current framebuffer resources to reference list? */
   virgl_reemit_res(ctx);
}

static void virgl_flush_from_st(struct pipe_context *ctx,
                               struct pipe_fence_handle **fence,
                               enum pipe_flush_flags flags)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *rs = virgl_screen(ctx->screen);
   struct virgl_buffer *buf, *tmp;

   if (fence)
      *fence = rs->vws->cs_create_fence(rs->vws);

   LIST_FOR_EACH_ENTRY_SAFE(buf, tmp, &vctx->to_flush_bufs, flush_list) {
      struct pipe_resource *res = &buf->base.u.b;
      virgl_buffer_flush(vctx, buf);
      list_del(&buf->flush_list);
      buf->on_list = FALSE;
      pipe_resource_reference(&res, NULL);

   }
   virgl_flush_eq(vctx, vctx);
}

static struct pipe_sampler_view *virgl_create_sampler_view(struct pipe_context *ctx,
                                      struct pipe_resource *texture,
                                      const struct pipe_sampler_view *state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_sampler_view *grview;
   uint32_t handle;
   struct virgl_resource *res;

   if (!state)
      return NULL;

   grview = CALLOC_STRUCT(virgl_sampler_view);
   if (!grview)
      return NULL;

   res = virgl_resource(texture);
   handle = virgl_object_assign_handle();
   virgl_encode_sampler_view(vctx, handle, res, state);

   grview->base = *state;
   grview->base.reference.count = 1;

   grview->base.texture = NULL;
   grview->base.context = ctx;
   pipe_resource_reference(&grview->base.texture, texture);
   grview->handle = handle;
   return &grview->base;
}

static void virgl_set_sampler_views(struct pipe_context *ctx,
                                   enum pipe_shader_type shader_type,
                                   unsigned start_slot,
                                   unsigned num_views,
                                   struct pipe_sampler_view **views)
{
   struct virgl_context *vctx = virgl_context(ctx);
   int i;
   uint32_t disable_mask = ~((1ull << num_views) - 1);
   struct virgl_textures_info *tinfo = &vctx->samplers[shader_type];
   uint32_t new_mask = 0;
   uint32_t remaining_mask;

   remaining_mask = tinfo->enabled_mask & disable_mask;

   while (remaining_mask) {
      i = u_bit_scan(&remaining_mask);
      assert(tinfo->views[i]);

      pipe_sampler_view_reference((struct pipe_sampler_view **)&tinfo->views[i], NULL);
   }

   for (i = 0; i < num_views; i++) {
      struct virgl_sampler_view *grview = virgl_sampler_view(views[i]);

      if (views[i] == (struct pipe_sampler_view *)tinfo->views[i])
         continue;

      if (grview) {
         new_mask |= 1 << i;
         pipe_sampler_view_reference((struct pipe_sampler_view **)&tinfo->views[i], views[i]);
      } else {
         pipe_sampler_view_reference((struct pipe_sampler_view **)&tinfo->views[i], NULL);
         disable_mask |= 1 << i;
      }
   }

   tinfo->enabled_mask &= ~disable_mask;
   tinfo->enabled_mask |= new_mask;
   virgl_encode_set_sampler_views(vctx, shader_type, start_slot, num_views, tinfo->views);
   virgl_attach_res_sampler_views(vctx, shader_type);
}

static void
virgl_texture_barrier(struct pipe_context *ctx, unsigned flags)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *rs = virgl_screen(ctx->screen);

   if (!(rs->caps.caps.v2.capability_bits & VIRGL_CAP_TEXTURE_BARRIER))
      return;
   virgl_encode_texture_barrier(vctx, flags);
}

static void virgl_destroy_sampler_view(struct pipe_context *ctx,
                                 struct pipe_sampler_view *view)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_sampler_view *grview = virgl_sampler_view(view);

   virgl_encode_delete_object(vctx, grview->handle, VIRGL_OBJECT_SAMPLER_VIEW);
   pipe_resource_reference(&view->texture, NULL);
   FREE(view);
}

static void *virgl_create_sampler_state(struct pipe_context *ctx,
                                        const struct pipe_sampler_state *state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle;

   handle = virgl_object_assign_handle();

   virgl_encode_sampler_state(vctx, handle, state);
   return (void *)(unsigned long)handle;
}

static void virgl_delete_sampler_state(struct pipe_context *ctx,
                                      void *ss)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle = (unsigned long)ss;

   virgl_encode_delete_object(vctx, handle, VIRGL_OBJECT_SAMPLER_STATE);
}

static void virgl_bind_sampler_states(struct pipe_context *ctx,
                                     enum pipe_shader_type shader,
                                     unsigned start_slot,
                                     unsigned num_samplers,
                                     void **samplers)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handles[32];
   int i;
   for (i = 0; i < num_samplers; i++) {
      handles[i] = (unsigned long)(samplers[i]);
   }
   virgl_encode_bind_sampler_states(vctx, shader, start_slot, num_samplers, handles);
}

static void virgl_set_polygon_stipple(struct pipe_context *ctx,
                                     const struct pipe_poly_stipple *ps)
{
   struct virgl_context *vctx = virgl_context(ctx);
   virgl_encoder_set_polygon_stipple(vctx, ps);
}

static void virgl_set_scissor_states(struct pipe_context *ctx,
                                    unsigned start_slot,
                                    unsigned num_scissor,
                                   const struct pipe_scissor_state *ss)
{
   struct virgl_context *vctx = virgl_context(ctx);
   virgl_encoder_set_scissor_state(vctx, start_slot, num_scissor, ss);
}

static void virgl_set_sample_mask(struct pipe_context *ctx,
                                 unsigned sample_mask)
{
   struct virgl_context *vctx = virgl_context(ctx);
   virgl_encoder_set_sample_mask(vctx, sample_mask);
}

static void virgl_set_min_samples(struct pipe_context *ctx,
                                 unsigned min_samples)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *rs = virgl_screen(ctx->screen);

   if (!(rs->caps.caps.v2.capability_bits & VIRGL_CAP_SET_MIN_SAMPLES))
      return;
   virgl_encoder_set_min_samples(vctx, min_samples);
}

static void virgl_set_clip_state(struct pipe_context *ctx,
                                const struct pipe_clip_state *clip)
{
   struct virgl_context *vctx = virgl_context(ctx);
   virgl_encoder_set_clip_state(vctx, clip);
}

static void virgl_set_tess_state(struct pipe_context *ctx,
                                 const float default_outer_level[4],
                                 const float default_inner_level[2])
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *rs = virgl_screen(ctx->screen);

   if (!rs->caps.caps.v1.bset.has_tessellation_shaders)
      return;
   virgl_encode_set_tess_state(vctx, default_outer_level, default_inner_level);
}

static void virgl_resource_copy_region(struct pipe_context *ctx,
                                      struct pipe_resource *dst,
                                      unsigned dst_level,
                                      unsigned dstx, unsigned dsty, unsigned dstz,
                                      struct pipe_resource *src,
                                      unsigned src_level,
                                      const struct pipe_box *src_box)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_resource *dres = virgl_resource(dst);
   struct virgl_resource *sres = virgl_resource(src);

   dres->clean = FALSE;
   virgl_encode_resource_copy_region(vctx, dres,
                                    dst_level, dstx, dsty, dstz,
                                    sres, src_level,
                                    src_box);
}

static void
virgl_flush_resource(struct pipe_context *pipe,
                    struct pipe_resource *resource)
{
}

static void virgl_blit(struct pipe_context *ctx,
                      const struct pipe_blit_info *blit)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_resource *dres = virgl_resource(blit->dst.resource);
   struct virgl_resource *sres = virgl_resource(blit->src.resource);

   dres->clean = FALSE;
   virgl_encode_blit(vctx, dres, sres,
                    blit);
}

static void virgl_set_hw_atomic_buffers(struct pipe_context *ctx,
                                        unsigned start_slot,
                                        unsigned count,
                                        const struct pipe_shader_buffer *buffers)
{
   struct virgl_context *vctx = virgl_context(ctx);

   for (unsigned i = 0; i < count; i++) {
      unsigned idx = start_slot + i;

      if (buffers) {
         if (buffers[i].buffer) {
            pipe_resource_reference(&vctx->atomic_buffers[idx],
                                    buffers[i].buffer);
            continue;
         }
      }
      pipe_resource_reference(&vctx->atomic_buffers[idx], NULL);
   }
   virgl_encode_set_hw_atomic_buffers(vctx, start_slot, count, buffers);
}

static void virgl_set_shader_buffers(struct pipe_context *ctx,
                                     enum pipe_shader_type shader,
                                     unsigned start_slot, unsigned count,
                                     const struct pipe_shader_buffer *buffers)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *rs = virgl_screen(ctx->screen);

   for (unsigned i = 0; i < count; i++) {
      unsigned idx = start_slot + i;

      if (buffers) {
         if (buffers[i].buffer) {
            pipe_resource_reference(&vctx->ssbos[shader][idx], buffers[i].buffer);
            continue;
         }
      }
      pipe_resource_reference(&vctx->ssbos[shader][idx], NULL);
   }

   uint32_t max_shader_buffer = (shader == PIPE_SHADER_FRAGMENT || shader == PIPE_SHADER_COMPUTE) ?
      rs->caps.caps.v2.max_shader_buffer_frag_compute :
      rs->caps.caps.v2.max_shader_buffer_other_stages;
   if (!max_shader_buffer)
      return;
   virgl_encode_set_shader_buffers(vctx, shader, start_slot, count, buffers);
}

static void virgl_set_shader_images(struct pipe_context *ctx,
                                    enum pipe_shader_type shader,
                                    unsigned start_slot, unsigned count,
                                    const struct pipe_image_view *images)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *rs = virgl_screen(ctx->screen);

   for (unsigned i = 0; i < count; i++) {
      unsigned idx = start_slot + i;

      if (images) {
         if (images[i].resource) {
            pipe_resource_reference(&vctx->images[shader][idx], images[i].resource);
            continue;
         }
      }
      pipe_resource_reference(&vctx->images[shader][idx], NULL);
   }

   uint32_t max_shader_images = (shader == PIPE_SHADER_FRAGMENT || shader == PIPE_SHADER_COMPUTE) ?
     rs->caps.caps.v2.max_shader_image_frag_compute :
     rs->caps.caps.v2.max_shader_image_other_stages;
   if (!max_shader_images)
      return;
   virgl_encode_set_shader_images(vctx, shader, start_slot, count, images);
}

static void virgl_memory_barrier(struct pipe_context *ctx,
                                 unsigned flags)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *rs = virgl_screen(ctx->screen);

   if (!(rs->caps.caps.v2.capability_bits & VIRGL_CAP_MEMORY_BARRIER))
      return;
   virgl_encode_memory_barrier(vctx, flags);
}

static void *virgl_create_compute_state(struct pipe_context *ctx,
                                        const struct pipe_compute_state *state)
{
   struct virgl_context *vctx = virgl_context(ctx);
   uint32_t handle;
   const struct tgsi_token *new_tokens = state->prog;
   struct pipe_stream_output_info so_info = {};
   int ret;

   handle = virgl_object_assign_handle();
   ret = virgl_encode_shader_state(vctx, handle, PIPE_SHADER_COMPUTE,
                                   &so_info,
                                   state->req_local_mem,
                                   new_tokens);
   if (ret) {
      return NULL;
   }

   return (void *)(unsigned long)handle;
}

static void virgl_bind_compute_state(struct pipe_context *ctx, void *state)
{
   uint32_t handle = (unsigned long)state;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_bind_shader(vctx, handle, PIPE_SHADER_COMPUTE);
}

static void virgl_delete_compute_state(struct pipe_context *ctx, void *state)
{
   uint32_t handle = (unsigned long)state;
   struct virgl_context *vctx = virgl_context(ctx);

   virgl_encode_delete_object(vctx, handle, VIRGL_OBJECT_SHADER);
}

static void virgl_launch_grid(struct pipe_context *ctx,
                              const struct pipe_grid_info *info)
{
   struct virgl_context *vctx = virgl_context(ctx);
   virgl_encode_launch_grid(vctx, info);
}

static void
virgl_context_destroy( struct pipe_context *ctx )
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *rs = virgl_screen(ctx->screen);

   vctx->framebuffer.zsbuf = NULL;
   vctx->framebuffer.nr_cbufs = 0;
   virgl_encoder_destroy_sub_ctx(vctx, vctx->hw_sub_ctx_id);
   virgl_flush_eq(vctx, vctx);

   rs->vws->cmd_buf_destroy(vctx->cbuf);
   if (vctx->uploader)
      u_upload_destroy(vctx->uploader);
   util_primconvert_destroy(vctx->primconvert);

   slab_destroy_child(&vctx->texture_transfer_pool);
   FREE(vctx);
}

static void virgl_get_sample_position(struct pipe_context *ctx,
				      unsigned sample_count,
				      unsigned index,
				      float *out_value)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *vs = virgl_screen(vctx->base.screen);

   if (sample_count > vs->caps.caps.v1.max_samples) {
      debug_printf("VIRGL: requested %d MSAA samples, but only %d supported\n",
		   sample_count, vs->caps.caps.v1.max_samples);
      return;
   }

   /* The following is basically copied from dri/i965gen6_get_sample_position
    * The only addition is that we hold the msaa positions for all sample
    * counts in a flat array. */
   uint32_t bits = 0;
   if (sample_count == 1) {
      out_value[0] = out_value[1] = 0.5f;
      return;
   } else if (sample_count == 2) {
      bits = vs->caps.caps.v2.sample_locations[0] >> (8 * index);
   } else if (sample_count <= 4) {
      bits = vs->caps.caps.v2.sample_locations[1] >> (8 * index);
   } else if (sample_count <= 8) {
      bits = vs->caps.caps.v2.sample_locations[2 + (index >> 2)] >> (8 * (index & 3));
   } else if (sample_count <= 16) {
      bits = vs->caps.caps.v2.sample_locations[4 + (index >> 2)] >> (8 * (index & 3));
   }
   out_value[0] = ((bits >> 4) & 0xf) / 16.0f;
   out_value[1] = (bits & 0xf) / 16.0f;

   if (virgl_debug & VIRGL_DEBUG_VERBOSE)
      debug_printf("VIRGL: sample postion [%2d/%2d] = (%f, %f)\n",
                   index, sample_count, out_value[0], out_value[1]);
}

struct pipe_context *virgl_context_create(struct pipe_screen *pscreen,
                                          void *priv,
                                          unsigned flags)
{
   struct virgl_context *vctx;
   struct virgl_screen *rs = virgl_screen(pscreen);
   vctx = CALLOC_STRUCT(virgl_context);

   vctx->cbuf = rs->vws->cmd_buf_create(rs->vws);
   if (!vctx->cbuf) {
      FREE(vctx);
      return NULL;
   }

   vctx->base.destroy = virgl_context_destroy;
   vctx->base.create_surface = virgl_create_surface;
   vctx->base.surface_destroy = virgl_surface_destroy;
   vctx->base.set_framebuffer_state = virgl_set_framebuffer_state;
   vctx->base.create_blend_state = virgl_create_blend_state;
   vctx->base.bind_blend_state = virgl_bind_blend_state;
   vctx->base.delete_blend_state = virgl_delete_blend_state;
   vctx->base.create_depth_stencil_alpha_state = virgl_create_depth_stencil_alpha_state;
   vctx->base.bind_depth_stencil_alpha_state = virgl_bind_depth_stencil_alpha_state;
   vctx->base.delete_depth_stencil_alpha_state = virgl_delete_depth_stencil_alpha_state;
   vctx->base.create_rasterizer_state = virgl_create_rasterizer_state;
   vctx->base.bind_rasterizer_state = virgl_bind_rasterizer_state;
   vctx->base.delete_rasterizer_state = virgl_delete_rasterizer_state;

   vctx->base.set_viewport_states = virgl_set_viewport_states;
   vctx->base.create_vertex_elements_state = virgl_create_vertex_elements_state;
   vctx->base.bind_vertex_elements_state = virgl_bind_vertex_elements_state;
   vctx->base.delete_vertex_elements_state = virgl_delete_vertex_elements_state;
   vctx->base.set_vertex_buffers = virgl_set_vertex_buffers;
   vctx->base.set_constant_buffer = virgl_set_constant_buffer;

   vctx->base.set_tess_state = virgl_set_tess_state;
   vctx->base.create_vs_state = virgl_create_vs_state;
   vctx->base.create_tcs_state = virgl_create_tcs_state;
   vctx->base.create_tes_state = virgl_create_tes_state;
   vctx->base.create_gs_state = virgl_create_gs_state;
   vctx->base.create_fs_state = virgl_create_fs_state;

   vctx->base.bind_vs_state = virgl_bind_vs_state;
   vctx->base.bind_tcs_state = virgl_bind_tcs_state;
   vctx->base.bind_tes_state = virgl_bind_tes_state;
   vctx->base.bind_gs_state = virgl_bind_gs_state;
   vctx->base.bind_fs_state = virgl_bind_fs_state;

   vctx->base.delete_vs_state = virgl_delete_vs_state;
   vctx->base.delete_tcs_state = virgl_delete_tcs_state;
   vctx->base.delete_tes_state = virgl_delete_tes_state;
   vctx->base.delete_gs_state = virgl_delete_gs_state;
   vctx->base.delete_fs_state = virgl_delete_fs_state;

   vctx->base.create_compute_state = virgl_create_compute_state;
   vctx->base.bind_compute_state = virgl_bind_compute_state;
   vctx->base.delete_compute_state = virgl_delete_compute_state;
   vctx->base.launch_grid = virgl_launch_grid;

   vctx->base.clear = virgl_clear;
   vctx->base.draw_vbo = virgl_draw_vbo;
   vctx->base.flush = virgl_flush_from_st;
   vctx->base.screen = pscreen;
   vctx->base.create_sampler_view = virgl_create_sampler_view;
   vctx->base.sampler_view_destroy = virgl_destroy_sampler_view;
   vctx->base.set_sampler_views = virgl_set_sampler_views;
   vctx->base.texture_barrier = virgl_texture_barrier;

   vctx->base.create_sampler_state = virgl_create_sampler_state;
   vctx->base.delete_sampler_state = virgl_delete_sampler_state;
   vctx->base.bind_sampler_states = virgl_bind_sampler_states;

   vctx->base.set_polygon_stipple = virgl_set_polygon_stipple;
   vctx->base.set_scissor_states = virgl_set_scissor_states;
   vctx->base.set_sample_mask = virgl_set_sample_mask;
   vctx->base.set_min_samples = virgl_set_min_samples;
   vctx->base.set_stencil_ref = virgl_set_stencil_ref;
   vctx->base.set_clip_state = virgl_set_clip_state;

   vctx->base.set_blend_color = virgl_set_blend_color;

   vctx->base.get_sample_position = virgl_get_sample_position;

   vctx->base.resource_copy_region = virgl_resource_copy_region;
   vctx->base.flush_resource = virgl_flush_resource;
   vctx->base.blit =  virgl_blit;

   vctx->base.set_shader_buffers = virgl_set_shader_buffers;
   vctx->base.set_hw_atomic_buffers = virgl_set_hw_atomic_buffers;
   vctx->base.set_shader_images = virgl_set_shader_images;
   vctx->base.memory_barrier = virgl_memory_barrier;

   virgl_init_context_resource_functions(&vctx->base);
   virgl_init_query_functions(vctx);
   virgl_init_so_functions(vctx);

   list_inithead(&vctx->to_flush_bufs);
   slab_create_child(&vctx->texture_transfer_pool, &rs->texture_transfer_pool);

   vctx->primconvert = util_primconvert_create(&vctx->base, rs->caps.caps.v1.prim_mask);
   vctx->uploader = u_upload_create(&vctx->base, 1024 * 1024,
                                     PIPE_BIND_INDEX_BUFFER, PIPE_USAGE_STREAM, 0);
   if (!vctx->uploader)
           goto fail;
   vctx->base.stream_uploader = vctx->uploader;
   vctx->base.const_uploader = vctx->uploader;

   vctx->hw_sub_ctx_id = rs->sub_ctx_id++;
   virgl_encoder_create_sub_ctx(vctx, vctx->hw_sub_ctx_id);

   virgl_encoder_set_sub_ctx(vctx, vctx->hw_sub_ctx_id);
   return &vctx->base;
fail:
   return NULL;
}
