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

#include "etnaviv_context.h"

#include "etnaviv_blend.h"
#include "etnaviv_clear_blit.h"
#include "etnaviv_compiler.h"
#include "etnaviv_debug.h"
#include "etnaviv_emit.h"
#include "etnaviv_fence.h"
#include "etnaviv_query.h"
#include "etnaviv_query_hw.h"
#include "etnaviv_rasterizer.h"
#include "etnaviv_screen.h"
#include "etnaviv_shader.h"
#include "etnaviv_state.h"
#include "etnaviv_surface.h"
#include "etnaviv_texture.h"
#include "etnaviv_transfer.h"
#include "etnaviv_translate.h"
#include "etnaviv_zsa.h"

#include "pipe/p_context.h"
#include "pipe/p_state.h"
#include "util/u_blitter.h"
#include "util/u_helpers.h"
#include "util/u_memory.h"
#include "util/u_prim.h"
#include "util/u_upload_mgr.h"

#include "hw/common.xml.h"

static void
etna_context_destroy(struct pipe_context *pctx)
{
   struct etna_context *ctx = etna_context(pctx);

   util_copy_framebuffer_state(&ctx->framebuffer_s, NULL);

   if (ctx->primconvert)
      util_primconvert_destroy(ctx->primconvert);

   if (ctx->blitter)
      util_blitter_destroy(ctx->blitter);

   if (pctx->stream_uploader)
      u_upload_destroy(pctx->stream_uploader);

   if (ctx->stream)
      etna_cmd_stream_del(ctx->stream);

   slab_destroy_child(&ctx->transfer_pool);

   if (ctx->in_fence_fd != -1)
      close(ctx->in_fence_fd);

   FREE(pctx);
}

/* Update render state where needed based on draw operation */
static void
etna_update_state_for_draw(struct etna_context *ctx, const struct pipe_draw_info *info)
{
   /* Handle primitive restart:
    * - If not an indexed draw, we don't care about the state of the primitive restart bit.
    * - Otherwise, set the bit in INDEX_STREAM_CONTROL in the index buffer state
    *   accordingly
    * - If the value of the INDEX_STREAM_CONTROL register changed due to this, or
    *   primitive restart is enabled and the restart index changed, mark the index
    *   buffer state as dirty
    */

   if (info->index_size) {
      uint32_t new_control = ctx->index_buffer.FE_INDEX_STREAM_CONTROL;

      if (info->primitive_restart)
         new_control |= VIVS_FE_INDEX_STREAM_CONTROL_PRIMITIVE_RESTART;
      else
         new_control &= ~VIVS_FE_INDEX_STREAM_CONTROL_PRIMITIVE_RESTART;

      if (ctx->index_buffer.FE_INDEX_STREAM_CONTROL != new_control ||
          (info->primitive_restart && ctx->index_buffer.FE_PRIMITIVE_RESTART_INDEX != info->restart_index)) {
         ctx->index_buffer.FE_INDEX_STREAM_CONTROL = new_control;
         ctx->index_buffer.FE_PRIMITIVE_RESTART_INDEX = info->restart_index;
         ctx->dirty |= ETNA_DIRTY_INDEX_BUFFER;
      }
   }
}

static bool
etna_get_vs(struct etna_context *ctx, struct etna_shader_key key)
{
   const struct etna_shader_variant *old = ctx->shader.vs;

   ctx->shader.vs = etna_shader_variant(ctx->shader.bind_vs, key, &ctx->debug);

   if (!ctx->shader.vs)
      return false;

   if (old != ctx->shader.vs)
      ctx->dirty |= ETNA_DIRTY_SHADER;

   return true;
}

static bool
etna_get_fs(struct etna_context *ctx, struct etna_shader_key key)
{
   const struct etna_shader_variant *old = ctx->shader.fs;

   ctx->shader.fs = etna_shader_variant(ctx->shader.bind_fs, key, &ctx->debug);

   if (!ctx->shader.fs)
      return false;

   if (old != ctx->shader.fs)
      ctx->dirty |= ETNA_DIRTY_SHADER;

   return true;
}

static void
etna_draw_vbo(struct pipe_context *pctx, const struct pipe_draw_info *info)
{
   struct etna_context *ctx = etna_context(pctx);
   struct pipe_framebuffer_state *pfb = &ctx->framebuffer_s;
   uint32_t draw_mode;
   unsigned i;

   if (!info->count_from_stream_output && !info->indirect &&
       !info->primitive_restart &&
       !u_trim_pipe_prim(info->mode, (unsigned*)&info->count))
      return;

   if (ctx->vertex_elements == NULL || ctx->vertex_elements->num_elements == 0)
      return; /* Nothing to do */

   if (!(ctx->prim_hwsupport & (1 << info->mode))) {
      struct primconvert_context *primconvert = ctx->primconvert;
      util_primconvert_save_rasterizer_state(primconvert, ctx->rasterizer);
      util_primconvert_draw_vbo(primconvert, info);
      return;
   }

   int prims = u_decomposed_prims_for_vertices(info->mode, info->count);
   if (unlikely(prims <= 0)) {
      DBG("Invalid draw primitive mode=%i or no primitives to be drawn", info->mode);
      return;
   }

   draw_mode = translate_draw_mode(info->mode);
   if (draw_mode == ETNA_NO_MATCH) {
      BUG("Unsupported draw mode");
      return;
   }

   /* Upload a user index buffer. */
   unsigned index_offset = 0;
   struct pipe_resource *indexbuf = NULL;

   if (info->index_size) {
      indexbuf = info->has_user_indices ? NULL : info->index.resource;
      if (info->has_user_indices &&
          !util_upload_index_buffer(pctx, info, &indexbuf, &index_offset)) {
         BUG("Index buffer upload failed.");
         return;
      }
      /* Add start to index offset, when rendering indexed */
      index_offset += info->start * info->index_size;

      ctx->index_buffer.FE_INDEX_STREAM_BASE_ADDR.bo = etna_resource(indexbuf)->bo;
      ctx->index_buffer.FE_INDEX_STREAM_BASE_ADDR.offset = index_offset;
      ctx->index_buffer.FE_INDEX_STREAM_BASE_ADDR.flags = ETNA_RELOC_READ;
      ctx->index_buffer.FE_INDEX_STREAM_CONTROL = translate_index_size(info->index_size);

      if (!ctx->index_buffer.FE_INDEX_STREAM_BASE_ADDR.bo) {
         BUG("Unsupported or no index buffer");
         return;
      }
   } else {
      ctx->index_buffer.FE_INDEX_STREAM_BASE_ADDR.bo = 0;
      ctx->index_buffer.FE_INDEX_STREAM_BASE_ADDR.offset = 0;
      ctx->index_buffer.FE_INDEX_STREAM_BASE_ADDR.flags = 0;
      ctx->index_buffer.FE_INDEX_STREAM_CONTROL = 0;
   }
   ctx->dirty |= ETNA_DIRTY_INDEX_BUFFER;

   struct etna_shader_key key = {};
   struct etna_surface *cbuf = etna_surface(pfb->cbufs[0]);

   if (cbuf) {
      struct etna_resource *res = etna_resource(cbuf->base.texture);

      key.frag_rb_swap = !!translate_rs_format_rb_swap(res->base.format);
   }

   if (!etna_get_vs(ctx, key) || !etna_get_fs(ctx, key)) {
      BUG("compiled shaders are not okay");
      return;
   }

   /* Update any derived state */
   if (!etna_state_update(ctx))
      return;

   /*
    * Figure out the buffers/features we need:
    */
   if (etna_depth_enabled(ctx))
      resource_written(ctx, pfb->zsbuf->texture);

   if (etna_stencil_enabled(ctx))
      resource_written(ctx, pfb->zsbuf->texture);

   for (i = 0; i < pfb->nr_cbufs; i++) {
      struct pipe_resource *surf;

      if (!pfb->cbufs[i])
         continue;

      surf = pfb->cbufs[i]->texture;
      resource_written(ctx, surf);
   }

   /* Mark constant buffers as being read */
   resource_read(ctx, ctx->constant_buffer[PIPE_SHADER_VERTEX].buffer);
   resource_read(ctx, ctx->constant_buffer[PIPE_SHADER_FRAGMENT].buffer);

   /* Mark VBOs as being read */
   for (i = 0; i < ctx->vertex_buffer.count; i++) {
      assert(!ctx->vertex_buffer.vb[i].is_user_buffer);
      resource_read(ctx, ctx->vertex_buffer.vb[i].buffer.resource);
   }

   /* Mark index buffer as being read */
   resource_read(ctx, indexbuf);

   /* Mark textures as being read */
   for (i = 0; i < PIPE_MAX_SAMPLERS; i++)
      if (ctx->sampler_view[i])
         resource_read(ctx, ctx->sampler_view[i]->texture);

   list_for_each_entry(struct etna_hw_query, hq, &ctx->active_hw_queries, node)
      resource_written(ctx, hq->prsc);

   ctx->stats.prims_emitted += u_reduced_prims_for_vertices(info->mode, info->count);
   ctx->stats.draw_calls++;

   /* Update state for this draw operation */
   etna_update_state_for_draw(ctx, info);

   /* First, sync state, then emit DRAW_PRIMITIVES or DRAW_INDEXED_PRIMITIVES */
   etna_emit_state(ctx);

   if (ctx->specs.halti >= 2) {
      /* On HALTI2+ (GC3000 and higher) only use instanced drawing commands, as the blob does */
      etna_draw_instanced(ctx->stream, info->index_size, draw_mode, 1,
         info->count, info->index_size ? info->index_bias : info->start);
   } else {
      if (info->index_size)
         etna_draw_indexed_primitives(ctx->stream, draw_mode, 0, prims, info->index_bias);
      else
         etna_draw_primitives(ctx->stream, draw_mode, info->start, prims);
   }

   if (DBG_ENABLED(ETNA_DBG_DRAW_STALL)) {
      /* Stall the FE after every draw operation.  This allows better
       * debug of GPU hang conditions, as the FE will indicate which
       * draw op has caused the hang. */
      etna_stall(ctx->stream, SYNC_RECIPIENT_FE, SYNC_RECIPIENT_PE);
   }

   if (DBG_ENABLED(ETNA_DBG_FLUSH_ALL))
      pctx->flush(pctx, NULL, 0);

   if (ctx->framebuffer_s.cbufs[0])
      etna_resource(ctx->framebuffer_s.cbufs[0]->texture)->seqno++;
   if (ctx->framebuffer_s.zsbuf)
      etna_resource(ctx->framebuffer_s.zsbuf->texture)->seqno++;
   if (info->index_size && indexbuf != info->index.resource)
      pipe_resource_reference(&indexbuf, NULL);
}

static void
etna_flush(struct pipe_context *pctx, struct pipe_fence_handle **fence,
           enum pipe_flush_flags flags)
{
   struct etna_context *ctx = etna_context(pctx);
   int out_fence_fd = -1;

   list_for_each_entry(struct etna_hw_query, hq, &ctx->active_hw_queries, node)
      etna_hw_query_suspend(hq, ctx);

   etna_cmd_stream_flush2(ctx->stream, ctx->in_fence_fd,
			  (flags & PIPE_FLUSH_FENCE_FD) ? &out_fence_fd :
			  NULL);

   list_for_each_entry(struct etna_hw_query, hq, &ctx->active_hw_queries, node)
      etna_hw_query_resume(hq, ctx);

   if (fence)
      *fence = etna_fence_create(pctx, out_fence_fd);
}

static void
etna_cmd_stream_reset_notify(struct etna_cmd_stream *stream, void *priv)
{
   struct etna_context *ctx = priv;
   struct etna_resource *rsc, *rsc_tmp;

   etna_set_state(stream, VIVS_GL_API_MODE, VIVS_GL_API_MODE_OPENGL);
   etna_set_state(stream, VIVS_GL_VERTEX_ELEMENT_CONFIG, 0x00000001);
   /* blob sets this to 0x40000031 on GC7000, seems to make no difference,
    * but keep it in mind if depth behaves strangely. */
   etna_set_state(stream, VIVS_RA_EARLY_DEPTH, 0x00000031);
   etna_set_state(stream, VIVS_PA_W_CLIP_LIMIT, 0x34000001);
   etna_set_state(stream, VIVS_PA_FLAGS, 0x00000000); /* blob sets ZCONVERT_BYPASS on GC3000+, this messes up z for us */
   etna_set_state(stream, VIVS_PA_VIEWPORT_UNK00A80, 0x38a01404);
   etna_set_state(stream, VIVS_PA_VIEWPORT_UNK00A84, fui(8192.0));
   etna_set_state(stream, VIVS_PA_ZFARCLIPPING, 0x00000000);
   etna_set_state(stream, VIVS_PE_ALPHA_COLOR_EXT0, 0x00000000);
   etna_set_state(stream, VIVS_PE_ALPHA_COLOR_EXT1, 0x00000000);
   etna_set_state(stream, VIVS_RA_HDEPTH_CONTROL, 0x00007000);
   etna_set_state(stream, VIVS_PE_STENCIL_CONFIG_EXT2, 0x00000000);
   etna_set_state(stream, VIVS_PS_CONTROL_EXT, 0x00000000);

   /* There is no HALTI0 specific state */
   if (ctx->specs.halti >= 1) { /* Only on HALTI1+ */
      etna_set_state(stream, VIVS_VS_HALTI1_UNK00884, 0x00000808);
   }
   if (ctx->specs.halti >= 2) { /* Only on HALTI2+ */
      etna_set_state(stream, VIVS_RA_UNK00E0C, 0x00000000);
   }
   if (ctx->specs.halti >= 3) { /* Only on HALTI3+ */
      etna_set_state(stream, VIVS_PE_MEM_CONFIG, 0x00000000); /* TODO: cache modes */
      etna_set_state(stream, VIVS_PS_HALTI3_UNK0103C, 0x76543210);
   }
   if (ctx->specs.halti >= 4) { /* Only on HALTI4+ */
      etna_set_state(stream, VIVS_PS_MSAA_CONFIG, 0x6fffffff & 0xf70fffff & 0xfff6ffff &
                                                  0xffff6fff & 0xfffff6ff & 0xffffff7f);
      etna_set_state(stream, VIVS_PE_HALTI4_UNK014C0, 0x00000000);
   }
   if (ctx->specs.halti >= 5) { /* Only on HALTI5+ */
      etna_set_state(stream, VIVS_NTE_DESCRIPTOR_UNK14C40, 0x00000001);
      etna_set_state(stream, VIVS_FE_HALTI5_UNK007D8, 0x00000002);
      etna_set_state(stream, VIVS_FE_HALTI5_UNK007C4, 0x00000000);
      etna_set_state(stream, VIVS_PS_SAMPLER_BASE, 0x00000000);
      etna_set_state(stream, VIVS_VS_SAMPLER_BASE, 0x00000020);
      etna_set_state(stream, VIVS_SH_CONFIG, VIVS_SH_CONFIG_RTNE_ROUNDING);
   } else { /* Only on pre-HALTI5 */
      etna_set_state(stream, VIVS_GL_UNK03834, 0x00000000);
      etna_set_state(stream, VIVS_GL_UNK03838, 0x00000000);
      etna_set_state(stream, VIVS_GL_UNK03854, 0x00000000);
   }

   if (!ctx->specs.use_blt) {
      /* Enable SINGLE_BUFFER for resolve, if supported */
      etna_set_state(stream, VIVS_RS_SINGLE_BUFFER, COND(ctx->specs.single_buffer, VIVS_RS_SINGLE_BUFFER_ENABLE));
   }

   ctx->dirty = ~0L;
   ctx->dirty_sampler_views = ~0L;

   /* go through all the used resources and clear their status flag */
   LIST_FOR_EACH_ENTRY_SAFE(rsc, rsc_tmp, &ctx->used_resources, list)
   {
      debug_assert(rsc->status != 0);
      rsc->status = 0;
      rsc->pending_ctx = NULL;
      list_delinit(&rsc->list);
   }

   assert(LIST_IS_EMPTY(&ctx->used_resources));
}

static void
etna_set_debug_callback(struct pipe_context *pctx,
                        const struct pipe_debug_callback *cb)
{
   struct etna_context *ctx = etna_context(pctx);

   if (cb)
      ctx->debug = *cb;
   else
      memset(&ctx->debug, 0, sizeof(ctx->debug));
}

struct pipe_context *
etna_context_create(struct pipe_screen *pscreen, void *priv, unsigned flags)
{
   struct etna_context *ctx = CALLOC_STRUCT(etna_context);
   struct etna_screen *screen;
   struct pipe_context *pctx;

   if (ctx == NULL)
      return NULL;

   pctx = &ctx->base;
   pctx->priv = ctx;
   pctx->screen = pscreen;
   pctx->stream_uploader = u_upload_create_default(pctx);
   if (!pctx->stream_uploader)
      goto fail;
   pctx->const_uploader = pctx->stream_uploader;

   screen = etna_screen(pscreen);
   ctx->stream = etna_cmd_stream_new(screen->pipe, 0x2000, &etna_cmd_stream_reset_notify, ctx);
   if (ctx->stream == NULL)
      goto fail;

   /* context ctxate setup */
   ctx->specs = screen->specs;
   ctx->screen = screen;
   /* need some sane default in case state tracker doesn't set some state: */
   ctx->sample_mask = 0xffff;

   list_inithead(&ctx->used_resources);

   /*  Set sensible defaults for state */
   etna_cmd_stream_reset_notify(ctx->stream, ctx);

   ctx->in_fence_fd = -1;

   pctx->destroy = etna_context_destroy;
   pctx->draw_vbo = etna_draw_vbo;
   pctx->flush = etna_flush;
   pctx->set_debug_callback = etna_set_debug_callback;
   pctx->create_fence_fd = etna_create_fence_fd;
   pctx->fence_server_sync = etna_fence_server_sync;

   /* creation of compile states */
   pctx->create_blend_state = etna_blend_state_create;
   pctx->create_rasterizer_state = etna_rasterizer_state_create;
   pctx->create_depth_stencil_alpha_state = etna_zsa_state_create;

   etna_clear_blit_init(pctx);
   etna_query_context_init(pctx);
   etna_state_init(pctx);
   etna_surface_init(pctx);
   etna_shader_init(pctx);
   etna_texture_init(pctx);
   etna_transfer_init(pctx);

   ctx->blitter = util_blitter_create(pctx);
   if (!ctx->blitter)
      goto fail;

   /* Generate the bitmask of supported draw primitives. */
   ctx->prim_hwsupport = 1 << PIPE_PRIM_POINTS |
                         1 << PIPE_PRIM_LINES |
                         1 << PIPE_PRIM_LINE_STRIP |
                         1 << PIPE_PRIM_TRIANGLES |
                         1 << PIPE_PRIM_TRIANGLE_STRIP |
                         1 << PIPE_PRIM_TRIANGLE_FAN;

   if (VIV_FEATURE(ctx->screen, chipMinorFeatures2, LINE_LOOP))
      ctx->prim_hwsupport |= 1 << PIPE_PRIM_LINE_LOOP;

   ctx->primconvert = util_primconvert_create(pctx, ctx->prim_hwsupport);
   if (!ctx->primconvert)
      goto fail;

   slab_create_child(&ctx->transfer_pool, &screen->transfer_pool);
   list_inithead(&ctx->active_hw_queries);

   return pctx;

fail:
   pctx->destroy(pctx);

   return NULL;
}
