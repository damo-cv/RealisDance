/****************************************************************************
 * Copyright (C) 2015 Intel Corporation.   All Rights Reserved.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 ***************************************************************************/

#include "swr_context.h"
#include "swr_query.h"

static void
swr_clear(struct pipe_context *pipe,
          unsigned buffers,
          const union pipe_color_union *color,
          double depth,
          unsigned stencil)
{
   struct swr_context *ctx = swr_context(pipe);
   struct pipe_framebuffer_state *fb = &ctx->framebuffer;

   UINT clearMask = 0;
   unsigned layers = 0;

   if (!swr_check_render_cond(pipe))
      return;

   swr_update_derived(pipe);

   if (buffers & PIPE_CLEAR_COLOR && fb->nr_cbufs) {
      for (unsigned i = 0; i < fb->nr_cbufs; ++i)
         if (fb->cbufs[i] && (buffers & (PIPE_CLEAR_COLOR0 << i))) {
            clearMask |= (SWR_ATTACHMENT_COLOR0_BIT << i);
            layers = std::max(layers, fb->cbufs[i]->u.tex.last_layer -
                                      fb->cbufs[i]->u.tex.first_layer + 1u);
         }
   }

   if (buffers & PIPE_CLEAR_DEPTH && fb->zsbuf) {
      clearMask |= SWR_ATTACHMENT_DEPTH_BIT;
      layers = std::max(layers, fb->zsbuf->u.tex.last_layer -
                                fb->zsbuf->u.tex.first_layer + 1u);
   }

   if (buffers & PIPE_CLEAR_STENCIL && fb->zsbuf) {
      clearMask |= SWR_ATTACHMENT_STENCIL_BIT;
      layers = std::max(layers, fb->zsbuf->u.tex.last_layer -
                                fb->zsbuf->u.tex.first_layer + 1u);
   }

#if 0 // XXX HACK, override clear color alpha. On ubuntu, clears are
      // transparent.
   ((union pipe_color_union *)color)->f[3] = 1.0; /* cast off your const'd-ness */
#endif

   SWR_RECT clear_rect;
   /* If enabled, clear to scissor; otherwise clear full surface */
   if (ctx->rasterizer && ctx->rasterizer->scissor) {
      clear_rect = ctx->swr_scissor;
   } else {
      clear_rect = {0, 0, (int32_t)fb->width, (int32_t)fb->height};
   }

   for (unsigned i = 0; i < layers; ++i) {
      swr_update_draw_context(ctx);
      ctx->api.pfnSwrClearRenderTarget(ctx->swrContext, clearMask, i,
                                       color->f, depth, stencil,
                                       clear_rect);

      // Mask out the attachments that are out of layers.
      if (fb->zsbuf &&
          (fb->zsbuf->u.tex.last_layer <= fb->zsbuf->u.tex.first_layer + i))
         clearMask &= ~(SWR_ATTACHMENT_DEPTH_BIT | SWR_ATTACHMENT_STENCIL_BIT);
      for (unsigned c = 0; c < fb->nr_cbufs; ++c) {
         const struct pipe_surface *sf = fb->cbufs[c];
         if (sf && (sf->u.tex.last_layer <= sf->u.tex.first_layer + i))
            clearMask &= ~(SWR_ATTACHMENT_COLOR0_BIT << c);
      }
   }
}


#if 0 // XXX, these don't get called. how to get these called?  Do we need
      // them?  Docs?
static void
swr_clear_render_target(struct pipe_context *pipe, struct pipe_surface *ps,
                        const union pipe_color_union *color,
                        unsigned x, unsigned y, unsigned w, unsigned h,
                        bool render_condition_enabled)
{
   struct swr_context *ctx = swr_context(pipe);
   fprintf(stderr, "SWR swr_clear_render_target!\n");

   ctx->dirty |= SWR_NEW_FRAMEBUFFER | SWR_NEW_SCISSOR;
}

static void
swr_clear_depth_stencil(struct pipe_context *pipe, struct pipe_surface *ps,
                        unsigned buffers, double depth, unsigned stencil,
                        unsigned x, unsigned y, unsigned w, unsigned h,
                        bool render_condition_enabled)
{
   struct swr_context *ctx = swr_context(pipe);
   fprintf(stderr, "SWR swr_clear_depth_stencil!\n");

   ctx->dirty |= SWR_NEW_FRAMEBUFFER | SWR_NEW_SCISSOR;
}

static void
swr_clear_buffer(struct pipe_context *pipe,
                 struct pipe_resource *res,
                 unsigned offset, unsigned size,
                 const void *data, int data_size)
{
   fprintf(stderr, "SWR swr_clear_buffer!\n");
   struct swr_context *ctx = swr_context(pipe);
   struct swr_resource *buf = swr_resource(res);
   union pipe_color_union color;
   enum pipe_format dst_fmt;
   unsigned width, height, elements;

   assert(res->target == PIPE_BUFFER);
   assert(buf);
   assert(size % data_size == 0);

   SWR_SURFACE_STATE &swr_buffer = buf->swr;

   ctx->dirty |= SWR_NEW_FRAMEBUFFER | SWR_NEW_SCISSOR;
}
#endif


void
swr_clear_init(struct pipe_context *pipe)
{
   pipe->clear = swr_clear;
#if 0 // XXX, these don't get called. how to get these called?  Do we need
      // them?  Docs?
   pipe->clear_render_target = swr_clear_render_target;
   pipe->clear_depth_stencil = swr_clear_depth_stencil;
   pipe->clear_buffer = swr_clear_buffer;
#endif
}
