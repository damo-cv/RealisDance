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

#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "virgl_context.h"
#include "virgl_resource.h"
#include "virgl_screen.h"

static void virgl_buffer_destroy(struct pipe_screen *screen,
                                 struct pipe_resource *buf)
{
   struct virgl_screen *vs = virgl_screen(screen);
   struct virgl_buffer *vbuf = virgl_buffer(buf);

   util_range_destroy(&vbuf->valid_buffer_range);
   vs->vws->resource_unref(vs->vws, vbuf->base.hw_res);
   FREE(vbuf);
}

static void *virgl_buffer_transfer_map(struct pipe_context *ctx,
                                       struct pipe_resource *resource,
                                       unsigned level,
                                       unsigned usage,
                                       const struct pipe_box *box,
                                       struct pipe_transfer **transfer)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *vs = virgl_screen(ctx->screen);
   struct virgl_buffer *vbuf = virgl_buffer(resource);
   struct virgl_transfer *trans;
   void *ptr;
   bool readback;
   uint32_t offset;
   bool doflushwait = false;

   if ((usage & PIPE_TRANSFER_READ) && (vbuf->on_list == TRUE))
      doflushwait = true;
   else
      doflushwait = virgl_res_needs_flush_wait(vctx, &vbuf->base, usage);

   if (doflushwait)
      ctx->flush(ctx, NULL, 0);

   trans = slab_alloc(&vctx->texture_transfer_pool);
   if (!trans)
      return NULL;

   trans->base.resource = resource;
   trans->base.level = level;
   trans->base.usage = usage;
   trans->base.box = *box;
   trans->base.stride = 0;
   trans->base.layer_stride = 0;

   offset = box->x;

   readback = virgl_res_needs_readback(vctx, &vbuf->base, usage);
   if (readback)
      vs->vws->transfer_get(vs->vws, vbuf->base.hw_res, box, trans->base.stride, trans->base.layer_stride, offset, level);

   if (!(usage & PIPE_TRANSFER_UNSYNCHRONIZED))
      doflushwait = true;

   if (doflushwait || readback)
      vs->vws->resource_wait(vs->vws, vbuf->base.hw_res);

   ptr = vs->vws->resource_map(vs->vws, vbuf->base.hw_res);
   if (!ptr) {
      return NULL;
   }

   trans->offset = offset;
   *transfer = &trans->base;

   return ptr + trans->offset;
}

static void virgl_buffer_transfer_unmap(struct pipe_context *ctx,
                                        struct pipe_transfer *transfer)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_transfer *trans = virgl_transfer(transfer);
   struct virgl_buffer *vbuf = virgl_buffer(transfer->resource);

   if (trans->base.usage & PIPE_TRANSFER_WRITE) {
      if (!(transfer->usage & PIPE_TRANSFER_FLUSH_EXPLICIT)) {
         struct virgl_screen *vs = virgl_screen(ctx->screen);
         vctx->num_transfers++;
         vs->vws->transfer_put(vs->vws, vbuf->base.hw_res,
                               &transfer->box, trans->base.stride, trans->base.layer_stride, trans->offset, transfer->level);

      }
   }

   slab_free(&vctx->texture_transfer_pool, trans);
}

static void virgl_buffer_transfer_flush_region(struct pipe_context *ctx,
                                               struct pipe_transfer *transfer,
                                               const struct pipe_box *box)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_buffer *vbuf = virgl_buffer(transfer->resource);

   if (!vbuf->on_list) {
       struct pipe_resource *res = NULL;

       list_addtail(&vbuf->flush_list, &vctx->to_flush_bufs);
       vbuf->on_list = TRUE;
       pipe_resource_reference(&res, &vbuf->base.u.b);
   }

   util_range_add(&vbuf->valid_buffer_range, transfer->box.x + box->x,
                  transfer->box.x + box->x + box->width);

   vbuf->base.clean = FALSE;
}

static const struct u_resource_vtbl virgl_buffer_vtbl =
{
   u_default_resource_get_handle,            /* get_handle */
   virgl_buffer_destroy,                     /* resource_destroy */
   virgl_buffer_transfer_map,                /* transfer_map */
   virgl_buffer_transfer_flush_region,       /* transfer_flush_region */
   virgl_buffer_transfer_unmap,              /* transfer_unmap */
};

struct pipe_resource *virgl_buffer_create(struct virgl_screen *vs,
                                          const struct pipe_resource *template)
{
   struct virgl_buffer *buf;
   uint32_t size;
   uint32_t vbind;
   buf = CALLOC_STRUCT(virgl_buffer);
   buf->base.clean = TRUE;
   buf->base.u.b = *template;
   buf->base.u.b.screen = &vs->base;
   buf->base.u.vtbl = &virgl_buffer_vtbl;
   pipe_reference_init(&buf->base.u.b.reference, 1);
   util_range_init(&buf->valid_buffer_range);

   vbind = pipe_to_virgl_bind(template->bind);
   size = template->width0;

   /* SSBOs and texture buffers can written to by host compute shaders. */
   if (vbind == VIRGL_BIND_SHADER_BUFFER || vbind == VIRGL_BIND_SAMPLER_VIEW)
      buf->base.clean = FALSE;
   buf->base.hw_res = vs->vws->resource_create(vs->vws, template->target, template->format, vbind, template->width0, 1, 1, 1, 0, 0, size);

   util_range_set_empty(&buf->valid_buffer_range);
   return &buf->base.u.b;
}
