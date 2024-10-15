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
#include "util/u_format.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"

#include "virgl_context.h"
#include "virgl_resource.h"
#include "virgl_screen.h"

static void virgl_copy_region_with_blit(struct pipe_context *pipe,
                                        struct pipe_resource *dst,
                                        unsigned dst_level,
                                        unsigned dstx, unsigned dsty, unsigned dstz,
                                        struct pipe_resource *src,
                                        unsigned src_level,
                                        const struct pipe_box *src_box)
{
   struct pipe_blit_info blit;

   memset(&blit, 0, sizeof(blit));
   blit.src.resource = src;
   blit.src.format = src->format;
   blit.src.level = src_level;
   blit.src.box = *src_box;
   blit.dst.resource = dst;
   blit.dst.format = dst->format;
   blit.dst.level = dst_level;
   blit.dst.box.x = dstx;
   blit.dst.box.y = dsty;
   blit.dst.box.z = dstz;
   blit.dst.box.width = src_box->width;
   blit.dst.box.height = src_box->height;
   blit.dst.box.depth = src_box->depth;
   blit.mask = util_format_get_mask(src->format) &
      util_format_get_mask(dst->format);
   blit.filter = PIPE_TEX_FILTER_NEAREST;

   if (blit.mask) {
      pipe->blit(pipe, &blit);
   }
}
static void virgl_init_temp_resource_from_box(struct pipe_resource *res,
                                              struct pipe_resource *orig,
                                              const struct pipe_box *box,
                                              unsigned level, unsigned flags)
{
   memset(res, 0, sizeof(*res));
   res->format = orig->format;
   res->width0 = box->width;
   res->height0 = box->height;
   res->depth0 = 1;
   res->array_size = 1;
   res->usage = PIPE_USAGE_STAGING;
   res->flags = flags;

   /* We must set the correct texture target and dimensions for a 3D box. */
   if (box->depth > 1 && util_max_layer(orig, level) > 0)
      res->target = orig->target;
   else
      res->target = PIPE_TEXTURE_2D;

   switch (res->target) {
   case PIPE_TEXTURE_1D_ARRAY:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_CUBE_ARRAY:
      res->array_size = box->depth;
      break;
   case PIPE_TEXTURE_3D:
      res->depth0 = box->depth;
      break;
   default:
      break;
   }
}

static unsigned
vrend_get_tex_image_offset(const struct virgl_texture *res,
                           unsigned level, unsigned layer)
{
   const struct pipe_resource *pres = &res->base.u.b;
   const unsigned hgt = u_minify(pres->height0, level);
   const unsigned nblocksy = util_format_get_nblocksy(pres->format, hgt);
   unsigned offset = res->level_offset[level];

   if (pres->target == PIPE_TEXTURE_CUBE ||
       pres->target == PIPE_TEXTURE_CUBE_ARRAY ||
       pres->target == PIPE_TEXTURE_3D ||
       pres->target == PIPE_TEXTURE_2D_ARRAY) {
      offset += layer * nblocksy * res->stride[level];
   }
   else if (pres->target == PIPE_TEXTURE_1D_ARRAY) {
      offset += layer * res->stride[level];
   }
   else {
      assert(layer == 0);
   }

   return offset;
}

static void *virgl_texture_transfer_map(struct pipe_context *ctx,
                                        struct pipe_resource *resource,
                                        unsigned level,
                                        unsigned usage,
                                        const struct pipe_box *box,
                                        struct pipe_transfer **transfer)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_screen *vs = virgl_screen(ctx->screen);
   struct virgl_texture *vtex = virgl_texture(resource);
   enum pipe_format format = resource->format;
   struct virgl_transfer *trans;
   void *ptr;
   boolean readback = TRUE;
   uint32_t offset;
   struct virgl_hw_res *hw_res;
   const unsigned h = u_minify(vtex->base.u.b.height0, level);
   const unsigned nblocksy = util_format_get_nblocksy(format, h);
   uint32_t l_stride;
   bool doflushwait;

   doflushwait = virgl_res_needs_flush_wait(vctx, &vtex->base, usage);
   if (doflushwait)
      ctx->flush(ctx, NULL, 0);

   trans = slab_alloc(&vctx->texture_transfer_pool);
   if (!trans)
      return NULL;

   trans->base.resource = resource;
   trans->base.level = level;
   trans->base.usage = usage;
   trans->base.box = *box;
   trans->base.stride = vtex->stride[level];
   trans->base.layer_stride = trans->base.stride * nblocksy;

   if (resource->target != PIPE_TEXTURE_3D &&
       resource->target != PIPE_TEXTURE_CUBE &&
       resource->target != PIPE_TEXTURE_1D_ARRAY &&
       resource->target != PIPE_TEXTURE_2D_ARRAY &&
       resource->target != PIPE_TEXTURE_CUBE_ARRAY)
      l_stride = 0;
   else
      l_stride = trans->base.layer_stride;

   if (resource->nr_samples > 1) {
      struct pipe_resource tmp_resource;
      virgl_init_temp_resource_from_box(&tmp_resource, resource, box,
                                        level, 0);

      trans->resolve_tmp = (struct virgl_resource *)ctx->screen->resource_create(ctx->screen, &tmp_resource);

      virgl_copy_region_with_blit(ctx, &trans->resolve_tmp->u.b, 0, 0, 0, 0, resource, level, box);
      ctx->flush(ctx, NULL, 0);
      /* we want to do a resolve blit into the temporary */
      hw_res = trans->resolve_tmp->hw_res;
      offset = 0;
      trans->base.stride = ((struct virgl_texture*)trans->resolve_tmp)->stride[level];
      trans->base.layer_stride = trans->base.stride * nblocksy;
   } else {
      offset = vrend_get_tex_image_offset(vtex, level, box->z);

      offset += box->y / util_format_get_blockheight(format) * trans->base.stride +
      box->x / util_format_get_blockwidth(format) * util_format_get_blocksize(format);
      hw_res = vtex->base.hw_res;
      trans->resolve_tmp = NULL;
   }

   readback = virgl_res_needs_readback(vctx, &vtex->base, usage);
   if (readback)
      vs->vws->transfer_get(vs->vws, hw_res, box, trans->base.stride, l_stride, offset, level);

   if (doflushwait || readback)
      vs->vws->resource_wait(vs->vws, vtex->base.hw_res);

   ptr = vs->vws->resource_map(vs->vws, hw_res);
   if (!ptr) {
      slab_free(&vctx->texture_transfer_pool, trans);
      return NULL;
   }

   trans->offset = offset;
   *transfer = &trans->base;

   return ptr + trans->offset;
}

static void virgl_texture_transfer_unmap(struct pipe_context *ctx,
                                         struct pipe_transfer *transfer)
{
   struct virgl_context *vctx = virgl_context(ctx);
   struct virgl_transfer *trans = virgl_transfer(transfer);
   struct virgl_texture *vtex = virgl_texture(transfer->resource);
   uint32_t l_stride;

   if (transfer->resource->target != PIPE_TEXTURE_3D &&
       transfer->resource->target != PIPE_TEXTURE_CUBE &&
       transfer->resource->target != PIPE_TEXTURE_1D_ARRAY &&
       transfer->resource->target != PIPE_TEXTURE_2D_ARRAY &&
       transfer->resource->target != PIPE_TEXTURE_CUBE_ARRAY)
      l_stride = 0;
   else
      l_stride = trans->base.layer_stride;

   if (trans->base.usage & PIPE_TRANSFER_WRITE) {
      if (!(transfer->usage & PIPE_TRANSFER_FLUSH_EXPLICIT)) {
         struct virgl_screen *vs = virgl_screen(ctx->screen);
         vtex->base.clean = FALSE;
         vctx->num_transfers++;
         vs->vws->transfer_put(vs->vws, vtex->base.hw_res,
                               &transfer->box, trans->base.stride, l_stride, trans->offset, transfer->level);

      }
   }

   if (trans->resolve_tmp)
      pipe_resource_reference((struct pipe_resource **)&trans->resolve_tmp, NULL);

   slab_free(&vctx->texture_transfer_pool, trans);
}


static void
vrend_resource_layout(struct virgl_texture *res,
                      uint32_t *total_size)
{
   struct pipe_resource *pt = &res->base.u.b;
   unsigned level;
   unsigned width = pt->width0;
   unsigned height = pt->height0;
   unsigned depth = pt->depth0;
   unsigned buffer_size = 0;

   for (level = 0; level <= pt->last_level; level++) {
      unsigned slices;

      if (pt->target == PIPE_TEXTURE_CUBE)
         slices = 6;
      else if (pt->target == PIPE_TEXTURE_3D)
         slices = depth;
      else
         slices = pt->array_size;

      res->stride[level] = util_format_get_stride(pt->format, width);
      res->level_offset[level] = buffer_size;

      buffer_size += (util_format_get_nblocksy(pt->format, height) *
                      slices * res->stride[level]);

      width = u_minify(width, 1);
      height = u_minify(height, 1);
      depth = u_minify(depth, 1);
   }

   if (pt->nr_samples <= 1)
      *total_size = buffer_size;
   else /* don't create guest backing store for MSAA */
      *total_size = 0;
}

static boolean virgl_texture_get_handle(struct pipe_screen *screen,
                                         struct pipe_resource *ptex,
                                         struct winsys_handle *whandle)
{
   struct virgl_screen *vs = virgl_screen(screen);
   struct virgl_texture *vtex = virgl_texture(ptex);

   return vs->vws->resource_get_handle(vs->vws, vtex->base.hw_res, vtex->stride[0], whandle);
}

static void virgl_texture_destroy(struct pipe_screen *screen,
                                  struct pipe_resource *res)
{
   struct virgl_screen *vs = virgl_screen(screen);
   struct virgl_texture *vtex = virgl_texture(res);
   vs->vws->resource_unref(vs->vws, vtex->base.hw_res);
   FREE(vtex);
}

static const struct u_resource_vtbl virgl_texture_vtbl =
{
   virgl_texture_get_handle,            /* get_handle */
   virgl_texture_destroy,               /* resource_destroy */
   virgl_texture_transfer_map,          /* transfer_map */
   NULL,                                /* transfer_flush_region */
   virgl_texture_transfer_unmap,        /* transfer_unmap */
};

struct pipe_resource *
virgl_texture_from_handle(struct virgl_screen *vs,
                          const struct pipe_resource *template,
                          struct winsys_handle *whandle)
{
   struct virgl_texture *tex = CALLOC_STRUCT(virgl_texture);
   tex->base.u.b = *template;
   tex->base.u.b.screen = &vs->base;
   pipe_reference_init(&tex->base.u.b.reference, 1);
   tex->base.u.vtbl = &virgl_texture_vtbl;

   tex->base.hw_res = vs->vws->resource_create_from_handle(vs->vws, whandle);
   return &tex->base.u.b;
}

struct pipe_resource *virgl_texture_create(struct virgl_screen *vs,
                                           const struct pipe_resource *template)
{
   struct virgl_texture *tex;
   uint32_t size;
   unsigned vbind;

   tex = CALLOC_STRUCT(virgl_texture);
   tex->base.clean = TRUE;
   tex->base.u.b = *template;
   tex->base.u.b.screen = &vs->base;
   pipe_reference_init(&tex->base.u.b.reference, 1);
   tex->base.u.vtbl = &virgl_texture_vtbl;
   vrend_resource_layout(tex, &size);

   vbind = pipe_to_virgl_bind(template->bind);
   tex->base.hw_res = vs->vws->resource_create(vs->vws, template->target, template->format, vbind, template->width0, template->height0, template->depth0, template->array_size, template->last_level, template->nr_samples, size);
   if (!tex->base.hw_res) {
      FREE(tex);
      return NULL;
   }
   return &tex->base.u.b;
}
