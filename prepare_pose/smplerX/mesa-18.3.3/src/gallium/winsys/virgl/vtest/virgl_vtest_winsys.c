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
#include <stdio.h>
#include "util/u_memory.h"
#include "util/u_format.h"
#include "util/u_inlines.h"
#include "util/os_time.h"
#include "state_tracker/sw_winsys.h"

#include "virgl_vtest_winsys.h"
#include "virgl_vtest_public.h"

static void *virgl_vtest_resource_map(struct virgl_winsys *vws,
                                      struct virgl_hw_res *res);
static void virgl_vtest_resource_unmap(struct virgl_winsys *vws,
                                       struct virgl_hw_res *res);

static inline boolean can_cache_resource(struct virgl_hw_res *res)
{
   return res->cacheable == TRUE;
}

static uint32_t vtest_get_transfer_size(struct virgl_hw_res *res,
                                        const struct pipe_box *box,
                                        uint32_t stride, uint32_t layer_stride,
                                        uint32_t level, uint32_t *valid_stride_p)
{
   uint32_t valid_stride, valid_layer_stride;

   valid_stride = util_format_get_stride(res->format, box->width);
   if (stride) {
      if (box->height > 1)
         valid_stride = stride;
   }

   valid_layer_stride = util_format_get_2d_size(res->format, valid_stride,
                                                box->height);
   if (layer_stride) {
      if (box->depth > 1)
         valid_layer_stride = layer_stride;
   }

   *valid_stride_p = valid_stride;
   return valid_layer_stride * box->depth;
}

static int
virgl_vtest_transfer_put(struct virgl_winsys *vws,
                         struct virgl_hw_res *res,
                         const struct pipe_box *box,
                         uint32_t stride, uint32_t layer_stride,
                         uint32_t buf_offset, uint32_t level)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);
   uint32_t size;
   void *ptr;
   uint32_t valid_stride;

   size = vtest_get_transfer_size(res, box, stride, layer_stride, level,
                                  &valid_stride);

   /* The size calculated above is the full box size, but if this box origin
    * is not zero we may have to correct the transfer size to not read past the
    * end of the resource. The correct adjustment depends on various factors
    * that are not documented, so instead of going though all the hops to get
    * the size right up-front, we just make sure we don't read past the end.
    * FIXME: figure out what it takes to actually get this right.
    */
   if (size + buf_offset > res->size)
      size = res->size - buf_offset;

   virgl_vtest_send_transfer_put(vtws, res->res_handle,
                                 level, stride, layer_stride,
                                 box, size, buf_offset);
   ptr = virgl_vtest_resource_map(vws, res);
   virgl_vtest_send_transfer_put_data(vtws, ptr + buf_offset, size);
   virgl_vtest_resource_unmap(vws, res);
   return 0;
}

static int
virgl_vtest_transfer_get(struct virgl_winsys *vws,
                         struct virgl_hw_res *res,
                         const struct pipe_box *box,
                         uint32_t stride, uint32_t layer_stride,
                         uint32_t buf_offset, uint32_t level)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);
   uint32_t size;
   void *ptr;
   uint32_t valid_stride;

   size = vtest_get_transfer_size(res, box, stride, layer_stride, level,
                                  &valid_stride);
   /* Don't ask for more pixels than available (see above) */
   if (size + buf_offset > res->size)
      size = res->size - buf_offset;

   virgl_vtest_send_transfer_get(vtws, res->res_handle,
                                 level, stride, layer_stride,
                                 box, size, buf_offset);

   ptr = virgl_vtest_resource_map(vws, res);

   /* This functions seems to be using a specific transfer resource that
    * has exactly the box size and hence its src stride is equal to the target
    * stride */
   virgl_vtest_recv_transfer_get_data(vtws, ptr + buf_offset, size,
                                      valid_stride, box, res->format, valid_stride);

   virgl_vtest_resource_unmap(vws, res);
   return 0;
}

static void virgl_hw_res_destroy(struct virgl_vtest_winsys *vtws,
                                 struct virgl_hw_res *res)
{
   virgl_vtest_send_resource_unref(vtws, res->res_handle);
   if (res->dt)
      vtws->sws->displaytarget_destroy(vtws->sws, res->dt);
   free(res->ptr);
   FREE(res);
}

static boolean virgl_vtest_resource_is_busy(struct virgl_vtest_winsys *vtws,
                                            struct virgl_hw_res *res)
{
   /* implement busy check */
   int ret;
   ret = virgl_vtest_busy_wait(vtws, res->res_handle, 0);

   if (ret < 0)
      return FALSE;

   return ret == 1 ? TRUE : FALSE;
}

static void
virgl_cache_flush(struct virgl_vtest_winsys *vtws)
{
   struct list_head *curr, *next;
   struct virgl_hw_res *res;

   mtx_lock(&vtws->mutex);
   curr = vtws->delayed.next;
   next = curr->next;

   while (curr != &vtws->delayed) {
      res = LIST_ENTRY(struct virgl_hw_res, curr, head);
      LIST_DEL(&res->head);
      virgl_hw_res_destroy(vtws, res);
      curr = next;
      next = curr->next;
   }
   mtx_unlock(&vtws->mutex);
}

static void
virgl_cache_list_check_free(struct virgl_vtest_winsys *vtws)
{
   struct list_head *curr, *next;
   struct virgl_hw_res *res;
   int64_t now;

   now = os_time_get();
   curr = vtws->delayed.next;
   next = curr->next;
   while (curr != &vtws->delayed) {
      res = LIST_ENTRY(struct virgl_hw_res, curr, head);
      if (!os_time_timeout(res->start, res->end, now))
         break;

      LIST_DEL(&res->head);
      virgl_hw_res_destroy(vtws, res);
      curr = next;
      next = curr->next;
   }
}

static void virgl_vtest_resource_reference(struct virgl_vtest_winsys *vtws,
                                           struct virgl_hw_res **dres,
                                           struct virgl_hw_res *sres)
{
   struct virgl_hw_res *old = *dres;
   if (pipe_reference(&(*dres)->reference, &sres->reference)) {
      if (!can_cache_resource(old)) {
         virgl_hw_res_destroy(vtws, old);
      } else {
         mtx_lock(&vtws->mutex);
         virgl_cache_list_check_free(vtws);

         old->start = os_time_get();
         old->end = old->start + vtws->usecs;
         LIST_ADDTAIL(&old->head, &vtws->delayed);
         vtws->num_delayed++;
         mtx_unlock(&vtws->mutex);
      }
   }
   *dres = sres;
}

static struct virgl_hw_res *
virgl_vtest_winsys_resource_create(struct virgl_winsys *vws,
                                   enum pipe_texture_target target,
                                   uint32_t format,
                                   uint32_t bind,
                                   uint32_t width,
                                   uint32_t height,
                                   uint32_t depth,
                                   uint32_t array_size,
                                   uint32_t last_level,
                                   uint32_t nr_samples,
                                   uint32_t size)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);
   struct virgl_hw_res *res;
   static int handle = 1;

   res = CALLOC_STRUCT(virgl_hw_res);
   if (!res)
      return NULL;

   if (bind & (VIRGL_BIND_DISPLAY_TARGET | VIRGL_BIND_SCANOUT)) {
      res->dt = vtws->sws->displaytarget_create(vtws->sws, bind, format,
                                                width, height, 64, NULL,
                                                &res->stride);

   } else {
      res->ptr = align_malloc(size, 64);
      if (!res->ptr) {
         FREE(res);
         return NULL;
      }
   }

   res->bind = bind;
   res->format = format;
   res->height = height;
   res->width = width;
   res->size = size;
   virgl_vtest_send_resource_create(vtws, handle, target, format, bind,
                                    width, height, depth, array_size,
                                    last_level, nr_samples, size);

   res->res_handle = handle++;
   pipe_reference_init(&res->reference, 1);
   return res;
}

static void virgl_vtest_winsys_resource_unref(struct virgl_winsys *vws,
                                              struct virgl_hw_res *hres)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);
   virgl_vtest_resource_reference(vtws, &hres, NULL);
}

static void *virgl_vtest_resource_map(struct virgl_winsys *vws,
                                      struct virgl_hw_res *res)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);

   if (res->dt) {
      return vtws->sws->displaytarget_map(vtws->sws, res->dt, 0);
   } else {
      res->mapped = res->ptr;
      return res->mapped;
   }
}

static void virgl_vtest_resource_unmap(struct virgl_winsys *vws,
                                       struct virgl_hw_res *res)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);
   if (res->mapped)
      res->mapped = NULL;

   if (res->dt)
      vtws->sws->displaytarget_unmap(vtws->sws, res->dt);
}

static void virgl_vtest_resource_wait(struct virgl_winsys *vws,
                                      struct virgl_hw_res *res)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);

   virgl_vtest_busy_wait(vtws, res->res_handle, VCMD_BUSY_WAIT_FLAG_WAIT);
}

static inline int virgl_is_res_compat(struct virgl_vtest_winsys *vtws,
                                      struct virgl_hw_res *res,
                                      uint32_t size, uint32_t bind,
                                      uint32_t format)
{
   if (res->bind != bind)
      return 0;
   if (res->format != format)
      return 0;
   if (res->size < size)
      return 0;
   if (res->size > size * 2)
      return 0;

   if (virgl_vtest_resource_is_busy(vtws, res)) {
      return -1;
   }

   return 1;
}

static struct virgl_hw_res *
virgl_vtest_winsys_resource_cache_create(struct virgl_winsys *vws,
                                         enum pipe_texture_target target,
                                         uint32_t format,
                                         uint32_t bind,
                                         uint32_t width,
                                         uint32_t height,
                                         uint32_t depth,
                                         uint32_t array_size,
                                         uint32_t last_level,
                                         uint32_t nr_samples,
                                         uint32_t size)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);
   struct virgl_hw_res *res, *curr_res;
   struct list_head *curr, *next;
   int64_t now;
   int ret;

   /* only store binds for vertex/index/const buffers */
   if (bind != VIRGL_BIND_CONSTANT_BUFFER && bind != VIRGL_BIND_INDEX_BUFFER &&
       bind != VIRGL_BIND_VERTEX_BUFFER && bind != VIRGL_BIND_CUSTOM)
      goto alloc;

   mtx_lock(&vtws->mutex);

   res = NULL;
   curr = vtws->delayed.next;
   next = curr->next;

   now = os_time_get();
   while (curr != &vtws->delayed) {
      curr_res = LIST_ENTRY(struct virgl_hw_res, curr, head);

      if (!res && ((ret = virgl_is_res_compat(vtws, curr_res, size, bind, format)) > 0))
         res = curr_res;
      else if (os_time_timeout(curr_res->start, curr_res->end, now)) {
         LIST_DEL(&curr_res->head);
         virgl_hw_res_destroy(vtws, curr_res);
      } else
         break;

      if (ret == -1)
         break;

      curr = next;
      next = curr->next;
   }

   if (!res && ret != -1) {
      while (curr != &vtws->delayed) {
         curr_res = LIST_ENTRY(struct virgl_hw_res, curr, head);
         ret = virgl_is_res_compat(vtws, curr_res, size, bind, format);
         if (ret > 0) {
            res = curr_res;
            break;
         }
         if (ret == -1)
            break;
         curr = next;
         next = curr->next;
      }
   }

   if (res) {
      LIST_DEL(&res->head);
      --vtws->num_delayed;
      mtx_unlock(&vtws->mutex);
      pipe_reference_init(&res->reference, 1);
      return res;
   }

   mtx_unlock(&vtws->mutex);

alloc:
   res = virgl_vtest_winsys_resource_create(vws, target, format, bind,
                                            width, height, depth, array_size,
                                            last_level, nr_samples, size);
   if (bind == VIRGL_BIND_CONSTANT_BUFFER || bind == VIRGL_BIND_INDEX_BUFFER ||
       bind == VIRGL_BIND_VERTEX_BUFFER)
      res->cacheable = TRUE;
   return res;
}

static struct virgl_cmd_buf *virgl_vtest_cmd_buf_create(struct virgl_winsys *vws)
{
   struct virgl_vtest_cmd_buf *cbuf;

   cbuf = CALLOC_STRUCT(virgl_vtest_cmd_buf);
   if (!cbuf)
      return NULL;

   cbuf->nres = 512;
   cbuf->res_bo = CALLOC(cbuf->nres, sizeof(struct virgl_hw_buf*));
   if (!cbuf->res_bo) {
      FREE(cbuf);
      return NULL;
   }
   cbuf->ws = vws;
   cbuf->base.buf = cbuf->buf;
   return &cbuf->base;
}

static void virgl_vtest_cmd_buf_destroy(struct virgl_cmd_buf *_cbuf)
{
   struct virgl_vtest_cmd_buf *cbuf = virgl_vtest_cmd_buf(_cbuf);

   FREE(cbuf->res_bo);
   FREE(cbuf);
}

static boolean virgl_vtest_lookup_res(struct virgl_vtest_cmd_buf *cbuf,
                                      struct virgl_hw_res *res)
{
   unsigned hash = res->res_handle & (sizeof(cbuf->is_handle_added)-1);
   int i;

   if (cbuf->is_handle_added[hash]) {
      i = cbuf->reloc_indices_hashlist[hash];
      if (cbuf->res_bo[i] == res)
         return true;

      for (i = 0; i < cbuf->cres; i++) {
         if (cbuf->res_bo[i] == res) {
            cbuf->reloc_indices_hashlist[hash] = i;
            return true;
         }
      }
   }
   return false;
}

static void virgl_vtest_release_all_res(struct virgl_vtest_winsys *vtws,
                                        struct virgl_vtest_cmd_buf *cbuf)
{
   int i;

   for (i = 0; i < cbuf->cres; i++) {
      p_atomic_dec(&cbuf->res_bo[i]->num_cs_references);
      virgl_vtest_resource_reference(vtws, &cbuf->res_bo[i], NULL);
   }
   cbuf->cres = 0;
}

static void virgl_vtest_add_res(struct virgl_vtest_winsys *vtws,
                                struct virgl_vtest_cmd_buf *cbuf,
                                struct virgl_hw_res *res)
{
   unsigned hash = res->res_handle & (sizeof(cbuf->is_handle_added)-1);

   if (cbuf->cres >= cbuf->nres) {
      unsigned new_nres = cbuf->nres + 256;
      struct virgl_hw_res **new_re_bo = REALLOC(cbuf->res_bo,
                                                cbuf->nres * sizeof(struct virgl_hw_buf*),
                                                new_nres * sizeof(struct virgl_hw_buf*));
      if (!new_re_bo) {
          fprintf(stderr,"failure to add relocation %d, %d\n", cbuf->cres, cbuf->nres);
          return;
      }

      cbuf->res_bo = new_re_bo;
      cbuf->nres = new_nres;
   }

   cbuf->res_bo[cbuf->cres] = NULL;
   virgl_vtest_resource_reference(vtws, &cbuf->res_bo[cbuf->cres], res);
   cbuf->is_handle_added[hash] = TRUE;

   cbuf->reloc_indices_hashlist[hash] = cbuf->cres;
   p_atomic_inc(&res->num_cs_references);
   cbuf->cres++;
}

static int virgl_vtest_winsys_submit_cmd(struct virgl_winsys *vws,
                                         struct virgl_cmd_buf *_cbuf)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);
   struct virgl_vtest_cmd_buf *cbuf = virgl_vtest_cmd_buf(_cbuf);
   int ret;

   if (cbuf->base.cdw == 0)
      return 0;

   ret = virgl_vtest_submit_cmd(vtws, cbuf);

   virgl_vtest_release_all_res(vtws, cbuf);
   memset(cbuf->is_handle_added, 0, sizeof(cbuf->is_handle_added));
   cbuf->base.cdw = 0;
   return ret;
}

static void virgl_vtest_emit_res(struct virgl_winsys *vws,
                                 struct virgl_cmd_buf *_cbuf,
                                 struct virgl_hw_res *res, boolean write_buf)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);
   struct virgl_vtest_cmd_buf *cbuf = virgl_vtest_cmd_buf(_cbuf);
   boolean already_in_list = virgl_vtest_lookup_res(cbuf, res);

   if (write_buf)
      cbuf->base.buf[cbuf->base.cdw++] = res->res_handle;
   if (!already_in_list)
      virgl_vtest_add_res(vtws, cbuf, res);
}

static boolean virgl_vtest_res_is_ref(struct virgl_winsys *vws,
                                      struct virgl_cmd_buf *_cbuf,
                                      struct virgl_hw_res *res)
{
   if (!res->num_cs_references)
      return FALSE;

   return TRUE;
}

static int virgl_vtest_get_caps(struct virgl_winsys *vws,
                                struct virgl_drm_caps *caps)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);

   virgl_ws_fill_new_caps_defaults(caps);
   return virgl_vtest_send_get_caps(vtws, caps);
}

static struct pipe_fence_handle *
virgl_cs_create_fence(struct virgl_winsys *vws)
{
   struct virgl_hw_res *res;

   res = virgl_vtest_winsys_resource_cache_create(vws,
                                                PIPE_BUFFER,
                                                PIPE_FORMAT_R8_UNORM,
                                                VIRGL_BIND_CUSTOM,
                                                8, 1, 1, 0, 0, 0, 8);

   return (struct pipe_fence_handle *)res;
}

static bool virgl_fence_wait(struct virgl_winsys *vws,
                             struct pipe_fence_handle *fence,
                             uint64_t timeout)
{
   struct virgl_vtest_winsys *vdws = virgl_vtest_winsys(vws);
   struct virgl_hw_res *res = virgl_hw_res(fence);

   if (timeout == 0)
      return !virgl_vtest_resource_is_busy(vdws, res);

   if (timeout != PIPE_TIMEOUT_INFINITE) {
      int64_t start_time = os_time_get();
      timeout /= 1000;
      while (virgl_vtest_resource_is_busy(vdws, res)) {
         if (os_time_get() - start_time >= timeout)
            return FALSE;
         os_time_sleep(10);
      }
      return TRUE;
   }
   virgl_vtest_resource_wait(vws, res);
   return TRUE;
}

static void virgl_fence_reference(struct virgl_winsys *vws,
                                  struct pipe_fence_handle **dst,
                                  struct pipe_fence_handle *src)
{
   struct virgl_vtest_winsys *vdws = virgl_vtest_winsys(vws);
   virgl_vtest_resource_reference(vdws, (struct virgl_hw_res **)dst,
                                  virgl_hw_res(src));
}

static void virgl_vtest_flush_frontbuffer(struct virgl_winsys *vws,
                                          struct virgl_hw_res *res,
                                          unsigned level, unsigned layer,
                                          void *winsys_drawable_handle,
                                          struct pipe_box *sub_box)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);
   struct pipe_box box;
   void *map;
   uint32_t size;
   uint32_t offset = 0, valid_stride;
   if (!res->dt)
      return;

   memset(&box, 0, sizeof(box));

   if (sub_box) {
      box = *sub_box;
      offset = box.y / util_format_get_blockheight(res->format) * res->stride +
               box.x / util_format_get_blockwidth(res->format) * util_format_get_blocksize(res->format);
   } else {
      box.z = layer;
      box.width = res->width;
      box.height = res->height;
      box.depth = 1;
   }

   size = vtest_get_transfer_size(res, &box, res->stride, 0, level, &valid_stride);

   virgl_vtest_busy_wait(vtws, res->res_handle, VCMD_BUSY_WAIT_FLAG_WAIT);
   map = vtws->sws->displaytarget_map(vtws->sws, res->dt, 0);

   /* execute a transfer */
   virgl_vtest_send_transfer_get(vtws, res->res_handle,
                                 level, res->stride, 0, &box, size, offset);

   /* This functions gets the resource from the hardware backend that may have
    * a hardware imposed stride that is different from the IOV stride used to
    * get the data. */
   virgl_vtest_recv_transfer_get_data(vtws, map + offset, size, valid_stride,
                                      &box, res->format,
                                      vtws->protocol_version == 0 ? valid_stride : util_format_get_stride(res->format, res->width));

   vtws->sws->displaytarget_unmap(vtws->sws, res->dt);

   vtws->sws->displaytarget_display(vtws->sws, res->dt, winsys_drawable_handle,
                                    sub_box);
}

static void
virgl_vtest_winsys_destroy(struct virgl_winsys *vws)
{
   struct virgl_vtest_winsys *vtws = virgl_vtest_winsys(vws);

   virgl_cache_flush(vtws);

   mtx_destroy(&vtws->mutex);
   FREE(vtws);
}

struct virgl_winsys *
virgl_vtest_winsys_wrap(struct sw_winsys *sws)
{
   struct virgl_vtest_winsys *vtws;

   vtws = CALLOC_STRUCT(virgl_vtest_winsys);
   if (!vtws)
      return NULL;

   virgl_vtest_connect(vtws);
   vtws->sws = sws;

   vtws->usecs = 1000000;
   LIST_INITHEAD(&vtws->delayed);
   (void) mtx_init(&vtws->mutex, mtx_plain);

   vtws->base.destroy = virgl_vtest_winsys_destroy;

   vtws->base.transfer_put = virgl_vtest_transfer_put;
   vtws->base.transfer_get = virgl_vtest_transfer_get;

   vtws->base.resource_create = virgl_vtest_winsys_resource_cache_create;
   vtws->base.resource_unref = virgl_vtest_winsys_resource_unref;
   vtws->base.resource_map = virgl_vtest_resource_map;
   vtws->base.resource_wait = virgl_vtest_resource_wait;
   vtws->base.cmd_buf_create = virgl_vtest_cmd_buf_create;
   vtws->base.cmd_buf_destroy = virgl_vtest_cmd_buf_destroy;
   vtws->base.submit_cmd = virgl_vtest_winsys_submit_cmd;

   vtws->base.emit_res = virgl_vtest_emit_res;
   vtws->base.res_is_referenced = virgl_vtest_res_is_ref;
   vtws->base.get_caps = virgl_vtest_get_caps;

   vtws->base.cs_create_fence = virgl_cs_create_fence;
   vtws->base.fence_wait = virgl_fence_wait;
   vtws->base.fence_reference = virgl_fence_reference;

   vtws->base.flush_frontbuffer = virgl_vtest_flush_frontbuffer;

   return &vtws->base;
}
