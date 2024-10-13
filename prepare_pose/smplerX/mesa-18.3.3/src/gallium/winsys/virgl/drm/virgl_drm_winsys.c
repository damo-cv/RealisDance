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

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/stat.h>

#include "os/os_mman.h"
#include "util/os_time.h"
#include "util/u_memory.h"
#include "util/u_format.h"
#include "util/u_hash_table.h"
#include "util/u_inlines.h"
#include "state_tracker/drm_driver.h"
#include "virgl/virgl_screen.h"
#include "virgl/virgl_public.h"

#include <xf86drm.h>
#include "virtgpu_drm.h"

#include "virgl_drm_winsys.h"
#include "virgl_drm_public.h"

static inline boolean can_cache_resource(struct virgl_hw_res *res)
{
   return res->cacheable == TRUE;
}

static void virgl_hw_res_destroy(struct virgl_drm_winsys *qdws,
                                 struct virgl_hw_res *res)
{
      struct drm_gem_close args;

      if (res->flinked) {
         mtx_lock(&qdws->bo_handles_mutex);
         util_hash_table_remove(qdws->bo_names,
                                (void *)(uintptr_t)res->flink);
         mtx_unlock(&qdws->bo_handles_mutex);
      }

      if (res->bo_handle) {
         mtx_lock(&qdws->bo_handles_mutex);
         util_hash_table_remove(qdws->bo_handles,
                                (void *)(uintptr_t)res->bo_handle);
         mtx_unlock(&qdws->bo_handles_mutex);
      }

      if (res->ptr)
         os_munmap(res->ptr, res->size);

      memset(&args, 0, sizeof(args));
      args.handle = res->bo_handle;
      drmIoctl(qdws->fd, DRM_IOCTL_GEM_CLOSE, &args);
      FREE(res);
}

static boolean virgl_drm_resource_is_busy(struct virgl_drm_winsys *qdws,
                                          struct virgl_hw_res *res)
{
   struct drm_virtgpu_3d_wait waitcmd;
   int ret;

   memset(&waitcmd, 0, sizeof(waitcmd));
   waitcmd.handle = res->bo_handle;
   waitcmd.flags = VIRTGPU_WAIT_NOWAIT;

   ret = drmIoctl(qdws->fd, DRM_IOCTL_VIRTGPU_WAIT, &waitcmd);
   if (ret && errno == EBUSY)
      return TRUE;
   return FALSE;
}

static void
virgl_cache_flush(struct virgl_drm_winsys *qdws)
{
   struct list_head *curr, *next;
   struct virgl_hw_res *res;

   mtx_lock(&qdws->mutex);
   curr = qdws->delayed.next;
   next = curr->next;

   while (curr != &qdws->delayed) {
      res = LIST_ENTRY(struct virgl_hw_res, curr, head);
      LIST_DEL(&res->head);
      virgl_hw_res_destroy(qdws, res);
      curr = next;
      next = curr->next;
   }
   mtx_unlock(&qdws->mutex);
}
static void
virgl_drm_winsys_destroy(struct virgl_winsys *qws)
{
   struct virgl_drm_winsys *qdws = virgl_drm_winsys(qws);

   virgl_cache_flush(qdws);

   util_hash_table_destroy(qdws->bo_handles);
   util_hash_table_destroy(qdws->bo_names);
   mtx_destroy(&qdws->bo_handles_mutex);
   mtx_destroy(&qdws->mutex);

   FREE(qdws);
}

static void
virgl_cache_list_check_free(struct virgl_drm_winsys *qdws)
{
   struct list_head *curr, *next;
   struct virgl_hw_res *res;
   int64_t now;

   now = os_time_get();
   curr = qdws->delayed.next;
   next = curr->next;
   while (curr != &qdws->delayed) {
      res = LIST_ENTRY(struct virgl_hw_res, curr, head);
      if (!os_time_timeout(res->start, res->end, now))
         break;

      LIST_DEL(&res->head);
      virgl_hw_res_destroy(qdws, res);
      curr = next;
      next = curr->next;
   }
}

static void virgl_drm_resource_reference(struct virgl_drm_winsys *qdws,
                                       struct virgl_hw_res **dres,
                                       struct virgl_hw_res *sres)
{
   struct virgl_hw_res *old = *dres;
   if (pipe_reference(&(*dres)->reference, &sres->reference)) {

      if (!can_cache_resource(old)) {
         virgl_hw_res_destroy(qdws, old);
      } else {
         mtx_lock(&qdws->mutex);
         virgl_cache_list_check_free(qdws);

         old->start = os_time_get();
         old->end = old->start + qdws->usecs;
         LIST_ADDTAIL(&old->head, &qdws->delayed);
         qdws->num_delayed++;
         mtx_unlock(&qdws->mutex);
      }
   }
   *dres = sres;
}

static struct virgl_hw_res *
virgl_drm_winsys_resource_create(struct virgl_winsys *qws,
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
   struct virgl_drm_winsys *qdws = virgl_drm_winsys(qws);
   struct drm_virtgpu_resource_create createcmd;
   int ret;
   struct virgl_hw_res *res;
   uint32_t stride = width * util_format_get_blocksize(format);

   res = CALLOC_STRUCT(virgl_hw_res);
   if (!res)
      return NULL;

   memset(&createcmd, 0, sizeof(createcmd));
   createcmd.target = target;
   createcmd.format = format;
   createcmd.bind = bind;
   createcmd.width = width;
   createcmd.height = height;
   createcmd.depth = depth;
   createcmd.array_size = array_size;
   createcmd.last_level = last_level;
   createcmd.nr_samples = nr_samples;
   createcmd.stride = stride;
   createcmd.size = size;

   ret = drmIoctl(qdws->fd, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE, &createcmd);
   if (ret != 0) {
      FREE(res);
      return NULL;
   }

   res->bind = bind;
   res->format = format;

   res->res_handle = createcmd.res_handle;
   res->bo_handle = createcmd.bo_handle;
   res->size = size;
   res->stride = stride;
   pipe_reference_init(&res->reference, 1);
   res->num_cs_references = 0;
   return res;
}

static inline int virgl_is_res_compat(struct virgl_drm_winsys *qdws,
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

   if (virgl_drm_resource_is_busy(qdws, res)) {
      return -1;
   }

   return 1;
}

static int
virgl_bo_transfer_put(struct virgl_winsys *vws,
                      struct virgl_hw_res *res,
                      const struct pipe_box *box,
                      uint32_t stride, uint32_t layer_stride,
                      uint32_t buf_offset, uint32_t level)
{
   struct virgl_drm_winsys *vdws = virgl_drm_winsys(vws);
   struct drm_virtgpu_3d_transfer_to_host tohostcmd;

   memset(&tohostcmd, 0, sizeof(tohostcmd));
   tohostcmd.bo_handle = res->bo_handle;
   tohostcmd.box.x = box->x;
   tohostcmd.box.y = box->y;
   tohostcmd.box.z = box->z;
   tohostcmd.box.w = box->width;
   tohostcmd.box.h = box->height;
   tohostcmd.box.d = box->depth;
   tohostcmd.offset = buf_offset;
   tohostcmd.level = level;
  // tohostcmd.stride = stride;
  // tohostcmd.layer_stride = stride;
   return drmIoctl(vdws->fd, DRM_IOCTL_VIRTGPU_TRANSFER_TO_HOST, &tohostcmd);
}

static int
virgl_bo_transfer_get(struct virgl_winsys *vws,
                      struct virgl_hw_res *res,
                      const struct pipe_box *box,
                      uint32_t stride, uint32_t layer_stride,
                      uint32_t buf_offset, uint32_t level)
{
   struct virgl_drm_winsys *vdws = virgl_drm_winsys(vws);
   struct drm_virtgpu_3d_transfer_from_host fromhostcmd;

   memset(&fromhostcmd, 0, sizeof(fromhostcmd));
   fromhostcmd.bo_handle = res->bo_handle;
   fromhostcmd.level = level;
   fromhostcmd.offset = buf_offset;
  // fromhostcmd.stride = stride;
  // fromhostcmd.layer_stride = layer_stride;
   fromhostcmd.box.x = box->x;
   fromhostcmd.box.y = box->y;
   fromhostcmd.box.z = box->z;
   fromhostcmd.box.w = box->width;
   fromhostcmd.box.h = box->height;
   fromhostcmd.box.d = box->depth;
   return drmIoctl(vdws->fd, DRM_IOCTL_VIRTGPU_TRANSFER_FROM_HOST, &fromhostcmd);
}

static struct virgl_hw_res *
virgl_drm_winsys_resource_cache_create(struct virgl_winsys *qws,
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
   struct virgl_drm_winsys *qdws = virgl_drm_winsys(qws);
   struct virgl_hw_res *res, *curr_res;
   struct list_head *curr, *next;
   int64_t now;
   int ret = 0;

   /* only store binds for vertex/index/const buffers */
   if (bind != VIRGL_BIND_CONSTANT_BUFFER && bind != VIRGL_BIND_INDEX_BUFFER &&
       bind != VIRGL_BIND_VERTEX_BUFFER && bind != VIRGL_BIND_CUSTOM)
      goto alloc;

   mtx_lock(&qdws->mutex);

   res = NULL;
   curr = qdws->delayed.next;
   next = curr->next;

   now = os_time_get();
   while (curr != &qdws->delayed) {
      curr_res = LIST_ENTRY(struct virgl_hw_res, curr, head);

      if (!res && ((ret = virgl_is_res_compat(qdws, curr_res, size, bind, format)) > 0))
         res = curr_res;
      else if (os_time_timeout(curr_res->start, curr_res->end, now)) {
         LIST_DEL(&curr_res->head);
         virgl_hw_res_destroy(qdws, curr_res);
      } else
         break;

      if (ret == -1)
         break;

      curr = next;
      next = curr->next;
   }

   if (!res && ret != -1) {
      while (curr != &qdws->delayed) {
         curr_res = LIST_ENTRY(struct virgl_hw_res, curr, head);
         ret = virgl_is_res_compat(qdws, curr_res, size, bind, format);
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
      --qdws->num_delayed;
      mtx_unlock(&qdws->mutex);
      pipe_reference_init(&res->reference, 1);
      return res;
   }

   mtx_unlock(&qdws->mutex);

alloc:
   res = virgl_drm_winsys_resource_create(qws, target, format, bind,
                                           width, height, depth, array_size,
                                           last_level, nr_samples, size);
   if (bind == VIRGL_BIND_CONSTANT_BUFFER || bind == VIRGL_BIND_INDEX_BUFFER ||
       bind == VIRGL_BIND_VERTEX_BUFFER)
      res->cacheable = TRUE;
   return res;
}

static struct virgl_hw_res *
virgl_drm_winsys_resource_create_handle(struct virgl_winsys *qws,
                                        struct winsys_handle *whandle)
{
   struct virgl_drm_winsys *qdws = virgl_drm_winsys(qws);
   struct drm_gem_open open_arg = {};
   struct drm_virtgpu_resource_info info_arg = {};
   struct virgl_hw_res *res;
   uint32_t handle = whandle->handle;

   if (whandle->offset != 0) {
      fprintf(stderr, "attempt to import unsupported winsys offset %u\n",
              whandle->offset);
      return NULL;
   }

   mtx_lock(&qdws->bo_handles_mutex);

   if (whandle->type == WINSYS_HANDLE_TYPE_SHARED) {
      res = util_hash_table_get(qdws->bo_names, (void*)(uintptr_t)handle);
      if (res) {
         struct virgl_hw_res *r = NULL;
         virgl_drm_resource_reference(qdws, &r, res);
         goto done;
      }
   }

   if (whandle->type == WINSYS_HANDLE_TYPE_FD) {
      int r;
      r = drmPrimeFDToHandle(qdws->fd, whandle->handle, &handle);
      if (r) {
         res = NULL;
         goto done;
      }
   }

   res = util_hash_table_get(qdws->bo_handles, (void*)(uintptr_t)handle);
   if (res) {
      struct virgl_hw_res *r = NULL;
      virgl_drm_resource_reference(qdws, &r, res);
      goto done;
   }

   res = CALLOC_STRUCT(virgl_hw_res);
   if (!res)
      goto done;

   if (whandle->type == WINSYS_HANDLE_TYPE_FD) {
      res->bo_handle = handle;
   } else {
      memset(&open_arg, 0, sizeof(open_arg));
      open_arg.name = whandle->handle;
      if (drmIoctl(qdws->fd, DRM_IOCTL_GEM_OPEN, &open_arg)) {
         FREE(res);
         res = NULL;
         goto done;
      }
      res->bo_handle = open_arg.handle;
   }
   res->name = handle;

   memset(&info_arg, 0, sizeof(info_arg));
   info_arg.bo_handle = res->bo_handle;

   if (drmIoctl(qdws->fd, DRM_IOCTL_VIRTGPU_RESOURCE_INFO, &info_arg)) {
      /* close */
      FREE(res);
      res = NULL;
      goto done;
   }

   res->res_handle = info_arg.res_handle;

   res->size = info_arg.size;
   res->stride = info_arg.stride;
   pipe_reference_init(&res->reference, 1);
   res->num_cs_references = 0;

   util_hash_table_set(qdws->bo_handles, (void *)(uintptr_t)handle, res);

done:
   mtx_unlock(&qdws->bo_handles_mutex);
   return res;
}

static boolean virgl_drm_winsys_resource_get_handle(struct virgl_winsys *qws,
                                                    struct virgl_hw_res *res,
                                                    uint32_t stride,
                                                    struct winsys_handle *whandle)
 {
   struct virgl_drm_winsys *qdws = virgl_drm_winsys(qws);
   struct drm_gem_flink flink;

   if (!res)
       return FALSE;

   if (whandle->type == WINSYS_HANDLE_TYPE_SHARED) {
      if (!res->flinked) {
         memset(&flink, 0, sizeof(flink));
         flink.handle = res->bo_handle;

         if (drmIoctl(qdws->fd, DRM_IOCTL_GEM_FLINK, &flink)) {
            return FALSE;
         }
         res->flinked = TRUE;
         res->flink = flink.name;

         mtx_lock(&qdws->bo_handles_mutex);
         util_hash_table_set(qdws->bo_names, (void *)(uintptr_t)res->flink, res);
         mtx_unlock(&qdws->bo_handles_mutex);
      }
      whandle->handle = res->flink;
   } else if (whandle->type == WINSYS_HANDLE_TYPE_KMS) {
      whandle->handle = res->bo_handle;
   } else if (whandle->type == WINSYS_HANDLE_TYPE_FD) {
      if (drmPrimeHandleToFD(qdws->fd, res->bo_handle, DRM_CLOEXEC, (int*)&whandle->handle))
            return FALSE;
      mtx_lock(&qdws->bo_handles_mutex);
      util_hash_table_set(qdws->bo_handles, (void *)(uintptr_t)res->bo_handle, res);
      mtx_unlock(&qdws->bo_handles_mutex);
   }
   whandle->stride = stride;
   return TRUE;
}

static void virgl_drm_winsys_resource_unref(struct virgl_winsys *qws,
                                            struct virgl_hw_res *hres)
{
   struct virgl_drm_winsys *qdws = virgl_drm_winsys(qws);

   virgl_drm_resource_reference(qdws, &hres, NULL);
}

static void *virgl_drm_resource_map(struct virgl_winsys *qws,
                                    struct virgl_hw_res *res)
{
   struct virgl_drm_winsys *qdws = virgl_drm_winsys(qws);
   struct drm_virtgpu_map mmap_arg;
   void *ptr;

   if (res->ptr)
      return res->ptr;

   memset(&mmap_arg, 0, sizeof(mmap_arg));
   mmap_arg.handle = res->bo_handle;
   if (drmIoctl(qdws->fd, DRM_IOCTL_VIRTGPU_MAP, &mmap_arg))
      return NULL;

   ptr = os_mmap(0, res->size, PROT_READ|PROT_WRITE, MAP_SHARED,
                 qdws->fd, mmap_arg.offset);
   if (ptr == MAP_FAILED)
      return NULL;

   res->ptr = ptr;
   return ptr;

}

static void virgl_drm_resource_wait(struct virgl_winsys *qws,
                                    struct virgl_hw_res *res)
{
   struct virgl_drm_winsys *qdws = virgl_drm_winsys(qws);
   struct drm_virtgpu_3d_wait waitcmd;
   int ret;

   memset(&waitcmd, 0, sizeof(waitcmd));
   waitcmd.handle = res->bo_handle;
 again:
   ret = drmIoctl(qdws->fd, DRM_IOCTL_VIRTGPU_WAIT, &waitcmd);
   if (ret == -EAGAIN)
      goto again;
}

static struct virgl_cmd_buf *virgl_drm_cmd_buf_create(struct virgl_winsys *qws)
{
   struct virgl_drm_cmd_buf *cbuf;

   cbuf = CALLOC_STRUCT(virgl_drm_cmd_buf);
   if (!cbuf)
      return NULL;

   cbuf->ws = qws;

   cbuf->nres = 512;
   cbuf->res_bo = CALLOC(cbuf->nres, sizeof(struct virgl_hw_buf*));
   if (!cbuf->res_bo) {
      FREE(cbuf);
      return NULL;
   }
   cbuf->res_hlist = MALLOC(cbuf->nres * sizeof(uint32_t));
   if (!cbuf->res_hlist) {
      FREE(cbuf->res_bo);
      FREE(cbuf);
      return NULL;
   }

   cbuf->base.buf = cbuf->buf;
   return &cbuf->base;
}

static void virgl_drm_cmd_buf_destroy(struct virgl_cmd_buf *_cbuf)
{
   struct virgl_drm_cmd_buf *cbuf = virgl_drm_cmd_buf(_cbuf);

   FREE(cbuf->res_hlist);
   FREE(cbuf->res_bo);
   FREE(cbuf);

}

static boolean virgl_drm_lookup_res(struct virgl_drm_cmd_buf *cbuf,
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

static void virgl_drm_add_res(struct virgl_drm_winsys *qdws,
                              struct virgl_drm_cmd_buf *cbuf,
                              struct virgl_hw_res *res)
{
   unsigned hash = res->res_handle & (sizeof(cbuf->is_handle_added)-1);

   if (cbuf->cres >= cbuf->nres) {
      unsigned new_nres = cbuf->nres + 256;
      void *new_ptr = REALLOC(cbuf->res_bo,
                              cbuf->nres * sizeof(struct virgl_hw_buf*),
                              new_nres * sizeof(struct virgl_hw_buf*));
      if (!new_ptr) {
          fprintf(stderr,"failure to add relocation %d, %d\n", cbuf->cres, new_nres);
          return;
      }
      cbuf->res_bo = new_ptr;

      new_ptr = REALLOC(cbuf->res_hlist,
                        cbuf->nres * sizeof(uint32_t),
                        new_nres * sizeof(uint32_t));
      if (!new_ptr) {
          fprintf(stderr,"failure to add hlist relocation %d, %d\n", cbuf->cres, cbuf->nres);
          return;
      }
      cbuf->res_hlist = new_ptr;
      cbuf->nres = new_nres;
   }

   cbuf->res_bo[cbuf->cres] = NULL;
   virgl_drm_resource_reference(qdws, &cbuf->res_bo[cbuf->cres], res);
   cbuf->res_hlist[cbuf->cres] = res->bo_handle;
   cbuf->is_handle_added[hash] = TRUE;

   cbuf->reloc_indices_hashlist[hash] = cbuf->cres;
   p_atomic_inc(&res->num_cs_references);
   cbuf->cres++;
}

static void virgl_drm_release_all_res(struct virgl_drm_winsys *qdws,
                                      struct virgl_drm_cmd_buf *cbuf)
{
   int i;

   for (i = 0; i < cbuf->cres; i++) {
      p_atomic_dec(&cbuf->res_bo[i]->num_cs_references);
      virgl_drm_resource_reference(qdws, &cbuf->res_bo[i], NULL);
   }
   cbuf->cres = 0;
}

static void virgl_drm_emit_res(struct virgl_winsys *qws,
                               struct virgl_cmd_buf *_cbuf,
                               struct virgl_hw_res *res, boolean write_buf)
{
   struct virgl_drm_winsys *qdws = virgl_drm_winsys(qws);
   struct virgl_drm_cmd_buf *cbuf = virgl_drm_cmd_buf(_cbuf);
   boolean already_in_list = virgl_drm_lookup_res(cbuf, res);

   if (write_buf)
      cbuf->base.buf[cbuf->base.cdw++] = res->res_handle;

   if (!already_in_list)
      virgl_drm_add_res(qdws, cbuf, res);
}

static boolean virgl_drm_res_is_ref(struct virgl_winsys *qws,
                                    struct virgl_cmd_buf *_cbuf,
                                    struct virgl_hw_res *res)
{
   if (!res->num_cs_references)
      return FALSE;

   return TRUE;
}

static int virgl_drm_winsys_submit_cmd(struct virgl_winsys *qws,
                                       struct virgl_cmd_buf *_cbuf)
{
   struct virgl_drm_winsys *qdws = virgl_drm_winsys(qws);
   struct virgl_drm_cmd_buf *cbuf = virgl_drm_cmd_buf(_cbuf);
   struct drm_virtgpu_execbuffer eb;
   int ret;

   if (cbuf->base.cdw == 0)
      return 0;

   memset(&eb, 0, sizeof(struct drm_virtgpu_execbuffer));
   eb.command = (unsigned long)(void*)cbuf->buf;
   eb.size = cbuf->base.cdw * 4;
   eb.num_bo_handles = cbuf->cres;
   eb.bo_handles = (unsigned long)(void *)cbuf->res_hlist;

   ret = drmIoctl(qdws->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &eb);
   if (ret == -1)
      fprintf(stderr,"got error from kernel - expect bad rendering %d\n", errno);
   cbuf->base.cdw = 0;

   virgl_drm_release_all_res(qdws, cbuf);

   memset(cbuf->is_handle_added, 0, sizeof(cbuf->is_handle_added));
   return ret;
}

static int virgl_drm_get_caps(struct virgl_winsys *vws,
                              struct virgl_drm_caps *caps)
{
   struct virgl_drm_winsys *vdws = virgl_drm_winsys(vws);
   struct drm_virtgpu_get_caps args;
   int ret;

   virgl_ws_fill_new_caps_defaults(caps);

   memset(&args, 0, sizeof(args));
   if (vdws->has_capset_query_fix) {
      /* if we have the query fix - try and get cap set id 2 first */
      args.cap_set_id = 2;
      args.size = sizeof(union virgl_caps);
   } else {
      args.cap_set_id = 1;
      args.size = sizeof(struct virgl_caps_v1);
   }
   args.addr = (unsigned long)&caps->caps;

   ret = drmIoctl(vdws->fd, DRM_IOCTL_VIRTGPU_GET_CAPS, &args);
   if (ret == -1 && errno == EINVAL) {
      /* Fallback to v1 */
      args.cap_set_id = 1;
      args.size = sizeof(struct virgl_caps_v1);
      ret = drmIoctl(vdws->fd, DRM_IOCTL_VIRTGPU_GET_CAPS, &args);
      if (ret == -1)
          return ret;
   }
   return ret;
}

#define PTR_TO_UINT(x) ((unsigned)((intptr_t)(x)))

static unsigned handle_hash(void *key)
{
    return PTR_TO_UINT(key);
}

static int handle_compare(void *key1, void *key2)
{
    return PTR_TO_UINT(key1) != PTR_TO_UINT(key2);
}

static struct pipe_fence_handle *
virgl_cs_create_fence(struct virgl_winsys *vws)
{
   struct virgl_hw_res *res;

   res = virgl_drm_winsys_resource_cache_create(vws,
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
   struct virgl_drm_winsys *vdws = virgl_drm_winsys(vws);
   struct virgl_hw_res *res = virgl_hw_res(fence);

   if (timeout == 0)
      return !virgl_drm_resource_is_busy(vdws, res);

   if (timeout != PIPE_TIMEOUT_INFINITE) {
      int64_t start_time = os_time_get();
      timeout /= 1000;
      while (virgl_drm_resource_is_busy(vdws, res)) {
         if (os_time_get() - start_time >= timeout)
            return FALSE;
         os_time_sleep(10);
      }
      return TRUE;
   }
   virgl_drm_resource_wait(vws, res);
   return TRUE;
}

static void virgl_fence_reference(struct virgl_winsys *vws,
                                  struct pipe_fence_handle **dst,
                                  struct pipe_fence_handle *src)
{
   struct virgl_drm_winsys *vdws = virgl_drm_winsys(vws);
   virgl_drm_resource_reference(vdws, (struct virgl_hw_res **)dst,
                                virgl_hw_res(src));
}


static struct virgl_winsys *
virgl_drm_winsys_create(int drmFD)
{
   struct virgl_drm_winsys *qdws;
   int ret;
   int gl = 0;
   struct drm_virtgpu_getparam getparam = {0};

   getparam.param = VIRTGPU_PARAM_3D_FEATURES;
   getparam.value = (uint64_t)(uintptr_t)&gl;
   ret = drmIoctl(drmFD, DRM_IOCTL_VIRTGPU_GETPARAM, &getparam);
   if (ret < 0 || !gl)
      return NULL;

   qdws = CALLOC_STRUCT(virgl_drm_winsys);
   if (!qdws)
      return NULL;

   qdws->fd = drmFD;
   qdws->num_delayed = 0;
   qdws->usecs = 1000000;
   LIST_INITHEAD(&qdws->delayed);
   (void) mtx_init(&qdws->mutex, mtx_plain);
   (void) mtx_init(&qdws->bo_handles_mutex, mtx_plain);
   qdws->bo_handles = util_hash_table_create(handle_hash, handle_compare);
   qdws->bo_names = util_hash_table_create(handle_hash, handle_compare);
   qdws->base.destroy = virgl_drm_winsys_destroy;

   qdws->base.transfer_put = virgl_bo_transfer_put;
   qdws->base.transfer_get = virgl_bo_transfer_get;
   qdws->base.resource_create = virgl_drm_winsys_resource_cache_create;
   qdws->base.resource_unref = virgl_drm_winsys_resource_unref;
   qdws->base.resource_create_from_handle = virgl_drm_winsys_resource_create_handle;
   qdws->base.resource_get_handle = virgl_drm_winsys_resource_get_handle;
   qdws->base.resource_map = virgl_drm_resource_map;
   qdws->base.resource_wait = virgl_drm_resource_wait;
   qdws->base.cmd_buf_create = virgl_drm_cmd_buf_create;
   qdws->base.cmd_buf_destroy = virgl_drm_cmd_buf_destroy;
   qdws->base.submit_cmd = virgl_drm_winsys_submit_cmd;
   qdws->base.emit_res = virgl_drm_emit_res;
   qdws->base.res_is_referenced = virgl_drm_res_is_ref;

   qdws->base.cs_create_fence = virgl_cs_create_fence;
   qdws->base.fence_wait = virgl_fence_wait;
   qdws->base.fence_reference = virgl_fence_reference;

   qdws->base.get_caps = virgl_drm_get_caps;

   uint32_t value = 0;
   getparam.param = VIRTGPU_PARAM_CAPSET_QUERY_FIX;
   getparam.value = (uint64_t)(uintptr_t)&value;
   ret = drmIoctl(qdws->fd, DRM_IOCTL_VIRTGPU_GETPARAM, &getparam);
   if (ret == 0) {
      if (value == 1)
         qdws->has_capset_query_fix = true;
   }

   return &qdws->base;

}

static struct util_hash_table *fd_tab = NULL;
static mtx_t virgl_screen_mutex = _MTX_INITIALIZER_NP;

static void
virgl_drm_screen_destroy(struct pipe_screen *pscreen)
{
   struct virgl_screen *screen = virgl_screen(pscreen);
   boolean destroy;

   mtx_lock(&virgl_screen_mutex);
   destroy = --screen->refcnt == 0;
   if (destroy) {
      int fd = virgl_drm_winsys(screen->vws)->fd;
      util_hash_table_remove(fd_tab, intptr_to_pointer(fd));
   }
   mtx_unlock(&virgl_screen_mutex);

   if (destroy) {
      pscreen->destroy = screen->winsys_priv;
      pscreen->destroy(pscreen);
   }
}

static unsigned hash_fd(void *key)
{
   int fd = pointer_to_intptr(key);
   struct stat stat;
   fstat(fd, &stat);

   return stat.st_dev ^ stat.st_ino ^ stat.st_rdev;
}

static int compare_fd(void *key1, void *key2)
{
   int fd1 = pointer_to_intptr(key1);
   int fd2 = pointer_to_intptr(key2);
   struct stat stat1, stat2;
   fstat(fd1, &stat1);
   fstat(fd2, &stat2);

   return stat1.st_dev != stat2.st_dev ||
         stat1.st_ino != stat2.st_ino ||
         stat1.st_rdev != stat2.st_rdev;
}

struct pipe_screen *
virgl_drm_screen_create(int fd)
{
   struct pipe_screen *pscreen = NULL;

   mtx_lock(&virgl_screen_mutex);
   if (!fd_tab) {
      fd_tab = util_hash_table_create(hash_fd, compare_fd);
      if (!fd_tab)
         goto unlock;
   }

   pscreen = util_hash_table_get(fd_tab, intptr_to_pointer(fd));
   if (pscreen) {
      virgl_screen(pscreen)->refcnt++;
   } else {
      struct virgl_winsys *vws;
      int dup_fd = fcntl(fd, F_DUPFD_CLOEXEC, 3);

      vws = virgl_drm_winsys_create(dup_fd);
      if (!vws) {
         close(dup_fd);
         goto unlock;
      }

      pscreen = virgl_create_screen(vws);
      if (pscreen) {
         util_hash_table_set(fd_tab, intptr_to_pointer(dup_fd), pscreen);

         /* Bit of a hack, to avoid circular linkage dependency,
          * ie. pipe driver having to call in to winsys, we
          * override the pipe drivers screen->destroy():
          */
         virgl_screen(pscreen)->winsys_priv = pscreen->destroy;
         pscreen->destroy = virgl_drm_screen_destroy;
      }
   }

unlock:
   mtx_unlock(&virgl_screen_mutex);
   return pscreen;
}
