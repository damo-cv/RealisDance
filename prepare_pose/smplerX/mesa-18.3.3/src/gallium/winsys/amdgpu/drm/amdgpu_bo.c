/*
 * Copyright © 2011 Marek Olšák <maraeo@gmail.com>
 * Copyright © 2015 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS, AUTHORS
 * AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 */

#include "amdgpu_cs.h"

#include "util/os_time.h"
#include "util/u_hash_table.h"
#include "state_tracker/drm_driver.h"
#include <amdgpu_drm.h>
#include <xf86drm.h>
#include <stdio.h>
#include <inttypes.h>

#ifndef AMDGPU_GEM_CREATE_VM_ALWAYS_VALID
#define AMDGPU_GEM_CREATE_VM_ALWAYS_VALID (1 << 6)
#endif

#ifndef AMDGPU_VA_RANGE_HIGH
#define AMDGPU_VA_RANGE_HIGH	0x2
#endif

/* Set to 1 for verbose output showing committed sparse buffer ranges. */
#define DEBUG_SPARSE_COMMITS 0

struct amdgpu_sparse_backing_chunk {
   uint32_t begin, end;
};

static struct pb_buffer *
amdgpu_bo_create(struct radeon_winsys *rws,
                 uint64_t size,
                 unsigned alignment,
                 enum radeon_bo_domain domain,
                 enum radeon_bo_flag flags);

static bool amdgpu_bo_wait(struct pb_buffer *_buf, uint64_t timeout,
                           enum radeon_bo_usage usage)
{
   struct amdgpu_winsys_bo *bo = amdgpu_winsys_bo(_buf);
   struct amdgpu_winsys *ws = bo->ws;
   int64_t abs_timeout;

   if (timeout == 0) {
      if (p_atomic_read(&bo->num_active_ioctls))
         return false;

   } else {
      abs_timeout = os_time_get_absolute_timeout(timeout);

      /* Wait if any ioctl is being submitted with this buffer. */
      if (!os_wait_until_zero_abs_timeout(&bo->num_active_ioctls, abs_timeout))
         return false;
   }

   if (bo->is_shared) {
      /* We can't use user fences for shared buffers, because user fences
       * are local to this process only. If we want to wait for all buffer
       * uses in all processes, we have to use amdgpu_bo_wait_for_idle.
       */
      bool buffer_busy = true;
      int r;

      r = amdgpu_bo_wait_for_idle(bo->bo, timeout, &buffer_busy);
      if (r)
         fprintf(stderr, "%s: amdgpu_bo_wait_for_idle failed %i\n", __func__,
                 r);
      return !buffer_busy;
   }

   if (timeout == 0) {
      unsigned idle_fences;
      bool buffer_idle;

      simple_mtx_lock(&ws->bo_fence_lock);

      for (idle_fences = 0; idle_fences < bo->num_fences; ++idle_fences) {
         if (!amdgpu_fence_wait(bo->fences[idle_fences], 0, false))
            break;
      }

      /* Release the idle fences to avoid checking them again later. */
      for (unsigned i = 0; i < idle_fences; ++i)
         amdgpu_fence_reference(&bo->fences[i], NULL);

      memmove(&bo->fences[0], &bo->fences[idle_fences],
              (bo->num_fences - idle_fences) * sizeof(*bo->fences));
      bo->num_fences -= idle_fences;

      buffer_idle = !bo->num_fences;
      simple_mtx_unlock(&ws->bo_fence_lock);

      return buffer_idle;
   } else {
      bool buffer_idle = true;

      simple_mtx_lock(&ws->bo_fence_lock);
      while (bo->num_fences && buffer_idle) {
         struct pipe_fence_handle *fence = NULL;
         bool fence_idle = false;

         amdgpu_fence_reference(&fence, bo->fences[0]);

         /* Wait for the fence. */
         simple_mtx_unlock(&ws->bo_fence_lock);
         if (amdgpu_fence_wait(fence, abs_timeout, true))
            fence_idle = true;
         else
            buffer_idle = false;
         simple_mtx_lock(&ws->bo_fence_lock);

         /* Release an idle fence to avoid checking it again later, keeping in
          * mind that the fence array may have been modified by other threads.
          */
         if (fence_idle && bo->num_fences && bo->fences[0] == fence) {
            amdgpu_fence_reference(&bo->fences[0], NULL);
            memmove(&bo->fences[0], &bo->fences[1],
                    (bo->num_fences - 1) * sizeof(*bo->fences));
            bo->num_fences--;
         }

         amdgpu_fence_reference(&fence, NULL);
      }
      simple_mtx_unlock(&ws->bo_fence_lock);

      return buffer_idle;
   }
}

static enum radeon_bo_domain amdgpu_bo_get_initial_domain(
      struct pb_buffer *buf)
{
   return ((struct amdgpu_winsys_bo*)buf)->initial_domain;
}

static void amdgpu_bo_remove_fences(struct amdgpu_winsys_bo *bo)
{
   for (unsigned i = 0; i < bo->num_fences; ++i)
      amdgpu_fence_reference(&bo->fences[i], NULL);

   FREE(bo->fences);
   bo->num_fences = 0;
   bo->max_fences = 0;
}

void amdgpu_bo_destroy(struct pb_buffer *_buf)
{
   struct amdgpu_winsys_bo *bo = amdgpu_winsys_bo(_buf);
   struct amdgpu_winsys *ws = bo->ws;

   assert(bo->bo && "must not be called for slab entries");

   if (ws->debug_all_bos) {
      simple_mtx_lock(&ws->global_bo_list_lock);
      LIST_DEL(&bo->u.real.global_list_item);
      ws->num_buffers--;
      simple_mtx_unlock(&ws->global_bo_list_lock);
   }

   simple_mtx_lock(&ws->bo_export_table_lock);
   util_hash_table_remove(ws->bo_export_table, bo->bo);
   simple_mtx_unlock(&ws->bo_export_table_lock);

   amdgpu_bo_va_op(bo->bo, 0, bo->base.size, bo->va, 0, AMDGPU_VA_OP_UNMAP);
   amdgpu_va_range_free(bo->u.real.va_handle);
   amdgpu_bo_free(bo->bo);

   amdgpu_bo_remove_fences(bo);

   if (bo->initial_domain & RADEON_DOMAIN_VRAM)
      ws->allocated_vram -= align64(bo->base.size, ws->info.gart_page_size);
   else if (bo->initial_domain & RADEON_DOMAIN_GTT)
      ws->allocated_gtt -= align64(bo->base.size, ws->info.gart_page_size);

   if (bo->u.real.map_count >= 1) {
      if (bo->initial_domain & RADEON_DOMAIN_VRAM)
         ws->mapped_vram -= bo->base.size;
      else if (bo->initial_domain & RADEON_DOMAIN_GTT)
         ws->mapped_gtt -= bo->base.size;
      ws->num_mapped_buffers--;
   }

   FREE(bo);
}

static void amdgpu_bo_destroy_or_cache(struct pb_buffer *_buf)
{
   struct amdgpu_winsys_bo *bo = amdgpu_winsys_bo(_buf);

   assert(bo->bo); /* slab buffers have a separate vtbl */

   if (bo->u.real.use_reusable_pool)
      pb_cache_add_buffer(&bo->u.real.cache_entry);
   else
      amdgpu_bo_destroy(_buf);
}

static void *amdgpu_bo_map(struct pb_buffer *buf,
                           struct radeon_cmdbuf *rcs,
                           enum pipe_transfer_usage usage)
{
   struct amdgpu_winsys_bo *bo = (struct amdgpu_winsys_bo*)buf;
   struct amdgpu_winsys_bo *real;
   struct amdgpu_cs *cs = (struct amdgpu_cs*)rcs;
   int r;
   void *cpu = NULL;
   uint64_t offset = 0;

   assert(!bo->sparse);

   /* If it's not unsynchronized bo_map, flush CS if needed and then wait. */
   if (!(usage & PIPE_TRANSFER_UNSYNCHRONIZED)) {
      /* DONTBLOCK doesn't make sense with UNSYNCHRONIZED. */
      if (usage & PIPE_TRANSFER_DONTBLOCK) {
         if (!(usage & PIPE_TRANSFER_WRITE)) {
            /* Mapping for read.
             *
             * Since we are mapping for read, we don't need to wait
             * if the GPU is using the buffer for read too
             * (neither one is changing it).
             *
             * Only check whether the buffer is being used for write. */
            if (cs && amdgpu_bo_is_referenced_by_cs_with_usage(cs, bo,
                                                               RADEON_USAGE_WRITE)) {
               cs->flush_cs(cs->flush_data,
			    RADEON_FLUSH_ASYNC_START_NEXT_GFX_IB_NOW, NULL);
               return NULL;
            }

            if (!amdgpu_bo_wait((struct pb_buffer*)bo, 0,
                                RADEON_USAGE_WRITE)) {
               return NULL;
            }
         } else {
            if (cs && amdgpu_bo_is_referenced_by_cs(cs, bo)) {
               cs->flush_cs(cs->flush_data,
			    RADEON_FLUSH_ASYNC_START_NEXT_GFX_IB_NOW, NULL);
               return NULL;
            }

            if (!amdgpu_bo_wait((struct pb_buffer*)bo, 0,
                                RADEON_USAGE_READWRITE)) {
               return NULL;
            }
         }
      } else {
         uint64_t time = os_time_get_nano();

         if (!(usage & PIPE_TRANSFER_WRITE)) {
            /* Mapping for read.
             *
             * Since we are mapping for read, we don't need to wait
             * if the GPU is using the buffer for read too
             * (neither one is changing it).
             *
             * Only check whether the buffer is being used for write. */
            if (cs) {
               if (amdgpu_bo_is_referenced_by_cs_with_usage(cs, bo,
                                                            RADEON_USAGE_WRITE)) {
                  cs->flush_cs(cs->flush_data,
			       RADEON_FLUSH_START_NEXT_GFX_IB_NOW, NULL);
               } else {
                  /* Try to avoid busy-waiting in amdgpu_bo_wait. */
                  if (p_atomic_read(&bo->num_active_ioctls))
                     amdgpu_cs_sync_flush(rcs);
               }
            }

            amdgpu_bo_wait((struct pb_buffer*)bo, PIPE_TIMEOUT_INFINITE,
                           RADEON_USAGE_WRITE);
         } else {
            /* Mapping for write. */
            if (cs) {
               if (amdgpu_bo_is_referenced_by_cs(cs, bo)) {
                  cs->flush_cs(cs->flush_data,
			       RADEON_FLUSH_START_NEXT_GFX_IB_NOW, NULL);
               } else {
                  /* Try to avoid busy-waiting in amdgpu_bo_wait. */
                  if (p_atomic_read(&bo->num_active_ioctls))
                     amdgpu_cs_sync_flush(rcs);
               }
            }

            amdgpu_bo_wait((struct pb_buffer*)bo, PIPE_TIMEOUT_INFINITE,
                           RADEON_USAGE_READWRITE);
         }

         bo->ws->buffer_wait_time += os_time_get_nano() - time;
      }
   }

   /* If the buffer is created from user memory, return the user pointer. */
   if (bo->user_ptr)
      return bo->user_ptr;

   if (bo->bo) {
      real = bo;
   } else {
      real = bo->u.slab.real;
      offset = bo->va - real->va;
   }

   r = amdgpu_bo_cpu_map(real->bo, &cpu);
   if (r) {
      /* Clear the cache and try again. */
      pb_cache_release_all_buffers(&real->ws->bo_cache);
      r = amdgpu_bo_cpu_map(real->bo, &cpu);
      if (r)
         return NULL;
   }

   if (p_atomic_inc_return(&real->u.real.map_count) == 1) {
      if (real->initial_domain & RADEON_DOMAIN_VRAM)
         real->ws->mapped_vram += real->base.size;
      else if (real->initial_domain & RADEON_DOMAIN_GTT)
         real->ws->mapped_gtt += real->base.size;
      real->ws->num_mapped_buffers++;
   }
   return (uint8_t*)cpu + offset;
}

static void amdgpu_bo_unmap(struct pb_buffer *buf)
{
   struct amdgpu_winsys_bo *bo = (struct amdgpu_winsys_bo*)buf;
   struct amdgpu_winsys_bo *real;

   assert(!bo->sparse);

   if (bo->user_ptr)
      return;

   real = bo->bo ? bo : bo->u.slab.real;

   if (p_atomic_dec_zero(&real->u.real.map_count)) {
      if (real->initial_domain & RADEON_DOMAIN_VRAM)
         real->ws->mapped_vram -= real->base.size;
      else if (real->initial_domain & RADEON_DOMAIN_GTT)
         real->ws->mapped_gtt -= real->base.size;
      real->ws->num_mapped_buffers--;
   }

   amdgpu_bo_cpu_unmap(real->bo);
}

static const struct pb_vtbl amdgpu_winsys_bo_vtbl = {
   amdgpu_bo_destroy_or_cache
   /* other functions are never called */
};

static void amdgpu_add_buffer_to_global_list(struct amdgpu_winsys_bo *bo)
{
   struct amdgpu_winsys *ws = bo->ws;

   assert(bo->bo);

   if (ws->debug_all_bos) {
      simple_mtx_lock(&ws->global_bo_list_lock);
      LIST_ADDTAIL(&bo->u.real.global_list_item, &ws->global_bo_list);
      ws->num_buffers++;
      simple_mtx_unlock(&ws->global_bo_list_lock);
   }
}

static struct amdgpu_winsys_bo *amdgpu_create_bo(struct amdgpu_winsys *ws,
                                                 uint64_t size,
                                                 unsigned alignment,
                                                 enum radeon_bo_domain initial_domain,
                                                 unsigned flags,
                                                 int heap)
{
   struct amdgpu_bo_alloc_request request = {0};
   amdgpu_bo_handle buf_handle;
   uint64_t va = 0;
   struct amdgpu_winsys_bo *bo;
   amdgpu_va_handle va_handle;
   unsigned va_gap_size;
   int r;

   /* VRAM or GTT must be specified, but not both at the same time. */
   assert(util_bitcount(initial_domain & RADEON_DOMAIN_VRAM_GTT) == 1);

   bo = CALLOC_STRUCT(amdgpu_winsys_bo);
   if (!bo) {
      return NULL;
   }

   if (heap >= 0) {
      pb_cache_init_entry(&ws->bo_cache, &bo->u.real.cache_entry, &bo->base,
                          heap);
   }
   request.alloc_size = size;
   request.phys_alignment = alignment;

   if (initial_domain & RADEON_DOMAIN_VRAM)
      request.preferred_heap |= AMDGPU_GEM_DOMAIN_VRAM;
   if (initial_domain & RADEON_DOMAIN_GTT)
      request.preferred_heap |= AMDGPU_GEM_DOMAIN_GTT;

   /* Since VRAM and GTT have almost the same performance on APUs, we could
    * just set GTT. However, in order to decrease GTT(RAM) usage, which is
    * shared with the OS, allow VRAM placements too. The idea is not to use
    * VRAM usefully, but to use it so that it's not unused and wasted.
    */
   if (!ws->info.has_dedicated_vram)
      request.preferred_heap |= AMDGPU_GEM_DOMAIN_GTT;

   if (flags & RADEON_FLAG_NO_CPU_ACCESS)
      request.flags |= AMDGPU_GEM_CREATE_NO_CPU_ACCESS;
   if (flags & RADEON_FLAG_GTT_WC)
      request.flags |= AMDGPU_GEM_CREATE_CPU_GTT_USWC;
   if (flags & RADEON_FLAG_NO_INTERPROCESS_SHARING &&
       ws->info.has_local_buffers)
      request.flags |= AMDGPU_GEM_CREATE_VM_ALWAYS_VALID;
   if (ws->zero_all_vram_allocs &&
       (request.preferred_heap & AMDGPU_GEM_DOMAIN_VRAM))
      request.flags |= AMDGPU_GEM_CREATE_VRAM_CLEARED;

   r = amdgpu_bo_alloc(ws->dev, &request, &buf_handle);
   if (r) {
      fprintf(stderr, "amdgpu: Failed to allocate a buffer:\n");
      fprintf(stderr, "amdgpu:    size      : %"PRIu64" bytes\n", size);
      fprintf(stderr, "amdgpu:    alignment : %u bytes\n", alignment);
      fprintf(stderr, "amdgpu:    domains   : %u\n", initial_domain);
      goto error_bo_alloc;
   }

   va_gap_size = ws->check_vm ? MAX2(4 * alignment, 64 * 1024) : 0;
   if (size > ws->info.pte_fragment_size)
	   alignment = MAX2(alignment, ws->info.pte_fragment_size);
   r = amdgpu_va_range_alloc(ws->dev, amdgpu_gpu_va_range_general,
                             size + va_gap_size, alignment, 0, &va, &va_handle,
                             (flags & RADEON_FLAG_32BIT ? AMDGPU_VA_RANGE_32_BIT : 0) |
			     AMDGPU_VA_RANGE_HIGH);
   if (r)
      goto error_va_alloc;

   unsigned vm_flags = AMDGPU_VM_PAGE_READABLE |
                       AMDGPU_VM_PAGE_EXECUTABLE;

   if (!(flags & RADEON_FLAG_READ_ONLY))
       vm_flags |= AMDGPU_VM_PAGE_WRITEABLE;

   r = amdgpu_bo_va_op_raw(ws->dev, buf_handle, 0, size, va, vm_flags,
			   AMDGPU_VA_OP_MAP);
   if (r)
      goto error_va_map;

   pipe_reference_init(&bo->base.reference, 1);
   bo->base.alignment = alignment;
   bo->base.usage = 0;
   bo->base.size = size;
   bo->base.vtbl = &amdgpu_winsys_bo_vtbl;
   bo->ws = ws;
   bo->bo = buf_handle;
   bo->va = va;
   bo->u.real.va_handle = va_handle;
   bo->initial_domain = initial_domain;
   bo->unique_id = __sync_fetch_and_add(&ws->next_bo_unique_id, 1);
   bo->is_local = !!(request.flags & AMDGPU_GEM_CREATE_VM_ALWAYS_VALID);

   if (initial_domain & RADEON_DOMAIN_VRAM)
      ws->allocated_vram += align64(size, ws->info.gart_page_size);
   else if (initial_domain & RADEON_DOMAIN_GTT)
      ws->allocated_gtt += align64(size, ws->info.gart_page_size);

   amdgpu_bo_export(bo->bo, amdgpu_bo_handle_type_kms_noimport, &bo->u.real.kms_handle);

   amdgpu_add_buffer_to_global_list(bo);

   return bo;

error_va_map:
   amdgpu_va_range_free(va_handle);

error_va_alloc:
   amdgpu_bo_free(buf_handle);

error_bo_alloc:
   FREE(bo);
   return NULL;
}

bool amdgpu_bo_can_reclaim(struct pb_buffer *_buf)
{
   struct amdgpu_winsys_bo *bo = amdgpu_winsys_bo(_buf);

   if (amdgpu_bo_is_referenced_by_any_cs(bo)) {
      return false;
   }

   return amdgpu_bo_wait(_buf, 0, RADEON_USAGE_READWRITE);
}

bool amdgpu_bo_can_reclaim_slab(void *priv, struct pb_slab_entry *entry)
{
   struct amdgpu_winsys_bo *bo = NULL; /* fix container_of */
   bo = container_of(entry, bo, u.slab.entry);

   return amdgpu_bo_can_reclaim(&bo->base);
}

static void amdgpu_bo_slab_destroy(struct pb_buffer *_buf)
{
   struct amdgpu_winsys_bo *bo = amdgpu_winsys_bo(_buf);

   assert(!bo->bo);

   pb_slab_free(&bo->ws->bo_slabs, &bo->u.slab.entry);
}

static const struct pb_vtbl amdgpu_winsys_bo_slab_vtbl = {
   amdgpu_bo_slab_destroy
   /* other functions are never called */
};

struct pb_slab *amdgpu_bo_slab_alloc(void *priv, unsigned heap,
                                     unsigned entry_size,
                                     unsigned group_index)
{
   struct amdgpu_winsys *ws = priv;
   struct amdgpu_slab *slab = CALLOC_STRUCT(amdgpu_slab);
   enum radeon_bo_domain domains = radeon_domain_from_heap(heap);
   enum radeon_bo_flag flags = radeon_flags_from_heap(heap);
   uint32_t base_id;

   if (!slab)
      return NULL;

   unsigned slab_size = 1 << AMDGPU_SLAB_BO_SIZE_LOG2;
   slab->buffer = amdgpu_winsys_bo(amdgpu_bo_create(&ws->base,
                                                    slab_size, slab_size,
                                                    domains, flags));
   if (!slab->buffer)
      goto fail;

   assert(slab->buffer->bo);

   slab->base.num_entries = slab->buffer->base.size / entry_size;
   slab->base.num_free = slab->base.num_entries;
   slab->entries = CALLOC(slab->base.num_entries, sizeof(*slab->entries));
   if (!slab->entries)
      goto fail_buffer;

   LIST_INITHEAD(&slab->base.free);

   base_id = __sync_fetch_and_add(&ws->next_bo_unique_id, slab->base.num_entries);

   for (unsigned i = 0; i < slab->base.num_entries; ++i) {
      struct amdgpu_winsys_bo *bo = &slab->entries[i];

      bo->base.alignment = entry_size;
      bo->base.usage = slab->buffer->base.usage;
      bo->base.size = entry_size;
      bo->base.vtbl = &amdgpu_winsys_bo_slab_vtbl;
      bo->ws = ws;
      bo->va = slab->buffer->va + i * entry_size;
      bo->initial_domain = domains;
      bo->unique_id = base_id + i;
      bo->u.slab.entry.slab = &slab->base;
      bo->u.slab.entry.group_index = group_index;
      bo->u.slab.real = slab->buffer;

      LIST_ADDTAIL(&bo->u.slab.entry.head, &slab->base.free);
   }

   return &slab->base;

fail_buffer:
   amdgpu_winsys_bo_reference(&slab->buffer, NULL);
fail:
   FREE(slab);
   return NULL;
}

void amdgpu_bo_slab_free(void *priv, struct pb_slab *pslab)
{
   struct amdgpu_slab *slab = amdgpu_slab(pslab);

   for (unsigned i = 0; i < slab->base.num_entries; ++i)
      amdgpu_bo_remove_fences(&slab->entries[i]);

   FREE(slab->entries);
   amdgpu_winsys_bo_reference(&slab->buffer, NULL);
   FREE(slab);
}

#if DEBUG_SPARSE_COMMITS
static void
sparse_dump(struct amdgpu_winsys_bo *bo, const char *func)
{
   fprintf(stderr, "%s: %p (size=%"PRIu64", num_va_pages=%u) @ %s\n"
                   "Commitments:\n",
           __func__, bo, bo->base.size, bo->u.sparse.num_va_pages, func);

   struct amdgpu_sparse_backing *span_backing = NULL;
   uint32_t span_first_backing_page = 0;
   uint32_t span_first_va_page = 0;
   uint32_t va_page = 0;

   for (;;) {
      struct amdgpu_sparse_backing *backing = 0;
      uint32_t backing_page = 0;

      if (va_page < bo->u.sparse.num_va_pages) {
         backing = bo->u.sparse.commitments[va_page].backing;
         backing_page = bo->u.sparse.commitments[va_page].page;
      }

      if (span_backing &&
          (backing != span_backing ||
           backing_page != span_first_backing_page + (va_page - span_first_va_page))) {
         fprintf(stderr, " %u..%u: backing=%p:%u..%u\n",
                 span_first_va_page, va_page - 1, span_backing,
                 span_first_backing_page,
                 span_first_backing_page + (va_page - span_first_va_page) - 1);

         span_backing = NULL;
      }

      if (va_page >= bo->u.sparse.num_va_pages)
         break;

      if (backing && !span_backing) {
         span_backing = backing;
         span_first_backing_page = backing_page;
         span_first_va_page = va_page;
      }

      va_page++;
   }

   fprintf(stderr, "Backing:\n");

   list_for_each_entry(struct amdgpu_sparse_backing, backing, &bo->u.sparse.backing, list) {
      fprintf(stderr, " %p (size=%"PRIu64")\n", backing, backing->bo->base.size);
      for (unsigned i = 0; i < backing->num_chunks; ++i)
         fprintf(stderr, "   %u..%u\n", backing->chunks[i].begin, backing->chunks[i].end);
   }
}
#endif

/*
 * Attempt to allocate the given number of backing pages. Fewer pages may be
 * allocated (depending on the fragmentation of existing backing buffers),
 * which will be reflected by a change to *pnum_pages.
 */
static struct amdgpu_sparse_backing *
sparse_backing_alloc(struct amdgpu_winsys_bo *bo, uint32_t *pstart_page, uint32_t *pnum_pages)
{
   struct amdgpu_sparse_backing *best_backing;
   unsigned best_idx;
   uint32_t best_num_pages;

   best_backing = NULL;
   best_idx = 0;
   best_num_pages = 0;

   /* This is a very simple and inefficient best-fit algorithm. */
   list_for_each_entry(struct amdgpu_sparse_backing, backing, &bo->u.sparse.backing, list) {
      for (unsigned idx = 0; idx < backing->num_chunks; ++idx) {
         uint32_t cur_num_pages = backing->chunks[idx].end - backing->chunks[idx].begin;
         if ((best_num_pages < *pnum_pages && cur_num_pages > best_num_pages) ||
            (best_num_pages > *pnum_pages && cur_num_pages < best_num_pages)) {
            best_backing = backing;
            best_idx = idx;
            best_num_pages = cur_num_pages;
         }
      }
   }

   /* Allocate a new backing buffer if necessary. */
   if (!best_backing) {
      struct pb_buffer *buf;
      uint64_t size;
      uint32_t pages;

      best_backing = CALLOC_STRUCT(amdgpu_sparse_backing);
      if (!best_backing)
         return NULL;

      best_backing->max_chunks = 4;
      best_backing->chunks = CALLOC(best_backing->max_chunks,
                                    sizeof(*best_backing->chunks));
      if (!best_backing->chunks) {
         FREE(best_backing);
         return NULL;
      }

      assert(bo->u.sparse.num_backing_pages < DIV_ROUND_UP(bo->base.size, RADEON_SPARSE_PAGE_SIZE));

      size = MIN3(bo->base.size / 16,
                  8 * 1024 * 1024,
                  bo->base.size - (uint64_t)bo->u.sparse.num_backing_pages * RADEON_SPARSE_PAGE_SIZE);
      size = MAX2(size, RADEON_SPARSE_PAGE_SIZE);

      buf = amdgpu_bo_create(&bo->ws->base, size, RADEON_SPARSE_PAGE_SIZE,
                             bo->initial_domain,
                             bo->u.sparse.flags | RADEON_FLAG_NO_SUBALLOC);
      if (!buf) {
         FREE(best_backing->chunks);
         FREE(best_backing);
         return NULL;
      }

      /* We might have gotten a bigger buffer than requested via caching. */
      pages = buf->size / RADEON_SPARSE_PAGE_SIZE;

      best_backing->bo = amdgpu_winsys_bo(buf);
      best_backing->num_chunks = 1;
      best_backing->chunks[0].begin = 0;
      best_backing->chunks[0].end = pages;

      list_add(&best_backing->list, &bo->u.sparse.backing);
      bo->u.sparse.num_backing_pages += pages;

      best_idx = 0;
      best_num_pages = pages;
   }

   *pnum_pages = MIN2(*pnum_pages, best_num_pages);
   *pstart_page = best_backing->chunks[best_idx].begin;
   best_backing->chunks[best_idx].begin += *pnum_pages;

   if (best_backing->chunks[best_idx].begin >= best_backing->chunks[best_idx].end) {
      memmove(&best_backing->chunks[best_idx], &best_backing->chunks[best_idx + 1],
              sizeof(*best_backing->chunks) * (best_backing->num_chunks - best_idx - 1));
      best_backing->num_chunks--;
   }

   return best_backing;
}

static void
sparse_free_backing_buffer(struct amdgpu_winsys_bo *bo,
                           struct amdgpu_sparse_backing *backing)
{
   struct amdgpu_winsys *ws = backing->bo->ws;

   bo->u.sparse.num_backing_pages -= backing->bo->base.size / RADEON_SPARSE_PAGE_SIZE;

   simple_mtx_lock(&ws->bo_fence_lock);
   amdgpu_add_fences(backing->bo, bo->num_fences, bo->fences);
   simple_mtx_unlock(&ws->bo_fence_lock);

   list_del(&backing->list);
   amdgpu_winsys_bo_reference(&backing->bo, NULL);
   FREE(backing->chunks);
   FREE(backing);
}

/*
 * Return a range of pages from the given backing buffer back into the
 * free structure.
 */
static bool
sparse_backing_free(struct amdgpu_winsys_bo *bo,
                    struct amdgpu_sparse_backing *backing,
                    uint32_t start_page, uint32_t num_pages)
{
   uint32_t end_page = start_page + num_pages;
   unsigned low = 0;
   unsigned high = backing->num_chunks;

   /* Find the first chunk with begin >= start_page. */
   while (low < high) {
      unsigned mid = low + (high - low) / 2;

      if (backing->chunks[mid].begin >= start_page)
         high = mid;
      else
         low = mid + 1;
   }

   assert(low >= backing->num_chunks || end_page <= backing->chunks[low].begin);
   assert(low == 0 || backing->chunks[low - 1].end <= start_page);

   if (low > 0 && backing->chunks[low - 1].end == start_page) {
      backing->chunks[low - 1].end = end_page;

      if (low < backing->num_chunks && end_page == backing->chunks[low].begin) {
         backing->chunks[low - 1].end = backing->chunks[low].end;
         memmove(&backing->chunks[low], &backing->chunks[low + 1],
                 sizeof(*backing->chunks) * (backing->num_chunks - low - 1));
         backing->num_chunks--;
      }
   } else if (low < backing->num_chunks && end_page == backing->chunks[low].begin) {
      backing->chunks[low].begin = start_page;
   } else {
      if (backing->num_chunks >= backing->max_chunks) {
         unsigned new_max_chunks = 2 * backing->max_chunks;
         struct amdgpu_sparse_backing_chunk *new_chunks =
            REALLOC(backing->chunks,
                    sizeof(*backing->chunks) * backing->max_chunks,
                    sizeof(*backing->chunks) * new_max_chunks);
         if (!new_chunks)
            return false;

         backing->max_chunks = new_max_chunks;
         backing->chunks = new_chunks;
      }

      memmove(&backing->chunks[low + 1], &backing->chunks[low],
              sizeof(*backing->chunks) * (backing->num_chunks - low));
      backing->chunks[low].begin = start_page;
      backing->chunks[low].end = end_page;
      backing->num_chunks++;
   }

   if (backing->num_chunks == 1 && backing->chunks[0].begin == 0 &&
       backing->chunks[0].end == backing->bo->base.size / RADEON_SPARSE_PAGE_SIZE)
      sparse_free_backing_buffer(bo, backing);

   return true;
}

static void amdgpu_bo_sparse_destroy(struct pb_buffer *_buf)
{
   struct amdgpu_winsys_bo *bo = amdgpu_winsys_bo(_buf);
   int r;

   assert(!bo->bo && bo->sparse);

   r = amdgpu_bo_va_op_raw(bo->ws->dev, NULL, 0,
                           (uint64_t)bo->u.sparse.num_va_pages * RADEON_SPARSE_PAGE_SIZE,
                           bo->va, 0, AMDGPU_VA_OP_CLEAR);
   if (r) {
      fprintf(stderr, "amdgpu: clearing PRT VA region on destroy failed (%d)\n", r);
   }

   while (!list_empty(&bo->u.sparse.backing)) {
      struct amdgpu_sparse_backing *dummy = NULL;
      sparse_free_backing_buffer(bo,
                                 container_of(bo->u.sparse.backing.next,
                                              dummy, list));
   }

   amdgpu_va_range_free(bo->u.sparse.va_handle);
   simple_mtx_destroy(&bo->u.sparse.commit_lock);
   FREE(bo->u.sparse.commitments);
   FREE(bo);
}

static const struct pb_vtbl amdgpu_winsys_bo_sparse_vtbl = {
   amdgpu_bo_sparse_destroy
   /* other functions are never called */
};

static struct pb_buffer *
amdgpu_bo_sparse_create(struct amdgpu_winsys *ws, uint64_t size,
                        enum radeon_bo_domain domain,
                        enum radeon_bo_flag flags)
{
   struct amdgpu_winsys_bo *bo;
   uint64_t map_size;
   uint64_t va_gap_size;
   int r;

   /* We use 32-bit page numbers; refuse to attempt allocating sparse buffers
    * that exceed this limit. This is not really a restriction: we don't have
    * that much virtual address space anyway.
    */
   if (size > (uint64_t)INT32_MAX * RADEON_SPARSE_PAGE_SIZE)
      return NULL;

   bo = CALLOC_STRUCT(amdgpu_winsys_bo);
   if (!bo)
      return NULL;

   pipe_reference_init(&bo->base.reference, 1);
   bo->base.alignment = RADEON_SPARSE_PAGE_SIZE;
   bo->base.size = size;
   bo->base.vtbl = &amdgpu_winsys_bo_sparse_vtbl;
   bo->ws = ws;
   bo->initial_domain = domain;
   bo->unique_id =  __sync_fetch_and_add(&ws->next_bo_unique_id, 1);
   bo->sparse = true;
   bo->u.sparse.flags = flags & ~RADEON_FLAG_SPARSE;

   bo->u.sparse.num_va_pages = DIV_ROUND_UP(size, RADEON_SPARSE_PAGE_SIZE);
   bo->u.sparse.commitments = CALLOC(bo->u.sparse.num_va_pages,
                                     sizeof(*bo->u.sparse.commitments));
   if (!bo->u.sparse.commitments)
      goto error_alloc_commitments;

   simple_mtx_init(&bo->u.sparse.commit_lock, mtx_plain);
   LIST_INITHEAD(&bo->u.sparse.backing);

   /* For simplicity, we always map a multiple of the page size. */
   map_size = align64(size, RADEON_SPARSE_PAGE_SIZE);
   va_gap_size = ws->check_vm ? 4 * RADEON_SPARSE_PAGE_SIZE : 0;
   r = amdgpu_va_range_alloc(ws->dev, amdgpu_gpu_va_range_general,
                             map_size + va_gap_size, RADEON_SPARSE_PAGE_SIZE,
                             0, &bo->va, &bo->u.sparse.va_handle,
			     AMDGPU_VA_RANGE_HIGH);
   if (r)
      goto error_va_alloc;

   r = amdgpu_bo_va_op_raw(bo->ws->dev, NULL, 0, size, bo->va,
                           AMDGPU_VM_PAGE_PRT, AMDGPU_VA_OP_MAP);
   if (r)
      goto error_va_map;

   return &bo->base;

error_va_map:
   amdgpu_va_range_free(bo->u.sparse.va_handle);
error_va_alloc:
   simple_mtx_destroy(&bo->u.sparse.commit_lock);
   FREE(bo->u.sparse.commitments);
error_alloc_commitments:
   FREE(bo);
   return NULL;
}

static bool
amdgpu_bo_sparse_commit(struct pb_buffer *buf, uint64_t offset, uint64_t size,
                        bool commit)
{
   struct amdgpu_winsys_bo *bo = amdgpu_winsys_bo(buf);
   struct amdgpu_sparse_commitment *comm;
   uint32_t va_page, end_va_page;
   bool ok = true;
   int r;

   assert(bo->sparse);
   assert(offset % RADEON_SPARSE_PAGE_SIZE == 0);
   assert(offset <= bo->base.size);
   assert(size <= bo->base.size - offset);
   assert(size % RADEON_SPARSE_PAGE_SIZE == 0 || offset + size == bo->base.size);

   comm = bo->u.sparse.commitments;
   va_page = offset / RADEON_SPARSE_PAGE_SIZE;
   end_va_page = va_page + DIV_ROUND_UP(size, RADEON_SPARSE_PAGE_SIZE);

   simple_mtx_lock(&bo->u.sparse.commit_lock);

#if DEBUG_SPARSE_COMMITS
   sparse_dump(bo, __func__);
#endif

   if (commit) {
      while (va_page < end_va_page) {
         uint32_t span_va_page;

         /* Skip pages that are already committed. */
         if (comm[va_page].backing) {
            va_page++;
            continue;
         }

         /* Determine length of uncommitted span. */
         span_va_page = va_page;
         while (va_page < end_va_page && !comm[va_page].backing)
            va_page++;

         /* Fill the uncommitted span with chunks of backing memory. */
         while (span_va_page < va_page) {
            struct amdgpu_sparse_backing *backing;
            uint32_t backing_start, backing_size;

            backing_size = va_page - span_va_page;
            backing = sparse_backing_alloc(bo, &backing_start, &backing_size);
            if (!backing) {
               ok = false;
               goto out;
            }

            r = amdgpu_bo_va_op_raw(bo->ws->dev, backing->bo->bo,
                                    (uint64_t)backing_start * RADEON_SPARSE_PAGE_SIZE,
                                    (uint64_t)backing_size * RADEON_SPARSE_PAGE_SIZE,
                                    bo->va + (uint64_t)span_va_page * RADEON_SPARSE_PAGE_SIZE,
                                    AMDGPU_VM_PAGE_READABLE |
                                    AMDGPU_VM_PAGE_WRITEABLE |
                                    AMDGPU_VM_PAGE_EXECUTABLE,
                                    AMDGPU_VA_OP_REPLACE);
            if (r) {
               ok = sparse_backing_free(bo, backing, backing_start, backing_size);
               assert(ok && "sufficient memory should already be allocated");

               ok = false;
               goto out;
            }

            while (backing_size) {
               comm[span_va_page].backing = backing;
               comm[span_va_page].page = backing_start;
               span_va_page++;
               backing_start++;
               backing_size--;
            }
         }
      }
   } else {
      r = amdgpu_bo_va_op_raw(bo->ws->dev, NULL, 0,
                              (uint64_t)(end_va_page - va_page) * RADEON_SPARSE_PAGE_SIZE,
                              bo->va + (uint64_t)va_page * RADEON_SPARSE_PAGE_SIZE,
                              AMDGPU_VM_PAGE_PRT, AMDGPU_VA_OP_REPLACE);
      if (r) {
         ok = false;
         goto out;
      }

      while (va_page < end_va_page) {
         struct amdgpu_sparse_backing *backing;
         uint32_t backing_start;
         uint32_t span_pages;

         /* Skip pages that are already uncommitted. */
         if (!comm[va_page].backing) {
            va_page++;
            continue;
         }

         /* Group contiguous spans of pages. */
         backing = comm[va_page].backing;
         backing_start = comm[va_page].page;
         comm[va_page].backing = NULL;

         span_pages = 1;
         va_page++;

         while (va_page < end_va_page &&
                comm[va_page].backing == backing &&
                comm[va_page].page == backing_start + span_pages) {
            comm[va_page].backing = NULL;
            va_page++;
            span_pages++;
         }

         if (!sparse_backing_free(bo, backing, backing_start, span_pages)) {
            /* Couldn't allocate tracking data structures, so we have to leak */
            fprintf(stderr, "amdgpu: leaking PRT backing memory\n");
            ok = false;
         }
      }
   }
out:

   simple_mtx_unlock(&bo->u.sparse.commit_lock);

   return ok;
}

static unsigned eg_tile_split(unsigned tile_split)
{
   switch (tile_split) {
   case 0:     tile_split = 64;    break;
   case 1:     tile_split = 128;   break;
   case 2:     tile_split = 256;   break;
   case 3:     tile_split = 512;   break;
   default:
   case 4:     tile_split = 1024;  break;
   case 5:     tile_split = 2048;  break;
   case 6:     tile_split = 4096;  break;
   }
   return tile_split;
}

static unsigned eg_tile_split_rev(unsigned eg_tile_split)
{
   switch (eg_tile_split) {
   case 64:    return 0;
   case 128:   return 1;
   case 256:   return 2;
   case 512:   return 3;
   default:
   case 1024:  return 4;
   case 2048:  return 5;
   case 4096:  return 6;
   }
}

static void amdgpu_buffer_get_metadata(struct pb_buffer *_buf,
                                       struct radeon_bo_metadata *md)
{
   struct amdgpu_winsys_bo *bo = amdgpu_winsys_bo(_buf);
   struct amdgpu_bo_info info = {0};
   uint64_t tiling_flags;
   int r;

   assert(bo->bo && "must not be called for slab entries");

   r = amdgpu_bo_query_info(bo->bo, &info);
   if (r)
      return;

   tiling_flags = info.metadata.tiling_info;

   if (bo->ws->info.chip_class >= GFX9) {
      md->u.gfx9.swizzle_mode = AMDGPU_TILING_GET(tiling_flags, SWIZZLE_MODE);
   } else {
      md->u.legacy.microtile = RADEON_LAYOUT_LINEAR;
      md->u.legacy.macrotile = RADEON_LAYOUT_LINEAR;

      if (AMDGPU_TILING_GET(tiling_flags, ARRAY_MODE) == 4)  /* 2D_TILED_THIN1 */
         md->u.legacy.macrotile = RADEON_LAYOUT_TILED;
      else if (AMDGPU_TILING_GET(tiling_flags, ARRAY_MODE) == 2) /* 1D_TILED_THIN1 */
         md->u.legacy.microtile = RADEON_LAYOUT_TILED;

      md->u.legacy.pipe_config = AMDGPU_TILING_GET(tiling_flags, PIPE_CONFIG);
      md->u.legacy.bankw = 1 << AMDGPU_TILING_GET(tiling_flags, BANK_WIDTH);
      md->u.legacy.bankh = 1 << AMDGPU_TILING_GET(tiling_flags, BANK_HEIGHT);
      md->u.legacy.tile_split = eg_tile_split(AMDGPU_TILING_GET(tiling_flags, TILE_SPLIT));
      md->u.legacy.mtilea = 1 << AMDGPU_TILING_GET(tiling_flags, MACRO_TILE_ASPECT);
      md->u.legacy.num_banks = 2 << AMDGPU_TILING_GET(tiling_flags, NUM_BANKS);
      md->u.legacy.scanout = AMDGPU_TILING_GET(tiling_flags, MICRO_TILE_MODE) == 0; /* DISPLAY */
   }

   md->size_metadata = info.metadata.size_metadata;
   memcpy(md->metadata, info.metadata.umd_metadata, sizeof(md->metadata));
}

static void amdgpu_buffer_set_metadata(struct pb_buffer *_buf,
                                       struct radeon_bo_metadata *md)
{
   struct amdgpu_winsys_bo *bo = amdgpu_winsys_bo(_buf);
   struct amdgpu_bo_metadata metadata = {0};
   uint64_t tiling_flags = 0;

   assert(bo->bo && "must not be called for slab entries");

   if (bo->ws->info.chip_class >= GFX9) {
      tiling_flags |= AMDGPU_TILING_SET(SWIZZLE_MODE, md->u.gfx9.swizzle_mode);
   } else {
      if (md->u.legacy.macrotile == RADEON_LAYOUT_TILED)
         tiling_flags |= AMDGPU_TILING_SET(ARRAY_MODE, 4); /* 2D_TILED_THIN1 */
      else if (md->u.legacy.microtile == RADEON_LAYOUT_TILED)
         tiling_flags |= AMDGPU_TILING_SET(ARRAY_MODE, 2); /* 1D_TILED_THIN1 */
      else
         tiling_flags |= AMDGPU_TILING_SET(ARRAY_MODE, 1); /* LINEAR_ALIGNED */

      tiling_flags |= AMDGPU_TILING_SET(PIPE_CONFIG, md->u.legacy.pipe_config);
      tiling_flags |= AMDGPU_TILING_SET(BANK_WIDTH, util_logbase2(md->u.legacy.bankw));
      tiling_flags |= AMDGPU_TILING_SET(BANK_HEIGHT, util_logbase2(md->u.legacy.bankh));
      if (md->u.legacy.tile_split)
         tiling_flags |= AMDGPU_TILING_SET(TILE_SPLIT, eg_tile_split_rev(md->u.legacy.tile_split));
      tiling_flags |= AMDGPU_TILING_SET(MACRO_TILE_ASPECT, util_logbase2(md->u.legacy.mtilea));
      tiling_flags |= AMDGPU_TILING_SET(NUM_BANKS, util_logbase2(md->u.legacy.num_banks)-1);

      if (md->u.legacy.scanout)
         tiling_flags |= AMDGPU_TILING_SET(MICRO_TILE_MODE, 0); /* DISPLAY_MICRO_TILING */
      else
         tiling_flags |= AMDGPU_TILING_SET(MICRO_TILE_MODE, 1); /* THIN_MICRO_TILING */
   }

   metadata.tiling_info = tiling_flags;
   metadata.size_metadata = md->size_metadata;
   memcpy(metadata.umd_metadata, md->metadata, sizeof(md->metadata));

   amdgpu_bo_set_metadata(bo->bo, &metadata);
}

static struct pb_buffer *
amdgpu_bo_create(struct radeon_winsys *rws,
                 uint64_t size,
                 unsigned alignment,
                 enum radeon_bo_domain domain,
                 enum radeon_bo_flag flags)
{
   struct amdgpu_winsys *ws = amdgpu_winsys(rws);
   struct amdgpu_winsys_bo *bo;
   int heap = -1;

   /* VRAM implies WC. This is not optional. */
   assert(!(domain & RADEON_DOMAIN_VRAM) || flags & RADEON_FLAG_GTT_WC);

   /* NO_CPU_ACCESS is valid with VRAM only. */
   assert(domain == RADEON_DOMAIN_VRAM || !(flags & RADEON_FLAG_NO_CPU_ACCESS));

   /* Sparse buffers must have NO_CPU_ACCESS set. */
   assert(!(flags & RADEON_FLAG_SPARSE) || flags & RADEON_FLAG_NO_CPU_ACCESS);

   /* Sub-allocate small buffers from slabs. */
   if (!(flags & (RADEON_FLAG_NO_SUBALLOC | RADEON_FLAG_SPARSE)) &&
       size <= (1 << AMDGPU_SLAB_MAX_SIZE_LOG2) &&
       alignment <= MAX2(1 << AMDGPU_SLAB_MIN_SIZE_LOG2, util_next_power_of_two(size))) {
      struct pb_slab_entry *entry;
      int heap = radeon_get_heap_index(domain, flags);

      if (heap < 0 || heap >= RADEON_MAX_SLAB_HEAPS)
         goto no_slab;

      entry = pb_slab_alloc(&ws->bo_slabs, size, heap);
      if (!entry) {
         /* Clear the cache and try again. */
         pb_cache_release_all_buffers(&ws->bo_cache);

         entry = pb_slab_alloc(&ws->bo_slabs, size, heap);
      }
      if (!entry)
         return NULL;

      bo = NULL;
      bo = container_of(entry, bo, u.slab.entry);

      pipe_reference_init(&bo->base.reference, 1);

      return &bo->base;
   }
no_slab:

   if (flags & RADEON_FLAG_SPARSE) {
      assert(RADEON_SPARSE_PAGE_SIZE % alignment == 0);

      return amdgpu_bo_sparse_create(ws, size, domain, flags);
   }

   /* This flag is irrelevant for the cache. */
   flags &= ~RADEON_FLAG_NO_SUBALLOC;

   /* Align size to page size. This is the minimum alignment for normal
    * BOs. Aligning this here helps the cached bufmgr. Especially small BOs,
    * like constant/uniform buffers, can benefit from better and more reuse.
    */
   size = align64(size, ws->info.gart_page_size);
   alignment = align(alignment, ws->info.gart_page_size);

   bool use_reusable_pool = flags & RADEON_FLAG_NO_INTERPROCESS_SHARING;

   if (use_reusable_pool) {
       heap = radeon_get_heap_index(domain, flags);
       assert(heap >= 0 && heap < RADEON_MAX_CACHED_HEAPS);

       /* Get a buffer from the cache. */
       bo = (struct amdgpu_winsys_bo*)
            pb_cache_reclaim_buffer(&ws->bo_cache, size, alignment, 0, heap);
       if (bo)
          return &bo->base;
   }

   /* Create a new one. */
   bo = amdgpu_create_bo(ws, size, alignment, domain, flags, heap);
   if (!bo) {
      /* Clear the cache and try again. */
      pb_slabs_reclaim(&ws->bo_slabs);
      pb_cache_release_all_buffers(&ws->bo_cache);
      bo = amdgpu_create_bo(ws, size, alignment, domain, flags, heap);
      if (!bo)
         return NULL;
   }

   bo->u.real.use_reusable_pool = use_reusable_pool;
   return &bo->base;
}

static struct pb_buffer *amdgpu_bo_from_handle(struct radeon_winsys *rws,
                                               struct winsys_handle *whandle,
                                               unsigned *stride,
                                               unsigned *offset)
{
   struct amdgpu_winsys *ws = amdgpu_winsys(rws);
   struct amdgpu_winsys_bo *bo = NULL;
   enum amdgpu_bo_handle_type type;
   struct amdgpu_bo_import_result result = {0};
   uint64_t va;
   amdgpu_va_handle va_handle = NULL;
   struct amdgpu_bo_info info = {0};
   enum radeon_bo_domain initial = 0;
   int r;

   switch (whandle->type) {
   case WINSYS_HANDLE_TYPE_SHARED:
      type = amdgpu_bo_handle_type_gem_flink_name;
      break;
   case WINSYS_HANDLE_TYPE_FD:
      type = amdgpu_bo_handle_type_dma_buf_fd;
      break;
   default:
      return NULL;
   }

   if (stride)
      *stride = whandle->stride;
   if (offset)
      *offset = whandle->offset;

   r = amdgpu_bo_import(ws->dev, type, whandle->handle, &result);
   if (r)
      return NULL;

   simple_mtx_lock(&ws->bo_export_table_lock);
   bo = util_hash_table_get(ws->bo_export_table, result.buf_handle);

   /* If the amdgpu_winsys_bo instance already exists, bump the reference
    * counter and return it.
    */
   if (bo) {
      p_atomic_inc(&bo->base.reference.count);
      simple_mtx_unlock(&ws->bo_export_table_lock);

      /* Release the buffer handle, because we don't need it anymore.
       * This function is returning an existing buffer, which has its own
       * handle.
       */
      amdgpu_bo_free(result.buf_handle);
      return &bo->base;
   }

   /* Get initial domains. */
   r = amdgpu_bo_query_info(result.buf_handle, &info);
   if (r)
      goto error;

   r = amdgpu_va_range_alloc(ws->dev, amdgpu_gpu_va_range_general,
                             result.alloc_size, 1 << 20, 0, &va, &va_handle,
			     AMDGPU_VA_RANGE_HIGH);
   if (r)
      goto error;

   bo = CALLOC_STRUCT(amdgpu_winsys_bo);
   if (!bo)
      goto error;

   r = amdgpu_bo_va_op(result.buf_handle, 0, result.alloc_size, va, 0, AMDGPU_VA_OP_MAP);
   if (r)
      goto error;

   if (info.preferred_heap & AMDGPU_GEM_DOMAIN_VRAM)
      initial |= RADEON_DOMAIN_VRAM;
   if (info.preferred_heap & AMDGPU_GEM_DOMAIN_GTT)
      initial |= RADEON_DOMAIN_GTT;

   /* Initialize the structure. */
   pipe_reference_init(&bo->base.reference, 1);
   bo->base.alignment = info.phys_alignment;
   bo->bo = result.buf_handle;
   bo->base.size = result.alloc_size;
   bo->base.vtbl = &amdgpu_winsys_bo_vtbl;
   bo->ws = ws;
   bo->va = va;
   bo->u.real.va_handle = va_handle;
   bo->initial_domain = initial;
   bo->unique_id = __sync_fetch_and_add(&ws->next_bo_unique_id, 1);
   bo->is_shared = true;

   if (bo->initial_domain & RADEON_DOMAIN_VRAM)
      ws->allocated_vram += align64(bo->base.size, ws->info.gart_page_size);
   else if (bo->initial_domain & RADEON_DOMAIN_GTT)
      ws->allocated_gtt += align64(bo->base.size, ws->info.gart_page_size);

   amdgpu_bo_export(bo->bo, amdgpu_bo_handle_type_kms_noimport, &bo->u.real.kms_handle);

   amdgpu_add_buffer_to_global_list(bo);

   util_hash_table_set(ws->bo_export_table, bo->bo, bo);
   simple_mtx_unlock(&ws->bo_export_table_lock);

   return &bo->base;

error:
   simple_mtx_unlock(&ws->bo_export_table_lock);
   if (bo)
      FREE(bo);
   if (va_handle)
      amdgpu_va_range_free(va_handle);
   amdgpu_bo_free(result.buf_handle);
   return NULL;
}

static bool amdgpu_bo_get_handle(struct pb_buffer *buffer,
                                 unsigned stride, unsigned offset,
                                 unsigned slice_size,
                                 struct winsys_handle *whandle)
{
   struct amdgpu_winsys_bo *bo = amdgpu_winsys_bo(buffer);
   struct amdgpu_winsys *ws = bo->ws;
   enum amdgpu_bo_handle_type type;
   int r;

   /* Don't allow exports of slab entries and sparse buffers. */
   if (!bo->bo)
      return false;

   bo->u.real.use_reusable_pool = false;

   switch (whandle->type) {
   case WINSYS_HANDLE_TYPE_SHARED:
      type = amdgpu_bo_handle_type_gem_flink_name;
      break;
   case WINSYS_HANDLE_TYPE_FD:
      type = amdgpu_bo_handle_type_dma_buf_fd;
      break;
   case WINSYS_HANDLE_TYPE_KMS:
      type = amdgpu_bo_handle_type_kms;
      break;
   default:
      return false;
   }

   r = amdgpu_bo_export(bo->bo, type, &whandle->handle);
   if (r)
      return false;

   simple_mtx_lock(&ws->bo_export_table_lock);
   util_hash_table_set(ws->bo_export_table, bo->bo, bo);
   simple_mtx_unlock(&ws->bo_export_table_lock);

   whandle->stride = stride;
   whandle->offset = offset;
   whandle->offset += slice_size * whandle->layer;
   bo->is_shared = true;
   return true;
}

static struct pb_buffer *amdgpu_bo_from_ptr(struct radeon_winsys *rws,
					    void *pointer, uint64_t size)
{
    struct amdgpu_winsys *ws = amdgpu_winsys(rws);
    amdgpu_bo_handle buf_handle;
    struct amdgpu_winsys_bo *bo;
    uint64_t va;
    amdgpu_va_handle va_handle;
    /* Avoid failure when the size is not page aligned */
    uint64_t aligned_size = align64(size, ws->info.gart_page_size);

    bo = CALLOC_STRUCT(amdgpu_winsys_bo);
    if (!bo)
        return NULL;

    if (amdgpu_create_bo_from_user_mem(ws->dev, pointer,
                                       aligned_size, &buf_handle))
        goto error;

    if (amdgpu_va_range_alloc(ws->dev, amdgpu_gpu_va_range_general,
                              aligned_size, 1 << 12, 0, &va, &va_handle,
			      AMDGPU_VA_RANGE_HIGH))
        goto error_va_alloc;

    if (amdgpu_bo_va_op(buf_handle, 0, aligned_size, va, 0, AMDGPU_VA_OP_MAP))
        goto error_va_map;

    /* Initialize it. */
    pipe_reference_init(&bo->base.reference, 1);
    bo->bo = buf_handle;
    bo->base.alignment = 0;
    bo->base.size = size;
    bo->base.vtbl = &amdgpu_winsys_bo_vtbl;
    bo->ws = ws;
    bo->user_ptr = pointer;
    bo->va = va;
    bo->u.real.va_handle = va_handle;
    bo->initial_domain = RADEON_DOMAIN_GTT;
    bo->unique_id = __sync_fetch_and_add(&ws->next_bo_unique_id, 1);

    ws->allocated_gtt += aligned_size;

    amdgpu_add_buffer_to_global_list(bo);

    amdgpu_bo_export(bo->bo, amdgpu_bo_handle_type_kms_noimport, &bo->u.real.kms_handle);

    return (struct pb_buffer*)bo;

error_va_map:
    amdgpu_va_range_free(va_handle);

error_va_alloc:
    amdgpu_bo_free(buf_handle);

error:
    FREE(bo);
    return NULL;
}

static bool amdgpu_bo_is_user_ptr(struct pb_buffer *buf)
{
   return ((struct amdgpu_winsys_bo*)buf)->user_ptr != NULL;
}

static bool amdgpu_bo_is_suballocated(struct pb_buffer *buf)
{
   struct amdgpu_winsys_bo *bo = (struct amdgpu_winsys_bo*)buf;

   return !bo->bo && !bo->sparse;
}

static uint64_t amdgpu_bo_get_va(struct pb_buffer *buf)
{
   return ((struct amdgpu_winsys_bo*)buf)->va;
}

void amdgpu_bo_init_functions(struct amdgpu_winsys *ws)
{
   ws->base.buffer_set_metadata = amdgpu_buffer_set_metadata;
   ws->base.buffer_get_metadata = amdgpu_buffer_get_metadata;
   ws->base.buffer_map = amdgpu_bo_map;
   ws->base.buffer_unmap = amdgpu_bo_unmap;
   ws->base.buffer_wait = amdgpu_bo_wait;
   ws->base.buffer_create = amdgpu_bo_create;
   ws->base.buffer_from_handle = amdgpu_bo_from_handle;
   ws->base.buffer_from_ptr = amdgpu_bo_from_ptr;
   ws->base.buffer_is_user_ptr = amdgpu_bo_is_user_ptr;
   ws->base.buffer_is_suballocated = amdgpu_bo_is_suballocated;
   ws->base.buffer_get_handle = amdgpu_bo_get_handle;
   ws->base.buffer_commit = amdgpu_bo_sparse_commit;
   ws->base.buffer_get_virtual_address = amdgpu_bo_get_va;
   ws->base.buffer_get_initial_domain = amdgpu_bo_get_initial_domain;
}
