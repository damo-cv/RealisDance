/*
 * Copyright © 2008 Jérôme Glisse
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

#ifndef AMDGPU_BO_H
#define AMDGPU_BO_H

#include "amdgpu_winsys.h"

#include "pipebuffer/pb_slab.h"

struct amdgpu_sparse_backing_chunk;

/*
 * Sub-allocation information for a real buffer used as backing memory of a
 * sparse buffer.
 */
struct amdgpu_sparse_backing {
   struct list_head list;

   struct amdgpu_winsys_bo *bo;

   /* Sorted list of free chunks. */
   struct amdgpu_sparse_backing_chunk *chunks;
   uint32_t max_chunks;
   uint32_t num_chunks;
};

struct amdgpu_sparse_commitment {
   struct amdgpu_sparse_backing *backing;
   uint32_t page;
};

struct amdgpu_winsys_bo {
   struct pb_buffer base;
   union {
      struct {
         struct pb_cache_entry cache_entry;

         amdgpu_va_handle va_handle;
         int map_count;
         bool use_reusable_pool;

         struct list_head global_list_item;

         uint32_t kms_handle;
      } real;
      struct {
         struct pb_slab_entry entry;
         struct amdgpu_winsys_bo *real;
      } slab;
      struct {
         simple_mtx_t commit_lock;
         amdgpu_va_handle va_handle;
         enum radeon_bo_flag flags;

         uint32_t num_va_pages;
         uint32_t num_backing_pages;

         struct list_head backing;

         /* Commitment information for each page of the virtual memory area. */
         struct amdgpu_sparse_commitment *commitments;
      } sparse;
   } u;

   struct amdgpu_winsys *ws;
   void *user_ptr; /* from buffer_from_ptr */

   amdgpu_bo_handle bo; /* NULL for slab entries and sparse buffers */
   bool sparse;
   uint32_t unique_id;
   uint64_t va;
   enum radeon_bo_domain initial_domain;

   /* how many command streams is this bo referenced in? */
   int num_cs_references;

   /* how many command streams, which are being emitted in a separate
    * thread, is this bo referenced in? */
   volatile int num_active_ioctls;

   /* whether buffer_get_handle or buffer_from_handle was called,
    * it can only transition from false to true
    */
   volatile int is_shared; /* bool (int for atomicity) */

   /* Fences for buffer synchronization. */
   unsigned num_fences;
   unsigned max_fences;
   struct pipe_fence_handle **fences;

   bool is_local;
};

struct amdgpu_slab {
   struct pb_slab base;
   struct amdgpu_winsys_bo *buffer;
   struct amdgpu_winsys_bo *entries;
};

bool amdgpu_bo_can_reclaim(struct pb_buffer *_buf);
void amdgpu_bo_destroy(struct pb_buffer *_buf);
void amdgpu_bo_init_functions(struct amdgpu_winsys *ws);

bool amdgpu_bo_can_reclaim_slab(void *priv, struct pb_slab_entry *entry);
struct pb_slab *amdgpu_bo_slab_alloc(void *priv, unsigned heap,
                                     unsigned entry_size,
                                     unsigned group_index);
void amdgpu_bo_slab_free(void *priv, struct pb_slab *slab);

static inline
struct amdgpu_winsys_bo *amdgpu_winsys_bo(struct pb_buffer *bo)
{
   return (struct amdgpu_winsys_bo *)bo;
}

static inline
struct amdgpu_slab *amdgpu_slab(struct pb_slab *slab)
{
   return (struct amdgpu_slab *)slab;
}

static inline
void amdgpu_winsys_bo_reference(struct amdgpu_winsys_bo **dst,
                                struct amdgpu_winsys_bo *src)
{
   pb_reference((struct pb_buffer**)dst, (struct pb_buffer*)src);
}

#endif
