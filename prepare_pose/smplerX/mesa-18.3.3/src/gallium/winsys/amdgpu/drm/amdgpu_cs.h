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

#ifndef AMDGPU_CS_H
#define AMDGPU_CS_H

#include "amdgpu_bo.h"
#include "util/u_memory.h"
#include <amdgpu_drm.h>

struct amdgpu_ctx {
   struct amdgpu_winsys *ws;
   amdgpu_context_handle ctx;
   amdgpu_bo_handle user_fence_bo;
   uint64_t *user_fence_cpu_address_base;
   int refcount;
   unsigned initial_num_total_rejected_cs;
   unsigned num_rejected_cs;
};

struct amdgpu_cs_buffer {
   struct amdgpu_winsys_bo *bo;
   union {
      struct {
         uint32_t priority_usage;
      } real;
      struct {
         uint32_t real_idx; /* index of underlying real BO */
      } slab;
   } u;
   enum radeon_bo_usage usage;
};

enum ib_type {
   IB_MAIN,
   IB_NUM,
};

struct amdgpu_ib {
   struct radeon_cmdbuf base;

   /* A buffer out of which new IBs are allocated. */
   struct pb_buffer        *big_ib_buffer;
   uint8_t                 *ib_mapped;
   unsigned                used_ib_space;
   unsigned                max_ib_size;
   uint32_t                *ptr_ib_size;
   bool                    ptr_ib_size_inside_ib;
   enum ib_type            ib_type;
};

struct amdgpu_cs_context {
   struct drm_amdgpu_cs_chunk_ib ib[IB_NUM];

   /* Buffers. */
   unsigned                    max_real_buffers;
   unsigned                    num_real_buffers;
   struct amdgpu_cs_buffer     *real_buffers;

   unsigned                    num_slab_buffers;
   unsigned                    max_slab_buffers;
   struct amdgpu_cs_buffer     *slab_buffers;

   unsigned                    num_sparse_buffers;
   unsigned                    max_sparse_buffers;
   struct amdgpu_cs_buffer     *sparse_buffers;

   int                         buffer_indices_hashlist[4096];

   struct amdgpu_winsys_bo     *last_added_bo;
   unsigned                    last_added_bo_index;
   unsigned                    last_added_bo_usage;
   uint32_t                    last_added_bo_priority_usage;

   struct pipe_fence_handle    **fence_dependencies;
   unsigned                    num_fence_dependencies;
   unsigned                    max_fence_dependencies;

   struct pipe_fence_handle    **syncobj_to_signal;
   unsigned                    num_syncobj_to_signal;
   unsigned                    max_syncobj_to_signal;

   struct pipe_fence_handle    *fence;

   /* the error returned from cs_flush for non-async submissions */
   int                         error_code;
};

struct amdgpu_cs {
   struct amdgpu_ib main; /* must be first because this is inherited */
   struct amdgpu_ctx *ctx;
   enum ring_type ring_type;
   struct drm_amdgpu_cs_chunk_fence fence_chunk;

   /* We flip between these two CS. While one is being consumed
    * by the kernel in another thread, the other one is being filled
    * by the pipe driver. */
   struct amdgpu_cs_context csc1;
   struct amdgpu_cs_context csc2;
   /* The currently-used CS. */
   struct amdgpu_cs_context *csc;
   /* The CS being currently-owned by the other thread. */
   struct amdgpu_cs_context *cst;

   /* Flush CS. */
   void (*flush_cs)(void *ctx, unsigned flags, struct pipe_fence_handle **fence);
   void *flush_data;

   struct util_queue_fence flush_completed;
   struct pipe_fence_handle *next_fence;
};

struct amdgpu_fence {
   struct pipe_reference reference;
   /* If ctx == NULL, this fence is syncobj-based. */
   uint32_t syncobj;

   struct amdgpu_winsys *ws;
   struct amdgpu_ctx *ctx;  /* submission context */
   struct amdgpu_cs_fence fence;
   uint64_t *user_fence_cpu_address;

   /* If the fence has been submitted. This is unsignalled for deferred fences
    * (cs->next_fence) and while an IB is still being submitted in the submit
    * thread. */
   struct util_queue_fence submitted;

   volatile int signalled;              /* bool (int for atomicity) */
};

static inline bool amdgpu_fence_is_syncobj(struct amdgpu_fence *fence)
{
   return fence->ctx == NULL;
}

static inline void amdgpu_ctx_unref(struct amdgpu_ctx *ctx)
{
   if (p_atomic_dec_zero(&ctx->refcount)) {
      amdgpu_cs_ctx_free(ctx->ctx);
      amdgpu_bo_free(ctx->user_fence_bo);
      FREE(ctx);
   }
}

static inline void amdgpu_fence_reference(struct pipe_fence_handle **dst,
                                          struct pipe_fence_handle *src)
{
   struct amdgpu_fence **rdst = (struct amdgpu_fence **)dst;
   struct amdgpu_fence *rsrc = (struct amdgpu_fence *)src;

   if (pipe_reference(&(*rdst)->reference, &rsrc->reference)) {
      struct amdgpu_fence *fence = *rdst;

      if (amdgpu_fence_is_syncobj(fence))
         amdgpu_cs_destroy_syncobj(fence->ws->dev, fence->syncobj);
      else
         amdgpu_ctx_unref(fence->ctx);

      util_queue_fence_destroy(&fence->submitted);
      FREE(fence);
   }
   *rdst = rsrc;
}

int amdgpu_lookup_buffer(struct amdgpu_cs_context *cs, struct amdgpu_winsys_bo *bo);

static inline struct amdgpu_ib *
amdgpu_ib(struct radeon_cmdbuf *base)
{
   return (struct amdgpu_ib *)base;
}

static inline struct amdgpu_cs *
amdgpu_cs(struct radeon_cmdbuf *base)
{
   assert(amdgpu_ib(base)->ib_type == IB_MAIN);
   return (struct amdgpu_cs*)base;
}

#define get_container(member_ptr, container_type, container_member) \
   (container_type *)((char *)(member_ptr) - offsetof(container_type, container_member))

static inline struct amdgpu_cs *
amdgpu_cs_from_ib(struct amdgpu_ib *ib)
{
   switch (ib->ib_type) {
   case IB_MAIN:
      return get_container(ib, struct amdgpu_cs, main);
   default:
      unreachable("bad ib_type");
   }
}

static inline bool
amdgpu_bo_is_referenced_by_cs(struct amdgpu_cs *cs,
                              struct amdgpu_winsys_bo *bo)
{
   int num_refs = bo->num_cs_references;
   return num_refs == bo->ws->num_cs ||
         (num_refs && amdgpu_lookup_buffer(cs->csc, bo) != -1);
}

static inline bool
amdgpu_bo_is_referenced_by_cs_with_usage(struct amdgpu_cs *cs,
                                         struct amdgpu_winsys_bo *bo,
                                         enum radeon_bo_usage usage)
{
   int index;
   struct amdgpu_cs_buffer *buffer;

   if (!bo->num_cs_references)
      return false;

   index = amdgpu_lookup_buffer(cs->csc, bo);
   if (index == -1)
      return false;

   buffer = bo->bo ? &cs->csc->real_buffers[index] :
            bo->sparse ? &cs->csc->sparse_buffers[index] :
            &cs->csc->slab_buffers[index];

   return (buffer->usage & usage) != 0;
}

static inline bool
amdgpu_bo_is_referenced_by_any_cs(struct amdgpu_winsys_bo *bo)
{
   return bo->num_cs_references != 0;
}

bool amdgpu_fence_wait(struct pipe_fence_handle *fence, uint64_t timeout,
                       bool absolute);
void amdgpu_add_fences(struct amdgpu_winsys_bo *bo,
                       unsigned num_fences,
                       struct pipe_fence_handle **fences);
void amdgpu_cs_sync_flush(struct radeon_cmdbuf *rcs);
void amdgpu_cs_init_functions(struct amdgpu_winsys *ws);
void amdgpu_cs_submit_ib(void *job, int thread_index);

#endif
