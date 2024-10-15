/*
 * Copyright © 2008 Jérôme Glisse
 * Copyright © 2010 Marek Olšák <maraeo@gmail.com>
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
#include <inttypes.h>
#include <stdio.h>

#include "amd/common/sid.h"

#ifndef AMDGPU_IB_FLAG_TC_WB_NOT_INVALIDATE
#define AMDGPU_IB_FLAG_TC_WB_NOT_INVALIDATE (1 << 3)
#endif

#ifndef AMDGPU_CHUNK_ID_BO_HANDLES
#define AMDGPU_CHUNK_ID_BO_HANDLES 0x06
#endif

DEBUG_GET_ONCE_BOOL_OPTION(noop, "RADEON_NOOP", false)

/* FENCES */

static struct pipe_fence_handle *
amdgpu_fence_create(struct amdgpu_ctx *ctx, unsigned ip_type,
                    unsigned ip_instance, unsigned ring)
{
   struct amdgpu_fence *fence = CALLOC_STRUCT(amdgpu_fence);

   fence->reference.count = 1;
   fence->ws = ctx->ws;
   fence->ctx = ctx;
   fence->fence.context = ctx->ctx;
   fence->fence.ip_type = ip_type;
   fence->fence.ip_instance = ip_instance;
   fence->fence.ring = ring;
   util_queue_fence_init(&fence->submitted);
   util_queue_fence_reset(&fence->submitted);
   p_atomic_inc(&ctx->refcount);
   return (struct pipe_fence_handle *)fence;
}

static struct pipe_fence_handle *
amdgpu_fence_import_syncobj(struct radeon_winsys *rws, int fd)
{
   struct amdgpu_winsys *ws = amdgpu_winsys(rws);
   struct amdgpu_fence *fence = CALLOC_STRUCT(amdgpu_fence);
   int r;

   if (!fence)
      return NULL;

   pipe_reference_init(&fence->reference, 1);
   fence->ws = ws;

   r = amdgpu_cs_import_syncobj(ws->dev, fd, &fence->syncobj);
   if (r) {
      FREE(fence);
      return NULL;
   }

   util_queue_fence_init(&fence->submitted);

   assert(amdgpu_fence_is_syncobj(fence));
   return (struct pipe_fence_handle*)fence;
}

static struct pipe_fence_handle *
amdgpu_fence_import_sync_file(struct radeon_winsys *rws, int fd)
{
   struct amdgpu_winsys *ws = amdgpu_winsys(rws);
   struct amdgpu_fence *fence = CALLOC_STRUCT(amdgpu_fence);

   if (!fence)
      return NULL;

   pipe_reference_init(&fence->reference, 1);
   fence->ws = ws;
   /* fence->ctx == NULL means that the fence is syncobj-based. */

   /* Convert sync_file into syncobj. */
   int r = amdgpu_cs_create_syncobj(ws->dev, &fence->syncobj);
   if (r) {
      FREE(fence);
      return NULL;
   }

   r = amdgpu_cs_syncobj_import_sync_file(ws->dev, fence->syncobj, fd);
   if (r) {
      amdgpu_cs_destroy_syncobj(ws->dev, fence->syncobj);
      FREE(fence);
      return NULL;
   }

   util_queue_fence_init(&fence->submitted);

   return (struct pipe_fence_handle*)fence;
}

static int amdgpu_fence_export_sync_file(struct radeon_winsys *rws,
					 struct pipe_fence_handle *pfence)
{
   struct amdgpu_winsys *ws = amdgpu_winsys(rws);
   struct amdgpu_fence *fence = (struct amdgpu_fence*)pfence;

   if (amdgpu_fence_is_syncobj(fence)) {
      int fd, r;

      /* Convert syncobj into sync_file. */
      r = amdgpu_cs_syncobj_export_sync_file(ws->dev, fence->syncobj, &fd);
      return r ? -1 : fd;
   }

   util_queue_fence_wait(&fence->submitted);

   /* Convert the amdgpu fence into a fence FD. */
   int fd;
   if (amdgpu_cs_fence_to_handle(ws->dev, &fence->fence,
                                 AMDGPU_FENCE_TO_HANDLE_GET_SYNC_FILE_FD,
                                 (uint32_t*)&fd))
      return -1;

   return fd;
}

static int amdgpu_export_signalled_sync_file(struct radeon_winsys *rws)
{
   struct amdgpu_winsys *ws = amdgpu_winsys(rws);
   uint32_t syncobj;
   int fd = -1;

   int r = amdgpu_cs_create_syncobj2(ws->dev, DRM_SYNCOBJ_CREATE_SIGNALED,
                                     &syncobj);
   if (r) {
      return -1;
   }

   r = amdgpu_cs_syncobj_export_sync_file(ws->dev, syncobj, &fd);
   if (r) {
      fd = -1;
   }

   amdgpu_cs_destroy_syncobj(ws->dev, syncobj);
   return fd;
}

static void amdgpu_fence_submitted(struct pipe_fence_handle *fence,
                                   uint64_t seq_no,
                                   uint64_t *user_fence_cpu_address)
{
   struct amdgpu_fence *rfence = (struct amdgpu_fence*)fence;

   rfence->fence.fence = seq_no;
   rfence->user_fence_cpu_address = user_fence_cpu_address;
   util_queue_fence_signal(&rfence->submitted);
}

static void amdgpu_fence_signalled(struct pipe_fence_handle *fence)
{
   struct amdgpu_fence *rfence = (struct amdgpu_fence*)fence;

   rfence->signalled = true;
   util_queue_fence_signal(&rfence->submitted);
}

bool amdgpu_fence_wait(struct pipe_fence_handle *fence, uint64_t timeout,
                       bool absolute)
{
   struct amdgpu_fence *rfence = (struct amdgpu_fence*)fence;
   uint32_t expired;
   int64_t abs_timeout;
   uint64_t *user_fence_cpu;
   int r;

   if (rfence->signalled)
      return true;

   /* Handle syncobjs. */
   if (amdgpu_fence_is_syncobj(rfence)) {
      /* Absolute timeouts are only be used by BO fences, which aren't
       * backed by syncobjs.
       */
      assert(!absolute);

      if (amdgpu_cs_syncobj_wait(rfence->ws->dev, &rfence->syncobj, 1,
                                 timeout, 0, NULL))
         return false;

      rfence->signalled = true;
      return true;
   }

   if (absolute)
      abs_timeout = timeout;
   else
      abs_timeout = os_time_get_absolute_timeout(timeout);

   /* The fence might not have a number assigned if its IB is being
    * submitted in the other thread right now. Wait until the submission
    * is done. */
   if (!util_queue_fence_wait_timeout(&rfence->submitted, abs_timeout))
      return false;

   user_fence_cpu = rfence->user_fence_cpu_address;
   if (user_fence_cpu) {
      if (*user_fence_cpu >= rfence->fence.fence) {
         rfence->signalled = true;
         return true;
      }

      /* No timeout, just query: no need for the ioctl. */
      if (!absolute && !timeout)
         return false;
   }

   /* Now use the libdrm query. */
   r = amdgpu_cs_query_fence_status(&rfence->fence,
				    abs_timeout,
				    AMDGPU_QUERY_FENCE_TIMEOUT_IS_ABSOLUTE,
				    &expired);
   if (r) {
      fprintf(stderr, "amdgpu: amdgpu_cs_query_fence_status failed.\n");
      return false;
   }

   if (expired) {
      /* This variable can only transition from false to true, so it doesn't
       * matter if threads race for it. */
      rfence->signalled = true;
      return true;
   }
   return false;
}

static bool amdgpu_fence_wait_rel_timeout(struct radeon_winsys *rws,
                                          struct pipe_fence_handle *fence,
                                          uint64_t timeout)
{
   return amdgpu_fence_wait(fence, timeout, false);
}

static struct pipe_fence_handle *
amdgpu_cs_get_next_fence(struct radeon_cmdbuf *rcs)
{
   struct amdgpu_cs *cs = amdgpu_cs(rcs);
   struct pipe_fence_handle *fence = NULL;

   if (debug_get_option_noop())
      return NULL;

   if (cs->next_fence) {
      amdgpu_fence_reference(&fence, cs->next_fence);
      return fence;
   }

   fence = amdgpu_fence_create(cs->ctx,
                               cs->csc->ib[IB_MAIN].ip_type,
                               cs->csc->ib[IB_MAIN].ip_instance,
                               cs->csc->ib[IB_MAIN].ring);
   if (!fence)
      return NULL;

   amdgpu_fence_reference(&cs->next_fence, fence);
   return fence;
}

/* CONTEXTS */

static struct radeon_winsys_ctx *amdgpu_ctx_create(struct radeon_winsys *ws)
{
   struct amdgpu_ctx *ctx = CALLOC_STRUCT(amdgpu_ctx);
   int r;
   struct amdgpu_bo_alloc_request alloc_buffer = {};
   amdgpu_bo_handle buf_handle;

   if (!ctx)
      return NULL;

   ctx->ws = amdgpu_winsys(ws);
   ctx->refcount = 1;
   ctx->initial_num_total_rejected_cs = ctx->ws->num_total_rejected_cs;

   r = amdgpu_cs_ctx_create(ctx->ws->dev, &ctx->ctx);
   if (r) {
      fprintf(stderr, "amdgpu: amdgpu_cs_ctx_create failed. (%i)\n", r);
      goto error_create;
   }

   alloc_buffer.alloc_size = ctx->ws->info.gart_page_size;
   alloc_buffer.phys_alignment = ctx->ws->info.gart_page_size;
   alloc_buffer.preferred_heap = AMDGPU_GEM_DOMAIN_GTT;

   r = amdgpu_bo_alloc(ctx->ws->dev, &alloc_buffer, &buf_handle);
   if (r) {
      fprintf(stderr, "amdgpu: amdgpu_bo_alloc failed. (%i)\n", r);
      goto error_user_fence_alloc;
   }

   r = amdgpu_bo_cpu_map(buf_handle, (void**)&ctx->user_fence_cpu_address_base);
   if (r) {
      fprintf(stderr, "amdgpu: amdgpu_bo_cpu_map failed. (%i)\n", r);
      goto error_user_fence_map;
   }

   memset(ctx->user_fence_cpu_address_base, 0, alloc_buffer.alloc_size);
   ctx->user_fence_bo = buf_handle;

   return (struct radeon_winsys_ctx*)ctx;

error_user_fence_map:
   amdgpu_bo_free(buf_handle);
error_user_fence_alloc:
   amdgpu_cs_ctx_free(ctx->ctx);
error_create:
   FREE(ctx);
   return NULL;
}

static void amdgpu_ctx_destroy(struct radeon_winsys_ctx *rwctx)
{
   amdgpu_ctx_unref((struct amdgpu_ctx*)rwctx);
}

static enum pipe_reset_status
amdgpu_ctx_query_reset_status(struct radeon_winsys_ctx *rwctx)
{
   struct amdgpu_ctx *ctx = (struct amdgpu_ctx*)rwctx;
   uint32_t result, hangs;
   int r;

   /* Return a failure due to a rejected command submission. */
   if (ctx->ws->num_total_rejected_cs > ctx->initial_num_total_rejected_cs) {
      return ctx->num_rejected_cs ? PIPE_GUILTY_CONTEXT_RESET :
                                    PIPE_INNOCENT_CONTEXT_RESET;
   }

   /* Return a failure due to a GPU hang. */
   r = amdgpu_cs_query_reset_state(ctx->ctx, &result, &hangs);
   if (r) {
      fprintf(stderr, "amdgpu: amdgpu_cs_query_reset_state failed. (%i)\n", r);
      return PIPE_NO_RESET;
   }

   switch (result) {
   case AMDGPU_CTX_GUILTY_RESET:
      return PIPE_GUILTY_CONTEXT_RESET;
   case AMDGPU_CTX_INNOCENT_RESET:
      return PIPE_INNOCENT_CONTEXT_RESET;
   case AMDGPU_CTX_UNKNOWN_RESET:
      return PIPE_UNKNOWN_CONTEXT_RESET;
   case AMDGPU_CTX_NO_RESET:
   default:
      return PIPE_NO_RESET;
   }
}

/* COMMAND SUBMISSION */

static bool amdgpu_cs_has_user_fence(struct amdgpu_cs_context *cs)
{
   return cs->ib[IB_MAIN].ip_type != AMDGPU_HW_IP_UVD &&
          cs->ib[IB_MAIN].ip_type != AMDGPU_HW_IP_VCE &&
          cs->ib[IB_MAIN].ip_type != AMDGPU_HW_IP_UVD_ENC &&
          cs->ib[IB_MAIN].ip_type != AMDGPU_HW_IP_VCN_DEC &&
          cs->ib[IB_MAIN].ip_type != AMDGPU_HW_IP_VCN_ENC;
}

static bool amdgpu_cs_has_chaining(struct amdgpu_cs *cs)
{
   return cs->ctx->ws->info.chip_class >= CIK &&
          cs->ring_type == RING_GFX;
}

static unsigned amdgpu_cs_epilog_dws(enum ring_type ring_type)
{
   if (ring_type == RING_GFX)
      return 4; /* for chaining */

   return 0;
}

int amdgpu_lookup_buffer(struct amdgpu_cs_context *cs, struct amdgpu_winsys_bo *bo)
{
   unsigned hash = bo->unique_id & (ARRAY_SIZE(cs->buffer_indices_hashlist)-1);
   int i = cs->buffer_indices_hashlist[hash];
   struct amdgpu_cs_buffer *buffers;
   int num_buffers;

   if (bo->bo) {
      buffers = cs->real_buffers;
      num_buffers = cs->num_real_buffers;
   } else if (!bo->sparse) {
      buffers = cs->slab_buffers;
      num_buffers = cs->num_slab_buffers;
   } else {
      buffers = cs->sparse_buffers;
      num_buffers = cs->num_sparse_buffers;
   }

   /* not found or found */
   if (i < 0 || (i < num_buffers && buffers[i].bo == bo))
      return i;

   /* Hash collision, look for the BO in the list of buffers linearly. */
   for (i = num_buffers - 1; i >= 0; i--) {
      if (buffers[i].bo == bo) {
         /* Put this buffer in the hash list.
          * This will prevent additional hash collisions if there are
          * several consecutive lookup_buffer calls for the same buffer.
          *
          * Example: Assuming buffers A,B,C collide in the hash list,
          * the following sequence of buffers:
          *         AAAAAAAAAAABBBBBBBBBBBBBBCCCCCCCC
          * will collide here: ^ and here:   ^,
          * meaning that we should get very few collisions in the end. */
         cs->buffer_indices_hashlist[hash] = i;
         return i;
      }
   }
   return -1;
}

static int
amdgpu_do_add_real_buffer(struct amdgpu_cs_context *cs, struct amdgpu_winsys_bo *bo)
{
   struct amdgpu_cs_buffer *buffer;
   int idx;

   /* New buffer, check if the backing array is large enough. */
   if (cs->num_real_buffers >= cs->max_real_buffers) {
      unsigned new_max =
         MAX2(cs->max_real_buffers + 16, (unsigned)(cs->max_real_buffers * 1.3));
      struct amdgpu_cs_buffer *new_buffers;

      new_buffers = MALLOC(new_max * sizeof(*new_buffers));

      if (!new_buffers) {
         fprintf(stderr, "amdgpu_do_add_buffer: allocation failed\n");
         FREE(new_buffers);
         return -1;
      }

      memcpy(new_buffers, cs->real_buffers, cs->num_real_buffers * sizeof(*new_buffers));

      FREE(cs->real_buffers);

      cs->max_real_buffers = new_max;
      cs->real_buffers = new_buffers;
   }

   idx = cs->num_real_buffers;
   buffer = &cs->real_buffers[idx];

   memset(buffer, 0, sizeof(*buffer));
   amdgpu_winsys_bo_reference(&buffer->bo, bo);
   p_atomic_inc(&bo->num_cs_references);
   cs->num_real_buffers++;

   return idx;
}

static int
amdgpu_lookup_or_add_real_buffer(struct amdgpu_cs *acs, struct amdgpu_winsys_bo *bo)
{
   struct amdgpu_cs_context *cs = acs->csc;
   unsigned hash;
   int idx = amdgpu_lookup_buffer(cs, bo);

   if (idx >= 0)
      return idx;

   idx = amdgpu_do_add_real_buffer(cs, bo);

   hash = bo->unique_id & (ARRAY_SIZE(cs->buffer_indices_hashlist)-1);
   cs->buffer_indices_hashlist[hash] = idx;

   if (bo->initial_domain & RADEON_DOMAIN_VRAM)
      acs->main.base.used_vram += bo->base.size;
   else if (bo->initial_domain & RADEON_DOMAIN_GTT)
      acs->main.base.used_gart += bo->base.size;

   return idx;
}

static int amdgpu_lookup_or_add_slab_buffer(struct amdgpu_cs *acs,
                                            struct amdgpu_winsys_bo *bo)
{
   struct amdgpu_cs_context *cs = acs->csc;
   struct amdgpu_cs_buffer *buffer;
   unsigned hash;
   int idx = amdgpu_lookup_buffer(cs, bo);
   int real_idx;

   if (idx >= 0)
      return idx;

   real_idx = amdgpu_lookup_or_add_real_buffer(acs, bo->u.slab.real);
   if (real_idx < 0)
      return -1;

   /* New buffer, check if the backing array is large enough. */
   if (cs->num_slab_buffers >= cs->max_slab_buffers) {
      unsigned new_max =
         MAX2(cs->max_slab_buffers + 16, (unsigned)(cs->max_slab_buffers * 1.3));
      struct amdgpu_cs_buffer *new_buffers;

      new_buffers = REALLOC(cs->slab_buffers,
                            cs->max_slab_buffers * sizeof(*new_buffers),
                            new_max * sizeof(*new_buffers));
      if (!new_buffers) {
         fprintf(stderr, "amdgpu_lookup_or_add_slab_buffer: allocation failed\n");
         return -1;
      }

      cs->max_slab_buffers = new_max;
      cs->slab_buffers = new_buffers;
   }

   idx = cs->num_slab_buffers;
   buffer = &cs->slab_buffers[idx];

   memset(buffer, 0, sizeof(*buffer));
   amdgpu_winsys_bo_reference(&buffer->bo, bo);
   buffer->u.slab.real_idx = real_idx;
   p_atomic_inc(&bo->num_cs_references);
   cs->num_slab_buffers++;

   hash = bo->unique_id & (ARRAY_SIZE(cs->buffer_indices_hashlist)-1);
   cs->buffer_indices_hashlist[hash] = idx;

   return idx;
}

static int amdgpu_lookup_or_add_sparse_buffer(struct amdgpu_cs *acs,
                                              struct amdgpu_winsys_bo *bo)
{
   struct amdgpu_cs_context *cs = acs->csc;
   struct amdgpu_cs_buffer *buffer;
   unsigned hash;
   int idx = amdgpu_lookup_buffer(cs, bo);

   if (idx >= 0)
      return idx;

   /* New buffer, check if the backing array is large enough. */
   if (cs->num_sparse_buffers >= cs->max_sparse_buffers) {
      unsigned new_max =
         MAX2(cs->max_sparse_buffers + 16, (unsigned)(cs->max_sparse_buffers * 1.3));
      struct amdgpu_cs_buffer *new_buffers;

      new_buffers = REALLOC(cs->sparse_buffers,
                            cs->max_sparse_buffers * sizeof(*new_buffers),
                            new_max * sizeof(*new_buffers));
      if (!new_buffers) {
         fprintf(stderr, "amdgpu_lookup_or_add_sparse_buffer: allocation failed\n");
         return -1;
      }

      cs->max_sparse_buffers = new_max;
      cs->sparse_buffers = new_buffers;
   }

   idx = cs->num_sparse_buffers;
   buffer = &cs->sparse_buffers[idx];

   memset(buffer, 0, sizeof(*buffer));
   amdgpu_winsys_bo_reference(&buffer->bo, bo);
   p_atomic_inc(&bo->num_cs_references);
   cs->num_sparse_buffers++;

   hash = bo->unique_id & (ARRAY_SIZE(cs->buffer_indices_hashlist)-1);
   cs->buffer_indices_hashlist[hash] = idx;

   /* We delay adding the backing buffers until we really have to. However,
    * we cannot delay accounting for memory use.
    */
   simple_mtx_lock(&bo->u.sparse.commit_lock);

   list_for_each_entry(struct amdgpu_sparse_backing, backing, &bo->u.sparse.backing, list) {
      if (bo->initial_domain & RADEON_DOMAIN_VRAM)
         acs->main.base.used_vram += backing->bo->base.size;
      else if (bo->initial_domain & RADEON_DOMAIN_GTT)
         acs->main.base.used_gart += backing->bo->base.size;
   }

   simple_mtx_unlock(&bo->u.sparse.commit_lock);

   return idx;
}

static unsigned amdgpu_cs_add_buffer(struct radeon_cmdbuf *rcs,
                                    struct pb_buffer *buf,
                                    enum radeon_bo_usage usage,
                                    enum radeon_bo_domain domains,
                                    enum radeon_bo_priority priority)
{
   /* Don't use the "domains" parameter. Amdgpu doesn't support changing
    * the buffer placement during command submission.
    */
   struct amdgpu_cs *acs = amdgpu_cs(rcs);
   struct amdgpu_cs_context *cs = acs->csc;
   struct amdgpu_winsys_bo *bo = (struct amdgpu_winsys_bo*)buf;
   struct amdgpu_cs_buffer *buffer;
   int index;

   /* Fast exit for no-op calls.
    * This is very effective with suballocators and linear uploaders that
    * are outside of the winsys.
    */
   if (bo == cs->last_added_bo &&
       (usage & cs->last_added_bo_usage) == usage &&
       (1u << priority) & cs->last_added_bo_priority_usage)
      return cs->last_added_bo_index;

   if (!bo->sparse) {
      if (!bo->bo) {
         index = amdgpu_lookup_or_add_slab_buffer(acs, bo);
         if (index < 0)
            return 0;

         buffer = &cs->slab_buffers[index];
         buffer->usage |= usage;

         usage &= ~RADEON_USAGE_SYNCHRONIZED;
         index = buffer->u.slab.real_idx;
      } else {
         index = amdgpu_lookup_or_add_real_buffer(acs, bo);
         if (index < 0)
            return 0;
      }

      buffer = &cs->real_buffers[index];
   } else {
      index = amdgpu_lookup_or_add_sparse_buffer(acs, bo);
      if (index < 0)
         return 0;

      buffer = &cs->sparse_buffers[index];
   }

   buffer->u.real.priority_usage |= 1u << priority;
   buffer->usage |= usage;

   cs->last_added_bo = bo;
   cs->last_added_bo_index = index;
   cs->last_added_bo_usage = buffer->usage;
   cs->last_added_bo_priority_usage = buffer->u.real.priority_usage;
   return index;
}

static bool amdgpu_ib_new_buffer(struct amdgpu_winsys *ws, struct amdgpu_ib *ib,
                                 enum ring_type ring_type)
{
   struct pb_buffer *pb;
   uint8_t *mapped;
   unsigned buffer_size;

   /* Always create a buffer that is at least as large as the maximum seen IB
    * size, aligned to a power of two (and multiplied by 4 to reduce internal
    * fragmentation if chaining is not available). Limit to 512k dwords, which
    * is the largest power of two that fits into the size field of the
    * INDIRECT_BUFFER packet.
    */
   if (amdgpu_cs_has_chaining(amdgpu_cs_from_ib(ib)))
      buffer_size = 4 *util_next_power_of_two(ib->max_ib_size);
   else
      buffer_size = 4 *util_next_power_of_two(4 * ib->max_ib_size);

   buffer_size = MIN2(buffer_size, 4 * 512 * 1024);

   switch (ib->ib_type) {
   case IB_MAIN:
      buffer_size = MAX2(buffer_size, 8 * 1024 * 4);
      break;
   default:
      unreachable("unhandled IB type");
   }

   pb = ws->base.buffer_create(&ws->base, buffer_size,
                               ws->info.gart_page_size,
                               RADEON_DOMAIN_GTT,
                               RADEON_FLAG_NO_INTERPROCESS_SHARING |
                               (ring_type == RING_GFX ||
                                ring_type == RING_COMPUTE ||
                                ring_type == RING_DMA ?
                                   RADEON_FLAG_READ_ONLY | RADEON_FLAG_GTT_WC : 0));
   if (!pb)
      return false;

   mapped = ws->base.buffer_map(pb, NULL, PIPE_TRANSFER_WRITE);
   if (!mapped) {
      pb_reference(&pb, NULL);
      return false;
   }

   pb_reference(&ib->big_ib_buffer, pb);
   pb_reference(&pb, NULL);

   ib->ib_mapped = mapped;
   ib->used_ib_space = 0;

   return true;
}

static unsigned amdgpu_ib_max_submit_dwords(enum ib_type ib_type)
{
   switch (ib_type) {
   case IB_MAIN:
      /* Smaller submits means the GPU gets busy sooner and there is less
       * waiting for buffers and fences. Proof:
       *   http://www.phoronix.com/scan.php?page=article&item=mesa-111-si&num=1
       */
      return 20 * 1024;
   default:
      unreachable("bad ib_type");
   }
}

static bool amdgpu_get_new_ib(struct radeon_winsys *ws, struct amdgpu_cs *cs,
                              enum ib_type ib_type)
{
   struct amdgpu_winsys *aws = (struct amdgpu_winsys*)ws;
   /* Small IBs are better than big IBs, because the GPU goes idle quicker
    * and there is less waiting for buffers and fences. Proof:
    *   http://www.phoronix.com/scan.php?page=article&item=mesa-111-si&num=1
    */
   struct amdgpu_ib *ib = NULL;
   struct drm_amdgpu_cs_chunk_ib *info = &cs->csc->ib[ib_type];
   unsigned ib_size = 0;

   switch (ib_type) {
   case IB_MAIN:
      ib = &cs->main;
      ib_size = 4 * 1024 * 4;
      break;
   default:
      unreachable("unhandled IB type");
   }

   if (!amdgpu_cs_has_chaining(cs)) {
      ib_size = MAX2(ib_size,
                     4 * MIN2(util_next_power_of_two(ib->max_ib_size),
                              amdgpu_ib_max_submit_dwords(ib_type)));
   }

   ib->max_ib_size = ib->max_ib_size - ib->max_ib_size / 32;

   ib->base.prev_dw = 0;
   ib->base.num_prev = 0;
   ib->base.current.cdw = 0;
   ib->base.current.buf = NULL;

   /* Allocate a new buffer for IBs if the current buffer is all used. */
   if (!ib->big_ib_buffer ||
       ib->used_ib_space + ib_size > ib->big_ib_buffer->size) {
      if (!amdgpu_ib_new_buffer(aws, ib, cs->ring_type))
         return false;
   }

   info->va_start = amdgpu_winsys_bo(ib->big_ib_buffer)->va + ib->used_ib_space;
   info->ib_bytes = 0;
   /* ib_bytes is in dwords and the conversion to bytes will be done before
    * the CS ioctl. */
   ib->ptr_ib_size = &info->ib_bytes;
   ib->ptr_ib_size_inside_ib = false;

   amdgpu_cs_add_buffer(&cs->main.base, ib->big_ib_buffer,
                        RADEON_USAGE_READ, 0, RADEON_PRIO_IB1);

   ib->base.current.buf = (uint32_t*)(ib->ib_mapped + ib->used_ib_space);

   ib_size = ib->big_ib_buffer->size - ib->used_ib_space;
   ib->base.current.max_dw = ib_size / 4 - amdgpu_cs_epilog_dws(cs->ring_type);
   return true;
}

static void amdgpu_set_ib_size(struct amdgpu_ib *ib)
{
   if (ib->ptr_ib_size_inside_ib) {
      *ib->ptr_ib_size = ib->base.current.cdw |
                         S_3F2_CHAIN(1) | S_3F2_VALID(1);
   } else {
      *ib->ptr_ib_size = ib->base.current.cdw;
   }
}

static void amdgpu_ib_finalize(struct amdgpu_winsys *ws, struct amdgpu_ib *ib)
{
   amdgpu_set_ib_size(ib);
   ib->used_ib_space += ib->base.current.cdw * 4;
   ib->used_ib_space = align(ib->used_ib_space, ws->info.ib_start_alignment);
   ib->max_ib_size = MAX2(ib->max_ib_size, ib->base.prev_dw + ib->base.current.cdw);
}

static bool amdgpu_init_cs_context(struct amdgpu_winsys *ws,
                                   struct amdgpu_cs_context *cs,
                                   enum ring_type ring_type)
{
   switch (ring_type) {
   case RING_DMA:
      cs->ib[IB_MAIN].ip_type = AMDGPU_HW_IP_DMA;
      break;

   case RING_UVD:
      cs->ib[IB_MAIN].ip_type = AMDGPU_HW_IP_UVD;
      break;

   case RING_UVD_ENC:
      cs->ib[IB_MAIN].ip_type = AMDGPU_HW_IP_UVD_ENC;
      break;

   case RING_VCE:
      cs->ib[IB_MAIN].ip_type = AMDGPU_HW_IP_VCE;
      break;

   case RING_VCN_DEC:
      cs->ib[IB_MAIN].ip_type = AMDGPU_HW_IP_VCN_DEC;
      break;

   case RING_VCN_ENC:
      cs->ib[IB_MAIN].ip_type = AMDGPU_HW_IP_VCN_ENC;
      break;

   case RING_VCN_JPEG:
      cs->ib[IB_MAIN].ip_type = AMDGPU_HW_IP_VCN_JPEG;
      break;

   case RING_COMPUTE:
   case RING_GFX:
      cs->ib[IB_MAIN].ip_type = ring_type == RING_GFX ? AMDGPU_HW_IP_GFX :
                                                        AMDGPU_HW_IP_COMPUTE;

      /* The kernel shouldn't invalidate L2 and vL1. The proper place for cache
       * invalidation is the beginning of IBs (the previous commit does that),
       * because completion of an IB doesn't care about the state of GPU caches,
       * but the beginning of an IB does. Draw calls from multiple IBs can be
       * executed in parallel, so draw calls from the current IB can finish after
       * the next IB starts drawing, and so the cache flush at the end of IB
       * is always late.
       */
      if (ws->info.drm_minor >= 26)
         cs->ib[IB_MAIN].flags = AMDGPU_IB_FLAG_TC_WB_NOT_INVALIDATE;
      break;

   default:
      assert(0);
   }

   memset(cs->buffer_indices_hashlist, -1, sizeof(cs->buffer_indices_hashlist));
   cs->last_added_bo = NULL;
   return true;
}

static void amdgpu_cs_context_cleanup(struct amdgpu_cs_context *cs)
{
   unsigned i;

   for (i = 0; i < cs->num_real_buffers; i++) {
      p_atomic_dec(&cs->real_buffers[i].bo->num_cs_references);
      amdgpu_winsys_bo_reference(&cs->real_buffers[i].bo, NULL);
   }
   for (i = 0; i < cs->num_slab_buffers; i++) {
      p_atomic_dec(&cs->slab_buffers[i].bo->num_cs_references);
      amdgpu_winsys_bo_reference(&cs->slab_buffers[i].bo, NULL);
   }
   for (i = 0; i < cs->num_sparse_buffers; i++) {
      p_atomic_dec(&cs->sparse_buffers[i].bo->num_cs_references);
      amdgpu_winsys_bo_reference(&cs->sparse_buffers[i].bo, NULL);
   }
   for (i = 0; i < cs->num_fence_dependencies; i++)
      amdgpu_fence_reference(&cs->fence_dependencies[i], NULL);
   for (i = 0; i < cs->num_syncobj_to_signal; i++)
      amdgpu_fence_reference(&cs->syncobj_to_signal[i], NULL);

   cs->num_real_buffers = 0;
   cs->num_slab_buffers = 0;
   cs->num_sparse_buffers = 0;
   cs->num_fence_dependencies = 0;
   cs->num_syncobj_to_signal = 0;
   amdgpu_fence_reference(&cs->fence, NULL);

   memset(cs->buffer_indices_hashlist, -1, sizeof(cs->buffer_indices_hashlist));
   cs->last_added_bo = NULL;
}

static void amdgpu_destroy_cs_context(struct amdgpu_cs_context *cs)
{
   amdgpu_cs_context_cleanup(cs);
   FREE(cs->real_buffers);
   FREE(cs->slab_buffers);
   FREE(cs->sparse_buffers);
   FREE(cs->fence_dependencies);
   FREE(cs->syncobj_to_signal);
}


static struct radeon_cmdbuf *
amdgpu_cs_create(struct radeon_winsys_ctx *rwctx,
                 enum ring_type ring_type,
                 void (*flush)(void *ctx, unsigned flags,
                               struct pipe_fence_handle **fence),
                 void *flush_ctx)
{
   struct amdgpu_ctx *ctx = (struct amdgpu_ctx*)rwctx;
   struct amdgpu_cs *cs;

   cs = CALLOC_STRUCT(amdgpu_cs);
   if (!cs) {
      return NULL;
   }

   util_queue_fence_init(&cs->flush_completed);

   cs->ctx = ctx;
   cs->flush_cs = flush;
   cs->flush_data = flush_ctx;
   cs->ring_type = ring_type;

   struct amdgpu_cs_fence_info fence_info;
   fence_info.handle = cs->ctx->user_fence_bo;
   fence_info.offset = cs->ring_type;
   amdgpu_cs_chunk_fence_info_to_data(&fence_info, (void*)&cs->fence_chunk);

   cs->main.ib_type = IB_MAIN;

   if (!amdgpu_init_cs_context(ctx->ws, &cs->csc1, ring_type)) {
      FREE(cs);
      return NULL;
   }

   if (!amdgpu_init_cs_context(ctx->ws, &cs->csc2, ring_type)) {
      amdgpu_destroy_cs_context(&cs->csc1);
      FREE(cs);
      return NULL;
   }

   /* Set the first submission context as current. */
   cs->csc = &cs->csc1;
   cs->cst = &cs->csc2;

   if (!amdgpu_get_new_ib(&ctx->ws->base, cs, IB_MAIN)) {
      amdgpu_destroy_cs_context(&cs->csc2);
      amdgpu_destroy_cs_context(&cs->csc1);
      FREE(cs);
      return NULL;
   }

   p_atomic_inc(&ctx->ws->num_cs);
   return &cs->main.base;
}

static bool amdgpu_cs_validate(struct radeon_cmdbuf *rcs)
{
   return true;
}

static bool amdgpu_cs_check_space(struct radeon_cmdbuf *rcs, unsigned dw)
{
   struct amdgpu_ib *ib = amdgpu_ib(rcs);
   struct amdgpu_cs *cs = amdgpu_cs_from_ib(ib);
   unsigned requested_size = rcs->prev_dw + rcs->current.cdw + dw;
   uint64_t va;
   uint32_t *new_ptr_ib_size;

   assert(rcs->current.cdw <= rcs->current.max_dw);

   if (requested_size > amdgpu_ib_max_submit_dwords(ib->ib_type))
      return false;

   ib->max_ib_size = MAX2(ib->max_ib_size, requested_size);

   if (rcs->current.max_dw - rcs->current.cdw >= dw)
      return true;

   if (!amdgpu_cs_has_chaining(cs))
      return false;

   /* Allocate a new chunk */
   if (rcs->num_prev >= rcs->max_prev) {
      unsigned new_max_prev = MAX2(1, 2 * rcs->max_prev);
      struct radeon_cmdbuf_chunk *new_prev;

      new_prev = REALLOC(rcs->prev,
                         sizeof(*new_prev) * rcs->max_prev,
                         sizeof(*new_prev) * new_max_prev);
      if (!new_prev)
         return false;

      rcs->prev = new_prev;
      rcs->max_prev = new_max_prev;
   }

   if (!amdgpu_ib_new_buffer(cs->ctx->ws, ib, cs->ring_type))
      return false;

   assert(ib->used_ib_space == 0);
   va = amdgpu_winsys_bo(ib->big_ib_buffer)->va;

   /* This space was originally reserved. */
   rcs->current.max_dw += 4;
   assert(ib->used_ib_space + 4 * rcs->current.max_dw <= ib->big_ib_buffer->size);

   /* Pad with NOPs and add INDIRECT_BUFFER packet */
   while ((rcs->current.cdw & 7) != 4)
      radeon_emit(rcs, 0xffff1000); /* type3 nop packet */

   radeon_emit(rcs, PKT3(ib->ib_type == IB_MAIN ? PKT3_INDIRECT_BUFFER_CIK
                                           : PKT3_INDIRECT_BUFFER_CONST, 2, 0));
   radeon_emit(rcs, va);
   radeon_emit(rcs, va >> 32);
   new_ptr_ib_size = &rcs->current.buf[rcs->current.cdw++];

   assert((rcs->current.cdw & 7) == 0);
   assert(rcs->current.cdw <= rcs->current.max_dw);

   amdgpu_set_ib_size(ib);
   ib->ptr_ib_size = new_ptr_ib_size;
   ib->ptr_ib_size_inside_ib = true;

   /* Hook up the new chunk */
   rcs->prev[rcs->num_prev].buf = rcs->current.buf;
   rcs->prev[rcs->num_prev].cdw = rcs->current.cdw;
   rcs->prev[rcs->num_prev].max_dw = rcs->current.cdw; /* no modifications */
   rcs->num_prev++;

   ib->base.prev_dw += ib->base.current.cdw;
   ib->base.current.cdw = 0;

   ib->base.current.buf = (uint32_t*)(ib->ib_mapped + ib->used_ib_space);
   ib->base.current.max_dw = ib->big_ib_buffer->size / 4 - amdgpu_cs_epilog_dws(cs->ring_type);

   amdgpu_cs_add_buffer(&cs->main.base, ib->big_ib_buffer,
                        RADEON_USAGE_READ, 0, RADEON_PRIO_IB1);

   return true;
}

static unsigned amdgpu_cs_get_buffer_list(struct radeon_cmdbuf *rcs,
                                          struct radeon_bo_list_item *list)
{
    struct amdgpu_cs_context *cs = amdgpu_cs(rcs)->csc;
    int i;

    if (list) {
        for (i = 0; i < cs->num_real_buffers; i++) {
            list[i].bo_size = cs->real_buffers[i].bo->base.size;
            list[i].vm_address = cs->real_buffers[i].bo->va;
            list[i].priority_usage = cs->real_buffers[i].u.real.priority_usage;
        }
    }
    return cs->num_real_buffers;
}

static unsigned add_fence_dependency_entry(struct amdgpu_cs_context *cs)
{
   unsigned idx = cs->num_fence_dependencies++;

   if (idx >= cs->max_fence_dependencies) {
      unsigned size;
      const unsigned increment = 8;

      cs->max_fence_dependencies = idx + increment;
      size = cs->max_fence_dependencies * sizeof(cs->fence_dependencies[0]);
      cs->fence_dependencies = realloc(cs->fence_dependencies, size);
      /* Clear the newly-allocated elements. */
      memset(cs->fence_dependencies + idx, 0,
             increment * sizeof(cs->fence_dependencies[0]));
   }
   return idx;
}

static bool is_noop_fence_dependency(struct amdgpu_cs *acs,
                                     struct amdgpu_fence *fence)
{
   struct amdgpu_cs_context *cs = acs->csc;

   if (!amdgpu_fence_is_syncobj(fence) &&
       fence->ctx == acs->ctx &&
       fence->fence.ip_type == cs->ib[IB_MAIN].ip_type &&
       fence->fence.ip_instance == cs->ib[IB_MAIN].ip_instance &&
       fence->fence.ring == cs->ib[IB_MAIN].ring)
      return true;

   return amdgpu_fence_wait((void *)fence, 0, false);
}

static void amdgpu_cs_add_fence_dependency(struct radeon_cmdbuf *rws,
                                           struct pipe_fence_handle *pfence)
{
   struct amdgpu_cs *acs = amdgpu_cs(rws);
   struct amdgpu_cs_context *cs = acs->csc;
   struct amdgpu_fence *fence = (struct amdgpu_fence*)pfence;

   util_queue_fence_wait(&fence->submitted);

   if (is_noop_fence_dependency(acs, fence))
      return;

   unsigned idx = add_fence_dependency_entry(cs);
   amdgpu_fence_reference(&cs->fence_dependencies[idx],
                          (struct pipe_fence_handle*)fence);
}

static void amdgpu_add_bo_fence_dependencies(struct amdgpu_cs *acs,
                                             struct amdgpu_cs_buffer *buffer)
{
   struct amdgpu_cs_context *cs = acs->csc;
   struct amdgpu_winsys_bo *bo = buffer->bo;
   unsigned new_num_fences = 0;

   for (unsigned j = 0; j < bo->num_fences; ++j) {
      struct amdgpu_fence *bo_fence = (void *)bo->fences[j];

      if (is_noop_fence_dependency(acs, bo_fence))
         continue;

      amdgpu_fence_reference(&bo->fences[new_num_fences], bo->fences[j]);
      new_num_fences++;

      if (!(buffer->usage & RADEON_USAGE_SYNCHRONIZED))
         continue;

      unsigned idx = add_fence_dependency_entry(cs);
      amdgpu_fence_reference(&cs->fence_dependencies[idx],
                             (struct pipe_fence_handle*)bo_fence);
   }

   for (unsigned j = new_num_fences; j < bo->num_fences; ++j)
      amdgpu_fence_reference(&bo->fences[j], NULL);

   bo->num_fences = new_num_fences;
}

/* Add the given list of fences to the buffer's fence list.
 *
 * Must be called with the winsys bo_fence_lock held.
 */
void amdgpu_add_fences(struct amdgpu_winsys_bo *bo,
                       unsigned num_fences,
                       struct pipe_fence_handle **fences)
{
   if (bo->num_fences + num_fences > bo->max_fences) {
      unsigned new_max_fences = MAX2(bo->num_fences + num_fences, bo->max_fences * 2);
      struct pipe_fence_handle **new_fences =
         REALLOC(bo->fences,
                 bo->num_fences * sizeof(*new_fences),
                 new_max_fences * sizeof(*new_fences));
      if (likely(new_fences)) {
         bo->fences = new_fences;
         bo->max_fences = new_max_fences;
      } else {
         unsigned drop;

         fprintf(stderr, "amdgpu_add_fences: allocation failure, dropping fence(s)\n");
         if (!bo->num_fences)
            return;

         bo->num_fences--; /* prefer to keep the most recent fence if possible */
         amdgpu_fence_reference(&bo->fences[bo->num_fences], NULL);

         drop = bo->num_fences + num_fences - bo->max_fences;
         num_fences -= drop;
         fences += drop;
      }
   }

   for (unsigned i = 0; i < num_fences; ++i) {
      bo->fences[bo->num_fences] = NULL;
      amdgpu_fence_reference(&bo->fences[bo->num_fences], fences[i]);
      bo->num_fences++;
   }
}

static void amdgpu_add_fence_dependencies_bo_list(struct amdgpu_cs *acs,
                                                  struct pipe_fence_handle *fence,
                                                  unsigned num_buffers,
                                                  struct amdgpu_cs_buffer *buffers)
{
   for (unsigned i = 0; i < num_buffers; i++) {
      struct amdgpu_cs_buffer *buffer = &buffers[i];
      struct amdgpu_winsys_bo *bo = buffer->bo;

      amdgpu_add_bo_fence_dependencies(acs, buffer);
      p_atomic_inc(&bo->num_active_ioctls);
      amdgpu_add_fences(bo, 1, &fence);
   }
}

/* Since the kernel driver doesn't synchronize execution between different
 * rings automatically, we have to add fence dependencies manually.
 */
static void amdgpu_add_fence_dependencies_bo_lists(struct amdgpu_cs *acs)
{
   struct amdgpu_cs_context *cs = acs->csc;

   cs->num_fence_dependencies = 0;

   amdgpu_add_fence_dependencies_bo_list(acs, cs->fence, cs->num_real_buffers, cs->real_buffers);
   amdgpu_add_fence_dependencies_bo_list(acs, cs->fence, cs->num_slab_buffers, cs->slab_buffers);
   amdgpu_add_fence_dependencies_bo_list(acs, cs->fence, cs->num_sparse_buffers, cs->sparse_buffers);
}

static unsigned add_syncobj_to_signal_entry(struct amdgpu_cs_context *cs)
{
   unsigned idx = cs->num_syncobj_to_signal++;

   if (idx >= cs->max_syncobj_to_signal) {
      unsigned size;
      const unsigned increment = 8;

      cs->max_syncobj_to_signal = idx + increment;
      size = cs->max_syncobj_to_signal * sizeof(cs->syncobj_to_signal[0]);
      cs->syncobj_to_signal = realloc(cs->syncobj_to_signal, size);
      /* Clear the newly-allocated elements. */
      memset(cs->syncobj_to_signal + idx, 0,
             increment * sizeof(cs->syncobj_to_signal[0]));
   }
   return idx;
}

static void amdgpu_cs_add_syncobj_signal(struct radeon_cmdbuf *rws,
                                         struct pipe_fence_handle *fence)
{
   struct amdgpu_cs *acs = amdgpu_cs(rws);
   struct amdgpu_cs_context *cs = acs->csc;

   assert(amdgpu_fence_is_syncobj((struct amdgpu_fence *)fence));

   unsigned idx = add_syncobj_to_signal_entry(cs);
   amdgpu_fence_reference(&cs->syncobj_to_signal[idx], fence);
}

/* Add backing of sparse buffers to the buffer list.
 *
 * This is done late, during submission, to keep the buffer list short before
 * submit, and to avoid managing fences for the backing buffers.
 */
static bool amdgpu_add_sparse_backing_buffers(struct amdgpu_cs_context *cs)
{
   for (unsigned i = 0; i < cs->num_sparse_buffers; ++i) {
      struct amdgpu_cs_buffer *buffer = &cs->sparse_buffers[i];
      struct amdgpu_winsys_bo *bo = buffer->bo;

      simple_mtx_lock(&bo->u.sparse.commit_lock);

      list_for_each_entry(struct amdgpu_sparse_backing, backing, &bo->u.sparse.backing, list) {
         /* We can directly add the buffer here, because we know that each
          * backing buffer occurs only once.
          */
         int idx = amdgpu_do_add_real_buffer(cs, backing->bo);
         if (idx < 0) {
            fprintf(stderr, "%s: failed to add buffer\n", __FUNCTION__);
            simple_mtx_unlock(&bo->u.sparse.commit_lock);
            return false;
         }

         cs->real_buffers[idx].usage = buffer->usage & ~RADEON_USAGE_SYNCHRONIZED;
         cs->real_buffers[idx].u.real.priority_usage = buffer->u.real.priority_usage;
         p_atomic_inc(&backing->bo->num_active_ioctls);
      }

      simple_mtx_unlock(&bo->u.sparse.commit_lock);
   }

   return true;
}

void amdgpu_cs_submit_ib(void *job, int thread_index)
{
   struct amdgpu_cs *acs = (struct amdgpu_cs*)job;
   struct amdgpu_winsys *ws = acs->ctx->ws;
   struct amdgpu_cs_context *cs = acs->cst;
   int i, r;
   amdgpu_bo_list_handle bo_list = NULL;
   uint64_t seq_no = 0;
   bool has_user_fence = amdgpu_cs_has_user_fence(cs);
   bool use_bo_list_create = ws->info.drm_minor < 27;
   struct drm_amdgpu_bo_list_in bo_list_in;

   /* Prepare the buffer list. */
   if (ws->debug_all_bos) {
      /* The buffer list contains all buffers. This is a slow path that
       * ensures that no buffer is missing in the BO list.
       */
      struct amdgpu_winsys_bo *bo;
      amdgpu_bo_handle *handles;
      unsigned num = 0;

      simple_mtx_lock(&ws->global_bo_list_lock);
      handles = alloca(sizeof(handles[0]) * ws->num_buffers);

      LIST_FOR_EACH_ENTRY(bo, &ws->global_bo_list, u.real.global_list_item) {
         assert(num < ws->num_buffers);
         handles[num++] = bo->bo;
      }

      r = amdgpu_bo_list_create(ws->dev, ws->num_buffers,
                                handles, NULL, &bo_list);
      simple_mtx_unlock(&ws->global_bo_list_lock);
      if (r) {
         fprintf(stderr, "amdgpu: buffer list creation failed (%d)\n", r);
         goto cleanup;
      }
   } else if (!use_bo_list_create) {
      /* Standard path passing the buffer list via the CS ioctl. */
      if (!amdgpu_add_sparse_backing_buffers(cs)) {
         fprintf(stderr, "amdgpu: amdgpu_add_sparse_backing_buffers failed\n");
         r = -ENOMEM;
         goto cleanup;
      }

      struct drm_amdgpu_bo_list_entry *list =
         alloca(cs->num_real_buffers * sizeof(struct drm_amdgpu_bo_list_entry));

      unsigned num_handles = 0;
      for (i = 0; i < cs->num_real_buffers; ++i) {
         struct amdgpu_cs_buffer *buffer = &cs->real_buffers[i];

         if (buffer->bo->is_local)
            continue;

         assert(buffer->u.real.priority_usage != 0);

         list[num_handles].bo_handle = buffer->bo->u.real.kms_handle;
         list[num_handles].bo_priority = (util_last_bit(buffer->u.real.priority_usage) - 1) / 2;
         ++num_handles;
      }

      bo_list_in.operation = ~0;
      bo_list_in.list_handle = ~0;
      bo_list_in.bo_number = num_handles;
      bo_list_in.bo_info_size = sizeof(struct drm_amdgpu_bo_list_entry);
      bo_list_in.bo_info_ptr = (uint64_t)(uintptr_t)list;
   } else {
      /* Legacy path creating the buffer list handle and passing it to the CS ioctl. */
      unsigned num_handles;

      if (!amdgpu_add_sparse_backing_buffers(cs)) {
         fprintf(stderr, "amdgpu: amdgpu_add_sparse_backing_buffers failed\n");
         r = -ENOMEM;
         goto cleanup;
      }

      amdgpu_bo_handle *handles = alloca(sizeof(*handles) * cs->num_real_buffers);
      uint8_t *flags = alloca(sizeof(*flags) * cs->num_real_buffers);

      num_handles = 0;
      for (i = 0; i < cs->num_real_buffers; ++i) {
         struct amdgpu_cs_buffer *buffer = &cs->real_buffers[i];

	 if (buffer->bo->is_local)
            continue;

         assert(buffer->u.real.priority_usage != 0);

         handles[num_handles] = buffer->bo->bo;
         flags[num_handles] = (util_last_bit(buffer->u.real.priority_usage) - 1) / 2;
	 ++num_handles;
      }

      if (num_handles) {
         r = amdgpu_bo_list_create(ws->dev, num_handles,
                                   handles, flags, &bo_list);
         if (r) {
            fprintf(stderr, "amdgpu: buffer list creation failed (%d)\n", r);
            goto cleanup;
         }
      }
   }

   if (acs->ring_type == RING_GFX)
      ws->gfx_bo_list_counter += cs->num_real_buffers;

   if (acs->ctx->num_rejected_cs) {
      r = -ECANCELED;
   } else {
      struct drm_amdgpu_cs_chunk chunks[6];
      unsigned num_chunks = 0;

      /* Convert from dwords to bytes. */
      cs->ib[IB_MAIN].ib_bytes *= 4;

      /* IB */
      chunks[num_chunks].chunk_id = AMDGPU_CHUNK_ID_IB;
      chunks[num_chunks].length_dw = sizeof(struct drm_amdgpu_cs_chunk_ib) / 4;
      chunks[num_chunks].chunk_data = (uintptr_t)&cs->ib[IB_MAIN];
      num_chunks++;

      /* Fence */
      if (has_user_fence) {
         chunks[num_chunks].chunk_id = AMDGPU_CHUNK_ID_FENCE;
         chunks[num_chunks].length_dw = sizeof(struct drm_amdgpu_cs_chunk_fence) / 4;
         chunks[num_chunks].chunk_data = (uintptr_t)&acs->fence_chunk;
         num_chunks++;
      }

      /* Dependencies */
      unsigned num_dependencies = cs->num_fence_dependencies;
      unsigned num_syncobj_dependencies = 0;

      if (num_dependencies) {
         struct drm_amdgpu_cs_chunk_dep *dep_chunk =
            alloca(num_dependencies * sizeof(*dep_chunk));
         unsigned num = 0;

         for (unsigned i = 0; i < num_dependencies; i++) {
            struct amdgpu_fence *fence =
               (struct amdgpu_fence*)cs->fence_dependencies[i];

            if (amdgpu_fence_is_syncobj(fence)) {
               num_syncobj_dependencies++;
               continue;
            }

            assert(util_queue_fence_is_signalled(&fence->submitted));
            amdgpu_cs_chunk_fence_to_dep(&fence->fence, &dep_chunk[num++]);
         }

         chunks[num_chunks].chunk_id = AMDGPU_CHUNK_ID_DEPENDENCIES;
         chunks[num_chunks].length_dw = sizeof(dep_chunk[0]) / 4 * num;
         chunks[num_chunks].chunk_data = (uintptr_t)dep_chunk;
         num_chunks++;
      }

      /* Syncobj dependencies. */
      if (num_syncobj_dependencies) {
         struct drm_amdgpu_cs_chunk_sem *sem_chunk =
            alloca(num_syncobj_dependencies * sizeof(sem_chunk[0]));
         unsigned num = 0;

         for (unsigned i = 0; i < num_dependencies; i++) {
            struct amdgpu_fence *fence =
               (struct amdgpu_fence*)cs->fence_dependencies[i];

            if (!amdgpu_fence_is_syncobj(fence))
               continue;

            assert(util_queue_fence_is_signalled(&fence->submitted));
            sem_chunk[num++].handle = fence->syncobj;
         }

         chunks[num_chunks].chunk_id = AMDGPU_CHUNK_ID_SYNCOBJ_IN;
         chunks[num_chunks].length_dw = sizeof(sem_chunk[0]) / 4 * num;
         chunks[num_chunks].chunk_data = (uintptr_t)sem_chunk;
         num_chunks++;
      }

      /* Syncobj sygnals. */
      if (cs->num_syncobj_to_signal) {
         struct drm_amdgpu_cs_chunk_sem *sem_chunk =
            alloca(cs->num_syncobj_to_signal * sizeof(sem_chunk[0]));

         for (unsigned i = 0; i < cs->num_syncobj_to_signal; i++) {
            struct amdgpu_fence *fence =
               (struct amdgpu_fence*)cs->syncobj_to_signal[i];

            assert(amdgpu_fence_is_syncobj(fence));
            sem_chunk[i].handle = fence->syncobj;
         }

         chunks[num_chunks].chunk_id = AMDGPU_CHUNK_ID_SYNCOBJ_OUT;
         chunks[num_chunks].length_dw = sizeof(sem_chunk[0]) / 4
                                        * cs->num_syncobj_to_signal;
         chunks[num_chunks].chunk_data = (uintptr_t)sem_chunk;
         num_chunks++;
      }

      /* BO list */
      if (!use_bo_list_create) {
         chunks[num_chunks].chunk_id = AMDGPU_CHUNK_ID_BO_HANDLES;
         chunks[num_chunks].length_dw = sizeof(struct drm_amdgpu_bo_list_in) / 4;
         chunks[num_chunks].chunk_data = (uintptr_t)&bo_list_in;
         num_chunks++;
      }

      assert(num_chunks <= ARRAY_SIZE(chunks));

      r = amdgpu_cs_submit_raw(ws->dev, acs->ctx->ctx, bo_list,
                               num_chunks, chunks, &seq_no);
   }

   if (r) {
      if (r == -ENOMEM)
         fprintf(stderr, "amdgpu: Not enough memory for command submission.\n");
      else if (r == -ECANCELED)
         fprintf(stderr, "amdgpu: The CS has been cancelled because the context is lost.\n");
      else
         fprintf(stderr, "amdgpu: The CS has been rejected, "
                 "see dmesg for more information (%i).\n", r);

      acs->ctx->num_rejected_cs++;
      ws->num_total_rejected_cs++;
   } else {
      /* Success. */
      uint64_t *user_fence = NULL;

      if (has_user_fence)
         user_fence = acs->ctx->user_fence_cpu_address_base + acs->ring_type;
      amdgpu_fence_submitted(cs->fence, seq_no, user_fence);
   }

   /* Cleanup. */
   if (bo_list)
      amdgpu_bo_list_destroy(bo_list);

cleanup:
   /* If there was an error, signal the fence, because it won't be signalled
    * by the hardware. */
   if (r)
      amdgpu_fence_signalled(cs->fence);

   cs->error_code = r;

   for (i = 0; i < cs->num_real_buffers; i++)
      p_atomic_dec(&cs->real_buffers[i].bo->num_active_ioctls);
   for (i = 0; i < cs->num_slab_buffers; i++)
      p_atomic_dec(&cs->slab_buffers[i].bo->num_active_ioctls);
   for (i = 0; i < cs->num_sparse_buffers; i++)
      p_atomic_dec(&cs->sparse_buffers[i].bo->num_active_ioctls);

   amdgpu_cs_context_cleanup(cs);
}

/* Make sure the previous submission is completed. */
void amdgpu_cs_sync_flush(struct radeon_cmdbuf *rcs)
{
   struct amdgpu_cs *cs = amdgpu_cs(rcs);

   /* Wait for any pending ioctl of this CS to complete. */
   util_queue_fence_wait(&cs->flush_completed);
}

static int amdgpu_cs_flush(struct radeon_cmdbuf *rcs,
                           unsigned flags,
                           struct pipe_fence_handle **fence)
{
   struct amdgpu_cs *cs = amdgpu_cs(rcs);
   struct amdgpu_winsys *ws = cs->ctx->ws;
   int error_code = 0;

   rcs->current.max_dw += amdgpu_cs_epilog_dws(cs->ring_type);

   switch (cs->ring_type) {
   case RING_DMA:
      /* pad DMA ring to 8 DWs */
      if (ws->info.chip_class <= SI) {
         while (rcs->current.cdw & 7)
            radeon_emit(rcs, 0xf0000000); /* NOP packet */
      } else {
         while (rcs->current.cdw & 7)
            radeon_emit(rcs, 0x00000000); /* NOP packet */
      }
      break;
   case RING_GFX:
   case RING_COMPUTE:
      /* pad GFX ring to 8 DWs to meet CP fetch alignment requirements */
      if (ws->info.gfx_ib_pad_with_type2) {
         while (rcs->current.cdw & 7)
            radeon_emit(rcs, 0x80000000); /* type2 nop packet */
      } else {
         while (rcs->current.cdw & 7)
            radeon_emit(rcs, 0xffff1000); /* type3 nop packet */
      }
      if (cs->ring_type == RING_GFX)
         ws->gfx_ib_size_counter += (rcs->prev_dw + rcs->current.cdw) * 4;
      break;
   case RING_UVD:
   case RING_UVD_ENC:
      while (rcs->current.cdw & 15)
         radeon_emit(rcs, 0x80000000); /* type2 nop packet */
      break;
   case RING_VCN_JPEG:
      if (rcs->current.cdw % 2)
         assert(0);
      while (rcs->current.cdw & 15) {
         radeon_emit(rcs, 0x60000000); /* nop packet */
         radeon_emit(rcs, 0x00000000);
      }
      break;
   case RING_VCN_DEC:
      while (rcs->current.cdw & 15)
         radeon_emit(rcs, 0x81ff); /* nop packet */
      break;
   default:
      break;
   }

   if (rcs->current.cdw > rcs->current.max_dw) {
      fprintf(stderr, "amdgpu: command stream overflowed\n");
   }

   /* If the CS is not empty or overflowed.... */
   if (likely(radeon_emitted(&cs->main.base, 0) &&
       cs->main.base.current.cdw <= cs->main.base.current.max_dw &&
       !debug_get_option_noop())) {
      struct amdgpu_cs_context *cur = cs->csc;

      /* Set IB sizes. */
      amdgpu_ib_finalize(ws, &cs->main);

      /* Create a fence. */
      amdgpu_fence_reference(&cur->fence, NULL);
      if (cs->next_fence) {
         /* just move the reference */
         cur->fence = cs->next_fence;
         cs->next_fence = NULL;
      } else {
         cur->fence = amdgpu_fence_create(cs->ctx,
                                          cur->ib[IB_MAIN].ip_type,
                                          cur->ib[IB_MAIN].ip_instance,
                                          cur->ib[IB_MAIN].ring);
      }
      if (fence)
         amdgpu_fence_reference(fence, cur->fence);

      amdgpu_cs_sync_flush(rcs);

      /* Prepare buffers.
       *
       * This fence must be held until the submission is queued to ensure
       * that the order of fence dependency updates matches the order of
       * submissions.
       */
      simple_mtx_lock(&ws->bo_fence_lock);
      amdgpu_add_fence_dependencies_bo_lists(cs);

      /* Swap command streams. "cst" is going to be submitted. */
      cs->csc = cs->cst;
      cs->cst = cur;

      /* Submit. */
      util_queue_add_job(&ws->cs_queue, cs, &cs->flush_completed,
                         amdgpu_cs_submit_ib, NULL);
      /* The submission has been queued, unlock the fence now. */
      simple_mtx_unlock(&ws->bo_fence_lock);

      if (!(flags & PIPE_FLUSH_ASYNC)) {
         amdgpu_cs_sync_flush(rcs);
         error_code = cur->error_code;
      }
   } else {
      amdgpu_cs_context_cleanup(cs->csc);
   }

   amdgpu_get_new_ib(&ws->base, cs, IB_MAIN);

   cs->main.base.used_gart = 0;
   cs->main.base.used_vram = 0;

   if (cs->ring_type == RING_GFX)
      ws->num_gfx_IBs++;
   else if (cs->ring_type == RING_DMA)
      ws->num_sdma_IBs++;

   return error_code;
}

static void amdgpu_cs_destroy(struct radeon_cmdbuf *rcs)
{
   struct amdgpu_cs *cs = amdgpu_cs(rcs);

   amdgpu_cs_sync_flush(rcs);
   util_queue_fence_destroy(&cs->flush_completed);
   p_atomic_dec(&cs->ctx->ws->num_cs);
   pb_reference(&cs->main.big_ib_buffer, NULL);
   FREE(cs->main.base.prev);
   amdgpu_destroy_cs_context(&cs->csc1);
   amdgpu_destroy_cs_context(&cs->csc2);
   amdgpu_fence_reference(&cs->next_fence, NULL);
   FREE(cs);
}

static bool amdgpu_bo_is_referenced(struct radeon_cmdbuf *rcs,
                                    struct pb_buffer *_buf,
                                    enum radeon_bo_usage usage)
{
   struct amdgpu_cs *cs = amdgpu_cs(rcs);
   struct amdgpu_winsys_bo *bo = (struct amdgpu_winsys_bo*)_buf;

   return amdgpu_bo_is_referenced_by_cs_with_usage(cs, bo, usage);
}

void amdgpu_cs_init_functions(struct amdgpu_winsys *ws)
{
   ws->base.ctx_create = amdgpu_ctx_create;
   ws->base.ctx_destroy = amdgpu_ctx_destroy;
   ws->base.ctx_query_reset_status = amdgpu_ctx_query_reset_status;
   ws->base.cs_create = amdgpu_cs_create;
   ws->base.cs_destroy = amdgpu_cs_destroy;
   ws->base.cs_add_buffer = amdgpu_cs_add_buffer;
   ws->base.cs_validate = amdgpu_cs_validate;
   ws->base.cs_check_space = amdgpu_cs_check_space;
   ws->base.cs_get_buffer_list = amdgpu_cs_get_buffer_list;
   ws->base.cs_flush = amdgpu_cs_flush;
   ws->base.cs_get_next_fence = amdgpu_cs_get_next_fence;
   ws->base.cs_is_buffer_referenced = amdgpu_bo_is_referenced;
   ws->base.cs_sync_flush = amdgpu_cs_sync_flush;
   ws->base.cs_add_fence_dependency = amdgpu_cs_add_fence_dependency;
   ws->base.cs_add_syncobj_signal = amdgpu_cs_add_syncobj_signal;
   ws->base.fence_wait = amdgpu_fence_wait_rel_timeout;
   ws->base.fence_reference = amdgpu_fence_reference;
   ws->base.fence_import_syncobj = amdgpu_fence_import_syncobj;
   ws->base.fence_import_sync_file = amdgpu_fence_import_sync_file;
   ws->base.fence_export_sync_file = amdgpu_fence_export_sync_file;
   ws->base.export_signalled_sync_file = amdgpu_export_signalled_sync_file;
}
