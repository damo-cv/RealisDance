/*
 * Copyright © 2015 Intel Corporation
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
 */

/**
 * This file implements VkQueue, VkFence, and VkSemaphore
 */

#include <fcntl.h>
#include <unistd.h>
#include <sys/eventfd.h>

#include "anv_private.h"
#include "vk_util.h"

#include "genxml/gen7_pack.h"

VkResult
anv_device_execbuf(struct anv_device *device,
                   struct drm_i915_gem_execbuffer2 *execbuf,
                   struct anv_bo **execbuf_bos)
{
   int ret = device->no_hw ? 0 : anv_gem_execbuffer(device, execbuf);
   if (ret != 0) {
      /* We don't know the real error. */
      return anv_device_set_lost(device, "execbuf2 failed: %m");
   }

   struct drm_i915_gem_exec_object2 *objects =
      (void *)(uintptr_t)execbuf->buffers_ptr;
   for (uint32_t k = 0; k < execbuf->buffer_count; k++) {
      if (execbuf_bos[k]->flags & EXEC_OBJECT_PINNED)
         assert(execbuf_bos[k]->offset == objects[k].offset);
      execbuf_bos[k]->offset = objects[k].offset;
   }

   return VK_SUCCESS;
}

VkResult
anv_device_submit_simple_batch(struct anv_device *device,
                               struct anv_batch *batch)
{
   struct drm_i915_gem_execbuffer2 execbuf;
   struct drm_i915_gem_exec_object2 exec2_objects[1];
   struct anv_bo bo, *exec_bos[1];
   VkResult result = VK_SUCCESS;
   uint32_t size;

   /* Kernel driver requires 8 byte aligned batch length */
   size = align_u32(batch->next - batch->start, 8);
   result = anv_bo_pool_alloc(&device->batch_bo_pool, &bo, size);
   if (result != VK_SUCCESS)
      return result;

   memcpy(bo.map, batch->start, size);
   if (!device->info.has_llc)
      gen_flush_range(bo.map, size);

   exec_bos[0] = &bo;
   exec2_objects[0].handle = bo.gem_handle;
   exec2_objects[0].relocation_count = 0;
   exec2_objects[0].relocs_ptr = 0;
   exec2_objects[0].alignment = 0;
   exec2_objects[0].offset = bo.offset;
   exec2_objects[0].flags = bo.flags;
   exec2_objects[0].rsvd1 = 0;
   exec2_objects[0].rsvd2 = 0;

   execbuf.buffers_ptr = (uintptr_t) exec2_objects;
   execbuf.buffer_count = 1;
   execbuf.batch_start_offset = 0;
   execbuf.batch_len = size;
   execbuf.cliprects_ptr = 0;
   execbuf.num_cliprects = 0;
   execbuf.DR1 = 0;
   execbuf.DR4 = 0;

   execbuf.flags =
      I915_EXEC_HANDLE_LUT | I915_EXEC_NO_RELOC | I915_EXEC_RENDER;
   execbuf.rsvd1 = device->context_id;
   execbuf.rsvd2 = 0;

   result = anv_device_execbuf(device, &execbuf, exec_bos);
   if (result != VK_SUCCESS)
      goto fail;

   result = anv_device_wait(device, &bo, INT64_MAX);

 fail:
   anv_bo_pool_free(&device->batch_bo_pool, &bo);

   return result;
}

VkResult anv_QueueSubmit(
    VkQueue                                     _queue,
    uint32_t                                    submitCount,
    const VkSubmitInfo*                         pSubmits,
    VkFence                                     fence)
{
   ANV_FROM_HANDLE(anv_queue, queue, _queue);
   struct anv_device *device = queue->device;

   /* Query for device status prior to submitting.  Technically, we don't need
    * to do this.  However, if we have a client that's submitting piles of
    * garbage, we would rather break as early as possible to keep the GPU
    * hanging contained.  If we don't check here, we'll either be waiting for
    * the kernel to kick us or we'll have to wait until the client waits on a
    * fence before we actually know whether or not we've hung.
    */
   VkResult result = anv_device_query_status(device);
   if (result != VK_SUCCESS)
      return result;

   /* We lock around QueueSubmit for three main reasons:
    *
    *  1) When a block pool is resized, we create a new gem handle with a
    *     different size and, in the case of surface states, possibly a
    *     different center offset but we re-use the same anv_bo struct when
    *     we do so.  If this happens in the middle of setting up an execbuf,
    *     we could end up with our list of BOs out of sync with our list of
    *     gem handles.
    *
    *  2) The algorithm we use for building the list of unique buffers isn't
    *     thread-safe.  While the client is supposed to syncronize around
    *     QueueSubmit, this would be extremely difficult to debug if it ever
    *     came up in the wild due to a broken app.  It's better to play it
    *     safe and just lock around QueueSubmit.
    *
    *  3)  The anv_cmd_buffer_execbuf function may perform relocations in
    *      userspace.  Due to the fact that the surface state buffer is shared
    *      between batches, we can't afford to have that happen from multiple
    *      threads at the same time.  Even though the user is supposed to
    *      ensure this doesn't happen, we play it safe as in (2) above.
    *
    * Since the only other things that ever take the device lock such as block
    * pool resize only rarely happen, this will almost never be contended so
    * taking a lock isn't really an expensive operation in this case.
    */
   pthread_mutex_lock(&device->mutex);

   if (fence && submitCount == 0) {
      /* If we don't have any command buffers, we need to submit a dummy
       * batch to give GEM something to wait on.  We could, potentially,
       * come up with something more efficient but this shouldn't be a
       * common case.
       */
      result = anv_cmd_buffer_execbuf(device, NULL, NULL, 0, NULL, 0, fence);
      goto out;
   }

   for (uint32_t i = 0; i < submitCount; i++) {
      /* Fence for this submit.  NULL for all but the last one */
      VkFence submit_fence = (i == submitCount - 1) ? fence : VK_NULL_HANDLE;

      if (pSubmits[i].commandBufferCount == 0) {
         /* If we don't have any command buffers, we need to submit a dummy
          * batch to give GEM something to wait on.  We could, potentially,
          * come up with something more efficient but this shouldn't be a
          * common case.
          */
         result = anv_cmd_buffer_execbuf(device, NULL,
                                         pSubmits[i].pWaitSemaphores,
                                         pSubmits[i].waitSemaphoreCount,
                                         pSubmits[i].pSignalSemaphores,
                                         pSubmits[i].signalSemaphoreCount,
                                         submit_fence);
         if (result != VK_SUCCESS)
            goto out;

         continue;
      }

      for (uint32_t j = 0; j < pSubmits[i].commandBufferCount; j++) {
         ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer,
                         pSubmits[i].pCommandBuffers[j]);
         assert(cmd_buffer->level == VK_COMMAND_BUFFER_LEVEL_PRIMARY);
         assert(!anv_batch_has_error(&cmd_buffer->batch));

         /* Fence for this execbuf.  NULL for all but the last one */
         VkFence execbuf_fence =
            (j == pSubmits[i].commandBufferCount - 1) ?
            submit_fence : VK_NULL_HANDLE;

         const VkSemaphore *in_semaphores = NULL, *out_semaphores = NULL;
         uint32_t num_in_semaphores = 0, num_out_semaphores = 0;
         if (j == 0) {
            /* Only the first batch gets the in semaphores */
            in_semaphores = pSubmits[i].pWaitSemaphores;
            num_in_semaphores = pSubmits[i].waitSemaphoreCount;
         }

         if (j == pSubmits[i].commandBufferCount - 1) {
            /* Only the last batch gets the out semaphores */
            out_semaphores = pSubmits[i].pSignalSemaphores;
            num_out_semaphores = pSubmits[i].signalSemaphoreCount;
         }

         result = anv_cmd_buffer_execbuf(device, cmd_buffer,
                                         in_semaphores, num_in_semaphores,
                                         out_semaphores, num_out_semaphores,
                                         execbuf_fence);
         if (result != VK_SUCCESS)
            goto out;
      }
   }

   pthread_cond_broadcast(&device->queue_submit);

out:
   if (result != VK_SUCCESS) {
      /* In the case that something has gone wrong we may end up with an
       * inconsistent state from which it may not be trivial to recover.
       * For example, we might have computed address relocations and
       * any future attempt to re-submit this job will need to know about
       * this and avoid computing relocation addresses again.
       *
       * To avoid this sort of issues, we assume that if something was
       * wrong during submission we must already be in a really bad situation
       * anyway (such us being out of memory) and return
       * VK_ERROR_DEVICE_LOST to ensure that clients do not attempt to
       * submit the same job again to this device.
       */
      result = anv_device_set_lost(device, "vkQueueSubmit() failed");
   }

   pthread_mutex_unlock(&device->mutex);

   return result;
}

VkResult anv_QueueWaitIdle(
    VkQueue                                     _queue)
{
   ANV_FROM_HANDLE(anv_queue, queue, _queue);

   return anv_DeviceWaitIdle(anv_device_to_handle(queue->device));
}

VkResult anv_CreateFence(
    VkDevice                                    _device,
    const VkFenceCreateInfo*                    pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_fence *fence;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);

   fence = vk_zalloc2(&device->alloc, pAllocator, sizeof(*fence), 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (fence == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   if (device->instance->physicalDevice.has_syncobj_wait) {
      fence->permanent.type = ANV_FENCE_TYPE_SYNCOBJ;

      uint32_t create_flags = 0;
      if (pCreateInfo->flags & VK_FENCE_CREATE_SIGNALED_BIT)
         create_flags |= DRM_SYNCOBJ_CREATE_SIGNALED;

      fence->permanent.syncobj = anv_gem_syncobj_create(device, create_flags);
      if (!fence->permanent.syncobj)
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
   } else {
      fence->permanent.type = ANV_FENCE_TYPE_BO;

      VkResult result = anv_bo_pool_alloc(&device->batch_bo_pool,
                                          &fence->permanent.bo.bo, 4096);
      if (result != VK_SUCCESS)
         return result;

      if (pCreateInfo->flags & VK_FENCE_CREATE_SIGNALED_BIT) {
         fence->permanent.bo.state = ANV_BO_FENCE_STATE_SIGNALED;
      } else {
         fence->permanent.bo.state = ANV_BO_FENCE_STATE_RESET;
      }
   }

   *pFence = anv_fence_to_handle(fence);

   return VK_SUCCESS;
}

static void
anv_fence_impl_cleanup(struct anv_device *device,
                       struct anv_fence_impl *impl)
{
   switch (impl->type) {
   case ANV_FENCE_TYPE_NONE:
      /* Dummy.  Nothing to do */
      break;

   case ANV_FENCE_TYPE_BO:
      anv_bo_pool_free(&device->batch_bo_pool, &impl->bo.bo);
      break;

   case ANV_FENCE_TYPE_SYNCOBJ:
      anv_gem_syncobj_destroy(device, impl->syncobj);
      break;

   case ANV_FENCE_TYPE_WSI:
      impl->fence_wsi->destroy(impl->fence_wsi);
      break;

   default:
      unreachable("Invalid fence type");
   }

   impl->type = ANV_FENCE_TYPE_NONE;
}

void anv_DestroyFence(
    VkDevice                                    _device,
    VkFence                                     _fence,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_fence, fence, _fence);

   if (!fence)
      return;

   anv_fence_impl_cleanup(device, &fence->temporary);
   anv_fence_impl_cleanup(device, &fence->permanent);

   vk_free2(&device->alloc, pAllocator, fence);
}

VkResult anv_ResetFences(
    VkDevice                                    _device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   for (uint32_t i = 0; i < fenceCount; i++) {
      ANV_FROM_HANDLE(anv_fence, fence, pFences[i]);

      /* From the Vulkan 1.0.53 spec:
       *
       *    "If any member of pFences currently has its payload imported with
       *    temporary permanence, that fence’s prior permanent payload is
       *    first restored. The remaining operations described therefore
       *    operate on the restored payload.
       */
      if (fence->temporary.type != ANV_FENCE_TYPE_NONE)
         anv_fence_impl_cleanup(device, &fence->temporary);

      struct anv_fence_impl *impl = &fence->permanent;

      switch (impl->type) {
      case ANV_FENCE_TYPE_BO:
         impl->bo.state = ANV_BO_FENCE_STATE_RESET;
         break;

      case ANV_FENCE_TYPE_SYNCOBJ:
         anv_gem_syncobj_reset(device, impl->syncobj);
         break;

      default:
         unreachable("Invalid fence type");
      }
   }

   return VK_SUCCESS;
}

VkResult anv_GetFenceStatus(
    VkDevice                                    _device,
    VkFence                                     _fence)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_fence, fence, _fence);

   if (anv_device_is_lost(device))
      return VK_ERROR_DEVICE_LOST;

   struct anv_fence_impl *impl =
      fence->temporary.type != ANV_FENCE_TYPE_NONE ?
      &fence->temporary : &fence->permanent;

   switch (impl->type) {
   case ANV_FENCE_TYPE_BO:
      /* BO fences don't support import/export */
      assert(fence->temporary.type == ANV_FENCE_TYPE_NONE);
      switch (impl->bo.state) {
      case ANV_BO_FENCE_STATE_RESET:
         /* If it hasn't even been sent off to the GPU yet, it's not ready */
         return VK_NOT_READY;

      case ANV_BO_FENCE_STATE_SIGNALED:
         /* It's been signaled, return success */
         return VK_SUCCESS;

      case ANV_BO_FENCE_STATE_SUBMITTED: {
         VkResult result = anv_device_bo_busy(device, &impl->bo.bo);
         if (result == VK_SUCCESS) {
            impl->bo.state = ANV_BO_FENCE_STATE_SIGNALED;
            return VK_SUCCESS;
         } else {
            return result;
         }
      }
      default:
         unreachable("Invalid fence status");
      }

   case ANV_FENCE_TYPE_SYNCOBJ: {
      int ret = anv_gem_syncobj_wait(device, &impl->syncobj, 1, 0, true);
      if (ret == -1) {
         if (errno == ETIME) {
            return VK_NOT_READY;
         } else {
            /* We don't know the real error. */
            return anv_device_set_lost(device, "drm_syncobj_wait failed: %m");
         }
      } else {
         return VK_SUCCESS;
      }
   }

   default:
      unreachable("Invalid fence type");
   }
}

#define NSEC_PER_SEC 1000000000
#define INT_TYPE_MAX(type) ((1ull << (sizeof(type) * 8 - 1)) - 1)

static uint64_t
gettime_ns(void)
{
   struct timespec current;
   clock_gettime(CLOCK_MONOTONIC, &current);
   return (uint64_t)current.tv_sec * NSEC_PER_SEC + current.tv_nsec;
}

static uint64_t anv_get_absolute_timeout(uint64_t timeout)
{
   if (timeout == 0)
      return 0;
   uint64_t current_time = gettime_ns();
   uint64_t max_timeout = (uint64_t) INT64_MAX - current_time;

   timeout = MIN2(max_timeout, timeout);

   return (current_time + timeout);
}

static int64_t anv_get_relative_timeout(uint64_t abs_timeout)
{
   uint64_t now = gettime_ns();

   /* We don't want negative timeouts.
    *
    * DRM_IOCTL_I915_GEM_WAIT uses a signed 64 bit timeout and is
    * supposed to block indefinitely timeouts < 0.  Unfortunately,
    * this was broken for a couple of kernel releases.  Since there's
    * no way to know whether or not the kernel we're using is one of
    * the broken ones, the best we can do is to clamp the timeout to
    * INT64_MAX.  This limits the maximum timeout from 584 years to
    * 292 years - likely not a big deal.
    */
   if (abs_timeout < now)
      return 0;

   uint64_t rel_timeout = abs_timeout - now;
   if (rel_timeout > (uint64_t) INT64_MAX)
      rel_timeout = INT64_MAX;

   return rel_timeout;
}

static VkResult
anv_wait_for_syncobj_fences(struct anv_device *device,
                            uint32_t fenceCount,
                            const VkFence *pFences,
                            bool waitAll,
                            uint64_t abs_timeout_ns)
{
   uint32_t *syncobjs = vk_zalloc(&device->alloc,
                                  sizeof(*syncobjs) * fenceCount, 8,
                                  VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
   if (!syncobjs)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   for (uint32_t i = 0; i < fenceCount; i++) {
      ANV_FROM_HANDLE(anv_fence, fence, pFences[i]);
      assert(fence->permanent.type == ANV_FENCE_TYPE_SYNCOBJ);

      struct anv_fence_impl *impl =
         fence->temporary.type != ANV_FENCE_TYPE_NONE ?
         &fence->temporary : &fence->permanent;

      assert(impl->type == ANV_FENCE_TYPE_SYNCOBJ);
      syncobjs[i] = impl->syncobj;
   }

   /* The gem_syncobj_wait ioctl may return early due to an inherent
    * limitation in the way it computes timeouts.  Loop until we've actually
    * passed the timeout.
    */
   int ret;
   do {
      ret = anv_gem_syncobj_wait(device, syncobjs, fenceCount,
                                 abs_timeout_ns, waitAll);
   } while (ret == -1 && errno == ETIME && gettime_ns() < abs_timeout_ns);

   vk_free(&device->alloc, syncobjs);

   if (ret == -1) {
      if (errno == ETIME) {
         return VK_TIMEOUT;
      } else {
         /* We don't know the real error. */
         return anv_device_set_lost(device, "drm_syncobj_wait failed: %m");
      }
   } else {
      return VK_SUCCESS;
   }
}

static VkResult
anv_wait_for_bo_fences(struct anv_device *device,
                       uint32_t fenceCount,
                       const VkFence *pFences,
                       bool waitAll,
                       uint64_t abs_timeout_ns)
{
   VkResult result = VK_SUCCESS;
   uint32_t pending_fences = fenceCount;
   while (pending_fences) {
      pending_fences = 0;
      bool signaled_fences = false;
      for (uint32_t i = 0; i < fenceCount; i++) {
         ANV_FROM_HANDLE(anv_fence, fence, pFences[i]);

         /* This function assumes that all fences are BO fences and that they
          * have no temporary state.  Since BO fences will never be exported,
          * this should be a safe assumption.
          */
         assert(fence->permanent.type == ANV_FENCE_TYPE_BO);
         assert(fence->temporary.type == ANV_FENCE_TYPE_NONE);
         struct anv_fence_impl *impl = &fence->permanent;

         switch (impl->bo.state) {
         case ANV_BO_FENCE_STATE_RESET:
            /* This fence hasn't been submitted yet, we'll catch it the next
             * time around.  Yes, this may mean we dead-loop but, short of
             * lots of locking and a condition variable, there's not much that
             * we can do about that.
             */
            pending_fences++;
            continue;

         case ANV_BO_FENCE_STATE_SIGNALED:
            /* This fence is not pending.  If waitAll isn't set, we can return
             * early.  Otherwise, we have to keep going.
             */
            if (!waitAll) {
               result = VK_SUCCESS;
               goto done;
            }
            continue;

         case ANV_BO_FENCE_STATE_SUBMITTED:
            /* These are the fences we really care about.  Go ahead and wait
             * on it until we hit a timeout.
             */
            result = anv_device_wait(device, &impl->bo.bo,
                                     anv_get_relative_timeout(abs_timeout_ns));
            switch (result) {
            case VK_SUCCESS:
               impl->bo.state = ANV_BO_FENCE_STATE_SIGNALED;
               signaled_fences = true;
               if (!waitAll)
                  goto done;
               break;

            case VK_TIMEOUT:
               goto done;

            default:
               return result;
            }
         }
      }

      if (pending_fences && !signaled_fences) {
         /* If we've hit this then someone decided to vkWaitForFences before
          * they've actually submitted any of them to a queue.  This is a
          * fairly pessimal case, so it's ok to lock here and use a standard
          * pthreads condition variable.
          */
         pthread_mutex_lock(&device->mutex);

         /* It's possible that some of the fences have changed state since the
          * last time we checked.  Now that we have the lock, check for
          * pending fences again and don't wait if it's changed.
          */
         uint32_t now_pending_fences = 0;
         for (uint32_t i = 0; i < fenceCount; i++) {
            ANV_FROM_HANDLE(anv_fence, fence, pFences[i]);
            if (fence->permanent.bo.state == ANV_BO_FENCE_STATE_RESET)
               now_pending_fences++;
         }
         assert(now_pending_fences <= pending_fences);

         if (now_pending_fences == pending_fences) {
            struct timespec abstime = {
               .tv_sec = abs_timeout_ns / NSEC_PER_SEC,
               .tv_nsec = abs_timeout_ns % NSEC_PER_SEC,
            };

            MAYBE_UNUSED int ret;
            ret = pthread_cond_timedwait(&device->queue_submit,
                                         &device->mutex, &abstime);
            assert(ret != EINVAL);
            if (gettime_ns() >= abs_timeout_ns) {
               pthread_mutex_unlock(&device->mutex);
               result = VK_TIMEOUT;
               goto done;
            }
         }

         pthread_mutex_unlock(&device->mutex);
      }
   }

done:
   if (anv_device_is_lost(device))
      return VK_ERROR_DEVICE_LOST;

   return result;
}

static VkResult
anv_wait_for_wsi_fence(struct anv_device *device,
                       const VkFence _fence,
                       uint64_t abs_timeout)
{
   ANV_FROM_HANDLE(anv_fence, fence, _fence);
   struct anv_fence_impl *impl = &fence->permanent;

   return impl->fence_wsi->wait(impl->fence_wsi, abs_timeout);
}

static VkResult
anv_wait_for_fences(struct anv_device *device,
                    uint32_t fenceCount,
                    const VkFence *pFences,
                    bool waitAll,
                    uint64_t abs_timeout)
{
   VkResult result = VK_SUCCESS;

   if (fenceCount <= 1 || waitAll) {
      for (uint32_t i = 0; i < fenceCount; i++) {
         ANV_FROM_HANDLE(anv_fence, fence, pFences[i]);
         switch (fence->permanent.type) {
         case ANV_FENCE_TYPE_BO:
            result = anv_wait_for_bo_fences(device, 1, &pFences[i],
                                            true, abs_timeout);
            break;
         case ANV_FENCE_TYPE_SYNCOBJ:
            result = anv_wait_for_syncobj_fences(device, 1, &pFences[i],
                                                 true, abs_timeout);
            break;
         case ANV_FENCE_TYPE_WSI:
            result = anv_wait_for_wsi_fence(device, pFences[i], abs_timeout);
            break;
         case ANV_FENCE_TYPE_NONE:
            result = VK_SUCCESS;
            break;
         }
         if (result != VK_SUCCESS)
            return result;
      }
   } else {
      do {
         for (uint32_t i = 0; i < fenceCount; i++) {
            if (anv_wait_for_fences(device, 1, &pFences[i], true, 0) == VK_SUCCESS)
               return VK_SUCCESS;
         }
      } while (gettime_ns() < abs_timeout);
      result = VK_TIMEOUT;
   }
   return result;
}

static bool anv_all_fences_syncobj(uint32_t fenceCount, const VkFence *pFences)
{
   for (uint32_t i = 0; i < fenceCount; ++i) {
      ANV_FROM_HANDLE(anv_fence, fence, pFences[i]);
      if (fence->permanent.type != ANV_FENCE_TYPE_SYNCOBJ)
         return false;
   }
   return true;
}

static bool anv_all_fences_bo(uint32_t fenceCount, const VkFence *pFences)
{
   for (uint32_t i = 0; i < fenceCount; ++i) {
      ANV_FROM_HANDLE(anv_fence, fence, pFences[i]);
      if (fence->permanent.type != ANV_FENCE_TYPE_BO)
         return false;
   }
   return true;
}

VkResult anv_WaitForFences(
    VkDevice                                    _device,
    uint32_t                                    fenceCount,
    const VkFence*                              pFences,
    VkBool32                                    waitAll,
    uint64_t                                    timeout)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   if (anv_device_is_lost(device))
      return VK_ERROR_DEVICE_LOST;

   uint64_t abs_timeout = anv_get_absolute_timeout(timeout);
   if (anv_all_fences_syncobj(fenceCount, pFences)) {
      return anv_wait_for_syncobj_fences(device, fenceCount, pFences,
                                         waitAll, abs_timeout);
   } else if (anv_all_fences_bo(fenceCount, pFences)) {
      return anv_wait_for_bo_fences(device, fenceCount, pFences,
                                    waitAll, abs_timeout);
   } else {
      return anv_wait_for_fences(device, fenceCount, pFences,
                                 waitAll, abs_timeout);
   }
}

void anv_GetPhysicalDeviceExternalFenceProperties(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceExternalFenceInfoKHR* pExternalFenceInfo,
    VkExternalFencePropertiesKHR*               pExternalFenceProperties)
{
   ANV_FROM_HANDLE(anv_physical_device, device, physicalDevice);

   switch (pExternalFenceInfo->handleType) {
   case VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT:
   case VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT:
      if (device->has_syncobj_wait) {
         pExternalFenceProperties->exportFromImportedHandleTypes =
            VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT |
            VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
         pExternalFenceProperties->compatibleHandleTypes =
            VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT |
            VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
         pExternalFenceProperties->externalFenceFeatures =
            VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT |
            VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT;
         return;
      }
      break;

   default:
      break;
   }

   pExternalFenceProperties->exportFromImportedHandleTypes = 0;
   pExternalFenceProperties->compatibleHandleTypes = 0;
   pExternalFenceProperties->externalFenceFeatures = 0;
}

VkResult anv_ImportFenceFdKHR(
    VkDevice                                    _device,
    const VkImportFenceFdInfoKHR*               pImportFenceFdInfo)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_fence, fence, pImportFenceFdInfo->fence);
   int fd = pImportFenceFdInfo->fd;

   assert(pImportFenceFdInfo->sType ==
          VK_STRUCTURE_TYPE_IMPORT_FENCE_FD_INFO_KHR);

   struct anv_fence_impl new_impl = {
      .type = ANV_FENCE_TYPE_NONE,
   };

   switch (pImportFenceFdInfo->handleType) {
   case VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT:
      new_impl.type = ANV_FENCE_TYPE_SYNCOBJ;

      new_impl.syncobj = anv_gem_syncobj_fd_to_handle(device, fd);
      if (!new_impl.syncobj)
         return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE);

      break;

   case VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT:
      /* Sync files are a bit tricky.  Because we want to continue using the
       * syncobj implementation of WaitForFences, we don't use the sync file
       * directly but instead import it into a syncobj.
       */
      new_impl.type = ANV_FENCE_TYPE_SYNCOBJ;

      new_impl.syncobj = anv_gem_syncobj_create(device, 0);
      if (!new_impl.syncobj)
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

      if (anv_gem_syncobj_import_sync_file(device, new_impl.syncobj, fd)) {
         anv_gem_syncobj_destroy(device, new_impl.syncobj);
         return vk_errorf(device->instance, NULL,
                          VK_ERROR_INVALID_EXTERNAL_HANDLE,
                          "syncobj sync file import failed: %m");
      }
      break;

   default:
      return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }

   /* From the Vulkan 1.0.53 spec:
    *
    *    "Importing a fence payload from a file descriptor transfers
    *    ownership of the file descriptor from the application to the
    *    Vulkan implementation. The application must not perform any
    *    operations on the file descriptor after a successful import."
    *
    * If the import fails, we leave the file descriptor open.
    */
   close(fd);

   if (pImportFenceFdInfo->flags & VK_FENCE_IMPORT_TEMPORARY_BIT) {
      anv_fence_impl_cleanup(device, &fence->temporary);
      fence->temporary = new_impl;
   } else {
      anv_fence_impl_cleanup(device, &fence->permanent);
      fence->permanent = new_impl;
   }

   return VK_SUCCESS;
}

VkResult anv_GetFenceFdKHR(
    VkDevice                                    _device,
    const VkFenceGetFdInfoKHR*                  pGetFdInfo,
    int*                                        pFd)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_fence, fence, pGetFdInfo->fence);

   assert(pGetFdInfo->sType == VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR);

   struct anv_fence_impl *impl =
      fence->temporary.type != ANV_FENCE_TYPE_NONE ?
      &fence->temporary : &fence->permanent;

   assert(impl->type == ANV_FENCE_TYPE_SYNCOBJ);
   switch (pGetFdInfo->handleType) {
   case VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT: {
      int fd = anv_gem_syncobj_handle_to_fd(device, impl->syncobj);
      if (fd < 0)
         return vk_error(VK_ERROR_TOO_MANY_OBJECTS);

      *pFd = fd;
      break;
   }

   case VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT: {
      int fd = anv_gem_syncobj_export_sync_file(device, impl->syncobj);
      if (fd < 0)
         return vk_error(VK_ERROR_TOO_MANY_OBJECTS);

      *pFd = fd;
      break;
   }

   default:
      unreachable("Invalid fence export handle type");
   }

   /* From the Vulkan 1.0.53 spec:
    *
    *    "Export operations have the same transference as the specified handle
    *    type’s import operations. [...] If the fence was using a
    *    temporarily imported payload, the fence’s prior permanent payload
    *    will be restored.
    */
   if (impl == &fence->temporary)
      anv_fence_impl_cleanup(device, impl);

   return VK_SUCCESS;
}

// Queue semaphore functions

VkResult anv_CreateSemaphore(
    VkDevice                                    _device,
    const VkSemaphoreCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSemaphore*                                pSemaphore)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_semaphore *semaphore;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);

   semaphore = vk_alloc2(&device->alloc, pAllocator, sizeof(*semaphore), 8,
                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (semaphore == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   const VkExportSemaphoreCreateInfoKHR *export =
      vk_find_struct_const(pCreateInfo->pNext, EXPORT_SEMAPHORE_CREATE_INFO);
    VkExternalSemaphoreHandleTypeFlagsKHR handleTypes =
      export ? export->handleTypes : 0;

   if (handleTypes == 0) {
      /* The DRM execbuffer ioctl always execute in-oder so long as you stay
       * on the same ring.  Since we don't expose the blit engine as a DMA
       * queue, a dummy no-op semaphore is a perfectly valid implementation.
       */
      semaphore->permanent.type = ANV_SEMAPHORE_TYPE_DUMMY;
   } else if (handleTypes & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
      assert(handleTypes == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT);
      if (device->instance->physicalDevice.has_syncobj) {
         semaphore->permanent.type = ANV_SEMAPHORE_TYPE_DRM_SYNCOBJ;
         semaphore->permanent.syncobj = anv_gem_syncobj_create(device, 0);
         if (!semaphore->permanent.syncobj) {
            vk_free2(&device->alloc, pAllocator, semaphore);
            return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
         }
      } else {
         semaphore->permanent.type = ANV_SEMAPHORE_TYPE_BO;
         VkResult result = anv_bo_cache_alloc(device, &device->bo_cache,
                                              4096, ANV_BO_EXTERNAL,
                                              &semaphore->permanent.bo);
         if (result != VK_SUCCESS) {
            vk_free2(&device->alloc, pAllocator, semaphore);
            return result;
         }

         /* If we're going to use this as a fence, we need to *not* have the
          * EXEC_OBJECT_ASYNC bit set.
          */
         assert(!(semaphore->permanent.bo->flags & EXEC_OBJECT_ASYNC));
      }
   } else if (handleTypes & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT) {
      assert(handleTypes == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT);

      semaphore->permanent.type = ANV_SEMAPHORE_TYPE_SYNC_FILE;
      semaphore->permanent.fd = -1;
   } else {
      assert(!"Unknown handle type");
      vk_free2(&device->alloc, pAllocator, semaphore);
      return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }

   semaphore->temporary.type = ANV_SEMAPHORE_TYPE_NONE;

   *pSemaphore = anv_semaphore_to_handle(semaphore);

   return VK_SUCCESS;
}

static void
anv_semaphore_impl_cleanup(struct anv_device *device,
                           struct anv_semaphore_impl *impl)
{
   switch (impl->type) {
   case ANV_SEMAPHORE_TYPE_NONE:
   case ANV_SEMAPHORE_TYPE_DUMMY:
      /* Dummy.  Nothing to do */
      break;

   case ANV_SEMAPHORE_TYPE_BO:
      anv_bo_cache_release(device, &device->bo_cache, impl->bo);
      break;

   case ANV_SEMAPHORE_TYPE_SYNC_FILE:
      close(impl->fd);
      break;

   case ANV_SEMAPHORE_TYPE_DRM_SYNCOBJ:
      anv_gem_syncobj_destroy(device, impl->syncobj);
      break;

   default:
      unreachable("Invalid semaphore type");
   }

   impl->type = ANV_SEMAPHORE_TYPE_NONE;
}

void
anv_semaphore_reset_temporary(struct anv_device *device,
                              struct anv_semaphore *semaphore)
{
   if (semaphore->temporary.type == ANV_SEMAPHORE_TYPE_NONE)
      return;

   anv_semaphore_impl_cleanup(device, &semaphore->temporary);
}

void anv_DestroySemaphore(
    VkDevice                                    _device,
    VkSemaphore                                 _semaphore,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_semaphore, semaphore, _semaphore);

   if (semaphore == NULL)
      return;

   anv_semaphore_impl_cleanup(device, &semaphore->temporary);
   anv_semaphore_impl_cleanup(device, &semaphore->permanent);

   vk_free2(&device->alloc, pAllocator, semaphore);
}

void anv_GetPhysicalDeviceExternalSemaphoreProperties(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceExternalSemaphoreInfoKHR* pExternalSemaphoreInfo,
    VkExternalSemaphorePropertiesKHR*           pExternalSemaphoreProperties)
{
   ANV_FROM_HANDLE(anv_physical_device, device, physicalDevice);

   switch (pExternalSemaphoreInfo->handleType) {
   case VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT:
      pExternalSemaphoreProperties->exportFromImportedHandleTypes =
         VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
      pExternalSemaphoreProperties->compatibleHandleTypes =
         VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
      pExternalSemaphoreProperties->externalSemaphoreFeatures =
         VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT |
         VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
      return;

   case VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT:
      if (device->has_exec_fence) {
         pExternalSemaphoreProperties->exportFromImportedHandleTypes = 0;
         pExternalSemaphoreProperties->compatibleHandleTypes =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
         pExternalSemaphoreProperties->externalSemaphoreFeatures =
            VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT |
            VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
         return;
      }
      break;

   default:
      break;
   }

   pExternalSemaphoreProperties->exportFromImportedHandleTypes = 0;
   pExternalSemaphoreProperties->compatibleHandleTypes = 0;
   pExternalSemaphoreProperties->externalSemaphoreFeatures = 0;
}

VkResult anv_ImportSemaphoreFdKHR(
    VkDevice                                    _device,
    const VkImportSemaphoreFdInfoKHR*           pImportSemaphoreFdInfo)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_semaphore, semaphore, pImportSemaphoreFdInfo->semaphore);
   int fd = pImportSemaphoreFdInfo->fd;

   struct anv_semaphore_impl new_impl = {
      .type = ANV_SEMAPHORE_TYPE_NONE,
   };

   switch (pImportSemaphoreFdInfo->handleType) {
   case VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT:
      if (device->instance->physicalDevice.has_syncobj) {
         new_impl.type = ANV_SEMAPHORE_TYPE_DRM_SYNCOBJ;

         new_impl.syncobj = anv_gem_syncobj_fd_to_handle(device, fd);
         if (!new_impl.syncobj)
            return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE);
      } else {
         new_impl.type = ANV_SEMAPHORE_TYPE_BO;

         VkResult result = anv_bo_cache_import(device, &device->bo_cache,
                                               fd, ANV_BO_EXTERNAL,
                                               &new_impl.bo);
         if (result != VK_SUCCESS)
            return result;

         if (new_impl.bo->size < 4096) {
            anv_bo_cache_release(device, &device->bo_cache, new_impl.bo);
            return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR);
         }

         /* If we're going to use this as a fence, we need to *not* have the
          * EXEC_OBJECT_ASYNC bit set.
          */
         assert(!(new_impl.bo->flags & EXEC_OBJECT_ASYNC));
      }

      /* From the Vulkan spec:
       *
       *    "Importing semaphore state from a file descriptor transfers
       *    ownership of the file descriptor from the application to the
       *    Vulkan implementation. The application must not perform any
       *    operations on the file descriptor after a successful import."
       *
       * If the import fails, we leave the file descriptor open.
       */
      close(fd);
      break;

   case VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT:
      new_impl = (struct anv_semaphore_impl) {
         .type = ANV_SEMAPHORE_TYPE_SYNC_FILE,
         .fd = fd,
      };
      break;

   default:
      return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }

   if (pImportSemaphoreFdInfo->flags & VK_SEMAPHORE_IMPORT_TEMPORARY_BIT) {
      anv_semaphore_impl_cleanup(device, &semaphore->temporary);
      semaphore->temporary = new_impl;
   } else {
      anv_semaphore_impl_cleanup(device, &semaphore->permanent);
      semaphore->permanent = new_impl;
   }

   return VK_SUCCESS;
}

VkResult anv_GetSemaphoreFdKHR(
    VkDevice                                    _device,
    const VkSemaphoreGetFdInfoKHR*              pGetFdInfo,
    int*                                        pFd)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_semaphore, semaphore, pGetFdInfo->semaphore);
   VkResult result;
   int fd;

   assert(pGetFdInfo->sType == VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR);

   struct anv_semaphore_impl *impl =
      semaphore->temporary.type != ANV_SEMAPHORE_TYPE_NONE ?
      &semaphore->temporary : &semaphore->permanent;

   switch (impl->type) {
   case ANV_SEMAPHORE_TYPE_BO:
      result = anv_bo_cache_export(device, &device->bo_cache, impl->bo, pFd);
      if (result != VK_SUCCESS)
         return result;
      break;

   case ANV_SEMAPHORE_TYPE_SYNC_FILE:
      /* There are two reasons why this could happen:
       *
       *  1) The user is trying to export without submitting something that
       *     signals the semaphore.  If this is the case, it's their bug so
       *     what we return here doesn't matter.
       *
       *  2) The kernel didn't give us a file descriptor.  The most likely
       *     reason for this is running out of file descriptors.
       */
      if (impl->fd < 0)
         return vk_error(VK_ERROR_TOO_MANY_OBJECTS);

      *pFd = impl->fd;

      /* From the Vulkan 1.0.53 spec:
       *
       *    "...exporting a semaphore payload to a handle with copy
       *    transference has the same side effects on the source
       *    semaphore’s payload as executing a semaphore wait operation."
       *
       * In other words, it may still be a SYNC_FD semaphore, but it's now
       * considered to have been waited on and no longer has a sync file
       * attached.
       */
      impl->fd = -1;
      return VK_SUCCESS;

   case ANV_SEMAPHORE_TYPE_DRM_SYNCOBJ:
      fd = anv_gem_syncobj_handle_to_fd(device, impl->syncobj);
      if (fd < 0)
         return vk_error(VK_ERROR_TOO_MANY_OBJECTS);
      *pFd = fd;
      break;

   default:
      return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }

   /* From the Vulkan 1.0.53 spec:
    *
    *    "Export operations have the same transference as the specified handle
    *    type’s import operations. [...] If the semaphore was using a
    *    temporarily imported payload, the semaphore’s prior permanent payload
    *    will be restored.
    */
   if (impl == &semaphore->temporary)
      anv_semaphore_impl_cleanup(device, impl);

   return VK_SUCCESS;
}
