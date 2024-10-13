/*
 * Copyright © 2017, Google Inc.
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

#include <hardware/gralloc.h>
#include <hardware/hardware.h>
#include <hardware/hwvulkan.h>
#include <vulkan/vk_android_native_buffer.h>
#include <vulkan/vk_icd.h>
#include <sync/sync.h>

#include "anv_private.h"

static int anv_hal_open(const struct hw_module_t* mod, const char* id, struct hw_device_t** dev);
static int anv_hal_close(struct hw_device_t *dev);

static void UNUSED
static_asserts(void)
{
   STATIC_ASSERT(HWVULKAN_DISPATCH_MAGIC == ICD_LOADER_MAGIC);
}

PUBLIC struct hwvulkan_module_t HAL_MODULE_INFO_SYM = {
   .common = {
      .tag = HARDWARE_MODULE_TAG,
      .module_api_version = HWVULKAN_MODULE_API_VERSION_0_1,
      .hal_api_version = HARDWARE_MAKE_API_VERSION(1, 0),
      .id = HWVULKAN_HARDWARE_MODULE_ID,
      .name = "Intel Vulkan HAL",
      .author = "Intel",
      .methods = &(hw_module_methods_t) {
         .open = anv_hal_open,
      },
   },
};

/* If any bits in test_mask are set, then unset them and return true. */
static inline bool
unmask32(uint32_t *inout_mask, uint32_t test_mask)
{
   uint32_t orig_mask = *inout_mask;
   *inout_mask &= ~test_mask;
   return *inout_mask != orig_mask;
}

static int
anv_hal_open(const struct hw_module_t* mod, const char* id,
             struct hw_device_t** dev)
{
   assert(mod == &HAL_MODULE_INFO_SYM.common);
   assert(strcmp(id, HWVULKAN_DEVICE_0) == 0);

   hwvulkan_device_t *hal_dev = malloc(sizeof(*hal_dev));
   if (!hal_dev)
      return -1;

   *hal_dev = (hwvulkan_device_t) {
      .common = {
         .tag = HARDWARE_DEVICE_TAG,
         .version = HWVULKAN_DEVICE_API_VERSION_0_1,
         .module = &HAL_MODULE_INFO_SYM.common,
         .close = anv_hal_close,
      },
     .EnumerateInstanceExtensionProperties = anv_EnumerateInstanceExtensionProperties,
     .CreateInstance = anv_CreateInstance,
     .GetInstanceProcAddr = anv_GetInstanceProcAddr,
   };

   *dev = &hal_dev->common;
   return 0;
}

static int
anv_hal_close(struct hw_device_t *dev)
{
   /* hwvulkan.h claims that hw_device_t::close() is never called. */
   return -1;
}

VkResult
anv_image_from_gralloc(VkDevice device_h,
                       const VkImageCreateInfo *base_info,
                       const VkNativeBufferANDROID *gralloc_info,
                       const VkAllocationCallbacks *alloc,
                       VkImage *out_image_h)

{
   ANV_FROM_HANDLE(anv_device, device, device_h);
   VkImage image_h = VK_NULL_HANDLE;
   struct anv_image *image = NULL;
   struct anv_bo *bo = NULL;
   VkResult result;

   struct anv_image_create_info anv_info = {
      .vk_info = base_info,
      .isl_extra_usage_flags = ISL_SURF_USAGE_DISABLE_AUX_BIT,
   };

   if (gralloc_info->handle->numFds != 1) {
      return vk_errorf(device->instance, device,
                       VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR,
                       "VkNativeBufferANDROID::handle::numFds is %d, "
                       "expected 1", gralloc_info->handle->numFds);
   }

   /* Do not close the gralloc handle's dma_buf. The lifetime of the dma_buf
    * must exceed that of the gralloc handle, and we do not own the gralloc
    * handle.
    */
   int dma_buf = gralloc_info->handle->data[0];

   uint64_t bo_flags = ANV_BO_EXTERNAL;
   if (device->instance->physicalDevice.supports_48bit_addresses)
      bo_flags |= EXEC_OBJECT_SUPPORTS_48B_ADDRESS;
   if (device->instance->physicalDevice.use_softpin)
      bo_flags |= EXEC_OBJECT_PINNED;

   result = anv_bo_cache_import(device, &device->bo_cache, dma_buf, bo_flags, &bo);
   if (result != VK_SUCCESS) {
      return vk_errorf(device->instance, device, result,
                       "failed to import dma-buf from VkNativeBufferANDROID");
   }

   int i915_tiling = anv_gem_get_tiling(device, bo->gem_handle);
   switch (i915_tiling) {
   case I915_TILING_NONE:
      anv_info.isl_tiling_flags = ISL_TILING_LINEAR_BIT;
      break;
   case I915_TILING_X:
      anv_info.isl_tiling_flags = ISL_TILING_X_BIT;
      break;
   case I915_TILING_Y:
      anv_info.isl_tiling_flags = ISL_TILING_Y0_BIT;
      break;
   case -1:
      result = vk_errorf(device->instance, device,
                         VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR,
                         "DRM_IOCTL_I915_GEM_GET_TILING failed for "
                         "VkNativeBufferANDROID");
      goto fail_tiling;
   default:
      result = vk_errorf(device->instance, device,
                         VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR,
                         "DRM_IOCTL_I915_GEM_GET_TILING returned unknown "
                         "tiling %d for VkNativeBufferANDROID", i915_tiling);
      goto fail_tiling;
   }

   enum isl_format format = anv_get_isl_format(&device->info,
                                               base_info->format,
                                               VK_IMAGE_ASPECT_COLOR_BIT,
                                               base_info->tiling);
   assert(format != ISL_FORMAT_UNSUPPORTED);

   anv_info.stride = gralloc_info->stride *
                     (isl_format_get_layout(format)->bpb / 8);

   result = anv_image_create(device_h, &anv_info, alloc, &image_h);
   image = anv_image_from_handle(image_h);
   if (result != VK_SUCCESS)
      goto fail_create;

   if (bo->size < image->size) {
      result = vk_errorf(device->instance, device,
                         VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR,
                         "dma-buf from VkNativeBufferANDROID is too small for "
                         "VkImage: %"PRIu64"B < %"PRIu64"B",
                         bo->size, image->size);
      goto fail_size;
   }

   assert(image->n_planes == 1);
   assert(image->planes[0].address.offset == 0);

   image->planes[0].address.bo = bo;
   image->planes[0].bo_is_owned = true;

   /* We need to set the WRITE flag on window system buffers so that GEM will
    * know we're writing to them and synchronize uses on other rings (for
    * example, if the display server uses the blitter ring).
    *
    * If this function fails and if the imported bo was resident in the cache,
    * we should avoid updating the bo's flags. Therefore, we defer updating
    * the flags until success is certain.
    *
    */
   bo->flags &= ~EXEC_OBJECT_ASYNC;
   bo->flags |= EXEC_OBJECT_WRITE;

   /* Don't clobber the out-parameter until success is certain. */
   *out_image_h = image_h;

   return VK_SUCCESS;

 fail_size:
   anv_DestroyImage(device_h, image_h, alloc);
 fail_create:
 fail_tiling:
   anv_bo_cache_release(device, &device->bo_cache, bo);

   return result;
}

VkResult anv_GetSwapchainGrallocUsageANDROID(
    VkDevice            device_h,
    VkFormat            format,
    VkImageUsageFlags   imageUsage,
    int*                grallocUsage)
{
   ANV_FROM_HANDLE(anv_device, device, device_h);
   struct anv_physical_device *phys_dev = &device->instance->physicalDevice;
   VkPhysicalDevice phys_dev_h = anv_physical_device_to_handle(phys_dev);
   VkResult result;

   *grallocUsage = 0;
   intel_logd("%s: format=%d, usage=0x%x", __func__, format, imageUsage);

   /* WARNING: Android's libvulkan.so hardcodes the VkImageUsageFlags
    * returned to applications via VkSurfaceCapabilitiesKHR::supportedUsageFlags.
    * The relevant code in libvulkan/swapchain.cpp contains this fun comment:
    *
    *     TODO(jessehall): I think these are right, but haven't thought hard
    *     about it. Do we need to query the driver for support of any of
    *     these?
    *
    * Any disagreement between this function and the hardcoded
    * VkSurfaceCapabilitiesKHR:supportedUsageFlags causes tests
    * dEQP-VK.wsi.android.swapchain.*.image_usage to fail.
    */

   const VkPhysicalDeviceImageFormatInfo2KHR image_format_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR,
      .format = format,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = imageUsage,
   };

   VkImageFormatProperties2KHR image_format_props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR,
   };

   /* Check that requested format and usage are supported. */
   result = anv_GetPhysicalDeviceImageFormatProperties2(phys_dev_h,
               &image_format_info, &image_format_props);
   if (result != VK_SUCCESS) {
      return vk_errorf(device->instance, device, result,
                       "anv_GetPhysicalDeviceImageFormatProperties2 failed "
                       "inside %s", __func__);
   }

   if (unmask32(&imageUsage, VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
      *grallocUsage |= GRALLOC_USAGE_HW_RENDER;

   if (unmask32(&imageUsage, VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_STORAGE_BIT |
                             VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT))
      *grallocUsage |= GRALLOC_USAGE_HW_TEXTURE;

   /* All VkImageUsageFlags not explicitly checked here are unsupported for
    * gralloc swapchains.
    */
   if (imageUsage != 0) {
      return vk_errorf(device->instance, device, VK_ERROR_FORMAT_NOT_SUPPORTED,
                       "unsupported VkImageUsageFlags(0x%x) for gralloc "
                       "swapchain", imageUsage);
   }

   /* The below formats support GRALLOC_USAGE_HW_FB (that is, display
    * scanout). This short list of formats is univserally supported on Intel
    * but is incomplete.  The full set of supported formats is dependent on
    * kernel and hardware.
    *
    * FINISHME: Advertise all display-supported formats.
    */
   switch (format) {
      case VK_FORMAT_B8G8R8A8_UNORM:
      case VK_FORMAT_B5G6R5_UNORM_PACK16:
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_R8G8B8A8_SRGB:
         *grallocUsage |= GRALLOC_USAGE_HW_FB |
                          GRALLOC_USAGE_HW_COMPOSER |
                          GRALLOC_USAGE_EXTERNAL_DISP;
         break;
      default:
         intel_logw("%s: unsupported format=%d", __func__, format);
   }

   if (*grallocUsage == 0)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   return VK_SUCCESS;
}

VkResult
anv_AcquireImageANDROID(
      VkDevice            device_h,
      VkImage             image_h,
      int                 nativeFenceFd,
      VkSemaphore         semaphore_h,
      VkFence             fence_h)
{
   ANV_FROM_HANDLE(anv_device, device, device_h);
   VkResult result = VK_SUCCESS;

   if (nativeFenceFd != -1) {
      /* As a simple, firstpass implementation of VK_ANDROID_native_buffer, we
       * block on the nativeFenceFd. This may introduce latency and is
       * definitiely inefficient, yet it's correct.
       *
       * FINISHME(chadv): Import the nativeFenceFd into the VkSemaphore and
       * VkFence.
       */
      if (sync_wait(nativeFenceFd, /*timeout*/ -1) < 0) {
         result = vk_errorf(device->instance, device, VK_ERROR_DEVICE_LOST,
                            "%s: failed to wait on nativeFenceFd=%d",
                            __func__, nativeFenceFd);
      }

      /* From VK_ANDROID_native_buffer's pseudo spec
       * (https://source.android.com/devices/graphics/implement-vulkan):
       *
       *    The driver takes ownership of the fence fd and is responsible for
       *    closing it [...] even if vkAcquireImageANDROID fails and returns
       *    an error.
       */
      close(nativeFenceFd);

      if (result != VK_SUCCESS)
         return result;
   }

   if (semaphore_h || fence_h) {
      /* Thanks to implicit sync, the image is ready for GPU access.  But we
       * must still put the semaphore into the "submit" state; otherwise the
       * client may get unexpected behavior if the client later uses it as
       * a wait semaphore.
       *
       * Because we blocked above on the nativeFenceFd, the image is also
       * ready for foreign-device access (including CPU access). But we must
       * still signal the fence; otherwise the client may get unexpected
       * behavior if the client later waits on it.
       *
       * For some values of anv_semaphore_type, we must submit the semaphore
       * to execbuf in order to signal it.  Likewise for anv_fence_type.
       * Instead of open-coding here the signal operation for each
       * anv_semaphore_type and anv_fence_type, we piggy-back on
       * vkQueueSubmit.
       */
      const VkSubmitInfo submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .waitSemaphoreCount = 0,
         .commandBufferCount = 0,
         .signalSemaphoreCount = (semaphore_h ? 1 : 0),
         .pSignalSemaphores = &semaphore_h,
      };

      result = anv_QueueSubmit(anv_queue_to_handle(&device->queue), 1,
                               &submit, fence_h);
      if (result != VK_SUCCESS) {
         return vk_errorf(device->instance, device, result,
                          "anv_QueueSubmit failed inside %s", __func__);
      }
   }

   return VK_SUCCESS;
}

VkResult
anv_QueueSignalReleaseImageANDROID(
      VkQueue             queue,
      uint32_t            waitSemaphoreCount,
      const VkSemaphore*  pWaitSemaphores,
      VkImage             image,
      int*                pNativeFenceFd)
{
   VkResult result;

   if (waitSemaphoreCount == 0)
      goto done;

   result = anv_QueueSubmit(queue, 1,
      &(VkSubmitInfo) {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = pWaitSemaphores,
      },
      (VkFence) VK_NULL_HANDLE);
   if (result != VK_SUCCESS)
      return result;

 done:
   if (pNativeFenceFd) {
      /* We can rely implicit on sync because above we submitted all
       * semaphores to the queue.
       */
      *pNativeFenceFd = -1;
   }

   return VK_SUCCESS;
}
