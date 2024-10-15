/*
 * Copyright 2017 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "anv_private.h"

#include "vk_util.h"

/* Convert the VK_USE_PLATFORM_* defines to booleans */
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#   undef VK_USE_PLATFORM_ANDROID_KHR
#   define VK_USE_PLATFORM_ANDROID_KHR true
#else
#   define VK_USE_PLATFORM_ANDROID_KHR false
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
#   undef VK_USE_PLATFORM_WAYLAND_KHR
#   define VK_USE_PLATFORM_WAYLAND_KHR true
#else
#   define VK_USE_PLATFORM_WAYLAND_KHR false
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
#   undef VK_USE_PLATFORM_XCB_KHR
#   define VK_USE_PLATFORM_XCB_KHR true
#else
#   define VK_USE_PLATFORM_XCB_KHR false
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
#   undef VK_USE_PLATFORM_XLIB_KHR
#   define VK_USE_PLATFORM_XLIB_KHR true
#else
#   define VK_USE_PLATFORM_XLIB_KHR false
#endif
#ifdef VK_USE_PLATFORM_DISPLAY_KHR
#   undef VK_USE_PLATFORM_DISPLAY_KHR
#   define VK_USE_PLATFORM_DISPLAY_KHR true
#else
#   define VK_USE_PLATFORM_DISPLAY_KHR false
#endif
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
#   undef VK_USE_PLATFORM_XLIB_XRANDR_EXT
#   define VK_USE_PLATFORM_XLIB_XRANDR_EXT true
#else
#   define VK_USE_PLATFORM_XLIB_XRANDR_EXT false
#endif

/* And ANDROID too */
#ifdef ANDROID
#   undef ANDROID
#   define ANDROID true
#else
#   define ANDROID false
#endif

#define ANV_HAS_SURFACE (VK_USE_PLATFORM_WAYLAND_KHR ||                          VK_USE_PLATFORM_XCB_KHR ||                          VK_USE_PLATFORM_XLIB_KHR ||                          VK_USE_PLATFORM_DISPLAY_KHR)

static const uint32_t MAX_API_VERSION = VK_MAKE_VERSION(1, 1, 90);

VkResult anv_EnumerateInstanceVersion(
    uint32_t*                                   pApiVersion)
{
    *pApiVersion = MAX_API_VERSION;
    return VK_SUCCESS;
}

const VkExtensionProperties anv_instance_extensions[ANV_INSTANCE_EXTENSION_COUNT] = {
   {"VK_KHR_device_group_creation", 1},
   {"VK_KHR_external_fence_capabilities", 1},
   {"VK_KHR_external_memory_capabilities", 1},
   {"VK_KHR_external_semaphore_capabilities", 1},
   {"VK_KHR_get_display_properties2", 1},
   {"VK_KHR_get_physical_device_properties2", 1},
   {"VK_KHR_get_surface_capabilities2", 1},
   {"VK_KHR_surface", 25},
   {"VK_KHR_wayland_surface", 6},
   {"VK_KHR_xcb_surface", 6},
   {"VK_KHR_xlib_surface", 6},
   {"VK_KHR_display", 23},
   {"VK_EXT_acquire_xlib_display", 1},
   {"VK_EXT_debug_report", 8},
   {"VK_EXT_direct_mode_display", 1},
   {"VK_EXT_display_surface_counter", 1},
};

const struct anv_instance_extension_table anv_instance_extensions_supported = {
   .KHR_device_group_creation = true,
   .KHR_external_fence_capabilities = true,
   .KHR_external_memory_capabilities = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_get_display_properties2 = VK_USE_PLATFORM_DISPLAY_KHR,
   .KHR_get_physical_device_properties2 = true,
   .KHR_get_surface_capabilities2 = ANV_HAS_SURFACE,
   .KHR_surface = ANV_HAS_SURFACE,
   .KHR_wayland_surface = VK_USE_PLATFORM_WAYLAND_KHR,
   .KHR_xcb_surface = VK_USE_PLATFORM_XCB_KHR,
   .KHR_xlib_surface = VK_USE_PLATFORM_XLIB_KHR,
   .KHR_display = VK_USE_PLATFORM_DISPLAY_KHR,
   .EXT_acquire_xlib_display = VK_USE_PLATFORM_XLIB_XRANDR_EXT,
   .EXT_debug_report = true,
   .EXT_direct_mode_display = VK_USE_PLATFORM_DISPLAY_KHR,
   .EXT_display_surface_counter = VK_USE_PLATFORM_DISPLAY_KHR,
};

uint32_t
anv_physical_device_api_version(struct anv_physical_device *device)
{
    uint32_t version = 0;

    uint32_t override = vk_get_version_override();
    if (override)
        return MIN2(override, MAX_API_VERSION);

    if (!(true))
        return version;
    version = VK_MAKE_VERSION(1, 0, 90);

    if (!(device->has_syncobj_wait))
        return version;
    version = VK_MAKE_VERSION(1, 1, 90);

    return version;
}

const VkExtensionProperties anv_device_extensions[ANV_DEVICE_EXTENSION_COUNT] = {
   {"VK_ANDROID_native_buffer", 5},
   {"VK_KHR_16bit_storage", 1},
   {"VK_KHR_8bit_storage", 1},
   {"VK_KHR_bind_memory2", 1},
   {"VK_KHR_create_renderpass2", 1},
   {"VK_KHR_dedicated_allocation", 1},
   {"VK_KHR_descriptor_update_template", 1},
   {"VK_KHR_device_group", 1},
   {"VK_KHR_driver_properties", 1},
   {"VK_KHR_external_fence", 1},
   {"VK_KHR_external_fence_fd", 1},
   {"VK_KHR_external_memory", 1},
   {"VK_KHR_external_memory_fd", 1},
   {"VK_KHR_external_semaphore", 1},
   {"VK_KHR_external_semaphore_fd", 1},
   {"VK_KHR_get_memory_requirements2", 1},
   {"VK_KHR_image_format_list", 1},
   {"VK_KHR_incremental_present", 1},
   {"VK_KHR_maintenance1", 1},
   {"VK_KHR_maintenance2", 1},
   {"VK_KHR_maintenance3", 1},
   {"VK_KHR_push_descriptor", 1},
   {"VK_KHR_relaxed_block_layout", 1},
   {"VK_KHR_sampler_mirror_clamp_to_edge", 1},
   {"VK_KHR_sampler_ycbcr_conversion", 1},
   {"VK_KHR_shader_draw_parameters", 1},
   {"VK_KHR_storage_buffer_storage_class", 1},
   {"VK_KHR_swapchain", 68},
   {"VK_KHR_variable_pointers", 1},
   {"VK_KHR_multiview", 1},
   {"VK_EXT_display_control", 1},
   {"VK_EXT_external_memory_dma_buf", 1},
   {"VK_EXT_global_priority", 1},
   {"VK_EXT_pci_bus_info", 1},
   {"VK_EXT_shader_viewport_index_layer", 1},
   {"VK_EXT_shader_stencil_export", 1},
   {"VK_EXT_vertex_attribute_divisor", 3},
   {"VK_EXT_post_depth_coverage", 1},
   {"VK_EXT_sampler_filter_minmax", 1},
   {"VK_EXT_calibrated_timestamps", 1},
   {"VK_GOOGLE_decorate_string", 1},
   {"VK_GOOGLE_hlsl_functionality1", 1},
};

void
anv_physical_device_get_supported_extensions(const struct anv_physical_device *device,
                                             struct anv_device_extension_table *extensions)
{
   *extensions = (struct anv_device_extension_table) {
      .ANDROID_native_buffer = ANDROID,
      .KHR_16bit_storage = device->info.gen >= 8,
      .KHR_8bit_storage = device->info.gen >= 8,
      .KHR_bind_memory2 = true,
      .KHR_create_renderpass2 = true,
      .KHR_dedicated_allocation = true,
      .KHR_descriptor_update_template = true,
      .KHR_device_group = true,
      .KHR_driver_properties = true,
      .KHR_external_fence = device->has_syncobj_wait,
      .KHR_external_fence_fd = device->has_syncobj_wait,
      .KHR_external_memory = true,
      .KHR_external_memory_fd = true,
      .KHR_external_semaphore = true,
      .KHR_external_semaphore_fd = true,
      .KHR_get_memory_requirements2 = true,
      .KHR_image_format_list = true,
      .KHR_incremental_present = ANV_HAS_SURFACE,
      .KHR_maintenance1 = true,
      .KHR_maintenance2 = true,
      .KHR_maintenance3 = true,
      .KHR_push_descriptor = true,
      .KHR_relaxed_block_layout = true,
      .KHR_sampler_mirror_clamp_to_edge = true,
      .KHR_sampler_ycbcr_conversion = true,
      .KHR_shader_draw_parameters = true,
      .KHR_storage_buffer_storage_class = true,
      .KHR_swapchain = ANV_HAS_SURFACE,
      .KHR_variable_pointers = true,
      .KHR_multiview = true,
      .EXT_display_control = VK_USE_PLATFORM_DISPLAY_KHR,
      .EXT_external_memory_dma_buf = true,
      .EXT_global_priority = device->has_context_priority,
      .EXT_pci_bus_info = false,
      .EXT_shader_viewport_index_layer = true,
      .EXT_shader_stencil_export = device->info.gen >= 9,
      .EXT_vertex_attribute_divisor = true,
      .EXT_post_depth_coverage = device->info.gen >= 9,
      .EXT_sampler_filter_minmax = device->info.gen >= 9,
      .EXT_calibrated_timestamps = true,
      .GOOGLE_decorate_string = true,
      .GOOGLE_hlsl_functionality1 = true,
   };
}
