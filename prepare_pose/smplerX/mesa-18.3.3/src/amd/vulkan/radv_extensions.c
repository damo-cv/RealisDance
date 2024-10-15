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

#include "radv_private.h"

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

#define RADV_HAS_SURFACE (VK_USE_PLATFORM_WAYLAND_KHR ||                          VK_USE_PLATFORM_XCB_KHR ||                          VK_USE_PLATFORM_XLIB_KHR ||                          VK_USE_PLATFORM_DISPLAY_KHR)


const VkExtensionProperties radv_instance_extensions[RADV_INSTANCE_EXTENSION_COUNT] = {
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
   {"VK_EXT_direct_mode_display", 1},
   {"VK_EXT_acquire_xlib_display", 1},
   {"VK_EXT_display_surface_counter", 1},
   {"VK_EXT_debug_report", 9},
};

const VkExtensionProperties radv_device_extensions[RADV_DEVICE_EXTENSION_COUNT] = {
   {"VK_ANDROID_native_buffer", 5},
   {"VK_KHR_16bit_storage", 1},
   {"VK_KHR_bind_memory2", 1},
   {"VK_KHR_create_renderpass2", 1},
   {"VK_KHR_dedicated_allocation", 1},
   {"VK_KHR_descriptor_update_template", 1},
   {"VK_KHR_device_group", 1},
   {"VK_KHR_draw_indirect_count", 1},
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
   {"VK_KHR_shader_draw_parameters", 1},
   {"VK_KHR_storage_buffer_storage_class", 1},
   {"VK_KHR_swapchain", 68},
   {"VK_KHR_variable_pointers", 1},
   {"VK_KHR_multiview", 1},
   {"VK_EXT_calibrated_timestamps", 1},
   {"VK_EXT_conditional_rendering", 1},
   {"VK_EXT_conservative_rasterization", 1},
   {"VK_EXT_display_control", 1},
   {"VK_EXT_depth_range_unrestricted", 1},
   {"VK_EXT_descriptor_indexing", 2},
   {"VK_EXT_discard_rectangles", 1},
   {"VK_EXT_external_memory_dma_buf", 1},
   {"VK_EXT_external_memory_host", 1},
   {"VK_EXT_global_priority", 1},
   {"VK_EXT_pci_bus_info", 1},
   {"VK_EXT_sampler_filter_minmax", 1},
   {"VK_EXT_shader_viewport_index_layer", 1},
   {"VK_EXT_shader_stencil_export", 1},
   {"VK_EXT_transform_feedback", 1},
   {"VK_EXT_vertex_attribute_divisor", 3},
   {"VK_AMD_draw_indirect_count", 1},
   {"VK_AMD_gcn_shader", 1},
   {"VK_AMD_rasterization_order", 1},
   {"VK_AMD_shader_core_properties", 1},
   {"VK_AMD_shader_info", 1},
   {"VK_AMD_shader_trinary_minmax", 1},
   {"VK_GOOGLE_decorate_string", 1},
   {"VK_GOOGLE_hlsl_functionality1", 1},
};

const struct radv_instance_extension_table radv_supported_instance_extensions = {
   .KHR_device_group_creation = true,
   .KHR_external_fence_capabilities = true,
   .KHR_external_memory_capabilities = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_get_display_properties2 = VK_USE_PLATFORM_DISPLAY_KHR,
   .KHR_get_physical_device_properties2 = true,
   .KHR_get_surface_capabilities2 = RADV_HAS_SURFACE,
   .KHR_surface = RADV_HAS_SURFACE,
   .KHR_wayland_surface = VK_USE_PLATFORM_WAYLAND_KHR,
   .KHR_xcb_surface = VK_USE_PLATFORM_XCB_KHR,
   .KHR_xlib_surface = VK_USE_PLATFORM_XLIB_KHR,
   .KHR_display = VK_USE_PLATFORM_DISPLAY_KHR,
   .EXT_direct_mode_display = VK_USE_PLATFORM_DISPLAY_KHR,
   .EXT_acquire_xlib_display = VK_USE_PLATFORM_XLIB_XRANDR_EXT,
   .EXT_display_surface_counter = VK_USE_PLATFORM_DISPLAY_KHR,
   .EXT_debug_report = true,
};

void radv_fill_device_extension_table(const struct radv_physical_device *device,
                                      struct radv_device_extension_table* table)
{
   table->ANDROID_native_buffer = ANDROID && device->rad_info.has_syncobj_wait_for_submit;
   table->KHR_16bit_storage = HAVE_LLVM >= 0x0700;
   table->KHR_bind_memory2 = true;
   table->KHR_create_renderpass2 = true;
   table->KHR_dedicated_allocation = true;
   table->KHR_descriptor_update_template = true;
   table->KHR_device_group = true;
   table->KHR_draw_indirect_count = true;
   table->KHR_driver_properties = true;
   table->KHR_external_fence = device->rad_info.has_syncobj_wait_for_submit;
   table->KHR_external_fence_fd = device->rad_info.has_syncobj_wait_for_submit;
   table->KHR_external_memory = true;
   table->KHR_external_memory_fd = true;
   table->KHR_external_semaphore = device->rad_info.has_syncobj;
   table->KHR_external_semaphore_fd = device->rad_info.has_syncobj;
   table->KHR_get_memory_requirements2 = true;
   table->KHR_image_format_list = true;
   table->KHR_incremental_present = RADV_HAS_SURFACE;
   table->KHR_maintenance1 = true;
   table->KHR_maintenance2 = true;
   table->KHR_maintenance3 = true;
   table->KHR_push_descriptor = true;
   table->KHR_relaxed_block_layout = true;
   table->KHR_sampler_mirror_clamp_to_edge = true;
   table->KHR_shader_draw_parameters = true;
   table->KHR_storage_buffer_storage_class = true;
   table->KHR_swapchain = RADV_HAS_SURFACE;
   table->KHR_variable_pointers = true;
   table->KHR_multiview = true;
   table->EXT_calibrated_timestamps = true;
   table->EXT_conditional_rendering = true;
   table->EXT_conservative_rasterization = device->rad_info.chip_class >= GFX9;
   table->EXT_display_control = VK_USE_PLATFORM_DISPLAY_KHR;
   table->EXT_depth_range_unrestricted = true;
   table->EXT_descriptor_indexing = true;
   table->EXT_discard_rectangles = true;
   table->EXT_external_memory_dma_buf = true;
   table->EXT_external_memory_host = device->rad_info.has_userptr;
   table->EXT_global_priority = device->rad_info.has_ctx_priority;
   table->EXT_pci_bus_info = false;
   table->EXT_sampler_filter_minmax = device->rad_info.chip_class >= CIK;
   table->EXT_shader_viewport_index_layer = true;
   table->EXT_shader_stencil_export = true;
   table->EXT_transform_feedback = true;
   table->EXT_vertex_attribute_divisor = true;
   table->AMD_draw_indirect_count = true;
   table->AMD_gcn_shader = true;
   table->AMD_rasterization_order = device->has_out_of_order_rast;
   table->AMD_shader_core_properties = true;
   table->AMD_shader_info = true;
   table->AMD_shader_trinary_minmax = true;
   table->GOOGLE_decorate_string = true;
   table->GOOGLE_hlsl_functionality1 = true;
}

VkResult radv_EnumerateInstanceVersion(
    uint32_t*                                   pApiVersion)
{
    *pApiVersion = VK_MAKE_VERSION(1, 1, 70);
    return VK_SUCCESS;
}

uint32_t
radv_physical_device_api_version(struct radv_physical_device *dev)
{
    if (!ANDROID && dev->rad_info.has_syncobj_wait_for_submit)
        return VK_MAKE_VERSION(1, 1, 70);
    return VK_MAKE_VERSION(1, 0, 68);
}
