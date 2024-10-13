COPYRIGHT = """\
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
"""

import argparse
import copy
import re
import xml.etree.cElementTree as et

def _bool_to_c_expr(b):
    if b is True:
        return 'true'
    if b is False:
        return 'false'
    return b

class Extension:
    def __init__(self, name, ext_version, enable):
        self.name = name
        self.ext_version = int(ext_version)
        self.enable = _bool_to_c_expr(enable)

class ApiVersion:
    def __init__(self, version, enable):
        self.version = version
        self.enable = _bool_to_c_expr(enable)

API_PATCH_VERSION = 90

# Supported API versions.  Each one is the maximum patch version for the given
# version.  Version come in increasing order and each version is available if
# it's provided "enable" condition is true and all previous versions are
# available.
API_VERSIONS = [
    ApiVersion('1.0',   True),

    # DRM_IOCTL_SYNCOBJ_WAIT is required for VK_KHR_external_fence which is a
    # required core feature in Vulkan 1.1
    ApiVersion('1.1',   'device->has_syncobj_wait'),
]

MAX_API_VERSION = None # Computed later

# On Android, we disable all surface and swapchain extensions. Android's Vulkan
# loader implements VK_KHR_surface and VK_KHR_swapchain, and applications
# cannot access the driver's implementation. Moreoever, if the driver exposes
# the those extension strings, then tests dEQP-VK.api.info.instance.extensions
# and dEQP-VK.api.info.device fail due to the duplicated strings.
EXTENSIONS = [
    Extension('VK_ANDROID_native_buffer',                 5, 'ANDROID'),
    Extension('VK_KHR_16bit_storage',                     1, 'device->info.gen >= 8'),
    Extension('VK_KHR_8bit_storage',                      1, 'device->info.gen >= 8'),
    Extension('VK_KHR_bind_memory2',                      1, True),
    Extension('VK_KHR_create_renderpass2',                1, True),
    Extension('VK_KHR_dedicated_allocation',              1, True),
    Extension('VK_KHR_descriptor_update_template',        1, True),
    Extension('VK_KHR_device_group',                      1, True),
    Extension('VK_KHR_device_group_creation',             1, True),
    Extension('VK_KHR_driver_properties',                 1, True),
    Extension('VK_KHR_external_fence',                    1,
              'device->has_syncobj_wait'),
    Extension('VK_KHR_external_fence_capabilities',       1, True),
    Extension('VK_KHR_external_fence_fd',                 1,
              'device->has_syncobj_wait'),
    Extension('VK_KHR_external_memory',                   1, True),
    Extension('VK_KHR_external_memory_capabilities',      1, True),
    Extension('VK_KHR_external_memory_fd',                1, True),
    Extension('VK_KHR_external_semaphore',                1, True),
    Extension('VK_KHR_external_semaphore_capabilities',   1, True),
    Extension('VK_KHR_external_semaphore_fd',             1, True),
    Extension('VK_KHR_get_display_properties2',           1, 'VK_USE_PLATFORM_DISPLAY_KHR'),
    Extension('VK_KHR_get_memory_requirements2',          1, True),
    Extension('VK_KHR_get_physical_device_properties2',   1, True),
    Extension('VK_KHR_get_surface_capabilities2',         1, 'ANV_HAS_SURFACE'),
    Extension('VK_KHR_image_format_list',                 1, True),
    Extension('VK_KHR_incremental_present',               1, 'ANV_HAS_SURFACE'),
    Extension('VK_KHR_maintenance1',                      1, True),
    Extension('VK_KHR_maintenance2',                      1, True),
    Extension('VK_KHR_maintenance3',                      1, True),
    Extension('VK_KHR_push_descriptor',                   1, True),
    Extension('VK_KHR_relaxed_block_layout',              1, True),
    Extension('VK_KHR_sampler_mirror_clamp_to_edge',      1, True),
    Extension('VK_KHR_sampler_ycbcr_conversion',          1, True),
    Extension('VK_KHR_shader_draw_parameters',            1, True),
    Extension('VK_KHR_storage_buffer_storage_class',      1, True),
    Extension('VK_KHR_surface',                          25, 'ANV_HAS_SURFACE'),
    Extension('VK_KHR_swapchain',                        68, 'ANV_HAS_SURFACE'),
    Extension('VK_KHR_variable_pointers',                 1, True),
    Extension('VK_KHR_wayland_surface',                   6, 'VK_USE_PLATFORM_WAYLAND_KHR'),
    Extension('VK_KHR_xcb_surface',                       6, 'VK_USE_PLATFORM_XCB_KHR'),
    Extension('VK_KHR_xlib_surface',                      6, 'VK_USE_PLATFORM_XLIB_KHR'),
    Extension('VK_KHR_multiview',                         1, True),
    Extension('VK_KHR_display',                          23, 'VK_USE_PLATFORM_DISPLAY_KHR'),
    Extension('VK_EXT_acquire_xlib_display',              1, 'VK_USE_PLATFORM_XLIB_XRANDR_EXT'),
    Extension('VK_EXT_debug_report',                      8, True),
    Extension('VK_EXT_direct_mode_display',               1, 'VK_USE_PLATFORM_DISPLAY_KHR'),
    Extension('VK_EXT_display_control',                   1, 'VK_USE_PLATFORM_DISPLAY_KHR'),
    Extension('VK_EXT_display_surface_counter',           1, 'VK_USE_PLATFORM_DISPLAY_KHR'),
    Extension('VK_EXT_external_memory_dma_buf',           1, True),
    Extension('VK_EXT_global_priority',                   1,
              'device->has_context_priority'),
    Extension('VK_EXT_pci_bus_info',                      1, False),
    Extension('VK_EXT_shader_viewport_index_layer',       1, True),
    Extension('VK_EXT_shader_stencil_export',             1, 'device->info.gen >= 9'),
    Extension('VK_EXT_vertex_attribute_divisor',          3, True),
    Extension('VK_EXT_post_depth_coverage',               1, 'device->info.gen >= 9'),
    Extension('VK_EXT_sampler_filter_minmax',             1, 'device->info.gen >= 9'),
    Extension('VK_EXT_calibrated_timestamps',             1, True),
    Extension('VK_GOOGLE_decorate_string',                1, True),
    Extension('VK_GOOGLE_hlsl_functionality1',            1, True),
]

class VkVersion:
    def __init__(self, string):
        split = string.split('.')
        self.major = int(split[0])
        self.minor = int(split[1])
        if len(split) > 2:
            assert len(split) == 3
            self.patch = int(split[2])
        else:
            self.patch = None

        # Sanity check.  The range bits are required by the definition of the
        # VK_MAKE_VERSION macro
        assert self.major < 1024 and self.minor < 1024
        assert self.patch is None or self.patch < 4096
        assert str(self) == string

    def __str__(self):
        ver_list = [str(self.major), str(self.minor)]
        if self.patch is not None:
            ver_list.append(str(self.patch))
        return '.'.join(ver_list)

    def c_vk_version(self):
        patch = self.patch if self.patch is not None else 0
        ver_list = [str(self.major), str(self.minor), str(patch)]
        return 'VK_MAKE_VERSION(' + ', '.join(ver_list) + ')'

    def __int_ver(self):
        # This is just an expansion of VK_VERSION
        patch = self.patch if self.patch is not None else 0
        return (self.major << 22) | (self.minor << 12) | patch

    def __gt__(self, other):
        # If only one of them has a patch version, "ignore" it by making
        # other's patch version match self.
        if (self.patch is None) != (other.patch is None):
            other = copy.copy(other)
            other.patch = self.patch

        return self.__int_ver() > other.__int_ver()



MAX_API_VERSION = VkVersion('0.0.0')
for version in API_VERSIONS:
    version.version = VkVersion(version.version)
    version.version.patch = API_PATCH_VERSION
    assert version.version > MAX_API_VERSION
    MAX_API_VERSION = version.version
