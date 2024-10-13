//
// Copyright 2012 Francisco Jerez
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//

#include <unistd.h>
#include "core/device.hpp"
#include "core/platform.hpp"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/u_debug.h"

using namespace clover;

namespace {
   template<typename T>
   std::vector<T>
   get_compute_param(pipe_screen *pipe, pipe_shader_ir ir_format,
                     pipe_compute_cap cap) {
      int sz = pipe->get_compute_param(pipe, ir_format, cap, NULL);
      std::vector<T> v(sz / sizeof(T));

      pipe->get_compute_param(pipe, ir_format, cap, &v.front());
      return v;
   }
}

device::device(clover::platform &platform, pipe_loader_device *ldev) :
   platform(platform), ldev(ldev) {
   pipe = pipe_loader_create_screen(ldev);
   if (!pipe || !pipe->get_param(pipe, PIPE_CAP_COMPUTE)) {
      if (pipe)
         pipe->destroy(pipe);
      throw error(CL_INVALID_DEVICE);
   }
}

device::~device() {
   if (pipe)
      pipe->destroy(pipe);
   if (ldev)
      pipe_loader_release(&ldev, 1);
}

bool
device::operator==(const device &dev) const {
   return this == &dev;
}

cl_device_type
device::type() const {
   switch (ldev->type) {
   case PIPE_LOADER_DEVICE_SOFTWARE:
      return CL_DEVICE_TYPE_CPU;
   case PIPE_LOADER_DEVICE_PCI:
   case PIPE_LOADER_DEVICE_PLATFORM:
      return CL_DEVICE_TYPE_GPU;
   default:
      unreachable("Unknown device type.");
   }
}

cl_uint
device::vendor_id() const {
   switch (ldev->type) {
   case PIPE_LOADER_DEVICE_SOFTWARE:
   case PIPE_LOADER_DEVICE_PLATFORM:
      return 0;
   case PIPE_LOADER_DEVICE_PCI:
      return ldev->u.pci.vendor_id;
   default:
      unreachable("Unknown device type.");
   }
}

size_t
device::max_images_read() const {
   return PIPE_MAX_SHADER_IMAGES;
}

size_t
device::max_images_write() const {
   return PIPE_MAX_SHADER_IMAGES;
}

size_t
device::max_image_buffer_size() const {
   return pipe->get_param(pipe, PIPE_CAP_MAX_TEXTURE_BUFFER_SIZE);
}

cl_uint
device::max_image_levels_2d() const {
   return pipe->get_param(pipe, PIPE_CAP_MAX_TEXTURE_2D_LEVELS);
}

cl_uint
device::max_image_levels_3d() const {
   return pipe->get_param(pipe, PIPE_CAP_MAX_TEXTURE_3D_LEVELS);
}

size_t
device::max_image_array_number() const {
   return pipe->get_param(pipe, PIPE_CAP_MAX_TEXTURE_ARRAY_LAYERS);
}

cl_uint
device::max_samplers() const {
   return pipe->get_shader_param(pipe, PIPE_SHADER_COMPUTE,
                                 PIPE_SHADER_CAP_MAX_TEXTURE_SAMPLERS);
}

cl_ulong
device::max_mem_global() const {
   return get_compute_param<uint64_t>(pipe, ir_format(),
                                      PIPE_COMPUTE_CAP_MAX_GLOBAL_SIZE)[0];
}

cl_ulong
device::max_mem_local() const {
   return get_compute_param<uint64_t>(pipe, ir_format(),
                                      PIPE_COMPUTE_CAP_MAX_LOCAL_SIZE)[0];
}

cl_ulong
device::max_mem_input() const {
   return get_compute_param<uint64_t>(pipe, ir_format(),
                                      PIPE_COMPUTE_CAP_MAX_INPUT_SIZE)[0];
}

cl_ulong
device::max_const_buffer_size() const {
   return pipe->get_shader_param(pipe, PIPE_SHADER_COMPUTE,
                                 PIPE_SHADER_CAP_MAX_CONST_BUFFER_SIZE);
}

cl_uint
device::max_const_buffers() const {
   return pipe->get_shader_param(pipe, PIPE_SHADER_COMPUTE,
                                 PIPE_SHADER_CAP_MAX_CONST_BUFFERS);
}

size_t
device::max_threads_per_block() const {
   return get_compute_param<uint64_t>(
      pipe, ir_format(), PIPE_COMPUTE_CAP_MAX_THREADS_PER_BLOCK)[0];
}

cl_ulong
device::max_mem_alloc_size() const {
   return get_compute_param<uint64_t>(pipe, ir_format(),
                                      PIPE_COMPUTE_CAP_MAX_MEM_ALLOC_SIZE)[0];
}

cl_uint
device::max_clock_frequency() const {
   return get_compute_param<uint32_t>(pipe, ir_format(),
                                      PIPE_COMPUTE_CAP_MAX_CLOCK_FREQUENCY)[0];
}

cl_uint
device::max_compute_units() const {
   return get_compute_param<uint32_t>(pipe, ir_format(),
                                      PIPE_COMPUTE_CAP_MAX_COMPUTE_UNITS)[0];
}

bool
device::image_support() const {
   return get_compute_param<uint32_t>(pipe, ir_format(),
                                      PIPE_COMPUTE_CAP_IMAGES_SUPPORTED)[0];
}

bool
device::has_doubles() const {
   return pipe->get_param(pipe, PIPE_CAP_DOUBLES);
}

bool
device::has_halves() const {
   return pipe->get_shader_param(pipe, PIPE_SHADER_COMPUTE,
                                 PIPE_SHADER_CAP_FP16);
}

bool
device::has_int64_atomics() const {
   return pipe->get_shader_param(pipe, PIPE_SHADER_COMPUTE,
                                 PIPE_SHADER_CAP_INT64_ATOMICS);
}

bool
device::has_unified_memory() const {
   return pipe->get_param(pipe, PIPE_CAP_UMA);
}

cl_uint
device::mem_base_addr_align() const {
   return sysconf(_SC_PAGESIZE);
}

std::vector<size_t>
device::max_block_size() const {
   auto v = get_compute_param<uint64_t>(pipe, ir_format(),
                                        PIPE_COMPUTE_CAP_MAX_BLOCK_SIZE);
   return { v.begin(), v.end() };
}

cl_uint
device::subgroup_size() const {
   return get_compute_param<uint32_t>(pipe, ir_format(),
                                      PIPE_COMPUTE_CAP_SUBGROUP_SIZE)[0];
}

cl_uint
device::address_bits() const {
   return get_compute_param<uint32_t>(pipe, ir_format(),
                                      PIPE_COMPUTE_CAP_ADDRESS_BITS)[0];
}

std::string
device::device_name() const {
   return pipe->get_name(pipe);
}

std::string
device::vendor_name() const {
   return pipe->get_device_vendor(pipe);
}

enum pipe_shader_ir
device::ir_format() const {
   int supported_irs =
      pipe->get_shader_param(pipe, PIPE_SHADER_COMPUTE,
                             PIPE_SHADER_CAP_SUPPORTED_IRS);

   if (supported_irs & (1 << PIPE_SHADER_IR_NATIVE)) {
      return PIPE_SHADER_IR_NATIVE;
   }

   return PIPE_SHADER_IR_TGSI;
}

std::string
device::ir_target() const {
   std::vector<char> target = get_compute_param<char>(
      pipe, ir_format(), PIPE_COMPUTE_CAP_IR_TARGET);
   return { target.data() };
}

enum pipe_endian
device::endianness() const {
   return (enum pipe_endian)pipe->get_param(pipe, PIPE_CAP_ENDIANNESS);
}

std::string
device::device_version() const {
   static const std::string device_version =
         debug_get_option("CLOVER_DEVICE_VERSION_OVERRIDE", "1.1");
   return device_version;
}

std::string
device::device_clc_version() const {
   static const std::string device_clc_version =
         debug_get_option("CLOVER_DEVICE_CLC_VERSION_OVERRIDE", "1.1");
   return device_clc_version;
}
