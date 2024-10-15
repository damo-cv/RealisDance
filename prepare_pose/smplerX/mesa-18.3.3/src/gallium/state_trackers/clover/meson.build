# Copyright © 2017-2018 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

clover_cpp_args = []
clover_incs = [inc_include, inc_src, inc_gallium, inc_gallium_aux]

if with_opencl_icd
  clover_cpp_args += '-DHAVE_CLOVER_ICD'
endif

libcltgsi = static_library(
  'cltgsi',
  files('tgsi/compiler.cpp', 'tgsi/invocation.hpp'),
  include_directories : clover_incs,
  cpp_args : [cpp_vis_args],
  override_options : clover_cpp_std,
)

libclllvm = static_library(
  'clllvm',
  files(
    'llvm/codegen/bitcode.cpp',
    'llvm/codegen/common.cpp',
    'llvm/codegen/native.cpp',
    'llvm/codegen.hpp',
    'llvm/compat.hpp',
    'llvm/invocation.cpp',
    'llvm/invocation.hpp',
    'llvm/metadata.hpp',
    'llvm/util.hpp',
  ),
  include_directories : clover_incs,
  cpp_args : [
    cpp_vis_args,
    '-DLIBCLC_INCLUDEDIR="@0@/"'.format(dep_clc.get_pkgconfig_variable('includedir')),
    '-DLIBCLC_LIBEXECDIR="@0@/"'.format(dep_clc.get_pkgconfig_variable('libexecdir')),
    '-DCLANG_RESOURCE_DIR="@0@"'.format(join_paths(
      dep_llvm.get_configtool_variable('libdir'), 'clang',
      dep_llvm.get_configtool_variable('version'), 'include',
    )),
  ],
  dependencies : [dep_llvm, dep_elf],
  override_options : clover_cpp_std,
)

clover_files = files(
  'api/context.cpp',
  'api/device.cpp',
  'api/dispatch.cpp',
  'api/dispatch.hpp',
  'api/event.cpp',
  'api/interop.cpp',
  'api/kernel.cpp',
  'api/memory.cpp',
  'api/platform.cpp',
  'api/program.cpp',
  'api/queue.cpp',
  'api/sampler.cpp',
  'api/transfer.cpp',
  'api/util.hpp',
  'core/context.cpp',
  'core/context.hpp',
  'core/device.cpp',
  'core/device.hpp',
  'core/error.hpp',
  'core/event.cpp',
  'core/event.hpp',
  'core/format.cpp',
  'core/format.hpp',
  'core/kernel.cpp',
  'core/kernel.hpp',
  'core/memory.cpp',
  'core/memory.hpp',
  'core/module.cpp',
  'core/module.hpp',
  'core/object.hpp',
  'core/platform.cpp',
  'core/platform.hpp',
  'core/program.cpp',
  'core/program.hpp',
  'core/property.hpp',
  'core/queue.cpp',
  'core/queue.hpp',
  'core/resource.cpp',
  'core/resource.hpp',
  'core/sampler.cpp',
  'core/sampler.hpp',
  'core/timestamp.cpp',
  'core/timestamp.hpp',
  'util/adaptor.hpp',
  'util/algebra.hpp',
  'util/algorithm.hpp',
  'util/factor.hpp',
  'util/functional.hpp',
  'util/lazy.hpp',
  'util/pointer.hpp',
  'util/range.hpp',
  'util/tuple.hpp',
)

libclover = static_library(
  'clover',
  [clover_files, sha1_h],
  include_directories : clover_incs,
  cpp_args : [clover_cpp_args, cpp_vis_args],
  link_with : [libcltgsi, libclllvm],
  override_options : clover_cpp_std,
)
