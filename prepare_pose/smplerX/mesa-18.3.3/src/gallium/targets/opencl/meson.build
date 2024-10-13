# Copyright © 2017 Intel Corporation

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

opencl_link_args = []
opencl_link_deps = []
opencl_version = '1'

if with_ld_version_script
  opencl_link_args += [
    '-Wl,--version-script', join_paths(meson.current_source_dir(), 'opencl.sym')
  ]
  opencl_link_deps += files('opencl.sym')
endif

llvm_libdir = dep_llvm.get_configtool_variable('libdir')

opencl_libname = with_opencl_icd ? 'MesaOpenCL' : 'OpenCL'

libopencl = shared_library(
  opencl_libname,
  [],
  link_args : [ld_args_gc_sections, opencl_link_args],
  link_depends : opencl_link_deps,
  link_whole : libclover,
  link_with : [libpipe_loader_dynamic, libgallium, libmesa_util],
  dependencies : [
    dep_thread, dep_clock, dep_dl, dep_unwind, dep_elf, dep_expat,
    cpp.find_library('clangCodeGen', dirs : llvm_libdir),
    cpp.find_library('clangFrontendTool', dirs : llvm_libdir),
    cpp.find_library('clangFrontend', dirs : llvm_libdir),
    cpp.find_library('clangDriver', dirs : llvm_libdir),
    cpp.find_library('clangSerialization', dirs : llvm_libdir),
    cpp.find_library('clangParse', dirs : llvm_libdir),
    cpp.find_library('clangSema', dirs : llvm_libdir),
    cpp.find_library('clangAnalysis', dirs : llvm_libdir),
    cpp.find_library('clangAST', dirs : llvm_libdir),
    cpp.find_library('clangEdit', dirs : llvm_libdir),
    cpp.find_library('clangLex', dirs : llvm_libdir),
    cpp.find_library('clangBasic', dirs : llvm_libdir),
  ],
  version : '@0@.0.0'.format(opencl_version),
  install : true,
)

if with_opencl_icd
  _config = configuration_data()
  _config.set('OPENCL_LIBNAME', 'MesaOpenCL')
  _config.set('OPENCL_VERSION', opencl_version)
  configure_file(
    configuration : _config,
    input : 'mesa.icd.in',
    output : 'mesa.icd',
    install : true,
    install_dir : join_paths(get_option('sysconfdir'), 'OpenCL', 'vendors'),
  )
endif
