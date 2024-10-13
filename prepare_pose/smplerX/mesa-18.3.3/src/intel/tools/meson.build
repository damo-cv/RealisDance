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

libaub = static_library(
  'aub',
  files('aub_read.c', 'aub_mem.c'),
  include_directories : [inc_common, inc_intel],
  link_with : [libintel_common, libintel_dev, libmesa_util],
  c_args : [c_vis_args, no_override_init_args],
  install : false
)

aubinator = executable(
  'aubinator',
  files('aubinator.c'),
  dependencies : [dep_expat, dep_zlib, dep_dl, dep_thread, dep_m],
  include_directories : [inc_common, inc_intel],
  link_with : [libintel_common, libintel_compiler, libintel_dev, libmesa_util, libaub],
  c_args : [c_vis_args, no_override_init_args],
  build_by_default : true,
  install : true
)

aubinator_error_decode = executable(
  'aubinator_error_decode',
  files('aubinator_error_decode.c'),
  dependencies : [dep_zlib, dep_thread],
  include_directories : [inc_common, inc_intel],
  link_with : [libintel_common, libintel_compiler, libintel_dev, libmesa_util],
  c_args : [c_vis_args, no_override_init_args],
  build_by_default : true,
  install : true
)

i965_disasm = executable(
  'i965_disasm',
  files('i965_disasm.c'),
  dependencies : [dep_thread],
  include_directories : [inc_common, inc_intel],
  link_with : [libintel_common, libintel_compiler, libintel_dev, libmesa_util],
  c_args : [c_vis_args, no_override_init_args],
  build_by_default : true,
  install : true
)

error2aub = executable(
  'intel_error2aub',
  files('aub_write.h', 'aub_write.c', 'error2aub.c'),
  dependencies : [dep_zlib, dep_dl, dep_thread, dep_m],
  include_directories : [inc_common, inc_intel, inc_drm_uapi],
  link_with : [libintel_dev],
  c_args : [c_vis_args, no_override_init_args],
  build_by_default : true,
  install : true
)

sanitize_data = configuration_data()
sanitize_data.set(
  'install_libexecdir',
  join_paths(get_option('prefix'), get_option('libexecdir'))
)
configure_file(
  input: 'intel_sanitize_gpu.in',
  output: '@BASENAME@',
  install_dir: get_option('bindir'),
  install: true,
  configuration: sanitize_data
)

libintel_sanitize_gpu = shared_library(
  'intel_sanitize_gpu',
  files('intel_sanitize_gpu.c'),
  dependencies : [dep_dl, dep_thread],
  include_directories : [inc_common, inc_intel, inc_drm_uapi],
  link_with : [libintel_common, libmesa_util],
  c_args : [c_vis_args, no_override_init_args, c_sse2_args],
  build_by_default : true,
  install_dir: get_option('libexecdir'),
  install: true
)

configure_file(
  input : 'intel_dump_gpu.in',
  output : '@BASENAME@',
  install_dir: get_option('bindir'),
  configuration: sanitize_data,
  install: true
)

libintel_dump_gpu = shared_library(
  'intel_dump_gpu',
  files('gen_context.h', 'gen8_context.h', 'gen10_context.h',
        'intel_aub.h', 'aub_write.h', 'aub_write.c', 'intel_dump_gpu.c'),
  dependencies : dep_dl,
  include_directories : [inc_common, inc_intel, inc_drm_uapi],
  link_with : libintel_dev,
  c_args : [c_vis_args, no_override_init_args],
  build_by_default : true,
  install_dir: get_option('libexecdir'),
  install: true
)

if with_tools.contains('intel-ui')
  subdir('imgui')

  aubinator_viewer = executable(
    'aubinator_viewer',
    files('aubinator_viewer.cpp', 'aubinator_viewer_decoder.cpp'),
    dependencies : [dep_zlib, dep_dl, dep_thread, dep_m, libintel_imgui_gtk_dep],
    include_directories : [inc_common, inc_intel],
    link_with : [libintel_common, libintel_compiler, libintel_dev, libmesa_util, libaub],
    c_args : [c_vis_args, no_override_init_args],
    cpp_args : [ '-fpermissive', '-std=c++11', '-Wno-parentheses', '-Wno-class-memaccess' ],
    build_by_default : true,
    install : true
  )
endif
