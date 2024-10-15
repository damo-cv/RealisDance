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

files_mapi_glapi = files(
  '../entry.c',
  '../mapi_glapi.c',
  '../stub.c',
  '../stub.h',
  '../table.c',
  '../table.h',
)

shared_glapi_mapi_tmp_h = custom_target(
  'shared_glapi_mapi_tmp.h',
  input : [mapi_abi_py, gl_and_es_api_files],
  output : 'glapi_mapi_tmp.h',
  command : [prog_python, '@INPUT0@', '--printer', 'shared-glapi', '@INPUT1@'],
  depend_files : api_xml_files,
  capture : true,
)

libglapi = shared_library(
  'glapi',
  [files_mapi_glapi, files_mapi_util, shared_glapi_mapi_tmp_h],
  c_args : [
    c_msvc_compat_args, c_vis_args, '-DMAPI_MODE_GLAPI',
    '-DMAPI_ABI_HEADER="@0@"'.format(shared_glapi_mapi_tmp_h.full_path()),
  ],
  link_args : [ld_args_gc_sections],
  include_directories : [inc_src, inc_include, inc_mapi],
  dependencies : [dep_thread, dep_selinux],
  version : '0.0.0',
  install : true,
)

if with_any_opengl and with_tests
  test(
    'shared-glapi-test',
    executable(
      ['shared-glapi-test', glapitable_h],
      'tests/check_table.cpp',
      cpp_args : [cpp_msvc_compat_args],
      include_directories : [inc_src, inc_include, inc_mapi],
      link_with : [libglapi],
      dependencies : [dep_thread, idep_gtest],
    )
  )
endif
