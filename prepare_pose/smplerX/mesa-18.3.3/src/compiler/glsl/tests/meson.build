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

test(
  'blob_test',
  executable(
    'blob_test',
    'blob_test.c',
    c_args : [c_vis_args, c_msvc_compat_args, no_override_init_args],
    include_directories : [inc_common, inc_compiler],
    link_with : [libglsl],
  )
)

test(
  'cache_test',
  executable(
    'cache_test',
    'cache_test.c',
    c_args : [c_vis_args, c_msvc_compat_args, no_override_init_args],
    include_directories : [inc_common, inc_glsl],
    link_with : [libglsl],
    dependencies : [dep_clock, dep_thread],
  )
)


test(
  'general_ir_test',
  executable(
    'general_ir_test',
    ['array_refcount_test.cpp', 'builtin_variable_test.cpp',
     'invalidate_locations_test.cpp', 'general_ir_test.cpp',
     'lower_int64_test.cpp', 'opt_add_neg_to_sub_test.cpp',
     'varyings_test.cpp', ir_expression_operation_h],
    cpp_args : [cpp_vis_args, cpp_msvc_compat_args],
    include_directories : [inc_common, inc_glsl],
    link_with : [libglsl, libglsl_standalone, libglsl_util],
    dependencies : [dep_clock, dep_thread, idep_gtest],
  )
)

test(
  'uniform_initializer_test',
  executable(
    'uniform_initializer_test',
    ['copy_constant_to_storage_tests.cpp', 'set_uniform_initializer_tests.cpp',
     'uniform_initializer_utils.cpp', 'uniform_initializer_utils.h',
     ir_expression_operation_h],
    cpp_args : [cpp_vis_args, cpp_msvc_compat_args],
    include_directories : [inc_common, inc_glsl],
    link_with : [libglsl, libglsl_util],
    dependencies : [dep_thread, idep_gtest],
  )
)

test(
  'sampler_types_test',
  executable(
    'sampler_types_test',
    ['sampler_types_test.cpp', ir_expression_operation_h],
    cpp_args : [cpp_vis_args, cpp_msvc_compat_args],
    include_directories : [inc_common, inc_glsl],
    link_with : [libglsl, libglsl_util],
    dependencies : [dep_thread, idep_gtest],
  )
)

test(
  'glsl compiler warnings',
  prog_python,
  args : [
    join_paths(meson.current_source_dir(), 'warnings_test.py'),
    '--glsl-compiler', glsl_compiler,
    '--test-directory', join_paths(
      meson.source_root(), 'src', 'compiler', 'glsl', 'tests', 'warnings'
    ),
  ],
)
test(
  'glsl optimization',
  prog_python,
  args : [
    join_paths(meson.current_source_dir(), 'optimization_test.py'),
    '--test-runner', glsl_test
  ],
)
