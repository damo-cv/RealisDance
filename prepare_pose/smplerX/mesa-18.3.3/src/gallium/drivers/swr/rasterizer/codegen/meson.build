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

gen_knobs_cpp = custom_target(
  'gen_knobs.cpp',
  input : ['gen_knobs.py'],
  output : 'gen_knobs.cpp',
  command : [prog_python, '@INPUT0@', '--output', '@OUTPUT@', '--gen_cpp'],
  depend_files : files(
    'knob_defs.py', 'gen_common.py',
    'templates/gen_knobs.cpp',
  ),
)

gen_knobs_h = custom_target(
  'gen_knobs.h',
  input : ['gen_knobs.py'],
  output : 'gen_knobs.h',
  command : [prog_python, '@INPUT0@', '--output', '@OUTPUT@', '--gen_h'],
  depend_files : files(
    'knob_defs.py', 'gen_common.py',
    'templates/gen_knobs.h',
  ),
)


# The generators above this are needed individually, while the below generators
# are all inputs to the same lib, so they don't need unique names.
files_swr_common += [
  gen_builder_hpp, gen_builder_meta_hpp, gen_knobs_h, gen_knobs_cpp
]

foreach x : [[swr_context_files, 'gen_swr_context_llvm.h'],
             [swr_state_files, 'gen_state_llvm.h']]
  files_swr_common += custom_target(
    x[1],
    input : ['gen_llvm_types.py', x[0]],
    output : x[1],
    command : [prog_python, '@INPUT0@', '--input', '@INPUT1@', '--output', '@OUTPUT@'],
    depend_files : files(
      'templates/gen_llvm.hpp',
      'gen_common.py',
    ),
  )
endforeach

foreach x : [['gen_ar_event.hpp', '--gen_event_hpp'],
             ['gen_ar_event.cpp', '--gen_event_cpp'],
             ['gen_ar_eventhandler.hpp', '--gen_eventhandler_h'],
             ['gen_ar_eventhandlerfile.hpp', '--gen_eventhandlerfile_h']]
  files_swr_common += custom_target(
    x[0],
    input : ['gen_archrast.py', swr_event_proto_files, swr_event_pproto_files],
    output : x[0],
    command : [
      prog_python,
      '@INPUT0@',
      '--proto', '@INPUT1@',
      '--proto_private', '@INPUT2@',
      '--output', '@OUTPUT@',
      x[1],
    ],
    depend_files : files(
      join_paths('templates', x[0]),
      'gen_common.py',
    ),
  )
endforeach

