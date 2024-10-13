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

files_r600 = files(
  'r600d_common.h',
  'compute_memory_pool.c',
  'compute_memory_pool.h',
  'eg_asm.c',
  'eg_debug.c',
  'eg_sq.h',
  'evergreen_compute.c',
  'evergreen_compute.h',
  'evergreen_compute_internal.h',
  'evergreend.h',
  'evergreen_hw_context.c',
  'evergreen_state.c',
  'r600_asm.c',
  'r600_asm.h',
  'r600_blit.c',
  'r600d.h',
  'r600_formats.h',
  'r600_hw_context.c',
  'r600_isa.c',
  'r600_isa.h',
  'r600_opcodes.h',
  'r600_pipe.c',
  'r600_pipe.h',
  'r600_public.h',
  'r600_shader.c',
  'r600_shader.h',
  'r600_sq.h',
  'r600_state.c',
  'r600_state_common.c',
  'r600_uvd.c',
  'r700_asm.c',
  'r700_sq.h',
  'cayman_msaa.c',
  'r600_buffer_common.c',
  'r600_cs.h',
  'r600_gpu_load.c',
  'r600_perfcounter.c',
  'r600_pipe_common.c',
  'r600_pipe_common.h',
  'r600_query.c',
  'r600_query.h',
  'r600_streamout.c',
  'r600_test_dma.c',
  'r600_texture.c',
  'r600_viewport.c',
  'radeon_uvd.c',
  'radeon_uvd.h',
  'radeon_vce.c',
  'radeon_vce.h',
  'radeon_video.c',
  'radeon_video.h',
  'sb/sb_bc_builder.cpp',
  'sb/sb_bc_decoder.cpp',
  'sb/sb_bc_dump.cpp',
  'sb/sb_bc_finalize.cpp',
  'sb/sb_bc.h',
  'sb/sb_bc_parser.cpp',
  'sb/sb_context.cpp',
  'sb/sb_core.cpp',
  'sb/sb_dce_cleanup.cpp',
  'sb/sb_def_use.cpp',
  'sb/sb_dump.cpp',
  'sb/sb_expr.cpp',
  'sb/sb_expr.h',
  'sb/sb_gcm.cpp',
  'sb/sb_gvn.cpp',
  'sb/sb_if_conversion.cpp',
  'sb/sb_ir.cpp',
  'sb/sb_ir.h',
  'sb/sb_liveness.cpp',
  'sb/sb_pass.cpp',
  'sb/sb_pass.h',
  'sb/sb_peephole.cpp',
  'sb/sb_psi_ops.cpp',
  'sb/sb_public.h',
  'sb/sb_ra_checker.cpp',
  'sb/sb_ra_coalesce.cpp',
  'sb/sb_ra_init.cpp',
  'sb/sb_sched.cpp',
  'sb/sb_sched.h',
  'sb/sb_shader.cpp',
  'sb/sb_shader.h',
  'sb/sb_ssa_builder.cpp',
  'sb/sb_valtable.cpp',
)

egd_tables_h = custom_target(
  'egd_tables.h',
  input : ['egd_tables.py', 'evergreend.h'],
  output : 'egd_tables.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
)

r600_c_args = []
if with_gallium_opencl
  r600_c_args += '-DHAVE_OPENCL'
endif

libr600 = static_library(
  'r600',
  [files_r600, egd_tables_h],
  c_args : [c_vis_args, r600_c_args],
  cpp_args : [cpp_vis_args],
  include_directories : [
    inc_src, inc_include, inc_gallium, inc_gallium_aux, inc_amd_common,
    inc_gallium_drivers,
  ],
  dependencies: [dep_libdrm_radeon, dep_elf, dep_llvm],
)

driver_r600 = declare_dependency(
  compile_args : '-DGALLIUM_R600',
  link_with : [libr600, libradeonwinsys],
)
