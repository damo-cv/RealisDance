# Copyright © 2017 Dylan Baker

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

files_libradeonsi = files(
  'cik_sdma.c',
  'driinfo_radeonsi.h',
  'si_blit.c',
  'si_buffer.c',
  'si_build_pm4.h',
  'si_clear.c',
  'si_compute.c',
  'si_compute.h',
  'si_compute_blit.c',
  'si_cp_dma.c',
  'si_debug.c',
  'si_descriptors.c',
  'si_dma.c',
  'si_dma_cs.c',
  'si_fence.c',
  'si_get.c',
  'si_gfx_cs.c',
  'si_gpu_load.c',
  'si_perfcounter.c',
  'si_pipe.c',
  'si_pipe.h',
  'si_pm4.c',
  'si_pm4.h',
  'si_public.h',
  'si_query.c',
  'si_query.h',
  'si_shader.c',
  'si_shader.h',
  'si_shader_internal.h',
  'si_shader_nir.c',
  'si_shader_tgsi_alu.c',
  'si_shader_tgsi_mem.c',
  'si_shader_tgsi_setup.c',
  'si_shaderlib_tgsi.c',
  'si_state.c',
  'si_state.h',
  'si_state_binning.c',
  'si_state_draw.c',
  'si_state_msaa.c',
  'si_state_shaders.c',
  'si_state_streamout.c',
  'si_state_viewport.c',
  'si_test_dma.c',
  'si_test_dma_perf.c',
  'si_texture.c',
  'si_uvd.c',
  '../radeon/r600_perfcounter.c',
  '../radeon/radeon_uvd.c',
  '../radeon/radeon_uvd.h',
  '../radeon/radeon_vcn_enc_1_2.c',
  '../radeon/radeon_vcn_enc.c',
  '../radeon/radeon_vcn_enc.h',
  '../radeon/radeon_vcn_dec_jpeg.c',
  '../radeon/radeon_vcn_dec.c',
  '../radeon/radeon_vcn_dec.h',
  '../radeon/radeon_uvd_enc_1_1.c',
  '../radeon/radeon_uvd_enc.c',
  '../radeon/radeon_uvd_enc.h',
  '../radeon/radeon_vce_40_2_2.c',
  '../radeon/radeon_vce_50.c',
  '../radeon/radeon_vce_52.c',
  '../radeon/radeon_vce.c',
  '../radeon/radeon_vce.h',
  '../radeon/radeon_video.c',
  '../radeon/radeon_video.h',
  '../radeon/radeon_winsys.h',
)

si_driinfo_h = custom_target(
  'si_driinfo.h',
  input : files(
    '../../../util/merge_driinfo.py',
    '../../auxiliary/pipe-loader/driinfo_gallium.h', 'driinfo_radeonsi.h'
  ),
  output : 'si_driinfo.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
)

libradeonsi = static_library(
  'radeonsi',
  [files_libradeonsi, si_driinfo_h, sid_tables_h],
  include_directories : [
    inc_src, inc_include, inc_gallium, inc_gallium_aux, inc_amd_common,
    inc_gallium_drivers,
  ],
  c_args : ['-Wstrict-overflow=0', c_vis_args],
  cpp_args : [cpp_vis_args],
  dependencies : [dep_llvm, dep_clock, dep_libdrm_radeon, idep_nir_headers],
)

driver_radeonsi = declare_dependency(
  compile_args : '-DGALLIUM_RADEONSI',
  sources : si_driinfo_h,
  link_with : [
    libradeonsi, libradeonwinsys, libamdgpuwinsys, libamd_common,
  ],
  dependencies : idep_nir,
)
