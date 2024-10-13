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

files_i965 = files(
  'brw_binding_tables.c',
  'brw_blorp.c',
  'brw_blorp.h',
  'brw_bufmgr.c',
  'brw_bufmgr.h',
  'brw_clear.c',
  'brw_clip.c',
  'brw_compute.c',
  'brw_conditional_render.c',
  'brw_context.c',
  'brw_context.h',
  'brw_cs.c',
  'brw_cs.h',
  'brw_curbe.c',
  'brw_defines.h',
  'brw_disk_cache.c',
  'brw_draw.c',
  'brw_draw.h',
  'brw_draw_upload.c',
  'brw_ff_gs.c',
  'brw_ff_gs_emit.c',
  'brw_ff_gs.h',
  'brw_formatquery.c',
  'brw_generate_mipmap.c',
  'brw_gs.c',
  'brw_gs.h',
  'brw_gs_surface_state.c',
  'brw_link.cpp',
  'brw_meta_util.c',
  'brw_meta_util.h',
  'brw_misc_state.c',
  'brw_multisample_state.h',
  'brw_nir_uniforms.cpp',
  'brw_object_purgeable.c',
  'brw_pipe_control.c',
  'brw_performance_query.h',
  'brw_performance_query.c',
  'brw_performance_query_mdapi.c',
  'brw_performance_query_metrics.h',
  'brw_program.c',
  'brw_program.h',
  'brw_program_binary.c',
  'brw_program_cache.c',
  'brw_primitive_restart.c',
  'brw_queryobj.c',
  'brw_reset.c',
  'brw_sf.c',
  'brw_state.h',
  'brw_state_upload.c',
  'brw_structs.h',
  'brw_surface_formats.c',
  'brw_sync.c',
  'brw_tcs.c',
  'brw_tcs_surface_state.c',
  'brw_tes.c',
  'brw_tes_surface_state.c',
  'brw_urb.c',
  'brw_util.c',
  'brw_util.h',
  'brw_vs.c',
  'brw_vs.h',
  'brw_vs_surface_state.c',
  'brw_wm.c',
  'brw_wm.h',
  'brw_wm_surface_state.c',
  'gen4_blorp_exec.h',
  'gen6_clip_state.c',
  'gen6_constant_state.c',
  'gen6_multisample_state.c',
  'gen6_queryobj.c',
  'gen6_sampler_state.c',
  'gen6_sol.c',
  'gen6_urb.c',
  'gen7_l3_state.c',
  'gen7_sol_state.c',
  'gen7_urb.c',
  'gen8_depth_state.c',
  'gen8_multisample_state.c',
  'hsw_queryobj.c',
  'hsw_sol.c',
  'intel_batchbuffer.c',
  'intel_batchbuffer.h',
  'intel_blit.c',
  'intel_blit.h',
  'intel_buffer_objects.c',
  'intel_buffer_objects.h',
  'intel_buffers.c',
  'intel_buffers.h',
  'intel_copy_image.c',
  'intel_extensions.c',
  'intel_fbo.c',
  'intel_fbo.h',
  'intel_image.h',
  'intel_mipmap_tree.c',
  'intel_mipmap_tree.h',
  'intel_pixel_bitmap.c',
  'intel_pixel.c',
  'intel_pixel_copy.c',
  'intel_pixel_draw.c',
  'intel_pixel.h',
  'intel_pixel_read.c',
  'intel_screen.c',
  'intel_screen.h',
  'intel_state.c',
  'intel_tex.c',
  'intel_tex_copy.c',
  'intel_tex.h',
  'intel_tex_image.c',
  'intel_tex_obj.h',
  'intel_tex_validate.c',
  'intel_upload.c',
  'libdrm_macros.h',
)

files_intel_tiled_memcpy = files(
  'intel_tiled_memcpy_normal.c',
  'intel_tiled_memcpy.h',
)

files_intel_tiled_memcpy_sse41 = files(
  'intel_tiled_memcpy_sse41.c',
  'intel_tiled_memcpy_sse41.h',
)

i965_gen_libs = []
foreach v : ['40', '45', '50', '60', '70', '75', '80', '90', '100', '110']
  i965_gen_libs += static_library(
    'i965_gen@0@'.format(v),
    ['genX_blorp_exec.c', 'genX_state_upload.c', gen_xml_pack],
    include_directories : [inc_common, inc_intel, inc_dri_common],
    c_args : [
      c_vis_args, no_override_init_args, c_sse2_args,
      '-DGEN_VERSIONx10=@0@'.format(v),
    ],
    dependencies : [dep_libdrm, idep_nir_headers],
  )
endforeach


i965_hw_metrics = [
  'hsw',
  'bdw', 'chv',
  'sklgt2', 'sklgt3', 'sklgt4',
  'kblgt2', 'kblgt3',
  'cflgt2', 'cflgt3',
  'bxt', 'glk',
  'cnl',
]

i965_hw_metrics_xml_files = []
foreach hw : i965_hw_metrics
  i965_hw_metrics_xml_files += 'brw_oa_@0@.xml'.format(hw)
endforeach

i965_oa_sources = custom_target(
  'i965-oa-sources',
  input : i965_hw_metrics_xml_files,
  output : [ 'brw_oa_metrics.c', 'brw_oa_metrics.h' ],
  command : [
    prog_python, files('brw_oa.py'),
    '--code', '@OUTPUT0@', '--header', '@OUTPUT1@',
    '@INPUT@',
  ],
)

intel_tiled_memcpy = static_library(
  'intel_tiled_memcpy',
  [files_intel_tiled_memcpy],
  include_directories : [
    inc_common, inc_intel, inc_dri_common, inc_drm_uapi,
  ],
  c_args : [c_vis_args, no_override_init_args, '-msse2'],
  extra_files : ['intel_tiled_memcpy.c']
)

if with_sse41
  intel_tiled_memcpy_sse41 = static_library(
    'intel_tiled_memcpy_sse41',
    [files_intel_tiled_memcpy_sse41],
    include_directories : [
      inc_common, inc_intel, inc_dri_common, inc_drm_uapi,
    ],
    link_args : ['-Wl,--exclude-libs=ALL'],
    c_args : [c_vis_args, no_override_init_args, '-Wl,--exclude-libs=ALL', '-msse2', sse41_args],
    extra_files : ['intel_tiled_memcpy.c']
  )
else
  intel_tiled_memcpy_sse41 = []
endif


libi965 = static_library(
  'i965',
  [files_i965, i965_oa_sources, ir_expression_operation_h,
   xmlpool_options_h],
  include_directories : [
    inc_common, inc_intel, inc_dri_common, inc_util, inc_drm_uapi,
  ],
  c_args : [c_vis_args, no_override_init_args, c_sse2_args],
  cpp_args : [cpp_vis_args, c_sse2_args],
  link_with : [
    i965_gen_libs, libintel_common, libintel_dev, libisl, libintel_compiler,
    libblorp, intel_tiled_memcpy, intel_tiled_memcpy_sse41
  ],
  dependencies : [dep_libdrm, dep_valgrind, idep_nir_headers],
)

dri_drivers += libi965
dri_link += 'i965_dri.so'
