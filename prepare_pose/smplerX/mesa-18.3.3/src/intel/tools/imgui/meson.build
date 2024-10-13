libintel_imgui_core = static_library(
  'intel_imgui_core',
  files('imgui.cpp', 'imgui_demo.cpp', 'imgui_draw.cpp'),
  cpp_args : ['-w'],
  install: false
)

libintel_imgui_gtk_deps = [
  dependency('epoxy'),
  dependency('gtk+-3.0'),
]

libintel_imgui_gtk = static_library(
  'intel_imgui_gtk',
  files('imgui_impl_gtk3.cpp', 'imgui_impl_opengl3.cpp'),
  dependencies : libintel_imgui_gtk_deps,
  cpp_args : ['-w'],
  install: false
)

libintel_imgui_gtk_dep = declare_dependency(
  link_with: [libintel_imgui_core, libintel_imgui_gtk],
  dependencies: libintel_imgui_gtk_deps,
  include_directories: include_directories('.')
)
