# Copyright © 2016 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

include Makefile.sources

# The gallium includes are for the util/u_math.h include from main/macros.h
AM_CPPFLAGS = \
	-I$(top_srcdir)/include \
	-I$(top_builddir)/src \
	-I$(top_srcdir)/src \
	-I$(top_builddir)/src/compiler/glsl \
	-I$(top_srcdir)/src/compiler/glsl \
	-I$(top_builddir)/src/compiler/nir \
	-I$(top_srcdir)/src/compiler/nir \
	-I$(top_srcdir)/src/gtest/include \
	-I$(top_builddir)/src/intel \
	-I$(top_srcdir)/src/intel \
	-I$(top_srcdir)/include/drm-uapi \
	-I$(top_srcdir)/src/mapi \
	-I$(top_srcdir)/src/mesa \
	-I$(top_srcdir)/src/gallium/auxiliary \
	-I$(top_srcdir)/src/gallium/include \
	$(VALGRIND_CFLAGS) \
	$(LIBDRM_CFLAGS) \
	$(DEFINES)

AM_CFLAGS = \
	$(VISIBILITY_CFLAGS) \
	$(WNO_OVERRIDE_INIT)

AM_CXXFLAGS = \
	$(VISIBILITY_CXXFLAGS)

MKDIR_GEN = $(AM_V_at)$(MKDIR_P) $(@D)
PYTHON_GEN = $(AM_V_GEN)$(PYTHON) $(PYTHON_FLAGS)

lib_LTLIBRARIES =
check_LTLIBRARIES =
noinst_DATA =
noinst_HEADERS =
noinst_LTLIBRARIES =
noinst_PROGRAMS =
check_PROGRAMS =
TESTS =
BUILT_SOURCES =
CLEANFILES =
EXTRA_DIST = \
	blorp/meson.build \
	common/meson.build \
	compiler/meson.build \
	dev/meson.build \
	genxml/meson.build \
	isl/meson.build \
	tools/aubinator_viewer.cpp \
	tools/aubinator_viewer.h \
	tools/aubinator_viewer_decoder.cpp \
	tools/aubinator_viewer_urb.h \
	tools/imgui/README \
	tools/imgui/stb_textedit.h \
	tools/imgui/meson.build \
	tools/imgui/imconfig.h \
	tools/imgui/imgui.cpp \
	tools/imgui/imgui.h \
	tools/imgui/imgui_draw.cpp \
	tools/imgui/imgui_demo.cpp \
	tools/imgui/imgui_internal.h \
	tools/imgui/imgui_impl_gtk3.cpp \
	tools/imgui/imgui_impl_gtk3.h \
	tools/imgui/imgui_impl_opengl3.cpp \
	tools/imgui/imgui_impl_opengl3.h \
	tools/imgui/imgui_memory_editor.h \
	tools/imgui/stb_truetype.h \
	tools/imgui/stb_rect_pack.h \
	tools/imgui/LICENSE.txt \
	tools/intel_sanitize_gpu.c \
	tools/intel_sanitize_gpu.in \
	tools/intel_dump_gpu.c \
	tools/intel_dump_gpu.in \
	tools/meson.build \
	vulkan/meson.build \
	meson.build

include Makefile.blorp.am
include Makefile.common.am
include Makefile.compiler.am
include Makefile.dev.am
include Makefile.genxml.am
include Makefile.isl.am
include Makefile.tools.am

include Makefile.vulkan.am

CLEANFILES += $(BUILT_SOURCES)
