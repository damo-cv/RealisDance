# Copyright © 2012 Intel Corporation
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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

if HAVE_SHARED_GLAPI
SHARED_GLAPI_LIB = $(top_builddir)/src/mapi/shared-glapi/libglapi.la
endif

AM_CPPFLAGS = \
	-I$(top_srcdir)/include \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/src/mapi \
	-I$(top_srcdir)/src/mesa \
	-I$(top_srcdir)/src/mesa/main \
	-I$(top_srcdir)/src/gallium/include \
	-I$(top_srcdir)/src/gallium/drivers \
	-I$(top_srcdir)/src/gallium/state_trackers/glx/xlib \
	-I$(top_srcdir)/src/gallium/auxiliary \
	-I$(top_srcdir)/src/gallium/winsys \
	-DGALLIUM_SOFTPIPE

AM_CFLAGS = $(X11_INCLUDES)

lib_LTLIBRARIES = lib@GL_LIB@.la

nodist_EXTRA_lib@GL_LIB@_la_SOURCES = dummy.cpp
lib@GL_LIB@_la_SOURCES = xlib.c
lib@GL_LIB@_la_LDFLAGS = \
	-no-undefined \
	-version-number 1:5:0 \
	$(BSYMBOLIC) \
	$(GC_SECTIONS) \
	$(LD_NO_UNDEFINED)

if HAVE_LD_VERSION_SCRIPT
lib@GL_LIB@_la_LDFLAGS += \
	-Wl,--version-script=$(top_srcdir)/src/gallium/targets/libgl-xlib/libgl-xlib.sym
endif

lib@GL_LIB@_la_LIBADD = \
	$(top_builddir)/src/gallium/state_trackers/glx/xlib/libxlib.la \
	$(top_builddir)/src/gallium/winsys/sw/xlib/libws_xlib.la \
	$(top_builddir)/src/gallium/drivers/softpipe/libsoftpipe.la \
	$(top_builddir)/src/mapi/glapi/libglapi.la \
	$(top_builddir)/src/mesa/libmesagallium.la \
	$(top_builddir)/src/gallium/auxiliary/libgallium.la \
	$(SHARED_GLAPI_LIB) \
	$(GL_LIB_DEPS) \
	$(CLOCK_LIB) \
	$(LIBUNWIND_LIBS)

if HAVE_GALLIUM_LLVM
lib@GL_LIB@_la_LIBADD += $(top_builddir)/src/gallium/drivers/llvmpipe/libllvmpipe.la $(LLVM_LIBS)
AM_CPPFLAGS += -DGALLIUM_LLVMPIPE
lib@GL_LIB@_la_LDFLAGS += $(LLVM_LDFLAGS)
endif

if HAVE_GALLIUM_SWR
lib@GL_LIB@_la_LIBADD += $(top_builddir)/src/gallium/drivers/swr/libmesaswr.la $(LLVM_LIBS)
AM_CPPFLAGS += -DGALLIUM_SWR
endif

EXTRA_lib@GL_LIB@_la_DEPENDENCIES = libgl-xlib.sym
EXTRA_DIST = SConscript libgl-xlib.sym meson.build

include $(top_srcdir)/install-gallium-links.mk
