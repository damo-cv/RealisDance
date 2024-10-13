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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


SUBDIRS =

EXTRA_DIST = SConscript meson.build

AM_CFLAGS = \
	-I$(top_srcdir)/include \
	-I$(top_srcdir)/include/GL/internal \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/src/loader \
	-I$(top_builddir)/src/mapi \
	-I$(top_srcdir)/src/mapi \
	-I$(top_builddir)/src/mapi/glapi \
	-I$(top_srcdir)/src/mapi/glapi \
	$(VISIBILITY_CFLAGS) \
	-D_REENTRANT \
	-DDEFAULT_DRIVER_DIR=\"$(DRI_DRIVER_SEARCH_DIR)\" \
	$(DEFINES) \
	$(LIBDRM_CFLAGS) \
	$(DRI2PROTO_CFLAGS) \
	$(GLPROTO_CFLAGS) \
	$(X11_INCLUDES)

lib_LTLIBRARIES = lib@GL_LIB@.la

noinst_LTLIBRARIES = libglx.la

libglx_la_SOURCES = \
	clientattrib.c \
	clientinfo.c \
	compsize.c \
	create_context.c \
	eval.c \
	glxclient.h \
	glxcmds.c \
	glxconfig.c \
	glxconfig.h \
	glxcurrent.c \
	glx_error.c \
	glx_error.h \
	glxext.c \
	glxextensions.c \
	glxextensions.h \
	glxhash.c \
	glxhash.h \
	glx_pbuffer.c \
	glx_query.c \
	indirect.c \
	indirect_glx.c \
	indirect_init.c \
	indirect_init.h \
	indirect_size.c \
	indirect_texture_compression.c \
	indirect_transpose_matrix.c \
	indirect_vertex_array.c \
	indirect_vertex_array.h \
	indirect_vertex_array_priv.h \
	indirect_vertex_program.c \
	indirect_window_pos.c \
	packrender.h \
	packsingle.h \
	pixel.c \
	pixelstore.c \
	query_renderer.c \
	render2.c \
	renderpix.c \
	single2.c \
	singlepix.c \
	vertarr.c

libglx_la_LIBADD = \
	$(top_builddir)/src/loader/libloader.la \
	$(top_builddir)/src/util/libmesautil.la \
	$(top_builddir)/src/util/libxmlconfig.la

if HAVE_DRISW
libglx_la_SOURCES += \
	drisw_glx.c \
	drisw_priv.h
endif

if HAVE_DRICOMMON
libglx_la_SOURCES += \
	dri_common.c \
	dri_common.h \
	dri_common_query_renderer.c \
	dri_common_interop.c \
	xfont.c
endif

if HAVE_DRI2
libglx_la_SOURCES += \
	dri2.c \
	dri2_glx.c \
	dri2.h \
	dri2_priv.h \
	dri_glx.c \
	dri_sarea.h \
	XF86dri.c \
	xf86dri.h \
	xf86dristr.h
endif

if HAVE_DRI3
libglx_la_SOURCES += \
	dri3_glx.c \
	dri3_priv.h

libglx_la_LIBADD += $(top_builddir)/src/loader/libloader_dri3_helper.la
endif

if HAVE_APPLEDRI
libglx_la_SOURCES += \
	applegl_glx.c

SUBDIRS += apple
libglx_la_LIBADD += $(builddir)/apple/libappleglx.la
endif

if HAVE_WINDOWSDRI
libglx_la_SOURCES += \
	  driwindows_glx.c

SUBDIRS += windows
libglx_la_LIBADD += \
	  $(builddir)/windows/libwindowsdri.la \
	  $(builddir)/windows/libwindowsglx.la
endif

if USE_LIBGLVND
AM_CFLAGS += \
	-DGL_LIB_NAME=\"lib@GL_LIB@.so.0\" \
	$(GLVND_CFLAGS)

libglx_la_SOURCES += \
	g_glxglvnddispatchfuncs.c \
	g_glxglvnddispatchindices.h \
	glxglvnd.c \
	glxglvnd.h \
	glxglvnddispatchfuncs.h

GL_LIB_VERSION=0
else
AM_CFLAGS += \
	-DGL_LIB_NAME=\"lib@GL_LIB@.so.1\"
GL_LIB_VERSION=1:2
endif

# Push the libdrm furhter up, as due to libtool bug we might end up relinking
# against the system one.
GL_LIBS = \
	$(LIBDRM_LIBS) \
	libglx.la \
	$(top_builddir)/src/mapi/glapi/libglapi.la \
	$(top_builddir)/src/mapi/shared-glapi/libglapi.la \
	$(GL_LIB_DEPS)

GL_LDFLAGS = \
	-no-undefined \
	-version-number $(GL_LIB_VERSION) \
	$(BSYMBOLIC) \
	$(GC_SECTIONS) \
	$(LD_NO_UNDEFINED)

lib@GL_LIB@_la_SOURCES =
lib@GL_LIB@_la_LIBADD = $(GL_LIBS)
lib@GL_LIB@_la_LDFLAGS = $(GL_LDFLAGS)

if HAVE_WINDOWSDRI
lib@GL_LIB@_la_LDFLAGS += -lgdi32 -lopengl32 -Wl,--disable-stdcall-fixup
endif

SUBDIRS += . tests

include $(top_srcdir)/install-lib-links.mk
