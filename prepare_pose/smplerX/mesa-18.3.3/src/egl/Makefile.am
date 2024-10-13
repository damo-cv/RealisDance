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

include Makefile.sources

MKDIR_GEN = $(AM_V_at)$(MKDIR_P) $(@D)

BUILT_SOURCES =

AM_CFLAGS = \
	-I$(top_srcdir)/include \
	-I$(top_srcdir)/include/drm-uapi \
	-I$(top_srcdir)/src/mapi \
	-I$(top_srcdir)/src/egl/main \
	-I$(top_srcdir)/src/gbm/main \
	-I$(top_srcdir)/src \
	$(DEFINES) \
	$(VISIBILITY_CFLAGS) \
	$(LIBDRM_CFLAGS) \
	$(EGL_CFLAGS) \
	-D_EGL_NATIVE_PLATFORM=$(EGL_NATIVE_PLATFORM)

# Depending on whether libglvnd is enabled, we'll build the EGL library as
# either libEGL.so.1 or libEGL_mesa.so.0. Using an automake substitution
# in the variable names breaks "make dist" target, so use a conenience library
# instead.
noinst_LTLIBRARIES = libEGL_common.la
libEGL_common_la_SOURCES = \
	$(LIBEGL_C_FILES)

libEGL_common_la_LIBADD = \
	$(top_builddir)/src/util/libmesautil.la \
	$(EGL_LIB_DEPS)

dri2_backend_FILES =
dri2_backend_GENERATED_FILES =
dri3_backend_FILES =

if HAVE_PLATFORM_X11
AM_CFLAGS += $(XCB_DRI2_CFLAGS)
libEGL_common_la_LIBADD += $(XCB_DRI2_LIBS)
dri2_backend_FILES += drivers/dri2/platform_x11.c

if HAVE_DRI3
dri3_backend_FILES += \
	drivers/dri2/platform_x11_dri3.c \
	drivers/dri2/platform_x11_dri3.h

libEGL_common_la_LIBADD += $(top_builddir)/src/loader/libloader_dri3_helper.la
endif
endif

WL_DMABUF_XML = $(WAYLAND_PROTOCOLS_DATADIR)/unstable/linux-dmabuf/linux-dmabuf-unstable-v1.xml

drivers/dri2/linux-dmabuf-unstable-v1-protocol.c: $(WL_DMABUF_XML)
	$(MKDIR_GEN)
	$(AM_V_GEN)$(WAYLAND_SCANNER) $(SCANNER_ARG) $< $@

drivers/dri2/linux-dmabuf-unstable-v1-client-protocol.h: $(WL_DMABUF_XML)
	$(MKDIR_GEN)
	$(AM_V_GEN)$(WAYLAND_SCANNER) client-header $< $@

if HAVE_PLATFORM_WAYLAND
drivers/dri2/linux-dmabuf-unstable-v1-protocol.lo: drivers/dri2/linux-dmabuf-unstable-v1-client-protocol.h
drivers/dri2/egl_dri2.lo: drivers/dri2/linux-dmabuf-unstable-v1-client-protocol.h
drivers/dri2/platform_wayland.lo: drivers/dri2/linux-dmabuf-unstable-v1-client-protocol.h

AM_CFLAGS += $(WAYLAND_CLIENT_CFLAGS)
libEGL_common_la_LIBADD += $(WAYLAND_CLIENT_LIBS)
libEGL_common_la_LIBADD += $(LIBDRM_LIBS)
AM_CFLAGS += $(WAYLAND_EGL_CFLAGS)
libEGL_common_la_LIBADD += $(WAYLAND_EGL_LIBS)
AM_CFLAGS += $(WAYLAND_SERVER_CFLAGS)
libEGL_common_la_LIBADD += $(top_builddir)/src/egl/wayland/wayland-drm/libwayland-drm.la
libEGL_common_la_LIBADD += $(WAYLAND_SERVER_LIBS)
dri2_backend_FILES += \
	drivers/dri2/platform_wayland.c
dri2_backend_GENERATED_FILES += \
	drivers/dri2/linux-dmabuf-unstable-v1-protocol.c \
	drivers/dri2/linux-dmabuf-unstable-v1-client-protocol.h
endif

if HAVE_PLATFORM_DRM
libEGL_common_la_LIBADD += $(top_builddir)/src/gbm/libgbm.la
dri2_backend_FILES += drivers/dri2/platform_drm.c
endif

if HAVE_PLATFORM_SURFACELESS
dri2_backend_FILES += drivers/dri2/platform_surfaceless.c
endif

if HAVE_PLATFORM_ANDROID
AM_CFLAGS += $(ANDROID_CFLAGS)
libEGL_common_la_LIBADD += $(ANDROID_LIBS)
dri2_backend_FILES += drivers/dri2/platform_android.c
endif

AM_CFLAGS += \
	-I$(top_srcdir)/src/loader \
	-I$(top_builddir)/src/egl/drivers/dri2 \
	-I$(top_srcdir)/src/egl/drivers/dri2 \
	-I$(top_srcdir)/src/gbm/backends/dri \
	-I$(top_builddir)/src/egl/wayland/wayland-drm \
	-I$(top_srcdir)/src/egl/wayland/wayland-drm \
	-DDEFAULT_DRIVER_DIR=\"$(DRI_DRIVER_SEARCH_DIR)\"

nodist_libEGL_common_la_SOURCES = \
	$(dri2_backend_GENERATED_FILES)

libEGL_common_la_SOURCES += \
	$(dri2_backend_core_FILES) \
	$(dri2_backend_FILES) \
	$(dri3_backend_FILES)

libEGL_common_la_LIBADD += \
	$(top_builddir)/src/loader/libloader.la \
	$(top_builddir)/src/util/libxmlconfig.la \
	$(DLOPEN_LIBS) \
	$(LIBDRM_LIBS) \
	$(CLOCK_LIB)

GLVND_GEN_DEPS = generate/gen_egl_dispatch.py \
	generate/egl.xml generate/eglFunctionList.py generate/genCommon.py \
	generate/egl_other.xml

PYTHON_GEN = $(AM_V_GEN)$(PYTHON) $(PYTHON_FLAGS)
g_egldispatchstubs.c: $(GLVND_GEN_DEPS)
	$(PYTHON_GEN) $(top_srcdir)/src/egl/generate/gen_egl_dispatch.py source \
		$(top_srcdir)/src/egl/generate/eglFunctionList.py \
		$(top_srcdir)/src/egl/generate/egl.xml \
		$(top_srcdir)/src/egl/generate/egl_other.xml > $@

g_egldispatchstubs.h: $(GLVND_GEN_DEPS)
	$(PYTHON_GEN) $(top_srcdir)/src/egl/generate/gen_egl_dispatch.py header \
		$(top_srcdir)/src/egl/generate/eglFunctionList.py \
		$(top_srcdir)/src/egl/generate/egl.xml \
		$(top_srcdir)/src/egl/generate/egl_other.xml > $@

BUILT_SOURCES += g_egldispatchstubs.c g_egldispatchstubs.h

CLEANFILES = \
	$(BUILT_SOURCES) \
	$(dri2_backend_GENERATED_FILES)

if USE_LIBGLVND
AM_CFLAGS += \
	$(GLVND_CFLAGS)

vendorjsondir = @LIBGLVND_DATADIR@/glvnd/egl_vendor.d
vendorjson_DATA = main/50_mesa.json

lib_LTLIBRARIES = libEGL_mesa.la
libEGL_mesa_la_SOURCES = \
	main/eglglvnd.c \
	main/egldispatchstubs.h \
	main/egldispatchstubs.c \
	g_egldispatchstubs.c \
	g_egldispatchstubs.h
libEGL_mesa_la_LIBADD = \
	libEGL_common.la \
	$(top_builddir)/src/mapi/shared-glapi/libglapi.la
libEGL_mesa_la_LDFLAGS = \
	-no-undefined \
	-version-number 0 \
	$(BSYMBOLIC) \
	$(GC_SECTIONS) \
	$(LD_NO_UNDEFINED)

else # USE_LIBGLVND

lib_LTLIBRARIES = libEGL.la
libEGL_la_SOURCES =
libEGL_la_LIBADD = \
	libEGL_common.la \
	$(top_builddir)/src/mapi/shared-glapi/libglapi.la
libEGL_la_LDFLAGS = \
	-no-undefined \
	-version-number 1:0 \
	$(BSYMBOLIC) \
	$(GC_SECTIONS) \
	$(LD_NO_UNDEFINED)

endif # USE_LIBGLVND

include $(top_srcdir)/install-lib-links.mk

# TODO: When building with glvnd, glvnd itself should be providing egl.pc and the headers
pkgconfigdir = $(libdir)/pkgconfig

pkgconfig_DATA = main/egl.pc

egldir = $(includedir)/EGL
egl_HEADERS = \
	$(top_srcdir)/include/EGL/eglext.h \
	$(top_srcdir)/include/EGL/egl.h \
	$(top_srcdir)/include/EGL/eglextchromium.h \
	$(top_srcdir)/include/EGL/eglmesaext.h \
	$(top_srcdir)/include/EGL/eglplatform.h

TESTS = egl-entrypoint-check

if USE_LIBGLVND
#TODO: glvnd symbol check
else
TESTS += egl-symbols-check
endif

EXTRA_DIST = \
	$(TESTS) \
	SConscript \
	drivers/haiku \
	main/egl.def \
	main/README.txt \
	$(GLVND_GEN_DEPS) \
	main/50_mesa.json \
	meson.build
