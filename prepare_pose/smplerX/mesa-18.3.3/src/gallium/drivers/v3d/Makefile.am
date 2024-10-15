# Copyright © 2014 Broadcom
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
include $(top_srcdir)/src/gallium/Automake.inc

AM_CFLAGS = \
	-I$(top_builddir)/src/compiler/nir \
	-I$(top_srcdir)/include/drm-uapi \
	-I$(top_srcdir)/src/broadcom \
	-I$(top_builddir)/src/broadcom \
	-I$(top_builddir)/src \
	$(LIBDRM_CFLAGS) \
	$(V3D_SIMULATOR_CFLAGS) \
	$(GALLIUM_DRIVER_CFLAGS) \
	$(VALGRIND_CFLAGS) \
	$()

noinst_LTLIBRARIES = \
	libv3d.la \
	libv3d_v33.la \
	libv3d_v41.la \
	$()

libv3d_v33_la_SOURCES = $(V3D_PER_VERSION_SOURCES)
libv3d_v33_la_CFLAGS = $(AM_CFLAGS) -DV3D_VERSION=33

libv3d_v41_la_SOURCES = $(V3D_PER_VERSION_SOURCES)
libv3d_v41_la_CFLAGS = $(AM_CFLAGS) -DV3D_VERSION=41

libv3d_la_SOURCES = $(C_SOURCES)

libv3d_la_LDFLAGS = \
	$(V3D_SIMULATOR_LIBS) \
	$(NULL)
libv3d_la_LIBADD = \
	libv3d_v33.la \
	libv3d_v41.la \
	$()

EXTRA_DIST = meson.build
