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

include Makefile.sources
include $(top_srcdir)/src/gallium/Automake.inc

AM_CFLAGS = \
	-I$(top_srcdir)/include \
	-I$(top_srcdir)/src/mapi \
	-I$(top_srcdir)/src/mesa \
	-I$(top_builddir)/src/mesa/drivers/dri/common \
	-I$(top_srcdir)/src/mesa/drivers/dri/common \
	-I$(top_srcdir)/src/egl \
	-I$(top_srcdir)/src/egl/drivers/dri2 \
	-I$(top_srcdir)/src/egl/main \
	-I$(top_srcdir)/src/gbm/main \
	-I$(top_srcdir)/src/loader \
	-I$(top_srcdir)/src/gbm/backends/dri \
	-I$(top_srcdir)/src/gallium/state_trackers/dri \
	-I$(top_srcdir)/src/gallium/state_trackers/omx \
	$(GALLIUM_CFLAGS) \
	$(LIBDRM_CFLAGS) \
	$(WAYLAND_EGL_CFLAGS) \
	$(VISIBILITY_CFLAGS) \
	$(VL_CFLAGS) \
	$(XCB_DRI3_CFLAGS) \
	$(OMX_TIZONIA_CFLAGS) \
	$(OMX_TIZILHEADERS_CFLAGS) \
	$(OMX_TIZPLATFORM_CFLAGS)

noinst_LTLIBRARIES = libomxtiztracker.la

libomxtiztracker_la_SOURCES = $(C_SOURCES)
