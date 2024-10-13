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

include Makefile.sources
include $(top_srcdir)/src/gallium/Automake.inc

AM_CFLAGS = \
	$(GALLIUM_DRIVER_CFLAGS) \
	-I$(top_builddir)/src/amd/common \
	-I$(top_builddir)/src/compiler/nir \
	-I$(top_srcdir)/src/amd/common \
	$(RADEON_CFLAGS) \
	$(LLVM_CFLAGS) \
	-Wstrict-overflow=0
# ^^ disable warnings about overflows (os_time_timeout)

noinst_LTLIBRARIES = libradeonsi.la

libradeonsi_la_SOURCES = $(C_SOURCES)

libradeonsi_la_LIBADD = \
	$(CLOCK_LIB) \
	$(LLVM_LIBS)

libradeonsi_la_LDFLAGS = \
	$(LLVM_LDFLAGS)

GEN_DRIINFO_INPUTS = \
	$(top_srcdir)/src/gallium/auxiliary/pipe-loader/driinfo_gallium.h \
	$(srcdir)/driinfo_radeonsi.h

PYTHON_GEN = $(AM_V_GEN)$(PYTHON) $(PYTHON_FLAGS)
MERGE_DRIINFO = $(top_srcdir)/src/util/merge_driinfo.py

si_driinfo.h: $(MERGE_DRIINFO) $(GEN_DRIINFO_INPUTS)
	$(PYTHON_GEN) $(MERGE_DRIINFO) $(GEN_DRIINFO_INPUTS) > $@ || ($(RM) $@; false)

BUILT_SOURCES = $(GENERATED_SOURCES)
CLEANFILES = $(GENERATED_SOURCES)
EXTRA_DIST = meson.build
