include Makefile.sources
include $(top_srcdir)/src/gallium/Automake.inc

AM_CFLAGS = \
	$(GALLIUM_WINSYS_CFLAGS) \
	$(AMDGPU_CFLAGS) \
	-I$(top_srcdir)/src/amd/

AM_CXXFLAGS = $(AM_CFLAGS)

noinst_LTLIBRARIES = libamdgpuwinsys.la

libamdgpuwinsys_la_LIBADD = \
   $(top_builddir)/src/amd/addrlib/libamdgpu_addrlib.la

libamdgpuwinsys_la_SOURCES = $(C_SOURCES)

EXTRA_DIST = meson.build
