include Makefile.sources
include $(top_srcdir)/src/gallium/Automake.inc

AM_CFLAGS = \
	-I$(top_srcdir)/include/drm-uapi \
	$(GALLIUM_DRIVER_CFLAGS)

noinst_LTLIBRARIES = libtegra.la

libtegra_la_SOURCES = \
	$(C_SOURCES)

EXTRA_DIST = \
	meson.build
