include Makefile.sources
include $(top_srcdir)/src/gallium/Automake.inc

AM_CFLAGS = \
	-I$(top_srcdir)/src/gallium/drivers \
	$(GALLIUM_WINSYS_CFLAGS)

noinst_LTLIBRARIES = libtegradrm.la

libtegradrm_la_SOURCES = $(C_SOURCES)

EXTRA_DIST = \
	meson.build
