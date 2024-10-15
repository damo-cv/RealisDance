include $(top_srcdir)/src/gallium/Automake.inc

PIPE_SRC_DIR = $(top_builddir)/src/gallium/targets/pipe-loader

AM_CFLAGS = \
	$(GALLIUM_CFLAGS)

LDADD = \
	$(top_builddir)/src/gallium/auxiliary/pipe-loader/libpipe_loader_dynamic.la \
	$(top_builddir)/src/gallium/auxiliary/libgallium.la \
	$(top_builddir)/src/util/libmesautil.la \
	$(GALLIUM_COMMON_LIB_DEPS)

noinst_PROGRAMS = compute tri quad-tex

compute_SOURCES = compute.c

tri_SOURCES = tri.c

quad_tex_SOURCES = quad-tex.c

EXTRA_DIST = meson.build

clean-local:
	-rm -f result.bmp
