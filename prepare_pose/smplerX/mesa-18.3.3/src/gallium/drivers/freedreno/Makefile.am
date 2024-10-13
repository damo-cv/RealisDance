include Makefile.sources
include $(top_srcdir)/src/gallium/Automake.inc

AM_CFLAGS = \
	-Wno-packed-bitfield-compat \
	-I$(top_srcdir)/src/gallium/drivers/freedreno/ir3 \
	-I$(top_builddir)/src/compiler/nir \
	-I$(top_srcdir)/src/compiler/nir \
	$(GALLIUM_DRIVER_CFLAGS) \
	$(LIBDRM_CFLAGS) \
	$(VALGRIND_CFLAGS)

MKDIR_GEN = $(AM_V_at)$(MKDIR_P) $(@D)
ir3/ir3_nir_trig.c: ir3/ir3_nir_trig.py $(top_srcdir)/src/compiler/nir/nir_algebraic.py
	$(MKDIR_GEN)
	$(AM_V_GEN) $(PYTHON) $(PYTHON_FLAGS) $(srcdir)/ir3/ir3_nir_trig.py -p $(top_srcdir)/src/compiler/nir > $@ || ($(RM) $@; false)

noinst_LTLIBRARIES = libfreedreno.la

libfreedreno_la_SOURCES = \
	$(C_SOURCES) \
	$(drm_SOURCES) \
	$(a2xx_SOURCES) \
	$(a3xx_SOURCES) \
	$(a4xx_SOURCES) \
	$(a5xx_SOURCES) \
	$(a6xx_SOURCES) \
	$(ir3_SOURCES) \
	$(ir3_GENERATED_FILES)

BUILT_SOURCES := $(ir3_GENERATED_FILES)
CLEANFILES := $(BUILT_SOURCES)
EXTRA_DIST = ir3/ir3_nir_trig.py

noinst_PROGRAMS = ir3_compiler

# XXX: Required due to the C++ sources in libnir
nodist_EXTRA_ir3_compiler_SOURCES = dummy.cpp
ir3_compiler_SOURCES = \
	ir3/ir3_cmdline.c

ir3_compiler_LDADD = \
	libfreedreno.la \
	$(top_builddir)/src/gallium/auxiliary/libgallium.la \
	$(top_builddir)/src/compiler/nir/libnir.la \
	$(top_builddir)/src/compiler/glsl/libstandalone.la \
	$(top_builddir)/src/util/libmesautil.la \
	$(top_builddir)/src/mesa/libmesagallium.la \
	$(GALLIUM_COMMON_LIB_DEPS) \
	$(LIBDRM_LIBS)

EXTRA_DIST += meson.build
