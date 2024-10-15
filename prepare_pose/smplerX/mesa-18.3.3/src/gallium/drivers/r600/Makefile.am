include Makefile.sources
include $(top_srcdir)/src/gallium/Automake.inc

egd_tables.h: $(srcdir)/egd_tables.py $(srcdir)/evergreend.h
	$(AM_V_at)$(MKDIR_P) $(@D)
	$(AM_V_GEN) $(PYTHON) $(srcdir)/egd_tables.py $(srcdir)/evergreend.h > $@

BUILT_SOURCES = $(R600_GENERATED_FILES)
AM_CFLAGS = \
	$(GALLIUM_DRIVER_CFLAGS) \
	$(RADEON_CFLAGS) \
	$(LIBELF_CFLAGS) \
	-I$(top_srcdir)/src/amd/common -Wstrict-overflow=0

AM_CXXFLAGS = \
	$(GALLIUM_DRIVER_CXXFLAGS) \
	$(RADEON_CFLAGS) \
	$(LIBELF_CFLAGS) \
	-I$(top_srcdir)/src/amd/common

noinst_LTLIBRARIES = libr600.la

libr600_la_SOURCES = \
	$(C_SOURCES) \
	$(CXX_SOURCES)

if HAVE_GALLIUM_LLVM

AM_CFLAGS += \
	$(LLVM_CFLAGS)

endif

if HAVE_GALLIUM_COMPUTE
AM_CFLAGS += \
	-DHAVE_OPENCL
endif

CLEANFILES = \
	egd_tables.h

EXTRA_DIST = \
	egd_tables.py \
	sb/notes.markdown \
	sb/sb_bc_fmt_def.inc \
	meson.build
