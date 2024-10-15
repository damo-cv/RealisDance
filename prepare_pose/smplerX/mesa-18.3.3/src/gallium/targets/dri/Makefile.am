include $(top_srcdir)/src/gallium/Automake.inc

AM_CFLAGS = \
	-I$(top_srcdir)/src/mapi \
	-I$(top_srcdir)/src/mesa \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/src/mesa/drivers/dri/common \
	-I$(top_srcdir)/src/gallium/state_trackers/dri \
	$(GALLIUM_TARGET_CFLAGS)

AM_CPPFLAGS = \
	$(DEFINES)

dridir = $(DRI_DRIVER_INSTALL_DIR)
dri_LTLIBRARIES = gallium_dri.la

nodist_EXTRA_gallium_dri_la_SOURCES = dummy.cpp
gallium_dri_la_SOURCES =

gallium_dri_la_LDFLAGS = \
	-shared \
	-shrext .so \
	-module \
	-no-undefined \
	-avoid-version \
	$(GC_SECTIONS) \
	$(LD_NO_UNDEFINED)

if HAVE_LD_VERSION_SCRIPT
gallium_dri_la_LDFLAGS += \
	-Wl,--version-script=$(top_srcdir)/src/gallium/targets/dri/dri.sym
endif # HAVE_LD_VERSION_SCRIPT

if HAVE_LD_DYNAMIC_LIST
gallium_dri_la_LDFLAGS += \
	-Wl,--dynamic-list=$(top_srcdir)/src/gallium/targets/dri-vdpau.dyn
endif # HAVE_LD_DYNAMIC_LIST

gallium_dri_la_LIBADD = \
	$(top_builddir)/src/mesa/libmesagallium.la \
	$(top_builddir)/src/mesa/drivers/dri/common/libdricommon.la \
	$(top_builddir)/src/mesa/drivers/dri/common/libmegadriver_stub.la \
	$(top_builddir)/src/gallium/state_trackers/dri/libdri.la \
	$(top_builddir)/src/gallium/auxiliary/libgalliumvl.la \
	$(top_builddir)/src/gallium/auxiliary/libgallium.la \
	$(top_builddir)/src/mapi/shared-glapi/libglapi.la \
	$(LIBDRM_LIBS) \
	$(SELINUX_LIBS) \
	$(EXPAT_LIBS) \
	$(GALLIUM_COMMON_LIB_DEPS)

EXTRA_gallium_dri_la_DEPENDENCIES = \
	dri.sym \
	$(top_srcdir)/src/gallium/targets/dri-vdpau.dyn
EXTRA_DIST = \
	SConscript \
	meson.build \
	dri.sym \
	$(top_srcdir)/src/gallium/targets/dri-vdpau.dyn

TARGET_DRIVERS =
TARGET_CPPFLAGS =
TARGET_LIB_DEPS =

include $(top_srcdir)/src/gallium/drivers/i915/Automake.inc

include $(top_srcdir)/src/gallium/drivers/nouveau/Automake.inc

include $(top_srcdir)/src/gallium/drivers/r300/Automake.inc
include $(top_srcdir)/src/gallium/drivers/r600/Automake.inc
include $(top_srcdir)/src/gallium/drivers/radeonsi/Automake.inc

include $(top_srcdir)/src/gallium/drivers/svga/Automake.inc

include $(top_srcdir)/src/gallium/drivers/freedreno/Automake.inc

include $(top_srcdir)/src/gallium/drivers/tegra/Automake.inc

include $(top_srcdir)/src/gallium/drivers/v3d/Automake.inc
include $(top_srcdir)/src/gallium/drivers/vc4/Automake.inc
include $(top_srcdir)/src/gallium/drivers/pl111/Automake.inc

include $(top_srcdir)/src/gallium/drivers/virgl/Automake.inc

include $(top_srcdir)/src/gallium/drivers/etnaviv/Automake.inc
include $(top_srcdir)/src/gallium/drivers/imx/Automake.inc

include $(top_srcdir)/src/gallium/drivers/softpipe/Automake.inc
include $(top_srcdir)/src/gallium/drivers/llvmpipe/Automake.inc
include $(top_srcdir)/src/gallium/drivers/swr/Automake.inc

if HAVE_GALLIUM_STATIC_TARGETS

gallium_dri_la_SOURCES += target.c
gallium_dri_la_CPPFLAGS = $(AM_CPPFLAGS) $(TARGET_CPPFLAGS)
gallium_dri_la_LIBADD += \
	$(top_builddir)/src/gallium/auxiliary/pipe-loader/libpipe_loader_static.la \
	$(GALLIUM_PIPE_LOADER_WINSYS_LIBS) \
	$(TARGET_LIB_DEPS) \
	$(TARGET_RADEON_WINSYS) $(TARGET_RADEON_COMMON)

else # HAVE_GALLIUM_STATIC_TARGETS

gallium_dri_la_LIBADD += \
	$(top_builddir)/src/gallium/auxiliary/pipe-loader/libpipe_loader_dynamic.la

endif # HAVE_GALLIUM_STATIC_TARGETS

if HAVE_GALLIUM_LLVM
gallium_dri_la_LIBADD += $(LLVM_LIBS)
gallium_dri_la_LDFLAGS += $(LLVM_LDFLAGS)
endif

if HAVE_COMPAT_SYMLINKS
# Add a link to allow setting LIBGL_DRIVERS_PATH to /lib/gallium of the build tree.
all-local: $(dri_LTLIBRARIES)
	$(AM_V_GEN)link_dir=$(top_builddir)/$(LIB_DIR)/gallium;         \
	$(MKDIR_P) $${link_dir};                                        \
	for i in $(TARGET_DRIVERS); do                                  \
		ln -f .libs/gallium_dri.so                              \
			$${link_dir}/$${i}_dri.so;                      \
	done

clean-local:
	$(AM_V_GEN)link_dir=$(top_builddir)/$(LIB_DIR)/gallium;         \
	$(AM_V_GEN)for i in $(TARGET_DRIVERS); do                       \
		$(RM) $${link_dir}/$${i}_dri.so;                        \
	done;
endif

# hardlink each megadriver instance, but don't actually have
# gallium_dri.so in the set of final installed files.
install-data-hook:
	for i in $(TARGET_DRIVERS); do                                  \
		ln -f $(DESTDIR)$(dridir)/gallium_dri.so                \
		      $(DESTDIR)$(dridir)/$${i}_dri.so;                 \
	done;                                                           \
	$(RM) $(DESTDIR)$(dridir)/gallium_dri.*;                        \
	$(RM) -d $(DESTDIR)$(dridir) &>/dev/null || true

uninstall-hook:
	for i in $(TARGET_DRIVERS); do                                  \
		$(RM) $(DESTDIR)$(dridir)/$${i}_dri.so;                 \
	done;
