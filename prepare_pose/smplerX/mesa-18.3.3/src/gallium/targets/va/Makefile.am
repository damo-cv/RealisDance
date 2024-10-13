include $(top_srcdir)/src/gallium/Automake.inc

AM_CFLAGS = \
	$(GALLIUM_TARGET_CFLAGS)

vadir = $(VA_LIB_INSTALL_DIR)
va_LTLIBRARIES = gallium_drv_video.la

nodist_EXTRA_gallium_drv_video_la_SOURCES = dummy.cpp
gallium_drv_video_la_SOURCES =

gallium_drv_video_la_LDFLAGS = \
	-shared \
	-module \
	-no-undefined \
	-avoid-version \
	$(GC_SECTIONS) \
	$(LD_NO_UNDEFINED)

if HAVE_LD_VERSION_SCRIPT
gallium_drv_video_la_LDFLAGS += \
	-Wl,--version-script=$(top_srcdir)/src/gallium/targets/va/va.sym
endif # HAVE_LD_VERSION_SCRIPT

# NOTE: gallium_drv_video does not use(link against) libva
gallium_drv_video_la_LIBADD = \
	$(top_builddir)/src/gallium/state_trackers/va/libvatracker.la \
	$(top_builddir)/src/gallium/auxiliary/libgalliumvlwinsys.la \
	$(top_builddir)/src/gallium/auxiliary/libgalliumvl.la \
	$(top_builddir)/src/gallium/auxiliary/libgallium.la \
	$(top_builddir)/src/util/libmesautil.la \
	$(LIBDRM_LIBS) \
	$(GALLIUM_COMMON_LIB_DEPS)

if HAVE_PLATFORM_X11
gallium_drv_video_la_LIBADD += \
	$(VL_LIBS) \
	$(XCB_DRI3_LIBS)
endif

EXTRA_gallium_drv_video_la_DEPENDENCIES = va.sym
EXTRA_DIST = va.sym meson.build

if HAVE_GALLIUM_STATIC_TARGETS

TARGET_DRIVERS =
TARGET_CPPFLAGS =
TARGET_LIB_DEPS =

include $(top_srcdir)/src/gallium/drivers/nouveau/Automake.inc

include $(top_srcdir)/src/gallium/drivers/r600/Automake.inc
include $(top_srcdir)/src/gallium/drivers/radeonsi/Automake.inc

gallium_drv_video_la_SOURCES += target.c
gallium_drv_video_la_CPPFLAGS = $(TARGET_CPPFLAGS)
gallium_drv_video_la_LIBADD += \
	$(top_builddir)/src/gallium/auxiliary/pipe-loader/libpipe_loader_static.la \
	$(GALLIUM_PIPE_LOADER_WINSYS_LIBS) \
	$(TARGET_LIB_DEPS) \
	$(TARGET_COMPILER_LIB_DEPS) \
	$(TARGET_RADEON_WINSYS) $(TARGET_RADEON_COMMON)

else # HAVE_GALLIUM_STATIC_TARGETS

gallium_drv_video_la_LIBADD += \
	$(top_builddir)/src/gallium/auxiliary/pipe-loader/libpipe_loader_dynamic.la

endif # HAVE_GALLIUM_STATIC_TARGETS

if HAVE_GALLIUM_LLVM
gallium_drv_video_la_LIBADD += $(LLVM_LIBS)
gallium_drv_video_la_LDFLAGS += $(LLVM_LDFLAGS)
endif

# hardlink each megadriver instance, but don't actually have
# gallium_drv_video.so in the set of final installed files.
install-data-hook:
	for i in $(TARGET_DRIVERS); do                                  \
		ln -f $(DESTDIR)$(vadir)/gallium_drv_video.so                \
		      $(DESTDIR)$(vadir)/$${i}_drv_video.so;                 \
	done;                                                           \
	$(RM) $(DESTDIR)$(vadir)/gallium_drv_video.*

uninstall-hook:
	for i in $(TARGET_DRIVERS); do                                  \
		$(RM) $(DESTDIR)$(vadir)/$${i}_drv_video.so;                 \
	done;
