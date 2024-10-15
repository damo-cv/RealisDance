if HAVE_GALLIUM_PL111

TARGET_DRIVERS += pl111
TARGET_CPPFLAGS += -DGALLIUM_PL111
TARGET_LIB_DEPS += \
    $(top_builddir)/src/gallium/winsys/pl111/drm/libpl111drm.la \
    $(LIBDRM_LIBS)

endif
