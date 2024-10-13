if HAVE_GALLIUM_IMX

TARGET_DRIVERS += imx-drm
TARGET_CPPFLAGS += -DGALLIUM_IMX
TARGET_LIB_DEPS += \
    $(top_builddir)/src/gallium/winsys/imx/drm/libimxdrm.la \
    $(LIBDRM_LIBS)

endif
