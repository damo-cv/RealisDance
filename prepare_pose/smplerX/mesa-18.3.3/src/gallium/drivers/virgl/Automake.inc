if HAVE_GALLIUM_VIRGL

TARGET_DRIVERS += virtio_gpu
TARGET_CPPFLAGS += -DGALLIUM_VIRGL
TARGET_LIB_DEPS += \
	$(top_builddir)/src/gallium/drivers/virgl/libvirgl.la \
	$(top_builddir)/src/gallium/winsys/virgl/drm/libvirgldrm.la \
	$(top_builddir)/src/gallium/winsys/virgl/vtest/libvirglvtest.la \
	$(LIBDRM_LIBS)

endif
