if HAVE_GALLIUM_V3D

TARGET_DRIVERS += v3d
TARGET_CPPFLAGS += -DGALLIUM_V3D
TARGET_LIB_DEPS += \
	$(top_builddir)/src/gallium/winsys/v3d/drm/libv3ddrm.la \
	$(top_builddir)/src/gallium/drivers/v3d/libv3d.la \
	$(top_builddir)/src/broadcom/libbroadcom.la \
	$(top_builddir)/src/broadcom/libbroadcom_v33.la \
	$(top_builddir)/src/broadcom/libbroadcom_v41.la

if !HAVE_GALLIUM_VC4
TARGET_LIB_DEPS += $(top_builddir)/src/broadcom/cle/libbroadcom_cle.la
endif

endif
