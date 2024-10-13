if HAVE_GALLIUM_VC4

TARGET_DRIVERS += vc4
TARGET_CPPFLAGS += -DGALLIUM_VC4

if !HAVE_GALLIUM_V3D
TARGET_LIB_DEPS += \
	$(top_builddir)/src/broadcom/libbroadcom.la \
	$(top_builddir)/src/broadcom/libbroadcom_v33.la \
	$(top_builddir)/src/broadcom/libbroadcom_v41.la
endif

TARGET_LIB_DEPS += \
	$(top_builddir)/src/gallium/winsys/vc4/drm/libvc4drm.la \
	$(top_builddir)/src/gallium/drivers/vc4/libvc4.la \
	$(top_builddir)/src/broadcom/cle/libbroadcom_cle.la


endif
