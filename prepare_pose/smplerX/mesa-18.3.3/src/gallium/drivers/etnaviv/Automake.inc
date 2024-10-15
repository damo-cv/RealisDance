if HAVE_GALLIUM_ETNAVIV

TARGET_DRIVERS += etnaviv
TARGET_CPPFLAGS += -DGALLIUM_ETNAVIV
TARGET_LIB_DEPS += \
	$(top_builddir)/src/gallium/winsys/etnaviv/drm/libetnavivdrm.la \
	$(top_builddir)/src/gallium/drivers/etnaviv/libetnaviv.la \
	$(ETNAVIV_LIBS) \
	$(LIBDRM_LIBS)

endif
