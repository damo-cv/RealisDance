if HAVE_GALLIUM_SOFTPIPE

TARGET_DRIVERS += swrast
TARGET_CPPFLAGS += -DGALLIUM_SOFTPIPE
TARGET_LIB_DEPS += \
	$(top_builddir)/src/gallium/drivers/softpipe/libsoftpipe.la

if HAVE_DRISW_KMS
TARGET_DRIVERS += kms_swrast

endif
endif
