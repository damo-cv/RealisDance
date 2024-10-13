if HAVE_GALLIUM_SWR

TARGET_DRIVERS += swrast
TARGET_CPPFLAGS += -DGALLIUM_SWR
TARGET_LIB_DEPS += \
	$(top_builddir)/src/gallium/drivers/swr/libmesaswr.la

endif
