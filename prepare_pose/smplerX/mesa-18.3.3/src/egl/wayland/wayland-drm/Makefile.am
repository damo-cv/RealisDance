AM_CFLAGS = -I$(top_srcdir)/include \
	    $(DEFINES) \
	    $(VISIBILITY_CFLAGS) \
	    $(WAYLAND_SERVER_CFLAGS)

if HAVE_PLATFORM_WAYLAND
wayland-drm.lo: wayland-drm-client-protocol.h wayland-drm-server-protocol.h
endif

noinst_LTLIBRARIES = libwayland-drm.la

nodist_libwayland_drm_la_SOURCES = \
	wayland-drm-protocol.c \
	wayland-drm-client-protocol.h \
	wayland-drm-server-protocol.h

libwayland_drm_la_SOURCES = \
	wayland-drm.c \
	wayland-drm.h

noinst_HEADERS = wayland-drm.h

EXTRA_DIST = wayland-drm.xml meson.build

CLEANFILES = \
	wayland-drm-protocol.c \
	wayland-drm-client-protocol.h \
	wayland-drm-server-protocol.h

%-protocol.c : %.xml
	$(AM_V_GEN)$(WAYLAND_SCANNER) $(SCANNER_ARG) $< $@

%-server-protocol.h : %.xml
	$(AM_V_GEN)$(WAYLAND_SCANNER) server-header $< $@

%-client-protocol.h : %.xml
	$(AM_V_GEN)$(WAYLAND_SCANNER) client-header $< $@
