#include "egldispatchstubs.h"
#include "g_egldispatchstubs.h"

static EGLBoolean EGLAPIENTRY dispatch_eglBindWaylandDisplayWL(EGLDisplay dpy, struct wl_display *display)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglBindWaylandDisplayWL)(EGLDisplay dpy, struct wl_display *display);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglBindWaylandDisplayWL _ptr_eglBindWaylandDisplayWL = (_pfn_eglBindWaylandDisplayWL) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglBindWaylandDisplayWL);
    if(_ptr_eglBindWaylandDisplayWL != NULL) {
        _ret = _ptr_eglBindWaylandDisplayWL(dpy, display);
    }
    return _ret;
}
static EGLint EGLAPIENTRY dispatch_eglClientWaitSyncKHR(EGLDisplay dpy, EGLSyncKHR sync, EGLint flags, EGLTimeKHR timeout)
{
    typedef EGLint EGLAPIENTRY (* _pfn_eglClientWaitSyncKHR)(EGLDisplay dpy, EGLSyncKHR sync, EGLint flags, EGLTimeKHR timeout);
    EGLint _ret = 0;
    _pfn_eglClientWaitSyncKHR _ptr_eglClientWaitSyncKHR = (_pfn_eglClientWaitSyncKHR) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglClientWaitSyncKHR);
    if(_ptr_eglClientWaitSyncKHR != NULL) {
        _ret = _ptr_eglClientWaitSyncKHR(dpy, sync, flags, timeout);
    }
    return _ret;
}
static EGLImageKHR EGLAPIENTRY dispatch_eglCreateDRMImageMESA(EGLDisplay dpy, const EGLint *attrib_list)
{
    typedef EGLImageKHR EGLAPIENTRY (* _pfn_eglCreateDRMImageMESA)(EGLDisplay dpy, const EGLint *attrib_list);
    EGLImageKHR _ret = 0;
    _pfn_eglCreateDRMImageMESA _ptr_eglCreateDRMImageMESA = (_pfn_eglCreateDRMImageMESA) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglCreateDRMImageMESA);
    if(_ptr_eglCreateDRMImageMESA != NULL) {
        _ret = _ptr_eglCreateDRMImageMESA(dpy, attrib_list);
    }
    return _ret;
}
static EGLImageKHR EGLAPIENTRY dispatch_eglCreateImageKHR(EGLDisplay dpy, EGLContext ctx, EGLenum target, EGLClientBuffer buffer, const EGLint *attrib_list)
{
    typedef EGLImageKHR EGLAPIENTRY (* _pfn_eglCreateImageKHR)(EGLDisplay dpy, EGLContext ctx, EGLenum target, EGLClientBuffer buffer, const EGLint *attrib_list);
    EGLImageKHR _ret = 0;
    _pfn_eglCreateImageKHR _ptr_eglCreateImageKHR = (_pfn_eglCreateImageKHR) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglCreateImageKHR);
    if(_ptr_eglCreateImageKHR != NULL) {
        _ret = _ptr_eglCreateImageKHR(dpy, ctx, target, buffer, attrib_list);
    }
    return _ret;
}
static EGLSurface EGLAPIENTRY dispatch_eglCreatePlatformPixmapSurfaceEXT(EGLDisplay dpy, EGLConfig config, void *native_pixmap, const EGLint *attrib_list)
{
    typedef EGLSurface EGLAPIENTRY (* _pfn_eglCreatePlatformPixmapSurfaceEXT)(EGLDisplay dpy, EGLConfig config, void *native_pixmap, const EGLint *attrib_list);
    EGLSurface _ret = EGL_NO_SURFACE;
    _pfn_eglCreatePlatformPixmapSurfaceEXT _ptr_eglCreatePlatformPixmapSurfaceEXT = (_pfn_eglCreatePlatformPixmapSurfaceEXT) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglCreatePlatformPixmapSurfaceEXT);
    if(_ptr_eglCreatePlatformPixmapSurfaceEXT != NULL) {
        _ret = _ptr_eglCreatePlatformPixmapSurfaceEXT(dpy, config, native_pixmap, attrib_list);
    }
    return _ret;
}
static EGLSurface EGLAPIENTRY dispatch_eglCreatePlatformWindowSurfaceEXT(EGLDisplay dpy, EGLConfig config, void *native_window, const EGLint *attrib_list)
{
    typedef EGLSurface EGLAPIENTRY (* _pfn_eglCreatePlatformWindowSurfaceEXT)(EGLDisplay dpy, EGLConfig config, void *native_window, const EGLint *attrib_list);
    EGLSurface _ret = EGL_NO_SURFACE;
    _pfn_eglCreatePlatformWindowSurfaceEXT _ptr_eglCreatePlatformWindowSurfaceEXT = (_pfn_eglCreatePlatformWindowSurfaceEXT) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglCreatePlatformWindowSurfaceEXT);
    if(_ptr_eglCreatePlatformWindowSurfaceEXT != NULL) {
        _ret = _ptr_eglCreatePlatformWindowSurfaceEXT(dpy, config, native_window, attrib_list);
    }
    return _ret;
}
static EGLSyncKHR EGLAPIENTRY dispatch_eglCreateSync64KHR(EGLDisplay dpy, EGLenum type, const EGLAttribKHR *attrib_list)
{
    typedef EGLSyncKHR EGLAPIENTRY (* _pfn_eglCreateSync64KHR)(EGLDisplay dpy, EGLenum type, const EGLAttribKHR *attrib_list);
    EGLSyncKHR _ret = 0;
    _pfn_eglCreateSync64KHR _ptr_eglCreateSync64KHR = (_pfn_eglCreateSync64KHR) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglCreateSync64KHR);
    if(_ptr_eglCreateSync64KHR != NULL) {
        _ret = _ptr_eglCreateSync64KHR(dpy, type, attrib_list);
    }
    return _ret;
}
static EGLSyncKHR EGLAPIENTRY dispatch_eglCreateSyncKHR(EGLDisplay dpy, EGLenum type, const EGLint *attrib_list)
{
    typedef EGLSyncKHR EGLAPIENTRY (* _pfn_eglCreateSyncKHR)(EGLDisplay dpy, EGLenum type, const EGLint *attrib_list);
    EGLSyncKHR _ret = 0;
    _pfn_eglCreateSyncKHR _ptr_eglCreateSyncKHR = (_pfn_eglCreateSyncKHR) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglCreateSyncKHR);
    if(_ptr_eglCreateSyncKHR != NULL) {
        _ret = _ptr_eglCreateSyncKHR(dpy, type, attrib_list);
    }
    return _ret;
}
static struct wl_buffer * EGLAPIENTRY dispatch_eglCreateWaylandBufferFromImageWL(EGLDisplay dpy, EGLImage image)
{
    typedef struct wl_buffer * EGLAPIENTRY (* _pfn_eglCreateWaylandBufferFromImageWL)(EGLDisplay dpy, EGLImage image);
    struct wl_buffer * _ret = NULL;
    _pfn_eglCreateWaylandBufferFromImageWL _ptr_eglCreateWaylandBufferFromImageWL = (_pfn_eglCreateWaylandBufferFromImageWL) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglCreateWaylandBufferFromImageWL);
    if(_ptr_eglCreateWaylandBufferFromImageWL != NULL) {
        _ret = _ptr_eglCreateWaylandBufferFromImageWL(dpy, image);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglDestroyImageKHR(EGLDisplay dpy, EGLImageKHR image)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglDestroyImageKHR)(EGLDisplay dpy, EGLImageKHR image);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglDestroyImageKHR _ptr_eglDestroyImageKHR = (_pfn_eglDestroyImageKHR) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglDestroyImageKHR);
    if(_ptr_eglDestroyImageKHR != NULL) {
        _ret = _ptr_eglDestroyImageKHR(dpy, image);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglDestroySyncKHR(EGLDisplay dpy, EGLSyncKHR sync)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglDestroySyncKHR)(EGLDisplay dpy, EGLSyncKHR sync);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglDestroySyncKHR _ptr_eglDestroySyncKHR = (_pfn_eglDestroySyncKHR) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglDestroySyncKHR);
    if(_ptr_eglDestroySyncKHR != NULL) {
        _ret = _ptr_eglDestroySyncKHR(dpy, sync);
    }
    return _ret;
}
static EGLint EGLAPIENTRY dispatch_eglDupNativeFenceFDANDROID(EGLDisplay dpy, EGLSyncKHR sync)
{
    typedef EGLint EGLAPIENTRY (* _pfn_eglDupNativeFenceFDANDROID)(EGLDisplay dpy, EGLSyncKHR sync);
    EGLint _ret = 0;
    _pfn_eglDupNativeFenceFDANDROID _ptr_eglDupNativeFenceFDANDROID = (_pfn_eglDupNativeFenceFDANDROID) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglDupNativeFenceFDANDROID);
    if(_ptr_eglDupNativeFenceFDANDROID != NULL) {
        _ret = _ptr_eglDupNativeFenceFDANDROID(dpy, sync);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglExportDMABUFImageMESA(EGLDisplay dpy, EGLImageKHR image, int *fds, EGLint *strides, EGLint *offsets)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglExportDMABUFImageMESA)(EGLDisplay dpy, EGLImageKHR image, int *fds, EGLint *strides, EGLint *offsets);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglExportDMABUFImageMESA _ptr_eglExportDMABUFImageMESA = (_pfn_eglExportDMABUFImageMESA) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglExportDMABUFImageMESA);
    if(_ptr_eglExportDMABUFImageMESA != NULL) {
        _ret = _ptr_eglExportDMABUFImageMESA(dpy, image, fds, strides, offsets);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglExportDMABUFImageQueryMESA(EGLDisplay dpy, EGLImageKHR image, int *fourcc, int *num_planes, EGLuint64KHR *modifiers)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglExportDMABUFImageQueryMESA)(EGLDisplay dpy, EGLImageKHR image, int *fourcc, int *num_planes, EGLuint64KHR *modifiers);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglExportDMABUFImageQueryMESA _ptr_eglExportDMABUFImageQueryMESA = (_pfn_eglExportDMABUFImageQueryMESA) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglExportDMABUFImageQueryMESA);
    if(_ptr_eglExportDMABUFImageQueryMESA != NULL) {
        _ret = _ptr_eglExportDMABUFImageQueryMESA(dpy, image, fourcc, num_planes, modifiers);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglExportDRMImageMESA(EGLDisplay dpy, EGLImageKHR image, EGLint *name, EGLint *handle, EGLint *stride)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglExportDRMImageMESA)(EGLDisplay dpy, EGLImageKHR image, EGLint *name, EGLint *handle, EGLint *stride);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglExportDRMImageMESA _ptr_eglExportDRMImageMESA = (_pfn_eglExportDRMImageMESA) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglExportDRMImageMESA);
    if(_ptr_eglExportDRMImageMESA != NULL) {
        _ret = _ptr_eglExportDRMImageMESA(dpy, image, name, handle, stride);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglGetSyncAttribKHR(EGLDisplay dpy, EGLSyncKHR sync, EGLint attribute, EGLint *value)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglGetSyncAttribKHR)(EGLDisplay dpy, EGLSyncKHR sync, EGLint attribute, EGLint *value);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglGetSyncAttribKHR _ptr_eglGetSyncAttribKHR = (_pfn_eglGetSyncAttribKHR) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglGetSyncAttribKHR);
    if(_ptr_eglGetSyncAttribKHR != NULL) {
        _ret = _ptr_eglGetSyncAttribKHR(dpy, sync, attribute, value);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglGetSyncValuesCHROMIUM(EGLDisplay display, EGLSurface surface, EGLuint64KHR *ust, EGLuint64KHR *msc, EGLuint64KHR *sbc)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglGetSyncValuesCHROMIUM)(EGLDisplay display, EGLSurface surface, EGLuint64KHR *ust, EGLuint64KHR *msc, EGLuint64KHR *sbc);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglGetSyncValuesCHROMIUM _ptr_eglGetSyncValuesCHROMIUM = (_pfn_eglGetSyncValuesCHROMIUM) __eglDispatchFetchByDisplay(display, __EGL_DISPATCH_eglGetSyncValuesCHROMIUM);
    if(_ptr_eglGetSyncValuesCHROMIUM != NULL) {
        _ret = _ptr_eglGetSyncValuesCHROMIUM(display, surface, ust, msc, sbc);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglPostSubBufferNV(EGLDisplay dpy, EGLSurface surface, EGLint x, EGLint y, EGLint width, EGLint height)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglPostSubBufferNV)(EGLDisplay dpy, EGLSurface surface, EGLint x, EGLint y, EGLint width, EGLint height);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglPostSubBufferNV _ptr_eglPostSubBufferNV = (_pfn_eglPostSubBufferNV) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglPostSubBufferNV);
    if(_ptr_eglPostSubBufferNV != NULL) {
        _ret = _ptr_eglPostSubBufferNV(dpy, surface, x, y, width, height);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglQueryDeviceAttribEXT(EGLDeviceEXT device, EGLint attribute, EGLAttrib *value)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglQueryDeviceAttribEXT)(EGLDeviceEXT device, EGLint attribute, EGLAttrib *value);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglQueryDeviceAttribEXT _ptr_eglQueryDeviceAttribEXT = (_pfn_eglQueryDeviceAttribEXT) __eglDispatchFetchByDevice(device, __EGL_DISPATCH_eglQueryDeviceAttribEXT);
    if(_ptr_eglQueryDeviceAttribEXT != NULL) {
        _ret = _ptr_eglQueryDeviceAttribEXT(device, attribute, value);
    }
    return _ret;
}
static const char * EGLAPIENTRY dispatch_eglQueryDeviceStringEXT(EGLDeviceEXT device, EGLint name)
{
    typedef const char * EGLAPIENTRY (* _pfn_eglQueryDeviceStringEXT)(EGLDeviceEXT device, EGLint name);
    const char * _ret = NULL;
    _pfn_eglQueryDeviceStringEXT _ptr_eglQueryDeviceStringEXT = (_pfn_eglQueryDeviceStringEXT) __eglDispatchFetchByDevice(device, __EGL_DISPATCH_eglQueryDeviceStringEXT);
    if(_ptr_eglQueryDeviceStringEXT != NULL) {
        _ret = _ptr_eglQueryDeviceStringEXT(device, name);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglQueryDisplayAttribEXT(EGLDisplay dpy, EGLint attribute, EGLAttrib *value)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglQueryDisplayAttribEXT)(EGLDisplay dpy, EGLint attribute, EGLAttrib *value);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglQueryDisplayAttribEXT _ptr_eglQueryDisplayAttribEXT = (_pfn_eglQueryDisplayAttribEXT) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglQueryDisplayAttribEXT);
    if(_ptr_eglQueryDisplayAttribEXT != NULL) {
        _ret = _ptr_eglQueryDisplayAttribEXT(dpy, attribute, value);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglQueryDmaBufFormatsEXT(EGLDisplay dpy, EGLint max_formats, EGLint *formats, EGLint *num_formats)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglQueryDmaBufFormatsEXT)(EGLDisplay dpy, EGLint max_formats, EGLint *formats, EGLint *num_formats);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglQueryDmaBufFormatsEXT _ptr_eglQueryDmaBufFormatsEXT = (_pfn_eglQueryDmaBufFormatsEXT) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglQueryDmaBufFormatsEXT);
    if(_ptr_eglQueryDmaBufFormatsEXT != NULL) {
        _ret = _ptr_eglQueryDmaBufFormatsEXT(dpy, max_formats, formats, num_formats);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglQueryDmaBufModifiersEXT(EGLDisplay dpy, EGLint format, EGLint max_modifiers, EGLuint64KHR *modifiers, EGLBoolean *external_only, EGLint *num_modifiers)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglQueryDmaBufModifiersEXT)(EGLDisplay dpy, EGLint format, EGLint max_modifiers, EGLuint64KHR *modifiers, EGLBoolean *external_only, EGLint *num_modifiers);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglQueryDmaBufModifiersEXT _ptr_eglQueryDmaBufModifiersEXT = (_pfn_eglQueryDmaBufModifiersEXT) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglQueryDmaBufModifiersEXT);
    if(_ptr_eglQueryDmaBufModifiersEXT != NULL) {
        _ret = _ptr_eglQueryDmaBufModifiersEXT(dpy, format, max_modifiers, modifiers, external_only, num_modifiers);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglQueryWaylandBufferWL(EGLDisplay dpy, struct wl_resource *buffer, EGLint attribute, EGLint *value)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglQueryWaylandBufferWL)(EGLDisplay dpy, struct wl_resource *buffer, EGLint attribute, EGLint *value);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglQueryWaylandBufferWL _ptr_eglQueryWaylandBufferWL = (_pfn_eglQueryWaylandBufferWL) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglQueryWaylandBufferWL);
    if(_ptr_eglQueryWaylandBufferWL != NULL) {
        _ret = _ptr_eglQueryWaylandBufferWL(dpy, buffer, attribute, value);
    }
    return _ret;
}
static void EGLAPIENTRY dispatch_eglSetBlobCacheFuncsANDROID(EGLDisplay dpy, EGLSetBlobFuncANDROID set, EGLGetBlobFuncANDROID get)
{
    typedef void EGLAPIENTRY (* _pfn_eglSetBlobCacheFuncsANDROID)(EGLDisplay dpy, EGLSetBlobFuncANDROID set, EGLGetBlobFuncANDROID get);
    _pfn_eglSetBlobCacheFuncsANDROID _ptr_eglSetBlobCacheFuncsANDROID = (_pfn_eglSetBlobCacheFuncsANDROID) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglSetBlobCacheFuncsANDROID);
    if(_ptr_eglSetBlobCacheFuncsANDROID != NULL) {
        _ptr_eglSetBlobCacheFuncsANDROID(dpy, set, get);
    }
}
static EGLBoolean EGLAPIENTRY dispatch_eglSignalSyncKHR(EGLDisplay dpy, EGLSyncKHR sync, EGLenum mode)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglSignalSyncKHR)(EGLDisplay dpy, EGLSyncKHR sync, EGLenum mode);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglSignalSyncKHR _ptr_eglSignalSyncKHR = (_pfn_eglSignalSyncKHR) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglSignalSyncKHR);
    if(_ptr_eglSignalSyncKHR != NULL) {
        _ret = _ptr_eglSignalSyncKHR(dpy, sync, mode);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglSwapBuffersRegionNOK(EGLDisplay dpy, EGLSurface surface, EGLint numRects, const EGLint *rects)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglSwapBuffersRegionNOK)(EGLDisplay dpy, EGLSurface surface, EGLint numRects, const EGLint *rects);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglSwapBuffersRegionNOK _ptr_eglSwapBuffersRegionNOK = (_pfn_eglSwapBuffersRegionNOK) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglSwapBuffersRegionNOK);
    if(_ptr_eglSwapBuffersRegionNOK != NULL) {
        _ret = _ptr_eglSwapBuffersRegionNOK(dpy, surface, numRects, rects);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglSwapBuffersWithDamageEXT(EGLDisplay dpy, EGLSurface surface, EGLint *rects, EGLint n_rects)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglSwapBuffersWithDamageEXT)(EGLDisplay dpy, EGLSurface surface, EGLint *rects, EGLint n_rects);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglSwapBuffersWithDamageEXT _ptr_eglSwapBuffersWithDamageEXT = (_pfn_eglSwapBuffersWithDamageEXT) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglSwapBuffersWithDamageEXT);
    if(_ptr_eglSwapBuffersWithDamageEXT != NULL) {
        _ret = _ptr_eglSwapBuffersWithDamageEXT(dpy, surface, rects, n_rects);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglSwapBuffersWithDamageKHR(EGLDisplay dpy, EGLSurface surface, EGLint *rects, EGLint n_rects)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglSwapBuffersWithDamageKHR)(EGLDisplay dpy, EGLSurface surface, EGLint *rects, EGLint n_rects);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglSwapBuffersWithDamageKHR _ptr_eglSwapBuffersWithDamageKHR = (_pfn_eglSwapBuffersWithDamageKHR) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglSwapBuffersWithDamageKHR);
    if(_ptr_eglSwapBuffersWithDamageKHR != NULL) {
        _ret = _ptr_eglSwapBuffersWithDamageKHR(dpy, surface, rects, n_rects);
    }
    return _ret;
}
static EGLBoolean EGLAPIENTRY dispatch_eglUnbindWaylandDisplayWL(EGLDisplay dpy, struct wl_display *display)
{
    typedef EGLBoolean EGLAPIENTRY (* _pfn_eglUnbindWaylandDisplayWL)(EGLDisplay dpy, struct wl_display *display);
    EGLBoolean _ret = EGL_FALSE;
    _pfn_eglUnbindWaylandDisplayWL _ptr_eglUnbindWaylandDisplayWL = (_pfn_eglUnbindWaylandDisplayWL) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglUnbindWaylandDisplayWL);
    if(_ptr_eglUnbindWaylandDisplayWL != NULL) {
        _ret = _ptr_eglUnbindWaylandDisplayWL(dpy, display);
    }
    return _ret;
}
static EGLint EGLAPIENTRY dispatch_eglWaitSyncKHR(EGLDisplay dpy, EGLSyncKHR sync, EGLint flags)
{
    typedef EGLint EGLAPIENTRY (* _pfn_eglWaitSyncKHR)(EGLDisplay dpy, EGLSyncKHR sync, EGLint flags);
    EGLint _ret = 0;
    _pfn_eglWaitSyncKHR _ptr_eglWaitSyncKHR = (_pfn_eglWaitSyncKHR) __eglDispatchFetchByDisplay(dpy, __EGL_DISPATCH_eglWaitSyncKHR);
    if(_ptr_eglWaitSyncKHR != NULL) {
        _ret = _ptr_eglWaitSyncKHR(dpy, sync, flags);
    }
    return _ret;
}

const char * const __EGL_DISPATCH_FUNC_NAMES[__EGL_DISPATCH_COUNT + 1] = {
    "eglBindAPI",
    "eglBindTexImage",
    "eglBindWaylandDisplayWL",
    "eglChooseConfig",
    "eglClientWaitSync",
    "eglClientWaitSyncKHR",
    "eglCopyBuffers",
    "eglCreateContext",
    "eglCreateDRMImageMESA",
    "eglCreateImage",
    "eglCreateImageKHR",
    "eglCreatePbufferFromClientBuffer",
    "eglCreatePbufferSurface",
    "eglCreatePixmapSurface",
    "eglCreatePlatformPixmapSurface",
    "eglCreatePlatformPixmapSurfaceEXT",
    "eglCreatePlatformWindowSurface",
    "eglCreatePlatformWindowSurfaceEXT",
    "eglCreateSync",
    "eglCreateSync64KHR",
    "eglCreateSyncKHR",
    "eglCreateWaylandBufferFromImageWL",
    "eglCreateWindowSurface",
    "eglDestroyContext",
    "eglDestroyImage",
    "eglDestroyImageKHR",
    "eglDestroySurface",
    "eglDestroySync",
    "eglDestroySyncKHR",
    "eglDupNativeFenceFDANDROID",
    "eglExportDMABUFImageMESA",
    "eglExportDMABUFImageQueryMESA",
    "eglExportDRMImageMESA",
    "eglGetConfigAttrib",
    "eglGetConfigs",
    "eglGetCurrentContext",
    "eglGetCurrentDisplay",
    "eglGetCurrentSurface",
    "eglGetDisplay",
    "eglGetError",
    "eglGetPlatformDisplay",
    "eglGetPlatformDisplayEXT",
    "eglGetProcAddress",
    "eglGetSyncAttrib",
    "eglGetSyncAttribKHR",
    "eglGetSyncValuesCHROMIUM",
    "eglInitialize",
    "eglMakeCurrent",
    "eglPostSubBufferNV",
    "eglQueryAPI",
    "eglQueryContext",
    "eglQueryDeviceAttribEXT",
    "eglQueryDeviceStringEXT",
    "eglQueryDevicesEXT",
    "eglQueryDisplayAttribEXT",
    "eglQueryDmaBufFormatsEXT",
    "eglQueryDmaBufModifiersEXT",
    "eglQueryString",
    "eglQuerySurface",
    "eglQueryWaylandBufferWL",
    "eglReleaseTexImage",
    "eglReleaseThread",
    "eglSetBlobCacheFuncsANDROID",
    "eglSignalSyncKHR",
    "eglSurfaceAttrib",
    "eglSwapBuffers",
    "eglSwapBuffersRegionNOK",
    "eglSwapBuffersWithDamageEXT",
    "eglSwapBuffersWithDamageKHR",
    "eglSwapInterval",
    "eglTerminate",
    "eglUnbindWaylandDisplayWL",
    "eglWaitClient",
    "eglWaitGL",
    "eglWaitNative",
    "eglWaitSync",
    "eglWaitSyncKHR",
    NULL
};
const __eglMustCastToProperFunctionPointerType __EGL_DISPATCH_FUNCS[__EGL_DISPATCH_COUNT + 1] = {
    NULL, // eglBindAPI
    NULL, // eglBindTexImage
    (__eglMustCastToProperFunctionPointerType) dispatch_eglBindWaylandDisplayWL,
    NULL, // eglChooseConfig
    NULL, // eglClientWaitSync
    (__eglMustCastToProperFunctionPointerType) dispatch_eglClientWaitSyncKHR,
    NULL, // eglCopyBuffers
    NULL, // eglCreateContext
    (__eglMustCastToProperFunctionPointerType) dispatch_eglCreateDRMImageMESA,
    NULL, // eglCreateImage
    (__eglMustCastToProperFunctionPointerType) dispatch_eglCreateImageKHR,
    NULL, // eglCreatePbufferFromClientBuffer
    NULL, // eglCreatePbufferSurface
    NULL, // eglCreatePixmapSurface
    NULL, // eglCreatePlatformPixmapSurface
    (__eglMustCastToProperFunctionPointerType) dispatch_eglCreatePlatformPixmapSurfaceEXT,
    NULL, // eglCreatePlatformWindowSurface
    (__eglMustCastToProperFunctionPointerType) dispatch_eglCreatePlatformWindowSurfaceEXT,
    NULL, // eglCreateSync
    (__eglMustCastToProperFunctionPointerType) dispatch_eglCreateSync64KHR,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglCreateSyncKHR,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglCreateWaylandBufferFromImageWL,
    NULL, // eglCreateWindowSurface
    NULL, // eglDestroyContext
    NULL, // eglDestroyImage
    (__eglMustCastToProperFunctionPointerType) dispatch_eglDestroyImageKHR,
    NULL, // eglDestroySurface
    NULL, // eglDestroySync
    (__eglMustCastToProperFunctionPointerType) dispatch_eglDestroySyncKHR,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglDupNativeFenceFDANDROID,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglExportDMABUFImageMESA,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglExportDMABUFImageQueryMESA,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglExportDRMImageMESA,
    NULL, // eglGetConfigAttrib
    NULL, // eglGetConfigs
    NULL, // eglGetCurrentContext
    NULL, // eglGetCurrentDisplay
    NULL, // eglGetCurrentSurface
    NULL, // eglGetDisplay
    NULL, // eglGetError
    NULL, // eglGetPlatformDisplay
    NULL, // eglGetPlatformDisplayEXT
    NULL, // eglGetProcAddress
    NULL, // eglGetSyncAttrib
    (__eglMustCastToProperFunctionPointerType) dispatch_eglGetSyncAttribKHR,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglGetSyncValuesCHROMIUM,
    NULL, // eglInitialize
    NULL, // eglMakeCurrent
    (__eglMustCastToProperFunctionPointerType) dispatch_eglPostSubBufferNV,
    NULL, // eglQueryAPI
    NULL, // eglQueryContext
    (__eglMustCastToProperFunctionPointerType) dispatch_eglQueryDeviceAttribEXT,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglQueryDeviceStringEXT,
    NULL, // eglQueryDevicesEXT
    (__eglMustCastToProperFunctionPointerType) dispatch_eglQueryDisplayAttribEXT,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglQueryDmaBufFormatsEXT,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglQueryDmaBufModifiersEXT,
    NULL, // eglQueryString
    NULL, // eglQuerySurface
    (__eglMustCastToProperFunctionPointerType) dispatch_eglQueryWaylandBufferWL,
    NULL, // eglReleaseTexImage
    NULL, // eglReleaseThread
    (__eglMustCastToProperFunctionPointerType) dispatch_eglSetBlobCacheFuncsANDROID,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglSignalSyncKHR,
    NULL, // eglSurfaceAttrib
    NULL, // eglSwapBuffers
    (__eglMustCastToProperFunctionPointerType) dispatch_eglSwapBuffersRegionNOK,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglSwapBuffersWithDamageEXT,
    (__eglMustCastToProperFunctionPointerType) dispatch_eglSwapBuffersWithDamageKHR,
    NULL, // eglSwapInterval
    NULL, // eglTerminate
    (__eglMustCastToProperFunctionPointerType) dispatch_eglUnbindWaylandDisplayWL,
    NULL, // eglWaitClient
    NULL, // eglWaitGL
    NULL, // eglWaitNative
    NULL, // eglWaitSync
    (__eglMustCastToProperFunctionPointerType) dispatch_eglWaitSyncKHR,
    NULL
};
