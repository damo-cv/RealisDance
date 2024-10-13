#ifndef G_EGLDISPATCH_STUBS_H
#define G_EGLDISPATCH_STUBS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "glvnd/libeglabi.h"

enum {
    __EGL_DISPATCH_eglBindAPI,
    __EGL_DISPATCH_eglBindTexImage,
    __EGL_DISPATCH_eglBindWaylandDisplayWL,
    __EGL_DISPATCH_eglChooseConfig,
    __EGL_DISPATCH_eglClientWaitSync,
    __EGL_DISPATCH_eglClientWaitSyncKHR,
    __EGL_DISPATCH_eglCopyBuffers,
    __EGL_DISPATCH_eglCreateContext,
    __EGL_DISPATCH_eglCreateDRMImageMESA,
    __EGL_DISPATCH_eglCreateImage,
    __EGL_DISPATCH_eglCreateImageKHR,
    __EGL_DISPATCH_eglCreatePbufferFromClientBuffer,
    __EGL_DISPATCH_eglCreatePbufferSurface,
    __EGL_DISPATCH_eglCreatePixmapSurface,
    __EGL_DISPATCH_eglCreatePlatformPixmapSurface,
    __EGL_DISPATCH_eglCreatePlatformPixmapSurfaceEXT,
    __EGL_DISPATCH_eglCreatePlatformWindowSurface,
    __EGL_DISPATCH_eglCreatePlatformWindowSurfaceEXT,
    __EGL_DISPATCH_eglCreateSync,
    __EGL_DISPATCH_eglCreateSync64KHR,
    __EGL_DISPATCH_eglCreateSyncKHR,
    __EGL_DISPATCH_eglCreateWaylandBufferFromImageWL,
    __EGL_DISPATCH_eglCreateWindowSurface,
    __EGL_DISPATCH_eglDestroyContext,
    __EGL_DISPATCH_eglDestroyImage,
    __EGL_DISPATCH_eglDestroyImageKHR,
    __EGL_DISPATCH_eglDestroySurface,
    __EGL_DISPATCH_eglDestroySync,
    __EGL_DISPATCH_eglDestroySyncKHR,
    __EGL_DISPATCH_eglDupNativeFenceFDANDROID,
    __EGL_DISPATCH_eglExportDMABUFImageMESA,
    __EGL_DISPATCH_eglExportDMABUFImageQueryMESA,
    __EGL_DISPATCH_eglExportDRMImageMESA,
    __EGL_DISPATCH_eglGetConfigAttrib,
    __EGL_DISPATCH_eglGetConfigs,
    __EGL_DISPATCH_eglGetCurrentContext,
    __EGL_DISPATCH_eglGetCurrentDisplay,
    __EGL_DISPATCH_eglGetCurrentSurface,
    __EGL_DISPATCH_eglGetDisplay,
    __EGL_DISPATCH_eglGetError,
    __EGL_DISPATCH_eglGetPlatformDisplay,
    __EGL_DISPATCH_eglGetPlatformDisplayEXT,
    __EGL_DISPATCH_eglGetProcAddress,
    __EGL_DISPATCH_eglGetSyncAttrib,
    __EGL_DISPATCH_eglGetSyncAttribKHR,
    __EGL_DISPATCH_eglGetSyncValuesCHROMIUM,
    __EGL_DISPATCH_eglInitialize,
    __EGL_DISPATCH_eglMakeCurrent,
    __EGL_DISPATCH_eglPostSubBufferNV,
    __EGL_DISPATCH_eglQueryAPI,
    __EGL_DISPATCH_eglQueryContext,
    __EGL_DISPATCH_eglQueryDeviceAttribEXT,
    __EGL_DISPATCH_eglQueryDeviceStringEXT,
    __EGL_DISPATCH_eglQueryDevicesEXT,
    __EGL_DISPATCH_eglQueryDisplayAttribEXT,
    __EGL_DISPATCH_eglQueryDmaBufFormatsEXT,
    __EGL_DISPATCH_eglQueryDmaBufModifiersEXT,
    __EGL_DISPATCH_eglQueryString,
    __EGL_DISPATCH_eglQuerySurface,
    __EGL_DISPATCH_eglQueryWaylandBufferWL,
    __EGL_DISPATCH_eglReleaseTexImage,
    __EGL_DISPATCH_eglReleaseThread,
    __EGL_DISPATCH_eglSetBlobCacheFuncsANDROID,
    __EGL_DISPATCH_eglSignalSyncKHR,
    __EGL_DISPATCH_eglSurfaceAttrib,
    __EGL_DISPATCH_eglSwapBuffers,
    __EGL_DISPATCH_eglSwapBuffersRegionNOK,
    __EGL_DISPATCH_eglSwapBuffersWithDamageEXT,
    __EGL_DISPATCH_eglSwapBuffersWithDamageKHR,
    __EGL_DISPATCH_eglSwapInterval,
    __EGL_DISPATCH_eglTerminate,
    __EGL_DISPATCH_eglUnbindWaylandDisplayWL,
    __EGL_DISPATCH_eglWaitClient,
    __EGL_DISPATCH_eglWaitGL,
    __EGL_DISPATCH_eglWaitNative,
    __EGL_DISPATCH_eglWaitSync,
    __EGL_DISPATCH_eglWaitSyncKHR,
    __EGL_DISPATCH_COUNT
};

#ifdef __cplusplus
}
#endif
#endif // G_EGLDISPATCH_STUBS_H
