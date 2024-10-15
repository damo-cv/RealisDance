#include <string.h>
#include <assert.h>

#include <glvnd/libeglabi.h>

#include "eglcurrent.h"
#include "egldispatchstubs.h"
#include "eglglobals.h"

static const __EGLapiExports *__eglGLVNDApiExports = NULL;

static const char * EGLAPIENTRY
__eglGLVNDQueryString(EGLDisplay dpy, EGLenum name)
{
   // For client extensions, return the list of non-platform extensions. The
   // platform extensions are returned by __eglGLVNDGetVendorString.
   if (dpy == EGL_NO_DISPLAY && name == EGL_EXTENSIONS)
      return _eglGlobal.ClientOnlyExtensionString;

   // For everything else, forward to the normal eglQueryString function.
   return eglQueryString(dpy, name);
}

static const char *
__eglGLVNDGetVendorString(int name)
{
   if (name == __EGL_VENDOR_STRING_PLATFORM_EXTENSIONS) {
      const char *str = _eglGlobal.PlatformExtensionString;
      // The platform extension string may have a leading space. If it does,
      // then skip over it.
      while (*str == ' ') {
         str++;
      }
      return str;
   }

   return NULL;
}

static EGLDisplay
__eglGLVNDGetPlatformDisplay(EGLenum platform, void *native_display,
      const EGLAttrib *attrib_list)
{
   if (platform == EGL_NONE) {
      assert(native_display == (void *) EGL_DEFAULT_DISPLAY);
      assert(attrib_list == NULL);
      return eglGetDisplay((EGLNativeDisplayType) native_display);
   } else {
      return eglGetPlatformDisplay(platform, native_display, attrib_list);
   }
}

static void *
__eglGLVNDGetProcAddress(const char *procName)
{
   if (strcmp(procName, "eglQueryString") == 0)
      return (void *) __eglGLVNDQueryString;

   return (void *) eglGetProcAddress(procName);
}

EGLAPI EGLBoolean
__egl_Main(uint32_t version, const __EGLapiExports *exports,
     __EGLvendorInfo *vendor, __EGLapiImports *imports)
{
   if (EGL_VENDOR_ABI_GET_MAJOR_VERSION(version) !=
       EGL_VENDOR_ABI_MAJOR_VERSION)
      return EGL_FALSE;

   __eglGLVNDApiExports = exports;
   __eglInitDispatchStubs(exports);

   imports->getPlatformDisplay = __eglGLVNDGetPlatformDisplay;
   imports->getSupportsAPI = _eglIsApiValid;
   imports->getVendorString = __eglGLVNDGetVendorString;
   imports->getProcAddress = __eglGLVNDGetProcAddress;
   imports->getDispatchAddress = __eglDispatchFindDispatchFunction;
   imports->setDispatchIndex = __eglSetDispatchIndex;

   return EGL_TRUE;
}

