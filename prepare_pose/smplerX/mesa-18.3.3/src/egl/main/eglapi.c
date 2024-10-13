/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
 * Copyright 2009-2010 Chia-I Wu <olvaffe@gmail.com>
 * Copyright 2010-2011 LunarG, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


/**
 * Public EGL API entrypoints
 *
 * Generally, we use the EGLDisplay parameter as a key to lookup the
 * appropriate device driver handle, then jump though the driver's
 * dispatch table to handle the function.
 *
 * That allows us the option of supporting multiple, simultaneous,
 * heterogeneous hardware devices in the future.
 *
 * The EGLDisplay, EGLConfig, EGLContext and EGLSurface types are
 * opaque handles. Internal objects are linked to a display to
 * create the handles.
 *
 * For each public API entry point, the opaque handles are looked up
 * before being dispatched to the drivers.  When it fails to look up
 * a handle, one of
 *
 * EGL_BAD_DISPLAY
 * EGL_BAD_CONFIG
 * EGL_BAD_CONTEXT
 * EGL_BAD_SURFACE
 * EGL_BAD_SCREEN_MESA
 * EGL_BAD_MODE_MESA
 *
 * is generated and the driver function is not called. An
 * uninitialized EGLDisplay has no driver associated with it. When
 * such display is detected,
 *
 * EGL_NOT_INITIALIZED
 *
 * is generated.
 *
 * Some of the entry points use current display, context, or surface
 * implicitly.  For such entry points, the implicit objects are also
 * checked before calling the driver function.  Other than the
 * errors listed above,
 *
 * EGL_BAD_CURRENT_SURFACE
 *
 * may also be generated.
 *
 * Notes on naming conventions:
 *
 * eglFooBar    - public EGL function
 * EGL_FOO_BAR  - public EGL token
 * EGLDatatype  - public EGL datatype
 *
 * _eglFooBar   - private EGL function
 * _EGLDatatype - private EGL datatype, typedef'd struct
 * _egl_struct  - private EGL struct, non-typedef'd
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c99_compat.h"
#include "c11/threads.h"
#include "util/macros.h"

#include "eglglobals.h"
#include "eglcontext.h"
#include "egldisplay.h"
#include "egltypedefs.h"
#include "eglcurrent.h"
#include "egldevice.h"
#include "egldriver.h"
#include "eglsurface.h"
#include "eglconfig.h"
#include "eglimage.h"
#include "eglsync.h"

#include "GL/mesa_glinterop.h"

/**
 * Macros to help return an API entrypoint.
 *
 * These macros will unlock the display and record the error code.
 */
#define RETURN_EGL_ERROR(disp, err, ret)        \
   do {                                         \
      if (disp)                                 \
         _eglUnlockDisplay(disp);               \
      /* EGL error codes are non-zero */        \
      if (err)                                  \
         _eglError(err, __func__);              \
      return ret;                               \
   } while (0)

#define RETURN_EGL_SUCCESS(disp, ret) \
   RETURN_EGL_ERROR(disp, EGL_SUCCESS, ret)

/* record EGL_SUCCESS only when ret evaluates to true */
#define RETURN_EGL_EVAL(disp, ret) \
   RETURN_EGL_ERROR(disp, (ret) ? EGL_SUCCESS : 0, ret)


/*
 * A bunch of macros and checks to simplify error checking.
 */

#define _EGL_CHECK_DISPLAY(disp, ret, drv)         \
   do {                                            \
      drv = _eglCheckDisplay(disp, __func__);      \
      if (!drv)                                    \
         RETURN_EGL_ERROR(disp, 0, ret);           \
   } while (0)

#define _EGL_CHECK_OBJECT(disp, type, obj, ret, drv)      \
   do {                                                   \
      drv = _eglCheck ## type(disp, obj, __func__);       \
      if (!drv)                                           \
         RETURN_EGL_ERROR(disp, 0, ret);                  \
   } while (0)

#define _EGL_CHECK_SURFACE(disp, surf, ret, drv) \
   _EGL_CHECK_OBJECT(disp, Surface, surf, ret, drv)

#define _EGL_CHECK_CONTEXT(disp, context, ret, drv) \
   _EGL_CHECK_OBJECT(disp, Context, context, ret, drv)

#define _EGL_CHECK_CONFIG(disp, conf, ret, drv) \
   _EGL_CHECK_OBJECT(disp, Config, conf, ret, drv)

#define _EGL_CHECK_SYNC(disp, s, ret, drv) \
   _EGL_CHECK_OBJECT(disp, Sync, s, ret, drv)


struct _egl_entrypoint {
   const char *name;
   _EGLProc function;
};


static inline _EGLDriver *
_eglCheckDisplay(_EGLDisplay *disp, const char *msg)
{
   if (!disp) {
      _eglError(EGL_BAD_DISPLAY, msg);
      return NULL;
   }
   if (!disp->Initialized) {
      _eglError(EGL_NOT_INITIALIZED, msg);
      return NULL;
   }
   return disp->Driver;
}


static inline _EGLDriver *
_eglCheckSurface(_EGLDisplay *disp, _EGLSurface *surf, const char *msg)
{
   _EGLDriver *drv = _eglCheckDisplay(disp, msg);
   if (!drv)
      return NULL;
   if (!surf) {
      _eglError(EGL_BAD_SURFACE, msg);
      return NULL;
   }
   return drv;
}


static inline _EGLDriver *
_eglCheckContext(_EGLDisplay *disp, _EGLContext *context, const char *msg)
{
   _EGLDriver *drv = _eglCheckDisplay(disp, msg);
   if (!drv)
      return NULL;
   if (!context) {
      _eglError(EGL_BAD_CONTEXT, msg);
      return NULL;
   }
   return drv;
}


static inline _EGLDriver *
_eglCheckConfig(_EGLDisplay *disp, _EGLConfig *conf, const char *msg)
{
   _EGLDriver *drv = _eglCheckDisplay(disp, msg);
   if (!drv)
      return NULL;
   if (!conf) {
      _eglError(EGL_BAD_CONFIG, msg);
      return NULL;
   }
   return drv;
}


static inline _EGLDriver *
_eglCheckSync(_EGLDisplay *disp, _EGLSync *s, const char *msg)
{
   _EGLDriver *drv = _eglCheckDisplay(disp, msg);
   if (!drv)
      return NULL;
   if (!s) {
      _eglError(EGL_BAD_PARAMETER, msg);
      return NULL;
   }
   return drv;
}


/**
 * Lookup and lock a display.
 */
static inline _EGLDisplay *
_eglLockDisplay(EGLDisplay display)
{
   _EGLDisplay *dpy = _eglLookupDisplay(display);
   if (dpy)
      mtx_lock(&dpy->Mutex);
   return dpy;
}


/**
 * Unlock a display.
 */
static inline void
_eglUnlockDisplay(_EGLDisplay *dpy)
{
   mtx_unlock(&dpy->Mutex);
}

static EGLBoolean
_eglSetFuncName(const char *funcName, _EGLDisplay *disp, EGLenum objectType, _EGLResource *object)
{
   _EGLThreadInfo *thr = _eglGetCurrentThread();
   if (!_eglIsCurrentThreadDummy()) {
      thr->CurrentFuncName = funcName;
      thr->CurrentObjectLabel = NULL;

      if (objectType == EGL_OBJECT_THREAD_KHR)
         thr->CurrentObjectLabel = thr->Label;
      else if (objectType == EGL_OBJECT_DISPLAY_KHR && disp)
         thr->CurrentObjectLabel = disp->Label;
      else if (object)
         thr->CurrentObjectLabel = object->Label;

      return EGL_TRUE;
   }

   _eglDebugReport(EGL_BAD_ALLOC, funcName, EGL_DEBUG_MSG_CRITICAL_KHR, NULL);
   return EGL_FALSE;
}

#define _EGL_FUNC_START(disp, objectType, object, ret) \
   do { \
      if (!_eglSetFuncName(__func__, disp, objectType, (_EGLResource *) object)) { \
         if (disp)                                 \
            _eglUnlockDisplay(disp);               \
         return ret; \
      } \
   } while(0)

/**
 * Convert an attribute list from EGLint[] to EGLAttrib[].
 *
 * Return an EGL error code. The output parameter out_attrib_list is modified
 * only on success.
 */
static EGLint
_eglConvertIntsToAttribs(const EGLint *int_list, EGLAttrib **out_attrib_list)
{
   size_t len = 0;
   EGLAttrib *attrib_list;

   if (int_list) {
      while (int_list[2*len] != EGL_NONE)
         ++len;
   }

   if (len == 0) {
      *out_attrib_list = NULL;
      return EGL_SUCCESS;
   }

   if (2*len + 1 > SIZE_MAX / sizeof(EGLAttrib))
      return EGL_BAD_ALLOC;

   attrib_list = malloc((2*len + 1) * sizeof(EGLAttrib));
   if (!attrib_list)
      return EGL_BAD_ALLOC;

   for (size_t i = 0; i < len; ++i) {
      attrib_list[2*i + 0] = int_list[2*i + 0];
      attrib_list[2*i + 1] = int_list[2*i + 1];
   }

   attrib_list[2*len] = EGL_NONE;

   *out_attrib_list = attrib_list;
   return EGL_SUCCESS;
}


static EGLint *
_eglConvertAttribsToInt(const EGLAttrib *attr_list)
{
   EGLint *int_attribs = NULL;

   /* Convert attributes from EGLAttrib[] to EGLint[] */
   if (attr_list) {
      int i, size = 0;

      while (attr_list[size] != EGL_NONE)
         size += 2;

      size += 1; /* add space for EGL_NONE */

      int_attribs = calloc(size, sizeof(int_attribs[0]));
      if (!int_attribs)
         return NULL;

      for (i = 0; i < size; i++)
         int_attribs[i] = attr_list[i];
   }
   return int_attribs;
}


/**
 * This is typically the first EGL function that an application calls.
 * It associates a private _EGLDisplay object to the native display.
 */
EGLDisplay EGLAPIENTRY
eglGetDisplay(EGLNativeDisplayType nativeDisplay)
{
   _EGLPlatformType plat;
   _EGLDisplay *dpy;
   void *native_display_ptr;

   _EGL_FUNC_START(NULL, EGL_OBJECT_THREAD_KHR, NULL, EGL_NO_DISPLAY);

   STATIC_ASSERT(sizeof(void*) == sizeof(nativeDisplay));
   native_display_ptr = (void*) nativeDisplay;

   plat = _eglGetNativePlatform(native_display_ptr);
   dpy = _eglFindDisplay(plat, native_display_ptr);
   return _eglGetDisplayHandle(dpy);
}

static EGLDisplay
_eglGetPlatformDisplayCommon(EGLenum platform, void *native_display,
                             const EGLAttrib *attrib_list)
{
   _EGLDisplay *dpy;

   switch (platform) {
#ifdef HAVE_X11_PLATFORM
   case EGL_PLATFORM_X11_EXT:
      dpy = _eglGetX11Display((Display*) native_display, attrib_list);
      break;
#endif
#ifdef HAVE_DRM_PLATFORM
   case EGL_PLATFORM_GBM_MESA:
      dpy = _eglGetGbmDisplay((struct gbm_device*) native_display,
                              attrib_list);
      break;
#endif
#ifdef HAVE_WAYLAND_PLATFORM
   case EGL_PLATFORM_WAYLAND_EXT:
      dpy = _eglGetWaylandDisplay((struct wl_display*) native_display,
                                  attrib_list);
      break;
#endif
#ifdef HAVE_SURFACELESS_PLATFORM
   case EGL_PLATFORM_SURFACELESS_MESA:
      dpy = _eglGetSurfacelessDisplay(native_display, attrib_list);
      break;
#endif
   default:
      RETURN_EGL_ERROR(NULL, EGL_BAD_PARAMETER, NULL);
   }

   return _eglGetDisplayHandle(dpy);
}

static EGLDisplay EGLAPIENTRY
eglGetPlatformDisplayEXT(EGLenum platform, void *native_display,
                         const EGLint *int_attribs)
{
   EGLAttrib *attrib_list;
   EGLDisplay display;

   _EGL_FUNC_START(NULL, EGL_OBJECT_THREAD_KHR, NULL, EGL_NO_DISPLAY);

   if (_eglConvertIntsToAttribs(int_attribs, &attrib_list) != EGL_SUCCESS)
      RETURN_EGL_ERROR(NULL, EGL_BAD_ALLOC, NULL);

   display = _eglGetPlatformDisplayCommon(platform, native_display, attrib_list);
   free(attrib_list);
   return display;
}

EGLDisplay EGLAPIENTRY
eglGetPlatformDisplay(EGLenum platform, void *native_display,
                      const EGLAttrib *attrib_list)
{
   _EGL_FUNC_START(NULL, EGL_OBJECT_THREAD_KHR, NULL, EGL_NO_DISPLAY);
   return _eglGetPlatformDisplayCommon(platform, native_display, attrib_list);
}

/**
 * Copy the extension into the string and update the string pointer.
 */
static EGLint
_eglAppendExtension(char **str, const char *ext)
{
   char *s = *str;
   size_t len = strlen(ext);

   if (s) {
      memcpy(s, ext, len);
      s[len++] = ' ';
      s[len] = '\0';

      *str += len;
   }
   else {
      len++;
   }

   return (EGLint) len;
}

/**
 * Examine the individual extension enable/disable flags and recompute
 * the driver's Extensions string.
 */
static void
_eglCreateExtensionsString(_EGLDisplay *dpy)
{
#define _EGL_CHECK_EXTENSION(ext)                                          \
   do {                                                                    \
      if (dpy->Extensions.ext) {                                           \
         _eglAppendExtension(&exts, "EGL_" #ext);                          \
         assert(exts <= dpy->ExtensionsString + _EGL_MAX_EXTENSIONS_LEN);  \
      }                                                                    \
   } while (0)

   char *exts = dpy->ExtensionsString;

   /* Please keep these sorted alphabetically. */
   _EGL_CHECK_EXTENSION(ANDROID_blob_cache);
   _EGL_CHECK_EXTENSION(ANDROID_framebuffer_target);
   _EGL_CHECK_EXTENSION(ANDROID_image_native_buffer);
   _EGL_CHECK_EXTENSION(ANDROID_native_fence_sync);
   _EGL_CHECK_EXTENSION(ANDROID_recordable);

   _EGL_CHECK_EXTENSION(CHROMIUM_sync_control);

   _EGL_CHECK_EXTENSION(EXT_buffer_age);
   _EGL_CHECK_EXTENSION(EXT_create_context_robustness);
   _EGL_CHECK_EXTENSION(EXT_image_dma_buf_import);
   _EGL_CHECK_EXTENSION(EXT_image_dma_buf_import_modifiers);
   _EGL_CHECK_EXTENSION(EXT_surface_CTA861_3_metadata);
   _EGL_CHECK_EXTENSION(EXT_surface_SMPTE2086_metadata);
   _EGL_CHECK_EXTENSION(EXT_swap_buffers_with_damage);

   _EGL_CHECK_EXTENSION(IMG_context_priority);

   _EGL_CHECK_EXTENSION(KHR_cl_event2);
   _EGL_CHECK_EXTENSION(KHR_config_attribs);
   _EGL_CHECK_EXTENSION(KHR_context_flush_control);
   _EGL_CHECK_EXTENSION(KHR_create_context);
   _EGL_CHECK_EXTENSION(KHR_create_context_no_error);
   _EGL_CHECK_EXTENSION(KHR_fence_sync);
   _EGL_CHECK_EXTENSION(KHR_get_all_proc_addresses);
   _EGL_CHECK_EXTENSION(KHR_gl_colorspace);
   _EGL_CHECK_EXTENSION(KHR_gl_renderbuffer_image);
   _EGL_CHECK_EXTENSION(KHR_gl_texture_2D_image);
   _EGL_CHECK_EXTENSION(KHR_gl_texture_3D_image);
   _EGL_CHECK_EXTENSION(KHR_gl_texture_cubemap_image);
   if (dpy->Extensions.KHR_image_base && dpy->Extensions.KHR_image_pixmap)
      dpy->Extensions.KHR_image = EGL_TRUE;
   _EGL_CHECK_EXTENSION(KHR_image);
   _EGL_CHECK_EXTENSION(KHR_image_base);
   _EGL_CHECK_EXTENSION(KHR_image_pixmap);
   _EGL_CHECK_EXTENSION(KHR_mutable_render_buffer);
   _EGL_CHECK_EXTENSION(KHR_no_config_context);
   _EGL_CHECK_EXTENSION(KHR_partial_update);
   _EGL_CHECK_EXTENSION(KHR_reusable_sync);
   _EGL_CHECK_EXTENSION(KHR_surfaceless_context);
   if (dpy->Extensions.EXT_swap_buffers_with_damage)
      _eglAppendExtension(&exts, "EGL_KHR_swap_buffers_with_damage");
   _EGL_CHECK_EXTENSION(EXT_pixel_format_float);
   _EGL_CHECK_EXTENSION(KHR_wait_sync);

   if (dpy->Extensions.KHR_no_config_context)
      _eglAppendExtension(&exts, "EGL_MESA_configless_context");
   _EGL_CHECK_EXTENSION(MESA_drm_image);
   _EGL_CHECK_EXTENSION(MESA_image_dma_buf_export);

   _EGL_CHECK_EXTENSION(NOK_swap_region);
   _EGL_CHECK_EXTENSION(NOK_texture_from_pixmap);

   _EGL_CHECK_EXTENSION(NV_post_sub_buffer);

   _EGL_CHECK_EXTENSION(WL_bind_wayland_display);
   _EGL_CHECK_EXTENSION(WL_create_wayland_buffer_from_image);

#undef _EGL_CHECK_EXTENSION
}

static void
_eglCreateAPIsString(_EGLDisplay *dpy)
{
#define addstr(str) \
   { \
      const size_t old_len = strlen(dpy->ClientAPIsString); \
      const size_t add_len = sizeof(str); \
      const size_t max_len = sizeof(dpy->ClientAPIsString) - 1; \
      if (old_len + add_len <= max_len) \
         strcat(dpy->ClientAPIsString, str " "); \
      else \
         assert(!"dpy->ClientAPIsString is not large enough"); \
   }

   if (dpy->ClientAPIs & EGL_OPENGL_BIT)
      addstr("OpenGL");

   if (dpy->ClientAPIs & EGL_OPENGL_ES_BIT ||
       dpy->ClientAPIs & EGL_OPENGL_ES2_BIT ||
       dpy->ClientAPIs & EGL_OPENGL_ES3_BIT_KHR) {
      addstr("OpenGL_ES");
   }

   if (dpy->ClientAPIs & EGL_OPENVG_BIT)
      addstr("OpenVG");

#undef addstr
}

static void
_eglComputeVersion(_EGLDisplay *disp)
{
   disp->Version = 14;

   if (disp->Extensions.KHR_fence_sync &&
       disp->Extensions.KHR_cl_event2 &&
       disp->Extensions.KHR_wait_sync &&
       disp->Extensions.KHR_image_base &&
       disp->Extensions.KHR_gl_texture_2D_image &&
       disp->Extensions.KHR_gl_texture_3D_image &&
       disp->Extensions.KHR_gl_texture_cubemap_image &&
       disp->Extensions.KHR_gl_renderbuffer_image &&
       disp->Extensions.KHR_create_context &&
       disp->Extensions.EXT_create_context_robustness &&
       disp->Extensions.KHR_get_all_proc_addresses &&
       disp->Extensions.KHR_gl_colorspace &&
       disp->Extensions.KHR_surfaceless_context)
      disp->Version = 15;
}

/**
 * This is typically the second EGL function that an application calls.
 * Here we load/initialize the actual hardware driver.
 */
EGLBoolean EGLAPIENTRY
eglInitialize(EGLDisplay dpy, EGLint *major, EGLint *minor)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   if (!disp)
      RETURN_EGL_ERROR(NULL, EGL_BAD_DISPLAY, EGL_FALSE);

   if (!disp->Initialized) {
      if (!_eglMatchDriver(disp))
         RETURN_EGL_ERROR(disp, EGL_NOT_INITIALIZED, EGL_FALSE);

      /* limit to APIs supported by core */
      disp->ClientAPIs &= _EGL_API_ALL_BITS;

      /* EGL_KHR_get_all_proc_addresses is a corner-case extension. The spec
       * classifies it as an EGL display extension, though conceptually it's an
       * EGL client extension.
       *
       * From the EGL_KHR_get_all_proc_addresses spec:
       *
       *    The EGL implementation must expose the name
       *    EGL_KHR_client_get_all_proc_addresses if and only if it exposes
       *    EGL_KHR_get_all_proc_addresses and supports
       *    EGL_EXT_client_extensions.
       *
       * Mesa unconditionally exposes both client extensions mentioned above,
       * so the spec requires that each EGLDisplay unconditionally expose
       * EGL_KHR_get_all_proc_addresses also.
       */
      disp->Extensions.KHR_get_all_proc_addresses = EGL_TRUE;

      /* Extensions is used to provide EGL 1.3 functionality for 1.2 aware
       * programs. It is driver agnostic and handled in the main EGL code.
       */
      disp->Extensions.KHR_config_attribs = EGL_TRUE;

      _eglComputeVersion(disp);
      _eglCreateExtensionsString(disp);
      _eglCreateAPIsString(disp);
      snprintf(disp->VersionString, sizeof(disp->VersionString),
               "%d.%d", disp->Version / 10, disp->Version % 10);
   }

   /* Update applications version of major and minor if not NULL */
   if ((major != NULL) && (minor != NULL)) {
      *major = disp->Version / 10;
      *minor = disp->Version % 10;
   }

   RETURN_EGL_SUCCESS(disp, EGL_TRUE);
}


EGLBoolean EGLAPIENTRY
eglTerminate(EGLDisplay dpy)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   if (!disp)
      RETURN_EGL_ERROR(NULL, EGL_BAD_DISPLAY, EGL_FALSE);

   if (disp->Initialized) {
      _EGLDriver *drv = disp->Driver;

      drv->API.Terminate(drv, disp);
      /* do not reset disp->Driver */
      disp->ClientAPIsString[0] = 0;
      disp->Initialized = EGL_FALSE;
   }

   RETURN_EGL_SUCCESS(disp, EGL_TRUE);
}


const char * EGLAPIENTRY
eglQueryString(EGLDisplay dpy, EGLint name)
{
   _EGLDisplay *disp;
   _EGLDriver *drv;

   if (dpy == EGL_NO_DISPLAY && name == EGL_EXTENSIONS) {
      const char *ret = _eglGetClientExtensionString();
      if (ret != NULL)
         RETURN_EGL_SUCCESS(NULL, ret);
      else
         RETURN_EGL_ERROR(NULL, EGL_BAD_ALLOC, NULL);
   }

   disp = _eglLockDisplay(dpy);
   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, NULL);
   _EGL_CHECK_DISPLAY(disp, NULL, drv);

   switch (name) {
   case EGL_VENDOR:
      RETURN_EGL_SUCCESS(disp, _EGL_VENDOR_STRING);
   case EGL_VERSION:
      RETURN_EGL_SUCCESS(disp, disp->VersionString);
   case EGL_EXTENSIONS:
      RETURN_EGL_SUCCESS(disp, disp->ExtensionsString);
   case EGL_CLIENT_APIS:
      RETURN_EGL_SUCCESS(disp, disp->ClientAPIsString);
   default:
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, NULL);
   }
}


EGLBoolean EGLAPIENTRY
eglGetConfigs(EGLDisplay dpy, EGLConfig *configs,
              EGLint config_size, EGLint *num_config)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);
   ret = drv->API.GetConfigs(drv, disp, configs, config_size, num_config);

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglChooseConfig(EGLDisplay dpy, const EGLint *attrib_list, EGLConfig *configs,
                EGLint config_size, EGLint *num_config)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);
   ret = drv->API.ChooseConfig(drv, disp, attrib_list, configs,
                               config_size, num_config);

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglGetConfigAttrib(EGLDisplay dpy, EGLConfig config,
                   EGLint attribute, EGLint *value)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLConfig *conf = _eglLookupConfig(config, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   _EGL_CHECK_CONFIG(disp, conf, EGL_FALSE, drv);
   ret = drv->API.GetConfigAttrib(drv, disp, conf, attribute, value);

   RETURN_EGL_EVAL(disp, ret);
}


EGLContext EGLAPIENTRY
eglCreateContext(EGLDisplay dpy, EGLConfig config, EGLContext share_list,
                 const EGLint *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLConfig *conf = _eglLookupConfig(config, disp);
   _EGLContext *share = _eglLookupContext(share_list, disp);
   _EGLDriver *drv;
   _EGLContext *context;
   EGLContext ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_CONTEXT);

   _EGL_CHECK_DISPLAY(disp, EGL_NO_CONTEXT, drv);

   if (config != EGL_NO_CONFIG_KHR)
      _EGL_CHECK_CONFIG(disp, conf, EGL_NO_CONTEXT, drv);
   else if (!disp->Extensions.KHR_no_config_context)
      RETURN_EGL_ERROR(disp, EGL_BAD_CONFIG, EGL_NO_CONTEXT);

   if (!share && share_list != EGL_NO_CONTEXT)
      RETURN_EGL_ERROR(disp, EGL_BAD_CONTEXT, EGL_NO_CONTEXT);

   context = drv->API.CreateContext(drv, disp, conf, share, attrib_list);
   ret = (context) ? _eglLinkContext(context) : EGL_NO_CONTEXT;

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglDestroyContext(EGLDisplay dpy, EGLContext ctx)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLContext *context = _eglLookupContext(ctx, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_CONTEXT_KHR, context, EGL_FALSE);

   _EGL_CHECK_CONTEXT(disp, context, EGL_FALSE, drv);
   _eglUnlinkContext(context);
   ret = drv->API.DestroyContext(drv, disp, context);

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglMakeCurrent(EGLDisplay dpy, EGLSurface draw, EGLSurface read,
               EGLContext ctx)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLContext *context = _eglLookupContext(ctx, disp);
   _EGLSurface *draw_surf = _eglLookupSurface(draw, disp);
   _EGLSurface *read_surf = _eglLookupSurface(read, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_CONTEXT_KHR, context, EGL_FALSE);

   if (!disp)
      RETURN_EGL_ERROR(disp, EGL_BAD_DISPLAY, EGL_FALSE);
   drv = disp->Driver;

   /* display is allowed to be uninitialized under certain condition */
   if (!disp->Initialized) {
      if (draw != EGL_NO_SURFACE || read != EGL_NO_SURFACE ||
          ctx != EGL_NO_CONTEXT)
         RETURN_EGL_ERROR(disp, EGL_BAD_DISPLAY, EGL_FALSE);
   }
   if (!drv)
      RETURN_EGL_SUCCESS(disp, EGL_TRUE);

   if (!context && ctx != EGL_NO_CONTEXT)
      RETURN_EGL_ERROR(disp, EGL_BAD_CONTEXT, EGL_FALSE);
   if (!draw_surf || !read_surf) {
      /* From the EGL 1.4 (20130211) spec:
       *
       *    To release the current context without assigning a new one, set ctx
       *    to EGL_NO_CONTEXT and set draw and read to EGL_NO_SURFACE.
       */
      if (!disp->Extensions.KHR_surfaceless_context && ctx != EGL_NO_CONTEXT)
         RETURN_EGL_ERROR(disp, EGL_BAD_SURFACE, EGL_FALSE);

      if ((!draw_surf && draw != EGL_NO_SURFACE) ||
          (!read_surf && read != EGL_NO_SURFACE))
         RETURN_EGL_ERROR(disp, EGL_BAD_SURFACE, EGL_FALSE);
      if (draw_surf || read_surf)
         RETURN_EGL_ERROR(disp, EGL_BAD_MATCH, EGL_FALSE);
   }

   /*    If a native window underlying either draw or read is no longer valid,
    *    an EGL_BAD_NATIVE_WINDOW error is generated.
    */
   if (draw_surf && draw_surf->Lost)
      RETURN_EGL_ERROR(disp, EGL_BAD_NATIVE_WINDOW, EGL_FALSE);
   if (read_surf && read_surf->Lost)
      RETURN_EGL_ERROR(disp, EGL_BAD_NATIVE_WINDOW, EGL_FALSE);

   ret = drv->API.MakeCurrent(drv, disp, draw_surf, read_surf, context);

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglQueryContext(EGLDisplay dpy, EGLContext ctx,
                EGLint attribute, EGLint *value)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLContext *context = _eglLookupContext(ctx, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_CONTEXT_KHR, context, EGL_FALSE);

   _EGL_CHECK_CONTEXT(disp, context, EGL_FALSE, drv);
   ret = drv->API.QueryContext(drv, disp, context, attribute, value);

   RETURN_EGL_EVAL(disp, ret);
}


static EGLSurface
_eglCreateWindowSurfaceCommon(_EGLDisplay *disp, EGLConfig config,
                              void *native_window, const EGLint *attrib_list)
{
   _EGLConfig *conf = _eglLookupConfig(config, disp);
   _EGLDriver *drv;
   _EGLSurface *surf;
   EGLSurface ret;


   if (native_window == NULL)
      RETURN_EGL_ERROR(disp, EGL_BAD_NATIVE_WINDOW, EGL_NO_SURFACE);

#ifdef HAVE_SURFACELESS_PLATFORM
   if (disp && disp->Platform == _EGL_PLATFORM_SURFACELESS) {
      /* From the EGL_MESA_platform_surfaceless spec (v1):
       *
       *    eglCreatePlatformWindowSurface fails when called with a <display>
       *    that belongs to the surfaceless platform. It returns
       *    EGL_NO_SURFACE and generates EGL_BAD_NATIVE_WINDOW. The
       *    justification for this unconditional failure is that the
       *    surfaceless platform has no native windows, and therefore the
       *    <native_window> parameter is always invalid.
       *
       * This check must occur before checking the EGLConfig, which emits
       * EGL_BAD_CONFIG.
       */
      RETURN_EGL_ERROR(disp, EGL_BAD_NATIVE_WINDOW, EGL_NO_SURFACE);
   }
#endif

   _EGL_CHECK_CONFIG(disp, conf, EGL_NO_SURFACE, drv);

   if ((conf->SurfaceType & EGL_WINDOW_BIT) == 0)
      RETURN_EGL_ERROR(disp, EGL_BAD_MATCH, EGL_NO_SURFACE);

   surf = drv->API.CreateWindowSurface(drv, disp, conf, native_window,
                                       attrib_list);
   ret = (surf) ? _eglLinkSurface(surf) : EGL_NO_SURFACE;

   RETURN_EGL_EVAL(disp, ret);
}


EGLSurface EGLAPIENTRY
eglCreateWindowSurface(EGLDisplay dpy, EGLConfig config,
                       EGLNativeWindowType window, const EGLint *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_SURFACE);
   STATIC_ASSERT(sizeof(void*) == sizeof(window));
   return _eglCreateWindowSurfaceCommon(disp, config, (void*) window,
                                        attrib_list);
}

static void *
_fixupNativeWindow(_EGLDisplay *disp, void *native_window)
{
#ifdef HAVE_X11_PLATFORM
   if (disp && disp->Platform == _EGL_PLATFORM_X11 && native_window != NULL) {
      /* The `native_window` parameter for the X11 platform differs between
       * eglCreateWindowSurface() and eglCreatePlatformPixmapSurfaceEXT(). In
       * eglCreateWindowSurface(), the type of `native_window` is an Xlib
       * `Window`. In eglCreatePlatformWindowSurfaceEXT(), the type is
       * `Window*`.  Convert `Window*` to `Window` because that's what
       * dri2_x11_create_window_surface() expects.
       */
      return (void *)(* (Window*) native_window);
   }
#endif
   return native_window;
}

static EGLSurface EGLAPIENTRY
eglCreatePlatformWindowSurfaceEXT(EGLDisplay dpy, EGLConfig config,
                                  void *native_window,
                                  const EGLint *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);

   native_window = _fixupNativeWindow(disp, native_window);

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_SURFACE);
   return _eglCreateWindowSurfaceCommon(disp, config, native_window,
                                        attrib_list);
}


EGLSurface EGLAPIENTRY
eglCreatePlatformWindowSurface(EGLDisplay dpy, EGLConfig config,
                               void *native_window,
                               const EGLAttrib *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   EGLSurface surface;
   EGLint *int_attribs;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_SURFACE);

   int_attribs = _eglConvertAttribsToInt(attrib_list);
   if (attrib_list && !int_attribs)
      RETURN_EGL_ERROR(disp, EGL_BAD_ALLOC, EGL_NO_SURFACE);

   native_window = _fixupNativeWindow(disp, native_window);
   surface = _eglCreateWindowSurfaceCommon(disp, config, native_window,
                                           int_attribs);
   free(int_attribs);
   return surface;
}

static void *
_fixupNativePixmap(_EGLDisplay *disp, void *native_pixmap)
{
#ifdef HAVE_X11_PLATFORM
   /* The `native_pixmap` parameter for the X11 platform differs between
    * eglCreatePixmapSurface() and eglCreatePlatformPixmapSurfaceEXT(). In
    * eglCreatePixmapSurface(), the type of `native_pixmap` is an Xlib
    * `Pixmap`. In eglCreatePlatformPixmapSurfaceEXT(), the type is
    * `Pixmap*`.  Convert `Pixmap*` to `Pixmap` because that's what
    * dri2_x11_create_pixmap_surface() expects.
    */
   if (disp && disp->Platform == _EGL_PLATFORM_X11 && native_pixmap != NULL)
      return (void *)(* (Pixmap*) native_pixmap);
#endif
   return native_pixmap;
}

static EGLSurface
_eglCreatePixmapSurfaceCommon(_EGLDisplay *disp, EGLConfig config,
                              void *native_pixmap, const EGLint *attrib_list)
{
   _EGLConfig *conf = _eglLookupConfig(config, disp);
   _EGLDriver *drv;
   _EGLSurface *surf;
   EGLSurface ret;

#if HAVE_SURFACELESS_PLATFORM
   if (disp && disp->Platform == _EGL_PLATFORM_SURFACELESS) {
      /* From the EGL_MESA_platform_surfaceless spec (v1):
       *
       *   [Like eglCreatePlatformWindowSurface,] eglCreatePlatformPixmapSurface
       *   also fails when called with a <display> that belongs to the
       *   surfaceless platform.  It returns EGL_NO_SURFACE and generates
       *   EGL_BAD_NATIVE_PIXMAP.
       *
       * This check must occur before checking the EGLConfig, which emits
       * EGL_BAD_CONFIG.
       */
      RETURN_EGL_ERROR(disp, EGL_BAD_NATIVE_PIXMAP, EGL_NO_SURFACE);
   }
#endif

   _EGL_CHECK_CONFIG(disp, conf, EGL_NO_SURFACE, drv);

   if ((conf->SurfaceType & EGL_PIXMAP_BIT) == 0)
      RETURN_EGL_ERROR(disp, EGL_BAD_MATCH, EGL_NO_SURFACE);

   if (native_pixmap == NULL)
      RETURN_EGL_ERROR(disp, EGL_BAD_NATIVE_PIXMAP, EGL_NO_SURFACE);

   surf = drv->API.CreatePixmapSurface(drv, disp, conf, native_pixmap,
                                       attrib_list);
   ret = (surf) ? _eglLinkSurface(surf) : EGL_NO_SURFACE;

   RETURN_EGL_EVAL(disp, ret);
}


EGLSurface EGLAPIENTRY
eglCreatePixmapSurface(EGLDisplay dpy, EGLConfig config,
                       EGLNativePixmapType pixmap, const EGLint *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_SURFACE);
   STATIC_ASSERT(sizeof(void*) == sizeof(pixmap));
   return _eglCreatePixmapSurfaceCommon(disp, config, (void*) pixmap,
                                        attrib_list);
}

static EGLSurface EGLAPIENTRY
eglCreatePlatformPixmapSurfaceEXT(EGLDisplay dpy, EGLConfig config,
                                  void *native_pixmap,
                                  const EGLint *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_SURFACE);
   native_pixmap = _fixupNativePixmap(disp, native_pixmap);
   return _eglCreatePixmapSurfaceCommon(disp, config, native_pixmap,
                                        attrib_list);
}


EGLSurface EGLAPIENTRY
eglCreatePlatformPixmapSurface(EGLDisplay dpy, EGLConfig config,
                               void *native_pixmap,
                               const EGLAttrib *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   EGLSurface surface;
   EGLint *int_attribs;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_SURFACE);

   int_attribs = _eglConvertAttribsToInt(attrib_list);
   if (attrib_list && !int_attribs)
      RETURN_EGL_ERROR(disp, EGL_BAD_ALLOC, EGL_NO_SURFACE);

   native_pixmap = _fixupNativePixmap(disp, native_pixmap);
   surface = _eglCreatePixmapSurfaceCommon(disp, config, native_pixmap,
                                           int_attribs);
   free(int_attribs);
   return surface;
}


EGLSurface EGLAPIENTRY
eglCreatePbufferSurface(EGLDisplay dpy, EGLConfig config,
                        const EGLint *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLConfig *conf = _eglLookupConfig(config, disp);
   _EGLDriver *drv;
   _EGLSurface *surf;
   EGLSurface ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_SURFACE);
   _EGL_CHECK_CONFIG(disp, conf, EGL_NO_SURFACE, drv);

   if ((conf->SurfaceType & EGL_PBUFFER_BIT) == 0)
      RETURN_EGL_ERROR(disp, EGL_BAD_MATCH, EGL_NO_SURFACE);

   surf = drv->API.CreatePbufferSurface(drv, disp, conf, attrib_list);
   ret = (surf) ? _eglLinkSurface(surf) : EGL_NO_SURFACE;

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglDestroySurface(EGLDisplay dpy, EGLSurface surface)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);
   _eglUnlinkSurface(surf);
   ret = drv->API.DestroySurface(drv, disp, surf);

   RETURN_EGL_EVAL(disp, ret);
}

EGLBoolean EGLAPIENTRY
eglQuerySurface(EGLDisplay dpy, EGLSurface surface,
                EGLint attribute, EGLint *value)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);
   ret = drv->API.QuerySurface(drv, disp, surf, attribute, value);

   RETURN_EGL_EVAL(disp, ret);
}

EGLBoolean EGLAPIENTRY
eglSurfaceAttrib(EGLDisplay dpy, EGLSurface surface,
                 EGLint attribute, EGLint value)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);
   ret = drv->API.SurfaceAttrib(drv, disp, surf, attribute, value);

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglBindTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);
   ret = drv->API.BindTexImage(drv, disp, surf, buffer);

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglReleaseTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);
   ret = drv->API.ReleaseTexImage(drv, disp, surf, buffer);

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglSwapInterval(EGLDisplay dpy, EGLint interval)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLContext *ctx = _eglGetCurrentContext();
   _EGLSurface *surf = ctx ? ctx->DrawSurface : NULL;
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);

   if (_eglGetContextHandle(ctx) == EGL_NO_CONTEXT ||
       ctx->Resource.Display != disp)
      RETURN_EGL_ERROR(disp, EGL_BAD_CONTEXT, EGL_FALSE);

   if (_eglGetSurfaceHandle(surf) == EGL_NO_SURFACE)
      RETURN_EGL_ERROR(disp, EGL_BAD_SURFACE, EGL_FALSE);

   if (surf->Type != EGL_WINDOW_BIT)
      RETURN_EGL_EVAL(disp, EGL_TRUE);

   interval = CLAMP(interval,
                    surf->Config->MinSwapInterval,
                    surf->Config->MaxSwapInterval);

   if (surf->SwapInterval != interval)
      ret = drv->API.SwapInterval(drv, disp, surf, interval);
   else
      ret = EGL_TRUE;

   if (ret)
      surf->SwapInterval = interval;

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglSwapBuffers(EGLDisplay dpy, EGLSurface surface)
{
   _EGLContext *ctx = _eglGetCurrentContext();
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);

   /* surface must be bound to current context in EGL 1.4 */
   #ifndef _EGL_BUILT_IN_DRIVER_HAIKU
   if (_eglGetContextHandle(ctx) == EGL_NO_CONTEXT ||
       surf != ctx->DrawSurface)
      RETURN_EGL_ERROR(disp, EGL_BAD_SURFACE, EGL_FALSE);
   #endif

   if (surf->Type != EGL_WINDOW_BIT)
      RETURN_EGL_EVAL(disp, EGL_TRUE);

   /* From the EGL 1.5 spec:
    *
    *    If eglSwapBuffers is called and the native window associated with
    *    surface is no longer valid, an EGL_BAD_NATIVE_WINDOW error is
    *    generated.
    */
   if (surf->Lost)
      RETURN_EGL_ERROR(disp, EGL_BAD_NATIVE_WINDOW, EGL_FALSE);

   ret = drv->API.SwapBuffers(drv, disp, surf);

   /* EGL_KHR_partial_update
    * Frame boundary successfully reached,
    * reset damage region and reset BufferAgeRead
    */
   if (ret) {
      surf->SetDamageRegionCalled = EGL_FALSE;
      surf->BufferAgeRead = EGL_FALSE;
   }

   RETURN_EGL_EVAL(disp, ret);
}


static EGLBoolean
_eglSwapBuffersWithDamageCommon(_EGLDisplay *disp, _EGLSurface *surf,
                                EGLint *rects, EGLint n_rects)
{
   _EGLContext *ctx = _eglGetCurrentContext();
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);

   /* surface must be bound to current context in EGL 1.4 */
   if (_eglGetContextHandle(ctx) == EGL_NO_CONTEXT ||
       surf != ctx->DrawSurface)
      RETURN_EGL_ERROR(disp, EGL_BAD_SURFACE, EGL_FALSE);

   if (surf->Type != EGL_WINDOW_BIT)
      RETURN_EGL_EVAL(disp, EGL_TRUE);

   if ((n_rects > 0 && rects == NULL) || n_rects < 0)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   ret = drv->API.SwapBuffersWithDamageEXT(drv, disp, surf, rects, n_rects);

   /* EGL_KHR_partial_update
    * Frame boundary successfully reached,
    * reset damage region and reset BufferAgeRead
    */
   if (ret) {
      surf->SetDamageRegionCalled = EGL_FALSE;
      surf->BufferAgeRead = EGL_FALSE;
   }

   RETURN_EGL_EVAL(disp, ret);
}

static EGLBoolean EGLAPIENTRY
eglSwapBuffersWithDamageEXT(EGLDisplay dpy, EGLSurface surface,
                            EGLint *rects, EGLint n_rects)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   return _eglSwapBuffersWithDamageCommon(disp, surf, rects, n_rects);
}

static EGLBoolean EGLAPIENTRY
eglSwapBuffersWithDamageKHR(EGLDisplay dpy, EGLSurface surface,
                            EGLint *rects, EGLint n_rects)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   return _eglSwapBuffersWithDamageCommon(disp, surf, rects, n_rects);
}

/**
 * Clamp the rectangles so that they lie within the surface.
 */

static void
_eglSetDamageRegionKHRClampRects(_EGLDisplay* disp, _EGLSurface* surf,
                                 EGLint *rects, EGLint n_rects)
{
   EGLint i;
   EGLint surf_height = surf->Height;
   EGLint surf_width = surf->Width;

   for (i = 0; i < (4 * n_rects); i += 4) {
      EGLint x1, y1, x2, y2;
      x1 = rects[i];
      y1 = rects[i + 1];
      x2 = rects[i + 2] + x1;
      y2 = rects[i + 3] + y1;

      rects[i] = CLAMP(x1, 0, surf_width);
      rects[i + 1] = CLAMP(y1, 0, surf_height);
      rects[i + 2] = CLAMP(x2, 0, surf_width) - rects[i];
      rects[i + 3] = CLAMP(y2, 0, surf_height) - rects[i + 1];
   }
}

static EGLBoolean EGLAPIENTRY
eglSetDamageRegionKHR(EGLDisplay dpy, EGLSurface surface,
                      EGLint *rects, EGLint n_rects)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   _EGLContext *ctx = _eglGetCurrentContext();
   _EGLDriver *drv;
   EGLBoolean ret;
   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);

   if (_eglGetContextHandle(ctx) == EGL_NO_CONTEXT ||
       surf->Type != EGL_WINDOW_BIT ||
       ctx->DrawSurface != surf ||
       surf->SwapBehavior != EGL_BUFFER_DESTROYED)
      RETURN_EGL_ERROR(disp, EGL_BAD_MATCH, EGL_FALSE);

   /* If the damage region is already set or
    * buffer age is not queried between
    * frame boundaries, throw bad access error
    */

   if (surf->SetDamageRegionCalled || !surf->BufferAgeRead)
      RETURN_EGL_ERROR(disp, EGL_BAD_ACCESS, EGL_FALSE);

   _eglSetDamageRegionKHRClampRects(disp, surf, rects, n_rects);
   ret = drv->API.SetDamageRegion(drv, disp, surf, rects, n_rects);

   if (ret)
      surf->SetDamageRegionCalled = EGL_TRUE;

   RETURN_EGL_EVAL(disp, ret);
}

EGLBoolean EGLAPIENTRY
eglCopyBuffers(EGLDisplay dpy, EGLSurface surface, EGLNativePixmapType target)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGLDriver *drv;
   EGLBoolean ret;
   void *native_pixmap_ptr;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);
   STATIC_ASSERT(sizeof(void*) == sizeof(target));
   native_pixmap_ptr = (void*) target;

   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);
   ret = drv->API.CopyBuffers(drv, disp, surf, native_pixmap_ptr);

   RETURN_EGL_EVAL(disp, ret);
}


static EGLBoolean
_eglWaitClientCommon(void)
{
   _EGLContext *ctx = _eglGetCurrentContext();
   _EGLDisplay *disp;
   _EGLDriver *drv;
   EGLBoolean ret;

   if (!ctx)
      RETURN_EGL_SUCCESS(NULL, EGL_TRUE);

   disp = ctx->Resource.Display;
   mtx_lock(&disp->Mutex);

   /* let bad current context imply bad current surface */
   if (_eglGetContextHandle(ctx) == EGL_NO_CONTEXT ||
       _eglGetSurfaceHandle(ctx->DrawSurface) == EGL_NO_SURFACE)
      RETURN_EGL_ERROR(disp, EGL_BAD_CURRENT_SURFACE, EGL_FALSE);

   /* a valid current context implies an initialized current display */
   assert(disp->Initialized);
   drv = disp->Driver;
   ret = drv->API.WaitClient(drv, disp, ctx);

   RETURN_EGL_EVAL(disp, ret);
}

EGLBoolean EGLAPIENTRY
eglWaitClient(void)
{
   _EGL_FUNC_START(NULL, EGL_OBJECT_CONTEXT_KHR, _eglGetCurrentContext(), EGL_FALSE);
   return _eglWaitClientCommon();
}

EGLBoolean EGLAPIENTRY
eglWaitGL(void)
{
   /* Since we only support OpenGL and GLES, eglWaitGL is equivalent to eglWaitClient. */
   _EGL_FUNC_START(NULL, EGL_OBJECT_CONTEXT_KHR, _eglGetCurrentContext(), EGL_FALSE);
   return _eglWaitClientCommon();
}


EGLBoolean EGLAPIENTRY
eglWaitNative(EGLint engine)
{
   _EGLContext *ctx = _eglGetCurrentContext();
   _EGLDisplay *disp;
   _EGLDriver *drv;
   EGLBoolean ret;

   if (!ctx)
      RETURN_EGL_SUCCESS(NULL, EGL_TRUE);

   _EGL_FUNC_START(NULL, EGL_OBJECT_THREAD_KHR, NULL, EGL_FALSE);

   disp = ctx->Resource.Display;
   mtx_lock(&disp->Mutex);

   /* let bad current context imply bad current surface */
   if (_eglGetContextHandle(ctx) == EGL_NO_CONTEXT ||
       _eglGetSurfaceHandle(ctx->DrawSurface) == EGL_NO_SURFACE)
      RETURN_EGL_ERROR(disp, EGL_BAD_CURRENT_SURFACE, EGL_FALSE);

   /* a valid current context implies an initialized current display */
   assert(disp->Initialized);
   drv = disp->Driver;
   ret = drv->API.WaitNative(drv, disp, engine);

   RETURN_EGL_EVAL(disp, ret);
}


EGLDisplay EGLAPIENTRY
eglGetCurrentDisplay(void)
{
   _EGLContext *ctx = _eglGetCurrentContext();
   EGLDisplay ret;

   ret = (ctx) ? _eglGetDisplayHandle(ctx->Resource.Display) : EGL_NO_DISPLAY;

   RETURN_EGL_SUCCESS(NULL, ret);
}


EGLContext EGLAPIENTRY
eglGetCurrentContext(void)
{
   _EGLContext *ctx = _eglGetCurrentContext();
   EGLContext ret;

   ret = _eglGetContextHandle(ctx);

   RETURN_EGL_SUCCESS(NULL, ret);
}


EGLSurface EGLAPIENTRY
eglGetCurrentSurface(EGLint readdraw)
{
   _EGLContext *ctx = _eglGetCurrentContext();
   EGLint err = EGL_SUCCESS;
   _EGLSurface *surf;
   EGLSurface ret;

   _EGL_FUNC_START(NULL, EGL_NONE, NULL, EGL_NO_SURFACE);

   if (!ctx)
      RETURN_EGL_SUCCESS(NULL, EGL_NO_SURFACE);

   switch (readdraw) {
   case EGL_DRAW:
      surf = ctx->DrawSurface;
      break;
   case EGL_READ:
      surf = ctx->ReadSurface;
      break;
   default:
      surf = NULL;
      err = EGL_BAD_PARAMETER;
      break;
   }

   ret = _eglGetSurfaceHandle(surf);

   RETURN_EGL_ERROR(NULL, err, ret);
}


EGLint EGLAPIENTRY
eglGetError(void)
{
   _EGLThreadInfo *t = _eglGetCurrentThread();
   EGLint e = t->LastError;
   if (!_eglIsCurrentThreadDummy())
      t->LastError = EGL_SUCCESS;
   return e;
}


/**
 ** EGL 1.2
 **/

/**
 * Specify the client API to use for subsequent calls including:
 *  eglCreateContext()
 *  eglGetCurrentContext()
 *  eglGetCurrentDisplay()
 *  eglGetCurrentSurface()
 *  eglMakeCurrent(when the ctx parameter is EGL NO CONTEXT)
 *  eglWaitClient()
 *  eglWaitNative()
 * See section 3.7 "Rendering Context" in the EGL specification for details.
 */
EGLBoolean EGLAPIENTRY
eglBindAPI(EGLenum api)
{
   _EGLThreadInfo *t;

   _EGL_FUNC_START(NULL, EGL_OBJECT_THREAD_KHR, NULL, EGL_FALSE);

   t = _eglGetCurrentThread();
   if (_eglIsCurrentThreadDummy())
      RETURN_EGL_ERROR(NULL, EGL_BAD_ALLOC, EGL_FALSE);

   if (!_eglIsApiValid(api))
      RETURN_EGL_ERROR(NULL, EGL_BAD_PARAMETER, EGL_FALSE);

   t->CurrentAPI = api;

   RETURN_EGL_SUCCESS(NULL, EGL_TRUE);
}


/**
 * Return the last value set with eglBindAPI().
 */
EGLenum EGLAPIENTRY
eglQueryAPI(void)
{
   _EGLThreadInfo *t = _eglGetCurrentThread();
   EGLenum ret;

   /* returns one of EGL_OPENGL_API, EGL_OPENGL_ES_API or EGL_OPENVG_API */
   ret = t->CurrentAPI;

   RETURN_EGL_SUCCESS(NULL, ret);
}


EGLSurface EGLAPIENTRY
eglCreatePbufferFromClientBuffer(EGLDisplay dpy, EGLenum buftype,
                                 EGLClientBuffer buffer, EGLConfig config,
                                 const EGLint *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLConfig *conf = _eglLookupConfig(config, disp);
   _EGLDriver *drv;
   _EGLSurface *surf;
   EGLSurface ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_SURFACE);

   _EGL_CHECK_CONFIG(disp, conf, EGL_NO_SURFACE, drv);

   surf = drv->API.CreatePbufferFromClientBuffer(drv, disp, buftype, buffer,
                                                 conf, attrib_list);
   ret = (surf) ? _eglLinkSurface(surf) : EGL_NO_SURFACE;

   RETURN_EGL_EVAL(disp, ret);
}


EGLBoolean EGLAPIENTRY
eglReleaseThread(void)
{
   /* unbind current contexts */
   if (!_eglIsCurrentThreadDummy()) {
      _EGLThreadInfo *t = _eglGetCurrentThread();
      _EGLContext *ctx = t->CurrentContext;

      _EGL_FUNC_START(NULL, EGL_OBJECT_THREAD_KHR, NULL, EGL_FALSE);

      if (ctx) {
         _EGLDisplay *disp = ctx->Resource.Display;
         _EGLDriver *drv;

         mtx_lock(&disp->Mutex);
         drv = disp->Driver;
         (void) drv->API.MakeCurrent(drv, disp, NULL, NULL, NULL);
         mtx_unlock(&disp->Mutex);
      }
   }

   _eglDestroyCurrentThread();

   RETURN_EGL_SUCCESS(NULL, EGL_TRUE);
}


static EGLImage
_eglCreateImageCommon(_EGLDisplay *disp, EGLContext ctx, EGLenum target,
                      EGLClientBuffer buffer, const EGLint *attr_list)
{
   _EGLContext *context = _eglLookupContext(ctx, disp);
   _EGLDriver *drv;
   _EGLImage *img;
   EGLImage ret;

   _EGL_CHECK_DISPLAY(disp, EGL_NO_IMAGE_KHR, drv);
   if (!disp->Extensions.KHR_image_base)
      RETURN_EGL_EVAL(disp, EGL_NO_IMAGE_KHR);
   if (!context && ctx != EGL_NO_CONTEXT)
      RETURN_EGL_ERROR(disp, EGL_BAD_CONTEXT, EGL_NO_IMAGE_KHR);
   /* "If <target> is EGL_LINUX_DMA_BUF_EXT, <dpy> must be a valid display,
    *  <ctx> must be EGL_NO_CONTEXT..."
    */
   if (ctx != EGL_NO_CONTEXT && target == EGL_LINUX_DMA_BUF_EXT)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_NO_IMAGE_KHR);

   img = drv->API.CreateImageKHR(drv, disp, context, target,
                                 buffer, attr_list);
   ret = (img) ? _eglLinkImage(img) : EGL_NO_IMAGE_KHR;

   RETURN_EGL_EVAL(disp, ret);
}

static EGLImage EGLAPIENTRY
eglCreateImageKHR(EGLDisplay dpy, EGLContext ctx, EGLenum target,
                  EGLClientBuffer buffer, const EGLint *attr_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_IMAGE_KHR);
   return _eglCreateImageCommon(disp, ctx, target, buffer, attr_list);
}


EGLImage EGLAPIENTRY
eglCreateImage(EGLDisplay dpy, EGLContext ctx, EGLenum target,
               EGLClientBuffer buffer, const EGLAttrib *attr_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   EGLImage image;
   EGLint *int_attribs;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_NO_IMAGE_KHR);

   int_attribs = _eglConvertAttribsToInt(attr_list);
   if (attr_list && !int_attribs)
      RETURN_EGL_ERROR(disp, EGL_BAD_ALLOC, EGL_NO_IMAGE);

   image = _eglCreateImageCommon(disp, ctx, target, buffer, int_attribs);
   free(int_attribs);
   return image;
}


static EGLBoolean
_eglDestroyImageCommon(_EGLDisplay *disp, _EGLImage *img)
{
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);
   if (!disp->Extensions.KHR_image_base)
      RETURN_EGL_EVAL(disp, EGL_FALSE);
   if (!img)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   _eglUnlinkImage(img);
   ret = drv->API.DestroyImageKHR(drv, disp, img);

   RETURN_EGL_EVAL(disp, ret);
}

EGLBoolean EGLAPIENTRY
eglDestroyImage(EGLDisplay dpy, EGLImage image)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLImage *img = _eglLookupImage(image, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_IMAGE_KHR, img, EGL_FALSE);
   return _eglDestroyImageCommon(disp, img);
}

static EGLBoolean EGLAPIENTRY
eglDestroyImageKHR(EGLDisplay dpy, EGLImage image)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLImage *img = _eglLookupImage(image, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_IMAGE_KHR, img, EGL_FALSE);
   return _eglDestroyImageCommon(disp, img);
}


static EGLSync
_eglCreateSync(_EGLDisplay *disp, EGLenum type, const EGLAttrib *attrib_list,
               EGLBoolean orig_is_EGLAttrib,
               EGLenum invalid_type_error)
{
   _EGLContext *ctx = _eglGetCurrentContext();
   _EGLDriver *drv;
   _EGLSync *sync;
   EGLSync ret;

   _EGL_CHECK_DISPLAY(disp, EGL_NO_SYNC_KHR, drv);

   if (!disp->Extensions.KHR_cl_event2 && orig_is_EGLAttrib) {
      /* There exist two EGLAttrib variants of eglCreateSync*:
       * eglCreateSync64KHR which requires EGL_KHR_cl_event2, and eglCreateSync
       * which requires EGL 1.5. Here we use the presence of EGL_KHR_cl_event2
       * support as a proxy for EGL 1.5 support, even though that's not
       * entirely correct (though _eglComputeVersion does the same).
       *
       * The EGL spec provides no guidance on how to handle unsupported
       * functions. EGL_BAD_MATCH seems reasonable.
       */
      RETURN_EGL_ERROR(disp, EGL_BAD_MATCH, EGL_NO_SYNC_KHR);
   }

   /* If type is EGL_SYNC_FENCE and no context is current for the bound API
    * (i.e., eglGetCurrentContext returns EGL_NO_CONTEXT ), an EGL_BAD_MATCH
    * error is generated.
    */
   if (!ctx &&
       (type == EGL_SYNC_FENCE_KHR || type == EGL_SYNC_NATIVE_FENCE_ANDROID))
      RETURN_EGL_ERROR(disp, EGL_BAD_MATCH, EGL_NO_SYNC_KHR);

   /* return an error if the client API doesn't support GL_OES_EGL_sync */
   if (ctx && (ctx->Resource.Display != disp ||
               ctx->ClientAPI != EGL_OPENGL_ES_API))
      RETURN_EGL_ERROR(disp, EGL_BAD_MATCH, EGL_NO_SYNC_KHR);

   switch (type) {
   case EGL_SYNC_FENCE_KHR:
      if (!disp->Extensions.KHR_fence_sync)
         RETURN_EGL_ERROR(disp, invalid_type_error, EGL_NO_SYNC_KHR);
      break;
   case EGL_SYNC_REUSABLE_KHR:
      if (!disp->Extensions.KHR_reusable_sync)
         RETURN_EGL_ERROR(disp, invalid_type_error, EGL_NO_SYNC_KHR);
      break;
   case EGL_SYNC_CL_EVENT_KHR:
      if (!disp->Extensions.KHR_cl_event2)
         RETURN_EGL_ERROR(disp, invalid_type_error, EGL_NO_SYNC_KHR);
      break;
   case EGL_SYNC_NATIVE_FENCE_ANDROID:
      if (!disp->Extensions.ANDROID_native_fence_sync)
         RETURN_EGL_ERROR(disp, invalid_type_error, EGL_NO_SYNC_KHR);
      break;
   default:
      RETURN_EGL_ERROR(disp, invalid_type_error, EGL_NO_SYNC_KHR);
   }

   sync = drv->API.CreateSyncKHR(drv, disp, type, attrib_list);
   ret = (sync) ? _eglLinkSync(sync) : EGL_NO_SYNC_KHR;

   RETURN_EGL_EVAL(disp, ret);
}


static EGLSync EGLAPIENTRY
eglCreateSyncKHR(EGLDisplay dpy, EGLenum type, const EGLint *int_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   EGLSync sync;
   EGLAttrib *attrib_list;
   EGLint err;

   if (sizeof(int_list[0]) == sizeof(attrib_list[0])) {
      attrib_list = (EGLAttrib *) int_list;
   } else {
      err = _eglConvertIntsToAttribs(int_list, &attrib_list);
      if (err != EGL_SUCCESS)
         RETURN_EGL_ERROR(disp, err, EGL_NO_SYNC);
   }

   sync = _eglCreateSync(disp, type, attrib_list, EGL_FALSE,
                         EGL_BAD_ATTRIBUTE);

   if (sizeof(int_list[0]) != sizeof(attrib_list[0]))
      free(attrib_list);

   /* Don't double-unlock the display. _eglCreateSync already unlocked it. */
   return sync;
}


static EGLSync EGLAPIENTRY
eglCreateSync64KHR(EGLDisplay dpy, EGLenum type, const EGLAttrib *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);
   return _eglCreateSync(disp, type, attrib_list, EGL_TRUE,
                         EGL_BAD_ATTRIBUTE);
}


EGLSync EGLAPIENTRY
eglCreateSync(EGLDisplay dpy, EGLenum type, const EGLAttrib *attrib_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);
   return _eglCreateSync(disp, type, attrib_list, EGL_TRUE,
                         EGL_BAD_PARAMETER);
}


static EGLBoolean
_eglDestroySync(_EGLDisplay *disp, _EGLSync *s)
{
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_CHECK_SYNC(disp, s, EGL_FALSE, drv);
   assert(disp->Extensions.KHR_reusable_sync ||
          disp->Extensions.KHR_fence_sync ||
          disp->Extensions.ANDROID_native_fence_sync);

   _eglUnlinkSync(s);
   ret = drv->API.DestroySyncKHR(drv, disp, s);

   RETURN_EGL_EVAL(disp, ret);
}

EGLBoolean EGLAPIENTRY
eglDestroySync(EGLDisplay dpy, EGLSync sync)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSync *s = _eglLookupSync(sync, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_SYNC_KHR, s, EGL_FALSE);
   return _eglDestroySync(disp, s);
}

static EGLBoolean EGLAPIENTRY
eglDestroySyncKHR(EGLDisplay dpy, EGLSync sync)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSync *s = _eglLookupSync(sync, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_SYNC_KHR, s, EGL_FALSE);
   return _eglDestroySync(disp, s);
}


static EGLint
_eglClientWaitSyncCommon(_EGLDisplay *disp, EGLDisplay dpy,
                         _EGLSync *s, EGLint flags, EGLTime timeout)
{
   _EGLDriver *drv;
   EGLint ret;

   _EGL_CHECK_SYNC(disp, s, EGL_FALSE, drv);
   assert(disp->Extensions.KHR_reusable_sync ||
          disp->Extensions.KHR_fence_sync ||
          disp->Extensions.ANDROID_native_fence_sync);

   if (s->SyncStatus == EGL_SIGNALED_KHR)
      RETURN_EGL_EVAL(disp, EGL_CONDITION_SATISFIED_KHR);

   /* if sync type is EGL_SYNC_REUSABLE_KHR, dpy should be
    * unlocked here to allow other threads also to be able to
    * go into waiting state.
    */

   if (s->Type == EGL_SYNC_REUSABLE_KHR)
      _eglUnlockDisplay(dpy);

   ret = drv->API.ClientWaitSyncKHR(drv, disp, s, flags, timeout);

   /*
    * 'disp' is already unlocked for reusable sync type,
    * so passing 'NULL' to bypass unlocking display.
    */
   if (s->Type == EGL_SYNC_REUSABLE_KHR)
      RETURN_EGL_EVAL(NULL, ret);
   else
      RETURN_EGL_EVAL(disp, ret);
}

EGLint EGLAPIENTRY
eglClientWaitSync(EGLDisplay dpy, EGLSync sync,
                  EGLint flags, EGLTime timeout)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSync *s = _eglLookupSync(sync, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_SYNC_KHR, s, EGL_FALSE);
   return _eglClientWaitSyncCommon(disp, dpy, s, flags, timeout);
}

static EGLint EGLAPIENTRY
eglClientWaitSyncKHR(EGLDisplay dpy, EGLSync sync,
                     EGLint flags, EGLTime timeout)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSync *s = _eglLookupSync(sync, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_SYNC_KHR, s, EGL_FALSE);
   return _eglClientWaitSyncCommon(disp, dpy, s, flags, timeout);
}


static EGLint
_eglWaitSyncCommon(_EGLDisplay *disp, _EGLSync *s, EGLint flags)
{
   _EGLContext *ctx = _eglGetCurrentContext();
   _EGLDriver *drv;
   EGLint ret;

   _EGL_CHECK_SYNC(disp, s, EGL_FALSE, drv);
   assert(disp->Extensions.KHR_wait_sync);

   /* return an error if the client API doesn't support GL_OES_EGL_sync */
   if (ctx == EGL_NO_CONTEXT || ctx->ClientAPI != EGL_OPENGL_ES_API)
      RETURN_EGL_ERROR(disp, EGL_BAD_MATCH, EGL_FALSE);

   /* the API doesn't allow any flags yet */
   if (flags != 0)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   ret = drv->API.WaitSyncKHR(drv, disp, s);

   RETURN_EGL_EVAL(disp, ret);
}

static EGLint EGLAPIENTRY
eglWaitSyncKHR(EGLDisplay dpy, EGLSync sync, EGLint flags)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSync *s = _eglLookupSync(sync, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_SYNC_KHR, s, EGL_FALSE);
   return _eglWaitSyncCommon(disp, s, flags);
}


EGLBoolean EGLAPIENTRY
eglWaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags)
{
   /* The KHR version returns EGLint, while the core version returns
    * EGLBoolean. In both cases, the return values can only be EGL_FALSE and
    * EGL_TRUE.
    */
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSync *s = _eglLookupSync(sync, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_SYNC_KHR, s, EGL_FALSE);
   return _eglWaitSyncCommon(disp, s, flags);
}


static EGLBoolean EGLAPIENTRY
eglSignalSyncKHR(EGLDisplay dpy, EGLSync sync, EGLenum mode)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSync *s = _eglLookupSync(sync, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SYNC_KHR, s, EGL_FALSE);

   _EGL_CHECK_SYNC(disp, s, EGL_FALSE, drv);
   assert(disp->Extensions.KHR_reusable_sync);
   ret = drv->API.SignalSyncKHR(drv, disp, s, mode);

   RETURN_EGL_EVAL(disp, ret);
}


static EGLBoolean
_eglGetSyncAttribCommon(_EGLDisplay *disp, _EGLSync *s, EGLint attribute, EGLAttrib *value)
{
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_CHECK_SYNC(disp, s, EGL_FALSE, drv);
   assert(disp->Extensions.KHR_reusable_sync ||
          disp->Extensions.KHR_fence_sync ||
          disp->Extensions.ANDROID_native_fence_sync);
   ret = drv->API.GetSyncAttrib(drv, disp, s, attribute, value);

   RETURN_EGL_EVAL(disp, ret);
}

EGLBoolean EGLAPIENTRY
eglGetSyncAttrib(EGLDisplay dpy, EGLSync sync, EGLint attribute, EGLAttrib *value)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSync *s = _eglLookupSync(sync, disp);
   _EGL_FUNC_START(disp, EGL_OBJECT_SYNC_KHR, s, EGL_FALSE);
   return _eglGetSyncAttribCommon(disp, s, attribute, value);
}


static EGLBoolean EGLAPIENTRY
eglGetSyncAttribKHR(EGLDisplay dpy, EGLSync sync, EGLint attribute, EGLint *value)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSync *s = _eglLookupSync(sync, disp);
   EGLAttrib attrib;
   EGLBoolean result;

   _EGL_FUNC_START(disp, EGL_OBJECT_SYNC_KHR, s, EGL_FALSE);

   if (!value)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   attrib = *value;
   result = _eglGetSyncAttribCommon(disp, s, attribute, &attrib);

   /* The EGL_KHR_fence_sync spec says this about eglGetSyncAttribKHR:
    *
    *    If any error occurs, <*value> is not modified.
    */
   if (result == EGL_FALSE)
      return result;

   *value = attrib;
   return result;
}

static EGLint EGLAPIENTRY
eglDupNativeFenceFDANDROID(EGLDisplay dpy, EGLSync sync)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSync *s = _eglLookupSync(sync, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SYNC_KHR, s, EGL_FALSE);

   /* the spec doesn't seem to specify what happens if the fence
    * type is not EGL_SYNC_NATIVE_FENCE_ANDROID, but this seems
    * sensible:
    */
   if (!(s && (s->Type == EGL_SYNC_NATIVE_FENCE_ANDROID)))
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_NO_NATIVE_FENCE_FD_ANDROID);

   _EGL_CHECK_SYNC(disp, s, EGL_NO_NATIVE_FENCE_FD_ANDROID, drv);
   assert(disp->Extensions.ANDROID_native_fence_sync);
   ret = drv->API.DupNativeFenceFDANDROID(drv, disp, s);

   RETURN_EGL_EVAL(disp, ret);
}

static EGLBoolean EGLAPIENTRY
eglSwapBuffersRegionNOK(EGLDisplay dpy, EGLSurface surface,
                        EGLint numRects, const EGLint *rects)
{
   _EGLContext *ctx = _eglGetCurrentContext();
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);

   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);

   if (!disp->Extensions.NOK_swap_region)
      RETURN_EGL_EVAL(disp, EGL_FALSE);

   /* surface must be bound to current context in EGL 1.4 */
   if (_eglGetContextHandle(ctx) == EGL_NO_CONTEXT ||
       surf != ctx->DrawSurface)
      RETURN_EGL_ERROR(disp, EGL_BAD_SURFACE, EGL_FALSE);

   ret = drv->API.SwapBuffersRegionNOK(drv, disp, surf, numRects, rects);

   RETURN_EGL_EVAL(disp, ret);
}


static EGLImage EGLAPIENTRY
eglCreateDRMImageMESA(EGLDisplay dpy, const EGLint *attr_list)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLDriver *drv;
   _EGLImage *img;
   EGLImage ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_NO_IMAGE_KHR, drv);
   if (!disp->Extensions.MESA_drm_image)
      RETURN_EGL_EVAL(disp, EGL_NO_IMAGE_KHR);

   img = drv->API.CreateDRMImageMESA(drv, disp, attr_list);
   ret = (img) ? _eglLinkImage(img) : EGL_NO_IMAGE_KHR;

   RETURN_EGL_EVAL(disp, ret);
}

static EGLBoolean EGLAPIENTRY
eglExportDRMImageMESA(EGLDisplay dpy, EGLImage image,
                      EGLint *name, EGLint *handle, EGLint *stride)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLImage *img = _eglLookupImage(image, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_IMAGE_KHR, img, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);
   assert(disp->Extensions.MESA_drm_image);

   if (!img)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   ret = drv->API.ExportDRMImageMESA(drv, disp, img, name, handle, stride);

   RETURN_EGL_EVAL(disp, ret);
}


struct wl_display;

static EGLBoolean EGLAPIENTRY
eglBindWaylandDisplayWL(EGLDisplay dpy, struct wl_display *display)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);
   assert(disp->Extensions.WL_bind_wayland_display);

   if (!display)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   ret = drv->API.BindWaylandDisplayWL(drv, disp, display);

   RETURN_EGL_EVAL(disp, ret);
}

static EGLBoolean EGLAPIENTRY
eglUnbindWaylandDisplayWL(EGLDisplay dpy, struct wl_display *display)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);
   assert(disp->Extensions.WL_bind_wayland_display);

   if (!display)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   ret = drv->API.UnbindWaylandDisplayWL(drv, disp, display);

   RETURN_EGL_EVAL(disp, ret);
}

static EGLBoolean EGLAPIENTRY
eglQueryWaylandBufferWL(EGLDisplay dpy, struct wl_resource *buffer,
                        EGLint attribute, EGLint *value)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);
   assert(disp->Extensions.WL_bind_wayland_display);

   if (!buffer)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   ret = drv->API.QueryWaylandBufferWL(drv, disp, buffer, attribute, value);

   RETURN_EGL_EVAL(disp, ret);
}


static struct wl_buffer * EGLAPIENTRY
eglCreateWaylandBufferFromImageWL(EGLDisplay dpy, EGLImage image)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLImage *img;
   _EGLDriver *drv;
   struct wl_buffer *ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_DISPLAY_KHR, NULL, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, NULL, drv);
   if (!disp->Extensions.WL_create_wayland_buffer_from_image)
      RETURN_EGL_EVAL(disp, NULL);

   img = _eglLookupImage(image, disp);

   if (!img)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, NULL);

   ret = drv->API.CreateWaylandBufferFromImageWL(drv, disp, img);

   RETURN_EGL_EVAL(disp, ret);
}

static EGLBoolean EGLAPIENTRY
eglPostSubBufferNV(EGLDisplay dpy, EGLSurface surface,
                   EGLint x, EGLint y, EGLint width, EGLint height)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);

   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);

   if (!disp->Extensions.NV_post_sub_buffer)
      RETURN_EGL_EVAL(disp, EGL_FALSE);

   ret = drv->API.PostSubBufferNV(drv, disp, surf, x, y, width, height);

   RETURN_EGL_EVAL(disp, ret);
}

static EGLBoolean EGLAPIENTRY
eglGetSyncValuesCHROMIUM(EGLDisplay display, EGLSurface surface,
                         EGLuint64KHR *ust, EGLuint64KHR *msc,
                         EGLuint64KHR *sbc)
{
   _EGLDisplay *disp = _eglLockDisplay(display);
   _EGLSurface *surf = _eglLookupSurface(surface, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_SURFACE_KHR, surf, EGL_FALSE);

   _EGL_CHECK_SURFACE(disp, surf, EGL_FALSE, drv);
   if (!disp->Extensions.CHROMIUM_sync_control)
      RETURN_EGL_EVAL(disp, EGL_FALSE);

   if (!ust || !msc || !sbc)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   ret = drv->API.GetSyncValuesCHROMIUM(disp, surf, ust, msc, sbc);

   RETURN_EGL_EVAL(disp, ret);
}

static EGLBoolean EGLAPIENTRY
eglExportDMABUFImageQueryMESA(EGLDisplay dpy, EGLImage image,
                              EGLint *fourcc, EGLint *nplanes,
                              EGLuint64KHR *modifiers)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLImage *img = _eglLookupImage(image, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_IMAGE_KHR, img, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);
   assert(disp->Extensions.MESA_image_dma_buf_export);

   if (!img)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   ret = drv->API.ExportDMABUFImageQueryMESA(drv, disp, img, fourcc, nplanes,
                                             modifiers);

   RETURN_EGL_EVAL(disp, ret);
}

static EGLBoolean EGLAPIENTRY
eglExportDMABUFImageMESA(EGLDisplay dpy, EGLImage image,
                         int *fds, EGLint *strides, EGLint *offsets)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLImage *img = _eglLookupImage(image, disp);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(disp, EGL_OBJECT_IMAGE_KHR, img, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);
   assert(disp->Extensions.MESA_image_dma_buf_export);

   if (!img)
      RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_FALSE);

   ret = drv->API.ExportDMABUFImageMESA(drv, disp, img, fds, strides, offsets);

   RETURN_EGL_EVAL(disp, ret);
}

static EGLint EGLAPIENTRY
eglLabelObjectKHR(EGLDisplay dpy, EGLenum objectType, EGLObjectKHR object,
                  EGLLabelKHR label)
{
   _EGLDisplay *disp = NULL;
   _EGLResourceType type;

   _EGL_FUNC_START(NULL, EGL_NONE, NULL, EGL_BAD_ALLOC);

   if (objectType == EGL_OBJECT_THREAD_KHR) {
      _EGLThreadInfo *t = _eglGetCurrentThread();

      if (!_eglIsCurrentThreadDummy()) {
         t->Label = label;
         return EGL_SUCCESS;
      }

      RETURN_EGL_ERROR(NULL, EGL_BAD_ALLOC, EGL_BAD_ALLOC);
   }

   disp = _eglLockDisplay(dpy);
   if (disp == NULL)
      RETURN_EGL_ERROR(disp, EGL_BAD_DISPLAY, EGL_BAD_DISPLAY);

   if (objectType == EGL_OBJECT_DISPLAY_KHR) {
      if (dpy != (EGLDisplay) object)
         RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_BAD_PARAMETER);

      disp->Label = label;
      RETURN_EGL_EVAL(disp, EGL_SUCCESS);
   }

   switch (objectType) {
      case EGL_OBJECT_CONTEXT_KHR:
         type = _EGL_RESOURCE_CONTEXT;
         break;
      case EGL_OBJECT_SURFACE_KHR:
         type = _EGL_RESOURCE_SURFACE;
         break;
      case EGL_OBJECT_IMAGE_KHR:
         type = _EGL_RESOURCE_IMAGE;
         break;
      case EGL_OBJECT_SYNC_KHR:
         type = _EGL_RESOURCE_SYNC;
         break;
      case EGL_OBJECT_STREAM_KHR:
      default:
         RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_BAD_PARAMETER);
   }

   if (_eglCheckResource(object, type, disp)) {
      _EGLResource *res = (_EGLResource *) object;

      res->Label = label;
      RETURN_EGL_EVAL(disp, EGL_SUCCESS);
   }

   RETURN_EGL_ERROR(disp, EGL_BAD_PARAMETER, EGL_BAD_PARAMETER);
}

static EGLint EGLAPIENTRY
eglDebugMessageControlKHR(EGLDEBUGPROCKHR callback,
                          const EGLAttrib *attrib_list)
{
   unsigned int newEnabled;

   _EGL_FUNC_START(NULL, EGL_NONE, NULL, EGL_BAD_ALLOC);

   mtx_lock(_eglGlobal.Mutex);

   newEnabled = _eglGlobal.debugTypesEnabled;
   if (attrib_list != NULL) {
      int i;

      for (i = 0; attrib_list[i] != EGL_NONE; i += 2) {
         switch (attrib_list[i]) {
         case EGL_DEBUG_MSG_CRITICAL_KHR:
         case EGL_DEBUG_MSG_ERROR_KHR:
         case EGL_DEBUG_MSG_WARN_KHR:
         case EGL_DEBUG_MSG_INFO_KHR:
            if (attrib_list[i + 1])
               newEnabled |= DebugBitFromType(attrib_list[i]);
            else
               newEnabled &= ~DebugBitFromType(attrib_list[i]);
            break;
         default:
            // On error, set the last error code, call the current
            // debug callback, and return the error code.
            mtx_unlock(_eglGlobal.Mutex);
            _eglReportError(EGL_BAD_ATTRIBUTE, NULL,
                  "Invalid attribute 0x%04lx", (unsigned long) attrib_list[i]);
            return EGL_BAD_ATTRIBUTE;
         }
      }
   }

   if (callback != NULL) {
      _eglGlobal.debugCallback = callback;
      _eglGlobal.debugTypesEnabled = newEnabled;
   } else {
      _eglGlobal.debugCallback = NULL;
      _eglGlobal.debugTypesEnabled = _EGL_DEBUG_BIT_CRITICAL | _EGL_DEBUG_BIT_ERROR;
   }

   mtx_unlock(_eglGlobal.Mutex);
   return EGL_SUCCESS;
}

static EGLBoolean EGLAPIENTRY
eglQueryDebugKHR(EGLint attribute, EGLAttrib *value)
{
   _EGL_FUNC_START(NULL, EGL_NONE, NULL, EGL_BAD_ALLOC);

   mtx_lock(_eglGlobal.Mutex);

   switch (attribute) {
   case EGL_DEBUG_MSG_CRITICAL_KHR:
   case EGL_DEBUG_MSG_ERROR_KHR:
   case EGL_DEBUG_MSG_WARN_KHR:
   case EGL_DEBUG_MSG_INFO_KHR:
      if (_eglGlobal.debugTypesEnabled & DebugBitFromType(attribute))
         *value = EGL_TRUE;
      else
         *value = EGL_FALSE;
      break;
   case EGL_DEBUG_CALLBACK_KHR:
      *value = (EGLAttrib) _eglGlobal.debugCallback;
      break;
   default:
      mtx_unlock(_eglGlobal.Mutex);
      _eglReportError(EGL_BAD_ATTRIBUTE, NULL,
                      "Invalid attribute 0x%04lx", (unsigned long) attribute);
      return EGL_FALSE;
   }

   mtx_unlock(_eglGlobal.Mutex);
   return EGL_TRUE;
}

static int
_eglFunctionCompare(const void *key, const void *elem)
{
   const char *procname = key;
   const struct _egl_entrypoint *entrypoint = elem;
   return strcmp(procname, entrypoint->name);
}

static EGLBoolean EGLAPIENTRY
eglQueryDmaBufFormatsEXT(EGLDisplay dpy, EGLint max_formats,
                         EGLint *formats, EGLint *num_formats)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(NULL, EGL_NONE, NULL, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);

   ret = drv->API.QueryDmaBufFormatsEXT(drv, disp, max_formats, formats,
                                        num_formats);

   RETURN_EGL_EVAL(disp, ret);
}

static EGLBoolean EGLAPIENTRY
eglQueryDmaBufModifiersEXT(EGLDisplay dpy, EGLint format, EGLint max_modifiers,
                           EGLuint64KHR *modifiers, EGLBoolean *external_only,
                           EGLint *num_modifiers)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLDriver *drv;
   EGLBoolean ret;

   _EGL_FUNC_START(NULL, EGL_NONE, NULL, EGL_FALSE);

   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);

   ret = drv->API.QueryDmaBufModifiersEXT(drv, disp, format, max_modifiers,
                                          modifiers, external_only,
                                          num_modifiers);

   RETURN_EGL_EVAL(disp, ret);
}

static void EGLAPIENTRY
eglSetBlobCacheFuncsANDROID(EGLDisplay *dpy, EGLSetBlobFuncANDROID set,
                            EGLGetBlobFuncANDROID get)
{
   /* This function does not return anything so we cannot
    * utilize the helper macros _EGL_FUNC_START or _EGL_CHECK_DISPLAY.
    */
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   if (!_eglSetFuncName(__func__, disp, EGL_OBJECT_DISPLAY_KHR, NULL)) {
      if (disp)
         _eglUnlockDisplay(disp);
      return;
   }

   _EGLDriver *drv = _eglCheckDisplay(disp, __func__);
   if (!drv) {
      if (disp)
         _eglUnlockDisplay(disp);
      return;
   }

   if (!set || !get) {
      _eglError(EGL_BAD_PARAMETER,
                "eglSetBlobCacheFuncsANDROID: NULL handler given");
      _eglUnlockDisplay(disp);
      return;
   }

   if (disp->BlobCacheSet) {
      _eglError(EGL_BAD_PARAMETER,
                "eglSetBlobCacheFuncsANDROID: functions already set");
      _eglUnlockDisplay(disp);
      return;
   }

   disp->BlobCacheSet = set;
   disp->BlobCacheGet = get;

   drv->API.SetBlobCacheFuncsANDROID(drv, disp, set, get);

   _eglUnlockDisplay(disp);
}

static EGLBoolean EGLAPIENTRY
eglQueryDeviceAttribEXT(EGLDeviceEXT device,
                        EGLint attribute,
                        EGLAttrib *value)
{
   _EGLDevice *dev = _eglLookupDevice(device);
   EGLBoolean ret;

   _EGL_FUNC_START(NULL, EGL_NONE, NULL, EGL_FALSE);
   if (!dev)
      RETURN_EGL_ERROR(NULL, EGL_BAD_DEVICE_EXT, EGL_FALSE);

   ret = _eglQueryDeviceAttribEXT(dev, attribute, value);
   RETURN_EGL_EVAL(NULL, ret);
}

static const char * EGLAPIENTRY
eglQueryDeviceStringEXT(EGLDeviceEXT device,
                        EGLint name)
{
   _EGLDevice *dev = _eglLookupDevice(device);

   _EGL_FUNC_START(NULL, EGL_NONE, NULL, NULL);
   if (!dev)
      RETURN_EGL_ERROR(NULL, EGL_BAD_DEVICE_EXT, NULL);

   RETURN_EGL_EVAL(NULL, _eglQueryDeviceStringEXT(dev, name));
}

static EGLBoolean EGLAPIENTRY
eglQueryDevicesEXT(EGLint max_devices,
                   EGLDeviceEXT *devices,
                   EGLint *num_devices)
{
   EGLBoolean ret;

   _EGL_FUNC_START(NULL, EGL_NONE, NULL, EGL_FALSE);
   ret = _eglQueryDevicesEXT(max_devices, (_EGLDevice **) devices,
                             num_devices);
   RETURN_EGL_EVAL(NULL, ret);
}

static EGLBoolean EGLAPIENTRY
eglQueryDisplayAttribEXT(EGLDisplay dpy,
                         EGLint attribute,
                         EGLAttrib *value)
{
   _EGLDisplay *disp = _eglLockDisplay(dpy);
   _EGLDriver *drv;

   _EGL_FUNC_START(NULL, EGL_NONE, NULL, EGL_FALSE);
   _EGL_CHECK_DISPLAY(disp, EGL_FALSE, drv);

   switch (attribute) {
   case EGL_DEVICE_EXT:
      *value = (EGLAttrib) disp->Device;
      break;
   default:
      RETURN_EGL_ERROR(disp, EGL_BAD_ATTRIBUTE, EGL_FALSE);
   }
   RETURN_EGL_SUCCESS(disp, EGL_TRUE);
}

__eglMustCastToProperFunctionPointerType EGLAPIENTRY
eglGetProcAddress(const char *procname)
{
   static const struct _egl_entrypoint egl_functions[] = {
#define EGL_ENTRYPOINT(f) { .name = #f, .function = (_EGLProc) f },
#include "eglentrypoint.h"
#undef EGL_ENTRYPOINT
   };
   _EGLProc ret = NULL;

   if (!procname)
      RETURN_EGL_SUCCESS(NULL, NULL);

   _EGL_FUNC_START(NULL, EGL_NONE, NULL, NULL);

   if (strncmp(procname, "egl", 3) == 0) {
      const struct _egl_entrypoint *entrypoint =
         bsearch(procname,
                 egl_functions, ARRAY_SIZE(egl_functions),
                 sizeof(egl_functions[0]),
                 _eglFunctionCompare);
      if (entrypoint)
         ret = entrypoint->function;
   }

   if (!ret)
      ret = _eglGetDriverProc(procname);

   RETURN_EGL_SUCCESS(NULL, ret);
}

static int
_eglLockDisplayInterop(EGLDisplay dpy, EGLContext context,
                       _EGLDisplay **disp, _EGLDriver **drv,
                       _EGLContext **ctx)
{

   *disp = _eglLockDisplay(dpy);
   if (!*disp || !(*disp)->Initialized || !(*disp)->Driver) {
      if (*disp)
         _eglUnlockDisplay(*disp);
      return MESA_GLINTEROP_INVALID_DISPLAY;
   }

   *drv = (*disp)->Driver;

   *ctx = _eglLookupContext(context, *disp);
   if (!*ctx ||
       ((*ctx)->ClientAPI != EGL_OPENGL_API &&
        (*ctx)->ClientAPI != EGL_OPENGL_ES_API)) {
      _eglUnlockDisplay(*disp);
      return MESA_GLINTEROP_INVALID_CONTEXT;
   }

   return MESA_GLINTEROP_SUCCESS;
}

PUBLIC int
MesaGLInteropEGLQueryDeviceInfo(EGLDisplay dpy, EGLContext context,
                                struct mesa_glinterop_device_info *out)
{
   _EGLDisplay *disp;
   _EGLDriver *drv;
   _EGLContext *ctx;
   int ret;

   ret = _eglLockDisplayInterop(dpy, context, &disp, &drv, &ctx);
   if (ret != MESA_GLINTEROP_SUCCESS)
      return ret;

   if (drv->API.GLInteropQueryDeviceInfo)
      ret = drv->API.GLInteropQueryDeviceInfo(disp, ctx, out);
   else
      ret = MESA_GLINTEROP_UNSUPPORTED;

   _eglUnlockDisplay(disp);
   return ret;
}

PUBLIC int
MesaGLInteropEGLExportObject(EGLDisplay dpy, EGLContext context,
                             struct mesa_glinterop_export_in *in,
                             struct mesa_glinterop_export_out *out)
{
   _EGLDisplay *disp;
   _EGLDriver *drv;
   _EGLContext *ctx;
   int ret;

   ret = _eglLockDisplayInterop(dpy, context, &disp, &drv, &ctx);
   if (ret != MESA_GLINTEROP_SUCCESS)
      return ret;

   if (drv->API.GLInteropExportObject)
      ret = drv->API.GLInteropExportObject(disp, ctx, in, out);
   else
      ret = MESA_GLINTEROP_UNSUPPORTED;

   _eglUnlockDisplay(disp);
   return ret;
}
