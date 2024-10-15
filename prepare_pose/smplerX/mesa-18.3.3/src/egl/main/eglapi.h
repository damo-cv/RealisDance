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


#ifndef EGLAPI_INCLUDED
#define EGLAPI_INCLUDED


#ifdef __cplusplus
extern "C" {
#endif

/**
 * A generic function ptr type
 */
typedef void (*_EGLProc)(void);

struct wl_display;
struct mesa_glinterop_device_info;
struct mesa_glinterop_export_in;
struct mesa_glinterop_export_out;

/**
 * The API dispatcher jumps through these functions
 */
struct _egl_api
{
   /* driver funcs */
   EGLBoolean (*Initialize)(_EGLDriver *, _EGLDisplay *dpy);
   EGLBoolean (*Terminate)(_EGLDriver *, _EGLDisplay *dpy);

   /* config funcs */
   EGLBoolean (*GetConfigs)(_EGLDriver *drv, _EGLDisplay *dpy,
                            EGLConfig *configs, EGLint config_size,
                            EGLint *num_config);
   EGLBoolean (*ChooseConfig)(_EGLDriver *drv, _EGLDisplay *dpy,
                              const EGLint *attrib_list, EGLConfig *configs,
                              EGLint config_size, EGLint *num_config);
   EGLBoolean (*GetConfigAttrib)(_EGLDriver *drv, _EGLDisplay *dpy,
                                 _EGLConfig *config, EGLint attribute,
                                 EGLint *value);

   /* context funcs */
   _EGLContext *(*CreateContext)(_EGLDriver *drv, _EGLDisplay *dpy,
                                 _EGLConfig *config, _EGLContext *share_list,
                                 const EGLint *attrib_list);
   EGLBoolean (*DestroyContext)(_EGLDriver *drv, _EGLDisplay *dpy,
                                _EGLContext *ctx);
   /* this is the only function (other than Initialize) that may be called
    * with an uninitialized display
    */
   EGLBoolean (*MakeCurrent)(_EGLDriver *drv, _EGLDisplay *dpy,
                             _EGLSurface *draw, _EGLSurface *read,
                             _EGLContext *ctx);
   EGLBoolean (*QueryContext)(_EGLDriver *drv, _EGLDisplay *dpy,
                              _EGLContext *ctx, EGLint attribute,
                              EGLint *value);

   /* surface funcs */
   _EGLSurface *(*CreateWindowSurface)(_EGLDriver *drv, _EGLDisplay *dpy,
                                       _EGLConfig *config, void *native_window,
                                       const EGLint *attrib_list);
   _EGLSurface *(*CreatePixmapSurface)(_EGLDriver *drv, _EGLDisplay *dpy,
                                       _EGLConfig *config, void *native_pixmap,
                                       const EGLint *attrib_list);
   _EGLSurface *(*CreatePbufferSurface)(_EGLDriver *drv, _EGLDisplay *dpy,
                                        _EGLConfig *config,
                                        const EGLint *attrib_list);
   EGLBoolean (*DestroySurface)(_EGLDriver *drv, _EGLDisplay *dpy,
                                _EGLSurface *surface);
   EGLBoolean (*QuerySurface)(_EGLDriver *drv, _EGLDisplay *dpy,
                              _EGLSurface *surface, EGLint attribute,
                              EGLint *value);
   EGLBoolean (*SurfaceAttrib)(_EGLDriver *drv, _EGLDisplay *dpy,
                               _EGLSurface *surface, EGLint attribute,
                               EGLint value);
   EGLBoolean (*BindTexImage)(_EGLDriver *drv, _EGLDisplay *dpy,
                              _EGLSurface *surface, EGLint buffer);
   EGLBoolean (*ReleaseTexImage)(_EGLDriver *drv, _EGLDisplay *dpy,
                                 _EGLSurface *surface, EGLint buffer);
   EGLBoolean (*SwapInterval)(_EGLDriver *drv, _EGLDisplay *dpy,
                              _EGLSurface *surf, EGLint interval);
   EGLBoolean (*SwapBuffers)(_EGLDriver *drv, _EGLDisplay *dpy,
                             _EGLSurface *draw);
   EGLBoolean (*CopyBuffers)(_EGLDriver *drv, _EGLDisplay *dpy,
                             _EGLSurface *surface, void *native_pixmap_target);
   EGLBoolean (*SetDamageRegion)(_EGLDriver *drv, _EGLDisplay *dpy,
                                 _EGLSurface *surface, EGLint *rects, EGLint n_rects);

   /* misc functions */
   EGLBoolean (*WaitClient)(_EGLDriver *drv, _EGLDisplay *dpy,
                            _EGLContext *ctx);
   EGLBoolean (*WaitNative)(_EGLDriver *drv, _EGLDisplay *dpy,
                            EGLint engine);

   /* this function may be called from multiple threads at the same time */
   _EGLProc (*GetProcAddress)(_EGLDriver *drv, const char *procname);

   _EGLSurface *(*CreatePbufferFromClientBuffer)(_EGLDriver *drv,
                                                 _EGLDisplay *dpy,
                                                 EGLenum buftype,
                                                 EGLClientBuffer buffer,
                                                 _EGLConfig *config,
                                                 const EGLint *attrib_list);

   _EGLImage *(*CreateImageKHR)(_EGLDriver *drv, _EGLDisplay *dpy,
                                _EGLContext *ctx, EGLenum target,
                                EGLClientBuffer buffer,
                                const EGLint *attr_list);
   EGLBoolean (*DestroyImageKHR)(_EGLDriver *drv, _EGLDisplay *dpy,
                                 _EGLImage *image);

   _EGLSync *(*CreateSyncKHR)(_EGLDriver *drv, _EGLDisplay *dpy, EGLenum type,
                              const EGLAttrib *attrib_list);
   EGLBoolean (*DestroySyncKHR)(_EGLDriver *drv, _EGLDisplay *dpy,
                                _EGLSync *sync);
   EGLint (*ClientWaitSyncKHR)(_EGLDriver *drv, _EGLDisplay *dpy,
                               _EGLSync *sync, EGLint flags, EGLTime timeout);
   EGLint (*WaitSyncKHR)(_EGLDriver *drv, _EGLDisplay *dpy, _EGLSync *sync);
   EGLBoolean (*SignalSyncKHR)(_EGLDriver *drv, _EGLDisplay *dpy,
                               _EGLSync *sync, EGLenum mode);
   EGLBoolean (*GetSyncAttrib)(_EGLDriver *drv, _EGLDisplay *dpy,
                               _EGLSync *sync, EGLint attribute,
                               EGLAttrib *value);
   EGLint (*DupNativeFenceFDANDROID)(_EGLDriver *drv, _EGLDisplay *dpy,
                                     _EGLSync *sync);

   EGLBoolean (*SwapBuffersRegionNOK)(_EGLDriver *drv, _EGLDisplay *disp,
                                      _EGLSurface *surf, EGLint numRects,
                                      const EGLint *rects);

   _EGLImage *(*CreateDRMImageMESA)(_EGLDriver *drv, _EGLDisplay *disp,
                                    const EGLint *attr_list);
   EGLBoolean (*ExportDRMImageMESA)(_EGLDriver *drv, _EGLDisplay *disp,
                                    _EGLImage *img, EGLint *name,
                                    EGLint *handle, EGLint *stride);

   EGLBoolean (*BindWaylandDisplayWL)(_EGLDriver *drv, _EGLDisplay *disp,
                                      struct wl_display *display);
   EGLBoolean (*UnbindWaylandDisplayWL)(_EGLDriver *drv, _EGLDisplay *disp,
                                        struct wl_display *display);
   EGLBoolean (*QueryWaylandBufferWL)(_EGLDriver *drv, _EGLDisplay *displ,
                                      struct wl_resource *buffer,
                                      EGLint attribute, EGLint *value);

   struct wl_buffer *(*CreateWaylandBufferFromImageWL)(_EGLDriver *drv,
                                                       _EGLDisplay *disp,
                                                       _EGLImage *img);

   EGLBoolean (*SwapBuffersWithDamageEXT)(_EGLDriver *drv, _EGLDisplay *dpy,
                                          _EGLSurface *surface,
                                          const EGLint *rects, EGLint n_rects);

   EGLBoolean (*PostSubBufferNV)(_EGLDriver *drv, _EGLDisplay *disp,
                                 _EGLSurface *surface, EGLint x, EGLint y,
                                 EGLint width, EGLint height);

   EGLint (*QueryBufferAge)(_EGLDriver *drv,
                            _EGLDisplay *dpy, _EGLSurface *surface);
   EGLBoolean (*GetSyncValuesCHROMIUM)(_EGLDisplay *dpy, _EGLSurface *surface,
                                       EGLuint64KHR *ust, EGLuint64KHR *msc,
                                       EGLuint64KHR *sbc);

   EGLBoolean (*ExportDMABUFImageQueryMESA)(_EGLDriver *drv, _EGLDisplay *disp,
                                            _EGLImage *img, EGLint *fourcc,
                                            EGLint *nplanes,
                                            EGLuint64KHR *modifiers);
   EGLBoolean (*ExportDMABUFImageMESA)(_EGLDriver *drv, _EGLDisplay *disp,
                                       _EGLImage *img, EGLint *fds,
                                       EGLint *strides, EGLint *offsets);

   int (*GLInteropQueryDeviceInfo)(_EGLDisplay *dpy, _EGLContext *ctx,
                                   struct mesa_glinterop_device_info *out);
   int (*GLInteropExportObject)(_EGLDisplay *dpy, _EGLContext *ctx,
                                struct mesa_glinterop_export_in *in,
                                struct mesa_glinterop_export_out *out);

   EGLBoolean (*QueryDmaBufFormatsEXT)(_EGLDriver *drv, _EGLDisplay *dpy,
                                       EGLint max_formats, EGLint *formats,
                                       EGLint *num_formats);
   EGLBoolean (*QueryDmaBufModifiersEXT) (_EGLDriver *drv, _EGLDisplay *dpy,
                                          EGLint format, EGLint max_modifiers,
                                          EGLuint64KHR *modifiers,
                                          EGLBoolean *external_only,
                                          EGLint *num_modifiers);

   void (*SetBlobCacheFuncsANDROID) (_EGLDriver *drv, _EGLDisplay *dpy,
                                     EGLSetBlobFuncANDROID set,
                                     EGLGetBlobFuncANDROID get);
};

#ifdef __cplusplus
}
#endif

#endif /* EGLAPI_INCLUDED */
