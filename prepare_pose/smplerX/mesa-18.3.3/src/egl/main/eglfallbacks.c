/**************************************************************************
 *
 * Copyright 2010 LunarG, Inc.
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


#include <string.h>
#include "egltypedefs.h"
#include "egldriver.h"
#include "eglconfig.h"
#include "eglcontext.h"
#include "eglsurface.h"
#include "eglsync.h"


static EGLBoolean
_eglReturnFalse(void)
{
   return EGL_FALSE;
}


/**
 * Plug all the available fallback routines into the given driver's
 * dispatch table.
 */
void
_eglInitDriverFallbacks(_EGLDriver *drv)
{
   /* the driver has to implement these */
   drv->API.Initialize = NULL;
   drv->API.Terminate = NULL;

   drv->API.GetConfigs = _eglGetConfigs;
   drv->API.ChooseConfig = _eglChooseConfig;
   drv->API.GetConfigAttrib = _eglGetConfigAttrib;

   drv->API.CreateContext = (void*) _eglReturnFalse;
   drv->API.DestroyContext = (void*) _eglReturnFalse;
   drv->API.MakeCurrent = (void*) _eglReturnFalse;
   drv->API.QueryContext = _eglQueryContext;

   drv->API.CreateWindowSurface = (void*) _eglReturnFalse;
   drv->API.CreatePixmapSurface = (void*) _eglReturnFalse;
   drv->API.CreatePbufferSurface = (void*) _eglReturnFalse;
   drv->API.CreatePbufferFromClientBuffer =
      (void*) _eglReturnFalse;
   drv->API.DestroySurface = (void*) _eglReturnFalse;
   drv->API.QuerySurface = _eglQuerySurface;
   drv->API.SurfaceAttrib = _eglSurfaceAttrib;

   drv->API.BindTexImage = (void*) _eglReturnFalse;
   drv->API.ReleaseTexImage = (void*) _eglReturnFalse;
   drv->API.CopyBuffers = (void*) _eglReturnFalse;
   drv->API.SwapBuffers = (void*) _eglReturnFalse;
   drv->API.SetDamageRegion = (void*) _eglReturnFalse;
   drv->API.SwapInterval = _eglSwapInterval;

   drv->API.WaitClient = (void*) _eglReturnFalse;
   drv->API.WaitNative = (void*) _eglReturnFalse;
   drv->API.GetProcAddress = (void*) _eglReturnFalse;

   drv->API.CreateImageKHR = NULL;
   drv->API.DestroyImageKHR = NULL;

   drv->API.CreateSyncKHR = NULL;
   drv->API.DestroySyncKHR = NULL;
   drv->API.ClientWaitSyncKHR = NULL;
   drv->API.WaitSyncKHR = NULL;
   drv->API.SignalSyncKHR = NULL;
   drv->API.GetSyncAttrib = _eglGetSyncAttrib;
   drv->API.DupNativeFenceFDANDROID = NULL;

   drv->API.CreateDRMImageMESA = NULL;
   drv->API.ExportDRMImageMESA = NULL;

   drv->API.SwapBuffersRegionNOK = NULL;

   drv->API.ExportDMABUFImageQueryMESA = NULL;
   drv->API.ExportDMABUFImageMESA = NULL;
}
