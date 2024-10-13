/*
 * Copyright (C) 1999-2007  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <dlfcn.h>
#include "util/u_memory.h"
#include "pipe/p_screen.h"
#include "state_tracker/st_texture.h"
#include "state_tracker/st_context.h"
#include "state_tracker/st_cb_fbo.h"
#include "main/texobj.h"

#include "dri_helpers.h"

static bool
dri2_is_opencl_interop_loaded_locked(struct dri_screen *screen)
{
   return screen->opencl_dri_event_add_ref &&
          screen->opencl_dri_event_release &&
          screen->opencl_dri_event_wait &&
          screen->opencl_dri_event_get_fence;
}

static bool
dri2_load_opencl_interop(struct dri_screen *screen)
{
#if defined(RTLD_DEFAULT)
   bool success;

   mtx_lock(&screen->opencl_func_mutex);

   if (dri2_is_opencl_interop_loaded_locked(screen)) {
      mtx_unlock(&screen->opencl_func_mutex);
      return true;
   }

   screen->opencl_dri_event_add_ref =
      dlsym(RTLD_DEFAULT, "opencl_dri_event_add_ref");
   screen->opencl_dri_event_release =
      dlsym(RTLD_DEFAULT, "opencl_dri_event_release");
   screen->opencl_dri_event_wait =
      dlsym(RTLD_DEFAULT, "opencl_dri_event_wait");
   screen->opencl_dri_event_get_fence =
      dlsym(RTLD_DEFAULT, "opencl_dri_event_get_fence");

   success = dri2_is_opencl_interop_loaded_locked(screen);
   mtx_unlock(&screen->opencl_func_mutex);
   return success;
#else
   return false;
#endif
}

struct dri2_fence {
   struct dri_screen *driscreen;
   struct pipe_fence_handle *pipe_fence;
   void *cl_event;
};

static unsigned dri2_fence_get_caps(__DRIscreen *_screen)
{
   struct dri_screen *driscreen = dri_screen(_screen);
   struct pipe_screen *screen = driscreen->base.screen;
   unsigned caps = 0;

   if (screen->get_param(screen, PIPE_CAP_NATIVE_FENCE_FD))
      caps |= __DRI_FENCE_CAP_NATIVE_FD;

   return caps;
}

static void *
dri2_create_fence(__DRIcontext *_ctx)
{
   struct st_context_iface *stapi = dri_context(_ctx)->st;
   struct dri2_fence *fence = CALLOC_STRUCT(dri2_fence);

   if (!fence)
      return NULL;

   stapi->flush(stapi, 0, &fence->pipe_fence);

   if (!fence->pipe_fence) {
      FREE(fence);
      return NULL;
   }

   fence->driscreen = dri_screen(_ctx->driScreenPriv);
   return fence;
}

static void *
dri2_create_fence_fd(__DRIcontext *_ctx, int fd)
{
   struct st_context_iface *stapi = dri_context(_ctx)->st;
   struct pipe_context *ctx = stapi->pipe;
   struct dri2_fence *fence = CALLOC_STRUCT(dri2_fence);

   if (fd == -1) {
      /* exporting driver created fence, flush: */
      stapi->flush(stapi, ST_FLUSH_FENCE_FD, &fence->pipe_fence);
   } else {
      /* importing a foreign fence fd: */
      ctx->create_fence_fd(ctx, &fence->pipe_fence, fd, PIPE_FD_TYPE_NATIVE_SYNC);
   }
   if (!fence->pipe_fence) {
      FREE(fence);
      return NULL;
   }

   fence->driscreen = dri_screen(_ctx->driScreenPriv);
   return fence;
}

static int
dri2_get_fence_fd(__DRIscreen *_screen, void *_fence)
{
   struct dri_screen *driscreen = dri_screen(_screen);
   struct pipe_screen *screen = driscreen->base.screen;
   struct dri2_fence *fence = (struct dri2_fence*)_fence;

   return screen->fence_get_fd(screen, fence->pipe_fence);
}

static void *
dri2_get_fence_from_cl_event(__DRIscreen *_screen, intptr_t cl_event)
{
   struct dri_screen *driscreen = dri_screen(_screen);
   struct dri2_fence *fence;

   if (!dri2_load_opencl_interop(driscreen))
      return NULL;

   fence = CALLOC_STRUCT(dri2_fence);
   if (!fence)
      return NULL;

   fence->cl_event = (void*)cl_event;

   if (!driscreen->opencl_dri_event_add_ref(fence->cl_event)) {
      free(fence);
      return NULL;
   }

   fence->driscreen = driscreen;
   return fence;
}

static void
dri2_destroy_fence(__DRIscreen *_screen, void *_fence)
{
   struct dri_screen *driscreen = dri_screen(_screen);
   struct pipe_screen *screen = driscreen->base.screen;
   struct dri2_fence *fence = (struct dri2_fence*)_fence;

   if (fence->pipe_fence)
      screen->fence_reference(screen, &fence->pipe_fence, NULL);
   else if (fence->cl_event)
      driscreen->opencl_dri_event_release(fence->cl_event);
   else
      assert(0);

   FREE(fence);
}

static GLboolean
dri2_client_wait_sync(__DRIcontext *_ctx, void *_fence, unsigned flags,
                      uint64_t timeout)
{
   struct dri2_fence *fence = (struct dri2_fence*)_fence;
   struct dri_screen *driscreen = fence->driscreen;
   struct pipe_screen *screen = driscreen->base.screen;

   /* No need to flush. The context was flushed when the fence was created. */

   if (fence->pipe_fence)
      return screen->fence_finish(screen, NULL, fence->pipe_fence, timeout);
   else if (fence->cl_event) {
      struct pipe_fence_handle *pipe_fence =
         driscreen->opencl_dri_event_get_fence(fence->cl_event);

      if (pipe_fence)
         return screen->fence_finish(screen, NULL, pipe_fence, timeout);
      else
         return driscreen->opencl_dri_event_wait(fence->cl_event, timeout);
   }
   else {
      assert(0);
      return false;
   }
}

static void
dri2_server_wait_sync(__DRIcontext *_ctx, void *_fence, unsigned flags)
{
   struct pipe_context *ctx = dri_context(_ctx)->st->pipe;
   struct dri2_fence *fence = (struct dri2_fence*)_fence;

   /* We might be called here with a NULL fence as a result of WaitSyncKHR
    * on a EGL_KHR_reusable_sync fence. Nothing to do here in such case.
    */
   if (!fence)
      return;

   if (ctx->fence_server_sync)
      ctx->fence_server_sync(ctx, fence->pipe_fence);
}

const __DRI2fenceExtension dri2FenceExtension = {
   .base = { __DRI2_FENCE, 2 },

   .create_fence = dri2_create_fence,
   .get_fence_from_cl_event = dri2_get_fence_from_cl_event,
   .destroy_fence = dri2_destroy_fence,
   .client_wait_sync = dri2_client_wait_sync,
   .server_wait_sync = dri2_server_wait_sync,
   .get_capabilities = dri2_fence_get_caps,
   .create_fence_fd = dri2_create_fence_fd,
   .get_fence_fd = dri2_get_fence_fd,
};

__DRIimage *
dri2_lookup_egl_image(struct dri_screen *screen, void *handle)
{
   const __DRIimageLookupExtension *loader = screen->sPriv->dri2.image;
   __DRIimage *img;

   if (!loader->lookupEGLImage)
      return NULL;

   img = loader->lookupEGLImage(screen->sPriv,
				handle, screen->sPriv->loaderPrivate);

   return img;
}

__DRIimage *
dri2_create_image_from_renderbuffer2(__DRIcontext *context,
				     int renderbuffer, void *loaderPrivate,
                                     unsigned *error)
{
   struct gl_context *ctx = ((struct st_context *)dri_context(context)->st)->ctx;
   struct gl_renderbuffer *rb;
   struct pipe_resource *tex;
   __DRIimage *img;

   /* Section 3.9 (EGLImage Specification and Management) of the EGL 1.5
    * specification says:
    *
    *   "If target is EGL_GL_RENDERBUFFER and buffer is not the name of a
    *    renderbuffer object, or if buffer is the name of a multisampled
    *    renderbuffer object, the error EGL_BAD_PARAMETER is generated."
    *
    *   "If target is EGL_GL_TEXTURE_2D , EGL_GL_TEXTURE_CUBE_MAP_*,
    *    EGL_GL_RENDERBUFFER or EGL_GL_TEXTURE_3D and buffer refers to the
    *    default GL texture object (0) for the corresponding GL target, the
    *    error EGL_BAD_PARAMETER is generated."
    *   (rely on _mesa_lookup_renderbuffer returning NULL in this case)
    */
   rb = _mesa_lookup_renderbuffer(ctx, renderbuffer);
   if (!rb || rb->NumSamples > 0) {
      *error = __DRI_IMAGE_ERROR_BAD_PARAMETER;
      return NULL;
   }

   tex = st_get_renderbuffer_resource(rb);
   if (!tex) {
      *error = __DRI_IMAGE_ERROR_BAD_PARAMETER;
      return NULL;
   }

   img = CALLOC_STRUCT(__DRIimageRec);
   if (!img) {
      *error = __DRI_IMAGE_ERROR_BAD_ALLOC;
      return NULL;
   }

   img->dri_format = driGLFormatToImageFormat(rb->Format);
   img->loader_private = loaderPrivate;

   pipe_resource_reference(&img->texture, tex);

   *error = __DRI_IMAGE_ERROR_SUCCESS;
   return img;
}

__DRIimage *
dri2_create_image_from_renderbuffer(__DRIcontext *context,
				    int renderbuffer, void *loaderPrivate)
{
   unsigned error;
   return dri2_create_image_from_renderbuffer2(context, renderbuffer,
                                               loaderPrivate, &error);
}

void
dri2_destroy_image(__DRIimage *img)
{
   pipe_resource_reference(&img->texture, NULL);
   FREE(img);
}


__DRIimage *
dri2_create_from_texture(__DRIcontext *context, int target, unsigned texture,
                         int depth, int level, unsigned *error,
                         void *loaderPrivate)
{
   __DRIimage *img;
   struct gl_context *ctx = ((struct st_context *)dri_context(context)->st)->ctx;
   struct gl_texture_object *obj;
   struct pipe_resource *tex;
   GLuint face = 0;

   obj = _mesa_lookup_texture(ctx, texture);
   if (!obj || obj->Target != target) {
      *error = __DRI_IMAGE_ERROR_BAD_PARAMETER;
      return NULL;
   }

   tex = st_get_texobj_resource(obj);
   if (!tex) {
      *error = __DRI_IMAGE_ERROR_BAD_PARAMETER;
      return NULL;
   }

   if (target == GL_TEXTURE_CUBE_MAP)
      face = depth;

   _mesa_test_texobj_completeness(ctx, obj);
   if (!obj->_BaseComplete || (level > 0 && !obj->_MipmapComplete)) {
      *error = __DRI_IMAGE_ERROR_BAD_PARAMETER;
      return NULL;
   }

   if (level < obj->BaseLevel || level > obj->_MaxLevel) {
      *error = __DRI_IMAGE_ERROR_BAD_MATCH;
      return NULL;
   }

   if (target == GL_TEXTURE_3D && obj->Image[face][level]->Depth < depth) {
      *error = __DRI_IMAGE_ERROR_BAD_MATCH;
      return NULL;
   }

   img = CALLOC_STRUCT(__DRIimageRec);
   if (!img) {
      *error = __DRI_IMAGE_ERROR_BAD_ALLOC;
      return NULL;
   }

   img->level = level;
   img->layer = depth;
   img->dri_format = driGLFormatToImageFormat(obj->Image[face][level]->TexFormat);

   img->loader_private = loaderPrivate;

   pipe_resource_reference(&img->texture, tex);

   *error = __DRI_IMAGE_ERROR_SUCCESS;
   return img;
}

/* vim: set sw=3 ts=8 sts=3 expandtab: */
