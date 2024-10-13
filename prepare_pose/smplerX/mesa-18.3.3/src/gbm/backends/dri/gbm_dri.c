/*
 * Copyright © 2011 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Benjamin Franzke <benjaminfranzke@googlemail.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <assert.h>

#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>
#include <xf86drm.h>
#include <drm_fourcc.h>

#include <GL/gl.h> /* dri_interface needs GL types */
#include <GL/internal/dri_interface.h>

#include "gbm_driint.h"

#include "gbmint.h"
#include "loader.h"
#include "util/debug.h"
#include "util/macros.h"

/* For importing wl_buffer */
#if HAVE_WAYLAND_PLATFORM
#include "wayland-drm.h"
#endif

#ifndef DRM_FORMAT_MOD_INVALID
#define DRM_FORMAT_MOD_INVALID ((1ULL<<56) - 1)
#endif

#ifndef DRM_FORMAT_MOD_LINEAR
#define DRM_FORMAT_MOD_LINEAR 0
#endif

static __DRIimage *
dri_lookup_egl_image(__DRIscreen *screen, void *image, void *data)
{
   struct gbm_dri_device *dri = data;

   if (dri->lookup_image == NULL)
      return NULL;

   return dri->lookup_image(screen, image, dri->lookup_user_data);
}

static __DRIbuffer *
dri_get_buffers(__DRIdrawable * driDrawable,
		 int *width, int *height,
		 unsigned int *attachments, int count,
		 int *out_count, void *data)
{
   struct gbm_dri_surface *surf = data;
   struct gbm_dri_device *dri = gbm_dri_device(surf->base.gbm);

   if (dri->get_buffers == NULL)
      return NULL;

   return dri->get_buffers(driDrawable, width, height, attachments,
                           count, out_count, surf->dri_private);
}

static void
dri_flush_front_buffer(__DRIdrawable * driDrawable, void *data)
{
   struct gbm_dri_surface *surf = data;
   struct gbm_dri_device *dri = gbm_dri_device(surf->base.gbm);

   if (dri->flush_front_buffer != NULL)
      dri->flush_front_buffer(driDrawable, surf->dri_private);
}

static __DRIbuffer *
dri_get_buffers_with_format(__DRIdrawable * driDrawable,
                            int *width, int *height,
                            unsigned int *attachments, int count,
                            int *out_count, void *data)
{
   struct gbm_dri_surface *surf = data;
   struct gbm_dri_device *dri = gbm_dri_device(surf->base.gbm);

   if (dri->get_buffers_with_format == NULL)
      return NULL;

   return
      dri->get_buffers_with_format(driDrawable, width, height, attachments,
                                   count, out_count, surf->dri_private);
}

static int
image_get_buffers(__DRIdrawable *driDrawable,
                  unsigned int format,
                  uint32_t *stamp,
                  void *loaderPrivate,
                  uint32_t buffer_mask,
                  struct __DRIimageList *buffers)
{
   struct gbm_dri_surface *surf = loaderPrivate;
   struct gbm_dri_device *dri = gbm_dri_device(surf->base.gbm);

   if (dri->image_get_buffers == NULL)
      return 0;

   return dri->image_get_buffers(driDrawable, format, stamp,
                                 surf->dri_private, buffer_mask, buffers);
}

static void
swrast_get_drawable_info(__DRIdrawable *driDrawable,
                         int           *x,
                         int           *y,
                         int           *width,
                         int           *height,
                         void          *loaderPrivate)
{
   struct gbm_dri_surface *surf = loaderPrivate;

   *x = 0;
   *y = 0;
   *width = surf->base.width;
   *height = surf->base.height;
}

static void
swrast_put_image2(__DRIdrawable *driDrawable,
                  int            op,
                  int            x,
                  int            y,
                  int            width,
                  int            height,
                  int            stride,
                  char          *data,
                  void          *loaderPrivate)
{
   struct gbm_dri_surface *surf = loaderPrivate;
   struct gbm_dri_device *dri = gbm_dri_device(surf->base.gbm);

   dri->swrast_put_image2(driDrawable,
                          op, x, y,
                          width, height, stride,
                          data, surf->dri_private);
}

static void
swrast_put_image(__DRIdrawable *driDrawable,
                 int            op,
                 int            x,
                 int            y,
                 int            width,
                 int            height,
                 char          *data,
                 void          *loaderPrivate)
{
   return swrast_put_image2(driDrawable, op, x, y, width, height,
                            width * 4, data, loaderPrivate);
}

static void
swrast_get_image(__DRIdrawable *driDrawable,
                 int            x,
                 int            y,
                 int            width,
                 int            height,
                 char          *data,
                 void          *loaderPrivate)
{
   struct gbm_dri_surface *surf = loaderPrivate;
   struct gbm_dri_device *dri = gbm_dri_device(surf->base.gbm);

   dri->swrast_get_image(driDrawable,
                         x, y,
                         width, height,
                         data, surf->dri_private);
}

static const __DRIuseInvalidateExtension use_invalidate = {
   .base = { __DRI_USE_INVALIDATE, 1 }
};

static const __DRIimageLookupExtension image_lookup_extension = {
   .base = { __DRI_IMAGE_LOOKUP, 1 },

   .lookupEGLImage          = dri_lookup_egl_image
};

static const __DRIdri2LoaderExtension dri2_loader_extension = {
   .base = { __DRI_DRI2_LOADER, 3 },

   .getBuffers              = dri_get_buffers,
   .flushFrontBuffer        = dri_flush_front_buffer,
   .getBuffersWithFormat    = dri_get_buffers_with_format,
};

static const __DRIimageLoaderExtension image_loader_extension = {
   .base = { __DRI_IMAGE_LOADER, 1 },

   .getBuffers          = image_get_buffers,
   .flushFrontBuffer    = dri_flush_front_buffer,
};

static const __DRIswrastLoaderExtension swrast_loader_extension = {
   .base = { __DRI_SWRAST_LOADER, 2 },

   .getDrawableInfo = swrast_get_drawable_info,
   .putImage        = swrast_put_image,
   .getImage        = swrast_get_image,
   .putImage2       = swrast_put_image2
};

static const __DRIextension *gbm_dri_screen_extensions[] = {
   &image_lookup_extension.base,
   &use_invalidate.base,
   &dri2_loader_extension.base,
   &image_loader_extension.base,
   &swrast_loader_extension.base,
   NULL,
};

struct dri_extension_match {
   const char *name;
   int version;
   int offset;
   int optional;
};

static struct dri_extension_match dri_core_extensions[] = {
   { __DRI2_FLUSH, 1, offsetof(struct gbm_dri_device, flush) },
   { __DRI_IMAGE, 1, offsetof(struct gbm_dri_device, image) },
   { __DRI2_FENCE, 1, offsetof(struct gbm_dri_device, fence), 1 },
   { NULL, 0, 0 }
};

static struct dri_extension_match gbm_dri_device_extensions[] = {
   { __DRI_CORE, 1, offsetof(struct gbm_dri_device, core) },
   { __DRI_DRI2, 1, offsetof(struct gbm_dri_device, dri2) },
   { NULL, 0, 0 }
};

static struct dri_extension_match gbm_swrast_device_extensions[] = {
   { __DRI_CORE, 1, offsetof(struct gbm_dri_device, core), },
   { __DRI_SWRAST, 1, offsetof(struct gbm_dri_device, swrast) },
   { NULL, 0, 0 }
};

static int
dri_bind_extensions(struct gbm_dri_device *dri,
                    struct dri_extension_match *matches,
                    const __DRIextension **extensions)
{
   int i, j, ret = 0;
   void *field;

   for (i = 0; extensions[i]; i++) {
      for (j = 0; matches[j].name; j++) {
         if (strcmp(extensions[i]->name, matches[j].name) == 0 &&
             extensions[i]->version >= matches[j].version) {
            field = ((char *) dri + matches[j].offset);
            *(const __DRIextension **) field = extensions[i];
         }
      }
   }

   for (j = 0; matches[j].name; j++) {
      field = ((char *) dri + matches[j].offset);
      if ((*(const __DRIextension **) field == NULL) && !matches[j].optional) {
         ret = -1;
      }
   }

   return ret;
}

static const __DRIextension **
dri_open_driver(struct gbm_dri_device *dri)
{
   const __DRIextension **extensions = NULL;
   char path[PATH_MAX], *search_paths, *p, *next, *end;
   char *get_extensions_name;

   search_paths = NULL;
   /* don't allow setuid apps to use LIBGL_DRIVERS_PATH or GBM_DRIVERS_PATH */
   if (geteuid() == getuid()) {
      /* Read GBM_DRIVERS_PATH first for compatibility, but LIBGL_DRIVERS_PATH
       * is recommended over GBM_DRIVERS_PATH.
       */
      search_paths = getenv("GBM_DRIVERS_PATH");

      /* Read LIBGL_DRIVERS_PATH if GBM_DRIVERS_PATH was not set.
       * LIBGL_DRIVERS_PATH is recommended over GBM_DRIVERS_PATH.
       */
      if (search_paths == NULL) {
         search_paths = getenv("LIBGL_DRIVERS_PATH");
      }
   }
   if (search_paths == NULL)
      search_paths = DEFAULT_DRIVER_DIR;

   /* Temporarily work around dri driver libs that need symbols in libglapi
    * but don't automatically link it in.
    */
   /* XXX: Library name differs on per platforms basis. Update this as
    * osx/cygwin/windows/bsd gets support for GBM..
    */
   dlopen("libglapi.so.0", RTLD_LAZY | RTLD_GLOBAL);

   dri->driver = NULL;
   end = search_paths + strlen(search_paths);
   for (p = search_paths; p < end && dri->driver == NULL; p = next + 1) {
      int len;
      next = strchr(p, ':');
      if (next == NULL)
         next = end;

      len = next - p;
#if GLX_USE_TLS
      snprintf(path, sizeof path,
               "%.*s/tls/%s_dri.so", len, p, dri->driver_name);
      dri->driver = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
#endif
      if (dri->driver == NULL) {
         snprintf(path, sizeof path,
                  "%.*s/%s_dri.so", len, p, dri->driver_name);
         dri->driver = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
      }
      /* not need continue to loop all paths once the driver is found */
      if (dri->driver != NULL)
         break;
   }

   if (dri->driver == NULL) {
      fprintf(stderr, "gbm: failed to open any driver (search paths %s)\n",
              search_paths);
      fprintf(stderr, "gbm: Last dlopen error: %s\n", dlerror());
      return NULL;
   }

   get_extensions_name = loader_get_extensions_name(dri->driver_name);
   if (get_extensions_name) {
      const __DRIextension **(*get_extensions)(void);

      get_extensions = dlsym(dri->driver, get_extensions_name);
      free(get_extensions_name);

      if (get_extensions)
         extensions = get_extensions();
   }

   if (!extensions)
      extensions = dlsym(dri->driver, __DRI_DRIVER_EXTENSIONS);
   if (extensions == NULL) {
      fprintf(stderr, "gbm: driver exports no extensions (%s)", dlerror());
      dlclose(dri->driver);
   }

   return extensions;
}

static int
dri_load_driver(struct gbm_dri_device *dri)
{
   const __DRIextension **extensions;

   extensions = dri_open_driver(dri);
   if (!extensions)
      return -1;

   if (dri_bind_extensions(dri, gbm_dri_device_extensions, extensions) < 0) {
      dlclose(dri->driver);
      fprintf(stderr, "failed to bind extensions\n");
      return -1;
   }

   dri->driver_extensions = extensions;

   return 0;
}

static int
dri_load_driver_swrast(struct gbm_dri_device *dri)
{
   const __DRIextension **extensions;

   extensions = dri_open_driver(dri);
   if (!extensions)
      return -1;

   if (dri_bind_extensions(dri, gbm_swrast_device_extensions, extensions) < 0) {
      dlclose(dri->driver);
      fprintf(stderr, "failed to bind extensions\n");
      return -1;
   }

   dri->driver_extensions = extensions;

   return 0;
}

static int
dri_screen_create_dri2(struct gbm_dri_device *dri, char *driver_name)
{
   const __DRIextension **extensions;
   int ret = 0;

   dri->driver_name = driver_name;
   if (dri->driver_name == NULL)
      return -1;

   ret = dri_load_driver(dri);
   if (ret) {
      fprintf(stderr, "failed to load driver: %s\n", dri->driver_name);
      return ret;
   }

   dri->loader_extensions = gbm_dri_screen_extensions;

   if (dri->dri2 == NULL)
      return -1;

   if (dri->dri2->base.version >= 4) {
      dri->screen = dri->dri2->createNewScreen2(0, dri->base.fd,
                                                dri->loader_extensions,
                                                dri->driver_extensions,
                                                &dri->driver_configs, dri);
   } else {
      dri->screen = dri->dri2->createNewScreen(0, dri->base.fd,
                                               dri->loader_extensions,
                                               &dri->driver_configs, dri);
   }
   if (dri->screen == NULL)
      return -1;

   extensions = dri->core->getExtensions(dri->screen);
   if (dri_bind_extensions(dri, dri_core_extensions, extensions) < 0) {
      ret = -1;
      goto free_screen;
   }

   dri->lookup_image = NULL;
   dri->lookup_user_data = NULL;

   return 0;

free_screen:
   dri->core->destroyScreen(dri->screen);

   return ret;
}

static int
dri_screen_create_swrast(struct gbm_dri_device *dri)
{
   int ret;

   dri->driver_name = strdup("swrast");
   if (dri->driver_name == NULL)
      return -1;

   ret = dri_load_driver_swrast(dri);
   if (ret) {
      fprintf(stderr, "failed to load swrast driver\n");
      return ret;
   }

   dri->loader_extensions = gbm_dri_screen_extensions;

   if (dri->swrast == NULL)
      return -1;

   if (dri->swrast->base.version >= 4) {
      dri->screen = dri->swrast->createNewScreen2(0, dri->loader_extensions,
                                                  dri->driver_extensions,
                                                  &dri->driver_configs, dri);
   } else {
      dri->screen = dri->swrast->createNewScreen(0, dri->loader_extensions,
                                                 &dri->driver_configs, dri);
   }
   if (dri->screen == NULL)
      return -1;

   dri->lookup_image = NULL;
   dri->lookup_user_data = NULL;

   return 0;
}

static int
dri_screen_create(struct gbm_dri_device *dri)
{
   char *driver_name;

   driver_name = loader_get_driver_for_fd(dri->base.fd);
   if (!driver_name)
      return -1;

   return dri_screen_create_dri2(dri, driver_name);
}

static int
dri_screen_create_sw(struct gbm_dri_device *dri)
{
   char *driver_name;
   int ret;

   driver_name = strdup("kms_swrast");
   if (!driver_name)
      return -errno;

   ret = dri_screen_create_dri2(dri, driver_name);
   if (ret == 0)
      return ret;

   return dri_screen_create_swrast(dri);
}

static const struct gbm_dri_visual gbm_dri_visuals_table[] = {
   {
     GBM_FORMAT_R8, __DRI_IMAGE_FORMAT_R8,
     { 0x000000ff, 0x00000000, 0x00000000, 0x00000000 },
   },
   {
     GBM_FORMAT_GR88, __DRI_IMAGE_FORMAT_GR88,
     { 0x000000ff, 0x0000ff00, 0x00000000, 0x00000000 },
   },
   {
     GBM_FORMAT_ARGB1555, __DRI_IMAGE_FORMAT_ARGB1555,
     { 0x00007c00, 0x000003e0, 0x0000001f, 0x00008000 },
   },
   {
     GBM_FORMAT_RGB565, __DRI_IMAGE_FORMAT_RGB565,
     { 0x0000f800, 0x000007e0, 0x0000001f, 0x00000000 },
   },
   {
     GBM_FORMAT_XRGB8888, __DRI_IMAGE_FORMAT_XRGB8888,
     { 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000 },
   },
   {
     GBM_FORMAT_ARGB8888, __DRI_IMAGE_FORMAT_ARGB8888,
     { 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000 },
   },
   {
     GBM_FORMAT_XBGR8888, __DRI_IMAGE_FORMAT_XBGR8888,
     { 0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000 },
   },
   {
     GBM_FORMAT_ABGR8888, __DRI_IMAGE_FORMAT_ABGR8888,
     { 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000 },
   },
   {
     GBM_FORMAT_XRGB2101010, __DRI_IMAGE_FORMAT_XRGB2101010,
     { 0x3ff00000, 0x000ffc00, 0x000003ff, 0x00000000 },
   },
   {
     GBM_FORMAT_ARGB2101010, __DRI_IMAGE_FORMAT_ARGB2101010,
     { 0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000 },
   },
   {
     GBM_FORMAT_XBGR2101010, __DRI_IMAGE_FORMAT_XBGR2101010,
     { 0x000003ff, 0x000ffc00, 0x3ff00000, 0x00000000 },
   },
   {
     GBM_FORMAT_ABGR2101010, __DRI_IMAGE_FORMAT_ABGR2101010,
     { 0x000003ff, 0x000ffc00, 0x3ff00000, 0xc0000000 },
   },
};

/* The two GBM_BO_FORMAT_[XA]RGB8888 formats alias the GBM_FORMAT_*
 * formats of the same name. We want to accept them whenever someone
 * has a GBM format, but never return them to the user. */
static int
gbm_format_canonicalize(uint32_t gbm_format)
{
   switch (gbm_format) {
   case GBM_BO_FORMAT_XRGB8888:
      return GBM_FORMAT_XRGB8888;
   case GBM_BO_FORMAT_ARGB8888:
      return GBM_FORMAT_ARGB8888;
   default:
      return gbm_format;
   }
}

static int
gbm_format_to_dri_format(uint32_t gbm_format)
{
   int i;

   gbm_format = gbm_format_canonicalize(gbm_format);
   for (i = 0; i < ARRAY_SIZE(gbm_dri_visuals_table); i++) {
      if (gbm_dri_visuals_table[i].gbm_format == gbm_format)
         return gbm_dri_visuals_table[i].dri_image_format;
   }

   return 0;
}

static uint32_t
gbm_dri_to_gbm_format(int dri_format)
{
   int i;

   for (i = 0; i < ARRAY_SIZE(gbm_dri_visuals_table); i++) {
      if (gbm_dri_visuals_table[i].dri_image_format == dri_format)
         return gbm_dri_visuals_table[i].gbm_format;
   }

   return 0;
}

static int
gbm_dri_is_format_supported(struct gbm_device *gbm,
                            uint32_t format,
                            uint32_t usage)
{
   struct gbm_dri_device *dri = gbm_dri_device(gbm);
   int count;

   if ((usage & GBM_BO_USE_CURSOR) && (usage & GBM_BO_USE_RENDERING))
      return 0;

   format = gbm_format_canonicalize(format);
   if (gbm_format_to_dri_format(format) == 0)
      return 0;

   /* If there is no query, fall back to the small table which was originally
    * here. */
   if (dri->image->base.version <= 15 || !dri->image->queryDmaBufModifiers) {
      switch (format) {
      case GBM_FORMAT_XRGB8888:
      case GBM_FORMAT_ARGB8888:
      case GBM_FORMAT_XBGR8888:
         return 1;
      default:
         return 0;
      }
   }

   /* Check if the driver returns any modifiers for this format; since linear
    * is counted as a modifier, we will have at least one modifier for any
    * supported format. */
   if (!dri->image->queryDmaBufModifiers(dri->screen, format, 0, NULL, NULL,
                                         &count))
      return 0;

   return (count > 0);
}

static int
gbm_dri_get_format_modifier_plane_count(struct gbm_device *gbm,
                                        uint32_t format,
                                        uint64_t modifier)
{
   struct gbm_dri_device *dri = gbm_dri_device(gbm);
   uint64_t plane_count;

   if (dri->image->base.version < 16 ||
       !dri->image->queryDmaBufFormatModifierAttribs)
      return -1;

   format = gbm_format_canonicalize(format);
   if (gbm_format_to_dri_format(format) == 0)
      return -1;

   if (!dri->image->queryDmaBufFormatModifierAttribs(
         dri->screen, format, modifier,
         __DRI_IMAGE_FORMAT_MODIFIER_ATTRIB_PLANE_COUNT, &plane_count))
      return -1;

   return plane_count;
}

static int
gbm_dri_bo_write(struct gbm_bo *_bo, const void *buf, size_t count)
{
   struct gbm_dri_bo *bo = gbm_dri_bo(_bo);

   if (bo->image != NULL) {
      errno = EINVAL;
      return -1;
   }

   memcpy(bo->map, buf, count);

   return 0;
}

static int
gbm_dri_bo_get_fd(struct gbm_bo *_bo)
{
   struct gbm_dri_device *dri = gbm_dri_device(_bo->gbm);
   struct gbm_dri_bo *bo = gbm_dri_bo(_bo);
   int fd;

   if (bo->image == NULL)
      return -1;

   if (!dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_FD, &fd))
      return -1;

   return fd;
}

static int
get_number_planes(struct gbm_dri_device *dri, __DRIimage *image)
{
   int num_planes = 0;

   /* Dumb buffers are single-plane only. */
   if (!image)
      return 1;

   dri->image->queryImage(image, __DRI_IMAGE_ATTRIB_NUM_PLANES, &num_planes);

   if (num_planes <= 0)
      num_planes = 1;

   return num_planes;
}

static int
gbm_dri_bo_get_planes(struct gbm_bo *_bo)
{
   struct gbm_dri_device *dri = gbm_dri_device(_bo->gbm);
   struct gbm_dri_bo *bo = gbm_dri_bo(_bo);

   return get_number_planes(dri, bo->image);
}

static union gbm_bo_handle
gbm_dri_bo_get_handle_for_plane(struct gbm_bo *_bo, int plane)
{
   struct gbm_dri_device *dri = gbm_dri_device(_bo->gbm);
   struct gbm_dri_bo *bo = gbm_dri_bo(_bo);
   union gbm_bo_handle ret;
   ret.s32 = -1;

   if (!dri->image || dri->image->base.version < 13 || !dri->image->fromPlanar) {
      errno = ENOSYS;
      return ret;
   }

   if (plane >= get_number_planes(dri, bo->image)) {
      errno = EINVAL;
      return ret;
   }

   /* dumb BOs can only utilize non-planar formats */
   if (!bo->image) {
      assert(plane == 0);
      ret.s32 = bo->handle;
      return ret;
   }

   __DRIimage *image = dri->image->fromPlanar(bo->image, plane, NULL);
   if (image) {
      dri->image->queryImage(image, __DRI_IMAGE_ATTRIB_HANDLE, &ret.s32);
      dri->image->destroyImage(image);
   } else {
      assert(plane == 0);
      dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_HANDLE, &ret.s32);
   }

   return ret;
}

static uint32_t
gbm_dri_bo_get_stride(struct gbm_bo *_bo, int plane)
{
   struct gbm_dri_device *dri = gbm_dri_device(_bo->gbm);
   struct gbm_dri_bo *bo = gbm_dri_bo(_bo);
   __DRIimage *image;
   int stride = 0;

   if (!dri->image || dri->image->base.version < 11 || !dri->image->fromPlanar) {
      /* Preserve legacy behavior if plane is 0 */
      if (plane == 0)
         return _bo->stride;

      errno = ENOSYS;
      return 0;
   }

   if (plane >= get_number_planes(dri, bo->image)) {
      errno = EINVAL;
      return 0;
   }

   if (bo->image == NULL) {
      assert(plane == 0);
      return _bo->stride;
   }

   image = dri->image->fromPlanar(bo->image, plane, NULL);
   if (image) {
      dri->image->queryImage(image, __DRI_IMAGE_ATTRIB_STRIDE, &stride);
      dri->image->destroyImage(image);
   } else {
      assert(plane == 0);
      dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_STRIDE, &stride);
   }

   return (uint32_t)stride;
}

static uint32_t
gbm_dri_bo_get_offset(struct gbm_bo *_bo, int plane)
{
   struct gbm_dri_device *dri = gbm_dri_device(_bo->gbm);
   struct gbm_dri_bo *bo = gbm_dri_bo(_bo);
   int offset = 0;

   /* These error cases do not actually return an error code, as the user
    * will also fail to obtain the handle/FD from the BO. In that case, the
    * offset is irrelevant, as they have no buffer to offset into, so
    * returning 0 is harmless.
    */
   if (!dri->image || dri->image->base.version < 13 || !dri->image->fromPlanar)
      return 0;

   if (plane >= get_number_planes(dri, bo->image))
      return 0;

    /* Dumb images have no offset */
   if (bo->image == NULL) {
      assert(plane == 0);
      return 0;
   }

   __DRIimage *image = dri->image->fromPlanar(bo->image, plane, NULL);
   if (image) {
      dri->image->queryImage(image, __DRI_IMAGE_ATTRIB_OFFSET, &offset);
      dri->image->destroyImage(image);
   } else {
      assert(plane == 0);
      dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_OFFSET, &offset);
   }

   return (uint32_t)offset;
}

static uint64_t
gbm_dri_bo_get_modifier(struct gbm_bo *_bo)
{
   struct gbm_dri_device *dri = gbm_dri_device(_bo->gbm);
   struct gbm_dri_bo *bo = gbm_dri_bo(_bo);

   if (!dri->image || dri->image->base.version < 14) {
      errno = ENOSYS;
      return DRM_FORMAT_MOD_INVALID;
   }

   /* Dumb buffers have no modifiers */
   if (!bo->image)
      return DRM_FORMAT_MOD_LINEAR;

   uint64_t ret = 0;
   int mod;
   if (!dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_MODIFIER_UPPER,
                               &mod))
      return DRM_FORMAT_MOD_INVALID;

   ret = (uint64_t)mod << 32;

   if (!dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_MODIFIER_LOWER,
                               &mod))
      return DRM_FORMAT_MOD_INVALID;

   ret |= (uint64_t)(mod & 0xffffffff);

   return ret;
}

static void
gbm_dri_bo_destroy(struct gbm_bo *_bo)
{
   struct gbm_dri_device *dri = gbm_dri_device(_bo->gbm);
   struct gbm_dri_bo *bo = gbm_dri_bo(_bo);
   struct drm_mode_destroy_dumb arg;

   if (bo->image != NULL) {
      dri->image->destroyImage(bo->image);
   } else {
      gbm_dri_bo_unmap_dumb(bo);
      memset(&arg, 0, sizeof(arg));
      arg.handle = bo->handle;
      drmIoctl(dri->base.fd, DRM_IOCTL_MODE_DESTROY_DUMB, &arg);
   }

   free(bo);
}

static struct gbm_bo *
gbm_dri_bo_import(struct gbm_device *gbm,
                  uint32_t type, void *buffer, uint32_t usage)
{
   struct gbm_dri_device *dri = gbm_dri_device(gbm);
   struct gbm_dri_bo *bo;
   __DRIimage *image;
   unsigned dri_use = 0;
   int gbm_format;

   /* Required for query image WIDTH & HEIGHT */
   if (dri->image == NULL || dri->image->base.version < 4) {
      errno = ENOSYS;
      return NULL;
   }

   switch (type) {
#if HAVE_WAYLAND_PLATFORM
   case GBM_BO_IMPORT_WL_BUFFER:
   {
      struct wl_drm_buffer *wb;

      if (!dri->wl_drm) {
         errno = EINVAL;
         return NULL;
      }

      wb = wayland_drm_buffer_get(dri->wl_drm, (struct wl_resource *) buffer);
      if (!wb) {
         errno = EINVAL;
         return NULL;
      }

      image = dri->image->dupImage(wb->driver_buffer, NULL);

      /* GBM_FORMAT_* is identical to WL_DRM_FORMAT_*, so no conversion
       * required. */
      gbm_format = wb->format;
      break;
   }
#endif

   case GBM_BO_IMPORT_EGL_IMAGE:
   {
      int dri_format;
      if (dri->lookup_image == NULL) {
         errno = EINVAL;
         return NULL;
      }

      image = dri->lookup_image(dri->screen, buffer, dri->lookup_user_data);
      image = dri->image->dupImage(image, NULL);
      dri->image->queryImage(image, __DRI_IMAGE_ATTRIB_FORMAT, &dri_format);
      gbm_format = gbm_dri_to_gbm_format(dri_format);
      if (gbm_format == 0) {
         errno = EINVAL;
         dri->image->destroyImage(image);
         return NULL;
      }
      break;
   }

   case GBM_BO_IMPORT_FD:
   {
      struct gbm_import_fd_data *fd_data = buffer;
      int stride = fd_data->stride, offset = 0;
      int fourcc;

      /* GBM's GBM_FORMAT_* tokens are a strict superset of the DRI FourCC
       * tokens accepted by createImageFromFds, except for not supporting
       * the sARGB format. */
      fourcc = gbm_format_canonicalize(fd_data->format);

      image = dri->image->createImageFromFds(dri->screen,
                                             fd_data->width,
                                             fd_data->height,
                                             fourcc,
                                             &fd_data->fd, 1,
                                             &stride, &offset,
                                             NULL);
      if (image == NULL) {
         errno = EINVAL;
         return NULL;
      }
      gbm_format = fd_data->format;
      break;
   }

   case GBM_BO_IMPORT_FD_MODIFIER:
   {
      struct gbm_import_fd_modifier_data *fd_data = buffer;
      unsigned int error;
      int fourcc;

      /* Import with modifier requires createImageFromDmaBufs2 */
      if (dri->image == NULL || dri->image->base.version < 15 ||
          dri->image->createImageFromDmaBufs2 == NULL) {
         errno = ENOSYS;
         return NULL;
      }

      /* GBM's GBM_FORMAT_* tokens are a strict superset of the DRI FourCC
       * tokens accepted by createImageFromDmaBufs2, except for not supporting
       * the sARGB format. */
      fourcc = gbm_format_canonicalize(fd_data->format);

      image = dri->image->createImageFromDmaBufs2(dri->screen, fd_data->width,
                                                  fd_data->height, fourcc,
                                                  fd_data->modifier,
                                                  fd_data->fds,
                                                  fd_data->num_fds,
                                                  fd_data->strides,
                                                  fd_data->offsets,
                                                  0, 0, 0, 0,
                                                  &error, NULL);
      if (image == NULL) {
         errno = ENOSYS;
         return NULL;
      }

      gbm_format = fourcc;
      break;
   }

   default:
      errno = ENOSYS;
      return NULL;
   }


   bo = calloc(1, sizeof *bo);
   if (bo == NULL) {
      dri->image->destroyImage(image);
      return NULL;
   }

   bo->image = image;

   if (usage & GBM_BO_USE_SCANOUT)
      dri_use |= __DRI_IMAGE_USE_SCANOUT;
   if (usage & GBM_BO_USE_CURSOR)
      dri_use |= __DRI_IMAGE_USE_CURSOR;
   if (dri->image->base.version >= 2 &&
       !dri->image->validateUsage(bo->image, dri_use)) {
      errno = EINVAL;
      dri->image->destroyImage(bo->image);
      free(bo);
      return NULL;
   }

   bo->base.gbm = gbm;
   bo->base.format = gbm_format;

   dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_WIDTH,
                          (int*)&bo->base.width);
   dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_HEIGHT,
                          (int*)&bo->base.height);
   dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_STRIDE,
                          (int*)&bo->base.stride);
   dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_HANDLE,
                          &bo->base.handle.s32);

   return &bo->base;
}

static struct gbm_bo *
create_dumb(struct gbm_device *gbm,
                  uint32_t width, uint32_t height,
                  uint32_t format, uint32_t usage)
{
   struct gbm_dri_device *dri = gbm_dri_device(gbm);
   struct drm_mode_create_dumb create_arg;
   struct gbm_dri_bo *bo;
   struct drm_mode_destroy_dumb destroy_arg;
   int ret;
   int is_cursor, is_scanout;

   is_cursor = (usage & GBM_BO_USE_CURSOR) != 0 &&
      format == GBM_FORMAT_ARGB8888;
   is_scanout = (usage & GBM_BO_USE_SCANOUT) != 0 &&
      (format == GBM_FORMAT_XRGB8888 || format == GBM_FORMAT_XBGR8888);
   if (!is_cursor && !is_scanout) {
      errno = EINVAL;
      return NULL;
   }

   bo = calloc(1, sizeof *bo);
   if (bo == NULL)
      return NULL;

   memset(&create_arg, 0, sizeof(create_arg));
   create_arg.bpp = 32;
   create_arg.width = width;
   create_arg.height = height;

   ret = drmIoctl(dri->base.fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_arg);
   if (ret)
      goto free_bo;

   bo->base.gbm = gbm;
   bo->base.width = width;
   bo->base.height = height;
   bo->base.stride = create_arg.pitch;
   bo->base.format = format;
   bo->base.handle.u32 = create_arg.handle;
   bo->handle = create_arg.handle;
   bo->size = create_arg.size;

   if (gbm_dri_bo_map_dumb(bo) == NULL)
      goto destroy_dumb;

   return &bo->base;

destroy_dumb:
   memset(&destroy_arg, 0, sizeof destroy_arg);
   destroy_arg.handle = create_arg.handle;
   drmIoctl(dri->base.fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_arg);
free_bo:
   free(bo);

   return NULL;
}

static struct gbm_bo *
gbm_dri_bo_create(struct gbm_device *gbm,
                  uint32_t width, uint32_t height,
                  uint32_t format, uint32_t usage,
                  const uint64_t *modifiers,
                  const unsigned int count)
{
   struct gbm_dri_device *dri = gbm_dri_device(gbm);
   struct gbm_dri_bo *bo;
   int dri_format;
   unsigned dri_use = 0;

   /* Callers of this may specify a modifier, or a dri usage, but not both. The
    * newer modifier interface deprecates the older usage flags.
    */
   assert(!(usage && count));

   format = gbm_format_canonicalize(format);

   if (usage & GBM_BO_USE_WRITE || dri->image == NULL)
      return create_dumb(gbm, width, height, format, usage);

   bo = calloc(1, sizeof *bo);
   if (bo == NULL)
      return NULL;

   bo->base.gbm = gbm;
   bo->base.width = width;
   bo->base.height = height;
   bo->base.format = format;

   dri_format = gbm_format_to_dri_format(format);
   if (dri_format == 0) {
      errno = EINVAL;
      goto failed;
   }

   if (usage & GBM_BO_USE_SCANOUT)
      dri_use |= __DRI_IMAGE_USE_SCANOUT;
   if (usage & GBM_BO_USE_CURSOR)
      dri_use |= __DRI_IMAGE_USE_CURSOR;
   if (usage & GBM_BO_USE_LINEAR)
      dri_use |= __DRI_IMAGE_USE_LINEAR;

   /* Gallium drivers requires shared in order to get the handle/stride */
   dri_use |= __DRI_IMAGE_USE_SHARE;

   if (modifiers) {
      if (!dri->image || dri->image->base.version < 14 ||
          !dri->image->createImageWithModifiers) {
         fprintf(stderr, "Modifiers specified, but DRI is too old\n");
         errno = ENOSYS;
         goto failed;
      }

      /* It's acceptable to create an image with INVALID modifier in the list,
       * but it cannot be on the only modifier (since it will certainly fail
       * later). While we could easily catch this after modifier creation, doing
       * the check here is a convenient debug check likely pointing at whatever
       * interface the client is using to build its modifier list.
       */
      if (count == 1 && modifiers[0] == DRM_FORMAT_MOD_INVALID) {
         fprintf(stderr, "Only invalid modifier specified\n");
         errno = EINVAL;
         goto failed;
      }

      bo->image =
         dri->image->createImageWithModifiers(dri->screen,
                                              width, height,
                                              dri_format,
                                              modifiers, count,
                                              bo);

      if (bo->image) {
         /* The client passed in a list of invalid modifiers */
         assert(gbm_dri_bo_get_modifier(&bo->base) != DRM_FORMAT_MOD_INVALID);
      }
   } else {
      bo->image = dri->image->createImage(dri->screen, width, height,
                                          dri_format, dri_use, bo);
   }

   if (bo->image == NULL)
      goto failed;

   dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_HANDLE,
                          &bo->base.handle.s32);
   dri->image->queryImage(bo->image, __DRI_IMAGE_ATTRIB_STRIDE,
                          (int *) &bo->base.stride);

   return &bo->base;

failed:
   free(bo);
   return NULL;
}

static void *
gbm_dri_bo_map(struct gbm_bo *_bo,
              uint32_t x, uint32_t y,
              uint32_t width, uint32_t height,
              uint32_t flags, uint32_t *stride, void **map_data)
{
   struct gbm_dri_device *dri = gbm_dri_device(_bo->gbm);
   struct gbm_dri_bo *bo = gbm_dri_bo(_bo);

   /* If it's a dumb buffer, we already have a mapping */
   if (bo->map) {
      *map_data = (char *)bo->map + (bo->base.stride * y) + (x * 4);
      *stride = bo->base.stride;
      return *map_data;
   }

   if (!dri->image || dri->image->base.version < 12 || !dri->image->mapImage) {
      errno = ENOSYS;
      return NULL;
   }

   mtx_lock(&dri->mutex);
   if (!dri->context)
      dri->context = dri->dri2->createNewContext(dri->screen, NULL,
                                                 NULL, NULL);
   assert(dri->context);
   mtx_unlock(&dri->mutex);

   /* GBM flags and DRI flags are the same, so just pass them on */
   return dri->image->mapImage(dri->context, bo->image, x, y,
                               width, height, flags, (int *)stride,
                               map_data);
}

static void
gbm_dri_bo_unmap(struct gbm_bo *_bo, void *map_data)
{
   struct gbm_dri_device *dri = gbm_dri_device(_bo->gbm);
   struct gbm_dri_bo *bo = gbm_dri_bo(_bo);

   /* Check if it's a dumb buffer and check the pointer is in range */
   if (bo->map) {
      assert(map_data >= bo->map);
      assert(map_data < (bo->map + bo->size));
      return;
   }

   if (!dri->context || !dri->image ||
       dri->image->base.version < 12 || !dri->image->unmapImage)
      return;

   dri->image->unmapImage(dri->context, bo->image, map_data);

   /*
    * Not all DRI drivers use direct maps. They may queue up DMA operations
    * on the mapping context. Since there is no explicit gbm flush
    * mechanism, we need to flush here.
    */
   if (dri->flush->base.version >= 4)
      dri->flush->flush_with_flags(dri->context, NULL, __DRI2_FLUSH_CONTEXT, 0);
}


static struct gbm_surface *
gbm_dri_surface_create(struct gbm_device *gbm,
                       uint32_t width, uint32_t height,
		       uint32_t format, uint32_t flags,
                       const uint64_t *modifiers, const unsigned count)
{
   struct gbm_dri_device *dri = gbm_dri_device(gbm);
   struct gbm_dri_surface *surf;

   if (modifiers &&
       (!dri->image || dri->image->base.version < 14 ||
        !dri->image->createImageWithModifiers)) {
      errno = ENOSYS;
      return NULL;
   }

   if (count)
      assert(modifiers);

   /* It's acceptable to create an image with INVALID modifier in the list,
    * but it cannot be on the only modifier (since it will certainly fail
    * later). While we could easily catch this after modifier creation, doing
    * the check here is a convenient debug check likely pointing at whatever
    * interface the client is using to build its modifier list.
    */
   if (count == 1 && modifiers[0] == DRM_FORMAT_MOD_INVALID) {
      fprintf(stderr, "Only invalid modifier specified\n");
      errno = EINVAL;
   }

   surf = calloc(1, sizeof *surf);
   if (surf == NULL) {
      errno = ENOMEM;
      return NULL;
   }

   surf->base.gbm = gbm;
   surf->base.width = width;
   surf->base.height = height;
   surf->base.format = gbm_format_canonicalize(format);
   surf->base.flags = flags;
   if (!modifiers) {
      assert(!count);
      return &surf->base;
   }

   surf->base.modifiers = calloc(count, sizeof(*modifiers));
   if (count && !surf->base.modifiers) {
      errno = ENOMEM;
      free(surf);
      return NULL;
   }

   /* TODO: We are deferring validation of modifiers until the image is actually
    * created. This deferred creation can fail due to a modifier-format
    * mismatch. The result is the client has a surface but no object to back it.
    */
   surf->base.count = count;
   memcpy(surf->base.modifiers, modifiers, count * sizeof(*modifiers));

   return &surf->base;
}

static void
gbm_dri_surface_destroy(struct gbm_surface *_surf)
{
   struct gbm_dri_surface *surf = gbm_dri_surface(_surf);

   free(surf->base.modifiers);
   free(surf);
}

static void
dri_destroy(struct gbm_device *gbm)
{
   struct gbm_dri_device *dri = gbm_dri_device(gbm);
   unsigned i;

   if (dri->context)
      dri->core->destroyContext(dri->context);

   dri->core->destroyScreen(dri->screen);
   for (i = 0; dri->driver_configs[i]; i++)
      free((__DRIconfig *) dri->driver_configs[i]);
   free(dri->driver_configs);
   dlclose(dri->driver);
   free(dri->driver_name);

   free(dri);
}

static struct gbm_device *
dri_device_create(int fd)
{
   struct gbm_dri_device *dri;
   int ret;
   bool force_sw;

   dri = calloc(1, sizeof *dri);
   if (!dri)
      return NULL;

   dri->base.fd = fd;
   dri->base.bo_create = gbm_dri_bo_create;
   dri->base.bo_import = gbm_dri_bo_import;
   dri->base.bo_map = gbm_dri_bo_map;
   dri->base.bo_unmap = gbm_dri_bo_unmap;
   dri->base.is_format_supported = gbm_dri_is_format_supported;
   dri->base.get_format_modifier_plane_count =
      gbm_dri_get_format_modifier_plane_count;
   dri->base.bo_write = gbm_dri_bo_write;
   dri->base.bo_get_fd = gbm_dri_bo_get_fd;
   dri->base.bo_get_planes = gbm_dri_bo_get_planes;
   dri->base.bo_get_handle = gbm_dri_bo_get_handle_for_plane;
   dri->base.bo_get_stride = gbm_dri_bo_get_stride;
   dri->base.bo_get_offset = gbm_dri_bo_get_offset;
   dri->base.bo_get_modifier = gbm_dri_bo_get_modifier;
   dri->base.bo_destroy = gbm_dri_bo_destroy;
   dri->base.destroy = dri_destroy;
   dri->base.surface_create = gbm_dri_surface_create;
   dri->base.surface_destroy = gbm_dri_surface_destroy;

   dri->base.name = "drm";

   dri->visual_table = gbm_dri_visuals_table;
   dri->num_visuals = ARRAY_SIZE(gbm_dri_visuals_table);

   mtx_init(&dri->mutex, mtx_plain);

   force_sw = env_var_as_boolean("GBM_ALWAYS_SOFTWARE", false);
   if (!force_sw) {
      ret = dri_screen_create(dri);
      if (ret)
         ret = dri_screen_create_sw(dri);
   } else {
      ret = dri_screen_create_sw(dri);
   }

   if (ret)
      goto err_dri;

   return &dri->base;

err_dri:
   free(dri);

   return NULL;
}

struct gbm_backend gbm_dri_backend = {
   .backend_name = "dri",
   .create_device = dri_device_create,
};
