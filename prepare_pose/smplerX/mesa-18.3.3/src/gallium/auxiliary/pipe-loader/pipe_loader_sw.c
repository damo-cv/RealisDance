/**************************************************************************
 *
 * Copyright 2012 Francisco Jerez
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifdef HAVE_PIPE_LOADER_KMS
#include <fcntl.h>
#endif

#include "pipe_loader_priv.h"

#include "util/u_memory.h"
#include "util/u_dl.h"
#include "sw/dri/dri_sw_winsys.h"
#include "sw/kms-dri/kms_dri_sw_winsys.h"
#include "sw/null/null_sw_winsys.h"
#include "sw/wrapper/wrapper_sw_winsys.h"
#include "target-helpers/sw_helper_public.h"
#include "state_tracker/drisw_api.h"
#include "state_tracker/sw_driver.h"
#include "state_tracker/sw_winsys.h"

struct pipe_loader_sw_device {
   struct pipe_loader_device base;
   const struct sw_driver_descriptor *dd;
#ifndef GALLIUM_STATIC_TARGETS
   struct util_dl_library *lib;
#endif
   struct sw_winsys *ws;
   int fd;
};

#define pipe_loader_sw_device(dev) ((struct pipe_loader_sw_device *)dev)

static const struct pipe_loader_ops pipe_loader_sw_ops;

#ifdef GALLIUM_STATIC_TARGETS
static const struct sw_driver_descriptor driver_descriptors = {
   .create_screen = sw_screen_create,
   .winsys = {
#ifdef HAVE_PIPE_LOADER_DRI
      {
         .name = "dri",
         .create_winsys = dri_create_sw_winsys,
      },
#endif
#ifdef HAVE_PIPE_LOADER_KMS
      {
         .name = "kms_dri",
         .create_winsys = kms_dri_create_winsys,
      },
#endif
/**
 * XXX: Do not include these two for non autotools builds.
 * They don't have neither opencl nor nine, where these are used.
 */
#ifndef DROP_PIPE_LOADER_MISC
      {
         .name = "null",
         .create_winsys = null_sw_create,
      },
      {
         .name = "wrapped",
         .create_winsys = wrapper_sw_winsys_wrap_pipe_screen,
      },
#endif
      { 0 },
   }
};
#endif

static bool
pipe_loader_sw_probe_init_common(struct pipe_loader_sw_device *sdev)
{
   sdev->base.type = PIPE_LOADER_DEVICE_SOFTWARE;
   sdev->base.driver_name = "swrast";
   sdev->base.ops = &pipe_loader_sw_ops;
   sdev->fd = -1;

#ifdef GALLIUM_STATIC_TARGETS
   sdev->dd = &driver_descriptors;
   if (!sdev->dd)
      return false;
#else
   sdev->lib = pipe_loader_find_module("swrast", PIPE_SEARCH_DIR);
   if (!sdev->lib)
      return false;

   sdev->dd = (const struct sw_driver_descriptor *)
      util_dl_get_proc_address(sdev->lib, "swrast_driver_descriptor");

   if (!sdev->dd){
      util_dl_close(sdev->lib);
      sdev->lib = NULL;
      return false;
   }
#endif

   return true;
}

static void
pipe_loader_sw_probe_teardown_common(struct pipe_loader_sw_device *sdev)
{
#ifndef GALLIUM_STATIC_TARGETS
   if (sdev->lib)
      util_dl_close(sdev->lib);
#endif
}

#ifdef HAVE_PIPE_LOADER_DRI
bool
pipe_loader_sw_probe_dri(struct pipe_loader_device **devs, const struct drisw_loader_funcs *drisw_lf)
{
   struct pipe_loader_sw_device *sdev = CALLOC_STRUCT(pipe_loader_sw_device);
   int i;

   if (!sdev)
      return false;

   if (!pipe_loader_sw_probe_init_common(sdev))
      goto fail;

   for (i = 0; sdev->dd->winsys[i].name; i++) {
      if (strcmp(sdev->dd->winsys[i].name, "dri") == 0) {
         sdev->ws = sdev->dd->winsys[i].create_winsys(drisw_lf);
         break;
      }
   }
   if (!sdev->ws)
      goto fail;

   *devs = &sdev->base;
   return true;

fail:
   pipe_loader_sw_probe_teardown_common(sdev);
   FREE(sdev);
   return false;
}
#endif

#ifdef HAVE_PIPE_LOADER_KMS
bool
pipe_loader_sw_probe_kms(struct pipe_loader_device **devs, int fd)
{
   struct pipe_loader_sw_device *sdev = CALLOC_STRUCT(pipe_loader_sw_device);
   int i;

   if (!sdev)
      return false;

   if (!pipe_loader_sw_probe_init_common(sdev))
      goto fail;

   if (fd < 0 || (sdev->fd = fcntl(fd, F_DUPFD_CLOEXEC, 3)) < 0)
      goto fail;

   for (i = 0; sdev->dd->winsys[i].name; i++) {
      if (strcmp(sdev->dd->winsys[i].name, "kms_dri") == 0) {
         sdev->ws = sdev->dd->winsys[i].create_winsys(sdev->fd);
         break;
      }
   }
   if (!sdev->ws)
      goto fail;

   *devs = &sdev->base;
   return true;

fail:
   pipe_loader_sw_probe_teardown_common(sdev);
   if (sdev->fd != -1)
      close(sdev->fd);
   FREE(sdev);
   return false;
}
#endif

bool
pipe_loader_sw_probe_null(struct pipe_loader_device **devs)
{
   struct pipe_loader_sw_device *sdev = CALLOC_STRUCT(pipe_loader_sw_device);
   int i;

   if (!sdev)
      return false;

   if (!pipe_loader_sw_probe_init_common(sdev))
      goto fail;

   for (i = 0; sdev->dd->winsys[i].name; i++) {
      if (strcmp(sdev->dd->winsys[i].name, "null") == 0) {
         sdev->ws = sdev->dd->winsys[i].create_winsys();
         break;
      }
   }
   if (!sdev->ws)
      goto fail;

   *devs = &sdev->base;
   return true;

fail:
   pipe_loader_sw_probe_teardown_common(sdev);
   FREE(sdev);
   return false;
}

int
pipe_loader_sw_probe(struct pipe_loader_device **devs, int ndev)
{
   int i = 1;

   if (i <= ndev) {
      if (!pipe_loader_sw_probe_null(devs)) {
         i--;
      }
   }

   return i;
}

boolean
pipe_loader_sw_probe_wrapped(struct pipe_loader_device **dev,
                             struct pipe_screen *screen)
{
   struct pipe_loader_sw_device *sdev = CALLOC_STRUCT(pipe_loader_sw_device);
   int i;

   if (!sdev)
      return false;

   if (!pipe_loader_sw_probe_init_common(sdev))
      goto fail;

   for (i = 0; sdev->dd->winsys[i].name; i++) {
      if (strcmp(sdev->dd->winsys[i].name, "wrapped") == 0) {
         sdev->ws = sdev->dd->winsys[i].create_winsys(screen);
         break;
      }
   }
   if (!sdev->ws)
      goto fail;

   *dev = &sdev->base;
   return true;

fail:
   pipe_loader_sw_probe_teardown_common(sdev);
   FREE(sdev);
   return false;
}

static void
pipe_loader_sw_release(struct pipe_loader_device **dev)
{
   MAYBE_UNUSED struct pipe_loader_sw_device *sdev =
      pipe_loader_sw_device(*dev);

#ifndef GALLIUM_STATIC_TARGETS
   if (sdev->lib)
      util_dl_close(sdev->lib);
#endif

#ifdef HAVE_PIPE_LOADER_KMS
   if (sdev->fd != -1)
      close(sdev->fd);
#endif

   pipe_loader_base_release(dev);
}

static const struct drm_conf_ret *
pipe_loader_sw_configuration(struct pipe_loader_device *dev,
                             enum drm_conf conf)
{
   return NULL;
}

static struct pipe_screen *
pipe_loader_sw_create_screen(struct pipe_loader_device *dev,
                             const struct pipe_screen_config *config)
{
   struct pipe_loader_sw_device *sdev = pipe_loader_sw_device(dev);
   struct pipe_screen *screen;

   screen = sdev->dd->create_screen(sdev->ws);
   if (!screen)
      sdev->ws->destroy(sdev->ws);

   return screen;
}

static const struct pipe_loader_ops pipe_loader_sw_ops = {
   .create_screen = pipe_loader_sw_create_screen,
   .configuration = pipe_loader_sw_configuration,
   .release = pipe_loader_sw_release
};
