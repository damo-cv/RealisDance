/*
 * Copyright (C) 2016 Christian Gmeiner <christian.gmeiner@gmail.com>
 * Copyright (C) 2017 Broadcom
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <fcntl.h>
#include <unistd.h>

#include "pl111_drm_public.h"
#include "vc4/drm/vc4_drm_public.h"
#include "xf86drm.h"

#include "pipe/p_screen.h"
#include "renderonly/renderonly.h"

struct pipe_screen *pl111_drm_screen_create(int fd)
{
   struct renderonly ro = {
      /* Passes the vc4-allocated BO through to the pl111 DRM device using
       * PRIME buffer sharing.  The VC4 BO must be linear, which the SCANOUT
       * flag on allocation will have ensured.
       */
      .create_for_resource = renderonly_create_gpu_import_for_resource,
      .kms_fd = fd,
      .gpu_fd = drmOpenWithType("vc4", NULL, DRM_NODE_RENDER),
   };

   if (ro.gpu_fd < 0)
      return NULL;

   struct pipe_screen *screen = vc4_drm_screen_create_renderonly(&ro);
   if (!screen)
      close(ro.gpu_fd);

   return screen;
}
