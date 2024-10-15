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

#include "pipe_loader_priv.h"

#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_string.h"
#include "util/u_dl.h"
#include "util/xmlconfig.h"
#include "util/xmlpool.h"

#include <string.h>

#ifdef _MSC_VER
#include <stdlib.h>
#define PATH_MAX _MAX_PATH
#endif

#define MODULE_PREFIX "pipe_"

static int (*backends[])(struct pipe_loader_device **, int) = {
#ifdef HAVE_LIBDRM
   &pipe_loader_drm_probe,
#endif
   &pipe_loader_sw_probe
};

const char gallium_driinfo_xml[] =
   DRI_CONF_BEGIN
#include "driinfo_gallium.h"
   DRI_CONF_END
;

int
pipe_loader_probe(struct pipe_loader_device **devs, int ndev)
{
   int i, n = 0;

   for (i = 0; i < ARRAY_SIZE(backends); i++)
      n += backends[i](&devs[n], MAX2(0, ndev - n));

   return n;
}

void
pipe_loader_release(struct pipe_loader_device **devs, int ndev)
{
   int i;

   for (i = 0; i < ndev; i++)
      devs[i]->ops->release(&devs[i]);
}

void
pipe_loader_base_release(struct pipe_loader_device **dev)
{
   driDestroyOptionCache(&(*dev)->option_cache);
   driDestroyOptionInfo(&(*dev)->option_info);

   FREE(*dev);
   *dev = NULL;
}

const struct drm_conf_ret *
pipe_loader_configuration(struct pipe_loader_device *dev,
                          enum drm_conf conf)
{
   return dev->ops->configuration(dev, conf);
}

void
pipe_loader_load_options(struct pipe_loader_device *dev)
{
   if (dev->option_info.info)
      return;

   const char *xml_options = gallium_driinfo_xml;
   const struct drm_conf_ret *xml_options_conf =
      pipe_loader_configuration(dev, DRM_CONF_XML_OPTIONS);

   if (xml_options_conf)
      xml_options = xml_options_conf->val.val_pointer;

   driParseOptionInfo(&dev->option_info, xml_options);
   driParseConfigFiles(&dev->option_cache, &dev->option_info, 0,
                       dev->driver_name, NULL);
}

char *
pipe_loader_get_driinfo_xml(const char *driver_name)
{
#ifdef HAVE_LIBDRM
   char *xml = pipe_loader_drm_get_driinfo_xml(driver_name);
#else
   char *xml = NULL;
#endif

   if (!xml)
      xml = strdup(gallium_driinfo_xml);

   return xml;
}

struct pipe_screen *
pipe_loader_create_screen(struct pipe_loader_device *dev)
{
   struct pipe_screen_config config;

   pipe_loader_load_options(dev);
   config.options = &dev->option_cache;

   return dev->ops->create_screen(dev, &config);
}

struct util_dl_library *
pipe_loader_find_module(const char *driver_name,
                        const char *library_paths)
{
   struct util_dl_library *lib;
   const char *next;
   char path[PATH_MAX];
   int len, ret;

   for (next = library_paths; *next; library_paths = next + 1) {
      next = util_strchrnul(library_paths, ':');
      len = next - library_paths;

      if (len)
         ret = util_snprintf(path, sizeof(path), "%.*s/%s%s%s",
                             len, library_paths,
                             MODULE_PREFIX, driver_name, UTIL_DL_EXT);
      else
         ret = util_snprintf(path, sizeof(path), "%s%s%s",
                             MODULE_PREFIX, driver_name, UTIL_DL_EXT);

      if (ret > 0 && ret < sizeof(path)) {
         lib = util_dl_open(path);
         if (lib) {
            return lib;
         }
      }
   }

   return NULL;
}
