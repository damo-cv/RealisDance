#include "state_tracker/drm_driver.h"
#include "target-helpers/inline_debug_helper.h"
#include "radeon/drm/radeon_drm_public.h"
#include "radeon/radeon_winsys.h"
#include "amdgpu/drm/amdgpu_public.h"
#include "radeonsi/si_public.h"
#include "util/xmlpool.h"

static struct pipe_screen *
create_screen(int fd, const struct pipe_screen_config *config)
{
   struct radeon_winsys *rw;

   /* First, try amdgpu. */
   rw = amdgpu_winsys_create(fd, config, radeonsi_screen_create);

   if (!rw)
      rw = radeon_drm_winsys_create(fd, config, radeonsi_screen_create);

   return rw ? debug_screen_wrap(rw->screen) : NULL;
}

static const struct drm_conf_ret throttle_ret = {
   .type = DRM_CONF_INT,
   .val.val_int = 2,
};

static const struct drm_conf_ret share_fd_ret = {
   .type = DRM_CONF_BOOL,
   .val.val_bool = true,
};

static const struct drm_conf_ret *drm_configuration(enum drm_conf conf)
{
   static const struct drm_conf_ret xml_options_ret = {
      .type = DRM_CONF_POINTER,
      .val.val_pointer =
#include "radeonsi/si_driinfo.h"
   };

   switch (conf) {
   case DRM_CONF_THROTTLE:
      return &throttle_ret;
   case DRM_CONF_SHARE_FD:
      return &share_fd_ret;
   case DRM_CONF_XML_OPTIONS:
      return &xml_options_ret;
   default:
      break;
   }
   return NULL;
}

PUBLIC
DRM_DRIVER_DESCRIPTOR("radeonsi", create_screen, drm_configuration)
