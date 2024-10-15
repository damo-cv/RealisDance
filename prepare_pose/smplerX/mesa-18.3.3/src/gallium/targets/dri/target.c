#include "target-helpers/drm_helper.h"
#include "target-helpers/sw_helper.h"

#include "dri_screen.h"

#define DEFINE_LOADER_DRM_ENTRYPOINT(drivername)                          \
const __DRIextension **__driDriverGetExtensions_##drivername(void);       \
PUBLIC const __DRIextension **__driDriverGetExtensions_##drivername(void) \
{                                                                         \
   globalDriverAPI = &galliumdrm_driver_api;                              \
   return galliumdrm_driver_extensions;                                   \
}

#if defined(GALLIUM_SOFTPIPE)

const __DRIextension **__driDriverGetExtensions_swrast(void);

PUBLIC const __DRIextension **__driDriverGetExtensions_swrast(void)
{
   globalDriverAPI = &galliumsw_driver_api;
   return galliumsw_driver_extensions;
}

#if defined(HAVE_LIBDRM)

const __DRIextension **__driDriverGetExtensions_kms_swrast(void);

PUBLIC const __DRIextension **__driDriverGetExtensions_kms_swrast(void)
{
   globalDriverAPI = &dri_kms_driver_api;
   return galliumdrm_driver_extensions;
}

#endif
#endif

#if defined(GALLIUM_I915)
DEFINE_LOADER_DRM_ENTRYPOINT(i915)
#endif

#if defined(GALLIUM_ILO)
DEFINE_LOADER_DRM_ENTRYPOINT(i965)
#endif

#if defined(GALLIUM_NOUVEAU)
DEFINE_LOADER_DRM_ENTRYPOINT(nouveau)
#endif

#if defined(GALLIUM_R300)
DEFINE_LOADER_DRM_ENTRYPOINT(r300)
#endif

#if defined(GALLIUM_R600)
DEFINE_LOADER_DRM_ENTRYPOINT(r600)
#endif

#if defined(GALLIUM_RADEONSI)
DEFINE_LOADER_DRM_ENTRYPOINT(radeonsi)
#endif

#if defined(GALLIUM_VMWGFX)
DEFINE_LOADER_DRM_ENTRYPOINT(vmwgfx)
#endif

#if defined(GALLIUM_FREEDRENO)
DEFINE_LOADER_DRM_ENTRYPOINT(msm)
DEFINE_LOADER_DRM_ENTRYPOINT(kgsl)
#endif

#if defined(GALLIUM_VIRGL)
DEFINE_LOADER_DRM_ENTRYPOINT(virtio_gpu)
#endif

#if defined(GALLIUM_V3D)
DEFINE_LOADER_DRM_ENTRYPOINT(v3d)
#endif

#if defined(GALLIUM_VC4)
DEFINE_LOADER_DRM_ENTRYPOINT(vc4)
#if defined(GALLIUM_PL111)
DEFINE_LOADER_DRM_ENTRYPOINT(pl111)
#endif
#endif

#if defined(GALLIUM_ETNAVIV)
DEFINE_LOADER_DRM_ENTRYPOINT(imx_drm)
DEFINE_LOADER_DRM_ENTRYPOINT(etnaviv)
#endif

#if defined(GALLIUM_TEGRA)
DEFINE_LOADER_DRM_ENTRYPOINT(tegra);
#endif
