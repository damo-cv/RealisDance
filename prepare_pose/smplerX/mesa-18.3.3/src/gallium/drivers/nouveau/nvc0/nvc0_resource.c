#include <drm_fourcc.h>

#include "pipe/p_context.h"
#include "nvc0/nvc0_resource.h"
#include "nouveau_screen.h"


static struct pipe_resource *
nvc0_resource_create(struct pipe_screen *screen,
                     const struct pipe_resource *templ)
{
   const uint64_t modifier = DRM_FORMAT_MOD_INVALID;

   switch (templ->target) {
   case PIPE_BUFFER:
      return nouveau_buffer_create(screen, templ);
   default:
      return nvc0_miptree_create(screen, templ, &modifier, 1);
   }
}

static struct pipe_resource *
nvc0_resource_create_with_modifiers(struct pipe_screen *screen,
                                    const struct pipe_resource *templ,
                                    const uint64_t *modifiers, int count)
{
   switch (templ->target) {
   case PIPE_BUFFER:
      return nouveau_buffer_create(screen, templ);
   default:
      return nvc0_miptree_create(screen, templ, modifiers, count);
   }
}

static void
nvc0_query_dmabuf_modifiers(struct pipe_screen *screen,
                            enum pipe_format format, int max,
                            uint64_t *modifiers, unsigned int *external_only,
                            int *count)
{
   static const uint64_t supported_modifiers[] = {
      DRM_FORMAT_MOD_LINEAR,
      DRM_FORMAT_MOD_NVIDIA_16BX2_BLOCK_ONE_GOB,
      DRM_FORMAT_MOD_NVIDIA_16BX2_BLOCK_TWO_GOB,
      DRM_FORMAT_MOD_NVIDIA_16BX2_BLOCK_FOUR_GOB,
      DRM_FORMAT_MOD_NVIDIA_16BX2_BLOCK_EIGHT_GOB,
      DRM_FORMAT_MOD_NVIDIA_16BX2_BLOCK_SIXTEEN_GOB,
      DRM_FORMAT_MOD_NVIDIA_16BX2_BLOCK_THIRTYTWO_GOB,
   };
   int i, num = 0;

   if (max > ARRAY_SIZE(supported_modifiers))
      max = ARRAY_SIZE(supported_modifiers);

   if (!max) {
      max = ARRAY_SIZE(supported_modifiers);
      external_only = NULL;
      modifiers = NULL;
   }

   for (i = 0; i < max; i++) {
      if (modifiers)
         modifiers[num] = supported_modifiers[i];

      if (external_only)
         external_only[num] = 0;

      num++;
   }

   *count = num;
}

static struct pipe_resource *
nvc0_resource_from_handle(struct pipe_screen * screen,
                          const struct pipe_resource *templ,
                          struct winsys_handle *whandle,
                          unsigned usage)
{
   if (templ->target == PIPE_BUFFER) {
      return NULL;
   } else {
      struct pipe_resource *res = nv50_miptree_from_handle(screen,
                                                           templ, whandle);
      if (res)
         nv04_resource(res)->vtbl = &nvc0_miptree_vtbl;
      return res;
   }
}

static struct pipe_surface *
nvc0_surface_create(struct pipe_context *pipe,
                    struct pipe_resource *pres,
                    const struct pipe_surface *templ)
{
   if (unlikely(pres->target == PIPE_BUFFER))
      return nv50_surface_from_buffer(pipe, pres, templ);
   return nvc0_miptree_surface_new(pipe, pres, templ);
}

void
nvc0_init_resource_functions(struct pipe_context *pcontext)
{
   pcontext->transfer_map = u_transfer_map_vtbl;
   pcontext->transfer_flush_region = u_transfer_flush_region_vtbl;
   pcontext->transfer_unmap = u_transfer_unmap_vtbl;
   pcontext->buffer_subdata = u_default_buffer_subdata;
   pcontext->texture_subdata = u_default_texture_subdata;
   pcontext->create_surface = nvc0_surface_create;
   pcontext->surface_destroy = nv50_surface_destroy;
   pcontext->invalidate_resource = nv50_invalidate_resource;
}

void
nvc0_screen_init_resource_functions(struct pipe_screen *pscreen)
{
   pscreen->resource_create = nvc0_resource_create;
   pscreen->resource_create_with_modifiers = nvc0_resource_create_with_modifiers;
   pscreen->query_dmabuf_modifiers = nvc0_query_dmabuf_modifiers;
   pscreen->resource_from_handle = nvc0_resource_from_handle;
   pscreen->resource_get_handle = u_resource_get_handle_vtbl;
   pscreen->resource_destroy = u_resource_destroy_vtbl;
}
