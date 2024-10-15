/*
 * Copyright 2016 Intel Corporation
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without limitation
 *  the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *  and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice (including the next
 *  paragraph) shall be included in all copies or substantial portions of the
 *  Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 *  IN THE SOFTWARE.
 */

#include <stdint.h>

#define __gen_address_type uint64_t
#define __gen_user_data void

static uint64_t
__gen_combine_address(__attribute__((unused)) void *data,
                      __attribute__((unused)) void *loc, uint64_t addr,
                      uint32_t delta)
{
   return addr + delta;
}

#include "genxml/gen_macros.h"
#include "genxml/genX_pack.h"

#include "isl_priv.h"

static const uint32_t isl_to_gen_ds_surftype[] = {
#if GEN_GEN >= 9
   /* From the SKL PRM, "3DSTATE_DEPTH_STENCIL::SurfaceType":
    *
    *    "If depth/stencil is enabled with 1D render target, depth/stencil
    *    surface type needs to be set to 2D surface type and height set to 1.
    *    Depth will use (legacy) TileY and stencil will use TileW. For this
    *    case only, the Surface Type of the depth buffer can be 2D while the
    *    Surface Type of the render target(s) are 1D, representing an
    *    exception to a programming note above.
    */
   [ISL_SURF_DIM_1D] = SURFTYPE_2D,
#else
   [ISL_SURF_DIM_1D] = SURFTYPE_1D,
#endif
   [ISL_SURF_DIM_2D] = SURFTYPE_2D,
   [ISL_SURF_DIM_3D] = SURFTYPE_3D,
};

void
isl_genX(emit_depth_stencil_hiz_s)(const struct isl_device *dev, void *batch,
                                   const struct isl_depth_stencil_hiz_emit_info *restrict info)
{
   struct GENX(3DSTATE_DEPTH_BUFFER) db = {
      GENX(3DSTATE_DEPTH_BUFFER_header),
   };

   if (info->depth_surf) {
      db.SurfaceType = isl_to_gen_ds_surftype[info->depth_surf->dim];
      db.SurfaceFormat = isl_surf_get_depth_format(dev, info->depth_surf);
      db.Width = info->depth_surf->logical_level0_px.width - 1;
      db.Height = info->depth_surf->logical_level0_px.height - 1;
   } else if (info->stencil_surf) {
      db.SurfaceType = isl_to_gen_ds_surftype[info->stencil_surf->dim];
      db.SurfaceFormat = D32_FLOAT;
      db.Width = info->stencil_surf->logical_level0_px.width - 1;
      db.Height = info->stencil_surf->logical_level0_px.height - 1;
   } else {
      db.SurfaceType = SURFTYPE_NULL;
      db.SurfaceFormat = D32_FLOAT;
   }

   if (info->depth_surf || info->stencil_surf) {
      /* These are based entirely on the view */
      db.Depth = db.RenderTargetViewExtent = info->view->array_len - 1;
      db.LOD                  = info->view->base_level;
      db.MinimumArrayElement  = info->view->base_array_layer;
   }

   if (info->depth_surf) {
#if GEN_GEN >= 7
      db.DepthWriteEnable = true;
#endif
      db.SurfaceBaseAddress = info->depth_address;
#if GEN_GEN >= 6
      db.DepthBufferMOCS = info->mocs;
#endif

#if GEN_GEN <= 6
      db.TiledSurface = info->depth_surf->tiling != ISL_TILING_LINEAR;
      db.TileWalk = info->depth_surf->tiling == ISL_TILING_Y0 ? TILEWALK_YMAJOR :
                                                                TILEWALK_XMAJOR;
      db.MIPMapLayoutMode = MIPLAYOUT_BELOW;
#endif

      db.SurfacePitch = info->depth_surf->row_pitch_B - 1;
#if GEN_GEN >= 8
      db.SurfaceQPitch =
         isl_surf_get_array_pitch_el_rows(info->depth_surf) >> 2;
#endif
   }

#if GEN_GEN == 5 || GEN_GEN == 6
   const bool separate_stencil =
      info->stencil_surf && info->stencil_surf->format == ISL_FORMAT_R8_UINT;
   if (separate_stencil || info->hiz_usage == ISL_AUX_USAGE_HIZ) {
      assert(ISL_DEV_USE_SEPARATE_STENCIL(dev));
      db.SeparateStencilBufferEnable = true;
      db.HierarchicalDepthBufferEnable = true;
   }
#endif

#if GEN_GEN >= 6
   struct GENX(3DSTATE_STENCIL_BUFFER) sb = {
      GENX(3DSTATE_STENCIL_BUFFER_header),
   };
#else
#  define sb db
#endif

   if (info->stencil_surf) {
#if GEN_GEN >= 7
      db.StencilWriteEnable = true;
#endif
#if GEN_GEN >= 8 || GEN_IS_HASWELL
      sb.StencilBufferEnable = true;
#endif
      sb.SurfaceBaseAddress = info->stencil_address;
#if GEN_GEN >= 6
      sb.StencilBufferMOCS = info->mocs;
#endif
      sb.SurfacePitch = info->stencil_surf->row_pitch_B - 1;
#if GEN_GEN >= 8
      sb.SurfaceQPitch =
         isl_surf_get_array_pitch_el_rows(info->stencil_surf) >> 2;
#endif
   }

#if GEN_GEN >= 6
   struct GENX(3DSTATE_HIER_DEPTH_BUFFER) hiz = {
      GENX(3DSTATE_HIER_DEPTH_BUFFER_header),
   };
   struct GENX(3DSTATE_CLEAR_PARAMS) clear = {
      GENX(3DSTATE_CLEAR_PARAMS_header),
   };

   assert(info->hiz_usage == ISL_AUX_USAGE_NONE ||
          info->hiz_usage == ISL_AUX_USAGE_HIZ);
   if (info->hiz_usage == ISL_AUX_USAGE_HIZ) {
      db.HierarchicalDepthBufferEnable = true;

      hiz.SurfaceBaseAddress = info->hiz_address;
      hiz.HierarchicalDepthBufferMOCS = info->mocs;
      hiz.SurfacePitch = info->hiz_surf->row_pitch_B - 1;
#if GEN_GEN >= 8
      /* From the SKL PRM Vol2a:
       *
       *    The interpretation of this field is dependent on Surface Type
       *    as follows:
       *    - SURFTYPE_1D: distance in pixels between array slices
       *    - SURFTYPE_2D/CUBE: distance in rows between array slices
       *    - SURFTYPE_3D: distance in rows between R - slices
       *
       * Unfortunately, the docs aren't 100% accurate here.  They fail to
       * mention that the 1-D rule only applies to linear 1-D images.
       * Since depth and HiZ buffers are always tiled, they are treated as
       * 2-D images.  Prior to Sky Lake, this field is always in rows.
       */
      hiz.SurfaceQPitch =
         isl_surf_get_array_pitch_sa_rows(info->hiz_surf) >> 2;
#endif

      clear.DepthClearValueValid = true;
#if GEN_GEN >= 8
      clear.DepthClearValue = info->depth_clear_value;
#else
      switch (info->depth_surf->format) {
      case ISL_FORMAT_R32_FLOAT: {
         union { float f; uint32_t u; } fu;
         fu.f = info->depth_clear_value;
         clear.DepthClearValue = fu.u;
         break;
      }
      case ISL_FORMAT_R24_UNORM_X8_TYPELESS:
         clear.DepthClearValue = info->depth_clear_value * ((1u << 24) - 1);
         break;
      case ISL_FORMAT_R16_UNORM:
         clear.DepthClearValue = info->depth_clear_value * ((1u << 16) - 1);
         break;
      default:
         unreachable("Invalid depth type");
      }
#endif
   }
#endif /* GEN_GEN >= 6 */

   /* Pack everything into the batch */
   uint32_t *dw = batch;
   GENX(3DSTATE_DEPTH_BUFFER_pack)(NULL, dw, &db);
   dw += GENX(3DSTATE_DEPTH_BUFFER_length);

#if GEN_GEN >= 6
   GENX(3DSTATE_STENCIL_BUFFER_pack)(NULL, dw, &sb);
   dw += GENX(3DSTATE_STENCIL_BUFFER_length);

   GENX(3DSTATE_HIER_DEPTH_BUFFER_pack)(NULL, dw, &hiz);
   dw += GENX(3DSTATE_HIER_DEPTH_BUFFER_length);

   GENX(3DSTATE_CLEAR_PARAMS_pack)(NULL, dw, &clear);
   dw += GENX(3DSTATE_CLEAR_PARAMS_length);
#endif
}
