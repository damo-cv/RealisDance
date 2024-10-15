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

#if GEN_GEN >= 8
static const uint8_t isl_to_gen_halign[] = {
    [4] = HALIGN4,
    [8] = HALIGN8,
    [16] = HALIGN16,
};
#elif GEN_GEN >= 7
static const uint8_t isl_to_gen_halign[] = {
    [4] = HALIGN_4,
    [8] = HALIGN_8,
};
#endif

#if GEN_GEN >= 8
static const uint8_t isl_to_gen_valign[] = {
    [4] = VALIGN4,
    [8] = VALIGN8,
    [16] = VALIGN16,
};
#elif GEN_GEN >= 6
static const uint8_t isl_to_gen_valign[] = {
    [2] = VALIGN_2,
    [4] = VALIGN_4,
};
#endif

#if GEN_GEN >= 8
static const uint8_t isl_to_gen_tiling[] = {
   [ISL_TILING_LINEAR]  = LINEAR,
   [ISL_TILING_X]       = XMAJOR,
   [ISL_TILING_Y0]      = YMAJOR,
   [ISL_TILING_Yf]      = YMAJOR,
   [ISL_TILING_Ys]      = YMAJOR,
   [ISL_TILING_W]       = WMAJOR,
};
#endif

#if GEN_GEN >= 7
static const uint32_t isl_to_gen_multisample_layout[] = {
   [ISL_MSAA_LAYOUT_NONE]           = MSFMT_MSS,
   [ISL_MSAA_LAYOUT_INTERLEAVED]    = MSFMT_DEPTH_STENCIL,
   [ISL_MSAA_LAYOUT_ARRAY]          = MSFMT_MSS,
};
#endif

#if GEN_GEN >= 9
static const uint32_t isl_to_gen_aux_mode[] = {
   [ISL_AUX_USAGE_NONE] = AUX_NONE,
   [ISL_AUX_USAGE_HIZ] = AUX_HIZ,
   [ISL_AUX_USAGE_MCS] = AUX_CCS_D,
   [ISL_AUX_USAGE_CCS_D] = AUX_CCS_D,
   [ISL_AUX_USAGE_CCS_E] = AUX_CCS_E,
};
#elif GEN_GEN >= 8
static const uint32_t isl_to_gen_aux_mode[] = {
   [ISL_AUX_USAGE_NONE] = AUX_NONE,
   [ISL_AUX_USAGE_HIZ] = AUX_HIZ,
   [ISL_AUX_USAGE_MCS] = AUX_MCS,
   [ISL_AUX_USAGE_CCS_D] = AUX_MCS,
};
#endif

static uint8_t
get_surftype(enum isl_surf_dim dim, isl_surf_usage_flags_t usage)
{
   switch (dim) {
   default:
      unreachable("bad isl_surf_dim");
   case ISL_SURF_DIM_1D:
      assert(!(usage & ISL_SURF_USAGE_CUBE_BIT));
      return SURFTYPE_1D;
   case ISL_SURF_DIM_2D:
      if ((usage & ISL_SURF_USAGE_CUBE_BIT) &&
          (usage & ISL_SURF_USAGE_TEXTURE_BIT)) {
         /* We need SURFTYPE_CUBE to make cube sampling work */
         return SURFTYPE_CUBE;
      } else {
         /* Everything else (render and storage) treat cubes as plain
          * 2D array textures
          */
         return SURFTYPE_2D;
      }
   case ISL_SURF_DIM_3D:
      assert(!(usage & ISL_SURF_USAGE_CUBE_BIT));
      return SURFTYPE_3D;
   }
}

/**
 * Get the horizontal and vertical alignment in the units expected by the
 * hardware.  Note that this does NOT give you the actual hardware enum values
 * but an index into the isl_to_gen_[hv]align arrays above.
 */
UNUSED static struct isl_extent3d
get_image_alignment(const struct isl_surf *surf)
{
   if (GEN_GEN >= 9) {
      if (isl_tiling_is_std_y(surf->tiling) ||
          surf->dim_layout == ISL_DIM_LAYOUT_GEN9_1D) {
         /* The hardware ignores the alignment values. Anyway, the surface's
          * true alignment is likely outside the enum range of HALIGN* and
          * VALIGN*.
          */
         return isl_extent3d(4, 4, 1);
      } else {
         /* In Skylake, RENDER_SUFFACE_STATE.SurfaceVerticalAlignment is in units
          * of surface elements (not pixels nor samples). For compressed formats,
          * a "surface element" is defined as a compression block.  For example,
          * if SurfaceVerticalAlignment is VALIGN_4 and SurfaceFormat is an ETC2
          * format (ETC2 has a block height of 4), then the vertical alignment is
          * 4 compression blocks or, equivalently, 16 pixels.
          */
         return isl_surf_get_image_alignment_el(surf);
      }
   } else {
      /* Pre-Skylake, RENDER_SUFFACE_STATE.SurfaceVerticalAlignment is in
       * units of surface samples.  For example, if SurfaceVerticalAlignment
       * is VALIGN_4 and the surface is singlesampled, then for any surface
       * format (compressed or not) the vertical alignment is
       * 4 pixels.
       */
      return isl_surf_get_image_alignment_sa(surf);
   }
}

#if GEN_GEN >= 8
static uint32_t
get_qpitch(const struct isl_surf *surf)
{
   switch (surf->dim_layout) {
   default:
      unreachable("Bad isl_surf_dim");
   case ISL_DIM_LAYOUT_GEN4_2D:
      if (GEN_GEN >= 9) {
         if (surf->dim == ISL_SURF_DIM_3D && surf->tiling == ISL_TILING_W) {
            /* This is rather annoying and completely undocumented.  It
             * appears that the hardware has a bug (or undocumented feature)
             * regarding stencil buffers most likely related to the way
             * W-tiling is handled as modified Y-tiling.  If you bind a 3-D
             * stencil buffer normally, and use texelFetch on it, the z or
             * array index will get implicitly multiplied by 2 for no obvious
             * reason.  The fix appears to be to divide qpitch by 2 for
             * W-tiled surfaces.
             */
            return isl_surf_get_array_pitch_el_rows(surf) / 2;
         } else {
            return isl_surf_get_array_pitch_el_rows(surf);
         }
      } else {
         /* From the Broadwell PRM for RENDER_SURFACE_STATE.QPitch
          *
          *    "This field must be set to an integer multiple of the Surface
          *    Vertical Alignment. For compressed textures (BC*, FXT1,
          *    ETC*, and EAC* Surface Formats), this field is in units of
          *    rows in the uncompressed surface, and must be set to an
          *    integer multiple of the vertical alignment parameter "j"
          *    defined in the Common Surface Formats section."
          */
         return isl_surf_get_array_pitch_sa_rows(surf);
      }
   case ISL_DIM_LAYOUT_GEN9_1D:
      /* QPitch is usually expressed as rows of surface elements (where
       * a surface element is an compression block or a single surface
       * sample). Skylake 1D is an outlier.
       *
       * From the Skylake BSpec >> Memory Views >> Common Surface
       * Formats >> Surface Layout and Tiling >> 1D Surfaces:
       *
       *    Surface QPitch specifies the distance in pixels between array
       *    slices.
       */
      return isl_surf_get_array_pitch_el(surf);
   case ISL_DIM_LAYOUT_GEN4_3D:
      /* QPitch doesn't make sense for ISL_DIM_LAYOUT_GEN4_3D since it uses a
       * different pitch at each LOD.  Also, the QPitch field is ignored for
       * these surfaces.  From the Broadwell PRM documentation for QPitch:
       *
       *    This field specifies the distance in rows between array slices. It
       *    is used only in the following cases:
       *     - Surface Array is enabled OR
       *     - Number of Mulitsamples is not NUMSAMPLES_1 and Multisampled
       *       Surface Storage Format set to MSFMT_MSS OR
       *     - Surface Type is SURFTYPE_CUBE
       *
       * None of the three conditions above can possibly apply to a 3D surface
       * so it is safe to just set QPitch to 0.
       */
      return 0;
   }
}
#endif /* GEN_GEN >= 8 */

void
isl_genX(surf_fill_state_s)(const struct isl_device *dev, void *state,
                            const struct isl_surf_fill_state_info *restrict info)
{
   struct GENX(RENDER_SURFACE_STATE) s = { 0 };

   s.SurfaceType = get_surftype(info->surf->dim, info->view->usage);

   if (info->view->usage & ISL_SURF_USAGE_RENDER_TARGET_BIT)
      assert(isl_format_supports_rendering(dev->info, info->view->format));
   else if (info->view->usage & ISL_SURF_USAGE_TEXTURE_BIT)
      assert(isl_format_supports_sampling(dev->info, info->view->format));

   /* From the Sky Lake PRM Vol. 2d, RENDER_SURFACE_STATE::SurfaceFormat
    *
    *    This field cannot be a compressed (BC*, DXT*, FXT*, ETC*, EAC*)
    *    format if the Surface Type is SURFTYPE_1D
    */
   if (info->surf->dim == ISL_SURF_DIM_1D)
      assert(!isl_format_is_compressed(info->view->format));

   if (isl_format_is_compressed(info->surf->format)) {
      /* You're not allowed to make a view of a compressed format with any
       * format other than the surface format.  None of the userspace APIs
       * allow for this directly and doing so would mess up a number of
       * surface parameters such as Width, Height, and alignments.  Ideally,
       * we'd like to assert that the two formats match.  However, we have an
       * S3TC workaround that requires us to do reinterpretation.  So assert
       * that they're at least the same bpb and block size.
       */
      MAYBE_UNUSED const struct isl_format_layout *surf_fmtl =
         isl_format_get_layout(info->surf->format);
      MAYBE_UNUSED const struct isl_format_layout *view_fmtl =
         isl_format_get_layout(info->surf->format);
      assert(surf_fmtl->bpb == view_fmtl->bpb);
      assert(surf_fmtl->bw == view_fmtl->bw);
      assert(surf_fmtl->bh == view_fmtl->bh);
   }

   s.SurfaceFormat = info->view->format;

#if GEN_GEN <= 5
   s.ColorBufferComponentWriteDisables = info->write_disables;
#else
   assert(info->write_disables == 0);
#endif

#if GEN_IS_HASWELL
   s.IntegerSurfaceFormat =
      isl_format_has_int_channel((enum isl_format) s.SurfaceFormat);
#endif

   assert(info->surf->logical_level0_px.width > 0 &&
          info->surf->logical_level0_px.height > 0);

   s.Width = info->surf->logical_level0_px.width - 1;
   s.Height = info->surf->logical_level0_px.height - 1;

   /* In the gen6 PRM Volume 1 Part 1: Graphics Core, Section 7.18.3.7.1
    * (Surface Arrays For all surfaces other than separate stencil buffer):
    *
    * "[DevSNB] Errata: Sampler MSAA Qpitch will be 4 greater than the value
    *  calculated in the equation above , for every other odd Surface Height
    *  starting from 1 i.e. 1,5,9,13"
    *
    * Since this Qpitch errata only impacts the sampler, we have to adjust the
    * input for the rendering surface to achieve the same qpitch. For the
    * affected heights, we increment the height by 1 for the rendering
    * surface.
    */
   if (GEN_GEN == 6 && (info->view->usage & ISL_SURF_USAGE_RENDER_TARGET_BIT) &&
       info->surf->samples > 1 &&
       (info->surf->logical_level0_px.height % 4) == 1)
      s.Height++;

   switch (s.SurfaceType) {
   case SURFTYPE_1D:
   case SURFTYPE_2D:
      /* From the Ivy Bridge PRM >> RENDER_SURFACE_STATE::MinimumArrayElement:
       *
       *    "If Number of Multisamples is not MULTISAMPLECOUNT_1, this field
       *    must be set to zero if this surface is used with sampling engine
       *    messages."
       *
       * This restriction appears to exist only on Ivy Bridge.
       */
      if (GEN_GEN == 7 && !GEN_IS_HASWELL && !ISL_DEV_IS_BAYTRAIL(dev) &&
          (info->view->usage & ISL_SURF_USAGE_TEXTURE_BIT) &&
          info->surf->samples > 1)
         assert(info->view->base_array_layer == 0);

      s.MinimumArrayElement = info->view->base_array_layer;

      /* From the Broadwell PRM >> RENDER_SURFACE_STATE::Depth:
       *
       *    For SURFTYPE_1D, 2D, and CUBE: The range of this field is reduced
       *    by one for each increase from zero of Minimum Array Element. For
       *    example, if Minimum Array Element is set to 1024 on a 2D surface,
       *    the range of this field is reduced to [0,1023].
       *
       * In other words, 'Depth' is the number of array layers.
       */
      s.Depth = info->view->array_len - 1;

      /* From the Broadwell PRM >> RENDER_SURFACE_STATE::RenderTargetViewExtent:
       *
       *    For Render Target and Typed Dataport 1D and 2D Surfaces:
       *    This field must be set to the same value as the Depth field.
       */
      if (info->view->usage & (ISL_SURF_USAGE_RENDER_TARGET_BIT |
                               ISL_SURF_USAGE_STORAGE_BIT))
         s.RenderTargetViewExtent = s.Depth;
      break;
   case SURFTYPE_CUBE:
      s.MinimumArrayElement = info->view->base_array_layer;
      /* Same as SURFTYPE_2D, but divided by 6 */
      s.Depth = info->view->array_len / 6 - 1;
      if (info->view->usage & (ISL_SURF_USAGE_RENDER_TARGET_BIT |
                               ISL_SURF_USAGE_STORAGE_BIT))
         s.RenderTargetViewExtent = s.Depth;
      break;
   case SURFTYPE_3D:
      /* From the Broadwell PRM >> RENDER_SURFACE_STATE::Depth:
       *
       *    If the volume texture is MIP-mapped, this field specifies the
       *    depth of the base MIP level.
       */
      s.Depth = info->surf->logical_level0_px.depth - 1;

      /* From the Broadwell PRM >> RENDER_SURFACE_STATE::RenderTargetViewExtent:
       *
       *    For Render Target and Typed Dataport 3D Surfaces: This field
       *    indicates the extent of the accessible 'R' coordinates minus 1 on
       *    the LOD currently being rendered to.
       *
       * The docs specify that this only matters for render targets and
       * surfaces used with typed dataport messages.  Prior to Ivy Bridge, the
       * Depth field has more bits than RenderTargetViewExtent so we can have
       * textures with more levels than we can render to.  In order to prevent
       * assert-failures in the packing function below, we only set the field
       * when it's actually going to be used by the hardware.
       *
       * Similaraly, the MinimumArrayElement field is ignored by all hardware
       * prior to Sky Lake when texturing and we want it set to 0 anyway.
       * Since it's already initialized to 0, we can just leave it alone for
       * texture surfaces.
       */
      if (info->view->usage & (ISL_SURF_USAGE_RENDER_TARGET_BIT |
                               ISL_SURF_USAGE_STORAGE_BIT)) {
         s.MinimumArrayElement = info->view->base_array_layer;
         s.RenderTargetViewExtent = info->view->array_len - 1;
      }
      break;
   default:
      unreachable("bad SurfaceType");
   }

#if GEN_GEN >= 7
   s.SurfaceArray = info->surf->dim != ISL_SURF_DIM_3D;
#endif

   if (info->view->usage & ISL_SURF_USAGE_RENDER_TARGET_BIT) {
      /* For render target surfaces, the hardware interprets field
       * MIPCount/LOD as LOD. The Broadwell PRM says:
       *
       *    MIPCountLOD defines the LOD that will be rendered into.
       *    SurfaceMinLOD is ignored.
       */
      s.MIPCountLOD = info->view->base_level;
      s.SurfaceMinLOD = 0;
   } else {
      /* For non render target surfaces, the hardware interprets field
       * MIPCount/LOD as MIPCount.  The range of levels accessible by the
       * sampler engine is [SurfaceMinLOD, SurfaceMinLOD + MIPCountLOD].
       */
      s.SurfaceMinLOD = info->view->base_level;
      s.MIPCountLOD = MAX(info->view->levels, 1) - 1;
   }

#if GEN_GEN >= 9
   /* We don't use miptails yet.  The PRM recommends that you set "Mip Tail
    * Start LOD" to 15 to prevent the hardware from trying to use them.
    */
   s.TiledResourceMode = NONE;
   s.MipTailStartLOD = 15;
#endif

#if GEN_GEN >= 6
   const struct isl_extent3d image_align = get_image_alignment(info->surf);
   s.SurfaceVerticalAlignment = isl_to_gen_valign[image_align.height];
#if GEN_GEN >= 7
   s.SurfaceHorizontalAlignment = isl_to_gen_halign[image_align.width];
#endif
#endif

   if (info->surf->dim_layout == ISL_DIM_LAYOUT_GEN9_1D) {
      /* For gen9 1-D textures, surface pitch is ignored */
      s.SurfacePitch = 0;
   } else {
      s.SurfacePitch = info->surf->row_pitch_B - 1;
   }

#if GEN_GEN >= 8
   s.SurfaceQPitch = get_qpitch(info->surf) >> 2;
#elif GEN_GEN == 7
   s.SurfaceArraySpacing = info->surf->array_pitch_span ==
                           ISL_ARRAY_PITCH_SPAN_COMPACT;
#endif

#if GEN_GEN >= 8
   s.TileMode = isl_to_gen_tiling[info->surf->tiling];
#else
   s.TiledSurface = info->surf->tiling != ISL_TILING_LINEAR,
   s.TileWalk = info->surf->tiling == ISL_TILING_Y0 ? TILEWALK_YMAJOR :
                                                      TILEWALK_XMAJOR,
#endif

#if GEN_GEN >= 8
   s.RenderCacheReadWriteMode = WriteOnlyCache;
#else
   s.RenderCacheReadWriteMode = 0;
#endif

   s.CubeFaceEnablePositiveZ = 1;
   s.CubeFaceEnableNegativeZ = 1;
   s.CubeFaceEnablePositiveY = 1;
   s.CubeFaceEnableNegativeY = 1;
   s.CubeFaceEnablePositiveX = 1;
   s.CubeFaceEnableNegativeX = 1;

#if GEN_GEN >= 6
   s.NumberofMultisamples = ffs(info->surf->samples) - 1;
#if GEN_GEN >= 7
   s.MultisampledSurfaceStorageFormat =
      isl_to_gen_multisample_layout[info->surf->msaa_layout];
#endif
#endif

#if (GEN_GEN >= 8 || GEN_IS_HASWELL)
   if (info->view->usage & ISL_SURF_USAGE_RENDER_TARGET_BIT)
      assert(isl_swizzle_supports_rendering(dev->info, info->view->swizzle));

   s.ShaderChannelSelectRed = (enum GENX(ShaderChannelSelect)) info->view->swizzle.r;
   s.ShaderChannelSelectGreen = (enum GENX(ShaderChannelSelect)) info->view->swizzle.g;
   s.ShaderChannelSelectBlue = (enum GENX(ShaderChannelSelect)) info->view->swizzle.b;
   s.ShaderChannelSelectAlpha = (enum GENX(ShaderChannelSelect)) info->view->swizzle.a;
#else
   assert(isl_swizzle_is_identity(info->view->swizzle));
#endif

   s.SurfaceBaseAddress = info->address;

#if GEN_GEN >= 6
   s.MOCS = info->mocs;
#endif

#if GEN_GEN > 4 || GEN_IS_G4X
   if (info->x_offset_sa != 0 || info->y_offset_sa != 0) {
      /* There are fairly strict rules about when the offsets can be used.
       * These are mostly taken from the Sky Lake PRM documentation for
       * RENDER_SURFACE_STATE.
       */
      assert(info->surf->tiling != ISL_TILING_LINEAR);
      assert(info->surf->dim == ISL_SURF_DIM_2D);
      assert(isl_is_pow2(isl_format_get_layout(info->view->format)->bpb));
      assert(info->surf->levels == 1);
      assert(info->surf->logical_level0_px.array_len == 1);
      assert(info->aux_usage == ISL_AUX_USAGE_NONE);

      if (GEN_GEN >= 8) {
         /* Broadwell added more rules. */
         assert(info->surf->samples == 1);
         if (isl_format_get_layout(info->view->format)->bpb == 8)
            assert(info->x_offset_sa % 16 == 0);
         if (isl_format_get_layout(info->view->format)->bpb == 16)
            assert(info->x_offset_sa % 8 == 0);
      }

#if GEN_GEN >= 7
      s.SurfaceArray = false;
#endif
   }

   const unsigned x_div = 4;
   const unsigned y_div = GEN_GEN >= 8 ? 4 : 2;
   assert(info->x_offset_sa % x_div == 0);
   assert(info->y_offset_sa % y_div == 0);
   s.XOffset = info->x_offset_sa / x_div;
   s.YOffset = info->y_offset_sa / y_div;
#else
   assert(info->x_offset_sa == 0);
   assert(info->y_offset_sa == 0);
#endif

#if GEN_GEN >= 7
   if (info->aux_surf && info->aux_usage != ISL_AUX_USAGE_NONE) {
      /* The docs don't appear to say anything whatsoever about compression
       * and the data port.  Testing seems to indicate that the data port
       * completely ignores the AuxiliarySurfaceMode field.
       */
      assert(!(info->view->usage & ISL_SURF_USAGE_STORAGE_BIT));

      struct isl_tile_info tile_info;
      isl_surf_get_tile_info(info->aux_surf, &tile_info);
      uint32_t pitch_in_tiles =
         info->aux_surf->row_pitch_B / tile_info.phys_extent_B.width;

      s.AuxiliarySurfaceBaseAddress = info->aux_address;
      s.AuxiliarySurfacePitch = pitch_in_tiles - 1;

#if GEN_GEN >= 8
      assert(GEN_GEN >= 9 || info->aux_usage != ISL_AUX_USAGE_CCS_E);
      /* Auxiliary surfaces in ISL have compressed formats but the hardware
       * doesn't expect our definition of the compression, it expects qpitch
       * in units of samples on the main surface.
       */
      s.AuxiliarySurfaceQPitch =
         isl_surf_get_array_pitch_sa_rows(info->aux_surf) >> 2;

      if (info->aux_usage == ISL_AUX_USAGE_HIZ) {
         /* The number of samples must be 1 */
         assert(info->surf->samples == 1);

         /* The dimension must not be 3D */
         assert(info->surf->dim != ISL_SURF_DIM_3D);

         /* The format must be one of the following: */
         switch (info->view->format) {
         case ISL_FORMAT_R32_FLOAT:
         case ISL_FORMAT_R24_UNORM_X8_TYPELESS:
         case ISL_FORMAT_R16_UNORM:
            break;
         default:
            assert(!"Incompatible HiZ Sampling format");
            break;
         }
      }

      s.AuxiliarySurfaceMode = isl_to_gen_aux_mode[info->aux_usage];
#else
      assert(info->aux_usage == ISL_AUX_USAGE_MCS ||
             info->aux_usage == ISL_AUX_USAGE_CCS_D);
      s.MCSEnable = true;
#endif
   }
#endif

#if GEN_GEN >= 8
   /* From the CHV PRM, Volume 2d, page 321 (RENDER_SURFACE_STATE dword 0
    * bit 9 "Sampler L2 Bypass Mode Disable" Programming Notes):
    *
    *    This bit must be set for the following surface types: BC2_UNORM
    *    BC3_UNORM BC5_UNORM BC5_SNORM BC7_UNORM
    */
   if (GEN_GEN >= 9 || dev->info->is_cherryview) {
      switch (info->view->format) {
      case ISL_FORMAT_BC2_UNORM:
      case ISL_FORMAT_BC3_UNORM:
      case ISL_FORMAT_BC5_UNORM:
      case ISL_FORMAT_BC5_SNORM:
      case ISL_FORMAT_BC7_UNORM:
         s.SamplerL2BypassModeDisable = true;
         break;
      default:
         /* From the SKL PRM, Programming Note under Sampler Output Channel
          * Mapping:
          *
          *    If a surface has an associated HiZ Auxilliary surface, the
          *    Sampler L2 Bypass Mode Disable field in the RENDER_SURFACE_STATE
          *    must be set.
          */
         if (GEN_GEN >= 9 && info->aux_usage == ISL_AUX_USAGE_HIZ)
            s.SamplerL2BypassModeDisable = true;
         break;
      }
   }
#endif

   if (info->aux_usage != ISL_AUX_USAGE_NONE) {
      if (info->use_clear_address) {
#if GEN_GEN >= 10
         s.ClearValueAddressEnable = true;
         s.ClearValueAddress = info->clear_address;
#else
         unreachable("Gen9 and earlier do not support indirect clear colors");
#endif
      }
#if GEN_GEN >= 9
      if (!info->use_clear_address) {
         s.RedClearColor = info->clear_color.u32[0];
         s.GreenClearColor = info->clear_color.u32[1];
         s.BlueClearColor = info->clear_color.u32[2];
         s.AlphaClearColor = info->clear_color.u32[3];
      }
#elif GEN_GEN >= 7
      /* Prior to Sky Lake, we only have one bit for the clear color which
       * gives us 0 or 1 in whatever the surface's format happens to be.
       */
      if (isl_format_has_int_channel(info->view->format)) {
         for (unsigned i = 0; i < 4; i++) {
            assert(info->clear_color.u32[i] == 0 ||
                   info->clear_color.u32[i] == 1);
         }
         s.RedClearColor = info->clear_color.u32[0] != 0;
         s.GreenClearColor = info->clear_color.u32[1] != 0;
         s.BlueClearColor = info->clear_color.u32[2] != 0;
         s.AlphaClearColor = info->clear_color.u32[3] != 0;
      } else {
         for (unsigned i = 0; i < 4; i++) {
            assert(info->clear_color.f32[i] == 0.0f ||
                   info->clear_color.f32[i] == 1.0f);
         }
         s.RedClearColor = info->clear_color.f32[0] != 0.0f;
         s.GreenClearColor = info->clear_color.f32[1] != 0.0f;
         s.BlueClearColor = info->clear_color.f32[2] != 0.0f;
         s.AlphaClearColor = info->clear_color.f32[3] != 0.0f;
      }
#endif
   }

   GENX(RENDER_SURFACE_STATE_pack)(NULL, state, &s);
}

void
isl_genX(buffer_fill_state_s)(void *state,
                              const struct isl_buffer_fill_state_info *restrict info)
{
   uint64_t buffer_size = info->size_B;

   /* Uniform and Storage buffers need to have surface size not less that the
    * aligned 32-bit size of the buffer. To calculate the array lenght on
    * unsized arrays in StorageBuffer the last 2 bits store the padding size
    * added to the surface, so we can calculate latter the original buffer
    * size to know the number of elements.
    *
    *  surface_size = isl_align(buffer_size, 4) +
    *                 (isl_align(buffer_size) - buffer_size)
    *
    *  buffer_size = (surface_size & ~3) - (surface_size & 3)
    */
   if (info->format == ISL_FORMAT_RAW  ||
       info->stride_B < isl_format_get_layout(info->format)->bpb / 8) {
      assert(info->stride_B == 1);
      uint64_t aligned_size = isl_align(buffer_size, 4);
      buffer_size = aligned_size + (aligned_size - buffer_size);
   }

   uint32_t num_elements = buffer_size / info->stride_B;

   if (GEN_GEN >= 7) {
      /* From the IVB PRM, SURFACE_STATE::Height,
       *
       *    For typed buffer and structured buffer surfaces, the number
       *    of entries in the buffer ranges from 1 to 2^27. For raw buffer
       *    surfaces, the number of entries in the buffer is the number of bytes
       *    which can range from 1 to 2^30.
       */
      if (info->format == ISL_FORMAT_RAW) {
         assert(num_elements <= (1ull << 30));
         assert(num_elements > 0);
      } else {
         assert(num_elements <= (1ull << 27));
      }
   } else {
      assert(num_elements <= (1ull << 27));
   }

   struct GENX(RENDER_SURFACE_STATE) s = { 0, };

   s.SurfaceType = SURFTYPE_BUFFER;
   s.SurfaceFormat = info->format;

#if GEN_GEN >= 6
   s.SurfaceVerticalAlignment = isl_to_gen_valign[4];
#if GEN_GEN >= 7
   s.SurfaceHorizontalAlignment = isl_to_gen_halign[4];
   s.SurfaceArray = false;
#endif
#endif

#if GEN_GEN >= 7
   s.Height = ((num_elements - 1) >> 7) & 0x3fff;
   s.Width = (num_elements - 1) & 0x7f;
   s.Depth = ((num_elements - 1) >> 21) & 0x3ff;
#else
   s.Height = ((num_elements - 1) >> 7) & 0x1fff;
   s.Width = (num_elements - 1) & 0x7f;
   s.Depth = ((num_elements - 1) >> 20) & 0x7f;
#endif

   s.SurfacePitch = info->stride_B - 1;

#if GEN_GEN >= 6
   s.NumberofMultisamples = MULTISAMPLECOUNT_1;
#endif

#if (GEN_GEN >= 8)
   s.TileMode = LINEAR;
#else
   s.TiledSurface = false;
#endif

#if (GEN_GEN >= 8)
   s.RenderCacheReadWriteMode = WriteOnlyCache;
#else
   s.RenderCacheReadWriteMode = 0;
#endif

   s.SurfaceBaseAddress = info->address;
#if GEN_GEN >= 6
   s.MOCS = info->mocs;
#endif

#if (GEN_GEN >= 8 || GEN_IS_HASWELL)
   s.ShaderChannelSelectRed = SCS_RED;
   s.ShaderChannelSelectGreen = SCS_GREEN;
   s.ShaderChannelSelectBlue = SCS_BLUE;
   s.ShaderChannelSelectAlpha = SCS_ALPHA;
#endif

   GENX(RENDER_SURFACE_STATE_pack)(NULL, state, &s);
}

void
isl_genX(null_fill_state)(void *state, struct isl_extent3d size)
{
   struct GENX(RENDER_SURFACE_STATE) s = {
      .SurfaceType = SURFTYPE_NULL,
      .SurfaceFormat = ISL_FORMAT_B8G8R8A8_UNORM,
#if GEN_GEN >= 7
      .SurfaceArray = size.depth > 0,
#endif
#if GEN_GEN >= 8
      .TileMode = YMAJOR,
#else
      .TiledSurface = true,
      .TileWalk = TILEWALK_YMAJOR,
#endif
      .Width = size.width - 1,
      .Height = size.height - 1,
      .Depth = size.depth - 1,
      .RenderTargetViewExtent = size.depth - 1,
#if GEN_GEN <= 5
      .ColorBufferComponentWriteDisables = 0xf,
#endif
   };
   GENX(RENDER_SURFACE_STATE_pack)(NULL, state, &s);
}
