/*
 * Copyright 2015 Intel Corporation
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

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

#include "genxml/genX_bits.h"

#include "isl.h"
#include "isl_gen4.h"
#include "isl_gen6.h"
#include "isl_gen7.h"
#include "isl_gen8.h"
#include "isl_gen9.h"
#include "isl_priv.h"

void PRINTFLIKE(3, 4) UNUSED
__isl_finishme(const char *file, int line, const char *fmt, ...)
{
   va_list ap;
   char buf[512];

   va_start(ap, fmt);
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);

   fprintf(stderr, "%s:%d: FINISHME: %s\n", file, line, buf);
}

void
isl_device_init(struct isl_device *dev,
                const struct gen_device_info *info,
                bool has_bit6_swizzling)
{
   dev->info = info;
   dev->use_separate_stencil = ISL_DEV_GEN(dev) >= 6;
   dev->has_bit6_swizzling = has_bit6_swizzling;

   /* The ISL_DEV macros may be defined in the CFLAGS, thus hardcoding some
    * device properties at buildtime. Verify that the macros with the device
    * properties chosen during runtime.
    */
   ISL_DEV_GEN_SANITIZE(dev);
   ISL_DEV_USE_SEPARATE_STENCIL_SANITIZE(dev);

   /* Did we break hiz or stencil? */
   if (ISL_DEV_USE_SEPARATE_STENCIL(dev))
      assert(info->has_hiz_and_separate_stencil);
   if (info->must_use_separate_stencil)
      assert(ISL_DEV_USE_SEPARATE_STENCIL(dev));

   dev->ss.size = RENDER_SURFACE_STATE_length(info) * 4;
   dev->ss.align = isl_align(dev->ss.size, 32);

   dev->ss.clear_color_state_size = CLEAR_COLOR_length(info) * 4;
   dev->ss.clear_color_state_offset =
      RENDER_SURFACE_STATE_ClearValueAddress_start(info) / 32 * 4;

   dev->ss.clear_value_size =
      isl_align(RENDER_SURFACE_STATE_RedClearColor_bits(info) +
                RENDER_SURFACE_STATE_GreenClearColor_bits(info) +
                RENDER_SURFACE_STATE_BlueClearColor_bits(info) +
                RENDER_SURFACE_STATE_AlphaClearColor_bits(info), 32) / 8;

   dev->ss.clear_value_offset =
      RENDER_SURFACE_STATE_RedClearColor_start(info) / 32 * 4;

   assert(RENDER_SURFACE_STATE_SurfaceBaseAddress_start(info) % 8 == 0);
   dev->ss.addr_offset =
      RENDER_SURFACE_STATE_SurfaceBaseAddress_start(info) / 8;

   /* The "Auxiliary Surface Base Address" field starts a bit higher up
    * because the bottom 12 bits are used for other things.  Round down to
    * the nearest dword before.
    */
   dev->ss.aux_addr_offset =
      (RENDER_SURFACE_STATE_AuxiliarySurfaceBaseAddress_start(info) & ~31) / 8;

   dev->ds.size = _3DSTATE_DEPTH_BUFFER_length(info) * 4;
   assert(_3DSTATE_DEPTH_BUFFER_SurfaceBaseAddress_start(info) % 8 == 0);
   dev->ds.depth_offset =
      _3DSTATE_DEPTH_BUFFER_SurfaceBaseAddress_start(info) / 8;

   if (dev->use_separate_stencil) {
      dev->ds.size += _3DSTATE_STENCIL_BUFFER_length(info) * 4 +
                      _3DSTATE_HIER_DEPTH_BUFFER_length(info) * 4 +
                      _3DSTATE_CLEAR_PARAMS_length(info) * 4;

      assert(_3DSTATE_STENCIL_BUFFER_SurfaceBaseAddress_start(info) % 8 == 0);
      dev->ds.stencil_offset =
         _3DSTATE_DEPTH_BUFFER_length(info) * 4 +
         _3DSTATE_STENCIL_BUFFER_SurfaceBaseAddress_start(info) / 8;

      assert(_3DSTATE_HIER_DEPTH_BUFFER_SurfaceBaseAddress_start(info) % 8 == 0);
      dev->ds.hiz_offset =
         _3DSTATE_DEPTH_BUFFER_length(info) * 4 +
         _3DSTATE_STENCIL_BUFFER_length(info) * 4 +
         _3DSTATE_HIER_DEPTH_BUFFER_SurfaceBaseAddress_start(info) / 8;
   } else {
      dev->ds.stencil_offset = 0;
      dev->ds.hiz_offset = 0;
   }
}

/**
 * @brief Query the set of multisamples supported by the device.
 *
 * This function always returns non-zero, as ISL_SAMPLE_COUNT_1_BIT is always
 * supported.
 */
isl_sample_count_mask_t ATTRIBUTE_CONST
isl_device_get_sample_counts(struct isl_device *dev)
{
   if (ISL_DEV_GEN(dev) >= 9) {
      return ISL_SAMPLE_COUNT_1_BIT |
             ISL_SAMPLE_COUNT_2_BIT |
             ISL_SAMPLE_COUNT_4_BIT |
             ISL_SAMPLE_COUNT_8_BIT |
             ISL_SAMPLE_COUNT_16_BIT;
   } else if (ISL_DEV_GEN(dev) >= 8) {
      return ISL_SAMPLE_COUNT_1_BIT |
             ISL_SAMPLE_COUNT_2_BIT |
             ISL_SAMPLE_COUNT_4_BIT |
             ISL_SAMPLE_COUNT_8_BIT;
   } else if (ISL_DEV_GEN(dev) >= 7) {
      return ISL_SAMPLE_COUNT_1_BIT |
             ISL_SAMPLE_COUNT_4_BIT |
             ISL_SAMPLE_COUNT_8_BIT;
   } else if (ISL_DEV_GEN(dev) >= 6) {
      return ISL_SAMPLE_COUNT_1_BIT |
             ISL_SAMPLE_COUNT_4_BIT;
   } else {
      return ISL_SAMPLE_COUNT_1_BIT;
   }
}

/**
 * @param[out] info is written only on success
 */
static void
isl_tiling_get_info(enum isl_tiling tiling,
                    uint32_t format_bpb,
                    struct isl_tile_info *tile_info)
{
   const uint32_t bs = format_bpb / 8;
   struct isl_extent2d logical_el, phys_B;

   if (tiling != ISL_TILING_LINEAR && !isl_is_pow2(format_bpb)) {
      /* It is possible to have non-power-of-two formats in a tiled buffer.
       * The easiest way to handle this is to treat the tile as if it is three
       * times as wide.  This way no pixel will ever cross a tile boundary.
       * This really only works on legacy X and Y tiling formats.
       */
      assert(tiling == ISL_TILING_X || tiling == ISL_TILING_Y0);
      assert(bs % 3 == 0 && isl_is_pow2(format_bpb / 3));
      isl_tiling_get_info(tiling, format_bpb / 3, tile_info);
      return;
   }

   switch (tiling) {
   case ISL_TILING_LINEAR:
      assert(bs > 0);
      logical_el = isl_extent2d(1, 1);
      phys_B = isl_extent2d(bs, 1);
      break;

   case ISL_TILING_X:
      assert(bs > 0);
      logical_el = isl_extent2d(512 / bs, 8);
      phys_B = isl_extent2d(512, 8);
      break;

   case ISL_TILING_Y0:
      assert(bs > 0);
      logical_el = isl_extent2d(128 / bs, 32);
      phys_B = isl_extent2d(128, 32);
      break;

   case ISL_TILING_W:
      assert(bs == 1);
      logical_el = isl_extent2d(64, 64);
      /* From the Broadwell PRM Vol 2d, RENDER_SURFACE_STATE::SurfacePitch:
       *
       *    "If the surface is a stencil buffer (and thus has Tile Mode set
       *    to TILEMODE_WMAJOR), the pitch must be set to 2x the value
       *    computed based on width, as the stencil buffer is stored with two
       *    rows interleaved."
       *
       * This, together with the fact that stencil buffers are referred to as
       * being Y-tiled in the PRMs for older hardware implies that the
       * physical size of a W-tile is actually the same as for a Y-tile.
       */
      phys_B = isl_extent2d(128, 32);
      break;

   case ISL_TILING_Yf:
   case ISL_TILING_Ys: {
      bool is_Ys = tiling == ISL_TILING_Ys;

      assert(bs > 0);
      unsigned width = 1 << (6 + (ffs(bs) / 2) + (2 * is_Ys));
      unsigned height = 1 << (6 - (ffs(bs) / 2) + (2 * is_Ys));

      logical_el = isl_extent2d(width / bs, height);
      phys_B = isl_extent2d(width, height);
      break;
   }

   case ISL_TILING_HIZ:
      /* HiZ buffers are required to have ISL_FORMAT_HIZ which is an 8x4
       * 128bpb format.  The tiling has the same physical dimensions as
       * Y-tiling but actually has two HiZ columns per Y-tiled column.
       */
      assert(bs == 16);
      logical_el = isl_extent2d(16, 16);
      phys_B = isl_extent2d(128, 32);
      break;

   case ISL_TILING_CCS:
      /* CCS surfaces are required to have one of the GENX_CCS_* formats which
       * have a block size of 1 or 2 bits per block and each CCS element
       * corresponds to one cache-line pair in the main surface.  From the Sky
       * Lake PRM Vol. 12 in the section on planes:
       *
       *    "The Color Control Surface (CCS) contains the compression status
       *    of the cache-line pairs. The compression state of the cache-line
       *    pair is specified by 2 bits in the CCS.  Each CCS cache-line
       *    represents an area on the main surface of 16x16 sets of 128 byte
       *    Y-tiled cache-line-pairs. CCS is always Y tiled."
       *
       * The CCS being Y-tiled implies that it's an 8x8 grid of cache-lines.
       * Since each cache line corresponds to a 16x16 set of cache-line pairs,
       * that yields total tile area of 128x128 cache-line pairs or CCS
       * elements.  On older hardware, each CCS element is 1 bit and the tile
       * is 128x256 elements.
       */
      assert(format_bpb == 1 || format_bpb == 2);
      logical_el = isl_extent2d(128, 256 / format_bpb);
      phys_B = isl_extent2d(128, 32);
      break;

   default:
      unreachable("not reached");
   } /* end switch */

   *tile_info = (struct isl_tile_info) {
      .tiling = tiling,
      .format_bpb = format_bpb,
      .logical_extent_el = logical_el,
      .phys_extent_B = phys_B,
   };
}

bool
isl_color_value_is_zero(union isl_color_value value,
                        enum isl_format format)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(format);

#define RETURN_FALSE_IF_NOT_0(c, i) \
   if (fmtl->channels.c.bits && value.u32[i] != 0) \
      return false

   RETURN_FALSE_IF_NOT_0(r, 0);
   RETURN_FALSE_IF_NOT_0(g, 1);
   RETURN_FALSE_IF_NOT_0(b, 2);
   RETURN_FALSE_IF_NOT_0(a, 3);

#undef RETURN_FALSE_IF_NOT_0

   return true;
}

bool
isl_color_value_is_zero_one(union isl_color_value value,
                            enum isl_format format)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(format);

#define RETURN_FALSE_IF_NOT_0_1(c, i, field) \
   if (fmtl->channels.c.bits && value.field[i] != 0 && value.field[i] != 1) \
      return false

   if (isl_format_has_int_channel(format)) {
      RETURN_FALSE_IF_NOT_0_1(r, 0, u32);
      RETURN_FALSE_IF_NOT_0_1(g, 1, u32);
      RETURN_FALSE_IF_NOT_0_1(b, 2, u32);
      RETURN_FALSE_IF_NOT_0_1(a, 3, u32);
   } else {
      RETURN_FALSE_IF_NOT_0_1(r, 0, f32);
      RETURN_FALSE_IF_NOT_0_1(g, 1, f32);
      RETURN_FALSE_IF_NOT_0_1(b, 2, f32);
      RETURN_FALSE_IF_NOT_0_1(a, 3, f32);
   }

#undef RETURN_FALSE_IF_NOT_0_1

   return true;
}

/**
 * @param[out] tiling is set only on success
 */
static bool
isl_surf_choose_tiling(const struct isl_device *dev,
                       const struct isl_surf_init_info *restrict info,
                       enum isl_tiling *tiling)
{
   isl_tiling_flags_t tiling_flags = info->tiling_flags;

   /* HiZ surfaces always use the HiZ tiling */
   if (info->usage & ISL_SURF_USAGE_HIZ_BIT) {
      assert(info->format == ISL_FORMAT_HIZ);
      assert(tiling_flags == ISL_TILING_HIZ_BIT);
      *tiling = ISL_TILING_HIZ;
      return true;
   }

   /* CCS surfaces always use the CCS tiling */
   if (info->usage & ISL_SURF_USAGE_CCS_BIT) {
      assert(isl_format_get_layout(info->format)->txc == ISL_TXC_CCS);
      assert(tiling_flags == ISL_TILING_CCS_BIT);
      *tiling = ISL_TILING_CCS;
      return true;
   }

   if (ISL_DEV_GEN(dev) >= 6) {
      isl_gen6_filter_tiling(dev, info, &tiling_flags);
   } else {
      isl_gen4_filter_tiling(dev, info, &tiling_flags);
   }

   #define CHOOSE(__tiling) \
      do { \
         if (tiling_flags & (1u << (__tiling))) { \
            *tiling = (__tiling); \
            return true; \
          } \
      } while (0)

   /* Of the tiling modes remaining, choose the one that offers the best
    * performance.
    */

   if (info->dim == ISL_SURF_DIM_1D) {
      /* Prefer linear for 1D surfaces because they do not benefit from
       * tiling. To the contrary, tiling leads to wasted memory and poor
       * memory locality due to the swizzling and alignment restrictions
       * required in tiled surfaces.
       */
      CHOOSE(ISL_TILING_LINEAR);
   }

   CHOOSE(ISL_TILING_Ys);
   CHOOSE(ISL_TILING_Yf);
   CHOOSE(ISL_TILING_Y0);
   CHOOSE(ISL_TILING_X);
   CHOOSE(ISL_TILING_W);
   CHOOSE(ISL_TILING_LINEAR);

   #undef CHOOSE

   /* No tiling mode accomodates the inputs. */
   return false;
}

static bool
isl_choose_msaa_layout(const struct isl_device *dev,
                 const struct isl_surf_init_info *info,
                 enum isl_tiling tiling,
                 enum isl_msaa_layout *msaa_layout)
{
   if (ISL_DEV_GEN(dev) >= 8) {
      return isl_gen8_choose_msaa_layout(dev, info, tiling, msaa_layout);
   } else if (ISL_DEV_GEN(dev) >= 7) {
      return isl_gen7_choose_msaa_layout(dev, info, tiling, msaa_layout);
   } else if (ISL_DEV_GEN(dev) >= 6) {
      return isl_gen6_choose_msaa_layout(dev, info, tiling, msaa_layout);
   } else {
      return isl_gen4_choose_msaa_layout(dev, info, tiling, msaa_layout);
   }
}

struct isl_extent2d
isl_get_interleaved_msaa_px_size_sa(uint32_t samples)
{
   assert(isl_is_pow2(samples));

   /* From the Broadwell PRM >> Volume 5: Memory Views >> Computing Mip Level
    * Sizes (p133):
    *
    *    If the surface is multisampled and it is a depth or stencil surface
    *    or Multisampled Surface StorageFormat in SURFACE_STATE is
    *    MSFMT_DEPTH_STENCIL, W_L and H_L must be adjusted as follows before
    *    proceeding: [...]
    */
   return (struct isl_extent2d) {
      .width = 1 << ((ffs(samples) - 0) / 2),
      .height = 1 << ((ffs(samples) - 1) / 2),
   };
}

static void
isl_msaa_interleaved_scale_px_to_sa(uint32_t samples,
                                    uint32_t *width, uint32_t *height)
{
   const struct isl_extent2d px_size_sa =
      isl_get_interleaved_msaa_px_size_sa(samples);

   if (width)
      *width = isl_align(*width, 2) * px_size_sa.width;
   if (height)
      *height = isl_align(*height, 2) * px_size_sa.height;
}

static enum isl_array_pitch_span
isl_choose_array_pitch_span(const struct isl_device *dev,
                            const struct isl_surf_init_info *restrict info,
                            enum isl_dim_layout dim_layout,
                            const struct isl_extent4d *phys_level0_sa)
{
   switch (dim_layout) {
   case ISL_DIM_LAYOUT_GEN9_1D:
   case ISL_DIM_LAYOUT_GEN4_2D:
      if (ISL_DEV_GEN(dev) >= 8) {
         /* QPitch becomes programmable in Broadwell. So choose the
          * most compact QPitch possible in order to conserve memory.
          *
          * From the Broadwell PRM >> Volume 2d: Command Reference: Structures
          * >> RENDER_SURFACE_STATE Surface QPitch (p325):
          *
          *    - Software must ensure that this field is set to a value
          *      sufficiently large such that the array slices in the surface
          *      do not overlap. Refer to the Memory Data Formats section for
          *      information on how surfaces are stored in memory.
          *
          *    - This field specifies the distance in rows between array
          *      slices.  It is used only in the following cases:
          *
          *          - Surface Array is enabled OR
          *          - Number of Mulitsamples is not NUMSAMPLES_1 and
          *            Multisampled Surface Storage Format set to MSFMT_MSS OR
          *          - Surface Type is SURFTYPE_CUBE
          */
         return ISL_ARRAY_PITCH_SPAN_COMPACT;
      } else if (ISL_DEV_GEN(dev) >= 7) {
         /* Note that Ivybridge introduces
          * RENDER_SURFACE_STATE.SurfaceArraySpacing, which provides the
          * driver more control over the QPitch.
          */

         if (phys_level0_sa->array_len == 1) {
            /* The hardware will never use the QPitch. So choose the most
             * compact QPitch possible in order to conserve memory.
             */
            return ISL_ARRAY_PITCH_SPAN_COMPACT;
         }

         if (isl_surf_usage_is_depth_or_stencil(info->usage) ||
             (info->usage & ISL_SURF_USAGE_HIZ_BIT)) {
            /* From the Ivybridge PRM >> Volume 1 Part 1: Graphics Core >>
             * Section 6.18.4.7: Surface Arrays (p112):
             *
             *    If Surface Array Spacing is set to ARYSPC_FULL (note that
             *    the depth buffer and stencil buffer have an implied value of
             *    ARYSPC_FULL):
             */
            return ISL_ARRAY_PITCH_SPAN_FULL;
         }

         if (info->levels == 1) {
            /* We are able to set RENDER_SURFACE_STATE.SurfaceArraySpacing
             * to ARYSPC_LOD0.
             */
            return ISL_ARRAY_PITCH_SPAN_COMPACT;
         }

         return ISL_ARRAY_PITCH_SPAN_FULL;
      } else if ((ISL_DEV_GEN(dev) == 5 || ISL_DEV_GEN(dev) == 6) &&
                 ISL_DEV_USE_SEPARATE_STENCIL(dev) &&
                 isl_surf_usage_is_stencil(info->usage)) {
         /* [ILK-SNB] Errata from the Sandy Bridge PRM >> Volume 4 Part 1:
          * Graphics Core >> Section 7.18.3.7: Surface Arrays:
          *
          *    The separate stencil buffer does not support mip mapping, thus
          *    the storage for LODs other than LOD 0 is not needed.
          */
         assert(info->levels == 1);
         return ISL_ARRAY_PITCH_SPAN_COMPACT;
      } else {
         if ((ISL_DEV_GEN(dev) == 5 || ISL_DEV_GEN(dev) == 6) &&
             ISL_DEV_USE_SEPARATE_STENCIL(dev) &&
             isl_surf_usage_is_stencil(info->usage)) {
            /* [ILK-SNB] Errata from the Sandy Bridge PRM >> Volume 4 Part 1:
             * Graphics Core >> Section 7.18.3.7: Surface Arrays:
             *
             *    The separate stencil buffer does not support mip mapping,
             *    thus the storage for LODs other than LOD 0 is not needed.
             */
            assert(info->levels == 1);
            assert(phys_level0_sa->array_len == 1);
            return ISL_ARRAY_PITCH_SPAN_COMPACT;
         }

         if (phys_level0_sa->array_len == 1) {
            /* The hardware will never use the QPitch. So choose the most
             * compact QPitch possible in order to conserve memory.
             */
            return ISL_ARRAY_PITCH_SPAN_COMPACT;
         }

         return ISL_ARRAY_PITCH_SPAN_FULL;
      }

   case ISL_DIM_LAYOUT_GEN4_3D:
      /* The hardware will never use the QPitch. So choose the most
       * compact QPitch possible in order to conserve memory.
       */
      return ISL_ARRAY_PITCH_SPAN_COMPACT;

   case ISL_DIM_LAYOUT_GEN6_STENCIL_HIZ:
      /* Each array image in the gen6 stencil of HiZ surface is compact in the
       * sense that every LOD is a compact array of the same size as LOD0.
       */
      return ISL_ARRAY_PITCH_SPAN_COMPACT;
   }

   unreachable("bad isl_dim_layout");
   return ISL_ARRAY_PITCH_SPAN_FULL;
}

static void
isl_choose_image_alignment_el(const struct isl_device *dev,
                              const struct isl_surf_init_info *restrict info,
                              enum isl_tiling tiling,
                              enum isl_dim_layout dim_layout,
                              enum isl_msaa_layout msaa_layout,
                              struct isl_extent3d *image_align_el)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(info->format);
   if (fmtl->txc == ISL_TXC_MCS) {
      assert(tiling == ISL_TILING_Y0);

      /*
       * IvyBrigde PRM Vol 2, Part 1, "11.7 MCS Buffer for Render Target(s)":
       *
       * Height, width, and layout of MCS buffer in this case must match with
       * Render Target height, width, and layout. MCS buffer is tiledY.
       *
       * To avoid wasting memory, choose the smallest alignment possible:
       * HALIGN_4 and VALIGN_4.
       */
      *image_align_el = isl_extent3d(4, 4, 1);
      return;
   } else if (info->format == ISL_FORMAT_HIZ) {
      assert(ISL_DEV_GEN(dev) >= 6);
      if (ISL_DEV_GEN(dev) == 6) {
         /* HiZ surfaces on Sandy Bridge are packed tightly. */
         *image_align_el = isl_extent3d(1, 1, 1);
      } else {
         /* On gen7+, HiZ surfaces are always aligned to 16x8 pixels in the
          * primary surface which works out to 2x2 HiZ elments.
          */
         *image_align_el = isl_extent3d(2, 2, 1);
      }
      return;
   }

   if (ISL_DEV_GEN(dev) >= 9) {
      isl_gen9_choose_image_alignment_el(dev, info, tiling, dim_layout,
                                         msaa_layout, image_align_el);
   } else if (ISL_DEV_GEN(dev) >= 8) {
      isl_gen8_choose_image_alignment_el(dev, info, tiling, dim_layout,
                                         msaa_layout, image_align_el);
   } else if (ISL_DEV_GEN(dev) >= 7) {
      isl_gen7_choose_image_alignment_el(dev, info, tiling, dim_layout,
                                          msaa_layout, image_align_el);
   } else if (ISL_DEV_GEN(dev) >= 6) {
      isl_gen6_choose_image_alignment_el(dev, info, tiling, dim_layout,
                                         msaa_layout, image_align_el);
   } else {
      isl_gen4_choose_image_alignment_el(dev, info, tiling, dim_layout,
                                         msaa_layout, image_align_el);
   }
}

static enum isl_dim_layout
isl_surf_choose_dim_layout(const struct isl_device *dev,
                           enum isl_surf_dim logical_dim,
                           enum isl_tiling tiling,
                           isl_surf_usage_flags_t usage)
{
   /* Sandy bridge needs a special layout for HiZ and stencil. */
   if (ISL_DEV_GEN(dev) == 6 &&
       (tiling == ISL_TILING_W || tiling == ISL_TILING_HIZ))
      return ISL_DIM_LAYOUT_GEN6_STENCIL_HIZ;

   if (ISL_DEV_GEN(dev) >= 9) {
      switch (logical_dim) {
      case ISL_SURF_DIM_1D:
         /* From the Sky Lake PRM Vol. 5, "1D Surfaces":
          *
          *    One-dimensional surfaces use a tiling mode of linear.
          *    Technically, they are not tiled resources, but the Tiled
          *    Resource Mode field in RENDER_SURFACE_STATE is still used to
          *    indicate the alignment requirements for this linear surface
          *    (See 1D Alignment requirements for how 4K and 64KB Tiled
          *    Resource Modes impact alignment). Alternatively, a 1D surface
          *    can be defined as a 2D tiled surface (e.g. TileY or TileX) with
          *    a height of 0.
          *
          * In other words, ISL_DIM_LAYOUT_GEN9_1D is only used for linear
          * surfaces and, for tiled surfaces, ISL_DIM_LAYOUT_GEN4_2D is used.
          */
         if (tiling == ISL_TILING_LINEAR)
            return ISL_DIM_LAYOUT_GEN9_1D;
         else
            return ISL_DIM_LAYOUT_GEN4_2D;
      case ISL_SURF_DIM_2D:
      case ISL_SURF_DIM_3D:
         return ISL_DIM_LAYOUT_GEN4_2D;
      }
   } else {
      switch (logical_dim) {
      case ISL_SURF_DIM_1D:
      case ISL_SURF_DIM_2D:
         /* From the G45 PRM Vol. 1a, "6.17.4.1 Hardware Cube Map Layout":
          *
          * The cube face textures are stored in the same way as 3D surfaces
          * are stored (see section 6.17.5 for details).  For cube surfaces,
          * however, the depth is equal to the number of faces (always 6) and 
          * is not reduced for each MIP.
          */
         if (ISL_DEV_GEN(dev) == 4 && (usage & ISL_SURF_USAGE_CUBE_BIT))
            return ISL_DIM_LAYOUT_GEN4_3D;

         return ISL_DIM_LAYOUT_GEN4_2D;
      case ISL_SURF_DIM_3D:
         return ISL_DIM_LAYOUT_GEN4_3D;
      }
   }

   unreachable("bad isl_surf_dim");
   return ISL_DIM_LAYOUT_GEN4_2D;
}

/**
 * Calculate the physical extent of the surface's first level, in units of
 * surface samples. The result is aligned to the format's compression block.
 */
static void
isl_calc_phys_level0_extent_sa(const struct isl_device *dev,
                               const struct isl_surf_init_info *restrict info,
                               enum isl_dim_layout dim_layout,
                               enum isl_tiling tiling,
                               enum isl_msaa_layout msaa_layout,
                               struct isl_extent4d *phys_level0_sa)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(info->format);

   if (isl_format_is_yuv(info->format))
      isl_finishme("%s:%s: YUV format", __FILE__, __func__);

   switch (info->dim) {
   case ISL_SURF_DIM_1D:
      assert(info->height == 1);
      assert(info->depth == 1);
      assert(info->samples == 1);

      switch (dim_layout) {
      case ISL_DIM_LAYOUT_GEN4_3D:
         unreachable("bad isl_dim_layout");

      case ISL_DIM_LAYOUT_GEN9_1D:
      case ISL_DIM_LAYOUT_GEN4_2D:
      case ISL_DIM_LAYOUT_GEN6_STENCIL_HIZ:
         *phys_level0_sa = (struct isl_extent4d) {
            .w = isl_align_npot(info->width, fmtl->bw),
            .h = fmtl->bh,
            .d = 1,
            .a = info->array_len,
         };
         break;
      }
      break;

   case ISL_SURF_DIM_2D:
      if (ISL_DEV_GEN(dev) == 4 && (info->usage & ISL_SURF_USAGE_CUBE_BIT))
         assert(dim_layout == ISL_DIM_LAYOUT_GEN4_3D);
      else
         assert(dim_layout == ISL_DIM_LAYOUT_GEN4_2D ||
                dim_layout == ISL_DIM_LAYOUT_GEN6_STENCIL_HIZ);

      if (tiling == ISL_TILING_Ys && info->samples > 1)
         isl_finishme("%s:%s: multisample TileYs layout", __FILE__, __func__);

      switch (msaa_layout) {
      case ISL_MSAA_LAYOUT_NONE:
         assert(info->depth == 1);
         assert(info->samples == 1);

         *phys_level0_sa = (struct isl_extent4d) {
            .w = isl_align_npot(info->width, fmtl->bw),
            .h = isl_align_npot(info->height, fmtl->bh),
            .d = 1,
            .a = info->array_len,
         };
         break;

      case ISL_MSAA_LAYOUT_ARRAY:
         assert(info->depth == 1);
         assert(info->levels == 1);
         assert(isl_format_supports_multisampling(dev->info, info->format));
         assert(fmtl->bw == 1 && fmtl->bh == 1);

         *phys_level0_sa = (struct isl_extent4d) {
            .w = info->width,
            .h = info->height,
            .d = 1,
            .a = info->array_len * info->samples,
         };
         break;

      case ISL_MSAA_LAYOUT_INTERLEAVED:
         assert(info->depth == 1);
         assert(info->levels == 1);
         assert(isl_format_supports_multisampling(dev->info, info->format));

         *phys_level0_sa = (struct isl_extent4d) {
            .w = info->width,
            .h = info->height,
            .d = 1,
            .a = info->array_len,
         };

         isl_msaa_interleaved_scale_px_to_sa(info->samples,
                                             &phys_level0_sa->w,
                                             &phys_level0_sa->h);

         phys_level0_sa->w = isl_align(phys_level0_sa->w, fmtl->bw);
         phys_level0_sa->h = isl_align(phys_level0_sa->h, fmtl->bh);
         break;
      }
      break;

   case ISL_SURF_DIM_3D:
      assert(info->array_len == 1);
      assert(info->samples == 1);

      if (fmtl->bd > 1) {
         isl_finishme("%s:%s: compression block with depth > 1",
                      __FILE__, __func__);
      }

      switch (dim_layout) {
      case ISL_DIM_LAYOUT_GEN9_1D:
      case ISL_DIM_LAYOUT_GEN6_STENCIL_HIZ:
         unreachable("bad isl_dim_layout");

      case ISL_DIM_LAYOUT_GEN4_2D:
         assert(ISL_DEV_GEN(dev) >= 9);

         *phys_level0_sa = (struct isl_extent4d) {
            .w = isl_align_npot(info->width, fmtl->bw),
            .h = isl_align_npot(info->height, fmtl->bh),
            .d = 1,
            .a = info->depth,
         };
         break;

      case ISL_DIM_LAYOUT_GEN4_3D:
         assert(ISL_DEV_GEN(dev) < 9);
         *phys_level0_sa = (struct isl_extent4d) {
            .w = isl_align(info->width, fmtl->bw),
            .h = isl_align(info->height, fmtl->bh),
            .d = info->depth,
            .a = 1,
         };
         break;
      }
      break;
   }
}

/**
 * Calculate the pitch between physical array slices, in units of rows of
 * surface elements.
 */
static uint32_t
isl_calc_array_pitch_el_rows_gen4_2d(
      const struct isl_device *dev,
      const struct isl_surf_init_info *restrict info,
      const struct isl_tile_info *tile_info,
      const struct isl_extent3d *image_align_sa,
      const struct isl_extent4d *phys_level0_sa,
      enum isl_array_pitch_span array_pitch_span,
      const struct isl_extent2d *phys_slice0_sa)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(info->format);
   uint32_t pitch_sa_rows = 0;

   switch (array_pitch_span) {
   case ISL_ARRAY_PITCH_SPAN_COMPACT:
      pitch_sa_rows = isl_align_npot(phys_slice0_sa->h, image_align_sa->h);
      break;
   case ISL_ARRAY_PITCH_SPAN_FULL: {
      /* The QPitch equation is found in the Broadwell PRM >> Volume 5:
       * Memory Views >> Common Surface Formats >> Surface Layout >> 2D
       * Surfaces >> Surface Arrays.
       */
      uint32_t H0_sa = phys_level0_sa->h;
      uint32_t H1_sa = isl_minify(H0_sa, 1);

      uint32_t h0_sa = isl_align_npot(H0_sa, image_align_sa->h);
      uint32_t h1_sa = isl_align_npot(H1_sa, image_align_sa->h);

      uint32_t m;
      if (ISL_DEV_GEN(dev) >= 7) {
         /* The QPitch equation changed slightly in Ivybridge. */
         m = 12;
      } else {
         m = 11;
      }

      pitch_sa_rows = h0_sa + h1_sa + (m * image_align_sa->h);

      if (ISL_DEV_GEN(dev) == 6 && info->samples > 1 &&
          (info->height % 4 == 1)) {
         /* [SNB] Errata from the Sandy Bridge PRM >> Volume 4 Part 1:
          * Graphics Core >> Section 7.18.3.7: Surface Arrays:
          *
          *    [SNB] Errata: Sampler MSAA Qpitch will be 4 greater than
          *    the value calculated in the equation above , for every
          *    other odd Surface Height starting from 1 i.e. 1,5,9,13.
          *
          * XXX(chadv): Is the errata natural corollary of the physical
          * layout of interleaved samples?
          */
         pitch_sa_rows += 4;
      }

      pitch_sa_rows = isl_align_npot(pitch_sa_rows, fmtl->bh);
      } /* end case */
      break;
   }

   assert(pitch_sa_rows % fmtl->bh == 0);
   uint32_t pitch_el_rows = pitch_sa_rows / fmtl->bh;

   if (ISL_DEV_GEN(dev) >= 9 && fmtl->txc == ISL_TXC_CCS) {
      /*
       * From the Sky Lake PRM Vol 7, "MCS Buffer for Render Target(s)" (p. 632):
       *
       *    "Mip-mapped and arrayed surfaces are supported with MCS buffer
       *    layout with these alignments in the RT space: Horizontal
       *    Alignment = 128 and Vertical Alignment = 64."
       *
       * From the Sky Lake PRM Vol. 2d, "RENDER_SURFACE_STATE" (p. 435):
       *
       *    "For non-multisampled render target's CCS auxiliary surface,
       *    QPitch must be computed with Horizontal Alignment = 128 and
       *    Surface Vertical Alignment = 256. These alignments are only for
       *    CCS buffer and not for associated render target."
       *
       * The first restriction is already handled by isl_choose_image_alignment_el
       * but the second restriction, which is an extension of the first, only
       * applies to qpitch and must be applied here.
       */
      assert(fmtl->bh == 4);
      pitch_el_rows = isl_align(pitch_el_rows, 256 / 4);
   }

   if (ISL_DEV_GEN(dev) >= 9 &&
       info->dim == ISL_SURF_DIM_3D &&
       tile_info->tiling != ISL_TILING_LINEAR) {
      /* From the Skylake BSpec >> RENDER_SURFACE_STATE >> Surface QPitch:
       *
       *    Tile Mode != Linear: This field must be set to an integer multiple
       *    of the tile height
       */
      pitch_el_rows = isl_align(pitch_el_rows, tile_info->logical_extent_el.height);
   }

   return pitch_el_rows;
}

/**
 * A variant of isl_calc_phys_slice0_extent_sa() specific to
 * ISL_DIM_LAYOUT_GEN4_2D.
 */
static void
isl_calc_phys_slice0_extent_sa_gen4_2d(
      const struct isl_device *dev,
      const struct isl_surf_init_info *restrict info,
      enum isl_msaa_layout msaa_layout,
      const struct isl_extent3d *image_align_sa,
      const struct isl_extent4d *phys_level0_sa,
      struct isl_extent2d *phys_slice0_sa)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(info->format);

   assert(phys_level0_sa->depth == 1);

   if (info->levels == 1) {
      /* Do not pad the surface to the image alignment. Instead, pad it only
       * to the pixel format's block alignment.
       *
       * For tiled surfaces, using a reduced alignment here avoids wasting CPU
       * cycles on the below mipmap layout caluclations. Reducing the
       * alignment here is safe because we later align the row pitch and array
       * pitch to the tile boundary. It is safe even for
       * ISL_MSAA_LAYOUT_INTERLEAVED, because phys_level0_sa is already scaled
       * to accomodate the interleaved samples.
       *
       * For linear surfaces, reducing the alignment here permits us to later
       * choose an arbitrary, non-aligned row pitch. If the surface backs
       * a VkBuffer, then an arbitrary pitch may be needed to accomodate
       * VkBufferImageCopy::bufferRowLength.
       */
      *phys_slice0_sa = (struct isl_extent2d) {
         .w = isl_align_npot(phys_level0_sa->w, fmtl->bw),
         .h = isl_align_npot(phys_level0_sa->h, fmtl->bh),
      };
      return;
   }

   uint32_t slice_top_w = 0;
   uint32_t slice_bottom_w = 0;
   uint32_t slice_left_h = 0;
   uint32_t slice_right_h = 0;

   uint32_t W0 = phys_level0_sa->w;
   uint32_t H0 = phys_level0_sa->h;

   for (uint32_t l = 0; l < info->levels; ++l) {
      uint32_t W = isl_minify(W0, l);
      uint32_t H = isl_minify(H0, l);

      uint32_t w = isl_align_npot(W, image_align_sa->w);
      uint32_t h = isl_align_npot(H, image_align_sa->h);

      if (l == 0) {
         slice_top_w = w;
         slice_left_h = h;
         slice_right_h = h;
      } else if (l == 1) {
         slice_bottom_w = w;
         slice_left_h += h;
      } else if (l == 2) {
         slice_bottom_w += w;
         slice_right_h += h;
      } else {
         slice_right_h += h;
      }
   }

   *phys_slice0_sa = (struct isl_extent2d) {
      .w = MAX(slice_top_w, slice_bottom_w),
      .h = MAX(slice_left_h, slice_right_h),
   };
}

static void
isl_calc_phys_total_extent_el_gen4_2d(
      const struct isl_device *dev,
      const struct isl_surf_init_info *restrict info,
      const struct isl_tile_info *tile_info,
      enum isl_msaa_layout msaa_layout,
      const struct isl_extent3d *image_align_sa,
      const struct isl_extent4d *phys_level0_sa,
      enum isl_array_pitch_span array_pitch_span,
      uint32_t *array_pitch_el_rows,
      struct isl_extent2d *total_extent_el)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(info->format);

   struct isl_extent2d phys_slice0_sa;
   isl_calc_phys_slice0_extent_sa_gen4_2d(dev, info, msaa_layout,
                                          image_align_sa, phys_level0_sa,
                                          &phys_slice0_sa);
   *array_pitch_el_rows =
      isl_calc_array_pitch_el_rows_gen4_2d(dev, info, tile_info,
                                           image_align_sa, phys_level0_sa,
                                           array_pitch_span,
                                           &phys_slice0_sa);
   *total_extent_el = (struct isl_extent2d) {
      .w = isl_assert_div(phys_slice0_sa.w, fmtl->bw),
      .h = *array_pitch_el_rows * (phys_level0_sa->array_len - 1) +
           isl_assert_div(phys_slice0_sa.h, fmtl->bh),
   };
}

/**
 * A variant of isl_calc_phys_slice0_extent_sa() specific to
 * ISL_DIM_LAYOUT_GEN4_3D.
 */
static void
isl_calc_phys_total_extent_el_gen4_3d(
      const struct isl_device *dev,
      const struct isl_surf_init_info *restrict info,
      const struct isl_extent3d *image_align_sa,
      const struct isl_extent4d *phys_level0_sa,
      uint32_t *array_pitch_el_rows,
      struct isl_extent2d *phys_total_el)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(info->format);

   assert(info->samples == 1);

   if (info->dim != ISL_SURF_DIM_3D) {
      /* From the G45 PRM Vol. 1a, "6.17.4.1 Hardware Cube Map Layout":
       *
       * The cube face textures are stored in the same way as 3D surfaces
       * are stored (see section 6.17.5 for details).  For cube surfaces,
       * however, the depth is equal to the number of faces (always 6) and
       * is not reduced for each MIP.
       */
      assert(ISL_DEV_GEN(dev) == 4);
      assert(info->usage & ISL_SURF_USAGE_CUBE_BIT);
      assert(phys_level0_sa->array_len == 6);
   } else {
      assert(phys_level0_sa->array_len == 1);
   }

   uint32_t total_w = 0;
   uint32_t total_h = 0;

   uint32_t W0 = phys_level0_sa->w;
   uint32_t H0 = phys_level0_sa->h;
   uint32_t D0 = phys_level0_sa->d;
   uint32_t A0 = phys_level0_sa->a;

   for (uint32_t l = 0; l < info->levels; ++l) {
      uint32_t level_w = isl_align_npot(isl_minify(W0, l), image_align_sa->w);
      uint32_t level_h = isl_align_npot(isl_minify(H0, l), image_align_sa->h);
      uint32_t level_d = info->dim == ISL_SURF_DIM_3D ? isl_minify(D0, l) : A0;

      uint32_t max_layers_horiz = MIN(level_d, 1u << l);
      uint32_t max_layers_vert = isl_align(level_d, 1u << l) / (1u << l);

      total_w = MAX(total_w, level_w * max_layers_horiz);
      total_h += level_h * max_layers_vert;
   }

   /* GEN4_3D layouts don't really have an array pitch since each LOD has a
    * different number of horizontal and vertical layers.  We have to set it
    * to something, so at least make it true for LOD0.
    */
   *array_pitch_el_rows =
      isl_align_npot(phys_level0_sa->h, image_align_sa->h) / fmtl->bw;
   *phys_total_el = (struct isl_extent2d) {
      .w = isl_assert_div(total_w, fmtl->bw),
      .h = isl_assert_div(total_h, fmtl->bh),
   };
}

/**
 * A variant of isl_calc_phys_slice0_extent_sa() specific to
 * ISL_DIM_LAYOUT_GEN6_STENCIL_HIZ.
 */
static void
isl_calc_phys_total_extent_el_gen6_stencil_hiz(
      const struct isl_device *dev,
      const struct isl_surf_init_info *restrict info,
      const struct isl_tile_info *tile_info,
      const struct isl_extent3d *image_align_sa,
      const struct isl_extent4d *phys_level0_sa,
      uint32_t *array_pitch_el_rows,
      struct isl_extent2d *phys_total_el)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(info->format);

   const struct isl_extent2d tile_extent_sa = {
      .w = tile_info->logical_extent_el.w * fmtl->bw,
      .h = tile_info->logical_extent_el.h * fmtl->bh,
   };
   /* Tile size is a multiple of image alignment */
   assert(tile_extent_sa.w % image_align_sa->w == 0);
   assert(tile_extent_sa.h % image_align_sa->h == 0);

   const uint32_t W0 = phys_level0_sa->w;
   const uint32_t H0 = phys_level0_sa->h;

   /* Each image has the same height as LOD0 because the hardware thinks
    * everything is LOD0
    */
   const uint32_t H = isl_align(H0, image_align_sa->h) * phys_level0_sa->a;

   uint32_t total_top_w = 0;
   uint32_t total_bottom_w = 0;
   uint32_t total_h = 0;

   for (uint32_t l = 0; l < info->levels; ++l) {
      const uint32_t W = isl_minify(W0, l);

      const uint32_t w = isl_align(W, tile_extent_sa.w);
      const uint32_t h = isl_align(H, tile_extent_sa.h);

      if (l == 0) {
         total_top_w = w;
         total_h = h;
      } else if (l == 1) {
         total_bottom_w = w;
         total_h += h;
      } else {
         total_bottom_w += w;
      }
   }

   *array_pitch_el_rows =
      isl_assert_div(isl_align(H0, image_align_sa->h), fmtl->bh);
   *phys_total_el = (struct isl_extent2d) {
      .w = isl_assert_div(MAX(total_top_w, total_bottom_w), fmtl->bw),
      .h = isl_assert_div(total_h, fmtl->bh),
   };
}

/**
 * A variant of isl_calc_phys_slice0_extent_sa() specific to
 * ISL_DIM_LAYOUT_GEN9_1D.
 */
static void
isl_calc_phys_total_extent_el_gen9_1d(
      const struct isl_device *dev,
      const struct isl_surf_init_info *restrict info,
      const struct isl_extent3d *image_align_sa,
      const struct isl_extent4d *phys_level0_sa,
      uint32_t *array_pitch_el_rows,
      struct isl_extent2d *phys_total_el)
{
   MAYBE_UNUSED const struct isl_format_layout *fmtl = isl_format_get_layout(info->format);

   assert(phys_level0_sa->height / fmtl->bh == 1);
   assert(phys_level0_sa->depth == 1);
   assert(info->samples == 1);
   assert(image_align_sa->w >= fmtl->bw);

   uint32_t slice_w = 0;
   const uint32_t W0 = phys_level0_sa->w;

   for (uint32_t l = 0; l < info->levels; ++l) {
      uint32_t W = isl_minify(W0, l);
      uint32_t w = isl_align_npot(W, image_align_sa->w);

      slice_w += w;
   }

   *array_pitch_el_rows = 1;
   *phys_total_el = (struct isl_extent2d) {
      .w = isl_assert_div(slice_w, fmtl->bw),
      .h = phys_level0_sa->array_len,
   };
}

/**
 * Calculate the two-dimensional total physical extent of the surface, in
 * units of surface elements.
 */
static void
isl_calc_phys_total_extent_el(const struct isl_device *dev,
                              const struct isl_surf_init_info *restrict info,
                              const struct isl_tile_info *tile_info,
                              enum isl_dim_layout dim_layout,
                              enum isl_msaa_layout msaa_layout,
                              const struct isl_extent3d *image_align_sa,
                              const struct isl_extent4d *phys_level0_sa,
                              enum isl_array_pitch_span array_pitch_span,
                              uint32_t *array_pitch_el_rows,
                              struct isl_extent2d *total_extent_el)
{
   switch (dim_layout) {
   case ISL_DIM_LAYOUT_GEN9_1D:
      assert(array_pitch_span == ISL_ARRAY_PITCH_SPAN_COMPACT);
      isl_calc_phys_total_extent_el_gen9_1d(dev, info,
                                            image_align_sa, phys_level0_sa,
                                            array_pitch_el_rows,
                                            total_extent_el);
      return;
   case ISL_DIM_LAYOUT_GEN4_2D:
      isl_calc_phys_total_extent_el_gen4_2d(dev, info, tile_info, msaa_layout,
                                            image_align_sa, phys_level0_sa,
                                            array_pitch_span,
                                            array_pitch_el_rows,
                                            total_extent_el);
      return;
   case ISL_DIM_LAYOUT_GEN6_STENCIL_HIZ:
      assert(array_pitch_span == ISL_ARRAY_PITCH_SPAN_COMPACT);
      isl_calc_phys_total_extent_el_gen6_stencil_hiz(dev, info, tile_info,
                                                     image_align_sa,
                                                     phys_level0_sa,
                                                     array_pitch_el_rows,
                                                     total_extent_el);
      return;
   case ISL_DIM_LAYOUT_GEN4_3D:
      assert(array_pitch_span == ISL_ARRAY_PITCH_SPAN_COMPACT);
      isl_calc_phys_total_extent_el_gen4_3d(dev, info,
                                            image_align_sa, phys_level0_sa,
                                            array_pitch_el_rows,
                                            total_extent_el);
      return;
   }
}

static uint32_t
isl_calc_row_pitch_alignment(const struct isl_surf_init_info *surf_info,
                             const struct isl_tile_info *tile_info)
{
   if (tile_info->tiling != ISL_TILING_LINEAR)
      return tile_info->phys_extent_B.width;

   /* From the Broadwel PRM >> Volume 2d: Command Reference: Structures >>
    * RENDER_SURFACE_STATE Surface Pitch (p349):
    *
    *    - For linear render target surfaces and surfaces accessed with the
    *      typed data port messages, the pitch must be a multiple of the
    *      element size for non-YUV surface formats.  Pitch must be
    *      a multiple of 2 * element size for YUV surface formats.
    *
    *    - [Requirements for SURFTYPE_BUFFER and SURFTYPE_STRBUF, which we
    *      ignore because isl doesn't do buffers.]
    *
    *    - For other linear surfaces, the pitch can be any multiple of
    *      bytes.
    */
   const struct isl_format_layout *fmtl = isl_format_get_layout(surf_info->format);
   const uint32_t bs = fmtl->bpb / 8;

   if (surf_info->usage & ISL_SURF_USAGE_RENDER_TARGET_BIT) {
      if (isl_format_is_yuv(surf_info->format)) {
         return 2 * bs;
      } else  {
         return bs;
      }
   }

   return 1;
}

static uint32_t
isl_calc_linear_min_row_pitch(const struct isl_device *dev,
                              const struct isl_surf_init_info *info,
                              const struct isl_extent2d *phys_total_el,
                              uint32_t alignment_B)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(info->format);
   const uint32_t bs = fmtl->bpb / 8;

   return isl_align_npot(bs * phys_total_el->w, alignment_B);
}

static uint32_t
isl_calc_tiled_min_row_pitch(const struct isl_device *dev,
                             const struct isl_surf_init_info *surf_info,
                             const struct isl_tile_info *tile_info,
                             const struct isl_extent2d *phys_total_el,
                             uint32_t alignment_B)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(surf_info->format);

   assert(fmtl->bpb % tile_info->format_bpb == 0);

   const uint32_t tile_el_scale = fmtl->bpb / tile_info->format_bpb;
   const uint32_t total_w_tl =
      isl_align_div(phys_total_el->w * tile_el_scale,
                    tile_info->logical_extent_el.width);

   assert(alignment_B == tile_info->phys_extent_B.width);
   return total_w_tl * tile_info->phys_extent_B.width;
}

static uint32_t
isl_calc_min_row_pitch(const struct isl_device *dev,
                       const struct isl_surf_init_info *surf_info,
                       const struct isl_tile_info *tile_info,
                       const struct isl_extent2d *phys_total_el,
                       uint32_t alignment_B)
{
   if (tile_info->tiling == ISL_TILING_LINEAR) {
      return isl_calc_linear_min_row_pitch(dev, surf_info, phys_total_el,
                                           alignment_B);
   } else {
      return isl_calc_tiled_min_row_pitch(dev, surf_info, tile_info,
                                          phys_total_el, alignment_B);
   }
}

/**
 * Is `pitch` in the valid range for a hardware bitfield, if the bitfield's
 * size is `bits` bits?
 *
 * Hardware pitch fields are offset by 1. For example, if the size of
 * RENDER_SURFACE_STATE::SurfacePitch is B bits, then the range of valid
 * pitches is [1, 2^b] inclusive.  If the surface pitch is N, then
 * RENDER_SURFACE_STATE::SurfacePitch must be set to N-1.
 */
static bool
pitch_in_range(uint32_t n, uint32_t bits)
{
   assert(n != 0);
   return likely(bits != 0 && 1 <= n && n <= (1 << bits));
}

static bool
isl_calc_row_pitch(const struct isl_device *dev,
                   const struct isl_surf_init_info *surf_info,
                   const struct isl_tile_info *tile_info,
                   enum isl_dim_layout dim_layout,
                   const struct isl_extent2d *phys_total_el,
                   uint32_t *out_row_pitch_B)
{
   uint32_t alignment_B =
      isl_calc_row_pitch_alignment(surf_info, tile_info);

   /* If pitch isn't given and it can be chosen freely, align it by cache line
    * allowing one to use blit engine on the surface.
    */
   if (surf_info->row_pitch_B == 0 && tile_info->tiling == ISL_TILING_LINEAR) {
      /* From the Broadwell PRM docs for XY_SRC_COPY_BLT::SourceBaseAddress:
       *
       *    "Base address of the destination surface: X=0, Y=0. Lower 32bits
       *    of the 48bit addressing. When Src Tiling is enabled (Bit_15
       *    enabled), this address must be 4KB-aligned. When Tiling is not
       *    enabled, this address should be CL (64byte) aligned."
       */
      alignment_B = MAX2(alignment_B, 64);
   }

   const uint32_t min_row_pitch_B =
      isl_calc_min_row_pitch(dev, surf_info, tile_info, phys_total_el,
                             alignment_B);

   uint32_t row_pitch_B = min_row_pitch_B;

   if (surf_info->row_pitch_B != 0) {
      row_pitch_B = surf_info->row_pitch_B;

      if (row_pitch_B < min_row_pitch_B)
         return false;

      if (row_pitch_B % alignment_B != 0)
         return false;
   }

   const uint32_t row_pitch_tl = row_pitch_B / tile_info->phys_extent_B.width;

   if (row_pitch_B == 0)
      return false;

   if (dim_layout == ISL_DIM_LAYOUT_GEN9_1D) {
      /* SurfacePitch is ignored for this layout. */
      goto done;
   }

   if ((surf_info->usage & (ISL_SURF_USAGE_RENDER_TARGET_BIT |
                            ISL_SURF_USAGE_TEXTURE_BIT |
                            ISL_SURF_USAGE_STORAGE_BIT)) &&
       !pitch_in_range(row_pitch_B, RENDER_SURFACE_STATE_SurfacePitch_bits(dev->info)))
      return false;

   if ((surf_info->usage & (ISL_SURF_USAGE_CCS_BIT |
                            ISL_SURF_USAGE_MCS_BIT)) &&
       !pitch_in_range(row_pitch_tl, RENDER_SURFACE_STATE_AuxiliarySurfacePitch_bits(dev->info)))
      return false;

   if ((surf_info->usage & ISL_SURF_USAGE_DEPTH_BIT) &&
       !pitch_in_range(row_pitch_B, _3DSTATE_DEPTH_BUFFER_SurfacePitch_bits(dev->info)))
      return false;

   if ((surf_info->usage & ISL_SURF_USAGE_HIZ_BIT) &&
       !pitch_in_range(row_pitch_B, _3DSTATE_HIER_DEPTH_BUFFER_SurfacePitch_bits(dev->info)))
      return false;

   const uint32_t stencil_pitch_bits = dev->use_separate_stencil ?
      _3DSTATE_STENCIL_BUFFER_SurfacePitch_bits(dev->info) :
      _3DSTATE_DEPTH_BUFFER_SurfacePitch_bits(dev->info);

   if ((surf_info->usage & ISL_SURF_USAGE_STENCIL_BIT) &&
       !pitch_in_range(row_pitch_B, stencil_pitch_bits))
      return false;

 done:
   *out_row_pitch_B = row_pitch_B;
   return true;
}

bool
isl_surf_init_s(const struct isl_device *dev,
                struct isl_surf *surf,
                const struct isl_surf_init_info *restrict info)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(info->format);

   const struct isl_extent4d logical_level0_px = {
      .w = info->width,
      .h = info->height,
      .d = info->depth,
      .a = info->array_len,
   };

   enum isl_tiling tiling;
   if (!isl_surf_choose_tiling(dev, info, &tiling))
      return false;

   struct isl_tile_info tile_info;
   isl_tiling_get_info(tiling, fmtl->bpb, &tile_info);

   const enum isl_dim_layout dim_layout =
      isl_surf_choose_dim_layout(dev, info->dim, tiling, info->usage);

   enum isl_msaa_layout msaa_layout;
   if (!isl_choose_msaa_layout(dev, info, tiling, &msaa_layout))
       return false;

   struct isl_extent3d image_align_el;
   isl_choose_image_alignment_el(dev, info, tiling, dim_layout, msaa_layout,
                                 &image_align_el);

   struct isl_extent3d image_align_sa =
      isl_extent3d_el_to_sa(info->format, image_align_el);

   struct isl_extent4d phys_level0_sa;
   isl_calc_phys_level0_extent_sa(dev, info, dim_layout, tiling, msaa_layout,
                                  &phys_level0_sa);
   assert(phys_level0_sa.w % fmtl->bw == 0);
   assert(phys_level0_sa.h % fmtl->bh == 0);

   enum isl_array_pitch_span array_pitch_span =
      isl_choose_array_pitch_span(dev, info, dim_layout, &phys_level0_sa);

   uint32_t array_pitch_el_rows;
   struct isl_extent2d phys_total_el;
   isl_calc_phys_total_extent_el(dev, info, &tile_info,
                                 dim_layout, msaa_layout,
                                 &image_align_sa, &phys_level0_sa,
                                 array_pitch_span, &array_pitch_el_rows,
                                 &phys_total_el);

   uint32_t row_pitch_B;
   if (!isl_calc_row_pitch(dev, info, &tile_info, dim_layout,
                           &phys_total_el, &row_pitch_B))
      return false;

   uint32_t base_alignment_B;
   uint64_t size_B;
   if (tiling == ISL_TILING_LINEAR) {
      size_B = (uint64_t) row_pitch_B * phys_total_el.h;

      /* From the Broadwell PRM Vol 2d, RENDER_SURFACE_STATE::SurfaceBaseAddress:
       *
       *    "The Base Address for linear render target surfaces and surfaces
       *    accessed with the typed surface read/write data port messages must
       *    be element-size aligned, for non-YUV surface formats, or a
       *    multiple of 2 element-sizes for YUV surface formats. Other linear
       *    surfaces have no alignment requirements (byte alignment is
       *    sufficient.)"
       */
      base_alignment_B = MAX(1, info->min_alignment_B);
      if (info->usage & ISL_SURF_USAGE_RENDER_TARGET_BIT) {
         if (isl_format_is_yuv(info->format)) {
            base_alignment_B = MAX(base_alignment_B, fmtl->bpb / 4);
         } else {
            base_alignment_B = MAX(base_alignment_B, fmtl->bpb / 8);
         }
      }
      base_alignment_B = isl_round_up_to_power_of_two(base_alignment_B);
   } else {
      const uint32_t total_h_tl =
         isl_align_div(phys_total_el.h, tile_info.logical_extent_el.height);

      size_B = (uint64_t) total_h_tl * tile_info.phys_extent_B.height * row_pitch_B;

      const uint32_t tile_size_B = tile_info.phys_extent_B.width *
                                   tile_info.phys_extent_B.height;
      assert(isl_is_pow2(info->min_alignment_B) && isl_is_pow2(tile_size_B));
      base_alignment_B = MAX(info->min_alignment_B, tile_size_B);
   }

   if (ISL_DEV_GEN(dev) < 9) {
      /* From the Broadwell PRM Vol 5, Surface Layout:
       *
       *    "In addition to restrictions on maximum height, width, and depth,
       *     surfaces are also restricted to a maximum size in bytes. This
       *     maximum is 2 GB for all products and all surface types."
       *
       * This comment is applicable to all Pre-gen9 platforms.
       */
      if (size_B > (uint64_t) 1 << 31)
         return false;
   } else if (ISL_DEV_GEN(dev) < 11) {
      /* From the Skylake PRM Vol 5, Maximum Surface Size in Bytes:
       *    "In addition to restrictions on maximum height, width, and depth,
       *     surfaces are also restricted to a maximum size of 2^38 bytes.
       *     All pixels within the surface must be contained within 2^38 bytes
       *     of the base address."
       */
      if (size_B > (uint64_t) 1 << 38)
         return false;
   } else {
      /* gen11+ platforms raised this limit to 2^44 bytes. */
      if (size_B > (uint64_t) 1 << 44)
         return false;
   }

   *surf = (struct isl_surf) {
      .dim = info->dim,
      .dim_layout = dim_layout,
      .msaa_layout = msaa_layout,
      .tiling = tiling,
      .format = info->format,

      .levels = info->levels,
      .samples = info->samples,

      .image_alignment_el = image_align_el,
      .logical_level0_px = logical_level0_px,
      .phys_level0_sa = phys_level0_sa,

      .size_B = size_B,
      .alignment_B = base_alignment_B,
      .row_pitch_B = row_pitch_B,
      .array_pitch_el_rows = array_pitch_el_rows,
      .array_pitch_span = array_pitch_span,

      .usage = info->usage,
   };

   return true;
}

void
isl_surf_get_tile_info(const struct isl_surf *surf,
                       struct isl_tile_info *tile_info)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(surf->format);
   isl_tiling_get_info(surf->tiling, fmtl->bpb, tile_info);
}

bool
isl_surf_get_hiz_surf(const struct isl_device *dev,
                      const struct isl_surf *surf,
                      struct isl_surf *hiz_surf)
{
   assert(ISL_DEV_GEN(dev) >= 5 && ISL_DEV_USE_SEPARATE_STENCIL(dev));

   /* Multisampled depth is always interleaved */
   assert(surf->msaa_layout == ISL_MSAA_LAYOUT_NONE ||
          surf->msaa_layout == ISL_MSAA_LAYOUT_INTERLEAVED);

   /* From the Broadwell PRM Vol. 7, "Hierarchical Depth Buffer":
    *
    *    "The Surface Type, Height, Width, Depth, Minimum Array Element, Render
    *    Target View Extent, and Depth Coordinate Offset X/Y of the
    *    hierarchical depth buffer are inherited from the depth buffer. The
    *    height and width of the hierarchical depth buffer that must be
    *    allocated are computed by the following formulas, where HZ is the
    *    hierarchical depth buffer and Z is the depth buffer. The Z_Height,
    *    Z_Width, and Z_Depth values given in these formulas are those present
    *    in 3DSTATE_DEPTH_BUFFER incremented by one.
    *
    *    "The value of Z_Height and Z_Width must each be multiplied by 2 before
    *    being applied to the table below if Number of Multisamples is set to
    *    NUMSAMPLES_4. The value of Z_Height must be multiplied by 2 and
    *    Z_Width must be multiplied by 4 before being applied to the table
    *    below if Number of Multisamples is set to NUMSAMPLES_8."
    *
    * In the Sky Lake PRM, the second paragraph is replaced with this:
    *
    *    "The Z_Height and Z_Width values must equal those present in
    *    3DSTATE_DEPTH_BUFFER incremented by one."
    *
    * In other words, on Sandy Bridge through Broadwell, each 128-bit HiZ
    * block corresponds to a region of 8x4 samples in the primary depth
    * surface.  On Sky Lake, on the other hand, each HiZ block corresponds to
    * a region of 8x4 pixels in the primary depth surface regardless of the
    * number of samples.  The dimensions of a HiZ block in both pixels and
    * samples are given in the table below:
    *
    *                    | SNB - BDW |     SKL+
    *              ------+-----------+-------------
    *                1x  |  8 x 4 sa |   8 x 4 sa
    *               MSAA |  8 x 4 px |   8 x 4 px
    *              ------+-----------+-------------
    *                2x  |  8 x 4 sa |  16 x 4 sa
    *               MSAA |  4 x 4 px |   8 x 4 px
    *              ------+-----------+-------------
    *                4x  |  8 x 4 sa |  16 x 8 sa
    *               MSAA |  4 x 2 px |   8 x 4 px
    *              ------+-----------+-------------
    *                8x  |  8 x 4 sa |  32 x 8 sa
    *               MSAA |  2 x 2 px |   8 x 4 px
    *              ------+-----------+-------------
    *               16x  |    N/A    | 32 x 16 sa
    *               MSAA |    N/A    |  8 x  4 px
    *              ------+-----------+-------------
    *
    * There are a number of different ways that this discrepency could be
    * handled.  The way we have chosen is to simply make MSAA HiZ have the
    * same number of samples as the parent surface pre-Sky Lake and always be
    * single-sampled on Sky Lake and above.  Since the block sizes of
    * compressed formats are given in samples, this neatly handles everything
    * without the need for additional HiZ formats with different block sizes
    * on SKL+.
    */
   const unsigned samples = ISL_DEV_GEN(dev) >= 9 ? 1 : surf->samples;

   return isl_surf_init(dev, hiz_surf,
                        .dim = surf->dim,
                        .format = ISL_FORMAT_HIZ,
                        .width = surf->logical_level0_px.width,
                        .height = surf->logical_level0_px.height,
                        .depth = surf->logical_level0_px.depth,
                        .levels = surf->levels,
                        .array_len = surf->logical_level0_px.array_len,
                        .samples = samples,
                        .usage = ISL_SURF_USAGE_HIZ_BIT,
                        .tiling_flags = ISL_TILING_HIZ_BIT);
}

bool
isl_surf_get_mcs_surf(const struct isl_device *dev,
                      const struct isl_surf *surf,
                      struct isl_surf *mcs_surf)
{
   assert(ISL_DEV_GEN(dev) >= 7);

   /* It must be multisampled with an array layout */
   assert(surf->samples > 1 && surf->msaa_layout == ISL_MSAA_LAYOUT_ARRAY);

   /* The following are true of all multisampled surfaces */
   assert(surf->dim == ISL_SURF_DIM_2D);
   assert(surf->levels == 1);
   assert(surf->logical_level0_px.depth == 1);

   /* The "Auxiliary Surface Pitch" field in RENDER_SURFACE_STATE is only 9
    * bits which means the maximum pitch of a compression surface is 512
    * tiles or 64KB (since MCS is always Y-tiled).  Since a 16x MCS buffer is
    * 64bpp, this gives us a maximum width of 8192 pixels.  We can create
    * larger multisampled surfaces, we just can't compress them.   For 2x, 4x,
    * and 8x, we have enough room for the full 16k supported by the hardware.
    */
   if (surf->samples == 16 && surf->logical_level0_px.width > 8192)
      return false;

   enum isl_format mcs_format;
   switch (surf->samples) {
   case 2:  mcs_format = ISL_FORMAT_MCS_2X;  break;
   case 4:  mcs_format = ISL_FORMAT_MCS_4X;  break;
   case 8:  mcs_format = ISL_FORMAT_MCS_8X;  break;
   case 16: mcs_format = ISL_FORMAT_MCS_16X; break;
   default:
      unreachable("Invalid sample count");
   }

   return isl_surf_init(dev, mcs_surf,
                        .dim = ISL_SURF_DIM_2D,
                        .format = mcs_format,
                        .width = surf->logical_level0_px.width,
                        .height = surf->logical_level0_px.height,
                        .depth = 1,
                        .levels = 1,
                        .array_len = surf->logical_level0_px.array_len,
                        .samples = 1, /* MCS surfaces are really single-sampled */
                        .usage = ISL_SURF_USAGE_MCS_BIT,
                        .tiling_flags = ISL_TILING_Y0_BIT);
}

bool
isl_surf_get_ccs_surf(const struct isl_device *dev,
                      const struct isl_surf *surf,
                      struct isl_surf *ccs_surf,
                      uint32_t row_pitch_B)
{
   assert(surf->samples == 1 && surf->msaa_layout == ISL_MSAA_LAYOUT_NONE);
   assert(ISL_DEV_GEN(dev) >= 7);

   if (surf->usage & ISL_SURF_USAGE_DISABLE_AUX_BIT)
      return false;

   /* The PRM doesn't say this explicitly, but fast-clears don't appear to
    * work for 3D textures until gen9 where the layout of 3D textures changes
    * to match 2D array textures.
    */
   if (ISL_DEV_GEN(dev) <= 8 && surf->dim != ISL_SURF_DIM_2D)
      return false;

   /* From the HSW PRM Volume 7: 3D-Media-GPGPU, page 652 (Color Clear of
    * Non-MultiSampler Render Target Restrictions):
    *
    *    "Support is for non-mip-mapped and non-array surface types only."
    *
    * This restriction is lifted on gen8+.  Technically, it may be possible to
    * create a CCS for an arrayed or mipmapped image and only enable CCS_D
    * when rendering to the base slice.  However, there is no documentation
    * tell us what the hardware would do in that case or what it does if you
    * walk off the bases slice.  (Does it ignore CCS or does it start
    * scribbling over random memory?)  We play it safe and just follow the
    * docs and don't allow CCS_D for arrayed or mip-mapped surfaces.
    */
   if (ISL_DEV_GEN(dev) <= 7 &&
       (surf->levels > 1 || surf->logical_level0_px.array_len > 1))
      return false;

   if (isl_format_is_compressed(surf->format))
      return false;

   /* TODO: More conditions where it can fail. */

   enum isl_format ccs_format;
   if (ISL_DEV_GEN(dev) >= 9) {
      if (!isl_tiling_is_any_y(surf->tiling))
         return false;

      switch (isl_format_get_layout(surf->format)->bpb) {
      case 32:    ccs_format = ISL_FORMAT_GEN9_CCS_32BPP;   break;
      case 64:    ccs_format = ISL_FORMAT_GEN9_CCS_64BPP;   break;
      case 128:   ccs_format = ISL_FORMAT_GEN9_CCS_128BPP;  break;
      default:
         return false;
      }
   } else if (surf->tiling == ISL_TILING_Y0) {
      switch (isl_format_get_layout(surf->format)->bpb) {
      case 32:    ccs_format = ISL_FORMAT_GEN7_CCS_32BPP_Y;    break;
      case 64:    ccs_format = ISL_FORMAT_GEN7_CCS_64BPP_Y;    break;
      case 128:   ccs_format = ISL_FORMAT_GEN7_CCS_128BPP_Y;   break;
      default:
         return false;
      }
   } else if (surf->tiling == ISL_TILING_X) {
      switch (isl_format_get_layout(surf->format)->bpb) {
      case 32:    ccs_format = ISL_FORMAT_GEN7_CCS_32BPP_X;    break;
      case 64:    ccs_format = ISL_FORMAT_GEN7_CCS_64BPP_X;    break;
      case 128:   ccs_format = ISL_FORMAT_GEN7_CCS_128BPP_X;   break;
      default:
         return false;
      }
   } else {
      return false;
   }

   return isl_surf_init(dev, ccs_surf,
                        .dim = surf->dim,
                        .format = ccs_format,
                        .width = surf->logical_level0_px.width,
                        .height = surf->logical_level0_px.height,
                        .depth = surf->logical_level0_px.depth,
                        .levels = surf->levels,
                        .array_len = surf->logical_level0_px.array_len,
                        .samples = 1,
                        .row_pitch_B = row_pitch_B,
                        .usage = ISL_SURF_USAGE_CCS_BIT,
                        .tiling_flags = ISL_TILING_CCS_BIT);
}

#define isl_genX_call(dev, func, ...)              \
   switch (ISL_DEV_GEN(dev)) {                     \
   case 4:                                         \
      /* G45 surface state is the same as gen5 */  \
      if (ISL_DEV_IS_G4X(dev)) {                   \
         isl_gen5_##func(__VA_ARGS__);             \
      } else {                                     \
         isl_gen4_##func(__VA_ARGS__);             \
      }                                            \
      break;                                       \
   case 5:                                         \
      isl_gen5_##func(__VA_ARGS__);                \
      break;                                       \
   case 6:                                         \
      isl_gen6_##func(__VA_ARGS__);                \
      break;                                       \
   case 7:                                         \
      if (ISL_DEV_IS_HASWELL(dev)) {               \
         isl_gen75_##func(__VA_ARGS__);            \
      } else {                                     \
         isl_gen7_##func(__VA_ARGS__);             \
      }                                            \
      break;                                       \
   case 8:                                         \
      isl_gen8_##func(__VA_ARGS__);                \
      break;                                       \
   case 9:                                         \
      isl_gen9_##func(__VA_ARGS__);                \
      break;                                       \
   case 10:                                        \
      isl_gen10_##func(__VA_ARGS__);               \
      break;                                       \
   case 11:                                        \
      isl_gen11_##func(__VA_ARGS__);               \
      break;                                       \
   default:                                        \
      assert(!"Unknown hardware generation");      \
   }

void
isl_surf_fill_state_s(const struct isl_device *dev, void *state,
                      const struct isl_surf_fill_state_info *restrict info)
{
#ifndef NDEBUG
   isl_surf_usage_flags_t _base_usage =
      info->view->usage & (ISL_SURF_USAGE_RENDER_TARGET_BIT |
                           ISL_SURF_USAGE_TEXTURE_BIT |
                           ISL_SURF_USAGE_STORAGE_BIT);
   /* They may only specify one of the above bits at a time */
   assert(__builtin_popcount(_base_usage) == 1);
   /* The only other allowed bit is ISL_SURF_USAGE_CUBE_BIT */
   assert((info->view->usage & ~ISL_SURF_USAGE_CUBE_BIT) == _base_usage);
#endif

   if (info->surf->dim == ISL_SURF_DIM_3D) {
      assert(info->view->base_array_layer + info->view->array_len <=
             info->surf->logical_level0_px.depth);
   } else {
      assert(info->view->base_array_layer + info->view->array_len <=
             info->surf->logical_level0_px.array_len);
   }

   isl_genX_call(dev, surf_fill_state_s, dev, state, info);
}

void
isl_buffer_fill_state_s(const struct isl_device *dev, void *state,
                        const struct isl_buffer_fill_state_info *restrict info)
{
   isl_genX_call(dev, buffer_fill_state_s, state, info);
}

void
isl_null_fill_state(const struct isl_device *dev, void *state,
                    struct isl_extent3d size)
{
   isl_genX_call(dev, null_fill_state, state, size);
}

void
isl_emit_depth_stencil_hiz_s(const struct isl_device *dev, void *batch,
                             const struct isl_depth_stencil_hiz_emit_info *restrict info)
{
   if (info->depth_surf && info->stencil_surf) {
      if (!dev->info->has_hiz_and_separate_stencil) {
         assert(info->depth_surf == info->stencil_surf);
         assert(info->depth_address == info->stencil_address);
      }
      assert(info->depth_surf->dim == info->stencil_surf->dim);
   }

   if (info->depth_surf) {
      assert((info->depth_surf->usage & ISL_SURF_USAGE_DEPTH_BIT));
      if (info->depth_surf->dim == ISL_SURF_DIM_3D) {
         assert(info->view->base_array_layer + info->view->array_len <=
                info->depth_surf->logical_level0_px.depth);
      } else {
         assert(info->view->base_array_layer + info->view->array_len <=
                info->depth_surf->logical_level0_px.array_len);
      }
   }

   if (info->stencil_surf) {
      assert((info->stencil_surf->usage & ISL_SURF_USAGE_STENCIL_BIT));
      if (info->stencil_surf->dim == ISL_SURF_DIM_3D) {
         assert(info->view->base_array_layer + info->view->array_len <=
                info->stencil_surf->logical_level0_px.depth);
      } else {
         assert(info->view->base_array_layer + info->view->array_len <=
                info->stencil_surf->logical_level0_px.array_len);
      }
   }

   isl_genX_call(dev, emit_depth_stencil_hiz_s, dev, batch, info);
}

/**
 * A variant of isl_surf_get_image_offset_sa() specific to
 * ISL_DIM_LAYOUT_GEN4_2D.
 */
static void
get_image_offset_sa_gen4_2d(const struct isl_surf *surf,
                            uint32_t level, uint32_t logical_array_layer,
                            uint32_t *x_offset_sa,
                            uint32_t *y_offset_sa)
{
   assert(level < surf->levels);
   if (surf->dim == ISL_SURF_DIM_3D)
      assert(logical_array_layer < surf->logical_level0_px.depth);
   else
      assert(logical_array_layer < surf->logical_level0_px.array_len);

   const struct isl_extent3d image_align_sa =
      isl_surf_get_image_alignment_sa(surf);

   const uint32_t W0 = surf->phys_level0_sa.width;
   const uint32_t H0 = surf->phys_level0_sa.height;

   const uint32_t phys_layer = logical_array_layer *
      (surf->msaa_layout == ISL_MSAA_LAYOUT_ARRAY ? surf->samples : 1);

   uint32_t x = 0;
   uint32_t y = phys_layer * isl_surf_get_array_pitch_sa_rows(surf);

   for (uint32_t l = 0; l < level; ++l) {
      if (l == 1) {
         uint32_t W = isl_minify(W0, l);
         x += isl_align_npot(W, image_align_sa.w);
      } else {
         uint32_t H = isl_minify(H0, l);
         y += isl_align_npot(H, image_align_sa.h);
      }
   }

   *x_offset_sa = x;
   *y_offset_sa = y;
}

/**
 * A variant of isl_surf_get_image_offset_sa() specific to
 * ISL_DIM_LAYOUT_GEN4_3D.
 */
static void
get_image_offset_sa_gen4_3d(const struct isl_surf *surf,
                            uint32_t level, uint32_t logical_z_offset_px,
                            uint32_t *x_offset_sa,
                            uint32_t *y_offset_sa)
{
   assert(level < surf->levels);
   if (surf->dim == ISL_SURF_DIM_3D) {
      assert(surf->phys_level0_sa.array_len == 1);
      assert(logical_z_offset_px < isl_minify(surf->phys_level0_sa.depth, level));
   } else {
      assert(surf->dim == ISL_SURF_DIM_2D);
      assert(surf->usage & ISL_SURF_USAGE_CUBE_BIT);
      assert(surf->phys_level0_sa.array_len == 6);
      assert(logical_z_offset_px < surf->phys_level0_sa.array_len);
   }

   const struct isl_extent3d image_align_sa =
      isl_surf_get_image_alignment_sa(surf);

   const uint32_t W0 = surf->phys_level0_sa.width;
   const uint32_t H0 = surf->phys_level0_sa.height;
   const uint32_t D0 = surf->phys_level0_sa.depth;
   const uint32_t AL = surf->phys_level0_sa.array_len;

   uint32_t x = 0;
   uint32_t y = 0;

   for (uint32_t l = 0; l < level; ++l) {
      const uint32_t level_h = isl_align_npot(isl_minify(H0, l), image_align_sa.h);
      const uint32_t level_d =
         isl_align_npot(surf->dim == ISL_SURF_DIM_3D ? isl_minify(D0, l) : AL,
                        image_align_sa.d);
      const uint32_t max_layers_vert = isl_align(level_d, 1u << l) / (1u << l);

      y += level_h * max_layers_vert;
   }

   const uint32_t level_w = isl_align_npot(isl_minify(W0, level), image_align_sa.w);
   const uint32_t level_h = isl_align_npot(isl_minify(H0, level), image_align_sa.h);
   const uint32_t level_d =
      isl_align_npot(surf->dim == ISL_SURF_DIM_3D ? isl_minify(D0, level) : AL,
                     image_align_sa.d);

   const uint32_t max_layers_horiz = MIN(level_d, 1u << level);

   x += level_w * (logical_z_offset_px % max_layers_horiz);
   y += level_h * (logical_z_offset_px / max_layers_horiz);

   *x_offset_sa = x;
   *y_offset_sa = y;
}

static void
get_image_offset_sa_gen6_stencil_hiz(const struct isl_surf *surf,
                                     uint32_t level,
                                     uint32_t logical_array_layer,
                                     uint32_t *x_offset_sa,
                                     uint32_t *y_offset_sa)
{
   assert(level < surf->levels);
   assert(surf->logical_level0_px.depth == 1);
   assert(logical_array_layer < surf->logical_level0_px.array_len);

   const struct isl_format_layout *fmtl = isl_format_get_layout(surf->format);

   const struct isl_extent3d image_align_sa =
      isl_surf_get_image_alignment_sa(surf);

   struct isl_tile_info tile_info;
   isl_tiling_get_info(surf->tiling, fmtl->bpb, &tile_info);
   const struct isl_extent2d tile_extent_sa = {
      .w = tile_info.logical_extent_el.w * fmtl->bw,
      .h = tile_info.logical_extent_el.h * fmtl->bh,
   };
   /* Tile size is a multiple of image alignment */
   assert(tile_extent_sa.w % image_align_sa.w == 0);
   assert(tile_extent_sa.h % image_align_sa.h == 0);

   const uint32_t W0 = surf->phys_level0_sa.w;
   const uint32_t H0 = surf->phys_level0_sa.h;

   /* Each image has the same height as LOD0 because the hardware thinks
    * everything is LOD0
    */
   const uint32_t H = isl_align(H0, image_align_sa.h);

   /* Quick sanity check for consistency */
   if (surf->phys_level0_sa.array_len > 1)
      assert(surf->array_pitch_el_rows == isl_assert_div(H, fmtl->bh));

   uint32_t x = 0, y = 0;
   for (uint32_t l = 0; l < level; ++l) {
      const uint32_t W = isl_minify(W0, l);

      const uint32_t w = isl_align(W, tile_extent_sa.w);
      const uint32_t h = isl_align(H * surf->phys_level0_sa.a,
                                   tile_extent_sa.h);

      if (l == 0) {
         y += h;
      } else {
         x += w;
      }
   }

   y += H * logical_array_layer;

   *x_offset_sa = x;
   *y_offset_sa = y;
}

/**
 * A variant of isl_surf_get_image_offset_sa() specific to
 * ISL_DIM_LAYOUT_GEN9_1D.
 */
static void
get_image_offset_sa_gen9_1d(const struct isl_surf *surf,
                            uint32_t level, uint32_t layer,
                            uint32_t *x_offset_sa,
                            uint32_t *y_offset_sa)
{
   assert(level < surf->levels);
   assert(layer < surf->phys_level0_sa.array_len);
   assert(surf->phys_level0_sa.height == 1);
   assert(surf->phys_level0_sa.depth == 1);
   assert(surf->samples == 1);

   const uint32_t W0 = surf->phys_level0_sa.width;
   const struct isl_extent3d image_align_sa =
      isl_surf_get_image_alignment_sa(surf);

   uint32_t x = 0;

   for (uint32_t l = 0; l < level; ++l) {
      uint32_t W = isl_minify(W0, l);
      uint32_t w = isl_align_npot(W, image_align_sa.w);

      x += w;
   }

   *x_offset_sa = x;
   *y_offset_sa = layer * isl_surf_get_array_pitch_sa_rows(surf);
}

/**
 * Calculate the offset, in units of surface samples, to a subimage in the
 * surface.
 *
 * @invariant level < surface levels
 * @invariant logical_array_layer < logical array length of surface
 * @invariant logical_z_offset_px < logical depth of surface at level
 */
void
isl_surf_get_image_offset_sa(const struct isl_surf *surf,
                             uint32_t level,
                             uint32_t logical_array_layer,
                             uint32_t logical_z_offset_px,
                             uint32_t *x_offset_sa,
                             uint32_t *y_offset_sa)
{
   assert(level < surf->levels);
   assert(logical_array_layer < surf->logical_level0_px.array_len);
   assert(logical_z_offset_px
          < isl_minify(surf->logical_level0_px.depth, level));

   switch (surf->dim_layout) {
   case ISL_DIM_LAYOUT_GEN9_1D:
      get_image_offset_sa_gen9_1d(surf, level, logical_array_layer,
                                  x_offset_sa, y_offset_sa);
      break;
   case ISL_DIM_LAYOUT_GEN4_2D:
      get_image_offset_sa_gen4_2d(surf, level, logical_array_layer
                                  + logical_z_offset_px,
                                  x_offset_sa, y_offset_sa);
      break;
   case ISL_DIM_LAYOUT_GEN4_3D:
      get_image_offset_sa_gen4_3d(surf, level, logical_array_layer +
                                  logical_z_offset_px,
                                  x_offset_sa, y_offset_sa);
      break;
   case ISL_DIM_LAYOUT_GEN6_STENCIL_HIZ:
      get_image_offset_sa_gen6_stencil_hiz(surf, level, logical_array_layer +
                                           logical_z_offset_px,
                                           x_offset_sa, y_offset_sa);
      break;

   default:
      unreachable("not reached");
   }
}

void
isl_surf_get_image_offset_el(const struct isl_surf *surf,
                             uint32_t level,
                             uint32_t logical_array_layer,
                             uint32_t logical_z_offset_px,
                             uint32_t *x_offset_el,
                             uint32_t *y_offset_el)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(surf->format);

   assert(level < surf->levels);
   assert(logical_array_layer < surf->logical_level0_px.array_len);
   assert(logical_z_offset_px
          < isl_minify(surf->logical_level0_px.depth, level));

   uint32_t x_offset_sa, y_offset_sa;
   isl_surf_get_image_offset_sa(surf, level,
                                logical_array_layer,
                                logical_z_offset_px,
                                &x_offset_sa,
                                &y_offset_sa);

   *x_offset_el = x_offset_sa / fmtl->bw;
   *y_offset_el = y_offset_sa / fmtl->bh;
}

void
isl_surf_get_image_offset_B_tile_sa(const struct isl_surf *surf,
                                    uint32_t level,
                                    uint32_t logical_array_layer,
                                    uint32_t logical_z_offset_px,
                                    uint32_t *offset_B,
                                    uint32_t *x_offset_sa,
                                    uint32_t *y_offset_sa)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(surf->format);

   uint32_t total_x_offset_el, total_y_offset_el;
   isl_surf_get_image_offset_el(surf, level, logical_array_layer,
                                logical_z_offset_px,
                                &total_x_offset_el,
                                &total_y_offset_el);

   uint32_t x_offset_el, y_offset_el;
   isl_tiling_get_intratile_offset_el(surf->tiling, fmtl->bpb,
                                      surf->row_pitch_B,
                                      total_x_offset_el,
                                      total_y_offset_el,
                                      offset_B,
                                      &x_offset_el,
                                      &y_offset_el);

   if (x_offset_sa) {
      *x_offset_sa = x_offset_el * fmtl->bw;
   } else {
      assert(x_offset_el == 0);
   }

   if (y_offset_sa) {
      *y_offset_sa = y_offset_el * fmtl->bh;
   } else {
      assert(y_offset_el == 0);
   }
}

void
isl_surf_get_image_surf(const struct isl_device *dev,
                        const struct isl_surf *surf,
                        uint32_t level,
                        uint32_t logical_array_layer,
                        uint32_t logical_z_offset_px,
                        struct isl_surf *image_surf,
                        uint32_t *offset_B,
                        uint32_t *x_offset_sa,
                        uint32_t *y_offset_sa)
{
   isl_surf_get_image_offset_B_tile_sa(surf,
                                       level,
                                       logical_array_layer,
                                       logical_z_offset_px,
                                       offset_B,
                                       x_offset_sa,
                                       y_offset_sa);

   /* Even for cube maps there will be only single face, therefore drop the
    * corresponding flag if present.
    */
   const isl_surf_usage_flags_t usage =
      surf->usage & (~ISL_SURF_USAGE_CUBE_BIT);

   bool ok UNUSED;
   ok = isl_surf_init(dev, image_surf,
                      .dim = ISL_SURF_DIM_2D,
                      .format = surf->format,
                      .width = isl_minify(surf->logical_level0_px.w, level),
                      .height = isl_minify(surf->logical_level0_px.h, level),
                      .depth = 1,
                      .levels = 1,
                      .array_len = 1,
                      .samples = surf->samples,
                      .row_pitch_B = surf->row_pitch_B,
                      .usage = usage,
                      .tiling_flags = (1 << surf->tiling));
   assert(ok);
}

void
isl_tiling_get_intratile_offset_el(enum isl_tiling tiling,
                                   uint32_t bpb,
                                   uint32_t row_pitch_B,
                                   uint32_t total_x_offset_el,
                                   uint32_t total_y_offset_el,
                                   uint32_t *base_address_offset,
                                   uint32_t *x_offset_el,
                                   uint32_t *y_offset_el)
{
   if (tiling == ISL_TILING_LINEAR) {
      assert(bpb % 8 == 0);
      *base_address_offset = total_y_offset_el * row_pitch_B +
                             total_x_offset_el * (bpb / 8);
      *x_offset_el = 0;
      *y_offset_el = 0;
      return;
   }

   struct isl_tile_info tile_info;
   isl_tiling_get_info(tiling, bpb, &tile_info);

   assert(row_pitch_B % tile_info.phys_extent_B.width == 0);

   /* For non-power-of-two formats, we need the address to be both tile and
    * element-aligned.  The easiest way to achieve this is to work with a tile
    * that is three times as wide as the regular tile.
    *
    * The tile info returned by get_tile_info has a logical size that is an
    * integer number of tile_info.format_bpb size elements.  To scale the
    * tile, we scale up the physical width and then treat the logical tile
    * size as if it has bpb size elements.
    */
   const uint32_t tile_el_scale = bpb / tile_info.format_bpb;
   tile_info.phys_extent_B.width *= tile_el_scale;

   /* Compute the offset into the tile */
   *x_offset_el = total_x_offset_el % tile_info.logical_extent_el.w;
   *y_offset_el = total_y_offset_el % tile_info.logical_extent_el.h;

   /* Compute the offset of the tile in units of whole tiles */
   uint32_t x_offset_tl = total_x_offset_el / tile_info.logical_extent_el.w;
   uint32_t y_offset_tl = total_y_offset_el / tile_info.logical_extent_el.h;

   *base_address_offset =
      y_offset_tl * tile_info.phys_extent_B.h * row_pitch_B +
      x_offset_tl * tile_info.phys_extent_B.h * tile_info.phys_extent_B.w;
}

uint32_t
isl_surf_get_depth_format(const struct isl_device *dev,
                          const struct isl_surf *surf)
{
   /* Support for separate stencil buffers began in gen5. Support for
    * interleaved depthstencil buffers ceased in gen7. The intermediate gens,
    * those that supported separate and interleaved stencil, were gen5 and
    * gen6.
    *
    * For a list of all available formats, see the Sandybridge PRM >> Volume
    * 2 Part 1: 3D/Media - 3D Pipeline >> 3DSTATE_DEPTH_BUFFER >> Surface
    * Format (p321).
    */

   bool has_stencil = surf->usage & ISL_SURF_USAGE_STENCIL_BIT;

   assert(surf->usage & ISL_SURF_USAGE_DEPTH_BIT);

   if (has_stencil)
      assert(ISL_DEV_GEN(dev) < 7);

   switch (surf->format) {
   default:
      unreachable("bad isl depth format");
   case ISL_FORMAT_R32_FLOAT_X8X24_TYPELESS:
      assert(ISL_DEV_GEN(dev) < 7);
      return 0; /* D32_FLOAT_S8X24_UINT */
   case ISL_FORMAT_R32_FLOAT:
      assert(!has_stencil);
      return 1; /* D32_FLOAT */
   case ISL_FORMAT_R24_UNORM_X8_TYPELESS:
      if (has_stencil) {
         assert(ISL_DEV_GEN(dev) < 7);
         return 2; /* D24_UNORM_S8_UINT */
      } else {
         assert(ISL_DEV_GEN(dev) >= 5);
         return 3; /* D24_UNORM_X8_UINT */
      }
   case ISL_FORMAT_R16_UNORM:
      assert(!has_stencil);
      return 5; /* D16_UNORM */
   }
}

bool
isl_swizzle_supports_rendering(const struct gen_device_info *devinfo,
                               struct isl_swizzle swizzle)
{
   if (devinfo->is_haswell) {
      /* From the Haswell PRM,
       * RENDER_SURFACE_STATE::Shader Channel Select Red
       *
       *    "The Shader channel selects also define which shader channels are
       *    written to which surface channel. If the Shader channel select is
       *    SCS_ZERO or SCS_ONE then it is not written to the surface. If the
       *    shader channel select is SCS_RED it is written to the surface red
       *    channel and so on. If more than one shader channel select is set
       *    to the same surface channel only the first shader channel in RGBA
       *    order will be written."
       */
      return true;
   } else if (devinfo->gen <= 7) {
      /* Ivy Bridge and early doesn't have any swizzling */
      return isl_swizzle_is_identity(swizzle);
   } else {
      /* From the Sky Lake PRM Vol. 2d,
       * RENDER_SURFACE_STATE::Shader Channel Select Red
       *
       *    "For Render Target, Red, Green and Blue Shader Channel Selects
       *    MUST be such that only valid components can be swapped i.e. only
       *    change the order of components in the pixel. Any other values for
       *    these Shader Channel Select fields are not valid for Render
       *    Targets. This also means that there MUST not be multiple shader
       *    channels mapped to the same RT channel."
       *
       * From the Sky Lake PRM Vol. 2d,
       * RENDER_SURFACE_STATE::Shader Channel Select Alpha
       *
       *    "For Render Target, this field MUST be programmed to
       *    value = SCS_ALPHA."
       */
      return (swizzle.r == ISL_CHANNEL_SELECT_RED ||
              swizzle.r == ISL_CHANNEL_SELECT_GREEN ||
              swizzle.r == ISL_CHANNEL_SELECT_BLUE) &&
             (swizzle.g == ISL_CHANNEL_SELECT_RED ||
              swizzle.g == ISL_CHANNEL_SELECT_GREEN ||
              swizzle.g == ISL_CHANNEL_SELECT_BLUE) &&
             (swizzle.b == ISL_CHANNEL_SELECT_RED ||
              swizzle.b == ISL_CHANNEL_SELECT_GREEN ||
              swizzle.b == ISL_CHANNEL_SELECT_BLUE) &&
             swizzle.r != swizzle.g &&
             swizzle.r != swizzle.b &&
             swizzle.g != swizzle.b &&
             swizzle.a == ISL_CHANNEL_SELECT_ALPHA;
   }
}

static enum isl_channel_select
swizzle_select(enum isl_channel_select chan, struct isl_swizzle swizzle)
{
   switch (chan) {
   case ISL_CHANNEL_SELECT_ZERO:
   case ISL_CHANNEL_SELECT_ONE:
      return chan;
   case ISL_CHANNEL_SELECT_RED:
      return swizzle.r;
   case ISL_CHANNEL_SELECT_GREEN:
      return swizzle.g;
   case ISL_CHANNEL_SELECT_BLUE:
      return swizzle.b;
   case ISL_CHANNEL_SELECT_ALPHA:
      return swizzle.a;
   default:
      unreachable("Invalid swizzle component");
   }
}

/**
 * Returns the single swizzle that is equivalent to applying the two given
 * swizzles in sequence.
 */
struct isl_swizzle
isl_swizzle_compose(struct isl_swizzle first, struct isl_swizzle second)
{
   return (struct isl_swizzle) {
      .r = swizzle_select(first.r, second),
      .g = swizzle_select(first.g, second),
      .b = swizzle_select(first.b, second),
      .a = swizzle_select(first.a, second),
   };
}

/**
 * Returns a swizzle that is the pseudo-inverse of this swizzle.
 */
struct isl_swizzle
isl_swizzle_invert(struct isl_swizzle swizzle)
{
   /* Default to zero for channels which do not show up in the swizzle */
   enum isl_channel_select chans[4] = {
      ISL_CHANNEL_SELECT_ZERO,
      ISL_CHANNEL_SELECT_ZERO,
      ISL_CHANNEL_SELECT_ZERO,
      ISL_CHANNEL_SELECT_ZERO,
   };

   /* We go in ABGR order so that, if there are any duplicates, the first one
    * is taken if you look at it in RGBA order.  This is what Haswell hardware
    * does for render target swizzles.
    */
   if ((unsigned)(swizzle.a - ISL_CHANNEL_SELECT_RED) < 4)
      chans[swizzle.a - ISL_CHANNEL_SELECT_RED] = ISL_CHANNEL_SELECT_ALPHA;
   if ((unsigned)(swizzle.b - ISL_CHANNEL_SELECT_RED) < 4)
      chans[swizzle.b - ISL_CHANNEL_SELECT_RED] = ISL_CHANNEL_SELECT_BLUE;
   if ((unsigned)(swizzle.g - ISL_CHANNEL_SELECT_RED) < 4)
      chans[swizzle.g - ISL_CHANNEL_SELECT_RED] = ISL_CHANNEL_SELECT_GREEN;
   if ((unsigned)(swizzle.r - ISL_CHANNEL_SELECT_RED) < 4)
      chans[swizzle.r - ISL_CHANNEL_SELECT_RED] = ISL_CHANNEL_SELECT_RED;

   return (struct isl_swizzle) { chans[0], chans[1], chans[2], chans[3] };
}
