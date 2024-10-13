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

#include "isl_gen4.h"
#include "isl_priv.h"

bool
isl_gen4_choose_msaa_layout(const struct isl_device *dev,
                            const struct isl_surf_init_info *info,
                            enum isl_tiling tiling,
                            enum isl_msaa_layout *msaa_layout)
{
   /* Gen4 and Gen5 do not support MSAA */
   assert(info->samples >= 1);

   *msaa_layout = ISL_MSAA_LAYOUT_NONE;
   return true;
}

void
isl_gen4_filter_tiling(const struct isl_device *dev,
                       const struct isl_surf_init_info *restrict info,
                       isl_tiling_flags_t *flags)
{
   /* Gen4-5 only support linear, X, and Y-tiling. */
   *flags &= (ISL_TILING_LINEAR_BIT | ISL_TILING_X_BIT | ISL_TILING_Y0_BIT);

   if (isl_surf_usage_is_depth_or_stencil(info->usage)) {
      assert(!ISL_DEV_USE_SEPARATE_STENCIL(dev));

      /* From the g35 PRM Vol. 2, 3DSTATE_DEPTH_BUFFER::Tile Walk:
       *
       *    "The Depth Buffer, if tiled, must use Y-Major tiling"
       *
       *    Errata   Description    Project
       *    BWT014   The Depth Buffer Must be Tiled, it cannot be linear. This
       *    field must be set to 1 on DevBW-A.  [DevBW -A,B]
       *
       * In testing, the linear configuration doesn't seem to work on gen4.
       */
      *flags &= (ISL_DEV_GEN(dev) == 4 && !ISL_DEV_IS_G4X(dev)) ?
                ISL_TILING_Y0_BIT : (ISL_TILING_Y0_BIT | ISL_TILING_LINEAR_BIT);
   }

   if (info->usage & (ISL_SURF_USAGE_DISPLAY_ROTATE_90_BIT |
                      ISL_SURF_USAGE_DISPLAY_ROTATE_180_BIT |
                      ISL_SURF_USAGE_DISPLAY_ROTATE_270_BIT)) {
      assert(*flags & ISL_SURF_USAGE_DISPLAY_BIT);
      isl_finishme("%s:%s: handle rotated display surfaces",
                   __FILE__, __func__);
   }

   if (info->usage & (ISL_SURF_USAGE_DISPLAY_FLIP_X_BIT |
                      ISL_SURF_USAGE_DISPLAY_FLIP_Y_BIT)) {
      assert(*flags & ISL_SURF_USAGE_DISPLAY_BIT);
      isl_finishme("%s:%s: handle flipped display surfaces",
                   __FILE__, __func__);
   }

   if (info->usage & ISL_SURF_USAGE_DISPLAY_BIT) {
      /* Before Skylake, the display engine does not accept Y */
      *flags &= (ISL_TILING_LINEAR_BIT | ISL_TILING_X_BIT);
   }

   assert(info->samples == 1);

   /* From the g35 PRM, Volume 1, 11.5.5, "Per-Stream Tile Format Support":
    *
    *    "NOTE: 128BPE Format Color buffer ( render target ) MUST be either
    *    TileX or Linear."
    *
    * This is required all the way up to Sandy Bridge.
    */
   if (isl_format_get_layout(info->format)->bpb >= 128)
      *flags &= ~ISL_TILING_Y0_BIT;
}

void
isl_gen4_choose_image_alignment_el(const struct isl_device *dev,
                                   const struct isl_surf_init_info *restrict info,
                                   enum isl_tiling tiling,
                                   enum isl_dim_layout dim_layout,
                                   enum isl_msaa_layout msaa_layout,
                                   struct isl_extent3d *image_align_el)
{
   assert(info->samples == 1);
   assert(msaa_layout == ISL_MSAA_LAYOUT_NONE);
   assert(!isl_tiling_is_std_y(tiling));

   /* Note that neither the surface's horizontal nor vertical image alignment
    * is programmable on gen4 nor gen5.
    *
    * From the G35 PRM (2008-01), Volume 1 Graphics Core, Section 6.17.3.4
    * Alignment Unit Size:
    *
    *    Note that the compressed formats are padded to a full compression
    *    cell.
    *
    *    +------------------------+--------+--------+
    *    | format                 | halign | valign |
    *    +------------------------+--------+--------+
    *    | YUV 4:2:2 formats      |      4 |      2 |
    *    | uncompressed formats   |      4 |      2 |
    *    +------------------------+--------+--------+
    */

   if (isl_format_is_compressed(info->format)) {
      *image_align_el = isl_extent3d(1, 1, 1);
      return;
   }

   *image_align_el = isl_extent3d(4, 2, 1);
}
