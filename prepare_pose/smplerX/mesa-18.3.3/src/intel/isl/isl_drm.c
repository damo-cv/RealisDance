/*
 * Copyright 2017 Intel Corporation
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
#include <stdlib.h>

#include <drm_fourcc.h>
#include <i915_drm.h>

#include "isl.h"
#include "dev/gen_device_info.h"

uint32_t
isl_tiling_to_i915_tiling(enum isl_tiling tiling)
{
   switch (tiling) {
   case ISL_TILING_LINEAR:
      return I915_TILING_NONE;

   case ISL_TILING_X:
      return I915_TILING_X;

   case ISL_TILING_Y0:
      return I915_TILING_Y;

   case ISL_TILING_W:
   case ISL_TILING_Yf:
   case ISL_TILING_Ys:
   case ISL_TILING_HIZ:
   case ISL_TILING_CCS:
      return I915_TILING_NONE;
   }

   unreachable("Invalid ISL tiling");
}

enum isl_tiling
isl_tiling_from_i915_tiling(uint32_t tiling)
{
   switch (tiling) {
   case I915_TILING_NONE:
      return ISL_TILING_LINEAR;

   case I915_TILING_X:
      return ISL_TILING_X;

   case I915_TILING_Y:
      return ISL_TILING_Y0;
   }

   unreachable("Invalid i915 tiling");
}

static const struct isl_drm_modifier_info modifier_info[] = {
   {
      .modifier = DRM_FORMAT_MOD_NONE,
      .name = "DRM_FORMAT_MOD_NONE",
      .tiling = ISL_TILING_LINEAR,
   },
   {
      .modifier = I915_FORMAT_MOD_X_TILED,
      .name = "I915_FORMAT_MOD_X_TILED",
      .tiling = ISL_TILING_X,
   },
   {
      .modifier = I915_FORMAT_MOD_Y_TILED,
      .name = "I915_FORMAT_MOD_Y_TILED",
      .tiling = ISL_TILING_Y0,
   },
   {
      .modifier = I915_FORMAT_MOD_Y_TILED_CCS,
      .name = "I915_FORMAT_MOD_Y_TILED_CCS",
      .tiling = ISL_TILING_Y0,
      .aux_usage = ISL_AUX_USAGE_CCS_E,
      .supports_clear_color = false,
   },
};

const struct isl_drm_modifier_info *
isl_drm_modifier_get_info(uint64_t modifier)
{
   for (unsigned i = 0; i < ARRAY_SIZE(modifier_info); i++) {
      if (modifier_info[i].modifier == modifier)
         return &modifier_info[i];
   }

   return NULL;
}
