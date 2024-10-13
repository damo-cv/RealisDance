/*
 * Mesa 3-D graphics library
 *
 * Copyright 2012 Intel Corporation
 * Copyright 2013 Google
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Chad Versace <chad.versace@linux.intel.com>
 *    Frank Henigman <fjhenigman@google.com>
 */

#ifndef INTEL_TILED_MEMCPY_H
#define INTEL_TILED_MEMCPY_H

#include <stdint.h>
#include "main/mtypes.h"

typedef enum {
  INTEL_COPY_MEMCPY = 0,
  INTEL_COPY_RGBA8,
  INTEL_COPY_STREAMING_LOAD,
  INTEL_COPY_INVALID,
} mem_copy_fn_type;

typedef void *(*mem_copy_fn)(void *dest, const void *src, size_t n);

typedef void (*tiled_to_linear_fn)
   (uint32_t xt1, uint32_t xt2,
    uint32_t yt1, uint32_t yt2,
    char *dst, const char *src,
    int32_t dst_pitch, uint32_t src_pitch,
    bool has_swizzling,
    enum isl_tiling tiling,
    mem_copy_fn_type copy_type);

void
linear_to_tiled(uint32_t xt1, uint32_t xt2,
                uint32_t yt1, uint32_t yt2,
                char *dst, const char *src,
                uint32_t dst_pitch, int32_t src_pitch,
                bool has_swizzling,
                enum isl_tiling tiling,
                mem_copy_fn_type copy_type);

void
tiled_to_linear(uint32_t xt1, uint32_t xt2,
                uint32_t yt1, uint32_t yt2,
                char *dst, const char *src,
                int32_t dst_pitch, uint32_t src_pitch,
                bool has_swizzling,
                enum isl_tiling tiling,
                mem_copy_fn_type copy_type);

/**
 * Determine which copy function to use for the given format combination
 *
 * The only two possible copy functions which are ever returned are a
 * direct memcpy and a RGBA <-> BGRA copy function.  Since RGBA -> BGRA and
 * BGRA -> RGBA are exactly the same operation (and memcpy is obviously
 * symmetric), it doesn't matter whether the copy is from the tiled image
 * to the untiled or vice versa.  The copy function required is the same in
 * either case so this function can be used.
 *
 * \param[in]  tiledFormat The format of the tiled image
 * \param[in]  format      The GL format of the client data
 * \param[in]  type        The GL type of the client data
 * \param[out] mem_copy    Will be set to one of either the standard
 *                         library's memcpy or a different copy function
 *                         that performs an RGBA to BGRA conversion
 * \param[out] cpp         Number of bytes per channel
 *
 * \return true if the format and type combination are valid
 */
static MAYBE_UNUSED bool
intel_get_memcpy_type(mesa_format tiledFormat, GLenum format, GLenum type,
                      mem_copy_fn_type *copy_type, uint32_t *cpp)
{
   *copy_type = INTEL_COPY_INVALID;

   if (type == GL_UNSIGNED_INT_8_8_8_8_REV &&
       !(format == GL_RGBA || format == GL_BGRA))
      return false; /* Invalid type/format combination */

   if ((tiledFormat == MESA_FORMAT_L_UNORM8 && format == GL_LUMINANCE) ||
       (tiledFormat == MESA_FORMAT_A_UNORM8 && format == GL_ALPHA)) {
      *cpp = 1;
      *copy_type = INTEL_COPY_MEMCPY;
   } else if ((tiledFormat == MESA_FORMAT_B8G8R8A8_UNORM) ||
              (tiledFormat == MESA_FORMAT_B8G8R8X8_UNORM) ||
              (tiledFormat == MESA_FORMAT_B8G8R8A8_SRGB) ||
              (tiledFormat == MESA_FORMAT_B8G8R8X8_SRGB)) {
      *cpp = 4;
      if (format == GL_BGRA) {
         *copy_type = INTEL_COPY_MEMCPY;
      } else if (format == GL_RGBA) {
         *copy_type = INTEL_COPY_RGBA8;
      }
   } else if ((tiledFormat == MESA_FORMAT_R8G8B8A8_UNORM) ||
              (tiledFormat == MESA_FORMAT_R8G8B8X8_UNORM) ||
              (tiledFormat == MESA_FORMAT_R8G8B8A8_SRGB) ||
              (tiledFormat == MESA_FORMAT_R8G8B8X8_SRGB)) {
      *cpp = 4;
      if (format == GL_BGRA) {
         /* Copying from RGBA to BGRA is the same as BGRA to RGBA so we can
          * use the same function.
          */
         *copy_type = INTEL_COPY_RGBA8;
      } else if (format == GL_RGBA) {
         *copy_type = INTEL_COPY_MEMCPY;
      }
   }

   if (*copy_type == INTEL_COPY_INVALID)
      return false;

   return true;
}

#endif /* INTEL_TILED_MEMCPY */
