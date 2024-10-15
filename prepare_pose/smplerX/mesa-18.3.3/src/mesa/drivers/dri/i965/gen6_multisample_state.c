/*
 * Copyright © 2012 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "intel_batchbuffer.h"

#include "brw_context.h"
#include "brw_defines.h"
#include "brw_multisample_state.h"
#include "main/framebuffer.h"

void
gen6_get_sample_position(struct gl_context *ctx,
                         struct gl_framebuffer *fb,
                         GLuint index, GLfloat *result)
{
   uint8_t bits;

   switch (_mesa_geometric_samples(fb)) {
   case 1:
      result[0] = result[1] = 0.5f;
      return;
   case 2:
      bits = brw_multisample_positions_1x_2x >> (8 * index);
      break;
   case 4:
      bits = brw_multisample_positions_4x >> (8 * index);
      break;
   case 8:
      bits = brw_multisample_positions_8x[index >> 2] >> (8 * (index & 3));
      break;
   case 16:
      bits = brw_multisample_positions_16x[index >> 2] >> (8 * (index & 3));
      break;
   default:
      unreachable("Not implemented");
   }

   /* Convert from U0.4 back to a floating point coordinate. */
   result[0] = ((bits >> 4) & 0xf) / 16.0f;
   result[1] = (bits & 0xf) / 16.0f;
}

/**
 * Sample index layout shows the numbering of slots in a rectangular
 * grid of samples with in a pixel. Sample number layout shows the
 * rectangular grid of samples roughly corresponding to the real sample
 * locations with in a pixel. Sample number layout matches the sample
 * index layout in case of 2X and 4x MSAA, but they are different in
 * case of 8X MSAA.
 *
 * 8X MSAA sample index layout    8x MSAA sample number layout
 *           ---------                      ---------
 *           | 0 | 1 |                      | 1 | 0 |
 *           ---------                      ---------
 *
 * 4X MSAA sample index / number layout
 *           ---------
 *           | 0 | 1 |
 *           ---------
 *           | 2 | 3 |
 *           ---------
 *
 * 8X MSAA sample index layout    8x MSAA sample number layout
 *           ---------                      ---------
 *           | 0 | 1 |                      | 5 | 2 |
 *           ---------                      ---------
 *           | 2 | 3 |                      | 4 | 6 |
 *           ---------                      ---------
 *           | 4 | 5 |                      | 0 | 3 |
 *           ---------                      ---------
 *           | 6 | 7 |                      | 7 | 1 |
 *           ---------                      ---------
 *
 * 16X MSAA sample index layout  16x MSAA sample number layout
 *         -----------------            -----------------
 *         | 0 | 1 | 2 | 3 |            |15 |10 | 9 | 7 |
 *         -----------------            -----------------
 *         | 4 | 5 | 6 | 7 |            | 4 | 1 | 3 |13 |
 *         -----------------            -----------------
 *         | 8 | 9 |10 |11 |            |12 | 2 | 0 | 6 |
 *         -----------------            -----------------
 *         |12 |13 |14 |15 |            |11 | 8 | 5 |14 |
 *         -----------------            -----------------
 *
 * A sample map is used to map sample indices to sample numbers.
 */
void
gen6_set_sample_maps(struct gl_context *ctx)
{
   uint8_t map_2x[2] = {1, 0};
   uint8_t map_4x[4] = {0, 1, 2, 3};
   uint8_t map_8x[8] = {3, 7, 5, 0, 1, 2, 4, 6};
   uint8_t map_16x[16] = { 15, 10, 9, 7, 4, 1, 3, 13,
                           12, 2, 0, 6, 11, 8, 5, 14 };

   memcpy(ctx->Const.SampleMap2x, map_2x, sizeof(map_2x));
   memcpy(ctx->Const.SampleMap4x, map_4x, sizeof(map_4x));
   memcpy(ctx->Const.SampleMap8x, map_8x, sizeof(map_8x));
   memcpy(ctx->Const.SampleMap16x, map_16x, sizeof(map_16x));
}
