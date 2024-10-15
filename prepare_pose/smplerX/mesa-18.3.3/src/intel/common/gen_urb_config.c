/*
 * Copyright (c) 2011 Intel Corporation
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

#include <stdlib.h>
#include <math.h>

#include "util/macros.h"
#include "main/macros.h"
#include "compiler/shader_enums.h"

#include "gen_l3_config.h"

/**
 * The following diagram shows how we partition the URB:
 *
 *        16kb or 32kb               Rest of the URB space
 *   __________-__________   _________________-_________________
 *  /                     \ /                                   \
 * +-------------------------------------------------------------+
 * |  VS/HS/DS/GS/FS Push  |           VS/HS/DS/GS URB           |
 * |       Constants       |               Entries               |
 * +-------------------------------------------------------------+
 *
 * Push constants must be stored at the beginning of the URB space,
 * while URB entries can be stored anywhere.  We choose to lay them
 * out in pipeline order (VS -> HS -> DS -> GS).
 */

/**
 * Decide how to partition the URB among the various stages.
 *
 * \param[in] push_constant_bytes - space allocate for push constants.
 * \param[in] urb_size_bytes - total size of the URB (from L3 config).
 * \param[in] tess_present - are tessellation shaders active?
 * \param[in] gs_present - are geometry shaders active?
 * \param[in] entry_size - the URB entry size (from the shader compiler)
 * \param[out] entries - the number of URB entries for each stage
 * \param[out] start - the starting offset for each stage
 */
void
gen_get_urb_config(const struct gen_device_info *devinfo,
                   unsigned push_constant_bytes, unsigned urb_size_bytes,
                   bool tess_present, bool gs_present,
                   const unsigned entry_size[4],
                   unsigned entries[4], unsigned start[4])
{
   const bool active[4] = { true, tess_present, tess_present, gs_present };

   /* URB allocations must be done in 8k chunks. */
   const unsigned chunk_size_bytes = 8192;

   const unsigned push_constant_chunks =
      push_constant_bytes / chunk_size_bytes;
   const unsigned urb_chunks = urb_size_bytes / chunk_size_bytes;

   /* From p35 of the Ivy Bridge PRM (section 1.7.1: 3DSTATE_URB_GS):
    *
    *     VS Number of URB Entries must be divisible by 8 if the VS URB Entry
    *     Allocation Size is less than 9 512-bit URB entries.
    *
    * Similar text exists for HS, DS and GS.
    */
   unsigned granularity[4];
   for (int i = MESA_SHADER_VERTEX; i <= MESA_SHADER_GEOMETRY; i++) {
      granularity[i] = (entry_size[i] < 9) ? 8 : 1;
   }

   unsigned min_entries[4] = {
      /* VS has a lower limit on the number of URB entries.
       *
       * From the Broadwell PRM, 3DSTATE_URB_VS instruction:
       * "When tessellation is enabled, the VS Number of URB Entries must be
       *  greater than or equal to 192."
       */
      [MESA_SHADER_VERTEX] = tess_present && devinfo->gen == 8 ?
         192 : devinfo->urb.min_entries[MESA_SHADER_VERTEX],

      /* There are two constraints on the minimum amount of URB space we can
       * allocate:
       *
       * (1) We need room for at least 2 URB entries, since we always operate
       * the GS in DUAL_OBJECT mode.
       *
       * (2) We can't allocate less than nr_gs_entries_granularity.
       */
      [MESA_SHADER_GEOMETRY] = gs_present ? 2 : 0,

      [MESA_SHADER_TESS_CTRL] = tess_present ? 1 : 0,

      [MESA_SHADER_TESS_EVAL] = tess_present ?
         devinfo->urb.min_entries[MESA_SHADER_TESS_EVAL] : 0,
   };

   /* Min VS Entries isn't a multiple of 8 on Cherryview/Broxton; round up.
    * Round them all up.
    */
   for (int i = MESA_SHADER_VERTEX; i <= MESA_SHADER_GEOMETRY; i++) {
      min_entries[i] = ALIGN(min_entries[i], granularity[i]);
   }

   unsigned entry_size_bytes[4];
   for (int i = MESA_SHADER_VERTEX; i <= MESA_SHADER_GEOMETRY; i++) {
      entry_size_bytes[i] = 64 * entry_size[i];
   }

   /* Initially, assign each stage the minimum amount of URB space it needs,
    * and make a note of how much additional space it "wants" (the amount of
    * additional space it could actually make use of).
    */
   unsigned chunks[4];
   unsigned wants[4];
   unsigned total_needs = push_constant_chunks;
   unsigned total_wants = 0;

   for (int i = MESA_SHADER_VERTEX; i <= MESA_SHADER_GEOMETRY; i++) {
      if (active[i]) {
         chunks[i] = DIV_ROUND_UP(min_entries[i] * entry_size_bytes[i],
                                  chunk_size_bytes);

         wants[i] =
            DIV_ROUND_UP(devinfo->urb.max_entries[i] * entry_size_bytes[i],
                         chunk_size_bytes) - chunks[i];
      } else {
         chunks[i] = 0;
         wants[i] = 0;
      }

      total_needs += chunks[i];
      total_wants += wants[i];
   }

   assert(total_needs <= urb_chunks);

   /* Mete out remaining space (if any) in proportion to "wants". */
   unsigned remaining_space = MIN2(urb_chunks - total_needs, total_wants);

   if (remaining_space > 0) {
      for (int i = MESA_SHADER_VERTEX;
           total_wants > 0 && i <= MESA_SHADER_TESS_EVAL; i++) {
         unsigned additional = (unsigned)
            roundf(wants[i] * (((float) remaining_space) / total_wants));
         chunks[i] += additional;
         remaining_space -= additional;
         total_wants -= wants[i];
      }

      chunks[MESA_SHADER_GEOMETRY] += remaining_space;
   }

   /* Sanity check that we haven't over-allocated. */
   unsigned total_chunks = push_constant_chunks;
   for (int i = MESA_SHADER_VERTEX; i <= MESA_SHADER_GEOMETRY; i++) {
      total_chunks += chunks[i];
   }
   assert(total_chunks <= urb_chunks);

   /* Finally, compute the number of entries that can fit in the space
    * allocated to each stage.
    */
   for (int i = MESA_SHADER_VERTEX; i <= MESA_SHADER_GEOMETRY; i++) {
      entries[i] = chunks[i] * chunk_size_bytes / entry_size_bytes[i];

      /* Since we rounded up when computing wants[], this may be slightly
       * more than the maximum allowed amount, so correct for that.
       */
      entries[i] = MIN2(entries[i], devinfo->urb.max_entries[i]);

      /* Ensure that we program a multiple of the granularity. */
      entries[i] = ROUND_DOWN_TO(entries[i], granularity[i]);

      /* Finally, sanity check to make sure we have at least the minimum
       * number of entries needed for each stage.
       */
      assert(entries[i] >= min_entries[i]);
   }

   /* Lay out the URB in pipeline order: push constants, VS, HS, DS, GS. */
   start[0] = push_constant_chunks;
   for (int i = MESA_SHADER_TESS_CTRL; i <= MESA_SHADER_GEOMETRY; i++) {
      start[i] = start[i - 1] + chunks[i - 1];
   }
}
