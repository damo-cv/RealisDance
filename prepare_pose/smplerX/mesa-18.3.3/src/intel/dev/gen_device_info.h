 /*
  * Copyright © 2013 Intel Corporation
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
  *
  */

#ifndef GEN_DEVICE_INFO_H
#define GEN_DEVICE_INFO_H

#include <stdbool.h>
#include <stdint.h>

#include "util/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

struct drm_i915_query_topology_info;

#define GEN_DEVICE_MAX_SLICES           (6)  /* Maximum on gen10 */
#define GEN_DEVICE_MAX_SUBSLICES        (8)  /* Maximum on gen11 */
#define GEN_DEVICE_MAX_EUS_PER_SUBSLICE (10) /* Maximum on Haswell */

/**
 * Intel hardware information and quirks
 */
struct gen_device_info
{
   int gen; /**< Generation number: 4, 5, 6, 7, ... */
   int gt;

   bool is_g4x;
   bool is_ivybridge;
   bool is_baytrail;
   bool is_haswell;
   bool is_broadwell;
   bool is_cherryview;
   bool is_skylake;
   bool is_broxton;
   bool is_kabylake;
   bool is_geminilake;
   bool is_coffeelake;
   bool is_cannonlake;

   bool has_hiz_and_separate_stencil;
   bool must_use_separate_stencil;
   bool has_sample_with_hiz;
   bool has_llc;

   bool has_pln;
   bool has_64bit_types;
   bool has_integer_dword_mul;
   bool has_compr4;
   bool has_surface_tile_offset;
   bool supports_simd16_3src;
   bool has_resource_streamer;

   /**
    * \name Intel hardware quirks
    *  @{
    */
   bool has_negative_rhw_bug;

   /**
    * Some versions of Gen hardware don't do centroid interpolation correctly
    * on unlit pixels, causing incorrect values for derivatives near triangle
    * edges.  Enabling this flag causes the fragment shader to use
    * non-centroid interpolation for unlit pixels, at the expense of two extra
    * fragment shader instructions.
    */
   bool needs_unlit_centroid_workaround;
   /** @} */

   /**
    * \name GPU hardware limits
    *
    * In general, you can find shader thread maximums by looking at the "Maximum
    * Number of Threads" field in the Intel PRM description of the 3DSTATE_VS,
    * 3DSTATE_GS, 3DSTATE_HS, 3DSTATE_DS, and 3DSTATE_PS commands. URB entry
    * limits come from the "Number of URB Entries" field in the
    * 3DSTATE_URB_VS command and friends.
    *
    * These fields are used to calculate the scratch space to allocate.  The
    * amount of scratch space can be larger without being harmful on modern
    * GPUs, however, prior to Haswell, programming the maximum number of threads
    * to greater than the hardware maximum would cause GPU performance to tank.
    *
    *  @{
    */
   /**
    * Total number of slices present on the device whether or not they've been
    * fused off.
    *
    * XXX: CS thread counts are limited by the inability to do cross subslice
    * communication. It is the effectively the number of logical threads which
    * can be executed in a subslice. Fuse configurations may cause this number
    * to change, so we program @max_cs_threads as the lower maximum.
    */
   unsigned num_slices;

   /**
    * Number of subslices for each slice (used to be uniform until CNL).
    */
   unsigned num_subslices[GEN_DEVICE_MAX_SUBSLICES];

   /**
    * Upper bound of number of EU per subslice (some SKUs might have just 1 EU
    * fused across all subslices, like 47 EUs, in which case this number won't
    * be acurate for one subslice).
    */
   unsigned num_eu_per_subslice;

   /**
    * Number of threads per eu, varies between 4 and 8 between generations.
    */
   unsigned num_thread_per_eu;

   /**
    * A bit mask of the slices available.
    */
   uint8_t slice_masks;

   /**
    * An array of bit mask of the subslices available, use subslice_slice_stride
    * to access this array.
    */
   uint8_t subslice_masks[GEN_DEVICE_MAX_SLICES *
                          DIV_ROUND_UP(GEN_DEVICE_MAX_SUBSLICES, 8)];

   /**
    * An array of bit mask of EUs available, use eu_slice_stride &
    * eu_subslice_stride to access this array.
    */
   uint8_t eu_masks[GEN_DEVICE_MAX_SLICES *
                    GEN_DEVICE_MAX_SUBSLICES *
                    DIV_ROUND_UP(GEN_DEVICE_MAX_EUS_PER_SUBSLICE, 8)];

   /**
    * Stride to access subslice_masks[].
    */
   uint16_t subslice_slice_stride;

   /**
    * Strides to access eu_masks[].
    */
   uint16_t eu_slice_stride;
   uint16_t eu_subslice_stride;

   unsigned l3_banks;
   unsigned max_vs_threads;   /**< Maximum Vertex Shader threads */
   unsigned max_tcs_threads;  /**< Maximum Hull Shader threads */
   unsigned max_tes_threads;  /**< Maximum Domain Shader threads */
   unsigned max_gs_threads;   /**< Maximum Geometry Shader threads. */
   /**
    * Theoretical maximum number of Pixel Shader threads.
    *
    * PSD means Pixel Shader Dispatcher. On modern Intel GPUs, hardware will
    * automatically scale pixel shader thread count, based on a single value
    * programmed into 3DSTATE_PS.
    *
    * To calculate the maximum number of threads for Gen8 beyond (which have
    * multiple Pixel Shader Dispatchers):
    *
    * - Look up 3DSTATE_PS and find "Maximum Number of Threads Per PSD"
    * - Usually there's only one PSD per subslice, so use the number of
    *   subslices for number of PSDs.
    * - For max_wm_threads, the total should be PSD threads * #PSDs.
    */
   unsigned max_wm_threads;

   /**
    * Maximum Compute Shader threads.
    *
    * Thread count * number of EUs per subslice
    */
   unsigned max_cs_threads;

   struct {
      /**
       * Hardware default URB size.
       *
       * The units this is expressed in are somewhat inconsistent: 512b units
       * on Gen4-5, KB on Gen6-7, and KB times the slice count on Gen8+.
       *
       * Look up "URB Size" in the "Device Attributes" page, and take the
       * maximum.  Look up the slice count for each GT SKU on the same page.
       * urb.size = URB Size (kbytes) / slice count
       */
      unsigned size;

      /**
       * The minimum number of URB entries.  See the 3DSTATE_URB_<XS> docs.
       */
      unsigned min_entries[4];

      /**
       * The maximum number of URB entries.  See the 3DSTATE_URB_<XS> docs.
       */
      unsigned max_entries[4];
   } urb;

   /**
    * For the longest time the timestamp frequency for Gen's timestamp counter
    * could be assumed to be 12.5MHz, where the least significant bit neatly
    * corresponded to 80 nanoseconds.
    *
    * Since Gen9 the numbers aren't so round, with a a frequency of 12MHz for
    * SKL (or scale factor of 83.33333333) and a frequency of 19200000Hz for
    * BXT.
    *
    * For simplicty to fit with the current code scaling by a single constant
    * to map from raw timestamps to nanoseconds we now do the conversion in
    * floating point instead of integer arithmetic.
    *
    * In general it's probably worth noting that the documented constants we
    * have for the per-platform timestamp frequencies aren't perfect and
    * shouldn't be trusted for scaling and comparing timestamps with a large
    * delta.
    *
    * E.g. with crude testing on my system using the 'correct' scale factor I'm
    * seeing a drift of ~2 milliseconds per second.
    */
   uint64_t timestamp_frequency;

   /**
    * ID to put into the .aub files.
    */
   int simulator_id;

   /** @} */
};

#define gen_device_info_is_9lp(devinfo) \
   ((devinfo)->is_broxton || (devinfo)->is_geminilake)

static inline bool
gen_device_info_subslice_available(const struct gen_device_info *devinfo,
                                   int slice, int subslice)
{
   return (devinfo->subslice_masks[slice * devinfo->subslice_slice_stride +
                                   subslice / 8] & (1U << (subslice % 8))) != 0;
}

int gen_get_pci_device_id_override(void);
int gen_device_name_to_pci_device_id(const char *name);
bool gen_get_device_info(int devid, struct gen_device_info *devinfo);
const char *gen_get_device_name(int devid);

/* Used with SLICE_MASK/SUBSLICE_MASK values from DRM_I915_GETPARAM. */
void gen_device_info_update_from_masks(struct gen_device_info *devinfo,
                                       uint32_t slice_mask,
                                       uint32_t subslice_mask,
                                       uint32_t n_eus);
/* Used with DRM_IOCTL_I915_QUERY & DRM_I915_QUERY_TOPOLOGY_INFO. */
void gen_device_info_update_from_topology(struct gen_device_info *devinfo,
                                          const struct drm_i915_query_topology_info *topology);

#ifdef __cplusplus
}
#endif

#endif /* GEN_DEVICE_INFO_H */
