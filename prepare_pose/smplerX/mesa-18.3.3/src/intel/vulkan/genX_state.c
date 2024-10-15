/*
 * Copyright © 2015 Intel Corporation
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

#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include "anv_private.h"

#include "common/gen_sample_positions.h"
#include "genxml/gen_macros.h"
#include "genxml/genX_pack.h"

#include "vk_util.h"

#if GEN_GEN == 10
/**
 * From Gen10 Workarounds page in h/w specs:
 * WaSampleOffsetIZ:
 *    "Prior to the 3DSTATE_SAMPLE_PATTERN driver must ensure there are no
 *     markers in the pipeline by programming a PIPE_CONTROL with stall."
 */
static void
gen10_emit_wa_cs_stall_flush(struct anv_batch *batch)
{

   anv_batch_emit(batch, GENX(PIPE_CONTROL), pc) {
      pc.CommandStreamerStallEnable = true;
      pc.StallAtPixelScoreboard = true;
   }
}

/**
 * From Gen10 Workarounds page in h/w specs:
 * WaSampleOffsetIZ:_cs_stall_flush
 *    "When 3DSTATE_SAMPLE_PATTERN is programmed, driver must then issue an
 *     MI_LOAD_REGISTER_IMM command to an offset between 0x7000 and 0x7FFF(SVL)
 *     after the command to ensure the state has been delivered prior to any
 *     command causing a marker in the pipeline."
 */
static void
gen10_emit_wa_lri_to_cache_mode_zero(struct anv_batch *batch)
{
   /* Before changing the value of CACHE_MODE_0 register, GFX pipeline must
    * be idle; i.e., full flush is required.
    */
   anv_batch_emit(batch, GENX(PIPE_CONTROL), pc) {
      pc.DepthCacheFlushEnable = true;
      pc.DCFlushEnable = true;
      pc.RenderTargetCacheFlushEnable = true;
      pc.InstructionCacheInvalidateEnable = true;
      pc.StateCacheInvalidationEnable = true;
      pc.TextureCacheInvalidationEnable = true;
      pc.VFCacheInvalidationEnable = true;
      pc.ConstantCacheInvalidationEnable =true;
   }

   /* Write to CACHE_MODE_0 (0x7000) */
   uint32_t cache_mode_0 = 0;
   anv_pack_struct(&cache_mode_0, GENX(CACHE_MODE_0));

   anv_batch_emit(batch, GENX(MI_LOAD_REGISTER_IMM), lri) {
      lri.RegisterOffset = GENX(CACHE_MODE_0_num);
      lri.DataDWord      = cache_mode_0;
   }
}
#endif

VkResult
genX(init_device_state)(struct anv_device *device)
{
   GENX(MEMORY_OBJECT_CONTROL_STATE_pack)(NULL, &device->default_mocs,
                                          &GENX(MOCS));
#if GEN_GEN >= 8
   GENX(MEMORY_OBJECT_CONTROL_STATE_pack)(NULL, &device->external_mocs,
                                          &GENX(EXTERNAL_MOCS));
#else
   device->external_mocs = device->default_mocs;
#endif

   struct anv_batch batch;

   uint32_t cmds[64];
   batch.start = batch.next = cmds;
   batch.end = (void *) cmds + sizeof(cmds);

   anv_batch_emit(&batch, GENX(PIPELINE_SELECT), ps) {
#if GEN_GEN >= 9
      ps.MaskBits = 3;
#endif
      ps.PipelineSelection = _3D;
   }

#if GEN_GEN == 9
   uint32_t cache_mode_1;
   anv_pack_struct(&cache_mode_1, GENX(CACHE_MODE_1),
                   .FloatBlendOptimizationEnable = true,
                   .FloatBlendOptimizationEnableMask = true,
                   .PartialResolveDisableInVC = true,
                   .PartialResolveDisableInVCMask = true);

   anv_batch_emit(&batch, GENX(MI_LOAD_REGISTER_IMM), lri) {
      lri.RegisterOffset = GENX(CACHE_MODE_1_num);
      lri.DataDWord      = cache_mode_1;
   }
#endif

   anv_batch_emit(&batch, GENX(3DSTATE_AA_LINE_PARAMETERS), aa);

   anv_batch_emit(&batch, GENX(3DSTATE_DRAWING_RECTANGLE), rect) {
      rect.ClippedDrawingRectangleYMin = 0;
      rect.ClippedDrawingRectangleXMin = 0;
      rect.ClippedDrawingRectangleYMax = UINT16_MAX;
      rect.ClippedDrawingRectangleXMax = UINT16_MAX;
      rect.DrawingRectangleOriginY = 0;
      rect.DrawingRectangleOriginX = 0;
   }

#if GEN_GEN >= 8
   anv_batch_emit(&batch, GENX(3DSTATE_WM_CHROMAKEY), ck);

#if GEN_GEN == 10
   gen10_emit_wa_cs_stall_flush(&batch);
#endif

   /* See the Vulkan 1.0 spec Table 24.1 "Standard sample locations" and
    * VkPhysicalDeviceFeatures::standardSampleLocations.
    */
   anv_batch_emit(&batch, GENX(3DSTATE_SAMPLE_PATTERN), sp) {
      GEN_SAMPLE_POS_1X(sp._1xSample);
      GEN_SAMPLE_POS_2X(sp._2xSample);
      GEN_SAMPLE_POS_4X(sp._4xSample);
      GEN_SAMPLE_POS_8X(sp._8xSample);
#if GEN_GEN >= 9
      GEN_SAMPLE_POS_16X(sp._16xSample);
#endif
   }

   /* The BDW+ docs describe how to use the 3DSTATE_WM_HZ_OP instruction in the
    * section titled, "Optimized Depth Buffer Clear and/or Stencil Buffer
    * Clear." It mentions that the packet overrides GPU state for the clear
    * operation and needs to be reset to 0s to clear the overrides. Depending
    * on the kernel, we may not get a context with the state for this packet
    * zeroed. Do it ourselves just in case. We've observed this to prevent a
    * number of GPU hangs on ICL.
    */
   anv_batch_emit(&batch, GENX(3DSTATE_WM_HZ_OP), hzp);
#endif

#if GEN_GEN == 10
   gen10_emit_wa_lri_to_cache_mode_zero(&batch);
#endif

#if GEN_GEN == 11
   /* The default behavior of bit 5 "Headerless Message for Pre-emptable
    * Contexts" in SAMPLER MODE register is set to 0, which means
    * headerless sampler messages are not allowed for pre-emptable
    * contexts. Set the bit 5 to 1 to allow them.
    */
   uint32_t sampler_mode;
   anv_pack_struct(&sampler_mode, GENX(SAMPLER_MODE),
                   .HeaderlessMessageforPreemptableContexts = true,
                   .HeaderlessMessageforPreemptableContextsMask = true);

    anv_batch_emit(&batch, GENX(MI_LOAD_REGISTER_IMM), lri) {
      lri.RegisterOffset = GENX(SAMPLER_MODE_num);
      lri.DataDWord      = sampler_mode;
   }

   /* Bit 1 "Enabled Texel Offset Precision Fix" must be set in
    * HALF_SLICE_CHICKEN7 register.
    */
   uint32_t half_slice_chicken7;
   anv_pack_struct(&half_slice_chicken7, GENX(HALF_SLICE_CHICKEN7),
                   .EnabledTexelOffsetPrecisionFix = true,
                   .EnabledTexelOffsetPrecisionFixMask = true);

    anv_batch_emit(&batch, GENX(MI_LOAD_REGISTER_IMM), lri) {
      lri.RegisterOffset = GENX(HALF_SLICE_CHICKEN7_num);
      lri.DataDWord      = half_slice_chicken7;
   }

#endif

   /* Set the "CONSTANT_BUFFER Address Offset Disable" bit, so
    * 3DSTATE_CONSTANT_XS buffer 0 is an absolute address.
    *
    * This is only safe on kernels with context isolation support.
    */
   if (GEN_GEN >= 8 &&
       device->instance->physicalDevice.has_context_isolation) {
      UNUSED uint32_t tmp_reg;
#if GEN_GEN >= 9
      anv_pack_struct(&tmp_reg, GENX(CS_DEBUG_MODE2),
                      .CONSTANT_BUFFERAddressOffsetDisable = true,
                      .CONSTANT_BUFFERAddressOffsetDisableMask = true);
      anv_batch_emit(&batch, GENX(MI_LOAD_REGISTER_IMM), lri) {
         lri.RegisterOffset = GENX(CS_DEBUG_MODE2_num);
         lri.DataDWord      = tmp_reg;
      }
#elif GEN_GEN == 8
      anv_pack_struct(&tmp_reg, GENX(INSTPM),
                      .CONSTANT_BUFFERAddressOffsetDisable = true,
                      .CONSTANT_BUFFERAddressOffsetDisableMask = true);
      anv_batch_emit(&batch, GENX(MI_LOAD_REGISTER_IMM), lri) {
         lri.RegisterOffset = GENX(INSTPM_num);
         lri.DataDWord      = tmp_reg;
      }
#endif
   }

   anv_batch_emit(&batch, GENX(MI_BATCH_BUFFER_END), bbe);

   assert(batch.next <= batch.end);

   return anv_device_submit_simple_batch(device, &batch);
}

static uint32_t
vk_to_gen_tex_filter(VkFilter filter, bool anisotropyEnable)
{
   switch (filter) {
   default:
      assert(!"Invalid filter");
   case VK_FILTER_NEAREST:
      return anisotropyEnable ? MAPFILTER_ANISOTROPIC : MAPFILTER_NEAREST;
   case VK_FILTER_LINEAR:
      return anisotropyEnable ? MAPFILTER_ANISOTROPIC : MAPFILTER_LINEAR;
   }
}

static uint32_t
vk_to_gen_max_anisotropy(float ratio)
{
   return (anv_clamp_f(ratio, 2, 16) - 2) / 2;
}

static const uint32_t vk_to_gen_mipmap_mode[] = {
   [VK_SAMPLER_MIPMAP_MODE_NEAREST]          = MIPFILTER_NEAREST,
   [VK_SAMPLER_MIPMAP_MODE_LINEAR]           = MIPFILTER_LINEAR
};

static const uint32_t vk_to_gen_tex_address[] = {
   [VK_SAMPLER_ADDRESS_MODE_REPEAT]          = TCM_WRAP,
   [VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT] = TCM_MIRROR,
   [VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE]   = TCM_CLAMP,
   [VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE] = TCM_MIRROR_ONCE,
   [VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER] = TCM_CLAMP_BORDER,
};

/* Vulkan specifies the result of shadow comparisons as:
 *     1     if   ref <op> texel,
 *     0     otherwise.
 *
 * The hardware does:
 *     0     if texel <op> ref,
 *     1     otherwise.
 *
 * So, these look a bit strange because there's both a negation
 * and swapping of the arguments involved.
 */
static const uint32_t vk_to_gen_shadow_compare_op[] = {
   [VK_COMPARE_OP_NEVER]                        = PREFILTEROPALWAYS,
   [VK_COMPARE_OP_LESS]                         = PREFILTEROPLEQUAL,
   [VK_COMPARE_OP_EQUAL]                        = PREFILTEROPNOTEQUAL,
   [VK_COMPARE_OP_LESS_OR_EQUAL]                = PREFILTEROPLESS,
   [VK_COMPARE_OP_GREATER]                      = PREFILTEROPGEQUAL,
   [VK_COMPARE_OP_NOT_EQUAL]                    = PREFILTEROPEQUAL,
   [VK_COMPARE_OP_GREATER_OR_EQUAL]             = PREFILTEROPGREATER,
   [VK_COMPARE_OP_ALWAYS]                       = PREFILTEROPNEVER,
};

#if GEN_GEN >= 9
static const uint32_t vk_to_gen_sampler_reduction_mode[] = {
   [VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE_EXT] = STD_FILTER,
   [VK_SAMPLER_REDUCTION_MODE_MIN_EXT]              = MINIMUM,
   [VK_SAMPLER_REDUCTION_MODE_MAX_EXT]              = MAXIMUM,
};
#endif

VkResult genX(CreateSampler)(
    VkDevice                                    _device,
    const VkSamplerCreateInfo*                  pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSampler*                                  pSampler)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_sampler *sampler;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);

   sampler = vk_zalloc2(&device->alloc, pAllocator, sizeof(*sampler), 8,
                        VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!sampler)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   sampler->n_planes = 1;

   uint32_t border_color_offset = device->border_colors.offset +
                                  pCreateInfo->borderColor * 64;

#if GEN_GEN >= 9
   unsigned sampler_reduction_mode = STD_FILTER;
   bool enable_sampler_reduction = false;
#endif

   vk_foreach_struct(ext, pCreateInfo->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO: {
         VkSamplerYcbcrConversionInfo *pSamplerConversion =
            (VkSamplerYcbcrConversionInfo *) ext;
         ANV_FROM_HANDLE(anv_ycbcr_conversion, conversion,
                         pSamplerConversion->conversion);

         if (conversion == NULL)
            break;

         sampler->n_planes = conversion->format->n_planes;
         sampler->conversion = conversion;
         break;
      }
#if GEN_GEN >= 9
      case VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT: {
         struct VkSamplerReductionModeCreateInfoEXT *sampler_reduction =
            (struct VkSamplerReductionModeCreateInfoEXT *) ext;
         sampler_reduction_mode =
            vk_to_gen_sampler_reduction_mode[sampler_reduction->reductionMode];
         enable_sampler_reduction = true;
         break;
      }
#endif
      default:
         anv_debug_ignored_stype(ext->sType);
         break;
      }
   }

   for (unsigned p = 0; p < sampler->n_planes; p++) {
      const bool plane_has_chroma =
         sampler->conversion && sampler->conversion->format->planes[p].has_chroma;
      const VkFilter min_filter =
         plane_has_chroma ? sampler->conversion->chroma_filter : pCreateInfo->minFilter;
      const VkFilter mag_filter =
         plane_has_chroma ? sampler->conversion->chroma_filter : pCreateInfo->magFilter;
      const bool enable_min_filter_addr_rounding = min_filter != VK_FILTER_NEAREST;
      const bool enable_mag_filter_addr_rounding = mag_filter != VK_FILTER_NEAREST;
      /* From Broadwell PRM, SAMPLER_STATE:
       *   "Mip Mode Filter must be set to MIPFILTER_NONE for Planar YUV surfaces."
       */
      const uint32_t mip_filter_mode =
         (sampler->conversion &&
          isl_format_is_yuv(sampler->conversion->format->planes[0].isl_format)) ?
         MIPFILTER_NONE : vk_to_gen_mipmap_mode[pCreateInfo->mipmapMode];

      struct GENX(SAMPLER_STATE) sampler_state = {
         .SamplerDisable = false,
         .TextureBorderColorMode = DX10OGL,

#if GEN_GEN >= 8
         .LODPreClampMode = CLAMP_MODE_OGL,
#else
         .LODPreClampEnable = CLAMP_ENABLE_OGL,
#endif

#if GEN_GEN == 8
         .BaseMipLevel = 0.0,
#endif
         .MipModeFilter = mip_filter_mode,
         .MagModeFilter = vk_to_gen_tex_filter(mag_filter, pCreateInfo->anisotropyEnable),
         .MinModeFilter = vk_to_gen_tex_filter(min_filter, pCreateInfo->anisotropyEnable),
         .TextureLODBias = anv_clamp_f(pCreateInfo->mipLodBias, -16, 15.996),
         .AnisotropicAlgorithm = EWAApproximation,
         .MinLOD = anv_clamp_f(pCreateInfo->minLod, 0, 14),
         .MaxLOD = anv_clamp_f(pCreateInfo->maxLod, 0, 14),
         .ChromaKeyEnable = 0,
         .ChromaKeyIndex = 0,
         .ChromaKeyMode = 0,
         .ShadowFunction = vk_to_gen_shadow_compare_op[pCreateInfo->compareOp],
         .CubeSurfaceControlMode = OVERRIDE,

         .BorderColorPointer = border_color_offset,

#if GEN_GEN >= 8
         .LODClampMagnificationMode = MIPNONE,
#endif

         .MaximumAnisotropy = vk_to_gen_max_anisotropy(pCreateInfo->maxAnisotropy),
         .RAddressMinFilterRoundingEnable = enable_min_filter_addr_rounding,
         .RAddressMagFilterRoundingEnable = enable_mag_filter_addr_rounding,
         .VAddressMinFilterRoundingEnable = enable_min_filter_addr_rounding,
         .VAddressMagFilterRoundingEnable = enable_mag_filter_addr_rounding,
         .UAddressMinFilterRoundingEnable = enable_min_filter_addr_rounding,
         .UAddressMagFilterRoundingEnable = enable_mag_filter_addr_rounding,
         .TrilinearFilterQuality = 0,
         .NonnormalizedCoordinateEnable = pCreateInfo->unnormalizedCoordinates,
         .TCXAddressControlMode = vk_to_gen_tex_address[pCreateInfo->addressModeU],
         .TCYAddressControlMode = vk_to_gen_tex_address[pCreateInfo->addressModeV],
         .TCZAddressControlMode = vk_to_gen_tex_address[pCreateInfo->addressModeW],

#if GEN_GEN >= 9
         .ReductionType = sampler_reduction_mode,
         .ReductionTypeEnable = enable_sampler_reduction,
#endif
      };

      GENX(SAMPLER_STATE_pack)(NULL, sampler->state[p], &sampler_state);
   }

   *pSampler = anv_sampler_to_handle(sampler);

   return VK_SUCCESS;
}
