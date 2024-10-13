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

#include "anv_private.h"

#include "genxml/gen_macros.h"
#include "genxml/genX_pack.h"

#include "common/gen_l3_config.h"
#include "common/gen_sample_positions.h"
#include "vk_util.h"
#include "vk_format_info.h"

static uint32_t
vertex_element_comp_control(enum isl_format format, unsigned comp)
{
   uint8_t bits;
   switch (comp) {
   case 0: bits = isl_format_layouts[format].channels.r.bits; break;
   case 1: bits = isl_format_layouts[format].channels.g.bits; break;
   case 2: bits = isl_format_layouts[format].channels.b.bits; break;
   case 3: bits = isl_format_layouts[format].channels.a.bits; break;
   default: unreachable("Invalid component");
   }

   /*
    * Take in account hardware restrictions when dealing with 64-bit floats.
    *
    * From Broadwell spec, command reference structures, page 586:
    *  "When SourceElementFormat is set to one of the *64*_PASSTHRU formats,
    *   64-bit components are stored * in the URB without any conversion. In
    *   this case, vertex elements must be written as 128 or 256 bits, with
    *   VFCOMP_STORE_0 being used to pad the output as required. E.g., if
    *   R64_PASSTHRU is used to copy a 64-bit Red component into the URB,
    *   Component 1 must be specified as VFCOMP_STORE_0 (with Components 2,3
    *   set to VFCOMP_NOSTORE) in order to output a 128-bit vertex element, or
    *   Components 1-3 must be specified as VFCOMP_STORE_0 in order to output
    *   a 256-bit vertex element. Likewise, use of R64G64B64_PASSTHRU requires
    *   Component 3 to be specified as VFCOMP_STORE_0 in order to output a
    *   256-bit vertex element."
    */
   if (bits) {
      return VFCOMP_STORE_SRC;
   } else if (comp >= 2 &&
              !isl_format_layouts[format].channels.b.bits &&
              isl_format_layouts[format].channels.r.type == ISL_RAW) {
      /* When emitting 64-bit attributes, we need to write either 128 or 256
       * bit chunks, using VFCOMP_NOSTORE when not writing the chunk, and
       * VFCOMP_STORE_0 to pad the written chunk */
      return VFCOMP_NOSTORE;
   } else if (comp < 3 ||
              isl_format_layouts[format].channels.r.type == ISL_RAW) {
      /* Note we need to pad with value 0, not 1, due hardware restrictions
       * (see comment above) */
      return VFCOMP_STORE_0;
   } else if (isl_format_layouts[format].channels.r.type == ISL_UINT ||
            isl_format_layouts[format].channels.r.type == ISL_SINT) {
      assert(comp == 3);
      return VFCOMP_STORE_1_INT;
   } else {
      assert(comp == 3);
      return VFCOMP_STORE_1_FP;
   }
}

static void
emit_vertex_input(struct anv_pipeline *pipeline,
                  const VkPipelineVertexInputStateCreateInfo *info)
{
   const struct brw_vs_prog_data *vs_prog_data = get_vs_prog_data(pipeline);

   /* Pull inputs_read out of the VS prog data */
   const uint64_t inputs_read = vs_prog_data->inputs_read;
   const uint64_t double_inputs_read =
      vs_prog_data->double_inputs_read & inputs_read;
   assert((inputs_read & ((1 << VERT_ATTRIB_GENERIC0) - 1)) == 0);
   const uint32_t elements = inputs_read >> VERT_ATTRIB_GENERIC0;
   const uint32_t elements_double = double_inputs_read >> VERT_ATTRIB_GENERIC0;
   const bool needs_svgs_elem = vs_prog_data->uses_vertexid ||
                                vs_prog_data->uses_instanceid ||
                                vs_prog_data->uses_firstvertex ||
                                vs_prog_data->uses_baseinstance;

   uint32_t elem_count = __builtin_popcount(elements) -
      __builtin_popcount(elements_double) / 2;

   const uint32_t total_elems =
      elem_count + needs_svgs_elem + vs_prog_data->uses_drawid;
   if (total_elems == 0)
      return;

   uint32_t *p;

   const uint32_t num_dwords = 1 + total_elems * 2;
   p = anv_batch_emitn(&pipeline->batch, num_dwords,
                       GENX(3DSTATE_VERTEX_ELEMENTS));
   if (!p)
      return;

   for (uint32_t i = 0; i < total_elems; i++) {
      /* The SKL docs for VERTEX_ELEMENT_STATE say:
       *
       *    "All elements must be valid from Element[0] to the last valid
       *    element. (I.e. if Element[2] is valid then Element[1] and
       *    Element[0] must also be valid)."
       *
       * The SKL docs for 3D_Vertex_Component_Control say:
       *
       *    "Don't store this component. (Not valid for Component 0, but can
       *    be used for Component 1-3)."
       *
       * So we can't just leave a vertex element blank and hope for the best.
       * We have to tell the VF hardware to put something in it; so we just
       * store a bunch of zero.
       *
       * TODO: Compact vertex elements so we never end up with holes.
       */
      struct GENX(VERTEX_ELEMENT_STATE) element = {
         .Valid = true,
         .Component0Control = VFCOMP_STORE_0,
         .Component1Control = VFCOMP_STORE_0,
         .Component2Control = VFCOMP_STORE_0,
         .Component3Control = VFCOMP_STORE_0,
      };
      GENX(VERTEX_ELEMENT_STATE_pack)(NULL, &p[1 + i * 2], &element);
   }

   for (uint32_t i = 0; i < info->vertexAttributeDescriptionCount; i++) {
      const VkVertexInputAttributeDescription *desc =
         &info->pVertexAttributeDescriptions[i];
      enum isl_format format = anv_get_isl_format(&pipeline->device->info,
                                                  desc->format,
                                                  VK_IMAGE_ASPECT_COLOR_BIT,
                                                  VK_IMAGE_TILING_LINEAR);

      assert(desc->binding < MAX_VBS);

      if ((elements & (1 << desc->location)) == 0)
         continue; /* Binding unused */

      uint32_t slot =
         __builtin_popcount(elements & ((1 << desc->location) - 1)) -
         DIV_ROUND_UP(__builtin_popcount(elements_double &
                                        ((1 << desc->location) -1)), 2);

      struct GENX(VERTEX_ELEMENT_STATE) element = {
         .VertexBufferIndex = desc->binding,
         .Valid = true,
         .SourceElementFormat = format,
         .EdgeFlagEnable = false,
         .SourceElementOffset = desc->offset,
         .Component0Control = vertex_element_comp_control(format, 0),
         .Component1Control = vertex_element_comp_control(format, 1),
         .Component2Control = vertex_element_comp_control(format, 2),
         .Component3Control = vertex_element_comp_control(format, 3),
      };
      GENX(VERTEX_ELEMENT_STATE_pack)(NULL, &p[1 + slot * 2], &element);

#if GEN_GEN >= 8
      /* On Broadwell and later, we have a separate VF_INSTANCING packet
       * that controls instancing.  On Haswell and prior, that's part of
       * VERTEX_BUFFER_STATE which we emit later.
       */
      anv_batch_emit(&pipeline->batch, GENX(3DSTATE_VF_INSTANCING), vfi) {
         vfi.InstancingEnable = pipeline->vb[desc->binding].instanced;
         vfi.VertexElementIndex = slot;
         vfi.InstanceDataStepRate =
            pipeline->vb[desc->binding].instance_divisor;
      }
#endif
   }

   const uint32_t id_slot = elem_count;
   if (needs_svgs_elem) {
      /* From the Broadwell PRM for the 3D_Vertex_Component_Control enum:
       *    "Within a VERTEX_ELEMENT_STATE structure, if a Component
       *    Control field is set to something other than VFCOMP_STORE_SRC,
       *    no higher-numbered Component Control fields may be set to
       *    VFCOMP_STORE_SRC"
       *
       * This means, that if we have BaseInstance, we need BaseVertex as
       * well.  Just do all or nothing.
       */
      uint32_t base_ctrl = (vs_prog_data->uses_firstvertex ||
                            vs_prog_data->uses_baseinstance) ?
                           VFCOMP_STORE_SRC : VFCOMP_STORE_0;

      struct GENX(VERTEX_ELEMENT_STATE) element = {
         .VertexBufferIndex = ANV_SVGS_VB_INDEX,
         .Valid = true,
         .SourceElementFormat = ISL_FORMAT_R32G32_UINT,
         .Component0Control = base_ctrl,
         .Component1Control = base_ctrl,
#if GEN_GEN >= 8
         .Component2Control = VFCOMP_STORE_0,
         .Component3Control = VFCOMP_STORE_0,
#else
         .Component2Control = VFCOMP_STORE_VID,
         .Component3Control = VFCOMP_STORE_IID,
#endif
      };
      GENX(VERTEX_ELEMENT_STATE_pack)(NULL, &p[1 + id_slot * 2], &element);
   }

#if GEN_GEN >= 8
   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_VF_SGVS), sgvs) {
      sgvs.VertexIDEnable              = vs_prog_data->uses_vertexid;
      sgvs.VertexIDComponentNumber     = 2;
      sgvs.VertexIDElementOffset       = id_slot;
      sgvs.InstanceIDEnable            = vs_prog_data->uses_instanceid;
      sgvs.InstanceIDComponentNumber   = 3;
      sgvs.InstanceIDElementOffset     = id_slot;
   }
#endif

   const uint32_t drawid_slot = elem_count + needs_svgs_elem;
   if (vs_prog_data->uses_drawid) {
      struct GENX(VERTEX_ELEMENT_STATE) element = {
         .VertexBufferIndex = ANV_DRAWID_VB_INDEX,
         .Valid = true,
         .SourceElementFormat = ISL_FORMAT_R32_UINT,
         .Component0Control = VFCOMP_STORE_SRC,
         .Component1Control = VFCOMP_STORE_0,
         .Component2Control = VFCOMP_STORE_0,
         .Component3Control = VFCOMP_STORE_0,
      };
      GENX(VERTEX_ELEMENT_STATE_pack)(NULL,
                                      &p[1 + drawid_slot * 2],
                                      &element);

#if GEN_GEN >= 8
      anv_batch_emit(&pipeline->batch, GENX(3DSTATE_VF_INSTANCING), vfi) {
         vfi.VertexElementIndex = drawid_slot;
      }
#endif
   }
}

void
genX(emit_urb_setup)(struct anv_device *device, struct anv_batch *batch,
                     const struct gen_l3_config *l3_config,
                     VkShaderStageFlags active_stages,
                     const unsigned entry_size[4])
{
   const struct gen_device_info *devinfo = &device->info;
#if GEN_IS_HASWELL
   const unsigned push_constant_kb = devinfo->gt == 3 ? 32 : 16;
#else
   const unsigned push_constant_kb = GEN_GEN >= 8 ? 32 : 16;
#endif

   const unsigned urb_size_kb = gen_get_l3_config_urb_size(devinfo, l3_config);

   unsigned entries[4];
   unsigned start[4];
   gen_get_urb_config(devinfo,
                      1024 * push_constant_kb, 1024 * urb_size_kb,
                      active_stages &
                         VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                      active_stages & VK_SHADER_STAGE_GEOMETRY_BIT,
                      entry_size, entries, start);

#if GEN_GEN == 7 && !GEN_IS_HASWELL
   /* From the IVB PRM Vol. 2, Part 1, Section 3.2.1:
    *
    *    "A PIPE_CONTROL with Post-Sync Operation set to 1h and a depth stall
    *    needs to be sent just prior to any 3DSTATE_VS, 3DSTATE_URB_VS,
    *    3DSTATE_CONSTANT_VS, 3DSTATE_BINDING_TABLE_POINTER_VS,
    *    3DSTATE_SAMPLER_STATE_POINTER_VS command.  Only one PIPE_CONTROL
    *    needs to be sent before any combination of VS associated 3DSTATE."
    */
   anv_batch_emit(batch, GEN7_PIPE_CONTROL, pc) {
      pc.DepthStallEnable  = true;
      pc.PostSyncOperation = WriteImmediateData;
      pc.Address           = (struct anv_address) { &device->workaround_bo, 0 };
   }
#endif

   for (int i = 0; i <= MESA_SHADER_GEOMETRY; i++) {
      anv_batch_emit(batch, GENX(3DSTATE_URB_VS), urb) {
         urb._3DCommandSubOpcode      += i;
         urb.VSURBStartingAddress      = start[i];
         urb.VSURBEntryAllocationSize  = entry_size[i] - 1;
         urb.VSNumberofURBEntries      = entries[i];
      }
   }
}

static void
emit_urb_setup(struct anv_pipeline *pipeline)
{
   unsigned entry_size[4];
   for (int i = MESA_SHADER_VERTEX; i <= MESA_SHADER_GEOMETRY; i++) {
      const struct brw_vue_prog_data *prog_data =
         !anv_pipeline_has_stage(pipeline, i) ? NULL :
         (const struct brw_vue_prog_data *) pipeline->shaders[i]->prog_data;

      entry_size[i] = prog_data ? prog_data->urb_entry_size : 1;
   }

   genX(emit_urb_setup)(pipeline->device, &pipeline->batch,
                        pipeline->urb.l3_config,
                        pipeline->active_stages, entry_size);
}

static void
emit_3dstate_sbe(struct anv_pipeline *pipeline)
{
   const struct brw_wm_prog_data *wm_prog_data = get_wm_prog_data(pipeline);

   if (!anv_pipeline_has_stage(pipeline, MESA_SHADER_FRAGMENT)) {
      anv_batch_emit(&pipeline->batch, GENX(3DSTATE_SBE), sbe);
#if GEN_GEN >= 8
      anv_batch_emit(&pipeline->batch, GENX(3DSTATE_SBE_SWIZ), sbe);
#endif
      return;
   }

   const struct brw_vue_map *fs_input_map =
      &anv_pipeline_get_last_vue_prog_data(pipeline)->vue_map;

   struct GENX(3DSTATE_SBE) sbe = {
      GENX(3DSTATE_SBE_header),
      .AttributeSwizzleEnable = true,
      .PointSpriteTextureCoordinateOrigin = UPPERLEFT,
      .NumberofSFOutputAttributes = wm_prog_data->num_varying_inputs,
      .ConstantInterpolationEnable = wm_prog_data->flat_inputs,
   };

#if GEN_GEN >= 9
   for (unsigned i = 0; i < 32; i++)
      sbe.AttributeActiveComponentFormat[i] = ACF_XYZW;
#endif

#if GEN_GEN >= 8
   /* On Broadwell, they broke 3DSTATE_SBE into two packets */
   struct GENX(3DSTATE_SBE_SWIZ) swiz = {
      GENX(3DSTATE_SBE_SWIZ_header),
   };
#else
#  define swiz sbe
#endif

   /* Skip the VUE header and position slots by default */
   unsigned urb_entry_read_offset = 1;
   int max_source_attr = 0;
   for (int attr = 0; attr < VARYING_SLOT_MAX; attr++) {
      int input_index = wm_prog_data->urb_setup[attr];

      if (input_index < 0)
         continue;

      /* gl_Layer is stored in the VUE header */
      if (attr == VARYING_SLOT_LAYER) {
         urb_entry_read_offset = 0;
         continue;
      }

      if (attr == VARYING_SLOT_PNTC) {
         sbe.PointSpriteTextureCoordinateEnable = 1 << input_index;
         continue;
      }

      const int slot = fs_input_map->varying_to_slot[attr];

      if (input_index >= 16)
         continue;

      if (slot == -1) {
         /* This attribute does not exist in the VUE--that means that the
          * vertex shader did not write to it.  It could be that it's a
          * regular varying read by the fragment shader but not written by
          * the vertex shader or it's gl_PrimitiveID. In the first case the
          * value is undefined, in the second it needs to be
          * gl_PrimitiveID.
          */
         swiz.Attribute[input_index].ConstantSource = PRIM_ID;
         swiz.Attribute[input_index].ComponentOverrideX = true;
         swiz.Attribute[input_index].ComponentOverrideY = true;
         swiz.Attribute[input_index].ComponentOverrideZ = true;
         swiz.Attribute[input_index].ComponentOverrideW = true;
      } else {
         /* We have to subtract two slots to accout for the URB entry output
          * read offset in the VS and GS stages.
          */
         const int source_attr = slot - 2 * urb_entry_read_offset;
         assert(source_attr >= 0 && source_attr < 32);
         max_source_attr = MAX2(max_source_attr, source_attr);
         swiz.Attribute[input_index].SourceAttribute = source_attr;
      }
   }

   sbe.VertexURBEntryReadOffset = urb_entry_read_offset;
   sbe.VertexURBEntryReadLength = DIV_ROUND_UP(max_source_attr + 1, 2);
#if GEN_GEN >= 8
   sbe.ForceVertexURBEntryReadOffset = true;
   sbe.ForceVertexURBEntryReadLength = true;
#endif

   uint32_t *dw = anv_batch_emit_dwords(&pipeline->batch,
                                        GENX(3DSTATE_SBE_length));
   if (!dw)
      return;
   GENX(3DSTATE_SBE_pack)(&pipeline->batch, dw, &sbe);

#if GEN_GEN >= 8
   dw = anv_batch_emit_dwords(&pipeline->batch, GENX(3DSTATE_SBE_SWIZ_length));
   if (!dw)
      return;
   GENX(3DSTATE_SBE_SWIZ_pack)(&pipeline->batch, dw, &swiz);
#endif
}

static const uint32_t vk_to_gen_cullmode[] = {
   [VK_CULL_MODE_NONE]                       = CULLMODE_NONE,
   [VK_CULL_MODE_FRONT_BIT]                  = CULLMODE_FRONT,
   [VK_CULL_MODE_BACK_BIT]                   = CULLMODE_BACK,
   [VK_CULL_MODE_FRONT_AND_BACK]             = CULLMODE_BOTH
};

static const uint32_t vk_to_gen_fillmode[] = {
   [VK_POLYGON_MODE_FILL]                    = FILL_MODE_SOLID,
   [VK_POLYGON_MODE_LINE]                    = FILL_MODE_WIREFRAME,
   [VK_POLYGON_MODE_POINT]                   = FILL_MODE_POINT,
};

static const uint32_t vk_to_gen_front_face[] = {
   [VK_FRONT_FACE_COUNTER_CLOCKWISE]         = 1,
   [VK_FRONT_FACE_CLOCKWISE]                 = 0
};

static void
emit_rs_state(struct anv_pipeline *pipeline,
              const VkPipelineRasterizationStateCreateInfo *rs_info,
              const VkPipelineMultisampleStateCreateInfo *ms_info,
              const struct anv_render_pass *pass,
              const struct anv_subpass *subpass)
{
   struct GENX(3DSTATE_SF) sf = {
      GENX(3DSTATE_SF_header),
   };

   sf.ViewportTransformEnable = true;
   sf.StatisticsEnable = true;
   sf.TriangleStripListProvokingVertexSelect = 0;
   sf.LineStripListProvokingVertexSelect = 0;
   sf.TriangleFanProvokingVertexSelect = 1;

   const struct brw_vue_prog_data *last_vue_prog_data =
      anv_pipeline_get_last_vue_prog_data(pipeline);

   if (last_vue_prog_data->vue_map.slots_valid & VARYING_BIT_PSIZ) {
      sf.PointWidthSource = Vertex;
   } else {
      sf.PointWidthSource = State;
      sf.PointWidth = 1.0;
   }

#if GEN_GEN >= 8
   struct GENX(3DSTATE_RASTER) raster = {
      GENX(3DSTATE_RASTER_header),
   };
#else
#  define raster sf
#endif

   /* For details on 3DSTATE_RASTER multisample state, see the BSpec table
    * "Multisample Modes State".
    */
#if GEN_GEN >= 8
   raster.DXMultisampleRasterizationEnable = true;
   /* NOTE: 3DSTATE_RASTER::ForcedSampleCount affects the BDW and SKL PMA fix
    * computations.  If we ever set this bit to a different value, they will
    * need to be updated accordingly.
    */
   raster.ForcedSampleCount = FSC_NUMRASTSAMPLES_0;
   raster.ForceMultisampling = false;
#else
   raster.MultisampleRasterizationMode =
      (ms_info && ms_info->rasterizationSamples > 1) ?
      MSRASTMODE_ON_PATTERN : MSRASTMODE_OFF_PIXEL;
#endif

   raster.FrontWinding = vk_to_gen_front_face[rs_info->frontFace];
   raster.CullMode = vk_to_gen_cullmode[rs_info->cullMode];
   raster.FrontFaceFillMode = vk_to_gen_fillmode[rs_info->polygonMode];
   raster.BackFaceFillMode = vk_to_gen_fillmode[rs_info->polygonMode];
   raster.ScissorRectangleEnable = true;

#if GEN_GEN >= 9
   /* GEN9+ splits ViewportZClipTestEnable into near and far enable bits */
   raster.ViewportZFarClipTestEnable = !pipeline->depth_clamp_enable;
   raster.ViewportZNearClipTestEnable = !pipeline->depth_clamp_enable;
#elif GEN_GEN >= 8
   raster.ViewportZClipTestEnable = !pipeline->depth_clamp_enable;
#endif

   raster.GlobalDepthOffsetEnableSolid = rs_info->depthBiasEnable;
   raster.GlobalDepthOffsetEnableWireframe = rs_info->depthBiasEnable;
   raster.GlobalDepthOffsetEnablePoint = rs_info->depthBiasEnable;

#if GEN_GEN == 7
   /* Gen7 requires that we provide the depth format in 3DSTATE_SF so that it
    * can get the depth offsets correct.
    */
   if (subpass->depth_stencil_attachment) {
      VkFormat vk_format =
         pass->attachments[subpass->depth_stencil_attachment->attachment].format;
      assert(vk_format_is_depth_or_stencil(vk_format));
      if (vk_format_aspects(vk_format) & VK_IMAGE_ASPECT_DEPTH_BIT) {
         enum isl_format isl_format =
            anv_get_isl_format(&pipeline->device->info, vk_format,
                               VK_IMAGE_ASPECT_DEPTH_BIT,
                               VK_IMAGE_TILING_OPTIMAL);
         sf.DepthBufferSurfaceFormat =
            isl_format_get_depth_format(isl_format, false);
      }
   }
#endif

#if GEN_GEN >= 8
   GENX(3DSTATE_SF_pack)(NULL, pipeline->gen8.sf, &sf);
   GENX(3DSTATE_RASTER_pack)(NULL, pipeline->gen8.raster, &raster);
#else
#  undef raster
   GENX(3DSTATE_SF_pack)(NULL, &pipeline->gen7.sf, &sf);
#endif
}

static void
emit_ms_state(struct anv_pipeline *pipeline,
              const VkPipelineMultisampleStateCreateInfo *info)
{
   uint32_t samples = 1;
   uint32_t log2_samples = 0;

   /* From the Vulkan 1.0 spec:
    *    If pSampleMask is NULL, it is treated as if the mask has all bits
    *    enabled, i.e. no coverage is removed from fragments.
    *
    * 3DSTATE_SAMPLE_MASK.SampleMask is 16 bits.
    */
#if GEN_GEN >= 8
   uint32_t sample_mask = 0xffff;
#else
   uint32_t sample_mask = 0xff;
#endif

   if (info) {
      samples = info->rasterizationSamples;
      log2_samples = __builtin_ffs(samples) - 1;
   }

   if (info && info->pSampleMask)
      sample_mask &= info->pSampleMask[0];

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_MULTISAMPLE), ms) {
      ms.NumberofMultisamples       = log2_samples;

      ms.PixelLocation              = CENTER;
#if GEN_GEN >= 8
      /* The PRM says that this bit is valid only for DX9:
       *
       *    SW can choose to set this bit only for DX9 API. DX10/OGL API's
       *    should not have any effect by setting or not setting this bit.
       */
      ms.PixelPositionOffsetEnable  = false;
#else

      switch (samples) {
      case 1:
         GEN_SAMPLE_POS_1X(ms.Sample);
         break;
      case 2:
         GEN_SAMPLE_POS_2X(ms.Sample);
         break;
      case 4:
         GEN_SAMPLE_POS_4X(ms.Sample);
         break;
      case 8:
         GEN_SAMPLE_POS_8X(ms.Sample);
         break;
      default:
         break;
      }
#endif
   }

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_SAMPLE_MASK), sm) {
      sm.SampleMask = sample_mask;
   }
}

static const uint32_t vk_to_gen_logic_op[] = {
   [VK_LOGIC_OP_COPY]                        = LOGICOP_COPY,
   [VK_LOGIC_OP_CLEAR]                       = LOGICOP_CLEAR,
   [VK_LOGIC_OP_AND]                         = LOGICOP_AND,
   [VK_LOGIC_OP_AND_REVERSE]                 = LOGICOP_AND_REVERSE,
   [VK_LOGIC_OP_AND_INVERTED]                = LOGICOP_AND_INVERTED,
   [VK_LOGIC_OP_NO_OP]                       = LOGICOP_NOOP,
   [VK_LOGIC_OP_XOR]                         = LOGICOP_XOR,
   [VK_LOGIC_OP_OR]                          = LOGICOP_OR,
   [VK_LOGIC_OP_NOR]                         = LOGICOP_NOR,
   [VK_LOGIC_OP_EQUIVALENT]                  = LOGICOP_EQUIV,
   [VK_LOGIC_OP_INVERT]                      = LOGICOP_INVERT,
   [VK_LOGIC_OP_OR_REVERSE]                  = LOGICOP_OR_REVERSE,
   [VK_LOGIC_OP_COPY_INVERTED]               = LOGICOP_COPY_INVERTED,
   [VK_LOGIC_OP_OR_INVERTED]                 = LOGICOP_OR_INVERTED,
   [VK_LOGIC_OP_NAND]                        = LOGICOP_NAND,
   [VK_LOGIC_OP_SET]                         = LOGICOP_SET,
};

static const uint32_t vk_to_gen_blend[] = {
   [VK_BLEND_FACTOR_ZERO]                    = BLENDFACTOR_ZERO,
   [VK_BLEND_FACTOR_ONE]                     = BLENDFACTOR_ONE,
   [VK_BLEND_FACTOR_SRC_COLOR]               = BLENDFACTOR_SRC_COLOR,
   [VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR]     = BLENDFACTOR_INV_SRC_COLOR,
   [VK_BLEND_FACTOR_DST_COLOR]               = BLENDFACTOR_DST_COLOR,
   [VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR]     = BLENDFACTOR_INV_DST_COLOR,
   [VK_BLEND_FACTOR_SRC_ALPHA]               = BLENDFACTOR_SRC_ALPHA,
   [VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA]     = BLENDFACTOR_INV_SRC_ALPHA,
   [VK_BLEND_FACTOR_DST_ALPHA]               = BLENDFACTOR_DST_ALPHA,
   [VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA]     = BLENDFACTOR_INV_DST_ALPHA,
   [VK_BLEND_FACTOR_CONSTANT_COLOR]          = BLENDFACTOR_CONST_COLOR,
   [VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR]= BLENDFACTOR_INV_CONST_COLOR,
   [VK_BLEND_FACTOR_CONSTANT_ALPHA]          = BLENDFACTOR_CONST_ALPHA,
   [VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA]= BLENDFACTOR_INV_CONST_ALPHA,
   [VK_BLEND_FACTOR_SRC_ALPHA_SATURATE]      = BLENDFACTOR_SRC_ALPHA_SATURATE,
   [VK_BLEND_FACTOR_SRC1_COLOR]              = BLENDFACTOR_SRC1_COLOR,
   [VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR]    = BLENDFACTOR_INV_SRC1_COLOR,
   [VK_BLEND_FACTOR_SRC1_ALPHA]              = BLENDFACTOR_SRC1_ALPHA,
   [VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA]    = BLENDFACTOR_INV_SRC1_ALPHA,
};

static const uint32_t vk_to_gen_blend_op[] = {
   [VK_BLEND_OP_ADD]                         = BLENDFUNCTION_ADD,
   [VK_BLEND_OP_SUBTRACT]                    = BLENDFUNCTION_SUBTRACT,
   [VK_BLEND_OP_REVERSE_SUBTRACT]            = BLENDFUNCTION_REVERSE_SUBTRACT,
   [VK_BLEND_OP_MIN]                         = BLENDFUNCTION_MIN,
   [VK_BLEND_OP_MAX]                         = BLENDFUNCTION_MAX,
};

static const uint32_t vk_to_gen_compare_op[] = {
   [VK_COMPARE_OP_NEVER]                        = PREFILTEROPNEVER,
   [VK_COMPARE_OP_LESS]                         = PREFILTEROPLESS,
   [VK_COMPARE_OP_EQUAL]                        = PREFILTEROPEQUAL,
   [VK_COMPARE_OP_LESS_OR_EQUAL]                = PREFILTEROPLEQUAL,
   [VK_COMPARE_OP_GREATER]                      = PREFILTEROPGREATER,
   [VK_COMPARE_OP_NOT_EQUAL]                    = PREFILTEROPNOTEQUAL,
   [VK_COMPARE_OP_GREATER_OR_EQUAL]             = PREFILTEROPGEQUAL,
   [VK_COMPARE_OP_ALWAYS]                       = PREFILTEROPALWAYS,
};

static const uint32_t vk_to_gen_stencil_op[] = {
   [VK_STENCIL_OP_KEEP]                         = STENCILOP_KEEP,
   [VK_STENCIL_OP_ZERO]                         = STENCILOP_ZERO,
   [VK_STENCIL_OP_REPLACE]                      = STENCILOP_REPLACE,
   [VK_STENCIL_OP_INCREMENT_AND_CLAMP]          = STENCILOP_INCRSAT,
   [VK_STENCIL_OP_DECREMENT_AND_CLAMP]          = STENCILOP_DECRSAT,
   [VK_STENCIL_OP_INVERT]                       = STENCILOP_INVERT,
   [VK_STENCIL_OP_INCREMENT_AND_WRAP]           = STENCILOP_INCR,
   [VK_STENCIL_OP_DECREMENT_AND_WRAP]           = STENCILOP_DECR,
};

/* This function sanitizes the VkStencilOpState by looking at the compare ops
 * and trying to determine whether or not a given stencil op can ever actually
 * occur.  Stencil ops which can never occur are set to VK_STENCIL_OP_KEEP.
 * This function returns true if, after sanitation, any of the stencil ops are
 * set to something other than VK_STENCIL_OP_KEEP.
 */
static bool
sanitize_stencil_face(VkStencilOpState *face,
                      VkCompareOp depthCompareOp)
{
   /* If compareOp is ALWAYS then the stencil test will never fail and failOp
    * will never happen.  Set failOp to KEEP in this case.
    */
   if (face->compareOp == VK_COMPARE_OP_ALWAYS)
      face->failOp = VK_STENCIL_OP_KEEP;

   /* If compareOp is NEVER or depthCompareOp is NEVER then one of the depth
    * or stencil tests will fail and passOp will never happen.
    */
   if (face->compareOp == VK_COMPARE_OP_NEVER ||
       depthCompareOp == VK_COMPARE_OP_NEVER)
      face->passOp = VK_STENCIL_OP_KEEP;

   /* If compareOp is NEVER or depthCompareOp is ALWAYS then either the
    * stencil test will fail or the depth test will pass.  In either case,
    * depthFailOp will never happen.
    */
   if (face->compareOp == VK_COMPARE_OP_NEVER ||
       depthCompareOp == VK_COMPARE_OP_ALWAYS)
      face->depthFailOp = VK_STENCIL_OP_KEEP;

   return face->failOp != VK_STENCIL_OP_KEEP ||
          face->depthFailOp != VK_STENCIL_OP_KEEP ||
          face->passOp != VK_STENCIL_OP_KEEP;
}

/* Intel hardware is fairly sensitive to whether or not depth/stencil writes
 * are enabled.  In the presence of discards, it's fairly easy to get into the
 * non-promoted case which means a fairly big performance hit.  From the Iron
 * Lake PRM, Vol 2, pt. 1, section 8.4.3.2, "Early Depth Test Cases":
 *
 *    "Non-promoted depth (N) is active whenever the depth test can be done
 *    early but it cannot determine whether or not to write source depth to
 *    the depth buffer, therefore the depth write must be performed post pixel
 *    shader. This includes cases where the pixel shader can kill pixels,
 *    including via sampler chroma key, as well as cases where the alpha test
 *    function is enabled, which kills pixels based on a programmable alpha
 *    test. In this case, even if the depth test fails, the pixel cannot be
 *    killed if a stencil write is indicated. Whether or not the stencil write
 *    happens depends on whether or not the pixel is killed later. In these
 *    cases if stencil test fails and stencil writes are off, the pixels can
 *    also be killed early. If stencil writes are enabled, the pixels must be
 *    treated as Computed depth (described above)."
 *
 * The same thing as mentioned in the stencil case can happen in the depth
 * case as well if it thinks it writes depth but, thanks to the depth test
 * being GL_EQUAL, the write doesn't actually matter.  A little extra work
 * up-front to try and disable depth and stencil writes can make a big
 * difference.
 *
 * Unfortunately, the way depth and stencil testing is specified, there are
 * many case where, regardless of depth/stencil writes being enabled, nothing
 * actually gets written due to some other bit of state being set.  This
 * function attempts to "sanitize" the depth stencil state and disable writes
 * and sometimes even testing whenever possible.
 */
static void
sanitize_ds_state(VkPipelineDepthStencilStateCreateInfo *state,
                  bool *stencilWriteEnable,
                  VkImageAspectFlags ds_aspects)
{
   *stencilWriteEnable = state->stencilTestEnable;

   /* If the depth test is disabled, we won't be writing anything. Make sure we
    * treat the test as always passing later on as well.
    *
    * Also, the Vulkan spec requires that if either depth or stencil is not
    * present, the pipeline is to act as if the test silently passes. In that
    * case we won't write either.
    */
   if (!state->depthTestEnable || !(ds_aspects & VK_IMAGE_ASPECT_DEPTH_BIT)) {
      state->depthWriteEnable = false;
      state->depthCompareOp = VK_COMPARE_OP_ALWAYS;
   }

   if (!(ds_aspects & VK_IMAGE_ASPECT_STENCIL_BIT)) {
      *stencilWriteEnable = false;
      state->front.compareOp = VK_COMPARE_OP_ALWAYS;
      state->back.compareOp = VK_COMPARE_OP_ALWAYS;
   }

   /* If the stencil test is enabled and always fails, then we will never get
    * to the depth test so we can just disable the depth test entirely.
    */
   if (state->stencilTestEnable &&
       state->front.compareOp == VK_COMPARE_OP_NEVER &&
       state->back.compareOp == VK_COMPARE_OP_NEVER) {
      state->depthTestEnable = false;
      state->depthWriteEnable = false;
   }

   /* If depthCompareOp is EQUAL then the value we would be writing to the
    * depth buffer is the same as the value that's already there so there's no
    * point in writing it.
    */
   if (state->depthCompareOp == VK_COMPARE_OP_EQUAL)
      state->depthWriteEnable = false;

   /* If the stencil ops are such that we don't actually ever modify the
    * stencil buffer, we should disable writes.
    */
   if (!sanitize_stencil_face(&state->front, state->depthCompareOp) &&
       !sanitize_stencil_face(&state->back, state->depthCompareOp))
      *stencilWriteEnable = false;

   /* If the depth test always passes and we never write out depth, that's the
    * same as if the depth test is disabled entirely.
    */
   if (state->depthCompareOp == VK_COMPARE_OP_ALWAYS &&
       !state->depthWriteEnable)
      state->depthTestEnable = false;

   /* If the stencil test always passes and we never write out stencil, that's
    * the same as if the stencil test is disabled entirely.
    */
   if (state->front.compareOp == VK_COMPARE_OP_ALWAYS &&
       state->back.compareOp == VK_COMPARE_OP_ALWAYS &&
       !*stencilWriteEnable)
      state->stencilTestEnable = false;
}

static void
emit_ds_state(struct anv_pipeline *pipeline,
              const VkPipelineDepthStencilStateCreateInfo *pCreateInfo,
              const struct anv_render_pass *pass,
              const struct anv_subpass *subpass)
{
#if GEN_GEN == 7
#  define depth_stencil_dw pipeline->gen7.depth_stencil_state
#elif GEN_GEN == 8
#  define depth_stencil_dw pipeline->gen8.wm_depth_stencil
#else
#  define depth_stencil_dw pipeline->gen9.wm_depth_stencil
#endif

   if (pCreateInfo == NULL) {
      /* We're going to OR this together with the dynamic state.  We need
       * to make sure it's initialized to something useful.
       */
      pipeline->writes_stencil = false;
      pipeline->stencil_test_enable = false;
      pipeline->writes_depth = false;
      pipeline->depth_test_enable = false;
      memset(depth_stencil_dw, 0, sizeof(depth_stencil_dw));
      return;
   }

   VkImageAspectFlags ds_aspects = 0;
   if (subpass->depth_stencil_attachment) {
      VkFormat depth_stencil_format =
         pass->attachments[subpass->depth_stencil_attachment->attachment].format;
      ds_aspects = vk_format_aspects(depth_stencil_format);
   }

   VkPipelineDepthStencilStateCreateInfo info = *pCreateInfo;
   sanitize_ds_state(&info, &pipeline->writes_stencil, ds_aspects);
   pipeline->stencil_test_enable = info.stencilTestEnable;
   pipeline->writes_depth = info.depthWriteEnable;
   pipeline->depth_test_enable = info.depthTestEnable;

   /* VkBool32 depthBoundsTestEnable; // optional (depth_bounds_test) */

#if GEN_GEN <= 7
   struct GENX(DEPTH_STENCIL_STATE) depth_stencil = {
#else
   struct GENX(3DSTATE_WM_DEPTH_STENCIL) depth_stencil = {
#endif
      .DepthTestEnable = info.depthTestEnable,
      .DepthBufferWriteEnable = info.depthWriteEnable,
      .DepthTestFunction = vk_to_gen_compare_op[info.depthCompareOp],
      .DoubleSidedStencilEnable = true,

      .StencilTestEnable = info.stencilTestEnable,
      .StencilFailOp = vk_to_gen_stencil_op[info.front.failOp],
      .StencilPassDepthPassOp = vk_to_gen_stencil_op[info.front.passOp],
      .StencilPassDepthFailOp = vk_to_gen_stencil_op[info.front.depthFailOp],
      .StencilTestFunction = vk_to_gen_compare_op[info.front.compareOp],
      .BackfaceStencilFailOp = vk_to_gen_stencil_op[info.back.failOp],
      .BackfaceStencilPassDepthPassOp = vk_to_gen_stencil_op[info.back.passOp],
      .BackfaceStencilPassDepthFailOp =vk_to_gen_stencil_op[info.back.depthFailOp],
      .BackfaceStencilTestFunction = vk_to_gen_compare_op[info.back.compareOp],
   };

#if GEN_GEN <= 7
   GENX(DEPTH_STENCIL_STATE_pack)(NULL, depth_stencil_dw, &depth_stencil);
#else
   GENX(3DSTATE_WM_DEPTH_STENCIL_pack)(NULL, depth_stencil_dw, &depth_stencil);
#endif
}

MAYBE_UNUSED static bool
is_dual_src_blend_factor(VkBlendFactor factor)
{
   return factor == VK_BLEND_FACTOR_SRC1_COLOR ||
          factor == VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR ||
          factor == VK_BLEND_FACTOR_SRC1_ALPHA ||
          factor == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
}

static void
emit_cb_state(struct anv_pipeline *pipeline,
              const VkPipelineColorBlendStateCreateInfo *info,
              const VkPipelineMultisampleStateCreateInfo *ms_info)
{
   struct anv_device *device = pipeline->device;
   const struct brw_wm_prog_data *wm_prog_data = get_wm_prog_data(pipeline);

   struct GENX(BLEND_STATE) blend_state = {
#if GEN_GEN >= 8
      .AlphaToCoverageEnable = ms_info && ms_info->alphaToCoverageEnable,
      .AlphaToOneEnable = ms_info && ms_info->alphaToOneEnable,
#endif
   };

   uint32_t surface_count = 0;
   struct anv_pipeline_bind_map *map;
   if (anv_pipeline_has_stage(pipeline, MESA_SHADER_FRAGMENT)) {
      map = &pipeline->shaders[MESA_SHADER_FRAGMENT]->bind_map;
      surface_count = map->surface_count;
   }

   const uint32_t num_dwords = GENX(BLEND_STATE_length) +
      GENX(BLEND_STATE_ENTRY_length) * surface_count;
   pipeline->blend_state =
      anv_state_pool_alloc(&device->dynamic_state_pool, num_dwords * 4, 64);

   bool has_writeable_rt = false;
   uint32_t *state_pos = pipeline->blend_state.map;
   state_pos += GENX(BLEND_STATE_length);
#if GEN_GEN >= 8
   struct GENX(BLEND_STATE_ENTRY) bs0 = { 0 };
#endif
   for (unsigned i = 0; i < surface_count; i++) {
      struct anv_pipeline_binding *binding = &map->surface_to_descriptor[i];

      /* All color attachments are at the beginning of the binding table */
      if (binding->set != ANV_DESCRIPTOR_SET_COLOR_ATTACHMENTS)
         break;

      /* We can have at most 8 attachments */
      assert(i < 8);

      if (info == NULL || binding->index >= info->attachmentCount) {
         /* Default everything to disabled */
         struct GENX(BLEND_STATE_ENTRY) entry = {
            .WriteDisableAlpha = true,
            .WriteDisableRed = true,
            .WriteDisableGreen = true,
            .WriteDisableBlue = true,
         };
         GENX(BLEND_STATE_ENTRY_pack)(NULL, state_pos, &entry);
         state_pos += GENX(BLEND_STATE_ENTRY_length);
         continue;
      }

      assert(binding->binding == 0);
      const VkPipelineColorBlendAttachmentState *a =
         &info->pAttachments[binding->index];

      struct GENX(BLEND_STATE_ENTRY) entry = {
#if GEN_GEN < 8
         .AlphaToCoverageEnable = ms_info && ms_info->alphaToCoverageEnable,
         .AlphaToOneEnable = ms_info && ms_info->alphaToOneEnable,
#endif
         .LogicOpEnable = info->logicOpEnable,
         .LogicOpFunction = vk_to_gen_logic_op[info->logicOp],
         .ColorBufferBlendEnable = a->blendEnable,
         .ColorClampRange = COLORCLAMP_RTFORMAT,
         .PreBlendColorClampEnable = true,
         .PostBlendColorClampEnable = true,
         .SourceBlendFactor = vk_to_gen_blend[a->srcColorBlendFactor],
         .DestinationBlendFactor = vk_to_gen_blend[a->dstColorBlendFactor],
         .ColorBlendFunction = vk_to_gen_blend_op[a->colorBlendOp],
         .SourceAlphaBlendFactor = vk_to_gen_blend[a->srcAlphaBlendFactor],
         .DestinationAlphaBlendFactor = vk_to_gen_blend[a->dstAlphaBlendFactor],
         .AlphaBlendFunction = vk_to_gen_blend_op[a->alphaBlendOp],
         .WriteDisableAlpha = !(a->colorWriteMask & VK_COLOR_COMPONENT_A_BIT),
         .WriteDisableRed = !(a->colorWriteMask & VK_COLOR_COMPONENT_R_BIT),
         .WriteDisableGreen = !(a->colorWriteMask & VK_COLOR_COMPONENT_G_BIT),
         .WriteDisableBlue = !(a->colorWriteMask & VK_COLOR_COMPONENT_B_BIT),
      };

      if (a->srcColorBlendFactor != a->srcAlphaBlendFactor ||
          a->dstColorBlendFactor != a->dstAlphaBlendFactor ||
          a->colorBlendOp != a->alphaBlendOp) {
#if GEN_GEN >= 8
         blend_state.IndependentAlphaBlendEnable = true;
#else
         entry.IndependentAlphaBlendEnable = true;
#endif
      }

      /* The Dual Source Blending documentation says:
       *
       * "If SRC1 is included in a src/dst blend factor and
       * a DualSource RT Write message is not used, results
       * are UNDEFINED. (This reflects the same restriction in DX APIs,
       * where undefined results are produced if “o1” is not written
       * by a PS – there are no default values defined)."
       *
       * There is no way to gracefully fix this undefined situation
       * so we just disable the blending to prevent possible issues.
       */
      if (!wm_prog_data->dual_src_blend &&
          (is_dual_src_blend_factor(a->srcColorBlendFactor) ||
           is_dual_src_blend_factor(a->dstColorBlendFactor) ||
           is_dual_src_blend_factor(a->srcAlphaBlendFactor) ||
           is_dual_src_blend_factor(a->dstAlphaBlendFactor))) {
         vk_debug_report(&device->instance->debug_report_callbacks,
                         VK_DEBUG_REPORT_WARNING_BIT_EXT,
                         VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT,
                         (uint64_t)(uintptr_t)device,
                         0, 0, "anv",
                         "Enabled dual-src blend factors without writing both targets "
                         "in the shader.  Disabling blending to avoid GPU hangs.");
         entry.ColorBufferBlendEnable = false;
      }

      if (a->colorWriteMask != 0)
         has_writeable_rt = true;

      /* Our hardware applies the blend factor prior to the blend function
       * regardless of what function is used.  Technically, this means the
       * hardware can do MORE than GL or Vulkan specify.  However, it also
       * means that, for MIN and MAX, we have to stomp the blend factor to
       * ONE to make it a no-op.
       */
      if (a->colorBlendOp == VK_BLEND_OP_MIN ||
          a->colorBlendOp == VK_BLEND_OP_MAX) {
         entry.SourceBlendFactor = BLENDFACTOR_ONE;
         entry.DestinationBlendFactor = BLENDFACTOR_ONE;
      }
      if (a->alphaBlendOp == VK_BLEND_OP_MIN ||
          a->alphaBlendOp == VK_BLEND_OP_MAX) {
         entry.SourceAlphaBlendFactor = BLENDFACTOR_ONE;
         entry.DestinationAlphaBlendFactor = BLENDFACTOR_ONE;
      }
      GENX(BLEND_STATE_ENTRY_pack)(NULL, state_pos, &entry);
      state_pos += GENX(BLEND_STATE_ENTRY_length);
#if GEN_GEN >= 8
      if (i == 0)
         bs0 = entry;
#endif
   }

#if GEN_GEN >= 8
   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_PS_BLEND), blend) {
      blend.AlphaToCoverageEnable         = blend_state.AlphaToCoverageEnable;
      blend.HasWriteableRT                = has_writeable_rt;
      blend.ColorBufferBlendEnable        = bs0.ColorBufferBlendEnable;
      blend.SourceAlphaBlendFactor        = bs0.SourceAlphaBlendFactor;
      blend.DestinationAlphaBlendFactor   = bs0.DestinationAlphaBlendFactor;
      blend.SourceBlendFactor             = bs0.SourceBlendFactor;
      blend.DestinationBlendFactor        = bs0.DestinationBlendFactor;
      blend.AlphaTestEnable               = false;
      blend.IndependentAlphaBlendEnable   =
         blend_state.IndependentAlphaBlendEnable;
   }
#else
   (void)has_writeable_rt;
#endif

   GENX(BLEND_STATE_pack)(NULL, pipeline->blend_state.map, &blend_state);
   anv_state_flush(device, pipeline->blend_state);

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_BLEND_STATE_POINTERS), bsp) {
      bsp.BlendStatePointer      = pipeline->blend_state.offset;
#if GEN_GEN >= 8
      bsp.BlendStatePointerValid = true;
#endif
   }
}

static void
emit_3dstate_clip(struct anv_pipeline *pipeline,
                  const VkPipelineViewportStateCreateInfo *vp_info,
                  const VkPipelineRasterizationStateCreateInfo *rs_info)
{
   const struct brw_wm_prog_data *wm_prog_data = get_wm_prog_data(pipeline);
   (void) wm_prog_data;
   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_CLIP), clip) {
      clip.ClipEnable               = true;
      clip.StatisticsEnable         = true;
      clip.EarlyCullEnable          = true;
      clip.APIMode                  = APIMODE_D3D,
      clip.ViewportXYClipTestEnable = true;

      clip.ClipMode = CLIPMODE_NORMAL;

      clip.TriangleStripListProvokingVertexSelect = 0;
      clip.LineStripListProvokingVertexSelect     = 0;
      clip.TriangleFanProvokingVertexSelect       = 1;

      clip.MinimumPointWidth = 0.125;
      clip.MaximumPointWidth = 255.875;

      const struct brw_vue_prog_data *last =
         anv_pipeline_get_last_vue_prog_data(pipeline);

      /* From the Vulkan 1.0.45 spec:
       *
       *    "If the last active vertex processing stage shader entry point's
       *    interface does not include a variable decorated with
       *    ViewportIndex, then the first viewport is used."
       */
      if (vp_info && (last->vue_map.slots_valid & VARYING_BIT_VIEWPORT)) {
         clip.MaximumVPIndex = vp_info->viewportCount - 1;
      } else {
         clip.MaximumVPIndex = 0;
      }

      /* From the Vulkan 1.0.45 spec:
       *
       *    "If the last active vertex processing stage shader entry point's
       *    interface does not include a variable decorated with Layer, then
       *    the first layer is used."
       */
      clip.ForceZeroRTAIndexEnable =
         !(last->vue_map.slots_valid & VARYING_BIT_LAYER);

#if GEN_GEN == 7
      clip.FrontWinding            = vk_to_gen_front_face[rs_info->frontFace];
      clip.CullMode                = vk_to_gen_cullmode[rs_info->cullMode];
      clip.ViewportZClipTestEnable = !pipeline->depth_clamp_enable;
      if (last) {
         clip.UserClipDistanceClipTestEnableBitmask = last->clip_distance_mask;
         clip.UserClipDistanceCullTestEnableBitmask = last->cull_distance_mask;
      }
#else
      clip.NonPerspectiveBarycentricEnable = wm_prog_data ?
         (wm_prog_data->barycentric_interp_modes &
          BRW_BARYCENTRIC_NONPERSPECTIVE_BITS) != 0 : 0;
#endif
   }
}

static void
emit_3dstate_streamout(struct anv_pipeline *pipeline,
                       const VkPipelineRasterizationStateCreateInfo *rs_info)
{
   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_STREAMOUT), so) {
      so.RenderingDisable = rs_info->rasterizerDiscardEnable;
   }
}

static uint32_t
get_sampler_count(const struct anv_shader_bin *bin)
{
   uint32_t count_by_4 = DIV_ROUND_UP(bin->bind_map.sampler_count, 4);

   /* We can potentially have way more than 32 samplers and that's ok.
    * However, the 3DSTATE_XS packets only have 3 bits to specify how
    * many to pre-fetch and all values above 4 are marked reserved.
    */
   return MIN2(count_by_4, 4);
}

static uint32_t
get_binding_table_entry_count(const struct anv_shader_bin *bin)
{
   return DIV_ROUND_UP(bin->bind_map.surface_count, 32);
}

static struct anv_address
get_scratch_address(struct anv_pipeline *pipeline,
                    gl_shader_stage stage,
                    const struct anv_shader_bin *bin)
{
   return (struct anv_address) {
      .bo = anv_scratch_pool_alloc(pipeline->device,
                                   &pipeline->device->scratch_pool,
                                   stage, bin->prog_data->total_scratch),
      .offset = 0,
   };
}

static uint32_t
get_scratch_space(const struct anv_shader_bin *bin)
{
   return ffs(bin->prog_data->total_scratch / 2048);
}

static void
emit_3dstate_vs(struct anv_pipeline *pipeline)
{
   const struct gen_device_info *devinfo = &pipeline->device->info;
   const struct brw_vs_prog_data *vs_prog_data = get_vs_prog_data(pipeline);
   const struct anv_shader_bin *vs_bin =
      pipeline->shaders[MESA_SHADER_VERTEX];

   assert(anv_pipeline_has_stage(pipeline, MESA_SHADER_VERTEX));

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_VS), vs) {
      vs.Enable               = true;
      vs.StatisticsEnable     = true;
      vs.KernelStartPointer   = vs_bin->kernel.offset;
#if GEN_GEN >= 8
      vs.SIMD8DispatchEnable  =
         vs_prog_data->base.dispatch_mode == DISPATCH_MODE_SIMD8;
#endif

      assert(!vs_prog_data->base.base.use_alt_mode);
#if GEN_GEN < 11
      vs.SingleVertexDispatch       = false;
#endif
      vs.VectorMaskEnable           = false;
      vs.SamplerCount               = get_sampler_count(vs_bin);
     /* Gen 11 workarounds table #2056 WABTPPrefetchDisable suggests to
      * disable prefetching of binding tables on A0 and B0 steppings.
      * TODO: Revisit this WA on newer steppings.
      */
      vs.BindingTableEntryCount     = GEN_GEN == 11 ? 0 : get_binding_table_entry_count(vs_bin);
      vs.FloatingPointMode          = IEEE754;
      vs.IllegalOpcodeExceptionEnable = false;
      vs.SoftwareExceptionEnable    = false;
      vs.MaximumNumberofThreads     = devinfo->max_vs_threads - 1;

      if (GEN_GEN == 9 && devinfo->gt == 4 &&
          anv_pipeline_has_stage(pipeline, MESA_SHADER_TESS_EVAL)) {
         /* On Sky Lake GT4, we have experienced some hangs related to the VS
          * cache and tessellation.  It is unknown exactly what is happening
          * but the Haswell docs for the "VS Reference Count Full Force Miss
          * Enable" field of the "Thread Mode" register refer to a HSW bug in
          * which the VUE handle reference count would overflow resulting in
          * internal reference counting bugs.  My (Jason's) best guess is that
          * this bug cropped back up on SKL GT4 when we suddenly had more
          * threads in play than any previous gen9 hardware.
          *
          * What we do know for sure is that setting this bit when
          * tessellation shaders are in use fixes a GPU hang in Batman: Arkham
          * City when playing with DXVK (https://bugs.freedesktop.org/107280).
          * Disabling the vertex cache with tessellation shaders should only
          * have a minor performance impact as the tessellation shaders are
          * likely generating and processing far more geometry than the vertex
          * stage.
          */
         vs.VertexCacheDisable = true;
      }

      vs.VertexURBEntryReadLength      = vs_prog_data->base.urb_read_length;
      vs.VertexURBEntryReadOffset      = 0;
      vs.DispatchGRFStartRegisterForURBData =
         vs_prog_data->base.base.dispatch_grf_start_reg;

#if GEN_GEN >= 8
      vs.UserClipDistanceClipTestEnableBitmask =
         vs_prog_data->base.clip_distance_mask;
      vs.UserClipDistanceCullTestEnableBitmask =
         vs_prog_data->base.cull_distance_mask;
#endif

      vs.PerThreadScratchSpace   = get_scratch_space(vs_bin);
      vs.ScratchSpaceBasePointer =
         get_scratch_address(pipeline, MESA_SHADER_VERTEX, vs_bin);
   }
}

static void
emit_3dstate_hs_te_ds(struct anv_pipeline *pipeline,
                      const VkPipelineTessellationStateCreateInfo *tess_info)
{
   if (!anv_pipeline_has_stage(pipeline, MESA_SHADER_TESS_EVAL)) {
      anv_batch_emit(&pipeline->batch, GENX(3DSTATE_HS), hs);
      anv_batch_emit(&pipeline->batch, GENX(3DSTATE_TE), te);
      anv_batch_emit(&pipeline->batch, GENX(3DSTATE_DS), ds);
      return;
   }

   const struct gen_device_info *devinfo = &pipeline->device->info;
   const struct anv_shader_bin *tcs_bin =
      pipeline->shaders[MESA_SHADER_TESS_CTRL];
   const struct anv_shader_bin *tes_bin =
      pipeline->shaders[MESA_SHADER_TESS_EVAL];

   const struct brw_tcs_prog_data *tcs_prog_data = get_tcs_prog_data(pipeline);
   const struct brw_tes_prog_data *tes_prog_data = get_tes_prog_data(pipeline);

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_HS), hs) {
      hs.Enable = true;
      hs.StatisticsEnable = true;
      hs.KernelStartPointer = tcs_bin->kernel.offset;

      hs.SamplerCount = get_sampler_count(tcs_bin);
      /* Gen 11 workarounds table #2056 WABTPPrefetchDisable */
      hs.BindingTableEntryCount = GEN_GEN == 11 ? 0 : get_binding_table_entry_count(tcs_bin);
      hs.MaximumNumberofThreads = devinfo->max_tcs_threads - 1;
      hs.IncludeVertexHandles = true;
      hs.InstanceCount = tcs_prog_data->instances - 1;

      hs.VertexURBEntryReadLength = 0;
      hs.VertexURBEntryReadOffset = 0;
      hs.DispatchGRFStartRegisterForURBData =
         tcs_prog_data->base.base.dispatch_grf_start_reg;

      hs.PerThreadScratchSpace = get_scratch_space(tcs_bin);
      hs.ScratchSpaceBasePointer =
         get_scratch_address(pipeline, MESA_SHADER_TESS_CTRL, tcs_bin);
   }

   const VkPipelineTessellationDomainOriginStateCreateInfoKHR *domain_origin_state =
      tess_info ? vk_find_struct_const(tess_info, PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO_KHR) : NULL;

   VkTessellationDomainOriginKHR uv_origin =
      domain_origin_state ? domain_origin_state->domainOrigin :
                            VK_TESSELLATION_DOMAIN_ORIGIN_UPPER_LEFT_KHR;

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_TE), te) {
      te.Partitioning = tes_prog_data->partitioning;

      if (uv_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT_KHR) {
         te.OutputTopology = tes_prog_data->output_topology;
      } else {
         /* When the origin is upper-left, we have to flip the winding order */
         if (tes_prog_data->output_topology == OUTPUT_TRI_CCW) {
            te.OutputTopology = OUTPUT_TRI_CW;
         } else if (tes_prog_data->output_topology == OUTPUT_TRI_CW) {
            te.OutputTopology = OUTPUT_TRI_CCW;
         } else {
            te.OutputTopology = tes_prog_data->output_topology;
         }
      }

      te.TEDomain = tes_prog_data->domain;
      te.TEEnable = true;
      te.MaximumTessellationFactorOdd = 63.0;
      te.MaximumTessellationFactorNotOdd = 64.0;
   }

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_DS), ds) {
      ds.Enable = true;
      ds.StatisticsEnable = true;
      ds.KernelStartPointer = tes_bin->kernel.offset;

      ds.SamplerCount = get_sampler_count(tes_bin);
      /* Gen 11 workarounds table #2056 WABTPPrefetchDisable */
      ds.BindingTableEntryCount = GEN_GEN == 11 ? 0 : get_binding_table_entry_count(tes_bin);
      ds.MaximumNumberofThreads = devinfo->max_tes_threads - 1;

      ds.ComputeWCoordinateEnable =
         tes_prog_data->domain == BRW_TESS_DOMAIN_TRI;

      ds.PatchURBEntryReadLength = tes_prog_data->base.urb_read_length;
      ds.PatchURBEntryReadOffset = 0;
      ds.DispatchGRFStartRegisterForURBData =
         tes_prog_data->base.base.dispatch_grf_start_reg;

#if GEN_GEN >= 8
#if GEN_GEN < 11
      ds.DispatchMode =
         tes_prog_data->base.dispatch_mode == DISPATCH_MODE_SIMD8 ?
            DISPATCH_MODE_SIMD8_SINGLE_PATCH :
            DISPATCH_MODE_SIMD4X2;
#else
      assert(tes_prog_data->base.dispatch_mode == DISPATCH_MODE_SIMD8);
      ds.DispatchMode = DISPATCH_MODE_SIMD8_SINGLE_PATCH;
#endif

      ds.UserClipDistanceClipTestEnableBitmask =
         tes_prog_data->base.clip_distance_mask;
      ds.UserClipDistanceCullTestEnableBitmask =
         tes_prog_data->base.cull_distance_mask;
#endif

      ds.PerThreadScratchSpace = get_scratch_space(tes_bin);
      ds.ScratchSpaceBasePointer =
         get_scratch_address(pipeline, MESA_SHADER_TESS_EVAL, tes_bin);
   }
}

static void
emit_3dstate_gs(struct anv_pipeline *pipeline)
{
   const struct gen_device_info *devinfo = &pipeline->device->info;
   const struct anv_shader_bin *gs_bin =
      pipeline->shaders[MESA_SHADER_GEOMETRY];

   if (!anv_pipeline_has_stage(pipeline, MESA_SHADER_GEOMETRY)) {
      anv_batch_emit(&pipeline->batch, GENX(3DSTATE_GS), gs);
      return;
   }

   const struct brw_gs_prog_data *gs_prog_data = get_gs_prog_data(pipeline);

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_GS), gs) {
      gs.Enable                  = true;
      gs.StatisticsEnable        = true;
      gs.KernelStartPointer      = gs_bin->kernel.offset;
      gs.DispatchMode            = gs_prog_data->base.dispatch_mode;

      gs.SingleProgramFlow       = false;
      gs.VectorMaskEnable        = false;
      gs.SamplerCount            = get_sampler_count(gs_bin);
      /* Gen 11 workarounds table #2056 WABTPPrefetchDisable */
      gs.BindingTableEntryCount  = GEN_GEN == 11 ? 0 : get_binding_table_entry_count(gs_bin);
      gs.IncludeVertexHandles    = gs_prog_data->base.include_vue_handles;
      gs.IncludePrimitiveID      = gs_prog_data->include_primitive_id;

      if (GEN_GEN == 8) {
         /* Broadwell is weird.  It needs us to divide by 2. */
         gs.MaximumNumberofThreads = devinfo->max_gs_threads / 2 - 1;
      } else {
         gs.MaximumNumberofThreads = devinfo->max_gs_threads - 1;
      }

      gs.OutputVertexSize        = gs_prog_data->output_vertex_size_hwords * 2 - 1;
      gs.OutputTopology          = gs_prog_data->output_topology;
      gs.VertexURBEntryReadLength = gs_prog_data->base.urb_read_length;
      gs.ControlDataFormat       = gs_prog_data->control_data_format;
      gs.ControlDataHeaderSize   = gs_prog_data->control_data_header_size_hwords;
      gs.InstanceControl         = MAX2(gs_prog_data->invocations, 1) - 1;
      gs.ReorderMode             = TRAILING;

#if GEN_GEN >= 8
      gs.ExpectedVertexCount     = gs_prog_data->vertices_in;
      gs.StaticOutput            = gs_prog_data->static_vertex_count >= 0;
      gs.StaticOutputVertexCount = gs_prog_data->static_vertex_count >= 0 ?
                                   gs_prog_data->static_vertex_count : 0;
#endif

      gs.VertexURBEntryReadOffset = 0;
      gs.VertexURBEntryReadLength = gs_prog_data->base.urb_read_length;
      gs.DispatchGRFStartRegisterForURBData =
         gs_prog_data->base.base.dispatch_grf_start_reg;

#if GEN_GEN >= 8
      gs.UserClipDistanceClipTestEnableBitmask =
         gs_prog_data->base.clip_distance_mask;
      gs.UserClipDistanceCullTestEnableBitmask =
         gs_prog_data->base.cull_distance_mask;
#endif

      gs.PerThreadScratchSpace   = get_scratch_space(gs_bin);
      gs.ScratchSpaceBasePointer =
         get_scratch_address(pipeline, MESA_SHADER_GEOMETRY, gs_bin);
   }
}

static bool
has_color_buffer_write_enabled(const struct anv_pipeline *pipeline,
                               const VkPipelineColorBlendStateCreateInfo *blend)
{
   const struct anv_shader_bin *shader_bin =
      pipeline->shaders[MESA_SHADER_FRAGMENT];
   if (!shader_bin)
      return false;

   const struct anv_pipeline_bind_map *bind_map = &shader_bin->bind_map;
   for (int i = 0; i < bind_map->surface_count; i++) {
      struct anv_pipeline_binding *binding = &bind_map->surface_to_descriptor[i];

      if (binding->set != ANV_DESCRIPTOR_SET_COLOR_ATTACHMENTS)
         continue;

      if (binding->index == UINT32_MAX)
         continue;

      if (blend && blend->pAttachments[binding->index].colorWriteMask != 0)
         return true;
   }

   return false;
}

static void
emit_3dstate_wm(struct anv_pipeline *pipeline, struct anv_subpass *subpass,
                const VkPipelineColorBlendStateCreateInfo *blend,
                const VkPipelineMultisampleStateCreateInfo *multisample)
{
   const struct brw_wm_prog_data *wm_prog_data = get_wm_prog_data(pipeline);

   MAYBE_UNUSED uint32_t samples =
      multisample ? multisample->rasterizationSamples : 1;

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_WM), wm) {
      wm.StatisticsEnable                    = true;
      wm.LineEndCapAntialiasingRegionWidth   = _05pixels;
      wm.LineAntialiasingRegionWidth         = _10pixels;
      wm.PointRasterizationRule              = RASTRULE_UPPER_RIGHT;

      if (anv_pipeline_has_stage(pipeline, MESA_SHADER_FRAGMENT)) {
         if (wm_prog_data->early_fragment_tests) {
            wm.EarlyDepthStencilControl         = EDSC_PREPS;
         } else if (wm_prog_data->has_side_effects) {
            wm.EarlyDepthStencilControl         = EDSC_PSEXEC;
         } else {
            wm.EarlyDepthStencilControl         = EDSC_NORMAL;
         }

#if GEN_GEN >= 8
         /* Gen8 hardware tries to compute ThreadDispatchEnable for us but
          * doesn't take into account KillPixels when no depth or stencil
          * writes are enabled.  In order for occlusion queries to work
          * correctly with no attachments, we need to force-enable PS thread
          * dispatch.
          *
          * The BDW docs are pretty clear that that this bit isn't validated
          * and probably shouldn't be used in production:
          *
          *    "This must always be set to Normal. This field should not be
          *    tested for functional validation."
          *
          * Unfortunately, however, the other mechanism we have for doing this
          * is 3DSTATE_PS_EXTRA::PixelShaderHasUAV which causes hangs on BDW.
          * Given two bad options, we choose the one which works.
          */
         if ((wm_prog_data->has_side_effects || wm_prog_data->uses_kill) &&
             !has_color_buffer_write_enabled(pipeline, blend))
            wm.ForceThreadDispatchEnable = ForceON;
#endif

         wm.BarycentricInterpolationMode =
            wm_prog_data->barycentric_interp_modes;

#if GEN_GEN < 8
         wm.PixelShaderComputedDepthMode  = wm_prog_data->computed_depth_mode;
         wm.PixelShaderUsesSourceDepth    = wm_prog_data->uses_src_depth;
         wm.PixelShaderUsesSourceW        = wm_prog_data->uses_src_w;
         wm.PixelShaderUsesInputCoverageMask = wm_prog_data->uses_sample_mask;

         /* If the subpass has a depth or stencil self-dependency, then we
          * need to force the hardware to do the depth/stencil write *after*
          * fragment shader execution.  Otherwise, the writes may hit memory
          * before we get around to fetching from the input attachment and we
          * may get the depth or stencil value from the current draw rather
          * than the previous one.
          */
         wm.PixelShaderKillsPixel         = subpass->has_ds_self_dep ||
                                            wm_prog_data->uses_kill;

         if (wm.PixelShaderComputedDepthMode != PSCDEPTH_OFF ||
             wm_prog_data->has_side_effects ||
             wm.PixelShaderKillsPixel ||
             has_color_buffer_write_enabled(pipeline, blend))
            wm.ThreadDispatchEnable = true;

         if (samples > 1) {
            wm.MultisampleRasterizationMode = MSRASTMODE_ON_PATTERN;
            if (wm_prog_data->persample_dispatch) {
               wm.MultisampleDispatchMode = MSDISPMODE_PERSAMPLE;
            } else {
               wm.MultisampleDispatchMode = MSDISPMODE_PERPIXEL;
            }
         } else {
            wm.MultisampleRasterizationMode = MSRASTMODE_OFF_PIXEL;
            wm.MultisampleDispatchMode = MSDISPMODE_PERSAMPLE;
         }
#endif
      }
   }
}

static void
emit_3dstate_ps(struct anv_pipeline *pipeline,
                const VkPipelineColorBlendStateCreateInfo *blend,
                const VkPipelineMultisampleStateCreateInfo *multisample)
{
   MAYBE_UNUSED const struct gen_device_info *devinfo = &pipeline->device->info;
   const struct anv_shader_bin *fs_bin =
      pipeline->shaders[MESA_SHADER_FRAGMENT];

   if (!anv_pipeline_has_stage(pipeline, MESA_SHADER_FRAGMENT)) {
      anv_batch_emit(&pipeline->batch, GENX(3DSTATE_PS), ps) {
#if GEN_GEN == 7
         /* Even if no fragments are ever dispatched, gen7 hardware hangs if
          * we don't at least set the maximum number of threads.
          */
         ps.MaximumNumberofThreads = devinfo->max_wm_threads - 1;
#endif
      }
      return;
   }

   const struct brw_wm_prog_data *wm_prog_data = get_wm_prog_data(pipeline);

#if GEN_GEN < 8
   /* The hardware wedges if you have this bit set but don't turn on any dual
    * source blend factors.
    */
   bool dual_src_blend = false;
   if (wm_prog_data->dual_src_blend && blend) {
      for (uint32_t i = 0; i < blend->attachmentCount; i++) {
         const VkPipelineColorBlendAttachmentState *bstate =
            &blend->pAttachments[i];

         if (bstate->blendEnable &&
             (is_dual_src_blend_factor(bstate->srcColorBlendFactor) ||
              is_dual_src_blend_factor(bstate->dstColorBlendFactor) ||
              is_dual_src_blend_factor(bstate->srcAlphaBlendFactor) ||
              is_dual_src_blend_factor(bstate->dstAlphaBlendFactor))) {
            dual_src_blend = true;
            break;
         }
      }
   }
#endif

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_PS), ps) {
      ps._8PixelDispatchEnable      = wm_prog_data->dispatch_8;
      ps._16PixelDispatchEnable     = wm_prog_data->dispatch_16;
      ps._32PixelDispatchEnable     = wm_prog_data->dispatch_32;

      /* From the Sky Lake PRM 3DSTATE_PS::32 Pixel Dispatch Enable:
       *
       *    "When NUM_MULTISAMPLES = 16 or FORCE_SAMPLE_COUNT = 16, SIMD32
       *    Dispatch must not be enabled for PER_PIXEL dispatch mode."
       *
       * Since 16x MSAA is first introduced on SKL, we don't need to apply
       * the workaround on any older hardware.
       */
      if (GEN_GEN >= 9 && !wm_prog_data->persample_dispatch &&
          multisample && multisample->rasterizationSamples == 16) {
         assert(ps._8PixelDispatchEnable || ps._16PixelDispatchEnable);
         ps._32PixelDispatchEnable = false;
      }

      ps.KernelStartPointer0 = fs_bin->kernel.offset +
                               brw_wm_prog_data_prog_offset(wm_prog_data, ps, 0);
      ps.KernelStartPointer1 = fs_bin->kernel.offset +
                               brw_wm_prog_data_prog_offset(wm_prog_data, ps, 1);
      ps.KernelStartPointer2 = fs_bin->kernel.offset +
                               brw_wm_prog_data_prog_offset(wm_prog_data, ps, 2);

      ps.SingleProgramFlow          = false;
      ps.VectorMaskEnable           = true;
      ps.SamplerCount               = get_sampler_count(fs_bin);
      /* Gen 11 workarounds table #2056 WABTPPrefetchDisable */
      ps.BindingTableEntryCount     = GEN_GEN == 11 ? 0 : get_binding_table_entry_count(fs_bin);
      ps.PushConstantEnable         = wm_prog_data->base.nr_params > 0 ||
                                      wm_prog_data->base.ubo_ranges[0].length;
      ps.PositionXYOffsetSelect     = wm_prog_data->uses_pos_offset ?
                                      POSOFFSET_SAMPLE: POSOFFSET_NONE;
#if GEN_GEN < 8
      ps.AttributeEnable            = wm_prog_data->num_varying_inputs > 0;
      ps.oMaskPresenttoRenderTarget = wm_prog_data->uses_omask;
      ps.DualSourceBlendEnable      = dual_src_blend;
#endif

#if GEN_IS_HASWELL
      /* Haswell requires the sample mask to be set in this packet as well
       * as in 3DSTATE_SAMPLE_MASK; the values should match.
       */
      ps.SampleMask                 = 0xff;
#endif

#if GEN_GEN >= 9
      ps.MaximumNumberofThreadsPerPSD  = 64 - 1;
#elif GEN_GEN >= 8
      ps.MaximumNumberofThreadsPerPSD  = 64 - 2;
#else
      ps.MaximumNumberofThreads        = devinfo->max_wm_threads - 1;
#endif

      ps.DispatchGRFStartRegisterForConstantSetupData0 =
         brw_wm_prog_data_dispatch_grf_start_reg(wm_prog_data, ps, 0);
      ps.DispatchGRFStartRegisterForConstantSetupData1 =
         brw_wm_prog_data_dispatch_grf_start_reg(wm_prog_data, ps, 1);
      ps.DispatchGRFStartRegisterForConstantSetupData2 =
         brw_wm_prog_data_dispatch_grf_start_reg(wm_prog_data, ps, 2);

      ps.PerThreadScratchSpace   = get_scratch_space(fs_bin);
      ps.ScratchSpaceBasePointer =
         get_scratch_address(pipeline, MESA_SHADER_FRAGMENT, fs_bin);
   }
}

#if GEN_GEN >= 8
static void
emit_3dstate_ps_extra(struct anv_pipeline *pipeline,
                      struct anv_subpass *subpass,
                      const VkPipelineColorBlendStateCreateInfo *blend)
{
   const struct brw_wm_prog_data *wm_prog_data = get_wm_prog_data(pipeline);

   if (!anv_pipeline_has_stage(pipeline, MESA_SHADER_FRAGMENT)) {
      anv_batch_emit(&pipeline->batch, GENX(3DSTATE_PS_EXTRA), ps);
      return;
   }

   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_PS_EXTRA), ps) {
      ps.PixelShaderValid              = true;
      ps.AttributeEnable               = wm_prog_data->num_varying_inputs > 0;
      ps.oMaskPresenttoRenderTarget    = wm_prog_data->uses_omask;
      ps.PixelShaderIsPerSample        = wm_prog_data->persample_dispatch;
      ps.PixelShaderComputedDepthMode  = wm_prog_data->computed_depth_mode;
      ps.PixelShaderUsesSourceDepth    = wm_prog_data->uses_src_depth;
      ps.PixelShaderUsesSourceW        = wm_prog_data->uses_src_w;

      /* If the subpass has a depth or stencil self-dependency, then we need
       * to force the hardware to do the depth/stencil write *after* fragment
       * shader execution.  Otherwise, the writes may hit memory before we get
       * around to fetching from the input attachment and we may get the depth
       * or stencil value from the current draw rather than the previous one.
       */
      ps.PixelShaderKillsPixel         = subpass->has_ds_self_dep ||
                                         wm_prog_data->uses_kill;

#if GEN_GEN >= 9
      ps.PixelShaderComputesStencil = wm_prog_data->computed_stencil;
      ps.PixelShaderPullsBary    = wm_prog_data->pulls_bary;

      ps.InputCoverageMaskState  = ICMS_NONE;
      if (wm_prog_data->uses_sample_mask) {
         if (wm_prog_data->post_depth_coverage)
            ps.InputCoverageMaskState  = ICMS_DEPTH_COVERAGE;
         else
            ps.InputCoverageMaskState  = ICMS_INNER_CONSERVATIVE;
      }
#else
      ps.PixelShaderUsesInputCoverageMask = wm_prog_data->uses_sample_mask;
#endif
   }
}

static void
emit_3dstate_vf_topology(struct anv_pipeline *pipeline)
{
   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_VF_TOPOLOGY), vft) {
      vft.PrimitiveTopologyType = pipeline->topology;
   }
}
#endif

static void
emit_3dstate_vf_statistics(struct anv_pipeline *pipeline)
{
   anv_batch_emit(&pipeline->batch, GENX(3DSTATE_VF_STATISTICS), vfs) {
      vfs.StatisticsEnable = true;
   }
}

static void
compute_kill_pixel(struct anv_pipeline *pipeline,
                   const VkPipelineMultisampleStateCreateInfo *ms_info,
                   const struct anv_subpass *subpass)
{
   if (!anv_pipeline_has_stage(pipeline, MESA_SHADER_FRAGMENT)) {
      pipeline->kill_pixel = false;
      return;
   }

   const struct brw_wm_prog_data *wm_prog_data = get_wm_prog_data(pipeline);

   /* This computes the KillPixel portion of the computation for whether or
    * not we want to enable the PMA fix on gen8 or gen9.  It's given by this
    * chunk of the giant formula:
    *
    *    (3DSTATE_PS_EXTRA::PixelShaderKillsPixels ||
    *     3DSTATE_PS_EXTRA::oMask Present to RenderTarget ||
    *     3DSTATE_PS_BLEND::AlphaToCoverageEnable ||
    *     3DSTATE_PS_BLEND::AlphaTestEnable ||
    *     3DSTATE_WM_CHROMAKEY::ChromaKeyKillEnable)
    *
    * 3DSTATE_WM_CHROMAKEY::ChromaKeyKillEnable is always false and so is
    * 3DSTATE_PS_BLEND::AlphaTestEnable since Vulkan doesn't have a concept
    * of an alpha test.
    */
   pipeline->kill_pixel =
      subpass->has_ds_self_dep || wm_prog_data->uses_kill ||
      wm_prog_data->uses_omask ||
      (ms_info && ms_info->alphaToCoverageEnable);
}

static VkResult
genX(graphics_pipeline_create)(
    VkDevice                                    _device,
    struct anv_pipeline_cache *                 cache,
    const VkGraphicsPipelineCreateInfo*         pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipeline)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_render_pass, pass, pCreateInfo->renderPass);
   struct anv_subpass *subpass = &pass->subpasses[pCreateInfo->subpass];
   struct anv_pipeline *pipeline;
   VkResult result;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO);

   /* Use the default pipeline cache if none is specified */
   if (cache == NULL && device->instance->pipeline_cache_enabled)
      cache = &device->default_pipeline_cache;

   pipeline = vk_alloc2(&device->alloc, pAllocator, sizeof(*pipeline), 8,
                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pipeline == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   result = anv_pipeline_init(pipeline, device, cache,
                              pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free2(&device->alloc, pAllocator, pipeline);
      return result;
   }

   assert(pCreateInfo->pVertexInputState);
   emit_vertex_input(pipeline, pCreateInfo->pVertexInputState);
   assert(pCreateInfo->pRasterizationState);
   emit_rs_state(pipeline, pCreateInfo->pRasterizationState,
                 pCreateInfo->pMultisampleState, pass, subpass);
   emit_ms_state(pipeline, pCreateInfo->pMultisampleState);
   emit_ds_state(pipeline, pCreateInfo->pDepthStencilState, pass, subpass);
   emit_cb_state(pipeline, pCreateInfo->pColorBlendState,
                           pCreateInfo->pMultisampleState);
   compute_kill_pixel(pipeline, pCreateInfo->pMultisampleState, subpass);

   emit_urb_setup(pipeline);

   emit_3dstate_clip(pipeline, pCreateInfo->pViewportState,
                     pCreateInfo->pRasterizationState);
   emit_3dstate_streamout(pipeline, pCreateInfo->pRasterizationState);

#if 0
   /* From gen7_vs_state.c */

   /**
    * From Graphics BSpec: 3D-Media-GPGPU Engine > 3D Pipeline Stages >
    * Geometry > Geometry Shader > State:
    *
    *     "Note: Because of corruption in IVB:GT2, software needs to flush the
    *     whole fixed function pipeline when the GS enable changes value in
    *     the 3DSTATE_GS."
    *
    * The hardware architects have clarified that in this context "flush the
    * whole fixed function pipeline" means to emit a PIPE_CONTROL with the "CS
    * Stall" bit set.
    */
   if (!device->info.is_haswell && !device->info.is_baytrail)
      gen7_emit_vs_workaround_flush(brw);
#endif

   emit_3dstate_vs(pipeline);
   emit_3dstate_hs_te_ds(pipeline, pCreateInfo->pTessellationState);
   emit_3dstate_gs(pipeline);
   emit_3dstate_sbe(pipeline);
   emit_3dstate_wm(pipeline, subpass, pCreateInfo->pColorBlendState,
                   pCreateInfo->pMultisampleState);
   emit_3dstate_ps(pipeline, pCreateInfo->pColorBlendState,
                   pCreateInfo->pMultisampleState);
#if GEN_GEN >= 8
   emit_3dstate_ps_extra(pipeline, subpass, pCreateInfo->pColorBlendState);
   emit_3dstate_vf_topology(pipeline);
#endif
   emit_3dstate_vf_statistics(pipeline);

   *pPipeline = anv_pipeline_to_handle(pipeline);

   return pipeline->batch.status;
}

static VkResult
compute_pipeline_create(
    VkDevice                                    _device,
    struct anv_pipeline_cache *                 cache,
    const VkComputePipelineCreateInfo*          pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipeline)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   const struct anv_physical_device *physical_device =
      &device->instance->physicalDevice;
   const struct gen_device_info *devinfo = &physical_device->info;
   struct anv_pipeline *pipeline;
   VkResult result;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);

   /* Use the default pipeline cache if none is specified */
   if (cache == NULL && device->instance->pipeline_cache_enabled)
      cache = &device->default_pipeline_cache;

   pipeline = vk_alloc2(&device->alloc, pAllocator, sizeof(*pipeline), 8,
                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pipeline == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   pipeline->device = device;

   pipeline->blend_state.map = NULL;

   result = anv_reloc_list_init(&pipeline->batch_relocs,
                                pAllocator ? pAllocator : &device->alloc);
   if (result != VK_SUCCESS) {
      vk_free2(&device->alloc, pAllocator, pipeline);
      return result;
   }
   pipeline->batch.next = pipeline->batch.start = pipeline->batch_data;
   pipeline->batch.end = pipeline->batch.start + sizeof(pipeline->batch_data);
   pipeline->batch.relocs = &pipeline->batch_relocs;
   pipeline->batch.status = VK_SUCCESS;

   /* When we free the pipeline, we detect stages based on the NULL status
    * of various prog_data pointers.  Make them NULL by default.
    */
   memset(pipeline->shaders, 0, sizeof(pipeline->shaders));

   pipeline->needs_data_cache = false;

   assert(pCreateInfo->stage.stage == VK_SHADER_STAGE_COMPUTE_BIT);
   pipeline->active_stages |= VK_SHADER_STAGE_COMPUTE_BIT;
   ANV_FROM_HANDLE(anv_shader_module, module,  pCreateInfo->stage.module);
   result = anv_pipeline_compile_cs(pipeline, cache, pCreateInfo, module,
                                    pCreateInfo->stage.pName,
                                    pCreateInfo->stage.pSpecializationInfo);
   if (result != VK_SUCCESS) {
      vk_free2(&device->alloc, pAllocator, pipeline);
      return result;
   }

   const struct brw_cs_prog_data *cs_prog_data = get_cs_prog_data(pipeline);

   anv_pipeline_setup_l3_config(pipeline, cs_prog_data->base.total_shared > 0);

   uint32_t group_size = cs_prog_data->local_size[0] *
      cs_prog_data->local_size[1] * cs_prog_data->local_size[2];
   uint32_t remainder = group_size & (cs_prog_data->simd_size - 1);

   if (remainder > 0)
      pipeline->cs_right_mask = ~0u >> (32 - remainder);
   else
      pipeline->cs_right_mask = ~0u >> (32 - cs_prog_data->simd_size);

   const uint32_t vfe_curbe_allocation =
      ALIGN(cs_prog_data->push.per_thread.regs * cs_prog_data->threads +
            cs_prog_data->push.cross_thread.regs, 2);

   const uint32_t subslices = MAX2(physical_device->subslice_total, 1);

   const struct anv_shader_bin *cs_bin =
      pipeline->shaders[MESA_SHADER_COMPUTE];

   anv_batch_emit(&pipeline->batch, GENX(MEDIA_VFE_STATE), vfe) {
#if GEN_GEN > 7
      vfe.StackSize              = 0;
#else
      vfe.GPGPUMode              = true;
#endif
      vfe.MaximumNumberofThreads =
         devinfo->max_cs_threads * subslices - 1;
      vfe.NumberofURBEntries     = GEN_GEN <= 7 ? 0 : 2;
#if GEN_GEN < 11
      vfe.ResetGatewayTimer      = true;
#endif
#if GEN_GEN <= 8
      vfe.BypassGatewayControl   = true;
#endif
      vfe.URBEntryAllocationSize = GEN_GEN <= 7 ? 0 : 2;
      vfe.CURBEAllocationSize    = vfe_curbe_allocation;

      vfe.PerThreadScratchSpace = get_scratch_space(cs_bin);
      vfe.ScratchSpaceBasePointer =
         get_scratch_address(pipeline, MESA_SHADER_COMPUTE, cs_bin);
   }

   struct GENX(INTERFACE_DESCRIPTOR_DATA) desc = {
      .KernelStartPointer     = cs_bin->kernel.offset,

      .SamplerCount           = get_sampler_count(cs_bin),
      /* Gen 11 workarounds table #2056 WABTPPrefetchDisable */
      .BindingTableEntryCount = GEN_GEN == 11 ? 0 : get_binding_table_entry_count(cs_bin),
      .BarrierEnable          = cs_prog_data->uses_barrier,
      .SharedLocalMemorySize  =
         encode_slm_size(GEN_GEN, cs_prog_data->base.total_shared),

#if !GEN_IS_HASWELL
      .ConstantURBEntryReadOffset = 0,
#endif
      .ConstantURBEntryReadLength = cs_prog_data->push.per_thread.regs,
#if GEN_GEN >= 8 || GEN_IS_HASWELL
      .CrossThreadConstantDataReadLength =
         cs_prog_data->push.cross_thread.regs,
#endif

      .NumberofThreadsinGPGPUThreadGroup = cs_prog_data->threads,
   };
   GENX(INTERFACE_DESCRIPTOR_DATA_pack)(NULL,
                                        pipeline->interface_descriptor_data,
                                        &desc);

   *pPipeline = anv_pipeline_to_handle(pipeline);

   return pipeline->batch.status;
}

VkResult genX(CreateGraphicsPipelines)(
    VkDevice                                    _device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    count,
    const VkGraphicsPipelineCreateInfo*         pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines)
{
   ANV_FROM_HANDLE(anv_pipeline_cache, pipeline_cache, pipelineCache);

   VkResult result = VK_SUCCESS;

   unsigned i;
   for (i = 0; i < count; i++) {
      result = genX(graphics_pipeline_create)(_device,
                                              pipeline_cache,
                                              &pCreateInfos[i],
                                              pAllocator, &pPipelines[i]);

      /* Bail out on the first error as it is not obvious what error should be
       * report upon 2 different failures. */
      if (result != VK_SUCCESS)
         break;
   }

   for (; i < count; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}

VkResult genX(CreateComputePipelines)(
    VkDevice                                    _device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    count,
    const VkComputePipelineCreateInfo*          pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines)
{
   ANV_FROM_HANDLE(anv_pipeline_cache, pipeline_cache, pipelineCache);

   VkResult result = VK_SUCCESS;

   unsigned i;
   for (i = 0; i < count; i++) {
      result = compute_pipeline_create(_device, pipeline_cache,
                                       &pCreateInfos[i],
                                       pAllocator, &pPipelines[i]);

      /* Bail out on the first error as it is not obvious what error should be
       * report upon 2 different failures. */
      if (result != VK_SUCCESS)
         break;
   }

   for (; i < count; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}
