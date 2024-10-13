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

#include "util/mesa-sha1.h"
#include "common/gen_l3_config.h"
#include "anv_private.h"
#include "compiler/brw_nir.h"
#include "anv_nir.h"
#include "spirv/nir_spirv.h"
#include "vk_util.h"

/* Needed for SWIZZLE macros */
#include "program/prog_instruction.h"

// Shader functions

VkResult anv_CreateShaderModule(
    VkDevice                                    _device,
    const VkShaderModuleCreateInfo*             pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkShaderModule*                             pShaderModule)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_shader_module *module;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
   assert(pCreateInfo->flags == 0);

   module = vk_alloc2(&device->alloc, pAllocator,
                       sizeof(*module) + pCreateInfo->codeSize, 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (module == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   module->size = pCreateInfo->codeSize;
   memcpy(module->data, pCreateInfo->pCode, module->size);

   _mesa_sha1_compute(module->data, module->size, module->sha1);

   *pShaderModule = anv_shader_module_to_handle(module);

   return VK_SUCCESS;
}

void anv_DestroyShaderModule(
    VkDevice                                    _device,
    VkShaderModule                              _module,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_shader_module, module, _module);

   if (!module)
      return;

   vk_free2(&device->alloc, pAllocator, module);
}

#define SPIR_V_MAGIC_NUMBER 0x07230203

static const uint64_t stage_to_debug[] = {
   [MESA_SHADER_VERTEX] = DEBUG_VS,
   [MESA_SHADER_TESS_CTRL] = DEBUG_TCS,
   [MESA_SHADER_TESS_EVAL] = DEBUG_TES,
   [MESA_SHADER_GEOMETRY] = DEBUG_GS,
   [MESA_SHADER_FRAGMENT] = DEBUG_WM,
   [MESA_SHADER_COMPUTE] = DEBUG_CS,
};

/* Eventually, this will become part of anv_CreateShader.  Unfortunately,
 * we can't do that yet because we don't have the ability to copy nir.
 */
static nir_shader *
anv_shader_compile_to_nir(struct anv_pipeline *pipeline,
                          void *mem_ctx,
                          const struct anv_shader_module *module,
                          const char *entrypoint_name,
                          gl_shader_stage stage,
                          const VkSpecializationInfo *spec_info)
{
   const struct anv_device *device = pipeline->device;

   const struct brw_compiler *compiler =
      device->instance->physicalDevice.compiler;
   const nir_shader_compiler_options *nir_options =
      compiler->glsl_compiler_options[stage].NirOptions;

   uint32_t *spirv = (uint32_t *) module->data;
   assert(spirv[0] == SPIR_V_MAGIC_NUMBER);
   assert(module->size % 4 == 0);

   uint32_t num_spec_entries = 0;
   struct nir_spirv_specialization *spec_entries = NULL;
   if (spec_info && spec_info->mapEntryCount > 0) {
      num_spec_entries = spec_info->mapEntryCount;
      spec_entries = malloc(num_spec_entries * sizeof(*spec_entries));
      for (uint32_t i = 0; i < num_spec_entries; i++) {
         VkSpecializationMapEntry entry = spec_info->pMapEntries[i];
         const void *data = spec_info->pData + entry.offset;
         assert(data + entry.size <= spec_info->pData + spec_info->dataSize);

         spec_entries[i].id = spec_info->pMapEntries[i].constantID;
         if (spec_info->dataSize == 8)
            spec_entries[i].data64 = *(const uint64_t *)data;
         else
            spec_entries[i].data32 = *(const uint32_t *)data;
      }
   }

   struct spirv_to_nir_options spirv_options = {
      .lower_workgroup_access_to_offsets = true,
      .caps = {
         .float64 = device->instance->physicalDevice.info.gen >= 8,
         .int64 = device->instance->physicalDevice.info.gen >= 8,
         .tessellation = true,
         .device_group = true,
         .draw_parameters = true,
         .image_write_without_format = true,
         .multiview = true,
         .variable_pointers = true,
         .storage_16bit = device->instance->physicalDevice.info.gen >= 8,
         .int16 = device->instance->physicalDevice.info.gen >= 8,
         .shader_viewport_index_layer = true,
         .subgroup_arithmetic = true,
         .subgroup_basic = true,
         .subgroup_ballot = true,
         .subgroup_quad = true,
         .subgroup_shuffle = true,
         .subgroup_vote = true,
         .stencil_export = device->instance->physicalDevice.info.gen >= 9,
         .storage_8bit = device->instance->physicalDevice.info.gen >= 8,
         .post_depth_coverage = device->instance->physicalDevice.info.gen >= 9,
      },
   };

   nir_function *entry_point =
      spirv_to_nir(spirv, module->size / 4,
                   spec_entries, num_spec_entries,
                   stage, entrypoint_name, &spirv_options, nir_options);
   nir_shader *nir = entry_point->shader;
   assert(nir->info.stage == stage);
   nir_validate_shader(nir, "after spirv_to_nir");
   ralloc_steal(mem_ctx, nir);

   free(spec_entries);

   if (unlikely(INTEL_DEBUG & stage_to_debug[stage])) {
      fprintf(stderr, "NIR (from SPIR-V) for %s shader:\n",
              gl_shader_stage_name(stage));
      nir_print_shader(nir, stderr);
   }

   /* We have to lower away local constant initializers right before we
    * inline functions.  That way they get properly initialized at the top
    * of the function and not at the top of its caller.
    */
   NIR_PASS_V(nir, nir_lower_constant_initializers, nir_var_local);
   NIR_PASS_V(nir, nir_lower_returns);
   NIR_PASS_V(nir, nir_inline_functions);
   NIR_PASS_V(nir, nir_copy_prop);

   /* Pick off the single entrypoint that we want */
   foreach_list_typed_safe(nir_function, func, node, &nir->functions) {
      if (func != entry_point)
         exec_node_remove(&func->node);
   }
   assert(exec_list_length(&nir->functions) == 1);

   /* Now that we've deleted all but the main function, we can go ahead and
    * lower the rest of the constant initializers.  We do this here so that
    * nir_remove_dead_variables and split_per_member_structs below see the
    * corresponding stores.
    */
   NIR_PASS_V(nir, nir_lower_constant_initializers, ~0);

   /* Split member structs.  We do this before lower_io_to_temporaries so that
    * it doesn't lower system values to temporaries by accident.
    */
   NIR_PASS_V(nir, nir_split_var_copies);
   NIR_PASS_V(nir, nir_split_per_member_structs);

   NIR_PASS_V(nir, nir_remove_dead_variables,
              nir_var_shader_in | nir_var_shader_out | nir_var_system_value);

   if (stage == MESA_SHADER_FRAGMENT)
      NIR_PASS_V(nir, nir_lower_wpos_center, pipeline->sample_shading_enable);

   NIR_PASS_V(nir, nir_propagate_invariant);
   NIR_PASS_V(nir, nir_lower_io_to_temporaries,
              entry_point->impl, true, false);

   /* Vulkan uses the separate-shader linking model */
   nir->info.separate_shader = true;

   nir = brw_preprocess_nir(compiler, nir);

   if (stage == MESA_SHADER_FRAGMENT)
      NIR_PASS_V(nir, anv_nir_lower_input_attachments);

   return nir;
}

void anv_DestroyPipeline(
    VkDevice                                    _device,
    VkPipeline                                  _pipeline,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_pipeline, pipeline, _pipeline);

   if (!pipeline)
      return;

   anv_reloc_list_finish(&pipeline->batch_relocs,
                         pAllocator ? pAllocator : &device->alloc);
   if (pipeline->blend_state.map)
      anv_state_pool_free(&device->dynamic_state_pool, pipeline->blend_state);

   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
      if (pipeline->shaders[s])
         anv_shader_bin_unref(device, pipeline->shaders[s]);
   }

   vk_free2(&device->alloc, pAllocator, pipeline);
}

static const uint32_t vk_to_gen_primitive_type[] = {
   [VK_PRIMITIVE_TOPOLOGY_POINT_LIST]                    = _3DPRIM_POINTLIST,
   [VK_PRIMITIVE_TOPOLOGY_LINE_LIST]                     = _3DPRIM_LINELIST,
   [VK_PRIMITIVE_TOPOLOGY_LINE_STRIP]                    = _3DPRIM_LINESTRIP,
   [VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST]                 = _3DPRIM_TRILIST,
   [VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP]                = _3DPRIM_TRISTRIP,
   [VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN]                  = _3DPRIM_TRIFAN,
   [VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY]      = _3DPRIM_LINELIST_ADJ,
   [VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY]     = _3DPRIM_LINESTRIP_ADJ,
   [VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY]  = _3DPRIM_TRILIST_ADJ,
   [VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY] = _3DPRIM_TRISTRIP_ADJ,
};

static void
populate_sampler_prog_key(const struct gen_device_info *devinfo,
                          struct brw_sampler_prog_key_data *key)
{
   /* Almost all multisampled textures are compressed.  The only time when we
    * don't compress a multisampled texture is for 16x MSAA with a surface
    * width greater than 8k which is a bit of an edge case.  Since the sampler
    * just ignores the MCS parameter to ld2ms when MCS is disabled, it's safe
    * to tell the compiler to always assume compression.
    */
   key->compressed_multisample_layout_mask = ~0;

   /* SkyLake added support for 16x MSAA.  With this came a new message for
    * reading from a 16x MSAA surface with compression.  The new message was
    * needed because now the MCS data is 64 bits instead of 32 or lower as is
    * the case for 8x, 4x, and 2x.  The key->msaa_16 bit-field controls which
    * message we use.  Fortunately, the 16x message works for 8x, 4x, and 2x
    * so we can just use it unconditionally.  This may not be quite as
    * efficient but it saves us from recompiling.
    */
   if (devinfo->gen >= 9)
      key->msaa_16 = ~0;

   /* XXX: Handle texture swizzle on HSW- */
   for (int i = 0; i < MAX_SAMPLERS; i++) {
      /* Assume color sampler, no swizzling. (Works for BDW+) */
      key->swizzles[i] = SWIZZLE_XYZW;
   }
}

static void
populate_vs_prog_key(const struct gen_device_info *devinfo,
                     struct brw_vs_prog_key *key)
{
   memset(key, 0, sizeof(*key));

   populate_sampler_prog_key(devinfo, &key->tex);

   /* XXX: Handle vertex input work-arounds */

   /* XXX: Handle sampler_prog_key */
}

static void
populate_tcs_prog_key(const struct gen_device_info *devinfo,
                      unsigned input_vertices,
                      struct brw_tcs_prog_key *key)
{
   memset(key, 0, sizeof(*key));

   populate_sampler_prog_key(devinfo, &key->tex);

   key->input_vertices = input_vertices;
}

static void
populate_tes_prog_key(const struct gen_device_info *devinfo,
                      struct brw_tes_prog_key *key)
{
   memset(key, 0, sizeof(*key));

   populate_sampler_prog_key(devinfo, &key->tex);
}

static void
populate_gs_prog_key(const struct gen_device_info *devinfo,
                     struct brw_gs_prog_key *key)
{
   memset(key, 0, sizeof(*key));

   populate_sampler_prog_key(devinfo, &key->tex);
}

static void
populate_wm_prog_key(const struct gen_device_info *devinfo,
                     const struct anv_subpass *subpass,
                     const VkPipelineMultisampleStateCreateInfo *ms_info,
                     struct brw_wm_prog_key *key)
{
   memset(key, 0, sizeof(*key));

   populate_sampler_prog_key(devinfo, &key->tex);

   /* We set this to 0 here and set to the actual value before we call
    * brw_compile_fs.
    */
   key->input_slots_valid = 0;

   /* Vulkan doesn't specify a default */
   key->high_quality_derivatives = false;

   /* XXX Vulkan doesn't appear to specify */
   key->clamp_fragment_color = false;

   assert(subpass->color_count <= MAX_RTS);
   for (uint32_t i = 0; i < subpass->color_count; i++) {
      if (subpass->color_attachments[i].attachment != VK_ATTACHMENT_UNUSED)
         key->color_outputs_valid |= (1 << i);
   }

   key->nr_color_regions = util_bitcount(key->color_outputs_valid);

   key->replicate_alpha = key->nr_color_regions > 1 &&
                          ms_info && ms_info->alphaToCoverageEnable;

   if (ms_info) {
      /* We should probably pull this out of the shader, but it's fairly
       * harmless to compute it and then let dead-code take care of it.
       */
      if (ms_info->rasterizationSamples > 1) {
         key->persample_interp =
            (ms_info->minSampleShading * ms_info->rasterizationSamples) > 1;
         key->multisample_fbo = true;
      }

      key->frag_coord_adds_sample_pos = ms_info->sampleShadingEnable;
   }
}

static void
populate_cs_prog_key(const struct gen_device_info *devinfo,
                     struct brw_cs_prog_key *key)
{
   memset(key, 0, sizeof(*key));

   populate_sampler_prog_key(devinfo, &key->tex);
}

struct anv_pipeline_stage {
   gl_shader_stage stage;

   const struct anv_shader_module *module;
   const char *entrypoint;
   const VkSpecializationInfo *spec_info;

   union brw_any_prog_key key;

   struct {
      gl_shader_stage stage;
      unsigned char sha1[20];
   } cache_key;

   nir_shader *nir;

   struct anv_pipeline_binding surface_to_descriptor[256];
   struct anv_pipeline_binding sampler_to_descriptor[256];
   struct anv_pipeline_bind_map bind_map;

   union brw_any_prog_data prog_data;
};

static void
anv_pipeline_hash_shader(struct mesa_sha1 *ctx,
                         struct anv_pipeline_stage *stage)
{
   _mesa_sha1_update(ctx, stage->module->sha1, sizeof(stage->module->sha1));
   _mesa_sha1_update(ctx, stage->entrypoint, strlen(stage->entrypoint));
   _mesa_sha1_update(ctx, &stage->stage, sizeof(stage->stage));
   if (stage->spec_info) {
      _mesa_sha1_update(ctx, stage->spec_info->pMapEntries,
                        stage->spec_info->mapEntryCount *
                        sizeof(*stage->spec_info->pMapEntries));
      _mesa_sha1_update(ctx, stage->spec_info->pData,
                        stage->spec_info->dataSize);
   }
   _mesa_sha1_update(ctx, &stage->key, brw_prog_key_size(stage->stage));
}

static void
anv_pipeline_hash_graphics(struct anv_pipeline *pipeline,
                           struct anv_pipeline_layout *layout,
                           struct anv_pipeline_stage *stages,
                           unsigned char *sha1_out)
{
   struct mesa_sha1 ctx;
   _mesa_sha1_init(&ctx);

   _mesa_sha1_update(&ctx, &pipeline->subpass->view_mask,
                     sizeof(pipeline->subpass->view_mask));

   if (layout)
      _mesa_sha1_update(&ctx, layout->sha1, sizeof(layout->sha1));

   const bool rba = pipeline->device->robust_buffer_access;
   _mesa_sha1_update(&ctx, &rba, sizeof(rba));

   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
      if (stages[s].entrypoint)
         anv_pipeline_hash_shader(&ctx, &stages[s]);
   }

   _mesa_sha1_final(&ctx, sha1_out);
}

static void
anv_pipeline_hash_compute(struct anv_pipeline *pipeline,
                          struct anv_pipeline_layout *layout,
                          struct anv_pipeline_stage *stage,
                          unsigned char *sha1_out)
{
   struct mesa_sha1 ctx;
   _mesa_sha1_init(&ctx);

   if (layout)
      _mesa_sha1_update(&ctx, layout->sha1, sizeof(layout->sha1));

   const bool rba = pipeline->device->robust_buffer_access;
   _mesa_sha1_update(&ctx, &rba, sizeof(rba));

   anv_pipeline_hash_shader(&ctx, stage);

   _mesa_sha1_final(&ctx, sha1_out);
}

static void
anv_pipeline_lower_nir(struct anv_pipeline *pipeline,
                       void *mem_ctx,
                       struct anv_pipeline_stage *stage,
                       struct anv_pipeline_layout *layout)
{
   const struct brw_compiler *compiler =
      pipeline->device->instance->physicalDevice.compiler;

   struct brw_stage_prog_data *prog_data = &stage->prog_data.base;
   nir_shader *nir = stage->nir;

   NIR_PASS_V(nir, anv_nir_lower_ycbcr_textures, layout);

   NIR_PASS_V(nir, anv_nir_lower_push_constants);

   if (nir->info.stage != MESA_SHADER_COMPUTE)
      NIR_PASS_V(nir, anv_nir_lower_multiview, pipeline->subpass->view_mask);

   if (nir->info.stage == MESA_SHADER_COMPUTE)
      prog_data->total_shared = nir->num_shared;

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   if (nir->num_uniforms > 0) {
      assert(prog_data->nr_params == 0);

      /* If the shader uses any push constants at all, we'll just give
       * them the maximum possible number
       */
      assert(nir->num_uniforms <= MAX_PUSH_CONSTANTS_SIZE);
      nir->num_uniforms = MAX_PUSH_CONSTANTS_SIZE;
      prog_data->nr_params += MAX_PUSH_CONSTANTS_SIZE / sizeof(float);
      prog_data->param = ralloc_array(mem_ctx, uint32_t, prog_data->nr_params);

      /* We now set the param values to be offsets into a
       * anv_push_constant_data structure.  Since the compiler doesn't
       * actually dereference any of the gl_constant_value pointers in the
       * params array, it doesn't really matter what we put here.
       */
      struct anv_push_constants *null_data = NULL;
      /* Fill out the push constants section of the param array */
      for (unsigned i = 0; i < MAX_PUSH_CONSTANTS_SIZE / sizeof(float); i++) {
         prog_data->param[i] = ANV_PARAM_PUSH(
            (uintptr_t)&null_data->client_data[i * sizeof(float)]);
      }
   }

   if (nir->info.num_ssbos > 0 || nir->info.num_images > 0)
      pipeline->needs_data_cache = true;

   NIR_PASS_V(nir, brw_nir_lower_image_load_store, compiler->devinfo);

   /* Apply the actual pipeline layout to UBOs, SSBOs, and textures */
   if (layout) {
      anv_nir_apply_pipeline_layout(&pipeline->device->instance->physicalDevice,
                                    pipeline->device->robust_buffer_access,
                                    layout, nir, prog_data,
                                    &stage->bind_map);
   }

   if (nir->info.stage != MESA_SHADER_COMPUTE)
      brw_nir_analyze_ubo_ranges(compiler, nir, NULL, prog_data->ubo_ranges);

   assert(nir->num_uniforms == prog_data->nr_params * 4);

   stage->nir = nir;
}

static void
anv_fill_binding_table(struct brw_stage_prog_data *prog_data, unsigned bias)
{
   prog_data->binding_table.size_bytes = 0;
   prog_data->binding_table.texture_start = bias;
   prog_data->binding_table.gather_texture_start = bias;
   prog_data->binding_table.ubo_start = bias;
   prog_data->binding_table.ssbo_start = bias;
   prog_data->binding_table.image_start = bias;
}

static void
anv_pipeline_link_vs(const struct brw_compiler *compiler,
                     struct anv_pipeline_stage *vs_stage,
                     struct anv_pipeline_stage *next_stage)
{
   anv_fill_binding_table(&vs_stage->prog_data.vs.base.base, 0);

   if (next_stage)
      brw_nir_link_shaders(compiler, &vs_stage->nir, &next_stage->nir);
}

static const unsigned *
anv_pipeline_compile_vs(const struct brw_compiler *compiler,
                        void *mem_ctx,
                        struct anv_pipeline_stage *vs_stage)
{
   brw_compute_vue_map(compiler->devinfo,
                       &vs_stage->prog_data.vs.base.vue_map,
                       vs_stage->nir->info.outputs_written,
                       vs_stage->nir->info.separate_shader);

   return brw_compile_vs(compiler, NULL, mem_ctx, &vs_stage->key.vs,
                         &vs_stage->prog_data.vs, vs_stage->nir, -1, NULL);
}

static void
merge_tess_info(struct shader_info *tes_info,
                const struct shader_info *tcs_info)
{
   /* The Vulkan 1.0.38 spec, section 21.1 Tessellator says:
    *
    *    "PointMode. Controls generation of points rather than triangles
    *     or lines. This functionality defaults to disabled, and is
    *     enabled if either shader stage includes the execution mode.
    *
    * and about Triangles, Quads, IsoLines, VertexOrderCw, VertexOrderCcw,
    * PointMode, SpacingEqual, SpacingFractionalEven, SpacingFractionalOdd,
    * and OutputVertices, it says:
    *
    *    "One mode must be set in at least one of the tessellation
    *     shader stages."
    *
    * So, the fields can be set in either the TCS or TES, but they must
    * agree if set in both.  Our backend looks at TES, so bitwise-or in
    * the values from the TCS.
    */
   assert(tcs_info->tess.tcs_vertices_out == 0 ||
          tes_info->tess.tcs_vertices_out == 0 ||
          tcs_info->tess.tcs_vertices_out == tes_info->tess.tcs_vertices_out);
   tes_info->tess.tcs_vertices_out |= tcs_info->tess.tcs_vertices_out;

   assert(tcs_info->tess.spacing == TESS_SPACING_UNSPECIFIED ||
          tes_info->tess.spacing == TESS_SPACING_UNSPECIFIED ||
          tcs_info->tess.spacing == tes_info->tess.spacing);
   tes_info->tess.spacing |= tcs_info->tess.spacing;

   assert(tcs_info->tess.primitive_mode == 0 ||
          tes_info->tess.primitive_mode == 0 ||
          tcs_info->tess.primitive_mode == tes_info->tess.primitive_mode);
   tes_info->tess.primitive_mode |= tcs_info->tess.primitive_mode;
   tes_info->tess.ccw |= tcs_info->tess.ccw;
   tes_info->tess.point_mode |= tcs_info->tess.point_mode;
}

static void
anv_pipeline_link_tcs(const struct brw_compiler *compiler,
                      struct anv_pipeline_stage *tcs_stage,
                      struct anv_pipeline_stage *tes_stage)
{
   assert(tes_stage && tes_stage->stage == MESA_SHADER_TESS_EVAL);

   anv_fill_binding_table(&tcs_stage->prog_data.tcs.base.base, 0);

   brw_nir_link_shaders(compiler, &tcs_stage->nir, &tes_stage->nir);

   nir_lower_patch_vertices(tes_stage->nir,
                            tcs_stage->nir->info.tess.tcs_vertices_out,
                            NULL);

   /* Copy TCS info into the TES info */
   merge_tess_info(&tes_stage->nir->info, &tcs_stage->nir->info);

   anv_fill_binding_table(&tcs_stage->prog_data.tcs.base.base, 0);
   anv_fill_binding_table(&tes_stage->prog_data.tes.base.base, 0);

   /* Whacking the key after cache lookup is a bit sketchy, but all of
    * this comes from the SPIR-V, which is part of the hash used for the
    * pipeline cache.  So it should be safe.
    */
   tcs_stage->key.tcs.tes_primitive_mode =
      tes_stage->nir->info.tess.primitive_mode;
   tcs_stage->key.tcs.quads_workaround =
      compiler->devinfo->gen < 9 &&
      tes_stage->nir->info.tess.primitive_mode == 7 /* GL_QUADS */ &&
      tes_stage->nir->info.tess.spacing == TESS_SPACING_EQUAL;
}

static const unsigned *
anv_pipeline_compile_tcs(const struct brw_compiler *compiler,
                         void *mem_ctx,
                         struct anv_pipeline_stage *tcs_stage,
                         struct anv_pipeline_stage *prev_stage)
{
   tcs_stage->key.tcs.outputs_written =
      tcs_stage->nir->info.outputs_written;
   tcs_stage->key.tcs.patch_outputs_written =
      tcs_stage->nir->info.patch_outputs_written;

   return brw_compile_tcs(compiler, NULL, mem_ctx, &tcs_stage->key.tcs,
                          &tcs_stage->prog_data.tcs, tcs_stage->nir,
                          -1, NULL);
}

static void
anv_pipeline_link_tes(const struct brw_compiler *compiler,
                      struct anv_pipeline_stage *tes_stage,
                      struct anv_pipeline_stage *next_stage)
{
   anv_fill_binding_table(&tes_stage->prog_data.tes.base.base, 0);

   if (next_stage)
      brw_nir_link_shaders(compiler, &tes_stage->nir, &next_stage->nir);
}

static const unsigned *
anv_pipeline_compile_tes(const struct brw_compiler *compiler,
                         void *mem_ctx,
                         struct anv_pipeline_stage *tes_stage,
                         struct anv_pipeline_stage *tcs_stage)
{
   tes_stage->key.tes.inputs_read =
      tcs_stage->nir->info.outputs_written;
   tes_stage->key.tes.patch_inputs_read =
      tcs_stage->nir->info.patch_outputs_written;

   return brw_compile_tes(compiler, NULL, mem_ctx, &tes_stage->key.tes,
                          &tcs_stage->prog_data.tcs.base.vue_map,
                          &tes_stage->prog_data.tes, tes_stage->nir,
                          NULL, -1, NULL);
}

static void
anv_pipeline_link_gs(const struct brw_compiler *compiler,
                     struct anv_pipeline_stage *gs_stage,
                     struct anv_pipeline_stage *next_stage)
{
   anv_fill_binding_table(&gs_stage->prog_data.gs.base.base, 0);

   if (next_stage)
      brw_nir_link_shaders(compiler, &gs_stage->nir, &next_stage->nir);
}

static const unsigned *
anv_pipeline_compile_gs(const struct brw_compiler *compiler,
                        void *mem_ctx,
                        struct anv_pipeline_stage *gs_stage,
                        struct anv_pipeline_stage *prev_stage)
{
   brw_compute_vue_map(compiler->devinfo,
                       &gs_stage->prog_data.gs.base.vue_map,
                       gs_stage->nir->info.outputs_written,
                       gs_stage->nir->info.separate_shader);

   return brw_compile_gs(compiler, NULL, mem_ctx, &gs_stage->key.gs,
                         &gs_stage->prog_data.gs, gs_stage->nir,
                         NULL, -1, NULL);
}

static void
anv_pipeline_link_fs(const struct brw_compiler *compiler,
                     struct anv_pipeline_stage *stage)
{
   unsigned num_rts = 0;
   const int max_rt = FRAG_RESULT_DATA7 - FRAG_RESULT_DATA0 + 1;
   struct anv_pipeline_binding rt_bindings[max_rt];
   nir_function_impl *impl = nir_shader_get_entrypoint(stage->nir);
   int rt_to_bindings[max_rt];
   memset(rt_to_bindings, -1, sizeof(rt_to_bindings));
   bool rt_used[max_rt];
   memset(rt_used, 0, sizeof(rt_used));

   /* Flag used render targets */
   nir_foreach_variable_safe(var, &stage->nir->outputs) {
      if (var->data.location < FRAG_RESULT_DATA0)
         continue;

      const unsigned rt = var->data.location - FRAG_RESULT_DATA0;
      /* Unused or out-of-bounds */
      if (rt >= MAX_RTS || !(stage->key.wm.color_outputs_valid & (1 << rt)))
         continue;

      const unsigned array_len =
         glsl_type_is_array(var->type) ? glsl_get_length(var->type) : 1;
      assert(rt + array_len <= max_rt);

      for (unsigned i = 0; i < array_len; i++)
         rt_used[rt + i] = true;
   }

   /* Set new, compacted, location */
   for (unsigned i = 0; i < max_rt; i++) {
      if (!rt_used[i])
         continue;

      rt_to_bindings[i] = num_rts;
      rt_bindings[rt_to_bindings[i]] = (struct anv_pipeline_binding) {
         .set = ANV_DESCRIPTOR_SET_COLOR_ATTACHMENTS,
         .binding = 0,
         .index = i,
      };
      num_rts++;
   }

   bool deleted_output = false;
   nir_foreach_variable_safe(var, &stage->nir->outputs) {
      if (var->data.location < FRAG_RESULT_DATA0)
         continue;

      const unsigned rt = var->data.location - FRAG_RESULT_DATA0;
      if (rt >= MAX_RTS ||
          !(stage->key.wm.color_outputs_valid & (1 << rt))) {
         /* Unused or out-of-bounds, throw it away */
         deleted_output = true;
         var->data.mode = nir_var_local;
         exec_node_remove(&var->node);
         exec_list_push_tail(&impl->locals, &var->node);
         continue;
      }

      /* Give it the new location */
      assert(rt_to_bindings[rt] != -1);
      var->data.location = rt_to_bindings[rt] + FRAG_RESULT_DATA0;
   }

   if (deleted_output)
      nir_fixup_deref_modes(stage->nir);

   if (num_rts == 0) {
      /* If we have no render targets, we need a null render target */
      rt_bindings[0] = (struct anv_pipeline_binding) {
         .set = ANV_DESCRIPTOR_SET_COLOR_ATTACHMENTS,
         .binding = 0,
         .index = UINT32_MAX,
      };
      num_rts = 1;
   }

   /* Now that we've determined the actual number of render targets, adjust
    * the key accordingly.
    */
   stage->key.wm.nr_color_regions = num_rts;
   stage->key.wm.color_outputs_valid = (1 << num_rts) - 1;

   assert(num_rts <= max_rt);
   assert(stage->bind_map.surface_count == 0);
   typed_memcpy(stage->bind_map.surface_to_descriptor,
                rt_bindings, num_rts);
   stage->bind_map.surface_count += num_rts;

   anv_fill_binding_table(&stage->prog_data.wm.base, 0);
}

static const unsigned *
anv_pipeline_compile_fs(const struct brw_compiler *compiler,
                        void *mem_ctx,
                        struct anv_pipeline_stage *fs_stage,
                        struct anv_pipeline_stage *prev_stage)
{
   /* TODO: we could set this to 0 based on the information in nir_shader, but
    * we need this before we call spirv_to_nir.
    */
   assert(prev_stage);
   fs_stage->key.wm.input_slots_valid =
      prev_stage->prog_data.vue.vue_map.slots_valid;

   const unsigned *code =
      brw_compile_fs(compiler, NULL, mem_ctx, &fs_stage->key.wm,
                     &fs_stage->prog_data.wm, fs_stage->nir,
                     NULL, -1, -1, -1, true, false, NULL, NULL);

   if (fs_stage->key.wm.nr_color_regions == 0 &&
       !fs_stage->prog_data.wm.has_side_effects &&
       !fs_stage->prog_data.wm.uses_kill &&
       fs_stage->prog_data.wm.computed_depth_mode == BRW_PSCDEPTH_OFF &&
       !fs_stage->prog_data.wm.computed_stencil) {
      /* This fragment shader has no outputs and no side effects.  Go ahead
       * and return the code pointer so we don't accidentally think the
       * compile failed but zero out prog_data which will set program_size to
       * zero and disable the stage.
       */
      memset(&fs_stage->prog_data, 0, sizeof(fs_stage->prog_data));
   }

   return code;
}

static VkResult
anv_pipeline_compile_graphics(struct anv_pipeline *pipeline,
                              struct anv_pipeline_cache *cache,
                              const VkGraphicsPipelineCreateInfo *info)
{
   const struct brw_compiler *compiler =
      pipeline->device->instance->physicalDevice.compiler;
   struct anv_pipeline_stage stages[MESA_SHADER_STAGES] = {};

   pipeline->active_stages = 0;

   VkResult result;
   for (uint32_t i = 0; i < info->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *sinfo = &info->pStages[i];
      gl_shader_stage stage = vk_to_mesa_shader_stage(sinfo->stage);

      pipeline->active_stages |= sinfo->stage;

      stages[stage].stage = stage;
      stages[stage].module = anv_shader_module_from_handle(sinfo->module);
      stages[stage].entrypoint = sinfo->pName;
      stages[stage].spec_info = sinfo->pSpecializationInfo;

      const struct gen_device_info *devinfo = &pipeline->device->info;
      switch (stage) {
      case MESA_SHADER_VERTEX:
         populate_vs_prog_key(devinfo, &stages[stage].key.vs);
         break;
      case MESA_SHADER_TESS_CTRL:
         populate_tcs_prog_key(devinfo,
                               info->pTessellationState->patchControlPoints,
                               &stages[stage].key.tcs);
         break;
      case MESA_SHADER_TESS_EVAL:
         populate_tes_prog_key(devinfo, &stages[stage].key.tes);
         break;
      case MESA_SHADER_GEOMETRY:
         populate_gs_prog_key(devinfo, &stages[stage].key.gs);
         break;
      case MESA_SHADER_FRAGMENT:
         populate_wm_prog_key(devinfo, pipeline->subpass,
                              info->pMultisampleState,
                              &stages[stage].key.wm);
         break;
      default:
         unreachable("Invalid graphics shader stage");
      }
   }

   if (pipeline->active_stages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
      pipeline->active_stages |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

   assert(pipeline->active_stages & VK_SHADER_STAGE_VERTEX_BIT);

   ANV_FROM_HANDLE(anv_pipeline_layout, layout, info->layout);

   unsigned char sha1[20];
   anv_pipeline_hash_graphics(pipeline, layout, stages, sha1);

   unsigned found = 0;
   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
      if (!stages[s].entrypoint)
         continue;

      stages[s].cache_key.stage = s;
      memcpy(stages[s].cache_key.sha1, sha1, sizeof(sha1));

      struct anv_shader_bin *bin =
         anv_device_search_for_kernel(pipeline->device, cache,
                                      &stages[s].cache_key,
                                      sizeof(stages[s].cache_key));
      if (bin) {
         found++;
         pipeline->shaders[s] = bin;
      }
   }

   if (found == __builtin_popcount(pipeline->active_stages)) {
      /* We found all our shaders in the cache.  We're done. */
      goto done;
   } else if (found > 0) {
      /* We found some but not all of our shaders.  This shouldn't happen
       * most of the time but it can if we have a partially populated
       * pipeline cache.
       */
      assert(found < __builtin_popcount(pipeline->active_stages));

      vk_debug_report(&pipeline->device->instance->debug_report_callbacks,
                      VK_DEBUG_REPORT_WARNING_BIT_EXT |
                      VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
                      VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT,
                      (uint64_t)(uintptr_t)cache,
                      0, 0, "anv",
                      "Found a partial pipeline in the cache.  This is "
                      "most likely caused by an incomplete pipeline cache "
                      "import or export");

      /* We're going to have to recompile anyway, so just throw away our
       * references to the shaders in the cache.  We'll get them out of the
       * cache again as part of the compilation process.
       */
      for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
         if (pipeline->shaders[s]) {
            anv_shader_bin_unref(pipeline->device, pipeline->shaders[s]);
            pipeline->shaders[s] = NULL;
         }
      }
   }

   void *pipeline_ctx = ralloc_context(NULL);

   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
      if (!stages[s].entrypoint)
         continue;

      assert(stages[s].stage == s);
      assert(pipeline->shaders[s] == NULL);

      stages[s].bind_map = (struct anv_pipeline_bind_map) {
         .surface_to_descriptor = stages[s].surface_to_descriptor,
         .sampler_to_descriptor = stages[s].sampler_to_descriptor
      };

      stages[s].nir = anv_shader_compile_to_nir(pipeline, pipeline_ctx,
                                                stages[s].module,
                                                stages[s].entrypoint,
                                                stages[s].stage,
                                                stages[s].spec_info);
      if (stages[s].nir == NULL) {
         result = vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
         goto fail;
      }
   }

   /* Walk backwards to link */
   struct anv_pipeline_stage *next_stage = NULL;
   for (int s = MESA_SHADER_STAGES - 1; s >= 0; s--) {
      if (!stages[s].entrypoint)
         continue;

      switch (s) {
      case MESA_SHADER_VERTEX:
         anv_pipeline_link_vs(compiler, &stages[s], next_stage);
         break;
      case MESA_SHADER_TESS_CTRL:
         anv_pipeline_link_tcs(compiler, &stages[s], next_stage);
         break;
      case MESA_SHADER_TESS_EVAL:
         anv_pipeline_link_tes(compiler, &stages[s], next_stage);
         break;
      case MESA_SHADER_GEOMETRY:
         anv_pipeline_link_gs(compiler, &stages[s], next_stage);
         break;
      case MESA_SHADER_FRAGMENT:
         anv_pipeline_link_fs(compiler, &stages[s]);
         break;
      default:
         unreachable("Invalid graphics shader stage");
      }

      next_stage = &stages[s];
   }

   struct anv_pipeline_stage *prev_stage = NULL;
   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
      if (!stages[s].entrypoint)
         continue;

      void *stage_ctx = ralloc_context(NULL);

      anv_pipeline_lower_nir(pipeline, stage_ctx, &stages[s], layout);

      const unsigned *code;
      switch (s) {
      case MESA_SHADER_VERTEX:
         code = anv_pipeline_compile_vs(compiler, stage_ctx, &stages[s]);
         break;
      case MESA_SHADER_TESS_CTRL:
         code = anv_pipeline_compile_tcs(compiler, stage_ctx,
                                         &stages[s], prev_stage);
         break;
      case MESA_SHADER_TESS_EVAL:
         code = anv_pipeline_compile_tes(compiler, stage_ctx,
                                         &stages[s], prev_stage);
         break;
      case MESA_SHADER_GEOMETRY:
         code = anv_pipeline_compile_gs(compiler, stage_ctx,
                                        &stages[s], prev_stage);
         break;
      case MESA_SHADER_FRAGMENT:
         code = anv_pipeline_compile_fs(compiler, stage_ctx,
                                        &stages[s], prev_stage);
         break;
      default:
         unreachable("Invalid graphics shader stage");
      }
      if (code == NULL) {
         ralloc_free(stage_ctx);
         result = vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
         goto fail;
      }

      struct anv_shader_bin *bin =
         anv_device_upload_kernel(pipeline->device, cache,
                                  &stages[s].cache_key,
                                  sizeof(stages[s].cache_key),
                                  code, stages[s].prog_data.base.program_size,
                                  stages[s].nir->constant_data,
                                  stages[s].nir->constant_data_size,
                                  &stages[s].prog_data.base,
                                  brw_prog_data_size(s),
                                  &stages[s].bind_map);
      if (!bin) {
         ralloc_free(stage_ctx);
         result = vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
         goto fail;
      }

      pipeline->shaders[s] = bin;
      ralloc_free(stage_ctx);

      prev_stage = &stages[s];
   }

   ralloc_free(pipeline_ctx);

done:

   if (pipeline->shaders[MESA_SHADER_FRAGMENT] &&
       pipeline->shaders[MESA_SHADER_FRAGMENT]->prog_data->program_size == 0) {
      /* This can happen if we decided to implicitly disable the fragment
       * shader.  See anv_pipeline_compile_fs().
       */
      anv_shader_bin_unref(pipeline->device,
                           pipeline->shaders[MESA_SHADER_FRAGMENT]);
      pipeline->shaders[MESA_SHADER_FRAGMENT] = NULL;
      pipeline->active_stages &= ~VK_SHADER_STAGE_FRAGMENT_BIT;
   }

   return VK_SUCCESS;

fail:
   ralloc_free(pipeline_ctx);

   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
      if (pipeline->shaders[s])
         anv_shader_bin_unref(pipeline->device, pipeline->shaders[s]);
   }

   return result;
}

VkResult
anv_pipeline_compile_cs(struct anv_pipeline *pipeline,
                        struct anv_pipeline_cache *cache,
                        const VkComputePipelineCreateInfo *info,
                        const struct anv_shader_module *module,
                        const char *entrypoint,
                        const VkSpecializationInfo *spec_info)
{
   const struct brw_compiler *compiler =
      pipeline->device->instance->physicalDevice.compiler;

   struct anv_pipeline_stage stage = {
      .stage = MESA_SHADER_COMPUTE,
      .module = module,
      .entrypoint = entrypoint,
      .spec_info = spec_info,
      .cache_key = {
         .stage = MESA_SHADER_COMPUTE,
      }
   };

   struct anv_shader_bin *bin = NULL;

   populate_cs_prog_key(&pipeline->device->info, &stage.key.cs);

   ANV_FROM_HANDLE(anv_pipeline_layout, layout, info->layout);

   anv_pipeline_hash_compute(pipeline, layout, &stage, stage.cache_key.sha1);
   bin = anv_device_search_for_kernel(pipeline->device, cache, &stage.cache_key,
                                      sizeof(stage.cache_key));

   if (bin == NULL) {
      stage.bind_map = (struct anv_pipeline_bind_map) {
         .surface_to_descriptor = stage.surface_to_descriptor,
         .sampler_to_descriptor = stage.sampler_to_descriptor
      };

      void *mem_ctx = ralloc_context(NULL);

      stage.nir = anv_shader_compile_to_nir(pipeline, mem_ctx,
                                            stage.module,
                                            stage.entrypoint,
                                            stage.stage,
                                            stage.spec_info);
      if (stage.nir == NULL) {
         ralloc_free(mem_ctx);
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
      }

      anv_pipeline_lower_nir(pipeline, mem_ctx, &stage, layout);

      NIR_PASS_V(stage.nir, anv_nir_add_base_work_group_id,
                 &stage.prog_data.cs);

      anv_fill_binding_table(&stage.prog_data.cs.base, 1);

      const unsigned *shader_code =
         brw_compile_cs(compiler, NULL, mem_ctx, &stage.key.cs,
                        &stage.prog_data.cs, stage.nir, -1, NULL);
      if (shader_code == NULL) {
         ralloc_free(mem_ctx);
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
      }

      const unsigned code_size = stage.prog_data.base.program_size;
      bin = anv_device_upload_kernel(pipeline->device, cache,
                                     &stage.cache_key, sizeof(stage.cache_key),
                                     shader_code, code_size,
                                     stage.nir->constant_data,
                                     stage.nir->constant_data_size,
                                     &stage.prog_data.base,
                                     sizeof(stage.prog_data.cs),
                                     &stage.bind_map);
      if (!bin) {
         ralloc_free(mem_ctx);
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
      }

      ralloc_free(mem_ctx);
   }

   pipeline->active_stages = VK_SHADER_STAGE_COMPUTE_BIT;
   pipeline->shaders[MESA_SHADER_COMPUTE] = bin;

   return VK_SUCCESS;
}

/**
 * Copy pipeline state not marked as dynamic.
 * Dynamic state is pipeline state which hasn't been provided at pipeline
 * creation time, but is dynamically provided afterwards using various
 * vkCmdSet* functions.
 *
 * The set of state considered "non_dynamic" is determined by the pieces of
 * state that have their corresponding VkDynamicState enums omitted from
 * VkPipelineDynamicStateCreateInfo::pDynamicStates.
 *
 * @param[out] pipeline    Destination non_dynamic state.
 * @param[in]  pCreateInfo Source of non_dynamic state to be copied.
 */
static void
copy_non_dynamic_state(struct anv_pipeline *pipeline,
                       const VkGraphicsPipelineCreateInfo *pCreateInfo)
{
   anv_cmd_dirty_mask_t states = ANV_CMD_DIRTY_DYNAMIC_ALL;
   struct anv_subpass *subpass = pipeline->subpass;

   pipeline->dynamic_state = default_dynamic_state;

   if (pCreateInfo->pDynamicState) {
      /* Remove all of the states that are marked as dynamic */
      uint32_t count = pCreateInfo->pDynamicState->dynamicStateCount;
      for (uint32_t s = 0; s < count; s++)
         states &= ~(1 << pCreateInfo->pDynamicState->pDynamicStates[s]);
   }

   struct anv_dynamic_state *dynamic = &pipeline->dynamic_state;

   /* Section 9.2 of the Vulkan 1.0.15 spec says:
    *
    *    pViewportState is [...] NULL if the pipeline
    *    has rasterization disabled.
    */
   if (!pCreateInfo->pRasterizationState->rasterizerDiscardEnable) {
      assert(pCreateInfo->pViewportState);

      dynamic->viewport.count = pCreateInfo->pViewportState->viewportCount;
      if (states & (1 << VK_DYNAMIC_STATE_VIEWPORT)) {
         typed_memcpy(dynamic->viewport.viewports,
                     pCreateInfo->pViewportState->pViewports,
                     pCreateInfo->pViewportState->viewportCount);
      }

      dynamic->scissor.count = pCreateInfo->pViewportState->scissorCount;
      if (states & (1 << VK_DYNAMIC_STATE_SCISSOR)) {
         typed_memcpy(dynamic->scissor.scissors,
                     pCreateInfo->pViewportState->pScissors,
                     pCreateInfo->pViewportState->scissorCount);
      }
   }

   if (states & (1 << VK_DYNAMIC_STATE_LINE_WIDTH)) {
      assert(pCreateInfo->pRasterizationState);
      dynamic->line_width = pCreateInfo->pRasterizationState->lineWidth;
   }

   if (states & (1 << VK_DYNAMIC_STATE_DEPTH_BIAS)) {
      assert(pCreateInfo->pRasterizationState);
      dynamic->depth_bias.bias =
         pCreateInfo->pRasterizationState->depthBiasConstantFactor;
      dynamic->depth_bias.clamp =
         pCreateInfo->pRasterizationState->depthBiasClamp;
      dynamic->depth_bias.slope =
         pCreateInfo->pRasterizationState->depthBiasSlopeFactor;
   }

   /* Section 9.2 of the Vulkan 1.0.15 spec says:
    *
    *    pColorBlendState is [...] NULL if the pipeline has rasterization
    *    disabled or if the subpass of the render pass the pipeline is
    *    created against does not use any color attachments.
    */
   bool uses_color_att = false;
   for (unsigned i = 0; i < subpass->color_count; ++i) {
      if (subpass->color_attachments[i].attachment != VK_ATTACHMENT_UNUSED) {
         uses_color_att = true;
         break;
      }
   }

   if (uses_color_att &&
       !pCreateInfo->pRasterizationState->rasterizerDiscardEnable) {
      assert(pCreateInfo->pColorBlendState);

      if (states & (1 << VK_DYNAMIC_STATE_BLEND_CONSTANTS))
         typed_memcpy(dynamic->blend_constants,
                     pCreateInfo->pColorBlendState->blendConstants, 4);
   }

   /* If there is no depthstencil attachment, then don't read
    * pDepthStencilState. The Vulkan spec states that pDepthStencilState may
    * be NULL in this case. Even if pDepthStencilState is non-NULL, there is
    * no need to override the depthstencil defaults in
    * anv_pipeline::dynamic_state when there is no depthstencil attachment.
    *
    * Section 9.2 of the Vulkan 1.0.15 spec says:
    *
    *    pDepthStencilState is [...] NULL if the pipeline has rasterization
    *    disabled or if the subpass of the render pass the pipeline is created
    *    against does not use a depth/stencil attachment.
    */
   if (!pCreateInfo->pRasterizationState->rasterizerDiscardEnable &&
       subpass->depth_stencil_attachment) {
      assert(pCreateInfo->pDepthStencilState);

      if (states & (1 << VK_DYNAMIC_STATE_DEPTH_BOUNDS)) {
         dynamic->depth_bounds.min =
            pCreateInfo->pDepthStencilState->minDepthBounds;
         dynamic->depth_bounds.max =
            pCreateInfo->pDepthStencilState->maxDepthBounds;
      }

      if (states & (1 << VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK)) {
         dynamic->stencil_compare_mask.front =
            pCreateInfo->pDepthStencilState->front.compareMask;
         dynamic->stencil_compare_mask.back =
            pCreateInfo->pDepthStencilState->back.compareMask;
      }

      if (states & (1 << VK_DYNAMIC_STATE_STENCIL_WRITE_MASK)) {
         dynamic->stencil_write_mask.front =
            pCreateInfo->pDepthStencilState->front.writeMask;
         dynamic->stencil_write_mask.back =
            pCreateInfo->pDepthStencilState->back.writeMask;
      }

      if (states & (1 << VK_DYNAMIC_STATE_STENCIL_REFERENCE)) {
         dynamic->stencil_reference.front =
            pCreateInfo->pDepthStencilState->front.reference;
         dynamic->stencil_reference.back =
            pCreateInfo->pDepthStencilState->back.reference;
      }
   }

   pipeline->dynamic_state_mask = states;
}

static void
anv_pipeline_validate_create_info(const VkGraphicsPipelineCreateInfo *info)
{
#ifdef DEBUG
   struct anv_render_pass *renderpass = NULL;
   struct anv_subpass *subpass = NULL;

   /* Assert that all required members of VkGraphicsPipelineCreateInfo are
    * present.  See the Vulkan 1.0.28 spec, Section 9.2 Graphics Pipelines.
    */
   assert(info->sType == VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO);

   renderpass = anv_render_pass_from_handle(info->renderPass);
   assert(renderpass);

   assert(info->subpass < renderpass->subpass_count);
   subpass = &renderpass->subpasses[info->subpass];

   assert(info->stageCount >= 1);
   assert(info->pVertexInputState);
   assert(info->pInputAssemblyState);
   assert(info->pRasterizationState);
   if (!info->pRasterizationState->rasterizerDiscardEnable) {
      assert(info->pViewportState);
      assert(info->pMultisampleState);

      if (subpass && subpass->depth_stencil_attachment)
         assert(info->pDepthStencilState);

      if (subpass && subpass->color_count > 0) {
         bool all_color_unused = true;
         for (int i = 0; i < subpass->color_count; i++) {
            if (subpass->color_attachments[i].attachment != VK_ATTACHMENT_UNUSED)
               all_color_unused = false;
         }
         /* pColorBlendState is ignored if the pipeline has rasterization
          * disabled or if the subpass of the render pass the pipeline is
          * created against does not use any color attachments.
          */
         assert(info->pColorBlendState || all_color_unused);
      }
   }

   for (uint32_t i = 0; i < info->stageCount; ++i) {
      switch (info->pStages[i].stage) {
      case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
      case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
         assert(info->pTessellationState);
         break;
      default:
         break;
      }
   }
#endif
}

/**
 * Calculate the desired L3 partitioning based on the current state of the
 * pipeline.  For now this simply returns the conservative defaults calculated
 * by get_default_l3_weights(), but we could probably do better by gathering
 * more statistics from the pipeline state (e.g. guess of expected URB usage
 * and bound surfaces), or by using feed-back from performance counters.
 */
void
anv_pipeline_setup_l3_config(struct anv_pipeline *pipeline, bool needs_slm)
{
   const struct gen_device_info *devinfo = &pipeline->device->info;

   const struct gen_l3_weights w =
      gen_get_default_l3_weights(devinfo, pipeline->needs_data_cache, needs_slm);

   pipeline->urb.l3_config = gen_get_l3_config(devinfo, w);
   pipeline->urb.total_size =
      gen_get_l3_config_urb_size(devinfo, pipeline->urb.l3_config);
}

VkResult
anv_pipeline_init(struct anv_pipeline *pipeline,
                  struct anv_device *device,
                  struct anv_pipeline_cache *cache,
                  const VkGraphicsPipelineCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *alloc)
{
   VkResult result;

   anv_pipeline_validate_create_info(pCreateInfo);

   if (alloc == NULL)
      alloc = &device->alloc;

   pipeline->device = device;

   ANV_FROM_HANDLE(anv_render_pass, render_pass, pCreateInfo->renderPass);
   assert(pCreateInfo->subpass < render_pass->subpass_count);
   pipeline->subpass = &render_pass->subpasses[pCreateInfo->subpass];

   result = anv_reloc_list_init(&pipeline->batch_relocs, alloc);
   if (result != VK_SUCCESS)
      return result;

   pipeline->batch.alloc = alloc;
   pipeline->batch.next = pipeline->batch.start = pipeline->batch_data;
   pipeline->batch.end = pipeline->batch.start + sizeof(pipeline->batch_data);
   pipeline->batch.relocs = &pipeline->batch_relocs;
   pipeline->batch.status = VK_SUCCESS;

   copy_non_dynamic_state(pipeline, pCreateInfo);
   pipeline->depth_clamp_enable = pCreateInfo->pRasterizationState &&
                                  pCreateInfo->pRasterizationState->depthClampEnable;

   pipeline->sample_shading_enable = pCreateInfo->pMultisampleState &&
                                     pCreateInfo->pMultisampleState->sampleShadingEnable;

   pipeline->needs_data_cache = false;

   /* When we free the pipeline, we detect stages based on the NULL status
    * of various prog_data pointers.  Make them NULL by default.
    */
   memset(pipeline->shaders, 0, sizeof(pipeline->shaders));

   result = anv_pipeline_compile_graphics(pipeline, cache, pCreateInfo);
   if (result != VK_SUCCESS) {
      anv_reloc_list_finish(&pipeline->batch_relocs, alloc);
      return result;
   }

   assert(pipeline->shaders[MESA_SHADER_VERTEX]);

   anv_pipeline_setup_l3_config(pipeline, false);

   const VkPipelineVertexInputStateCreateInfo *vi_info =
      pCreateInfo->pVertexInputState;

   const uint64_t inputs_read = get_vs_prog_data(pipeline)->inputs_read;

   pipeline->vb_used = 0;
   for (uint32_t i = 0; i < vi_info->vertexAttributeDescriptionCount; i++) {
      const VkVertexInputAttributeDescription *desc =
         &vi_info->pVertexAttributeDescriptions[i];

      if (inputs_read & (1ull << (VERT_ATTRIB_GENERIC0 + desc->location)))
         pipeline->vb_used |= 1 << desc->binding;
   }

   for (uint32_t i = 0; i < vi_info->vertexBindingDescriptionCount; i++) {
      const VkVertexInputBindingDescription *desc =
         &vi_info->pVertexBindingDescriptions[i];

      pipeline->vb[desc->binding].stride = desc->stride;

      /* Step rate is programmed per vertex element (attribute), not
       * binding. Set up a map of which bindings step per instance, for
       * reference by vertex element setup. */
      switch (desc->inputRate) {
      default:
      case VK_VERTEX_INPUT_RATE_VERTEX:
         pipeline->vb[desc->binding].instanced = false;
         break;
      case VK_VERTEX_INPUT_RATE_INSTANCE:
         pipeline->vb[desc->binding].instanced = true;
         break;
      }

      pipeline->vb[desc->binding].instance_divisor = 1;
   }

   const VkPipelineVertexInputDivisorStateCreateInfoEXT *vi_div_state =
      vk_find_struct_const(vi_info->pNext,
                           PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT);
   if (vi_div_state) {
      for (uint32_t i = 0; i < vi_div_state->vertexBindingDivisorCount; i++) {
         const VkVertexInputBindingDivisorDescriptionEXT *desc =
            &vi_div_state->pVertexBindingDivisors[i];

         pipeline->vb[desc->binding].instance_divisor = desc->divisor;
      }
   }

   /* Our implementation of VK_KHR_multiview uses instancing to draw the
    * different views.  If the client asks for instancing, we need to multiply
    * the instance divisor by the number of views ensure that we repeat the
    * client's per-instance data once for each view.
    */
   if (pipeline->subpass->view_mask) {
      const uint32_t view_count = anv_subpass_view_count(pipeline->subpass);
      for (uint32_t vb = 0; vb < MAX_VBS; vb++) {
         if (pipeline->vb[vb].instanced)
            pipeline->vb[vb].instance_divisor *= view_count;
      }
   }

   const VkPipelineInputAssemblyStateCreateInfo *ia_info =
      pCreateInfo->pInputAssemblyState;
   const VkPipelineTessellationStateCreateInfo *tess_info =
      pCreateInfo->pTessellationState;
   pipeline->primitive_restart = ia_info->primitiveRestartEnable;

   if (anv_pipeline_has_stage(pipeline, MESA_SHADER_TESS_EVAL))
      pipeline->topology = _3DPRIM_PATCHLIST(tess_info->patchControlPoints);
   else
      pipeline->topology = vk_to_gen_primitive_type[ia_info->topology];

   return VK_SUCCESS;
}
