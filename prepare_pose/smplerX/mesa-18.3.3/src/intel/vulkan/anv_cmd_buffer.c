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

#include "vk_format_info.h"

/** \file anv_cmd_buffer.c
 *
 * This file contains all of the stuff for emitting commands into a command
 * buffer.  This includes implementations of most of the vkCmd*
 * entrypoints.  This file is concerned entirely with state emission and
 * not with the command buffer data structure itself.  As far as this file
 * is concerned, most of anv_cmd_buffer is magic.
 */

/* TODO: These are taken from GLES.  We should check the Vulkan spec */
const struct anv_dynamic_state default_dynamic_state = {
   .viewport = {
      .count = 0,
   },
   .scissor = {
      .count = 0,
   },
   .line_width = 1.0f,
   .depth_bias = {
      .bias = 0.0f,
      .clamp = 0.0f,
      .slope = 0.0f,
   },
   .blend_constants = { 0.0f, 0.0f, 0.0f, 0.0f },
   .depth_bounds = {
      .min = 0.0f,
      .max = 1.0f,
   },
   .stencil_compare_mask = {
      .front = ~0u,
      .back = ~0u,
   },
   .stencil_write_mask = {
      .front = ~0u,
      .back = ~0u,
   },
   .stencil_reference = {
      .front = 0u,
      .back = 0u,
   },
};

void
anv_dynamic_state_copy(struct anv_dynamic_state *dest,
                       const struct anv_dynamic_state *src,
                       uint32_t copy_mask)
{
   if (copy_mask & (1 << VK_DYNAMIC_STATE_VIEWPORT)) {
      dest->viewport.count = src->viewport.count;
      typed_memcpy(dest->viewport.viewports, src->viewport.viewports,
                   src->viewport.count);
   }

   if (copy_mask & (1 << VK_DYNAMIC_STATE_SCISSOR)) {
      dest->scissor.count = src->scissor.count;
      typed_memcpy(dest->scissor.scissors, src->scissor.scissors,
                   src->scissor.count);
   }

   if (copy_mask & (1 << VK_DYNAMIC_STATE_LINE_WIDTH))
      dest->line_width = src->line_width;

   if (copy_mask & (1 << VK_DYNAMIC_STATE_DEPTH_BIAS))
      dest->depth_bias = src->depth_bias;

   if (copy_mask & (1 << VK_DYNAMIC_STATE_BLEND_CONSTANTS))
      typed_memcpy(dest->blend_constants, src->blend_constants, 4);

   if (copy_mask & (1 << VK_DYNAMIC_STATE_DEPTH_BOUNDS))
      dest->depth_bounds = src->depth_bounds;

   if (copy_mask & (1 << VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK))
      dest->stencil_compare_mask = src->stencil_compare_mask;

   if (copy_mask & (1 << VK_DYNAMIC_STATE_STENCIL_WRITE_MASK))
      dest->stencil_write_mask = src->stencil_write_mask;

   if (copy_mask & (1 << VK_DYNAMIC_STATE_STENCIL_REFERENCE))
      dest->stencil_reference = src->stencil_reference;
}

static void
anv_cmd_state_init(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_cmd_state *state = &cmd_buffer->state;

   memset(state, 0, sizeof(*state));

   state->current_pipeline = UINT32_MAX;
   state->restart_index = UINT32_MAX;
   state->gfx.dynamic = default_dynamic_state;
}

static void
anv_cmd_pipeline_state_finish(struct anv_cmd_buffer *cmd_buffer,
                              struct anv_cmd_pipeline_state *pipe_state)
{
   for (uint32_t i = 0; i < ARRAY_SIZE(pipe_state->push_descriptors); i++)
      vk_free(&cmd_buffer->pool->alloc, pipe_state->push_descriptors[i]);
}

static void
anv_cmd_state_finish(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_cmd_state *state = &cmd_buffer->state;

   anv_cmd_pipeline_state_finish(cmd_buffer, &state->gfx.base);
   anv_cmd_pipeline_state_finish(cmd_buffer, &state->compute.base);

   for (uint32_t i = 0; i < MESA_SHADER_STAGES; i++)
      vk_free(&cmd_buffer->pool->alloc, state->push_constants[i]);

   vk_free(&cmd_buffer->pool->alloc, state->attachments);
}

static void
anv_cmd_state_reset(struct anv_cmd_buffer *cmd_buffer)
{
   anv_cmd_state_finish(cmd_buffer);
   anv_cmd_state_init(cmd_buffer);
}

/**
 * This function updates the size of the push constant buffer we need to emit.
 * This is called in various parts of the driver to ensure that different
 * pieces of push constant data get emitted as needed. However, it is important
 * that we never shrink the size of the buffer. For example, a compute shader
 * dispatch will always call this for the base group id, which has an
 * offset in the push constant buffer that is smaller than the offset for
 * storage image data. If the compute shader has storage images, we will call
 * this again with a larger size during binding table emission. However,
 * if we dispatch the compute shader again without dirtying our descriptors,
 * we would still call this function with a smaller size for the base group
 * id, and not for the images, which would incorrectly shrink the size of the
 * push constant data we emit with that dispatch, making us drop the image data.
 */
VkResult
anv_cmd_buffer_ensure_push_constants_size(struct anv_cmd_buffer *cmd_buffer,
                                          gl_shader_stage stage, uint32_t size)
{
   struct anv_push_constants **ptr = &cmd_buffer->state.push_constants[stage];

   if (*ptr == NULL) {
      *ptr = vk_alloc(&cmd_buffer->pool->alloc, size, 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (*ptr == NULL) {
         anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_HOST_MEMORY);
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      (*ptr)->size = size;
   } else if ((*ptr)->size < size) {
      *ptr = vk_realloc(&cmd_buffer->pool->alloc, *ptr, size, 8,
                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (*ptr == NULL) {
         anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_HOST_MEMORY);
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      (*ptr)->size = size;
   }

   return VK_SUCCESS;
}

static VkResult anv_create_cmd_buffer(
    struct anv_device *                         device,
    struct anv_cmd_pool *                       pool,
    VkCommandBufferLevel                        level,
    VkCommandBuffer*                            pCommandBuffer)
{
   struct anv_cmd_buffer *cmd_buffer;
   VkResult result;

   cmd_buffer = vk_alloc(&pool->alloc, sizeof(*cmd_buffer), 8,
                          VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (cmd_buffer == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   cmd_buffer->batch.status = VK_SUCCESS;

   cmd_buffer->_loader_data.loaderMagic = ICD_LOADER_MAGIC;
   cmd_buffer->device = device;
   cmd_buffer->pool = pool;
   cmd_buffer->level = level;

   result = anv_cmd_buffer_init_batch_bo_chain(cmd_buffer);
   if (result != VK_SUCCESS)
      goto fail;

   anv_state_stream_init(&cmd_buffer->surface_state_stream,
                         &device->surface_state_pool, 4096);
   anv_state_stream_init(&cmd_buffer->dynamic_state_stream,
                         &device->dynamic_state_pool, 16384);

   anv_cmd_state_init(cmd_buffer);

   if (pool) {
      list_addtail(&cmd_buffer->pool_link, &pool->cmd_buffers);
   } else {
      /* Init the pool_link so we can safefly call list_del when we destroy
       * the command buffer
       */
      list_inithead(&cmd_buffer->pool_link);
   }

   *pCommandBuffer = anv_cmd_buffer_to_handle(cmd_buffer);

   return VK_SUCCESS;

 fail:
   vk_free(&cmd_buffer->pool->alloc, cmd_buffer);

   return result;
}

VkResult anv_AllocateCommandBuffers(
    VkDevice                                    _device,
    const VkCommandBufferAllocateInfo*          pAllocateInfo,
    VkCommandBuffer*                            pCommandBuffers)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_cmd_pool, pool, pAllocateInfo->commandPool);

   VkResult result = VK_SUCCESS;
   uint32_t i;

   for (i = 0; i < pAllocateInfo->commandBufferCount; i++) {
      result = anv_create_cmd_buffer(device, pool, pAllocateInfo->level,
                                     &pCommandBuffers[i]);
      if (result != VK_SUCCESS)
         break;
   }

   if (result != VK_SUCCESS) {
      anv_FreeCommandBuffers(_device, pAllocateInfo->commandPool,
                             i, pCommandBuffers);
      for (i = 0; i < pAllocateInfo->commandBufferCount; i++)
         pCommandBuffers[i] = VK_NULL_HANDLE;
   }

   return result;
}

static void
anv_cmd_buffer_destroy(struct anv_cmd_buffer *cmd_buffer)
{
   list_del(&cmd_buffer->pool_link);

   anv_cmd_buffer_fini_batch_bo_chain(cmd_buffer);

   anv_state_stream_finish(&cmd_buffer->surface_state_stream);
   anv_state_stream_finish(&cmd_buffer->dynamic_state_stream);

   anv_cmd_state_finish(cmd_buffer);

   vk_free(&cmd_buffer->pool->alloc, cmd_buffer);
}

void anv_FreeCommandBuffers(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    uint32_t                                    commandBufferCount,
    const VkCommandBuffer*                      pCommandBuffers)
{
   for (uint32_t i = 0; i < commandBufferCount; i++) {
      ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, pCommandBuffers[i]);

      if (!cmd_buffer)
         continue;

      anv_cmd_buffer_destroy(cmd_buffer);
   }
}

VkResult
anv_cmd_buffer_reset(struct anv_cmd_buffer *cmd_buffer)
{
   cmd_buffer->usage_flags = 0;
   anv_cmd_buffer_reset_batch_bo_chain(cmd_buffer);
   anv_cmd_state_reset(cmd_buffer);

   anv_state_stream_finish(&cmd_buffer->surface_state_stream);
   anv_state_stream_init(&cmd_buffer->surface_state_stream,
                         &cmd_buffer->device->surface_state_pool, 4096);

   anv_state_stream_finish(&cmd_buffer->dynamic_state_stream);
   anv_state_stream_init(&cmd_buffer->dynamic_state_stream,
                         &cmd_buffer->device->dynamic_state_pool, 16384);
   return VK_SUCCESS;
}

VkResult anv_ResetCommandBuffer(
    VkCommandBuffer                             commandBuffer,
    VkCommandBufferResetFlags                   flags)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   return anv_cmd_buffer_reset(cmd_buffer);
}

#define anv_genX_call(devinfo, func, ...)          \
   switch ((devinfo)->gen) {                       \
   case 7:                                         \
      if ((devinfo)->is_haswell) {                 \
         gen75_##func(__VA_ARGS__);                \
      } else {                                     \
         gen7_##func(__VA_ARGS__);                 \
      }                                            \
      break;                                       \
   case 8:                                         \
      gen8_##func(__VA_ARGS__);                    \
      break;                                       \
   case 9:                                         \
      gen9_##func(__VA_ARGS__);                    \
      break;                                       \
   case 10:                                        \
      gen10_##func(__VA_ARGS__);                   \
      break;                                       \
   case 11:                                        \
      gen11_##func(__VA_ARGS__);                   \
      break;                                       \
   default:                                        \
      assert(!"Unknown hardware generation");      \
   }

void
anv_cmd_buffer_emit_state_base_address(struct anv_cmd_buffer *cmd_buffer)
{
   anv_genX_call(&cmd_buffer->device->info,
                 cmd_buffer_emit_state_base_address,
                 cmd_buffer);
}

void
anv_cmd_buffer_mark_image_written(struct anv_cmd_buffer *cmd_buffer,
                                  const struct anv_image *image,
                                  VkImageAspectFlagBits aspect,
                                  enum isl_aux_usage aux_usage,
                                  uint32_t level,
                                  uint32_t base_layer,
                                  uint32_t layer_count)
{
   anv_genX_call(&cmd_buffer->device->info,
                 cmd_buffer_mark_image_written,
                 cmd_buffer, image, aspect, aux_usage,
                 level, base_layer, layer_count);
}

void anv_CmdBindPipeline(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipeline                                  _pipeline)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_pipeline, pipeline, _pipeline);

   switch (pipelineBindPoint) {
   case VK_PIPELINE_BIND_POINT_COMPUTE:
      cmd_buffer->state.compute.base.pipeline = pipeline;
      cmd_buffer->state.compute.pipeline_dirty = true;
      cmd_buffer->state.push_constants_dirty |= VK_SHADER_STAGE_COMPUTE_BIT;
      cmd_buffer->state.descriptors_dirty |= VK_SHADER_STAGE_COMPUTE_BIT;
      break;

   case VK_PIPELINE_BIND_POINT_GRAPHICS:
      cmd_buffer->state.gfx.base.pipeline = pipeline;
      cmd_buffer->state.gfx.vb_dirty |= pipeline->vb_used;
      cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_PIPELINE;
      cmd_buffer->state.push_constants_dirty |= pipeline->active_stages;
      cmd_buffer->state.descriptors_dirty |= pipeline->active_stages;

      /* Apply the dynamic state from the pipeline */
      cmd_buffer->state.gfx.dirty |= pipeline->dynamic_state_mask;
      anv_dynamic_state_copy(&cmd_buffer->state.gfx.dynamic,
                             &pipeline->dynamic_state,
                             pipeline->dynamic_state_mask);
      break;

   default:
      assert(!"invalid bind point");
      break;
   }
}

void anv_CmdSetViewport(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstViewport,
    uint32_t                                    viewportCount,
    const VkViewport*                           pViewports)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   const uint32_t total_count = firstViewport + viewportCount;
   if (cmd_buffer->state.gfx.dynamic.viewport.count < total_count)
      cmd_buffer->state.gfx.dynamic.viewport.count = total_count;

   memcpy(cmd_buffer->state.gfx.dynamic.viewport.viewports + firstViewport,
          pViewports, viewportCount * sizeof(*pViewports));

   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_DYNAMIC_VIEWPORT;
}

void anv_CmdSetScissor(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstScissor,
    uint32_t                                    scissorCount,
    const VkRect2D*                             pScissors)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   const uint32_t total_count = firstScissor + scissorCount;
   if (cmd_buffer->state.gfx.dynamic.scissor.count < total_count)
      cmd_buffer->state.gfx.dynamic.scissor.count = total_count;

   memcpy(cmd_buffer->state.gfx.dynamic.scissor.scissors + firstScissor,
          pScissors, scissorCount * sizeof(*pScissors));

   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_DYNAMIC_SCISSOR;
}

void anv_CmdSetLineWidth(
    VkCommandBuffer                             commandBuffer,
    float                                       lineWidth)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   cmd_buffer->state.gfx.dynamic.line_width = lineWidth;
   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_DYNAMIC_LINE_WIDTH;
}

void anv_CmdSetDepthBias(
    VkCommandBuffer                             commandBuffer,
    float                                       depthBiasConstantFactor,
    float                                       depthBiasClamp,
    float                                       depthBiasSlopeFactor)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   cmd_buffer->state.gfx.dynamic.depth_bias.bias = depthBiasConstantFactor;
   cmd_buffer->state.gfx.dynamic.depth_bias.clamp = depthBiasClamp;
   cmd_buffer->state.gfx.dynamic.depth_bias.slope = depthBiasSlopeFactor;

   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_DYNAMIC_DEPTH_BIAS;
}

void anv_CmdSetBlendConstants(
    VkCommandBuffer                             commandBuffer,
    const float                                 blendConstants[4])
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   memcpy(cmd_buffer->state.gfx.dynamic.blend_constants,
          blendConstants, sizeof(float) * 4);

   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_DYNAMIC_BLEND_CONSTANTS;
}

void anv_CmdSetDepthBounds(
    VkCommandBuffer                             commandBuffer,
    float                                       minDepthBounds,
    float                                       maxDepthBounds)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   cmd_buffer->state.gfx.dynamic.depth_bounds.min = minDepthBounds;
   cmd_buffer->state.gfx.dynamic.depth_bounds.max = maxDepthBounds;

   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_DYNAMIC_DEPTH_BOUNDS;
}

void anv_CmdSetStencilCompareMask(
    VkCommandBuffer                             commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    compareMask)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   if (faceMask & VK_STENCIL_FACE_FRONT_BIT)
      cmd_buffer->state.gfx.dynamic.stencil_compare_mask.front = compareMask;
   if (faceMask & VK_STENCIL_FACE_BACK_BIT)
      cmd_buffer->state.gfx.dynamic.stencil_compare_mask.back = compareMask;

   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_DYNAMIC_STENCIL_COMPARE_MASK;
}

void anv_CmdSetStencilWriteMask(
    VkCommandBuffer                             commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    writeMask)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   if (faceMask & VK_STENCIL_FACE_FRONT_BIT)
      cmd_buffer->state.gfx.dynamic.stencil_write_mask.front = writeMask;
   if (faceMask & VK_STENCIL_FACE_BACK_BIT)
      cmd_buffer->state.gfx.dynamic.stencil_write_mask.back = writeMask;

   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_DYNAMIC_STENCIL_WRITE_MASK;
}

void anv_CmdSetStencilReference(
    VkCommandBuffer                             commandBuffer,
    VkStencilFaceFlags                          faceMask,
    uint32_t                                    reference)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   if (faceMask & VK_STENCIL_FACE_FRONT_BIT)
      cmd_buffer->state.gfx.dynamic.stencil_reference.front = reference;
   if (faceMask & VK_STENCIL_FACE_BACK_BIT)
      cmd_buffer->state.gfx.dynamic.stencil_reference.back = reference;

   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_DYNAMIC_STENCIL_REFERENCE;
}

static void
anv_cmd_buffer_bind_descriptor_set(struct anv_cmd_buffer *cmd_buffer,
                                   VkPipelineBindPoint bind_point,
                                   struct anv_pipeline_layout *layout,
                                   uint32_t set_index,
                                   struct anv_descriptor_set *set,
                                   uint32_t *dynamic_offset_count,
                                   const uint32_t **dynamic_offsets)
{
   struct anv_descriptor_set_layout *set_layout =
      layout->set[set_index].layout;

   struct anv_cmd_pipeline_state *pipe_state;
   if (bind_point == VK_PIPELINE_BIND_POINT_COMPUTE) {
      pipe_state = &cmd_buffer->state.compute.base;
   } else {
      assert(bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS);
      pipe_state = &cmd_buffer->state.gfx.base;
   }
   pipe_state->descriptors[set_index] = set;

   if (dynamic_offsets) {
      if (set_layout->dynamic_offset_count > 0) {
         uint32_t dynamic_offset_start =
            layout->set[set_index].dynamic_offset_start;

         /* Assert that everything is in range */
         assert(set_layout->dynamic_offset_count <= *dynamic_offset_count);
         assert(dynamic_offset_start + set_layout->dynamic_offset_count <=
                ARRAY_SIZE(pipe_state->dynamic_offsets));

         typed_memcpy(&pipe_state->dynamic_offsets[dynamic_offset_start],
                      *dynamic_offsets, set_layout->dynamic_offset_count);

         *dynamic_offsets += set_layout->dynamic_offset_count;
         *dynamic_offset_count -= set_layout->dynamic_offset_count;
      }
   }

   if (bind_point == VK_PIPELINE_BIND_POINT_COMPUTE) {
      cmd_buffer->state.descriptors_dirty |= VK_SHADER_STAGE_COMPUTE_BIT;
   } else {
      assert(bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS);
      cmd_buffer->state.descriptors_dirty |=
         set_layout->shader_stages & VK_SHADER_STAGE_ALL_GRAPHICS;
   }

   /* Pipeline layout objects are required to live at least while any command
    * buffers that use them are in recording state. We need to grab a reference
    * to the pipeline layout being bound here so we can compute correct dynamic
    * offsets for VK_DESCRIPTOR_TYPE_*_DYNAMIC in dynamic_offset_for_binding()
    * when we record draw commands that come after this.
    */
   pipe_state->layout = layout;
}

void anv_CmdBindDescriptorSets(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            _layout,
    uint32_t                                    firstSet,
    uint32_t                                    descriptorSetCount,
    const VkDescriptorSet*                      pDescriptorSets,
    uint32_t                                    dynamicOffsetCount,
    const uint32_t*                             pDynamicOffsets)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_pipeline_layout, layout, _layout);

   assert(firstSet + descriptorSetCount <= MAX_SETS);

   for (uint32_t i = 0; i < descriptorSetCount; i++) {
      ANV_FROM_HANDLE(anv_descriptor_set, set, pDescriptorSets[i]);
      anv_cmd_buffer_bind_descriptor_set(cmd_buffer, pipelineBindPoint,
                                         layout, firstSet + i, set,
                                         &dynamicOffsetCount,
                                         &pDynamicOffsets);
   }
}

void anv_CmdBindVertexBuffers(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstBinding,
    uint32_t                                    bindingCount,
    const VkBuffer*                             pBuffers,
    const VkDeviceSize*                         pOffsets)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   struct anv_vertex_binding *vb = cmd_buffer->state.vertex_bindings;

   /* We have to defer setting up vertex buffer since we need the buffer
    * stride from the pipeline. */

   assert(firstBinding + bindingCount <= MAX_VBS);
   for (uint32_t i = 0; i < bindingCount; i++) {
      vb[firstBinding + i].buffer = anv_buffer_from_handle(pBuffers[i]);
      vb[firstBinding + i].offset = pOffsets[i];
      cmd_buffer->state.gfx.vb_dirty |= 1 << (firstBinding + i);
   }
}

enum isl_format
anv_isl_format_for_descriptor_type(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      return ISL_FORMAT_R32G32B32A32_FLOAT;

   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      return ISL_FORMAT_RAW;

   default:
      unreachable("Invalid descriptor type");
   }
}

struct anv_state
anv_cmd_buffer_emit_dynamic(struct anv_cmd_buffer *cmd_buffer,
                            const void *data, uint32_t size, uint32_t alignment)
{
   struct anv_state state;

   state = anv_cmd_buffer_alloc_dynamic_state(cmd_buffer, size, alignment);
   memcpy(state.map, data, size);

   anv_state_flush(cmd_buffer->device, state);

   VG(VALGRIND_CHECK_MEM_IS_DEFINED(state.map, size));

   return state;
}

struct anv_state
anv_cmd_buffer_merge_dynamic(struct anv_cmd_buffer *cmd_buffer,
                             uint32_t *a, uint32_t *b,
                             uint32_t dwords, uint32_t alignment)
{
   struct anv_state state;
   uint32_t *p;

   state = anv_cmd_buffer_alloc_dynamic_state(cmd_buffer,
                                              dwords * 4, alignment);
   p = state.map;
   for (uint32_t i = 0; i < dwords; i++)
      p[i] = a[i] | b[i];

   anv_state_flush(cmd_buffer->device, state);

   VG(VALGRIND_CHECK_MEM_IS_DEFINED(p, dwords * 4));

   return state;
}

static uint32_t
anv_push_constant_value(struct anv_push_constants *data, uint32_t param)
{
   if (BRW_PARAM_IS_BUILTIN(param)) {
      switch (param) {
      case BRW_PARAM_BUILTIN_ZERO:
         return 0;
      case BRW_PARAM_BUILTIN_BASE_WORK_GROUP_ID_X:
         return data->base_work_group_id[0];
      case BRW_PARAM_BUILTIN_BASE_WORK_GROUP_ID_Y:
         return data->base_work_group_id[1];
      case BRW_PARAM_BUILTIN_BASE_WORK_GROUP_ID_Z:
         return data->base_work_group_id[2];
      default:
         unreachable("Invalid param builtin");
      }
   } else {
      uint32_t offset = ANV_PARAM_PUSH_OFFSET(param);
      assert(offset % sizeof(uint32_t) == 0);
      if (offset < data->size)
         return *(uint32_t *)((uint8_t *)data + offset);
      else
         return 0;
   }
}

struct anv_state
anv_cmd_buffer_push_constants(struct anv_cmd_buffer *cmd_buffer,
                              gl_shader_stage stage)
{
   struct anv_pipeline *pipeline = cmd_buffer->state.gfx.base.pipeline;

   /* If we don't have this stage, bail. */
   if (!anv_pipeline_has_stage(pipeline, stage))
      return (struct anv_state) { .offset = 0 };

   struct anv_push_constants *data =
      cmd_buffer->state.push_constants[stage];
   const struct brw_stage_prog_data *prog_data =
      pipeline->shaders[stage]->prog_data;

   /* If we don't actually have any push constants, bail. */
   if (data == NULL || prog_data == NULL || prog_data->nr_params == 0)
      return (struct anv_state) { .offset = 0 };

   struct anv_state state =
      anv_cmd_buffer_alloc_dynamic_state(cmd_buffer,
                                         prog_data->nr_params * sizeof(float),
                                         32 /* bottom 5 bits MBZ */);

   /* Walk through the param array and fill the buffer with data */
   uint32_t *u32_map = state.map;
   for (unsigned i = 0; i < prog_data->nr_params; i++)
      u32_map[i] = anv_push_constant_value(data, prog_data->param[i]);

   anv_state_flush(cmd_buffer->device, state);

   return state;
}

struct anv_state
anv_cmd_buffer_cs_push_constants(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_push_constants *data =
      cmd_buffer->state.push_constants[MESA_SHADER_COMPUTE];
   struct anv_pipeline *pipeline = cmd_buffer->state.compute.base.pipeline;
   const struct brw_cs_prog_data *cs_prog_data = get_cs_prog_data(pipeline);
   const struct brw_stage_prog_data *prog_data = &cs_prog_data->base;

   /* If we don't actually have any push constants, bail. */
   if (cs_prog_data->push.total.size == 0)
      return (struct anv_state) { .offset = 0 };

   const unsigned push_constant_alignment =
      cmd_buffer->device->info.gen < 8 ? 32 : 64;
   const unsigned aligned_total_push_constants_size =
      ALIGN(cs_prog_data->push.total.size, push_constant_alignment);
   struct anv_state state =
      anv_cmd_buffer_alloc_dynamic_state(cmd_buffer,
                                         aligned_total_push_constants_size,
                                         push_constant_alignment);

   /* Walk through the param array and fill the buffer with data */
   uint32_t *u32_map = state.map;

   if (cs_prog_data->push.cross_thread.size > 0) {
      for (unsigned i = 0;
           i < cs_prog_data->push.cross_thread.dwords;
           i++) {
         assert(prog_data->param[i] != BRW_PARAM_BUILTIN_SUBGROUP_ID);
         u32_map[i] = anv_push_constant_value(data, prog_data->param[i]);
      }
   }

   if (cs_prog_data->push.per_thread.size > 0) {
      for (unsigned t = 0; t < cs_prog_data->threads; t++) {
         unsigned dst =
            8 * (cs_prog_data->push.per_thread.regs * t +
                 cs_prog_data->push.cross_thread.regs);
         unsigned src = cs_prog_data->push.cross_thread.dwords;
         for ( ; src < prog_data->nr_params; src++, dst++) {
            if (prog_data->param[src] == BRW_PARAM_BUILTIN_SUBGROUP_ID) {
               u32_map[dst] = t;
            } else {
               u32_map[dst] =
                  anv_push_constant_value(data, prog_data->param[src]);
            }
         }
      }
   }

   anv_state_flush(cmd_buffer->device, state);

   return state;
}

void anv_CmdPushConstants(
    VkCommandBuffer                             commandBuffer,
    VkPipelineLayout                            layout,
    VkShaderStageFlags                          stageFlags,
    uint32_t                                    offset,
    uint32_t                                    size,
    const void*                                 pValues)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);

   anv_foreach_stage(stage, stageFlags) {
      VkResult result =
         anv_cmd_buffer_ensure_push_constant_field(cmd_buffer,
                                                   stage, client_data);
      if (result != VK_SUCCESS)
         return;

      memcpy(cmd_buffer->state.push_constants[stage]->client_data + offset,
             pValues, size);
   }

   cmd_buffer->state.push_constants_dirty |= stageFlags;
}

VkResult anv_CreateCommandPool(
    VkDevice                                    _device,
    const VkCommandPoolCreateInfo*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkCommandPool*                              pCmdPool)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_cmd_pool *pool;

   pool = vk_alloc2(&device->alloc, pAllocator, sizeof(*pool), 8,
                     VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pool == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   if (pAllocator)
      pool->alloc = *pAllocator;
   else
      pool->alloc = device->alloc;

   list_inithead(&pool->cmd_buffers);

   *pCmdPool = anv_cmd_pool_to_handle(pool);

   return VK_SUCCESS;
}

void anv_DestroyCommandPool(
    VkDevice                                    _device,
    VkCommandPool                               commandPool,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_cmd_pool, pool, commandPool);

   if (!pool)
      return;

   list_for_each_entry_safe(struct anv_cmd_buffer, cmd_buffer,
                            &pool->cmd_buffers, pool_link) {
      anv_cmd_buffer_destroy(cmd_buffer);
   }

   vk_free2(&device->alloc, pAllocator, pool);
}

VkResult anv_ResetCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolResetFlags                     flags)
{
   ANV_FROM_HANDLE(anv_cmd_pool, pool, commandPool);

   list_for_each_entry(struct anv_cmd_buffer, cmd_buffer,
                       &pool->cmd_buffers, pool_link) {
      anv_cmd_buffer_reset(cmd_buffer);
   }

   return VK_SUCCESS;
}

void anv_TrimCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolTrimFlags                      flags)
{
   /* Nothing for us to do here.  Our pools stay pretty tidy. */
}

/**
 * Return NULL if the current subpass has no depthstencil attachment.
 */
const struct anv_image_view *
anv_cmd_buffer_get_depth_stencil_view(const struct anv_cmd_buffer *cmd_buffer)
{
   const struct anv_subpass *subpass = cmd_buffer->state.subpass;
   const struct anv_framebuffer *fb = cmd_buffer->state.framebuffer;

   if (subpass->depth_stencil_attachment == NULL)
      return NULL;

   const struct anv_image_view *iview =
      fb->attachments[subpass->depth_stencil_attachment->attachment];

   assert(iview->aspect_mask & (VK_IMAGE_ASPECT_DEPTH_BIT |
                                VK_IMAGE_ASPECT_STENCIL_BIT));

   return iview;
}

static struct anv_push_descriptor_set *
anv_cmd_buffer_get_push_descriptor_set(struct anv_cmd_buffer *cmd_buffer,
                                       VkPipelineBindPoint bind_point,
                                       uint32_t set)
{
   struct anv_cmd_pipeline_state *pipe_state;
   if (bind_point == VK_PIPELINE_BIND_POINT_COMPUTE) {
      pipe_state = &cmd_buffer->state.compute.base;
   } else {
      assert(bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS);
      pipe_state = &cmd_buffer->state.gfx.base;
   }

   struct anv_push_descriptor_set **push_set =
      &pipe_state->push_descriptors[set];

   if (*push_set == NULL) {
      *push_set = vk_alloc(&cmd_buffer->pool->alloc,
                           sizeof(struct anv_push_descriptor_set), 8,
                           VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (*push_set == NULL) {
         anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_HOST_MEMORY);
         return NULL;
      }
   }

   return *push_set;
}

void anv_CmdPushDescriptorSetKHR(
    VkCommandBuffer commandBuffer,
    VkPipelineBindPoint pipelineBindPoint,
    VkPipelineLayout _layout,
    uint32_t _set,
    uint32_t descriptorWriteCount,
    const VkWriteDescriptorSet* pDescriptorWrites)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_pipeline_layout, layout, _layout);

   assert(_set < MAX_SETS);

   struct anv_descriptor_set_layout *set_layout = layout->set[_set].layout;

   struct anv_push_descriptor_set *push_set =
      anv_cmd_buffer_get_push_descriptor_set(cmd_buffer,
                                             pipelineBindPoint, _set);
   if (!push_set)
      return;

   struct anv_descriptor_set *set = &push_set->set;

   set->layout = set_layout;
   set->size = anv_descriptor_set_layout_size(set_layout);
   set->buffer_count = set_layout->buffer_count;
   set->buffer_views = push_set->buffer_views;

   /* Go through the user supplied descriptors. */
   for (uint32_t i = 0; i < descriptorWriteCount; i++) {
      const VkWriteDescriptorSet *write = &pDescriptorWrites[i];

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            anv_descriptor_set_write_image_view(set, &cmd_buffer->device->info,
                                                write->pImageInfo + j,
                                                write->descriptorType,
                                                write->dstBinding,
                                                write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            ANV_FROM_HANDLE(anv_buffer_view, bview,
                            write->pTexelBufferView[j]);

            anv_descriptor_set_write_buffer_view(set,
                                                 write->descriptorType,
                                                 bview,
                                                 write->dstBinding,
                                                 write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            assert(write->pBufferInfo[j].buffer);
            ANV_FROM_HANDLE(anv_buffer, buffer, write->pBufferInfo[j].buffer);
            assert(buffer);

            anv_descriptor_set_write_buffer(set,
                                            cmd_buffer->device,
                                            &cmd_buffer->surface_state_stream,
                                            write->descriptorType,
                                            buffer,
                                            write->dstBinding,
                                            write->dstArrayElement + j,
                                            write->pBufferInfo[j].offset,
                                            write->pBufferInfo[j].range);
         }
         break;

      default:
         break;
      }
   }

   anv_cmd_buffer_bind_descriptor_set(cmd_buffer, pipelineBindPoint,
                                      layout, _set, set, NULL, NULL);
}

void anv_CmdPushDescriptorSetWithTemplateKHR(
    VkCommandBuffer                             commandBuffer,
    VkDescriptorUpdateTemplate                  descriptorUpdateTemplate,
    VkPipelineLayout                            _layout,
    uint32_t                                    _set,
    const void*                                 pData)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_descriptor_update_template, template,
                   descriptorUpdateTemplate);
   ANV_FROM_HANDLE(anv_pipeline_layout, layout, _layout);

   assert(_set < MAX_PUSH_DESCRIPTORS);

   struct anv_descriptor_set_layout *set_layout = layout->set[_set].layout;

   struct anv_push_descriptor_set *push_set =
      anv_cmd_buffer_get_push_descriptor_set(cmd_buffer,
                                             template->bind_point, _set);
   if (!push_set)
      return;

   struct anv_descriptor_set *set = &push_set->set;

   set->layout = set_layout;
   set->size = anv_descriptor_set_layout_size(set_layout);
   set->buffer_count = set_layout->buffer_count;
   set->buffer_views = push_set->buffer_views;

   anv_descriptor_set_write_template(set,
                                     cmd_buffer->device,
                                     &cmd_buffer->surface_state_stream,
                                     template,
                                     pData);

   anv_cmd_buffer_bind_descriptor_set(cmd_buffer, template->bind_point,
                                      layout, _set, set, NULL, NULL);
}

void anv_CmdSetDeviceMask(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    deviceMask)
{
   /* No-op */
}
