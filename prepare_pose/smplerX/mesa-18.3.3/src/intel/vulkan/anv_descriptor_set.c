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

#include "anv_private.h"

/*
 * Descriptor set layouts.
 */

void anv_GetDescriptorSetLayoutSupport(
    VkDevice                                    device,
    const VkDescriptorSetLayoutCreateInfo*      pCreateInfo,
    VkDescriptorSetLayoutSupport*               pSupport)
{
   uint32_t surface_count[MESA_SHADER_STAGES] = { 0, };

   for (uint32_t b = 0; b < pCreateInfo->bindingCount; b++) {
      const VkDescriptorSetLayoutBinding *binding = &pCreateInfo->pBindings[b];

      switch (binding->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
         /* There is no real limit on samplers */
         break;

      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         if (binding->pImmutableSamplers) {
            for (uint32_t i = 0; i < binding->descriptorCount; i++) {
               ANV_FROM_HANDLE(anv_sampler, sampler,
                               binding->pImmutableSamplers[i]);
               anv_foreach_stage(s, binding->stageFlags)
                  surface_count[s] += sampler->n_planes;
            }
         }
         break;

      default:
         anv_foreach_stage(s, binding->stageFlags)
            surface_count[s] += binding->descriptorCount;
         break;
      }
   }

   bool supported = true;
   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
      /* Our maximum binding table size is 250 and we need to reserve 8 for
       * render targets.  240 is a nice round number.
       */
      if (surface_count[s] >= 240)
         supported = false;
   }

   pSupport->supported = supported;
}

VkResult anv_CreateDescriptorSetLayout(
    VkDevice                                    _device,
    const VkDescriptorSetLayoutCreateInfo*      pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorSetLayout*                      pSetLayout)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);

   uint32_t max_binding = 0;
   uint32_t immutable_sampler_count = 0;
   for (uint32_t j = 0; j < pCreateInfo->bindingCount; j++) {
      max_binding = MAX2(max_binding, pCreateInfo->pBindings[j].binding);

      /* From the Vulkan 1.1.97 spec for VkDescriptorSetLayoutBinding:
       *
       *    "If descriptorType specifies a VK_DESCRIPTOR_TYPE_SAMPLER or
       *    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER type descriptor, then
       *    pImmutableSamplers can be used to initialize a set of immutable
       *    samplers. [...]  If descriptorType is not one of these descriptor
       *    types, then pImmutableSamplers is ignored.
       *
       * We need to be careful here and only parse pImmutableSamplers if we
       * have one of the right descriptor types.
       */
      VkDescriptorType desc_type = pCreateInfo->pBindings[j].descriptorType;
      if ((desc_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
           desc_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) &&
          pCreateInfo->pBindings[j].pImmutableSamplers)
         immutable_sampler_count += pCreateInfo->pBindings[j].descriptorCount;
   }

   struct anv_descriptor_set_layout *set_layout;
   struct anv_descriptor_set_binding_layout *bindings;
   struct anv_sampler **samplers;

   /* We need to allocate decriptor set layouts off the device allocator
    * with DEVICE scope because they are reference counted and may not be
    * destroyed when vkDestroyDescriptorSetLayout is called.
    */
   ANV_MULTIALLOC(ma);
   anv_multialloc_add(&ma, &set_layout, 1);
   anv_multialloc_add(&ma, &bindings, max_binding + 1);
   anv_multialloc_add(&ma, &samplers, immutable_sampler_count);

   if (!anv_multialloc_alloc(&ma, &device->alloc,
                             VK_SYSTEM_ALLOCATION_SCOPE_DEVICE))
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   memset(set_layout, 0, sizeof(*set_layout));
   set_layout->ref_cnt = 1;
   set_layout->binding_count = max_binding + 1;

   for (uint32_t b = 0; b <= max_binding; b++) {
      /* Initialize all binding_layout entries to -1 */
      memset(&set_layout->binding[b], -1, sizeof(set_layout->binding[b]));

      set_layout->binding[b].array_size = 0;
      set_layout->binding[b].immutable_samplers = NULL;
   }

   /* Initialize all samplers to 0 */
   memset(samplers, 0, immutable_sampler_count * sizeof(*samplers));

   uint32_t sampler_count[MESA_SHADER_STAGES] = { 0, };
   uint32_t surface_count[MESA_SHADER_STAGES] = { 0, };
   uint32_t image_count[MESA_SHADER_STAGES] = { 0, };
   uint32_t buffer_count = 0;
   uint32_t dynamic_offset_count = 0;

   for (uint32_t j = 0; j < pCreateInfo->bindingCount; j++) {
      const VkDescriptorSetLayoutBinding *binding = &pCreateInfo->pBindings[j];
      uint32_t b = binding->binding;
      /* We temporarily store the pointer to the binding in the
       * immutable_samplers pointer.  This provides us with a quick-and-dirty
       * way to sort the bindings by binding number.
       */
      set_layout->binding[b].immutable_samplers = (void *)binding;
   }

   for (uint32_t b = 0; b <= max_binding; b++) {
      const VkDescriptorSetLayoutBinding *binding =
         (void *)set_layout->binding[b].immutable_samplers;

      if (binding == NULL)
         continue;

      /* We temporarily stashed the pointer to the binding in the
       * immutable_samplers pointer.  Now that we've pulled it back out
       * again, we reset immutable_samplers to NULL.
       */
      set_layout->binding[b].immutable_samplers = NULL;

      if (binding->descriptorCount == 0)
         continue;

#ifndef NDEBUG
      set_layout->binding[b].type = binding->descriptorType;
#endif
      set_layout->binding[b].array_size = binding->descriptorCount;
      set_layout->binding[b].descriptor_index = set_layout->size;
      set_layout->size += binding->descriptorCount;

      switch (binding->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         anv_foreach_stage(s, binding->stageFlags) {
            set_layout->binding[b].stage[s].sampler_index = sampler_count[s];
            sampler_count[s] += binding->descriptorCount;
         }

         if (binding->pImmutableSamplers) {
            set_layout->binding[b].immutable_samplers = samplers;
            samplers += binding->descriptorCount;

            for (uint32_t i = 0; i < binding->descriptorCount; i++)
               set_layout->binding[b].immutable_samplers[i] =
                  anv_sampler_from_handle(binding->pImmutableSamplers[i]);
         }
         break;
      default:
         break;
      }

      switch (binding->descriptorType) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         set_layout->binding[b].buffer_index = buffer_count;
         buffer_count += binding->descriptorCount;
         /* fall through */

      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         anv_foreach_stage(s, binding->stageFlags) {
            set_layout->binding[b].stage[s].surface_index = surface_count[s];
            surface_count[s] += binding->descriptorCount;
         }
         break;
      default:
         break;
      }

      switch (binding->descriptorType) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         set_layout->binding[b].dynamic_offset_index = dynamic_offset_count;
         dynamic_offset_count += binding->descriptorCount;
         break;
      default:
         break;
      }

      switch (binding->descriptorType) {
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         anv_foreach_stage(s, binding->stageFlags) {
            set_layout->binding[b].stage[s].image_index = image_count[s];
            image_count[s] += binding->descriptorCount;
         }
         break;
      default:
         break;
      }

      set_layout->shader_stages |= binding->stageFlags;
   }

   set_layout->buffer_count = buffer_count;
   set_layout->dynamic_offset_count = dynamic_offset_count;

   *pSetLayout = anv_descriptor_set_layout_to_handle(set_layout);

   return VK_SUCCESS;
}

void anv_DestroyDescriptorSetLayout(
    VkDevice                                    _device,
    VkDescriptorSetLayout                       _set_layout,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_descriptor_set_layout, set_layout, _set_layout);

   if (!set_layout)
      return;

   anv_descriptor_set_layout_unref(device, set_layout);
}

#define SHA1_UPDATE_VALUE(ctx, x) _mesa_sha1_update(ctx, &(x), sizeof(x));

static void
sha1_update_immutable_sampler(struct mesa_sha1 *ctx,
                              const struct anv_sampler *sampler)
{
   if (!sampler->conversion)
      return;

   /* The only thing that affects the shader is ycbcr conversion */
   _mesa_sha1_update(ctx, sampler->conversion,
                     sizeof(*sampler->conversion));
}

static void
sha1_update_descriptor_set_binding_layout(struct mesa_sha1 *ctx,
   const struct anv_descriptor_set_binding_layout *layout)
{
   SHA1_UPDATE_VALUE(ctx, layout->array_size);
   SHA1_UPDATE_VALUE(ctx, layout->descriptor_index);
   SHA1_UPDATE_VALUE(ctx, layout->dynamic_offset_index);
   SHA1_UPDATE_VALUE(ctx, layout->buffer_index);
   _mesa_sha1_update(ctx, layout->stage, sizeof(layout->stage));

   if (layout->immutable_samplers) {
      for (uint16_t i = 0; i < layout->array_size; i++)
         sha1_update_immutable_sampler(ctx, layout->immutable_samplers[i]);
   }
}

static void
sha1_update_descriptor_set_layout(struct mesa_sha1 *ctx,
                                  const struct anv_descriptor_set_layout *layout)
{
   SHA1_UPDATE_VALUE(ctx, layout->binding_count);
   SHA1_UPDATE_VALUE(ctx, layout->size);
   SHA1_UPDATE_VALUE(ctx, layout->shader_stages);
   SHA1_UPDATE_VALUE(ctx, layout->buffer_count);
   SHA1_UPDATE_VALUE(ctx, layout->dynamic_offset_count);

   for (uint16_t i = 0; i < layout->binding_count; i++)
      sha1_update_descriptor_set_binding_layout(ctx, &layout->binding[i]);
}

/*
 * Pipeline layouts.  These have nothing to do with the pipeline.  They are
 * just multiple descriptor set layouts pasted together
 */

VkResult anv_CreatePipelineLayout(
    VkDevice                                    _device,
    const VkPipelineLayoutCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineLayout*                           pPipelineLayout)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_pipeline_layout *layout;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);

   layout = vk_alloc2(&device->alloc, pAllocator, sizeof(*layout), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (layout == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   layout->num_sets = pCreateInfo->setLayoutCount;

   unsigned dynamic_offset_count = 0;

   memset(layout->stage, 0, sizeof(layout->stage));
   for (uint32_t set = 0; set < pCreateInfo->setLayoutCount; set++) {
      ANV_FROM_HANDLE(anv_descriptor_set_layout, set_layout,
                      pCreateInfo->pSetLayouts[set]);
      layout->set[set].layout = set_layout;
      anv_descriptor_set_layout_ref(set_layout);

      layout->set[set].dynamic_offset_start = dynamic_offset_count;
      for (uint32_t b = 0; b < set_layout->binding_count; b++) {
         if (set_layout->binding[b].dynamic_offset_index < 0)
            continue;

         dynamic_offset_count += set_layout->binding[b].array_size;
         for (gl_shader_stage s = 0; s < MESA_SHADER_STAGES; s++) {
            if (set_layout->binding[b].stage[s].surface_index >= 0)
               layout->stage[s].has_dynamic_offsets = true;
         }
      }
   }

   struct mesa_sha1 ctx;
   _mesa_sha1_init(&ctx);
   for (unsigned s = 0; s < layout->num_sets; s++) {
      sha1_update_descriptor_set_layout(&ctx, layout->set[s].layout);
      _mesa_sha1_update(&ctx, &layout->set[s].dynamic_offset_start,
                        sizeof(layout->set[s].dynamic_offset_start));
   }
   _mesa_sha1_update(&ctx, &layout->num_sets, sizeof(layout->num_sets));
   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
      _mesa_sha1_update(&ctx, &layout->stage[s].has_dynamic_offsets,
                        sizeof(layout->stage[s].has_dynamic_offsets));
   }
   _mesa_sha1_final(&ctx, layout->sha1);

   *pPipelineLayout = anv_pipeline_layout_to_handle(layout);

   return VK_SUCCESS;
}

void anv_DestroyPipelineLayout(
    VkDevice                                    _device,
    VkPipelineLayout                            _pipelineLayout,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_pipeline_layout, pipeline_layout, _pipelineLayout);

   if (!pipeline_layout)
      return;

   for (uint32_t i = 0; i < pipeline_layout->num_sets; i++)
      anv_descriptor_set_layout_unref(device, pipeline_layout->set[i].layout);

   vk_free2(&device->alloc, pAllocator, pipeline_layout);
}

/*
 * Descriptor pools.
 *
 * These are implemented using a big pool of memory and a free-list for the
 * host memory allocations and a state_stream and a free list for the buffer
 * view surface state. The spec allows us to fail to allocate due to
 * fragmentation in all cases but two: 1) after pool reset, allocating up
 * until the pool size with no freeing must succeed and 2) allocating and
 * freeing only descriptor sets with the same layout. Case 1) is easy enogh,
 * and the free lists lets us recycle blocks for case 2).
 */

#define EMPTY 1

VkResult anv_CreateDescriptorPool(
    VkDevice                                    _device,
    const VkDescriptorPoolCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorPool*                           pDescriptorPool)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_descriptor_pool *pool;

   uint32_t descriptor_count = 0;
   uint32_t buffer_count = 0;
   for (uint32_t i = 0; i < pCreateInfo->poolSizeCount; i++) {
      switch (pCreateInfo->pPoolSizes[i].type) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         buffer_count += pCreateInfo->pPoolSizes[i].descriptorCount;
      default:
         descriptor_count += pCreateInfo->pPoolSizes[i].descriptorCount;
         break;
      }
   }

   const size_t pool_size =
      pCreateInfo->maxSets * sizeof(struct anv_descriptor_set) +
      descriptor_count * sizeof(struct anv_descriptor) +
      buffer_count * sizeof(struct anv_buffer_view);
   const size_t total_size = sizeof(*pool) + pool_size;

   pool = vk_alloc2(&device->alloc, pAllocator, total_size, 8,
                     VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!pool)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   pool->size = pool_size;
   pool->next = 0;
   pool->free_list = EMPTY;

   anv_state_stream_init(&pool->surface_state_stream,
                         &device->surface_state_pool, 4096);
   pool->surface_state_free_list = NULL;

   *pDescriptorPool = anv_descriptor_pool_to_handle(pool);

   return VK_SUCCESS;
}

void anv_DestroyDescriptorPool(
    VkDevice                                    _device,
    VkDescriptorPool                            _pool,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_descriptor_pool, pool, _pool);

   if (!pool)
      return;

   anv_state_stream_finish(&pool->surface_state_stream);
   vk_free2(&device->alloc, pAllocator, pool);
}

VkResult anv_ResetDescriptorPool(
    VkDevice                                    _device,
    VkDescriptorPool                            descriptorPool,
    VkDescriptorPoolResetFlags                  flags)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_descriptor_pool, pool, descriptorPool);

   pool->next = 0;
   pool->free_list = EMPTY;
   anv_state_stream_finish(&pool->surface_state_stream);
   anv_state_stream_init(&pool->surface_state_stream,
                         &device->surface_state_pool, 4096);
   pool->surface_state_free_list = NULL;

   return VK_SUCCESS;
}

struct pool_free_list_entry {
   uint32_t next;
   uint32_t size;
};

size_t
anv_descriptor_set_layout_size(const struct anv_descriptor_set_layout *layout)
{
   return
      sizeof(struct anv_descriptor_set) +
      layout->size * sizeof(struct anv_descriptor) +
      layout->buffer_count * sizeof(struct anv_buffer_view);
}

struct surface_state_free_list_entry {
   void *next;
   struct anv_state state;
};

VkResult
anv_descriptor_set_create(struct anv_device *device,
                          struct anv_descriptor_pool *pool,
                          struct anv_descriptor_set_layout *layout,
                          struct anv_descriptor_set **out_set)
{
   struct anv_descriptor_set *set;
   const size_t size = anv_descriptor_set_layout_size(layout);

   set = NULL;
   if (size <= pool->size - pool->next) {
      set = (struct anv_descriptor_set *) (pool->data + pool->next);
      pool->next += size;
   } else {
      struct pool_free_list_entry *entry;
      uint32_t *link = &pool->free_list;
      for (uint32_t f = pool->free_list; f != EMPTY; f = entry->next) {
         entry = (struct pool_free_list_entry *) (pool->data + f);
         if (size <= entry->size) {
            *link = entry->next;
            set = (struct anv_descriptor_set *) entry;
            break;
         }
         link = &entry->next;
      }
   }

   if (set == NULL) {
      if (pool->free_list != EMPTY) {
         return vk_error(VK_ERROR_FRAGMENTED_POOL);
      } else {
         return vk_error(VK_ERROR_OUT_OF_POOL_MEMORY);
      }
   }

   set->layout = layout;
   anv_descriptor_set_layout_ref(layout);

   set->size = size;
   set->buffer_views =
      (struct anv_buffer_view *) &set->descriptors[layout->size];
   set->buffer_count = layout->buffer_count;

   /* By defining the descriptors to be zero now, we can later verify that
    * a descriptor has not been populated with user data.
    */
   memset(set->descriptors, 0, sizeof(struct anv_descriptor) * layout->size);

   /* Go through and fill out immutable samplers if we have any */
   struct anv_descriptor *desc = set->descriptors;
   for (uint32_t b = 0; b < layout->binding_count; b++) {
      if (layout->binding[b].immutable_samplers) {
         for (uint32_t i = 0; i < layout->binding[b].array_size; i++) {
            /* The type will get changed to COMBINED_IMAGE_SAMPLER in
             * UpdateDescriptorSets if needed.  However, if the descriptor
             * set has an immutable sampler, UpdateDescriptorSets may never
             * touch it, so we need to make sure it's 100% valid now.
             */
            desc[i] = (struct anv_descriptor) {
               .type = VK_DESCRIPTOR_TYPE_SAMPLER,
               .sampler = layout->binding[b].immutable_samplers[i],
            };
         }
      }
      desc += layout->binding[b].array_size;
   }

   /* Allocate surface state for the buffer views. */
   for (uint32_t b = 0; b < layout->buffer_count; b++) {
      struct surface_state_free_list_entry *entry =
         pool->surface_state_free_list;
      struct anv_state state;

      if (entry) {
         state = entry->state;
         pool->surface_state_free_list = entry->next;
         assert(state.alloc_size == 64);
      } else {
         state = anv_state_stream_alloc(&pool->surface_state_stream, 64, 64);
      }

      set->buffer_views[b].surface_state = state;
   }

   *out_set = set;

   return VK_SUCCESS;
}

void
anv_descriptor_set_destroy(struct anv_device *device,
                           struct anv_descriptor_pool *pool,
                           struct anv_descriptor_set *set)
{
   anv_descriptor_set_layout_unref(device, set->layout);

   /* Put the buffer view surface state back on the free list. */
   for (uint32_t b = 0; b < set->buffer_count; b++) {
      struct surface_state_free_list_entry *entry =
         set->buffer_views[b].surface_state.map;
      entry->next = pool->surface_state_free_list;
      entry->state = set->buffer_views[b].surface_state;
      pool->surface_state_free_list = entry;
   }

   /* Put the descriptor set allocation back on the free list. */
   const uint32_t index = (char *) set - pool->data;
   if (index + set->size == pool->next) {
      pool->next = index;
   } else {
      struct pool_free_list_entry *entry = (struct pool_free_list_entry *) set;
      entry->next = pool->free_list;
      entry->size = set->size;
      pool->free_list = (char *) entry - pool->data;
   }
}

VkResult anv_AllocateDescriptorSets(
    VkDevice                                    _device,
    const VkDescriptorSetAllocateInfo*          pAllocateInfo,
    VkDescriptorSet*                            pDescriptorSets)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_descriptor_pool, pool, pAllocateInfo->descriptorPool);

   VkResult result = VK_SUCCESS;
   struct anv_descriptor_set *set;
   uint32_t i;

   for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      ANV_FROM_HANDLE(anv_descriptor_set_layout, layout,
                      pAllocateInfo->pSetLayouts[i]);

      result = anv_descriptor_set_create(device, pool, layout, &set);
      if (result != VK_SUCCESS)
         break;

      pDescriptorSets[i] = anv_descriptor_set_to_handle(set);
   }

   if (result != VK_SUCCESS)
      anv_FreeDescriptorSets(_device, pAllocateInfo->descriptorPool,
                             i, pDescriptorSets);

   return result;
}

VkResult anv_FreeDescriptorSets(
    VkDevice                                    _device,
    VkDescriptorPool                            descriptorPool,
    uint32_t                                    count,
    const VkDescriptorSet*                      pDescriptorSets)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_descriptor_pool, pool, descriptorPool);

   for (uint32_t i = 0; i < count; i++) {
      ANV_FROM_HANDLE(anv_descriptor_set, set, pDescriptorSets[i]);

      if (!set)
         continue;

      anv_descriptor_set_destroy(device, pool, set);
   }

   return VK_SUCCESS;
}

void
anv_descriptor_set_write_image_view(struct anv_descriptor_set *set,
                                    const struct gen_device_info * const devinfo,
                                    const VkDescriptorImageInfo * const info,
                                    VkDescriptorType type,
                                    uint32_t binding,
                                    uint32_t element)
{
   const struct anv_descriptor_set_binding_layout *bind_layout =
      &set->layout->binding[binding];
   struct anv_descriptor *desc =
      &set->descriptors[bind_layout->descriptor_index + element];
   struct anv_image_view *image_view = NULL;
   struct anv_sampler *sampler = NULL;

   assert(type == bind_layout->type);

   switch (type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER:
      sampler = anv_sampler_from_handle(info->sampler);
      break;

   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      image_view = anv_image_view_from_handle(info->imageView);
      sampler = anv_sampler_from_handle(info->sampler);
      break;

   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      image_view = anv_image_view_from_handle(info->imageView);
      break;

   default:
      unreachable("invalid descriptor type");
   }

   /* If this descriptor has an immutable sampler, we don't want to stomp on
    * it.
    */
   sampler = bind_layout->immutable_samplers ?
             bind_layout->immutable_samplers[element] :
             sampler;

   *desc = (struct anv_descriptor) {
      .type = type,
      .layout = info->imageLayout,
      .image_view = image_view,
      .sampler = sampler,
   };
}

void
anv_descriptor_set_write_buffer_view(struct anv_descriptor_set *set,
                                     VkDescriptorType type,
                                     struct anv_buffer_view *buffer_view,
                                     uint32_t binding,
                                     uint32_t element)
{
   const struct anv_descriptor_set_binding_layout *bind_layout =
      &set->layout->binding[binding];
   struct anv_descriptor *desc =
      &set->descriptors[bind_layout->descriptor_index + element];

   assert(type == bind_layout->type);

   *desc = (struct anv_descriptor) {
      .type = type,
      .buffer_view = buffer_view,
   };
}

void
anv_descriptor_set_write_buffer(struct anv_descriptor_set *set,
                                struct anv_device *device,
                                struct anv_state_stream *alloc_stream,
                                VkDescriptorType type,
                                struct anv_buffer *buffer,
                                uint32_t binding,
                                uint32_t element,
                                VkDeviceSize offset,
                                VkDeviceSize range)
{
   const struct anv_descriptor_set_binding_layout *bind_layout =
      &set->layout->binding[binding];
   struct anv_descriptor *desc =
      &set->descriptors[bind_layout->descriptor_index + element];

   assert(type == bind_layout->type);

   if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
       type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
      *desc = (struct anv_descriptor) {
         .type = type,
         .buffer = buffer,
         .offset = offset,
         .range = range,
      };
   } else {
      struct anv_buffer_view *bview =
         &set->buffer_views[bind_layout->buffer_index + element];

      bview->format = anv_isl_format_for_descriptor_type(type);
      bview->range = anv_buffer_get_range(buffer, offset, range);
      bview->address = anv_address_add(buffer->address, offset);

      /* If we're writing descriptors through a push command, we need to
       * allocate the surface state from the command buffer. Otherwise it will
       * be allocated by the descriptor pool when calling
       * vkAllocateDescriptorSets. */
      if (alloc_stream)
         bview->surface_state = anv_state_stream_alloc(alloc_stream, 64, 64);

      anv_fill_buffer_surface_state(device, bview->surface_state,
                                    bview->format,
                                    bview->address, bview->range, 1);

      *desc = (struct anv_descriptor) {
         .type = type,
         .buffer_view = bview,
      };
   }
}

void anv_UpdateDescriptorSets(
    VkDevice                                    _device,
    uint32_t                                    descriptorWriteCount,
    const VkWriteDescriptorSet*                 pDescriptorWrites,
    uint32_t                                    descriptorCopyCount,
    const VkCopyDescriptorSet*                  pDescriptorCopies)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   for (uint32_t i = 0; i < descriptorWriteCount; i++) {
      const VkWriteDescriptorSet *write = &pDescriptorWrites[i];
      ANV_FROM_HANDLE(anv_descriptor_set, set, write->dstSet);

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            anv_descriptor_set_write_image_view(set, &device->info,
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
                                            device,
                                            NULL,
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

   for (uint32_t i = 0; i < descriptorCopyCount; i++) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      ANV_FROM_HANDLE(anv_descriptor_set, src, copy->srcSet);
      ANV_FROM_HANDLE(anv_descriptor_set, dst, copy->dstSet);

      const struct anv_descriptor_set_binding_layout *src_layout =
         &src->layout->binding[copy->srcBinding];
      struct anv_descriptor *src_desc =
         &src->descriptors[src_layout->descriptor_index];
      src_desc += copy->srcArrayElement;

      const struct anv_descriptor_set_binding_layout *dst_layout =
         &dst->layout->binding[copy->dstBinding];
      struct anv_descriptor *dst_desc =
         &dst->descriptors[dst_layout->descriptor_index];
      dst_desc += copy->dstArrayElement;

      for (uint32_t j = 0; j < copy->descriptorCount; j++)
         dst_desc[j] = src_desc[j];
   }
}

/*
 * Descriptor update templates.
 */

void
anv_descriptor_set_write_template(struct anv_descriptor_set *set,
                                  struct anv_device *device,
                                  struct anv_state_stream *alloc_stream,
                                  const struct anv_descriptor_update_template *template,
                                  const void *data)
{
   for (uint32_t i = 0; i < template->entry_count; i++) {
      const struct anv_descriptor_template_entry *entry =
         &template->entries[i];

      switch (entry->type) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;
            anv_descriptor_set_write_image_view(set, &device->info,
                                                info, entry->type,
                                                entry->binding,
                                                entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkBufferView *_bview =
               data + entry->offset + j * entry->stride;
            ANV_FROM_HANDLE(anv_buffer_view, bview, *_bview);

            anv_descriptor_set_write_buffer_view(set,
                                                 entry->type,
                                                 bview,
                                                 entry->binding,
                                                 entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorBufferInfo *info =
               data + entry->offset + j * entry->stride;
            ANV_FROM_HANDLE(anv_buffer, buffer, info->buffer);

            anv_descriptor_set_write_buffer(set,
                                            device,
                                            alloc_stream,
                                            entry->type,
                                            buffer,
                                            entry->binding,
                                            entry->array_element + j,
                                            info->offset, info->range);
         }
         break;

      default:
         break;
      }
   }
}

VkResult anv_CreateDescriptorUpdateTemplate(
    VkDevice                                    _device,
    const VkDescriptorUpdateTemplateCreateInfo* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorUpdateTemplate*                 pDescriptorUpdateTemplate)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_descriptor_update_template *template;

   size_t size = sizeof(*template) +
      pCreateInfo->descriptorUpdateEntryCount * sizeof(template->entries[0]);
   template = vk_alloc2(&device->alloc, pAllocator, size, 8,
                        VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (template == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   template->bind_point = pCreateInfo->pipelineBindPoint;

   if (pCreateInfo->templateType == VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET)
      template->set = pCreateInfo->set;

   template->entry_count = pCreateInfo->descriptorUpdateEntryCount;
   for (uint32_t i = 0; i < template->entry_count; i++) {
      const VkDescriptorUpdateTemplateEntryKHR *pEntry =
         &pCreateInfo->pDescriptorUpdateEntries[i];

      template->entries[i] = (struct anv_descriptor_template_entry) {
         .type = pEntry->descriptorType,
         .binding = pEntry->dstBinding,
         .array_element = pEntry->dstArrayElement,
         .array_count = pEntry->descriptorCount,
         .offset = pEntry->offset,
         .stride = pEntry->stride,
      };
   }

   *pDescriptorUpdateTemplate =
      anv_descriptor_update_template_to_handle(template);

   return VK_SUCCESS;
}

void anv_DestroyDescriptorUpdateTemplate(
    VkDevice                                    _device,
    VkDescriptorUpdateTemplate                  descriptorUpdateTemplate,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_descriptor_update_template, template,
                   descriptorUpdateTemplate);

   vk_free2(&device->alloc, pAllocator, template);
}

void anv_UpdateDescriptorSetWithTemplate(
    VkDevice                                    _device,
    VkDescriptorSet                             descriptorSet,
    VkDescriptorUpdateTemplate                  descriptorUpdateTemplate,
    const void*                                 pData)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_descriptor_set, set, descriptorSet);
   ANV_FROM_HANDLE(anv_descriptor_update_template, template,
                   descriptorUpdateTemplate);

   anv_descriptor_set_write_template(set, device, NULL, template, pData);
}
