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

#include "vk_util.h"

static void
anv_render_pass_add_subpass_dep(struct anv_render_pass *pass,
                                const VkSubpassDependency2KHR *dep)
{
   if (dep->dstSubpass == VK_SUBPASS_EXTERNAL) {
      pass->subpass_flushes[pass->subpass_count] |=
         anv_pipe_invalidate_bits_for_access_flags(dep->dstAccessMask);
   } else {
      assert(dep->dstSubpass < pass->subpass_count);
      pass->subpass_flushes[dep->dstSubpass] |=
         anv_pipe_invalidate_bits_for_access_flags(dep->dstAccessMask);
   }

   if (dep->srcSubpass == VK_SUBPASS_EXTERNAL) {
      pass->subpass_flushes[0] |=
         anv_pipe_flush_bits_for_access_flags(dep->srcAccessMask);
   } else {
      assert(dep->srcSubpass < pass->subpass_count);
      pass->subpass_flushes[dep->srcSubpass + 1] |=
         anv_pipe_flush_bits_for_access_flags(dep->srcAccessMask);
   }
}

/* Do a second "compile" step on a render pass */
static void
anv_render_pass_compile(struct anv_render_pass *pass)
{
   /* The CreateRenderPass code zeros the entire render pass and also uses a
    * designated initializer for filling these out.  There's no need for us to
    * do it again.
    *
    * for (uint32_t i = 0; i < pass->attachment_count; i++) {
    *    pass->attachments[i].usage = 0;
    *    pass->attachments[i].first_subpass_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    * }
    */

   VkImageUsageFlags all_usage = 0;
   for (uint32_t i = 0; i < pass->subpass_count; i++) {
      struct anv_subpass *subpass = &pass->subpasses[i];

      /* We don't allow depth_stencil_attachment to be non-NULL and be
       * VK_ATTACHMENT_UNUSED.  This way something can just check for NULL
       * and be guaranteed that they have a valid attachment.
       */
      if (subpass->depth_stencil_attachment &&
          subpass->depth_stencil_attachment->attachment == VK_ATTACHMENT_UNUSED)
         subpass->depth_stencil_attachment = NULL;

      for (uint32_t j = 0; j < subpass->attachment_count; j++) {
         struct anv_subpass_attachment *subpass_att = &subpass->attachments[j];
         if (subpass_att->attachment == VK_ATTACHMENT_UNUSED)
            continue;

         struct anv_render_pass_attachment *pass_att =
            &pass->attachments[subpass_att->attachment];

         assert(__builtin_popcount(subpass_att->usage) == 1);
         pass_att->usage |= subpass_att->usage;
         pass_att->last_subpass_idx = i;

         all_usage |= subpass_att->usage;

         if (pass_att->first_subpass_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            pass_att->first_subpass_layout = subpass_att->layout;
            assert(pass_att->first_subpass_layout != VK_IMAGE_LAYOUT_UNDEFINED);
         }

         if (subpass_att->usage == VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT &&
             subpass->depth_stencil_attachment &&
             subpass_att->attachment == subpass->depth_stencil_attachment->attachment)
            subpass->has_ds_self_dep = true;
      }

      /* We have to handle resolve attachments specially */
      subpass->has_resolve = false;
      if (subpass->resolve_attachments) {
         for (uint32_t j = 0; j < subpass->color_count; j++) {
            struct anv_subpass_attachment *color_att =
               &subpass->color_attachments[j];
            struct anv_subpass_attachment *resolve_att =
               &subpass->resolve_attachments[j];
            if (resolve_att->attachment == VK_ATTACHMENT_UNUSED)
               continue;

            subpass->has_resolve = true;

            assert(resolve_att->usage == VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            color_att->usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
         }
      }
   }

   /* From the Vulkan 1.0.39 spec:
    *
    *    If there is no subpass dependency from VK_SUBPASS_EXTERNAL to the
    *    first subpass that uses an attachment, then an implicit subpass
    *    dependency exists from VK_SUBPASS_EXTERNAL to the first subpass it is
    *    used in. The subpass dependency operates as if defined with the
    *    following parameters:
    *
    *    VkSubpassDependency implicitDependency = {
    *        .srcSubpass = VK_SUBPASS_EXTERNAL;
    *        .dstSubpass = firstSubpass; // First subpass attachment is used in
    *        .srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    *        .dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    *        .srcAccessMask = 0;
    *        .dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
    *                         VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
    *                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
    *                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
    *                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    *        .dependencyFlags = 0;
    *    };
    *
    *    Similarly, if there is no subpass dependency from the last subpass
    *    that uses an attachment to VK_SUBPASS_EXTERNAL, then an implicit
    *    subpass dependency exists from the last subpass it is used in to
    *    VK_SUBPASS_EXTERNAL. The subpass dependency operates as if defined
    *    with the following parameters:
    *
    *    VkSubpassDependency implicitDependency = {
    *        .srcSubpass = lastSubpass; // Last subpass attachment is used in
    *        .dstSubpass = VK_SUBPASS_EXTERNAL;
    *        .srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    *        .dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    *        .srcAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
    *                         VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
    *                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
    *                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
    *                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    *        .dstAccessMask = 0;
    *        .dependencyFlags = 0;
    *    };
    *
    * We could implement this by walking over all of the attachments and
    * subpasses and checking to see if any of them don't have an external
    * dependency.  Or, we could just be lazy and add a couple extra flushes.
    * We choose to be lazy.
    */
   if (all_usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) {
      pass->subpass_flushes[0] |=
         ANV_PIPE_TEXTURE_CACHE_INVALIDATE_BIT;
   }
   if (all_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
      pass->subpass_flushes[pass->subpass_count] |=
         ANV_PIPE_RENDER_TARGET_CACHE_FLUSH_BIT;
   }
   if (all_usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
      pass->subpass_flushes[pass->subpass_count] |=
         ANV_PIPE_DEPTH_CACHE_FLUSH_BIT;
   }
}

static unsigned
num_subpass_attachments(const VkSubpassDescription *desc)
{
   return desc->inputAttachmentCount +
          desc->colorAttachmentCount +
          (desc->pResolveAttachments ? desc->colorAttachmentCount : 0) +
          (desc->pDepthStencilAttachment != NULL);
}

VkResult anv_CreateRenderPass(
    VkDevice                                    _device,
    const VkRenderPassCreateInfo*               pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkRenderPass*                               pRenderPass)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO);

   struct anv_render_pass *pass;
   struct anv_subpass *subpasses;
   struct anv_render_pass_attachment *attachments;
   enum anv_pipe_bits *subpass_flushes;

   ANV_MULTIALLOC(ma);
   anv_multialloc_add(&ma, &pass, 1);
   anv_multialloc_add(&ma, &subpasses, pCreateInfo->subpassCount);
   anv_multialloc_add(&ma, &attachments, pCreateInfo->attachmentCount);
   anv_multialloc_add(&ma, &subpass_flushes, pCreateInfo->subpassCount + 1);

   struct anv_subpass_attachment *subpass_attachments;
   uint32_t subpass_attachment_count = 0;
   for (uint32_t i = 0; i < pCreateInfo->subpassCount; i++) {
      subpass_attachment_count +=
         num_subpass_attachments(&pCreateInfo->pSubpasses[i]);
   }
   anv_multialloc_add(&ma, &subpass_attachments, subpass_attachment_count);

   if (!anv_multialloc_alloc2(&ma, &device->alloc, pAllocator,
                              VK_SYSTEM_ALLOCATION_SCOPE_OBJECT))
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Clear the subpasses along with the parent pass. This required because
    * each array member of anv_subpass must be a valid pointer if not NULL.
    */
   memset(pass, 0, ma.size);
   pass->attachment_count = pCreateInfo->attachmentCount;
   pass->subpass_count = pCreateInfo->subpassCount;
   pass->attachments = attachments;
   pass->subpass_flushes = subpass_flushes;

   for (uint32_t i = 0; i < pCreateInfo->attachmentCount; i++) {
      pass->attachments[i] = (struct anv_render_pass_attachment) {
         .format                 = pCreateInfo->pAttachments[i].format,
         .samples                = pCreateInfo->pAttachments[i].samples,
         .load_op                = pCreateInfo->pAttachments[i].loadOp,
         .store_op               = pCreateInfo->pAttachments[i].storeOp,
         .stencil_load_op        = pCreateInfo->pAttachments[i].stencilLoadOp,
         .initial_layout         = pCreateInfo->pAttachments[i].initialLayout,
         .final_layout           = pCreateInfo->pAttachments[i].finalLayout,
      };
   }

   for (uint32_t i = 0; i < pCreateInfo->subpassCount; i++) {
      const VkSubpassDescription *desc = &pCreateInfo->pSubpasses[i];
      struct anv_subpass *subpass = &pass->subpasses[i];

      subpass->input_count = desc->inputAttachmentCount;
      subpass->color_count = desc->colorAttachmentCount;
      subpass->attachment_count = num_subpass_attachments(desc);
      subpass->attachments = subpass_attachments;
      subpass->view_mask = 0;

      if (desc->inputAttachmentCount > 0) {
         subpass->input_attachments = subpass_attachments;
         subpass_attachments += desc->inputAttachmentCount;

         for (uint32_t j = 0; j < desc->inputAttachmentCount; j++) {
            subpass->input_attachments[j] = (struct anv_subpass_attachment) {
               .usage =       VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
               .attachment =  desc->pInputAttachments[j].attachment,
               .layout =      desc->pInputAttachments[j].layout,
            };
         }
      }

      if (desc->colorAttachmentCount > 0) {
         subpass->color_attachments = subpass_attachments;
         subpass_attachments += desc->colorAttachmentCount;

         for (uint32_t j = 0; j < desc->colorAttachmentCount; j++) {
            subpass->color_attachments[j] = (struct anv_subpass_attachment) {
               .usage =       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
               .attachment =  desc->pColorAttachments[j].attachment,
               .layout =      desc->pColorAttachments[j].layout,
            };
         }
      }

      if (desc->pResolveAttachments) {
         subpass->resolve_attachments = subpass_attachments;
         subpass_attachments += desc->colorAttachmentCount;

         for (uint32_t j = 0; j < desc->colorAttachmentCount; j++) {
            subpass->resolve_attachments[j] = (struct anv_subpass_attachment) {
               .usage =       VK_IMAGE_USAGE_TRANSFER_DST_BIT,
               .attachment =  desc->pResolveAttachments[j].attachment,
               .layout =      desc->pResolveAttachments[j].layout,
            };
         }
      }

      if (desc->pDepthStencilAttachment) {
         subpass->depth_stencil_attachment = subpass_attachments++;

         *subpass->depth_stencil_attachment = (struct anv_subpass_attachment) {
            .usage =       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .attachment =  desc->pDepthStencilAttachment->attachment,
            .layout =      desc->pDepthStencilAttachment->layout,
         };
      }
   }

   for (uint32_t i = 0; i < pCreateInfo->dependencyCount; i++) {
      /* Convert to a Dependency2KHR */
      struct VkSubpassDependency2KHR dep2 = {
         .srcSubpass       = pCreateInfo->pDependencies[i].srcSubpass,
         .dstSubpass       = pCreateInfo->pDependencies[i].dstSubpass,
         .srcStageMask     = pCreateInfo->pDependencies[i].srcStageMask,
         .dstStageMask     = pCreateInfo->pDependencies[i].dstStageMask,
         .srcAccessMask    = pCreateInfo->pDependencies[i].srcAccessMask,
         .dstAccessMask    = pCreateInfo->pDependencies[i].dstAccessMask,
         .dependencyFlags  = pCreateInfo->pDependencies[i].dependencyFlags,
      };
      anv_render_pass_add_subpass_dep(pass, &dep2);
   }

   vk_foreach_struct(ext, pCreateInfo->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO_KHR: {
         VkRenderPassMultiviewCreateInfoKHR *mv = (void *)ext;

         for (uint32_t i = 0; i < mv->subpassCount; i++) {
            pass->subpasses[i].view_mask = mv->pViewMasks[i];
         }
         break;
      }

      default:
         anv_debug_ignored_stype(ext->sType);
      }
   }

   anv_render_pass_compile(pass);

   *pRenderPass = anv_render_pass_to_handle(pass);

   return VK_SUCCESS;
}

static unsigned
num_subpass_attachments2(const VkSubpassDescription2KHR *desc)
{
   return desc->inputAttachmentCount +
          desc->colorAttachmentCount +
          (desc->pResolveAttachments ? desc->colorAttachmentCount : 0) +
          (desc->pDepthStencilAttachment != NULL);
}

VkResult anv_CreateRenderPass2KHR(
    VkDevice                                    _device,
    const VkRenderPassCreateInfo2KHR*           pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkRenderPass*                               pRenderPass)
{
   ANV_FROM_HANDLE(anv_device, device, _device);

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2_KHR);

   struct anv_render_pass *pass;
   struct anv_subpass *subpasses;
   struct anv_render_pass_attachment *attachments;
   enum anv_pipe_bits *subpass_flushes;

   ANV_MULTIALLOC(ma);
   anv_multialloc_add(&ma, &pass, 1);
   anv_multialloc_add(&ma, &subpasses, pCreateInfo->subpassCount);
   anv_multialloc_add(&ma, &attachments, pCreateInfo->attachmentCount);
   anv_multialloc_add(&ma, &subpass_flushes, pCreateInfo->subpassCount + 1);

   struct anv_subpass_attachment *subpass_attachments;
   uint32_t subpass_attachment_count = 0;
   for (uint32_t i = 0; i < pCreateInfo->subpassCount; i++) {
      subpass_attachment_count +=
         num_subpass_attachments2(&pCreateInfo->pSubpasses[i]);
   }
   anv_multialloc_add(&ma, &subpass_attachments, subpass_attachment_count);

   if (!anv_multialloc_alloc2(&ma, &device->alloc, pAllocator,
                              VK_SYSTEM_ALLOCATION_SCOPE_OBJECT))
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Clear the subpasses along with the parent pass. This required because
    * each array member of anv_subpass must be a valid pointer if not NULL.
    */
   memset(pass, 0, ma.size);
   pass->attachment_count = pCreateInfo->attachmentCount;
   pass->subpass_count = pCreateInfo->subpassCount;
   pass->attachments = attachments;
   pass->subpass_flushes = subpass_flushes;

   for (uint32_t i = 0; i < pCreateInfo->attachmentCount; i++) {
      pass->attachments[i] = (struct anv_render_pass_attachment) {
         .format                 = pCreateInfo->pAttachments[i].format,
         .samples                = pCreateInfo->pAttachments[i].samples,
         .load_op                = pCreateInfo->pAttachments[i].loadOp,
         .store_op               = pCreateInfo->pAttachments[i].storeOp,
         .stencil_load_op        = pCreateInfo->pAttachments[i].stencilLoadOp,
         .initial_layout         = pCreateInfo->pAttachments[i].initialLayout,
         .final_layout           = pCreateInfo->pAttachments[i].finalLayout,
      };
   }

   for (uint32_t i = 0; i < pCreateInfo->subpassCount; i++) {
      const VkSubpassDescription2KHR *desc = &pCreateInfo->pSubpasses[i];
      struct anv_subpass *subpass = &pass->subpasses[i];

      subpass->input_count = desc->inputAttachmentCount;
      subpass->color_count = desc->colorAttachmentCount;
      subpass->attachment_count = num_subpass_attachments2(desc);
      subpass->attachments = subpass_attachments;
      subpass->view_mask = desc->viewMask;

      if (desc->inputAttachmentCount > 0) {
         subpass->input_attachments = subpass_attachments;
         subpass_attachments += desc->inputAttachmentCount;

         for (uint32_t j = 0; j < desc->inputAttachmentCount; j++) {
            subpass->input_attachments[j] = (struct anv_subpass_attachment) {
               .usage =       VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
               .attachment =  desc->pInputAttachments[j].attachment,
               .layout =      desc->pInputAttachments[j].layout,
            };
         }
      }

      if (desc->colorAttachmentCount > 0) {
         subpass->color_attachments = subpass_attachments;
         subpass_attachments += desc->colorAttachmentCount;

         for (uint32_t j = 0; j < desc->colorAttachmentCount; j++) {
            subpass->color_attachments[j] = (struct anv_subpass_attachment) {
               .usage =       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
               .attachment =  desc->pColorAttachments[j].attachment,
               .layout =      desc->pColorAttachments[j].layout,
            };
         }
      }

      if (desc->pResolveAttachments) {
         subpass->resolve_attachments = subpass_attachments;
         subpass_attachments += desc->colorAttachmentCount;

         for (uint32_t j = 0; j < desc->colorAttachmentCount; j++) {
            subpass->resolve_attachments[j] = (struct anv_subpass_attachment) {
               .usage =       VK_IMAGE_USAGE_TRANSFER_DST_BIT,
               .attachment =  desc->pResolveAttachments[j].attachment,
               .layout =      desc->pResolveAttachments[j].layout,
            };
         }
      }

      if (desc->pDepthStencilAttachment) {
         subpass->depth_stencil_attachment = subpass_attachments++;

         *subpass->depth_stencil_attachment = (struct anv_subpass_attachment) {
            .usage =       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .attachment =  desc->pDepthStencilAttachment->attachment,
            .layout =      desc->pDepthStencilAttachment->layout,
         };
      }
   }

   for (uint32_t i = 0; i < pCreateInfo->dependencyCount; i++)
      anv_render_pass_add_subpass_dep(pass, &pCreateInfo->pDependencies[i]);

   vk_foreach_struct(ext, pCreateInfo->pNext) {
      switch (ext->sType) {
      default:
         anv_debug_ignored_stype(ext->sType);
      }
   }

   anv_render_pass_compile(pass);

   *pRenderPass = anv_render_pass_to_handle(pass);

   return VK_SUCCESS;
}

void anv_DestroyRenderPass(
    VkDevice                                    _device,
    VkRenderPass                                _pass,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_render_pass, pass, _pass);

   vk_free2(&device->alloc, pAllocator, pass);
}

void anv_GetRenderAreaGranularity(
    VkDevice                                    device,
    VkRenderPass                                renderPass,
    VkExtent2D*                                 pGranularity)
{
   ANV_FROM_HANDLE(anv_render_pass, pass, renderPass);

   /* This granularity satisfies HiZ fast clear alignment requirements
    * for all sample counts.
    */
   for (unsigned i = 0; i < pass->subpass_count; ++i) {
      if (pass->subpasses[i].depth_stencil_attachment) {
         *pGranularity = (VkExtent2D) { .width = 8, .height = 4 };
         return;
      }
   }

   *pGranularity = (VkExtent2D) { 1, 1 };
}
