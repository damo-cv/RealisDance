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

#include "genxml/gen8_pack.h"

#include "util/debug.h"

/** \file anv_batch_chain.c
 *
 * This file contains functions related to anv_cmd_buffer as a data
 * structure.  This involves everything required to create and destroy
 * the actual batch buffers as well as link them together and handle
 * relocations and surface state.  It specifically does *not* contain any
 * handling of actual vkCmd calls beyond vkCmdExecuteCommands.
 */

/*-----------------------------------------------------------------------*
 * Functions related to anv_reloc_list
 *-----------------------------------------------------------------------*/

static VkResult
anv_reloc_list_init_clone(struct anv_reloc_list *list,
                          const VkAllocationCallbacks *alloc,
                          const struct anv_reloc_list *other_list)
{
   if (other_list) {
      list->num_relocs = other_list->num_relocs;
      list->array_length = other_list->array_length;
   } else {
      list->num_relocs = 0;
      list->array_length = 256;
   }

   list->relocs =
      vk_alloc(alloc, list->array_length * sizeof(*list->relocs), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

   if (list->relocs == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   list->reloc_bos =
      vk_alloc(alloc, list->array_length * sizeof(*list->reloc_bos), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

   if (list->reloc_bos == NULL) {
      vk_free(alloc, list->relocs);
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   list->deps = _mesa_set_create(NULL, _mesa_hash_pointer,
                                 _mesa_key_pointer_equal);

   if (!list->deps) {
      vk_free(alloc, list->relocs);
      vk_free(alloc, list->reloc_bos);
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   if (other_list) {
      memcpy(list->relocs, other_list->relocs,
             list->array_length * sizeof(*list->relocs));
      memcpy(list->reloc_bos, other_list->reloc_bos,
             list->array_length * sizeof(*list->reloc_bos));
      set_foreach(other_list->deps, entry) {
         _mesa_set_add_pre_hashed(list->deps, entry->hash, entry->key);
      }
   }

   return VK_SUCCESS;
}

VkResult
anv_reloc_list_init(struct anv_reloc_list *list,
                    const VkAllocationCallbacks *alloc)
{
   return anv_reloc_list_init_clone(list, alloc, NULL);
}

void
anv_reloc_list_finish(struct anv_reloc_list *list,
                      const VkAllocationCallbacks *alloc)
{
   vk_free(alloc, list->relocs);
   vk_free(alloc, list->reloc_bos);
   _mesa_set_destroy(list->deps, NULL);
}

static VkResult
anv_reloc_list_grow(struct anv_reloc_list *list,
                    const VkAllocationCallbacks *alloc,
                    size_t num_additional_relocs)
{
   if (list->num_relocs + num_additional_relocs <= list->array_length)
      return VK_SUCCESS;

   size_t new_length = list->array_length * 2;
   while (new_length < list->num_relocs + num_additional_relocs)
      new_length *= 2;

   struct drm_i915_gem_relocation_entry *new_relocs =
      vk_alloc(alloc, new_length * sizeof(*list->relocs), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (new_relocs == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   struct anv_bo **new_reloc_bos =
      vk_alloc(alloc, new_length * sizeof(*list->reloc_bos), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (new_reloc_bos == NULL) {
      vk_free(alloc, new_relocs);
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   memcpy(new_relocs, list->relocs, list->num_relocs * sizeof(*list->relocs));
   memcpy(new_reloc_bos, list->reloc_bos,
          list->num_relocs * sizeof(*list->reloc_bos));

   vk_free(alloc, list->relocs);
   vk_free(alloc, list->reloc_bos);

   list->array_length = new_length;
   list->relocs = new_relocs;
   list->reloc_bos = new_reloc_bos;

   return VK_SUCCESS;
}

VkResult
anv_reloc_list_add(struct anv_reloc_list *list,
                   const VkAllocationCallbacks *alloc,
                   uint32_t offset, struct anv_bo *target_bo, uint32_t delta)
{
   struct drm_i915_gem_relocation_entry *entry;
   int index;

   if (target_bo->flags & EXEC_OBJECT_PINNED) {
      _mesa_set_add(list->deps, target_bo);
      return VK_SUCCESS;
   }

   VkResult result = anv_reloc_list_grow(list, alloc, 1);
   if (result != VK_SUCCESS)
      return result;

   /* XXX: Can we use I915_EXEC_HANDLE_LUT? */
   index = list->num_relocs++;
   list->reloc_bos[index] = target_bo;
   entry = &list->relocs[index];
   entry->target_handle = target_bo->gem_handle;
   entry->delta = delta;
   entry->offset = offset;
   entry->presumed_offset = target_bo->offset;
   entry->read_domains = 0;
   entry->write_domain = 0;
   VG(VALGRIND_CHECK_MEM_IS_DEFINED(entry, sizeof(*entry)));

   return VK_SUCCESS;
}

static VkResult
anv_reloc_list_append(struct anv_reloc_list *list,
                      const VkAllocationCallbacks *alloc,
                      struct anv_reloc_list *other, uint32_t offset)
{
   VkResult result = anv_reloc_list_grow(list, alloc, other->num_relocs);
   if (result != VK_SUCCESS)
      return result;

   memcpy(&list->relocs[list->num_relocs], &other->relocs[0],
          other->num_relocs * sizeof(other->relocs[0]));
   memcpy(&list->reloc_bos[list->num_relocs], &other->reloc_bos[0],
          other->num_relocs * sizeof(other->reloc_bos[0]));

   for (uint32_t i = 0; i < other->num_relocs; i++)
      list->relocs[i + list->num_relocs].offset += offset;

   list->num_relocs += other->num_relocs;

   set_foreach(other->deps, entry) {
      _mesa_set_add_pre_hashed(list->deps, entry->hash, entry->key);
   }

   return VK_SUCCESS;
}

/*-----------------------------------------------------------------------*
 * Functions related to anv_batch
 *-----------------------------------------------------------------------*/

void *
anv_batch_emit_dwords(struct anv_batch *batch, int num_dwords)
{
   if (batch->next + num_dwords * 4 > batch->end) {
      VkResult result = batch->extend_cb(batch, batch->user_data);
      if (result != VK_SUCCESS) {
         anv_batch_set_error(batch, result);
         return NULL;
      }
   }

   void *p = batch->next;

   batch->next += num_dwords * 4;
   assert(batch->next <= batch->end);

   return p;
}

uint64_t
anv_batch_emit_reloc(struct anv_batch *batch,
                     void *location, struct anv_bo *bo, uint32_t delta)
{
   VkResult result = anv_reloc_list_add(batch->relocs, batch->alloc,
                                        location - batch->start, bo, delta);
   if (result != VK_SUCCESS) {
      anv_batch_set_error(batch, result);
      return 0;
   }

   return bo->offset + delta;
}

void
anv_batch_emit_batch(struct anv_batch *batch, struct anv_batch *other)
{
   uint32_t size, offset;

   size = other->next - other->start;
   assert(size % 4 == 0);

   if (batch->next + size > batch->end) {
      VkResult result = batch->extend_cb(batch, batch->user_data);
      if (result != VK_SUCCESS) {
         anv_batch_set_error(batch, result);
         return;
      }
   }

   assert(batch->next + size <= batch->end);

   VG(VALGRIND_CHECK_MEM_IS_DEFINED(other->start, size));
   memcpy(batch->next, other->start, size);

   offset = batch->next - batch->start;
   VkResult result = anv_reloc_list_append(batch->relocs, batch->alloc,
                                           other->relocs, offset);
   if (result != VK_SUCCESS) {
      anv_batch_set_error(batch, result);
      return;
   }

   batch->next += size;
}

/*-----------------------------------------------------------------------*
 * Functions related to anv_batch_bo
 *-----------------------------------------------------------------------*/

static VkResult
anv_batch_bo_create(struct anv_cmd_buffer *cmd_buffer,
                    struct anv_batch_bo **bbo_out)
{
   VkResult result;

   struct anv_batch_bo *bbo = vk_alloc(&cmd_buffer->pool->alloc, sizeof(*bbo),
                                        8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (bbo == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   result = anv_bo_pool_alloc(&cmd_buffer->device->batch_bo_pool, &bbo->bo,
                              ANV_CMD_BUFFER_BATCH_SIZE);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   result = anv_reloc_list_init(&bbo->relocs, &cmd_buffer->pool->alloc);
   if (result != VK_SUCCESS)
      goto fail_bo_alloc;

   *bbo_out = bbo;

   return VK_SUCCESS;

 fail_bo_alloc:
   anv_bo_pool_free(&cmd_buffer->device->batch_bo_pool, &bbo->bo);
 fail_alloc:
   vk_free(&cmd_buffer->pool->alloc, bbo);

   return result;
}

static VkResult
anv_batch_bo_clone(struct anv_cmd_buffer *cmd_buffer,
                   const struct anv_batch_bo *other_bbo,
                   struct anv_batch_bo **bbo_out)
{
   VkResult result;

   struct anv_batch_bo *bbo = vk_alloc(&cmd_buffer->pool->alloc, sizeof(*bbo),
                                        8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (bbo == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   result = anv_bo_pool_alloc(&cmd_buffer->device->batch_bo_pool, &bbo->bo,
                              other_bbo->bo.size);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   result = anv_reloc_list_init_clone(&bbo->relocs, &cmd_buffer->pool->alloc,
                                      &other_bbo->relocs);
   if (result != VK_SUCCESS)
      goto fail_bo_alloc;

   bbo->length = other_bbo->length;
   memcpy(bbo->bo.map, other_bbo->bo.map, other_bbo->length);

   *bbo_out = bbo;

   return VK_SUCCESS;

 fail_bo_alloc:
   anv_bo_pool_free(&cmd_buffer->device->batch_bo_pool, &bbo->bo);
 fail_alloc:
   vk_free(&cmd_buffer->pool->alloc, bbo);

   return result;
}

static void
anv_batch_bo_start(struct anv_batch_bo *bbo, struct anv_batch *batch,
                   size_t batch_padding)
{
   batch->next = batch->start = bbo->bo.map;
   batch->end = bbo->bo.map + bbo->bo.size - batch_padding;
   batch->relocs = &bbo->relocs;
   bbo->relocs.num_relocs = 0;
   _mesa_set_clear(bbo->relocs.deps, NULL);
}

static void
anv_batch_bo_continue(struct anv_batch_bo *bbo, struct anv_batch *batch,
                      size_t batch_padding)
{
   batch->start = bbo->bo.map;
   batch->next = bbo->bo.map + bbo->length;
   batch->end = bbo->bo.map + bbo->bo.size - batch_padding;
   batch->relocs = &bbo->relocs;
}

static void
anv_batch_bo_finish(struct anv_batch_bo *bbo, struct anv_batch *batch)
{
   assert(batch->start == bbo->bo.map);
   bbo->length = batch->next - batch->start;
   VG(VALGRIND_CHECK_MEM_IS_DEFINED(batch->start, bbo->length));
}

static VkResult
anv_batch_bo_grow(struct anv_cmd_buffer *cmd_buffer, struct anv_batch_bo *bbo,
                  struct anv_batch *batch, size_t aditional,
                  size_t batch_padding)
{
   assert(batch->start == bbo->bo.map);
   bbo->length = batch->next - batch->start;

   size_t new_size = bbo->bo.size;
   while (new_size <= bbo->length + aditional + batch_padding)
      new_size *= 2;

   if (new_size == bbo->bo.size)
      return VK_SUCCESS;

   struct anv_bo new_bo;
   VkResult result = anv_bo_pool_alloc(&cmd_buffer->device->batch_bo_pool,
                                       &new_bo, new_size);
   if (result != VK_SUCCESS)
      return result;

   memcpy(new_bo.map, bbo->bo.map, bbo->length);

   anv_bo_pool_free(&cmd_buffer->device->batch_bo_pool, &bbo->bo);

   bbo->bo = new_bo;
   anv_batch_bo_continue(bbo, batch, batch_padding);

   return VK_SUCCESS;
}

static void
anv_batch_bo_link(struct anv_cmd_buffer *cmd_buffer,
                  struct anv_batch_bo *prev_bbo,
                  struct anv_batch_bo *next_bbo,
                  uint32_t next_bbo_offset)
{
   MAYBE_UNUSED const uint32_t bb_start_offset =
      prev_bbo->length - GEN8_MI_BATCH_BUFFER_START_length * 4;
   MAYBE_UNUSED const uint32_t *bb_start = prev_bbo->bo.map + bb_start_offset;

   /* Make sure we're looking at a MI_BATCH_BUFFER_START */
   assert(((*bb_start >> 29) & 0x07) == 0);
   assert(((*bb_start >> 23) & 0x3f) == 49);

   if (cmd_buffer->device->instance->physicalDevice.use_softpin) {
      assert(prev_bbo->bo.flags & EXEC_OBJECT_PINNED);
      assert(next_bbo->bo.flags & EXEC_OBJECT_PINNED);

      write_reloc(cmd_buffer->device,
                  prev_bbo->bo.map + bb_start_offset + 4,
                  next_bbo->bo.offset + next_bbo_offset, true);
   } else {
      uint32_t reloc_idx = prev_bbo->relocs.num_relocs - 1;
      assert(prev_bbo->relocs.relocs[reloc_idx].offset == bb_start_offset + 4);

      prev_bbo->relocs.reloc_bos[reloc_idx] = &next_bbo->bo;
      prev_bbo->relocs.relocs[reloc_idx].delta = next_bbo_offset;

      /* Use a bogus presumed offset to force a relocation */
      prev_bbo->relocs.relocs[reloc_idx].presumed_offset = -1;
   }
}

static void
anv_batch_bo_destroy(struct anv_batch_bo *bbo,
                     struct anv_cmd_buffer *cmd_buffer)
{
   anv_reloc_list_finish(&bbo->relocs, &cmd_buffer->pool->alloc);
   anv_bo_pool_free(&cmd_buffer->device->batch_bo_pool, &bbo->bo);
   vk_free(&cmd_buffer->pool->alloc, bbo);
}

static VkResult
anv_batch_bo_list_clone(const struct list_head *list,
                        struct anv_cmd_buffer *cmd_buffer,
                        struct list_head *new_list)
{
   VkResult result = VK_SUCCESS;

   list_inithead(new_list);

   struct anv_batch_bo *prev_bbo = NULL;
   list_for_each_entry(struct anv_batch_bo, bbo, list, link) {
      struct anv_batch_bo *new_bbo = NULL;
      result = anv_batch_bo_clone(cmd_buffer, bbo, &new_bbo);
      if (result != VK_SUCCESS)
         break;
      list_addtail(&new_bbo->link, new_list);

      if (prev_bbo)
         anv_batch_bo_link(cmd_buffer, prev_bbo, new_bbo, 0);

      prev_bbo = new_bbo;
   }

   if (result != VK_SUCCESS) {
      list_for_each_entry_safe(struct anv_batch_bo, bbo, new_list, link)
         anv_batch_bo_destroy(bbo, cmd_buffer);
   }

   return result;
}

/*-----------------------------------------------------------------------*
 * Functions related to anv_batch_bo
 *-----------------------------------------------------------------------*/

static struct anv_batch_bo *
anv_cmd_buffer_current_batch_bo(struct anv_cmd_buffer *cmd_buffer)
{
   return LIST_ENTRY(struct anv_batch_bo, cmd_buffer->batch_bos.prev, link);
}

struct anv_address
anv_cmd_buffer_surface_base_address(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_state *bt_block = u_vector_head(&cmd_buffer->bt_block_states);
   return (struct anv_address) {
      .bo = &anv_binding_table_pool(cmd_buffer->device)->block_pool.bo,
      .offset = bt_block->offset,
   };
}

static void
emit_batch_buffer_start(struct anv_cmd_buffer *cmd_buffer,
                        struct anv_bo *bo, uint32_t offset)
{
   /* In gen8+ the address field grew to two dwords to accomodate 48 bit
    * offsets. The high 16 bits are in the last dword, so we can use the gen8
    * version in either case, as long as we set the instruction length in the
    * header accordingly.  This means that we always emit three dwords here
    * and all the padding and adjustment we do in this file works for all
    * gens.
    */

#define GEN7_MI_BATCH_BUFFER_START_length      2
#define GEN7_MI_BATCH_BUFFER_START_length_bias      2

   const uint32_t gen7_length =
      GEN7_MI_BATCH_BUFFER_START_length - GEN7_MI_BATCH_BUFFER_START_length_bias;
   const uint32_t gen8_length =
      GEN8_MI_BATCH_BUFFER_START_length - GEN8_MI_BATCH_BUFFER_START_length_bias;

   anv_batch_emit(&cmd_buffer->batch, GEN8_MI_BATCH_BUFFER_START, bbs) {
      bbs.DWordLength               = cmd_buffer->device->info.gen < 8 ?
                                      gen7_length : gen8_length;
      bbs.SecondLevelBatchBuffer    = Firstlevelbatch;
      bbs.AddressSpaceIndicator     = ASI_PPGTT;
      bbs.BatchBufferStartAddress   = (struct anv_address) { bo, offset };
   }
}

static void
cmd_buffer_chain_to_batch_bo(struct anv_cmd_buffer *cmd_buffer,
                             struct anv_batch_bo *bbo)
{
   struct anv_batch *batch = &cmd_buffer->batch;
   struct anv_batch_bo *current_bbo =
      anv_cmd_buffer_current_batch_bo(cmd_buffer);

   /* We set the end of the batch a little short so we would be sure we
    * have room for the chaining command.  Since we're about to emit the
    * chaining command, let's set it back where it should go.
    */
   batch->end += GEN8_MI_BATCH_BUFFER_START_length * 4;
   assert(batch->end == current_bbo->bo.map + current_bbo->bo.size);

   emit_batch_buffer_start(cmd_buffer, &bbo->bo, 0);

   anv_batch_bo_finish(current_bbo, batch);
}

static VkResult
anv_cmd_buffer_chain_batch(struct anv_batch *batch, void *_data)
{
   struct anv_cmd_buffer *cmd_buffer = _data;
   struct anv_batch_bo *new_bbo;

   VkResult result = anv_batch_bo_create(cmd_buffer, &new_bbo);
   if (result != VK_SUCCESS)
      return result;

   struct anv_batch_bo **seen_bbo = u_vector_add(&cmd_buffer->seen_bbos);
   if (seen_bbo == NULL) {
      anv_batch_bo_destroy(new_bbo, cmd_buffer);
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   *seen_bbo = new_bbo;

   cmd_buffer_chain_to_batch_bo(cmd_buffer, new_bbo);

   list_addtail(&new_bbo->link, &cmd_buffer->batch_bos);

   anv_batch_bo_start(new_bbo, batch, GEN8_MI_BATCH_BUFFER_START_length * 4);

   return VK_SUCCESS;
}

static VkResult
anv_cmd_buffer_grow_batch(struct anv_batch *batch, void *_data)
{
   struct anv_cmd_buffer *cmd_buffer = _data;
   struct anv_batch_bo *bbo = anv_cmd_buffer_current_batch_bo(cmd_buffer);

   anv_batch_bo_grow(cmd_buffer, bbo, &cmd_buffer->batch, 4096,
                     GEN8_MI_BATCH_BUFFER_START_length * 4);

   return VK_SUCCESS;
}

/** Allocate a binding table
 *
 * This function allocates a binding table.  This is a bit more complicated
 * than one would think due to a combination of Vulkan driver design and some
 * unfortunate hardware restrictions.
 *
 * The 3DSTATE_BINDING_TABLE_POINTERS_* packets only have a 16-bit field for
 * the binding table pointer which means that all binding tables need to live
 * in the bottom 64k of surface state base address.  The way the GL driver has
 * classically dealt with this restriction is to emit all surface states
 * on-the-fly into the batch and have a batch buffer smaller than 64k.  This
 * isn't really an option in Vulkan for a couple of reasons:
 *
 *  1) In Vulkan, we have growing (or chaining) batches so surface states have
 *     to live in their own buffer and we have to be able to re-emit
 *     STATE_BASE_ADDRESS as needed which requires a full pipeline stall.  In
 *     order to avoid emitting STATE_BASE_ADDRESS any more often than needed
 *     (it's not that hard to hit 64k of just binding tables), we allocate
 *     surface state objects up-front when VkImageView is created.  In order
 *     for this to work, surface state objects need to be allocated from a
 *     global buffer.
 *
 *  2) We tried to design the surface state system in such a way that it's
 *     already ready for bindless texturing.  The way bindless texturing works
 *     on our hardware is that you have a big pool of surface state objects
 *     (with its own state base address) and the bindless handles are simply
 *     offsets into that pool.  With the architecture we chose, we already
 *     have that pool and it's exactly the same pool that we use for regular
 *     surface states so we should already be ready for bindless.
 *
 *  3) For render targets, we need to be able to fill out the surface states
 *     later in vkBeginRenderPass so that we can assign clear colors
 *     correctly.  One way to do this would be to just create the surface
 *     state data and then repeatedly copy it into the surface state BO every
 *     time we have to re-emit STATE_BASE_ADDRESS.  While this works, it's
 *     rather annoying and just being able to allocate them up-front and
 *     re-use them for the entire render pass.
 *
 * While none of these are technically blockers for emitting state on the fly
 * like we do in GL, the ability to have a single surface state pool is
 * simplifies things greatly.  Unfortunately, it comes at a cost...
 *
 * Because of the 64k limitation of 3DSTATE_BINDING_TABLE_POINTERS_*, we can't
 * place the binding tables just anywhere in surface state base address.
 * Because 64k isn't a whole lot of space, we can't simply restrict the
 * surface state buffer to 64k, we have to be more clever.  The solution we've
 * chosen is to have a block pool with a maximum size of 2G that starts at
 * zero and grows in both directions.  All surface states are allocated from
 * the top of the pool (positive offsets) and we allocate blocks (< 64k) of
 * binding tables from the bottom of the pool (negative offsets).  Every time
 * we allocate a new binding table block, we set surface state base address to
 * point to the bottom of the binding table block.  This way all of the
 * binding tables in the block are in the bottom 64k of surface state base
 * address.  When we fill out the binding table, we add the distance between
 * the bottom of our binding table block and zero of the block pool to the
 * surface state offsets so that they are correct relative to out new surface
 * state base address at the bottom of the binding table block.
 *
 * \see adjust_relocations_from_block_pool()
 * \see adjust_relocations_too_block_pool()
 *
 * \param[in]  entries        The number of surface state entries the binding
 *                            table should be able to hold.
 *
 * \param[out] state_offset   The offset surface surface state base address
 *                            where the surface states live.  This must be
 *                            added to the surface state offset when it is
 *                            written into the binding table entry.
 *
 * \return                    An anv_state representing the binding table
 */
struct anv_state
anv_cmd_buffer_alloc_binding_table(struct anv_cmd_buffer *cmd_buffer,
                                   uint32_t entries, uint32_t *state_offset)
{
   struct anv_device *device = cmd_buffer->device;
   struct anv_state_pool *state_pool = &device->surface_state_pool;
   struct anv_state *bt_block = u_vector_head(&cmd_buffer->bt_block_states);
   struct anv_state state;

   state.alloc_size = align_u32(entries * 4, 32);

   if (cmd_buffer->bt_next + state.alloc_size > state_pool->block_size)
      return (struct anv_state) { 0 };

   state.offset = cmd_buffer->bt_next;
   state.map = anv_binding_table_pool(device)->block_pool.map +
      bt_block->offset + state.offset;

   cmd_buffer->bt_next += state.alloc_size;

   if (device->instance->physicalDevice.use_softpin) {
      assert(bt_block->offset >= 0);
      *state_offset = device->surface_state_pool.block_pool.start_address -
         device->binding_table_pool.block_pool.start_address - bt_block->offset;
   } else {
      assert(bt_block->offset < 0);
      *state_offset = -bt_block->offset;
   }

   return state;
}

struct anv_state
anv_cmd_buffer_alloc_surface_state(struct anv_cmd_buffer *cmd_buffer)
{
   struct isl_device *isl_dev = &cmd_buffer->device->isl_dev;
   return anv_state_stream_alloc(&cmd_buffer->surface_state_stream,
                                 isl_dev->ss.size, isl_dev->ss.align);
}

struct anv_state
anv_cmd_buffer_alloc_dynamic_state(struct anv_cmd_buffer *cmd_buffer,
                                   uint32_t size, uint32_t alignment)
{
   return anv_state_stream_alloc(&cmd_buffer->dynamic_state_stream,
                                 size, alignment);
}

VkResult
anv_cmd_buffer_new_binding_table_block(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_state *bt_block = u_vector_add(&cmd_buffer->bt_block_states);
   if (bt_block == NULL) {
      anv_batch_set_error(&cmd_buffer->batch, VK_ERROR_OUT_OF_HOST_MEMORY);
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   *bt_block = anv_binding_table_pool_alloc(cmd_buffer->device);
   cmd_buffer->bt_next = 0;

   return VK_SUCCESS;
}

VkResult
anv_cmd_buffer_init_batch_bo_chain(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_batch_bo *batch_bo;
   VkResult result;

   list_inithead(&cmd_buffer->batch_bos);

   result = anv_batch_bo_create(cmd_buffer, &batch_bo);
   if (result != VK_SUCCESS)
      return result;

   list_addtail(&batch_bo->link, &cmd_buffer->batch_bos);

   cmd_buffer->batch.alloc = &cmd_buffer->pool->alloc;
   cmd_buffer->batch.user_data = cmd_buffer;

   if (cmd_buffer->device->can_chain_batches) {
      cmd_buffer->batch.extend_cb = anv_cmd_buffer_chain_batch;
   } else {
      cmd_buffer->batch.extend_cb = anv_cmd_buffer_grow_batch;
   }

   anv_batch_bo_start(batch_bo, &cmd_buffer->batch,
                      GEN8_MI_BATCH_BUFFER_START_length * 4);

   int success = u_vector_init(&cmd_buffer->seen_bbos,
                                 sizeof(struct anv_bo *),
                                 8 * sizeof(struct anv_bo *));
   if (!success)
      goto fail_batch_bo;

   *(struct anv_batch_bo **)u_vector_add(&cmd_buffer->seen_bbos) = batch_bo;

   /* u_vector requires power-of-two size elements */
   unsigned pow2_state_size = util_next_power_of_two(sizeof(struct anv_state));
   success = u_vector_init(&cmd_buffer->bt_block_states,
                           pow2_state_size, 8 * pow2_state_size);
   if (!success)
      goto fail_seen_bbos;

   result = anv_reloc_list_init(&cmd_buffer->surface_relocs,
                                &cmd_buffer->pool->alloc);
   if (result != VK_SUCCESS)
      goto fail_bt_blocks;
   cmd_buffer->last_ss_pool_center = 0;

   result = anv_cmd_buffer_new_binding_table_block(cmd_buffer);
   if (result != VK_SUCCESS)
      goto fail_bt_blocks;

   return VK_SUCCESS;

 fail_bt_blocks:
   u_vector_finish(&cmd_buffer->bt_block_states);
 fail_seen_bbos:
   u_vector_finish(&cmd_buffer->seen_bbos);
 fail_batch_bo:
   anv_batch_bo_destroy(batch_bo, cmd_buffer);

   return result;
}

void
anv_cmd_buffer_fini_batch_bo_chain(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_state *bt_block;
   u_vector_foreach(bt_block, &cmd_buffer->bt_block_states)
      anv_binding_table_pool_free(cmd_buffer->device, *bt_block);
   u_vector_finish(&cmd_buffer->bt_block_states);

   anv_reloc_list_finish(&cmd_buffer->surface_relocs, &cmd_buffer->pool->alloc);

   u_vector_finish(&cmd_buffer->seen_bbos);

   /* Destroy all of the batch buffers */
   list_for_each_entry_safe(struct anv_batch_bo, bbo,
                            &cmd_buffer->batch_bos, link) {
      anv_batch_bo_destroy(bbo, cmd_buffer);
   }
}

void
anv_cmd_buffer_reset_batch_bo_chain(struct anv_cmd_buffer *cmd_buffer)
{
   /* Delete all but the first batch bo */
   assert(!list_empty(&cmd_buffer->batch_bos));
   while (cmd_buffer->batch_bos.next != cmd_buffer->batch_bos.prev) {
      struct anv_batch_bo *bbo = anv_cmd_buffer_current_batch_bo(cmd_buffer);
      list_del(&bbo->link);
      anv_batch_bo_destroy(bbo, cmd_buffer);
   }
   assert(!list_empty(&cmd_buffer->batch_bos));

   anv_batch_bo_start(anv_cmd_buffer_current_batch_bo(cmd_buffer),
                      &cmd_buffer->batch,
                      GEN8_MI_BATCH_BUFFER_START_length * 4);

   while (u_vector_length(&cmd_buffer->bt_block_states) > 1) {
      struct anv_state *bt_block = u_vector_remove(&cmd_buffer->bt_block_states);
      anv_binding_table_pool_free(cmd_buffer->device, *bt_block);
   }
   assert(u_vector_length(&cmd_buffer->bt_block_states) == 1);
   cmd_buffer->bt_next = 0;

   cmd_buffer->surface_relocs.num_relocs = 0;
   _mesa_set_clear(cmd_buffer->surface_relocs.deps, NULL);
   cmd_buffer->last_ss_pool_center = 0;

   /* Reset the list of seen buffers */
   cmd_buffer->seen_bbos.head = 0;
   cmd_buffer->seen_bbos.tail = 0;

   *(struct anv_batch_bo **)u_vector_add(&cmd_buffer->seen_bbos) =
      anv_cmd_buffer_current_batch_bo(cmd_buffer);
}

void
anv_cmd_buffer_end_batch_buffer(struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_batch_bo *batch_bo = anv_cmd_buffer_current_batch_bo(cmd_buffer);

   if (cmd_buffer->level == VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
      /* When we start a batch buffer, we subtract a certain amount of
       * padding from the end to ensure that we always have room to emit a
       * BATCH_BUFFER_START to chain to the next BO.  We need to remove
       * that padding before we end the batch; otherwise, we may end up
       * with our BATCH_BUFFER_END in another BO.
       */
      cmd_buffer->batch.end += GEN8_MI_BATCH_BUFFER_START_length * 4;
      assert(cmd_buffer->batch.end == batch_bo->bo.map + batch_bo->bo.size);

      anv_batch_emit(&cmd_buffer->batch, GEN8_MI_BATCH_BUFFER_END, bbe);

      /* Round batch up to an even number of dwords. */
      if ((cmd_buffer->batch.next - cmd_buffer->batch.start) & 4)
         anv_batch_emit(&cmd_buffer->batch, GEN8_MI_NOOP, noop);

      cmd_buffer->exec_mode = ANV_CMD_BUFFER_EXEC_MODE_PRIMARY;
   } else {
      assert(cmd_buffer->level == VK_COMMAND_BUFFER_LEVEL_SECONDARY);
      /* If this is a secondary command buffer, we need to determine the
       * mode in which it will be executed with vkExecuteCommands.  We
       * determine this statically here so that this stays in sync with the
       * actual ExecuteCommands implementation.
       */
      const uint32_t length = cmd_buffer->batch.next - cmd_buffer->batch.start;
      if (!cmd_buffer->device->can_chain_batches) {
         cmd_buffer->exec_mode = ANV_CMD_BUFFER_EXEC_MODE_GROW_AND_EMIT;
      } else if ((cmd_buffer->batch_bos.next == cmd_buffer->batch_bos.prev) &&
                 (length < ANV_CMD_BUFFER_BATCH_SIZE / 2)) {
         /* If the secondary has exactly one batch buffer in its list *and*
          * that batch buffer is less than half of the maximum size, we're
          * probably better of simply copying it into our batch.
          */
         cmd_buffer->exec_mode = ANV_CMD_BUFFER_EXEC_MODE_EMIT;
      } else if (!(cmd_buffer->usage_flags &
                   VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)) {
         cmd_buffer->exec_mode = ANV_CMD_BUFFER_EXEC_MODE_CHAIN;

         /* In order to chain, we need this command buffer to contain an
          * MI_BATCH_BUFFER_START which will jump back to the calling batch.
          * It doesn't matter where it points now so long as has a valid
          * relocation.  We'll adjust it later as part of the chaining
          * process.
          *
          * We set the end of the batch a little short so we would be sure we
          * have room for the chaining command.  Since we're about to emit the
          * chaining command, let's set it back where it should go.
          */
         cmd_buffer->batch.end += GEN8_MI_BATCH_BUFFER_START_length * 4;
         assert(cmd_buffer->batch.start == batch_bo->bo.map);
         assert(cmd_buffer->batch.end == batch_bo->bo.map + batch_bo->bo.size);

         emit_batch_buffer_start(cmd_buffer, &batch_bo->bo, 0);
         assert(cmd_buffer->batch.start == batch_bo->bo.map);
      } else {
         cmd_buffer->exec_mode = ANV_CMD_BUFFER_EXEC_MODE_COPY_AND_CHAIN;
      }
   }

   anv_batch_bo_finish(batch_bo, &cmd_buffer->batch);
}

static VkResult
anv_cmd_buffer_add_seen_bbos(struct anv_cmd_buffer *cmd_buffer,
                             struct list_head *list)
{
   list_for_each_entry(struct anv_batch_bo, bbo, list, link) {
      struct anv_batch_bo **bbo_ptr = u_vector_add(&cmd_buffer->seen_bbos);
      if (bbo_ptr == NULL)
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

      *bbo_ptr = bbo;
   }

   return VK_SUCCESS;
}

void
anv_cmd_buffer_add_secondary(struct anv_cmd_buffer *primary,
                             struct anv_cmd_buffer *secondary)
{
   switch (secondary->exec_mode) {
   case ANV_CMD_BUFFER_EXEC_MODE_EMIT:
      anv_batch_emit_batch(&primary->batch, &secondary->batch);
      break;
   case ANV_CMD_BUFFER_EXEC_MODE_GROW_AND_EMIT: {
      struct anv_batch_bo *bbo = anv_cmd_buffer_current_batch_bo(primary);
      unsigned length = secondary->batch.end - secondary->batch.start;
      anv_batch_bo_grow(primary, bbo, &primary->batch, length,
                        GEN8_MI_BATCH_BUFFER_START_length * 4);
      anv_batch_emit_batch(&primary->batch, &secondary->batch);
      break;
   }
   case ANV_CMD_BUFFER_EXEC_MODE_CHAIN: {
      struct anv_batch_bo *first_bbo =
         list_first_entry(&secondary->batch_bos, struct anv_batch_bo, link);
      struct anv_batch_bo *last_bbo =
         list_last_entry(&secondary->batch_bos, struct anv_batch_bo, link);

      emit_batch_buffer_start(primary, &first_bbo->bo, 0);

      struct anv_batch_bo *this_bbo = anv_cmd_buffer_current_batch_bo(primary);
      assert(primary->batch.start == this_bbo->bo.map);
      uint32_t offset = primary->batch.next - primary->batch.start;

      /* Make the tail of the secondary point back to right after the
       * MI_BATCH_BUFFER_START in the primary batch.
       */
      anv_batch_bo_link(primary, last_bbo, this_bbo, offset);

      anv_cmd_buffer_add_seen_bbos(primary, &secondary->batch_bos);
      break;
   }
   case ANV_CMD_BUFFER_EXEC_MODE_COPY_AND_CHAIN: {
      struct list_head copy_list;
      VkResult result = anv_batch_bo_list_clone(&secondary->batch_bos,
                                                secondary,
                                                &copy_list);
      if (result != VK_SUCCESS)
         return; /* FIXME */

      anv_cmd_buffer_add_seen_bbos(primary, &copy_list);

      struct anv_batch_bo *first_bbo =
         list_first_entry(&copy_list, struct anv_batch_bo, link);
      struct anv_batch_bo *last_bbo =
         list_last_entry(&copy_list, struct anv_batch_bo, link);

      cmd_buffer_chain_to_batch_bo(primary, first_bbo);

      list_splicetail(&copy_list, &primary->batch_bos);

      anv_batch_bo_continue(last_bbo, &primary->batch,
                            GEN8_MI_BATCH_BUFFER_START_length * 4);
      break;
   }
   default:
      assert(!"Invalid execution mode");
   }

   anv_reloc_list_append(&primary->surface_relocs, &primary->pool->alloc,
                         &secondary->surface_relocs, 0);
}

struct anv_execbuf {
   struct drm_i915_gem_execbuffer2           execbuf;

   struct drm_i915_gem_exec_object2 *        objects;
   uint32_t                                  bo_count;
   struct anv_bo **                          bos;

   /* Allocated length of the 'objects' and 'bos' arrays */
   uint32_t                                  array_length;

   bool                                      has_relocs;

   uint32_t                                  fence_count;
   uint32_t                                  fence_array_length;
   struct drm_i915_gem_exec_fence *          fences;
   struct anv_syncobj **                     syncobjs;
};

static void
anv_execbuf_init(struct anv_execbuf *exec)
{
   memset(exec, 0, sizeof(*exec));
}

static void
anv_execbuf_finish(struct anv_execbuf *exec,
                   const VkAllocationCallbacks *alloc)
{
   vk_free(alloc, exec->objects);
   vk_free(alloc, exec->bos);
   vk_free(alloc, exec->fences);
   vk_free(alloc, exec->syncobjs);
}

static int
_compare_bo_handles(const void *_bo1, const void *_bo2)
{
   struct anv_bo * const *bo1 = _bo1;
   struct anv_bo * const *bo2 = _bo2;

   return (*bo1)->gem_handle - (*bo2)->gem_handle;
}

static VkResult
anv_execbuf_add_bo(struct anv_execbuf *exec,
                   struct anv_bo *bo,
                   struct anv_reloc_list *relocs,
                   uint32_t extra_flags,
                   const VkAllocationCallbacks *alloc)
{
   struct drm_i915_gem_exec_object2 *obj = NULL;

   if (bo->index < exec->bo_count && exec->bos[bo->index] == bo)
      obj = &exec->objects[bo->index];

   if (obj == NULL) {
      /* We've never seen this one before.  Add it to the list and assign
       * an id that we can use later.
       */
      if (exec->bo_count >= exec->array_length) {
         uint32_t new_len = exec->objects ? exec->array_length * 2 : 64;

         struct drm_i915_gem_exec_object2 *new_objects =
            vk_alloc(alloc, new_len * sizeof(*new_objects),
                     8, VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
         if (new_objects == NULL)
            return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

         struct anv_bo **new_bos =
            vk_alloc(alloc, new_len * sizeof(*new_bos),
                      8, VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
         if (new_bos == NULL) {
            vk_free(alloc, new_objects);
            return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);
         }

         if (exec->objects) {
            memcpy(new_objects, exec->objects,
                   exec->bo_count * sizeof(*new_objects));
            memcpy(new_bos, exec->bos,
                   exec->bo_count * sizeof(*new_bos));
         }

         vk_free(alloc, exec->objects);
         vk_free(alloc, exec->bos);

         exec->objects = new_objects;
         exec->bos = new_bos;
         exec->array_length = new_len;
      }

      assert(exec->bo_count < exec->array_length);

      bo->index = exec->bo_count++;
      obj = &exec->objects[bo->index];
      exec->bos[bo->index] = bo;

      obj->handle = bo->gem_handle;
      obj->relocation_count = 0;
      obj->relocs_ptr = 0;
      obj->alignment = 0;
      obj->offset = bo->offset;
      obj->flags = (bo->flags & ~ANV_BO_FLAG_MASK) | extra_flags;
      obj->rsvd1 = 0;
      obj->rsvd2 = 0;
   }

   if (relocs != NULL) {
      assert(obj->relocation_count == 0);

      if (relocs->num_relocs > 0) {
         /* This is the first time we've ever seen a list of relocations for
          * this BO.  Go ahead and set the relocations and then walk the list
          * of relocations and add them all.
          */
         exec->has_relocs = true;
         obj->relocation_count = relocs->num_relocs;
         obj->relocs_ptr = (uintptr_t) relocs->relocs;

         for (size_t i = 0; i < relocs->num_relocs; i++) {
            VkResult result;

            /* A quick sanity check on relocations */
            assert(relocs->relocs[i].offset < bo->size);
            result = anv_execbuf_add_bo(exec, relocs->reloc_bos[i], NULL,
                                        extra_flags, alloc);

            if (result != VK_SUCCESS)
               return result;
         }
      }

      if (relocs->deps && relocs->deps->entries > 0) {
         const uint32_t entries = relocs->deps->entries;
         struct anv_bo **bos =
            vk_alloc(alloc, entries * sizeof(*bos),
                     8, VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
         if (bos == NULL)
            return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

         struct anv_bo **bo = bos;
         set_foreach(relocs->deps, entry) {
            *bo++ = (void *)entry->key;
         }

         qsort(bos, entries, sizeof(struct anv_bo*), _compare_bo_handles);

         VkResult result = VK_SUCCESS;
         for (bo = bos; bo < bos + entries; bo++) {
            result = anv_execbuf_add_bo(exec, *bo, NULL, extra_flags, alloc);
            if (result != VK_SUCCESS)
               break;
         }

         vk_free(alloc, bos);

         if (result != VK_SUCCESS)
            return result;
      }
   }

   return VK_SUCCESS;
}

static VkResult
anv_execbuf_add_syncobj(struct anv_execbuf *exec,
                        uint32_t handle, uint32_t flags,
                        const VkAllocationCallbacks *alloc)
{
   assert(flags != 0);

   if (exec->fence_count >= exec->fence_array_length) {
      uint32_t new_len = MAX2(exec->fence_array_length * 2, 64);

      exec->fences = vk_realloc(alloc, exec->fences,
                                new_len * sizeof(*exec->fences),
                                8, VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
      if (exec->fences == NULL)
         return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

      exec->fence_array_length = new_len;
   }

   exec->fences[exec->fence_count] = (struct drm_i915_gem_exec_fence) {
      .handle = handle,
      .flags = flags,
   };

   exec->fence_count++;

   return VK_SUCCESS;
}

static void
anv_cmd_buffer_process_relocs(struct anv_cmd_buffer *cmd_buffer,
                              struct anv_reloc_list *list)
{
   for (size_t i = 0; i < list->num_relocs; i++)
      list->relocs[i].target_handle = list->reloc_bos[i]->index;
}

static void
adjust_relocations_from_state_pool(struct anv_state_pool *pool,
                                   struct anv_reloc_list *relocs,
                                   uint32_t last_pool_center_bo_offset)
{
   assert(last_pool_center_bo_offset <= pool->block_pool.center_bo_offset);
   uint32_t delta = pool->block_pool.center_bo_offset - last_pool_center_bo_offset;

   for (size_t i = 0; i < relocs->num_relocs; i++) {
      /* All of the relocations from this block pool to other BO's should
       * have been emitted relative to the surface block pool center.  We
       * need to add the center offset to make them relative to the
       * beginning of the actual GEM bo.
       */
      relocs->relocs[i].offset += delta;
   }
}

static void
adjust_relocations_to_state_pool(struct anv_state_pool *pool,
                                 struct anv_bo *from_bo,
                                 struct anv_reloc_list *relocs,
                                 uint32_t last_pool_center_bo_offset)
{
   assert(last_pool_center_bo_offset <= pool->block_pool.center_bo_offset);
   uint32_t delta = pool->block_pool.center_bo_offset - last_pool_center_bo_offset;

   /* When we initially emit relocations into a block pool, we don't
    * actually know what the final center_bo_offset will be so we just emit
    * it as if center_bo_offset == 0.  Now that we know what the center
    * offset is, we need to walk the list of relocations and adjust any
    * relocations that point to the pool bo with the correct offset.
    */
   for (size_t i = 0; i < relocs->num_relocs; i++) {
      if (relocs->reloc_bos[i] == &pool->block_pool.bo) {
         /* Adjust the delta value in the relocation to correctly
          * correspond to the new delta.  Initially, this value may have
          * been negative (if treated as unsigned), but we trust in
          * uint32_t roll-over to fix that for us at this point.
          */
         relocs->relocs[i].delta += delta;

         /* Since the delta has changed, we need to update the actual
          * relocated value with the new presumed value.  This function
          * should only be called on batch buffers, so we know it isn't in
          * use by the GPU at the moment.
          */
         assert(relocs->relocs[i].offset < from_bo->size);
         write_reloc(pool->block_pool.device,
                     from_bo->map + relocs->relocs[i].offset,
                     relocs->relocs[i].presumed_offset +
                     relocs->relocs[i].delta, false);
      }
   }
}

static void
anv_reloc_list_apply(struct anv_device *device,
                     struct anv_reloc_list *list,
                     struct anv_bo *bo,
                     bool always_relocate)
{
   for (size_t i = 0; i < list->num_relocs; i++) {
      struct anv_bo *target_bo = list->reloc_bos[i];
      if (list->relocs[i].presumed_offset == target_bo->offset &&
          !always_relocate)
         continue;

      void *p = bo->map + list->relocs[i].offset;
      write_reloc(device, p, target_bo->offset + list->relocs[i].delta, true);
      list->relocs[i].presumed_offset = target_bo->offset;
   }
}

/**
 * This function applies the relocation for a command buffer and writes the
 * actual addresses into the buffers as per what we were told by the kernel on
 * the previous execbuf2 call.  This should be safe to do because, for each
 * relocated address, we have two cases:
 *
 *  1) The target BO is inactive (as seen by the kernel).  In this case, it is
 *     not in use by the GPU so updating the address is 100% ok.  It won't be
 *     in-use by the GPU (from our context) again until the next execbuf2
 *     happens.  If the kernel decides to move it in the next execbuf2, it
 *     will have to do the relocations itself, but that's ok because it should
 *     have all of the information needed to do so.
 *
 *  2) The target BO is active (as seen by the kernel).  In this case, it
 *     hasn't moved since the last execbuffer2 call because GTT shuffling
 *     *only* happens when the BO is idle. (From our perspective, it only
 *     happens inside the execbuffer2 ioctl, but the shuffling may be
 *     triggered by another ioctl, with full-ppgtt this is limited to only
 *     execbuffer2 ioctls on the same context, or memory pressure.)  Since the
 *     target BO hasn't moved, our anv_bo::offset exactly matches the BO's GTT
 *     address and the relocated value we are writing into the BO will be the
 *     same as the value that is already there.
 *
 *     There is also a possibility that the target BO is active but the exact
 *     RENDER_SURFACE_STATE object we are writing the relocation into isn't in
 *     use.  In this case, the address currently in the RENDER_SURFACE_STATE
 *     may be stale but it's still safe to write the relocation because that
 *     particular RENDER_SURFACE_STATE object isn't in-use by the GPU and
 *     won't be until the next execbuf2 call.
 *
 * By doing relocations on the CPU, we can tell the kernel that it doesn't
 * need to bother.  We want to do this because the surface state buffer is
 * used by every command buffer so, if the kernel does the relocations, it
 * will always be busy and the kernel will always stall.  This is also
 * probably the fastest mechanism for doing relocations since the kernel would
 * have to make a full copy of all the relocations lists.
 */
static bool
relocate_cmd_buffer(struct anv_cmd_buffer *cmd_buffer,
                    struct anv_execbuf *exec)
{
   if (!exec->has_relocs)
      return true;

   static int userspace_relocs = -1;
   if (userspace_relocs < 0)
      userspace_relocs = env_var_as_boolean("ANV_USERSPACE_RELOCS", true);
   if (!userspace_relocs)
      return false;

   /* First, we have to check to see whether or not we can even do the
    * relocation.  New buffers which have never been submitted to the kernel
    * don't have a valid offset so we need to let the kernel do relocations so
    * that we can get offsets for them.  On future execbuf2 calls, those
    * buffers will have offsets and we will be able to skip relocating.
    * Invalid offsets are indicated by anv_bo::offset == (uint64_t)-1.
    */
   for (uint32_t i = 0; i < exec->bo_count; i++) {
      if (exec->bos[i]->offset == (uint64_t)-1)
         return false;
   }

   /* Since surface states are shared between command buffers and we don't
    * know what order they will be submitted to the kernel, we don't know
    * what address is actually written in the surface state object at any
    * given time.  The only option is to always relocate them.
    */
   anv_reloc_list_apply(cmd_buffer->device, &cmd_buffer->surface_relocs,
                        &cmd_buffer->device->surface_state_pool.block_pool.bo,
                        true /* always relocate surface states */);

   /* Since we own all of the batch buffers, we know what values are stored
    * in the relocated addresses and only have to update them if the offsets
    * have changed.
    */
   struct anv_batch_bo **bbo;
   u_vector_foreach(bbo, &cmd_buffer->seen_bbos) {
      anv_reloc_list_apply(cmd_buffer->device,
                           &(*bbo)->relocs, &(*bbo)->bo, false);
   }

   for (uint32_t i = 0; i < exec->bo_count; i++)
      exec->objects[i].offset = exec->bos[i]->offset;

   return true;
}

static VkResult
setup_execbuf_for_cmd_buffer(struct anv_execbuf *execbuf,
                             struct anv_cmd_buffer *cmd_buffer)
{
   struct anv_batch *batch = &cmd_buffer->batch;
   struct anv_state_pool *ss_pool =
      &cmd_buffer->device->surface_state_pool;

   adjust_relocations_from_state_pool(ss_pool, &cmd_buffer->surface_relocs,
                                      cmd_buffer->last_ss_pool_center);
   VkResult result = anv_execbuf_add_bo(execbuf, &ss_pool->block_pool.bo,
                                        &cmd_buffer->surface_relocs, 0,
                                        &cmd_buffer->device->alloc);
   if (result != VK_SUCCESS)
      return result;

   /* First, we walk over all of the bos we've seen and add them and their
    * relocations to the validate list.
    */
   struct anv_batch_bo **bbo;
   u_vector_foreach(bbo, &cmd_buffer->seen_bbos) {
      adjust_relocations_to_state_pool(ss_pool, &(*bbo)->bo, &(*bbo)->relocs,
                                       cmd_buffer->last_ss_pool_center);

      result = anv_execbuf_add_bo(execbuf, &(*bbo)->bo, &(*bbo)->relocs, 0,
                                  &cmd_buffer->device->alloc);
      if (result != VK_SUCCESS)
         return result;
   }

   /* Now that we've adjusted all of the surface state relocations, we need to
    * record the surface state pool center so future executions of the command
    * buffer can adjust correctly.
    */
   cmd_buffer->last_ss_pool_center = ss_pool->block_pool.center_bo_offset;

   struct anv_batch_bo *first_batch_bo =
      list_first_entry(&cmd_buffer->batch_bos, struct anv_batch_bo, link);

   /* The kernel requires that the last entry in the validation list be the
    * batch buffer to execute.  We can simply swap the element
    * corresponding to the first batch_bo in the chain with the last
    * element in the list.
    */
   if (first_batch_bo->bo.index != execbuf->bo_count - 1) {
      uint32_t idx = first_batch_bo->bo.index;
      uint32_t last_idx = execbuf->bo_count - 1;

      struct drm_i915_gem_exec_object2 tmp_obj = execbuf->objects[idx];
      assert(execbuf->bos[idx] == &first_batch_bo->bo);

      execbuf->objects[idx] = execbuf->objects[last_idx];
      execbuf->bos[idx] = execbuf->bos[last_idx];
      execbuf->bos[idx]->index = idx;

      execbuf->objects[last_idx] = tmp_obj;
      execbuf->bos[last_idx] = &first_batch_bo->bo;
      first_batch_bo->bo.index = last_idx;
   }

   /* If we are pinning our BOs, we shouldn't have to relocate anything */
   if (cmd_buffer->device->instance->physicalDevice.use_softpin)
      assert(!execbuf->has_relocs);

   /* Now we go through and fixup all of the relocation lists to point to
    * the correct indices in the object array.  We have to do this after we
    * reorder the list above as some of the indices may have changed.
    */
   if (execbuf->has_relocs) {
      u_vector_foreach(bbo, &cmd_buffer->seen_bbos)
         anv_cmd_buffer_process_relocs(cmd_buffer, &(*bbo)->relocs);

      anv_cmd_buffer_process_relocs(cmd_buffer, &cmd_buffer->surface_relocs);
   }

   if (!cmd_buffer->device->info.has_llc) {
      __builtin_ia32_mfence();
      u_vector_foreach(bbo, &cmd_buffer->seen_bbos) {
         for (uint32_t i = 0; i < (*bbo)->length; i += CACHELINE_SIZE)
            __builtin_ia32_clflush((*bbo)->bo.map + i);
      }
   }

   execbuf->execbuf = (struct drm_i915_gem_execbuffer2) {
      .buffers_ptr = (uintptr_t) execbuf->objects,
      .buffer_count = execbuf->bo_count,
      .batch_start_offset = 0,
      .batch_len = batch->next - batch->start,
      .cliprects_ptr = 0,
      .num_cliprects = 0,
      .DR1 = 0,
      .DR4 = 0,
      .flags = I915_EXEC_HANDLE_LUT | I915_EXEC_RENDER,
      .rsvd1 = cmd_buffer->device->context_id,
      .rsvd2 = 0,
   };

   if (relocate_cmd_buffer(cmd_buffer, execbuf)) {
      /* If we were able to successfully relocate everything, tell the kernel
       * that it can skip doing relocations. The requirement for using
       * NO_RELOC is:
       *
       *  1) The addresses written in the objects must match the corresponding
       *     reloc.presumed_offset which in turn must match the corresponding
       *     execobject.offset.
       *
       *  2) To avoid stalling, execobject.offset should match the current
       *     address of that object within the active context.
       *
       * In order to satisfy all of the invariants that make userspace
       * relocations to be safe (see relocate_cmd_buffer()), we need to
       * further ensure that the addresses we use match those used by the
       * kernel for the most recent execbuf2.
       *
       * The kernel may still choose to do relocations anyway if something has
       * moved in the GTT. In this case, the relocation list still needs to be
       * valid.  All relocations on the batch buffers are already valid and
       * kept up-to-date.  For surface state relocations, by applying the
       * relocations in relocate_cmd_buffer, we ensured that the address in
       * the RENDER_SURFACE_STATE matches presumed_offset, so it should be
       * safe for the kernel to relocate them as needed.
       */
      execbuf->execbuf.flags |= I915_EXEC_NO_RELOC;
   } else {
      /* In the case where we fall back to doing kernel relocations, we need
       * to ensure that the relocation list is valid.  All relocations on the
       * batch buffers are already valid and kept up-to-date.  Since surface
       * states are shared between command buffers and we don't know what
       * order they will be submitted to the kernel, we don't know what
       * address is actually written in the surface state object at any given
       * time.  The only option is to set a bogus presumed offset and let the
       * kernel relocate them.
       */
      for (size_t i = 0; i < cmd_buffer->surface_relocs.num_relocs; i++)
         cmd_buffer->surface_relocs.relocs[i].presumed_offset = -1;
   }

   return VK_SUCCESS;
}

static VkResult
setup_empty_execbuf(struct anv_execbuf *execbuf, struct anv_device *device)
{
   VkResult result = anv_execbuf_add_bo(execbuf, &device->trivial_batch_bo,
                                        NULL, 0, &device->alloc);
   if (result != VK_SUCCESS)
      return result;

   execbuf->execbuf = (struct drm_i915_gem_execbuffer2) {
      .buffers_ptr = (uintptr_t) execbuf->objects,
      .buffer_count = execbuf->bo_count,
      .batch_start_offset = 0,
      .batch_len = 8, /* GEN7_MI_BATCH_BUFFER_END and NOOP */
      .flags = I915_EXEC_HANDLE_LUT | I915_EXEC_RENDER,
      .rsvd1 = device->context_id,
      .rsvd2 = 0,
   };

   return VK_SUCCESS;
}

VkResult
anv_cmd_buffer_execbuf(struct anv_device *device,
                       struct anv_cmd_buffer *cmd_buffer,
                       const VkSemaphore *in_semaphores,
                       uint32_t num_in_semaphores,
                       const VkSemaphore *out_semaphores,
                       uint32_t num_out_semaphores,
                       VkFence _fence)
{
   ANV_FROM_HANDLE(anv_fence, fence, _fence);

   struct anv_execbuf execbuf;
   anv_execbuf_init(&execbuf);

   int in_fence = -1;
   VkResult result = VK_SUCCESS;
   for (uint32_t i = 0; i < num_in_semaphores; i++) {
      ANV_FROM_HANDLE(anv_semaphore, semaphore, in_semaphores[i]);
      struct anv_semaphore_impl *impl =
         semaphore->temporary.type != ANV_SEMAPHORE_TYPE_NONE ?
         &semaphore->temporary : &semaphore->permanent;

      switch (impl->type) {
      case ANV_SEMAPHORE_TYPE_BO:
         result = anv_execbuf_add_bo(&execbuf, impl->bo, NULL,
                                     0, &device->alloc);
         if (result != VK_SUCCESS)
            return result;
         break;

      case ANV_SEMAPHORE_TYPE_SYNC_FILE:
         if (in_fence == -1) {
            in_fence = impl->fd;
         } else {
            int merge = anv_gem_sync_file_merge(device, in_fence, impl->fd);
            if (merge == -1)
               return vk_error(VK_ERROR_INVALID_EXTERNAL_HANDLE);

            close(impl->fd);
            close(in_fence);
            in_fence = merge;
         }

         impl->fd = -1;
         break;

      case ANV_SEMAPHORE_TYPE_DRM_SYNCOBJ:
         result = anv_execbuf_add_syncobj(&execbuf, impl->syncobj,
                                          I915_EXEC_FENCE_WAIT,
                                          &device->alloc);
         if (result != VK_SUCCESS)
            return result;
         break;

      default:
         break;
      }
   }

   bool need_out_fence = false;
   for (uint32_t i = 0; i < num_out_semaphores; i++) {
      ANV_FROM_HANDLE(anv_semaphore, semaphore, out_semaphores[i]);

      /* Under most circumstances, out fences won't be temporary.  However,
       * the spec does allow it for opaque_fd.  From the Vulkan 1.0.53 spec:
       *
       *    "If the import is temporary, the implementation must restore the
       *    semaphore to its prior permanent state after submitting the next
       *    semaphore wait operation."
       *
       * The spec says nothing whatsoever about signal operations on
       * temporarily imported semaphores so it appears they are allowed.
       * There are also CTS tests that require this to work.
       */
      struct anv_semaphore_impl *impl =
         semaphore->temporary.type != ANV_SEMAPHORE_TYPE_NONE ?
         &semaphore->temporary : &semaphore->permanent;

      switch (impl->type) {
      case ANV_SEMAPHORE_TYPE_BO:
         result = anv_execbuf_add_bo(&execbuf, impl->bo, NULL,
                                     EXEC_OBJECT_WRITE, &device->alloc);
         if (result != VK_SUCCESS)
            return result;
         break;

      case ANV_SEMAPHORE_TYPE_SYNC_FILE:
         need_out_fence = true;
         break;

      case ANV_SEMAPHORE_TYPE_DRM_SYNCOBJ:
         result = anv_execbuf_add_syncobj(&execbuf, impl->syncobj,
                                          I915_EXEC_FENCE_SIGNAL,
                                          &device->alloc);
         if (result != VK_SUCCESS)
            return result;
         break;

      default:
         break;
      }
   }

   if (fence) {
      /* Under most circumstances, out fences won't be temporary.  However,
       * the spec does allow it for opaque_fd.  From the Vulkan 1.0.53 spec:
       *
       *    "If the import is temporary, the implementation must restore the
       *    semaphore to its prior permanent state after submitting the next
       *    semaphore wait operation."
       *
       * The spec says nothing whatsoever about signal operations on
       * temporarily imported semaphores so it appears they are allowed.
       * There are also CTS tests that require this to work.
       */
      struct anv_fence_impl *impl =
         fence->temporary.type != ANV_FENCE_TYPE_NONE ?
         &fence->temporary : &fence->permanent;

      switch (impl->type) {
      case ANV_FENCE_TYPE_BO:
         result = anv_execbuf_add_bo(&execbuf, &impl->bo.bo, NULL,
                                     EXEC_OBJECT_WRITE, &device->alloc);
         if (result != VK_SUCCESS)
            return result;
         break;

      case ANV_FENCE_TYPE_SYNCOBJ:
         result = anv_execbuf_add_syncobj(&execbuf, impl->syncobj,
                                          I915_EXEC_FENCE_SIGNAL,
                                          &device->alloc);
         if (result != VK_SUCCESS)
            return result;
         break;

      default:
         unreachable("Invalid fence type");
      }
   }

   if (cmd_buffer)
      result = setup_execbuf_for_cmd_buffer(&execbuf, cmd_buffer);
   else
      result = setup_empty_execbuf(&execbuf, device);

   if (result != VK_SUCCESS)
      return result;

   if (execbuf.fence_count > 0) {
      assert(device->instance->physicalDevice.has_syncobj);
      execbuf.execbuf.flags |= I915_EXEC_FENCE_ARRAY;
      execbuf.execbuf.num_cliprects = execbuf.fence_count;
      execbuf.execbuf.cliprects_ptr = (uintptr_t) execbuf.fences;
   }

   if (in_fence != -1) {
      execbuf.execbuf.flags |= I915_EXEC_FENCE_IN;
      execbuf.execbuf.rsvd2 |= (uint32_t)in_fence;
   }

   if (need_out_fence)
      execbuf.execbuf.flags |= I915_EXEC_FENCE_OUT;

   result = anv_device_execbuf(device, &execbuf.execbuf, execbuf.bos);

   /* Execbuf does not consume the in_fence.  It's our job to close it. */
   if (in_fence != -1)
      close(in_fence);

   for (uint32_t i = 0; i < num_in_semaphores; i++) {
      ANV_FROM_HANDLE(anv_semaphore, semaphore, in_semaphores[i]);
      /* From the Vulkan 1.0.53 spec:
       *
       *    "If the import is temporary, the implementation must restore the
       *    semaphore to its prior permanent state after submitting the next
       *    semaphore wait operation."
       *
       * This has to happen after the execbuf in case we close any syncobjs in
       * the process.
       */
      anv_semaphore_reset_temporary(device, semaphore);
   }

   if (fence && fence->permanent.type == ANV_FENCE_TYPE_BO) {
      /* BO fences can't be shared, so they can't be temporary. */
      assert(fence->temporary.type == ANV_FENCE_TYPE_NONE);

      /* Once the execbuf has returned, we need to set the fence state to
       * SUBMITTED.  We can't do this before calling execbuf because
       * anv_GetFenceStatus does take the global device lock before checking
       * fence->state.
       *
       * We set the fence state to SUBMITTED regardless of whether or not the
       * execbuf succeeds because we need to ensure that vkWaitForFences() and
       * vkGetFenceStatus() return a valid result (VK_ERROR_DEVICE_LOST or
       * VK_SUCCESS) in a finite amount of time even if execbuf fails.
       */
      fence->permanent.bo.state = ANV_BO_FENCE_STATE_SUBMITTED;
   }

   if (result == VK_SUCCESS && need_out_fence) {
      int out_fence = execbuf.execbuf.rsvd2 >> 32;
      for (uint32_t i = 0; i < num_out_semaphores; i++) {
         ANV_FROM_HANDLE(anv_semaphore, semaphore, out_semaphores[i]);
         /* Out fences can't have temporary state because that would imply
          * that we imported a sync file and are trying to signal it.
          */
         assert(semaphore->temporary.type == ANV_SEMAPHORE_TYPE_NONE);
         struct anv_semaphore_impl *impl = &semaphore->permanent;

         if (impl->type == ANV_SEMAPHORE_TYPE_SYNC_FILE) {
            assert(impl->fd == -1);
            impl->fd = dup(out_fence);
         }
      }
      close(out_fence);
   }

   anv_execbuf_finish(&execbuf, &device->alloc);

   return result;
}
