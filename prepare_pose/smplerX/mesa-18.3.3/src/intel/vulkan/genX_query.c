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

#include "genxml/gen_macros.h"
#include "genxml/genX_pack.h"

VkResult genX(CreateQueryPool)(
    VkDevice                                    _device,
    const VkQueryPoolCreateInfo*                pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkQueryPool*                                pQueryPool)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   const struct anv_physical_device *pdevice = &device->instance->physicalDevice;
   struct anv_query_pool *pool;
   VkResult result;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);

   /* Query pool slots are made up of some number of 64-bit values packed
    * tightly together.  The first 64-bit value is always the "available" bit
    * which is 0 when the query is unavailable and 1 when it is available.
    * The 64-bit values that follow are determined by the type of query.
    */
   uint32_t uint64s_per_slot = 1;

   VkQueryPipelineStatisticFlags pipeline_statistics = 0;
   switch (pCreateInfo->queryType) {
   case VK_QUERY_TYPE_OCCLUSION:
      /* Occlusion queries have two values: begin and end. */
      uint64s_per_slot += 2;
      break;
   case VK_QUERY_TYPE_TIMESTAMP:
      /* Timestamps just have the one timestamp value */
      uint64s_per_slot += 1;
      break;
   case VK_QUERY_TYPE_PIPELINE_STATISTICS:
      pipeline_statistics = pCreateInfo->pipelineStatistics;
      /* We're going to trust this field implicitly so we need to ensure that
       * no unhandled extension bits leak in.
       */
      pipeline_statistics &= ANV_PIPELINE_STATISTICS_MASK;

      /* Statistics queries have a min and max for every statistic */
      uint64s_per_slot += 2 * util_bitcount(pipeline_statistics);
      break;
   default:
      assert(!"Invalid query type");
   }

   pool = vk_alloc2(&device->alloc, pAllocator, sizeof(*pool), 8,
                     VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pool == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   pool->type = pCreateInfo->queryType;
   pool->pipeline_statistics = pipeline_statistics;
   pool->stride = uint64s_per_slot * sizeof(uint64_t);
   pool->slots = pCreateInfo->queryCount;

   uint64_t size = pool->slots * pool->stride;
   result = anv_bo_init_new(&pool->bo, device, size);
   if (result != VK_SUCCESS)
      goto fail;

   if (pdevice->supports_48bit_addresses)
      pool->bo.flags |= EXEC_OBJECT_SUPPORTS_48B_ADDRESS;

   if (pdevice->use_softpin)
      pool->bo.flags |= EXEC_OBJECT_PINNED;

   if (pdevice->has_exec_async)
      pool->bo.flags |= EXEC_OBJECT_ASYNC;

   anv_vma_alloc(device, &pool->bo);

   /* For query pools, we set the caching mode to I915_CACHING_CACHED.  On LLC
    * platforms, this does nothing.  On non-LLC platforms, this means snooping
    * which comes at a slight cost.  However, the buffers aren't big, won't be
    * written frequently, and trying to handle the flushing manually without
    * doing too much flushing is extremely painful.
    */
   anv_gem_set_caching(device, pool->bo.gem_handle, I915_CACHING_CACHED);

   pool->bo.map = anv_gem_mmap(device, pool->bo.gem_handle, 0, size, 0);

   *pQueryPool = anv_query_pool_to_handle(pool);

   return VK_SUCCESS;

 fail:
   vk_free2(&device->alloc, pAllocator, pool);

   return result;
}

void genX(DestroyQueryPool)(
    VkDevice                                    _device,
    VkQueryPool                                 _pool,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_query_pool, pool, _pool);

   if (!pool)
      return;

   anv_gem_munmap(pool->bo.map, pool->bo.size);
   anv_vma_free(device, &pool->bo);
   anv_gem_close(device, pool->bo.gem_handle);
   vk_free2(&device->alloc, pAllocator, pool);
}

static struct anv_address
anv_query_address(struct anv_query_pool *pool, uint32_t query)
{
   return (struct anv_address) {
      .bo = &pool->bo,
      .offset = query * pool->stride,
   };
}

static void
cpu_write_query_result(void *dst_slot, VkQueryResultFlags flags,
                       uint32_t value_index, uint64_t result)
{
   if (flags & VK_QUERY_RESULT_64_BIT) {
      uint64_t *dst64 = dst_slot;
      dst64[value_index] = result;
   } else {
      uint32_t *dst32 = dst_slot;
      dst32[value_index] = result;
   }
}

static bool
query_is_available(uint64_t *slot)
{
   return *(volatile uint64_t *)slot;
}

static VkResult
wait_for_available(struct anv_device *device,
                   struct anv_query_pool *pool, uint64_t *slot)
{
   while (true) {
      if (query_is_available(slot))
         return VK_SUCCESS;

      int ret = anv_gem_busy(device, pool->bo.gem_handle);
      if (ret == 1) {
         /* The BO is still busy, keep waiting. */
         continue;
      } else if (ret == -1) {
         /* We don't know the real error. */
         return anv_device_set_lost(device, "gem wait failed: %m");
      } else {
         assert(ret == 0);
         /* The BO is no longer busy. */
         if (query_is_available(slot)) {
            return VK_SUCCESS;
         } else {
            VkResult status = anv_device_query_status(device);
            if (status != VK_SUCCESS)
               return status;

            /* If we haven't seen availability yet, then we never will.  This
             * can only happen if we have a client error where they call
             * GetQueryPoolResults on a query that they haven't submitted to
             * the GPU yet.  The spec allows us to do anything in this case,
             * but returning VK_SUCCESS doesn't seem right and we shouldn't
             * just keep spinning.
             */
            return VK_NOT_READY;
         }
      }
   }
}

VkResult genX(GetQueryPoolResults)(
    VkDevice                                    _device,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount,
    size_t                                      dataSize,
    void*                                       pData,
    VkDeviceSize                                stride,
    VkQueryResultFlags                          flags)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);

   assert(pool->type == VK_QUERY_TYPE_OCCLUSION ||
          pool->type == VK_QUERY_TYPE_PIPELINE_STATISTICS ||
          pool->type == VK_QUERY_TYPE_TIMESTAMP);

   if (anv_device_is_lost(device))
      return VK_ERROR_DEVICE_LOST;

   if (pData == NULL)
      return VK_SUCCESS;

   void *data_end = pData + dataSize;

   VkResult status = VK_SUCCESS;
   for (uint32_t i = 0; i < queryCount; i++) {
      uint64_t *slot = pool->bo.map + (firstQuery + i) * pool->stride;

      /* Availability is always at the start of the slot */
      bool available = slot[0];

      if (!available && (flags & VK_QUERY_RESULT_WAIT_BIT)) {
         status = wait_for_available(device, pool, slot);
         if (status != VK_SUCCESS)
            return status;

         available = true;
      }

      /* From the Vulkan 1.0.42 spec:
       *
       *    "If VK_QUERY_RESULT_WAIT_BIT and VK_QUERY_RESULT_PARTIAL_BIT are
       *    both not set then no result values are written to pData for
       *    queries that are in the unavailable state at the time of the call,
       *    and vkGetQueryPoolResults returns VK_NOT_READY. However,
       *    availability state is still written to pData for those queries if
       *    VK_QUERY_RESULT_WITH_AVAILABILITY_BIT is set."
       */
      bool write_results = available || (flags & VK_QUERY_RESULT_PARTIAL_BIT);

      uint32_t idx = 0;
      switch (pool->type) {
      case VK_QUERY_TYPE_OCCLUSION:
         if (write_results)
            cpu_write_query_result(pData, flags, idx, slot[2] - slot[1]);
         idx++;
         break;

      case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
         uint32_t statistics = pool->pipeline_statistics;
         while (statistics) {
            uint32_t stat = u_bit_scan(&statistics);
            if (write_results) {
               uint64_t result = slot[idx * 2 + 2] - slot[idx * 2 + 1];

               /* WaDividePSInvocationCountBy4:HSW,BDW */
               if ((device->info.gen == 8 || device->info.is_haswell) &&
                   (1 << stat) == VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT)
                  result >>= 2;

               cpu_write_query_result(pData, flags, idx, result);
            }
            idx++;
         }
         assert(idx == util_bitcount(pool->pipeline_statistics));
         break;
      }

      case VK_QUERY_TYPE_TIMESTAMP:
         if (write_results)
            cpu_write_query_result(pData, flags, idx, slot[1]);
         idx++;
         break;

      default:
         unreachable("invalid pool type");
      }

      if (!write_results)
         status = VK_NOT_READY;

      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
         cpu_write_query_result(pData, flags, idx, available);

      pData += stride;
      if (pData >= data_end)
         break;
   }

   return status;
}

static void
emit_srm32(struct anv_batch *batch, struct anv_address addr, uint32_t reg)
{
   anv_batch_emit(batch, GENX(MI_STORE_REGISTER_MEM), srm) {
      srm.MemoryAddress    = addr;
      srm.RegisterAddress  = reg;
   }
}

static void
emit_srm64(struct anv_batch *batch, struct anv_address addr, uint32_t reg)
{
   emit_srm32(batch, anv_address_add(addr, 0), reg + 0);
   emit_srm32(batch, anv_address_add(addr, 4), reg + 4);
}

static void
emit_ps_depth_count(struct anv_cmd_buffer *cmd_buffer,
                    struct anv_address addr)
{
   anv_batch_emit(&cmd_buffer->batch, GENX(PIPE_CONTROL), pc) {
      pc.DestinationAddressType  = DAT_PPGTT;
      pc.PostSyncOperation       = WritePSDepthCount;
      pc.DepthStallEnable        = true;
      pc.Address                 = addr;

      if (GEN_GEN == 9 && cmd_buffer->device->info.gt == 4)
         pc.CommandStreamerStallEnable = true;
   }
}

static void
emit_query_availability(struct anv_cmd_buffer *cmd_buffer,
                        struct anv_address addr)
{
   anv_batch_emit(&cmd_buffer->batch, GENX(PIPE_CONTROL), pc) {
      pc.DestinationAddressType  = DAT_PPGTT;
      pc.PostSyncOperation       = WriteImmediateData;
      pc.Address                 = addr;
      pc.ImmediateData           = 1;
   }
}

/**
 * Goes through a series of consecutive query indices in the given pool
 * setting all element values to 0 and emitting them as available.
 */
static void
emit_zero_queries(struct anv_cmd_buffer *cmd_buffer,
                  struct anv_query_pool *pool,
                  uint32_t first_index, uint32_t num_queries)
{
   for (uint32_t i = 0; i < num_queries; i++) {
      struct anv_address slot_addr =
         anv_query_address(pool, first_index + i);
      genX(cmd_buffer_mi_memset)(cmd_buffer, anv_address_add(slot_addr, 8),
                                 0, pool->stride - 8);
      emit_query_availability(cmd_buffer, slot_addr);
   }
}

void genX(CmdResetQueryPool)(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);

   for (uint32_t i = 0; i < queryCount; i++) {
      anv_batch_emit(&cmd_buffer->batch, GENX(MI_STORE_DATA_IMM), sdm) {
         sdm.Address = anv_query_address(pool, firstQuery + i);
         sdm.ImmediateData = 0;
      }
   }
}

static const uint32_t vk_pipeline_stat_to_reg[] = {
   GENX(IA_VERTICES_COUNT_num),
   GENX(IA_PRIMITIVES_COUNT_num),
   GENX(VS_INVOCATION_COUNT_num),
   GENX(GS_INVOCATION_COUNT_num),
   GENX(GS_PRIMITIVES_COUNT_num),
   GENX(CL_INVOCATION_COUNT_num),
   GENX(CL_PRIMITIVES_COUNT_num),
   GENX(PS_INVOCATION_COUNT_num),
   GENX(HS_INVOCATION_COUNT_num),
   GENX(DS_INVOCATION_COUNT_num),
   GENX(CS_INVOCATION_COUNT_num),
};

static void
emit_pipeline_stat(struct anv_cmd_buffer *cmd_buffer, uint32_t stat,
                   struct anv_address addr)
{
   STATIC_ASSERT(ANV_PIPELINE_STATISTICS_MASK ==
                 (1 << ARRAY_SIZE(vk_pipeline_stat_to_reg)) - 1);

   assert(stat < ARRAY_SIZE(vk_pipeline_stat_to_reg));
   emit_srm64(&cmd_buffer->batch, addr, vk_pipeline_stat_to_reg[stat]);
}

void genX(CmdBeginQuery)(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query,
    VkQueryControlFlags                         flags)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);
   struct anv_address query_addr = anv_query_address(pool, query);

   switch (pool->type) {
   case VK_QUERY_TYPE_OCCLUSION:
      emit_ps_depth_count(cmd_buffer, anv_address_add(query_addr, 8));
      break;

   case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
      /* TODO: This might only be necessary for certain stats */
      anv_batch_emit(&cmd_buffer->batch, GENX(PIPE_CONTROL), pc) {
         pc.CommandStreamerStallEnable = true;
         pc.StallAtPixelScoreboard = true;
      }

      uint32_t statistics = pool->pipeline_statistics;
      uint32_t offset = 8;
      while (statistics) {
         uint32_t stat = u_bit_scan(&statistics);
         emit_pipeline_stat(cmd_buffer, stat,
                            anv_address_add(query_addr, offset));
         offset += 16;
      }
      break;
   }

   default:
      unreachable("");
   }
}

void genX(CmdEndQuery)(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);
   struct anv_address query_addr = anv_query_address(pool, query);

   switch (pool->type) {
   case VK_QUERY_TYPE_OCCLUSION:
      emit_ps_depth_count(cmd_buffer, anv_address_add(query_addr, 16));
      emit_query_availability(cmd_buffer, query_addr);
      break;

   case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
      /* TODO: This might only be necessary for certain stats */
      anv_batch_emit(&cmd_buffer->batch, GENX(PIPE_CONTROL), pc) {
         pc.CommandStreamerStallEnable = true;
         pc.StallAtPixelScoreboard = true;
      }

      uint32_t statistics = pool->pipeline_statistics;
      uint32_t offset = 16;
      while (statistics) {
         uint32_t stat = u_bit_scan(&statistics);
         emit_pipeline_stat(cmd_buffer, stat,
                            anv_address_add(query_addr, offset));
         offset += 16;
      }

      emit_query_availability(cmd_buffer, query_addr);
      break;
   }

   default:
      unreachable("");
   }

   /* When multiview is active the spec requires that N consecutive query
    * indices are used, where N is the number of active views in the subpass.
    * The spec allows that we only write the results to one of the queries
    * but we still need to manage result availability for all the query indices.
    * Since we only emit a single query for all active views in the
    * first index, mark the other query indices as being already available
    * with result 0.
    */
   if (cmd_buffer->state.subpass && cmd_buffer->state.subpass->view_mask) {
      const uint32_t num_queries =
         util_bitcount(cmd_buffer->state.subpass->view_mask);
      if (num_queries > 1)
         emit_zero_queries(cmd_buffer, pool, query + 1, num_queries - 1);
   }
}

#define TIMESTAMP 0x2358

void genX(CmdWriteTimestamp)(
    VkCommandBuffer                             commandBuffer,
    VkPipelineStageFlagBits                     pipelineStage,
    VkQueryPool                                 queryPool,
    uint32_t                                    query)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);
   struct anv_address query_addr = anv_query_address(pool, query);

   assert(pool->type == VK_QUERY_TYPE_TIMESTAMP);

   switch (pipelineStage) {
   case VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT:
      emit_srm64(&cmd_buffer->batch, anv_address_add(query_addr, 8), TIMESTAMP);
      break;

   default:
      /* Everything else is bottom-of-pipe */
      anv_batch_emit(&cmd_buffer->batch, GENX(PIPE_CONTROL), pc) {
         pc.DestinationAddressType  = DAT_PPGTT;
         pc.PostSyncOperation       = WriteTimestamp;
         pc.Address                 = anv_address_add(query_addr, 8);

         if (GEN_GEN == 9 && cmd_buffer->device->info.gt == 4)
            pc.CommandStreamerStallEnable = true;
      }
      break;
   }

   emit_query_availability(cmd_buffer, query_addr);

   /* When multiview is active the spec requires that N consecutive query
    * indices are used, where N is the number of active views in the subpass.
    * The spec allows that we only write the results to one of the queries
    * but we still need to manage result availability for all the query indices.
    * Since we only emit a single query for all active views in the
    * first index, mark the other query indices as being already available
    * with result 0.
    */
   if (cmd_buffer->state.subpass && cmd_buffer->state.subpass->view_mask) {
      const uint32_t num_queries =
         util_bitcount(cmd_buffer->state.subpass->view_mask);
      if (num_queries > 1)
         emit_zero_queries(cmd_buffer, pool, query + 1, num_queries - 1);
   }
}

#if GEN_GEN > 7 || GEN_IS_HASWELL

static uint32_t
mi_alu(uint32_t opcode, uint32_t operand1, uint32_t operand2)
{
   struct GENX(MI_MATH_ALU_INSTRUCTION) instr = {
      .ALUOpcode = opcode,
      .Operand1 = operand1,
      .Operand2 = operand2,
   };

   uint32_t dw;
   GENX(MI_MATH_ALU_INSTRUCTION_pack)(NULL, &dw, &instr);

   return dw;
}

#define CS_GPR(n) (0x2600 + (n) * 8)

static void
emit_load_alu_reg_u64(struct anv_batch *batch, uint32_t reg,
                      struct anv_address addr)
{
   anv_batch_emit(batch, GENX(MI_LOAD_REGISTER_MEM), lrm) {
      lrm.RegisterAddress  = reg;
      lrm.MemoryAddress    = anv_address_add(addr, 0);
   }
   anv_batch_emit(batch, GENX(MI_LOAD_REGISTER_MEM), lrm) {
      lrm.RegisterAddress  = reg + 4;
      lrm.MemoryAddress    = anv_address_add(addr, 4);
   }
}

static void
emit_load_alu_reg_imm32(struct anv_batch *batch, uint32_t reg, uint32_t imm)
{
   anv_batch_emit(batch, GENX(MI_LOAD_REGISTER_IMM), lri) {
      lri.RegisterOffset   = reg;
      lri.DataDWord        = imm;
   }
}

static void
emit_load_alu_reg_imm64(struct anv_batch *batch, uint32_t reg, uint64_t imm)
{
   emit_load_alu_reg_imm32(batch, reg, (uint32_t)imm);
   emit_load_alu_reg_imm32(batch, reg + 4, (uint32_t)(imm >> 32));
}

static void
emit_load_alu_reg_reg32(struct anv_batch *batch, uint32_t src, uint32_t dst)
{
   anv_batch_emit(batch, GENX(MI_LOAD_REGISTER_REG), lrr) {
      lrr.SourceRegisterAddress      = src;
      lrr.DestinationRegisterAddress = dst;
   }
}

/*
 * GPR0 = GPR0 & ((1ull << n) - 1);
 */
static void
keep_gpr0_lower_n_bits(struct anv_batch *batch, uint32_t n)
{
   assert(n < 64);
   emit_load_alu_reg_imm64(batch, CS_GPR(1), (1ull << n) - 1);

   uint32_t *dw = anv_batch_emitn(batch, 5, GENX(MI_MATH));
   if (!dw) {
      anv_batch_set_error(batch, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }

   dw[1] = mi_alu(MI_ALU_LOAD, MI_ALU_SRCA, MI_ALU_REG0);
   dw[2] = mi_alu(MI_ALU_LOAD, MI_ALU_SRCB, MI_ALU_REG1);
   dw[3] = mi_alu(MI_ALU_AND, 0, 0);
   dw[4] = mi_alu(MI_ALU_STORE, MI_ALU_REG0, MI_ALU_ACCU);
}

/*
 * GPR0 = GPR0 << 30;
 */
static void
shl_gpr0_by_30_bits(struct anv_batch *batch)
{
   /* First we mask 34 bits of GPR0 to prevent overflow */
   keep_gpr0_lower_n_bits(batch, 34);

   const uint32_t outer_count = 5;
   const uint32_t inner_count = 6;
   STATIC_ASSERT(outer_count * inner_count == 30);
   const uint32_t cmd_len = 1 + inner_count * 4;

   /* We'll emit 5 commands, each shifting GPR0 left by 6 bits, for a total of
    * 30 left shifts.
    */
   for (int o = 0; o < outer_count; o++) {
      /* Submit one MI_MATH to shift left by 6 bits */
      uint32_t *dw = anv_batch_emitn(batch, cmd_len, GENX(MI_MATH));
      if (!dw) {
         anv_batch_set_error(batch, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }

      dw++;
      for (int i = 0; i < inner_count; i++, dw += 4) {
         dw[0] = mi_alu(MI_ALU_LOAD, MI_ALU_SRCA, MI_ALU_REG0);
         dw[1] = mi_alu(MI_ALU_LOAD, MI_ALU_SRCB, MI_ALU_REG0);
         dw[2] = mi_alu(MI_ALU_ADD, 0, 0);
         dw[3] = mi_alu(MI_ALU_STORE, MI_ALU_REG0, MI_ALU_ACCU);
      }
   }
}

/*
 * GPR0 = GPR0 >> 2;
 *
 * Note that the upper 30 bits of GPR are lost!
 */
static void
shr_gpr0_by_2_bits(struct anv_batch *batch)
{
   shl_gpr0_by_30_bits(batch);
   emit_load_alu_reg_reg32(batch, CS_GPR(0) + 4, CS_GPR(0));
   emit_load_alu_reg_imm32(batch, CS_GPR(0) + 4, 0);
}

static void
gpu_write_query_result(struct anv_batch *batch,
                       struct anv_address dst_addr,
                       VkQueryResultFlags flags,
                       uint32_t value_index, uint32_t reg)
{
   if (flags & VK_QUERY_RESULT_64_BIT) {
      emit_srm64(batch, anv_address_add(dst_addr, value_index * 8), reg);
   } else {
      emit_srm32(batch, anv_address_add(dst_addr, value_index * 4), reg);
   }
}

static void
compute_query_result(struct anv_batch *batch, uint32_t dst_reg,
                     struct anv_address addr)
{
   emit_load_alu_reg_u64(batch, CS_GPR(0), anv_address_add(addr, 0));
   emit_load_alu_reg_u64(batch, CS_GPR(1), anv_address_add(addr, 8));

   /* FIXME: We need to clamp the result for 32 bit. */

   uint32_t *dw = anv_batch_emitn(batch, 5, GENX(MI_MATH));
   if (!dw) {
      anv_batch_set_error(batch, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }

   dw[1] = mi_alu(MI_ALU_LOAD, MI_ALU_SRCA, MI_ALU_REG1);
   dw[2] = mi_alu(MI_ALU_LOAD, MI_ALU_SRCB, MI_ALU_REG0);
   dw[3] = mi_alu(MI_ALU_SUB, 0, 0);
   dw[4] = mi_alu(MI_ALU_STORE, dst_reg, MI_ALU_ACCU);
}

void genX(CmdCopyQueryPoolResults)(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount,
    VkBuffer                                    destBuffer,
    VkDeviceSize                                destOffset,
    VkDeviceSize                                destStride,
    VkQueryResultFlags                          flags)
{
   ANV_FROM_HANDLE(anv_cmd_buffer, cmd_buffer, commandBuffer);
   ANV_FROM_HANDLE(anv_query_pool, pool, queryPool);
   ANV_FROM_HANDLE(anv_buffer, buffer, destBuffer);

   /* If render target writes are ongoing, request a render target cache flush
    * to ensure proper ordering of the commands from the 3d pipe and the
    * command streamer.
    */
   if (cmd_buffer->state.pending_pipe_bits & ANV_PIPE_RENDER_TARGET_WRITES) {
      cmd_buffer->state.pending_pipe_bits |=
         ANV_PIPE_RENDER_TARGET_CACHE_FLUSH_BIT;
   }

   if ((flags & VK_QUERY_RESULT_WAIT_BIT) ||
       (cmd_buffer->state.pending_pipe_bits & ANV_PIPE_FLUSH_BITS)) {
      cmd_buffer->state.pending_pipe_bits |= ANV_PIPE_CS_STALL_BIT;
      genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);
   }

   struct anv_address dest_addr = anv_address_add(buffer->address, destOffset);
   for (uint32_t i = 0; i < queryCount; i++) {
      struct anv_address query_addr = anv_query_address(pool, firstQuery + i);
      uint32_t idx = 0;
      switch (pool->type) {
      case VK_QUERY_TYPE_OCCLUSION:
         compute_query_result(&cmd_buffer->batch, MI_ALU_REG2,
                              anv_address_add(query_addr, 8));
         gpu_write_query_result(&cmd_buffer->batch, dest_addr,
                                flags, idx++, CS_GPR(2));
         break;

      case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
         uint32_t statistics = pool->pipeline_statistics;
         while (statistics) {
            uint32_t stat = u_bit_scan(&statistics);

            compute_query_result(&cmd_buffer->batch, MI_ALU_REG0,
                                 anv_address_add(query_addr, idx * 16 + 8));

            /* WaDividePSInvocationCountBy4:HSW,BDW */
            if ((cmd_buffer->device->info.gen == 8 ||
                 cmd_buffer->device->info.is_haswell) &&
                (1 << stat) == VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT) {
               shr_gpr0_by_2_bits(&cmd_buffer->batch);
            }

            gpu_write_query_result(&cmd_buffer->batch, dest_addr,
                                   flags, idx++, CS_GPR(0));
         }
         assert(idx == util_bitcount(pool->pipeline_statistics));
         break;
      }

      case VK_QUERY_TYPE_TIMESTAMP:
         emit_load_alu_reg_u64(&cmd_buffer->batch,
                               CS_GPR(2), anv_address_add(query_addr, 8));
         gpu_write_query_result(&cmd_buffer->batch, dest_addr,
                                flags, 0, CS_GPR(2));
         break;

      default:
         unreachable("unhandled query type");
      }

      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
         emit_load_alu_reg_u64(&cmd_buffer->batch, CS_GPR(0), query_addr);
         gpu_write_query_result(&cmd_buffer->batch, dest_addr,
                                flags, idx, CS_GPR(0));
      }

      dest_addr = anv_address_add(dest_addr, destStride);
   }
}

#else
void genX(CmdCopyQueryPoolResults)(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount,
    VkBuffer                                    destBuffer,
    VkDeviceSize                                destOffset,
    VkDeviceSize                                destStride,
    VkQueryResultFlags                          flags)
{
   anv_finishme("Queries not yet supported on Ivy Bridge");
}
#endif
