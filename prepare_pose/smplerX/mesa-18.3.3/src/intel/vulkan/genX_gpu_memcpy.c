/*
 * Copyright © 2016 Intel Corporation
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

/**
 * This file implements some lightweight memcpy/memset operations on the GPU
 * using a vertex buffer and streamout.
 */

/**
 * Returns the greatest common divisor of a and b that is a power of two.
 */
static uint64_t
gcd_pow2_u64(uint64_t a, uint64_t b)
{
   assert(a > 0 || b > 0);

   unsigned a_log2 = ffsll(a) - 1;
   unsigned b_log2 = ffsll(b) - 1;

   /* If either a or b is 0, then a_log2 or b_log2 will be UINT_MAX in which
    * case, the MIN2() will take the other one.  If both are 0 then we will
    * hit the assert above.
    */
   return 1 << MIN2(a_log2, b_log2);
}

void
genX(cmd_buffer_mi_memcpy)(struct anv_cmd_buffer *cmd_buffer,
                           struct anv_address dst, struct anv_address src,
                           uint32_t size)
{
   /* This memcpy operates in units of dwords. */
   assert(size % 4 == 0);
   assert(dst.offset % 4 == 0);
   assert(src.offset % 4 == 0);

#if GEN_GEN == 7
   /* On gen7, the combination of commands used here(MI_LOAD_REGISTER_MEM
    * and MI_STORE_REGISTER_MEM) can cause GPU hangs if any rendering is
    * in-flight when they are issued even if the memory touched is not
    * currently active for rendering.  The weird bit is that it is not the
    * MI_LOAD/STORE_REGISTER_MEM commands which hang but rather the in-flight
    * rendering hangs such that the next stalling command after the
    * MI_LOAD/STORE_REGISTER_MEM commands will catch the hang.
    *
    * It is unclear exactly why this hang occurs.  Both MI commands come with
    * warnings about the 3D pipeline but that doesn't seem to fully explain
    * it.  My (Jason's) best theory is that it has something to do with the
    * fact that we're using a GPU state register as our temporary and that
    * something with reading/writing it is causing problems.
    *
    * In order to work around this issue, we emit a PIPE_CONTROL with the
    * command streamer stall bit set.
    */
   cmd_buffer->state.pending_pipe_bits |= ANV_PIPE_CS_STALL_BIT;
   genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);
#endif

   for (uint32_t i = 0; i < size; i += 4) {
#if GEN_GEN >= 8
      anv_batch_emit(&cmd_buffer->batch, GENX(MI_COPY_MEM_MEM), cp) {
         cp.DestinationMemoryAddress = anv_address_add(dst, i);
         cp.SourceMemoryAddress = anv_address_add(src, i);
      }
#else
      /* IVB does not have a general purpose register for command streamer
       * commands. Therefore, we use an alternate temporary register.
       */
#define TEMP_REG 0x2440 /* GEN7_3DPRIM_BASE_VERTEX */
      anv_batch_emit(&cmd_buffer->batch, GENX(MI_LOAD_REGISTER_MEM), load) {
         load.RegisterAddress = TEMP_REG;
         load.MemoryAddress = anv_address_add(src, i);
      }
      anv_batch_emit(&cmd_buffer->batch, GENX(MI_STORE_REGISTER_MEM), store) {
         store.RegisterAddress = TEMP_REG;
         store.MemoryAddress = anv_address_add(dst, i);
      }
#undef TEMP_REG
#endif
   }
   return;
}

void
genX(cmd_buffer_mi_memset)(struct anv_cmd_buffer *cmd_buffer,
                           struct anv_address dst, uint32_t value,
                           uint32_t size)
{
   /* This memset operates in units of dwords. */
   assert(size % 4 == 0);
   assert(dst.offset % 4 == 0);

   for (uint32_t i = 0; i < size; i += 4) {
      anv_batch_emit(&cmd_buffer->batch, GENX(MI_STORE_DATA_IMM), sdi) {
         sdi.Address = anv_address_add(dst, i);
         sdi.ImmediateData = value;
      }
   }
}

void
genX(cmd_buffer_so_memcpy)(struct anv_cmd_buffer *cmd_buffer,
                           struct anv_address dst, struct anv_address src,
                           uint32_t size)
{
   if (size == 0)
      return;

   assert(dst.offset + size <= dst.bo->size);
   assert(src.offset + size <= src.bo->size);

   /* The maximum copy block size is 4 32-bit components at a time. */
   assert(size % 4 == 0);
   unsigned bs = gcd_pow2_u64(16, size);

   enum isl_format format;
   switch (bs) {
   case 4:  format = ISL_FORMAT_R32_UINT;          break;
   case 8:  format = ISL_FORMAT_R32G32_UINT;       break;
   case 16: format = ISL_FORMAT_R32G32B32A32_UINT; break;
   default:
      unreachable("Invalid size");
   }

   if (!cmd_buffer->state.current_l3_config) {
      const struct gen_l3_config *cfg =
         gen_get_default_l3_config(&cmd_buffer->device->info);
      genX(cmd_buffer_config_l3)(cmd_buffer, cfg);
   }

   genX(cmd_buffer_apply_pipe_flushes)(cmd_buffer);

   genX(flush_pipeline_select_3d)(cmd_buffer);

   uint32_t *dw;
   dw = anv_batch_emitn(&cmd_buffer->batch, 5, GENX(3DSTATE_VERTEX_BUFFERS));
   GENX(VERTEX_BUFFER_STATE_pack)(&cmd_buffer->batch, dw + 1,
      &(struct GENX(VERTEX_BUFFER_STATE)) {
         .VertexBufferIndex = 32, /* Reserved for this */
         .AddressModifyEnable = true,
         .BufferStartingAddress = src,
         .BufferPitch = bs,
         .VertexBufferMOCS = anv_mocs_for_bo(cmd_buffer->device, src.bo),
#if (GEN_GEN >= 8)
         .BufferSize = size,
#else
         .EndAddress = anv_address_add(src, size - 1),
#endif
      });

   dw = anv_batch_emitn(&cmd_buffer->batch, 3, GENX(3DSTATE_VERTEX_ELEMENTS));
   GENX(VERTEX_ELEMENT_STATE_pack)(&cmd_buffer->batch, dw + 1,
      &(struct GENX(VERTEX_ELEMENT_STATE)) {
         .VertexBufferIndex = 32,
         .Valid = true,
         .SourceElementFormat = format,
         .SourceElementOffset = 0,
         .Component0Control = (bs >= 4) ? VFCOMP_STORE_SRC : VFCOMP_STORE_0,
         .Component1Control = (bs >= 8) ? VFCOMP_STORE_SRC : VFCOMP_STORE_0,
         .Component2Control = (bs >= 12) ? VFCOMP_STORE_SRC : VFCOMP_STORE_0,
         .Component3Control = (bs >= 16) ? VFCOMP_STORE_SRC : VFCOMP_STORE_0,
      });

#if GEN_GEN >= 8
   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_VF_SGVS), sgvs);
#endif

   /* Disable all shader stages */
   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_VS), vs);
   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_HS), hs);
   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_TE), te);
   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_DS), DS);
   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_GS), gs);
   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_PS), gs);

   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_SBE), sbe) {
      sbe.VertexURBEntryReadOffset = 1;
      sbe.NumberofSFOutputAttributes = 1;
      sbe.VertexURBEntryReadLength = 1;
#if GEN_GEN >= 8
      sbe.ForceVertexURBEntryReadLength = true;
      sbe.ForceVertexURBEntryReadOffset = true;
#endif

#if GEN_GEN >= 9
      for (unsigned i = 0; i < 32; i++)
         sbe.AttributeActiveComponentFormat[i] = ACF_XYZW;
#endif
   }

   /* Emit URB setup.  We tell it that the VS is active because we want it to
    * allocate space for the VS.  Even though one isn't run, we need VUEs to
    * store the data that VF is going to pass to SOL.
    */
   const unsigned entry_size[4] = { DIV_ROUND_UP(32, 64), 1, 1, 1 };

   genX(emit_urb_setup)(cmd_buffer->device, &cmd_buffer->batch,
                        cmd_buffer->state.current_l3_config,
                        VK_SHADER_STAGE_VERTEX_BIT, entry_size);

   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_SO_BUFFER), sob) {
      sob.SOBufferIndex = 0;
      sob.SOBufferMOCS = anv_mocs_for_bo(cmd_buffer->device, dst.bo),
      sob.SurfaceBaseAddress = dst;

#if GEN_GEN >= 8
      sob.SOBufferEnable = true;
      sob.SurfaceSize = size / 4 - 1;
#else
      sob.SurfacePitch = bs;
      sob.SurfaceEndAddress = anv_address_add(dst, size);
#endif

#if GEN_GEN >= 8
      /* As SOL writes out data, it updates the SO_WRITE_OFFSET registers with
       * the end position of the stream.  We need to reset this value to 0 at
       * the beginning of the run or else SOL will start at the offset from
       * the previous draw.
       */
      sob.StreamOffsetWriteEnable = true;
      sob.StreamOffset = 0;
#endif
   }

#if GEN_GEN <= 7
   /* The hardware can do this for us on BDW+ (see above) */
   anv_batch_emit(&cmd_buffer->batch, GENX(MI_LOAD_REGISTER_IMM), load) {
      load.RegisterOffset = GENX(SO_WRITE_OFFSET0_num);
      load.DataDWord = 0;
   }
#endif

   dw = anv_batch_emitn(&cmd_buffer->batch, 5, GENX(3DSTATE_SO_DECL_LIST),
                        .StreamtoBufferSelects0 = (1 << 0),
                        .NumEntries0 = 1);
   GENX(SO_DECL_ENTRY_pack)(&cmd_buffer->batch, dw + 3,
      &(struct GENX(SO_DECL_ENTRY)) {
         .Stream0Decl = {
            .OutputBufferSlot = 0,
            .RegisterIndex = 0,
            .ComponentMask = (1 << (bs / 4)) - 1,
         },
      });

   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_STREAMOUT), so) {
      so.SOFunctionEnable = true;
      so.RenderingDisable = true;
      so.Stream0VertexReadOffset = 0;
      so.Stream0VertexReadLength = DIV_ROUND_UP(32, 64);
#if GEN_GEN >= 8
      so.Buffer0SurfacePitch = bs;
#else
      so.SOBufferEnable0 = true;
#endif
   }

#if GEN_GEN >= 8
   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_VF_TOPOLOGY), topo) {
      topo.PrimitiveTopologyType = _3DPRIM_POINTLIST;
   }
#endif

   anv_batch_emit(&cmd_buffer->batch, GENX(3DSTATE_VF_STATISTICS), vf) {
      vf.StatisticsEnable = false;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(3DPRIMITIVE), prim) {
      prim.VertexAccessType         = SEQUENTIAL;
      prim.PrimitiveTopologyType    = _3DPRIM_POINTLIST;
      prim.VertexCountPerInstance   = size / bs;
      prim.StartVertexLocation      = 0;
      prim.InstanceCount            = 1;
      prim.StartInstanceLocation    = 0;
      prim.BaseVertexLocation       = 0;
   }

   cmd_buffer->state.gfx.dirty |= ANV_CMD_DIRTY_PIPELINE;
   cmd_buffer->state.pending_pipe_bits |= ANV_PIPE_RENDER_TARGET_WRITES;
}
