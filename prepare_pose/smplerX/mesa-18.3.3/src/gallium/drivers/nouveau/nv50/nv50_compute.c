/*
 * Copyright 2012 Francisco Jerez
 * Copyright 2015 Samuel Pitoiset
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE COPYRIGHT OWNER(S) AND/OR ITS SUPPLIERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "nv50/nv50_context.h"
#include "nv50/nv50_compute.xml.h"

#include "codegen/nv50_ir_driver.h"

int
nv50_screen_compute_setup(struct nv50_screen *screen,
                          struct nouveau_pushbuf *push)
{
   struct nouveau_device *dev = screen->base.device;
   struct nouveau_object *chan = screen->base.channel;
   struct nv04_fifo *fifo = (struct nv04_fifo *)chan->data;
   unsigned obj_class;
   int i, ret;

   switch (dev->chipset & 0xf0) {
   case 0x50:
   case 0x80:
   case 0x90:
      obj_class = NV50_COMPUTE_CLASS;
      break;
   case 0xa0:
      switch (dev->chipset) {
      case 0xa3:
      case 0xa5:
      case 0xa8:
         obj_class = NVA3_COMPUTE_CLASS;
         break;
      default:
         obj_class = NV50_COMPUTE_CLASS;
         break;
      }
      break;
   default:
      NOUVEAU_ERR("unsupported chipset: NV%02x\n", dev->chipset);
      return -1;
   }

   ret = nouveau_object_new(chan, 0xbeef50c0, obj_class, NULL, 0,
                            &screen->compute);
   if (ret)
      return ret;

   BEGIN_NV04(push, SUBC_CP(NV01_SUBCHAN_OBJECT), 1);
   PUSH_DATA (push, screen->compute->handle);

   BEGIN_NV04(push, NV50_CP(UNK02A0), 1);
   PUSH_DATA (push, 1);
   BEGIN_NV04(push, NV50_CP(DMA_STACK), 1);
   PUSH_DATA (push, fifo->vram);
   BEGIN_NV04(push, NV50_CP(STACK_ADDRESS_HIGH), 2);
   PUSH_DATAh(push, screen->stack_bo->offset);
   PUSH_DATA (push, screen->stack_bo->offset);
   BEGIN_NV04(push, NV50_CP(STACK_SIZE_LOG), 1);
   PUSH_DATA (push, 4);

   BEGIN_NV04(push, NV50_CP(UNK0290), 1);
   PUSH_DATA (push, 1);
   BEGIN_NV04(push, NV50_CP(LANES32_ENABLE), 1);
   PUSH_DATA (push, 1);
   BEGIN_NV04(push, NV50_CP(REG_MODE), 1);
   PUSH_DATA (push, NV50_COMPUTE_REG_MODE_STRIPED);
   BEGIN_NV04(push, NV50_CP(UNK0384), 1);
   PUSH_DATA (push, 0x100);
   BEGIN_NV04(push, NV50_CP(DMA_GLOBAL), 1);
   PUSH_DATA (push, fifo->vram);

   for (i = 0; i < 15; i++) {
      BEGIN_NV04(push, NV50_CP(GLOBAL_ADDRESS_HIGH(i)), 2);
      PUSH_DATA (push, 0);
      PUSH_DATA (push, 0);
      BEGIN_NV04(push, NV50_CP(GLOBAL_LIMIT(i)), 1);
      PUSH_DATA (push, 0);
      BEGIN_NV04(push, NV50_CP(GLOBAL_MODE(i)), 1);
      PUSH_DATA (push, NV50_COMPUTE_GLOBAL_MODE_LINEAR);
   }

   BEGIN_NV04(push, NV50_CP(GLOBAL_ADDRESS_HIGH(15)), 2);
   PUSH_DATA (push, 0);
   PUSH_DATA (push, 0);
   BEGIN_NV04(push, NV50_CP(GLOBAL_LIMIT(15)), 1);
   PUSH_DATA (push, ~0);
   BEGIN_NV04(push, NV50_CP(GLOBAL_MODE(15)), 1);
   PUSH_DATA (push, NV50_COMPUTE_GLOBAL_MODE_LINEAR);

   BEGIN_NV04(push, NV50_CP(LOCAL_WARPS_LOG_ALLOC), 1);
   PUSH_DATA (push, 7);
   BEGIN_NV04(push, NV50_CP(LOCAL_WARPS_NO_CLAMP), 1);
   PUSH_DATA (push, 1);
   BEGIN_NV04(push, NV50_CP(STACK_WARPS_LOG_ALLOC), 1);
   PUSH_DATA (push, 7);
   BEGIN_NV04(push, NV50_CP(STACK_WARPS_NO_CLAMP), 1);
   PUSH_DATA (push, 1);
   BEGIN_NV04(push, NV50_CP(USER_PARAM_COUNT), 1);
   PUSH_DATA (push, 0);

   BEGIN_NV04(push, NV50_CP(DMA_TEXTURE), 1);
   PUSH_DATA (push, fifo->vram);
   BEGIN_NV04(push, NV50_CP(TEX_LIMITS), 1);
   PUSH_DATA (push, 0x54);
   BEGIN_NV04(push, NV50_CP(LINKED_TSC), 1);
   PUSH_DATA (push, 0);

   BEGIN_NV04(push, NV50_CP(DMA_TIC), 1);
   PUSH_DATA (push, fifo->vram);
   BEGIN_NV04(push, NV50_CP(TIC_ADDRESS_HIGH), 3);
   PUSH_DATAh(push, screen->txc->offset);
   PUSH_DATA (push, screen->txc->offset);
   PUSH_DATA (push, NV50_TIC_MAX_ENTRIES - 1);

   BEGIN_NV04(push, NV50_CP(DMA_TSC), 1);
   PUSH_DATA (push, fifo->vram);
   BEGIN_NV04(push, NV50_CP(TSC_ADDRESS_HIGH), 3);
   PUSH_DATAh(push, screen->txc->offset + 65536);
   PUSH_DATA (push, screen->txc->offset + 65536);
   PUSH_DATA (push, NV50_TSC_MAX_ENTRIES - 1);

   BEGIN_NV04(push, NV50_CP(DMA_CODE_CB), 1);
   PUSH_DATA (push, fifo->vram);

   BEGIN_NV04(push, NV50_CP(DMA_LOCAL), 1);
   PUSH_DATA (push, fifo->vram);
   BEGIN_NV04(push, NV50_CP(LOCAL_ADDRESS_HIGH), 2);
   PUSH_DATAh(push, screen->tls_bo->offset + 65536);
   PUSH_DATA (push, screen->tls_bo->offset + 65536);
   BEGIN_NV04(push, NV50_CP(LOCAL_SIZE_LOG), 1);
   PUSH_DATA (push, util_logbase2((screen->max_tls_space / ONE_TEMP_SIZE) * 2));

   return 0;
}

static void
nv50_compute_validate_globals(struct nv50_context *nv50)
{
   unsigned i;

   for (i = 0; i < nv50->global_residents.size / sizeof(struct pipe_resource *);
        ++i) {
      struct pipe_resource *res = *util_dynarray_element(
         &nv50->global_residents, struct pipe_resource *, i);
      if (res)
         nv50_add_bufctx_resident(nv50->bufctx_cp, NV50_BIND_CP_GLOBAL,
                                  nv04_resource(res), NOUVEAU_BO_RDWR);
   }
}

static struct nv50_state_validate
validate_list_cp[] = {
   { nv50_compprog_validate,              NV50_NEW_CP_PROGRAM     },
   { nv50_compute_validate_globals,       NV50_NEW_CP_GLOBALS     },
};

static bool
nv50_state_validate_cp(struct nv50_context *nv50, uint32_t mask)
{
   bool ret;

   /* TODO: validate textures, samplers, surfaces */
   ret = nv50_state_validate(nv50, mask, validate_list_cp,
                             ARRAY_SIZE(validate_list_cp), &nv50->dirty_cp,
                             nv50->bufctx_cp);

   if (unlikely(nv50->state.flushed))
      nv50_bufctx_fence(nv50->bufctx_cp, true);
   return ret;
}

static void
nv50_compute_upload_input(struct nv50_context *nv50, const uint32_t *input)
{
   struct nv50_screen *screen = nv50->screen;
   struct nouveau_pushbuf *push = screen->base.pushbuf;
   unsigned size = align(nv50->compprog->parm_size, 0x4);

   BEGIN_NV04(push, NV50_CP(USER_PARAM_COUNT), 1);
   PUSH_DATA (push, (size / 4) << 8);

   if (size) {
      struct nouveau_mm_allocation *mm;
      struct nouveau_bo *bo = NULL;
      unsigned offset;

      mm = nouveau_mm_allocate(screen->base.mm_GART, size, &bo, &offset);
      assert(mm);

      nouveau_bo_map(bo, 0, screen->base.client);
      memcpy(bo->map + offset, input, size);

      nouveau_bufctx_refn(nv50->bufctx, 0, bo, NOUVEAU_BO_GART | NOUVEAU_BO_RD);
      nouveau_pushbuf_bufctx(push, nv50->bufctx);
      nouveau_pushbuf_validate(push);

      BEGIN_NV04(push, NV50_CP(USER_PARAM(0)), size / 4);
      nouveau_pushbuf_data(push, bo, offset, size);

      nouveau_fence_work(screen->base.fence.current, nouveau_mm_free_work, mm);
      nouveau_bo_ref(NULL, &bo);
      nouveau_bufctx_reset(nv50->bufctx, 0);
   }
}

static uint32_t
nv50_compute_find_symbol(struct nv50_context *nv50, uint32_t label)
{
   struct nv50_program *prog = nv50->compprog;
   const struct nv50_ir_prog_symbol *syms =
      (const struct nv50_ir_prog_symbol *)prog->cp.syms;
   unsigned i;

   for (i = 0; i < prog->cp.num_syms; ++i) {
      if (syms[i].label == label)
         return prog->code_base + syms[i].offset;
   }
   return prog->code_base; /* no symbols or symbol not found */
}

void
nv50_launch_grid(struct pipe_context *pipe, const struct pipe_grid_info *info)
{
   struct nv50_context *nv50 = nv50_context(pipe);
   struct nouveau_pushbuf *push = nv50->base.pushbuf;
   unsigned block_size = info->block[0] * info->block[1] * info->block[2];
   struct nv50_program *cp = nv50->compprog;
   bool ret;

   ret = !nv50_state_validate_cp(nv50, ~0);
   if (ret) {
      NOUVEAU_ERR("Failed to launch grid !\n");
      return;
   }

   nv50_compute_upload_input(nv50, info->input);

   BEGIN_NV04(push, NV50_CP(CP_START_ID), 1);
   PUSH_DATA (push, nv50_compute_find_symbol(nv50, info->pc));

   BEGIN_NV04(push, NV50_CP(SHARED_SIZE), 1);
   PUSH_DATA (push, align(cp->cp.smem_size + cp->parm_size + 0x10, 0x40));
   BEGIN_NV04(push, NV50_CP(CP_REG_ALLOC_TEMP), 1);
   PUSH_DATA (push, cp->max_gpr);

   /* grid/block setup */
   BEGIN_NV04(push, NV50_CP(BLOCKDIM_XY), 2);
   PUSH_DATA (push, info->block[1] << 16 | info->block[0]);
   PUSH_DATA (push, info->block[2]);
   BEGIN_NV04(push, NV50_CP(BLOCK_ALLOC), 1);
   PUSH_DATA (push, 1 << 16 | block_size);
   BEGIN_NV04(push, NV50_CP(BLOCKDIM_LATCH), 1);
   PUSH_DATA (push, 1);
   BEGIN_NV04(push, NV50_CP(GRIDDIM), 1);
   PUSH_DATA (push, info->grid[1] << 16 | info->grid[0]);
   BEGIN_NV04(push, NV50_CP(GRIDID), 1);
   PUSH_DATA (push, 1);

   /* kernel launching */
   BEGIN_NV04(push, NV50_CP(LAUNCH), 1);
   PUSH_DATA (push, 0);
   BEGIN_NV04(push, SUBC_CP(NV50_GRAPH_SERIALIZE), 1);
   PUSH_DATA (push, 0);

   /* bind a compute shader clobbers fragment shader state */
   nv50->dirty_3d |= NV50_NEW_3D_FRAGPROG;
}
