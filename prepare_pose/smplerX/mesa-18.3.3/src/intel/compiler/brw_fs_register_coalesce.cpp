/*
 * Copyright © 2012 Intel Corporation
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

/** @file brw_fs_register_coalesce.cpp
 *
 * Implements register coalescing: Checks if the two registers involved in a
 * raw move don't interfere, in which case they can both be stored in the same
 * place and the MOV removed.
 *
 * To do this, all uses of the source of the MOV in the shader are replaced
 * with the destination of the MOV. For example:
 *
 * add vgrf3:F, vgrf1:F, vgrf2:F
 * mov vgrf4:F, vgrf3:F
 * mul vgrf5:F, vgrf5:F, vgrf4:F
 *
 * becomes
 *
 * add vgrf4:F, vgrf1:F, vgrf2:F
 * mul vgrf5:F, vgrf5:F, vgrf4:F
 */

#include "brw_fs.h"
#include "brw_cfg.h"
#include "brw_fs_live_variables.h"

static bool
is_nop_mov(const fs_inst *inst)
{
   if (inst->opcode == SHADER_OPCODE_LOAD_PAYLOAD) {
      fs_reg dst = inst->dst;
      for (int i = 0; i < inst->sources; i++) {
         if (!dst.equals(inst->src[i])) {
            return false;
         }
         dst.offset += (i < inst->header_size ? REG_SIZE :
                        inst->exec_size * dst.stride *
                        type_sz(inst->src[i].type));
      }
      return true;
   } else if (inst->opcode == BRW_OPCODE_MOV) {
      return inst->dst.equals(inst->src[0]);
   }

   return false;
}

static bool
is_coalesce_candidate(const fs_visitor *v, const fs_inst *inst)
{
   if ((inst->opcode != BRW_OPCODE_MOV &&
        inst->opcode != SHADER_OPCODE_LOAD_PAYLOAD) ||
       inst->is_partial_write() ||
       inst->saturate ||
       inst->src[0].file != VGRF ||
       inst->src[0].negate ||
       inst->src[0].abs ||
       !inst->src[0].is_contiguous() ||
       inst->dst.file != VGRF ||
       inst->dst.type != inst->src[0].type) {
      return false;
   }

   if (v->alloc.sizes[inst->src[0].nr] >
       v->alloc.sizes[inst->dst.nr])
      return false;

   if (inst->opcode == SHADER_OPCODE_LOAD_PAYLOAD) {
      if (!inst->is_copy_payload(v->alloc)) {
         return false;
      }
   }

   return true;
}

static bool
can_coalesce_vars(brw::fs_live_variables *live_intervals,
                  const cfg_t *cfg, const fs_inst *inst,
                  int dst_var, int src_var)
{
   if (!live_intervals->vars_interfere(src_var, dst_var))
      return true;

   int dst_start = live_intervals->start[dst_var];
   int dst_end = live_intervals->end[dst_var];
   int src_start = live_intervals->start[src_var];
   int src_end = live_intervals->end[src_var];

   /* Variables interfere and one line range isn't a subset of the other. */
   if ((dst_end > src_end && src_start < dst_start) ||
       (src_end > dst_end && dst_start < src_start))
      return false;

   /* Check for a write to either register in the intersection of their live
    * ranges.
    */
   int start_ip = MAX2(dst_start, src_start);
   int end_ip = MIN2(dst_end, src_end);

   foreach_block(block, cfg) {
      if (block->end_ip < start_ip)
         continue;

      int scan_ip = block->start_ip - 1;

      foreach_inst_in_block(fs_inst, scan_inst, block) {
         scan_ip++;

         /* Ignore anything before the intersection of the live ranges */
         if (scan_ip < start_ip)
            continue;

         /* Ignore the copying instruction itself */
         if (scan_inst == inst)
            continue;

         if (scan_ip > end_ip)
            return true; /* registers do not interfere */

         if (regions_overlap(scan_inst->dst, scan_inst->size_written,
                             inst->dst, inst->size_written) ||
             regions_overlap(scan_inst->dst, scan_inst->size_written,
                             inst->src[0], inst->size_read(0)))
            return false; /* registers interfere */
      }
   }

   return true;
}

bool
fs_visitor::register_coalesce()
{
   bool progress = false;

   calculate_live_intervals();

   int src_size = 0;
   int channels_remaining = 0;
   int src_reg = -1, dst_reg = -1;
   int dst_reg_offset[MAX_VGRF_SIZE];
   fs_inst *mov[MAX_VGRF_SIZE];
   int dst_var[MAX_VGRF_SIZE];
   int src_var[MAX_VGRF_SIZE];

   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      if (!is_coalesce_candidate(this, inst))
         continue;

      if (is_nop_mov(inst)) {
         inst->opcode = BRW_OPCODE_NOP;
         progress = true;
         continue;
      }

      if (src_reg != inst->src[0].nr) {
         src_reg = inst->src[0].nr;

         src_size = alloc.sizes[inst->src[0].nr];
         assert(src_size <= MAX_VGRF_SIZE);

         channels_remaining = src_size;
         memset(mov, 0, sizeof(mov));

         dst_reg = inst->dst.nr;
      }

      if (dst_reg != inst->dst.nr)
         continue;

      if (inst->opcode == SHADER_OPCODE_LOAD_PAYLOAD) {
         for (int i = 0; i < src_size; i++) {
            dst_reg_offset[i] = i;
         }
         mov[0] = inst;
         channels_remaining -= regs_written(inst);
      } else {
         const int offset = inst->src[0].offset / REG_SIZE;
         if (mov[offset]) {
            /* This is the second time that this offset in the register has
             * been set.  This means, in particular, that inst->dst was
             * live before this instruction and that the live ranges of
             * inst->dst and inst->src[0] overlap and we can't coalesce the
             * two variables.  Let's ensure that doesn't happen.
             */
            channels_remaining = -1;
            continue;
         }
         for (unsigned i = 0; i < MAX2(inst->size_written / REG_SIZE, 1); i++)
            dst_reg_offset[offset + i] = inst->dst.offset / REG_SIZE + i;
         mov[offset] = inst;
         channels_remaining -= regs_written(inst);
      }

      if (channels_remaining)
         continue;

      bool can_coalesce = true;
      for (int i = 0; i < src_size; i++) {
         if (dst_reg_offset[i] != dst_reg_offset[0] + i) {
            /* Registers are out-of-order. */
            can_coalesce = false;
            src_reg = -1;
            break;
         }

         dst_var[i] = live_intervals->var_from_vgrf[dst_reg] + dst_reg_offset[i];
         src_var[i] = live_intervals->var_from_vgrf[src_reg] + i;

         if (!can_coalesce_vars(live_intervals, cfg, inst,
                                dst_var[i], src_var[i])) {
            can_coalesce = false;
            src_reg = -1;
            break;
         }
      }

      if (!can_coalesce)
         continue;

      progress = true;

      for (int i = 0; i < src_size; i++) {
         if (mov[i]) {
            mov[i]->opcode = BRW_OPCODE_NOP;
            mov[i]->conditional_mod = BRW_CONDITIONAL_NONE;
            mov[i]->dst = reg_undef;
            for (int j = 0; j < mov[i]->sources; j++) {
               mov[i]->src[j] = reg_undef;
            }
         }
      }

      foreach_block_and_inst(block, fs_inst, scan_inst, cfg) {
         if (scan_inst->dst.file == VGRF &&
             scan_inst->dst.nr == src_reg) {
            scan_inst->dst.nr = dst_reg;
            scan_inst->dst.offset = scan_inst->dst.offset % REG_SIZE +
               dst_reg_offset[scan_inst->dst.offset / REG_SIZE] * REG_SIZE;
         }

         for (int j = 0; j < scan_inst->sources; j++) {
            if (scan_inst->src[j].file == VGRF &&
                scan_inst->src[j].nr == src_reg) {
               scan_inst->src[j].nr = dst_reg;
               scan_inst->src[j].offset = scan_inst->src[j].offset % REG_SIZE +
                  dst_reg_offset[scan_inst->src[j].offset / REG_SIZE] * REG_SIZE;
            }
         }
      }

      for (int i = 0; i < src_size; i++) {
         live_intervals->start[dst_var[i]] =
            MIN2(live_intervals->start[dst_var[i]],
                 live_intervals->start[src_var[i]]);
         live_intervals->end[dst_var[i]] =
            MAX2(live_intervals->end[dst_var[i]],
                 live_intervals->end[src_var[i]]);
      }
      src_reg = -1;
   }

   if (progress) {
      foreach_block_and_inst_safe (block, backend_instruction, inst, cfg) {
         if (inst->opcode == BRW_OPCODE_NOP) {
            inst->remove(block);
         }
      }

      invalidate_live_intervals();
   }

   return progress;
}
