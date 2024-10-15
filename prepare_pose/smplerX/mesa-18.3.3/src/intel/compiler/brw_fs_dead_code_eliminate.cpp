/*
 * Copyright © 2014 Intel Corporation
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

#include "brw_fs.h"
#include "brw_fs_live_variables.h"
#include "brw_cfg.h"

/** @file brw_fs_dead_code_eliminate.cpp
 *
 * Dataflow-aware dead code elimination.
 *
 * Walks the instruction list from the bottom, removing instructions that
 * have results that both aren't used in later blocks and haven't been read
 * yet in the tail end of this block.
 */

/**
 * Is it safe to eliminate the instruction?
 */
static bool
can_eliminate(const fs_inst *inst, BITSET_WORD *flag_live)
{
    return !inst->is_control_flow() &&
           !inst->has_side_effects() &&
           !(flag_live[0] & inst->flags_written()) &&
           !inst->writes_accumulator;
}

/**
 * Is it safe to omit the write, making the destination ARF null?
 */
static bool
can_omit_write(const fs_inst *inst)
{
   switch (inst->opcode) {
   case SHADER_OPCODE_UNTYPED_ATOMIC:
   case SHADER_OPCODE_UNTYPED_ATOMIC_LOGICAL:
   case SHADER_OPCODE_UNTYPED_ATOMIC_FLOAT:
   case SHADER_OPCODE_UNTYPED_ATOMIC_FLOAT_LOGICAL:
   case SHADER_OPCODE_TYPED_ATOMIC:
   case SHADER_OPCODE_TYPED_ATOMIC_LOGICAL:
      return true;
   default:
      /* We can eliminate the destination write for ordinary instructions,
       * but not most SENDs.
       */
      if (inst->opcode < 128 && inst->mlen == 0)
         return true;

      /* It might not be safe for other virtual opcodes. */
      return false;
   }
}

bool
fs_visitor::dead_code_eliminate()
{
   bool progress = false;

   calculate_live_intervals();

   int num_vars = live_intervals->num_vars;
   BITSET_WORD *live = rzalloc_array(NULL, BITSET_WORD, BITSET_WORDS(num_vars));
   BITSET_WORD *flag_live = rzalloc_array(NULL, BITSET_WORD, 1);

   foreach_block_reverse_safe(block, cfg) {
      memcpy(live, live_intervals->block_data[block->num].liveout,
             sizeof(BITSET_WORD) * BITSET_WORDS(num_vars));
      memcpy(flag_live, live_intervals->block_data[block->num].flag_liveout,
             sizeof(BITSET_WORD));

      foreach_inst_in_block_reverse_safe(fs_inst, inst, block) {
         if (inst->dst.file == VGRF) {
            const unsigned var = live_intervals->var_from_reg(inst->dst);
            bool result_live = false;

            for (unsigned i = 0; i < regs_written(inst); i++)
               result_live |= BITSET_TEST(live, var + i);

            if (!result_live &&
                (can_omit_write(inst) || can_eliminate(inst, flag_live))) {
               inst->dst = fs_reg(retype(brw_null_reg(), inst->dst.type));
               progress = true;
            }
         }

         if (inst->dst.is_null() && can_eliminate(inst, flag_live)) {
            inst->opcode = BRW_OPCODE_NOP;
            progress = true;
         }

         if (inst->dst.file == VGRF) {
            if (!inst->is_partial_write()) {
               int var = live_intervals->var_from_reg(inst->dst);
               for (unsigned i = 0; i < regs_written(inst); i++) {
                  BITSET_CLEAR(live, var + i);
               }
            }
         }

         if (!inst->predicate && inst->exec_size >= 8)
            flag_live[0] &= ~inst->flags_written();

         if (inst->opcode == BRW_OPCODE_NOP) {
            inst->remove(block);
            continue;
         }

         for (int i = 0; i < inst->sources; i++) {
            if (inst->src[i].file == VGRF) {
               int var = live_intervals->var_from_reg(inst->src[i]);

               for (unsigned j = 0; j < regs_read(inst, i); j++) {
                  BITSET_SET(live, var + j);
               }
            }
         }

         flag_live[0] |= inst->flags_read(devinfo);
      }
   }

   ralloc_free(live);
   ralloc_free(flag_live);

   if (progress)
      invalidate_live_intervals();

   return progress;
}
