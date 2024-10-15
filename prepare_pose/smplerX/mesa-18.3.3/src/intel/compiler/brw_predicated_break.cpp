/*
 * Copyright © 2013 Intel Corporation
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

#include "brw_cfg.h"

using namespace brw;

/** @file brw_predicated_break.cpp
 *
 * Loops are often structured as
 *
 * loop:
 *    CMP.f0
 *    (+f0) IF
 *    BREAK
 *    ENDIF
 *    ...
 *    WHILE loop
 *
 * This peephole pass removes the IF and ENDIF instructions and predicates the
 * BREAK, dropping two instructions from the loop body.
 *
 * If the loop was a DO { ... } WHILE loop, it looks like
 *
 * loop:
 *    ...
 *    CMP.f0
 *    (+f0) IF
 *    BREAK
 *    ENDIF
 *    WHILE loop
 *
 * and we can remove the BREAK instruction and predicate the WHILE.
 */

bool
opt_predicated_break(backend_shader *s)
{
   bool progress = false;

   foreach_block (block, s->cfg) {
      if (block->start_ip != block->end_ip)
         continue;

      /* BREAK and CONTINUE instructions, by definition, can only be found at
       * the ends of basic blocks.
       */
      backend_instruction *jump_inst = block->end();
      if (jump_inst->opcode != BRW_OPCODE_BREAK &&
          jump_inst->opcode != BRW_OPCODE_CONTINUE)
         continue;

      backend_instruction *if_inst = block->prev()->end();
      if (if_inst->opcode != BRW_OPCODE_IF)
         continue;

      backend_instruction *endif_inst = block->next()->start();
      if (endif_inst->opcode != BRW_OPCODE_ENDIF)
         continue;

      bblock_t *jump_block = block;
      bblock_t *if_block = jump_block->prev();
      bblock_t *endif_block = jump_block->next();

      jump_inst->predicate = if_inst->predicate;
      jump_inst->predicate_inverse = if_inst->predicate_inverse;

      bblock_t *earlier_block = if_block;
      if (if_block->start_ip == if_block->end_ip) {
         earlier_block = if_block->prev();
      }

      if_inst->remove(if_block);

      bblock_t *later_block = endif_block;
      if (endif_block->start_ip == endif_block->end_ip) {
         later_block = endif_block->next();
      }
      endif_inst->remove(endif_block);

      if (!earlier_block->ends_with_control_flow()) {
         earlier_block->children.make_empty();
         earlier_block->add_successor(s->cfg->mem_ctx, jump_block);
      }

      if (!later_block->starts_with_control_flow()) {
         later_block->parents.make_empty();
      }
      jump_block->add_successor(s->cfg->mem_ctx, later_block);

      if (earlier_block->can_combine_with(jump_block)) {
         earlier_block->combine_with(jump_block);

         block = earlier_block;
      }

      /* Now look at the first instruction of the block following the BREAK. If
       * it's a WHILE, we can delete the break, predicate the WHILE, and join
       * the two basic blocks.
       */
      bblock_t *while_block = earlier_block->next();
      backend_instruction *while_inst = while_block->start();

      if (jump_inst->opcode == BRW_OPCODE_BREAK &&
          while_inst->opcode == BRW_OPCODE_WHILE &&
          while_inst->predicate == BRW_PREDICATE_NONE) {
         jump_inst->remove(earlier_block);
         while_inst->predicate = jump_inst->predicate;
         while_inst->predicate_inverse = !jump_inst->predicate_inverse;

         earlier_block->children.make_empty();
         earlier_block->add_successor(s->cfg->mem_ctx, while_block);

         assert(earlier_block->can_combine_with(while_block));
         earlier_block->combine_with(while_block);

         earlier_block->next()->parents.make_empty();
         earlier_block->add_successor(s->cfg->mem_ctx, earlier_block->next());
      }

      progress = true;
   }

   if (progress)
      s->invalidate_live_intervals();

   return progress;
}
