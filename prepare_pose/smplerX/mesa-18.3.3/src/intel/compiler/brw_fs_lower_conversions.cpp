/*
 * Copyright © 2015 Connor Abbott
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
#include "brw_cfg.h"
#include "brw_fs_builder.h"

using namespace brw;

static bool
supports_type_conversion(const fs_inst *inst) {
   switch (inst->opcode) {
   case BRW_OPCODE_MOV:
   case SHADER_OPCODE_MOV_INDIRECT:
      return true;
   case BRW_OPCODE_SEL:
      return inst->dst.type == get_exec_type(inst);
   default:
      /* FIXME: We assume the opcodes don't explicitly mentioned
       * before just work fine with arbitrary conversions.
       */
      return true;
   }
}

/* From the SKL PRM Vol 2a, "Move":
 *
 *    "A mov with the same source and destination type, no source modifier,
 *     and no saturation is a raw move. A packed byte destination region (B
 *     or UB type with HorzStride == 1 and ExecSize > 1) can only be written
 *     using raw move."
 */
static bool
is_byte_raw_mov (const fs_inst *inst)
{
   return type_sz(inst->dst.type) == 1 &&
          inst->opcode == BRW_OPCODE_MOV &&
          inst->src[0].type == inst->dst.type &&
          !inst->saturate &&
          !inst->src[0].negate &&
          !inst->src[0].abs;
}

bool
fs_visitor::lower_conversions()
{
   bool progress = false;

   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      const fs_builder ibld(this, block, inst);
      fs_reg dst = inst->dst;
      bool saturate = inst->saturate;

      if (supports_type_conversion(inst)) {
         if (type_sz(inst->dst.type) < get_exec_type_size(inst) &&
             !is_byte_raw_mov(inst)) {
            /* From the Broadwell PRM, 3D Media GPGPU, "Double Precision Float to
             * Single Precision Float":
             *
             *    The upper Dword of every Qword will be written with undefined
             *    value when converting DF to F.
             *
             * So we need to allocate a temporary that's two registers, and then do
             * a strided MOV to get the lower DWord of every Qword that has the
             * result.
             *
             * This restriction applies, in general, whenever we convert to
             * a type with a smaller bit-size.
             */
            fs_reg temp = ibld.vgrf(get_exec_type(inst));
            fs_reg strided_temp = subscript(temp, dst.type, 0);

            assert(inst->size_written == inst->dst.component_size(inst->exec_size));
            inst->dst = strided_temp;
            inst->saturate = false;
            /* As it is an strided destination, we write n-times more being n the
             * size ratio between source and destination types. Update
             * size_written accordingly.
             */
            inst->size_written = inst->dst.component_size(inst->exec_size);

            fs_inst *mov = ibld.at(block, inst->next).MOV(dst, strided_temp);
            mov->saturate = saturate;
            mov->predicate = inst->predicate;

            progress = true;
         }
      } else {
         fs_reg temp0 = ibld.vgrf(get_exec_type(inst));

         assert(inst->size_written == inst->dst.component_size(inst->exec_size));
         inst->dst = temp0;
         /* As it is an strided destination, we write n-times more being n the
          * size ratio between source and destination types. Update
          * size_written accordingly.
          */
         inst->size_written = inst->dst.component_size(inst->exec_size);
         inst->saturate = false;
         /* Now, do the conversion to original destination's type. In next iteration,
          * we will lower it if it is a d2f conversion.
          */
         ibld.at(block, inst->next).MOV(dst, temp0)->saturate = saturate;

         progress = true;
      }
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}
