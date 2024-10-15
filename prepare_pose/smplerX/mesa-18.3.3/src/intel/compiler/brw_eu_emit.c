/*
 Copyright (C) Intel Corp.  2006.  All Rights Reserved.
 Intel funded Tungsten Graphics to
 develop this 3D driver.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice (including the
 next paragraph) shall be included in all copies or substantial
 portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE COPYRIGHT OWNER(S) AND/OR ITS SUPPLIERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 **********************************************************************/
 /*
  * Authors:
  *   Keith Whitwell <keithw@vmware.com>
  */


#include "brw_eu_defines.h"
#include "brw_eu.h"

#include "util/ralloc.h"

/**
 * Prior to Sandybridge, the SEND instruction accepted non-MRF source
 * registers, implicitly moving the operand to a message register.
 *
 * On Sandybridge, this is no longer the case.  This function performs the
 * explicit move; it should be called before emitting a SEND instruction.
 */
void
gen6_resolve_implied_move(struct brw_codegen *p,
			  struct brw_reg *src,
			  unsigned msg_reg_nr)
{
   const struct gen_device_info *devinfo = p->devinfo;
   if (devinfo->gen < 6)
      return;

   if (src->file == BRW_MESSAGE_REGISTER_FILE)
      return;

   if (src->file != BRW_ARCHITECTURE_REGISTER_FILE || src->nr != BRW_ARF_NULL) {
      brw_push_insn_state(p);
      brw_set_default_exec_size(p, BRW_EXECUTE_8);
      brw_set_default_mask_control(p, BRW_MASK_DISABLE);
      brw_set_default_compression_control(p, BRW_COMPRESSION_NONE);
      brw_MOV(p, retype(brw_message_reg(msg_reg_nr), BRW_REGISTER_TYPE_UD),
	      retype(*src, BRW_REGISTER_TYPE_UD));
      brw_pop_insn_state(p);
   }
   *src = brw_message_reg(msg_reg_nr);
}

static void
gen7_convert_mrf_to_grf(struct brw_codegen *p, struct brw_reg *reg)
{
   /* From the Ivybridge PRM, Volume 4 Part 3, page 218 ("send"):
    * "The send with EOT should use register space R112-R127 for <src>. This is
    *  to enable loading of a new thread into the same slot while the message
    *  with EOT for current thread is pending dispatch."
    *
    * Since we're pretending to have 16 MRFs anyway, we may as well use the
    * registers required for messages with EOT.
    */
   const struct gen_device_info *devinfo = p->devinfo;
   if (devinfo->gen >= 7 && reg->file == BRW_MESSAGE_REGISTER_FILE) {
      reg->file = BRW_GENERAL_REGISTER_FILE;
      reg->nr += GEN7_MRF_HACK_START;
   }
}

void
brw_set_dest(struct brw_codegen *p, brw_inst *inst, struct brw_reg dest)
{
   const struct gen_device_info *devinfo = p->devinfo;

   if (dest.file == BRW_MESSAGE_REGISTER_FILE)
      assert((dest.nr & ~BRW_MRF_COMPR4) < BRW_MAX_MRF(devinfo->gen));
   else if (dest.file != BRW_ARCHITECTURE_REGISTER_FILE)
      assert(dest.nr < 128);

   gen7_convert_mrf_to_grf(p, &dest);

   brw_inst_set_dst_file_type(devinfo, inst, dest.file, dest.type);
   brw_inst_set_dst_address_mode(devinfo, inst, dest.address_mode);

   if (dest.address_mode == BRW_ADDRESS_DIRECT) {
      brw_inst_set_dst_da_reg_nr(devinfo, inst, dest.nr);

      if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
         brw_inst_set_dst_da1_subreg_nr(devinfo, inst, dest.subnr);
	 if (dest.hstride == BRW_HORIZONTAL_STRIDE_0)
	    dest.hstride = BRW_HORIZONTAL_STRIDE_1;
         brw_inst_set_dst_hstride(devinfo, inst, dest.hstride);
      } else {
         brw_inst_set_dst_da16_subreg_nr(devinfo, inst, dest.subnr / 16);
         brw_inst_set_da16_writemask(devinfo, inst, dest.writemask);
         if (dest.file == BRW_GENERAL_REGISTER_FILE ||
             dest.file == BRW_MESSAGE_REGISTER_FILE) {
            assert(dest.writemask != 0);
         }
	 /* From the Ivybridge PRM, Vol 4, Part 3, Section 5.2.4.1:
	  *    Although Dst.HorzStride is a don't care for Align16, HW needs
	  *    this to be programmed as "01".
	  */
         brw_inst_set_dst_hstride(devinfo, inst, 1);
      }
   } else {
      brw_inst_set_dst_ia_subreg_nr(devinfo, inst, dest.subnr);

      /* These are different sizes in align1 vs align16:
       */
      if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
         brw_inst_set_dst_ia1_addr_imm(devinfo, inst,
                                       dest.indirect_offset);
	 if (dest.hstride == BRW_HORIZONTAL_STRIDE_0)
	    dest.hstride = BRW_HORIZONTAL_STRIDE_1;
         brw_inst_set_dst_hstride(devinfo, inst, dest.hstride);
      } else {
         brw_inst_set_dst_ia16_addr_imm(devinfo, inst,
                                        dest.indirect_offset);
	 /* even ignored in da16, still need to set as '01' */
         brw_inst_set_dst_hstride(devinfo, inst, 1);
      }
   }

   /* Generators should set a default exec_size of either 8 (SIMD4x2 or SIMD8)
    * or 16 (SIMD16), as that's normally correct.  However, when dealing with
    * small registers, it can be useful for us to automatically reduce it to
    * match the register size.
    */
   if (p->automatic_exec_sizes) {
      /*
       * In platforms that support fp64 we can emit instructions with a width
       * of 4 that need two SIMD8 registers and an exec_size of 8 or 16. In
       * these cases we need to make sure that these instructions have their
       * exec sizes set properly when they are emitted and we can't rely on
       * this code to fix it.
       */
      bool fix_exec_size;
      if (devinfo->gen >= 6)
         fix_exec_size = dest.width < BRW_EXECUTE_4;
      else
         fix_exec_size = dest.width < BRW_EXECUTE_8;

      if (fix_exec_size)
         brw_inst_set_exec_size(devinfo, inst, dest.width);
   }
}

void
brw_set_src0(struct brw_codegen *p, brw_inst *inst, struct brw_reg reg)
{
   const struct gen_device_info *devinfo = p->devinfo;

   if (reg.file == BRW_MESSAGE_REGISTER_FILE)
      assert((reg.nr & ~BRW_MRF_COMPR4) < BRW_MAX_MRF(devinfo->gen));
   else if (reg.file != BRW_ARCHITECTURE_REGISTER_FILE)
      assert(reg.nr < 128);

   gen7_convert_mrf_to_grf(p, &reg);

   if (devinfo->gen >= 6 && (brw_inst_opcode(devinfo, inst) == BRW_OPCODE_SEND ||
                             brw_inst_opcode(devinfo, inst) == BRW_OPCODE_SENDC)) {
      /* Any source modifiers or regions will be ignored, since this just
       * identifies the MRF/GRF to start reading the message contents from.
       * Check for some likely failures.
       */
      assert(!reg.negate);
      assert(!reg.abs);
      assert(reg.address_mode == BRW_ADDRESS_DIRECT);
   }

   brw_inst_set_src0_file_type(devinfo, inst, reg.file, reg.type);
   brw_inst_set_src0_abs(devinfo, inst, reg.abs);
   brw_inst_set_src0_negate(devinfo, inst, reg.negate);
   brw_inst_set_src0_address_mode(devinfo, inst, reg.address_mode);

   if (reg.file == BRW_IMMEDIATE_VALUE) {
      if (reg.type == BRW_REGISTER_TYPE_DF ||
          brw_inst_opcode(devinfo, inst) == BRW_OPCODE_DIM)
         brw_inst_set_imm_df(devinfo, inst, reg.df);
      else if (reg.type == BRW_REGISTER_TYPE_UQ ||
               reg.type == BRW_REGISTER_TYPE_Q)
         brw_inst_set_imm_uq(devinfo, inst, reg.u64);
      else
         brw_inst_set_imm_ud(devinfo, inst, reg.ud);

      if (type_sz(reg.type) < 8) {
         brw_inst_set_src1_reg_file(devinfo, inst,
                                    BRW_ARCHITECTURE_REGISTER_FILE);
         brw_inst_set_src1_reg_hw_type(devinfo, inst,
                                       brw_inst_src0_reg_hw_type(devinfo, inst));
      }
   } else {
      if (reg.address_mode == BRW_ADDRESS_DIRECT) {
         brw_inst_set_src0_da_reg_nr(devinfo, inst, reg.nr);
         if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
             brw_inst_set_src0_da1_subreg_nr(devinfo, inst, reg.subnr);
	 } else {
            brw_inst_set_src0_da16_subreg_nr(devinfo, inst, reg.subnr / 16);
	 }
      } else {
         brw_inst_set_src0_ia_subreg_nr(devinfo, inst, reg.subnr);

         if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
            brw_inst_set_src0_ia1_addr_imm(devinfo, inst, reg.indirect_offset);
	 } else {
            brw_inst_set_src0_ia16_addr_imm(devinfo, inst, reg.indirect_offset);
	 }
      }

      if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
	 if (reg.width == BRW_WIDTH_1 &&
             brw_inst_exec_size(devinfo, inst) == BRW_EXECUTE_1) {
            brw_inst_set_src0_hstride(devinfo, inst, BRW_HORIZONTAL_STRIDE_0);
            brw_inst_set_src0_width(devinfo, inst, BRW_WIDTH_1);
            brw_inst_set_src0_vstride(devinfo, inst, BRW_VERTICAL_STRIDE_0);
	 } else {
            brw_inst_set_src0_hstride(devinfo, inst, reg.hstride);
            brw_inst_set_src0_width(devinfo, inst, reg.width);
            brw_inst_set_src0_vstride(devinfo, inst, reg.vstride);
	 }
      } else {
         brw_inst_set_src0_da16_swiz_x(devinfo, inst,
            BRW_GET_SWZ(reg.swizzle, BRW_CHANNEL_X));
         brw_inst_set_src0_da16_swiz_y(devinfo, inst,
            BRW_GET_SWZ(reg.swizzle, BRW_CHANNEL_Y));
         brw_inst_set_src0_da16_swiz_z(devinfo, inst,
            BRW_GET_SWZ(reg.swizzle, BRW_CHANNEL_Z));
         brw_inst_set_src0_da16_swiz_w(devinfo, inst,
            BRW_GET_SWZ(reg.swizzle, BRW_CHANNEL_W));

         if (reg.vstride == BRW_VERTICAL_STRIDE_8) {
            /* This is an oddity of the fact we're using the same
             * descriptions for registers in align_16 as align_1:
             */
            brw_inst_set_src0_vstride(devinfo, inst, BRW_VERTICAL_STRIDE_4);
         } else if (devinfo->gen == 7 && !devinfo->is_haswell &&
                    reg.type == BRW_REGISTER_TYPE_DF &&
                    reg.vstride == BRW_VERTICAL_STRIDE_2) {
            /* From SNB PRM:
             *
             * "For Align16 access mode, only encodings of 0000 and 0011
             *  are allowed. Other codes are reserved."
             *
             * Presumably the DevSNB behavior applies to IVB as well.
             */
            brw_inst_set_src0_vstride(devinfo, inst, BRW_VERTICAL_STRIDE_4);
         } else {
            brw_inst_set_src0_vstride(devinfo, inst, reg.vstride);
         }
      }
   }
}


void
brw_set_src1(struct brw_codegen *p, brw_inst *inst, struct brw_reg reg)
{
   const struct gen_device_info *devinfo = p->devinfo;

   if (reg.file != BRW_ARCHITECTURE_REGISTER_FILE)
      assert(reg.nr < 128);

   /* From the IVB PRM Vol. 4, Pt. 3, Section 3.3.3.5:
    *
    *    "Accumulator registers may be accessed explicitly as src0
    *    operands only."
    */
   assert(reg.file != BRW_ARCHITECTURE_REGISTER_FILE ||
          reg.nr != BRW_ARF_ACCUMULATOR);

   gen7_convert_mrf_to_grf(p, &reg);
   assert(reg.file != BRW_MESSAGE_REGISTER_FILE);

   brw_inst_set_src1_file_type(devinfo, inst, reg.file, reg.type);
   brw_inst_set_src1_abs(devinfo, inst, reg.abs);
   brw_inst_set_src1_negate(devinfo, inst, reg.negate);

   /* Only src1 can be immediate in two-argument instructions.
    */
   assert(brw_inst_src0_reg_file(devinfo, inst) != BRW_IMMEDIATE_VALUE);

   if (reg.file == BRW_IMMEDIATE_VALUE) {
      /* two-argument instructions can only use 32-bit immediates */
      assert(type_sz(reg.type) < 8);
      brw_inst_set_imm_ud(devinfo, inst, reg.ud);
   } else {
      /* This is a hardware restriction, which may or may not be lifted
       * in the future:
       */
      assert (reg.address_mode == BRW_ADDRESS_DIRECT);
      /* assert (reg.file == BRW_GENERAL_REGISTER_FILE); */

      brw_inst_set_src1_da_reg_nr(devinfo, inst, reg.nr);
      if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
         brw_inst_set_src1_da1_subreg_nr(devinfo, inst, reg.subnr);
      } else {
         brw_inst_set_src1_da16_subreg_nr(devinfo, inst, reg.subnr / 16);
      }

      if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
	 if (reg.width == BRW_WIDTH_1 &&
             brw_inst_exec_size(devinfo, inst) == BRW_EXECUTE_1) {
            brw_inst_set_src1_hstride(devinfo, inst, BRW_HORIZONTAL_STRIDE_0);
            brw_inst_set_src1_width(devinfo, inst, BRW_WIDTH_1);
            brw_inst_set_src1_vstride(devinfo, inst, BRW_VERTICAL_STRIDE_0);
	 } else {
            brw_inst_set_src1_hstride(devinfo, inst, reg.hstride);
            brw_inst_set_src1_width(devinfo, inst, reg.width);
            brw_inst_set_src1_vstride(devinfo, inst, reg.vstride);
	 }
      } else {
         brw_inst_set_src1_da16_swiz_x(devinfo, inst,
            BRW_GET_SWZ(reg.swizzle, BRW_CHANNEL_X));
         brw_inst_set_src1_da16_swiz_y(devinfo, inst,
            BRW_GET_SWZ(reg.swizzle, BRW_CHANNEL_Y));
         brw_inst_set_src1_da16_swiz_z(devinfo, inst,
            BRW_GET_SWZ(reg.swizzle, BRW_CHANNEL_Z));
         brw_inst_set_src1_da16_swiz_w(devinfo, inst,
            BRW_GET_SWZ(reg.swizzle, BRW_CHANNEL_W));

         if (reg.vstride == BRW_VERTICAL_STRIDE_8) {
            /* This is an oddity of the fact we're using the same
             * descriptions for registers in align_16 as align_1:
             */
            brw_inst_set_src1_vstride(devinfo, inst, BRW_VERTICAL_STRIDE_4);
         } else if (devinfo->gen == 7 && !devinfo->is_haswell &&
                    reg.type == BRW_REGISTER_TYPE_DF &&
                    reg.vstride == BRW_VERTICAL_STRIDE_2) {
            /* From SNB PRM:
             *
             * "For Align16 access mode, only encodings of 0000 and 0011
             *  are allowed. Other codes are reserved."
             *
             * Presumably the DevSNB behavior applies to IVB as well.
             */
            brw_inst_set_src1_vstride(devinfo, inst, BRW_VERTICAL_STRIDE_4);
         } else {
            brw_inst_set_src1_vstride(devinfo, inst, reg.vstride);
         }
      }
   }
}

/**
 * Specify the descriptor and extended descriptor immediate for a SEND(C)
 * message instruction.
 */
void
brw_set_desc_ex(struct brw_codegen *p, brw_inst *inst,
                unsigned desc, unsigned ex_desc)
{
   const struct gen_device_info *devinfo = p->devinfo;
   assert(brw_inst_opcode(devinfo, inst) == BRW_OPCODE_SEND ||
          brw_inst_opcode(devinfo, inst) == BRW_OPCODE_SENDC);
   brw_inst_set_src1_file_type(devinfo, inst,
                               BRW_IMMEDIATE_VALUE, BRW_REGISTER_TYPE_UD);
   brw_inst_set_send_desc(devinfo, inst, desc);
   if (devinfo->gen >= 9)
      brw_inst_set_send_ex_desc(devinfo, inst, ex_desc);
}

static void brw_set_math_message( struct brw_codegen *p,
				  brw_inst *inst,
				  unsigned function,
				  unsigned integer_type,
				  bool low_precision,
				  unsigned dataType )
{
   const struct gen_device_info *devinfo = p->devinfo;
   unsigned msg_length;
   unsigned response_length;

   /* Infer message length from the function */
   switch (function) {
   case BRW_MATH_FUNCTION_POW:
   case BRW_MATH_FUNCTION_INT_DIV_QUOTIENT:
   case BRW_MATH_FUNCTION_INT_DIV_REMAINDER:
   case BRW_MATH_FUNCTION_INT_DIV_QUOTIENT_AND_REMAINDER:
      msg_length = 2;
      break;
   default:
      msg_length = 1;
      break;
   }

   /* Infer response length from the function */
   switch (function) {
   case BRW_MATH_FUNCTION_SINCOS:
   case BRW_MATH_FUNCTION_INT_DIV_QUOTIENT_AND_REMAINDER:
      response_length = 2;
      break;
   default:
      response_length = 1;
      break;
   }

   brw_set_desc(p, inst, brw_message_desc(
                   devinfo, msg_length, response_length, false));

   brw_inst_set_sfid(devinfo, inst, BRW_SFID_MATH);
   brw_inst_set_math_msg_function(devinfo, inst, function);
   brw_inst_set_math_msg_signed_int(devinfo, inst, integer_type);
   brw_inst_set_math_msg_precision(devinfo, inst, low_precision);
   brw_inst_set_math_msg_saturate(devinfo, inst, brw_inst_saturate(devinfo, inst));
   brw_inst_set_math_msg_data_type(devinfo, inst, dataType);
   brw_inst_set_saturate(devinfo, inst, 0);
}


static void brw_set_ff_sync_message(struct brw_codegen *p,
				    brw_inst *insn,
				    bool allocate,
				    unsigned response_length,
				    bool end_of_thread)
{
   const struct gen_device_info *devinfo = p->devinfo;

   brw_set_desc(p, insn, brw_message_desc(
                   devinfo, 1, response_length, true));

   brw_inst_set_sfid(devinfo, insn, BRW_SFID_URB);
   brw_inst_set_eot(devinfo, insn, end_of_thread);
   brw_inst_set_urb_opcode(devinfo, insn, 1); /* FF_SYNC */
   brw_inst_set_urb_allocate(devinfo, insn, allocate);
   /* The following fields are not used by FF_SYNC: */
   brw_inst_set_urb_global_offset(devinfo, insn, 0);
   brw_inst_set_urb_swizzle_control(devinfo, insn, 0);
   brw_inst_set_urb_used(devinfo, insn, 0);
   brw_inst_set_urb_complete(devinfo, insn, 0);
}

static void brw_set_urb_message( struct brw_codegen *p,
				 brw_inst *insn,
                                 enum brw_urb_write_flags flags,
				 unsigned msg_length,
				 unsigned response_length,
				 unsigned offset,
				 unsigned swizzle_control )
{
   const struct gen_device_info *devinfo = p->devinfo;

   assert(devinfo->gen < 7 || swizzle_control != BRW_URB_SWIZZLE_TRANSPOSE);
   assert(devinfo->gen < 7 || !(flags & BRW_URB_WRITE_ALLOCATE));
   assert(devinfo->gen >= 7 || !(flags & BRW_URB_WRITE_PER_SLOT_OFFSET));

   brw_set_desc(p, insn, brw_message_desc(
                   devinfo, msg_length, response_length, true));

   brw_inst_set_sfid(devinfo, insn, BRW_SFID_URB);
   brw_inst_set_eot(devinfo, insn, !!(flags & BRW_URB_WRITE_EOT));

   if (flags & BRW_URB_WRITE_OWORD) {
      assert(msg_length == 2); /* header + one OWORD of data */
      brw_inst_set_urb_opcode(devinfo, insn, BRW_URB_OPCODE_WRITE_OWORD);
   } else {
      brw_inst_set_urb_opcode(devinfo, insn, BRW_URB_OPCODE_WRITE_HWORD);
   }

   brw_inst_set_urb_global_offset(devinfo, insn, offset);
   brw_inst_set_urb_swizzle_control(devinfo, insn, swizzle_control);

   if (devinfo->gen < 8) {
      brw_inst_set_urb_complete(devinfo, insn, !!(flags & BRW_URB_WRITE_COMPLETE));
   }

   if (devinfo->gen < 7) {
      brw_inst_set_urb_allocate(devinfo, insn, !!(flags & BRW_URB_WRITE_ALLOCATE));
      brw_inst_set_urb_used(devinfo, insn, !(flags & BRW_URB_WRITE_UNUSED));
   } else {
      brw_inst_set_urb_per_slot_offset(devinfo, insn,
         !!(flags & BRW_URB_WRITE_PER_SLOT_OFFSET));
   }
}

static void
gen7_set_dp_scratch_message(struct brw_codegen *p,
                            brw_inst *inst,
                            bool write,
                            bool dword,
                            bool invalidate_after_read,
                            unsigned num_regs,
                            unsigned addr_offset,
                            unsigned mlen,
                            unsigned rlen,
                            bool header_present)
{
   const struct gen_device_info *devinfo = p->devinfo;
   assert(num_regs == 1 || num_regs == 2 || num_regs == 4 ||
          (devinfo->gen >= 8 && num_regs == 8));
   const unsigned block_size = (devinfo->gen >= 8 ? _mesa_logbase2(num_regs) :
                                num_regs - 1);

   brw_set_desc(p, inst, brw_message_desc(
                   devinfo, mlen, rlen, header_present));

   brw_inst_set_sfid(devinfo, inst, GEN7_SFID_DATAPORT_DATA_CACHE);
   brw_inst_set_dp_category(devinfo, inst, 1); /* Scratch Block Read/Write msgs */
   brw_inst_set_scratch_read_write(devinfo, inst, write);
   brw_inst_set_scratch_type(devinfo, inst, dword);
   brw_inst_set_scratch_invalidate_after_read(devinfo, inst, invalidate_after_read);
   brw_inst_set_scratch_block_size(devinfo, inst, block_size);
   brw_inst_set_scratch_addr_offset(devinfo, inst, addr_offset);
}

static void
brw_inst_set_state(const struct gen_device_info *devinfo,
                   brw_inst *insn,
                   const struct brw_insn_state *state)
{
   brw_inst_set_exec_size(devinfo, insn, state->exec_size);
   brw_inst_set_group(devinfo, insn, state->group);
   brw_inst_set_compression(devinfo, insn, state->compressed);
   brw_inst_set_access_mode(devinfo, insn, state->access_mode);
   brw_inst_set_mask_control(devinfo, insn, state->mask_control);
   brw_inst_set_saturate(devinfo, insn, state->saturate);
   brw_inst_set_pred_control(devinfo, insn, state->predicate);
   brw_inst_set_pred_inv(devinfo, insn, state->pred_inv);

   if (is_3src(devinfo, brw_inst_opcode(devinfo, insn)) &&
       state->access_mode == BRW_ALIGN_16) {
      brw_inst_set_3src_a16_flag_subreg_nr(devinfo, insn, state->flag_subreg % 2);
      if (devinfo->gen >= 7)
         brw_inst_set_3src_a16_flag_reg_nr(devinfo, insn, state->flag_subreg / 2);
   } else {
      brw_inst_set_flag_subreg_nr(devinfo, insn, state->flag_subreg % 2);
      if (devinfo->gen >= 7)
         brw_inst_set_flag_reg_nr(devinfo, insn, state->flag_subreg / 2);
   }

   if (devinfo->gen >= 6)
      brw_inst_set_acc_wr_control(devinfo, insn, state->acc_wr_control);
}

#define next_insn brw_next_insn
brw_inst *
brw_next_insn(struct brw_codegen *p, unsigned opcode)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn;

   if (p->nr_insn + 1 > p->store_size) {
      p->store_size <<= 1;
      p->store = reralloc(p->mem_ctx, p->store, brw_inst, p->store_size);
   }

   p->next_insn_offset += 16;
   insn = &p->store[p->nr_insn++];

   memset(insn, 0, sizeof(*insn));
   brw_inst_set_opcode(devinfo, insn, opcode);

   /* Apply the default instruction state */
   brw_inst_set_state(devinfo, insn, p->current);

   return insn;
}

static brw_inst *
brw_alu1(struct brw_codegen *p, unsigned opcode,
         struct brw_reg dest, struct brw_reg src)
{
   brw_inst *insn = next_insn(p, opcode);
   brw_set_dest(p, insn, dest);
   brw_set_src0(p, insn, src);
   return insn;
}

static brw_inst *
brw_alu2(struct brw_codegen *p, unsigned opcode,
         struct brw_reg dest, struct brw_reg src0, struct brw_reg src1)
{
   /* 64-bit immediates are only supported on 1-src instructions */
   assert(src0.file != BRW_IMMEDIATE_VALUE || type_sz(src0.type) <= 4);
   assert(src1.file != BRW_IMMEDIATE_VALUE || type_sz(src1.type) <= 4);

   brw_inst *insn = next_insn(p, opcode);
   brw_set_dest(p, insn, dest);
   brw_set_src0(p, insn, src0);
   brw_set_src1(p, insn, src1);
   return insn;
}

static int
get_3src_subreg_nr(struct brw_reg reg)
{
   /* Normally, SubRegNum is in bytes (0..31).  However, 3-src instructions
    * use 32-bit units (components 0..7).  Since they only support F/D/UD
    * types, this doesn't lose any flexibility, but uses fewer bits.
    */
   return reg.subnr / 4;
}

static enum gen10_align1_3src_vertical_stride
to_3src_align1_vstride(enum brw_vertical_stride vstride)
{
   switch (vstride) {
   case BRW_VERTICAL_STRIDE_0:
      return BRW_ALIGN1_3SRC_VERTICAL_STRIDE_0;
   case BRW_VERTICAL_STRIDE_2:
      return BRW_ALIGN1_3SRC_VERTICAL_STRIDE_2;
   case BRW_VERTICAL_STRIDE_4:
      return BRW_ALIGN1_3SRC_VERTICAL_STRIDE_4;
   case BRW_VERTICAL_STRIDE_8:
   case BRW_VERTICAL_STRIDE_16:
      return BRW_ALIGN1_3SRC_VERTICAL_STRIDE_8;
   default:
      unreachable("invalid vstride");
   }
}


static enum gen10_align1_3src_src_horizontal_stride
to_3src_align1_hstride(enum brw_horizontal_stride hstride)
{
   switch (hstride) {
   case BRW_HORIZONTAL_STRIDE_0:
      return BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_0;
   case BRW_HORIZONTAL_STRIDE_1:
      return BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_1;
   case BRW_HORIZONTAL_STRIDE_2:
      return BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_2;
   case BRW_HORIZONTAL_STRIDE_4:
      return BRW_ALIGN1_3SRC_SRC_HORIZONTAL_STRIDE_4;
   default:
      unreachable("invalid hstride");
   }
}

static brw_inst *
brw_alu3(struct brw_codegen *p, unsigned opcode, struct brw_reg dest,
         struct brw_reg src0, struct brw_reg src1, struct brw_reg src2)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *inst = next_insn(p, opcode);

   gen7_convert_mrf_to_grf(p, &dest);

   assert(dest.nr < 128);
   assert(src0.nr < 128);
   assert(src1.nr < 128);
   assert(src2.nr < 128);
   assert(dest.address_mode == BRW_ADDRESS_DIRECT);
   assert(src0.address_mode == BRW_ADDRESS_DIRECT);
   assert(src1.address_mode == BRW_ADDRESS_DIRECT);
   assert(src2.address_mode == BRW_ADDRESS_DIRECT);

   if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
      assert(dest.file == BRW_GENERAL_REGISTER_FILE ||
             dest.file == BRW_ARCHITECTURE_REGISTER_FILE);

      if (dest.file == BRW_ARCHITECTURE_REGISTER_FILE) {
         brw_inst_set_3src_a1_dst_reg_file(devinfo, inst,
                                           BRW_ALIGN1_3SRC_ACCUMULATOR);
         brw_inst_set_3src_dst_reg_nr(devinfo, inst, BRW_ARF_ACCUMULATOR);
      } else {
         brw_inst_set_3src_a1_dst_reg_file(devinfo, inst,
                                           BRW_ALIGN1_3SRC_GENERAL_REGISTER_FILE);
         brw_inst_set_3src_dst_reg_nr(devinfo, inst, dest.nr);
      }
      brw_inst_set_3src_a1_dst_subreg_nr(devinfo, inst, dest.subnr / 8);

      brw_inst_set_3src_a1_dst_hstride(devinfo, inst, BRW_ALIGN1_3SRC_DST_HORIZONTAL_STRIDE_1);

      if (brw_reg_type_is_floating_point(dest.type)) {
         brw_inst_set_3src_a1_exec_type(devinfo, inst,
                                        BRW_ALIGN1_3SRC_EXEC_TYPE_FLOAT);
      } else {
         brw_inst_set_3src_a1_exec_type(devinfo, inst,
                                        BRW_ALIGN1_3SRC_EXEC_TYPE_INT);
      }

      brw_inst_set_3src_a1_dst_type(devinfo, inst, dest.type);
      brw_inst_set_3src_a1_src0_type(devinfo, inst, src0.type);
      brw_inst_set_3src_a1_src1_type(devinfo, inst, src1.type);
      brw_inst_set_3src_a1_src2_type(devinfo, inst, src2.type);

      brw_inst_set_3src_a1_src0_vstride(devinfo, inst,
                                        to_3src_align1_vstride(src0.vstride));
      brw_inst_set_3src_a1_src1_vstride(devinfo, inst,
                                        to_3src_align1_vstride(src1.vstride));
      /* no vstride on src2 */

      brw_inst_set_3src_a1_src0_hstride(devinfo, inst,
                                        to_3src_align1_hstride(src0.hstride));
      brw_inst_set_3src_a1_src1_hstride(devinfo, inst,
                                        to_3src_align1_hstride(src1.hstride));
      brw_inst_set_3src_a1_src2_hstride(devinfo, inst,
                                        to_3src_align1_hstride(src2.hstride));

      brw_inst_set_3src_a1_src0_subreg_nr(devinfo, inst, src0.subnr);
      if (src0.type == BRW_REGISTER_TYPE_NF) {
         brw_inst_set_3src_src0_reg_nr(devinfo, inst, BRW_ARF_ACCUMULATOR);
      } else {
         brw_inst_set_3src_src0_reg_nr(devinfo, inst, src0.nr);
      }
      brw_inst_set_3src_src0_abs(devinfo, inst, src0.abs);
      brw_inst_set_3src_src0_negate(devinfo, inst, src0.negate);

      brw_inst_set_3src_a1_src1_subreg_nr(devinfo, inst, src1.subnr);
      if (src1.file == BRW_ARCHITECTURE_REGISTER_FILE) {
         brw_inst_set_3src_src1_reg_nr(devinfo, inst, BRW_ARF_ACCUMULATOR);
      } else {
         brw_inst_set_3src_src1_reg_nr(devinfo, inst, src1.nr);
      }
      brw_inst_set_3src_src1_abs(devinfo, inst, src1.abs);
      brw_inst_set_3src_src1_negate(devinfo, inst, src1.negate);

      brw_inst_set_3src_a1_src2_subreg_nr(devinfo, inst, src2.subnr);
      brw_inst_set_3src_src2_reg_nr(devinfo, inst, src2.nr);
      brw_inst_set_3src_src2_abs(devinfo, inst, src2.abs);
      brw_inst_set_3src_src2_negate(devinfo, inst, src2.negate);

      assert(src0.file == BRW_GENERAL_REGISTER_FILE ||
             src0.file == BRW_IMMEDIATE_VALUE ||
             (src0.file == BRW_ARCHITECTURE_REGISTER_FILE &&
              src0.type == BRW_REGISTER_TYPE_NF));
      assert(src1.file == BRW_GENERAL_REGISTER_FILE ||
             src1.file == BRW_ARCHITECTURE_REGISTER_FILE);
      assert(src2.file == BRW_GENERAL_REGISTER_FILE ||
             src2.file == BRW_IMMEDIATE_VALUE);

      brw_inst_set_3src_a1_src0_reg_file(devinfo, inst,
                                         src0.file == BRW_GENERAL_REGISTER_FILE ?
                                         BRW_ALIGN1_3SRC_GENERAL_REGISTER_FILE :
                                         BRW_ALIGN1_3SRC_IMMEDIATE_VALUE);
      brw_inst_set_3src_a1_src1_reg_file(devinfo, inst,
                                         src1.file == BRW_GENERAL_REGISTER_FILE ?
                                         BRW_ALIGN1_3SRC_GENERAL_REGISTER_FILE :
                                         BRW_ALIGN1_3SRC_ACCUMULATOR);
      brw_inst_set_3src_a1_src2_reg_file(devinfo, inst,
                                         src2.file == BRW_GENERAL_REGISTER_FILE ?
                                         BRW_ALIGN1_3SRC_GENERAL_REGISTER_FILE :
                                         BRW_ALIGN1_3SRC_IMMEDIATE_VALUE);
   } else {
      assert(dest.file == BRW_GENERAL_REGISTER_FILE ||
             dest.file == BRW_MESSAGE_REGISTER_FILE);
      assert(dest.type == BRW_REGISTER_TYPE_F  ||
             dest.type == BRW_REGISTER_TYPE_DF ||
             dest.type == BRW_REGISTER_TYPE_D  ||
             dest.type == BRW_REGISTER_TYPE_UD);
      if (devinfo->gen == 6) {
         brw_inst_set_3src_a16_dst_reg_file(devinfo, inst,
                                            dest.file == BRW_MESSAGE_REGISTER_FILE);
      }
      brw_inst_set_3src_dst_reg_nr(devinfo, inst, dest.nr);
      brw_inst_set_3src_a16_dst_subreg_nr(devinfo, inst, dest.subnr / 16);
      brw_inst_set_3src_a16_dst_writemask(devinfo, inst, dest.writemask);

      assert(src0.file == BRW_GENERAL_REGISTER_FILE);
      brw_inst_set_3src_a16_src0_swizzle(devinfo, inst, src0.swizzle);
      brw_inst_set_3src_a16_src0_subreg_nr(devinfo, inst, get_3src_subreg_nr(src0));
      brw_inst_set_3src_src0_reg_nr(devinfo, inst, src0.nr);
      brw_inst_set_3src_src0_abs(devinfo, inst, src0.abs);
      brw_inst_set_3src_src0_negate(devinfo, inst, src0.negate);
      brw_inst_set_3src_a16_src0_rep_ctrl(devinfo, inst,
                                          src0.vstride == BRW_VERTICAL_STRIDE_0);

      assert(src1.file == BRW_GENERAL_REGISTER_FILE);
      brw_inst_set_3src_a16_src1_swizzle(devinfo, inst, src1.swizzle);
      brw_inst_set_3src_a16_src1_subreg_nr(devinfo, inst, get_3src_subreg_nr(src1));
      brw_inst_set_3src_src1_reg_nr(devinfo, inst, src1.nr);
      brw_inst_set_3src_src1_abs(devinfo, inst, src1.abs);
      brw_inst_set_3src_src1_negate(devinfo, inst, src1.negate);
      brw_inst_set_3src_a16_src1_rep_ctrl(devinfo, inst,
                                          src1.vstride == BRW_VERTICAL_STRIDE_0);

      assert(src2.file == BRW_GENERAL_REGISTER_FILE);
      brw_inst_set_3src_a16_src2_swizzle(devinfo, inst, src2.swizzle);
      brw_inst_set_3src_a16_src2_subreg_nr(devinfo, inst, get_3src_subreg_nr(src2));
      brw_inst_set_3src_src2_reg_nr(devinfo, inst, src2.nr);
      brw_inst_set_3src_src2_abs(devinfo, inst, src2.abs);
      brw_inst_set_3src_src2_negate(devinfo, inst, src2.negate);
      brw_inst_set_3src_a16_src2_rep_ctrl(devinfo, inst,
                                          src2.vstride == BRW_VERTICAL_STRIDE_0);

      if (devinfo->gen >= 7) {
         /* Set both the source and destination types based on dest.type,
          * ignoring the source register types.  The MAD and LRP emitters ensure
          * that all four types are float.  The BFE and BFI2 emitters, however,
          * may send us mixed D and UD types and want us to ignore that and use
          * the destination type.
          */
         brw_inst_set_3src_a16_src_type(devinfo, inst, dest.type);
         brw_inst_set_3src_a16_dst_type(devinfo, inst, dest.type);
      }
   }

   return inst;
}


/***********************************************************************
 * Convenience routines.
 */
#define ALU1(OP)					\
brw_inst *brw_##OP(struct brw_codegen *p,		\
	      struct brw_reg dest,			\
	      struct brw_reg src0)   			\
{							\
   return brw_alu1(p, BRW_OPCODE_##OP, dest, src0);    	\
}

#define ALU2(OP)					\
brw_inst *brw_##OP(struct brw_codegen *p,		\
	      struct brw_reg dest,			\
	      struct brw_reg src0,			\
	      struct brw_reg src1)   			\
{							\
   return brw_alu2(p, BRW_OPCODE_##OP, dest, src0, src1);	\
}

#define ALU3(OP)					\
brw_inst *brw_##OP(struct brw_codegen *p,		\
	      struct brw_reg dest,			\
	      struct brw_reg src0,			\
	      struct brw_reg src1,			\
	      struct brw_reg src2)   			\
{							\
   return brw_alu3(p, BRW_OPCODE_##OP, dest, src0, src1, src2);	\
}

#define ALU3F(OP)                                               \
brw_inst *brw_##OP(struct brw_codegen *p,         \
                                 struct brw_reg dest,           \
                                 struct brw_reg src0,           \
                                 struct brw_reg src1,           \
                                 struct brw_reg src2)           \
{                                                               \
   assert(dest.type == BRW_REGISTER_TYPE_F ||                   \
          dest.type == BRW_REGISTER_TYPE_DF);                   \
   if (dest.type == BRW_REGISTER_TYPE_F) {                      \
      assert(src0.type == BRW_REGISTER_TYPE_F);                 \
      assert(src1.type == BRW_REGISTER_TYPE_F);                 \
      assert(src2.type == BRW_REGISTER_TYPE_F);                 \
   } else if (dest.type == BRW_REGISTER_TYPE_DF) {              \
      assert(src0.type == BRW_REGISTER_TYPE_DF);                \
      assert(src1.type == BRW_REGISTER_TYPE_DF);                \
      assert(src2.type == BRW_REGISTER_TYPE_DF);                \
   }                                                            \
   return brw_alu3(p, BRW_OPCODE_##OP, dest, src0, src1, src2); \
}

/* Rounding operations (other than RNDD) require two instructions - the first
 * stores a rounded value (possibly the wrong way) in the dest register, but
 * also sets a per-channel "increment bit" in the flag register.  A predicated
 * add of 1.0 fixes dest to contain the desired result.
 *
 * Sandybridge and later appear to round correctly without an ADD.
 */
#define ROUND(OP)							      \
void brw_##OP(struct brw_codegen *p,					      \
	      struct brw_reg dest,					      \
	      struct brw_reg src)					      \
{									      \
   const struct gen_device_info *devinfo = p->devinfo;					      \
   brw_inst *rnd, *add;							      \
   rnd = next_insn(p, BRW_OPCODE_##OP);					      \
   brw_set_dest(p, rnd, dest);						      \
   brw_set_src0(p, rnd, src);						      \
									      \
   if (devinfo->gen < 6) {							      \
      /* turn on round-increments */					      \
      brw_inst_set_cond_modifier(devinfo, rnd, BRW_CONDITIONAL_R);            \
      add = brw_ADD(p, dest, dest, brw_imm_f(1.0f));			      \
      brw_inst_set_pred_control(devinfo, add, BRW_PREDICATE_NORMAL);          \
   }									      \
}


ALU2(SEL)
ALU1(NOT)
ALU2(AND)
ALU2(OR)
ALU2(XOR)
ALU2(SHR)
ALU2(SHL)
ALU1(DIM)
ALU2(ASR)
ALU3(CSEL)
ALU1(FRC)
ALU1(RNDD)
ALU2(MAC)
ALU2(MACH)
ALU1(LZD)
ALU2(DP4)
ALU2(DPH)
ALU2(DP3)
ALU2(DP2)
ALU3(MAD)
ALU3F(LRP)
ALU1(BFREV)
ALU3(BFE)
ALU2(BFI1)
ALU3(BFI2)
ALU1(FBH)
ALU1(FBL)
ALU1(CBIT)
ALU2(ADDC)
ALU2(SUBB)

ROUND(RNDZ)
ROUND(RNDE)

brw_inst *
brw_MOV(struct brw_codegen *p, struct brw_reg dest, struct brw_reg src0)
{
   const struct gen_device_info *devinfo = p->devinfo;

   /* When converting F->DF on IVB/BYT, every odd source channel is ignored.
    * To avoid the problems that causes, we use an <X,2,0> source region to
    * read each element twice.
    */
   if (devinfo->gen == 7 && !devinfo->is_haswell &&
       brw_get_default_access_mode(p) == BRW_ALIGN_1 &&
       dest.type == BRW_REGISTER_TYPE_DF &&
       (src0.type == BRW_REGISTER_TYPE_F ||
        src0.type == BRW_REGISTER_TYPE_D ||
        src0.type == BRW_REGISTER_TYPE_UD) &&
       !has_scalar_region(src0)) {
      assert(src0.vstride == src0.width + src0.hstride);
      src0.vstride = src0.hstride;
      src0.width = BRW_WIDTH_2;
      src0.hstride = BRW_HORIZONTAL_STRIDE_0;
   }

   return brw_alu1(p, BRW_OPCODE_MOV, dest, src0);
}

brw_inst *
brw_ADD(struct brw_codegen *p, struct brw_reg dest,
        struct brw_reg src0, struct brw_reg src1)
{
   /* 6.2.2: add */
   if (src0.type == BRW_REGISTER_TYPE_F ||
       (src0.file == BRW_IMMEDIATE_VALUE &&
	src0.type == BRW_REGISTER_TYPE_VF)) {
      assert(src1.type != BRW_REGISTER_TYPE_UD);
      assert(src1.type != BRW_REGISTER_TYPE_D);
   }

   if (src1.type == BRW_REGISTER_TYPE_F ||
       (src1.file == BRW_IMMEDIATE_VALUE &&
	src1.type == BRW_REGISTER_TYPE_VF)) {
      assert(src0.type != BRW_REGISTER_TYPE_UD);
      assert(src0.type != BRW_REGISTER_TYPE_D);
   }

   return brw_alu2(p, BRW_OPCODE_ADD, dest, src0, src1);
}

brw_inst *
brw_AVG(struct brw_codegen *p, struct brw_reg dest,
        struct brw_reg src0, struct brw_reg src1)
{
   assert(dest.type == src0.type);
   assert(src0.type == src1.type);
   switch (src0.type) {
   case BRW_REGISTER_TYPE_B:
   case BRW_REGISTER_TYPE_UB:
   case BRW_REGISTER_TYPE_W:
   case BRW_REGISTER_TYPE_UW:
   case BRW_REGISTER_TYPE_D:
   case BRW_REGISTER_TYPE_UD:
      break;
   default:
      unreachable("Bad type for brw_AVG");
   }

   return brw_alu2(p, BRW_OPCODE_AVG, dest, src0, src1);
}

brw_inst *
brw_MUL(struct brw_codegen *p, struct brw_reg dest,
        struct brw_reg src0, struct brw_reg src1)
{
   /* 6.32.38: mul */
   if (src0.type == BRW_REGISTER_TYPE_D ||
       src0.type == BRW_REGISTER_TYPE_UD ||
       src1.type == BRW_REGISTER_TYPE_D ||
       src1.type == BRW_REGISTER_TYPE_UD) {
      assert(dest.type != BRW_REGISTER_TYPE_F);
   }

   if (src0.type == BRW_REGISTER_TYPE_F ||
       (src0.file == BRW_IMMEDIATE_VALUE &&
	src0.type == BRW_REGISTER_TYPE_VF)) {
      assert(src1.type != BRW_REGISTER_TYPE_UD);
      assert(src1.type != BRW_REGISTER_TYPE_D);
   }

   if (src1.type == BRW_REGISTER_TYPE_F ||
       (src1.file == BRW_IMMEDIATE_VALUE &&
	src1.type == BRW_REGISTER_TYPE_VF)) {
      assert(src0.type != BRW_REGISTER_TYPE_UD);
      assert(src0.type != BRW_REGISTER_TYPE_D);
   }

   assert(src0.file != BRW_ARCHITECTURE_REGISTER_FILE ||
	  src0.nr != BRW_ARF_ACCUMULATOR);
   assert(src1.file != BRW_ARCHITECTURE_REGISTER_FILE ||
	  src1.nr != BRW_ARF_ACCUMULATOR);

   return brw_alu2(p, BRW_OPCODE_MUL, dest, src0, src1);
}

brw_inst *
brw_LINE(struct brw_codegen *p, struct brw_reg dest,
         struct brw_reg src0, struct brw_reg src1)
{
   src0.vstride = BRW_VERTICAL_STRIDE_0;
   src0.width = BRW_WIDTH_1;
   src0.hstride = BRW_HORIZONTAL_STRIDE_0;
   return brw_alu2(p, BRW_OPCODE_LINE, dest, src0, src1);
}

brw_inst *
brw_PLN(struct brw_codegen *p, struct brw_reg dest,
        struct brw_reg src0, struct brw_reg src1)
{
   src0.vstride = BRW_VERTICAL_STRIDE_0;
   src0.width = BRW_WIDTH_1;
   src0.hstride = BRW_HORIZONTAL_STRIDE_0;
   src1.vstride = BRW_VERTICAL_STRIDE_8;
   src1.width = BRW_WIDTH_8;
   src1.hstride = BRW_HORIZONTAL_STRIDE_1;
   return brw_alu2(p, BRW_OPCODE_PLN, dest, src0, src1);
}

brw_inst *
brw_F32TO16(struct brw_codegen *p, struct brw_reg dst, struct brw_reg src)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const bool align16 = brw_get_default_access_mode(p) == BRW_ALIGN_16;
   /* The F32TO16 instruction doesn't support 32-bit destination types in
    * Align1 mode, and neither does the Gen8 implementation in terms of a
    * converting MOV.  Gen7 does zero out the high 16 bits in Align16 mode as
    * an undocumented feature.
    */
   const bool needs_zero_fill = (dst.type == BRW_REGISTER_TYPE_UD &&
                                 (!align16 || devinfo->gen >= 8));
   brw_inst *inst;

   if (align16) {
      assert(dst.type == BRW_REGISTER_TYPE_UD);
   } else {
      assert(dst.type == BRW_REGISTER_TYPE_UD ||
             dst.type == BRW_REGISTER_TYPE_W ||
             dst.type == BRW_REGISTER_TYPE_UW ||
             dst.type == BRW_REGISTER_TYPE_HF);
   }

   brw_push_insn_state(p);

   if (needs_zero_fill) {
      brw_set_default_access_mode(p, BRW_ALIGN_1);
      dst = spread(retype(dst, BRW_REGISTER_TYPE_W), 2);
   }

   if (devinfo->gen >= 8) {
      inst = brw_MOV(p, retype(dst, BRW_REGISTER_TYPE_HF), src);
   } else {
      assert(devinfo->gen == 7);
      inst = brw_alu1(p, BRW_OPCODE_F32TO16, dst, src);
   }

   if (needs_zero_fill) {
      brw_inst_set_no_dd_clear(devinfo, inst, true);
      inst = brw_MOV(p, suboffset(dst, 1), brw_imm_w(0));
      brw_inst_set_no_dd_check(devinfo, inst, true);
   }

   brw_pop_insn_state(p);
   return inst;
}

brw_inst *
brw_F16TO32(struct brw_codegen *p, struct brw_reg dst, struct brw_reg src)
{
   const struct gen_device_info *devinfo = p->devinfo;
   bool align16 = brw_get_default_access_mode(p) == BRW_ALIGN_16;

   if (align16) {
      assert(src.type == BRW_REGISTER_TYPE_UD);
   } else {
      /* From the Ivybridge PRM, Vol4, Part3, Section 6.26 f16to32:
       *
       *   Because this instruction does not have a 16-bit floating-point
       *   type, the source data type must be Word (W). The destination type
       *   must be F (Float).
       */
      if (src.type == BRW_REGISTER_TYPE_UD)
         src = spread(retype(src, BRW_REGISTER_TYPE_W), 2);

      assert(src.type == BRW_REGISTER_TYPE_W ||
             src.type == BRW_REGISTER_TYPE_UW ||
             src.type == BRW_REGISTER_TYPE_HF);
   }

   if (devinfo->gen >= 8) {
      return brw_MOV(p, dst, retype(src, BRW_REGISTER_TYPE_HF));
   } else {
      assert(devinfo->gen == 7);
      return brw_alu1(p, BRW_OPCODE_F16TO32, dst, src);
   }
}


void brw_NOP(struct brw_codegen *p)
{
   brw_inst *insn = next_insn(p, BRW_OPCODE_NOP);
   memset(insn, 0, sizeof(*insn));
   brw_inst_set_opcode(p->devinfo, insn, BRW_OPCODE_NOP);
}





/***********************************************************************
 * Comparisons, if/else/endif
 */

brw_inst *
brw_JMPI(struct brw_codegen *p, struct brw_reg index,
         unsigned predicate_control)
{
   const struct gen_device_info *devinfo = p->devinfo;
   struct brw_reg ip = brw_ip_reg();
   brw_inst *inst = brw_alu2(p, BRW_OPCODE_JMPI, ip, ip, index);

   brw_inst_set_exec_size(devinfo, inst, BRW_EXECUTE_1);
   brw_inst_set_qtr_control(devinfo, inst, BRW_COMPRESSION_NONE);
   brw_inst_set_mask_control(devinfo, inst, BRW_MASK_DISABLE);
   brw_inst_set_pred_control(devinfo, inst, predicate_control);

   return inst;
}

static void
push_if_stack(struct brw_codegen *p, brw_inst *inst)
{
   p->if_stack[p->if_stack_depth] = inst - p->store;

   p->if_stack_depth++;
   if (p->if_stack_array_size <= p->if_stack_depth) {
      p->if_stack_array_size *= 2;
      p->if_stack = reralloc(p->mem_ctx, p->if_stack, int,
			     p->if_stack_array_size);
   }
}

static brw_inst *
pop_if_stack(struct brw_codegen *p)
{
   p->if_stack_depth--;
   return &p->store[p->if_stack[p->if_stack_depth]];
}

static void
push_loop_stack(struct brw_codegen *p, brw_inst *inst)
{
   if (p->loop_stack_array_size <= (p->loop_stack_depth + 1)) {
      p->loop_stack_array_size *= 2;
      p->loop_stack = reralloc(p->mem_ctx, p->loop_stack, int,
			       p->loop_stack_array_size);
      p->if_depth_in_loop = reralloc(p->mem_ctx, p->if_depth_in_loop, int,
				     p->loop_stack_array_size);
   }

   p->loop_stack[p->loop_stack_depth] = inst - p->store;
   p->loop_stack_depth++;
   p->if_depth_in_loop[p->loop_stack_depth] = 0;
}

static brw_inst *
get_inner_do_insn(struct brw_codegen *p)
{
   return &p->store[p->loop_stack[p->loop_stack_depth - 1]];
}

/* EU takes the value from the flag register and pushes it onto some
 * sort of a stack (presumably merging with any flag value already on
 * the stack).  Within an if block, the flags at the top of the stack
 * control execution on each channel of the unit, eg. on each of the
 * 16 pixel values in our wm programs.
 *
 * When the matching 'else' instruction is reached (presumably by
 * countdown of the instruction count patched in by our ELSE/ENDIF
 * functions), the relevant flags are inverted.
 *
 * When the matching 'endif' instruction is reached, the flags are
 * popped off.  If the stack is now empty, normal execution resumes.
 */
brw_inst *
brw_IF(struct brw_codegen *p, unsigned execute_size)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn;

   insn = next_insn(p, BRW_OPCODE_IF);

   /* Override the defaults for this instruction:
    */
   if (devinfo->gen < 6) {
      brw_set_dest(p, insn, brw_ip_reg());
      brw_set_src0(p, insn, brw_ip_reg());
      brw_set_src1(p, insn, brw_imm_d(0x0));
   } else if (devinfo->gen == 6) {
      brw_set_dest(p, insn, brw_imm_w(0));
      brw_inst_set_gen6_jump_count(devinfo, insn, 0);
      brw_set_src0(p, insn, vec1(retype(brw_null_reg(), BRW_REGISTER_TYPE_D)));
      brw_set_src1(p, insn, vec1(retype(brw_null_reg(), BRW_REGISTER_TYPE_D)));
   } else if (devinfo->gen == 7) {
      brw_set_dest(p, insn, vec1(retype(brw_null_reg(), BRW_REGISTER_TYPE_D)));
      brw_set_src0(p, insn, vec1(retype(brw_null_reg(), BRW_REGISTER_TYPE_D)));
      brw_set_src1(p, insn, brw_imm_w(0));
      brw_inst_set_jip(devinfo, insn, 0);
      brw_inst_set_uip(devinfo, insn, 0);
   } else {
      brw_set_dest(p, insn, vec1(retype(brw_null_reg(), BRW_REGISTER_TYPE_D)));
      brw_set_src0(p, insn, brw_imm_d(0));
      brw_inst_set_jip(devinfo, insn, 0);
      brw_inst_set_uip(devinfo, insn, 0);
   }

   brw_inst_set_exec_size(devinfo, insn, execute_size);
   brw_inst_set_qtr_control(devinfo, insn, BRW_COMPRESSION_NONE);
   brw_inst_set_pred_control(devinfo, insn, BRW_PREDICATE_NORMAL);
   brw_inst_set_mask_control(devinfo, insn, BRW_MASK_ENABLE);
   if (!p->single_program_flow && devinfo->gen < 6)
      brw_inst_set_thread_control(devinfo, insn, BRW_THREAD_SWITCH);

   push_if_stack(p, insn);
   p->if_depth_in_loop[p->loop_stack_depth]++;
   return insn;
}

/* This function is only used for gen6-style IF instructions with an
 * embedded comparison (conditional modifier).  It is not used on gen7.
 */
brw_inst *
gen6_IF(struct brw_codegen *p, enum brw_conditional_mod conditional,
	struct brw_reg src0, struct brw_reg src1)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn;

   insn = next_insn(p, BRW_OPCODE_IF);

   brw_set_dest(p, insn, brw_imm_w(0));
   brw_inst_set_exec_size(devinfo, insn, brw_get_default_exec_size(p));
   brw_inst_set_gen6_jump_count(devinfo, insn, 0);
   brw_set_src0(p, insn, src0);
   brw_set_src1(p, insn, src1);

   assert(brw_inst_qtr_control(devinfo, insn) == BRW_COMPRESSION_NONE);
   assert(brw_inst_pred_control(devinfo, insn) == BRW_PREDICATE_NONE);
   brw_inst_set_cond_modifier(devinfo, insn, conditional);

   push_if_stack(p, insn);
   return insn;
}

/**
 * In single-program-flow (SPF) mode, convert IF and ELSE into ADDs.
 */
static void
convert_IF_ELSE_to_ADD(struct brw_codegen *p,
                       brw_inst *if_inst, brw_inst *else_inst)
{
   const struct gen_device_info *devinfo = p->devinfo;

   /* The next instruction (where the ENDIF would be, if it existed) */
   brw_inst *next_inst = &p->store[p->nr_insn];

   assert(p->single_program_flow);
   assert(if_inst != NULL && brw_inst_opcode(devinfo, if_inst) == BRW_OPCODE_IF);
   assert(else_inst == NULL || brw_inst_opcode(devinfo, else_inst) == BRW_OPCODE_ELSE);
   assert(brw_inst_exec_size(devinfo, if_inst) == BRW_EXECUTE_1);

   /* Convert IF to an ADD instruction that moves the instruction pointer
    * to the first instruction of the ELSE block.  If there is no ELSE
    * block, point to where ENDIF would be.  Reverse the predicate.
    *
    * There's no need to execute an ENDIF since we don't need to do any
    * stack operations, and if we're currently executing, we just want to
    * continue normally.
    */
   brw_inst_set_opcode(devinfo, if_inst, BRW_OPCODE_ADD);
   brw_inst_set_pred_inv(devinfo, if_inst, true);

   if (else_inst != NULL) {
      /* Convert ELSE to an ADD instruction that points where the ENDIF
       * would be.
       */
      brw_inst_set_opcode(devinfo, else_inst, BRW_OPCODE_ADD);

      brw_inst_set_imm_ud(devinfo, if_inst, (else_inst - if_inst + 1) * 16);
      brw_inst_set_imm_ud(devinfo, else_inst, (next_inst - else_inst) * 16);
   } else {
      brw_inst_set_imm_ud(devinfo, if_inst, (next_inst - if_inst) * 16);
   }
}

/**
 * Patch IF and ELSE instructions with appropriate jump targets.
 */
static void
patch_IF_ELSE(struct brw_codegen *p,
              brw_inst *if_inst, brw_inst *else_inst, brw_inst *endif_inst)
{
   const struct gen_device_info *devinfo = p->devinfo;

   /* We shouldn't be patching IF and ELSE instructions in single program flow
    * mode when gen < 6, because in single program flow mode on those
    * platforms, we convert flow control instructions to conditional ADDs that
    * operate on IP (see brw_ENDIF).
    *
    * However, on Gen6, writing to IP doesn't work in single program flow mode
    * (see the SandyBridge PRM, Volume 4 part 2, p79: "When SPF is ON, IP may
    * not be updated by non-flow control instructions.").  And on later
    * platforms, there is no significant benefit to converting control flow
    * instructions to conditional ADDs.  So we do patch IF and ELSE
    * instructions in single program flow mode on those platforms.
    */
   if (devinfo->gen < 6)
      assert(!p->single_program_flow);

   assert(if_inst != NULL && brw_inst_opcode(devinfo, if_inst) == BRW_OPCODE_IF);
   assert(endif_inst != NULL);
   assert(else_inst == NULL || brw_inst_opcode(devinfo, else_inst) == BRW_OPCODE_ELSE);

   unsigned br = brw_jump_scale(devinfo);

   assert(brw_inst_opcode(devinfo, endif_inst) == BRW_OPCODE_ENDIF);
   brw_inst_set_exec_size(devinfo, endif_inst, brw_inst_exec_size(devinfo, if_inst));

   if (else_inst == NULL) {
      /* Patch IF -> ENDIF */
      if (devinfo->gen < 6) {
	 /* Turn it into an IFF, which means no mask stack operations for
	  * all-false and jumping past the ENDIF.
	  */
         brw_inst_set_opcode(devinfo, if_inst, BRW_OPCODE_IFF);
         brw_inst_set_gen4_jump_count(devinfo, if_inst,
                                      br * (endif_inst - if_inst + 1));
         brw_inst_set_gen4_pop_count(devinfo, if_inst, 0);
      } else if (devinfo->gen == 6) {
	 /* As of gen6, there is no IFF and IF must point to the ENDIF. */
         brw_inst_set_gen6_jump_count(devinfo, if_inst, br*(endif_inst - if_inst));
      } else {
         brw_inst_set_uip(devinfo, if_inst, br * (endif_inst - if_inst));
         brw_inst_set_jip(devinfo, if_inst, br * (endif_inst - if_inst));
      }
   } else {
      brw_inst_set_exec_size(devinfo, else_inst, brw_inst_exec_size(devinfo, if_inst));

      /* Patch IF -> ELSE */
      if (devinfo->gen < 6) {
         brw_inst_set_gen4_jump_count(devinfo, if_inst,
                                      br * (else_inst - if_inst));
         brw_inst_set_gen4_pop_count(devinfo, if_inst, 0);
      } else if (devinfo->gen == 6) {
         brw_inst_set_gen6_jump_count(devinfo, if_inst,
                                      br * (else_inst - if_inst + 1));
      }

      /* Patch ELSE -> ENDIF */
      if (devinfo->gen < 6) {
	 /* BRW_OPCODE_ELSE pre-gen6 should point just past the
	  * matching ENDIF.
	  */
         brw_inst_set_gen4_jump_count(devinfo, else_inst,
                                      br * (endif_inst - else_inst + 1));
         brw_inst_set_gen4_pop_count(devinfo, else_inst, 1);
      } else if (devinfo->gen == 6) {
	 /* BRW_OPCODE_ELSE on gen6 should point to the matching ENDIF. */
         brw_inst_set_gen6_jump_count(devinfo, else_inst,
                                      br * (endif_inst - else_inst));
      } else {
	 /* The IF instruction's JIP should point just past the ELSE */
         brw_inst_set_jip(devinfo, if_inst, br * (else_inst - if_inst + 1));
	 /* The IF instruction's UIP and ELSE's JIP should point to ENDIF */
         brw_inst_set_uip(devinfo, if_inst, br * (endif_inst - if_inst));
         brw_inst_set_jip(devinfo, else_inst, br * (endif_inst - else_inst));
         if (devinfo->gen >= 8) {
            /* Since we don't set branch_ctrl, the ELSE's JIP and UIP both
             * should point to ENDIF.
             */
            brw_inst_set_uip(devinfo, else_inst, br * (endif_inst - else_inst));
         }
      }
   }
}

void
brw_ELSE(struct brw_codegen *p)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn;

   insn = next_insn(p, BRW_OPCODE_ELSE);

   if (devinfo->gen < 6) {
      brw_set_dest(p, insn, brw_ip_reg());
      brw_set_src0(p, insn, brw_ip_reg());
      brw_set_src1(p, insn, brw_imm_d(0x0));
   } else if (devinfo->gen == 6) {
      brw_set_dest(p, insn, brw_imm_w(0));
      brw_inst_set_gen6_jump_count(devinfo, insn, 0);
      brw_set_src0(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src1(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
   } else if (devinfo->gen == 7) {
      brw_set_dest(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src0(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src1(p, insn, brw_imm_w(0));
      brw_inst_set_jip(devinfo, insn, 0);
      brw_inst_set_uip(devinfo, insn, 0);
   } else {
      brw_set_dest(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src0(p, insn, brw_imm_d(0));
      brw_inst_set_jip(devinfo, insn, 0);
      brw_inst_set_uip(devinfo, insn, 0);
   }

   brw_inst_set_qtr_control(devinfo, insn, BRW_COMPRESSION_NONE);
   brw_inst_set_mask_control(devinfo, insn, BRW_MASK_ENABLE);
   if (!p->single_program_flow && devinfo->gen < 6)
      brw_inst_set_thread_control(devinfo, insn, BRW_THREAD_SWITCH);

   push_if_stack(p, insn);
}

void
brw_ENDIF(struct brw_codegen *p)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn = NULL;
   brw_inst *else_inst = NULL;
   brw_inst *if_inst = NULL;
   brw_inst *tmp;
   bool emit_endif = true;

   /* In single program flow mode, we can express IF and ELSE instructions
    * equivalently as ADD instructions that operate on IP.  On platforms prior
    * to Gen6, flow control instructions cause an implied thread switch, so
    * this is a significant savings.
    *
    * However, on Gen6, writing to IP doesn't work in single program flow mode
    * (see the SandyBridge PRM, Volume 4 part 2, p79: "When SPF is ON, IP may
    * not be updated by non-flow control instructions.").  And on later
    * platforms, there is no significant benefit to converting control flow
    * instructions to conditional ADDs.  So we only do this trick on Gen4 and
    * Gen5.
    */
   if (devinfo->gen < 6 && p->single_program_flow)
      emit_endif = false;

   /*
    * A single next_insn() may change the base address of instruction store
    * memory(p->store), so call it first before referencing the instruction
    * store pointer from an index
    */
   if (emit_endif)
      insn = next_insn(p, BRW_OPCODE_ENDIF);

   /* Pop the IF and (optional) ELSE instructions from the stack */
   p->if_depth_in_loop[p->loop_stack_depth]--;
   tmp = pop_if_stack(p);
   if (brw_inst_opcode(devinfo, tmp) == BRW_OPCODE_ELSE) {
      else_inst = tmp;
      tmp = pop_if_stack(p);
   }
   if_inst = tmp;

   if (!emit_endif) {
      /* ENDIF is useless; don't bother emitting it. */
      convert_IF_ELSE_to_ADD(p, if_inst, else_inst);
      return;
   }

   if (devinfo->gen < 6) {
      brw_set_dest(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src0(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src1(p, insn, brw_imm_d(0x0));
   } else if (devinfo->gen == 6) {
      brw_set_dest(p, insn, brw_imm_w(0));
      brw_set_src0(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src1(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
   } else if (devinfo->gen == 7) {
      brw_set_dest(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src0(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src1(p, insn, brw_imm_w(0));
   } else {
      brw_set_src0(p, insn, brw_imm_d(0));
   }

   brw_inst_set_qtr_control(devinfo, insn, BRW_COMPRESSION_NONE);
   brw_inst_set_mask_control(devinfo, insn, BRW_MASK_ENABLE);
   if (devinfo->gen < 6)
      brw_inst_set_thread_control(devinfo, insn, BRW_THREAD_SWITCH);

   /* Also pop item off the stack in the endif instruction: */
   if (devinfo->gen < 6) {
      brw_inst_set_gen4_jump_count(devinfo, insn, 0);
      brw_inst_set_gen4_pop_count(devinfo, insn, 1);
   } else if (devinfo->gen == 6) {
      brw_inst_set_gen6_jump_count(devinfo, insn, 2);
   } else {
      brw_inst_set_jip(devinfo, insn, 2);
   }
   patch_IF_ELSE(p, if_inst, else_inst, insn);
}

brw_inst *
brw_BREAK(struct brw_codegen *p)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn;

   insn = next_insn(p, BRW_OPCODE_BREAK);
   if (devinfo->gen >= 8) {
      brw_set_dest(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src0(p, insn, brw_imm_d(0x0));
   } else if (devinfo->gen >= 6) {
      brw_set_dest(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src0(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src1(p, insn, brw_imm_d(0x0));
   } else {
      brw_set_dest(p, insn, brw_ip_reg());
      brw_set_src0(p, insn, brw_ip_reg());
      brw_set_src1(p, insn, brw_imm_d(0x0));
      brw_inst_set_gen4_pop_count(devinfo, insn,
                                  p->if_depth_in_loop[p->loop_stack_depth]);
   }
   brw_inst_set_qtr_control(devinfo, insn, BRW_COMPRESSION_NONE);
   brw_inst_set_exec_size(devinfo, insn, brw_get_default_exec_size(p));

   return insn;
}

brw_inst *
brw_CONT(struct brw_codegen *p)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn;

   insn = next_insn(p, BRW_OPCODE_CONTINUE);
   brw_set_dest(p, insn, brw_ip_reg());
   if (devinfo->gen >= 8) {
      brw_set_src0(p, insn, brw_imm_d(0x0));
   } else {
      brw_set_src0(p, insn, brw_ip_reg());
      brw_set_src1(p, insn, brw_imm_d(0x0));
   }

   if (devinfo->gen < 6) {
      brw_inst_set_gen4_pop_count(devinfo, insn,
                                  p->if_depth_in_loop[p->loop_stack_depth]);
   }
   brw_inst_set_qtr_control(devinfo, insn, BRW_COMPRESSION_NONE);
   brw_inst_set_exec_size(devinfo, insn, brw_get_default_exec_size(p));
   return insn;
}

brw_inst *
gen6_HALT(struct brw_codegen *p)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn;

   insn = next_insn(p, BRW_OPCODE_HALT);
   brw_set_dest(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
   if (devinfo->gen >= 8) {
      brw_set_src0(p, insn, brw_imm_d(0x0));
   } else {
      brw_set_src0(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      brw_set_src1(p, insn, brw_imm_d(0x0)); /* UIP and JIP, updated later. */
   }

   brw_inst_set_qtr_control(devinfo, insn, BRW_COMPRESSION_NONE);
   brw_inst_set_exec_size(devinfo, insn, brw_get_default_exec_size(p));
   return insn;
}

/* DO/WHILE loop:
 *
 * The DO/WHILE is just an unterminated loop -- break or continue are
 * used for control within the loop.  We have a few ways they can be
 * done.
 *
 * For uniform control flow, the WHILE is just a jump, so ADD ip, ip,
 * jip and no DO instruction.
 *
 * For non-uniform control flow pre-gen6, there's a DO instruction to
 * push the mask, and a WHILE to jump back, and BREAK to get out and
 * pop the mask.
 *
 * For gen6, there's no more mask stack, so no need for DO.  WHILE
 * just points back to the first instruction of the loop.
 */
brw_inst *
brw_DO(struct brw_codegen *p, unsigned execute_size)
{
   const struct gen_device_info *devinfo = p->devinfo;

   if (devinfo->gen >= 6 || p->single_program_flow) {
      push_loop_stack(p, &p->store[p->nr_insn]);
      return &p->store[p->nr_insn];
   } else {
      brw_inst *insn = next_insn(p, BRW_OPCODE_DO);

      push_loop_stack(p, insn);

      /* Override the defaults for this instruction:
       */
      brw_set_dest(p, insn, brw_null_reg());
      brw_set_src0(p, insn, brw_null_reg());
      brw_set_src1(p, insn, brw_null_reg());

      brw_inst_set_qtr_control(devinfo, insn, BRW_COMPRESSION_NONE);
      brw_inst_set_exec_size(devinfo, insn, execute_size);
      brw_inst_set_pred_control(devinfo, insn, BRW_PREDICATE_NONE);

      return insn;
   }
}

/**
 * For pre-gen6, we patch BREAK/CONT instructions to point at the WHILE
 * instruction here.
 *
 * For gen6+, see brw_set_uip_jip(), which doesn't care so much about the loop
 * nesting, since it can always just point to the end of the block/current loop.
 */
static void
brw_patch_break_cont(struct brw_codegen *p, brw_inst *while_inst)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *do_inst = get_inner_do_insn(p);
   brw_inst *inst;
   unsigned br = brw_jump_scale(devinfo);

   assert(devinfo->gen < 6);

   for (inst = while_inst - 1; inst != do_inst; inst--) {
      /* If the jump count is != 0, that means that this instruction has already
       * been patched because it's part of a loop inside of the one we're
       * patching.
       */
      if (brw_inst_opcode(devinfo, inst) == BRW_OPCODE_BREAK &&
          brw_inst_gen4_jump_count(devinfo, inst) == 0) {
         brw_inst_set_gen4_jump_count(devinfo, inst, br*((while_inst - inst) + 1));
      } else if (brw_inst_opcode(devinfo, inst) == BRW_OPCODE_CONTINUE &&
                 brw_inst_gen4_jump_count(devinfo, inst) == 0) {
         brw_inst_set_gen4_jump_count(devinfo, inst, br * (while_inst - inst));
      }
   }
}

brw_inst *
brw_WHILE(struct brw_codegen *p)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn, *do_insn;
   unsigned br = brw_jump_scale(devinfo);

   if (devinfo->gen >= 6) {
      insn = next_insn(p, BRW_OPCODE_WHILE);
      do_insn = get_inner_do_insn(p);

      if (devinfo->gen >= 8) {
         brw_set_dest(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
         brw_set_src0(p, insn, brw_imm_d(0));
         brw_inst_set_jip(devinfo, insn, br * (do_insn - insn));
      } else if (devinfo->gen == 7) {
         brw_set_dest(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
         brw_set_src0(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
         brw_set_src1(p, insn, brw_imm_w(0));
         brw_inst_set_jip(devinfo, insn, br * (do_insn - insn));
      } else {
         brw_set_dest(p, insn, brw_imm_w(0));
         brw_inst_set_gen6_jump_count(devinfo, insn, br * (do_insn - insn));
         brw_set_src0(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
         brw_set_src1(p, insn, retype(brw_null_reg(), BRW_REGISTER_TYPE_D));
      }

      brw_inst_set_exec_size(devinfo, insn, brw_get_default_exec_size(p));

   } else {
      if (p->single_program_flow) {
	 insn = next_insn(p, BRW_OPCODE_ADD);
         do_insn = get_inner_do_insn(p);

	 brw_set_dest(p, insn, brw_ip_reg());
	 brw_set_src0(p, insn, brw_ip_reg());
	 brw_set_src1(p, insn, brw_imm_d((do_insn - insn) * 16));
         brw_inst_set_exec_size(devinfo, insn, BRW_EXECUTE_1);
      } else {
	 insn = next_insn(p, BRW_OPCODE_WHILE);
         do_insn = get_inner_do_insn(p);

         assert(brw_inst_opcode(devinfo, do_insn) == BRW_OPCODE_DO);

	 brw_set_dest(p, insn, brw_ip_reg());
	 brw_set_src0(p, insn, brw_ip_reg());
	 brw_set_src1(p, insn, brw_imm_d(0));

         brw_inst_set_exec_size(devinfo, insn, brw_inst_exec_size(devinfo, do_insn));
         brw_inst_set_gen4_jump_count(devinfo, insn, br * (do_insn - insn + 1));
         brw_inst_set_gen4_pop_count(devinfo, insn, 0);

	 brw_patch_break_cont(p, insn);
      }
   }
   brw_inst_set_qtr_control(devinfo, insn, BRW_COMPRESSION_NONE);

   p->loop_stack_depth--;

   return insn;
}

/* FORWARD JUMPS:
 */
void brw_land_fwd_jump(struct brw_codegen *p, int jmp_insn_idx)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *jmp_insn = &p->store[jmp_insn_idx];
   unsigned jmpi = 1;

   if (devinfo->gen >= 5)
      jmpi = 2;

   assert(brw_inst_opcode(devinfo, jmp_insn) == BRW_OPCODE_JMPI);
   assert(brw_inst_src1_reg_file(devinfo, jmp_insn) == BRW_IMMEDIATE_VALUE);

   brw_inst_set_gen4_jump_count(devinfo, jmp_insn,
                                jmpi * (p->nr_insn - jmp_insn_idx - 1));
}

/* To integrate with the above, it makes sense that the comparison
 * instruction should populate the flag register.  It might be simpler
 * just to use the flag reg for most WM tasks?
 */
void brw_CMP(struct brw_codegen *p,
	     struct brw_reg dest,
	     unsigned conditional,
	     struct brw_reg src0,
	     struct brw_reg src1)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn = next_insn(p, BRW_OPCODE_CMP);

   brw_inst_set_cond_modifier(devinfo, insn, conditional);
   brw_set_dest(p, insn, dest);
   brw_set_src0(p, insn, src0);
   brw_set_src1(p, insn, src1);

   /* Item WaCMPInstNullDstForcesThreadSwitch in the Haswell Bspec workarounds
    * page says:
    *    "Any CMP instruction with a null destination must use a {switch}."
    *
    * It also applies to other Gen7 platforms (IVB, BYT) even though it isn't
    * mentioned on their work-arounds pages.
    */
   if (devinfo->gen == 7) {
      if (dest.file == BRW_ARCHITECTURE_REGISTER_FILE &&
          dest.nr == BRW_ARF_NULL) {
         brw_inst_set_thread_control(devinfo, insn, BRW_THREAD_SWITCH);
      }
   }
}

/***********************************************************************
 * Helpers for the various SEND message types:
 */

/** Extended math function, float[8].
 */
void gen4_math(struct brw_codegen *p,
	       struct brw_reg dest,
	       unsigned function,
	       unsigned msg_reg_nr,
	       struct brw_reg src,
	       unsigned precision )
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn = next_insn(p, BRW_OPCODE_SEND);
   unsigned data_type;
   if (has_scalar_region(src)) {
      data_type = BRW_MATH_DATA_SCALAR;
   } else {
      data_type = BRW_MATH_DATA_VECTOR;
   }

   assert(devinfo->gen < 6);

   /* Example code doesn't set predicate_control for send
    * instructions.
    */
   brw_inst_set_pred_control(devinfo, insn, 0);
   brw_inst_set_base_mrf(devinfo, insn, msg_reg_nr);

   brw_set_dest(p, insn, dest);
   brw_set_src0(p, insn, src);
   brw_set_math_message(p,
                        insn,
                        function,
                        src.type == BRW_REGISTER_TYPE_D,
                        precision,
                        data_type);
}

void gen6_math(struct brw_codegen *p,
	       struct brw_reg dest,
	       unsigned function,
	       struct brw_reg src0,
	       struct brw_reg src1)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn = next_insn(p, BRW_OPCODE_MATH);

   assert(devinfo->gen >= 6);

   assert(dest.file == BRW_GENERAL_REGISTER_FILE ||
          (devinfo->gen >= 7 && dest.file == BRW_MESSAGE_REGISTER_FILE));

   assert(dest.hstride == BRW_HORIZONTAL_STRIDE_1);
   if (devinfo->gen == 6) {
      assert(src0.hstride == BRW_HORIZONTAL_STRIDE_1);
      assert(src1.hstride == BRW_HORIZONTAL_STRIDE_1);
   }

   if (function == BRW_MATH_FUNCTION_INT_DIV_QUOTIENT ||
       function == BRW_MATH_FUNCTION_INT_DIV_REMAINDER ||
       function == BRW_MATH_FUNCTION_INT_DIV_QUOTIENT_AND_REMAINDER) {
      assert(src0.type != BRW_REGISTER_TYPE_F);
      assert(src1.type != BRW_REGISTER_TYPE_F);
      assert(src1.file == BRW_GENERAL_REGISTER_FILE ||
             (devinfo->gen >= 8 && src1.file == BRW_IMMEDIATE_VALUE));
   } else {
      assert(src0.type == BRW_REGISTER_TYPE_F);
      assert(src1.type == BRW_REGISTER_TYPE_F);
   }

   /* Source modifiers are ignored for extended math instructions on Gen6. */
   if (devinfo->gen == 6) {
      assert(!src0.negate);
      assert(!src0.abs);
      assert(!src1.negate);
      assert(!src1.abs);
   }

   brw_inst_set_math_function(devinfo, insn, function);

   brw_set_dest(p, insn, dest);
   brw_set_src0(p, insn, src0);
   brw_set_src1(p, insn, src1);
}

/**
 * Return the right surface index to access the thread scratch space using
 * stateless dataport messages.
 */
unsigned
brw_scratch_surface_idx(const struct brw_codegen *p)
{
   /* The scratch space is thread-local so IA coherency is unnecessary. */
   if (p->devinfo->gen >= 8)
      return GEN8_BTI_STATELESS_NON_COHERENT;
   else
      return BRW_BTI_STATELESS;
}

/**
 * Write a block of OWORDs (half a GRF each) from the scratch buffer,
 * using a constant offset per channel.
 *
 * The offset must be aligned to oword size (16 bytes).  Used for
 * register spilling.
 */
void brw_oword_block_write_scratch(struct brw_codegen *p,
				   struct brw_reg mrf,
				   int num_regs,
				   unsigned offset)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned target_cache =
      (devinfo->gen >= 7 ? GEN7_SFID_DATAPORT_DATA_CACHE :
       devinfo->gen >= 6 ? GEN6_SFID_DATAPORT_RENDER_CACHE :
       BRW_SFID_DATAPORT_WRITE);
   uint32_t msg_type;

   if (devinfo->gen >= 6)
      offset /= 16;

   mrf = retype(mrf, BRW_REGISTER_TYPE_UD);

   const unsigned mlen = 1 + num_regs;

   /* Set up the message header.  This is g0, with g0.2 filled with
    * the offset.  We don't want to leave our offset around in g0 or
    * it'll screw up texture samples, so set it up inside the message
    * reg.
    */
   {
      brw_push_insn_state(p);
      brw_set_default_exec_size(p, BRW_EXECUTE_8);
      brw_set_default_mask_control(p, BRW_MASK_DISABLE);
      brw_set_default_compression_control(p, BRW_COMPRESSION_NONE);

      brw_MOV(p, mrf, retype(brw_vec8_grf(0, 0), BRW_REGISTER_TYPE_UD));

      /* set message header global offset field (reg 0, element 2) */
      brw_set_default_exec_size(p, BRW_EXECUTE_1);
      brw_MOV(p,
	      retype(brw_vec1_reg(BRW_MESSAGE_REGISTER_FILE,
				  mrf.nr,
				  2), BRW_REGISTER_TYPE_UD),
	      brw_imm_ud(offset));

      brw_pop_insn_state(p);
   }

   {
      struct brw_reg dest;
      brw_inst *insn = next_insn(p, BRW_OPCODE_SEND);
      int send_commit_msg;
      struct brw_reg src_header = retype(brw_vec8_grf(0, 0),
					 BRW_REGISTER_TYPE_UW);

      brw_inst_set_sfid(devinfo, insn, target_cache);
      brw_inst_set_compression(devinfo, insn, false);

      if (brw_inst_exec_size(devinfo, insn) >= 16)
	 src_header = vec16(src_header);

      assert(brw_inst_pred_control(devinfo, insn) == BRW_PREDICATE_NONE);
      if (devinfo->gen < 6)
         brw_inst_set_base_mrf(devinfo, insn, mrf.nr);

      /* Until gen6, writes followed by reads from the same location
       * are not guaranteed to be ordered unless write_commit is set.
       * If set, then a no-op write is issued to the destination
       * register to set a dependency, and a read from the destination
       * can be used to ensure the ordering.
       *
       * For gen6, only writes between different threads need ordering
       * protection.  Our use of DP writes is all about register
       * spilling within a thread.
       */
      if (devinfo->gen >= 6) {
	 dest = retype(vec16(brw_null_reg()), BRW_REGISTER_TYPE_UW);
	 send_commit_msg = 0;
      } else {
	 dest = src_header;
	 send_commit_msg = 1;
      }

      brw_set_dest(p, insn, dest);
      if (devinfo->gen >= 6) {
	 brw_set_src0(p, insn, mrf);
      } else {
	 brw_set_src0(p, insn, brw_null_reg());
      }

      if (devinfo->gen >= 6)
	 msg_type = GEN6_DATAPORT_WRITE_MESSAGE_OWORD_BLOCK_WRITE;
      else
	 msg_type = BRW_DATAPORT_WRITE_MESSAGE_OWORD_BLOCK_WRITE;

      brw_set_desc(p, insn,
                   brw_message_desc(devinfo, mlen, send_commit_msg, true) |
                   brw_dp_write_desc(devinfo, brw_scratch_surface_idx(p),
                                     BRW_DATAPORT_OWORD_BLOCK_DWORDS(num_regs * 8),
                                     msg_type, 0, /* not a render target */
                                     send_commit_msg));
   }
}


/**
 * Read a block of owords (half a GRF each) from the scratch buffer
 * using a constant index per channel.
 *
 * Offset must be aligned to oword size (16 bytes).  Used for register
 * spilling.
 */
void
brw_oword_block_read_scratch(struct brw_codegen *p,
			     struct brw_reg dest,
			     struct brw_reg mrf,
			     int num_regs,
			     unsigned offset)
{
   const struct gen_device_info *devinfo = p->devinfo;

   if (devinfo->gen >= 6)
      offset /= 16;

   if (p->devinfo->gen >= 7) {
      /* On gen 7 and above, we no longer have message registers and we can
       * send from any register we want.  By using the destination register
       * for the message, we guarantee that the implied message write won't
       * accidentally overwrite anything.  This has been a problem because
       * the MRF registers and source for the final FB write are both fixed
       * and may overlap.
       */
      mrf = retype(dest, BRW_REGISTER_TYPE_UD);
   } else {
      mrf = retype(mrf, BRW_REGISTER_TYPE_UD);
   }
   dest = retype(dest, BRW_REGISTER_TYPE_UW);

   const unsigned rlen = num_regs;
   const unsigned target_cache =
      (devinfo->gen >= 7 ? GEN7_SFID_DATAPORT_DATA_CACHE :
       devinfo->gen >= 6 ? GEN6_SFID_DATAPORT_RENDER_CACHE :
       BRW_SFID_DATAPORT_READ);

   {
      brw_push_insn_state(p);
      brw_set_default_exec_size(p, BRW_EXECUTE_8);
      brw_set_default_compression_control(p, BRW_COMPRESSION_NONE);
      brw_set_default_mask_control(p, BRW_MASK_DISABLE);

      brw_MOV(p, mrf, retype(brw_vec8_grf(0, 0), BRW_REGISTER_TYPE_UD));

      /* set message header global offset field (reg 0, element 2) */
      brw_set_default_exec_size(p, BRW_EXECUTE_1);
      brw_MOV(p, get_element_ud(mrf, 2), brw_imm_ud(offset));

      brw_pop_insn_state(p);
   }

   {
      brw_inst *insn = next_insn(p, BRW_OPCODE_SEND);

      brw_inst_set_sfid(devinfo, insn, target_cache);
      assert(brw_inst_pred_control(devinfo, insn) == 0);
      brw_inst_set_compression(devinfo, insn, false);

      brw_set_dest(p, insn, dest);	/* UW? */
      if (devinfo->gen >= 6) {
	 brw_set_src0(p, insn, mrf);
      } else {
	 brw_set_src0(p, insn, brw_null_reg());
         brw_inst_set_base_mrf(devinfo, insn, mrf.nr);
      }

      brw_set_desc(p, insn,
                   brw_message_desc(devinfo, 1, rlen, true) |
                   brw_dp_read_desc(devinfo, brw_scratch_surface_idx(p),
                                    BRW_DATAPORT_OWORD_BLOCK_DWORDS(num_regs * 8),
                                    BRW_DATAPORT_READ_MESSAGE_OWORD_BLOCK_READ,
                                    BRW_DATAPORT_READ_TARGET_RENDER_CACHE));
   }
}

void
gen7_block_read_scratch(struct brw_codegen *p,
                        struct brw_reg dest,
                        int num_regs,
                        unsigned offset)
{
   brw_inst *insn = next_insn(p, BRW_OPCODE_SEND);
   assert(brw_inst_pred_control(p->devinfo, insn) == BRW_PREDICATE_NONE);

   brw_set_dest(p, insn, retype(dest, BRW_REGISTER_TYPE_UW));

   /* The HW requires that the header is present; this is to get the g0.5
    * scratch offset.
    */
   brw_set_src0(p, insn, brw_vec8_grf(0, 0));

   /* According to the docs, offset is "A 12-bit HWord offset into the memory
    * Immediate Memory buffer as specified by binding table 0xFF."  An HWORD
    * is 32 bytes, which happens to be the size of a register.
    */
   offset /= REG_SIZE;
   assert(offset < (1 << 12));

   gen7_set_dp_scratch_message(p, insn,
                               false, /* scratch read */
                               false, /* OWords */
                               false, /* invalidate after read */
                               num_regs,
                               offset,
                               1,        /* mlen: just g0 */
                               num_regs, /* rlen */
                               true);    /* header present */
}

/**
 * Read float[4] vectors from the data port constant cache.
 * Location (in buffer) should be a multiple of 16.
 * Used for fetching shader constants.
 */
void brw_oword_block_read(struct brw_codegen *p,
			  struct brw_reg dest,
			  struct brw_reg mrf,
			  uint32_t offset,
			  uint32_t bind_table_index)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned target_cache =
      (devinfo->gen >= 6 ? GEN6_SFID_DATAPORT_CONSTANT_CACHE :
       BRW_SFID_DATAPORT_READ);
   const unsigned exec_size = 1 << brw_get_default_exec_size(p);

   /* On newer hardware, offset is in units of owords. */
   if (devinfo->gen >= 6)
      offset /= 16;

   mrf = retype(mrf, BRW_REGISTER_TYPE_UD);

   brw_push_insn_state(p);
   brw_set_default_predicate_control(p, BRW_PREDICATE_NONE);
   brw_set_default_compression_control(p, BRW_COMPRESSION_NONE);
   brw_set_default_mask_control(p, BRW_MASK_DISABLE);

   brw_push_insn_state(p);
   brw_set_default_exec_size(p, BRW_EXECUTE_8);
   brw_MOV(p, mrf, retype(brw_vec8_grf(0, 0), BRW_REGISTER_TYPE_UD));

   /* set message header global offset field (reg 0, element 2) */
   brw_set_default_exec_size(p, BRW_EXECUTE_1);
   brw_MOV(p,
	   retype(brw_vec1_reg(BRW_MESSAGE_REGISTER_FILE,
			       mrf.nr,
			       2), BRW_REGISTER_TYPE_UD),
	   brw_imm_ud(offset));
   brw_pop_insn_state(p);

   brw_inst *insn = next_insn(p, BRW_OPCODE_SEND);

   brw_inst_set_sfid(devinfo, insn, target_cache);

   /* cast dest to a uword[8] vector */
   dest = retype(vec8(dest), BRW_REGISTER_TYPE_UW);

   brw_set_dest(p, insn, dest);
   if (devinfo->gen >= 6) {
      brw_set_src0(p, insn, mrf);
   } else {
      brw_set_src0(p, insn, brw_null_reg());
      brw_inst_set_base_mrf(devinfo, insn, mrf.nr);
   }

   brw_set_desc(p, insn,
                brw_message_desc(devinfo, 1, DIV_ROUND_UP(exec_size, 8), true) |
                brw_dp_read_desc(devinfo, bind_table_index,
                                 BRW_DATAPORT_OWORD_BLOCK_DWORDS(exec_size),
                                 BRW_DATAPORT_READ_MESSAGE_OWORD_BLOCK_READ,
                                 BRW_DATAPORT_READ_TARGET_DATA_CACHE));

   brw_pop_insn_state(p);
}

brw_inst *
brw_fb_WRITE(struct brw_codegen *p,
             struct brw_reg payload,
             struct brw_reg implied_header,
             unsigned msg_control,
             unsigned binding_table_index,
             unsigned msg_length,
             unsigned response_length,
             bool eot,
             bool last_render_target,
             bool header_present)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned target_cache =
      (devinfo->gen >= 6 ? GEN6_SFID_DATAPORT_RENDER_CACHE :
       BRW_SFID_DATAPORT_WRITE);
   brw_inst *insn;
   unsigned msg_type;
   struct brw_reg dest, src0;

   if (brw_get_default_exec_size(p) >= BRW_EXECUTE_16)
      dest = retype(vec16(brw_null_reg()), BRW_REGISTER_TYPE_UW);
   else
      dest = retype(vec8(brw_null_reg()), BRW_REGISTER_TYPE_UW);

   if (devinfo->gen >= 6) {
      insn = next_insn(p, BRW_OPCODE_SENDC);
   } else {
      insn = next_insn(p, BRW_OPCODE_SEND);
   }
   brw_inst_set_sfid(devinfo, insn, target_cache);
   brw_inst_set_compression(devinfo, insn, false);

   if (devinfo->gen >= 6) {
      /* headerless version, just submit color payload */
      src0 = payload;

      msg_type = GEN6_DATAPORT_WRITE_MESSAGE_RENDER_TARGET_WRITE;
   } else {
      assert(payload.file == BRW_MESSAGE_REGISTER_FILE);
      brw_inst_set_base_mrf(devinfo, insn, payload.nr);
      src0 = implied_header;

      msg_type = BRW_DATAPORT_WRITE_MESSAGE_RENDER_TARGET_WRITE;
   }

   brw_set_dest(p, insn, dest);
   brw_set_src0(p, insn, src0);
   brw_set_desc(p, insn,
                brw_message_desc(devinfo, msg_length, response_length,
                                 header_present) |
                brw_dp_write_desc(devinfo, binding_table_index, msg_control,
                                  msg_type, last_render_target,
                                  0 /* send_commit_msg */));
   brw_inst_set_eot(devinfo, insn, eot);

   return insn;
}

brw_inst *
gen9_fb_READ(struct brw_codegen *p,
             struct brw_reg dst,
             struct brw_reg payload,
             unsigned binding_table_index,
             unsigned msg_length,
             unsigned response_length,
             bool per_sample)
{
   const struct gen_device_info *devinfo = p->devinfo;
   assert(devinfo->gen >= 9);
   const unsigned msg_subtype =
      brw_get_default_exec_size(p) == BRW_EXECUTE_16 ? 0 : 1;
   brw_inst *insn = next_insn(p, BRW_OPCODE_SENDC);

   brw_inst_set_sfid(devinfo, insn, GEN6_SFID_DATAPORT_RENDER_CACHE);
   brw_set_dest(p, insn, dst);
   brw_set_src0(p, insn, payload);
   brw_set_desc(
      p, insn,
      brw_message_desc(devinfo, msg_length, response_length, true) |
      brw_dp_read_desc(devinfo, binding_table_index,
                       per_sample << 5 | msg_subtype,
                       GEN9_DATAPORT_RC_RENDER_TARGET_READ,
                       BRW_DATAPORT_READ_TARGET_RENDER_CACHE));
   brw_inst_set_rt_slot_group(devinfo, insn, brw_get_default_group(p) / 16);

   return insn;
}

/**
 * Texture sample instruction.
 * Note: the msg_type plus msg_length values determine exactly what kind
 * of sampling operation is performed.  See volume 4, page 161 of docs.
 */
void brw_SAMPLE(struct brw_codegen *p,
		struct brw_reg dest,
		unsigned msg_reg_nr,
		struct brw_reg src0,
		unsigned binding_table_index,
		unsigned sampler,
		unsigned msg_type,
		unsigned response_length,
		unsigned msg_length,
		unsigned header_present,
		unsigned simd_mode,
		unsigned return_format)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn;

   if (msg_reg_nr != -1)
      gen6_resolve_implied_move(p, &src0, msg_reg_nr);

   insn = next_insn(p, BRW_OPCODE_SEND);
   brw_inst_set_sfid(devinfo, insn, BRW_SFID_SAMPLER);
   brw_inst_set_pred_control(devinfo, insn, BRW_PREDICATE_NONE); /* XXX */

   /* From the 965 PRM (volume 4, part 1, section 14.2.41):
    *
    *    "Instruction compression is not allowed for this instruction (that
    *     is, send). The hardware behavior is undefined if this instruction is
    *     set as compressed. However, compress control can be set to "SecHalf"
    *     to affect the EMask generation."
    *
    * No similar wording is found in later PRMs, but there are examples
    * utilizing send with SecHalf.  More importantly, SIMD8 sampler messages
    * are allowed in SIMD16 mode and they could not work without SecHalf.  For
    * these reasons, we allow BRW_COMPRESSION_2NDHALF here.
    */
   brw_inst_set_compression(devinfo, insn, false);

   if (devinfo->gen < 6)
      brw_inst_set_base_mrf(devinfo, insn, msg_reg_nr);

   brw_set_dest(p, insn, dest);
   brw_set_src0(p, insn, src0);
   brw_set_desc(p, insn,
                brw_message_desc(devinfo, msg_length, response_length,
                                 header_present) |
                brw_sampler_desc(devinfo, binding_table_index, sampler,
                                 msg_type, simd_mode, return_format));
}

/* Adjust the message header's sampler state pointer to
 * select the correct group of 16 samplers.
 */
void brw_adjust_sampler_state_pointer(struct brw_codegen *p,
                                      struct brw_reg header,
                                      struct brw_reg sampler_index)
{
   /* The "Sampler Index" field can only store values between 0 and 15.
    * However, we can add an offset to the "Sampler State Pointer"
    * field, effectively selecting a different set of 16 samplers.
    *
    * The "Sampler State Pointer" needs to be aligned to a 32-byte
    * offset, and each sampler state is only 16-bytes, so we can't
    * exclusively use the offset - we have to use both.
    */

   const struct gen_device_info *devinfo = p->devinfo;

   if (sampler_index.file == BRW_IMMEDIATE_VALUE) {
      const int sampler_state_size = 16; /* 16 bytes */
      uint32_t sampler = sampler_index.ud;

      if (sampler >= 16) {
         assert(devinfo->is_haswell || devinfo->gen >= 8);
         brw_ADD(p,
                 get_element_ud(header, 3),
                 get_element_ud(brw_vec8_grf(0, 0), 3),
                 brw_imm_ud(16 * (sampler / 16) * sampler_state_size));
      }
   } else {
      /* Non-const sampler array indexing case */
      if (devinfo->gen < 8 && !devinfo->is_haswell) {
         return;
      }

      struct brw_reg temp = get_element_ud(header, 3);

      brw_AND(p, temp, get_element_ud(sampler_index, 0), brw_imm_ud(0x0f0));
      brw_SHL(p, temp, temp, brw_imm_ud(4));
      brw_ADD(p,
              get_element_ud(header, 3),
              get_element_ud(brw_vec8_grf(0, 0), 3),
              temp);
   }
}

/* All these variables are pretty confusing - we might be better off
 * using bitmasks and macros for this, in the old style.  Or perhaps
 * just having the caller instantiate the fields in dword3 itself.
 */
void brw_urb_WRITE(struct brw_codegen *p,
		   struct brw_reg dest,
		   unsigned msg_reg_nr,
		   struct brw_reg src0,
                   enum brw_urb_write_flags flags,
		   unsigned msg_length,
		   unsigned response_length,
		   unsigned offset,
		   unsigned swizzle)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn;

   gen6_resolve_implied_move(p, &src0, msg_reg_nr);

   if (devinfo->gen >= 7 && !(flags & BRW_URB_WRITE_USE_CHANNEL_MASKS)) {
      /* Enable Channel Masks in the URB_WRITE_HWORD message header */
      brw_push_insn_state(p);
      brw_set_default_access_mode(p, BRW_ALIGN_1);
      brw_set_default_mask_control(p, BRW_MASK_DISABLE);
      brw_set_default_exec_size(p, BRW_EXECUTE_1);
      brw_OR(p, retype(brw_vec1_reg(BRW_MESSAGE_REGISTER_FILE, msg_reg_nr, 5),
		       BRW_REGISTER_TYPE_UD),
	        retype(brw_vec1_grf(0, 5), BRW_REGISTER_TYPE_UD),
		brw_imm_ud(0xff00));
      brw_pop_insn_state(p);
   }

   insn = next_insn(p, BRW_OPCODE_SEND);

   assert(msg_length < BRW_MAX_MRF(devinfo->gen));

   brw_set_dest(p, insn, dest);
   brw_set_src0(p, insn, src0);
   brw_set_src1(p, insn, brw_imm_d(0));

   if (devinfo->gen < 6)
      brw_inst_set_base_mrf(devinfo, insn, msg_reg_nr);

   brw_set_urb_message(p,
		       insn,
		       flags,
		       msg_length,
		       response_length,
		       offset,
		       swizzle);
}

void
brw_send_indirect_message(struct brw_codegen *p,
                          unsigned sfid,
                          struct brw_reg dst,
                          struct brw_reg payload,
                          struct brw_reg desc,
                          unsigned desc_imm)
{
   const struct gen_device_info *devinfo = p->devinfo;
   struct brw_inst *send;

   dst = retype(dst, BRW_REGISTER_TYPE_UW);

   assert(desc.type == BRW_REGISTER_TYPE_UD);

   if (desc.file == BRW_IMMEDIATE_VALUE) {
      send = next_insn(p, BRW_OPCODE_SEND);
      brw_set_desc(p, send, desc.ud | desc_imm);

   } else {
      struct brw_reg addr = retype(brw_address_reg(0), BRW_REGISTER_TYPE_UD);

      brw_push_insn_state(p);
      brw_set_default_access_mode(p, BRW_ALIGN_1);
      brw_set_default_mask_control(p, BRW_MASK_DISABLE);
      brw_set_default_exec_size(p, BRW_EXECUTE_1);
      brw_set_default_predicate_control(p, BRW_PREDICATE_NONE);

      /* Load the indirect descriptor to an address register using OR so the
       * caller can specify additional descriptor bits with the desc_imm
       * immediate.
       */
      brw_OR(p, addr, desc, brw_imm_ud(desc_imm));

      brw_pop_insn_state(p);

      send = next_insn(p, BRW_OPCODE_SEND);
      brw_set_src1(p, send, addr);
   }

   if (dst.width < BRW_EXECUTE_8)
      brw_inst_set_exec_size(devinfo, send, dst.width);

   brw_set_dest(p, send, dst);
   brw_set_src0(p, send, retype(payload, BRW_REGISTER_TYPE_UD));
   brw_inst_set_sfid(devinfo, send, sfid);
}

static void
brw_send_indirect_surface_message(struct brw_codegen *p,
                                  unsigned sfid,
                                  struct brw_reg dst,
                                  struct brw_reg payload,
                                  struct brw_reg surface,
                                  unsigned desc_imm)
{
   if (surface.file != BRW_IMMEDIATE_VALUE) {
      struct brw_reg addr = retype(brw_address_reg(0), BRW_REGISTER_TYPE_UD);

      brw_push_insn_state(p);
      brw_set_default_access_mode(p, BRW_ALIGN_1);
      brw_set_default_mask_control(p, BRW_MASK_DISABLE);
      brw_set_default_exec_size(p, BRW_EXECUTE_1);
      brw_set_default_predicate_control(p, BRW_PREDICATE_NONE);

      /* Mask out invalid bits from the surface index to avoid hangs e.g. when
       * some surface array is accessed out of bounds.
       */
      brw_AND(p, addr,
              suboffset(vec1(retype(surface, BRW_REGISTER_TYPE_UD)),
                        BRW_GET_SWZ(surface.swizzle, 0)),
              brw_imm_ud(0xff));

      brw_pop_insn_state(p);

      surface = addr;
   }

   brw_send_indirect_message(p, sfid, dst, payload, surface, desc_imm);
}

static bool
while_jumps_before_offset(const struct gen_device_info *devinfo,
                          brw_inst *insn, int while_offset, int start_offset)
{
   int scale = 16 / brw_jump_scale(devinfo);
   int jip = devinfo->gen == 6 ? brw_inst_gen6_jump_count(devinfo, insn)
                               : brw_inst_jip(devinfo, insn);
   assert(jip < 0);
   return while_offset + jip * scale <= start_offset;
}


static int
brw_find_next_block_end(struct brw_codegen *p, int start_offset)
{
   int offset;
   void *store = p->store;
   const struct gen_device_info *devinfo = p->devinfo;

   int depth = 0;

   for (offset = next_offset(devinfo, store, start_offset);
        offset < p->next_insn_offset;
        offset = next_offset(devinfo, store, offset)) {
      brw_inst *insn = store + offset;

      switch (brw_inst_opcode(devinfo, insn)) {
      case BRW_OPCODE_IF:
         depth++;
         break;
      case BRW_OPCODE_ENDIF:
         if (depth == 0)
            return offset;
         depth--;
         break;
      case BRW_OPCODE_WHILE:
         /* If the while doesn't jump before our instruction, it's the end
          * of a sibling do...while loop.  Ignore it.
          */
         if (!while_jumps_before_offset(devinfo, insn, offset, start_offset))
            continue;
         /* fallthrough */
      case BRW_OPCODE_ELSE:
      case BRW_OPCODE_HALT:
         if (depth == 0)
            return offset;
      }
   }

   return 0;
}

/* There is no DO instruction on gen6, so to find the end of the loop
 * we have to see if the loop is jumping back before our start
 * instruction.
 */
static int
brw_find_loop_end(struct brw_codegen *p, int start_offset)
{
   const struct gen_device_info *devinfo = p->devinfo;
   int offset;
   void *store = p->store;

   assert(devinfo->gen >= 6);

   /* Always start after the instruction (such as a WHILE) we're trying to fix
    * up.
    */
   for (offset = next_offset(devinfo, store, start_offset);
        offset < p->next_insn_offset;
        offset = next_offset(devinfo, store, offset)) {
      brw_inst *insn = store + offset;

      if (brw_inst_opcode(devinfo, insn) == BRW_OPCODE_WHILE) {
	 if (while_jumps_before_offset(devinfo, insn, offset, start_offset))
	    return offset;
      }
   }
   assert(!"not reached");
   return start_offset;
}

/* After program generation, go back and update the UIP and JIP of
 * BREAK, CONT, and HALT instructions to their correct locations.
 */
void
brw_set_uip_jip(struct brw_codegen *p, int start_offset)
{
   const struct gen_device_info *devinfo = p->devinfo;
   int offset;
   int br = brw_jump_scale(devinfo);
   int scale = 16 / br;
   void *store = p->store;

   if (devinfo->gen < 6)
      return;

   for (offset = start_offset; offset < p->next_insn_offset; offset += 16) {
      brw_inst *insn = store + offset;
      assert(brw_inst_cmpt_control(devinfo, insn) == 0);

      int block_end_offset = brw_find_next_block_end(p, offset);
      switch (brw_inst_opcode(devinfo, insn)) {
      case BRW_OPCODE_BREAK:
         assert(block_end_offset != 0);
         brw_inst_set_jip(devinfo, insn, (block_end_offset - offset) / scale);
	 /* Gen7 UIP points to WHILE; Gen6 points just after it */
         brw_inst_set_uip(devinfo, insn,
	    (brw_find_loop_end(p, offset) - offset +
             (devinfo->gen == 6 ? 16 : 0)) / scale);
	 break;
      case BRW_OPCODE_CONTINUE:
         assert(block_end_offset != 0);
         brw_inst_set_jip(devinfo, insn, (block_end_offset - offset) / scale);
         brw_inst_set_uip(devinfo, insn,
            (brw_find_loop_end(p, offset) - offset) / scale);

         assert(brw_inst_uip(devinfo, insn) != 0);
         assert(brw_inst_jip(devinfo, insn) != 0);
	 break;

      case BRW_OPCODE_ENDIF: {
         int32_t jump = (block_end_offset == 0) ?
                        1 * br : (block_end_offset - offset) / scale;
         if (devinfo->gen >= 7)
            brw_inst_set_jip(devinfo, insn, jump);
         else
            brw_inst_set_gen6_jump_count(devinfo, insn, jump);
	 break;
      }

      case BRW_OPCODE_HALT:
	 /* From the Sandy Bridge PRM (volume 4, part 2, section 8.3.19):
	  *
	  *    "In case of the halt instruction not inside any conditional
	  *     code block, the value of <JIP> and <UIP> should be the
	  *     same. In case of the halt instruction inside conditional code
	  *     block, the <UIP> should be the end of the program, and the
	  *     <JIP> should be end of the most inner conditional code block."
	  *
	  * The uip will have already been set by whoever set up the
	  * instruction.
	  */
	 if (block_end_offset == 0) {
            brw_inst_set_jip(devinfo, insn, brw_inst_uip(devinfo, insn));
	 } else {
            brw_inst_set_jip(devinfo, insn, (block_end_offset - offset) / scale);
	 }
         assert(brw_inst_uip(devinfo, insn) != 0);
         assert(brw_inst_jip(devinfo, insn) != 0);
	 break;
      }
   }
}

void brw_ff_sync(struct brw_codegen *p,
		   struct brw_reg dest,
		   unsigned msg_reg_nr,
		   struct brw_reg src0,
		   bool allocate,
		   unsigned response_length,
		   bool eot)
{
   const struct gen_device_info *devinfo = p->devinfo;
   brw_inst *insn;

   gen6_resolve_implied_move(p, &src0, msg_reg_nr);

   insn = next_insn(p, BRW_OPCODE_SEND);
   brw_set_dest(p, insn, dest);
   brw_set_src0(p, insn, src0);
   brw_set_src1(p, insn, brw_imm_d(0));

   if (devinfo->gen < 6)
      brw_inst_set_base_mrf(devinfo, insn, msg_reg_nr);

   brw_set_ff_sync_message(p,
			   insn,
			   allocate,
			   response_length,
			   eot);
}

/**
 * Emit the SEND instruction necessary to generate stream output data on Gen6
 * (for transform feedback).
 *
 * If send_commit_msg is true, this is the last piece of stream output data
 * from this thread, so send the data as a committed write.  According to the
 * Sandy Bridge PRM (volume 2 part 1, section 4.5.1):
 *
 *   "Prior to End of Thread with a URB_WRITE, the kernel must ensure all
 *   writes are complete by sending the final write as a committed write."
 */
void
brw_svb_write(struct brw_codegen *p,
              struct brw_reg dest,
              unsigned msg_reg_nr,
              struct brw_reg src0,
              unsigned binding_table_index,
              bool   send_commit_msg)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned target_cache =
      (devinfo->gen >= 7 ? GEN7_SFID_DATAPORT_DATA_CACHE :
       devinfo->gen >= 6 ? GEN6_SFID_DATAPORT_RENDER_CACHE :
       BRW_SFID_DATAPORT_WRITE);
   brw_inst *insn;

   gen6_resolve_implied_move(p, &src0, msg_reg_nr);

   insn = next_insn(p, BRW_OPCODE_SEND);
   brw_inst_set_sfid(devinfo, insn, target_cache);
   brw_set_dest(p, insn, dest);
   brw_set_src0(p, insn, src0);
   brw_set_desc(p, insn,
                brw_message_desc(devinfo, 1, send_commit_msg, true) |
                brw_dp_write_desc(devinfo, binding_table_index,
                                  0, /* msg_control: ignored */
                                  GEN6_DATAPORT_WRITE_MESSAGE_STREAMED_VB_WRITE,
                                  0, /* last_render_target: ignored */
                                  send_commit_msg)); /* send_commit_msg */
}

static unsigned
brw_surface_payload_size(struct brw_codegen *p,
                         unsigned num_channels,
                         bool has_simd4x2,
                         bool has_simd16)
{
   if (has_simd4x2 && brw_get_default_access_mode(p) == BRW_ALIGN_16)
      return 1;
   else if (has_simd16 && brw_get_default_exec_size(p) == BRW_EXECUTE_16)
      return 2 * num_channels;
   else
      return num_channels;
}

static uint32_t
brw_dp_untyped_atomic_desc(struct brw_codegen *p,
                           unsigned atomic_op,
                           bool response_expected)
{
   const struct gen_device_info *devinfo = p->devinfo;
   unsigned msg_control =
      atomic_op | /* Atomic Operation Type: BRW_AOP_* */
      (response_expected ? 1 << 5 : 0); /* Return data expected */
   unsigned msg_type;

   if (devinfo->gen >= 8 || devinfo->is_haswell) {
      if (brw_get_default_access_mode(p) == BRW_ALIGN_1) {
         if (brw_get_default_exec_size(p) != BRW_EXECUTE_16)
            msg_control |= 1 << 4; /* SIMD8 mode */

         msg_type = HSW_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_OP;
      } else {
         msg_type = HSW_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_OP_SIMD4X2;
      }
   } else {
      if (brw_get_default_exec_size(p) != BRW_EXECUTE_16)
         msg_control |= 1 << 4; /* SIMD8 mode */

      msg_type = GEN7_DATAPORT_DC_UNTYPED_ATOMIC_OP;
   }

   return brw_dp_surface_desc(devinfo, msg_type, msg_control);
}

void
brw_untyped_atomic(struct brw_codegen *p,
                   struct brw_reg dst,
                   struct brw_reg payload,
                   struct brw_reg surface,
                   unsigned atomic_op,
                   unsigned msg_length,
                   bool response_expected,
                   bool header_present)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned sfid = (devinfo->gen >= 8 || devinfo->is_haswell ?
                          HSW_SFID_DATAPORT_DATA_CACHE_1 :
                          GEN7_SFID_DATAPORT_DATA_CACHE);
   const unsigned response_length = brw_surface_payload_size(
      p, response_expected, devinfo->gen >= 8 || devinfo->is_haswell, true);
   const unsigned desc =
      brw_message_desc(devinfo, msg_length, response_length, header_present) |
      brw_dp_untyped_atomic_desc(p, atomic_op, response_expected);
   const bool align1 = brw_get_default_access_mode(p) == BRW_ALIGN_1;
   /* Mask out unused components -- This is especially important in Align16
    * mode on generations that don't have native support for SIMD4x2 atomics,
    * because unused but enabled components will cause the dataport to perform
    * additional atomic operations on the addresses that happen to be in the
    * uninitialized Y, Z and W coordinates of the payload.
    */
   const unsigned mask = align1 ? WRITEMASK_XYZW : WRITEMASK_X;

   brw_send_indirect_surface_message(p, sfid, brw_writemask(dst, mask),
                                     payload, surface, desc);
}

static uint32_t
brw_dp_untyped_atomic_float_desc(struct brw_codegen *p,
                                 unsigned atomic_op,
                                 bool response_expected)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned msg_type = GEN9_DATAPORT_DC_PORT1_UNTYPED_ATOMIC_FLOAT_OP;
   unsigned msg_control =
      atomic_op | /* Atomic Operation Type: BRW_AOP_F* */
      (response_expected ? 1 << 5 : 0); /* Return data expected */

   assert(devinfo->gen >= 9);
   assert(brw_get_default_access_mode(p) == BRW_ALIGN_1);

   if (brw_get_default_exec_size(p) != BRW_EXECUTE_16)
      msg_control |= 1 << 4; /* SIMD8 mode */

   return brw_dp_surface_desc(devinfo, msg_type, msg_control);
}

void
brw_untyped_atomic_float(struct brw_codegen *p,
                         struct brw_reg dst,
                         struct brw_reg payload,
                         struct brw_reg surface,
                         unsigned atomic_op,
                         unsigned msg_length,
                         bool response_expected,
                         bool header_present)
{
   const struct gen_device_info *devinfo = p->devinfo;

   assert(devinfo->gen >= 9);
   assert(brw_get_default_access_mode(p) == BRW_ALIGN_1);

   const unsigned sfid = HSW_SFID_DATAPORT_DATA_CACHE_1;
   const unsigned response_length = brw_surface_payload_size(
      p, response_expected, true, true);
   const unsigned desc =
      brw_message_desc(devinfo, msg_length, response_length, header_present) |
      brw_dp_untyped_atomic_float_desc(p, atomic_op, response_expected);

   brw_send_indirect_surface_message(p, sfid,
                                     brw_writemask(dst, WRITEMASK_XYZW),
                                     payload, surface, desc);
}

static uint32_t
brw_dp_untyped_surface_read_desc(struct brw_codegen *p,
                                 unsigned num_channels)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned msg_type = (devinfo->gen >= 8 || devinfo->is_haswell ?
                              HSW_DATAPORT_DC_PORT1_UNTYPED_SURFACE_READ :
                              GEN7_DATAPORT_DC_UNTYPED_SURFACE_READ);
   /* Set mask of 32-bit channels to drop. */
   unsigned msg_control = 0xf & (0xf << num_channels);

   if (brw_get_default_access_mode(p) == BRW_ALIGN_1) {
      if (brw_get_default_exec_size(p) == BRW_EXECUTE_16)
         msg_control |= 1 << 4; /* SIMD16 mode */
      else
         msg_control |= 2 << 4; /* SIMD8 mode */
   }

   return brw_dp_surface_desc(devinfo, msg_type, msg_control);
}

void
brw_untyped_surface_read(struct brw_codegen *p,
                         struct brw_reg dst,
                         struct brw_reg payload,
                         struct brw_reg surface,
                         unsigned msg_length,
                         unsigned num_channels)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned sfid = (devinfo->gen >= 8 || devinfo->is_haswell ?
                          HSW_SFID_DATAPORT_DATA_CACHE_1 :
                          GEN7_SFID_DATAPORT_DATA_CACHE);
   const unsigned response_length =
      brw_surface_payload_size(p, num_channels, true, true);
   const unsigned desc =
      brw_message_desc(devinfo, msg_length, response_length, false) |
      brw_dp_untyped_surface_read_desc(p, num_channels);

   brw_send_indirect_surface_message(p, sfid, dst, payload, surface, desc);
}

static uint32_t
brw_dp_untyped_surface_write_desc(struct brw_codegen *p,
                                  unsigned num_channels)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned msg_type = (devinfo->gen >= 8 || devinfo->is_haswell ?
                              HSW_DATAPORT_DC_PORT1_UNTYPED_SURFACE_WRITE :
                              GEN7_DATAPORT_DC_UNTYPED_SURFACE_WRITE);
   /* Set mask of 32-bit channels to drop. */
   unsigned msg_control = 0xf & (0xf << num_channels);

   if (brw_get_default_access_mode(p) == BRW_ALIGN_1) {
      if (brw_get_default_exec_size(p) == BRW_EXECUTE_16)
         msg_control |= 1 << 4; /* SIMD16 mode */
      else
         msg_control |= 2 << 4; /* SIMD8 mode */
   } else {
      if (devinfo->gen >= 8 || devinfo->is_haswell)
         msg_control |= 0 << 4; /* SIMD4x2 mode */
      else
         msg_control |= 2 << 4; /* SIMD8 mode */
   }

   return brw_dp_surface_desc(devinfo, msg_type, msg_control);
}

void
brw_untyped_surface_write(struct brw_codegen *p,
                          struct brw_reg payload,
                          struct brw_reg surface,
                          unsigned msg_length,
                          unsigned num_channels,
                          bool header_present)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned sfid = (devinfo->gen >= 8 || devinfo->is_haswell ?
                          HSW_SFID_DATAPORT_DATA_CACHE_1 :
                          GEN7_SFID_DATAPORT_DATA_CACHE);
   const unsigned desc =
      brw_message_desc(devinfo, msg_length, 0, header_present) |
      brw_dp_untyped_surface_write_desc(p, num_channels);
   const bool align1 = brw_get_default_access_mode(p) == BRW_ALIGN_1;
   /* Mask out unused components -- See comment in brw_untyped_atomic(). */
   const unsigned mask = devinfo->gen == 7 && !devinfo->is_haswell && !align1 ?
                          WRITEMASK_X : WRITEMASK_XYZW;

   brw_send_indirect_surface_message(p, sfid, brw_writemask(brw_null_reg(), mask),
                                     payload, surface, desc);
}

static unsigned
brw_byte_scattered_data_element_from_bit_size(unsigned bit_size)
{
   switch (bit_size) {
   case 8:
      return GEN7_BYTE_SCATTERED_DATA_ELEMENT_BYTE;
   case 16:
      return GEN7_BYTE_SCATTERED_DATA_ELEMENT_WORD;
   case 32:
      return GEN7_BYTE_SCATTERED_DATA_ELEMENT_DWORD;
   default:
      unreachable("Unsupported bit_size for byte scattered messages");
   }
}

static uint32_t
brw_dp_byte_scattered_desc(struct brw_codegen *p, unsigned bit_size,
                           unsigned msg_type)
{
   const struct gen_device_info *devinfo = p->devinfo;
   unsigned msg_control =
      brw_byte_scattered_data_element_from_bit_size(bit_size) << 2;

   if (brw_get_default_exec_size(p) == BRW_EXECUTE_16)
      msg_control |= 1; /* SIMD16 mode */
   else
      msg_control |= 0; /* SIMD8 mode */

   return brw_dp_surface_desc(devinfo, msg_type, msg_control);
}

void
brw_byte_scattered_read(struct brw_codegen *p,
                        struct brw_reg dst,
                        struct brw_reg payload,
                        struct brw_reg surface,
                        unsigned msg_length,
                        unsigned bit_size)
{
   const struct gen_device_info *devinfo = p->devinfo;
   assert(devinfo->gen > 7 || devinfo->is_haswell);
   assert(brw_get_default_access_mode(p) == BRW_ALIGN_1);
   const unsigned response_length =
      brw_surface_payload_size(p, 1, true, true);
   const unsigned desc =
      brw_message_desc(devinfo, msg_length, response_length, false) |
      brw_dp_byte_scattered_desc(p, bit_size,
                                 HSW_DATAPORT_DC_PORT0_BYTE_SCATTERED_READ);

   brw_send_indirect_surface_message(p, GEN7_SFID_DATAPORT_DATA_CACHE,
                                     dst, payload, surface, desc);
}

void
brw_byte_scattered_write(struct brw_codegen *p,
                         struct brw_reg payload,
                         struct brw_reg surface,
                         unsigned msg_length,
                         unsigned bit_size,
                         bool header_present)
{
   const struct gen_device_info *devinfo = p->devinfo;
   assert(devinfo->gen > 7 || devinfo->is_haswell);
   assert(brw_get_default_access_mode(p) == BRW_ALIGN_1);
   const unsigned desc =
      brw_message_desc(devinfo, msg_length, 0, header_present) |
      brw_dp_byte_scattered_desc(p, bit_size,
                                 HSW_DATAPORT_DC_PORT0_BYTE_SCATTERED_WRITE);

   brw_send_indirect_surface_message(p, GEN7_SFID_DATAPORT_DATA_CACHE,
                                     brw_writemask(brw_null_reg(),
                                                   WRITEMASK_XYZW),
                                     payload, surface, desc);
}

static uint32_t
brw_dp_typed_atomic_desc(struct brw_codegen *p,
                         unsigned atomic_op,
                         bool response_expected)
{
   const struct gen_device_info *devinfo = p->devinfo;
   unsigned msg_control =
      atomic_op | /* Atomic Operation Type: BRW_AOP_* */
      (response_expected ? 1 << 5 : 0); /* Return data expected */
   unsigned msg_type;

   if (devinfo->gen >= 8 || devinfo->is_haswell) {
      if (brw_get_default_access_mode(p) == BRW_ALIGN_1) {
         if ((brw_get_default_group(p) / 8) % 2 == 1)
            msg_control |= 1 << 4; /* Use high 8 slots of the sample mask */

         msg_type = HSW_DATAPORT_DC_PORT1_TYPED_ATOMIC_OP;
      } else {
         msg_type = HSW_DATAPORT_DC_PORT1_TYPED_ATOMIC_OP_SIMD4X2;
      }

   } else {
      if ((brw_get_default_group(p) / 8) % 2 == 1)
         msg_control |= 1 << 4; /* Use high 8 slots of the sample mask */

      msg_type = GEN7_DATAPORT_RC_TYPED_ATOMIC_OP;
   }

   return brw_dp_surface_desc(devinfo, msg_type, msg_control);
}

void
brw_typed_atomic(struct brw_codegen *p,
                 struct brw_reg dst,
                 struct brw_reg payload,
                 struct brw_reg surface,
                 unsigned atomic_op,
                 unsigned msg_length,
                 bool response_expected,
                 bool header_present) {
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned sfid = (devinfo->gen >= 8 || devinfo->is_haswell ?
                          HSW_SFID_DATAPORT_DATA_CACHE_1 :
                          GEN6_SFID_DATAPORT_RENDER_CACHE);
   const unsigned response_length = brw_surface_payload_size(
      p, response_expected, devinfo->gen >= 8 || devinfo->is_haswell, false);
   const unsigned desc =
      brw_message_desc(devinfo, msg_length, response_length, header_present) |
      brw_dp_typed_atomic_desc(p, atomic_op, response_expected);
   const bool align1 = brw_get_default_access_mode(p) == BRW_ALIGN_1;
   /* Mask out unused components -- See comment in brw_untyped_atomic(). */
   const unsigned mask = align1 ? WRITEMASK_XYZW : WRITEMASK_X;

   brw_send_indirect_surface_message(p, sfid, brw_writemask(dst, mask),
                                     payload, surface, desc);
}

static uint32_t
brw_dp_typed_surface_read_desc(struct brw_codegen *p,
                               unsigned num_channels)
{
   const struct gen_device_info *devinfo = p->devinfo;
   /* Set mask of unused channels. */
   unsigned msg_control = 0xf & (0xf << num_channels);
   unsigned msg_type;

   if (devinfo->gen >= 8 || devinfo->is_haswell) {
      if (brw_get_default_access_mode(p) == BRW_ALIGN_1) {
         if ((brw_get_default_group(p) / 8) % 2 == 1)
            msg_control |= 2 << 4; /* Use high 8 slots of the sample mask */
         else
            msg_control |= 1 << 4; /* Use low 8 slots of the sample mask */
      }

      msg_type = HSW_DATAPORT_DC_PORT1_TYPED_SURFACE_READ;
   } else {
      if (brw_get_default_access_mode(p) == BRW_ALIGN_1) {
         if ((brw_get_default_group(p) / 8) % 2 == 1)
            msg_control |= 1 << 5; /* Use high 8 slots of the sample mask */
      }

      msg_type = GEN7_DATAPORT_RC_TYPED_SURFACE_READ;
   }

   return brw_dp_surface_desc(devinfo, msg_type, msg_control);
}

void
brw_typed_surface_read(struct brw_codegen *p,
                       struct brw_reg dst,
                       struct brw_reg payload,
                       struct brw_reg surface,
                       unsigned msg_length,
                       unsigned num_channels,
                       bool header_present)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned sfid = (devinfo->gen >= 8 || devinfo->is_haswell ?
                          HSW_SFID_DATAPORT_DATA_CACHE_1 :
                          GEN6_SFID_DATAPORT_RENDER_CACHE);
   const unsigned response_length = brw_surface_payload_size(
      p, num_channels, devinfo->gen >= 8 || devinfo->is_haswell, false);
   const unsigned desc =
      brw_message_desc(devinfo, msg_length, response_length, header_present) |
      brw_dp_typed_surface_read_desc(p, num_channels);

   brw_send_indirect_surface_message(p, sfid, dst, payload, surface, desc);
}

static uint32_t
brw_dp_typed_surface_write_desc(struct brw_codegen *p,
                                unsigned num_channels)
{
   const struct gen_device_info *devinfo = p->devinfo;
   /* Set mask of unused channels. */
   unsigned msg_control = 0xf & (0xf << num_channels);
   unsigned msg_type;

   if (devinfo->gen >= 8 || devinfo->is_haswell) {
      if (brw_get_default_access_mode(p) == BRW_ALIGN_1) {
         if ((brw_get_default_group(p) / 8) % 2 == 1)
            msg_control |= 2 << 4; /* Use high 8 slots of the sample mask */
         else
            msg_control |= 1 << 4; /* Use low 8 slots of the sample mask */
      }

      msg_type = HSW_DATAPORT_DC_PORT1_TYPED_SURFACE_WRITE;

   } else {
      if (brw_get_default_access_mode(p) == BRW_ALIGN_1) {
         if ((brw_get_default_group(p) / 8) % 2 == 1)
            msg_control |= 1 << 5; /* Use high 8 slots of the sample mask */
      }

      msg_type = GEN7_DATAPORT_RC_TYPED_SURFACE_WRITE;
   }

   return brw_dp_surface_desc(devinfo, msg_type, msg_control);
}

void
brw_typed_surface_write(struct brw_codegen *p,
                        struct brw_reg payload,
                        struct brw_reg surface,
                        unsigned msg_length,
                        unsigned num_channels,
                        bool header_present)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned sfid = (devinfo->gen >= 8 || devinfo->is_haswell ?
                          HSW_SFID_DATAPORT_DATA_CACHE_1 :
                          GEN6_SFID_DATAPORT_RENDER_CACHE);
   const unsigned desc =
      brw_message_desc(devinfo, msg_length, 0, header_present) |
      brw_dp_typed_surface_write_desc(p, num_channels);
   const bool align1 = brw_get_default_access_mode(p) == BRW_ALIGN_1;
   /* Mask out unused components -- See comment in brw_untyped_atomic(). */
   const unsigned mask = (devinfo->gen == 7 && !devinfo->is_haswell && !align1 ?
                          WRITEMASK_X : WRITEMASK_XYZW);

   brw_send_indirect_surface_message(p, sfid, brw_writemask(brw_null_reg(), mask),
                                     payload, surface, desc);
}

static void
brw_set_memory_fence_message(struct brw_codegen *p,
                             struct brw_inst *insn,
                             enum brw_message_target sfid,
                             bool commit_enable)
{
   const struct gen_device_info *devinfo = p->devinfo;

   brw_set_desc(p, insn, brw_message_desc(
                   devinfo, 1, (commit_enable ? 1 : 0), true));

   brw_inst_set_sfid(devinfo, insn, sfid);

   switch (sfid) {
   case GEN6_SFID_DATAPORT_RENDER_CACHE:
      brw_inst_set_dp_msg_type(devinfo, insn, GEN7_DATAPORT_RC_MEMORY_FENCE);
      break;
   case GEN7_SFID_DATAPORT_DATA_CACHE:
      brw_inst_set_dp_msg_type(devinfo, insn, GEN7_DATAPORT_DC_MEMORY_FENCE);
      break;
   default:
      unreachable("Not reached");
   }

   if (commit_enable)
      brw_inst_set_dp_msg_control(devinfo, insn, 1 << 5);
}

void
brw_memory_fence(struct brw_codegen *p,
                 struct brw_reg dst,
                 enum opcode send_op)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const bool commit_enable =
      devinfo->gen >= 10 || /* HSD ES # 1404612949 */
      (devinfo->gen == 7 && !devinfo->is_haswell);
   struct brw_inst *insn;

   brw_push_insn_state(p);
   brw_set_default_mask_control(p, BRW_MASK_DISABLE);
   brw_set_default_exec_size(p, BRW_EXECUTE_1);
   dst = vec1(dst);

   /* Set dst as destination for dependency tracking, the MEMORY_FENCE
    * message doesn't write anything back.
    */
   insn = next_insn(p, send_op);
   dst = retype(dst, BRW_REGISTER_TYPE_UW);
   brw_set_dest(p, insn, dst);
   brw_set_src0(p, insn, dst);
   brw_set_memory_fence_message(p, insn, GEN7_SFID_DATAPORT_DATA_CACHE,
                                commit_enable);

   if (devinfo->gen == 7 && !devinfo->is_haswell) {
      /* IVB does typed surface access through the render cache, so we need to
       * flush it too.  Use a different register so both flushes can be
       * pipelined by the hardware.
       */
      insn = next_insn(p, send_op);
      brw_set_dest(p, insn, offset(dst, 1));
      brw_set_src0(p, insn, offset(dst, 1));
      brw_set_memory_fence_message(p, insn, GEN6_SFID_DATAPORT_RENDER_CACHE,
                                   commit_enable);

      /* Now write the response of the second message into the response of the
       * first to trigger a pipeline stall -- This way future render and data
       * cache messages will be properly ordered with respect to past data and
       * render cache messages.
       */
      brw_MOV(p, dst, offset(dst, 1));
   }

   brw_pop_insn_state(p);
}

void
brw_pixel_interpolator_query(struct brw_codegen *p,
                             struct brw_reg dest,
                             struct brw_reg mrf,
                             bool noperspective,
                             unsigned mode,
                             struct brw_reg data,
                             unsigned msg_length,
                             unsigned response_length)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const uint16_t exec_size = brw_get_default_exec_size(p);
   const unsigned slot_group = brw_get_default_group(p) / 16;
   const unsigned simd_mode = (exec_size == BRW_EXECUTE_16);
   const unsigned desc =
      brw_message_desc(devinfo, msg_length, response_length, false) |
      brw_pixel_interp_desc(devinfo, mode, noperspective, simd_mode,
                            slot_group);

   /* brw_send_indirect_message will automatically use a direct send message
    * if data is actually immediate.
    */
   brw_send_indirect_message(p,
                             GEN7_SFID_PIXEL_INTERPOLATOR,
                             dest,
                             mrf,
                             vec1(data),
                             desc);
}

void
brw_find_live_channel(struct brw_codegen *p, struct brw_reg dst,
                      struct brw_reg mask)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned exec_size = 1 << brw_get_default_exec_size(p);
   const unsigned qtr_control = brw_get_default_group(p) / 8;
   brw_inst *inst;

   assert(devinfo->gen >= 7);
   assert(mask.type == BRW_REGISTER_TYPE_UD);

   brw_push_insn_state(p);

   if (brw_get_default_access_mode(p) == BRW_ALIGN_1) {
      brw_set_default_mask_control(p, BRW_MASK_DISABLE);

      if (devinfo->gen >= 8) {
         /* Getting the first active channel index is easy on Gen8: Just find
          * the first bit set in the execution mask.  The register exists on
          * HSW already but it reads back as all ones when the current
          * instruction has execution masking disabled, so it's kind of
          * useless.
          */
         struct brw_reg exec_mask =
            retype(brw_mask_reg(0), BRW_REGISTER_TYPE_UD);

         brw_set_default_exec_size(p, BRW_EXECUTE_1);
         if (mask.file != BRW_IMMEDIATE_VALUE || mask.ud != 0xffffffff) {
            /* Unfortunately, ce0 does not take into account the thread
             * dispatch mask, which may be a problem in cases where it's not
             * tightly packed (i.e. it doesn't have the form '2^n - 1' for
             * some n).  Combine ce0 with the given dispatch (or vector) mask
             * to mask off those channels which were never dispatched by the
             * hardware.
             */
            brw_SHR(p, vec1(dst), mask, brw_imm_ud(qtr_control * 8));
            brw_AND(p, vec1(dst), exec_mask, vec1(dst));
            exec_mask = vec1(dst);
         }

         /* Quarter control has the effect of magically shifting the value of
          * ce0 so you'll get the first active channel relative to the
          * specified quarter control as result.
          */
         inst = brw_FBL(p, vec1(dst), exec_mask);
      } else {
         const struct brw_reg flag = brw_flag_reg(p->current->flag_subreg / 2,
                                                  p->current->flag_subreg % 2);

         brw_set_default_exec_size(p, BRW_EXECUTE_1);
         brw_MOV(p, retype(flag, BRW_REGISTER_TYPE_UD), brw_imm_ud(0));

         /* Run enough instructions returning zero with execution masking and
          * a conditional modifier enabled in order to get the full execution
          * mask in f1.0.  We could use a single 32-wide move here if it
          * weren't because of the hardware bug that causes channel enables to
          * be applied incorrectly to the second half of 32-wide instructions
          * on Gen7.
          */
         const unsigned lower_size = MIN2(16, exec_size);
         for (unsigned i = 0; i < exec_size / lower_size; i++) {
            inst = brw_MOV(p, retype(brw_null_reg(), BRW_REGISTER_TYPE_UW),
                           brw_imm_uw(0));
            brw_inst_set_mask_control(devinfo, inst, BRW_MASK_ENABLE);
            brw_inst_set_group(devinfo, inst, lower_size * i + 8 * qtr_control);
            brw_inst_set_cond_modifier(devinfo, inst, BRW_CONDITIONAL_Z);
            brw_inst_set_exec_size(devinfo, inst, cvt(lower_size) - 1);
         }

         /* Find the first bit set in the exec_size-wide portion of the flag
          * register that was updated by the last sequence of MOV
          * instructions.
          */
         const enum brw_reg_type type = brw_int_type(exec_size / 8, false);
         brw_set_default_exec_size(p, BRW_EXECUTE_1);
         brw_FBL(p, vec1(dst), byte_offset(retype(flag, type), qtr_control));
      }
   } else {
      brw_set_default_mask_control(p, BRW_MASK_DISABLE);

      if (devinfo->gen >= 8 &&
          mask.file == BRW_IMMEDIATE_VALUE && mask.ud == 0xffffffff) {
         /* In SIMD4x2 mode the first active channel index is just the
          * negation of the first bit of the mask register.  Note that ce0
          * doesn't take into account the dispatch mask, so the Gen7 path
          * should be used instead unless you have the guarantee that the
          * dispatch mask is tightly packed (i.e. it has the form '2^n - 1'
          * for some n).
          */
         inst = brw_AND(p, brw_writemask(dst, WRITEMASK_X),
                        negate(retype(brw_mask_reg(0), BRW_REGISTER_TYPE_UD)),
                        brw_imm_ud(1));

      } else {
         /* Overwrite the destination without and with execution masking to
          * find out which of the channels is active.
          */
         brw_push_insn_state(p);
         brw_set_default_exec_size(p, BRW_EXECUTE_4);
         brw_MOV(p, brw_writemask(vec4(dst), WRITEMASK_X),
                 brw_imm_ud(1));

         inst = brw_MOV(p, brw_writemask(vec4(dst), WRITEMASK_X),
                        brw_imm_ud(0));
         brw_pop_insn_state(p);
         brw_inst_set_mask_control(devinfo, inst, BRW_MASK_ENABLE);
      }
   }

   brw_pop_insn_state(p);
}

void
brw_broadcast(struct brw_codegen *p,
              struct brw_reg dst,
              struct brw_reg src,
              struct brw_reg idx)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const bool align1 = brw_get_default_access_mode(p) == BRW_ALIGN_1;
   brw_inst *inst;

   brw_push_insn_state(p);
   brw_set_default_mask_control(p, BRW_MASK_DISABLE);
   brw_set_default_exec_size(p, align1 ? BRW_EXECUTE_1 : BRW_EXECUTE_4);

   assert(src.file == BRW_GENERAL_REGISTER_FILE &&
          src.address_mode == BRW_ADDRESS_DIRECT);
   assert(!src.abs && !src.negate);
   assert(src.type == dst.type);

   if ((src.vstride == 0 && (src.hstride == 0 || !align1)) ||
       idx.file == BRW_IMMEDIATE_VALUE) {
      /* Trivial, the source is already uniform or the index is a constant.
       * We will typically not get here if the optimizer is doing its job, but
       * asserting would be mean.
       */
      const unsigned i = idx.file == BRW_IMMEDIATE_VALUE ? idx.ud : 0;
      brw_MOV(p, dst,
              (align1 ? stride(suboffset(src, i), 0, 1, 0) :
               stride(suboffset(src, 4 * i), 0, 4, 1)));
   } else {
      /* From the Haswell PRM section "Register Region Restrictions":
       *
       *    "The lower bits of the AddressImmediate must not overflow to
       *    change the register address.  The lower 5 bits of Address
       *    Immediate when added to lower 5 bits of address register gives
       *    the sub-register offset. The upper bits of Address Immediate
       *    when added to upper bits of address register gives the register
       *    address. Any overflow from sub-register offset is dropped."
       *
       * Fortunately, for broadcast, we never have a sub-register offset so
       * this isn't an issue.
       */
      assert(src.subnr == 0);

      if (align1) {
         const struct brw_reg addr =
            retype(brw_address_reg(0), BRW_REGISTER_TYPE_UD);
         unsigned offset = src.nr * REG_SIZE + src.subnr;
         /* Limit in bytes of the signed indirect addressing immediate. */
         const unsigned limit = 512;

         brw_push_insn_state(p);
         brw_set_default_mask_control(p, BRW_MASK_DISABLE);
         brw_set_default_predicate_control(p, BRW_PREDICATE_NONE);

         /* Take into account the component size and horizontal stride. */
         assert(src.vstride == src.hstride + src.width);
         brw_SHL(p, addr, vec1(idx),
                 brw_imm_ud(_mesa_logbase2(type_sz(src.type)) +
                            src.hstride - 1));

         /* We can only address up to limit bytes using the indirect
          * addressing immediate, account for the difference if the source
          * register is above this limit.
          */
         if (offset >= limit) {
            brw_ADD(p, addr, addr, brw_imm_ud(offset - offset % limit));
            offset = offset % limit;
         }

         brw_pop_insn_state(p);

         /* Use indirect addressing to fetch the specified component. */
         if (type_sz(src.type) > 4 &&
             (devinfo->is_cherryview || gen_device_info_is_9lp(devinfo))) {
            /* From the Cherryview PRM Vol 7. "Register Region Restrictions":
             *
             *    "When source or destination datatype is 64b or operation is
             *    integer DWord multiply, indirect addressing must not be
             *    used."
             *
             * To work around both of this issue, we do two integer MOVs
             * insead of one 64-bit MOV.  Because no double value should ever
             * cross a register boundary, it's safe to use the immediate
             * offset in the indirect here to handle adding 4 bytes to the
             * offset and avoid the extra ADD to the register file.
             */
            brw_MOV(p, subscript(dst, BRW_REGISTER_TYPE_D, 0),
                       retype(brw_vec1_indirect(addr.subnr, offset),
                              BRW_REGISTER_TYPE_D));
            brw_MOV(p, subscript(dst, BRW_REGISTER_TYPE_D, 1),
                       retype(brw_vec1_indirect(addr.subnr, offset + 4),
                              BRW_REGISTER_TYPE_D));
         } else {
            brw_MOV(p, dst,
                    retype(brw_vec1_indirect(addr.subnr, offset), src.type));
         }
      } else {
         /* In SIMD4x2 mode the index can be either zero or one, replicate it
          * to all bits of a flag register,
          */
         inst = brw_MOV(p,
                        brw_null_reg(),
                        stride(brw_swizzle(idx, BRW_SWIZZLE_XXXX), 4, 4, 1));
         brw_inst_set_pred_control(devinfo, inst, BRW_PREDICATE_NONE);
         brw_inst_set_cond_modifier(devinfo, inst, BRW_CONDITIONAL_NZ);
         brw_inst_set_flag_reg_nr(devinfo, inst, 1);

         /* and use predicated SEL to pick the right channel. */
         inst = brw_SEL(p, dst,
                        stride(suboffset(src, 4), 4, 4, 1),
                        stride(src, 4, 4, 1));
         brw_inst_set_pred_control(devinfo, inst, BRW_PREDICATE_NORMAL);
         brw_inst_set_flag_reg_nr(devinfo, inst, 1);
      }
   }

   brw_pop_insn_state(p);
}

/**
 * This instruction is generated as a single-channel align1 instruction by
 * both the VS and FS stages when using INTEL_DEBUG=shader_time.
 *
 * We can't use the typed atomic op in the FS because that has the execution
 * mask ANDed with the pixel mask, but we just want to write the one dword for
 * all the pixels.
 *
 * We don't use the SIMD4x2 atomic ops in the VS because want to just write
 * one u32.  So we use the same untyped atomic write message as the pixel
 * shader.
 *
 * The untyped atomic operation requires a BUFFER surface type with RAW
 * format, and is only accessible through the legacy DATA_CACHE dataport
 * messages.
 */
void brw_shader_time_add(struct brw_codegen *p,
                         struct brw_reg payload,
                         uint32_t surf_index)
{
   const struct gen_device_info *devinfo = p->devinfo;
   const unsigned sfid = (devinfo->gen >= 8 || devinfo->is_haswell ?
                          HSW_SFID_DATAPORT_DATA_CACHE_1 :
                          GEN7_SFID_DATAPORT_DATA_CACHE);
   assert(devinfo->gen >= 7);

   brw_push_insn_state(p);
   brw_set_default_access_mode(p, BRW_ALIGN_1);
   brw_set_default_mask_control(p, BRW_MASK_DISABLE);
   brw_set_default_compression_control(p, BRW_COMPRESSION_NONE);
   brw_inst *send = brw_next_insn(p, BRW_OPCODE_SEND);

   /* We use brw_vec1_reg and unmasked because we want to increment the given
    * offset only once.
    */
   brw_set_dest(p, send, brw_vec1_reg(BRW_ARCHITECTURE_REGISTER_FILE,
                                      BRW_ARF_NULL, 0));
   brw_set_src0(p, send, brw_vec1_reg(payload.file,
                                      payload.nr, 0));
   brw_set_desc(p, send, (brw_message_desc(devinfo, 2, 0, false) |
                          brw_dp_untyped_atomic_desc(p, BRW_AOP_ADD, false)));

   brw_inst_set_sfid(devinfo, send, sfid);
   brw_inst_set_binding_table_index(devinfo, send, surf_index);

   brw_pop_insn_state(p);
}


/**
 * Emit the SEND message for a barrier
 */
void
brw_barrier(struct brw_codegen *p, struct brw_reg src)
{
   const struct gen_device_info *devinfo = p->devinfo;
   struct brw_inst *inst;

   assert(devinfo->gen >= 7);

   brw_push_insn_state(p);
   brw_set_default_access_mode(p, BRW_ALIGN_1);
   inst = next_insn(p, BRW_OPCODE_SEND);
   brw_set_dest(p, inst, retype(brw_null_reg(), BRW_REGISTER_TYPE_UW));
   brw_set_src0(p, inst, src);
   brw_set_src1(p, inst, brw_null_reg());
   brw_set_desc(p, inst, brw_message_desc(devinfo, 1, 0, false));

   brw_inst_set_sfid(devinfo, inst, BRW_SFID_MESSAGE_GATEWAY);
   brw_inst_set_gateway_notify(devinfo, inst, 1);
   brw_inst_set_gateway_subfuncid(devinfo, inst,
                                  BRW_MESSAGE_GATEWAY_SFID_BARRIER_MSG);

   brw_inst_set_mask_control(devinfo, inst, BRW_MASK_DISABLE);
   brw_pop_insn_state(p);
}


/**
 * Emit the wait instruction for a barrier
 */
void
brw_WAIT(struct brw_codegen *p)
{
   const struct gen_device_info *devinfo = p->devinfo;
   struct brw_inst *insn;

   struct brw_reg src = brw_notification_reg();

   insn = next_insn(p, BRW_OPCODE_WAIT);
   brw_set_dest(p, insn, src);
   brw_set_src0(p, insn, src);
   brw_set_src1(p, insn, brw_null_reg());

   brw_inst_set_exec_size(devinfo, insn, BRW_EXECUTE_1);
   brw_inst_set_mask_control(devinfo, insn, BRW_MASK_DISABLE);
}

/**
 * Changes the floating point rounding mode updating the control register
 * field defined at cr0.0[5-6] bits. This function supports the changes to
 * RTNE (00), RU (01), RD (10) and RTZ (11) rounding using bitwise operations.
 * Only RTNE and RTZ rounding are enabled at nir.
 */
void
brw_rounding_mode(struct brw_codegen *p,
                  enum brw_rnd_mode mode)
{
   const unsigned bits = mode << BRW_CR0_RND_MODE_SHIFT;

   if (bits != BRW_CR0_RND_MODE_MASK) {
      brw_inst *inst = brw_AND(p, brw_cr0_reg(0), brw_cr0_reg(0),
                               brw_imm_ud(~BRW_CR0_RND_MODE_MASK));
      brw_inst_set_exec_size(p->devinfo, inst, BRW_EXECUTE_1);

      /* From the Skylake PRM, Volume 7, page 760:
       *  "Implementation Restriction on Register Access: When the control
       *   register is used as an explicit source and/or destination, hardware
       *   does not ensure execution pipeline coherency. Software must set the
       *   thread control field to ‘switch’ for an instruction that uses
       *   control register as an explicit operand."
       */
      brw_inst_set_thread_control(p->devinfo, inst, BRW_THREAD_SWITCH);
    }

   if (bits) {
      brw_inst *inst = brw_OR(p, brw_cr0_reg(0), brw_cr0_reg(0),
                              brw_imm_ud(bits));
      brw_inst_set_exec_size(p->devinfo, inst, BRW_EXECUTE_1);
      brw_inst_set_thread_control(p->devinfo, inst, BRW_THREAD_SWITCH);
   }
}
