/*
 * Copyright © 2010 Intel Corporation
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

#include "compiler/glsl/ir.h"
#include "brw_fs.h"
#include "brw_fs_surface_builder.h"
#include "brw_nir.h"
#include "util/u_math.h"

using namespace brw;
using namespace brw::surface_access;

void
fs_visitor::emit_nir_code()
{
   /* emit the arrays used for inputs and outputs - load/store intrinsics will
    * be converted to reads/writes of these arrays
    */
   nir_setup_outputs();
   nir_setup_uniforms();
   nir_emit_system_values();

   nir_emit_impl(nir_shader_get_entrypoint((nir_shader *)nir));
}

void
fs_visitor::nir_setup_outputs()
{
   if (stage == MESA_SHADER_TESS_CTRL || stage == MESA_SHADER_FRAGMENT)
      return;

   unsigned vec4s[VARYING_SLOT_TESS_MAX] = { 0, };

   /* Calculate the size of output registers in a separate pass, before
    * allocating them.  With ARB_enhanced_layouts, multiple output variables
    * may occupy the same slot, but have different type sizes.
    */
   nir_foreach_variable(var, &nir->outputs) {
      const int loc = var->data.driver_location;
      const unsigned var_vec4s =
         var->data.compact ? DIV_ROUND_UP(glsl_get_length(var->type), 4)
                           : type_size_vec4(var->type);
      vec4s[loc] = MAX2(vec4s[loc], var_vec4s);
   }

   for (unsigned loc = 0; loc < ARRAY_SIZE(vec4s);) {
      if (vec4s[loc] == 0) {
         loc++;
         continue;
      }

      unsigned reg_size = vec4s[loc];

      /* Check if there are any ranges that start within this range and extend
       * past it. If so, include them in this allocation.
       */
      for (unsigned i = 1; i < reg_size; i++)
         reg_size = MAX2(vec4s[i + loc] + i, reg_size);

      fs_reg reg = bld.vgrf(BRW_REGISTER_TYPE_F, 4 * reg_size);
      for (unsigned i = 0; i < reg_size; i++)
         outputs[loc + i] = offset(reg, bld, 4 * i);

      loc += reg_size;
   }
}

void
fs_visitor::nir_setup_uniforms()
{
   /* Only the first compile gets to set up uniforms. */
   if (push_constant_loc) {
      assert(pull_constant_loc);
      return;
   }

   uniforms = nir->num_uniforms / 4;

   if (stage == MESA_SHADER_COMPUTE) {
      /* Add a uniform for the thread local id.  It must be the last uniform
       * on the list.
       */
      assert(uniforms == prog_data->nr_params);
      uint32_t *param = brw_stage_prog_data_add_params(prog_data, 1);
      *param = BRW_PARAM_BUILTIN_SUBGROUP_ID;
      subgroup_id = fs_reg(UNIFORM, uniforms++, BRW_REGISTER_TYPE_UD);
   }
}

static bool
emit_system_values_block(nir_block *block, fs_visitor *v)
{
   fs_reg *reg;

   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      switch (intrin->intrinsic) {
      case nir_intrinsic_load_vertex_id:
      case nir_intrinsic_load_base_vertex:
         unreachable("should be lowered by nir_lower_system_values().");

      case nir_intrinsic_load_vertex_id_zero_base:
      case nir_intrinsic_load_is_indexed_draw:
      case nir_intrinsic_load_first_vertex:
      case nir_intrinsic_load_instance_id:
      case nir_intrinsic_load_base_instance:
      case nir_intrinsic_load_draw_id:
         unreachable("should be lowered by brw_nir_lower_vs_inputs().");

      case nir_intrinsic_load_invocation_id:
         if (v->stage == MESA_SHADER_TESS_CTRL)
            break;
         assert(v->stage == MESA_SHADER_GEOMETRY);
         reg = &v->nir_system_values[SYSTEM_VALUE_INVOCATION_ID];
         if (reg->file == BAD_FILE) {
            const fs_builder abld = v->bld.annotate("gl_InvocationID", NULL);
            fs_reg g1(retype(brw_vec8_grf(1, 0), BRW_REGISTER_TYPE_UD));
            fs_reg iid = abld.vgrf(BRW_REGISTER_TYPE_UD, 1);
            abld.SHR(iid, g1, brw_imm_ud(27u));
            *reg = iid;
         }
         break;

      case nir_intrinsic_load_sample_pos:
         assert(v->stage == MESA_SHADER_FRAGMENT);
         reg = &v->nir_system_values[SYSTEM_VALUE_SAMPLE_POS];
         if (reg->file == BAD_FILE)
            *reg = *v->emit_samplepos_setup();
         break;

      case nir_intrinsic_load_sample_id:
         assert(v->stage == MESA_SHADER_FRAGMENT);
         reg = &v->nir_system_values[SYSTEM_VALUE_SAMPLE_ID];
         if (reg->file == BAD_FILE)
            *reg = *v->emit_sampleid_setup();
         break;

      case nir_intrinsic_load_sample_mask_in:
         assert(v->stage == MESA_SHADER_FRAGMENT);
         assert(v->devinfo->gen >= 7);
         reg = &v->nir_system_values[SYSTEM_VALUE_SAMPLE_MASK_IN];
         if (reg->file == BAD_FILE)
            *reg = *v->emit_samplemaskin_setup();
         break;

      case nir_intrinsic_load_work_group_id:
         assert(v->stage == MESA_SHADER_COMPUTE);
         reg = &v->nir_system_values[SYSTEM_VALUE_WORK_GROUP_ID];
         if (reg->file == BAD_FILE)
            *reg = *v->emit_cs_work_group_id_setup();
         break;

      case nir_intrinsic_load_helper_invocation:
         assert(v->stage == MESA_SHADER_FRAGMENT);
         reg = &v->nir_system_values[SYSTEM_VALUE_HELPER_INVOCATION];
         if (reg->file == BAD_FILE) {
            const fs_builder abld =
               v->bld.annotate("gl_HelperInvocation", NULL);

            /* On Gen6+ (gl_HelperInvocation is only exposed on Gen7+) the
             * pixel mask is in g1.7 of the thread payload.
             *
             * We move the per-channel pixel enable bit to the low bit of each
             * channel by shifting the byte containing the pixel mask by the
             * vector immediate 0x76543210UV.
             *
             * The region of <1,8,0> reads only 1 byte (the pixel masks for
             * subspans 0 and 1) in SIMD8 and an additional byte (the pixel
             * masks for 2 and 3) in SIMD16.
             */
            fs_reg shifted = abld.vgrf(BRW_REGISTER_TYPE_UW, 1);

            for (unsigned i = 0; i < DIV_ROUND_UP(v->dispatch_width, 16); i++) {
               const fs_builder hbld = abld.group(MIN2(16, v->dispatch_width), i);
               hbld.SHR(offset(shifted, hbld, i),
                        stride(retype(brw_vec1_grf(1 + i, 7),
                                      BRW_REGISTER_TYPE_UB),
                               1, 8, 0),
                        brw_imm_v(0x76543210));
            }

            /* A set bit in the pixel mask means the channel is enabled, but
             * that is the opposite of gl_HelperInvocation so we need to invert
             * the mask.
             *
             * The negate source-modifier bit of logical instructions on Gen8+
             * performs 1's complement negation, so we can use that instead of
             * a NOT instruction.
             */
            fs_reg inverted = negate(shifted);
            if (v->devinfo->gen < 8) {
               inverted = abld.vgrf(BRW_REGISTER_TYPE_UW);
               abld.NOT(inverted, shifted);
            }

            /* We then resolve the 0/1 result to 0/~0 boolean values by ANDing
             * with 1 and negating.
             */
            fs_reg anded = abld.vgrf(BRW_REGISTER_TYPE_UD, 1);
            abld.AND(anded, inverted, brw_imm_uw(1));

            fs_reg dst = abld.vgrf(BRW_REGISTER_TYPE_D, 1);
            abld.MOV(dst, negate(retype(anded, BRW_REGISTER_TYPE_D)));
            *reg = dst;
         }
         break;

      default:
         break;
      }
   }

   return true;
}

void
fs_visitor::nir_emit_system_values()
{
   nir_system_values = ralloc_array(mem_ctx, fs_reg, SYSTEM_VALUE_MAX);
   for (unsigned i = 0; i < SYSTEM_VALUE_MAX; i++) {
      nir_system_values[i] = fs_reg();
   }

   /* Always emit SUBGROUP_INVOCATION.  Dead code will clean it up if we
    * never end up using it.
    */
   {
      const fs_builder abld = bld.annotate("gl_SubgroupInvocation", NULL);
      fs_reg &reg = nir_system_values[SYSTEM_VALUE_SUBGROUP_INVOCATION];
      reg = abld.vgrf(BRW_REGISTER_TYPE_UW);

      const fs_builder allbld8 = abld.group(8, 0).exec_all();
      allbld8.MOV(reg, brw_imm_v(0x76543210));
      if (dispatch_width > 8)
         allbld8.ADD(byte_offset(reg, 16), reg, brw_imm_uw(8u));
      if (dispatch_width > 16) {
         const fs_builder allbld16 = abld.group(16, 0).exec_all();
         allbld16.ADD(byte_offset(reg, 32), reg, brw_imm_uw(16u));
      }
   }

   nir_function_impl *impl = nir_shader_get_entrypoint((nir_shader *)nir);
   nir_foreach_block(block, impl)
      emit_system_values_block(block, this);
}

/*
 * Returns a type based on a reference_type (word, float, half-float) and a
 * given bit_size.
 *
 * Reference BRW_REGISTER_TYPE are HF,F,DF,W,D,UW,UD.
 *
 * @FIXME: 64-bit return types are always DF on integer types to maintain
 * compability with uses of DF previously to the introduction of int64
 * support.
 */
static brw_reg_type
brw_reg_type_from_bit_size(const unsigned bit_size,
                           const brw_reg_type reference_type)
{
   switch(reference_type) {
   case BRW_REGISTER_TYPE_HF:
   case BRW_REGISTER_TYPE_F:
   case BRW_REGISTER_TYPE_DF:
      switch(bit_size) {
      case 16:
         return BRW_REGISTER_TYPE_HF;
      case 32:
         return BRW_REGISTER_TYPE_F;
      case 64:
         return BRW_REGISTER_TYPE_DF;
      default:
         unreachable("Invalid bit size");
      }
   case BRW_REGISTER_TYPE_B:
   case BRW_REGISTER_TYPE_W:
   case BRW_REGISTER_TYPE_D:
   case BRW_REGISTER_TYPE_Q:
      switch(bit_size) {
      case 8:
         return BRW_REGISTER_TYPE_B;
      case 16:
         return BRW_REGISTER_TYPE_W;
      case 32:
         return BRW_REGISTER_TYPE_D;
      case 64:
         return BRW_REGISTER_TYPE_Q;
      default:
         unreachable("Invalid bit size");
      }
   case BRW_REGISTER_TYPE_UB:
   case BRW_REGISTER_TYPE_UW:
   case BRW_REGISTER_TYPE_UD:
   case BRW_REGISTER_TYPE_UQ:
      switch(bit_size) {
      case 8:
         return BRW_REGISTER_TYPE_UB;
      case 16:
         return BRW_REGISTER_TYPE_UW;
      case 32:
         return BRW_REGISTER_TYPE_UD;
      case 64:
         return BRW_REGISTER_TYPE_UQ;
      default:
         unreachable("Invalid bit size");
      }
   default:
      unreachable("Unknown type");
   }
}

void
fs_visitor::nir_emit_impl(nir_function_impl *impl)
{
   nir_locals = ralloc_array(mem_ctx, fs_reg, impl->reg_alloc);
   for (unsigned i = 0; i < impl->reg_alloc; i++) {
      nir_locals[i] = fs_reg();
   }

   foreach_list_typed(nir_register, reg, node, &impl->registers) {
      unsigned array_elems =
         reg->num_array_elems == 0 ? 1 : reg->num_array_elems;
      unsigned size = array_elems * reg->num_components;
      const brw_reg_type reg_type =
         brw_reg_type_from_bit_size(reg->bit_size, BRW_REGISTER_TYPE_F);
      nir_locals[reg->index] = bld.vgrf(reg_type, size);
   }

   nir_ssa_values = reralloc(mem_ctx, nir_ssa_values, fs_reg,
                             impl->ssa_alloc);

   nir_emit_cf_list(&impl->body);
}

void
fs_visitor::nir_emit_cf_list(exec_list *list)
{
   exec_list_validate(list);
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if:
         nir_emit_if(nir_cf_node_as_if(node));
         break;

      case nir_cf_node_loop:
         nir_emit_loop(nir_cf_node_as_loop(node));
         break;

      case nir_cf_node_block:
         nir_emit_block(nir_cf_node_as_block(node));
         break;

      default:
         unreachable("Invalid CFG node block");
      }
   }
}

void
fs_visitor::nir_emit_if(nir_if *if_stmt)
{
   /* first, put the condition into f0 */
   fs_inst *inst = bld.MOV(bld.null_reg_d(),
                            retype(get_nir_src(if_stmt->condition),
                                   BRW_REGISTER_TYPE_D));
   inst->conditional_mod = BRW_CONDITIONAL_NZ;

   bld.IF(BRW_PREDICATE_NORMAL);

   nir_emit_cf_list(&if_stmt->then_list);

   /* note: if the else is empty, dead CF elimination will remove it */
   bld.emit(BRW_OPCODE_ELSE);

   nir_emit_cf_list(&if_stmt->else_list);

   bld.emit(BRW_OPCODE_ENDIF);

   if (devinfo->gen < 7)
      limit_dispatch_width(16, "Non-uniform control flow unsupported "
                           "in SIMD32 mode.");
}

void
fs_visitor::nir_emit_loop(nir_loop *loop)
{
   bld.emit(BRW_OPCODE_DO);

   nir_emit_cf_list(&loop->body);

   bld.emit(BRW_OPCODE_WHILE);

   if (devinfo->gen < 7)
      limit_dispatch_width(16, "Non-uniform control flow unsupported "
                           "in SIMD32 mode.");
}

void
fs_visitor::nir_emit_block(nir_block *block)
{
   nir_foreach_instr(instr, block) {
      nir_emit_instr(instr);
   }
}

void
fs_visitor::nir_emit_instr(nir_instr *instr)
{
   const fs_builder abld = bld.annotate(NULL, instr);

   switch (instr->type) {
   case nir_instr_type_alu:
      nir_emit_alu(abld, nir_instr_as_alu(instr));
      break;

   case nir_instr_type_deref:
      /* Derefs can exist for images but they do nothing */
      break;

   case nir_instr_type_intrinsic:
      switch (stage) {
      case MESA_SHADER_VERTEX:
         nir_emit_vs_intrinsic(abld, nir_instr_as_intrinsic(instr));
         break;
      case MESA_SHADER_TESS_CTRL:
         nir_emit_tcs_intrinsic(abld, nir_instr_as_intrinsic(instr));
         break;
      case MESA_SHADER_TESS_EVAL:
         nir_emit_tes_intrinsic(abld, nir_instr_as_intrinsic(instr));
         break;
      case MESA_SHADER_GEOMETRY:
         nir_emit_gs_intrinsic(abld, nir_instr_as_intrinsic(instr));
         break;
      case MESA_SHADER_FRAGMENT:
         nir_emit_fs_intrinsic(abld, nir_instr_as_intrinsic(instr));
         break;
      case MESA_SHADER_COMPUTE:
         nir_emit_cs_intrinsic(abld, nir_instr_as_intrinsic(instr));
         break;
      default:
         unreachable("unsupported shader stage");
      }
      break;

   case nir_instr_type_tex:
      nir_emit_texture(abld, nir_instr_as_tex(instr));
      break;

   case nir_instr_type_load_const:
      nir_emit_load_const(abld, nir_instr_as_load_const(instr));
      break;

   case nir_instr_type_ssa_undef:
      /* We create a new VGRF for undefs on every use (by handling
       * them in get_nir_src()), rather than for each definition.
       * This helps register coalescing eliminate MOVs from undef.
       */
      break;

   case nir_instr_type_jump:
      nir_emit_jump(abld, nir_instr_as_jump(instr));
      break;

   default:
      unreachable("unknown instruction type");
   }
}

/**
 * Recognizes a parent instruction of nir_op_extract_* and changes the type to
 * match instr.
 */
bool
fs_visitor::optimize_extract_to_float(nir_alu_instr *instr,
                                      const fs_reg &result)
{
   if (!instr->src[0].src.is_ssa ||
       !instr->src[0].src.ssa->parent_instr)
      return false;

   if (instr->src[0].src.ssa->parent_instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *src0 =
      nir_instr_as_alu(instr->src[0].src.ssa->parent_instr);

   if (src0->op != nir_op_extract_u8 && src0->op != nir_op_extract_u16 &&
       src0->op != nir_op_extract_i8 && src0->op != nir_op_extract_i16)
      return false;

   nir_const_value *element = nir_src_as_const_value(src0->src[1].src);
   assert(element != NULL);

   /* Element type to extract.*/
   const brw_reg_type type = brw_int_type(
      src0->op == nir_op_extract_u16 || src0->op == nir_op_extract_i16 ? 2 : 1,
      src0->op == nir_op_extract_i16 || src0->op == nir_op_extract_i8);

   fs_reg op0 = get_nir_src(src0->src[0].src);
   op0.type = brw_type_for_nir_type(devinfo,
      (nir_alu_type)(nir_op_infos[src0->op].input_types[0] |
                     nir_src_bit_size(src0->src[0].src)));
   op0 = offset(op0, bld, src0->src[0].swizzle[0]);

   set_saturate(instr->dest.saturate,
                bld.MOV(result, subscript(op0, type, element->u32[0])));
   return true;
}

bool
fs_visitor::optimize_frontfacing_ternary(nir_alu_instr *instr,
                                         const fs_reg &result)
{
   if (!instr->src[0].src.is_ssa ||
       instr->src[0].src.ssa->parent_instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *src0 =
      nir_instr_as_intrinsic(instr->src[0].src.ssa->parent_instr);

   if (src0->intrinsic != nir_intrinsic_load_front_face)
      return false;

   nir_const_value *value1 = nir_src_as_const_value(instr->src[1].src);
   if (!value1 || fabsf(value1->f32[0]) != 1.0f)
      return false;

   nir_const_value *value2 = nir_src_as_const_value(instr->src[2].src);
   if (!value2 || fabsf(value2->f32[0]) != 1.0f)
      return false;

   fs_reg tmp = vgrf(glsl_type::int_type);

   if (devinfo->gen >= 6) {
      /* Bit 15 of g0.0 is 0 if the polygon is front facing. */
      fs_reg g0 = fs_reg(retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_W));

      /* For (gl_FrontFacing ? 1.0 : -1.0), emit:
       *
       *    or(8)  tmp.1<2>W  g0.0<0,1,0>W  0x00003f80W
       *    and(8) dst<1>D    tmp<8,8,1>D   0xbf800000D
       *
       * and negate g0.0<0,1,0>W for (gl_FrontFacing ? -1.0 : 1.0).
       *
       * This negation looks like it's safe in practice, because bits 0:4 will
       * surely be TRIANGLES
       */

      if (value1->f32[0] == -1.0f) {
         g0.negate = true;
      }

      bld.OR(subscript(tmp, BRW_REGISTER_TYPE_W, 1),
             g0, brw_imm_uw(0x3f80));
   } else {
      /* Bit 31 of g1.6 is 0 if the polygon is front facing. */
      fs_reg g1_6 = fs_reg(retype(brw_vec1_grf(1, 6), BRW_REGISTER_TYPE_D));

      /* For (gl_FrontFacing ? 1.0 : -1.0), emit:
       *
       *    or(8)  tmp<1>D  g1.6<0,1,0>D  0x3f800000D
       *    and(8) dst<1>D  tmp<8,8,1>D   0xbf800000D
       *
       * and negate g1.6<0,1,0>D for (gl_FrontFacing ? -1.0 : 1.0).
       *
       * This negation looks like it's safe in practice, because bits 0:4 will
       * surely be TRIANGLES
       */

      if (value1->f32[0] == -1.0f) {
         g1_6.negate = true;
      }

      bld.OR(tmp, g1_6, brw_imm_d(0x3f800000));
   }
   bld.AND(retype(result, BRW_REGISTER_TYPE_D), tmp, brw_imm_d(0xbf800000));

   return true;
}

static void
emit_find_msb_using_lzd(const fs_builder &bld,
                        const fs_reg &result,
                        const fs_reg &src,
                        bool is_signed)
{
   fs_inst *inst;
   fs_reg temp = src;

   if (is_signed) {
      /* LZD of an absolute value source almost always does the right
       * thing.  There are two problem values:
       *
       * * 0x80000000.  Since abs(0x80000000) == 0x80000000, LZD returns
       *   0.  However, findMSB(int(0x80000000)) == 30.
       *
       * * 0xffffffff.  Since abs(0xffffffff) == 1, LZD returns
       *   31.  Section 8.8 (Integer Functions) of the GLSL 4.50 spec says:
       *
       *    For a value of zero or negative one, -1 will be returned.
       *
       * * Negative powers of two.  LZD(abs(-(1<<x))) returns x, but
       *   findMSB(-(1<<x)) should return x-1.
       *
       * For all negative number cases, including 0x80000000 and
       * 0xffffffff, the correct value is obtained from LZD if instead of
       * negating the (already negative) value the logical-not is used.  A
       * conditonal logical-not can be achieved in two instructions.
       */
      temp = bld.vgrf(BRW_REGISTER_TYPE_D);

      bld.ASR(temp, src, brw_imm_d(31));
      bld.XOR(temp, temp, src);
   }

   bld.LZD(retype(result, BRW_REGISTER_TYPE_UD),
           retype(temp, BRW_REGISTER_TYPE_UD));

   /* LZD counts from the MSB side, while GLSL's findMSB() wants the count
    * from the LSB side. Subtract the result from 31 to convert the MSB
    * count into an LSB count.  If no bits are set, LZD will return 32.
    * 31-32 = -1, which is exactly what findMSB() is supposed to return.
    */
   inst = bld.ADD(result, retype(result, BRW_REGISTER_TYPE_D), brw_imm_d(31));
   inst->src[0].negate = true;
}

static brw_rnd_mode
brw_rnd_mode_from_nir_op (const nir_op op) {
   switch (op) {
   case nir_op_f2f16_rtz:
      return BRW_RND_MODE_RTZ;
   case nir_op_f2f16_rtne:
      return BRW_RND_MODE_RTNE;
   default:
      unreachable("Operation doesn't support rounding mode");
   }
}

void
fs_visitor::nir_emit_alu(const fs_builder &bld, nir_alu_instr *instr)
{
   struct brw_wm_prog_key *fs_key = (struct brw_wm_prog_key *) this->key;
   fs_inst *inst;

   fs_reg result = get_nir_dest(instr->dest.dest);
   result.type = brw_type_for_nir_type(devinfo,
      (nir_alu_type)(nir_op_infos[instr->op].output_type |
                     nir_dest_bit_size(instr->dest.dest)));

   fs_reg op[4];
   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      op[i] = get_nir_src(instr->src[i].src);
      op[i].type = brw_type_for_nir_type(devinfo,
         (nir_alu_type)(nir_op_infos[instr->op].input_types[i] |
                        nir_src_bit_size(instr->src[i].src)));
      op[i].abs = instr->src[i].abs;
      op[i].negate = instr->src[i].negate;
   }

   /* We get a bunch of mov's out of the from_ssa pass and they may still
    * be vectorized.  We'll handle them as a special-case.  We'll also
    * handle vecN here because it's basically the same thing.
    */
   switch (instr->op) {
   case nir_op_imov:
   case nir_op_fmov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4: {
      fs_reg temp = result;
      bool need_extra_copy = false;
      for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
         if (!instr->src[i].src.is_ssa &&
             instr->dest.dest.reg.reg == instr->src[i].src.reg.reg) {
            need_extra_copy = true;
            temp = bld.vgrf(result.type, 4);
            break;
         }
      }

      for (unsigned i = 0; i < 4; i++) {
         if (!(instr->dest.write_mask & (1 << i)))
            continue;

         if (instr->op == nir_op_imov || instr->op == nir_op_fmov) {
            inst = bld.MOV(offset(temp, bld, i),
                           offset(op[0], bld, instr->src[0].swizzle[i]));
         } else {
            inst = bld.MOV(offset(temp, bld, i),
                           offset(op[i], bld, instr->src[i].swizzle[0]));
         }
         inst->saturate = instr->dest.saturate;
      }

      /* In this case the source and destination registers were the same,
       * so we need to insert an extra set of moves in order to deal with
       * any swizzling.
       */
      if (need_extra_copy) {
         for (unsigned i = 0; i < 4; i++) {
            if (!(instr->dest.write_mask & (1 << i)))
               continue;

            bld.MOV(offset(result, bld, i), offset(temp, bld, i));
         }
      }
      return;
   }
   default:
      break;
   }

   /* At this point, we have dealt with any instruction that operates on
    * more than a single channel.  Therefore, we can just adjust the source
    * and destination registers for that channel and emit the instruction.
    */
   unsigned channel = 0;
   if (nir_op_infos[instr->op].output_size == 0) {
      /* Since NIR is doing the scalarizing for us, we should only ever see
       * vectorized operations with a single channel.
       */
      assert(util_bitcount(instr->dest.write_mask) == 1);
      channel = ffs(instr->dest.write_mask) - 1;

      result = offset(result, bld, channel);
   }

   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      assert(nir_op_infos[instr->op].input_sizes[i] < 2);
      op[i] = offset(op[i], bld, instr->src[i].swizzle[channel]);
   }

   switch (instr->op) {
   case nir_op_i2f32:
   case nir_op_u2f32:
      if (optimize_extract_to_float(instr, result))
         return;
      inst = bld.MOV(result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_f2f16_rtne:
   case nir_op_f2f16_rtz:
      bld.emit(SHADER_OPCODE_RND_MODE, bld.null_reg_ud(),
               brw_imm_d(brw_rnd_mode_from_nir_op(instr->op)));
      /* fallthrough */

      /* In theory, it would be better to use BRW_OPCODE_F32TO16. Depending
       * on the HW gen, it is a special hw opcode or just a MOV, and
       * brw_F32TO16 (at brw_eu_emit) would do the work to chose.
       *
       * But if we want to use that opcode, we need to provide support on
       * different optimizations and lowerings. As right now HF support is
       * only for gen8+, it will be better to use directly the MOV, and use
       * BRW_OPCODE_F32TO16 when/if we work for HF support on gen7.
       */

   case nir_op_f2f16:
      inst = bld.MOV(result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_b2i:
   case nir_op_b2f:
      op[0].type = BRW_REGISTER_TYPE_D;
      op[0].negate = !op[0].negate;
      /* fallthrough */
   case nir_op_f2f64:
   case nir_op_f2i64:
   case nir_op_f2u64:
   case nir_op_i2f64:
   case nir_op_i2i64:
   case nir_op_u2f64:
   case nir_op_u2u64:
      /* CHV PRM, vol07, 3D Media GPGPU Engine, Register Region Restrictions:
       *
       *    "When source or destination is 64b (...), regioning in Align1
       *     must follow these rules:
       *
       *     1. Source and destination horizontal stride must be aligned to
       *        the same qword.
       *     (...)"
       *
       * This means that conversions from bit-sizes smaller than 64-bit to
       * 64-bit need to have the source data elements aligned to 64-bit.
       * This restriction does not apply to BDW and later.
       */
      if (nir_dest_bit_size(instr->dest.dest) == 64 &&
          nir_src_bit_size(instr->src[0].src) < 64 &&
          (devinfo->is_cherryview || gen_device_info_is_9lp(devinfo))) {
         fs_reg tmp = bld.vgrf(result.type, 1);
         tmp = subscript(tmp, op[0].type, 0);
         inst = bld.MOV(tmp, op[0]);
         inst = bld.MOV(result, tmp);
         inst->saturate = instr->dest.saturate;
         break;
      }
      /* fallthrough */
   case nir_op_f2f32:
   case nir_op_f2i32:
   case nir_op_f2u32:
   case nir_op_f2i16:
   case nir_op_f2u16:
   case nir_op_i2i32:
   case nir_op_u2u32:
   case nir_op_i2i16:
   case nir_op_u2u16:
   case nir_op_i2f16:
   case nir_op_u2f16:
   case nir_op_i2i8:
   case nir_op_u2u8:
      inst = bld.MOV(result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fsign: {
      assert(!instr->dest.saturate);
      if (op[0].abs) {
         /* Straightforward since the source can be assumed to be either
          * strictly >= 0 or strictly <= 0 depending on the setting of the
          * negate flag.
          */
         set_condmod(BRW_CONDITIONAL_NZ, bld.MOV(result, op[0]));

         inst = (op[0].negate)
            ? bld.MOV(result, brw_imm_f(-1.0f))
            : bld.MOV(result, brw_imm_f(1.0f));

         set_predicate(BRW_PREDICATE_NORMAL, inst);
      } else if (type_sz(op[0].type) < 8) {
         /* AND(val, 0x80000000) gives the sign bit.
          *
          * Predicated OR ORs 1.0 (0x3f800000) with the sign bit if val is not
          * zero.
          */
         bld.CMP(bld.null_reg_f(), op[0], brw_imm_f(0.0f), BRW_CONDITIONAL_NZ);

         fs_reg result_int = retype(result, BRW_REGISTER_TYPE_UD);
         op[0].type = BRW_REGISTER_TYPE_UD;
         result.type = BRW_REGISTER_TYPE_UD;
         bld.AND(result_int, op[0], brw_imm_ud(0x80000000u));

         inst = bld.OR(result_int, result_int, brw_imm_ud(0x3f800000u));
         inst->predicate = BRW_PREDICATE_NORMAL;
      } else {
         /* For doubles we do the same but we need to consider:
          *
          * - 2-src instructions can't operate with 64-bit immediates
          * - The sign is encoded in the high 32-bit of each DF
          * - We need to produce a DF result.
          */

         fs_reg zero = vgrf(glsl_type::double_type);
         bld.MOV(zero, setup_imm_df(bld, 0.0));
         bld.CMP(bld.null_reg_df(), op[0], zero, BRW_CONDITIONAL_NZ);

         bld.MOV(result, zero);

         fs_reg r = subscript(result, BRW_REGISTER_TYPE_UD, 1);
         bld.AND(r, subscript(op[0], BRW_REGISTER_TYPE_UD, 1),
                 brw_imm_ud(0x80000000u));

         set_predicate(BRW_PREDICATE_NORMAL,
                       bld.OR(r, r, brw_imm_ud(0x3ff00000u)));
      }
      break;
   }

   case nir_op_isign: {
      /*  ASR(val, 31) -> negative val generates 0xffffffff (signed -1).
       *               -> non-negative val generates 0x00000000.
       *  Predicated OR sets 1 if val is positive.
       */
      uint32_t bit_size = nir_dest_bit_size(instr->dest.dest);
      assert(bit_size == 32 || bit_size == 16);

      fs_reg zero = bit_size == 32 ? brw_imm_d(0) : brw_imm_w(0);
      fs_reg one = bit_size == 32 ? brw_imm_d(1) : brw_imm_w(1);
      fs_reg shift = bit_size == 32 ? brw_imm_d(31) : brw_imm_w(15);

      bld.CMP(bld.null_reg_d(), op[0], zero, BRW_CONDITIONAL_G);
      bld.ASR(result, op[0], shift);
      inst = bld.OR(result, result, one);
      inst->predicate = BRW_PREDICATE_NORMAL;
      break;
   }

   case nir_op_frcp:
      inst = bld.emit(SHADER_OPCODE_RCP, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fexp2:
      inst = bld.emit(SHADER_OPCODE_EXP2, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_flog2:
      inst = bld.emit(SHADER_OPCODE_LOG2, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fsin:
      inst = bld.emit(SHADER_OPCODE_SIN, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fcos:
      inst = bld.emit(SHADER_OPCODE_COS, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fddx:
      if (fs_key->high_quality_derivatives) {
         inst = bld.emit(FS_OPCODE_DDX_FINE, result, op[0]);
      } else {
         inst = bld.emit(FS_OPCODE_DDX_COARSE, result, op[0]);
      }
      inst->saturate = instr->dest.saturate;
      break;
   case nir_op_fddx_fine:
      inst = bld.emit(FS_OPCODE_DDX_FINE, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;
   case nir_op_fddx_coarse:
      inst = bld.emit(FS_OPCODE_DDX_COARSE, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;
   case nir_op_fddy:
      if (fs_key->high_quality_derivatives) {
         inst = bld.emit(FS_OPCODE_DDY_FINE, result, op[0]);
      } else {
         inst = bld.emit(FS_OPCODE_DDY_COARSE, result, op[0]);
      }
      inst->saturate = instr->dest.saturate;
      break;
   case nir_op_fddy_fine:
      inst = bld.emit(FS_OPCODE_DDY_FINE, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;
   case nir_op_fddy_coarse:
      inst = bld.emit(FS_OPCODE_DDY_COARSE, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_iadd:
   case nir_op_fadd:
      inst = bld.ADD(result, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fmul:
      inst = bld.MUL(result, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_imul:
      assert(nir_dest_bit_size(instr->dest.dest) < 64);
      bld.MUL(result, op[0], op[1]);
      break;

   case nir_op_imul_high:
   case nir_op_umul_high:
      assert(nir_dest_bit_size(instr->dest.dest) < 64);
      bld.emit(SHADER_OPCODE_MULH, result, op[0], op[1]);
      break;

   case nir_op_idiv:
   case nir_op_udiv:
      assert(nir_dest_bit_size(instr->dest.dest) < 64);
      bld.emit(SHADER_OPCODE_INT_QUOTIENT, result, op[0], op[1]);
      break;

   case nir_op_uadd_carry:
      unreachable("Should have been lowered by carry_to_arith().");

   case nir_op_usub_borrow:
      unreachable("Should have been lowered by borrow_to_arith().");

   case nir_op_umod:
   case nir_op_irem:
      /* According to the sign table for INT DIV in the Ivy Bridge PRM, it
       * appears that our hardware just does the right thing for signed
       * remainder.
       */
      assert(nir_dest_bit_size(instr->dest.dest) < 64);
      bld.emit(SHADER_OPCODE_INT_REMAINDER, result, op[0], op[1]);
      break;

   case nir_op_imod: {
      /* Get a regular C-style remainder.  If a % b == 0, set the predicate. */
      bld.emit(SHADER_OPCODE_INT_REMAINDER, result, op[0], op[1]);

      /* Math instructions don't support conditional mod */
      inst = bld.MOV(bld.null_reg_d(), result);
      inst->conditional_mod = BRW_CONDITIONAL_NZ;

      /* Now, we need to determine if signs of the sources are different.
       * When we XOR the sources, the top bit is 0 if they are the same and 1
       * if they are different.  We can then use a conditional modifier to
       * turn that into a predicate.  This leads us to an XOR.l instruction.
       *
       * Technically, according to the PRM, you're not allowed to use .l on a
       * XOR instruction.  However, emperical experiments and Curro's reading
       * of the simulator source both indicate that it's safe.
       */
      fs_reg tmp = bld.vgrf(BRW_REGISTER_TYPE_D);
      inst = bld.XOR(tmp, op[0], op[1]);
      inst->predicate = BRW_PREDICATE_NORMAL;
      inst->conditional_mod = BRW_CONDITIONAL_L;

      /* If the result of the initial remainder operation is non-zero and the
       * two sources have different signs, add in a copy of op[1] to get the
       * final integer modulus value.
       */
      inst = bld.ADD(result, result, op[1]);
      inst->predicate = BRW_PREDICATE_NORMAL;
      break;
   }

   case nir_op_flt:
   case nir_op_fge:
   case nir_op_feq:
   case nir_op_fne: {
      fs_reg dest = result;

      const uint32_t bit_size =  nir_src_bit_size(instr->src[0].src);
      if (bit_size != 32)
         dest = bld.vgrf(op[0].type, 1);

      brw_conditional_mod cond;
      switch (instr->op) {
      case nir_op_flt:
         cond = BRW_CONDITIONAL_L;
         break;
      case nir_op_fge:
         cond = BRW_CONDITIONAL_GE;
         break;
      case nir_op_feq:
         cond = BRW_CONDITIONAL_Z;
         break;
      case nir_op_fne:
         cond = BRW_CONDITIONAL_NZ;
         break;
      default:
         unreachable("bad opcode");
      }

      bld.CMP(dest, op[0], op[1], cond);

      if (bit_size > 32) {
         bld.MOV(result, subscript(dest, BRW_REGISTER_TYPE_UD, 0));
      } else if(bit_size < 32) {
         /* When we convert the result to 32-bit we need to be careful and do
          * it as a signed conversion to get sign extension (for 32-bit true)
          */
         const brw_reg_type src_type =
            brw_reg_type_from_bit_size(bit_size, BRW_REGISTER_TYPE_D);

         bld.MOV(retype(result, BRW_REGISTER_TYPE_D), retype(dest, src_type));
      }
      break;
   }

   case nir_op_ilt:
   case nir_op_ult:
   case nir_op_ige:
   case nir_op_uge:
   case nir_op_ieq:
   case nir_op_ine: {
      fs_reg dest = result;

      const uint32_t bit_size = nir_src_bit_size(instr->src[0].src);
      if (bit_size != 32)
         dest = bld.vgrf(op[0].type, 1);

      brw_conditional_mod cond;
      switch (instr->op) {
      case nir_op_ilt:
      case nir_op_ult:
         cond = BRW_CONDITIONAL_L;
         break;
      case nir_op_ige:
      case nir_op_uge:
         cond = BRW_CONDITIONAL_GE;
         break;
      case nir_op_ieq:
         cond = BRW_CONDITIONAL_Z;
         break;
      case nir_op_ine:
         cond = BRW_CONDITIONAL_NZ;
         break;
      default:
         unreachable("bad opcode");
      }
      bld.CMP(dest, op[0], op[1], cond);

      if (bit_size > 32) {
         bld.MOV(result, subscript(dest, BRW_REGISTER_TYPE_UD, 0));
      } else if (bit_size < 32) {
         /* When we convert the result to 32-bit we need to be careful and do
          * it as a signed conversion to get sign extension (for 32-bit true)
          */
         const brw_reg_type src_type =
            brw_reg_type_from_bit_size(bit_size, BRW_REGISTER_TYPE_D);

         bld.MOV(retype(result, BRW_REGISTER_TYPE_D), retype(dest, src_type));
      }
      break;
   }

   case nir_op_inot:
      if (devinfo->gen >= 8) {
         op[0] = resolve_source_modifiers(op[0]);
      }
      bld.NOT(result, op[0]);
      break;
   case nir_op_ixor:
      if (devinfo->gen >= 8) {
         op[0] = resolve_source_modifiers(op[0]);
         op[1] = resolve_source_modifiers(op[1]);
      }
      bld.XOR(result, op[0], op[1]);
      break;
   case nir_op_ior:
      if (devinfo->gen >= 8) {
         op[0] = resolve_source_modifiers(op[0]);
         op[1] = resolve_source_modifiers(op[1]);
      }
      bld.OR(result, op[0], op[1]);
      break;
   case nir_op_iand:
      if (devinfo->gen >= 8) {
         op[0] = resolve_source_modifiers(op[0]);
         op[1] = resolve_source_modifiers(op[1]);
      }
      bld.AND(result, op[0], op[1]);
      break;

   case nir_op_fdot2:
   case nir_op_fdot3:
   case nir_op_fdot4:
   case nir_op_ball_fequal2:
   case nir_op_ball_iequal2:
   case nir_op_ball_fequal3:
   case nir_op_ball_iequal3:
   case nir_op_ball_fequal4:
   case nir_op_ball_iequal4:
   case nir_op_bany_fnequal2:
   case nir_op_bany_inequal2:
   case nir_op_bany_fnequal3:
   case nir_op_bany_inequal3:
   case nir_op_bany_fnequal4:
   case nir_op_bany_inequal4:
      unreachable("Lowered by nir_lower_alu_reductions");

   case nir_op_fnoise1_1:
   case nir_op_fnoise1_2:
   case nir_op_fnoise1_3:
   case nir_op_fnoise1_4:
   case nir_op_fnoise2_1:
   case nir_op_fnoise2_2:
   case nir_op_fnoise2_3:
   case nir_op_fnoise2_4:
   case nir_op_fnoise3_1:
   case nir_op_fnoise3_2:
   case nir_op_fnoise3_3:
   case nir_op_fnoise3_4:
   case nir_op_fnoise4_1:
   case nir_op_fnoise4_2:
   case nir_op_fnoise4_3:
   case nir_op_fnoise4_4:
      unreachable("not reached: should be handled by lower_noise");

   case nir_op_ldexp:
      unreachable("not reached: should be handled by ldexp_to_arith()");

   case nir_op_fsqrt:
      inst = bld.emit(SHADER_OPCODE_SQRT, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_frsq:
      inst = bld.emit(SHADER_OPCODE_RSQ, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_i2b:
   case nir_op_f2b: {
      uint32_t bit_size = nir_src_bit_size(instr->src[0].src);
      if (bit_size == 64) {
         /* two-argument instructions can't take 64-bit immediates */
         fs_reg zero;
         fs_reg tmp;

         if (instr->op == nir_op_f2b) {
            zero = vgrf(glsl_type::double_type);
            tmp = vgrf(glsl_type::double_type);
            bld.MOV(zero, setup_imm_df(bld, 0.0));
         } else {
            zero = vgrf(glsl_type::int64_t_type);
            tmp = vgrf(glsl_type::int64_t_type);
            bld.MOV(zero, brw_imm_q(0));
         }

         /* A SIMD16 execution needs to be split in two instructions, so use
          * a vgrf instead of the flag register as dst so instruction splitting
          * works
          */
         bld.CMP(tmp, op[0], zero, BRW_CONDITIONAL_NZ);
         bld.MOV(result, subscript(tmp, BRW_REGISTER_TYPE_UD, 0));
      } else {
         fs_reg zero;
         if (bit_size == 32) {
            zero = instr->op == nir_op_f2b ? brw_imm_f(0.0f) : brw_imm_d(0);
         } else {
            assert(bit_size == 16);
            zero = instr->op == nir_op_f2b ?
               retype(brw_imm_w(0), BRW_REGISTER_TYPE_HF) : brw_imm_w(0);
         }
         bld.CMP(result, op[0], zero, BRW_CONDITIONAL_NZ);
      }
      break;
   }

   case nir_op_ftrunc:
      inst = bld.RNDZ(result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fceil: {
      op[0].negate = !op[0].negate;
      fs_reg temp = vgrf(glsl_type::float_type);
      bld.RNDD(temp, op[0]);
      temp.negate = true;
      inst = bld.MOV(result, temp);
      inst->saturate = instr->dest.saturate;
      break;
   }
   case nir_op_ffloor:
      inst = bld.RNDD(result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;
   case nir_op_ffract:
      inst = bld.FRC(result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;
   case nir_op_fround_even:
      inst = bld.RNDE(result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_fquantize2f16: {
      fs_reg tmp16 = bld.vgrf(BRW_REGISTER_TYPE_D);
      fs_reg tmp32 = bld.vgrf(BRW_REGISTER_TYPE_F);
      fs_reg zero = bld.vgrf(BRW_REGISTER_TYPE_F);

      /* The destination stride must be at least as big as the source stride. */
      tmp16.type = BRW_REGISTER_TYPE_W;
      tmp16.stride = 2;

      /* Check for denormal */
      fs_reg abs_src0 = op[0];
      abs_src0.abs = true;
      bld.CMP(bld.null_reg_f(), abs_src0, brw_imm_f(ldexpf(1.0, -14)),
              BRW_CONDITIONAL_L);
      /* Get the appropriately signed zero */
      bld.AND(retype(zero, BRW_REGISTER_TYPE_UD),
              retype(op[0], BRW_REGISTER_TYPE_UD),
              brw_imm_ud(0x80000000));
      /* Do the actual F32 -> F16 -> F32 conversion */
      bld.emit(BRW_OPCODE_F32TO16, tmp16, op[0]);
      bld.emit(BRW_OPCODE_F16TO32, tmp32, tmp16);
      /* Select that or zero based on normal status */
      inst = bld.SEL(result, zero, tmp32);
      inst->predicate = BRW_PREDICATE_NORMAL;
      inst->saturate = instr->dest.saturate;
      break;
   }

   case nir_op_imin:
   case nir_op_umin:
   case nir_op_fmin:
      inst = bld.emit_minmax(result, op[0], op[1], BRW_CONDITIONAL_L);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_imax:
   case nir_op_umax:
   case nir_op_fmax:
      inst = bld.emit_minmax(result, op[0], op[1], BRW_CONDITIONAL_GE);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_pack_snorm_2x16:
   case nir_op_pack_snorm_4x8:
   case nir_op_pack_unorm_2x16:
   case nir_op_pack_unorm_4x8:
   case nir_op_unpack_snorm_2x16:
   case nir_op_unpack_snorm_4x8:
   case nir_op_unpack_unorm_2x16:
   case nir_op_unpack_unorm_4x8:
   case nir_op_unpack_half_2x16:
   case nir_op_pack_half_2x16:
      unreachable("not reached: should be handled by lower_packing_builtins");

   case nir_op_unpack_half_2x16_split_x:
      inst = bld.emit(FS_OPCODE_UNPACK_HALF_2x16_SPLIT_X, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;
   case nir_op_unpack_half_2x16_split_y:
      inst = bld.emit(FS_OPCODE_UNPACK_HALF_2x16_SPLIT_Y, result, op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_pack_64_2x32_split:
   case nir_op_pack_32_2x16_split:
      bld.emit(FS_OPCODE_PACK, result, op[0], op[1]);
      break;

   case nir_op_unpack_64_2x32_split_x:
   case nir_op_unpack_64_2x32_split_y: {
      if (instr->op == nir_op_unpack_64_2x32_split_x)
         bld.MOV(result, subscript(op[0], BRW_REGISTER_TYPE_UD, 0));
      else
         bld.MOV(result, subscript(op[0], BRW_REGISTER_TYPE_UD, 1));
      break;
   }

   case nir_op_unpack_32_2x16_split_x:
   case nir_op_unpack_32_2x16_split_y: {
      if (instr->op == nir_op_unpack_32_2x16_split_x)
         bld.MOV(result, subscript(op[0], BRW_REGISTER_TYPE_UW, 0));
      else
         bld.MOV(result, subscript(op[0], BRW_REGISTER_TYPE_UW, 1));
      break;
   }

   case nir_op_fpow:
      inst = bld.emit(SHADER_OPCODE_POW, result, op[0], op[1]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_bitfield_reverse:
      assert(nir_dest_bit_size(instr->dest.dest) < 64);
      bld.BFREV(result, op[0]);
      break;

   case nir_op_bit_count:
      assert(nir_dest_bit_size(instr->dest.dest) < 64);
      bld.CBIT(result, op[0]);
      break;

   case nir_op_ufind_msb: {
      assert(nir_dest_bit_size(instr->dest.dest) < 64);
      emit_find_msb_using_lzd(bld, result, op[0], false);
      break;
   }

   case nir_op_ifind_msb: {
      assert(nir_dest_bit_size(instr->dest.dest) < 64);

      if (devinfo->gen < 7) {
         emit_find_msb_using_lzd(bld, result, op[0], true);
      } else {
         bld.FBH(retype(result, BRW_REGISTER_TYPE_UD), op[0]);

         /* FBH counts from the MSB side, while GLSL's findMSB() wants the
          * count from the LSB side. If FBH didn't return an error
          * (0xFFFFFFFF), then subtract the result from 31 to convert the MSB
          * count into an LSB count.
          */
         bld.CMP(bld.null_reg_d(), result, brw_imm_d(-1), BRW_CONDITIONAL_NZ);

         inst = bld.ADD(result, result, brw_imm_d(31));
         inst->predicate = BRW_PREDICATE_NORMAL;
         inst->src[0].negate = true;
      }
      break;
   }

   case nir_op_find_lsb:
      assert(nir_dest_bit_size(instr->dest.dest) < 64);

      if (devinfo->gen < 7) {
         fs_reg temp = vgrf(glsl_type::int_type);

         /* (x & -x) generates a value that consists of only the LSB of x.
          * For all powers of 2, findMSB(y) == findLSB(y).
          */
         fs_reg src = retype(op[0], BRW_REGISTER_TYPE_D);
         fs_reg negated_src = src;

         /* One must be negated, and the other must be non-negated.  It
          * doesn't matter which is which.
          */
         negated_src.negate = true;
         src.negate = false;

         bld.AND(temp, src, negated_src);
         emit_find_msb_using_lzd(bld, result, temp, false);
      } else {
         bld.FBL(result, op[0]);
      }
      break;

   case nir_op_ubitfield_extract:
   case nir_op_ibitfield_extract:
      unreachable("should have been lowered");
   case nir_op_ubfe:
   case nir_op_ibfe:
      assert(nir_dest_bit_size(instr->dest.dest) < 64);
      bld.BFE(result, op[2], op[1], op[0]);
      break;
   case nir_op_bfm:
      assert(nir_dest_bit_size(instr->dest.dest) < 64);
      bld.BFI1(result, op[0], op[1]);
      break;
   case nir_op_bfi:
      assert(nir_dest_bit_size(instr->dest.dest) < 64);
      bld.BFI2(result, op[0], op[1], op[2]);
      break;

   case nir_op_bitfield_insert:
      unreachable("not reached: should have been lowered");

   case nir_op_ishl:
   case nir_op_ishr:
   case nir_op_ushr: {
      fs_reg shift_count = op[1];

      if (devinfo->is_cherryview || gen_device_info_is_9lp(devinfo)) {
         if (op[1].file == VGRF &&
             (result.type == BRW_REGISTER_TYPE_Q ||
              result.type == BRW_REGISTER_TYPE_UQ)) {
            shift_count = fs_reg(VGRF, alloc.allocate(dispatch_width / 4),
                                 BRW_REGISTER_TYPE_UD);
            shift_count.stride = 2;
            bld.MOV(shift_count, op[1]);
         }
      }

      switch (instr->op) {
      case nir_op_ishl:
         bld.SHL(result, op[0], shift_count);
         break;
      case nir_op_ishr:
         bld.ASR(result, op[0], shift_count);
         break;
      case nir_op_ushr:
         bld.SHR(result, op[0], shift_count);
         break;
      default:
         unreachable("not reached");
      }
      break;
   }

   case nir_op_pack_half_2x16_split:
      bld.emit(FS_OPCODE_PACK_HALF_2x16_SPLIT, result, op[0], op[1]);
      break;

   case nir_op_ffma:
      inst = bld.MAD(result, op[2], op[1], op[0]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_flrp:
      inst = bld.LRP(result, op[0], op[1], op[2]);
      inst->saturate = instr->dest.saturate;
      break;

   case nir_op_bcsel:
      if (optimize_frontfacing_ternary(instr, result))
         return;

      bld.CMP(bld.null_reg_d(), op[0], brw_imm_d(0), BRW_CONDITIONAL_NZ);
      inst = bld.SEL(result, op[1], op[2]);
      inst->predicate = BRW_PREDICATE_NORMAL;
      break;

   case nir_op_extract_u8:
   case nir_op_extract_i8: {
      nir_const_value *byte = nir_src_as_const_value(instr->src[1].src);
      assert(byte != NULL);

      /* The PRMs say:
       *
       *    BDW+
       *    There is no direct conversion from B/UB to Q/UQ or Q/UQ to B/UB.
       *    Use two instructions and a word or DWord intermediate integer type.
       */
      if (nir_dest_bit_size(instr->dest.dest) == 64) {
         const brw_reg_type type = brw_int_type(2, instr->op == nir_op_extract_i8);

         if (instr->op == nir_op_extract_i8) {
            /* If we need to sign extend, extract to a word first */
            fs_reg w_temp = bld.vgrf(BRW_REGISTER_TYPE_W);
            bld.MOV(w_temp, subscript(op[0], type, byte->u32[0]));
            bld.MOV(result, w_temp);
         } else {
            /* Otherwise use an AND with 0xff and a word type */
            bld.AND(result, subscript(op[0], type, byte->u32[0] / 2), brw_imm_uw(0xff));
         }
      } else {
         const brw_reg_type type = brw_int_type(1, instr->op == nir_op_extract_i8);
         bld.MOV(result, subscript(op[0], type, byte->u32[0]));
      }
      break;
   }

   case nir_op_extract_u16:
   case nir_op_extract_i16: {
      const brw_reg_type type = brw_int_type(2, instr->op == nir_op_extract_i16);
      nir_const_value *word = nir_src_as_const_value(instr->src[1].src);
      assert(word != NULL);
      bld.MOV(result, subscript(op[0], type, word->u32[0]));
      break;
   }

   default:
      unreachable("unhandled instruction");
   }

   /* If we need to do a boolean resolve, replace the result with -(x & 1)
    * to sign extend the low bit to 0/~0
    */
   if (devinfo->gen <= 5 &&
       (instr->instr.pass_flags & BRW_NIR_BOOLEAN_MASK) == BRW_NIR_BOOLEAN_NEEDS_RESOLVE) {
      fs_reg masked = vgrf(glsl_type::int_type);
      bld.AND(masked, result, brw_imm_d(1));
      masked.negate = true;
      bld.MOV(retype(result, BRW_REGISTER_TYPE_D), masked);
   }
}

void
fs_visitor::nir_emit_load_const(const fs_builder &bld,
                                nir_load_const_instr *instr)
{
   const brw_reg_type reg_type =
      brw_reg_type_from_bit_size(instr->def.bit_size, BRW_REGISTER_TYPE_D);
   fs_reg reg = bld.vgrf(reg_type, instr->def.num_components);

   switch (instr->def.bit_size) {
   case 8:
      for (unsigned i = 0; i < instr->def.num_components; i++)
         bld.MOV(offset(reg, bld, i), setup_imm_b(bld, instr->value.i8[i]));
      break;

   case 16:
      for (unsigned i = 0; i < instr->def.num_components; i++)
         bld.MOV(offset(reg, bld, i), brw_imm_w(instr->value.i16[i]));
      break;

   case 32:
      for (unsigned i = 0; i < instr->def.num_components; i++)
         bld.MOV(offset(reg, bld, i), brw_imm_d(instr->value.i32[i]));
      break;

   case 64:
      assert(devinfo->gen >= 7);
      if (devinfo->gen == 7) {
         /* We don't get 64-bit integer types until gen8 */
         for (unsigned i = 0; i < instr->def.num_components; i++) {
            bld.MOV(retype(offset(reg, bld, i), BRW_REGISTER_TYPE_DF),
                    setup_imm_df(bld, instr->value.f64[i]));
         }
      } else {
         for (unsigned i = 0; i < instr->def.num_components; i++)
            bld.MOV(offset(reg, bld, i), brw_imm_q(instr->value.i64[i]));
      }
      break;

   default:
      unreachable("Invalid bit size");
   }

   nir_ssa_values[instr->def.index] = reg;
}

fs_reg
fs_visitor::get_nir_src(const nir_src &src)
{
   fs_reg reg;
   if (src.is_ssa) {
      if (src.ssa->parent_instr->type == nir_instr_type_ssa_undef) {
         const brw_reg_type reg_type =
            brw_reg_type_from_bit_size(src.ssa->bit_size, BRW_REGISTER_TYPE_D);
         reg = bld.vgrf(reg_type, src.ssa->num_components);
      } else {
         reg = nir_ssa_values[src.ssa->index];
      }
   } else {
      /* We don't handle indirects on locals */
      assert(src.reg.indirect == NULL);
      reg = offset(nir_locals[src.reg.reg->index], bld,
                   src.reg.base_offset * src.reg.reg->num_components);
   }

   if (nir_src_bit_size(src) == 64 && devinfo->gen == 7) {
      /* The only 64-bit type available on gen7 is DF, so use that. */
      reg.type = BRW_REGISTER_TYPE_DF;
   } else {
      /* To avoid floating-point denorm flushing problems, set the type by
       * default to an integer type - instructions that need floating point
       * semantics will set this to F if they need to
       */
      reg.type = brw_reg_type_from_bit_size(nir_src_bit_size(src),
                                            BRW_REGISTER_TYPE_D);
   }

   return reg;
}

/**
 * Return an IMM for constants; otherwise call get_nir_src() as normal.
 *
 * This function should not be called on any value which may be 64 bits.
 * We could theoretically support 64-bit on gen8+ but we choose not to
 * because it wouldn't work in general (no gen7 support) and there are
 * enough restrictions in 64-bit immediates that you can't take the return
 * value and treat it the same as the result of get_nir_src().
 */
fs_reg
fs_visitor::get_nir_src_imm(const nir_src &src)
{
   nir_const_value *val = nir_src_as_const_value(src);
   assert(nir_src_bit_size(src) == 32);
   return val ? fs_reg(brw_imm_d(val->i32[0])) : get_nir_src(src);
}

fs_reg
fs_visitor::get_nir_dest(const nir_dest &dest)
{
   if (dest.is_ssa) {
      const brw_reg_type reg_type =
         brw_reg_type_from_bit_size(dest.ssa.bit_size,
                                    dest.ssa.bit_size == 8 ?
                                    BRW_REGISTER_TYPE_D :
                                    BRW_REGISTER_TYPE_F);
      nir_ssa_values[dest.ssa.index] =
         bld.vgrf(reg_type, dest.ssa.num_components);
      return nir_ssa_values[dest.ssa.index];
   } else {
      /* We don't handle indirects on locals */
      assert(dest.reg.indirect == NULL);
      return offset(nir_locals[dest.reg.reg->index], bld,
                    dest.reg.base_offset * dest.reg.reg->num_components);
   }
}

void
fs_visitor::emit_percomp(const fs_builder &bld, const fs_inst &inst,
                         unsigned wr_mask)
{
   for (unsigned i = 0; i < 4; i++) {
      if (!((wr_mask >> i) & 1))
         continue;

      fs_inst *new_inst = new(mem_ctx) fs_inst(inst);
      new_inst->dst = offset(new_inst->dst, bld, i);
      for (unsigned j = 0; j < new_inst->sources; j++)
         if (new_inst->src[j].file == VGRF)
            new_inst->src[j] = offset(new_inst->src[j], bld, i);

      bld.emit(new_inst);
   }
}

static fs_inst *
emit_pixel_interpolater_send(const fs_builder &bld,
                             enum opcode opcode,
                             const fs_reg &dst,
                             const fs_reg &src,
                             const fs_reg &desc,
                             glsl_interp_mode interpolation)
{
   struct brw_wm_prog_data *wm_prog_data =
      brw_wm_prog_data(bld.shader->stage_prog_data);

   fs_inst *inst = bld.emit(opcode, dst, src, desc);
   /* 2 floats per slot returned */
   inst->size_written = 2 * dst.component_size(inst->exec_size);
   inst->pi_noperspective = interpolation == INTERP_MODE_NOPERSPECTIVE;

   wm_prog_data->pulls_bary = true;

   return inst;
}

/**
 * Computes 1 << x, given a D/UD register containing some value x.
 */
static fs_reg
intexp2(const fs_builder &bld, const fs_reg &x)
{
   assert(x.type == BRW_REGISTER_TYPE_UD || x.type == BRW_REGISTER_TYPE_D);

   fs_reg result = bld.vgrf(x.type, 1);
   fs_reg one = bld.vgrf(x.type, 1);

   bld.MOV(one, retype(brw_imm_d(1), one.type));
   bld.SHL(result, one, x);
   return result;
}

void
fs_visitor::emit_gs_end_primitive(const nir_src &vertex_count_nir_src)
{
   assert(stage == MESA_SHADER_GEOMETRY);

   struct brw_gs_prog_data *gs_prog_data = brw_gs_prog_data(prog_data);

   if (gs_compile->control_data_header_size_bits == 0)
      return;

   /* We can only do EndPrimitive() functionality when the control data
    * consists of cut bits.  Fortunately, the only time it isn't is when the
    * output type is points, in which case EndPrimitive() is a no-op.
    */
   if (gs_prog_data->control_data_format !=
       GEN7_GS_CONTROL_DATA_FORMAT_GSCTL_CUT) {
      return;
   }

   /* Cut bits use one bit per vertex. */
   assert(gs_compile->control_data_bits_per_vertex == 1);

   fs_reg vertex_count = get_nir_src(vertex_count_nir_src);
   vertex_count.type = BRW_REGISTER_TYPE_UD;

   /* Cut bit n should be set to 1 if EndPrimitive() was called after emitting
    * vertex n, 0 otherwise.  So all we need to do here is mark bit
    * (vertex_count - 1) % 32 in the cut_bits register to indicate that
    * EndPrimitive() was called after emitting vertex (vertex_count - 1);
    * vec4_gs_visitor::emit_control_data_bits() will take care of the rest.
    *
    * Note that if EndPrimitive() is called before emitting any vertices, this
    * will cause us to set bit 31 of the control_data_bits register to 1.
    * That's fine because:
    *
    * - If max_vertices < 32, then vertex number 31 (zero-based) will never be
    *   output, so the hardware will ignore cut bit 31.
    *
    * - If max_vertices == 32, then vertex number 31 is guaranteed to be the
    *   last vertex, so setting cut bit 31 has no effect (since the primitive
    *   is automatically ended when the GS terminates).
    *
    * - If max_vertices > 32, then the ir_emit_vertex visitor will reset the
    *   control_data_bits register to 0 when the first vertex is emitted.
    */

   const fs_builder abld = bld.annotate("end primitive");

   /* control_data_bits |= 1 << ((vertex_count - 1) % 32) */
   fs_reg prev_count = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
   abld.ADD(prev_count, vertex_count, brw_imm_ud(0xffffffffu));
   fs_reg mask = intexp2(abld, prev_count);
   /* Note: we're relying on the fact that the GEN SHL instruction only pays
    * attention to the lower 5 bits of its second source argument, so on this
    * architecture, 1 << (vertex_count - 1) is equivalent to 1 <<
    * ((vertex_count - 1) % 32).
    */
   abld.OR(this->control_data_bits, this->control_data_bits, mask);
}

void
fs_visitor::emit_gs_control_data_bits(const fs_reg &vertex_count)
{
   assert(stage == MESA_SHADER_GEOMETRY);
   assert(gs_compile->control_data_bits_per_vertex != 0);

   struct brw_gs_prog_data *gs_prog_data = brw_gs_prog_data(prog_data);

   const fs_builder abld = bld.annotate("emit control data bits");
   const fs_builder fwa_bld = bld.exec_all();

   /* We use a single UD register to accumulate control data bits (32 bits
    * for each of the SIMD8 channels).  So we need to write a DWord (32 bits)
    * at a time.
    *
    * Unfortunately, the URB_WRITE_SIMD8 message uses 128-bit (OWord) offsets.
    * We have select a 128-bit group via the Global and Per-Slot Offsets, then
    * use the Channel Mask phase to enable/disable which DWord within that
    * group to write.  (Remember, different SIMD8 channels may have emitted
    * different numbers of vertices, so we may need per-slot offsets.)
    *
    * Channel masking presents an annoying problem: we may have to replicate
    * the data up to 4 times:
    *
    * Msg = Handles, Per-Slot Offsets, Channel Masks, Data, Data, Data, Data.
    *
    * To avoid penalizing shaders that emit a small number of vertices, we
    * can avoid these sometimes: if the size of the control data header is
    * <= 128 bits, then there is only 1 OWord.  All SIMD8 channels will land
    * land in the same 128-bit group, so we can skip per-slot offsets.
    *
    * Similarly, if the control data header is <= 32 bits, there is only one
    * DWord, so we can skip channel masks.
    */
   enum opcode opcode = SHADER_OPCODE_URB_WRITE_SIMD8;

   fs_reg channel_mask, per_slot_offset;

   if (gs_compile->control_data_header_size_bits > 32) {
      opcode = SHADER_OPCODE_URB_WRITE_SIMD8_MASKED;
      channel_mask = vgrf(glsl_type::uint_type);
   }

   if (gs_compile->control_data_header_size_bits > 128) {
      opcode = SHADER_OPCODE_URB_WRITE_SIMD8_MASKED_PER_SLOT;
      per_slot_offset = vgrf(glsl_type::uint_type);
   }

   /* Figure out which DWord we're trying to write to using the formula:
    *
    *    dword_index = (vertex_count - 1) * bits_per_vertex / 32
    *
    * Since bits_per_vertex is a power of two, and is known at compile
    * time, this can be optimized to:
    *
    *    dword_index = (vertex_count - 1) >> (6 - log2(bits_per_vertex))
    */
   if (opcode != SHADER_OPCODE_URB_WRITE_SIMD8) {
      fs_reg dword_index = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
      fs_reg prev_count = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
      abld.ADD(prev_count, vertex_count, brw_imm_ud(0xffffffffu));
      unsigned log2_bits_per_vertex =
         util_last_bit(gs_compile->control_data_bits_per_vertex);
      abld.SHR(dword_index, prev_count, brw_imm_ud(6u - log2_bits_per_vertex));

      if (per_slot_offset.file != BAD_FILE) {
         /* Set the per-slot offset to dword_index / 4, so that we'll write to
          * the appropriate OWord within the control data header.
          */
         abld.SHR(per_slot_offset, dword_index, brw_imm_ud(2u));
      }

      /* Set the channel masks to 1 << (dword_index % 4), so that we'll
       * write to the appropriate DWORD within the OWORD.
       */
      fs_reg channel = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
      fwa_bld.AND(channel, dword_index, brw_imm_ud(3u));
      channel_mask = intexp2(fwa_bld, channel);
      /* Then the channel masks need to be in bits 23:16. */
      fwa_bld.SHL(channel_mask, channel_mask, brw_imm_ud(16u));
   }

   /* Store the control data bits in the message payload and send it. */
   int mlen = 2;
   if (channel_mask.file != BAD_FILE)
      mlen += 4; /* channel masks, plus 3 extra copies of the data */
   if (per_slot_offset.file != BAD_FILE)
      mlen++;

   fs_reg payload = bld.vgrf(BRW_REGISTER_TYPE_UD, mlen);
   fs_reg *sources = ralloc_array(mem_ctx, fs_reg, mlen);
   int i = 0;
   sources[i++] = fs_reg(retype(brw_vec8_grf(1, 0), BRW_REGISTER_TYPE_UD));
   if (per_slot_offset.file != BAD_FILE)
      sources[i++] = per_slot_offset;
   if (channel_mask.file != BAD_FILE)
      sources[i++] = channel_mask;
   while (i < mlen) {
      sources[i++] = this->control_data_bits;
   }

   abld.LOAD_PAYLOAD(payload, sources, mlen, mlen);
   fs_inst *inst = abld.emit(opcode, reg_undef, payload);
   inst->mlen = mlen;
   /* We need to increment Global Offset by 256-bits to make room for
    * Broadwell's extra "Vertex Count" payload at the beginning of the
    * URB entry.  Since this is an OWord message, Global Offset is counted
    * in 128-bit units, so we must set it to 2.
    */
   if (gs_prog_data->static_vertex_count == -1)
      inst->offset = 2;
}

void
fs_visitor::set_gs_stream_control_data_bits(const fs_reg &vertex_count,
                                            unsigned stream_id)
{
   /* control_data_bits |= stream_id << ((2 * (vertex_count - 1)) % 32) */

   /* Note: we are calling this *before* increasing vertex_count, so
    * this->vertex_count == vertex_count - 1 in the formula above.
    */

   /* Stream mode uses 2 bits per vertex */
   assert(gs_compile->control_data_bits_per_vertex == 2);

   /* Must be a valid stream */
   assert(stream_id < MAX_VERTEX_STREAMS);

   /* Control data bits are initialized to 0 so we don't have to set any
    * bits when sending vertices to stream 0.
    */
   if (stream_id == 0)
      return;

   const fs_builder abld = bld.annotate("set stream control data bits", NULL);

   /* reg::sid = stream_id */
   fs_reg sid = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
   abld.MOV(sid, brw_imm_ud(stream_id));

   /* reg:shift_count = 2 * (vertex_count - 1) */
   fs_reg shift_count = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
   abld.SHL(shift_count, vertex_count, brw_imm_ud(1u));

   /* Note: we're relying on the fact that the GEN SHL instruction only pays
    * attention to the lower 5 bits of its second source argument, so on this
    * architecture, stream_id << 2 * (vertex_count - 1) is equivalent to
    * stream_id << ((2 * (vertex_count - 1)) % 32).
    */
   fs_reg mask = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
   abld.SHL(mask, sid, shift_count);
   abld.OR(this->control_data_bits, this->control_data_bits, mask);
}

void
fs_visitor::emit_gs_vertex(const nir_src &vertex_count_nir_src,
                           unsigned stream_id)
{
   assert(stage == MESA_SHADER_GEOMETRY);

   struct brw_gs_prog_data *gs_prog_data = brw_gs_prog_data(prog_data);

   fs_reg vertex_count = get_nir_src(vertex_count_nir_src);
   vertex_count.type = BRW_REGISTER_TYPE_UD;

   /* Haswell and later hardware ignores the "Render Stream Select" bits
    * from the 3DSTATE_STREAMOUT packet when the SOL stage is disabled,
    * and instead sends all primitives down the pipeline for rasterization.
    * If the SOL stage is enabled, "Render Stream Select" is honored and
    * primitives bound to non-zero streams are discarded after stream output.
    *
    * Since the only purpose of primives sent to non-zero streams is to
    * be recorded by transform feedback, we can simply discard all geometry
    * bound to these streams when transform feedback is disabled.
    */
   if (stream_id > 0 && !nir->info.has_transform_feedback_varyings)
      return;

   /* If we're outputting 32 control data bits or less, then we can wait
    * until the shader is over to output them all.  Otherwise we need to
    * output them as we go.  Now is the time to do it, since we're about to
    * output the vertex_count'th vertex, so it's guaranteed that the
    * control data bits associated with the (vertex_count - 1)th vertex are
    * correct.
    */
   if (gs_compile->control_data_header_size_bits > 32) {
      const fs_builder abld =
         bld.annotate("emit vertex: emit control data bits");

      /* Only emit control data bits if we've finished accumulating a batch
       * of 32 bits.  This is the case when:
       *
       *     (vertex_count * bits_per_vertex) % 32 == 0
       *
       * (in other words, when the last 5 bits of vertex_count *
       * bits_per_vertex are 0).  Assuming bits_per_vertex == 2^n for some
       * integer n (which is always the case, since bits_per_vertex is
       * always 1 or 2), this is equivalent to requiring that the last 5-n
       * bits of vertex_count are 0:
       *
       *     vertex_count & (2^(5-n) - 1) == 0
       *
       * 2^(5-n) == 2^5 / 2^n == 32 / bits_per_vertex, so this is
       * equivalent to:
       *
       *     vertex_count & (32 / bits_per_vertex - 1) == 0
       *
       * TODO: If vertex_count is an immediate, we could do some of this math
       *       at compile time...
       */
      fs_inst *inst =
         abld.AND(bld.null_reg_d(), vertex_count,
                  brw_imm_ud(32u / gs_compile->control_data_bits_per_vertex - 1u));
      inst->conditional_mod = BRW_CONDITIONAL_Z;

      abld.IF(BRW_PREDICATE_NORMAL);
      /* If vertex_count is 0, then no control data bits have been
       * accumulated yet, so we can skip emitting them.
       */
      abld.CMP(bld.null_reg_d(), vertex_count, brw_imm_ud(0u),
               BRW_CONDITIONAL_NEQ);
      abld.IF(BRW_PREDICATE_NORMAL);
      emit_gs_control_data_bits(vertex_count);
      abld.emit(BRW_OPCODE_ENDIF);

      /* Reset control_data_bits to 0 so we can start accumulating a new
       * batch.
       *
       * Note: in the case where vertex_count == 0, this neutralizes the
       * effect of any call to EndPrimitive() that the shader may have
       * made before outputting its first vertex.
       */
      inst = abld.MOV(this->control_data_bits, brw_imm_ud(0u));
      inst->force_writemask_all = true;
      abld.emit(BRW_OPCODE_ENDIF);
   }

   emit_urb_writes(vertex_count);

   /* In stream mode we have to set control data bits for all vertices
    * unless we have disabled control data bits completely (which we do
    * do for GL_POINTS outputs that don't use streams).
    */
   if (gs_compile->control_data_header_size_bits > 0 &&
       gs_prog_data->control_data_format ==
          GEN7_GS_CONTROL_DATA_FORMAT_GSCTL_SID) {
      set_gs_stream_control_data_bits(vertex_count, stream_id);
   }
}

void
fs_visitor::emit_gs_input_load(const fs_reg &dst,
                               const nir_src &vertex_src,
                               unsigned base_offset,
                               const nir_src &offset_src,
                               unsigned num_components,
                               unsigned first_component)
{
   struct brw_gs_prog_data *gs_prog_data = brw_gs_prog_data(prog_data);

   nir_const_value *vertex_const = nir_src_as_const_value(vertex_src);
   nir_const_value *offset_const = nir_src_as_const_value(offset_src);
   const unsigned push_reg_count = gs_prog_data->base.urb_read_length * 8;

   /* TODO: figure out push input layout for invocations == 1 */
   /* TODO: make this work with 64-bit inputs */
   if (gs_prog_data->invocations == 1 &&
       type_sz(dst.type) <= 4 &&
       offset_const != NULL && vertex_const != NULL &&
       4 * (base_offset + offset_const->u32[0]) < push_reg_count) {
      int imm_offset = (base_offset + offset_const->u32[0]) * 4 +
                       vertex_const->u32[0] * push_reg_count;
      for (unsigned i = 0; i < num_components; i++) {
         bld.MOV(offset(dst, bld, i),
                 fs_reg(ATTR, imm_offset + i + first_component, dst.type));
      }
      return;
   }

   /* Resort to the pull model.  Ensure the VUE handles are provided. */
   assert(gs_prog_data->base.include_vue_handles);

   unsigned first_icp_handle = gs_prog_data->include_primitive_id ? 3 : 2;
   fs_reg icp_handle = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);

   if (gs_prog_data->invocations == 1) {
      if (vertex_const) {
         /* The vertex index is constant; just select the proper URB handle. */
         icp_handle =
            retype(brw_vec8_grf(first_icp_handle + vertex_const->i32[0], 0),
                   BRW_REGISTER_TYPE_UD);
      } else {
         /* The vertex index is non-constant.  We need to use indirect
          * addressing to fetch the proper URB handle.
          *
          * First, we start with the sequence <7, 6, 5, 4, 3, 2, 1, 0>
          * indicating that channel <n> should read the handle from
          * DWord <n>.  We convert that to bytes by multiplying by 4.
          *
          * Next, we convert the vertex index to bytes by multiplying
          * by 32 (shifting by 5), and add the two together.  This is
          * the final indirect byte offset.
          */
         fs_reg sequence = bld.vgrf(BRW_REGISTER_TYPE_UW, 1);
         fs_reg channel_offsets = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
         fs_reg vertex_offset_bytes = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
         fs_reg icp_offset_bytes = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);

         /* sequence = <7, 6, 5, 4, 3, 2, 1, 0> */
         bld.MOV(sequence, fs_reg(brw_imm_v(0x76543210)));
         /* channel_offsets = 4 * sequence = <28, 24, 20, 16, 12, 8, 4, 0> */
         bld.SHL(channel_offsets, sequence, brw_imm_ud(2u));
         /* Convert vertex_index to bytes (multiply by 32) */
         bld.SHL(vertex_offset_bytes,
                 retype(get_nir_src(vertex_src), BRW_REGISTER_TYPE_UD),
                 brw_imm_ud(5u));
         bld.ADD(icp_offset_bytes, vertex_offset_bytes, channel_offsets);

         /* Use first_icp_handle as the base offset.  There is one register
          * of URB handles per vertex, so inform the register allocator that
          * we might read up to nir->info.gs.vertices_in registers.
          */
         bld.emit(SHADER_OPCODE_MOV_INDIRECT, icp_handle,
                  retype(brw_vec8_grf(first_icp_handle, 0), icp_handle.type),
                  fs_reg(icp_offset_bytes),
                  brw_imm_ud(nir->info.gs.vertices_in * REG_SIZE));
      }
   } else {
      assert(gs_prog_data->invocations > 1);

      if (vertex_const) {
         assert(devinfo->gen >= 9 || vertex_const->i32[0] <= 5);
         bld.MOV(icp_handle,
                 retype(brw_vec1_grf(first_icp_handle +
                                     vertex_const->i32[0] / 8,
                                     vertex_const->i32[0] % 8),
                        BRW_REGISTER_TYPE_UD));
      } else {
         /* The vertex index is non-constant.  We need to use indirect
          * addressing to fetch the proper URB handle.
          *
          */
         fs_reg icp_offset_bytes = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);

         /* Convert vertex_index to bytes (multiply by 4) */
         bld.SHL(icp_offset_bytes,
                 retype(get_nir_src(vertex_src), BRW_REGISTER_TYPE_UD),
                 brw_imm_ud(2u));

         /* Use first_icp_handle as the base offset.  There is one DWord
          * of URB handles per vertex, so inform the register allocator that
          * we might read up to ceil(nir->info.gs.vertices_in / 8) registers.
          */
         bld.emit(SHADER_OPCODE_MOV_INDIRECT, icp_handle,
                  retype(brw_vec8_grf(first_icp_handle, 0), icp_handle.type),
                  fs_reg(icp_offset_bytes),
                  brw_imm_ud(DIV_ROUND_UP(nir->info.gs.vertices_in, 8) *
                             REG_SIZE));
      }
   }

   fs_inst *inst;

   fs_reg tmp_dst = dst;
   fs_reg indirect_offset = get_nir_src(offset_src);
   unsigned num_iterations = 1;
   unsigned orig_num_components = num_components;

   if (type_sz(dst.type) == 8) {
      if (num_components > 2) {
         num_iterations = 2;
         num_components = 2;
      }
      fs_reg tmp = fs_reg(VGRF, alloc.allocate(4), dst.type);
      tmp_dst = tmp;
      first_component = first_component / 2;
   }

   for (unsigned iter = 0; iter < num_iterations; iter++) {
      if (offset_const) {
         /* Constant indexing - use global offset. */
         if (first_component != 0) {
            unsigned read_components = num_components + first_component;
            fs_reg tmp = bld.vgrf(dst.type, read_components);
            inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8, tmp, icp_handle);
            inst->size_written = read_components *
                                 tmp.component_size(inst->exec_size);
            for (unsigned i = 0; i < num_components; i++) {
               bld.MOV(offset(tmp_dst, bld, i),
                       offset(tmp, bld, i + first_component));
            }
         } else {
            inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8, tmp_dst,
                            icp_handle);
            inst->size_written = num_components *
                                 tmp_dst.component_size(inst->exec_size);
         }
         inst->offset = base_offset + offset_const->u32[0];
         inst->mlen = 1;
      } else {
         /* Indirect indexing - use per-slot offsets as well. */
         const fs_reg srcs[] = { icp_handle, indirect_offset };
         unsigned read_components = num_components + first_component;
         fs_reg tmp = bld.vgrf(dst.type, read_components);
         fs_reg payload = bld.vgrf(BRW_REGISTER_TYPE_UD, 2);
         bld.LOAD_PAYLOAD(payload, srcs, ARRAY_SIZE(srcs), 0);
         if (first_component != 0) {
            inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT, tmp,
                            payload);
            inst->size_written = read_components *
                                 tmp.component_size(inst->exec_size);
            for (unsigned i = 0; i < num_components; i++) {
               bld.MOV(offset(tmp_dst, bld, i),
                       offset(tmp, bld, i + first_component));
            }
         } else {
            inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT, tmp_dst,
                         payload);
            inst->size_written = num_components *
                                 tmp_dst.component_size(inst->exec_size);
         }
         inst->offset = base_offset;
         inst->mlen = 2;
      }

      if (type_sz(dst.type) == 8) {
         shuffle_from_32bit_read(bld,
                                 offset(dst, bld, iter * 2),
                                 retype(tmp_dst, BRW_REGISTER_TYPE_D),
                                 0,
                                 num_components);
      }

      if (num_iterations > 1) {
         num_components = orig_num_components - 2;
         if(offset_const) {
            base_offset++;
         } else {
            fs_reg new_indirect = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
            bld.ADD(new_indirect, indirect_offset, brw_imm_ud(1u));
            indirect_offset = new_indirect;
         }
      }
   }
}

fs_reg
fs_visitor::get_indirect_offset(nir_intrinsic_instr *instr)
{
   nir_src *offset_src = nir_get_io_offset_src(instr);
   nir_const_value *const_value = nir_src_as_const_value(*offset_src);

   if (const_value) {
      /* The only constant offset we should find is 0.  brw_nir.c's
       * add_const_offset_to_base() will fold other constant offsets
       * into instr->const_index[0].
       */
      assert(const_value->u32[0] == 0);
      return fs_reg();
   }

   return get_nir_src(*offset_src);
}

static void
do_untyped_vector_read(const fs_builder &bld,
                       const fs_reg dest,
                       const fs_reg surf_index,
                       const fs_reg offset_reg,
                       unsigned num_components)
{
   if (type_sz(dest.type) <= 2) {
      assert(dest.stride == 1);
      boolean is_const_offset = offset_reg.file == BRW_IMMEDIATE_VALUE;

      if (is_const_offset) {
         uint32_t start = offset_reg.ud & ~3;
         uint32_t end = offset_reg.ud + num_components * type_sz(dest.type);
         end = ALIGN(end, 4);
         assert (end - start <= 16);

         /* At this point we have 16-bit component/s that have constant
          * offset aligned to 4-bytes that can be read with untyped_reads.
          * untyped_read message requires 32-bit aligned offsets.
          */
         unsigned first_component = (offset_reg.ud & 3) / type_sz(dest.type);
         unsigned num_components_32bit = (end - start) / 4;

         fs_reg read_result =
            emit_untyped_read(bld, surf_index, brw_imm_ud(start),
                              1 /* dims */,
                              num_components_32bit,
                              BRW_PREDICATE_NONE);
         shuffle_from_32bit_read(bld, dest, read_result, first_component,
                                 num_components);
      } else {
         fs_reg read_offset = bld.vgrf(BRW_REGISTER_TYPE_UD);
         for (unsigned i = 0; i < num_components; i++) {
            if (i == 0) {
               bld.MOV(read_offset, offset_reg);
            } else {
               bld.ADD(read_offset, offset_reg,
                       brw_imm_ud(i * type_sz(dest.type)));
            }
            /* Non constant offsets are not guaranteed to be aligned 32-bits
             * so they are read using one byte_scattered_read message
             * for each component.
             */
            fs_reg read_result =
               emit_byte_scattered_read(bld, surf_index, read_offset,
                                        1 /* dims */, 1,
                                        type_sz(dest.type) * 8 /* bit_size */,
                                        BRW_PREDICATE_NONE);
            bld.MOV(offset(dest, bld, i),
                    subscript (read_result, dest.type, 0));
         }
      }
   } else if (type_sz(dest.type) == 4) {
      fs_reg read_result = emit_untyped_read(bld, surf_index, offset_reg,
                                             1 /* dims */,
                                             num_components,
                                             BRW_PREDICATE_NONE);
      read_result.type = dest.type;
      for (unsigned i = 0; i < num_components; i++)
         bld.MOV(offset(dest, bld, i), offset(read_result, bld, i));
   } else if (type_sz(dest.type) == 8) {
      /* Reading a dvec, so we need to:
       *
       * 1. Multiply num_components by 2, to account for the fact that we
       *    need to read 64-bit components.
       * 2. Shuffle the result of the load to form valid 64-bit elements
       * 3. Emit a second load (for components z/w) if needed.
       */
      fs_reg read_offset = bld.vgrf(BRW_REGISTER_TYPE_UD);
      bld.MOV(read_offset, offset_reg);

      int iters = num_components <= 2 ? 1 : 2;

      /* Load the dvec, the first iteration loads components x/y, the second
       * iteration, if needed, loads components z/w
       */
      for (int it = 0; it < iters; it++) {
         /* Compute number of components to read in this iteration */
         int iter_components = MIN2(2, num_components);
         num_components -= iter_components;

         /* Read. Since this message reads 32-bit components, we need to
          * read twice as many components.
          */
         fs_reg read_result = emit_untyped_read(bld, surf_index, read_offset,
                                                1 /* dims */,
                                                iter_components * 2,
                                                BRW_PREDICATE_NONE);

         /* Shuffle the 32-bit load result into valid 64-bit data */
         shuffle_from_32bit_read(bld, offset(dest, bld, it * 2),
                                 read_result, 0, iter_components);

         bld.ADD(read_offset, read_offset, brw_imm_ud(16));
      }
   } else {
      unreachable("Unsupported type");
   }
}

void
fs_visitor::nir_emit_vs_intrinsic(const fs_builder &bld,
                                  nir_intrinsic_instr *instr)
{
   assert(stage == MESA_SHADER_VERTEX);

   fs_reg dest;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dest = get_nir_dest(instr->dest);

   switch (instr->intrinsic) {
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_base_vertex:
      unreachable("should be lowered by nir_lower_system_values()");

   case nir_intrinsic_load_input: {
      fs_reg src = fs_reg(ATTR, nir_intrinsic_base(instr) * 4, dest.type);
      unsigned first_component = nir_intrinsic_component(instr);
      unsigned num_components = instr->num_components;

      nir_const_value *const_offset = nir_src_as_const_value(instr->src[0]);
      assert(const_offset && "Indirect input loads not allowed");
      src = offset(src, bld, const_offset->u32[0]);

      if (type_sz(dest.type) == 8)
         first_component /= 2;

      /* For 16-bit support maybe a temporary will be needed to copy from
       * the ATTR file.
       */
      shuffle_from_32bit_read(bld, dest, retype(src, BRW_REGISTER_TYPE_D),
                              first_component, num_components);
      break;
   }

   case nir_intrinsic_load_vertex_id_zero_base:
   case nir_intrinsic_load_instance_id:
   case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_draw_id:
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_is_indexed_draw:
      unreachable("lowered by brw_nir_lower_vs_inputs");

   default:
      nir_emit_intrinsic(bld, instr);
      break;
   }
}

void
fs_visitor::nir_emit_tcs_intrinsic(const fs_builder &bld,
                                   nir_intrinsic_instr *instr)
{
   assert(stage == MESA_SHADER_TESS_CTRL);
   struct brw_tcs_prog_key *tcs_key = (struct brw_tcs_prog_key *) key;
   struct brw_tcs_prog_data *tcs_prog_data = brw_tcs_prog_data(prog_data);

   fs_reg dst;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dst = get_nir_dest(instr->dest);

   switch (instr->intrinsic) {
   case nir_intrinsic_load_primitive_id:
      bld.MOV(dst, fs_reg(brw_vec1_grf(0, 1)));
      break;
   case nir_intrinsic_load_invocation_id:
      bld.MOV(retype(dst, invocation_id.type), invocation_id);
      break;
   case nir_intrinsic_load_patch_vertices_in:
      bld.MOV(retype(dst, BRW_REGISTER_TYPE_D),
              brw_imm_d(tcs_key->input_vertices));
      break;

   case nir_intrinsic_barrier: {
      if (tcs_prog_data->instances == 1)
         break;

      fs_reg m0 = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
      fs_reg m0_2 = component(m0, 2);

      const fs_builder chanbld = bld.exec_all().group(1, 0);

      /* Zero the message header */
      bld.exec_all().MOV(m0, brw_imm_ud(0u));

      /* Copy "Barrier ID" from r0.2, bits 16:13 */
      chanbld.AND(m0_2, retype(brw_vec1_grf(0, 2), BRW_REGISTER_TYPE_UD),
                  brw_imm_ud(INTEL_MASK(16, 13)));

      /* Shift it up to bits 27:24. */
      chanbld.SHL(m0_2, m0_2, brw_imm_ud(11));

      /* Set the Barrier Count and the enable bit */
      chanbld.OR(m0_2, m0_2,
                 brw_imm_ud(tcs_prog_data->instances << 9 | (1 << 15)));

      bld.emit(SHADER_OPCODE_BARRIER, bld.null_reg_ud(), m0);
      break;
   }

   case nir_intrinsic_load_input:
      unreachable("nir_lower_io should never give us these.");
      break;

   case nir_intrinsic_load_per_vertex_input: {
      fs_reg indirect_offset = get_indirect_offset(instr);
      unsigned imm_offset = instr->const_index[0];

      const nir_src &vertex_src = instr->src[0];
      nir_const_value *vertex_const = nir_src_as_const_value(vertex_src);

      fs_inst *inst;

      fs_reg icp_handle;

      if (vertex_const) {
         /* Emit a MOV to resolve <0,1,0> regioning. */
         icp_handle = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
         bld.MOV(icp_handle,
                 retype(brw_vec1_grf(1 + (vertex_const->i32[0] >> 3),
                                     vertex_const->i32[0] & 7),
                        BRW_REGISTER_TYPE_UD));
      } else if (tcs_prog_data->instances == 1 &&
                 vertex_src.is_ssa &&
                 vertex_src.ssa->parent_instr->type == nir_instr_type_intrinsic &&
                 nir_instr_as_intrinsic(vertex_src.ssa->parent_instr)->intrinsic == nir_intrinsic_load_invocation_id) {
         /* For the common case of only 1 instance, an array index of
          * gl_InvocationID means reading g1.  Skip all the indirect work.
          */
         icp_handle = retype(brw_vec8_grf(1, 0), BRW_REGISTER_TYPE_UD);
      } else {
         /* The vertex index is non-constant.  We need to use indirect
          * addressing to fetch the proper URB handle.
          */
         icp_handle = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);

         /* Each ICP handle is a single DWord (4 bytes) */
         fs_reg vertex_offset_bytes = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
         bld.SHL(vertex_offset_bytes,
                 retype(get_nir_src(vertex_src), BRW_REGISTER_TYPE_UD),
                 brw_imm_ud(2u));

         /* Start at g1.  We might read up to 4 registers. */
         bld.emit(SHADER_OPCODE_MOV_INDIRECT, icp_handle,
                  retype(brw_vec8_grf(1, 0), icp_handle.type), vertex_offset_bytes,
                  brw_imm_ud(4 * REG_SIZE));
      }

      /* We can only read two double components with each URB read, so
       * we send two read messages in that case, each one loading up to
       * two double components.
       */
      unsigned num_iterations = 1;
      unsigned num_components = instr->num_components;
      unsigned first_component = nir_intrinsic_component(instr);
      fs_reg orig_dst = dst;
      if (type_sz(dst.type) == 8) {
         first_component = first_component / 2;
         if (instr->num_components > 2) {
            num_iterations = 2;
            num_components = 2;
         }

         fs_reg tmp = fs_reg(VGRF, alloc.allocate(4), dst.type);
         dst = tmp;
      }

      for (unsigned iter = 0; iter < num_iterations; iter++) {
         if (indirect_offset.file == BAD_FILE) {
            /* Constant indexing - use global offset. */
            if (first_component != 0) {
               unsigned read_components = num_components + first_component;
               fs_reg tmp = bld.vgrf(dst.type, read_components);
               inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8, tmp, icp_handle);
               for (unsigned i = 0; i < num_components; i++) {
                  bld.MOV(offset(dst, bld, i),
                          offset(tmp, bld, i + first_component));
               }
            } else {
               inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8, dst, icp_handle);
            }
            inst->offset = imm_offset;
            inst->mlen = 1;
         } else {
            /* Indirect indexing - use per-slot offsets as well. */
            const fs_reg srcs[] = { icp_handle, indirect_offset };
            fs_reg payload = bld.vgrf(BRW_REGISTER_TYPE_UD, 2);
            bld.LOAD_PAYLOAD(payload, srcs, ARRAY_SIZE(srcs), 0);
            if (first_component != 0) {
               unsigned read_components = num_components + first_component;
               fs_reg tmp = bld.vgrf(dst.type, read_components);
               inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT, tmp,
                               payload);
               for (unsigned i = 0; i < num_components; i++) {
                  bld.MOV(offset(dst, bld, i),
                          offset(tmp, bld, i + first_component));
               }
            } else {
               inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT, dst,
                               payload);
            }
            inst->offset = imm_offset;
            inst->mlen = 2;
         }
         inst->size_written = (num_components + first_component) *
                              inst->dst.component_size(inst->exec_size);

         /* If we are reading 64-bit data using 32-bit read messages we need
          * build proper 64-bit data elements by shuffling the low and high
          * 32-bit components around like we do for other things like UBOs
          * or SSBOs.
          */
         if (type_sz(dst.type) == 8) {
            shuffle_from_32bit_read(bld,
                                    offset(orig_dst, bld, iter * 2),
                                    retype(dst, BRW_REGISTER_TYPE_D),
                                    0, num_components);
         }

         /* Copy the temporary to the destination to deal with writemasking.
          *
          * Also attempt to deal with gl_PointSize being in the .w component.
          */
         if (inst->offset == 0 && indirect_offset.file == BAD_FILE) {
            assert(type_sz(dst.type) < 8);
            inst->dst = bld.vgrf(dst.type, 4);
            inst->size_written = 4 * REG_SIZE;
            bld.MOV(dst, offset(inst->dst, bld, 3));
         }

         /* If we are loading double data and we need a second read message
          * adjust the write offset
          */
         if (num_iterations > 1) {
            num_components = instr->num_components - 2;
            imm_offset++;
         }
      }
      break;
   }

   case nir_intrinsic_load_output:
   case nir_intrinsic_load_per_vertex_output: {
      fs_reg indirect_offset = get_indirect_offset(instr);
      unsigned imm_offset = instr->const_index[0];
      unsigned first_component = nir_intrinsic_component(instr);

      fs_inst *inst;
      if (indirect_offset.file == BAD_FILE) {
         /* Replicate the patch handle to all enabled channels */
         fs_reg patch_handle = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
         bld.MOV(patch_handle,
                 retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_UD));

         {
            if (first_component != 0) {
               unsigned read_components =
                  instr->num_components + first_component;
               fs_reg tmp = bld.vgrf(dst.type, read_components);
               inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8, tmp,
                               patch_handle);
               inst->size_written = read_components * REG_SIZE;
               for (unsigned i = 0; i < instr->num_components; i++) {
                  bld.MOV(offset(dst, bld, i),
                          offset(tmp, bld, i + first_component));
               }
            } else {
               inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8, dst,
                               patch_handle);
               inst->size_written = instr->num_components * REG_SIZE;
            }
            inst->offset = imm_offset;
            inst->mlen = 1;
         }
      } else {
         /* Indirect indexing - use per-slot offsets as well. */
         const fs_reg srcs[] = {
            retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_UD),
            indirect_offset
         };
         fs_reg payload = bld.vgrf(BRW_REGISTER_TYPE_UD, 2);
         bld.LOAD_PAYLOAD(payload, srcs, ARRAY_SIZE(srcs), 0);
         if (first_component != 0) {
            unsigned read_components =
               instr->num_components + first_component;
            fs_reg tmp = bld.vgrf(dst.type, read_components);
            inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT, tmp,
                            payload);
            inst->size_written = read_components * REG_SIZE;
            for (unsigned i = 0; i < instr->num_components; i++) {
               bld.MOV(offset(dst, bld, i),
                       offset(tmp, bld, i + first_component));
            }
         } else {
            inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT, dst,
                            payload);
            inst->size_written = instr->num_components * REG_SIZE;
         }
         inst->offset = imm_offset;
         inst->mlen = 2;
      }
      break;
   }

   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output: {
      fs_reg value = get_nir_src(instr->src[0]);
      bool is_64bit = (instr->src[0].is_ssa ?
         instr->src[0].ssa->bit_size : instr->src[0].reg.reg->bit_size) == 64;
      fs_reg indirect_offset = get_indirect_offset(instr);
      unsigned imm_offset = instr->const_index[0];
      unsigned mask = instr->const_index[1];
      unsigned header_regs = 0;
      fs_reg srcs[7];
      srcs[header_regs++] = retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_UD);

      if (indirect_offset.file != BAD_FILE) {
         srcs[header_regs++] = indirect_offset;
      }

      if (mask == 0)
         break;

      unsigned num_components = util_last_bit(mask);
      enum opcode opcode;

      /* We can only pack two 64-bit components in a single message, so send
       * 2 messages if we have more components
       */
      unsigned num_iterations = 1;
      unsigned iter_components = num_components;
      unsigned first_component = nir_intrinsic_component(instr);
      if (is_64bit) {
         first_component = first_component / 2;
         if (instr->num_components > 2) {
            num_iterations = 2;
            iter_components = 2;
         }
      }

      mask = mask << first_component;

      for (unsigned iter = 0; iter < num_iterations; iter++) {
         if (!is_64bit && mask != WRITEMASK_XYZW) {
            srcs[header_regs++] = brw_imm_ud(mask << 16);
            opcode = indirect_offset.file != BAD_FILE ?
               SHADER_OPCODE_URB_WRITE_SIMD8_MASKED_PER_SLOT :
               SHADER_OPCODE_URB_WRITE_SIMD8_MASKED;
         } else if (is_64bit && ((mask & WRITEMASK_XY) != WRITEMASK_XY)) {
            /* Expand the 64-bit mask to 32-bit channels. We only handle
             * two channels in each iteration, so we only care about X/Y.
             */
            unsigned mask32 = 0;
            if (mask & WRITEMASK_X)
               mask32 |= WRITEMASK_XY;
            if (mask & WRITEMASK_Y)
               mask32 |= WRITEMASK_ZW;

            /* If the mask does not include any of the channels X or Y there
             * is nothing to do in this iteration. Move on to the next couple
             * of 64-bit channels.
             */
            if (!mask32) {
               mask >>= 2;
               imm_offset++;
               continue;
            }

            srcs[header_regs++] = brw_imm_ud(mask32 << 16);
            opcode = indirect_offset.file != BAD_FILE ?
               SHADER_OPCODE_URB_WRITE_SIMD8_MASKED_PER_SLOT :
               SHADER_OPCODE_URB_WRITE_SIMD8_MASKED;
         } else {
            opcode = indirect_offset.file != BAD_FILE ?
               SHADER_OPCODE_URB_WRITE_SIMD8_PER_SLOT :
               SHADER_OPCODE_URB_WRITE_SIMD8;
         }

         for (unsigned i = 0; i < iter_components; i++) {
            if (!(mask & (1 << (i + first_component))))
               continue;

            if (!is_64bit) {
               srcs[header_regs + i + first_component] = offset(value, bld, i);
            } else {
               /* We need to shuffle the 64-bit data to match the layout
                * expected by our 32-bit URB write messages. We use a temporary
                * for that.
                */
               unsigned channel = iter * 2 + i;
               fs_reg dest = shuffle_for_32bit_write(bld, value, channel, 1);

               srcs[header_regs + (i + first_component) * 2] = dest;
               srcs[header_regs + (i + first_component) * 2 + 1] =
                  offset(dest, bld, 1);
            }
         }

         unsigned mlen =
            header_regs + (is_64bit ? 2 * iter_components : iter_components) +
            (is_64bit ? 2 * first_component : first_component);
         fs_reg payload =
            bld.vgrf(BRW_REGISTER_TYPE_UD, mlen);
         bld.LOAD_PAYLOAD(payload, srcs, mlen, header_regs);

         fs_inst *inst = bld.emit(opcode, bld.null_reg_ud(), payload);
         inst->offset = imm_offset;
         inst->mlen = mlen;

         /* If this is a 64-bit attribute, select the next two 64-bit channels
          * to be handled in the next iteration.
          */
         if (is_64bit) {
            mask >>= 2;
            imm_offset++;
         }
      }
      break;
   }

   default:
      nir_emit_intrinsic(bld, instr);
      break;
   }
}

void
fs_visitor::nir_emit_tes_intrinsic(const fs_builder &bld,
                                   nir_intrinsic_instr *instr)
{
   assert(stage == MESA_SHADER_TESS_EVAL);
   struct brw_tes_prog_data *tes_prog_data = brw_tes_prog_data(prog_data);

   fs_reg dest;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dest = get_nir_dest(instr->dest);

   switch (instr->intrinsic) {
   case nir_intrinsic_load_primitive_id:
      bld.MOV(dest, fs_reg(brw_vec1_grf(0, 1)));
      break;
   case nir_intrinsic_load_tess_coord:
      /* gl_TessCoord is part of the payload in g1-3 */
      for (unsigned i = 0; i < 3; i++) {
         bld.MOV(offset(dest, bld, i), fs_reg(brw_vec8_grf(1 + i, 0)));
      }
      break;

   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input: {
      fs_reg indirect_offset = get_indirect_offset(instr);
      unsigned imm_offset = instr->const_index[0];
      unsigned first_component = nir_intrinsic_component(instr);

      if (type_sz(dest.type) == 8) {
         first_component = first_component / 2;
      }

      fs_inst *inst;
      if (indirect_offset.file == BAD_FILE) {
         /* Arbitrarily only push up to 32 vec4 slots worth of data,
          * which is 16 registers (since each holds 2 vec4 slots).
          */
         unsigned slot_count = 1;
         if (type_sz(dest.type) == 8 && instr->num_components > 2)
            slot_count++;

         const unsigned max_push_slots = 32;
         if (imm_offset + slot_count <= max_push_slots) {
            fs_reg src = fs_reg(ATTR, imm_offset / 2, dest.type);
            for (int i = 0; i < instr->num_components; i++) {
               unsigned comp = 16 / type_sz(dest.type) * (imm_offset % 2) +
                  i + first_component;
               bld.MOV(offset(dest, bld, i), component(src, comp));
            }

            tes_prog_data->base.urb_read_length =
               MAX2(tes_prog_data->base.urb_read_length,
                    DIV_ROUND_UP(imm_offset + slot_count, 2));
         } else {
            /* Replicate the patch handle to all enabled channels */
            const fs_reg srcs[] = {
               retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_UD)
            };
            fs_reg patch_handle = bld.vgrf(BRW_REGISTER_TYPE_UD, 1);
            bld.LOAD_PAYLOAD(patch_handle, srcs, ARRAY_SIZE(srcs), 0);

            if (first_component != 0) {
               unsigned read_components =
                  instr->num_components + first_component;
               fs_reg tmp = bld.vgrf(dest.type, read_components);
               inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8, tmp,
                               patch_handle);
               inst->size_written = read_components * REG_SIZE;
               for (unsigned i = 0; i < instr->num_components; i++) {
                  bld.MOV(offset(dest, bld, i),
                          offset(tmp, bld, i + first_component));
               }
            } else {
               inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8, dest,
                               patch_handle);
               inst->size_written = instr->num_components * REG_SIZE;
            }
            inst->mlen = 1;
            inst->offset = imm_offset;
         }
      } else {
         /* Indirect indexing - use per-slot offsets as well. */

         /* We can only read two double components with each URB read, so
          * we send two read messages in that case, each one loading up to
          * two double components.
          */
         unsigned num_iterations = 1;
         unsigned num_components = instr->num_components;
         fs_reg orig_dest = dest;
         if (type_sz(dest.type) == 8) {
            if (instr->num_components > 2) {
               num_iterations = 2;
               num_components = 2;
            }
            fs_reg tmp = fs_reg(VGRF, alloc.allocate(4), dest.type);
            dest = tmp;
         }

         for (unsigned iter = 0; iter < num_iterations; iter++) {
            const fs_reg srcs[] = {
               retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_UD),
               indirect_offset
            };
            fs_reg payload = bld.vgrf(BRW_REGISTER_TYPE_UD, 2);
            bld.LOAD_PAYLOAD(payload, srcs, ARRAY_SIZE(srcs), 0);

            if (first_component != 0) {
               unsigned read_components =
                   num_components + first_component;
               fs_reg tmp = bld.vgrf(dest.type, read_components);
               inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT, tmp,
                               payload);
               for (unsigned i = 0; i < num_components; i++) {
                  bld.MOV(offset(dest, bld, i),
                          offset(tmp, bld, i + first_component));
               }
            } else {
               inst = bld.emit(SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT, dest,
                               payload);
            }
            inst->mlen = 2;
            inst->offset = imm_offset;
            inst->size_written = (num_components + first_component) *
                                 inst->dst.component_size(inst->exec_size);

            /* If we are reading 64-bit data using 32-bit read messages we need
             * build proper 64-bit data elements by shuffling the low and high
             * 32-bit components around like we do for other things like UBOs
             * or SSBOs.
             */
            if (type_sz(dest.type) == 8) {
               shuffle_from_32bit_read(bld,
                                       offset(orig_dest, bld, iter * 2),
                                       retype(dest, BRW_REGISTER_TYPE_D),
                                       0, num_components);
            }

            /* If we are loading double data and we need a second read message
             * adjust the offset
             */
            if (num_iterations > 1) {
               num_components = instr->num_components - 2;
               imm_offset++;
            }
         }
      }
      break;
   }
   default:
      nir_emit_intrinsic(bld, instr);
      break;
   }
}

void
fs_visitor::nir_emit_gs_intrinsic(const fs_builder &bld,
                                  nir_intrinsic_instr *instr)
{
   assert(stage == MESA_SHADER_GEOMETRY);
   fs_reg indirect_offset;

   fs_reg dest;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dest = get_nir_dest(instr->dest);

   switch (instr->intrinsic) {
   case nir_intrinsic_load_primitive_id:
      assert(stage == MESA_SHADER_GEOMETRY);
      assert(brw_gs_prog_data(prog_data)->include_primitive_id);
      bld.MOV(retype(dest, BRW_REGISTER_TYPE_UD),
              retype(fs_reg(brw_vec8_grf(2, 0)), BRW_REGISTER_TYPE_UD));
      break;

   case nir_intrinsic_load_input:
      unreachable("load_input intrinsics are invalid for the GS stage");

   case nir_intrinsic_load_per_vertex_input:
      emit_gs_input_load(dest, instr->src[0], instr->const_index[0],
                         instr->src[1], instr->num_components,
                         nir_intrinsic_component(instr));
      break;

   case nir_intrinsic_emit_vertex_with_counter:
      emit_gs_vertex(instr->src[0], instr->const_index[0]);
      break;

   case nir_intrinsic_end_primitive_with_counter:
      emit_gs_end_primitive(instr->src[0]);
      break;

   case nir_intrinsic_set_vertex_count:
      bld.MOV(this->final_gs_vertex_count, get_nir_src(instr->src[0]));
      break;

   case nir_intrinsic_load_invocation_id: {
      fs_reg val = nir_system_values[SYSTEM_VALUE_INVOCATION_ID];
      assert(val.file != BAD_FILE);
      dest.type = val.type;
      bld.MOV(dest, val);
      break;
   }

   default:
      nir_emit_intrinsic(bld, instr);
      break;
   }
}

/**
 * Fetch the current render target layer index.
 */
static fs_reg
fetch_render_target_array_index(const fs_builder &bld)
{
   if (bld.shader->devinfo->gen >= 6) {
      /* The render target array index is provided in the thread payload as
       * bits 26:16 of r0.0.
       */
      const fs_reg idx = bld.vgrf(BRW_REGISTER_TYPE_UD);
      bld.AND(idx, brw_uw1_reg(BRW_GENERAL_REGISTER_FILE, 0, 1),
              brw_imm_uw(0x7ff));
      return idx;
   } else {
      /* Pre-SNB we only ever render into the first layer of the framebuffer
       * since layered rendering is not implemented.
       */
      return brw_imm_ud(0);
   }
}

/**
 * Fake non-coherent framebuffer read implemented using TXF to fetch from the
 * framebuffer at the current fragment coordinates and sample index.
 */
fs_inst *
fs_visitor::emit_non_coherent_fb_read(const fs_builder &bld, const fs_reg &dst,
                                      unsigned target)
{
   const struct gen_device_info *devinfo = bld.shader->devinfo;

   assert(bld.shader->stage == MESA_SHADER_FRAGMENT);
   const brw_wm_prog_key *wm_key =
      reinterpret_cast<const brw_wm_prog_key *>(key);
   assert(!wm_key->coherent_fb_fetch);
   const struct brw_wm_prog_data *wm_prog_data =
      brw_wm_prog_data(stage_prog_data);

   /* Calculate the surface index relative to the start of the texture binding
    * table block, since that's what the texturing messages expect.
    */
   const unsigned surface = target +
      wm_prog_data->binding_table.render_target_read_start -
      wm_prog_data->base.binding_table.texture_start;

   brw_mark_surface_used(
      bld.shader->stage_prog_data,
      wm_prog_data->binding_table.render_target_read_start + target);

   /* Calculate the fragment coordinates. */
   const fs_reg coords = bld.vgrf(BRW_REGISTER_TYPE_UD, 3);
   bld.MOV(offset(coords, bld, 0), pixel_x);
   bld.MOV(offset(coords, bld, 1), pixel_y);
   bld.MOV(offset(coords, bld, 2), fetch_render_target_array_index(bld));

   /* Calculate the sample index and MCS payload when multisampling.  Luckily
    * the MCS fetch message behaves deterministically for UMS surfaces, so it
    * shouldn't be necessary to recompile based on whether the framebuffer is
    * CMS or UMS.
    */
   if (wm_key->multisample_fbo &&
       nir_system_values[SYSTEM_VALUE_SAMPLE_ID].file == BAD_FILE)
      nir_system_values[SYSTEM_VALUE_SAMPLE_ID] = *emit_sampleid_setup();

   const fs_reg sample = nir_system_values[SYSTEM_VALUE_SAMPLE_ID];
   const fs_reg mcs = wm_key->multisample_fbo ?
      emit_mcs_fetch(coords, 3, brw_imm_ud(surface)) : fs_reg();

   /* Use either a normal or a CMS texel fetch message depending on whether
    * the framebuffer is single or multisample.  On SKL+ use the wide CMS
    * message just in case the framebuffer uses 16x multisampling, it should
    * be equivalent to the normal CMS fetch for lower multisampling modes.
    */
   const opcode op = !wm_key->multisample_fbo ? SHADER_OPCODE_TXF_LOGICAL :
                     devinfo->gen >= 9 ? SHADER_OPCODE_TXF_CMS_W_LOGICAL :
                     SHADER_OPCODE_TXF_CMS_LOGICAL;

   /* Emit the instruction. */
   const fs_reg srcs[] = { coords, fs_reg(), brw_imm_ud(0), fs_reg(),
                           sample, mcs,
                           brw_imm_ud(surface), brw_imm_ud(0),
                           fs_reg(), brw_imm_ud(3), brw_imm_ud(0) };
   STATIC_ASSERT(ARRAY_SIZE(srcs) == TEX_LOGICAL_NUM_SRCS);

   fs_inst *inst = bld.emit(op, dst, srcs, ARRAY_SIZE(srcs));
   inst->size_written = 4 * inst->dst.component_size(inst->exec_size);

   return inst;
}

/**
 * Actual coherent framebuffer read implemented using the native render target
 * read message.  Requires SKL+.
 */
static fs_inst *
emit_coherent_fb_read(const fs_builder &bld, const fs_reg &dst, unsigned target)
{
   assert(bld.shader->devinfo->gen >= 9);
   fs_inst *inst = bld.emit(FS_OPCODE_FB_READ_LOGICAL, dst);
   inst->target = target;
   inst->size_written = 4 * inst->dst.component_size(inst->exec_size);

   return inst;
}

static fs_reg
alloc_temporary(const fs_builder &bld, unsigned size, fs_reg *regs, unsigned n)
{
   if (n && regs[0].file != BAD_FILE) {
      return regs[0];

   } else {
      const fs_reg tmp = bld.vgrf(BRW_REGISTER_TYPE_F, size);

      for (unsigned i = 0; i < n; i++)
         regs[i] = tmp;

      return tmp;
   }
}

static fs_reg
alloc_frag_output(fs_visitor *v, unsigned location)
{
   assert(v->stage == MESA_SHADER_FRAGMENT);
   const brw_wm_prog_key *const key =
      reinterpret_cast<const brw_wm_prog_key *>(v->key);
   const unsigned l = GET_FIELD(location, BRW_NIR_FRAG_OUTPUT_LOCATION);
   const unsigned i = GET_FIELD(location, BRW_NIR_FRAG_OUTPUT_INDEX);

   if (i > 0 || (key->force_dual_color_blend && l == FRAG_RESULT_DATA1))
      return alloc_temporary(v->bld, 4, &v->dual_src_output, 1);

   else if (l == FRAG_RESULT_COLOR)
      return alloc_temporary(v->bld, 4, v->outputs,
                             MAX2(key->nr_color_regions, 1));

   else if (l == FRAG_RESULT_DEPTH)
      return alloc_temporary(v->bld, 1, &v->frag_depth, 1);

   else if (l == FRAG_RESULT_STENCIL)
      return alloc_temporary(v->bld, 1, &v->frag_stencil, 1);

   else if (l == FRAG_RESULT_SAMPLE_MASK)
      return alloc_temporary(v->bld, 1, &v->sample_mask, 1);

   else if (l >= FRAG_RESULT_DATA0 &&
            l < FRAG_RESULT_DATA0 + BRW_MAX_DRAW_BUFFERS)
      return alloc_temporary(v->bld, 4,
                             &v->outputs[l - FRAG_RESULT_DATA0], 1);

   else
      unreachable("Invalid location");
}

void
fs_visitor::nir_emit_fs_intrinsic(const fs_builder &bld,
                                  nir_intrinsic_instr *instr)
{
   assert(stage == MESA_SHADER_FRAGMENT);

   fs_reg dest;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dest = get_nir_dest(instr->dest);

   switch (instr->intrinsic) {
   case nir_intrinsic_load_front_face:
      bld.MOV(retype(dest, BRW_REGISTER_TYPE_D),
              *emit_frontfacing_interpolation());
      break;

   case nir_intrinsic_load_sample_pos: {
      fs_reg sample_pos = nir_system_values[SYSTEM_VALUE_SAMPLE_POS];
      assert(sample_pos.file != BAD_FILE);
      dest.type = sample_pos.type;
      bld.MOV(dest, sample_pos);
      bld.MOV(offset(dest, bld, 1), offset(sample_pos, bld, 1));
      break;
   }

   case nir_intrinsic_load_layer_id:
      dest.type = BRW_REGISTER_TYPE_UD;
      bld.MOV(dest, fetch_render_target_array_index(bld));
      break;

   case nir_intrinsic_load_helper_invocation:
   case nir_intrinsic_load_sample_mask_in:
   case nir_intrinsic_load_sample_id: {
      gl_system_value sv = nir_system_value_from_intrinsic(instr->intrinsic);
      fs_reg val = nir_system_values[sv];
      assert(val.file != BAD_FILE);
      dest.type = val.type;
      bld.MOV(dest, val);
      break;
   }

   case nir_intrinsic_store_output: {
      const fs_reg src = get_nir_src(instr->src[0]);
      const nir_const_value *const_offset = nir_src_as_const_value(instr->src[1]);
      assert(const_offset && "Indirect output stores not allowed");
      const unsigned location = nir_intrinsic_base(instr) +
         SET_FIELD(const_offset->u32[0], BRW_NIR_FRAG_OUTPUT_LOCATION);
      const fs_reg new_dest = retype(alloc_frag_output(this, location),
                                     src.type);

      for (unsigned j = 0; j < instr->num_components; j++)
         bld.MOV(offset(new_dest, bld, nir_intrinsic_component(instr) + j),
                 offset(src, bld, j));

      break;
   }

   case nir_intrinsic_load_output: {
      const unsigned l = GET_FIELD(nir_intrinsic_base(instr),
                                   BRW_NIR_FRAG_OUTPUT_LOCATION);
      assert(l >= FRAG_RESULT_DATA0);
      nir_const_value *const_offset = nir_src_as_const_value(instr->src[0]);
      assert(const_offset && "Indirect output loads not allowed");
      const unsigned target = l - FRAG_RESULT_DATA0 + const_offset->u32[0];
      const fs_reg tmp = bld.vgrf(dest.type, 4);

      if (reinterpret_cast<const brw_wm_prog_key *>(key)->coherent_fb_fetch)
         emit_coherent_fb_read(bld, tmp, target);
      else
         emit_non_coherent_fb_read(bld, tmp, target);

      for (unsigned j = 0; j < instr->num_components; j++) {
         bld.MOV(offset(dest, bld, j),
                 offset(tmp, bld, nir_intrinsic_component(instr) + j));
      }

      break;
   }

   case nir_intrinsic_discard:
   case nir_intrinsic_discard_if: {
      /* We track our discarded pixels in f0.1.  By predicating on it, we can
       * update just the flag bits that aren't yet discarded.  If there's no
       * condition, we emit a CMP of g0 != g0, so all currently executing
       * channels will get turned off.
       */
      fs_inst *cmp;
      if (instr->intrinsic == nir_intrinsic_discard_if) {
         cmp = bld.CMP(bld.null_reg_f(), get_nir_src(instr->src[0]),
                       brw_imm_d(0), BRW_CONDITIONAL_Z);
      } else {
         fs_reg some_reg = fs_reg(retype(brw_vec8_grf(0, 0),
                                       BRW_REGISTER_TYPE_UW));
         cmp = bld.CMP(bld.null_reg_f(), some_reg, some_reg, BRW_CONDITIONAL_NZ);
      }
      cmp->predicate = BRW_PREDICATE_NORMAL;
      cmp->flag_subreg = 1;

      if (devinfo->gen >= 6) {
         emit_discard_jump();
      }

      limit_dispatch_width(16, "Fragment discard not implemented in SIMD32 mode.");
      break;
   }

   case nir_intrinsic_load_input: {
      /* load_input is only used for flat inputs */
      unsigned base = nir_intrinsic_base(instr);
      unsigned comp = nir_intrinsic_component(instr);
      unsigned num_components = instr->num_components;
      fs_reg orig_dest = dest;
      enum brw_reg_type type = dest.type;

      /* Special case fields in the VUE header */
      if (base == VARYING_SLOT_LAYER)
         comp = 1;
      else if (base == VARYING_SLOT_VIEWPORT)
         comp = 2;

      if (nir_dest_bit_size(instr->dest) == 64) {
         /* const_index is in 32-bit type size units that could not be aligned
          * with DF. We need to read the double vector as if it was a float
          * vector of twice the number of components to fetch the right data.
          */
         type = BRW_REGISTER_TYPE_F;
         num_components *= 2;
         dest = bld.vgrf(type, num_components);
      }

      for (unsigned int i = 0; i < num_components; i++) {
         bld.MOV(offset(retype(dest, type), bld, i),
                 retype(component(interp_reg(base, comp + i), 3), type));
      }

      if (nir_dest_bit_size(instr->dest) == 64) {
         shuffle_from_32bit_read(bld, orig_dest, dest, 0,
                                 instr->num_components);
      }
      break;
   }

   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_sample:
      /* Do nothing - load_interpolated_input handling will handle it later. */
      break;

   case nir_intrinsic_load_barycentric_at_sample: {
      const glsl_interp_mode interpolation =
         (enum glsl_interp_mode) nir_intrinsic_interp_mode(instr);

      nir_const_value *const_sample = nir_src_as_const_value(instr->src[0]);

      if (const_sample) {
         unsigned msg_data = const_sample->i32[0] << 4;

         emit_pixel_interpolater_send(bld,
                                      FS_OPCODE_INTERPOLATE_AT_SAMPLE,
                                      dest,
                                      fs_reg(), /* src */
                                      brw_imm_ud(msg_data),
                                      interpolation);
      } else {
         const fs_reg sample_src = retype(get_nir_src(instr->src[0]),
                                          BRW_REGISTER_TYPE_UD);

         if (nir_src_is_dynamically_uniform(instr->src[0])) {
            const fs_reg sample_id = bld.emit_uniformize(sample_src);
            const fs_reg msg_data = vgrf(glsl_type::uint_type);
            bld.exec_all().group(1, 0)
               .SHL(msg_data, sample_id, brw_imm_ud(4u));
            emit_pixel_interpolater_send(bld,
                                         FS_OPCODE_INTERPOLATE_AT_SAMPLE,
                                         dest,
                                         fs_reg(), /* src */
                                         msg_data,
                                         interpolation);
         } else {
            /* Make a loop that sends a message to the pixel interpolater
             * for the sample number in each live channel. If there are
             * multiple channels with the same sample number then these
             * will be handled simultaneously with a single interation of
             * the loop.
             */
            bld.emit(BRW_OPCODE_DO);

            /* Get the next live sample number into sample_id_reg */
            const fs_reg sample_id = bld.emit_uniformize(sample_src);

            /* Set the flag register so that we can perform the send
             * message on all channels that have the same sample number
             */
            bld.CMP(bld.null_reg_ud(),
                    sample_src, sample_id,
                    BRW_CONDITIONAL_EQ);
            const fs_reg msg_data = vgrf(glsl_type::uint_type);
            bld.exec_all().group(1, 0)
               .SHL(msg_data, sample_id, brw_imm_ud(4u));
            fs_inst *inst =
               emit_pixel_interpolater_send(bld,
                                            FS_OPCODE_INTERPOLATE_AT_SAMPLE,
                                            dest,
                                            fs_reg(), /* src */
                                            component(msg_data, 0),
                                            interpolation);
            set_predicate(BRW_PREDICATE_NORMAL, inst);

            /* Continue the loop if there are any live channels left */
            set_predicate_inv(BRW_PREDICATE_NORMAL,
                              true, /* inverse */
                              bld.emit(BRW_OPCODE_WHILE));
         }
      }
      break;
   }

   case nir_intrinsic_load_barycentric_at_offset: {
      const glsl_interp_mode interpolation =
         (enum glsl_interp_mode) nir_intrinsic_interp_mode(instr);

      nir_const_value *const_offset = nir_src_as_const_value(instr->src[0]);

      if (const_offset) {
         unsigned off_x = MIN2((int)(const_offset->f32[0] * 16), 7) & 0xf;
         unsigned off_y = MIN2((int)(const_offset->f32[1] * 16), 7) & 0xf;

         emit_pixel_interpolater_send(bld,
                                      FS_OPCODE_INTERPOLATE_AT_SHARED_OFFSET,
                                      dest,
                                      fs_reg(), /* src */
                                      brw_imm_ud(off_x | (off_y << 4)),
                                      interpolation);
      } else {
         fs_reg src = vgrf(glsl_type::ivec2_type);
         fs_reg offset_src = retype(get_nir_src(instr->src[0]),
                                    BRW_REGISTER_TYPE_F);
         for (int i = 0; i < 2; i++) {
            fs_reg temp = vgrf(glsl_type::float_type);
            bld.MUL(temp, offset(offset_src, bld, i), brw_imm_f(16.0f));
            fs_reg itemp = vgrf(glsl_type::int_type);
            /* float to int */
            bld.MOV(itemp, temp);

            /* Clamp the upper end of the range to +7/16.
             * ARB_gpu_shader5 requires that we support a maximum offset
             * of +0.5, which isn't representable in a S0.4 value -- if
             * we didn't clamp it, we'd end up with -8/16, which is the
             * opposite of what the shader author wanted.
             *
             * This is legal due to ARB_gpu_shader5's quantization
             * rules:
             *
             * "Not all values of <offset> may be supported; x and y
             * offsets may be rounded to fixed-point values with the
             * number of fraction bits given by the
             * implementation-dependent constant
             * FRAGMENT_INTERPOLATION_OFFSET_BITS"
             */
            set_condmod(BRW_CONDITIONAL_L,
                        bld.SEL(offset(src, bld, i), itemp, brw_imm_d(7)));
         }

         const enum opcode opcode = FS_OPCODE_INTERPOLATE_AT_PER_SLOT_OFFSET;
         emit_pixel_interpolater_send(bld,
                                      opcode,
                                      dest,
                                      src,
                                      brw_imm_ud(0u),
                                      interpolation);
      }
      break;
   }

   case nir_intrinsic_load_interpolated_input: {
      if (nir_intrinsic_base(instr) == VARYING_SLOT_POS) {
         emit_fragcoord_interpolation(dest);
         break;
      }

      assert(instr->src[0].ssa &&
             instr->src[0].ssa->parent_instr->type == nir_instr_type_intrinsic);
      nir_intrinsic_instr *bary_intrinsic =
         nir_instr_as_intrinsic(instr->src[0].ssa->parent_instr);
      nir_intrinsic_op bary_intrin = bary_intrinsic->intrinsic;
      enum glsl_interp_mode interp_mode =
         (enum glsl_interp_mode) nir_intrinsic_interp_mode(bary_intrinsic);
      fs_reg dst_xy;

      if (bary_intrin == nir_intrinsic_load_barycentric_at_offset ||
          bary_intrin == nir_intrinsic_load_barycentric_at_sample) {
         /* Use the result of the PI message */
         dst_xy = retype(get_nir_src(instr->src[0]), BRW_REGISTER_TYPE_F);
      } else {
         /* Use the delta_xy values computed from the payload */
         enum brw_barycentric_mode bary =
            brw_barycentric_mode(interp_mode, bary_intrin);

         dst_xy = this->delta_xy[bary];
      }

      for (unsigned int i = 0; i < instr->num_components; i++) {
         fs_reg interp =
            component(interp_reg(nir_intrinsic_base(instr),
                                 nir_intrinsic_component(instr) + i), 0);
         interp.type = BRW_REGISTER_TYPE_F;
         dest.type = BRW_REGISTER_TYPE_F;

         if (devinfo->gen < 6 && interp_mode == INTERP_MODE_SMOOTH) {
            fs_reg tmp = vgrf(glsl_type::float_type);
            bld.emit(FS_OPCODE_LINTERP, tmp, dst_xy, interp);
            bld.MUL(offset(dest, bld, i), tmp, this->pixel_w);
         } else {
            bld.emit(FS_OPCODE_LINTERP, offset(dest, bld, i), dst_xy, interp);
         }
      }
      break;
   }

   default:
      nir_emit_intrinsic(bld, instr);
      break;
   }
}

static int
get_op_for_atomic_add(nir_intrinsic_instr *instr, unsigned src)
{
   const nir_const_value *const val = nir_src_as_const_value(instr->src[src]);

   if (val != NULL) {
      if (val->i32[0] == 1)
         return BRW_AOP_INC;
      else if (val->i32[0] == -1)
         return BRW_AOP_DEC;
   }

   return BRW_AOP_ADD;
}

void
fs_visitor::nir_emit_cs_intrinsic(const fs_builder &bld,
                                  nir_intrinsic_instr *instr)
{
   assert(stage == MESA_SHADER_COMPUTE);
   struct brw_cs_prog_data *cs_prog_data = brw_cs_prog_data(prog_data);

   fs_reg dest;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dest = get_nir_dest(instr->dest);

   switch (instr->intrinsic) {
   case nir_intrinsic_barrier:
      emit_barrier();
      cs_prog_data->uses_barrier = true;
      break;

   case nir_intrinsic_load_subgroup_id:
      bld.MOV(retype(dest, BRW_REGISTER_TYPE_UD), subgroup_id);
      break;

   case nir_intrinsic_load_local_invocation_id:
   case nir_intrinsic_load_work_group_id: {
      gl_system_value sv = nir_system_value_from_intrinsic(instr->intrinsic);
      fs_reg val = nir_system_values[sv];
      assert(val.file != BAD_FILE);
      dest.type = val.type;
      for (unsigned i = 0; i < 3; i++)
         bld.MOV(offset(dest, bld, i), offset(val, bld, i));
      break;
   }

   case nir_intrinsic_load_num_work_groups: {
      const unsigned surface =
         cs_prog_data->binding_table.work_groups_start;

      cs_prog_data->uses_num_work_groups = true;

      fs_reg surf_index = brw_imm_ud(surface);
      brw_mark_surface_used(prog_data, surface);

      /* Read the 3 GLuint components of gl_NumWorkGroups */
      for (unsigned i = 0; i < 3; i++) {
         fs_reg read_result =
            emit_untyped_read(bld, surf_index,
                              brw_imm_ud(i << 2),
                              1 /* dims */, 1 /* size */,
                              BRW_PREDICATE_NONE);
         read_result.type = dest.type;
         bld.MOV(dest, read_result);
         dest = offset(dest, bld, 1);
      }
      break;
   }

   case nir_intrinsic_shared_atomic_add:
      nir_emit_shared_atomic(bld, get_op_for_atomic_add(instr, 1), instr);
      break;
   case nir_intrinsic_shared_atomic_imin:
      nir_emit_shared_atomic(bld, BRW_AOP_IMIN, instr);
      break;
   case nir_intrinsic_shared_atomic_umin:
      nir_emit_shared_atomic(bld, BRW_AOP_UMIN, instr);
      break;
   case nir_intrinsic_shared_atomic_imax:
      nir_emit_shared_atomic(bld, BRW_AOP_IMAX, instr);
      break;
   case nir_intrinsic_shared_atomic_umax:
      nir_emit_shared_atomic(bld, BRW_AOP_UMAX, instr);
      break;
   case nir_intrinsic_shared_atomic_and:
      nir_emit_shared_atomic(bld, BRW_AOP_AND, instr);
      break;
   case nir_intrinsic_shared_atomic_or:
      nir_emit_shared_atomic(bld, BRW_AOP_OR, instr);
      break;
   case nir_intrinsic_shared_atomic_xor:
      nir_emit_shared_atomic(bld, BRW_AOP_XOR, instr);
      break;
   case nir_intrinsic_shared_atomic_exchange:
      nir_emit_shared_atomic(bld, BRW_AOP_MOV, instr);
      break;
   case nir_intrinsic_shared_atomic_comp_swap:
      nir_emit_shared_atomic(bld, BRW_AOP_CMPWR, instr);
      break;
   case nir_intrinsic_shared_atomic_fmin:
      nir_emit_shared_atomic_float(bld, BRW_AOP_FMIN, instr);
      break;
   case nir_intrinsic_shared_atomic_fmax:
      nir_emit_shared_atomic_float(bld, BRW_AOP_FMAX, instr);
      break;
   case nir_intrinsic_shared_atomic_fcomp_swap:
      nir_emit_shared_atomic_float(bld, BRW_AOP_FCMPWR, instr);
      break;

   case nir_intrinsic_load_shared: {
      assert(devinfo->gen >= 7);

      fs_reg surf_index = brw_imm_ud(GEN7_BTI_SLM);

      /* Get the offset to read from */
      fs_reg offset_reg;
      nir_const_value *const_offset = nir_src_as_const_value(instr->src[0]);
      if (const_offset) {
         offset_reg = brw_imm_ud(instr->const_index[0] + const_offset->u32[0]);
      } else {
         offset_reg = vgrf(glsl_type::uint_type);
         bld.ADD(offset_reg,
                 retype(get_nir_src(instr->src[0]), BRW_REGISTER_TYPE_UD),
                 brw_imm_ud(instr->const_index[0]));
      }

      /* Read the vector */
      do_untyped_vector_read(bld, dest, surf_index, offset_reg,
                             instr->num_components);
      break;
   }

   case nir_intrinsic_store_shared: {
      assert(devinfo->gen >= 7);

      /* Block index */
      fs_reg surf_index = brw_imm_ud(GEN7_BTI_SLM);

      /* Value */
      fs_reg val_reg = get_nir_src(instr->src[0]);

      /* Writemask */
      unsigned writemask = instr->const_index[1];

      /* get_nir_src() retypes to integer. Be wary of 64-bit types though
       * since the untyped writes below operate in units of 32-bits, which
       * means that we need to write twice as many components each time.
       * Also, we have to suffle 64-bit data to be in the appropriate layout
       * expected by our 32-bit write messages.
       */
      unsigned type_size = 4;
      if (nir_src_bit_size(instr->src[0]) == 64) {
         type_size = 8;
         val_reg = shuffle_for_32bit_write(bld, val_reg, 0,
                                           instr->num_components);
      }

      unsigned type_slots = type_size / 4;

      /* Combine groups of consecutive enabled channels in one write
       * message. We use ffs to find the first enabled channel and then ffs on
       * the bit-inverse, down-shifted writemask to determine the length of
       * the block of enabled bits.
       */
      while (writemask) {
         unsigned first_component = ffs(writemask) - 1;
         unsigned length = ffs(~(writemask >> first_component)) - 1;

         /* We can't write more than 2 64-bit components at once. Limit the
          * length of the write to what we can do and let the next iteration
          * handle the rest
          */
         if (type_size > 4)
            length = MIN2(2, length);

         fs_reg offset_reg;
         nir_const_value *const_offset = nir_src_as_const_value(instr->src[1]);
         if (const_offset) {
            offset_reg = brw_imm_ud(instr->const_index[0] + const_offset->u32[0] +
                                    type_size * first_component);
         } else {
            offset_reg = vgrf(glsl_type::uint_type);
            bld.ADD(offset_reg,
                    retype(get_nir_src(instr->src[1]), BRW_REGISTER_TYPE_UD),
                    brw_imm_ud(instr->const_index[0] + type_size * first_component));
         }

         emit_untyped_write(bld, surf_index, offset_reg,
                            offset(val_reg, bld, first_component * type_slots),
                            1 /* dims */, length * type_slots,
                            BRW_PREDICATE_NONE);

         /* Clear the bits in the writemask that we just wrote, then try
          * again to see if more channels are left.
          */
         writemask &= (15 << (first_component + length));
      }

      break;
   }

   default:
      nir_emit_intrinsic(bld, instr);
      break;
   }
}

static fs_reg
brw_nir_reduction_op_identity(const fs_builder &bld,
                              nir_op op, brw_reg_type type)
{
   nir_const_value value = nir_alu_binop_identity(op, type_sz(type) * 8);
   switch (type_sz(type)) {
   case 2:
      assert(type != BRW_REGISTER_TYPE_HF);
      return retype(brw_imm_uw(value.u16[0]), type);
   case 4:
      return retype(brw_imm_ud(value.u32[0]), type);
   case 8:
      if (type == BRW_REGISTER_TYPE_DF)
         return setup_imm_df(bld, value.f64[0]);
      else
         return retype(brw_imm_u64(value.u64[0]), type);
   default:
      unreachable("Invalid type size");
   }
}

static opcode
brw_op_for_nir_reduction_op(nir_op op)
{
   switch (op) {
   case nir_op_iadd: return BRW_OPCODE_ADD;
   case nir_op_fadd: return BRW_OPCODE_ADD;
   case nir_op_imul: return BRW_OPCODE_MUL;
   case nir_op_fmul: return BRW_OPCODE_MUL;
   case nir_op_imin: return BRW_OPCODE_SEL;
   case nir_op_umin: return BRW_OPCODE_SEL;
   case nir_op_fmin: return BRW_OPCODE_SEL;
   case nir_op_imax: return BRW_OPCODE_SEL;
   case nir_op_umax: return BRW_OPCODE_SEL;
   case nir_op_fmax: return BRW_OPCODE_SEL;
   case nir_op_iand: return BRW_OPCODE_AND;
   case nir_op_ior:  return BRW_OPCODE_OR;
   case nir_op_ixor: return BRW_OPCODE_XOR;
   default:
      unreachable("Invalid reduction operation");
   }
}

static brw_conditional_mod
brw_cond_mod_for_nir_reduction_op(nir_op op)
{
   switch (op) {
   case nir_op_iadd: return BRW_CONDITIONAL_NONE;
   case nir_op_fadd: return BRW_CONDITIONAL_NONE;
   case nir_op_imul: return BRW_CONDITIONAL_NONE;
   case nir_op_fmul: return BRW_CONDITIONAL_NONE;
   case nir_op_imin: return BRW_CONDITIONAL_L;
   case nir_op_umin: return BRW_CONDITIONAL_L;
   case nir_op_fmin: return BRW_CONDITIONAL_L;
   case nir_op_imax: return BRW_CONDITIONAL_GE;
   case nir_op_umax: return BRW_CONDITIONAL_GE;
   case nir_op_fmax: return BRW_CONDITIONAL_GE;
   case nir_op_iand: return BRW_CONDITIONAL_NONE;
   case nir_op_ior:  return BRW_CONDITIONAL_NONE;
   case nir_op_ixor: return BRW_CONDITIONAL_NONE;
   default:
      unreachable("Invalid reduction operation");
   }
}

fs_reg
fs_visitor::get_nir_image_intrinsic_image(const brw::fs_builder &bld,
                                          nir_intrinsic_instr *instr)
{
   fs_reg image = retype(get_nir_src_imm(instr->src[0]), BRW_REGISTER_TYPE_UD);

   if (stage_prog_data->binding_table.image_start > 0) {
      if (image.file == BRW_IMMEDIATE_VALUE) {
         image.d += stage_prog_data->binding_table.image_start;
      } else {
         bld.ADD(image, image,
                 brw_imm_d(stage_prog_data->binding_table.image_start));
      }
   }

   return bld.emit_uniformize(image);
}

static unsigned
image_intrinsic_coord_components(nir_intrinsic_instr *instr)
{
   switch (nir_intrinsic_image_dim(instr)) {
   case GLSL_SAMPLER_DIM_1D:
      return 1 + nir_intrinsic_image_array(instr);
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_RECT:
      return 2 + nir_intrinsic_image_array(instr);
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
      return 3;
   case GLSL_SAMPLER_DIM_BUF:
      return 1;
   case GLSL_SAMPLER_DIM_MS:
      return 2 + nir_intrinsic_image_array(instr);
   default:
      unreachable("Invalid image dimension");
   }
}

void
fs_visitor::nir_emit_intrinsic(const fs_builder &bld, nir_intrinsic_instr *instr)
{
   fs_reg dest;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dest = get_nir_dest(instr->dest);

   switch (instr->intrinsic) {
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_store:
   case nir_intrinsic_image_atomic_add:
   case nir_intrinsic_image_atomic_min:
   case nir_intrinsic_image_atomic_max:
   case nir_intrinsic_image_atomic_and:
   case nir_intrinsic_image_atomic_or:
   case nir_intrinsic_image_atomic_xor:
   case nir_intrinsic_image_atomic_exchange:
   case nir_intrinsic_image_atomic_comp_swap: {
      if (stage == MESA_SHADER_FRAGMENT &&
          instr->intrinsic != nir_intrinsic_image_load)
         brw_wm_prog_data(prog_data)->has_side_effects = true;

      /* Get some metadata from the image intrinsic. */
      const nir_intrinsic_info *info = &nir_intrinsic_infos[instr->intrinsic];
      const unsigned dims = image_intrinsic_coord_components(instr);
      const GLenum format = nir_intrinsic_format(instr);
      const unsigned dest_components = nir_intrinsic_dest_components(instr);

      /* Get the arguments of the image intrinsic. */
      const fs_reg image = get_nir_image_intrinsic_image(bld, instr);
      const fs_reg coords = retype(get_nir_src(instr->src[1]),
                                   BRW_REGISTER_TYPE_UD);
      fs_reg tmp;

      /* Emit an image load, store or atomic op. */
      if (instr->intrinsic == nir_intrinsic_image_load) {
         tmp = emit_typed_read(bld, image, coords, dims,
                               instr->num_components);
      } else if (instr->intrinsic == nir_intrinsic_image_store) {
         const fs_reg src0 = get_nir_src(instr->src[3]);
         emit_typed_write(bld, image, coords, src0, dims,
                          instr->num_components);
      } else {
         int op;
         unsigned num_srcs = info->num_srcs;

         switch (instr->intrinsic) {
         case nir_intrinsic_image_atomic_add:
            assert(num_srcs == 4);

            op = get_op_for_atomic_add(instr, 3);

            if (op != BRW_AOP_ADD)
               num_srcs = 3;
            break;
         case nir_intrinsic_image_atomic_min:
            assert(format == GL_R32UI || format == GL_R32I);
            op = (format == GL_R32I) ? BRW_AOP_IMIN : BRW_AOP_UMIN;
            break;
         case nir_intrinsic_image_atomic_max:
            assert(format == GL_R32UI || format == GL_R32I);
            op = (format == GL_R32I) ? BRW_AOP_IMAX : BRW_AOP_UMAX;
            break;
         case nir_intrinsic_image_atomic_and:
            op = BRW_AOP_AND;
            break;
         case nir_intrinsic_image_atomic_or:
            op = BRW_AOP_OR;
            break;
         case nir_intrinsic_image_atomic_xor:
            op = BRW_AOP_XOR;
            break;
         case nir_intrinsic_image_atomic_exchange:
            op = BRW_AOP_MOV;
            break;
         case nir_intrinsic_image_atomic_comp_swap:
            op = BRW_AOP_CMPWR;
            break;
         default:
            unreachable("Not reachable.");
         }

         const fs_reg src0 = (num_srcs >= 4 ?
                              get_nir_src(instr->src[3]) : fs_reg());
         const fs_reg src1 = (num_srcs >= 5 ?
                              get_nir_src(instr->src[4]) : fs_reg());

         tmp = emit_typed_atomic(bld, image, coords, src0, src1, dims, 1, op);
      }

      /* Assign the result. */
      for (unsigned c = 0; c < dest_components; ++c) {
         bld.MOV(offset(retype(dest, tmp.type), bld, c),
                 offset(tmp, bld, c));
      }
      break;
   }

   case nir_intrinsic_image_size: {
      /* Unlike the [un]typed load and store opcodes, the TXS that this turns
       * into will handle the binding table index for us in the geneerator.
       */
      fs_reg image = retype(get_nir_src_imm(instr->src[0]),
                            BRW_REGISTER_TYPE_UD);
      image = bld.emit_uniformize(image);

      /* Since the image size is always uniform, we can just emit a SIMD8
       * query instruction and splat the result out.
       */
      const fs_builder ubld = bld.exec_all().group(8, 0);

      /* The LOD also serves as the message payload */
      fs_reg lod = ubld.vgrf(BRW_REGISTER_TYPE_UD);
      ubld.MOV(lod, brw_imm_ud(0));

      fs_reg tmp = ubld.vgrf(BRW_REGISTER_TYPE_UD, 4);
      fs_inst *inst = ubld.emit(SHADER_OPCODE_IMAGE_SIZE, tmp, lod, image);
      inst->mlen = 1;
      inst->size_written = 4 * REG_SIZE;

      for (unsigned c = 0; c < instr->dest.ssa.num_components; ++c) {
         if (c == 2 && nir_intrinsic_image_dim(instr) == GLSL_SAMPLER_DIM_CUBE) {
            bld.emit(SHADER_OPCODE_INT_QUOTIENT,
                     offset(retype(dest, tmp.type), bld, c),
                     component(offset(tmp, ubld, c), 0), brw_imm_ud(6));
         } else {
            bld.MOV(offset(retype(dest, tmp.type), bld, c),
                    component(offset(tmp, ubld, c), 0));
         }
      }
      break;
   }

   case nir_intrinsic_image_load_raw_intel: {
      const fs_reg image = get_nir_image_intrinsic_image(bld, instr);
      const fs_reg addr = retype(get_nir_src(instr->src[1]),
                                 BRW_REGISTER_TYPE_UD);

      fs_reg tmp = emit_untyped_read(bld, image, addr, 1,
                                     instr->num_components);

      for (unsigned c = 0; c < instr->num_components; ++c) {
         bld.MOV(offset(retype(dest, tmp.type), bld, c),
                 offset(tmp, bld, c));
      }
      break;
   }

   case nir_intrinsic_image_store_raw_intel: {
      const fs_reg image = get_nir_image_intrinsic_image(bld, instr);
      const fs_reg addr = retype(get_nir_src(instr->src[1]),
                                 BRW_REGISTER_TYPE_UD);
      const fs_reg data = retype(get_nir_src(instr->src[2]),
                                 BRW_REGISTER_TYPE_UD);

      brw_wm_prog_data(prog_data)->has_side_effects = true;

      emit_untyped_write(bld, image, addr, data, 1,
                         instr->num_components);
      break;
   }

   case nir_intrinsic_group_memory_barrier:
   case nir_intrinsic_memory_barrier_shared:
   case nir_intrinsic_memory_barrier_atomic_counter:
   case nir_intrinsic_memory_barrier_buffer:
   case nir_intrinsic_memory_barrier_image:
   case nir_intrinsic_memory_barrier: {
      const fs_builder ubld = bld.group(8, 0);
      const fs_reg tmp = ubld.vgrf(BRW_REGISTER_TYPE_UD, 2);
      ubld.emit(SHADER_OPCODE_MEMORY_FENCE, tmp)
         ->size_written = 2 * REG_SIZE;
      break;
   }

   case nir_intrinsic_shader_clock: {
      /* We cannot do anything if there is an event, so ignore it for now */
      const fs_reg shader_clock = get_timestamp(bld);
      const fs_reg srcs[] = { component(shader_clock, 0),
                              component(shader_clock, 1) };
      bld.LOAD_PAYLOAD(dest, srcs, ARRAY_SIZE(srcs), 0);
      break;
   }

   case nir_intrinsic_image_samples:
      /* The driver does not support multi-sampled images. */
      bld.MOV(retype(dest, BRW_REGISTER_TYPE_D), brw_imm_d(1));
      break;

   case nir_intrinsic_load_uniform: {
      /* Offsets are in bytes but they should always aligned to
       * the type size
       */
      assert(instr->const_index[0] % 4 == 0 ||
             instr->const_index[0] % type_sz(dest.type) == 0);

      fs_reg src(UNIFORM, instr->const_index[0] / 4, dest.type);

      nir_const_value *const_offset = nir_src_as_const_value(instr->src[0]);
      if (const_offset) {
         assert(const_offset->u32[0] % type_sz(dest.type) == 0);
         /* For 16-bit types we add the module of the const_index[0]
          * offset to access to not 32-bit aligned element
          */
         src.offset = const_offset->u32[0] + instr->const_index[0] % 4;

         for (unsigned j = 0; j < instr->num_components; j++) {
            bld.MOV(offset(dest, bld, j), offset(src, bld, j));
         }
      } else {
         fs_reg indirect = retype(get_nir_src(instr->src[0]),
                                  BRW_REGISTER_TYPE_UD);

         /* We need to pass a size to the MOV_INDIRECT but we don't want it to
          * go past the end of the uniform.  In order to keep the n'th
          * component from running past, we subtract off the size of all but
          * one component of the vector.
          */
         assert(instr->const_index[1] >=
                instr->num_components * (int) type_sz(dest.type));
         unsigned read_size = instr->const_index[1] -
            (instr->num_components - 1) * type_sz(dest.type);

         bool supports_64bit_indirects =
            !devinfo->is_cherryview && !gen_device_info_is_9lp(devinfo);

         if (type_sz(dest.type) != 8 || supports_64bit_indirects) {
            for (unsigned j = 0; j < instr->num_components; j++) {
               bld.emit(SHADER_OPCODE_MOV_INDIRECT,
                        offset(dest, bld, j), offset(src, bld, j),
                        indirect, brw_imm_ud(read_size));
            }
         } else {
            const unsigned num_mov_indirects =
               type_sz(dest.type) / type_sz(BRW_REGISTER_TYPE_UD);
            /* We read a little bit less per MOV INDIRECT, as they are now
             * 32-bits ones instead of 64-bit. Fix read_size then.
             */
            const unsigned read_size_32bit = read_size -
                (num_mov_indirects - 1) * type_sz(BRW_REGISTER_TYPE_UD);
            for (unsigned j = 0; j < instr->num_components; j++) {
               for (unsigned i = 0; i < num_mov_indirects; i++) {
                  bld.emit(SHADER_OPCODE_MOV_INDIRECT,
                           subscript(offset(dest, bld, j), BRW_REGISTER_TYPE_UD, i),
                           subscript(offset(src, bld, j), BRW_REGISTER_TYPE_UD, i),
                           indirect, brw_imm_ud(read_size_32bit));
               }
            }
         }
      }
      break;
   }

   case nir_intrinsic_load_ubo: {
      nir_const_value *const_index = nir_src_as_const_value(instr->src[0]);
      fs_reg surf_index;

      if (const_index) {
         const unsigned index = stage_prog_data->binding_table.ubo_start +
                                const_index->u32[0];
         surf_index = brw_imm_ud(index);
         brw_mark_surface_used(prog_data, index);
      } else {
         /* The block index is not a constant. Evaluate the index expression
          * per-channel and add the base UBO index; we have to select a value
          * from any live channel.
          */
         surf_index = vgrf(glsl_type::uint_type);
         bld.ADD(surf_index, get_nir_src(instr->src[0]),
                 brw_imm_ud(stage_prog_data->binding_table.ubo_start));
         surf_index = bld.emit_uniformize(surf_index);

         /* Assume this may touch any UBO. It would be nice to provide
          * a tighter bound, but the array information is already lowered away.
          */
         brw_mark_surface_used(prog_data,
                               stage_prog_data->binding_table.ubo_start +
                               nir->info.num_ubos - 1);
      }

      nir_const_value *const_offset = nir_src_as_const_value(instr->src[1]);
      if (const_offset == NULL) {
         fs_reg base_offset = retype(get_nir_src(instr->src[1]),
                                     BRW_REGISTER_TYPE_UD);

         for (int i = 0; i < instr->num_components; i++)
            VARYING_PULL_CONSTANT_LOAD(bld, offset(dest, bld, i), surf_index,
                                       base_offset, i * type_sz(dest.type));
      } else {
         /* Even if we are loading doubles, a pull constant load will load
          * a 32-bit vec4, so should only reserve vgrf space for that. If we
          * need to load a full dvec4 we will have to emit 2 loads. This is
          * similar to demote_pull_constants(), except that in that case we
          * see individual accesses to each component of the vector and then
          * we let CSE deal with duplicate loads. Here we see a vector access
          * and we have to split it if necessary.
          */
         const unsigned type_size = type_sz(dest.type);

         /* See if we've selected this as a push constant candidate */
         if (const_index) {
            const unsigned ubo_block = const_index->u32[0];
            const unsigned offset_256b = const_offset->u32[0] / 32;

            fs_reg push_reg;
            for (int i = 0; i < 4; i++) {
               const struct brw_ubo_range *range = &prog_data->ubo_ranges[i];
               if (range->block == ubo_block &&
                   offset_256b >= range->start &&
                   offset_256b < range->start + range->length) {

                  push_reg = fs_reg(UNIFORM, UBO_START + i, dest.type);
                  push_reg.offset = const_offset->u32[0] - 32 * range->start;
                  break;
               }
            }

            if (push_reg.file != BAD_FILE) {
               for (unsigned i = 0; i < instr->num_components; i++) {
                  bld.MOV(offset(dest, bld, i),
                          byte_offset(push_reg, i * type_size));
               }
               break;
            }
         }

         const unsigned block_sz = 64; /* Fetch one cacheline at a time. */
         const fs_builder ubld = bld.exec_all().group(block_sz / 4, 0);
         const fs_reg packed_consts = ubld.vgrf(BRW_REGISTER_TYPE_UD);

         for (unsigned c = 0; c < instr->num_components;) {
            const unsigned base = const_offset->u32[0] + c * type_size;
            /* Number of usable components in the next block-aligned load. */
            const unsigned count = MIN2(instr->num_components - c,
                                        (block_sz - base % block_sz) / type_size);

            ubld.emit(FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD,
                      packed_consts, surf_index,
                      brw_imm_ud(base & ~(block_sz - 1)));

            const fs_reg consts =
               retype(byte_offset(packed_consts, base & (block_sz - 1)),
                      dest.type);

            for (unsigned d = 0; d < count; d++)
               bld.MOV(offset(dest, bld, c + d), component(consts, d));

            c += count;
         }
      }
      break;
   }

   case nir_intrinsic_load_ssbo: {
      assert(devinfo->gen >= 7);

      nir_const_value *const_uniform_block =
         nir_src_as_const_value(instr->src[0]);

      fs_reg surf_index;
      if (const_uniform_block) {
         unsigned index = stage_prog_data->binding_table.ssbo_start +
                          const_uniform_block->u32[0];
         surf_index = brw_imm_ud(index);
         brw_mark_surface_used(prog_data, index);
      } else {
         surf_index = vgrf(glsl_type::uint_type);
         bld.ADD(surf_index, get_nir_src(instr->src[0]),
                 brw_imm_ud(stage_prog_data->binding_table.ssbo_start));

         /* Assume this may touch any UBO. It would be nice to provide
          * a tighter bound, but the array information is already lowered away.
          */
         brw_mark_surface_used(prog_data,
                               stage_prog_data->binding_table.ssbo_start +
                               nir->info.num_ssbos - 1);
      }

      fs_reg offset_reg;
      nir_const_value *const_offset = nir_src_as_const_value(instr->src[1]);
      if (const_offset) {
         offset_reg = brw_imm_ud(const_offset->u32[0]);
      } else {
         offset_reg = retype(get_nir_src(instr->src[1]), BRW_REGISTER_TYPE_UD);
      }

      /* Read the vector */
      do_untyped_vector_read(bld, dest, surf_index, offset_reg,
                             instr->num_components);

      break;
   }

   case nir_intrinsic_store_ssbo: {
      assert(devinfo->gen >= 7);

      if (stage == MESA_SHADER_FRAGMENT)
         brw_wm_prog_data(prog_data)->has_side_effects = true;

      /* Block index */
      fs_reg surf_index;
      nir_const_value *const_uniform_block =
         nir_src_as_const_value(instr->src[1]);
      if (const_uniform_block) {
         unsigned index = stage_prog_data->binding_table.ssbo_start +
                          const_uniform_block->u32[0];
         surf_index = brw_imm_ud(index);
         brw_mark_surface_used(prog_data, index);
      } else {
         surf_index = vgrf(glsl_type::uint_type);
         bld.ADD(surf_index, get_nir_src(instr->src[1]),
                  brw_imm_ud(stage_prog_data->binding_table.ssbo_start));

         brw_mark_surface_used(prog_data,
                               stage_prog_data->binding_table.ssbo_start +
                               nir->info.num_ssbos - 1);
      }

      /* Value */
      fs_reg val_reg = get_nir_src(instr->src[0]);

      /* Writemask */
      unsigned writemask = instr->const_index[0];

      /* get_nir_src() retypes to integer. Be wary of 64-bit types though
       * since the untyped writes below operate in units of 32-bits, which
       * means that we need to write twice as many components each time.
       * Also, we have to suffle 64-bit data to be in the appropriate layout
       * expected by our 32-bit write messages.
       */
      unsigned bit_size = nir_src_bit_size(instr->src[0]);
      unsigned type_size = bit_size / 8;

      /* Combine groups of consecutive enabled channels in one write
       * message. We use ffs to find the first enabled channel and then ffs on
       * the bit-inverse, down-shifted writemask to determine the num_components
       * of the block of enabled bits.
       */
      while (writemask) {
         unsigned first_component = ffs(writemask) - 1;
         unsigned num_components = ffs(~(writemask >> first_component)) - 1;
         fs_reg write_src = offset(val_reg, bld, first_component);

         nir_const_value *const_offset = nir_src_as_const_value(instr->src[2]);

         if (type_size > 4) {
            /* We can't write more than 2 64-bit components at once. Limit
             * the num_components of the write to what we can do and let the next
             * iteration handle the rest.
             */
            num_components = MIN2(2, num_components);
            write_src = shuffle_for_32bit_write(bld, write_src, 0,
                                                num_components);
         } else if (type_size < 4) {
            /* For 16-bit types we pack two consecutive values into a 32-bit
             * word and use an untyped write message. For single values or not
             * 32-bit-aligned we need to use byte-scattered writes because
             * untyped writes works with 32-bit components with 32-bit
             * alignment. byte_scattered_write messages only support one
             * 16-bit component at a time. As VK_KHR_relaxed_block_layout
             * could be enabled we can not guarantee that not constant offsets
             * to be 32-bit aligned for 16-bit types. For example an array, of
             * 16-bit vec3 with array element stride of 6.
             *
             * In the case of 32-bit aligned constant offsets if there is
             * a 3-components vector we submit one untyped-write message
             * of 32-bit (first two components), and one byte-scattered
             * write message (the last component).
             */

            if ( !const_offset || ((const_offset->u32[0] +
                                   type_size * first_component) % 4)) {
               /* If we use a .yz writemask we also need to emit 2
                * byte-scattered write messages because of y-component not
                * being aligned to 32-bit.
                */
               num_components = 1;
            } else if (num_components * type_size > 4 &&
                       (num_components * type_size % 4)) {
               /* If the pending components size is not a multiple of 4 bytes
                * we left the not aligned components for following emits of
                * length == 1 with byte_scattered_write.
                */
               num_components -= (num_components * type_size % 4) / type_size;
            } else if (num_components * type_size < 4) {
               num_components = 1;
            }
            /* For num_components == 1 we are also shuffling the component
             * because byte scattered writes of 16-bit need values to be dword
             * aligned. Shuffling only one component would be the same as
             * striding it.
             */
            write_src = shuffle_for_32bit_write(bld, write_src, 0,
                                                num_components);
         }

         fs_reg offset_reg;

         if (const_offset) {
            offset_reg = brw_imm_ud(const_offset->u32[0] +
                                    type_size * first_component);
         } else {
            offset_reg = vgrf(glsl_type::uint_type);
            bld.ADD(offset_reg,
                    retype(get_nir_src(instr->src[2]), BRW_REGISTER_TYPE_UD),
                    brw_imm_ud(type_size * first_component));
         }

         if (type_size < 4 && num_components == 1) {
            /* Untyped Surface messages have a fixed 32-bit size, so we need
             * to rely on byte scattered in order to write 16-bit elements.
             * The byte_scattered_write message needs that every written 16-bit
             * type to be aligned 32-bits (stride=2).
             */
            emit_byte_scattered_write(bld, surf_index, offset_reg,
                                      write_src,
                                      1 /* dims */,
                                      bit_size,
                                      BRW_PREDICATE_NONE);
         } else {
            assert(num_components * type_size <= 16);
            assert((num_components * type_size) % 4 == 0);
            assert(offset_reg.file != BRW_IMMEDIATE_VALUE ||
                   offset_reg.ud % 4 == 0);
            unsigned num_slots = (num_components * type_size) / 4;

            emit_untyped_write(bld, surf_index, offset_reg,
                               write_src,
                               1 /* dims */, num_slots,
                               BRW_PREDICATE_NONE);
         }

         /* Clear the bits in the writemask that we just wrote, then try
          * again to see if more channels are left.
          */
         writemask &= (15 << (first_component + num_components));
      }
      break;
   }

   case nir_intrinsic_store_output: {
      fs_reg src = get_nir_src(instr->src[0]);

      nir_const_value *const_offset = nir_src_as_const_value(instr->src[1]);
      assert(const_offset && "Indirect output stores not allowed");

      unsigned num_components = instr->num_components;
      unsigned first_component = nir_intrinsic_component(instr);
      if (nir_src_bit_size(instr->src[0]) == 64) {
         src = shuffle_for_32bit_write(bld, src, 0, num_components);
         num_components *= 2;
      }

      fs_reg new_dest = retype(offset(outputs[instr->const_index[0]], bld,
                                      4 * const_offset->u32[0]), src.type);
      for (unsigned j = 0; j < num_components; j++) {
         bld.MOV(offset(new_dest, bld, j + first_component),
                 offset(src, bld, j));
      }
      break;
   }

   case nir_intrinsic_ssbo_atomic_add:
      nir_emit_ssbo_atomic(bld, get_op_for_atomic_add(instr, 2), instr);
      break;
   case nir_intrinsic_ssbo_atomic_imin:
      nir_emit_ssbo_atomic(bld, BRW_AOP_IMIN, instr);
      break;
   case nir_intrinsic_ssbo_atomic_umin:
      nir_emit_ssbo_atomic(bld, BRW_AOP_UMIN, instr);
      break;
   case nir_intrinsic_ssbo_atomic_imax:
      nir_emit_ssbo_atomic(bld, BRW_AOP_IMAX, instr);
      break;
   case nir_intrinsic_ssbo_atomic_umax:
      nir_emit_ssbo_atomic(bld, BRW_AOP_UMAX, instr);
      break;
   case nir_intrinsic_ssbo_atomic_and:
      nir_emit_ssbo_atomic(bld, BRW_AOP_AND, instr);
      break;
   case nir_intrinsic_ssbo_atomic_or:
      nir_emit_ssbo_atomic(bld, BRW_AOP_OR, instr);
      break;
   case nir_intrinsic_ssbo_atomic_xor:
      nir_emit_ssbo_atomic(bld, BRW_AOP_XOR, instr);
      break;
   case nir_intrinsic_ssbo_atomic_exchange:
      nir_emit_ssbo_atomic(bld, BRW_AOP_MOV, instr);
      break;
   case nir_intrinsic_ssbo_atomic_comp_swap:
      nir_emit_ssbo_atomic(bld, BRW_AOP_CMPWR, instr);
      break;
   case nir_intrinsic_ssbo_atomic_fmin:
      nir_emit_ssbo_atomic_float(bld, BRW_AOP_FMIN, instr);
      break;
   case nir_intrinsic_ssbo_atomic_fmax:
      nir_emit_ssbo_atomic_float(bld, BRW_AOP_FMAX, instr);
      break;
   case nir_intrinsic_ssbo_atomic_fcomp_swap:
      nir_emit_ssbo_atomic_float(bld, BRW_AOP_FCMPWR, instr);
      break;

   case nir_intrinsic_get_buffer_size: {
      nir_const_value *const_uniform_block = nir_src_as_const_value(instr->src[0]);
      unsigned ssbo_index = const_uniform_block ? const_uniform_block->u32[0] : 0;

      /* A resinfo's sampler message is used to get the buffer size.  The
       * SIMD8's writeback message consists of four registers and SIMD16's
       * writeback message consists of 8 destination registers (two per each
       * component).  Because we are only interested on the first channel of
       * the first returned component, where resinfo returns the buffer size
       * for SURFTYPE_BUFFER, we can just use the SIMD8 variant regardless of
       * the dispatch width.
       */
      const fs_builder ubld = bld.exec_all().group(8, 0);
      fs_reg src_payload = ubld.vgrf(BRW_REGISTER_TYPE_UD);
      fs_reg ret_payload = ubld.vgrf(BRW_REGISTER_TYPE_UD, 4);

      /* Set LOD = 0 */
      ubld.MOV(src_payload, brw_imm_d(0));

      const unsigned index = prog_data->binding_table.ssbo_start + ssbo_index;
      fs_inst *inst = ubld.emit(SHADER_OPCODE_GET_BUFFER_SIZE, ret_payload,
                                src_payload, brw_imm_ud(index));
      inst->header_size = 0;
      inst->mlen = 1;
      inst->size_written = 4 * REG_SIZE;

      /* SKL PRM, vol07, 3D Media GPGPU Engine, Bounds Checking and Faulting:
       *
       * "Out-of-bounds checking is always performed at a DWord granularity. If
       * any part of the DWord is out-of-bounds then the whole DWord is
       * considered out-of-bounds."
       *
       * This implies that types with size smaller than 4-bytes need to be
       * padded if they don't complete the last dword of the buffer. But as we
       * need to maintain the original size we need to reverse the padding
       * calculation to return the correct size to know the number of elements
       * of an unsized array. As we stored in the last two bits of the surface
       * size the needed padding for the buffer, we calculate here the
       * original buffer_size reversing the surface_size calculation:
       *
       * surface_size = isl_align(buffer_size, 4) +
       *                (isl_align(buffer_size) - buffer_size)
       *
       * buffer_size = surface_size & ~3 - surface_size & 3
       */

      fs_reg size_aligned4 = ubld.vgrf(BRW_REGISTER_TYPE_UD);
      fs_reg size_padding = ubld.vgrf(BRW_REGISTER_TYPE_UD);
      fs_reg buffer_size = ubld.vgrf(BRW_REGISTER_TYPE_UD);

      ubld.AND(size_padding, ret_payload, brw_imm_ud(3));
      ubld.AND(size_aligned4, ret_payload, brw_imm_ud(~3));
      ubld.ADD(buffer_size, size_aligned4, negate(size_padding));

      bld.MOV(retype(dest, ret_payload.type), component(buffer_size, 0));

      brw_mark_surface_used(prog_data, index);
      break;
   }

   case nir_intrinsic_load_subgroup_invocation:
      bld.MOV(retype(dest, BRW_REGISTER_TYPE_D),
              nir_system_values[SYSTEM_VALUE_SUBGROUP_INVOCATION]);
      break;

   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
      unreachable("not reached");

   case nir_intrinsic_vote_any: {
      const fs_builder ubld = bld.exec_all().group(1, 0);

      /* The any/all predicates do not consider channel enables. To prevent
       * dead channels from affecting the result, we initialize the flag with
       * with the identity value for the logical operation.
       */
      if (dispatch_width == 32) {
         /* For SIMD32, we use a UD type so we fill both f0.0 and f0.1. */
         ubld.MOV(retype(brw_flag_reg(0, 0), BRW_REGISTER_TYPE_UD),
                         brw_imm_ud(0));
      } else {
         ubld.MOV(brw_flag_reg(0, 0), brw_imm_uw(0));
      }
      bld.CMP(bld.null_reg_d(), get_nir_src(instr->src[0]), brw_imm_d(0), BRW_CONDITIONAL_NZ);

      /* For some reason, the any/all predicates don't work properly with
       * SIMD32.  In particular, it appears that a SEL with a QtrCtrl of 2H
       * doesn't read the correct subset of the flag register and you end up
       * getting garbage in the second half.  Work around this by using a pair
       * of 1-wide MOVs and scattering the result.
       */
      fs_reg res1 = ubld.vgrf(BRW_REGISTER_TYPE_D);
      ubld.MOV(res1, brw_imm_d(0));
      set_predicate(dispatch_width == 8  ? BRW_PREDICATE_ALIGN1_ANY8H :
                    dispatch_width == 16 ? BRW_PREDICATE_ALIGN1_ANY16H :
                                           BRW_PREDICATE_ALIGN1_ANY32H,
                    ubld.MOV(res1, brw_imm_d(-1)));

      bld.MOV(retype(dest, BRW_REGISTER_TYPE_D), component(res1, 0));
      break;
   }
   case nir_intrinsic_vote_all: {
      const fs_builder ubld = bld.exec_all().group(1, 0);

      /* The any/all predicates do not consider channel enables. To prevent
       * dead channels from affecting the result, we initialize the flag with
       * with the identity value for the logical operation.
       */
      if (dispatch_width == 32) {
         /* For SIMD32, we use a UD type so we fill both f0.0 and f0.1. */
         ubld.MOV(retype(brw_flag_reg(0, 0), BRW_REGISTER_TYPE_UD),
                         brw_imm_ud(0xffffffff));
      } else {
         ubld.MOV(brw_flag_reg(0, 0), brw_imm_uw(0xffff));
      }
      bld.CMP(bld.null_reg_d(), get_nir_src(instr->src[0]), brw_imm_d(0), BRW_CONDITIONAL_NZ);

      /* For some reason, the any/all predicates don't work properly with
       * SIMD32.  In particular, it appears that a SEL with a QtrCtrl of 2H
       * doesn't read the correct subset of the flag register and you end up
       * getting garbage in the second half.  Work around this by using a pair
       * of 1-wide MOVs and scattering the result.
       */
      fs_reg res1 = ubld.vgrf(BRW_REGISTER_TYPE_D);
      ubld.MOV(res1, brw_imm_d(0));
      set_predicate(dispatch_width == 8  ? BRW_PREDICATE_ALIGN1_ALL8H :
                    dispatch_width == 16 ? BRW_PREDICATE_ALIGN1_ALL16H :
                                           BRW_PREDICATE_ALIGN1_ALL32H,
                    ubld.MOV(res1, brw_imm_d(-1)));

      bld.MOV(retype(dest, BRW_REGISTER_TYPE_D), component(res1, 0));
      break;
   }
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq: {
      fs_reg value = get_nir_src(instr->src[0]);
      if (instr->intrinsic == nir_intrinsic_vote_feq) {
         const unsigned bit_size = nir_src_bit_size(instr->src[0]);
         value.type = brw_reg_type_from_bit_size(bit_size, BRW_REGISTER_TYPE_F);
      }

      fs_reg uniformized = bld.emit_uniformize(value);
      const fs_builder ubld = bld.exec_all().group(1, 0);

      /* The any/all predicates do not consider channel enables. To prevent
       * dead channels from affecting the result, we initialize the flag with
       * with the identity value for the logical operation.
       */
      if (dispatch_width == 32) {
         /* For SIMD32, we use a UD type so we fill both f0.0 and f0.1. */
         ubld.MOV(retype(brw_flag_reg(0, 0), BRW_REGISTER_TYPE_UD),
                         brw_imm_ud(0xffffffff));
      } else {
         ubld.MOV(brw_flag_reg(0, 0), brw_imm_uw(0xffff));
      }
      bld.CMP(bld.null_reg_d(), value, uniformized, BRW_CONDITIONAL_Z);

      /* For some reason, the any/all predicates don't work properly with
       * SIMD32.  In particular, it appears that a SEL with a QtrCtrl of 2H
       * doesn't read the correct subset of the flag register and you end up
       * getting garbage in the second half.  Work around this by using a pair
       * of 1-wide MOVs and scattering the result.
       */
      fs_reg res1 = ubld.vgrf(BRW_REGISTER_TYPE_D);
      ubld.MOV(res1, brw_imm_d(0));
      set_predicate(dispatch_width == 8  ? BRW_PREDICATE_ALIGN1_ALL8H :
                    dispatch_width == 16 ? BRW_PREDICATE_ALIGN1_ALL16H :
                                           BRW_PREDICATE_ALIGN1_ALL32H,
                    ubld.MOV(res1, brw_imm_d(-1)));

      bld.MOV(retype(dest, BRW_REGISTER_TYPE_D), component(res1, 0));
      break;
   }

   case nir_intrinsic_ballot: {
      const fs_reg value = retype(get_nir_src(instr->src[0]),
                                  BRW_REGISTER_TYPE_UD);
      struct brw_reg flag = brw_flag_reg(0, 0);
      /* FIXME: For SIMD32 programs, this causes us to stomp on f0.1 as well
       * as f0.0.  This is a problem for fragment programs as we currently use
       * f0.1 for discards.  Fortunately, we don't support SIMD32 fragment
       * programs yet so this isn't a problem.  When we do, something will
       * have to change.
       */
      if (dispatch_width == 32)
         flag.type = BRW_REGISTER_TYPE_UD;

      bld.exec_all().group(1, 0).MOV(flag, brw_imm_ud(0u));
      bld.CMP(bld.null_reg_ud(), value, brw_imm_ud(0u), BRW_CONDITIONAL_NZ);

      if (instr->dest.ssa.bit_size > 32) {
         dest.type = BRW_REGISTER_TYPE_UQ;
      } else {
         dest.type = BRW_REGISTER_TYPE_UD;
      }
      bld.MOV(dest, flag);
      break;
   }

   case nir_intrinsic_read_invocation: {
      const fs_reg value = get_nir_src(instr->src[0]);
      const fs_reg invocation = get_nir_src(instr->src[1]);
      fs_reg tmp = bld.vgrf(value.type);

      bld.exec_all().emit(SHADER_OPCODE_BROADCAST, tmp, value,
                          bld.emit_uniformize(invocation));

      bld.MOV(retype(dest, value.type), fs_reg(component(tmp, 0)));
      break;
   }

   case nir_intrinsic_read_first_invocation: {
      const fs_reg value = get_nir_src(instr->src[0]);
      bld.MOV(retype(dest, value.type), bld.emit_uniformize(value));
      break;
   }

   case nir_intrinsic_shuffle: {
      const fs_reg value = get_nir_src(instr->src[0]);
      const fs_reg index = get_nir_src(instr->src[1]);

      bld.emit(SHADER_OPCODE_SHUFFLE, retype(dest, value.type), value, index);
      break;
   }

   case nir_intrinsic_first_invocation: {
      fs_reg tmp = bld.vgrf(BRW_REGISTER_TYPE_UD);
      bld.exec_all().emit(SHADER_OPCODE_FIND_LIVE_CHANNEL, tmp);
      bld.MOV(retype(dest, BRW_REGISTER_TYPE_UD),
              fs_reg(component(tmp, 0)));
      break;
   }

   case nir_intrinsic_quad_broadcast: {
      const fs_reg value = get_nir_src(instr->src[0]);
      nir_const_value *index = nir_src_as_const_value(instr->src[1]);
      assert(nir_src_bit_size(instr->src[1]) == 32);

      bld.emit(SHADER_OPCODE_CLUSTER_BROADCAST, retype(dest, value.type),
               value, brw_imm_ud(index->u32[0]), brw_imm_ud(4));
      break;
   }

   case nir_intrinsic_quad_swap_horizontal: {
      const fs_reg value = get_nir_src(instr->src[0]);
      const fs_reg tmp = bld.vgrf(value.type);
      const fs_builder ubld = bld.exec_all().group(dispatch_width / 2, 0);

      const fs_reg src_left = horiz_stride(value, 2);
      const fs_reg src_right = horiz_stride(horiz_offset(value, 1), 2);
      const fs_reg tmp_left = horiz_stride(tmp, 2);
      const fs_reg tmp_right = horiz_stride(horiz_offset(tmp, 1), 2);

      /* From the Cherryview PRM Vol. 7, "Register Region Restrictiosn":
       *
       *    "When source or destination datatype is 64b or operation is
       *    integer DWord multiply, regioning in Align1 must follow
       *    these rules:
       *
       *    [...]
       *
       *    3. Source and Destination offset must be the same, except
       *       the case of scalar source."
       *
       * In order to work around this, we have to emit two 32-bit MOVs instead
       * of a single 64-bit MOV to do the shuffle.
       */
      if (type_sz(value.type) > 4 &&
          (devinfo->is_cherryview || gen_device_info_is_9lp(devinfo))) {
         ubld.MOV(subscript(tmp_left, BRW_REGISTER_TYPE_D, 0),
                  subscript(src_right, BRW_REGISTER_TYPE_D, 0));
         ubld.MOV(subscript(tmp_left, BRW_REGISTER_TYPE_D, 1),
                  subscript(src_right, BRW_REGISTER_TYPE_D, 1));
         ubld.MOV(subscript(tmp_right, BRW_REGISTER_TYPE_D, 0),
                  subscript(src_left, BRW_REGISTER_TYPE_D, 0));
         ubld.MOV(subscript(tmp_right, BRW_REGISTER_TYPE_D, 1),
                  subscript(src_left, BRW_REGISTER_TYPE_D, 1));
      } else {
         ubld.MOV(tmp_left, src_right);
         ubld.MOV(tmp_right, src_left);
      }
      bld.MOV(retype(dest, value.type), tmp);
      break;
   }

   case nir_intrinsic_quad_swap_vertical: {
      const fs_reg value = get_nir_src(instr->src[0]);
      if (nir_src_bit_size(instr->src[0]) == 32) {
         /* For 32-bit, we can use a SIMD4x2 instruction to do this easily */
         const fs_reg tmp = bld.vgrf(value.type);
         const fs_builder ubld = bld.exec_all();
         ubld.emit(SHADER_OPCODE_QUAD_SWIZZLE, tmp, value,
                   brw_imm_ud(BRW_SWIZZLE4(2,3,0,1)));
         bld.MOV(retype(dest, value.type), tmp);
      } else {
         /* For larger data types, we have to either emit dispatch_width many
          * MOVs or else fall back to doing indirects.
          */
         fs_reg idx = bld.vgrf(BRW_REGISTER_TYPE_W);
         bld.XOR(idx, nir_system_values[SYSTEM_VALUE_SUBGROUP_INVOCATION],
                      brw_imm_w(0x2));
         bld.emit(SHADER_OPCODE_SHUFFLE, retype(dest, value.type), value, idx);
      }
      break;
   }

   case nir_intrinsic_quad_swap_diagonal: {
      const fs_reg value = get_nir_src(instr->src[0]);
      if (nir_src_bit_size(instr->src[0]) == 32) {
         /* For 32-bit, we can use a SIMD4x2 instruction to do this easily */
         const fs_reg tmp = bld.vgrf(value.type);
         const fs_builder ubld = bld.exec_all();
         ubld.emit(SHADER_OPCODE_QUAD_SWIZZLE, tmp, value,
                   brw_imm_ud(BRW_SWIZZLE4(3,2,1,0)));
         bld.MOV(retype(dest, value.type), tmp);
      } else {
         /* For larger data types, we have to either emit dispatch_width many
          * MOVs or else fall back to doing indirects.
          */
         fs_reg idx = bld.vgrf(BRW_REGISTER_TYPE_W);
         bld.XOR(idx, nir_system_values[SYSTEM_VALUE_SUBGROUP_INVOCATION],
                      brw_imm_w(0x3));
         bld.emit(SHADER_OPCODE_SHUFFLE, retype(dest, value.type), value, idx);
      }
      break;
   }

   case nir_intrinsic_reduce: {
      fs_reg src = get_nir_src(instr->src[0]);
      nir_op redop = (nir_op)nir_intrinsic_reduction_op(instr);
      unsigned cluster_size = nir_intrinsic_cluster_size(instr);
      if (cluster_size == 0 || cluster_size > dispatch_width)
         cluster_size = dispatch_width;

      /* Figure out the source type */
      src.type = brw_type_for_nir_type(devinfo,
         (nir_alu_type)(nir_op_infos[redop].input_types[0] |
                        nir_src_bit_size(instr->src[0])));

      fs_reg identity = brw_nir_reduction_op_identity(bld, redop, src.type);
      opcode brw_op = brw_op_for_nir_reduction_op(redop);
      brw_conditional_mod cond_mod = brw_cond_mod_for_nir_reduction_op(redop);

      /* Set up a register for all of our scratching around and initialize it
       * to reduction operation's identity value.
       */
      fs_reg scan = bld.vgrf(src.type);
      bld.exec_all().emit(SHADER_OPCODE_SEL_EXEC, scan, src, identity);

      bld.emit_scan(brw_op, scan, cluster_size, cond_mod);

      dest.type = src.type;
      if (cluster_size * type_sz(src.type) >= REG_SIZE * 2) {
         /* In this case, CLUSTER_BROADCAST instruction isn't needed because
          * the distance between clusters is at least 2 GRFs.  In this case,
          * we don't need the weird striding of the CLUSTER_BROADCAST
          * instruction and can just do regular MOVs.
          */
         assert((cluster_size * type_sz(src.type)) % (REG_SIZE * 2) == 0);
         const unsigned groups =
            (dispatch_width * type_sz(src.type)) / (REG_SIZE * 2);
         const unsigned group_size = dispatch_width / groups;
         for (unsigned i = 0; i < groups; i++) {
            const unsigned cluster = (i * group_size) / cluster_size;
            const unsigned comp = cluster * cluster_size + (cluster_size - 1);
            bld.group(group_size, i).MOV(horiz_offset(dest, i * group_size),
                                         component(scan, comp));
         }
      } else {
         bld.emit(SHADER_OPCODE_CLUSTER_BROADCAST, dest, scan,
                  brw_imm_ud(cluster_size - 1), brw_imm_ud(cluster_size));
      }
      break;
   }

   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan: {
      fs_reg src = get_nir_src(instr->src[0]);
      nir_op redop = (nir_op)nir_intrinsic_reduction_op(instr);

      /* Figure out the source type */
      src.type = brw_type_for_nir_type(devinfo,
         (nir_alu_type)(nir_op_infos[redop].input_types[0] |
                        nir_src_bit_size(instr->src[0])));

      fs_reg identity = brw_nir_reduction_op_identity(bld, redop, src.type);
      opcode brw_op = brw_op_for_nir_reduction_op(redop);
      brw_conditional_mod cond_mod = brw_cond_mod_for_nir_reduction_op(redop);

      /* Set up a register for all of our scratching around and initialize it
       * to reduction operation's identity value.
       */
      fs_reg scan = bld.vgrf(src.type);
      const fs_builder allbld = bld.exec_all();
      allbld.emit(SHADER_OPCODE_SEL_EXEC, scan, src, identity);

      if (instr->intrinsic == nir_intrinsic_exclusive_scan) {
         /* Exclusive scan is a bit harder because we have to do an annoying
          * shift of the contents before we can begin.  To make things worse,
          * we can't do this with a normal stride; we have to use indirects.
          */
         fs_reg shifted = bld.vgrf(src.type);
         fs_reg idx = bld.vgrf(BRW_REGISTER_TYPE_W);
         allbld.ADD(idx, nir_system_values[SYSTEM_VALUE_SUBGROUP_INVOCATION],
                         brw_imm_w(-1));
         allbld.emit(SHADER_OPCODE_SHUFFLE, shifted, scan, idx);
         allbld.group(1, 0).MOV(component(shifted, 0), identity);
         scan = shifted;
      }

      bld.emit_scan(brw_op, scan, dispatch_width, cond_mod);

      bld.MOV(retype(dest, src.type), scan);
      break;
   }

   case nir_intrinsic_begin_invocation_interlock: {
      const fs_builder ubld = bld.group(8, 0);
      const fs_reg tmp = ubld.vgrf(BRW_REGISTER_TYPE_UD, 2);

      ubld.emit(SHADER_OPCODE_INTERLOCK, tmp)->size_written = 2 *
         REG_SIZE;

      break;
   }

   case nir_intrinsic_end_invocation_interlock: {
      /* We don't need to do anything here */
      break;
   }

   default:
      unreachable("unknown intrinsic");
   }
}

void
fs_visitor::nir_emit_ssbo_atomic(const fs_builder &bld,
                                 int op, nir_intrinsic_instr *instr)
{
   if (stage == MESA_SHADER_FRAGMENT)
      brw_wm_prog_data(prog_data)->has_side_effects = true;

   fs_reg dest;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dest = get_nir_dest(instr->dest);

   fs_reg surface;
   nir_const_value *const_surface = nir_src_as_const_value(instr->src[0]);
   if (const_surface) {
      unsigned surf_index = stage_prog_data->binding_table.ssbo_start +
                            const_surface->u32[0];
      surface = brw_imm_ud(surf_index);
      brw_mark_surface_used(prog_data, surf_index);
   } else {
      surface = vgrf(glsl_type::uint_type);
      bld.ADD(surface, get_nir_src(instr->src[0]),
              brw_imm_ud(stage_prog_data->binding_table.ssbo_start));

      /* Assume this may touch any SSBO. This is the same we do for other
       * UBO/SSBO accesses with non-constant surface.
       */
      brw_mark_surface_used(prog_data,
                            stage_prog_data->binding_table.ssbo_start +
                            nir->info.num_ssbos - 1);
   }

   fs_reg offset = get_nir_src(instr->src[1]);
   fs_reg data1;
   if (op != BRW_AOP_INC && op != BRW_AOP_DEC && op != BRW_AOP_PREDEC)
      data1 = get_nir_src(instr->src[2]);
   fs_reg data2;
   if (op == BRW_AOP_CMPWR)
      data2 = get_nir_src(instr->src[3]);

   /* Emit the actual atomic operation */

   fs_reg atomic_result = emit_untyped_atomic(bld, surface, offset,
                                              data1, data2,
                                              1 /* dims */, 1 /* rsize */,
                                              op,
                                              BRW_PREDICATE_NONE);
   dest.type = atomic_result.type;
   bld.MOV(dest, atomic_result);
}

void
fs_visitor::nir_emit_ssbo_atomic_float(const fs_builder &bld,
                                       int op, nir_intrinsic_instr *instr)
{
   if (stage == MESA_SHADER_FRAGMENT)
      brw_wm_prog_data(prog_data)->has_side_effects = true;

   fs_reg dest;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dest = get_nir_dest(instr->dest);

   fs_reg surface;
   nir_const_value *const_surface = nir_src_as_const_value(instr->src[0]);
   if (const_surface) {
      unsigned surf_index = stage_prog_data->binding_table.ssbo_start +
                            const_surface->u32[0];
      surface = brw_imm_ud(surf_index);
      brw_mark_surface_used(prog_data, surf_index);
   } else {
      surface = vgrf(glsl_type::uint_type);
      bld.ADD(surface, get_nir_src(instr->src[0]),
              brw_imm_ud(stage_prog_data->binding_table.ssbo_start));

      /* Assume this may touch any SSBO. This is the same we do for other
       * UBO/SSBO accesses with non-constant surface.
       */
      brw_mark_surface_used(prog_data,
                            stage_prog_data->binding_table.ssbo_start +
                            nir->info.num_ssbos - 1);
   }

   fs_reg offset = get_nir_src(instr->src[1]);
   fs_reg data1 = get_nir_src(instr->src[2]);
   fs_reg data2;
   if (op == BRW_AOP_FCMPWR)
      data2 = get_nir_src(instr->src[3]);

   /* Emit the actual atomic operation */

   fs_reg atomic_result = emit_untyped_atomic_float(bld, surface, offset,
                                                    data1, data2,
                                                    1 /* dims */, 1 /* rsize */,
                                                    op,
                                                    BRW_PREDICATE_NONE);
   dest.type = atomic_result.type;
   bld.MOV(dest, atomic_result);
}

void
fs_visitor::nir_emit_shared_atomic(const fs_builder &bld,
                                   int op, nir_intrinsic_instr *instr)
{
   fs_reg dest;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dest = get_nir_dest(instr->dest);

   fs_reg surface = brw_imm_ud(GEN7_BTI_SLM);
   fs_reg offset;
   fs_reg data1;
   if (op != BRW_AOP_INC && op != BRW_AOP_DEC && op != BRW_AOP_PREDEC)
      data1 = get_nir_src(instr->src[1]);
   fs_reg data2;
   if (op == BRW_AOP_CMPWR)
      data2 = get_nir_src(instr->src[2]);

   /* Get the offset */
   nir_const_value *const_offset = nir_src_as_const_value(instr->src[0]);
   if (const_offset) {
      offset = brw_imm_ud(instr->const_index[0] + const_offset->u32[0]);
   } else {
      offset = vgrf(glsl_type::uint_type);
      bld.ADD(offset,
	      retype(get_nir_src(instr->src[0]), BRW_REGISTER_TYPE_UD),
	      brw_imm_ud(instr->const_index[0]));
   }

   /* Emit the actual atomic operation operation */

   fs_reg atomic_result = emit_untyped_atomic(bld, surface, offset,
                                              data1, data2,
                                              1 /* dims */, 1 /* rsize */,
                                              op,
                                              BRW_PREDICATE_NONE);
   dest.type = atomic_result.type;
   bld.MOV(dest, atomic_result);
}

void
fs_visitor::nir_emit_shared_atomic_float(const fs_builder &bld,
                                         int op, nir_intrinsic_instr *instr)
{
   fs_reg dest;
   if (nir_intrinsic_infos[instr->intrinsic].has_dest)
      dest = get_nir_dest(instr->dest);

   fs_reg surface = brw_imm_ud(GEN7_BTI_SLM);
   fs_reg offset;
   fs_reg data1 = get_nir_src(instr->src[1]);
   fs_reg data2;
   if (op == BRW_AOP_FCMPWR)
      data2 = get_nir_src(instr->src[2]);

   /* Get the offset */
   nir_const_value *const_offset = nir_src_as_const_value(instr->src[0]);
   if (const_offset) {
      offset = brw_imm_ud(instr->const_index[0] + const_offset->u32[0]);
   } else {
      offset = vgrf(glsl_type::uint_type);
      bld.ADD(offset,
              retype(get_nir_src(instr->src[0]), BRW_REGISTER_TYPE_UD),
              brw_imm_ud(instr->const_index[0]));
   }

   /* Emit the actual atomic operation operation */

   fs_reg atomic_result = emit_untyped_atomic_float(bld, surface, offset,
                                                    data1, data2,
                                                    1 /* dims */, 1 /* rsize */,
                                                    op,
                                                    BRW_PREDICATE_NONE);
   dest.type = atomic_result.type;
   bld.MOV(dest, atomic_result);
}

void
fs_visitor::nir_emit_texture(const fs_builder &bld, nir_tex_instr *instr)
{
   unsigned texture = instr->texture_index;
   unsigned sampler = instr->sampler_index;

   fs_reg srcs[TEX_LOGICAL_NUM_SRCS];

   srcs[TEX_LOGICAL_SRC_SURFACE] = brw_imm_ud(texture);
   srcs[TEX_LOGICAL_SRC_SAMPLER] = brw_imm_ud(sampler);

   int lod_components = 0;

   /* The hardware requires a LOD for buffer textures */
   if (instr->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      srcs[TEX_LOGICAL_SRC_LOD] = brw_imm_d(0);

   uint32_t header_bits = 0;
   for (unsigned i = 0; i < instr->num_srcs; i++) {
      fs_reg src = get_nir_src(instr->src[i].src);
      switch (instr->src[i].src_type) {
      case nir_tex_src_bias:
         srcs[TEX_LOGICAL_SRC_LOD] =
            retype(get_nir_src_imm(instr->src[i].src), BRW_REGISTER_TYPE_F);
         break;
      case nir_tex_src_comparator:
         srcs[TEX_LOGICAL_SRC_SHADOW_C] = retype(src, BRW_REGISTER_TYPE_F);
         break;
      case nir_tex_src_coord:
         switch (instr->op) {
         case nir_texop_txf:
         case nir_texop_txf_ms:
         case nir_texop_txf_ms_mcs:
         case nir_texop_samples_identical:
            srcs[TEX_LOGICAL_SRC_COORDINATE] = retype(src, BRW_REGISTER_TYPE_D);
            break;
         default:
            srcs[TEX_LOGICAL_SRC_COORDINATE] = retype(src, BRW_REGISTER_TYPE_F);
            break;
         }
         break;
      case nir_tex_src_ddx:
         srcs[TEX_LOGICAL_SRC_LOD] = retype(src, BRW_REGISTER_TYPE_F);
         lod_components = nir_tex_instr_src_size(instr, i);
         break;
      case nir_tex_src_ddy:
         srcs[TEX_LOGICAL_SRC_LOD2] = retype(src, BRW_REGISTER_TYPE_F);
         break;
      case nir_tex_src_lod:
         switch (instr->op) {
         case nir_texop_txs:
            srcs[TEX_LOGICAL_SRC_LOD] =
               retype(get_nir_src_imm(instr->src[i].src), BRW_REGISTER_TYPE_UD);
            break;
         case nir_texop_txf:
            srcs[TEX_LOGICAL_SRC_LOD] =
               retype(get_nir_src_imm(instr->src[i].src), BRW_REGISTER_TYPE_D);
            break;
         default:
            srcs[TEX_LOGICAL_SRC_LOD] =
               retype(get_nir_src_imm(instr->src[i].src), BRW_REGISTER_TYPE_F);
            break;
         }
         break;
      case nir_tex_src_ms_index:
         srcs[TEX_LOGICAL_SRC_SAMPLE_INDEX] = retype(src, BRW_REGISTER_TYPE_UD);
         break;

      case nir_tex_src_offset: {
         nir_const_value *const_offset =
            nir_src_as_const_value(instr->src[i].src);
         unsigned offset_bits = 0;
         if (const_offset &&
             brw_texture_offset(const_offset->i32,
                                nir_tex_instr_src_size(instr, i),
                                &offset_bits)) {
            header_bits |= offset_bits;
         } else {
            srcs[TEX_LOGICAL_SRC_TG4_OFFSET] =
               retype(src, BRW_REGISTER_TYPE_D);
         }
         break;
      }

      case nir_tex_src_projector:
         unreachable("should be lowered");

      case nir_tex_src_texture_offset: {
         /* Figure out the highest possible texture index and mark it as used */
         uint32_t max_used = texture + instr->texture_array_size - 1;
         if (instr->op == nir_texop_tg4 && devinfo->gen < 8) {
            max_used += stage_prog_data->binding_table.gather_texture_start;
         } else {
            max_used += stage_prog_data->binding_table.texture_start;
         }
         brw_mark_surface_used(prog_data, max_used);

         /* Emit code to evaluate the actual indexing expression */
         fs_reg tmp = vgrf(glsl_type::uint_type);
         bld.ADD(tmp, src, brw_imm_ud(texture));
         srcs[TEX_LOGICAL_SRC_SURFACE] = bld.emit_uniformize(tmp);
         break;
      }

      case nir_tex_src_sampler_offset: {
         /* Emit code to evaluate the actual indexing expression */
         fs_reg tmp = vgrf(glsl_type::uint_type);
         bld.ADD(tmp, src, brw_imm_ud(sampler));
         srcs[TEX_LOGICAL_SRC_SAMPLER] = bld.emit_uniformize(tmp);
         break;
      }

      case nir_tex_src_ms_mcs:
         assert(instr->op == nir_texop_txf_ms);
         srcs[TEX_LOGICAL_SRC_MCS] = retype(src, BRW_REGISTER_TYPE_D);
         break;

      case nir_tex_src_plane: {
         nir_const_value *const_plane =
            nir_src_as_const_value(instr->src[i].src);
         const uint32_t plane = const_plane->u32[0];
         const uint32_t texture_index =
            instr->texture_index +
            stage_prog_data->binding_table.plane_start[plane] -
            stage_prog_data->binding_table.texture_start;

         srcs[TEX_LOGICAL_SRC_SURFACE] = brw_imm_ud(texture_index);
         break;
      }

      default:
         unreachable("unknown texture source");
      }
   }

   if (srcs[TEX_LOGICAL_SRC_MCS].file == BAD_FILE &&
       (instr->op == nir_texop_txf_ms ||
        instr->op == nir_texop_samples_identical)) {
      if (devinfo->gen >= 7 &&
          key_tex->compressed_multisample_layout_mask & (1 << texture)) {
         srcs[TEX_LOGICAL_SRC_MCS] =
            emit_mcs_fetch(srcs[TEX_LOGICAL_SRC_COORDINATE],
                           instr->coord_components,
                           srcs[TEX_LOGICAL_SRC_SURFACE]);
      } else {
         srcs[TEX_LOGICAL_SRC_MCS] = brw_imm_ud(0u);
      }
   }

   srcs[TEX_LOGICAL_SRC_COORD_COMPONENTS] = brw_imm_d(instr->coord_components);
   srcs[TEX_LOGICAL_SRC_GRAD_COMPONENTS] = brw_imm_d(lod_components);

   enum opcode opcode;
   switch (instr->op) {
   case nir_texop_tex:
      opcode = (stage == MESA_SHADER_FRAGMENT ? SHADER_OPCODE_TEX_LOGICAL :
                SHADER_OPCODE_TXL_LOGICAL);
      break;
   case nir_texop_txb:
      opcode = FS_OPCODE_TXB_LOGICAL;
      break;
   case nir_texop_txl:
      opcode = SHADER_OPCODE_TXL_LOGICAL;
      break;
   case nir_texop_txd:
      opcode = SHADER_OPCODE_TXD_LOGICAL;
      break;
   case nir_texop_txf:
      opcode = SHADER_OPCODE_TXF_LOGICAL;
      break;
   case nir_texop_txf_ms:
      if ((key_tex->msaa_16 & (1 << sampler)))
         opcode = SHADER_OPCODE_TXF_CMS_W_LOGICAL;
      else
         opcode = SHADER_OPCODE_TXF_CMS_LOGICAL;
      break;
   case nir_texop_txf_ms_mcs:
      opcode = SHADER_OPCODE_TXF_MCS_LOGICAL;
      break;
   case nir_texop_query_levels:
   case nir_texop_txs:
      opcode = SHADER_OPCODE_TXS_LOGICAL;
      break;
   case nir_texop_lod:
      opcode = SHADER_OPCODE_LOD_LOGICAL;
      break;
   case nir_texop_tg4:
      if (srcs[TEX_LOGICAL_SRC_TG4_OFFSET].file != BAD_FILE)
         opcode = SHADER_OPCODE_TG4_OFFSET_LOGICAL;
      else
         opcode = SHADER_OPCODE_TG4_LOGICAL;
      break;
   case nir_texop_texture_samples:
      opcode = SHADER_OPCODE_SAMPLEINFO_LOGICAL;
      break;
   case nir_texop_samples_identical: {
      fs_reg dst = retype(get_nir_dest(instr->dest), BRW_REGISTER_TYPE_D);

      /* If mcs is an immediate value, it means there is no MCS.  In that case
       * just return false.
       */
      if (srcs[TEX_LOGICAL_SRC_MCS].file == BRW_IMMEDIATE_VALUE) {
         bld.MOV(dst, brw_imm_ud(0u));
      } else if ((key_tex->msaa_16 & (1 << sampler))) {
         fs_reg tmp = vgrf(glsl_type::uint_type);
         bld.OR(tmp, srcs[TEX_LOGICAL_SRC_MCS],
                offset(srcs[TEX_LOGICAL_SRC_MCS], bld, 1));
         bld.CMP(dst, tmp, brw_imm_ud(0u), BRW_CONDITIONAL_EQ);
      } else {
         bld.CMP(dst, srcs[TEX_LOGICAL_SRC_MCS], brw_imm_ud(0u),
                 BRW_CONDITIONAL_EQ);
      }
      return;
   }
   default:
      unreachable("unknown texture opcode");
   }

   if (instr->op == nir_texop_tg4) {
      if (instr->component == 1 &&
          key_tex->gather_channel_quirk_mask & (1 << texture)) {
         /* gather4 sampler is broken for green channel on RG32F --
          * we must ask for blue instead.
          */
         header_bits |= 2 << 16;
      } else {
         header_bits |= instr->component << 16;
      }
   }

   fs_reg dst = bld.vgrf(brw_type_for_nir_type(devinfo, instr->dest_type), 4);
   fs_inst *inst = bld.emit(opcode, dst, srcs, ARRAY_SIZE(srcs));
   inst->offset = header_bits;

   const unsigned dest_size = nir_tex_instr_dest_size(instr);
   if (devinfo->gen >= 9 &&
       instr->op != nir_texop_tg4 && instr->op != nir_texop_query_levels) {
      unsigned write_mask = instr->dest.is_ssa ?
                            nir_ssa_def_components_read(&instr->dest.ssa):
                            (1 << dest_size) - 1;
      assert(write_mask != 0); /* dead code should have been eliminated */
      inst->size_written = util_last_bit(write_mask) *
                           inst->dst.component_size(inst->exec_size);
   } else {
      inst->size_written = 4 * inst->dst.component_size(inst->exec_size);
   }

   if (srcs[TEX_LOGICAL_SRC_SHADOW_C].file != BAD_FILE)
      inst->shadow_compare = true;

   if (instr->op == nir_texop_tg4 && devinfo->gen == 6)
      emit_gen6_gather_wa(key_tex->gen6_gather_wa[texture], dst);

   fs_reg nir_dest[4];
   for (unsigned i = 0; i < dest_size; i++)
      nir_dest[i] = offset(dst, bld, i);

   if (instr->op == nir_texop_query_levels) {
      /* # levels is in .w */
      nir_dest[0] = offset(dst, bld, 3);
   } else if (instr->op == nir_texop_txs &&
              dest_size >= 3 && devinfo->gen < 7) {
      /* Gen4-6 return 0 instead of 1 for single layer surfaces. */
      fs_reg depth = offset(dst, bld, 2);
      nir_dest[2] = vgrf(glsl_type::int_type);
      bld.emit_minmax(nir_dest[2], depth, brw_imm_d(1), BRW_CONDITIONAL_GE);
   }

   bld.LOAD_PAYLOAD(get_nir_dest(instr->dest), nir_dest, dest_size, 0);
}

void
fs_visitor::nir_emit_jump(const fs_builder &bld, nir_jump_instr *instr)
{
   switch (instr->type) {
   case nir_jump_break:
      bld.emit(BRW_OPCODE_BREAK);
      break;
   case nir_jump_continue:
      bld.emit(BRW_OPCODE_CONTINUE);
      break;
   case nir_jump_return:
   default:
      unreachable("unknown jump");
   }
}

/*
 * This helper takes a source register and un/shuffles it into the destination
 * register.
 *
 * If source type size is smaller than destination type size the operation
 * needed is a component shuffle. The opposite case would be an unshuffle. If
 * source/destination type size is equal a shuffle is done that would be
 * equivalent to a simple MOV.
 *
 * For example, if source is a 16-bit type and destination is 32-bit. A 3
 * components .xyz 16-bit vector on SIMD8 would be.
 *
 *    |x1|x2|x3|x4|x5|x6|x7|x8|y1|y2|y3|y4|y5|y6|y7|y8|
 *    |z1|z2|z3|z4|z5|z6|z7|z8|  |  |  |  |  |  |  |  |
 *
 * This helper will return the following 2 32-bit components with the 16-bit
 * values shuffled:
 *
 *    |x1 y1|x2 y2|x3 y3|x4 y4|x5 y5|x6 y6|x7 y7|x8 y8|
 *    |z1   |z2   |z3   |z4   |z5   |z6   |z7   |z8   |
 *
 * For unshuffle, the example would be the opposite, a 64-bit type source
 * and a 32-bit destination. A 2 component .xy 64-bit vector on SIMD8
 * would be:
 *
 *    | x1l   x1h | x2l   x2h | x3l   x3h | x4l   x4h |
 *    | x5l   x5h | x6l   x6h | x7l   x7h | x8l   x8h |
 *    | y1l   y1h | y2l   y2h | y3l   y3h | y4l   y4h |
 *    | y5l   y5h | y6l   y6h | y7l   y7h | y8l   y8h |
 *
 * The returned result would be the following 4 32-bit components unshuffled:
 *
 *    | x1l | x2l | x3l | x4l | x5l | x6l | x7l | x8l |
 *    | x1h | x2h | x3h | x4h | x5h | x6h | x7h | x8h |
 *    | y1l | y2l | y3l | y4l | y5l | y6l | y7l | y8l |
 *    | y1h | y2h | y3h | y4h | y5h | y6h | y7h | y8h |
 *
 * - Source and destination register must not be overlapped.
 * - components units are measured in terms of the smaller type between
 *   source and destination because we are un/shuffling the smaller
 *   components from/into the bigger ones.
 * - first_component parameter allows skipping source components.
 */
void
shuffle_src_to_dst(const fs_builder &bld,
                   const fs_reg &dst,
                   const fs_reg &src,
                   uint32_t first_component,
                   uint32_t components)
{
   if (type_sz(src.type) == type_sz(dst.type)) {
      assert(!regions_overlap(dst,
         type_sz(dst.type) * bld.dispatch_width() * components,
         offset(src, bld, first_component),
         type_sz(src.type) * bld.dispatch_width() * components));
      for (unsigned i = 0; i < components; i++) {
         bld.MOV(retype(offset(dst, bld, i), src.type),
                 offset(src, bld, i + first_component));
      }
   } else if (type_sz(src.type) < type_sz(dst.type)) {
      /* Source is shuffled into destination */
      unsigned size_ratio = type_sz(dst.type) / type_sz(src.type);
      assert(!regions_overlap(dst,
         type_sz(dst.type) * bld.dispatch_width() *
         DIV_ROUND_UP(components, size_ratio),
         offset(src, bld, first_component),
         type_sz(src.type) * bld.dispatch_width() * components));

      brw_reg_type shuffle_type =
         brw_reg_type_from_bit_size(8 * type_sz(src.type),
                                    BRW_REGISTER_TYPE_D);
      for (unsigned i = 0; i < components; i++) {
         fs_reg shuffle_component_i =
            subscript(offset(dst, bld, i / size_ratio),
                      shuffle_type, i % size_ratio);
         bld.MOV(shuffle_component_i,
                 retype(offset(src, bld, i + first_component), shuffle_type));
      }
   } else {
      /* Source is unshuffled into destination */
      unsigned size_ratio = type_sz(src.type) / type_sz(dst.type);
      assert(!regions_overlap(dst,
         type_sz(dst.type) * bld.dispatch_width() * components,
         offset(src, bld, first_component / size_ratio),
         type_sz(src.type) * bld.dispatch_width() *
         DIV_ROUND_UP(components + (first_component % size_ratio),
                      size_ratio)));

      brw_reg_type shuffle_type =
         brw_reg_type_from_bit_size(8 * type_sz(dst.type),
                                    BRW_REGISTER_TYPE_D);
      for (unsigned i = 0; i < components; i++) {
         fs_reg shuffle_component_i =
            subscript(offset(src, bld, (first_component + i) / size_ratio),
                      shuffle_type, (first_component + i) % size_ratio);
         bld.MOV(retype(offset(dst, bld, i), shuffle_type),
                 shuffle_component_i);
      }
   }
}

void
shuffle_from_32bit_read(const fs_builder &bld,
                        const fs_reg &dst,
                        const fs_reg &src,
                        uint32_t first_component,
                        uint32_t components)
{
   assert(type_sz(src.type) == 4);

   /* This function takes components in units of the destination type while
    * shuffle_src_to_dst takes components in units of the smallest type
    */
   if (type_sz(dst.type) > 4) {
      assert(type_sz(dst.type) == 8);
      first_component *= 2;
      components *= 2;
   }

   shuffle_src_to_dst(bld, dst, src, first_component, components);
}

fs_reg
shuffle_for_32bit_write(const fs_builder &bld,
                        const fs_reg &src,
                        uint32_t first_component,
                        uint32_t components)
{
   fs_reg dst = bld.vgrf(BRW_REGISTER_TYPE_D,
                         DIV_ROUND_UP (components * type_sz(src.type), 4));
   /* This function takes components in units of the source type while
    * shuffle_src_to_dst takes components in units of the smallest type
    */
   if (type_sz(src.type) > 4) {
      assert(type_sz(src.type) == 8);
      first_component *= 2;
      components *= 2;
   }

   shuffle_src_to_dst(bld, dst, src, first_component, components);

   return dst;
}

fs_reg
setup_imm_df(const fs_builder &bld, double v)
{
   const struct gen_device_info *devinfo = bld.shader->devinfo;
   assert(devinfo->gen >= 7);

   if (devinfo->gen >= 8)
      return brw_imm_df(v);

   /* gen7.5 does not support DF immediates straighforward but the DIM
    * instruction allows to set the 64-bit immediate value.
    */
   if (devinfo->is_haswell) {
      const fs_builder ubld = bld.exec_all().group(1, 0);
      fs_reg dst = ubld.vgrf(BRW_REGISTER_TYPE_DF, 1);
      ubld.DIM(dst, brw_imm_df(v));
      return component(dst, 0);
   }

   /* gen7 does not support DF immediates, so we generate a 64-bit constant by
    * writing the low 32-bit of the constant to suboffset 0 of a VGRF and
    * the high 32-bit to suboffset 4 and then applying a stride of 0.
    *
    * Alternatively, we could also produce a normal VGRF (without stride 0)
    * by writing to all the channels in the VGRF, however, that would hit the
    * gen7 bug where we have to split writes that span more than 1 register
    * into instructions with a width of 4 (otherwise the write to the second
    * register written runs into an execmask hardware bug) which isn't very
    * nice.
    */
   union {
      double d;
      struct {
         uint32_t i1;
         uint32_t i2;
      };
   } di;

   di.d = v;

   const fs_builder ubld = bld.exec_all().group(1, 0);
   const fs_reg tmp = ubld.vgrf(BRW_REGISTER_TYPE_UD, 2);
   ubld.MOV(tmp, brw_imm_ud(di.i1));
   ubld.MOV(horiz_offset(tmp, 1), brw_imm_ud(di.i2));

   return component(retype(tmp, BRW_REGISTER_TYPE_DF), 0);
}

fs_reg
setup_imm_b(const fs_builder &bld, int8_t v)
{
   const fs_reg tmp = bld.vgrf(BRW_REGISTER_TYPE_B);
   bld.MOV(tmp, brw_imm_w(v));
   return tmp;
}

fs_reg
setup_imm_ub(const fs_builder &bld, uint8_t v)
{
   const fs_reg tmp = bld.vgrf(BRW_REGISTER_TYPE_UB);
   bld.MOV(tmp, brw_imm_uw(v));
   return tmp;
}
