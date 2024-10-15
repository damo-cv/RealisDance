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

#include "brw_vec4_gs_visitor.h"

namespace brw {

void
vec4_gs_visitor::nir_setup_inputs()
{
}

void
vec4_gs_visitor::nir_emit_intrinsic(nir_intrinsic_instr *instr)
{
   dst_reg dest;
   src_reg src;

   switch (instr->intrinsic) {
   case nir_intrinsic_load_per_vertex_input: {
      /* The EmitNoIndirectInput flag guarantees our vertex index will
       * be constant.  We should handle indirects someday.
       */
      nir_const_value *vertex = nir_src_as_const_value(instr->src[0]);
      nir_const_value *offset_reg = nir_src_as_const_value(instr->src[1]);

      const unsigned input_array_stride = prog_data->urb_read_length * 2;

      if (nir_dest_bit_size(instr->dest) == 64) {
         src = src_reg(ATTR, input_array_stride * vertex->u32[0] +
                       instr->const_index[0] + offset_reg->u32[0],
                       glsl_type::dvec4_type);

         dst_reg tmp = dst_reg(this, glsl_type::dvec4_type);
         shuffle_64bit_data(tmp, src, false);

         src = src_reg(tmp);
         src.swizzle = BRW_SWZ_COMP_INPUT(nir_intrinsic_component(instr) / 2);

         /* Write to dst reg taking into account original writemask */
         dest = get_nir_dest(instr->dest, BRW_REGISTER_TYPE_DF);
         dest.writemask = brw_writemask_for_size(instr->num_components);
         emit(MOV(dest, src));
      } else {
         /* Make up a type...we have no way of knowing... */
         const glsl_type *const type = glsl_type::ivec(instr->num_components);

         src = src_reg(ATTR, input_array_stride * vertex->u32[0] +
                       instr->const_index[0] + offset_reg->u32[0],
                       type);
         src.swizzle = BRW_SWZ_COMP_INPUT(nir_intrinsic_component(instr));

         dest = get_nir_dest(instr->dest, src.type);
         dest.writemask = brw_writemask_for_size(instr->num_components);
         emit(MOV(dest, src));
      }
      break;
   }

   case nir_intrinsic_load_input:
      unreachable("nir_lower_io should have produced per_vertex intrinsics");

   case nir_intrinsic_emit_vertex_with_counter: {
      this->vertex_count =
         retype(get_nir_src(instr->src[0], 1), BRW_REGISTER_TYPE_UD);
      int stream_id = instr->const_index[0];
      gs_emit_vertex(stream_id);
      break;
   }

   case nir_intrinsic_end_primitive_with_counter:
      this->vertex_count =
         retype(get_nir_src(instr->src[0], 1), BRW_REGISTER_TYPE_UD);
      gs_end_primitive();
      break;

   case nir_intrinsic_set_vertex_count:
      this->vertex_count =
         retype(get_nir_src(instr->src[0], 1), BRW_REGISTER_TYPE_UD);
      break;

   case nir_intrinsic_load_primitive_id:
      assert(gs_prog_data->include_primitive_id);
      dest = get_nir_dest(instr->dest, BRW_REGISTER_TYPE_D);
      emit(MOV(dest, retype(brw_vec4_grf(1, 0), BRW_REGISTER_TYPE_D)));
      break;

   case nir_intrinsic_load_invocation_id: {
      dest = get_nir_dest(instr->dest, BRW_REGISTER_TYPE_D);
      if (gs_prog_data->invocations > 1)
         emit(GS_OPCODE_GET_INSTANCE_ID, dest);
      else
         emit(MOV(dest, brw_imm_ud(0)));
      break;
   }

   default:
      vec4_visitor::nir_emit_intrinsic(instr);
   }
}
}
