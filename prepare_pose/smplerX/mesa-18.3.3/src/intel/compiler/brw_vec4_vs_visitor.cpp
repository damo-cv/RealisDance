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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include "brw_vec4_vs.h"
#include "common/gen_debug.h"

namespace brw {

void
vec4_vs_visitor::emit_prolog()
{
}


void
vec4_vs_visitor::emit_urb_write_header(int mrf)
{
   /* No need to do anything for VS; an implied write to this MRF will be
    * performed by VS_OPCODE_URB_WRITE.
    */
   (void) mrf;
}


vec4_instruction *
vec4_vs_visitor::emit_urb_write_opcode(bool complete)
{
   /* For VS, the URB writes end the thread. */
   if (complete) {
      if (INTEL_DEBUG & DEBUG_SHADER_TIME)
         emit_shader_time_end();
   }

   vec4_instruction *inst = emit(VS_OPCODE_URB_WRITE);
   inst->urb_write_flags = complete ?
      BRW_URB_WRITE_EOT_COMPLETE : BRW_URB_WRITE_NO_FLAGS;

   return inst;
}


void
vec4_vs_visitor::emit_urb_slot(dst_reg reg, int varying)
{
   reg.type = BRW_REGISTER_TYPE_F;
   output_reg[varying][0].type = reg.type;

   switch (varying) {
   case VARYING_SLOT_COL0:
   case VARYING_SLOT_COL1:
   case VARYING_SLOT_BFC0:
   case VARYING_SLOT_BFC1: {
      /* These built-in varyings are only supported in compatibility mode,
       * and we only support GS in core profile.  So, this must be a vertex
       * shader.
       */
      vec4_instruction *inst = emit_generic_urb_slot(reg, varying, 0);
      if (inst && key->clamp_vertex_color)
         inst->saturate = true;
      break;
   }
   default:
      return vec4_visitor::emit_urb_slot(reg, varying);
   }
}


void
vec4_vs_visitor::emit_clip_distances(dst_reg reg, int offset)
{
   /* From the GLSL 1.30 spec, section 7.1 (Vertex Shader Special Variables):
    *
    *     "If a linked set of shaders forming the vertex stage contains no
    *     static write to gl_ClipVertex or gl_ClipDistance, but the
    *     application has requested clipping against user clip planes through
    *     the API, then the coordinate written to gl_Position is used for
    *     comparison against the user clip planes."
    *
    * This function is only called if the shader didn't write to
    * gl_ClipDistance.  Accordingly, we use gl_ClipVertex to perform clipping
    * if the user wrote to it; otherwise we use gl_Position.
    */
   gl_varying_slot clip_vertex = VARYING_SLOT_CLIP_VERTEX;
   if (!(prog_data->vue_map.slots_valid & VARYING_BIT_CLIP_VERTEX)) {
      clip_vertex = VARYING_SLOT_POS;
   }

   for (int i = 0; i + offset < key->nr_userclip_plane_consts && i < 4;
        ++i) {
      reg.writemask = 1 << i;
      emit(DP4(reg,
               src_reg(output_reg[clip_vertex][0]),
               src_reg(this->userplane[i + offset])));
   }
}


void
vec4_vs_visitor::setup_uniform_clipplane_values()
{
   if (key->nr_userclip_plane_consts == 0)
      return;

   assert(stage_prog_data->nr_params == (unsigned)this->uniforms * 4);
   brw_stage_prog_data_add_params(stage_prog_data,
                                  key->nr_userclip_plane_consts * 4);

   for (int i = 0; i < key->nr_userclip_plane_consts; ++i) {
      this->userplane[i] = dst_reg(UNIFORM, this->uniforms);
      this->userplane[i].type = BRW_REGISTER_TYPE_F;
      for (int j = 0; j < 4; ++j) {
         stage_prog_data->param[this->uniforms * 4 + j] =
            BRW_PARAM_BUILTIN_CLIP_PLANE(i, j);
      }
      ++this->uniforms;
   }
}


void
vec4_vs_visitor::emit_thread_end()
{
   setup_uniform_clipplane_values();

   /* Lower legacy ff and ClipVertex clipping to clip distances */
   if (key->nr_userclip_plane_consts > 0) {
      current_annotation = "user clip distances";

      output_reg[VARYING_SLOT_CLIP_DIST0][0] =
         dst_reg(this, glsl_type::vec4_type);
      output_reg[VARYING_SLOT_CLIP_DIST1][0] =
         dst_reg(this, glsl_type::vec4_type);
      output_num_components[VARYING_SLOT_CLIP_DIST0][0] = 4;
      output_num_components[VARYING_SLOT_CLIP_DIST1][0] = 4;

      emit_clip_distances(output_reg[VARYING_SLOT_CLIP_DIST0][0], 0);
      emit_clip_distances(output_reg[VARYING_SLOT_CLIP_DIST1][0], 4);
   }

   /* For VS, we always end the thread by emitting a single vertex.
    * emit_urb_write_opcode() will take care of setting the eot flag on the
    * SEND instruction.
    */
   emit_vertex();
}


vec4_vs_visitor::vec4_vs_visitor(const struct brw_compiler *compiler,
                                 void *log_data,
                                 const struct brw_vs_prog_key *key,
                                 struct brw_vs_prog_data *vs_prog_data,
                                 const nir_shader *shader,
                                 void *mem_ctx,
                                 int shader_time_index)
   : vec4_visitor(compiler, log_data, &key->tex, &vs_prog_data->base, shader,
                  mem_ctx, false /* no_spills */, shader_time_index),
     key(key),
     vs_prog_data(vs_prog_data)
{
}


} /* namespace brw */
