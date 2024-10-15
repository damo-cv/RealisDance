/*
 * Copyright (c) 2016 Intel Corporation
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

#include "brw_nir.h"
#include "compiler/nir/nir_builder.h"

struct lower_intrinsics_state {
   nir_shader *nir;
   unsigned dispatch_width;
   nir_function_impl *impl;
   bool progress;
   nir_builder builder;
   unsigned local_workgroup_size;
};

static bool
lower_cs_intrinsics_convert_block(struct lower_intrinsics_state *state,
                                  nir_block *block)
{
   bool progress = false;
   nir_builder *b = &state->builder;
   nir_shader *nir = state->nir;

   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);

      b->cursor = nir_after_instr(&intrinsic->instr);

      nir_ssa_def *sysval;
      switch (intrinsic->intrinsic) {
      case nir_intrinsic_load_local_invocation_index: {
         /* We construct the local invocation index from:
          *
          *    gl_LocalInvocationIndex =
          *       cs_thread_local_id + subgroup_invocation;
          */
         nir_ssa_def *subgroup_id;
         if (state->local_workgroup_size <= state->dispatch_width)
            subgroup_id = nir_imm_int(b, 0);
         else
            subgroup_id = nir_load_subgroup_id(b);

         nir_ssa_def *thread_local_id =
            nir_imul(b, subgroup_id, nir_imm_int(b, state->dispatch_width));
         nir_ssa_def *channel = nir_load_subgroup_invocation(b);
         sysval = nir_iadd(b, channel, thread_local_id);
         break;
      }

      case nir_intrinsic_load_local_invocation_id: {
         /* We lower gl_LocalInvocationID from gl_LocalInvocationIndex based
          * on this formula:
          *
          *    gl_LocalInvocationID.x =
          *       gl_LocalInvocationIndex % gl_WorkGroupSize.x;
          *    gl_LocalInvocationID.y =
          *       (gl_LocalInvocationIndex / gl_WorkGroupSize.x) %
          *       gl_WorkGroupSize.y;
          *    gl_LocalInvocationID.z =
          *       (gl_LocalInvocationIndex /
          *        (gl_WorkGroupSize.x * gl_WorkGroupSize.y)) %
          *       gl_WorkGroupSize.z;
          */
         unsigned *size = nir->info.cs.local_size;

         nir_ssa_def *local_index = nir_load_local_invocation_index(b);

         nir_const_value uvec3;
         memset(&uvec3, 0, sizeof(uvec3));
         uvec3.u32[0] = 1;
         uvec3.u32[1] = size[0];
         uvec3.u32[2] = size[0] * size[1];
         nir_ssa_def *div_val = nir_build_imm(b, 3, 32, uvec3);
         uvec3.u32[0] = size[0];
         uvec3.u32[1] = size[1];
         uvec3.u32[2] = size[2];
         nir_ssa_def *mod_val = nir_build_imm(b, 3, 32, uvec3);

         sysval = nir_umod(b, nir_udiv(b, local_index, div_val), mod_val);
         break;
      }

      case nir_intrinsic_load_subgroup_id:
         if (state->local_workgroup_size > 8)
            continue;

         /* For small workgroup sizes, we know subgroup_id will be zero */
         sysval = nir_imm_int(b, 0);
         break;

      case nir_intrinsic_load_num_subgroups: {
         unsigned local_workgroup_size =
            nir->info.cs.local_size[0] * nir->info.cs.local_size[1] *
            nir->info.cs.local_size[2];
         unsigned num_subgroups =
            DIV_ROUND_UP(local_workgroup_size, state->dispatch_width);
         sysval = nir_imm_int(b, num_subgroups);
         break;
      }

      default:
         continue;
      }

      nir_ssa_def_rewrite_uses(&intrinsic->dest.ssa, nir_src_for_ssa(sysval));
      nir_instr_remove(&intrinsic->instr);

      state->progress = true;
   }

   return progress;
}

static void
lower_cs_intrinsics_convert_impl(struct lower_intrinsics_state *state)
{
   nir_builder_init(&state->builder, state->impl);

   nir_foreach_block(block, state->impl) {
      lower_cs_intrinsics_convert_block(state, block);
   }

   nir_metadata_preserve(state->impl,
                         nir_metadata_block_index | nir_metadata_dominance);
}

bool
brw_nir_lower_cs_intrinsics(nir_shader *nir,
                            unsigned dispatch_width)
{
   assert(nir->info.stage == MESA_SHADER_COMPUTE);

   bool progress = false;
   struct lower_intrinsics_state state;
   memset(&state, 0, sizeof(state));
   state.nir = nir;
   state.dispatch_width = dispatch_width;
   state.local_workgroup_size = nir->info.cs.local_size[0] *
                                nir->info.cs.local_size[1] *
                                nir->info.cs.local_size[2];

   do {
      state.progress = false;
      nir_foreach_function(function, nir) {
         if (function->impl) {
            state.impl = function->impl;
            lower_cs_intrinsics_convert_impl(&state);
         }
      }
      progress |= state.progress;
   } while (state.progress);

   return progress;
}
