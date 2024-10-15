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

/** @file brw_fs_visitor.cpp
 *
 * This file supports generating the FS LIR from the GLSL IR.  The LIR
 * makes it easier to do backend-specific optimizations than doing so
 * in the GLSL IR or in the native code.
 */
#include "brw_fs.h"
#include "compiler/glsl_types.h"

using namespace brw;

/* Sample from the MCS surface attached to this multisample texture. */
fs_reg
fs_visitor::emit_mcs_fetch(const fs_reg &coordinate, unsigned components,
                           const fs_reg &texture)
{
   const fs_reg dest = vgrf(glsl_type::uvec4_type);

   fs_reg srcs[TEX_LOGICAL_NUM_SRCS];
   srcs[TEX_LOGICAL_SRC_COORDINATE] = coordinate;
   srcs[TEX_LOGICAL_SRC_SURFACE] = texture;
   srcs[TEX_LOGICAL_SRC_SAMPLER] = texture;
   srcs[TEX_LOGICAL_SRC_COORD_COMPONENTS] = brw_imm_d(components);
   srcs[TEX_LOGICAL_SRC_GRAD_COMPONENTS] = brw_imm_d(0);

   fs_inst *inst = bld.emit(SHADER_OPCODE_TXF_MCS_LOGICAL, dest, srcs,
                            ARRAY_SIZE(srcs));

   /* We only care about one or two regs of response, but the sampler always
    * writes 4/8.
    */
   inst->size_written = 4 * dest.component_size(inst->exec_size);

   return dest;
}

/**
 * Apply workarounds for Gen6 gather with UINT/SINT
 */
void
fs_visitor::emit_gen6_gather_wa(uint8_t wa, fs_reg dst)
{
   if (!wa)
      return;

   int width = (wa & WA_8BIT) ? 8 : 16;

   for (int i = 0; i < 4; i++) {
      fs_reg dst_f = retype(dst, BRW_REGISTER_TYPE_F);
      /* Convert from UNORM to UINT */
      bld.MUL(dst_f, dst_f, brw_imm_f((1 << width) - 1));
      bld.MOV(dst, dst_f);

      if (wa & WA_SIGN) {
         /* Reinterpret the UINT value as a signed INT value by
          * shifting the sign bit into place, then shifting back
          * preserving sign.
          */
         bld.SHL(dst, dst, brw_imm_d(32 - width));
         bld.ASR(dst, dst, brw_imm_d(32 - width));
      }

      dst = offset(dst, bld, 1);
   }
}

/** Emits a dummy fragment shader consisting of magenta for bringup purposes. */
void
fs_visitor::emit_dummy_fs()
{
   int reg_width = dispatch_width / 8;

   /* Everyone's favorite color. */
   const float color[4] = { 1.0, 0.0, 1.0, 0.0 };
   for (int i = 0; i < 4; i++) {
      bld.MOV(fs_reg(MRF, 2 + i * reg_width, BRW_REGISTER_TYPE_F),
              brw_imm_f(color[i]));
   }

   fs_inst *write;
   write = bld.emit(FS_OPCODE_FB_WRITE);
   write->eot = true;
   write->last_rt = true;
   if (devinfo->gen >= 6) {
      write->base_mrf = 2;
      write->mlen = 4 * reg_width;
   } else {
      write->header_size = 2;
      write->base_mrf = 0;
      write->mlen = 2 + 4 * reg_width;
   }

   /* Tell the SF we don't have any inputs.  Gen4-5 require at least one
    * varying to avoid GPU hangs, so set that.
    */
   struct brw_wm_prog_data *wm_prog_data = brw_wm_prog_data(this->prog_data);
   wm_prog_data->num_varying_inputs = devinfo->gen < 6 ? 1 : 0;
   memset(wm_prog_data->urb_setup, -1,
          sizeof(wm_prog_data->urb_setup[0]) * VARYING_SLOT_MAX);

   /* We don't have any uniforms. */
   stage_prog_data->nr_params = 0;
   stage_prog_data->nr_pull_params = 0;
   stage_prog_data->curb_read_length = 0;
   stage_prog_data->dispatch_grf_start_reg = 2;
   wm_prog_data->dispatch_grf_start_reg_16 = 2;
   wm_prog_data->dispatch_grf_start_reg_32 = 2;
   grf_used = 1; /* Gen4-5 don't allow zero GRF blocks */

   calculate_cfg();
}

/* The register location here is relative to the start of the URB
 * data.  It will get adjusted to be a real location before
 * generate_code() time.
 */
fs_reg
fs_visitor::interp_reg(int location, int channel)
{
   assert(stage == MESA_SHADER_FRAGMENT);
   struct brw_wm_prog_data *prog_data = brw_wm_prog_data(this->prog_data);
   int regnr = prog_data->urb_setup[location] * 4 + channel;
   assert(prog_data->urb_setup[location] != -1);

   return fs_reg(ATTR, regnr, BRW_REGISTER_TYPE_F);
}

/** Emits the interpolation for the varying inputs. */
void
fs_visitor::emit_interpolation_setup_gen4()
{
   struct brw_reg g1_uw = retype(brw_vec1_grf(1, 0), BRW_REGISTER_TYPE_UW);

   fs_builder abld = bld.annotate("compute pixel centers");
   this->pixel_x = vgrf(glsl_type::uint_type);
   this->pixel_y = vgrf(glsl_type::uint_type);
   this->pixel_x.type = BRW_REGISTER_TYPE_UW;
   this->pixel_y.type = BRW_REGISTER_TYPE_UW;
   abld.ADD(this->pixel_x,
            fs_reg(stride(suboffset(g1_uw, 4), 2, 4, 0)),
            fs_reg(brw_imm_v(0x10101010)));
   abld.ADD(this->pixel_y,
            fs_reg(stride(suboffset(g1_uw, 5), 2, 4, 0)),
            fs_reg(brw_imm_v(0x11001100)));

   abld = bld.annotate("compute pixel deltas from v0");

   this->delta_xy[BRW_BARYCENTRIC_PERSPECTIVE_PIXEL] =
      vgrf(glsl_type::vec2_type);
   const fs_reg &delta_xy = this->delta_xy[BRW_BARYCENTRIC_PERSPECTIVE_PIXEL];
   const fs_reg xstart(negate(brw_vec1_grf(1, 0)));
   const fs_reg ystart(negate(brw_vec1_grf(1, 1)));

   if (devinfo->has_pln && dispatch_width == 16) {
      for (unsigned i = 0; i < 2; i++) {
         abld.half(i).ADD(half(offset(delta_xy, abld, i), 0),
                          half(this->pixel_x, i), xstart);
         abld.half(i).ADD(half(offset(delta_xy, abld, i), 1),
                          half(this->pixel_y, i), ystart);
      }
   } else {
      abld.ADD(offset(delta_xy, abld, 0), this->pixel_x, xstart);
      abld.ADD(offset(delta_xy, abld, 1), this->pixel_y, ystart);
   }

   abld = bld.annotate("compute pos.w and 1/pos.w");
   /* Compute wpos.w.  It's always in our setup, since it's needed to
    * interpolate the other attributes.
    */
   this->wpos_w = vgrf(glsl_type::float_type);
   abld.emit(FS_OPCODE_LINTERP, wpos_w, delta_xy,
             component(interp_reg(VARYING_SLOT_POS, 3), 0));
   /* Compute the pixel 1/W value from wpos.w. */
   this->pixel_w = vgrf(glsl_type::float_type);
   abld.emit(SHADER_OPCODE_RCP, this->pixel_w, wpos_w);
}

/** Emits the interpolation for the varying inputs. */
void
fs_visitor::emit_interpolation_setup_gen6()
{
   fs_builder abld = bld.annotate("compute pixel centers");

   this->pixel_x = vgrf(glsl_type::float_type);
   this->pixel_y = vgrf(glsl_type::float_type);

   for (unsigned i = 0; i < DIV_ROUND_UP(dispatch_width, 16); i++) {
      const fs_builder hbld = abld.group(MIN2(16, dispatch_width), i);
      struct brw_reg gi_uw = retype(brw_vec1_grf(1 + i, 0), BRW_REGISTER_TYPE_UW);

      if (devinfo->gen >= 8 || dispatch_width == 8) {
         /* The "Register Region Restrictions" page says for BDW (and newer,
          * presumably):
          *
          *     "When destination spans two registers, the source may be one or
          *      two registers. The destination elements must be evenly split
          *      between the two registers."
          *
          * Thus we can do a single add(16) in SIMD8 or an add(32) in SIMD16
          * to compute our pixel centers.
          */
         const fs_builder dbld =
            abld.exec_all().group(hbld.dispatch_width() * 2, 0);
         fs_reg int_pixel_xy = dbld.vgrf(BRW_REGISTER_TYPE_UW);

         dbld.ADD(int_pixel_xy,
                  fs_reg(stride(suboffset(gi_uw, 4), 1, 4, 0)),
                  fs_reg(brw_imm_v(0x11001010)));

         hbld.emit(FS_OPCODE_PIXEL_X, offset(pixel_x, hbld, i), int_pixel_xy);
         hbld.emit(FS_OPCODE_PIXEL_Y, offset(pixel_y, hbld, i), int_pixel_xy);
      } else {
         /* The "Register Region Restrictions" page says for SNB, IVB, HSW:
          *
          *     "When destination spans two registers, the source MUST span
          *      two registers."
          *
          * Since the GRF source of the ADD will only read a single register,
          * we must do two separate ADDs in SIMD16.
          */
         const fs_reg int_pixel_x = hbld.vgrf(BRW_REGISTER_TYPE_UW);
         const fs_reg int_pixel_y = hbld.vgrf(BRW_REGISTER_TYPE_UW);

         hbld.ADD(int_pixel_x,
                  fs_reg(stride(suboffset(gi_uw, 4), 2, 4, 0)),
                  fs_reg(brw_imm_v(0x10101010)));
         hbld.ADD(int_pixel_y,
                  fs_reg(stride(suboffset(gi_uw, 5), 2, 4, 0)),
                  fs_reg(brw_imm_v(0x11001100)));

         /* As of gen6, we can no longer mix float and int sources.  We have
          * to turn the integer pixel centers into floats for their actual
          * use.
          */
         hbld.MOV(offset(pixel_x, hbld, i), int_pixel_x);
         hbld.MOV(offset(pixel_y, hbld, i), int_pixel_y);
      }
   }

   abld = bld.annotate("compute pos.w");
   this->pixel_w = fetch_payload_reg(abld, payload.source_w_reg);
   this->wpos_w = vgrf(glsl_type::float_type);
   abld.emit(SHADER_OPCODE_RCP, this->wpos_w, this->pixel_w);

   struct brw_wm_prog_data *wm_prog_data = brw_wm_prog_data(prog_data);

   for (int i = 0; i < BRW_BARYCENTRIC_MODE_COUNT; ++i) {
      this->delta_xy[i] = fetch_payload_reg(
         bld, payload.barycentric_coord_reg[i], BRW_REGISTER_TYPE_F, 2);
   }

   uint32_t centroid_modes = wm_prog_data->barycentric_interp_modes &
      (1 << BRW_BARYCENTRIC_PERSPECTIVE_CENTROID |
       1 << BRW_BARYCENTRIC_NONPERSPECTIVE_CENTROID);

   if (devinfo->needs_unlit_centroid_workaround && centroid_modes) {
      /* Get the pixel/sample mask into f0 so that we know which
       * pixels are lit.  Then, for each channel that is unlit,
       * replace the centroid data with non-centroid data.
       */
      for (unsigned i = 0; i < DIV_ROUND_UP(dispatch_width, 16); i++) {
         bld.exec_all().group(1, 0)
            .MOV(retype(brw_flag_reg(0, i), BRW_REGISTER_TYPE_UW),
                 retype(brw_vec1_grf(1 + i, 7), BRW_REGISTER_TYPE_UW));
      }

      for (int i = 0; i < BRW_BARYCENTRIC_MODE_COUNT; ++i) {
         if (!(centroid_modes & (1 << i)))
            continue;

         const fs_reg &pixel_delta_xy = delta_xy[i - 1];

         for (unsigned q = 0; q < dispatch_width / 8; q++) {
            for (unsigned c = 0; c < 2; c++) {
               const unsigned idx = c + (q & 2) + (q & 1) * dispatch_width / 8;
               set_predicate_inv(
                  BRW_PREDICATE_NORMAL, true,
                  bld.half(q).MOV(horiz_offset(delta_xy[i], idx * 8),
                                  horiz_offset(pixel_delta_xy, idx * 8)));
            }
         }
      }
   }
}

static enum brw_conditional_mod
cond_for_alpha_func(GLenum func)
{
   switch(func) {
      case GL_GREATER:
         return BRW_CONDITIONAL_G;
      case GL_GEQUAL:
         return BRW_CONDITIONAL_GE;
      case GL_LESS:
         return BRW_CONDITIONAL_L;
      case GL_LEQUAL:
         return BRW_CONDITIONAL_LE;
      case GL_EQUAL:
         return BRW_CONDITIONAL_EQ;
      case GL_NOTEQUAL:
         return BRW_CONDITIONAL_NEQ;
      default:
         unreachable("Not reached");
   }
}

/**
 * Alpha test support for when we compile it into the shader instead
 * of using the normal fixed-function alpha test.
 */
void
fs_visitor::emit_alpha_test()
{
   assert(stage == MESA_SHADER_FRAGMENT);
   brw_wm_prog_key *key = (brw_wm_prog_key*) this->key;
   const fs_builder abld = bld.annotate("Alpha test");

   fs_inst *cmp;
   if (key->alpha_test_func == GL_ALWAYS)
      return;

   if (key->alpha_test_func == GL_NEVER) {
      /* f0.1 = 0 */
      fs_reg some_reg = fs_reg(retype(brw_vec8_grf(0, 0),
                                      BRW_REGISTER_TYPE_UW));
      cmp = abld.CMP(bld.null_reg_f(), some_reg, some_reg,
                     BRW_CONDITIONAL_NEQ);
   } else {
      /* RT0 alpha */
      fs_reg color = offset(outputs[0], bld, 3);

      /* f0.1 &= func(color, ref) */
      cmp = abld.CMP(bld.null_reg_f(), color, brw_imm_f(key->alpha_test_ref),
                     cond_for_alpha_func(key->alpha_test_func));
   }
   cmp->predicate = BRW_PREDICATE_NORMAL;
   cmp->flag_subreg = 1;
}

fs_inst *
fs_visitor::emit_single_fb_write(const fs_builder &bld,
                                 fs_reg color0, fs_reg color1,
                                 fs_reg src0_alpha, unsigned components)
{
   assert(stage == MESA_SHADER_FRAGMENT);
   struct brw_wm_prog_data *prog_data = brw_wm_prog_data(this->prog_data);

   /* Hand over gl_FragDepth or the payload depth. */
   const fs_reg dst_depth = fetch_payload_reg(bld, payload.dest_depth_reg);
   fs_reg src_depth, src_stencil;

   if (source_depth_to_render_target) {
      if (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH))
         src_depth = frag_depth;
      else
         src_depth = fetch_payload_reg(bld, payload.source_depth_reg);
   }

   if (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_STENCIL))
      src_stencil = frag_stencil;

   const fs_reg sources[] = {
      color0, color1, src0_alpha, src_depth, dst_depth, src_stencil,
      (prog_data->uses_omask ? sample_mask : fs_reg()),
      brw_imm_ud(components)
   };
   assert(ARRAY_SIZE(sources) - 1 == FB_WRITE_LOGICAL_SRC_COMPONENTS);
   fs_inst *write = bld.emit(FS_OPCODE_FB_WRITE_LOGICAL, fs_reg(),
                             sources, ARRAY_SIZE(sources));

   if (prog_data->uses_kill) {
      write->predicate = BRW_PREDICATE_NORMAL;
      write->flag_subreg = 1;
   }

   return write;
}

void
fs_visitor::emit_fb_writes()
{
   assert(stage == MESA_SHADER_FRAGMENT);
   struct brw_wm_prog_data *prog_data = brw_wm_prog_data(this->prog_data);
   brw_wm_prog_key *key = (brw_wm_prog_key*) this->key;

   fs_inst *inst = NULL;

   if (source_depth_to_render_target && devinfo->gen == 6) {
      /* For outputting oDepth on gen6, SIMD8 writes have to be used.  This
       * would require SIMD8 moves of each half to message regs, e.g. by using
       * the SIMD lowering pass.  Unfortunately this is more difficult than it
       * sounds because the SIMD8 single-source message lacks channel selects
       * for the second and third subspans.
       */
      limit_dispatch_width(8, "Depth writes unsupported in SIMD16+ mode.\n");
   }

   if (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_STENCIL)) {
      /* From the 'Render Target Write message' section of the docs:
       * "Output Stencil is not supported with SIMD16 Render Target Write
       * Messages."
       */
      limit_dispatch_width(8, "gl_FragStencilRefARB unsupported "
                           "in SIMD16+ mode.\n");
   }

   for (int target = 0; target < key->nr_color_regions; target++) {
      /* Skip over outputs that weren't written. */
      if (this->outputs[target].file == BAD_FILE)
         continue;

      const fs_builder abld = bld.annotate(
         ralloc_asprintf(this->mem_ctx, "FB write target %d", target));

      fs_reg src0_alpha;
      if (devinfo->gen >= 6 && key->replicate_alpha && target != 0)
         src0_alpha = offset(outputs[0], bld, 3);

      inst = emit_single_fb_write(abld, this->outputs[target],
                                  this->dual_src_output, src0_alpha, 4);
      inst->target = target;
   }

   prog_data->dual_src_blend = (this->dual_src_output.file != BAD_FILE &&
                                this->outputs[0].file != BAD_FILE);
   assert(!prog_data->dual_src_blend || key->nr_color_regions == 1);

   if (inst == NULL) {
      /* Even if there's no color buffers enabled, we still need to send
       * alpha out the pipeline to our null renderbuffer to support
       * alpha-testing, alpha-to-coverage, and so on.
       */
      /* FINISHME: Factor out this frequently recurring pattern into a
       * helper function.
       */
      const fs_reg srcs[] = { reg_undef, reg_undef,
                              reg_undef, offset(this->outputs[0], bld, 3) };
      const fs_reg tmp = bld.vgrf(BRW_REGISTER_TYPE_UD, 4);
      bld.LOAD_PAYLOAD(tmp, srcs, 4, 0);

      inst = emit_single_fb_write(bld, tmp, reg_undef, reg_undef, 4);
      inst->target = 0;
   }

   inst->last_rt = true;
   inst->eot = true;
}

void
fs_visitor::setup_uniform_clipplane_values()
{
   const struct brw_vs_prog_key *key =
      (const struct brw_vs_prog_key *) this->key;

   if (key->nr_userclip_plane_consts == 0)
      return;

   assert(stage_prog_data->nr_params == uniforms);
   brw_stage_prog_data_add_params(stage_prog_data,
                                  key->nr_userclip_plane_consts * 4);

   for (int i = 0; i < key->nr_userclip_plane_consts; i++) {
      this->userplane[i] = fs_reg(UNIFORM, uniforms);
      for (int j = 0; j < 4; ++j) {
         stage_prog_data->param[uniforms + j] =
            BRW_PARAM_BUILTIN_CLIP_PLANE(i, j);
      }
      uniforms += 4;
   }
}

/**
 * Lower legacy fixed-function and gl_ClipVertex clipping to clip distances.
 *
 * This does nothing if the shader uses gl_ClipDistance or user clipping is
 * disabled altogether.
 */
void fs_visitor::compute_clip_distance()
{
   struct brw_vue_prog_data *vue_prog_data = brw_vue_prog_data(prog_data);
   const struct brw_vs_prog_key *key =
      (const struct brw_vs_prog_key *) this->key;

   /* Bail unless some sort of legacy clipping is enabled */
   if (key->nr_userclip_plane_consts == 0)
      return;

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
   if (!(vue_prog_data->vue_map.slots_valid & VARYING_BIT_CLIP_VERTEX))
      clip_vertex = VARYING_SLOT_POS;

   /* If the clip vertex isn't written, skip this.  Typically this means
    * the GS will set up clipping. */
   if (outputs[clip_vertex].file == BAD_FILE)
      return;

   setup_uniform_clipplane_values();

   const fs_builder abld = bld.annotate("user clip distances");

   this->outputs[VARYING_SLOT_CLIP_DIST0] = vgrf(glsl_type::vec4_type);
   this->outputs[VARYING_SLOT_CLIP_DIST1] = vgrf(glsl_type::vec4_type);

   for (int i = 0; i < key->nr_userclip_plane_consts; i++) {
      fs_reg u = userplane[i];
      const fs_reg output = offset(outputs[VARYING_SLOT_CLIP_DIST0 + i / 4],
                                   bld, i & 3);

      abld.MUL(output, outputs[clip_vertex], u);
      for (int j = 1; j < 4; j++) {
         u.nr = userplane[i].nr + j;
         abld.MAD(output, output, offset(outputs[clip_vertex], bld, j), u);
      }
   }
}

void
fs_visitor::emit_urb_writes(const fs_reg &gs_vertex_count)
{
   int slot, urb_offset, length;
   int starting_urb_offset = 0;
   const struct brw_vue_prog_data *vue_prog_data =
      brw_vue_prog_data(this->prog_data);
   const struct brw_vs_prog_key *vs_key =
      (const struct brw_vs_prog_key *) this->key;
   const GLbitfield64 psiz_mask =
      VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT | VARYING_BIT_PSIZ;
   const struct brw_vue_map *vue_map = &vue_prog_data->vue_map;
   bool flush;
   fs_reg sources[8];
   fs_reg urb_handle;

   if (stage == MESA_SHADER_TESS_EVAL)
      urb_handle = fs_reg(retype(brw_vec8_grf(4, 0), BRW_REGISTER_TYPE_UD));
   else
      urb_handle = fs_reg(retype(brw_vec8_grf(1, 0), BRW_REGISTER_TYPE_UD));

   opcode opcode = SHADER_OPCODE_URB_WRITE_SIMD8;
   int header_size = 1;
   fs_reg per_slot_offsets;

   if (stage == MESA_SHADER_GEOMETRY) {
      const struct brw_gs_prog_data *gs_prog_data =
         brw_gs_prog_data(this->prog_data);

      /* We need to increment the Global Offset to skip over the control data
       * header and the extra "Vertex Count" field (1 HWord) at the beginning
       * of the VUE.  We're counting in OWords, so the units are doubled.
       */
      starting_urb_offset = 2 * gs_prog_data->control_data_header_size_hwords;
      if (gs_prog_data->static_vertex_count == -1)
         starting_urb_offset += 2;

      /* We also need to use per-slot offsets.  The per-slot offset is the
       * Vertex Count.  SIMD8 mode processes 8 different primitives at a
       * time; each may output a different number of vertices.
       */
      opcode = SHADER_OPCODE_URB_WRITE_SIMD8_PER_SLOT;
      header_size++;

      /* The URB offset is in 128-bit units, so we need to multiply by 2 */
      const int output_vertex_size_owords =
         gs_prog_data->output_vertex_size_hwords * 2;

      if (gs_vertex_count.file == IMM) {
         per_slot_offsets = brw_imm_ud(output_vertex_size_owords *
                                       gs_vertex_count.ud);
      } else {
         per_slot_offsets = vgrf(glsl_type::uint_type);
         bld.MUL(per_slot_offsets, gs_vertex_count,
                 brw_imm_ud(output_vertex_size_owords));
      }
   }

   length = 0;
   urb_offset = starting_urb_offset;
   flush = false;

   /* SSO shaders can have VUE slots allocated which are never actually
    * written to, so ignore them when looking for the last (written) slot.
    */
   int last_slot = vue_map->num_slots - 1;
   while (last_slot > 0 &&
          (vue_map->slot_to_varying[last_slot] == BRW_VARYING_SLOT_PAD ||
           outputs[vue_map->slot_to_varying[last_slot]].file == BAD_FILE)) {
      last_slot--;
   }

   bool urb_written = false;
   for (slot = 0; slot < vue_map->num_slots; slot++) {
      int varying = vue_map->slot_to_varying[slot];
      switch (varying) {
      case VARYING_SLOT_PSIZ: {
         /* The point size varying slot is the vue header and is always in the
          * vue map.  But often none of the special varyings that live there
          * are written and in that case we can skip writing to the vue
          * header, provided the corresponding state properly clamps the
          * values further down the pipeline. */
         if ((vue_map->slots_valid & psiz_mask) == 0) {
            assert(length == 0);
            urb_offset++;
            break;
         }

         fs_reg zero(VGRF, alloc.allocate(1), BRW_REGISTER_TYPE_UD);
         bld.MOV(zero, brw_imm_ud(0u));

         sources[length++] = zero;
         if (vue_map->slots_valid & VARYING_BIT_LAYER)
            sources[length++] = this->outputs[VARYING_SLOT_LAYER];
         else
            sources[length++] = zero;

         if (vue_map->slots_valid & VARYING_BIT_VIEWPORT)
            sources[length++] = this->outputs[VARYING_SLOT_VIEWPORT];
         else
            sources[length++] = zero;

         if (vue_map->slots_valid & VARYING_BIT_PSIZ)
            sources[length++] = this->outputs[VARYING_SLOT_PSIZ];
         else
            sources[length++] = zero;
         break;
      }
      case BRW_VARYING_SLOT_NDC:
      case VARYING_SLOT_EDGE:
         unreachable("unexpected scalar vs output");
         break;

      default:
         /* gl_Position is always in the vue map, but isn't always written by
          * the shader.  Other varyings (clip distances) get added to the vue
          * map but don't always get written.  In those cases, the
          * corresponding this->output[] slot will be invalid we and can skip
          * the urb write for the varying.  If we've already queued up a vue
          * slot for writing we flush a mlen 5 urb write, otherwise we just
          * advance the urb_offset.
          */
         if (varying == BRW_VARYING_SLOT_PAD ||
             this->outputs[varying].file == BAD_FILE) {
            if (length > 0)
               flush = true;
            else
               urb_offset++;
            break;
         }

         if (stage == MESA_SHADER_VERTEX && vs_key->clamp_vertex_color &&
             (varying == VARYING_SLOT_COL0 ||
              varying == VARYING_SLOT_COL1 ||
              varying == VARYING_SLOT_BFC0 ||
              varying == VARYING_SLOT_BFC1)) {
            /* We need to clamp these guys, so do a saturating MOV into a
             * temp register and use that for the payload.
             */
            for (int i = 0; i < 4; i++) {
               fs_reg reg = fs_reg(VGRF, alloc.allocate(1), outputs[varying].type);
               fs_reg src = offset(this->outputs[varying], bld, i);
               set_saturate(true, bld.MOV(reg, src));
               sources[length++] = reg;
            }
         } else {
            for (unsigned i = 0; i < 4; i++)
               sources[length++] = offset(this->outputs[varying], bld, i);
         }
         break;
      }

      const fs_builder abld = bld.annotate("URB write");

      /* If we've queued up 8 registers of payload (2 VUE slots), if this is
       * the last slot or if we need to flush (see BAD_FILE varying case
       * above), emit a URB write send now to flush out the data.
       */
      if (length == 8 || (length > 0 && slot == last_slot))
         flush = true;
      if (flush) {
         fs_reg *payload_sources =
            ralloc_array(mem_ctx, fs_reg, length + header_size);
         fs_reg payload = fs_reg(VGRF, alloc.allocate(length + header_size),
                                 BRW_REGISTER_TYPE_F);
         payload_sources[0] = urb_handle;

         if (opcode == SHADER_OPCODE_URB_WRITE_SIMD8_PER_SLOT)
            payload_sources[1] = per_slot_offsets;

         memcpy(&payload_sources[header_size], sources,
                length * sizeof sources[0]);

         abld.LOAD_PAYLOAD(payload, payload_sources, length + header_size,
                           header_size);

         fs_inst *inst = abld.emit(opcode, reg_undef, payload);
         inst->eot = slot == last_slot && stage != MESA_SHADER_GEOMETRY;
         inst->mlen = length + header_size;
         inst->offset = urb_offset;
         urb_offset = starting_urb_offset + slot + 1;
         length = 0;
         flush = false;
         urb_written = true;
      }
   }

   /* If we don't have any valid slots to write, just do a minimal urb write
    * send to terminate the shader.  This includes 1 slot of undefined data,
    * because it's invalid to write 0 data:
    *
    * From the Broadwell PRM, Volume 7: 3D Media GPGPU, Shared Functions -
    * Unified Return Buffer (URB) > URB_SIMD8_Write and URB_SIMD8_Read >
    * Write Data Payload:
    *
    *    "The write data payload can be between 1 and 8 message phases long."
    */
   if (!urb_written) {
      /* For GS, just turn EmitVertex() into a no-op.  We don't want it to
       * end the thread, and emit_gs_thread_end() already emits a SEND with
       * EOT at the end of the program for us.
       */
      if (stage == MESA_SHADER_GEOMETRY)
         return;

      fs_reg payload = fs_reg(VGRF, alloc.allocate(2), BRW_REGISTER_TYPE_UD);
      bld.exec_all().MOV(payload, urb_handle);

      fs_inst *inst = bld.emit(SHADER_OPCODE_URB_WRITE_SIMD8, reg_undef, payload);
      inst->eot = true;
      inst->mlen = 2;
      inst->offset = 1;
      return;
   }
}

void
fs_visitor::emit_cs_terminate()
{
   assert(devinfo->gen >= 7);

   /* We are getting the thread ID from the compute shader header */
   assert(stage == MESA_SHADER_COMPUTE);

   /* We can't directly send from g0, since sends with EOT have to use
    * g112-127. So, copy it to a virtual register, The register allocator will
    * make sure it uses the appropriate register range.
    */
   struct brw_reg g0 = retype(brw_vec8_grf(0, 0), BRW_REGISTER_TYPE_UD);
   fs_reg payload = fs_reg(VGRF, alloc.allocate(1), BRW_REGISTER_TYPE_UD);
   bld.group(8, 0).exec_all().MOV(payload, g0);

   /* Send a message to the thread spawner to terminate the thread. */
   fs_inst *inst = bld.exec_all()
                      .emit(CS_OPCODE_CS_TERMINATE, reg_undef, payload);
   inst->eot = true;
}

void
fs_visitor::emit_barrier()
{
   uint32_t barrier_id_mask;
   switch (devinfo->gen) {
   case 7:
   case 8:
      barrier_id_mask = 0x0f000000u; break;
   case 9:
   case 10:
      barrier_id_mask = 0x8f000000u; break;
   case 11:
      barrier_id_mask = 0x7f000000u; break;
   default:
      unreachable("barrier is only available on gen >= 7");
   }

   /* We are getting the barrier ID from the compute shader header */
   assert(stage == MESA_SHADER_COMPUTE);

   fs_reg payload = fs_reg(VGRF, alloc.allocate(1), BRW_REGISTER_TYPE_UD);

   /* Clear the message payload */
   bld.exec_all().group(8, 0).MOV(payload, brw_imm_ud(0u));

   /* Copy the barrier id from r0.2 to the message payload reg.2 */
   fs_reg r0_2 = fs_reg(retype(brw_vec1_grf(0, 2), BRW_REGISTER_TYPE_UD));
   bld.exec_all().group(1, 0).AND(component(payload, 2), r0_2,
                                  brw_imm_ud(barrier_id_mask));

   /* Emit a gateway "barrier" message using the payload we set up, followed
    * by a wait instruction.
    */
   bld.exec_all().emit(SHADER_OPCODE_BARRIER, reg_undef, payload);
}

fs_visitor::fs_visitor(const struct brw_compiler *compiler, void *log_data,
                       void *mem_ctx,
                       const void *key,
                       struct brw_stage_prog_data *prog_data,
                       struct gl_program *prog,
                       const nir_shader *shader,
                       unsigned dispatch_width,
                       int shader_time_index,
                       const struct brw_vue_map *input_vue_map)
   : backend_shader(compiler, log_data, mem_ctx, shader, prog_data),
     key(key), gs_compile(NULL), prog_data(prog_data), prog(prog),
     input_vue_map(input_vue_map),
     dispatch_width(dispatch_width),
     shader_time_index(shader_time_index),
     bld(fs_builder(this, dispatch_width).at_end())
{
   init();
}

fs_visitor::fs_visitor(const struct brw_compiler *compiler, void *log_data,
                       void *mem_ctx,
                       struct brw_gs_compile *c,
                       struct brw_gs_prog_data *prog_data,
                       const nir_shader *shader,
                       int shader_time_index)
   : backend_shader(compiler, log_data, mem_ctx, shader,
                    &prog_data->base.base),
     key(&c->key), gs_compile(c),
     prog_data(&prog_data->base.base), prog(NULL),
     dispatch_width(8),
     shader_time_index(shader_time_index),
     bld(fs_builder(this, dispatch_width).at_end())
{
   init();
}


void
fs_visitor::init()
{
   switch (stage) {
   case MESA_SHADER_FRAGMENT:
      key_tex = &((const brw_wm_prog_key *) key)->tex;
      break;
   case MESA_SHADER_VERTEX:
      key_tex = &((const brw_vs_prog_key *) key)->tex;
      break;
   case MESA_SHADER_TESS_CTRL:
      key_tex = &((const brw_tcs_prog_key *) key)->tex;
      break;
   case MESA_SHADER_TESS_EVAL:
      key_tex = &((const brw_tes_prog_key *) key)->tex;
      break;
   case MESA_SHADER_GEOMETRY:
      key_tex = &((const brw_gs_prog_key *) key)->tex;
      break;
   case MESA_SHADER_COMPUTE:
      key_tex = &((const brw_cs_prog_key*) key)->tex;
      break;
   default:
      unreachable("unhandled shader stage");
   }

   this->max_dispatch_width = 32;
   this->prog_data = this->stage_prog_data;

   this->failed = false;

   this->nir_locals = NULL;
   this->nir_ssa_values = NULL;

   memset(&this->payload, 0, sizeof(this->payload));
   this->source_depth_to_render_target = false;
   this->runtime_check_aads_emit = false;
   this->first_non_payload_grf = 0;
   this->max_grf = devinfo->gen >= 7 ? GEN7_MRF_HACK_START : BRW_MAX_GRF;

   this->virtual_grf_start = NULL;
   this->virtual_grf_end = NULL;
   this->live_intervals = NULL;
   this->regs_live_at_ip = NULL;

   this->uniforms = 0;
   this->last_scratch = 0;
   this->pull_constant_loc = NULL;
   this->push_constant_loc = NULL;

   this->promoted_constants = 0,

   this->grf_used = 0;
   this->spilled_any_registers = false;
}

fs_visitor::~fs_visitor()
{
}
