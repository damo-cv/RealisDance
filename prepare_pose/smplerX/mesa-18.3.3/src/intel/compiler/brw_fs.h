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
 *
 * Authors:
 *    Eric Anholt <eric@anholt.net>
 *
 */

#ifndef BRW_FS_H
#define BRW_FS_H

#include "brw_shader.h"
#include "brw_ir_fs.h"
#include "brw_fs_builder.h"
#include "compiler/nir/nir.h"

struct bblock_t;
namespace {
   struct acp_entry;
}

namespace brw {
   class fs_live_variables;
}

struct brw_gs_compile;

static inline fs_reg
offset(const fs_reg &reg, const brw::fs_builder &bld, unsigned delta)
{
   return offset(reg, bld.dispatch_width(), delta);
}

#define UBO_START ((1 << 16) - 4)

/**
 * The fragment shader front-end.
 *
 * Translates either GLSL IR or Mesa IR (for ARB_fragment_program) into FS IR.
 */
class fs_visitor : public backend_shader
{
public:
   fs_visitor(const struct brw_compiler *compiler, void *log_data,
              void *mem_ctx,
              const void *key,
              struct brw_stage_prog_data *prog_data,
              struct gl_program *prog,
              const nir_shader *shader,
              unsigned dispatch_width,
              int shader_time_index,
              const struct brw_vue_map *input_vue_map = NULL);
   fs_visitor(const struct brw_compiler *compiler, void *log_data,
              void *mem_ctx,
              struct brw_gs_compile *gs_compile,
              struct brw_gs_prog_data *prog_data,
              const nir_shader *shader,
              int shader_time_index);
   void init();
   ~fs_visitor();

   fs_reg vgrf(const glsl_type *const type);
   void import_uniforms(fs_visitor *v);
   void setup_uniform_clipplane_values();
   void compute_clip_distance();

   void VARYING_PULL_CONSTANT_LOAD(const brw::fs_builder &bld,
                                   const fs_reg &dst,
                                   const fs_reg &surf_index,
                                   const fs_reg &varying_offset,
                                   uint32_t const_offset);
   void DEP_RESOLVE_MOV(const brw::fs_builder &bld, int grf);

   bool run_fs(bool allow_spilling, bool do_rep_send);
   bool run_vs();
   bool run_tcs_single_patch();
   bool run_tes();
   bool run_gs();
   bool run_cs(unsigned min_dispatch_width);
   void optimize();
   void allocate_registers(unsigned min_dispatch_width, bool allow_spilling);
   void setup_fs_payload_gen4();
   void setup_fs_payload_gen6();
   void setup_vs_payload();
   void setup_gs_payload();
   void setup_cs_payload();
   void fixup_3src_null_dest();
   void assign_curb_setup();
   void calculate_urb_setup();
   void assign_urb_setup();
   void convert_attr_sources_to_hw_regs(fs_inst *inst);
   void assign_vs_urb_setup();
   void assign_tcs_single_patch_urb_setup();
   void assign_tes_urb_setup();
   void assign_gs_urb_setup();
   bool assign_regs(bool allow_spilling, bool spill_all);
   void assign_regs_trivial();
   void calculate_payload_ranges(int payload_node_count,
                                 int *payload_last_use_ip);
   void setup_payload_interference(struct ra_graph *g, int payload_reg_count,
                                   int first_payload_node);
   int choose_spill_reg(struct ra_graph *g);
   void spill_reg(int spill_reg);
   void split_virtual_grfs();
   bool compact_virtual_grfs();
   void assign_constant_locations();
   bool get_pull_locs(const fs_reg &src, unsigned *out_surf_index,
                      unsigned *out_pull_index);
   void lower_constant_loads();
   void invalidate_live_intervals();
   void calculate_live_intervals();
   void calculate_register_pressure();
   void validate();
   bool opt_algebraic();
   bool opt_redundant_discard_jumps();
   bool opt_cse();
   bool opt_cse_local(bblock_t *block);
   bool opt_copy_propagation();
   bool try_copy_propagate(fs_inst *inst, int arg, acp_entry *entry);
   bool try_constant_propagate(fs_inst *inst, acp_entry *entry);
   bool opt_copy_propagation_local(void *mem_ctx, bblock_t *block,
                                   exec_list *acp);
   bool opt_drop_redundant_mov_to_flags();
   bool opt_register_renaming();
   bool opt_bank_conflicts();
   unsigned bank_conflict_cycles(const fs_inst *inst) const;
   bool register_coalesce();
   bool compute_to_mrf();
   bool eliminate_find_live_channel();
   bool dead_code_eliminate();
   bool remove_duplicate_mrf_writes();
   bool remove_extra_rounding_modes();

   bool opt_sampler_eot();
   bool virtual_grf_interferes(int a, int b);
   void schedule_instructions(instruction_scheduler_mode mode);
   void insert_gen4_send_dependency_workarounds();
   void insert_gen4_pre_send_dependency_workarounds(bblock_t *block,
                                                    fs_inst *inst);
   void insert_gen4_post_send_dependency_workarounds(bblock_t *block,
                                                     fs_inst *inst);
   void vfail(const char *msg, va_list args);
   void fail(const char *msg, ...);
   void limit_dispatch_width(unsigned n, const char *msg);
   void lower_uniform_pull_constant_loads();
   bool lower_load_payload();
   bool lower_pack();
   bool lower_conversions();
   bool lower_logical_sends();
   bool lower_integer_multiplication();
   bool lower_minmax();
   bool lower_simd_width();
   bool opt_combine_constants();

   void emit_dummy_fs();
   void emit_repclear_shader();
   void emit_fragcoord_interpolation(fs_reg wpos);
   fs_reg *emit_frontfacing_interpolation();
   fs_reg *emit_samplepos_setup();
   fs_reg *emit_sampleid_setup();
   fs_reg *emit_samplemaskin_setup();
   void emit_interpolation_setup_gen4();
   void emit_interpolation_setup_gen6();
   void compute_sample_position(fs_reg dst, fs_reg int_sample_pos);
   fs_reg emit_mcs_fetch(const fs_reg &coordinate, unsigned components,
                         const fs_reg &sampler);
   void emit_gen6_gather_wa(uint8_t wa, fs_reg dst);
   fs_reg resolve_source_modifiers(const fs_reg &src);
   void emit_discard_jump();
   bool opt_peephole_sel();
   bool opt_peephole_csel();
   bool opt_peephole_predicated_break();
   bool opt_saturate_propagation();
   bool opt_cmod_propagation();
   bool opt_zero_samples();

   void emit_nir_code();
   void nir_setup_outputs();
   void nir_setup_uniforms();
   void nir_emit_system_values();
   void nir_emit_impl(nir_function_impl *impl);
   void nir_emit_cf_list(exec_list *list);
   void nir_emit_if(nir_if *if_stmt);
   void nir_emit_loop(nir_loop *loop);
   void nir_emit_block(nir_block *block);
   void nir_emit_instr(nir_instr *instr);
   void nir_emit_alu(const brw::fs_builder &bld, nir_alu_instr *instr);
   void nir_emit_load_const(const brw::fs_builder &bld,
                            nir_load_const_instr *instr);
   void nir_emit_vs_intrinsic(const brw::fs_builder &bld,
                              nir_intrinsic_instr *instr);
   void nir_emit_tcs_intrinsic(const brw::fs_builder &bld,
                               nir_intrinsic_instr *instr);
   void nir_emit_gs_intrinsic(const brw::fs_builder &bld,
                              nir_intrinsic_instr *instr);
   void nir_emit_fs_intrinsic(const brw::fs_builder &bld,
                              nir_intrinsic_instr *instr);
   void nir_emit_cs_intrinsic(const brw::fs_builder &bld,
                              nir_intrinsic_instr *instr);
   fs_reg get_nir_image_intrinsic_image(const brw::fs_builder &bld,
                                        nir_intrinsic_instr *instr);
   void nir_emit_intrinsic(const brw::fs_builder &bld,
                           nir_intrinsic_instr *instr);
   void nir_emit_tes_intrinsic(const brw::fs_builder &bld,
                               nir_intrinsic_instr *instr);
   void nir_emit_ssbo_atomic(const brw::fs_builder &bld,
                             int op, nir_intrinsic_instr *instr);
   void nir_emit_ssbo_atomic_float(const brw::fs_builder &bld,
                                   int op, nir_intrinsic_instr *instr);
   void nir_emit_shared_atomic(const brw::fs_builder &bld,
                               int op, nir_intrinsic_instr *instr);
   void nir_emit_shared_atomic_float(const brw::fs_builder &bld,
                                     int op, nir_intrinsic_instr *instr);
   void nir_emit_texture(const brw::fs_builder &bld,
                         nir_tex_instr *instr);
   void nir_emit_jump(const brw::fs_builder &bld,
                      nir_jump_instr *instr);
   fs_reg get_nir_src(const nir_src &src);
   fs_reg get_nir_src_imm(const nir_src &src);
   fs_reg get_nir_dest(const nir_dest &dest);
   fs_reg get_indirect_offset(nir_intrinsic_instr *instr);
   void emit_percomp(const brw::fs_builder &bld, const fs_inst &inst,
                     unsigned wr_mask);

   bool optimize_extract_to_float(nir_alu_instr *instr,
                                  const fs_reg &result);
   bool optimize_frontfacing_ternary(nir_alu_instr *instr,
                                     const fs_reg &result);

   void emit_alpha_test();
   fs_inst *emit_single_fb_write(const brw::fs_builder &bld,
                                 fs_reg color1, fs_reg color2,
                                 fs_reg src0_alpha, unsigned components);
   void emit_fb_writes();
   fs_inst *emit_non_coherent_fb_read(const brw::fs_builder &bld,
                                      const fs_reg &dst, unsigned target);
   void emit_urb_writes(const fs_reg &gs_vertex_count = fs_reg());
   void set_gs_stream_control_data_bits(const fs_reg &vertex_count,
                                        unsigned stream_id);
   void emit_gs_control_data_bits(const fs_reg &vertex_count);
   void emit_gs_end_primitive(const nir_src &vertex_count_nir_src);
   void emit_gs_vertex(const nir_src &vertex_count_nir_src,
                       unsigned stream_id);
   void emit_gs_thread_end();
   void emit_gs_input_load(const fs_reg &dst, const nir_src &vertex_src,
                           unsigned base_offset, const nir_src &offset_src,
                           unsigned num_components, unsigned first_component);
   void emit_cs_terminate();
   fs_reg *emit_cs_work_group_id_setup();

   void emit_barrier();

   void emit_shader_time_begin();
   void emit_shader_time_end();
   void SHADER_TIME_ADD(const brw::fs_builder &bld,
                        int shader_time_subindex,
                        fs_reg value);

   fs_reg get_timestamp(const brw::fs_builder &bld);

   fs_reg interp_reg(int location, int channel);

   int implied_mrf_writes(fs_inst *inst) const;

   virtual void dump_instructions();
   virtual void dump_instructions(const char *name);
   void dump_instruction(backend_instruction *inst);
   void dump_instruction(backend_instruction *inst, FILE *file);

   const void *const key;
   const struct brw_sampler_prog_key_data *key_tex;

   struct brw_gs_compile *gs_compile;

   struct brw_stage_prog_data *prog_data;
   struct gl_program *prog;

   const struct brw_vue_map *input_vue_map;

   int *virtual_grf_start;
   int *virtual_grf_end;
   brw::fs_live_variables *live_intervals;

   int *regs_live_at_ip;

   /** Number of uniform variable components visited. */
   unsigned uniforms;

   /** Byte-offset for the next available spot in the scratch space buffer. */
   unsigned last_scratch;

   /**
    * Array mapping UNIFORM register numbers to the pull parameter index,
    * or -1 if this uniform register isn't being uploaded as a pull constant.
    */
   int *pull_constant_loc;

   /**
    * Array mapping UNIFORM register numbers to the push parameter index,
    * or -1 if this uniform register isn't being uploaded as a push constant.
    */
   int *push_constant_loc;

   fs_reg subgroup_id;
   fs_reg frag_depth;
   fs_reg frag_stencil;
   fs_reg sample_mask;
   fs_reg outputs[VARYING_SLOT_MAX];
   fs_reg dual_src_output;
   int first_non_payload_grf;
   /** Either BRW_MAX_GRF or GEN7_MRF_HACK_START */
   unsigned max_grf;

   fs_reg *nir_locals;
   fs_reg *nir_ssa_values;
   fs_reg *nir_system_values;

   bool failed;
   char *fail_msg;

   /** Register numbers for thread payload fields. */
   struct thread_payload {
      uint8_t subspan_coord_reg[2];
      uint8_t source_depth_reg[2];
      uint8_t source_w_reg[2];
      uint8_t aa_dest_stencil_reg[2];
      uint8_t dest_depth_reg[2];
      uint8_t sample_pos_reg[2];
      uint8_t sample_mask_in_reg[2];
      uint8_t barycentric_coord_reg[BRW_BARYCENTRIC_MODE_COUNT][2];
      uint8_t local_invocation_id_reg[2];

      /** The number of thread payload registers the hardware will supply. */
      uint8_t num_regs;
   } payload;

   bool source_depth_to_render_target;
   bool runtime_check_aads_emit;

   fs_reg pixel_x;
   fs_reg pixel_y;
   fs_reg wpos_w;
   fs_reg pixel_w;
   fs_reg delta_xy[BRW_BARYCENTRIC_MODE_COUNT];
   fs_reg shader_start_time;
   fs_reg userplane[MAX_CLIP_PLANES];
   fs_reg final_gs_vertex_count;
   fs_reg control_data_bits;
   fs_reg invocation_id;

   unsigned grf_used;
   bool spilled_any_registers;

   const unsigned dispatch_width; /**< 8, 16 or 32 */
   unsigned max_dispatch_width;

   int shader_time_index;

   unsigned promoted_constants;
   brw::fs_builder bld;
};

/**
 * The fragment shader code generator.
 *
 * Translates FS IR to actual i965 assembly code.
 */
class fs_generator
{
public:
   fs_generator(const struct brw_compiler *compiler, void *log_data,
                void *mem_ctx,
                struct brw_stage_prog_data *prog_data,
                unsigned promoted_constants,
                bool runtime_check_aads_emit,
                gl_shader_stage stage);
   ~fs_generator();

   void enable_debug(const char *shader_name);
   int generate_code(const cfg_t *cfg, int dispatch_width);
   const unsigned *get_assembly();

private:
   void fire_fb_write(fs_inst *inst,
                      struct brw_reg payload,
                      struct brw_reg implied_header,
                      GLuint nr);
   void generate_fb_write(fs_inst *inst, struct brw_reg payload);
   void generate_fb_read(fs_inst *inst, struct brw_reg dst,
                         struct brw_reg payload);
   void generate_urb_read(fs_inst *inst, struct brw_reg dst, struct brw_reg payload);
   void generate_urb_write(fs_inst *inst, struct brw_reg payload);
   void generate_cs_terminate(fs_inst *inst, struct brw_reg payload);
   void generate_barrier(fs_inst *inst, struct brw_reg src);
   bool generate_linterp(fs_inst *inst, struct brw_reg dst,
			 struct brw_reg *src);
   void generate_tex(fs_inst *inst, struct brw_reg dst, struct brw_reg src,
                     struct brw_reg surface_index,
                     struct brw_reg sampler_index);
   void generate_get_buffer_size(fs_inst *inst, struct brw_reg dst,
                                 struct brw_reg src,
                                 struct brw_reg surf_index);
   void generate_ddx(const fs_inst *inst,
                     struct brw_reg dst, struct brw_reg src);
   void generate_ddy(const fs_inst *inst,
                     struct brw_reg dst, struct brw_reg src);
   void generate_scratch_write(fs_inst *inst, struct brw_reg src);
   void generate_scratch_read(fs_inst *inst, struct brw_reg dst);
   void generate_scratch_read_gen7(fs_inst *inst, struct brw_reg dst);
   void generate_uniform_pull_constant_load(fs_inst *inst, struct brw_reg dst,
                                            struct brw_reg index,
                                            struct brw_reg offset);
   void generate_uniform_pull_constant_load_gen7(fs_inst *inst,
                                                 struct brw_reg dst,
                                                 struct brw_reg surf_index,
                                                 struct brw_reg payload);
   void generate_varying_pull_constant_load_gen4(fs_inst *inst,
                                                 struct brw_reg dst,
                                                 struct brw_reg index);
   void generate_varying_pull_constant_load_gen7(fs_inst *inst,
                                                 struct brw_reg dst,
                                                 struct brw_reg index,
                                                 struct brw_reg offset);
   void generate_mov_dispatch_to_flags(fs_inst *inst);

   void generate_pixel_interpolator_query(fs_inst *inst,
                                          struct brw_reg dst,
                                          struct brw_reg src,
                                          struct brw_reg msg_data,
                                          unsigned msg_type);

   void generate_set_sample_id(fs_inst *inst,
                               struct brw_reg dst,
                               struct brw_reg src0,
                               struct brw_reg src1);

   void generate_discard_jump(fs_inst *inst);

   void generate_pack_half_2x16_split(fs_inst *inst,
                                      struct brw_reg dst,
                                      struct brw_reg x,
                                      struct brw_reg y);
   void generate_unpack_half_2x16_split(fs_inst *inst,
                                        struct brw_reg dst,
                                        struct brw_reg src);

   void generate_shader_time_add(fs_inst *inst,
                                 struct brw_reg payload,
                                 struct brw_reg offset,
                                 struct brw_reg value);

   void generate_mov_indirect(fs_inst *inst,
                              struct brw_reg dst,
                              struct brw_reg reg,
                              struct brw_reg indirect_byte_offset);

   void generate_shuffle(fs_inst *inst,
                         struct brw_reg dst,
                         struct brw_reg src,
                         struct brw_reg idx);

   void generate_quad_swizzle(const fs_inst *inst,
                              struct brw_reg dst, struct brw_reg src,
                              unsigned swiz);

   bool patch_discard_jumps_to_fb_writes();

   const struct brw_compiler *compiler;
   void *log_data; /* Passed to compiler->*_log functions */

   const struct gen_device_info *devinfo;

   struct brw_codegen *p;
   struct brw_stage_prog_data * const prog_data;

   unsigned dispatch_width; /**< 8, 16 or 32 */

   exec_list discard_halt_patches;
   unsigned promoted_constants;
   bool runtime_check_aads_emit;
   bool debug_flag;
   const char *shader_name;
   gl_shader_stage stage;
   void *mem_ctx;
};

namespace brw {
   inline fs_reg
   fetch_payload_reg(const brw::fs_builder &bld, uint8_t regs[2],
                     brw_reg_type type = BRW_REGISTER_TYPE_F, unsigned n = 1)
   {
      if (!regs[0])
         return fs_reg();

      if (bld.dispatch_width() > 16) {
         const fs_reg tmp = bld.vgrf(type, n);
         const brw::fs_builder hbld = bld.exec_all().group(16, 0);
         const unsigned m = bld.dispatch_width() / hbld.dispatch_width();
         fs_reg *const components = new fs_reg[n * m];

         for (unsigned c = 0; c < n; c++) {
            for (unsigned g = 0; g < m; g++) {
               components[c * m + g] =
                  offset(retype(brw_vec8_grf(regs[g], 0), type), hbld, c);
            }
         }

         hbld.LOAD_PAYLOAD(tmp, components, n * m, 0);

         delete[] components;
         return tmp;

      } else {
         return fs_reg(retype(brw_vec8_grf(regs[0], 0), type));
      }
   }

   /**
    * Remove any modifiers from the \p i-th source region of the instruction,
    * including negate, abs and any implicit type conversion to the execution
    * type.  Instead any source modifiers will be implemented as a separate
    * MOV instruction prior to the original instruction.
    */
   inline bool
   lower_src_modifiers(fs_visitor *v, bblock_t *block, fs_inst *inst, unsigned i)
   {
      assert(inst->components_read(i) == 1);
      const fs_builder ibld(v, block, inst);
      const fs_reg tmp = ibld.vgrf(get_exec_type(inst));

      ibld.MOV(tmp, inst->src[i]);
      inst->src[i] = tmp;

      return true;
   }
}

void shuffle_from_32bit_read(const brw::fs_builder &bld,
                             const fs_reg &dst,
                             const fs_reg &src,
                             uint32_t first_component,
                             uint32_t components);

fs_reg shuffle_for_32bit_write(const brw::fs_builder &bld,
                               const fs_reg &src,
                               uint32_t first_component,
                               uint32_t components);

fs_reg setup_imm_df(const brw::fs_builder &bld,
                    double v);

fs_reg setup_imm_b(const brw::fs_builder &bld,
                   int8_t v);

fs_reg setup_imm_ub(const brw::fs_builder &bld,
                   uint8_t v);

enum brw_barycentric_mode brw_barycentric_mode(enum glsl_interp_mode mode,
                                               nir_intrinsic_op op);

#endif /* BRW_FS_H */
