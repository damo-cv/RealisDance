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

#ifndef BRW_SHADER_H
#define BRW_SHADER_H

#include <stdint.h>
#include "brw_reg.h"
#include "brw_compiler.h"
#include "brw_eu_defines.h"
#include "brw_inst.h"
#include "compiler/nir/nir.h"

#ifdef __cplusplus
#include "brw_ir_allocator.h"
#endif

#define MAX_SAMPLER_MESSAGE_SIZE 11
#define MAX_VGRF_SIZE 16

#ifdef __cplusplus
struct backend_reg : private brw_reg
{
   backend_reg() {}
   backend_reg(const struct brw_reg &reg) : brw_reg(reg) {}

   const brw_reg &as_brw_reg() const
   {
      assert(file == ARF || file == FIXED_GRF || file == MRF || file == IMM);
      assert(offset == 0);
      return static_cast<const brw_reg &>(*this);
   }

   brw_reg &as_brw_reg()
   {
      assert(file == ARF || file == FIXED_GRF || file == MRF || file == IMM);
      assert(offset == 0);
      return static_cast<brw_reg &>(*this);
   }

   bool equals(const backend_reg &r) const;
   bool negative_equals(const backend_reg &r) const;

   bool is_zero() const;
   bool is_one() const;
   bool is_negative_one() const;
   bool is_null() const;
   bool is_accumulator() const;

   /** Offset from the start of the (virtual) register in bytes. */
   uint16_t offset;

   using brw_reg::type;
   using brw_reg::file;
   using brw_reg::negate;
   using brw_reg::abs;
   using brw_reg::address_mode;
   using brw_reg::subnr;
   using brw_reg::nr;

   using brw_reg::swizzle;
   using brw_reg::writemask;
   using brw_reg::indirect_offset;
   using brw_reg::vstride;
   using brw_reg::width;
   using brw_reg::hstride;

   using brw_reg::df;
   using brw_reg::f;
   using brw_reg::d;
   using brw_reg::ud;
   using brw_reg::d64;
   using brw_reg::u64;
};
#endif

struct cfg_t;
struct bblock_t;

#ifdef __cplusplus
struct backend_instruction : public exec_node {
   bool is_3src(const struct gen_device_info *devinfo) const;
   bool is_tex() const;
   bool is_math() const;
   bool is_control_flow() const;
   bool is_commutative() const;
   bool can_do_source_mods() const;
   bool can_do_saturate() const;
   bool can_do_cmod() const;
   bool reads_accumulator_implicitly() const;
   bool writes_accumulator_implicitly(const struct gen_device_info *devinfo) const;

   void remove(bblock_t *block);
   void insert_after(bblock_t *block, backend_instruction *inst);
   void insert_before(bblock_t *block, backend_instruction *inst);
   void insert_before(bblock_t *block, exec_list *list);

   /**
    * True if the instruction has side effects other than writing to
    * its destination registers.  You are expected not to reorder or
    * optimize these out unless you know what you are doing.
    */
   bool has_side_effects() const;

   /**
    * True if the instruction might be affected by side effects of other
    * instructions.
    */
   bool is_volatile() const;
#else
struct backend_instruction {
   struct exec_node link;
#endif
   /** @{
    * Annotation for the generated IR.  One of the two can be set.
    */
   const void *ir;
   const char *annotation;
   /** @} */

   /**
    * Execution size of the instruction.  This is used by the generator to
    * generate the correct binary for the given instruction.  Current valid
    * values are 1, 4, 8, 16, 32.
    */
   uint8_t exec_size;

   /**
    * Channel group from the hardware execution and predication mask that
    * should be applied to the instruction.  The subset of channel enable
    * signals (calculated from the EU control flow and predication state)
    * given by [group, group + exec_size) will be used to mask GRF writes and
    * any other side effects of the instruction.
    */
   uint8_t group;

   uint32_t offset; /**< spill/unspill offset or texture offset bitfield */
   uint8_t mlen; /**< SEND message length */
   int8_t base_mrf; /**< First MRF in the SEND message, if mlen is nonzero. */
   uint8_t target; /**< MRT target. */
   unsigned size_written; /**< Data written to the destination register in bytes. */

   enum opcode opcode; /* BRW_OPCODE_* or FS_OPCODE_* */
   enum brw_conditional_mod conditional_mod; /**< BRW_CONDITIONAL_* */
   enum brw_predicate predicate;
   bool predicate_inverse:1;
   bool writes_accumulator:1; /**< instruction implicitly writes accumulator */
   bool force_writemask_all:1;
   bool no_dd_clear:1;
   bool no_dd_check:1;
   bool saturate:1;
   bool shadow_compare:1;
   bool eot:1;

   /* Chooses which flag subregister (f0.0 to f1.1) is used for conditional
    * mod and predication.
    */
   unsigned flag_subreg:2;

   /** The number of hardware registers used for a message header. */
   uint8_t header_size;
};

#ifdef __cplusplus

enum instruction_scheduler_mode {
   SCHEDULE_PRE,
   SCHEDULE_PRE_NON_LIFO,
   SCHEDULE_PRE_LIFO,
   SCHEDULE_POST,
};

struct backend_shader {
protected:

   backend_shader(const struct brw_compiler *compiler,
                  void *log_data,
                  void *mem_ctx,
                  const nir_shader *shader,
                  struct brw_stage_prog_data *stage_prog_data);

public:
   virtual ~backend_shader();

   const struct brw_compiler *compiler;
   void *log_data; /* Passed to compiler->*_log functions */

   const struct gen_device_info * const devinfo;
   const nir_shader *nir;
   struct brw_stage_prog_data * const stage_prog_data;

   /** ralloc context for temporary data used during compile */
   void *mem_ctx;

   /**
    * List of either fs_inst or vec4_instruction (inheriting from
    * backend_instruction)
    */
   exec_list instructions;

   cfg_t *cfg;

   gl_shader_stage stage;
   bool debug_enabled;
   const char *stage_name;
   const char *stage_abbrev;

   brw::simple_allocator alloc;

   virtual void dump_instruction(backend_instruction *inst) = 0;
   virtual void dump_instruction(backend_instruction *inst, FILE *file) = 0;
   virtual void dump_instructions();
   virtual void dump_instructions(const char *name);

   void calculate_cfg();

   virtual void invalidate_live_intervals() = 0;
};

bool brw_texture_offset(int *offsets,
                        unsigned num_components,
                        uint32_t *offset_bits);

#else
struct backend_shader;
#endif /* __cplusplus */

enum brw_reg_type brw_type_for_base_type(const struct glsl_type *type);
enum brw_conditional_mod brw_conditional_for_comparison(unsigned int op);
uint32_t brw_math_function(enum opcode op);
const char *brw_instruction_name(const struct gen_device_info *devinfo,
                                 enum opcode op);
bool brw_saturate_immediate(enum brw_reg_type type, struct brw_reg *reg);
bool brw_negate_immediate(enum brw_reg_type type, struct brw_reg *reg);
bool brw_abs_immediate(enum brw_reg_type type, struct brw_reg *reg);

bool opt_predicated_break(struct backend_shader *s);

#ifdef __cplusplus
extern "C" {
#endif

/* brw_fs_reg_allocate.cpp */
void brw_fs_alloc_reg_sets(struct brw_compiler *compiler);

/* brw_vec4_reg_allocate.cpp */
void brw_vec4_alloc_reg_set(struct brw_compiler *compiler);

/* brw_disasm.c */
extern const char *const conditional_modifier[16];
extern const char *const pred_ctrl_align16[16];

/* Per-thread scratch space is a power-of-two multiple of 1KB. */
static inline int
brw_get_scratch_size(int size)
{
   return MAX2(1024, util_next_power_of_two(size));
}

/**
 * Scratch data used when compiling a GLSL geometry shader.
 */
struct brw_gs_compile
{
   struct brw_gs_prog_key key;
   struct brw_vue_map input_vue_map;

   unsigned control_data_bits_per_vertex;
   unsigned control_data_header_size_bits;
};

#ifdef __cplusplus
}
#endif

#endif /* BRW_SHADER_H */
