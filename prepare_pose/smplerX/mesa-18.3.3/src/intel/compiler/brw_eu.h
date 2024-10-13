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


#ifndef BRW_EU_H
#define BRW_EU_H

#include <stdbool.h>
#include <stdio.h>
#include "brw_inst.h"
#include "brw_eu_defines.h"
#include "brw_reg.h"
#include "brw_disasm_info.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BRW_EU_MAX_INSN_STACK 5

struct brw_insn_state {
   /* One of BRW_EXECUTE_* */
   unsigned exec_size:3;

   /* Group in units of channels */
   unsigned group:5;

   /* Compression control on gen4-5 */
   bool compressed:1;

   /* One of BRW_MASK_* */
   unsigned mask_control:1;

   bool saturate:1;

   /* One of BRW_ALIGN_* */
   unsigned access_mode:1;

   /* One of BRW_PREDICATE_* */
   enum brw_predicate predicate:4;

   bool pred_inv:1;

   /* Flag subreg.  Bottom bit is subreg, top bit is reg */
   unsigned flag_subreg:2;

   bool acc_wr_control:1;
};


/* A helper for accessing the last instruction emitted.  This makes it easy
 * to set various bits on an instruction without having to create temporary
 * variable and assign the emitted instruction to those.
 */
#define brw_last_inst (&p->store[p->nr_insn - 1])

struct brw_codegen {
   brw_inst *store;
   int store_size;
   unsigned nr_insn;
   unsigned int next_insn_offset;

   void *mem_ctx;

   /* Allow clients to push/pop instruction state:
    */
   struct brw_insn_state stack[BRW_EU_MAX_INSN_STACK];
   struct brw_insn_state *current;

   /** Whether or not the user wants automatic exec sizes
    *
    * If true, codegen will try to automatically infer the exec size of an
    * instruction from the width of the destination register.  If false, it
    * will take whatever is set by brw_set_default_exec_size verbatim.
    *
    * This is set to true by default in brw_init_codegen.
    */
   bool automatic_exec_sizes;

   bool single_program_flow;
   const struct gen_device_info *devinfo;

   /* Control flow stacks:
    * - if_stack contains IF and ELSE instructions which must be patched
    *   (and popped) once the matching ENDIF instruction is encountered.
    *
    *   Just store the instruction pointer(an index).
    */
   int *if_stack;
   int if_stack_depth;
   int if_stack_array_size;

   /**
    * loop_stack contains the instruction pointers of the starts of loops which
    * must be patched (and popped) once the matching WHILE instruction is
    * encountered.
    */
   int *loop_stack;
   /**
    * pre-gen6, the BREAK and CONT instructions had to tell how many IF/ENDIF
    * blocks they were popping out of, to fix up the mask stack.  This tracks
    * the IF/ENDIF nesting in each current nested loop level.
    */
   int *if_depth_in_loop;
   int loop_stack_depth;
   int loop_stack_array_size;
};

void brw_pop_insn_state( struct brw_codegen *p );
void brw_push_insn_state( struct brw_codegen *p );
unsigned brw_get_default_exec_size(struct brw_codegen *p);
unsigned brw_get_default_group(struct brw_codegen *p);
unsigned brw_get_default_access_mode(struct brw_codegen *p);
void brw_set_default_exec_size(struct brw_codegen *p, unsigned value);
void brw_set_default_mask_control( struct brw_codegen *p, unsigned value );
void brw_set_default_saturate( struct brw_codegen *p, bool enable );
void brw_set_default_access_mode( struct brw_codegen *p, unsigned access_mode );
void brw_inst_set_compression(const struct gen_device_info *devinfo,
                              brw_inst *inst, bool on);
void brw_set_default_compression(struct brw_codegen *p, bool on);
void brw_inst_set_group(const struct gen_device_info *devinfo,
                        brw_inst *inst, unsigned group);
void brw_set_default_group(struct brw_codegen *p, unsigned group);
void brw_set_default_compression_control(struct brw_codegen *p, enum brw_compression c);
void brw_set_default_predicate_control( struct brw_codegen *p, unsigned pc );
void brw_set_default_predicate_inverse(struct brw_codegen *p, bool predicate_inverse);
void brw_set_default_flag_reg(struct brw_codegen *p, int reg, int subreg);
void brw_set_default_acc_write_control(struct brw_codegen *p, unsigned value);

void brw_init_codegen(const struct gen_device_info *, struct brw_codegen *p,
		      void *mem_ctx);
int brw_disassemble_inst(FILE *file, const struct gen_device_info *devinfo,
                         const struct brw_inst *inst, bool is_compacted);
void brw_disassemble(const struct gen_device_info *devinfo,
                     const void *assembly, int start, int end, FILE *out);
const unsigned *brw_get_program( struct brw_codegen *p, unsigned *sz );

brw_inst *brw_next_insn(struct brw_codegen *p, unsigned opcode);
void brw_set_dest(struct brw_codegen *p, brw_inst *insn, struct brw_reg dest);
void brw_set_src0(struct brw_codegen *p, brw_inst *insn, struct brw_reg reg);

void gen6_resolve_implied_move(struct brw_codegen *p,
			       struct brw_reg *src,
			       unsigned msg_reg_nr);

/* Helpers for regular instructions:
 */
#define ALU1(OP)				\
brw_inst *brw_##OP(struct brw_codegen *p,	\
	      struct brw_reg dest,		\
	      struct brw_reg src0);

#define ALU2(OP)				\
brw_inst *brw_##OP(struct brw_codegen *p,	\
	      struct brw_reg dest,		\
	      struct brw_reg src0,		\
	      struct brw_reg src1);

#define ALU3(OP)				\
brw_inst *brw_##OP(struct brw_codegen *p,	\
	      struct brw_reg dest,		\
	      struct brw_reg src0,		\
	      struct brw_reg src1,		\
	      struct brw_reg src2);

#define ROUND(OP) \
void brw_##OP(struct brw_codegen *p, struct brw_reg dest, struct brw_reg src0);

ALU1(MOV)
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
ALU1(F32TO16)
ALU1(F16TO32)
ALU2(ADD)
ALU2(AVG)
ALU2(MUL)
ALU1(FRC)
ALU1(RNDD)
ALU2(MAC)
ALU2(MACH)
ALU1(LZD)
ALU2(DP4)
ALU2(DPH)
ALU2(DP3)
ALU2(DP2)
ALU2(LINE)
ALU2(PLN)
ALU3(MAD)
ALU3(LRP)
ALU1(BFREV)
ALU3(BFE)
ALU2(BFI1)
ALU3(BFI2)
ALU1(FBH)
ALU1(FBL)
ALU1(CBIT)
ALU2(ADDC)
ALU2(SUBB)
ALU2(MAC)

ROUND(RNDZ)
ROUND(RNDE)

#undef ALU1
#undef ALU2
#undef ALU3
#undef ROUND


/* Helpers for SEND instruction:
 */

/**
 * Construct a message descriptor immediate with the specified common
 * descriptor controls.
 */
static inline uint32_t
brw_message_desc(const struct gen_device_info *devinfo,
                 unsigned msg_length,
                 unsigned response_length,
                 bool header_present)
{
   if (devinfo->gen >= 5) {
      return (SET_BITS(msg_length, 28, 25) |
              SET_BITS(response_length, 24, 20) |
              SET_BITS(header_present, 19, 19));
   } else {
      return (SET_BITS(msg_length, 23, 20) |
              SET_BITS(response_length, 19, 16));
   }
}

/**
 * Construct a message descriptor immediate with the specified sampler
 * function controls.
 */
static inline uint32_t
brw_sampler_desc(const struct gen_device_info *devinfo,
                 unsigned binding_table_index,
                 unsigned sampler,
                 unsigned msg_type,
                 unsigned simd_mode,
                 unsigned return_format)
{
   const unsigned desc = (SET_BITS(binding_table_index, 7, 0) |
                          SET_BITS(sampler, 11, 8));
   if (devinfo->gen >= 7)
      return (desc | SET_BITS(msg_type, 16, 12) |
              SET_BITS(simd_mode, 18, 17));
   else if (devinfo->gen >= 5)
      return (desc | SET_BITS(msg_type, 15, 12) |
              SET_BITS(simd_mode, 17, 16));
   else if (devinfo->is_g4x)
      return desc | SET_BITS(msg_type, 15, 12);
   else
      return (desc | SET_BITS(return_format, 13, 12) |
              SET_BITS(msg_type, 15, 14));
}

/**
 * Construct a message descriptor immediate with the specified dataport read
 * function controls.
 */
static inline uint32_t
brw_dp_read_desc(const struct gen_device_info *devinfo,
                 unsigned binding_table_index,
                 unsigned msg_control,
                 unsigned msg_type,
                 unsigned target_cache)
{
   const unsigned desc = SET_BITS(binding_table_index, 7, 0);
   if (devinfo->gen >= 7)
      return (desc | SET_BITS(msg_control, 13, 8) |
              SET_BITS(msg_type, 17, 14));
   else if (devinfo->gen >= 6)
      return (desc | SET_BITS(msg_control, 12, 8) |
              SET_BITS(msg_type, 16, 13));
   else if (devinfo->gen >= 5 || devinfo->is_g4x)
      return (desc | SET_BITS(msg_control, 10, 8) |
              SET_BITS(msg_type, 13, 11) |
              SET_BITS(target_cache, 15, 14));
   else
      return (desc | SET_BITS(msg_control, 11, 8) |
              SET_BITS(msg_type, 13, 12) |
              SET_BITS(target_cache, 15, 14));
}

/**
 * Construct a message descriptor immediate with the specified dataport write
 * function controls.
 */
static inline uint32_t
brw_dp_write_desc(const struct gen_device_info *devinfo,
                  unsigned binding_table_index,
                  unsigned msg_control,
                  unsigned msg_type,
                  unsigned last_render_target,
                  unsigned send_commit_msg)
{
   const unsigned desc = SET_BITS(binding_table_index, 7, 0);
   if (devinfo->gen >= 7)
      return (desc | SET_BITS(msg_control, 13, 8) |
              SET_BITS(last_render_target, 12, 12) |
              SET_BITS(msg_type, 17, 14));
   else if (devinfo->gen >= 6)
      return (desc | SET_BITS(msg_control, 12, 8) |
              SET_BITS(last_render_target, 12, 12) |
              SET_BITS(msg_type, 16, 13) |
              SET_BITS(send_commit_msg, 17, 17));
   else
      return (desc | SET_BITS(msg_control, 11, 8) |
              SET_BITS(last_render_target, 11, 11) |
              SET_BITS(msg_type, 14, 12) |
              SET_BITS(send_commit_msg, 15, 15));
}

/**
 * Construct a message descriptor immediate with the specified dataport
 * surface function controls.
 */
static inline uint32_t
brw_dp_surface_desc(const struct gen_device_info *devinfo,
                    unsigned msg_type,
                    unsigned msg_control)
{
   assert(devinfo->gen >= 7);
   if (devinfo->gen >= 8) {
      return (SET_BITS(msg_control, 13, 8) |
              SET_BITS(msg_type, 18, 14));
   } else {
      return (SET_BITS(msg_control, 13, 8) |
              SET_BITS(msg_type, 17, 14));
   }
}

/**
 * Construct a message descriptor immediate with the specified pixel
 * interpolator function controls.
 */
static inline uint32_t
brw_pixel_interp_desc(UNUSED const struct gen_device_info *devinfo,
                      unsigned msg_type,
                      bool noperspective,
                      unsigned simd_mode,
                      unsigned slot_group)
{
   return (SET_BITS(slot_group, 11, 11) |
           SET_BITS(msg_type, 13, 12) |
           SET_BITS(!!noperspective, 14, 14) |
           SET_BITS(simd_mode, 16, 16));
}

void brw_urb_WRITE(struct brw_codegen *p,
		   struct brw_reg dest,
		   unsigned msg_reg_nr,
		   struct brw_reg src0,
                   enum brw_urb_write_flags flags,
		   unsigned msg_length,
		   unsigned response_length,
		   unsigned offset,
		   unsigned swizzle);

/**
 * Send message to shared unit \p sfid with a possibly indirect descriptor \p
 * desc.  If \p desc is not an immediate it will be transparently loaded to an
 * address register using an OR instruction.
 */
void
brw_send_indirect_message(struct brw_codegen *p,
                          unsigned sfid,
                          struct brw_reg dst,
                          struct brw_reg payload,
                          struct brw_reg desc,
                          unsigned desc_imm);

void brw_ff_sync(struct brw_codegen *p,
		   struct brw_reg dest,
		   unsigned msg_reg_nr,
		   struct brw_reg src0,
		   bool allocate,
		   unsigned response_length,
		   bool eot);

void brw_svb_write(struct brw_codegen *p,
                   struct brw_reg dest,
                   unsigned msg_reg_nr,
                   struct brw_reg src0,
                   unsigned binding_table_index,
                   bool   send_commit_msg);

brw_inst *brw_fb_WRITE(struct brw_codegen *p,
                       struct brw_reg payload,
                       struct brw_reg implied_header,
                       unsigned msg_control,
                       unsigned binding_table_index,
                       unsigned msg_length,
                       unsigned response_length,
                       bool eot,
                       bool last_render_target,
                       bool header_present);

brw_inst *gen9_fb_READ(struct brw_codegen *p,
                       struct brw_reg dst,
                       struct brw_reg payload,
                       unsigned binding_table_index,
                       unsigned msg_length,
                       unsigned response_length,
                       bool per_sample);

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
		unsigned return_format);

void brw_adjust_sampler_state_pointer(struct brw_codegen *p,
                                      struct brw_reg header,
                                      struct brw_reg sampler_index);

void gen4_math(struct brw_codegen *p,
	       struct brw_reg dest,
	       unsigned function,
	       unsigned msg_reg_nr,
	       struct brw_reg src,
	       unsigned precision );

void gen6_math(struct brw_codegen *p,
	       struct brw_reg dest,
	       unsigned function,
	       struct brw_reg src0,
	       struct brw_reg src1);

void brw_oword_block_read(struct brw_codegen *p,
			  struct brw_reg dest,
			  struct brw_reg mrf,
			  uint32_t offset,
			  uint32_t bind_table_index);

unsigned brw_scratch_surface_idx(const struct brw_codegen *p);

void brw_oword_block_read_scratch(struct brw_codegen *p,
				  struct brw_reg dest,
				  struct brw_reg mrf,
				  int num_regs,
				  unsigned offset);

void brw_oword_block_write_scratch(struct brw_codegen *p,
				   struct brw_reg mrf,
				   int num_regs,
				   unsigned offset);

void gen7_block_read_scratch(struct brw_codegen *p,
                             struct brw_reg dest,
                             int num_regs,
                             unsigned offset);

void brw_shader_time_add(struct brw_codegen *p,
                         struct brw_reg payload,
                         uint32_t surf_index);

/**
 * Return the generation-specific jump distance scaling factor.
 *
 * Given the number of instructions to jump, we need to scale by
 * some number to obtain the actual jump distance to program in an
 * instruction.
 */
static inline unsigned
brw_jump_scale(const struct gen_device_info *devinfo)
{
   /* Broadwell measures jump targets in bytes. */
   if (devinfo->gen >= 8)
      return 16;

   /* Ironlake and later measure jump targets in 64-bit data chunks (in order
    * (to support compaction), so each 128-bit instruction requires 2 chunks.
    */
   if (devinfo->gen >= 5)
      return 2;

   /* Gen4 simply uses the number of 128-bit instructions. */
   return 1;
}

void brw_barrier(struct brw_codegen *p, struct brw_reg src);

/* If/else/endif.  Works by manipulating the execution flags on each
 * channel.
 */
brw_inst *brw_IF(struct brw_codegen *p, unsigned execute_size);
brw_inst *gen6_IF(struct brw_codegen *p, enum brw_conditional_mod conditional,
                  struct brw_reg src0, struct brw_reg src1);

void brw_ELSE(struct brw_codegen *p);
void brw_ENDIF(struct brw_codegen *p);

/* DO/WHILE loops:
 */
brw_inst *brw_DO(struct brw_codegen *p, unsigned execute_size);

brw_inst *brw_WHILE(struct brw_codegen *p);

brw_inst *brw_BREAK(struct brw_codegen *p);
brw_inst *brw_CONT(struct brw_codegen *p);
brw_inst *gen6_HALT(struct brw_codegen *p);

/* Forward jumps:
 */
void brw_land_fwd_jump(struct brw_codegen *p, int jmp_insn_idx);

brw_inst *brw_JMPI(struct brw_codegen *p, struct brw_reg index,
                   unsigned predicate_control);

void brw_NOP(struct brw_codegen *p);

void brw_WAIT(struct brw_codegen *p);

/* Special case: there is never a destination, execution size will be
 * taken from src0:
 */
void brw_CMP(struct brw_codegen *p,
	     struct brw_reg dest,
	     unsigned conditional,
	     struct brw_reg src0,
	     struct brw_reg src1);

void
brw_untyped_atomic(struct brw_codegen *p,
                   struct brw_reg dst,
                   struct brw_reg payload,
                   struct brw_reg surface,
                   unsigned atomic_op,
                   unsigned msg_length,
                   bool response_expected,
                   bool header_present);

void
brw_untyped_atomic_float(struct brw_codegen *p,
                         struct brw_reg dst,
                         struct brw_reg payload,
                         struct brw_reg surface,
                         unsigned atomic_op,
                         unsigned msg_length,
                         bool response_expected,
                         bool header_present);


void
brw_untyped_surface_read(struct brw_codegen *p,
                         struct brw_reg dst,
                         struct brw_reg payload,
                         struct brw_reg surface,
                         unsigned msg_length,
                         unsigned num_channels);

void
brw_untyped_surface_write(struct brw_codegen *p,
                          struct brw_reg payload,
                          struct brw_reg surface,
                          unsigned msg_length,
                          unsigned num_channels,
                          bool header_present);

void
brw_typed_atomic(struct brw_codegen *p,
                 struct brw_reg dst,
                 struct brw_reg payload,
                 struct brw_reg surface,
                 unsigned atomic_op,
                 unsigned msg_length,
                 bool response_expected,
                 bool header_present);

void
brw_typed_surface_read(struct brw_codegen *p,
                       struct brw_reg dst,
                       struct brw_reg payload,
                       struct brw_reg surface,
                       unsigned msg_length,
                       unsigned num_channels,
                       bool header_present);

void
brw_typed_surface_write(struct brw_codegen *p,
                        struct brw_reg payload,
                        struct brw_reg surface,
                        unsigned msg_length,
                        unsigned num_channels,
                        bool header_present);

void
brw_byte_scattered_read(struct brw_codegen *p,
                        struct brw_reg dst,
                        struct brw_reg payload,
                        struct brw_reg surface,
                        unsigned msg_length,
                        unsigned bit_size);

void
brw_byte_scattered_write(struct brw_codegen *p,
                         struct brw_reg payload,
                         struct brw_reg surface,
                         unsigned msg_length,
                         unsigned bit_size,
                         bool header_present);

void
brw_memory_fence(struct brw_codegen *p,
                 struct brw_reg dst,
                 enum opcode send_op);

void
brw_pixel_interpolator_query(struct brw_codegen *p,
                             struct brw_reg dest,
                             struct brw_reg mrf,
                             bool noperspective,
                             unsigned mode,
                             struct brw_reg data,
                             unsigned msg_length,
                             unsigned response_length);

void
brw_find_live_channel(struct brw_codegen *p,
                      struct brw_reg dst,
                      struct brw_reg mask);

void
brw_broadcast(struct brw_codegen *p,
              struct brw_reg dst,
              struct brw_reg src,
              struct brw_reg idx);

void
brw_rounding_mode(struct brw_codegen *p,
                  enum brw_rnd_mode mode);

/***********************************************************************
 * brw_eu_util.c:
 */

void brw_copy_indirect_to_indirect(struct brw_codegen *p,
				   struct brw_indirect dst_ptr,
				   struct brw_indirect src_ptr,
				   unsigned count);

void brw_copy_from_indirect(struct brw_codegen *p,
			    struct brw_reg dst,
			    struct brw_indirect ptr,
			    unsigned count);

void brw_copy4(struct brw_codegen *p,
	       struct brw_reg dst,
	       struct brw_reg src,
	       unsigned count);

void brw_copy8(struct brw_codegen *p,
	       struct brw_reg dst,
	       struct brw_reg src,
	       unsigned count);

void brw_math_invert( struct brw_codegen *p,
		      struct brw_reg dst,
		      struct brw_reg src);

void brw_set_src1(struct brw_codegen *p, brw_inst *insn, struct brw_reg reg);

void brw_set_desc_ex(struct brw_codegen *p, brw_inst *insn,
                     unsigned desc, unsigned ex_desc);

static inline void
brw_set_desc(struct brw_codegen *p, brw_inst *insn, unsigned desc)
{
   brw_set_desc_ex(p, insn, desc, 0);
}

void brw_set_uip_jip(struct brw_codegen *p, int start_offset);

enum brw_conditional_mod brw_negate_cmod(uint32_t cmod);
enum brw_conditional_mod brw_swap_cmod(uint32_t cmod);

/* brw_eu_compact.c */
void brw_init_compaction_tables(const struct gen_device_info *devinfo);
void brw_compact_instructions(struct brw_codegen *p, int start_offset,
                              struct disasm_info *disasm);
void brw_uncompact_instruction(const struct gen_device_info *devinfo,
                               brw_inst *dst, brw_compact_inst *src);
bool brw_try_compact_instruction(const struct gen_device_info *devinfo,
                                 brw_compact_inst *dst, const brw_inst *src);

void brw_debug_compact_uncompact(const struct gen_device_info *devinfo,
                                 brw_inst *orig, brw_inst *uncompacted);

/* brw_eu_validate.c */
bool brw_validate_instructions(const struct gen_device_info *devinfo,
                               const void *assembly, int start_offset, int end_offset,
                               struct disasm_info *disasm);

static inline int
next_offset(const struct gen_device_info *devinfo, void *store, int offset)
{
   brw_inst *insn = (brw_inst *)((char *)store + offset);

   if (brw_inst_cmpt_control(devinfo, insn))
      return offset + 8;
   else
      return offset + 16;
}

struct opcode_desc {
   /* The union is an implementation detail used by brw_opcode_desc() to handle
    * opcodes that have been reused for different instructions across hardware
    * generations.
    *
    * The gens field acts as a tag. If it is non-zero, name points to a string
    * containing the instruction mnemonic. If it is zero, the table field is
    * valid and either points to a secondary opcode_desc table with 'size'
    * elements or is NULL and no such instruction exists for the opcode.
    */
   union {
      struct {
         char    *name;
         int      nsrc;
      };
      struct {
         const struct opcode_desc *table;
         unsigned size;
      };
   };
   int      ndst;
   int      gens;
};

const struct opcode_desc *
brw_opcode_desc(const struct gen_device_info *devinfo, enum opcode opcode);

static inline bool
is_3src(const struct gen_device_info *devinfo, enum opcode opcode)
{
   const struct opcode_desc *desc = brw_opcode_desc(devinfo, opcode);
   return desc && desc->nsrc == 3;
}

/** Maximum SEND message length */
#define BRW_MAX_MSG_LENGTH 15

/** First MRF register used by pull loads */
#define FIRST_SPILL_MRF(gen) ((gen) == 6 ? 21 : 13)

/** First MRF register used by spills */
#define FIRST_PULL_LOAD_MRF(gen) ((gen) == 6 ? 16 : 13)

#ifdef __cplusplus
}
#endif

#endif
