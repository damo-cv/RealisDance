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
#include "brw_shader.h"
#include "common/gen_debug.h"

#include "util/ralloc.h"

/* Returns a conditional modifier that negates the condition. */
enum brw_conditional_mod
brw_negate_cmod(uint32_t cmod)
{
   switch (cmod) {
   case BRW_CONDITIONAL_Z:
      return BRW_CONDITIONAL_NZ;
   case BRW_CONDITIONAL_NZ:
      return BRW_CONDITIONAL_Z;
   case BRW_CONDITIONAL_G:
      return BRW_CONDITIONAL_LE;
   case BRW_CONDITIONAL_GE:
      return BRW_CONDITIONAL_L;
   case BRW_CONDITIONAL_L:
      return BRW_CONDITIONAL_GE;
   case BRW_CONDITIONAL_LE:
      return BRW_CONDITIONAL_G;
   default:
      return ~0;
   }
}

/* Returns the corresponding conditional mod for swapping src0 and
 * src1 in e.g. CMP.
 */
enum brw_conditional_mod
brw_swap_cmod(uint32_t cmod)
{
   switch (cmod) {
   case BRW_CONDITIONAL_Z:
   case BRW_CONDITIONAL_NZ:
      return cmod;
   case BRW_CONDITIONAL_G:
      return BRW_CONDITIONAL_L;
   case BRW_CONDITIONAL_GE:
      return BRW_CONDITIONAL_LE;
   case BRW_CONDITIONAL_L:
      return BRW_CONDITIONAL_G;
   case BRW_CONDITIONAL_LE:
      return BRW_CONDITIONAL_GE;
   default:
      return BRW_CONDITIONAL_NONE;
   }
}

/**
 * Get the least significant bit offset of the i+1-th component of immediate
 * type \p type.  For \p i equal to the two's complement of j, return the
 * offset of the j-th component starting from the end of the vector.  For
 * scalar register types return zero.
 */
static unsigned
imm_shift(enum brw_reg_type type, unsigned i)
{
   assert(type != BRW_REGISTER_TYPE_UV && type != BRW_REGISTER_TYPE_V &&
          "Not implemented.");

   if (type == BRW_REGISTER_TYPE_VF)
      return 8 * (i & 3);
   else
      return 0;
}

/**
 * Swizzle an arbitrary immediate \p x of the given type according to the
 * permutation specified as \p swz.
 */
uint32_t
brw_swizzle_immediate(enum brw_reg_type type, uint32_t x, unsigned swz)
{
   if (imm_shift(type, 1)) {
      const unsigned n = 32 / imm_shift(type, 1);
      uint32_t y = 0;

      for (unsigned i = 0; i < n; i++) {
         /* Shift the specified component all the way to the right and left to
          * discard any undesired L/MSBs, then shift it right into component i.
          */
         y |= x >> imm_shift(type, (i & ~3) + BRW_GET_SWZ(swz, i & 3))
                << imm_shift(type, ~0u)
                >> imm_shift(type, ~0u - i);
      }

      return y;
   } else {
      return x;
   }
}

unsigned
brw_get_default_exec_size(struct brw_codegen *p)
{
   return p->current->exec_size;
}

unsigned
brw_get_default_group(struct brw_codegen *p)
{
   return p->current->group;
}

unsigned
brw_get_default_access_mode(struct brw_codegen *p)
{
   return p->current->access_mode;
}

void
brw_set_default_exec_size(struct brw_codegen *p, unsigned value)
{
   p->current->exec_size = value;
}

void brw_set_default_predicate_control( struct brw_codegen *p, unsigned pc )
{
   p->current->predicate = pc;
}

void brw_set_default_predicate_inverse(struct brw_codegen *p, bool predicate_inverse)
{
   p->current->pred_inv = predicate_inverse;
}

void brw_set_default_flag_reg(struct brw_codegen *p, int reg, int subreg)
{
   assert(subreg < 2);
   p->current->flag_subreg = reg * 2 + subreg;
}

void brw_set_default_access_mode( struct brw_codegen *p, unsigned access_mode )
{
   p->current->access_mode = access_mode;
}

void
brw_set_default_compression_control(struct brw_codegen *p,
			    enum brw_compression compression_control)
{
   switch (compression_control) {
   case BRW_COMPRESSION_NONE:
      /* This is the "use the first set of bits of dmask/vmask/arf
       * according to execsize" option.
       */
      p->current->group = 0;
      break;
   case BRW_COMPRESSION_2NDHALF:
      /* For SIMD8, this is "use the second set of 8 bits." */
      p->current->group = 8;
      break;
   case BRW_COMPRESSION_COMPRESSED:
      /* For SIMD16 instruction compression, use the first set of 16 bits
       * since we don't do SIMD32 dispatch.
       */
      p->current->group = 0;
      break;
   default:
      unreachable("not reached");
   }

   if (p->devinfo->gen <= 6) {
      p->current->compressed =
         (compression_control == BRW_COMPRESSION_COMPRESSED);
   }
}

/**
 * Enable or disable instruction compression on the given instruction leaving
 * the currently selected channel enable group untouched.
 */
void
brw_inst_set_compression(const struct gen_device_info *devinfo,
                         brw_inst *inst, bool on)
{
   if (devinfo->gen >= 6) {
      /* No-op, the EU will figure out for us whether the instruction needs to
       * be compressed.
       */
   } else {
      /* The channel group and compression controls are non-orthogonal, there
       * are two possible representations for uncompressed instructions and we
       * may need to preserve the current one to avoid changing the selected
       * channel group inadvertently.
       */
      if (on)
         brw_inst_set_qtr_control(devinfo, inst, BRW_COMPRESSION_COMPRESSED);
      else if (brw_inst_qtr_control(devinfo, inst)
               == BRW_COMPRESSION_COMPRESSED)
         brw_inst_set_qtr_control(devinfo, inst, BRW_COMPRESSION_NONE);
   }
}

void
brw_set_default_compression(struct brw_codegen *p, bool on)
{
   p->current->compressed = on;
}

/**
 * Apply the range of channel enable signals given by
 * [group, group + exec_size) to the instruction passed as argument.
 */
void
brw_inst_set_group(const struct gen_device_info *devinfo,
                   brw_inst *inst, unsigned group)
{
   if (devinfo->gen >= 7) {
      assert(group % 4 == 0 && group < 32);
      brw_inst_set_qtr_control(devinfo, inst, group / 8);
      brw_inst_set_nib_control(devinfo, inst, (group / 4) % 2);

   } else if (devinfo->gen == 6) {
      assert(group % 8 == 0 && group < 32);
      brw_inst_set_qtr_control(devinfo, inst, group / 8);

   } else {
      assert(group % 8 == 0 && group < 16);
      /* The channel group and compression controls are non-orthogonal, there
       * are two possible representations for group zero and we may need to
       * preserve the current one to avoid changing the selected compression
       * enable inadvertently.
       */
      if (group == 8)
         brw_inst_set_qtr_control(devinfo, inst, BRW_COMPRESSION_2NDHALF);
      else if (brw_inst_qtr_control(devinfo, inst) == BRW_COMPRESSION_2NDHALF)
         brw_inst_set_qtr_control(devinfo, inst, BRW_COMPRESSION_NONE);
   }
}

void
brw_set_default_group(struct brw_codegen *p, unsigned group)
{
   p->current->group = group;
}

void brw_set_default_mask_control( struct brw_codegen *p, unsigned value )
{
   p->current->mask_control = value;
}

void brw_set_default_saturate( struct brw_codegen *p, bool enable )
{
   p->current->saturate = enable;
}

void brw_set_default_acc_write_control(struct brw_codegen *p, unsigned value)
{
   p->current->acc_wr_control = value;
}

void brw_push_insn_state( struct brw_codegen *p )
{
   assert(p->current != &p->stack[BRW_EU_MAX_INSN_STACK-1]);
   *(p->current + 1) = *p->current;
   p->current++;
}

void brw_pop_insn_state( struct brw_codegen *p )
{
   assert(p->current != p->stack);
   p->current--;
}


/***********************************************************************
 */
void
brw_init_codegen(const struct gen_device_info *devinfo,
                 struct brw_codegen *p, void *mem_ctx)
{
   memset(p, 0, sizeof(*p));

   p->devinfo = devinfo;
   p->automatic_exec_sizes = true;
   /*
    * Set the initial instruction store array size to 1024, if found that
    * isn't enough, then it will double the store size at brw_next_insn()
    * until out of memory.
    */
   p->store_size = 1024;
   p->store = rzalloc_array(mem_ctx, brw_inst, p->store_size);
   p->nr_insn = 0;
   p->current = p->stack;
   memset(p->current, 0, sizeof(p->current[0]));

   p->mem_ctx = mem_ctx;

   /* Some defaults?
    */
   brw_set_default_exec_size(p, BRW_EXECUTE_8);
   brw_set_default_mask_control(p, BRW_MASK_ENABLE); /* what does this do? */
   brw_set_default_saturate(p, 0);
   brw_set_default_compression_control(p, BRW_COMPRESSION_NONE);

   /* Set up control flow stack */
   p->if_stack_depth = 0;
   p->if_stack_array_size = 16;
   p->if_stack = rzalloc_array(mem_ctx, int, p->if_stack_array_size);

   p->loop_stack_depth = 0;
   p->loop_stack_array_size = 16;
   p->loop_stack = rzalloc_array(mem_ctx, int, p->loop_stack_array_size);
   p->if_depth_in_loop = rzalloc_array(mem_ctx, int, p->loop_stack_array_size);
}


const unsigned *brw_get_program( struct brw_codegen *p,
			       unsigned *sz )
{
   *sz = p->next_insn_offset;
   return (const unsigned *)p->store;
}

void
brw_disassemble(const struct gen_device_info *devinfo,
                const void *assembly, int start, int end, FILE *out)
{
   bool dump_hex = (INTEL_DEBUG & DEBUG_HEX) != 0;

   for (int offset = start; offset < end;) {
      const brw_inst *insn = assembly + offset;
      brw_inst uncompacted;
      bool compacted = brw_inst_cmpt_control(devinfo, insn);
      if (0)
         fprintf(out, "0x%08x: ", offset);

      if (compacted) {
         brw_compact_inst *compacted = (void *)insn;
         if (dump_hex) {
            unsigned char * insn_ptr = ((unsigned char *)&insn[0]);
            const unsigned int blank_spaces = 24;
            for (int i = 0 ; i < 8; i = i + 4) {
               fprintf(out, "%02x %02x %02x %02x ",
                       insn_ptr[i],
                       insn_ptr[i + 1],
                       insn_ptr[i + 2],
                       insn_ptr[i + 3]);
            }
            /* Make compacted instructions hex value output vertically aligned
             * with uncompacted instructions hex value
             */
            fprintf(out, "%*c", blank_spaces, ' ');
         }

         brw_uncompact_instruction(devinfo, &uncompacted, compacted);
         insn = &uncompacted;
         offset += 8;
      } else {
         if (dump_hex) {
            unsigned char * insn_ptr = ((unsigned char *)&insn[0]);
            for (int i = 0 ; i < 16; i = i + 4) {
               fprintf(out, "%02x %02x %02x %02x ",
                       insn_ptr[i],
                       insn_ptr[i + 1],
                       insn_ptr[i + 2],
                       insn_ptr[i + 3]);
            }
         }
         offset += 16;
      }

      brw_disassemble_inst(out, devinfo, insn, compacted);
   }
}

enum gen {
   GEN4  = (1 << 0),
   GEN45 = (1 << 1),
   GEN5  = (1 << 2),
   GEN6  = (1 << 3),
   GEN7  = (1 << 4),
   GEN75 = (1 << 5),
   GEN8  = (1 << 6),
   GEN9  = (1 << 7),
   GEN10 = (1 << 8),
   GEN11 = (1 << 9),
   GEN_ALL = ~0
};

#define GEN_LT(gen) ((gen) - 1)
#define GEN_GE(gen) (~GEN_LT(gen))
#define GEN_LE(gen) (GEN_LT(gen) | (gen))

static const struct opcode_desc opcode_10_descs[] = {
   { .name = "dim",   .nsrc = 1, .ndst = 1, .gens = GEN75 },
   { .name = "smov",  .nsrc = 0, .ndst = 0, .gens = GEN_GE(GEN8) },
};

static const struct opcode_desc opcode_35_descs[] = {
   { .name = "iff",   .nsrc = 0, .ndst = 0, .gens = GEN_LE(GEN5) },
   { .name = "brc",   .nsrc = 0, .ndst = 0, .gens = GEN_GE(GEN7) },
};

static const struct opcode_desc opcode_38_descs[] = {
   { .name = "do",    .nsrc = 0, .ndst = 0, .gens = GEN_LE(GEN5) },
   { .name = "case",  .nsrc = 0, .ndst = 0, .gens = GEN6 },
};

static const struct opcode_desc opcode_44_descs[] = {
   { .name = "msave", .nsrc = 0, .ndst = 0, .gens = GEN_LE(GEN5) },
   { .name = "call",  .nsrc = 0, .ndst = 0, .gens = GEN_GE(GEN6) },
};

static const struct opcode_desc opcode_45_descs[] = {
   { .name = "mrest", .nsrc = 0, .ndst = 0, .gens = GEN_LE(GEN5) },
   { .name = "ret",   .nsrc = 0, .ndst = 0, .gens = GEN_GE(GEN6) },
};

static const struct opcode_desc opcode_46_descs[] = {
   { .name = "push",  .nsrc = 0, .ndst = 0, .gens = GEN_LE(GEN5) },
   { .name = "fork",  .nsrc = 0, .ndst = 0, .gens = GEN6 },
   { .name = "goto",  .nsrc = 0, .ndst = 0, .gens = GEN_GE(GEN8) },
};

static const struct opcode_desc opcode_descs[128] = {
   [BRW_OPCODE_ILLEGAL] = {
      .name = "illegal", .nsrc = 0, .ndst = 0, .gens = GEN_ALL,
   },
   [BRW_OPCODE_MOV] = {
      .name = "mov",     .nsrc = 1, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_SEL] = {
      .name = "sel",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_MOVI] = {
      .name = "movi",    .nsrc = 2, .ndst = 1, .gens = GEN_GE(GEN45),
   },
   [BRW_OPCODE_NOT] = {
      .name = "not",     .nsrc = 1, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_AND] = {
      .name = "and",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_OR] = {
      .name = "or",      .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_XOR] = {
      .name = "xor",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_SHR] = {
      .name = "shr",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_SHL] = {
      .name = "shl",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [10] = {
      .table = opcode_10_descs, .size = ARRAY_SIZE(opcode_10_descs),
   },
   /* Reserved - 11 */
   [BRW_OPCODE_ASR] = {
      .name = "asr",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   /* Reserved - 13-15 */
   [BRW_OPCODE_CMP] = {
      .name = "cmp",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_CMPN] = {
      .name = "cmpn",    .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_CSEL] = {
      .name = "csel",    .nsrc = 3, .ndst = 1, .gens = GEN_GE(GEN8),
   },
   [BRW_OPCODE_F32TO16] = {
      .name = "f32to16", .nsrc = 1, .ndst = 1, .gens = GEN7 | GEN75,
   },
   [BRW_OPCODE_F16TO32] = {
      .name = "f16to32", .nsrc = 1, .ndst = 1, .gens = GEN7 | GEN75,
   },
   /* Reserved - 21-22 */
   [BRW_OPCODE_BFREV] = {
      .name = "bfrev",   .nsrc = 1, .ndst = 1, .gens = GEN_GE(GEN7),
   },
   [BRW_OPCODE_BFE] = {
      .name = "bfe",     .nsrc = 3, .ndst = 1, .gens = GEN_GE(GEN7),
   },
   [BRW_OPCODE_BFI1] = {
      .name = "bfi1",    .nsrc = 2, .ndst = 1, .gens = GEN_GE(GEN7),
   },
   [BRW_OPCODE_BFI2] = {
      .name = "bfi2",    .nsrc = 3, .ndst = 1, .gens = GEN_GE(GEN7),
   },
   /* Reserved - 27-31 */
   [BRW_OPCODE_JMPI] = {
      .name = "jmpi",    .nsrc = 0, .ndst = 0, .gens = GEN_ALL,
   },
   [33] = {
      .name = "brd",     .nsrc = 0, .ndst = 0, .gens = GEN_GE(GEN7),
   },
   [BRW_OPCODE_IF] = {
      .name = "if",      .nsrc = 0, .ndst = 0, .gens = GEN_ALL,
   },
   [35] = {
      .table = opcode_35_descs, .size = ARRAY_SIZE(opcode_35_descs),
   },
   [BRW_OPCODE_ELSE] = {
      .name = "else",    .nsrc = 0, .ndst = 0, .gens = GEN_ALL,
   },
   [BRW_OPCODE_ENDIF] = {
      .name = "endif",   .nsrc = 0, .ndst = 0, .gens = GEN_ALL,
   },
   [38] = {
      .table = opcode_38_descs, .size = ARRAY_SIZE(opcode_38_descs),
   },
   [BRW_OPCODE_WHILE] = {
      .name = "while",   .nsrc = 0, .ndst = 0, .gens = GEN_ALL,
   },
   [BRW_OPCODE_BREAK] = {
      .name = "break",   .nsrc = 0, .ndst = 0, .gens = GEN_ALL,
   },
   [BRW_OPCODE_CONTINUE] = {
      .name = "cont",    .nsrc = 0, .ndst = 0, .gens = GEN_ALL,
   },
   [BRW_OPCODE_HALT] = {
      .name = "halt",    .nsrc = 0, .ndst = 0, .gens = GEN_ALL,
   },
   [43] = {
      .name = "calla",   .nsrc = 0, .ndst = 0, .gens = GEN_GE(GEN75),
   },
   [44] = {
      .table = opcode_44_descs, .size = ARRAY_SIZE(opcode_44_descs),
   },
   [45] = {
      .table = opcode_45_descs, .size = ARRAY_SIZE(opcode_45_descs),
   },
   [46] = {
      .table = opcode_46_descs, .size = ARRAY_SIZE(opcode_46_descs),
   },
   [47] = {
      .name = "pop",     .nsrc = 2, .ndst = 0, .gens = GEN_LE(GEN5),
   },
   [BRW_OPCODE_WAIT] = {
      .name = "wait",    .nsrc = 1, .ndst = 0, .gens = GEN_ALL,
   },
   [BRW_OPCODE_SEND] = {
      .name = "send",    .nsrc = 1, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_SENDC] = {
      .name = "sendc",   .nsrc = 1, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_SENDS] = {
      .name = "sends",   .nsrc = 2, .ndst = 1, .gens = GEN_GE(GEN9),
   },
   [BRW_OPCODE_SENDSC] = {
      .name = "sendsc",  .nsrc = 2, .ndst = 1, .gens = GEN_GE(GEN9),
   },
   /* Reserved 53-55 */
   [BRW_OPCODE_MATH] = {
      .name = "math",    .nsrc = 2, .ndst = 1, .gens = GEN_GE(GEN6),
   },
   /* Reserved 57-63 */
   [BRW_OPCODE_ADD] = {
      .name = "add",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_MUL] = {
      .name = "mul",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_AVG] = {
      .name = "avg",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_FRC] = {
      .name = "frc",     .nsrc = 1, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_RNDU] = {
      .name = "rndu",    .nsrc = 1, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_RNDD] = {
      .name = "rndd",    .nsrc = 1, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_RNDE] = {
      .name = "rnde",    .nsrc = 1, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_RNDZ] = {
      .name = "rndz",    .nsrc = 1, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_MAC] = {
      .name = "mac",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_MACH] = {
      .name = "mach",    .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_LZD] = {
      .name = "lzd",     .nsrc = 1, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_FBH] = {
      .name = "fbh",     .nsrc = 1, .ndst = 1, .gens = GEN_GE(GEN7),
   },
   [BRW_OPCODE_FBL] = {
      .name = "fbl",     .nsrc = 1, .ndst = 1, .gens = GEN_GE(GEN7),
   },
   [BRW_OPCODE_CBIT] = {
      .name = "cbit",    .nsrc = 1, .ndst = 1, .gens = GEN_GE(GEN7),
   },
   [BRW_OPCODE_ADDC] = {
      .name = "addc",    .nsrc = 2, .ndst = 1, .gens = GEN_GE(GEN7),
   },
   [BRW_OPCODE_SUBB] = {
      .name = "subb",    .nsrc = 2, .ndst = 1, .gens = GEN_GE(GEN7),
   },
   [BRW_OPCODE_SAD2] = {
      .name = "sad2",    .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_SADA2] = {
      .name = "sada2",   .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   /* Reserved 82-83 */
   [BRW_OPCODE_DP4] = {
      .name = "dp4",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_DPH] = {
      .name = "dph",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_DP3] = {
      .name = "dp3",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   [BRW_OPCODE_DP2] = {
      .name = "dp2",     .nsrc = 2, .ndst = 1, .gens = GEN_ALL,
   },
   /* Reserved 88 */
   [BRW_OPCODE_LINE] = {
      .name = "line",    .nsrc = 2, .ndst = 1, .gens = GEN_LE(GEN10),
   },
   [BRW_OPCODE_PLN] = {
      .name = "pln",     .nsrc = 2, .ndst = 1, .gens = GEN_GE(GEN45) & GEN_LE(GEN10),
   },
   [BRW_OPCODE_MAD] = {
      .name = "mad",     .nsrc = 3, .ndst = 1, .gens = GEN_GE(GEN6),
   },
   [BRW_OPCODE_LRP] = {
      .name = "lrp",     .nsrc = 3, .ndst = 1, .gens = GEN_GE(GEN6) & GEN_LE(GEN10),
   },
   [93] = {
      .name = "madm",    .nsrc = 3, .ndst = 1, .gens = GEN_GE(GEN8),
   },
   /* Reserved 94-124 */
   [BRW_OPCODE_NENOP] = {
      .name = "nenop",   .nsrc = 0, .ndst = 0, .gens = GEN45,
   },
   [BRW_OPCODE_NOP] = {
      .name = "nop",     .nsrc = 0, .ndst = 0, .gens = GEN_ALL,
   },
};

static enum gen
gen_from_devinfo(const struct gen_device_info *devinfo)
{
   switch (devinfo->gen) {
   case 4: return devinfo->is_g4x ? GEN45 : GEN4;
   case 5: return GEN5;
   case 6: return GEN6;
   case 7: return devinfo->is_haswell ? GEN75 : GEN7;
   case 8: return GEN8;
   case 9: return GEN9;
   case 10: return GEN10;
   case 11: return GEN11;
   default:
      unreachable("not reached");
   }
}

/* Return the matching opcode_desc for the specified opcode number and
 * hardware generation, or NULL if the opcode is not supported by the device.
 */
const struct opcode_desc *
brw_opcode_desc(const struct gen_device_info *devinfo, enum opcode opcode)
{
   if (opcode >= ARRAY_SIZE(opcode_descs))
      return NULL;

   enum gen gen = gen_from_devinfo(devinfo);
   if (opcode_descs[opcode].gens != 0) {
      if ((opcode_descs[opcode].gens & gen) != 0) {
         return &opcode_descs[opcode];
      }
   } else if (opcode_descs[opcode].table != NULL) {
      const struct opcode_desc *table = opcode_descs[opcode].table;
      for (unsigned i = 0; i < opcode_descs[opcode].size; i++) {
         if ((table[i].gens & gen) != 0) {
            return &table[i];
         }
      }
   }
   return NULL;
}
