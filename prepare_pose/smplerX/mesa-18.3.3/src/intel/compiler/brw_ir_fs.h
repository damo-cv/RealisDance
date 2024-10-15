/* -*- c++ -*- */
/*
 * Copyright © 2010-2015 Intel Corporation
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

#ifndef BRW_IR_FS_H
#define BRW_IR_FS_H

#include "brw_shader.h"

class fs_inst;

class fs_reg : public backend_reg {
public:
   DECLARE_RALLOC_CXX_OPERATORS(fs_reg)

   void init();

   fs_reg();
   fs_reg(struct ::brw_reg reg);
   fs_reg(enum brw_reg_file file, int nr);
   fs_reg(enum brw_reg_file file, int nr, enum brw_reg_type type);

   bool equals(const fs_reg &r) const;
   bool negative_equals(const fs_reg &r) const;
   bool is_contiguous() const;

   /**
    * Return the size in bytes of a single logical component of the
    * register assuming the given execution width.
    */
   unsigned component_size(unsigned width) const;

   /** Register region horizontal stride */
   uint8_t stride;
};

static inline fs_reg
negate(fs_reg reg)
{
   assert(reg.file != IMM);
   reg.negate = !reg.negate;
   return reg;
}

static inline fs_reg
retype(fs_reg reg, enum brw_reg_type type)
{
   reg.type = type;
   return reg;
}

static inline fs_reg
byte_offset(fs_reg reg, unsigned delta)
{
   switch (reg.file) {
   case BAD_FILE:
      break;
   case VGRF:
   case ATTR:
   case UNIFORM:
      reg.offset += delta;
      break;
   case MRF: {
      const unsigned suboffset = reg.offset + delta;
      reg.nr += suboffset / REG_SIZE;
      reg.offset = suboffset % REG_SIZE;
      break;
   }
   case ARF:
   case FIXED_GRF: {
      const unsigned suboffset = reg.subnr + delta;
      reg.nr += suboffset / REG_SIZE;
      reg.subnr = suboffset % REG_SIZE;
      break;
   }
   case IMM:
   default:
      assert(delta == 0);
   }
   return reg;
}

static inline fs_reg
horiz_offset(const fs_reg &reg, unsigned delta)
{
   switch (reg.file) {
   case BAD_FILE:
   case UNIFORM:
   case IMM:
      /* These only have a single component that is implicitly splatted.  A
       * horizontal offset should be a harmless no-op.
       * XXX - Handle vector immediates correctly.
       */
      return reg;
   case VGRF:
   case MRF:
   case ATTR:
      return byte_offset(reg, delta * reg.stride * type_sz(reg.type));
   case ARF:
   case FIXED_GRF:
      if (reg.is_null()) {
         return reg;
      } else {
         const unsigned stride = reg.hstride ? 1 << (reg.hstride - 1) : 0;
         return byte_offset(reg, delta * stride * type_sz(reg.type));
      }
   }
   unreachable("Invalid register file");
}

static inline fs_reg
offset(fs_reg reg, unsigned width, unsigned delta)
{
   switch (reg.file) {
   case BAD_FILE:
      break;
   case ARF:
   case FIXED_GRF:
   case MRF:
   case VGRF:
   case ATTR:
   case UNIFORM:
      return byte_offset(reg, delta * reg.component_size(width));
   case IMM:
      assert(delta == 0);
   }
   return reg;
}

/**
 * Get the scalar channel of \p reg given by \p idx and replicate it to all
 * channels of the result.
 */
static inline fs_reg
component(fs_reg reg, unsigned idx)
{
   reg = horiz_offset(reg, idx);
   reg.stride = 0;
   return reg;
}

/**
 * Return an integer identifying the discrete address space a register is
 * contained in.  A register is by definition fully contained in the single
 * reg_space it belongs to, so two registers with different reg_space ids are
 * guaranteed not to overlap.  Most register files are a single reg_space of
 * its own, only the VGRF file is composed of multiple discrete address
 * spaces, one for each VGRF allocation.
 */
static inline uint32_t
reg_space(const fs_reg &r)
{
   return r.file << 16 | (r.file == VGRF ? r.nr : 0);
}

/**
 * Return the base offset in bytes of a register relative to the start of its
 * reg_space().
 */
static inline unsigned
reg_offset(const fs_reg &r)
{
   return (r.file == VGRF || r.file == IMM ? 0 : r.nr) *
          (r.file == UNIFORM ? 4 : REG_SIZE) + r.offset +
          (r.file == ARF || r.file == FIXED_GRF ? r.subnr : 0);
}

/**
 * Return the amount of padding in bytes left unused between individual
 * components of register \p r due to a (horizontal) stride value greater than
 * one, or zero if components are tightly packed in the register file.
 */
static inline unsigned
reg_padding(const fs_reg &r)
{
   const unsigned stride = ((r.file != ARF && r.file != FIXED_GRF) ? r.stride :
                            r.hstride == 0 ? 0 :
                            1 << (r.hstride - 1));
   return (MAX2(1, stride) - 1) * type_sz(r.type);
}

/**
 * Return whether the register region starting at \p r and spanning \p dr
 * bytes could potentially overlap the register region starting at \p s and
 * spanning \p ds bytes.
 */
static inline bool
regions_overlap(const fs_reg &r, unsigned dr, const fs_reg &s, unsigned ds)
{
   if (r.file == MRF && (r.nr & BRW_MRF_COMPR4)) {
      fs_reg t = r;
      t.nr &= ~BRW_MRF_COMPR4;
      /* COMPR4 regions are translated by the hardware during decompression
       * into two separate half-regions 4 MRFs apart from each other.
       */
      return regions_overlap(t, dr / 2, s, ds) ||
             regions_overlap(byte_offset(t, 4 * REG_SIZE), dr / 2, s, ds);

   } else if (s.file == MRF && (s.nr & BRW_MRF_COMPR4)) {
      return regions_overlap(s, ds, r, dr);

   } else {
      return reg_space(r) == reg_space(s) &&
             !(reg_offset(r) + dr <= reg_offset(s) ||
               reg_offset(s) + ds <= reg_offset(r));
   }
}

/**
 * Check that the register region given by r [r.offset, r.offset + dr[
 * is fully contained inside the register region given by s
 * [s.offset, s.offset + ds[.
 */
static inline bool
region_contained_in(const fs_reg &r, unsigned dr, const fs_reg &s, unsigned ds)
{
   return reg_space(r) == reg_space(s) &&
          reg_offset(r) >= reg_offset(s) &&
          reg_offset(r) + dr <= reg_offset(s) + ds;
}

/**
 * Return whether the given register region is n-periodic, i.e. whether the
 * original region remains invariant after shifting it by \p n scalar
 * channels.
 */
static inline bool
is_periodic(const fs_reg &reg, unsigned n)
{
   if (reg.file == BAD_FILE || reg.is_null()) {
      return true;

   } else if (reg.file == IMM) {
      const unsigned period = (reg.type == BRW_REGISTER_TYPE_UV ||
                               reg.type == BRW_REGISTER_TYPE_V ? 8 :
                               reg.type == BRW_REGISTER_TYPE_VF ? 4 :
                               1);
      return n % period == 0;

   } else if (reg.file == ARF || reg.file == FIXED_GRF) {
      const unsigned period = (reg.hstride == 0 && reg.vstride == 0 ? 1 :
                               reg.vstride == 0 ? 1 << reg.width :
                               ~0);
      return n % period == 0;

   } else {
      return reg.stride == 0;
   }
}

static inline bool
is_uniform(const fs_reg &reg)
{
   return is_periodic(reg, 1);
}

/**
 * Get the specified 8-component quarter of a register.
 * XXX - Maybe come up with a less misleading name for this (e.g. quarter())?
 */
static inline fs_reg
half(const fs_reg &reg, unsigned idx)
{
   assert(idx < 2);
   return horiz_offset(reg, 8 * idx);
}

/**
 * Reinterpret each channel of register \p reg as a vector of values of the
 * given smaller type and take the i-th subcomponent from each.
 */
static inline fs_reg
subscript(fs_reg reg, brw_reg_type type, unsigned i)
{
   assert((i + 1) * type_sz(type) <= type_sz(reg.type));

   if (reg.file == ARF || reg.file == FIXED_GRF) {
      /* The stride is encoded inconsistently for fixed GRF and ARF registers
       * as the log2 of the actual vertical and horizontal strides.
       */
      const int delta = _mesa_logbase2(type_sz(reg.type)) -
                        _mesa_logbase2(type_sz(type));
      reg.hstride += (reg.hstride ? delta : 0);
      reg.vstride += (reg.vstride ? delta : 0);

   } else if (reg.file == IMM) {
      assert(reg.type == type);

   } else {
      reg.stride *= type_sz(reg.type) / type_sz(type);
   }

   return byte_offset(retype(reg, type), i * type_sz(type));
}

static inline fs_reg
horiz_stride(fs_reg reg, unsigned s)
{
   reg.stride *= s;
   return reg;
}

static const fs_reg reg_undef;

class fs_inst : public backend_instruction {
   fs_inst &operator=(const fs_inst &);

   void init(enum opcode opcode, uint8_t exec_width, const fs_reg &dst,
             const fs_reg *src, unsigned sources);

public:
   DECLARE_RALLOC_CXX_OPERATORS(fs_inst)

   fs_inst();
   fs_inst(enum opcode opcode, uint8_t exec_size);
   fs_inst(enum opcode opcode, uint8_t exec_size, const fs_reg &dst);
   fs_inst(enum opcode opcode, uint8_t exec_size, const fs_reg &dst,
           const fs_reg &src0);
   fs_inst(enum opcode opcode, uint8_t exec_size, const fs_reg &dst,
           const fs_reg &src0, const fs_reg &src1);
   fs_inst(enum opcode opcode, uint8_t exec_size, const fs_reg &dst,
           const fs_reg &src0, const fs_reg &src1, const fs_reg &src2);
   fs_inst(enum opcode opcode, uint8_t exec_size, const fs_reg &dst,
           const fs_reg src[], unsigned sources);
   fs_inst(const fs_inst &that);
   ~fs_inst();

   void resize_sources(uint8_t num_sources);

   bool equals(fs_inst *inst) const;
   bool is_send_from_grf() const;
   bool is_partial_write() const;
   bool is_copy_payload(const brw::simple_allocator &grf_alloc) const;
   unsigned components_read(unsigned i) const;
   unsigned size_read(int arg) const;
   bool can_do_source_mods(const struct gen_device_info *devinfo);
   bool can_do_cmod();
   bool can_change_types() const;
   bool has_source_and_destination_hazard() const;

   /**
    * Return the subset of flag registers read by the instruction as a bitset
    * with byte granularity.
    */
   unsigned flags_read(const gen_device_info *devinfo) const;

   /**
    * Return the subset of flag registers updated by the instruction (either
    * partially or fully) as a bitset with byte granularity.
    */
   unsigned flags_written() const;

   fs_reg dst;
   fs_reg *src;

   uint8_t sources; /**< Number of fs_reg sources. */

   bool last_rt:1;
   bool pi_noperspective:1;   /**< Pixel interpolator noperspective flag */
};

/**
 * Make the execution of \p inst dependent on the evaluation of a possibly
 * inverted predicate.
 */
static inline fs_inst *
set_predicate_inv(enum brw_predicate pred, bool inverse,
                  fs_inst *inst)
{
   inst->predicate = pred;
   inst->predicate_inverse = inverse;
   return inst;
}

/**
 * Make the execution of \p inst dependent on the evaluation of a predicate.
 */
static inline fs_inst *
set_predicate(enum brw_predicate pred, fs_inst *inst)
{
   return set_predicate_inv(pred, false, inst);
}

/**
 * Write the result of evaluating the condition given by \p mod to a flag
 * register.
 */
static inline fs_inst *
set_condmod(enum brw_conditional_mod mod, fs_inst *inst)
{
   inst->conditional_mod = mod;
   return inst;
}

/**
 * Clamp the result of \p inst to the saturation range of its destination
 * datatype.
 */
static inline fs_inst *
set_saturate(bool saturate, fs_inst *inst)
{
   inst->saturate = saturate;
   return inst;
}

/**
 * Return the number of dataflow registers written by the instruction (either
 * fully or partially) counted from 'floor(reg_offset(inst->dst) /
 * register_size)'.  The somewhat arbitrary register size unit is 4B for the
 * UNIFORM and IMM files and 32B for all other files.
 */
inline unsigned
regs_written(const fs_inst *inst)
{
   assert(inst->dst.file != UNIFORM && inst->dst.file != IMM);
   return DIV_ROUND_UP(reg_offset(inst->dst) % REG_SIZE +
                       inst->size_written -
                       MIN2(inst->size_written, reg_padding(inst->dst)),
                       REG_SIZE);
}

/**
 * Return the number of dataflow registers read by the instruction (either
 * fully or partially) counted from 'floor(reg_offset(inst->src[i]) /
 * register_size)'.  The somewhat arbitrary register size unit is 4B for the
 * UNIFORM and IMM files and 32B for all other files.
 */
inline unsigned
regs_read(const fs_inst *inst, unsigned i)
{
   const unsigned reg_size =
      inst->src[i].file == UNIFORM || inst->src[i].file == IMM ? 4 : REG_SIZE;
   return DIV_ROUND_UP(reg_offset(inst->src[i]) % reg_size +
                       inst->size_read(i) -
                       MIN2(inst->size_read(i), reg_padding(inst->src[i])),
                       reg_size);
}

static inline enum brw_reg_type
get_exec_type(const fs_inst *inst)
{
   brw_reg_type exec_type = BRW_REGISTER_TYPE_B;

   for (int i = 0; i < inst->sources; i++) {
      if (inst->src[i].file != BAD_FILE) {
         const brw_reg_type t = get_exec_type(inst->src[i].type);
         if (type_sz(t) > type_sz(exec_type))
            exec_type = t;
         else if (type_sz(t) == type_sz(exec_type) &&
                  brw_reg_type_is_floating_point(t))
            exec_type = t;
      }
   }

   if (exec_type == BRW_REGISTER_TYPE_B)
      exec_type = inst->dst.type;

   assert(exec_type != BRW_REGISTER_TYPE_B);

   return exec_type;
}

static inline unsigned
get_exec_type_size(const fs_inst *inst)
{
   return type_sz(get_exec_type(inst));
}

/**
 * Return whether the following regioning restriction applies to the specified
 * instruction.  From the Cherryview PRM Vol 7. "Register Region
 * Restrictions":
 *
 * "When source or destination datatype is 64b or operation is integer DWord
 *  multiply, regioning in Align1 must follow these rules:
 *
 *  1. Source and Destination horizontal stride must be aligned to the same qword.
 *  2. Regioning must ensure Src.Vstride = Src.Width * Src.Hstride.
 *  3. Source and Destination offset must be the same, except the case of
 *     scalar source."
 */
static inline bool
has_dst_aligned_region_restriction(const gen_device_info *devinfo,
                                   const fs_inst *inst)
{
   const brw_reg_type exec_type = get_exec_type(inst);
   const bool is_int_multiply = !brw_reg_type_is_floating_point(exec_type) &&
         (inst->opcode == BRW_OPCODE_MUL || inst->opcode == BRW_OPCODE_MAD);

   if (type_sz(inst->dst.type) > 4 || type_sz(exec_type) > 4 ||
       (type_sz(exec_type) == 4 && is_int_multiply))
      return devinfo->is_cherryview || gen_device_info_is_9lp(devinfo);
   else
      return false;
}

#endif
