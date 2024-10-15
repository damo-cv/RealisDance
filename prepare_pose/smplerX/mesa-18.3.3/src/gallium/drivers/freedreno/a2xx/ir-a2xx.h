/*
 * Copyright (c) 2012 Rob Clark <robdclark@gmail.com>
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef IR2_H_
#define IR2_H_

#include <stdint.h>
#include <stdbool.h>

#include "instr-a2xx.h"

/* low level intermediate representation of an adreno a2xx shader program */

struct ir2_shader;

#define REG_MASK 0xff

struct ir2_shader_info {
	uint16_t sizedwords;
	int8_t   max_reg;   /* highest GPR # used by shader */
};

struct ir2_register {
	int16_t write_idx, write_idx2, read_idx, reg;
	/* bitmask of variables on which this one depends
	 * XXX: use bitmask util?
	 */
	uint32_t regmask[REG_MASK/32+1];
};

struct ir2_src_register {
	enum {
		IR2_REG_INPUT  = 0x1,
		IR2_REG_CONST  = 0x2,
		IR2_REG_NEGATE = 0x4,
		IR2_REG_ABS    = 0x8,
	} flags;
	int num;
	char *swizzle;
};

struct ir2_dst_register {
	enum {
		IR2_REG_EXPORT = 0x1,
	} flags;
	int num;
	char *swizzle;
};

enum ir2_pred {
	IR2_PRED_NONE = 0,
	IR2_PRED_EQ = 1,
	IR2_PRED_NE = 2,
};

struct ir2_instruction {
	struct ir2_shader *shader;
	unsigned idx;
	enum {
		IR2_FETCH,
		IR2_ALU_VECTOR,
		IR2_ALU_SCALAR,
	} instr_type;
	enum ir2_pred pred;
	int sync;
	unsigned src_reg_count;
	struct ir2_dst_register dst_reg;
	struct ir2_src_register src_reg[3];
	union {
		/* FETCH specific: */
		struct {
			instr_fetch_opc_t opc;
			unsigned const_idx;
			/* texture fetch specific: */
			bool is_cube : 1;
			bool is_rect : 1;
			/* vertex fetch specific: */
			unsigned const_idx_sel;
			enum a2xx_sq_surfaceformat fmt;
			bool is_signed : 1;
			bool is_normalized : 1;
			uint32_t stride;
			uint32_t offset;
		} fetch;
		/* ALU-Vector specific: */
		struct {
			instr_vector_opc_t opc;
			bool clamp;
		} alu_vector;
		/* ALU-Scalar specific: */
		struct {
			instr_scalar_opc_t opc;
			bool clamp;
		} alu_scalar;
	};
};

struct ir2_shader {
	unsigned instr_count;
	int max_reg;
	struct ir2_register reg[REG_MASK+1];

	struct ir2_instruction *instr[0x200];
	uint32_t heap[100 * 4096];
	unsigned heap_idx;

	enum ir2_pred pred;  /* pred inherited by newly created instrs */
};

struct ir2_shader * ir2_shader_create(void);
void ir2_shader_destroy(struct ir2_shader *shader);
void * ir2_shader_assemble(struct ir2_shader *shader,
		struct ir2_shader_info *info);

struct ir2_instruction * ir2_instr_create(struct ir2_shader *shader,
		int instr_type);

struct ir2_dst_register * ir2_dst_create(struct ir2_instruction *instr,
		int num, const char *swizzle, int flags);
struct ir2_src_register * ir2_reg_create(struct ir2_instruction *instr,
		int num, const char *swizzle, int flags);

/* some helper fxns: */

static inline struct ir2_instruction *
ir2_instr_create_alu_v(struct ir2_shader *shader, instr_vector_opc_t vop)
{
	struct ir2_instruction *instr = ir2_instr_create(shader, IR2_ALU_VECTOR);
	if (!instr)
		return instr;
	instr->alu_vector.opc = vop;
	return instr;
}

static inline struct ir2_instruction *
ir2_instr_create_alu_s(struct ir2_shader *shader, instr_scalar_opc_t sop)
{
	struct ir2_instruction *instr = ir2_instr_create(shader, IR2_ALU_SCALAR);
	if (!instr)
		return instr;
	instr->alu_scalar.opc = sop;
	return instr;
}

static inline struct ir2_instruction *
ir2_instr_create_vtx_fetch(struct ir2_shader *shader, int ci, int cis,
		enum a2xx_sq_surfaceformat fmt, bool is_signed, int stride)
{
	struct ir2_instruction *instr = ir2_instr_create(shader, IR2_FETCH);
	instr->fetch.opc = VTX_FETCH;
	instr->fetch.const_idx = ci;
	instr->fetch.const_idx_sel = cis;
	instr->fetch.fmt = fmt;
	instr->fetch.is_signed = is_signed;
	instr->fetch.stride = stride;
	return instr;
}
static inline struct ir2_instruction *
ir2_instr_create_tex_fetch(struct ir2_shader *shader, int ci)
{
	struct ir2_instruction *instr = ir2_instr_create(shader, IR2_FETCH);
	instr->fetch.opc = TEX_FETCH;
	instr->fetch.const_idx = ci;
	return instr;
}


#endif /* IR2_H_ */
