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

#include "ir-a2xx.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "freedreno_util.h"
#include "instr-a2xx.h"

#define DEBUG_MSG(f, ...)  do { if (0) DBG(f, ##__VA_ARGS__); } while (0)
#define WARN_MSG(f, ...)   DBG("WARN:  "f, ##__VA_ARGS__)
#define ERROR_MSG(f, ...)  DBG("ERROR: "f, ##__VA_ARGS__)

static int instr_emit(struct ir2_instruction *instr, uint32_t *dwords,
		uint32_t idx, struct ir2_shader_info *info);

static uint32_t reg_fetch_src_swiz(struct ir2_src_register *reg, uint32_t n);
static uint32_t reg_fetch_dst_swiz(struct ir2_dst_register *reg);
static uint32_t reg_alu_dst_swiz(struct ir2_dst_register *reg);
static uint32_t reg_alu_src_swiz(struct ir2_src_register *reg);

/* simple allocator to carve allocations out of an up-front allocated heap,
 * so that we can free everything easily in one shot.
 */
static void * ir2_alloc(struct ir2_shader *shader, int sz)
{
	void *ptr = &shader->heap[shader->heap_idx];
	shader->heap_idx += align(sz, 4) / 4;
	return ptr;
}

static char * ir2_strdup(struct ir2_shader *shader, const char *str)
{
	char *ptr = NULL;
	if (str) {
		int len = strlen(str);
		ptr = ir2_alloc(shader, len+1);
		memcpy(ptr, str, len);
		ptr[len] = '\0';
	}
	return ptr;
}

struct ir2_shader * ir2_shader_create(void)
{
	DEBUG_MSG("");
	struct ir2_shader *shader = calloc(1, sizeof(struct ir2_shader));
	shader->max_reg = -1;
	return shader;
}

void ir2_shader_destroy(struct ir2_shader *shader)
{
	DEBUG_MSG("");
	free(shader);
}

/* check if an instruction is a simple MOV
 */
static struct ir2_instruction * simple_mov(struct ir2_instruction *instr,
		bool output)
{
    struct ir2_src_register *src_reg = instr->src_reg;
    struct ir2_dst_register *dst_reg = &instr->dst_reg;
    struct ir2_register *reg;
    unsigned i;

    /* MAXv used for MOV */
    if (instr->instr_type != IR2_ALU_VECTOR ||
		instr->alu_vector.opc != MAXv)
		return NULL;

	/* non identical srcs */
	if (src_reg[0].num != src_reg[1].num)
		return NULL;

	/* flags */
	int flags = IR2_REG_NEGATE | IR2_REG_ABS;
	if (output)
		flags |= IR2_REG_INPUT | IR2_REG_CONST;
	if ((src_reg[0].flags & flags) || (src_reg[1].flags & flags))
		return NULL;

	/* clamping */
	if (instr->alu_vector.clamp)
		return NULL;

	/* swizzling */
    for (i = 0; i < 4; i++) {
		char swiz = (dst_reg->swizzle ? dst_reg->swizzle : "xyzw")[i];
		if (swiz == '_')
			continue;

		if (swiz != (src_reg[0].swizzle ? src_reg[0].swizzle : "xyzw")[i] ||
			swiz != (src_reg[1].swizzle ? src_reg[1].swizzle : "xyzw")[i])
			return NULL;
    }

    if (output)
		reg = &instr->shader->reg[src_reg[0].num];
	else
		reg = &instr->shader->reg[dst_reg->num];

	assert(reg->write_idx >= 0);
    if (reg->write_idx != reg->write_idx2)
		return NULL;

	if (!output)
		return instr;

	instr = instr->shader->instr[reg->write_idx];
	return instr->instr_type != IR2_ALU_VECTOR ? NULL : instr;
}

static int src_to_reg(struct ir2_instruction *instr,
		struct ir2_src_register *reg)
{
	if (reg->flags & IR2_REG_CONST)
		return reg->num;

	return instr->shader->reg[reg->num].reg;
}

static int dst_to_reg(struct ir2_instruction *instr,
		struct ir2_dst_register *reg)
{
	if (reg->flags & IR2_REG_EXPORT)
		return reg->num;

	return instr->shader->reg[reg->num].reg;
}

static bool mask_get(uint32_t *mask, unsigned index)
{
    return !!(mask[index / 32] & 1 << index % 32);
}

static void mask_set(uint32_t *mask, struct ir2_register *reg, int index)
{
	if (reg) {
		unsigned i;
		for (i = 0; i < ARRAY_SIZE(reg->regmask); i++)
			mask[i] |= reg->regmask[i];
	}
	if (index >= 0)
		mask[index / 32] |= 1 << index % 32;
}

static bool sets_pred(struct ir2_instruction *instr)
{
    return instr->instr_type == IR2_ALU_SCALAR &&
		instr->alu_scalar.opc >= PRED_SETEs &&
		instr->alu_scalar.opc <= PRED_SET_RESTOREs;
}



void* ir2_shader_assemble(struct ir2_shader *shader,
		struct ir2_shader_info *info)
{
	/* NOTES
	 * blob compiler seems to always puts PRED_* instrs in a CF by
	 * themselves, and wont combine EQ/NE in the same CF
	 * (not doing this - doesn't seem to make a difference)
	 *
	 * TODO: implement scheduling for combining vector+scalar instructions
	 * -some vector instructions can be replaced by scalar
	 */

	/* first step:
	 * 1. remove "NOP" MOV instructions generated by TGSI for input/output:
	 * 2. track information for register allocation, and to remove
	 * the dead code when some exports are not needed
	 * 3. add additional instructions for a20x hw binning if needed
	 * NOTE: modifies the shader instrs
	 * this step could be done as instructions are added by compiler instead
	 */

	/* mask of exports that must be generated
	 * used to avoid calculating ps exports with hw binning
	*/
	uint64_t export = ~0ull;
	/* bitmask of variables required for exports defined by "export" */
	uint32_t export_mask[REG_MASK/32+1] = {};

	unsigned idx, reg_idx;
	unsigned max_input = 0;
	int export_size = -1;

	for (idx = 0; idx < shader->instr_count; idx++) {
		struct ir2_instruction *instr = shader->instr[idx], *prev;
		struct ir2_dst_register dst_reg = instr->dst_reg;

		if (dst_reg.flags & IR2_REG_EXPORT) {
			if (dst_reg.num < 32)
				export_size++;

			if ((prev = simple_mov(instr, true))) {
				/* copy instruction but keep dst */
				*instr = *prev;
				instr->dst_reg = dst_reg;
			}
		}

		for (reg_idx = 0; reg_idx < instr->src_reg_count; reg_idx++) {
			struct ir2_src_register *src_reg = &instr->src_reg[reg_idx];
			struct ir2_register *reg;
			int num;

			if (src_reg->flags & IR2_REG_CONST)
				continue;

			num = src_reg->num;
			reg = &shader->reg[num];
			reg->read_idx = idx;

			if (src_reg->flags & IR2_REG_INPUT) {
				max_input = MAX2(max_input, num);
			} else {
				/* bypass simple mov used to set src_reg */
				assert(reg->write_idx >= 0);
				prev = shader->instr[reg->write_idx];
				if (simple_mov(prev, false)) {
					*src_reg = prev->src_reg[0];
					/* process same src_reg again */
					reg_idx -= 1;
					continue;
				}
			}

			/* update dependencies */
			uint32_t *mask = (dst_reg.flags & IR2_REG_EXPORT) ?
					export_mask : shader->reg[dst_reg.num].regmask;
			mask_set(mask, reg, num);
			if (sets_pred(instr))
				mask_set(export_mask, reg, num);
		}
	}

	/* second step:
	 * emit instructions (with CFs) + RA
	 */
	instr_cf_t cfs[128], *cf = cfs;
	uint32_t alufetch[3*256], *af = alufetch;

	/* RA is done on write, so inputs must be allocated here */
	for (reg_idx = 0; reg_idx <= max_input; reg_idx++)
		shader->reg[reg_idx].reg = reg_idx;
	info->max_reg = max_input;

	/* CF instr state */
	instr_cf_exec_t exec = { .opc = EXEC };
	instr_cf_alloc_t alloc = { .opc = ALLOC };
	bool need_alloc = 0;
	bool pos_export = 0;

	export_size = MAX2(export_size, 0);

	for (idx = 0; idx < shader->instr_count; idx++) {
		struct ir2_instruction *instr = shader->instr[idx];
		struct ir2_dst_register *dst_reg = &instr->dst_reg;
		unsigned num = dst_reg->num;
		struct ir2_register *reg;

		/* a2xx only has 64 registers, so we can use a single 64-bit mask */
		uint64_t regmask = 0ull;

		/* compute the current regmask */
		for (reg_idx = 0; (int) reg_idx <= shader->max_reg; reg_idx++) {
			reg = &shader->reg[reg_idx];
			if ((int) idx > reg->write_idx && idx < reg->read_idx)
				regmask |= (1ull << reg->reg);
		}

		if (dst_reg->flags & IR2_REG_EXPORT) {
			/* skip if export is not needed */
			if (!(export & (1ull << num)))
				continue;

            /* ALLOC CF:
             * want to alloc all < 32 at once
			 * 32/33 and 62/63 come in pairs
			 * XXX assuming all 3 types are never interleaved
			 */
            if (num < 32) {
				alloc.size = export_size;
				alloc.buffer_select = SQ_PARAMETER_PIXEL;
				need_alloc = export_size >= 0;
				export_size = -1;
			} else if (num == 32 || num == 33) {
				alloc.size = 0;
				alloc.buffer_select = SQ_MEMORY;
				need_alloc = num != 33;
			} else {
				alloc.size = 0;
				alloc.buffer_select = SQ_POSITION;
				need_alloc = !pos_export;
				pos_export = true;
			}

		} else {
			/* skip if dst register not needed to compute exports */
			if (!mask_get(export_mask, num))
				continue;

			/* RA on first write */
			reg = &shader->reg[num];
			if (reg->write_idx == idx) {
				reg->reg = ffsll(~regmask) - 1;
				info->max_reg = MAX2(info->max_reg, reg->reg);
			}
		}

		if (exec.count == 6 || (exec.count && need_alloc)) {
			*cf++ = *(instr_cf_t*) &exec;
			exec.address += exec.count;
			exec.serialize = 0;
			exec.count = 0;
		}

		if (need_alloc) {
			*cf++ = *(instr_cf_t*) &alloc;
			need_alloc = false;
		}

		int ret = instr_emit(instr, af, idx, info); af += 3;
		assert(!ret);

		if (instr->instr_type == IR2_FETCH)
			exec.serialize |= 0x1 << exec.count * 2;
		if (instr->sync)
			exec.serialize |= 0x2 << exec.count * 2;
		 exec.count += 1;
	}


	exec.opc = !export_size ? EXEC : EXEC_END;
	*cf++ = *(instr_cf_t*) &exec;
	exec.address += exec.count;
	exec.serialize = 0;
	exec.count = 0;

	/* GPU will hang without at least one pixel alloc */
	if (!export_size) {
		alloc.size = 0;
		alloc.buffer_select = SQ_PARAMETER_PIXEL;
		*cf++ = *(instr_cf_t*) &alloc;

		exec.opc = EXEC_END;
		*cf++ = *(instr_cf_t*) &exec;
	}

	unsigned num_cfs = cf - cfs;

	/* insert nop to get an even # of CFs */
	if (num_cfs % 2) {
		*cf++ = (instr_cf_t) { .opc = NOP };
		num_cfs++;
	}

	/* offset cf addrs */
	for (idx = 0; idx < num_cfs; idx++) {
        switch (cfs[idx].opc) {
		case EXEC:
		case EXEC_END:
			cfs[idx].exec.address += num_cfs / 2;
			break;
		default:
			break;
		/* XXX  and any other address using cf that gets implemented */
		}
	}

	/* concatenate cfs+alufetchs */
	uint32_t cfdwords = num_cfs / 2 * 3;
	uint32_t alufetchdwords = exec.address * 3;
	info->sizedwords = cfdwords + alufetchdwords;
	uint32_t *dwords = malloc(info->sizedwords * 4);
	assert(dwords);
	memcpy(dwords, cfs, cfdwords * 4);
	memcpy(&dwords[cfdwords], alufetch, alufetchdwords * 4);
	return dwords;
}

struct ir2_instruction * ir2_instr_create(struct ir2_shader *shader,
		int instr_type)
{
	struct ir2_instruction *instr =
			ir2_alloc(shader, sizeof(struct ir2_instruction));
	DEBUG_MSG("%d", instr_type);
	instr->shader = shader;
	instr->idx = shader->instr_count;
	instr->pred = shader->pred;
	instr->instr_type = instr_type;
	shader->instr[shader->instr_count++] = instr;
	return instr;
}


/*
 * FETCH instructions:
 */

static int instr_emit_fetch(struct ir2_instruction *instr,
		uint32_t *dwords, uint32_t idx,
		struct ir2_shader_info *info)
{
	instr_fetch_t *fetch = (instr_fetch_t *)dwords;
	struct ir2_dst_register *dst_reg = &instr->dst_reg;
	struct ir2_src_register *src_reg = &instr->src_reg[0];

	memset(fetch, 0, sizeof(*fetch));

	fetch->opc = instr->fetch.opc;

	if (instr->fetch.opc == VTX_FETCH) {
		instr_fetch_vtx_t *vtx = &fetch->vtx;

		assert(instr->fetch.stride <= 0xff);
		assert(instr->fetch.fmt <= 0x3f);
		assert(instr->fetch.const_idx <= 0x1f);
		assert(instr->fetch.const_idx_sel <= 0x3);

		vtx->src_reg = src_to_reg(instr, src_reg);
		vtx->src_swiz = reg_fetch_src_swiz(src_reg, 1);
		vtx->dst_reg = dst_to_reg(instr, dst_reg);
		vtx->dst_swiz = reg_fetch_dst_swiz(dst_reg);
		vtx->must_be_one = 1;
		vtx->const_index = instr->fetch.const_idx;
		vtx->const_index_sel = instr->fetch.const_idx_sel;
		vtx->format_comp_all = !!instr->fetch.is_signed;
		vtx->num_format_all = !instr->fetch.is_normalized;
		vtx->format = instr->fetch.fmt;
		vtx->stride = instr->fetch.stride;
		vtx->offset = instr->fetch.offset;

		if (instr->pred != IR2_PRED_NONE) {
			vtx->pred_select = 1;
			vtx->pred_condition = (instr->pred == IR2_PRED_EQ) ? 1 : 0;
		}

		/* XXX seems like every FETCH but the first has
		 * this bit set:
		 */
		vtx->reserved3 = (idx > 0) ? 0x1 : 0x0;
		vtx->reserved0 = (idx > 0) ? 0x2 : 0x3;
	} else if (instr->fetch.opc == TEX_FETCH) {
		instr_fetch_tex_t *tex = &fetch->tex;

		assert(instr->fetch.const_idx <= 0x1f);

		tex->src_reg = src_to_reg(instr, src_reg);
		tex->src_swiz = reg_fetch_src_swiz(src_reg, 3);
		tex->dst_reg = dst_to_reg(instr, dst_reg);
		tex->dst_swiz = reg_fetch_dst_swiz(dst_reg);
		tex->const_idx = instr->fetch.const_idx;
		tex->mag_filter = TEX_FILTER_USE_FETCH_CONST;
		tex->min_filter = TEX_FILTER_USE_FETCH_CONST;
		tex->mip_filter = TEX_FILTER_USE_FETCH_CONST;
		tex->aniso_filter = ANISO_FILTER_USE_FETCH_CONST;
		tex->arbitrary_filter = ARBITRARY_FILTER_USE_FETCH_CONST;
		tex->vol_mag_filter = TEX_FILTER_USE_FETCH_CONST;
		tex->vol_min_filter = TEX_FILTER_USE_FETCH_CONST;
		tex->use_comp_lod = 1;
		tex->use_reg_lod = !instr->fetch.is_cube;
		tex->sample_location = SAMPLE_CENTER;
		tex->tx_coord_denorm = instr->fetch.is_rect;

		if (instr->pred != IR2_PRED_NONE) {
			tex->pred_select = 1;
			tex->pred_condition = (instr->pred == IR2_PRED_EQ) ? 1 : 0;
		}

	} else {
		ERROR_MSG("invalid fetch opc: %d\n", instr->fetch.opc);
		return -1;
	}

	return 0;
}

/*
 * ALU instructions:
 */

static int instr_emit_alu(struct ir2_instruction *instr_v,
		struct ir2_instruction *instr_s, uint32_t *dwords,
		struct ir2_shader_info *info)
{
	instr_alu_t *alu = (instr_alu_t *)dwords;
	struct ir2_dst_register *vdst_reg, *sdst_reg;
	struct ir2_src_register *src1_reg, *src2_reg, *src3_reg;
	struct ir2_shader *shader = instr_v ? instr_v->shader : instr_s->shader;
	enum ir2_pred pred = IR2_PRED_NONE;

	memset(alu, 0, sizeof(*alu));

	vdst_reg = NULL;
	sdst_reg = NULL;
	src1_reg = NULL;
	src2_reg = NULL;
	src3_reg = NULL;

	if (instr_v) {
		vdst_reg = &instr_v->dst_reg;
		assert(instr_v->src_reg_count >= 2);
		src1_reg = &instr_v->src_reg[0];
		src2_reg = &instr_v->src_reg[1];
		if (instr_v->src_reg_count > 2)
			src3_reg = &instr_v->src_reg[2];
		pred = instr_v->pred;
	}

	if (instr_s) {
		sdst_reg = &instr_s->dst_reg;
		assert(instr_s->src_reg_count == 1);
		assert(!instr_v || vdst_reg->flags == sdst_reg->flags);
		assert(!instr_v || pred == instr_s->pred);
		if (src3_reg) {
			assert(src3_reg->flags == instr_s->src_reg[0].flags);
			assert(src3_reg->num == instr_s->src_reg[0].num);
			assert(!strcmp(src3_reg->swizzle, instr_s->src_reg[0].swizzle));
		}
		src3_reg = &instr_s->src_reg[0];
		pred = instr_s->pred;
	}

	if (vdst_reg) {
		assert((vdst_reg->flags & ~IR2_REG_EXPORT) == 0);
		assert(!vdst_reg->swizzle || (strlen(vdst_reg->swizzle) == 4));
		alu->vector_opc          = instr_v->alu_vector.opc;
		alu->vector_write_mask   = reg_alu_dst_swiz(vdst_reg);
		alu->vector_dest         = dst_to_reg(instr_v, vdst_reg);
	} else {
		alu->vector_opc          = MAXv;
	}

	if (sdst_reg) {
		alu->scalar_opc          = instr_s->alu_scalar.opc;
		alu->scalar_write_mask   = reg_alu_dst_swiz(sdst_reg);
		alu->scalar_dest         = dst_to_reg(instr_s, sdst_reg);
	} else {
		/* not sure if this is required, but adreno compiler seems
		 * to always set scalar opc to MAXs if it is not used:
		 */
		alu->scalar_opc = MAXs;
	}

	alu->export_data =
		!!((instr_v ? vdst_reg : sdst_reg)->flags & IR2_REG_EXPORT);

	/* export32 has this bit set.. it seems to do more than just set
	 * the base address of the constants used to zero
	 * TODO make this less of a hack
	 */
	if (alu->export_data && alu->vector_dest == 32) {
		assert(!instr_s);
		alu->relative_addr = 1;
	}

	if (src1_reg) {
		if (src1_reg->flags & IR2_REG_CONST) {
			assert(!(src1_reg->flags & IR2_REG_ABS));
			alu->src1_reg_const  = src1_reg->num;
		} else {
			alu->src1_reg        = shader->reg[src1_reg->num].reg;
			alu->src1_reg_abs    = !!(src1_reg->flags & IR2_REG_ABS);
		}
		alu->src1_swiz           = reg_alu_src_swiz(src1_reg);
		alu->src1_reg_negate     = !!(src1_reg->flags & IR2_REG_NEGATE);
		alu->src1_sel            = !(src1_reg->flags & IR2_REG_CONST);
    }  else {
		alu->src1_sel = 1;
	}

    if (src2_reg) {
		if (src2_reg->flags & IR2_REG_CONST) {
			assert(!(src2_reg->flags & IR2_REG_ABS));
			alu->src2_reg_const  = src2_reg->num;
		} else {
			alu->src2_reg        = shader->reg[src2_reg->num].reg;
			alu->src2_reg_abs    = !!(src2_reg->flags & IR2_REG_ABS);
		}
		alu->src2_swiz           = reg_alu_src_swiz(src2_reg);
		alu->src2_reg_negate     = !!(src2_reg->flags & IR2_REG_NEGATE);
		alu->src2_sel            = !(src2_reg->flags & IR2_REG_CONST);
    } else {
		alu->src2_sel = 1;
    }

    if (src3_reg) {
		if (src3_reg->flags & IR2_REG_CONST) {
			assert(!(src3_reg->flags & IR2_REG_ABS));
			alu->src3_reg_const  = src3_reg->num;
		} else {
			alu->src3_reg        = shader->reg[src3_reg->num].reg;
			alu->src3_reg_abs    = !!(src3_reg->flags & IR2_REG_ABS);
		}
		alu->src3_swiz           = reg_alu_src_swiz(src3_reg);
		alu->src3_reg_negate     = !!(src3_reg->flags & IR2_REG_NEGATE);
		alu->src3_sel            = !(src3_reg->flags & IR2_REG_CONST);
	} else {
		/* not sure if this is required, but adreno compiler seems
		 * to always set register bank for 3rd src if unused:
		 */
		alu->src3_sel = 1;
	}

	alu->vector_clamp = instr_v ? instr_v->alu_vector.clamp : 0;
	alu->scalar_clamp = instr_s ? instr_s->alu_scalar.clamp : 0;

	if (pred != IR2_PRED_NONE)
		alu->pred_select = (pred == IR2_PRED_EQ) ? 3 : 2;

	return 0;
}

static int instr_emit(struct ir2_instruction *instr, uint32_t *dwords,
		uint32_t idx, struct ir2_shader_info *info)
{
	switch (instr->instr_type) {
	case IR2_FETCH: return instr_emit_fetch(instr, dwords, idx, info);
	case IR2_ALU_VECTOR: return instr_emit_alu(instr, NULL, dwords, info);
	case IR2_ALU_SCALAR: return instr_emit_alu(NULL, instr, dwords, info);
	}
	return -1;
}

struct ir2_dst_register * ir2_dst_create(struct ir2_instruction *instr,
		int num, const char *swizzle, int flags)
{
	if (!(flags & IR2_REG_EXPORT)) {
		struct ir2_register *reg = &instr->shader->reg[num];

		unsigned i;
		for (i = instr->shader->max_reg + 1; i <= num; i++)
			instr->shader->reg[i].write_idx = -1;
		instr->shader->max_reg = i - 1;

		if (reg->write_idx < 0)
            reg->write_idx = instr->idx;
		reg->write_idx2 = instr->idx;
	}

	struct ir2_dst_register *reg = &instr->dst_reg;
	reg->flags = flags;
	reg->num = num;
	reg->swizzle = ir2_strdup(instr->shader, swizzle);
	return reg;
}

struct ir2_src_register * ir2_reg_create(struct ir2_instruction *instr,
		int num, const char *swizzle, int flags)
{
	assert(instr->src_reg_count + 1 <= ARRAY_SIZE(instr->src_reg));
	if (!(flags & IR2_REG_CONST)) {
		struct ir2_register *reg = &instr->shader->reg[num];

		reg->read_idx = instr->idx;

		unsigned i;
		for (i = instr->shader->max_reg + 1; i <= num; i++)
			instr->shader->reg[i].write_idx = -1;
		instr->shader->max_reg = i - 1;
	}

	struct ir2_src_register *reg = &instr->src_reg[instr->src_reg_count++];
	reg->flags = flags;
	reg->num = num;
	reg->swizzle = ir2_strdup(instr->shader, swizzle);
	return reg;
}

static uint32_t reg_fetch_src_swiz(struct ir2_src_register *reg, uint32_t n)
{
	uint32_t swiz = 0;
	int i;

	assert((reg->flags & ~IR2_REG_INPUT) == 0);
	assert(reg->swizzle);

	DEBUG_MSG("fetch src R%d.%s", reg->num, reg->swizzle);

	for (i = n-1; i >= 0; i--) {
		swiz <<= 2;
		switch (reg->swizzle[i]) {
		default:
			ERROR_MSG("invalid fetch src swizzle: %s", reg->swizzle);
		case 'x': swiz |= 0x0; break;
		case 'y': swiz |= 0x1; break;
		case 'z': swiz |= 0x2; break;
		case 'w': swiz |= 0x3; break;
		}
	}

	return swiz;
}

static uint32_t reg_fetch_dst_swiz(struct ir2_dst_register *reg)
{
	uint32_t swiz = 0;
	int i;

	assert(reg->flags == 0);
	assert(!reg->swizzle || (strlen(reg->swizzle) == 4));

	DEBUG_MSG("fetch dst R%d.%s", reg->num, reg->swizzle);

	if (reg->swizzle) {
		for (i = 3; i >= 0; i--) {
			swiz <<= 3;
			switch (reg->swizzle[i]) {
			default:
				ERROR_MSG("invalid dst swizzle: %s", reg->swizzle);
			case 'x': swiz |= 0x0; break;
			case 'y': swiz |= 0x1; break;
			case 'z': swiz |= 0x2; break;
			case 'w': swiz |= 0x3; break;
			case '0': swiz |= 0x4; break;
			case '1': swiz |= 0x5; break;
			case '_': swiz |= 0x7; break;
			}
		}
	} else {
		swiz = 0x688;
	}

	return swiz;
}

/* actually, a write-mask */
static uint32_t reg_alu_dst_swiz(struct ir2_dst_register *reg)
{
	uint32_t swiz = 0;
	int i;

	assert((reg->flags & ~IR2_REG_EXPORT) == 0);
	assert(!reg->swizzle || (strlen(reg->swizzle) == 4));

	DEBUG_MSG("alu dst R%d.%s", reg->num, reg->swizzle);

	if (reg->swizzle) {
		for (i = 3; i >= 0; i--) {
			swiz <<= 1;
			if (reg->swizzle[i] == "xyzw"[i]) {
				swiz |= 0x1;
			} else if (reg->swizzle[i] != '_') {
				ERROR_MSG("invalid dst swizzle: %s", reg->swizzle);
				break;
			}
		}
	} else {
		swiz = 0xf;
	}

	return swiz;
}

static uint32_t reg_alu_src_swiz(struct ir2_src_register *reg)
{
	uint32_t swiz = 0;
	int i;

	assert(!reg->swizzle || (strlen(reg->swizzle) == 4));

	DEBUG_MSG("vector src R%d.%s", reg->num, reg->swizzle);

	if (reg->swizzle) {
		for (i = 3; i >= 0; i--) {
			swiz <<= 2;
			switch (reg->swizzle[i]) {
			default:
				ERROR_MSG("invalid vector src swizzle: %s", reg->swizzle);
			case 'x': swiz |= (0x0 - i) & 0x3; break;
			case 'y': swiz |= (0x1 - i) & 0x3; break;
			case 'z': swiz |= (0x2 - i) & 0x3; break;
			case 'w': swiz |= (0x3 - i) & 0x3; break;
			}
		}
	} else {
		swiz = 0x0;
	}

	return swiz;
}
