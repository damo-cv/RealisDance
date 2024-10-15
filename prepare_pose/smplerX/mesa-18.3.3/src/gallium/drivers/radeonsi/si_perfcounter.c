/*
 * Copyright 2015 Advanced Micro Devices, Inc.
 * All Rights Reserved.
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

#include "si_build_pm4.h"
#include "si_query.h"
#include "util/u_memory.h"


enum si_pc_reg_layout {
	/* All secondary selector dwords follow as one block after the primary
	 * selector dwords for the counters that have secondary selectors.
	 */
	SI_PC_MULTI_BLOCK = 0,

	/* Each secondary selector dword follows immediately afters the
	 * corresponding primary.
	 */
	SI_PC_MULTI_ALTERNATE = 1,

	/* All secondary selector dwords follow as one block after all primary
	 * selector dwords.
	 */
	SI_PC_MULTI_TAIL = 2,

	/* Free-form arrangement of selector registers. */
	SI_PC_MULTI_CUSTOM = 3,

	SI_PC_MULTI_MASK = 3,

	/* Registers are laid out in decreasing rather than increasing order. */
	SI_PC_REG_REVERSE = 4,

	SI_PC_FAKE = 8,
};

struct si_pc_block_base {
	const char *name;
	unsigned num_counters;
	unsigned flags;

	unsigned select_or;
	unsigned select0;
	unsigned counter0_lo;
	unsigned *select;
	unsigned *counters;
	unsigned num_multi;
	unsigned num_prelude;
	unsigned layout;
};

struct si_pc_block {
	struct si_pc_block_base *b;
	unsigned selectors;
	unsigned instances;
};

/* The order is chosen to be compatible with GPUPerfStudio's hardcoding of
 * performance counter group IDs.
 */
static const char * const si_pc_shader_type_suffixes[] = {
	"", "_ES", "_GS", "_VS", "_PS", "_LS", "_HS", "_CS"
};

static const unsigned si_pc_shader_type_bits[] = {
	0x7f,
	S_036780_ES_EN(1),
	S_036780_GS_EN(1),
	S_036780_VS_EN(1),
	S_036780_PS_EN(1),
	S_036780_LS_EN(1),
	S_036780_HS_EN(1),
	S_036780_CS_EN(1),
};

static struct si_pc_block_base cik_CB = {
	.name = "CB",
	.num_counters = 4,
	.flags = SI_PC_BLOCK_SE | SI_PC_BLOCK_INSTANCE_GROUPS,

	.select0 = R_037000_CB_PERFCOUNTER_FILTER,
	.counter0_lo = R_035018_CB_PERFCOUNTER0_LO,
	.num_multi = 1,
	.num_prelude = 1,
	.layout = SI_PC_MULTI_ALTERNATE,
};

static unsigned cik_CPC_select[] = {
	R_036024_CPC_PERFCOUNTER0_SELECT,
	R_036010_CPC_PERFCOUNTER0_SELECT1,
	R_03600C_CPC_PERFCOUNTER1_SELECT,
};
static struct si_pc_block_base cik_CPC = {
	.name = "CPC",
	.num_counters = 2,

	.select = cik_CPC_select,
	.counter0_lo = R_034018_CPC_PERFCOUNTER0_LO,
	.num_multi = 1,
	.layout = SI_PC_MULTI_CUSTOM | SI_PC_REG_REVERSE,
};

static struct si_pc_block_base cik_CPF = {
	.name = "CPF",
	.num_counters = 2,

	.select0 = R_03601C_CPF_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034028_CPF_PERFCOUNTER0_LO,
	.num_multi = 1,
	.layout = SI_PC_MULTI_ALTERNATE | SI_PC_REG_REVERSE,
};

static struct si_pc_block_base cik_CPG = {
	.name = "CPG",
	.num_counters = 2,

	.select0 = R_036008_CPG_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034008_CPG_PERFCOUNTER0_LO,
	.num_multi = 1,
	.layout = SI_PC_MULTI_ALTERNATE | SI_PC_REG_REVERSE,
};

static struct si_pc_block_base cik_DB = {
	.name = "DB",
	.num_counters = 4,
	.flags = SI_PC_BLOCK_SE | SI_PC_BLOCK_INSTANCE_GROUPS,

	.select0 = R_037100_DB_PERFCOUNTER0_SELECT,
	.counter0_lo = R_035100_DB_PERFCOUNTER0_LO,
	.num_multi = 3, // really only 2, but there's a gap between registers
	.layout = SI_PC_MULTI_ALTERNATE,
};

static struct si_pc_block_base cik_GDS = {
	.name = "GDS",
	.num_counters = 4,

	.select0 = R_036A00_GDS_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034A00_GDS_PERFCOUNTER0_LO,
	.num_multi = 1,
	.layout = SI_PC_MULTI_TAIL,
};

static unsigned cik_GRBM_counters[] = {
	R_034100_GRBM_PERFCOUNTER0_LO,
	R_03410C_GRBM_PERFCOUNTER1_LO,
};
static struct si_pc_block_base cik_GRBM = {
	.name = "GRBM",
	.num_counters = 2,

	.select0 = R_036100_GRBM_PERFCOUNTER0_SELECT,
	.counters = cik_GRBM_counters,
};

static struct si_pc_block_base cik_GRBMSE = {
	.name = "GRBMSE",
	.num_counters = 4,

	.select0 = R_036108_GRBM_SE0_PERFCOUNTER_SELECT,
	.counter0_lo = R_034114_GRBM_SE0_PERFCOUNTER_LO,
};

static struct si_pc_block_base cik_IA = {
	.name = "IA",
	.num_counters = 4,

	.select0 = R_036210_IA_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034220_IA_PERFCOUNTER0_LO,
	.num_multi = 1,
	.layout = SI_PC_MULTI_TAIL,
};

static struct si_pc_block_base cik_PA_SC = {
	.name = "PA_SC",
	.num_counters = 8,
	.flags = SI_PC_BLOCK_SE,

	.select0 = R_036500_PA_SC_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034500_PA_SC_PERFCOUNTER0_LO,
	.num_multi = 1,
	.layout = SI_PC_MULTI_ALTERNATE,
};

/* According to docs, PA_SU counters are only 48 bits wide. */
static struct si_pc_block_base cik_PA_SU = {
	.name = "PA_SU",
	.num_counters = 4,
	.flags = SI_PC_BLOCK_SE,

	.select0 = R_036400_PA_SU_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034400_PA_SU_PERFCOUNTER0_LO,
	.num_multi = 2,
	.layout = SI_PC_MULTI_ALTERNATE,
};

static struct si_pc_block_base cik_SPI = {
	.name = "SPI",
	.num_counters = 6,
	.flags = SI_PC_BLOCK_SE,

	.select0 = R_036600_SPI_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034604_SPI_PERFCOUNTER0_LO,
	.num_multi = 4,
	.layout = SI_PC_MULTI_BLOCK,
};

static struct si_pc_block_base cik_SQ = {
	.name = "SQ",
	.num_counters = 16,
	.flags = SI_PC_BLOCK_SE | SI_PC_BLOCK_SHADER,

	.select0 = R_036700_SQ_PERFCOUNTER0_SELECT,
	.select_or = S_036700_SQC_BANK_MASK(15) |
			S_036700_SQC_CLIENT_MASK(15) |
			S_036700_SIMD_MASK(15),
	.counter0_lo = R_034700_SQ_PERFCOUNTER0_LO,
};

static struct si_pc_block_base cik_SX = {
	.name = "SX",
	.num_counters = 4,
	.flags = SI_PC_BLOCK_SE,

	.select0 = R_036900_SX_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034900_SX_PERFCOUNTER0_LO,
	.num_multi = 2,
	.layout = SI_PC_MULTI_TAIL,
};

static struct si_pc_block_base cik_TA = {
	.name = "TA",
	.num_counters = 2,
	.flags = SI_PC_BLOCK_SE | SI_PC_BLOCK_INSTANCE_GROUPS | SI_PC_BLOCK_SHADER_WINDOWED,

	.select0 = R_036B00_TA_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034B00_TA_PERFCOUNTER0_LO,
	.num_multi = 1,
	.layout = SI_PC_MULTI_ALTERNATE,
};

static struct si_pc_block_base cik_TD = {
	.name = "TD",
	.num_counters = 2,
	.flags = SI_PC_BLOCK_SE | SI_PC_BLOCK_INSTANCE_GROUPS | SI_PC_BLOCK_SHADER_WINDOWED,

	.select0 = R_036C00_TD_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034C00_TD_PERFCOUNTER0_LO,
	.num_multi = 1,
	.layout = SI_PC_MULTI_ALTERNATE,
};

static struct si_pc_block_base cik_TCA = {
	.name = "TCA",
	.num_counters = 4,
	.flags = SI_PC_BLOCK_INSTANCE_GROUPS,

	.select0 = R_036E40_TCA_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034E40_TCA_PERFCOUNTER0_LO,
	.num_multi = 2,
	.layout = SI_PC_MULTI_ALTERNATE,
};

static struct si_pc_block_base cik_TCC = {
	.name = "TCC",
	.num_counters = 4,
	.flags = SI_PC_BLOCK_INSTANCE_GROUPS,

	.select0 = R_036E00_TCC_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034E00_TCC_PERFCOUNTER0_LO,
	.num_multi = 2,
	.layout = SI_PC_MULTI_ALTERNATE,
};

static struct si_pc_block_base cik_TCP = {
	.name = "TCP",
	.num_counters = 4,
	.flags = SI_PC_BLOCK_SE | SI_PC_BLOCK_INSTANCE_GROUPS | SI_PC_BLOCK_SHADER_WINDOWED,

	.select0 = R_036D00_TCP_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034D00_TCP_PERFCOUNTER0_LO,
	.num_multi = 2,
	.layout = SI_PC_MULTI_ALTERNATE,
};

static struct si_pc_block_base cik_VGT = {
	.name = "VGT",
	.num_counters = 4,
	.flags = SI_PC_BLOCK_SE,

	.select0 = R_036230_VGT_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034240_VGT_PERFCOUNTER0_LO,
	.num_multi = 1,
	.layout = SI_PC_MULTI_TAIL,
};

static struct si_pc_block_base cik_WD = {
	.name = "WD",
	.num_counters = 4,

	.select0 = R_036200_WD_PERFCOUNTER0_SELECT,
	.counter0_lo = R_034200_WD_PERFCOUNTER0_LO,
};

static struct si_pc_block_base cik_MC = {
	.name = "MC",
	.num_counters = 4,

	.layout = SI_PC_FAKE,
};

static struct si_pc_block_base cik_SRBM = {
	.name = "SRBM",
	.num_counters = 2,

	.layout = SI_PC_FAKE,
};

/* Both the number of instances and selectors varies between chips of the same
 * class. We only differentiate by class here and simply expose the maximum
 * number over all chips in a class.
 *
 * Unfortunately, GPUPerfStudio uses the order of performance counter groups
 * blindly once it believes it has identified the hardware, so the order of
 * blocks here matters.
 */
static struct si_pc_block groups_CIK[] = {
	{ &cik_CB, 226},
	{ &cik_CPF, 17 },
	{ &cik_DB, 257},
	{ &cik_GRBM, 34 },
	{ &cik_GRBMSE, 15 },
	{ &cik_PA_SU, 153 },
	{ &cik_PA_SC, 395 },
	{ &cik_SPI, 186 },
	{ &cik_SQ, 252 },
	{ &cik_SX, 32 },
	{ &cik_TA, 111, 11 },
	{ &cik_TCA, 39, 2 },
	{ &cik_TCC, 160},
	{ &cik_TD, 55, 11 },
	{ &cik_TCP, 154, 11 },
	{ &cik_GDS, 121 },
	{ &cik_VGT, 140 },
	{ &cik_IA, 22 },
	{ &cik_MC, 22 },
	{ &cik_SRBM, 19 },
	{ &cik_WD, 22 },
	{ &cik_CPG, 46 },
	{ &cik_CPC, 22 },

};

static struct si_pc_block groups_VI[] = {
	{ &cik_CB, 405},
	{ &cik_CPF, 19 },
	{ &cik_DB, 257},
	{ &cik_GRBM, 34 },
	{ &cik_GRBMSE, 15 },
	{ &cik_PA_SU, 154 },
	{ &cik_PA_SC, 397 },
	{ &cik_SPI, 197 },
	{ &cik_SQ, 273 },
	{ &cik_SX, 34 },
	{ &cik_TA, 119, 16 },
	{ &cik_TCA, 35, 2 },
	{ &cik_TCC, 192},
	{ &cik_TD, 55, 16 },
	{ &cik_TCP, 180, 16 },
	{ &cik_GDS, 121 },
	{ &cik_VGT, 147 },
	{ &cik_IA, 24 },
	{ &cik_MC, 22 },
	{ &cik_SRBM, 27 },
	{ &cik_WD, 37 },
	{ &cik_CPG, 48 },
	{ &cik_CPC, 24 },

};

static struct si_pc_block groups_gfx9[] = {
	{ &cik_CB, 438},
	{ &cik_CPF, 32 },
	{ &cik_DB, 328},
	{ &cik_GRBM, 38 },
	{ &cik_GRBMSE, 16 },
	{ &cik_PA_SU, 292 },
	{ &cik_PA_SC, 491 },
	{ &cik_SPI, 196 },
	{ &cik_SQ, 374 },
	{ &cik_SX, 208 },
	{ &cik_TA, 119, 16 },
	{ &cik_TCA, 35, 2 },
	{ &cik_TCC, 256},
	{ &cik_TD, 57, 16 },
	{ &cik_TCP, 85, 16 },
	{ &cik_GDS, 121 },
	{ &cik_VGT, 148 },
	{ &cik_IA, 32 },
	{ &cik_WD, 58 },
	{ &cik_CPG, 59 },
	{ &cik_CPC, 35 },
};

static void si_pc_emit_instance(struct si_context *sctx,
				int se, int instance)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	unsigned value = S_030800_SH_BROADCAST_WRITES(1);

	if (se >= 0) {
		value |= S_030800_SE_INDEX(se);
	} else {
		value |= S_030800_SE_BROADCAST_WRITES(1);
	}

	if (instance >= 0) {
		value |= S_030800_INSTANCE_INDEX(instance);
	} else {
		value |= S_030800_INSTANCE_BROADCAST_WRITES(1);
	}

	radeon_set_uconfig_reg(cs, R_030800_GRBM_GFX_INDEX, value);
}

static void si_pc_emit_shaders(struct si_context *sctx,
			       unsigned shaders)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;

	radeon_set_uconfig_reg_seq(cs, R_036780_SQ_PERFCOUNTER_CTRL, 2);
	radeon_emit(cs, shaders & 0x7f);
	radeon_emit(cs, 0xffffffff);
}

static void si_pc_emit_select(struct si_context *sctx,
		        struct si_perfcounter_block *group,
		        unsigned count, unsigned *selectors)
{
	struct si_pc_block *sigroup = (struct si_pc_block *)group->data;
	struct si_pc_block_base *regs = sigroup->b;
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	unsigned idx;
	unsigned layout_multi = regs->layout & SI_PC_MULTI_MASK;
	unsigned dw;

	assert(count <= regs->num_counters);

	if (regs->layout & SI_PC_FAKE)
		return;

	if (layout_multi == SI_PC_MULTI_BLOCK) {
		assert(!(regs->layout & SI_PC_REG_REVERSE));

		dw = count + regs->num_prelude;
		if (count >= regs->num_multi)
			dw += regs->num_multi;
		radeon_set_uconfig_reg_seq(cs, regs->select0, dw);
		for (idx = 0; idx < regs->num_prelude; ++idx)
			radeon_emit(cs, 0);
		for (idx = 0; idx < MIN2(count, regs->num_multi); ++idx)
			radeon_emit(cs, selectors[idx] | regs->select_or);

		if (count < regs->num_multi) {
			unsigned select1 =
				regs->select0 + 4 * regs->num_multi;
			radeon_set_uconfig_reg_seq(cs, select1, count);
		}

		for (idx = 0; idx < MIN2(count, regs->num_multi); ++idx)
			radeon_emit(cs, 0);

		if (count > regs->num_multi) {
			for (idx = regs->num_multi; idx < count; ++idx)
				radeon_emit(cs, selectors[idx] | regs->select_or);
		}
	} else if (layout_multi == SI_PC_MULTI_TAIL) {
		unsigned select1, select1_count;

		assert(!(regs->layout & SI_PC_REG_REVERSE));

		radeon_set_uconfig_reg_seq(cs, regs->select0, count + regs->num_prelude);
		for (idx = 0; idx < regs->num_prelude; ++idx)
			radeon_emit(cs, 0);
		for (idx = 0; idx < count; ++idx)
			radeon_emit(cs, selectors[idx] | regs->select_or);

		select1 = regs->select0 + 4 * regs->num_counters;
		select1_count = MIN2(count, regs->num_multi);
		radeon_set_uconfig_reg_seq(cs, select1, select1_count);
		for (idx = 0; idx < select1_count; ++idx)
			radeon_emit(cs, 0);
	} else if (layout_multi == SI_PC_MULTI_CUSTOM) {
		unsigned *reg = regs->select;
		for (idx = 0; idx < count; ++idx) {
			radeon_set_uconfig_reg(cs, *reg++, selectors[idx] | regs->select_or);
			if (idx < regs->num_multi)
				radeon_set_uconfig_reg(cs, *reg++, 0);
		}
	} else {
		assert(layout_multi == SI_PC_MULTI_ALTERNATE);

		unsigned reg_base = regs->select0;
		unsigned reg_count = count + MIN2(count, regs->num_multi);
		reg_count += regs->num_prelude;

		if (!(regs->layout & SI_PC_REG_REVERSE)) {
			radeon_set_uconfig_reg_seq(cs, reg_base, reg_count);

			for (idx = 0; idx < regs->num_prelude; ++idx)
				radeon_emit(cs, 0);
			for (idx = 0; idx < count; ++idx) {
				radeon_emit(cs, selectors[idx] | regs->select_or);
				if (idx < regs->num_multi)
					radeon_emit(cs, 0);
			}
		} else {
			reg_base -= (reg_count - 1) * 4;
			radeon_set_uconfig_reg_seq(cs, reg_base, reg_count);

			for (idx = count; idx > 0; --idx) {
				if (idx <= regs->num_multi)
					radeon_emit(cs, 0);
				radeon_emit(cs, selectors[idx - 1] | regs->select_or);
			}
			for (idx = 0; idx < regs->num_prelude; ++idx)
				radeon_emit(cs, 0);
		}
	}
}

static void si_pc_emit_start(struct si_context *sctx,
			     struct r600_resource *buffer, uint64_t va)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;

	radeon_add_to_buffer_list(sctx, sctx->gfx_cs, buffer,
				  RADEON_USAGE_WRITE, RADEON_PRIO_QUERY);

	radeon_emit(cs, PKT3(PKT3_COPY_DATA, 4, 0));
	radeon_emit(cs, COPY_DATA_SRC_SEL(COPY_DATA_IMM) |
			COPY_DATA_DST_SEL(COPY_DATA_DST_MEM_GRBM));
	radeon_emit(cs, 1); /* immediate */
	radeon_emit(cs, 0); /* unused */
	radeon_emit(cs, va);
	radeon_emit(cs, va >> 32);

	radeon_set_uconfig_reg(cs, R_036020_CP_PERFMON_CNTL,
			       S_036020_PERFMON_STATE(V_036020_DISABLE_AND_RESET));
	radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
	radeon_emit(cs, EVENT_TYPE(V_028A90_PERFCOUNTER_START) | EVENT_INDEX(0));
	radeon_set_uconfig_reg(cs, R_036020_CP_PERFMON_CNTL,
			       S_036020_PERFMON_STATE(V_036020_START_COUNTING));
}

/* Note: The buffer was already added in si_pc_emit_start, so we don't have to
 * do it again in here. */
static void si_pc_emit_stop(struct si_context *sctx,
			    struct r600_resource *buffer, uint64_t va)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;

	si_cp_release_mem(sctx, V_028A90_BOTTOM_OF_PIPE_TS, 0,
			  EOP_DST_SEL_MEM,
			  EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
			  EOP_DATA_SEL_VALUE_32BIT,
			  buffer, va, 0, SI_NOT_QUERY);
	si_cp_wait_mem(sctx, va, 0, 0xffffffff, 0);

	radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
	radeon_emit(cs, EVENT_TYPE(V_028A90_PERFCOUNTER_SAMPLE) | EVENT_INDEX(0));
	radeon_emit(cs, PKT3(PKT3_EVENT_WRITE, 0, 0));
	radeon_emit(cs, EVENT_TYPE(V_028A90_PERFCOUNTER_STOP) | EVENT_INDEX(0));
	radeon_set_uconfig_reg(cs, R_036020_CP_PERFMON_CNTL,
			       S_036020_PERFMON_STATE(V_036020_STOP_COUNTING) |
			       S_036020_PERFMON_SAMPLE_ENABLE(1));
}

static void si_pc_emit_read(struct si_context *sctx,
			    struct si_perfcounter_block *group,
			    unsigned count, unsigned *selectors,
			    struct r600_resource *buffer, uint64_t va)
{
	struct si_pc_block *sigroup = (struct si_pc_block *)group->data;
	struct si_pc_block_base *regs = sigroup->b;
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	unsigned idx;
	unsigned reg = regs->counter0_lo;
	unsigned reg_delta = 8;

	if (!(regs->layout & SI_PC_FAKE)) {
		if (regs->layout & SI_PC_REG_REVERSE)
			reg_delta = -reg_delta;

		for (idx = 0; idx < count; ++idx) {
			if (regs->counters)
				reg = regs->counters[idx];

			radeon_emit(cs, PKT3(PKT3_COPY_DATA, 4, 0));
			radeon_emit(cs, COPY_DATA_SRC_SEL(COPY_DATA_PERF) |
					COPY_DATA_DST_SEL(COPY_DATA_DST_MEM_GRBM) |
					COPY_DATA_COUNT_SEL); /* 64 bits */
			radeon_emit(cs, reg >> 2);
			radeon_emit(cs, 0); /* unused */
			radeon_emit(cs, va);
			radeon_emit(cs, va >> 32);
			va += sizeof(uint64_t);
			reg += reg_delta;
		}
	} else {
		for (idx = 0; idx < count; ++idx) {
			radeon_emit(cs, PKT3(PKT3_COPY_DATA, 4, 0));
			radeon_emit(cs, COPY_DATA_SRC_SEL(COPY_DATA_IMM) |
					COPY_DATA_DST_SEL(COPY_DATA_DST_MEM_GRBM) |
					COPY_DATA_COUNT_SEL);
			radeon_emit(cs, 0); /* immediate */
			radeon_emit(cs, 0);
			radeon_emit(cs, va);
			radeon_emit(cs, va >> 32);
			va += sizeof(uint64_t);
		}
	}
}

static void si_pc_cleanup(struct si_screen *sscreen)
{
	si_perfcounters_do_destroy(sscreen->perfcounters);
	sscreen->perfcounters = NULL;
}

void si_init_perfcounters(struct si_screen *screen)
{
	struct si_perfcounters *pc;
	struct si_pc_block *blocks;
	unsigned num_blocks;
	unsigned i;

	switch (screen->info.chip_class) {
	case CIK:
		blocks = groups_CIK;
		num_blocks = ARRAY_SIZE(groups_CIK);
		break;
	case VI:
		blocks = groups_VI;
		num_blocks = ARRAY_SIZE(groups_VI);
		break;
	case GFX9:
		blocks = groups_gfx9;
		num_blocks = ARRAY_SIZE(groups_gfx9);
		break;
	case SI:
	default:
		return; /* not implemented */
	}

	if (screen->info.max_sh_per_se != 1) {
		/* This should not happen on non-SI chips. */
		fprintf(stderr, "si_init_perfcounters: max_sh_per_se = %d not "
			"supported (inaccurate performance counters)\n",
			screen->info.max_sh_per_se);
	}

	pc = CALLOC_STRUCT(si_perfcounters);
	if (!pc)
		return;

	pc->num_stop_cs_dwords = 14 + si_cp_write_fence_dwords(screen);
	pc->num_instance_cs_dwords = 3;

	pc->num_shader_types = ARRAY_SIZE(si_pc_shader_type_bits);
	pc->shader_type_suffixes = si_pc_shader_type_suffixes;
	pc->shader_type_bits = si_pc_shader_type_bits;

	pc->emit_instance = si_pc_emit_instance;
	pc->emit_shaders = si_pc_emit_shaders;
	pc->emit_select = si_pc_emit_select;
	pc->emit_start = si_pc_emit_start;
	pc->emit_stop = si_pc_emit_stop;
	pc->emit_read = si_pc_emit_read;
	pc->cleanup = si_pc_cleanup;

	if (!si_perfcounters_init(pc, num_blocks))
		goto error;

	for (i = 0; i < num_blocks; ++i) {
		struct si_pc_block *block = &blocks[i];
		unsigned instances = block->instances;

		if (!strcmp(block->b->name, "CB") ||
		    !strcmp(block->b->name, "DB"))
			instances = screen->info.max_se;
		else if (!strcmp(block->b->name, "TCC"))
			instances = screen->info.num_tcc_blocks;
		else if (!strcmp(block->b->name, "IA"))
			instances = MAX2(1, screen->info.max_se / 2);

		si_perfcounters_add_block(screen, pc,
					    block->b->name,
					    block->b->flags,
					    block->b->num_counters,
					    block->selectors,
					    instances,
					    block);
	}

	screen->perfcounters = pc;
	return;

error:
	si_perfcounters_do_destroy(pc);
}
