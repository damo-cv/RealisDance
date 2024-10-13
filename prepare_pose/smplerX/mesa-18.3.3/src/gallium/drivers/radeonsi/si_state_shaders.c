/*
 * Copyright 2012 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "si_build_pm4.h"
#include "gfx9d.h"

#include "compiler/nir/nir_serialize.h"
#include "tgsi/tgsi_parse.h"
#include "util/hash_table.h"
#include "util/crc32.h"
#include "util/u_async_debug.h"
#include "util/u_memory.h"
#include "util/u_prim.h"

#include "util/disk_cache.h"
#include "util/mesa-sha1.h"
#include "ac_exp_param.h"
#include "ac_shader_util.h"

/* SHADER_CACHE */

/**
 * Return the IR binary in a buffer. For TGSI the first 4 bytes contain its
 * size as integer.
 */
void *si_get_ir_binary(struct si_shader_selector *sel)
{
	struct blob blob;
	unsigned ir_size;
	void *ir_binary;

	if (sel->tokens) {
		ir_binary = sel->tokens;
		ir_size = tgsi_num_tokens(sel->tokens) *
					  sizeof(struct tgsi_token);
	} else {
		assert(sel->nir);

		blob_init(&blob);
		nir_serialize(&blob, sel->nir);
		ir_binary = blob.data;
		ir_size = blob.size;
	}

	unsigned size = 4 + ir_size + sizeof(sel->so);
	char *result = (char*)MALLOC(size);
	if (!result)
		return NULL;

	*((uint32_t*)result) = size;
	memcpy(result + 4, ir_binary, ir_size);
	memcpy(result + 4 + ir_size, &sel->so, sizeof(sel->so));

	if (sel->nir)
		blob_finish(&blob);

	return result;
}

/** Copy "data" to "ptr" and return the next dword following copied data. */
static uint32_t *write_data(uint32_t *ptr, const void *data, unsigned size)
{
	/* data may be NULL if size == 0 */
	if (size)
		memcpy(ptr, data, size);
	ptr += DIV_ROUND_UP(size, 4);
	return ptr;
}

/** Read data from "ptr". Return the next dword following the data. */
static uint32_t *read_data(uint32_t *ptr, void *data, unsigned size)
{
	memcpy(data, ptr, size);
	ptr += DIV_ROUND_UP(size, 4);
	return ptr;
}

/**
 * Write the size as uint followed by the data. Return the next dword
 * following the copied data.
 */
static uint32_t *write_chunk(uint32_t *ptr, const void *data, unsigned size)
{
	*ptr++ = size;
	return write_data(ptr, data, size);
}

/**
 * Read the size as uint followed by the data. Return both via parameters.
 * Return the next dword following the data.
 */
static uint32_t *read_chunk(uint32_t *ptr, void **data, unsigned *size)
{
	*size = *ptr++;
	assert(*data == NULL);
	if (!*size)
		return ptr;
	*data = malloc(*size);
	return read_data(ptr, *data, *size);
}

/**
 * Return the shader binary in a buffer. The first 4 bytes contain its size
 * as integer.
 */
static void *si_get_shader_binary(struct si_shader *shader)
{
	/* There is always a size of data followed by the data itself. */
	unsigned relocs_size = shader->binary.reloc_count *
			       sizeof(shader->binary.relocs[0]);
	unsigned disasm_size = shader->binary.disasm_string ?
			       strlen(shader->binary.disasm_string) + 1 : 0;
	unsigned llvm_ir_size = shader->binary.llvm_ir_string ?
				strlen(shader->binary.llvm_ir_string) + 1 : 0;
	unsigned size =
		4 + /* total size */
		4 + /* CRC32 of the data below */
		align(sizeof(shader->config), 4) +
		align(sizeof(shader->info), 4) +
		4 + align(shader->binary.code_size, 4) +
		4 + align(shader->binary.rodata_size, 4) +
		4 + align(relocs_size, 4) +
		4 + align(disasm_size, 4) +
		4 + align(llvm_ir_size, 4);
	void *buffer = CALLOC(1, size);
	uint32_t *ptr = (uint32_t*)buffer;

	if (!buffer)
		return NULL;

	*ptr++ = size;
	ptr++; /* CRC32 is calculated at the end. */

	ptr = write_data(ptr, &shader->config, sizeof(shader->config));
	ptr = write_data(ptr, &shader->info, sizeof(shader->info));
	ptr = write_chunk(ptr, shader->binary.code, shader->binary.code_size);
	ptr = write_chunk(ptr, shader->binary.rodata, shader->binary.rodata_size);
	ptr = write_chunk(ptr, shader->binary.relocs, relocs_size);
	ptr = write_chunk(ptr, shader->binary.disasm_string, disasm_size);
	ptr = write_chunk(ptr, shader->binary.llvm_ir_string, llvm_ir_size);
	assert((char *)ptr - (char *)buffer == size);

	/* Compute CRC32. */
	ptr = (uint32_t*)buffer;
	ptr++;
	*ptr = util_hash_crc32(ptr + 1, size - 8);

	return buffer;
}

static bool si_load_shader_binary(struct si_shader *shader, void *binary)
{
	uint32_t *ptr = (uint32_t*)binary;
	uint32_t size = *ptr++;
	uint32_t crc32 = *ptr++;
	unsigned chunk_size;

	if (util_hash_crc32(ptr, size - 8) != crc32) {
		fprintf(stderr, "radeonsi: binary shader has invalid CRC32\n");
		return false;
	}

	ptr = read_data(ptr, &shader->config, sizeof(shader->config));
	ptr = read_data(ptr, &shader->info, sizeof(shader->info));
	ptr = read_chunk(ptr, (void**)&shader->binary.code,
			 &shader->binary.code_size);
	ptr = read_chunk(ptr, (void**)&shader->binary.rodata,
			 &shader->binary.rodata_size);
	ptr = read_chunk(ptr, (void**)&shader->binary.relocs, &chunk_size);
	shader->binary.reloc_count = chunk_size / sizeof(shader->binary.relocs[0]);
	ptr = read_chunk(ptr, (void**)&shader->binary.disasm_string, &chunk_size);
	ptr = read_chunk(ptr, (void**)&shader->binary.llvm_ir_string, &chunk_size);

	return true;
}

/**
 * Insert a shader into the cache. It's assumed the shader is not in the cache.
 * Use si_shader_cache_load_shader before calling this.
 *
 * Returns false on failure, in which case the ir_binary should be freed.
 */
bool si_shader_cache_insert_shader(struct si_screen *sscreen, void *ir_binary,
				   struct si_shader *shader,
				   bool insert_into_disk_cache)
{
	void *hw_binary;
	struct hash_entry *entry;
	uint8_t key[CACHE_KEY_SIZE];

	entry = _mesa_hash_table_search(sscreen->shader_cache, ir_binary);
	if (entry)
		return false; /* already added */

	hw_binary = si_get_shader_binary(shader);
	if (!hw_binary)
		return false;

	if (_mesa_hash_table_insert(sscreen->shader_cache, ir_binary,
				    hw_binary) == NULL) {
		FREE(hw_binary);
		return false;
	}

	if (sscreen->disk_shader_cache && insert_into_disk_cache) {
		disk_cache_compute_key(sscreen->disk_shader_cache, ir_binary,
				       *((uint32_t *)ir_binary), key);
		disk_cache_put(sscreen->disk_shader_cache, key, hw_binary,
			       *((uint32_t *) hw_binary), NULL);
	}

	return true;
}

bool si_shader_cache_load_shader(struct si_screen *sscreen, void *ir_binary,
				 struct si_shader *shader)
{
	struct hash_entry *entry =
		_mesa_hash_table_search(sscreen->shader_cache, ir_binary);
	if (!entry) {
		if (sscreen->disk_shader_cache) {
			unsigned char sha1[CACHE_KEY_SIZE];
			size_t tg_size = *((uint32_t *) ir_binary);

			disk_cache_compute_key(sscreen->disk_shader_cache,
					       ir_binary, tg_size, sha1);

			size_t binary_size;
			uint8_t *buffer =
				disk_cache_get(sscreen->disk_shader_cache,
					       sha1, &binary_size);
			if (!buffer)
				return false;

			if (binary_size < sizeof(uint32_t) ||
			    *((uint32_t*)buffer) != binary_size) {
				 /* Something has gone wrong discard the item
				  * from the cache and rebuild/link from
				  * source.
				  */
				assert(!"Invalid radeonsi shader disk cache "
				       "item!");

				disk_cache_remove(sscreen->disk_shader_cache,
						  sha1);
				free(buffer);

				return false;
			}

			if (!si_load_shader_binary(shader, buffer)) {
				free(buffer);
				return false;
			}
			free(buffer);

			if (!si_shader_cache_insert_shader(sscreen, ir_binary,
							   shader, false))
				FREE(ir_binary);
		} else {
			return false;
		}
	} else {
		if (si_load_shader_binary(shader, entry->data))
			FREE(ir_binary);
		else
			return false;
	}
	p_atomic_inc(&sscreen->num_shader_cache_hits);
	return true;
}

static uint32_t si_shader_cache_key_hash(const void *key)
{
	/* The first dword is the key size. */
	return util_hash_crc32(key, *(uint32_t*)key);
}

static bool si_shader_cache_key_equals(const void *a, const void *b)
{
	uint32_t *keya = (uint32_t*)a;
	uint32_t *keyb = (uint32_t*)b;

	/* The first dword is the key size. */
	if (*keya != *keyb)
		return false;

	return memcmp(keya, keyb, *keya) == 0;
}

static void si_destroy_shader_cache_entry(struct hash_entry *entry)
{
	FREE((void*)entry->key);
	FREE(entry->data);
}

bool si_init_shader_cache(struct si_screen *sscreen)
{
	(void) mtx_init(&sscreen->shader_cache_mutex, mtx_plain);
	sscreen->shader_cache =
		_mesa_hash_table_create(NULL,
					si_shader_cache_key_hash,
					si_shader_cache_key_equals);

	return sscreen->shader_cache != NULL;
}

void si_destroy_shader_cache(struct si_screen *sscreen)
{
	if (sscreen->shader_cache)
		_mesa_hash_table_destroy(sscreen->shader_cache,
					 si_destroy_shader_cache_entry);
	mtx_destroy(&sscreen->shader_cache_mutex);
}

/* SHADER STATES */

static void si_set_tesseval_regs(struct si_screen *sscreen,
				 struct si_shader_selector *tes,
				 struct si_pm4_state *pm4)
{
	struct tgsi_shader_info *info = &tes->info;
	unsigned tes_prim_mode = info->properties[TGSI_PROPERTY_TES_PRIM_MODE];
	unsigned tes_spacing = info->properties[TGSI_PROPERTY_TES_SPACING];
	bool tes_vertex_order_cw = info->properties[TGSI_PROPERTY_TES_VERTEX_ORDER_CW];
	bool tes_point_mode = info->properties[TGSI_PROPERTY_TES_POINT_MODE];
	unsigned type, partitioning, topology, distribution_mode;

	switch (tes_prim_mode) {
	case PIPE_PRIM_LINES:
		type = V_028B6C_TESS_ISOLINE;
		break;
	case PIPE_PRIM_TRIANGLES:
		type = V_028B6C_TESS_TRIANGLE;
		break;
	case PIPE_PRIM_QUADS:
		type = V_028B6C_TESS_QUAD;
		break;
	default:
		assert(0);
		return;
	}

	switch (tes_spacing) {
	case PIPE_TESS_SPACING_FRACTIONAL_ODD:
		partitioning = V_028B6C_PART_FRAC_ODD;
		break;
	case PIPE_TESS_SPACING_FRACTIONAL_EVEN:
		partitioning = V_028B6C_PART_FRAC_EVEN;
		break;
	case PIPE_TESS_SPACING_EQUAL:
		partitioning = V_028B6C_PART_INTEGER;
		break;
	default:
		assert(0);
		return;
	}

	if (tes_point_mode)
		topology = V_028B6C_OUTPUT_POINT;
	else if (tes_prim_mode == PIPE_PRIM_LINES)
		topology = V_028B6C_OUTPUT_LINE;
	else if (tes_vertex_order_cw)
		/* for some reason, this must be the other way around */
		topology = V_028B6C_OUTPUT_TRIANGLE_CCW;
	else
		topology = V_028B6C_OUTPUT_TRIANGLE_CW;

	if (sscreen->has_distributed_tess) {
		if (sscreen->info.family == CHIP_FIJI ||
		    sscreen->info.family >= CHIP_POLARIS10)
			distribution_mode = V_028B6C_DISTRIBUTION_MODE_TRAPEZOIDS;
		else
			distribution_mode = V_028B6C_DISTRIBUTION_MODE_DONUTS;
	} else
		distribution_mode = V_028B6C_DISTRIBUTION_MODE_NO_DIST;

	assert(pm4->shader);
	pm4->shader->vgt_tf_param = S_028B6C_TYPE(type) |
				    S_028B6C_PARTITIONING(partitioning) |
				    S_028B6C_TOPOLOGY(topology) |
				    S_028B6C_DISTRIBUTION_MODE(distribution_mode);
}

/* Polaris needs different VTX_REUSE_DEPTH settings depending on
 * whether the "fractional odd" tessellation spacing is used.
 *
 * Possible VGT configurations and which state should set the register:
 *
 *   Reg set in | VGT shader configuration   | Value
 * ------------------------------------------------------
 *     VS as VS | VS                         | 30
 *     VS as ES | ES -> GS -> VS             | 30
 *    TES as VS | LS -> HS -> VS             | 14 or 30
 *    TES as ES | LS -> HS -> ES -> GS -> VS | 14 or 30
 *
 * If "shader" is NULL, it's assumed it's not LS or GS copy shader.
 */
static void polaris_set_vgt_vertex_reuse(struct si_screen *sscreen,
					 struct si_shader_selector *sel,
					 struct si_shader *shader,
					 struct si_pm4_state *pm4)
{
	unsigned type = sel->type;

	if (sscreen->info.family < CHIP_POLARIS10)
		return;

	/* VS as VS, or VS as ES: */
	if ((type == PIPE_SHADER_VERTEX &&
	     (!shader ||
	      (!shader->key.as_ls && !shader->is_gs_copy_shader))) ||
	    /* TES as VS, or TES as ES: */
	    type == PIPE_SHADER_TESS_EVAL) {
		unsigned vtx_reuse_depth = 30;

		if (type == PIPE_SHADER_TESS_EVAL &&
		    sel->info.properties[TGSI_PROPERTY_TES_SPACING] ==
		    PIPE_TESS_SPACING_FRACTIONAL_ODD)
			vtx_reuse_depth = 14;

		assert(pm4->shader);
		pm4->shader->vgt_vertex_reuse_block_cntl = vtx_reuse_depth;
	}
}

static struct si_pm4_state *si_get_shader_pm4_state(struct si_shader *shader)
{
	if (shader->pm4)
		si_pm4_clear_state(shader->pm4);
	else
		shader->pm4 = CALLOC_STRUCT(si_pm4_state);

	if (shader->pm4) {
		shader->pm4->shader = shader;
		return shader->pm4;
	} else {
		fprintf(stderr, "radeonsi: Failed to create pm4 state.\n");
		return NULL;
	}
}

static unsigned si_get_num_vs_user_sgprs(unsigned num_always_on_user_sgprs)
{
	/* Add the pointer to VBO descriptors. */
	if (HAVE_32BIT_POINTERS) {
		return num_always_on_user_sgprs + 1;
	} else {
		assert(num_always_on_user_sgprs % 2 == 0);
		return num_always_on_user_sgprs + 2;
	}
}

static void si_shader_ls(struct si_screen *sscreen, struct si_shader *shader)
{
	struct si_pm4_state *pm4;
	unsigned vgpr_comp_cnt;
	uint64_t va;

	assert(sscreen->info.chip_class <= VI);

	pm4 = si_get_shader_pm4_state(shader);
	if (!pm4)
		return;

	va = shader->bo->gpu_address;
	si_pm4_add_bo(pm4, shader->bo, RADEON_USAGE_READ, RADEON_PRIO_SHADER_BINARY);

	/* We need at least 2 components for LS.
	 * VGPR0-3: (VertexID, RelAutoindex, InstanceID / StepRate0, InstanceID).
	 * StepRate0 is set to 1. so that VGPR3 doesn't have to be loaded.
	 */
	vgpr_comp_cnt = shader->info.uses_instanceid ? 2 : 1;

	si_pm4_set_reg(pm4, R_00B520_SPI_SHADER_PGM_LO_LS, va >> 8);
	si_pm4_set_reg(pm4, R_00B524_SPI_SHADER_PGM_HI_LS, S_00B524_MEM_BASE(va >> 40));

	shader->config.rsrc1 = S_00B528_VGPRS((shader->config.num_vgprs - 1) / 4) |
			   S_00B528_SGPRS((shader->config.num_sgprs - 1) / 8) |
		           S_00B528_VGPR_COMP_CNT(vgpr_comp_cnt) |
			   S_00B528_DX10_CLAMP(1) |
			   S_00B528_FLOAT_MODE(shader->config.float_mode);
	shader->config.rsrc2 = S_00B52C_USER_SGPR(si_get_num_vs_user_sgprs(SI_VS_NUM_USER_SGPR)) |
			   S_00B52C_SCRATCH_EN(shader->config.scratch_bytes_per_wave > 0);
}

static void si_shader_hs(struct si_screen *sscreen, struct si_shader *shader)
{
	struct si_pm4_state *pm4;
	uint64_t va;
	unsigned ls_vgpr_comp_cnt = 0;

	pm4 = si_get_shader_pm4_state(shader);
	if (!pm4)
		return;

	va = shader->bo->gpu_address;
	si_pm4_add_bo(pm4, shader->bo, RADEON_USAGE_READ, RADEON_PRIO_SHADER_BINARY);

	if (sscreen->info.chip_class >= GFX9) {
		si_pm4_set_reg(pm4, R_00B410_SPI_SHADER_PGM_LO_LS, va >> 8);
		si_pm4_set_reg(pm4, R_00B414_SPI_SHADER_PGM_HI_LS, S_00B414_MEM_BASE(va >> 40));

		/* We need at least 2 components for LS.
		 * VGPR0-3: (VertexID, RelAutoindex, InstanceID / StepRate0, InstanceID).
		 * StepRate0 is set to 1. so that VGPR3 doesn't have to be loaded.
		 */
		ls_vgpr_comp_cnt = shader->info.uses_instanceid ? 2 : 1;

		unsigned num_user_sgprs =
			si_get_num_vs_user_sgprs(GFX9_TCS_NUM_USER_SGPR);

		shader->config.rsrc2 =
			S_00B42C_USER_SGPR(num_user_sgprs) |
			S_00B42C_USER_SGPR_MSB(num_user_sgprs >> 5) |
			S_00B42C_SCRATCH_EN(shader->config.scratch_bytes_per_wave > 0);
	} else {
		si_pm4_set_reg(pm4, R_00B420_SPI_SHADER_PGM_LO_HS, va >> 8);
		si_pm4_set_reg(pm4, R_00B424_SPI_SHADER_PGM_HI_HS, S_00B424_MEM_BASE(va >> 40));

		shader->config.rsrc2 =
			S_00B42C_USER_SGPR(GFX6_TCS_NUM_USER_SGPR) |
			S_00B42C_OC_LDS_EN(1) |
			S_00B42C_SCRATCH_EN(shader->config.scratch_bytes_per_wave > 0);
	}

	si_pm4_set_reg(pm4, R_00B428_SPI_SHADER_PGM_RSRC1_HS,
		       S_00B428_VGPRS((shader->config.num_vgprs - 1) / 4) |
		       S_00B428_SGPRS((shader->config.num_sgprs - 1) / 8) |
		       S_00B428_DX10_CLAMP(1) |
		       S_00B428_FLOAT_MODE(shader->config.float_mode) |
		       S_00B428_LS_VGPR_COMP_CNT(ls_vgpr_comp_cnt));

	if (sscreen->info.chip_class <= VI) {
		si_pm4_set_reg(pm4, R_00B42C_SPI_SHADER_PGM_RSRC2_HS,
			       shader->config.rsrc2);
	}
}

static void si_emit_shader_es(struct si_context *sctx)
{
	struct si_shader *shader = sctx->queued.named.es->shader;
	unsigned initial_cdw = sctx->gfx_cs->current.cdw;

	if (!shader)
		return;

	radeon_opt_set_context_reg(sctx, R_028AAC_VGT_ESGS_RING_ITEMSIZE,
				   SI_TRACKED_VGT_ESGS_RING_ITEMSIZE,
				   shader->selector->esgs_itemsize / 4);

	if (shader->selector->type == PIPE_SHADER_TESS_EVAL)
		radeon_opt_set_context_reg(sctx, R_028B6C_VGT_TF_PARAM,
					   SI_TRACKED_VGT_TF_PARAM,
					   shader->vgt_tf_param);

	if (shader->vgt_vertex_reuse_block_cntl)
		radeon_opt_set_context_reg(sctx, R_028C58_VGT_VERTEX_REUSE_BLOCK_CNTL,
					   SI_TRACKED_VGT_VERTEX_REUSE_BLOCK_CNTL,
					   shader->vgt_vertex_reuse_block_cntl);

	if (initial_cdw != sctx->gfx_cs->current.cdw)
		sctx->context_roll_counter++;
}

static void si_shader_es(struct si_screen *sscreen, struct si_shader *shader)
{
	struct si_pm4_state *pm4;
	unsigned num_user_sgprs;
	unsigned vgpr_comp_cnt;
	uint64_t va;
	unsigned oc_lds_en;

	assert(sscreen->info.chip_class <= VI);

	pm4 = si_get_shader_pm4_state(shader);
	if (!pm4)
		return;

	pm4->atom.emit = si_emit_shader_es;
	va = shader->bo->gpu_address;
	si_pm4_add_bo(pm4, shader->bo, RADEON_USAGE_READ, RADEON_PRIO_SHADER_BINARY);

	if (shader->selector->type == PIPE_SHADER_VERTEX) {
		/* VGPR0-3: (VertexID, InstanceID / StepRate0, ...) */
		vgpr_comp_cnt = shader->info.uses_instanceid ? 1 : 0;
		num_user_sgprs = si_get_num_vs_user_sgprs(SI_VS_NUM_USER_SGPR);
	} else if (shader->selector->type == PIPE_SHADER_TESS_EVAL) {
		vgpr_comp_cnt = shader->selector->info.uses_primid ? 3 : 2;
		num_user_sgprs = SI_TES_NUM_USER_SGPR;
	} else
		unreachable("invalid shader selector type");

	oc_lds_en = shader->selector->type == PIPE_SHADER_TESS_EVAL ? 1 : 0;

	si_pm4_set_reg(pm4, R_00B320_SPI_SHADER_PGM_LO_ES, va >> 8);
	si_pm4_set_reg(pm4, R_00B324_SPI_SHADER_PGM_HI_ES, S_00B324_MEM_BASE(va >> 40));
	si_pm4_set_reg(pm4, R_00B328_SPI_SHADER_PGM_RSRC1_ES,
		       S_00B328_VGPRS((shader->config.num_vgprs - 1) / 4) |
		       S_00B328_SGPRS((shader->config.num_sgprs - 1) / 8) |
		       S_00B328_VGPR_COMP_CNT(vgpr_comp_cnt) |
		       S_00B328_DX10_CLAMP(1) |
		       S_00B328_FLOAT_MODE(shader->config.float_mode));
	si_pm4_set_reg(pm4, R_00B32C_SPI_SHADER_PGM_RSRC2_ES,
		       S_00B32C_USER_SGPR(num_user_sgprs) |
		       S_00B32C_OC_LDS_EN(oc_lds_en) |
		       S_00B32C_SCRATCH_EN(shader->config.scratch_bytes_per_wave > 0));

	if (shader->selector->type == PIPE_SHADER_TESS_EVAL)
		si_set_tesseval_regs(sscreen, shader->selector, pm4);

	polaris_set_vgt_vertex_reuse(sscreen, shader->selector, shader, pm4);
}

static unsigned si_conv_prim_to_gs_out(unsigned mode)
{
	static const int prim_conv[] = {
		[PIPE_PRIM_POINTS]			= V_028A6C_OUTPRIM_TYPE_POINTLIST,
		[PIPE_PRIM_LINES]			= V_028A6C_OUTPRIM_TYPE_LINESTRIP,
		[PIPE_PRIM_LINE_LOOP]			= V_028A6C_OUTPRIM_TYPE_LINESTRIP,
		[PIPE_PRIM_LINE_STRIP]			= V_028A6C_OUTPRIM_TYPE_LINESTRIP,
		[PIPE_PRIM_TRIANGLES]			= V_028A6C_OUTPRIM_TYPE_TRISTRIP,
		[PIPE_PRIM_TRIANGLE_STRIP]		= V_028A6C_OUTPRIM_TYPE_TRISTRIP,
		[PIPE_PRIM_TRIANGLE_FAN]		= V_028A6C_OUTPRIM_TYPE_TRISTRIP,
		[PIPE_PRIM_QUADS]			= V_028A6C_OUTPRIM_TYPE_TRISTRIP,
		[PIPE_PRIM_QUAD_STRIP]			= V_028A6C_OUTPRIM_TYPE_TRISTRIP,
		[PIPE_PRIM_POLYGON]			= V_028A6C_OUTPRIM_TYPE_TRISTRIP,
		[PIPE_PRIM_LINES_ADJACENCY]		= V_028A6C_OUTPRIM_TYPE_LINESTRIP,
		[PIPE_PRIM_LINE_STRIP_ADJACENCY]	= V_028A6C_OUTPRIM_TYPE_LINESTRIP,
		[PIPE_PRIM_TRIANGLES_ADJACENCY]		= V_028A6C_OUTPRIM_TYPE_TRISTRIP,
		[PIPE_PRIM_TRIANGLE_STRIP_ADJACENCY]	= V_028A6C_OUTPRIM_TYPE_TRISTRIP,
		[PIPE_PRIM_PATCHES]			= V_028A6C_OUTPRIM_TYPE_POINTLIST,
	};
	assert(mode < ARRAY_SIZE(prim_conv));

	return prim_conv[mode];
}

struct gfx9_gs_info {
	unsigned es_verts_per_subgroup;
	unsigned gs_prims_per_subgroup;
	unsigned gs_inst_prims_in_subgroup;
	unsigned max_prims_per_subgroup;
	unsigned lds_size;
};

static void gfx9_get_gs_info(struct si_shader_selector *es,
				   struct si_shader_selector *gs,
				   struct gfx9_gs_info *out)
{
	unsigned gs_num_invocations = MAX2(gs->gs_num_invocations, 1);
	unsigned input_prim = gs->info.properties[TGSI_PROPERTY_GS_INPUT_PRIM];
	bool uses_adjacency = input_prim >= PIPE_PRIM_LINES_ADJACENCY &&
			      input_prim <= PIPE_PRIM_TRIANGLE_STRIP_ADJACENCY;

	/* All these are in dwords: */
	/* We can't allow using the whole LDS, because GS waves compete with
	 * other shader stages for LDS space. */
	const unsigned max_lds_size = 8 * 1024;
	const unsigned esgs_itemsize = es->esgs_itemsize / 4;
	unsigned esgs_lds_size;

	/* All these are per subgroup: */
	const unsigned max_out_prims = 32 * 1024;
	const unsigned max_es_verts = 255;
	const unsigned ideal_gs_prims = 64;
	unsigned max_gs_prims, gs_prims;
	unsigned min_es_verts, es_verts, worst_case_es_verts;

	if (uses_adjacency || gs_num_invocations > 1)
		max_gs_prims = 127 / gs_num_invocations;
	else
		max_gs_prims = 255;

	/* MAX_PRIMS_PER_SUBGROUP = gs_prims * max_vert_out * gs_invocations.
	 * Make sure we don't go over the maximum value.
	 */
	if (gs->gs_max_out_vertices > 0) {
		max_gs_prims = MIN2(max_gs_prims,
				    max_out_prims /
				    (gs->gs_max_out_vertices * gs_num_invocations));
	}
	assert(max_gs_prims > 0);

	/* If the primitive has adjacency, halve the number of vertices
	 * that will be reused in multiple primitives.
	 */
	min_es_verts = gs->gs_input_verts_per_prim / (uses_adjacency ? 2 : 1);

	gs_prims = MIN2(ideal_gs_prims, max_gs_prims);
	worst_case_es_verts = MIN2(min_es_verts * gs_prims, max_es_verts);

	/* Compute ESGS LDS size based on the worst case number of ES vertices
	 * needed to create the target number of GS prims per subgroup.
	 */
	esgs_lds_size = esgs_itemsize * worst_case_es_verts;

	/* If total LDS usage is too big, refactor partitions based on ratio
	 * of ESGS item sizes.
	 */
	if (esgs_lds_size > max_lds_size) {
		/* Our target GS Prims Per Subgroup was too large. Calculate
		 * the maximum number of GS Prims Per Subgroup that will fit
		 * into LDS, capped by the maximum that the hardware can support.
		 */
		gs_prims = MIN2((max_lds_size / (esgs_itemsize * min_es_verts)),
				max_gs_prims);
		assert(gs_prims > 0);
		worst_case_es_verts = MIN2(min_es_verts * gs_prims,
					   max_es_verts);

		esgs_lds_size = esgs_itemsize * worst_case_es_verts;
		assert(esgs_lds_size <= max_lds_size);
	}

	/* Now calculate remaining ESGS information. */
	if (esgs_lds_size)
		es_verts = MIN2(esgs_lds_size / esgs_itemsize, max_es_verts);
	else
		es_verts = max_es_verts;

	/* Vertices for adjacency primitives are not always reused, so restore
	 * it for ES_VERTS_PER_SUBGRP.
	 */
	min_es_verts = gs->gs_input_verts_per_prim;

	/* For normal primitives, the VGT only checks if they are past the ES
	 * verts per subgroup after allocating a full GS primitive and if they
	 * are, kick off a new subgroup.  But if those additional ES verts are
	 * unique (e.g. not reused) we need to make sure there is enough LDS
	 * space to account for those ES verts beyond ES_VERTS_PER_SUBGRP.
	 */
	es_verts -= min_es_verts - 1;

	out->es_verts_per_subgroup = es_verts;
	out->gs_prims_per_subgroup = gs_prims;
	out->gs_inst_prims_in_subgroup = gs_prims * gs_num_invocations;
	out->max_prims_per_subgroup = out->gs_inst_prims_in_subgroup *
				      gs->gs_max_out_vertices;
	out->lds_size = align(esgs_lds_size, 128) / 128;

	assert(out->max_prims_per_subgroup <= max_out_prims);
}

static void si_emit_shader_gs(struct si_context *sctx)
{
	struct si_shader *shader = sctx->queued.named.gs->shader;
	unsigned initial_cdw = sctx->gfx_cs->current.cdw;

	if (!shader)
		return;

	/* R_028A60_VGT_GSVS_RING_OFFSET_1, R_028A64_VGT_GSVS_RING_OFFSET_2
	 * R_028A68_VGT_GSVS_RING_OFFSET_3, R_028A6C_VGT_GS_OUT_PRIM_TYPE */
	radeon_opt_set_context_reg4(sctx, R_028A60_VGT_GSVS_RING_OFFSET_1,
				    SI_TRACKED_VGT_GSVS_RING_OFFSET_1,
				    shader->ctx_reg.gs.vgt_gsvs_ring_offset_1,
				    shader->ctx_reg.gs.vgt_gsvs_ring_offset_2,
				    shader->ctx_reg.gs.vgt_gsvs_ring_offset_3,
				    shader->ctx_reg.gs.vgt_gs_out_prim_type);


	/* R_028AB0_VGT_GSVS_RING_ITEMSIZE */
	radeon_opt_set_context_reg(sctx, R_028AB0_VGT_GSVS_RING_ITEMSIZE,
				   SI_TRACKED_VGT_GSVS_RING_ITEMSIZE,
				   shader->ctx_reg.gs.vgt_gsvs_ring_itemsize);

	/* R_028B38_VGT_GS_MAX_VERT_OUT */
	radeon_opt_set_context_reg(sctx, R_028B38_VGT_GS_MAX_VERT_OUT,
				   SI_TRACKED_VGT_GS_MAX_VERT_OUT,
				   shader->ctx_reg.gs.vgt_gs_max_vert_out);

	/* R_028B5C_VGT_GS_VERT_ITEMSIZE, R_028B60_VGT_GS_VERT_ITEMSIZE_1
	 * R_028B64_VGT_GS_VERT_ITEMSIZE_2, R_028B68_VGT_GS_VERT_ITEMSIZE_3 */
	radeon_opt_set_context_reg4(sctx, R_028B5C_VGT_GS_VERT_ITEMSIZE,
				    SI_TRACKED_VGT_GS_VERT_ITEMSIZE,
				    shader->ctx_reg.gs.vgt_gs_vert_itemsize,
				    shader->ctx_reg.gs.vgt_gs_vert_itemsize_1,
				    shader->ctx_reg.gs.vgt_gs_vert_itemsize_2,
				    shader->ctx_reg.gs.vgt_gs_vert_itemsize_3);

	/* R_028B90_VGT_GS_INSTANCE_CNT */
	radeon_opt_set_context_reg(sctx, R_028B90_VGT_GS_INSTANCE_CNT,
				   SI_TRACKED_VGT_GS_INSTANCE_CNT,
				   shader->ctx_reg.gs.vgt_gs_instance_cnt);

	if (sctx->chip_class >= GFX9) {
		/* R_028A44_VGT_GS_ONCHIP_CNTL */
		radeon_opt_set_context_reg(sctx, R_028A44_VGT_GS_ONCHIP_CNTL,
					   SI_TRACKED_VGT_GS_ONCHIP_CNTL,
					   shader->ctx_reg.gs.vgt_gs_onchip_cntl);
		/* R_028A94_VGT_GS_MAX_PRIMS_PER_SUBGROUP */
		radeon_opt_set_context_reg(sctx, R_028A94_VGT_GS_MAX_PRIMS_PER_SUBGROUP,
					   SI_TRACKED_VGT_GS_MAX_PRIMS_PER_SUBGROUP,
					   shader->ctx_reg.gs.vgt_gs_max_prims_per_subgroup);
		/* R_028AAC_VGT_ESGS_RING_ITEMSIZE */
		radeon_opt_set_context_reg(sctx, R_028AAC_VGT_ESGS_RING_ITEMSIZE,
					   SI_TRACKED_VGT_ESGS_RING_ITEMSIZE,
					   shader->ctx_reg.gs.vgt_esgs_ring_itemsize);

		if (shader->key.part.gs.es->type == PIPE_SHADER_TESS_EVAL)
			radeon_opt_set_context_reg(sctx, R_028B6C_VGT_TF_PARAM,
						   SI_TRACKED_VGT_TF_PARAM,
						   shader->vgt_tf_param);
		if (shader->vgt_vertex_reuse_block_cntl)
			radeon_opt_set_context_reg(sctx, R_028C58_VGT_VERTEX_REUSE_BLOCK_CNTL,
						   SI_TRACKED_VGT_VERTEX_REUSE_BLOCK_CNTL,
						   shader->vgt_vertex_reuse_block_cntl);
	}

	if (initial_cdw != sctx->gfx_cs->current.cdw)
		sctx->context_roll_counter++;
}

static void si_shader_gs(struct si_screen *sscreen, struct si_shader *shader)
{
	struct si_shader_selector *sel = shader->selector;
	const ubyte *num_components = sel->info.num_stream_output_components;
	unsigned gs_num_invocations = sel->gs_num_invocations;
	struct si_pm4_state *pm4;
	uint64_t va;
	unsigned max_stream = sel->max_gs_stream;
	unsigned offset;

	pm4 = si_get_shader_pm4_state(shader);
	if (!pm4)
		return;

	pm4->atom.emit = si_emit_shader_gs;

	offset = num_components[0] * sel->gs_max_out_vertices;
	shader->ctx_reg.gs.vgt_gsvs_ring_offset_1 = offset;

	if (max_stream >= 1)
		offset += num_components[1] * sel->gs_max_out_vertices;
	shader->ctx_reg.gs.vgt_gsvs_ring_offset_2 = offset;

	if (max_stream >= 2)
		offset += num_components[2] * sel->gs_max_out_vertices;
	shader->ctx_reg.gs.vgt_gsvs_ring_offset_3 = offset;

	shader->ctx_reg.gs.vgt_gs_out_prim_type =
		si_conv_prim_to_gs_out(sel->gs_output_prim);

	if (max_stream >= 3)
		offset += num_components[3] * sel->gs_max_out_vertices;
	shader->ctx_reg.gs.vgt_gsvs_ring_itemsize = offset;

	/* The GSVS_RING_ITEMSIZE register takes 15 bits */
	assert(offset < (1 << 15));

	shader->ctx_reg.gs.vgt_gs_max_vert_out = sel->gs_max_out_vertices;

	shader->ctx_reg.gs.vgt_gs_vert_itemsize = num_components[0];
	shader->ctx_reg.gs.vgt_gs_vert_itemsize_1 = (max_stream >= 1) ? num_components[1] : 0;
	shader->ctx_reg.gs.vgt_gs_vert_itemsize_2 = (max_stream >= 2) ? num_components[2] : 0;
	shader->ctx_reg.gs.vgt_gs_vert_itemsize_3 = (max_stream >= 3) ? num_components[3] : 0;

	shader->ctx_reg.gs.vgt_gs_instance_cnt = S_028B90_CNT(MIN2(gs_num_invocations, 127)) |
						 S_028B90_ENABLE(gs_num_invocations > 0);

	va = shader->bo->gpu_address;
	si_pm4_add_bo(pm4, shader->bo, RADEON_USAGE_READ, RADEON_PRIO_SHADER_BINARY);

	if (sscreen->info.chip_class >= GFX9) {
		unsigned input_prim = sel->info.properties[TGSI_PROPERTY_GS_INPUT_PRIM];
		unsigned es_type = shader->key.part.gs.es->type;
		unsigned es_vgpr_comp_cnt, gs_vgpr_comp_cnt;
		struct gfx9_gs_info gs_info;

		if (es_type == PIPE_SHADER_VERTEX)
			/* VGPR0-3: (VertexID, InstanceID / StepRate0, ...) */
			es_vgpr_comp_cnt = shader->info.uses_instanceid ? 1 : 0;
		else if (es_type == PIPE_SHADER_TESS_EVAL)
			es_vgpr_comp_cnt = shader->key.part.gs.es->info.uses_primid ? 3 : 2;
		else
			unreachable("invalid shader selector type");

		/* If offsets 4, 5 are used, GS_VGPR_COMP_CNT is ignored and
		 * VGPR[0:4] are always loaded.
		 */
		if (sel->info.uses_invocationid)
			gs_vgpr_comp_cnt = 3; /* VGPR3 contains InvocationID. */
		else if (sel->info.uses_primid)
			gs_vgpr_comp_cnt = 2; /* VGPR2 contains PrimitiveID. */
		else if (input_prim >= PIPE_PRIM_TRIANGLES)
			gs_vgpr_comp_cnt = 1; /* VGPR1 contains offsets 2, 3 */
		else
			gs_vgpr_comp_cnt = 0; /* VGPR0 contains offsets 0, 1 */

		unsigned num_user_sgprs;
		if (es_type == PIPE_SHADER_VERTEX)
			num_user_sgprs = si_get_num_vs_user_sgprs(GFX9_VSGS_NUM_USER_SGPR);
		else
			num_user_sgprs = GFX9_TESGS_NUM_USER_SGPR;

		gfx9_get_gs_info(shader->key.part.gs.es, sel, &gs_info);

		si_pm4_set_reg(pm4, R_00B210_SPI_SHADER_PGM_LO_ES, va >> 8);
		si_pm4_set_reg(pm4, R_00B214_SPI_SHADER_PGM_HI_ES, S_00B214_MEM_BASE(va >> 40));

		si_pm4_set_reg(pm4, R_00B228_SPI_SHADER_PGM_RSRC1_GS,
			       S_00B228_VGPRS((shader->config.num_vgprs - 1) / 4) |
			       S_00B228_SGPRS((shader->config.num_sgprs - 1) / 8) |
			       S_00B228_DX10_CLAMP(1) |
			       S_00B228_FLOAT_MODE(shader->config.float_mode) |
			       S_00B228_GS_VGPR_COMP_CNT(gs_vgpr_comp_cnt));
		si_pm4_set_reg(pm4, R_00B22C_SPI_SHADER_PGM_RSRC2_GS,
			       S_00B22C_USER_SGPR(num_user_sgprs) |
			       S_00B22C_USER_SGPR_MSB(num_user_sgprs >> 5) |
			       S_00B22C_ES_VGPR_COMP_CNT(es_vgpr_comp_cnt) |
			       S_00B22C_OC_LDS_EN(es_type == PIPE_SHADER_TESS_EVAL) |
			       S_00B22C_LDS_SIZE(gs_info.lds_size) |
			       S_00B22C_SCRATCH_EN(shader->config.scratch_bytes_per_wave > 0));

		shader->ctx_reg.gs.vgt_gs_onchip_cntl =
			S_028A44_ES_VERTS_PER_SUBGRP(gs_info.es_verts_per_subgroup) |
			S_028A44_GS_PRIMS_PER_SUBGRP(gs_info.gs_prims_per_subgroup) |
			S_028A44_GS_INST_PRIMS_IN_SUBGRP(gs_info.gs_inst_prims_in_subgroup);
		shader->ctx_reg.gs.vgt_gs_max_prims_per_subgroup =
			S_028A94_MAX_PRIMS_PER_SUBGROUP(gs_info.max_prims_per_subgroup);
		shader->ctx_reg.gs.vgt_esgs_ring_itemsize =
			shader->key.part.gs.es->esgs_itemsize / 4;

		if (es_type == PIPE_SHADER_TESS_EVAL)
			si_set_tesseval_regs(sscreen, shader->key.part.gs.es, pm4);

		polaris_set_vgt_vertex_reuse(sscreen, shader->key.part.gs.es,
					     NULL, pm4);
	} else {
		si_pm4_set_reg(pm4, R_00B220_SPI_SHADER_PGM_LO_GS, va >> 8);
		si_pm4_set_reg(pm4, R_00B224_SPI_SHADER_PGM_HI_GS, S_00B224_MEM_BASE(va >> 40));

		si_pm4_set_reg(pm4, R_00B228_SPI_SHADER_PGM_RSRC1_GS,
			       S_00B228_VGPRS((shader->config.num_vgprs - 1) / 4) |
			       S_00B228_SGPRS((shader->config.num_sgprs - 1) / 8) |
			       S_00B228_DX10_CLAMP(1) |
			       S_00B228_FLOAT_MODE(shader->config.float_mode));
		si_pm4_set_reg(pm4, R_00B22C_SPI_SHADER_PGM_RSRC2_GS,
			       S_00B22C_USER_SGPR(GFX6_GS_NUM_USER_SGPR) |
			       S_00B22C_SCRATCH_EN(shader->config.scratch_bytes_per_wave > 0));
	}
}

static void si_emit_shader_vs(struct si_context *sctx)
{
	struct si_shader *shader = sctx->queued.named.vs->shader;
	unsigned initial_cdw = sctx->gfx_cs->current.cdw;

	if (!shader)
		return;

	radeon_opt_set_context_reg(sctx, R_028A40_VGT_GS_MODE,
				   SI_TRACKED_VGT_GS_MODE,
				   shader->ctx_reg.vs.vgt_gs_mode);
	radeon_opt_set_context_reg(sctx, R_028A84_VGT_PRIMITIVEID_EN,
				   SI_TRACKED_VGT_PRIMITIVEID_EN,
				   shader->ctx_reg.vs.vgt_primitiveid_en);

	if (sctx->chip_class <= VI) {
		radeon_opt_set_context_reg(sctx, R_028AB4_VGT_REUSE_OFF,
					   SI_TRACKED_VGT_REUSE_OFF,
					   shader->ctx_reg.vs.vgt_reuse_off);
	}

	radeon_opt_set_context_reg(sctx, R_0286C4_SPI_VS_OUT_CONFIG,
				   SI_TRACKED_SPI_VS_OUT_CONFIG,
				   shader->ctx_reg.vs.spi_vs_out_config);

	radeon_opt_set_context_reg(sctx, R_02870C_SPI_SHADER_POS_FORMAT,
				   SI_TRACKED_SPI_SHADER_POS_FORMAT,
				   shader->ctx_reg.vs.spi_shader_pos_format);

	radeon_opt_set_context_reg(sctx, R_028818_PA_CL_VTE_CNTL,
				   SI_TRACKED_PA_CL_VTE_CNTL,
				   shader->ctx_reg.vs.pa_cl_vte_cntl);

	if (shader->selector->type == PIPE_SHADER_TESS_EVAL)
		radeon_opt_set_context_reg(sctx, R_028B6C_VGT_TF_PARAM,
					   SI_TRACKED_VGT_TF_PARAM,
					   shader->vgt_tf_param);

	if (shader->vgt_vertex_reuse_block_cntl)
		radeon_opt_set_context_reg(sctx, R_028C58_VGT_VERTEX_REUSE_BLOCK_CNTL,
					   SI_TRACKED_VGT_VERTEX_REUSE_BLOCK_CNTL,
					   shader->vgt_vertex_reuse_block_cntl);

	if (initial_cdw != sctx->gfx_cs->current.cdw)
		sctx->context_roll_counter++;
}

/**
 * Compute the state for \p shader, which will run as a vertex shader on the
 * hardware.
 *
 * If \p gs is non-NULL, it points to the geometry shader for which this shader
 * is the copy shader.
 */
static void si_shader_vs(struct si_screen *sscreen, struct si_shader *shader,
                         struct si_shader_selector *gs)
{
	const struct tgsi_shader_info *info = &shader->selector->info;
	struct si_pm4_state *pm4;
	unsigned num_user_sgprs, vgpr_comp_cnt;
	uint64_t va;
	unsigned nparams, oc_lds_en;
	unsigned window_space =
		info->properties[TGSI_PROPERTY_VS_WINDOW_SPACE_POSITION];
	bool enable_prim_id = shader->key.mono.u.vs_export_prim_id || info->uses_primid;

	pm4 = si_get_shader_pm4_state(shader);
	if (!pm4)
		return;

	pm4->atom.emit = si_emit_shader_vs;

	/* We always write VGT_GS_MODE in the VS state, because every switch
	 * between different shader pipelines involving a different GS or no
	 * GS at all involves a switch of the VS (different GS use different
	 * copy shaders). On the other hand, when the API switches from a GS to
	 * no GS and then back to the same GS used originally, the GS state is
	 * not sent again.
	 */
	if (!gs) {
		unsigned mode = V_028A40_GS_OFF;

		/* PrimID needs GS scenario A. */
		if (enable_prim_id)
			mode = V_028A40_GS_SCENARIO_A;

		shader->ctx_reg.vs.vgt_gs_mode = S_028A40_MODE(mode);
		shader->ctx_reg.vs.vgt_primitiveid_en = enable_prim_id;
	} else {
		shader->ctx_reg.vs.vgt_gs_mode = ac_vgt_gs_mode(gs->gs_max_out_vertices,
								sscreen->info.chip_class);
		shader->ctx_reg.vs.vgt_primitiveid_en = 0;
	}

	if (sscreen->info.chip_class <= VI) {
		/* Reuse needs to be set off if we write oViewport. */
		shader->ctx_reg.vs.vgt_reuse_off =
				S_028AB4_REUSE_OFF(info->writes_viewport_index);
	}

	va = shader->bo->gpu_address;
	si_pm4_add_bo(pm4, shader->bo, RADEON_USAGE_READ, RADEON_PRIO_SHADER_BINARY);

	if (gs) {
		vgpr_comp_cnt = 0; /* only VertexID is needed for GS-COPY. */
		num_user_sgprs = SI_GSCOPY_NUM_USER_SGPR;
	} else if (shader->selector->type == PIPE_SHADER_VERTEX) {
		/* VGPR0-3: (VertexID, InstanceID / StepRate0, PrimID, InstanceID)
		 * If PrimID is disabled. InstanceID / StepRate1 is loaded instead.
		 * StepRate0 is set to 1. so that VGPR3 doesn't have to be loaded.
		 */
		vgpr_comp_cnt = enable_prim_id ? 2 : (shader->info.uses_instanceid ? 1 : 0);

		if (info->properties[TGSI_PROPERTY_VS_BLIT_SGPRS]) {
			num_user_sgprs = SI_SGPR_VS_BLIT_DATA +
					 info->properties[TGSI_PROPERTY_VS_BLIT_SGPRS];
		} else {
			num_user_sgprs = si_get_num_vs_user_sgprs(SI_VS_NUM_USER_SGPR);
		}
	} else if (shader->selector->type == PIPE_SHADER_TESS_EVAL) {
		vgpr_comp_cnt = enable_prim_id ? 3 : 2;
		num_user_sgprs = SI_TES_NUM_USER_SGPR;
	} else
		unreachable("invalid shader selector type");

	/* VS is required to export at least one param. */
	nparams = MAX2(shader->info.nr_param_exports, 1);
	shader->ctx_reg.vs.spi_vs_out_config = S_0286C4_VS_EXPORT_COUNT(nparams - 1);

	shader->ctx_reg.vs.spi_shader_pos_format =
			S_02870C_POS0_EXPORT_FORMAT(V_02870C_SPI_SHADER_4COMP) |
			S_02870C_POS1_EXPORT_FORMAT(shader->info.nr_pos_exports > 1 ?
						    V_02870C_SPI_SHADER_4COMP :
						    V_02870C_SPI_SHADER_NONE) |
			S_02870C_POS2_EXPORT_FORMAT(shader->info.nr_pos_exports > 2 ?
						    V_02870C_SPI_SHADER_4COMP :
						    V_02870C_SPI_SHADER_NONE) |
			S_02870C_POS3_EXPORT_FORMAT(shader->info.nr_pos_exports > 3 ?
						    V_02870C_SPI_SHADER_4COMP :
						    V_02870C_SPI_SHADER_NONE);

	oc_lds_en = shader->selector->type == PIPE_SHADER_TESS_EVAL ? 1 : 0;

	si_pm4_set_reg(pm4, R_00B120_SPI_SHADER_PGM_LO_VS, va >> 8);
	si_pm4_set_reg(pm4, R_00B124_SPI_SHADER_PGM_HI_VS, S_00B124_MEM_BASE(va >> 40));
	si_pm4_set_reg(pm4, R_00B128_SPI_SHADER_PGM_RSRC1_VS,
		       S_00B128_VGPRS((shader->config.num_vgprs - 1) / 4) |
		       S_00B128_SGPRS((shader->config.num_sgprs - 1) / 8) |
		       S_00B128_VGPR_COMP_CNT(vgpr_comp_cnt) |
		       S_00B128_DX10_CLAMP(1) |
		       S_00B128_FLOAT_MODE(shader->config.float_mode));
	si_pm4_set_reg(pm4, R_00B12C_SPI_SHADER_PGM_RSRC2_VS,
		       S_00B12C_USER_SGPR(num_user_sgprs) |
		       S_00B12C_OC_LDS_EN(oc_lds_en) |
		       S_00B12C_SO_BASE0_EN(!!shader->selector->so.stride[0]) |
		       S_00B12C_SO_BASE1_EN(!!shader->selector->so.stride[1]) |
		       S_00B12C_SO_BASE2_EN(!!shader->selector->so.stride[2]) |
		       S_00B12C_SO_BASE3_EN(!!shader->selector->so.stride[3]) |
		       S_00B12C_SO_EN(!!shader->selector->so.num_outputs) |
		       S_00B12C_SCRATCH_EN(shader->config.scratch_bytes_per_wave > 0));

	if (window_space)
		shader->ctx_reg.vs.pa_cl_vte_cntl =
				S_028818_VTX_XY_FMT(1) | S_028818_VTX_Z_FMT(1);
	else
		shader->ctx_reg.vs.pa_cl_vte_cntl =
				S_028818_VTX_W0_FMT(1) |
				S_028818_VPORT_X_SCALE_ENA(1) | S_028818_VPORT_X_OFFSET_ENA(1) |
				S_028818_VPORT_Y_SCALE_ENA(1) | S_028818_VPORT_Y_OFFSET_ENA(1) |
				S_028818_VPORT_Z_SCALE_ENA(1) | S_028818_VPORT_Z_OFFSET_ENA(1);

	if (shader->selector->type == PIPE_SHADER_TESS_EVAL)
		si_set_tesseval_regs(sscreen, shader->selector, pm4);

	polaris_set_vgt_vertex_reuse(sscreen, shader->selector, shader, pm4);
}

static unsigned si_get_ps_num_interp(struct si_shader *ps)
{
	struct tgsi_shader_info *info = &ps->selector->info;
	unsigned num_colors = !!(info->colors_read & 0x0f) +
			      !!(info->colors_read & 0xf0);
	unsigned num_interp = ps->selector->info.num_inputs +
			      (ps->key.part.ps.prolog.color_two_side ? num_colors : 0);

	assert(num_interp <= 32);
	return MIN2(num_interp, 32);
}

static unsigned si_get_spi_shader_col_format(struct si_shader *shader)
{
	unsigned value = shader->key.part.ps.epilog.spi_shader_col_format;
	unsigned i, num_targets = (util_last_bit(value) + 3) / 4;

	/* If the i-th target format is set, all previous target formats must
	 * be non-zero to avoid hangs.
	 */
	for (i = 0; i < num_targets; i++)
		if (!(value & (0xf << (i * 4))))
			value |= V_028714_SPI_SHADER_32_R << (i * 4);

	return value;
}

static void si_emit_shader_ps(struct si_context *sctx)
{
	struct si_shader *shader = sctx->queued.named.ps->shader;
	unsigned initial_cdw = sctx->gfx_cs->current.cdw;

	if (!shader)
		return;

	/* R_0286CC_SPI_PS_INPUT_ENA, R_0286D0_SPI_PS_INPUT_ADDR*/
	radeon_opt_set_context_reg2(sctx, R_0286CC_SPI_PS_INPUT_ENA,
				    SI_TRACKED_SPI_PS_INPUT_ENA,
				    shader->ctx_reg.ps.spi_ps_input_ena,
				    shader->ctx_reg.ps.spi_ps_input_addr);

	radeon_opt_set_context_reg(sctx, R_0286E0_SPI_BARYC_CNTL,
				   SI_TRACKED_SPI_BARYC_CNTL,
				   shader->ctx_reg.ps.spi_baryc_cntl);
	radeon_opt_set_context_reg(sctx, R_0286D8_SPI_PS_IN_CONTROL,
				   SI_TRACKED_SPI_PS_IN_CONTROL,
				   shader->ctx_reg.ps.spi_ps_in_control);

	/* R_028710_SPI_SHADER_Z_FORMAT, R_028714_SPI_SHADER_COL_FORMAT */
	radeon_opt_set_context_reg2(sctx, R_028710_SPI_SHADER_Z_FORMAT,
				    SI_TRACKED_SPI_SHADER_Z_FORMAT,
				    shader->ctx_reg.ps.spi_shader_z_format,
				    shader->ctx_reg.ps.spi_shader_col_format);

	radeon_opt_set_context_reg(sctx, R_02823C_CB_SHADER_MASK,
				   SI_TRACKED_CB_SHADER_MASK,
				   shader->ctx_reg.ps.cb_shader_mask);

	if (initial_cdw != sctx->gfx_cs->current.cdw)
		sctx->context_roll_counter++;
}

static void si_shader_ps(struct si_shader *shader)
{
	struct tgsi_shader_info *info = &shader->selector->info;
	struct si_pm4_state *pm4;
	unsigned spi_ps_in_control, spi_shader_col_format, cb_shader_mask;
	unsigned spi_baryc_cntl = S_0286E0_FRONT_FACE_ALL_BITS(1);
	uint64_t va;
	unsigned input_ena = shader->config.spi_ps_input_ena;

	/* we need to enable at least one of them, otherwise we hang the GPU */
	assert(G_0286CC_PERSP_SAMPLE_ENA(input_ena) ||
	       G_0286CC_PERSP_CENTER_ENA(input_ena) ||
	       G_0286CC_PERSP_CENTROID_ENA(input_ena) ||
	       G_0286CC_PERSP_PULL_MODEL_ENA(input_ena) ||
	       G_0286CC_LINEAR_SAMPLE_ENA(input_ena) ||
	       G_0286CC_LINEAR_CENTER_ENA(input_ena) ||
	       G_0286CC_LINEAR_CENTROID_ENA(input_ena) ||
	       G_0286CC_LINE_STIPPLE_TEX_ENA(input_ena));
	/* POS_W_FLOAT_ENA requires one of the perspective weights. */
	assert(!G_0286CC_POS_W_FLOAT_ENA(input_ena) ||
	       G_0286CC_PERSP_SAMPLE_ENA(input_ena) ||
	       G_0286CC_PERSP_CENTER_ENA(input_ena) ||
	       G_0286CC_PERSP_CENTROID_ENA(input_ena) ||
	       G_0286CC_PERSP_PULL_MODEL_ENA(input_ena));

	/* Validate interpolation optimization flags (read as implications). */
	assert(!shader->key.part.ps.prolog.bc_optimize_for_persp ||
	       (G_0286CC_PERSP_CENTER_ENA(input_ena) &&
		G_0286CC_PERSP_CENTROID_ENA(input_ena)));
	assert(!shader->key.part.ps.prolog.bc_optimize_for_linear ||
	       (G_0286CC_LINEAR_CENTER_ENA(input_ena) &&
		G_0286CC_LINEAR_CENTROID_ENA(input_ena)));
	assert(!shader->key.part.ps.prolog.force_persp_center_interp ||
	       (!G_0286CC_PERSP_SAMPLE_ENA(input_ena) &&
		!G_0286CC_PERSP_CENTROID_ENA(input_ena)));
	assert(!shader->key.part.ps.prolog.force_linear_center_interp ||
	       (!G_0286CC_LINEAR_SAMPLE_ENA(input_ena) &&
		!G_0286CC_LINEAR_CENTROID_ENA(input_ena)));
	assert(!shader->key.part.ps.prolog.force_persp_sample_interp ||
	       (!G_0286CC_PERSP_CENTER_ENA(input_ena) &&
		!G_0286CC_PERSP_CENTROID_ENA(input_ena)));
	assert(!shader->key.part.ps.prolog.force_linear_sample_interp ||
	       (!G_0286CC_LINEAR_CENTER_ENA(input_ena) &&
		!G_0286CC_LINEAR_CENTROID_ENA(input_ena)));

	/* Validate cases when the optimizations are off (read as implications). */
	assert(shader->key.part.ps.prolog.bc_optimize_for_persp ||
	       !G_0286CC_PERSP_CENTER_ENA(input_ena) ||
	       !G_0286CC_PERSP_CENTROID_ENA(input_ena));
	assert(shader->key.part.ps.prolog.bc_optimize_for_linear ||
	       !G_0286CC_LINEAR_CENTER_ENA(input_ena) ||
	       !G_0286CC_LINEAR_CENTROID_ENA(input_ena));

	pm4 = si_get_shader_pm4_state(shader);
	if (!pm4)
		return;

	pm4->atom.emit = si_emit_shader_ps;

	/* SPI_BARYC_CNTL.POS_FLOAT_LOCATION
	 * Possible vaules:
	 * 0 -> Position = pixel center
	 * 1 -> Position = pixel centroid
	 * 2 -> Position = at sample position
	 *
	 * From GLSL 4.5 specification, section 7.1:
	 *   "The variable gl_FragCoord is available as an input variable from
	 *    within fragment shaders and it holds the window relative coordinates
	 *    (x, y, z, 1/w) values for the fragment. If multi-sampling, this
	 *    value can be for any location within the pixel, or one of the
	 *    fragment samples. The use of centroid does not further restrict
	 *    this value to be inside the current primitive."
	 *
	 * Meaning that centroid has no effect and we can return anything within
	 * the pixel. Thus, return the value at sample position, because that's
	 * the most accurate one shaders can get.
	 */
	spi_baryc_cntl |= S_0286E0_POS_FLOAT_LOCATION(2);

	if (info->properties[TGSI_PROPERTY_FS_COORD_PIXEL_CENTER] ==
	    TGSI_FS_COORD_PIXEL_CENTER_INTEGER)
		spi_baryc_cntl |= S_0286E0_POS_FLOAT_ULC(1);

	spi_shader_col_format = si_get_spi_shader_col_format(shader);
	cb_shader_mask = ac_get_cb_shader_mask(spi_shader_col_format);

	/* Ensure that some export memory is always allocated, for two reasons:
	 *
	 * 1) Correctness: The hardware ignores the EXEC mask if no export
	 *    memory is allocated, so KILL and alpha test do not work correctly
	 *    without this.
	 * 2) Performance: Every shader needs at least a NULL export, even when
	 *    it writes no color/depth output. The NULL export instruction
	 *    stalls without this setting.
	 *
	 * Don't add this to CB_SHADER_MASK.
	 */
	if (!spi_shader_col_format &&
	    !info->writes_z && !info->writes_stencil && !info->writes_samplemask)
		spi_shader_col_format = V_028714_SPI_SHADER_32_R;

	shader->ctx_reg.ps.spi_ps_input_ena = input_ena;
	shader->ctx_reg.ps.spi_ps_input_addr = shader->config.spi_ps_input_addr;

	/* Set interpolation controls. */
	spi_ps_in_control = S_0286D8_NUM_INTERP(si_get_ps_num_interp(shader));

	shader->ctx_reg.ps.spi_baryc_cntl = spi_baryc_cntl;
	shader->ctx_reg.ps.spi_ps_in_control = spi_ps_in_control;
	shader->ctx_reg.ps.spi_shader_z_format =
			ac_get_spi_shader_z_format(info->writes_z,
						   info->writes_stencil,
						   info->writes_samplemask);
	shader->ctx_reg.ps.spi_shader_col_format = spi_shader_col_format;
	shader->ctx_reg.ps.cb_shader_mask = cb_shader_mask;

	va = shader->bo->gpu_address;
	si_pm4_add_bo(pm4, shader->bo, RADEON_USAGE_READ, RADEON_PRIO_SHADER_BINARY);
	si_pm4_set_reg(pm4, R_00B020_SPI_SHADER_PGM_LO_PS, va >> 8);
	si_pm4_set_reg(pm4, R_00B024_SPI_SHADER_PGM_HI_PS, S_00B024_MEM_BASE(va >> 40));

	si_pm4_set_reg(pm4, R_00B028_SPI_SHADER_PGM_RSRC1_PS,
		       S_00B028_VGPRS((shader->config.num_vgprs - 1) / 4) |
		       S_00B028_SGPRS((shader->config.num_sgprs - 1) / 8) |
		       S_00B028_DX10_CLAMP(1) |
		       S_00B028_FLOAT_MODE(shader->config.float_mode));
	si_pm4_set_reg(pm4, R_00B02C_SPI_SHADER_PGM_RSRC2_PS,
		       S_00B02C_EXTRA_LDS_SIZE(shader->config.lds_size) |
		       S_00B02C_USER_SGPR(SI_PS_NUM_USER_SGPR) |
		       S_00B32C_SCRATCH_EN(shader->config.scratch_bytes_per_wave > 0));
}

static void si_shader_init_pm4_state(struct si_screen *sscreen,
                                     struct si_shader *shader)
{
	switch (shader->selector->type) {
	case PIPE_SHADER_VERTEX:
		if (shader->key.as_ls)
			si_shader_ls(sscreen, shader);
		else if (shader->key.as_es)
			si_shader_es(sscreen, shader);
		else
			si_shader_vs(sscreen, shader, NULL);
		break;
	case PIPE_SHADER_TESS_CTRL:
		si_shader_hs(sscreen, shader);
		break;
	case PIPE_SHADER_TESS_EVAL:
		if (shader->key.as_es)
			si_shader_es(sscreen, shader);
		else
			si_shader_vs(sscreen, shader, NULL);
		break;
	case PIPE_SHADER_GEOMETRY:
		si_shader_gs(sscreen, shader);
		break;
	case PIPE_SHADER_FRAGMENT:
		si_shader_ps(shader);
		break;
	default:
		assert(0);
	}
}

static unsigned si_get_alpha_test_func(struct si_context *sctx)
{
	/* Alpha-test should be disabled if colorbuffer 0 is integer. */
	if (sctx->queued.named.dsa)
		return sctx->queued.named.dsa->alpha_func;

	return PIPE_FUNC_ALWAYS;
}

static void si_shader_selector_key_vs(struct si_context *sctx,
				      struct si_shader_selector *vs,
				      struct si_shader_key *key,
				      struct si_vs_prolog_bits *prolog_key)
{
	if (!sctx->vertex_elements ||
	    vs->info.properties[TGSI_PROPERTY_VS_BLIT_SGPRS])
		return;

	prolog_key->instance_divisor_is_one =
		sctx->vertex_elements->instance_divisor_is_one;
	prolog_key->instance_divisor_is_fetched =
		sctx->vertex_elements->instance_divisor_is_fetched;

	/* Prefer a monolithic shader to allow scheduling divisions around
	 * VBO loads. */
	if (prolog_key->instance_divisor_is_fetched)
		key->opt.prefer_mono = 1;

	unsigned count = MIN2(vs->info.num_inputs,
			      sctx->vertex_elements->count);
	memcpy(key->mono.vs_fix_fetch, sctx->vertex_elements->fix_fetch, count);
}

static void si_shader_selector_key_hw_vs(struct si_context *sctx,
					 struct si_shader_selector *vs,
					 struct si_shader_key *key)
{
	struct si_shader_selector *ps = sctx->ps_shader.cso;

	key->opt.clip_disable =
		sctx->queued.named.rasterizer->clip_plane_enable == 0 &&
		(vs->info.clipdist_writemask ||
		 vs->info.writes_clipvertex) &&
		!vs->info.culldist_writemask;

	/* Find out if PS is disabled. */
	bool ps_disabled = true;
	if (ps) {
		const struct si_state_blend *blend = sctx->queued.named.blend;
		bool alpha_to_coverage = blend && blend->alpha_to_coverage;
		bool ps_modifies_zs = ps->info.uses_kill ||
				      ps->info.writes_z ||
				      ps->info.writes_stencil ||
				      ps->info.writes_samplemask ||
				      alpha_to_coverage ||
				      si_get_alpha_test_func(sctx) != PIPE_FUNC_ALWAYS;
		unsigned ps_colormask = si_get_total_colormask(sctx);

		ps_disabled = sctx->queued.named.rasterizer->rasterizer_discard ||
			      (!ps_colormask &&
			       !ps_modifies_zs &&
			       !ps->info.writes_memory);
	}

	/* Find out which VS outputs aren't used by the PS. */
	uint64_t outputs_written = vs->outputs_written_before_ps;
	uint64_t inputs_read = 0;

	/* Ignore outputs that are not passed from VS to PS. */
	outputs_written &= ~((1ull << si_shader_io_get_unique_index(TGSI_SEMANTIC_POSITION, 0, true)) |
			     (1ull << si_shader_io_get_unique_index(TGSI_SEMANTIC_PSIZE, 0, true)) |
			     (1ull << si_shader_io_get_unique_index(TGSI_SEMANTIC_CLIPVERTEX, 0, true)));

	if (!ps_disabled) {
		inputs_read = ps->inputs_read;
	}

	uint64_t linked = outputs_written & inputs_read;

	key->opt.kill_outputs = ~linked & outputs_written;
}

/* Compute the key for the hw shader variant */
static inline void si_shader_selector_key(struct pipe_context *ctx,
					  struct si_shader_selector *sel,
					  struct si_shader_key *key)
{
	struct si_context *sctx = (struct si_context *)ctx;

	memset(key, 0, sizeof(*key));

	switch (sel->type) {
	case PIPE_SHADER_VERTEX:
		si_shader_selector_key_vs(sctx, sel, key, &key->part.vs.prolog);

		if (sctx->tes_shader.cso)
			key->as_ls = 1;
		else if (sctx->gs_shader.cso)
			key->as_es = 1;
		else {
			si_shader_selector_key_hw_vs(sctx, sel, key);

			if (sctx->ps_shader.cso && sctx->ps_shader.cso->info.uses_primid)
				key->mono.u.vs_export_prim_id = 1;
		}
		break;
	case PIPE_SHADER_TESS_CTRL:
		if (sctx->chip_class >= GFX9) {
			si_shader_selector_key_vs(sctx, sctx->vs_shader.cso,
						  key, &key->part.tcs.ls_prolog);
			key->part.tcs.ls = sctx->vs_shader.cso;

			/* When the LS VGPR fix is needed, monolithic shaders
			 * can:
			 *  - avoid initializing EXEC in both the LS prolog
			 *    and the LS main part when !vs_needs_prolog
			 *  - remove the fixup for unused input VGPRs
			 */
			key->part.tcs.ls_prolog.ls_vgpr_fix = sctx->ls_vgpr_fix;

			/* The LS output / HS input layout can be communicated
			 * directly instead of via user SGPRs for merged LS-HS.
			 * The LS VGPR fix prefers this too.
			 */
			key->opt.prefer_mono = 1;
		}

		key->part.tcs.epilog.prim_mode =
			sctx->tes_shader.cso->info.properties[TGSI_PROPERTY_TES_PRIM_MODE];
		key->part.tcs.epilog.invoc0_tess_factors_are_def =
			sel->tcs_info.tessfactors_are_def_in_all_invocs;
		key->part.tcs.epilog.tes_reads_tess_factors =
			sctx->tes_shader.cso->info.reads_tess_factors;

		if (sel == sctx->fixed_func_tcs_shader.cso)
			key->mono.u.ff_tcs_inputs_to_copy = sctx->vs_shader.cso->outputs_written;
		break;
	case PIPE_SHADER_TESS_EVAL:
		if (sctx->gs_shader.cso)
			key->as_es = 1;
		else {
			si_shader_selector_key_hw_vs(sctx, sel, key);

			if (sctx->ps_shader.cso && sctx->ps_shader.cso->info.uses_primid)
				key->mono.u.vs_export_prim_id = 1;
		}
		break;
	case PIPE_SHADER_GEOMETRY:
		if (sctx->chip_class >= GFX9) {
			if (sctx->tes_shader.cso) {
				key->part.gs.es = sctx->tes_shader.cso;
			} else {
				si_shader_selector_key_vs(sctx, sctx->vs_shader.cso,
							  key, &key->part.gs.vs_prolog);
				key->part.gs.es = sctx->vs_shader.cso;
				key->part.gs.prolog.gfx9_prev_is_vs = 1;
			}

			/* Merged ES-GS can have unbalanced wave usage.
			 *
			 * ES threads are per-vertex, while GS threads are
			 * per-primitive. So without any amplification, there
			 * are fewer GS threads than ES threads, which can result
			 * in empty (no-op) GS waves. With too much amplification,
			 * there are more GS threads than ES threads, which
			 * can result in empty (no-op) ES waves.
			 *
			 * Non-monolithic shaders are implemented by setting EXEC
			 * at the beginning of shader parts, and don't jump to
			 * the end if EXEC is 0.
			 *
			 * Monolithic shaders use conditional blocks, so they can
			 * jump and skip empty waves of ES or GS. So set this to
			 * always use optimized variants, which are monolithic.
			 */
			key->opt.prefer_mono = 1;
		}
		key->part.gs.prolog.tri_strip_adj_fix = sctx->gs_tri_strip_adj_fix;
		break;
	case PIPE_SHADER_FRAGMENT: {
		struct si_state_rasterizer *rs = sctx->queued.named.rasterizer;
		struct si_state_blend *blend = sctx->queued.named.blend;

		if (sel->info.properties[TGSI_PROPERTY_FS_COLOR0_WRITES_ALL_CBUFS] &&
		    sel->info.colors_written == 0x1)
			key->part.ps.epilog.last_cbuf = MAX2(sctx->framebuffer.state.nr_cbufs, 1) - 1;

		if (blend) {
			/* Select the shader color format based on whether
			 * blending or alpha are needed.
			 */
			key->part.ps.epilog.spi_shader_col_format =
				(blend->blend_enable_4bit & blend->need_src_alpha_4bit &
				 sctx->framebuffer.spi_shader_col_format_blend_alpha) |
				(blend->blend_enable_4bit & ~blend->need_src_alpha_4bit &
				 sctx->framebuffer.spi_shader_col_format_blend) |
				(~blend->blend_enable_4bit & blend->need_src_alpha_4bit &
				 sctx->framebuffer.spi_shader_col_format_alpha) |
				(~blend->blend_enable_4bit & ~blend->need_src_alpha_4bit &
				 sctx->framebuffer.spi_shader_col_format);
			key->part.ps.epilog.spi_shader_col_format &= blend->cb_target_enabled_4bit;

			/* The output for dual source blending should have
			 * the same format as the first output.
			 */
			if (blend->dual_src_blend)
				key->part.ps.epilog.spi_shader_col_format |=
					(key->part.ps.epilog.spi_shader_col_format & 0xf) << 4;
		} else
			key->part.ps.epilog.spi_shader_col_format = sctx->framebuffer.spi_shader_col_format;

		/* If alpha-to-coverage is enabled, we have to export alpha
		 * even if there is no color buffer.
		 */
		if (!(key->part.ps.epilog.spi_shader_col_format & 0xf) &&
		    blend && blend->alpha_to_coverage)
			key->part.ps.epilog.spi_shader_col_format |= V_028710_SPI_SHADER_32_AR;

		/* On SI and CIK except Hawaii, the CB doesn't clamp outputs
		 * to the range supported by the type if a channel has less
		 * than 16 bits and the export format is 16_ABGR.
		 */
		if (sctx->chip_class <= CIK && sctx->family != CHIP_HAWAII) {
			key->part.ps.epilog.color_is_int8 = sctx->framebuffer.color_is_int8;
			key->part.ps.epilog.color_is_int10 = sctx->framebuffer.color_is_int10;
		}

		/* Disable unwritten outputs (if WRITE_ALL_CBUFS isn't enabled). */
		if (!key->part.ps.epilog.last_cbuf) {
			key->part.ps.epilog.spi_shader_col_format &= sel->colors_written_4bit;
			key->part.ps.epilog.color_is_int8 &= sel->info.colors_written;
			key->part.ps.epilog.color_is_int10 &= sel->info.colors_written;
		}

		bool is_poly = !util_prim_is_points_or_lines(sctx->current_rast_prim);
		bool is_line = util_prim_is_lines(sctx->current_rast_prim);

		key->part.ps.prolog.color_two_side = rs->two_side && sel->info.colors_read;
		key->part.ps.prolog.flatshade_colors = rs->flatshade && sel->info.colors_read;

		if (sctx->queued.named.blend) {
			key->part.ps.epilog.alpha_to_one = sctx->queued.named.blend->alpha_to_one &&
							   rs->multisample_enable;
		}

		key->part.ps.prolog.poly_stipple = rs->poly_stipple_enable && is_poly;
		key->part.ps.epilog.poly_line_smoothing = ((is_poly && rs->poly_smooth) ||
							   (is_line && rs->line_smooth)) &&
							  sctx->framebuffer.nr_samples <= 1;
		key->part.ps.epilog.clamp_color = rs->clamp_fragment_color;

		if (sctx->ps_iter_samples > 1 &&
		    sel->info.reads_samplemask) {
			key->part.ps.prolog.samplemask_log_ps_iter =
				util_logbase2(sctx->ps_iter_samples);
		}

		if (rs->force_persample_interp &&
		    rs->multisample_enable &&
		    sctx->framebuffer.nr_samples > 1 &&
		    sctx->ps_iter_samples > 1) {
			key->part.ps.prolog.force_persp_sample_interp =
				sel->info.uses_persp_center ||
				sel->info.uses_persp_centroid;

			key->part.ps.prolog.force_linear_sample_interp =
				sel->info.uses_linear_center ||
				sel->info.uses_linear_centroid;
		} else if (rs->multisample_enable &&
			   sctx->framebuffer.nr_samples > 1) {
			key->part.ps.prolog.bc_optimize_for_persp =
				sel->info.uses_persp_center &&
				sel->info.uses_persp_centroid;
			key->part.ps.prolog.bc_optimize_for_linear =
				sel->info.uses_linear_center &&
				sel->info.uses_linear_centroid;
		} else {
			/* Make sure SPI doesn't compute more than 1 pair
			 * of (i,j), which is the optimization here. */
			key->part.ps.prolog.force_persp_center_interp =
				sel->info.uses_persp_center +
				sel->info.uses_persp_centroid +
				sel->info.uses_persp_sample > 1;

			key->part.ps.prolog.force_linear_center_interp =
				sel->info.uses_linear_center +
				sel->info.uses_linear_centroid +
				sel->info.uses_linear_sample > 1;

			if (sel->info.opcode_count[TGSI_OPCODE_INTERP_SAMPLE])
				key->mono.u.ps.interpolate_at_sample_force_center = 1;
		}

		key->part.ps.epilog.alpha_func = si_get_alpha_test_func(sctx);

		/* ps_uses_fbfetch is true only if the color buffer is bound. */
		if (sctx->ps_uses_fbfetch && !sctx->blitter->running) {
			struct pipe_surface *cb0 = sctx->framebuffer.state.cbufs[0];
			struct pipe_resource *tex = cb0->texture;

			/* 1D textures are allocated and used as 2D on GFX9. */
			key->mono.u.ps.fbfetch_msaa = sctx->framebuffer.nr_samples > 1;
			key->mono.u.ps.fbfetch_is_1D = sctx->chip_class != GFX9 &&
						       (tex->target == PIPE_TEXTURE_1D ||
							tex->target == PIPE_TEXTURE_1D_ARRAY);
			key->mono.u.ps.fbfetch_layered = tex->target == PIPE_TEXTURE_1D_ARRAY ||
							 tex->target == PIPE_TEXTURE_2D_ARRAY ||
							 tex->target == PIPE_TEXTURE_CUBE ||
							 tex->target == PIPE_TEXTURE_CUBE_ARRAY ||
							 tex->target == PIPE_TEXTURE_3D;
		}
		break;
	}
	default:
		assert(0);
	}

	if (unlikely(sctx->screen->debug_flags & DBG(NO_OPT_VARIANT)))
		memset(&key->opt, 0, sizeof(key->opt));
}

static void si_build_shader_variant(struct si_shader *shader,
				    int thread_index,
				    bool low_priority)
{
	struct si_shader_selector *sel = shader->selector;
	struct si_screen *sscreen = sel->screen;
	struct ac_llvm_compiler *compiler;
	struct pipe_debug_callback *debug = &shader->compiler_ctx_state.debug;
	int r;

	if (thread_index >= 0) {
		if (low_priority) {
			assert(thread_index < ARRAY_SIZE(sscreen->compiler_lowp));
			compiler = &sscreen->compiler_lowp[thread_index];
		} else {
			assert(thread_index < ARRAY_SIZE(sscreen->compiler));
			compiler = &sscreen->compiler[thread_index];
		}
		if (!debug->async)
			debug = NULL;
	} else {
		assert(!low_priority);
		compiler = shader->compiler_ctx_state.compiler;
	}

	r = si_shader_create(sscreen, compiler, shader, debug);
	if (unlikely(r)) {
		PRINT_ERR("Failed to build shader variant (type=%u) %d\n",
			 sel->type, r);
		shader->compilation_failed = true;
		return;
	}

	if (shader->compiler_ctx_state.is_debug_context) {
		FILE *f = open_memstream(&shader->shader_log,
					 &shader->shader_log_size);
		if (f) {
			si_shader_dump(sscreen, shader, NULL, sel->type, f, false);
			fclose(f);
		}
	}

	si_shader_init_pm4_state(sscreen, shader);
}

static void si_build_shader_variant_low_priority(void *job, int thread_index)
{
	struct si_shader *shader = (struct si_shader *)job;

	assert(thread_index >= 0);

	si_build_shader_variant(shader, thread_index, true);
}

static const struct si_shader_key zeroed;

static bool si_check_missing_main_part(struct si_screen *sscreen,
				       struct si_shader_selector *sel,
				       struct si_compiler_ctx_state *compiler_state,
				       struct si_shader_key *key)
{
	struct si_shader **mainp = si_get_main_shader_part(sel, key);

	if (!*mainp) {
		struct si_shader *main_part = CALLOC_STRUCT(si_shader);

		if (!main_part)
			return false;

		/* We can leave the fence as permanently signaled because the
		 * main part becomes visible globally only after it has been
		 * compiled. */
		util_queue_fence_init(&main_part->ready);

		main_part->selector = sel;
		main_part->key.as_es = key->as_es;
		main_part->key.as_ls = key->as_ls;
		main_part->is_monolithic = false;

		if (si_compile_tgsi_shader(sscreen, compiler_state->compiler,
					   main_part, &compiler_state->debug) != 0) {
			FREE(main_part);
			return false;
		}
		*mainp = main_part;
	}
	return true;
}

/* Select the hw shader variant depending on the current state. */
static int si_shader_select_with_key(struct si_screen *sscreen,
				     struct si_shader_ctx_state *state,
				     struct si_compiler_ctx_state *compiler_state,
				     struct si_shader_key *key,
				     int thread_index)
{
	struct si_shader_selector *sel = state->cso;
	struct si_shader_selector *previous_stage_sel = NULL;
	struct si_shader *current = state->current;
	struct si_shader *iter, *shader = NULL;

again:
	/* Check if we don't need to change anything.
	 * This path is also used for most shaders that don't need multiple
	 * variants, it will cost just a computation of the key and this
	 * test. */
	if (likely(current &&
		   memcmp(&current->key, key, sizeof(*key)) == 0)) {
		if (unlikely(!util_queue_fence_is_signalled(&current->ready))) {
			if (current->is_optimized) {
				memset(&key->opt, 0, sizeof(key->opt));
				goto current_not_ready;
			}

			util_queue_fence_wait(&current->ready);
		}

		return current->compilation_failed ? -1 : 0;
	}
current_not_ready:

	/* This must be done before the mutex is locked, because async GS
	 * compilation calls this function too, and therefore must enter
	 * the mutex first.
	 *
	 * Only wait if we are in a draw call. Don't wait if we are
	 * in a compiler thread.
	 */
	if (thread_index < 0)
		util_queue_fence_wait(&sel->ready);

	mtx_lock(&sel->mutex);

	/* Find the shader variant. */
	for (iter = sel->first_variant; iter; iter = iter->next_variant) {
		/* Don't check the "current" shader. We checked it above. */
		if (current != iter &&
		    memcmp(&iter->key, key, sizeof(*key)) == 0) {
			mtx_unlock(&sel->mutex);

			if (unlikely(!util_queue_fence_is_signalled(&iter->ready))) {
				/* If it's an optimized shader and its compilation has
				 * been started but isn't done, use the unoptimized
				 * shader so as not to cause a stall due to compilation.
				 */
				if (iter->is_optimized) {
					memset(&key->opt, 0, sizeof(key->opt));
					goto again;
				}

				util_queue_fence_wait(&iter->ready);
			}

			if (iter->compilation_failed) {
				return -1; /* skip the draw call */
			}

			state->current = iter;
			return 0;
		}
	}

	/* Build a new shader. */
	shader = CALLOC_STRUCT(si_shader);
	if (!shader) {
		mtx_unlock(&sel->mutex);
		return -ENOMEM;
	}

	util_queue_fence_init(&shader->ready);

	shader->selector = sel;
	shader->key = *key;
	shader->compiler_ctx_state = *compiler_state;

	/* If this is a merged shader, get the first shader's selector. */
	if (sscreen->info.chip_class >= GFX9) {
		if (sel->type == PIPE_SHADER_TESS_CTRL)
			previous_stage_sel = key->part.tcs.ls;
		else if (sel->type == PIPE_SHADER_GEOMETRY)
			previous_stage_sel = key->part.gs.es;

		/* We need to wait for the previous shader. */
		if (previous_stage_sel && thread_index < 0)
			util_queue_fence_wait(&previous_stage_sel->ready);
	}

	/* Compile the main shader part if it doesn't exist. This can happen
	 * if the initial guess was wrong. */
	bool is_pure_monolithic =
		sscreen->use_monolithic_shaders ||
		memcmp(&key->mono, &zeroed.mono, sizeof(key->mono)) != 0;

	if (!is_pure_monolithic) {
		bool ok;

		/* Make sure the main shader part is present. This is needed
		 * for shaders that can be compiled as VS, LS, or ES, and only
		 * one of them is compiled at creation.
		 *
		 * For merged shaders, check that the starting shader's main
		 * part is present.
		 */
		if (previous_stage_sel) {
			struct si_shader_key shader1_key = zeroed;

			if (sel->type == PIPE_SHADER_TESS_CTRL)
				shader1_key.as_ls = 1;
			else if (sel->type == PIPE_SHADER_GEOMETRY)
				shader1_key.as_es = 1;
			else
				assert(0);

			mtx_lock(&previous_stage_sel->mutex);
			ok = si_check_missing_main_part(sscreen,
							previous_stage_sel,
							compiler_state, &shader1_key);
			mtx_unlock(&previous_stage_sel->mutex);
		} else {
			ok = si_check_missing_main_part(sscreen, sel,
							compiler_state, key);
		}
		if (!ok) {
			FREE(shader);
			mtx_unlock(&sel->mutex);
			return -ENOMEM; /* skip the draw call */
		}
	}

	/* Keep the reference to the 1st shader of merged shaders, so that
	 * Gallium can't destroy it before we destroy the 2nd shader.
	 *
	 * Set sctx = NULL, because it's unused if we're not releasing
	 * the shader, and we don't have any sctx here.
	 */
	si_shader_selector_reference(NULL, &shader->previous_stage_sel,
				     previous_stage_sel);

	/* Monolithic-only shaders don't make a distinction between optimized
	 * and unoptimized. */
	shader->is_monolithic =
		is_pure_monolithic ||
		memcmp(&key->opt, &zeroed.opt, sizeof(key->opt)) != 0;

	shader->is_optimized =
		!is_pure_monolithic &&
		memcmp(&key->opt, &zeroed.opt, sizeof(key->opt)) != 0;

	/* If it's an optimized shader, compile it asynchronously. */
	if (shader->is_optimized &&
	    !is_pure_monolithic &&
	    thread_index < 0) {
		/* Compile it asynchronously. */
		util_queue_add_job(&sscreen->shader_compiler_queue_low_priority,
				   shader, &shader->ready,
				   si_build_shader_variant_low_priority, NULL);

		/* Add only after the ready fence was reset, to guard against a
		 * race with si_bind_XX_shader. */
		if (!sel->last_variant) {
			sel->first_variant = shader;
			sel->last_variant = shader;
		} else {
			sel->last_variant->next_variant = shader;
			sel->last_variant = shader;
		}

		/* Use the default (unoptimized) shader for now. */
		memset(&key->opt, 0, sizeof(key->opt));
		mtx_unlock(&sel->mutex);
		goto again;
	}

	/* Reset the fence before adding to the variant list. */
	util_queue_fence_reset(&shader->ready);

	if (!sel->last_variant) {
		sel->first_variant = shader;
		sel->last_variant = shader;
	} else {
		sel->last_variant->next_variant = shader;
		sel->last_variant = shader;
	}

	mtx_unlock(&sel->mutex);

	assert(!shader->is_optimized);
	si_build_shader_variant(shader, thread_index, false);

	util_queue_fence_signal(&shader->ready);

	if (!shader->compilation_failed)
		state->current = shader;

	return shader->compilation_failed ? -1 : 0;
}

static int si_shader_select(struct pipe_context *ctx,
			    struct si_shader_ctx_state *state,
			    struct si_compiler_ctx_state *compiler_state)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_shader_key key;

	si_shader_selector_key(ctx, state->cso, &key);
	return si_shader_select_with_key(sctx->screen, state, compiler_state,
					 &key, -1);
}

static void si_parse_next_shader_property(const struct tgsi_shader_info *info,
					  bool streamout,
					  struct si_shader_key *key)
{
	unsigned next_shader = info->properties[TGSI_PROPERTY_NEXT_SHADER];

	switch (info->processor) {
	case PIPE_SHADER_VERTEX:
		switch (next_shader) {
		case PIPE_SHADER_GEOMETRY:
			key->as_es = 1;
			break;
		case PIPE_SHADER_TESS_CTRL:
		case PIPE_SHADER_TESS_EVAL:
			key->as_ls = 1;
			break;
		default:
			/* If POSITION isn't written, it can only be a HW VS
			 * if streamout is used. If streamout isn't used,
			 * assume that it's a HW LS. (the next shader is TCS)
			 * This heuristic is needed for separate shader objects.
			 */
			if (!info->writes_position && !streamout)
				key->as_ls = 1;
		}
		break;

	case PIPE_SHADER_TESS_EVAL:
		if (next_shader == PIPE_SHADER_GEOMETRY ||
		    !info->writes_position)
			key->as_es = 1;
		break;
	}
}

/**
 * Compile the main shader part or the monolithic shader as part of
 * si_shader_selector initialization. Since it can be done asynchronously,
 * there is no way to report compile failures to applications.
 */
static void si_init_shader_selector_async(void *job, int thread_index)
{
	struct si_shader_selector *sel = (struct si_shader_selector *)job;
	struct si_screen *sscreen = sel->screen;
	struct ac_llvm_compiler *compiler;
	struct pipe_debug_callback *debug = &sel->compiler_ctx_state.debug;

	assert(!debug->debug_message || debug->async);
	assert(thread_index >= 0);
	assert(thread_index < ARRAY_SIZE(sscreen->compiler));
	compiler = &sscreen->compiler[thread_index];

	/* Compile the main shader part for use with a prolog and/or epilog.
	 * If this fails, the driver will try to compile a monolithic shader
	 * on demand.
	 */
	if (!sscreen->use_monolithic_shaders) {
		struct si_shader *shader = CALLOC_STRUCT(si_shader);
		void *ir_binary = NULL;

		if (!shader) {
			fprintf(stderr, "radeonsi: can't allocate a main shader part\n");
			return;
		}

		/* We can leave the fence signaled because use of the default
		 * main part is guarded by the selector's ready fence. */
		util_queue_fence_init(&shader->ready);

		shader->selector = sel;
		shader->is_monolithic = false;
		si_parse_next_shader_property(&sel->info,
					      sel->so.num_outputs != 0,
					      &shader->key);

		if (sel->tokens || sel->nir)
			ir_binary = si_get_ir_binary(sel);

		/* Try to load the shader from the shader cache. */
		mtx_lock(&sscreen->shader_cache_mutex);

		if (ir_binary &&
		    si_shader_cache_load_shader(sscreen, ir_binary, shader)) {
			mtx_unlock(&sscreen->shader_cache_mutex);
			si_shader_dump_stats_for_shader_db(shader, debug);
		} else {
			mtx_unlock(&sscreen->shader_cache_mutex);

			/* Compile the shader if it hasn't been loaded from the cache. */
			if (si_compile_tgsi_shader(sscreen, compiler, shader,
						   debug) != 0) {
				FREE(shader);
				FREE(ir_binary);
				fprintf(stderr, "radeonsi: can't compile a main shader part\n");
				return;
			}

			if (ir_binary) {
				mtx_lock(&sscreen->shader_cache_mutex);
				if (!si_shader_cache_insert_shader(sscreen, ir_binary, shader, true))
					FREE(ir_binary);
				mtx_unlock(&sscreen->shader_cache_mutex);
			}
		}

		*si_get_main_shader_part(sel, &shader->key) = shader;

		/* Unset "outputs_written" flags for outputs converted to
		 * DEFAULT_VAL, so that later inter-shader optimizations don't
		 * try to eliminate outputs that don't exist in the final
		 * shader.
		 *
		 * This is only done if non-monolithic shaders are enabled.
		 */
		if ((sel->type == PIPE_SHADER_VERTEX ||
		     sel->type == PIPE_SHADER_TESS_EVAL) &&
		    !shader->key.as_ls &&
		    !shader->key.as_es) {
			unsigned i;

			for (i = 0; i < sel->info.num_outputs; i++) {
				unsigned offset = shader->info.vs_output_param_offset[i];

				if (offset <= AC_EXP_PARAM_OFFSET_31)
					continue;

				unsigned name = sel->info.output_semantic_name[i];
				unsigned index = sel->info.output_semantic_index[i];
				unsigned id;

				switch (name) {
				case TGSI_SEMANTIC_GENERIC:
					/* don't process indices the function can't handle */
					if (index >= SI_MAX_IO_GENERIC)
						break;
					/* fall through */
				default:
					id = si_shader_io_get_unique_index(name, index, true);
					sel->outputs_written_before_ps &= ~(1ull << id);
					break;
				case TGSI_SEMANTIC_POSITION: /* ignore these */
				case TGSI_SEMANTIC_PSIZE:
				case TGSI_SEMANTIC_CLIPVERTEX:
				case TGSI_SEMANTIC_EDGEFLAG:
					break;
				}
			}
		}
	}

	/* The GS copy shader is always pre-compiled. */
	if (sel->type == PIPE_SHADER_GEOMETRY) {
		sel->gs_copy_shader = si_generate_gs_copy_shader(sscreen, compiler, sel, debug);
		if (!sel->gs_copy_shader) {
			fprintf(stderr, "radeonsi: can't create GS copy shader\n");
			return;
		}

		si_shader_vs(sscreen, sel->gs_copy_shader, sel);
	}
}

void si_schedule_initial_compile(struct si_context *sctx, unsigned processor,
				 struct util_queue_fence *ready_fence,
				 struct si_compiler_ctx_state *compiler_ctx_state,
				 void *job, util_queue_execute_func execute)
{
	util_queue_fence_init(ready_fence);

	struct util_async_debug_callback async_debug;
	bool wait =
		(sctx->debug.debug_message && !sctx->debug.async) ||
		sctx->is_debug ||
		si_can_dump_shader(sctx->screen, processor);

	if (wait) {
		u_async_debug_init(&async_debug);
		compiler_ctx_state->debug = async_debug.base;
	}

	util_queue_add_job(&sctx->screen->shader_compiler_queue, job,
			   ready_fence, execute, NULL);

	if (wait) {
		util_queue_fence_wait(ready_fence);
		u_async_debug_drain(&async_debug, &sctx->debug);
		u_async_debug_cleanup(&async_debug);
	}
}

/* Return descriptor slot usage masks from the given shader info. */
void si_get_active_slot_masks(const struct tgsi_shader_info *info,
			      uint32_t *const_and_shader_buffers,
			      uint64_t *samplers_and_images)
{
	unsigned start, num_shaderbufs, num_constbufs, num_images, num_samplers;

	num_shaderbufs = util_last_bit(info->shader_buffers_declared);
	num_constbufs = util_last_bit(info->const_buffers_declared);
	/* two 8-byte images share one 16-byte slot */
	num_images = align(util_last_bit(info->images_declared), 2);
	num_samplers = util_last_bit(info->samplers_declared);

	/* The layout is: sb[last] ... sb[0], cb[0] ... cb[last] */
	start = si_get_shaderbuf_slot(num_shaderbufs - 1);
	*const_and_shader_buffers =
		u_bit_consecutive(start, num_shaderbufs + num_constbufs);

	/* The layout is: image[last] ... image[0], sampler[0] ... sampler[last] */
	start = si_get_image_slot(num_images - 1) / 2;
	*samplers_and_images =
		u_bit_consecutive64(start, num_images / 2 + num_samplers);
}

static void *si_create_shader_selector(struct pipe_context *ctx,
				       const struct pipe_shader_state *state)
{
	struct si_screen *sscreen = (struct si_screen *)ctx->screen;
	struct si_context *sctx = (struct si_context*)ctx;
	struct si_shader_selector *sel = CALLOC_STRUCT(si_shader_selector);
	int i;

	if (!sel)
		return NULL;

	pipe_reference_init(&sel->reference, 1);
	sel->screen = sscreen;
	sel->compiler_ctx_state.debug = sctx->debug;
	sel->compiler_ctx_state.is_debug_context = sctx->is_debug;

	sel->so = state->stream_output;

	if (state->type == PIPE_SHADER_IR_TGSI) {
		sel->tokens = tgsi_dup_tokens(state->tokens);
		if (!sel->tokens) {
			FREE(sel);
			return NULL;
		}

		tgsi_scan_shader(state->tokens, &sel->info);
		tgsi_scan_tess_ctrl(state->tokens, &sel->info, &sel->tcs_info);
	} else {
		assert(state->type == PIPE_SHADER_IR_NIR);

		sel->nir = state->ir.nir;

		si_nir_scan_shader(sel->nir, &sel->info);
		si_nir_scan_tess_ctrl(sel->nir, &sel->info, &sel->tcs_info);

		si_lower_nir(sel);
	}

	sel->type = sel->info.processor;
	p_atomic_inc(&sscreen->num_shaders_created);
	si_get_active_slot_masks(&sel->info,
				 &sel->active_const_and_shader_buffers,
				 &sel->active_samplers_and_images);

	/* Record which streamout buffers are enabled. */
	for (i = 0; i < sel->so.num_outputs; i++) {
		sel->enabled_streamout_buffer_mask |=
			(1 << sel->so.output[i].output_buffer) <<
			(sel->so.output[i].stream * 4);
	}

	/* The prolog is a no-op if there are no inputs. */
	sel->vs_needs_prolog = sel->type == PIPE_SHADER_VERTEX &&
			       sel->info.num_inputs &&
			       !sel->info.properties[TGSI_PROPERTY_VS_BLIT_SGPRS];

	sel->force_correct_derivs_after_kill =
		sel->type == PIPE_SHADER_FRAGMENT &&
		sel->info.uses_derivatives &&
		sel->info.uses_kill &&
		sctx->screen->debug_flags & DBG(FS_CORRECT_DERIVS_AFTER_KILL);

	/* Set which opcode uses which (i,j) pair. */
	if (sel->info.uses_persp_opcode_interp_centroid)
		sel->info.uses_persp_centroid = true;

	if (sel->info.uses_linear_opcode_interp_centroid)
		sel->info.uses_linear_centroid = true;

	if (sel->info.uses_persp_opcode_interp_offset ||
	    sel->info.uses_persp_opcode_interp_sample)
		sel->info.uses_persp_center = true;

	if (sel->info.uses_linear_opcode_interp_offset ||
	    sel->info.uses_linear_opcode_interp_sample)
		sel->info.uses_linear_center = true;

	switch (sel->type) {
	case PIPE_SHADER_GEOMETRY:
		sel->gs_output_prim =
			sel->info.properties[TGSI_PROPERTY_GS_OUTPUT_PRIM];
		sel->gs_max_out_vertices =
			sel->info.properties[TGSI_PROPERTY_GS_MAX_OUTPUT_VERTICES];
		sel->gs_num_invocations =
			sel->info.properties[TGSI_PROPERTY_GS_INVOCATIONS];
		sel->gsvs_vertex_size = sel->info.num_outputs * 16;
		sel->max_gsvs_emit_size = sel->gsvs_vertex_size *
					  sel->gs_max_out_vertices;

		sel->max_gs_stream = 0;
		for (i = 0; i < sel->so.num_outputs; i++)
			sel->max_gs_stream = MAX2(sel->max_gs_stream,
						  sel->so.output[i].stream);

		sel->gs_input_verts_per_prim =
			u_vertices_per_prim(sel->info.properties[TGSI_PROPERTY_GS_INPUT_PRIM]);
		break;

	case PIPE_SHADER_TESS_CTRL:
		/* Always reserve space for these. */
		sel->patch_outputs_written |=
			(1ull << si_shader_io_get_unique_index_patch(TGSI_SEMANTIC_TESSINNER, 0)) |
			(1ull << si_shader_io_get_unique_index_patch(TGSI_SEMANTIC_TESSOUTER, 0));
		/* fall through */
	case PIPE_SHADER_VERTEX:
	case PIPE_SHADER_TESS_EVAL:
		for (i = 0; i < sel->info.num_outputs; i++) {
			unsigned name = sel->info.output_semantic_name[i];
			unsigned index = sel->info.output_semantic_index[i];

			switch (name) {
			case TGSI_SEMANTIC_TESSINNER:
			case TGSI_SEMANTIC_TESSOUTER:
			case TGSI_SEMANTIC_PATCH:
				sel->patch_outputs_written |=
					1ull << si_shader_io_get_unique_index_patch(name, index);
				break;

			case TGSI_SEMANTIC_GENERIC:
				/* don't process indices the function can't handle */
				if (index >= SI_MAX_IO_GENERIC)
					break;
				/* fall through */
			default:
				sel->outputs_written |=
					1ull << si_shader_io_get_unique_index(name, index, false);
				sel->outputs_written_before_ps |=
					1ull << si_shader_io_get_unique_index(name, index, true);
				break;
			case TGSI_SEMANTIC_EDGEFLAG:
				break;
			}
		}
		sel->esgs_itemsize = util_last_bit64(sel->outputs_written) * 16;
		sel->lshs_vertex_stride = sel->esgs_itemsize;

		/* Add 1 dword to reduce LDS bank conflicts, so that each vertex
		 * will start on a different bank. (except for the maximum 32*16).
		 */
		if (sel->lshs_vertex_stride < 32*16)
			sel->lshs_vertex_stride += 4;

		/* For the ESGS ring in LDS, add 1 dword to reduce LDS bank
		 * conflicts, i.e. each vertex will start at a different bank.
		 */
		if (sctx->chip_class >= GFX9)
			sel->esgs_itemsize += 4;

		assert(((sel->esgs_itemsize / 4) & C_028AAC_ITEMSIZE) == 0);
		break;

	case PIPE_SHADER_FRAGMENT:
		for (i = 0; i < sel->info.num_inputs; i++) {
			unsigned name = sel->info.input_semantic_name[i];
			unsigned index = sel->info.input_semantic_index[i];

			switch (name) {
			case TGSI_SEMANTIC_GENERIC:
				/* don't process indices the function can't handle */
				if (index >= SI_MAX_IO_GENERIC)
					break;
				/* fall through */
			default:
				sel->inputs_read |=
					1ull << si_shader_io_get_unique_index(name, index, true);
				break;
			case TGSI_SEMANTIC_PCOORD: /* ignore this */
				break;
			}
		}

		for (i = 0; i < 8; i++)
			if (sel->info.colors_written & (1 << i))
				sel->colors_written_4bit |= 0xf << (4 * i);

		for (i = 0; i < sel->info.num_inputs; i++) {
			if (sel->info.input_semantic_name[i] == TGSI_SEMANTIC_COLOR) {
				int index = sel->info.input_semantic_index[i];
				sel->color_attr_index[index] = i;
			}
		}
		break;
	}

	/* PA_CL_VS_OUT_CNTL */
	bool misc_vec_ena =
		sel->info.writes_psize || sel->info.writes_edgeflag ||
		sel->info.writes_layer || sel->info.writes_viewport_index;
	sel->pa_cl_vs_out_cntl =
		S_02881C_USE_VTX_POINT_SIZE(sel->info.writes_psize) |
		S_02881C_USE_VTX_EDGE_FLAG(sel->info.writes_edgeflag) |
		S_02881C_USE_VTX_RENDER_TARGET_INDX(sel->info.writes_layer) |
		S_02881C_USE_VTX_VIEWPORT_INDX(sel->info.writes_viewport_index) |
		S_02881C_VS_OUT_MISC_VEC_ENA(misc_vec_ena) |
		S_02881C_VS_OUT_MISC_SIDE_BUS_ENA(misc_vec_ena);
	sel->clipdist_mask = sel->info.writes_clipvertex ?
				     SIX_BITS : sel->info.clipdist_writemask;
	sel->culldist_mask = sel->info.culldist_writemask <<
			     sel->info.num_written_clipdistance;

	/* DB_SHADER_CONTROL */
	sel->db_shader_control =
		S_02880C_Z_EXPORT_ENABLE(sel->info.writes_z) |
		S_02880C_STENCIL_TEST_VAL_EXPORT_ENABLE(sel->info.writes_stencil) |
		S_02880C_MASK_EXPORT_ENABLE(sel->info.writes_samplemask) |
		S_02880C_KILL_ENABLE(sel->info.uses_kill);

	switch (sel->info.properties[TGSI_PROPERTY_FS_DEPTH_LAYOUT]) {
	case TGSI_FS_DEPTH_LAYOUT_GREATER:
		sel->db_shader_control |=
			S_02880C_CONSERVATIVE_Z_EXPORT(V_02880C_EXPORT_GREATER_THAN_Z);
		break;
	case TGSI_FS_DEPTH_LAYOUT_LESS:
		sel->db_shader_control |=
			S_02880C_CONSERVATIVE_Z_EXPORT(V_02880C_EXPORT_LESS_THAN_Z);
		break;
	}

	/* Z_ORDER, EXEC_ON_HIER_FAIL and EXEC_ON_NOOP should be set as following:
	 *
	 *   | early Z/S | writes_mem | allow_ReZ? |      Z_ORDER       | EXEC_ON_HIER_FAIL | EXEC_ON_NOOP
	 * --|-----------|------------|------------|--------------------|-------------------|-------------
	 * 1a|   false   |   false    |   true     | EarlyZ_Then_ReZ    |         0         |     0
	 * 1b|   false   |   false    |   false    | EarlyZ_Then_LateZ  |         0         |     0
	 * 2 |   false   |   true     |   n/a      |       LateZ        |         1         |     0
	 * 3 |   true    |   false    |   n/a      | EarlyZ_Then_LateZ  |         0         |     0
	 * 4 |   true    |   true     |   n/a      | EarlyZ_Then_LateZ  |         0         |     1
	 *
	 * In cases 3 and 4, HW will force Z_ORDER to EarlyZ regardless of what's set in the register.
	 * In case 2, NOOP_CULL is a don't care field. In case 2, 3 and 4, ReZ doesn't make sense.
	 *
	 * Don't use ReZ without profiling !!!
	 *
	 * ReZ decreases performance by 15% in DiRT: Showdown on Ultra settings, which has pretty complex
	 * shaders.
	 */
	if (sel->info.properties[TGSI_PROPERTY_FS_EARLY_DEPTH_STENCIL]) {
		/* Cases 3, 4. */
		sel->db_shader_control |= S_02880C_DEPTH_BEFORE_SHADER(1) |
					  S_02880C_Z_ORDER(V_02880C_EARLY_Z_THEN_LATE_Z) |
					  S_02880C_EXEC_ON_NOOP(sel->info.writes_memory);
	} else if (sel->info.writes_memory) {
		/* Case 2. */
		sel->db_shader_control |= S_02880C_Z_ORDER(V_02880C_LATE_Z) |
					  S_02880C_EXEC_ON_HIER_FAIL(1);
	} else {
		/* Case 1. */
		sel->db_shader_control |= S_02880C_Z_ORDER(V_02880C_EARLY_Z_THEN_LATE_Z);
	}

	(void) mtx_init(&sel->mutex, mtx_plain);

	si_schedule_initial_compile(sctx, sel->info.processor, &sel->ready,
				    &sel->compiler_ctx_state, sel,
				    si_init_shader_selector_async);
	return sel;
}

static void si_update_streamout_state(struct si_context *sctx)
{
	struct si_shader_selector *shader_with_so = si_get_vs(sctx)->cso;

	if (!shader_with_so)
		return;

	sctx->streamout.enabled_stream_buffers_mask =
		shader_with_so->enabled_streamout_buffer_mask;
	sctx->streamout.stride_in_dw = shader_with_so->so.stride;
}

static void si_update_clip_regs(struct si_context *sctx,
				struct si_shader_selector *old_hw_vs,
				struct si_shader *old_hw_vs_variant,
				struct si_shader_selector *next_hw_vs,
				struct si_shader *next_hw_vs_variant)
{
	if (next_hw_vs &&
	    (!old_hw_vs ||
	     old_hw_vs->info.properties[TGSI_PROPERTY_VS_WINDOW_SPACE_POSITION] !=
	     next_hw_vs->info.properties[TGSI_PROPERTY_VS_WINDOW_SPACE_POSITION] ||
	     old_hw_vs->pa_cl_vs_out_cntl != next_hw_vs->pa_cl_vs_out_cntl ||
	     old_hw_vs->clipdist_mask != next_hw_vs->clipdist_mask ||
	     old_hw_vs->culldist_mask != next_hw_vs->culldist_mask ||
	     !old_hw_vs_variant ||
	     !next_hw_vs_variant ||
	     old_hw_vs_variant->key.opt.clip_disable !=
	     next_hw_vs_variant->key.opt.clip_disable))
		si_mark_atom_dirty(sctx, &sctx->atoms.s.clip_regs);
}

static void si_update_common_shader_state(struct si_context *sctx)
{
	sctx->uses_bindless_samplers =
		si_shader_uses_bindless_samplers(sctx->vs_shader.cso)  ||
		si_shader_uses_bindless_samplers(sctx->gs_shader.cso)  ||
		si_shader_uses_bindless_samplers(sctx->ps_shader.cso)  ||
		si_shader_uses_bindless_samplers(sctx->tcs_shader.cso) ||
		si_shader_uses_bindless_samplers(sctx->tes_shader.cso);
	sctx->uses_bindless_images =
		si_shader_uses_bindless_images(sctx->vs_shader.cso)  ||
		si_shader_uses_bindless_images(sctx->gs_shader.cso)  ||
		si_shader_uses_bindless_images(sctx->ps_shader.cso)  ||
		si_shader_uses_bindless_images(sctx->tcs_shader.cso) ||
		si_shader_uses_bindless_images(sctx->tes_shader.cso);
	sctx->do_update_shaders = true;
}

static void si_bind_vs_shader(struct pipe_context *ctx, void *state)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_shader_selector *old_hw_vs = si_get_vs(sctx)->cso;
	struct si_shader *old_hw_vs_variant = si_get_vs_state(sctx);
	struct si_shader_selector *sel = state;

	if (sctx->vs_shader.cso == sel)
		return;

	sctx->vs_shader.cso = sel;
	sctx->vs_shader.current = sel ? sel->first_variant : NULL;
	sctx->num_vs_blit_sgprs = sel ? sel->info.properties[TGSI_PROPERTY_VS_BLIT_SGPRS] : 0;

	si_update_common_shader_state(sctx);
	si_update_vs_viewport_state(sctx);
	si_set_active_descriptors_for_shader(sctx, sel);
	si_update_streamout_state(sctx);
	si_update_clip_regs(sctx, old_hw_vs, old_hw_vs_variant,
			    si_get_vs(sctx)->cso, si_get_vs_state(sctx));
}

static void si_update_tess_uses_prim_id(struct si_context *sctx)
{
	sctx->ia_multi_vgt_param_key.u.tess_uses_prim_id =
		(sctx->tes_shader.cso &&
		 sctx->tes_shader.cso->info.uses_primid) ||
		(sctx->tcs_shader.cso &&
		 sctx->tcs_shader.cso->info.uses_primid) ||
		(sctx->gs_shader.cso &&
		 sctx->gs_shader.cso->info.uses_primid) ||
		(sctx->ps_shader.cso && !sctx->gs_shader.cso &&
		 sctx->ps_shader.cso->info.uses_primid);
}

static void si_bind_gs_shader(struct pipe_context *ctx, void *state)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_shader_selector *old_hw_vs = si_get_vs(sctx)->cso;
	struct si_shader *old_hw_vs_variant = si_get_vs_state(sctx);
	struct si_shader_selector *sel = state;
	bool enable_changed = !!sctx->gs_shader.cso != !!sel;

	if (sctx->gs_shader.cso == sel)
		return;

	sctx->gs_shader.cso = sel;
	sctx->gs_shader.current = sel ? sel->first_variant : NULL;
	sctx->ia_multi_vgt_param_key.u.uses_gs = sel != NULL;

	si_update_common_shader_state(sctx);
	sctx->last_rast_prim = -1; /* reset this so that it gets updated */

	if (enable_changed) {
		si_shader_change_notify(sctx);
		if (sctx->ia_multi_vgt_param_key.u.uses_tess)
			si_update_tess_uses_prim_id(sctx);
	}
	si_update_vs_viewport_state(sctx);
	si_set_active_descriptors_for_shader(sctx, sel);
	si_update_streamout_state(sctx);
	si_update_clip_regs(sctx, old_hw_vs, old_hw_vs_variant,
			    si_get_vs(sctx)->cso, si_get_vs_state(sctx));
}

static void si_bind_tcs_shader(struct pipe_context *ctx, void *state)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_shader_selector *sel = state;
	bool enable_changed = !!sctx->tcs_shader.cso != !!sel;

	if (sctx->tcs_shader.cso == sel)
		return;

	sctx->tcs_shader.cso = sel;
	sctx->tcs_shader.current = sel ? sel->first_variant : NULL;
	si_update_tess_uses_prim_id(sctx);

	si_update_common_shader_state(sctx);

	if (enable_changed)
		sctx->last_tcs = NULL; /* invalidate derived tess state */

	si_set_active_descriptors_for_shader(sctx, sel);
}

static void si_bind_tes_shader(struct pipe_context *ctx, void *state)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_shader_selector *old_hw_vs = si_get_vs(sctx)->cso;
	struct si_shader *old_hw_vs_variant = si_get_vs_state(sctx);
	struct si_shader_selector *sel = state;
	bool enable_changed = !!sctx->tes_shader.cso != !!sel;

	if (sctx->tes_shader.cso == sel)
		return;

	sctx->tes_shader.cso = sel;
	sctx->tes_shader.current = sel ? sel->first_variant : NULL;
	sctx->ia_multi_vgt_param_key.u.uses_tess = sel != NULL;
	si_update_tess_uses_prim_id(sctx);

	si_update_common_shader_state(sctx);
	sctx->last_rast_prim = -1; /* reset this so that it gets updated */

	if (enable_changed) {
		si_shader_change_notify(sctx);
		sctx->last_tes_sh_base = -1; /* invalidate derived tess state */
	}
	si_update_vs_viewport_state(sctx);
	si_set_active_descriptors_for_shader(sctx, sel);
	si_update_streamout_state(sctx);
	si_update_clip_regs(sctx, old_hw_vs, old_hw_vs_variant,
			    si_get_vs(sctx)->cso, si_get_vs_state(sctx));
}

static void si_bind_ps_shader(struct pipe_context *ctx, void *state)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_shader_selector *old_sel = sctx->ps_shader.cso;
	struct si_shader_selector *sel = state;

	/* skip if supplied shader is one already in use */
	if (old_sel == sel)
		return;

	sctx->ps_shader.cso = sel;
	sctx->ps_shader.current = sel ? sel->first_variant : NULL;

	si_update_common_shader_state(sctx);
	if (sel) {
		if (sctx->ia_multi_vgt_param_key.u.uses_tess)
			si_update_tess_uses_prim_id(sctx);

		if (!old_sel ||
		    old_sel->info.colors_written != sel->info.colors_written)
			si_mark_atom_dirty(sctx, &sctx->atoms.s.cb_render_state);

		if (sctx->screen->has_out_of_order_rast &&
		    (!old_sel ||
		     old_sel->info.writes_memory != sel->info.writes_memory ||
		     old_sel->info.properties[TGSI_PROPERTY_FS_EARLY_DEPTH_STENCIL] !=
		     sel->info.properties[TGSI_PROPERTY_FS_EARLY_DEPTH_STENCIL]))
			si_mark_atom_dirty(sctx, &sctx->atoms.s.msaa_config);
	}
	si_set_active_descriptors_for_shader(sctx, sel);
	si_update_ps_colorbuf0_slot(sctx);
}

static void si_delete_shader(struct si_context *sctx, struct si_shader *shader)
{
	if (shader->is_optimized) {
		util_queue_drop_job(&sctx->screen->shader_compiler_queue_low_priority,
				    &shader->ready);
	}

	util_queue_fence_destroy(&shader->ready);

	if (shader->pm4) {
		switch (shader->selector->type) {
		case PIPE_SHADER_VERTEX:
			if (shader->key.as_ls) {
				assert(sctx->chip_class <= VI);
				si_pm4_delete_state(sctx, ls, shader->pm4);
			} else if (shader->key.as_es) {
				assert(sctx->chip_class <= VI);
				si_pm4_delete_state(sctx, es, shader->pm4);
			} else {
				si_pm4_delete_state(sctx, vs, shader->pm4);
			}
			break;
		case PIPE_SHADER_TESS_CTRL:
			si_pm4_delete_state(sctx, hs, shader->pm4);
			break;
		case PIPE_SHADER_TESS_EVAL:
			if (shader->key.as_es) {
				assert(sctx->chip_class <= VI);
				si_pm4_delete_state(sctx, es, shader->pm4);
			} else {
				si_pm4_delete_state(sctx, vs, shader->pm4);
			}
			break;
		case PIPE_SHADER_GEOMETRY:
			if (shader->is_gs_copy_shader)
				si_pm4_delete_state(sctx, vs, shader->pm4);
			else
				si_pm4_delete_state(sctx, gs, shader->pm4);
			break;
		case PIPE_SHADER_FRAGMENT:
			si_pm4_delete_state(sctx, ps, shader->pm4);
			break;
		}
	}

	si_shader_selector_reference(sctx, &shader->previous_stage_sel, NULL);
	si_shader_destroy(shader);
	free(shader);
}

void si_destroy_shader_selector(struct si_context *sctx,
				struct si_shader_selector *sel)
{
	struct si_shader *p = sel->first_variant, *c;
	struct si_shader_ctx_state *current_shader[SI_NUM_SHADERS] = {
		[PIPE_SHADER_VERTEX] = &sctx->vs_shader,
		[PIPE_SHADER_TESS_CTRL] = &sctx->tcs_shader,
		[PIPE_SHADER_TESS_EVAL] = &sctx->tes_shader,
		[PIPE_SHADER_GEOMETRY] = &sctx->gs_shader,
		[PIPE_SHADER_FRAGMENT] = &sctx->ps_shader,
	};

	util_queue_drop_job(&sctx->screen->shader_compiler_queue, &sel->ready);

	if (current_shader[sel->type]->cso == sel) {
		current_shader[sel->type]->cso = NULL;
		current_shader[sel->type]->current = NULL;
	}

	while (p) {
		c = p->next_variant;
		si_delete_shader(sctx, p);
		p = c;
	}

	if (sel->main_shader_part)
		si_delete_shader(sctx, sel->main_shader_part);
	if (sel->main_shader_part_ls)
		si_delete_shader(sctx, sel->main_shader_part_ls);
	if (sel->main_shader_part_es)
		si_delete_shader(sctx, sel->main_shader_part_es);
	if (sel->gs_copy_shader)
		si_delete_shader(sctx, sel->gs_copy_shader);

	util_queue_fence_destroy(&sel->ready);
	mtx_destroy(&sel->mutex);
	free(sel->tokens);
	ralloc_free(sel->nir);
	free(sel);
}

static void si_delete_shader_selector(struct pipe_context *ctx, void *state)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_shader_selector *sel = (struct si_shader_selector *)state;

	si_shader_selector_reference(sctx, &sel, NULL);
}

static unsigned si_get_ps_input_cntl(struct si_context *sctx,
				     struct si_shader *vs, unsigned name,
				     unsigned index, unsigned interpolate)
{
	struct tgsi_shader_info *vsinfo = &vs->selector->info;
	unsigned j, offset, ps_input_cntl = 0;

	if (interpolate == TGSI_INTERPOLATE_CONSTANT ||
	    (interpolate == TGSI_INTERPOLATE_COLOR && sctx->flatshade))
		ps_input_cntl |= S_028644_FLAT_SHADE(1);

	if (name == TGSI_SEMANTIC_PCOORD ||
	    (name == TGSI_SEMANTIC_TEXCOORD &&
	     sctx->sprite_coord_enable & (1 << index))) {
		ps_input_cntl |= S_028644_PT_SPRITE_TEX(1);
	}

	for (j = 0; j < vsinfo->num_outputs; j++) {
		if (name == vsinfo->output_semantic_name[j] &&
		    index == vsinfo->output_semantic_index[j]) {
			offset = vs->info.vs_output_param_offset[j];

			if (offset <= AC_EXP_PARAM_OFFSET_31) {
				/* The input is loaded from parameter memory. */
				ps_input_cntl |= S_028644_OFFSET(offset);
			} else if (!G_028644_PT_SPRITE_TEX(ps_input_cntl)) {
				if (offset == AC_EXP_PARAM_UNDEFINED) {
					/* This can happen with depth-only rendering. */
					offset = 0;
				} else {
					/* The input is a DEFAULT_VAL constant. */
					assert(offset >= AC_EXP_PARAM_DEFAULT_VAL_0000 &&
					       offset <= AC_EXP_PARAM_DEFAULT_VAL_1111);
					offset -= AC_EXP_PARAM_DEFAULT_VAL_0000;
				}

				ps_input_cntl = S_028644_OFFSET(0x20) |
						S_028644_DEFAULT_VAL(offset);
			}
			break;
		}
	}

	if (name == TGSI_SEMANTIC_PRIMID)
		/* PrimID is written after the last output. */
		ps_input_cntl |= S_028644_OFFSET(vs->info.vs_output_param_offset[vsinfo->num_outputs]);
	else if (j == vsinfo->num_outputs && !G_028644_PT_SPRITE_TEX(ps_input_cntl)) {
		/* No corresponding output found, load defaults into input.
		 * Don't set any other bits.
		 * (FLAT_SHADE=1 completely changes behavior) */
		ps_input_cntl = S_028644_OFFSET(0x20);
		/* D3D 9 behaviour. GL is undefined */
		if (name == TGSI_SEMANTIC_COLOR && index == 0)
			ps_input_cntl |= S_028644_DEFAULT_VAL(3);
	}
	return ps_input_cntl;
}

static void si_emit_spi_map(struct si_context *sctx)
{
	struct si_shader *ps = sctx->ps_shader.current;
	struct si_shader *vs = si_get_vs_state(sctx);
	struct tgsi_shader_info *psinfo = ps ? &ps->selector->info : NULL;
	unsigned i, num_interp, num_written = 0, bcol_interp[2];
	unsigned spi_ps_input_cntl[32];

	if (!ps || !ps->selector->info.num_inputs)
		return;

	num_interp = si_get_ps_num_interp(ps);
	assert(num_interp > 0);

	for (i = 0; i < psinfo->num_inputs; i++) {
		unsigned name = psinfo->input_semantic_name[i];
		unsigned index = psinfo->input_semantic_index[i];
		unsigned interpolate = psinfo->input_interpolate[i];

		spi_ps_input_cntl[num_written++] = si_get_ps_input_cntl(sctx, vs, name,
							    index, interpolate);

		if (name == TGSI_SEMANTIC_COLOR) {
			assert(index < ARRAY_SIZE(bcol_interp));
			bcol_interp[index] = interpolate;
		}
	}

	if (ps->key.part.ps.prolog.color_two_side) {
		unsigned bcol = TGSI_SEMANTIC_BCOLOR;

		for (i = 0; i < 2; i++) {
			if (!(psinfo->colors_read & (0xf << (i * 4))))
				continue;

			spi_ps_input_cntl[num_written++] =
			  si_get_ps_input_cntl(sctx, vs, bcol, i, bcol_interp[i]);

		}
	}
	assert(num_interp == num_written);

	/* R_028644_SPI_PS_INPUT_CNTL_0 */
	/* Dota 2: Only ~16% of SPI map updates set different values. */
	/* Talos: Only ~9% of SPI map updates set different values. */
	unsigned initial_cdw = sctx->gfx_cs->current.cdw;
	radeon_opt_set_context_regn(sctx, R_028644_SPI_PS_INPUT_CNTL_0,
				    spi_ps_input_cntl,
				    sctx->tracked_regs.spi_ps_input_cntl, num_interp);

	if (initial_cdw != sctx->gfx_cs->current.cdw)
		sctx->context_roll_counter++;
}

/**
 * Writing CONFIG or UCONFIG VGT registers requires VGT_FLUSH before that.
 */
static void si_init_config_add_vgt_flush(struct si_context *sctx)
{
	if (sctx->init_config_has_vgt_flush)
		return;

	/* Done by Vulkan before VGT_FLUSH. */
	si_pm4_cmd_begin(sctx->init_config, PKT3_EVENT_WRITE);
	si_pm4_cmd_add(sctx->init_config,
		       EVENT_TYPE(V_028A90_VS_PARTIAL_FLUSH) | EVENT_INDEX(4));
	si_pm4_cmd_end(sctx->init_config, false);

	/* VGT_FLUSH is required even if VGT is idle. It resets VGT pointers. */
	si_pm4_cmd_begin(sctx->init_config, PKT3_EVENT_WRITE);
	si_pm4_cmd_add(sctx->init_config, EVENT_TYPE(V_028A90_VGT_FLUSH) | EVENT_INDEX(0));
	si_pm4_cmd_end(sctx->init_config, false);
	sctx->init_config_has_vgt_flush = true;
}

/* Initialize state related to ESGS / GSVS ring buffers */
static bool si_update_gs_ring_buffers(struct si_context *sctx)
{
	struct si_shader_selector *es =
		sctx->tes_shader.cso ? sctx->tes_shader.cso : sctx->vs_shader.cso;
	struct si_shader_selector *gs = sctx->gs_shader.cso;
	struct si_pm4_state *pm4;

	/* Chip constants. */
	unsigned num_se = sctx->screen->info.max_se;
	unsigned wave_size = 64;
	unsigned max_gs_waves = 32 * num_se; /* max 32 per SE on GCN */
	/* On SI-CI, the value comes from VGT_GS_VERTEX_REUSE = 16.
	 * On VI+, the value comes from VGT_VERTEX_REUSE_BLOCK_CNTL = 30 (+2).
	 */
	unsigned gs_vertex_reuse = (sctx->chip_class >= VI ? 32 : 16) * num_se;
	unsigned alignment = 256 * num_se;
	/* The maximum size is 63.999 MB per SE. */
	unsigned max_size = ((unsigned)(63.999 * 1024 * 1024) & ~255) * num_se;

	/* Calculate the minimum size. */
	unsigned min_esgs_ring_size = align(es->esgs_itemsize * gs_vertex_reuse *
					    wave_size, alignment);

	/* These are recommended sizes, not minimum sizes. */
	unsigned esgs_ring_size = max_gs_waves * 2 * wave_size *
				  es->esgs_itemsize * gs->gs_input_verts_per_prim;
	unsigned gsvs_ring_size = max_gs_waves * 2 * wave_size *
				  gs->max_gsvs_emit_size;

	min_esgs_ring_size = align(min_esgs_ring_size, alignment);
	esgs_ring_size = align(esgs_ring_size, alignment);
	gsvs_ring_size = align(gsvs_ring_size, alignment);

	esgs_ring_size = CLAMP(esgs_ring_size, min_esgs_ring_size, max_size);
	gsvs_ring_size = MIN2(gsvs_ring_size, max_size);

	/* Some rings don't have to be allocated if shaders don't use them.
	 * (e.g. no varyings between ES and GS or GS and VS)
	 *
	 * GFX9 doesn't have the ESGS ring.
	 */
	bool update_esgs = sctx->chip_class <= VI &&
			   esgs_ring_size &&
			   (!sctx->esgs_ring ||
			    sctx->esgs_ring->width0 < esgs_ring_size);
	bool update_gsvs = gsvs_ring_size &&
			   (!sctx->gsvs_ring ||
			    sctx->gsvs_ring->width0 < gsvs_ring_size);

	if (!update_esgs && !update_gsvs)
		return true;

	if (update_esgs) {
		pipe_resource_reference(&sctx->esgs_ring, NULL);
		sctx->esgs_ring =
			pipe_aligned_buffer_create(sctx->b.screen,
						   SI_RESOURCE_FLAG_UNMAPPABLE,
						   PIPE_USAGE_DEFAULT,
						   esgs_ring_size, alignment);
		if (!sctx->esgs_ring)
			return false;
	}

	if (update_gsvs) {
		pipe_resource_reference(&sctx->gsvs_ring, NULL);
		sctx->gsvs_ring =
			pipe_aligned_buffer_create(sctx->b.screen,
						   SI_RESOURCE_FLAG_UNMAPPABLE,
						   PIPE_USAGE_DEFAULT,
						   gsvs_ring_size, alignment);
		if (!sctx->gsvs_ring)
			return false;
	}

	/* Create the "init_config_gs_rings" state. */
	pm4 = CALLOC_STRUCT(si_pm4_state);
	if (!pm4)
		return false;

	if (sctx->chip_class >= CIK) {
		if (sctx->esgs_ring) {
			assert(sctx->chip_class <= VI);
			si_pm4_set_reg(pm4, R_030900_VGT_ESGS_RING_SIZE,
				       sctx->esgs_ring->width0 / 256);
		}
		if (sctx->gsvs_ring)
			si_pm4_set_reg(pm4, R_030904_VGT_GSVS_RING_SIZE,
				       sctx->gsvs_ring->width0 / 256);
	} else {
		if (sctx->esgs_ring)
			si_pm4_set_reg(pm4, R_0088C8_VGT_ESGS_RING_SIZE,
				       sctx->esgs_ring->width0 / 256);
		if (sctx->gsvs_ring)
			si_pm4_set_reg(pm4, R_0088CC_VGT_GSVS_RING_SIZE,
				       sctx->gsvs_ring->width0 / 256);
	}

	/* Set the state. */
	if (sctx->init_config_gs_rings)
		si_pm4_free_state(sctx, sctx->init_config_gs_rings, ~0);
	sctx->init_config_gs_rings = pm4;

	if (!sctx->init_config_has_vgt_flush) {
		si_init_config_add_vgt_flush(sctx);
		si_pm4_upload_indirect_buffer(sctx, sctx->init_config);
	}

	/* Flush the context to re-emit both init_config states. */
	sctx->initial_gfx_cs_size = 0; /* force flush */
	si_flush_gfx_cs(sctx, RADEON_FLUSH_ASYNC_START_NEXT_GFX_IB_NOW, NULL);

	/* Set ring bindings. */
	if (sctx->esgs_ring) {
		assert(sctx->chip_class <= VI);
		si_set_ring_buffer(sctx, SI_ES_RING_ESGS,
				   sctx->esgs_ring, 0, sctx->esgs_ring->width0,
				   true, true, 4, 64, 0);
		si_set_ring_buffer(sctx, SI_GS_RING_ESGS,
				   sctx->esgs_ring, 0, sctx->esgs_ring->width0,
				   false, false, 0, 0, 0);
	}
	if (sctx->gsvs_ring) {
		si_set_ring_buffer(sctx, SI_RING_GSVS,
				   sctx->gsvs_ring, 0, sctx->gsvs_ring->width0,
				   false, false, 0, 0, 0);
	}

	return true;
}

static void si_shader_lock(struct si_shader *shader)
{
	mtx_lock(&shader->selector->mutex);
	if (shader->previous_stage_sel) {
		assert(shader->previous_stage_sel != shader->selector);
		mtx_lock(&shader->previous_stage_sel->mutex);
	}
}

static void si_shader_unlock(struct si_shader *shader)
{
	if (shader->previous_stage_sel)
		mtx_unlock(&shader->previous_stage_sel->mutex);
	mtx_unlock(&shader->selector->mutex);
}

/**
 * @returns 1 if \p sel has been updated to use a new scratch buffer
 *          0 if not
 *          < 0 if there was a failure
 */
static int si_update_scratch_buffer(struct si_context *sctx,
				    struct si_shader *shader)
{
	uint64_t scratch_va = sctx->scratch_buffer->gpu_address;
	int r;

	if (!shader)
		return 0;

	/* This shader doesn't need a scratch buffer */
	if (shader->config.scratch_bytes_per_wave == 0)
		return 0;

	/* Prevent race conditions when updating:
	 * - si_shader::scratch_bo
	 * - si_shader::binary::code
	 * - si_shader::previous_stage::binary::code.
	 */
	si_shader_lock(shader);

	/* This shader is already configured to use the current
	 * scratch buffer. */
	if (shader->scratch_bo == sctx->scratch_buffer) {
		si_shader_unlock(shader);
		return 0;
	}

	assert(sctx->scratch_buffer);

	if (shader->previous_stage)
		si_shader_apply_scratch_relocs(shader->previous_stage, scratch_va);

	si_shader_apply_scratch_relocs(shader, scratch_va);

	/* Replace the shader bo with a new bo that has the relocs applied. */
	r = si_shader_binary_upload(sctx->screen, shader);
	if (r) {
		si_shader_unlock(shader);
		return r;
	}

	/* Update the shader state to use the new shader bo. */
	si_shader_init_pm4_state(sctx->screen, shader);

	r600_resource_reference(&shader->scratch_bo, sctx->scratch_buffer);

	si_shader_unlock(shader);
	return 1;
}

static unsigned si_get_current_scratch_buffer_size(struct si_context *sctx)
{
	return sctx->scratch_buffer ? sctx->scratch_buffer->b.b.width0 : 0;
}

static unsigned si_get_scratch_buffer_bytes_per_wave(struct si_shader *shader)
{
	return shader ? shader->config.scratch_bytes_per_wave : 0;
}

static struct si_shader *si_get_tcs_current(struct si_context *sctx)
{
	if (!sctx->tes_shader.cso)
		return NULL; /* tessellation disabled */

	return sctx->tcs_shader.cso ? sctx->tcs_shader.current :
				      sctx->fixed_func_tcs_shader.current;
}

static unsigned si_get_max_scratch_bytes_per_wave(struct si_context *sctx)
{
	unsigned bytes = 0;

	bytes = MAX2(bytes, si_get_scratch_buffer_bytes_per_wave(sctx->ps_shader.current));
	bytes = MAX2(bytes, si_get_scratch_buffer_bytes_per_wave(sctx->gs_shader.current));
	bytes = MAX2(bytes, si_get_scratch_buffer_bytes_per_wave(sctx->vs_shader.current));
	bytes = MAX2(bytes, si_get_scratch_buffer_bytes_per_wave(sctx->tes_shader.current));

	if (sctx->tes_shader.cso) {
		struct si_shader *tcs = si_get_tcs_current(sctx);

		bytes = MAX2(bytes, si_get_scratch_buffer_bytes_per_wave(tcs));
	}
	return bytes;
}

static bool si_update_scratch_relocs(struct si_context *sctx)
{
	struct si_shader *tcs = si_get_tcs_current(sctx);
	int r;

	/* Update the shaders, so that they are using the latest scratch.
	 * The scratch buffer may have been changed since these shaders were
	 * last used, so we still need to try to update them, even if they
	 * require scratch buffers smaller than the current size.
	 */
	r = si_update_scratch_buffer(sctx, sctx->ps_shader.current);
	if (r < 0)
		return false;
	if (r == 1)
		si_pm4_bind_state(sctx, ps, sctx->ps_shader.current->pm4);

	r = si_update_scratch_buffer(sctx, sctx->gs_shader.current);
	if (r < 0)
		return false;
	if (r == 1)
		si_pm4_bind_state(sctx, gs, sctx->gs_shader.current->pm4);

	r = si_update_scratch_buffer(sctx, tcs);
	if (r < 0)
		return false;
	if (r == 1)
		si_pm4_bind_state(sctx, hs, tcs->pm4);

	/* VS can be bound as LS, ES, or VS. */
	r = si_update_scratch_buffer(sctx, sctx->vs_shader.current);
	if (r < 0)
		return false;
	if (r == 1) {
		if (sctx->tes_shader.current)
			si_pm4_bind_state(sctx, ls, sctx->vs_shader.current->pm4);
		else if (sctx->gs_shader.current)
			si_pm4_bind_state(sctx, es, sctx->vs_shader.current->pm4);
		else
			si_pm4_bind_state(sctx, vs, sctx->vs_shader.current->pm4);
	}

	/* TES can be bound as ES or VS. */
	r = si_update_scratch_buffer(sctx, sctx->tes_shader.current);
	if (r < 0)
		return false;
	if (r == 1) {
		if (sctx->gs_shader.current)
			si_pm4_bind_state(sctx, es, sctx->tes_shader.current->pm4);
		else
			si_pm4_bind_state(sctx, vs, sctx->tes_shader.current->pm4);
	}

	return true;
}

static bool si_update_spi_tmpring_size(struct si_context *sctx)
{
	unsigned current_scratch_buffer_size =
		si_get_current_scratch_buffer_size(sctx);
	unsigned scratch_bytes_per_wave =
		si_get_max_scratch_bytes_per_wave(sctx);
	unsigned scratch_needed_size = scratch_bytes_per_wave *
		sctx->scratch_waves;
	unsigned spi_tmpring_size;

	if (scratch_needed_size > 0) {
		if (scratch_needed_size > current_scratch_buffer_size) {
			/* Create a bigger scratch buffer */
			r600_resource_reference(&sctx->scratch_buffer, NULL);

			sctx->scratch_buffer =
				si_aligned_buffer_create(&sctx->screen->b,
							   SI_RESOURCE_FLAG_UNMAPPABLE,
							   PIPE_USAGE_DEFAULT,
							   scratch_needed_size, 256);
			if (!sctx->scratch_buffer)
				return false;

			si_mark_atom_dirty(sctx, &sctx->atoms.s.scratch_state);
			si_context_add_resource_size(sctx,
						     &sctx->scratch_buffer->b.b);
		}

		if (!si_update_scratch_relocs(sctx))
			return false;
	}

	/* The LLVM shader backend should be reporting aligned scratch_sizes. */
	assert((scratch_needed_size & ~0x3FF) == scratch_needed_size &&
		"scratch size should already be aligned correctly.");

	spi_tmpring_size = S_0286E8_WAVES(sctx->scratch_waves) |
			   S_0286E8_WAVESIZE(scratch_bytes_per_wave >> 10);
	if (spi_tmpring_size != sctx->spi_tmpring_size) {
		sctx->spi_tmpring_size = spi_tmpring_size;
		si_mark_atom_dirty(sctx, &sctx->atoms.s.scratch_state);
	}
	return true;
}

static void si_init_tess_factor_ring(struct si_context *sctx)
{
	assert(!sctx->tess_rings);

	/* The address must be aligned to 2^19, because the shader only
	 * receives the high 13 bits.
	 */
	sctx->tess_rings = pipe_aligned_buffer_create(sctx->b.screen,
						    SI_RESOURCE_FLAG_32BIT,
						    PIPE_USAGE_DEFAULT,
						    sctx->screen->tess_offchip_ring_size +
						    sctx->screen->tess_factor_ring_size,
						    1 << 19);
	if (!sctx->tess_rings)
		return;

	si_init_config_add_vgt_flush(sctx);

	si_pm4_add_bo(sctx->init_config, r600_resource(sctx->tess_rings),
		      RADEON_USAGE_READWRITE, RADEON_PRIO_SHADER_RINGS);

	uint64_t factor_va = r600_resource(sctx->tess_rings)->gpu_address +
			     sctx->screen->tess_offchip_ring_size;

	/* Append these registers to the init config state. */
	if (sctx->chip_class >= CIK) {
		si_pm4_set_reg(sctx->init_config, R_030938_VGT_TF_RING_SIZE,
			       S_030938_SIZE(sctx->screen->tess_factor_ring_size / 4));
		si_pm4_set_reg(sctx->init_config, R_030940_VGT_TF_MEMORY_BASE,
			       factor_va >> 8);
		if (sctx->chip_class >= GFX9)
			si_pm4_set_reg(sctx->init_config, R_030944_VGT_TF_MEMORY_BASE_HI,
				       S_030944_BASE_HI(factor_va >> 40));
		si_pm4_set_reg(sctx->init_config, R_03093C_VGT_HS_OFFCHIP_PARAM,
			       sctx->screen->vgt_hs_offchip_param);
	} else {
		si_pm4_set_reg(sctx->init_config, R_008988_VGT_TF_RING_SIZE,
			       S_008988_SIZE(sctx->screen->tess_factor_ring_size / 4));
		si_pm4_set_reg(sctx->init_config, R_0089B8_VGT_TF_MEMORY_BASE,
			       factor_va >> 8);
		si_pm4_set_reg(sctx->init_config, R_0089B0_VGT_HS_OFFCHIP_PARAM,
		               sctx->screen->vgt_hs_offchip_param);
	}

	/* Flush the context to re-emit the init_config state.
	 * This is done only once in a lifetime of a context.
	 */
	si_pm4_upload_indirect_buffer(sctx, sctx->init_config);
	sctx->initial_gfx_cs_size = 0; /* force flush */
	si_flush_gfx_cs(sctx, RADEON_FLUSH_ASYNC_START_NEXT_GFX_IB_NOW, NULL);
}

static void si_update_vgt_shader_config(struct si_context *sctx)
{
	/* Calculate the index of the config.
	 * 0 = VS, 1 = VS+GS, 2 = VS+Tess, 3 = VS+Tess+GS */
	unsigned index = 2*!!sctx->tes_shader.cso + !!sctx->gs_shader.cso;
	struct si_pm4_state **pm4 = &sctx->vgt_shader_config[index];

	if (!*pm4) {
		uint32_t stages = 0;

		*pm4 = CALLOC_STRUCT(si_pm4_state);

		if (sctx->tes_shader.cso) {
			stages |= S_028B54_LS_EN(V_028B54_LS_STAGE_ON) |
				  S_028B54_HS_EN(1) | S_028B54_DYNAMIC_HS(1);

			if (sctx->gs_shader.cso)
				stages |= S_028B54_ES_EN(V_028B54_ES_STAGE_DS) |
					  S_028B54_GS_EN(1) |
				          S_028B54_VS_EN(V_028B54_VS_STAGE_COPY_SHADER);
			else
				stages |= S_028B54_VS_EN(V_028B54_VS_STAGE_DS);
		} else if (sctx->gs_shader.cso) {
			stages |= S_028B54_ES_EN(V_028B54_ES_STAGE_REAL) |
				  S_028B54_GS_EN(1) |
			          S_028B54_VS_EN(V_028B54_VS_STAGE_COPY_SHADER);
		}

		if (sctx->chip_class >= GFX9)
			stages |= S_028B54_MAX_PRIMGRP_IN_WAVE(2);

		si_pm4_set_reg(*pm4, R_028B54_VGT_SHADER_STAGES_EN, stages);
	}
	si_pm4_bind_state(sctx, vgt_shader_config, *pm4);
}

bool si_update_shaders(struct si_context *sctx)
{
	struct pipe_context *ctx = (struct pipe_context*)sctx;
	struct si_compiler_ctx_state compiler_state;
	struct si_state_rasterizer *rs = sctx->queued.named.rasterizer;
	struct si_shader *old_vs = si_get_vs_state(sctx);
	bool old_clip_disable = old_vs ? old_vs->key.opt.clip_disable : false;
	struct si_shader *old_ps = sctx->ps_shader.current;
	unsigned old_spi_shader_col_format =
		old_ps ? old_ps->key.part.ps.epilog.spi_shader_col_format : 0;
	int r;

	compiler_state.compiler = &sctx->compiler;
	compiler_state.debug = sctx->debug;
	compiler_state.is_debug_context = sctx->is_debug;

	/* Update stages before GS. */
	if (sctx->tes_shader.cso) {
		if (!sctx->tess_rings) {
			si_init_tess_factor_ring(sctx);
			if (!sctx->tess_rings)
				return false;
		}

		/* VS as LS */
		if (sctx->chip_class <= VI) {
			r = si_shader_select(ctx, &sctx->vs_shader,
					     &compiler_state);
			if (r)
				return false;
			si_pm4_bind_state(sctx, ls, sctx->vs_shader.current->pm4);
		}

		if (sctx->tcs_shader.cso) {
			r = si_shader_select(ctx, &sctx->tcs_shader,
					     &compiler_state);
			if (r)
				return false;
			si_pm4_bind_state(sctx, hs, sctx->tcs_shader.current->pm4);
		} else {
			if (!sctx->fixed_func_tcs_shader.cso) {
				sctx->fixed_func_tcs_shader.cso =
					si_create_fixed_func_tcs(sctx);
				if (!sctx->fixed_func_tcs_shader.cso)
					return false;
			}

			r = si_shader_select(ctx, &sctx->fixed_func_tcs_shader,
					     &compiler_state);
			if (r)
				return false;
			si_pm4_bind_state(sctx, hs,
					  sctx->fixed_func_tcs_shader.current->pm4);
		}

		if (sctx->gs_shader.cso) {
			/* TES as ES */
			if (sctx->chip_class <= VI) {
				r = si_shader_select(ctx, &sctx->tes_shader,
						     &compiler_state);
				if (r)
					return false;
				si_pm4_bind_state(sctx, es, sctx->tes_shader.current->pm4);
			}
		} else {
			/* TES as VS */
			r = si_shader_select(ctx, &sctx->tes_shader,
					     &compiler_state);
			if (r)
				return false;
			si_pm4_bind_state(sctx, vs, sctx->tes_shader.current->pm4);
		}
	} else if (sctx->gs_shader.cso) {
		if (sctx->chip_class <= VI) {
			/* VS as ES */
			r = si_shader_select(ctx, &sctx->vs_shader,
					     &compiler_state);
			if (r)
				return false;
			si_pm4_bind_state(sctx, es, sctx->vs_shader.current->pm4);

			si_pm4_bind_state(sctx, ls, NULL);
			si_pm4_bind_state(sctx, hs, NULL);
		}
	} else {
		/* VS as VS */
		r = si_shader_select(ctx, &sctx->vs_shader, &compiler_state);
		if (r)
			return false;
		si_pm4_bind_state(sctx, vs, sctx->vs_shader.current->pm4);
		si_pm4_bind_state(sctx, ls, NULL);
		si_pm4_bind_state(sctx, hs, NULL);
	}

	/* Update GS. */
	if (sctx->gs_shader.cso) {
		r = si_shader_select(ctx, &sctx->gs_shader, &compiler_state);
		if (r)
			return false;
		si_pm4_bind_state(sctx, gs, sctx->gs_shader.current->pm4);
		si_pm4_bind_state(sctx, vs, sctx->gs_shader.cso->gs_copy_shader->pm4);

		if (!si_update_gs_ring_buffers(sctx))
			return false;
	} else {
		si_pm4_bind_state(sctx, gs, NULL);
		if (sctx->chip_class <= VI)
			si_pm4_bind_state(sctx, es, NULL);
	}

	si_update_vgt_shader_config(sctx);

	if (old_clip_disable != si_get_vs_state(sctx)->key.opt.clip_disable)
		si_mark_atom_dirty(sctx, &sctx->atoms.s.clip_regs);

	if (sctx->ps_shader.cso) {
		unsigned db_shader_control;

		r = si_shader_select(ctx, &sctx->ps_shader, &compiler_state);
		if (r)
			return false;
		si_pm4_bind_state(sctx, ps, sctx->ps_shader.current->pm4);

		db_shader_control =
			sctx->ps_shader.cso->db_shader_control |
			S_02880C_KILL_ENABLE(si_get_alpha_test_func(sctx) != PIPE_FUNC_ALWAYS);

		if (si_pm4_state_changed(sctx, ps) || si_pm4_state_changed(sctx, vs) ||
		    sctx->sprite_coord_enable != rs->sprite_coord_enable ||
		    sctx->flatshade != rs->flatshade) {
			sctx->sprite_coord_enable = rs->sprite_coord_enable;
			sctx->flatshade = rs->flatshade;
			si_mark_atom_dirty(sctx, &sctx->atoms.s.spi_map);
		}

		if (sctx->screen->rbplus_allowed &&
		    si_pm4_state_changed(sctx, ps) &&
		    (!old_ps ||
		     old_spi_shader_col_format !=
		     sctx->ps_shader.current->key.part.ps.epilog.spi_shader_col_format))
			si_mark_atom_dirty(sctx, &sctx->atoms.s.cb_render_state);

		if (sctx->ps_db_shader_control != db_shader_control) {
			sctx->ps_db_shader_control = db_shader_control;
			si_mark_atom_dirty(sctx, &sctx->atoms.s.db_render_state);
			if (sctx->screen->dpbb_allowed)
				si_mark_atom_dirty(sctx, &sctx->atoms.s.dpbb_state);
		}

		if (sctx->smoothing_enabled != sctx->ps_shader.current->key.part.ps.epilog.poly_line_smoothing) {
			sctx->smoothing_enabled = sctx->ps_shader.current->key.part.ps.epilog.poly_line_smoothing;
			si_mark_atom_dirty(sctx, &sctx->atoms.s.msaa_config);

			if (sctx->chip_class == SI)
				si_mark_atom_dirty(sctx, &sctx->atoms.s.db_render_state);

			if (sctx->framebuffer.nr_samples <= 1)
				si_mark_atom_dirty(sctx, &sctx->atoms.s.msaa_sample_locs);
		}
	}

	if (si_pm4_state_enabled_and_changed(sctx, ls) ||
	    si_pm4_state_enabled_and_changed(sctx, hs) ||
	    si_pm4_state_enabled_and_changed(sctx, es) ||
	    si_pm4_state_enabled_and_changed(sctx, gs) ||
	    si_pm4_state_enabled_and_changed(sctx, vs) ||
	    si_pm4_state_enabled_and_changed(sctx, ps)) {
		if (!si_update_spi_tmpring_size(sctx))
			return false;
	}

	if (sctx->chip_class >= CIK) {
		if (si_pm4_state_enabled_and_changed(sctx, ls))
			sctx->prefetch_L2_mask |= SI_PREFETCH_LS;
		else if (!sctx->queued.named.ls)
			sctx->prefetch_L2_mask &= ~SI_PREFETCH_LS;

		if (si_pm4_state_enabled_and_changed(sctx, hs))
			sctx->prefetch_L2_mask |= SI_PREFETCH_HS;
		else if (!sctx->queued.named.hs)
			sctx->prefetch_L2_mask &= ~SI_PREFETCH_HS;

		if (si_pm4_state_enabled_and_changed(sctx, es))
			sctx->prefetch_L2_mask |= SI_PREFETCH_ES;
		else if (!sctx->queued.named.es)
			sctx->prefetch_L2_mask &= ~SI_PREFETCH_ES;

		if (si_pm4_state_enabled_and_changed(sctx, gs))
			sctx->prefetch_L2_mask |= SI_PREFETCH_GS;
		else if (!sctx->queued.named.gs)
			sctx->prefetch_L2_mask &= ~SI_PREFETCH_GS;

		if (si_pm4_state_enabled_and_changed(sctx, vs))
			sctx->prefetch_L2_mask |= SI_PREFETCH_VS;
		else if (!sctx->queued.named.vs)
			sctx->prefetch_L2_mask &= ~SI_PREFETCH_VS;

		if (si_pm4_state_enabled_and_changed(sctx, ps))
			sctx->prefetch_L2_mask |= SI_PREFETCH_PS;
		else if (!sctx->queued.named.ps)
			sctx->prefetch_L2_mask &= ~SI_PREFETCH_PS;
	}

	sctx->do_update_shaders = false;
	return true;
}

static void si_emit_scratch_state(struct si_context *sctx)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;

	radeon_set_context_reg(cs, R_0286E8_SPI_TMPRING_SIZE,
			       sctx->spi_tmpring_size);

	if (sctx->scratch_buffer) {
		radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
				      sctx->scratch_buffer, RADEON_USAGE_READWRITE,
				      RADEON_PRIO_SCRATCH_BUFFER);
	}
}

void si_init_shader_functions(struct si_context *sctx)
{
	sctx->atoms.s.spi_map.emit = si_emit_spi_map;
	sctx->atoms.s.scratch_state.emit = si_emit_scratch_state;

	sctx->b.create_vs_state = si_create_shader_selector;
	sctx->b.create_tcs_state = si_create_shader_selector;
	sctx->b.create_tes_state = si_create_shader_selector;
	sctx->b.create_gs_state = si_create_shader_selector;
	sctx->b.create_fs_state = si_create_shader_selector;

	sctx->b.bind_vs_state = si_bind_vs_shader;
	sctx->b.bind_tcs_state = si_bind_tcs_shader;
	sctx->b.bind_tes_state = si_bind_tes_shader;
	sctx->b.bind_gs_state = si_bind_gs_shader;
	sctx->b.bind_fs_state = si_bind_ps_shader;

	sctx->b.delete_vs_state = si_delete_shader_selector;
	sctx->b.delete_tcs_state = si_delete_shader_selector;
	sctx->b.delete_tes_state = si_delete_shader_selector;
	sctx->b.delete_gs_state = si_delete_shader_selector;
	sctx->b.delete_fs_state = si_delete_shader_selector;
}
