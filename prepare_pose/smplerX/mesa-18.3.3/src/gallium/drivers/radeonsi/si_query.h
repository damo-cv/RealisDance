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

#ifndef SI_QUERY_H
#define SI_QUERY_H

#include "util/u_threaded_context.h"

struct pipe_context;
struct pipe_query;
struct pipe_resource;

struct si_screen;
struct si_context;
struct si_query;
struct si_query_hw;
struct r600_resource;

enum {
	SI_QUERY_DRAW_CALLS = PIPE_QUERY_DRIVER_SPECIFIC,
	SI_QUERY_DECOMPRESS_CALLS,
	SI_QUERY_MRT_DRAW_CALLS,
	SI_QUERY_PRIM_RESTART_CALLS,
	SI_QUERY_SPILL_DRAW_CALLS,
	SI_QUERY_COMPUTE_CALLS,
	SI_QUERY_SPILL_COMPUTE_CALLS,
	SI_QUERY_DMA_CALLS,
	SI_QUERY_CP_DMA_CALLS,
	SI_QUERY_NUM_VS_FLUSHES,
	SI_QUERY_NUM_PS_FLUSHES,
	SI_QUERY_NUM_CS_FLUSHES,
	SI_QUERY_NUM_CB_CACHE_FLUSHES,
	SI_QUERY_NUM_DB_CACHE_FLUSHES,
	SI_QUERY_NUM_L2_INVALIDATES,
	SI_QUERY_NUM_L2_WRITEBACKS,
	SI_QUERY_NUM_RESIDENT_HANDLES,
	SI_QUERY_TC_OFFLOADED_SLOTS,
	SI_QUERY_TC_DIRECT_SLOTS,
	SI_QUERY_TC_NUM_SYNCS,
	SI_QUERY_CS_THREAD_BUSY,
	SI_QUERY_GALLIUM_THREAD_BUSY,
	SI_QUERY_REQUESTED_VRAM,
	SI_QUERY_REQUESTED_GTT,
	SI_QUERY_MAPPED_VRAM,
	SI_QUERY_MAPPED_GTT,
	SI_QUERY_BUFFER_WAIT_TIME,
	SI_QUERY_NUM_MAPPED_BUFFERS,
	SI_QUERY_NUM_GFX_IBS,
	SI_QUERY_NUM_SDMA_IBS,
	SI_QUERY_GFX_BO_LIST_SIZE,
	SI_QUERY_GFX_IB_SIZE,
	SI_QUERY_NUM_BYTES_MOVED,
	SI_QUERY_NUM_EVICTIONS,
	SI_QUERY_NUM_VRAM_CPU_PAGE_FAULTS,
	SI_QUERY_VRAM_USAGE,
	SI_QUERY_VRAM_VIS_USAGE,
	SI_QUERY_GTT_USAGE,
	SI_QUERY_GPU_TEMPERATURE,
	SI_QUERY_CURRENT_GPU_SCLK,
	SI_QUERY_CURRENT_GPU_MCLK,
	SI_QUERY_GPU_LOAD,
	SI_QUERY_GPU_SHADERS_BUSY,
	SI_QUERY_GPU_TA_BUSY,
	SI_QUERY_GPU_GDS_BUSY,
	SI_QUERY_GPU_VGT_BUSY,
	SI_QUERY_GPU_IA_BUSY,
	SI_QUERY_GPU_SX_BUSY,
	SI_QUERY_GPU_WD_BUSY,
	SI_QUERY_GPU_BCI_BUSY,
	SI_QUERY_GPU_SC_BUSY,
	SI_QUERY_GPU_PA_BUSY,
	SI_QUERY_GPU_DB_BUSY,
	SI_QUERY_GPU_CP_BUSY,
	SI_QUERY_GPU_CB_BUSY,
	SI_QUERY_GPU_SDMA_BUSY,
	SI_QUERY_GPU_PFP_BUSY,
	SI_QUERY_GPU_MEQ_BUSY,
	SI_QUERY_GPU_ME_BUSY,
	SI_QUERY_GPU_SURF_SYNC_BUSY,
	SI_QUERY_GPU_CP_DMA_BUSY,
	SI_QUERY_GPU_SCRATCH_RAM_BUSY,
	SI_QUERY_NUM_COMPILATIONS,
	SI_QUERY_NUM_SHADERS_CREATED,
	SI_QUERY_BACK_BUFFER_PS_DRAW_RATIO,
	SI_QUERY_NUM_SHADER_CACHE_HITS,
	SI_QUERY_GPIN_ASIC_ID,
	SI_QUERY_GPIN_NUM_SIMD,
	SI_QUERY_GPIN_NUM_RB,
	SI_QUERY_GPIN_NUM_SPI,
	SI_QUERY_GPIN_NUM_SE,
	SI_QUERY_TIME_ELAPSED_SDMA,
	SI_QUERY_TIME_ELAPSED_SDMA_SI, /* emulated, measured on the CPU */

	SI_QUERY_FIRST_PERFCOUNTER = PIPE_QUERY_DRIVER_SPECIFIC + 100,
};

enum {
	SI_QUERY_GROUP_GPIN = 0,
	SI_NUM_SW_QUERY_GROUPS
};

struct si_query_ops {
	void (*destroy)(struct si_screen *, struct si_query *);
	bool (*begin)(struct si_context *, struct si_query *);
	bool (*end)(struct si_context *, struct si_query *);
	bool (*get_result)(struct si_context *,
			   struct si_query *, bool wait,
			   union pipe_query_result *result);
	void (*get_result_resource)(struct si_context *,
				    struct si_query *, bool wait,
				    enum pipe_query_value_type result_type,
				    int index,
				    struct pipe_resource *resource,
				    unsigned offset);
};

struct si_query {
	struct threaded_query b;
	struct si_query_ops *ops;

	/* The type of query */
	unsigned type;
};

enum {
	SI_QUERY_HW_FLAG_NO_START = (1 << 0),
	/* gap */
	/* whether begin_query doesn't clear the result */
	SI_QUERY_HW_FLAG_BEGIN_RESUMES = (1 << 2),
};

struct si_query_hw_ops {
	bool (*prepare_buffer)(struct si_screen *,
			       struct si_query_hw *,
			       struct r600_resource *);
	void (*emit_start)(struct si_context *,
			   struct si_query_hw *,
			   struct r600_resource *buffer, uint64_t va);
	void (*emit_stop)(struct si_context *,
			  struct si_query_hw *,
			  struct r600_resource *buffer, uint64_t va);
	void (*clear_result)(struct si_query_hw *, union pipe_query_result *);
	void (*add_result)(struct si_screen *screen,
			   struct si_query_hw *, void *buffer,
			   union pipe_query_result *result);
};

struct si_query_buffer {
	/* The buffer where query results are stored. */
	struct r600_resource		*buf;
	/* Offset of the next free result after current query data */
	unsigned			results_end;
	/* If a query buffer is full, a new buffer is created and the old one
	 * is put in here. When we calculate the result, we sum up the samples
	 * from all buffers. */
	struct si_query_buffer	*previous;
};

struct si_query_hw {
	struct si_query b;
	struct si_query_hw_ops *ops;
	unsigned flags;

	/* The query buffer and how many results are in it. */
	struct si_query_buffer buffer;
	/* Size of the result in memory for both begin_query and end_query,
	 * this can be one or two numbers, or it could even be a size of a structure. */
	unsigned result_size;
	/* The number of dwords for end_query. */
	unsigned num_cs_dw_end;
	/* Linked list of queries */
	struct list_head list;
	/* For transform feedback: which stream the query is for */
	unsigned stream;

	/* Workaround via compute shader */
	struct r600_resource *workaround_buf;
	unsigned workaround_offset;
};

bool si_query_hw_init(struct si_screen *sscreen,
		      struct si_query_hw *query);
void si_query_hw_destroy(struct si_screen *sscreen,
			 struct si_query *rquery);
bool si_query_hw_begin(struct si_context *sctx,
		       struct si_query *rquery);
bool si_query_hw_end(struct si_context *sctx,
		     struct si_query *rquery);
bool si_query_hw_get_result(struct si_context *sctx,
			    struct si_query *rquery,
			    bool wait,
			    union pipe_query_result *result);

/* Performance counters */
enum {
	/* This block is part of the shader engine */
	SI_PC_BLOCK_SE = (1 << 0),

	/* Expose per-instance groups instead of summing all instances (within
	 * an SE). */
	SI_PC_BLOCK_INSTANCE_GROUPS = (1 << 1),

	/* Expose per-SE groups instead of summing instances across SEs. */
	SI_PC_BLOCK_SE_GROUPS = (1 << 2),

	/* Shader block */
	SI_PC_BLOCK_SHADER = (1 << 3),

	/* Non-shader block with perfcounters windowed by shaders. */
	SI_PC_BLOCK_SHADER_WINDOWED = (1 << 4),
};

/* Describes a hardware block with performance counters. Multiple instances of
 * each block, possibly per-SE, may exist on the chip. Depending on the block
 * and on the user's configuration, we either
 *  (a) expose every instance as a performance counter group,
 *  (b) expose a single performance counter group that reports the sum over all
 *      instances, or
 *  (c) expose one performance counter group per instance, but summed over all
 *      shader engines.
 */
struct si_perfcounter_block {
	const char *basename;
	unsigned flags;
	unsigned num_counters;
	unsigned num_selectors;
	unsigned num_instances;

	unsigned num_groups;
	char *group_names;
	unsigned group_name_stride;

	char *selector_names;
	unsigned selector_name_stride;

	void *data;
};

struct si_perfcounters {
	unsigned num_groups;
	unsigned num_blocks;
	struct si_perfcounter_block *blocks;

	unsigned num_stop_cs_dwords;
	unsigned num_instance_cs_dwords;

	unsigned num_shader_types;
	const char * const *shader_type_suffixes;
	const unsigned *shader_type_bits;

	void (*emit_instance)(struct si_context *,
			      int se, int instance);
	void (*emit_shaders)(struct si_context *, unsigned shaders);
	void (*emit_select)(struct si_context *,
			    struct si_perfcounter_block *,
			    unsigned count, unsigned *selectors);
	void (*emit_start)(struct si_context *,
			  struct r600_resource *buffer, uint64_t va);
	void (*emit_stop)(struct si_context *,
			  struct r600_resource *buffer, uint64_t va);
	void (*emit_read)(struct si_context *,
			  struct si_perfcounter_block *,
			  unsigned count, unsigned *selectors,
			  struct r600_resource *buffer, uint64_t va);

	void (*cleanup)(struct si_screen *);

	bool separate_se;
	bool separate_instance;
};

struct pipe_query *si_create_batch_query(struct pipe_context *ctx,
					 unsigned num_queries,
					 unsigned *query_types);

int si_get_perfcounter_info(struct si_screen *,
			    unsigned index,
			    struct pipe_driver_query_info *info);
int si_get_perfcounter_group_info(struct si_screen *,
				  unsigned index,
				  struct pipe_driver_query_group_info *info);

bool si_perfcounters_init(struct si_perfcounters *, unsigned num_blocks);
void si_perfcounters_add_block(struct si_screen *,
			       struct si_perfcounters *,
			       const char *name, unsigned flags,
			       unsigned counters, unsigned selectors,
			       unsigned instances, void *data);
void si_perfcounters_do_destroy(struct si_perfcounters *);
void si_query_hw_reset_buffers(struct si_context *sctx,
			       struct si_query_hw *query);

struct si_qbo_state {
	void *saved_compute;
	struct pipe_constant_buffer saved_const0;
	struct pipe_shader_buffer saved_ssbo[3];
};

#endif /* SI_QUERY_H */
