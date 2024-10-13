/*
 * Copyright (c) 2012-2015 Etnaviv Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Wladimir J. van der Laan <laanwj@gmail.com>
 */

#ifndef H_ETNAVIV_COMPILER
#define H_ETNAVIV_COMPILER

#include "etnaviv_context.h"
#include "etnaviv_internal.h"
#include "etnaviv_shader.h"
#include "pipe/p_compiler.h"
#include "pipe/p_shader_tokens.h"

/* XXX some of these are pretty arbitrary limits, may be better to switch
 * to dynamic allocation at some point.
 */
#define ETNA_MAX_TEMPS (64) /* max temp register count of all Vivante hw */
#define ETNA_MAX_TOKENS (2048)
#define ETNA_MAX_IMM (1024) /* max const+imm in 32-bit words */
#define ETNA_MAX_DECL (2048) /* max declarations */
#define ETNA_MAX_DEPTH (32)
#define ETNA_MAX_INSTRUCTIONS (2048)

/* compiler output per input/output */
struct etna_shader_inout {
   int reg; /* native register */
   struct tgsi_declaration_semantic semantic; /* tgsi semantic name and index */
   int num_components;
};

struct etna_shader_io_file {
   size_t num_reg;
   struct etna_shader_inout reg[ETNA_NUM_INPUTS];
};

/* shader object, for linking */
struct etna_shader_variant {
   uint32_t id; /* for debug */

   uint processor; /* TGSI_PROCESSOR_... */
   uint32_t code_size; /* code size in uint32 words */
   uint32_t *code;
   unsigned num_loops;
   unsigned num_temps;

   struct etna_shader_uniform_info uniforms;

   /* ETNA_DIRTY_* flags that, when set in context dirty, mean that the
    * uniforms have to get (partial) reloaded. */
   uint32_t uniforms_dirty_bits;

   /* inputs (for linking) for fs, the inputs must be in register 1..N */
   struct etna_shader_io_file infile;

   /* outputs (for linking) */
   struct etna_shader_io_file outfile;

   /* index into outputs (for linking) */
   int output_count_per_semantic[TGSI_SEMANTIC_COUNT];
   struct etna_shader_inout * *output_per_semantic_list; /* list of pointers to outputs */
   struct etna_shader_inout **output_per_semantic[TGSI_SEMANTIC_COUNT];

   /* special outputs (vs only) */
   int vs_pos_out_reg; /* VS position output */
   int vs_pointsize_out_reg; /* VS point size output */
   uint32_t vs_load_balancing;

   /* special outputs (ps only) */
   int ps_color_out_reg; /* color output register */
   int ps_depth_out_reg; /* depth output register */

   /* unknown input property (XX_INPUT_COUNT, field UNK8) */
   uint32_t input_count_unk8;

   /* shader is larger than GPU instruction limit, thus needs icache */
   bool needs_icache;

   /* shader variants form a linked list */
   struct etna_shader_variant *next;

   /* replicated here to avoid passing extra ptrs everywhere */
   struct etna_shader *shader;
   struct etna_shader_key key;

   struct etna_bo *bo; /* cached code memory bo handle (for icache) */
};

struct etna_varying {
   uint32_t pa_attributes;
   uint8_t num_components;
   uint8_t use[4];
   uint8_t reg;
};

struct etna_shader_link_info {
   /* each PS input is annotated with the VS output reg */
   unsigned num_varyings;
   struct etna_varying varyings[ETNA_NUM_INPUTS];
   int pcoord_varying_comp_ofs;
};

bool
etna_compile_shader(struct etna_shader_variant *shader);

void
etna_dump_shader(const struct etna_shader_variant *shader);

bool
etna_link_shader(struct etna_shader_link_info *info,
                 const struct etna_shader_variant *vs, const struct etna_shader_variant *fs);

void
etna_destroy_shader(struct etna_shader_variant *shader);

#endif
