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

#ifndef H_ETNAVIV_TEXTURE_PLAIN
#define H_ETNAVIV_TEXTURE_PLAIN

#include <etnaviv_drmif.h>

#include "etnaviv_texture.h"

#include "pipe/p_context.h"
#include "pipe/p_state.h"

#include "hw/state_3d.xml.h"

struct etna_sampler_state {
   struct pipe_sampler_state base;

   /* sampler offset +4*sampler, interleave when committing state */
   uint32_t TE_SAMPLER_CONFIG0;
   uint32_t TE_SAMPLER_CONFIG1;
   uint32_t TE_SAMPLER_LOD_CONFIG;
   unsigned min_lod, max_lod;
};

static inline struct etna_sampler_state *
etna_sampler_state(struct pipe_sampler_state *samp)
{
   return (struct etna_sampler_state *)samp;
}

struct etna_sampler_view {
   struct pipe_sampler_view base;

   /* sampler offset +4*sampler, interleave when committing state */
   uint32_t TE_SAMPLER_CONFIG0;
   uint32_t TE_SAMPLER_CONFIG0_MASK;
   uint32_t TE_SAMPLER_CONFIG1;
   uint32_t TE_SAMPLER_SIZE;
   uint32_t TE_SAMPLER_LOG_SIZE;
   uint32_t TE_SAMPLER_ASTC0;
   struct etna_reloc TE_SAMPLER_LOD_ADDR[VIVS_TE_SAMPLER_LOD_ADDR__LEN];
   unsigned min_lod, max_lod; /* 5.5 fixp */

   struct etna_sampler_ts ts;
};

static inline struct etna_sampler_view *
etna_sampler_view(struct pipe_sampler_view *view)
{
   return (struct etna_sampler_view *)view;
}

/* Initialize context for "plain" (non-descriptor, state-based) texture views
 * and descriptors */
void
etna_texture_state_init(struct pipe_context *pctx);

#endif
