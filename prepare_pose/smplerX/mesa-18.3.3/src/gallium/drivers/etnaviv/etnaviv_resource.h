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

#ifndef H_ETNAVIV_RESOURCE
#define H_ETNAVIV_RESOURCE

#include "etnaviv_internal.h"
#include "etnaviv_tiling.h"
#include "pipe/p_state.h"
#include "util/list.h"

struct pipe_screen;

struct etna_resource_level {
   unsigned width, padded_width; /* in pixels */
   unsigned height, padded_height; /* in samples */
   unsigned offset; /* offset into memory area */
   uint32_t stride; /* row stride */
   uint32_t layer_stride; /* layer stride */
   unsigned size; /* total size of memory area */

   uint32_t ts_offset;
   uint32_t ts_layer_stride;
   uint32_t ts_size;
   uint32_t clear_value; /* clear value of resource level (mainly for TS) */
   bool ts_valid;
};

/* status of queued up but not flushed reads and write operations.
 * In _transfer_map() we need to know if queued up rendering needs
 * to be flushed to preserve the order of cpu and gpu access. */
enum etna_resource_status {
   ETNA_PENDING_WRITE = 0x01,
   ETNA_PENDING_READ = 0x02,
};

struct etna_resource {
   struct pipe_resource base;
   struct renderonly_scanout *scanout;
   uint32_t seqno;
   uint32_t flush_seqno;

   /* only lod 0 used for non-texture buffers */
   /* Layout for surface (tiled, multitiled, split tiled, ...) */
   enum etna_surface_layout layout;
   /* Horizontal alignment for texture unit (TEXTURE_HALIGN_*) */
   unsigned halign;
   struct etna_bo *bo; /* Surface video memory */
   struct etna_bo *ts_bo; /* Tile status video memory */

   struct etna_resource_level levels[ETNA_NUM_LOD];

   /* When we are rendering to a texture, we need a differently tiled resource */
   struct pipe_resource *texture;
   /*
    * If imported resources have an render/sampler incompatible tiling, we keep
    * them as an external resource, which is blitted as needed.
    */
   struct pipe_resource *external;

   enum etna_resource_status status;

   /* resources accessed by queued but not flushed draws are tracked
    * in the used_resources list. */
   struct list_head list;
   struct etna_context *pending_ctx;
};

/* returns TRUE if a is newer than b */
static inline bool
etna_resource_newer(struct etna_resource *a, struct etna_resource *b)
{
   return (int)(a->seqno - b->seqno) > 0;
}

/* returns TRUE if a is older than b */
static inline bool
etna_resource_older(struct etna_resource *a, struct etna_resource *b)
{
   return (int)(a->seqno - b->seqno) < 0;
}

/* returns TRUE if a resource has a TS, and it is valid for at least one level */
bool
etna_resource_has_valid_ts(struct etna_resource *res);

/* returns TRUE if the resource needs a resolve to itself */
static inline bool
etna_resource_needs_flush(struct etna_resource *res)
{
   return etna_resource_has_valid_ts(res) && ((int)(res->seqno - res->flush_seqno) > 0);
}

/* is the resource only used on the sampler? */
static inline bool
etna_resource_sampler_only(const struct pipe_resource *pres)
{
   return (pres->bind & (PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET |
                         PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_BLENDABLE)) ==
          PIPE_BIND_SAMPLER_VIEW;
}

static inline struct etna_resource *
etna_resource(struct pipe_resource *p)
{
   return (struct etna_resource *)p;
}

void
etna_resource_used(struct etna_context *ctx, struct pipe_resource *prsc,
                   enum etna_resource_status status);

static inline void
resource_read(struct etna_context *ctx, struct pipe_resource *prsc)
{
   etna_resource_used(ctx, prsc, ETNA_PENDING_READ);
}

static inline void
resource_written(struct etna_context *ctx, struct pipe_resource *prsc)
{
   etna_resource_used(ctx, prsc, ETNA_PENDING_WRITE);
}

/* Allocate Tile Status for an etna resource.
 * Tile status is a cache of the clear status per tile. This means a smaller
 * surface has to be cleared which is faster.
 * This is also called "fast clear". */
bool
etna_screen_resource_alloc_ts(struct pipe_screen *pscreen,
                              struct etna_resource *prsc);

struct pipe_resource *
etna_resource_alloc(struct pipe_screen *pscreen, unsigned layout,
                    uint64_t modifier, const struct pipe_resource *templat);

void
etna_resource_screen_init(struct pipe_screen *pscreen);

#endif
