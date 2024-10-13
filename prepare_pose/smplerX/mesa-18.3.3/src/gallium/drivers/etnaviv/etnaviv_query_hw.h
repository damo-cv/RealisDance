/*
 * Copyright (c) 2017 Etnaviv Project
 * Copyright (C) 2017 Zodiac Inflight Innovations
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
 *    Rob Clark <robclark@freedesktop.org>
 *    Christian Gmeiner <christian.gmeiner@gmail.com>
 */

#ifndef H_ETNAVIV_QUERY_HW
#define H_ETNAVIV_QUERY_HW

#include "etnaviv_query.h"

struct etna_hw_query;

struct etna_hw_sample_provider {
   void (*start)(struct etna_hw_query *hq, struct etna_context *ctx);
   void (*stop)(struct etna_hw_query *hq, struct etna_context *ctx);
   void (*suspend)(struct etna_hw_query *hq, struct etna_context *ctx);
   void (*resume)(struct etna_hw_query *hq, struct etna_context *ctx);

   void (*result)(struct etna_hw_query *hq, void *buf,
           union pipe_query_result *result);
};

struct etna_hw_query {
   struct etna_query base;

   struct pipe_resource *prsc;
   unsigned samples;        /* number of samples stored in resource */
   unsigned no_wait_cnt;    /* see etna_hw_get_query_result() */
   struct list_head node;   /* list-node in ctx->active_hw_queries */

   const struct etna_hw_sample_provider *provider;
};

static inline struct etna_hw_query *
etna_hw_query(struct etna_query *q)
{
   return (struct etna_hw_query *)q;
}

struct etna_query *
etna_hw_create_query(struct etna_context *ctx, unsigned query_type);

static inline void
etna_hw_query_suspend(struct etna_hw_query *hq, struct etna_context *ctx)
{
   const struct etna_hw_sample_provider *p = hq->provider;

   if (!hq->base.active)
      return;

   p->suspend(hq, ctx);
}

static inline void
etna_hw_query_resume(struct etna_hw_query *hq, struct etna_context *ctx)
{
   const struct etna_hw_sample_provider *p = hq->provider;

   if (!hq->base.active)
      return;

   p->resume(hq, ctx);
}

#endif
