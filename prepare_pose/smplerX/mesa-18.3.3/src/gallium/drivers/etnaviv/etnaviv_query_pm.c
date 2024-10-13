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
 *    Christian Gmeiner <christian.gmeiner@gmail.com>
 */

#include "util/u_inlines.h"
#include "util/u_memory.h"

#include "etnaviv_context.h"
#include "etnaviv_query_pm.h"
#include "etnaviv_screen.h"

struct etna_perfmon_source
{
   const char *domain;
   const char *signal;
};

struct etna_perfmon_config
{
   const char *name;
   unsigned type;
   unsigned group_id;
   const struct etna_perfmon_source *source;
};

static const char *group_names[] = {
   [ETNA_QUERY_HI_GROUP_ID] = "HI",
   [ETNA_QUERY_PE_GROUP_ID] = "PE",
   [ETNA_QUERY_SH_GROUP_ID] = "SH",
   [ETNA_QUERY_PA_GROUP_ID] = "PA",
   [ETNA_QUERY_SE_GROUP_ID] = "SE",
   [ETNA_QUERY_RA_GROUP_ID] = "RA",
   [ETNA_QUERY_TX_GROUP_ID] = "TX",
   [ETNA_QUERY_MC_GROUP_ID] = "MC",
};

static const struct etna_perfmon_config query_config[] = {
   {
      .name = "hi-total-cycles",
      .type = ETNA_QUERY_HI_TOTAL_CYCLES,
      .group_id = ETNA_QUERY_HI_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "HI", "TOTAL_CYCLES" }
      }
   },
   {
      .name = "hi-idle-cycles",
      .type = ETNA_QUERY_HI_IDLE_CYCLES,
      .group_id = ETNA_QUERY_HI_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "HI", "IDLE_CYCLES" }
      }
   },
   {
      .name = "hi-axi-cycles-read-request-stalled",
      .type = ETNA_QUERY_HI_AXI_CYCLES_READ_REQUEST_STALLED,
      .group_id = ETNA_QUERY_HI_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "HI", "AXI_CYCLES_READ_REQUEST_STALLED" }
      }
   },
   {
      .name = "hi-axi-cycles-write-request-stalled",
      .type = ETNA_QUERY_HI_AXI_CYCLES_WRITE_REQUEST_STALLED,
      .group_id = ETNA_QUERY_HI_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "HI", "AXI_CYCLES_WRITE_REQUEST_STALLED" }
      }
   },
   {
      .name = "hi-axi-cycles-write-data-stalled",
      .type = ETNA_QUERY_HI_AXI_CYCLES_WRITE_DATA_STALLED,
      .group_id = ETNA_QUERY_HI_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "HI", "AXI_CYCLES_WRITE_DATA_STALLED" }
      }
   },
   {
      .name = "pe-pixel-count-killed-by-color-pipe",
      .type = ETNA_QUERY_PE_PIXEL_COUNT_KILLED_BY_COLOR_PIPE,
      .group_id = ETNA_QUERY_PE_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "PE", "PIXEL_COUNT_KILLED_BY_COLOR_PIPE" }
      }
   },
   {
      .name = "pe-pixel-count-killed-by-depth-pipe",
      .type = ETNA_QUERY_PE_PIXEL_COUNT_KILLED_BY_DEPTH_PIPE,
      .group_id = ETNA_QUERY_PE_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "PE", "PIXEL_COUNT_KILLED_BY_DEPTH_PIPE" }
      }
   },
   {
      .name = "pe-pixel-count-drawn-by-color-pipe",
      .type = ETNA_QUERY_PE_PIXEL_COUNT_DRAWN_BY_COLOR_PIPE,
      .group_id = ETNA_QUERY_PE_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "PE", "PIXEL_COUNT_DRAWN_BY_COLOR_PIPE" }
      }
   },
   {
      .name = "pe-pixel-count-drawn-by-depth-pipe",
      .type = ETNA_QUERY_PE_PIXEL_COUNT_DRAWN_BY_DEPTH_PIPE,
      .group_id = ETNA_QUERY_PE_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "PE", "PIXEL_COUNT_DRAWN_BY_DEPTH_PIPE" }
      }
   },
   {
      .name = "sh-shader-cycles",
      .type = ETNA_QUERY_SH_SHADER_CYCLES,
      .group_id = ETNA_QUERY_SH_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SH", "SHADER_CYCLES" }
      }
   },
   {
      .name = "sh-ps-inst-counter",
      .type = ETNA_QUERY_SH_PS_INST_COUNTER,
      .group_id = ETNA_QUERY_SH_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SH", "PS_INST_COUNTER" }
      }
   },
   {
      .name = "sh-rendered-pixel-counter",
      .type = ETNA_QUERY_SH_RENDERED_PIXEL_COUNTER,
      .group_id = ETNA_QUERY_SH_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SH", "RENDERED_PIXEL_COUNTER" }
      }
   },
   {
      .name = "sh-vs-inst-counter",
      .type = ETNA_QUERY_SH_VS_INST_COUNTER,
      .group_id = ETNA_QUERY_SH_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SH", "VS_INST_COUNTER" }
      }
   },
   {
      .name = "sh-rendered-vertice-counter",
      .type = ETNA_QUERY_SH_RENDERED_VERTICE_COUNTER,
      .group_id = ETNA_QUERY_SH_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SH", "RENDERED_VERTICE_COUNTER" }
      }
   },
   {
      .name = "sh-vtx-branch-inst-counter",
      .type = ETNA_QUERY_SH_RENDERED_VERTICE_COUNTER,
      .group_id = ETNA_QUERY_SH_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SH", "VTX_BRANCH_INST_COUNTER" }
      }
   },
   {
      .name = "sh-vtx-texld-inst-counter",
      .type = ETNA_QUERY_SH_RENDERED_VERTICE_COUNTER,
      .group_id = ETNA_QUERY_SH_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SH", "VTX_TEXLD_INST_COUNTER" }
      }
   },
   {
      .name = "sh-plx-branch-inst-counter",
      .type = ETNA_QUERY_SH_RENDERED_VERTICE_COUNTER,
      .group_id = ETNA_QUERY_SH_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SH", "PXL_BRANCH_INST_COUNTER" }
      }
   },
   {
      .name = "sh-plx-texld-inst-counter",
      .type = ETNA_QUERY_SH_RENDERED_VERTICE_COUNTER,
      .group_id = ETNA_QUERY_SH_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SH", "PXL_TEXLD_INST_COUNTER" }
      }
   },
   {
      .name = "pa-input-vtx-counter",
      .type = ETNA_QUERY_PA_INPUT_VTX_COUNTER,
      .group_id = ETNA_QUERY_PA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "PA", "INPUT_VTX_COUNTER" }
      }
   },
   {
      .name = "pa-input-prim-counter",
      .type = ETNA_QUERY_PA_INPUT_PRIM_COUNTER,
      .group_id = ETNA_QUERY_PA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "PA", "INPUT_PRIM_COUNTER" }
      }
   },
   {
      .name = "pa-output-prim-counter",
      .type = ETNA_QUERY_PA_OUTPUT_PRIM_COUNTER,
      .group_id = ETNA_QUERY_PA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "PA", "OUTPUT_PRIM_COUNTER" }
      }
   },
   {
      .name = "pa-depth-clipped-counter",
      .type = ETNA_QUERY_PA_DEPTH_CLIPPED_COUNTER,
      .group_id = ETNA_QUERY_PA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "PA", "DEPTH_CLIPPED_COUNTER" }
      }
   },
   {
      .name = "pa-trivial-rejected-counter",
      .type = ETNA_QUERY_PA_TRIVIAL_REJECTED_COUNTER,
      .group_id = ETNA_QUERY_PA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "PA", "TRIVIAL_REJECTED_COUNTER" }
      }
   },
   {
      .name = "pa-culled-counter",
      .type = ETNA_QUERY_PA_CULLED_COUNTER,
      .group_id = ETNA_QUERY_PA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "PA", "CULLED_COUNTER" }
      }
   },
   {
      .name = "se-culled-triangle-count",
      .type = ETNA_QUERY_SE_CULLED_TRIANGLE_COUNT,
      .group_id = ETNA_QUERY_SE_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SE", "CULLED_TRIANGLE_COUNT" }
      }
   },
   {
      .name = "se-culled-lines-count",
      .type = ETNA_QUERY_SE_CULLED_LINES_COUNT,
      .group_id = ETNA_QUERY_SE_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "SE", "CULLED_LINES_COUNT" }
      }
   },
   {
      .name = "ra-valid-pixel-count",
      .type = ETNA_QUERY_RA_VALID_PIXEL_COUNT,
      .group_id = ETNA_QUERY_RA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "RA", "VALID_PIXEL_COUNT" }
      }
   },
   {
      .name = "ra-total-quad-count",
      .type = ETNA_QUERY_RA_TOTAL_QUAD_COUNT,
      .group_id = ETNA_QUERY_RA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "RA", "TOTAL_QUAD_COUNT" }
      }
   },
   {
      .name = "ra-valid-quad-count-after-early-z",
      .type = ETNA_QUERY_RA_VALID_QUAD_COUNT_AFTER_EARLY_Z,
      .group_id = ETNA_QUERY_RA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "RA", "VALID_QUAD_COUNT_AFTER_EARLY_Z" }
      }
   },
   {
      .name = "ra-total-primitive-count",
      .type = ETNA_QUERY_RA_TOTAL_PRIMITIVE_COUNT,
      .group_id = ETNA_QUERY_RA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "RA", "TOTAL_PRIMITIVE_COUNT" }
      }
   },
   {
      .name = "ra-pipe-cache-miss-counter",
      .type = ETNA_QUERY_RA_PIPE_CACHE_MISS_COUNTER,
      .group_id = ETNA_QUERY_RA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "RA", "PIPE_CACHE_MISS_COUNTER" }
      }
   },
   {
      .name = "ra-prefetch-cache-miss-counter",
      .type = ETNA_QUERY_RA_PREFETCH_CACHE_MISS_COUNTER,
      .group_id = ETNA_QUERY_RA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "RA", "PREFETCH_CACHE_MISS_COUNTER" }
      }
   },
   {
      .name = "ra-pculled-quad-count",
      .type = ETNA_QUERY_RA_CULLED_QUAD_COUNT,
      .group_id = ETNA_QUERY_RA_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "RA", "CULLED_QUAD_COUNT" }
      }
   },
   {
      .name = "tx-total-bilinear-requests",
      .type = ETNA_QUERY_TX_TOTAL_BILINEAR_REQUESTS,
      .group_id = ETNA_QUERY_TX_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "TX", "TOTAL_BILINEAR_REQUESTS" }
      }
   },
   {
      .name = "tx-total-trilinear-requests",
      .type = ETNA_QUERY_TX_TOTAL_TRILINEAR_REQUESTS,
      .group_id = ETNA_QUERY_TX_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "TX", "TOTAL_TRILINEAR_REQUESTS" }
      }
   },
   {
      .name = "tx-total-discarded-texture-requests",
      .type = ETNA_QUERY_TX_TOTAL_DISCARDED_TEXTURE_REQUESTS,
      .group_id = ETNA_QUERY_TX_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "TX", "TOTAL_DISCARDED_TEXTURE_REQUESTS" }
      }
   },
   {
      .name = "tx-total-texture-requests",
      .type = ETNA_QUERY_TX_TOTAL_TEXTURE_REQUESTS,
      .group_id = ETNA_QUERY_TX_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "TX", "TOTAL_TEXTURE_REQUESTS" }
      }
   },
   {
      .name = "tx-mem-read-count",
      .type = ETNA_QUERY_TX_MEM_READ_COUNT,
      .group_id = ETNA_QUERY_TX_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "TX", "MEM_READ_COUNT" }
      }
   },
   {
      .name = "tx-mem-read-in-8b-count",
      .type = ETNA_QUERY_TX_MEM_READ_IN_8B_COUNT,
      .group_id = ETNA_QUERY_TX_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "TX", "MEM_READ_IN_8B_COUNT" }
      }
   },
   {
      .name = "tx-cache-miss-count",
      .type = ETNA_QUERY_TX_CACHE_MISS_COUNT,
      .group_id = ETNA_QUERY_TX_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "TX", "CACHE_MISS_COUNT" }
      }
   },
   {
      .name = "tx-cache-hit-texel-count",
      .type = ETNA_QUERY_TX_CACHE_HIT_TEXEL_COUNT,
      .group_id = ETNA_QUERY_TX_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "TX", "CACHE_HIT_TEXEL_COUNT" }
      }
   },
   {
      .name = "tx-cache-miss-texel-count",
      .type = ETNA_QUERY_TX_CACHE_MISS_TEXEL_COUNT,
      .group_id = ETNA_QUERY_TX_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "TX", "CACHE_MISS_TEXEL_COUNT" }
      }
   },
   {
      .name = "mc-total-read-req-8b-from-pipeline",
      .type = ETNA_QUERY_MC_TOTAL_READ_REQ_8B_FROM_PIPELINE,
      .group_id = ETNA_QUERY_MC_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "MC", "TOTAL_READ_REQ_8B_FROM_PIPELINE" }
      }
   },
   {
      .name = "mc-total-read-req-8b-from-ip",
      .type = ETNA_QUERY_MC_TOTAL_READ_REQ_8B_FROM_IP,
      .group_id = ETNA_QUERY_MC_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "MC", "TOTAL_READ_REQ_8B_FROM_IP" }
      }
   },
   {
      .name = "mc-total-write-req-8b-from-pipeline",
      .type = ETNA_QUERY_MC_TOTAL_WRITE_REQ_8B_FROM_PIPELINE,
      .group_id = ETNA_QUERY_MC_GROUP_ID,
      .source = (const struct etna_perfmon_source[]) {
         { "MC", "TOTAL_WRITE_REQ_8B_FROM_PIPELINE" }
      }
   }
};

static const struct etna_perfmon_config *
etna_pm_query_config(unsigned type)
{
   for (unsigned i = 0; i < ARRAY_SIZE(query_config); i++)
      if (query_config[i].type == type)
         return &query_config[i];

   return NULL;
}

static struct etna_perfmon_signal *
etna_pm_query_signal(struct etna_perfmon *perfmon,
                     const struct etna_perfmon_source *source)
{
   struct etna_perfmon_domain *domain;

   domain = etna_perfmon_get_dom_by_name(perfmon, source->domain);
   if (!domain)
      return NULL;

   return etna_perfmon_get_sig_by_name(domain, source->signal);
}

static inline bool
etna_pm_cfg_supported(struct etna_perfmon *perfmon,
                      const struct etna_perfmon_config *cfg)
{
   struct etna_perfmon_signal *signal = etna_pm_query_signal(perfmon, cfg->source);

   return !!signal;
}

static inline void
etna_pm_add_signal(struct etna_pm_query *pq, struct etna_perfmon *perfmon,
                   const struct etna_perfmon_config *cfg)
{
   struct etna_perfmon_signal *signal = etna_pm_query_signal(perfmon, cfg->source);

   pq->signal = signal;
}

static bool
realloc_query_bo(struct etna_context *ctx, struct etna_pm_query *pq)
{
   if (pq->bo)
      etna_bo_del(pq->bo);

   pq->bo = etna_bo_new(ctx->screen->dev, 64, DRM_ETNA_GEM_CACHE_WC);
   if (unlikely(!pq->bo))
      return false;

   pq->data = etna_bo_map(pq->bo);

   return true;
}

static void
etna_pm_query_get(struct etna_cmd_stream *stream, struct etna_query *q,
                  unsigned flags)
{
   struct etna_pm_query *pq = etna_pm_query(q);
   unsigned offset;
   assert(flags);

   if (flags == ETNA_PM_PROCESS_PRE)
      offset = 2;
   else
      offset = 3;

   struct etna_perf p = {
      .flags = flags,
      .sequence = pq->sequence,
      .bo = pq->bo,
      .signal = pq->signal,
      .offset = offset
   };

   etna_cmd_stream_perf(stream, &p);
}

static inline void
etna_pm_query_update(struct etna_query *q)
{
   struct etna_pm_query *pq = etna_pm_query(q);

   if (pq->data[0] == pq->sequence)
      pq->ready = true;
}

static void
etna_pm_destroy_query(struct etna_context *ctx, struct etna_query *q)
{
   struct etna_pm_query *pq = etna_pm_query(q);

   etna_bo_del(pq->bo);
   FREE(pq);
}

static boolean
etna_pm_begin_query(struct etna_context *ctx, struct etna_query *q)
{
   struct etna_pm_query *pq = etna_pm_query(q);

   pq->ready = false;
   pq->sequence++;

   etna_pm_query_get(ctx->stream, q, ETNA_PM_PROCESS_PRE);

   return true;
}

static void
etna_pm_end_query(struct etna_context *ctx, struct etna_query *q)
{
   etna_pm_query_get(ctx->stream, q, ETNA_PM_PROCESS_POST);
}

static boolean
etna_pm_get_query_result(struct etna_context *ctx, struct etna_query *q,
                         boolean wait, union pipe_query_result *result)
{
   struct etna_pm_query *pq = etna_pm_query(q);

   etna_pm_query_update(q);

   if (!pq->ready) {
      if (!wait)
         return false;

      if (!etna_bo_cpu_prep(pq->bo, DRM_ETNA_PREP_READ))
         return false;

      pq->ready = true;
      etna_bo_cpu_fini(pq->bo);
   }

   result->u32 = pq->data[2] - pq->data[1];

   return true;
}

static const struct etna_query_funcs hw_query_funcs = {
   .destroy_query = etna_pm_destroy_query,
   .begin_query = etna_pm_begin_query,
   .end_query = etna_pm_end_query,
   .get_query_result = etna_pm_get_query_result,
};

struct etna_query *
etna_pm_create_query(struct etna_context *ctx, unsigned query_type)
{
   struct etna_perfmon *perfmon = ctx->screen->perfmon;
   const struct etna_perfmon_config *cfg;
   struct etna_pm_query *pq;
   struct etna_query *q;

   cfg = etna_pm_query_config(query_type);
   if (!cfg)
      return NULL;

   if (!etna_pm_cfg_supported(perfmon, cfg))
      return NULL;

   pq = CALLOC_STRUCT(etna_pm_query);
   if (!pq)
      return NULL;

   if (!realloc_query_bo(ctx, pq)) {
      FREE(pq);
      return NULL;
   }

   q = &pq->base;
   q->funcs = &hw_query_funcs;
   q->type = query_type;

   etna_pm_add_signal(pq, perfmon, cfg);

   return q;
}

void
etna_pm_query_setup(struct etna_screen *screen)
{
   screen->perfmon = etna_perfmon_create(screen->pipe);

   if (!screen->perfmon)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(query_config); i++) {
      const struct etna_perfmon_config *cfg = &query_config[i];

      if (!etna_pm_cfg_supported(screen->perfmon, cfg))
         continue;

      util_dynarray_append(&screen->supported_pm_queries, unsigned, i);
   }
}

int
etna_pm_get_driver_query_info(struct pipe_screen *pscreen, unsigned index,
                              struct pipe_driver_query_info *info)
{
   const struct etna_screen *screen = etna_screen(pscreen);
   const unsigned num = screen->supported_pm_queries.size / sizeof(unsigned);
   unsigned i;

   if (!info)
      return num;

   if (index >= num)
      return 0;

   i = *util_dynarray_element(&screen->supported_pm_queries, unsigned, index);
   assert(i < ARRAY_SIZE(query_config));

   info->name = query_config[i].name;
   info->query_type = query_config[i].type;
   info->group_id = query_config[i].group_id;

   return 1;
}

static
unsigned query_count(unsigned group)
{
   unsigned count = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(query_config); i++)
      if (query_config[i].group_id == group)
         count++;

   assert(count);

   return count;
}

int
etna_pm_get_driver_query_group_info(struct pipe_screen *pscreen,
                                    unsigned index,
                                    struct pipe_driver_query_group_info *info)
{
   if (!info)
      return ARRAY_SIZE(group_names);

   if (index >= ARRAY_SIZE(group_names))
      return 0;

   unsigned count = query_count(index);

   info->name = group_names[index];
   info->max_active_queries = count;
   info->num_queries = count;

   return 1;
}
