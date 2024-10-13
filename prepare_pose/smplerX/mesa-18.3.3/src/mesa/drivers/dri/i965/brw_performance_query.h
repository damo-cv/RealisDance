/*
 * Copyright © 2015 Intel Corporation
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef BRW_PERFORMANCE_QUERY_H
#define BRW_PERFORMANCE_QUERY_H

#include <stdint.h>

#include "brw_context.h"
#include "brw_performance_query_metrics.h"

/*
 * When currently allocate only one page for pipeline statistics queries. Here
 * we derived the maximum number of counters for that amount.
 */
#define STATS_BO_SIZE               4096
#define STATS_BO_END_OFFSET_BYTES   (STATS_BO_SIZE / 2)
#define MAX_STAT_COUNTERS           (STATS_BO_END_OFFSET_BYTES / 8)

/*
 * The largest OA formats we can use include:
 * For Haswell:
 *   1 timestamp, 45 A counters, 8 B counters and 8 C counters.
 * For Gen8+
 *   1 timestamp, 1 clock, 36 A counters, 8 B counters and 8 C counters
 */
#define MAX_OA_REPORT_COUNTERS 62

/**
 * i965 representation of a performance query object.
 *
 * NB: We want to keep this structure relatively lean considering that
 * applications may expect to allocate enough objects to be able to
 * query around all draw calls in a frame.
 */
struct brw_perf_query_object
{
   struct gl_perf_query_object base;

   const struct brw_perf_query_info *query;

   /* See query->kind to know which state below is in use... */
   union {
      struct {

         /**
          * BO containing OA counter snapshots at query Begin/End time.
          */
         struct brw_bo *bo;

         /**
          * Address of mapped of @bo
          */
         void *map;

         /**
          * The MI_REPORT_PERF_COUNT command lets us specify a unique
          * ID that will be reflected in the resulting OA report
          * that's written by the GPU. This is the ID we're expecting
          * in the begin report and the the end report should be
          * @begin_report_id + 1.
          */
         int begin_report_id;

         /**
          * Reference the head of the brw->perfquery.sample_buffers
          * list at the time that the query started (so we only need
          * to look at nodes after this point when looking for samples
          * related to this query)
          *
          * (See struct brw_oa_sample_buf description for more details)
          */
         struct exec_node *samples_head;

         /**
          * Storage for the final accumulated OA counters.
          */
         uint64_t accumulator[MAX_OA_REPORT_COUNTERS];

         /**
          * Hw ID used by the context on which the query was running.
          */
         uint32_t hw_id;

         /**
          * false while in the unaccumulated_elements list, and set to
          * true when the final, end MI_RPC snapshot has been
          * accumulated.
          */
         bool results_accumulated;

         /**
          * Number of reports accumulated to produce the results.
          */
         uint32_t reports_accumulated;

         /**
          * Frequency of the GT at begin and end of the query.
          */
         uint64_t gt_frequency[2];

         /**
          * Frequency in the slices of the GT at the begin and end of the
          * query.
          */
         uint64_t slice_frequency[2];

         /**
          * Frequency in the unslice of the GT at the begin and end of the
          * query.
          */
         uint64_t unslice_frequency[2];
      } oa;

      struct {
         /**
          * BO containing starting and ending snapshots for the
          * statistics counters.
          */
         struct brw_bo *bo;
      } pipeline_stats;
   };
};

static inline struct brw_perf_query_info *
brw_perf_query_append_query_info(struct brw_context *brw)
{
   brw->perfquery.queries =
      reralloc(brw, brw->perfquery.queries,
               struct brw_perf_query_info, ++brw->perfquery.n_queries);

   return &brw->perfquery.queries[brw->perfquery.n_queries - 1];
}

static inline void
brw_perf_query_info_add_stat_reg(struct brw_perf_query_info *query,
                                 uint32_t reg,
                                 uint32_t numerator,
                                 uint32_t denominator,
                                 const char *name,
                                 const char *description)
{
   struct brw_perf_query_counter *counter;

   assert(query->n_counters < MAX_STAT_COUNTERS);

   counter = &query->counters[query->n_counters];
   counter->name = name;
   counter->desc = description;
   counter->type = GL_PERFQUERY_COUNTER_RAW_INTEL;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;
   counter->size = sizeof(uint64_t);
   counter->offset = sizeof(uint64_t) * query->n_counters;
   counter->pipeline_stat.reg = reg;
   counter->pipeline_stat.numerator = numerator;
   counter->pipeline_stat.denominator = denominator;

   query->n_counters++;
}

static inline void
brw_perf_query_info_add_basic_stat_reg(struct brw_perf_query_info *query,
                                       uint32_t reg, const char *name)
{
   brw_perf_query_info_add_stat_reg(query, reg, 1, 1, name, name);
}

/* Accumulate 32bits OA counters */
static inline void
brw_perf_query_accumulate_uint32(const uint32_t *report0,
                                 const uint32_t *report1,
                                 uint64_t *accumulator)
{
   *accumulator += (uint32_t)(*report1 - *report0);
}

/* Accumulate 40bits OA counters */
static inline void
brw_perf_query_accumulate_uint40(int a_index,
                                 const uint32_t *report0,
                                 const uint32_t *report1,
                                 uint64_t *accumulator)
{
   const uint8_t *high_bytes0 = (uint8_t *)(report0 + 40);
   const uint8_t *high_bytes1 = (uint8_t *)(report1 + 40);
   uint64_t high0 = (uint64_t)(high_bytes0[a_index]) << 32;
   uint64_t high1 = (uint64_t)(high_bytes1[a_index]) << 32;
   uint64_t value0 = report0[a_index + 4] | high0;
   uint64_t value1 = report1[a_index + 4] | high1;
   uint64_t delta;

   if (value0 > value1)
      delta = (1ULL << 40) + value1 - value0;
   else
      delta = value1 - value0;

   *accumulator += delta;
}

int brw_perf_query_get_mdapi_oa_data(struct brw_context *brw,
                                     struct brw_perf_query_object *obj,
                                     size_t data_size,
                                     uint8_t *data);
void brw_perf_query_register_mdapi_oa_query(struct brw_context *brw);
void brw_perf_query_register_mdapi_statistic_query(struct brw_context *brw);

#endif /* BRW_PERFORMANCE_QUERY_H */
