/*
 * Copyright © 2018 Intel Corporation
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

#include "brw_defines.h"
#include "brw_performance_query.h"

/**
 * Data format expected by MDAPI.
 */

struct mdapi_gen7_metrics {
   uint64_t TotalTime;

   uint64_t ACounters[45];
   uint64_t NOACounters[16];

   uint64_t PerfCounter1;
   uint64_t PerfCounter2;
   uint32_t SplitOccured;
   uint32_t CoreFrequencyChanged;
   uint64_t CoreFrequency;
   uint32_t ReportId;
   uint32_t ReportsCount;
};

#define GTDI_QUERY_BDW_METRICS_OA_COUNT         36
#define GTDI_QUERY_BDW_METRICS_OA_40b_COUNT     32
#define GTDI_QUERY_BDW_METRICS_NOA_COUNT        16
struct mdapi_gen8_metrics {
   uint64_t TotalTime;
   uint64_t GPUTicks;
   uint64_t OaCntr[GTDI_QUERY_BDW_METRICS_OA_COUNT];
   uint64_t NoaCntr[GTDI_QUERY_BDW_METRICS_NOA_COUNT];
   uint64_t BeginTimestamp;
   uint64_t Reserved1;
   uint64_t Reserved2;
   uint32_t Reserved3;
   uint32_t OverrunOccured;
   uint64_t MarkerUser;
   uint64_t MarkerDriver;

   uint64_t SliceFrequency;
   uint64_t UnsliceFrequency;
   uint64_t PerfCounter1;
   uint64_t PerfCounter2;
   uint32_t SplitOccured;
   uint32_t CoreFrequencyChanged;
   uint64_t CoreFrequency;
   uint32_t ReportId;
   uint32_t ReportsCount;
};

#define GTDI_MAX_READ_REGS 16

struct mdapi_gen9_metrics {
   uint64_t TotalTime;
   uint64_t GPUTicks;
   uint64_t OaCntr[GTDI_QUERY_BDW_METRICS_OA_COUNT];
   uint64_t NoaCntr[GTDI_QUERY_BDW_METRICS_NOA_COUNT];
   uint64_t BeginTimestamp;
   uint64_t Reserved1;
   uint64_t Reserved2;
   uint32_t Reserved3;
   uint32_t OverrunOccured;
   uint64_t MarkerUser;
   uint64_t MarkerDriver;

   uint64_t SliceFrequency;
   uint64_t UnsliceFrequency;
   uint64_t PerfCounter1;
   uint64_t PerfCounter2;
   uint32_t SplitOccured;
   uint32_t CoreFrequencyChanged;
   uint64_t CoreFrequency;
   uint32_t ReportId;
   uint32_t ReportsCount;

   uint64_t UserCntr[GTDI_MAX_READ_REGS];
   uint32_t UserCntrCfgId;
   uint32_t Reserved4;
};

struct mdapi_pipeline_metrics {
   uint64_t IAVertices;
   uint64_t IAPrimitives;
   uint64_t VSInvocations;
   uint64_t GSInvocations;
   uint64_t GSPrimitives;
   uint64_t CInvocations;
   uint64_t CPrimitives;
   uint64_t PSInvocations;
   uint64_t HSInvocations;
   uint64_t DSInvocations;
   uint64_t CSInvocations;
};

int
brw_perf_query_get_mdapi_oa_data(struct brw_context *brw,
                                 struct brw_perf_query_object *obj,
                                 size_t data_size,
                                 uint8_t *data)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   switch (devinfo->gen) {
   case 7: {
      struct mdapi_gen7_metrics *mdapi_data = (struct mdapi_gen7_metrics *) data;

      if (data_size < sizeof(*mdapi_data))
         return 0;

      assert(devinfo->is_haswell);

      for (int i = 0; i < ARRAY_SIZE(mdapi_data->ACounters); i++)
         mdapi_data->ACounters[i] = obj->oa.accumulator[1 + i];

      for (int i = 0; i < ARRAY_SIZE(mdapi_data->NOACounters); i++) {
         mdapi_data->NOACounters[i] =
            obj->oa.accumulator[1 + ARRAY_SIZE(mdapi_data->ACounters) + i];
      }

      mdapi_data->ReportsCount = obj->oa.reports_accumulated;
      mdapi_data->TotalTime = brw_timebase_scale(brw, obj->oa.accumulator[0]);
      mdapi_data->CoreFrequency = obj->oa.gt_frequency[1];
      mdapi_data->CoreFrequencyChanged = obj->oa.gt_frequency[0] != obj->oa.gt_frequency[1];
      return sizeof(*mdapi_data);
   }
   case 8: {
      struct mdapi_gen8_metrics *mdapi_data = (struct mdapi_gen8_metrics *) data;

      if (data_size < sizeof(*mdapi_data))
         return 0;

      for (int i = 0; i < ARRAY_SIZE(mdapi_data->OaCntr); i++)
         mdapi_data->OaCntr[i] = obj->oa.accumulator[2 + i];
      for (int i = 0; i < ARRAY_SIZE(mdapi_data->NoaCntr); i++) {
         mdapi_data->NoaCntr[i] =
            obj->oa.accumulator[2 + ARRAY_SIZE(mdapi_data->OaCntr) + i];
      }

      mdapi_data->ReportId = obj->oa.hw_id;
      mdapi_data->ReportsCount = obj->oa.reports_accumulated;
      mdapi_data->TotalTime = brw_timebase_scale(brw, obj->oa.accumulator[0]);
      mdapi_data->GPUTicks = obj->oa.accumulator[1];
      mdapi_data->CoreFrequency = obj->oa.gt_frequency[1];
      mdapi_data->CoreFrequencyChanged = obj->oa.gt_frequency[0] != obj->oa.gt_frequency[1];
      mdapi_data->SliceFrequency = (obj->oa.slice_frequency[0] + obj->oa.slice_frequency[1]) / 2ULL;
      mdapi_data->UnsliceFrequency = (obj->oa.unslice_frequency[0] + obj->oa.unslice_frequency[1]) / 2ULL;

      return sizeof(*mdapi_data);
   }
   case 9:
   case 10:
   case 11: {
      struct mdapi_gen9_metrics *mdapi_data = (struct mdapi_gen9_metrics *) data;

      if (data_size < sizeof(*mdapi_data))
         return 0;

      for (int i = 0; i < ARRAY_SIZE(mdapi_data->OaCntr); i++)
         mdapi_data->OaCntr[i] = obj->oa.accumulator[2 + i];
      for (int i = 0; i < ARRAY_SIZE(mdapi_data->NoaCntr); i++) {
         mdapi_data->NoaCntr[i] =
            obj->oa.accumulator[2 + ARRAY_SIZE(mdapi_data->OaCntr) + i];
      }

      mdapi_data->ReportId = obj->oa.hw_id;
      mdapi_data->ReportsCount = obj->oa.reports_accumulated;
      mdapi_data->TotalTime = brw_timebase_scale(brw, obj->oa.accumulator[0]);
      mdapi_data->GPUTicks = obj->oa.accumulator[1];
      mdapi_data->CoreFrequency = obj->oa.gt_frequency[1];
      mdapi_data->CoreFrequencyChanged = obj->oa.gt_frequency[0] != obj->oa.gt_frequency[1];
      mdapi_data->SliceFrequency = (obj->oa.slice_frequency[0] + obj->oa.slice_frequency[1]) / 2ULL;
      mdapi_data->UnsliceFrequency = (obj->oa.unslice_frequency[0] + obj->oa.unslice_frequency[1]) / 2ULL;

      return sizeof(*mdapi_data);
   }
   default:
      unreachable("unexpected gen");
   }

   return 0;
}

static void
fill_mdapi_perf_query_counter(struct brw_perf_query_info *query,
                              const char *name,
                              uint32_t data_offset,
                              uint32_t data_size,
                              GLenum data_type)
{
   struct brw_perf_query_counter *counter = &query->counters[query->n_counters];

   counter->name = name;
   counter->desc = "Raw counter value";
   counter->data_type = data_type;
   counter->offset = data_offset;
   counter->size = data_size;
   assert(counter->offset + counter->size <= query->data_size);

   query->n_counters++;
}

#define MDAPI_QUERY_ADD_COUNTER(query, struct_name, field_name, type_name) \
   fill_mdapi_perf_query_counter(query, #field_name,                    \
                                 (uint8_t *) &struct_name.field_name -  \
                                 (uint8_t *) &struct_name,              \
                                 sizeof(struct_name.field_name),        \
                                 GL_PERFQUERY_COUNTER_DATA_##type_name##_INTEL)
#define MDAPI_QUERY_ADD_ARRAY_COUNTER(ctx, query, struct_name, field_name, idx, type_name) \
   fill_mdapi_perf_query_counter(query,                                 \
                                 ralloc_asprintf(ctx, "%s%i", #field_name, idx), \
                                 (uint8_t *) &struct_name.field_name[idx] - \
                                 (uint8_t *) &struct_name,              \
                                 sizeof(struct_name.field_name[0]),     \
                                 GL_PERFQUERY_COUNTER_DATA_##type_name##_INTEL)

void
brw_perf_query_register_mdapi_oa_query(struct brw_context *brw)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   /* MDAPI requires different structures for pretty much every generation
    * (right now we have definitions for gen 7 to 11).
    */
   if (!(devinfo->gen >= 7 && devinfo->gen <= 11))
      return;

   struct brw_perf_query_info *query = brw_perf_query_append_query_info(brw);

   query->kind = OA_COUNTERS_RAW;
   query->name = "Intel_Raw_Hardware_Counters_Set_0_Query";
   /* Guid has to matches with MDAPI's. */
   query->guid = "2f01b241-7014-42a7-9eb6-a925cad3daba";
   query->n_counters = 0;
   query->oa_metrics_set_id = 0; /* Set by MDAPI */

   int n_counters;
   switch (devinfo->gen) {
   case 7: {
      query->oa_format = I915_OA_FORMAT_A45_B8_C8;

      struct mdapi_gen7_metrics metric_data;
      query->data_size = sizeof(metric_data);

      n_counters = 1 + 45 + 16 + 7;
      query->counters =
         rzalloc_array_size(brw->perfquery.queries,
                            sizeof(*query->counters), n_counters);

      MDAPI_QUERY_ADD_COUNTER(query, metric_data, TotalTime, UINT64);
      for (int i = 0; i < ARRAY_SIZE(metric_data.ACounters); i++) {
         MDAPI_QUERY_ADD_ARRAY_COUNTER(brw->perfquery.queries,
                                       query, metric_data, ACounters, i, UINT64);
      }
      for (int i = 0; i < ARRAY_SIZE(metric_data.NOACounters); i++) {
         MDAPI_QUERY_ADD_ARRAY_COUNTER(brw->perfquery.queries,
                                       query, metric_data, NOACounters, i, UINT64);
      }
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, PerfCounter1, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, PerfCounter2, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, SplitOccured, BOOL32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, CoreFrequencyChanged, BOOL32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, CoreFrequency, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, ReportId, UINT32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, ReportsCount, UINT32);
      break;
   }
   case 8: {
      query->oa_format = I915_OA_FORMAT_A32u40_A4u32_B8_C8;

      struct mdapi_gen8_metrics metric_data;
      query->data_size = sizeof(metric_data);

      n_counters = 2 + 36 + 16 + 16;
      query->counters =
         rzalloc_array_size(brw->perfquery.queries,
                            sizeof(*query->counters), n_counters);

      MDAPI_QUERY_ADD_COUNTER(query, metric_data, TotalTime, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, GPUTicks, UINT64);
      for (int i = 0; i < ARRAY_SIZE(metric_data.OaCntr); i++) {
         MDAPI_QUERY_ADD_ARRAY_COUNTER(brw->perfquery.queries,
                                       query, metric_data, OaCntr, i, UINT64);
      }
      for (int i = 0; i < ARRAY_SIZE(metric_data.NoaCntr); i++) {
         MDAPI_QUERY_ADD_ARRAY_COUNTER(brw->perfquery.queries,
                                       query, metric_data, NoaCntr, i, UINT64);
      }
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, BeginTimestamp, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, Reserved1, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, Reserved2, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, Reserved3, UINT32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, OverrunOccured, BOOL32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, MarkerUser, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, MarkerDriver, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, SliceFrequency, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, UnsliceFrequency, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, PerfCounter1, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, PerfCounter2, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, SplitOccured, BOOL32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, CoreFrequencyChanged, BOOL32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, CoreFrequency, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, ReportId, UINT32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, ReportsCount, UINT32);
      break;
   }
   case 9:
   case 10:
   case 11: {
      query->oa_format = I915_OA_FORMAT_A32u40_A4u32_B8_C8;

      struct mdapi_gen9_metrics metric_data;
      query->data_size = sizeof(metric_data);

      n_counters = 2 + 36 + 16 + 16 + 16 + 2;
      query->counters =
         rzalloc_array_size(brw->perfquery.queries,
                            sizeof(*query->counters), n_counters);

      MDAPI_QUERY_ADD_COUNTER(query, metric_data, TotalTime, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, GPUTicks, UINT64);
      for (int i = 0; i < ARRAY_SIZE(metric_data.OaCntr); i++) {
         MDAPI_QUERY_ADD_ARRAY_COUNTER(brw->perfquery.queries,
                                       query, metric_data, OaCntr, i, UINT64);
      }
      for (int i = 0; i < ARRAY_SIZE(metric_data.NoaCntr); i++) {
         MDAPI_QUERY_ADD_ARRAY_COUNTER(brw->perfquery.queries,
                                       query, metric_data, NoaCntr, i, UINT64);
      }
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, BeginTimestamp, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, Reserved1, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, Reserved2, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, Reserved3, UINT32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, OverrunOccured, BOOL32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, MarkerUser, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, MarkerDriver, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, SliceFrequency, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, UnsliceFrequency, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, PerfCounter1, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, PerfCounter2, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, SplitOccured, BOOL32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, CoreFrequencyChanged, BOOL32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, CoreFrequency, UINT64);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, ReportId, UINT32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, ReportsCount, UINT32);
      for (int i = 0; i < ARRAY_SIZE(metric_data.UserCntr); i++) {
         MDAPI_QUERY_ADD_ARRAY_COUNTER(brw->perfquery.queries,
                                       query, metric_data, UserCntr, i, UINT64);
      }
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, UserCntrCfgId, UINT32);
      MDAPI_QUERY_ADD_COUNTER(query, metric_data, Reserved4, UINT32);
      break;
   }
   default:
      unreachable("Unsupported gen");
      break;
   }

   assert(query->n_counters <= n_counters);

   {
      /* Accumulation buffer offsets copied from an actual query... */
      const struct brw_perf_query_info *copy_query =
         &brw->perfquery.queries[0];

      query->gpu_time_offset = copy_query->gpu_time_offset;
      query->gpu_clock_offset = copy_query->gpu_clock_offset;
      query->a_offset = copy_query->a_offset;
      query->b_offset = copy_query->b_offset;
      query->c_offset = copy_query->c_offset;
   }
}

void
brw_perf_query_register_mdapi_statistic_query(struct brw_context *brw)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (!(devinfo->gen >= 7 && devinfo->gen <= 9))
      return;

   struct brw_perf_query_info *query = brw_perf_query_append_query_info(brw);

   query->kind = PIPELINE_STATS;
   query->name = "Intel_Raw_Pipeline_Statistics_Query";
   query->n_counters = 0;
   query->counters =
      rzalloc_array(brw, struct brw_perf_query_counter, MAX_STAT_COUNTERS);

   /* The order has to match mdapi_pipeline_metrics. */
   brw_perf_query_info_add_basic_stat_reg(query, IA_VERTICES_COUNT,
                                          "N vertices submitted");
   brw_perf_query_info_add_basic_stat_reg(query, IA_PRIMITIVES_COUNT,
                                          "N primitives submitted");
   brw_perf_query_info_add_basic_stat_reg(query, VS_INVOCATION_COUNT,
                                          "N vertex shader invocations");
   brw_perf_query_info_add_basic_stat_reg(query, GS_INVOCATION_COUNT,
                                          "N geometry shader invocations");
   brw_perf_query_info_add_basic_stat_reg(query, GS_PRIMITIVES_COUNT,
                                          "N geometry shader primitives emitted");
   brw_perf_query_info_add_basic_stat_reg(query, CL_INVOCATION_COUNT,
                                          "N primitives entering clipping");
   brw_perf_query_info_add_basic_stat_reg(query, CL_PRIMITIVES_COUNT,
                                          "N primitives leaving clipping");
   if (devinfo->is_haswell || devinfo->gen == 8) {
      brw_perf_query_info_add_stat_reg(query, PS_INVOCATION_COUNT, 1, 4,
                                       "N fragment shader invocations",
                                       "N fragment shader invocations");
   } else {
      brw_perf_query_info_add_basic_stat_reg(query, PS_INVOCATION_COUNT,
                                             "N fragment shader invocations");
   }
   brw_perf_query_info_add_basic_stat_reg(query, HS_INVOCATION_COUNT,
                                          "N TCS shader invocations");
   brw_perf_query_info_add_basic_stat_reg(query, DS_INVOCATION_COUNT,
                                          "N TES shader invocations");
   if (devinfo->gen >= 7) {
      brw_perf_query_info_add_basic_stat_reg(query, CS_INVOCATION_COUNT,
                                             "N compute shader invocations");
   }

   query->data_size = sizeof(uint64_t) * query->n_counters;
}
