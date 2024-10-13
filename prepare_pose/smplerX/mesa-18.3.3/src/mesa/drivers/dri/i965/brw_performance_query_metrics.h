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

#ifndef BRW_PERFORMANCE_QUERY_METRICS_H
#define BRW_PERFORMANCE_QUERY_METRICS_H

#include <stdint.h>

struct brw_pipeline_stat
{
   uint32_t reg;
   uint32_t numerator;
   uint32_t denominator;
};

struct brw_perf_query_counter
{
   const char *name;
   const char *desc;
   GLenum type;
   GLenum data_type;
   uint64_t raw_max;
   size_t offset;
   size_t size;

   union {
      uint64_t (*oa_counter_read_uint64)(struct brw_context *brw,
                                         const struct brw_perf_query_info *query,
                                         uint64_t *accumulator);
      float (*oa_counter_read_float)(struct brw_context *brw,
                                     const struct brw_perf_query_info *query,
                                     uint64_t *accumulator);
      struct brw_pipeline_stat pipeline_stat;
   };
};

#endif /* BRW_PERFORMANCE_QUERY_METRICS_H */
