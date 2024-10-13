/*
 * Copyright (C) 2018 Rob Clark <robclark@freedesktop.org>
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
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#ifndef FREEDRENO_PERFCNTR_H_
#define FREEDRENO_PERFCNTR_H_

#include "pipe/p_defines.h"

/*
 * Mapping very closely to the AMD_performance_monitor extension, adreno has
 * groups of performance counters where each group has N counters, which can
 * select from M different countables (things that can be counted), where
 * generally M > N.
 */

/* Describes a single counter: */
struct fd_perfcntr_counter {
	/* offset of the select register to choose what to count: */
	unsigned select_reg;
	/* offset of the lo/hi 32b to read current counter value: */
	unsigned counter_reg_lo;
	unsigned counter_reg_hi;
	/* Optional, most counters don't have enable/clear registers: */
	unsigned enable;
	unsigned clear;
};

/* Describes a single countable: */
struct fd_perfcntr_countable {
	const char *name;
	/* selector register enum value to select this countable: */
	unsigned selector;

	/* description of the countable: */
	enum pipe_driver_query_type query_type;
	enum pipe_driver_query_result_type result_type;
};

/* Describes an entire counter group: */
struct fd_perfcntr_group {
	const char *name;
	unsigned num_counters;
	const struct fd_perfcntr_counter *counters;
	unsigned num_countables;
	const struct fd_perfcntr_countable *countables;
};


#endif /* FREEDRENO_PERFCNTR_H_ */
