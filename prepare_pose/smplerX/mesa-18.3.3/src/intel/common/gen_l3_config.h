/*
 * Copyright (c) 2015 Intel Corporation
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

#ifndef GEN_L3_CONFIG_H
#define GEN_L3_CONFIG_H

#include <stdio.h>

#include "dev/gen_device_info.h"

/**
 * Chunk of L3 cache reserved for some specific purpose.
 */
enum gen_l3_partition {
   /** Shared local memory. */
   GEN_L3P_SLM = 0,
   /** Unified return buffer. */
   GEN_L3P_URB,
   /** Union of DC and RO. */
   GEN_L3P_ALL,
   /** Data cluster RW partition. */
   GEN_L3P_DC,
   /** Union of IS, C and T. */
   GEN_L3P_RO,
   /** Instruction and state cache. */
   GEN_L3P_IS,
   /** Constant cache. */
   GEN_L3P_C,
   /** Texture cache. */
   GEN_L3P_T,
   /** Number of supported L3 partitions. */
   GEN_NUM_L3P
};

/**
 * L3 configuration represented as the number of ways allocated for each
 * partition.  \sa get_l3_way_size().
 */
struct gen_l3_config {
   unsigned n[GEN_NUM_L3P];
};

/**
 * L3 configuration represented as a vector of weights giving the desired
 * relative size of each partition.  The scale is arbitrary, only the ratios
 * between weights will have an influence on the selection of the closest L3
 * configuration.
 */
struct gen_l3_weights {
   float w[GEN_NUM_L3P];
};

float gen_diff_l3_weights(struct gen_l3_weights w0, struct gen_l3_weights w1);

struct gen_l3_weights
gen_get_default_l3_weights(const struct gen_device_info *devinfo,
                           bool needs_dc, bool needs_slm);

struct gen_l3_weights
gen_get_l3_config_weights(const struct gen_l3_config *cfg);

const struct gen_l3_config *
gen_get_default_l3_config(const struct gen_device_info *devinfo);

const struct gen_l3_config *
gen_get_l3_config(const struct gen_device_info *devinfo,
                  struct gen_l3_weights w0);

unsigned
gen_get_l3_config_urb_size(const struct gen_device_info *devinfo,
                           const struct gen_l3_config *cfg);

void gen_dump_l3_config(const struct gen_l3_config *cfg, FILE *fp);

void gen_get_urb_config(const struct gen_device_info *devinfo,
                        unsigned push_constant_bytes, unsigned urb_size_bytes,
                        bool tess_present, bool gs_present,
                        const unsigned entry_size[4],
                        unsigned entries[4], unsigned start[4]);

#endif /* GEN_L3_CONFIG_H */
