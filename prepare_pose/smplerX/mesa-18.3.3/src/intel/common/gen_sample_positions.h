/*
 * Copyright © 2016 Intel Corporation
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
#ifndef GEN_SAMPLE_POSITIONS_H
#define GEN_SAMPLE_POSITIONS_H

/*
 * This file defines the standard multisample positions used by both GL and
 * Vulkan.  These correspond to the Vulkan "standard sample locations".
 */

/**
 * 1x MSAA has a single sample at the center: (0.5, 0.5) -> (0x8, 0x8).
 */
#define GEN_SAMPLE_POS_1X(prefix) \
prefix##0XOffset   = 0.5; \
prefix##0YOffset   = 0.5;

/**
 * 2x MSAA sample positions are (0.25, 0.25) and (0.75, 0.75):
 *   4 c
 * 4 0
 * c   1
 */
#define GEN_SAMPLE_POS_2X(prefix) \
prefix##0XOffset   = 0.75; \
prefix##0YOffset   = 0.75; \
prefix##1XOffset   = 0.25; \
prefix##1YOffset   = 0.25;

/**
 * Sample positions:
 *   2 6 a e
 * 2   0
 * 6       1
 * a 2
 * e     3
 */
#define GEN_SAMPLE_POS_4X(prefix) \
prefix##0XOffset   = 0.375; \
prefix##0YOffset   = 0.125; \
prefix##1XOffset   = 0.875; \
prefix##1YOffset   = 0.375; \
prefix##2XOffset   = 0.125; \
prefix##2YOffset   = 0.625; \
prefix##3XOffset   = 0.625; \
prefix##3YOffset   = 0.875;

/**
 * Sample positions:
 *
 * From the Ivy Bridge PRM, Vol2 Part1 p304 (3DSTATE_MULTISAMPLE:
 * Programming Notes):
 *     "When programming the sample offsets (for NUMSAMPLES_4 or _8 and
 *     MSRASTMODE_xxx_PATTERN), the order of the samples 0 to 3 (or 7
 *     for 8X) must have monotonically increasing distance from the
 *     pixel center. This is required to get the correct centroid
 *     computation in the device."
 *
 * Sample positions:
 *   1 3 5 7 9 b d f
 * 1               7
 * 3     3
 * 5         0
 * 7 5
 * 9             2
 * b       1
 * d   4
 * f           6
 */
#define GEN_SAMPLE_POS_8X(prefix) \
prefix##0XOffset   = 0.5625; \
prefix##0YOffset   = 0.3125; \
prefix##1XOffset   = 0.4375; \
prefix##1YOffset   = 0.6875; \
prefix##2XOffset   = 0.8125; \
prefix##2YOffset   = 0.5625; \
prefix##3XOffset   = 0.3125; \
prefix##3YOffset   = 0.1875; \
prefix##4XOffset   = 0.1875; \
prefix##4YOffset   = 0.8125; \
prefix##5XOffset   = 0.0625; \
prefix##5YOffset   = 0.4375; \
prefix##6XOffset   = 0.6875; \
prefix##6YOffset   = 0.9375; \
prefix##7XOffset   = 0.9375; \
prefix##7YOffset   = 0.0625;

/**
 * Sample positions:
 *
 *    0 1 2 3 4 5 6 7 8 9 a b c d e f
 * 0   15
 * 1                  9
 * 2         10
 * 3                        7
 * 4                               13
 * 5                1
 * 6        4
 * 7                          3
 * 8 12
 * 9                    0
 * a            2
 * b                            6
 * c     11
 * d                      5
 * e              8
 * f                             14
 */
#define GEN_SAMPLE_POS_16X(prefix) \
prefix##0XOffset   = 0.5625; \
prefix##0YOffset   = 0.5625; \
prefix##1XOffset   = 0.4375; \
prefix##1YOffset   = 0.3125; \
prefix##2XOffset   = 0.3125; \
prefix##2YOffset   = 0.6250; \
prefix##3XOffset   = 0.7500; \
prefix##3YOffset   = 0.4375; \
prefix##4XOffset   = 0.1875; \
prefix##4YOffset   = 0.3750; \
prefix##5XOffset   = 0.6250; \
prefix##5YOffset   = 0.8125; \
prefix##6XOffset   = 0.8125; \
prefix##6YOffset   = 0.6875; \
prefix##7XOffset   = 0.6875; \
prefix##7YOffset   = 0.1875; \
prefix##8XOffset   = 0.3750; \
prefix##8YOffset   = 0.8750; \
prefix##9XOffset   = 0.5000; \
prefix##9YOffset   = 0.0625; \
prefix##10XOffset  = 0.2500; \
prefix##10YOffset  = 0.1250; \
prefix##11XOffset  = 0.1250; \
prefix##11YOffset  = 0.7500; \
prefix##12XOffset  = 0.0000; \
prefix##12YOffset  = 0.5000; \
prefix##13XOffset  = 0.9375; \
prefix##13YOffset  = 0.2500; \
prefix##14XOffset  = 0.8750; \
prefix##14YOffset  = 0.9375; \
prefix##15XOffset  = 0.0625; \
prefix##15YOffset  = 0.0000;

#endif /* GEN_SAMPLE_POSITIONS_H */
