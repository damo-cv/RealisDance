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

#ifndef GEN_GEM_H
#define GEN_GEM_H

#include <stdint.h>

static inline uint64_t
gen_canonical_address(uint64_t v)
{
   /* From the Broadwell PRM Vol. 2a, MI_LOAD_REGISTER_MEM::MemoryAddress:
    *
    *    "This field specifies the address of the memory location where the
    *    register value specified in the DWord above will read from. The
    *    address specifies the DWord location of the data. Range =
    *    GraphicsVirtualAddress[63:2] for a DWord register GraphicsAddress
    *    [63:48] are ignored by the HW and assumed to be in correct
    *    canonical form [63:48] == [47]."
    */
   const int shift = 63 - 47;
   return (int64_t)(v << shift) >> shift;
}

/**
 * This returns a 48-bit address with the high 16 bits zeroed.
 *
 * It's the opposite of gen_canonicalize_address.
 */
static inline uint64_t
gen_48b_address(uint64_t v)
{
   const int shift = 63 - 47;
   return (uint64_t)(v << shift) >> shift;
}

#endif /* GEN_GEM_H */
