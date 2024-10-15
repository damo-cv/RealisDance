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

#ifndef GEN_CONTEXT_H
#define GEN_CONTEXT_H

#include <stdint.h>

#define RING_SIZE         (1 * 4096)
#define PPHWSP_SIZE         (1 * 4096)

#define GEN11_LR_CONTEXT_RENDER_SIZE    (14 * 4096)
#define GEN10_LR_CONTEXT_RENDER_SIZE    (19 * 4096)
#define GEN9_LR_CONTEXT_RENDER_SIZE     (22 * 4096)
#define GEN8_LR_CONTEXT_RENDER_SIZE     (20 * 4096)
#define GEN8_LR_CONTEXT_OTHER_SIZE      (2 * 4096)

#define CONTEXT_RENDER_SIZE GEN9_LR_CONTEXT_RENDER_SIZE /* largest size */
#define CONTEXT_OTHER_SIZE GEN8_LR_CONTEXT_OTHER_SIZE

#define MI_LOAD_REGISTER_IMM_n(n) ((0x22 << 23) | (2 * (n) - 1))
#define MI_LRI_FORCE_POSTED       (1<<12)

#define MI_BATCH_BUFFER_END (0xA << 23)

#define HWS_PGA_RCSUNIT      0x02080
#define HWS_PGA_VCSUNIT0   0x12080
#define HWS_PGA_BCSUNIT      0x22080

#define GFX_MODE_RCSUNIT   0x0229c
#define GFX_MODE_VCSUNIT0   0x1229c
#define GFX_MODE_BCSUNIT   0x2229c

#define EXECLIST_SUBMITPORT_RCSUNIT   0x02230
#define EXECLIST_SUBMITPORT_VCSUNIT0   0x12230
#define EXECLIST_SUBMITPORT_BCSUNIT   0x22230

#define EXECLIST_STATUS_RCSUNIT      0x02234
#define EXECLIST_STATUS_VCSUNIT0   0x12234
#define EXECLIST_STATUS_BCSUNIT      0x22234

#define EXECLIST_SQ_CONTENTS0_RCSUNIT   0x02510
#define EXECLIST_SQ_CONTENTS0_VCSUNIT0   0x12510
#define EXECLIST_SQ_CONTENTS0_BCSUNIT   0x22510

#define EXECLIST_CONTROL_RCSUNIT   0x02550
#define EXECLIST_CONTROL_VCSUNIT0   0x12550
#define EXECLIST_CONTROL_BCSUNIT   0x22550

#define MEMORY_MAP_SIZE (64 /* MiB */ * 1024 * 1024)

#define PTE_SIZE 4
#define GEN8_PTE_SIZE 8

#define NUM_PT_ENTRIES (ALIGN(MEMORY_MAP_SIZE, 4096) / 4096)
#define PT_SIZE ALIGN(NUM_PT_ENTRIES * GEN8_PTE_SIZE, 4096)

#define STATIC_GGTT_MAP_START 0

#define RENDER_RING_ADDR STATIC_GGTT_MAP_START
#define RENDER_CONTEXT_ADDR (RENDER_RING_ADDR + RING_SIZE)

#define BLITTER_RING_ADDR (RENDER_CONTEXT_ADDR + PPHWSP_SIZE + GEN9_LR_CONTEXT_RENDER_SIZE)
#define BLITTER_CONTEXT_ADDR (BLITTER_RING_ADDR + RING_SIZE)

#define VIDEO_RING_ADDR (BLITTER_CONTEXT_ADDR + PPHWSP_SIZE + GEN8_LR_CONTEXT_OTHER_SIZE)
#define VIDEO_CONTEXT_ADDR (VIDEO_RING_ADDR + RING_SIZE)

#define STATIC_GGTT_MAP_END (VIDEO_CONTEXT_ADDR + PPHWSP_SIZE + GEN8_LR_CONTEXT_OTHER_SIZE)
#define STATIC_GGTT_MAP_SIZE (STATIC_GGTT_MAP_END - STATIC_GGTT_MAP_START)

#define PML4_PHYS_ADDR ((uint64_t)(STATIC_GGTT_MAP_END))

#define CONTEXT_FLAGS (0x339)   /* Normal Priority | L3-LLC Coherency |
                                 * PPGTT Enabled |
                                 * Legacy Context with 64 bit VA support |
                                 * Valid
                                 */

#define RENDER_CONTEXT_DESCRIPTOR  ((uint64_t)1 << 62 | RENDER_CONTEXT_ADDR  | CONTEXT_FLAGS)
#define BLITTER_CONTEXT_DESCRIPTOR ((uint64_t)2 << 62 | BLITTER_CONTEXT_ADDR | CONTEXT_FLAGS)
#define VIDEO_CONTEXT_DESCRIPTOR   ((uint64_t)3 << 62 | VIDEO_CONTEXT_ADDR   | CONTEXT_FLAGS)

#include "gen8_context.h"
#include "gen10_context.h"

#endif /* GEN_CONTEXT_H */
