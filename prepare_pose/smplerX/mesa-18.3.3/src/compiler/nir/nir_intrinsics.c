/* Copyright (C) 2018 Red Hat
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

#include "nir.h"

const nir_intrinsic_info nir_intrinsic_infos[nir_num_intrinsics] = {
{
   .name = "atomic_counter_add",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_add_deref",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "atomic_counter_and",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_and_deref",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "atomic_counter_comp_swap",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_comp_swap_deref",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "atomic_counter_exchange",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_exchange_deref",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "atomic_counter_inc",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_inc_deref",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "atomic_counter_max",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_max_deref",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "atomic_counter_min",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_min_deref",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "atomic_counter_or",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_or_deref",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "atomic_counter_post_dec",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_post_dec_deref",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "atomic_counter_pre_dec",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_pre_dec_deref",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "atomic_counter_read",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "atomic_counter_read_deref",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "atomic_counter_xor",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "atomic_counter_xor_deref",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ballot",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "ballot_bit_count_exclusive",
   .num_srcs = 1,
   .src_components = {
      4
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "ballot_bit_count_inclusive",
   .num_srcs = 1,
   .src_components = {
      4
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "ballot_bit_count_reduce",
   .num_srcs = 1,
   .src_components = {
      4
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "ballot_bitfield_extract",
   .num_srcs = 2,
   .src_components = {
      4, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "ballot_find_lsb",
   .num_srcs = 1,
   .src_components = {
      4
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "ballot_find_msb",
   .num_srcs = 1,
   .src_components = {
      4
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "barrier",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "begin_invocation_interlock",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "copy_deref",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_add",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_and",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_comp_swap",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_exchange",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_fadd",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_fcomp_swap",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_fmax",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_fmin",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_imax",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_imin",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_or",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_umax",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_umin",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "deref_atomic_xor",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "discard",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "discard_if",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "elect",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "emit_vertex",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_STREAM_ID] = 1,
    },
   .flags = 0,
},
{
   .name = "emit_vertex_with_counter",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_STREAM_ID] = 1,
    },
   .flags = 0,
},
{
   .name = "end_invocation_interlock",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "end_primitive",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_STREAM_ID] = 1,
    },
   .flags = 0,
},
{
   .name = "end_primitive_with_counter",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_STREAM_ID] = 1,
    },
   .flags = 0,
},
{
   .name = "exclusive_scan",
   .num_srcs = 1,
   .src_components = {
      0
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_REDUCTION_OP] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "first_invocation",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "get_buffer_size",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "group_memory_barrier",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_atomic_add",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "image_atomic_and",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "image_atomic_comp_swap",
   .num_srcs = 5,
   .src_components = {
      1, 4, 1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "image_atomic_exchange",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "image_atomic_fadd",
   .num_srcs = 5,
   .src_components = {
      1, 1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "image_atomic_max",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "image_atomic_min",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "image_atomic_or",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "image_atomic_xor",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "image_deref_atomic_add",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_deref_atomic_and",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_deref_atomic_comp_swap",
   .num_srcs = 5,
   .src_components = {
      1, 4, 1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_deref_atomic_exchange",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_deref_atomic_fadd",
   .num_srcs = 5,
   .src_components = {
      1, 1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_deref_atomic_max",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_deref_atomic_min",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_deref_atomic_or",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_deref_atomic_xor",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_deref_load",
   .num_srcs = 3,
   .src_components = {
      1, 4, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "image_deref_load_param_intel",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "image_deref_load_raw_intel",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "image_deref_samples",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "image_deref_size",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "image_deref_store",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 0
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_deref_store_raw_intel",
   .num_srcs = 3,
   .src_components = {
      1, 1, 0
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "image_load",
   .num_srcs = 3,
   .src_components = {
      1, 4, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "image_load_raw_intel",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "image_samples",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "image_size",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "image_store",
   .num_srcs = 4,
   .src_components = {
      1, 4, 1, 0
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "image_store_raw_intel",
   .num_srcs = 3,
   .src_components = {
      1, 1, 0
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 4,
   .index_map = {
      [NIR_INTRINSIC_IMAGE_DIM] = 1,
      [NIR_INTRINSIC_IMAGE_ARRAY] = 2,
      [NIR_INTRINSIC_FORMAT] = 3,
      [NIR_INTRINSIC_ACCESS] = 4,
    },
   .flags = 0,
},
{
   .name = "inclusive_scan",
   .num_srcs = 1,
   .src_components = {
      0
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_REDUCTION_OP] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "interp_deref_at_centroid",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "interp_deref_at_offset",
   .num_srcs = 2,
   .src_components = {
      1, 2
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "interp_deref_at_sample",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_alpha_ref_float",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_barycentric_at_offset",
   .num_srcs = 1,
   .src_components = {
      2
   },
   .has_dest = true,
   .dest_components = 2,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_INTERP_MODE] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_barycentric_at_sample",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 2,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_INTERP_MODE] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_barycentric_centroid",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 2,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_INTERP_MODE] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_barycentric_pixel",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 2,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_INTERP_MODE] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_barycentric_sample",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 2,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_INTERP_MODE] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_base_instance",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_base_vertex",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_blend_const_color_a_float",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_blend_const_color_aaaa8888_unorm",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_blend_const_color_b_float",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_blend_const_color_g_float",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_blend_const_color_r_float",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_blend_const_color_rgba8888_unorm",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_constant",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_RANGE] = 2,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_deref",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "load_draw_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_first_vertex",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_frag_coord",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 4,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_front_face",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_global_invocation_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 3,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_helper_invocation",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_input",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_COMPONENT] = 2,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_instance_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_interpolated_input",
   .num_srcs = 2,
   .src_components = {
      2, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_COMPONENT] = 2,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_invocation_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_is_indexed_draw",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_layer_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_local_group_size",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 3,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_local_invocation_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 3,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_local_invocation_index",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_num_subgroups",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_num_work_groups",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 3,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_output",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_COMPONENT] = 2,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "load_param",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_PARAM_IDX] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "load_patch_vertices_in",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_per_vertex_input",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_COMPONENT] = 2,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_per_vertex_output",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_COMPONENT] = 2,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "load_primitive_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_push_constant",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_RANGE] = 2,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_sample_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_sample_id_no_per_sample",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_sample_mask_in",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_sample_pos",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 2,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_shared",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "load_ssbo",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_ACCESS] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "load_subgroup_eq_mask",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_subgroup_ge_mask",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_subgroup_gt_mask",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_subgroup_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_subgroup_invocation",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_subgroup_le_mask",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_subgroup_lt_mask",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_subgroup_size",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_tess_coord",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 3,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_tess_level_inner",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 2,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_tess_level_outer",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 4,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_ubo",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_uniform",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_RANGE] = 2,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_user_clip_plane",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 4,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_UCP_ID] = 1,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_vertex_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_vertex_id_zero_base",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_view_index",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_work_dim",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "load_work_group_id",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 3,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "memory_barrier",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "memory_barrier_atomic_counter",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "memory_barrier_buffer",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "memory_barrier_image",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "memory_barrier_shared",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "nop",
   .num_srcs = 0,
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "quad_broadcast",
   .num_srcs = 2,
   .src_components = {
      0, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "quad_swap_diagonal",
   .num_srcs = 1,
   .src_components = {
      0
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "quad_swap_horizontal",
   .num_srcs = 1,
   .src_components = {
      0
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "quad_swap_vertical",
   .num_srcs = 1,
   .src_components = {
      0
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "read_first_invocation",
   .num_srcs = 1,
   .src_components = {
      0
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "read_invocation",
   .num_srcs = 2,
   .src_components = {
      0, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "reduce",
   .num_srcs = 1,
   .src_components = {
      0
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_REDUCTION_OP] = 1,
      [NIR_INTRINSIC_CLUSTER_SIZE] = 2,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "set_vertex_count",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "shader_clock",
   .num_srcs = 0,
   .has_dest = true,
   .dest_components = 2,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "shared_atomic_add",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_and",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_comp_swap",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_exchange",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_fadd",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_fcomp_swap",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_fmax",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_fmin",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_imax",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_imin",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_or",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_umax",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_umin",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shared_atomic_xor",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
    },
   .flags = 0,
},
{
   .name = "shuffle",
   .num_srcs = 2,
   .src_components = {
      0, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "shuffle_down",
   .num_srcs = 2,
   .src_components = {
      0, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "shuffle_up",
   .num_srcs = 2,
   .src_components = {
      0, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "shuffle_xor",
   .num_srcs = 2,
   .src_components = {
      0, 1
   },
   .has_dest = true,
   .dest_components = 0,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "ssbo_atomic_add",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_and",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_comp_swap",
   .num_srcs = 4,
   .src_components = {
      1, 1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_exchange",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_fadd",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_fcomp_swap",
   .num_srcs = 4,
   .src_components = {
      1, 1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_fmax",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_fmin",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_imax",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_imin",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_or",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_umax",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_umin",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "ssbo_atomic_xor",
   .num_srcs = 3,
   .src_components = {
      1, 1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = 0,
},
{
   .name = "store_deref",
   .num_srcs = 2,
   .src_components = {
      1, 0
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 1,
   .index_map = {
      [NIR_INTRINSIC_WRMASK] = 1,
    },
   .flags = 0,
},
{
   .name = "store_output",
   .num_srcs = 2,
   .src_components = {
      0, 1
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 3,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_WRMASK] = 2,
      [NIR_INTRINSIC_COMPONENT] = 3,
    },
   .flags = 0,
},
{
   .name = "store_per_vertex_output",
   .num_srcs = 3,
   .src_components = {
      0, 1, 1
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 3,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_WRMASK] = 2,
      [NIR_INTRINSIC_COMPONENT] = 3,
    },
   .flags = 0,
},
{
   .name = "store_shared",
   .num_srcs = 2,
   .src_components = {
      0, 1
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_BASE] = 1,
      [NIR_INTRINSIC_WRMASK] = 2,
    },
   .flags = 0,
},
{
   .name = "store_ssbo",
   .num_srcs = 3,
   .src_components = {
      0, 1, 1
   },
   .has_dest = false,
   .dest_components = 0,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_WRMASK] = 1,
      [NIR_INTRINSIC_ACCESS] = 2,
    },
   .flags = 0,
},
{
   .name = "vote_all",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "vote_any",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "vote_feq",
   .num_srcs = 1,
   .src_components = {
      0
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "vote_ieq",
   .num_srcs = 1,
   .src_components = {
      0
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE,
},
{
   .name = "vulkan_resource_index",
   .num_srcs = 1,
   .src_components = {
      1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 2,
   .index_map = {
      [NIR_INTRINSIC_DESC_SET] = 1,
      [NIR_INTRINSIC_BINDING] = 2,
    },
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
{
   .name = "vulkan_resource_reindex",
   .num_srcs = 2,
   .src_components = {
      1, 1
   },
   .has_dest = true,
   .dest_components = 1,
   .num_indices = 0,
   .flags = NIR_INTRINSIC_CAN_ELIMINATE | NIR_INTRINSIC_CAN_REORDER,
},
};
