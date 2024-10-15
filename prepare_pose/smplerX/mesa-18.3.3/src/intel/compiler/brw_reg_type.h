/*
 * Copyright © 2017 Intel Corporation
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

#ifndef BRW_REG_TYPE_H
#define BRW_REG_TYPE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_FUNC_ATTRIBUTE_PURE
#define ATTRIBUTE_PURE __attribute__((__pure__))
#else
#define ATTRIBUTE_PURE
#endif

enum brw_reg_file;
struct gen_device_info;

/*
 * The ordering has been chosen so that no enum value is the same as a
 * compatible hardware encoding.
 */
enum PACKED brw_reg_type {
   /** Floating-point types: @{ */
   BRW_REGISTER_TYPE_NF,
   BRW_REGISTER_TYPE_DF,
   BRW_REGISTER_TYPE_F,
   BRW_REGISTER_TYPE_HF,
   BRW_REGISTER_TYPE_VF,
   /** @} */

   /** Integer types: @{ */
   BRW_REGISTER_TYPE_Q,
   BRW_REGISTER_TYPE_UQ,
   BRW_REGISTER_TYPE_D,
   BRW_REGISTER_TYPE_UD,
   BRW_REGISTER_TYPE_W,
   BRW_REGISTER_TYPE_UW,
   BRW_REGISTER_TYPE_B,
   BRW_REGISTER_TYPE_UB,
   BRW_REGISTER_TYPE_V,
   BRW_REGISTER_TYPE_UV,
   /** @} */

   BRW_REGISTER_TYPE_LAST = BRW_REGISTER_TYPE_UV
};

static inline bool
brw_reg_type_is_floating_point(enum brw_reg_type type)
{
   switch (type) {
   case BRW_REGISTER_TYPE_NF:
   case BRW_REGISTER_TYPE_DF:
   case BRW_REGISTER_TYPE_F:
   case BRW_REGISTER_TYPE_HF:
      return true;
   default:
      return false;
   }
}

unsigned
brw_reg_type_to_hw_type(const struct gen_device_info *devinfo,
                        enum brw_reg_file file, enum brw_reg_type type);

enum brw_reg_type ATTRIBUTE_PURE
brw_hw_type_to_reg_type(const struct gen_device_info *devinfo,
                        enum brw_reg_file file, unsigned hw_type);

unsigned
brw_reg_type_to_a16_hw_3src_type(const struct gen_device_info *devinfo,
                                 enum brw_reg_type type);

unsigned
brw_reg_type_to_a1_hw_3src_type(const struct gen_device_info *devinfo,
                                enum brw_reg_type type);

enum brw_reg_type
brw_a16_hw_3src_type_to_reg_type(const struct gen_device_info *devinfo,
                                 unsigned hw_type);

enum brw_reg_type
brw_a1_hw_3src_type_to_reg_type(const struct gen_device_info *devinfo,
                                unsigned hw_type, unsigned exec_type);

unsigned
brw_reg_type_to_size(enum brw_reg_type type);

const char *
brw_reg_type_to_letters(enum brw_reg_type type);

#ifdef __cplusplus
}
#endif

#endif
