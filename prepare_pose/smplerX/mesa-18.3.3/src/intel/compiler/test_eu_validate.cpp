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

#include <gtest/gtest.h>
#include "brw_eu.h"
#include "util/ralloc.h"

static const struct gen_info {
   const char *name;
} gens[] = {
   { "brw", },
   { "g4x", },
   { "ilk", },
   { "snb", },
   { "ivb", },
   { "byt", },
   { "hsw", },
   { "bdw", },
   { "chv", },
   { "skl", },
   { "bxt", },
   { "kbl", },
   { "aml", },
   { "glk", },
   { "cfl", },
   { "whl", },
   { "cnl", },
   { "icl", },
};

class validation_test: public ::testing::TestWithParam<struct gen_info> {
   virtual void SetUp();

public:
   validation_test();
   virtual ~validation_test();

   struct brw_codegen *p;
   struct gen_device_info devinfo;
};

validation_test::validation_test()
{
   p = rzalloc(NULL, struct brw_codegen);
   memset(&devinfo, 0, sizeof(devinfo));
}

validation_test::~validation_test()
{
   ralloc_free(p);
}

void validation_test::SetUp()
{
   struct gen_info info = GetParam();
   int devid = gen_device_name_to_pci_device_id(info.name);

   gen_get_device_info(devid, &devinfo);

   brw_init_codegen(&devinfo, p, p);
}

struct gen_name {
   template <class ParamType>
   std::string
   operator()(const ::testing::TestParamInfo<ParamType>& info) const {
      return info.param.name;
   }
};

INSTANTIATE_TEST_CASE_P(eu_assembly, validation_test,
                        ::testing::ValuesIn(gens),
                        gen_name());

static bool
validate(struct brw_codegen *p)
{
   const bool print = getenv("TEST_DEBUG");
   struct disasm_info *disasm = disasm_initialize(p->devinfo, NULL);

   if (print) {
      disasm_new_inst_group(disasm, 0);
      disasm_new_inst_group(disasm, p->next_insn_offset);
   }

   bool ret = brw_validate_instructions(p->devinfo, p->store, 0,
                                        p->next_insn_offset, disasm);

   if (print) {
      dump_assembly(p->store, disasm);
   }
   ralloc_free(disasm);

   return ret;
}

#define last_inst    (&p->store[p->nr_insn - 1])
#define g0           brw_vec8_grf(0, 0)
#define acc0         brw_acc_reg(8)
#define null         brw_null_reg()
#define zero         brw_imm_f(0.0f)

static void
clear_instructions(struct brw_codegen *p)
{
   p->next_insn_offset = 0;
   p->nr_insn = 0;
}

TEST_P(validation_test, sanity)
{
   brw_ADD(p, g0, g0, g0);

   EXPECT_TRUE(validate(p));
}

TEST_P(validation_test, src0_null_reg)
{
   brw_MOV(p, g0, null);

   EXPECT_FALSE(validate(p));
}

TEST_P(validation_test, src1_null_reg)
{
   brw_ADD(p, g0, g0, null);

   EXPECT_FALSE(validate(p));
}

TEST_P(validation_test, math_src0_null_reg)
{
   if (devinfo.gen >= 6) {
      gen6_math(p, g0, BRW_MATH_FUNCTION_SIN, null, null);
   } else {
      gen4_math(p, g0, BRW_MATH_FUNCTION_SIN, 0, null, BRW_MATH_PRECISION_FULL);
   }

   EXPECT_FALSE(validate(p));
}

TEST_P(validation_test, math_src1_null_reg)
{
   if (devinfo.gen >= 6) {
      gen6_math(p, g0, BRW_MATH_FUNCTION_POW, g0, null);
      EXPECT_FALSE(validate(p));
   } else {
      /* Math instructions on Gen4/5 are actually SEND messages with payloads.
       * src1 is an immediate message descriptor set by gen4_math.
       */
   }
}

TEST_P(validation_test, opcode46)
{
   /* opcode 46 is "push" on Gen 4 and 5
    *              "fork" on Gen 6
    *              reserved on Gen 7
    *              "goto" on Gen8+
    */
   brw_next_insn(p, 46);

   if (devinfo.gen == 7) {
      EXPECT_FALSE(validate(p));
   } else {
      EXPECT_TRUE(validate(p));
   }
}

/* When the Execution Data Type is wider than the destination data type, the
 * destination must [...] specify a HorzStride equal to the ratio in sizes of
 * the two data types.
 */
TEST_P(validation_test, dest_stride_must_be_equal_to_the_ratio_of_exec_size_to_dest_size)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_D);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_D);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_D);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_D);

   EXPECT_TRUE(validate(p));
}

/* When the Execution Data Type is wider than the destination data type, the
 * destination must be aligned as required by the wider execution data type
 * [...]
 */
TEST_P(validation_test, dst_subreg_must_be_aligned_to_exec_type_size)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 2);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_D);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_D);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_4);
   brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 8);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_D);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_D);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_TRUE(validate(p));
}

/* ExecSize must be greater than or equal to Width. */
TEST_P(validation_test, exec_size_less_than_width)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_16);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_16);

   EXPECT_FALSE(validate(p));
}

/* If ExecSize = Width and HorzStride ≠ 0,
 * VertStride must be set to Width * HorzStride.
 */
TEST_P(validation_test, vertical_stride_is_width_by_horizontal_stride)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);

   EXPECT_FALSE(validate(p));
}

/* If Width = 1, HorzStride must be 0 regardless of the values
 * of ExecSize and VertStride.
 */
TEST_P(validation_test, horizontal_stride_must_be_0_if_width_is_1)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_FALSE(validate(p));
}

/* If ExecSize = Width = 1, both VertStride and HorzStride must be 0. */
TEST_P(validation_test, scalar_region_must_be_0_1_0)
{
   struct brw_reg g0_0 = brw_vec1_grf(0, 0);

   brw_ADD(p, g0, g0, g0_0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_1);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_1);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0_0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_1);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_1);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));
}

/* If VertStride = HorzStride = 0, Width must be 1 regardless of the value
 * of ExecSize.
 */
TEST_P(validation_test, zero_stride_implies_0_1_0)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_2);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_2);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));
}

/* Dst.HorzStride must not be 0. */
TEST_P(validation_test, dst_horizontal_stride_0)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   /* Align16 does not exist on Gen11+ */
   if (devinfo.gen >= 11)
      return;

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   EXPECT_FALSE(validate(p));
}

/* VertStride must be used to cross BRW_GENERAL_REGISTER_FILE register boundaries. This rule implies
 * that elements within a 'Width' cannot cross BRW_GENERAL_REGISTER_FILE boundaries.
 */
TEST_P(validation_test, must_not_cross_grf_boundary_in_a_width)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, 4);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src1_da1_subreg_nr(&devinfo, last_inst, 4);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_FALSE(validate(p));
}

/* Destination Horizontal must be 1 in Align16 */
TEST_P(validation_test, dst_hstride_on_align16_must_be_1)
{
   /* Align16 does not exist on Gen11+ */
   if (devinfo.gen >= 11)
      return;

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_TRUE(validate(p));
}

/* VertStride must be 0 or 4 in Align16 */
TEST_P(validation_test, vstride_on_align16_must_be_0_or_4)
{
   /* Align16 does not exist on Gen11+ */
   if (devinfo.gen >= 11)
      return;

   const struct {
      enum brw_vertical_stride vstride;
      bool expected_result;
   } vstride[] = {
      { BRW_VERTICAL_STRIDE_0, true },
      { BRW_VERTICAL_STRIDE_1, false },
      { BRW_VERTICAL_STRIDE_2, devinfo.is_haswell || devinfo.gen >= 8 },
      { BRW_VERTICAL_STRIDE_4, true },
      { BRW_VERTICAL_STRIDE_8, false },
      { BRW_VERTICAL_STRIDE_16, false },
      { BRW_VERTICAL_STRIDE_32, false },
      { BRW_VERTICAL_STRIDE_ONE_DIMENSIONAL, false },
   };

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   for (unsigned i = 0; i < sizeof(vstride) / sizeof(vstride[0]); i++) {
      brw_ADD(p, g0, g0, g0);
      brw_inst_set_src0_vstride(&devinfo, last_inst, vstride[i].vstride);

      EXPECT_EQ(vstride[i].expected_result, validate(p));

      clear_instructions(p);
   }

   for (unsigned i = 0; i < sizeof(vstride) / sizeof(vstride[0]); i++) {
      brw_ADD(p, g0, g0, g0);
      brw_inst_set_src1_vstride(&devinfo, last_inst, vstride[i].vstride);

      EXPECT_EQ(vstride[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

/* In Direct Addressing mode, a source cannot span more than 2 adjacent BRW_GENERAL_REGISTER_FILE
 * registers.
 */
TEST_P(validation_test, source_cannot_span_more_than_2_registers)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_32);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_8);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_8);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_inst_set_src1_da1_subreg_nr(&devinfo, last_inst, 2);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);

   EXPECT_TRUE(validate(p));
}

/* A destination cannot span more than 2 adjacent BRW_GENERAL_REGISTER_FILE registers. */
TEST_P(validation_test, destination_cannot_span_more_than_2_registers)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_32);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_8);
   brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 6);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_4);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_TRUE(validate(p));
}

TEST_P(validation_test, src_region_spans_two_regs_dst_region_spans_one)
{
   /* Writes to dest are to the lower OWord */
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   /* Writes to dest are to the upper OWord */
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 16);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   /* Writes to dest are evenly split between OWords */
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_8);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   /* Writes to dest are uneven between OWords */
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_4);
   brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 10);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_16);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_2);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   if (devinfo.gen >= 9) {
      EXPECT_TRUE(validate(p));
   } else {
      EXPECT_FALSE(validate(p));
   }
}

TEST_P(validation_test, dst_elements_must_be_evenly_split_between_registers)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 4);

   if (devinfo.gen >= 9) {
      EXPECT_TRUE(validate(p));
   } else {
      EXPECT_FALSE(validate(p));
   }

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   if (devinfo.gen >= 6) {
      gen6_math(p, g0, BRW_MATH_FUNCTION_SIN, g0, null);

      EXPECT_TRUE(validate(p));

      clear_instructions(p);

      gen6_math(p, g0, BRW_MATH_FUNCTION_SIN, g0, null);
      brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 4);

      EXPECT_FALSE(validate(p));
   }
}

TEST_P(validation_test, two_src_two_dst_source_offsets_must_be_same)
{
   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_4);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_4);
   brw_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, 16);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_2);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   if (devinfo.gen <= 7) {
      EXPECT_FALSE(validate(p));
   } else {
      EXPECT_TRUE(validate(p));
   }

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_4);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_4);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_8);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_2);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   EXPECT_TRUE(validate(p));
}

TEST_P(validation_test, two_src_two_dst_each_dst_must_be_derived_from_one_src)
{
   brw_MOV(p, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, 8);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_4);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_4);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   if (devinfo.gen <= 7) {
      EXPECT_FALSE(validate(p));
   } else {
      EXPECT_TRUE(validate(p));
   }

   clear_instructions(p);

   brw_MOV(p, g0, g0);
   brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 16);
   brw_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, 8);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_2);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_2);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_1);

   if (devinfo.gen <= 7) {
      EXPECT_FALSE(validate(p));
   } else {
      EXPECT_TRUE(validate(p));
   }
}

TEST_P(validation_test, one_src_two_dst)
{
   struct brw_reg g0_0 = brw_vec1_grf(0, 0);

   brw_ADD(p, g0, g0_0, g0_0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_D);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src1_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_inst_set_src1_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_inst_set_src1_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);

   if (devinfo.gen >= 8) {
      EXPECT_TRUE(validate(p));
   } else {
      EXPECT_FALSE(validate(p));
   }

   clear_instructions(p);

   brw_ADD(p, g0, g0, g0);
   brw_inst_set_exec_size(&devinfo, last_inst, BRW_EXECUTE_16);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_inst_set_dst_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);
   brw_inst_set_src0_vstride(&devinfo, last_inst, BRW_VERTICAL_STRIDE_0);
   brw_inst_set_src0_width(&devinfo, last_inst, BRW_WIDTH_1);
   brw_inst_set_src0_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_0);
   brw_inst_set_src1_file_type(&devinfo, last_inst, BRW_GENERAL_REGISTER_FILE, BRW_REGISTER_TYPE_W);

   if (devinfo.gen >= 8) {
      EXPECT_TRUE(validate(p));
   } else {
      EXPECT_FALSE(validate(p));
   }
}

TEST_P(validation_test, packed_byte_destination)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src_type;
      bool neg, abs, sat;
      bool expected_result;
   } move[] = {
      { BRW_REGISTER_TYPE_UB, BRW_REGISTER_TYPE_UB, 0, 0, 0, true },
      { BRW_REGISTER_TYPE_B , BRW_REGISTER_TYPE_B , 0, 0, 0, true },
      { BRW_REGISTER_TYPE_UB, BRW_REGISTER_TYPE_B , 0, 0, 0, true },
      { BRW_REGISTER_TYPE_B , BRW_REGISTER_TYPE_UB, 0, 0, 0, true },

      { BRW_REGISTER_TYPE_UB, BRW_REGISTER_TYPE_UB, 1, 0, 0, false },
      { BRW_REGISTER_TYPE_B , BRW_REGISTER_TYPE_B , 1, 0, 0, false },
      { BRW_REGISTER_TYPE_UB, BRW_REGISTER_TYPE_B , 1, 0, 0, false },
      { BRW_REGISTER_TYPE_B , BRW_REGISTER_TYPE_UB, 1, 0, 0, false },

      { BRW_REGISTER_TYPE_UB, BRW_REGISTER_TYPE_UB, 0, 1, 0, false },
      { BRW_REGISTER_TYPE_B , BRW_REGISTER_TYPE_B , 0, 1, 0, false },
      { BRW_REGISTER_TYPE_UB, BRW_REGISTER_TYPE_B , 0, 1, 0, false },
      { BRW_REGISTER_TYPE_B , BRW_REGISTER_TYPE_UB, 0, 1, 0, false },

      { BRW_REGISTER_TYPE_UB, BRW_REGISTER_TYPE_UB, 0, 0, 1, false },
      { BRW_REGISTER_TYPE_B , BRW_REGISTER_TYPE_B , 0, 0, 1, false },
      { BRW_REGISTER_TYPE_UB, BRW_REGISTER_TYPE_B , 0, 0, 1, false },
      { BRW_REGISTER_TYPE_B , BRW_REGISTER_TYPE_UB, 0, 0, 1, false },

      { BRW_REGISTER_TYPE_UB, BRW_REGISTER_TYPE_UW, 0, 0, 0, false },
      { BRW_REGISTER_TYPE_B , BRW_REGISTER_TYPE_W , 0, 0, 0, false },
      { BRW_REGISTER_TYPE_UB, BRW_REGISTER_TYPE_UD, 0, 0, 0, false },
      { BRW_REGISTER_TYPE_B , BRW_REGISTER_TYPE_D , 0, 0, 0, false },
   };

   for (unsigned i = 0; i < sizeof(move) / sizeof(move[0]); i++) {
      brw_MOV(p, retype(g0, move[i].dst_type), retype(g0, move[i].src_type));
      brw_inst_set_src0_negate(&devinfo, last_inst, move[i].neg);
      brw_inst_set_src0_abs(&devinfo, last_inst, move[i].abs);
      brw_inst_set_saturate(&devinfo, last_inst, move[i].sat);

      EXPECT_EQ(move[i].expected_result, validate(p));

      clear_instructions(p);
   }

   brw_SEL(p, retype(g0, BRW_REGISTER_TYPE_UB),
              retype(g0, BRW_REGISTER_TYPE_UB),
              retype(g0, BRW_REGISTER_TYPE_UB));
   brw_inst_set_pred_control(&devinfo, last_inst, BRW_PREDICATE_NORMAL);

   EXPECT_FALSE(validate(p));

   clear_instructions(p);

   brw_SEL(p, retype(g0, BRW_REGISTER_TYPE_B),
              retype(g0, BRW_REGISTER_TYPE_B),
              retype(g0, BRW_REGISTER_TYPE_B));
   brw_inst_set_pred_control(&devinfo, last_inst, BRW_PREDICATE_NORMAL);

   EXPECT_FALSE(validate(p));
}

TEST_P(validation_test, byte_destination_relaxed_alignment)
{
   brw_SEL(p, retype(g0, BRW_REGISTER_TYPE_B),
              retype(g0, BRW_REGISTER_TYPE_W),
              retype(g0, BRW_REGISTER_TYPE_W));
   brw_inst_set_pred_control(&devinfo, last_inst, BRW_PREDICATE_NORMAL);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);

   EXPECT_TRUE(validate(p));

   clear_instructions(p);

   brw_SEL(p, retype(g0, BRW_REGISTER_TYPE_B),
              retype(g0, BRW_REGISTER_TYPE_W),
              retype(g0, BRW_REGISTER_TYPE_W));
   brw_inst_set_pred_control(&devinfo, last_inst, BRW_PREDICATE_NORMAL);
   brw_inst_set_dst_hstride(&devinfo, last_inst, BRW_HORIZONTAL_STRIDE_2);
   brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, 1);

   if (devinfo.gen > 4 || devinfo.is_g4x) {
      EXPECT_TRUE(validate(p));
   } else {
      EXPECT_FALSE(validate(p));
   }
}

TEST_P(validation_test, vector_immediate_destination_alignment)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src_type;
      unsigned subnr;
      unsigned exec_size;
      bool expected_result;
   } move[] = {
      { BRW_REGISTER_TYPE_F, BRW_REGISTER_TYPE_VF,  0, BRW_EXECUTE_4, true  },
      { BRW_REGISTER_TYPE_F, BRW_REGISTER_TYPE_VF, 16, BRW_EXECUTE_4, true  },
      { BRW_REGISTER_TYPE_F, BRW_REGISTER_TYPE_VF,  1, BRW_EXECUTE_4, false },

      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_V,   0, BRW_EXECUTE_8, true  },
      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_V,  16, BRW_EXECUTE_8, true  },
      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_V,   1, BRW_EXECUTE_8, false },

      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_UV,  0, BRW_EXECUTE_8, true  },
      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_UV, 16, BRW_EXECUTE_8, true  },
      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_UV,  1, BRW_EXECUTE_8, false },
   };

   for (unsigned i = 0; i < sizeof(move) / sizeof(move[0]); i++) {
      /* UV type is Gen6+ */
      if (devinfo.gen < 6 &&
          move[i].src_type == BRW_REGISTER_TYPE_UV)
         continue;

      brw_MOV(p, retype(g0, move[i].dst_type), retype(zero, move[i].src_type));
      brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, move[i].subnr);
      brw_inst_set_exec_size(&devinfo, last_inst, move[i].exec_size);

      EXPECT_EQ(move[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, vector_immediate_destination_stride)
{
   static const struct {
      enum brw_reg_type dst_type;
      enum brw_reg_type src_type;
      unsigned stride;
      bool expected_result;
   } move[] = {
      { BRW_REGISTER_TYPE_F, BRW_REGISTER_TYPE_VF, BRW_HORIZONTAL_STRIDE_1, true  },
      { BRW_REGISTER_TYPE_F, BRW_REGISTER_TYPE_VF, BRW_HORIZONTAL_STRIDE_2, false },
      { BRW_REGISTER_TYPE_D, BRW_REGISTER_TYPE_VF, BRW_HORIZONTAL_STRIDE_1, true  },
      { BRW_REGISTER_TYPE_D, BRW_REGISTER_TYPE_VF, BRW_HORIZONTAL_STRIDE_2, false },
      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_VF, BRW_HORIZONTAL_STRIDE_2, true  },
      { BRW_REGISTER_TYPE_B, BRW_REGISTER_TYPE_VF, BRW_HORIZONTAL_STRIDE_4, true  },

      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_V,  BRW_HORIZONTAL_STRIDE_1, true  },
      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_V,  BRW_HORIZONTAL_STRIDE_2, false },
      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_V,  BRW_HORIZONTAL_STRIDE_4, false },
      { BRW_REGISTER_TYPE_B, BRW_REGISTER_TYPE_V,  BRW_HORIZONTAL_STRIDE_2, true  },

      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_UV, BRW_HORIZONTAL_STRIDE_1, true  },
      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_UV, BRW_HORIZONTAL_STRIDE_2, false },
      { BRW_REGISTER_TYPE_W, BRW_REGISTER_TYPE_UV, BRW_HORIZONTAL_STRIDE_4, false },
      { BRW_REGISTER_TYPE_B, BRW_REGISTER_TYPE_UV, BRW_HORIZONTAL_STRIDE_2, true  },
   };

   for (unsigned i = 0; i < sizeof(move) / sizeof(move[0]); i++) {
      /* UV type is Gen6+ */
      if (devinfo.gen < 6 &&
          move[i].src_type == BRW_REGISTER_TYPE_UV)
         continue;

      brw_MOV(p, retype(g0, move[i].dst_type), retype(zero, move[i].src_type));
      brw_inst_set_dst_hstride(&devinfo, last_inst, move[i].stride);

      EXPECT_EQ(move[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, qword_low_power_align1_regioning_restrictions)
{
   static const struct {
      enum opcode opcode;
      unsigned exec_size;

      enum brw_reg_type dst_type;
      unsigned dst_subreg;
      unsigned dst_stride;

      enum brw_reg_type src_type;
      unsigned src_subreg;
      unsigned src_vstride;
      unsigned src_width;
      unsigned src_hstride;

      bool expected_result;
   } inst[] = {
#define INST(opcode, exec_size, dst_type, dst_subreg, dst_stride, src_type,    \
             src_subreg, src_vstride, src_width, src_hstride, expected_result) \
      {                                                                        \
         BRW_OPCODE_##opcode,                                                  \
         BRW_EXECUTE_##exec_size,                                              \
         BRW_REGISTER_TYPE_##dst_type,                                         \
         dst_subreg,                                                           \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                                   \
         BRW_REGISTER_TYPE_##src_type,                                         \
         src_subreg,                                                           \
         BRW_VERTICAL_STRIDE_##src_vstride,                                    \
         BRW_WIDTH_##src_width,                                                \
         BRW_HORIZONTAL_STRIDE_##src_hstride,                                  \
         expected_result,                                                      \
      }

      /* Some instruction that violate no restrictions, as a control */
      INST(MOV, 4, DF, 0, 1, DF, 0, 4, 4, 1, true ),
      INST(MOV, 4, Q,  0, 1, Q,  0, 4, 4, 1, true ),
      INST(MOV, 4, UQ, 0, 1, UQ, 0, 4, 4, 1, true ),

      INST(MOV, 4, DF, 0, 1, F,  0, 8, 4, 2, true ),
      INST(MOV, 4, Q,  0, 1, D,  0, 8, 4, 2, true ),
      INST(MOV, 4, UQ, 0, 1, UD, 0, 8, 4, 2, true ),

      INST(MOV, 4, F,  0, 2, DF, 0, 4, 4, 1, true ),
      INST(MOV, 4, D,  0, 2, Q,  0, 4, 4, 1, true ),
      INST(MOV, 4, UD, 0, 2, UQ, 0, 4, 4, 1, true ),

      INST(MUL, 8, D,  0, 2, D,  0, 8, 4, 2, true ),
      INST(MUL, 8, UD, 0, 2, UD, 0, 8, 4, 2, true ),

      /* Something with subreg nrs */
      INST(MOV, 2, DF, 8, 1, DF, 8, 2, 2, 1, true ),
      INST(MOV, 2, Q,  8, 1, Q,  8, 2, 2, 1, true ),
      INST(MOV, 2, UQ, 8, 1, UQ, 8, 2, 2, 1, true ),

      INST(MUL, 2, D,  4, 2, D,  4, 4, 2, 2, true ),
      INST(MUL, 2, UD, 4, 2, UD, 4, 4, 2, 2, true ),

      /* The PRMs say that for CHV, BXT:
       *
       *    When source or destination datatype is 64b or operation is integer
       *    DWord multiply, regioning in Align1 must follow these rules:
       *
       *    1. Source and Destination horizontal stride must be aligned to the
       *       same qword.
       */
      INST(MOV, 4, DF, 0, 2, DF, 0, 4, 4, 1, false),
      INST(MOV, 4, Q,  0, 2, Q,  0, 4, 4, 1, false),
      INST(MOV, 4, UQ, 0, 2, UQ, 0, 4, 4, 1, false),

      INST(MOV, 4, DF, 0, 2, F,  0, 8, 4, 2, false),
      INST(MOV, 4, Q,  0, 2, D,  0, 8, 4, 2, false),
      INST(MOV, 4, UQ, 0, 2, UD, 0, 8, 4, 2, false),

      INST(MOV, 4, DF, 0, 2, F,  0, 4, 4, 1, false),
      INST(MOV, 4, Q,  0, 2, D,  0, 4, 4, 1, false),
      INST(MOV, 4, UQ, 0, 2, UD, 0, 4, 4, 1, false),

      INST(MUL, 4, D,  0, 2, D,  0, 4, 4, 1, false),
      INST(MUL, 4, UD, 0, 2, UD, 0, 4, 4, 1, false),

      INST(MUL, 4, D,  0, 1, D,  0, 8, 4, 2, false),
      INST(MUL, 4, UD, 0, 1, UD, 0, 8, 4, 2, false),

      /*    2. Regioning must ensure Src.Vstride = Src.Width * Src.Hstride. */
      INST(MOV, 4, DF, 0, 1, DF, 0, 0, 2, 1, false),
      INST(MOV, 4, Q,  0, 1, Q,  0, 0, 2, 1, false),
      INST(MOV, 4, UQ, 0, 1, UQ, 0, 0, 2, 1, false),

      INST(MOV, 4, DF, 0, 1, F,  0, 0, 2, 2, false),
      INST(MOV, 4, Q,  0, 1, D,  0, 0, 2, 2, false),
      INST(MOV, 4, UQ, 0, 1, UD, 0, 0, 2, 2, false),

      INST(MOV, 8, F,  0, 2, DF, 0, 0, 2, 1, false),
      INST(MOV, 8, D,  0, 2, Q,  0, 0, 2, 1, false),
      INST(MOV, 8, UD, 0, 2, UQ, 0, 0, 2, 1, false),

      INST(MUL, 8, D,  0, 2, D,  0, 0, 4, 2, false),
      INST(MUL, 8, UD, 0, 2, UD, 0, 0, 4, 2, false),

      INST(MUL, 8, D,  0, 2, D,  0, 0, 4, 2, false),
      INST(MUL, 8, UD, 0, 2, UD, 0, 0, 4, 2, false),

      /*    3. Source and Destination offset must be the same, except the case
       *       of scalar source.
       */
      INST(MOV, 2, DF, 8, 1, DF, 0, 2, 2, 1, false),
      INST(MOV, 2, Q,  8, 1, Q,  0, 2, 2, 1, false),
      INST(MOV, 2, UQ, 8, 1, UQ, 0, 2, 2, 1, false),

      INST(MOV, 2, DF, 0, 1, DF, 8, 2, 2, 1, false),
      INST(MOV, 2, Q,  0, 1, Q,  8, 2, 2, 1, false),
      INST(MOV, 2, UQ, 0, 1, UQ, 8, 2, 2, 1, false),

      INST(MUL, 4, D,  4, 2, D,  0, 4, 2, 2, false),
      INST(MUL, 4, UD, 4, 2, UD, 0, 4, 2, 2, false),

      INST(MUL, 4, D,  0, 2, D,  4, 4, 2, 2, false),
      INST(MUL, 4, UD, 0, 2, UD, 4, 4, 2, 2, false),

      INST(MOV, 2, DF, 8, 1, DF, 0, 0, 1, 0, true ),
      INST(MOV, 2, Q,  8, 1, Q,  0, 0, 1, 0, true ),
      INST(MOV, 2, UQ, 8, 1, UQ, 0, 0, 1, 0, true ),

      INST(MOV, 2, DF, 8, 1, F,  4, 0, 1, 0, true ),
      INST(MOV, 2, Q,  8, 1, D,  4, 0, 1, 0, true ),
      INST(MOV, 2, UQ, 8, 1, UD, 4, 0, 1, 0, true ),

      INST(MUL, 4, D,  4, 1, D,  0, 0, 1, 0, true ),
      INST(MUL, 4, UD, 4, 1, UD, 0, 0, 1, 0, true ),

      INST(MUL, 4, D,  0, 1, D,  4, 0, 1, 0, true ),
      INST(MUL, 4, UD, 0, 1, UD, 4, 0, 1, 0, true ),

#undef INST
   };

   /* These restrictions only apply to Gen8+ */
   if (devinfo.gen < 8)
      return;

   for (unsigned i = 0; i < sizeof(inst) / sizeof(inst[0]); i++) {
      if (!devinfo.has_64bit_types &&
          (type_sz(inst[i].dst_type) == 8 || type_sz(inst[i].src_type) == 8))
         continue;

      if (inst[i].opcode == BRW_OPCODE_MOV) {
         brw_MOV(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_MUL);
         brw_MUL(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type),
                    retype(zero, inst[i].src_type));
      }
      brw_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      brw_inst_set_dst_da1_subreg_nr(&devinfo, last_inst, inst[i].dst_subreg);
      brw_inst_set_src0_da1_subreg_nr(&devinfo, last_inst, inst[i].src_subreg);

      brw_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      brw_inst_set_src0_vstride(&devinfo, last_inst, inst[i].src_vstride);
      brw_inst_set_src0_width(&devinfo, last_inst, inst[i].src_width);
      brw_inst_set_src0_hstride(&devinfo, last_inst, inst[i].src_hstride);

      if (devinfo.is_cherryview || gen_device_info_is_9lp(&devinfo)) {
         EXPECT_EQ(inst[i].expected_result, validate(p));
      } else {
         EXPECT_TRUE(validate(p));
      }

      clear_instructions(p);
   }
}

TEST_P(validation_test, qword_low_power_no_indirect_addressing)
{
   static const struct {
      enum opcode opcode;
      unsigned exec_size;

      enum brw_reg_type dst_type;
      bool dst_is_indirect;
      unsigned dst_stride;

      enum brw_reg_type src_type;
      bool src_is_indirect;
      unsigned src_vstride;
      unsigned src_width;
      unsigned src_hstride;

      bool expected_result;
   } inst[] = {
#define INST(opcode, exec_size, dst_type, dst_is_indirect, dst_stride,         \
             src_type, src_is_indirect, src_vstride, src_width, src_hstride,   \
             expected_result)                                                  \
      {                                                                        \
         BRW_OPCODE_##opcode,                                                  \
         BRW_EXECUTE_##exec_size,                                              \
         BRW_REGISTER_TYPE_##dst_type,                                         \
         dst_is_indirect,                                                      \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                                   \
         BRW_REGISTER_TYPE_##src_type,                                         \
         src_is_indirect,                                                      \
         BRW_VERTICAL_STRIDE_##src_vstride,                                    \
         BRW_WIDTH_##src_width,                                                \
         BRW_HORIZONTAL_STRIDE_##src_hstride,                                  \
         expected_result,                                                      \
      }

      /* Some instruction that violate no restrictions, as a control */
      INST(MOV, 4, DF, 0, 1, DF, 0, 4, 4, 1, true ),
      INST(MOV, 4, Q,  0, 1, Q,  0, 4, 4, 1, true ),
      INST(MOV, 4, UQ, 0, 1, UQ, 0, 4, 4, 1, true ),

      INST(MUL, 8, D,  0, 2, D,  0, 8, 4, 2, true ),
      INST(MUL, 8, UD, 0, 2, UD, 0, 8, 4, 2, true ),

      INST(MOV, 4, F,  1, 1, F,  0, 4, 4, 1, true ),
      INST(MOV, 4, F,  0, 1, F,  1, 4, 4, 1, true ),
      INST(MOV, 4, F,  1, 1, F,  1, 4, 4, 1, true ),

      /* The PRMs say that for CHV, BXT:
       *
       *    When source or destination datatype is 64b or operation is integer
       *    DWord multiply, indirect addressing must not be used.
       */
      INST(MOV, 4, DF, 1, 1, DF, 0, 4, 4, 1, false),
      INST(MOV, 4, Q,  1, 1, Q,  0, 4, 4, 1, false),
      INST(MOV, 4, UQ, 1, 1, UQ, 0, 4, 4, 1, false),

      INST(MOV, 4, DF, 0, 1, DF, 1, 4, 4, 1, false),
      INST(MOV, 4, Q,  0, 1, Q,  1, 4, 4, 1, false),
      INST(MOV, 4, UQ, 0, 1, UQ, 1, 4, 4, 1, false),

      INST(MOV, 4, DF, 1, 1, F,  0, 8, 4, 2, false),
      INST(MOV, 4, Q,  1, 1, D,  0, 8, 4, 2, false),
      INST(MOV, 4, UQ, 1, 1, UD, 0, 8, 4, 2, false),

      INST(MOV, 4, DF, 0, 1, F,  1, 8, 4, 2, false),
      INST(MOV, 4, Q,  0, 1, D,  1, 8, 4, 2, false),
      INST(MOV, 4, UQ, 0, 1, UD, 1, 8, 4, 2, false),

      INST(MOV, 4, F,  1, 2, DF, 0, 4, 4, 1, false),
      INST(MOV, 4, D,  1, 2, Q,  0, 4, 4, 1, false),
      INST(MOV, 4, UD, 1, 2, UQ, 0, 4, 4, 1, false),

      INST(MOV, 4, F,  0, 2, DF, 1, 4, 4, 1, false),
      INST(MOV, 4, D,  0, 2, Q,  1, 4, 4, 1, false),
      INST(MOV, 4, UD, 0, 2, UQ, 1, 4, 4, 1, false),

      INST(MUL, 8, D,  1, 2, D,  0, 8, 4, 2, false),
      INST(MUL, 8, UD, 1, 2, UD, 0, 8, 4, 2, false),

      INST(MUL, 8, D,  0, 2, D,  1, 8, 4, 2, false),
      INST(MUL, 8, UD, 0, 2, UD, 1, 8, 4, 2, false),

#undef INST
   };

   /* These restrictions only apply to Gen8+ */
   if (devinfo.gen < 8)
      return;

   for (unsigned i = 0; i < sizeof(inst) / sizeof(inst[0]); i++) {
      if (!devinfo.has_64bit_types &&
          (type_sz(inst[i].dst_type) == 8 || type_sz(inst[i].src_type) == 8))
         continue;

      if (inst[i].opcode == BRW_OPCODE_MOV) {
         brw_MOV(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_MUL);
         brw_MUL(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type),
                    retype(zero, inst[i].src_type));
      }
      brw_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      brw_inst_set_dst_address_mode(&devinfo, last_inst, inst[i].dst_is_indirect);
      brw_inst_set_src0_address_mode(&devinfo, last_inst, inst[i].src_is_indirect);

      brw_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      brw_inst_set_src0_vstride(&devinfo, last_inst, inst[i].src_vstride);
      brw_inst_set_src0_width(&devinfo, last_inst, inst[i].src_width);
      brw_inst_set_src0_hstride(&devinfo, last_inst, inst[i].src_hstride);

      if (devinfo.is_cherryview || gen_device_info_is_9lp(&devinfo)) {
         EXPECT_EQ(inst[i].expected_result, validate(p));
      } else {
         EXPECT_TRUE(validate(p));
      }

      clear_instructions(p);
   }
}

TEST_P(validation_test, qword_low_power_no_64bit_arf)
{
   static const struct {
      enum opcode opcode;
      unsigned exec_size;

      struct brw_reg dst;
      enum brw_reg_type dst_type;
      unsigned dst_stride;

      struct brw_reg src;
      enum brw_reg_type src_type;
      unsigned src_vstride;
      unsigned src_width;
      unsigned src_hstride;

      bool acc_wr;
      bool expected_result;
   } inst[] = {
#define INST(opcode, exec_size, dst, dst_type, dst_stride,                     \
             src, src_type, src_vstride, src_width, src_hstride,               \
             acc_wr, expected_result)                                          \
      {                                                                        \
         BRW_OPCODE_##opcode,                                                  \
         BRW_EXECUTE_##exec_size,                                              \
         dst,                                                                  \
         BRW_REGISTER_TYPE_##dst_type,                                         \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                                   \
         src,                                                                  \
         BRW_REGISTER_TYPE_##src_type,                                         \
         BRW_VERTICAL_STRIDE_##src_vstride,                                    \
         BRW_WIDTH_##src_width,                                                \
         BRW_HORIZONTAL_STRIDE_##src_hstride,                                  \
         acc_wr,                                                               \
         expected_result,                                                      \
      }

      /* Some instruction that violate no restrictions, as a control */
      INST(MOV, 4, g0,   DF, 1, g0,   F,  4, 2, 2, 0, true ),
      INST(MOV, 4, g0,   F,  2, g0,   DF, 4, 4, 1, 0, true ),

      INST(MOV, 4, g0,   Q,  1, g0,   D,  4, 2, 2, 0, true ),
      INST(MOV, 4, g0,   D,  2, g0,   Q,  4, 4, 1, 0, true ),

      INST(MOV, 4, g0,   UQ, 1, g0,   UD, 4, 2, 2, 0, true ),
      INST(MOV, 4, g0,   UD, 2, g0,   UQ, 4, 4, 1, 0, true ),

      INST(MOV, 4, null, F,  1, g0,   F,  4, 4, 1, 0, true ),
      INST(MOV, 4, acc0, F,  1, g0,   F,  4, 4, 1, 0, true ),
      INST(MOV, 4, g0,   F,  1, acc0, F,  4, 4, 1, 0, true ),

      INST(MOV, 4, null, D,  1, g0,   D,  4, 4, 1, 0, true ),
      INST(MOV, 4, acc0, D,  1, g0,   D,  4, 4, 1, 0, true ),
      INST(MOV, 4, g0,   D,  1, acc0, D,  4, 4, 1, 0, true ),

      INST(MOV, 4, null, UD, 1, g0,   UD, 4, 4, 1, 0, true ),
      INST(MOV, 4, acc0, UD, 1, g0,   UD, 4, 4, 1, 0, true ),
      INST(MOV, 4, g0,   UD, 1, acc0, UD, 4, 4, 1, 0, true ),

      INST(MUL, 4, g0,   D,  2, g0,   D,  4, 2, 2, 0, true ),
      INST(MUL, 4, g0,   UD, 2, g0,   UD, 4, 2, 2, 0, true ),

      /* The PRMs say that for CHV, BXT:
       *
       *    ARF registers must never be used with 64b datatype or when
       *    operation is integer DWord multiply.
       */
      INST(MOV, 4, acc0, DF, 1, g0,   F,  4, 2, 2, 0, false),
      INST(MOV, 4, g0,   DF, 1, acc0, F,  4, 2, 2, 0, false),

      INST(MOV, 4, acc0, Q,  1, g0,   D,  4, 2, 2, 0, false),
      INST(MOV, 4, g0,   Q,  1, acc0, D,  4, 2, 2, 0, false),

      INST(MOV, 4, acc0, UQ, 1, g0,   UD, 4, 2, 2, 0, false),
      INST(MOV, 4, g0,   UQ, 1, acc0, UD, 4, 2, 2, 0, false),

      INST(MOV, 4, acc0, F,  2, g0,   DF, 4, 4, 1, 0, false),
      INST(MOV, 4, g0,   F,  2, acc0, DF, 4, 4, 1, 0, false),

      INST(MOV, 4, acc0, D,  2, g0,   Q,  4, 4, 1, 0, false),
      INST(MOV, 4, g0,   D,  2, acc0, Q,  4, 4, 1, 0, false),

      INST(MOV, 4, acc0, UD, 2, g0,   UQ, 4, 4, 1, 0, false),
      INST(MOV, 4, g0,   UD, 2, acc0, UQ, 4, 4, 1, 0, false),

      INST(MUL, 4, acc0, D,  2, g0,   D,  4, 2, 2, 0, false),
      INST(MUL, 4, acc0, UD, 2, g0,   UD, 4, 2, 2, 0, false),
      /* MUL cannot have integer accumulator sources, so don't test that */

      /* We assume that the restriction does not apply to the null register */
      INST(MOV, 4, null, DF, 1, g0,   F,  4, 2, 2, 0, true ),
      INST(MOV, 4, null, Q,  1, g0,   D,  4, 2, 2, 0, true ),
      INST(MOV, 4, null, UQ, 1, g0,   UD, 4, 2, 2, 0, true ),

      /* Check implicit accumulator write control */
      INST(MOV, 4, null, DF, 1, g0,   F,  4, 2, 2, 1, false),
      INST(MUL, 4, null, DF, 1, g0,   F,  4, 2, 2, 1, false),

#undef INST
   };

   /* These restrictions only apply to Gen8+ */
   if (devinfo.gen < 8)
      return;

   for (unsigned i = 0; i < sizeof(inst) / sizeof(inst[0]); i++) {
      if (!devinfo.has_64bit_types &&
          (type_sz(inst[i].dst_type) == 8 || type_sz(inst[i].src_type) == 8))
         continue;

      if (inst[i].opcode == BRW_OPCODE_MOV) {
         brw_MOV(p, retype(inst[i].dst, inst[i].dst_type),
                    retype(inst[i].src, inst[i].src_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_MUL);
         brw_MUL(p, retype(inst[i].dst, inst[i].dst_type),
                    retype(inst[i].src, inst[i].src_type),
                    retype(zero, inst[i].src_type));
         brw_inst_set_opcode(&devinfo, last_inst, inst[i].opcode);
      }
      brw_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);
      brw_inst_set_acc_wr_control(&devinfo, last_inst, inst[i].acc_wr);

      brw_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      brw_inst_set_src0_vstride(&devinfo, last_inst, inst[i].src_vstride);
      brw_inst_set_src0_width(&devinfo, last_inst, inst[i].src_width);
      brw_inst_set_src0_hstride(&devinfo, last_inst, inst[i].src_hstride);

      if (devinfo.is_cherryview || gen_device_info_is_9lp(&devinfo)) {
         EXPECT_EQ(inst[i].expected_result, validate(p));
      } else {
         EXPECT_TRUE(validate(p));
      }

      clear_instructions(p);
   }

   if (!devinfo.has_64bit_types)
      return;

   /* MAC implicitly reads the accumulator */
   brw_MAC(p, retype(g0, BRW_REGISTER_TYPE_DF),
              retype(stride(g0, 4, 4, 1), BRW_REGISTER_TYPE_DF),
              retype(stride(g0, 4, 4, 1), BRW_REGISTER_TYPE_DF));
   if (devinfo.is_cherryview || gen_device_info_is_9lp(&devinfo)) {
      EXPECT_FALSE(validate(p));
   } else {
      EXPECT_TRUE(validate(p));
   }
}

TEST_P(validation_test, align16_64_bit_integer)
{
   static const struct {
      enum opcode opcode;
      unsigned exec_size;

      enum brw_reg_type dst_type;
      enum brw_reg_type src_type;

      bool expected_result;
   } inst[] = {
#define INST(opcode, exec_size, dst_type, src_type, expected_result)           \
      {                                                                        \
         BRW_OPCODE_##opcode,                                                  \
         BRW_EXECUTE_##exec_size,                                              \
         BRW_REGISTER_TYPE_##dst_type,                                         \
         BRW_REGISTER_TYPE_##src_type,                                         \
         expected_result,                                                      \
      }

      /* Some instruction that violate no restrictions, as a control */
      INST(MOV, 2, Q,  D,  true ),
      INST(MOV, 2, UQ, UD, true ),
      INST(MOV, 2, DF, F,  true ),

      INST(ADD, 2, Q,  D,  true ),
      INST(ADD, 2, UQ, UD, true ),
      INST(ADD, 2, DF, F,  true ),

      /* The PRMs say that for BDW, SKL:
       *
       *    If Align16 is required for an operation with QW destination and non-QW
       *    source datatypes, the execution size cannot exceed 2.
       */

      INST(MOV, 4, Q,  D,  false),
      INST(MOV, 4, UQ, UD, false),
      INST(MOV, 4, DF, F,  false),

      INST(ADD, 4, Q,  D,  false),
      INST(ADD, 4, UQ, UD, false),
      INST(ADD, 4, DF, F,  false),

#undef INST
   };

   /* 64-bit integer types exist on Gen8+ */
   if (devinfo.gen < 8)
      return;

   /* Align16 does not exist on Gen11+ */
   if (devinfo.gen >= 11)
      return;

   brw_set_default_access_mode(p, BRW_ALIGN_16);

   for (unsigned i = 0; i < sizeof(inst) / sizeof(inst[0]); i++) {
      if (inst[i].opcode == BRW_OPCODE_MOV) {
         brw_MOV(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_ADD);
         brw_ADD(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type),
                    retype(g0, inst[i].src_type));
      }
      brw_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      EXPECT_EQ(inst[i].expected_result, validate(p));

      clear_instructions(p);
   }
}

TEST_P(validation_test, qword_low_power_no_depctrl)
{
   static const struct {
      enum opcode opcode;
      unsigned exec_size;

      enum brw_reg_type dst_type;
      unsigned dst_stride;

      enum brw_reg_type src_type;
      unsigned src_vstride;
      unsigned src_width;
      unsigned src_hstride;

      bool no_dd_check;
      bool no_dd_clear;

      bool expected_result;
   } inst[] = {
#define INST(opcode, exec_size, dst_type, dst_stride,                          \
             src_type, src_vstride, src_width, src_hstride,                    \
             no_dd_check, no_dd_clear, expected_result)                        \
      {                                                                        \
         BRW_OPCODE_##opcode,                                                  \
         BRW_EXECUTE_##exec_size,                                              \
         BRW_REGISTER_TYPE_##dst_type,                                         \
         BRW_HORIZONTAL_STRIDE_##dst_stride,                                   \
         BRW_REGISTER_TYPE_##src_type,                                         \
         BRW_VERTICAL_STRIDE_##src_vstride,                                    \
         BRW_WIDTH_##src_width,                                                \
         BRW_HORIZONTAL_STRIDE_##src_hstride,                                  \
         no_dd_check,                                                          \
         no_dd_clear,                                                          \
         expected_result,                                                      \
      }

      /* Some instruction that violate no restrictions, as a control */
      INST(MOV, 4, DF, 1, F,  8, 4, 2, 0, 0, true ),
      INST(MOV, 4, Q,  1, D,  8, 4, 2, 0, 0, true ),
      INST(MOV, 4, UQ, 1, UD, 8, 4, 2, 0, 0, true ),

      INST(MOV, 4, F,  2, DF, 4, 4, 1, 0, 0, true ),
      INST(MOV, 4, D,  2, Q,  4, 4, 1, 0, 0, true ),
      INST(MOV, 4, UD, 2, UQ, 4, 4, 1, 0, 0, true ),

      INST(MUL, 8, D,  2, D,  8, 4, 2, 0, 0, true ),
      INST(MUL, 8, UD, 2, UD, 8, 4, 2, 0, 0, true ),

      INST(MOV, 4, F,  1, F,  4, 4, 1, 1, 1, true ),

      /* The PRMs say that for CHV, BXT:
       *
       *    When source or destination datatype is 64b or operation is integer
       *    DWord multiply, DepCtrl must not be used.
       */
      INST(MOV, 4, DF, 1, F,  8, 4, 2, 1, 0, false),
      INST(MOV, 4, Q,  1, D,  8, 4, 2, 1, 0, false),
      INST(MOV, 4, UQ, 1, UD, 8, 4, 2, 1, 0, false),

      INST(MOV, 4, F,  2, DF, 4, 4, 1, 1, 0, false),
      INST(MOV, 4, D,  2, Q,  4, 4, 1, 1, 0, false),
      INST(MOV, 4, UD, 2, UQ, 4, 4, 1, 1, 0, false),

      INST(MOV, 4, DF, 1, F,  8, 4, 2, 0, 1, false),
      INST(MOV, 4, Q,  1, D,  8, 4, 2, 0, 1, false),
      INST(MOV, 4, UQ, 1, UD, 8, 4, 2, 0, 1, false),

      INST(MOV, 4, F,  2, DF, 4, 4, 1, 0, 1, false),
      INST(MOV, 4, D,  2, Q,  4, 4, 1, 0, 1, false),
      INST(MOV, 4, UD, 2, UQ, 4, 4, 1, 0, 1, false),

      INST(MUL, 8, D,  2, D,  8, 4, 2, 1, 0, false),
      INST(MUL, 8, UD, 2, UD, 8, 4, 2, 1, 0, false),

      INST(MUL, 8, D,  2, D,  8, 4, 2, 0, 1, false),
      INST(MUL, 8, UD, 2, UD, 8, 4, 2, 0, 1, false),

#undef INST
   };

   /* These restrictions only apply to Gen8+ */
   if (devinfo.gen < 8)
      return;

   for (unsigned i = 0; i < sizeof(inst) / sizeof(inst[0]); i++) {
      if (!devinfo.has_64bit_types &&
          (type_sz(inst[i].dst_type) == 8 || type_sz(inst[i].src_type) == 8))
         continue;

      if (inst[i].opcode == BRW_OPCODE_MOV) {
         brw_MOV(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type));
      } else {
         assert(inst[i].opcode == BRW_OPCODE_MUL);
         brw_MUL(p, retype(g0, inst[i].dst_type),
                    retype(g0, inst[i].src_type),
                    retype(zero, inst[i].src_type));
      }
      brw_inst_set_exec_size(&devinfo, last_inst, inst[i].exec_size);

      brw_inst_set_dst_hstride(&devinfo, last_inst, inst[i].dst_stride);

      brw_inst_set_src0_vstride(&devinfo, last_inst, inst[i].src_vstride);
      brw_inst_set_src0_width(&devinfo, last_inst, inst[i].src_width);
      brw_inst_set_src0_hstride(&devinfo, last_inst, inst[i].src_hstride);

      brw_inst_set_no_dd_check(&devinfo, last_inst, inst[i].no_dd_check);
      brw_inst_set_no_dd_clear(&devinfo, last_inst, inst[i].no_dd_clear);

      if (devinfo.is_cherryview || gen_device_info_is_9lp(&devinfo)) {
         EXPECT_EQ(inst[i].expected_result, validate(p));
      } else {
         EXPECT_TRUE(validate(p));
      }

      clear_instructions(p);
   }
}
