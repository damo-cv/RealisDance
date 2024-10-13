/*
 * Copyright © 2015 Intel Corporation
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
#include "brw_fs.h"
#include "brw_cfg.h"
#include "program/program.h"

using namespace brw;

class cmod_propagation_test : public ::testing::Test {
   virtual void SetUp();

public:
   struct brw_compiler *compiler;
   struct gen_device_info *devinfo;
   struct gl_context *ctx;
   struct brw_wm_prog_data *prog_data;
   struct gl_shader_program *shader_prog;
   fs_visitor *v;
};

class cmod_propagation_fs_visitor : public fs_visitor
{
public:
   cmod_propagation_fs_visitor(struct brw_compiler *compiler,
                               struct brw_wm_prog_data *prog_data,
                               nir_shader *shader)
      : fs_visitor(compiler, NULL, NULL, NULL,
                   &prog_data->base, (struct gl_program *) NULL,
                   shader, 8, -1) {}
};


void cmod_propagation_test::SetUp()
{
   ctx = (struct gl_context *)calloc(1, sizeof(*ctx));
   compiler = (struct brw_compiler *)calloc(1, sizeof(*compiler));
   devinfo = (struct gen_device_info *)calloc(1, sizeof(*devinfo));
   compiler->devinfo = devinfo;

   prog_data = ralloc(NULL, struct brw_wm_prog_data);
   nir_shader *shader =
      nir_shader_create(NULL, MESA_SHADER_FRAGMENT, NULL, NULL);

   v = new cmod_propagation_fs_visitor(compiler, prog_data, shader);

   devinfo->gen = 4;
}

static fs_inst *
instruction(bblock_t *block, int num)
{
   fs_inst *inst = (fs_inst *)block->start();
   for (int i = 0; i < num; i++) {
      inst = (fs_inst *)inst->next;
   }
   return inst;
}

static bool
cmod_propagation(fs_visitor *v)
{
   const bool print = getenv("TEST_DEBUG");

   if (print) {
      fprintf(stderr, "= Before =\n");
      v->cfg->dump(v);
   }

   bool ret = v->opt_cmod_propagation();

   if (print) {
      fprintf(stderr, "\n= After =\n");
      v->cfg->dump(v);
   }

   return ret;
}

TEST_F(cmod_propagation_test, basic)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   fs_reg zero(brw_imm_f(0.0f));
   bld.ADD(dest, src0, src1);
   bld.CMP(bld.null_reg_f(), dest, zero, BRW_CONDITIONAL_GE);

   /* = Before =
    *
    * 0: add(8)        dest  src0  src1
    * 1: cmp.ge.f0(8)  null  dest  0.0f
    *
    * = After =
    * 0: add.ge.f0(8)  dest  src0  src1
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(0, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_GE, instruction(block0, 0)->conditional_mod);
}

TEST_F(cmod_propagation_test, cmp_nonzero)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   fs_reg nonzero(brw_imm_f(1.0f));
   bld.ADD(dest, src0, src1);
   bld.CMP(bld.null_reg_f(), dest, nonzero, BRW_CONDITIONAL_GE);

   /* = Before =
    *
    * 0: add(8)        dest  src0  src1
    * 1: cmp.ge.f0(8)  null  dest  1.0f
    *
    * = After =
    * (no changes)
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_GE, instruction(block0, 1)->conditional_mod);
}

TEST_F(cmod_propagation_test, non_cmod_instruction)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::uint_type);
   fs_reg src0 = v->vgrf(glsl_type::uint_type);
   fs_reg zero(brw_imm_ud(0u));
   bld.FBL(dest, src0);
   bld.CMP(bld.null_reg_ud(), dest, zero, BRW_CONDITIONAL_GE);

   /* = Before =
    *
    * 0: fbl(8)        dest  src0
    * 1: cmp.ge.f0(8)  null  dest  0u
    *
    * = After =
    * (no changes)
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_FBL, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_GE, instruction(block0, 1)->conditional_mod);
}

TEST_F(cmod_propagation_test, intervening_flag_write)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   fs_reg src2 = v->vgrf(glsl_type::float_type);
   fs_reg zero(brw_imm_f(0.0f));
   bld.ADD(dest, src0, src1);
   bld.CMP(bld.null_reg_f(), src2, zero, BRW_CONDITIONAL_GE);
   bld.CMP(bld.null_reg_f(), dest, zero, BRW_CONDITIONAL_GE);

   /* = Before =
    *
    * 0: add(8)        dest  src0  src1
    * 1: cmp.ge.f0(8)  null  src2  0.0f
    * 2: cmp.ge.f0(8)  null  dest  0.0f
    *
    * = After =
    * (no changes)
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_GE, instruction(block0, 1)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 2)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_GE, instruction(block0, 2)->conditional_mod);
}

TEST_F(cmod_propagation_test, intervening_flag_read)
{
   const fs_builder &bld = v->bld;
   fs_reg dest0 = v->vgrf(glsl_type::float_type);
   fs_reg dest1 = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   fs_reg src2 = v->vgrf(glsl_type::float_type);
   fs_reg zero(brw_imm_f(0.0f));
   bld.ADD(dest0, src0, src1);
   set_predicate(BRW_PREDICATE_NORMAL, bld.SEL(dest1, src2, zero));
   bld.CMP(bld.null_reg_f(), dest0, zero, BRW_CONDITIONAL_GE);

   /* = Before =
    *
    * 0: add(8)        dest0 src0  src1
    * 1: (+f0) sel(8)  dest1 src2  0.0f
    * 2: cmp.ge.f0(8)  null  dest0 0.0f
    *
    * = After =
    * (no changes)
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_OPCODE_SEL, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_PREDICATE_NORMAL, instruction(block0, 1)->predicate);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 2)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_GE, instruction(block0, 2)->conditional_mod);
}

TEST_F(cmod_propagation_test, intervening_dest_write)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::vec4_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   fs_reg src2 = v->vgrf(glsl_type::vec2_type);
   fs_reg zero(brw_imm_f(0.0f));
   bld.ADD(offset(dest, bld, 2), src0, src1);
   bld.emit(SHADER_OPCODE_TEX, dest, src2)
      ->size_written = 4 * REG_SIZE;
   bld.CMP(bld.null_reg_f(), offset(dest, bld, 2), zero, BRW_CONDITIONAL_GE);

   /* = Before =
    *
    * 0: add(8)        dest+2  src0    src1
    * 1: tex(8) rlen 4 dest+0  src2
    * 2: cmp.ge.f0(8)  null    dest+2  0.0f
    *
    * = After =
    * (no changes)
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_NONE, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(SHADER_OPCODE_TEX, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_NONE, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 2)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_GE, instruction(block0, 2)->conditional_mod);
}

TEST_F(cmod_propagation_test, intervening_flag_read_same_value)
{
   const fs_builder &bld = v->bld;
   fs_reg dest0 = v->vgrf(glsl_type::float_type);
   fs_reg dest1 = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   fs_reg src2 = v->vgrf(glsl_type::float_type);
   fs_reg zero(brw_imm_f(0.0f));
   set_condmod(BRW_CONDITIONAL_GE, bld.ADD(dest0, src0, src1));
   set_predicate(BRW_PREDICATE_NORMAL, bld.SEL(dest1, src2, zero));
   bld.CMP(bld.null_reg_f(), dest0, zero, BRW_CONDITIONAL_GE);

   /* = Before =
    *
    * 0: add.ge.f0(8)  dest0 src0  src1
    * 1: (+f0) sel(8)  dest1 src2  0.0f
    * 2: cmp.ge.f0(8)  null  dest0 0.0f
    *
    * = After =
    * 0: add.ge.f0(8)  dest0 src0  src1
    * 1: (+f0) sel(8)  dest1 src2  0.0f
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_GE, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_SEL, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_PREDICATE_NORMAL, instruction(block0, 1)->predicate);
}

TEST_F(cmod_propagation_test, negate)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   fs_reg zero(brw_imm_f(0.0f));
   bld.ADD(dest, src0, src1);
   dest.negate = true;
   bld.CMP(bld.null_reg_f(), dest, zero, BRW_CONDITIONAL_GE);

   /* = Before =
    *
    * 0: add(8)        dest  src0  src1
    * 1: cmp.ge.f0(8)  null  -dest 0.0f
    *
    * = After =
    * 0: add.le.f0(8)  dest  src0  src1
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(0, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_LE, instruction(block0, 0)->conditional_mod);
}

TEST_F(cmod_propagation_test, movnz)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   bld.CMP(dest, src0, src1, BRW_CONDITIONAL_GE);
   set_condmod(BRW_CONDITIONAL_NZ,
               bld.MOV(bld.null_reg_f(), dest));

   /* = Before =
    *
    * 0: cmp.ge.f0(8)  dest  src0  src1
    * 1: mov.nz.f0(8)  null  dest
    *
    * = After =
    * 0: cmp.ge.f0(8)  dest  src0  src1
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(0, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_GE, instruction(block0, 0)->conditional_mod);
}

TEST_F(cmod_propagation_test, different_types_cmod_with_zero)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::int_type);
   fs_reg src0 = v->vgrf(glsl_type::int_type);
   fs_reg src1 = v->vgrf(glsl_type::int_type);
   fs_reg zero(brw_imm_f(0.0f));
   bld.ADD(dest, src0, src1);
   bld.CMP(bld.null_reg_f(), retype(dest, BRW_REGISTER_TYPE_F), zero,
           BRW_CONDITIONAL_GE);

   /* = Before =
    *
    * 0: add(8)        dest:D  src0:D  src1:D
    * 1: cmp.ge.f0(8)  null:F  dest:F  0.0f
    *
    * = After =
    * (no changes)
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_GE, instruction(block0, 1)->conditional_mod);
}

TEST_F(cmod_propagation_test, andnz_one)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::int_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg zero(brw_imm_f(0.0f));
   fs_reg one(brw_imm_d(1));

   bld.CMP(retype(dest, BRW_REGISTER_TYPE_F), src0, zero, BRW_CONDITIONAL_L);
   set_condmod(BRW_CONDITIONAL_NZ,
               bld.AND(bld.null_reg_d(), dest, one));

   /* = Before =
    * 0: cmp.l.f0(8)     dest:F  src0:F  0F
    * 1: and.nz.f0(8)    null:D  dest:D  1D
    *
    * = After =
    * 0: cmp.l.f0(8)     dest:F  src0:F  0F
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(0, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 0)->conditional_mod);
   EXPECT_TRUE(retype(dest, BRW_REGISTER_TYPE_F)
               .equals(instruction(block0, 0)->dst));
}

TEST_F(cmod_propagation_test, andnz_non_one)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::int_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg zero(brw_imm_f(0.0f));
   fs_reg nonone(brw_imm_d(38));

   bld.CMP(retype(dest, BRW_REGISTER_TYPE_F), src0, zero, BRW_CONDITIONAL_L);
   set_condmod(BRW_CONDITIONAL_NZ,
               bld.AND(bld.null_reg_d(), dest, nonone));

   /* = Before =
    * 0: cmp.l.f0(8)     dest:F  src0:F  0F
    * 1: and.nz.f0(8)    null:D  dest:D  38D
    *
    * = After =
    * (no changes)
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_AND, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_NZ, instruction(block0, 1)->conditional_mod);
}

TEST_F(cmod_propagation_test, andz_one)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::int_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg zero(brw_imm_f(0.0f));
   fs_reg one(brw_imm_d(1));

   bld.CMP(retype(dest, BRW_REGISTER_TYPE_F), src0, zero, BRW_CONDITIONAL_L);
   set_condmod(BRW_CONDITIONAL_Z,
               bld.AND(bld.null_reg_d(), dest, one));

   /* = Before =
    * 0: cmp.l.f0(8)     dest:F  src0:F  0F
    * 1: and.z.f0(8)     null:D  dest:D  1D
    *
    * = After =
    * (no changes)
    */

   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_AND, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_EQ, instruction(block0, 1)->conditional_mod);
}

TEST_F(cmod_propagation_test, add_not_merge_with_compare)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   bld.ADD(dest, src0, src1);
   bld.CMP(bld.null_reg_f(), src0, src1, BRW_CONDITIONAL_L);

   /* The addition and the implicit subtraction in the compare do not compute
    * related values.
    *
    * = Before =
    * 0: add(8)          dest:F  src0:F  src1:F
    * 1: cmp.l.f0(8)     null:F  src0:F  src1:F
    *
    * = After =
    * (no changes)
    */
   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_NONE, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 1)->conditional_mod);
}

TEST_F(cmod_propagation_test, subtract_merge_with_compare)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   bld.ADD(dest, src0, negate(src1));
   bld.CMP(bld.null_reg_f(), src0, src1, BRW_CONDITIONAL_L);

   /* = Before =
    * 0: add(8)          dest:F  src0:F  -src1:F
    * 1: cmp.l.f0(8)     null:F  src0:F  src1:F
    *
    * = After =
    * 0: add.l.f0(8)     dest:F  src0:F  -src1:F
    */
   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(0, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 0)->conditional_mod);
}

TEST_F(cmod_propagation_test, subtract_immediate_merge_with_compare)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg one(brw_imm_f(1.0f));
   fs_reg negative_one(brw_imm_f(-1.0f));

   bld.ADD(dest, src0, negative_one);
   bld.CMP(bld.null_reg_f(), src0, one, BRW_CONDITIONAL_NZ);

   /* = Before =
    * 0: add(8)          dest:F  src0:F  -1.0f
    * 1: cmp.nz.f0(8)    null:F  src0:F  1.0f
    *
    * = After =
    * 0: add.nz.f0(8)    dest:F  src0:F  -1.0f
    */
   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(0, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_NZ, instruction(block0, 0)->conditional_mod);
}

TEST_F(cmod_propagation_test, subtract_merge_with_compare_intervening_add)
{
   const fs_builder &bld = v->bld;
   fs_reg dest0 = v->vgrf(glsl_type::float_type);
   fs_reg dest1 = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   bld.ADD(dest0, src0, negate(src1));
   bld.ADD(dest1, src0, src1);
   bld.CMP(bld.null_reg_f(), src0, src1, BRW_CONDITIONAL_L);

   /* = Before =
    * 0: add(8)          dest0:F src0:F  -src1:F
    * 1: add(8)          dest1:F src0:F  src1:F
    * 2: cmp.l.f0(8)     null:F  src0:F  src1:F
    *
    * = After =
    * 0: add.l.f0(8)     dest0:F src0:F  -src1:F
    * 1: add(8)          dest1:F src0:F  src1:F
    */
   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_NONE, instruction(block0, 1)->conditional_mod);
}

TEST_F(cmod_propagation_test, subtract_not_merge_with_compare_intervening_partial_write)
{
   const fs_builder &bld = v->bld;
   fs_reg dest0 = v->vgrf(glsl_type::float_type);
   fs_reg dest1 = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   bld.ADD(dest0, src0, negate(src1));
   set_predicate(BRW_PREDICATE_NORMAL, bld.ADD(dest1, src0, negate(src1)));
   bld.CMP(bld.null_reg_f(), src0, src1, BRW_CONDITIONAL_L);

   /* = Before =
    * 0: add(8)          dest0:F src0:F  -src1:F
    * 1: (+f0) add(8)    dest1:F src0:F  -src1:F
    * 2: cmp.l.f0(8)     null:F  src0:F  src1:F
    *
    * = After =
    * (no changes)
    */
   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_NONE, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_NONE, instruction(block0, 1)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 2)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 2)->conditional_mod);
}

TEST_F(cmod_propagation_test, subtract_not_merge_with_compare_intervening_add)
{
   const fs_builder &bld = v->bld;
   fs_reg dest0 = v->vgrf(glsl_type::float_type);
   fs_reg dest1 = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   bld.ADD(dest0, src0, negate(src1));
   set_condmod(BRW_CONDITIONAL_EQ, bld.ADD(dest1, src0, src1));
   bld.CMP(bld.null_reg_f(), src0, src1, BRW_CONDITIONAL_L);

   /* = Before =
    * 0: add(8)          dest0:F src0:F  -src1:F
    * 1: add.z.f0(8)     dest1:F src0:F  src1:F
    * 2: cmp.l.f0(8)     null:F  src0:F  src1:F
    *
    * = After =
    * (no changes)
    */
   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);

   EXPECT_FALSE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_NONE, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_EQ, instruction(block0, 1)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_CMP, instruction(block0, 2)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 2)->conditional_mod);
}

TEST_F(cmod_propagation_test, add_merge_with_compare)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   bld.ADD(dest, src0, src1);
   bld.CMP(bld.null_reg_f(), src0, negate(src1), BRW_CONDITIONAL_L);

   /* = Before =
    * 0: add(8)          dest:F  src0:F  src1:F
    * 1: cmp.l.f0(8)     null:F  src0:F  -src1:F
    *
    * = After =
    * 0: add.l.f0(8)     dest:F  src0:F  src1:F
    */
   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(0, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 0)->conditional_mod);
}

TEST_F(cmod_propagation_test, negative_subtract_merge_with_compare)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   bld.ADD(dest, src1, negate(src0));
   bld.CMP(bld.null_reg_f(), src0, src1, BRW_CONDITIONAL_L);

   /* The result of the subtract is the negatiion of the result of the
    * implicit subtract in the compare, so the condition must change.
    *
    * = Before =
    * 0: add(8)          dest:F  src1:F  -src0:F
    * 1: cmp.l.f0(8)     null:F  src0:F  src1:F
    *
    * = After =
    * 0: add.g.f0(8)     dest:F  src0:F  -src1:F
    */
   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(0, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_G, instruction(block0, 0)->conditional_mod);
}

TEST_F(cmod_propagation_test, subtract_delete_compare)
{
   const fs_builder &bld = v->bld;
   fs_reg dest = v->vgrf(glsl_type::float_type);
   fs_reg dest1 = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);
   fs_reg src2 = v->vgrf(glsl_type::float_type);

   set_condmod(BRW_CONDITIONAL_L, bld.ADD(dest, src0, negate(src1)));
   set_predicate(BRW_PREDICATE_NORMAL, bld.MOV(dest1, src2));
   bld.CMP(bld.null_reg_f(), src0, src1, BRW_CONDITIONAL_L);

   /* = Before =
    * 0: add.l.f0(8)     dest0:F src0:F  -src1:F
    * 1: (+f0) mov(0)    dest1:F src2:F
    * 2: cmp.l.f0(8)     null:F  src0:F  src1:F
    *
    * = After =
    * 0: add.l.f0(8)     dest:F  src0:F  -src1:F
    * 1: (+f0) mov(0)    dest1:F src2:F
    */
   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_MOV, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_PREDICATE_NORMAL, instruction(block0, 1)->predicate);
}

TEST_F(cmod_propagation_test, subtract_delete_compare_derp)
{
   const fs_builder &bld = v->bld;
   fs_reg dest0 = v->vgrf(glsl_type::float_type);
   fs_reg dest1 = v->vgrf(glsl_type::float_type);
   fs_reg src0 = v->vgrf(glsl_type::float_type);
   fs_reg src1 = v->vgrf(glsl_type::float_type);

   set_condmod(BRW_CONDITIONAL_L, bld.ADD(dest0, src0, negate(src1)));
   set_predicate(BRW_PREDICATE_NORMAL, bld.ADD(dest1, negate(src0), src1));
   bld.CMP(bld.null_reg_f(), src0, src1, BRW_CONDITIONAL_L);

   /* = Before =
    * 0: add.l.f0(8)     dest0:F src0:F  -src1:F
    * 1: (+f0) add(0)    dest1:F -src0:F src1:F
    * 2: cmp.l.f0(8)     null:F  src0:F  src1:F
    *
    * = After =
    * 0: add.l.f0(8)     dest0:F src0:F  -src1:F
    * 1: (+f0) add(0)    dest1:F -src0:F src1:F
    */
   v->calculate_cfg();
   bblock_t *block0 = v->cfg->blocks[0];

   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(2, block0->end_ip);

   EXPECT_TRUE(cmod_propagation(v));
   EXPECT_EQ(0, block0->start_ip);
   EXPECT_EQ(1, block0->end_ip);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 0)->opcode);
   EXPECT_EQ(BRW_CONDITIONAL_L, instruction(block0, 0)->conditional_mod);
   EXPECT_EQ(BRW_OPCODE_ADD, instruction(block0, 1)->opcode);
   EXPECT_EQ(BRW_PREDICATE_NORMAL, instruction(block0, 1)->predicate);
}
