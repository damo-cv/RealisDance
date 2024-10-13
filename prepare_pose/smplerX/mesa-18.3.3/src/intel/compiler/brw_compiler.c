/*
 * Copyright © 2015-2016 Intel Corporation
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

#include "brw_compiler.h"
#include "brw_shader.h"
#include "brw_eu.h"
#include "common/gen_debug.h"
#include "compiler/nir/nir.h"
#include "main/errors.h"
#include "util/debug.h"

#define COMMON_OPTIONS                                                        \
   .lower_sub = true,                                                         \
   .lower_fdiv = true,                                                        \
   .lower_scmp = true,                                                        \
   .lower_fmod32 = true,                                                      \
   .lower_fmod64 = false,                                                     \
   .lower_bitfield_extract = true,                                            \
   .lower_bitfield_insert = true,                                             \
   .lower_uadd_carry = true,                                                  \
   .lower_usub_borrow = true,                                                 \
   .lower_fdiv = true,                                                        \
   .lower_flrp64 = true,                                                      \
   .lower_ldexp = true,                                                       \
   .lower_device_index_to_zero = true,                                        \
   .native_integers = true,                                                   \
   .use_interpolated_input_intrinsics = true,                                 \
   .vertex_id_zero_based = true,                                              \
   .lower_base_vertex = true

#define COMMON_SCALAR_OPTIONS                                                 \
   .lower_pack_half_2x16 = true,                                              \
   .lower_pack_snorm_2x16 = true,                                             \
   .lower_pack_snorm_4x8 = true,                                              \
   .lower_pack_unorm_2x16 = true,                                             \
   .lower_pack_unorm_4x8 = true,                                              \
   .lower_unpack_half_2x16 = true,                                            \
   .lower_unpack_snorm_2x16 = true,                                           \
   .lower_unpack_snorm_4x8 = true,                                            \
   .lower_unpack_unorm_2x16 = true,                                           \
   .lower_unpack_unorm_4x8 = true,                                            \
   .max_unroll_iterations = 32

static const struct nir_shader_compiler_options scalar_nir_options = {
   COMMON_OPTIONS,
   COMMON_SCALAR_OPTIONS,
};

static const struct nir_shader_compiler_options scalar_nir_options_gen11 = {
   COMMON_OPTIONS,
   COMMON_SCALAR_OPTIONS,
   .lower_flrp32 = true,
};

static const struct nir_shader_compiler_options vector_nir_options = {
   COMMON_OPTIONS,

   /* In the vec4 backend, our dpN instruction replicates its result to all the
    * components of a vec4.  We would like NIR to give us replicated fdot
    * instructions because it can optimize better for us.
    */
   .fdot_replicates = true,

   /* Prior to Gen6, there are no three source operations for SIMD4x2. */
   .lower_flrp32 = true,

   .lower_pack_snorm_2x16 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .max_unroll_iterations = 32,
};

static const struct nir_shader_compiler_options vector_nir_options_gen6 = {
   COMMON_OPTIONS,

   /* In the vec4 backend, our dpN instruction replicates its result to all the
    * components of a vec4.  We would like NIR to give us replicated fdot
    * instructions because it can optimize better for us.
    */
   .fdot_replicates = true,

   .lower_pack_snorm_2x16 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .max_unroll_iterations = 32,
};

struct brw_compiler *
brw_compiler_create(void *mem_ctx, const struct gen_device_info *devinfo)
{
   struct brw_compiler *compiler = rzalloc(mem_ctx, struct brw_compiler);

   compiler->devinfo = devinfo;

   brw_fs_alloc_reg_sets(compiler);
   brw_vec4_alloc_reg_set(compiler);
   brw_init_compaction_tables(devinfo);

   compiler->precise_trig = env_var_as_boolean("INTEL_PRECISE_TRIG", false);

   if (devinfo->gen >= 10) {
      /* We don't support vec4 mode on Cannonlake. */
      for (int i = MESA_SHADER_VERTEX; i < MESA_SHADER_STAGES; i++)
         compiler->scalar_stage[i] = true;
   } else {
      compiler->scalar_stage[MESA_SHADER_VERTEX] =
         devinfo->gen >= 8 && env_var_as_boolean("INTEL_SCALAR_VS", true);
      compiler->scalar_stage[MESA_SHADER_TESS_CTRL] =
         devinfo->gen >= 8 && env_var_as_boolean("INTEL_SCALAR_TCS", true);
      compiler->scalar_stage[MESA_SHADER_TESS_EVAL] =
         devinfo->gen >= 8 && env_var_as_boolean("INTEL_SCALAR_TES", true);
      compiler->scalar_stage[MESA_SHADER_GEOMETRY] =
         devinfo->gen >= 8 && env_var_as_boolean("INTEL_SCALAR_GS", true);
      compiler->scalar_stage[MESA_SHADER_FRAGMENT] = true;
      compiler->scalar_stage[MESA_SHADER_COMPUTE] = true;
   }

   /* We want the GLSL compiler to emit code that uses condition codes */
   for (int i = 0; i < MESA_SHADER_STAGES; i++) {
      compiler->glsl_compiler_options[i].MaxUnrollIterations = 0;
      compiler->glsl_compiler_options[i].MaxIfDepth =
         devinfo->gen < 6 ? 16 : UINT_MAX;

      compiler->glsl_compiler_options[i].EmitNoIndirectInput = true;
      compiler->glsl_compiler_options[i].EmitNoIndirectUniform = false;

      bool is_scalar = compiler->scalar_stage[i];

      compiler->glsl_compiler_options[i].EmitNoIndirectOutput = is_scalar;
      compiler->glsl_compiler_options[i].EmitNoIndirectTemp = is_scalar;
      compiler->glsl_compiler_options[i].OptimizeForAOS = !is_scalar;

      if (is_scalar) {
         compiler->glsl_compiler_options[i].NirOptions =
            devinfo->gen < 11 ? &scalar_nir_options : &scalar_nir_options_gen11;
      } else {
         compiler->glsl_compiler_options[i].NirOptions =
            devinfo->gen < 6 ? &vector_nir_options : &vector_nir_options_gen6;
      }

      compiler->glsl_compiler_options[i].LowerBufferInterfaceBlocks = true;
      compiler->glsl_compiler_options[i].ClampBlockIndicesToArrayBounds = true;
   }

   compiler->glsl_compiler_options[MESA_SHADER_TESS_CTRL].EmitNoIndirectInput = false;
   compiler->glsl_compiler_options[MESA_SHADER_TESS_EVAL].EmitNoIndirectInput = false;
   compiler->glsl_compiler_options[MESA_SHADER_TESS_CTRL].EmitNoIndirectOutput = false;

   if (compiler->scalar_stage[MESA_SHADER_GEOMETRY])
      compiler->glsl_compiler_options[MESA_SHADER_GEOMETRY].EmitNoIndirectInput = false;

   return compiler;
}

static void
insert_u64_bit(uint64_t *val, bool add)
{
   *val = (*val << 1) | !!add;
}

uint64_t
brw_get_compiler_config_value(const struct brw_compiler *compiler)
{
   uint64_t config = 0;
   insert_u64_bit(&config, compiler->precise_trig);
   if (compiler->devinfo->gen >= 8 && compiler->devinfo->gen < 10) {
      insert_u64_bit(&config, compiler->scalar_stage[MESA_SHADER_VERTEX]);
      insert_u64_bit(&config, compiler->scalar_stage[MESA_SHADER_TESS_CTRL]);
      insert_u64_bit(&config, compiler->scalar_stage[MESA_SHADER_TESS_EVAL]);
      insert_u64_bit(&config, compiler->scalar_stage[MESA_SHADER_GEOMETRY]);
   }
   uint64_t debug_bits = INTEL_DEBUG;
   uint64_t mask = DEBUG_DISK_CACHE_MASK;
   while (mask != 0) {
      const uint64_t bit = 1ULL << (ffsll(mask) - 1);
      insert_u64_bit(&config, (debug_bits & bit) != 0);
      mask &= ~bit;
   }
   return config;
}

unsigned
brw_prog_data_size(gl_shader_stage stage)
{
   STATIC_ASSERT(MESA_SHADER_VERTEX == 0);
   STATIC_ASSERT(MESA_SHADER_TESS_CTRL == 1);
   STATIC_ASSERT(MESA_SHADER_TESS_EVAL == 2);
   STATIC_ASSERT(MESA_SHADER_GEOMETRY == 3);
   STATIC_ASSERT(MESA_SHADER_FRAGMENT == 4);
   STATIC_ASSERT(MESA_SHADER_COMPUTE == 5);
   static const size_t stage_sizes[] = {
      sizeof(struct brw_vs_prog_data),
      sizeof(struct brw_tcs_prog_data),
      sizeof(struct brw_tes_prog_data),
      sizeof(struct brw_gs_prog_data),
      sizeof(struct brw_wm_prog_data),
      sizeof(struct brw_cs_prog_data),
   };
   assert((int)stage >= 0 && stage < ARRAY_SIZE(stage_sizes));
   return stage_sizes[stage];
}

unsigned
brw_prog_key_size(gl_shader_stage stage)
{
   static const size_t stage_sizes[] = {
      sizeof(struct brw_vs_prog_key),
      sizeof(struct brw_tcs_prog_key),
      sizeof(struct brw_tes_prog_key),
      sizeof(struct brw_gs_prog_key),
      sizeof(struct brw_wm_prog_key),
      sizeof(struct brw_cs_prog_key),
   };
   assert((int)stage >= 0 && stage < ARRAY_SIZE(stage_sizes));
   return stage_sizes[stage];
}
