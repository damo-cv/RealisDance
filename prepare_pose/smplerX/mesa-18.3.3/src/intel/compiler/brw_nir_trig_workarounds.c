#include "brw_nir.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_search.h"
#include "nir_search_helpers.h"

#ifndef NIR_OPT_ALGEBRAIC_STRUCT_DEFS
#define NIR_OPT_ALGEBRAIC_STRUCT_DEFS

struct transform {
   const nir_search_expression *search;
   const nir_search_value *replace;
   unsigned condition_offset;
};

#endif

   
static const nir_search_variable search0_0 = {
   { nir_search_value_variable, 0 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsin,
   { &search0_0.value },
   NULL,
};
   
static const nir_search_variable replace0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsin,
   { &replace0_0_0.value },
   NULL,
};

static const nir_search_constant replace0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3fefffc115df6556 /* 0.99997 */ },
};
static const nir_search_expression replace0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace0_0.value, &replace0_1.value },
   NULL,
};

static const struct transform brw_nir_apply_trig_workarounds_fsin_xforms[] = {
   { &search0, &replace0.value, 0 },
};
   
static const nir_search_variable search1_0 = {
   { nir_search_value_variable, 0 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fcos,
   { &search1_0.value },
   NULL,
};
   
static const nir_search_variable replace1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fcos,
   { &replace1_0_0.value },
   NULL,
};

static const nir_search_constant replace1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3fefffc115df6556 /* 0.99997 */ },
};
static const nir_search_expression replace1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace1_0.value, &replace1_1.value },
   NULL,
};

static const struct transform brw_nir_apply_trig_workarounds_fcos_xforms[] = {
   { &search1, &replace1.value, 0 },
};

static bool
brw_nir_apply_trig_workarounds_block(nir_builder *build, nir_block *block,
                   const bool *condition_flags)
{
   bool progress = false;

   nir_foreach_instr_reverse_safe(instr, block) {
      if (instr->type != nir_instr_type_alu)
         continue;

      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (!alu->dest.dest.is_ssa)
         continue;

      switch (alu->op) {
      case nir_op_fsin:
         for (unsigned i = 0; i < ARRAY_SIZE(brw_nir_apply_trig_workarounds_fsin_xforms); i++) {
            const struct transform *xform = &brw_nir_apply_trig_workarounds_fsin_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fcos:
         for (unsigned i = 0; i < ARRAY_SIZE(brw_nir_apply_trig_workarounds_fcos_xforms); i++) {
            const struct transform *xform = &brw_nir_apply_trig_workarounds_fcos_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      default:
         break;
      }
   }

   return progress;
}

static bool
brw_nir_apply_trig_workarounds_impl(nir_function_impl *impl, const bool *condition_flags)
{
   bool progress = false;

   nir_builder build;
   nir_builder_init(&build, impl);

   nir_foreach_block_reverse(block, impl) {
      progress |= brw_nir_apply_trig_workarounds_block(&build, block, condition_flags);
   }

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);

   return progress;
}


bool
brw_nir_apply_trig_workarounds(nir_shader *shader)
{
   bool progress = false;
   bool condition_flags[1];
   const nir_shader_compiler_options *options = shader->options;
   (void) options;

   condition_flags[0] = true;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= brw_nir_apply_trig_workarounds_impl(function->impl, condition_flags);
   }

   return progress;
}

