
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
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search0_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   true,
   nir_type_invalid,
   (is_pos_power_of_two),
};
static const nir_search_expression search0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search0_0.value, &search0_1.value },
   NULL,
};
   
static const nir_search_variable replace0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &replace0_1_0.value },
   NULL,
};
static const nir_search_expression replace0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace0_0.value, &replace0_1.value },
   NULL,
};
   
static const nir_search_variable search1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search1_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   true,
   nir_type_invalid,
   (is_neg_power_of_two),
};
static const nir_search_expression search1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search1_0.value, &search1_1.value },
   NULL,
};
   
static const nir_search_variable replace1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace1_0_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace1_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace1_0_1_0_0.value },
   NULL,
};
static const nir_search_expression replace1_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &replace1_0_1_0.value },
   NULL,
};
static const nir_search_expression replace1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace1_0_0.value, &replace1_0_1.value },
   NULL,
};
static const nir_search_expression replace1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &replace1_0.value },
   NULL,
};
   
static const nir_search_variable search30_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search30_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search30 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search30_0.value, &search30_1.value },
   NULL,
};
   
static const nir_search_constant replace30 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
   
static const nir_search_variable search34_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search34_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression search34 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search34_0.value, &search34_1.value },
   NULL,
};
   
static const nir_search_variable replace34 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search36_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search36_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search36 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search36_0.value, &search36_1.value },
   NULL,
};
   
static const nir_search_variable replace36_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace36 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &replace36_0.value },
   NULL,
};
   
static const nir_search_variable search275_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search275_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search275_0_0.value },
   NULL,
};

static const nir_search_variable search275_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search275_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search275_1_0.value },
   NULL,
};
static const nir_search_expression search275 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search275_0.value, &search275_1.value },
   NULL,
};
   
static const nir_search_variable replace275_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace275_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace275_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace275_0_0.value, &replace275_0_1.value },
   NULL,
};
static const nir_search_expression replace275 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace275_0.value },
   NULL,
};
   
static const nir_search_variable search401_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search401_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search401_0_0.value },
   NULL,
};

static const nir_search_variable search401_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search401 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search401_0.value, &search401_1.value },
   NULL,
};
   
static const nir_search_variable replace401_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace401_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace401_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace401_0_0.value, &replace401_0_1.value },
   NULL,
};
static const nir_search_expression replace401 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &replace401_0.value },
   NULL,
};
   
static const nir_search_variable search403_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   (is_not_const),
};

static const nir_search_variable search403_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   (is_not_const),
};
static const nir_search_expression search403_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search403_0_0.value, &search403_0_1.value },
   (is_used_once),
};

static const nir_search_variable search403_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search403 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search403_0.value, &search403_1.value },
   (is_used_once),
};
   
static const nir_search_variable replace403_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace403_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace403_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace403_0_0.value, &replace403_0_1.value },
   NULL,
};

static const nir_search_variable replace403_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace403 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace403_0.value, &replace403_1.value },
   NULL,
};
   
static const nir_search_variable search407_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search407_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search407_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search407_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search407_1_0.value, &search407_1_1.value },
   NULL,
};
static const nir_search_expression search407 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search407_0.value, &search407_1.value },
   NULL,
};
   
static const nir_search_variable replace407_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace407_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace407_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace407_0_0.value, &replace407_0_1.value },
   NULL,
};

static const nir_search_variable replace407_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace407 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace407_0.value, &replace407_1.value },
   NULL,
};
   
static const nir_search_variable search523_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search523_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search523_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search523_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search523_0_0.value, &search523_0_1.value, &search523_0_2.value },
   (is_used_once),
};

static const nir_search_variable search523_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search523 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search523_0.value, &search523_1.value },
   NULL,
};
   
static const nir_search_variable replace523_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace523_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace523_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace523_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace523_1_0.value, &replace523_1_1.value },
   NULL,
};

static const nir_search_variable replace523_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace523_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace523_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace523_2_0.value, &replace523_2_1.value },
   NULL,
};
static const nir_search_expression replace523 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace523_0.value, &replace523_1.value, &replace523_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_imul_xforms[] = {
   { &search0, &replace0.value, 0 },
   { &search1, &replace1.value, 0 },
   { &search30, &replace30.value, 0 },
   { &search34, &replace34.value, 0 },
   { &search36, &replace36.value, 0 },
   { &search275, &replace275.value, 0 },
   { &search401, &replace401.value, 0 },
   { &search403, &replace403.value, 0 },
   { &search407, &replace407.value, 0 },
   { &search523, &replace523.value, 0 },
};
   
static const nir_search_variable search2_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search2_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression search2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_udiv,
   { &search2_0.value, &search2_1.value },
   NULL,
};
   
static const nir_search_variable replace2 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search6_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search6_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   true,
   nir_type_invalid,
   (is_pos_power_of_two),
};
static const nir_search_expression search6 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_udiv,
   { &search6_0.value, &search6_1.value },
   NULL,
};
   
static const nir_search_variable replace6_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace6_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace6_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &replace6_1_0.value },
   NULL,
};
static const nir_search_expression replace6 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &replace6_0.value, &replace6_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_udiv_xforms[] = {
   { &search2, &replace2.value, 0 },
   { &search6, &replace6.value, 0 },
};
   
static const nir_search_variable search3_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search3_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression search3 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_idiv,
   { &search3_0.value, &search3_1.value },
   NULL,
};
   
static const nir_search_variable replace3 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search7_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search7_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   true,
   nir_type_invalid,
   (is_pos_power_of_two),
};
static const nir_search_expression search7 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_idiv,
   { &search7_0.value, &search7_1.value },
   NULL,
};
   
static const nir_search_variable replace7_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace7_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isign,
   { &replace7_0_0.value },
   NULL,
};

static const nir_search_variable replace7_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace7_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace7_1_0_0.value },
   NULL,
};

static const nir_search_variable replace7_1_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace7_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &replace7_1_1_0.value },
   NULL,
};
static const nir_search_expression replace7_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &replace7_1_0.value, &replace7_1_1.value },
   NULL,
};
static const nir_search_expression replace7 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace7_0.value, &replace7_1.value },
   NULL,
};
   
static const nir_search_variable search8_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search8_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   true,
   nir_type_invalid,
   (is_neg_power_of_two),
};
static const nir_search_expression search8 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_idiv,
   { &search8_0.value, &search8_1.value },
   NULL,
};
   
static const nir_search_variable replace8_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace8_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isign,
   { &replace8_0_0_0.value },
   NULL,
};

static const nir_search_variable replace8_0_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace8_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace8_0_1_0_0.value },
   NULL,
};

static const nir_search_variable replace8_0_1_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace8_0_1_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace8_0_1_1_0_0.value },
   NULL,
};
static const nir_search_expression replace8_0_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &replace8_0_1_1_0.value },
   NULL,
};
static const nir_search_expression replace8_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &replace8_0_1_0.value, &replace8_0_1_1.value },
   NULL,
};
static const nir_search_expression replace8_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace8_0_0.value, &replace8_0_1.value },
   NULL,
};
static const nir_search_expression replace8 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &replace8_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_idiv_xforms[] = {
   { &search3, &replace3.value, 0 },
   { &search7, &replace7.value, 1 },
   { &search8, &replace8.value, 1 },
};
   
static const nir_search_variable search4_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search4_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression search4 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umod,
   { &search4_0.value, &search4_1.value },
   NULL,
};
   
static const nir_search_constant replace4 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
   
static const nir_search_variable search9_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search9_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   (is_pos_power_of_two),
};
static const nir_search_expression search9 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umod,
   { &search9_0.value, &search9_1.value },
   NULL,
};
   
static const nir_search_variable replace9_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace9_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace9_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace9_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace9_1_0.value, &replace9_1_1.value },
   NULL,
};
static const nir_search_expression replace9 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace9_0.value, &replace9_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_umod_xforms[] = {
   { &search4, &replace4.value, 0 },
   { &search9, &replace9.value, 0 },
};
   
static const nir_search_variable search5_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search5_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression search5 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imod,
   { &search5_0.value, &search5_1.value },
   NULL,
};
   
static const nir_search_constant replace5 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const struct transform nir_opt_algebraic_imod_xforms[] = {
   { &search5, &replace5.value, 0 },
};
   
static const nir_search_variable search10_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search10_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search10_0_0.value },
   NULL,
};
static const nir_search_expression search10 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search10_0.value },
   NULL,
};
   
static const nir_search_variable replace10 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search394_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search394 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search394_0.value },
   NULL,
};
   
static const nir_search_constant replace394_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable replace394_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace394 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &replace394_0.value, &replace394_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fneg_xforms[] = {
   { &search10, &replace10.value, 0 },
   { &search394, &replace394.value, 21 },
};
   
static const nir_search_variable search11_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search11_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search11_0_0.value },
   NULL,
};
static const nir_search_expression search11 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search11_0.value },
   NULL,
};
   
static const nir_search_variable replace11 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search279_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search279_0 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_b2i,
   { &search279_0_0.value },
   NULL,
};
static const nir_search_expression search279 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search279_0.value },
   NULL,
};
   
static const nir_search_variable replace279 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search395_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search395 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search395_0.value },
   NULL,
};
   
static const nir_search_constant replace395_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable replace395_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace395 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace395_0.value, &replace395_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ineg_xforms[] = {
   { &search11, &replace11.value, 0 },
   { &search279, &replace279.value, 0 },
   { &search395, &replace395.value, 21 },
};
   
static const nir_search_variable search12_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search12_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search12_0_0.value },
   NULL,
};
static const nir_search_expression search12 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search12_0.value },
   NULL,
};
   
static const nir_search_variable replace12_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace12 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace12_0.value },
   NULL,
};
   
static const nir_search_variable search13_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search13_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search13_0_0.value },
   NULL,
};
static const nir_search_expression search13 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search13_0.value },
   NULL,
};
   
static const nir_search_variable replace13_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace13 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace13_0.value },
   NULL,
};
   
static const nir_search_variable search14_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search14_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &search14_0_0.value },
   NULL,
};
static const nir_search_expression search14 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search14_0.value },
   NULL,
};
   
static const nir_search_variable replace14_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace14 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &replace14_0.value },
   NULL,
};
   
static const nir_search_variable search265_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search265_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search265_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_slt,
   { &search265_0_0.value, &search265_0_1.value },
   NULL,
};
static const nir_search_expression search265 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search265_0.value },
   NULL,
};
   
static const nir_search_variable replace265_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace265_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace265 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_slt,
   { &replace265_0.value, &replace265_1.value },
   NULL,
};
   
static const nir_search_variable search266_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search266_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search266_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_sge,
   { &search266_0_0.value, &search266_0_1.value },
   NULL,
};
static const nir_search_expression search266 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search266_0.value },
   NULL,
};
   
static const nir_search_variable replace266_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace266_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace266 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_sge,
   { &replace266_0.value, &replace266_1.value },
   NULL,
};
   
static const nir_search_variable search267_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search267_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search267_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_seq,
   { &search267_0_0.value, &search267_0_1.value },
   NULL,
};
static const nir_search_expression search267 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search267_0.value },
   NULL,
};
   
static const nir_search_variable replace267_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace267_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace267 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_seq,
   { &replace267_0.value, &replace267_1.value },
   NULL,
};
   
static const nir_search_variable search268_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search268_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search268_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_sne,
   { &search268_0_0.value, &search268_0_1.value },
   NULL,
};
static const nir_search_expression search268 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search268_0.value },
   NULL,
};
   
static const nir_search_variable replace268_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace268_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace268 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_sne,
   { &replace268_0.value, &replace268_1.value },
   NULL,
};
   
static const nir_search_variable search362_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search362_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search362_0_0.value },
   NULL,
};
static const nir_search_expression search362 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search362_0.value },
   NULL,
};
   
static const nir_search_variable replace362_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace362 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace362_0.value },
   NULL,
};
   
static const nir_search_constant search398_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search398_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search398_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &search398_0_0.value, &search398_0_1.value },
   NULL,
};
static const nir_search_expression search398 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search398_0.value },
   NULL,
};
   
static const nir_search_variable replace398_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace398 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace398_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fabs_xforms[] = {
   { &search12, &replace12.value, 0 },
   { &search13, &replace13.value, 0 },
   { &search14, &replace14.value, 0 },
   { &search265, &replace265.value, 0 },
   { &search266, &replace266.value, 0 },
   { &search267, &replace267.value, 0 },
   { &search268, &replace268.value, 0 },
   { &search362, &replace362.value, 0 },
   { &search398, &replace398.value, 0 },
};
   
static const nir_search_variable search15_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search15_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &search15_0_0.value },
   NULL,
};
static const nir_search_expression search15 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &search15_0.value },
   NULL,
};
   
static const nir_search_variable replace15_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace15 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace15_0.value },
   NULL,
};
   
static const nir_search_variable search16_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search16_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search16_0_0.value },
   NULL,
};
static const nir_search_expression search16 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &search16_0.value },
   NULL,
};
   
static const nir_search_variable replace16_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace16 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace16_0.value },
   NULL,
};
   
static const nir_search_variable search363_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search363_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search363_0_0.value },
   NULL,
};
static const nir_search_expression search363 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &search363_0.value },
   NULL,
};
   
static const nir_search_variable replace363_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace363 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace363_0.value },
   NULL,
};
   
static const nir_search_constant search399_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable search399_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search399_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &search399_0_0.value, &search399_0_1.value },
   NULL,
};
static const nir_search_expression search399 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &search399_0.value },
   NULL,
};
   
static const nir_search_variable replace399_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace399 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace399_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_iabs_xforms[] = {
   { &search15, &replace15.value, 0 },
   { &search16, &replace16.value, 0 },
   { &search363, &replace363.value, 0 },
   { &search399, &replace399.value, 0 },
};
   
static const nir_search_variable search17_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search17_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search17 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search17_0.value, &search17_1.value },
   NULL,
};
   
static const nir_search_variable replace17 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search21_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search21_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search21_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search21_0_0.value, &search21_0_1.value },
   NULL,
};

static const nir_search_variable search21_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search21_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search21_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search21_1_0.value, &search21_1_1.value },
   NULL,
};
static const nir_search_expression search21 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search21_0.value, &search21_1.value },
   NULL,
};
   
static const nir_search_variable replace21_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace21_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace21_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace21_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace21_1_0.value, &replace21_1_1.value },
   NULL,
};
static const nir_search_expression replace21 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace21_0.value, &replace21_1.value },
   NULL,
};
   
static const nir_search_variable search23_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search23_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search23_0_0.value },
   NULL,
};

static const nir_search_variable search23_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search23 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search23_0.value, &search23_1.value },
   NULL,
};
   
static const nir_search_constant replace23 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
   
static const nir_search_variable search27_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search27_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search27_0_0.value },
   NULL,
};

static const nir_search_variable search27_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search27_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search27_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search27_1_0.value, &search27_1_1.value },
   NULL,
};
static const nir_search_expression search27 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search27_0.value, &search27_1.value },
   NULL,
};
   
static const nir_search_variable replace27 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search28_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search28_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search28_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search28_1_0_0.value },
   NULL,
};

static const nir_search_variable search28_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search28_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search28_1_0.value, &search28_1_1.value },
   NULL,
};
static const nir_search_expression search28 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search28_0.value, &search28_1.value },
   NULL,
};
   
static const nir_search_variable replace28 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search53_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search53_0_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_variable search53_0_1_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search53_0_1_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search53_0_1_1_0_0.value },
   NULL,
};
static const nir_search_expression search53_0_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search53_0_1_1_0.value },
   NULL,
};
static const nir_search_expression search53_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search53_0_1_0.value, &search53_0_1_1.value },
   NULL,
};
static const nir_search_expression search53_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search53_0_0.value, &search53_0_1.value },
   NULL,
};

static const nir_search_variable search53_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search53_1_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search53_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search53_1_1_0.value },
   NULL,
};
static const nir_search_expression search53_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search53_1_0.value, &search53_1_1.value },
   NULL,
};
static const nir_search_expression search53 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search53_0.value, &search53_1.value },
   NULL,
};
   
static const nir_search_variable replace53_0 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace53_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace53_2 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace53 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace53_0.value, &replace53_1.value, &replace53_2.value },
   NULL,
};
   
static const nir_search_variable search54_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search54_0_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_variable search54_0_1_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search54_0_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search54_0_1_1_0.value },
   NULL,
};
static const nir_search_expression search54_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search54_0_1_0.value, &search54_0_1_1.value },
   NULL,
};
static const nir_search_expression search54_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search54_0_0.value, &search54_0_1.value },
   NULL,
};

static const nir_search_variable search54_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search54_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search54_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search54_1_0.value, &search54_1_1.value },
   NULL,
};
static const nir_search_expression search54 = {
   { nir_search_value_expression, 32 },
   true,
   nir_op_fadd,
   { &search54_0.value, &search54_1.value },
   NULL,
};
   
static const nir_search_variable replace54_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace54_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace54_2 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace54 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flrp,
   { &replace54_0.value, &replace54_1.value, &replace54_2.value },
   NULL,
};
   
static const nir_search_variable search55_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search55_0_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_variable search55_0_1_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search55_0_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search55_0_1_1_0.value },
   NULL,
};
static const nir_search_expression search55_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search55_0_1_0.value, &search55_0_1_1.value },
   NULL,
};
static const nir_search_expression search55_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search55_0_0.value, &search55_0_1.value },
   NULL,
};

static const nir_search_variable search55_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search55_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search55_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search55_1_0.value, &search55_1_1.value },
   NULL,
};
static const nir_search_expression search55 = {
   { nir_search_value_expression, 64 },
   true,
   nir_op_fadd,
   { &search55_0.value, &search55_1.value },
   NULL,
};
   
static const nir_search_variable replace55_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace55_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace55_2 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace55 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flrp,
   { &replace55_0.value, &replace55_1.value, &replace55_2.value },
   NULL,
};
   
static const nir_search_variable search56_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search56_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search56_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search56_1_0_0.value },
   NULL,
};

static const nir_search_variable search56_1_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search56_1_1_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search56_1_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search56_1_1_1_0.value },
   NULL,
};
static const nir_search_expression search56_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search56_1_1_0.value, &search56_1_1_1.value },
   NULL,
};
static const nir_search_expression search56_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search56_1_0.value, &search56_1_1.value },
   NULL,
};
static const nir_search_expression search56 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search56_0.value, &search56_1.value },
   NULL,
};
   
static const nir_search_variable replace56_0 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace56_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace56_2 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace56 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace56_0.value, &replace56_1.value, &replace56_2.value },
   NULL,
};
   
static const nir_search_variable search57_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search57_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search57_1_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search57_1_1_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search57_1_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search57_1_1_1_0.value },
   NULL,
};
static const nir_search_expression search57_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search57_1_1_0.value, &search57_1_1_1.value },
   NULL,
};
static const nir_search_expression search57_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search57_1_0.value, &search57_1_1.value },
   NULL,
};
static const nir_search_expression search57 = {
   { nir_search_value_expression, 32 },
   true,
   nir_op_fadd,
   { &search57_0.value, &search57_1.value },
   NULL,
};
   
static const nir_search_variable replace57_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace57_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace57_2 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace57 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flrp,
   { &replace57_0.value, &replace57_1.value, &replace57_2.value },
   NULL,
};
   
static const nir_search_variable search58_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search58_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search58_1_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search58_1_1_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search58_1_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search58_1_1_1_0.value },
   NULL,
};
static const nir_search_expression search58_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search58_1_1_0.value, &search58_1_1_1.value },
   NULL,
};
static const nir_search_expression search58_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search58_1_0.value, &search58_1_1.value },
   NULL,
};
static const nir_search_expression search58 = {
   { nir_search_value_expression, 64 },
   true,
   nir_op_fadd,
   { &search58_0.value, &search58_1.value },
   NULL,
};
   
static const nir_search_variable replace58_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace58_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace58_2 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace58 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flrp,
   { &replace58_0.value, &replace58_1.value, &replace58_2.value },
   NULL,
};
   
static const nir_search_variable search60_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search60_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search60_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search60_0_0.value, &search60_0_1.value },
   NULL,
};

static const nir_search_variable search60_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search60 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search60_0.value, &search60_1.value },
   NULL,
};
   
static const nir_search_variable replace60_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace60_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace60_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace60 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ffma,
   { &replace60_0.value, &replace60_1.value, &replace60_2.value },
   NULL,
};
   
static const nir_search_variable search396_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search396_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search396_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search396_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &search396_1_0.value, &search396_1_1.value },
   NULL,
};
static const nir_search_expression search396 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search396_0.value, &search396_1.value },
   NULL,
};
   
static const nir_search_variable replace396_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace396_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace396 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &replace396_0.value, &replace396_1.value },
   NULL,
};
   
static const nir_search_variable search404_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   (is_not_const),
};

static const nir_search_variable search404_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   (is_not_const),
};
static const nir_search_expression search404_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search404_0_0.value, &search404_0_1.value },
   (is_used_once),
};

static const nir_search_variable search404_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search404 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search404_0.value, &search404_1.value },
   (is_used_once),
};
   
static const nir_search_variable replace404_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace404_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace404_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace404_0_0.value, &replace404_0_1.value },
   NULL,
};

static const nir_search_variable replace404_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace404 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace404_0.value, &replace404_1.value },
   NULL,
};
   
static const nir_search_variable search408_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search408_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search408_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search408_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search408_1_0.value, &search408_1_1.value },
   NULL,
};
static const nir_search_expression search408 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search408_0.value, &search408_1.value },
   NULL,
};
   
static const nir_search_variable replace408_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace408_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace408_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace408_0_0.value, &replace408_0_1.value },
   NULL,
};

static const nir_search_variable replace408_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace408 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace408_0.value, &replace408_1.value },
   NULL,
};
   
static const nir_search_variable search409_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search409_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search409_1_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search409_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search409_1_0_0.value, &search409_1_0_1.value },
   NULL,
};
static const nir_search_expression search409_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search409_1_0.value },
   NULL,
};
static const nir_search_expression search409 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search409_0.value, &search409_1.value },
   NULL,
};
   
static const nir_search_variable replace409_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace409_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace409_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace409_0_1_0.value },
   NULL,
};
static const nir_search_expression replace409_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace409_0_0.value, &replace409_0_1.value },
   NULL,
};

static const nir_search_variable replace409_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace409_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace409_1_0.value },
   NULL,
};
static const nir_search_expression replace409 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace409_0.value, &replace409_1.value },
   NULL,
};
   
static const nir_search_variable search520_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search520_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search520_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search520_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search520_0_0.value, &search520_0_1.value, &search520_0_2.value },
   (is_used_once),
};

static const nir_search_variable search520_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search520 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search520_0.value, &search520_1.value },
   NULL,
};
   
static const nir_search_variable replace520_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace520_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace520_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace520_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace520_1_0.value, &replace520_1_1.value },
   NULL,
};

static const nir_search_variable replace520_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace520_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace520_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace520_2_0.value, &replace520_2_1.value },
   NULL,
};
static const nir_search_expression replace520 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace520_0.value, &replace520_1.value, &replace520_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fadd_xforms[] = {
   { &search17, &replace17.value, 0 },
   { &search21, &replace21.value, 0 },
   { &search23, &replace23.value, 0 },
   { &search27, &replace27.value, 0 },
   { &search28, &replace28.value, 0 },
   { &search53, &replace53.value, 2 },
   { &search54, &replace54.value, 5 },
   { &search55, &replace55.value, 6 },
   { &search56, &replace56.value, 2 },
   { &search57, &replace57.value, 5 },
   { &search58, &replace58.value, 6 },
   { &search60, &replace60.value, 8 },
   { &search396, &replace396.value, 0 },
   { &search404, &replace404.value, 0 },
   { &search408, &replace408.value, 0 },
   { &search409, &replace409.value, 0 },
   { &search520, &replace520.value, 0 },
};
   
static const nir_search_variable search18_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search18_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search18 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search18_0.value, &search18_1.value },
   NULL,
};
   
static const nir_search_variable replace18 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search22_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search22_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search22_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search22_0_0.value, &search22_0_1.value },
   NULL,
};

static const nir_search_variable search22_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search22_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search22_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search22_1_0.value, &search22_1_1.value },
   NULL,
};
static const nir_search_expression search22 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search22_0.value, &search22_1.value },
   NULL,
};
   
static const nir_search_variable replace22_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace22_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace22_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace22_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace22_1_0.value, &replace22_1_1.value },
   NULL,
};
static const nir_search_expression replace22 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace22_0.value, &replace22_1.value },
   NULL,
};
   
static const nir_search_variable search24_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search24_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search24_0_0.value },
   NULL,
};

static const nir_search_variable search24_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search24 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search24_0.value, &search24_1.value },
   NULL,
};
   
static const nir_search_constant replace24 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
   
static const nir_search_variable search25_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search25_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search25_0_0.value },
   NULL,
};

static const nir_search_variable search25_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search25_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search25_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search25_1_0.value, &search25_1_1.value },
   NULL,
};
static const nir_search_expression search25 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search25_0.value, &search25_1.value },
   NULL,
};
   
static const nir_search_variable replace25 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search26_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search26_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search26_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search26_1_0_0.value },
   NULL,
};

static const nir_search_variable search26_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search26_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search26_1_0.value, &search26_1_1.value },
   NULL,
};
static const nir_search_expression search26 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search26_0.value, &search26_1.value },
   NULL,
};
   
static const nir_search_variable replace26 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search397_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search397_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable search397_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search397_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &search397_1_0.value, &search397_1_1.value },
   NULL,
};
static const nir_search_expression search397 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search397_0.value, &search397_1.value },
   NULL,
};
   
static const nir_search_variable replace397_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace397_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace397 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace397_0.value, &replace397_1.value },
   NULL,
};
   
static const nir_search_variable search405_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   (is_not_const),
};

static const nir_search_variable search405_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   (is_not_const),
};
static const nir_search_expression search405_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search405_0_0.value, &search405_0_1.value },
   (is_used_once),
};

static const nir_search_variable search405_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search405 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search405_0.value, &search405_1.value },
   (is_used_once),
};
   
static const nir_search_variable replace405_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace405_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace405_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace405_0_0.value, &replace405_0_1.value },
   NULL,
};

static const nir_search_variable replace405_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace405 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace405_0.value, &replace405_1.value },
   NULL,
};
   
static const nir_search_variable search410_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search410_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search410_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search410_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search410_1_0.value, &search410_1_1.value },
   NULL,
};
static const nir_search_expression search410 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search410_0.value, &search410_1.value },
   NULL,
};
   
static const nir_search_variable replace410_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace410_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace410_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace410_0_0.value, &replace410_0_1.value },
   NULL,
};

static const nir_search_variable replace410_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace410 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace410_0.value, &replace410_1.value },
   NULL,
};
   
static const nir_search_variable search522_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search522_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search522_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search522_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search522_0_0.value, &search522_0_1.value, &search522_0_2.value },
   (is_used_once),
};

static const nir_search_variable search522_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search522 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search522_0.value, &search522_1.value },
   NULL,
};
   
static const nir_search_variable replace522_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace522_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace522_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace522_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace522_1_0.value, &replace522_1_1.value },
   NULL,
};

static const nir_search_variable replace522_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace522_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace522_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace522_2_0.value, &replace522_2_1.value },
   NULL,
};
static const nir_search_expression replace522 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace522_0.value, &replace522_1.value, &replace522_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_iadd_xforms[] = {
   { &search18, &replace18.value, 0 },
   { &search22, &replace22.value, 0 },
   { &search24, &replace24.value, 0 },
   { &search25, &replace25.value, 0 },
   { &search26, &replace26.value, 0 },
   { &search397, &replace397.value, 0 },
   { &search405, &replace405.value, 0 },
   { &search410, &replace410.value, 0 },
   { &search522, &replace522.value, 0 },
};
   
static const nir_search_variable search19_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search19_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search19 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_usadd_4x8,
   { &search19_0.value, &search19_1.value },
   NULL,
};
   
static const nir_search_variable replace19 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search20_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search20_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search20 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_usadd_4x8,
   { &search20_0.value, &search20_1.value },
   NULL,
};
   
static const nir_search_constant replace20 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};

static const struct transform nir_opt_algebraic_usadd_4x8_xforms[] = {
   { &search19, &replace19.value, 0 },
   { &search20, &replace20.value, 0 },
};
   
static const nir_search_variable search29_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search29_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search29 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fmul,
   { &search29_0.value, &search29_1.value },
   NULL,
};
   
static const nir_search_constant replace29 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
   
static const nir_search_variable search33_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search33_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression search33 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search33_0.value, &search33_1.value },
   NULL,
};
   
static const nir_search_variable replace33 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search35_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search35_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0xbff0000000000000L /* -1.0 */ },
};
static const nir_search_expression search35 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search35_0.value, &search35_1.value },
   NULL,
};
   
static const nir_search_variable replace35_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace35 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace35_0.value },
   NULL,
};
   
static const nir_search_variable search37_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search37_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsign,
   { &search37_0_0.value },
   NULL,
};

static const nir_search_variable search37_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search37_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search37_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search37_1_0.value, &search37_1_1.value },
   NULL,
};
static const nir_search_expression search37 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search37_0.value, &search37_1.value },
   NULL,
};
   
static const nir_search_variable replace37_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace37_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace37_0_0.value },
   NULL,
};

static const nir_search_variable replace37_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace37 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace37_0.value, &replace37_1.value },
   NULL,
};
   
static const nir_search_variable search38_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search38_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsign,
   { &search38_0_0_0.value },
   NULL,
};

static const nir_search_variable search38_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search38_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search38_0_0.value, &search38_0_1.value },
   NULL,
};

static const nir_search_variable search38_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search38 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search38_0.value, &search38_1.value },
   NULL,
};
   
static const nir_search_variable replace38_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace38_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace38_0_0.value },
   NULL,
};

static const nir_search_variable replace38_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace38 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace38_0.value, &replace38_1.value },
   NULL,
};
   
static const nir_search_variable search276_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search276_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search276_0_0.value },
   NULL,
};

static const nir_search_variable search276_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search276_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search276_1_0.value },
   NULL,
};
static const nir_search_expression search276 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search276_0.value, &search276_1.value },
   NULL,
};
   
static const nir_search_variable replace276_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace276_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace276_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace276_0_0.value, &replace276_0_1.value },
   NULL,
};
static const nir_search_expression replace276 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace276_0.value },
   NULL,
};
   
static const nir_search_variable search333_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search333_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &search333_0_0.value },
   (is_used_once),
};

static const nir_search_variable search333_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search333_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &search333_1_0.value },
   (is_used_once),
};
static const nir_search_expression search333 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fmul,
   { &search333_0.value, &search333_1.value },
   NULL,
};
   
static const nir_search_variable replace333_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace333_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace333_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace333_0_0.value, &replace333_0_1.value },
   NULL,
};
static const nir_search_expression replace333 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &replace333_0.value },
   NULL,
};
   
static const nir_search_variable search400_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search400_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search400_0_0.value },
   NULL,
};

static const nir_search_variable search400_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search400 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search400_0.value, &search400_1.value },
   NULL,
};
   
static const nir_search_variable replace400_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace400_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace400_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace400_0_0.value, &replace400_0_1.value },
   NULL,
};
static const nir_search_expression replace400 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace400_0.value },
   NULL,
};
   
static const nir_search_variable search402_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   (is_not_const),
};

static const nir_search_variable search402_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   (is_not_const),
};
static const nir_search_expression search402_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search402_0_0.value, &search402_0_1.value },
   (is_used_once),
};

static const nir_search_variable search402_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search402 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fmul,
   { &search402_0.value, &search402_1.value },
   (is_used_once),
};
   
static const nir_search_variable replace402_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace402_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace402_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace402_0_0.value, &replace402_0_1.value },
   NULL,
};

static const nir_search_variable replace402_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace402 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace402_0.value, &replace402_1.value },
   NULL,
};
   
static const nir_search_variable search406_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search406_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search406_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search406_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search406_1_0.value, &search406_1_1.value },
   NULL,
};
static const nir_search_expression search406 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fmul,
   { &search406_0.value, &search406_1.value },
   NULL,
};
   
static const nir_search_variable replace406_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace406_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace406_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace406_0_0.value, &replace406_0_1.value },
   NULL,
};

static const nir_search_variable replace406_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace406 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace406_0.value, &replace406_1.value },
   NULL,
};
   
static const nir_search_variable search521_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search521_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search521_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search521_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search521_0_0.value, &search521_0_1.value, &search521_0_2.value },
   (is_used_once),
};

static const nir_search_variable search521_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search521 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search521_0.value, &search521_1.value },
   NULL,
};
   
static const nir_search_variable replace521_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace521_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace521_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace521_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace521_1_0.value, &replace521_1_1.value },
   NULL,
};

static const nir_search_variable replace521_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace521_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace521_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace521_2_0.value, &replace521_2_1.value },
   NULL,
};
static const nir_search_expression replace521 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace521_0.value, &replace521_1.value, &replace521_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fmul_xforms[] = {
   { &search29, &replace29.value, 0 },
   { &search33, &replace33.value, 0 },
   { &search35, &replace35.value, 0 },
   { &search37, &replace37.value, 0 },
   { &search38, &replace38.value, 0 },
   { &search276, &replace276.value, 0 },
   { &search333, &replace333.value, 0 },
   { &search400, &replace400.value, 0 },
   { &search402, &replace402.value, 0 },
   { &search406, &replace406.value, 0 },
   { &search521, &replace521.value, 0 },
};
   
static const nir_search_variable search31_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search31_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search31 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umul_unorm_4x8,
   { &search31_0.value, &search31_1.value },
   NULL,
};
   
static const nir_search_constant replace31 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
   
static const nir_search_variable search32_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search32_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search32 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umul_unorm_4x8,
   { &search32_0.value, &search32_1.value },
   NULL,
};
   
static const nir_search_variable replace32 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_umul_unorm_4x8_xforms[] = {
   { &search31, &replace31.value, 0 },
   { &search32, &replace32.value, 0 },
};
   
static const nir_search_constant search39_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search39_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search39_2 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search39 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ffma,
   { &search39_0.value, &search39_1.value, &search39_2.value },
   NULL,
};
   
static const nir_search_variable replace39 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search40_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search40_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search40_2 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search40 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ffma,
   { &search40_0.value, &search40_1.value, &search40_2.value },
   NULL,
};
   
static const nir_search_variable replace40 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search41_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search41_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search41_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search41 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ffma,
   { &search41_0.value, &search41_1.value, &search41_2.value },
   NULL,
};
   
static const nir_search_variable replace41_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace41_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace41 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace41_0.value, &replace41_1.value },
   NULL,
};
   
static const nir_search_variable search42_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search42_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_variable search42_2 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search42 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ffma,
   { &search42_0.value, &search42_1.value, &search42_2.value },
   NULL,
};
   
static const nir_search_variable replace42_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace42_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace42 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace42_0.value, &replace42_1.value },
   NULL,
};
   
static const nir_search_constant search43_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_variable search43_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search43_2 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search43 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ffma,
   { &search43_0.value, &search43_1.value, &search43_2.value },
   NULL,
};
   
static const nir_search_variable replace43_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace43_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace43 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace43_0.value, &replace43_1.value },
   NULL,
};
   
static const nir_search_variable search59_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search59_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search59_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search59 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ffma,
   { &search59_0.value, &search59_1.value, &search59_2.value },
   NULL,
};
   
static const nir_search_variable replace59_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace59_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace59_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace59_0_0.value, &replace59_0_1.value },
   NULL,
};

static const nir_search_variable replace59_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace59 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace59_0.value, &replace59_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ffma_xforms[] = {
   { &search39, &replace39.value, 0 },
   { &search40, &replace40.value, 0 },
   { &search41, &replace41.value, 0 },
   { &search42, &replace42.value, 0 },
   { &search43, &replace43.value, 0 },
   { &search59, &replace59.value, 7 },
};
   
static const nir_search_variable search44_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search44_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search44_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search44 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flrp,
   { &search44_0.value, &search44_1.value, &search44_2.value },
   NULL,
};
   
static const nir_search_variable replace44 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search45_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search45_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search45_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression search45 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flrp,
   { &search45_0.value, &search45_1.value, &search45_2.value },
   NULL,
};
   
static const nir_search_variable replace45 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search46_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search46_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search46_2 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search46 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flrp,
   { &search46_0.value, &search46_1.value, &search46_2.value },
   NULL,
};
   
static const nir_search_variable replace46 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_constant search47_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search47_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search47_2 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search47 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flrp,
   { &search47_0.value, &search47_1.value, &search47_2.value },
   NULL,
};
   
static const nir_search_variable replace47_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace47_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace47 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace47_0.value, &replace47_1.value },
   NULL,
};
   
static const nir_search_variable search48_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search48_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search48_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search48_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search48_2_0.value },
   NULL,
};
static const nir_search_expression search48 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flrp,
   { &search48_0.value, &search48_1.value, &search48_2.value },
   NULL,
};
   
static const nir_search_variable replace48_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace48_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace48_2 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace48 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace48_0.value, &replace48_1.value, &replace48_2.value },
   NULL,
};
   
static const nir_search_variable search49_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search49_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search49_2 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search49 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flrp,
   { &search49_0.value, &search49_1.value, &search49_2.value },
   NULL,
};
   
static const nir_search_variable replace49_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace49_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace49_0_0_0.value },
   NULL,
};

static const nir_search_variable replace49_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace49_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace49_0_0.value, &replace49_0_1.value },
   NULL,
};

static const nir_search_variable replace49_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace49 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace49_0.value, &replace49_1.value },
   NULL,
};
   
static const nir_search_variable search50_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search50_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search50_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search50 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_flrp,
   { &search50_0.value, &search50_1.value, &search50_2.value },
   NULL,
};
   
static const nir_search_variable replace50_0_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace50_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace50_0_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace50_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &replace50_0_1_0.value, &replace50_0_1_1.value },
   NULL,
};
static const nir_search_expression replace50_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace50_0_0.value, &replace50_0_1.value },
   NULL,
};

static const nir_search_variable replace50_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace50 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace50_0.value, &replace50_1.value },
   NULL,
};
   
static const nir_search_variable search51_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search51_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search51_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search51 = {
   { nir_search_value_expression, 64 },
   false,
   nir_op_flrp,
   { &search51_0.value, &search51_1.value, &search51_2.value },
   NULL,
};
   
static const nir_search_variable replace51_0_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace51_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace51_0_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace51_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &replace51_0_1_0.value, &replace51_0_1_1.value },
   NULL,
};
static const nir_search_expression replace51_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace51_0_0.value, &replace51_0_1.value },
   NULL,
};

static const nir_search_variable replace51_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace51 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace51_0.value, &replace51_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_flrp_xforms[] = {
   { &search44, &replace44.value, 0 },
   { &search45, &replace45.value, 0 },
   { &search46, &replace46.value, 0 },
   { &search47, &replace47.value, 0 },
   { &search48, &replace48.value, 2 },
   { &search49, &replace49.value, 0 },
   { &search50, &replace50.value, 2 },
   { &search51, &replace51.value, 3 },
};
   
static const nir_search_variable search52_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search52 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ffract,
   { &search52_0.value },
   NULL,
};
   
static const nir_search_variable replace52_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace52_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace52_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ffloor,
   { &replace52_1_0.value },
   NULL,
};
static const nir_search_expression replace52 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &replace52_0.value, &replace52_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ffract_xforms[] = {
   { &search52, &replace52.value, 4 },
};
   
static const nir_search_variable search61_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search61_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search61_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search61_0_3 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression search61_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec4,
   { &search61_0_0.value, &search61_0_1.value, &search61_0_2.value, &search61_0_3.value },
   NULL,
};

static const nir_search_variable search61_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search61 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot4,
   { &search61_0.value, &search61_1.value },
   NULL,
};
   
static const nir_search_variable replace61_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace61_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace61_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace61_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec3,
   { &replace61_0_0.value, &replace61_0_1.value, &replace61_0_2.value },
   NULL,
};

static const nir_search_variable replace61_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace61 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdph,
   { &replace61_0.value, &replace61_1.value },
   NULL,
};
   
static const nir_search_variable search62_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search62_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_constant search62_0_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_constant search62_0_3 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search62_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec4,
   { &search62_0_0.value, &search62_0_1.value, &search62_0_2.value, &search62_0_3.value },
   NULL,
};

static const nir_search_variable search62_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search62 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot4,
   { &search62_0.value, &search62_1.value },
   NULL,
};
   
static const nir_search_variable replace62_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace62_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace62 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace62_0.value, &replace62_1.value },
   NULL,
};
   
static const nir_search_variable search63_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search63_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search63_0_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_constant search63_0_3 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search63_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec4,
   { &search63_0_0.value, &search63_0_1.value, &search63_0_2.value, &search63_0_3.value },
   NULL,
};

static const nir_search_variable search63_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search63 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot4,
   { &search63_0.value, &search63_1.value },
   NULL,
};
   
static const nir_search_variable replace63_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace63_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace63_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec2,
   { &replace63_0_0.value, &replace63_0_1.value },
   NULL,
};

static const nir_search_variable replace63_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace63 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot2,
   { &replace63_0.value, &replace63_1.value },
   NULL,
};
   
static const nir_search_variable search64_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search64_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search64_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search64_0_3 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search64_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec4,
   { &search64_0_0.value, &search64_0_1.value, &search64_0_2.value, &search64_0_3.value },
   NULL,
};

static const nir_search_variable search64_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search64 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot4,
   { &search64_0.value, &search64_1.value },
   NULL,
};
   
static const nir_search_variable replace64_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace64_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace64_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace64_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec3,
   { &replace64_0_0.value, &replace64_0_1.value, &replace64_0_2.value },
   NULL,
};

static const nir_search_variable replace64_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace64 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot3,
   { &replace64_0.value, &replace64_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fdot4_xforms[] = {
   { &search61, &replace61.value, 0 },
   { &search62, &replace62.value, 0 },
   { &search63, &replace63.value, 0 },
   { &search64, &replace64.value, 0 },
};
   
static const nir_search_variable search65_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search65_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_constant search65_0_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search65_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec3,
   { &search65_0_0.value, &search65_0_1.value, &search65_0_2.value },
   NULL,
};

static const nir_search_variable search65_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search65 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot3,
   { &search65_0.value, &search65_1.value },
   NULL,
};
   
static const nir_search_variable replace65_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace65_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace65 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace65_0.value, &replace65_1.value },
   NULL,
};
   
static const nir_search_variable search66_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search66_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search66_0_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search66_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec3,
   { &search66_0_0.value, &search66_0_1.value, &search66_0_2.value },
   NULL,
};

static const nir_search_variable search66_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search66 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot3,
   { &search66_0.value, &search66_1.value },
   NULL,
};
   
static const nir_search_variable replace66_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace66_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace66_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec2,
   { &replace66_0_0.value, &replace66_0_1.value },
   NULL,
};

static const nir_search_variable replace66_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace66 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot2,
   { &replace66_0.value, &replace66_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fdot3_xforms[] = {
   { &search65, &replace65.value, 0 },
   { &search66, &replace66.value, 0 },
};
   
static const nir_search_variable search67_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search67_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search67_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search67_0_0_0.value, &search67_0_0_1.value },
   NULL,
};

static const nir_search_variable search67_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search67_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search67_0_0.value, &search67_0_1.value },
   NULL,
};

static const nir_search_variable search67_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search67 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search67_0.value, &search67_1.value },
   NULL,
};
   
static const nir_search_variable replace67_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace67_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace67_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace67_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace67_0_1_0.value, &replace67_0_1_1.value },
   NULL,
};
static const nir_search_expression replace67_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace67_0_0.value, &replace67_0_1.value },
   NULL,
};

static const nir_search_variable replace67_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace67_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace67_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace67_1_0.value, &replace67_1_1.value },
   NULL,
};
static const nir_search_expression replace67 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace67_0.value, &replace67_1.value },
   NULL,
};
   
static const nir_search_variable search68_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search68_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search68_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search68_0_0.value, &search68_0_1.value },
   NULL,
};

static const nir_search_variable search68_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search68 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search68_0.value, &search68_1.value },
   NULL,
};
   
static const nir_search_variable replace68_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace68_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace68_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace68_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace68_1_0.value, &replace68_1_1.value },
   NULL,
};
static const nir_search_expression replace68 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace68_0.value, &replace68_1.value },
   NULL,
};
   
static const nir_search_constant search305_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable search305_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search305 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search305_0.value, &search305_1.value },
   NULL,
};
   
static const nir_search_constant replace305 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
   
static const nir_search_variable search306_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search306_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search306 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search306_0.value, &search306_1.value },
   NULL,
};
   
static const nir_search_variable replace306 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_ishl_xforms[] = {
   { &search67, &replace67.value, 0 },
   { &search68, &replace68.value, 0 },
   { &search305, &replace305.value, 0 },
   { &search306, &replace306.value, 0 },
};
   
static const nir_search_variable search69_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search69_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search69_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search69_0_0.value, &search69_0_1.value },
   NULL,
};
static const nir_search_expression search69 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_inot,
   { &search69_0.value },
   NULL,
};
   
static const nir_search_variable replace69_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace69_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace69 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace69_0.value, &replace69_1.value },
   NULL,
};
   
static const nir_search_variable search70_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search70_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search70_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search70_0_0.value, &search70_0_1.value },
   NULL,
};
static const nir_search_expression search70 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_inot,
   { &search70_0.value },
   NULL,
};
   
static const nir_search_variable replace70_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace70_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace70 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace70_0.value, &replace70_1.value },
   NULL,
};
   
static const nir_search_variable search71_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search71_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search71_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search71_0_0.value, &search71_0_1.value },
   NULL,
};
static const nir_search_expression search71 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_inot,
   { &search71_0.value },
   NULL,
};
   
static const nir_search_variable replace71_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace71_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace71 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace71_0.value, &replace71_1.value },
   NULL,
};
   
static const nir_search_variable search72_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search72_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search72_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search72_0_0.value, &search72_0_1.value },
   NULL,
};
static const nir_search_expression search72 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_inot,
   { &search72_0.value },
   NULL,
};
   
static const nir_search_variable replace72_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace72_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace72 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace72_0.value, &replace72_1.value },
   NULL,
};
   
static const nir_search_variable search73_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search73_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search73_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search73_0_0.value, &search73_0_1.value },
   NULL,
};
static const nir_search_expression search73 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search73_0.value },
   NULL,
};
   
static const nir_search_variable replace73_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace73_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace73 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace73_0.value, &replace73_1.value },
   NULL,
};
   
static const nir_search_variable search74_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search74_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search74_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search74_0_0.value, &search74_0_1.value },
   NULL,
};
static const nir_search_expression search74 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search74_0.value },
   NULL,
};
   
static const nir_search_variable replace74_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace74_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace74 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace74_0.value, &replace74_1.value },
   NULL,
};
   
static const nir_search_variable search75_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search75_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search75_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search75_0_0.value, &search75_0_1.value },
   NULL,
};
static const nir_search_expression search75 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search75_0.value },
   NULL,
};
   
static const nir_search_variable replace75_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace75_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace75 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace75_0.value, &replace75_1.value },
   NULL,
};
   
static const nir_search_variable search76_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search76_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search76_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search76_0_0.value, &search76_0_1.value },
   NULL,
};
static const nir_search_expression search76 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search76_0.value },
   NULL,
};
   
static const nir_search_variable replace76_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace76_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace76 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace76_0.value, &replace76_1.value },
   NULL,
};
   
static const nir_search_variable search77_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search77_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search77_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search77_0_0.value, &search77_0_1.value },
   NULL,
};
static const nir_search_expression search77 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search77_0.value },
   NULL,
};
   
static const nir_search_variable replace77_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace77_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace77 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &replace77_0.value, &replace77_1.value },
   NULL,
};
   
static const nir_search_variable search78_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search78_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search78_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search78_0_0.value, &search78_0_1.value },
   NULL,
};
static const nir_search_expression search78 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search78_0.value },
   NULL,
};
   
static const nir_search_variable replace78_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace78_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace78 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace78_0.value, &replace78_1.value },
   NULL,
};
   
static const nir_search_variable search298_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search298_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search298_0_0.value },
   NULL,
};
static const nir_search_expression search298 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search298_0.value },
   NULL,
};
   
static const nir_search_variable replace298 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search364_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search364_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2b,
   { &search364_0_0.value },
   NULL,
};
static const nir_search_expression search364 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search364_0.value },
   NULL,
};
   
static const nir_search_variable replace364_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace364_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace364 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace364_0.value, &replace364_1.value },
   NULL,
};
   
static const nir_search_variable search444_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search444_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search444_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search444_0_0_0.value, &search444_0_0_1.value },
   NULL,
};

static const nir_search_variable search444_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search444_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search444_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search444_0_1_0.value, &search444_0_1_1.value },
   NULL,
};
static const nir_search_expression search444_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search444_0_0.value, &search444_0_1.value },
   (is_used_once),
};
static const nir_search_expression search444 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search444_0.value },
   NULL,
};
   
static const nir_search_variable replace444_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace444_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace444_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace444_0_0.value, &replace444_0_1.value },
   NULL,
};

static const nir_search_variable replace444_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace444_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace444_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace444_1_0.value, &replace444_1_1.value },
   NULL,
};
static const nir_search_expression replace444 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace444_0.value, &replace444_1.value },
   NULL,
};
   
static const nir_search_variable search445_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search445_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search445_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search445_0_0_0.value, &search445_0_0_1.value },
   NULL,
};

static const nir_search_variable search445_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search445_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search445_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search445_0_1_0.value, &search445_0_1_1.value },
   NULL,
};
static const nir_search_expression search445_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search445_0_0.value, &search445_0_1.value },
   (is_used_once),
};
static const nir_search_expression search445 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search445_0.value },
   NULL,
};
   
static const nir_search_variable replace445_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace445_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace445_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace445_0_0.value, &replace445_0_1.value },
   NULL,
};

static const nir_search_variable replace445_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace445_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace445_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace445_1_0.value, &replace445_1_1.value },
   NULL,
};
static const nir_search_expression replace445 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace445_0.value, &replace445_1.value },
   NULL,
};
   
static const nir_search_variable search446_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search446_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search446_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search446_0_0_0.value, &search446_0_0_1.value },
   NULL,
};

static const nir_search_variable search446_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search446_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search446_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search446_0_1_0.value, &search446_0_1_1.value },
   NULL,
};
static const nir_search_expression search446_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search446_0_0.value, &search446_0_1.value },
   (is_used_once),
};
static const nir_search_expression search446 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search446_0.value },
   NULL,
};
   
static const nir_search_variable replace446_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace446_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace446_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace446_0_0.value, &replace446_0_1.value },
   NULL,
};

static const nir_search_variable replace446_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace446_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace446_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace446_1_0.value, &replace446_1_1.value },
   NULL,
};
static const nir_search_expression replace446 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace446_0.value, &replace446_1.value },
   NULL,
};
   
static const nir_search_variable search447_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search447_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search447_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search447_0_0_0.value, &search447_0_0_1.value },
   NULL,
};

static const nir_search_variable search447_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search447_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search447_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search447_0_1_0.value, &search447_0_1_1.value },
   NULL,
};
static const nir_search_expression search447_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search447_0_0.value, &search447_0_1.value },
   (is_used_once),
};
static const nir_search_expression search447 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search447_0.value },
   NULL,
};
   
static const nir_search_variable replace447_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace447_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace447_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace447_0_0.value, &replace447_0_1.value },
   NULL,
};

static const nir_search_variable replace447_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace447_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace447_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace447_1_0.value, &replace447_1_1.value },
   NULL,
};
static const nir_search_expression replace447 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace447_0.value, &replace447_1.value },
   NULL,
};
   
static const nir_search_variable search448_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search448_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search448_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search448_0_0_0.value, &search448_0_0_1.value },
   NULL,
};

static const nir_search_variable search448_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search448_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search448_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search448_0_1_0.value, &search448_0_1_1.value },
   NULL,
};
static const nir_search_expression search448_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search448_0_0.value, &search448_0_1.value },
   (is_used_once),
};
static const nir_search_expression search448 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search448_0.value },
   NULL,
};
   
static const nir_search_variable replace448_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace448_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace448_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace448_0_0.value, &replace448_0_1.value },
   NULL,
};

static const nir_search_variable replace448_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace448_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace448_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace448_1_0.value, &replace448_1_1.value },
   NULL,
};
static const nir_search_expression replace448 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace448_0.value, &replace448_1.value },
   NULL,
};
   
static const nir_search_variable search449_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search449_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search449_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search449_0_0_0.value, &search449_0_0_1.value },
   NULL,
};

static const nir_search_variable search449_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search449_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search449_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search449_0_1_0.value, &search449_0_1_1.value },
   NULL,
};
static const nir_search_expression search449_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search449_0_0.value, &search449_0_1.value },
   (is_used_once),
};
static const nir_search_expression search449 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search449_0.value },
   NULL,
};
   
static const nir_search_variable replace449_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace449_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace449_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace449_0_0.value, &replace449_0_1.value },
   NULL,
};

static const nir_search_variable replace449_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace449_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace449_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace449_1_0.value, &replace449_1_1.value },
   NULL,
};
static const nir_search_expression replace449 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace449_0.value, &replace449_1.value },
   NULL,
};
   
static const nir_search_variable search450_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search450_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search450_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search450_0_0_0.value, &search450_0_0_1.value },
   NULL,
};

static const nir_search_variable search450_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search450_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search450_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search450_0_1_0.value, &search450_0_1_1.value },
   NULL,
};
static const nir_search_expression search450_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search450_0_0.value, &search450_0_1.value },
   (is_used_once),
};
static const nir_search_expression search450 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search450_0.value },
   NULL,
};
   
static const nir_search_variable replace450_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace450_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace450_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace450_0_0.value, &replace450_0_1.value },
   NULL,
};

static const nir_search_variable replace450_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace450_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace450_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace450_1_0.value, &replace450_1_1.value },
   NULL,
};
static const nir_search_expression replace450 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace450_0.value, &replace450_1.value },
   NULL,
};
   
static const nir_search_variable search451_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search451_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search451_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search451_0_0_0.value, &search451_0_0_1.value },
   NULL,
};

static const nir_search_variable search451_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search451_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search451_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search451_0_1_0.value, &search451_0_1_1.value },
   NULL,
};
static const nir_search_expression search451_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search451_0_0.value, &search451_0_1.value },
   (is_used_once),
};
static const nir_search_expression search451 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search451_0.value },
   NULL,
};
   
static const nir_search_variable replace451_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace451_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace451_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace451_0_0.value, &replace451_0_1.value },
   NULL,
};

static const nir_search_variable replace451_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace451_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace451_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace451_1_0.value, &replace451_1_1.value },
   NULL,
};
static const nir_search_expression replace451 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace451_0.value, &replace451_1.value },
   NULL,
};
   
static const nir_search_variable search452_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search452_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search452_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search452_0_0_0.value, &search452_0_0_1.value },
   NULL,
};

static const nir_search_variable search452_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search452_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search452_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search452_0_1_0.value, &search452_0_1_1.value },
   NULL,
};
static const nir_search_expression search452_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search452_0_0.value, &search452_0_1.value },
   (is_used_once),
};
static const nir_search_expression search452 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search452_0.value },
   NULL,
};
   
static const nir_search_variable replace452_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace452_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace452_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace452_0_0.value, &replace452_0_1.value },
   NULL,
};

static const nir_search_variable replace452_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace452_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace452_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace452_1_0.value, &replace452_1_1.value },
   NULL,
};
static const nir_search_expression replace452 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace452_0.value, &replace452_1.value },
   NULL,
};
   
static const nir_search_variable search453_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search453_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search453_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search453_0_0_0.value, &search453_0_0_1.value },
   NULL,
};

static const nir_search_variable search453_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search453_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search453_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search453_0_1_0.value, &search453_0_1_1.value },
   NULL,
};
static const nir_search_expression search453_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search453_0_0.value, &search453_0_1.value },
   (is_used_once),
};
static const nir_search_expression search453 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search453_0.value },
   NULL,
};
   
static const nir_search_variable replace453_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace453_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace453_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace453_0_0.value, &replace453_0_1.value },
   NULL,
};

static const nir_search_variable replace453_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace453_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace453_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace453_1_0.value, &replace453_1_1.value },
   NULL,
};
static const nir_search_expression replace453 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace453_0.value, &replace453_1.value },
   NULL,
};
   
static const nir_search_variable search454_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search454_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search454_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search454_0_0_0.value, &search454_0_0_1.value },
   NULL,
};

static const nir_search_variable search454_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search454_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search454_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search454_0_1_0.value, &search454_0_1_1.value },
   NULL,
};
static const nir_search_expression search454_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search454_0_0.value, &search454_0_1.value },
   (is_used_once),
};
static const nir_search_expression search454 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search454_0.value },
   NULL,
};
   
static const nir_search_variable replace454_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace454_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace454_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace454_0_0.value, &replace454_0_1.value },
   NULL,
};

static const nir_search_variable replace454_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace454_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace454_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace454_1_0.value, &replace454_1_1.value },
   NULL,
};
static const nir_search_expression replace454 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace454_0.value, &replace454_1.value },
   NULL,
};
   
static const nir_search_variable search455_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search455_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search455_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search455_0_0_0.value, &search455_0_0_1.value },
   NULL,
};

static const nir_search_variable search455_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search455_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search455_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search455_0_1_0.value, &search455_0_1_1.value },
   NULL,
};
static const nir_search_expression search455_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search455_0_0.value, &search455_0_1.value },
   (is_used_once),
};
static const nir_search_expression search455 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search455_0.value },
   NULL,
};
   
static const nir_search_variable replace455_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace455_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace455_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace455_0_0.value, &replace455_0_1.value },
   NULL,
};

static const nir_search_variable replace455_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace455_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace455_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace455_1_0.value, &replace455_1_1.value },
   NULL,
};
static const nir_search_expression replace455 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace455_0.value, &replace455_1.value },
   NULL,
};
   
static const nir_search_variable search456_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search456_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search456_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search456_0_0_0.value, &search456_0_0_1.value },
   NULL,
};

static const nir_search_variable search456_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search456_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search456_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search456_0_1_0.value, &search456_0_1_1.value },
   NULL,
};
static const nir_search_expression search456_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search456_0_0.value, &search456_0_1.value },
   (is_used_once),
};
static const nir_search_expression search456 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search456_0.value },
   NULL,
};
   
static const nir_search_variable replace456_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace456_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace456_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace456_0_0.value, &replace456_0_1.value },
   NULL,
};

static const nir_search_variable replace456_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace456_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace456_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace456_1_0.value, &replace456_1_1.value },
   NULL,
};
static const nir_search_expression replace456 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace456_0.value, &replace456_1.value },
   NULL,
};
   
static const nir_search_variable search457_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search457_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search457_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search457_0_0_0.value, &search457_0_0_1.value },
   NULL,
};

static const nir_search_variable search457_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search457_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search457_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search457_0_1_0.value, &search457_0_1_1.value },
   NULL,
};
static const nir_search_expression search457_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search457_0_0.value, &search457_0_1.value },
   (is_used_once),
};
static const nir_search_expression search457 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search457_0.value },
   NULL,
};
   
static const nir_search_variable replace457_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace457_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace457_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace457_0_0.value, &replace457_0_1.value },
   NULL,
};

static const nir_search_variable replace457_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace457_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace457_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace457_1_0.value, &replace457_1_1.value },
   NULL,
};
static const nir_search_expression replace457 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace457_0.value, &replace457_1.value },
   NULL,
};
   
static const nir_search_variable search458_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search458_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search458_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search458_0_0_0.value, &search458_0_0_1.value },
   NULL,
};

static const nir_search_variable search458_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search458_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search458_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search458_0_1_0.value, &search458_0_1_1.value },
   NULL,
};
static const nir_search_expression search458_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search458_0_0.value, &search458_0_1.value },
   (is_used_once),
};
static const nir_search_expression search458 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search458_0.value },
   NULL,
};
   
static const nir_search_variable replace458_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace458_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace458_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace458_0_0.value, &replace458_0_1.value },
   NULL,
};

static const nir_search_variable replace458_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace458_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace458_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace458_1_0.value, &replace458_1_1.value },
   NULL,
};
static const nir_search_expression replace458 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace458_0.value, &replace458_1.value },
   NULL,
};
   
static const nir_search_variable search459_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search459_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search459_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search459_0_0_0.value, &search459_0_0_1.value },
   NULL,
};

static const nir_search_variable search459_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search459_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search459_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search459_0_1_0.value, &search459_0_1_1.value },
   NULL,
};
static const nir_search_expression search459_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search459_0_0.value, &search459_0_1.value },
   (is_used_once),
};
static const nir_search_expression search459 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search459_0.value },
   NULL,
};
   
static const nir_search_variable replace459_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace459_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace459_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace459_0_0.value, &replace459_0_1.value },
   NULL,
};

static const nir_search_variable replace459_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace459_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace459_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace459_1_0.value, &replace459_1_1.value },
   NULL,
};
static const nir_search_expression replace459 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace459_0.value, &replace459_1.value },
   NULL,
};
   
static const nir_search_variable search460_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search460_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search460_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search460_0_0_0.value, &search460_0_0_1.value },
   NULL,
};

static const nir_search_variable search460_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search460_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search460_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search460_0_1_0.value, &search460_0_1_1.value },
   NULL,
};
static const nir_search_expression search460_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search460_0_0.value, &search460_0_1.value },
   (is_used_once),
};
static const nir_search_expression search460 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search460_0.value },
   NULL,
};
   
static const nir_search_variable replace460_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace460_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace460_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace460_0_0.value, &replace460_0_1.value },
   NULL,
};

static const nir_search_variable replace460_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace460_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace460_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace460_1_0.value, &replace460_1_1.value },
   NULL,
};
static const nir_search_expression replace460 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace460_0.value, &replace460_1.value },
   NULL,
};
   
static const nir_search_variable search461_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search461_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search461_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search461_0_0_0.value, &search461_0_0_1.value },
   NULL,
};

static const nir_search_variable search461_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search461_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search461_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search461_0_1_0.value, &search461_0_1_1.value },
   NULL,
};
static const nir_search_expression search461_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search461_0_0.value, &search461_0_1.value },
   (is_used_once),
};
static const nir_search_expression search461 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search461_0.value },
   NULL,
};
   
static const nir_search_variable replace461_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace461_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace461_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace461_0_0.value, &replace461_0_1.value },
   NULL,
};

static const nir_search_variable replace461_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace461_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace461_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace461_1_0.value, &replace461_1_1.value },
   NULL,
};
static const nir_search_expression replace461 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace461_0.value, &replace461_1.value },
   NULL,
};
   
static const nir_search_variable search462_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search462_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search462_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search462_0_0_0.value, &search462_0_0_1.value },
   NULL,
};

static const nir_search_variable search462_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search462_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search462_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search462_0_1_0.value, &search462_0_1_1.value },
   NULL,
};
static const nir_search_expression search462_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search462_0_0.value, &search462_0_1.value },
   (is_used_once),
};
static const nir_search_expression search462 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search462_0.value },
   NULL,
};
   
static const nir_search_variable replace462_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace462_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace462_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace462_0_0.value, &replace462_0_1.value },
   NULL,
};

static const nir_search_variable replace462_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace462_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace462_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace462_1_0.value, &replace462_1_1.value },
   NULL,
};
static const nir_search_expression replace462 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace462_0.value, &replace462_1.value },
   NULL,
};
   
static const nir_search_variable search463_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search463_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search463_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search463_0_0_0.value, &search463_0_0_1.value },
   NULL,
};

static const nir_search_variable search463_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search463_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search463_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search463_0_1_0.value, &search463_0_1_1.value },
   NULL,
};
static const nir_search_expression search463_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search463_0_0.value, &search463_0_1.value },
   (is_used_once),
};
static const nir_search_expression search463 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search463_0.value },
   NULL,
};
   
static const nir_search_variable replace463_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace463_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace463_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace463_0_0.value, &replace463_0_1.value },
   NULL,
};

static const nir_search_variable replace463_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace463_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace463_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace463_1_0.value, &replace463_1_1.value },
   NULL,
};
static const nir_search_expression replace463 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace463_0.value, &replace463_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_inot_xforms[] = {
   { &search69, &replace69.value, 0 },
   { &search70, &replace70.value, 0 },
   { &search71, &replace71.value, 0 },
   { &search72, &replace72.value, 0 },
   { &search73, &replace73.value, 0 },
   { &search74, &replace74.value, 0 },
   { &search75, &replace75.value, 0 },
   { &search76, &replace76.value, 0 },
   { &search77, &replace77.value, 0 },
   { &search78, &replace78.value, 0 },
   { &search298, &replace298.value, 0 },
   { &search364, &replace364.value, 0 },
   { &search444, &replace444.value, 0 },
   { &search445, &replace445.value, 0 },
   { &search446, &replace446.value, 0 },
   { &search447, &replace447.value, 0 },
   { &search448, &replace448.value, 0 },
   { &search449, &replace449.value, 0 },
   { &search450, &replace450.value, 0 },
   { &search451, &replace451.value, 0 },
   { &search452, &replace452.value, 0 },
   { &search453, &replace453.value, 0 },
   { &search454, &replace454.value, 0 },
   { &search455, &replace455.value, 0 },
   { &search456, &replace456.value, 0 },
   { &search457, &replace457.value, 0 },
   { &search458, &replace458.value, 0 },
   { &search459, &replace459.value, 0 },
   { &search460, &replace460.value, 0 },
   { &search461, &replace461.value, 0 },
   { &search462, &replace462.value, 0 },
   { &search463, &replace463.value, 0 },
};
   
static const nir_search_constant search79_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search79_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search79_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search79_1_0.value },
   NULL,
};
static const nir_search_expression search79 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search79_0.value, &search79_1.value },
   NULL,
};
   
static const nir_search_variable replace79_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace79 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace79_0.value },
   NULL,
};
   
static const nir_search_variable search80_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search80_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search80_0_0_0.value },
   NULL,
};
static const nir_search_expression search80_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search80_0_0.value },
   NULL,
};

static const nir_search_constant search80_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search80 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search80_0.value, &search80_1.value },
   NULL,
};
   
static const nir_search_variable replace80_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace80 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace80_0.value },
   NULL,
};
   
static const nir_search_variable search103_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search103_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search103_0_0_0_0.value },
   NULL,
};

static const nir_search_variable search103_0_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search103_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search103_0_0_1_0.value },
   NULL,
};
static const nir_search_expression search103_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search103_0_0_0.value, &search103_0_0_1.value },
   NULL,
};
static const nir_search_expression search103_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search103_0_0.value },
   NULL,
};

static const nir_search_constant search103_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search103 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search103_0.value, &search103_1.value },
   NULL,
};
   
static const nir_search_variable replace103_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace103_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace103_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace103_0_0.value, &replace103_0_1.value },
   NULL,
};
static const nir_search_expression replace103 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace103_0.value },
   NULL,
};
   
static const nir_search_constant search104_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search104_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search104_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search104_1_0_0.value },
   NULL,
};

static const nir_search_variable search104_1_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search104_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search104_1_1_0.value },
   NULL,
};
static const nir_search_expression search104_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search104_1_0.value, &search104_1_1.value },
   NULL,
};
static const nir_search_expression search104 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search104_0.value, &search104_1.value },
   NULL,
};
   
static const nir_search_variable replace104_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace104_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace104_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace104_0_0.value, &replace104_0_1.value },
   NULL,
};
static const nir_search_expression replace104 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace104_0.value },
   NULL,
};
   
static const nir_search_variable search107_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search107_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search107_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search107_0_0.value, &search107_0_1.value },
   NULL,
};

static const nir_search_variable search107_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search107 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fge,
   { &search107_0.value, &search107_1.value },
   NULL,
};
   
static const nir_search_variable replace107_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace107_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace107 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace107_0.value, &replace107_1.value },
   NULL,
};
   
static const nir_search_variable search112_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search112_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search112_0_0_0_0.value },
   NULL,
};
static const nir_search_expression search112_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search112_0_0_0.value },
   NULL,
};

static const nir_search_variable search112_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search112_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search112_0_0.value, &search112_0_1.value },
   NULL,
};

static const nir_search_constant search112_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search112 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search112_0.value, &search112_1.value },
   NULL,
};
   
static const nir_search_variable replace112_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace112_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace112_0_0.value },
   NULL,
};

static const nir_search_variable replace112_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace112_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace112_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace112_1_0.value, &replace112_1_1.value },
   NULL,
};
static const nir_search_expression replace112 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace112_0.value, &replace112_1.value },
   NULL,
};
   
static const nir_search_variable search120_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search120_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &search120_0_0.value },
   NULL,
};

static const nir_search_constant search120_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search120 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search120_0.value, &search120_1.value },
   NULL,
};
   
static const nir_search_constant replace120 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_constant search121_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search121_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search121_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &search121_1_0.value },
   NULL,
};
static const nir_search_expression search121 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search121_0.value, &search121_1.value },
   NULL,
};
   
static const nir_search_constant replace121_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable replace121_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace121 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace121_0.value, &replace121_1.value },
   NULL,
};
   
static const nir_search_variable search126_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search126_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &search126_0_0.value },
   NULL,
};

static const nir_search_constant search126_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search126 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search126_0.value, &search126_1.value },
   NULL,
};
   
static const nir_search_variable replace126_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace126_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace126 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace126_0.value, &replace126_1.value },
   NULL,
};
   
static const nir_search_constant search127_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search127_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search127_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &search127_1_0.value },
   NULL,
};
static const nir_search_expression search127 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search127_0.value, &search127_1.value },
   NULL,
};
   
static const nir_search_constant replace127_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable replace127_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace127 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace127_0.value, &replace127_1.value },
   NULL,
};
   
static const nir_search_constant search132_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search132_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search132_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search132_1_0.value },
   NULL,
};
static const nir_search_expression search132 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search132_0.value, &search132_1.value },
   NULL,
};
   
static const nir_search_variable replace132_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace132_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace132 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace132_0.value, &replace132_1.value },
   NULL,
};
   
static const nir_search_variable search133_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search133_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search133_0_0_0.value },
   NULL,
};
static const nir_search_expression search133_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search133_0_0.value },
   NULL,
};

static const nir_search_constant search133_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search133 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search133_0.value, &search133_1.value },
   NULL,
};
   
static const nir_search_variable replace133_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace133_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace133 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace133_0.value, &replace133_1.value },
   NULL,
};
   
static const nir_search_variable search140_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search140_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search140_0_0_0.value },
   NULL,
};
static const nir_search_expression search140_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search140_0_0.value },
   NULL,
};

static const nir_search_constant search140_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search140 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search140_0.value, &search140_1.value },
   NULL,
};
   
static const nir_search_variable replace140_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace140_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace140 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace140_0.value, &replace140_1.value },
   NULL,
};
   
static const nir_search_variable search227_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search227_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search227_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search227_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search227_1_0.value, &search227_1_1.value },
   NULL,
};
static const nir_search_expression search227 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fge,
   { &search227_0.value, &search227_1.value },
   NULL,
};
   
static const nir_search_constant replace227 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search228_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search228_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search228_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search228_0_0.value, &search228_0_1.value },
   NULL,
};

static const nir_search_variable search228_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search228 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fge,
   { &search228_0.value, &search228_1.value },
   NULL,
};
   
static const nir_search_constant replace228 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search231_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search231_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search231_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search231_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search231_1_0.value, &search231_1_1.value },
   NULL,
};
static const nir_search_expression search231 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fge,
   { &search231_0.value, &search231_1.value },
   NULL,
};
   
static const nir_search_variable replace231_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace231_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace231 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace231_0.value, &replace231_1.value },
   NULL,
};
   
static const nir_search_variable search232_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search232_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search232_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search232_0_0.value, &search232_0_1.value },
   NULL,
};

static const nir_search_variable search232_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search232 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fge,
   { &search232_0.value, &search232_1.value },
   NULL,
};
   
static const nir_search_variable replace232_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace232_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace232 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace232_0.value, &replace232_1.value },
   NULL,
};
   
static const nir_search_variable search502_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search502_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search502_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search502_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search502_0_0.value, &search502_0_1.value, &search502_0_2.value },
   NULL,
};

static const nir_search_variable search502_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search502 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search502_0.value, &search502_1.value },
   NULL,
};
   
static const nir_search_variable replace502_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace502_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace502_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace502_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace502_1_0.value, &replace502_1_1.value },
   NULL,
};

static const nir_search_variable replace502_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace502_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace502_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace502_2_0.value, &replace502_2_1.value },
   NULL,
};
static const nir_search_expression replace502 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace502_0.value, &replace502_1.value, &replace502_2.value },
   NULL,
};
   
static const nir_search_variable search503_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search503_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search503_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search503_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search503_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search503_1_0.value, &search503_1_1.value, &search503_1_2.value },
   NULL,
};
static const nir_search_expression search503 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search503_0.value, &search503_1.value },
   NULL,
};
   
static const nir_search_variable replace503_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace503_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace503_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace503_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace503_1_0.value, &replace503_1_1.value },
   NULL,
};

static const nir_search_variable replace503_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace503_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace503_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace503_2_0.value, &replace503_2_1.value },
   NULL,
};
static const nir_search_expression replace503 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace503_0.value, &replace503_1.value, &replace503_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fge_xforms[] = {
   { &search79, &replace79.value, 0 },
   { &search80, &replace80.value, 0 },
   { &search103, &replace103.value, 0 },
   { &search104, &replace104.value, 0 },
   { &search107, &replace107.value, 0 },
   { &search112, &replace112.value, 0 },
   { &search120, &replace120.value, 0 },
   { &search121, &replace121.value, 0 },
   { &search126, &replace126.value, 0 },
   { &search127, &replace127.value, 0 },
   { &search132, &replace132.value, 0 },
   { &search133, &replace133.value, 0 },
   { &search140, &replace140.value, 0 },
   { &search227, &replace227.value, 0 },
   { &search228, &replace228.value, 0 },
   { &search231, &replace231.value, 0 },
   { &search232, &replace232.value, 0 },
   { &search502, &replace502.value, 0 },
   { &search503, &replace503.value, 0 },
};
   
static const nir_search_variable search81_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search81_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search81_0_0_0.value },
   NULL,
};

static const nir_search_variable search81_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search81_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search81_0_1_0.value },
   NULL,
};
static const nir_search_expression search81_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search81_0_0.value, &search81_0_1.value },
   NULL,
};

static const nir_search_constant search81_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search81 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search81_0.value, &search81_1.value },
   NULL,
};
   
static const nir_search_variable replace81_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace81_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace81 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace81_0.value, &replace81_1.value },
   NULL,
};
   
static const nir_search_variable search82_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search82_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search82_0_0_0.value },
   NULL,
};

static const nir_search_variable search82_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search82_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search82_0_1_0.value },
   NULL,
};
static const nir_search_expression search82_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search82_0_0.value, &search82_0_1.value },
   NULL,
};

static const nir_search_constant search82_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search82 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search82_0.value, &search82_1.value },
   NULL,
};
   
static const nir_search_variable replace82_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace82_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace82 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace82_0.value, &replace82_1.value },
   NULL,
};
   
static const nir_search_variable search83_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search83_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_variable search83_0_2_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search83_0_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search83_0_2_0.value },
   NULL,
};
static const nir_search_expression search83_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search83_0_0.value, &search83_0_1.value, &search83_0_2.value },
   NULL,
};

static const nir_search_constant search83_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search83 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search83_0.value, &search83_1.value },
   NULL,
};
   
static const nir_search_variable replace83_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace83_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace83 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace83_0.value, &replace83_1.value },
   NULL,
};
   
static const nir_search_variable search84_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search84_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search84_0_0.value },
   NULL,
};

static const nir_search_variable search84_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search84_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search84_1_0_0.value },
   NULL,
};
static const nir_search_expression search84_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search84_1_0.value },
   NULL,
};
static const nir_search_expression search84 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search84_0.value, &search84_1.value },
   NULL,
};
   
static const nir_search_variable replace84_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace84_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace84 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace84_0.value, &replace84_1.value },
   NULL,
};
   
static const nir_search_variable search85_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search85_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search85_0_0_0.value },
   NULL,
};

static const nir_search_variable search85_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search85_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search85_0_1_0.value },
   NULL,
};
static const nir_search_expression search85_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search85_0_0.value, &search85_0_1.value },
   NULL,
};

static const nir_search_constant search85_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search85 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search85_0.value, &search85_1.value },
   NULL,
};
   
static const nir_search_variable replace85_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace85_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace85 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace85_0.value, &replace85_1.value },
   NULL,
};
   
static const nir_search_variable search86_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search86_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search86_0_0_0.value },
   NULL,
};

static const nir_search_variable search86_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search86_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search86_0_1_0.value },
   NULL,
};
static const nir_search_expression search86_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search86_0_0.value, &search86_0_1.value },
   NULL,
};

static const nir_search_constant search86_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search86 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search86_0.value, &search86_1.value },
   NULL,
};
   
static const nir_search_variable replace86_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace86_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace86 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace86_0.value, &replace86_1.value },
   NULL,
};
   
static const nir_search_variable search87_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search87_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search87_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search87_0_1_0.value },
   NULL,
};

static const nir_search_constant search87_0_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search87_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search87_0_0.value, &search87_0_1.value, &search87_0_2.value },
   NULL,
};

static const nir_search_constant search87_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search87 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search87_0.value, &search87_1.value },
   NULL,
};
   
static const nir_search_variable replace87_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace87_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace87 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace87_0.value, &replace87_1.value },
   NULL,
};
   
static const nir_search_variable search88_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search88_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search88_0_0_0.value },
   NULL,
};

static const nir_search_variable search88_0_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search88_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search88_0_1_0_0.value },
   NULL,
};
static const nir_search_expression search88_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search88_0_1_0.value },
   NULL,
};
static const nir_search_expression search88_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search88_0_0.value, &search88_0_1.value },
   NULL,
};

static const nir_search_constant search88_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search88 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search88_0.value, &search88_1.value },
   NULL,
};
   
static const nir_search_variable replace88_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace88_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace88 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ixor,
   { &replace88_0.value, &replace88_1.value },
   NULL,
};
   
static const nir_search_variable search89_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search89_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search89_0_0.value },
   NULL,
};

static const nir_search_variable search89_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search89_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search89_1_0.value },
   NULL,
};
static const nir_search_expression search89 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search89_0.value, &search89_1.value },
   NULL,
};
   
static const nir_search_variable replace89_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace89_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace89 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ixor,
   { &replace89_0.value, &replace89_1.value },
   NULL,
};
   
static const nir_search_variable search90_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search90_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search90_0_0_0.value },
   NULL,
};
static const nir_search_expression search90_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search90_0_0.value },
   NULL,
};

static const nir_search_variable search90_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search90_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search90_1_0_0.value },
   NULL,
};
static const nir_search_expression search90_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search90_1_0.value },
   NULL,
};
static const nir_search_expression search90 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search90_0.value, &search90_1.value },
   NULL,
};
   
static const nir_search_variable replace90_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace90_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace90 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ixor,
   { &replace90_0.value, &replace90_1.value },
   NULL,
};
   
static const nir_search_variable search109_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search109_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search109_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search109_0_0.value, &search109_0_1.value },
   NULL,
};

static const nir_search_variable search109_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search109 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fne,
   { &search109_0.value, &search109_1.value },
   NULL,
};
   
static const nir_search_variable replace109_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace109_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace109 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace109_0.value, &replace109_1.value },
   NULL,
};
   
static const nir_search_variable search115_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search115_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search115_0_0.value },
   NULL,
};

static const nir_search_constant search115_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search115 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search115_0.value, &search115_1.value },
   NULL,
};
   
static const nir_search_variable replace115 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search118_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search118_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &search118_0_0.value },
   NULL,
};

static const nir_search_constant search118_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search118 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search118_0.value, &search118_1.value },
   NULL,
};
   
static const nir_search_variable replace118_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace118_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace118 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &replace118_0.value, &replace118_1.value },
   NULL,
};
   
static const nir_search_variable search124_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search124_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &search124_0_0.value },
   NULL,
};

static const nir_search_constant search124_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search124 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search124_0.value, &search124_1.value },
   NULL,
};
   
static const nir_search_variable replace124_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace124_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace124 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &replace124_0.value, &replace124_1.value },
   NULL,
};
   
static const nir_search_variable search273_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search273_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search273_0_0.value },
   NULL,
};

static const nir_search_variable search273_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search273 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search273_0.value, &search273_1.value },
   NULL,
};
   
static const nir_search_variable replace273_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace273_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace273 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace273_0.value, &replace273_1.value },
   NULL,
};
   
static const nir_search_variable search506_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search506_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search506_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search506_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search506_0_0.value, &search506_0_1.value, &search506_0_2.value },
   NULL,
};

static const nir_search_variable search506_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search506 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search506_0.value, &search506_1.value },
   NULL,
};
   
static const nir_search_variable replace506_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace506_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace506_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace506_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace506_1_0.value, &replace506_1_1.value },
   NULL,
};

static const nir_search_variable replace506_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace506_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace506_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace506_2_0.value, &replace506_2_1.value },
   NULL,
};
static const nir_search_expression replace506 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace506_0.value, &replace506_1.value, &replace506_2.value },
   NULL,
};
   
static const nir_search_variable search507_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search507_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search507_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search507_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search507_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search507_1_0.value, &search507_1_1.value, &search507_1_2.value },
   NULL,
};
static const nir_search_expression search507 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &search507_0.value, &search507_1.value },
   NULL,
};
   
static const nir_search_variable replace507_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace507_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace507_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace507_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace507_1_0.value, &replace507_1_1.value },
   NULL,
};

static const nir_search_variable replace507_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace507_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace507_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace507_2_0.value, &replace507_2_1.value },
   NULL,
};
static const nir_search_expression replace507 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace507_0.value, &replace507_1.value, &replace507_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fne_xforms[] = {
   { &search81, &replace81.value, 0 },
   { &search82, &replace82.value, 0 },
   { &search83, &replace83.value, 0 },
   { &search84, &replace84.value, 0 },
   { &search85, &replace85.value, 0 },
   { &search86, &replace86.value, 0 },
   { &search87, &replace87.value, 0 },
   { &search88, &replace88.value, 0 },
   { &search89, &replace89.value, 0 },
   { &search90, &replace90.value, 0 },
   { &search109, &replace109.value, 0 },
   { &search115, &replace115.value, 0 },
   { &search118, &replace118.value, 0 },
   { &search124, &replace124.value, 0 },
   { &search273, &replace273.value, 0 },
   { &search506, &replace506.value, 0 },
   { &search507, &replace507.value, 0 },
};
   
static const nir_search_variable search91_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search91_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search91_0_0_0.value },
   NULL,
};

static const nir_search_variable search91_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search91_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search91_0_1_0.value },
   NULL,
};
static const nir_search_expression search91_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search91_0_0.value, &search91_0_1.value },
   NULL,
};

static const nir_search_constant search91_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search91 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search91_0.value, &search91_1.value },
   NULL,
};
   
static const nir_search_variable replace91_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace91_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace91_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace91_0_0.value, &replace91_0_1.value },
   NULL,
};
static const nir_search_expression replace91 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace91_0.value },
   NULL,
};
   
static const nir_search_variable search92_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search92_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search92_0_0_0.value },
   NULL,
};

static const nir_search_variable search92_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search92_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search92_0_1_0.value },
   NULL,
};
static const nir_search_expression search92_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search92_0_0.value, &search92_0_1.value },
   NULL,
};

static const nir_search_constant search92_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search92 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search92_0.value, &search92_1.value },
   NULL,
};
   
static const nir_search_variable replace92_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace92_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace92_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace92_0_0.value, &replace92_0_1.value },
   NULL,
};
static const nir_search_expression replace92 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace92_0.value },
   NULL,
};
   
static const nir_search_variable search93_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search93_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_variable search93_0_2_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search93_0_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search93_0_2_0.value },
   NULL,
};
static const nir_search_expression search93_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search93_0_0.value, &search93_0_1.value, &search93_0_2.value },
   NULL,
};

static const nir_search_constant search93_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search93 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search93_0.value, &search93_1.value },
   NULL,
};
   
static const nir_search_variable replace93_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace93_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace93_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace93_0_0.value, &replace93_0_1.value },
   NULL,
};
static const nir_search_expression replace93 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace93_0.value },
   NULL,
};
   
static const nir_search_variable search94_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search94_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search94_0_0.value },
   NULL,
};

static const nir_search_variable search94_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search94_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search94_1_0_0.value },
   NULL,
};
static const nir_search_expression search94_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search94_1_0.value },
   NULL,
};
static const nir_search_expression search94 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search94_0.value, &search94_1.value },
   NULL,
};
   
static const nir_search_variable replace94_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace94_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace94_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace94_0_0.value, &replace94_0_1.value },
   NULL,
};
static const nir_search_expression replace94 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace94_0.value },
   NULL,
};
   
static const nir_search_variable search95_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search95_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search95_0_0_0.value },
   NULL,
};

static const nir_search_variable search95_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search95_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search95_0_1_0.value },
   NULL,
};
static const nir_search_expression search95_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search95_0_0.value, &search95_0_1.value },
   NULL,
};

static const nir_search_constant search95_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search95 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search95_0.value, &search95_1.value },
   NULL,
};
   
static const nir_search_variable replace95_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace95_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace95_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace95_0_0.value, &replace95_0_1.value },
   NULL,
};
static const nir_search_expression replace95 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace95_0.value },
   NULL,
};
   
static const nir_search_variable search96_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search96_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search96_0_0_0.value },
   NULL,
};

static const nir_search_variable search96_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search96_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search96_0_1_0.value },
   NULL,
};
static const nir_search_expression search96_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search96_0_0.value, &search96_0_1.value },
   NULL,
};

static const nir_search_constant search96_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search96 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search96_0.value, &search96_1.value },
   NULL,
};
   
static const nir_search_variable replace96_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace96_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace96_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace96_0_0.value, &replace96_0_1.value },
   NULL,
};
static const nir_search_expression replace96 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace96_0.value },
   NULL,
};
   
static const nir_search_variable search97_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search97_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search97_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search97_0_1_0.value },
   NULL,
};

static const nir_search_constant search97_0_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search97_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search97_0_0.value, &search97_0_1.value, &search97_0_2.value },
   NULL,
};

static const nir_search_constant search97_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search97 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search97_0.value, &search97_1.value },
   NULL,
};
   
static const nir_search_variable replace97_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace97_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace97_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace97_0_0.value, &replace97_0_1.value },
   NULL,
};
static const nir_search_expression replace97 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace97_0.value },
   NULL,
};
   
static const nir_search_variable search98_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search98_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search98_0_0_0.value },
   NULL,
};

static const nir_search_variable search98_0_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search98_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search98_0_1_0_0.value },
   NULL,
};
static const nir_search_expression search98_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search98_0_1_0.value },
   NULL,
};
static const nir_search_expression search98_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search98_0_0.value, &search98_0_1.value },
   NULL,
};

static const nir_search_constant search98_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search98 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search98_0.value, &search98_1.value },
   NULL,
};
   
static const nir_search_variable replace98_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace98_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace98 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace98_0.value, &replace98_1.value },
   NULL,
};
   
static const nir_search_variable search99_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search99_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search99_0_0.value },
   NULL,
};

static const nir_search_variable search99_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search99_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search99_1_0.value },
   NULL,
};
static const nir_search_expression search99 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search99_0.value, &search99_1.value },
   NULL,
};
   
static const nir_search_variable replace99_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace99_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace99 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace99_0.value, &replace99_1.value },
   NULL,
};
   
static const nir_search_variable search100_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search100_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search100_0_0_0.value },
   NULL,
};
static const nir_search_expression search100_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search100_0_0.value },
   NULL,
};

static const nir_search_variable search100_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search100_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search100_1_0_0.value },
   NULL,
};
static const nir_search_expression search100_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search100_1_0.value },
   NULL,
};
static const nir_search_expression search100 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search100_0.value, &search100_1.value },
   NULL,
};
   
static const nir_search_variable replace100_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace100_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace100 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace100_0.value, &replace100_1.value },
   NULL,
};
   
static const nir_search_variable search108_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search108_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search108_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search108_0_0.value, &search108_0_1.value },
   NULL,
};

static const nir_search_variable search108_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search108 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_feq,
   { &search108_0.value, &search108_1.value },
   NULL,
};
   
static const nir_search_variable replace108_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace108_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace108 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace108_0.value, &replace108_1.value },
   NULL,
};
   
static const nir_search_variable search113_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search113_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search113_0_0_0_0.value },
   NULL,
};
static const nir_search_expression search113_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search113_0_0_0.value },
   NULL,
};

static const nir_search_variable search113_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search113_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search113_0_0.value, &search113_0_1.value },
   NULL,
};

static const nir_search_constant search113_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search113 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search113_0.value, &search113_1.value },
   NULL,
};
   
static const nir_search_variable replace113_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace113_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace113_0_0.value },
   NULL,
};

static const nir_search_variable replace113_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace113_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace113_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace113_1_0.value, &replace113_1_1.value },
   NULL,
};
static const nir_search_expression replace113 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace113_0.value, &replace113_1.value },
   NULL,
};
   
static const nir_search_variable search114_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search114_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search114_0_0.value },
   NULL,
};

static const nir_search_constant search114_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search114 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search114_0.value, &search114_1.value },
   NULL,
};
   
static const nir_search_variable replace114_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace114 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace114_0.value },
   NULL,
};
   
static const nir_search_variable search119_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search119_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &search119_0_0.value },
   NULL,
};

static const nir_search_constant search119_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search119 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search119_0.value, &search119_1.value },
   NULL,
};
   
static const nir_search_variable replace119_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace119_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace119 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace119_0.value, &replace119_1.value },
   NULL,
};
   
static const nir_search_variable search125_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search125_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &search125_0_0.value },
   NULL,
};

static const nir_search_constant search125_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search125 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search125_0.value, &search125_1.value },
   NULL,
};
   
static const nir_search_variable replace125_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace125_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace125 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace125_0.value, &replace125_1.value },
   NULL,
};
   
static const nir_search_variable search274_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search274_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search274_0_0.value },
   NULL,
};

static const nir_search_variable search274_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search274 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search274_0.value, &search274_1.value },
   NULL,
};
   
static const nir_search_variable replace274_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace274_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace274 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace274_0.value, &replace274_1.value },
   NULL,
};
   
static const nir_search_variable search504_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search504_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search504_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search504_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search504_0_0.value, &search504_0_1.value, &search504_0_2.value },
   NULL,
};

static const nir_search_variable search504_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search504 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search504_0.value, &search504_1.value },
   NULL,
};
   
static const nir_search_variable replace504_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace504_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace504_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace504_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace504_1_0.value, &replace504_1_1.value },
   NULL,
};

static const nir_search_variable replace504_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace504_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace504_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace504_2_0.value, &replace504_2_1.value },
   NULL,
};
static const nir_search_expression replace504 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace504_0.value, &replace504_1.value, &replace504_2.value },
   NULL,
};
   
static const nir_search_variable search505_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search505_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search505_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search505_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search505_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search505_1_0.value, &search505_1_1.value, &search505_1_2.value },
   NULL,
};
static const nir_search_expression search505 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search505_0.value, &search505_1.value },
   NULL,
};
   
static const nir_search_variable replace505_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace505_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace505_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace505_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace505_1_0.value, &replace505_1_1.value },
   NULL,
};

static const nir_search_variable replace505_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace505_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace505_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace505_2_0.value, &replace505_2_1.value },
   NULL,
};
static const nir_search_expression replace505 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace505_0.value, &replace505_1.value, &replace505_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_feq_xforms[] = {
   { &search91, &replace91.value, 0 },
   { &search92, &replace92.value, 0 },
   { &search93, &replace93.value, 0 },
   { &search94, &replace94.value, 0 },
   { &search95, &replace95.value, 0 },
   { &search96, &replace96.value, 0 },
   { &search97, &replace97.value, 0 },
   { &search98, &replace98.value, 0 },
   { &search99, &replace99.value, 0 },
   { &search100, &replace100.value, 0 },
   { &search108, &replace108.value, 0 },
   { &search113, &replace113.value, 0 },
   { &search114, &replace114.value, 0 },
   { &search119, &replace119.value, 0 },
   { &search125, &replace125.value, 0 },
   { &search274, &replace274.value, 0 },
   { &search504, &replace504.value, 0 },
   { &search505, &replace505.value, 0 },
};
   
static const nir_search_variable search101_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search101_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search101_0_0_0_0.value },
   NULL,
};

static const nir_search_variable search101_0_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search101_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search101_0_0_1_0.value },
   NULL,
};
static const nir_search_expression search101_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search101_0_0_0.value, &search101_0_0_1.value },
   NULL,
};
static const nir_search_expression search101_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search101_0_0.value },
   NULL,
};

static const nir_search_constant search101_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search101 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search101_0.value, &search101_1.value },
   NULL,
};
   
static const nir_search_variable replace101_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace101_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace101 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace101_0.value, &replace101_1.value },
   NULL,
};
   
static const nir_search_constant search102_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search102_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search102_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search102_1_0_0.value },
   NULL,
};

static const nir_search_variable search102_1_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search102_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search102_1_1_0.value },
   NULL,
};
static const nir_search_expression search102_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search102_1_0.value, &search102_1_1.value },
   NULL,
};
static const nir_search_expression search102 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search102_0.value, &search102_1.value },
   NULL,
};
   
static const nir_search_variable replace102_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace102_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace102 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace102_0.value, &replace102_1.value },
   NULL,
};
   
static const nir_search_variable search105_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search105_0_1_0_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search105_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search105_0_1_0_0_0.value },
   NULL,
};

static const nir_search_variable search105_0_1_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search105_0_1_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search105_0_1_0_1_0.value },
   NULL,
};
static const nir_search_expression search105_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search105_0_1_0_0.value, &search105_0_1_0_1.value },
   NULL,
};
static const nir_search_expression search105_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search105_0_1_0.value },
   NULL,
};
static const nir_search_expression search105_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search105_0_0.value, &search105_0_1.value },
   NULL,
};

static const nir_search_constant search105_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search105 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search105_0.value, &search105_1.value },
   NULL,
};
   
static const nir_search_variable replace105_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace105_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace105_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace105_0_0.value, &replace105_0_1.value },
   NULL,
};

static const nir_search_variable replace105_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace105_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace105_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace105_1_0.value, &replace105_1_1.value },
   NULL,
};
static const nir_search_expression replace105 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace105_0.value, &replace105_1.value },
   NULL,
};
   
static const nir_search_variable search106_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search106_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search106_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search106_0_0.value, &search106_0_1.value },
   NULL,
};

static const nir_search_variable search106_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search106 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flt,
   { &search106_0.value, &search106_1.value },
   NULL,
};
   
static const nir_search_variable replace106_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace106_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace106 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace106_0.value, &replace106_1.value },
   NULL,
};
   
static const nir_search_variable search122_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search122_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &search122_0_0.value },
   NULL,
};

static const nir_search_constant search122_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search122 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search122_0.value, &search122_1.value },
   NULL,
};
   
static const nir_search_constant replace122 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
   
static const nir_search_constant search123_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search123_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search123_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &search123_1_0.value },
   NULL,
};
static const nir_search_expression search123 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search123_0.value, &search123_1.value },
   NULL,
};
   
static const nir_search_constant replace123_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable replace123_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace123 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace123_0.value, &replace123_1.value },
   NULL,
};
   
static const nir_search_variable search128_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search128_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &search128_0_0.value },
   NULL,
};

static const nir_search_constant search128_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search128 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search128_0.value, &search128_1.value },
   NULL,
};
   
static const nir_search_variable replace128_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace128_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace128 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace128_0.value, &replace128_1.value },
   NULL,
};
   
static const nir_search_constant search129_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search129_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search129_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &search129_1_0.value },
   NULL,
};
static const nir_search_expression search129 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search129_0.value, &search129_1.value },
   NULL,
};
   
static const nir_search_constant replace129_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable replace129_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace129 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace129_0.value, &replace129_1.value },
   NULL,
};
   
static const nir_search_constant search130_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search130_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search130_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search130_1_0.value },
   NULL,
};
static const nir_search_expression search130 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flt,
   { &search130_0.value, &search130_1.value },
   NULL,
};
   
static const nir_search_variable replace130_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace130_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace130 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace130_0.value, &replace130_1.value },
   NULL,
};
   
static const nir_search_variable search131_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search131_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search131_0_0_0.value },
   NULL,
};
static const nir_search_expression search131_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search131_0_0.value },
   NULL,
};

static const nir_search_constant search131_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search131 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flt,
   { &search131_0.value, &search131_1.value },
   NULL,
};
   
static const nir_search_variable replace131_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace131_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace131 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace131_0.value, &replace131_1.value },
   NULL,
};
   
static const nir_search_variable search139_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search139_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search139_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search139_0_1_0.value },
   NULL,
};
static const nir_search_expression search139_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search139_0_0.value, &search139_0_1.value },
   (is_used_once),
};

static const nir_search_constant search139_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search139 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search139_0.value, &search139_1.value },
   NULL,
};
   
static const nir_search_variable replace139_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace139_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace139 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace139_0.value, &replace139_1.value },
   NULL,
};
   
static const nir_search_variable search225_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search225_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search225_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search225_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search225_1_0.value, &search225_1_1.value },
   NULL,
};
static const nir_search_expression search225 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flt,
   { &search225_0.value, &search225_1.value },
   NULL,
};
   
static const nir_search_variable replace225_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace225_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace225 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace225_0.value, &replace225_1.value },
   NULL,
};
   
static const nir_search_variable search226_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search226_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search226_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search226_0_0.value, &search226_0_1.value },
   NULL,
};

static const nir_search_variable search226_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search226 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flt,
   { &search226_0.value, &search226_1.value },
   NULL,
};
   
static const nir_search_variable replace226_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace226_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace226 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace226_0.value, &replace226_1.value },
   NULL,
};
   
static const nir_search_variable search229_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search229_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search229_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search229_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search229_1_0.value, &search229_1_1.value },
   NULL,
};
static const nir_search_expression search229 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flt,
   { &search229_0.value, &search229_1.value },
   NULL,
};
   
static const nir_search_constant replace229 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
   
static const nir_search_variable search230_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search230_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search230_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search230_0_0.value, &search230_0_1.value },
   NULL,
};

static const nir_search_variable search230_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search230 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flt,
   { &search230_0.value, &search230_1.value },
   NULL,
};
   
static const nir_search_constant replace230 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
   
static const nir_search_variable search280_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search280_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search280_0_0_0.value },
   NULL,
};
static const nir_search_expression search280_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search280_0_0.value },
   NULL,
};

static const nir_search_constant search280_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search280 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search280_0.value, &search280_1.value },
   NULL,
};
   
static const nir_search_variable replace280 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_constant search281_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search281_0_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search281_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search281_0_1_0.value },
   NULL,
};
static const nir_search_expression search281_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &search281_0_0.value, &search281_0_1.value },
   NULL,
};

static const nir_search_constant search281_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search281 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search281_0.value, &search281_1.value },
   NULL,
};
   
static const nir_search_variable replace281 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search500_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search500_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search500_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search500_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search500_0_0.value, &search500_0_1.value, &search500_0_2.value },
   NULL,
};

static const nir_search_variable search500_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search500 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search500_0.value, &search500_1.value },
   NULL,
};
   
static const nir_search_variable replace500_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace500_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace500_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace500_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace500_1_0.value, &replace500_1_1.value },
   NULL,
};

static const nir_search_variable replace500_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace500_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace500_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace500_2_0.value, &replace500_2_1.value },
   NULL,
};
static const nir_search_expression replace500 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace500_0.value, &replace500_1.value, &replace500_2.value },
   NULL,
};
   
static const nir_search_variable search501_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search501_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search501_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search501_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search501_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search501_1_0.value, &search501_1_1.value, &search501_1_2.value },
   NULL,
};
static const nir_search_expression search501 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search501_0.value, &search501_1.value },
   NULL,
};
   
static const nir_search_variable replace501_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace501_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace501_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace501_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace501_1_0.value, &replace501_1_1.value },
   NULL,
};

static const nir_search_variable replace501_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace501_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace501_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace501_2_0.value, &replace501_2_1.value },
   NULL,
};
static const nir_search_expression replace501 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace501_0.value, &replace501_1.value, &replace501_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_flt_xforms[] = {
   { &search101, &replace101.value, 0 },
   { &search102, &replace102.value, 0 },
   { &search105, &replace105.value, 0 },
   { &search106, &replace106.value, 0 },
   { &search122, &replace122.value, 0 },
   { &search123, &replace123.value, 0 },
   { &search128, &replace128.value, 0 },
   { &search129, &replace129.value, 0 },
   { &search130, &replace130.value, 0 },
   { &search131, &replace131.value, 0 },
   { &search139, &replace139.value, 0 },
   { &search225, &replace225.value, 0 },
   { &search226, &replace226.value, 0 },
   { &search229, &replace229.value, 0 },
   { &search230, &replace230.value, 0 },
   { &search280, &replace280.value, 0 },
   { &search281, &replace281.value, 0 },
   { &search500, &replace500.value, 0 },
   { &search501, &replace501.value, 0 },
};
   
static const nir_search_variable search110_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search110_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search110_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search110_0_0.value, &search110_0_1.value },
   NULL,
};

static const nir_search_variable search110_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search110 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search110_0.value, &search110_1.value },
   NULL,
};
   
static const nir_search_variable replace110_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace110_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace110 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace110_0.value, &replace110_1.value },
   NULL,
};
   
static const nir_search_variable search116_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search116_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search116_0_0.value },
   NULL,
};

static const nir_search_constant search116_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search116 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search116_0.value, &search116_1.value },
   NULL,
};
   
static const nir_search_variable replace116_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace116 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace116_0.value },
   NULL,
};
   
static const nir_search_variable search284_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search284_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search284 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search284_0.value, &search284_1.value },
   NULL,
};
   
static const nir_search_constant replace284 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search340_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_bool,
   NULL,
};

static const nir_search_constant search340_1 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
static const nir_search_expression search340 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search340_0.value, &search340_1.value },
   NULL,
};
   
static const nir_search_variable replace340 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search343_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_bool,
   NULL,
};

static const nir_search_constant search343_1 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
static const nir_search_expression search343 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search343_0.value, &search343_1.value },
   (is_not_used_by_if),
};
   
static const nir_search_variable replace343_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace343 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace343_0.value },
   NULL,
};
   
static const nir_search_variable search512_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search512_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search512_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search512_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search512_0_0.value, &search512_0_1.value, &search512_0_2.value },
   NULL,
};

static const nir_search_variable search512_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search512 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search512_0.value, &search512_1.value },
   NULL,
};
   
static const nir_search_variable replace512_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace512_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace512_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace512_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace512_1_0.value, &replace512_1_1.value },
   NULL,
};

static const nir_search_variable replace512_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace512_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace512_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace512_2_0.value, &replace512_2_1.value },
   NULL,
};
static const nir_search_expression replace512 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace512_0.value, &replace512_1.value, &replace512_2.value },
   NULL,
};
   
static const nir_search_variable search513_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search513_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search513_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search513_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search513_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search513_1_0.value, &search513_1_1.value, &search513_1_2.value },
   NULL,
};
static const nir_search_expression search513 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search513_0.value, &search513_1.value },
   NULL,
};
   
static const nir_search_variable replace513_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace513_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace513_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace513_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace513_1_0.value, &replace513_1_1.value },
   NULL,
};

static const nir_search_variable replace513_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace513_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace513_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace513_2_0.value, &replace513_2_1.value },
   NULL,
};
static const nir_search_expression replace513 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace513_0.value, &replace513_1.value, &replace513_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ieq_xforms[] = {
   { &search110, &replace110.value, 0 },
   { &search116, &replace116.value, 0 },
   { &search284, &replace284.value, 0 },
   { &search340, &replace340.value, 0 },
   { &search343, &replace343.value, 0 },
   { &search512, &replace512.value, 0 },
   { &search513, &replace513.value, 0 },
};
   
static const nir_search_variable search111_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search111_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search111_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search111_0_0.value, &search111_0_1.value },
   NULL,
};

static const nir_search_variable search111_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search111 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search111_0.value, &search111_1.value },
   NULL,
};
   
static const nir_search_variable replace111_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace111_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace111 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &replace111_0.value, &replace111_1.value },
   NULL,
};
   
static const nir_search_variable search117_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search117_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search117_0_0.value },
   NULL,
};

static const nir_search_constant search117_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search117 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search117_0.value, &search117_1.value },
   NULL,
};
   
static const nir_search_variable replace117 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search285_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search285_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search285 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search285_0.value, &search285_1.value },
   NULL,
};
   
static const nir_search_constant replace285 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
   
static const nir_search_variable search341_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_bool,
   NULL,
};

static const nir_search_constant search341_1 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
static const nir_search_expression search341 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search341_0.value, &search341_1.value },
   (is_not_used_by_if),
};
   
static const nir_search_variable replace341_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace341 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace341_0.value },
   NULL,
};
   
static const nir_search_variable search342_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_bool,
   NULL,
};

static const nir_search_constant search342_1 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
static const nir_search_expression search342 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search342_0.value, &search342_1.value },
   NULL,
};
   
static const nir_search_variable replace342 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search514_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search514_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search514_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search514_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search514_0_0.value, &search514_0_1.value, &search514_0_2.value },
   NULL,
};

static const nir_search_variable search514_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search514 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search514_0.value, &search514_1.value },
   NULL,
};
   
static const nir_search_variable replace514_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace514_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace514_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace514_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &replace514_1_0.value, &replace514_1_1.value },
   NULL,
};

static const nir_search_variable replace514_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace514_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace514_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &replace514_2_0.value, &replace514_2_1.value },
   NULL,
};
static const nir_search_expression replace514 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace514_0.value, &replace514_1.value, &replace514_2.value },
   NULL,
};
   
static const nir_search_variable search515_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search515_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search515_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search515_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search515_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search515_1_0.value, &search515_1_1.value, &search515_1_2.value },
   NULL,
};
static const nir_search_expression search515 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search515_0.value, &search515_1.value },
   NULL,
};
   
static const nir_search_variable replace515_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace515_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace515_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace515_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &replace515_1_0.value, &replace515_1_1.value },
   NULL,
};

static const nir_search_variable replace515_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace515_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace515_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &replace515_2_0.value, &replace515_2_1.value },
   NULL,
};
static const nir_search_expression replace515 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace515_0.value, &replace515_1.value, &replace515_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ine_xforms[] = {
   { &search111, &replace111.value, 0 },
   { &search117, &replace117.value, 0 },
   { &search285, &replace285.value, 0 },
   { &search341, &replace341.value, 0 },
   { &search342, &replace342.value, 0 },
   { &search514, &replace514.value, 0 },
   { &search515, &replace515.value, 0 },
};
   
static const nir_search_variable search134_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search134_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search134_0_0.value },
   (is_used_once),
};

static const nir_search_variable search134_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search134_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search134_1_0.value },
   NULL,
};
static const nir_search_expression search134 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search134_0.value, &search134_1.value },
   NULL,
};
   
static const nir_search_variable replace134_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace134_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace134_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace134_0_0.value, &replace134_0_1.value },
   NULL,
};
static const nir_search_expression replace134 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace134_0.value },
   NULL,
};
   
static const nir_search_variable search135_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search135_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search135_0_0_0.value },
   (is_used_once),
};
static const nir_search_expression search135_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search135_0_0.value },
   (is_used_once),
};

static const nir_search_variable search135_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search135_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search135_1_0_0.value },
   NULL,
};
static const nir_search_expression search135_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search135_1_0.value },
   NULL,
};
static const nir_search_expression search135 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search135_0.value, &search135_1.value },
   NULL,
};
   
static const nir_search_variable replace135_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace135_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace135_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace135_0_0_0.value, &replace135_0_0_1.value },
   NULL,
};
static const nir_search_expression replace135_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace135_0_0.value },
   NULL,
};
static const nir_search_expression replace135 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace135_0.value },
   NULL,
};
   
static const nir_search_variable search154_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search154_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search154 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search154_0.value, &search154_1.value },
   NULL,
};
   
static const nir_search_variable replace154 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search159_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search159_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search159_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search159_0_0.value, &search159_0_1.value },
   NULL,
};

static const nir_search_variable search159_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search159 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search159_0.value, &search159_1.value },
   NULL,
};
   
static const nir_search_variable replace159_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace159_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace159 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace159_0.value, &replace159_1.value },
   NULL,
};
   
static const nir_search_variable search165_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search165_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search165_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search165_1_0.value },
   NULL,
};
static const nir_search_expression search165 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search165_0.value, &search165_1.value },
   NULL,
};
   
static const nir_search_variable replace165_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace165 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace165_0.value },
   NULL,
};
   
static const nir_search_variable search173_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search173_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search173_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search173_1_0_0.value },
   NULL,
};
static const nir_search_expression search173_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search173_1_0.value },
   NULL,
};
static const nir_search_expression search173 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search173_0.value, &search173_1.value },
   NULL,
};
   
static const nir_search_variable replace173 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search175_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search175_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search175_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search175_1_0.value },
   NULL,
};
static const nir_search_expression search175 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search175_0.value, &search175_1.value },
   NULL,
};
   
static const nir_search_variable replace175_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace175 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace175_0.value },
   NULL,
};
   
static const nir_search_variable search177_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search177_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search177_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search177_1_0.value },
   NULL,
};
static const nir_search_expression search177 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search177_0.value, &search177_1.value },
   NULL,
};
   
static const nir_search_variable replace177_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace177 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace177_0.value },
   NULL,
};
   
static const nir_search_variable search180_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search180_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression search180_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search180_0_0.value, &search180_0_1.value },
   NULL,
};

static const nir_search_constant search180_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search180 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fmax,
   { &search180_0.value, &search180_1.value },
   NULL,
};
   
static const nir_search_variable replace180_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace180 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &replace180_0.value },
   NULL,
};
   
static const nir_search_variable search187_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search187_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &search187_0_0.value },
   NULL,
};

static const nir_search_variable search187_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   true,
   nir_type_invalid,
   (is_zero_to_one),
};
static const nir_search_expression search187 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search187_0.value, &search187_1.value },
   NULL,
};
   
static const nir_search_variable replace187_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace187_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace187_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace187_0_0.value, &replace187_0_1.value },
   NULL,
};
static const nir_search_expression replace187 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &replace187_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fmax_xforms[] = {
   { &search134, &replace134.value, 0 },
   { &search135, &replace135.value, 0 },
   { &search154, &replace154.value, 0 },
   { &search159, &replace159.value, 0 },
   { &search165, &replace165.value, 0 },
   { &search173, &replace173.value, 0 },
   { &search175, &replace175.value, 0 },
   { &search177, &replace177.value, 0 },
   { &search180, &replace180.value, 9 },
   { &search187, &replace187.value, 0 },
};
   
static const nir_search_variable search136_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search136_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search136_0_0.value },
   (is_used_once),
};

static const nir_search_variable search136_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search136_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search136_1_0.value },
   NULL,
};
static const nir_search_expression search136 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search136_0.value, &search136_1.value },
   NULL,
};
   
static const nir_search_variable replace136_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace136_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace136_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace136_0_0.value, &replace136_0_1.value },
   NULL,
};
static const nir_search_expression replace136 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace136_0.value },
   NULL,
};
   
static const nir_search_variable search137_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search137_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search137_0_0_0.value },
   (is_used_once),
};
static const nir_search_expression search137_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search137_0_0.value },
   (is_used_once),
};

static const nir_search_variable search137_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search137_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search137_1_0_0.value },
   NULL,
};
static const nir_search_expression search137_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search137_1_0.value },
   NULL,
};
static const nir_search_expression search137 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search137_0.value, &search137_1.value },
   NULL,
};
   
static const nir_search_variable replace137_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace137_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace137_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace137_0_0_0.value, &replace137_0_0_1.value },
   NULL,
};
static const nir_search_expression replace137_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace137_0_0.value },
   NULL,
};
static const nir_search_expression replace137 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace137_0.value },
   NULL,
};
   
static const nir_search_variable search138_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search138_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search138_0_0.value },
   NULL,
};

static const nir_search_variable search138_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search138 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search138_0.value, &search138_1.value },
   NULL,
};
   
static const nir_search_variable replace138_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace138_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace138_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression replace138_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace138_1_0.value, &replace138_1_1.value },
   NULL,
};

static const nir_search_variable replace138_2_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace138_2_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace138_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace138_2_0.value, &replace138_2_1.value },
   NULL,
};
static const nir_search_expression replace138 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace138_0.value, &replace138_1.value, &replace138_2.value },
   NULL,
};
   
static const nir_search_variable search153_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search153_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search153 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search153_0.value, &search153_1.value },
   NULL,
};
   
static const nir_search_variable replace153 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search162_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search162_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search162_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search162_0_0.value, &search162_0_1.value },
   NULL,
};

static const nir_search_variable search162_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search162 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search162_0.value, &search162_1.value },
   NULL,
};
   
static const nir_search_variable replace162_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace162_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace162 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace162_0.value, &replace162_1.value },
   NULL,
};
   
static const nir_search_variable search167_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search167_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search167_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search167_1_0.value },
   NULL,
};
static const nir_search_expression search167 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search167_0.value, &search167_1.value },
   NULL,
};
   
static const nir_search_variable replace167_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace167_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace167_0_0.value },
   NULL,
};
static const nir_search_expression replace167 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace167_0.value },
   NULL,
};
   
static const nir_search_variable search169_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search169_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search169_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search169_1_0_0.value },
   NULL,
};
static const nir_search_expression search169_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search169_1_0.value },
   NULL,
};
static const nir_search_expression search169 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search169_0.value, &search169_1.value },
   NULL,
};
   
static const nir_search_variable replace169_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace169_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace169_0_0.value },
   NULL,
};
static const nir_search_expression replace169 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace169_0.value },
   NULL,
};
   
static const nir_search_variable search171_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search171_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search171_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search171_1_0.value },
   NULL,
};
static const nir_search_expression search171 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search171_0.value, &search171_1.value },
   NULL,
};
   
static const nir_search_variable replace171 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search179_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search179_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search179_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search179_0_0.value, &search179_0_1.value },
   NULL,
};

static const nir_search_constant search179_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression search179 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fmin,
   { &search179_0.value, &search179_1.value },
   NULL,
};
   
static const nir_search_variable replace179_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace179 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &replace179_0.value },
   NULL,
};
   
static const nir_search_variable search184_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search184_0_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search184_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search184_0_0_0_0.value, &search184_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search184_0_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search184_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search184_0_0_0.value, &search184_0_0_1.value },
   NULL,
};

static const nir_search_variable search184_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search184_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search184_0_0.value, &search184_0_1.value },
   NULL,
};

static const nir_search_variable search184_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search184 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search184_0.value, &search184_1.value },
   NULL,
};
   
static const nir_search_variable replace184_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace184_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace184_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace184_0_0.value, &replace184_0_1.value },
   NULL,
};

static const nir_search_variable replace184_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace184 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace184_0.value, &replace184_1.value },
   NULL,
};
   
static const nir_search_variable search188_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search188_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &search188_0_0.value },
   NULL,
};

static const nir_search_variable search188_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   true,
   nir_type_invalid,
   (is_zero_to_one),
};
static const nir_search_expression search188 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search188_0.value, &search188_1.value },
   NULL,
};
   
static const nir_search_variable replace188_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace188_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace188_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace188_0_0.value, &replace188_0_1.value },
   NULL,
};
static const nir_search_expression replace188 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &replace188_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fmin_xforms[] = {
   { &search136, &replace136.value, 0 },
   { &search137, &replace137.value, 0 },
   { &search138, &replace138.value, 0 },
   { &search153, &replace153.value, 0 },
   { &search162, &replace162.value, 0 },
   { &search167, &replace167.value, 0 },
   { &search169, &replace169.value, 0 },
   { &search171, &replace171.value, 0 },
   { &search179, &replace179.value, 9 },
   { &search184, &replace184.value, 0 },
   { &search188, &replace188.value, 0 },
};
   
static const nir_search_variable search141_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search141_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search141_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search141_0_0.value, &search141_0_1.value },
   NULL,
};

static const nir_search_variable search141_1 = {
   { nir_search_value_variable, 0 },
   0, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search141_2 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search141 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_bcsel,
   { &search141_0.value, &search141_1.value, &search141_2.value },
   NULL,
};
   
static const nir_search_variable replace141_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace141_1 = {
   { nir_search_value_variable, 0 },
   0, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace141 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace141_0.value, &replace141_1.value },
   NULL,
};
   
static const nir_search_variable search142_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search142_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search142_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search142_0_0.value, &search142_0_1.value },
   NULL,
};

static const nir_search_variable search142_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search142_2 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search142 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_bcsel,
   { &search142_0.value, &search142_1.value, &search142_2.value },
   NULL,
};
   
static const nir_search_variable replace142_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace142_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace142 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace142_0.value, &replace142_1.value },
   NULL,
};
   
static const nir_search_variable search143_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search143_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search143_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search143_0_0.value, &search143_0_1.value },
   NULL,
};

static const nir_search_variable search143_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search143_2 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search143 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_bcsel,
   { &search143_0.value, &search143_1.value, &search143_2.value },
   NULL,
};
   
static const nir_search_variable replace143_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace143_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace143 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace143_0.value, &replace143_1.value },
   NULL,
};
   
static const nir_search_variable search144_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search144_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search144_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search144_0_0.value, &search144_0_1.value },
   NULL,
};

static const nir_search_variable search144_1 = {
   { nir_search_value_variable, 0 },
   0, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search144_2 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search144 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_bcsel,
   { &search144_0.value, &search144_1.value, &search144_2.value },
   NULL,
};
   
static const nir_search_variable replace144_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace144_1 = {
   { nir_search_value_variable, 0 },
   0, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace144 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace144_0.value, &replace144_1.value },
   NULL,
};
   
static const nir_search_variable search145_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search145_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search145_0_0.value },
   NULL,
};

static const nir_search_variable search145_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search145_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search145 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search145_0.value, &search145_1.value, &search145_2.value },
   NULL,
};
   
static const nir_search_variable replace145_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace145_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace145_2 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace145 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace145_0.value, &replace145_1.value, &replace145_2.value },
   NULL,
};
   
static const nir_search_variable search146_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search146_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search146_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search146_1_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search146_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search146_1_0.value, &search146_1_1.value, &search146_1_2.value },
   NULL,
};

static const nir_search_variable search146_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search146 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search146_0.value, &search146_1.value, &search146_2.value },
   NULL,
};
   
static const nir_search_variable replace146_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace146_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace146_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace146 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace146_0.value, &replace146_1.value, &replace146_2.value },
   NULL,
};
   
static const nir_search_variable search147_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search147_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search147_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search147_2_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search147_2_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search147_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search147_2_0.value, &search147_2_1.value, &search147_2_2.value },
   NULL,
};
static const nir_search_expression search147 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search147_0.value, &search147_1.value, &search147_2.value },
   NULL,
};
   
static const nir_search_variable replace147_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace147_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace147_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace147 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace147_0.value, &replace147_1.value, &replace147_2.value },
   NULL,
};
   
static const nir_search_variable search148_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search148_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search148_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search148_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search148_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search148_1_0.value, &search148_1_1.value, &search148_1_2.value },
   NULL,
};

static const nir_search_variable search148_2_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search148_2_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search148_2_2 = {
   { nir_search_value_variable, 0 },
   4, /* e */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search148_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search148_2_0.value, &search148_2_1.value, &search148_2_2.value },
   (is_used_once),
};
static const nir_search_expression search148 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search148_0.value, &search148_1.value, &search148_2.value },
   NULL,
};
   
static const nir_search_variable replace148_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace148_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace148_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace148_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace148_2_2 = {
   { nir_search_value_variable, 0 },
   4, /* e */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace148_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace148_2_0.value, &replace148_2_1.value, &replace148_2_2.value },
   NULL,
};
static const nir_search_expression replace148 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace148_0.value, &replace148_1.value, &replace148_2.value },
   NULL,
};
   
static const nir_search_variable search149_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search149_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search149_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search149_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search149_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search149_1_0.value, &search149_1_1.value, &search149_1_2.value },
   (is_used_once),
};

static const nir_search_variable search149_2_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search149_2_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search149_2_2 = {
   { nir_search_value_variable, 0 },
   4, /* e */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search149_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search149_2_0.value, &search149_2_1.value, &search149_2_2.value },
   NULL,
};
static const nir_search_expression search149 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search149_0.value, &search149_1.value, &search149_2.value },
   NULL,
};
   
static const nir_search_variable replace149_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace149_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace149_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace149_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace149_2_2 = {
   { nir_search_value_variable, 0 },
   4, /* e */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace149_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace149_2_0.value, &replace149_2_1.value, &replace149_2_2.value },
   NULL,
};
static const nir_search_expression replace149 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace149_0.value, &replace149_1.value, &replace149_2.value },
   NULL,
};
   
static const nir_search_variable search150_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search150_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search150_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search150_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search150_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search150_1_0.value, &search150_1_1.value, &search150_1_2.value },
   NULL,
};

static const nir_search_variable search150_2_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search150_2_1 = {
   { nir_search_value_variable, 0 },
   4, /* e */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search150_2_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search150_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search150_2_0.value, &search150_2_1.value, &search150_2_2.value },
   (is_used_once),
};
static const nir_search_expression search150 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search150_0.value, &search150_1.value, &search150_2.value },
   NULL,
};
   
static const nir_search_variable replace150_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace150_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace150_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace150_1_2 = {
   { nir_search_value_variable, 0 },
   4, /* e */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace150_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace150_1_0.value, &replace150_1_1.value, &replace150_1_2.value },
   NULL,
};

static const nir_search_variable replace150_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace150 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace150_0.value, &replace150_1.value, &replace150_2.value },
   NULL,
};
   
static const nir_search_variable search151_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search151_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search151_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search151_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search151_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search151_1_0.value, &search151_1_1.value, &search151_1_2.value },
   (is_used_once),
};

static const nir_search_variable search151_2_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search151_2_1 = {
   { nir_search_value_variable, 0 },
   4, /* e */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search151_2_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search151_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search151_2_0.value, &search151_2_1.value, &search151_2_2.value },
   NULL,
};
static const nir_search_expression search151 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search151_0.value, &search151_1.value, &search151_2.value },
   NULL,
};
   
static const nir_search_variable replace151_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace151_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace151_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace151_1_2 = {
   { nir_search_value_variable, 0 },
   4, /* e */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace151_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace151_1_0.value, &replace151_1_1.value, &replace151_1_2.value },
   NULL,
};

static const nir_search_variable replace151_2 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace151 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace151_0.value, &replace151_1.value, &replace151_2.value },
   NULL,
};
   
static const nir_search_variable search152_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search152_1 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};

static const nir_search_variable search152_2 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   false,
   nir_type_bool,
   NULL,
};
static const nir_search_expression search152 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search152_0.value, &search152_1.value, &search152_2.value },
   NULL,
};
   
static const nir_search_variable replace152_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace152_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace152 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace152_0.value, &replace152_1.value },
   NULL,
};
   
static const nir_search_variable search344_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search344_1 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};

static const nir_search_constant search344_2 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
static const nir_search_expression search344 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search344_0.value, &search344_1.value, &search344_2.value },
   NULL,
};
   
static const nir_search_variable replace344 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search345_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search345_1 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};

static const nir_search_constant search345_2 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
static const nir_search_expression search345 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search345_0.value, &search345_1.value, &search345_2.value },
   NULL,
};
   
static const nir_search_variable replace345_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace345 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace345_0.value },
   NULL,
};
   
static const nir_search_variable search346_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search346_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_constant search346_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search346 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_bcsel,
   { &search346_0.value, &search346_1.value, &search346_2.value },
   NULL,
};
   
static const nir_search_variable replace346_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace346 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace346_0.value },
   NULL,
};
   
static const nir_search_variable search347_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search347_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_constant search347_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression search347 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_bcsel,
   { &search347_0.value, &search347_1.value, &search347_2.value },
   NULL,
};
   
static const nir_search_variable replace347_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace347_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace347_0_0.value },
   NULL,
};
static const nir_search_expression replace347 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace347_0.value },
   NULL,
};
   
static const nir_search_variable search348_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search348_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0xbff0000000000000L /* -1.0 */ },
};

static const nir_search_constant search348_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x8000000000000000L /* -0.0 */ },
};
static const nir_search_expression search348 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_bcsel,
   { &search348_0.value, &search348_1.value, &search348_2.value },
   NULL,
};
   
static const nir_search_variable replace348_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace348_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace348_0_0.value },
   NULL,
};
static const nir_search_expression replace348 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace348_0.value },
   NULL,
};
   
static const nir_search_variable search349_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search349_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x8000000000000000L /* -0.0 */ },
};

static const nir_search_constant search349_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0xbff0000000000000L /* -1.0 */ },
};
static const nir_search_expression search349 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_bcsel,
   { &search349_0.value, &search349_1.value, &search349_2.value },
   NULL,
};
   
static const nir_search_variable replace349_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace349_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace349_0_0_0.value },
   NULL,
};
static const nir_search_expression replace349_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace349_0_0.value },
   NULL,
};
static const nir_search_expression replace349 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace349_0.value },
   NULL,
};
   
static const nir_search_constant search350_0 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};

static const nir_search_variable search350_1 = {
   { nir_search_value_variable, 0 },
   0, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search350_2 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search350 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search350_0.value, &search350_1.value, &search350_2.value },
   NULL,
};
   
static const nir_search_variable replace350 = {
   { nir_search_value_variable, 0 },
   0, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_constant search351_0 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};

static const nir_search_variable search351_1 = {
   { nir_search_value_variable, 0 },
   0, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search351_2 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search351 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search351_0.value, &search351_1.value, &search351_2.value },
   NULL,
};
   
static const nir_search_variable replace351 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search352_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search352_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search352_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search352_1_0.value },
   (is_used_once),
};

static const nir_search_variable search352_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search352_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search352_2_0.value },
   NULL,
};
static const nir_search_expression search352 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search352_0.value, &search352_1.value, &search352_2.value },
   NULL,
};
   
static const nir_search_variable replace352_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace352_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace352_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace352_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace352_0_0.value, &replace352_0_1.value, &replace352_0_2.value },
   NULL,
};
static const nir_search_expression replace352 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace352_0.value },
   NULL,
};
   
static const nir_search_variable search353_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search353_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search353_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search353 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search353_0.value, &search353_1.value, &search353_2.value },
   NULL,
};
   
static const nir_search_variable replace353_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace353_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace353_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &replace353_0_0.value, &replace353_0_1.value },
   NULL,
};

static const nir_search_variable replace353_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace353_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace353 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace353_0.value, &replace353_1.value, &replace353_2.value },
   NULL,
};
   
static const nir_search_variable search354_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search354_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search354_2 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search354 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search354_0.value, &search354_1.value, &search354_2.value },
   NULL,
};
   
static const nir_search_variable replace354 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search411_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search411_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &search411_0_0_0.value },
   NULL,
};

static const nir_search_constant search411_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search411_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search411_0_0.value, &search411_0_1.value },
   NULL,
};

static const nir_search_variable search411_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search411_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &search411_1_0.value },
   NULL,
};

static const nir_search_constant search411_2 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search411 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search411_0.value, &search411_1.value, &search411_2.value },
   NULL,
};
   
static const nir_search_variable replace411_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace411 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &replace411_0.value },
   NULL,
};
   
static const nir_search_variable search412_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search412_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ifind_msb,
   { &search412_0_0_0.value },
   NULL,
};

static const nir_search_constant search412_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search412_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search412_0_0.value, &search412_0_1.value },
   NULL,
};

static const nir_search_variable search412_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search412_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ifind_msb,
   { &search412_1_0.value },
   NULL,
};

static const nir_search_constant search412_2 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search412 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search412_0.value, &search412_1.value, &search412_2.value },
   NULL,
};
   
static const nir_search_variable replace412_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace412 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ifind_msb,
   { &replace412_0.value },
   NULL,
};
   
static const nir_search_variable search413_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search413_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ufind_msb,
   { &search413_0_0_0.value },
   NULL,
};

static const nir_search_constant search413_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search413_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search413_0_0.value, &search413_0_1.value },
   NULL,
};

static const nir_search_variable search413_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search413_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ufind_msb,
   { &search413_1_0.value },
   NULL,
};

static const nir_search_constant search413_2 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search413 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search413_0.value, &search413_1.value, &search413_2.value },
   NULL,
};
   
static const nir_search_variable replace413_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace413 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ufind_msb,
   { &replace413_0.value },
   NULL,
};
   
static const nir_search_variable search414_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search414_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search414_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search414_0_0.value, &search414_0_1.value },
   NULL,
};

static const nir_search_variable search414_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search414_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &search414_1_0.value },
   NULL,
};

static const nir_search_constant search414_2 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search414 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search414_0.value, &search414_1.value, &search414_2.value },
   NULL,
};
   
static const nir_search_variable replace414_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace414 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &replace414_0.value },
   NULL,
};
   
static const nir_search_variable search415_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search415_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search415_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search415_0_0.value, &search415_0_1.value },
   NULL,
};

static const nir_search_variable search415_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search415_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ifind_msb,
   { &search415_1_0.value },
   NULL,
};

static const nir_search_constant search415_2 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search415 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search415_0.value, &search415_1.value, &search415_2.value },
   NULL,
};
   
static const nir_search_variable replace415_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace415 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ifind_msb,
   { &replace415_0.value },
   NULL,
};
   
static const nir_search_variable search416_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search416_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search416_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search416_0_0.value, &search416_0_1.value },
   NULL,
};

static const nir_search_variable search416_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search416_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ufind_msb,
   { &search416_1_0.value },
   NULL,
};

static const nir_search_constant search416_2 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search416 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search416_0.value, &search416_1.value, &search416_2.value },
   NULL,
};
   
static const nir_search_variable replace416_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace416 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ufind_msb,
   { &replace416_0.value },
   NULL,
};
   
static const nir_search_variable search417_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search417_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search417_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ine,
   { &search417_0_0.value, &search417_0_1.value },
   NULL,
};

static const nir_search_variable search417_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search417_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ifind_msb,
   { &search417_1_0.value },
   NULL,
};

static const nir_search_constant search417_2 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search417 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search417_0.value, &search417_1.value, &search417_2.value },
   NULL,
};
   
static const nir_search_variable replace417_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace417 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ifind_msb,
   { &replace417_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_bcsel_xforms[] = {
   { &search141, &replace141.value, 0 },
   { &search142, &replace142.value, 0 },
   { &search143, &replace143.value, 0 },
   { &search144, &replace144.value, 0 },
   { &search145, &replace145.value, 0 },
   { &search146, &replace146.value, 0 },
   { &search147, &replace147.value, 0 },
   { &search148, &replace148.value, 0 },
   { &search149, &replace149.value, 0 },
   { &search150, &replace150.value, 0 },
   { &search151, &replace151.value, 0 },
   { &search152, &replace152.value, 0 },
   { &search344, &replace344.value, 0 },
   { &search345, &replace345.value, 0 },
   { &search346, &replace346.value, 0 },
   { &search347, &replace347.value, 0 },
   { &search348, &replace348.value, 0 },
   { &search349, &replace349.value, 0 },
   { &search350, &replace350.value, 0 },
   { &search351, &replace351.value, 0 },
   { &search352, &replace352.value, 0 },
   { &search353, &replace353.value, 0 },
   { &search354, &replace354.value, 0 },
   { &search411, &replace411.value, 0 },
   { &search412, &replace412.value, 0 },
   { &search413, &replace413.value, 0 },
   { &search414, &replace414.value, 0 },
   { &search415, &replace415.value, 0 },
   { &search416, &replace416.value, 0 },
   { &search417, &replace417.value, 0 },
};
   
static const nir_search_variable search155_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search155_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search155 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search155_0.value, &search155_1.value },
   NULL,
};
   
static const nir_search_variable replace155 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search164_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search164_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search164_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search164_0_0.value, &search164_0_1.value },
   NULL,
};

static const nir_search_variable search164_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search164 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search164_0.value, &search164_1.value },
   NULL,
};
   
static const nir_search_variable replace164_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace164_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace164 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace164_0.value, &replace164_1.value },
   NULL,
};
   
static const nir_search_variable search168_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search168_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search168_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search168_1_0.value },
   NULL,
};
static const nir_search_expression search168 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search168_0.value, &search168_1.value },
   NULL,
};
   
static const nir_search_variable replace168_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace168_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace168_0_0.value },
   NULL,
};
static const nir_search_expression replace168 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &replace168_0.value },
   NULL,
};
   
static const nir_search_variable search170_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search170_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search170_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &search170_1_0_0.value },
   NULL,
};
static const nir_search_expression search170_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search170_1_0.value },
   NULL,
};
static const nir_search_expression search170 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search170_0.value, &search170_1.value },
   NULL,
};
   
static const nir_search_variable replace170_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace170_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace170_0_0.value },
   NULL,
};
static const nir_search_expression replace170 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &replace170_0.value },
   NULL,
};
   
static const nir_search_variable search172_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search172_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search172_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &search172_1_0.value },
   NULL,
};
static const nir_search_expression search172 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search172_0.value, &search172_1.value },
   NULL,
};
   
static const nir_search_variable replace172 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search185_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search185_0_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search185_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search185_0_0_0_0.value, &search185_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search185_0_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search185_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search185_0_0_0.value, &search185_0_0_1.value },
   NULL,
};

static const nir_search_variable search185_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search185_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search185_0_0.value, &search185_0_1.value },
   NULL,
};

static const nir_search_variable search185_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search185 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search185_0.value, &search185_1.value },
   NULL,
};
   
static const nir_search_variable replace185_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace185_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace185_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace185_0_0.value, &replace185_0_1.value },
   NULL,
};

static const nir_search_variable replace185_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace185 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace185_0.value, &replace185_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_imin_xforms[] = {
   { &search155, &replace155.value, 0 },
   { &search164, &replace164.value, 0 },
   { &search168, &replace168.value, 0 },
   { &search170, &replace170.value, 0 },
   { &search172, &replace172.value, 0 },
   { &search185, &replace185.value, 0 },
};
   
static const nir_search_variable search156_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search156_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search156 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search156_0.value, &search156_1.value },
   NULL,
};
   
static const nir_search_variable replace156 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search161_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search161_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search161_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search161_0_0.value, &search161_0_1.value },
   NULL,
};

static const nir_search_variable search161_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search161 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search161_0.value, &search161_1.value },
   NULL,
};
   
static const nir_search_variable replace161_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace161_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace161 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace161_0.value, &replace161_1.value },
   NULL,
};
   
static const nir_search_variable search166_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search166_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search166_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search166_1_0.value },
   NULL,
};
static const nir_search_expression search166 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search166_0.value, &search166_1.value },
   NULL,
};
   
static const nir_search_variable replace166_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace166 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace166_0.value },
   NULL,
};
   
static const nir_search_variable search174_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search174_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search174_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &search174_1_0_0.value },
   NULL,
};
static const nir_search_expression search174_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search174_1_0.value },
   NULL,
};
static const nir_search_expression search174 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search174_0.value, &search174_1.value },
   NULL,
};
   
static const nir_search_variable replace174 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search176_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search176_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search176_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &search176_1_0.value },
   NULL,
};
static const nir_search_expression search176 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search176_0.value, &search176_1.value },
   NULL,
};
   
static const nir_search_variable replace176_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace176 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace176_0.value },
   NULL,
};
   
static const nir_search_variable search178_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search178_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search178_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search178_1_0.value },
   NULL,
};
static const nir_search_expression search178 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search178_0.value, &search178_1.value },
   NULL,
};
   
static const nir_search_variable replace178_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace178 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &replace178_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_imax_xforms[] = {
   { &search156, &replace156.value, 0 },
   { &search161, &replace161.value, 0 },
   { &search166, &replace166.value, 0 },
   { &search174, &replace174.value, 0 },
   { &search176, &replace176.value, 0 },
   { &search178, &replace178.value, 0 },
};
   
static const nir_search_variable search157_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search157_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search157 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search157_0.value, &search157_1.value },
   NULL,
};
   
static const nir_search_variable replace157 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search163_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search163_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search163_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search163_0_0.value, &search163_0_1.value },
   NULL,
};

static const nir_search_variable search163_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search163 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search163_0.value, &search163_1.value },
   NULL,
};
   
static const nir_search_variable replace163_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace163_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace163 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &replace163_0.value, &replace163_1.value },
   NULL,
};
   
static const nir_search_variable search186_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search186_0_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search186_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search186_0_0_0_0.value, &search186_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search186_0_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search186_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search186_0_0_0.value, &search186_0_0_1.value },
   NULL,
};

static const nir_search_variable search186_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search186_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search186_0_0.value, &search186_0_1.value },
   NULL,
};

static const nir_search_variable search186_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search186 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search186_0.value, &search186_1.value },
   NULL,
};
   
static const nir_search_variable replace186_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace186_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace186_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &replace186_0_0.value, &replace186_0_1.value },
   NULL,
};

static const nir_search_variable replace186_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace186 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &replace186_0.value, &replace186_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_umin_xforms[] = {
   { &search157, &replace157.value, 0 },
   { &search163, &replace163.value, 0 },
   { &search186, &replace186.value, 0 },
};
   
static const nir_search_variable search158_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search158_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search158 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search158_0.value, &search158_1.value },
   NULL,
};
   
static const nir_search_variable replace158 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search160_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search160_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search160_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search160_0_0.value, &search160_0_1.value },
   NULL,
};

static const nir_search_variable search160_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search160 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search160_0.value, &search160_1.value },
   NULL,
};
   
static const nir_search_variable replace160_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace160_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace160 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &replace160_0.value, &replace160_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_umax_xforms[] = {
   { &search158, &replace158.value, 0 },
   { &search160, &replace160.value, 0 },
};
   
static const nir_search_variable search181_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search181_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsign,
   { &search181_0_0.value },
   NULL,
};
static const nir_search_expression search181 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &search181_0.value },
   NULL,
};
   
static const nir_search_constant replace181_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable replace181_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace181_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace181_0_0.value, &replace181_0_1.value },
   NULL,
};
static const nir_search_expression replace181 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace181_0.value },
   NULL,
};
   
static const nir_search_variable search182_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search182 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &search182_0.value },
   NULL,
};
   
static const nir_search_variable replace182_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace182_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression replace182_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace182_0_0.value, &replace182_0_1.value },
   NULL,
};

static const nir_search_constant replace182_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression replace182 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace182_0.value, &replace182_1.value },
   NULL,
};
   
static const nir_search_variable search183_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search183_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &search183_0_0.value },
   NULL,
};
static const nir_search_expression search183 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &search183_0.value },
   NULL,
};
   
static const nir_search_variable replace183_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace183 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &replace183_0.value },
   NULL,
};
   
static const nir_search_variable search277_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search277_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search277_0_0_0.value },
   NULL,
};

static const nir_search_variable search277_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search277_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search277_0_1_0.value },
   NULL,
};
static const nir_search_expression search277_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search277_0_0.value, &search277_0_1.value },
   NULL,
};
static const nir_search_expression search277 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &search277_0.value },
   NULL,
};
   
static const nir_search_variable replace277_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace277_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace277_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace277_0_0.value, &replace277_0_1.value },
   NULL,
};
static const nir_search_expression replace277 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace277_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fsat_xforms[] = {
   { &search181, &replace181.value, 0 },
   { &search182, &replace182.value, 10 },
   { &search183, &replace183.value, 0 },
   { &search277, &replace277.value, 0 },
};
   
static const nir_search_variable search189_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search189_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search189_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search189_0_0_0.value, &search189_0_0_1.value },
   NULL,
};

static const nir_search_constant search189_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff /* 255 */ },
};
static const nir_search_expression search189_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search189_0_0.value, &search189_0_1.value },
   NULL,
};

static const nir_search_constant search189_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search189 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &search189_0.value, &search189_1.value },
   NULL,
};
   
static const nir_search_variable replace189_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace189_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace189_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace189_0_0.value, &replace189_0_1.value },
   NULL,
};

static const nir_search_constant replace189_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff /* 255 */ },
};
static const nir_search_expression replace189 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace189_0.value, &replace189_1.value },
   NULL,
};
   
static const nir_search_variable search433_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search433_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search433 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &search433_0.value, &search433_1.value },
   NULL,
};
   
static const nir_search_variable replace433_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace433_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace433_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression replace433_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace433_0_1_0.value, &replace433_0_1_1.value },
   NULL,
};
static const nir_search_expression replace433_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &replace433_0_0.value, &replace433_0_1.value },
   NULL,
};

static const nir_search_constant replace433_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff /* 255 */ },
};
static const nir_search_expression replace433 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace433_0.value, &replace433_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_extract_u8_xforms[] = {
   { &search189, &replace189.value, 0 },
   { &search433, &replace433.value, 33 },
};
   
static const nir_search_variable search190_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search190_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search190_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search190_0_0.value, &search190_0_1.value },
   (is_used_once),
};

static const nir_search_variable search190_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search190_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search190_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search190_1_0.value, &search190_1_1.value },
   NULL,
};
static const nir_search_expression search190 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ior,
   { &search190_0.value, &search190_1.value },
   NULL,
};
   
static const nir_search_variable replace190_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace190_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace190_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace190_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace190_1_0.value, &replace190_1_1.value },
   NULL,
};
static const nir_search_expression replace190 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace190_0.value, &replace190_1.value },
   NULL,
};
   
static const nir_search_variable search191_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search191_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search191_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search191_0_0.value, &search191_0_1.value },
   (is_used_once),
};

static const nir_search_variable search191_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search191_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search191_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search191_1_0.value, &search191_1_1.value },
   NULL,
};
static const nir_search_expression search191 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ior,
   { &search191_0.value, &search191_1.value },
   NULL,
};
   
static const nir_search_variable replace191_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace191_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace191_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace191_0_0.value, &replace191_0_1.value },
   NULL,
};

static const nir_search_variable replace191_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace191 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace191_0.value, &replace191_1.value },
   NULL,
};
   
static const nir_search_variable search192_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search192_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search192_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search192_0_0.value, &search192_0_1.value },
   (is_used_once),
};

static const nir_search_variable search192_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search192_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search192_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search192_1_0.value, &search192_1_1.value },
   NULL,
};
static const nir_search_expression search192 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ior,
   { &search192_0.value, &search192_1.value },
   NULL,
};
   
static const nir_search_variable replace192_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace192_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace192_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace192_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace192_1_0.value, &replace192_1_1.value },
   NULL,
};
static const nir_search_expression replace192 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace192_0.value, &replace192_1.value },
   NULL,
};
   
static const nir_search_variable search193_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search193_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search193_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search193_0_0.value, &search193_0_1.value },
   (is_used_once),
};

static const nir_search_variable search193_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search193_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search193_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search193_1_0.value, &search193_1_1.value },
   NULL,
};
static const nir_search_expression search193 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ior,
   { &search193_0.value, &search193_1.value },
   NULL,
};
   
static const nir_search_variable replace193_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace193_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace193_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace193_0_0.value, &replace193_0_1.value },
   NULL,
};

static const nir_search_variable replace193_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace193 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace193_0.value, &replace193_1.value },
   NULL,
};
   
static const nir_search_variable search194_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search194_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search194_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search194_0_0.value, &search194_0_1.value },
   NULL,
};

static const nir_search_variable search194_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search194_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search194_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search194_1_0.value, &search194_1_1.value },
   NULL,
};
static const nir_search_expression search194 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ior,
   { &search194_0.value, &search194_1.value },
   NULL,
};
   
static const nir_search_variable replace194_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace194_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace194_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace194_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace194_1_0.value, &replace194_1_1.value },
   NULL,
};
static const nir_search_expression replace194 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace194_0.value, &replace194_1.value },
   NULL,
};
   
static const nir_search_variable search195_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search195_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search195_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search195_0_0.value, &search195_0_1.value },
   NULL,
};

static const nir_search_variable search195_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search195_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search195_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search195_1_0.value, &search195_1_1.value },
   NULL,
};
static const nir_search_expression search195 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ior,
   { &search195_0.value, &search195_1.value },
   NULL,
};
   
static const nir_search_variable replace195_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace195_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace195_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace195_0_0.value, &replace195_0_1.value },
   NULL,
};

static const nir_search_variable replace195_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace195 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace195_0.value, &replace195_1.value },
   NULL,
};
   
static const nir_search_variable search196_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search196_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search196_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search196_0_0.value, &search196_0_1.value },
   NULL,
};

static const nir_search_variable search196_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search196_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search196_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search196_1_0.value, &search196_1_1.value },
   NULL,
};
static const nir_search_expression search196 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ior,
   { &search196_0.value, &search196_1.value },
   NULL,
};
   
static const nir_search_variable replace196_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace196_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace196_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace196_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace196_1_0.value, &replace196_1_1.value },
   NULL,
};
static const nir_search_expression replace196 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace196_0.value, &replace196_1.value },
   NULL,
};
   
static const nir_search_variable search197_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search197_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search197_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search197_0_0.value, &search197_0_1.value },
   NULL,
};

static const nir_search_variable search197_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search197_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search197_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search197_1_0.value, &search197_1_1.value },
   NULL,
};
static const nir_search_expression search197 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_ior,
   { &search197_0.value, &search197_1.value },
   NULL,
};
   
static const nir_search_variable replace197_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace197_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace197_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace197_0_0.value, &replace197_0_1.value },
   NULL,
};

static const nir_search_variable replace197_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace197 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace197_0.value, &replace197_1.value },
   NULL,
};
   
static const nir_search_variable search206_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search206_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search206_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search206_0_0.value, &search206_0_1.value },
   (is_used_once),
};

static const nir_search_variable search206_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search206_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search206_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search206_1_0.value, &search206_1_1.value },
   NULL,
};
static const nir_search_expression search206 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search206_0.value, &search206_1.value },
   NULL,
};
   
static const nir_search_variable replace206_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace206_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace206_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace206_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace206_1_0.value, &replace206_1_1.value },
   NULL,
};
static const nir_search_expression replace206 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace206_0.value, &replace206_1.value },
   NULL,
};
   
static const nir_search_variable search207_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search207_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search207_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search207_0_0.value, &search207_0_1.value },
   (is_used_once),
};

static const nir_search_variable search207_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search207_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search207_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search207_1_0.value, &search207_1_1.value },
   NULL,
};
static const nir_search_expression search207 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search207_0.value, &search207_1.value },
   NULL,
};
   
static const nir_search_variable replace207_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace207_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace207_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace207_0_0.value, &replace207_0_1.value },
   NULL,
};

static const nir_search_variable replace207_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace207 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace207_0.value, &replace207_1.value },
   NULL,
};
   
static const nir_search_variable search208_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search208_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search208_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search208_0_0.value, &search208_0_1.value },
   (is_used_once),
};

static const nir_search_variable search208_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search208_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search208_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search208_1_0.value, &search208_1_1.value },
   NULL,
};
static const nir_search_expression search208 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search208_0.value, &search208_1.value },
   NULL,
};
   
static const nir_search_variable replace208_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace208_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace208_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace208_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace208_1_0.value, &replace208_1_1.value },
   NULL,
};
static const nir_search_expression replace208 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace208_0.value, &replace208_1.value },
   NULL,
};
   
static const nir_search_variable search209_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search209_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search209_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search209_0_0.value, &search209_0_1.value },
   (is_used_once),
};

static const nir_search_variable search209_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search209_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search209_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search209_1_0.value, &search209_1_1.value },
   NULL,
};
static const nir_search_expression search209 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search209_0.value, &search209_1.value },
   NULL,
};
   
static const nir_search_variable replace209_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace209_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace209_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace209_0_0.value, &replace209_0_1.value },
   NULL,
};

static const nir_search_variable replace209_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace209 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace209_0.value, &replace209_1.value },
   NULL,
};
   
static const nir_search_variable search210_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search210_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search210_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search210_0_0.value, &search210_0_1.value },
   (is_used_once),
};

static const nir_search_variable search210_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search210_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search210_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search210_1_0.value, &search210_1_1.value },
   NULL,
};
static const nir_search_expression search210 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search210_0.value, &search210_1.value },
   NULL,
};
   
static const nir_search_variable replace210_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace210_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace210_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace210_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &replace210_1_0.value, &replace210_1_1.value },
   NULL,
};
static const nir_search_expression replace210 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace210_0.value, &replace210_1.value },
   NULL,
};
   
static const nir_search_variable search211_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search211_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search211_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search211_0_0.value, &search211_0_1.value },
   (is_used_once),
};

static const nir_search_variable search211_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search211_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search211_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search211_1_0.value, &search211_1_1.value },
   NULL,
};
static const nir_search_expression search211 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search211_0.value, &search211_1.value },
   NULL,
};
   
static const nir_search_variable replace211_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace211_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace211_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &replace211_0_0.value, &replace211_0_1.value },
   NULL,
};

static const nir_search_variable replace211_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace211 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace211_0.value, &replace211_1.value },
   NULL,
};
   
static const nir_search_variable search212_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search212_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search212_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search212_0_0.value, &search212_0_1.value },
   (is_used_once),
};

static const nir_search_variable search212_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search212_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search212_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search212_1_0.value, &search212_1_1.value },
   NULL,
};
static const nir_search_expression search212 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search212_0.value, &search212_1.value },
   NULL,
};
   
static const nir_search_variable replace212_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace212_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace212_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace212_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &replace212_1_0.value, &replace212_1_1.value },
   NULL,
};
static const nir_search_expression replace212 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace212_0.value, &replace212_1.value },
   NULL,
};
   
static const nir_search_variable search213_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search213_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search213_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search213_0_0.value, &search213_0_1.value },
   (is_used_once),
};

static const nir_search_variable search213_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search213_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search213_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search213_1_0.value, &search213_1_1.value },
   NULL,
};
static const nir_search_expression search213 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search213_0.value, &search213_1.value },
   NULL,
};
   
static const nir_search_variable replace213_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace213_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace213_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &replace213_0_0.value, &replace213_0_1.value },
   NULL,
};

static const nir_search_variable replace213_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace213 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace213_0.value, &replace213_1.value },
   NULL,
};
   
static const nir_search_variable search222_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_bool,
   NULL,
};

static const nir_search_variable search222_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search222_1_1 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
static const nir_search_expression search222_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search222_1_0.value, &search222_1_1.value },
   NULL,
};
static const nir_search_expression search222 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search222_0.value, &search222_1.value },
   NULL,
};
   
static const nir_search_constant replace222 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search223_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search223_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search223_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search223_1_0.value },
   NULL,
};
static const nir_search_expression search223 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search223_0.value, &search223_1.value },
   NULL,
};
   
static const nir_search_constant replace223 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
   
static const nir_search_variable search292_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search292_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search292 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search292_0.value, &search292_1.value },
   NULL,
};
   
static const nir_search_variable replace292 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search293_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search293_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search293 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search293_0.value, &search293_1.value },
   NULL,
};
   
static const nir_search_variable replace293 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search294_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search294_1 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
static const nir_search_expression search294 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search294_0.value, &search294_1.value },
   NULL,
};
   
static const nir_search_constant replace294 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search299_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search299_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search299_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search299_0_0.value, &search299_0_1.value },
   NULL,
};

static const nir_search_variable search299_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search299 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search299_0.value, &search299_1.value },
   NULL,
};
   
static const nir_search_variable replace299 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search300_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search300_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search300_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search300_0_0.value, &search300_0_1.value },
   NULL,
};

static const nir_search_variable search300_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search300 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search300_0.value, &search300_1.value },
   NULL,
};
   
static const nir_search_variable replace300_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace300_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace300 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace300_0.value, &replace300_1.value },
   NULL,
};
   
static const nir_search_variable search304_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search304_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search304_0_0.value },
   NULL,
};

static const nir_search_variable search304_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search304_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search304_1_0.value },
   NULL,
};
static const nir_search_expression search304 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search304_0.value, &search304_1.value },
   NULL,
};
   
static const nir_search_variable replace304_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace304_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace304_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace304_0_0.value, &replace304_0_1.value },
   NULL,
};
static const nir_search_expression replace304 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace304_0.value },
   NULL,
};
   
static const nir_search_variable search493_0_0_0_0_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_0_0_0_0_0_0_0_0_0_0_0.value, &search493_0_0_0_0_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_0_0_0_0_0_0_0_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_0_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_0_0_0_0_0_0_0_0_0_1_0.value, &search493_0_0_0_0_0_0_0_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_0_0_0_0_0_0_0_0_0_0.value, &search493_0_0_0_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff /* 16711935 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_0_0_0_0_0_0_0_0_0.value, &search493_0_0_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_0_0_0_0_0_0_0_0.value, &search493_0_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_0_0_0_0_0_0_1_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_0_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_0_0_0_0_0_0_1_0_0_0_0.value, &search493_0_0_0_0_0_0_0_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_0_0_0_0_0_0_1_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_0_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_0_0_0_0_0_0_1_0_0_1_0.value, &search493_0_0_0_0_0_0_0_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_0_0_0_0_0_0_1_0_0_0.value, &search493_0_0_0_0_0_0_0_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_0_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff00 /* 4278255360 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_0_0_0_0_0_0_1_0_0.value, &search493_0_0_0_0_0_0_0_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_0_0_0_0_0_0_1_0.value, &search493_0_0_0_0_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_0_0_0_0_0_0_0.value, &search493_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xf0f0f0f /* 252645135 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_0_0_0_0_0_0.value, &search493_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x4 /* 4 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_0_0_0_0_0.value, &search493_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_0_0_0_1_0_0_0_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_1_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_0_0_0_1_0_0_0_0_0_0_0.value, &search493_0_0_0_0_0_0_1_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_0_0_0_1_0_0_0_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_1_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_0_0_0_1_0_0_0_0_0_1_0.value, &search493_0_0_0_0_0_0_1_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_0_0_0_1_0_0_0_0_0_0.value, &search493_0_0_0_0_0_0_1_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff /* 16711935 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_0_0_0_1_0_0_0_0_0.value, &search493_0_0_0_0_0_0_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_0_0_0_1_0_0_0_0.value, &search493_0_0_0_0_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_0_0_0_1_0_0_1_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_1_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_0_0_0_1_0_0_1_0_0_0_0.value, &search493_0_0_0_0_0_0_1_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_0_0_0_1_0_0_1_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_1_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_0_0_0_1_0_0_1_0_0_1_0.value, &search493_0_0_0_0_0_0_1_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_0_0_0_1_0_0_1_0_0_0.value, &search493_0_0_0_0_0_0_1_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_1_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff00 /* 4278255360 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_0_0_0_1_0_0_1_0_0.value, &search493_0_0_0_0_0_0_1_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_0_0_0_1_0_0_1_0.value, &search493_0_0_0_0_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_0_0_0_1_0_0_0.value, &search493_0_0_0_0_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xf0f0f0f0 /* 4042322160 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_0_0_0_1_0_0.value, &search493_0_0_0_0_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x4 /* 4 */ },
};
static const nir_search_expression search493_0_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_0_0_0_1_0.value, &search493_0_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_0_0_0_0.value, &search493_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x33333333 /* 858993459 */ },
};
static const nir_search_expression search493_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_0_0_0.value, &search493_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x2 /* 2 */ },
};
static const nir_search_expression search493_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_0_0.value, &search493_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_1_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_1_0_0_0_0_0_0_0_0_0_0.value, &search493_0_0_0_1_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_1_0_0_0_0_0_0_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_0_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_1_0_0_0_0_0_0_0_0_1_0.value, &search493_0_0_0_1_0_0_0_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_1_0_0_0_0_0_0_0_0_0.value, &search493_0_0_0_1_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff /* 16711935 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_1_0_0_0_0_0_0_0_0.value, &search493_0_0_0_1_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_1_0_0_0_0_0_0_0.value, &search493_0_0_0_1_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_1_0_0_0_0_0_1_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_0_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_1_0_0_0_0_0_1_0_0_0_0.value, &search493_0_0_0_1_0_0_0_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_1_0_0_0_0_0_1_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_0_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_1_0_0_0_0_0_1_0_0_1_0.value, &search493_0_0_0_1_0_0_0_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_1_0_0_0_0_0_1_0_0_0.value, &search493_0_0_0_1_0_0_0_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_0_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff00 /* 4278255360 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_1_0_0_0_0_0_1_0_0.value, &search493_0_0_0_1_0_0_0_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_1_0_0_0_0_0_1_0.value, &search493_0_0_0_1_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_1_0_0_0_0_0_0.value, &search493_0_0_0_1_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xf0f0f0f /* 252645135 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_1_0_0_0_0_0.value, &search493_0_0_0_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x4 /* 4 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_1_0_0_0_0.value, &search493_0_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_1_0_0_1_0_0_0_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_1_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_1_0_0_1_0_0_0_0_0_0_0.value, &search493_0_0_0_1_0_0_1_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_1_0_0_1_0_0_0_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_1_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_1_0_0_1_0_0_0_0_0_1_0.value, &search493_0_0_0_1_0_0_1_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_1_0_0_1_0_0_0_0_0_0.value, &search493_0_0_0_1_0_0_1_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff /* 16711935 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_1_0_0_1_0_0_0_0_0.value, &search493_0_0_0_1_0_0_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_1_0_0_1_0_0_0_0.value, &search493_0_0_0_1_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_1_0_0_1_0_0_1_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_1_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0_0_1_0_0_1_0_0_1_0_0_0_0.value, &search493_0_0_0_1_0_0_1_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_0_0_0_1_0_0_1_0_0_1_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_1_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_1_0_0_1_0_0_1_0_0_1_0.value, &search493_0_0_0_1_0_0_1_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_1_0_0_1_0_0_1_0_0_0.value, &search493_0_0_0_1_0_0_1_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_1_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff00 /* 4278255360 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_1_0_0_1_0_0_1_0_0.value, &search493_0_0_0_1_0_0_1_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_1_0_0_1_0_0_1_0.value, &search493_0_0_0_1_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_1_0_0_1_0_0_0.value, &search493_0_0_0_1_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xf0f0f0f0 /* 4042322160 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_1_0_0_1_0_0.value, &search493_0_0_0_1_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x4 /* 4 */ },
};
static const nir_search_expression search493_0_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_1_0_0_1_0.value, &search493_0_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_1_0_0_0.value, &search493_0_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xcccccccc /* 3435973836 */ },
};
static const nir_search_expression search493_0_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0_1_0_0.value, &search493_0_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x2 /* 2 */ },
};
static const nir_search_expression search493_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_0_0_0_1_0.value, &search493_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0_0_0_0.value, &search493_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x55555555 /* 1431655765 */ },
};
static const nir_search_expression search493_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_0_0_0.value, &search493_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression search493_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_0_0.value, &search493_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_0_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_0_0_0_0_0_0_0_0_0_0_0.value, &search493_1_0_0_0_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_0_0_0_0_0_0_0_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_0_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_0_0_0_0_0_0_0_0_0_1_0.value, &search493_1_0_0_0_0_0_0_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_0_0_0_0_0_0_0_0_0_0.value, &search493_1_0_0_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff /* 16711935 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_0_0_0_0_0_0_0_0_0.value, &search493_1_0_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_0_0_0_0_0_0_0_0.value, &search493_1_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_0_0_0_0_0_0_1_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_0_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_0_0_0_0_0_0_1_0_0_0_0.value, &search493_1_0_0_0_0_0_0_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_0_0_0_0_0_0_1_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_0_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_0_0_0_0_0_0_1_0_0_1_0.value, &search493_1_0_0_0_0_0_0_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_0_0_0_0_0_0_1_0_0_0.value, &search493_1_0_0_0_0_0_0_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_0_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff00 /* 4278255360 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_0_0_0_0_0_0_1_0_0.value, &search493_1_0_0_0_0_0_0_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_0_0_0_0_0_0_1_0.value, &search493_1_0_0_0_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_0_0_0_0_0_0_0.value, &search493_1_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xf0f0f0f /* 252645135 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_0_0_0_0_0_0.value, &search493_1_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x4 /* 4 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_0_0_0_0_0.value, &search493_1_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_0_0_0_1_0_0_0_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_1_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_0_0_0_1_0_0_0_0_0_0_0.value, &search493_1_0_0_0_0_0_1_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_0_0_0_1_0_0_0_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_1_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_0_0_0_1_0_0_0_0_0_1_0.value, &search493_1_0_0_0_0_0_1_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_0_0_0_1_0_0_0_0_0_0.value, &search493_1_0_0_0_0_0_1_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff /* 16711935 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_0_0_0_1_0_0_0_0_0.value, &search493_1_0_0_0_0_0_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_0_0_0_1_0_0_0_0.value, &search493_1_0_0_0_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_0_0_0_1_0_0_1_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_1_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_0_0_0_1_0_0_1_0_0_0_0.value, &search493_1_0_0_0_0_0_1_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_0_0_0_1_0_0_1_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_1_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_0_0_0_1_0_0_1_0_0_1_0.value, &search493_1_0_0_0_0_0_1_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_0_0_0_1_0_0_1_0_0_0.value, &search493_1_0_0_0_0_0_1_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_1_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff00 /* 4278255360 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_0_0_0_1_0_0_1_0_0.value, &search493_1_0_0_0_0_0_1_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_0_0_0_1_0_0_1_0.value, &search493_1_0_0_0_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_0_0_0_1_0_0_0.value, &search493_1_0_0_0_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xf0f0f0f0 /* 4042322160 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_0_0_0_1_0_0.value, &search493_1_0_0_0_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x4 /* 4 */ },
};
static const nir_search_expression search493_1_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_0_0_0_1_0.value, &search493_1_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_0_0_0_0.value, &search493_1_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x33333333 /* 858993459 */ },
};
static const nir_search_expression search493_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_0_0_0.value, &search493_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x2 /* 2 */ },
};
static const nir_search_expression search493_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_0_0.value, &search493_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_1_0_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_1_0_0_0_0_0_0_0_0_0_0.value, &search493_1_0_0_1_0_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_1_0_0_0_0_0_0_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_0_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_1_0_0_0_0_0_0_0_0_1_0.value, &search493_1_0_0_1_0_0_0_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_1_0_0_0_0_0_0_0_0_0.value, &search493_1_0_0_1_0_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff /* 16711935 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_1_0_0_0_0_0_0_0_0.value, &search493_1_0_0_1_0_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_1_0_0_0_0_0_0_0.value, &search493_1_0_0_1_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_1_0_0_0_0_0_1_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_0_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_1_0_0_0_0_0_1_0_0_0_0.value, &search493_1_0_0_1_0_0_0_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_1_0_0_0_0_0_1_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_0_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_1_0_0_0_0_0_1_0_0_1_0.value, &search493_1_0_0_1_0_0_0_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_1_0_0_0_0_0_1_0_0_0.value, &search493_1_0_0_1_0_0_0_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_0_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff00 /* 4278255360 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_1_0_0_0_0_0_1_0_0.value, &search493_1_0_0_1_0_0_0_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_1_0_0_0_0_0_1_0.value, &search493_1_0_0_1_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_1_0_0_0_0_0_0.value, &search493_1_0_0_1_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xf0f0f0f /* 252645135 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_1_0_0_0_0_0.value, &search493_1_0_0_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x4 /* 4 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_1_0_0_0_0.value, &search493_1_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_1_0_0_1_0_0_0_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_1_0_0_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_1_0_0_1_0_0_0_0_0_0_0.value, &search493_1_0_0_1_0_0_1_0_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_1_0_0_1_0_0_0_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_1_0_0_0_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_1_0_0_1_0_0_0_0_0_1_0.value, &search493_1_0_0_1_0_0_1_0_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_1_0_0_1_0_0_0_0_0_0.value, &search493_1_0_0_1_0_0_1_0_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff /* 16711935 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_1_0_0_1_0_0_0_0_0.value, &search493_1_0_0_1_0_0_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_1_0_0_1_0_0_0_0.value, &search493_1_0_0_1_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_1_0_0_1_0_0_1_0_0_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_1_0_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search493_1_0_0_1_0_0_1_0_0_1_0_0_0_0.value, &search493_1_0_0_1_0_0_1_0_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable search493_1_0_0_1_0_0_1_0_0_1_0_0_1_0 = {
   { nir_search_value_variable, 32 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_1_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_1_0_0_1_0_0_1_0_0_1_0.value, &search493_1_0_0_1_0_0_1_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_1_0_0_1_0_0_1_0_0_0.value, &search493_1_0_0_1_0_0_1_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_1_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff00ff00 /* 4278255360 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_1_0_0_1_0_0_1_0_0.value, &search493_1_0_0_1_0_0_1_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_1_0_0_1_0_0_1_0.value, &search493_1_0_0_1_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_1_0_0_1_0_0_0.value, &search493_1_0_0_1_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xf0f0f0f0 /* 4042322160 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_1_0_0_1_0_0.value, &search493_1_0_0_1_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x4 /* 4 */ },
};
static const nir_search_expression search493_1_0_0_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_1_0_0_1_0.value, &search493_1_0_0_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_1_0_0_0.value, &search493_1_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xcccccccc /* 3435973836 */ },
};
static const nir_search_expression search493_1_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0_1_0_0.value, &search493_1_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x2 /* 2 */ },
};
static const nir_search_expression search493_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0_0_1_0.value, &search493_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression search493_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_1_0_0_0.value, &search493_1_0_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xaaaaaaaa /* 2863311530 */ },
};
static const nir_search_expression search493_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search493_1_0_0.value, &search493_1_0_1.value },
   NULL,
};

static const nir_search_constant search493_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression search493_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search493_1_0.value, &search493_1_1.value },
   NULL,
};
static const nir_search_expression search493 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search493_0.value, &search493_1.value },
   NULL,
};
   
static const nir_search_variable replace493_0 = {
   { nir_search_value_variable, 0 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace493 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bitfield_reverse,
   { &replace493_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ior_xforms[] = {
   { &search190, &replace190.value, 0 },
   { &search191, &replace191.value, 0 },
   { &search192, &replace192.value, 0 },
   { &search193, &replace193.value, 0 },
   { &search194, &replace194.value, 0 },
   { &search195, &replace195.value, 0 },
   { &search196, &replace196.value, 0 },
   { &search197, &replace197.value, 0 },
   { &search206, &replace206.value, 0 },
   { &search207, &replace207.value, 0 },
   { &search208, &replace208.value, 0 },
   { &search209, &replace209.value, 0 },
   { &search210, &replace210.value, 0 },
   { &search211, &replace211.value, 0 },
   { &search212, &replace212.value, 0 },
   { &search213, &replace213.value, 0 },
   { &search222, &replace222.value, 0 },
   { &search223, &replace223.value, 0 },
   { &search292, &replace292.value, 0 },
   { &search293, &replace293.value, 0 },
   { &search294, &replace294.value, 0 },
   { &search299, &replace299.value, 0 },
   { &search300, &replace300.value, 0 },
   { &search304, &replace304.value, 0 },
   { &search493, &replace493.value, 0 },
};
   
static const nir_search_variable search198_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search198_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search198_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search198_0_0.value, &search198_0_1.value },
   (is_used_once),
};

static const nir_search_variable search198_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search198_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search198_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search198_1_0.value, &search198_1_1.value },
   NULL,
};
static const nir_search_expression search198 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_iand,
   { &search198_0.value, &search198_1.value },
   NULL,
};
   
static const nir_search_variable replace198_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace198_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace198_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace198_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace198_1_0.value, &replace198_1_1.value },
   NULL,
};
static const nir_search_expression replace198 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace198_0.value, &replace198_1.value },
   NULL,
};
   
static const nir_search_variable search199_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search199_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search199_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search199_0_0.value, &search199_0_1.value },
   (is_used_once),
};

static const nir_search_variable search199_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search199_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search199_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search199_1_0.value, &search199_1_1.value },
   NULL,
};
static const nir_search_expression search199 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_iand,
   { &search199_0.value, &search199_1.value },
   NULL,
};
   
static const nir_search_variable replace199_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace199_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace199_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace199_0_0.value, &replace199_0_1.value },
   NULL,
};

static const nir_search_variable replace199_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace199 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace199_0.value, &replace199_1.value },
   NULL,
};
   
static const nir_search_variable search200_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search200_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search200_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search200_0_0.value, &search200_0_1.value },
   (is_used_once),
};

static const nir_search_variable search200_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search200_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search200_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search200_1_0.value, &search200_1_1.value },
   NULL,
};
static const nir_search_expression search200 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_iand,
   { &search200_0.value, &search200_1.value },
   NULL,
};
   
static const nir_search_variable replace200_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace200_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace200_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace200_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace200_1_0.value, &replace200_1_1.value },
   NULL,
};
static const nir_search_expression replace200 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace200_0.value, &replace200_1.value },
   NULL,
};
   
static const nir_search_variable search201_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search201_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search201_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search201_0_0.value, &search201_0_1.value },
   (is_used_once),
};

static const nir_search_variable search201_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search201_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search201_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search201_1_0.value, &search201_1_1.value },
   NULL,
};
static const nir_search_expression search201 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_iand,
   { &search201_0.value, &search201_1.value },
   NULL,
};
   
static const nir_search_variable replace201_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace201_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace201_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace201_0_0.value, &replace201_0_1.value },
   NULL,
};

static const nir_search_variable replace201_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace201 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace201_0.value, &replace201_1.value },
   NULL,
};
   
static const nir_search_variable search202_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search202_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search202_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search202_0_0.value, &search202_0_1.value },
   NULL,
};

static const nir_search_variable search202_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search202_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search202_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search202_1_0.value, &search202_1_1.value },
   NULL,
};
static const nir_search_expression search202 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_iand,
   { &search202_0.value, &search202_1.value },
   NULL,
};
   
static const nir_search_variable replace202_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace202_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace202_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace202_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace202_1_0.value, &replace202_1_1.value },
   NULL,
};
static const nir_search_expression replace202 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace202_0.value, &replace202_1.value },
   NULL,
};
   
static const nir_search_variable search203_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search203_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search203_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search203_0_0.value, &search203_0_1.value },
   NULL,
};

static const nir_search_variable search203_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search203_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search203_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search203_1_0.value, &search203_1_1.value },
   NULL,
};
static const nir_search_expression search203 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_iand,
   { &search203_0.value, &search203_1.value },
   NULL,
};
   
static const nir_search_variable replace203_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace203_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace203_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace203_0_0.value, &replace203_0_1.value },
   NULL,
};

static const nir_search_variable replace203_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace203 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace203_0.value, &replace203_1.value },
   NULL,
};
   
static const nir_search_variable search204_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search204_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search204_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search204_0_0.value, &search204_0_1.value },
   NULL,
};

static const nir_search_variable search204_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search204_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search204_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search204_1_0.value, &search204_1_1.value },
   NULL,
};
static const nir_search_expression search204 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_iand,
   { &search204_0.value, &search204_1.value },
   NULL,
};
   
static const nir_search_variable replace204_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace204_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace204_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace204_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace204_1_0.value, &replace204_1_1.value },
   NULL,
};
static const nir_search_expression replace204 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace204_0.value, &replace204_1.value },
   NULL,
};
   
static const nir_search_variable search205_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search205_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search205_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search205_0_0.value, &search205_0_1.value },
   NULL,
};

static const nir_search_variable search205_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search205_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search205_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search205_1_0.value, &search205_1_1.value },
   NULL,
};
static const nir_search_expression search205 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_iand,
   { &search205_0.value, &search205_1.value },
   NULL,
};
   
static const nir_search_variable replace205_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace205_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace205_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace205_0_0.value, &replace205_0_1.value },
   NULL,
};

static const nir_search_variable replace205_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace205 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace205_0.value, &replace205_1.value },
   NULL,
};
   
static const nir_search_variable search214_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search214_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search214_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search214_0_0.value, &search214_0_1.value },
   (is_used_once),
};

static const nir_search_variable search214_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search214_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search214_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search214_1_0.value, &search214_1_1.value },
   NULL,
};
static const nir_search_expression search214 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search214_0.value, &search214_1.value },
   NULL,
};
   
static const nir_search_variable replace214_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace214_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace214_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace214_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace214_1_0.value, &replace214_1_1.value },
   NULL,
};
static const nir_search_expression replace214 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace214_0.value, &replace214_1.value },
   NULL,
};
   
static const nir_search_variable search215_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search215_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search215_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search215_0_0.value, &search215_0_1.value },
   (is_used_once),
};

static const nir_search_variable search215_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search215_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search215_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search215_1_0.value, &search215_1_1.value },
   NULL,
};
static const nir_search_expression search215 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search215_0.value, &search215_1.value },
   NULL,
};
   
static const nir_search_variable replace215_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace215_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace215_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace215_0_0.value, &replace215_0_1.value },
   NULL,
};

static const nir_search_variable replace215_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace215 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace215_0.value, &replace215_1.value },
   NULL,
};
   
static const nir_search_variable search216_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search216_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search216_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search216_0_0.value, &search216_0_1.value },
   (is_used_once),
};

static const nir_search_variable search216_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search216_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search216_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search216_1_0.value, &search216_1_1.value },
   NULL,
};
static const nir_search_expression search216 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search216_0.value, &search216_1.value },
   NULL,
};
   
static const nir_search_variable replace216_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace216_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace216_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace216_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace216_1_0.value, &replace216_1_1.value },
   NULL,
};
static const nir_search_expression replace216 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace216_0.value, &replace216_1.value },
   NULL,
};
   
static const nir_search_variable search217_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search217_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search217_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search217_0_0.value, &search217_0_1.value },
   (is_used_once),
};

static const nir_search_variable search217_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search217_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search217_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search217_1_0.value, &search217_1_1.value },
   NULL,
};
static const nir_search_expression search217 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search217_0.value, &search217_1.value },
   NULL,
};
   
static const nir_search_variable replace217_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace217_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace217_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace217_0_0.value, &replace217_0_1.value },
   NULL,
};

static const nir_search_variable replace217_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace217 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace217_0.value, &replace217_1.value },
   NULL,
};
   
static const nir_search_variable search218_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search218_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search218_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search218_0_0.value, &search218_0_1.value },
   (is_used_once),
};

static const nir_search_variable search218_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search218_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search218_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search218_1_0.value, &search218_1_1.value },
   NULL,
};
static const nir_search_expression search218 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search218_0.value, &search218_1.value },
   NULL,
};
   
static const nir_search_variable replace218_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace218_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace218_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace218_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &replace218_1_0.value, &replace218_1_1.value },
   NULL,
};
static const nir_search_expression replace218 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace218_0.value, &replace218_1.value },
   NULL,
};
   
static const nir_search_variable search219_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search219_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search219_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search219_0_0.value, &search219_0_1.value },
   (is_used_once),
};

static const nir_search_variable search219_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search219_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search219_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search219_1_0.value, &search219_1_1.value },
   NULL,
};
static const nir_search_expression search219 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search219_0.value, &search219_1.value },
   NULL,
};
   
static const nir_search_variable replace219_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace219_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace219_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &replace219_0_0.value, &replace219_0_1.value },
   NULL,
};

static const nir_search_variable replace219_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace219 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace219_0.value, &replace219_1.value },
   NULL,
};
   
static const nir_search_variable search220_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search220_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search220_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search220_0_0.value, &search220_0_1.value },
   (is_used_once),
};

static const nir_search_variable search220_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search220_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search220_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search220_1_0.value, &search220_1_1.value },
   NULL,
};
static const nir_search_expression search220 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search220_0.value, &search220_1.value },
   NULL,
};
   
static const nir_search_variable replace220_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace220_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace220_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace220_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &replace220_1_0.value, &replace220_1_1.value },
   NULL,
};
static const nir_search_expression replace220 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace220_0.value, &replace220_1.value },
   NULL,
};
   
static const nir_search_variable search221_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search221_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search221_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search221_0_0.value, &search221_0_1.value },
   (is_used_once),
};

static const nir_search_variable search221_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search221_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search221_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search221_1_0.value, &search221_1_1.value },
   NULL,
};
static const nir_search_expression search221 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search221_0.value, &search221_1.value },
   NULL,
};
   
static const nir_search_variable replace221_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace221_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace221_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &replace221_0_0.value, &replace221_0_1.value },
   NULL,
};

static const nir_search_variable replace221_1 = {
   { nir_search_value_variable, 0 },
   1, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace221 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace221_0.value, &replace221_1.value },
   NULL,
};
   
static const nir_search_variable search224_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search224_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search224_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search224_0_0.value, &search224_0_1.value },
   NULL,
};

static const nir_search_variable search224_1_0 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search224_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search224_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &search224_1_0.value, &search224_1_1.value },
   NULL,
};
static const nir_search_expression search224 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search224_0.value, &search224_1.value },
   NULL,
};
   
static const nir_search_variable replace224_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace224_0_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace224_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace224_0_0.value, &replace224_0_1.value },
   NULL,
};

static const nir_search_constant replace224_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace224 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace224_0.value, &replace224_1.value },
   NULL,
};
   
static const nir_search_variable search278_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_bool,
   NULL,
};

static const nir_search_constant search278_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression search278 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search278_0.value, &search278_1.value },
   NULL,
};
   
static const nir_search_variable replace278_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace278 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace278_0.value },
   NULL,
};
   
static const nir_search_variable search289_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search289_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search289 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search289_0.value, &search289_1.value },
   NULL,
};
   
static const nir_search_variable replace289 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search290_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search290_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search290 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search290_0.value, &search290_1.value },
   NULL,
};
   
static const nir_search_variable replace290 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search291_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search291_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search291 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search291_0.value, &search291_1.value },
   NULL,
};
   
static const nir_search_constant replace291 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
   
static const nir_search_variable search301_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search301_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search301_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &search301_0_0.value, &search301_0_1.value },
   NULL,
};

static const nir_search_variable search301_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search301 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search301_0.value, &search301_1.value },
   NULL,
};
   
static const nir_search_variable replace301 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search302_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search302_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search302_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search302_0_0.value, &search302_0_1.value },
   NULL,
};

static const nir_search_variable search302_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search302 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search302_0.value, &search302_1.value },
   NULL,
};
   
static const nir_search_variable replace302_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace302_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace302 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace302_0.value, &replace302_1.value },
   NULL,
};
   
static const nir_search_variable search303_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search303_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search303_0_0.value },
   NULL,
};

static const nir_search_variable search303_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search303_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search303_1_0.value },
   NULL,
};
static const nir_search_expression search303 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search303_0.value, &search303_1.value },
   NULL,
};
   
static const nir_search_variable replace303_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace303_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace303_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace303_0_0.value, &replace303_0_1.value },
   NULL,
};
static const nir_search_expression replace303 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace303_0.value },
   NULL,
};
   
static const nir_search_constant search311_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff /* 255 */ },
};

static const nir_search_variable search311_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search311_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search311_1 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_ushr,
   { &search311_1_0.value, &search311_1_1.value },
   NULL,
};
static const nir_search_expression search311 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search311_0.value, &search311_1.value },
   NULL,
};
   
static const nir_search_variable replace311_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace311_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression replace311 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &replace311_0.value, &replace311_1.value },
   NULL,
};
   
static const nir_search_constant search312_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xffff /* 65535 */ },
};

static const nir_search_variable search312_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search312_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search312_1 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_ushr,
   { &search312_1_0.value, &search312_1_1.value },
   NULL,
};
static const nir_search_expression search312 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search312_0.value, &search312_1.value },
   NULL,
};
   
static const nir_search_variable replace312_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace312_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression replace312 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &replace312_0.value, &replace312_1.value },
   NULL,
};
   
static const nir_search_constant search380_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff /* 255 */ },
};

static const nir_search_variable search380_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search380_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search380_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search380_1_0.value, &search380_1_1.value },
   NULL,
};
static const nir_search_expression search380 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search380_0.value, &search380_1.value },
   NULL,
};
   
static const nir_search_variable replace380_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace380_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x2 /* 2 */ },
};
static const nir_search_expression replace380 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace380_0.value, &replace380_1.value },
   NULL,
};
   
static const nir_search_constant search381_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff /* 255 */ },
};

static const nir_search_variable search381_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search381_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search381_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search381_1_0.value, &search381_1_1.value },
   NULL,
};
static const nir_search_expression search381 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search381_0.value, &search381_1.value },
   NULL,
};
   
static const nir_search_variable replace381_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace381_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace381 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace381_0.value, &replace381_1.value },
   NULL,
};
   
static const nir_search_constant search382_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xff /* 255 */ },
};

static const nir_search_variable search382_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search382 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search382_0.value, &search382_1.value },
   NULL,
};
   
static const nir_search_variable replace382_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace382_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace382 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace382_0.value, &replace382_1.value },
   NULL,
};
   
static const nir_search_constant search387_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xffff /* 65535 */ },
};

static const nir_search_variable search387_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search387 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search387_0.value, &search387_1.value },
   NULL,
};
   
static const nir_search_variable replace387_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace387_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace387 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u16,
   { &replace387_0.value, &replace387_1.value },
   NULL,
};
   
static const nir_search_variable search494_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search494_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search494_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search494_0_0.value, &search494_0_1.value },
   NULL,
};

static const nir_search_variable search494_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search494_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search494_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search494_1_0.value, &search494_1_1.value },
   NULL,
};
static const nir_search_expression search494 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search494_0.value, &search494_1.value },
   NULL,
};
   
static const nir_search_variable replace494_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace494_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace494 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace494_0.value, &replace494_1.value },
   NULL,
};
   
static const nir_search_variable search495_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search495_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search495_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search495_0_0.value, &search495_0_1.value },
   NULL,
};

static const nir_search_variable search495_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search495_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search495_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search495_1_0.value, &search495_1_1.value },
   NULL,
};
static const nir_search_expression search495 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search495_0.value, &search495_1.value },
   NULL,
};
   
static const nir_search_variable replace495_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace495_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace495 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace495_0.value, &replace495_1.value },
   NULL,
};
   
static const nir_search_variable search496_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search496_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search496_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search496_0_0.value, &search496_0_1.value },
   NULL,
};

static const nir_search_variable search496_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search496_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search496_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search496_1_0.value, &search496_1_1.value },
   NULL,
};
static const nir_search_expression search496 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search496_0.value, &search496_1.value },
   NULL,
};
   
static const nir_search_variable replace496_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace496_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace496 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace496_0.value, &replace496_1.value },
   NULL,
};
   
static const nir_search_variable search497_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search497_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search497_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search497_0_0.value, &search497_0_1.value },
   NULL,
};

static const nir_search_variable search497_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search497_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search497_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &search497_1_0.value, &search497_1_1.value },
   NULL,
};
static const nir_search_expression search497 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search497_0.value, &search497_1.value },
   NULL,
};
   
static const nir_search_variable replace497_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace497_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace497 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace497_0.value, &replace497_1.value },
   NULL,
};
   
static const nir_search_variable search498_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search498_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search498_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search498_0_0.value, &search498_0_1.value },
   NULL,
};

static const nir_search_variable search498_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search498_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search498_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search498_1_0.value, &search498_1_1.value },
   NULL,
};
static const nir_search_expression search498 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search498_0.value, &search498_1.value },
   NULL,
};
   
static const nir_search_variable replace498_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace498_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace498 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace498_0.value, &replace498_1.value },
   NULL,
};
   
static const nir_search_variable search499_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search499_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search499_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search499_0_0.value, &search499_0_1.value },
   NULL,
};

static const nir_search_variable search499_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search499_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search499_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &search499_1_0.value, &search499_1_1.value },
   NULL,
};
static const nir_search_expression search499 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &search499_0.value, &search499_1.value },
   NULL,
};
   
static const nir_search_variable replace499_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace499_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace499 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace499_0.value, &replace499_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_iand_xforms[] = {
   { &search198, &replace198.value, 0 },
   { &search199, &replace199.value, 0 },
   { &search200, &replace200.value, 0 },
   { &search201, &replace201.value, 0 },
   { &search202, &replace202.value, 0 },
   { &search203, &replace203.value, 0 },
   { &search204, &replace204.value, 0 },
   { &search205, &replace205.value, 0 },
   { &search214, &replace214.value, 0 },
   { &search215, &replace215.value, 0 },
   { &search216, &replace216.value, 0 },
   { &search217, &replace217.value, 0 },
   { &search218, &replace218.value, 0 },
   { &search219, &replace219.value, 0 },
   { &search220, &replace220.value, 0 },
   { &search221, &replace221.value, 0 },
   { &search224, &replace224.value, 0 },
   { &search278, &replace278.value, 12 },
   { &search289, &replace289.value, 0 },
   { &search290, &replace290.value, 0 },
   { &search291, &replace291.value, 0 },
   { &search301, &replace301.value, 0 },
   { &search302, &replace302.value, 0 },
   { &search303, &replace303.value, 0 },
   { &search311, &replace311.value, 0 },
   { &search312, &replace312.value, 0 },
   { &search380, &replace380.value, 18 },
   { &search381, &replace381.value, 18 },
   { &search382, &replace382.value, 18 },
   { &search387, &replace387.value, 19 },
   { &search494, &replace494.value, 0 },
   { &search495, &replace495.value, 0 },
   { &search496, &replace496.value, 0 },
   { &search497, &replace497.value, 0 },
   { &search498, &replace498.value, 0 },
   { &search499, &replace499.value, 0 },
};
   
static const nir_search_variable search233_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search233_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search233_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search233_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search233_1_0.value, &search233_1_1.value },
   NULL,
};
static const nir_search_expression search233 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search233_0.value, &search233_1.value },
   NULL,
};
   
static const nir_search_variable replace233_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace233_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace233 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace233_0.value, &replace233_1.value },
   NULL,
};
   
static const nir_search_variable search234_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search234_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search234_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search234_0_0.value, &search234_0_1.value },
   NULL,
};

static const nir_search_variable search234_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search234 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search234_0.value, &search234_1.value },
   NULL,
};
   
static const nir_search_variable replace234_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace234_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace234 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace234_0.value, &replace234_1.value },
   NULL,
};
   
static const nir_search_variable search241_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search241_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search241_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search241_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search241_1_0.value, &search241_1_1.value },
   NULL,
};
static const nir_search_expression search241 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search241_0.value, &search241_1.value },
   NULL,
};
   
static const nir_search_constant replace241 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
   
static const nir_search_variable search242_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search242_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search242_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search242_0_0.value, &search242_0_1.value },
   NULL,
};

static const nir_search_variable search242_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search242 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search242_0.value, &search242_1.value },
   NULL,
};
   
static const nir_search_constant replace242 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
   
static const nir_search_variable search249_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search249_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search249_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search249_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search249_1_0.value, &search249_1_1.value },
   NULL,
};
static const nir_search_expression search249 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search249_0.value, &search249_1.value },
   NULL,
};
   
static const nir_search_variable replace249_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace249_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace249_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace249_0_0.value, &replace249_0_1.value },
   NULL,
};

static const nir_search_variable replace249_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace249_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace249_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace249_1_0.value, &replace249_1_1.value },
   NULL,
};
static const nir_search_expression replace249 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace249_0.value, &replace249_1.value },
   NULL,
};
   
static const nir_search_variable search250_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search250_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search250_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search250_0_0.value, &search250_0_1.value },
   NULL,
};

static const nir_search_variable search250_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search250 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search250_0.value, &search250_1.value },
   NULL,
};
   
static const nir_search_variable replace250_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace250_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace250_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace250_0_0.value, &replace250_0_1.value },
   NULL,
};

static const nir_search_variable replace250_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace250_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace250_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace250_1_0.value, &replace250_1_1.value },
   NULL,
};
static const nir_search_expression replace250 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace250_0.value, &replace250_1.value },
   NULL,
};
   
static const nir_search_variable search257_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search257_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search257_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search257_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search257_1_0.value, &search257_1_1.value },
   NULL,
};
static const nir_search_expression search257 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search257_0.value, &search257_1.value },
   NULL,
};
   
static const nir_search_variable replace257_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace257_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace257_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace257_0_0.value, &replace257_0_1.value },
   NULL,
};

static const nir_search_variable replace257_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace257_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace257_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace257_1_0.value, &replace257_1_1.value },
   NULL,
};
static const nir_search_expression replace257 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace257_0.value, &replace257_1.value },
   NULL,
};
   
static const nir_search_variable search258_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search258_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search258_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search258_0_0.value, &search258_0_1.value },
   NULL,
};

static const nir_search_variable search258_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search258 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search258_0.value, &search258_1.value },
   NULL,
};
   
static const nir_search_variable replace258_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace258_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace258_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace258_0_0.value, &replace258_0_1.value },
   NULL,
};

static const nir_search_variable replace258_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace258_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace258_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace258_1_0.value, &replace258_1_1.value },
   NULL,
};
static const nir_search_expression replace258 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace258_0.value, &replace258_1.value },
   NULL,
};
   
static const nir_search_variable search282_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search282_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search282 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search282_0.value, &search282_1.value },
   NULL,
};
   
static const nir_search_constant replace282 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
   
static const nir_search_variable search508_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search508_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search508_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search508_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search508_0_0.value, &search508_0_1.value, &search508_0_2.value },
   NULL,
};

static const nir_search_variable search508_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search508 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search508_0.value, &search508_1.value },
   NULL,
};
   
static const nir_search_variable replace508_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace508_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace508_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace508_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace508_1_0.value, &replace508_1_1.value },
   NULL,
};

static const nir_search_variable replace508_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace508_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace508_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace508_2_0.value, &replace508_2_1.value },
   NULL,
};
static const nir_search_expression replace508 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace508_0.value, &replace508_1.value, &replace508_2.value },
   NULL,
};
   
static const nir_search_variable search509_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search509_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search509_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search509_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search509_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search509_1_0.value, &search509_1_1.value, &search509_1_2.value },
   NULL,
};
static const nir_search_expression search509 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &search509_0.value, &search509_1.value },
   NULL,
};
   
static const nir_search_variable replace509_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace509_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace509_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace509_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace509_1_0.value, &replace509_1_1.value },
   NULL,
};

static const nir_search_variable replace509_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace509_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace509_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace509_2_0.value, &replace509_2_1.value },
   NULL,
};
static const nir_search_expression replace509 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace509_0.value, &replace509_1.value, &replace509_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ilt_xforms[] = {
   { &search233, &replace233.value, 0 },
   { &search234, &replace234.value, 0 },
   { &search241, &replace241.value, 0 },
   { &search242, &replace242.value, 0 },
   { &search249, &replace249.value, 0 },
   { &search250, &replace250.value, 0 },
   { &search257, &replace257.value, 0 },
   { &search258, &replace258.value, 0 },
   { &search282, &replace282.value, 0 },
   { &search508, &replace508.value, 0 },
   { &search509, &replace509.value, 0 },
};
   
static const nir_search_variable search235_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search235_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search235_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search235_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search235_1_0.value, &search235_1_1.value },
   NULL,
};
static const nir_search_expression search235 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search235_0.value, &search235_1.value },
   NULL,
};
   
static const nir_search_constant replace235 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search236_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search236_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search236_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search236_0_0.value, &search236_0_1.value },
   NULL,
};

static const nir_search_variable search236_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search236 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search236_0.value, &search236_1.value },
   NULL,
};
   
static const nir_search_constant replace236 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search243_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search243_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search243_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search243_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search243_1_0.value, &search243_1_1.value },
   NULL,
};
static const nir_search_expression search243 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search243_0.value, &search243_1.value },
   NULL,
};
   
static const nir_search_variable replace243_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace243_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace243 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace243_0.value, &replace243_1.value },
   NULL,
};
   
static const nir_search_variable search244_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search244_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search244_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search244_0_0.value, &search244_0_1.value },
   NULL,
};

static const nir_search_variable search244_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search244 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search244_0.value, &search244_1.value },
   NULL,
};
   
static const nir_search_variable replace244_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace244_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace244 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace244_0.value, &replace244_1.value },
   NULL,
};
   
static const nir_search_variable search251_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search251_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search251_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search251_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search251_1_0.value, &search251_1_1.value },
   NULL,
};
static const nir_search_expression search251 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search251_0.value, &search251_1.value },
   NULL,
};
   
static const nir_search_variable replace251_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace251_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace251_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace251_0_0.value, &replace251_0_1.value },
   NULL,
};

static const nir_search_variable replace251_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace251_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace251_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace251_1_0.value, &replace251_1_1.value },
   NULL,
};
static const nir_search_expression replace251 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace251_0.value, &replace251_1.value },
   NULL,
};
   
static const nir_search_variable search252_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search252_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search252_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search252_0_0.value, &search252_0_1.value },
   NULL,
};

static const nir_search_variable search252_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search252 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search252_0.value, &search252_1.value },
   NULL,
};
   
static const nir_search_variable replace252_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace252_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace252_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace252_0_0.value, &replace252_0_1.value },
   NULL,
};

static const nir_search_variable replace252_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace252_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace252_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace252_1_0.value, &replace252_1_1.value },
   NULL,
};
static const nir_search_expression replace252 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace252_0.value, &replace252_1.value },
   NULL,
};
   
static const nir_search_variable search259_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search259_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search259_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search259_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &search259_1_0.value, &search259_1_1.value },
   NULL,
};
static const nir_search_expression search259 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search259_0.value, &search259_1.value },
   NULL,
};
   
static const nir_search_variable replace259_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace259_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace259_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace259_0_0.value, &replace259_0_1.value },
   NULL,
};

static const nir_search_variable replace259_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace259_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace259_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace259_1_0.value, &replace259_1_1.value },
   NULL,
};
static const nir_search_expression replace259 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace259_0.value, &replace259_1.value },
   NULL,
};
   
static const nir_search_variable search260_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search260_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search260_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &search260_0_0.value, &search260_0_1.value },
   NULL,
};

static const nir_search_variable search260_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search260 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search260_0.value, &search260_1.value },
   NULL,
};
   
static const nir_search_variable replace260_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace260_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace260_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace260_0_0.value, &replace260_0_1.value },
   NULL,
};

static const nir_search_variable replace260_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace260_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace260_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace260_1_0.value, &replace260_1_1.value },
   NULL,
};
static const nir_search_expression replace260 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace260_0.value, &replace260_1.value },
   NULL,
};
   
static const nir_search_variable search283_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search283_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search283 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search283_0.value, &search283_1.value },
   NULL,
};
   
static const nir_search_constant replace283 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search510_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search510_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search510_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search510_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search510_0_0.value, &search510_0_1.value, &search510_0_2.value },
   NULL,
};

static const nir_search_variable search510_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search510 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search510_0.value, &search510_1.value },
   NULL,
};
   
static const nir_search_variable replace510_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace510_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace510_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace510_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace510_1_0.value, &replace510_1_1.value },
   NULL,
};

static const nir_search_variable replace510_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace510_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace510_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace510_2_0.value, &replace510_2_1.value },
   NULL,
};
static const nir_search_expression replace510 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace510_0.value, &replace510_1.value, &replace510_2.value },
   NULL,
};
   
static const nir_search_variable search511_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search511_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search511_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search511_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search511_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search511_1_0.value, &search511_1_1.value, &search511_1_2.value },
   NULL,
};
static const nir_search_expression search511 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &search511_0.value, &search511_1.value },
   NULL,
};
   
static const nir_search_variable replace511_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace511_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace511_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace511_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace511_1_0.value, &replace511_1_1.value },
   NULL,
};

static const nir_search_variable replace511_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace511_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace511_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ige,
   { &replace511_2_0.value, &replace511_2_1.value },
   NULL,
};
static const nir_search_expression replace511 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace511_0.value, &replace511_1.value, &replace511_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ige_xforms[] = {
   { &search235, &replace235.value, 0 },
   { &search236, &replace236.value, 0 },
   { &search243, &replace243.value, 0 },
   { &search244, &replace244.value, 0 },
   { &search251, &replace251.value, 0 },
   { &search252, &replace252.value, 0 },
   { &search259, &replace259.value, 0 },
   { &search260, &replace260.value, 0 },
   { &search283, &replace283.value, 0 },
   { &search510, &replace510.value, 0 },
   { &search511, &replace511.value, 0 },
};
   
static const nir_search_variable search237_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search237_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search237_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search237_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search237_1_0.value, &search237_1_1.value },
   NULL,
};
static const nir_search_expression search237 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search237_0.value, &search237_1.value },
   NULL,
};
   
static const nir_search_variable replace237_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace237_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace237 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace237_0.value, &replace237_1.value },
   NULL,
};
   
static const nir_search_variable search238_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search238_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search238_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search238_0_0.value, &search238_0_1.value },
   NULL,
};

static const nir_search_variable search238_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search238 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search238_0.value, &search238_1.value },
   NULL,
};
   
static const nir_search_variable replace238_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace238_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace238 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace238_0.value, &replace238_1.value },
   NULL,
};
   
static const nir_search_variable search245_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search245_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search245_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search245_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search245_1_0.value, &search245_1_1.value },
   NULL,
};
static const nir_search_expression search245 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search245_0.value, &search245_1.value },
   NULL,
};
   
static const nir_search_constant replace245 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
   
static const nir_search_variable search246_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search246_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search246_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search246_0_0.value, &search246_0_1.value },
   NULL,
};

static const nir_search_variable search246_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search246 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search246_0.value, &search246_1.value },
   NULL,
};
   
static const nir_search_constant replace246 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
   
static const nir_search_variable search253_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search253_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search253_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search253_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search253_1_0.value, &search253_1_1.value },
   NULL,
};
static const nir_search_expression search253 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search253_0.value, &search253_1.value },
   NULL,
};
   
static const nir_search_variable replace253_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace253_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace253_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace253_0_0.value, &replace253_0_1.value },
   NULL,
};

static const nir_search_variable replace253_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace253_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace253_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace253_1_0.value, &replace253_1_1.value },
   NULL,
};
static const nir_search_expression replace253 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace253_0.value, &replace253_1.value },
   NULL,
};
   
static const nir_search_variable search254_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search254_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search254_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search254_0_0.value, &search254_0_1.value },
   NULL,
};

static const nir_search_variable search254_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search254 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search254_0.value, &search254_1.value },
   NULL,
};
   
static const nir_search_variable replace254_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace254_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace254_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace254_0_0.value, &replace254_0_1.value },
   NULL,
};

static const nir_search_variable replace254_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace254_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace254_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace254_1_0.value, &replace254_1_1.value },
   NULL,
};
static const nir_search_expression replace254 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace254_0.value, &replace254_1.value },
   NULL,
};
   
static const nir_search_variable search261_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search261_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search261_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search261_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search261_1_0.value, &search261_1_1.value },
   NULL,
};
static const nir_search_expression search261 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search261_0.value, &search261_1.value },
   NULL,
};
   
static const nir_search_variable replace261_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace261_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace261_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace261_0_0.value, &replace261_0_1.value },
   NULL,
};

static const nir_search_variable replace261_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace261_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace261_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace261_1_0.value, &replace261_1_1.value },
   NULL,
};
static const nir_search_expression replace261 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace261_0.value, &replace261_1.value },
   NULL,
};
   
static const nir_search_variable search262_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search262_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search262_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search262_0_0.value, &search262_0_1.value },
   NULL,
};

static const nir_search_variable search262_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search262 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search262_0.value, &search262_1.value },
   NULL,
};
   
static const nir_search_variable replace262_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace262_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace262_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace262_0_0.value, &replace262_0_1.value },
   NULL,
};

static const nir_search_variable replace262_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace262_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace262_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace262_1_0.value, &replace262_1_1.value },
   NULL,
};
static const nir_search_expression replace262 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace262_0.value, &replace262_1.value },
   NULL,
};
   
static const nir_search_variable search286_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search286_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search286 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search286_0.value, &search286_1.value },
   NULL,
};
   
static const nir_search_constant replace286 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_FALSE /* False */ },
};
   
static const nir_search_variable search516_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search516_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search516_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search516_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search516_0_0.value, &search516_0_1.value, &search516_0_2.value },
   NULL,
};

static const nir_search_variable search516_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search516 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search516_0.value, &search516_1.value },
   NULL,
};
   
static const nir_search_variable replace516_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace516_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace516_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace516_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace516_1_0.value, &replace516_1_1.value },
   NULL,
};

static const nir_search_variable replace516_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace516_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace516_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace516_2_0.value, &replace516_2_1.value },
   NULL,
};
static const nir_search_expression replace516 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace516_0.value, &replace516_1.value, &replace516_2.value },
   NULL,
};
   
static const nir_search_variable search517_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search517_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search517_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search517_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search517_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search517_1_0.value, &search517_1_1.value, &search517_1_2.value },
   NULL,
};
static const nir_search_expression search517 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &search517_0.value, &search517_1.value },
   NULL,
};
   
static const nir_search_variable replace517_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace517_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace517_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace517_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace517_1_0.value, &replace517_1_1.value },
   NULL,
};

static const nir_search_variable replace517_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace517_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace517_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace517_2_0.value, &replace517_2_1.value },
   NULL,
};
static const nir_search_expression replace517 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace517_0.value, &replace517_1.value, &replace517_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ult_xforms[] = {
   { &search237, &replace237.value, 0 },
   { &search238, &replace238.value, 0 },
   { &search245, &replace245.value, 0 },
   { &search246, &replace246.value, 0 },
   { &search253, &replace253.value, 0 },
   { &search254, &replace254.value, 0 },
   { &search261, &replace261.value, 0 },
   { &search262, &replace262.value, 0 },
   { &search286, &replace286.value, 0 },
   { &search516, &replace516.value, 0 },
   { &search517, &replace517.value, 0 },
};
   
static const nir_search_variable search239_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search239_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search239_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search239_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search239_1_0.value, &search239_1_1.value },
   NULL,
};
static const nir_search_expression search239 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search239_0.value, &search239_1.value },
   NULL,
};
   
static const nir_search_constant replace239 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search240_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search240_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search240_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search240_0_0.value, &search240_0_1.value },
   NULL,
};

static const nir_search_variable search240_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search240 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search240_0.value, &search240_1.value },
   NULL,
};
   
static const nir_search_constant replace240 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search247_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search247_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search247_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search247_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search247_1_0.value, &search247_1_1.value },
   NULL,
};
static const nir_search_expression search247 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search247_0.value, &search247_1.value },
   NULL,
};
   
static const nir_search_variable replace247_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace247_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace247 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace247_0.value, &replace247_1.value },
   NULL,
};
   
static const nir_search_variable search248_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search248_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search248_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search248_0_0.value, &search248_0_1.value },
   NULL,
};

static const nir_search_variable search248_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search248 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search248_0.value, &search248_1.value },
   NULL,
};
   
static const nir_search_variable replace248_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace248_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace248 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace248_0.value, &replace248_1.value },
   NULL,
};
   
static const nir_search_variable search255_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search255_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search255_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search255_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search255_1_0.value, &search255_1_1.value },
   NULL,
};
static const nir_search_expression search255 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search255_0.value, &search255_1.value },
   NULL,
};
   
static const nir_search_variable replace255_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace255_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace255_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace255_0_0.value, &replace255_0_1.value },
   NULL,
};

static const nir_search_variable replace255_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace255_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace255_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace255_1_0.value, &replace255_1_1.value },
   NULL,
};
static const nir_search_expression replace255 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace255_0.value, &replace255_1.value },
   NULL,
};
   
static const nir_search_variable search256_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search256_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search256_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search256_0_0.value, &search256_0_1.value },
   NULL,
};

static const nir_search_variable search256_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search256 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search256_0.value, &search256_1.value },
   NULL,
};
   
static const nir_search_variable replace256_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace256_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace256_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace256_0_0.value, &replace256_0_1.value },
   NULL,
};

static const nir_search_variable replace256_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace256_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace256_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace256_1_0.value, &replace256_1_1.value },
   NULL,
};
static const nir_search_expression replace256 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace256_0.value, &replace256_1.value },
   NULL,
};
   
static const nir_search_variable search263_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search263_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search263_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search263_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umax,
   { &search263_1_0.value, &search263_1_1.value },
   NULL,
};
static const nir_search_expression search263 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search263_0.value, &search263_1.value },
   NULL,
};
   
static const nir_search_variable replace263_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace263_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace263_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace263_0_0.value, &replace263_0_1.value },
   NULL,
};

static const nir_search_variable replace263_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace263_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace263_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace263_1_0.value, &replace263_1_1.value },
   NULL,
};
static const nir_search_expression replace263 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace263_0.value, &replace263_1.value },
   NULL,
};
   
static const nir_search_variable search264_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search264_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search264_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_umin,
   { &search264_0_0.value, &search264_0_1.value },
   NULL,
};

static const nir_search_variable search264_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search264 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search264_0.value, &search264_1.value },
   NULL,
};
   
static const nir_search_variable replace264_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace264_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace264_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace264_0_0.value, &replace264_0_1.value },
   NULL,
};

static const nir_search_variable replace264_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace264_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace264_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace264_1_0.value, &replace264_1_1.value },
   NULL,
};
static const nir_search_expression replace264 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace264_0.value, &replace264_1.value },
   NULL,
};
   
static const nir_search_variable search287_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search287_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search287 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search287_0.value, &search287_1.value },
   NULL,
};
   
static const nir_search_constant replace287 = {
   { nir_search_value_constant, 32 },
   nir_type_bool, { NIR_TRUE /* True */ },
};
   
static const nir_search_variable search518_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search518_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search518_0_2 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search518_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search518_0_0.value, &search518_0_1.value, &search518_0_2.value },
   NULL,
};

static const nir_search_variable search518_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search518 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search518_0.value, &search518_1.value },
   NULL,
};
   
static const nir_search_variable replace518_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace518_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace518_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace518_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace518_1_0.value, &replace518_1_1.value },
   NULL,
};

static const nir_search_variable replace518_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace518_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace518_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace518_2_0.value, &replace518_2_1.value },
   NULL,
};
static const nir_search_expression replace518 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace518_0.value, &replace518_1.value, &replace518_2.value },
   NULL,
};
   
static const nir_search_variable search519_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search519_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search519_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search519_1_2 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search519_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &search519_1_0.value, &search519_1_1.value, &search519_1_2.value },
   NULL,
};
static const nir_search_expression search519 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &search519_0.value, &search519_1.value },
   NULL,
};
   
static const nir_search_variable replace519_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace519_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace519_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace519_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace519_1_0.value, &replace519_1_1.value },
   NULL,
};

static const nir_search_variable replace519_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* d */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace519_2_1 = {
   { nir_search_value_variable, 0 },
   3, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace519_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_uge,
   { &replace519_2_0.value, &replace519_2_1.value },
   NULL,
};
static const nir_search_expression replace519 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace519_0.value, &replace519_1.value, &replace519_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_uge_xforms[] = {
   { &search239, &replace239.value, 0 },
   { &search240, &replace240.value, 0 },
   { &search247, &replace247.value, 0 },
   { &search248, &replace248.value, 0 },
   { &search255, &replace255.value, 0 },
   { &search256, &replace256.value, 0 },
   { &search263, &replace263.value, 0 },
   { &search264, &replace264.value, 0 },
   { &search287, &replace287.value, 0 },
   { &search518, &replace518.value, 0 },
   { &search519, &replace519.value, 0 },
};
   
static const nir_search_variable search269_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search269_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search269 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_slt,
   { &search269_0.value, &search269_1.value },
   NULL,
};
   
static const nir_search_variable replace269_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace269_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace269_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace269_0_0.value, &replace269_0_1.value },
   NULL,
};
static const nir_search_expression replace269 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace269_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_slt_xforms[] = {
   { &search269, &replace269.value, 11 },
};
   
static const nir_search_variable search270_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search270_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search270 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_sge,
   { &search270_0.value, &search270_1.value },
   NULL,
};
   
static const nir_search_variable replace270_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace270_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace270_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace270_0_0.value, &replace270_0_1.value },
   NULL,
};
static const nir_search_expression replace270 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace270_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_sge_xforms[] = {
   { &search270, &replace270.value, 11 },
};
   
static const nir_search_variable search271_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search271_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search271 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_seq,
   { &search271_0.value, &search271_1.value },
   NULL,
};
   
static const nir_search_variable replace271_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace271_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace271_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace271_0_0.value, &replace271_0_1.value },
   NULL,
};
static const nir_search_expression replace271 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace271_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_seq_xforms[] = {
   { &search271, &replace271.value, 11 },
};
   
static const nir_search_variable search272_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search272_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search272 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_sne,
   { &search272_0.value, &search272_1.value },
   NULL,
};
   
static const nir_search_variable replace272_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace272_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace272_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace272_0_0.value, &replace272_0_1.value },
   NULL,
};
static const nir_search_expression replace272 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace272_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_sne_xforms[] = {
   { &search272, &replace272.value, 11 },
};
   
static const nir_search_variable search288_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search288_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search288 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fand,
   { &search288_0.value, &search288_1.value },
   NULL,
};
   
static const nir_search_constant replace288 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const struct transform nir_opt_algebraic_fand_xforms[] = {
   { &search288, &replace288.value, 0 },
};
   
static const nir_search_variable search295_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search295_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search295 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fxor,
   { &search295_0.value, &search295_1.value },
   NULL,
};
   
static const nir_search_constant replace295 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const struct transform nir_opt_algebraic_fxor_xforms[] = {
   { &search295, &replace295.value, 0 },
};
   
static const nir_search_variable search296_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search296_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search296 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ixor,
   { &search296_0.value, &search296_1.value },
   NULL,
};
   
static const nir_search_constant replace296 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
   
static const nir_search_variable search297_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search297_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search297 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ixor,
   { &search297_0.value, &search297_1.value },
   NULL,
};
   
static const nir_search_variable replace297 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_ixor_xforms[] = {
   { &search296, &replace296.value, 0 },
   { &search297, &replace297.value, 0 },
};
   
static const nir_search_constant search307_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable search307_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search307 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &search307_0.value, &search307_1.value },
   NULL,
};
   
static const nir_search_constant replace307 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
   
static const nir_search_variable search308_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search308_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search308 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &search308_0.value, &search308_1.value },
   NULL,
};
   
static const nir_search_variable replace308 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search376_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search376_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search376_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search376_0_0.value, &search376_0_1.value },
   NULL,
};

static const nir_search_constant search376_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search376 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &search376_0.value, &search376_1.value },
   NULL,
};
   
static const nir_search_variable replace376_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace376_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace376 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i8,
   { &replace376_0.value, &replace376_1.value },
   NULL,
};
   
static const nir_search_variable search377_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search377_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search377_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search377_0_0.value, &search377_0_1.value },
   NULL,
};

static const nir_search_constant search377_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search377 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &search377_0.value, &search377_1.value },
   NULL,
};
   
static const nir_search_variable replace377_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace377_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace377 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i8,
   { &replace377_0.value, &replace377_1.value },
   NULL,
};
   
static const nir_search_variable search378_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search378_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search378_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search378_0_0.value, &search378_0_1.value },
   NULL,
};

static const nir_search_constant search378_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search378 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &search378_0.value, &search378_1.value },
   NULL,
};
   
static const nir_search_variable replace378_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace378_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x2 /* 2 */ },
};
static const nir_search_expression replace378 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i8,
   { &replace378_0.value, &replace378_1.value },
   NULL,
};
   
static const nir_search_variable search379_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search379_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search379 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &search379_0.value, &search379_1.value },
   NULL,
};
   
static const nir_search_variable replace379_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace379_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x3 /* 3 */ },
};
static const nir_search_expression replace379 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i8,
   { &replace379_0.value, &replace379_1.value },
   NULL,
};
   
static const nir_search_variable search385_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search385_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search385_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search385_0_0.value, &search385_0_1.value },
   NULL,
};

static const nir_search_constant search385_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search385 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &search385_0.value, &search385_1.value },
   NULL,
};
   
static const nir_search_variable replace385_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace385_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace385 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i16,
   { &replace385_0.value, &replace385_1.value },
   NULL,
};
   
static const nir_search_variable search386_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search386_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search386 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &search386_0.value, &search386_1.value },
   NULL,
};
   
static const nir_search_variable replace386_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace386_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace386 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i16,
   { &replace386_0.value, &replace386_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ishr_xforms[] = {
   { &search307, &replace307.value, 0 },
   { &search308, &replace308.value, 0 },
   { &search376, &replace376.value, 18 },
   { &search377, &replace377.value, 18 },
   { &search378, &replace378.value, 18 },
   { &search379, &replace379.value, 18 },
   { &search385, &replace385.value, 19 },
   { &search386, &replace386.value, 19 },
};
   
static const nir_search_constant search309_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable search309_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search309 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search309_0.value, &search309_1.value },
   NULL,
};
   
static const nir_search_constant replace309 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
   
static const nir_search_variable search310_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search310_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search310 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search310_0.value, &search310_1.value },
   NULL,
};
   
static const nir_search_variable replace310 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search372_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search372_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search372_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search372_0_0.value, &search372_0_1.value },
   NULL,
};

static const nir_search_constant search372_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search372 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search372_0.value, &search372_1.value },
   NULL,
};
   
static const nir_search_variable replace372_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace372_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace372 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace372_0.value, &replace372_1.value },
   NULL,
};
   
static const nir_search_variable search373_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search373_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search373_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search373_0_0.value, &search373_0_1.value },
   NULL,
};

static const nir_search_constant search373_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search373 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search373_0.value, &search373_1.value },
   NULL,
};
   
static const nir_search_variable replace373_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace373_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace373 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace373_0.value, &replace373_1.value },
   NULL,
};
   
static const nir_search_variable search374_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search374_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression search374_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search374_0_0.value, &search374_0_1.value },
   NULL,
};

static const nir_search_constant search374_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search374 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search374_0.value, &search374_1.value },
   NULL,
};
   
static const nir_search_variable replace374_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace374_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x2 /* 2 */ },
};
static const nir_search_expression replace374 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace374_0.value, &replace374_1.value },
   NULL,
};
   
static const nir_search_variable search375_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search375_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression search375 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search375_0.value, &search375_1.value },
   NULL,
};
   
static const nir_search_variable replace375_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace375_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x3 /* 3 */ },
};
static const nir_search_expression replace375 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace375_0.value, &replace375_1.value },
   NULL,
};
   
static const nir_search_variable search383_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search383_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search383_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &search383_0_0.value, &search383_0_1.value },
   NULL,
};

static const nir_search_constant search383_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search383 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search383_0.value, &search383_1.value },
   NULL,
};
   
static const nir_search_variable replace383_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace383_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace383 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u16,
   { &replace383_0.value, &replace383_1.value },
   NULL,
};
   
static const nir_search_variable search384_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search384_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression search384 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &search384_0.value, &search384_1.value },
   NULL,
};
   
static const nir_search_variable replace384_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace384_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace384 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u16,
   { &replace384_0.value, &replace384_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ushr_xforms[] = {
   { &search309, &replace309.value, 0 },
   { &search310, &replace310.value, 0 },
   { &search372, &replace372.value, 18 },
   { &search373, &replace373.value, 18 },
   { &search374, &replace374.value, 18 },
   { &search375, &replace375.value, 18 },
   { &search383, &replace383.value, 19 },
   { &search384, &replace384.value, 19 },
};
   
static const nir_search_variable search313_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search313_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &search313_0_0.value },
   NULL,
};
static const nir_search_expression search313 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fexp2,
   { &search313_0.value },
   NULL,
};
   
static const nir_search_variable replace313 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search316_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search316_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &search316_0_0_0.value },
   NULL,
};

static const nir_search_variable search316_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search316_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search316_0_0.value, &search316_0_1.value },
   NULL,
};
static const nir_search_expression search316 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fexp2,
   { &search316_0.value },
   NULL,
};
   
static const nir_search_variable replace316_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace316_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace316 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fpow,
   { &replace316_0.value, &replace316_1.value },
   NULL,
};
   
static const nir_search_variable search317_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search317_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &search317_0_0_0_0.value },
   NULL,
};

static const nir_search_variable search317_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search317_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search317_0_0_0.value, &search317_0_0_1.value },
   NULL,
};

static const nir_search_variable search317_0_1_0_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search317_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &search317_0_1_0_0.value },
   NULL,
};

static const nir_search_variable search317_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search317_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search317_0_1_0.value, &search317_0_1_1.value },
   NULL,
};
static const nir_search_expression search317_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search317_0_0.value, &search317_0_1.value },
   NULL,
};
static const nir_search_expression search317 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fexp2,
   { &search317_0.value },
   NULL,
};
   
static const nir_search_variable replace317_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace317_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace317_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fpow,
   { &replace317_0_0.value, &replace317_0_1.value },
   NULL,
};

static const nir_search_variable replace317_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace317_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace317_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fpow,
   { &replace317_1_0.value, &replace317_1_1.value },
   NULL,
};
static const nir_search_expression replace317 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fmul,
   { &replace317_0.value, &replace317_1.value },
   NULL,
};
   
static const nir_search_variable search318_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search318_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &search318_0_0_0.value },
   NULL,
};

static const nir_search_constant search318_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x4000000000000000 /* 2.0 */ },
};
static const nir_search_expression search318_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search318_0_0.value, &search318_0_1.value },
   NULL,
};
static const nir_search_expression search318 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fexp2,
   { &search318_0.value },
   NULL,
};
   
static const nir_search_variable replace318_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace318_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace318 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace318_0.value, &replace318_1.value },
   NULL,
};
   
static const nir_search_variable search319_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search319_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &search319_0_0_0.value },
   NULL,
};

static const nir_search_constant search319_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x4010000000000000 /* 4.0 */ },
};
static const nir_search_expression search319_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search319_0_0.value, &search319_0_1.value },
   NULL,
};
static const nir_search_expression search319 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fexp2,
   { &search319_0.value },
   NULL,
};
   
static const nir_search_variable replace319_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace319_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace319_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace319_0_0.value, &replace319_0_1.value },
   NULL,
};

static const nir_search_variable replace319_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace319_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace319_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace319_1_0.value, &replace319_1_1.value },
   NULL,
};
static const nir_search_expression replace319 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace319_0.value, &replace319_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fexp2_xforms[] = {
   { &search313, &replace313.value, 0 },
   { &search316, &replace316.value, 14 },
   { &search317, &replace317.value, 14 },
   { &search318, &replace318.value, 0 },
   { &search319, &replace319.value, 0 },
};
   
static const nir_search_variable search314_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search314_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &search314_0_0.value },
   NULL,
};
static const nir_search_expression search314 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flog2,
   { &search314_0.value },
   NULL,
};
   
static const nir_search_variable replace314 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search329_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search329_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsqrt,
   { &search329_0_0.value },
   NULL,
};
static const nir_search_expression search329 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flog2,
   { &search329_0.value },
   NULL,
};
   
static const nir_search_constant replace329_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3fe0000000000000 /* 0.5 */ },
};

static const nir_search_variable replace329_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace329_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &replace329_1_0.value },
   NULL,
};
static const nir_search_expression replace329 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace329_0.value, &replace329_1.value },
   NULL,
};
   
static const nir_search_variable search330_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search330_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_frcp,
   { &search330_0_0.value },
   NULL,
};
static const nir_search_expression search330 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flog2,
   { &search330_0.value },
   NULL,
};
   
static const nir_search_variable replace330_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace330_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &replace330_0_0.value },
   NULL,
};
static const nir_search_expression replace330 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace330_0.value },
   NULL,
};
   
static const nir_search_variable search331_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search331_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_frsq,
   { &search331_0_0.value },
   NULL,
};
static const nir_search_expression search331 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flog2,
   { &search331_0.value },
   NULL,
};
   
static const nir_search_constant replace331_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0xbfe0000000000000L /* -0.5 */ },
};

static const nir_search_variable replace331_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace331_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &replace331_1_0.value },
   NULL,
};
static const nir_search_expression replace331 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace331_0.value, &replace331_1.value },
   NULL,
};
   
static const nir_search_variable search332_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search332_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search332_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fpow,
   { &search332_0_0.value, &search332_0_1.value },
   NULL,
};
static const nir_search_expression search332 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_flog2,
   { &search332_0.value },
   NULL,
};
   
static const nir_search_variable replace332_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace332_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace332_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &replace332_1_0.value },
   NULL,
};
static const nir_search_expression replace332 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace332_0.value, &replace332_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_flog2_xforms[] = {
   { &search314, &replace314.value, 0 },
   { &search329, &replace329.value, 0 },
   { &search330, &replace330.value, 0 },
   { &search331, &replace331.value, 0 },
   { &search332, &replace332.value, 0 },
};
   
static const nir_search_variable search315_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search315_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search315 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fpow,
   { &search315_0.value, &search315_1.value },
   NULL,
};
   
static const nir_search_variable replace315_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace315_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flog2,
   { &replace315_0_0_0.value },
   NULL,
};

static const nir_search_variable replace315_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace315_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace315_0_0.value, &replace315_0_1.value },
   NULL,
};
static const nir_search_expression replace315 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &replace315_0.value },
   NULL,
};
   
static const nir_search_variable search320_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search320_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression search320 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fpow,
   { &search320_0.value, &search320_1.value },
   NULL,
};
   
static const nir_search_variable replace320 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search321_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search321_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x4000000000000000 /* 2.0 */ },
};
static const nir_search_expression search321 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fpow,
   { &search321_0.value, &search321_1.value },
   NULL,
};
   
static const nir_search_variable replace321_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace321_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace321 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace321_0.value, &replace321_1.value },
   NULL,
};
   
static const nir_search_variable search322_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search322_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x4010000000000000 /* 4.0 */ },
};
static const nir_search_expression search322 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fpow,
   { &search322_0.value, &search322_1.value },
   NULL,
};
   
static const nir_search_variable replace322_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace322_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace322_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace322_0_0.value, &replace322_0_1.value },
   NULL,
};

static const nir_search_variable replace322_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace322_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace322_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace322_1_0.value, &replace322_1_1.value },
   NULL,
};
static const nir_search_expression replace322 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace322_0.value, &replace322_1.value },
   NULL,
};
   
static const nir_search_constant search323_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x4000000000000000 /* 2.0 */ },
};

static const nir_search_variable search323_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search323 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fpow,
   { &search323_0.value, &search323_1.value },
   NULL,
};
   
static const nir_search_variable replace323_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace323 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &replace323_0.value },
   NULL,
};
   
static const nir_search_variable search324_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search324_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x400199999999999a /* 2.2 */ },
};
static const nir_search_expression search324_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fpow,
   { &search324_0_0.value, &search324_0_1.value },
   NULL,
};

static const nir_search_constant search324_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3fdd1743e963dc48 /* 0.454545 */ },
};
static const nir_search_expression search324 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fpow,
   { &search324_0.value, &search324_1.value },
   NULL,
};
   
static const nir_search_variable replace324 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search325_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search325_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x400199999999999a /* 2.2 */ },
};
static const nir_search_expression search325_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fpow,
   { &search325_0_0_0.value, &search325_0_0_1.value },
   NULL,
};
static const nir_search_expression search325_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &search325_0_0.value },
   NULL,
};

static const nir_search_constant search325_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3fdd1743e963dc48 /* 0.454545 */ },
};
static const nir_search_expression search325 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fpow,
   { &search325_0.value, &search325_1.value },
   NULL,
};
   
static const nir_search_variable replace325_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace325 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fabs,
   { &replace325_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fpow_xforms[] = {
   { &search315, &replace315.value, 13 },
   { &search320, &replace320.value, 0 },
   { &search321, &replace321.value, 0 },
   { &search322, &replace322.value, 0 },
   { &search323, &replace323.value, 0 },
   { &search324, &replace324.value, 0 },
   { &search325, &replace325.value, 0 },
};
   
static const nir_search_variable search326_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search326_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &search326_0_0.value },
   NULL,
};
static const nir_search_expression search326 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fsqrt,
   { &search326_0.value },
   NULL,
};
   
static const nir_search_constant replace326_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3fe0000000000000 /* 0.5 */ },
};

static const nir_search_variable replace326_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace326_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace326_0_0.value, &replace326_0_1.value },
   NULL,
};
static const nir_search_expression replace326 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &replace326_0.value },
   NULL,
};
   
static const nir_search_variable search338_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search338 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsqrt,
   { &search338_0.value },
   NULL,
};
   
static const nir_search_variable replace338_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace338_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_frsq,
   { &replace338_0_0.value },
   NULL,
};
static const nir_search_expression replace338 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_frcp,
   { &replace338_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fsqrt_xforms[] = {
   { &search326, &replace326.value, 0 },
   { &search338, &replace338.value, 16 },
};
   
static const nir_search_variable search327_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search327_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &search327_0_0.value },
   NULL,
};
static const nir_search_expression search327 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_frcp,
   { &search327_0.value },
   NULL,
};
   
static const nir_search_variable replace327_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace327_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace327_0_0.value },
   NULL,
};
static const nir_search_expression replace327 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &replace327_0.value },
   NULL,
};
   
static const nir_search_variable search336_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search336_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_frcp,
   { &search336_0_0.value },
   NULL,
};
static const nir_search_expression search336 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_frcp,
   { &search336_0.value },
   NULL,
};
   
static const nir_search_variable replace336 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search337_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search337_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsqrt,
   { &search337_0_0.value },
   NULL,
};
static const nir_search_expression search337 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_frcp,
   { &search337_0.value },
   NULL,
};
   
static const nir_search_variable replace337_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace337 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_frsq,
   { &replace337_0.value },
   NULL,
};
   
static const nir_search_variable search339_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search339_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_frsq,
   { &search339_0_0.value },
   NULL,
};
static const nir_search_expression search339 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_frcp,
   { &search339_0.value },
   NULL,
};
   
static const nir_search_variable replace339_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace339 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsqrt,
   { &replace339_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_frcp_xforms[] = {
   { &search327, &replace327.value, 0 },
   { &search336, &replace336.value, 0 },
   { &search337, &replace337.value, 0 },
   { &search339, &replace339.value, 17 },
};
   
static const nir_search_variable search328_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search328_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &search328_0_0.value },
   NULL,
};
static const nir_search_expression search328 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_frsq,
   { &search328_0.value },
   NULL,
};
   
static const nir_search_constant replace328_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0xbfe0000000000000L /* -0.5 */ },
};

static const nir_search_variable replace328_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace328_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace328_0_0.value, &replace328_0_1.value },
   NULL,
};
static const nir_search_expression replace328 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fexp2,
   { &replace328_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_frsq_xforms[] = {
   { &search328, &replace328.value, 0 },
};
   
static const nir_search_constant search334_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_variable search334_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search334 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fdiv,
   { &search334_0.value, &search334_1.value },
   NULL,
};
   
static const nir_search_variable replace334_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace334 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_frcp,
   { &replace334_0.value },
   NULL,
};
   
static const nir_search_variable search335_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search335_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search335 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdiv,
   { &search335_0.value, &search335_1.value },
   NULL,
};
   
static const nir_search_variable replace335_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace335_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace335_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_frcp,
   { &replace335_1_0.value },
   NULL,
};
static const nir_search_expression replace335 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace335_0.value, &replace335_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fdiv_xforms[] = {
   { &search334, &replace334.value, 0 },
   { &search335, &replace335.value, 15 },
};
   
static const nir_search_variable search355_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search355_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search355_2 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search355 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fcsel,
   { &search355_0.value, &search355_1.value, &search355_2.value },
   NULL,
};
   
static const nir_search_variable replace355 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_fcsel_xforms[] = {
   { &search355, &replace355.value, 0 },
};
   
static const nir_search_variable search356_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search356_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search356_0_0.value },
   NULL,
};
static const nir_search_expression search356 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2b,
   { &search356_0.value },
   NULL,
};
   
static const nir_search_variable replace356 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search357_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_bool,
   NULL,
};
static const nir_search_expression search357 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2b,
   { &search357_0.value },
   NULL,
};
   
static const nir_search_variable replace357 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search360_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search360_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search360_0_0.value },
   NULL,
};
static const nir_search_expression search360 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2b,
   { &search360_0.value },
   NULL,
};
   
static const nir_search_variable replace360_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace360 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2b,
   { &replace360_0.value },
   NULL,
};
   
static const nir_search_variable search361_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search361_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iabs,
   { &search361_0_0.value },
   NULL,
};
static const nir_search_expression search361 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2b,
   { &search361_0.value },
   NULL,
};
   
static const nir_search_variable replace361_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace361 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2b,
   { &replace361_0.value },
   NULL,
};
   
static const nir_search_variable search465_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search465_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search465_0_0.value },
   NULL,
};
static const nir_search_expression search465 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2b,
   { &search465_0.value },
   NULL,
};
   
static const nir_search_variable replace465 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_i2b_xforms[] = {
   { &search356, &replace356.value, 0 },
   { &search357, &replace357.value, 0 },
   { &search360, &replace360.value, 0 },
   { &search361, &replace361.value, 0 },
   { &search465, &replace465.value, 0 },
};
   
static const nir_search_variable search358_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search358_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ftrunc,
   { &search358_0_0.value },
   NULL,
};
static const nir_search_expression search358 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2i32,
   { &search358_0.value },
   NULL,
};
   
static const nir_search_variable replace358_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace358 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2i32,
   { &replace358_0.value },
   NULL,
};
   
static const nir_search_variable search365_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search365_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &search365_0_0.value },
   NULL,
};
static const nir_search_expression search365 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_f2i32,
   { &search365_0.value },
   NULL,
};
   
static const nir_search_variable replace365 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search366_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search366_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &search366_0_0.value },
   NULL,
};
static const nir_search_expression search366 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_f2i32,
   { &search366_0.value },
   NULL,
};
   
static const nir_search_variable replace366 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search475_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search475_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search475_0_0.value },
   NULL,
};
static const nir_search_expression search475 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2i32,
   { &search475_0.value },
   NULL,
};
   
static const nir_search_variable replace475_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace475 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace475_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2i32_xforms[] = {
   { &search358, &replace358.value, 0 },
   { &search365, &replace365.value, 0 },
   { &search366, &replace366.value, 0 },
   { &search475, &replace475.value, 0 },
};
   
static const nir_search_variable search359_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search359_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ftrunc,
   { &search359_0_0.value },
   NULL,
};
static const nir_search_expression search359 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2u32,
   { &search359_0.value },
   NULL,
};
   
static const nir_search_variable replace359_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace359 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2u32,
   { &replace359_0.value },
   NULL,
};
   
static const nir_search_variable search367_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search367_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &search367_0_0.value },
   NULL,
};
static const nir_search_expression search367 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_f2u32,
   { &search367_0.value },
   NULL,
};
   
static const nir_search_variable replace367 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search368_0_0 = {
   { nir_search_value_variable, 32 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search368_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &search368_0_0.value },
   NULL,
};
static const nir_search_expression search368 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_f2u32,
   { &search368_0.value },
   NULL,
};
   
static const nir_search_variable replace368 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search471_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search471_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search471_0_0.value },
   NULL,
};
static const nir_search_expression search471 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2u32,
   { &search471_0.value },
   NULL,
};
   
static const nir_search_variable replace471_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace471 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace471_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2u32_xforms[] = {
   { &search359, &replace359.value, 0 },
   { &search367, &replace367.value, 0 },
   { &search368, &replace368.value, 0 },
   { &search471, &replace471.value, 0 },
};
   
static const nir_search_variable search369_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search369_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search369_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_64_2x32_split,
   { &search369_0_0.value, &search369_0_1.value },
   NULL,
};
static const nir_search_expression search369 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_unpack_64_2x32_split_x,
   { &search369_0.value },
   NULL,
};
   
static const nir_search_variable replace369 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_unpack_64_2x32_split_x_xforms[] = {
   { &search369, &replace369.value, 0 },
};
   
static const nir_search_variable search370_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search370_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search370_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_64_2x32_split,
   { &search370_0_0.value, &search370_0_1.value },
   NULL,
};
static const nir_search_expression search370 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_unpack_64_2x32_split_y,
   { &search370_0.value },
   NULL,
};
   
static const nir_search_variable replace370 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_unpack_64_2x32_split_y_xforms[] = {
   { &search370, &replace370.value, 0 },
};
   
static const nir_search_variable search371_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search371_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_unpack_64_2x32_split_x,
   { &search371_0_0.value },
   NULL,
};

static const nir_search_variable search371_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search371_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_unpack_64_2x32_split_y,
   { &search371_1_0.value },
   NULL,
};
static const nir_search_expression search371 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_64_2x32_split,
   { &search371_0.value, &search371_1.value },
   NULL,
};
   
static const nir_search_variable replace371 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_pack_64_2x32_split_xforms[] = {
   { &search371, &replace371.value, 0 },
};
   
static const nir_search_variable search388_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search388_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_variable search388_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search388_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &search388_1_0.value, &search388_1_1.value },
   NULL,
};
static const nir_search_expression search388 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fsub,
   { &search388_0.value, &search388_1.value },
   NULL,
};
   
static const nir_search_variable replace388_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace388_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace388 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace388_0.value, &replace388_1.value },
   NULL,
};
   
static const nir_search_variable search392_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search392_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search392 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &search392_0.value, &search392_1.value },
   NULL,
};
   
static const nir_search_variable replace392_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace392_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace392_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace392_1_0.value },
   NULL,
};
static const nir_search_expression replace392 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace392_0.value, &replace392_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fsub_xforms[] = {
   { &search388, &replace388.value, 0 },
   { &search392, &replace392.value, 20 },
};
   
static const nir_search_variable search389_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search389_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable search389_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search389_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &search389_1_0.value, &search389_1_1.value },
   NULL,
};
static const nir_search_expression search389 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &search389_0.value, &search389_1.value },
   NULL,
};
   
static const nir_search_variable replace389_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace389_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace389 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace389_0.value, &replace389_1.value },
   NULL,
};
   
static const nir_search_variable search393_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search393_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search393 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &search393_0.value, &search393_1.value },
   NULL,
};
   
static const nir_search_variable replace393_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace393_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace393_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &replace393_1_0.value },
   NULL,
};
static const nir_search_expression replace393 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace393_0.value, &replace393_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_isub_xforms[] = {
   { &search389, &replace389.value, 0 },
   { &search393, &replace393.value, 20 },
};
   
static const nir_search_variable search390_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search390_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression search390 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ussub_4x8,
   { &search390_0.value, &search390_1.value },
   NULL,
};
   
static const nir_search_variable replace390 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search391_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant search391_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x1 /* -1 */ },
};
static const nir_search_expression search391 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ussub_4x8,
   { &search391_0.value, &search391_1.value },
   NULL,
};
   
static const nir_search_constant replace391 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const struct transform nir_opt_algebraic_ussub_4x8_xforms[] = {
   { &search390, &replace390.value, 0 },
   { &search391, &replace391.value, 0 },
};
   
static const nir_search_variable search418_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search418_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search418 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_fmod,
   { &search418_0.value, &search418_1.value },
   NULL,
};
   
static const nir_search_variable replace418_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace418_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace418_1_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace418_1_1_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace418_1_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdiv,
   { &replace418_1_1_0_0.value, &replace418_1_1_0_1.value },
   NULL,
};
static const nir_search_expression replace418_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ffloor,
   { &replace418_1_1_0.value },
   NULL,
};
static const nir_search_expression replace418_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace418_1_0.value, &replace418_1_1.value },
   NULL,
};
static const nir_search_expression replace418 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &replace418_0.value, &replace418_1.value },
   NULL,
};
   
static const nir_search_variable search419_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search419_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search419 = {
   { nir_search_value_expression, 64 },
   false,
   nir_op_fmod,
   { &search419_0.value, &search419_1.value },
   NULL,
};
   
static const nir_search_variable replace419_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace419_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace419_1_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace419_1_1_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace419_1_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdiv,
   { &replace419_1_1_0_0.value, &replace419_1_1_0_1.value },
   NULL,
};
static const nir_search_expression replace419_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ffloor,
   { &replace419_1_1_0.value },
   NULL,
};
static const nir_search_expression replace419_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace419_1_0.value, &replace419_1_1.value },
   NULL,
};
static const nir_search_expression replace419 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &replace419_0.value, &replace419_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_fmod_xforms[] = {
   { &search418, &replace418.value, 22 },
   { &search419, &replace419.value, 23 },
};
   
static const nir_search_variable search420_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search420_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search420 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_frem,
   { &search420_0.value, &search420_1.value },
   NULL,
};
   
static const nir_search_variable replace420_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace420_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace420_1_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace420_1_1_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace420_1_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdiv,
   { &replace420_1_1_0_0.value, &replace420_1_1_0_1.value },
   NULL,
};
static const nir_search_expression replace420_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ftrunc,
   { &replace420_1_1_0.value },
   NULL,
};
static const nir_search_expression replace420_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace420_1_0.value, &replace420_1_1.value },
   NULL,
};
static const nir_search_expression replace420 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsub,
   { &replace420_0.value, &replace420_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_frem_xforms[] = {
   { &search420, &replace420.value, 22 },
};
   
static const nir_search_variable search421_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search421_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search421 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_uadd_carry,
   { &search421_0.value, &search421_1.value },
   NULL,
};
   
static const nir_search_variable replace421_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace421_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace421_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace421_0_0_0.value, &replace421_0_0_1.value },
   NULL,
};

static const nir_search_variable replace421_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace421_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace421_0_0.value, &replace421_0_1.value },
   NULL,
};
static const nir_search_expression replace421 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace421_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_uadd_carry_xforms[] = {
   { &search421, &replace421.value, 24 },
};
   
static const nir_search_variable search422_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search422_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search422 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_usub_borrow,
   { &search422_0.value, &search422_1.value },
   NULL,
};
   
static const nir_search_variable replace422_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace422_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace422_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace422_0_0.value, &replace422_0_1.value },
   NULL,
};
static const nir_search_expression replace422 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace422_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_usub_borrow_xforms[] = {
   { &search422, &replace422.value, 25 },
};
   
static const nir_search_variable search423_0 = {
   { nir_search_value_variable, 0 },
   0, /* base */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search423_1 = {
   { nir_search_value_variable, 0 },
   1, /* insert */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search423_2 = {
   { nir_search_value_variable, 0 },
   2, /* offset */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search423_3 = {
   { nir_search_value_variable, 0 },
   3, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search423 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bitfield_insert,
   { &search423_0.value, &search423_1.value, &search423_2.value, &search423_3.value },
   NULL,
};
   
static const nir_search_constant replace423_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1f /* 31 */ },
};

static const nir_search_variable replace423_0_1 = {
   { nir_search_value_variable, 0 },
   3, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace423_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace423_0_0.value, &replace423_0_1.value },
   NULL,
};

static const nir_search_variable replace423_1 = {
   { nir_search_value_variable, 0 },
   1, /* insert */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace423_2_0_0 = {
   { nir_search_value_variable, 0 },
   3, /* bits */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace423_2_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* offset */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace423_2_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bfm,
   { &replace423_2_0_0.value, &replace423_2_0_1.value },
   NULL,
};

static const nir_search_variable replace423_2_1 = {
   { nir_search_value_variable, 0 },
   1, /* insert */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace423_2_2 = {
   { nir_search_value_variable, 0 },
   0, /* base */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace423_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bfi,
   { &replace423_2_0.value, &replace423_2_1.value, &replace423_2_2.value },
   NULL,
};
static const nir_search_expression replace423 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace423_0.value, &replace423_1.value, &replace423_2.value },
   NULL,
};
   
static const nir_search_variable search424_0 = {
   { nir_search_value_variable, 0 },
   0, /* base */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search424_1 = {
   { nir_search_value_variable, 0 },
   1, /* insert */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search424_2 = {
   { nir_search_value_variable, 0 },
   2, /* offset */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search424_3 = {
   { nir_search_value_variable, 0 },
   3, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search424 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bitfield_insert,
   { &search424_0.value, &search424_1.value, &search424_2.value, &search424_3.value },
   NULL,
};
   
static const nir_search_constant replace424_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1f /* 31 */ },
};

static const nir_search_variable replace424_0_1 = {
   { nir_search_value_variable, 0 },
   3, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace424_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace424_0_0.value, &replace424_0_1.value },
   NULL,
};

static const nir_search_variable replace424_1 = {
   { nir_search_value_variable, 0 },
   1, /* insert */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace424_2_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* base */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace424_2_0_1_0_0 = {
   { nir_search_value_variable, 0 },
   3, /* bits */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace424_2_0_1_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* offset */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace424_2_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bfm,
   { &replace424_2_0_1_0_0.value, &replace424_2_0_1_0_1.value },
   NULL,
};
static const nir_search_expression replace424_2_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace424_2_0_1_0.value },
   NULL,
};
static const nir_search_expression replace424_2_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace424_2_0_0.value, &replace424_2_0_1.value },
   NULL,
};

static const nir_search_variable replace424_2_1_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* insert */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace424_2_1_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* offset */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace424_2_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace424_2_1_0_0.value, &replace424_2_1_0_1.value },
   NULL,
};

static const nir_search_variable replace424_2_1_1_0 = {
   { nir_search_value_variable, 0 },
   3, /* bits */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace424_2_1_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* offset */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace424_2_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bfm,
   { &replace424_2_1_1_0.value, &replace424_2_1_1_1.value },
   NULL,
};
static const nir_search_expression replace424_2_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace424_2_1_0.value, &replace424_2_1_1.value },
   NULL,
};
static const nir_search_expression replace424_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ior,
   { &replace424_2_0.value, &replace424_2_1.value },
   NULL,
};
static const nir_search_expression replace424 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace424_0.value, &replace424_1.value, &replace424_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_bitfield_insert_xforms[] = {
   { &search423, &replace423.value, 26 },
   { &search424, &replace424.value, 27 },
};
   
static const nir_search_variable search425_0 = {
   { nir_search_value_variable, 0 },
   0, /* bits */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search425_1 = {
   { nir_search_value_variable, 0 },
   1, /* offset */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search425 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bfm,
   { &search425_0.value, &search425_1.value },
   NULL,
};
   
static const nir_search_constant replace425_0_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};

static const nir_search_variable replace425_0_0_1 = {
   { nir_search_value_variable, 0 },
   0, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace425_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace425_0_0_0.value, &replace425_0_0_1.value },
   NULL,
};

static const nir_search_constant replace425_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace425_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace425_0_0.value, &replace425_0_1.value },
   NULL,
};

static const nir_search_variable replace425_1 = {
   { nir_search_value_variable, 0 },
   1, /* offset */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace425 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace425_0.value, &replace425_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_bfm_xforms[] = {
   { &search425, &replace425.value, 28 },
};
   
static const nir_search_variable search426_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search426_1 = {
   { nir_search_value_variable, 0 },
   1, /* offset */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search426_2 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search426 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ibitfield_extract,
   { &search426_0.value, &search426_1.value, &search426_2.value },
   NULL,
};
   
static const nir_search_constant replace426_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1f /* 31 */ },
};

static const nir_search_variable replace426_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace426_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace426_0_0.value, &replace426_0_1.value },
   NULL,
};

static const nir_search_variable replace426_1 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace426_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace426_2_1 = {
   { nir_search_value_variable, 0 },
   1, /* offset */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace426_2_2 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace426_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ibfe,
   { &replace426_2_0.value, &replace426_2_1.value, &replace426_2_2.value },
   NULL,
};
static const nir_search_expression replace426 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace426_0.value, &replace426_1.value, &replace426_2.value },
   NULL,
};
   
static const nir_search_variable search428_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search428_1 = {
   { nir_search_value_variable, 0 },
   1, /* offset */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search428_2 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search428 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ibitfield_extract,
   { &search428_0.value, &search428_1.value, &search428_2.value },
   NULL,
};
   
static const nir_search_constant replace428_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable replace428_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace428_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace428_0_0.value, &replace428_0_1.value },
   NULL,
};

static const nir_search_constant replace428_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable replace428_2_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace428_2_0_1_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x20 /* 32 */ },
};

static const nir_search_variable replace428_2_0_1_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace428_2_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace428_2_0_1_0_0.value, &replace428_2_0_1_0_1.value },
   NULL,
};

static const nir_search_variable replace428_2_0_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* offset */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace428_2_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace428_2_0_1_0.value, &replace428_2_0_1_1.value },
   NULL,
};
static const nir_search_expression replace428_2_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace428_2_0_0.value, &replace428_2_0_1.value },
   NULL,
};

static const nir_search_constant replace428_2_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x20 /* 32 */ },
};

static const nir_search_variable replace428_2_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace428_2_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace428_2_1_0.value, &replace428_2_1_1.value },
   NULL,
};
static const nir_search_expression replace428_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &replace428_2_0.value, &replace428_2_1.value },
   NULL,
};
static const nir_search_expression replace428 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace428_0.value, &replace428_1.value, &replace428_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ibitfield_extract_xforms[] = {
   { &search426, &replace426.value, 29 },
   { &search428, &replace428.value, 30 },
};
   
static const nir_search_variable search427_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search427_1 = {
   { nir_search_value_variable, 0 },
   1, /* offset */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search427_2 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search427 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ubitfield_extract,
   { &search427_0.value, &search427_1.value, &search427_2.value },
   NULL,
};
   
static const nir_search_constant replace427_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1f /* 31 */ },
};

static const nir_search_variable replace427_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace427_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ult,
   { &replace427_0_0.value, &replace427_0_1.value },
   NULL,
};

static const nir_search_variable replace427_1 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace427_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace427_2_1 = {
   { nir_search_value_variable, 0 },
   1, /* offset */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace427_2_2 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace427_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ubfe,
   { &replace427_2_0.value, &replace427_2_1.value, &replace427_2_2.value },
   NULL,
};
static const nir_search_expression replace427 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace427_0.value, &replace427_1.value, &replace427_2.value },
   NULL,
};
   
static const nir_search_variable search429_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search429_1 = {
   { nir_search_value_variable, 0 },
   1, /* offset */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search429_2 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search429 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ubitfield_extract,
   { &search429_0.value, &search429_1.value, &search429_2.value },
   NULL,
};
   
static const nir_search_variable replace429_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace429_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* offset */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace429_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &replace429_0_0.value, &replace429_0_1.value },
   NULL,
};

static const nir_search_variable replace429_1_0_0 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace429_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x20 /* 32 */ },
};
static const nir_search_expression replace429_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ieq,
   { &replace429_1_0_0.value, &replace429_1_0_1.value },
   NULL,
};

static const nir_search_constant replace429_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xffffffff /* 4294967295 */ },
};

static const nir_search_variable replace429_1_2_0 = {
   { nir_search_value_variable, 0 },
   2, /* bits */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace429_1_2_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace429_1_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bfm,
   { &replace429_1_2_0.value, &replace429_1_2_1.value },
   NULL,
};
static const nir_search_expression replace429_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace429_1_0.value, &replace429_1_1.value, &replace429_1_2.value },
   NULL,
};
static const nir_search_expression replace429 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace429_0.value, &replace429_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ubitfield_extract_xforms[] = {
   { &search427, &replace427.value, 29 },
   { &search429, &replace429.value, 30 },
};
   
static const nir_search_variable search430_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search430 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ifind_msb,
   { &search430_0.value },
   NULL,
};
   
static const nir_search_variable replace430_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace430_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace430_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ilt,
   { &replace430_0_0_0.value, &replace430_0_0_1.value },
   NULL,
};

static const nir_search_variable replace430_0_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace430_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &replace430_0_1_0.value },
   NULL,
};

static const nir_search_variable replace430_0_2 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace430_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace430_0_0.value, &replace430_0_1.value, &replace430_0_2.value },
   NULL,
};
static const nir_search_expression replace430 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ufind_msb,
   { &replace430_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ifind_msb_xforms[] = {
   { &search430, &replace430.value, 31 },
};
   
static const nir_search_variable search431_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search431 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_find_lsb,
   { &search431_0.value },
   NULL,
};
   
static const nir_search_variable replace431_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace431_0_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* value */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace431_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &replace431_0_1_0.value },
   NULL,
};
static const nir_search_expression replace431_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace431_0_0.value, &replace431_0_1.value },
   NULL,
};
static const nir_search_expression replace431 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ufind_msb,
   { &replace431_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_find_lsb_xforms[] = {
   { &search431, &replace431.value, 32 },
};
   
static const nir_search_variable search432_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search432_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search432 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i8,
   { &search432_0.value, &search432_1.value },
   NULL,
};
   
static const nir_search_variable replace432_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace432_0_1_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x3 /* 3 */ },
};

static const nir_search_variable replace432_0_1_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace432_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace432_0_1_0_0.value, &replace432_0_1_0_1.value },
   NULL,
};

static const nir_search_constant replace432_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x8 /* 8 */ },
};
static const nir_search_expression replace432_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace432_0_1_0.value, &replace432_0_1_1.value },
   NULL,
};
static const nir_search_expression replace432_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace432_0_0.value, &replace432_0_1.value },
   NULL,
};

static const nir_search_constant replace432_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x18 /* 24 */ },
};
static const nir_search_expression replace432 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &replace432_0.value, &replace432_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_extract_i8_xforms[] = {
   { &search432, &replace432.value, 33 },
};
   
static const nir_search_variable search434_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search434_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search434 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i16,
   { &search434_0.value, &search434_1.value },
   NULL,
};
   
static const nir_search_variable replace434_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace434_0_1_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};

static const nir_search_variable replace434_0_1_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace434_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace434_0_1_0_0.value, &replace434_0_1_0_1.value },
   NULL,
};

static const nir_search_constant replace434_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression replace434_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace434_0_1_0.value, &replace434_0_1_1.value },
   NULL,
};
static const nir_search_expression replace434_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace434_0_0.value, &replace434_0_1.value },
   NULL,
};

static const nir_search_constant replace434_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression replace434 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &replace434_0.value, &replace434_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_extract_i16_xforms[] = {
   { &search434, &replace434.value, 34 },
};
   
static const nir_search_variable search435_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search435_1 = {
   { nir_search_value_variable, 32 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search435 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u16,
   { &search435_0.value, &search435_1.value },
   NULL,
};
   
static const nir_search_variable replace435_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace435_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace435_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x10 /* 16 */ },
};
static const nir_search_expression replace435_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace435_0_1_0.value, &replace435_0_1_1.value },
   NULL,
};
static const nir_search_expression replace435_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ushr,
   { &replace435_0_0.value, &replace435_0_1.value },
   NULL,
};

static const nir_search_constant replace435_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xffff /* 65535 */ },
};
static const nir_search_expression replace435 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace435_0.value, &replace435_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_extract_u16_xforms[] = {
   { &search435, &replace435.value, 34 },
};
   
static const nir_search_variable search436_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search436 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_unorm_2x16,
   { &search436_0.value },
   NULL,
};
   
static const nir_search_variable replace436_0_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace436_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &replace436_0_0_0_0_0.value },
   NULL,
};

static const nir_search_constant replace436_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x40efffe000000000 /* 65535.0 */ },
};
static const nir_search_expression replace436_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace436_0_0_0_0.value, &replace436_0_0_0_1.value },
   NULL,
};
static const nir_search_expression replace436_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fround_even,
   { &replace436_0_0_0.value },
   NULL,
};
static const nir_search_expression replace436_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2u32,
   { &replace436_0_0.value },
   NULL,
};
static const nir_search_expression replace436 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_uvec2_to_uint,
   { &replace436_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_pack_unorm_2x16_xforms[] = {
   { &search436, &replace436.value, 35 },
};
   
static const nir_search_variable search437_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search437 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_unorm_4x8,
   { &search437_0.value },
   NULL,
};
   
static const nir_search_variable replace437_0_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace437_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fsat,
   { &replace437_0_0_0_0_0.value },
   NULL,
};

static const nir_search_constant replace437_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x406fe00000000000 /* 255.0 */ },
};
static const nir_search_expression replace437_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace437_0_0_0_0.value, &replace437_0_0_0_1.value },
   NULL,
};
static const nir_search_expression replace437_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fround_even,
   { &replace437_0_0_0.value },
   NULL,
};
static const nir_search_expression replace437_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2u32,
   { &replace437_0_0.value },
   NULL,
};
static const nir_search_expression replace437 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_uvec4_to_uint,
   { &replace437_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_pack_unorm_4x8_xforms[] = {
   { &search437, &replace437.value, 36 },
};
   
static const nir_search_variable search438_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search438 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_snorm_2x16,
   { &search438_0.value },
   NULL,
};
   
static const nir_search_constant replace438_0_0_0_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_constant replace438_0_0_0_0_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0xbff0000000000000L /* -1.0 */ },
};

static const nir_search_variable replace438_0_0_0_0_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace438_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace438_0_0_0_0_1_0.value, &replace438_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression replace438_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace438_0_0_0_0_0.value, &replace438_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant replace438_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x40dfffc000000000 /* 32767.0 */ },
};
static const nir_search_expression replace438_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace438_0_0_0_0.value, &replace438_0_0_0_1.value },
   NULL,
};
static const nir_search_expression replace438_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fround_even,
   { &replace438_0_0_0.value },
   NULL,
};
static const nir_search_expression replace438_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2i32,
   { &replace438_0_0.value },
   NULL,
};
static const nir_search_expression replace438 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_uvec2_to_uint,
   { &replace438_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_pack_snorm_2x16_xforms[] = {
   { &search438, &replace438.value, 37 },
};
   
static const nir_search_variable search439_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search439 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_snorm_4x8,
   { &search439_0.value },
   NULL,
};
   
static const nir_search_constant replace439_0_0_0_0_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_constant replace439_0_0_0_0_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0xbff0000000000000L /* -1.0 */ },
};

static const nir_search_variable replace439_0_0_0_0_1_1 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace439_0_0_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace439_0_0_0_0_1_0.value, &replace439_0_0_0_0_1_1.value },
   NULL,
};
static const nir_search_expression replace439_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace439_0_0_0_0_0.value, &replace439_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant replace439_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x405fc00000000000 /* 127.0 */ },
};
static const nir_search_expression replace439_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace439_0_0_0_0.value, &replace439_0_0_0_1.value },
   NULL,
};
static const nir_search_expression replace439_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fround_even,
   { &replace439_0_0_0.value },
   NULL,
};
static const nir_search_expression replace439_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2i32,
   { &replace439_0_0.value },
   NULL,
};
static const nir_search_expression replace439 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_uvec4_to_uint,
   { &replace439_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_pack_snorm_4x8_xforms[] = {
   { &search439, &replace439.value, 38 },
};
   
static const nir_search_variable search440_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search440 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_unpack_unorm_2x16,
   { &search440_0.value },
   NULL,
};
   
static const nir_search_variable replace440_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace440_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace440_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u16,
   { &replace440_0_0_0_0.value, &replace440_0_0_0_1.value },
   NULL,
};

static const nir_search_variable replace440_0_0_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace440_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace440_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u16,
   { &replace440_0_0_1_0.value, &replace440_0_0_1_1.value },
   NULL,
};
static const nir_search_expression replace440_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec2,
   { &replace440_0_0_0.value, &replace440_0_0_1.value },
   NULL,
};
static const nir_search_expression replace440_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &replace440_0_0.value },
   NULL,
};

static const nir_search_constant replace440_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x40efffe000000000 /* 65535.0 */ },
};
static const nir_search_expression replace440 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdiv,
   { &replace440_0.value, &replace440_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_unpack_unorm_2x16_xforms[] = {
   { &search440, &replace440.value, 39 },
};
   
static const nir_search_variable search441_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search441 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_unpack_unorm_4x8,
   { &search441_0.value },
   NULL,
};
   
static const nir_search_variable replace441_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace441_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace441_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace441_0_0_0_0.value, &replace441_0_0_0_1.value },
   NULL,
};

static const nir_search_variable replace441_0_0_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace441_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace441_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace441_0_0_1_0.value, &replace441_0_0_1_1.value },
   NULL,
};

static const nir_search_variable replace441_0_0_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace441_0_0_2_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x2 /* 2 */ },
};
static const nir_search_expression replace441_0_0_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace441_0_0_2_0.value, &replace441_0_0_2_1.value },
   NULL,
};

static const nir_search_variable replace441_0_0_3_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace441_0_0_3_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x3 /* 3 */ },
};
static const nir_search_expression replace441_0_0_3 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_u8,
   { &replace441_0_0_3_0.value, &replace441_0_0_3_1.value },
   NULL,
};
static const nir_search_expression replace441_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec4,
   { &replace441_0_0_0.value, &replace441_0_0_1.value, &replace441_0_0_2.value, &replace441_0_0_3.value },
   NULL,
};
static const nir_search_expression replace441_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &replace441_0_0.value },
   NULL,
};

static const nir_search_constant replace441_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x406fe00000000000 /* 255.0 */ },
};
static const nir_search_expression replace441 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdiv,
   { &replace441_0.value, &replace441_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_unpack_unorm_4x8_xforms[] = {
   { &search441, &replace441.value, 40 },
};
   
static const nir_search_variable search442_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search442 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_unpack_snorm_2x16,
   { &search442_0.value },
   NULL,
};
   
static const nir_search_constant replace442_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_constant replace442_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0xbff0000000000000L /* -1.0 */ },
};

static const nir_search_variable replace442_1_1_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace442_1_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace442_1_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i16,
   { &replace442_1_1_0_0_0_0.value, &replace442_1_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable replace442_1_1_0_0_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace442_1_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace442_1_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i16,
   { &replace442_1_1_0_0_1_0.value, &replace442_1_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression replace442_1_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec2,
   { &replace442_1_1_0_0_0.value, &replace442_1_1_0_0_1.value },
   NULL,
};
static const nir_search_expression replace442_1_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &replace442_1_1_0_0.value },
   NULL,
};

static const nir_search_constant replace442_1_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x40dfffc000000000 /* 32767.0 */ },
};
static const nir_search_expression replace442_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdiv,
   { &replace442_1_1_0.value, &replace442_1_1_1.value },
   NULL,
};
static const nir_search_expression replace442_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace442_1_0.value, &replace442_1_1.value },
   NULL,
};
static const nir_search_expression replace442 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace442_0.value, &replace442_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_unpack_snorm_2x16_xforms[] = {
   { &search442, &replace442.value, 41 },
};
   
static const nir_search_variable search443_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search443 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_unpack_snorm_4x8,
   { &search443_0.value },
   NULL,
};
   
static const nir_search_constant replace443_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};

static const nir_search_constant replace443_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0xbff0000000000000L /* -1.0 */ },
};

static const nir_search_variable replace443_1_1_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace443_1_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
static const nir_search_expression replace443_1_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i8,
   { &replace443_1_1_0_0_0_0.value, &replace443_1_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable replace443_1_1_0_0_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace443_1_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace443_1_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i8,
   { &replace443_1_1_0_0_1_0.value, &replace443_1_1_0_0_1_1.value },
   NULL,
};

static const nir_search_variable replace443_1_1_0_0_2_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace443_1_1_0_0_2_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x2 /* 2 */ },
};
static const nir_search_expression replace443_1_1_0_0_2 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i8,
   { &replace443_1_1_0_0_2_0.value, &replace443_1_1_0_0_2_1.value },
   NULL,
};

static const nir_search_variable replace443_1_1_0_0_3_0 = {
   { nir_search_value_variable, 0 },
   0, /* v */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace443_1_1_0_0_3_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x3 /* 3 */ },
};
static const nir_search_expression replace443_1_1_0_0_3 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_extract_i8,
   { &replace443_1_1_0_0_3_0.value, &replace443_1_1_0_0_3_1.value },
   NULL,
};
static const nir_search_expression replace443_1_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_vec4,
   { &replace443_1_1_0_0_0.value, &replace443_1_1_0_0_1.value, &replace443_1_1_0_0_2.value, &replace443_1_1_0_0_3.value },
   NULL,
};
static const nir_search_expression replace443_1_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &replace443_1_1_0_0.value },
   NULL,
};

static const nir_search_constant replace443_1_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x405fc00000000000 /* 127.0 */ },
};
static const nir_search_expression replace443_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdiv,
   { &replace443_1_1_0.value, &replace443_1_1_1.value },
   NULL,
};
static const nir_search_expression replace443_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace443_1_0.value, &replace443_1_1.value },
   NULL,
};
static const nir_search_expression replace443 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace443_0.value, &replace443_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_unpack_snorm_4x8_xforms[] = {
   { &search443, &replace443.value, 42 },
};
   
static const nir_search_variable search464_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search464_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search464_0_0.value },
   NULL,
};
static const nir_search_expression search464 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2b,
   { &search464_0.value },
   NULL,
};
   
static const nir_search_variable replace464 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_f2b_xforms[] = {
   { &search464, &replace464.value, 0 },
};
   
static const nir_search_variable search466_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search466_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search466_0_0.value },
   NULL,
};
static const nir_search_expression search466 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2f16,
   { &search466_0.value },
   NULL,
};
   
static const nir_search_variable replace466_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace466 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace466_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2f16_xforms[] = {
   { &search466, &replace466.value, 0 },
};
   
static const nir_search_variable search467_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search467_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search467_0_0.value },
   NULL,
};
static const nir_search_expression search467 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2f32,
   { &search467_0.value },
   NULL,
};
   
static const nir_search_variable replace467_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace467 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace467_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2f32_xforms[] = {
   { &search467, &replace467.value, 0 },
};
   
static const nir_search_variable search468_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search468_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search468_0_0.value },
   NULL,
};
static const nir_search_expression search468 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2f64,
   { &search468_0.value },
   NULL,
};
   
static const nir_search_variable replace468_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace468 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace468_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2f64_xforms[] = {
   { &search468, &replace468.value, 0 },
};
   
static const nir_search_variable search469_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search469_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search469_0_0.value },
   NULL,
};
static const nir_search_expression search469 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2u8,
   { &search469_0.value },
   NULL,
};
   
static const nir_search_variable replace469_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace469 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace469_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2u8_xforms[] = {
   { &search469, &replace469.value, 0 },
};
   
static const nir_search_variable search470_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search470_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search470_0_0.value },
   NULL,
};
static const nir_search_expression search470 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2u16,
   { &search470_0.value },
   NULL,
};
   
static const nir_search_variable replace470_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace470 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace470_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2u16_xforms[] = {
   { &search470, &replace470.value, 0 },
};
   
static const nir_search_variable search472_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search472_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search472_0_0.value },
   NULL,
};
static const nir_search_expression search472 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2u64,
   { &search472_0.value },
   NULL,
};
   
static const nir_search_variable replace472_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace472 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace472_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2u64_xforms[] = {
   { &search472, &replace472.value, 0 },
};
   
static const nir_search_variable search473_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search473_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search473_0_0.value },
   NULL,
};
static const nir_search_expression search473 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2i8,
   { &search473_0.value },
   NULL,
};
   
static const nir_search_variable replace473_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace473 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace473_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2i8_xforms[] = {
   { &search473, &replace473.value, 0 },
};
   
static const nir_search_variable search474_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search474_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search474_0_0.value },
   NULL,
};
static const nir_search_expression search474 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2i16,
   { &search474_0.value },
   NULL,
};
   
static const nir_search_variable replace474_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace474 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace474_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2i16_xforms[] = {
   { &search474, &replace474.value, 0 },
};
   
static const nir_search_variable search476_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search476_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search476_0_0.value },
   NULL,
};
static const nir_search_expression search476 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_f2i64,
   { &search476_0.value },
   NULL,
};
   
static const nir_search_variable replace476_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace476 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace476_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_f2i64_xforms[] = {
   { &search476, &replace476.value, 0 },
};
   
static const nir_search_variable search477_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search477_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search477_0_0.value },
   NULL,
};
static const nir_search_expression search477 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f16,
   { &search477_0.value },
   NULL,
};
   
static const nir_search_variable replace477_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace477 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace477_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_u2f16_xforms[] = {
   { &search477, &replace477.value, 0 },
};
   
static const nir_search_variable search478_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search478_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search478_0_0.value },
   NULL,
};
static const nir_search_expression search478 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f32,
   { &search478_0.value },
   NULL,
};
   
static const nir_search_variable replace478_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace478 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace478_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_u2f32_xforms[] = {
   { &search478, &replace478.value, 0 },
};
   
static const nir_search_variable search479_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search479_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search479_0_0.value },
   NULL,
};
static const nir_search_expression search479 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2f64,
   { &search479_0.value },
   NULL,
};
   
static const nir_search_variable replace479_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace479 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace479_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_u2f64_xforms[] = {
   { &search479, &replace479.value, 0 },
};
   
static const nir_search_variable search480_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search480_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search480_0_0.value },
   NULL,
};
static const nir_search_expression search480 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2u8,
   { &search480_0.value },
   NULL,
};
   
static const nir_search_variable replace480_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace480 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace480_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_u2u8_xforms[] = {
   { &search480, &replace480.value, 0 },
};
   
static const nir_search_variable search481_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search481_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search481_0_0.value },
   NULL,
};
static const nir_search_expression search481 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2u16,
   { &search481_0.value },
   NULL,
};
   
static const nir_search_variable replace481_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace481 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace481_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_u2u16_xforms[] = {
   { &search481, &replace481.value, 0 },
};
   
static const nir_search_variable search482_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search482_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search482_0_0.value },
   NULL,
};
static const nir_search_expression search482 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2u32,
   { &search482_0.value },
   NULL,
};
   
static const nir_search_variable replace482_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace482 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace482_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_u2u32_xforms[] = {
   { &search482, &replace482.value, 0 },
};
   
static const nir_search_variable search483_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search483_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search483_0_0.value },
   NULL,
};
static const nir_search_expression search483 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_u2u64,
   { &search483_0.value },
   NULL,
};
   
static const nir_search_variable replace483_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace483 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace483_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_u2u64_xforms[] = {
   { &search483, &replace483.value, 0 },
};
   
static const nir_search_variable search484_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search484_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search484_0_0.value },
   NULL,
};
static const nir_search_expression search484 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f16,
   { &search484_0.value },
   NULL,
};
   
static const nir_search_variable replace484_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace484 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace484_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_i2f16_xforms[] = {
   { &search484, &replace484.value, 0 },
};
   
static const nir_search_variable search485_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search485_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search485_0_0.value },
   NULL,
};
static const nir_search_expression search485 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f32,
   { &search485_0.value },
   NULL,
};
   
static const nir_search_variable replace485_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace485 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace485_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_i2f32_xforms[] = {
   { &search485, &replace485.value, 0 },
};
   
static const nir_search_variable search486_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search486_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search486_0_0.value },
   NULL,
};
static const nir_search_expression search486 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2f64,
   { &search486_0.value },
   NULL,
};
   
static const nir_search_variable replace486_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace486 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &replace486_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_i2f64_xforms[] = {
   { &search486, &replace486.value, 0 },
};
   
static const nir_search_variable search487_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search487_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search487_0_0.value },
   NULL,
};
static const nir_search_expression search487 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2i8,
   { &search487_0.value },
   NULL,
};
   
static const nir_search_variable replace487_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace487 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace487_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_i2i8_xforms[] = {
   { &search487, &replace487.value, 0 },
};
   
static const nir_search_variable search488_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search488_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search488_0_0.value },
   NULL,
};
static const nir_search_expression search488 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2i16,
   { &search488_0.value },
   NULL,
};
   
static const nir_search_variable replace488_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace488 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace488_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_i2i16_xforms[] = {
   { &search488, &replace488.value, 0 },
};
   
static const nir_search_variable search489_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search489_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search489_0_0.value },
   NULL,
};
static const nir_search_expression search489 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2i32,
   { &search489_0.value },
   NULL,
};
   
static const nir_search_variable replace489_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace489 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace489_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_i2i32_xforms[] = {
   { &search489, &replace489.value, 0 },
};
   
static const nir_search_variable search490_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search490_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &search490_0_0.value },
   NULL,
};
static const nir_search_expression search490 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_i2i64,
   { &search490_0.value },
   NULL,
};
   
static const nir_search_variable replace490_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace490 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2i,
   { &replace490_0.value },
   NULL,
};

static const struct transform nir_opt_algebraic_i2i64_xforms[] = {
   { &search490, &replace490.value, 0 },
};
   
static const nir_search_variable search491_0 = {
   { nir_search_value_variable, 0 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search491_1 = {
   { nir_search_value_variable, 0 },
   1, /* exp */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search491 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_ldexp,
   { &search491_0.value, &search491_1.value },
   NULL,
};
   
static const nir_search_variable replace491_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace491_0_1_0_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* exp */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace491_0_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0xfc /* -252 */ },
};
static const nir_search_expression replace491_0_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace491_0_1_0_0_0_0_0.value, &replace491_0_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant replace491_0_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xfe /* 254 */ },
};
static const nir_search_expression replace491_0_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace491_0_1_0_0_0_0.value, &replace491_0_1_0_0_0_1.value },
   NULL,
};

static const nir_search_constant replace491_0_1_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace491_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &replace491_0_1_0_0_0.value, &replace491_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant replace491_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x7f /* 127 */ },
};
static const nir_search_expression replace491_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace491_0_1_0_0.value, &replace491_0_1_0_1.value },
   NULL,
};

static const nir_search_constant replace491_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x17 /* 23 */ },
};
static const nir_search_expression replace491_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace491_0_1_0.value, &replace491_0_1_1.value },
   NULL,
};
static const nir_search_expression replace491_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace491_0_0.value, &replace491_0_1.value },
   NULL,
};

static const nir_search_variable replace491_1_0_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* exp */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace491_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0xfc /* -252 */ },
};
static const nir_search_expression replace491_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace491_1_0_0_0_0_0.value, &replace491_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant replace491_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xfe /* 254 */ },
};
static const nir_search_expression replace491_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace491_1_0_0_0_0.value, &replace491_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable replace491_1_0_0_1_0_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* exp */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace491_1_0_0_1_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0xfc /* -252 */ },
};
static const nir_search_expression replace491_1_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace491_1_0_0_1_0_0_0.value, &replace491_1_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant replace491_1_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0xfe /* 254 */ },
};
static const nir_search_expression replace491_1_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace491_1_0_0_1_0_0.value, &replace491_1_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant replace491_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace491_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &replace491_1_0_0_1_0.value, &replace491_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression replace491_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace491_1_0_0_0.value, &replace491_1_0_0_1.value },
   NULL,
};

static const nir_search_constant replace491_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x7f /* 127 */ },
};
static const nir_search_expression replace491_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace491_1_0_0.value, &replace491_1_0_1.value },
   NULL,
};

static const nir_search_constant replace491_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x17 /* 23 */ },
};
static const nir_search_expression replace491_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace491_1_0.value, &replace491_1_1.value },
   NULL,
};
static const nir_search_expression replace491 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace491_0.value, &replace491_1.value },
   NULL,
};
   
static const nir_search_variable search492_0 = {
   { nir_search_value_variable, 0 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search492_1 = {
   { nir_search_value_variable, 0 },
   1, /* exp */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search492 = {
   { nir_search_value_expression, 64 },
   false,
   nir_op_ldexp,
   { &search492_0.value, &search492_1.value },
   NULL,
};
   
static const nir_search_variable replace492_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* x */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace492_0_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable replace492_0_1_1_0_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* exp */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace492_0_1_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x7fc /* -2044 */ },
};
static const nir_search_expression replace492_0_1_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace492_0_1_1_0_0_0_0_0.value, &replace492_0_1_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant replace492_0_1_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x7fe /* 2046 */ },
};
static const nir_search_expression replace492_0_1_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace492_0_1_1_0_0_0_0.value, &replace492_0_1_1_0_0_0_1.value },
   NULL,
};

static const nir_search_constant replace492_0_1_1_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace492_0_1_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &replace492_0_1_1_0_0_0.value, &replace492_0_1_1_0_0_1.value },
   NULL,
};

static const nir_search_constant replace492_0_1_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x3ff /* 1023 */ },
};
static const nir_search_expression replace492_0_1_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace492_0_1_1_0_0.value, &replace492_0_1_1_0_1.value },
   NULL,
};

static const nir_search_constant replace492_0_1_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x14 /* 20 */ },
};
static const nir_search_expression replace492_0_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace492_0_1_1_0.value, &replace492_0_1_1_1.value },
   NULL,
};
static const nir_search_expression replace492_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_64_2x32_split,
   { &replace492_0_1_0.value, &replace492_0_1_1.value },
   NULL,
};
static const nir_search_expression replace492_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace492_0_0.value, &replace492_0_1.value },
   NULL,
};

static const nir_search_constant replace492_1_0 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};

static const nir_search_variable replace492_1_1_0_0_0_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* exp */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace492_1_1_0_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x7fc /* -2044 */ },
};
static const nir_search_expression replace492_1_1_0_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace492_1_1_0_0_0_0_0.value, &replace492_1_1_0_0_0_0_1.value },
   NULL,
};

static const nir_search_constant replace492_1_1_0_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x7fe /* 2046 */ },
};
static const nir_search_expression replace492_1_1_0_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace492_1_1_0_0_0_0.value, &replace492_1_1_0_0_0_1.value },
   NULL,
};

static const nir_search_variable replace492_1_1_0_0_1_0_0_0 = {
   { nir_search_value_variable, 0 },
   1, /* exp */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace492_1_1_0_0_1_0_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { -0x7fc /* -2044 */ },
};
static const nir_search_expression replace492_1_1_0_0_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imax,
   { &replace492_1_1_0_0_1_0_0_0.value, &replace492_1_1_0_0_1_0_0_1.value },
   NULL,
};

static const nir_search_constant replace492_1_1_0_0_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x7fe /* 2046 */ },
};
static const nir_search_expression replace492_1_1_0_0_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imin,
   { &replace492_1_1_0_0_1_0_0.value, &replace492_1_1_0_0_1_0_1.value },
   NULL,
};

static const nir_search_constant replace492_1_1_0_0_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x1 /* 1 */ },
};
static const nir_search_expression replace492_1_1_0_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishr,
   { &replace492_1_1_0_0_1_0.value, &replace492_1_1_0_0_1_1.value },
   NULL,
};
static const nir_search_expression replace492_1_1_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_isub,
   { &replace492_1_1_0_0_0.value, &replace492_1_1_0_0_1.value },
   NULL,
};

static const nir_search_constant replace492_1_1_0_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x3ff /* 1023 */ },
};
static const nir_search_expression replace492_1_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace492_1_1_0_0.value, &replace492_1_1_0_1.value },
   NULL,
};

static const nir_search_constant replace492_1_1_1 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x14 /* 20 */ },
};
static const nir_search_expression replace492_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ishl,
   { &replace492_1_1_0.value, &replace492_1_1_1.value },
   NULL,
};
static const nir_search_expression replace492_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_pack_64_2x32_split,
   { &replace492_1_0.value, &replace492_1_1.value },
   NULL,
};
static const nir_search_expression replace492 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace492_0.value, &replace492_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_ldexp_xforms[] = {
   { &search491, &replace491.value, 43 },
   { &search492, &replace492.value, 43 },
};

static bool
nir_opt_algebraic_block(nir_builder *build, nir_block *block,
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
      case nir_op_imul:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_imul_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_imul_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_udiv:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_udiv_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_udiv_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_idiv:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_idiv_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_idiv_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_umod:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_umod_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_umod_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_imod:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_imod_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_imod_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fneg:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fneg_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fneg_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ineg:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ineg_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ineg_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fabs:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fabs_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fabs_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_iabs:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_iabs_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_iabs_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fadd:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fadd_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fadd_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_iadd:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_iadd_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_iadd_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_usadd_4x8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_usadd_4x8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_usadd_4x8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fmul:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fmul_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fmul_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_umul_unorm_4x8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_umul_unorm_4x8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_umul_unorm_4x8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ffma:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ffma_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ffma_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_flrp:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_flrp_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_flrp_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ffract:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ffract_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ffract_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fdot4:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fdot4_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fdot4_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fdot3:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fdot3_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fdot3_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ishl:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ishl_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ishl_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_inot:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_inot_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_inot_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fge:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fge_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fge_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fne:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fne_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fne_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_feq:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_feq_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_feq_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_flt:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_flt_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_flt_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ieq:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ieq_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ieq_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ine:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ine_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ine_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fmax:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fmax_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fmax_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fmin:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fmin_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fmin_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_bcsel:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_bcsel_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_bcsel_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_imin:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_imin_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_imin_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_imax:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_imax_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_imax_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_umin:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_umin_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_umin_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_umax:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_umax_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_umax_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fsat:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fsat_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fsat_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_extract_u8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_extract_u8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_extract_u8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ior:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ior_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ior_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_iand:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_iand_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_iand_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ilt:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ilt_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ilt_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ige:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ige_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ige_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ult:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ult_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ult_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_uge:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_uge_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_uge_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_slt:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_slt_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_slt_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_sge:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_sge_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_sge_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_seq:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_seq_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_seq_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_sne:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_sne_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_sne_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fand:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fand_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fand_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fxor:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fxor_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fxor_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ixor:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ixor_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ixor_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ishr:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ishr_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ishr_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ushr:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ushr_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ushr_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fexp2:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fexp2_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fexp2_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_flog2:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_flog2_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_flog2_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fpow:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fpow_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fpow_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fsqrt:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fsqrt_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fsqrt_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_frcp:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_frcp_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_frcp_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_frsq:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_frsq_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_frsq_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fdiv:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fdiv_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fdiv_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fcsel:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fcsel_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fcsel_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_i2b:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_i2b_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_i2b_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2i32:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2i32_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2i32_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2u32:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2u32_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2u32_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_unpack_64_2x32_split_x:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_unpack_64_2x32_split_x_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_unpack_64_2x32_split_x_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_unpack_64_2x32_split_y:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_unpack_64_2x32_split_y_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_unpack_64_2x32_split_y_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_pack_64_2x32_split:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_pack_64_2x32_split_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_pack_64_2x32_split_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fsub:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fsub_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fsub_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_isub:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_isub_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_isub_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ussub_4x8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ussub_4x8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ussub_4x8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fmod:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_fmod_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_fmod_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_frem:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_frem_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_frem_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_uadd_carry:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_uadd_carry_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_uadd_carry_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_usub_borrow:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_usub_borrow_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_usub_borrow_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_bitfield_insert:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_bitfield_insert_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_bitfield_insert_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_bfm:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_bfm_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_bfm_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ibitfield_extract:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ibitfield_extract_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ibitfield_extract_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ubitfield_extract:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ubitfield_extract_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ubitfield_extract_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ifind_msb:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ifind_msb_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ifind_msb_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_find_lsb:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_find_lsb_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_find_lsb_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_extract_i8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_extract_i8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_extract_i8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_extract_i16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_extract_i16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_extract_i16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_extract_u16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_extract_u16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_extract_u16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_pack_unorm_2x16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_pack_unorm_2x16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_pack_unorm_2x16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_pack_unorm_4x8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_pack_unorm_4x8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_pack_unorm_4x8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_pack_snorm_2x16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_pack_snorm_2x16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_pack_snorm_2x16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_pack_snorm_4x8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_pack_snorm_4x8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_pack_snorm_4x8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_unpack_unorm_2x16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_unpack_unorm_2x16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_unpack_unorm_2x16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_unpack_unorm_4x8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_unpack_unorm_4x8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_unpack_unorm_4x8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_unpack_snorm_2x16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_unpack_snorm_2x16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_unpack_snorm_2x16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_unpack_snorm_4x8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_unpack_snorm_4x8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_unpack_snorm_4x8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2b:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2b_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2b_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2f16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2f16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2f16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2f32:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2f32_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2f32_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2f64:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2f64_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2f64_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2u8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2u8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2u8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2u16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2u16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2u16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2u64:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2u64_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2u64_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2i8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2i8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2i8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2i16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2i16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2i16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_f2i64:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_f2i64_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_f2i64_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_u2f16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_u2f16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_u2f16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_u2f32:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_u2f32_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_u2f32_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_u2f64:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_u2f64_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_u2f64_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_u2u8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_u2u8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_u2u8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_u2u16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_u2u16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_u2u16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_u2u32:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_u2u32_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_u2u32_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_u2u64:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_u2u64_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_u2u64_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_i2f16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_i2f16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_i2f16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_i2f32:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_i2f32_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_i2f32_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_i2f64:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_i2f64_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_i2f64_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_i2i8:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_i2i8_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_i2i8_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_i2i16:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_i2i16_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_i2i16_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_i2i32:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_i2i32_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_i2i32_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_i2i64:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_i2i64_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_i2i64_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_ldexp:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_ldexp_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_ldexp_xforms[i];
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
nir_opt_algebraic_impl(nir_function_impl *impl, const bool *condition_flags)
{
   bool progress = false;

   nir_builder build;
   nir_builder_init(&build, impl);

   nir_foreach_block_reverse(block, impl) {
      progress |= nir_opt_algebraic_block(&build, block, condition_flags);
   }

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);

   return progress;
}


bool
nir_opt_algebraic(nir_shader *shader)
{
   bool progress = false;
   bool condition_flags[44];
   const nir_shader_compiler_options *options = shader->options;
   (void) options;

   condition_flags[0] = true;
   condition_flags[1] = options->lower_idiv;
   condition_flags[2] = options->lower_flrp32;
   condition_flags[3] = options->lower_flrp64;
   condition_flags[4] = options->lower_ffract;
   condition_flags[5] = !options->lower_flrp32;
   condition_flags[6] = !options->lower_flrp64;
   condition_flags[7] = options->lower_ffma;
   condition_flags[8] = options->fuse_ffma;
   condition_flags[9] = !options->lower_fsat;
   condition_flags[10] = options->lower_fsat;
   condition_flags[11] = options->lower_scmp;
   condition_flags[12] = !options->lower_b2f;
   condition_flags[13] = options->lower_fpow;
   condition_flags[14] = !options->lower_fpow;
   condition_flags[15] = options->lower_fdiv;
   condition_flags[16] = options->lower_fsqrt;
   condition_flags[17] = !options->lower_fsqrt;
   condition_flags[18] = !options->lower_extract_byte;
   condition_flags[19] = !options->lower_extract_word;
   condition_flags[20] = options->lower_sub;
   condition_flags[21] = options->lower_negate;
   condition_flags[22] = options->lower_fmod32;
   condition_flags[23] = options->lower_fmod64;
   condition_flags[24] = options->lower_uadd_carry;
   condition_flags[25] = options->lower_usub_borrow;
   condition_flags[26] = options->lower_bitfield_insert;
   condition_flags[27] = options->lower_bitfield_insert_to_shifts;
   condition_flags[28] = options->lower_bfm;
   condition_flags[29] = options->lower_bitfield_extract;
   condition_flags[30] = options->lower_bitfield_extract_to_shifts;
   condition_flags[31] = options->lower_ifind_msb;
   condition_flags[32] = options->lower_find_lsb;
   condition_flags[33] = options->lower_extract_byte;
   condition_flags[34] = options->lower_extract_word;
   condition_flags[35] = options->lower_pack_unorm_2x16;
   condition_flags[36] = options->lower_pack_unorm_4x8;
   condition_flags[37] = options->lower_pack_snorm_2x16;
   condition_flags[38] = options->lower_pack_snorm_4x8;
   condition_flags[39] = options->lower_unpack_unorm_2x16;
   condition_flags[40] = options->lower_unpack_unorm_4x8;
   condition_flags[41] = options->lower_unpack_snorm_2x16;
   condition_flags[42] = options->lower_unpack_snorm_4x8;
   condition_flags[43] = options->lower_ldexp;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_opt_algebraic_impl(function->impl, condition_flags);
   }

   return progress;
}


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

   
static const nir_search_variable search524_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   (is_not_const),
};

static const nir_search_variable search524_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search524_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search524_0_0.value, &search524_0_1.value },
   (is_used_once),
};

static const nir_search_variable search524_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   (is_not_const),
};
static const nir_search_expression search524 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fmul,
   { &search524_0.value, &search524_1.value },
   (is_used_once),
};
   
static const nir_search_variable replace524_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace524_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace524_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace524_0_0.value, &replace524_0_1.value },
   NULL,
};

static const nir_search_variable replace524_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace524 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace524_0.value, &replace524_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_before_ffma_fmul_xforms[] = {
   { &search524, &replace524.value, 0 },
};
   
static const nir_search_variable search525_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   (is_not_const),
};

static const nir_search_variable search525_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search525_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search525_0_0.value, &search525_0_1.value },
   (is_used_once),
};

static const nir_search_variable search525_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   (is_not_const),
};
static const nir_search_expression search525 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search525_0.value, &search525_1.value },
   (is_used_once),
};
   
static const nir_search_variable replace525_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace525_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace525_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace525_0_0.value, &replace525_0_1.value },
   NULL,
};

static const nir_search_variable replace525_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace525 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace525_0.value, &replace525_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_before_ffma_imul_xforms[] = {
   { &search525, &replace525.value, 0 },
};
   
static const nir_search_variable search526_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   (is_not_const),
};

static const nir_search_variable search526_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search526_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search526_0_0.value, &search526_0_1.value },
   (is_used_once),
};

static const nir_search_variable search526_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   (is_not_const),
};
static const nir_search_expression search526 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search526_0.value, &search526_1.value },
   (is_used_once),
};
   
static const nir_search_variable replace526_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace526_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace526_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace526_0_0.value, &replace526_0_1.value },
   NULL,
};

static const nir_search_variable replace526_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace526 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace526_0.value, &replace526_1.value },
   NULL,
};
   
static const nir_search_variable search528_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search528_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search528_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search528_0_0.value, &search528_0_1.value },
   NULL,
};

static const nir_search_variable search528_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search528_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search528_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &search528_1_0.value, &search528_1_1.value },
   NULL,
};
static const nir_search_expression search528 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search528_0.value, &search528_1.value },
   NULL,
};
   
static const nir_search_variable replace528_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace528_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace528_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace528_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace528_1_0.value, &replace528_1_1.value },
   NULL,
};
static const nir_search_expression replace528 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmul,
   { &replace528_0.value, &replace528_1.value },
   NULL,
};
   
static const nir_search_variable search530_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search530_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search530_0_0.value },
   NULL,
};

static const nir_search_variable search530_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search530 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search530_0.value, &search530_1.value },
   NULL,
};
   
static const nir_search_constant replace530 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
   
static const nir_search_variable search534_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search534_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search534_0_0.value },
   NULL,
};

static const nir_search_variable search534_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search534_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search534_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search534_1_0.value, &search534_1_1.value },
   NULL,
};
static const nir_search_expression search534 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search534_0.value, &search534_1.value },
   NULL,
};
   
static const nir_search_variable replace534 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search535_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search535_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search535_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search535_1_0_0.value },
   NULL,
};

static const nir_search_variable search535_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search535_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search535_1_0.value, &search535_1_1.value },
   NULL,
};
static const nir_search_expression search535 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fadd,
   { &search535_0.value, &search535_1.value },
   NULL,
};
   
static const nir_search_variable replace535 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_before_ffma_fadd_xforms[] = {
   { &search526, &replace526.value, 0 },
   { &search528, &replace528.value, 0 },
   { &search530, &replace530.value, 0 },
   { &search534, &replace534.value, 0 },
   { &search535, &replace535.value, 0 },
};
   
static const nir_search_variable search527_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   (is_not_const),
};

static const nir_search_variable search527_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   true,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search527_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search527_0_0.value, &search527_0_1.value },
   (is_used_once),
};

static const nir_search_variable search527_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   (is_not_const),
};
static const nir_search_expression search527 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search527_0.value, &search527_1.value },
   (is_used_once),
};
   
static const nir_search_variable replace527_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace527_0_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace527_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace527_0_0.value, &replace527_0_1.value },
   NULL,
};

static const nir_search_variable replace527_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace527 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace527_0.value, &replace527_1.value },
   NULL,
};
   
static const nir_search_variable search529_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search529_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search529_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search529_0_0.value, &search529_0_1.value },
   NULL,
};

static const nir_search_variable search529_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search529_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search529_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &search529_1_0.value, &search529_1_1.value },
   NULL,
};
static const nir_search_expression search529 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search529_0.value, &search529_1.value },
   NULL,
};
   
static const nir_search_variable replace529_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace529_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace529_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace529_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &replace529_1_0.value, &replace529_1_1.value },
   NULL,
};
static const nir_search_expression replace529 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_imul,
   { &replace529_0.value, &replace529_1.value },
   NULL,
};
   
static const nir_search_variable search531_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search531_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search531_0_0.value },
   NULL,
};

static const nir_search_variable search531_1 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search531 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search531_0.value, &search531_1.value },
   NULL,
};
   
static const nir_search_constant replace531 = {
   { nir_search_value_constant, 0 },
   nir_type_int, { 0x0 /* 0 */ },
};
   
static const nir_search_variable search532_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search532_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search532_0_0.value },
   NULL,
};

static const nir_search_variable search532_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search532_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search532_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search532_1_0.value, &search532_1_1.value },
   NULL,
};
static const nir_search_expression search532 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search532_0.value, &search532_1.value },
   NULL,
};
   
static const nir_search_variable replace532 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
   
static const nir_search_variable search533_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search533_1_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search533_1_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_ineg,
   { &search533_1_0_0.value },
   NULL,
};

static const nir_search_variable search533_1_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search533_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search533_1_0.value, &search533_1_1.value },
   NULL,
};
static const nir_search_expression search533 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iadd,
   { &search533_0.value, &search533_1.value },
   NULL,
};
   
static const nir_search_variable replace533 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};

static const struct transform nir_opt_algebraic_before_ffma_iadd_xforms[] = {
   { &search527, &replace527.value, 0 },
   { &search529, &replace529.value, 0 },
   { &search531, &replace531.value, 0 },
   { &search532, &replace532.value, 0 },
   { &search533, &replace533.value, 0 },
};

static bool
nir_opt_algebraic_before_ffma_block(nir_builder *build, nir_block *block,
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
      case nir_op_fmul:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_before_ffma_fmul_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_before_ffma_fmul_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_imul:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_before_ffma_imul_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_before_ffma_imul_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fadd:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_before_ffma_fadd_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_before_ffma_fadd_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_iadd:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_before_ffma_iadd_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_before_ffma_iadd_xforms[i];
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
nir_opt_algebraic_before_ffma_impl(nir_function_impl *impl, const bool *condition_flags)
{
   bool progress = false;

   nir_builder build;
   nir_builder_init(&build, impl);

   nir_foreach_block_reverse(block, impl) {
      progress |= nir_opt_algebraic_before_ffma_block(&build, block, condition_flags);
   }

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);

   return progress;
}


bool
nir_opt_algebraic_before_ffma(nir_shader *shader)
{
   bool progress = false;
   bool condition_flags[44];
   const nir_shader_compiler_options *options = shader->options;
   (void) options;

   condition_flags[0] = true;
   condition_flags[1] = options->lower_idiv;
   condition_flags[2] = options->lower_flrp32;
   condition_flags[3] = options->lower_flrp64;
   condition_flags[4] = options->lower_ffract;
   condition_flags[5] = !options->lower_flrp32;
   condition_flags[6] = !options->lower_flrp64;
   condition_flags[7] = options->lower_ffma;
   condition_flags[8] = options->fuse_ffma;
   condition_flags[9] = !options->lower_fsat;
   condition_flags[10] = options->lower_fsat;
   condition_flags[11] = options->lower_scmp;
   condition_flags[12] = !options->lower_b2f;
   condition_flags[13] = options->lower_fpow;
   condition_flags[14] = !options->lower_fpow;
   condition_flags[15] = options->lower_fdiv;
   condition_flags[16] = options->lower_fsqrt;
   condition_flags[17] = !options->lower_fsqrt;
   condition_flags[18] = !options->lower_extract_byte;
   condition_flags[19] = !options->lower_extract_word;
   condition_flags[20] = options->lower_sub;
   condition_flags[21] = options->lower_negate;
   condition_flags[22] = options->lower_fmod32;
   condition_flags[23] = options->lower_fmod64;
   condition_flags[24] = options->lower_uadd_carry;
   condition_flags[25] = options->lower_usub_borrow;
   condition_flags[26] = options->lower_bitfield_insert;
   condition_flags[27] = options->lower_bitfield_insert_to_shifts;
   condition_flags[28] = options->lower_bfm;
   condition_flags[29] = options->lower_bitfield_extract;
   condition_flags[30] = options->lower_bitfield_extract_to_shifts;
   condition_flags[31] = options->lower_ifind_msb;
   condition_flags[32] = options->lower_find_lsb;
   condition_flags[33] = options->lower_extract_byte;
   condition_flags[34] = options->lower_extract_word;
   condition_flags[35] = options->lower_pack_unorm_2x16;
   condition_flags[36] = options->lower_pack_unorm_4x8;
   condition_flags[37] = options->lower_pack_snorm_2x16;
   condition_flags[38] = options->lower_pack_snorm_4x8;
   condition_flags[39] = options->lower_unpack_unorm_2x16;
   condition_flags[40] = options->lower_unpack_unorm_4x8;
   condition_flags[41] = options->lower_unpack_snorm_2x16;
   condition_flags[42] = options->lower_unpack_snorm_4x8;
   condition_flags[43] = options->lower_ldexp;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_opt_algebraic_before_ffma_impl(function->impl, condition_flags);
   }

   return progress;
}


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

   
static const nir_search_variable search536_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search536_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search536_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search536_0_0.value, &search536_0_1.value },
   NULL,
};

static const nir_search_constant search536_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search536 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search536_0.value, &search536_1.value },
   NULL,
};
   
static const nir_search_variable replace536_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace536_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace536_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace536_1_0.value },
   NULL,
};
static const nir_search_expression replace536 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace536_0.value, &replace536_1.value },
   NULL,
};
   
static const nir_search_variable search537_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search537_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search537_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search537_0_0_0.value, &search537_0_0_1.value },
   NULL,
};
static const nir_search_expression search537_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search537_0_0.value },
   NULL,
};

static const nir_search_constant search537_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search537 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &search537_0.value, &search537_1.value },
   NULL,
};
   
static const nir_search_variable replace537_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace537_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace537_0_0.value },
   NULL,
};

static const nir_search_variable replace537_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace537 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_flt,
   { &replace537_0.value, &replace537_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_flt_xforms[] = {
   { &search536, &replace536.value, 0 },
   { &search537, &replace537.value, 0 },
};
   
static const nir_search_variable search538_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search538_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search538_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search538_0_0.value, &search538_0_1.value },
   NULL,
};

static const nir_search_constant search538_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search538 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fge,
   { &search538_0.value, &search538_1.value },
   NULL,
};
   
static const nir_search_variable replace538_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace538_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace538_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace538_1_0.value },
   NULL,
};
static const nir_search_expression replace538 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace538_0.value, &replace538_1.value },
   NULL,
};
   
static const nir_search_variable search539_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search539_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search539_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search539_0_0_0.value, &search539_0_0_1.value },
   NULL,
};
static const nir_search_expression search539_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search539_0_0.value },
   NULL,
};

static const nir_search_constant search539_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search539 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fge,
   { &search539_0.value, &search539_1.value },
   NULL,
};
   
static const nir_search_variable replace539_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace539_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace539_0_0.value },
   NULL,
};

static const nir_search_variable replace539_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace539 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace539_0.value, &replace539_1.value },
   NULL,
};
   
static const nir_search_variable search542_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search542_0_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search542_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search542_0_0_0.value, &search542_0_0_1.value },
   (is_used_once),
};

static const nir_search_variable search542_0_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search542_0_1_1 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search542_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search542_0_1_0.value, &search542_0_1_1.value },
   NULL,
};
static const nir_search_expression search542_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search542_0_0.value, &search542_0_1.value },
   (is_used_once),
};

static const nir_search_constant search542_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search542 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fge,
   { &search542_0.value, &search542_1.value },
   NULL,
};
   
static const nir_search_variable replace542_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace542_0_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace542_0_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace542_0_1_0.value },
   NULL,
};
static const nir_search_expression replace542_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace542_0_0.value, &replace542_0_1.value },
   NULL,
};

static const nir_search_variable replace542_1_0 = {
   { nir_search_value_variable, 0 },
   2, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace542_1_1_0 = {
   { nir_search_value_variable, 0 },
   3, /* d */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace542_1_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace542_1_1_0.value },
   NULL,
};
static const nir_search_expression replace542_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fge,
   { &replace542_1_0.value, &replace542_1_1.value },
   NULL,
};
static const nir_search_expression replace542 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace542_0.value, &replace542_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_fge_xforms[] = {
   { &search538, &replace538.value, 0 },
   { &search539, &replace539.value, 0 },
   { &search542, &replace542.value, 0 },
};
   
static const nir_search_variable search540_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search540_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search540_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search540_0_0.value, &search540_0_1.value },
   NULL,
};

static const nir_search_constant search540_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search540 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_feq,
   { &search540_0.value, &search540_1.value },
   NULL,
};
   
static const nir_search_variable replace540_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace540_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace540_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace540_1_0.value },
   NULL,
};
static const nir_search_expression replace540 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_feq,
   { &replace540_0.value, &replace540_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_feq_xforms[] = {
   { &search540, &replace540.value, 0 },
};
   
static const nir_search_variable search541_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search541_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search541_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search541_0_0.value, &search541_0_1.value },
   NULL,
};

static const nir_search_constant search541_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};
static const nir_search_expression search541 = {
   { nir_search_value_expression, 0 },
   true,
   nir_op_fne,
   { &search541_0.value, &search541_1.value },
   NULL,
};
   
static const nir_search_variable replace541_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace541_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace541_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &replace541_1_0.value },
   NULL,
};
static const nir_search_expression replace541 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fne,
   { &replace541_0.value, &replace541_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_fne_xforms[] = {
   { &search541, &replace541.value, 0 },
};
   
static const nir_search_variable search543_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search543_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search543 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot2,
   { &search543_0.value, &search543_1.value },
   NULL,
};
   
static const nir_search_variable replace543_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace543_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace543 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot_replicated2,
   { &replace543_0.value, &replace543_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_fdot2_xforms[] = {
   { &search543, &replace543.value, 44 },
};
   
static const nir_search_variable search544_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search544_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search544 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot3,
   { &search544_0.value, &search544_1.value },
   NULL,
};
   
static const nir_search_variable replace544_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace544_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace544 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot_replicated3,
   { &replace544_0.value, &replace544_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_fdot3_xforms[] = {
   { &search544, &replace544.value, 44 },
};
   
static const nir_search_variable search545_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search545_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search545 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot4,
   { &search545_0.value, &search545_1.value },
   NULL,
};
   
static const nir_search_variable replace545_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace545_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace545 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdot_replicated4,
   { &replace545_0.value, &replace545_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_fdot4_xforms[] = {
   { &search545, &replace545.value, 44 },
};
   
static const nir_search_variable search546_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search546_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search546 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdph,
   { &search546_0.value, &search546_1.value },
   NULL,
};
   
static const nir_search_variable replace546_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace546_1 = {
   { nir_search_value_variable, 0 },
   1, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace546 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fdph_replicated,
   { &replace546_0.value, &replace546_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_fdph_xforms[] = {
   { &search546, &replace546.value, 44 },
};
   
static const nir_search_variable search547_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search547_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search547_0_0.value },
   NULL,
};
static const nir_search_expression search547 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search547_0.value },
   (is_used_more_than_once),
};
   
static const nir_search_variable replace547_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace547_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x0 /* 0.0 */ },
};

static const nir_search_constant replace547_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression replace547 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace547_0.value, &replace547_1.value, &replace547_2.value },
   NULL,
};
   
static const nir_search_variable search551_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search551 = {
   { nir_search_value_expression, 32 },
   false,
   nir_op_b2f,
   { &search551_0.value },
   NULL,
};
   
static const nir_search_variable replace551_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace551_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x3ff0000000000000 /* 1.0 */ },
};
static const nir_search_expression replace551 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_iand,
   { &replace551_0.value, &replace551_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_b2f_xforms[] = {
   { &search547, &replace547.value, 0 },
   { &search551, &replace551.value, 45 },
};
   
static const nir_search_variable search548_0_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search548_0_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_inot,
   { &search548_0_0_0.value },
   NULL,
};
static const nir_search_expression search548_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_b2f,
   { &search548_0_0.value },
   NULL,
};
static const nir_search_expression search548 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fneg,
   { &search548_0.value },
   (is_used_more_than_once),
};
   
static const nir_search_variable replace548_0 = {
   { nir_search_value_variable, 0 },
   0, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_constant replace548_1 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0x8000000000000000L /* -0.0 */ },
};

static const nir_search_constant replace548_2 = {
   { nir_search_value_constant, 0 },
   nir_type_float, { 0xbff0000000000000L /* -1.0 */ },
};
static const nir_search_expression replace548 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_bcsel,
   { &replace548_0.value, &replace548_1.value, &replace548_2.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_fneg_xforms[] = {
   { &search548, &replace548.value, 0 },
};
   
static const nir_search_variable search549_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* c */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search549_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search549_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search549_0_0.value, &search549_0_1.value },
   (is_used_once),
};

static const nir_search_variable search549_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* c */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search549_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search549_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search549_1_0.value, &search549_1_1.value },
   (is_used_once),
};
static const nir_search_expression search549 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &search549_0.value, &search549_1.value },
   NULL,
};
   
static const nir_search_variable replace549_0 = {
   { nir_search_value_variable, 0 },
   0, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace549_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace549_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace549_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmin,
   { &replace549_1_0.value, &replace549_1_1.value },
   NULL,
};
static const nir_search_expression replace549 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace549_0.value, &replace549_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_fmin_xforms[] = {
   { &search549, &replace549.value, 0 },
};
   
static const nir_search_variable search550_0_0 = {
   { nir_search_value_variable, 0 },
   0, /* c */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search550_0_1 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search550_0 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search550_0_0.value, &search550_0_1.value },
   (is_used_once),
};

static const nir_search_variable search550_1_0 = {
   { nir_search_value_variable, 0 },
   0, /* c */
   true,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable search550_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression search550_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &search550_1_0.value, &search550_1_1.value },
   (is_used_once),
};
static const nir_search_expression search550 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &search550_0.value, &search550_1.value },
   NULL,
};
   
static const nir_search_variable replace550_0 = {
   { nir_search_value_variable, 0 },
   0, /* c */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace550_1_0 = {
   { nir_search_value_variable, 0 },
   1, /* a */
   false,
   nir_type_invalid,
   NULL,
};

static const nir_search_variable replace550_1_1 = {
   { nir_search_value_variable, 0 },
   2, /* b */
   false,
   nir_type_invalid,
   NULL,
};
static const nir_search_expression replace550_1 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fmax,
   { &replace550_1_0.value, &replace550_1_1.value },
   NULL,
};
static const nir_search_expression replace550 = {
   { nir_search_value_expression, 0 },
   false,
   nir_op_fadd,
   { &replace550_0.value, &replace550_1.value },
   NULL,
};

static const struct transform nir_opt_algebraic_late_fmax_xforms[] = {
   { &search550, &replace550.value, 0 },
};

static bool
nir_opt_algebraic_late_block(nir_builder *build, nir_block *block,
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
      case nir_op_flt:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_flt_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_flt_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fge:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_fge_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_fge_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_feq:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_feq_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_feq_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fne:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_fne_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_fne_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fdot2:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_fdot2_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_fdot2_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fdot3:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_fdot3_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_fdot3_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fdot4:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_fdot4_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_fdot4_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fdph:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_fdph_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_fdph_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_b2f:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_b2f_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_b2f_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fneg:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_fneg_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_fneg_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fmin:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_fmin_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_fmin_xforms[i];
            if (condition_flags[xform->condition_offset] &&
                nir_replace_instr(build, alu, xform->search, xform->replace)) {
               progress = true;
               break;
            }
         }
         break;
      case nir_op_fmax:
         for (unsigned i = 0; i < ARRAY_SIZE(nir_opt_algebraic_late_fmax_xforms); i++) {
            const struct transform *xform = &nir_opt_algebraic_late_fmax_xforms[i];
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
nir_opt_algebraic_late_impl(nir_function_impl *impl, const bool *condition_flags)
{
   bool progress = false;

   nir_builder build;
   nir_builder_init(&build, impl);

   nir_foreach_block_reverse(block, impl) {
      progress |= nir_opt_algebraic_late_block(&build, block, condition_flags);
   }

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);

   return progress;
}


bool
nir_opt_algebraic_late(nir_shader *shader)
{
   bool progress = false;
   bool condition_flags[46];
   const nir_shader_compiler_options *options = shader->options;
   (void) options;

   condition_flags[0] = true;
   condition_flags[1] = options->lower_idiv;
   condition_flags[2] = options->lower_flrp32;
   condition_flags[3] = options->lower_flrp64;
   condition_flags[4] = options->lower_ffract;
   condition_flags[5] = !options->lower_flrp32;
   condition_flags[6] = !options->lower_flrp64;
   condition_flags[7] = options->lower_ffma;
   condition_flags[8] = options->fuse_ffma;
   condition_flags[9] = !options->lower_fsat;
   condition_flags[10] = options->lower_fsat;
   condition_flags[11] = options->lower_scmp;
   condition_flags[12] = !options->lower_b2f;
   condition_flags[13] = options->lower_fpow;
   condition_flags[14] = !options->lower_fpow;
   condition_flags[15] = options->lower_fdiv;
   condition_flags[16] = options->lower_fsqrt;
   condition_flags[17] = !options->lower_fsqrt;
   condition_flags[18] = !options->lower_extract_byte;
   condition_flags[19] = !options->lower_extract_word;
   condition_flags[20] = options->lower_sub;
   condition_flags[21] = options->lower_negate;
   condition_flags[22] = options->lower_fmod32;
   condition_flags[23] = options->lower_fmod64;
   condition_flags[24] = options->lower_uadd_carry;
   condition_flags[25] = options->lower_usub_borrow;
   condition_flags[26] = options->lower_bitfield_insert;
   condition_flags[27] = options->lower_bitfield_insert_to_shifts;
   condition_flags[28] = options->lower_bfm;
   condition_flags[29] = options->lower_bitfield_extract;
   condition_flags[30] = options->lower_bitfield_extract_to_shifts;
   condition_flags[31] = options->lower_ifind_msb;
   condition_flags[32] = options->lower_find_lsb;
   condition_flags[33] = options->lower_extract_byte;
   condition_flags[34] = options->lower_extract_word;
   condition_flags[35] = options->lower_pack_unorm_2x16;
   condition_flags[36] = options->lower_pack_unorm_4x8;
   condition_flags[37] = options->lower_pack_snorm_2x16;
   condition_flags[38] = options->lower_pack_snorm_4x8;
   condition_flags[39] = options->lower_unpack_unorm_2x16;
   condition_flags[40] = options->lower_unpack_unorm_4x8;
   condition_flags[41] = options->lower_unpack_snorm_2x16;
   condition_flags[42] = options->lower_unpack_snorm_4x8;
   condition_flags[43] = options->lower_ldexp;
   condition_flags[44] = options->fdot_replicates;
   condition_flags[45] = options->lower_b2f;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_opt_algebraic_late_impl(function->impl, condition_flags);
   }

   return progress;
}

