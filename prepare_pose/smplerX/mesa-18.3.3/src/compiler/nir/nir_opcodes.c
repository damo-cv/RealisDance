
#include "nir.h"

nir_op
nir_type_conversion_op(nir_alu_type src, nir_alu_type dst, nir_rounding_mode rnd)
{
   nir_alu_type src_base = (nir_alu_type) nir_alu_type_get_base_type(src);
   nir_alu_type dst_base = (nir_alu_type) nir_alu_type_get_base_type(dst);
   unsigned src_bit_size = nir_alu_type_get_type_size(src);
   unsigned dst_bit_size = nir_alu_type_get_type_size(dst);

   if (src == dst && src_base == nir_type_float) {
      return nir_op_fmov;
   } else if ((src_base == nir_type_int || src_base == nir_type_uint) &&
              (dst_base == nir_type_int || dst_base == nir_type_uint) &&
              src_bit_size == dst_bit_size) {
      /* Integer <-> integer conversions with the same bit-size on both
       * ends are just no-op moves.
       */
      return nir_op_imov;
   }

   switch (src_base) {
      case nir_type_int:
         switch (dst_base) {
            case nir_type_int:
            case nir_type_uint:

               switch (dst_bit_size) {

                  case 8:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_i2i8;
                  case 16:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_i2i16;
                  case 32:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_i2i32;
                  case 64:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_i2i64;
                  default:
                     unreachable("Invalid nir alu bit size");
               }
            case nir_type_float:
               switch (dst_bit_size) {

                  case 16:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_i2f16;
                  case 32:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_i2f32;
                  case 64:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_i2f64;
                  default:
                     unreachable("Invalid nir alu bit size");
               }
            case nir_type_bool:
                  return nir_op_i2b;
            default:
               unreachable("Invalid nir alu base type");
         }
      case nir_type_uint:
         switch (dst_base) {
            case nir_type_int:
            case nir_type_uint:

               switch (dst_bit_size) {

                  case 8:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_u2u8;
                  case 16:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_u2u16;
                  case 32:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_u2u32;
                  case 64:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_u2u64;
                  default:
                     unreachable("Invalid nir alu bit size");
               }
            case nir_type_float:
               switch (dst_bit_size) {

                  case 16:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_u2f16;
                  case 32:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_u2f32;
                  case 64:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_u2f64;
                  default:
                     unreachable("Invalid nir alu bit size");
               }
            case nir_type_bool:
                  return nir_op_i2b;
            default:
               unreachable("Invalid nir alu base type");
         }
      case nir_type_float:
         switch (dst_base) {
            case nir_type_int:
               switch (dst_bit_size) {

                  case 8:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_f2i8;
                  case 16:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_f2i16;
                  case 32:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_f2i32;
                  case 64:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_f2i64;
                  default:
                     unreachable("Invalid nir alu bit size");
               }
            case nir_type_uint:
               switch (dst_bit_size) {

                  case 8:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_f2u8;
                  case 16:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_f2u16;
                  case 32:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_f2u32;
                  case 64:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_f2u64;
                  default:
                     unreachable("Invalid nir alu bit size");
               }
            case nir_type_float:
               switch (dst_bit_size) {

                  case 16:
                     switch(rnd) {
                        case nir_rounding_mode_rtne:
                           return nir_op_f2f16_rtne;
                        case nir_rounding_mode_rtz:
                           return nir_op_f2f16_rtz;
                        case nir_rounding_mode_undef:
                           return nir_op_f2f16;
                        default:
                           unreachable("Invalid 16-bit nir rounding mode");
                     }
                  case 32:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_f2f32;
                  case 64:
                     assert(rnd == nir_rounding_mode_undef);
                     return nir_op_f2f64;
                  default:
                     unreachable("Invalid nir alu bit size");
               }
            case nir_type_bool:
                  return nir_op_f2b;
            default:
               unreachable("Invalid nir alu base type");
         }
      case nir_type_bool:
         switch (dst_base) {
            case nir_type_int:
            case nir_type_uint:
               return nir_op_b2i;
            case nir_type_float:
               return nir_op_b2f;
            default:
               unreachable("Invalid nir alu base type");
         }
      default:
         unreachable("Invalid nir alu base type");
   }
}

const nir_op_info nir_op_infos[nir_num_opcodes] = {
{
   .name = "b2f",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_bool32
   },
   .algebraic_properties =
      0
},
{
   .name = "b2i",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_bool32
   },
   .algebraic_properties =
      0
},
{
   .name = "ball_fequal2",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      2, 2
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "ball_fequal3",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      3, 3
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "ball_fequal4",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      4, 4
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "ball_iequal2",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      2, 2
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "ball_iequal3",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      3, 3
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "ball_iequal4",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      4, 4
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "bany_fnequal2",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      2, 2
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "bany_fnequal3",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      3, 3
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "bany_fnequal4",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      4, 4
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "bany_inequal2",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      2, 2
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "bany_inequal3",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      3, 3
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "bany_inequal4",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_bool32,
   .input_sizes = {
      4, 4
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "bcsel",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_bool32, nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "bfi",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_uint32, nir_type_uint32, nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "bfm",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      0
},
{
   .name = "bit_count",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "bitfield_insert",
   .num_inputs = 4,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0, 0, 0, 0
   },
   .input_types = {
      nir_type_uint32, nir_type_uint32, nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      0
},
{
   .name = "bitfield_reverse",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "cube_face_coord",
   .num_inputs = 1,
   .output_size = 2,
   .output_type = nir_type_float32,
   .input_sizes = {
      3
   },
   .input_types = {
      nir_type_float32
   },
   .algebraic_properties =
      0
},
{
   .name = "cube_face_index",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_float32,
   .input_sizes = {
      3
   },
   .input_types = {
      nir_type_float32
   },
   .algebraic_properties =
      0
},
{
   .name = "extract_i16",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "extract_i8",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "extract_u16",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "extract_u8",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "f2b",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2f16",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2f16_rtne",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2f16_rtz",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2f32",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2f64",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float64,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2i16",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2i32",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2i64",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int64,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2i8",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int8,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2u16",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2u32",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2u64",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint64,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "f2u8",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint8,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fabs",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fadd",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "fall_equal2",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_float32,
   .input_sizes = {
      2, 2
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fall_equal3",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_float32,
   .input_sizes = {
      3, 3
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fall_equal4",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_float32,
   .input_sizes = {
      4, 4
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fand",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fany_nequal2",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_float32,
   .input_sizes = {
      2, 2
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fany_nequal3",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_float32,
   .input_sizes = {
      3, 3
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fany_nequal4",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_float32,
   .input_sizes = {
      4, 4
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fceil",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fcos",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fcsel",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_float32, nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      0
},
{
   .name = "fddx",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fddx_coarse",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fddx_fine",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fddy",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fddy_coarse",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fddy_fine",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fdiv",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fdot2",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_float,
   .input_sizes = {
      2, 2
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fdot3",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_float,
   .input_sizes = {
      3, 3
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fdot4",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_float,
   .input_sizes = {
      4, 4
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fdot_replicated2",
   .num_inputs = 2,
   .output_size = 4,
   .output_type = nir_type_float,
   .input_sizes = {
      2, 2
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fdot_replicated3",
   .num_inputs = 2,
   .output_size = 4,
   .output_type = nir_type_float,
   .input_sizes = {
      3, 3
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fdot_replicated4",
   .num_inputs = 2,
   .output_size = 4,
   .output_type = nir_type_float,
   .input_sizes = {
      4, 4
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fdph",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_float,
   .input_sizes = {
      3, 4
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fdph_replicated",
   .num_inputs = 2,
   .output_size = 4,
   .output_type = nir_type_float,
   .input_sizes = {
      3, 4
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "feq",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fexp2",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "ffloor",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "ffma",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "ffract",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fge",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "find_lsb",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "flog2",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "flrp",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "flt",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fmax",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fmax3",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fmed3",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fmin",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fmin3",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fmod",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fmov",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fmul",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "fne",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fneg",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise1_1",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_float,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise1_2",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_float,
   .input_sizes = {
      2
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise1_3",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_float,
   .input_sizes = {
      3
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise1_4",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_float,
   .input_sizes = {
      4
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise2_1",
   .num_inputs = 1,
   .output_size = 2,
   .output_type = nir_type_float,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise2_2",
   .num_inputs = 1,
   .output_size = 2,
   .output_type = nir_type_float,
   .input_sizes = {
      2
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise2_3",
   .num_inputs = 1,
   .output_size = 2,
   .output_type = nir_type_float,
   .input_sizes = {
      3
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise2_4",
   .num_inputs = 1,
   .output_size = 2,
   .output_type = nir_type_float,
   .input_sizes = {
      4
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise3_1",
   .num_inputs = 1,
   .output_size = 3,
   .output_type = nir_type_float,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise3_2",
   .num_inputs = 1,
   .output_size = 3,
   .output_type = nir_type_float,
   .input_sizes = {
      2
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise3_3",
   .num_inputs = 1,
   .output_size = 3,
   .output_type = nir_type_float,
   .input_sizes = {
      3
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise3_4",
   .num_inputs = 1,
   .output_size = 3,
   .output_type = nir_type_float,
   .input_sizes = {
      4
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise4_1",
   .num_inputs = 1,
   .output_size = 4,
   .output_type = nir_type_float,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise4_2",
   .num_inputs = 1,
   .output_size = 4,
   .output_type = nir_type_float,
   .input_sizes = {
      2
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise4_3",
   .num_inputs = 1,
   .output_size = 4,
   .output_type = nir_type_float,
   .input_sizes = {
      3
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnoise4_4",
   .num_inputs = 1,
   .output_size = 4,
   .output_type = nir_type_float,
   .input_sizes = {
      4
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fnot",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "for",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "fpow",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fquantize2f16",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "frcp",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "frem",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "frexp_exp",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float64
   },
   .algebraic_properties =
      0
},
{
   .name = "frexp_sig",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float64,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float64
   },
   .algebraic_properties =
      0
},
{
   .name = "fround_even",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "frsq",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fsat",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fsign",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fsin",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fsqrt",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fsub",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "ftrunc",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "fxor",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "i2b",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "i2f16",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "i2f32",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "i2f64",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float64,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "i2i16",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "i2i32",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "i2i64",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int64,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "i2i8",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int8,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "iabs",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "iadd",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "iand",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "ibfe",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_int32, nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      0
},
{
   .name = "ibitfield_extract",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_int32, nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      0
},
{
   .name = "idiv",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "ieq",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "ifind_msb",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int32
   },
   .algebraic_properties =
      0
},
{
   .name = "ige",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "ilt",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "imax",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "imax3",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "imed3",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "imin",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "imin3",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "imod",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "imov",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "imul",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "imul_high",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "ine",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "ineg",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "inot",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "ior",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "irem",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "ishl",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "ishr",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "isign",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "isub",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int, nir_type_int
   },
   .algebraic_properties =
      0
},
{
   .name = "ixor",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "ldexp",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_int32
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_32_2x16",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_uint32,
   .input_sizes = {
      2
   },
   .input_types = {
      nir_type_uint16
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_32_2x16_split",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint16, nir_type_uint16
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_64_2x32",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_uint64,
   .input_sizes = {
      2
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_64_2x32_split",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint64,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint32, nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_64_4x16",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_uint64,
   .input_sizes = {
      4
   },
   .input_types = {
      nir_type_uint16
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_half_2x16",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_uint32,
   .input_sizes = {
      2
   },
   .input_types = {
      nir_type_float32
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_half_2x16_split",
   .num_inputs = 2,
   .output_size = 1,
   .output_type = nir_type_uint32,
   .input_sizes = {
      1, 1
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_snorm_2x16",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_uint32,
   .input_sizes = {
      2
   },
   .input_types = {
      nir_type_float32
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_snorm_4x8",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_uint32,
   .input_sizes = {
      4
   },
   .input_types = {
      nir_type_float32
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_unorm_2x16",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_uint32,
   .input_sizes = {
      2
   },
   .input_types = {
      nir_type_float32
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_unorm_4x8",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_uint32,
   .input_sizes = {
      4
   },
   .input_types = {
      nir_type_float32
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_uvec2_to_uint",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_uint32,
   .input_sizes = {
      2
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "pack_uvec4_to_uint",
   .num_inputs = 1,
   .output_size = 1,
   .output_type = nir_type_uint32,
   .input_sizes = {
      4
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "seq",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "sge",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float, nir_type_float
   },
   .algebraic_properties =
      0
},
{
   .name = "slt",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      0
},
{
   .name = "sne",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_float32, nir_type_float32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "u2f16",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "u2f32",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "u2f64",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float64,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "u2u16",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "u2u32",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "u2u64",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint64,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "u2u8",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint8,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "uadd_carry",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "ubfe",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_uint32, nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      0
},
{
   .name = "ubitfield_extract",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_uint32, nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      0
},
{
   .name = "udiv",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "ufind_msb",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "uge",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "ult",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_bool32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "umax",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "umax3",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "umax_4x8",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "umed3",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "umin",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "umin3",
   .num_inputs = 3,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "umin_4x8",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "umod",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "umul_high",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint32, nir_type_uint32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE
},
{
   .name = "umul_unorm_4x8",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "unpack_32_2x16",
   .num_inputs = 1,
   .output_size = 2,
   .output_type = nir_type_uint16,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_32_2x16_split_x",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_32_2x16_split_y",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint16,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_64_2x32",
   .num_inputs = 1,
   .output_size = 2,
   .output_type = nir_type_uint32,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_uint64
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_64_2x32_split_x",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint64
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_64_2x32_split_y",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_uint32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint64
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_64_4x16",
   .num_inputs = 1,
   .output_size = 4,
   .output_type = nir_type_uint16,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_uint64
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_half_2x16",
   .num_inputs = 1,
   .output_size = 2,
   .output_type = nir_type_float32,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_half_2x16_split_x",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_half_2x16_split_y",
   .num_inputs = 1,
   .output_size = 0,
   .output_type = nir_type_float32,
   .input_sizes = {
      0
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_snorm_2x16",
   .num_inputs = 1,
   .output_size = 2,
   .output_type = nir_type_float32,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_snorm_4x8",
   .num_inputs = 1,
   .output_size = 4,
   .output_type = nir_type_float32,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_unorm_2x16",
   .num_inputs = 1,
   .output_size = 2,
   .output_type = nir_type_float32,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "unpack_unorm_4x8",
   .num_inputs = 1,
   .output_size = 4,
   .output_type = nir_type_float32,
   .input_sizes = {
      1
   },
   .input_types = {
      nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "usadd_4x8",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      NIR_OP_IS_COMMUTATIVE | NIR_OP_IS_ASSOCIATIVE
},
{
   .name = "ushr",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint32
   },
   .algebraic_properties =
      0
},
{
   .name = "ussub_4x8",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_int32,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_int32, nir_type_int32
   },
   .algebraic_properties =
      0
},
{
   .name = "usub_borrow",
   .num_inputs = 2,
   .output_size = 0,
   .output_type = nir_type_uint,
   .input_sizes = {
      0, 0
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "vec2",
   .num_inputs = 2,
   .output_size = 2,
   .output_type = nir_type_uint,
   .input_sizes = {
      1, 1
   },
   .input_types = {
      nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "vec3",
   .num_inputs = 3,
   .output_size = 3,
   .output_type = nir_type_uint,
   .input_sizes = {
      1, 1, 1
   },
   .input_types = {
      nir_type_uint, nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
{
   .name = "vec4",
   .num_inputs = 4,
   .output_size = 4,
   .output_type = nir_type_uint,
   .input_sizes = {
      1, 1, 1, 1
   },
   .input_types = {
      nir_type_uint, nir_type_uint, nir_type_uint, nir_type_uint
   },
   .algebraic_properties =
      0
},
};

