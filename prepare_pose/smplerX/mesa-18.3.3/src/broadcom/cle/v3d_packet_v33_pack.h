/* Generated code, see packets.xml and gen_packet_header.py */


/* Packets, enums and structures for V3D 3.3.
 *
 * This file has been generated, do not hand edit.
 */

#ifndef V3D33_PACK_H
#define V3D33_PACK_H

#include "cle/v3d_packet_helpers.h"


enum V3D33_Compare_Function {
        V3D_COMPARE_FUNC_NEVER               =      0,
        V3D_COMPARE_FUNC_LESS                =      1,
        V3D_COMPARE_FUNC_EQUAL               =      2,
        V3D_COMPARE_FUNC_LEQUAL              =      3,
        V3D_COMPARE_FUNC_GREATER             =      4,
        V3D_COMPARE_FUNC_NOTEQUAL            =      5,
        V3D_COMPARE_FUNC_GEQUAL              =      6,
        V3D_COMPARE_FUNC_ALWAYS              =      7,
};

enum V3D33_Blend_Factor {
        V3D_BLEND_FACTOR_ZERO                =      0,
        V3D_BLEND_FACTOR_ONE                 =      1,
        V3D_BLEND_FACTOR_SRC_COLOR           =      2,
        V3D_BLEND_FACTOR_INV_SRC_COLOR       =      3,
        V3D_BLEND_FACTOR_DST_COLOR           =      4,
        V3D_BLEND_FACTOR_INV_DST_COLOR       =      5,
        V3D_BLEND_FACTOR_SRC_ALPHA           =      6,
        V3D_BLEND_FACTOR_INV_SRC_ALPHA       =      7,
        V3D_BLEND_FACTOR_DST_ALPHA           =      8,
        V3D_BLEND_FACTOR_INV_DST_ALPHA       =      9,
        V3D_BLEND_FACTOR_CONST_COLOR         =     10,
        V3D_BLEND_FACTOR_INV_CONST_COLOR     =     11,
        V3D_BLEND_FACTOR_CONST_ALPHA         =     12,
        V3D_BLEND_FACTOR_INV_CONST_ALPHA     =     13,
        V3D_BLEND_FACTOR_SRC_ALPHA_SATURATE  =     14,
};

enum V3D33_Blend_Mode {
        V3D_BLEND_MODE_ADD                   =      0,
        V3D_BLEND_MODE_SUB                   =      1,
        V3D_BLEND_MODE_RSUB                  =      2,
        V3D_BLEND_MODE_MIN                   =      3,
        V3D_BLEND_MODE_MAX                   =      4,
        V3D_BLEND_MODE_MUL                   =      5,
        V3D_BLEND_MODE_SCREEN                =      6,
        V3D_BLEND_MODE_DARKEN                =      7,
        V3D_BLEND_MODE_LIGHTEN               =      8,
};

enum V3D33_Stencil_Op {
        V3D_STENCIL_OP_ZERO                  =      0,
        V3D_STENCIL_OP_KEEP                  =      1,
        V3D_STENCIL_OP_REPLACE               =      2,
        V3D_STENCIL_OP_INCR                  =      3,
        V3D_STENCIL_OP_DECR                  =      4,
        V3D_STENCIL_OP_INVERT                =      5,
        V3D_STENCIL_OP_INCWRAP               =      6,
        V3D_STENCIL_OP_DECWRAP               =      7,
};

enum V3D33_Primitive {
        V3D_PRIM_POINTS                      =      0,
        V3D_PRIM_LINES                       =      1,
        V3D_PRIM_LINE_LOOP                   =      2,
        V3D_PRIM_LINE_STRIP                  =      3,
        V3D_PRIM_TRIANGLES                   =      4,
        V3D_PRIM_TRIANGLE_STRIP              =      5,
        V3D_PRIM_TRIANGLE_FAN                =      6,
        V3D_PRIM_POINTS_TF                   =     16,
        V3D_PRIM_LINES_TF                    =     17,
        V3D_PRIM_LINE_LOOP_TF                =     18,
        V3D_PRIM_LINE_STRIP_TF               =     19,
        V3D_PRIM_TRIANGLES_TF                =     20,
        V3D_PRIM_TRIANGLE_STRIP_TF           =     21,
        V3D_PRIM_TRIANGLE_FAN_TF             =     22,
};

enum V3D33_TMU_Filter {
        V3D_TMU_FILTER_MIN_LIN_MIP_NONE_MAG_LIN =      0,
        V3D_TMU_FILTER_MIN_LIN_MIP_NONE_MAG_NEAR =      1,
        V3D_TMU_FILTER_MIN_NEAR_MIP_NONE_MAG_LIN =      2,
        V3D_TMU_FILTER_MIN_NEAR_MIP_NONE_MAG_NEAR =      3,
        V3D_TMU_FILTER_MIN_NEAR_MIP_NEAR_MAG_LIN =      4,
        V3D_TMU_FILTER_MIN_NEAR_MIP_NEAR_MAG_NEAR =      5,
        V3D_TMU_FILTER_MIN_NEAR_MIP_LIN_MAG_LIN =      6,
        V3D_TMU_FILTER_MIN_NEAR_MIP_LIN_MAG_NEAR =      7,
        V3D_TMU_FILTER_MIN_LIN_MIP_NEAR_MAG_LIN =      8,
        V3D_TMU_FILTER_MIN_LIN_MIP_NEAR_MAG_NEAR =      9,
        V3D_TMU_FILTER_MIN_LIN_MIP_LIN_MAG_LIN =     10,
        V3D_TMU_FILTER_MIN_LIN_MIP_LIN_MAG_NEAR =     11,
        V3D_TMU_FILTER_ANISOTROPIC_2_1       =     12,
        V3D_TMU_FILTER_ANISOTROPIC_4_1       =     13,
        V3D_TMU_FILTER_ANISOTROPIC_8_1       =     14,
        V3D_TMU_FILTER_ANISOTROPIC_16_1      =     15,
};

enum V3D33_Varying_Flags_Action {
        V3D_VARYING_FLAGS_ACTION_UNCHANGED   =      0,
        V3D_VARYING_FLAGS_ACTION_ZEROED      =      1,
        V3D_VARYING_FLAGS_ACTION_SET         =      2,
};

enum V3D33_Memory_Format {
        V3D_MEMORY_FORMAT_RASTER             =      0,
        V3D_MEMORY_FORMAT_LINEARTILE         =      1,
        V3D_MEMORY_FORMAT_UB_LINEAR_1_UIF_BLOCK_WIDE =      2,
        V3D_MEMORY_FORMAT_UB_LINEAR_2_UIF_BLOCKS_WIDE =      3,
        V3D_MEMORY_FORMAT_UIF_NO_XOR         =      4,
        V3D_MEMORY_FORMAT_UIF_XOR            =      5,
};

enum V3D33_Decimate_Mode {
        V3D_DECIMATE_MODE_SAMPLE_0           =      0,
        V3D_DECIMATE_MODE_4X                 =      1,
        V3D_DECIMATE_MODE_ALL_SAMPLES        =      3,
};

enum V3D33_Internal_Type {
        V3D_INTERNAL_TYPE_8I                 =      0,
        V3D_INTERNAL_TYPE_8UI                =      1,
        V3D_INTERNAL_TYPE_8                  =      2,
        V3D_INTERNAL_TYPE_16I                =      4,
        V3D_INTERNAL_TYPE_16UI               =      5,
        V3D_INTERNAL_TYPE_16F                =      6,
        V3D_INTERNAL_TYPE_32I                =      8,
        V3D_INTERNAL_TYPE_32UI               =      9,
        V3D_INTERNAL_TYPE_32F                =     10,
};

enum V3D33_Internal_BPP {
        V3D_INTERNAL_BPP_32                  =      0,
        V3D_INTERNAL_BPP_64                  =      1,
        V3D_INTERNAL_BPP_128                 =      2,
};

enum V3D33_Internal_Depth_Type {
        V3D_INTERNAL_TYPE_DEPTH_32F          =      0,
        V3D_INTERNAL_TYPE_DEPTH_24           =      1,
        V3D_INTERNAL_TYPE_DEPTH_16           =      2,
};

enum V3D33_Output_Image_Format {
        V3D_OUTPUT_IMAGE_FORMAT_SRGB8_ALPHA8 =      0,
        V3D_OUTPUT_IMAGE_FORMAT_SRGB         =      1,
        V3D_OUTPUT_IMAGE_FORMAT_RGB10_A2UI   =      2,
        V3D_OUTPUT_IMAGE_FORMAT_RGB10_A2     =      3,
        V3D_OUTPUT_IMAGE_FORMAT_ABGR1555     =      4,
        V3D_OUTPUT_IMAGE_FORMAT_ALPHA_MASKED_ABGR1555 =      5,
        V3D_OUTPUT_IMAGE_FORMAT_ABGR4444     =      6,
        V3D_OUTPUT_IMAGE_FORMAT_BGR565       =      7,
        V3D_OUTPUT_IMAGE_FORMAT_R11F_G11F_B10F =      8,
        V3D_OUTPUT_IMAGE_FORMAT_RGBA32F      =      9,
        V3D_OUTPUT_IMAGE_FORMAT_RG32F        =     10,
        V3D_OUTPUT_IMAGE_FORMAT_R32F         =     11,
        V3D_OUTPUT_IMAGE_FORMAT_RGBA32I      =     12,
        V3D_OUTPUT_IMAGE_FORMAT_RG32I        =     13,
        V3D_OUTPUT_IMAGE_FORMAT_R32I         =     14,
        V3D_OUTPUT_IMAGE_FORMAT_RGBA32UI     =     15,
        V3D_OUTPUT_IMAGE_FORMAT_RG32UI       =     16,
        V3D_OUTPUT_IMAGE_FORMAT_R32UI        =     17,
        V3D_OUTPUT_IMAGE_FORMAT_RGBA16F      =     18,
        V3D_OUTPUT_IMAGE_FORMAT_RG16F        =     19,
        V3D_OUTPUT_IMAGE_FORMAT_R16F         =     20,
        V3D_OUTPUT_IMAGE_FORMAT_RGBA16I      =     21,
        V3D_OUTPUT_IMAGE_FORMAT_RG16I        =     22,
        V3D_OUTPUT_IMAGE_FORMAT_R16I         =     23,
        V3D_OUTPUT_IMAGE_FORMAT_RGBA16UI     =     24,
        V3D_OUTPUT_IMAGE_FORMAT_RG16UI       =     25,
        V3D_OUTPUT_IMAGE_FORMAT_R16UI        =     26,
        V3D_OUTPUT_IMAGE_FORMAT_RGBA8        =     27,
        V3D_OUTPUT_IMAGE_FORMAT_RGB8         =     28,
        V3D_OUTPUT_IMAGE_FORMAT_RG8          =     29,
        V3D_OUTPUT_IMAGE_FORMAT_R8           =     30,
        V3D_OUTPUT_IMAGE_FORMAT_RGBA8I       =     31,
        V3D_OUTPUT_IMAGE_FORMAT_RG8I         =     32,
        V3D_OUTPUT_IMAGE_FORMAT_R8I          =     33,
        V3D_OUTPUT_IMAGE_FORMAT_RGBA8UI      =     34,
        V3D_OUTPUT_IMAGE_FORMAT_RG8UI        =     35,
        V3D_OUTPUT_IMAGE_FORMAT_R8UI         =     36,
        V3D_OUTPUT_IMAGE_FORMAT_SRGBX8       =     37,
        V3D_OUTPUT_IMAGE_FORMAT_RGBX8        =     38,
};

enum V3D33_Z_S_Output_Image_Format {
        V3D_OUTPUT_IMAGE_FORMAT_ZS_DEPTH_COMPONENT32F =      0,
        V3D_OUTPUT_IMAGE_FORMAT_ZS_DEPTH_COMPONENT24 =      1,
        V3D_OUTPUT_IMAGE_FORMAT_ZS_DEPTH_COMPONENT16 =      2,
        V3D_OUTPUT_IMAGE_FORMAT_ZS_DEPTH24_STENCIL8 =      3,
};

enum V3D33_Dither_Mode {
        V3D_DITHER_MODE_NONE                 =      0,
        V3D_DITHER_MODE_RGB                  =      1,
        V3D_DITHER_MODE_A                    =      2,
        V3D_DITHER_MODE_RGBA                 =      3,
};

#define V3D33_HALT_opcode                      0
#define V3D33_HALT_header                       \
   .opcode                              =      0

struct V3D33_HALT {
   uint32_t                             opcode;
};

static inline void
V3D33_HALT_pack(__gen_user_data *data, uint8_t * restrict cl,
                const struct V3D33_HALT * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_HALT_length                      1
#ifdef __gen_unpack_address
static inline void
V3D33_HALT_unpack(const uint8_t * restrict cl,
                  struct V3D33_HALT * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_NOP_opcode                       1
#define V3D33_NOP_header                        \
   .opcode                              =      1

struct V3D33_NOP {
   uint32_t                             opcode;
};

static inline void
V3D33_NOP_pack(__gen_user_data *data, uint8_t * restrict cl,
               const struct V3D33_NOP * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_NOP_length                       1
#ifdef __gen_unpack_address
static inline void
V3D33_NOP_unpack(const uint8_t * restrict cl,
                 struct V3D33_NOP * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_FLUSH_opcode                     4
#define V3D33_FLUSH_header                      \
   .opcode                              =      4

struct V3D33_FLUSH {
   uint32_t                             opcode;
};

static inline void
V3D33_FLUSH_pack(__gen_user_data *data, uint8_t * restrict cl,
                 const struct V3D33_FLUSH * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_FLUSH_length                     1
#ifdef __gen_unpack_address
static inline void
V3D33_FLUSH_unpack(const uint8_t * restrict cl,
                   struct V3D33_FLUSH * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_FLUSH_ALL_STATE_opcode           5
#define V3D33_FLUSH_ALL_STATE_header            \
   .opcode                              =      5

struct V3D33_FLUSH_ALL_STATE {
   uint32_t                             opcode;
};

static inline void
V3D33_FLUSH_ALL_STATE_pack(__gen_user_data *data, uint8_t * restrict cl,
                           const struct V3D33_FLUSH_ALL_STATE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_FLUSH_ALL_STATE_length           1
#ifdef __gen_unpack_address
static inline void
V3D33_FLUSH_ALL_STATE_unpack(const uint8_t * restrict cl,
                             struct V3D33_FLUSH_ALL_STATE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_START_TILE_BINNING_opcode        6
#define V3D33_START_TILE_BINNING_header         \
   .opcode                              =      6

struct V3D33_START_TILE_BINNING {
   uint32_t                             opcode;
};

static inline void
V3D33_START_TILE_BINNING_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D33_START_TILE_BINNING * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_START_TILE_BINNING_length        1
#ifdef __gen_unpack_address
static inline void
V3D33_START_TILE_BINNING_unpack(const uint8_t * restrict cl,
                                struct V3D33_START_TILE_BINNING * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_INCREMENT_SEMAPHORE_opcode       7
#define V3D33_INCREMENT_SEMAPHORE_header        \
   .opcode                              =      7

struct V3D33_INCREMENT_SEMAPHORE {
   uint32_t                             opcode;
};

static inline void
V3D33_INCREMENT_SEMAPHORE_pack(__gen_user_data *data, uint8_t * restrict cl,
                               const struct V3D33_INCREMENT_SEMAPHORE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_INCREMENT_SEMAPHORE_length       1
#ifdef __gen_unpack_address
static inline void
V3D33_INCREMENT_SEMAPHORE_unpack(const uint8_t * restrict cl,
                                 struct V3D33_INCREMENT_SEMAPHORE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_WAIT_ON_SEMAPHORE_opcode         8
#define V3D33_WAIT_ON_SEMAPHORE_header          \
   .opcode                              =      8

struct V3D33_WAIT_ON_SEMAPHORE {
   uint32_t                             opcode;
};

static inline void
V3D33_WAIT_ON_SEMAPHORE_pack(__gen_user_data *data, uint8_t * restrict cl,
                             const struct V3D33_WAIT_ON_SEMAPHORE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_WAIT_ON_SEMAPHORE_length         1
#ifdef __gen_unpack_address
static inline void
V3D33_WAIT_ON_SEMAPHORE_unpack(const uint8_t * restrict cl,
                               struct V3D33_WAIT_ON_SEMAPHORE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_WAIT_FOR_PREVIOUS_FRAME_opcode      9
#define V3D33_WAIT_FOR_PREVIOUS_FRAME_header    \
   .opcode                              =      9

struct V3D33_WAIT_FOR_PREVIOUS_FRAME {
   uint32_t                             opcode;
};

static inline void
V3D33_WAIT_FOR_PREVIOUS_FRAME_pack(__gen_user_data *data, uint8_t * restrict cl,
                                   const struct V3D33_WAIT_FOR_PREVIOUS_FRAME * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_WAIT_FOR_PREVIOUS_FRAME_length      1
#ifdef __gen_unpack_address
static inline void
V3D33_WAIT_FOR_PREVIOUS_FRAME_unpack(const uint8_t * restrict cl,
                                     struct V3D33_WAIT_FOR_PREVIOUS_FRAME * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_ENABLE_Z_ONLY_RENDERING_opcode     10
#define V3D33_ENABLE_Z_ONLY_RENDERING_header    \
   .opcode                              =     10

struct V3D33_ENABLE_Z_ONLY_RENDERING {
   uint32_t                             opcode;
};

static inline void
V3D33_ENABLE_Z_ONLY_RENDERING_pack(__gen_user_data *data, uint8_t * restrict cl,
                                   const struct V3D33_ENABLE_Z_ONLY_RENDERING * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_ENABLE_Z_ONLY_RENDERING_length      1
#ifdef __gen_unpack_address
static inline void
V3D33_ENABLE_Z_ONLY_RENDERING_unpack(const uint8_t * restrict cl,
                                     struct V3D33_ENABLE_Z_ONLY_RENDERING * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_DISABLE_Z_ONLY_RENDERING_opcode     11
#define V3D33_DISABLE_Z_ONLY_RENDERING_header   \
   .opcode                              =     11

struct V3D33_DISABLE_Z_ONLY_RENDERING {
   uint32_t                             opcode;
};

static inline void
V3D33_DISABLE_Z_ONLY_RENDERING_pack(__gen_user_data *data, uint8_t * restrict cl,
                                    const struct V3D33_DISABLE_Z_ONLY_RENDERING * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_DISABLE_Z_ONLY_RENDERING_length      1
#ifdef __gen_unpack_address
static inline void
V3D33_DISABLE_Z_ONLY_RENDERING_unpack(const uint8_t * restrict cl,
                                      struct V3D33_DISABLE_Z_ONLY_RENDERING * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_END_OF_Z_ONLY_RENDERING_IN_FRAME_opcode     12
#define V3D33_END_OF_Z_ONLY_RENDERING_IN_FRAME_header\
   .opcode                              =     12

struct V3D33_END_OF_Z_ONLY_RENDERING_IN_FRAME {
   uint32_t                             opcode;
};

static inline void
V3D33_END_OF_Z_ONLY_RENDERING_IN_FRAME_pack(__gen_user_data *data, uint8_t * restrict cl,
                                            const struct V3D33_END_OF_Z_ONLY_RENDERING_IN_FRAME * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_END_OF_Z_ONLY_RENDERING_IN_FRAME_length      1
#ifdef __gen_unpack_address
static inline void
V3D33_END_OF_Z_ONLY_RENDERING_IN_FRAME_unpack(const uint8_t * restrict cl,
                                              struct V3D33_END_OF_Z_ONLY_RENDERING_IN_FRAME * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_END_OF_RENDERING_opcode         13
#define V3D33_END_OF_RENDERING_header           \
   .opcode                              =     13

struct V3D33_END_OF_RENDERING {
   uint32_t                             opcode;
};

static inline void
V3D33_END_OF_RENDERING_pack(__gen_user_data *data, uint8_t * restrict cl,
                            const struct V3D33_END_OF_RENDERING * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_END_OF_RENDERING_length          1
#ifdef __gen_unpack_address
static inline void
V3D33_END_OF_RENDERING_unpack(const uint8_t * restrict cl,
                              struct V3D33_END_OF_RENDERING * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_WAIT_FOR_TRANSFORM_FEEDBACK_opcode     14
#define V3D33_WAIT_FOR_TRANSFORM_FEEDBACK_header\
   .opcode                              =     14

struct V3D33_WAIT_FOR_TRANSFORM_FEEDBACK {
   uint32_t                             opcode;
   uint32_t                             block_count;
};

static inline void
V3D33_WAIT_FOR_TRANSFORM_FEEDBACK_pack(__gen_user_data *data, uint8_t * restrict cl,
                                       const struct V3D33_WAIT_FOR_TRANSFORM_FEEDBACK * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->block_count, 0, 7);

}

#define V3D33_WAIT_FOR_TRANSFORM_FEEDBACK_length      2
#ifdef __gen_unpack_address
static inline void
V3D33_WAIT_FOR_TRANSFORM_FEEDBACK_unpack(const uint8_t * restrict cl,
                                         struct V3D33_WAIT_FOR_TRANSFORM_FEEDBACK * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->block_count = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D33_BRANCH_TO_AUTO_CHAINED_SUB_LIST_opcode     15
#define V3D33_BRANCH_TO_AUTO_CHAINED_SUB_LIST_header\
   .opcode                              =     15

struct V3D33_BRANCH_TO_AUTO_CHAINED_SUB_LIST {
   uint32_t                             opcode;
   __gen_address_type                   address;
};

static inline void
V3D33_BRANCH_TO_AUTO_CHAINED_SUB_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                                           const struct V3D33_BRANCH_TO_AUTO_CHAINED_SUB_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_BRANCH_TO_AUTO_CHAINED_SUB_LIST_length      5
#ifdef __gen_unpack_address
static inline void
V3D33_BRANCH_TO_AUTO_CHAINED_SUB_LIST_unpack(const uint8_t * restrict cl,
                                             struct V3D33_BRANCH_TO_AUTO_CHAINED_SUB_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 8, 39);
}
#endif


#define V3D33_BRANCH_opcode                   16
#define V3D33_BRANCH_header                     \
   .opcode                              =     16

struct V3D33_BRANCH {
   uint32_t                             opcode;
   __gen_address_type                   address;
};

static inline void
V3D33_BRANCH_pack(__gen_user_data *data, uint8_t * restrict cl,
                  const struct V3D33_BRANCH * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_BRANCH_length                    5
#ifdef __gen_unpack_address
static inline void
V3D33_BRANCH_unpack(const uint8_t * restrict cl,
                    struct V3D33_BRANCH * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 8, 39);
}
#endif


#define V3D33_BRANCH_TO_SUB_LIST_opcode       17
#define V3D33_BRANCH_TO_SUB_LIST_header         \
   .opcode                              =     17

struct V3D33_BRANCH_TO_SUB_LIST {
   uint32_t                             opcode;
   __gen_address_type                   address;
};

static inline void
V3D33_BRANCH_TO_SUB_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D33_BRANCH_TO_SUB_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_BRANCH_TO_SUB_LIST_length        5
#ifdef __gen_unpack_address
static inline void
V3D33_BRANCH_TO_SUB_LIST_unpack(const uint8_t * restrict cl,
                                struct V3D33_BRANCH_TO_SUB_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 8, 39);
}
#endif


#define V3D33_RETURN_FROM_SUB_LIST_opcode     18
#define V3D33_RETURN_FROM_SUB_LIST_header       \
   .opcode                              =     18

struct V3D33_RETURN_FROM_SUB_LIST {
   uint32_t                             opcode;
};

static inline void
V3D33_RETURN_FROM_SUB_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                                const struct V3D33_RETURN_FROM_SUB_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_RETURN_FROM_SUB_LIST_length      1
#ifdef __gen_unpack_address
static inline void
V3D33_RETURN_FROM_SUB_LIST_unpack(const uint8_t * restrict cl,
                                  struct V3D33_RETURN_FROM_SUB_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_FLUSH_VCD_CACHE_opcode          19
#define V3D33_FLUSH_VCD_CACHE_header            \
   .opcode                              =     19

struct V3D33_FLUSH_VCD_CACHE {
   uint32_t                             opcode;
};

static inline void
V3D33_FLUSH_VCD_CACHE_pack(__gen_user_data *data, uint8_t * restrict cl,
                           const struct V3D33_FLUSH_VCD_CACHE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_FLUSH_VCD_CACHE_length           1
#ifdef __gen_unpack_address
static inline void
V3D33_FLUSH_VCD_CACHE_unpack(const uint8_t * restrict cl,
                             struct V3D33_FLUSH_VCD_CACHE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_START_ADDRESS_OF_GENERIC_TILE_LIST_opcode     20
#define V3D33_START_ADDRESS_OF_GENERIC_TILE_LIST_header\
   .opcode                              =     20

struct V3D33_START_ADDRESS_OF_GENERIC_TILE_LIST {
   uint32_t                             opcode;
   __gen_address_type                   start;
   __gen_address_type                   end;
};

static inline void
V3D33_START_ADDRESS_OF_GENERIC_TILE_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                                              const struct V3D33_START_ADDRESS_OF_GENERIC_TILE_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->start);
   cl[ 1] = __gen_address_offset(&values->start);

   cl[ 2] = __gen_address_offset(&values->start) >> 8;

   cl[ 3] = __gen_address_offset(&values->start) >> 16;

   cl[ 4] = __gen_address_offset(&values->start) >> 24;

   __gen_emit_reloc(data, &values->end);
   cl[ 5] = __gen_address_offset(&values->end);

   cl[ 6] = __gen_address_offset(&values->end) >> 8;

   cl[ 7] = __gen_address_offset(&values->end) >> 16;

   cl[ 8] = __gen_address_offset(&values->end) >> 24;

}

#define V3D33_START_ADDRESS_OF_GENERIC_TILE_LIST_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_START_ADDRESS_OF_GENERIC_TILE_LIST_unpack(const uint8_t * restrict cl,
                                                struct V3D33_START_ADDRESS_OF_GENERIC_TILE_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->start = __gen_unpack_address(cl, 8, 39);
   values->end = __gen_unpack_address(cl, 40, 71);
}
#endif


#define V3D33_BRANCH_TO_IMPLICIT_TILE_LIST_opcode     21
#define V3D33_BRANCH_TO_IMPLICIT_TILE_LIST_header\
   .opcode                              =     21

struct V3D33_BRANCH_TO_IMPLICIT_TILE_LIST {
   uint32_t                             opcode;
   uint32_t                             tile_list_set_number;
};

static inline void
V3D33_BRANCH_TO_IMPLICIT_TILE_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D33_BRANCH_TO_IMPLICIT_TILE_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->tile_list_set_number, 0, 7);

}

#define V3D33_BRANCH_TO_IMPLICIT_TILE_LIST_length      2
#ifdef __gen_unpack_address
static inline void
V3D33_BRANCH_TO_IMPLICIT_TILE_LIST_unpack(const uint8_t * restrict cl,
                                          struct V3D33_BRANCH_TO_IMPLICIT_TILE_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->tile_list_set_number = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D33_BRANCH_TO_EXPLICIT_SUPERTILE_opcode     22
#define V3D33_BRANCH_TO_EXPLICIT_SUPERTILE_header\
   .opcode                              =     22

struct V3D33_BRANCH_TO_EXPLICIT_SUPERTILE {
   uint32_t                             opcode;
   __gen_address_type                   absolute_address_of_explicit_supertile_render_list;
   uint32_t                             explicit_supertile_number;
   uint32_t                             row_number;
   uint32_t                             column_number;
};

static inline void
V3D33_BRANCH_TO_EXPLICIT_SUPERTILE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D33_BRANCH_TO_EXPLICIT_SUPERTILE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->column_number, 0, 7);

   cl[ 2] = __gen_uint(values->row_number, 0, 7);

   cl[ 3] = __gen_uint(values->explicit_supertile_number, 0, 7);

   __gen_emit_reloc(data, &values->absolute_address_of_explicit_supertile_render_list);
   cl[ 4] = __gen_address_offset(&values->absolute_address_of_explicit_supertile_render_list);

   cl[ 5] = __gen_address_offset(&values->absolute_address_of_explicit_supertile_render_list) >> 8;

   cl[ 6] = __gen_address_offset(&values->absolute_address_of_explicit_supertile_render_list) >> 16;

   cl[ 7] = __gen_address_offset(&values->absolute_address_of_explicit_supertile_render_list) >> 24;

}

#define V3D33_BRANCH_TO_EXPLICIT_SUPERTILE_length      8
#ifdef __gen_unpack_address
static inline void
V3D33_BRANCH_TO_EXPLICIT_SUPERTILE_unpack(const uint8_t * restrict cl,
                                          struct V3D33_BRANCH_TO_EXPLICIT_SUPERTILE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->absolute_address_of_explicit_supertile_render_list = __gen_unpack_address(cl, 32, 63);
   values->explicit_supertile_number = __gen_unpack_uint(cl, 24, 31);
   values->row_number = __gen_unpack_uint(cl, 16, 23);
   values->column_number = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D33_SUPERTILE_COORDINATES_opcode     23
#define V3D33_SUPERTILE_COORDINATES_header      \
   .opcode                              =     23

struct V3D33_SUPERTILE_COORDINATES {
   uint32_t                             opcode;
   uint32_t                             row_number_in_supertiles;
   uint32_t                             column_number_in_supertiles;
};

static inline void
V3D33_SUPERTILE_COORDINATES_pack(__gen_user_data *data, uint8_t * restrict cl,
                                 const struct V3D33_SUPERTILE_COORDINATES * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->column_number_in_supertiles, 0, 7);

   cl[ 2] = __gen_uint(values->row_number_in_supertiles, 0, 7);

}

#define V3D33_SUPERTILE_COORDINATES_length      3
#ifdef __gen_unpack_address
static inline void
V3D33_SUPERTILE_COORDINATES_unpack(const uint8_t * restrict cl,
                                   struct V3D33_SUPERTILE_COORDINATES * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->row_number_in_supertiles = __gen_unpack_uint(cl, 16, 23);
   values->column_number_in_supertiles = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_opcode     24
#define V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_header\
   .opcode                              =     24

struct V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER {
   uint32_t                             opcode;
};

static inline void
V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                         const struct V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_length      1
#ifdef __gen_unpack_address
static inline void
V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_unpack(const uint8_t * restrict cl,
                                                           struct V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_EXTENDED_opcode     25
#define V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_EXTENDED_header\
   .opcode                              =     25

struct V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_EXTENDED {
   uint32_t                             opcode;
   uint32_t                             disable_color_buffer_write;
   bool                                 enable_z_write;
   bool                                 enable_stencil_write;
   bool                                 disable_color_buffers_clear_on_write;
   bool                                 disable_stencil_buffer_clear_on_write;
   bool                                 disable_z_buffer_clear_on_write;
   bool                                 disable_fast_opportunistic_write_out_in_multisample_mode;
   bool                                 last_tile_of_frame;
};

static inline void
V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_EXTENDED_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                                  const struct V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_EXTENDED * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->enable_z_write, 7, 7) |
            __gen_uint(values->enable_stencil_write, 6, 6) |
            __gen_uint(values->disable_color_buffers_clear_on_write, 4, 4) |
            __gen_uint(values->disable_stencil_buffer_clear_on_write, 3, 3) |
            __gen_uint(values->disable_z_buffer_clear_on_write, 2, 2) |
            __gen_uint(values->disable_fast_opportunistic_write_out_in_multisample_mode, 1, 1) |
            __gen_uint(values->last_tile_of_frame, 0, 0);

   cl[ 2] = __gen_uint(values->disable_color_buffer_write, 0, 7);

}

#define V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_EXTENDED_length      3
#ifdef __gen_unpack_address
static inline void
V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_EXTENDED_unpack(const uint8_t * restrict cl,
                                                                    struct V3D33_STORE_MULTI_SAMPLE_RESOLVED_TILE_COLOR_BUFFER_EXTENDED * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->disable_color_buffer_write = __gen_unpack_uint(cl, 16, 23);
   values->enable_z_write = __gen_unpack_uint(cl, 15, 15);
   values->enable_stencil_write = __gen_unpack_uint(cl, 14, 14);
   values->disable_color_buffers_clear_on_write = __gen_unpack_uint(cl, 12, 12);
   values->disable_stencil_buffer_clear_on_write = __gen_unpack_uint(cl, 11, 11);
   values->disable_z_buffer_clear_on_write = __gen_unpack_uint(cl, 10, 10);
   values->disable_fast_opportunistic_write_out_in_multisample_mode = __gen_unpack_uint(cl, 9, 9);
   values->last_tile_of_frame = __gen_unpack_uint(cl, 8, 8);
}
#endif


#define V3D33_RELOAD_TILE_COLOR_BUFFER_opcode     26
#define V3D33_RELOAD_TILE_COLOR_BUFFER_header   \
   .opcode                              =     26

struct V3D33_RELOAD_TILE_COLOR_BUFFER {
   uint32_t                             opcode;
   uint32_t                             disable_color_buffer_load;
   bool                                 enable_z_load;
   bool                                 enable_stencil_load;
};

static inline void
V3D33_RELOAD_TILE_COLOR_BUFFER_pack(__gen_user_data *data, uint8_t * restrict cl,
                                    const struct V3D33_RELOAD_TILE_COLOR_BUFFER * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->enable_z_load, 7, 7) |
            __gen_uint(values->enable_stencil_load, 6, 6);

   cl[ 2] = __gen_uint(values->disable_color_buffer_load, 0, 7);

}

#define V3D33_RELOAD_TILE_COLOR_BUFFER_length      3
#ifdef __gen_unpack_address
static inline void
V3D33_RELOAD_TILE_COLOR_BUFFER_unpack(const uint8_t * restrict cl,
                                      struct V3D33_RELOAD_TILE_COLOR_BUFFER * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->disable_color_buffer_load = __gen_unpack_uint(cl, 16, 23);
   values->enable_z_load = __gen_unpack_uint(cl, 15, 15);
   values->enable_stencil_load = __gen_unpack_uint(cl, 14, 14);
}
#endif


#define V3D33_END_OF_TILE_MARKER_opcode       27
#define V3D33_END_OF_TILE_MARKER_header         \
   .opcode                              =     27

struct V3D33_END_OF_TILE_MARKER {
   uint32_t                             opcode;
};

static inline void
V3D33_END_OF_TILE_MARKER_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D33_END_OF_TILE_MARKER * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_END_OF_TILE_MARKER_length        1
#ifdef __gen_unpack_address
static inline void
V3D33_END_OF_TILE_MARKER_unpack(const uint8_t * restrict cl,
                                struct V3D33_END_OF_TILE_MARKER * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_STORE_TILE_BUFFER_GENERAL_opcode     29
#define V3D33_STORE_TILE_BUFFER_GENERAL_header  \
   .opcode                              =     29

struct V3D33_STORE_TILE_BUFFER_GENERAL {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             padded_height_of_output_image_in_uif_blocks;
   bool                                 xor_uif;
   bool                                 last_tile_of_frame;
   bool                                 disable_color_buffers_clear_on_write;
   bool                                 disable_stencil_buffer_clear_on_write;
   bool                                 disable_z_buffer_clear_on_write;
   bool                                 raw_mode;
   uint32_t                             buffer_to_store;
#define RENDER_TARGET_0                          0
#define RENDER_TARGET_1                          1
#define RENDER_TARGET_2                          2
#define RENDER_TARGET_3                          3
#define NONE                                     8
#define Z                                        9
#define STENCIL                                  10
#define ZSTENCIL                                 11
};

static inline void
V3D33_STORE_TILE_BUFFER_GENERAL_pack(__gen_user_data *data, uint8_t * restrict cl,
                                     const struct V3D33_STORE_TILE_BUFFER_GENERAL * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->disable_color_buffers_clear_on_write, 7, 7) |
            __gen_uint(values->disable_stencil_buffer_clear_on_write, 6, 6) |
            __gen_uint(values->disable_z_buffer_clear_on_write, 5, 5) |
            __gen_uint(values->raw_mode, 4, 4) |
            __gen_uint(values->buffer_to_store, 0, 3);

   cl[ 2] = __gen_uint(values->padded_height_of_output_image_in_uif_blocks, 3, 15) |
            __gen_uint(values->xor_uif, 2, 2) |
            __gen_uint(values->last_tile_of_frame, 0, 0);

   cl[ 3] = __gen_uint(values->padded_height_of_output_image_in_uif_blocks, 3, 15) >> 8;

   __gen_emit_reloc(data, &values->address);
   cl[ 4] = __gen_address_offset(&values->address) >> 8;

   cl[ 5] = __gen_address_offset(&values->address) >> 16;

   cl[ 6] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_STORE_TILE_BUFFER_GENERAL_length      7
#ifdef __gen_unpack_address
static inline void
V3D33_STORE_TILE_BUFFER_GENERAL_unpack(const uint8_t * restrict cl,
                                       struct V3D33_STORE_TILE_BUFFER_GENERAL * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 32, 55);
   values->padded_height_of_output_image_in_uif_blocks = __gen_unpack_uint(cl, 19, 31);
   values->xor_uif = __gen_unpack_uint(cl, 18, 18);
   values->last_tile_of_frame = __gen_unpack_uint(cl, 16, 16);
   values->disable_color_buffers_clear_on_write = __gen_unpack_uint(cl, 15, 15);
   values->disable_stencil_buffer_clear_on_write = __gen_unpack_uint(cl, 14, 14);
   values->disable_z_buffer_clear_on_write = __gen_unpack_uint(cl, 13, 13);
   values->raw_mode = __gen_unpack_uint(cl, 12, 12);
   values->buffer_to_store = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_LOAD_TILE_BUFFER_GENERAL_opcode     30
#define V3D33_LOAD_TILE_BUFFER_GENERAL_header   \
   .opcode                              =     30

struct V3D33_LOAD_TILE_BUFFER_GENERAL {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             padded_height_of_output_image_in_uif_blocks;
   bool                                 xor_uif;
   bool                                 raw_mode;
   uint32_t                             buffer_to_load;
#define RENDER_TARGET_0                          0
#define RENDER_TARGET_1                          1
#define RENDER_TARGET_2                          2
#define RENDER_TARGET_3                          3
#define NONE                                     8
#define Z                                        9
#define STENCIL                                  10
#define ZSTENCIL                                 11
};

static inline void
V3D33_LOAD_TILE_BUFFER_GENERAL_pack(__gen_user_data *data, uint8_t * restrict cl,
                                    const struct V3D33_LOAD_TILE_BUFFER_GENERAL * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->raw_mode, 4, 4) |
            __gen_uint(values->buffer_to_load, 0, 3);

   cl[ 2] = __gen_uint(values->padded_height_of_output_image_in_uif_blocks, 3, 15) |
            __gen_uint(values->xor_uif, 2, 2);

   cl[ 3] = __gen_uint(values->padded_height_of_output_image_in_uif_blocks, 3, 15) >> 8;

   __gen_emit_reloc(data, &values->address);
   cl[ 4] = __gen_address_offset(&values->address) >> 8;

   cl[ 5] = __gen_address_offset(&values->address) >> 16;

   cl[ 6] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_LOAD_TILE_BUFFER_GENERAL_length      7
#ifdef __gen_unpack_address
static inline void
V3D33_LOAD_TILE_BUFFER_GENERAL_unpack(const uint8_t * restrict cl,
                                      struct V3D33_LOAD_TILE_BUFFER_GENERAL * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 32, 55);
   values->padded_height_of_output_image_in_uif_blocks = __gen_unpack_uint(cl, 19, 31);
   values->xor_uif = __gen_unpack_uint(cl, 18, 18);
   values->raw_mode = __gen_unpack_uint(cl, 12, 12);
   values->buffer_to_load = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT_opcode     31
#define V3D33_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT_header\
   .opcode                              =     31

struct V3D33_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT {
   uint32_t                             opcode;
};

static inline void
V3D33_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT_pack(__gen_user_data *data, uint8_t * restrict cl,
                                              const struct V3D33_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT_length      1
#ifdef __gen_unpack_address
static inline void
V3D33_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT_unpack(const uint8_t * restrict cl,
                                                struct V3D33_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_INDEXED_PRIM_LIST_opcode        32
#define V3D33_INDEXED_PRIM_LIST_header          \
   .opcode                              =     32

struct V3D33_INDEXED_PRIM_LIST {
   uint32_t                             opcode;
   uint32_t                             minimum_index;
   bool                                 enable_primitive_restarts;
   uint32_t                             maximum_index;
   __gen_address_type                   address_of_indices_list;
   uint32_t                             length;
   uint32_t                             index_type;
#define INDEX_TYPE_8_BIT                         0
#define INDEX_TYPE_16_BIT                        1
#define INDEX_TYPE_32_BIT                        2
   enum V3D33_Primitive                 mode;
};

static inline void
V3D33_INDEXED_PRIM_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                             const struct V3D33_INDEXED_PRIM_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->index_type, 6, 7) |
            __gen_uint(values->mode, 0, 4);


   memcpy(&cl[2], &values->length, sizeof(values->length));
   __gen_emit_reloc(data, &values->address_of_indices_list);
   cl[ 6] = __gen_address_offset(&values->address_of_indices_list);

   cl[ 7] = __gen_address_offset(&values->address_of_indices_list) >> 8;

   cl[ 8] = __gen_address_offset(&values->address_of_indices_list) >> 16;

   cl[ 9] = __gen_address_offset(&values->address_of_indices_list) >> 24;

   cl[10] = __gen_uint(values->maximum_index, 0, 30);

   cl[11] = __gen_uint(values->maximum_index, 0, 30) >> 8;

   cl[12] = __gen_uint(values->maximum_index, 0, 30) >> 16;

   cl[13] = __gen_uint(values->enable_primitive_restarts, 7, 7) |
            __gen_uint(values->maximum_index, 0, 30) >> 24;


   memcpy(&cl[14], &values->minimum_index, sizeof(values->minimum_index));
}

#define V3D33_INDEXED_PRIM_LIST_length        18
#ifdef __gen_unpack_address
static inline void
V3D33_INDEXED_PRIM_LIST_unpack(const uint8_t * restrict cl,
                               struct V3D33_INDEXED_PRIM_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->minimum_index = __gen_unpack_uint(cl, 112, 143);
   values->enable_primitive_restarts = __gen_unpack_uint(cl, 111, 111);
   values->maximum_index = __gen_unpack_uint(cl, 80, 110);
   values->address_of_indices_list = __gen_unpack_address(cl, 48, 79);
   values->length = __gen_unpack_uint(cl, 16, 47);
   values->index_type = __gen_unpack_uint(cl, 14, 15);
   values->mode = __gen_unpack_uint(cl, 8, 12);
}
#endif


#define V3D33_INDEXED_INSTANCED_PRIM_LIST_opcode     34
#define V3D33_INDEXED_INSTANCED_PRIM_LIST_header\
   .opcode                              =     34

struct V3D33_INDEXED_INSTANCED_PRIM_LIST {
   uint32_t                             opcode;
   bool                                 enable_primitive_restarts;
   uint32_t                             maximum_index;
   __gen_address_type                   address_of_indices_list;
   uint32_t                             number_of_instances;
   uint32_t                             instance_length;
   uint32_t                             index_type;
#define INDEX_TYPE_8_BIT                         0
#define INDEX_TYPE_16_BIT                        1
#define INDEX_TYPE_32_BIT                        2
   enum V3D33_Primitive                 mode;
};

static inline void
V3D33_INDEXED_INSTANCED_PRIM_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                                       const struct V3D33_INDEXED_INSTANCED_PRIM_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->index_type, 6, 7) |
            __gen_uint(values->mode, 0, 4);


   memcpy(&cl[2], &values->instance_length, sizeof(values->instance_length));

   memcpy(&cl[6], &values->number_of_instances, sizeof(values->number_of_instances));
   __gen_emit_reloc(data, &values->address_of_indices_list);
   cl[10] = __gen_address_offset(&values->address_of_indices_list);

   cl[11] = __gen_address_offset(&values->address_of_indices_list) >> 8;

   cl[12] = __gen_address_offset(&values->address_of_indices_list) >> 16;

   cl[13] = __gen_address_offset(&values->address_of_indices_list) >> 24;

   cl[14] = __gen_uint(values->maximum_index, 0, 30);

   cl[15] = __gen_uint(values->maximum_index, 0, 30) >> 8;

   cl[16] = __gen_uint(values->maximum_index, 0, 30) >> 16;

   cl[17] = __gen_uint(values->enable_primitive_restarts, 7, 7) |
            __gen_uint(values->maximum_index, 0, 30) >> 24;

}

#define V3D33_INDEXED_INSTANCED_PRIM_LIST_length     18
#ifdef __gen_unpack_address
static inline void
V3D33_INDEXED_INSTANCED_PRIM_LIST_unpack(const uint8_t * restrict cl,
                                         struct V3D33_INDEXED_INSTANCED_PRIM_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->enable_primitive_restarts = __gen_unpack_uint(cl, 143, 143);
   values->maximum_index = __gen_unpack_uint(cl, 112, 142);
   values->address_of_indices_list = __gen_unpack_address(cl, 80, 111);
   values->number_of_instances = __gen_unpack_uint(cl, 48, 79);
   values->instance_length = __gen_unpack_uint(cl, 16, 47);
   values->index_type = __gen_unpack_uint(cl, 14, 15);
   values->mode = __gen_unpack_uint(cl, 8, 12);
}
#endif


#define V3D33_VERTEX_ARRAY_PRIMS_opcode       36
#define V3D33_VERTEX_ARRAY_PRIMS_header         \
   .opcode                              =     36

struct V3D33_VERTEX_ARRAY_PRIMS {
   uint32_t                             opcode;
   uint32_t                             index_of_first_vertex;
   uint32_t                             length;
   enum V3D33_Primitive                 mode;
};

static inline void
V3D33_VERTEX_ARRAY_PRIMS_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D33_VERTEX_ARRAY_PRIMS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->mode, 0, 7);


   memcpy(&cl[2], &values->length, sizeof(values->length));

   memcpy(&cl[6], &values->index_of_first_vertex, sizeof(values->index_of_first_vertex));
}

#define V3D33_VERTEX_ARRAY_PRIMS_length       10
#ifdef __gen_unpack_address
static inline void
V3D33_VERTEX_ARRAY_PRIMS_unpack(const uint8_t * restrict cl,
                                struct V3D33_VERTEX_ARRAY_PRIMS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->index_of_first_vertex = __gen_unpack_uint(cl, 48, 79);
   values->length = __gen_unpack_uint(cl, 16, 47);
   values->mode = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D33_VERTEX_ARRAY_INSTANCED_PRIMS_opcode     38
#define V3D33_VERTEX_ARRAY_INSTANCED_PRIMS_header\
   .opcode                              =     38

struct V3D33_VERTEX_ARRAY_INSTANCED_PRIMS {
   uint32_t                             opcode;
   uint32_t                             index_of_first_vertex;
   uint32_t                             number_of_instances;
   uint32_t                             instance_length;
   enum V3D33_Primitive                 mode;
};

static inline void
V3D33_VERTEX_ARRAY_INSTANCED_PRIMS_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D33_VERTEX_ARRAY_INSTANCED_PRIMS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->mode, 0, 7);


   memcpy(&cl[2], &values->instance_length, sizeof(values->instance_length));

   memcpy(&cl[6], &values->number_of_instances, sizeof(values->number_of_instances));

   memcpy(&cl[10], &values->index_of_first_vertex, sizeof(values->index_of_first_vertex));
}

#define V3D33_VERTEX_ARRAY_INSTANCED_PRIMS_length     14
#ifdef __gen_unpack_address
static inline void
V3D33_VERTEX_ARRAY_INSTANCED_PRIMS_unpack(const uint8_t * restrict cl,
                                          struct V3D33_VERTEX_ARRAY_INSTANCED_PRIMS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->index_of_first_vertex = __gen_unpack_uint(cl, 80, 111);
   values->number_of_instances = __gen_unpack_uint(cl, 48, 79);
   values->instance_length = __gen_unpack_uint(cl, 16, 47);
   values->mode = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D33_BASE_VERTEX_BASE_INSTANCE_opcode     43
#define V3D33_BASE_VERTEX_BASE_INSTANCE_header  \
   .opcode                              =     43

struct V3D33_BASE_VERTEX_BASE_INSTANCE {
   uint32_t                             opcode;
   uint32_t                             base_instance;
   uint32_t                             base_vertex;
};

static inline void
V3D33_BASE_VERTEX_BASE_INSTANCE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                     const struct V3D33_BASE_VERTEX_BASE_INSTANCE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->base_vertex, sizeof(values->base_vertex));

   memcpy(&cl[5], &values->base_instance, sizeof(values->base_instance));
}

#define V3D33_BASE_VERTEX_BASE_INSTANCE_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_BASE_VERTEX_BASE_INSTANCE_unpack(const uint8_t * restrict cl,
                                       struct V3D33_BASE_VERTEX_BASE_INSTANCE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->base_instance = __gen_unpack_uint(cl, 40, 71);
   values->base_vertex = __gen_unpack_uint(cl, 8, 39);
}
#endif


#define V3D33_PRIM_LIST_FORMAT_opcode         56
#define V3D33_PRIM_LIST_FORMAT_header           \
   .opcode                              =     56

struct V3D33_PRIM_LIST_FORMAT {
   uint32_t                             opcode;
   bool                                 tri_strip_or_fan;
   uint32_t                             primitive_type;
#define LIST_POINTS                              0
#define LIST_LINES                               1
#define LIST_TRIANGLES                           2
};

static inline void
V3D33_PRIM_LIST_FORMAT_pack(__gen_user_data *data, uint8_t * restrict cl,
                            const struct V3D33_PRIM_LIST_FORMAT * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->tri_strip_or_fan, 7, 7) |
            __gen_uint(values->primitive_type, 0, 5);

}

#define V3D33_PRIM_LIST_FORMAT_length          2
#ifdef __gen_unpack_address
static inline void
V3D33_PRIM_LIST_FORMAT_unpack(const uint8_t * restrict cl,
                              struct V3D33_PRIM_LIST_FORMAT * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->tri_strip_or_fan = __gen_unpack_uint(cl, 15, 15);
   values->primitive_type = __gen_unpack_uint(cl, 8, 13);
}
#endif


#define V3D33_GL_SHADER_STATE_opcode          64
#define V3D33_GL_SHADER_STATE_header            \
   .opcode                              =     64

struct V3D33_GL_SHADER_STATE {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             number_of_attribute_arrays;
};

static inline void
V3D33_GL_SHADER_STATE_pack(__gen_user_data *data, uint8_t * restrict cl,
                           const struct V3D33_GL_SHADER_STATE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address) |
            __gen_uint(values->number_of_attribute_arrays, 0, 4);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_GL_SHADER_STATE_length           5
#ifdef __gen_unpack_address
static inline void
V3D33_GL_SHADER_STATE_unpack(const uint8_t * restrict cl,
                             struct V3D33_GL_SHADER_STATE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 13, 39);
   values->number_of_attribute_arrays = __gen_unpack_uint(cl, 8, 12);
}
#endif


#define V3D33_VCM_CACHE_SIZE_opcode           73
#define V3D33_VCM_CACHE_SIZE_header             \
   .opcode                              =     73

struct V3D33_VCM_CACHE_SIZE {
   uint32_t                             opcode;
   uint32_t                             number_of_16_vertex_batches_for_rendering;
   uint32_t                             number_of_16_vertex_batches_for_binning;
};

static inline void
V3D33_VCM_CACHE_SIZE_pack(__gen_user_data *data, uint8_t * restrict cl,
                          const struct V3D33_VCM_CACHE_SIZE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->number_of_16_vertex_batches_for_rendering, 4, 7) |
            __gen_uint(values->number_of_16_vertex_batches_for_binning, 0, 3);

}

#define V3D33_VCM_CACHE_SIZE_length            2
#ifdef __gen_unpack_address
static inline void
V3D33_VCM_CACHE_SIZE_unpack(const uint8_t * restrict cl,
                            struct V3D33_VCM_CACHE_SIZE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->number_of_16_vertex_batches_for_rendering = __gen_unpack_uint(cl, 12, 15);
   values->number_of_16_vertex_batches_for_binning = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_TRANSFORM_FEEDBACK_ENABLE_opcode     74
#define V3D33_TRANSFORM_FEEDBACK_ENABLE_header  \
   .opcode                              =     74

struct V3D33_TRANSFORM_FEEDBACK_ENABLE {
   uint32_t                             opcode;
   uint32_t                             number_of_32_bit_output_buffer_address_following;
   uint32_t                             number_of_16_bit_output_data_specs_following;
};

static inline void
V3D33_TRANSFORM_FEEDBACK_ENABLE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                     const struct V3D33_TRANSFORM_FEEDBACK_ENABLE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = 0;
   cl[ 2] = __gen_uint(values->number_of_32_bit_output_buffer_address_following, 0, 2) |
            __gen_uint(values->number_of_16_bit_output_data_specs_following, 3, 7);

}

#define V3D33_TRANSFORM_FEEDBACK_ENABLE_length      3
#ifdef __gen_unpack_address
static inline void
V3D33_TRANSFORM_FEEDBACK_ENABLE_unpack(const uint8_t * restrict cl,
                                       struct V3D33_TRANSFORM_FEEDBACK_ENABLE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->number_of_32_bit_output_buffer_address_following = __gen_unpack_uint(cl, 16, 18);
   values->number_of_16_bit_output_data_specs_following = __gen_unpack_uint(cl, 19, 23);
}
#endif


#define V3D33_FLUSH_TRANSFORM_FEEDBACK_DATA_opcode     75
#define V3D33_FLUSH_TRANSFORM_FEEDBACK_DATA_header\
   .opcode                              =     75

struct V3D33_FLUSH_TRANSFORM_FEEDBACK_DATA {
   uint32_t                             opcode;
};

static inline void
V3D33_FLUSH_TRANSFORM_FEEDBACK_DATA_pack(__gen_user_data *data, uint8_t * restrict cl,
                                         const struct V3D33_FLUSH_TRANSFORM_FEEDBACK_DATA * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_FLUSH_TRANSFORM_FEEDBACK_DATA_length      1
#ifdef __gen_unpack_address
static inline void
V3D33_FLUSH_TRANSFORM_FEEDBACK_DATA_unpack(const uint8_t * restrict cl,
                                           struct V3D33_FLUSH_TRANSFORM_FEEDBACK_DATA * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC_header\


struct V3D33_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC {
   uint32_t                             first_shaded_vertex_value_to_output;
   uint32_t                             number_of_consecutive_vertex_values_to_output_as_32_bit_values;
   uint32_t                             output_buffer_to_write_to;
};

static inline void
V3D33_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC_pack(__gen_user_data *data, uint8_t * restrict cl,
                                               const struct V3D33_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC * restrict values)
{
   assert(values->number_of_consecutive_vertex_values_to_output_as_32_bit_values >= 1);
   cl[ 0] = __gen_uint(values->first_shaded_vertex_value_to_output, 0, 7);

   cl[ 1] = __gen_uint(values->number_of_consecutive_vertex_values_to_output_as_32_bit_values - 1, 0, 3) |
            __gen_uint(values->output_buffer_to_write_to, 4, 5);

}

#define V3D33_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC_length      2
#ifdef __gen_unpack_address
static inline void
V3D33_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC_unpack(const uint8_t * restrict cl,
                                                 struct V3D33_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC * restrict values)
{
   values->first_shaded_vertex_value_to_output = __gen_unpack_uint(cl, 0, 7);
   values->number_of_consecutive_vertex_values_to_output_as_32_bit_values = __gen_unpack_uint(cl, 8, 11) + 1;
   values->output_buffer_to_write_to = __gen_unpack_uint(cl, 12, 13);
}
#endif


#define V3D33_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS_header\


struct V3D33_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS {
   __gen_address_type                   address;
};

static inline void
V3D33_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS_pack(__gen_user_data *data, uint8_t * restrict cl,
                                             const struct V3D33_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS * restrict values)
{
   __gen_emit_reloc(data, &values->address);
   cl[ 0] = __gen_address_offset(&values->address);

   cl[ 1] = __gen_address_offset(&values->address) >> 8;

   cl[ 2] = __gen_address_offset(&values->address) >> 16;

   cl[ 3] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS_length      4
#ifdef __gen_unpack_address
static inline void
V3D33_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS_unpack(const uint8_t * restrict cl,
                                               struct V3D33_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS * restrict values)
{
   values->address = __gen_unpack_address(cl, 0, 31);
}
#endif


#define V3D33_STENCIL_CFG_opcode              80
#define V3D33_STENCIL_CFG_header                \
   .opcode                              =     80

struct V3D33_STENCIL_CFG {
   uint32_t                             opcode;
   uint32_t                             stencil_write_mask;
   bool                                 back_config;
   bool                                 front_config;
   enum V3D33_Stencil_Op                stencil_pass_op;
   enum V3D33_Stencil_Op                depth_test_fail_op;
   enum V3D33_Stencil_Op                stencil_test_fail_op;
   enum V3D33_Compare_Function          stencil_test_function;
   uint32_t                             stencil_test_mask;
   uint32_t                             stencil_ref_value;
};

static inline void
V3D33_STENCIL_CFG_pack(__gen_user_data *data, uint8_t * restrict cl,
                       const struct V3D33_STENCIL_CFG * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->stencil_ref_value, 0, 7);

   cl[ 2] = __gen_uint(values->stencil_test_mask, 0, 7);

   cl[ 3] = __gen_uint(values->depth_test_fail_op, 6, 8) |
            __gen_uint(values->stencil_test_fail_op, 3, 5) |
            __gen_uint(values->stencil_test_function, 0, 2);

   cl[ 4] = __gen_uint(values->back_config, 5, 5) |
            __gen_uint(values->front_config, 4, 4) |
            __gen_uint(values->stencil_pass_op, 1, 3) |
            __gen_uint(values->depth_test_fail_op, 6, 8) >> 8;

   cl[ 5] = __gen_uint(values->stencil_write_mask, 0, 7);

}

#define V3D33_STENCIL_CFG_length               6
#ifdef __gen_unpack_address
static inline void
V3D33_STENCIL_CFG_unpack(const uint8_t * restrict cl,
                         struct V3D33_STENCIL_CFG * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->stencil_write_mask = __gen_unpack_uint(cl, 40, 47);
   values->back_config = __gen_unpack_uint(cl, 37, 37);
   values->front_config = __gen_unpack_uint(cl, 36, 36);
   values->stencil_pass_op = __gen_unpack_uint(cl, 33, 35);
   values->depth_test_fail_op = __gen_unpack_uint(cl, 30, 32);
   values->stencil_test_fail_op = __gen_unpack_uint(cl, 27, 29);
   values->stencil_test_function = __gen_unpack_uint(cl, 24, 26);
   values->stencil_test_mask = __gen_unpack_uint(cl, 16, 23);
   values->stencil_ref_value = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D33_BLEND_CFG_opcode                84
#define V3D33_BLEND_CFG_header                  \
   .opcode                              =     84

struct V3D33_BLEND_CFG {
   uint32_t                             opcode;
   enum V3D33_Blend_Factor              color_blend_dst_factor;
   enum V3D33_Blend_Factor              color_blend_src_factor;
   enum V3D33_Blend_Mode                color_blend_mode;
   enum V3D33_Blend_Factor              alpha_blend_dst_factor;
   enum V3D33_Blend_Factor              alpha_blend_src_factor;
   enum V3D33_Blend_Mode                alpha_blend_mode;
};

static inline void
V3D33_BLEND_CFG_pack(__gen_user_data *data, uint8_t * restrict cl,
                     const struct V3D33_BLEND_CFG * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->alpha_blend_src_factor, 4, 7) |
            __gen_uint(values->alpha_blend_mode, 0, 3);

   cl[ 2] = __gen_uint(values->color_blend_mode, 4, 7) |
            __gen_uint(values->alpha_blend_dst_factor, 0, 3);

   cl[ 3] = __gen_uint(values->color_blend_dst_factor, 4, 7) |
            __gen_uint(values->color_blend_src_factor, 0, 3);

}

#define V3D33_BLEND_CFG_length                 4
#ifdef __gen_unpack_address
static inline void
V3D33_BLEND_CFG_unpack(const uint8_t * restrict cl,
                       struct V3D33_BLEND_CFG * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->color_blend_dst_factor = __gen_unpack_uint(cl, 28, 31);
   values->color_blend_src_factor = __gen_unpack_uint(cl, 24, 27);
   values->color_blend_mode = __gen_unpack_uint(cl, 20, 23);
   values->alpha_blend_dst_factor = __gen_unpack_uint(cl, 16, 19);
   values->alpha_blend_src_factor = __gen_unpack_uint(cl, 12, 15);
   values->alpha_blend_mode = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_BLEND_CONSTANT_COLOR_opcode     86
#define V3D33_BLEND_CONSTANT_COLOR_header       \
   .opcode                              =     86

struct V3D33_BLEND_CONSTANT_COLOR {
   uint32_t                             opcode;
   uint32_t                             alpha_f16;
   uint32_t                             blue_f16;
   uint32_t                             green_f16;
   uint32_t                             red_f16;
};

static inline void
V3D33_BLEND_CONSTANT_COLOR_pack(__gen_user_data *data, uint8_t * restrict cl,
                                const struct V3D33_BLEND_CONSTANT_COLOR * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->red_f16, 0, 15);

   cl[ 2] = __gen_uint(values->red_f16, 0, 15) >> 8;

   cl[ 3] = __gen_uint(values->green_f16, 0, 15);

   cl[ 4] = __gen_uint(values->green_f16, 0, 15) >> 8;

   cl[ 5] = __gen_uint(values->blue_f16, 0, 15);

   cl[ 6] = __gen_uint(values->blue_f16, 0, 15) >> 8;

   cl[ 7] = __gen_uint(values->alpha_f16, 0, 15);

   cl[ 8] = __gen_uint(values->alpha_f16, 0, 15) >> 8;

}

#define V3D33_BLEND_CONSTANT_COLOR_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_BLEND_CONSTANT_COLOR_unpack(const uint8_t * restrict cl,
                                  struct V3D33_BLEND_CONSTANT_COLOR * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->alpha_f16 = __gen_unpack_uint(cl, 56, 71);
   values->blue_f16 = __gen_unpack_uint(cl, 40, 55);
   values->green_f16 = __gen_unpack_uint(cl, 24, 39);
   values->red_f16 = __gen_unpack_uint(cl, 8, 23);
}
#endif


#define V3D33_COLOR_WRITE_MASKS_opcode        87
#define V3D33_COLOR_WRITE_MASKS_header          \
   .opcode                              =     87

struct V3D33_COLOR_WRITE_MASKS {
   uint32_t                             opcode;
   uint32_t                             mask;
};

static inline void
V3D33_COLOR_WRITE_MASKS_pack(__gen_user_data *data, uint8_t * restrict cl,
                             const struct V3D33_COLOR_WRITE_MASKS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->mask, sizeof(values->mask));
}

#define V3D33_COLOR_WRITE_MASKS_length         5
#ifdef __gen_unpack_address
static inline void
V3D33_COLOR_WRITE_MASKS_unpack(const uint8_t * restrict cl,
                               struct V3D33_COLOR_WRITE_MASKS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->mask = __gen_unpack_uint(cl, 8, 39);
}
#endif


#define V3D33_OCCLUSION_QUERY_COUNTER_opcode     92
#define V3D33_OCCLUSION_QUERY_COUNTER_header    \
   .opcode                              =     92

struct V3D33_OCCLUSION_QUERY_COUNTER {
   uint32_t                             opcode;
   __gen_address_type                   address;
};

static inline void
V3D33_OCCLUSION_QUERY_COUNTER_pack(__gen_user_data *data, uint8_t * restrict cl,
                                   const struct V3D33_OCCLUSION_QUERY_COUNTER * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_OCCLUSION_QUERY_COUNTER_length      5
#ifdef __gen_unpack_address
static inline void
V3D33_OCCLUSION_QUERY_COUNTER_unpack(const uint8_t * restrict cl,
                                     struct V3D33_OCCLUSION_QUERY_COUNTER * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 8, 39);
}
#endif


#define V3D33_CFG_BITS_opcode                 96
#define V3D33_CFG_BITS_header                   \
   .opcode                              =     96

struct V3D33_CFG_BITS {
   uint32_t                             opcode;
   bool                                 direct3d_provoking_vertex;
   bool                                 direct3d_point_fill_mode;
   bool                                 blend_enable;
   bool                                 stencil_enable;
   bool                                 early_z_updates_enable;
   bool                                 early_z_enable;
   bool                                 z_updates_enable;
   enum V3D33_Compare_Function          depth_test_function;
   bool                                 direct3d_wireframe_triangles_mode;
   uint32_t                             rasterizer_oversample_mode;
   uint32_t                             line_rasterization;
   bool                                 enable_depth_offset;
   bool                                 clockwise_primitives;
   bool                                 enable_reverse_facing_primitive;
   bool                                 enable_forward_facing_primitive;
};

static inline void
V3D33_CFG_BITS_pack(__gen_user_data *data, uint8_t * restrict cl,
                    const struct V3D33_CFG_BITS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->rasterizer_oversample_mode, 6, 7) |
            __gen_uint(values->line_rasterization, 4, 5) |
            __gen_uint(values->enable_depth_offset, 3, 3) |
            __gen_uint(values->clockwise_primitives, 2, 2) |
            __gen_uint(values->enable_reverse_facing_primitive, 1, 1) |
            __gen_uint(values->enable_forward_facing_primitive, 0, 0);

   cl[ 2] = __gen_uint(values->z_updates_enable, 7, 7) |
            __gen_uint(values->depth_test_function, 4, 6) |
            __gen_uint(values->direct3d_wireframe_triangles_mode, 3, 3);

   cl[ 3] = __gen_uint(values->direct3d_provoking_vertex, 5, 5) |
            __gen_uint(values->direct3d_point_fill_mode, 4, 4) |
            __gen_uint(values->blend_enable, 3, 3) |
            __gen_uint(values->stencil_enable, 2, 2) |
            __gen_uint(values->early_z_updates_enable, 1, 1) |
            __gen_uint(values->early_z_enable, 0, 0);

}

#define V3D33_CFG_BITS_length                  4
#ifdef __gen_unpack_address
static inline void
V3D33_CFG_BITS_unpack(const uint8_t * restrict cl,
                      struct V3D33_CFG_BITS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->direct3d_provoking_vertex = __gen_unpack_uint(cl, 29, 29);
   values->direct3d_point_fill_mode = __gen_unpack_uint(cl, 28, 28);
   values->blend_enable = __gen_unpack_uint(cl, 27, 27);
   values->stencil_enable = __gen_unpack_uint(cl, 26, 26);
   values->early_z_updates_enable = __gen_unpack_uint(cl, 25, 25);
   values->early_z_enable = __gen_unpack_uint(cl, 24, 24);
   values->z_updates_enable = __gen_unpack_uint(cl, 23, 23);
   values->depth_test_function = __gen_unpack_uint(cl, 20, 22);
   values->direct3d_wireframe_triangles_mode = __gen_unpack_uint(cl, 19, 19);
   values->rasterizer_oversample_mode = __gen_unpack_uint(cl, 14, 15);
   values->line_rasterization = __gen_unpack_uint(cl, 12, 13);
   values->enable_depth_offset = __gen_unpack_uint(cl, 11, 11);
   values->clockwise_primitives = __gen_unpack_uint(cl, 10, 10);
   values->enable_reverse_facing_primitive = __gen_unpack_uint(cl, 9, 9);
   values->enable_forward_facing_primitive = __gen_unpack_uint(cl, 8, 8);
}
#endif


#define V3D33_ZERO_ALL_FLAT_SHADE_FLAGS_opcode     97
#define V3D33_ZERO_ALL_FLAT_SHADE_FLAGS_header  \
   .opcode                              =     97

struct V3D33_ZERO_ALL_FLAT_SHADE_FLAGS {
   uint32_t                             opcode;
};

static inline void
V3D33_ZERO_ALL_FLAT_SHADE_FLAGS_pack(__gen_user_data *data, uint8_t * restrict cl,
                                     const struct V3D33_ZERO_ALL_FLAT_SHADE_FLAGS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_ZERO_ALL_FLAT_SHADE_FLAGS_length      1
#ifdef __gen_unpack_address
static inline void
V3D33_ZERO_ALL_FLAT_SHADE_FLAGS_unpack(const uint8_t * restrict cl,
                                       struct V3D33_ZERO_ALL_FLAT_SHADE_FLAGS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_FLAT_SHADE_FLAGS_opcode         98
#define V3D33_FLAT_SHADE_FLAGS_header           \
   .opcode                              =     98

struct V3D33_FLAT_SHADE_FLAGS {
   uint32_t                             opcode;
   uint32_t                             flat_shade_flags_for_varyings_v024;
   enum V3D33_Varying_Flags_Action      action_for_flat_shade_flags_of_higher_numbered_varyings;
   enum V3D33_Varying_Flags_Action      action_for_flat_shade_flags_of_lower_numbered_varyings;
   uint32_t                             varying_offset_v0;
};

static inline void
V3D33_FLAT_SHADE_FLAGS_pack(__gen_user_data *data, uint8_t * restrict cl,
                            const struct V3D33_FLAT_SHADE_FLAGS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->action_for_flat_shade_flags_of_higher_numbered_varyings, 6, 7) |
            __gen_uint(values->action_for_flat_shade_flags_of_lower_numbered_varyings, 4, 5) |
            __gen_uint(values->varying_offset_v0, 0, 3);

   cl[ 2] = __gen_uint(values->flat_shade_flags_for_varyings_v024, 0, 23);

   cl[ 3] = __gen_uint(values->flat_shade_flags_for_varyings_v024, 0, 23) >> 8;

   cl[ 4] = __gen_uint(values->flat_shade_flags_for_varyings_v024, 0, 23) >> 16;

}

#define V3D33_FLAT_SHADE_FLAGS_length          5
#ifdef __gen_unpack_address
static inline void
V3D33_FLAT_SHADE_FLAGS_unpack(const uint8_t * restrict cl,
                              struct V3D33_FLAT_SHADE_FLAGS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->flat_shade_flags_for_varyings_v024 = __gen_unpack_uint(cl, 16, 39);
   values->action_for_flat_shade_flags_of_higher_numbered_varyings = __gen_unpack_uint(cl, 14, 15);
   values->action_for_flat_shade_flags_of_lower_numbered_varyings = __gen_unpack_uint(cl, 12, 13);
   values->varying_offset_v0 = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_POINT_SIZE_opcode              104
#define V3D33_POINT_SIZE_header                 \
   .opcode                              =    104

struct V3D33_POINT_SIZE {
   uint32_t                             opcode;
   float                                point_size;
};

static inline void
V3D33_POINT_SIZE_pack(__gen_user_data *data, uint8_t * restrict cl,
                      const struct V3D33_POINT_SIZE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->point_size, sizeof(values->point_size));
}

#define V3D33_POINT_SIZE_length                5
#ifdef __gen_unpack_address
static inline void
V3D33_POINT_SIZE_unpack(const uint8_t * restrict cl,
                        struct V3D33_POINT_SIZE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->point_size = __gen_unpack_float(cl, 8, 39);
}
#endif


#define V3D33_LINE_WIDTH_opcode              105
#define V3D33_LINE_WIDTH_header                 \
   .opcode                              =    105

struct V3D33_LINE_WIDTH {
   uint32_t                             opcode;
   float                                line_width;
};

static inline void
V3D33_LINE_WIDTH_pack(__gen_user_data *data, uint8_t * restrict cl,
                      const struct V3D33_LINE_WIDTH * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->line_width, sizeof(values->line_width));
}

#define V3D33_LINE_WIDTH_length                5
#ifdef __gen_unpack_address
static inline void
V3D33_LINE_WIDTH_unpack(const uint8_t * restrict cl,
                        struct V3D33_LINE_WIDTH * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->line_width = __gen_unpack_float(cl, 8, 39);
}
#endif


#define V3D33_DEPTH_OFFSET_opcode            106
#define V3D33_DEPTH_OFFSET_header               \
   .opcode                              =    106

struct V3D33_DEPTH_OFFSET {
   uint32_t                             opcode;
   float                                depth_offset_units;
   float                                depth_offset_factor;
};

static inline void
V3D33_DEPTH_OFFSET_pack(__gen_user_data *data, uint8_t * restrict cl,
                        const struct V3D33_DEPTH_OFFSET * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(fui(values->depth_offset_factor) >> 16, 0, 15);

   cl[ 2] = __gen_uint(fui(values->depth_offset_factor) >> 16, 0, 15) >> 8;

   cl[ 3] = __gen_uint(fui(values->depth_offset_units) >> 16, 0, 15);

   cl[ 4] = __gen_uint(fui(values->depth_offset_units) >> 16, 0, 15) >> 8;

}

#define V3D33_DEPTH_OFFSET_length              5
#ifdef __gen_unpack_address
static inline void
V3D33_DEPTH_OFFSET_unpack(const uint8_t * restrict cl,
                          struct V3D33_DEPTH_OFFSET * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->depth_offset_units = __gen_unpack_f187(cl, 24, 39);
   values->depth_offset_factor = __gen_unpack_f187(cl, 8, 23);
}
#endif


#define V3D33_CLIP_WINDOW_opcode             107
#define V3D33_CLIP_WINDOW_header                \
   .opcode                              =    107

struct V3D33_CLIP_WINDOW {
   uint32_t                             opcode;
   uint32_t                             clip_window_height_in_pixels;
   uint32_t                             clip_window_width_in_pixels;
   uint32_t                             clip_window_bottom_pixel_coordinate;
   uint32_t                             clip_window_left_pixel_coordinate;
};

static inline void
V3D33_CLIP_WINDOW_pack(__gen_user_data *data, uint8_t * restrict cl,
                       const struct V3D33_CLIP_WINDOW * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->clip_window_left_pixel_coordinate, 0, 15);

   cl[ 2] = __gen_uint(values->clip_window_left_pixel_coordinate, 0, 15) >> 8;

   cl[ 3] = __gen_uint(values->clip_window_bottom_pixel_coordinate, 0, 15);

   cl[ 4] = __gen_uint(values->clip_window_bottom_pixel_coordinate, 0, 15) >> 8;

   cl[ 5] = __gen_uint(values->clip_window_width_in_pixels, 0, 15);

   cl[ 6] = __gen_uint(values->clip_window_width_in_pixels, 0, 15) >> 8;

   cl[ 7] = __gen_uint(values->clip_window_height_in_pixels, 0, 15);

   cl[ 8] = __gen_uint(values->clip_window_height_in_pixels, 0, 15) >> 8;

}

#define V3D33_CLIP_WINDOW_length               9
#ifdef __gen_unpack_address
static inline void
V3D33_CLIP_WINDOW_unpack(const uint8_t * restrict cl,
                         struct V3D33_CLIP_WINDOW * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->clip_window_height_in_pixels = __gen_unpack_uint(cl, 56, 71);
   values->clip_window_width_in_pixels = __gen_unpack_uint(cl, 40, 55);
   values->clip_window_bottom_pixel_coordinate = __gen_unpack_uint(cl, 24, 39);
   values->clip_window_left_pixel_coordinate = __gen_unpack_uint(cl, 8, 23);
}
#endif


#define V3D33_VIEWPORT_OFFSET_opcode         108
#define V3D33_VIEWPORT_OFFSET_header            \
   .opcode                              =    108

struct V3D33_VIEWPORT_OFFSET {
   uint32_t                             opcode;
   float                                viewport_centre_y_coordinate;
   float                                viewport_centre_x_coordinate;
};

static inline void
V3D33_VIEWPORT_OFFSET_pack(__gen_user_data *data, uint8_t * restrict cl,
                           const struct V3D33_VIEWPORT_OFFSET * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_sfixed(values->viewport_centre_x_coordinate, 0, 31, 8);

   cl[ 2] = __gen_sfixed(values->viewport_centre_x_coordinate, 0, 31, 8) >> 8;

   cl[ 3] = __gen_sfixed(values->viewport_centre_x_coordinate, 0, 31, 8) >> 16;

   cl[ 4] = __gen_sfixed(values->viewport_centre_x_coordinate, 0, 31, 8) >> 24;

   cl[ 5] = __gen_sfixed(values->viewport_centre_y_coordinate, 0, 31, 8);

   cl[ 6] = __gen_sfixed(values->viewport_centre_y_coordinate, 0, 31, 8) >> 8;

   cl[ 7] = __gen_sfixed(values->viewport_centre_y_coordinate, 0, 31, 8) >> 16;

   cl[ 8] = __gen_sfixed(values->viewport_centre_y_coordinate, 0, 31, 8) >> 24;

}

#define V3D33_VIEWPORT_OFFSET_length           9
#ifdef __gen_unpack_address
static inline void
V3D33_VIEWPORT_OFFSET_unpack(const uint8_t * restrict cl,
                             struct V3D33_VIEWPORT_OFFSET * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->viewport_centre_y_coordinate = __gen_unpack_sfixed(cl, 40, 71, 8);
   values->viewport_centre_x_coordinate = __gen_unpack_sfixed(cl, 8, 39, 8);
}
#endif


#define V3D33_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES_opcode    109
#define V3D33_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES_header\
   .opcode                              =    109

struct V3D33_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES {
   uint32_t                             opcode;
   float                                maximum_zw;
   float                                minimum_zw;
};

static inline void
V3D33_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES_pack(__gen_user_data *data, uint8_t * restrict cl,
                                             const struct V3D33_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->minimum_zw, sizeof(values->minimum_zw));

   memcpy(&cl[5], &values->maximum_zw, sizeof(values->maximum_zw));
}

#define V3D33_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES_unpack(const uint8_t * restrict cl,
                                               struct V3D33_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->maximum_zw = __gen_unpack_float(cl, 40, 71);
   values->minimum_zw = __gen_unpack_float(cl, 8, 39);
}
#endif


#define V3D33_CLIPPER_XY_SCALING_opcode      110
#define V3D33_CLIPPER_XY_SCALING_header         \
   .opcode                              =    110

struct V3D33_CLIPPER_XY_SCALING {
   uint32_t                             opcode;
   float                                viewport_half_height_in_1_256th_of_pixel;
   float                                viewport_half_width_in_1_256th_of_pixel;
};

static inline void
V3D33_CLIPPER_XY_SCALING_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D33_CLIPPER_XY_SCALING * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->viewport_half_width_in_1_256th_of_pixel, sizeof(values->viewport_half_width_in_1_256th_of_pixel));

   memcpy(&cl[5], &values->viewport_half_height_in_1_256th_of_pixel, sizeof(values->viewport_half_height_in_1_256th_of_pixel));
}

#define V3D33_CLIPPER_XY_SCALING_length        9
#ifdef __gen_unpack_address
static inline void
V3D33_CLIPPER_XY_SCALING_unpack(const uint8_t * restrict cl,
                                struct V3D33_CLIPPER_XY_SCALING * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->viewport_half_height_in_1_256th_of_pixel = __gen_unpack_float(cl, 40, 71);
   values->viewport_half_width_in_1_256th_of_pixel = __gen_unpack_float(cl, 8, 39);
}
#endif


#define V3D33_CLIPPER_Z_SCALE_AND_OFFSET_opcode    111
#define V3D33_CLIPPER_Z_SCALE_AND_OFFSET_header \
   .opcode                              =    111

struct V3D33_CLIPPER_Z_SCALE_AND_OFFSET {
   uint32_t                             opcode;
   float                                viewport_z_offset_zc_to_zs;
   float                                viewport_z_scale_zc_to_zs;
};

static inline void
V3D33_CLIPPER_Z_SCALE_AND_OFFSET_pack(__gen_user_data *data, uint8_t * restrict cl,
                                      const struct V3D33_CLIPPER_Z_SCALE_AND_OFFSET * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->viewport_z_scale_zc_to_zs, sizeof(values->viewport_z_scale_zc_to_zs));

   memcpy(&cl[5], &values->viewport_z_offset_zc_to_zs, sizeof(values->viewport_z_offset_zc_to_zs));
}

#define V3D33_CLIPPER_Z_SCALE_AND_OFFSET_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_CLIPPER_Z_SCALE_AND_OFFSET_unpack(const uint8_t * restrict cl,
                                        struct V3D33_CLIPPER_Z_SCALE_AND_OFFSET * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->viewport_z_offset_zc_to_zs = __gen_unpack_float(cl, 40, 71);
   values->viewport_z_scale_zc_to_zs = __gen_unpack_float(cl, 8, 39);
}
#endif


#define V3D33_TILE_BINNING_MODE_CFG_PART1_opcode    120
#define V3D33_TILE_BINNING_MODE_CFG_PART1_header\
   .opcode                              =    120,  \
   .auto_initialize_tile_state_data_array =      1,  \
   .sub_id                              =      0

struct V3D33_TILE_BINNING_MODE_CFG_PART1 {
   uint32_t                             opcode;
   bool                                 double_buffer_in_non_ms_mode;
   bool                                 multisample_mode_4x;
   enum V3D33_Internal_BPP              maximum_bpp_of_all_render_targets;
   uint32_t                             number_of_render_targets;
   uint32_t                             height_in_tiles;
   uint32_t                             width_in_tiles;
   __gen_address_type                   tile_state_data_array_base_address;
   uint32_t                             tile_allocation_block_size;
#define TILE_ALLOCATION_BLOCK_SIZE_64B           0
#define TILE_ALLOCATION_BLOCK_SIZE_128B          1
#define TILE_ALLOCATION_BLOCK_SIZE_256B          2
   uint32_t                             tile_allocation_initial_block_size;
#define TILE_ALLOCATION_INITIAL_BLOCK_SIZE_64B   0
#define TILE_ALLOCATION_INITIAL_BLOCK_SIZE_128B  1
#define TILE_ALLOCATION_INITIAL_BLOCK_SIZE_256B  2
   bool                                 auto_initialize_tile_state_data_array;
   uint32_t                             sub_id;
};

static inline void
V3D33_TILE_BINNING_MODE_CFG_PART1_pack(__gen_user_data *data, uint8_t * restrict cl,
                                       const struct V3D33_TILE_BINNING_MODE_CFG_PART1 * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->tile_state_data_array_base_address);
   cl[ 1] = __gen_address_offset(&values->tile_state_data_array_base_address) |
            __gen_uint(values->tile_allocation_block_size, 4, 5) |
            __gen_uint(values->tile_allocation_initial_block_size, 2, 3) |
            __gen_uint(values->auto_initialize_tile_state_data_array, 1, 1) |
            __gen_uint(values->sub_id, 0, 0);

   cl[ 2] = __gen_address_offset(&values->tile_state_data_array_base_address) >> 8;

   cl[ 3] = __gen_address_offset(&values->tile_state_data_array_base_address) >> 16;

   cl[ 4] = __gen_address_offset(&values->tile_state_data_array_base_address) >> 24;

   cl[ 5] = __gen_uint(values->width_in_tiles, 0, 11);

   cl[ 6] = __gen_uint(values->height_in_tiles, 4, 15) |
            __gen_uint(values->width_in_tiles, 0, 11) >> 8;

   cl[ 7] = __gen_uint(values->height_in_tiles, 4, 15) >> 8;

   cl[ 8] = __gen_uint(values->double_buffer_in_non_ms_mode, 7, 7) |
            __gen_uint(values->multisample_mode_4x, 6, 6) |
            __gen_uint(values->maximum_bpp_of_all_render_targets, 4, 5) |
            __gen_uint(values->number_of_render_targets, 0, 3);

}

#define V3D33_TILE_BINNING_MODE_CFG_PART1_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_BINNING_MODE_CFG_PART1_unpack(const uint8_t * restrict cl,
                                         struct V3D33_TILE_BINNING_MODE_CFG_PART1 * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->double_buffer_in_non_ms_mode = __gen_unpack_uint(cl, 71, 71);
   values->multisample_mode_4x = __gen_unpack_uint(cl, 70, 70);
   values->maximum_bpp_of_all_render_targets = __gen_unpack_uint(cl, 68, 69);
   values->number_of_render_targets = __gen_unpack_uint(cl, 64, 67);
   values->height_in_tiles = __gen_unpack_uint(cl, 52, 63);
   values->width_in_tiles = __gen_unpack_uint(cl, 40, 51);
   values->tile_state_data_array_base_address = __gen_unpack_address(cl, 14, 39);
   values->tile_allocation_block_size = __gen_unpack_uint(cl, 12, 13);
   values->tile_allocation_initial_block_size = __gen_unpack_uint(cl, 10, 11);
   values->auto_initialize_tile_state_data_array = __gen_unpack_uint(cl, 9, 9);
   values->sub_id = __gen_unpack_uint(cl, 8, 8);
}
#endif


#define V3D33_TILE_BINNING_MODE_CFG_PART2_opcode    120
#define V3D33_TILE_BINNING_MODE_CFG_PART2_header\
   .opcode                              =    120,  \
   .sub_id                              =      1

struct V3D33_TILE_BINNING_MODE_CFG_PART2 {
   uint32_t                             opcode;
   __gen_address_type                   tile_allocation_memory_address;
   uint32_t                             tile_allocation_memory_size;
   uint32_t                             sub_id;
};

static inline void
V3D33_TILE_BINNING_MODE_CFG_PART2_pack(__gen_user_data *data, uint8_t * restrict cl,
                                       const struct V3D33_TILE_BINNING_MODE_CFG_PART2 * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->tile_allocation_memory_size, 0, 31) |
            __gen_uint(values->sub_id, 0, 0);

   cl[ 2] = __gen_uint(values->tile_allocation_memory_size, 0, 31) >> 8;

   cl[ 3] = __gen_uint(values->tile_allocation_memory_size, 0, 31) >> 16;

   cl[ 4] = __gen_uint(values->tile_allocation_memory_size, 0, 31) >> 24;

   __gen_emit_reloc(data, &values->tile_allocation_memory_address);
   cl[ 5] = __gen_address_offset(&values->tile_allocation_memory_address);

   cl[ 6] = __gen_address_offset(&values->tile_allocation_memory_address) >> 8;

   cl[ 7] = __gen_address_offset(&values->tile_allocation_memory_address) >> 16;

   cl[ 8] = __gen_address_offset(&values->tile_allocation_memory_address) >> 24;

}

#define V3D33_TILE_BINNING_MODE_CFG_PART2_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_BINNING_MODE_CFG_PART2_unpack(const uint8_t * restrict cl,
                                         struct V3D33_TILE_BINNING_MODE_CFG_PART2 * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->tile_allocation_memory_address = __gen_unpack_address(cl, 40, 71);
   values->tile_allocation_memory_size = __gen_unpack_uint(cl, 8, 39);
   values->sub_id = __gen_unpack_uint(cl, 8, 8);
}
#endif


#define V3D33_TILE_RENDERING_MODE_CFG_COMMON_opcode    121
#define V3D33_TILE_RENDERING_MODE_CFG_COMMON_header\
   .opcode                              =    121,  \
   .sub_id                              =      0

struct V3D33_TILE_RENDERING_MODE_CFG_COMMON {
   uint32_t                             opcode;
   uint32_t                             disable_render_target_stores;
   bool                                 enable_z_store;
   bool                                 enable_stencil_store;
   bool                                 early_z_disable;
   uint32_t                             early_z_test_and_update_direction;
#define EARLY_Z_DIRECTION_LT_LE                  0
#define EARLY_Z_DIRECTION_GT_GE                  1
   bool                                 double_buffer_in_non_ms_mode;
   bool                                 multisample_mode_4x;
   uint32_t                             maximum_bpp_of_all_render_targets;
#define RENDER_TARGET_MAXIMUM_32BPP              0
#define RENDER_TARGET_MAXIMUM_64BPP              1
#define RENDER_TARGET_MAXIMUM_128BPP             2
   uint32_t                             image_height_pixels;
   uint32_t                             image_width_pixels;
   uint32_t                             number_of_render_targets;
   uint32_t                             sub_id;
};

static inline void
V3D33_TILE_RENDERING_MODE_CFG_COMMON_pack(__gen_user_data *data, uint8_t * restrict cl,
                                          const struct V3D33_TILE_RENDERING_MODE_CFG_COMMON * restrict values)
{
   assert(values->number_of_render_targets >= 1);
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->number_of_render_targets - 1, 4, 7) |
            __gen_uint(values->sub_id, 0, 3);

   cl[ 2] = __gen_uint(values->image_width_pixels, 0, 15);

   cl[ 3] = __gen_uint(values->image_width_pixels, 0, 15) >> 8;

   cl[ 4] = __gen_uint(values->image_height_pixels, 0, 15);

   cl[ 5] = __gen_uint(values->image_height_pixels, 0, 15) >> 8;

   cl[ 6] = __gen_uint(values->early_z_disable, 6, 6) |
            __gen_uint(values->early_z_test_and_update_direction, 5, 5) |
            __gen_uint(values->double_buffer_in_non_ms_mode, 3, 3) |
            __gen_uint(values->multisample_mode_4x, 2, 2) |
            __gen_uint(values->maximum_bpp_of_all_render_targets, 0, 1);

   cl[ 7] = __gen_uint(values->enable_z_store, 7, 7) |
            __gen_uint(values->enable_stencil_store, 6, 6);

   cl[ 8] = __gen_uint(values->disable_render_target_stores, 0, 7);

}

#define V3D33_TILE_RENDERING_MODE_CFG_COMMON_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_RENDERING_MODE_CFG_COMMON_unpack(const uint8_t * restrict cl,
                                            struct V3D33_TILE_RENDERING_MODE_CFG_COMMON * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->disable_render_target_stores = __gen_unpack_uint(cl, 64, 71);
   values->enable_z_store = __gen_unpack_uint(cl, 63, 63);
   values->enable_stencil_store = __gen_unpack_uint(cl, 62, 62);
   values->early_z_disable = __gen_unpack_uint(cl, 54, 54);
   values->early_z_test_and_update_direction = __gen_unpack_uint(cl, 53, 53);
   values->double_buffer_in_non_ms_mode = __gen_unpack_uint(cl, 51, 51);
   values->multisample_mode_4x = __gen_unpack_uint(cl, 50, 50);
   values->maximum_bpp_of_all_render_targets = __gen_unpack_uint(cl, 48, 49);
   values->image_height_pixels = __gen_unpack_uint(cl, 32, 47);
   values->image_width_pixels = __gen_unpack_uint(cl, 16, 31);
   values->number_of_render_targets = __gen_unpack_uint(cl, 12, 15) + 1;
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_TILE_RENDERING_MODE_CFG_COLOR_opcode    121
#define V3D33_TILE_RENDERING_MODE_CFG_COLOR_header\
   .opcode                              =    121,  \
   .sub_id                              =      2

struct V3D33_TILE_RENDERING_MODE_CFG_COLOR {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             pad;
   bool                                 flip_y;
   enum V3D33_Memory_Format             memory_format;
   enum V3D33_Dither_Mode               dither_mode;
   enum V3D33_Output_Image_Format       output_image_format;
   enum V3D33_Decimate_Mode             decimate_mode;
   enum V3D33_Internal_Type             internal_type;
   enum V3D33_Internal_BPP              internal_bpp;
   uint32_t                             render_target_number;
   uint32_t                             sub_id;
};

static inline void
V3D33_TILE_RENDERING_MODE_CFG_COLOR_pack(__gen_user_data *data, uint8_t * restrict cl,
                                         const struct V3D33_TILE_RENDERING_MODE_CFG_COLOR * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->render_target_number, 4, 7) |
            __gen_uint(values->sub_id, 0, 3);

   cl[ 2] = __gen_uint(values->decimate_mode, 6, 7) |
            __gen_uint(values->internal_type, 2, 5) |
            __gen_uint(values->internal_bpp, 0, 1);

   cl[ 3] = __gen_uint(values->dither_mode, 6, 7) |
            __gen_uint(values->output_image_format, 0, 5);

   cl[ 4] = __gen_uint(values->pad, 4, 7) |
            __gen_uint(values->flip_y, 3, 3) |
            __gen_uint(values->memory_format, 0, 2);

   __gen_emit_reloc(data, &values->address);
   cl[ 5] = __gen_address_offset(&values->address);

   cl[ 6] = __gen_address_offset(&values->address) >> 8;

   cl[ 7] = __gen_address_offset(&values->address) >> 16;

   cl[ 8] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_TILE_RENDERING_MODE_CFG_COLOR_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_RENDERING_MODE_CFG_COLOR_unpack(const uint8_t * restrict cl,
                                           struct V3D33_TILE_RENDERING_MODE_CFG_COLOR * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 40, 71);
   values->pad = __gen_unpack_uint(cl, 36, 39);
   values->flip_y = __gen_unpack_uint(cl, 35, 35);
   values->memory_format = __gen_unpack_uint(cl, 32, 34);
   values->dither_mode = __gen_unpack_uint(cl, 30, 31);
   values->output_image_format = __gen_unpack_uint(cl, 24, 29);
   values->decimate_mode = __gen_unpack_uint(cl, 22, 23);
   values->internal_type = __gen_unpack_uint(cl, 18, 21);
   values->internal_bpp = __gen_unpack_uint(cl, 16, 17);
   values->render_target_number = __gen_unpack_uint(cl, 12, 15);
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_TILE_RENDERING_MODE_CFG_Z_STENCIL_opcode    121
#define V3D33_TILE_RENDERING_MODE_CFG_Z_STENCIL_header\
   .opcode                              =    121,  \
   .z_stencil_id                        =      0,  \
   .sub_id                              =      1

struct V3D33_TILE_RENDERING_MODE_CFG_Z_STENCIL {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             padded_height_of_output_image_in_uif_blocks;
   enum V3D33_Memory_Format             memory_format;
   enum V3D33_Z_S_Output_Image_Format   output_image_format;
   uint32_t                             decimate_mode;
   enum V3D33_Internal_Depth_Type       internal_type;
   uint32_t                             internal_bpp_ignored;
   uint32_t                             z_stencil_id;
   uint32_t                             sub_id;
};

static inline void
V3D33_TILE_RENDERING_MODE_CFG_Z_STENCIL_pack(__gen_user_data *data, uint8_t * restrict cl,
                                             const struct V3D33_TILE_RENDERING_MODE_CFG_Z_STENCIL * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->z_stencil_id, 4, 7) |
            __gen_uint(values->sub_id, 0, 3);

   cl[ 2] = __gen_uint(values->decimate_mode, 6, 7) |
            __gen_uint(values->internal_type, 2, 5) |
            __gen_uint(values->internal_bpp_ignored, 0, 1);

   cl[ 3] = __gen_uint(values->memory_format, 6, 8) |
            __gen_uint(values->output_image_format, 0, 5);

   cl[ 4] = __gen_uint(values->padded_height_of_output_image_in_uif_blocks, 1, 13) |
            __gen_uint(values->memory_format, 6, 8) >> 8;

   __gen_emit_reloc(data, &values->address);
   cl[ 5] = __gen_address_offset(&values->address) |
            __gen_uint(values->padded_height_of_output_image_in_uif_blocks, 1, 13) >> 8;

   cl[ 6] = __gen_address_offset(&values->address) >> 8;

   cl[ 7] = __gen_address_offset(&values->address) >> 16;

   cl[ 8] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_TILE_RENDERING_MODE_CFG_Z_STENCIL_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_RENDERING_MODE_CFG_Z_STENCIL_unpack(const uint8_t * restrict cl,
                                               struct V3D33_TILE_RENDERING_MODE_CFG_Z_STENCIL * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 46, 71);
   values->padded_height_of_output_image_in_uif_blocks = __gen_unpack_uint(cl, 33, 45);
   values->memory_format = __gen_unpack_uint(cl, 30, 32);
   values->output_image_format = __gen_unpack_uint(cl, 24, 29);
   values->decimate_mode = __gen_unpack_uint(cl, 22, 23);
   values->internal_type = __gen_unpack_uint(cl, 18, 21);
   values->internal_bpp_ignored = __gen_unpack_uint(cl, 16, 17);
   values->z_stencil_id = __gen_unpack_uint(cl, 12, 15);
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES_opcode    121
#define V3D33_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES_header\
   .opcode                              =    121,  \
   .sub_id                              =      3

struct V3D33_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES {
   uint32_t                             opcode;
   uint32_t                             unused;
   float                                z_clear_value;
   uint32_t                             stencil_clear_value;
   uint32_t                             sub_id;
};

static inline void
V3D33_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                   const struct V3D33_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->sub_id, 0, 3);

   cl[ 2] = __gen_uint(values->stencil_clear_value, 0, 7);


   memcpy(&cl[3], &values->z_clear_value, sizeof(values->z_clear_value));
   cl[ 7] = __gen_uint(values->unused, 0, 15);

   cl[ 8] = __gen_uint(values->unused, 0, 15) >> 8;

}

#define V3D33_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES_unpack(const uint8_t * restrict cl,
                                                     struct V3D33_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->unused = __gen_unpack_uint(cl, 56, 71);
   values->z_clear_value = __gen_unpack_float(cl, 24, 55);
   values->stencil_clear_value = __gen_unpack_uint(cl, 16, 23);
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1_opcode    121
#define V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1_header\
   .opcode                              =    121,  \
   .sub_id                              =      4

struct V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1 {
   uint32_t                             opcode;
   uint32_t                             clear_color_next_24_bits;
   uint32_t                             clear_color_low_32_bits;
   uint32_t                             render_target_number;
   uint32_t                             sub_id;
};

static inline void
V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                      const struct V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1 * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->render_target_number, 4, 7) |
            __gen_uint(values->sub_id, 0, 3);


   memcpy(&cl[2], &values->clear_color_low_32_bits, sizeof(values->clear_color_low_32_bits));
   cl[ 6] = __gen_uint(values->clear_color_next_24_bits, 0, 23);

   cl[ 7] = __gen_uint(values->clear_color_next_24_bits, 0, 23) >> 8;

   cl[ 8] = __gen_uint(values->clear_color_next_24_bits, 0, 23) >> 16;

}

#define V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1_unpack(const uint8_t * restrict cl,
                                                        struct V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1 * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->clear_color_next_24_bits = __gen_unpack_uint(cl, 48, 71);
   values->clear_color_low_32_bits = __gen_unpack_uint(cl, 16, 47);
   values->render_target_number = __gen_unpack_uint(cl, 12, 15);
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2_opcode    121
#define V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2_header\
   .opcode                              =    121,  \
   .sub_id                              =      5

struct V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2 {
   uint32_t                             opcode;
   uint32_t                             clear_color_mid_high_24_bits;
   uint32_t                             clear_color_mid_low_32_bits;
   uint32_t                             render_target_number;
   uint32_t                             sub_id;
};

static inline void
V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                      const struct V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2 * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->render_target_number, 4, 7) |
            __gen_uint(values->sub_id, 0, 3);


   memcpy(&cl[2], &values->clear_color_mid_low_32_bits, sizeof(values->clear_color_mid_low_32_bits));
   cl[ 6] = __gen_uint(values->clear_color_mid_high_24_bits, 0, 23);

   cl[ 7] = __gen_uint(values->clear_color_mid_high_24_bits, 0, 23) >> 8;

   cl[ 8] = __gen_uint(values->clear_color_mid_high_24_bits, 0, 23) >> 16;

}

#define V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2_unpack(const uint8_t * restrict cl,
                                                        struct V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2 * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->clear_color_mid_high_24_bits = __gen_unpack_uint(cl, 48, 71);
   values->clear_color_mid_low_32_bits = __gen_unpack_uint(cl, 16, 47);
   values->render_target_number = __gen_unpack_uint(cl, 12, 15);
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3_opcode    121
#define V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3_header\
   .opcode                              =    121,  \
   .sub_id                              =      6

struct V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3 {
   uint32_t                             opcode;
   uint32_t                             pad;
   uint32_t                             uif_padded_height_in_uif_blocks;
   uint32_t                             raster_row_stride_or_image_height_in_pixels;
   uint32_t                             clear_color_high_16_bits;
   uint32_t                             render_target_number;
   uint32_t                             sub_id;
};

static inline void
V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                      const struct V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3 * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->render_target_number, 4, 7) |
            __gen_uint(values->sub_id, 0, 3);

   cl[ 2] = __gen_uint(values->clear_color_high_16_bits, 0, 15);

   cl[ 3] = __gen_uint(values->clear_color_high_16_bits, 0, 15) >> 8;

   cl[ 4] = __gen_uint(values->raster_row_stride_or_image_height_in_pixels, 0, 15);

   cl[ 5] = __gen_uint(values->raster_row_stride_or_image_height_in_pixels, 0, 15) >> 8;

   cl[ 6] = __gen_uint(values->uif_padded_height_in_uif_blocks, 0, 12);

   cl[ 7] = __gen_uint(values->pad, 5, 15) |
            __gen_uint(values->uif_padded_height_in_uif_blocks, 0, 12) >> 8;

   cl[ 8] = __gen_uint(values->pad, 5, 15) >> 8;

}

#define V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3_unpack(const uint8_t * restrict cl,
                                                        struct V3D33_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3 * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->pad = __gen_unpack_uint(cl, 61, 71);
   values->uif_padded_height_in_uif_blocks = __gen_unpack_uint(cl, 48, 60);
   values->raster_row_stride_or_image_height_in_pixels = __gen_unpack_uint(cl, 32, 47);
   values->clear_color_high_16_bits = __gen_unpack_uint(cl, 16, 31);
   values->render_target_number = __gen_unpack_uint(cl, 12, 15);
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_TILE_COORDINATES_opcode        124
#define V3D33_TILE_COORDINATES_header           \
   .opcode                              =    124

struct V3D33_TILE_COORDINATES {
   uint32_t                             opcode;
   uint32_t                             tile_row_number;
   uint32_t                             tile_column_number;
};

static inline void
V3D33_TILE_COORDINATES_pack(__gen_user_data *data, uint8_t * restrict cl,
                            const struct V3D33_TILE_COORDINATES * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->tile_column_number, 0, 11);

   cl[ 2] = __gen_uint(values->tile_row_number, 4, 15) |
            __gen_uint(values->tile_column_number, 0, 11) >> 8;

   cl[ 3] = __gen_uint(values->tile_row_number, 4, 15) >> 8;

}

#define V3D33_TILE_COORDINATES_length          4
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_COORDINATES_unpack(const uint8_t * restrict cl,
                              struct V3D33_TILE_COORDINATES * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->tile_row_number = __gen_unpack_uint(cl, 20, 31);
   values->tile_column_number = __gen_unpack_uint(cl, 8, 19);
}
#endif


#define V3D33_MULTICORE_RENDERING_SUPERTILE_CFG_opcode    122
#define V3D33_MULTICORE_RENDERING_SUPERTILE_CFG_header\
   .opcode                              =    122

struct V3D33_MULTICORE_RENDERING_SUPERTILE_CFG {
   uint32_t                             opcode;
   uint32_t                             number_of_bin_tile_lists;
   bool                                 supertile_raster_order;
   bool                                 multicore_enable;
   uint32_t                             total_frame_height_in_tiles;
   uint32_t                             total_frame_width_in_tiles;
   uint32_t                             total_frame_height_in_supertiles;
   uint32_t                             total_frame_width_in_supertiles;
   uint32_t                             supertile_height_in_tiles;
   uint32_t                             supertile_width_in_tiles;
};

static inline void
V3D33_MULTICORE_RENDERING_SUPERTILE_CFG_pack(__gen_user_data *data, uint8_t * restrict cl,
                                             const struct V3D33_MULTICORE_RENDERING_SUPERTILE_CFG * restrict values)
{
   assert(values->number_of_bin_tile_lists >= 1);
   assert(values->supertile_height_in_tiles >= 1);
   assert(values->supertile_width_in_tiles >= 1);
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->supertile_width_in_tiles - 1, 0, 7);

   cl[ 2] = __gen_uint(values->supertile_height_in_tiles - 1, 0, 7);

   cl[ 3] = __gen_uint(values->total_frame_width_in_supertiles, 0, 7);

   cl[ 4] = __gen_uint(values->total_frame_height_in_supertiles, 0, 7);

   cl[ 5] = __gen_uint(values->total_frame_width_in_tiles, 0, 11);

   cl[ 6] = __gen_uint(values->total_frame_height_in_tiles, 4, 15) |
            __gen_uint(values->total_frame_width_in_tiles, 0, 11) >> 8;

   cl[ 7] = __gen_uint(values->total_frame_height_in_tiles, 4, 15) >> 8;

   cl[ 8] = __gen_uint(values->number_of_bin_tile_lists - 1, 5, 7) |
            __gen_uint(values->supertile_raster_order, 4, 4) |
            __gen_uint(values->multicore_enable, 0, 0);

}

#define V3D33_MULTICORE_RENDERING_SUPERTILE_CFG_length      9
#ifdef __gen_unpack_address
static inline void
V3D33_MULTICORE_RENDERING_SUPERTILE_CFG_unpack(const uint8_t * restrict cl,
                                               struct V3D33_MULTICORE_RENDERING_SUPERTILE_CFG * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->number_of_bin_tile_lists = __gen_unpack_uint(cl, 69, 71) + 1;
   values->supertile_raster_order = __gen_unpack_uint(cl, 68, 68);
   values->multicore_enable = __gen_unpack_uint(cl, 64, 64);
   values->total_frame_height_in_tiles = __gen_unpack_uint(cl, 52, 63);
   values->total_frame_width_in_tiles = __gen_unpack_uint(cl, 40, 51);
   values->total_frame_height_in_supertiles = __gen_unpack_uint(cl, 32, 39);
   values->total_frame_width_in_supertiles = __gen_unpack_uint(cl, 24, 31);
   values->supertile_height_in_tiles = __gen_unpack_uint(cl, 16, 23) + 1;
   values->supertile_width_in_tiles = __gen_unpack_uint(cl, 8, 15) + 1;
}
#endif


#define V3D33_MULTICORE_RENDERING_TILE_LIST_SET_BASE_opcode    123
#define V3D33_MULTICORE_RENDERING_TILE_LIST_SET_BASE_header\
   .opcode                              =    123

struct V3D33_MULTICORE_RENDERING_TILE_LIST_SET_BASE {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             tile_list_set_number;
};

static inline void
V3D33_MULTICORE_RENDERING_TILE_LIST_SET_BASE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                  const struct V3D33_MULTICORE_RENDERING_TILE_LIST_SET_BASE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address) |
            __gen_uint(values->tile_list_set_number, 0, 3);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D33_MULTICORE_RENDERING_TILE_LIST_SET_BASE_length      5
#ifdef __gen_unpack_address
static inline void
V3D33_MULTICORE_RENDERING_TILE_LIST_SET_BASE_unpack(const uint8_t * restrict cl,
                                                    struct V3D33_MULTICORE_RENDERING_TILE_LIST_SET_BASE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 14, 39);
   values->tile_list_set_number = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D33_TILE_COORDINATES_IMPLICIT_opcode    125
#define V3D33_TILE_COORDINATES_IMPLICIT_header  \
   .opcode                              =    125

struct V3D33_TILE_COORDINATES_IMPLICIT {
   uint32_t                             opcode;
};

static inline void
V3D33_TILE_COORDINATES_IMPLICIT_pack(__gen_user_data *data, uint8_t * restrict cl,
                                     const struct V3D33_TILE_COORDINATES_IMPLICIT * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D33_TILE_COORDINATES_IMPLICIT_length      1
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_COORDINATES_IMPLICIT_unpack(const uint8_t * restrict cl,
                                       struct V3D33_TILE_COORDINATES_IMPLICIT * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D33_TILE_LIST_INITIAL_BLOCK_SIZE_opcode    126
#define V3D33_TILE_LIST_INITIAL_BLOCK_SIZE_header\
   .opcode                              =    126

struct V3D33_TILE_LIST_INITIAL_BLOCK_SIZE {
   uint32_t                             opcode;
   bool                                 use_auto_chained_tile_lists;
   uint32_t                             size_of_first_block_in_chained_tile_lists;
#define TILE_ALLOCATION_BLOCK_SIZE_64B           0
#define TILE_ALLOCATION_BLOCK_SIZE_128B          1
#define TILE_ALLOCATION_BLOCK_SIZE_256B          2
};

static inline void
V3D33_TILE_LIST_INITIAL_BLOCK_SIZE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D33_TILE_LIST_INITIAL_BLOCK_SIZE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->use_auto_chained_tile_lists, 2, 2) |
            __gen_uint(values->size_of_first_block_in_chained_tile_lists, 0, 1);

}

#define V3D33_TILE_LIST_INITIAL_BLOCK_SIZE_length      2
#ifdef __gen_unpack_address
static inline void
V3D33_TILE_LIST_INITIAL_BLOCK_SIZE_unpack(const uint8_t * restrict cl,
                                          struct V3D33_TILE_LIST_INITIAL_BLOCK_SIZE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->use_auto_chained_tile_lists = __gen_unpack_uint(cl, 10, 10);
   values->size_of_first_block_in_chained_tile_lists = __gen_unpack_uint(cl, 8, 9);
}
#endif


#define V3D33_GL_SHADER_STATE_RECORD_header     \


struct V3D33_GL_SHADER_STATE_RECORD {
   bool                                 point_size_in_shaded_vertex_data;
   bool                                 enable_clipping;
   bool                                 vertex_id_read_by_coordinate_shader;
   bool                                 instance_id_read_by_coordinate_shader;
   bool                                 vertex_id_read_by_vertex_shader;
   bool                                 instance_id_read_by_vertex_shader;
   bool                                 fragment_shader_does_z_writes;
   bool                                 turn_off_early_z_test;
   bool                                 coordinate_shader_has_separate_input_and_output_vpm_blocks;
   bool                                 vertex_shader_has_separate_input_and_output_vpm_blocks;
   bool                                 fragment_shader_uses_real_pixel_centre_w_in_addition_to_centroid_w2;
   uint32_t                             number_of_varyings_in_fragment_shader;
   uint32_t                             coordinate_shader_output_vpm_segment_size;
   uint32_t                             coordinate_shader_input_vpm_segment_size;
   uint32_t                             vertex_shader_output_vpm_segment_size;
   uint32_t                             vertex_shader_input_vpm_segment_size;
   __gen_address_type                   address_of_default_attribute_values;
   __gen_address_type                   fragment_shader_code_address;
   bool                                 fragment_shader_2_way_threadable;
   bool                                 fragment_shader_4_way_threadable;
   bool                                 fragment_shader_propagate_nans;
   __gen_address_type                   fragment_shader_uniforms_address;
   __gen_address_type                   vertex_shader_code_address;
   bool                                 vertex_shader_2_way_threadable;
   bool                                 vertex_shader_4_way_threadable;
   bool                                 vertex_shader_propagate_nans;
   __gen_address_type                   vertex_shader_uniforms_address;
   __gen_address_type                   coordinate_shader_code_address;
   bool                                 coordinate_shader_2_way_threadable;
   bool                                 coordinate_shader_4_way_threadable;
   bool                                 coordinate_shader_propagate_nans;
   __gen_address_type                   coordinate_shader_uniforms_address;
};

static inline void
V3D33_GL_SHADER_STATE_RECORD_pack(__gen_user_data *data, uint8_t * restrict cl,
                                  const struct V3D33_GL_SHADER_STATE_RECORD * restrict values)
{
   cl[ 0] = __gen_uint(values->point_size_in_shaded_vertex_data, 0, 0) |
            __gen_uint(values->enable_clipping, 1, 1) |
            __gen_uint(values->vertex_id_read_by_coordinate_shader, 2, 2) |
            __gen_uint(values->instance_id_read_by_coordinate_shader, 3, 3) |
            __gen_uint(values->vertex_id_read_by_vertex_shader, 4, 4) |
            __gen_uint(values->instance_id_read_by_vertex_shader, 5, 5) |
            __gen_uint(values->fragment_shader_does_z_writes, 6, 6) |
            __gen_uint(values->turn_off_early_z_test, 7, 7);

   cl[ 1] = __gen_uint(values->coordinate_shader_has_separate_input_and_output_vpm_blocks, 0, 0) |
            __gen_uint(values->vertex_shader_has_separate_input_and_output_vpm_blocks, 1, 1) |
            __gen_uint(values->fragment_shader_uses_real_pixel_centre_w_in_addition_to_centroid_w2, 2, 2);

   cl[ 2] = __gen_uint(values->number_of_varyings_in_fragment_shader, 0, 7);

   cl[ 3] = 0;
   cl[ 4] = __gen_uint(values->coordinate_shader_output_vpm_segment_size, 0, 7);

   cl[ 5] = __gen_uint(values->coordinate_shader_input_vpm_segment_size, 0, 7);

   cl[ 6] = __gen_uint(values->vertex_shader_output_vpm_segment_size, 0, 7);

   cl[ 7] = __gen_uint(values->vertex_shader_input_vpm_segment_size, 0, 7);

   __gen_emit_reloc(data, &values->address_of_default_attribute_values);
   cl[ 8] = __gen_address_offset(&values->address_of_default_attribute_values);

   cl[ 9] = __gen_address_offset(&values->address_of_default_attribute_values) >> 8;

   cl[10] = __gen_address_offset(&values->address_of_default_attribute_values) >> 16;

   cl[11] = __gen_address_offset(&values->address_of_default_attribute_values) >> 24;

   __gen_emit_reloc(data, &values->fragment_shader_code_address);
   cl[12] = __gen_address_offset(&values->fragment_shader_code_address) |
            __gen_uint(values->fragment_shader_2_way_threadable, 0, 0) |
            __gen_uint(values->fragment_shader_4_way_threadable, 1, 1) |
            __gen_uint(values->fragment_shader_propagate_nans, 2, 2);

   cl[13] = __gen_address_offset(&values->fragment_shader_code_address) >> 8;

   cl[14] = __gen_address_offset(&values->fragment_shader_code_address) >> 16;

   cl[15] = __gen_address_offset(&values->fragment_shader_code_address) >> 24;

   __gen_emit_reloc(data, &values->fragment_shader_uniforms_address);
   cl[16] = __gen_address_offset(&values->fragment_shader_uniforms_address);

   cl[17] = __gen_address_offset(&values->fragment_shader_uniforms_address) >> 8;

   cl[18] = __gen_address_offset(&values->fragment_shader_uniforms_address) >> 16;

   cl[19] = __gen_address_offset(&values->fragment_shader_uniforms_address) >> 24;

   __gen_emit_reloc(data, &values->vertex_shader_code_address);
   cl[20] = __gen_address_offset(&values->vertex_shader_code_address) |
            __gen_uint(values->vertex_shader_2_way_threadable, 0, 0) |
            __gen_uint(values->vertex_shader_4_way_threadable, 1, 1) |
            __gen_uint(values->vertex_shader_propagate_nans, 2, 2);

   cl[21] = __gen_address_offset(&values->vertex_shader_code_address) >> 8;

   cl[22] = __gen_address_offset(&values->vertex_shader_code_address) >> 16;

   cl[23] = __gen_address_offset(&values->vertex_shader_code_address) >> 24;

   __gen_emit_reloc(data, &values->vertex_shader_uniforms_address);
   cl[24] = __gen_address_offset(&values->vertex_shader_uniforms_address);

   cl[25] = __gen_address_offset(&values->vertex_shader_uniforms_address) >> 8;

   cl[26] = __gen_address_offset(&values->vertex_shader_uniforms_address) >> 16;

   cl[27] = __gen_address_offset(&values->vertex_shader_uniforms_address) >> 24;

   __gen_emit_reloc(data, &values->coordinate_shader_code_address);
   cl[28] = __gen_address_offset(&values->coordinate_shader_code_address) |
            __gen_uint(values->coordinate_shader_2_way_threadable, 0, 0) |
            __gen_uint(values->coordinate_shader_4_way_threadable, 1, 1) |
            __gen_uint(values->coordinate_shader_propagate_nans, 2, 2);

   cl[29] = __gen_address_offset(&values->coordinate_shader_code_address) >> 8;

   cl[30] = __gen_address_offset(&values->coordinate_shader_code_address) >> 16;

   cl[31] = __gen_address_offset(&values->coordinate_shader_code_address) >> 24;

   __gen_emit_reloc(data, &values->coordinate_shader_uniforms_address);
   cl[32] = __gen_address_offset(&values->coordinate_shader_uniforms_address);

   cl[33] = __gen_address_offset(&values->coordinate_shader_uniforms_address) >> 8;

   cl[34] = __gen_address_offset(&values->coordinate_shader_uniforms_address) >> 16;

   cl[35] = __gen_address_offset(&values->coordinate_shader_uniforms_address) >> 24;

}

#define V3D33_GL_SHADER_STATE_RECORD_length     36
#ifdef __gen_unpack_address
static inline void
V3D33_GL_SHADER_STATE_RECORD_unpack(const uint8_t * restrict cl,
                                    struct V3D33_GL_SHADER_STATE_RECORD * restrict values)
{
   values->point_size_in_shaded_vertex_data = __gen_unpack_uint(cl, 0, 0);
   values->enable_clipping = __gen_unpack_uint(cl, 1, 1);
   values->vertex_id_read_by_coordinate_shader = __gen_unpack_uint(cl, 2, 2);
   values->instance_id_read_by_coordinate_shader = __gen_unpack_uint(cl, 3, 3);
   values->vertex_id_read_by_vertex_shader = __gen_unpack_uint(cl, 4, 4);
   values->instance_id_read_by_vertex_shader = __gen_unpack_uint(cl, 5, 5);
   values->fragment_shader_does_z_writes = __gen_unpack_uint(cl, 6, 6);
   values->turn_off_early_z_test = __gen_unpack_uint(cl, 7, 7);
   values->coordinate_shader_has_separate_input_and_output_vpm_blocks = __gen_unpack_uint(cl, 8, 8);
   values->vertex_shader_has_separate_input_and_output_vpm_blocks = __gen_unpack_uint(cl, 9, 9);
   values->fragment_shader_uses_real_pixel_centre_w_in_addition_to_centroid_w2 = __gen_unpack_uint(cl, 10, 10);
   values->number_of_varyings_in_fragment_shader = __gen_unpack_uint(cl, 16, 23);
   values->coordinate_shader_output_vpm_segment_size = __gen_unpack_uint(cl, 32, 39);
   values->coordinate_shader_input_vpm_segment_size = __gen_unpack_uint(cl, 40, 47);
   values->vertex_shader_output_vpm_segment_size = __gen_unpack_uint(cl, 48, 55);
   values->vertex_shader_input_vpm_segment_size = __gen_unpack_uint(cl, 56, 63);
   values->address_of_default_attribute_values = __gen_unpack_address(cl, 64, 95);
   values->fragment_shader_code_address = __gen_unpack_address(cl, 99, 127);
   values->fragment_shader_2_way_threadable = __gen_unpack_uint(cl, 96, 96);
   values->fragment_shader_4_way_threadable = __gen_unpack_uint(cl, 97, 97);
   values->fragment_shader_propagate_nans = __gen_unpack_uint(cl, 98, 98);
   values->fragment_shader_uniforms_address = __gen_unpack_address(cl, 128, 159);
   values->vertex_shader_code_address = __gen_unpack_address(cl, 160, 191);
   values->vertex_shader_2_way_threadable = __gen_unpack_uint(cl, 160, 160);
   values->vertex_shader_4_way_threadable = __gen_unpack_uint(cl, 161, 161);
   values->vertex_shader_propagate_nans = __gen_unpack_uint(cl, 162, 162);
   values->vertex_shader_uniforms_address = __gen_unpack_address(cl, 192, 223);
   values->coordinate_shader_code_address = __gen_unpack_address(cl, 224, 255);
   values->coordinate_shader_2_way_threadable = __gen_unpack_uint(cl, 224, 224);
   values->coordinate_shader_4_way_threadable = __gen_unpack_uint(cl, 225, 225);
   values->coordinate_shader_propagate_nans = __gen_unpack_uint(cl, 226, 226);
   values->coordinate_shader_uniforms_address = __gen_unpack_address(cl, 256, 287);
}
#endif


#define V3D33_GL_SHADER_STATE_ATTRIBUTE_RECORD_header\


struct V3D33_GL_SHADER_STATE_ATTRIBUTE_RECORD {
   __gen_address_type                   address;
   uint32_t                             vec_size;
   uint32_t                             type;
#define ATTRIBUTE_HALF_FLOAT                     1
#define ATTRIBUTE_FLOAT                          2
#define ATTRIBUTE_FIXED                          3
#define ATTRIBUTE_BYTE                           4
#define ATTRIBUTE_SHORT                          5
#define ATTRIBUTE_INT                            6
#define ATTRIBUTE_INT2_10_10_10                  7
   bool                                 signed_int_type;
   bool                                 normalized_int_type;
   bool                                 read_as_int_uint;
   uint32_t                             number_of_values_read_by_coordinate_shader;
   uint32_t                             number_of_values_read_by_vertex_shader;
   uint32_t                             instance_divisor;
   uint32_t                             stride;
};

static inline void
V3D33_GL_SHADER_STATE_ATTRIBUTE_RECORD_pack(__gen_user_data *data, uint8_t * restrict cl,
                                            const struct V3D33_GL_SHADER_STATE_ATTRIBUTE_RECORD * restrict values)
{
   __gen_emit_reloc(data, &values->address);
   cl[ 0] = __gen_address_offset(&values->address);

   cl[ 1] = __gen_address_offset(&values->address) >> 8;

   cl[ 2] = __gen_address_offset(&values->address) >> 16;

   cl[ 3] = __gen_address_offset(&values->address) >> 24;

   cl[ 4] = __gen_uint(values->vec_size, 0, 1) |
            __gen_uint(values->type, 2, 4) |
            __gen_uint(values->signed_int_type, 5, 5) |
            __gen_uint(values->normalized_int_type, 6, 6) |
            __gen_uint(values->read_as_int_uint, 7, 7);

   cl[ 5] = __gen_uint(values->number_of_values_read_by_coordinate_shader, 0, 3) |
            __gen_uint(values->number_of_values_read_by_vertex_shader, 4, 7);

   cl[ 6] = __gen_uint(values->instance_divisor, 0, 15);

   cl[ 7] = __gen_uint(values->instance_divisor, 0, 15) >> 8;


   memcpy(&cl[8], &values->stride, sizeof(values->stride));
}

#define V3D33_GL_SHADER_STATE_ATTRIBUTE_RECORD_length     12
#ifdef __gen_unpack_address
static inline void
V3D33_GL_SHADER_STATE_ATTRIBUTE_RECORD_unpack(const uint8_t * restrict cl,
                                              struct V3D33_GL_SHADER_STATE_ATTRIBUTE_RECORD * restrict values)
{
   values->address = __gen_unpack_address(cl, 0, 31);
   values->vec_size = __gen_unpack_uint(cl, 32, 33);
   values->type = __gen_unpack_uint(cl, 34, 36);
   values->signed_int_type = __gen_unpack_uint(cl, 37, 37);
   values->normalized_int_type = __gen_unpack_uint(cl, 38, 38);
   values->read_as_int_uint = __gen_unpack_uint(cl, 39, 39);
   values->number_of_values_read_by_coordinate_shader = __gen_unpack_uint(cl, 40, 43);
   values->number_of_values_read_by_vertex_shader = __gen_unpack_uint(cl, 44, 47);
   values->instance_divisor = __gen_unpack_uint(cl, 48, 63);
   values->stride = __gen_unpack_uint(cl, 64, 95);
}
#endif


#define V3D33_VPM_GENERIC_BLOCK_WRITE_SETUP_header\
   .id                                  =      0,  \
   .id0                                 =      0

struct V3D33_VPM_GENERIC_BLOCK_WRITE_SETUP {
   uint32_t                             id;
   uint32_t                             id0;
   bool                                 horiz;
   bool                                 laned;
   bool                                 segs;
   int32_t                              stride;
   uint32_t                             size;
#define VPM_SETUP_SIZE_8_BIT                     0
#define VPM_SETUP_SIZE_16_BIT                    1
#define VPM_SETUP_SIZE_32_BIT                    2
   uint32_t                             addr;
};

static inline void
V3D33_VPM_GENERIC_BLOCK_WRITE_SETUP_pack(__gen_user_data *data, uint8_t * restrict cl,
                                         const struct V3D33_VPM_GENERIC_BLOCK_WRITE_SETUP * restrict values)
{
   cl[ 0] = __gen_uint(values->addr, 0, 12);

   cl[ 1] = __gen_sint(values->stride, 7, 13) |
            __gen_uint(values->size, 5, 6) |
            __gen_uint(values->addr, 0, 12) >> 8;

   cl[ 2] = __gen_uint(values->laned, 7, 7) |
            __gen_uint(values->segs, 6, 6) |
            __gen_sint(values->stride, 7, 13) >> 8;

   cl[ 3] = __gen_uint(values->id, 6, 7) |
            __gen_uint(values->id0, 3, 5) |
            __gen_uint(values->horiz, 0, 0);

}

#define V3D33_VPM_GENERIC_BLOCK_WRITE_SETUP_length      4
#ifdef __gen_unpack_address
static inline void
V3D33_VPM_GENERIC_BLOCK_WRITE_SETUP_unpack(const uint8_t * restrict cl,
                                           struct V3D33_VPM_GENERIC_BLOCK_WRITE_SETUP * restrict values)
{
   values->id = __gen_unpack_uint(cl, 30, 31);
   values->id0 = __gen_unpack_uint(cl, 27, 29);
   values->horiz = __gen_unpack_uint(cl, 24, 24);
   values->laned = __gen_unpack_uint(cl, 23, 23);
   values->segs = __gen_unpack_uint(cl, 22, 22);
   values->stride = __gen_unpack_sint(cl, 15, 21);
   values->size = __gen_unpack_uint(cl, 13, 14);
   values->addr = __gen_unpack_uint(cl, 0, 12);
}
#endif


#define V3D33_VPM_GENERIC_BLOCK_READ_SETUP_header\
   .id                                  =      1

struct V3D33_VPM_GENERIC_BLOCK_READ_SETUP {
   uint32_t                             id;
   bool                                 horiz;
   bool                                 laned;
   bool                                 segs;
   uint32_t                             num;
   int32_t                              stride;
   uint32_t                             size;
#define VPM_SETUP_SIZE_8_BIT                     0
#define VPM_SETUP_SIZE_16_BIT                    1
#define VPM_SETUP_SIZE_32_BIT                    2
   uint32_t                             addr;
};

static inline void
V3D33_VPM_GENERIC_BLOCK_READ_SETUP_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D33_VPM_GENERIC_BLOCK_READ_SETUP * restrict values)
{
   cl[ 0] = __gen_uint(values->addr, 0, 12);

   cl[ 1] = __gen_sint(values->stride, 7, 13) |
            __gen_uint(values->size, 5, 6) |
            __gen_uint(values->addr, 0, 12) >> 8;

   cl[ 2] = __gen_uint(values->num, 6, 10) |
            __gen_sint(values->stride, 7, 13) >> 8;

   cl[ 3] = __gen_uint(values->id, 6, 7) |
            __gen_uint(values->horiz, 5, 5) |
            __gen_uint(values->laned, 4, 4) |
            __gen_uint(values->segs, 3, 3) |
            __gen_uint(values->num, 6, 10) >> 8;

}

#define V3D33_VPM_GENERIC_BLOCK_READ_SETUP_length      4
#ifdef __gen_unpack_address
static inline void
V3D33_VPM_GENERIC_BLOCK_READ_SETUP_unpack(const uint8_t * restrict cl,
                                          struct V3D33_VPM_GENERIC_BLOCK_READ_SETUP * restrict values)
{
   values->id = __gen_unpack_uint(cl, 30, 31);
   values->horiz = __gen_unpack_uint(cl, 29, 29);
   values->laned = __gen_unpack_uint(cl, 28, 28);
   values->segs = __gen_unpack_uint(cl, 27, 27);
   values->num = __gen_unpack_uint(cl, 22, 26);
   values->stride = __gen_unpack_sint(cl, 15, 21);
   values->size = __gen_unpack_uint(cl, 13, 14);
   values->addr = __gen_unpack_uint(cl, 0, 12);
}
#endif


#define V3D33_TEXTURE_UNIFORM_PARAMETER_0_CFG_MODE1_header\
   .new_configuration_mode              =      1

struct V3D33_TEXTURE_UNIFORM_PARAMETER_0_CFG_MODE1 {
   bool                                 per_pixel_mask_enable;
   int32_t                              texel_offset_for_r_coordinate;
   int32_t                              texel_offset_for_t_coordinate;
   int32_t                              texel_offset_for_s_coordinate;
   uint32_t                             r_wrap_mode;
#define WRAP_MODE_REPEAT                         0
#define WRAP_MODE_CLAMP                          1
#define WRAP_MODE_MIRROR                         2
#define WRAP_MODE_BORDER                         3
#define WRAP_MODE_MIRROR_ONCE                    4
   uint32_t                             t_wrap_mode;
#define WRAP_MODE_REPEAT                         0
#define WRAP_MODE_CLAMP                          1
#define WRAP_MODE_MIRROR                         2
#define WRAP_MODE_BORDER                         3
#define WRAP_MODE_MIRROR_ONCE                    4
   uint32_t                             s_wrap_mode;
#define WRAP_MODE_REPEAT                         0
#define WRAP_MODE_CLAMP                          1
#define WRAP_MODE_MIRROR                         2
#define WRAP_MODE_BORDER                         3
#define WRAP_MODE_MIRROR_ONCE                    4
   bool                                 new_configuration_mode;
   bool                                 shadow;
   bool                                 coefficient_lookup_mode;
   bool                                 disable_autolod_use_bias_only;
   bool                                 bias_supplied;
   bool                                 gather_sample_mode;
   bool                                 fetch_sample_mode;
   uint32_t                             lookup_type;
#define TEXTURE_2D                               0
#define TEXTURE_2D_ARRAY                         1
#define TEXTURE_3D                               2
#define TEXTURE_CUBE_MAP                         3
#define TEXTURE_1D                               4
#define TEXTURE_1D_ARRAY                         5
#define TEXTURE_CHILD_IMAGE                      6
};

static inline void
V3D33_TEXTURE_UNIFORM_PARAMETER_0_CFG_MODE1_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                 const struct V3D33_TEXTURE_UNIFORM_PARAMETER_0_CFG_MODE1 * restrict values)
{
   cl[ 0] = __gen_uint(values->coefficient_lookup_mode, 7, 7) |
            __gen_uint(values->disable_autolod_use_bias_only, 6, 6) |
            __gen_uint(values->bias_supplied, 5, 5) |
            __gen_uint(values->gather_sample_mode, 4, 4) |
            __gen_uint(values->fetch_sample_mode, 3, 3) |
            __gen_uint(values->lookup_type, 0, 2);

   cl[ 1] = __gen_uint(values->t_wrap_mode, 5, 7) |
            __gen_uint(values->s_wrap_mode, 2, 4) |
            __gen_uint(values->new_configuration_mode, 1, 1) |
            __gen_uint(values->shadow, 0, 0);

   cl[ 2] = __gen_sint(values->texel_offset_for_t_coordinate, 7, 10) |
            __gen_sint(values->texel_offset_for_s_coordinate, 3, 6) |
            __gen_uint(values->r_wrap_mode, 0, 2);

   cl[ 3] = __gen_uint(values->per_pixel_mask_enable, 7, 7) |
            __gen_sint(values->texel_offset_for_r_coordinate, 3, 6) |
            __gen_sint(values->texel_offset_for_t_coordinate, 7, 10) >> 8;

}

#define V3D33_TEXTURE_UNIFORM_PARAMETER_0_CFG_MODE1_length      4
#ifdef __gen_unpack_address
static inline void
V3D33_TEXTURE_UNIFORM_PARAMETER_0_CFG_MODE1_unpack(const uint8_t * restrict cl,
                                                   struct V3D33_TEXTURE_UNIFORM_PARAMETER_0_CFG_MODE1 * restrict values)
{
   values->per_pixel_mask_enable = __gen_unpack_uint(cl, 31, 31);
   values->texel_offset_for_r_coordinate = __gen_unpack_sint(cl, 27, 30);
   values->texel_offset_for_t_coordinate = __gen_unpack_sint(cl, 23, 26);
   values->texel_offset_for_s_coordinate = __gen_unpack_sint(cl, 19, 22);
   values->r_wrap_mode = __gen_unpack_uint(cl, 16, 18);
   values->t_wrap_mode = __gen_unpack_uint(cl, 13, 15);
   values->s_wrap_mode = __gen_unpack_uint(cl, 10, 12);
   values->new_configuration_mode = __gen_unpack_uint(cl, 9, 9);
   values->shadow = __gen_unpack_uint(cl, 8, 8);
   values->coefficient_lookup_mode = __gen_unpack_uint(cl, 7, 7);
   values->disable_autolod_use_bias_only = __gen_unpack_uint(cl, 6, 6);
   values->bias_supplied = __gen_unpack_uint(cl, 5, 5);
   values->gather_sample_mode = __gen_unpack_uint(cl, 4, 4);
   values->fetch_sample_mode = __gen_unpack_uint(cl, 3, 3);
   values->lookup_type = __gen_unpack_uint(cl, 0, 2);
}
#endif


#define V3D33_TEXTURE_UNIFORM_PARAMETER_1_CFG_MODE1_header\


struct V3D33_TEXTURE_UNIFORM_PARAMETER_1_CFG_MODE1 {
   __gen_address_type                   texture_state_record_base_address;
   uint32_t                             return_words_of_texture_data;
};

static inline void
V3D33_TEXTURE_UNIFORM_PARAMETER_1_CFG_MODE1_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                 const struct V3D33_TEXTURE_UNIFORM_PARAMETER_1_CFG_MODE1 * restrict values)
{
   __gen_emit_reloc(data, &values->texture_state_record_base_address);
   cl[ 0] = __gen_address_offset(&values->texture_state_record_base_address) |
            __gen_uint(values->return_words_of_texture_data, 0, 3);

   cl[ 1] = __gen_address_offset(&values->texture_state_record_base_address) >> 8;

   cl[ 2] = __gen_address_offset(&values->texture_state_record_base_address) >> 16;

   cl[ 3] = __gen_address_offset(&values->texture_state_record_base_address) >> 24;

}

#define V3D33_TEXTURE_UNIFORM_PARAMETER_1_CFG_MODE1_length      4
#ifdef __gen_unpack_address
static inline void
V3D33_TEXTURE_UNIFORM_PARAMETER_1_CFG_MODE1_unpack(const uint8_t * restrict cl,
                                                   struct V3D33_TEXTURE_UNIFORM_PARAMETER_1_CFG_MODE1 * restrict values)
{
   values->texture_state_record_base_address = __gen_unpack_address(cl, 4, 31);
   values->return_words_of_texture_data = __gen_unpack_uint(cl, 0, 3);
}
#endif


#define V3D33_TEXTURE_SHADER_STATE_header       \
   .flip_etc_y                          =      1

struct V3D33_TEXTURE_SHADER_STATE {
   bool                                 uif_xor_disable;
   bool                                 level_0_is_strictly_uif;
   bool                                 level_0_xor_enable;
   uint32_t                             level_0_ub_pad;
   bool                                 output_32_bit;
   uint32_t                             sample_number;
   uint32_t                             base_level;
   float                                fixed_bias;
   float                                max_level_of_detail;
   float                                min_level_of_detail;
   uint32_t                             border_color_alpha;
   uint32_t                             border_color_blue;
   uint32_t                             border_color_green;
   uint32_t                             border_color_red;
   bool                                 flip_s_and_t_on_incoming_request;
   bool                                 flip_etc_y;
   bool                                 flip_texture_y_axis;
   bool                                 flip_texture_x_axis;
   uint32_t                             swizzle_a;
#define SWIZZLE_ZERO                             0
#define SWIZZLE_ONE                              1
#define SWIZZLE_RED                              2
#define SWIZZLE_GREEN                            3
#define SWIZZLE_BLUE                             4
#define SWIZZLE_ALPHA                            5
   uint32_t                             swizzle_b;
   uint32_t                             swizzle_g;
   uint32_t                             swizzle_r;
   enum V3D33_Compare_Function          depth_compare_function;
   bool                                 srgb;
   uint32_t                             texture_type;
   uint32_t                             image_depth;
   uint32_t                             image_height;
   uint32_t                             image_width;
   uint32_t                             array_stride_64_byte_aligned;
   __gen_address_type                   texture_base_pointer;
   enum V3D33_TMU_Filter                filter;
};

static inline void
V3D33_TEXTURE_SHADER_STATE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                const struct V3D33_TEXTURE_SHADER_STATE * restrict values)
{
   __gen_emit_reloc(data, &values->texture_base_pointer);
   cl[ 0] = __gen_address_offset(&values->texture_base_pointer) |
            __gen_uint(values->filter, 0, 3);

   cl[ 1] = __gen_address_offset(&values->texture_base_pointer) >> 8;

   cl[ 2] = __gen_address_offset(&values->texture_base_pointer) >> 16;

   cl[ 3] = __gen_address_offset(&values->texture_base_pointer) >> 24;

   cl[ 4] = __gen_uint(values->array_stride_64_byte_aligned, 0, 25);

   cl[ 5] = __gen_uint(values->array_stride_64_byte_aligned, 0, 25) >> 8;

   cl[ 6] = __gen_uint(values->array_stride_64_byte_aligned, 0, 25) >> 16;

   cl[ 7] = __gen_uint(values->image_width, 2, 15) |
            __gen_uint(values->array_stride_64_byte_aligned, 0, 25) >> 24;

   cl[ 8] = __gen_uint(values->image_width, 2, 15) >> 8;

   cl[ 9] = __gen_uint(values->image_height, 0, 13);

   cl[10] = __gen_uint(values->image_depth, 6, 19) |
            __gen_uint(values->image_height, 0, 13) >> 8;

   cl[11] = __gen_uint(values->image_depth, 6, 19) >> 8;

   cl[12] = __gen_uint(values->texture_type, 4, 10) |
            __gen_uint(values->image_depth, 6, 19) >> 16;

   cl[13] = __gen_uint(values->depth_compare_function, 5, 7) |
            __gen_uint(values->srgb, 3, 3) |
            __gen_uint(values->texture_type, 4, 10) >> 8;

   cl[14] = __gen_uint(values->swizzle_b, 6, 8) |
            __gen_uint(values->swizzle_g, 3, 5) |
            __gen_uint(values->swizzle_r, 0, 2);

   cl[15] = __gen_uint(values->flip_s_and_t_on_incoming_request, 7, 7) |
            __gen_uint(values->flip_etc_y, 6, 6) |
            __gen_uint(values->flip_texture_y_axis, 5, 5) |
            __gen_uint(values->flip_texture_x_axis, 4, 4) |
            __gen_uint(values->swizzle_a, 1, 3) |
            __gen_uint(values->swizzle_b, 6, 8) >> 8;

   cl[16] = __gen_uint(values->border_color_red, 0, 15);

   cl[17] = __gen_uint(values->border_color_red, 0, 15) >> 8;

   cl[18] = __gen_uint(values->border_color_green, 0, 15);

   cl[19] = __gen_uint(values->border_color_green, 0, 15) >> 8;

   cl[20] = __gen_uint(values->border_color_blue, 0, 15);

   cl[21] = __gen_uint(values->border_color_blue, 0, 15) >> 8;

   cl[22] = __gen_uint(values->border_color_alpha, 0, 15);

   cl[23] = __gen_uint(values->border_color_alpha, 0, 15) >> 8;

   cl[24] = __gen_sfixed(values->min_level_of_detail, 0, 15, 8);

   cl[25] = __gen_sfixed(values->min_level_of_detail, 0, 15, 8) >> 8;

   cl[26] = __gen_sfixed(values->max_level_of_detail, 0, 15, 8);

   cl[27] = __gen_sfixed(values->max_level_of_detail, 0, 15, 8) >> 8;

   cl[28] = __gen_sfixed(values->fixed_bias, 0, 15, 8);

   cl[29] = __gen_sfixed(values->fixed_bias, 0, 15, 8) >> 8;

   cl[30] = __gen_uint(values->output_32_bit, 6, 6) |
            __gen_uint(values->sample_number, 4, 5) |
            __gen_uint(values->base_level, 0, 3);

   cl[31] = __gen_uint(values->uif_xor_disable, 7, 7) |
            __gen_uint(values->level_0_is_strictly_uif, 6, 6) |
            __gen_uint(values->level_0_xor_enable, 4, 4) |
            __gen_uint(values->level_0_ub_pad, 0, 3);

}

#define V3D33_TEXTURE_SHADER_STATE_length     32
#ifdef __gen_unpack_address
static inline void
V3D33_TEXTURE_SHADER_STATE_unpack(const uint8_t * restrict cl,
                                  struct V3D33_TEXTURE_SHADER_STATE * restrict values)
{
   values->uif_xor_disable = __gen_unpack_uint(cl, 255, 255);
   values->level_0_is_strictly_uif = __gen_unpack_uint(cl, 254, 254);
   values->level_0_xor_enable = __gen_unpack_uint(cl, 252, 252);
   values->level_0_ub_pad = __gen_unpack_uint(cl, 248, 251);
   values->output_32_bit = __gen_unpack_uint(cl, 246, 246);
   values->sample_number = __gen_unpack_uint(cl, 244, 245);
   values->base_level = __gen_unpack_uint(cl, 240, 243);
   values->fixed_bias = __gen_unpack_sfixed(cl, 224, 239, 8);
   values->max_level_of_detail = __gen_unpack_sfixed(cl, 208, 223, 8);
   values->min_level_of_detail = __gen_unpack_sfixed(cl, 192, 207, 8);
   values->border_color_alpha = __gen_unpack_uint(cl, 176, 191);
   values->border_color_blue = __gen_unpack_uint(cl, 160, 175);
   values->border_color_green = __gen_unpack_uint(cl, 144, 159);
   values->border_color_red = __gen_unpack_uint(cl, 128, 143);
   values->flip_s_and_t_on_incoming_request = __gen_unpack_uint(cl, 127, 127);
   values->flip_etc_y = __gen_unpack_uint(cl, 126, 126);
   values->flip_texture_y_axis = __gen_unpack_uint(cl, 125, 125);
   values->flip_texture_x_axis = __gen_unpack_uint(cl, 124, 124);
   values->swizzle_a = __gen_unpack_uint(cl, 121, 123);
   values->swizzle_b = __gen_unpack_uint(cl, 118, 120);
   values->swizzle_g = __gen_unpack_uint(cl, 115, 117);
   values->swizzle_r = __gen_unpack_uint(cl, 112, 114);
   values->depth_compare_function = __gen_unpack_uint(cl, 109, 111);
   values->srgb = __gen_unpack_uint(cl, 107, 107);
   values->texture_type = __gen_unpack_uint(cl, 100, 106);
   values->image_depth = __gen_unpack_uint(cl, 86, 99);
   values->image_height = __gen_unpack_uint(cl, 72, 85);
   values->image_width = __gen_unpack_uint(cl, 58, 71);
   values->array_stride_64_byte_aligned = __gen_unpack_uint(cl, 32, 57);
   values->texture_base_pointer = __gen_unpack_address(cl, 2, 31);
   values->filter = __gen_unpack_uint(cl, 0, 3);
}
#endif


enum V3D33_Texture_Data_Formats {
        TEXTURE_DATA_FORMAT_R8               =      0,
        TEXTURE_DATA_FORMAT_R8_SNORM         =      1,
        TEXTURE_DATA_FORMAT_RG8              =      2,
        TEXTURE_DATA_FORMAT_RG8_SNORM        =      3,
        TEXTURE_DATA_FORMAT_RGBA8            =      4,
        TEXTURE_DATA_FORMAT_RGBA8_SNORM      =      5,
        TEXTURE_DATA_FORMAT_RGB565           =      6,
        TEXTURE_DATA_FORMAT_RGBA4            =      7,
        TEXTURE_DATA_FORMAT_RGB5_A1          =      8,
        TEXTURE_DATA_FORMAT_RGB10_A2         =      9,
        TEXTURE_DATA_FORMAT_R16              =     10,
        TEXTURE_DATA_FORMAT_R16_SNORM        =     11,
        TEXTURE_DATA_FORMAT_RG16             =     12,
        TEXTURE_DATA_FORMAT_RG16_SNORM       =     13,
        TEXTURE_DATA_FORMAT_RGBA16           =     14,
        TEXTURE_DATA_FORMAT_RGBA16_SNORM     =     15,
        TEXTURE_DATA_FORMAT_R16F             =     16,
        TEXTURE_DATA_FORMAT_RG16F            =     17,
        TEXTURE_DATA_FORMAT_RGBA16F          =     18,
        TEXTURE_DATA_FORMAT_R11F_G11F_B10F   =     19,
        TEXTURE_DATA_FORMAT_RGB9_E5          =     20,
        TEXTURE_DATA_FORMAT_DEPTH_COMP16     =     21,
        TEXTURE_DATA_FORMAT_DEPTH_COMP24     =     22,
        TEXTURE_DATA_FORMAT_DEPTH_COMP32F    =     23,
        TEXTURE_DATA_FORMAT_DEPTH24_X8       =     24,
        TEXTURE_DATA_FORMAT_R4               =     25,
        TEXTURE_DATA_FORMAT_R1               =     26,
        TEXTURE_DATA_FORMAT_S8               =     27,
        TEXTURE_DATA_FORMAT_S16              =     28,
        TEXTURE_DATA_FORMAT_R32F             =     29,
        TEXTURE_DATA_FORMAT_RG32F            =     30,
        TEXTURE_DATA_FORMAT_RGBA32F          =     31,
        TEXTURE_DATA_FORMAT_RGB8_ETC2        =     32,
        TEXTURE_DATA_FORMAT_RGB8_PUNCHTHROUGH_ALPHA1 =     33,
        TEXTURE_DATA_FORMAT_R11_EAC          =     34,
        TEXTURE_DATA_FORMAT_SIGNED_R11_EAC   =     35,
        TEXTURE_DATA_FORMAT_RG11_EAC         =     36,
        TEXTURE_DATA_FORMAT_SIGNED_RG11_EAC  =     37,
        TEXTURE_DATA_FORMAT_RGBA8_ETC2_EAC   =     38,
        TEXTURE_DATA_FORMAT_YCBCR_LUMA       =     39,
        TEXTURE_DATA_FORMAT_YCBCR_420_CHROMA =     40,
        TEXTURE_DATA_FORMAT_BC1              =     48,
        TEXTURE_DATA_FORMAT_BC2              =     49,
        TEXTURE_DATA_FORMAT_BC3              =     50,
        TEXTURE_DATA_FORMAT_ASTC_4X4         =     64,
        TEXTURE_DATA_FORMAT_ASTC_5X4         =     65,
        TEXTURE_DATA_FORMAT_ASTC_5X5         =     66,
        TEXTURE_DATA_FORMAT_ASTC_6X5         =     67,
        TEXTURE_DATA_FORMAT_ASTC_6X6         =     68,
        TEXTURE_DATA_FORMAT_ASTC_8X5         =     69,
        TEXTURE_DATA_FORMAT_ASTC_8X6         =     70,
        TEXTURE_DATA_FORMAT_ASTC_8X8         =     71,
        TEXTURE_DATA_FORMAT_ASTC_10X5        =     72,
        TEXTURE_DATA_FORMAT_ASTC_10X6        =     73,
        TEXTURE_DATA_FORMAT_ASTC_10X8        =     74,
        TEXTURE_DATA_FORMAT_ASTC_10X10       =     75,
        TEXTURE_DATA_FORMAT_ASTC_12X10       =     76,
        TEXTURE_DATA_FORMAT_ASTC_12X12       =     77,
        TEXTURE_DATA_FORMAT_R8I              =     96,
        TEXTURE_DATA_FORMAT_R8UI             =     97,
        TEXTURE_DATA_FORMAT_RG8I             =     98,
        TEXTURE_DATA_FORMAT_RG8UI            =     99,
        TEXTURE_DATA_FORMAT_RGBA8I           =    100,
        TEXTURE_DATA_FORMAT_RGBA8UI          =    101,
        TEXTURE_DATA_FORMAT_R16I             =    102,
        TEXTURE_DATA_FORMAT_R16UI            =    103,
        TEXTURE_DATA_FORMAT_RG16I            =    104,
        TEXTURE_DATA_FORMAT_RG16UI           =    105,
        TEXTURE_DATA_FORMAT_RGBA16I          =    106,
        TEXTURE_DATA_FORMAT_RGBA16UI         =    107,
        TEXTURE_DATA_FORMAT_R32I             =    108,
        TEXTURE_DATA_FORMAT_R32UI            =    109,
        TEXTURE_DATA_FORMAT_RG32I            =    110,
        TEXTURE_DATA_FORMAT_RG32UI           =    111,
        TEXTURE_DATA_FORMAT_RGBA32I          =    112,
        TEXTURE_DATA_FORMAT_RGBA32UI         =    113,
        TEXTURE_DATA_FORMAT_RGB10_A2UI       =    114,
};

#endif /* V3D33_PACK_H */
