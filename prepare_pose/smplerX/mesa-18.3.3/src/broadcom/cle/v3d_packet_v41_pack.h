/* Generated code, see packets.xml and gen_packet_header.py */


/* Packets, enums and structures for V3D 4.1.
 *
 * This file has been generated, do not hand edit.
 */

#ifndef V3D41_PACK_H
#define V3D41_PACK_H

#include "cle/v3d_packet_helpers.h"


enum V3D41_Compare_Function {
        V3D_COMPARE_FUNC_NEVER               =      0,
        V3D_COMPARE_FUNC_LESS                =      1,
        V3D_COMPARE_FUNC_EQUAL               =      2,
        V3D_COMPARE_FUNC_LEQUAL              =      3,
        V3D_COMPARE_FUNC_GREATER             =      4,
        V3D_COMPARE_FUNC_NOTEQUAL            =      5,
        V3D_COMPARE_FUNC_GEQUAL              =      6,
        V3D_COMPARE_FUNC_ALWAYS              =      7,
};

enum V3D41_Blend_Factor {
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

enum V3D41_Blend_Mode {
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

enum V3D41_Stencil_Op {
        V3D_STENCIL_OP_ZERO                  =      0,
        V3D_STENCIL_OP_KEEP                  =      1,
        V3D_STENCIL_OP_REPLACE               =      2,
        V3D_STENCIL_OP_INCR                  =      3,
        V3D_STENCIL_OP_DECR                  =      4,
        V3D_STENCIL_OP_INVERT                =      5,
        V3D_STENCIL_OP_INCWRAP               =      6,
        V3D_STENCIL_OP_DECWRAP               =      7,
};

enum V3D41_Primitive {
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

enum V3D41_Border_Color_Mode {
        V3D_BORDER_COLOR_0000                =      0,
        V3D_BORDER_COLOR_0001                =      1,
        V3D_BORDER_COLOR_1111                =      2,
        V3D_BORDER_COLOR_FOLLOWS             =      7,
};

enum V3D41_Wrap_Mode {
        V3D_WRAP_MODE_WRAP_MODE_REPEAT       =      0,
        V3D_WRAP_MODE_WRAP_MODE_CLAMP        =      1,
        V3D_WRAP_MODE_WRAP_MODE_MIRROR       =      2,
        V3D_WRAP_MODE_WRAP_MODE_BORDER       =      3,
        V3D_WRAP_MODE_WRAP_MODE_MIRROR_ONCE  =      4,
};

enum V3D41_TMU_Op {
        V3D_TMU_OP_WRITE_ADD_READ_PREFETCH   =      0,
        V3D_TMU_OP_WRITE_SUB_READ_CLEAR      =      1,
        V3D_TMU_OP_WRITE_XCHG_READ_FLUSH     =      2,
        V3D_TMU_OP_WRITE_CMPXCHG_READ_FLUSH  =      3,
        V3D_TMU_OP_WRITE_UMIN_FULL_L1_CLEAR  =      4,
        V3D_TMU_OP_WRITE_UMAX                =      5,
        V3D_TMU_OP_WRITE_SMIN                =      6,
        V3D_TMU_OP_WRITE_SMAX                =      7,
        V3D_TMU_OP_WRITE_AND_READ_INC        =      8,
        V3D_TMU_OP_WRITE_OR_READ_DEC         =      9,
        V3D_TMU_OP_WRITE_XOR_READ_NOT        =     10,
        V3D_TMU_OP_REGULAR                   =     15,
};

enum V3D41_Varying_Flags_Action {
        V3D_VARYING_FLAGS_ACTION_UNCHANGED   =      0,
        V3D_VARYING_FLAGS_ACTION_ZEROED      =      1,
        V3D_VARYING_FLAGS_ACTION_SET         =      2,
};

enum V3D41_Memory_Format {
        V3D_MEMORY_FORMAT_RASTER             =      0,
        V3D_MEMORY_FORMAT_LINEARTILE         =      1,
        V3D_MEMORY_FORMAT_UB_LINEAR_1_UIF_BLOCK_WIDE =      2,
        V3D_MEMORY_FORMAT_UB_LINEAR_2_UIF_BLOCKS_WIDE =      3,
        V3D_MEMORY_FORMAT_UIF_NO_XOR         =      4,
        V3D_MEMORY_FORMAT_UIF_XOR            =      5,
};

enum V3D41_Decimate_Mode {
        V3D_DECIMATE_MODE_SAMPLE_0           =      0,
        V3D_DECIMATE_MODE_4X                 =      1,
        V3D_DECIMATE_MODE_ALL_SAMPLES        =      3,
};

enum V3D41_Internal_Type {
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

enum V3D41_Internal_BPP {
        V3D_INTERNAL_BPP_32                  =      0,
        V3D_INTERNAL_BPP_64                  =      1,
        V3D_INTERNAL_BPP_128                 =      2,
};

enum V3D41_Internal_Depth_Type {
        V3D_INTERNAL_TYPE_DEPTH_32F          =      0,
        V3D_INTERNAL_TYPE_DEPTH_24           =      1,
        V3D_INTERNAL_TYPE_DEPTH_16           =      2,
};

enum V3D41_Render_Target_Clamp {
        V3D_RENDER_TARGET_CLAMP_NONE         =      0,
        V3D_RENDER_TARGET_CLAMP_NORM         =      1,
        V3D_RENDER_TARGET_CLAMP_POS          =      2,
};

enum V3D41_Output_Image_Format {
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
        V3D_OUTPUT_IMAGE_FORMAT_BSTC         =     39,
        V3D_OUTPUT_IMAGE_FORMAT_D32F         =     40,
        V3D_OUTPUT_IMAGE_FORMAT_D24          =     41,
        V3D_OUTPUT_IMAGE_FORMAT_D16          =     42,
        V3D_OUTPUT_IMAGE_FORMAT_D24S8        =     43,
        V3D_OUTPUT_IMAGE_FORMAT_S8           =     44,
};

enum V3D41_Dither_Mode {
        V3D_DITHER_MODE_NONE                 =      0,
        V3D_DITHER_MODE_RGB                  =      1,
        V3D_DITHER_MODE_A                    =      2,
        V3D_DITHER_MODE_RGBA                 =      3,
};

#define V3D41_HALT_opcode                      0
#define V3D41_HALT_header                       \
   .opcode                              =      0

struct V3D41_HALT {
   uint32_t                             opcode;
};

static inline void
V3D41_HALT_pack(__gen_user_data *data, uint8_t * restrict cl,
                const struct V3D41_HALT * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_HALT_length                      1
#ifdef __gen_unpack_address
static inline void
V3D41_HALT_unpack(const uint8_t * restrict cl,
                  struct V3D41_HALT * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_NOP_opcode                       1
#define V3D41_NOP_header                        \
   .opcode                              =      1

struct V3D41_NOP {
   uint32_t                             opcode;
};

static inline void
V3D41_NOP_pack(__gen_user_data *data, uint8_t * restrict cl,
               const struct V3D41_NOP * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_NOP_length                       1
#ifdef __gen_unpack_address
static inline void
V3D41_NOP_unpack(const uint8_t * restrict cl,
                 struct V3D41_NOP * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_FLUSH_opcode                     4
#define V3D41_FLUSH_header                      \
   .opcode                              =      4

struct V3D41_FLUSH {
   uint32_t                             opcode;
};

static inline void
V3D41_FLUSH_pack(__gen_user_data *data, uint8_t * restrict cl,
                 const struct V3D41_FLUSH * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_FLUSH_length                     1
#ifdef __gen_unpack_address
static inline void
V3D41_FLUSH_unpack(const uint8_t * restrict cl,
                   struct V3D41_FLUSH * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_FLUSH_ALL_STATE_opcode           5
#define V3D41_FLUSH_ALL_STATE_header            \
   .opcode                              =      5

struct V3D41_FLUSH_ALL_STATE {
   uint32_t                             opcode;
};

static inline void
V3D41_FLUSH_ALL_STATE_pack(__gen_user_data *data, uint8_t * restrict cl,
                           const struct V3D41_FLUSH_ALL_STATE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_FLUSH_ALL_STATE_length           1
#ifdef __gen_unpack_address
static inline void
V3D41_FLUSH_ALL_STATE_unpack(const uint8_t * restrict cl,
                             struct V3D41_FLUSH_ALL_STATE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_START_TILE_BINNING_opcode        6
#define V3D41_START_TILE_BINNING_header         \
   .opcode                              =      6

struct V3D41_START_TILE_BINNING {
   uint32_t                             opcode;
};

static inline void
V3D41_START_TILE_BINNING_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D41_START_TILE_BINNING * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_START_TILE_BINNING_length        1
#ifdef __gen_unpack_address
static inline void
V3D41_START_TILE_BINNING_unpack(const uint8_t * restrict cl,
                                struct V3D41_START_TILE_BINNING * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_INCREMENT_SEMAPHORE_opcode       7
#define V3D41_INCREMENT_SEMAPHORE_header        \
   .opcode                              =      7

struct V3D41_INCREMENT_SEMAPHORE {
   uint32_t                             opcode;
};

static inline void
V3D41_INCREMENT_SEMAPHORE_pack(__gen_user_data *data, uint8_t * restrict cl,
                               const struct V3D41_INCREMENT_SEMAPHORE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_INCREMENT_SEMAPHORE_length       1
#ifdef __gen_unpack_address
static inline void
V3D41_INCREMENT_SEMAPHORE_unpack(const uint8_t * restrict cl,
                                 struct V3D41_INCREMENT_SEMAPHORE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_WAIT_ON_SEMAPHORE_opcode         8
#define V3D41_WAIT_ON_SEMAPHORE_header          \
   .opcode                              =      8

struct V3D41_WAIT_ON_SEMAPHORE {
   uint32_t                             opcode;
};

static inline void
V3D41_WAIT_ON_SEMAPHORE_pack(__gen_user_data *data, uint8_t * restrict cl,
                             const struct V3D41_WAIT_ON_SEMAPHORE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_WAIT_ON_SEMAPHORE_length         1
#ifdef __gen_unpack_address
static inline void
V3D41_WAIT_ON_SEMAPHORE_unpack(const uint8_t * restrict cl,
                               struct V3D41_WAIT_ON_SEMAPHORE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_WAIT_FOR_PREVIOUS_FRAME_opcode      9
#define V3D41_WAIT_FOR_PREVIOUS_FRAME_header    \
   .opcode                              =      9

struct V3D41_WAIT_FOR_PREVIOUS_FRAME {
   uint32_t                             opcode;
};

static inline void
V3D41_WAIT_FOR_PREVIOUS_FRAME_pack(__gen_user_data *data, uint8_t * restrict cl,
                                   const struct V3D41_WAIT_FOR_PREVIOUS_FRAME * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_WAIT_FOR_PREVIOUS_FRAME_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_WAIT_FOR_PREVIOUS_FRAME_unpack(const uint8_t * restrict cl,
                                     struct V3D41_WAIT_FOR_PREVIOUS_FRAME * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_ENABLE_Z_ONLY_RENDERING_opcode     10
#define V3D41_ENABLE_Z_ONLY_RENDERING_header    \
   .opcode                              =     10

struct V3D41_ENABLE_Z_ONLY_RENDERING {
   uint32_t                             opcode;
};

static inline void
V3D41_ENABLE_Z_ONLY_RENDERING_pack(__gen_user_data *data, uint8_t * restrict cl,
                                   const struct V3D41_ENABLE_Z_ONLY_RENDERING * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_ENABLE_Z_ONLY_RENDERING_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_ENABLE_Z_ONLY_RENDERING_unpack(const uint8_t * restrict cl,
                                     struct V3D41_ENABLE_Z_ONLY_RENDERING * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_DISABLE_Z_ONLY_RENDERING_opcode     11
#define V3D41_DISABLE_Z_ONLY_RENDERING_header   \
   .opcode                              =     11

struct V3D41_DISABLE_Z_ONLY_RENDERING {
   uint32_t                             opcode;
};

static inline void
V3D41_DISABLE_Z_ONLY_RENDERING_pack(__gen_user_data *data, uint8_t * restrict cl,
                                    const struct V3D41_DISABLE_Z_ONLY_RENDERING * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_DISABLE_Z_ONLY_RENDERING_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_DISABLE_Z_ONLY_RENDERING_unpack(const uint8_t * restrict cl,
                                      struct V3D41_DISABLE_Z_ONLY_RENDERING * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_END_OF_Z_ONLY_RENDERING_IN_FRAME_opcode     12
#define V3D41_END_OF_Z_ONLY_RENDERING_IN_FRAME_header\
   .opcode                              =     12

struct V3D41_END_OF_Z_ONLY_RENDERING_IN_FRAME {
   uint32_t                             opcode;
};

static inline void
V3D41_END_OF_Z_ONLY_RENDERING_IN_FRAME_pack(__gen_user_data *data, uint8_t * restrict cl,
                                            const struct V3D41_END_OF_Z_ONLY_RENDERING_IN_FRAME * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_END_OF_Z_ONLY_RENDERING_IN_FRAME_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_END_OF_Z_ONLY_RENDERING_IN_FRAME_unpack(const uint8_t * restrict cl,
                                              struct V3D41_END_OF_Z_ONLY_RENDERING_IN_FRAME * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_END_OF_RENDERING_opcode         13
#define V3D41_END_OF_RENDERING_header           \
   .opcode                              =     13

struct V3D41_END_OF_RENDERING {
   uint32_t                             opcode;
};

static inline void
V3D41_END_OF_RENDERING_pack(__gen_user_data *data, uint8_t * restrict cl,
                            const struct V3D41_END_OF_RENDERING * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_END_OF_RENDERING_length          1
#ifdef __gen_unpack_address
static inline void
V3D41_END_OF_RENDERING_unpack(const uint8_t * restrict cl,
                              struct V3D41_END_OF_RENDERING * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_WAIT_FOR_TRANSFORM_FEEDBACK_opcode     14
#define V3D41_WAIT_FOR_TRANSFORM_FEEDBACK_header\
   .opcode                              =     14

struct V3D41_WAIT_FOR_TRANSFORM_FEEDBACK {
   uint32_t                             opcode;
   uint32_t                             block_count;
};

static inline void
V3D41_WAIT_FOR_TRANSFORM_FEEDBACK_pack(__gen_user_data *data, uint8_t * restrict cl,
                                       const struct V3D41_WAIT_FOR_TRANSFORM_FEEDBACK * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->block_count, 0, 7);

}

#define V3D41_WAIT_FOR_TRANSFORM_FEEDBACK_length      2
#ifdef __gen_unpack_address
static inline void
V3D41_WAIT_FOR_TRANSFORM_FEEDBACK_unpack(const uint8_t * restrict cl,
                                         struct V3D41_WAIT_FOR_TRANSFORM_FEEDBACK * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->block_count = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D41_BRANCH_TO_AUTO_CHAINED_SUB_LIST_opcode     15
#define V3D41_BRANCH_TO_AUTO_CHAINED_SUB_LIST_header\
   .opcode                              =     15

struct V3D41_BRANCH_TO_AUTO_CHAINED_SUB_LIST {
   uint32_t                             opcode;
   __gen_address_type                   address;
};

static inline void
V3D41_BRANCH_TO_AUTO_CHAINED_SUB_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                                           const struct V3D41_BRANCH_TO_AUTO_CHAINED_SUB_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D41_BRANCH_TO_AUTO_CHAINED_SUB_LIST_length      5
#ifdef __gen_unpack_address
static inline void
V3D41_BRANCH_TO_AUTO_CHAINED_SUB_LIST_unpack(const uint8_t * restrict cl,
                                             struct V3D41_BRANCH_TO_AUTO_CHAINED_SUB_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 8, 39);
}
#endif


#define V3D41_BRANCH_opcode                   16
#define V3D41_BRANCH_header                     \
   .opcode                              =     16

struct V3D41_BRANCH {
   uint32_t                             opcode;
   __gen_address_type                   address;
};

static inline void
V3D41_BRANCH_pack(__gen_user_data *data, uint8_t * restrict cl,
                  const struct V3D41_BRANCH * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D41_BRANCH_length                    5
#ifdef __gen_unpack_address
static inline void
V3D41_BRANCH_unpack(const uint8_t * restrict cl,
                    struct V3D41_BRANCH * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 8, 39);
}
#endif


#define V3D41_BRANCH_TO_SUB_LIST_opcode       17
#define V3D41_BRANCH_TO_SUB_LIST_header         \
   .opcode                              =     17

struct V3D41_BRANCH_TO_SUB_LIST {
   uint32_t                             opcode;
   __gen_address_type                   address;
};

static inline void
V3D41_BRANCH_TO_SUB_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D41_BRANCH_TO_SUB_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D41_BRANCH_TO_SUB_LIST_length        5
#ifdef __gen_unpack_address
static inline void
V3D41_BRANCH_TO_SUB_LIST_unpack(const uint8_t * restrict cl,
                                struct V3D41_BRANCH_TO_SUB_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 8, 39);
}
#endif


#define V3D41_RETURN_FROM_SUB_LIST_opcode     18
#define V3D41_RETURN_FROM_SUB_LIST_header       \
   .opcode                              =     18

struct V3D41_RETURN_FROM_SUB_LIST {
   uint32_t                             opcode;
};

static inline void
V3D41_RETURN_FROM_SUB_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                                const struct V3D41_RETURN_FROM_SUB_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_RETURN_FROM_SUB_LIST_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_RETURN_FROM_SUB_LIST_unpack(const uint8_t * restrict cl,
                                  struct V3D41_RETURN_FROM_SUB_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_FLUSH_VCD_CACHE_opcode          19
#define V3D41_FLUSH_VCD_CACHE_header            \
   .opcode                              =     19

struct V3D41_FLUSH_VCD_CACHE {
   uint32_t                             opcode;
};

static inline void
V3D41_FLUSH_VCD_CACHE_pack(__gen_user_data *data, uint8_t * restrict cl,
                           const struct V3D41_FLUSH_VCD_CACHE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_FLUSH_VCD_CACHE_length           1
#ifdef __gen_unpack_address
static inline void
V3D41_FLUSH_VCD_CACHE_unpack(const uint8_t * restrict cl,
                             struct V3D41_FLUSH_VCD_CACHE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_START_ADDRESS_OF_GENERIC_TILE_LIST_opcode     20
#define V3D41_START_ADDRESS_OF_GENERIC_TILE_LIST_header\
   .opcode                              =     20

struct V3D41_START_ADDRESS_OF_GENERIC_TILE_LIST {
   uint32_t                             opcode;
   __gen_address_type                   start;
   __gen_address_type                   end;
};

static inline void
V3D41_START_ADDRESS_OF_GENERIC_TILE_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                                              const struct V3D41_START_ADDRESS_OF_GENERIC_TILE_LIST * restrict values)
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

#define V3D41_START_ADDRESS_OF_GENERIC_TILE_LIST_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_START_ADDRESS_OF_GENERIC_TILE_LIST_unpack(const uint8_t * restrict cl,
                                                struct V3D41_START_ADDRESS_OF_GENERIC_TILE_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->start = __gen_unpack_address(cl, 8, 39);
   values->end = __gen_unpack_address(cl, 40, 71);
}
#endif


#define V3D41_BRANCH_TO_IMPLICIT_TILE_LIST_opcode     21
#define V3D41_BRANCH_TO_IMPLICIT_TILE_LIST_header\
   .opcode                              =     21

struct V3D41_BRANCH_TO_IMPLICIT_TILE_LIST {
   uint32_t                             opcode;
   uint32_t                             tile_list_set_number;
};

static inline void
V3D41_BRANCH_TO_IMPLICIT_TILE_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D41_BRANCH_TO_IMPLICIT_TILE_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->tile_list_set_number, 0, 7);

}

#define V3D41_BRANCH_TO_IMPLICIT_TILE_LIST_length      2
#ifdef __gen_unpack_address
static inline void
V3D41_BRANCH_TO_IMPLICIT_TILE_LIST_unpack(const uint8_t * restrict cl,
                                          struct V3D41_BRANCH_TO_IMPLICIT_TILE_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->tile_list_set_number = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D41_BRANCH_TO_EXPLICIT_SUPERTILE_opcode     22
#define V3D41_BRANCH_TO_EXPLICIT_SUPERTILE_header\
   .opcode                              =     22

struct V3D41_BRANCH_TO_EXPLICIT_SUPERTILE {
   uint32_t                             opcode;
   __gen_address_type                   absolute_address_of_explicit_supertile_render_list;
   uint32_t                             explicit_supertile_number;
   uint32_t                             row_number;
   uint32_t                             column_number;
};

static inline void
V3D41_BRANCH_TO_EXPLICIT_SUPERTILE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D41_BRANCH_TO_EXPLICIT_SUPERTILE * restrict values)
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

#define V3D41_BRANCH_TO_EXPLICIT_SUPERTILE_length      8
#ifdef __gen_unpack_address
static inline void
V3D41_BRANCH_TO_EXPLICIT_SUPERTILE_unpack(const uint8_t * restrict cl,
                                          struct V3D41_BRANCH_TO_EXPLICIT_SUPERTILE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->absolute_address_of_explicit_supertile_render_list = __gen_unpack_address(cl, 32, 63);
   values->explicit_supertile_number = __gen_unpack_uint(cl, 24, 31);
   values->row_number = __gen_unpack_uint(cl, 16, 23);
   values->column_number = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D41_SUPERTILE_COORDINATES_opcode     23
#define V3D41_SUPERTILE_COORDINATES_header      \
   .opcode                              =     23

struct V3D41_SUPERTILE_COORDINATES {
   uint32_t                             opcode;
   uint32_t                             row_number_in_supertiles;
   uint32_t                             column_number_in_supertiles;
};

static inline void
V3D41_SUPERTILE_COORDINATES_pack(__gen_user_data *data, uint8_t * restrict cl,
                                 const struct V3D41_SUPERTILE_COORDINATES * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->column_number_in_supertiles, 0, 7);

   cl[ 2] = __gen_uint(values->row_number_in_supertiles, 0, 7);

}

#define V3D41_SUPERTILE_COORDINATES_length      3
#ifdef __gen_unpack_address
static inline void
V3D41_SUPERTILE_COORDINATES_unpack(const uint8_t * restrict cl,
                                   struct V3D41_SUPERTILE_COORDINATES * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->row_number_in_supertiles = __gen_unpack_uint(cl, 16, 23);
   values->column_number_in_supertiles = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D41_CLEAR_TILE_BUFFERS_opcode       25
#define V3D41_CLEAR_TILE_BUFFERS_header         \
   .opcode                              =     25

struct V3D41_CLEAR_TILE_BUFFERS {
   uint32_t                             opcode;
   bool                                 clear_z_stencil_buffer;
   bool                                 clear_all_render_targets;
};

static inline void
V3D41_CLEAR_TILE_BUFFERS_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D41_CLEAR_TILE_BUFFERS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->clear_z_stencil_buffer, 1, 1) |
            __gen_uint(values->clear_all_render_targets, 0, 0);

}

#define V3D41_CLEAR_TILE_BUFFERS_length        2
#ifdef __gen_unpack_address
static inline void
V3D41_CLEAR_TILE_BUFFERS_unpack(const uint8_t * restrict cl,
                                struct V3D41_CLEAR_TILE_BUFFERS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->clear_z_stencil_buffer = __gen_unpack_uint(cl, 9, 9);
   values->clear_all_render_targets = __gen_unpack_uint(cl, 8, 8);
}
#endif


#define V3D41_END_OF_LOADS_opcode             26
#define V3D41_END_OF_LOADS_header               \
   .opcode                              =     26

struct V3D41_END_OF_LOADS {
   uint32_t                             opcode;
};

static inline void
V3D41_END_OF_LOADS_pack(__gen_user_data *data, uint8_t * restrict cl,
                        const struct V3D41_END_OF_LOADS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_END_OF_LOADS_length              1
#ifdef __gen_unpack_address
static inline void
V3D41_END_OF_LOADS_unpack(const uint8_t * restrict cl,
                          struct V3D41_END_OF_LOADS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_END_OF_TILE_MARKER_opcode       27
#define V3D41_END_OF_TILE_MARKER_header         \
   .opcode                              =     27

struct V3D41_END_OF_TILE_MARKER {
   uint32_t                             opcode;
};

static inline void
V3D41_END_OF_TILE_MARKER_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D41_END_OF_TILE_MARKER * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_END_OF_TILE_MARKER_length        1
#ifdef __gen_unpack_address
static inline void
V3D41_END_OF_TILE_MARKER_unpack(const uint8_t * restrict cl,
                                struct V3D41_END_OF_TILE_MARKER * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_STORE_TILE_BUFFER_GENERAL_opcode     29
#define V3D41_STORE_TILE_BUFFER_GENERAL_header  \
   .opcode                              =     29

struct V3D41_STORE_TILE_BUFFER_GENERAL {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             height;
   uint32_t                             height_in_ub_or_stride;
   bool                                 r_b_swap;
   bool                                 channel_reverse;
   bool                                 clear_buffer_being_stored;
   enum V3D41_Output_Image_Format       output_image_format;
   enum V3D41_Decimate_Mode             decimate_mode;
   enum V3D41_Dither_Mode               dither_mode;
   bool                                 flip_y;
   enum V3D41_Memory_Format             memory_format;
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
V3D41_STORE_TILE_BUFFER_GENERAL_pack(__gen_user_data *data, uint8_t * restrict cl,
                                     const struct V3D41_STORE_TILE_BUFFER_GENERAL * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->flip_y, 7, 7) |
            __gen_uint(values->memory_format, 4, 6) |
            __gen_uint(values->buffer_to_store, 0, 3);

   cl[ 2] = __gen_uint(values->output_image_format, 4, 9) |
            __gen_uint(values->decimate_mode, 2, 3) |
            __gen_uint(values->dither_mode, 0, 1);

   cl[ 3] = __gen_uint(values->r_b_swap, 4, 4) |
            __gen_uint(values->channel_reverse, 3, 3) |
            __gen_uint(values->clear_buffer_being_stored, 2, 2) |
            __gen_uint(values->output_image_format, 4, 9) >> 8;

   cl[ 4] = __gen_uint(values->height_in_ub_or_stride, 4, 23);

   cl[ 5] = __gen_uint(values->height_in_ub_or_stride, 4, 23) >> 8;

   cl[ 6] = __gen_uint(values->height_in_ub_or_stride, 4, 23) >> 16;

   cl[ 7] = __gen_uint(values->height, 0, 15);

   cl[ 8] = __gen_uint(values->height, 0, 15) >> 8;

   __gen_emit_reloc(data, &values->address);
   cl[ 9] = __gen_address_offset(&values->address);

   cl[10] = __gen_address_offset(&values->address) >> 8;

   cl[11] = __gen_address_offset(&values->address) >> 16;

   cl[12] = __gen_address_offset(&values->address) >> 24;

}

#define V3D41_STORE_TILE_BUFFER_GENERAL_length     13
#ifdef __gen_unpack_address
static inline void
V3D41_STORE_TILE_BUFFER_GENERAL_unpack(const uint8_t * restrict cl,
                                       struct V3D41_STORE_TILE_BUFFER_GENERAL * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 72, 103);
   values->height = __gen_unpack_uint(cl, 56, 71);
   values->height_in_ub_or_stride = __gen_unpack_uint(cl, 36, 55);
   values->r_b_swap = __gen_unpack_uint(cl, 28, 28);
   values->channel_reverse = __gen_unpack_uint(cl, 27, 27);
   values->clear_buffer_being_stored = __gen_unpack_uint(cl, 26, 26);
   values->output_image_format = __gen_unpack_uint(cl, 20, 25);
   values->decimate_mode = __gen_unpack_uint(cl, 18, 19);
   values->dither_mode = __gen_unpack_uint(cl, 16, 17);
   values->flip_y = __gen_unpack_uint(cl, 15, 15);
   values->memory_format = __gen_unpack_uint(cl, 12, 14);
   values->buffer_to_store = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_LOAD_TILE_BUFFER_GENERAL_opcode     30
#define V3D41_LOAD_TILE_BUFFER_GENERAL_header   \
   .opcode                              =     30

struct V3D41_LOAD_TILE_BUFFER_GENERAL {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             height;
   uint32_t                             height_in_ub_or_stride;
   bool                                 r_b_swap;
   bool                                 channel_reverse;
   enum V3D41_Output_Image_Format       input_image_format;
   enum V3D41_Decimate_Mode             decimate_mode;
   bool                                 flip_y;
   enum V3D41_Memory_Format             memory_format;
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
V3D41_LOAD_TILE_BUFFER_GENERAL_pack(__gen_user_data *data, uint8_t * restrict cl,
                                    const struct V3D41_LOAD_TILE_BUFFER_GENERAL * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->flip_y, 7, 7) |
            __gen_uint(values->memory_format, 4, 6) |
            __gen_uint(values->buffer_to_load, 0, 3);

   cl[ 2] = __gen_uint(values->input_image_format, 4, 9) |
            __gen_uint(values->decimate_mode, 2, 3);

   cl[ 3] = __gen_uint(values->r_b_swap, 4, 4) |
            __gen_uint(values->channel_reverse, 3, 3) |
            __gen_uint(values->input_image_format, 4, 9) >> 8;

   cl[ 4] = __gen_uint(values->height_in_ub_or_stride, 4, 23);

   cl[ 5] = __gen_uint(values->height_in_ub_or_stride, 4, 23) >> 8;

   cl[ 6] = __gen_uint(values->height_in_ub_or_stride, 4, 23) >> 16;

   cl[ 7] = __gen_uint(values->height, 0, 15);

   cl[ 8] = __gen_uint(values->height, 0, 15) >> 8;

   __gen_emit_reloc(data, &values->address);
   cl[ 9] = __gen_address_offset(&values->address);

   cl[10] = __gen_address_offset(&values->address) >> 8;

   cl[11] = __gen_address_offset(&values->address) >> 16;

   cl[12] = __gen_address_offset(&values->address) >> 24;

}

#define V3D41_LOAD_TILE_BUFFER_GENERAL_length     13
#ifdef __gen_unpack_address
static inline void
V3D41_LOAD_TILE_BUFFER_GENERAL_unpack(const uint8_t * restrict cl,
                                      struct V3D41_LOAD_TILE_BUFFER_GENERAL * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 72, 103);
   values->height = __gen_unpack_uint(cl, 56, 71);
   values->height_in_ub_or_stride = __gen_unpack_uint(cl, 36, 55);
   values->r_b_swap = __gen_unpack_uint(cl, 28, 28);
   values->channel_reverse = __gen_unpack_uint(cl, 27, 27);
   values->input_image_format = __gen_unpack_uint(cl, 20, 25);
   values->decimate_mode = __gen_unpack_uint(cl, 18, 19);
   values->flip_y = __gen_unpack_uint(cl, 15, 15);
   values->memory_format = __gen_unpack_uint(cl, 12, 14);
   values->buffer_to_load = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT_opcode     31
#define V3D41_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT_header\
   .opcode                              =     31

struct V3D41_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT {
   uint32_t                             opcode;
};

static inline void
V3D41_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT_pack(__gen_user_data *data, uint8_t * restrict cl,
                                              const struct V3D41_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT_unpack(const uint8_t * restrict cl,
                                                struct V3D41_TRANSFORM_FEEDBACK_FLUSH_AND_COUNT * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_INDEXED_PRIM_LIST_opcode        32
#define V3D41_INDEXED_PRIM_LIST_header          \
   .opcode                              =     32

struct V3D41_INDEXED_PRIM_LIST {
   uint32_t                             opcode;
   uint32_t                             index_offset;
   bool                                 enable_primitive_restarts;
   uint32_t                             length;
   uint32_t                             index_type;
#define INDEX_TYPE_8_BIT                         0
#define INDEX_TYPE_16_BIT                        1
#define INDEX_TYPE_32_BIT                        2
   enum V3D41_Primitive                 mode;
};

static inline void
V3D41_INDEXED_PRIM_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                             const struct V3D41_INDEXED_PRIM_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->index_type, 6, 7) |
            __gen_uint(values->mode, 0, 5);

   cl[ 2] = __gen_uint(values->length, 0, 30);

   cl[ 3] = __gen_uint(values->length, 0, 30) >> 8;

   cl[ 4] = __gen_uint(values->length, 0, 30) >> 16;

   cl[ 5] = __gen_uint(values->enable_primitive_restarts, 7, 7) |
            __gen_uint(values->length, 0, 30) >> 24;


   memcpy(&cl[6], &values->index_offset, sizeof(values->index_offset));
}

#define V3D41_INDEXED_PRIM_LIST_length        10
#ifdef __gen_unpack_address
static inline void
V3D41_INDEXED_PRIM_LIST_unpack(const uint8_t * restrict cl,
                               struct V3D41_INDEXED_PRIM_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->index_offset = __gen_unpack_uint(cl, 48, 79);
   values->enable_primitive_restarts = __gen_unpack_uint(cl, 47, 47);
   values->length = __gen_unpack_uint(cl, 16, 46);
   values->index_type = __gen_unpack_uint(cl, 14, 15);
   values->mode = __gen_unpack_uint(cl, 8, 13);
}
#endif


#define V3D41_INDEXED_INSTANCED_PRIM_LIST_opcode     34
#define V3D41_INDEXED_INSTANCED_PRIM_LIST_header\
   .opcode                              =     34

struct V3D41_INDEXED_INSTANCED_PRIM_LIST {
   uint32_t                             opcode;
   uint32_t                             index_offset;
   uint32_t                             number_of_instances;
   bool                                 enable_primitive_restarts;
   uint32_t                             instance_length;
   uint32_t                             index_type;
#define INDEX_TYPE_8_BIT                         0
#define INDEX_TYPE_16_BIT                        1
#define INDEX_TYPE_32_BIT                        2
   enum V3D41_Primitive                 mode;
};

static inline void
V3D41_INDEXED_INSTANCED_PRIM_LIST_pack(__gen_user_data *data, uint8_t * restrict cl,
                                       const struct V3D41_INDEXED_INSTANCED_PRIM_LIST * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->index_type, 6, 7) |
            __gen_uint(values->mode, 0, 5);

   cl[ 2] = __gen_uint(values->instance_length, 0, 30);

   cl[ 3] = __gen_uint(values->instance_length, 0, 30) >> 8;

   cl[ 4] = __gen_uint(values->instance_length, 0, 30) >> 16;

   cl[ 5] = __gen_uint(values->enable_primitive_restarts, 7, 7) |
            __gen_uint(values->instance_length, 0, 30) >> 24;


   memcpy(&cl[6], &values->number_of_instances, sizeof(values->number_of_instances));

   memcpy(&cl[10], &values->index_offset, sizeof(values->index_offset));
}

#define V3D41_INDEXED_INSTANCED_PRIM_LIST_length     14
#ifdef __gen_unpack_address
static inline void
V3D41_INDEXED_INSTANCED_PRIM_LIST_unpack(const uint8_t * restrict cl,
                                         struct V3D41_INDEXED_INSTANCED_PRIM_LIST * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->index_offset = __gen_unpack_uint(cl, 80, 111);
   values->number_of_instances = __gen_unpack_uint(cl, 48, 79);
   values->enable_primitive_restarts = __gen_unpack_uint(cl, 47, 47);
   values->instance_length = __gen_unpack_uint(cl, 16, 46);
   values->index_type = __gen_unpack_uint(cl, 14, 15);
   values->mode = __gen_unpack_uint(cl, 8, 13);
}
#endif


#define V3D41_VERTEX_ARRAY_PRIMS_opcode       36
#define V3D41_VERTEX_ARRAY_PRIMS_header         \
   .opcode                              =     36

struct V3D41_VERTEX_ARRAY_PRIMS {
   uint32_t                             opcode;
   uint32_t                             index_of_first_vertex;
   uint32_t                             length;
   enum V3D41_Primitive                 mode;
};

static inline void
V3D41_VERTEX_ARRAY_PRIMS_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D41_VERTEX_ARRAY_PRIMS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->mode, 0, 7);


   memcpy(&cl[2], &values->length, sizeof(values->length));

   memcpy(&cl[6], &values->index_of_first_vertex, sizeof(values->index_of_first_vertex));
}

#define V3D41_VERTEX_ARRAY_PRIMS_length       10
#ifdef __gen_unpack_address
static inline void
V3D41_VERTEX_ARRAY_PRIMS_unpack(const uint8_t * restrict cl,
                                struct V3D41_VERTEX_ARRAY_PRIMS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->index_of_first_vertex = __gen_unpack_uint(cl, 48, 79);
   values->length = __gen_unpack_uint(cl, 16, 47);
   values->mode = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D41_VERTEX_ARRAY_INSTANCED_PRIMS_opcode     38
#define V3D41_VERTEX_ARRAY_INSTANCED_PRIMS_header\
   .opcode                              =     38

struct V3D41_VERTEX_ARRAY_INSTANCED_PRIMS {
   uint32_t                             opcode;
   uint32_t                             index_of_first_vertex;
   uint32_t                             number_of_instances;
   uint32_t                             instance_length;
   enum V3D41_Primitive                 mode;
};

static inline void
V3D41_VERTEX_ARRAY_INSTANCED_PRIMS_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D41_VERTEX_ARRAY_INSTANCED_PRIMS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->mode, 0, 7);


   memcpy(&cl[2], &values->instance_length, sizeof(values->instance_length));

   memcpy(&cl[6], &values->number_of_instances, sizeof(values->number_of_instances));

   memcpy(&cl[10], &values->index_of_first_vertex, sizeof(values->index_of_first_vertex));
}

#define V3D41_VERTEX_ARRAY_INSTANCED_PRIMS_length     14
#ifdef __gen_unpack_address
static inline void
V3D41_VERTEX_ARRAY_INSTANCED_PRIMS_unpack(const uint8_t * restrict cl,
                                          struct V3D41_VERTEX_ARRAY_INSTANCED_PRIMS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->index_of_first_vertex = __gen_unpack_uint(cl, 80, 111);
   values->number_of_instances = __gen_unpack_uint(cl, 48, 79);
   values->instance_length = __gen_unpack_uint(cl, 16, 47);
   values->mode = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D41_BASE_VERTEX_BASE_INSTANCE_opcode     43
#define V3D41_BASE_VERTEX_BASE_INSTANCE_header  \
   .opcode                              =     43

struct V3D41_BASE_VERTEX_BASE_INSTANCE {
   uint32_t                             opcode;
   uint32_t                             base_instance;
   uint32_t                             base_vertex;
};

static inline void
V3D41_BASE_VERTEX_BASE_INSTANCE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                     const struct V3D41_BASE_VERTEX_BASE_INSTANCE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->base_vertex, sizeof(values->base_vertex));

   memcpy(&cl[5], &values->base_instance, sizeof(values->base_instance));
}

#define V3D41_BASE_VERTEX_BASE_INSTANCE_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_BASE_VERTEX_BASE_INSTANCE_unpack(const uint8_t * restrict cl,
                                       struct V3D41_BASE_VERTEX_BASE_INSTANCE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->base_instance = __gen_unpack_uint(cl, 40, 71);
   values->base_vertex = __gen_unpack_uint(cl, 8, 39);
}
#endif


#define V3D41_INDEX_BUFFER_SETUP_opcode       44
#define V3D41_INDEX_BUFFER_SETUP_header         \
   .opcode                              =     44

struct V3D41_INDEX_BUFFER_SETUP {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             size;
};

static inline void
V3D41_INDEX_BUFFER_SETUP_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D41_INDEX_BUFFER_SETUP * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;


   memcpy(&cl[5], &values->size, sizeof(values->size));
}

#define V3D41_INDEX_BUFFER_SETUP_length        9
#ifdef __gen_unpack_address
static inline void
V3D41_INDEX_BUFFER_SETUP_unpack(const uint8_t * restrict cl,
                                struct V3D41_INDEX_BUFFER_SETUP * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 8, 39);
   values->size = __gen_unpack_uint(cl, 40, 71);
}
#endif


#define V3D41_PRIM_LIST_FORMAT_opcode         56
#define V3D41_PRIM_LIST_FORMAT_header           \
   .opcode                              =     56

struct V3D41_PRIM_LIST_FORMAT {
   uint32_t                             opcode;
   bool                                 tri_strip_or_fan;
   uint32_t                             primitive_type;
#define LIST_POINTS                              0
#define LIST_LINES                               1
#define LIST_TRIANGLES                           2
};

static inline void
V3D41_PRIM_LIST_FORMAT_pack(__gen_user_data *data, uint8_t * restrict cl,
                            const struct V3D41_PRIM_LIST_FORMAT * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->tri_strip_or_fan, 7, 7) |
            __gen_uint(values->primitive_type, 0, 5);

}

#define V3D41_PRIM_LIST_FORMAT_length          2
#ifdef __gen_unpack_address
static inline void
V3D41_PRIM_LIST_FORMAT_unpack(const uint8_t * restrict cl,
                              struct V3D41_PRIM_LIST_FORMAT * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->tri_strip_or_fan = __gen_unpack_uint(cl, 15, 15);
   values->primitive_type = __gen_unpack_uint(cl, 8, 13);
}
#endif


#define V3D41_GL_SHADER_STATE_opcode          64
#define V3D41_GL_SHADER_STATE_header            \
   .opcode                              =     64

struct V3D41_GL_SHADER_STATE {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             number_of_attribute_arrays;
};

static inline void
V3D41_GL_SHADER_STATE_pack(__gen_user_data *data, uint8_t * restrict cl,
                           const struct V3D41_GL_SHADER_STATE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address) |
            __gen_uint(values->number_of_attribute_arrays, 0, 4);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D41_GL_SHADER_STATE_length           5
#ifdef __gen_unpack_address
static inline void
V3D41_GL_SHADER_STATE_unpack(const uint8_t * restrict cl,
                             struct V3D41_GL_SHADER_STATE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 13, 39);
   values->number_of_attribute_arrays = __gen_unpack_uint(cl, 8, 12);
}
#endif


#define V3D41_VCM_CACHE_SIZE_opcode           71
#define V3D41_VCM_CACHE_SIZE_header             \
   .opcode                              =     71

struct V3D41_VCM_CACHE_SIZE {
   uint32_t                             opcode;
   uint32_t                             number_of_16_vertex_batches_for_rendering;
   uint32_t                             number_of_16_vertex_batches_for_binning;
};

static inline void
V3D41_VCM_CACHE_SIZE_pack(__gen_user_data *data, uint8_t * restrict cl,
                          const struct V3D41_VCM_CACHE_SIZE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->number_of_16_vertex_batches_for_rendering, 4, 7) |
            __gen_uint(values->number_of_16_vertex_batches_for_binning, 0, 3);

}

#define V3D41_VCM_CACHE_SIZE_length            2
#ifdef __gen_unpack_address
static inline void
V3D41_VCM_CACHE_SIZE_unpack(const uint8_t * restrict cl,
                            struct V3D41_VCM_CACHE_SIZE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->number_of_16_vertex_batches_for_rendering = __gen_unpack_uint(cl, 12, 15);
   values->number_of_16_vertex_batches_for_binning = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_TRANSFORM_FEEDBACK_BUFFER_opcode     73
#define V3D41_TRANSFORM_FEEDBACK_BUFFER_header  \
   .opcode                              =     73

struct V3D41_TRANSFORM_FEEDBACK_BUFFER {
   uint32_t                             opcode;
   __gen_address_type                   buffer_address;
   uint32_t                             buffer_size_in_32_bit_words;
   uint32_t                             buffer_number;
};

static inline void
V3D41_TRANSFORM_FEEDBACK_BUFFER_pack(__gen_user_data *data, uint8_t * restrict cl,
                                     const struct V3D41_TRANSFORM_FEEDBACK_BUFFER * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->buffer_size_in_32_bit_words, 2, 31) |
            __gen_uint(values->buffer_number, 0, 1);

   cl[ 2] = __gen_uint(values->buffer_size_in_32_bit_words, 2, 31) >> 8;

   cl[ 3] = __gen_uint(values->buffer_size_in_32_bit_words, 2, 31) >> 16;

   cl[ 4] = __gen_uint(values->buffer_size_in_32_bit_words, 2, 31) >> 24;

   __gen_emit_reloc(data, &values->buffer_address);
   cl[ 5] = __gen_address_offset(&values->buffer_address);

   cl[ 6] = __gen_address_offset(&values->buffer_address) >> 8;

   cl[ 7] = __gen_address_offset(&values->buffer_address) >> 16;

   cl[ 8] = __gen_address_offset(&values->buffer_address) >> 24;

}

#define V3D41_TRANSFORM_FEEDBACK_BUFFER_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_TRANSFORM_FEEDBACK_BUFFER_unpack(const uint8_t * restrict cl,
                                       struct V3D41_TRANSFORM_FEEDBACK_BUFFER * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->buffer_address = __gen_unpack_address(cl, 40, 71);
   values->buffer_size_in_32_bit_words = __gen_unpack_uint(cl, 10, 39);
   values->buffer_number = __gen_unpack_uint(cl, 8, 9);
}
#endif


#define V3D41_TRANSFORM_FEEDBACK_SPECS_opcode     74
#define V3D41_TRANSFORM_FEEDBACK_SPECS_header   \
   .opcode                              =     74

struct V3D41_TRANSFORM_FEEDBACK_SPECS {
   uint32_t                             opcode;
   bool                                 enable;
   uint32_t                             number_of_16_bit_output_data_specs_following;
};

static inline void
V3D41_TRANSFORM_FEEDBACK_SPECS_pack(__gen_user_data *data, uint8_t * restrict cl,
                                    const struct V3D41_TRANSFORM_FEEDBACK_SPECS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->enable, 7, 7) |
            __gen_uint(values->number_of_16_bit_output_data_specs_following, 0, 4);

}

#define V3D41_TRANSFORM_FEEDBACK_SPECS_length      2
#ifdef __gen_unpack_address
static inline void
V3D41_TRANSFORM_FEEDBACK_SPECS_unpack(const uint8_t * restrict cl,
                                      struct V3D41_TRANSFORM_FEEDBACK_SPECS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->enable = __gen_unpack_uint(cl, 15, 15);
   values->number_of_16_bit_output_data_specs_following = __gen_unpack_uint(cl, 8, 12);
}
#endif


#define V3D41_FLUSH_TRANSFORM_FEEDBACK_DATA_opcode     75
#define V3D41_FLUSH_TRANSFORM_FEEDBACK_DATA_header\
   .opcode                              =     75

struct V3D41_FLUSH_TRANSFORM_FEEDBACK_DATA {
   uint32_t                             opcode;
};

static inline void
V3D41_FLUSH_TRANSFORM_FEEDBACK_DATA_pack(__gen_user_data *data, uint8_t * restrict cl,
                                         const struct V3D41_FLUSH_TRANSFORM_FEEDBACK_DATA * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_FLUSH_TRANSFORM_FEEDBACK_DATA_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_FLUSH_TRANSFORM_FEEDBACK_DATA_unpack(const uint8_t * restrict cl,
                                           struct V3D41_FLUSH_TRANSFORM_FEEDBACK_DATA * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC_header\


struct V3D41_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC {
   uint32_t                             first_shaded_vertex_value_to_output;
   uint32_t                             number_of_consecutive_vertex_values_to_output_as_32_bit_values;
   uint32_t                             output_buffer_to_write_to;
   uint32_t                             stream_number;
};

static inline void
V3D41_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC_pack(__gen_user_data *data, uint8_t * restrict cl,
                                               const struct V3D41_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC * restrict values)
{
   assert(values->number_of_consecutive_vertex_values_to_output_as_32_bit_values >= 1);
   cl[ 0] = __gen_uint(values->first_shaded_vertex_value_to_output, 0, 7);

   cl[ 1] = __gen_uint(values->number_of_consecutive_vertex_values_to_output_as_32_bit_values - 1, 0, 3) |
            __gen_uint(values->output_buffer_to_write_to, 4, 5) |
            __gen_uint(values->stream_number, 6, 7);

}

#define V3D41_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC_length      2
#ifdef __gen_unpack_address
static inline void
V3D41_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC_unpack(const uint8_t * restrict cl,
                                                 struct V3D41_TRANSFORM_FEEDBACK_OUTPUT_DATA_SPEC * restrict values)
{
   values->first_shaded_vertex_value_to_output = __gen_unpack_uint(cl, 0, 7);
   values->number_of_consecutive_vertex_values_to_output_as_32_bit_values = __gen_unpack_uint(cl, 8, 11) + 1;
   values->output_buffer_to_write_to = __gen_unpack_uint(cl, 12, 13);
   values->stream_number = __gen_unpack_uint(cl, 14, 15);
}
#endif


#define V3D41_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS_header\


struct V3D41_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS {
   __gen_address_type                   address;
};

static inline void
V3D41_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS_pack(__gen_user_data *data, uint8_t * restrict cl,
                                             const struct V3D41_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS * restrict values)
{
   __gen_emit_reloc(data, &values->address);
   cl[ 0] = __gen_address_offset(&values->address);

   cl[ 1] = __gen_address_offset(&values->address) >> 8;

   cl[ 2] = __gen_address_offset(&values->address) >> 16;

   cl[ 3] = __gen_address_offset(&values->address) >> 24;

}

#define V3D41_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS_length      4
#ifdef __gen_unpack_address
static inline void
V3D41_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS_unpack(const uint8_t * restrict cl,
                                               struct V3D41_TRANSFORM_FEEDBACK_OUTPUT_ADDRESS * restrict values)
{
   values->address = __gen_unpack_address(cl, 0, 31);
}
#endif


#define V3D41_STENCIL_CFG_opcode              80
#define V3D41_STENCIL_CFG_header                \
   .opcode                              =     80

struct V3D41_STENCIL_CFG {
   uint32_t                             opcode;
   uint32_t                             stencil_write_mask;
   bool                                 back_config;
   bool                                 front_config;
   enum V3D41_Stencil_Op                stencil_pass_op;
   enum V3D41_Stencil_Op                depth_test_fail_op;
   enum V3D41_Stencil_Op                stencil_test_fail_op;
   enum V3D41_Compare_Function          stencil_test_function;
   uint32_t                             stencil_test_mask;
   uint32_t                             stencil_ref_value;
};

static inline void
V3D41_STENCIL_CFG_pack(__gen_user_data *data, uint8_t * restrict cl,
                       const struct V3D41_STENCIL_CFG * restrict values)
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

#define V3D41_STENCIL_CFG_length               6
#ifdef __gen_unpack_address
static inline void
V3D41_STENCIL_CFG_unpack(const uint8_t * restrict cl,
                         struct V3D41_STENCIL_CFG * restrict values)
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


#define V3D41_BLEND_ENABLES_opcode            83
#define V3D41_BLEND_ENABLES_header              \
   .opcode                              =     83

struct V3D41_BLEND_ENABLES {
   uint32_t                             opcode;
   uint32_t                             mask;
};

static inline void
V3D41_BLEND_ENABLES_pack(__gen_user_data *data, uint8_t * restrict cl,
                         const struct V3D41_BLEND_ENABLES * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->mask, 0, 7);

}

#define V3D41_BLEND_ENABLES_length             2
#ifdef __gen_unpack_address
static inline void
V3D41_BLEND_ENABLES_unpack(const uint8_t * restrict cl,
                           struct V3D41_BLEND_ENABLES * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->mask = __gen_unpack_uint(cl, 8, 15);
}
#endif


#define V3D41_BLEND_CFG_opcode                84
#define V3D41_BLEND_CFG_header                  \
   .opcode                              =     84

struct V3D41_BLEND_CFG {
   uint32_t                             opcode;
   uint32_t                             render_target_mask;
   enum V3D41_Blend_Factor              color_blend_dst_factor;
   enum V3D41_Blend_Factor              color_blend_src_factor;
   enum V3D41_Blend_Mode                color_blend_mode;
   enum V3D41_Blend_Factor              alpha_blend_dst_factor;
   enum V3D41_Blend_Factor              alpha_blend_src_factor;
   enum V3D41_Blend_Mode                alpha_blend_mode;
};

static inline void
V3D41_BLEND_CFG_pack(__gen_user_data *data, uint8_t * restrict cl,
                     const struct V3D41_BLEND_CFG * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->alpha_blend_src_factor, 4, 7) |
            __gen_uint(values->alpha_blend_mode, 0, 3);

   cl[ 2] = __gen_uint(values->color_blend_mode, 4, 7) |
            __gen_uint(values->alpha_blend_dst_factor, 0, 3);

   cl[ 3] = __gen_uint(values->color_blend_dst_factor, 4, 7) |
            __gen_uint(values->color_blend_src_factor, 0, 3);

   cl[ 4] = __gen_uint(values->render_target_mask, 0, 3);

}

#define V3D41_BLEND_CFG_length                 5
#ifdef __gen_unpack_address
static inline void
V3D41_BLEND_CFG_unpack(const uint8_t * restrict cl,
                       struct V3D41_BLEND_CFG * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->render_target_mask = __gen_unpack_uint(cl, 32, 35);
   values->color_blend_dst_factor = __gen_unpack_uint(cl, 28, 31);
   values->color_blend_src_factor = __gen_unpack_uint(cl, 24, 27);
   values->color_blend_mode = __gen_unpack_uint(cl, 20, 23);
   values->alpha_blend_dst_factor = __gen_unpack_uint(cl, 16, 19);
   values->alpha_blend_src_factor = __gen_unpack_uint(cl, 12, 15);
   values->alpha_blend_mode = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_BLEND_CONSTANT_COLOR_opcode     86
#define V3D41_BLEND_CONSTANT_COLOR_header       \
   .opcode                              =     86

struct V3D41_BLEND_CONSTANT_COLOR {
   uint32_t                             opcode;
   uint32_t                             alpha_f16;
   uint32_t                             blue_f16;
   uint32_t                             green_f16;
   uint32_t                             red_f16;
};

static inline void
V3D41_BLEND_CONSTANT_COLOR_pack(__gen_user_data *data, uint8_t * restrict cl,
                                const struct V3D41_BLEND_CONSTANT_COLOR * restrict values)
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

#define V3D41_BLEND_CONSTANT_COLOR_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_BLEND_CONSTANT_COLOR_unpack(const uint8_t * restrict cl,
                                  struct V3D41_BLEND_CONSTANT_COLOR * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->alpha_f16 = __gen_unpack_uint(cl, 56, 71);
   values->blue_f16 = __gen_unpack_uint(cl, 40, 55);
   values->green_f16 = __gen_unpack_uint(cl, 24, 39);
   values->red_f16 = __gen_unpack_uint(cl, 8, 23);
}
#endif


#define V3D41_COLOR_WRITE_MASKS_opcode        87
#define V3D41_COLOR_WRITE_MASKS_header          \
   .opcode                              =     87

struct V3D41_COLOR_WRITE_MASKS {
   uint32_t                             opcode;
   uint32_t                             mask;
};

static inline void
V3D41_COLOR_WRITE_MASKS_pack(__gen_user_data *data, uint8_t * restrict cl,
                             const struct V3D41_COLOR_WRITE_MASKS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->mask, sizeof(values->mask));
}

#define V3D41_COLOR_WRITE_MASKS_length         5
#ifdef __gen_unpack_address
static inline void
V3D41_COLOR_WRITE_MASKS_unpack(const uint8_t * restrict cl,
                               struct V3D41_COLOR_WRITE_MASKS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->mask = __gen_unpack_uint(cl, 8, 39);
}
#endif


#define V3D41_ZERO_ALL_CENTROID_FLAGS_opcode     88
#define V3D41_ZERO_ALL_CENTROID_FLAGS_header    \
   .opcode                              =     88

struct V3D41_ZERO_ALL_CENTROID_FLAGS {
   uint32_t                             opcode;
};

static inline void
V3D41_ZERO_ALL_CENTROID_FLAGS_pack(__gen_user_data *data, uint8_t * restrict cl,
                                   const struct V3D41_ZERO_ALL_CENTROID_FLAGS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_ZERO_ALL_CENTROID_FLAGS_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_ZERO_ALL_CENTROID_FLAGS_unpack(const uint8_t * restrict cl,
                                     struct V3D41_ZERO_ALL_CENTROID_FLAGS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_CENTROID_FLAGS_opcode           89
#define V3D41_CENTROID_FLAGS_header             \
   .opcode                              =     89

struct V3D41_CENTROID_FLAGS {
   uint32_t                             opcode;
   uint32_t                             centroid_flags_for_varyings_v024;
   enum V3D41_Varying_Flags_Action      action_for_centroid_flags_of_higher_numbered_varyings;
   enum V3D41_Varying_Flags_Action      action_for_centroid_flags_of_lower_numbered_varyings;
   uint32_t                             varying_offset_v0;
};

static inline void
V3D41_CENTROID_FLAGS_pack(__gen_user_data *data, uint8_t * restrict cl,
                          const struct V3D41_CENTROID_FLAGS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->action_for_centroid_flags_of_higher_numbered_varyings, 6, 7) |
            __gen_uint(values->action_for_centroid_flags_of_lower_numbered_varyings, 4, 5) |
            __gen_uint(values->varying_offset_v0, 0, 3);

   cl[ 2] = __gen_uint(values->centroid_flags_for_varyings_v024, 0, 23);

   cl[ 3] = __gen_uint(values->centroid_flags_for_varyings_v024, 0, 23) >> 8;

   cl[ 4] = __gen_uint(values->centroid_flags_for_varyings_v024, 0, 23) >> 16;

}

#define V3D41_CENTROID_FLAGS_length            5
#ifdef __gen_unpack_address
static inline void
V3D41_CENTROID_FLAGS_unpack(const uint8_t * restrict cl,
                            struct V3D41_CENTROID_FLAGS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->centroid_flags_for_varyings_v024 = __gen_unpack_uint(cl, 16, 39);
   values->action_for_centroid_flags_of_higher_numbered_varyings = __gen_unpack_uint(cl, 14, 15);
   values->action_for_centroid_flags_of_lower_numbered_varyings = __gen_unpack_uint(cl, 12, 13);
   values->varying_offset_v0 = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_SAMPLE_STATE_opcode             91
#define V3D41_SAMPLE_STATE_header               \
   .opcode                              =     91

struct V3D41_SAMPLE_STATE {
   uint32_t                             opcode;
   float                                coverage;
   uint32_t                             mask;
};

static inline void
V3D41_SAMPLE_STATE_pack(__gen_user_data *data, uint8_t * restrict cl,
                        const struct V3D41_SAMPLE_STATE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->mask, 0, 3);

   cl[ 2] = 0;
   cl[ 3] = __gen_uint(fui(values->coverage) >> 16, 0, 15);

   cl[ 4] = __gen_uint(fui(values->coverage) >> 16, 0, 15) >> 8;

}

#define V3D41_SAMPLE_STATE_length              5
#ifdef __gen_unpack_address
static inline void
V3D41_SAMPLE_STATE_unpack(const uint8_t * restrict cl,
                          struct V3D41_SAMPLE_STATE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->coverage = __gen_unpack_f187(cl, 24, 39);
   values->mask = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_OCCLUSION_QUERY_COUNTER_opcode     92
#define V3D41_OCCLUSION_QUERY_COUNTER_header    \
   .opcode                              =     92

struct V3D41_OCCLUSION_QUERY_COUNTER {
   uint32_t                             opcode;
   __gen_address_type                   address;
};

static inline void
V3D41_OCCLUSION_QUERY_COUNTER_pack(__gen_user_data *data, uint8_t * restrict cl,
                                   const struct V3D41_OCCLUSION_QUERY_COUNTER * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D41_OCCLUSION_QUERY_COUNTER_length      5
#ifdef __gen_unpack_address
static inline void
V3D41_OCCLUSION_QUERY_COUNTER_unpack(const uint8_t * restrict cl,
                                     struct V3D41_OCCLUSION_QUERY_COUNTER * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 8, 39);
}
#endif


#define V3D41_CFG_BITS_opcode                 96
#define V3D41_CFG_BITS_header                   \
   .opcode                              =     96

struct V3D41_CFG_BITS {
   uint32_t                             opcode;
   bool                                 direct3d_provoking_vertex;
   bool                                 direct3d_point_fill_mode;
   bool                                 blend_enable;
   bool                                 stencil_enable;
   bool                                 early_z_updates_enable;
   bool                                 early_z_enable;
   bool                                 z_updates_enable;
   enum V3D41_Compare_Function          depth_test_function;
   bool                                 direct3d_wireframe_triangles_mode;
   uint32_t                             rasterizer_oversample_mode;
   uint32_t                             line_rasterization;
   bool                                 enable_depth_offset;
   bool                                 clockwise_primitives;
   bool                                 enable_reverse_facing_primitive;
   bool                                 enable_forward_facing_primitive;
};

static inline void
V3D41_CFG_BITS_pack(__gen_user_data *data, uint8_t * restrict cl,
                    const struct V3D41_CFG_BITS * restrict values)
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

#define V3D41_CFG_BITS_length                  4
#ifdef __gen_unpack_address
static inline void
V3D41_CFG_BITS_unpack(const uint8_t * restrict cl,
                      struct V3D41_CFG_BITS * restrict values)
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


#define V3D41_ZERO_ALL_FLAT_SHADE_FLAGS_opcode     97
#define V3D41_ZERO_ALL_FLAT_SHADE_FLAGS_header  \
   .opcode                              =     97

struct V3D41_ZERO_ALL_FLAT_SHADE_FLAGS {
   uint32_t                             opcode;
};

static inline void
V3D41_ZERO_ALL_FLAT_SHADE_FLAGS_pack(__gen_user_data *data, uint8_t * restrict cl,
                                     const struct V3D41_ZERO_ALL_FLAT_SHADE_FLAGS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_ZERO_ALL_FLAT_SHADE_FLAGS_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_ZERO_ALL_FLAT_SHADE_FLAGS_unpack(const uint8_t * restrict cl,
                                       struct V3D41_ZERO_ALL_FLAT_SHADE_FLAGS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_FLAT_SHADE_FLAGS_opcode         98
#define V3D41_FLAT_SHADE_FLAGS_header           \
   .opcode                              =     98

struct V3D41_FLAT_SHADE_FLAGS {
   uint32_t                             opcode;
   uint32_t                             flat_shade_flags_for_varyings_v024;
   enum V3D41_Varying_Flags_Action      action_for_flat_shade_flags_of_higher_numbered_varyings;
   enum V3D41_Varying_Flags_Action      action_for_flat_shade_flags_of_lower_numbered_varyings;
   uint32_t                             varying_offset_v0;
};

static inline void
V3D41_FLAT_SHADE_FLAGS_pack(__gen_user_data *data, uint8_t * restrict cl,
                            const struct V3D41_FLAT_SHADE_FLAGS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->action_for_flat_shade_flags_of_higher_numbered_varyings, 6, 7) |
            __gen_uint(values->action_for_flat_shade_flags_of_lower_numbered_varyings, 4, 5) |
            __gen_uint(values->varying_offset_v0, 0, 3);

   cl[ 2] = __gen_uint(values->flat_shade_flags_for_varyings_v024, 0, 23);

   cl[ 3] = __gen_uint(values->flat_shade_flags_for_varyings_v024, 0, 23) >> 8;

   cl[ 4] = __gen_uint(values->flat_shade_flags_for_varyings_v024, 0, 23) >> 16;

}

#define V3D41_FLAT_SHADE_FLAGS_length          5
#ifdef __gen_unpack_address
static inline void
V3D41_FLAT_SHADE_FLAGS_unpack(const uint8_t * restrict cl,
                              struct V3D41_FLAT_SHADE_FLAGS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->flat_shade_flags_for_varyings_v024 = __gen_unpack_uint(cl, 16, 39);
   values->action_for_flat_shade_flags_of_higher_numbered_varyings = __gen_unpack_uint(cl, 14, 15);
   values->action_for_flat_shade_flags_of_lower_numbered_varyings = __gen_unpack_uint(cl, 12, 13);
   values->varying_offset_v0 = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_ZERO_ALL_NON_PERSPECTIVE_FLAGS_opcode     99
#define V3D41_ZERO_ALL_NON_PERSPECTIVE_FLAGS_header\
   .opcode                              =     99

struct V3D41_ZERO_ALL_NON_PERSPECTIVE_FLAGS {
   uint32_t                             opcode;
};

static inline void
V3D41_ZERO_ALL_NON_PERSPECTIVE_FLAGS_pack(__gen_user_data *data, uint8_t * restrict cl,
                                          const struct V3D41_ZERO_ALL_NON_PERSPECTIVE_FLAGS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_ZERO_ALL_NON_PERSPECTIVE_FLAGS_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_ZERO_ALL_NON_PERSPECTIVE_FLAGS_unpack(const uint8_t * restrict cl,
                                            struct V3D41_ZERO_ALL_NON_PERSPECTIVE_FLAGS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_NON_PERSPECTIVE_FLAGS_opcode    100
#define V3D41_NON_PERSPECTIVE_FLAGS_header      \
   .opcode                              =    100

struct V3D41_NON_PERSPECTIVE_FLAGS {
   uint32_t                             opcode;
   uint32_t                             non_perspective_flags_for_varyings_v024;
   enum V3D41_Varying_Flags_Action      action_for_non_perspective_flags_of_higher_numbered_varyings;
   enum V3D41_Varying_Flags_Action      action_for_non_perspective_flags_of_lower_numbered_varyings;
   uint32_t                             varying_offset_v0;
};

static inline void
V3D41_NON_PERSPECTIVE_FLAGS_pack(__gen_user_data *data, uint8_t * restrict cl,
                                 const struct V3D41_NON_PERSPECTIVE_FLAGS * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->action_for_non_perspective_flags_of_higher_numbered_varyings, 6, 7) |
            __gen_uint(values->action_for_non_perspective_flags_of_lower_numbered_varyings, 4, 5) |
            __gen_uint(values->varying_offset_v0, 0, 3);

   cl[ 2] = __gen_uint(values->non_perspective_flags_for_varyings_v024, 0, 23);

   cl[ 3] = __gen_uint(values->non_perspective_flags_for_varyings_v024, 0, 23) >> 8;

   cl[ 4] = __gen_uint(values->non_perspective_flags_for_varyings_v024, 0, 23) >> 16;

}

#define V3D41_NON_PERSPECTIVE_FLAGS_length      5
#ifdef __gen_unpack_address
static inline void
V3D41_NON_PERSPECTIVE_FLAGS_unpack(const uint8_t * restrict cl,
                                   struct V3D41_NON_PERSPECTIVE_FLAGS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->non_perspective_flags_for_varyings_v024 = __gen_unpack_uint(cl, 16, 39);
   values->action_for_non_perspective_flags_of_higher_numbered_varyings = __gen_unpack_uint(cl, 14, 15);
   values->action_for_non_perspective_flags_of_lower_numbered_varyings = __gen_unpack_uint(cl, 12, 13);
   values->varying_offset_v0 = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_POINT_SIZE_opcode              104
#define V3D41_POINT_SIZE_header                 \
   .opcode                              =    104

struct V3D41_POINT_SIZE {
   uint32_t                             opcode;
   float                                point_size;
};

static inline void
V3D41_POINT_SIZE_pack(__gen_user_data *data, uint8_t * restrict cl,
                      const struct V3D41_POINT_SIZE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->point_size, sizeof(values->point_size));
}

#define V3D41_POINT_SIZE_length                5
#ifdef __gen_unpack_address
static inline void
V3D41_POINT_SIZE_unpack(const uint8_t * restrict cl,
                        struct V3D41_POINT_SIZE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->point_size = __gen_unpack_float(cl, 8, 39);
}
#endif


#define V3D41_LINE_WIDTH_opcode              105
#define V3D41_LINE_WIDTH_header                 \
   .opcode                              =    105

struct V3D41_LINE_WIDTH {
   uint32_t                             opcode;
   float                                line_width;
};

static inline void
V3D41_LINE_WIDTH_pack(__gen_user_data *data, uint8_t * restrict cl,
                      const struct V3D41_LINE_WIDTH * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->line_width, sizeof(values->line_width));
}

#define V3D41_LINE_WIDTH_length                5
#ifdef __gen_unpack_address
static inline void
V3D41_LINE_WIDTH_unpack(const uint8_t * restrict cl,
                        struct V3D41_LINE_WIDTH * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->line_width = __gen_unpack_float(cl, 8, 39);
}
#endif


#define V3D41_DEPTH_OFFSET_opcode            106
#define V3D41_DEPTH_OFFSET_header               \
   .opcode                              =    106

struct V3D41_DEPTH_OFFSET {
   uint32_t                             opcode;
   float                                limit;
   float                                depth_offset_units;
   float                                depth_offset_factor;
};

static inline void
V3D41_DEPTH_OFFSET_pack(__gen_user_data *data, uint8_t * restrict cl,
                        const struct V3D41_DEPTH_OFFSET * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(fui(values->depth_offset_factor) >> 16, 0, 15);

   cl[ 2] = __gen_uint(fui(values->depth_offset_factor) >> 16, 0, 15) >> 8;

   cl[ 3] = __gen_uint(fui(values->depth_offset_units) >> 16, 0, 15);

   cl[ 4] = __gen_uint(fui(values->depth_offset_units) >> 16, 0, 15) >> 8;


   memcpy(&cl[5], &values->limit, sizeof(values->limit));
}

#define V3D41_DEPTH_OFFSET_length              9
#ifdef __gen_unpack_address
static inline void
V3D41_DEPTH_OFFSET_unpack(const uint8_t * restrict cl,
                          struct V3D41_DEPTH_OFFSET * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->limit = __gen_unpack_float(cl, 40, 71);
   values->depth_offset_units = __gen_unpack_f187(cl, 24, 39);
   values->depth_offset_factor = __gen_unpack_f187(cl, 8, 23);
}
#endif


#define V3D41_CLIP_WINDOW_opcode             107
#define V3D41_CLIP_WINDOW_header                \
   .opcode                              =    107

struct V3D41_CLIP_WINDOW {
   uint32_t                             opcode;
   uint32_t                             clip_window_height_in_pixels;
   uint32_t                             clip_window_width_in_pixels;
   uint32_t                             clip_window_bottom_pixel_coordinate;
   uint32_t                             clip_window_left_pixel_coordinate;
};

static inline void
V3D41_CLIP_WINDOW_pack(__gen_user_data *data, uint8_t * restrict cl,
                       const struct V3D41_CLIP_WINDOW * restrict values)
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

#define V3D41_CLIP_WINDOW_length               9
#ifdef __gen_unpack_address
static inline void
V3D41_CLIP_WINDOW_unpack(const uint8_t * restrict cl,
                         struct V3D41_CLIP_WINDOW * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->clip_window_height_in_pixels = __gen_unpack_uint(cl, 56, 71);
   values->clip_window_width_in_pixels = __gen_unpack_uint(cl, 40, 55);
   values->clip_window_bottom_pixel_coordinate = __gen_unpack_uint(cl, 24, 39);
   values->clip_window_left_pixel_coordinate = __gen_unpack_uint(cl, 8, 23);
}
#endif


#define V3D41_VIEWPORT_OFFSET_opcode         108
#define V3D41_VIEWPORT_OFFSET_header            \
   .opcode                              =    108

struct V3D41_VIEWPORT_OFFSET {
   uint32_t                             opcode;
   uint32_t                             coarse_y;
   float                                viewport_centre_y_coordinate;
   uint32_t                             coarse_x;
   float                                viewport_centre_x_coordinate;
};

static inline void
V3D41_VIEWPORT_OFFSET_pack(__gen_user_data *data, uint8_t * restrict cl,
                           const struct V3D41_VIEWPORT_OFFSET * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_sfixed(values->viewport_centre_x_coordinate, 0, 21, 8);

   cl[ 2] = __gen_sfixed(values->viewport_centre_x_coordinate, 0, 21, 8) >> 8;

   cl[ 3] = __gen_uint(values->coarse_x, 6, 15) |
            __gen_sfixed(values->viewport_centre_x_coordinate, 0, 21, 8) >> 16;

   cl[ 4] = __gen_uint(values->coarse_x, 6, 15) >> 8;

   cl[ 5] = __gen_sfixed(values->viewport_centre_y_coordinate, 0, 21, 8);

   cl[ 6] = __gen_sfixed(values->viewport_centre_y_coordinate, 0, 21, 8) >> 8;

   cl[ 7] = __gen_uint(values->coarse_y, 6, 15) |
            __gen_sfixed(values->viewport_centre_y_coordinate, 0, 21, 8) >> 16;

   cl[ 8] = __gen_uint(values->coarse_y, 6, 15) >> 8;

}

#define V3D41_VIEWPORT_OFFSET_length           9
#ifdef __gen_unpack_address
static inline void
V3D41_VIEWPORT_OFFSET_unpack(const uint8_t * restrict cl,
                             struct V3D41_VIEWPORT_OFFSET * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->coarse_y = __gen_unpack_uint(cl, 62, 71);
   values->viewport_centre_y_coordinate = __gen_unpack_sfixed(cl, 40, 61, 8);
   values->coarse_x = __gen_unpack_uint(cl, 30, 39);
   values->viewport_centre_x_coordinate = __gen_unpack_sfixed(cl, 8, 29, 8);
}
#endif


#define V3D41_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES_opcode    109
#define V3D41_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES_header\
   .opcode                              =    109

struct V3D41_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES {
   uint32_t                             opcode;
   float                                maximum_zw;
   float                                minimum_zw;
};

static inline void
V3D41_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES_pack(__gen_user_data *data, uint8_t * restrict cl,
                                             const struct V3D41_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->minimum_zw, sizeof(values->minimum_zw));

   memcpy(&cl[5], &values->maximum_zw, sizeof(values->maximum_zw));
}

#define V3D41_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES_unpack(const uint8_t * restrict cl,
                                               struct V3D41_CLIPPER_Z_MIN_MAX_CLIPPING_PLANES * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->maximum_zw = __gen_unpack_float(cl, 40, 71);
   values->minimum_zw = __gen_unpack_float(cl, 8, 39);
}
#endif


#define V3D41_CLIPPER_XY_SCALING_opcode      110
#define V3D41_CLIPPER_XY_SCALING_header         \
   .opcode                              =    110

struct V3D41_CLIPPER_XY_SCALING {
   uint32_t                             opcode;
   float                                viewport_half_height_in_1_256th_of_pixel;
   float                                viewport_half_width_in_1_256th_of_pixel;
};

static inline void
V3D41_CLIPPER_XY_SCALING_pack(__gen_user_data *data, uint8_t * restrict cl,
                              const struct V3D41_CLIPPER_XY_SCALING * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->viewport_half_width_in_1_256th_of_pixel, sizeof(values->viewport_half_width_in_1_256th_of_pixel));

   memcpy(&cl[5], &values->viewport_half_height_in_1_256th_of_pixel, sizeof(values->viewport_half_height_in_1_256th_of_pixel));
}

#define V3D41_CLIPPER_XY_SCALING_length        9
#ifdef __gen_unpack_address
static inline void
V3D41_CLIPPER_XY_SCALING_unpack(const uint8_t * restrict cl,
                                struct V3D41_CLIPPER_XY_SCALING * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->viewport_half_height_in_1_256th_of_pixel = __gen_unpack_float(cl, 40, 71);
   values->viewport_half_width_in_1_256th_of_pixel = __gen_unpack_float(cl, 8, 39);
}
#endif


#define V3D41_CLIPPER_Z_SCALE_AND_OFFSET_opcode    111
#define V3D41_CLIPPER_Z_SCALE_AND_OFFSET_header \
   .opcode                              =    111

struct V3D41_CLIPPER_Z_SCALE_AND_OFFSET {
   uint32_t                             opcode;
   float                                viewport_z_offset_zc_to_zs;
   float                                viewport_z_scale_zc_to_zs;
};

static inline void
V3D41_CLIPPER_Z_SCALE_AND_OFFSET_pack(__gen_user_data *data, uint8_t * restrict cl,
                                      const struct V3D41_CLIPPER_Z_SCALE_AND_OFFSET * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);


   memcpy(&cl[1], &values->viewport_z_scale_zc_to_zs, sizeof(values->viewport_z_scale_zc_to_zs));

   memcpy(&cl[5], &values->viewport_z_offset_zc_to_zs, sizeof(values->viewport_z_offset_zc_to_zs));
}

#define V3D41_CLIPPER_Z_SCALE_AND_OFFSET_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_CLIPPER_Z_SCALE_AND_OFFSET_unpack(const uint8_t * restrict cl,
                                        struct V3D41_CLIPPER_Z_SCALE_AND_OFFSET * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->viewport_z_offset_zc_to_zs = __gen_unpack_float(cl, 40, 71);
   values->viewport_z_scale_zc_to_zs = __gen_unpack_float(cl, 8, 39);
}
#endif


#define V3D41_NUMBER_OF_LAYERS_opcode        119
#define V3D41_NUMBER_OF_LAYERS_header           \
   .opcode                              =    119

struct V3D41_NUMBER_OF_LAYERS {
   uint32_t                             opcode;
   uint32_t                             number_of_layers;
};

static inline void
V3D41_NUMBER_OF_LAYERS_pack(__gen_user_data *data, uint8_t * restrict cl,
                            const struct V3D41_NUMBER_OF_LAYERS * restrict values)
{
   assert(values->number_of_layers >= 1);
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->number_of_layers - 1, 0, 7);

}

#define V3D41_NUMBER_OF_LAYERS_length          2
#ifdef __gen_unpack_address
static inline void
V3D41_NUMBER_OF_LAYERS_unpack(const uint8_t * restrict cl,
                              struct V3D41_NUMBER_OF_LAYERS * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->number_of_layers = __gen_unpack_uint(cl, 8, 15) + 1;
}
#endif


#define V3D41_TILE_BINNING_MODE_CFG_opcode    120
#define V3D41_TILE_BINNING_MODE_CFG_header      \
   .opcode                              =    120

struct V3D41_TILE_BINNING_MODE_CFG {
   uint32_t                             opcode;
   uint32_t                             height_in_pixels;
   uint32_t                             width_in_pixels;
   bool                                 double_buffer_in_non_ms_mode;
   bool                                 multisample_mode_4x;
   uint32_t                             maximum_bpp_of_all_render_targets;
#define RENDER_TARGET_MAXIMUM_32BPP              0
#define RENDER_TARGET_MAXIMUM_64BPP              1
#define RENDER_TARGET_MAXIMUM_128BPP             2
   uint32_t                             number_of_render_targets;
   uint32_t                             tile_allocation_block_size;
#define TILE_ALLOCATION_BLOCK_SIZE_64B           0
#define TILE_ALLOCATION_BLOCK_SIZE_128B          1
#define TILE_ALLOCATION_BLOCK_SIZE_256B          2
   uint32_t                             tile_allocation_initial_block_size;
#define TILE_ALLOCATION_INITIAL_BLOCK_SIZE_64B   0
#define TILE_ALLOCATION_INITIAL_BLOCK_SIZE_128B  1
#define TILE_ALLOCATION_INITIAL_BLOCK_SIZE_256B  2
};

static inline void
V3D41_TILE_BINNING_MODE_CFG_pack(__gen_user_data *data, uint8_t * restrict cl,
                                 const struct V3D41_TILE_BINNING_MODE_CFG * restrict values)
{
   assert(values->height_in_pixels >= 1);
   assert(values->width_in_pixels >= 1);
   assert(values->number_of_render_targets >= 1);
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->tile_allocation_block_size, 4, 5) |
            __gen_uint(values->tile_allocation_initial_block_size, 2, 3);

   cl[ 2] = __gen_uint(values->double_buffer_in_non_ms_mode, 7, 7) |
            __gen_uint(values->multisample_mode_4x, 6, 6) |
            __gen_uint(values->maximum_bpp_of_all_render_targets, 4, 5) |
            __gen_uint(values->number_of_render_targets - 1, 0, 3);

   cl[ 3] = 0;
   cl[ 4] = 0;
   cl[ 5] = __gen_uint(values->width_in_pixels - 1, 0, 11);

   cl[ 6] = __gen_uint(values->width_in_pixels - 1, 0, 11) >> 8;

   cl[ 7] = __gen_uint(values->height_in_pixels - 1, 0, 11);

   cl[ 8] = __gen_uint(values->height_in_pixels - 1, 0, 11) >> 8;

}

#define V3D41_TILE_BINNING_MODE_CFG_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_TILE_BINNING_MODE_CFG_unpack(const uint8_t * restrict cl,
                                   struct V3D41_TILE_BINNING_MODE_CFG * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->height_in_pixels = __gen_unpack_uint(cl, 56, 67) + 1;
   values->width_in_pixels = __gen_unpack_uint(cl, 40, 51) + 1;
   values->double_buffer_in_non_ms_mode = __gen_unpack_uint(cl, 23, 23);
   values->multisample_mode_4x = __gen_unpack_uint(cl, 22, 22);
   values->maximum_bpp_of_all_render_targets = __gen_unpack_uint(cl, 20, 21);
   values->number_of_render_targets = __gen_unpack_uint(cl, 16, 19) + 1;
   values->tile_allocation_block_size = __gen_unpack_uint(cl, 12, 13);
   values->tile_allocation_initial_block_size = __gen_unpack_uint(cl, 10, 11);
}
#endif


#define V3D41_TILE_RENDERING_MODE_CFG_COMMON_opcode    121
#define V3D41_TILE_RENDERING_MODE_CFG_COMMON_header\
   .opcode                              =    121,  \
   .sub_id                              =      0

struct V3D41_TILE_RENDERING_MODE_CFG_COMMON {
   uint32_t                             opcode;
   uint32_t                             pad;
   bool                                 early_depth_stencil_clear;
   enum V3D41_Internal_Depth_Type       internal_depth_type;
   bool                                 early_z_disable;
   uint32_t                             early_z_test_and_update_direction;
#define EARLY_Z_DIRECTION_LT_LE                  0
#define EARLY_Z_DIRECTION_GT_GE                  1
   bool                                 double_buffer_in_non_ms_mode;
   bool                                 multisample_mode_4x;
   enum V3D41_Internal_BPP              maximum_bpp_of_all_render_targets;
   uint32_t                             image_height_pixels;
   uint32_t                             image_width_pixels;
   uint32_t                             number_of_render_targets;
   uint32_t                             sub_id;
};

static inline void
V3D41_TILE_RENDERING_MODE_CFG_COMMON_pack(__gen_user_data *data, uint8_t * restrict cl,
                                          const struct V3D41_TILE_RENDERING_MODE_CFG_COMMON * restrict values)
{
   assert(values->number_of_render_targets >= 1);
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->number_of_render_targets - 1, 4, 7) |
            __gen_uint(values->sub_id, 0, 3);

   cl[ 2] = __gen_uint(values->image_width_pixels, 0, 15);

   cl[ 3] = __gen_uint(values->image_width_pixels, 0, 15) >> 8;

   cl[ 4] = __gen_uint(values->image_height_pixels, 0, 15);

   cl[ 5] = __gen_uint(values->image_height_pixels, 0, 15) >> 8;

   cl[ 6] = __gen_uint(values->internal_depth_type, 7, 10) |
            __gen_uint(values->early_z_disable, 6, 6) |
            __gen_uint(values->early_z_test_and_update_direction, 5, 5) |
            __gen_uint(values->double_buffer_in_non_ms_mode, 3, 3) |
            __gen_uint(values->multisample_mode_4x, 2, 2) |
            __gen_uint(values->maximum_bpp_of_all_render_targets, 0, 1);

   cl[ 7] = __gen_uint(values->pad, 4, 15) |
            __gen_uint(values->early_depth_stencil_clear, 3, 3) |
            __gen_uint(values->internal_depth_type, 7, 10) >> 8;

   cl[ 8] = __gen_uint(values->pad, 4, 15) >> 8;

}

#define V3D41_TILE_RENDERING_MODE_CFG_COMMON_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_TILE_RENDERING_MODE_CFG_COMMON_unpack(const uint8_t * restrict cl,
                                            struct V3D41_TILE_RENDERING_MODE_CFG_COMMON * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->pad = __gen_unpack_uint(cl, 60, 71);
   values->early_depth_stencil_clear = __gen_unpack_uint(cl, 59, 59);
   values->internal_depth_type = __gen_unpack_uint(cl, 55, 58);
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


#define V3D41_TILE_RENDERING_MODE_CFG_COLOR_opcode    121
#define V3D41_TILE_RENDERING_MODE_CFG_COLOR_header\
   .opcode                              =    121,  \
   .sub_id                              =      1

struct V3D41_TILE_RENDERING_MODE_CFG_COLOR {
   uint32_t                             opcode;
   uint32_t                             pad;
   enum V3D41_Render_Target_Clamp       render_target_3_clamp;
   enum V3D41_Internal_Type             render_target_3_internal_type;
   enum V3D41_Internal_BPP              render_target_3_internal_bpp;
   enum V3D41_Render_Target_Clamp       render_target_2_clamp;
   enum V3D41_Internal_Type             render_target_2_internal_type;
   enum V3D41_Internal_BPP              render_target_2_internal_bpp;
   enum V3D41_Render_Target_Clamp       render_target_1_clamp;
   enum V3D41_Internal_Type             render_target_1_internal_type;
   enum V3D41_Internal_BPP              render_target_1_internal_bpp;
   enum V3D41_Render_Target_Clamp       render_target_0_clamp;
   enum V3D41_Internal_Type             render_target_0_internal_type;
   enum V3D41_Internal_BPP              render_target_0_internal_bpp;
   uint32_t                             sub_id;
};

static inline void
V3D41_TILE_RENDERING_MODE_CFG_COLOR_pack(__gen_user_data *data, uint8_t * restrict cl,
                                         const struct V3D41_TILE_RENDERING_MODE_CFG_COLOR * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->render_target_0_internal_type, 6, 9) |
            __gen_uint(values->render_target_0_internal_bpp, 4, 5) |
            __gen_uint(values->sub_id, 0, 3);

   cl[ 2] = __gen_uint(values->render_target_1_internal_type, 6, 9) |
            __gen_uint(values->render_target_1_internal_bpp, 4, 5) |
            __gen_uint(values->render_target_0_clamp, 2, 3) |
            __gen_uint(values->render_target_0_internal_type, 6, 9) >> 8;

   cl[ 3] = __gen_uint(values->render_target_2_internal_type, 6, 9) |
            __gen_uint(values->render_target_2_internal_bpp, 4, 5) |
            __gen_uint(values->render_target_1_clamp, 2, 3) |
            __gen_uint(values->render_target_1_internal_type, 6, 9) >> 8;

   cl[ 4] = __gen_uint(values->render_target_3_internal_type, 6, 9) |
            __gen_uint(values->render_target_3_internal_bpp, 4, 5) |
            __gen_uint(values->render_target_2_clamp, 2, 3) |
            __gen_uint(values->render_target_2_internal_type, 6, 9) >> 8;

   cl[ 5] = __gen_uint(values->pad, 2, 29) |
            __gen_uint(values->render_target_3_clamp, 0, 1) |
            __gen_uint(values->render_target_3_internal_type, 6, 9) >> 8;

   cl[ 6] = __gen_uint(values->pad, 2, 29) >> 8;

   cl[ 7] = __gen_uint(values->pad, 2, 29) >> 16;

   cl[ 8] = __gen_uint(values->pad, 2, 29) >> 24;

}

#define V3D41_TILE_RENDERING_MODE_CFG_COLOR_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_TILE_RENDERING_MODE_CFG_COLOR_unpack(const uint8_t * restrict cl,
                                           struct V3D41_TILE_RENDERING_MODE_CFG_COLOR * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->pad = __gen_unpack_uint(cl, 42, 69);
   values->render_target_3_clamp = __gen_unpack_uint(cl, 40, 41);
   values->render_target_3_internal_type = __gen_unpack_uint(cl, 38, 41);
   values->render_target_3_internal_bpp = __gen_unpack_uint(cl, 36, 37);
   values->render_target_2_clamp = __gen_unpack_uint(cl, 34, 35);
   values->render_target_2_internal_type = __gen_unpack_uint(cl, 30, 33);
   values->render_target_2_internal_bpp = __gen_unpack_uint(cl, 28, 29);
   values->render_target_1_clamp = __gen_unpack_uint(cl, 26, 27);
   values->render_target_1_internal_type = __gen_unpack_uint(cl, 22, 25);
   values->render_target_1_internal_bpp = __gen_unpack_uint(cl, 20, 21);
   values->render_target_0_clamp = __gen_unpack_uint(cl, 18, 19);
   values->render_target_0_internal_type = __gen_unpack_uint(cl, 14, 17);
   values->render_target_0_internal_bpp = __gen_unpack_uint(cl, 12, 13);
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES_opcode    121
#define V3D41_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES_header\
   .opcode                              =    121,  \
   .sub_id                              =      2

struct V3D41_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES {
   uint32_t                             opcode;
   uint32_t                             unused;
   float                                z_clear_value;
   uint32_t                             stencil_clear_value;
   uint32_t                             sub_id;
};

static inline void
V3D41_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                   const struct V3D41_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->sub_id, 0, 3);

   cl[ 2] = __gen_uint(values->stencil_clear_value, 0, 7);


   memcpy(&cl[3], &values->z_clear_value, sizeof(values->z_clear_value));
   cl[ 7] = __gen_uint(values->unused, 0, 15);

   cl[ 8] = __gen_uint(values->unused, 0, 15) >> 8;

}

#define V3D41_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES_unpack(const uint8_t * restrict cl,
                                                     struct V3D41_TILE_RENDERING_MODE_CFG_ZS_CLEAR_VALUES * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->unused = __gen_unpack_uint(cl, 56, 71);
   values->z_clear_value = __gen_unpack_float(cl, 24, 55);
   values->stencil_clear_value = __gen_unpack_uint(cl, 16, 23);
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1_opcode    121
#define V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1_header\
   .opcode                              =    121,  \
   .sub_id                              =      3

struct V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1 {
   uint32_t                             opcode;
   uint32_t                             clear_color_next_24_bits;
   uint32_t                             clear_color_low_32_bits;
   uint32_t                             render_target_number;
   uint32_t                             sub_id;
};

static inline void
V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                      const struct V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1 * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->render_target_number, 4, 7) |
            __gen_uint(values->sub_id, 0, 3);


   memcpy(&cl[2], &values->clear_color_low_32_bits, sizeof(values->clear_color_low_32_bits));
   cl[ 6] = __gen_uint(values->clear_color_next_24_bits, 0, 23);

   cl[ 7] = __gen_uint(values->clear_color_next_24_bits, 0, 23) >> 8;

   cl[ 8] = __gen_uint(values->clear_color_next_24_bits, 0, 23) >> 16;

}

#define V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1_unpack(const uint8_t * restrict cl,
                                                        struct V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART1 * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->clear_color_next_24_bits = __gen_unpack_uint(cl, 48, 71);
   values->clear_color_low_32_bits = __gen_unpack_uint(cl, 16, 47);
   values->render_target_number = __gen_unpack_uint(cl, 12, 15);
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2_opcode    121
#define V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2_header\
   .opcode                              =    121,  \
   .sub_id                              =      4

struct V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2 {
   uint32_t                             opcode;
   uint32_t                             clear_color_mid_high_24_bits;
   uint32_t                             clear_color_mid_low_32_bits;
   uint32_t                             render_target_number;
   uint32_t                             sub_id;
};

static inline void
V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                      const struct V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2 * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->render_target_number, 4, 7) |
            __gen_uint(values->sub_id, 0, 3);


   memcpy(&cl[2], &values->clear_color_mid_low_32_bits, sizeof(values->clear_color_mid_low_32_bits));
   cl[ 6] = __gen_uint(values->clear_color_mid_high_24_bits, 0, 23);

   cl[ 7] = __gen_uint(values->clear_color_mid_high_24_bits, 0, 23) >> 8;

   cl[ 8] = __gen_uint(values->clear_color_mid_high_24_bits, 0, 23) >> 16;

}

#define V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2_unpack(const uint8_t * restrict cl,
                                                        struct V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART2 * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->clear_color_mid_high_24_bits = __gen_unpack_uint(cl, 48, 71);
   values->clear_color_mid_low_32_bits = __gen_unpack_uint(cl, 16, 47);
   values->render_target_number = __gen_unpack_uint(cl, 12, 15);
   values->sub_id = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3_opcode    121
#define V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3_header\
   .opcode                              =    121,  \
   .sub_id                              =      5

struct V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3 {
   uint32_t                             opcode;
   uint32_t                             pad;
   uint32_t                             uif_padded_height_in_uif_blocks;
   uint32_t                             raster_row_stride_or_image_height_in_pixels;
   uint32_t                             clear_color_high_16_bits;
   uint32_t                             render_target_number;
   uint32_t                             sub_id;
};

static inline void
V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                      const struct V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3 * restrict values)
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

#define V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3_unpack(const uint8_t * restrict cl,
                                                        struct V3D41_TILE_RENDERING_MODE_CFG_CLEAR_COLORS_PART3 * restrict values)
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


#define V3D41_TILE_COORDINATES_opcode        124
#define V3D41_TILE_COORDINATES_header           \
   .opcode                              =    124

struct V3D41_TILE_COORDINATES {
   uint32_t                             opcode;
   uint32_t                             tile_row_number;
   uint32_t                             tile_column_number;
};

static inline void
V3D41_TILE_COORDINATES_pack(__gen_user_data *data, uint8_t * restrict cl,
                            const struct V3D41_TILE_COORDINATES * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->tile_column_number, 0, 11);

   cl[ 2] = __gen_uint(values->tile_row_number, 4, 15) |
            __gen_uint(values->tile_column_number, 0, 11) >> 8;

   cl[ 3] = __gen_uint(values->tile_row_number, 4, 15) >> 8;

}

#define V3D41_TILE_COORDINATES_length          4
#ifdef __gen_unpack_address
static inline void
V3D41_TILE_COORDINATES_unpack(const uint8_t * restrict cl,
                              struct V3D41_TILE_COORDINATES * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->tile_row_number = __gen_unpack_uint(cl, 20, 31);
   values->tile_column_number = __gen_unpack_uint(cl, 8, 19);
}
#endif


#define V3D41_MULTICORE_RENDERING_SUPERTILE_CFG_opcode    122
#define V3D41_MULTICORE_RENDERING_SUPERTILE_CFG_header\
   .opcode                              =    122

struct V3D41_MULTICORE_RENDERING_SUPERTILE_CFG {
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
V3D41_MULTICORE_RENDERING_SUPERTILE_CFG_pack(__gen_user_data *data, uint8_t * restrict cl,
                                             const struct V3D41_MULTICORE_RENDERING_SUPERTILE_CFG * restrict values)
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

#define V3D41_MULTICORE_RENDERING_SUPERTILE_CFG_length      9
#ifdef __gen_unpack_address
static inline void
V3D41_MULTICORE_RENDERING_SUPERTILE_CFG_unpack(const uint8_t * restrict cl,
                                               struct V3D41_MULTICORE_RENDERING_SUPERTILE_CFG * restrict values)
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


#define V3D41_MULTICORE_RENDERING_TILE_LIST_SET_BASE_opcode    123
#define V3D41_MULTICORE_RENDERING_TILE_LIST_SET_BASE_header\
   .opcode                              =    123

struct V3D41_MULTICORE_RENDERING_TILE_LIST_SET_BASE {
   uint32_t                             opcode;
   __gen_address_type                   address;
   uint32_t                             tile_list_set_number;
};

static inline void
V3D41_MULTICORE_RENDERING_TILE_LIST_SET_BASE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                                  const struct V3D41_MULTICORE_RENDERING_TILE_LIST_SET_BASE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   __gen_emit_reloc(data, &values->address);
   cl[ 1] = __gen_address_offset(&values->address) |
            __gen_uint(values->tile_list_set_number, 0, 3);

   cl[ 2] = __gen_address_offset(&values->address) >> 8;

   cl[ 3] = __gen_address_offset(&values->address) >> 16;

   cl[ 4] = __gen_address_offset(&values->address) >> 24;

}

#define V3D41_MULTICORE_RENDERING_TILE_LIST_SET_BASE_length      5
#ifdef __gen_unpack_address
static inline void
V3D41_MULTICORE_RENDERING_TILE_LIST_SET_BASE_unpack(const uint8_t * restrict cl,
                                                    struct V3D41_MULTICORE_RENDERING_TILE_LIST_SET_BASE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->address = __gen_unpack_address(cl, 14, 39);
   values->tile_list_set_number = __gen_unpack_uint(cl, 8, 11);
}
#endif


#define V3D41_TILE_COORDINATES_IMPLICIT_opcode    125
#define V3D41_TILE_COORDINATES_IMPLICIT_header  \
   .opcode                              =    125

struct V3D41_TILE_COORDINATES_IMPLICIT {
   uint32_t                             opcode;
};

static inline void
V3D41_TILE_COORDINATES_IMPLICIT_pack(__gen_user_data *data, uint8_t * restrict cl,
                                     const struct V3D41_TILE_COORDINATES_IMPLICIT * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

}

#define V3D41_TILE_COORDINATES_IMPLICIT_length      1
#ifdef __gen_unpack_address
static inline void
V3D41_TILE_COORDINATES_IMPLICIT_unpack(const uint8_t * restrict cl,
                                       struct V3D41_TILE_COORDINATES_IMPLICIT * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
}
#endif


#define V3D41_TILE_LIST_INITIAL_BLOCK_SIZE_opcode    126
#define V3D41_TILE_LIST_INITIAL_BLOCK_SIZE_header\
   .opcode                              =    126

struct V3D41_TILE_LIST_INITIAL_BLOCK_SIZE {
   uint32_t                             opcode;
   bool                                 use_auto_chained_tile_lists;
   uint32_t                             size_of_first_block_in_chained_tile_lists;
#define TILE_ALLOCATION_BLOCK_SIZE_64B           0
#define TILE_ALLOCATION_BLOCK_SIZE_128B          1
#define TILE_ALLOCATION_BLOCK_SIZE_256B          2
};

static inline void
V3D41_TILE_LIST_INITIAL_BLOCK_SIZE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D41_TILE_LIST_INITIAL_BLOCK_SIZE * restrict values)
{
   cl[ 0] = __gen_uint(values->opcode, 0, 7);

   cl[ 1] = __gen_uint(values->use_auto_chained_tile_lists, 2, 2) |
            __gen_uint(values->size_of_first_block_in_chained_tile_lists, 0, 1);

}

#define V3D41_TILE_LIST_INITIAL_BLOCK_SIZE_length      2
#ifdef __gen_unpack_address
static inline void
V3D41_TILE_LIST_INITIAL_BLOCK_SIZE_unpack(const uint8_t * restrict cl,
                                          struct V3D41_TILE_LIST_INITIAL_BLOCK_SIZE * restrict values)
{
   values->opcode = __gen_unpack_uint(cl, 0, 7);
   values->use_auto_chained_tile_lists = __gen_unpack_uint(cl, 10, 10);
   values->size_of_first_block_in_chained_tile_lists = __gen_unpack_uint(cl, 8, 9);
}
#endif


#define V3D41_GL_SHADER_STATE_RECORD_header     \


struct V3D41_GL_SHADER_STATE_RECORD {
   bool                                 point_size_in_shaded_vertex_data;
   bool                                 enable_clipping;
   bool                                 vertex_id_read_by_coordinate_shader;
   bool                                 instance_id_read_by_coordinate_shader;
   bool                                 base_instance_id_read_by_coordinate_shader;
   bool                                 vertex_id_read_by_vertex_shader;
   bool                                 instance_id_read_by_vertex_shader;
   bool                                 base_instance_id_read_by_vertex_shader;
   bool                                 fragment_shader_does_z_writes;
   bool                                 turn_off_early_z_test;
   bool                                 coordinate_shader_has_separate_input_and_output_vpm_blocks;
   bool                                 vertex_shader_has_separate_input_and_output_vpm_blocks;
   bool                                 fragment_shader_uses_real_pixel_centre_w_in_addition_to_centroid_w2;
   bool                                 enable_sample_rate_shading;
   bool                                 any_shader_reads_hardware_written_primitive_id;
   bool                                 insert_primitive_id_as_first_varying_to_fragment_shader;
   bool                                 turn_off_scoreboard;
   bool                                 do_scoreboard_wait_on_first_thread_switch;
   bool                                 disable_implicit_point_line_varyings;
   bool                                 no_prim_pack;
   uint32_t                             number_of_varyings_in_fragment_shader;
   uint32_t                             coordinate_shader_output_vpm_segment_size;
   uint32_t                             min_coord_shader_output_segments_required_in_play_in_addition_to_vcm_cache_size;
   uint32_t                             coordinate_shader_input_vpm_segment_size;
   uint32_t                             min_coord_shader_input_segments_required_in_play;
   uint32_t                             vertex_shader_output_vpm_segment_size;
   uint32_t                             min_vertex_shader_output_segments_required_in_play_in_addition_to_vcm_cache_size;
   uint32_t                             vertex_shader_input_vpm_segment_size;
   uint32_t                             min_vertex_shader_input_segments_required_in_play;
   __gen_address_type                   address_of_default_attribute_values;
   __gen_address_type                   fragment_shader_code_address;
   bool                                 fragment_shader_4_way_threadable;
   bool                                 fragment_shader_start_in_final_thread_section;
   bool                                 fragment_shader_propagate_nans;
   __gen_address_type                   fragment_shader_uniforms_address;
   __gen_address_type                   vertex_shader_code_address;
   bool                                 vertex_shader_4_way_threadable;
   bool                                 vertex_shader_start_in_final_thread_section;
   bool                                 vertex_shader_propagate_nans;
   __gen_address_type                   vertex_shader_uniforms_address;
   __gen_address_type                   coordinate_shader_code_address;
   bool                                 coordinate_shader_4_way_threadable;
   bool                                 coordinate_shader_start_in_final_thread_section;
   bool                                 coordinate_shader_propagate_nans;
   __gen_address_type                   coordinate_shader_uniforms_address;
};

static inline void
V3D41_GL_SHADER_STATE_RECORD_pack(__gen_user_data *data, uint8_t * restrict cl,
                                  const struct V3D41_GL_SHADER_STATE_RECORD * restrict values)
{
   assert(values->min_coord_shader_input_segments_required_in_play >= 1);
   assert(values->min_vertex_shader_input_segments_required_in_play >= 1);
   cl[ 0] = __gen_uint(values->point_size_in_shaded_vertex_data, 0, 0) |
            __gen_uint(values->enable_clipping, 1, 1) |
            __gen_uint(values->vertex_id_read_by_coordinate_shader, 2, 2) |
            __gen_uint(values->instance_id_read_by_coordinate_shader, 3, 3) |
            __gen_uint(values->base_instance_id_read_by_coordinate_shader, 4, 4) |
            __gen_uint(values->vertex_id_read_by_vertex_shader, 5, 5) |
            __gen_uint(values->instance_id_read_by_vertex_shader, 6, 6) |
            __gen_uint(values->base_instance_id_read_by_vertex_shader, 7, 7);

   cl[ 1] = __gen_uint(values->fragment_shader_does_z_writes, 0, 0) |
            __gen_uint(values->turn_off_early_z_test, 1, 1) |
            __gen_uint(values->coordinate_shader_has_separate_input_and_output_vpm_blocks, 2, 2) |
            __gen_uint(values->vertex_shader_has_separate_input_and_output_vpm_blocks, 3, 3) |
            __gen_uint(values->fragment_shader_uses_real_pixel_centre_w_in_addition_to_centroid_w2, 4, 4) |
            __gen_uint(values->enable_sample_rate_shading, 5, 5) |
            __gen_uint(values->any_shader_reads_hardware_written_primitive_id, 6, 6) |
            __gen_uint(values->insert_primitive_id_as_first_varying_to_fragment_shader, 7, 7);

   cl[ 2] = __gen_uint(values->turn_off_scoreboard, 0, 0) |
            __gen_uint(values->do_scoreboard_wait_on_first_thread_switch, 1, 1) |
            __gen_uint(values->disable_implicit_point_line_varyings, 2, 2) |
            __gen_uint(values->no_prim_pack, 3, 3);

   cl[ 3] = __gen_uint(values->number_of_varyings_in_fragment_shader, 0, 7);

   cl[ 4] = __gen_uint(values->coordinate_shader_output_vpm_segment_size, 0, 3) |
            __gen_uint(values->min_coord_shader_output_segments_required_in_play_in_addition_to_vcm_cache_size, 4, 7);

   cl[ 5] = __gen_uint(values->coordinate_shader_input_vpm_segment_size, 0, 3) |
            __gen_uint(values->min_coord_shader_input_segments_required_in_play - 1, 4, 7);

   cl[ 6] = __gen_uint(values->vertex_shader_output_vpm_segment_size, 0, 3) |
            __gen_uint(values->min_vertex_shader_output_segments_required_in_play_in_addition_to_vcm_cache_size, 4, 7);

   cl[ 7] = __gen_uint(values->vertex_shader_input_vpm_segment_size, 0, 3) |
            __gen_uint(values->min_vertex_shader_input_segments_required_in_play - 1, 4, 7);

   __gen_emit_reloc(data, &values->address_of_default_attribute_values);
   cl[ 8] = __gen_address_offset(&values->address_of_default_attribute_values);

   cl[ 9] = __gen_address_offset(&values->address_of_default_attribute_values) >> 8;

   cl[10] = __gen_address_offset(&values->address_of_default_attribute_values) >> 16;

   cl[11] = __gen_address_offset(&values->address_of_default_attribute_values) >> 24;

   __gen_emit_reloc(data, &values->fragment_shader_code_address);
   cl[12] = __gen_address_offset(&values->fragment_shader_code_address) |
            __gen_uint(values->fragment_shader_4_way_threadable, 0, 0) |
            __gen_uint(values->fragment_shader_start_in_final_thread_section, 1, 1) |
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
            __gen_uint(values->vertex_shader_4_way_threadable, 0, 0) |
            __gen_uint(values->vertex_shader_start_in_final_thread_section, 1, 1) |
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
            __gen_uint(values->coordinate_shader_4_way_threadable, 0, 0) |
            __gen_uint(values->coordinate_shader_start_in_final_thread_section, 1, 1) |
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

#define V3D41_GL_SHADER_STATE_RECORD_length     36
#ifdef __gen_unpack_address
static inline void
V3D41_GL_SHADER_STATE_RECORD_unpack(const uint8_t * restrict cl,
                                    struct V3D41_GL_SHADER_STATE_RECORD * restrict values)
{
   values->point_size_in_shaded_vertex_data = __gen_unpack_uint(cl, 0, 0);
   values->enable_clipping = __gen_unpack_uint(cl, 1, 1);
   values->vertex_id_read_by_coordinate_shader = __gen_unpack_uint(cl, 2, 2);
   values->instance_id_read_by_coordinate_shader = __gen_unpack_uint(cl, 3, 3);
   values->base_instance_id_read_by_coordinate_shader = __gen_unpack_uint(cl, 4, 4);
   values->vertex_id_read_by_vertex_shader = __gen_unpack_uint(cl, 5, 5);
   values->instance_id_read_by_vertex_shader = __gen_unpack_uint(cl, 6, 6);
   values->base_instance_id_read_by_vertex_shader = __gen_unpack_uint(cl, 7, 7);
   values->fragment_shader_does_z_writes = __gen_unpack_uint(cl, 8, 8);
   values->turn_off_early_z_test = __gen_unpack_uint(cl, 9, 9);
   values->coordinate_shader_has_separate_input_and_output_vpm_blocks = __gen_unpack_uint(cl, 10, 10);
   values->vertex_shader_has_separate_input_and_output_vpm_blocks = __gen_unpack_uint(cl, 11, 11);
   values->fragment_shader_uses_real_pixel_centre_w_in_addition_to_centroid_w2 = __gen_unpack_uint(cl, 12, 12);
   values->enable_sample_rate_shading = __gen_unpack_uint(cl, 13, 13);
   values->any_shader_reads_hardware_written_primitive_id = __gen_unpack_uint(cl, 14, 14);
   values->insert_primitive_id_as_first_varying_to_fragment_shader = __gen_unpack_uint(cl, 15, 15);
   values->turn_off_scoreboard = __gen_unpack_uint(cl, 16, 16);
   values->do_scoreboard_wait_on_first_thread_switch = __gen_unpack_uint(cl, 17, 17);
   values->disable_implicit_point_line_varyings = __gen_unpack_uint(cl, 18, 18);
   values->no_prim_pack = __gen_unpack_uint(cl, 19, 19);
   values->number_of_varyings_in_fragment_shader = __gen_unpack_uint(cl, 24, 31);
   values->coordinate_shader_output_vpm_segment_size = __gen_unpack_uint(cl, 32, 35);
   values->min_coord_shader_output_segments_required_in_play_in_addition_to_vcm_cache_size = __gen_unpack_uint(cl, 36, 39);
   values->coordinate_shader_input_vpm_segment_size = __gen_unpack_uint(cl, 40, 43);
   values->min_coord_shader_input_segments_required_in_play = __gen_unpack_uint(cl, 44, 47) + 1;
   values->vertex_shader_output_vpm_segment_size = __gen_unpack_uint(cl, 48, 51);
   values->min_vertex_shader_output_segments_required_in_play_in_addition_to_vcm_cache_size = __gen_unpack_uint(cl, 52, 55);
   values->vertex_shader_input_vpm_segment_size = __gen_unpack_uint(cl, 56, 59);
   values->min_vertex_shader_input_segments_required_in_play = __gen_unpack_uint(cl, 60, 63) + 1;
   values->address_of_default_attribute_values = __gen_unpack_address(cl, 64, 95);
   values->fragment_shader_code_address = __gen_unpack_address(cl, 99, 127);
   values->fragment_shader_4_way_threadable = __gen_unpack_uint(cl, 96, 96);
   values->fragment_shader_start_in_final_thread_section = __gen_unpack_uint(cl, 97, 97);
   values->fragment_shader_propagate_nans = __gen_unpack_uint(cl, 98, 98);
   values->fragment_shader_uniforms_address = __gen_unpack_address(cl, 128, 159);
   values->vertex_shader_code_address = __gen_unpack_address(cl, 163, 191);
   values->vertex_shader_4_way_threadable = __gen_unpack_uint(cl, 160, 160);
   values->vertex_shader_start_in_final_thread_section = __gen_unpack_uint(cl, 161, 161);
   values->vertex_shader_propagate_nans = __gen_unpack_uint(cl, 162, 162);
   values->vertex_shader_uniforms_address = __gen_unpack_address(cl, 192, 223);
   values->coordinate_shader_code_address = __gen_unpack_address(cl, 227, 255);
   values->coordinate_shader_4_way_threadable = __gen_unpack_uint(cl, 224, 224);
   values->coordinate_shader_start_in_final_thread_section = __gen_unpack_uint(cl, 225, 225);
   values->coordinate_shader_propagate_nans = __gen_unpack_uint(cl, 226, 226);
   values->coordinate_shader_uniforms_address = __gen_unpack_address(cl, 256, 287);
}
#endif


#define V3D41_GEOMETRY_SHADER_STATE_RECORD_header\


struct V3D41_GEOMETRY_SHADER_STATE_RECORD {
   __gen_address_type                   geometry_bin_mode_shader_code_address;
   bool                                 _4_way_threadable;
   bool                                 start_in_final_thread_section;
   bool                                 propagate_nans;
   __gen_address_type                   geometry_bin_mode_shader_uniforms_address;
   __gen_address_type                   geometry_render_mode_shader_code_address;
   __gen_address_type                   geometry_render_mode_shader_uniforms_address;
};

static inline void
V3D41_GEOMETRY_SHADER_STATE_RECORD_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D41_GEOMETRY_SHADER_STATE_RECORD * restrict values)
{
   __gen_emit_reloc(data, &values->geometry_bin_mode_shader_code_address);
   cl[ 0] = __gen_address_offset(&values->geometry_bin_mode_shader_code_address) |
            __gen_uint(values->_4_way_threadable, 0, 0) |
            __gen_uint(values->start_in_final_thread_section, 1, 1) |
            __gen_uint(values->propagate_nans, 2, 2);

   cl[ 1] = __gen_address_offset(&values->geometry_bin_mode_shader_code_address) >> 8;

   cl[ 2] = __gen_address_offset(&values->geometry_bin_mode_shader_code_address) >> 16;

   cl[ 3] = __gen_address_offset(&values->geometry_bin_mode_shader_code_address) >> 24;

   __gen_emit_reloc(data, &values->geometry_bin_mode_shader_uniforms_address);
   cl[ 4] = __gen_address_offset(&values->geometry_bin_mode_shader_uniforms_address);

   cl[ 5] = __gen_address_offset(&values->geometry_bin_mode_shader_uniforms_address) >> 8;

   cl[ 6] = __gen_address_offset(&values->geometry_bin_mode_shader_uniforms_address) >> 16;

   cl[ 7] = __gen_address_offset(&values->geometry_bin_mode_shader_uniforms_address) >> 24;

   __gen_emit_reloc(data, &values->geometry_render_mode_shader_code_address);
   cl[ 8] = __gen_address_offset(&values->geometry_render_mode_shader_code_address);

   cl[ 9] = __gen_address_offset(&values->geometry_render_mode_shader_code_address) >> 8;

   cl[10] = __gen_address_offset(&values->geometry_render_mode_shader_code_address) >> 16;

   cl[11] = __gen_address_offset(&values->geometry_render_mode_shader_code_address) >> 24;

   __gen_emit_reloc(data, &values->geometry_render_mode_shader_uniforms_address);
   cl[12] = __gen_address_offset(&values->geometry_render_mode_shader_uniforms_address);

   cl[13] = __gen_address_offset(&values->geometry_render_mode_shader_uniforms_address) >> 8;

   cl[14] = __gen_address_offset(&values->geometry_render_mode_shader_uniforms_address) >> 16;

   cl[15] = __gen_address_offset(&values->geometry_render_mode_shader_uniforms_address) >> 24;

}

#define V3D41_GEOMETRY_SHADER_STATE_RECORD_length     16
#ifdef __gen_unpack_address
static inline void
V3D41_GEOMETRY_SHADER_STATE_RECORD_unpack(const uint8_t * restrict cl,
                                          struct V3D41_GEOMETRY_SHADER_STATE_RECORD * restrict values)
{
   values->geometry_bin_mode_shader_code_address = __gen_unpack_address(cl, 0, 31);
   values->_4_way_threadable = __gen_unpack_uint(cl, 0, 0);
   values->start_in_final_thread_section = __gen_unpack_uint(cl, 1, 1);
   values->propagate_nans = __gen_unpack_uint(cl, 2, 2);
   values->geometry_bin_mode_shader_uniforms_address = __gen_unpack_address(cl, 32, 63);
   values->geometry_render_mode_shader_code_address = __gen_unpack_address(cl, 64, 95);
   values->geometry_render_mode_shader_uniforms_address = __gen_unpack_address(cl, 96, 127);
}
#endif


#define V3D41_TESSELLATION_SHADER_STATE_RECORD_header\


struct V3D41_TESSELLATION_SHADER_STATE_RECORD {
   __gen_address_type                   tessellation_bin_mode_control_shader_code_address;
   __gen_address_type                   tessellation_bin_mode_control_shader_uniforms_address;
   __gen_address_type                   tessellation_render_mode_control_shader_code_address;
   __gen_address_type                   tessellation_render_mode_control_shader_uniforms_address;
   __gen_address_type                   tessellation_bin_mode_evaluation_shader_code_address;
   __gen_address_type                   tessellation_bin_mode_evaluation_shader_uniforms_address;
   __gen_address_type                   tessellation_render_mode_evaluation_shader_code_address;
   __gen_address_type                   tessellation_render_mode_evaluation_shader_uniforms_address;
};

static inline void
V3D41_TESSELLATION_SHADER_STATE_RECORD_pack(__gen_user_data *data, uint8_t * restrict cl,
                                            const struct V3D41_TESSELLATION_SHADER_STATE_RECORD * restrict values)
{
   __gen_emit_reloc(data, &values->tessellation_bin_mode_control_shader_code_address);
   cl[ 0] = __gen_address_offset(&values->tessellation_bin_mode_control_shader_code_address);

   cl[ 1] = __gen_address_offset(&values->tessellation_bin_mode_control_shader_code_address) >> 8;

   cl[ 2] = __gen_address_offset(&values->tessellation_bin_mode_control_shader_code_address) >> 16;

   cl[ 3] = __gen_address_offset(&values->tessellation_bin_mode_control_shader_code_address) >> 24;

   __gen_emit_reloc(data, &values->tessellation_bin_mode_control_shader_uniforms_address);
   cl[ 4] = __gen_address_offset(&values->tessellation_bin_mode_control_shader_uniforms_address);

   cl[ 5] = __gen_address_offset(&values->tessellation_bin_mode_control_shader_uniforms_address) >> 8;

   cl[ 6] = __gen_address_offset(&values->tessellation_bin_mode_control_shader_uniforms_address) >> 16;

   cl[ 7] = __gen_address_offset(&values->tessellation_bin_mode_control_shader_uniforms_address) >> 24;

   __gen_emit_reloc(data, &values->tessellation_render_mode_control_shader_code_address);
   cl[ 8] = __gen_address_offset(&values->tessellation_render_mode_control_shader_code_address);

   cl[ 9] = __gen_address_offset(&values->tessellation_render_mode_control_shader_code_address) >> 8;

   cl[10] = __gen_address_offset(&values->tessellation_render_mode_control_shader_code_address) >> 16;

   cl[11] = __gen_address_offset(&values->tessellation_render_mode_control_shader_code_address) >> 24;

   __gen_emit_reloc(data, &values->tessellation_render_mode_control_shader_uniforms_address);
   cl[12] = __gen_address_offset(&values->tessellation_render_mode_control_shader_uniforms_address);

   cl[13] = __gen_address_offset(&values->tessellation_render_mode_control_shader_uniforms_address) >> 8;

   cl[14] = __gen_address_offset(&values->tessellation_render_mode_control_shader_uniforms_address) >> 16;

   cl[15] = __gen_address_offset(&values->tessellation_render_mode_control_shader_uniforms_address) >> 24;

   __gen_emit_reloc(data, &values->tessellation_bin_mode_evaluation_shader_code_address);
   cl[16] = __gen_address_offset(&values->tessellation_bin_mode_evaluation_shader_code_address);

   cl[17] = __gen_address_offset(&values->tessellation_bin_mode_evaluation_shader_code_address) >> 8;

   cl[18] = __gen_address_offset(&values->tessellation_bin_mode_evaluation_shader_code_address) >> 16;

   cl[19] = __gen_address_offset(&values->tessellation_bin_mode_evaluation_shader_code_address) >> 24;

   __gen_emit_reloc(data, &values->tessellation_bin_mode_evaluation_shader_uniforms_address);
   cl[20] = __gen_address_offset(&values->tessellation_bin_mode_evaluation_shader_uniforms_address);

   cl[21] = __gen_address_offset(&values->tessellation_bin_mode_evaluation_shader_uniforms_address) >> 8;

   cl[22] = __gen_address_offset(&values->tessellation_bin_mode_evaluation_shader_uniforms_address) >> 16;

   cl[23] = __gen_address_offset(&values->tessellation_bin_mode_evaluation_shader_uniforms_address) >> 24;

   __gen_emit_reloc(data, &values->tessellation_render_mode_evaluation_shader_code_address);
   cl[24] = __gen_address_offset(&values->tessellation_render_mode_evaluation_shader_code_address);

   cl[25] = __gen_address_offset(&values->tessellation_render_mode_evaluation_shader_code_address) >> 8;

   cl[26] = __gen_address_offset(&values->tessellation_render_mode_evaluation_shader_code_address) >> 16;

   cl[27] = __gen_address_offset(&values->tessellation_render_mode_evaluation_shader_code_address) >> 24;

   __gen_emit_reloc(data, &values->tessellation_render_mode_evaluation_shader_uniforms_address);
   cl[28] = __gen_address_offset(&values->tessellation_render_mode_evaluation_shader_uniforms_address);

   cl[29] = __gen_address_offset(&values->tessellation_render_mode_evaluation_shader_uniforms_address) >> 8;

   cl[30] = __gen_address_offset(&values->tessellation_render_mode_evaluation_shader_uniforms_address) >> 16;

   cl[31] = __gen_address_offset(&values->tessellation_render_mode_evaluation_shader_uniforms_address) >> 24;

}

#define V3D41_TESSELLATION_SHADER_STATE_RECORD_length     32
#ifdef __gen_unpack_address
static inline void
V3D41_TESSELLATION_SHADER_STATE_RECORD_unpack(const uint8_t * restrict cl,
                                              struct V3D41_TESSELLATION_SHADER_STATE_RECORD * restrict values)
{
   values->tessellation_bin_mode_control_shader_code_address = __gen_unpack_address(cl, 0, 31);
   values->tessellation_bin_mode_control_shader_uniforms_address = __gen_unpack_address(cl, 32, 63);
   values->tessellation_render_mode_control_shader_code_address = __gen_unpack_address(cl, 64, 95);
   values->tessellation_render_mode_control_shader_uniforms_address = __gen_unpack_address(cl, 96, 127);
   values->tessellation_bin_mode_evaluation_shader_code_address = __gen_unpack_address(cl, 128, 159);
   values->tessellation_bin_mode_evaluation_shader_uniforms_address = __gen_unpack_address(cl, 160, 191);
   values->tessellation_render_mode_evaluation_shader_code_address = __gen_unpack_address(cl, 192, 223);
   values->tessellation_render_mode_evaluation_shader_uniforms_address = __gen_unpack_address(cl, 224, 255);
}
#endif


#define V3D41_GL_SHADER_STATE_ATTRIBUTE_RECORD_header\


struct V3D41_GL_SHADER_STATE_ATTRIBUTE_RECORD {
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
   uint32_t                             maximum_index;
};

static inline void
V3D41_GL_SHADER_STATE_ATTRIBUTE_RECORD_pack(__gen_user_data *data, uint8_t * restrict cl,
                                            const struct V3D41_GL_SHADER_STATE_ATTRIBUTE_RECORD * restrict values)
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

   memcpy(&cl[12], &values->maximum_index, sizeof(values->maximum_index));
}

#define V3D41_GL_SHADER_STATE_ATTRIBUTE_RECORD_length     16
#ifdef __gen_unpack_address
static inline void
V3D41_GL_SHADER_STATE_ATTRIBUTE_RECORD_unpack(const uint8_t * restrict cl,
                                              struct V3D41_GL_SHADER_STATE_ATTRIBUTE_RECORD * restrict values)
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
   values->maximum_index = __gen_unpack_uint(cl, 96, 127);
}
#endif


#define V3D41_VPM_GENERIC_BLOCK_WRITE_SETUP_header\
   .id                                  =      0,  \
   .id0                                 =      0

struct V3D41_VPM_GENERIC_BLOCK_WRITE_SETUP {
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
V3D41_VPM_GENERIC_BLOCK_WRITE_SETUP_pack(__gen_user_data *data, uint8_t * restrict cl,
                                         const struct V3D41_VPM_GENERIC_BLOCK_WRITE_SETUP * restrict values)
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

#define V3D41_VPM_GENERIC_BLOCK_WRITE_SETUP_length      4
#ifdef __gen_unpack_address
static inline void
V3D41_VPM_GENERIC_BLOCK_WRITE_SETUP_unpack(const uint8_t * restrict cl,
                                           struct V3D41_VPM_GENERIC_BLOCK_WRITE_SETUP * restrict values)
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


#define V3D41_VPM_GENERIC_BLOCK_READ_SETUP_header\
   .id                                  =      1

struct V3D41_VPM_GENERIC_BLOCK_READ_SETUP {
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
V3D41_VPM_GENERIC_BLOCK_READ_SETUP_pack(__gen_user_data *data, uint8_t * restrict cl,
                                        const struct V3D41_VPM_GENERIC_BLOCK_READ_SETUP * restrict values)
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

#define V3D41_VPM_GENERIC_BLOCK_READ_SETUP_length      4
#ifdef __gen_unpack_address
static inline void
V3D41_VPM_GENERIC_BLOCK_READ_SETUP_unpack(const uint8_t * restrict cl,
                                          struct V3D41_VPM_GENERIC_BLOCK_READ_SETUP * restrict values)
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


#define V3D41_TMU_CONFIG_PARAMETER_0_header     \


struct V3D41_TMU_CONFIG_PARAMETER_0 {
   __gen_address_type                   texture_state_address;
   uint32_t                             return_words_of_texture_data;
};

static inline void
V3D41_TMU_CONFIG_PARAMETER_0_pack(__gen_user_data *data, uint8_t * restrict cl,
                                  const struct V3D41_TMU_CONFIG_PARAMETER_0 * restrict values)
{
   __gen_emit_reloc(data, &values->texture_state_address);
   cl[ 0] = __gen_address_offset(&values->texture_state_address) |
            __gen_uint(values->return_words_of_texture_data, 0, 3);

   cl[ 1] = __gen_address_offset(&values->texture_state_address) >> 8;

   cl[ 2] = __gen_address_offset(&values->texture_state_address) >> 16;

   cl[ 3] = __gen_address_offset(&values->texture_state_address) >> 24;

}

#define V3D41_TMU_CONFIG_PARAMETER_0_length      4
#ifdef __gen_unpack_address
static inline void
V3D41_TMU_CONFIG_PARAMETER_0_unpack(const uint8_t * restrict cl,
                                    struct V3D41_TMU_CONFIG_PARAMETER_0 * restrict values)
{
   values->texture_state_address = __gen_unpack_address(cl, 0, 31);
   values->return_words_of_texture_data = __gen_unpack_uint(cl, 0, 3);
}
#endif


#define V3D41_TMU_CONFIG_PARAMETER_1_header     \


struct V3D41_TMU_CONFIG_PARAMETER_1 {
   __gen_address_type                   sampler_state_address;
   bool                                 per_pixel_mask_enable;
   bool                                 unnormalized_coordinates;
   bool                                 output_type_32_bit;
};

static inline void
V3D41_TMU_CONFIG_PARAMETER_1_pack(__gen_user_data *data, uint8_t * restrict cl,
                                  const struct V3D41_TMU_CONFIG_PARAMETER_1 * restrict values)
{
   __gen_emit_reloc(data, &values->sampler_state_address);
   cl[ 0] = __gen_address_offset(&values->sampler_state_address) |
            __gen_uint(values->per_pixel_mask_enable, 2, 2) |
            __gen_uint(values->unnormalized_coordinates, 1, 1) |
            __gen_uint(values->output_type_32_bit, 0, 0);

   cl[ 1] = __gen_address_offset(&values->sampler_state_address) >> 8;

   cl[ 2] = __gen_address_offset(&values->sampler_state_address) >> 16;

   cl[ 3] = __gen_address_offset(&values->sampler_state_address) >> 24;

}

#define V3D41_TMU_CONFIG_PARAMETER_1_length      4
#ifdef __gen_unpack_address
static inline void
V3D41_TMU_CONFIG_PARAMETER_1_unpack(const uint8_t * restrict cl,
                                    struct V3D41_TMU_CONFIG_PARAMETER_1 * restrict values)
{
   values->sampler_state_address = __gen_unpack_address(cl, 0, 31);
   values->per_pixel_mask_enable = __gen_unpack_uint(cl, 2, 2);
   values->unnormalized_coordinates = __gen_unpack_uint(cl, 1, 1);
   values->output_type_32_bit = __gen_unpack_uint(cl, 0, 0);
}
#endif


#define V3D41_TMU_CONFIG_PARAMETER_2_header     \


struct V3D41_TMU_CONFIG_PARAMETER_2 {
   uint32_t                             pad;
   enum V3D41_TMU_Op                    op;
   int32_t                              offset_r;
   int32_t                              offset_t;
   int32_t                              offset_s;
   bool                                 gather_mode;
   uint32_t                             gather_component;
   bool                                 coefficient_mode;
   uint32_t                             sample_number;
   bool                                 disable_autolod;
   bool                                 offset_format_8;
};

static inline void
V3D41_TMU_CONFIG_PARAMETER_2_pack(__gen_user_data *data, uint8_t * restrict cl,
                                  const struct V3D41_TMU_CONFIG_PARAMETER_2 * restrict values)
{
   cl[ 0] = __gen_uint(values->gather_mode, 7, 7) |
            __gen_uint(values->gather_component, 5, 6) |
            __gen_uint(values->coefficient_mode, 4, 4) |
            __gen_uint(values->sample_number, 2, 3) |
            __gen_uint(values->disable_autolod, 1, 1) |
            __gen_uint(values->offset_format_8, 0, 0);

   cl[ 1] = __gen_uint(values->pad, 0, 23) |
            __gen_sint(values->offset_t, 4, 7) |
            __gen_sint(values->offset_s, 0, 3);

   cl[ 2] = __gen_uint(values->pad, 0, 23) >> 8 |
            __gen_uint(values->op, 4, 7) |
            __gen_sint(values->offset_r, 0, 3);

   cl[ 3] = __gen_uint(values->pad, 0, 23) >> 16;

}

#define V3D41_TMU_CONFIG_PARAMETER_2_length      4
#ifdef __gen_unpack_address
static inline void
V3D41_TMU_CONFIG_PARAMETER_2_unpack(const uint8_t * restrict cl,
                                    struct V3D41_TMU_CONFIG_PARAMETER_2 * restrict values)
{
   values->pad = __gen_unpack_uint(cl, 8, 31);
   values->op = __gen_unpack_uint(cl, 20, 23);
   values->offset_r = __gen_unpack_sint(cl, 16, 19);
   values->offset_t = __gen_unpack_sint(cl, 12, 15);
   values->offset_s = __gen_unpack_sint(cl, 8, 11);
   values->gather_mode = __gen_unpack_uint(cl, 7, 7);
   values->gather_component = __gen_unpack_uint(cl, 5, 6);
   values->coefficient_mode = __gen_unpack_uint(cl, 4, 4);
   values->sample_number = __gen_unpack_uint(cl, 2, 3);
   values->disable_autolod = __gen_unpack_uint(cl, 1, 1);
   values->offset_format_8 = __gen_unpack_uint(cl, 0, 0);
}
#endif


#define V3D41_TEXTURE_SHADER_STATE_header       \


struct V3D41_TEXTURE_SHADER_STATE {
   uint64_t                             pad;
   bool                                 uif_xor_disable;
   bool                                 level_0_is_strictly_uif;
   bool                                 level_0_xor_enable;
   uint32_t                             level_0_ub_pad;
   uint32_t                             base_level;
   uint32_t                             max_level;
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
   bool                                 extended;
   uint32_t                             texture_type;
   uint32_t                             image_depth;
   uint32_t                             image_height;
   uint32_t                             image_width;
   uint32_t                             array_stride_64_byte_aligned;
   __gen_address_type                   texture_base_pointer;
   bool                                 reverse_standard_border_color;
   bool                                 ahdr;
   bool                                 srgb;
   bool                                 flip_s_and_t_on_incoming_request;
   bool                                 flip_texture_y_axis;
   bool                                 flip_texture_x_axis;
};

static inline void
V3D41_TEXTURE_SHADER_STATE_pack(__gen_user_data *data, uint8_t * restrict cl,
                                const struct V3D41_TEXTURE_SHADER_STATE * restrict values)
{
   __gen_emit_reloc(data, &values->texture_base_pointer);
   cl[ 0] = __gen_address_offset(&values->texture_base_pointer) |
            __gen_uint(values->reverse_standard_border_color, 5, 5) |
            __gen_uint(values->ahdr, 4, 4) |
            __gen_uint(values->srgb, 3, 3) |
            __gen_uint(values->flip_s_and_t_on_incoming_request, 2, 2) |
            __gen_uint(values->flip_texture_y_axis, 1, 1) |
            __gen_uint(values->flip_texture_x_axis, 0, 0);

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

   cl[13] = __gen_uint(values->swizzle_g, 7, 9) |
            __gen_uint(values->swizzle_r, 4, 6) |
            __gen_uint(values->extended, 3, 3) |
            __gen_uint(values->texture_type, 4, 10) >> 8;

   cl[14] = __gen_uint(values->swizzle_a, 5, 7) |
            __gen_uint(values->swizzle_b, 2, 4) |
            __gen_uint(values->swizzle_g, 7, 9) >> 8;

   cl[15] = __gen_uint(values->base_level, 4, 7) |
            __gen_uint(values->max_level, 0, 3);

   cl[16] = __gen_uint(values->uif_xor_disable, 7, 7) |
            __gen_uint(values->level_0_is_strictly_uif, 6, 6) |
            __gen_uint(values->level_0_xor_enable, 4, 4) |
            __gen_uint(values->level_0_ub_pad, 0, 3);

   cl[17] = __gen_uint(values->pad, 0, 55);

   cl[18] = __gen_uint(values->pad, 0, 55) >> 8;

   cl[19] = __gen_uint(values->pad, 0, 55) >> 16;

   cl[20] = __gen_uint(values->pad, 0, 55) >> 24;

   cl[21] = __gen_uint(values->pad, 0, 55) >> 32;

   cl[22] = __gen_uint(values->pad, 0, 55) >> 40;

   cl[23] = __gen_uint(values->pad, 0, 55) >> 48;

}

#define V3D41_TEXTURE_SHADER_STATE_length     24
#ifdef __gen_unpack_address
static inline void
V3D41_TEXTURE_SHADER_STATE_unpack(const uint8_t * restrict cl,
                                  struct V3D41_TEXTURE_SHADER_STATE * restrict values)
{
   values->pad = __gen_unpack_uint(cl, 136, 191);
   values->uif_xor_disable = __gen_unpack_uint(cl, 135, 135);
   values->level_0_is_strictly_uif = __gen_unpack_uint(cl, 134, 134);
   values->level_0_xor_enable = __gen_unpack_uint(cl, 132, 132);
   values->level_0_ub_pad = __gen_unpack_uint(cl, 128, 131);
   values->base_level = __gen_unpack_uint(cl, 124, 127);
   values->max_level = __gen_unpack_uint(cl, 120, 123);
   values->swizzle_a = __gen_unpack_uint(cl, 117, 119);
   values->swizzle_b = __gen_unpack_uint(cl, 114, 116);
   values->swizzle_g = __gen_unpack_uint(cl, 111, 113);
   values->swizzle_r = __gen_unpack_uint(cl, 108, 110);
   values->extended = __gen_unpack_uint(cl, 107, 107);
   values->texture_type = __gen_unpack_uint(cl, 100, 106);
   values->image_depth = __gen_unpack_uint(cl, 86, 99);
   values->image_height = __gen_unpack_uint(cl, 72, 85);
   values->image_width = __gen_unpack_uint(cl, 58, 71);
   values->array_stride_64_byte_aligned = __gen_unpack_uint(cl, 32, 57);
   values->texture_base_pointer = __gen_unpack_address(cl, 0, 31);
   values->reverse_standard_border_color = __gen_unpack_uint(cl, 5, 5);
   values->ahdr = __gen_unpack_uint(cl, 4, 4);
   values->srgb = __gen_unpack_uint(cl, 3, 3);
   values->flip_s_and_t_on_incoming_request = __gen_unpack_uint(cl, 2, 2);
   values->flip_texture_y_axis = __gen_unpack_uint(cl, 1, 1);
   values->flip_texture_x_axis = __gen_unpack_uint(cl, 0, 0);
}
#endif


#define V3D41_SAMPLER_STATE_header              \


struct V3D41_SAMPLER_STATE {
   uint32_t                             border_color_alpha;
   uint32_t                             border_color_blue;
   uint32_t                             border_color_green;
   uint32_t                             border_color_red;
   uint32_t                             maximum_anisotropy;
   enum V3D41_Border_Color_Mode         border_color_mode;
   bool                                 wrap_i_border;
   enum V3D41_Wrap_Mode                 wrap_r;
   enum V3D41_Wrap_Mode                 wrap_t;
   enum V3D41_Wrap_Mode                 wrap_s;
   float                                fixed_bias;
   float                                max_level_of_detail;
   float                                min_level_of_detail;
   bool                                 srgb_disable;
   enum V3D41_Compare_Function          depth_compare_function;
   bool                                 anisotropy_enable;
   bool                                 mip_filter_nearest;
   bool                                 min_filter_nearest;
   bool                                 mag_filter_nearest;
};

static inline void
V3D41_SAMPLER_STATE_pack(__gen_user_data *data, uint8_t * restrict cl,
                         const struct V3D41_SAMPLER_STATE * restrict values)
{
   cl[ 0] = __gen_uint(values->srgb_disable, 7, 7) |
            __gen_uint(values->depth_compare_function, 4, 6) |
            __gen_uint(values->anisotropy_enable, 3, 3) |
            __gen_uint(values->mip_filter_nearest, 2, 2) |
            __gen_uint(values->min_filter_nearest, 1, 1) |
            __gen_uint(values->mag_filter_nearest, 0, 0);

   cl[ 1] = __gen_ufixed(values->min_level_of_detail, 0, 11, 8);

   cl[ 2] = __gen_ufixed(values->max_level_of_detail, 4, 15, 8) |
            __gen_ufixed(values->min_level_of_detail, 0, 11, 8) >> 8;

   cl[ 3] = __gen_ufixed(values->max_level_of_detail, 4, 15, 8) >> 8;

   cl[ 4] = __gen_sfixed(values->fixed_bias, 0, 15, 8);

   cl[ 5] = __gen_sfixed(values->fixed_bias, 0, 15, 8) >> 8;

   cl[ 6] = __gen_uint(values->wrap_r, 6, 8) |
            __gen_uint(values->wrap_t, 3, 5) |
            __gen_uint(values->wrap_s, 0, 2);

   cl[ 7] = __gen_uint(values->maximum_anisotropy, 5, 6) |
            __gen_uint(values->border_color_mode, 2, 4) |
            __gen_uint(values->wrap_i_border, 1, 1) |
            __gen_uint(values->wrap_r, 6, 8) >> 8;


   memcpy(&cl[8], &values->border_color_red, sizeof(values->border_color_red));

   memcpy(&cl[12], &values->border_color_green, sizeof(values->border_color_green));

   memcpy(&cl[16], &values->border_color_blue, sizeof(values->border_color_blue));

   memcpy(&cl[20], &values->border_color_alpha, sizeof(values->border_color_alpha));
}

#define V3D41_SAMPLER_STATE_length            24
#ifdef __gen_unpack_address
static inline void
V3D41_SAMPLER_STATE_unpack(const uint8_t * restrict cl,
                           struct V3D41_SAMPLER_STATE * restrict values)
{
   values->border_color_alpha = __gen_unpack_uint(cl, 160, 191);
   values->border_color_blue = __gen_unpack_uint(cl, 128, 159);
   values->border_color_green = __gen_unpack_uint(cl, 96, 127);
   values->border_color_red = __gen_unpack_uint(cl, 64, 95);
   values->maximum_anisotropy = __gen_unpack_uint(cl, 61, 62);
   values->border_color_mode = __gen_unpack_uint(cl, 58, 60);
   values->wrap_i_border = __gen_unpack_uint(cl, 57, 57);
   values->wrap_r = __gen_unpack_uint(cl, 54, 56);
   values->wrap_t = __gen_unpack_uint(cl, 51, 53);
   values->wrap_s = __gen_unpack_uint(cl, 48, 50);
   values->fixed_bias = __gen_unpack_sfixed(cl, 32, 47, 8);
   values->max_level_of_detail = __gen_unpack_ufixed(cl, 20, 31, 8);
   values->min_level_of_detail = __gen_unpack_ufixed(cl, 8, 19, 8);
   values->srgb_disable = __gen_unpack_uint(cl, 7, 7);
   values->depth_compare_function = __gen_unpack_uint(cl, 4, 6);
   values->anisotropy_enable = __gen_unpack_uint(cl, 3, 3);
   values->mip_filter_nearest = __gen_unpack_uint(cl, 2, 2);
   values->min_filter_nearest = __gen_unpack_uint(cl, 1, 1);
   values->mag_filter_nearest = __gen_unpack_uint(cl, 0, 0);
}
#endif


enum V3D41_Texture_Data_Formats {
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

#endif /* V3D41_PACK_H */
