/**********************************************************
 * Copyright 2007-2015 VMware, Inc.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **********************************************************/

/*
 * VGPU10ShaderTokens.h --
 *
 *    VGPU10 shader token definitions.
 *
 */

#ifndef VGPU10SHADERTOKENS_H
#define VGPU10SHADERTOKENS_H

/* Shader limits */
#define VGPU10_MAX_VS_INPUTS 16
#define VGPU10_MAX_VS_OUTPUTS 16
#define VGPU10_MAX_GS_INPUTS 16
#define VGPU10_MAX_GS_OUTPUTS 32
#define VGPU10_MAX_FS_INPUTS 32
#define VGPU10_MAX_FS_OUTPUTS 8
#define VGPU10_MAX_TEMPS 4096
#define VGPU10_MAX_CONSTANT_BUFFERS 14
#define VGPU10_MAX_CONSTANT_BUFFER_ELEMENT_COUNT 4096
#define VGPU10_MAX_IMMEDIATE_CONSTANT_BUFFER_ELEMENT_COUNT 4096
#define VGPU10_MAX_SAMPLERS 16
#define VGPU10_MAX_RESOURCES 128
#define VGPU10_MIN_TEXEL_FETCH_OFFSET -8
#define VGPU10_MAX_TEXEL_FETCH_OFFSET 7

typedef enum {
   VGPU10_PIXEL_SHADER = 0,
   VGPU10_VERTEX_SHADER = 1,
   VGPU10_GEOMETRY_SHADER = 2
} VGPU10_PROGRAM_TYPE;

typedef union {
   struct {
      unsigned int minorVersion  : 4;
      unsigned int majorVersion  : 4;
      unsigned int               : 8;
      unsigned int programType   : 16; /* VGPU10_PROGRAM_TYPE */
   };
   uint32 value;
} VGPU10ProgramToken;


typedef enum {
   VGPU10_OPCODE_ADD                               = 0,
   VGPU10_OPCODE_AND                               = 1,
   VGPU10_OPCODE_BREAK                             = 2,
   VGPU10_OPCODE_BREAKC                            = 3,
   VGPU10_OPCODE_CALL                              = 4,
   VGPU10_OPCODE_CALLC                             = 5,
   VGPU10_OPCODE_CASE                              = 6,
   VGPU10_OPCODE_CONTINUE                          = 7,
   VGPU10_OPCODE_CONTINUEC                         = 8,
   VGPU10_OPCODE_CUT                               = 9,
   VGPU10_OPCODE_DEFAULT                           = 10,
   VGPU10_OPCODE_DERIV_RTX                         = 11,
   VGPU10_OPCODE_DERIV_RTY                         = 12,
   VGPU10_OPCODE_DISCARD                           = 13,
   VGPU10_OPCODE_DIV                               = 14,
   VGPU10_OPCODE_DP2                               = 15,
   VGPU10_OPCODE_DP3                               = 16,
   VGPU10_OPCODE_DP4                               = 17,
   VGPU10_OPCODE_ELSE                              = 18,
   VGPU10_OPCODE_EMIT                              = 19,
   VGPU10_OPCODE_EMITTHENCUT                       = 20,
   VGPU10_OPCODE_ENDIF                             = 21,
   VGPU10_OPCODE_ENDLOOP                           = 22,
   VGPU10_OPCODE_ENDSWITCH                         = 23,
   VGPU10_OPCODE_EQ                                = 24,
   VGPU10_OPCODE_EXP                               = 25,
   VGPU10_OPCODE_FRC                               = 26,
   VGPU10_OPCODE_FTOI                              = 27,
   VGPU10_OPCODE_FTOU                              = 28,
   VGPU10_OPCODE_GE                                = 29,
   VGPU10_OPCODE_IADD                              = 30,
   VGPU10_OPCODE_IF                                = 31,
   VGPU10_OPCODE_IEQ                               = 32,
   VGPU10_OPCODE_IGE                               = 33,
   VGPU10_OPCODE_ILT                               = 34,
   VGPU10_OPCODE_IMAD                              = 35,
   VGPU10_OPCODE_IMAX                              = 36,
   VGPU10_OPCODE_IMIN                              = 37,
   VGPU10_OPCODE_IMUL                              = 38,
   VGPU10_OPCODE_INE                               = 39,
   VGPU10_OPCODE_INEG                              = 40,
   VGPU10_OPCODE_ISHL                              = 41,
   VGPU10_OPCODE_ISHR                              = 42,
   VGPU10_OPCODE_ITOF                              = 43,
   VGPU10_OPCODE_LABEL                             = 44,
   VGPU10_OPCODE_LD                                = 45,
   VGPU10_OPCODE_LD_MS                             = 46,
   VGPU10_OPCODE_LOG                               = 47,
   VGPU10_OPCODE_LOOP                              = 48,
   VGPU10_OPCODE_LT                                = 49,
   VGPU10_OPCODE_MAD                               = 50,
   VGPU10_OPCODE_MIN                               = 51,
   VGPU10_OPCODE_MAX                               = 52,
   VGPU10_OPCODE_CUSTOMDATA                        = 53,
   VGPU10_OPCODE_MOV                               = 54,
   VGPU10_OPCODE_MOVC                              = 55,
   VGPU10_OPCODE_MUL                               = 56,
   VGPU10_OPCODE_NE                                = 57,
   VGPU10_OPCODE_NOP                               = 58,
   VGPU10_OPCODE_NOT                               = 59,
   VGPU10_OPCODE_OR                                = 60,
   VGPU10_OPCODE_RESINFO                           = 61,
   VGPU10_OPCODE_RET                               = 62,
   VGPU10_OPCODE_RETC                              = 63,
   VGPU10_OPCODE_ROUND_NE                          = 64,
   VGPU10_OPCODE_ROUND_NI                          = 65,
   VGPU10_OPCODE_ROUND_PI                          = 66,
   VGPU10_OPCODE_ROUND_Z                           = 67,
   VGPU10_OPCODE_RSQ                               = 68,
   VGPU10_OPCODE_SAMPLE                            = 69,
   VGPU10_OPCODE_SAMPLE_C                          = 70,
   VGPU10_OPCODE_SAMPLE_C_LZ                       = 71,
   VGPU10_OPCODE_SAMPLE_L                          = 72,
   VGPU10_OPCODE_SAMPLE_D                          = 73,
   VGPU10_OPCODE_SAMPLE_B                          = 74,
   VGPU10_OPCODE_SQRT                              = 75,
   VGPU10_OPCODE_SWITCH                            = 76,
   VGPU10_OPCODE_SINCOS                            = 77,
   VGPU10_OPCODE_UDIV                              = 78,
   VGPU10_OPCODE_ULT                               = 79,
   VGPU10_OPCODE_UGE                               = 80,
   VGPU10_OPCODE_UMUL                              = 81,
   VGPU10_OPCODE_UMAD                              = 82,
   VGPU10_OPCODE_UMAX                              = 83,
   VGPU10_OPCODE_UMIN                              = 84,
   VGPU10_OPCODE_USHR                              = 85,
   VGPU10_OPCODE_UTOF                              = 86,
   VGPU10_OPCODE_XOR                               = 87,
   VGPU10_OPCODE_DCL_RESOURCE                      = 88,
   VGPU10_OPCODE_DCL_CONSTANT_BUFFER               = 89,
   VGPU10_OPCODE_DCL_SAMPLER                       = 90,
   VGPU10_OPCODE_DCL_INDEX_RANGE                   = 91,
   VGPU10_OPCODE_DCL_GS_OUTPUT_PRIMITIVE_TOPOLOGY  = 92,
   VGPU10_OPCODE_DCL_GS_INPUT_PRIMITIVE            = 93,
   VGPU10_OPCODE_DCL_MAX_OUTPUT_VERTEX_COUNT       = 94,
   VGPU10_OPCODE_DCL_INPUT                         = 95,
   VGPU10_OPCODE_DCL_INPUT_SGV                     = 96,
   VGPU10_OPCODE_DCL_INPUT_SIV                     = 97,
   VGPU10_OPCODE_DCL_INPUT_PS                      = 98,
   VGPU10_OPCODE_DCL_INPUT_PS_SGV                  = 99,
   VGPU10_OPCODE_DCL_INPUT_PS_SIV                  = 100,
   VGPU10_OPCODE_DCL_OUTPUT                        = 101,
   VGPU10_OPCODE_DCL_OUTPUT_SGV                    = 102,
   VGPU10_OPCODE_DCL_OUTPUT_SIV                    = 103,
   VGPU10_OPCODE_DCL_TEMPS                         = 104,
   VGPU10_OPCODE_DCL_INDEXABLE_TEMP                = 105,
   VGPU10_OPCODE_DCL_GLOBAL_FLAGS                  = 106,
   VGPU10_OPCODE_IDIV                              = 107,
   VGPU10_OPCODE_LOD                               = 108,
   VGPU10_OPCODE_GATHER4                           = 109,
   VGPU10_OPCODE_SAMPLE_POS                        = 110,
   VGPU10_OPCODE_SAMPLE_INFO                       = 111,
   VGPU10_NUM_OPCODES                  /* Should be the last entry. */
} VGPU10_OPCODE_TYPE;

typedef enum {
   VGPU10_INTERPOLATION_UNDEFINED = 0,
   VGPU10_INTERPOLATION_CONSTANT = 1,
   VGPU10_INTERPOLATION_LINEAR = 2,
   VGPU10_INTERPOLATION_LINEAR_CENTROID = 3,
   VGPU10_INTERPOLATION_LINEAR_NOPERSPECTIVE = 4,
   VGPU10_INTERPOLATION_LINEAR_NOPERSPECTIVE_CENTROID = 5,
   VGPU10_INTERPOLATION_LINEAR_SAMPLE = 6,                  /* DX10.1 */
   VGPU10_INTERPOLATION_LINEAR_NOPERSPECTIVE_SAMPLE = 7     /* DX10.1 */
} VGPU10_INTERPOLATION_MODE;

typedef enum {
   VGPU10_RESOURCE_DIMENSION_UNKNOWN = 0,
   VGPU10_RESOURCE_DIMENSION_BUFFER = 1,
   VGPU10_RESOURCE_DIMENSION_TEXTURE1D = 2,
   VGPU10_RESOURCE_DIMENSION_TEXTURE2D = 3,
   VGPU10_RESOURCE_DIMENSION_TEXTURE2DMS = 4,
   VGPU10_RESOURCE_DIMENSION_TEXTURE3D = 5,
   VGPU10_RESOURCE_DIMENSION_TEXTURECUBE = 6,
   VGPU10_RESOURCE_DIMENSION_TEXTURE1DARRAY = 7,
   VGPU10_RESOURCE_DIMENSION_TEXTURE2DARRAY = 8,
   VGPU10_RESOURCE_DIMENSION_TEXTURE2DMSARRAY = 9,
   VGPU10_RESOURCE_DIMENSION_TEXTURECUBEARRAY = 10
} VGPU10_RESOURCE_DIMENSION;

typedef enum {
   VGPU10_SAMPLER_MODE_DEFAULT = 0,
   VGPU10_SAMPLER_MODE_COMPARISON = 1,
   VGPU10_SAMPLER_MODE_MONO = 2
} VGPU10_SAMPLER_MODE;

typedef enum {
   VGPU10_INSTRUCTION_TEST_ZERO     = 0,
   VGPU10_INSTRUCTION_TEST_NONZERO  = 1
} VGPU10_INSTRUCTION_TEST_BOOLEAN;

typedef enum {
   VGPU10_CB_IMMEDIATE_INDEXED   = 0,
   VGPU10_CB_DYNAMIC_INDEXED     = 1
} VGPU10_CB_ACCESS_PATTERN;

typedef enum {
   VGPU10_PRIMITIVE_UNDEFINED    = 0,
   VGPU10_PRIMITIVE_POINT        = 1,
   VGPU10_PRIMITIVE_LINE         = 2,
   VGPU10_PRIMITIVE_TRIANGLE     = 3,
   VGPU10_PRIMITIVE_LINE_ADJ     = 6,
   VGPU10_PRIMITIVE_TRIANGLE_ADJ = 7
} VGPU10_PRIMITIVE;

typedef enum {
   VGPU10_PRIMITIVE_TOPOLOGY_UNDEFINED          = 0,
   VGPU10_PRIMITIVE_TOPOLOGY_POINTLIST          = 1,
   VGPU10_PRIMITIVE_TOPOLOGY_LINELIST           = 2,
   VGPU10_PRIMITIVE_TOPOLOGY_LINESTRIP          = 3,
   VGPU10_PRIMITIVE_TOPOLOGY_TRIANGLELIST       = 4,
   VGPU10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP      = 5,
   VGPU10_PRIMITIVE_TOPOLOGY_LINELIST_ADJ       = 10,
   VGPU10_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ      = 11,
   VGPU10_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ   = 12,
   VGPU10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ  = 13
} VGPU10_PRIMITIVE_TOPOLOGY;

typedef enum {
   VGPU10_CUSTOMDATA_COMMENT                       = 0,
   VGPU10_CUSTOMDATA_DEBUGINFO                     = 1,
   VGPU10_CUSTOMDATA_OPAQUE                        = 2,
   VGPU10_CUSTOMDATA_DCL_IMMEDIATE_CONSTANT_BUFFER = 3
} VGPU10_CUSTOMDATA_CLASS;

typedef enum {
   VGPU10_RESINFO_RETURN_FLOAT      = 0,
   VGPU10_RESINFO_RETURN_RCPFLOAT   = 1,
   VGPU10_RESINFO_RETURN_UINT       = 2
} VGPU10_RESINFO_RETURN_TYPE;


typedef enum {
   VGPU10_INSTRUCTION_RETURN_FLOAT  = 0,
   VGPU10_INSTRUCTION_RETURN_UINT   = 1
} VGPU10_INSTRUCTION_RETURN_TYPE;

typedef union {
   struct {
      unsigned int opcodeType          : 11; /* VGPU10_OPCODE_TYPE */
      unsigned int interpolationMode   : 4;  /* VGPU10_INTERPOLATION_MODE */
      unsigned int                     : 3;
      unsigned int testBoolean         : 1;  /* VGPU10_INSTRUCTION_TEST_BOOLEAN */
      unsigned int                     : 5;
      unsigned int instructionLength   : 7;
      unsigned int extended            : 1;
   };
   struct {
      unsigned int                     : 11;
      unsigned int resourceDimension   : 5;  /* VGPU10_RESOURCE_DIMENSION */
      unsigned int sampleCount         : 7;
   };
   struct {
      unsigned int                     : 11;
      unsigned int samplerMode         : 4;  /* VGPU10_SAMPLER_MODE */
   };
   struct {
      unsigned int                     : 11;
      unsigned int accessPattern       : 1;  /* VGPU10_CB_ACCESS_PATTERN */
   };
   struct {
      unsigned int                     : 11;
      unsigned int primitive           : 6;  /* VGPU10_PRIMITIVE */
   };
   struct {
      unsigned int                     : 11;
      unsigned int primitiveTopology   : 6;  /* VGPU10_PRIMITIVE_TOPOLOGY */
   };
   struct {
      unsigned int                     : 11;
      unsigned int customDataClass     : 21; /* VGPU10_CUSTOMDATA_CLASS */
   };
   struct {
      unsigned int                     : 11;
      unsigned int resinfoReturnType   : 2;  /* VGPU10_RESINFO_RETURN_TYPE */
      unsigned int saturate            : 1;
   };
   struct {
      unsigned int                     : 11;
      unsigned int refactoringAllowed  : 1;
   };
   struct {
      unsigned int                     : 11;
      unsigned int instReturnType      : 2;  /* VGPU10_INSTRUCTION_RETURN_TYPE */
   };
   uint32 value;
} VGPU10OpcodeToken0;


typedef enum {
   VGPU10_EXTENDED_OPCODE_EMPTY = 0,
   VGPU10_EXTENDED_OPCODE_SAMPLE_CONTROLS
} VGPU10_EXTENDED_OPCODE_TYPE;

typedef union {
   struct {
      unsigned int opcodeType : 6;  /* VGPU10_EXTENDED_OPCODE_TYPE */
      unsigned int            : 3;
      unsigned int offsetU    : 4;  /* Two's complement. */
      unsigned int offsetV    : 4;  /* Two's complement. */
      unsigned int offsetW    : 4;  /* Two's complement. */
      unsigned int            : 10;
      unsigned int extended   : 1;
   };
   uint32 value;
} VGPU10OpcodeToken1;


typedef enum {
   VGPU10_OPERAND_0_COMPONENT = 0,
   VGPU10_OPERAND_1_COMPONENT = 1,
   VGPU10_OPERAND_4_COMPONENT = 2,
   VGPU10_OPERAND_N_COMPONENT = 3   /* Unused for now. */
} VGPU10_OPERAND_NUM_COMPONENTS;

typedef enum {
   VGPU10_OPERAND_4_COMPONENT_MASK_MODE = 0,
   VGPU10_OPERAND_4_COMPONENT_SWIZZLE_MODE = 1,
   VGPU10_OPERAND_4_COMPONENT_SELECT_1_MODE = 2
} VGPU10_OPERAND_4_COMPONENT_SELECTION_MODE;

#define VGPU10_OPERAND_4_COMPONENT_MASK_X    0x1
#define VGPU10_OPERAND_4_COMPONENT_MASK_Y    0x2
#define VGPU10_OPERAND_4_COMPONENT_MASK_Z    0x4
#define VGPU10_OPERAND_4_COMPONENT_MASK_W    0x8

#define VGPU10_OPERAND_4_COMPONENT_MASK_XY   (VGPU10_OPERAND_4_COMPONENT_MASK_X   | VGPU10_OPERAND_4_COMPONENT_MASK_Y)
#define VGPU10_OPERAND_4_COMPONENT_MASK_XZ   (VGPU10_OPERAND_4_COMPONENT_MASK_X   | VGPU10_OPERAND_4_COMPONENT_MASK_Z)
#define VGPU10_OPERAND_4_COMPONENT_MASK_XW   (VGPU10_OPERAND_4_COMPONENT_MASK_X   | VGPU10_OPERAND_4_COMPONENT_MASK_W)
#define VGPU10_OPERAND_4_COMPONENT_MASK_YZ   (VGPU10_OPERAND_4_COMPONENT_MASK_Y   | VGPU10_OPERAND_4_COMPONENT_MASK_Z)
#define VGPU10_OPERAND_4_COMPONENT_MASK_YW   (VGPU10_OPERAND_4_COMPONENT_MASK_Y   | VGPU10_OPERAND_4_COMPONENT_MASK_W)
#define VGPU10_OPERAND_4_COMPONENT_MASK_ZW   (VGPU10_OPERAND_4_COMPONENT_MASK_Z   | VGPU10_OPERAND_4_COMPONENT_MASK_W)
#define VGPU10_OPERAND_4_COMPONENT_MASK_XYZ  (VGPU10_OPERAND_4_COMPONENT_MASK_XY  | VGPU10_OPERAND_4_COMPONENT_MASK_Z)
#define VGPU10_OPERAND_4_COMPONENT_MASK_XYW  (VGPU10_OPERAND_4_COMPONENT_MASK_XY  | VGPU10_OPERAND_4_COMPONENT_MASK_W)
#define VGPU10_OPERAND_4_COMPONENT_MASK_XZW  (VGPU10_OPERAND_4_COMPONENT_MASK_XZ  | VGPU10_OPERAND_4_COMPONENT_MASK_W)
#define VGPU10_OPERAND_4_COMPONENT_MASK_YZW  (VGPU10_OPERAND_4_COMPONENT_MASK_YZ  | VGPU10_OPERAND_4_COMPONENT_MASK_W)
#define VGPU10_OPERAND_4_COMPONENT_MASK_XYZW (VGPU10_OPERAND_4_COMPONENT_MASK_XYZ | VGPU10_OPERAND_4_COMPONENT_MASK_W)
#define VGPU10_OPERAND_4_COMPONENT_MASK_ALL  VGPU10_OPERAND_4_COMPONENT_MASK_XYZW

#define VGPU10_REGISTER_INDEX_FROM_SEMANTIC  0xffffffff

typedef enum {
   VGPU10_COMPONENT_X = 0,
   VGPU10_COMPONENT_Y = 1,
   VGPU10_COMPONENT_Z = 2,
   VGPU10_COMPONENT_W = 3
} VGPU10_COMPONENT_NAME;

typedef enum {
   VGPU10_OPERAND_TYPE_TEMP = 0,
   VGPU10_OPERAND_TYPE_INPUT = 1,
   VGPU10_OPERAND_TYPE_OUTPUT = 2,
   VGPU10_OPERAND_TYPE_INDEXABLE_TEMP = 3,
   VGPU10_OPERAND_TYPE_IMMEDIATE32 = 4,
   VGPU10_OPERAND_TYPE_IMMEDIATE64 = 5,
   VGPU10_OPERAND_TYPE_SAMPLER = 6,
   VGPU10_OPERAND_TYPE_RESOURCE = 7,
   VGPU10_OPERAND_TYPE_CONSTANT_BUFFER = 8,
   VGPU10_OPERAND_TYPE_IMMEDIATE_CONSTANT_BUFFER = 9,
   VGPU10_OPERAND_TYPE_LABEL = 10,
   VGPU10_OPERAND_TYPE_INPUT_PRIMITIVEID = 11,
   VGPU10_OPERAND_TYPE_OUTPUT_DEPTH = 12,
   VGPU10_OPERAND_TYPE_NULL = 13,
   VGPU10_OPERAND_TYPE_RASTERIZER = 14,            /* DX10.1 */
   VGPU10_OPERAND_TYPE_OUTPUT_COVERAGE_MASK = 15   /* DX10.1 */
} VGPU10_OPERAND_TYPE;

typedef enum {
   VGPU10_OPERAND_INDEX_0D = 0,
   VGPU10_OPERAND_INDEX_1D = 1,
   VGPU10_OPERAND_INDEX_2D = 2,
   VGPU10_OPERAND_INDEX_3D = 3
} VGPU10_OPERAND_INDEX_DIMENSION;

typedef enum {
   VGPU10_OPERAND_INDEX_IMMEDIATE32 = 0,
   VGPU10_OPERAND_INDEX_IMMEDIATE64 = 1,
   VGPU10_OPERAND_INDEX_RELATIVE = 2,
   VGPU10_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE = 3,
   VGPU10_OPERAND_INDEX_IMMEDIATE64_PLUS_RELATIVE = 4
} VGPU10_OPERAND_INDEX_REPRESENTATION;

typedef union {
   struct {
      unsigned int numComponents          : 2;  /* VGPU10_OPERAND_NUM_COMPONENTS */
      unsigned int selectionMode          : 2;  /* VGPU10_OPERAND_4_COMPONENT_SELECTION_MODE */
      unsigned int mask                   : 4;  /* D3D10_SB_OPERAND_4_COMPONENT_MASK_* */
      unsigned int                        : 4;
      unsigned int operandType            : 8;  /* VGPU10_OPERAND_TYPE */
      unsigned int indexDimension         : 2;  /* VGPU10_OPERAND_INDEX_DIMENSION */
      unsigned int index0Representation   : 3;  /* VGPU10_OPERAND_INDEX_REPRESENTATION */
      unsigned int index1Representation   : 3;  /* VGPU10_OPERAND_INDEX_REPRESENTATION */
      unsigned int                        : 3;
      unsigned int extended               : 1;
   };
   struct {
      unsigned int                        : 4;
      unsigned int swizzleX               : 2;  /* VGPU10_COMPONENT_NAME */
      unsigned int swizzleY               : 2;  /* VGPU10_COMPONENT_NAME */
      unsigned int swizzleZ               : 2;  /* VGPU10_COMPONENT_NAME */
      unsigned int swizzleW               : 2;  /* VGPU10_COMPONENT_NAME */
   };
   struct {
      unsigned int                        : 4;
      unsigned int selectMask             : 2;  /* VGPU10_COMPONENT_NAME */
   };
   uint32 value;
} VGPU10OperandToken0;


typedef enum {
   VGPU10_EXTENDED_OPERAND_EMPTY = 0,
   VGPU10_EXTENDED_OPERAND_MODIFIER = 1
} VGPU10_EXTENDED_OPERAND_TYPE;

typedef enum {
   VGPU10_OPERAND_MODIFIER_NONE = 0,
   VGPU10_OPERAND_MODIFIER_NEG = 1,
   VGPU10_OPERAND_MODIFIER_ABS = 2,
   VGPU10_OPERAND_MODIFIER_ABSNEG = 3
} VGPU10_OPERAND_MODIFIER;

typedef union {
   struct {
      unsigned int extendedOperandType : 6;  /* VGPU10_EXTENDED_OPERAND_TYPE */
      unsigned int operandModifier     : 8;  /* VGPU10_OPERAND_MODIFIER */
      unsigned int                     : 17;
      unsigned int extended            : 1;
   };
   uint32 value;
} VGPU10OperandToken1;


typedef enum {
   VGPU10_RETURN_TYPE_MIN     = 1,
   VGPU10_RETURN_TYPE_UNORM   = 1,
   VGPU10_RETURN_TYPE_SNORM   = 2,
   VGPU10_RETURN_TYPE_SINT    = 3,
   VGPU10_RETURN_TYPE_UINT    = 4,
   VGPU10_RETURN_TYPE_FLOAT   = 5,
   VGPU10_RETURN_TYPE_MIXED   = 6,
   VGPU10_RETURN_TYPE_MAX     = 6
} VGPU10_RESOURCE_RETURN_TYPE;

typedef union {
   struct {
      unsigned int component0 : 4;  /* VGPU10_RESOURCE_RETURN_TYPE */
      unsigned int component1 : 4;  /* VGPU10_RESOURCE_RETURN_TYPE */
      unsigned int component2 : 4;  /* VGPU10_RESOURCE_RETURN_TYPE */
      unsigned int component3 : 4;  /* VGPU10_RESOURCE_RETURN_TYPE */
   };
   uint32 value;
} VGPU10ResourceReturnTypeToken;


typedef enum {
   VGPU10_NAME_MIN                        = 0,
   VGPU10_NAME_UNDEFINED                  = 0,
   VGPU10_NAME_POSITION                   = 1,
   VGPU10_NAME_CLIP_DISTANCE              = 2,
   VGPU10_NAME_CULL_DISTANCE              = 3,
   VGPU10_NAME_RENDER_TARGET_ARRAY_INDEX  = 4,
   VGPU10_NAME_VIEWPORT_ARRAY_INDEX       = 5,
   VGPU10_NAME_VERTEX_ID                  = 6,
   VGPU10_NAME_PRIMITIVE_ID               = 7,
   VGPU10_NAME_INSTANCE_ID                = 8,
   VGPU10_NAME_IS_FRONT_FACE              = 9,
   VGPU10_NAME_SAMPLE_INDEX               = 10,
   VGPU10_NAME_MAX                        = 10
} VGPU10_SYSTEM_NAME;

typedef union {
   struct {
      unsigned int name : 16; /* VGPU10_SYSTEM_NAME */
   };
   uint32 value;
} VGPU10NameToken;

#endif
