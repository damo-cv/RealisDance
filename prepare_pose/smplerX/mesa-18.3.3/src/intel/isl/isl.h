/*
 * Copyright 2015 Intel Corporation
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

/**
 * @file
 * @brief Intel Surface Layout
 *
 * Header Layout
 * -------------
 * The header is ordered as:
 *    - forward declarations
 *    - macros that may be overridden at compile-time for specific gens
 *    - enums and constants
 *    - structs and unions
 *    - functions
 */

#ifndef ISL_H
#define ISL_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "c99_compat.h"
#include "util/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gen_device_info;
struct brw_image_param;

#ifndef ISL_DEV_GEN
/**
 * @brief Get the hardware generation of isl_device.
 *
 * You can define this as a compile-time constant in the CFLAGS. For example,
 * `gcc -DISL_DEV_GEN(dev)=9 ...`.
 */
#define ISL_DEV_GEN(__dev) ((__dev)->info->gen)
#define ISL_DEV_GEN_SANITIZE(__dev)
#else
#define ISL_DEV_GEN_SANITIZE(__dev) \
   (assert(ISL_DEV_GEN(__dev) == (__dev)->info->gen))
#endif

#ifndef ISL_DEV_IS_G4X
#define ISL_DEV_IS_G4X(__dev) ((__dev)->info->is_g4x)
#endif

#ifndef ISL_DEV_IS_HASWELL
/**
 * @brief Get the hardware generation of isl_device.
 *
 * You can define this as a compile-time constant in the CFLAGS. For example,
 * `gcc -DISL_DEV_GEN(dev)=9 ...`.
 */
#define ISL_DEV_IS_HASWELL(__dev) ((__dev)->info->is_haswell)
#endif

#ifndef ISL_DEV_IS_BAYTRAIL
#define ISL_DEV_IS_BAYTRAIL(__dev) ((__dev)->info->is_baytrail)
#endif

#ifndef ISL_DEV_USE_SEPARATE_STENCIL
/**
 * You can define this as a compile-time constant in the CFLAGS. For example,
 * `gcc -DISL_DEV_USE_SEPARATE_STENCIL(dev)=1 ...`.
 */
#define ISL_DEV_USE_SEPARATE_STENCIL(__dev) ((__dev)->use_separate_stencil)
#define ISL_DEV_USE_SEPARATE_STENCIL_SANITIZE(__dev)
#else
#define ISL_DEV_USE_SEPARATE_STENCIL_SANITIZE(__dev) \
   (assert(ISL_DEV_USE_SEPARATE_STENCIL(__dev) == (__dev)->use_separate_stencil))
#endif

/**
 * Hardware enumeration SURFACE_FORMAT.
 *
 * For the official list, see Broadwell PRM: Volume 2b: Command Reference:
 * Enumerations: SURFACE_FORMAT.
 */
enum isl_format {
   ISL_FORMAT_R32G32B32A32_FLOAT =                               0,
   ISL_FORMAT_R32G32B32A32_SINT =                                1,
   ISL_FORMAT_R32G32B32A32_UINT =                                2,
   ISL_FORMAT_R32G32B32A32_UNORM =                               3,
   ISL_FORMAT_R32G32B32A32_SNORM =                               4,
   ISL_FORMAT_R64G64_FLOAT =                                     5,
   ISL_FORMAT_R32G32B32X32_FLOAT =                               6,
   ISL_FORMAT_R32G32B32A32_SSCALED =                             7,
   ISL_FORMAT_R32G32B32A32_USCALED =                             8,
   ISL_FORMAT_R32G32B32A32_SFIXED =                             32,
   ISL_FORMAT_R64G64_PASSTHRU =                                 33,
   ISL_FORMAT_R32G32B32_FLOAT =                                 64,
   ISL_FORMAT_R32G32B32_SINT =                                  65,
   ISL_FORMAT_R32G32B32_UINT =                                  66,
   ISL_FORMAT_R32G32B32_UNORM =                                 67,
   ISL_FORMAT_R32G32B32_SNORM =                                 68,
   ISL_FORMAT_R32G32B32_SSCALED =                               69,
   ISL_FORMAT_R32G32B32_USCALED =                               70,
   ISL_FORMAT_R32G32B32_SFIXED =                                80,
   ISL_FORMAT_R16G16B16A16_UNORM =                             128,
   ISL_FORMAT_R16G16B16A16_SNORM =                             129,
   ISL_FORMAT_R16G16B16A16_SINT =                              130,
   ISL_FORMAT_R16G16B16A16_UINT =                              131,
   ISL_FORMAT_R16G16B16A16_FLOAT =                             132,
   ISL_FORMAT_R32G32_FLOAT =                                   133,
   ISL_FORMAT_R32G32_SINT =                                    134,
   ISL_FORMAT_R32G32_UINT =                                    135,
   ISL_FORMAT_R32_FLOAT_X8X24_TYPELESS =                       136,
   ISL_FORMAT_X32_TYPELESS_G8X24_UINT =                        137,
   ISL_FORMAT_L32A32_FLOAT =                                   138,
   ISL_FORMAT_R32G32_UNORM =                                   139,
   ISL_FORMAT_R32G32_SNORM =                                   140,
   ISL_FORMAT_R64_FLOAT =                                      141,
   ISL_FORMAT_R16G16B16X16_UNORM =                             142,
   ISL_FORMAT_R16G16B16X16_FLOAT =                             143,
   ISL_FORMAT_A32X32_FLOAT =                                   144,
   ISL_FORMAT_L32X32_FLOAT =                                   145,
   ISL_FORMAT_I32X32_FLOAT =                                   146,
   ISL_FORMAT_R16G16B16A16_SSCALED =                           147,
   ISL_FORMAT_R16G16B16A16_USCALED =                           148,
   ISL_FORMAT_R32G32_SSCALED =                                 149,
   ISL_FORMAT_R32G32_USCALED =                                 150,
   ISL_FORMAT_R32G32_FLOAT_LD =                                151,
   ISL_FORMAT_R32G32_SFIXED =                                  160,
   ISL_FORMAT_R64_PASSTHRU =                                   161,
   ISL_FORMAT_B8G8R8A8_UNORM =                                 192,
   ISL_FORMAT_B8G8R8A8_UNORM_SRGB =                            193,
   ISL_FORMAT_R10G10B10A2_UNORM =                              194,
   ISL_FORMAT_R10G10B10A2_UNORM_SRGB =                         195,
   ISL_FORMAT_R10G10B10A2_UINT =                               196,
   ISL_FORMAT_R10G10B10_SNORM_A2_UNORM =                       197,
   ISL_FORMAT_R8G8B8A8_UNORM =                                 199,
   ISL_FORMAT_R8G8B8A8_UNORM_SRGB =                            200,
   ISL_FORMAT_R8G8B8A8_SNORM =                                 201,
   ISL_FORMAT_R8G8B8A8_SINT =                                  202,
   ISL_FORMAT_R8G8B8A8_UINT =                                  203,
   ISL_FORMAT_R16G16_UNORM =                                   204,
   ISL_FORMAT_R16G16_SNORM =                                   205,
   ISL_FORMAT_R16G16_SINT =                                    206,
   ISL_FORMAT_R16G16_UINT =                                    207,
   ISL_FORMAT_R16G16_FLOAT =                                   208,
   ISL_FORMAT_B10G10R10A2_UNORM =                              209,
   ISL_FORMAT_B10G10R10A2_UNORM_SRGB =                         210,
   ISL_FORMAT_R11G11B10_FLOAT =                                211,
   ISL_FORMAT_R32_SINT =                                       214,
   ISL_FORMAT_R32_UINT =                                       215,
   ISL_FORMAT_R32_FLOAT =                                      216,
   ISL_FORMAT_R24_UNORM_X8_TYPELESS =                          217,
   ISL_FORMAT_X24_TYPELESS_G8_UINT =                           218,
   ISL_FORMAT_L32_UNORM =                                      221,
   ISL_FORMAT_A32_UNORM =                                      222,
   ISL_FORMAT_L16A16_UNORM =                                   223,
   ISL_FORMAT_I24X8_UNORM =                                    224,
   ISL_FORMAT_L24X8_UNORM =                                    225,
   ISL_FORMAT_A24X8_UNORM =                                    226,
   ISL_FORMAT_I32_FLOAT =                                      227,
   ISL_FORMAT_L32_FLOAT =                                      228,
   ISL_FORMAT_A32_FLOAT =                                      229,
   ISL_FORMAT_X8B8_UNORM_G8R8_SNORM =                          230,
   ISL_FORMAT_A8X8_UNORM_G8R8_SNORM =                          231,
   ISL_FORMAT_B8X8_UNORM_G8R8_SNORM =                          232,
   ISL_FORMAT_B8G8R8X8_UNORM =                                 233,
   ISL_FORMAT_B8G8R8X8_UNORM_SRGB =                            234,
   ISL_FORMAT_R8G8B8X8_UNORM =                                 235,
   ISL_FORMAT_R8G8B8X8_UNORM_SRGB =                            236,
   ISL_FORMAT_R9G9B9E5_SHAREDEXP =                             237,
   ISL_FORMAT_B10G10R10X2_UNORM =                              238,
   ISL_FORMAT_L16A16_FLOAT =                                   240,
   ISL_FORMAT_R32_UNORM =                                      241,
   ISL_FORMAT_R32_SNORM =                                      242,
   ISL_FORMAT_R10G10B10X2_USCALED =                            243,
   ISL_FORMAT_R8G8B8A8_SSCALED =                               244,
   ISL_FORMAT_R8G8B8A8_USCALED =                               245,
   ISL_FORMAT_R16G16_SSCALED =                                 246,
   ISL_FORMAT_R16G16_USCALED =                                 247,
   ISL_FORMAT_R32_SSCALED =                                    248,
   ISL_FORMAT_R32_USCALED =                                    249,
   ISL_FORMAT_B5G6R5_UNORM =                                   256,
   ISL_FORMAT_B5G6R5_UNORM_SRGB =                              257,
   ISL_FORMAT_B5G5R5A1_UNORM =                                 258,
   ISL_FORMAT_B5G5R5A1_UNORM_SRGB =                            259,
   ISL_FORMAT_B4G4R4A4_UNORM =                                 260,
   ISL_FORMAT_B4G4R4A4_UNORM_SRGB =                            261,
   ISL_FORMAT_R8G8_UNORM =                                     262,
   ISL_FORMAT_R8G8_SNORM =                                     263,
   ISL_FORMAT_R8G8_SINT =                                      264,
   ISL_FORMAT_R8G8_UINT =                                      265,
   ISL_FORMAT_R16_UNORM =                                      266,
   ISL_FORMAT_R16_SNORM =                                      267,
   ISL_FORMAT_R16_SINT =                                       268,
   ISL_FORMAT_R16_UINT =                                       269,
   ISL_FORMAT_R16_FLOAT =                                      270,
   ISL_FORMAT_A8P8_UNORM_PALETTE0 =                            271,
   ISL_FORMAT_A8P8_UNORM_PALETTE1 =                            272,
   ISL_FORMAT_I16_UNORM =                                      273,
   ISL_FORMAT_L16_UNORM =                                      274,
   ISL_FORMAT_A16_UNORM =                                      275,
   ISL_FORMAT_L8A8_UNORM =                                     276,
   ISL_FORMAT_I16_FLOAT =                                      277,
   ISL_FORMAT_L16_FLOAT =                                      278,
   ISL_FORMAT_A16_FLOAT =                                      279,
   ISL_FORMAT_L8A8_UNORM_SRGB =                                280,
   ISL_FORMAT_R5G5_SNORM_B6_UNORM =                            281,
   ISL_FORMAT_B5G5R5X1_UNORM =                                 282,
   ISL_FORMAT_B5G5R5X1_UNORM_SRGB =                            283,
   ISL_FORMAT_R8G8_SSCALED =                                   284,
   ISL_FORMAT_R8G8_USCALED =                                   285,
   ISL_FORMAT_R16_SSCALED =                                    286,
   ISL_FORMAT_R16_USCALED =                                    287,
   ISL_FORMAT_P8A8_UNORM_PALETTE0 =                            290,
   ISL_FORMAT_P8A8_UNORM_PALETTE1 =                            291,
   ISL_FORMAT_A1B5G5R5_UNORM =                                 292,
   ISL_FORMAT_A4B4G4R4_UNORM =                                 293,
   ISL_FORMAT_L8A8_UINT =                                      294,
   ISL_FORMAT_L8A8_SINT =                                      295,
   ISL_FORMAT_R8_UNORM =                                       320,
   ISL_FORMAT_R8_SNORM =                                       321,
   ISL_FORMAT_R8_SINT =                                        322,
   ISL_FORMAT_R8_UINT =                                        323,
   ISL_FORMAT_A8_UNORM =                                       324,
   ISL_FORMAT_I8_UNORM =                                       325,
   ISL_FORMAT_L8_UNORM =                                       326,
   ISL_FORMAT_P4A4_UNORM_PALETTE0 =                            327,
   ISL_FORMAT_A4P4_UNORM_PALETTE0 =                            328,
   ISL_FORMAT_R8_SSCALED =                                     329,
   ISL_FORMAT_R8_USCALED =                                     330,
   ISL_FORMAT_P8_UNORM_PALETTE0 =                              331,
   ISL_FORMAT_L8_UNORM_SRGB =                                  332,
   ISL_FORMAT_P8_UNORM_PALETTE1 =                              333,
   ISL_FORMAT_P4A4_UNORM_PALETTE1 =                            334,
   ISL_FORMAT_A4P4_UNORM_PALETTE1 =                            335,
   ISL_FORMAT_Y8_UNORM =                                       336,
   ISL_FORMAT_L8_UINT =                                        338,
   ISL_FORMAT_L8_SINT =                                        339,
   ISL_FORMAT_I8_UINT =                                        340,
   ISL_FORMAT_I8_SINT =                                        341,
   ISL_FORMAT_DXT1_RGB_SRGB =                                  384,
   ISL_FORMAT_R1_UNORM =                                       385,
   ISL_FORMAT_YCRCB_NORMAL =                                   386,
   ISL_FORMAT_YCRCB_SWAPUVY =                                  387,
   ISL_FORMAT_P2_UNORM_PALETTE0 =                              388,
   ISL_FORMAT_P2_UNORM_PALETTE1 =                              389,
   ISL_FORMAT_BC1_UNORM =                                      390,
   ISL_FORMAT_BC2_UNORM =                                      391,
   ISL_FORMAT_BC3_UNORM =                                      392,
   ISL_FORMAT_BC4_UNORM =                                      393,
   ISL_FORMAT_BC5_UNORM =                                      394,
   ISL_FORMAT_BC1_UNORM_SRGB =                                 395,
   ISL_FORMAT_BC2_UNORM_SRGB =                                 396,
   ISL_FORMAT_BC3_UNORM_SRGB =                                 397,
   ISL_FORMAT_MONO8 =                                          398,
   ISL_FORMAT_YCRCB_SWAPUV =                                   399,
   ISL_FORMAT_YCRCB_SWAPY =                                    400,
   ISL_FORMAT_DXT1_RGB =                                       401,
   ISL_FORMAT_FXT1 =                                           402,
   ISL_FORMAT_R8G8B8_UNORM =                                   403,
   ISL_FORMAT_R8G8B8_SNORM =                                   404,
   ISL_FORMAT_R8G8B8_SSCALED =                                 405,
   ISL_FORMAT_R8G8B8_USCALED =                                 406,
   ISL_FORMAT_R64G64B64A64_FLOAT =                             407,
   ISL_FORMAT_R64G64B64_FLOAT =                                408,
   ISL_FORMAT_BC4_SNORM =                                      409,
   ISL_FORMAT_BC5_SNORM =                                      410,
   ISL_FORMAT_R16G16B16_FLOAT =                                411,
   ISL_FORMAT_R16G16B16_UNORM =                                412,
   ISL_FORMAT_R16G16B16_SNORM =                                413,
   ISL_FORMAT_R16G16B16_SSCALED =                              414,
   ISL_FORMAT_R16G16B16_USCALED =                              415,
   ISL_FORMAT_BC6H_SF16 =                                      417,
   ISL_FORMAT_BC7_UNORM =                                      418,
   ISL_FORMAT_BC7_UNORM_SRGB =                                 419,
   ISL_FORMAT_BC6H_UF16 =                                      420,
   ISL_FORMAT_PLANAR_420_8 =                                   421,
   ISL_FORMAT_R8G8B8_UNORM_SRGB =                              424,
   ISL_FORMAT_ETC1_RGB8 =                                      425,
   ISL_FORMAT_ETC2_RGB8 =                                      426,
   ISL_FORMAT_EAC_R11 =                                        427,
   ISL_FORMAT_EAC_RG11 =                                       428,
   ISL_FORMAT_EAC_SIGNED_R11 =                                 429,
   ISL_FORMAT_EAC_SIGNED_RG11 =                                430,
   ISL_FORMAT_ETC2_SRGB8 =                                     431,
   ISL_FORMAT_R16G16B16_UINT =                                 432,
   ISL_FORMAT_R16G16B16_SINT =                                 433,
   ISL_FORMAT_R32_SFIXED =                                     434,
   ISL_FORMAT_R10G10B10A2_SNORM =                              435,
   ISL_FORMAT_R10G10B10A2_USCALED =                            436,
   ISL_FORMAT_R10G10B10A2_SSCALED =                            437,
   ISL_FORMAT_R10G10B10A2_SINT =                               438,
   ISL_FORMAT_B10G10R10A2_SNORM =                              439,
   ISL_FORMAT_B10G10R10A2_USCALED =                            440,
   ISL_FORMAT_B10G10R10A2_SSCALED =                            441,
   ISL_FORMAT_B10G10R10A2_UINT =                               442,
   ISL_FORMAT_B10G10R10A2_SINT =                               443,
   ISL_FORMAT_R64G64B64A64_PASSTHRU =                          444,
   ISL_FORMAT_R64G64B64_PASSTHRU =                             445,
   ISL_FORMAT_ETC2_RGB8_PTA =                                  448,
   ISL_FORMAT_ETC2_SRGB8_PTA =                                 449,
   ISL_FORMAT_ETC2_EAC_RGBA8 =                                 450,
   ISL_FORMAT_ETC2_EAC_SRGB8_A8 =                              451,
   ISL_FORMAT_R8G8B8_UINT =                                    456,
   ISL_FORMAT_R8G8B8_SINT =                                    457,
   ISL_FORMAT_RAW =                                            511,
   ISL_FORMAT_ASTC_LDR_2D_4X4_U8SRGB =                         512,
   ISL_FORMAT_ASTC_LDR_2D_5X4_U8SRGB =                         520,
   ISL_FORMAT_ASTC_LDR_2D_5X5_U8SRGB =                         521,
   ISL_FORMAT_ASTC_LDR_2D_6X5_U8SRGB =                         529,
   ISL_FORMAT_ASTC_LDR_2D_6X6_U8SRGB =                         530,
   ISL_FORMAT_ASTC_LDR_2D_8X5_U8SRGB =                         545,
   ISL_FORMAT_ASTC_LDR_2D_8X6_U8SRGB =                         546,
   ISL_FORMAT_ASTC_LDR_2D_8X8_U8SRGB =                         548,
   ISL_FORMAT_ASTC_LDR_2D_10X5_U8SRGB =                        561,
   ISL_FORMAT_ASTC_LDR_2D_10X6_U8SRGB =                        562,
   ISL_FORMAT_ASTC_LDR_2D_10X8_U8SRGB =                        564,
   ISL_FORMAT_ASTC_LDR_2D_10X10_U8SRGB =                       566,
   ISL_FORMAT_ASTC_LDR_2D_12X10_U8SRGB =                       574,
   ISL_FORMAT_ASTC_LDR_2D_12X12_U8SRGB =                       575,
   ISL_FORMAT_ASTC_LDR_2D_4X4_FLT16 =                          576,
   ISL_FORMAT_ASTC_LDR_2D_5X4_FLT16 =                          584,
   ISL_FORMAT_ASTC_LDR_2D_5X5_FLT16 =                          585,
   ISL_FORMAT_ASTC_LDR_2D_6X5_FLT16 =                          593,
   ISL_FORMAT_ASTC_LDR_2D_6X6_FLT16 =                          594,
   ISL_FORMAT_ASTC_LDR_2D_8X5_FLT16 =                          609,
   ISL_FORMAT_ASTC_LDR_2D_8X6_FLT16 =                          610,
   ISL_FORMAT_ASTC_LDR_2D_8X8_FLT16 =                          612,
   ISL_FORMAT_ASTC_LDR_2D_10X5_FLT16 =                         625,
   ISL_FORMAT_ASTC_LDR_2D_10X6_FLT16 =                         626,
   ISL_FORMAT_ASTC_LDR_2D_10X8_FLT16 =                         628,
   ISL_FORMAT_ASTC_LDR_2D_10X10_FLT16 =                        630,
   ISL_FORMAT_ASTC_LDR_2D_12X10_FLT16 =                        638,
   ISL_FORMAT_ASTC_LDR_2D_12X12_FLT16 =                        639,
   ISL_FORMAT_ASTC_HDR_2D_4X4_FLT16 =                          832,
   ISL_FORMAT_ASTC_HDR_2D_5X4_FLT16 =                          840,
   ISL_FORMAT_ASTC_HDR_2D_5X5_FLT16 =                          841,
   ISL_FORMAT_ASTC_HDR_2D_6X5_FLT16 =                          849,
   ISL_FORMAT_ASTC_HDR_2D_6X6_FLT16 =                          850,
   ISL_FORMAT_ASTC_HDR_2D_8X5_FLT16 =                          865,
   ISL_FORMAT_ASTC_HDR_2D_8X6_FLT16 =                          866,
   ISL_FORMAT_ASTC_HDR_2D_8X8_FLT16 =                          868,
   ISL_FORMAT_ASTC_HDR_2D_10X5_FLT16 =                         881,
   ISL_FORMAT_ASTC_HDR_2D_10X6_FLT16 =                         882,
   ISL_FORMAT_ASTC_HDR_2D_10X8_FLT16 =                         884,
   ISL_FORMAT_ASTC_HDR_2D_10X10_FLT16 =                        886,
   ISL_FORMAT_ASTC_HDR_2D_12X10_FLT16 =                        894,
   ISL_FORMAT_ASTC_HDR_2D_12X12_FLT16 =                        895,

   /* The formats that follow are internal to ISL and as such don't have an
    * explicit number.  We'll just let the C compiler assign it for us.  Any
    * actual hardware formats *must* come before these in the list.
    */

   /* Formats for auxiliary surfaces */
   ISL_FORMAT_HIZ,
   ISL_FORMAT_MCS_2X,
   ISL_FORMAT_MCS_4X,
   ISL_FORMAT_MCS_8X,
   ISL_FORMAT_MCS_16X,
   ISL_FORMAT_GEN7_CCS_32BPP_X,
   ISL_FORMAT_GEN7_CCS_64BPP_X,
   ISL_FORMAT_GEN7_CCS_128BPP_X,
   ISL_FORMAT_GEN7_CCS_32BPP_Y,
   ISL_FORMAT_GEN7_CCS_64BPP_Y,
   ISL_FORMAT_GEN7_CCS_128BPP_Y,
   ISL_FORMAT_GEN9_CCS_32BPP,
   ISL_FORMAT_GEN9_CCS_64BPP,
   ISL_FORMAT_GEN9_CCS_128BPP,

   /* An upper bound on the supported format enumerations */
   ISL_NUM_FORMATS,

   /* Hardware doesn't understand this out-of-band value */
   ISL_FORMAT_UNSUPPORTED =                             UINT16_MAX,
};

/**
 * Numerical base type for channels of isl_format.
 */
enum isl_base_type {
   ISL_VOID,
   ISL_RAW,
   ISL_UNORM,
   ISL_SNORM,
   ISL_UFLOAT,
   ISL_SFLOAT,
   ISL_UFIXED,
   ISL_SFIXED,
   ISL_UINT,
   ISL_SINT,
   ISL_USCALED,
   ISL_SSCALED,
};

/**
 * Colorspace of isl_format.
 */
enum isl_colorspace {
   ISL_COLORSPACE_NONE = 0,
   ISL_COLORSPACE_LINEAR,
   ISL_COLORSPACE_SRGB,
   ISL_COLORSPACE_YUV,
};

/**
 * Texture compression mode of isl_format.
 */
enum isl_txc {
   ISL_TXC_NONE = 0,
   ISL_TXC_DXT1,
   ISL_TXC_DXT3,
   ISL_TXC_DXT5,
   ISL_TXC_FXT1,
   ISL_TXC_RGTC1,
   ISL_TXC_RGTC2,
   ISL_TXC_BPTC,
   ISL_TXC_ETC1,
   ISL_TXC_ETC2,
   ISL_TXC_ASTC,

   /* Used for auxiliary surface formats */
   ISL_TXC_HIZ,
   ISL_TXC_MCS,
   ISL_TXC_CCS,
};

/**
 * @brief Hardware tile mode
 *
 * WARNING: These values differ from the hardware enum values, which are
 * unstable across hardware generations.
 *
 * Note that legacy Y tiling is ISL_TILING_Y0 instead of ISL_TILING_Y, to
 * clearly distinguish it from Yf and Ys.
 */
enum isl_tiling {
   ISL_TILING_LINEAR = 0,
   ISL_TILING_W,
   ISL_TILING_X,
   ISL_TILING_Y0, /**< Legacy Y tiling */
   ISL_TILING_Yf, /**< Standard 4K tiling. The 'f' means "four". */
   ISL_TILING_Ys, /**< Standard 64K tiling. The 's' means "sixty-four". */
   ISL_TILING_HIZ, /**< Tiling format for HiZ surfaces */
   ISL_TILING_CCS, /**< Tiling format for CCS surfaces */
};

/**
 * @defgroup Tiling Flags
 * @{
 */
typedef uint32_t isl_tiling_flags_t;
#define ISL_TILING_LINEAR_BIT             (1u << ISL_TILING_LINEAR)
#define ISL_TILING_W_BIT                  (1u << ISL_TILING_W)
#define ISL_TILING_X_BIT                  (1u << ISL_TILING_X)
#define ISL_TILING_Y0_BIT                 (1u << ISL_TILING_Y0)
#define ISL_TILING_Yf_BIT                 (1u << ISL_TILING_Yf)
#define ISL_TILING_Ys_BIT                 (1u << ISL_TILING_Ys)
#define ISL_TILING_HIZ_BIT                (1u << ISL_TILING_HIZ)
#define ISL_TILING_CCS_BIT                (1u << ISL_TILING_CCS)
#define ISL_TILING_ANY_MASK               (~0u)
#define ISL_TILING_NON_LINEAR_MASK        (~ISL_TILING_LINEAR_BIT)

/** Any Y tiling, including legacy Y tiling. */
#define ISL_TILING_ANY_Y_MASK             (ISL_TILING_Y0_BIT | \
                                           ISL_TILING_Yf_BIT | \
                                           ISL_TILING_Ys_BIT)

/** The Skylake BSpec refers to Yf and Ys as "standard tiling formats". */
#define ISL_TILING_STD_Y_MASK             (ISL_TILING_Yf_BIT | \
                                           ISL_TILING_Ys_BIT)
/** @} */

/**
 * @brief Logical dimension of surface.
 *
 * Note: There is no dimension for cube map surfaces. ISL interprets cube maps
 * as 2D array surfaces.
 */
enum isl_surf_dim {
   ISL_SURF_DIM_1D,
   ISL_SURF_DIM_2D,
   ISL_SURF_DIM_3D,
};

/**
 * @brief Physical layout of the surface's dimensions.
 */
enum isl_dim_layout {
   /**
    * For details, see the G35 PRM >> Volume 1: Graphics Core >> Section
    * 6.17.3: 2D Surfaces.
    *
    * On many gens, 1D surfaces share the same layout as 2D surfaces.  From
    * the G35 PRM >> Volume 1: Graphics Core >> Section 6.17.2: 1D Surfaces:
    *
    *    One-dimensional surfaces are identical to 2D surfaces with height of
    *    one.
    *
    * @invariant isl_surf::phys_level0_sa::depth == 1
    */
   ISL_DIM_LAYOUT_GEN4_2D,

   /**
    * For details, see the G35 PRM >> Volume 1: Graphics Core >> Section
    * 6.17.5: 3D Surfaces.
    *
    * @invariant isl_surf::phys_level0_sa::array_len == 1
    */
   ISL_DIM_LAYOUT_GEN4_3D,

   /**
    * Special layout used for HiZ and stencil on Sandy Bridge to work around
    * the hardware's lack of mipmap support.  On gen6, HiZ and stencil buffers
    * work the same as on gen7+ except that they don't technically support
    * mipmapping.  That does not, however, stop us from doing it.  As far as
    * Sandy Bridge hardware is concerned, HiZ and stencil always operates on a
    * single miplevel 2D (possibly array) image.  The dimensions of that image
    * are NOT minified.
    *
    * In order to implement HiZ and stencil on Sandy Bridge, we create one
    * full-sized 2D (possibly array) image for every LOD with every image
    * aligned to a page boundary.  When the surface is used with the stencil
    * or HiZ hardware, we manually offset to the image for the given LOD.
    *
    * As a memory saving measure,  we pretend that the width of each miplevel
    * is minified and we place LOD1 and above below LOD0 but horizontally
    * adjacent to each other.  When considered as full-sized images, LOD1 and
    * above technically overlap.  However, since we only write to part of that
    * image, the hardware will never notice the overlap.
    *
    * This layout looks something like this:
    *
    *   +---------+
    *   |         |
    *   |         |
    *   +---------+
    *   |         |
    *   |         |
    *   +---------+
    *
    *   +----+ +-+ .
    *   |    | +-+
    *   +----+
    *
    *   +----+ +-+ .
    *   |    | +-+
    *   +----+
    */
   ISL_DIM_LAYOUT_GEN6_STENCIL_HIZ,

   /**
    * For details, see the Skylake BSpec >> Memory Views >> Common Surface
    * Formats >> Surface Layout and Tiling >> » 1D Surfaces.
    */
   ISL_DIM_LAYOUT_GEN9_1D,
};

enum isl_aux_usage {
   /** No Auxiliary surface is used */
   ISL_AUX_USAGE_NONE,

   /** The primary surface is a depth surface and the auxiliary surface is HiZ */
   ISL_AUX_USAGE_HIZ,

   /** The auxiliary surface is an MCS
    *
    * @invariant isl_surf::samples > 1
    */
   ISL_AUX_USAGE_MCS,

   /** The auxiliary surface is a fast-clear-only compression surface
    *
    * @invariant isl_surf::samples == 1
    */
   ISL_AUX_USAGE_CCS_D,

   /** The auxiliary surface provides full lossless color compression
    *
    * @invariant isl_surf::samples == 1
    */
   ISL_AUX_USAGE_CCS_E,
};

/**
 * Enum for keeping track of the state an auxiliary compressed surface.
 *
 * For any given auxiliary surface compression format (HiZ, CCS, or MCS), any
 * given slice (lod + array layer) can be in one of the six states described
 * by this enum.  Draw and resolve operations may cause the slice to change
 * from one state to another.  The six valid states are:
 *
 *    1) Clear:  In this state, each block in the auxiliary surface contains a
 *       magic value that indicates that the block is in the clear state.  If
 *       a block is in the clear state, it's values in the primary surface are
 *       ignored and the color of the samples in the block is taken either the
 *       RENDER_SURFACE_STATE packet for color or 3DSTATE_CLEAR_PARAMS for
 *       depth.  Since neither the primary surface nor the auxiliary surface
 *       contains the clear value, the surface can be cleared to a different
 *       color by simply changing the clear color without modifying either
 *       surface.
 *
 *    2) Partial Clear:  In this state, each block in the auxiliary surface
 *       contains either the magic clear or pass-through value.  See Clear and
 *       Pass-through for more details.
 *
 *    3) Compressed w/ Clear:  In this state, neither the auxiliary surface
 *       nor the primary surface has a complete representation of the data.
 *       Instead, both surfaces must be used together or else rendering
 *       corruption may occur.  Depending on the auxiliary compression format
 *       and the data, any given block in the primary surface may contain all,
 *       some, or none of the data required to reconstruct the actual sample
 *       values.  Blocks may also be in the clear state (see Clear) and have
 *       their value taken from outside the surface.
 *
 *    4) Compressed w/o Clear:  This state is identical to the state above
 *       except that no blocks are in the clear state.  In this state, all of
 *       the data required to reconstruct the final sample values is contained
 *       in the auxiliary and primary surface and the clear value is not
 *       considered.
 *
 *    5) Resolved:  In this state, the primary surface contains 100% of the
 *       data.  The auxiliary surface is also valid so the surface can be
 *       validly used with or without aux enabled.  The auxiliary surface may,
 *       however, contain non-trivial data and any update to the primary
 *       surface with aux disabled will cause the two to get out of sync.
 *
 *    6) Pass-through:  In this state, the primary surface contains 100% of the
 *       data and every block in the auxiliary surface contains a magic value
 *       which indicates that the auxiliary surface should be ignored and the
 *       only the primary surface should be considered.  Updating the primary
 *       surface without aux works fine and can be done repeatedly in this
 *       mode.  Writing to a surface in pass-through mode with aux enabled may
 *       cause the auxiliary buffer to contain non-trivial data and no longer
 *       be in the pass-through state.
 *
 *    7) Aux Invalid:  In this state, the primary surface contains 100% of the
 *       data and the auxiliary surface is completely bogus.  Any attempt to
 *       use the auxiliary surface is liable to result in rendering
 *       corruption.  The only thing that one can do to re-enable aux once
 *       this state is reached is to use an ambiguate pass to transition into
 *       the pass-through state.
 *
 * Drawing with or without aux enabled may implicitly cause the surface to
 * transition between these states.  There are also four types of auxiliary
 * compression operations which cause an explicit transition which are
 * described by the isl_aux_op enum below.
 *
 * Not all operations are valid or useful in all states.  The diagram below
 * contains a complete description of the states and all valid and useful
 * transitions except clear.
 *
 *   Draw w/ Aux
 *   +----------+
 *   |          |
 *   |       +-------------+    Draw w/ Aux     +-------------+
 *   +------>| Compressed  |<-------------------|    Clear    |
 *           |  w/ Clear   |----->----+         |             |
 *           +-------------+          |         +-------------+
 *                  |  /|\            |            |   |
 *                  |   |             |            |   |
 *                  |   |             +------<-----+   |  Draw w/
 *                  |   |             |                | Clear Only
 *                  |   |      Full   |                |   +----------+
 *          Partial |   |     Resolve |               \|/  |          |
 *          Resolve |   |             |         +-------------+       |
 *                  |   |             |         |   Partial   |<------+
 *                  |   |             |         |    Clear    |<----------+
 *                  |   |             |         +-------------+           |
 *                  |   |             |                |                  |
 *                  |   |             +------>---------+  Full            |
 *                  |   |                              | Resolve          |
 *   Draw w/ aux    |   |   Partial Fast Clear         |                  |
 *   +----------+   |   +--------------------------+   |                  |
 *   |          |  \|/                             |  \|/                 |
 *   |       +-------------+    Full Resolve    +-------------+           |
 *   +------>| Compressed  |------------------->|  Resolved   |           |
 *           |  w/o Clear  |<-------------------|             |           |
 *           +-------------+    Draw w/ Aux     +-------------+           |
 *                 /|\                             |   |                  |
 *                  |  Draw                        |   |  Draw            |
 *                  | w/ Aux                       |   | w/o Aux          |
 *                  |            Ambiguate         |   |                  |
 *                  |   +--------------------------+   |                  |
 *   Draw w/o Aux   |   |                              |   Draw w/o Aux   |
 *   +----------+   |   |                              |   +----------+   |
 *   |          |   |  \|/                            \|/  |          |   |
 *   |       +-------------+     Ambiguate      +-------------+       |   |
 *   +------>|    Pass-    |<-------------------|     Aux     |<------+   |
 *   +------>|   through   |                    |   Invalid   |           |
 *   |       +-------------+                    +-------------+           |
 *   |          |   |                                                     |
 *   +----------+   +-----------------------------------------------------+
 *     Draw w/                       Partial Fast Clear
 *    Clear Only
 *
 *
 * While the above general theory applies to all forms of auxiliary
 * compression on Intel hardware, not all states and operations are available
 * on all compression types.  However, each of the auxiliary states and
 * operations can be fairly easily mapped onto the above diagram:
 *
 * HiZ:     Hierarchical depth compression is capable of being in any of the
 *          states above.  Hardware provides three HiZ operations: "Depth
 *          Clear", "Depth Resolve", and "HiZ Resolve" which map to "Fast
 *          Clear", "Full Resolve", and "Ambiguate" respectively.  The
 *          hardware provides no HiZ partial resolve operation so the only way
 *          to get into the "Compressed w/o Clear" state is to render with HiZ
 *          when the surface is in the resolved or pass-through states.
 *
 * MCS:     Multisample compression is technically capable of being in any of
 *          the states above except that most of them aren't useful.  Both the
 *          render engine and the sampler support MCS compression and, apart
 *          from clear color, MCS is format-unaware so we leave the surface
 *          compressed 100% of the time.  The hardware provides no MCS
 *          operations.
 *
 * CCS_D:   Single-sample fast-clears (also called CCS_D in ISL) are one of
 *          the simplest forms of compression since they don't do anything
 *          beyond clear color tracking.  They really only support three of
 *          the six states: Clear, Partial Clear, and Pass-through.  The
 *          only CCS_D operation is "Resolve" which maps to a full resolve
 *          followed by an ambiguate.
 *
 * CCS_E:   Single-sample render target compression (also called CCS_E in ISL)
 *          is capable of being in almost all of the above states.  THe only
 *          exception is that it does not have separate resolved and pass-
 *          through states.  Instead, the CCS_E full resolve operation does
 *          both a resolve and an ambiguate so it goes directly into the
 *          pass-through state.  CCS_E also provides fast clear and partial
 *          resolve operations which work as described above.
 *
 *          While it is technically possible to perform a CCS_E ambiguate, it
 *          is not provided by Sky Lake hardware so we choose to avoid the aux
 *          invalid state.  If the aux invalid state were determined to be
 *          useful, a CCS ambiguate could be done by carefully rendering to
 *          the CCS and filling it with zeros.
 */
enum isl_aux_state {
   ISL_AUX_STATE_CLEAR = 0,
   ISL_AUX_STATE_PARTIAL_CLEAR,
   ISL_AUX_STATE_COMPRESSED_CLEAR,
   ISL_AUX_STATE_COMPRESSED_NO_CLEAR,
   ISL_AUX_STATE_RESOLVED,
   ISL_AUX_STATE_PASS_THROUGH,
   ISL_AUX_STATE_AUX_INVALID,
};

/**
 * Enum which describes explicit aux transition operations.
 */
enum isl_aux_op {
   ISL_AUX_OP_NONE,

   /** Fast Clear
    *
    * This operation writes the magic "clear" value to the auxiliary surface.
    * This operation will safely transition any slice of a surface from any
    * state to the clear state so long as the entire slice is fast cleared at
    * once.  A fast clear that only covers part of a slice of a surface is
    * called a partial fast clear.
    */
   ISL_AUX_OP_FAST_CLEAR,

   /** Full Resolve
    *
    * This operation combines the auxiliary surface data with the primary
    * surface data and writes the result to the primary.  For HiZ, the docs
    * call this a depth resolve.  For CCS, the hardware full resolve operation
    * does both a full resolve and an ambiguate so it actually takes you all
    * the way to the pass-through state.
    */
   ISL_AUX_OP_FULL_RESOLVE,

   /** Partial Resolve
    *
    * This operation considers blocks which are in the "clear" state and
    * writes the clear value directly into the primary or auxiliary surface.
    * Once this operation completes, the surface is still compressed but no
    * longer references the clear color.  This operation is only available
    * for CCS_E.
    */
   ISL_AUX_OP_PARTIAL_RESOLVE,

   /** Ambiguate
    *
    * This operation throws away the current auxiliary data and replaces it
    * with the magic pass-through value.  If an ambiguate operation is
    * performed when the primary surface does not contain 100% of the data,
    * data will be lost.  This operation is only implemented in hardware for
    * depth where it is called a HiZ resolve.
    */
   ISL_AUX_OP_AMBIGUATE,
};

/* TODO(chadv): Explain */
enum isl_array_pitch_span {
   ISL_ARRAY_PITCH_SPAN_FULL,
   ISL_ARRAY_PITCH_SPAN_COMPACT,
};

/**
 * @defgroup Surface Usage
 * @{
 */
typedef uint64_t isl_surf_usage_flags_t;
#define ISL_SURF_USAGE_RENDER_TARGET_BIT       (1u << 0)
#define ISL_SURF_USAGE_DEPTH_BIT               (1u << 1)
#define ISL_SURF_USAGE_STENCIL_BIT             (1u << 2)
#define ISL_SURF_USAGE_TEXTURE_BIT             (1u << 3)
#define ISL_SURF_USAGE_CUBE_BIT                (1u << 4)
#define ISL_SURF_USAGE_DISABLE_AUX_BIT         (1u << 5)
#define ISL_SURF_USAGE_DISPLAY_BIT             (1u << 6)
#define ISL_SURF_USAGE_DISPLAY_ROTATE_90_BIT   (1u << 7)
#define ISL_SURF_USAGE_DISPLAY_ROTATE_180_BIT  (1u << 8)
#define ISL_SURF_USAGE_DISPLAY_ROTATE_270_BIT  (1u << 9)
#define ISL_SURF_USAGE_DISPLAY_FLIP_X_BIT      (1u << 10)
#define ISL_SURF_USAGE_DISPLAY_FLIP_Y_BIT      (1u << 11)
#define ISL_SURF_USAGE_STORAGE_BIT             (1u << 12)
#define ISL_SURF_USAGE_HIZ_BIT                 (1u << 13)
#define ISL_SURF_USAGE_MCS_BIT                 (1u << 14)
#define ISL_SURF_USAGE_CCS_BIT                 (1u << 15)
/** @} */

/**
 * @defgroup Channel Mask
 *
 * These #define values are chosen to match the values of
 * RENDER_SURFACE_STATE::Color Buffer Component Write Disables
 *
 * @{
 */
typedef uint8_t isl_channel_mask_t;
#define ISL_CHANNEL_BLUE_BIT  (1 << 0)
#define ISL_CHANNEL_GREEN_BIT (1 << 1)
#define ISL_CHANNEL_RED_BIT   (1 << 2)
#define ISL_CHANNEL_ALPHA_BIT (1 << 3)
/** @} */

/**
 * @brief A channel select (also known as texture swizzle) value
 */
enum isl_channel_select {
   ISL_CHANNEL_SELECT_ZERO = 0,
   ISL_CHANNEL_SELECT_ONE = 1,
   ISL_CHANNEL_SELECT_RED = 4,
   ISL_CHANNEL_SELECT_GREEN = 5,
   ISL_CHANNEL_SELECT_BLUE = 6,
   ISL_CHANNEL_SELECT_ALPHA = 7,
};

/**
 * Identical to VkSampleCountFlagBits.
 */
enum isl_sample_count {
   ISL_SAMPLE_COUNT_1_BIT     = 1u,
   ISL_SAMPLE_COUNT_2_BIT     = 2u,
   ISL_SAMPLE_COUNT_4_BIT     = 4u,
   ISL_SAMPLE_COUNT_8_BIT     = 8u,
   ISL_SAMPLE_COUNT_16_BIT    = 16u,
};
typedef uint32_t isl_sample_count_mask_t;

/**
 * @brief Multisample Format
 */
enum isl_msaa_layout {
   /**
    * @brief Suface is single-sampled.
    */
   ISL_MSAA_LAYOUT_NONE,

   /**
    * @brief [SNB+] Interleaved Multisample Format
    *
    * In this format, multiple samples are interleaved into each cacheline.
    * In other words, the sample index is swizzled into the low 6 bits of the
    * surface's virtual address space.
    *
    * For example, suppose the surface is legacy Y tiled, is 4x multisampled,
    * and its pixel format is 32bpp. Then the first cacheline is arranged
    * thus:
    *
    *    (0,0,0) (0,1,0)   (0,0,1) (1,0,1)
    *    (1,0,0) (1,1,0)   (0,1,1) (1,1,1)
    *
    *    (0,0,2) (1,0,2)   (0,0,3) (1,0,3)
    *    (0,1,2) (1,1,2)   (0,1,3) (1,1,3)
    *
    * The hardware docs refer to this format with multiple terms.  In
    * Sandybridge, this is the only multisample format; so no term is used.
    * The Ivybridge docs refer to surfaces in this format as IMS (Interleaved
    * Multisample Surface). Later hardware docs additionally refer to this
    * format as MSFMT_DEPTH_STENCIL (because the format is deprecated for
    * color surfaces).
    *
    * See the Sandybridge PRM, Volume 4, Part 1, Section 2.7 "Multisampled
    * Surface Behavior".
    *
    * See the Ivybridge PRM, Volume 1, Part 1, Section 6.18.4.1 "Interleaved
    * Multisampled Surfaces".
    */
   ISL_MSAA_LAYOUT_INTERLEAVED,

   /**
    * @brief [IVB+] Array Multisample Format
    *
    * In this format, the surface's physical layout resembles that of a
    * 2D array surface.
    *
    * Suppose the multisample surface's logical extent is (w, h) and its
    * sample count is N. Then surface's physical extent is the same as
    * a singlesample 2D surface whose logical extent is (w, h) and array
    * length is N.  Array slice `i` contains the pixel values for sample
    * index `i`.
    *
    * The Ivybridge docs refer to surfaces in this format as UMS
    * (Uncompressed Multsample Layout) and CMS (Compressed Multisample
    * Surface). The Broadwell docs additionally refer to this format as
    * MSFMT_MSS (MSS=Multisample Surface Storage).
    *
    * See the Broadwell PRM, Volume 5 "Memory Views", Section "Uncompressed
    * Multisample Surfaces".
    *
    * See the Broadwell PRM, Volume 5 "Memory Views", Section "Compressed
    * Multisample Surfaces".
    */
   ISL_MSAA_LAYOUT_ARRAY,
};


struct isl_device {
   const struct gen_device_info *info;
   bool use_separate_stencil;
   bool has_bit6_swizzling;

   /**
    * Describes the layout of a RENDER_SURFACE_STATE structure for the
    * current gen.
    */
   struct {
      uint8_t size;
      uint8_t align;
      uint8_t addr_offset;
      uint8_t aux_addr_offset;

      /* Rounded up to the nearest dword to simplify GPU memcpy operations. */

      /* size of the state buffer used to store the clear color + extra
       * additional space used by the hardware */
      uint8_t clear_color_state_size;
      uint8_t clear_color_state_offset;
      /* size of the clear color itself - used to copy it to/from a BO */
      uint8_t clear_value_size;
      uint8_t clear_value_offset;
   } ss;

   /**
    * Describes the layout of the depth/stencil/hiz commands as emitted by
    * isl_emit_depth_stencil_hiz.
    */
   struct {
      uint8_t size;
      uint8_t depth_offset;
      uint8_t stencil_offset;
      uint8_t hiz_offset;
   } ds;
};

struct isl_extent2d {
   union { uint32_t w, width; };
   union { uint32_t h, height; };
};

struct isl_extent3d {
   union { uint32_t w, width; };
   union { uint32_t h, height; };
   union { uint32_t d, depth; };
};

struct isl_extent4d {
   union { uint32_t w, width; };
   union { uint32_t h, height; };
   union { uint32_t d, depth; };
   union { uint32_t a, array_len; };
};

struct isl_channel_layout {
   enum isl_base_type type;
   uint8_t start_bit; /**< Bit at which this channel starts */
   uint8_t bits; /**< Size in bits */
};

/**
 * Each format has 3D block extent (width, height, depth). The block extent of
 * compressed formats is that of the format's compression block. For example,
 * the block extent of ISL_FORMAT_ETC2_RGB8 is (w=4, h=4, d=1).  The block
 * extent of uncompressed pixel formats, such as ISL_FORMAT_R8G8B8A8_UNORM, is
 * is (w=1, h=1, d=1).
 */
struct isl_format_layout {
   enum isl_format format;
   const char *name;

   uint16_t bpb; /**< Bits per block */
   uint8_t bw; /**< Block width, in pixels */
   uint8_t bh; /**< Block height, in pixels */
   uint8_t bd; /**< Block depth, in pixels */

   union {
      struct {
         struct isl_channel_layout r; /**< Red channel */
         struct isl_channel_layout g; /**< Green channel */
         struct isl_channel_layout b; /**< Blue channel */
         struct isl_channel_layout a; /**< Alpha channel */
         struct isl_channel_layout l; /**< Luminance channel */
         struct isl_channel_layout i; /**< Intensity channel */
         struct isl_channel_layout p; /**< Palette channel */
      } channels;
      struct isl_channel_layout channels_array[7];
   };

   enum isl_colorspace colorspace;
   enum isl_txc txc;
};

struct isl_tile_info {
   enum isl_tiling tiling;

   /* The size (in bits per block) of a single surface element
    *
    * For surfaces with power-of-two formats, this is the same as
    * isl_format_layout::bpb.  For non-power-of-two formats it may be smaller.
    * The logical_extent_el field is in terms of elements of this size.
    *
    * For example, consider ISL_FORMAT_R32G32B32_FLOAT for which
    * isl_format_layout::bpb is 96 (a non-power-of-two).  In this case, none
    * of the tiling formats can actually hold an integer number of 96-bit
    * surface elements so isl_tiling_get_info returns an isl_tile_info for a
    * 32-bit element size.  It is the responsibility of the caller to
    * recognize that 32 != 96 ad adjust accordingly.  For instance, to compute
    * the width of a surface in tiles, you would do:
    *
    * width_tl = DIV_ROUND_UP(width_el * (format_bpb / tile_info.format_bpb),
    *                         tile_info.logical_extent_el.width);
    */
   uint32_t format_bpb;

   /** The logical size of the tile in units of format_bpb size elements
    *
    * This field determines how a given surface is cut up into tiles.  It is
    * used to compute the size of a surface in tiles and can be used to
    * determine the location of the tile containing any given surface element.
    * The exact value of this field depends heavily on the bits-per-block of
    * the format being used.
    */
   struct isl_extent2d logical_extent_el;

   /** The physical size of the tile in bytes and rows of bytes
    *
    * This field determines how the tiles of a surface are physically layed
    * out in memory.  The logical and physical tile extent are frequently the
    * same but this is not always the case.  For instance, a W-tile (which is
    * always used with ISL_FORMAT_R8) has a logical size of 64el x 64el but
    * its physical size is 128B x 32rows, the same as a Y-tile.
    *
    * @see isl_surf::row_pitch_B
    */
   struct isl_extent2d phys_extent_B;
};

/**
 * Metadata about a DRM format modifier.
 */
struct isl_drm_modifier_info {
   uint64_t modifier;

   /** Text name of the modifier */
   const char *name;

   /** ISL tiling implied by this modifier */
   enum isl_tiling tiling;

   /** ISL aux usage implied by this modifier */
   enum isl_aux_usage aux_usage;

   /** Whether or not this modifier supports clear color */
   bool supports_clear_color;
};

/**
 * @brief Input to surface initialization
 *
 * @invariant width >= 1
 * @invariant height >= 1
 * @invariant depth >= 1
 * @invariant levels >= 1
 * @invariant samples >= 1
 * @invariant array_len >= 1
 *
 * @invariant if 1D then height == 1 and depth == 1 and samples == 1
 * @invariant if 2D then depth == 1
 * @invariant if 3D then array_len == 1 and samples == 1
 */
struct isl_surf_init_info {
   enum isl_surf_dim dim;
   enum isl_format format;

   uint32_t width;
   uint32_t height;
   uint32_t depth;
   uint32_t levels;
   uint32_t array_len;
   uint32_t samples;

   /** Lower bound for isl_surf::alignment, in bytes. */
   uint32_t min_alignment_B;

   /**
    * Exact value for isl_surf::row_pitch. Ignored if zero.  isl_surf_init()
    * will fail if this is misaligned or out of bounds.
    */
   uint32_t row_pitch_B;

   isl_surf_usage_flags_t usage;

   /** Flags that alter how ISL selects isl_surf::tiling.  */
   isl_tiling_flags_t tiling_flags;
};

struct isl_surf {
   enum isl_surf_dim dim;
   enum isl_dim_layout dim_layout;
   enum isl_msaa_layout msaa_layout;
   enum isl_tiling tiling;
   enum isl_format format;

   /**
    * Alignment of the upper-left sample of each subimage, in units of surface
    * elements.
    */
   struct isl_extent3d image_alignment_el;

   /**
    * Logical extent of the surface's base level, in units of pixels.  This is
    * identical to the extent defined in isl_surf_init_info.
    */
   struct isl_extent4d logical_level0_px;

   /**
    * Physical extent of the surface's base level, in units of physical
    * surface samples and aligned to the format's compression block.
    *
    * Consider isl_dim_layout as an operator that transforms a logical surface
    * layout to a physical surface layout. Then
    *
    *    logical_layout := (isl_surf::dim, isl_surf::logical_level0_px)
    *    isl_surf::phys_level0_sa := isl_surf::dim_layout * logical_layout
    */
   struct isl_extent4d phys_level0_sa;

   uint32_t levels;
   uint32_t samples;

   /** Total size of the surface, in bytes. */
   uint64_t size_B;

   /** Required alignment for the surface's base address. */
   uint32_t alignment_B;

   /**
    * The interpretation of this field depends on the value of
    * isl_tile_info::physical_extent_B.  In particular, the width of the
    * surface in tiles is row_pitch_B / isl_tile_info::physical_extent_B.width
    * and the distance in bytes between vertically adjacent tiles in the image
    * is given by row_pitch_B * isl_tile_info::physical_extent_B.height.
    *
    * For linear images where isl_tile_info::physical_extent_B.height == 1,
    * this cleanly reduces to being the distance, in bytes, between vertically
    * adjacent surface elements.
    *
    * @see isl_tile_info::phys_extent_B;
    */
   uint32_t row_pitch_B;

   /**
    * Pitch between physical array slices, in rows of surface elements.
    */
   uint32_t array_pitch_el_rows;

   enum isl_array_pitch_span array_pitch_span;

   /** Copy of isl_surf_init_info::usage. */
   isl_surf_usage_flags_t usage;
};

struct isl_swizzle {
   enum isl_channel_select r:4;
   enum isl_channel_select g:4;
   enum isl_channel_select b:4;
   enum isl_channel_select a:4;
};

#define ISL_SWIZZLE(R, G, B, A) ((struct isl_swizzle) { \
      .r = ISL_CHANNEL_SELECT_##R, \
      .g = ISL_CHANNEL_SELECT_##G, \
      .b = ISL_CHANNEL_SELECT_##B, \
      .a = ISL_CHANNEL_SELECT_##A, \
   })

#define ISL_SWIZZLE_IDENTITY ISL_SWIZZLE(RED, GREEN, BLUE, ALPHA)

struct isl_view {
   /**
    * Indicates the usage of the particular view
    *
    * Normally, this is one bit.  However, for a cube map texture, it
    * should be ISL_SURF_USAGE_TEXTURE_BIT | ISL_SURF_USAGE_CUBE_BIT.
    */
   isl_surf_usage_flags_t usage;

   /**
    * The format to use in the view
    *
    * This may differ from the format of the actual isl_surf but must have
    * the same block size.
    */
   enum isl_format format;

   uint32_t base_level;
   uint32_t levels;

   /**
    * Base array layer
    *
    * For cube maps, both base_array_layer and array_len should be
    * specified in terms of 2-D layers and must be a multiple of 6.
    *
    * 3-D textures are effectively treated as 2-D arrays when used as a
    * storage image or render target.  If `usage` contains
    * ISL_SURF_USAGE_RENDER_TARGET_BIT or ISL_SURF_USAGE_STORAGE_BIT then
    * base_array_layer and array_len are applied.  If the surface is only used
    * for texturing, they are ignored.
    */
   uint32_t base_array_layer;

   /**
    * Array Length
    *
    * Indicates the number of array elements starting at  Base Array Layer.
    */
   uint32_t array_len;

   struct isl_swizzle swizzle;
};

union isl_color_value {
   float f32[4];
   uint32_t u32[4];
   int32_t i32[4];
};

struct isl_surf_fill_state_info {
   const struct isl_surf *surf;
   const struct isl_view *view;

   /**
    * The address of the surface in GPU memory.
    */
   uint64_t address;

   /**
    * The Memory Object Control state for the filled surface state.
    *
    * The exact format of this value depends on hardware generation.
    */
   uint32_t mocs;

   /**
    * The auxilary surface or NULL if no auxilary surface is to be used.
    */
   const struct isl_surf *aux_surf;
   enum isl_aux_usage aux_usage;
   uint64_t aux_address;

   /**
    * The clear color for this surface
    *
    * Valid values depend on hardware generation.
    */
   union isl_color_value clear_color;

   /**
    * Send only the clear value address
    *
    * If set, we only pass the clear address to the GPU and it will fetch it
    * from wherever it is.
    */
   bool use_clear_address;
   uint64_t clear_address;

   /**
    * Surface write disables for gen4-5
    */
   isl_channel_mask_t write_disables;

   /* Intra-tile offset */
   uint16_t x_offset_sa, y_offset_sa;
};

struct isl_buffer_fill_state_info {
   /**
    * The address of the surface in GPU memory.
    */
   uint64_t address;

   /**
    * The size of the buffer
    */
   uint64_t size_B;

   /**
    * The Memory Object Control state for the filled surface state.
    *
    * The exact format of this value depends on hardware generation.
    */
   uint32_t mocs;

   /**
    * The format to use in the surface state
    *
    * This may differ from the format of the actual isl_surf but have the
    * same block size.
    */
   enum isl_format format;

   uint32_t stride_B;
};

struct isl_depth_stencil_hiz_emit_info {
   /**
    * The depth surface
    */
   const struct isl_surf *depth_surf;

   /**
    * The stencil surface
    *
    * If separate stencil is not available, this must point to the same
    * isl_surf as depth_surf.
    */
   const struct isl_surf *stencil_surf;

   /**
    * The view into the depth and stencil surfaces.
    *
    * This view applies to both surfaces simultaneously.
    */
   const struct isl_view *view;

   /**
    * The address of the depth surface in GPU memory
    */
   uint64_t depth_address;

   /**
    * The address of the stencil surface in GPU memory
    *
    * If separate stencil is not available, this must have the same value as
    * depth_address.
    */
   uint64_t stencil_address;

   /**
    * The Memory Object Control state for depth and stencil buffers
    *
    * Both depth and stencil will get the same MOCS value.  The exact format
    * of this value depends on hardware generation.
    */
   uint32_t mocs;

   /**
    * The HiZ surface or NULL if HiZ is disabled.
    */
   const struct isl_surf *hiz_surf;
   enum isl_aux_usage hiz_usage;
   uint64_t hiz_address;

   /**
    * The depth clear value
    */
   float depth_clear_value;
};

extern const struct isl_format_layout isl_format_layouts[];

void
isl_device_init(struct isl_device *dev,
                const struct gen_device_info *info,
                bool has_bit6_swizzling);

isl_sample_count_mask_t ATTRIBUTE_CONST
isl_device_get_sample_counts(struct isl_device *dev);

static inline const struct isl_format_layout * ATTRIBUTE_CONST
isl_format_get_layout(enum isl_format fmt)
{
   assert(fmt != ISL_FORMAT_UNSUPPORTED);
   assert(fmt < ISL_NUM_FORMATS);
   return &isl_format_layouts[fmt];
}

bool isl_format_is_valid(enum isl_format);

static inline const char * ATTRIBUTE_CONST
isl_format_get_name(enum isl_format fmt)
{
   return isl_format_get_layout(fmt)->name;
}

bool isl_format_supports_rendering(const struct gen_device_info *devinfo,
                                   enum isl_format format);
bool isl_format_supports_alpha_blending(const struct gen_device_info *devinfo,
                                        enum isl_format format);
bool isl_format_supports_sampling(const struct gen_device_info *devinfo,
                                  enum isl_format format);
bool isl_format_supports_filtering(const struct gen_device_info *devinfo,
                                   enum isl_format format);
bool isl_format_supports_vertex_fetch(const struct gen_device_info *devinfo,
                                      enum isl_format format);
bool isl_format_supports_typed_writes(const struct gen_device_info *devinfo,
                                      enum isl_format format);
bool isl_format_supports_typed_reads(const struct gen_device_info *devinfo,
                                     enum isl_format format);
bool isl_format_supports_ccs_d(const struct gen_device_info *devinfo,
                               enum isl_format format);
bool isl_format_supports_ccs_e(const struct gen_device_info *devinfo,
                               enum isl_format format);
bool isl_format_supports_multisampling(const struct gen_device_info *devinfo,
                                       enum isl_format format);

bool isl_formats_are_ccs_e_compatible(const struct gen_device_info *devinfo,
                                      enum isl_format format1,
                                      enum isl_format format2);

bool isl_format_has_unorm_channel(enum isl_format fmt) ATTRIBUTE_CONST;
bool isl_format_has_snorm_channel(enum isl_format fmt) ATTRIBUTE_CONST;
bool isl_format_has_ufloat_channel(enum isl_format fmt) ATTRIBUTE_CONST;
bool isl_format_has_sfloat_channel(enum isl_format fmt) ATTRIBUTE_CONST;
bool isl_format_has_uint_channel(enum isl_format fmt) ATTRIBUTE_CONST;
bool isl_format_has_sint_channel(enum isl_format fmt) ATTRIBUTE_CONST;

static inline bool
isl_format_has_normalized_channel(enum isl_format fmt)
{
   return isl_format_has_unorm_channel(fmt) ||
          isl_format_has_snorm_channel(fmt);
}

static inline bool
isl_format_has_float_channel(enum isl_format fmt)
{
   return isl_format_has_ufloat_channel(fmt) ||
          isl_format_has_sfloat_channel(fmt);
}

static inline bool
isl_format_has_int_channel(enum isl_format fmt)
{
   return isl_format_has_uint_channel(fmt) ||
          isl_format_has_sint_channel(fmt);
}

unsigned isl_format_get_num_channels(enum isl_format fmt);

uint32_t isl_format_get_depth_format(enum isl_format fmt, bool has_stencil);

static inline bool
isl_format_is_compressed(enum isl_format fmt)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(fmt);

   return fmtl->txc != ISL_TXC_NONE;
}

static inline bool
isl_format_has_bc_compression(enum isl_format fmt)
{
   switch (isl_format_get_layout(fmt)->txc) {
   case ISL_TXC_DXT1:
   case ISL_TXC_DXT3:
   case ISL_TXC_DXT5:
      return true;
   case ISL_TXC_NONE:
   case ISL_TXC_FXT1:
   case ISL_TXC_RGTC1:
   case ISL_TXC_RGTC2:
   case ISL_TXC_BPTC:
   case ISL_TXC_ETC1:
   case ISL_TXC_ETC2:
   case ISL_TXC_ASTC:
      return false;

   case ISL_TXC_HIZ:
   case ISL_TXC_MCS:
   case ISL_TXC_CCS:
      unreachable("Should not be called on an aux surface");
   }

   unreachable("bad texture compression mode");
   return false;
}

static inline bool
isl_format_is_yuv(enum isl_format fmt)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(fmt);

   return fmtl->colorspace == ISL_COLORSPACE_YUV;
}

static inline bool
isl_format_block_is_1x1x1(enum isl_format fmt)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(fmt);

   return fmtl->bw == 1 && fmtl->bh == 1 && fmtl->bd == 1;
}

static inline bool
isl_format_is_srgb(enum isl_format fmt)
{
   return isl_format_get_layout(fmt)->colorspace == ISL_COLORSPACE_SRGB;
}

enum isl_format isl_format_srgb_to_linear(enum isl_format fmt);

static inline bool
isl_format_is_rgb(enum isl_format fmt)
{
   if (isl_format_is_yuv(fmt))
      return false;

   const struct isl_format_layout *fmtl = isl_format_get_layout(fmt);

   return fmtl->channels.r.bits > 0 &&
          fmtl->channels.g.bits > 0 &&
          fmtl->channels.b.bits > 0 &&
          fmtl->channels.a.bits == 0;
}

static inline bool
isl_format_is_rgbx(enum isl_format fmt)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(fmt);

   return fmtl->channels.r.bits > 0 &&
          fmtl->channels.g.bits > 0 &&
          fmtl->channels.b.bits > 0 &&
          fmtl->channels.a.bits > 0 &&
          fmtl->channels.a.type == ISL_VOID;
}

enum isl_format isl_format_rgb_to_rgba(enum isl_format rgb) ATTRIBUTE_CONST;
enum isl_format isl_format_rgb_to_rgbx(enum isl_format rgb) ATTRIBUTE_CONST;
enum isl_format isl_format_rgbx_to_rgba(enum isl_format rgb) ATTRIBUTE_CONST;

void isl_color_value_pack(const union isl_color_value *value,
                          enum isl_format format,
                          uint32_t *data_out);
void isl_color_value_unpack(union isl_color_value *value,
                            enum isl_format format,
                            const uint32_t *data_in);

bool isl_is_storage_image_format(enum isl_format fmt);

enum isl_format
isl_lower_storage_image_format(const struct gen_device_info *devinfo,
                               enum isl_format fmt);

/* Returns true if this hardware supports typed load/store on a format with
 * the same size as the given format.
 */
bool
isl_has_matching_typed_storage_image_format(const struct gen_device_info *devinfo,
                                            enum isl_format fmt);

static inline bool
isl_tiling_is_any_y(enum isl_tiling tiling)
{
   return (1u << tiling) & ISL_TILING_ANY_Y_MASK;
}

static inline bool
isl_tiling_is_std_y(enum isl_tiling tiling)
{
   return (1u << tiling) & ISL_TILING_STD_Y_MASK;
}

uint32_t
isl_tiling_to_i915_tiling(enum isl_tiling tiling);

enum isl_tiling 
isl_tiling_from_i915_tiling(uint32_t tiling);

const struct isl_drm_modifier_info * ATTRIBUTE_CONST
isl_drm_modifier_get_info(uint64_t modifier);

static inline bool
isl_drm_modifier_has_aux(uint64_t modifier)
{
   return isl_drm_modifier_get_info(modifier)->aux_usage != ISL_AUX_USAGE_NONE;
}

/** Returns the default isl_aux_state for the given modifier.
 *
 * If we have a modifier which supports compression, then the auxiliary data
 * could be in state other than ISL_AUX_STATE_AUX_INVALID.  In particular, it
 * can be in any of the following:
 *
 *  - ISL_AUX_STATE_CLEAR
 *  - ISL_AUX_STATE_PARTIAL_CLEAR
 *  - ISL_AUX_STATE_COMPRESSED_CLEAR
 *  - ISL_AUX_STATE_COMPRESSED_NO_CLEAR
 *  - ISL_AUX_STATE_RESOLVED
 *  - ISL_AUX_STATE_PASS_THROUGH
 *
 * If the modifier does not support fast-clears, then we are guaranteed
 * that the surface is at least partially resolved and the first three not
 * possible.  We return ISL_AUX_STATE_COMPRESSED_CLEAR if the modifier
 * supports fast clears and ISL_AUX_STATE_COMPRESSED_NO_CLEAR if it does not
 * because they are the least common denominator of the set of possible aux
 * states and will yield a valid interpretation of the aux data.
 *
 * For modifiers with no aux support, ISL_AUX_STATE_AUX_INVALID is returned.
 */
static inline enum isl_aux_state
isl_drm_modifier_get_default_aux_state(uint64_t modifier)
{
   const struct isl_drm_modifier_info *mod_info =
      isl_drm_modifier_get_info(modifier);

   if (!mod_info || mod_info->aux_usage == ISL_AUX_USAGE_NONE)
      return ISL_AUX_STATE_AUX_INVALID;

   assert(mod_info->aux_usage == ISL_AUX_USAGE_CCS_E);
   return mod_info->supports_clear_color ? ISL_AUX_STATE_COMPRESSED_CLEAR :
                                           ISL_AUX_STATE_COMPRESSED_NO_CLEAR;
}

struct isl_extent2d ATTRIBUTE_CONST
isl_get_interleaved_msaa_px_size_sa(uint32_t samples);

static inline bool
isl_surf_usage_is_display(isl_surf_usage_flags_t usage)
{
   return usage & ISL_SURF_USAGE_DISPLAY_BIT;
}

static inline bool
isl_surf_usage_is_depth(isl_surf_usage_flags_t usage)
{
   return usage & ISL_SURF_USAGE_DEPTH_BIT;
}

static inline bool
isl_surf_usage_is_stencil(isl_surf_usage_flags_t usage)
{
   return usage & ISL_SURF_USAGE_STENCIL_BIT;
}

static inline bool
isl_surf_usage_is_depth_and_stencil(isl_surf_usage_flags_t usage)
{
   return (usage & ISL_SURF_USAGE_DEPTH_BIT) &&
          (usage & ISL_SURF_USAGE_STENCIL_BIT);
}

static inline bool
isl_surf_usage_is_depth_or_stencil(isl_surf_usage_flags_t usage)
{
   return usage & (ISL_SURF_USAGE_DEPTH_BIT | ISL_SURF_USAGE_STENCIL_BIT);
}

static inline bool
isl_surf_info_is_z16(const struct isl_surf_init_info *info)
{
   return (info->usage & ISL_SURF_USAGE_DEPTH_BIT) &&
          (info->format == ISL_FORMAT_R16_UNORM);
}

static inline bool
isl_surf_info_is_z32_float(const struct isl_surf_init_info *info)
{
   return (info->usage & ISL_SURF_USAGE_DEPTH_BIT) &&
          (info->format == ISL_FORMAT_R32_FLOAT);
}

static inline struct isl_extent2d
isl_extent2d(uint32_t width, uint32_t height)
{
   struct isl_extent2d e = { { 0 } };

   e.width = width;
   e.height = height;

   return e;
}

static inline struct isl_extent3d
isl_extent3d(uint32_t width, uint32_t height, uint32_t depth)
{
   struct isl_extent3d e = { { 0 } };

   e.width = width;
   e.height = height;
   e.depth = depth;

   return e;
}

static inline struct isl_extent4d
isl_extent4d(uint32_t width, uint32_t height, uint32_t depth,
             uint32_t array_len)
{
   struct isl_extent4d e = { { 0 } };

   e.width = width;
   e.height = height;
   e.depth = depth;
   e.array_len = array_len;

   return e;
}

bool isl_color_value_is_zero(union isl_color_value value,
                             enum isl_format format);

bool isl_color_value_is_zero_one(union isl_color_value value,
                                 enum isl_format format);

static inline bool
isl_swizzle_is_identity(struct isl_swizzle swizzle)
{
   return swizzle.r == ISL_CHANNEL_SELECT_RED &&
          swizzle.g == ISL_CHANNEL_SELECT_GREEN &&
          swizzle.b == ISL_CHANNEL_SELECT_BLUE &&
          swizzle.a == ISL_CHANNEL_SELECT_ALPHA;
}

bool
isl_swizzle_supports_rendering(const struct gen_device_info *devinfo,
                               struct isl_swizzle swizzle);

struct isl_swizzle
isl_swizzle_compose(struct isl_swizzle first, struct isl_swizzle second);
struct isl_swizzle
isl_swizzle_invert(struct isl_swizzle swizzle);

#define isl_surf_init(dev, surf, ...) \
   isl_surf_init_s((dev), (surf), \
                   &(struct isl_surf_init_info) {  __VA_ARGS__ });

bool
isl_surf_init_s(const struct isl_device *dev,
                struct isl_surf *surf,
                const struct isl_surf_init_info *restrict info);

void
isl_surf_get_tile_info(const struct isl_surf *surf,
                       struct isl_tile_info *tile_info);

bool
isl_surf_get_hiz_surf(const struct isl_device *dev,
                      const struct isl_surf *surf,
                      struct isl_surf *hiz_surf);

bool
isl_surf_get_mcs_surf(const struct isl_device *dev,
                      const struct isl_surf *surf,
                      struct isl_surf *mcs_surf);

bool
isl_surf_get_ccs_surf(const struct isl_device *dev,
                      const struct isl_surf *surf,
                      struct isl_surf *ccs_surf,
                      uint32_t row_pitch_B /**< Ignored if 0 */);

#define isl_surf_fill_state(dev, state, ...) \
   isl_surf_fill_state_s((dev), (state), \
                         &(struct isl_surf_fill_state_info) {  __VA_ARGS__ });

void
isl_surf_fill_state_s(const struct isl_device *dev, void *state,
                      const struct isl_surf_fill_state_info *restrict info);

#define isl_buffer_fill_state(dev, state, ...) \
   isl_buffer_fill_state_s((dev), (state), \
                           &(struct isl_buffer_fill_state_info) {  __VA_ARGS__ });

void
isl_buffer_fill_state_s(const struct isl_device *dev, void *state,
                        const struct isl_buffer_fill_state_info *restrict info);

void
isl_null_fill_state(const struct isl_device *dev, void *state,
                    struct isl_extent3d size);

#define isl_emit_depth_stencil_hiz(dev, batch, ...) \
   isl_emit_depth_stencil_hiz_s((dev), (batch), \
                                &(struct isl_depth_stencil_hiz_emit_info) {  __VA_ARGS__ })

void
isl_emit_depth_stencil_hiz_s(const struct isl_device *dev, void *batch,
                             const struct isl_depth_stencil_hiz_emit_info *restrict info);

void
isl_surf_fill_image_param(const struct isl_device *dev,
                          struct brw_image_param *param,
                          const struct isl_surf *surf,
                          const struct isl_view *view);

void
isl_buffer_fill_image_param(const struct isl_device *dev,
                            struct brw_image_param *param,
                            enum isl_format format,
                            uint64_t size);

/**
 * Alignment of the upper-left sample of each subimage, in units of surface
 * elements.
 */
static inline struct isl_extent3d
isl_surf_get_image_alignment_el(const struct isl_surf *surf)
{
   return surf->image_alignment_el;
}

/**
 * Alignment of the upper-left sample of each subimage, in units of surface
 * samples.
 */
static inline struct isl_extent3d
isl_surf_get_image_alignment_sa(const struct isl_surf *surf)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(surf->format);

   return isl_extent3d(fmtl->bw * surf->image_alignment_el.w,
                       fmtl->bh * surf->image_alignment_el.h,
                       fmtl->bd * surf->image_alignment_el.d);
}

/**
 * Pitch between vertically adjacent surface elements, in bytes.
 */
static inline uint32_t
isl_surf_get_row_pitch_B(const struct isl_surf *surf)
{
   return surf->row_pitch_B;
}

/**
 * Pitch between vertically adjacent surface elements, in units of surface elements.
 */
static inline uint32_t
isl_surf_get_row_pitch_el(const struct isl_surf *surf)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(surf->format);

   assert(surf->row_pitch_B % (fmtl->bpb / 8) == 0);
   return surf->row_pitch_B / (fmtl->bpb / 8);
}

/**
 * Pitch between physical array slices, in rows of surface elements.
 */
static inline uint32_t
isl_surf_get_array_pitch_el_rows(const struct isl_surf *surf)
{
   return surf->array_pitch_el_rows;
}

/**
 * Pitch between physical array slices, in units of surface elements.
 */
static inline uint32_t
isl_surf_get_array_pitch_el(const struct isl_surf *surf)
{
   return isl_surf_get_array_pitch_el_rows(surf) *
          isl_surf_get_row_pitch_el(surf);
}

/**
 * Pitch between physical array slices, in rows of surface samples.
 */
static inline uint32_t
isl_surf_get_array_pitch_sa_rows(const struct isl_surf *surf)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(surf->format);
   return fmtl->bh * isl_surf_get_array_pitch_el_rows(surf);
}

/**
 * Pitch between physical array slices, in bytes.
 */
static inline uint32_t
isl_surf_get_array_pitch(const struct isl_surf *surf)
{
   return isl_surf_get_array_pitch_sa_rows(surf) * surf->row_pitch_B;
}

/**
 * Calculate the offset, in units of surface samples, to a subimage in the
 * surface.
 *
 * @invariant level < surface levels
 * @invariant logical_array_layer < logical array length of surface
 * @invariant logical_z_offset_px < logical depth of surface at level
 */
void
isl_surf_get_image_offset_sa(const struct isl_surf *surf,
                             uint32_t level,
                             uint32_t logical_array_layer,
                             uint32_t logical_z_offset_px,
                             uint32_t *x_offset_sa,
                             uint32_t *y_offset_sa);

/**
 * Calculate the offset, in units of surface elements, to a subimage in the
 * surface.
 *
 * @invariant level < surface levels
 * @invariant logical_array_layer < logical array length of surface
 * @invariant logical_z_offset_px < logical depth of surface at level
 */
void
isl_surf_get_image_offset_el(const struct isl_surf *surf,
                             uint32_t level,
                             uint32_t logical_array_layer,
                             uint32_t logical_z_offset_px,
                             uint32_t *x_offset_el,
                             uint32_t *y_offset_el);

/**
 * Calculate the offset, in bytes and intratile surface samples, to a
 * subimage in the surface.
 *
 * This is equivalent to calling isl_surf_get_image_offset_el, passing the
 * result to isl_tiling_get_intratile_offset_el, and converting the tile
 * offsets to samples.
 *
 * @invariant level < surface levels
 * @invariant logical_array_layer < logical array length of surface
 * @invariant logical_z_offset_px < logical depth of surface at level
 */
void
isl_surf_get_image_offset_B_tile_sa(const struct isl_surf *surf,
                                    uint32_t level,
                                    uint32_t logical_array_layer,
                                    uint32_t logical_z_offset_px,
                                    uint32_t *offset_B,
                                    uint32_t *x_offset_sa,
                                    uint32_t *y_offset_sa);

/**
 * Create an isl_surf that represents a particular subimage in the surface.
 *
 * The newly created surface will have a single miplevel and array slice.  The
 * surface lives at the returned byte and intratile offsets, in samples.
 *
 * It is safe to call this function with surf == image_surf.
 *
 * @invariant level < surface levels
 * @invariant logical_array_layer < logical array length of surface
 * @invariant logical_z_offset_px < logical depth of surface at level
 */
void
isl_surf_get_image_surf(const struct isl_device *dev,
                        const struct isl_surf *surf,
                        uint32_t level,
                        uint32_t logical_array_layer,
                        uint32_t logical_z_offset_px,
                        struct isl_surf *image_surf,
                        uint32_t *offset_B,
                        uint32_t *x_offset_sa,
                        uint32_t *y_offset_sa);

/**
 * @brief Calculate the intratile offsets to a surface.
 *
 * In @a base_address_offset return the offset from the base of the surface to
 * the base address of the first tile of the subimage. In @a x_offset_B and
 * @a y_offset_rows, return the offset, in units of bytes and rows, from the
 * tile's base to the subimage's first surface element. The x and y offsets
 * are intratile offsets; that is, they do not exceed the boundary of the
 * surface's tiling format.
 */
void
isl_tiling_get_intratile_offset_el(enum isl_tiling tiling,
                                   uint32_t bpb,
                                   uint32_t row_pitch_B,
                                   uint32_t total_x_offset_el,
                                   uint32_t total_y_offset_el,
                                   uint32_t *base_address_offset,
                                   uint32_t *x_offset_el,
                                   uint32_t *y_offset_el);

static inline void
isl_tiling_get_intratile_offset_sa(enum isl_tiling tiling,
                                   enum isl_format format,
                                   uint32_t row_pitch_B,
                                   uint32_t total_x_offset_sa,
                                   uint32_t total_y_offset_sa,
                                   uint32_t *base_address_offset,
                                   uint32_t *x_offset_sa,
                                   uint32_t *y_offset_sa)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(format);

   /* For computing the intratile offsets, we actually want a strange unit
    * which is samples for multisampled surfaces but elements for compressed
    * surfaces.
    */
   assert(total_x_offset_sa % fmtl->bw == 0);
   assert(total_y_offset_sa % fmtl->bh == 0);
   const uint32_t total_x_offset = total_x_offset_sa / fmtl->bw;
   const uint32_t total_y_offset = total_y_offset_sa / fmtl->bh;

   isl_tiling_get_intratile_offset_el(tiling, fmtl->bpb, row_pitch_B,
                                      total_x_offset, total_y_offset,
                                      base_address_offset,
                                      x_offset_sa, y_offset_sa);
   *x_offset_sa *= fmtl->bw;
   *y_offset_sa *= fmtl->bh;
}

/**
 * @brief Get value of 3DSTATE_DEPTH_BUFFER.SurfaceFormat
 *
 * @pre surf->usage has ISL_SURF_USAGE_DEPTH_BIT
 * @pre surf->format must be a valid format for depth surfaces
 */
uint32_t
isl_surf_get_depth_format(const struct isl_device *dev,
                          const struct isl_surf *surf);

#ifdef __cplusplus
}
#endif

#endif /* ISL_H */
