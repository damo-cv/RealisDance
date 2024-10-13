/*
 * Copyright (C) 2016 Intel Corporation
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


/* Instructions, enums and structures for ILK.
 *
 * This file has been generated, do not hand edit.
 */

#ifndef GEN5_PACK_H
#define GEN5_PACK_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#ifndef __gen_validate_value
#define __gen_validate_value(x)
#endif

#ifndef __gen_field_functions
#define __gen_field_functions

#ifdef NDEBUG
#define NDEBUG_UNUSED __attribute__((unused))
#else
#define NDEBUG_UNUSED
#endif

union __gen_value {
   float f;
   uint32_t dw;
};

static inline uint64_t
__gen_mbo(uint32_t start, uint32_t end)
{
   return (~0ull >> (64 - (end - start + 1))) << start;
}

static inline uint64_t
__gen_uint(uint64_t v, uint32_t start, NDEBUG_UNUSED uint32_t end)
{
   __gen_validate_value(v);

#ifndef NDEBUG
   const int width = end - start + 1;
   if (width < 64) {
      const uint64_t max = (1ull << width) - 1;
      assert(v <= max);
   }
#endif

   return v << start;
}

static inline uint64_t
__gen_sint(int64_t v, uint32_t start, uint32_t end)
{
   const int width = end - start + 1;

   __gen_validate_value(v);

#ifndef NDEBUG
   if (width < 64) {
      const int64_t max = (1ll << (width - 1)) - 1;
      const int64_t min = -(1ll << (width - 1));
      assert(min <= v && v <= max);
   }
#endif

   const uint64_t mask = ~0ull >> (64 - width);

   return (v & mask) << start;
}

static inline uint64_t
__gen_offset(uint64_t v, NDEBUG_UNUSED uint32_t start, NDEBUG_UNUSED uint32_t end)
{
   __gen_validate_value(v);
#ifndef NDEBUG
   uint64_t mask = (~0ull >> (64 - (end - start + 1))) << start;

   assert((v & ~mask) == 0);
#endif

   return v;
}

static inline uint32_t
__gen_float(float v)
{
   __gen_validate_value(v);
   return ((union __gen_value) { .f = (v) }).dw;
}

static inline uint64_t
__gen_sfixed(float v, uint32_t start, uint32_t end, uint32_t fract_bits)
{
   __gen_validate_value(v);

   const float factor = (1 << fract_bits);

#ifndef NDEBUG
   const float max = ((1 << (end - start)) - 1) / factor;
   const float min = -(1 << (end - start)) / factor;
   assert(min <= v && v <= max);
#endif

   const int64_t int_val = llroundf(v * factor);
   const uint64_t mask = ~0ull >> (64 - (end - start + 1));

   return (int_val & mask) << start;
}

static inline uint64_t
__gen_ufixed(float v, uint32_t start, NDEBUG_UNUSED uint32_t end, uint32_t fract_bits)
{
   __gen_validate_value(v);

   const float factor = (1 << fract_bits);

#ifndef NDEBUG
   const float max = ((1 << (end - start + 1)) - 1) / factor;
   const float min = 0.0f;
   assert(min <= v && v <= max);
#endif

   const uint64_t uint_val = llroundf(v * factor);

   return uint_val << start;
}

#ifndef __gen_address_type
#error #define __gen_address_type before including this file
#endif

#ifndef __gen_user_data
#error #define __gen_combine_address before including this file
#endif

#undef NDEBUG_UNUSED

#endif


enum GEN5_3D_Prim_Topo_Type {
   _3DPRIM_POINTLIST                    =      1,
   _3DPRIM_LINELIST                     =      2,
   _3DPRIM_LINESTRIP                    =      3,
   _3DPRIM_TRILIST                      =      4,
   _3DPRIM_TRISTRIP                     =      5,
   _3DPRIM_TRIFAN                       =      6,
   _3DPRIM_QUADLIST                     =      7,
   _3DPRIM_QUADSTRIP                    =      8,
   _3DPRIM_LINELIST_ADJ                 =      9,
   _3DPRIM_LINESTRIP_ADJ                =     10,
   _3DPRIM_TRILIST_ADJ                  =     11,
   _3DPRIM_TRISTRIP_ADJ                 =     12,
   _3DPRIM_TRISTRIP_REVERSE             =     13,
   _3DPRIM_POLYGON                      =     14,
   _3DPRIM_RECTLIST                     =     15,
   _3DPRIM_LINELOOP                     =     16,
   _3DPRIM_POINTLIST_BF                 =     17,
   _3DPRIM_LINESTRIP_CONT               =     18,
   _3DPRIM_LINESTRIP_BF                 =     19,
   _3DPRIM_LINESTRIP_CONT_BF            =     20,
   _3DPRIM_TRIFAN_NOSTIPPLE             =     22,
};

enum GEN5_3D_Vertex_Component_Control {
   VFCOMP_NOSTORE                       =      0,
   VFCOMP_STORE_SRC                     =      1,
   VFCOMP_STORE_0                       =      2,
   VFCOMP_STORE_1_FP                    =      3,
   VFCOMP_STORE_1_INT                   =      4,
   VFCOMP_STORE_VID                     =      5,
   VFCOMP_STORE_IID                     =      6,
   VFCOMP_STORE_PID                     =      7,
};

enum GEN5_3D_Color_Buffer_Blend_Function {
   BLENDFUNCTION_ADD                    =      0,
   BLENDFUNCTION_SUBTRACT               =      1,
   BLENDFUNCTION_REVERSE_SUBTRACT       =      2,
   BLENDFUNCTION_MIN                    =      3,
   BLENDFUNCTION_MAX                    =      4,
};

enum GEN5_3D_Color_Buffer_Blend_Factor {
   BLENDFACTOR_ONE                      =      1,
   BLENDFACTOR_SRC_COLOR                =      2,
   BLENDFACTOR_SRC_ALPHA                =      3,
   BLENDFACTOR_DST_ALPHA                =      4,
   BLENDFACTOR_DST_COLOR                =      5,
   BLENDFACTOR_SRC_ALPHA_SATURATE       =      6,
   BLENDFACTOR_CONST_COLOR              =      7,
   BLENDFACTOR_CONST_ALPHA              =      8,
   BLENDFACTOR_SRC1_COLOR               =      9,
   BLENDFACTOR_SRC1_ALPHA               =     10,
   BLENDFACTOR_ZERO                     =     17,
   BLENDFACTOR_INV_SRC_COLOR            =     18,
   BLENDFACTOR_INV_SRC_ALPHA            =     19,
   BLENDFACTOR_INV_DST_ALPHA            =     20,
   BLENDFACTOR_INV_DST_COLOR            =     21,
   BLENDFACTOR_INV_CONST_COLOR          =     23,
   BLENDFACTOR_INV_CONST_ALPHA          =     24,
   BLENDFACTOR_INV_SRC1_COLOR           =     25,
   BLENDFACTOR_INV_SRC1_ALPHA           =     26,
};

enum GEN5_3D_Compare_Function {
   COMPAREFUNCTION_ALWAYS               =      0,
   COMPAREFUNCTION_NEVER                =      1,
   COMPAREFUNCTION_LESS                 =      2,
   COMPAREFUNCTION_EQUAL                =      3,
   COMPAREFUNCTION_LEQUAL               =      4,
   COMPAREFUNCTION_GREATER              =      5,
   COMPAREFUNCTION_NOTEQUAL             =      6,
   COMPAREFUNCTION_GEQUAL               =      7,
};

enum GEN5_3D_Stencil_Operation {
   STENCILOP_KEEP                       =      0,
   STENCILOP_ZERO                       =      1,
   STENCILOP_REPLACE                    =      2,
   STENCILOP_INCRSAT                    =      3,
   STENCILOP_DECRSAT                    =      4,
   STENCILOP_INCR                       =      5,
   STENCILOP_DECR                       =      6,
   STENCILOP_INVERT                     =      7,
};

enum GEN5_3D_Logic_Op_Function {
   LOGICOP_CLEAR                        =      0,
   LOGICOP_NOR                          =      1,
   LOGICOP_AND_INVERTED                 =      2,
   LOGICOP_COPY_INVERTED                =      3,
   LOGICOP_AND_REVERSE                  =      4,
   LOGICOP_INVERT                       =      5,
   LOGICOP_XOR                          =      6,
   LOGICOP_NAND                         =      7,
   LOGICOP_AND                          =      8,
   LOGICOP_EQUIV                        =      9,
   LOGICOP_NOOP                         =     10,
   LOGICOP_OR_INVERTED                  =     11,
   LOGICOP_COPY                         =     12,
   LOGICOP_OR_REVERSE                   =     13,
   LOGICOP_OR                           =     14,
   LOGICOP_SET                          =     15,
};

enum GEN5_Texture_Coordinate_Mode {
   TCM_WRAP                             =      0,
   TCM_MIRROR                           =      1,
   TCM_CLAMP                            =      2,
   TCM_CUBE                             =      3,
   TCM_CLAMP_BORDER                     =      4,
   TCM_MIRROR_ONCE                      =      5,
};

#define GEN5_CC_VIEWPORT_length                2
struct GEN5_CC_VIEWPORT {
   float                                MinimumDepth;
   float                                MaximumDepth;
};

static inline void
GEN5_CC_VIEWPORT_pack(__attribute__((unused)) __gen_user_data *data,
                      __attribute__((unused)) void * restrict dst,
                      __attribute__((unused)) const struct GEN5_CC_VIEWPORT * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_float(values->MinimumDepth);

   dw[1] =
      __gen_float(values->MaximumDepth);
}

#define GEN5_CLIP_STATE_length                11
struct GEN5_CLIP_STATE {
   uint64_t                             KernelStartPointer;
   uint32_t                             GRFRegisterCount;
   bool                                 SingleProgramFlow;
   uint32_t                             BindingTableEntryCount;
   uint32_t                             ThreadPriority;
#define NormalPriority                           0
#define HighPriority                             1
   uint32_t                             FloatingPointMode;
#define FLOATING_POINT_MODE_IEEE754              0
#define FLOATING_POINT_MODE_Alternate            1
   bool                                 IllegalOpcodeExceptionEnable;
   bool                                 MaskStackExceptionEnable;
   bool                                 SoftwareExceptionEnable;
   __gen_address_type                   ScratchSpaceBasePointer;
   uint32_t                             PerThreadScratchSpace;
   uint32_t                             ConstantURBEntryReadLength;
   uint32_t                             ConstantURBEntryReadOffset;
   uint32_t                             VertexURBEntryReadLength;
   uint32_t                             VertexURBEntryReadOffset;
   uint32_t                             DispatchGRFStartRegisterForURBData;
   uint32_t                             MaximumNumberofThreads;
   uint32_t                             URBEntryAllocationSize;
   uint32_t                             NumberofURBEntries;
   uint32_t                             APIMode;
#define APIMODE_OGL                              0
#define APIMODE_D3D                              1
   uint32_t                             VertexPositionSpace;
#define VPOS_NDCSPACE                            0
#define VPOS_SCREENSPACE                         1
   bool                                 ViewportXYClipTestEnable;
   bool                                 ViewportZClipTestEnable;
   bool                                 GuardbandClipTestEnable;
   bool                                 NegativeWClipTestEnable;
   bool                                 UserClipFlagsMustClipEnable;
   uint32_t                             UserClipDistanceClipTestEnableBitmask;
   uint32_t                             ClipMode;
#define CLIPMODE_NORMAL                          0
#define CLIPMODE_ALL                             1
#define CLIPMODE_CLIP_NON_REJECTED               2
#define CLIPMODE_REJECT_ALL                      3
#define CLIPMODE_ACCEPT_ALL                      4
   __gen_address_type                   ClipperViewportStatePointer;
   float                                ScreenSpaceViewportXMin;
   float                                ScreenSpaceViewportXMax;
   float                                ScreenSpaceViewportYMin;
   float                                ScreenSpaceViewportYMax;
};

static inline void
GEN5_CLIP_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                     __attribute__((unused)) void * restrict dst,
                     __attribute__((unused)) const struct GEN5_CLIP_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_offset(values->KernelStartPointer, 6, 31) |
      __gen_uint(values->GRFRegisterCount, 1, 3);

   dw[1] =
      __gen_uint(values->SingleProgramFlow, 31, 31) |
      __gen_uint(values->BindingTableEntryCount, 18, 25) |
      __gen_uint(values->ThreadPriority, 17, 17) |
      __gen_uint(values->FloatingPointMode, 16, 16) |
      __gen_uint(values->IllegalOpcodeExceptionEnable, 13, 13) |
      __gen_uint(values->MaskStackExceptionEnable, 11, 11) |
      __gen_uint(values->SoftwareExceptionEnable, 7, 7);

   const uint32_t v2 =
      __gen_uint(values->PerThreadScratchSpace, 0, 3);
   dw[2] = __gen_combine_address(data, &dw[2], values->ScratchSpaceBasePointer, v2);

   dw[3] =
      __gen_uint(values->ConstantURBEntryReadLength, 25, 30) |
      __gen_uint(values->ConstantURBEntryReadOffset, 18, 23) |
      __gen_uint(values->VertexURBEntryReadLength, 11, 16) |
      __gen_uint(values->VertexURBEntryReadOffset, 4, 9) |
      __gen_uint(values->DispatchGRFStartRegisterForURBData, 0, 3);

   dw[4] =
      __gen_uint(values->MaximumNumberofThreads, 25, 30) |
      __gen_uint(values->URBEntryAllocationSize, 19, 23) |
      __gen_uint(values->NumberofURBEntries, 11, 18);

   dw[5] =
      __gen_uint(values->APIMode, 30, 30) |
      __gen_uint(values->VertexPositionSpace, 29, 29) |
      __gen_uint(values->ViewportXYClipTestEnable, 28, 28) |
      __gen_uint(values->ViewportZClipTestEnable, 27, 27) |
      __gen_uint(values->GuardbandClipTestEnable, 26, 26) |
      __gen_uint(values->NegativeWClipTestEnable, 25, 25) |
      __gen_uint(values->UserClipFlagsMustClipEnable, 24, 24) |
      __gen_uint(values->UserClipDistanceClipTestEnableBitmask, 16, 23) |
      __gen_uint(values->ClipMode, 13, 15);

   dw[6] = __gen_combine_address(data, &dw[6], values->ClipperViewportStatePointer, 0);

   dw[7] =
      __gen_float(values->ScreenSpaceViewportXMin);

   dw[8] =
      __gen_float(values->ScreenSpaceViewportXMax);

   dw[9] =
      __gen_float(values->ScreenSpaceViewportYMin);

   dw[10] =
      __gen_float(values->ScreenSpaceViewportYMax);
}

#define GEN5_CLIP_VIEWPORT_length              4
struct GEN5_CLIP_VIEWPORT {
   float                                XMinClipGuardband;
   float                                XMaxClipGuardband;
   float                                YMinClipGuardband;
   float                                YMaxClipGuardband;
};

static inline void
GEN5_CLIP_VIEWPORT_pack(__attribute__((unused)) __gen_user_data *data,
                        __attribute__((unused)) void * restrict dst,
                        __attribute__((unused)) const struct GEN5_CLIP_VIEWPORT * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_float(values->XMinClipGuardband);

   dw[1] =
      __gen_float(values->XMaxClipGuardband);

   dw[2] =
      __gen_float(values->YMinClipGuardband);

   dw[3] =
      __gen_float(values->YMaxClipGuardband);
}

#define GEN5_SCISSOR_RECT_length               2
struct GEN5_SCISSOR_RECT {
   uint32_t                             ScissorRectangleYMin;
   uint32_t                             ScissorRectangleXMin;
   uint32_t                             ScissorRectangleYMax;
   uint32_t                             ScissorRectangleXMax;
};

static inline void
GEN5_SCISSOR_RECT_pack(__attribute__((unused)) __gen_user_data *data,
                       __attribute__((unused)) void * restrict dst,
                       __attribute__((unused)) const struct GEN5_SCISSOR_RECT * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->ScissorRectangleYMin, 16, 31) |
      __gen_uint(values->ScissorRectangleXMin, 0, 15);

   dw[1] =
      __gen_uint(values->ScissorRectangleYMax, 16, 31) |
      __gen_uint(values->ScissorRectangleXMax, 0, 15);
}

#define GEN5_COLOR_CALC_STATE_length           8
struct GEN5_COLOR_CALC_STATE {
   bool                                 StencilTestEnable;
   enum GEN5_3D_Compare_Function        StencilTestFunction;
   enum GEN5_3D_Stencil_Operation       StencilFailOp;
   enum GEN5_3D_Stencil_Operation       StencilPassDepthFailOp;
   enum GEN5_3D_Stencil_Operation       StencilPassDepthPassOp;
   bool                                 StencilBufferWriteEnable;
   bool                                 DoubleSidedStencilEnable;
   enum GEN5_3D_Compare_Function        BackfaceStencilTestFunction;
   enum GEN5_3D_Stencil_Operation       BackfaceStencilFailOp;
   enum GEN5_3D_Stencil_Operation       BackfaceStencilPassDepthFailOp;
   enum GEN5_3D_Stencil_Operation       BackfaceStencilPassDepthPassOp;
   uint32_t                             StencilReferenceValue;
   uint32_t                             StencilTestMask;
   uint32_t                             StencilWriteMask;
   uint32_t                             BackfaceStencilReferenceValue;
   uint32_t                             BackfaceStencilTestMask;
   uint32_t                             BackfaceStencilWriteMask;
   bool                                 DepthTestEnable;
   enum GEN5_3D_Compare_Function        DepthTestFunction;
   bool                                 DepthBufferWriteEnable;
   bool                                 LogicOpEnable;
   uint32_t                             AlphaTestFormat;
#define ALPHATEST_UNORM8                         0
#define ALPHATEST_FLOAT32                        1
   bool                                 IndependentAlphaBlendEnable;
   bool                                 ColorBufferBlendEnable;
   bool                                 AlphaTestEnable;
   enum GEN5_3D_Compare_Function        AlphaTestFunction;
   __gen_address_type                   CCViewportStatePointer;
   bool                                 ColorDitherEnable;
   bool                                 RoundDisableFunctionDisable;
   enum GEN5_3D_Logic_Op_Function       LogicOpFunction;
   bool                                 StatisticsEnable;
   enum GEN5_3D_Color_Buffer_Blend_Function AlphaBlendFunction;
   enum GEN5_3D_Color_Buffer_Blend_Factor SourceAlphaBlendFactor;
   enum GEN5_3D_Color_Buffer_Blend_Factor DestinationAlphaBlendFactor;
   enum GEN5_3D_Color_Buffer_Blend_Function ColorBlendFunction;
   enum GEN5_3D_Color_Buffer_Blend_Factor SourceBlendFactor;
   enum GEN5_3D_Color_Buffer_Blend_Factor DestinationBlendFactor;
   uint32_t                             XDitherOffset;
   uint32_t                             YDitherOffset;
   uint32_t                             ColorClampRange;
#define COLORCLAMP_UNORM                         0
#define COLORCLAMP_SNORM                         1
#define COLORCLAMP_RTFORMAT                      2
   bool                                 PreBlendColorClampEnable;
   bool                                 PostBlendColorClampEnable;
   uint32_t                             AlphaReferenceValueAsUNORM8;
   float                                AlphaReferenceValueAsFLOAT32;
};

static inline void
GEN5_COLOR_CALC_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                           __attribute__((unused)) void * restrict dst,
                           __attribute__((unused)) const struct GEN5_COLOR_CALC_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->StencilTestEnable, 31, 31) |
      __gen_uint(values->StencilTestFunction, 28, 30) |
      __gen_uint(values->StencilFailOp, 25, 27) |
      __gen_uint(values->StencilPassDepthFailOp, 22, 24) |
      __gen_uint(values->StencilPassDepthPassOp, 19, 21) |
      __gen_uint(values->StencilBufferWriteEnable, 18, 18) |
      __gen_uint(values->DoubleSidedStencilEnable, 15, 15) |
      __gen_uint(values->BackfaceStencilTestFunction, 12, 14) |
      __gen_uint(values->BackfaceStencilFailOp, 9, 11) |
      __gen_uint(values->BackfaceStencilPassDepthFailOp, 6, 8) |
      __gen_uint(values->BackfaceStencilPassDepthPassOp, 3, 5);

   dw[1] =
      __gen_uint(values->StencilReferenceValue, 24, 31) |
      __gen_uint(values->StencilTestMask, 16, 23) |
      __gen_uint(values->StencilWriteMask, 8, 15) |
      __gen_uint(values->BackfaceStencilReferenceValue, 0, 7);

   dw[2] =
      __gen_uint(values->BackfaceStencilTestMask, 24, 31) |
      __gen_uint(values->BackfaceStencilWriteMask, 16, 23) |
      __gen_uint(values->DepthTestEnable, 15, 15) |
      __gen_uint(values->DepthTestFunction, 12, 14) |
      __gen_uint(values->DepthBufferWriteEnable, 11, 11) |
      __gen_uint(values->LogicOpEnable, 0, 0);

   dw[3] =
      __gen_uint(values->AlphaTestFormat, 15, 15) |
      __gen_uint(values->IndependentAlphaBlendEnable, 13, 13) |
      __gen_uint(values->ColorBufferBlendEnable, 12, 12) |
      __gen_uint(values->AlphaTestEnable, 11, 11) |
      __gen_uint(values->AlphaTestFunction, 8, 10);

   dw[4] = __gen_combine_address(data, &dw[4], values->CCViewportStatePointer, 0);

   dw[5] =
      __gen_uint(values->ColorDitherEnable, 31, 31) |
      __gen_uint(values->RoundDisableFunctionDisable, 30, 30) |
      __gen_uint(values->LogicOpFunction, 16, 19) |
      __gen_uint(values->StatisticsEnable, 15, 15) |
      __gen_uint(values->AlphaBlendFunction, 12, 14) |
      __gen_uint(values->SourceAlphaBlendFactor, 7, 11) |
      __gen_uint(values->DestinationAlphaBlendFactor, 2, 6);

   dw[6] =
      __gen_uint(values->ColorBlendFunction, 29, 31) |
      __gen_uint(values->SourceBlendFactor, 24, 28) |
      __gen_uint(values->DestinationBlendFactor, 19, 23) |
      __gen_uint(values->XDitherOffset, 17, 18) |
      __gen_uint(values->YDitherOffset, 15, 16) |
      __gen_uint(values->ColorClampRange, 2, 3) |
      __gen_uint(values->PreBlendColorClampEnable, 1, 1) |
      __gen_uint(values->PostBlendColorClampEnable, 0, 0);

   dw[7] =
      __gen_uint(values->AlphaReferenceValueAsUNORM8, 0, 31) |
      __gen_float(values->AlphaReferenceValueAsFLOAT32);
}

#define GEN5_GS_STATE_length                   7
struct GEN5_GS_STATE {
   uint64_t                             KernelStartPointer;
   uint32_t                             GRFRegisterCount;
   bool                                 SingleProgramFlow;
   uint32_t                             BindingTableEntryCount;
   uint32_t                             FloatingPointMode;
#define FLOATING_POINT_MODE_IEEE754              0
#define FLOATING_POINT_MODE_Alternate            1
   bool                                 IllegalOpcodeExceptionEnable;
   bool                                 MaskStackExceptionEnable;
   bool                                 SoftwareExceptionEnable;
   __gen_address_type                   ScratchSpaceBasePointer;
   uint32_t                             PerThreadScratchSpace;
   uint32_t                             ConstantURBEntryReadLength;
   uint32_t                             ConstantURBEntryReadOffset;
   uint32_t                             VertexURBEntryReadLength;
   uint32_t                             VertexURBEntryReadOffset;
   uint32_t                             DispatchGRFStartRegisterForURBData;
   uint32_t                             MaximumNumberofThreads;
   uint32_t                             URBEntryAllocationSize;
   uint32_t                             NumberofURBEntries;
   bool                                 GSStatisticsEnable;
   bool                                 SOStatisticsEnable;
   bool                                 RenderingEnabled;
   __gen_address_type                   SamplerStatePointer;
   uint32_t                             SamplerCount;
   bool                                 ReorderEnable;
   uint32_t                             MaximumVPIndex;
};

static inline void
GEN5_GS_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                   __attribute__((unused)) void * restrict dst,
                   __attribute__((unused)) const struct GEN5_GS_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_offset(values->KernelStartPointer, 6, 31) |
      __gen_uint(values->GRFRegisterCount, 1, 3);

   dw[1] =
      __gen_uint(values->SingleProgramFlow, 31, 31) |
      __gen_uint(values->BindingTableEntryCount, 18, 25) |
      __gen_uint(values->FloatingPointMode, 16, 16) |
      __gen_uint(values->IllegalOpcodeExceptionEnable, 13, 13) |
      __gen_uint(values->MaskStackExceptionEnable, 11, 11) |
      __gen_uint(values->SoftwareExceptionEnable, 7, 7);

   const uint32_t v2 =
      __gen_uint(values->PerThreadScratchSpace, 0, 3);
   dw[2] = __gen_combine_address(data, &dw[2], values->ScratchSpaceBasePointer, v2);

   dw[3] =
      __gen_uint(values->ConstantURBEntryReadLength, 25, 30) |
      __gen_uint(values->ConstantURBEntryReadOffset, 18, 23) |
      __gen_uint(values->VertexURBEntryReadLength, 11, 16) |
      __gen_uint(values->VertexURBEntryReadOffset, 4, 9) |
      __gen_uint(values->DispatchGRFStartRegisterForURBData, 0, 3);

   dw[4] =
      __gen_uint(values->MaximumNumberofThreads, 25, 30) |
      __gen_uint(values->URBEntryAllocationSize, 19, 23) |
      __gen_uint(values->NumberofURBEntries, 11, 18) |
      __gen_uint(values->GSStatisticsEnable, 10, 10) |
      __gen_uint(values->SOStatisticsEnable, 9, 9) |
      __gen_uint(values->RenderingEnabled, 8, 8);

   const uint32_t v5 =
      __gen_uint(values->SamplerCount, 0, 2);
   dw[5] = __gen_combine_address(data, &dw[5], values->SamplerStatePointer, v5);

   dw[6] =
      __gen_uint(values->ReorderEnable, 30, 30) |
      __gen_uint(values->MaximumVPIndex, 0, 3);
}

#define GEN5_MEMORY_OBJECT_CONTROL_STATE_length      1
struct GEN5_MEMORY_OBJECT_CONTROL_STATE {
   bool                                 EncryptedData;
   uint32_t                             GraphicsDataTypeGFDT;
   uint32_t                             CacheabilityControl;
};

static inline void
GEN5_MEMORY_OBJECT_CONTROL_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                                      __attribute__((unused)) void * restrict dst,
                                      __attribute__((unused)) const struct GEN5_MEMORY_OBJECT_CONTROL_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->EncryptedData, 3, 3) |
      __gen_uint(values->GraphicsDataTypeGFDT, 2, 2) |
      __gen_uint(values->CacheabilityControl, 0, 1);
}

#define GEN5_SAMPLER_BORDER_COLOR_STATE_length     12
struct GEN5_SAMPLER_BORDER_COLOR_STATE {
   uint32_t                             BorderColorUnormRed;
   uint32_t                             BorderColorUnormGreen;
   uint32_t                             BorderColorUnormBlue;
   uint32_t                             BorderColorUnormAlpha;
   float                                BorderColorFloatRed;
   float                                BorderColorFloatGreen;
   float                                BorderColorFloatBlue;
   float                                BorderColorFloatAlpha;
   uint32_t                             BorderColorFloat16Red;
   uint32_t                             BorderColorFloat16Green;
   uint32_t                             BorderColorFloat16Blue;
   uint32_t                             BorderColorFloat16Alpha;
   uint32_t                             BorderColorUnorm16Red;
   uint32_t                             BorderColorUnorm16Green;
   uint32_t                             BorderColorUnorm16Blue;
   uint32_t                             BorderColorUnorm16Alpha;
   int32_t                              BorderColorSnorm16Red;
   int32_t                              BorderColorSnorm16Green;
   int32_t                              BorderColorSnorm16Blue;
   int32_t                              BorderColorSnorm16Alpha;
   int32_t                              BorderColorSnorm8Red;
   int32_t                              BorderColorSnorm8Green;
   int32_t                              BorderColorSnorm8Blue;
   int32_t                              BorderColorSnorm8Alpha;
};

static inline void
GEN5_SAMPLER_BORDER_COLOR_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                                     __attribute__((unused)) void * restrict dst,
                                     __attribute__((unused)) const struct GEN5_SAMPLER_BORDER_COLOR_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->BorderColorUnormRed, 0, 7) |
      __gen_uint(values->BorderColorUnormGreen, 8, 15) |
      __gen_uint(values->BorderColorUnormBlue, 16, 23) |
      __gen_uint(values->BorderColorUnormAlpha, 24, 31);

   dw[1] =
      __gen_float(values->BorderColorFloatRed);

   dw[2] =
      __gen_float(values->BorderColorFloatGreen);

   dw[3] =
      __gen_float(values->BorderColorFloatBlue);

   dw[4] =
      __gen_float(values->BorderColorFloatAlpha);

   dw[5] =
      __gen_uint(values->BorderColorFloat16Red, 0, 15) |
      __gen_uint(values->BorderColorFloat16Green, 16, 31);

   dw[6] =
      __gen_uint(values->BorderColorFloat16Blue, 0, 15) |
      __gen_uint(values->BorderColorFloat16Alpha, 16, 31);

   dw[7] =
      __gen_uint(values->BorderColorUnorm16Red, 0, 15) |
      __gen_uint(values->BorderColorUnorm16Green, 16, 31);

   dw[8] =
      __gen_uint(values->BorderColorUnorm16Blue, 0, 15) |
      __gen_uint(values->BorderColorUnorm16Alpha, 16, 31);

   dw[9] =
      __gen_sint(values->BorderColorSnorm16Red, 0, 15) |
      __gen_sint(values->BorderColorSnorm16Green, 16, 31);

   dw[10] =
      __gen_sint(values->BorderColorSnorm16Blue, 0, 15) |
      __gen_sint(values->BorderColorSnorm16Alpha, 16, 31);

   dw[11] =
      __gen_sint(values->BorderColorSnorm8Red, 0, 7) |
      __gen_sint(values->BorderColorSnorm8Green, 8, 15) |
      __gen_sint(values->BorderColorSnorm8Blue, 16, 23) |
      __gen_sint(values->BorderColorSnorm8Alpha, 24, 31);
}

#define GEN5_SAMPLER_STATE_length              4
struct GEN5_SAMPLER_STATE {
   bool                                 SamplerDisable;
   bool                                 LODPreClampEnable;
   float                                BaseMipLevel;
   uint32_t                             MipModeFilter;
#define MIPFILTER_NONE                           0
#define MIPFILTER_NEAREST                        1
#define MIPFILTER_LINEAR                         3
   uint32_t                             MagModeFilter;
#define MAPFILTER_NEAREST                        0
#define MAPFILTER_LINEAR                         1
#define MAPFILTER_ANISOTROPIC                    2
#define MAPFILTER_MONO                           6
   uint32_t                             MinModeFilter;
   float                                TextureLODBias;
   uint32_t                             ShadowFunction;
#define PREFILTEROP_ALWAYS                       0
#define PREFILTEROP_NEVER                        1
#define PREFILTEROP_LESS                         2
#define PREFILTEROP_EQUAL                        3
#define PREFILTEROP_LEQUAL                       4
#define PREFILTEROP_GREATER                      5
#define PREFILTEROP_NOTEQUAL                     6
#define PREFILTEROP_GEQUAL                       7
   float                                MinLOD;
   float                                MaxLOD;
   uint32_t                             CubeSurfaceControlMode;
#define CUBECTRLMODE_PROGRAMMED                  0
#define CUBECTRLMODE_OVERRIDE                    1
   enum GEN5_Texture_Coordinate_Mode    TCXAddressControlMode;
   enum GEN5_Texture_Coordinate_Mode    TCYAddressControlMode;
   enum GEN5_Texture_Coordinate_Mode    TCZAddressControlMode;
   __gen_address_type                   BorderColorPointer;
   uint32_t                             MonochromeFilterHeight;
   uint32_t                             MonochromeFilterWidth;
   bool                                 ChromaKeyEnable;
   uint32_t                             ChromaKeyIndex;
   uint32_t                             ChromaKeyMode;
#define KEYFILTER_KILL_ON_ANY_MATCH              0
#define KEYFILTER_REPLACE_BLACK                  1
   uint32_t                             MaximumAnisotropy;
#define RATIO21                                  0
#define RATIO41                                  1
#define RATIO61                                  2
#define RATIO81                                  3
#define RATIO101                                 4
#define RATIO121                                 5
#define RATIO141                                 6
#define RATIO161                                 7
   bool                                 RAddressMinFilterRoundingEnable;
   bool                                 RAddressMagFilterRoundingEnable;
   bool                                 VAddressMinFilterRoundingEnable;
   bool                                 VAddressMagFilterRoundingEnable;
   bool                                 UAddressMinFilterRoundingEnable;
   bool                                 UAddressMagFilterRoundingEnable;
};

static inline void
GEN5_SAMPLER_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                        __attribute__((unused)) void * restrict dst,
                        __attribute__((unused)) const struct GEN5_SAMPLER_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->SamplerDisable, 31, 31) |
      __gen_uint(values->LODPreClampEnable, 28, 28) |
      __gen_ufixed(values->BaseMipLevel, 22, 26, 1) |
      __gen_uint(values->MipModeFilter, 20, 21) |
      __gen_uint(values->MagModeFilter, 17, 19) |
      __gen_uint(values->MinModeFilter, 14, 16) |
      __gen_sfixed(values->TextureLODBias, 3, 13, 6) |
      __gen_uint(values->ShadowFunction, 0, 2);

   dw[1] =
      __gen_ufixed(values->MinLOD, 22, 31, 6) |
      __gen_ufixed(values->MaxLOD, 12, 21, 6) |
      __gen_uint(values->CubeSurfaceControlMode, 9, 9) |
      __gen_uint(values->TCXAddressControlMode, 6, 8) |
      __gen_uint(values->TCYAddressControlMode, 3, 5) |
      __gen_uint(values->TCZAddressControlMode, 0, 2);

   dw[2] = __gen_combine_address(data, &dw[2], values->BorderColorPointer, 0);

   dw[3] =
      __gen_uint(values->MonochromeFilterHeight, 29, 31) |
      __gen_uint(values->MonochromeFilterWidth, 26, 28) |
      __gen_uint(values->ChromaKeyEnable, 25, 25) |
      __gen_uint(values->ChromaKeyIndex, 23, 24) |
      __gen_uint(values->ChromaKeyMode, 22, 22) |
      __gen_uint(values->MaximumAnisotropy, 19, 21) |
      __gen_uint(values->RAddressMinFilterRoundingEnable, 13, 13) |
      __gen_uint(values->RAddressMagFilterRoundingEnable, 14, 14) |
      __gen_uint(values->VAddressMinFilterRoundingEnable, 15, 15) |
      __gen_uint(values->VAddressMagFilterRoundingEnable, 16, 16) |
      __gen_uint(values->UAddressMinFilterRoundingEnable, 17, 17) |
      __gen_uint(values->UAddressMagFilterRoundingEnable, 18, 18);
}

#define GEN5_RENDER_SURFACE_STATE_length       6
struct GEN5_RENDER_SURFACE_STATE {
   uint32_t                             SurfaceType;
#define SURFTYPE_1D                              0
#define SURFTYPE_2D                              1
#define SURFTYPE_3D                              2
#define SURFTYPE_CUBE                            3
#define SURFTYPE_BUFFER                          4
#define SURFTYPE_NULL                            7
   uint32_t                             DataReturnFormat;
#define DATA_RETURN_FLOAT32                      0
#define DATA_RETURN_S114                         1
   uint32_t                             SurfaceFormat;
   uint32_t                             ColorBufferComponentWriteDisables;
#define WRITEDISABLE_ALPHA                       8
#define WRITEDISABLE_RED                         4
#define WRITEDISABLE_GREEN                       2
#define WRITEDISABLE_BLUE                        1
   bool                                 ColorBlendEnable;
   uint32_t                             VerticalLineStride;
   uint32_t                             VerticalLineStrideOffset;
   uint32_t                             MIPMapLayoutMode;
#define MIPLAYOUT_BELOW                          0
#define MIPLAYOUT_RIGHT                          1
   uint32_t                             CubeMapCornerMode;
#define CUBE_REPLICATE                           0
#define CUBE_AVERAGE                             1
   uint32_t                             RenderCacheReadWriteMode;
#define WRITE_ONLY                               0
#define READ_WRITE                               1
   uint32_t                             MediaBoundaryPixelMode;
#define NORMAL_MODE                              0
   bool                                 CubeFaceEnablePositiveZ;
   bool                                 CubeFaceEnableNegativeZ;
   bool                                 CubeFaceEnablePositiveY;
   bool                                 CubeFaceEnableNegativeY;
   bool                                 CubeFaceEnablePositiveX;
   bool                                 CubeFaceEnableNegativeX;
   __gen_address_type                   SurfaceBaseAddress;
   uint32_t                             Height;
   uint32_t                             Width;
   uint32_t                             MIPCountLOD;
   uint32_t                             RenderTargetRotation;
#define RTROTATE_0DEG                            0
#define RTROTATE_90DEG                           1
#define RTROTATE_270DEG                          3
   uint32_t                             Depth;
   uint32_t                             SurfacePitch;
   uint32_t                             TiledSurface;
   uint32_t                             TileWalk;
#define TILEWALK_XMAJOR                          0
#define TILEWALK_YMAJOR                          1
   uint32_t                             SurfaceMinLOD;
   uint32_t                             MinimumArrayElement;
   uint32_t                             RenderTargetViewExtent;
   uint32_t                             XOffset;
   uint32_t                             YOffset;
};

static inline void
GEN5_RENDER_SURFACE_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                               __attribute__((unused)) void * restrict dst,
                               __attribute__((unused)) const struct GEN5_RENDER_SURFACE_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->SurfaceType, 29, 31) |
      __gen_uint(values->DataReturnFormat, 27, 27) |
      __gen_uint(values->SurfaceFormat, 18, 26) |
      __gen_uint(values->ColorBufferComponentWriteDisables, 14, 17) |
      __gen_uint(values->ColorBlendEnable, 13, 13) |
      __gen_uint(values->VerticalLineStride, 12, 12) |
      __gen_uint(values->VerticalLineStrideOffset, 11, 11) |
      __gen_uint(values->MIPMapLayoutMode, 10, 10) |
      __gen_uint(values->CubeMapCornerMode, 9, 9) |
      __gen_uint(values->RenderCacheReadWriteMode, 8, 8) |
      __gen_uint(values->MediaBoundaryPixelMode, 6, 7) |
      __gen_uint(values->CubeFaceEnablePositiveZ, 0, 0) |
      __gen_uint(values->CubeFaceEnableNegativeZ, 1, 1) |
      __gen_uint(values->CubeFaceEnablePositiveY, 2, 2) |
      __gen_uint(values->CubeFaceEnableNegativeY, 3, 3) |
      __gen_uint(values->CubeFaceEnablePositiveX, 4, 4) |
      __gen_uint(values->CubeFaceEnableNegativeX, 5, 5);

   dw[1] = __gen_combine_address(data, &dw[1], values->SurfaceBaseAddress, 0);

   dw[2] =
      __gen_uint(values->Height, 19, 31) |
      __gen_uint(values->Width, 6, 18) |
      __gen_uint(values->MIPCountLOD, 2, 5) |
      __gen_uint(values->RenderTargetRotation, 0, 1);

   dw[3] =
      __gen_uint(values->Depth, 21, 31) |
      __gen_uint(values->SurfacePitch, 3, 19) |
      __gen_uint(values->TiledSurface, 1, 1) |
      __gen_uint(values->TileWalk, 0, 0);

   dw[4] =
      __gen_uint(values->SurfaceMinLOD, 28, 31) |
      __gen_uint(values->MinimumArrayElement, 17, 27) |
      __gen_uint(values->RenderTargetViewExtent, 8, 16);

   dw[5] =
      __gen_uint(values->XOffset, 25, 31) |
      __gen_uint(values->YOffset, 20, 23);
}

#define GEN5_3DSTATE_DEPTH_BUFFER_length       6
#define GEN5_3DSTATE_DEPTH_BUFFER_length_bias      2
#define GEN5_3DSTATE_DEPTH_BUFFER_header        \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =      5,  \
   .DWordLength                         =      4

struct GEN5_3DSTATE_DEPTH_BUFFER {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   uint32_t                             SurfaceType;
#define SURFTYPE_1D                              0
#define SURFTYPE_2D                              1
#define SURFTYPE_3D                              2
#define SURFTYPE_CUBE                            3
#define SURFTYPE_NULL                            7
   bool                                 TiledSurface;
   uint32_t                             TileWalk;
#define TILEWALK_YMAJOR                          1
   uint32_t                             SoftwareTiledRenderingMode;
#define NORMAL                                   0
#define STR1                                     1
#define STR2                                     3
   bool                                 HierarchicalDepthBufferEnable;
   bool                                 SeparateStencilBufferEnable;
   uint32_t                             SurfaceFormat;
#define D32_FLOAT_S8X24_UINT                     0
#define D32_FLOAT                                1
#define D24_UNORM_S8_UINT                        2
#define D24_UNORM_X8_UINT                        3
#define D16_UNORM                                5
   uint32_t                             SurfacePitch;
   __gen_address_type                   SurfaceBaseAddress;
   uint32_t                             Height;
   uint32_t                             Width;
   uint32_t                             LOD;
   uint32_t                             MIPMapLayoutMode;
#define MIPLAYOUT_BELOW                          0
#define MIPLAYOUT_RIGHT                          1
   uint32_t                             Depth;
   uint32_t                             MinimumArrayElement;
   uint32_t                             RenderTargetViewExtent;
   int32_t                              DepthCoordinateOffsetY;
   int32_t                              DepthCoordinateOffsetX;
};

static inline void
GEN5_3DSTATE_DEPTH_BUFFER_pack(__attribute__((unused)) __gen_user_data *data,
                               __attribute__((unused)) void * restrict dst,
                               __attribute__((unused)) const struct GEN5_3DSTATE_DEPTH_BUFFER * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_uint(values->SurfaceType, 29, 31) |
      __gen_uint(values->TiledSurface, 27, 27) |
      __gen_uint(values->TileWalk, 26, 26) |
      __gen_uint(values->SoftwareTiledRenderingMode, 23, 24) |
      __gen_uint(values->HierarchicalDepthBufferEnable, 22, 22) |
      __gen_uint(values->SeparateStencilBufferEnable, 21, 21) |
      __gen_uint(values->SurfaceFormat, 18, 20) |
      __gen_uint(values->SurfacePitch, 0, 16);

   dw[2] = __gen_combine_address(data, &dw[2], values->SurfaceBaseAddress, 0);

   dw[3] =
      __gen_uint(values->Height, 19, 31) |
      __gen_uint(values->Width, 6, 18) |
      __gen_uint(values->LOD, 2, 5) |
      __gen_uint(values->MIPMapLayoutMode, 1, 1);

   dw[4] =
      __gen_uint(values->Depth, 21, 31) |
      __gen_uint(values->MinimumArrayElement, 10, 20) |
      __gen_uint(values->RenderTargetViewExtent, 1, 9);

   dw[5] =
      __gen_sint(values->DepthCoordinateOffsetY, 16, 31) |
      __gen_sint(values->DepthCoordinateOffsetX, 0, 15);
}

#define GEN5_SF_STATE_length                   8
struct GEN5_SF_STATE {
   uint64_t                             KernelStartPointer;
   uint32_t                             GRFRegisterCount;
   bool                                 SingleProgramFlow;
   uint32_t                             BindingTableEntryCount;
   uint32_t                             ThreadPriority;
#define NormalPriority                           0
#define HighPriority                             1
   uint32_t                             FloatingPointMode;
#define FLOATING_POINT_MODE_IEEE754              0
#define FLOATING_POINT_MODE_Alternate            1
   bool                                 IllegalOpcodeExceptionEnable;
   bool                                 MaskStackExceptionEnable;
   bool                                 SoftwareExceptionEnable;
   __gen_address_type                   ScratchSpaceBasePointer;
   uint32_t                             PerThreadScratchSpace;
   uint32_t                             ConstantURBEntryReadLength;
   uint32_t                             ConstantURBEntryReadOffset;
   uint32_t                             VertexURBEntryReadLength;
   uint32_t                             VertexURBEntryReadOffset;
   uint32_t                             DispatchGRFStartRegisterForURBData;
   uint32_t                             MaximumNumberofThreads;
   uint32_t                             URBEntryAllocationSize;
   uint32_t                             NumberofURBEntries;
   __gen_address_type                   SetupViewportStateOffset;
   bool                                 ViewportTransformEnable;
   uint32_t                             FrontWinding;
#define FRONTWINDING_CW                          0
#define FRONTWINDING_CCW                         1
   bool                                 AntiAliasingEnable;
   uint32_t                             CullMode;
#define CULLMODE_BOTH                            0
#define CULLMODE_NONE                            1
#define CULLMODE_FRONT                           2
#define CULLMODE_BACK                            3
   bool                                 FastScissorClipDisable;
   float                                LineWidth;
   uint32_t                             LineEndCapAntialiasingRegionWidth;
#define _05pixels                                0
#define _10pixels                                1
#define _20pixels                                2
#define _40pixels                                3
   uint32_t                             PointRasterizationRule;
#define RASTRULE_UPPER_LEFT                      0
#define RASTRULE_UPPER_RIGHT                     1
   bool                                 ZeroPixelTriangleFilterDisable;
   bool                                 _2x2PixelTriangleFilterDisable;
   bool                                 ScissorRectangleEnable;
   float                                DestinationOriginHorizontalBias;
   float                                DestinationOriginVerticalBias;
   bool                                 LastPixelEnable;
   uint32_t                             TriangleStripListProvokingVertexSelect;
#define Vertex0                                  0
#define Vertex1                                  1
#define Vertex2                                  2
   uint32_t                             LineStripListProvokingVertexSelect;
#define Vertex0                                  0
#define Vertex1                                  1
   uint32_t                             TriangleFanProvokingVertexSelect;
#define Vertex0                                  0
#define Vertex1                                  1
#define Vertex2                                  2
   uint32_t                             AALineDistanceMode;
#define AALINEDISTANCE_MANHATTAN                 0
#define AALINEDISTANCE_TRUE                      1
   bool                                 SpritePointEnable;
   uint32_t                             VertexSubPixelPrecisionSelect;
#define _8SubPixelPrecisionBits                  0
#define _4SubPixelPrecisionBits                  1
   uint32_t                             PointWidthSource;
#define Vertex                                   0
#define State                                    1
   float                                PointWidth;
};

static inline void
GEN5_SF_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                   __attribute__((unused)) void * restrict dst,
                   __attribute__((unused)) const struct GEN5_SF_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_offset(values->KernelStartPointer, 6, 31) |
      __gen_uint(values->GRFRegisterCount, 1, 3);

   dw[1] =
      __gen_uint(values->SingleProgramFlow, 31, 31) |
      __gen_uint(values->BindingTableEntryCount, 18, 25) |
      __gen_uint(values->ThreadPriority, 17, 17) |
      __gen_uint(values->FloatingPointMode, 16, 16) |
      __gen_uint(values->IllegalOpcodeExceptionEnable, 13, 13) |
      __gen_uint(values->MaskStackExceptionEnable, 11, 11) |
      __gen_uint(values->SoftwareExceptionEnable, 7, 7);

   const uint32_t v2 =
      __gen_uint(values->PerThreadScratchSpace, 0, 3);
   dw[2] = __gen_combine_address(data, &dw[2], values->ScratchSpaceBasePointer, v2);

   dw[3] =
      __gen_uint(values->ConstantURBEntryReadLength, 25, 30) |
      __gen_uint(values->ConstantURBEntryReadOffset, 18, 23) |
      __gen_uint(values->VertexURBEntryReadLength, 11, 16) |
      __gen_uint(values->VertexURBEntryReadOffset, 4, 9) |
      __gen_uint(values->DispatchGRFStartRegisterForURBData, 0, 3);

   dw[4] =
      __gen_uint(values->MaximumNumberofThreads, 25, 30) |
      __gen_uint(values->URBEntryAllocationSize, 19, 23) |
      __gen_uint(values->NumberofURBEntries, 11, 18);

   const uint32_t v5 =
      __gen_uint(values->ViewportTransformEnable, 1, 1) |
      __gen_uint(values->FrontWinding, 0, 0);
   dw[5] = __gen_combine_address(data, &dw[5], values->SetupViewportStateOffset, v5);

   dw[6] =
      __gen_uint(values->AntiAliasingEnable, 31, 31) |
      __gen_uint(values->CullMode, 29, 30) |
      __gen_uint(values->FastScissorClipDisable, 28, 28) |
      __gen_ufixed(values->LineWidth, 24, 27, 1) |
      __gen_uint(values->LineEndCapAntialiasingRegionWidth, 22, 23) |
      __gen_uint(values->PointRasterizationRule, 20, 21) |
      __gen_uint(values->ZeroPixelTriangleFilterDisable, 19, 19) |
      __gen_uint(values->_2x2PixelTriangleFilterDisable, 18, 18) |
      __gen_uint(values->ScissorRectangleEnable, 17, 17) |
      __gen_ufixed(values->DestinationOriginHorizontalBias, 13, 16, 4) |
      __gen_ufixed(values->DestinationOriginVerticalBias, 9, 12, 4);

   dw[7] =
      __gen_uint(values->LastPixelEnable, 31, 31) |
      __gen_uint(values->TriangleStripListProvokingVertexSelect, 29, 30) |
      __gen_uint(values->LineStripListProvokingVertexSelect, 27, 28) |
      __gen_uint(values->TriangleFanProvokingVertexSelect, 25, 26) |
      __gen_uint(values->AALineDistanceMode, 14, 14) |
      __gen_uint(values->SpritePointEnable, 13, 13) |
      __gen_uint(values->VertexSubPixelPrecisionSelect, 12, 12) |
      __gen_uint(values->PointWidthSource, 11, 11) |
      __gen_ufixed(values->PointWidth, 0, 10, 3);
}

#define GEN5_SF_VIEWPORT_length                8
struct GEN5_SF_VIEWPORT {
   float                                ViewportMatrixElementm00;
   float                                ViewportMatrixElementm11;
   float                                ViewportMatrixElementm22;
   float                                ViewportMatrixElementm30;
   float                                ViewportMatrixElementm31;
   float                                ViewportMatrixElementm32;
   struct GEN5_SCISSOR_RECT             ScissorRectangle;
};

static inline void
GEN5_SF_VIEWPORT_pack(__attribute__((unused)) __gen_user_data *data,
                      __attribute__((unused)) void * restrict dst,
                      __attribute__((unused)) const struct GEN5_SF_VIEWPORT * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_float(values->ViewportMatrixElementm00);

   dw[1] =
      __gen_float(values->ViewportMatrixElementm11);

   dw[2] =
      __gen_float(values->ViewportMatrixElementm22);

   dw[3] =
      __gen_float(values->ViewportMatrixElementm30);

   dw[4] =
      __gen_float(values->ViewportMatrixElementm31);

   dw[5] =
      __gen_float(values->ViewportMatrixElementm32);

   GEN5_SCISSOR_RECT_pack(data, &dw[6], &values->ScissorRectangle);
}

#define GEN5_VERTEX_BUFFER_STATE_length        4
struct GEN5_VERTEX_BUFFER_STATE {
   uint32_t                             VertexBufferIndex;
   uint32_t                             BufferAccessType;
#define VERTEXDATA                               0
#define INSTANCEDATA                             1
   bool                                 NullVertexBuffer;
   uint32_t                             BufferPitch;
   __gen_address_type                   BufferStartingAddress;
   __gen_address_type                   EndAddress;
   uint32_t                             InstanceDataStepRate;
};

static inline void
GEN5_VERTEX_BUFFER_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                              __attribute__((unused)) void * restrict dst,
                              __attribute__((unused)) const struct GEN5_VERTEX_BUFFER_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->VertexBufferIndex, 27, 31) |
      __gen_uint(values->BufferAccessType, 26, 26) |
      __gen_uint(values->NullVertexBuffer, 13, 13) |
      __gen_uint(values->BufferPitch, 0, 11);

   dw[1] = __gen_combine_address(data, &dw[1], values->BufferStartingAddress, 0);

   dw[2] = __gen_combine_address(data, &dw[2], values->EndAddress, 0);

   dw[3] =
      __gen_uint(values->InstanceDataStepRate, 0, 31);
}

#define GEN5_VERTEX_ELEMENT_STATE_length       2
struct GEN5_VERTEX_ELEMENT_STATE {
   uint32_t                             VertexBufferIndex;
   bool                                 Valid;
   uint32_t                             SourceElementFormat;
   uint32_t                             SourceElementOffset;
   uint32_t                             DestinationElementOffset;
   enum GEN5_3D_Vertex_Component_Control Component0Control;
   enum GEN5_3D_Vertex_Component_Control Component1Control;
   enum GEN5_3D_Vertex_Component_Control Component2Control;
   enum GEN5_3D_Vertex_Component_Control Component3Control;
};

static inline void
GEN5_VERTEX_ELEMENT_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                               __attribute__((unused)) void * restrict dst,
                               __attribute__((unused)) const struct GEN5_VERTEX_ELEMENT_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->VertexBufferIndex, 27, 31) |
      __gen_uint(values->Valid, 26, 26) |
      __gen_uint(values->SourceElementFormat, 16, 24) |
      __gen_uint(values->SourceElementOffset, 0, 10);

   dw[1] =
      __gen_uint(values->DestinationElementOffset, 0, 7) |
      __gen_uint(values->Component0Control, 28, 30) |
      __gen_uint(values->Component1Control, 24, 26) |
      __gen_uint(values->Component2Control, 20, 22) |
      __gen_uint(values->Component3Control, 16, 18);
}

#define GEN5_VS_STATE_length                   7
struct GEN5_VS_STATE {
   uint64_t                             KernelStartPointer;
   uint32_t                             GRFRegisterCount;
   bool                                 SingleProgramFlow;
   uint32_t                             BindingTableEntryCount;
   uint32_t                             ThreadPriority;
#define NormalPriority                           0
#define HighPriority                             1
   uint32_t                             FloatingPointMode;
#define FLOATING_POINT_MODE_IEEE754              0
#define FLOATING_POINT_MODE_Alternate            1
   bool                                 IllegalOpcodeExceptionEnable;
   bool                                 MaskStackExceptionEnable;
   bool                                 SoftwareExceptionEnable;
   __gen_address_type                   ScratchSpaceBasePointer;
   uint32_t                             PerThreadScratchSpace;
   uint32_t                             ConstantURBEntryReadLength;
   uint32_t                             ConstantURBEntryReadOffset;
   uint32_t                             VertexURBEntryReadLength;
   uint32_t                             VertexURBEntryReadOffset;
   uint32_t                             DispatchGRFStartRegisterForURBData;
   uint32_t                             MaximumNumberofThreads;
   uint32_t                             URBEntryAllocationSize;
   uint32_t                             NumberofURBEntries;
   bool                                 StatisticsEnable;
   __gen_address_type                   SamplerStatePointer;
   uint32_t                             SamplerCount;
   bool                                 VertexCacheDisable;
   bool                                 Enable;
};

static inline void
GEN5_VS_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                   __attribute__((unused)) void * restrict dst,
                   __attribute__((unused)) const struct GEN5_VS_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_offset(values->KernelStartPointer, 6, 31) |
      __gen_uint(values->GRFRegisterCount, 1, 3);

   dw[1] =
      __gen_uint(values->SingleProgramFlow, 31, 31) |
      __gen_uint(values->BindingTableEntryCount, 18, 25) |
      __gen_uint(values->ThreadPriority, 17, 17) |
      __gen_uint(values->FloatingPointMode, 16, 16) |
      __gen_uint(values->IllegalOpcodeExceptionEnable, 13, 13) |
      __gen_uint(values->MaskStackExceptionEnable, 11, 11) |
      __gen_uint(values->SoftwareExceptionEnable, 7, 7);

   const uint32_t v2 =
      __gen_uint(values->PerThreadScratchSpace, 0, 3);
   dw[2] = __gen_combine_address(data, &dw[2], values->ScratchSpaceBasePointer, v2);

   dw[3] =
      __gen_uint(values->ConstantURBEntryReadLength, 25, 30) |
      __gen_uint(values->ConstantURBEntryReadOffset, 18, 23) |
      __gen_uint(values->VertexURBEntryReadLength, 11, 16) |
      __gen_uint(values->VertexURBEntryReadOffset, 4, 9) |
      __gen_uint(values->DispatchGRFStartRegisterForURBData, 0, 3);

   dw[4] =
      __gen_uint(values->MaximumNumberofThreads, 25, 30) |
      __gen_uint(values->URBEntryAllocationSize, 19, 23) |
      __gen_uint(values->NumberofURBEntries, 11, 18) |
      __gen_uint(values->StatisticsEnable, 10, 10);

   const uint32_t v5 =
      __gen_uint(values->SamplerCount, 0, 2);
   dw[5] = __gen_combine_address(data, &dw[5], values->SamplerStatePointer, v5);

   dw[6] =
      __gen_uint(values->VertexCacheDisable, 1, 1) |
      __gen_uint(values->Enable, 0, 0);
}

#define GEN5_WM_STATE_length                  11
struct GEN5_WM_STATE {
   uint64_t                             KernelStartPointer0;
   uint32_t                             GRFRegisterCount0;
   bool                                 SingleProgramFlow;
   uint32_t                             BindingTableEntryCount;
   uint32_t                             ThreadPriority;
#define High                                     1
   uint32_t                             FloatingPointMode;
#define FLOATING_POINT_MODE_IEEE754              0
#define FLOATING_POINT_MODE_Alternate            1
   uint32_t                             DepthCoefficientURBReadOffset;
   bool                                 IllegalOpcodeExceptionEnable;
   bool                                 MaskStackExceptionEnable;
   bool                                 SoftwareExceptionEnable;
   __gen_address_type                   ScratchSpaceBasePointer;
   uint32_t                             PerThreadScratchSpace;
   uint32_t                             ConstantURBEntryReadLength;
   uint32_t                             ConstantURBEntryReadOffset;
   uint32_t                             SetupURBEntryReadLength;
   uint32_t                             SetupURBEntryReadOffset;
   uint32_t                             DispatchGRFStartRegisterForConstantSetupData0;
   __gen_address_type                   SamplerStatePointer;
   uint32_t                             SamplerCount;
   bool                                 StatisticsEnable;
   uint32_t                             MaximumNumberofThreads;
   bool                                 LegacyDiamondLineRasterization;
   bool                                 PixelShaderKillsPixel;
   bool                                 PixelShaderComputedDepth;
   bool                                 PixelShaderUsesSourceDepth;
   bool                                 ThreadDispatchEnable;
   bool                                 EarlyDepthTestEnable;
   uint32_t                             LineEndCapAntialiasingRegionWidth;
#define _05pixels                                0
#define _10pixels                                1
#define _20pixels                                2
#define _40pixels                                3
   uint32_t                             LineAntialiasingRegionWidth;
#define _05pixels                                0
#define _10pixels                                1
#define _20pixels                                2
#define _40pixels                                3
   bool                                 PolygonStippleEnable;
   bool                                 GlobalDepthOffsetEnable;
   bool                                 LineStippleEnable;
   bool                                 LegacyGlobalDepthBiasEnable;
   bool                                 HierarchicalDepthBufferResolveEnable;
   bool                                 DepthBufferResolveEnable;
   bool                                 DepthBufferClear;
   bool                                 FastSpanCoverageEnable;
   bool                                 Contiguous64PixelDispatchEnable;
   bool                                 Contiguous32PixelDispatchEnable;
   bool                                 _32PixelDispatchEnable;
   bool                                 _16PixelDispatchEnable;
   bool                                 _8PixelDispatchEnable;
   float                                GlobalDepthOffsetConstant;
   float                                GlobalDepthOffsetScale;
   uint64_t                             KernelStartPointer1;
   uint32_t                             GRFRegisterCount1;
   uint64_t                             KernelStartPointer2;
   uint32_t                             GRFRegisterCount2;
   uint64_t                             KernelStartPointer3;
   uint32_t                             GRFRegisterCount3;
};

static inline void
GEN5_WM_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                   __attribute__((unused)) void * restrict dst,
                   __attribute__((unused)) const struct GEN5_WM_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_offset(values->KernelStartPointer0, 6, 31) |
      __gen_uint(values->GRFRegisterCount0, 1, 3);

   dw[1] =
      __gen_uint(values->SingleProgramFlow, 31, 31) |
      __gen_uint(values->BindingTableEntryCount, 18, 25) |
      __gen_uint(values->ThreadPriority, 17, 17) |
      __gen_uint(values->FloatingPointMode, 16, 16) |
      __gen_uint(values->DepthCoefficientURBReadOffset, 8, 13) |
      __gen_uint(values->IllegalOpcodeExceptionEnable, 4, 4) |
      __gen_uint(values->MaskStackExceptionEnable, 2, 2) |
      __gen_uint(values->SoftwareExceptionEnable, 1, 1);

   const uint32_t v2 =
      __gen_uint(values->PerThreadScratchSpace, 0, 3);
   dw[2] = __gen_combine_address(data, &dw[2], values->ScratchSpaceBasePointer, v2);

   dw[3] =
      __gen_uint(values->ConstantURBEntryReadLength, 25, 30) |
      __gen_uint(values->ConstantURBEntryReadOffset, 18, 23) |
      __gen_uint(values->SetupURBEntryReadLength, 11, 16) |
      __gen_uint(values->SetupURBEntryReadOffset, 4, 9) |
      __gen_uint(values->DispatchGRFStartRegisterForConstantSetupData0, 0, 3);

   const uint32_t v4 =
      __gen_uint(values->SamplerCount, 2, 4) |
      __gen_uint(values->StatisticsEnable, 0, 0);
   dw[4] = __gen_combine_address(data, &dw[4], values->SamplerStatePointer, v4);

   dw[5] =
      __gen_uint(values->MaximumNumberofThreads, 25, 31) |
      __gen_uint(values->LegacyDiamondLineRasterization, 23, 23) |
      __gen_uint(values->PixelShaderKillsPixel, 22, 22) |
      __gen_uint(values->PixelShaderComputedDepth, 21, 21) |
      __gen_uint(values->PixelShaderUsesSourceDepth, 20, 20) |
      __gen_uint(values->ThreadDispatchEnable, 19, 19) |
      __gen_uint(values->EarlyDepthTestEnable, 18, 18) |
      __gen_uint(values->LineEndCapAntialiasingRegionWidth, 16, 17) |
      __gen_uint(values->LineAntialiasingRegionWidth, 14, 15) |
      __gen_uint(values->PolygonStippleEnable, 13, 13) |
      __gen_uint(values->GlobalDepthOffsetEnable, 12, 12) |
      __gen_uint(values->LineStippleEnable, 11, 11) |
      __gen_uint(values->LegacyGlobalDepthBiasEnable, 10, 10) |
      __gen_uint(values->HierarchicalDepthBufferResolveEnable, 9, 9) |
      __gen_uint(values->DepthBufferResolveEnable, 8, 8) |
      __gen_uint(values->DepthBufferClear, 7, 7) |
      __gen_uint(values->FastSpanCoverageEnable, 6, 6) |
      __gen_uint(values->Contiguous64PixelDispatchEnable, 4, 4) |
      __gen_uint(values->Contiguous32PixelDispatchEnable, 3, 3) |
      __gen_uint(values->_32PixelDispatchEnable, 2, 2) |
      __gen_uint(values->_16PixelDispatchEnable, 1, 1) |
      __gen_uint(values->_8PixelDispatchEnable, 0, 0);

   dw[6] =
      __gen_float(values->GlobalDepthOffsetConstant);

   dw[7] =
      __gen_float(values->GlobalDepthOffsetScale);

   dw[8] =
      __gen_offset(values->KernelStartPointer1, 6, 31) |
      __gen_uint(values->GRFRegisterCount1, 1, 3);

   dw[9] =
      __gen_offset(values->KernelStartPointer2, 6, 31) |
      __gen_uint(values->GRFRegisterCount2, 1, 3);

   dw[10] =
      __gen_offset(values->KernelStartPointer3, 6, 31) |
      __gen_uint(values->GRFRegisterCount3, 1, 3);
}

#define GEN5_3DPRIMITIVE_length                6
#define GEN5_3DPRIMITIVE_length_bias           2
#define GEN5_3DPRIMITIVE_header                 \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      3,  \
   ._3DCommandSubOpcode                 =      0,  \
   .DWordLength                         =      4

struct GEN5_3DPRIMITIVE {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             VertexAccessType;
#define SEQUENTIAL                               0
#define RANDOM                                   1
   enum GEN5_3D_Prim_Topo_Type          PrimitiveTopologyType;
   uint32_t                             IndirectVertexCount;
   uint32_t                             DWordLength;
   uint32_t                             VertexCountPerInstance;
   uint32_t                             StartVertexLocation;
   uint32_t                             InstanceCount;
   uint32_t                             StartInstanceLocation;
   int32_t                              BaseVertexLocation;
};

static inline void
GEN5_3DPRIMITIVE_pack(__attribute__((unused)) __gen_user_data *data,
                      __attribute__((unused)) void * restrict dst,
                      __attribute__((unused)) const struct GEN5_3DPRIMITIVE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->VertexAccessType, 15, 15) |
      __gen_uint(values->PrimitiveTopologyType, 10, 14) |
      __gen_uint(values->IndirectVertexCount, 9, 9) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_uint(values->VertexCountPerInstance, 0, 31);

   dw[2] =
      __gen_uint(values->StartVertexLocation, 0, 31);

   dw[3] =
      __gen_uint(values->InstanceCount, 0, 31);

   dw[4] =
      __gen_uint(values->StartInstanceLocation, 0, 31);

   dw[5] =
      __gen_sint(values->BaseVertexLocation, 0, 31);
}

#define GEN5_3DSTATE_AA_LINE_PARAMETERS_length      3
#define GEN5_3DSTATE_AA_LINE_PARAMETERS_length_bias      2
#define GEN5_3DSTATE_AA_LINE_PARAMETERS_header  \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =     10,  \
   .DWordLength                         =      1

struct GEN5_3DSTATE_AA_LINE_PARAMETERS {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   float                                AACoverageBias;
   float                                AACoverageSlope;
   float                                AACoverageEndCapBias;
   float                                AACoverageEndCapSlope;
};

static inline void
GEN5_3DSTATE_AA_LINE_PARAMETERS_pack(__attribute__((unused)) __gen_user_data *data,
                                     __attribute__((unused)) void * restrict dst,
                                     __attribute__((unused)) const struct GEN5_3DSTATE_AA_LINE_PARAMETERS * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_ufixed(values->AACoverageBias, 16, 23, 8) |
      __gen_ufixed(values->AACoverageSlope, 0, 7, 8);

   dw[2] =
      __gen_ufixed(values->AACoverageEndCapBias, 16, 23, 8) |
      __gen_ufixed(values->AACoverageEndCapSlope, 0, 7, 8);
}

#define GEN5_3DSTATE_BINDING_TABLE_POINTERS_length      6
#define GEN5_3DSTATE_BINDING_TABLE_POINTERS_length_bias      2
#define GEN5_3DSTATE_BINDING_TABLE_POINTERS_header\
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      0,  \
   ._3DCommandSubOpcode                 =      1,  \
   .DWordLength                         =      4

struct GEN5_3DSTATE_BINDING_TABLE_POINTERS {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   uint64_t                             PointertoVSBindingTable;
   uint64_t                             PointertoGSBindingTable;
   uint64_t                             PointertoCLIPBindingTable;
   uint64_t                             PointertoSFBindingTable;
   uint64_t                             PointertoPSBindingTable;
};

static inline void
GEN5_3DSTATE_BINDING_TABLE_POINTERS_pack(__attribute__((unused)) __gen_user_data *data,
                                         __attribute__((unused)) void * restrict dst,
                                         __attribute__((unused)) const struct GEN5_3DSTATE_BINDING_TABLE_POINTERS * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_offset(values->PointertoVSBindingTable, 5, 31);

   dw[2] =
      __gen_offset(values->PointertoGSBindingTable, 5, 31);

   dw[3] =
      __gen_offset(values->PointertoCLIPBindingTable, 5, 31);

   dw[4] =
      __gen_offset(values->PointertoSFBindingTable, 5, 31);

   dw[5] =
      __gen_offset(values->PointertoPSBindingTable, 5, 31);
}

#define GEN5_3DSTATE_CLEAR_PARAMS_length       2
#define GEN5_3DSTATE_CLEAR_PARAMS_length_bias      2
#define GEN5_3DSTATE_CLEAR_PARAMS_header        \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =     16,  \
   .DWordLength                         =      0

struct GEN5_3DSTATE_CLEAR_PARAMS {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   bool                                 DepthClearValueValid;
   uint32_t                             DWordLength;
   uint32_t                             DepthClearValue;
};

static inline void
GEN5_3DSTATE_CLEAR_PARAMS_pack(__attribute__((unused)) __gen_user_data *data,
                               __attribute__((unused)) void * restrict dst,
                               __attribute__((unused)) const struct GEN5_3DSTATE_CLEAR_PARAMS * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DepthClearValueValid, 15, 15) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_uint(values->DepthClearValue, 0, 31);
}

#define GEN5_3DSTATE_CONSTANT_COLOR_length      5
#define GEN5_3DSTATE_CONSTANT_COLOR_length_bias      2
#define GEN5_3DSTATE_CONSTANT_COLOR_header      \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =      1,  \
   .DWordLength                         =      3

struct GEN5_3DSTATE_CONSTANT_COLOR {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   float                                BlendConstantColorRed;
   float                                BlendConstantColorGreen;
   float                                BlendConstantColorBlue;
   float                                BlendConstantColorAlpha;
};

static inline void
GEN5_3DSTATE_CONSTANT_COLOR_pack(__attribute__((unused)) __gen_user_data *data,
                                 __attribute__((unused)) void * restrict dst,
                                 __attribute__((unused)) const struct GEN5_3DSTATE_CONSTANT_COLOR * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_float(values->BlendConstantColorRed);

   dw[2] =
      __gen_float(values->BlendConstantColorGreen);

   dw[3] =
      __gen_float(values->BlendConstantColorBlue);

   dw[4] =
      __gen_float(values->BlendConstantColorAlpha);
}

#define GEN5_3DSTATE_DRAWING_RECTANGLE_length      4
#define GEN5_3DSTATE_DRAWING_RECTANGLE_length_bias      2
#define GEN5_3DSTATE_DRAWING_RECTANGLE_header   \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =      0,  \
   .DWordLength                         =      2

struct GEN5_3DSTATE_DRAWING_RECTANGLE {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   uint32_t                             ClippedDrawingRectangleYMin;
   uint32_t                             ClippedDrawingRectangleXMin;
   uint32_t                             ClippedDrawingRectangleYMax;
   uint32_t                             ClippedDrawingRectangleXMax;
   int32_t                              DrawingRectangleOriginY;
   int32_t                              DrawingRectangleOriginX;
};

static inline void
GEN5_3DSTATE_DRAWING_RECTANGLE_pack(__attribute__((unused)) __gen_user_data *data,
                                    __attribute__((unused)) void * restrict dst,
                                    __attribute__((unused)) const struct GEN5_3DSTATE_DRAWING_RECTANGLE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_uint(values->ClippedDrawingRectangleYMin, 16, 31) |
      __gen_uint(values->ClippedDrawingRectangleXMin, 0, 15);

   dw[2] =
      __gen_uint(values->ClippedDrawingRectangleYMax, 16, 31) |
      __gen_uint(values->ClippedDrawingRectangleXMax, 0, 15);

   dw[3] =
      __gen_sint(values->DrawingRectangleOriginY, 16, 31) |
      __gen_sint(values->DrawingRectangleOriginX, 0, 15);
}

#define GEN5_3DSTATE_GLOBAL_DEPTH_OFFSET_CLAMP_length      2
#define GEN5_3DSTATE_GLOBAL_DEPTH_OFFSET_CLAMP_length_bias      2
#define GEN5_3DSTATE_GLOBAL_DEPTH_OFFSET_CLAMP_header\
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =      9,  \
   .DWordLength                         =      0

struct GEN5_3DSTATE_GLOBAL_DEPTH_OFFSET_CLAMP {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   float                                GlobalDepthOffsetClamp;
};

static inline void
GEN5_3DSTATE_GLOBAL_DEPTH_OFFSET_CLAMP_pack(__attribute__((unused)) __gen_user_data *data,
                                            __attribute__((unused)) void * restrict dst,
                                            __attribute__((unused)) const struct GEN5_3DSTATE_GLOBAL_DEPTH_OFFSET_CLAMP * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_float(values->GlobalDepthOffsetClamp);
}

#define GEN5_3DSTATE_HIER_DEPTH_BUFFER_length      3
#define GEN5_3DSTATE_HIER_DEPTH_BUFFER_length_bias      2
#define GEN5_3DSTATE_HIER_DEPTH_BUFFER_header   \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =     15,  \
   .DWordLength                         =      1

struct GEN5_3DSTATE_HIER_DEPTH_BUFFER {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   uint32_t                             SurfacePitch;
   __gen_address_type                   SurfaceBaseAddress;
};

static inline void
GEN5_3DSTATE_HIER_DEPTH_BUFFER_pack(__attribute__((unused)) __gen_user_data *data,
                                    __attribute__((unused)) void * restrict dst,
                                    __attribute__((unused)) const struct GEN5_3DSTATE_HIER_DEPTH_BUFFER * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_uint(values->SurfacePitch, 0, 16);

   dw[2] = __gen_combine_address(data, &dw[2], values->SurfaceBaseAddress, 0);
}

#define GEN5_3DSTATE_INDEX_BUFFER_length       3
#define GEN5_3DSTATE_INDEX_BUFFER_length_bias      2
#define GEN5_3DSTATE_INDEX_BUFFER_header        \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      0,  \
   ._3DCommandSubOpcode                 =     10,  \
   .DWordLength                         =      1

struct GEN5_3DSTATE_INDEX_BUFFER {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   bool                                 CutIndexEnable;
   uint32_t                             IndexFormat;
#define INDEX_BYTE                               0
#define INDEX_WORD                               1
#define INDEX_DWORD                              2
   uint32_t                             DWordLength;
   __gen_address_type                   BufferStartingAddress;
   __gen_address_type                   BufferEndingAddress;
};

static inline void
GEN5_3DSTATE_INDEX_BUFFER_pack(__attribute__((unused)) __gen_user_data *data,
                               __attribute__((unused)) void * restrict dst,
                               __attribute__((unused)) const struct GEN5_3DSTATE_INDEX_BUFFER * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->CutIndexEnable, 10, 10) |
      __gen_uint(values->IndexFormat, 8, 9) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] = __gen_combine_address(data, &dw[1], values->BufferStartingAddress, 0);

   dw[2] = __gen_combine_address(data, &dw[2], values->BufferEndingAddress, 0);
}

#define GEN5_3DSTATE_LINE_STIPPLE_length       3
#define GEN5_3DSTATE_LINE_STIPPLE_length_bias      2
#define GEN5_3DSTATE_LINE_STIPPLE_header        \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =      8,  \
   .DWordLength                         =      1

struct GEN5_3DSTATE_LINE_STIPPLE {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   bool                                 ModifyEnable;
   uint32_t                             CurrentRepeatCounter;
   uint32_t                             CurrentStippleIndex;
   uint32_t                             LineStipplePattern;
   float                                LineStippleInverseRepeatCount;
   uint32_t                             LineStippleRepeatCount;
};

static inline void
GEN5_3DSTATE_LINE_STIPPLE_pack(__attribute__((unused)) __gen_user_data *data,
                               __attribute__((unused)) void * restrict dst,
                               __attribute__((unused)) const struct GEN5_3DSTATE_LINE_STIPPLE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_uint(values->ModifyEnable, 31, 31) |
      __gen_uint(values->CurrentRepeatCounter, 21, 29) |
      __gen_uint(values->CurrentStippleIndex, 16, 19) |
      __gen_uint(values->LineStipplePattern, 0, 15);

   dw[2] =
      __gen_ufixed(values->LineStippleInverseRepeatCount, 16, 31, 13) |
      __gen_uint(values->LineStippleRepeatCount, 0, 8);
}

#define GEN5_3DSTATE_POLY_STIPPLE_OFFSET_length      2
#define GEN5_3DSTATE_POLY_STIPPLE_OFFSET_length_bias      2
#define GEN5_3DSTATE_POLY_STIPPLE_OFFSET_header \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =      6,  \
   .DWordLength                         =      0

struct GEN5_3DSTATE_POLY_STIPPLE_OFFSET {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   uint32_t                             PolygonStippleXOffset;
   uint32_t                             PolygonStippleYOffset;
};

static inline void
GEN5_3DSTATE_POLY_STIPPLE_OFFSET_pack(__attribute__((unused)) __gen_user_data *data,
                                      __attribute__((unused)) void * restrict dst,
                                      __attribute__((unused)) const struct GEN5_3DSTATE_POLY_STIPPLE_OFFSET * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_uint(values->PolygonStippleXOffset, 8, 12) |
      __gen_uint(values->PolygonStippleYOffset, 0, 4);
}

#define GEN5_3DSTATE_POLY_STIPPLE_PATTERN_length     33
#define GEN5_3DSTATE_POLY_STIPPLE_PATTERN_length_bias      2
#define GEN5_3DSTATE_POLY_STIPPLE_PATTERN_header\
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =      7,  \
   .DWordLength                         =     31

struct GEN5_3DSTATE_POLY_STIPPLE_PATTERN {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   uint32_t                             PatternRow[32];
};

static inline void
GEN5_3DSTATE_POLY_STIPPLE_PATTERN_pack(__attribute__((unused)) __gen_user_data *data,
                                       __attribute__((unused)) void * restrict dst,
                                       __attribute__((unused)) const struct GEN5_3DSTATE_POLY_STIPPLE_PATTERN * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_uint(values->PatternRow[0], 0, 31);

   dw[2] =
      __gen_uint(values->PatternRow[1], 0, 31);

   dw[3] =
      __gen_uint(values->PatternRow[2], 0, 31);

   dw[4] =
      __gen_uint(values->PatternRow[3], 0, 31);

   dw[5] =
      __gen_uint(values->PatternRow[4], 0, 31);

   dw[6] =
      __gen_uint(values->PatternRow[5], 0, 31);

   dw[7] =
      __gen_uint(values->PatternRow[6], 0, 31);

   dw[8] =
      __gen_uint(values->PatternRow[7], 0, 31);

   dw[9] =
      __gen_uint(values->PatternRow[8], 0, 31);

   dw[10] =
      __gen_uint(values->PatternRow[9], 0, 31);

   dw[11] =
      __gen_uint(values->PatternRow[10], 0, 31);

   dw[12] =
      __gen_uint(values->PatternRow[11], 0, 31);

   dw[13] =
      __gen_uint(values->PatternRow[12], 0, 31);

   dw[14] =
      __gen_uint(values->PatternRow[13], 0, 31);

   dw[15] =
      __gen_uint(values->PatternRow[14], 0, 31);

   dw[16] =
      __gen_uint(values->PatternRow[15], 0, 31);

   dw[17] =
      __gen_uint(values->PatternRow[16], 0, 31);

   dw[18] =
      __gen_uint(values->PatternRow[17], 0, 31);

   dw[19] =
      __gen_uint(values->PatternRow[18], 0, 31);

   dw[20] =
      __gen_uint(values->PatternRow[19], 0, 31);

   dw[21] =
      __gen_uint(values->PatternRow[20], 0, 31);

   dw[22] =
      __gen_uint(values->PatternRow[21], 0, 31);

   dw[23] =
      __gen_uint(values->PatternRow[22], 0, 31);

   dw[24] =
      __gen_uint(values->PatternRow[23], 0, 31);

   dw[25] =
      __gen_uint(values->PatternRow[24], 0, 31);

   dw[26] =
      __gen_uint(values->PatternRow[25], 0, 31);

   dw[27] =
      __gen_uint(values->PatternRow[26], 0, 31);

   dw[28] =
      __gen_uint(values->PatternRow[27], 0, 31);

   dw[29] =
      __gen_uint(values->PatternRow[28], 0, 31);

   dw[30] =
      __gen_uint(values->PatternRow[29], 0, 31);

   dw[31] =
      __gen_uint(values->PatternRow[30], 0, 31);

   dw[32] =
      __gen_uint(values->PatternRow[31], 0, 31);
}

#define GEN5_3DSTATE_PIPELINED_POINTERS_length      7
#define GEN5_3DSTATE_PIPELINED_POINTERS_length_bias      2
#define GEN5_3DSTATE_PIPELINED_POINTERS_header  \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      0,  \
   ._3DCommandSubOpcode                 =      0,  \
   .DWordLength                         =      5

struct GEN5_3DSTATE_PIPELINED_POINTERS {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   __gen_address_type                   PointertoVSState;
   __gen_address_type                   PointertoGSState;
   bool                                 GSEnable;
   __gen_address_type                   PointertoCLIPState;
   bool                                 ClipEnable;
   __gen_address_type                   PointertoSFState;
   __gen_address_type                   PointertoWMState;
   __gen_address_type                   PointertoColorCalcState;
};

static inline void
GEN5_3DSTATE_PIPELINED_POINTERS_pack(__attribute__((unused)) __gen_user_data *data,
                                     __attribute__((unused)) void * restrict dst,
                                     __attribute__((unused)) const struct GEN5_3DSTATE_PIPELINED_POINTERS * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] = __gen_combine_address(data, &dw[1], values->PointertoVSState, 0);

   const uint32_t v2 =
      __gen_uint(values->GSEnable, 0, 0);
   dw[2] = __gen_combine_address(data, &dw[2], values->PointertoGSState, v2);

   const uint32_t v3 =
      __gen_uint(values->ClipEnable, 0, 0);
   dw[3] = __gen_combine_address(data, &dw[3], values->PointertoCLIPState, v3);

   dw[4] = __gen_combine_address(data, &dw[4], values->PointertoSFState, 0);

   dw[5] = __gen_combine_address(data, &dw[5], values->PointertoWMState, 0);

   dw[6] = __gen_combine_address(data, &dw[6], values->PointertoColorCalcState, 0);
}

#define GEN5_3DSTATE_STENCIL_BUFFER_length      3
#define GEN5_3DSTATE_STENCIL_BUFFER_length_bias      2
#define GEN5_3DSTATE_STENCIL_BUFFER_header      \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =     14,  \
   .DWordLength                         =      1

struct GEN5_3DSTATE_STENCIL_BUFFER {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   uint32_t                             SurfacePitch;
   __gen_address_type                   SurfaceBaseAddress;
};

static inline void
GEN5_3DSTATE_STENCIL_BUFFER_pack(__attribute__((unused)) __gen_user_data *data,
                                 __attribute__((unused)) void * restrict dst,
                                 __attribute__((unused)) const struct GEN5_3DSTATE_STENCIL_BUFFER * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_uint(values->SurfacePitch, 0, 16);

   dw[2] = __gen_combine_address(data, &dw[2], values->SurfaceBaseAddress, 0);
}

#define GEN5_3DSTATE_VERTEX_BUFFERS_length_bias      2
#define GEN5_3DSTATE_VERTEX_BUFFERS_header      \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      0,  \
   ._3DCommandSubOpcode                 =      8,  \
   .DWordLength                         =      3

struct GEN5_3DSTATE_VERTEX_BUFFERS {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   /* variable length fields follow */
};

static inline void
GEN5_3DSTATE_VERTEX_BUFFERS_pack(__attribute__((unused)) __gen_user_data *data,
                                 __attribute__((unused)) void * restrict dst,
                                 __attribute__((unused)) const struct GEN5_3DSTATE_VERTEX_BUFFERS * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);
}

#define GEN5_3DSTATE_VERTEX_ELEMENTS_length_bias      2
#define GEN5_3DSTATE_VERTEX_ELEMENTS_header     \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      0,  \
   ._3DCommandSubOpcode                 =      9,  \
   .DWordLength                         =      1

struct GEN5_3DSTATE_VERTEX_ELEMENTS {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   /* variable length fields follow */
};

static inline void
GEN5_3DSTATE_VERTEX_ELEMENTS_pack(__attribute__((unused)) __gen_user_data *data,
                                  __attribute__((unused)) void * restrict dst,
                                  __attribute__((unused)) const struct GEN5_3DSTATE_VERTEX_ELEMENTS * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);
}

#define GEN5_3DSTATE_VF_STATISTICS_length      1
#define GEN5_3DSTATE_VF_STATISTICS_length_bias      1
#define GEN5_3DSTATE_VF_STATISTICS_header       \
   .CommandType                         =      3,  \
   .CommandSubType                      =      1,  \
   ._3DCommandOpcode                    =      0,  \
   ._3DCommandSubOpcode                 =     11

struct GEN5_3DSTATE_VF_STATISTICS {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   bool                                 StatisticsEnable;
};

static inline void
GEN5_3DSTATE_VF_STATISTICS_pack(__attribute__((unused)) __gen_user_data *data,
                                __attribute__((unused)) void * restrict dst,
                                __attribute__((unused)) const struct GEN5_3DSTATE_VF_STATISTICS * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->StatisticsEnable, 0, 0);
}

#define GEN5_CS_URB_STATE_length               2
#define GEN5_CS_URB_STATE_length_bias          2
#define GEN5_CS_URB_STATE_header                \
   .CommandType                         =      3,  \
   .CommandSubType                      =      0,  \
   ._3DCommandOpcode                    =      0,  \
   ._3DCommandSubOpcode                 =      1,  \
   .DWordLength                         =      0

struct GEN5_CS_URB_STATE {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   uint32_t                             URBEntryAllocationSize;
   uint32_t                             NumberofURBEntries;
};

static inline void
GEN5_CS_URB_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                       __attribute__((unused)) void * restrict dst,
                       __attribute__((unused)) const struct GEN5_CS_URB_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_uint(values->URBEntryAllocationSize, 4, 8) |
      __gen_uint(values->NumberofURBEntries, 0, 2);
}

#define GEN5_CONSTANT_BUFFER_length            2
#define GEN5_CONSTANT_BUFFER_length_bias       2
#define GEN5_CONSTANT_BUFFER_header             \
   .CommandType                         =      3,  \
   .CommandSubType                      =      0,  \
   ._3DCommandOpcode                    =      0,  \
   ._3DCommandSubOpcode                 =      2,  \
   .DWordLength                         =      0

struct GEN5_CONSTANT_BUFFER {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   bool                                 Valid;
   __gen_address_type                   BufferStartingAddress;
   uint32_t                             BufferLength;
};

static inline void
GEN5_CONSTANT_BUFFER_pack(__attribute__((unused)) __gen_user_data *data,
                          __attribute__((unused)) void * restrict dst,
                          __attribute__((unused)) const struct GEN5_CONSTANT_BUFFER * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7) |
      __gen_uint(values->Valid, 8, 8);

   const uint32_t v1 =
      __gen_uint(values->BufferLength, 0, 5);
   dw[1] = __gen_combine_address(data, &dw[1], values->BufferStartingAddress, v1);
}

#define GEN5_MI_FLUSH_length                   1
#define GEN5_MI_FLUSH_length_bias              1
#define GEN5_MI_FLUSH_header                    \
   .CommandType                         =      0,  \
   .MICommandOpcode                     =      4

struct GEN5_MI_FLUSH {
   uint32_t                             CommandType;
   uint32_t                             MICommandOpcode;
   bool                                 ProtectedMemoryEnable;
   bool                                 IndirectStatePointersDisable;
   bool                                 GenericMediaStateClear;
   uint32_t                             GlobalSnapshotCountReset;
#define DontReset                                0
#define Reset                                    1
   uint32_t                             RenderCacheFlushInhibit;
#define Flush                                    0
#define DontFlush                                1
   uint32_t                             StateInstructionCacheInvalidate;
#define DontInvalidate                           0
#define Invalidate                               1
};

static inline void
GEN5_MI_FLUSH_pack(__attribute__((unused)) __gen_user_data *data,
                   __attribute__((unused)) void * restrict dst,
                   __attribute__((unused)) const struct GEN5_MI_FLUSH * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->MICommandOpcode, 23, 28) |
      __gen_uint(values->ProtectedMemoryEnable, 6, 6) |
      __gen_uint(values->IndirectStatePointersDisable, 5, 5) |
      __gen_uint(values->GenericMediaStateClear, 4, 4) |
      __gen_uint(values->GlobalSnapshotCountReset, 3, 3) |
      __gen_uint(values->RenderCacheFlushInhibit, 2, 2) |
      __gen_uint(values->StateInstructionCacheInvalidate, 1, 1);
}

#define GEN5_MI_LOAD_REGISTER_IMM_length       3
#define GEN5_MI_LOAD_REGISTER_IMM_length_bias      2
#define GEN5_MI_LOAD_REGISTER_IMM_header        \
   .CommandType                         =      0,  \
   .MICommandOpcode                     =     34,  \
   .DWordLength                         =      1

struct GEN5_MI_LOAD_REGISTER_IMM {
   uint32_t                             CommandType;
   uint32_t                             MICommandOpcode;
   uint32_t                             ByteWriteDisables;
   uint32_t                             DWordLength;
   uint64_t                             RegisterOffset;
   uint32_t                             DataDWord;
   /* variable length fields follow */
};

static inline void
GEN5_MI_LOAD_REGISTER_IMM_pack(__attribute__((unused)) __gen_user_data *data,
                               __attribute__((unused)) void * restrict dst,
                               __attribute__((unused)) const struct GEN5_MI_LOAD_REGISTER_IMM * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->MICommandOpcode, 23, 28) |
      __gen_uint(values->ByteWriteDisables, 8, 11) |
      __gen_uint(values->DWordLength, 0, 5);

   dw[1] =
      __gen_offset(values->RegisterOffset, 2, 31);

   dw[2] =
      __gen_uint(values->DataDWord, 0, 31);
}

#define GEN5_MI_STORE_DATA_IMM_length          5
#define GEN5_MI_STORE_DATA_IMM_length_bias      2
#define GEN5_MI_STORE_DATA_IMM_header           \
   .CommandType                         =      0,  \
   .MICommandOpcode                     =     32,  \
   .DWordLength                         =      2

struct GEN5_MI_STORE_DATA_IMM {
   uint32_t                             CommandType;
   uint32_t                             MICommandOpcode;
   bool                                 MemoryAddressType;
   uint32_t                             DWordLength;
   __gen_address_type                   PhysicalStartAddressExtension;
   __gen_address_type                   Address;
   uint32_t                             DataDWord0;
   uint32_t                             DataDWord1;
};

static inline void
GEN5_MI_STORE_DATA_IMM_pack(__attribute__((unused)) __gen_user_data *data,
                            __attribute__((unused)) void * restrict dst,
                            __attribute__((unused)) const struct GEN5_MI_STORE_DATA_IMM * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->MICommandOpcode, 23, 28) |
      __gen_uint(values->MemoryAddressType, 22, 22) |
      __gen_uint(values->DWordLength, 0, 5);

   dw[1] = __gen_combine_address(data, &dw[1], values->PhysicalStartAddressExtension, 0);

   dw[2] = __gen_combine_address(data, &dw[2], values->Address, 0);

   dw[3] =
      __gen_uint(values->DataDWord0, 0, 31);

   dw[4] =
      __gen_uint(values->DataDWord1, 0, 31);
}

#define GEN5_MI_STORE_REGISTER_MEM_length      3
#define GEN5_MI_STORE_REGISTER_MEM_length_bias      2
#define GEN5_MI_STORE_REGISTER_MEM_header       \
   .CommandType                         =      0,  \
   .MICommandOpcode                     =     36,  \
   .DWordLength                         =      1

struct GEN5_MI_STORE_REGISTER_MEM {
   uint32_t                             CommandType;
   uint32_t                             MICommandOpcode;
   bool                                 UseGlobalGTT;
   uint32_t                             DWordLength;
   uint64_t                             RegisterAddress;
   __gen_address_type                   MemoryAddress;
};

static inline void
GEN5_MI_STORE_REGISTER_MEM_pack(__attribute__((unused)) __gen_user_data *data,
                                __attribute__((unused)) void * restrict dst,
                                __attribute__((unused)) const struct GEN5_MI_STORE_REGISTER_MEM * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->MICommandOpcode, 23, 28) |
      __gen_uint(values->UseGlobalGTT, 22, 22) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_offset(values->RegisterAddress, 2, 25);

   dw[2] = __gen_combine_address(data, &dw[2], values->MemoryAddress, 0);
}

#define GEN5_PIPE_CONTROL_length               4
#define GEN5_PIPE_CONTROL_length_bias          2
#define GEN5_PIPE_CONTROL_header                \
   .CommandType                         =      3,  \
   .CommandSubType                      =      3,  \
   ._3DCommandOpcode                    =      2,  \
   ._3DCommandSubOpcode                 =      0,  \
   .DWordLength                         =      2

struct GEN5_PIPE_CONTROL {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   uint32_t                             PostSyncOperation;
#define NoWrite                                  0
#define WriteImmediateData                       1
#define WritePSDepthCount                        2
#define WriteTimestamp                           3
   bool                                 DepthStallEnable;
   bool                                 WriteCacheFlush;
   bool                                 InstructionCacheInvalidateEnable;
   bool                                 TextureCacheFlushEnable;
   bool                                 IndirectStatePointersDisable;
   bool                                 NotifyEnable;
   __gen_address_type                   Address;
   uint32_t                             DestinationAddressType;
#define DAT_PGTT                                 0
#define DAT_GGTT                                 1
   bool                                 StallAtPixelScoreboard;
   uint32_t                             DepthCacheFlushInhibit;
#define Flushed                                  0
#define NotFlushed                               1
   uint64_t                             ImmediateData;
};

static inline void
GEN5_PIPE_CONTROL_pack(__attribute__((unused)) __gen_user_data *data,
                       __attribute__((unused)) void * restrict dst,
                       __attribute__((unused)) const struct GEN5_PIPE_CONTROL * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7) |
      __gen_uint(values->PostSyncOperation, 14, 15) |
      __gen_uint(values->DepthStallEnable, 13, 13) |
      __gen_uint(values->WriteCacheFlush, 12, 12) |
      __gen_uint(values->InstructionCacheInvalidateEnable, 11, 11) |
      __gen_uint(values->TextureCacheFlushEnable, 10, 10) |
      __gen_uint(values->IndirectStatePointersDisable, 9, 9) |
      __gen_uint(values->NotifyEnable, 8, 8);

   const uint32_t v1 =
      __gen_uint(values->DestinationAddressType, 2, 2) |
      __gen_uint(values->StallAtPixelScoreboard, 1, 1) |
      __gen_uint(values->DepthCacheFlushInhibit, 0, 0);
   dw[1] = __gen_combine_address(data, &dw[1], values->Address, v1);

   const uint64_t v2 =
      __gen_uint(values->ImmediateData, 0, 63);
   dw[2] = v2;
   dw[3] = v2 >> 32;
}

#define GEN5_PIPELINE_SELECT_length            1
#define GEN5_PIPELINE_SELECT_length_bias       1
#define GEN5_PIPELINE_SELECT_header             \
   .CommandType                         =      3,  \
   .CommandSubType                      =      1,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =      4

struct GEN5_PIPELINE_SELECT {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             PipelineSelection;
#define _3D                                      0
#define Media                                    1
#define GPGPU                                    2
};

static inline void
GEN5_PIPELINE_SELECT_pack(__attribute__((unused)) __gen_user_data *data,
                          __attribute__((unused)) void * restrict dst,
                          __attribute__((unused)) const struct GEN5_PIPELINE_SELECT * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->PipelineSelection, 0, 1);
}

#define GEN5_STATE_BASE_ADDRESS_length         8
#define GEN5_STATE_BASE_ADDRESS_length_bias      2
#define GEN5_STATE_BASE_ADDRESS_header          \
   .CommandType                         =      3,  \
   .CommandSubType                      =      0,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =      1,  \
   .DWordLength                         =      6

struct GEN5_STATE_BASE_ADDRESS {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   __gen_address_type                   GeneralStateBaseAddress;
   bool                                 GeneralStateBaseAddressModifyEnable;
   __gen_address_type                   SurfaceStateBaseAddress;
   bool                                 SurfaceStateBaseAddressModifyEnable;
   __gen_address_type                   IndirectObjectBaseAddress;
   bool                                 IndirectObjectBaseAddressModifyEnable;
   __gen_address_type                   InstructionBaseAddress;
   bool                                 InstructionBaseAddressModifyEnable;
   __gen_address_type                   GeneralStateAccessUpperBound;
   bool                                 GeneralStateAccessUpperBoundModifyEnable;
   __gen_address_type                   IndirectObjectAccessUpperBound;
   bool                                 IndirectObjectAccessUpperBoundModifyEnable;
   __gen_address_type                   InstructionAccessUpperBound;
   bool                                 InstructionAccessUpperBoundModifyEnable;
};

static inline void
GEN5_STATE_BASE_ADDRESS_pack(__attribute__((unused)) __gen_user_data *data,
                             __attribute__((unused)) void * restrict dst,
                             __attribute__((unused)) const struct GEN5_STATE_BASE_ADDRESS * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   const uint32_t v1 =
      __gen_uint(values->GeneralStateBaseAddressModifyEnable, 0, 0);
   dw[1] = __gen_combine_address(data, &dw[1], values->GeneralStateBaseAddress, v1);

   const uint32_t v2 =
      __gen_uint(values->SurfaceStateBaseAddressModifyEnable, 0, 0);
   dw[2] = __gen_combine_address(data, &dw[2], values->SurfaceStateBaseAddress, v2);

   const uint32_t v3 =
      __gen_uint(values->IndirectObjectBaseAddressModifyEnable, 0, 0);
   dw[3] = __gen_combine_address(data, &dw[3], values->IndirectObjectBaseAddress, v3);

   const uint32_t v4 =
      __gen_uint(values->InstructionBaseAddressModifyEnable, 0, 0);
   dw[4] = __gen_combine_address(data, &dw[4], values->InstructionBaseAddress, v4);

   const uint32_t v5 =
      __gen_uint(values->GeneralStateAccessUpperBoundModifyEnable, 0, 0);
   dw[5] = __gen_combine_address(data, &dw[5], values->GeneralStateAccessUpperBound, v5);

   const uint32_t v6 =
      __gen_uint(values->IndirectObjectAccessUpperBoundModifyEnable, 0, 0);
   dw[6] = __gen_combine_address(data, &dw[6], values->IndirectObjectAccessUpperBound, v6);

   const uint32_t v7 =
      __gen_uint(values->InstructionAccessUpperBoundModifyEnable, 0, 0);
   dw[7] = __gen_combine_address(data, &dw[7], values->InstructionAccessUpperBound, v7);
}

#define GEN5_STATE_SIP_length                  2
#define GEN5_STATE_SIP_length_bias             2
#define GEN5_STATE_SIP_header                   \
   .CommandType                         =      3,  \
   .CommandSubType                      =      0,  \
   ._3DCommandOpcode                    =      1,  \
   ._3DCommandSubOpcode                 =      2,  \
   .DWordLength                         =      0

struct GEN5_STATE_SIP {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   uint64_t                             SystemInstructionPointer;
};

static inline void
GEN5_STATE_SIP_pack(__attribute__((unused)) __gen_user_data *data,
                    __attribute__((unused)) void * restrict dst,
                    __attribute__((unused)) const struct GEN5_STATE_SIP * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7);

   dw[1] =
      __gen_offset(values->SystemInstructionPointer, 4, 31);
}

#define GEN5_URB_FENCE_length                  3
#define GEN5_URB_FENCE_length_bias             2
#define GEN5_URB_FENCE_header                   \
   .CommandType                         =      3,  \
   .CommandSubType                      =      0,  \
   ._3DCommandOpcode                    =      0,  \
   ._3DCommandSubOpcode                 =      0,  \
   .DWordLength                         =      1

struct GEN5_URB_FENCE {
   uint32_t                             CommandType;
   uint32_t                             CommandSubType;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             DWordLength;
   bool                                 CSUnitURBReallocationRequest;
   bool                                 VFEUnitURBReallocationRequest;
   bool                                 SFUnitURBReallocationRequest;
   bool                                 CLIPUnitURBReallocationRequest;
   bool                                 GSUnitURBReallocationRequest;
   bool                                 VSUnitURBReallocationRequest;
   uint32_t                             CLIPFence;
   uint32_t                             GSFence;
   uint32_t                             VSFence;
   uint32_t                             CSFence;
   uint32_t                             VFEFence;
   uint32_t                             SFFence;
};

static inline void
GEN5_URB_FENCE_pack(__attribute__((unused)) __gen_user_data *data,
                    __attribute__((unused)) void * restrict dst,
                    __attribute__((unused)) const struct GEN5_URB_FENCE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->CommandSubType, 27, 28) |
      __gen_uint(values->_3DCommandOpcode, 24, 26) |
      __gen_uint(values->_3DCommandSubOpcode, 16, 23) |
      __gen_uint(values->DWordLength, 0, 7) |
      __gen_uint(values->CSUnitURBReallocationRequest, 13, 13) |
      __gen_uint(values->VFEUnitURBReallocationRequest, 12, 12) |
      __gen_uint(values->SFUnitURBReallocationRequest, 11, 11) |
      __gen_uint(values->CLIPUnitURBReallocationRequest, 10, 10) |
      __gen_uint(values->GSUnitURBReallocationRequest, 9, 9) |
      __gen_uint(values->VSUnitURBReallocationRequest, 8, 8);

   dw[1] =
      __gen_uint(values->CLIPFence, 20, 29) |
      __gen_uint(values->GSFence, 10, 19) |
      __gen_uint(values->VSFence, 0, 9);

   dw[2] =
      __gen_uint(values->CSFence, 20, 30) |
      __gen_uint(values->VFEFence, 10, 19) |
      __gen_uint(values->SFFence, 0, 9);
}

#define GEN5_XY_COLOR_BLT_length               6
#define GEN5_XY_COLOR_BLT_length_bias          2
#define GEN5_XY_COLOR_BLT_header                \
   .CommandType                         =      2,  \
   ._2DCommandOpcode                    =     80,  \
   .DWordLength                         =      4

struct GEN5_XY_COLOR_BLT {
   uint32_t                             CommandType;
   uint32_t                             _2DCommandOpcode;
   uint32_t                             DWordLength;
   uint32_t                             _32bppByteMask;
   bool                                 TilingEnable;
   bool                                 ClippingEnabled;
   uint32_t                             ColorDepth;
#define COLOR_DEPTH__8bit                        0
#define COLOR_DEPTH__565                         1
#define COLOR_DEPTH__1555                        2
#define COLOR_DEPTH__32bit                       3
   int32_t                              RasterOperation;
   int32_t                              DestinationPitch;
   int32_t                              DestinationY1Coordinate;
   int32_t                              DestinationX1Coordinate;
   int32_t                              DestinationY2Coordinate;
   int32_t                              DestinationX2Coordinate;
   __gen_address_type                   DestinationBaseAddress;
   int32_t                              SolidPatternColor;
};

static inline void
GEN5_XY_COLOR_BLT_pack(__attribute__((unused)) __gen_user_data *data,
                       __attribute__((unused)) void * restrict dst,
                       __attribute__((unused)) const struct GEN5_XY_COLOR_BLT * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->_2DCommandOpcode, 22, 28) |
      __gen_uint(values->DWordLength, 0, 7) |
      __gen_uint(values->_32bppByteMask, 20, 21) |
      __gen_uint(values->TilingEnable, 11, 11);

   dw[1] =
      __gen_uint(values->ClippingEnabled, 30, 30) |
      __gen_uint(values->ColorDepth, 24, 26) |
      __gen_sint(values->RasterOperation, 16, 23) |
      __gen_sint(values->DestinationPitch, 0, 15);

   dw[2] =
      __gen_sint(values->DestinationY1Coordinate, 16, 31) |
      __gen_sint(values->DestinationX1Coordinate, 0, 15);

   dw[3] =
      __gen_sint(values->DestinationY2Coordinate, 16, 31) |
      __gen_sint(values->DestinationX2Coordinate, 0, 15);

   dw[4] = __gen_combine_address(data, &dw[4], values->DestinationBaseAddress, 0);

   dw[5] =
      __gen_sint(values->SolidPatternColor, 0, 31);
}

#define GEN5_XY_SETUP_BLT_length               8
#define GEN5_XY_SETUP_BLT_length_bias          2
#define GEN5_XY_SETUP_BLT_header                \
   .CommandType                         =      2,  \
   ._2DCommandOpcode                    =      1,  \
   .DWordLength                         =      6

struct GEN5_XY_SETUP_BLT {
   uint32_t                             CommandType;
   uint32_t                             _2DCommandOpcode;
   uint32_t                             DWordLength;
   uint32_t                             _32bppByteMask;
   bool                                 TilingEnable;
   bool                                 ClippingEnabled;
   bool                                 MonoSourceTransparencyMode;
   uint32_t                             ColorDepth;
#define COLOR_DEPTH__8bit                        0
#define COLOR_DEPTH__565                         1
#define COLOR_DEPTH__1555                        2
#define COLOR_DEPTH__32bit                       3
   int32_t                              RasterOperation;
   int32_t                              DestinationPitch;
   int32_t                              ClipRectY1Coordinate;
   int32_t                              ClipRectX1Coordinate;
   int32_t                              ClipRectY2Coordinate;
   int32_t                              ClipRectX2Coordinate;
   __gen_address_type                   DestinationBaseAddress;
   uint32_t                             BackgroundColor;
   uint32_t                             ForegroundColor;
   uint32_t                             PatternBaseAddress;
};

static inline void
GEN5_XY_SETUP_BLT_pack(__attribute__((unused)) __gen_user_data *data,
                       __attribute__((unused)) void * restrict dst,
                       __attribute__((unused)) const struct GEN5_XY_SETUP_BLT * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->_2DCommandOpcode, 22, 28) |
      __gen_uint(values->DWordLength, 0, 7) |
      __gen_uint(values->_32bppByteMask, 20, 21) |
      __gen_uint(values->TilingEnable, 11, 11);

   dw[1] =
      __gen_uint(values->ClippingEnabled, 30, 30) |
      __gen_uint(values->MonoSourceTransparencyMode, 28, 28) |
      __gen_uint(values->ColorDepth, 24, 25) |
      __gen_sint(values->RasterOperation, 16, 23) |
      __gen_sint(values->DestinationPitch, 0, 15);

   dw[2] =
      __gen_sint(values->ClipRectY1Coordinate, 16, 31) |
      __gen_sint(values->ClipRectX1Coordinate, 0, 15);

   dw[3] =
      __gen_sint(values->ClipRectY2Coordinate, 16, 31) |
      __gen_sint(values->ClipRectX2Coordinate, 0, 15);

   dw[4] = __gen_combine_address(data, &dw[4], values->DestinationBaseAddress, 0);

   dw[5] =
      __gen_uint(values->BackgroundColor, 0, 31);

   dw[6] =
      __gen_uint(values->ForegroundColor, 0, 31);

   dw[7] =
      __gen_uint(values->PatternBaseAddress, 0, 31);
}

#define GEN5_XY_SRC_COPY_BLT_length            8
#define GEN5_XY_SRC_COPY_BLT_length_bias       2
#define GEN5_XY_SRC_COPY_BLT_header             \
   .CommandType                         =      2,  \
   ._2DCommandOpcode                    =     83,  \
   .DWordLength                         =      6

struct GEN5_XY_SRC_COPY_BLT {
   uint32_t                             CommandType;
   uint32_t                             _2DCommandOpcode;
   uint32_t                             DWordLength;
   uint32_t                             _32bppByteMask;
   bool                                 SourceTilingEnable;
   bool                                 DestinationTilingEnable;
   bool                                 ClippingEnabled;
   uint32_t                             ColorDepth;
#define COLOR_DEPTH__8bit                        0
#define COLOR_DEPTH__565                         1
#define COLOR_DEPTH__1555                        2
#define COLOR_DEPTH__32bit                       3
   int32_t                              RasterOperation;
   int32_t                              DestinationPitch;
   int32_t                              DestinationY1Coordinate;
   int32_t                              DestinationX1Coordinate;
   int32_t                              DestinationY2Coordinate;
   int32_t                              DestinationX2Coordinate;
   __gen_address_type                   DestinationBaseAddress;
   int32_t                              SourceY1Coordinate;
   int32_t                              SourceX1Coordinate;
   int32_t                              SourcePitch;
   __gen_address_type                   SourceBaseAddress;
};

static inline void
GEN5_XY_SRC_COPY_BLT_pack(__attribute__((unused)) __gen_user_data *data,
                          __attribute__((unused)) void * restrict dst,
                          __attribute__((unused)) const struct GEN5_XY_SRC_COPY_BLT * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->_2DCommandOpcode, 22, 28) |
      __gen_uint(values->DWordLength, 0, 7) |
      __gen_uint(values->_32bppByteMask, 20, 21) |
      __gen_uint(values->SourceTilingEnable, 15, 15) |
      __gen_uint(values->DestinationTilingEnable, 11, 11);

   dw[1] =
      __gen_uint(values->ClippingEnabled, 30, 30) |
      __gen_uint(values->ColorDepth, 24, 26) |
      __gen_sint(values->RasterOperation, 16, 23) |
      __gen_sint(values->DestinationPitch, 0, 15);

   dw[2] =
      __gen_sint(values->DestinationY1Coordinate, 16, 31) |
      __gen_sint(values->DestinationX1Coordinate, 0, 15);

   dw[3] =
      __gen_sint(values->DestinationY2Coordinate, 16, 31) |
      __gen_sint(values->DestinationX2Coordinate, 0, 15);

   dw[4] = __gen_combine_address(data, &dw[4], values->DestinationBaseAddress, 0);

   dw[5] =
      __gen_sint(values->SourceY1Coordinate, 16, 31) |
      __gen_sint(values->SourceX1Coordinate, 0, 15);

   dw[6] =
      __gen_sint(values->SourcePitch, 0, 15);

   dw[7] = __gen_combine_address(data, &dw[7], values->SourceBaseAddress, 0);
}

#define GEN5_XY_TEXT_IMMEDIATE_BLT_length      3
#define GEN5_XY_TEXT_IMMEDIATE_BLT_length_bias      2
#define GEN5_XY_TEXT_IMMEDIATE_BLT_header       \
   .CommandType                         =      2,  \
   ._2DCommandOpcode                    =     49,  \
   .DWordLength                         =      1

struct GEN5_XY_TEXT_IMMEDIATE_BLT {
   uint32_t                             CommandType;
   uint32_t                             _2DCommandOpcode;
   uint32_t                             DWordLength;
   uint32_t                             _32bppByteMask;
   uint32_t                             Packing;
#define BitPacked                                0
#define BytePacked                               1
   bool                                 TilingEnable;
   int32_t                              DestinationPitch;
   int32_t                              DestinationY1Coordinate;
   int32_t                              DestinationX1Coordinate;
   int32_t                              DestinationY2Coordinate;
   int32_t                              DestinationX2Coordinate;
};

static inline void
GEN5_XY_TEXT_IMMEDIATE_BLT_pack(__attribute__((unused)) __gen_user_data *data,
                                __attribute__((unused)) void * restrict dst,
                                __attribute__((unused)) const struct GEN5_XY_TEXT_IMMEDIATE_BLT * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_uint(values->CommandType, 29, 31) |
      __gen_uint(values->_2DCommandOpcode, 22, 28) |
      __gen_uint(values->DWordLength, 0, 7) |
      __gen_uint(values->_32bppByteMask, 20, 21) |
      __gen_uint(values->Packing, 16, 16) |
      __gen_uint(values->TilingEnable, 11, 11);

   dw[1] =
      __gen_sint(values->DestinationPitch, 0, 15) |
      __gen_sint(values->DestinationY1Coordinate, 16, 31) |
      __gen_sint(values->DestinationX1Coordinate, 0, 15);

   dw[2] =
      __gen_sint(values->DestinationY2Coordinate, 16, 31) |
      __gen_sint(values->DestinationX2Coordinate, 0, 15);
}

#endif /* GEN5_PACK_H */
