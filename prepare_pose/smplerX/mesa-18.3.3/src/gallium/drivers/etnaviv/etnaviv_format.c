/*
 * Copyright (c) 2016 Etnaviv Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Christian Gmeiner <christian.gmeiner@gmail.com>
 */

#include "etnaviv_format.h"

#include "hw/common_3d.xml.h"
#include "hw/state.xml.h"
#include "hw/state_3d.xml.h"

#include "pipe/p_defines.h"

/* Specifies the table of all the formats and their features. Also supplies
 * the helpers that look up various data in those tables.
 */

struct etna_format {
   unsigned vtx;
   unsigned tex;
   unsigned rs;
   unsigned ts;
   boolean present;
   const unsigned char tex_swiz[4];
};

#define RS_FORMAT_NONE ~0

#define RS_FORMAT_MASK        0xf
#define RS_FORMAT(x)          ((x) & RS_FORMAT_MASK)
#define RS_FORMAT_RB_SWAP     0x10

#define RS_FORMAT_X8B8G8R8    (RS_FORMAT_X8R8G8B8 | RS_FORMAT_RB_SWAP)
#define RS_FORMAT_A8B8G8R8    (RS_FORMAT_A8R8G8B8 | RS_FORMAT_RB_SWAP)

#define TS_SAMPLER_FORMAT_NONE      ETNA_NO_MATCH

#define SWIZ(x,y,z,w) {    \
   PIPE_SWIZZLE_##x,       \
   PIPE_SWIZZLE_##y,       \
   PIPE_SWIZZLE_##z,       \
   PIPE_SWIZZLE_##w        \
}

/* vertex + texture */
#define VT(pipe, vtxfmt, texfmt, texswiz, rsfmt, tsfmt)          \
   [PIPE_FORMAT_##pipe] = {                               \
      .vtx = FE_DATA_TYPE_##vtxfmt, \
      .tex = TEXTURE_FORMAT_##texfmt,                     \
      .rs = RS_FORMAT_##rsfmt,                            \
      .ts = TS_SAMPLER_FORMAT_##tsfmt,        \
      .present = 1,                                       \
      .tex_swiz = texswiz,                                \
   }

/* texture-only */
#define _T(pipe, fmt, swiz, rsfmt, tsfmt) \
   [PIPE_FORMAT_##pipe] = {        \
      .vtx = ETNA_NO_MATCH,        \
      .tex = TEXTURE_FORMAT_##fmt, \
      .rs = RS_FORMAT_##rsfmt,     \
      .ts = TS_SAMPLER_FORMAT_##tsfmt,     \
      .present = 1,                \
      .tex_swiz = swiz,            \
   }

/* vertex-only */
#define V_(pipe, fmt, rsfmt)                           \
   [PIPE_FORMAT_##pipe] = {                            \
      .vtx = FE_DATA_TYPE_##fmt, \
      .tex = ETNA_NO_MATCH,                            \
      .rs = RS_FORMAT_##rsfmt,                         \
      .present = 1,                                    \
   }

static struct etna_format formats[PIPE_FORMAT_COUNT] = {
   /* 8-bit */
   VT(R8_UNORM,   UNSIGNED_BYTE, L8, SWIZ(X, 0, 0, 1), NONE, NONE),
   V_(R8_SNORM,   BYTE,          NONE),
   V_(R8_UINT,    UNSIGNED_BYTE, NONE),
   V_(R8_SINT,    BYTE,          NONE),
   V_(R8_USCALED, UNSIGNED_BYTE, NONE),
   V_(R8_SSCALED, BYTE,          NONE),

   _T(A8_UNORM, A8, SWIZ(X, Y, Z, W), NONE, NONE),
   _T(L8_UNORM, L8, SWIZ(X, Y, Z, W), NONE, NONE),
   _T(I8_UNORM, I8, SWIZ(X, Y, Z, W), NONE, NONE),

   /* 16-bit */
   V_(R16_UNORM,   UNSIGNED_SHORT, NONE),
   V_(R16_SNORM,   SHORT,          NONE),
   V_(R16_UINT,    UNSIGNED_SHORT, NONE),
   V_(R16_SINT,    SHORT,          NONE),
   V_(R16_USCALED, UNSIGNED_SHORT, NONE),
   V_(R16_SSCALED, SHORT,          NONE),
   V_(R16_FLOAT,   HALF_FLOAT,     NONE),

   _T(B4G4R4A4_UNORM, A4R4G4B4, SWIZ(X, Y, Z, W), A4R4G4B4, A4R4G4B4),
   _T(B4G4R4X4_UNORM, X4R4G4B4, SWIZ(X, Y, Z, W), X4R4G4B4, A4R4G4B4),

   _T(L8A8_UNORM, A8L8, SWIZ(X, Y, Z, W), NONE, NONE),

   _T(Z16_UNORM,      D16,      SWIZ(X, Y, Z, W), A4R4G4B4, D16),
   _T(B5G6R5_UNORM,   R5G6B5,   SWIZ(X, Y, Z, W), R5G6B5, R5G6B5),
   _T(B5G5R5A1_UNORM, A1R5G5B5, SWIZ(X, Y, Z, W), A1R5G5B5, A1R5G5B5),
   _T(B5G5R5X1_UNORM, X1R5G5B5, SWIZ(X, Y, Z, W), X1R5G5B5, A1R5G5B5),

   VT(R8G8_UNORM,   UNSIGNED_BYTE,  EXT_G8R8 | EXT_FORMAT, SWIZ(X, Y, 0, 1), NONE, NONE),
   V_(R8G8_SNORM,   BYTE,           NONE),
   V_(R8G8_UINT,    UNSIGNED_BYTE,  NONE),
   V_(R8G8_SINT,    BYTE,           NONE),
   V_(R8G8_USCALED, UNSIGNED_BYTE,  NONE),
   V_(R8G8_SSCALED, BYTE,           NONE),

   /* 24-bit */
   V_(R8G8B8_UNORM,   UNSIGNED_BYTE, NONE),
   V_(R8G8B8_SNORM,   BYTE,          NONE),
   V_(R8G8B8_UINT,    UNSIGNED_BYTE, NONE),
   V_(R8G8B8_SINT,    BYTE,          NONE),
   V_(R8G8B8_USCALED, UNSIGNED_BYTE, NONE),
   V_(R8G8B8_SSCALED, BYTE,          NONE),

   /* 32-bit */
   V_(R32_UNORM,   UNSIGNED_INT, NONE),
   V_(R32_SNORM,   INT,          NONE),
   V_(R32_SINT,    INT,          NONE),
   V_(R32_UINT,    UNSIGNED_INT, NONE),
   V_(R32_USCALED, UNSIGNED_INT, NONE),
   V_(R32_SSCALED, INT,          NONE),
   V_(R32_FLOAT,   FLOAT,        NONE),
   V_(R32_FIXED,   FIXED,        NONE),

   V_(R16G16_UNORM,   UNSIGNED_SHORT, NONE),
   V_(R16G16_SNORM,   SHORT,          NONE),
   V_(R16G16_UINT,    UNSIGNED_SHORT, NONE),
   V_(R16G16_SINT,    SHORT,          NONE),
   V_(R16G16_USCALED, UNSIGNED_SHORT, NONE),
   V_(R16G16_SSCALED, SHORT,          NONE),
   V_(R16G16_FLOAT,   HALF_FLOAT,     NONE),

   V_(A8B8G8R8_UNORM,   UNSIGNED_BYTE, NONE),

   VT(R8G8B8A8_UNORM,   UNSIGNED_BYTE, A8B8G8R8, SWIZ(X, Y, Z, W), A8B8G8R8, A8R8G8B8),
   V_(R8G8B8A8_SNORM,   BYTE,          A8B8G8R8),
   _T(R8G8B8X8_UNORM,   X8B8G8R8,      SWIZ(X, Y, Z, W), X8B8G8R8, X8R8G8B8),
   V_(R8G8B8A8_UINT,    UNSIGNED_BYTE, A8B8G8R8),
   V_(R8G8B8A8_SINT,    BYTE,          A8B8G8R8),
   V_(R8G8B8A8_USCALED, UNSIGNED_BYTE, A8B8G8R8),
   V_(R8G8B8A8_SSCALED, BYTE,          A8B8G8R8),

   _T(B8G8R8A8_UNORM, A8R8G8B8, SWIZ(X, Y, Z, W), A8R8G8B8, A8R8G8B8),
   _T(B8G8R8X8_UNORM, X8R8G8B8, SWIZ(X, Y, Z, W), X8R8G8B8, X8R8G8B8),
   _T(B8G8R8A8_SRGB,  A8R8G8B8, SWIZ(X, Y, Z, W), A8R8G8B8, A8R8G8B8),
   _T(B8G8R8X8_SRGB,  X8R8G8B8, SWIZ(X, Y, Z, W), X8R8G8B8, X8R8G8B8),

   V_(R10G10B10A2_UNORM,   UNSIGNED_INT_10_10_10_2, NONE),
   V_(R10G10B10A2_SNORM,   INT_10_10_10_2,          NONE),
   V_(R10G10B10A2_USCALED, UNSIGNED_INT_10_10_10_2, NONE),
   V_(R10G10B10A2_SSCALED, INT_10_10_10_2,          NONE),

   _T(X8Z24_UNORM,       D24X8, SWIZ(X, Y, Z, W), A8R8G8B8, D24X8),
   _T(S8_UINT_Z24_UNORM, D24X8, SWIZ(X, Y, Z, W), A8R8G8B8, D24X8),

   /* 48-bit */
   V_(R16G16B16_UNORM,   UNSIGNED_SHORT, NONE),
   V_(R16G16B16_SNORM,   SHORT,          NONE),
   V_(R16G16B16_UINT,    UNSIGNED_SHORT, NONE),
   V_(R16G16B16_SINT,    SHORT,          NONE),
   V_(R16G16B16_USCALED, UNSIGNED_SHORT, NONE),
   V_(R16G16B16_SSCALED, SHORT,          NONE),
   V_(R16G16B16_FLOAT,   HALF_FLOAT,     NONE),

   /* 64-bit */
   V_(R16G16B16A16_UNORM,   UNSIGNED_SHORT, NONE),
   V_(R16G16B16A16_SNORM,   SHORT,          NONE),
   V_(R16G16B16A16_UINT,    UNSIGNED_SHORT, NONE),
   V_(R16G16B16A16_SINT,    SHORT,          NONE),
   V_(R16G16B16A16_USCALED, UNSIGNED_SHORT, NONE),
   V_(R16G16B16A16_SSCALED, SHORT,          NONE),
   V_(R16G16B16A16_FLOAT,   HALF_FLOAT,     NONE),

   V_(R32G32_UNORM,   UNSIGNED_INT, NONE),
   V_(R32G32_SNORM,   INT,          NONE),
   V_(R32G32_UINT,    UNSIGNED_INT, NONE),
   V_(R32G32_SINT,    INT,          NONE),
   V_(R32G32_USCALED, UNSIGNED_INT, NONE),
   V_(R32G32_SSCALED, INT,          NONE),
   V_(R32G32_FLOAT,   FLOAT,        NONE),
   V_(R32G32_FIXED,   FIXED,        NONE),

   /* 96-bit */
   V_(R32G32B32_UNORM,   UNSIGNED_INT, NONE),
   V_(R32G32B32_SNORM,   INT,          NONE),
   V_(R32G32B32_UINT,    UNSIGNED_INT, NONE),
   V_(R32G32B32_SINT,    INT,          NONE),
   V_(R32G32B32_USCALED, UNSIGNED_INT, NONE),
   V_(R32G32B32_SSCALED, INT,          NONE),
   V_(R32G32B32_FLOAT,   FLOAT,        NONE),
   V_(R32G32B32_FIXED,   FIXED,        NONE),

   /* 128-bit */
   V_(R32G32B32A32_UNORM,   UNSIGNED_INT, NONE),
   V_(R32G32B32A32_SNORM,   INT,          NONE),
   V_(R32G32B32A32_UINT,    UNSIGNED_INT, NONE),
   V_(R32G32B32A32_SINT,    INT,          NONE),
   V_(R32G32B32A32_USCALED, UNSIGNED_INT, NONE),
   V_(R32G32B32A32_SSCALED, INT,          NONE),
   V_(R32G32B32A32_FLOAT,   FLOAT,        NONE),
   V_(R32G32B32A32_FIXED,   FIXED,        NONE),

   /* compressed */
   _T(ETC1_RGB8, ETC1, SWIZ(X, Y, Z, W), NONE, NONE),

   _T(DXT1_RGB,  DXT1,      SWIZ(X, Y, Z, W), NONE, NONE),
   _T(DXT1_SRGBA,DXT1,      SWIZ(X, Y, Z, W), NONE, NONE),
   _T(DXT1_RGBA, DXT1,      SWIZ(X, Y, Z, W), NONE, NONE),
   _T(DXT3_SRGBA,DXT2_DXT3, SWIZ(X, Y, Z, W), NONE, NONE),
   _T(DXT3_RGBA, DXT2_DXT3, SWIZ(X, Y, Z, W), NONE, NONE),
   _T(DXT5_SRGBA,DXT4_DXT5, SWIZ(X, Y, Z, W), NONE, NONE),
   _T(DXT5_RGBA, DXT4_DXT5, SWIZ(X, Y, Z, W), NONE, NONE),

   _T(ETC2_RGB8,       EXT_NONE | EXT_FORMAT,                          SWIZ(X, Y, Z, W), NONE, NONE), /* Extd. format NONE doubles as ETC2_RGB8 */
   _T(ETC2_SRGB8,      EXT_NONE | EXT_FORMAT,                          SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ETC2_RGB8A1,     EXT_RGB8_PUNCHTHROUGH_ALPHA1_ETC2 | EXT_FORMAT, SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ETC2_SRGB8A1,    EXT_RGB8_PUNCHTHROUGH_ALPHA1_ETC2 | EXT_FORMAT, SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ETC2_RGBA8,      EXT_RGBA8_ETC2_EAC | EXT_FORMAT,                SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ETC2_SRGBA8,     EXT_RGBA8_ETC2_EAC | EXT_FORMAT,                SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ETC2_R11_UNORM,  EXT_R11_EAC | EXT_FORMAT,                       SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ETC2_R11_SNORM,  EXT_SIGNED_R11_EAC | EXT_FORMAT,                SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ETC2_RG11_UNORM, EXT_RG11_EAC | EXT_FORMAT,                      SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ETC2_RG11_SNORM, EXT_SIGNED_RG11_EAC | EXT_FORMAT,               SWIZ(X, Y, Z, W), NONE, NONE),

   _T(ASTC_4x4,        ASTC_RGBA_4x4 | ASTC_FORMAT,                    SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_5x4,        ASTC_RGBA_5x4 | ASTC_FORMAT,                    SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_5x5,        ASTC_RGBA_5x5 | ASTC_FORMAT,                    SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_6x5,        ASTC_RGBA_6x5 | ASTC_FORMAT,                    SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_6x6,        ASTC_RGBA_6x6 | ASTC_FORMAT,                    SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_8x5,        ASTC_RGBA_8x5 | ASTC_FORMAT,                    SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_8x6,        ASTC_RGBA_8x6 | ASTC_FORMAT,                    SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_8x8,        ASTC_RGBA_8x8 | ASTC_FORMAT,                    SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_10x5,       ASTC_RGBA_10x5 | ASTC_FORMAT,                   SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_10x6,       ASTC_RGBA_10x6 | ASTC_FORMAT,                   SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_10x8,       ASTC_RGBA_10x8 | ASTC_FORMAT,                   SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_10x10,      ASTC_RGBA_10x10 | ASTC_FORMAT,                  SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_12x10,      ASTC_RGBA_12x10 | ASTC_FORMAT,                  SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_12x12,      ASTC_RGBA_12x12 | ASTC_FORMAT,                  SWIZ(X, Y, Z, W), NONE, NONE),

   _T(ASTC_4x4_SRGB,   ASTC_SRGB8_ALPHA8_4x4 | ASTC_FORMAT,            SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_5x4_SRGB,   ASTC_SRGB8_ALPHA8_5x4 | ASTC_FORMAT,            SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_5x5_SRGB,   ASTC_SRGB8_ALPHA8_5x5 | ASTC_FORMAT,            SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_6x5_SRGB,   ASTC_SRGB8_ALPHA8_6x5 | ASTC_FORMAT,            SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_6x6_SRGB,   ASTC_SRGB8_ALPHA8_6x6 | ASTC_FORMAT,            SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_8x5_SRGB,   ASTC_SRGB8_ALPHA8_8x5 | ASTC_FORMAT,            SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_8x6_SRGB,   ASTC_SRGB8_ALPHA8_8x6 | ASTC_FORMAT,            SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_8x8_SRGB,   ASTC_SRGB8_ALPHA8_8x8 | ASTC_FORMAT,            SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_10x5_SRGB,  ASTC_SRGB8_ALPHA8_10x5 | ASTC_FORMAT,           SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_10x6_SRGB,  ASTC_SRGB8_ALPHA8_10x6 | ASTC_FORMAT,           SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_10x8_SRGB,  ASTC_SRGB8_ALPHA8_10x8 | ASTC_FORMAT,           SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_10x10_SRGB, ASTC_SRGB8_ALPHA8_10x10 | ASTC_FORMAT,          SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_12x10_SRGB, ASTC_SRGB8_ALPHA8_12x10 | ASTC_FORMAT,          SWIZ(X, Y, Z, W), NONE, NONE),
   _T(ASTC_12x12_SRGB, ASTC_SRGB8_ALPHA8_12x12 | ASTC_FORMAT,          SWIZ(X, Y, Z, W), NONE, NONE),

   /* YUV */
   _T(YUYV, YUY2, SWIZ(X, Y, Z, W), YUY2, NONE),
   _T(UYVY, UYVY, SWIZ(X, Y, Z, W), NONE, NONE),
};

uint32_t
translate_texture_format(enum pipe_format fmt)
{
   if (!formats[fmt].present)
      return ETNA_NO_MATCH;

   return formats[fmt].tex;
}

bool
texture_format_needs_swiz(enum pipe_format fmt)
{
   static const unsigned char def[4] = SWIZ(X, Y, Z, W);
   bool swiz = false;

   if (formats[fmt].present)
      swiz = !!memcmp(def, formats[fmt].tex_swiz, sizeof(formats[fmt].tex_swiz));

   return swiz;
}

uint32_t
get_texture_swiz(enum pipe_format fmt, unsigned swizzle_r,
                 unsigned swizzle_g, unsigned swizzle_b, unsigned swizzle_a)
{
   unsigned char swiz[4] = {
      swizzle_r, swizzle_g, swizzle_b, swizzle_a,
   }, rswiz[4];

   assert(formats[fmt].present);
   util_format_compose_swizzles(formats[fmt].tex_swiz, swiz, rswiz);

   /* PIPE_SWIZZLE_ maps 1:1 to TEXTURE_SWIZZLE_ */
   STATIC_ASSERT(PIPE_SWIZZLE_X == TEXTURE_SWIZZLE_RED);
   STATIC_ASSERT(PIPE_SWIZZLE_Y == TEXTURE_SWIZZLE_GREEN);
   STATIC_ASSERT(PIPE_SWIZZLE_Z == TEXTURE_SWIZZLE_BLUE);
   STATIC_ASSERT(PIPE_SWIZZLE_W == TEXTURE_SWIZZLE_ALPHA);
   STATIC_ASSERT(PIPE_SWIZZLE_0 == TEXTURE_SWIZZLE_ZERO);
   STATIC_ASSERT(PIPE_SWIZZLE_1 == TEXTURE_SWIZZLE_ONE);

   return VIVS_TE_SAMPLER_CONFIG1_SWIZZLE_R(rswiz[0]) |
          VIVS_TE_SAMPLER_CONFIG1_SWIZZLE_G(rswiz[1]) |
          VIVS_TE_SAMPLER_CONFIG1_SWIZZLE_B(rswiz[2]) |
          VIVS_TE_SAMPLER_CONFIG1_SWIZZLE_A(rswiz[3]);
}

uint32_t
translate_rs_format(enum pipe_format fmt)
{
   if (!formats[fmt].present)
      return ETNA_NO_MATCH;

   if (formats[fmt].rs == ETNA_NO_MATCH)
      return ETNA_NO_MATCH;

   return RS_FORMAT(formats[fmt].rs);
}

int
translate_rs_format_rb_swap(enum pipe_format fmt)
{
   assert(formats[fmt].present);

   return formats[fmt].rs & RS_FORMAT_RB_SWAP;
}

/* Return type flags for vertex element format */
uint32_t
translate_vertex_format_type(enum pipe_format fmt)
{
   if (!formats[fmt].present)
      return ETNA_NO_MATCH;

   return formats[fmt].vtx;
}

uint32_t
translate_ts_sampler_format(enum pipe_format fmt)
{
   if (!formats[fmt].present)
      return ETNA_NO_MATCH;

   return formats[fmt].ts;
}
