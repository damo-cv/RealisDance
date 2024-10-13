/*
 * Copyright 2015 Intel Corporation
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without limitation
 *  the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *  and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice (including the next
 *  paragraph) shall be included in all copies or substantial portions of the
 *  Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 *  IN THE SOFTWARE.
 */

#include <assert.h>

#include "isl.h"
#include "isl_priv.h"
#include "dev/gen_device_info.h"

#include "main/macros.h" /* Needed for MAX3 and MAX2 for format_rgb9e5 */
#include "util/format_srgb.h"
#include "util/format_rgb9e5.h"
#include "util/format_r11g11b10f.h"

/* Header-only format conversion include */
#include "main/format_utils.h"

struct surface_format_info {
   bool exists;
   uint8_t sampling;
   uint8_t filtering;
   uint8_t shadow_compare;
   uint8_t chroma_key;
   uint8_t render_target;
   uint8_t alpha_blend;
   uint8_t input_vb;
   uint8_t streamed_output_vb;
   uint8_t color_processing;
   uint8_t typed_write;
   uint8_t typed_read;
   uint8_t ccs_e;
};

/* This macro allows us to write the table almost as it appears in the PRM,
 * while restructuring it to turn it into the C code we want.
 */
#define SF(sampl, filt, shad, ck, rt, ab, vb, so, color, tw, tr, ccs_e, sf) \
   [ISL_FORMAT_##sf] = { true, sampl, filt, shad, ck, rt, ab, vb, so, color, tw, tr, ccs_e},

#define Y 0
#define x 255
/**
 * This is the table of support for surface (texture, renderbuffer, and vertex
 * buffer, but not depthbuffer) formats across the various hardware generations.
 *
 * The table is formatted to match the documentation, except that the docs have
 * this ridiculous mapping of Y[*+~^#&] for "supported on DevWhatever".  To put
 * it in our table, here's the mapping:
 *
 * Y*: 45
 * Y+: 45 (g45/gm45)
 * Y~: 50 (gen5)
 * Y^: 60 (gen6)
 * Y#: 70 (gen7)
 *
 * The abbreviations in the header below are:
 * smpl  - Sampling Engine
 * filt  - Sampling Engine Filtering
 * shad  - Sampling Engine Shadow Map
 * CK    - Sampling Engine Chroma Key
 * RT    - Render Target
 * AB    - Alpha Blend Render Target
 * VB    - Input Vertex Buffer
 * SO    - Steamed Output Vertex Buffers (transform feedback)
 * color - Color Processing
 * ccs_e - Lossless Compression Support (gen9+ only)
 * sf    - Surface Format
 *
 * See page 88 of the Sandybridge PRM VOL4_Part1 PDF.
 *
 * As of Ivybridge, the columns are no longer in that table and the
 * information can be found spread across:
 *
 * - VOL2_Part1 section 2.5.11 Format Conversion (vertex fetch).
 * - VOL4_Part1 section 2.12.2.1.2 Sampler Output Channel Mapping.
 * - VOL4_Part1 section 3.9.11 Render Target Write.
 * - Render Target Surface Types [SKL+]
 */
static const struct surface_format_info format_info[] = {
/*    smpl filt  shad  CK   RT   AB   VB   SO color TW   TR  ccs_e */
   SF(  Y,  50,   x,   x,   Y,   Y,   Y,   Y,   x,  70,  90,  90,   R32G32B32A32_FLOAT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   Y,   x,  70,  90,  90,   R32G32B32A32_SINT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   Y,   x,  70,  90,  90,   R32G32B32A32_UINT)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32B32A32_UNORM)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32B32A32_SNORM)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R64G64_FLOAT)
   SF(  Y,  50,   x,   x, 100, 100,   x,   x,   x,   x,   x, 100,   R32G32B32X32_FLOAT)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32B32A32_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32B32A32_USCALED)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R32G32B32A32_SFIXED)
   SF(  x,   x,   x,   x,   x,   x,  80,   x,   x,   x,   x,   x,   R64G64_PASSTHRU)
   SF(  Y,  50,   x,   x,   x,   x,   Y,   Y,   x,   x,   x,   x,   R32G32B32_FLOAT)
   SF(  Y,   x,   x,   x,   x,   x,   Y,   Y,   x,   x,   x,   x,   R32G32B32_SINT)
   SF(  Y,   x,   x,   x,   x,   x,   Y,   Y,   x,   x,   x,   x,   R32G32B32_UINT)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32B32_UNORM)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32B32_SNORM)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32B32_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32B32_USCALED)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R32G32B32_SFIXED)
   SF(  Y,   Y,   x,   x,   Y,  45,   Y,   x,  60,  70, 110,  90,   R16G16B16A16_UNORM)
   SF(  Y,   Y,   x,   x,   Y,  60,   Y,   x,   x,  70, 110,  90,   R16G16B16A16_SNORM)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  90,  90,   R16G16B16A16_SINT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  75,  90,   R16G16B16A16_UINT)
   SF(  Y,   Y,   x,   x,   Y,   Y,   Y,   x,   x,  70,  90,  90,   R16G16B16A16_FLOAT)
   SF(  Y,  50,   x,   x,   Y,   Y,   Y,   Y,   x,  70,  90,  90,   R32G32_FLOAT)
   SF(  Y,  70,   x,   x,   Y,   Y,   Y,   Y,   x,   x,   x,   x,   R32G32_FLOAT_LD)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   Y,   x,  70,  90,  90,   R32G32_SINT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   Y,   x,  70,  90,  90,   R32G32_UINT)
   SF(  Y,  50,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   R32_FLOAT_X8X24_TYPELESS)
   SF(  Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   X32_TYPELESS_G8X24_UINT)
   SF(  Y,  50,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L32A32_FLOAT)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32_UNORM)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32_SNORM)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R64_FLOAT)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   R16G16B16X16_UNORM)
   SF(  Y,   Y,   x,   x,  90,  90,   x,   x,   x,   x,   x,  90,   R16G16B16X16_FLOAT)
   SF(  Y,  50,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   A32X32_FLOAT)
   SF(  Y,  50,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L32X32_FLOAT)
   SF(  Y,  50,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   I32X32_FLOAT)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R16G16B16A16_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R16G16B16A16_USCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32G32_USCALED)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R32G32_SFIXED)
   SF(  x,   x,   x,   x,   x,   x,  80,   x,   x,   x,   x,   x,   R64_PASSTHRU)
   SF(  Y,   Y,   x,   Y,   Y,   Y,   Y,   x,  60,  70,   x,  90,   B8G8R8A8_UNORM)
   SF(  Y,   Y,   x,   x,   Y,   Y,   x,   x,   x,   x,   x, 100,   B8G8R8A8_UNORM_SRGB)
/*    smpl filt  shad  CK   RT   AB   VB   SO color TW   TR  ccs_e */
   SF(  Y,   Y,   x,   x,   Y,   Y,   Y,   x,  60,  70,   x, 100,   R10G10B10A2_UNORM)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,  60,   x,   x,   x,   R10G10B10A2_UNORM_SRGB)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,   x, 100,   R10G10B10A2_UINT)
   SF(  Y,   Y,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R10G10B10_SNORM_A2_UNORM)
   SF(  Y,   Y,   x,   x,   Y,   Y,   Y,   x,  60,  70, 110,  90,   R8G8B8A8_UNORM)
   SF(  Y,   Y,   x,   x,   Y,   Y,   x,   x,  60,   x,   x, 100,   R8G8B8A8_UNORM_SRGB)
   SF(  Y,   Y,   x,   x,   Y,  60,   Y,   x,   x,  70, 110,  90,   R8G8B8A8_SNORM)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  90,  90,   R8G8B8A8_SINT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  75,  90,   R8G8B8A8_UINT)
   SF(  Y,   Y,   x,   x,   Y,  45,   Y,   x,   x,  70, 110,  90,   R16G16_UNORM)
   SF(  Y,   Y,   x,   x,   Y,  60,   Y,   x,   x,  70, 110,  90,   R16G16_SNORM)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  90,  90,   R16G16_SINT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  75,  90,   R16G16_UINT)
   SF(  Y,   Y,   x,   x,   Y,   Y,   Y,   x,   x,  70,  90,  90,   R16G16_FLOAT)
   SF(  Y,   Y,   x,   x,   Y,   Y,  75,   x,  60,  70,   x, 100,   B10G10R10A2_UNORM)
   SF(  Y,   Y,   x,   x,   Y,   Y,   x,   x,  60,   x,   x, 100,   B10G10R10A2_UNORM_SRGB)
   SF(  Y,   Y,   x,   x,   Y,   Y,   Y,   x,   x,  70,   x, 100,   R11G11B10_FLOAT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   Y,   x,  70,  70,  90,   R32_SINT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   Y,   x,  70,  70,  90,   R32_UINT)
   SF(  Y,  50,   Y,   x,   Y,   Y,   Y,   Y,   x,  70,  70,  90,   R32_FLOAT)
   SF(  Y,  50,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   R24_UNORM_X8_TYPELESS)
   SF(  Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   X24_TYPELESS_G8_UINT)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L16A16_UNORM)
   SF(  Y,  50,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   I24X8_UNORM)
   SF(  Y,  50,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L24X8_UNORM)
   SF(  Y,  50,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   A24X8_UNORM)
   SF(  Y,  50,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   I32_FLOAT)
   SF(  Y,  50,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L32_FLOAT)
   SF(  Y,  50,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   A32_FLOAT)
   SF(  Y,   Y,   x,   Y,  80,  80,   x,   x,  60,   x,   x,  90,   B8G8R8X8_UNORM)
   SF(  Y,   Y,   x,   x,  80,  80,   x,   x,   x,   x,   x, 100,   B8G8R8X8_UNORM_SRGB)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   R8G8B8X8_UNORM)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   R8G8B8X8_UNORM_SRGB)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   R9G9B9E5_SHAREDEXP)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   B10G10R10X2_UNORM)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L16A16_FLOAT)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32_UNORM)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32_SNORM)
/*    smpl filt  shad  CK   RT   AB   VB   SO color TW   TR  ccs_e */
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R10G10B10X2_USCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R8G8B8A8_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R8G8B8A8_USCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R16G16_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R16G16_USCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R32_USCALED)
   SF(  Y,   Y,   x,   Y,   Y,   Y,   x,   x,   x,  70,   x,   x,   B5G6R5_UNORM)
   SF(  Y,   Y,   x,   x,   Y,   Y,   x,   x,   x,   x,   x,   x,   B5G6R5_UNORM_SRGB)
   SF(  Y,   Y,   x,   Y,   Y,   Y,   x,   x,   x,  70,   x,   x,   B5G5R5A1_UNORM)
   SF(  Y,   Y,   x,   x,   Y,   Y,   x,   x,   x,   x,   x,   x,   B5G5R5A1_UNORM_SRGB)
   SF(  Y,   Y,   x,   Y,   Y,   Y,   x,   x,   x,  70,   x,   x,   B4G4R4A4_UNORM)
   SF(  Y,   Y,   x,   x,   Y,   Y,   x,   x,   x,   x,   x,   x,   B4G4R4A4_UNORM_SRGB)
   SF(  Y,   Y,   x,   x,   Y,   Y,   Y,   x,   x,  70, 110,   x,   R8G8_UNORM)
   SF(  Y,   Y,   x,   Y,   Y,  60,   Y,   x,   x,  70, 110,   x,   R8G8_SNORM)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  90,   x,   R8G8_SINT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  75,   x,   R8G8_UINT)
   SF(  Y,   Y,   Y,   x,   Y,  45,   Y,   x,  70,  70, 110,   x,   R16_UNORM)
   SF(  Y,   Y,   x,   x,   Y,  60,   Y,   x,   x,  70, 110,   x,   R16_SNORM)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  90,   x,   R16_SINT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  75,   x,   R16_UINT)
   SF(  Y,   Y,   x,   x,   Y,   Y,   Y,   x,   x,  70,  90,   x,   R16_FLOAT)
   SF( 50,  50,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   A8P8_UNORM_PALETTE0)
   SF( 50,  50,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   A8P8_UNORM_PALETTE1)
   SF(  Y,   Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   I16_UNORM)
   SF(  Y,   Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L16_UNORM)
   SF(  Y,   Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   A16_UNORM)
   SF(  Y,   Y,   x,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   L8A8_UNORM)
   SF(  Y,   Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   I16_FLOAT)
   SF(  Y,   Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L16_FLOAT)
   SF(  Y,   Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   A16_FLOAT)
   SF( 45,  45,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L8A8_UNORM_SRGB)
   SF(  Y,   Y,   x,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   R5G5_SNORM_B6_UNORM)
   SF(  x,   x,   x,   x,   Y,   Y,   x,   x,   x,  70,   x,   x,   B5G5R5X1_UNORM)
   SF(  x,   x,   x,   x,   Y,   Y,   x,   x,   x,   x,   x,   x,   B5G5R5X1_UNORM_SRGB)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R8G8_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R8G8_USCALED)
/*    smpl filt  shad  CK   RT   AB   VB   SO color TW   TR  ccs_e */
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R16_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R16_USCALED)
   SF( 50,  50,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   P8A8_UNORM_PALETTE0)
   SF( 50,  50,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   P8A8_UNORM_PALETTE1)
   SF(  x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   A1B5G5R5_UNORM)
   /* According to the PRM, A4B4G4R4_UNORM isn't supported until Sky Lake
    * but empirical testing indicates that at least sampling works just fine
    * on Broadwell.
    */
   SF( 80,  80,   x,   x,  90,   x,   x,   x,   x,   x,   x,   x,   A4B4G4R4_UNORM)
   SF( 90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L8A8_UINT)
   SF( 90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L8A8_SINT)
   SF(  Y,   Y,   x,  45,   Y,   Y,   Y,   x,   x,  70, 110,   x,   R8_UNORM)
   SF(  Y,   Y,   x,   x,   Y,  60,   Y,   x,   x,  70, 110,   x,   R8_SNORM)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  90,   x,   R8_SINT)
   SF(  Y,   x,   x,   x,   Y,   x,   Y,   x,   x,  70,  75,   x,   R8_UINT)
   SF(  Y,   Y,   x,   Y,   Y,   Y,   x,   x,   x,  70, 110,   x,   A8_UNORM)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   I8_UNORM)
   SF(  Y,   Y,   x,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   L8_UNORM)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   P4A4_UNORM_PALETTE0)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   A4P4_UNORM_PALETTE0)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R8_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R8_USCALED)
   SF( 45,  45,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   P8_UNORM_PALETTE0)
   SF( 45,  45,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L8_UNORM_SRGB)
   SF( 45,  45,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   P8_UNORM_PALETTE1)
   SF( 45,  45,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   P4A4_UNORM_PALETTE1)
   SF( 45,  45,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   A4P4_UNORM_PALETTE1)
   SF(  x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   Y8_UNORM)
   SF( 90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L8_UINT)
   SF( 90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   L8_SINT)
   SF( 90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   I8_UINT)
   SF( 90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   I8_SINT)
   SF( 45,  45,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   DXT1_RGB_SRGB)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   R1_UNORM)
   SF(  Y,   Y,   x,   Y,   Y,   x,   x,   x,  60,   x,   x,   x,   YCRCB_NORMAL)
   SF(  Y,   Y,   x,   Y,   Y,   x,   x,   x,  60,   x,   x,   x,   YCRCB_SWAPUVY)
   SF( 45,  45,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   P2_UNORM_PALETTE0)
   SF( 45,  45,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   P2_UNORM_PALETTE1)
   SF(  Y,   Y,   x,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   BC1_UNORM)
   SF(  Y,   Y,   x,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   BC2_UNORM)
   SF(  Y,   Y,   x,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   BC3_UNORM)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC4_UNORM)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC5_UNORM)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC1_UNORM_SRGB)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC2_UNORM_SRGB)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC3_UNORM_SRGB)
   SF(  Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   MONO8)
   SF(  Y,   Y,   x,   x,   Y,   x,   x,   x,  60,   x,   x,   x,   YCRCB_SWAPUV)
   SF(  Y,   Y,   x,   x,   Y,   x,   x,   x,  60,   x,   x,   x,   YCRCB_SWAPY)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   DXT1_RGB)
/*    smpl filt  shad  CK   RT   AB   VB   SO color TW   TR  ccs_e */
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   FXT1)
   SF( 75,  75,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R8G8B8_UNORM)
   SF( 75,  75,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R8G8B8_SNORM)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R8G8B8_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R8G8B8_USCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R64G64B64A64_FLOAT)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R64G64B64_FLOAT)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC4_SNORM)
   SF(  Y,   Y,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC5_SNORM)
   SF( 50,  50,   x,   x,   x,   x,  60,   x,   x,   x,   x,   x,   R16G16B16_FLOAT)
   SF( 75,  75,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R16G16B16_UNORM)
   SF( 75,  75,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R16G16B16_SNORM)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R16G16B16_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,   Y,   x,   x,   x,   x,   x,   R16G16B16_USCALED)
   SF( 70,  70,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC6H_SF16)
   SF( 70,  70,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC7_UNORM)
   SF( 70,  70,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC7_UNORM_SRGB)
   SF( 70,  70,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   BC6H_UF16)
   SF(  x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   PLANAR_420_8)
   SF( 75,  75,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   R8G8B8_UNORM_SRGB)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ETC1_RGB8)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ETC2_RGB8)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   EAC_R11)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   EAC_RG11)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   EAC_SIGNED_R11)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   EAC_SIGNED_RG11)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ETC2_SRGB8)
   SF( 90,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R16G16B16_UINT)
   SF( 90,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R16G16B16_SINT)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R32_SFIXED)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R10G10B10A2_SNORM)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R10G10B10A2_USCALED)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R10G10B10A2_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R10G10B10A2_SINT)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   B10G10R10A2_SNORM)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   B10G10R10A2_USCALED)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   B10G10R10A2_SSCALED)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   B10G10R10A2_UINT)
   SF(  x,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   B10G10R10A2_SINT)
   SF(  x,   x,   x,   x,   x,   x,  80,   x,   x,   x,   x,   x,   R64G64B64A64_PASSTHRU)
   SF(  x,   x,   x,   x,   x,   x,  80,   x,   x,   x,   x,   x,   R64G64B64_PASSTHRU)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ETC2_RGB8_PTA)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ETC2_SRGB8_PTA)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ETC2_EAC_RGBA8)
   SF( 80,  80,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ETC2_EAC_SRGB8_A8)
   SF( 90,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R8G8B8_UINT)
   SF( 90,   x,   x,   x,   x,   x,  75,   x,   x,   x,   x,   x,   R8G8B8_SINT)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_4X4_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_5X4_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_5X5_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_6X5_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_6X6_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_8X5_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_8X6_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_8X8_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_10X5_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_10X6_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_10X8_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_10X10_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_12X10_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_12X12_FLT16)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_4X4_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_5X4_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_5X5_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_6X5_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_6X6_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_8X5_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_8X6_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_8X8_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_10X5_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_10X6_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_10X8_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_10X10_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_12X10_U8SRGB)
   SF( 90,  90,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_LDR_2D_12X12_U8SRGB)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_4X4_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_5X4_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_5X5_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_6X5_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_6X6_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_8X5_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_8X6_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_8X8_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_10X5_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_10X6_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_10X8_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_10X10_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_12X10_FLT16)
   SF(100, 100,   x,   x,   x,   x,   x,   x,   x,   x,   x,   x,   ASTC_HDR_2D_12X12_FLT16)
};
#undef x
#undef Y

static unsigned
format_gen(const struct gen_device_info *devinfo)
{
   return devinfo->gen * 10 + (devinfo->is_g4x || devinfo->is_haswell) * 5;
}

static bool
format_info_exists(enum isl_format format)
{
   assert(format != ISL_FORMAT_UNSUPPORTED);
   assert(format < ISL_NUM_FORMATS);
   return format < ARRAY_SIZE(format_info) && format_info[format].exists;
}

bool
isl_format_supports_rendering(const struct gen_device_info *devinfo,
                              enum isl_format format)
{
   if (!format_info_exists(format))
      return false;

   return format_gen(devinfo) >= format_info[format].render_target;
}

bool
isl_format_supports_alpha_blending(const struct gen_device_info *devinfo,
                                   enum isl_format format)
{
   if (!format_info_exists(format))
      return false;

   return format_gen(devinfo) >= format_info[format].alpha_blend;
}

bool
isl_format_supports_sampling(const struct gen_device_info *devinfo,
                             enum isl_format format)
{
   if (!format_info_exists(format))
      return false;

   if (devinfo->is_baytrail) {
      const struct isl_format_layout *fmtl = isl_format_get_layout(format);
      /* Support for ETC1 and ETC2 exists on Bay Trail even though big-core
       * GPUs didn't get it until Broadwell.
       */
      if (fmtl->txc == ISL_TXC_ETC1 || fmtl->txc == ISL_TXC_ETC2)
         return true;
   } else if (devinfo->is_cherryview) {
      const struct isl_format_layout *fmtl = isl_format_get_layout(format);
      /* Support for ASTC LDR exists on Cherry View even though big-core
       * GPUs didn't get it until Skylake.
       */
      if (fmtl->txc == ISL_TXC_ASTC)
         return format < ISL_FORMAT_ASTC_HDR_2D_4X4_FLT16;
   } else if (gen_device_info_is_9lp(devinfo)) {
      const struct isl_format_layout *fmtl = isl_format_get_layout(format);
      /* Support for ASTC HDR exists on Broxton even though big-core
       * GPUs didn't get it until Cannonlake.
       */
      if (fmtl->txc == ISL_TXC_ASTC)
         return true;
   }

   return format_gen(devinfo) >= format_info[format].sampling;
}

bool
isl_format_supports_filtering(const struct gen_device_info *devinfo,
                              enum isl_format format)
{
   if (!format_info_exists(format))
      return false;

   if (devinfo->is_baytrail) {
      const struct isl_format_layout *fmtl = isl_format_get_layout(format);
      /* Support for ETC1 and ETC2 exists on Bay Trail even though big-core
       * GPUs didn't get it until Broadwell.
       */
      if (fmtl->txc == ISL_TXC_ETC1 || fmtl->txc == ISL_TXC_ETC2)
         return true;
   } else if (devinfo->is_cherryview) {
      const struct isl_format_layout *fmtl = isl_format_get_layout(format);
      /* Support for ASTC LDR exists on Cherry View even though big-core
       * GPUs didn't get it until Skylake.
       */
      if (fmtl->txc == ISL_TXC_ASTC)
         return format < ISL_FORMAT_ASTC_HDR_2D_4X4_FLT16;
   } else if (gen_device_info_is_9lp(devinfo)) {
      const struct isl_format_layout *fmtl = isl_format_get_layout(format);
      /* Support for ASTC HDR exists on Broxton even though big-core
       * GPUs didn't get it until Cannonlake.
       */
      if (fmtl->txc == ISL_TXC_ASTC)
         return true;
   }

   return format_gen(devinfo) >= format_info[format].filtering;
}

bool
isl_format_supports_vertex_fetch(const struct gen_device_info *devinfo,
                                 enum isl_format format)
{
   if (!format_info_exists(format))
      return false;

   /* For vertex fetch, Bay Trail supports the same set of formats as Haswell
    * but is a superset of Ivy Bridge.
    */
   if (devinfo->is_baytrail)
      return 75 >= format_info[format].input_vb;

   return format_gen(devinfo) >= format_info[format].input_vb;
}

/**
 * Returns true if the given format can support typed writes.
 */
bool
isl_format_supports_typed_writes(const struct gen_device_info *devinfo,
                                 enum isl_format format)
{
   if (!format_info_exists(format))
      return false;

   return format_gen(devinfo) >= format_info[format].typed_write;
}


/**
 * Returns true if the given format can support typed reads with format
 * conversion fully handled by hardware.  On Sky Lake, all formats which are
 * supported for typed writes also support typed reads but some of them return
 * the raw image data and don't provide format conversion.
 *
 * For anyone looking to find this data in the PRM, the easiest way to find
 * format tables is to search for R11G11B10.  There are only a few
 * occurrences.
 */
bool
isl_format_supports_typed_reads(const struct gen_device_info *devinfo,
                                enum isl_format format)
{
   if (!format_info_exists(format))
      return false;

   return format_gen(devinfo) >= format_info[format].typed_read;
}

/**
 * Returns true if the given format can support single-sample fast clears.
 * This function only checks the format.  In order to determine if a surface
 * supports CCS_E, several other factors need to be considered such as tiling
 * and sample count.  See isl_surf_get_ccs_surf for details.
 */
bool
isl_format_supports_ccs_d(const struct gen_device_info *devinfo,
                          enum isl_format format)
{
   /* Fast clears were first added on Ivy Bridge */
   if (devinfo->gen < 7)
      return false;

   if (!isl_format_supports_rendering(devinfo, format))
      return false;

   const struct isl_format_layout *fmtl = isl_format_get_layout(format);

   return fmtl->bpb == 32 || fmtl->bpb == 64 || fmtl->bpb == 128;
}

/**
 * Returns true if the given format can support single-sample color
 * compression.  This function only checks the format.  In order to determine
 * if a surface supports CCS_E, several other factors need to be considered
 * such as tiling and sample count.  See isl_surf_get_ccs_surf for details.
 */
bool
isl_format_supports_ccs_e(const struct gen_device_info *devinfo,
                          enum isl_format format)
{
   if (!format_info_exists(format))
      return false;

   /* For simplicity, only report that a format supports CCS_E if blorp can
    * perform bit-for-bit copies with an image of that format while compressed.
    * This allows ISL users to avoid having to resolve the image before
    * performing such a copy. We may want to change this behavior in the
    * future.
    *
    * R11G11B10_FLOAT has no equivalent UINT format. Given how blorp_copy
    * currently works, bit-for-bit copy operations are not possible without an
    * intermediate resolve.
    */
   if (format == ISL_FORMAT_R11G11B10_FLOAT)
      return false;

   return format_gen(devinfo) >= format_info[format].ccs_e;
}

bool
isl_format_supports_multisampling(const struct gen_device_info *devinfo,
                                  enum isl_format format)
{
   /* From the Sandybridge PRM, Volume 4 Part 1 p72, SURFACE_STATE, Surface
    * Format:
    *
    *    If Number of Multisamples is set to a value other than
    *    MULTISAMPLECOUNT_1, this field cannot be set to the following
    *    formats:
    *
    *       - any format with greater than 64 bits per element
    *       - any compressed texture format (BC*)
    *       - any YCRCB* format
    *
    * The restriction on the format's size is removed on Broadwell. Moreover,
    * empirically it looks that even IvyBridge can handle multisampled surfaces
    * with format sizes all the way to 128-bits (RGBA32F, RGBA32I, RGBA32UI).
    *
    * Also, there is an exception for HiZ which we treat as a compressed
    * format and is allowed to be multisampled on Broadwell and earlier.
    */
   if (format == ISL_FORMAT_HIZ) {
      /* On SKL+, HiZ is always single-sampled even when the primary surface
       * is multisampled.  See also isl_surf_get_hiz_surf().
       */
      return devinfo->gen <= 8;
   } else if (devinfo->gen < 7 && isl_format_get_layout(format)->bpb > 64) {
      return false;
   } else if (isl_format_is_compressed(format)) {
      return false;
   } else if (isl_format_is_yuv(format)) {
      return false;
   } else {
      return true;
   }
}

/**
 * Returns true if the two formats are "CCS_E compatible" meaning that you can
 * render in one format with CCS_E enabled and then texture using the other
 * format without needing a resolve.
 *
 * Note: Even if the formats are compatible, special care must be taken if a
 * clear color is involved because the encoding of the clear color is heavily
 * format-dependent.
 */
bool
isl_formats_are_ccs_e_compatible(const struct gen_device_info *devinfo,
                                 enum isl_format format1,
                                 enum isl_format format2)
{
   /* They must support CCS_E */
   if (!isl_format_supports_ccs_e(devinfo, format1) ||
       !isl_format_supports_ccs_e(devinfo, format2))
      return false;

   const struct isl_format_layout *fmtl1 = isl_format_get_layout(format1);
   const struct isl_format_layout *fmtl2 = isl_format_get_layout(format2);

   /* The compression used by CCS is not dependent on the actual data encoding
    * of the format but only depends on the bit-layout of the channels.
    */
   return fmtl1->channels.r.bits == fmtl2->channels.r.bits &&
          fmtl1->channels.g.bits == fmtl2->channels.g.bits &&
          fmtl1->channels.b.bits == fmtl2->channels.b.bits &&
          fmtl1->channels.a.bits == fmtl2->channels.a.bits;
}

static bool
isl_format_has_channel_type(enum isl_format fmt, enum isl_base_type type)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(fmt);

   return fmtl->channels.r.type == type ||
          fmtl->channels.g.type == type ||
          fmtl->channels.b.type == type ||
          fmtl->channels.a.type == type ||
          fmtl->channels.l.type == type ||
          fmtl->channels.i.type == type ||
          fmtl->channels.p.type == type;
}

bool
isl_format_has_unorm_channel(enum isl_format fmt)
{
   return isl_format_has_channel_type(fmt, ISL_UNORM);
}

bool
isl_format_has_snorm_channel(enum isl_format fmt)
{
   return isl_format_has_channel_type(fmt, ISL_SNORM);
}

bool
isl_format_has_ufloat_channel(enum isl_format fmt)
{
   return isl_format_has_channel_type(fmt, ISL_UFLOAT);
}

bool
isl_format_has_sfloat_channel(enum isl_format fmt)
{
   return isl_format_has_channel_type(fmt, ISL_SFLOAT);
}

bool
isl_format_has_uint_channel(enum isl_format fmt)
{
   return isl_format_has_channel_type(fmt, ISL_UINT);
}

bool
isl_format_has_sint_channel(enum isl_format fmt)
{
   return isl_format_has_channel_type(fmt, ISL_SINT);
}

unsigned
isl_format_get_num_channels(enum isl_format fmt)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(fmt);

   assert(fmtl->channels.p.bits == 0);

   return (fmtl->channels.r.bits > 0) +
          (fmtl->channels.g.bits > 0) +
          (fmtl->channels.b.bits > 0) +
          (fmtl->channels.a.bits > 0) +
          (fmtl->channels.l.bits > 0) +
          (fmtl->channels.i.bits > 0);
}

uint32_t
isl_format_get_depth_format(enum isl_format fmt, bool has_stencil)
{
   switch (fmt) {
   default:
      unreachable("bad isl depth format");
   case ISL_FORMAT_R32_FLOAT_X8X24_TYPELESS:
      assert(has_stencil);
      return 0; /* D32_FLOAT_S8X24_UINT */
   case ISL_FORMAT_R32_FLOAT:
      assert(!has_stencil);
      return 1; /* D32_FLOAT */
   case ISL_FORMAT_R24_UNORM_X8_TYPELESS:
      if (has_stencil) {
         return 2; /* D24_UNORM_S8_UINT */
      } else {
         return 3; /* D24_UNORM_X8_UINT */
      }
   case ISL_FORMAT_R16_UNORM:
      assert(!has_stencil);
      return 5; /* D16_UNORM */
   }
}

enum isl_format
isl_format_rgb_to_rgba(enum isl_format rgb)
{
   assert(isl_format_is_rgb(rgb));

   switch (rgb) {
   case ISL_FORMAT_R32G32B32_FLOAT:    return ISL_FORMAT_R32G32B32A32_FLOAT;
   case ISL_FORMAT_R32G32B32_SINT:     return ISL_FORMAT_R32G32B32A32_SINT;
   case ISL_FORMAT_R32G32B32_UINT:     return ISL_FORMAT_R32G32B32A32_UINT;
   case ISL_FORMAT_R32G32B32_UNORM:    return ISL_FORMAT_R32G32B32A32_UNORM;
   case ISL_FORMAT_R32G32B32_SNORM:    return ISL_FORMAT_R32G32B32A32_SNORM;
   case ISL_FORMAT_R32G32B32_SSCALED:  return ISL_FORMAT_R32G32B32A32_SSCALED;
   case ISL_FORMAT_R32G32B32_USCALED:  return ISL_FORMAT_R32G32B32A32_USCALED;
   case ISL_FORMAT_R32G32B32_SFIXED:   return ISL_FORMAT_R32G32B32A32_SFIXED;
   case ISL_FORMAT_R8G8B8_UNORM:       return ISL_FORMAT_R8G8B8A8_UNORM;
   case ISL_FORMAT_R8G8B8_SNORM:       return ISL_FORMAT_R8G8B8A8_SNORM;
   case ISL_FORMAT_R8G8B8_SSCALED:     return ISL_FORMAT_R8G8B8A8_SSCALED;
   case ISL_FORMAT_R8G8B8_USCALED:     return ISL_FORMAT_R8G8B8A8_USCALED;
   case ISL_FORMAT_R16G16B16_FLOAT:    return ISL_FORMAT_R16G16B16A16_FLOAT;
   case ISL_FORMAT_R16G16B16_UNORM:    return ISL_FORMAT_R16G16B16A16_UNORM;
   case ISL_FORMAT_R16G16B16_SNORM:    return ISL_FORMAT_R16G16B16A16_SNORM;
   case ISL_FORMAT_R16G16B16_SSCALED:  return ISL_FORMAT_R16G16B16A16_SSCALED;
   case ISL_FORMAT_R16G16B16_USCALED:  return ISL_FORMAT_R16G16B16A16_USCALED;
   case ISL_FORMAT_R8G8B8_UNORM_SRGB:  return ISL_FORMAT_R8G8B8A8_UNORM_SRGB;
   case ISL_FORMAT_R16G16B16_UINT:     return ISL_FORMAT_R16G16B16A16_UINT;
   case ISL_FORMAT_R16G16B16_SINT:     return ISL_FORMAT_R16G16B16A16_SINT;
   case ISL_FORMAT_R8G8B8_UINT:        return ISL_FORMAT_R8G8B8A8_UINT;
   case ISL_FORMAT_R8G8B8_SINT:        return ISL_FORMAT_R8G8B8A8_SINT;
   default:
      return ISL_FORMAT_UNSUPPORTED;
   }
}

enum isl_format
isl_format_rgb_to_rgbx(enum isl_format rgb)
{
   assert(isl_format_is_rgb(rgb));

   switch (rgb) {
   case ISL_FORMAT_R32G32B32_FLOAT:
      return ISL_FORMAT_R32G32B32X32_FLOAT;
   case ISL_FORMAT_R16G16B16_UNORM:
      return ISL_FORMAT_R16G16B16X16_UNORM;
   case ISL_FORMAT_R16G16B16_FLOAT:
      return ISL_FORMAT_R16G16B16X16_FLOAT;
   case ISL_FORMAT_R8G8B8_UNORM:
      return ISL_FORMAT_R8G8B8X8_UNORM;
   case ISL_FORMAT_R8G8B8_UNORM_SRGB:
      return ISL_FORMAT_R8G8B8X8_UNORM_SRGB;
   default:
      return ISL_FORMAT_UNSUPPORTED;
   }
}

enum isl_format
isl_format_rgbx_to_rgba(enum isl_format rgbx)
{
   assert(isl_format_is_rgbx(rgbx));

   switch (rgbx) {
   case ISL_FORMAT_R32G32B32X32_FLOAT:
      return ISL_FORMAT_R32G32B32A32_FLOAT;
   case ISL_FORMAT_R16G16B16X16_UNORM:
      return ISL_FORMAT_R16G16B16A16_UNORM;
   case ISL_FORMAT_R16G16B16X16_FLOAT:
      return ISL_FORMAT_R16G16B16A16_FLOAT;
   case ISL_FORMAT_B8G8R8X8_UNORM:
      return ISL_FORMAT_B8G8R8A8_UNORM;
   case ISL_FORMAT_B8G8R8X8_UNORM_SRGB:
      return ISL_FORMAT_B8G8R8A8_UNORM_SRGB;
   case ISL_FORMAT_R8G8B8X8_UNORM:
      return ISL_FORMAT_R8G8B8A8_UNORM;
   case ISL_FORMAT_R8G8B8X8_UNORM_SRGB:
      return ISL_FORMAT_R8G8B8A8_UNORM_SRGB;
   case ISL_FORMAT_B10G10R10X2_UNORM:
      return ISL_FORMAT_B10G10R10A2_UNORM;
   case ISL_FORMAT_B5G5R5X1_UNORM:
      return ISL_FORMAT_B5G5R5A1_UNORM;
   case ISL_FORMAT_B5G5R5X1_UNORM_SRGB:
      return ISL_FORMAT_B5G5R5A1_UNORM_SRGB;
   default:
      assert(!"Invalid RGBX format");
      return rgbx;
   }
}

static inline void
pack_channel(const union isl_color_value *value, unsigned i,
             const struct isl_channel_layout *layout,
             enum isl_colorspace colorspace,
             uint32_t data_out[4])
{
   if (layout->type == ISL_VOID)
      return;

   if (colorspace == ISL_COLORSPACE_SRGB)
      assert(layout->type == ISL_UNORM);

   uint32_t packed;
   switch (layout->type) {
   case ISL_UNORM:
      if (colorspace == ISL_COLORSPACE_SRGB) {
         if (layout->bits == 8) {
            packed = util_format_linear_float_to_srgb_8unorm(value->f32[i]);
         } else {
            float srgb = util_format_linear_to_srgb_float(value->f32[i]);
            packed = _mesa_float_to_unorm(srgb, layout->bits);
         }
      } else {
         packed = _mesa_float_to_unorm(value->f32[i], layout->bits);
      }
      break;
   case ISL_SNORM:
      packed = _mesa_float_to_snorm(value->f32[i], layout->bits);
      break;
   case ISL_SFLOAT:
      assert(layout->bits == 16 || layout->bits == 32);
      if (layout->bits == 16) {
         packed = _mesa_float_to_half(value->f32[i]);
      } else {
         packed = value->u32[i];
      }
      break;
   case ISL_UINT:
      packed = MIN(value->u32[i], MAX_UINT(layout->bits));
      break;
   case ISL_SINT:
      packed = MIN(MAX(value->u32[i], MIN_INT(layout->bits)),
                   MAX_INT(layout->bits));
      break;

   default:
      unreachable("Invalid channel type");
   }

   unsigned dword = layout->start_bit / 32;
   unsigned bit = layout->start_bit % 32;
   assert(bit + layout->bits <= 32);
   data_out[dword] |= (packed & MAX_UINT(layout->bits)) << bit;
}

/**
 * Take an isl_color_value and pack it into the actual bits as specified by
 * the isl_format.  This function is very slow for a format conversion
 * function but should be fine for a single pixel worth of data.
 */
void
isl_color_value_pack(const union isl_color_value *value,
                     enum isl_format format,
                     uint32_t *data_out)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(format);
   assert(fmtl->colorspace == ISL_COLORSPACE_LINEAR ||
          fmtl->colorspace == ISL_COLORSPACE_SRGB);
   assert(!isl_format_is_compressed(format));

   memset(data_out, 0, isl_align(fmtl->bpb, 32) / 8);

   if (format == ISL_FORMAT_R9G9B9E5_SHAREDEXP) {
      data_out[0] = float3_to_rgb9e5(value->f32);
      return;
   } else if (format == ISL_FORMAT_R11G11B10_FLOAT) {
      data_out[0] = float3_to_r11g11b10f(value->f32);
      return;
   }

   pack_channel(value, 0, &fmtl->channels.r, fmtl->colorspace, data_out);
   pack_channel(value, 1, &fmtl->channels.g, fmtl->colorspace, data_out);
   pack_channel(value, 2, &fmtl->channels.b, fmtl->colorspace, data_out);
   pack_channel(value, 3, &fmtl->channels.a, ISL_COLORSPACE_LINEAR, data_out);
   pack_channel(value, 0, &fmtl->channels.l, fmtl->colorspace, data_out);
   pack_channel(value, 0, &fmtl->channels.i, ISL_COLORSPACE_LINEAR, data_out);
   assert(fmtl->channels.p.bits == 0);
}

/** Extend an N-bit signed integer to 32 bits */
static inline int32_t
sign_extend(int32_t x, unsigned bits)
{
   if (bits < 32) {
      unsigned shift = 32 - bits;
      return (x << shift) >> shift;
   } else {
      return x;
   }
}

static inline void
unpack_channel(union isl_color_value *value,
               unsigned start, unsigned count,
               const struct isl_channel_layout *layout,
               enum isl_colorspace colorspace,
               const uint32_t *data_in)
{
   if (layout->type == ISL_VOID)
      return;

   unsigned dword = layout->start_bit / 32;
   unsigned bit = layout->start_bit % 32;
   assert(bit + layout->bits <= 32);
   uint32_t packed = (data_in[dword] >> bit) & MAX_UINT(layout->bits);

   union {
      uint32_t u32;
      float f32;
   } unpacked;

   if (colorspace == ISL_COLORSPACE_SRGB)
      assert(layout->type == ISL_UNORM);

   switch (layout->type) {
   case ISL_UNORM:
      unpacked.f32 = _mesa_unorm_to_float(packed, layout->bits);
      if (colorspace == ISL_COLORSPACE_SRGB) {
         if (layout->bits == 8) {
            unpacked.f32 = util_format_srgb_8unorm_to_linear_float(packed);
         } else {
            float srgb = _mesa_unorm_to_float(packed, layout->bits);
            unpacked.f32 = util_format_srgb_to_linear_float(srgb);
         }
      } else {
         unpacked.f32 = _mesa_unorm_to_float(packed, layout->bits);
      }
      break;
   case ISL_SNORM:
      unpacked.f32 = _mesa_snorm_to_float(sign_extend(packed, layout->bits),
                                          layout->bits);
      break;
   case ISL_SFLOAT:
      assert(layout->bits == 16 || layout->bits == 32);
      if (layout->bits == 16) {
         unpacked.f32 = _mesa_half_to_float(packed);
      } else {
         unpacked.u32 = packed;
      }
      break;
   case ISL_UINT:
      unpacked.u32 = packed;
      break;
   case ISL_SINT:
      unpacked.u32 = sign_extend(packed, layout->bits);
      break;

   default:
      unreachable("Invalid channel type");
   }

   for (unsigned i = 0; i < count; i++)
      value->u32[start + i] = unpacked.u32;
}

/**
 * Take unpack an isl_color_value from the actual bits as specified by
 * the isl_format.  This function is very slow for a format conversion
 * function but should be fine for a single pixel worth of data.
 */
void
isl_color_value_unpack(union isl_color_value *value,
                       enum isl_format format,
                       const uint32_t data_in[4])
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(format);
   assert(fmtl->colorspace == ISL_COLORSPACE_LINEAR ||
          fmtl->colorspace == ISL_COLORSPACE_SRGB);
   assert(!isl_format_is_compressed(format));

   /* Default to opaque black. */
   memset(value, 0, sizeof(*value));
   if (isl_format_has_int_channel(format)) {
      value->u32[3] = 1u;
   } else {
      value->f32[3] = 1.0f;
   }

   if (format == ISL_FORMAT_R9G9B9E5_SHAREDEXP) {
      rgb9e5_to_float3(data_in[0], value->f32);
      return;
   } else if (format == ISL_FORMAT_R11G11B10_FLOAT) {
      r11g11b10f_to_float3(data_in[0], value->f32);
      return;
   }

   unpack_channel(value, 0, 1, &fmtl->channels.r, fmtl->colorspace, data_in);
   unpack_channel(value, 1, 1, &fmtl->channels.g, fmtl->colorspace, data_in);
   unpack_channel(value, 2, 1, &fmtl->channels.b, fmtl->colorspace, data_in);
   unpack_channel(value, 3, 1, &fmtl->channels.a, ISL_COLORSPACE_LINEAR, data_in);
   unpack_channel(value, 0, 3, &fmtl->channels.l, fmtl->colorspace, data_in);
   unpack_channel(value, 0, 4, &fmtl->channels.i, ISL_COLORSPACE_LINEAR, data_in);
   assert(fmtl->channels.p.bits == 0);
}
