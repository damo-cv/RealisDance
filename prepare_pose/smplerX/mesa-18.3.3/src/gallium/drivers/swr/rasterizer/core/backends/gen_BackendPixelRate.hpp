//============================================================================
// Copyright (C) 2017 Intel Corporation.   All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice (including the next
// paragraph) shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// @file gen_BackendPixelRate.hpp
//
// @brief auto-generated file
//
// DO NOT EDIT
//
// Generation Command Line:
//  ./rasterizer/codegen/gen_backends.py
//    --outdir
//    rasterizer/core/backends
//    --dim
//    5
//    2
//    3
//    2
//    2
//    2
//    --numfiles
//    4
//    --cpp
//    --hpp
//
//============================================================================

// clang-format off

void InitBackendPixelRate0();
void InitBackendPixelRate1();
void InitBackendPixelRate2();
void InitBackendPixelRate3();

static INLINE void InitBackendPixelRate()
{
    InitBackendPixelRate0();
    InitBackendPixelRate1();
    InitBackendPixelRate2();
    InitBackendPixelRate3();
}
// clang-format on
