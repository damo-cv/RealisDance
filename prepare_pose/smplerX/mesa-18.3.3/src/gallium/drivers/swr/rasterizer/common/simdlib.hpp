/****************************************************************************
 * Copyright (C) 2017 Intel Corporation.   All Rights Reserved.
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
 ****************************************************************************/
#pragma once

#include "simdlib_types.hpp"

// For documentation, please see the following include...
// #include "simdlib_interface.hpp"

namespace SIMDImpl
{
    namespace SIMD128Impl
    {
#if SIMD_ARCH >= SIMD_ARCH_AVX
        struct AVXImpl
        {
#define __SIMD_LIB_AVX_HPP__
#include "simdlib_128_avx.inl"
#undef __SIMD_LIB_AVX_HPP__
        }; // struct AVXImpl
#endif     // #if SIMD_ARCH >= SIMD_ARCH_AVX

#if SIMD_ARCH >= SIMD_ARCH_AVX2
        struct AVX2Impl : AVXImpl
        {
#define __SIMD_LIB_AVX2_HPP__
#include "simdlib_128_avx2.inl"
#undef __SIMD_LIB_AVX2_HPP__
        }; // struct AVX2Impl
#endif     // #if SIMD_ARCH >= SIMD_ARCH_AVX2

#if SIMD_ARCH >= SIMD_ARCH_AVX512
        struct AVX512Impl : AVX2Impl
        {
#if defined(SIMD_OPT_128_AVX512)
#define __SIMD_LIB_AVX512_HPP__
#include "simdlib_128_avx512.inl"
#if defined(SIMD_ARCH_KNIGHTS)
#include "simdlib_128_avx512_knights.inl"
#else // optimize for core
#include "simdlib_128_avx512_core.inl"
#endif // defined(SIMD_ARCH_KNIGHTS)
#undef __SIMD_LIB_AVX512_HPP__
#endif     // SIMD_OPT_128_AVX512
        }; // struct AVX2Impl
#endif     // #if SIMD_ARCH >= SIMD_ARCH_AVX512

        struct Traits : SIMDImpl::Traits
        {
#if SIMD_ARCH == SIMD_ARCH_AVX
            using IsaImpl = AVXImpl;
#elif SIMD_ARCH == SIMD_ARCH_AVX2
            using IsaImpl = AVX2Impl;
#elif SIMD_ARCH == SIMD_ARCH_AVX512
            using IsaImpl = AVX512Impl;
#else
#error Invalid value for SIMD_ARCH
#endif

            using Float   = SIMD128Impl::Float;
            using Double  = SIMD128Impl::Double;
            using Integer = SIMD128Impl::Integer;
            using Vec4    = SIMD128Impl::Vec4;
            using Mask    = SIMD128Impl::Mask;
        };
    } // namespace SIMD128Impl

    namespace SIMD256Impl
    {
#if SIMD_ARCH >= SIMD_ARCH_AVX
        struct AVXImpl
        {
#define __SIMD_LIB_AVX_HPP__
#include "simdlib_256_avx.inl"
#undef __SIMD_LIB_AVX_HPP__
        }; // struct AVXImpl
#endif     // #if SIMD_ARCH >= SIMD_ARCH_AVX

#if SIMD_ARCH >= SIMD_ARCH_AVX2
        struct AVX2Impl : AVXImpl
        {
#define __SIMD_LIB_AVX2_HPP__
#include "simdlib_256_avx2.inl"
#undef __SIMD_LIB_AVX2_HPP__
        }; // struct AVX2Impl
#endif     // #if SIMD_ARCH >= SIMD_ARCH_AVX2

#if SIMD_ARCH >= SIMD_ARCH_AVX512
        struct AVX512Impl : AVX2Impl
        {
#if defined(SIMD_OPT_256_AVX512)
#define __SIMD_LIB_AVX512_HPP__
#include "simdlib_256_avx512.inl"
#if defined(SIMD_ARCH_KNIGHTS)
#include "simdlib_256_avx512_knights.inl"
#else // optimize for core
#include "simdlib_256_avx512_core.inl"
#endif // defined(SIMD_ARCH_KNIGHTS)
#undef __SIMD_LIB_AVX512_HPP__
#endif     // SIMD_OPT_256_AVX512
        }; // struct AVX2Impl
#endif     // #if SIMD_ARCH >= SIMD_ARCH_AVX512

        struct Traits : SIMDImpl::Traits
        {
#if SIMD_ARCH == SIMD_ARCH_AVX
            using IsaImpl = AVXImpl;
#elif SIMD_ARCH == SIMD_ARCH_AVX2
            using IsaImpl = AVX2Impl;
#elif SIMD_ARCH == SIMD_ARCH_AVX512
            using IsaImpl = AVX512Impl;
#else
#error Invalid value for SIMD_ARCH
#endif

            using Float   = SIMD256Impl::Float;
            using Double  = SIMD256Impl::Double;
            using Integer = SIMD256Impl::Integer;
            using Vec4    = SIMD256Impl::Vec4;
            using Mask    = SIMD256Impl::Mask;
        };
    } // namespace SIMD256Impl

    namespace SIMD512Impl
    {
#if SIMD_ARCH >= SIMD_ARCH_AVX
        template <typename SIMD256T>
        struct AVXImplBase
        {
#define __SIMD_LIB_AVX_HPP__
#include "simdlib_512_emu.inl"
#include "simdlib_512_emu_masks.inl"
#undef __SIMD_LIB_AVX_HPP__
        }; // struct AVXImplBase
        using AVXImpl = AVXImplBase<SIMD256Impl::AVXImpl>;
#endif // #if SIMD_ARCH >= SIMD_ARCH_AVX

#if SIMD_ARCH >= SIMD_ARCH_AVX2
        using AVX2Impl = AVXImplBase<SIMD256Impl::AVX2Impl>;
#endif // #if SIMD_ARCH >= SIMD_ARCH_AVX2

#if SIMD_ARCH >= SIMD_ARCH_AVX512
        struct AVX512Impl : AVXImplBase<SIMD256Impl::AVX512Impl>
        {
#define __SIMD_LIB_AVX512_HPP__
#include "simdlib_512_avx512.inl"
#include "simdlib_512_avx512_masks.inl"
#if defined(SIMD_ARCH_KNIGHTS)
#include "simdlib_512_avx512_knights.inl"
#include "simdlib_512_avx512_masks_knights.inl"
#else // optimize for core
#include "simdlib_512_avx512_core.inl"
#include "simdlib_512_avx512_masks_core.inl"
#endif // defined(SIMD_ARCH_KNIGHTS)
#undef __SIMD_LIB_AVX512_HPP__
        }; // struct AVX512ImplBase
#endif     // #if SIMD_ARCH >= SIMD_ARCH_AVX512

        struct Traits : SIMDImpl::Traits
        {
#if SIMD_ARCH == SIMD_ARCH_AVX
            using IsaImpl = AVXImpl;
#elif SIMD_ARCH == SIMD_ARCH_AVX2
            using IsaImpl = AVX2Impl;
#elif SIMD_ARCH == SIMD_ARCH_AVX512
            using IsaImpl = AVX512Impl;
#else
#error Invalid value for SIMD_ARCH
#endif

            using Float   = SIMD512Impl::Float;
            using Double  = SIMD512Impl::Double;
            using Integer = SIMD512Impl::Integer;
            using Vec4    = SIMD512Impl::Vec4;
            using Mask    = SIMD512Impl::Mask;
        };
    } // namespace SIMD512Impl
} // namespace SIMDImpl

template <typename Traits>
struct SIMDBase : Traits::IsaImpl
{
    using CompareType = typename Traits::CompareType;
    using ScaleFactor = typename Traits::ScaleFactor;
    using RoundMode   = typename Traits::RoundMode;
    using SIMD        = typename Traits::IsaImpl;
    using Float       = typename Traits::Float;
    using Double      = typename Traits::Double;
    using Integer     = typename Traits::Integer;
    using Vec4        = typename Traits::Vec4;
    using Mask        = typename Traits::Mask;

    static const size_t VECTOR_BYTES = sizeof(Float);

    // Populates a SIMD Vec4 from a non-simd vector. So p = xyzw becomes xxxx yyyy zzzz wwww.
    static SIMDINLINE void vec4_load1_ps(Vec4& r, const float* p)
    {
        r[0] = SIMD::set1_ps(p[0]);
        r[1] = SIMD::set1_ps(p[1]);
        r[2] = SIMD::set1_ps(p[2]);
        r[3] = SIMD::set1_ps(p[3]);
    }

    static SIMDINLINE void vec4_set1_vps(Vec4& r, Float const& s)
    {
        r[0] = s;
        r[1] = s;
        r[2] = s;
        r[3] = s;
    }

    static SIMDINLINE Float vec4_dp3_ps(const Vec4& v0, const Vec4& v1)
    {
        Float tmp, r;
        r = SIMD::mul_ps(v0[0], v1[0]); // (v0.x*v1.x)

        tmp = SIMD::mul_ps(v0[1], v1[1]); // (v0.y*v1.y)
        r   = SIMD::add_ps(r, tmp);       // (v0.x*v1.x) + (v0.y*v1.y)

        tmp = SIMD::mul_ps(v0[2], v1[2]); // (v0.z*v1.z)
        r   = SIMD::add_ps(r, tmp);       // (v0.x*v1.x) + (v0.y*v1.y) + (v0.z*v1.z)

        return r;
    }

    static SIMDINLINE Float vec4_dp4_ps(const Vec4& v0, const Vec4& v1)
    {
        Float tmp, r;
        r = SIMD::mul_ps(v0[0], v1[0]); // (v0.x*v1.x)

        tmp = SIMD::mul_ps(v0[1], v1[1]); // (v0.y*v1.y)
        r   = SIMD::add_ps(r, tmp);       // (v0.x*v1.x) + (v0.y*v1.y)

        tmp = SIMD::mul_ps(v0[2], v1[2]); // (v0.z*v1.z)
        r   = SIMD::add_ps(r, tmp);       // (v0.x*v1.x) + (v0.y*v1.y) + (v0.z*v1.z)

        tmp = SIMD::mul_ps(v0[3], v1[3]); // (v0.w*v1.w)
        r   = SIMD::add_ps(r, tmp);       // (v0.x*v1.x) + (v0.y*v1.y) + (v0.z*v1.z)

        return r;
    }

    static SIMDINLINE Float vec4_rcp_length_ps(const Vec4& v)
    {
        Float length = vec4_dp4_ps(v, v);
        return SIMD::rsqrt_ps(length);
    }

    static SIMDINLINE void vec4_normalize_ps(Vec4& r, const Vec4& v)
    {
        Float rcpLength = vec4_rcp_length_ps(v);

        r[0] = SIMD::mul_ps(v[0], rcpLength);
        r[1] = SIMD::mul_ps(v[1], rcpLength);
        r[2] = SIMD::mul_ps(v[2], rcpLength);
        r[3] = SIMD::mul_ps(v[3], rcpLength);
    }

    static SIMDINLINE void vec4_mul_ps(Vec4& r, const Vec4& v, Float const& s)
    {
        r[0] = SIMD::mul_ps(v[0], s);
        r[1] = SIMD::mul_ps(v[1], s);
        r[2] = SIMD::mul_ps(v[2], s);
        r[3] = SIMD::mul_ps(v[3], s);
    }

    static SIMDINLINE void vec4_mul_ps(Vec4& r, const Vec4& v0, const Vec4& v1)
    {
        r[0] = SIMD::mul_ps(v0[0], v1[0]);
        r[1] = SIMD::mul_ps(v0[1], v1[1]);
        r[2] = SIMD::mul_ps(v0[2], v1[2]);
        r[3] = SIMD::mul_ps(v0[3], v1[3]);
    }

    static SIMDINLINE void vec4_add_ps(Vec4& r, const Vec4& v0, Float const& s)
    {
        r[0] = SIMD::add_ps(v0[0], s);
        r[1] = SIMD::add_ps(v0[1], s);
        r[2] = SIMD::add_ps(v0[2], s);
        r[3] = SIMD::add_ps(v0[3], s);
    }

    static SIMDINLINE void vec4_add_ps(Vec4& r, const Vec4& v0, const Vec4& v1)
    {
        r[0] = SIMD::add_ps(v0[0], v1[0]);
        r[1] = SIMD::add_ps(v0[1], v1[1]);
        r[2] = SIMD::add_ps(v0[2], v1[2]);
        r[3] = SIMD::add_ps(v0[3], v1[3]);
    }

    static SIMDINLINE void vec4_min_ps(Vec4& r, const Vec4& v0, Float const& s)
    {
        r[0] = SIMD::min_ps(v0[0], s);
        r[1] = SIMD::min_ps(v0[1], s);
        r[2] = SIMD::min_ps(v0[2], s);
        r[3] = SIMD::min_ps(v0[3], s);
    }

    static SIMDINLINE void vec4_max_ps(Vec4& r, const Vec4& v0, Float const& s)
    {
        r[0] = SIMD::max_ps(v0[0], s);
        r[1] = SIMD::max_ps(v0[1], s);
        r[2] = SIMD::max_ps(v0[2], s);
        r[3] = SIMD::max_ps(v0[3], s);
    }

    // Matrix4x4 * Vector4
    //   outVec.x = (m00 * v.x) + (m01 * v.y) + (m02 * v.z) + (m03 * v.w)
    //   outVec.y = (m10 * v.x) + (m11 * v.y) + (m12 * v.z) + (m13 * v.w)
    //   outVec.z = (m20 * v.x) + (m21 * v.y) + (m22 * v.z) + (m23 * v.w)
    //   outVec.w = (m30 * v.x) + (m31 * v.y) + (m32 * v.z) + (m33 * v.w)
    static SIMDINLINE void SIMDCALL mat4x4_vec4_multiply(Vec4&        result,
                                                         const float* pMatrix,
                                                         const Vec4&  v)
    {
        Float m;
        Float r0;
        Float r1;

        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 3); // m[row][3]
        r1        = SIMD::mul_ps(m, v[3]);               // (m3 * v.z)
        r0        = SIMD::add_ps(r0, r1); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * v.w)
        result[0] = r0;

        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 3); // m[row][3]
        r1        = SIMD::mul_ps(m, v[3]);               // (m3 * v.z)
        r0        = SIMD::add_ps(r0, r1); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * v.w)
        result[1] = r0;

        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 3); // m[row][3]
        r1        = SIMD::mul_ps(m, v[3]);               // (m3 * v.z)
        r0        = SIMD::add_ps(r0, r1); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * v.w)
        result[2] = r0;

        m         = SIMD::load1_ps(pMatrix + 3 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 3 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 3 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 3 * 4 + 3); // m[row][3]
        r1        = SIMD::mul_ps(m, v[3]);               // (m3 * v.z)
        r0        = SIMD::add_ps(r0, r1); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * v.w)
        result[3] = r0;
    }

    // Matrix4x4 * Vector3 - Direction Vector where w = 0.
    //   outVec.x = (m00 * v.x) + (m01 * v.y) + (m02 * v.z) + (m03 * 0)
    //   outVec.y = (m10 * v.x) + (m11 * v.y) + (m12 * v.z) + (m13 * 0)
    //   outVec.z = (m20 * v.x) + (m21 * v.y) + (m22 * v.z) + (m23 * 0)
    //   outVec.w = (m30 * v.x) + (m31 * v.y) + (m32 * v.z) + (m33 * 0)
    static SIMDINLINE void SIMDCALL mat3x3_vec3_w0_multiply(Vec4&        result,
                                                            const float* pMatrix,
                                                            const Vec4&  v)
    {
        Float m;
        Float r0;
        Float r1;

        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        result[0] = r0;

        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        result[1] = r0;

        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        result[2] = r0;

        result[3] = SIMD::setzero_ps();
    }

    // Matrix4x4 * Vector3 - Position vector where w = 1.
    //   outVec.x = (m00 * v.x) + (m01 * v.y) + (m02 * v.z) + (m03 * 1)
    //   outVec.y = (m10 * v.x) + (m11 * v.y) + (m12 * v.z) + (m13 * 1)
    //   outVec.z = (m20 * v.x) + (m21 * v.y) + (m22 * v.z) + (m23 * 1)
    //   outVec.w = (m30 * v.x) + (m31 * v.y) + (m32 * v.z) + (m33 * 1)
    static SIMDINLINE void SIMDCALL mat4x4_vec3_w1_multiply(Vec4&        result,
                                                            const float* pMatrix,
                                                            const Vec4&  v)
    {
        Float m;
        Float r0;
        Float r1;

        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 3); // m[row][3]
        r0        = SIMD::add_ps(r0, m); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * 1)
        result[0] = r0;

        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 3); // m[row][3]
        r0        = SIMD::add_ps(r0, m); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * 1)
        result[1] = r0;

        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 3); // m[row][3]
        r0        = SIMD::add_ps(r0, m); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * 1)
        result[2] = r0;

        m         = SIMD::load1_ps(pMatrix + 3 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 3 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 3 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 3 * 4 + 3); // m[row][3]
        result[3] = SIMD::add_ps(r0, m); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * 1)
    }

    static SIMDINLINE void SIMDCALL mat4x3_vec3_w1_multiply(Vec4&        result,
                                                            const float* pMatrix,
                                                            const Vec4&  v)
    {
        Float m;
        Float r0;
        Float r1;

        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 0 * 4 + 3); // m[row][3]
        r0        = SIMD::add_ps(r0, m); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * 1)
        result[0] = r0;

        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 1 * 4 + 3); // m[row][3]
        r0        = SIMD::add_ps(r0, m); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * 1)
        result[1] = r0;

        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 0); // m[row][0]
        r0        = SIMD::mul_ps(m, v[0]);               // (m00 * v.x)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 1); // m[row][1]
        r1        = SIMD::mul_ps(m, v[1]);               // (m1 * v.y)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 2); // m[row][2]
        r1        = SIMD::mul_ps(m, v[2]);               // (m2 * v.z)
        r0        = SIMD::add_ps(r0, r1);                // (m0 * v.x) + (m1 * v.y) + (m2 * v.z)
        m         = SIMD::load1_ps(pMatrix + 2 * 4 + 3); // m[row][3]
        r0        = SIMD::add_ps(r0, m); // (m0 * v.x) + (m1 * v.y) + (m2 * v.z) + (m2 * 1)
        result[2] = r0;
        result[3] = SIMD::set1_ps(1.0f);
    }
}; // struct SIMDBase

using SIMD128 = SIMDBase<SIMDImpl::SIMD128Impl::Traits>;
using SIMD256 = SIMDBase<SIMDImpl::SIMD256Impl::Traits>;
using SIMD512 = SIMDBase<SIMDImpl::SIMD512Impl::Traits>;

template <typename SIMD_T>
using CompareType = typename SIMD_T::CompareType;
template <typename SIMD_T>
using ScaleFactor = typename SIMD_T::ScaleFactor;
template <typename SIMD_T>
using RoundMode = typename SIMD_T::RoundMode;
template <typename SIMD_T>
using Float = typename SIMD_T::Float;
template <typename SIMD_T>
using Double = typename SIMD_T::Double;
template <typename SIMD_T>
using Integer = typename SIMD_T::Integer;
template <typename SIMD_T>
using Vec4 = typename SIMD_T::Vec4;
template <typename SIMD_T>
using Mask = typename SIMD_T::Mask;

template <typename SIMD_T>
struct SIMDVecEqual
{
    INLINE bool operator()(Integer<SIMD_T> a, Integer<SIMD_T> b) const
    {
        Integer<SIMD_T> c = SIMD_T::xor_si(a, b);
        return SIMD_T::testz_si(c, c);
    }

    INLINE bool operator()(Float<SIMD_T> a, Float<SIMD_T> b) const
    {
        return this->operator()(SIMD_T::castps_si(a), SIMD_T::castps_si(b));
    }

    INLINE bool operator()(Double<SIMD_T> a, Double<SIMD_T> b) const
    {
        return this->operator()(SIMD_T::castpd_si(a), SIMD_T::castpd_si(b));
    }
};

template <typename SIMD_T>
struct SIMDVecHash
{
    INLINE uint32_t operator()(Integer<SIMD_T> val) const
    {
#if defined(_WIN64) || !defined(_WIN32) // assume non-Windows is always 64-bit
        static_assert(sizeof(void*) == 8, "This path only meant for 64-bit code");

        uint64_t              crc32          = 0;
        const uint64_t*       pData          = reinterpret_cast<const uint64_t*>(&val);
        static const uint32_t loopIterations = sizeof(val) / sizeof(void*);
        static_assert(loopIterations * sizeof(void*) == sizeof(val), "bad vector size");

        for (uint32_t i = 0; i < loopIterations; ++i)
        {
            crc32 = _mm_crc32_u64(crc32, pData[i]);
        }

        return static_cast<uint32_t>(crc32);
#else
        static_assert(sizeof(void*) == 4, "This path only meant for 32-bit code");

        uint32_t crc32 = 0;
        const uint32_t* pData = reinterpret_cast<const uint32_t*>(&val);
        static const uint32_t loopIterations = sizeof(val) / sizeof(void*);
        static_assert(loopIterations * sizeof(void*) == sizeof(val), "bad vector size");

        for (uint32_t i = 0; i < loopIterations; ++i)
        {
            crc32 = _mm_crc32_u32(crc32, pData[i]);
        }

        return crc32;
#endif
    };

    INLINE uint32_t operator()(Float<SIMD_T> val) const
    {
        return operator()(SIMD_T::castps_si(val));
    };
    INLINE uint32_t operator()(Double<SIMD_T> val) const
    {
        return operator()(SIMD_T::castpd_si(val));
    }
};
