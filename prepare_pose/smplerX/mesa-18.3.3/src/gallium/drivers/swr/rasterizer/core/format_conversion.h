/****************************************************************************
 * Copyright (C) 2014-2015 Intel Corporation.   All Rights Reserved.
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
 *
 * @file format_conversion.h
 *
 * @brief API implementation
 *
 ******************************************************************************/
#include "format_types.h"
#include "format_traits.h"

//////////////////////////////////////////////////////////////////////////
/// @brief Load SIMD packed pixels in SOA format and converts to
///        SOA RGBA32_FLOAT format.
/// @param pSrc - source data in SOA form
/// @param dst - output data in SOA form
template <SWR_FORMAT SrcFormat>
INLINE void LoadSOA(const uint8_t* pSrc, simdvector& dst)
{
    // fast path for float32
    if ((FormatTraits<SrcFormat>::GetType(0) == SWR_TYPE_FLOAT) &&
        (FormatTraits<SrcFormat>::GetBPC(0) == 32))
    {
        auto lambda = [&](int comp) {
            simdscalar vComp = _simd_load_ps((const float*)(pSrc + comp * sizeof(simdscalar)));

            dst.v[FormatTraits<SrcFormat>::swizzle(comp)] = vComp;
        };

        UnrollerL<0, FormatTraits<SrcFormat>::numComps, 1>::step(lambda);
        return;
    }

    auto lambda = [&](int comp) {
        // load SIMD components
        simdscalar vComp = FormatTraits<SrcFormat>::loadSOA(comp, pSrc);

        // unpack
        vComp = FormatTraits<SrcFormat>::unpack(comp, vComp);

        // convert
        if (FormatTraits<SrcFormat>::isNormalized(comp))
        {
            vComp = _simd_cvtepi32_ps(_simd_castps_si(vComp));
            vComp = _simd_mul_ps(vComp, _simd_set1_ps(FormatTraits<SrcFormat>::toFloat(comp)));
        }

        dst.v[FormatTraits<SrcFormat>::swizzle(comp)] = vComp;

        pSrc += (FormatTraits<SrcFormat>::GetBPC(comp) * KNOB_SIMD_WIDTH) / 8;
    };

    UnrollerL<0, FormatTraits<SrcFormat>::numComps, 1>::step(lambda);
}

//////////////////////////////////////////////////////////////////////////
/// @brief Clamps the given component based on the requirements on the
///        Format template arg
/// @param vComp - SIMD vector of floats
/// @param Component - component
template <SWR_FORMAT Format>
INLINE simdscalar Clamp(simdscalar const& vC, uint32_t Component)
{
    simdscalar vComp = vC;
    if (FormatTraits<Format>::isNormalized(Component))
    {
        if (FormatTraits<Format>::GetType(Component) == SWR_TYPE_UNORM)
        {
            vComp = _simd_max_ps(vComp, _simd_setzero_ps());
        }

        if (FormatTraits<Format>::GetType(Component) == SWR_TYPE_SNORM)
        {
            vComp = _simd_max_ps(vComp, _simd_set1_ps(-1.0f));
        }
        vComp = _simd_min_ps(vComp, _simd_set1_ps(1.0f));
    }
    else if (FormatTraits<Format>::GetBPC(Component) < 32)
    {
        if (FormatTraits<Format>::GetType(Component) == SWR_TYPE_UINT)
        {
            int         iMax   = (1 << FormatTraits<Format>::GetBPC(Component)) - 1;
            int         iMin   = 0;
            simdscalari vCompi = _simd_castps_si(vComp);
            vCompi             = _simd_max_epu32(vCompi, _simd_set1_epi32(iMin));
            vCompi             = _simd_min_epu32(vCompi, _simd_set1_epi32(iMax));
            vComp              = _simd_castsi_ps(vCompi);
        }
        else if (FormatTraits<Format>::GetType(Component) == SWR_TYPE_SINT)
        {
            int         iMax   = (1 << (FormatTraits<Format>::GetBPC(Component) - 1)) - 1;
            int         iMin   = -1 - iMax;
            simdscalari vCompi = _simd_castps_si(vComp);
            vCompi             = _simd_max_epi32(vCompi, _simd_set1_epi32(iMin));
            vCompi             = _simd_min_epi32(vCompi, _simd_set1_epi32(iMax));
            vComp              = _simd_castsi_ps(vCompi);
        }
    }

    return vComp;
}

//////////////////////////////////////////////////////////////////////////
/// @brief Normalize the given component based on the requirements on the
///        Format template arg
/// @param vComp - SIMD vector of floats
/// @param Component - component
template <SWR_FORMAT Format>
INLINE simdscalar Normalize(simdscalar const& vC, uint32_t Component)
{
    simdscalar vComp = vC;
    if (FormatTraits<Format>::isNormalized(Component))
    {
        vComp = _simd_mul_ps(vComp, _simd_set1_ps(FormatTraits<Format>::fromFloat(Component)));
        vComp = _simd_castsi_ps(_simd_cvtps_epi32(vComp));
    }
    return vComp;
}

//////////////////////////////////////////////////////////////////////////
/// @brief Convert and store simdvector of pixels in SOA
///        RGBA32_FLOAT to SOA format
/// @param src - source data in SOA form
/// @param dst - output data in SOA form
template <SWR_FORMAT DstFormat>
INLINE void StoreSOA(const simdvector& src, uint8_t* pDst)
{
    // fast path for float32
    if ((FormatTraits<DstFormat>::GetType(0) == SWR_TYPE_FLOAT) &&
        (FormatTraits<DstFormat>::GetBPC(0) == 32))
    {
        for (uint32_t comp = 0; comp < FormatTraits<DstFormat>::numComps; ++comp)
        {
            simdscalar vComp = src.v[FormatTraits<DstFormat>::swizzle(comp)];

            // Gamma-correct
            if (FormatTraits<DstFormat>::isSRGB)
            {
                if (comp < 3) // Input format is always RGBA32_FLOAT.
                {
                    vComp = FormatTraits<R32G32B32A32_FLOAT>::convertSrgb(comp, vComp);
                }
            }

            _simd_store_ps((float*)(pDst + comp * sizeof(simdscalar)), vComp);
        }
        return;
    }

    auto lambda = [&](int comp) {
        simdscalar vComp = src.v[FormatTraits<DstFormat>::swizzle(comp)];

        // Gamma-correct
        if (FormatTraits<DstFormat>::isSRGB)
        {
            if (comp < 3) // Input format is always RGBA32_FLOAT.
            {
                vComp = FormatTraits<R32G32B32A32_FLOAT>::convertSrgb(comp, vComp);
            }
        }

        // clamp
        vComp = Clamp<DstFormat>(vComp, comp);

        // normalize
        vComp = Normalize<DstFormat>(vComp, comp);

        // pack
        vComp = FormatTraits<DstFormat>::pack(comp, vComp);

        // store
        FormatTraits<DstFormat>::storeSOA(comp, pDst, vComp);

        pDst += (FormatTraits<DstFormat>::GetBPC(comp) * KNOB_SIMD_WIDTH) / 8;
    };

    UnrollerL<0, FormatTraits<DstFormat>::numComps, 1>::step(lambda);
}

#if ENABLE_AVX512_SIMD16
//////////////////////////////////////////////////////////////////////////
/// @brief Load SIMD packed pixels in SOA format and converts to
///        SOA RGBA32_FLOAT format.
/// @param pSrc - source data in SOA form
/// @param dst - output data in SOA form
template <SWR_FORMAT SrcFormat>
INLINE void SIMDCALL LoadSOA(const uint8_t* pSrc, simd16vector& dst)
{
    // fast path for float32
    if ((FormatTraits<SrcFormat>::GetType(0) == SWR_TYPE_FLOAT) &&
        (FormatTraits<SrcFormat>::GetBPC(0) == 32))
    {
        auto lambda = [&](int comp) {
            simd16scalar vComp =
                _simd16_load_ps(reinterpret_cast<const float*>(pSrc + comp * sizeof(simd16scalar)));

            dst.v[FormatTraits<SrcFormat>::swizzle(comp)] = vComp;
        };

        UnrollerL<0, FormatTraits<SrcFormat>::numComps, 1>::step(lambda);
        return;
    }

    auto lambda = [&](int comp) {
        // load SIMD components
        simd16scalar vComp = FormatTraits<SrcFormat>::loadSOA_16(comp, pSrc);

        // unpack
        vComp = FormatTraits<SrcFormat>::unpack(comp, vComp);

        // convert
        if (FormatTraits<SrcFormat>::isNormalized(comp))
        {
            vComp = _simd16_cvtepi32_ps(_simd16_castps_si(vComp));
            vComp = _simd16_mul_ps(vComp, _simd16_set1_ps(FormatTraits<SrcFormat>::toFloat(comp)));
        }

        dst.v[FormatTraits<SrcFormat>::swizzle(comp)] = vComp;

        pSrc += (FormatTraits<SrcFormat>::GetBPC(comp) * KNOB_SIMD16_WIDTH) / 8;
    };

    UnrollerL<0, FormatTraits<SrcFormat>::numComps, 1>::step(lambda);
}

//////////////////////////////////////////////////////////////////////////
/// @brief Clamps the given component based on the requirements on the
///        Format template arg
/// @param vComp - SIMD vector of floats
/// @param Component - component
template <SWR_FORMAT Format>
INLINE simd16scalar SIMDCALL Clamp(simd16scalar const& v, uint32_t Component)
{
    simd16scalar vComp = v;
    if (FormatTraits<Format>::isNormalized(Component))
    {
        if (FormatTraits<Format>::GetType(Component) == SWR_TYPE_UNORM)
        {
            vComp = _simd16_max_ps(vComp, _simd16_setzero_ps());
        }

        if (FormatTraits<Format>::GetType(Component) == SWR_TYPE_SNORM)
        {
            vComp = _simd16_max_ps(vComp, _simd16_set1_ps(-1.0f));
        }
        vComp = _simd16_min_ps(vComp, _simd16_set1_ps(1.0f));
    }
    else if (FormatTraits<Format>::GetBPC(Component) < 32)
    {
        if (FormatTraits<Format>::GetType(Component) == SWR_TYPE_UINT)
        {
            int           iMax   = (1 << FormatTraits<Format>::GetBPC(Component)) - 1;
            int           iMin   = 0;
            simd16scalari vCompi = _simd16_castps_si(vComp);
            vCompi               = _simd16_max_epu32(vCompi, _simd16_set1_epi32(iMin));
            vCompi               = _simd16_min_epu32(vCompi, _simd16_set1_epi32(iMax));
            vComp                = _simd16_castsi_ps(vCompi);
        }
        else if (FormatTraits<Format>::GetType(Component) == SWR_TYPE_SINT)
        {
            int           iMax   = (1 << (FormatTraits<Format>::GetBPC(Component) - 1)) - 1;
            int           iMin   = -1 - iMax;
            simd16scalari vCompi = _simd16_castps_si(vComp);
            vCompi               = _simd16_max_epi32(vCompi, _simd16_set1_epi32(iMin));
            vCompi               = _simd16_min_epi32(vCompi, _simd16_set1_epi32(iMax));
            vComp                = _simd16_castsi_ps(vCompi);
        }
    }

    return vComp;
}

//////////////////////////////////////////////////////////////////////////
/// @brief Normalize the given component based on the requirements on the
///        Format template arg
/// @param vComp - SIMD vector of floats
/// @param Component - component
template <SWR_FORMAT Format>
INLINE simd16scalar SIMDCALL Normalize(simd16scalar const& vComp, uint32_t Component)
{
    simd16scalar r = vComp;
    if (FormatTraits<Format>::isNormalized(Component))
    {
        r = _simd16_mul_ps(r, _simd16_set1_ps(FormatTraits<Format>::fromFloat(Component)));
        r = _simd16_castsi_ps(_simd16_cvtps_epi32(r));
    }
    return r;
}

//////////////////////////////////////////////////////////////////////////
/// @brief Convert and store simdvector of pixels in SOA
///        RGBA32_FLOAT to SOA format
/// @param src - source data in SOA form
/// @param dst - output data in SOA form
template <SWR_FORMAT DstFormat>
INLINE void SIMDCALL StoreSOA(const simd16vector& src, uint8_t* pDst)
{
    // fast path for float32
    if ((FormatTraits<DstFormat>::GetType(0) == SWR_TYPE_FLOAT) &&
        (FormatTraits<DstFormat>::GetBPC(0) == 32))
    {
        for (uint32_t comp = 0; comp < FormatTraits<DstFormat>::numComps; ++comp)
        {
            simd16scalar vComp = src.v[FormatTraits<DstFormat>::swizzle(comp)];

            // Gamma-correct
            if (FormatTraits<DstFormat>::isSRGB)
            {
                if (comp < 3) // Input format is always RGBA32_FLOAT.
                {
                    vComp = FormatTraits<R32G32B32A32_FLOAT>::convertSrgb(comp, vComp);
                }
            }

            _simd16_store_ps(reinterpret_cast<float*>(pDst + comp * sizeof(simd16scalar)), vComp);
        }
        return;
    }

    auto lambda = [&](int comp) {
        simd16scalar vComp = src.v[FormatTraits<DstFormat>::swizzle(comp)];

        // Gamma-correct
        if (FormatTraits<DstFormat>::isSRGB)
        {
            if (comp < 3) // Input format is always RGBA32_FLOAT.
            {
                vComp = FormatTraits<R32G32B32A32_FLOAT>::convertSrgb(comp, vComp);
            }
        }

        // clamp
        vComp = Clamp<DstFormat>(vComp, comp);

        // normalize
        vComp = Normalize<DstFormat>(vComp, comp);

        // pack
        vComp = FormatTraits<DstFormat>::pack(comp, vComp);

        // store
        FormatTraits<DstFormat>::storeSOA(comp, pDst, vComp);

        pDst += (FormatTraits<DstFormat>::GetBPC(comp) * KNOB_SIMD16_WIDTH) / 8;
    };

    UnrollerL<0, FormatTraits<DstFormat>::numComps, 1>::step(lambda);
}

#endif
