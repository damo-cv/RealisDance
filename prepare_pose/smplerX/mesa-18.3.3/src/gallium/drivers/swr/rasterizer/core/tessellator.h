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
 * @file tessellator.h
 *
 * @brief Tessellator fixed function unit interface definition
 *
 ******************************************************************************/
#pragma once

/// Allocate and initialize a new tessellation context
HANDLE SWR_API
       TSInitCtx(SWR_TS_DOMAIN          tsDomain, ///< [IN] Tessellation domain (isoline, quad, triangle)
                 SWR_TS_PARTITIONING    tsPartitioning, ///< [IN] Tessellation partitioning algorithm
                 SWR_TS_OUTPUT_TOPOLOGY tsOutputTopology, ///< [IN] Tessellation output topology
                 void*                  pContextMem,      ///< [IN] Memory to use for the context
                 size_t& memSize); ///< [INOUT] In: Amount of memory in pContextMem. Out: Mem required

/// Destroy & de-allocate tessellation context
void SWR_API TSDestroyCtx(HANDLE tsCtx); ///< [IN] Tessellation context to be destroyed

struct SWR_TS_TESSELLATED_DATA
{
    uint32_t NumPrimitives;
    uint32_t NumDomainPoints;

    uint32_t* ppIndices[3];
    float*    pDomainPointsU;
    float*    pDomainPointsV;
    // For Tri: pDomainPointsW[i] = 1.0f - pDomainPointsU[i] - pDomainPointsV[i]
};

/// Perform Tessellation
void SWR_API
     TSTessellate(HANDLE                          tsCtx,         ///< [IN] Tessellation Context
                  const SWR_TESSELLATION_FACTORS& tsTessFactors, ///< [IN] Tessellation Factors
                  SWR_TS_TESSELLATED_DATA&        tsTessellatedData);   ///< [OUT] Tessellated Data


/// @TODO - Implement OSS tessellator

INLINE HANDLE SWR_API TSInitCtx(SWR_TS_DOMAIN          tsDomain,
                                SWR_TS_PARTITIONING    tsPartitioning,
                                SWR_TS_OUTPUT_TOPOLOGY tsOutputTopology,
                                void*                  pContextMem,
                                size_t&                memSize)
{
    SWR_NOT_IMPL;
    return NULL;
}

INLINE void SWR_API TSDestroyCtx(HANDLE tsCtx)
{
    SWR_NOT_IMPL;
}

INLINE void SWR_API TSTessellate(HANDLE                          tsCtx,
                                 const SWR_TESSELLATION_FACTORS& tsTessFactors,
                                 SWR_TS_TESSELLATED_DATA&        tsTessellatedData)
{
    SWR_NOT_IMPL;
}

