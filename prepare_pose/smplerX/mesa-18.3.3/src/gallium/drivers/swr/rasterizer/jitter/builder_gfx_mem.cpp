/****************************************************************************
 * Copyright (C) 2014-2018 Intel Corporation.   All Rights Reserved.
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
 * @file builder_gfx_mem.cpp
 *
 * @brief Definition of the gfx mem builder
 *
 * Notes:
 *
 ******************************************************************************/
#include "jit_pch.hpp"
#include "builder.h"
#include "common/rdtsc_buckets.h"
#include "builder_gfx_mem.h"


namespace SwrJit
{
    using namespace llvm;

    BuilderGfxMem::BuilderGfxMem(JitManager* pJitMgr) : Builder(pJitMgr)
    {
        mpTranslationFuncTy             = nullptr;
        mpfnTranslateGfxAddressForRead  = nullptr;
        mpfnTranslateGfxAddressForWrite = nullptr;
        mpParamSimDC                    = nullptr;

    }

    void BuilderGfxMem::NotifyPrivateContextSet()
    {
    }

    void BuilderGfxMem::AssertGFXMemoryParams(Value* ptr, Builder::JIT_MEM_CLIENT usage)
    {
        SWR_ASSERT(!(ptr->getType() == mInt64Ty && usage == MEM_CLIENT_INTERNAL),
                   "Internal memory should not be gfxptr_t.");
    }


    //////////////////////////////////////////////////////////////////////////
    /// @brief Generate a masked gather operation in LLVM IR.  If not
    /// supported on the underlying platform, emulate it with loads
    /// @param vSrc - SIMD wide value that will be loaded if mask is invalid
    /// @param pBase - Int8* base VB address pointer value
    /// @param vIndices - SIMD wide value of VB byte offsets
    /// @param vMask - SIMD wide mask that controls whether to access memory or the src values
    /// @param scale - value to scale indices by
    Value* BuilderGfxMem::GATHERPS(Value*         vSrc,
                                   Value*         pBase,
                                   Value*         vIndices,
                                   Value*         vMask,
                                   uint8_t        scale,
                                   JIT_MEM_CLIENT usage)
    {
       // address may be coming in as 64bit int now so get the pointer
        if (pBase->getType() == mInt64Ty)
        {
            pBase = INT_TO_PTR(pBase, PointerType::get(mInt8Ty, 0));
        }

        Value* vGather = Builder::GATHERPS(vSrc, pBase, vIndices, vMask, scale);
        return vGather;
    }

    //////////////////////////////////////////////////////////////////////////
    /// @brief Generate a masked gather operation in LLVM IR.  If not
    /// supported on the underlying platform, emulate it with loads
    /// @param vSrc - SIMD wide value that will be loaded if mask is invalid
    /// @param pBase - Int8* base VB address pointer value
    /// @param vIndices - SIMD wide value of VB byte offsets
    /// @param vMask - SIMD wide mask that controls whether to access memory or the src values
    /// @param scale - value to scale indices by
    Value* BuilderGfxMem::GATHERDD(Value*         vSrc,
                                   Value*         pBase,
                                   Value*         vIndices,
                                   Value*         vMask,
                                   uint8_t        scale,
                                   JIT_MEM_CLIENT usage)
    {

        // address may be coming in as 64bit int now so get the pointer
        if (pBase->getType() == mInt64Ty)
        {
            pBase = INT_TO_PTR(pBase, PointerType::get(mInt8Ty, 0));
        }

        Value* vGather = Builder::GATHERDD(vSrc, pBase, vIndices, vMask, scale);
        return vGather;
    }

    void BuilderGfxMem::SCATTERPS(
        Value* pDst, Value* vSrc, Value* vOffsets, Value* vMask, JIT_MEM_CLIENT usage)
    {

        // address may be coming in as 64bit int now so get the pointer
        if (pDst->getType() == mInt64Ty)
        {
            pDst = INT_TO_PTR(pDst, PointerType::get(mInt8Ty, 0));
        }

        Builder::SCATTERPS(pDst, vSrc, vOffsets, vMask, usage);
    }


    Value* BuilderGfxMem::OFFSET_TO_NEXT_COMPONENT(Value* base, Constant* offset)
    {
        return ADD(base, offset);
    }

    Value* BuilderGfxMem::GEP(Value* Ptr, Value* Idx, Type* Ty, const Twine& Name)
    {
        Ptr = TranslationHelper(Ptr, Ty);
        return Builder::GEP(Ptr, Idx, nullptr, Name);
    }

    Value* BuilderGfxMem::GEP(Type* Ty, Value* Ptr, Value* Idx, const Twine& Name)
    {
        Ptr = TranslationHelper(Ptr, Ty);
        return Builder::GEP(Ty, Ptr, Idx, Name);
    }

    Value* BuilderGfxMem::GEP(Value* Ptr, const std::initializer_list<Value*>& indexList, Type* Ty)
    {
        Ptr = TranslationHelper(Ptr, Ty);
        return Builder::GEP(Ptr, indexList);
    }

    Value*
    BuilderGfxMem::GEP(Value* Ptr, const std::initializer_list<uint32_t>& indexList, Type* Ty)
    {
        Ptr = TranslationHelper(Ptr, Ty);
        return Builder::GEP(Ptr, indexList);
    }

    Value* BuilderGfxMem::TranslationHelper(Value* Ptr, Type* Ty)
    {
        SWR_ASSERT(!(Ptr->getType() == mInt64Ty && Ty == nullptr),
                   "Access of GFX pointers must have non-null type specified.");


        // address may be coming in as 64bit int now so get the pointer
        if (Ptr->getType() == mInt64Ty)
        {
            Ptr = INT_TO_PTR(Ptr, Ty);
        }

        return Ptr;
    }

    LoadInst* BuilderGfxMem::LOAD(Value* Ptr, const char* Name, Type* Ty, JIT_MEM_CLIENT usage)
    {
        AssertGFXMemoryParams(Ptr, usage);

        Ptr = TranslationHelper(Ptr, Ty);
        return Builder::LOAD(Ptr, Name);
    }

    LoadInst* BuilderGfxMem::LOAD(Value* Ptr, const Twine& Name, Type* Ty, JIT_MEM_CLIENT usage)
    {
        AssertGFXMemoryParams(Ptr, usage);

        Ptr = TranslationHelper(Ptr, Ty);
        return Builder::LOAD(Ptr, Name);
    }


    LoadInst* BuilderGfxMem::LOAD(
        Value* Ptr, bool isVolatile, const Twine& Name, Type* Ty, JIT_MEM_CLIENT usage)
    {
        AssertGFXMemoryParams(Ptr, usage);

        Ptr = TranslationHelper(Ptr, Ty);
        return Builder::LOAD(Ptr, isVolatile, Name);
    }

    LoadInst* BuilderGfxMem::LOAD(Value*                                 BasePtr,
                                  const std::initializer_list<uint32_t>& offset,
                                  const llvm::Twine&                     name,
                                  Type*                                  Ty,
                                  JIT_MEM_CLIENT                         usage)
    {
        AssertGFXMemoryParams(BasePtr, usage);

        bool bNeedTranslation = false;
        if (BasePtr->getType() == mInt64Ty)
        {
            SWR_ASSERT(Ty);
            BasePtr          = INT_TO_PTR(BasePtr, Ty, name);
            bNeedTranslation = true;
        }
        std::vector<Value*> valIndices;
        for (auto i : offset)
        {
            valIndices.push_back(C(i));
        }
        BasePtr = Builder::GEPA(BasePtr, valIndices, name);
        if (bNeedTranslation)
        {
            BasePtr = PTR_TO_INT(BasePtr, mInt64Ty, name);
        }

        return LOAD(BasePtr, name, Ty, usage);
    }


    CallInst* BuilderGfxMem::MASKED_LOAD(Value*         Ptr,
                                         unsigned       Align,
                                         Value*         Mask,
                                         Value*         PassThru,
                                         const Twine&   Name,
                                         Type*          Ty,
                                         JIT_MEM_CLIENT usage)
    {
        AssertGFXMemoryParams(Ptr, usage);

        Ptr = TranslationHelper(Ptr, Ty);
        return Builder::MASKED_LOAD(Ptr, Align, Mask, PassThru, Name, Ty, usage);
    }

    Value* BuilderGfxMem::TranslateGfxAddressForRead(Value*       xpGfxAddress,
                                                     Type*        PtrTy,
                                                     const Twine& Name,
                                                     JIT_MEM_CLIENT /* usage */)
    {
        if (PtrTy == nullptr)
        {
            PtrTy = mInt8PtrTy;
        }
        return INT_TO_PTR(xpGfxAddress, PtrTy, Name);
    }

    Value* BuilderGfxMem::TranslateGfxAddressForWrite(Value*       xpGfxAddress,
                                                      Type*        PtrTy,
                                                      const Twine& Name,
                                                      JIT_MEM_CLIENT /* usage */)
    {
        if (PtrTy == nullptr)
        {
            PtrTy = mInt8PtrTy;
        }
        return INT_TO_PTR(xpGfxAddress, PtrTy, Name);
    }

} // namespace SwrJit
