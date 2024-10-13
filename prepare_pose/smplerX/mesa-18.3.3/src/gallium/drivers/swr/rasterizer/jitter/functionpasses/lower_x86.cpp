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
 * @file lower_x86.cpp
 *
 * @brief llvm pass to lower meta code to x86
 *
 * Notes:
 *
 ******************************************************************************/

#include "jit_pch.hpp"
#include "passes.h"
#include "JitManager.h"

#include <unordered_map>

namespace llvm
{
    // foward declare the initializer
    void initializeLowerX86Pass(PassRegistry&);
} // namespace llvm

namespace SwrJit
{
    using namespace llvm;

    enum TargetArch
    {
        AVX    = 0,
        AVX2   = 1,
        AVX512 = 2
    };

    enum TargetWidth
    {
        W256       = 0,
        W512       = 1,
        NUM_WIDTHS = 2
    };

    struct LowerX86;

    typedef std::function<Instruction*(LowerX86*, TargetArch, TargetWidth, CallInst*)> EmuFunc;

    struct X86Intrinsic
    {
        Intrinsic::ID intrin[NUM_WIDTHS];
        EmuFunc       emuFunc;
    };

    // Map of intrinsics that haven't been moved to the new mechanism yet. If used, these get the
    // previous behavior of mapping directly to avx/avx2 intrinsics.
    static std::map<std::string, Intrinsic::ID> intrinsicMap = {
        {"meta.intrinsic.BEXTR_32", Intrinsic::x86_bmi_bextr_32},
        {"meta.intrinsic.VPSHUFB", Intrinsic::x86_avx2_pshuf_b},
        {"meta.intrinsic.VCVTPS2PH", Intrinsic::x86_vcvtps2ph_256},
        {"meta.intrinsic.VPTESTC", Intrinsic::x86_avx_ptestc_256},
        {"meta.intrinsic.VPTESTZ", Intrinsic::x86_avx_ptestz_256},
        {"meta.intrinsic.VPHADDD", Intrinsic::x86_avx2_phadd_d},
        {"meta.intrinsic.PDEP32", Intrinsic::x86_bmi_pdep_32},
        {"meta.intrinsic.RDTSC", Intrinsic::x86_rdtsc},
    };

    // Forward decls
    Instruction* NO_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst);
    Instruction*
    VPERM_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst);
    Instruction*
    VGATHER_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst);
    Instruction*
    VROUND_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst);
    Instruction*
    VHSUB_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst);
    Instruction*
    VCONVERT_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst);

    Instruction* DOUBLE_EMU(LowerX86*     pThis,
                            TargetArch    arch,
                            TargetWidth   width,
                            CallInst*     pCallInst,
                            Intrinsic::ID intrin);

    static Intrinsic::ID DOUBLE = (Intrinsic::ID)-1;

    static std::map<std::string, X86Intrinsic> intrinsicMap2[] = {
        //                              256 wide                                    512 wide
        {
            // AVX
            {"meta.intrinsic.VRCPPS", {{Intrinsic::x86_avx_rcp_ps_256, DOUBLE}, NO_EMU}},
            {"meta.intrinsic.VPERMPS",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VPERM_EMU}},
            {"meta.intrinsic.VPERMD",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VPERM_EMU}},
            {"meta.intrinsic.VGATHERPD",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VGATHER_EMU}},
            {"meta.intrinsic.VGATHERPS",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VGATHER_EMU}},
            {"meta.intrinsic.VGATHERDD",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VGATHER_EMU}},
            {"meta.intrinsic.VCVTPD2PS",
             {{Intrinsic::x86_avx_cvt_pd2_ps_256, Intrinsic::not_intrinsic}, NO_EMU}},
            {"meta.intrinsic.VCVTPH2PS",
             {{Intrinsic::x86_vcvtph2ps_256, Intrinsic::not_intrinsic}, NO_EMU}},
            {"meta.intrinsic.VROUND", {{Intrinsic::x86_avx_round_ps_256, DOUBLE}, NO_EMU}},
            {"meta.intrinsic.VHSUBPS", {{Intrinsic::x86_avx_hsub_ps_256, DOUBLE}, NO_EMU}},
        },
        {
            // AVX2
            {"meta.intrinsic.VRCPPS", {{Intrinsic::x86_avx_rcp_ps_256, DOUBLE}, NO_EMU}},
            {"meta.intrinsic.VPERMPS",
             {{Intrinsic::x86_avx2_permps, Intrinsic::not_intrinsic}, VPERM_EMU}},
            {"meta.intrinsic.VPERMD",
             {{Intrinsic::x86_avx2_permd, Intrinsic::not_intrinsic}, VPERM_EMU}},
            {"meta.intrinsic.VGATHERPD",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VGATHER_EMU}},
            {"meta.intrinsic.VGATHERPS",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VGATHER_EMU}},
            {"meta.intrinsic.VGATHERDD",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VGATHER_EMU}},
            {"meta.intrinsic.VCVTPD2PS", {{Intrinsic::x86_avx_cvt_pd2_ps_256, DOUBLE}, NO_EMU}},
            {"meta.intrinsic.VCVTPH2PS",
             {{Intrinsic::x86_vcvtph2ps_256, Intrinsic::not_intrinsic}, NO_EMU}},
            {"meta.intrinsic.VROUND", {{Intrinsic::x86_avx_round_ps_256, DOUBLE}, NO_EMU}},
            {"meta.intrinsic.VHSUBPS", {{Intrinsic::x86_avx_hsub_ps_256, DOUBLE}, NO_EMU}},
        },
        {
            // AVX512
            {"meta.intrinsic.VRCPPS",
             {{Intrinsic::x86_avx512_rcp14_ps_256, Intrinsic::x86_avx512_rcp14_ps_512}, NO_EMU}},
#if LLVM_VERSION_MAJOR < 7
            {"meta.intrinsic.VPERMPS",
             {{Intrinsic::x86_avx512_mask_permvar_sf_256,
               Intrinsic::x86_avx512_mask_permvar_sf_512},
              NO_EMU}},
            {"meta.intrinsic.VPERMD",
             {{Intrinsic::x86_avx512_mask_permvar_si_256,
               Intrinsic::x86_avx512_mask_permvar_si_512},
              NO_EMU}},
#else
            {"meta.intrinsic.VPERMPS",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VPERM_EMU}},
            {"meta.intrinsic.VPERMD",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VPERM_EMU}},
#endif
            {"meta.intrinsic.VGATHERPD",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VGATHER_EMU}},
            {"meta.intrinsic.VGATHERPS",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VGATHER_EMU}},
            {"meta.intrinsic.VGATHERDD",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VGATHER_EMU}},
#if LLVM_VERSION_MAJOR < 7
            {"meta.intrinsic.VCVTPD2PS",
             {{Intrinsic::x86_avx512_mask_cvtpd2ps_256, Intrinsic::x86_avx512_mask_cvtpd2ps_512},
              NO_EMU}},
#else
            {"meta.intrinsic.VCVTPD2PS",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VCONVERT_EMU}},
#endif
            {"meta.intrinsic.VCVTPH2PS",
             {{Intrinsic::x86_avx512_mask_vcvtph2ps_256, Intrinsic::x86_avx512_mask_vcvtph2ps_512},
              NO_EMU}},
            {"meta.intrinsic.VROUND",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VROUND_EMU}},
            {"meta.intrinsic.VHSUBPS",
             {{Intrinsic::not_intrinsic, Intrinsic::not_intrinsic}, VHSUB_EMU}},
        }};

    struct LowerX86 : public FunctionPass
    {
        LowerX86(Builder* b = nullptr) : FunctionPass(ID), B(b)
        {
            initializeLowerX86Pass(*PassRegistry::getPassRegistry());

            // Determine target arch
            if (JM()->mArch.AVX512F())
            {
                mTarget = AVX512;
            }
            else if (JM()->mArch.AVX2())
            {
                mTarget = AVX2;
            }
            else if (JM()->mArch.AVX())
            {
                mTarget = AVX;
            }
            else
            {
                SWR_ASSERT(false, "Unsupported AVX architecture.");
                mTarget = AVX;
            }
        }

        // Try to decipher the vector type of the instruction. This does not work properly
        // across all intrinsics, and will have to be rethought. Probably need something
        // similar to llvm's getDeclaration() utility to map a set of inputs to a specific typed
        // intrinsic.
        void GetRequestedWidthAndType(CallInst*       pCallInst,
                                      const StringRef intrinName,
                                      TargetWidth*    pWidth,
                                      Type**          pTy)
        {
            Type* pVecTy = pCallInst->getType();

            // Check for intrinsic specific types
            // VCVTPD2PS type comes from src, not dst
            if (intrinName.equals("meta.intrinsic.VCVTPD2PS"))
            {
                pVecTy = pCallInst->getOperand(0)->getType();
            }

            if (!pVecTy->isVectorTy())
            {
                for (auto& op : pCallInst->arg_operands())
                {
                    if (op.get()->getType()->isVectorTy())
                    {
                        pVecTy = op.get()->getType();
                        break;
                    }
                }
            }
            SWR_ASSERT(pVecTy->isVectorTy(), "Couldn't determine vector size");

            uint32_t width = cast<VectorType>(pVecTy)->getBitWidth();
            switch (width)
            {
            case 256:
                *pWidth = W256;
                break;
            case 512:
                *pWidth = W512;
                break;
            default:
                SWR_ASSERT(false, "Unhandled vector width %d", width);
                *pWidth = W256;
            }

            *pTy = pVecTy->getScalarType();
        }

        Value* GetZeroVec(TargetWidth width, Type* pTy)
        {
            uint32_t numElem = 0;
            switch (width)
            {
            case W256:
                numElem = 8;
                break;
            case W512:
                numElem = 16;
                break;
            default:
                SWR_ASSERT(false, "Unhandled vector width type %d\n", width);
            }

            return ConstantVector::getNullValue(VectorType::get(pTy, numElem));
        }

        Value* GetMask(TargetWidth width)
        {
            Value* mask;
            switch (width)
            {
            case W256:
                mask = B->C((uint8_t)-1);
                break;
            case W512:
                mask = B->C((uint16_t)-1);
                break;
            default:
                SWR_ASSERT(false, "Unhandled vector width type %d\n", width);
            }
            return mask;
        }

        // Convert <N x i1> mask to <N x i32> x86 mask
        Value* VectorMask(Value* vi1Mask)
        {
            uint32_t numElem = vi1Mask->getType()->getVectorNumElements();
            return B->S_EXT(vi1Mask, VectorType::get(B->mInt32Ty, numElem));
        }

        Instruction* ProcessIntrinsicAdvanced(CallInst* pCallInst)
        {
            Function*   pFunc     = pCallInst->getCalledFunction();
            auto&       intrinsic = intrinsicMap2[mTarget][pFunc->getName()];
            TargetWidth vecWidth;
            Type*       pElemTy;
            GetRequestedWidthAndType(pCallInst, pFunc->getName(), &vecWidth, &pElemTy);

            // Check if there is a native intrinsic for this instruction
            Intrinsic::ID id = intrinsic.intrin[vecWidth];
            if (id == DOUBLE)
            {
                // Double pump the next smaller SIMD intrinsic
                SWR_ASSERT(vecWidth != 0, "Cannot double pump smallest SIMD width.");
                Intrinsic::ID id2 = intrinsic.intrin[vecWidth - 1];
                SWR_ASSERT(id2 != Intrinsic::not_intrinsic,
                           "Cannot find intrinsic to double pump.");
                return DOUBLE_EMU(this, mTarget, vecWidth, pCallInst, id2);
            }
            else if (id != Intrinsic::not_intrinsic)
            {
                Function* pIntrin = Intrinsic::getDeclaration(B->JM()->mpCurrentModule, id);
                SmallVector<Value*, 8> args;
                for (auto& arg : pCallInst->arg_operands())
                {
                    args.push_back(arg.get());
                }

                // If AVX512, all instructions add a src operand and mask. We'll pass in 0 src and
                // full mask for now Assuming the intrinsics are consistent and place the src
                // operand and mask last in the argument list.
                if (mTarget == AVX512)
                {
                    if (pFunc->getName().equals("meta.intrinsic.VCVTPD2PS"))
                    {
                        args.push_back(GetZeroVec(W256, pCallInst->getType()->getScalarType()));
                        args.push_back(GetMask(W256));
                        // for AVX512 VCVTPD2PS, we also have to add rounding mode
                        args.push_back(B->C(_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                    }
                    else
                    {
                        args.push_back(GetZeroVec(vecWidth, pElemTy));
                        args.push_back(GetMask(vecWidth));
                    }
                }

                return B->CALLA(pIntrin, args);
            }
            else
            {
                // No native intrinsic, call emulation function
                return intrinsic.emuFunc(this, mTarget, vecWidth, pCallInst);
            }

            SWR_ASSERT(false);
            return nullptr;
        }

        Instruction* ProcessIntrinsic(CallInst* pCallInst)
        {
            Function* pFunc = pCallInst->getCalledFunction();

            // Forward to the advanced support if found
            if (intrinsicMap2[mTarget].find(pFunc->getName()) != intrinsicMap2[mTarget].end())
            {
                return ProcessIntrinsicAdvanced(pCallInst);
            }

            SWR_ASSERT(intrinsicMap.find(pFunc->getName()) != intrinsicMap.end(),
                       "Unimplemented intrinsic %s.",
                       pFunc->getName());

            Intrinsic::ID x86Intrinsic = intrinsicMap[pFunc->getName()];
            Function*     pX86IntrinFunc =
                Intrinsic::getDeclaration(B->JM()->mpCurrentModule, x86Intrinsic);

            SmallVector<Value*, 8> args;
            for (auto& arg : pCallInst->arg_operands())
            {
                args.push_back(arg.get());
            }
            return B->CALLA(pX86IntrinFunc, args);
        }

        //////////////////////////////////////////////////////////////////////////
        /// @brief LLVM funtion pass run method.
        /// @param f- The function we're working on with this pass.
        virtual bool runOnFunction(Function& F)
        {
            std::vector<Instruction*> toRemove;

            for (auto& BB : F.getBasicBlockList())
            {
                for (auto& I : BB.getInstList())
                {
                    if (CallInst* pCallInst = dyn_cast<CallInst>(&I))
                    {
                        Function* pFunc = pCallInst->getCalledFunction();
                        if (pFunc)
                        {
                            if (pFunc->getName().startswith("meta.intrinsic"))
                            {
                                B->IRB()->SetInsertPoint(&I);
                                Instruction* pReplace = ProcessIntrinsic(pCallInst);
                                SWR_ASSERT(pReplace);
                                toRemove.push_back(pCallInst);
                                pCallInst->replaceAllUsesWith(pReplace);
                            }
                        }
                    }
                }
            }

            for (auto* pInst : toRemove)
            {
                pInst->eraseFromParent();
            }

            JitManager::DumpToFile(&F, "lowerx86");

            return true;
        }

        virtual void getAnalysisUsage(AnalysisUsage& AU) const {}

        JitManager* JM() { return B->JM(); }

        Builder* B;

        TargetArch mTarget;

        static char ID; ///< Needed by LLVM to generate ID for FunctionPass.
    };

    char LowerX86::ID = 0; // LLVM uses address of ID as the actual ID.

    FunctionPass* createLowerX86Pass(Builder* b) { return new LowerX86(b); }

    Instruction* NO_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst)
    {
        SWR_ASSERT(false, "Unimplemented intrinsic emulation.");
        return nullptr;
    }

    Instruction* VPERM_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst)
    {
        // Only need vperm emulation for AVX
        SWR_ASSERT(arch == AVX);

        Builder* B         = pThis->B;
        auto     v32A      = pCallInst->getArgOperand(0);
        auto     vi32Index = pCallInst->getArgOperand(1);

        Value* v32Result;
        if (isa<Constant>(vi32Index))
        {
            // Can use llvm shuffle vector directly with constant shuffle indices
            v32Result = B->VSHUFFLE(v32A, v32A, vi32Index);
        }
        else
        {
            v32Result = UndefValue::get(v32A->getType());
            for (uint32_t l = 0; l < v32A->getType()->getVectorNumElements(); ++l)
            {
                auto i32Index = B->VEXTRACT(vi32Index, B->C(l));
                auto val      = B->VEXTRACT(v32A, i32Index);
                v32Result     = B->VINSERT(v32Result, val, B->C(l));
            }
        }
        return cast<Instruction>(v32Result);
    }

    Instruction*
    VGATHER_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst)
    {
        Builder* B           = pThis->B;
        auto     vSrc        = pCallInst->getArgOperand(0);
        auto     pBase       = pCallInst->getArgOperand(1);
        auto     vi32Indices = pCallInst->getArgOperand(2);
        auto     vi1Mask     = pCallInst->getArgOperand(3);
        auto     i8Scale     = pCallInst->getArgOperand(4);

        pBase             = B->POINTER_CAST(pBase, PointerType::get(B->mInt8Ty, 0));
        uint32_t numElem  = vSrc->getType()->getVectorNumElements();
        auto     i32Scale = B->Z_EXT(i8Scale, B->mInt32Ty);
        auto     srcTy    = vSrc->getType()->getVectorElementType();
        Value*   v32Gather;
        if (arch == AVX)
        {
            // Full emulation for AVX
            // Store source on stack to provide a valid address to load from inactive lanes
            auto pStack = B->STACKSAVE();
            auto pTmp   = B->ALLOCA(vSrc->getType());
            B->STORE(vSrc, pTmp);

            v32Gather        = UndefValue::get(vSrc->getType());
            auto vi32Scale   = ConstantVector::getSplat(numElem, cast<ConstantInt>(i32Scale));
            auto vi32Offsets = B->MUL(vi32Indices, vi32Scale);

            for (uint32_t i = 0; i < numElem; ++i)
            {
                auto i32Offset          = B->VEXTRACT(vi32Offsets, B->C(i));
                auto pLoadAddress       = B->GEP(pBase, i32Offset);
                pLoadAddress            = B->BITCAST(pLoadAddress, PointerType::get(srcTy, 0));
                auto pMaskedLoadAddress = B->GEP(pTmp, {0, i});
                auto i1Mask             = B->VEXTRACT(vi1Mask, B->C(i));
                auto pValidAddress      = B->SELECT(i1Mask, pLoadAddress, pMaskedLoadAddress);
                auto val                = B->LOAD(pValidAddress);
                v32Gather               = B->VINSERT(v32Gather, val, B->C(i));
            }

            B->STACKRESTORE(pStack);
        }
        else if (arch == AVX2 || (arch == AVX512 && width == W256))
        {
            Function* pX86IntrinFunc;
            if (srcTy == B->mFP32Ty)
            {
                pX86IntrinFunc = Intrinsic::getDeclaration(B->JM()->mpCurrentModule,
                                                           Intrinsic::x86_avx2_gather_d_ps_256);
            }
            else if (srcTy == B->mInt32Ty)
            {
                pX86IntrinFunc = Intrinsic::getDeclaration(B->JM()->mpCurrentModule,
                                                           Intrinsic::x86_avx2_gather_d_d_256);
            }
            else if (srcTy == B->mDoubleTy)
            {
                pX86IntrinFunc = Intrinsic::getDeclaration(B->JM()->mpCurrentModule,
                                                           Intrinsic::x86_avx2_gather_d_q_256);
            }
            else
            {
                SWR_ASSERT(false, "Unsupported vector element type for gather.");
            }

            if (width == W256)
            {
                auto v32Mask = B->BITCAST(pThis->VectorMask(vi1Mask), vSrc->getType());
                v32Gather = B->CALL(pX86IntrinFunc, {vSrc, pBase, vi32Indices, v32Mask, i8Scale});
            }
            else if (width == W512)
            {
                // Double pump 4-wide for 64bit elements
                if (vSrc->getType()->getVectorElementType() == B->mDoubleTy)
                {
                    auto v64Mask = pThis->VectorMask(vi1Mask);
                    v64Mask      = B->S_EXT(
                        v64Mask,
                        VectorType::get(B->mInt64Ty, v64Mask->getType()->getVectorNumElements()));
                    v64Mask = B->BITCAST(v64Mask, vSrc->getType());

                    Value* src0 = B->VSHUFFLE(vSrc, vSrc, B->C({0, 1, 2, 3}));
                    Value* src1 = B->VSHUFFLE(vSrc, vSrc, B->C({4, 5, 6, 7}));

                    Value* indices0 = B->VSHUFFLE(vi32Indices, vi32Indices, B->C({0, 1, 2, 3}));
                    Value* indices1 = B->VSHUFFLE(vi32Indices, vi32Indices, B->C({4, 5, 6, 7}));

                    Value* mask0 = B->VSHUFFLE(v64Mask, v64Mask, B->C({0, 1, 2, 3}));
                    Value* mask1 = B->VSHUFFLE(v64Mask, v64Mask, B->C({4, 5, 6, 7}));

                    src0 = B->BITCAST(
                        src0,
                        VectorType::get(B->mInt64Ty, src0->getType()->getVectorNumElements()));
                    mask0 = B->BITCAST(
                        mask0,
                        VectorType::get(B->mInt64Ty, mask0->getType()->getVectorNumElements()));
                    Value* gather0 =
                        B->CALL(pX86IntrinFunc, {src0, pBase, indices0, mask0, i8Scale});
                    src1 = B->BITCAST(
                        src1,
                        VectorType::get(B->mInt64Ty, src1->getType()->getVectorNumElements()));
                    mask1 = B->BITCAST(
                        mask1,
                        VectorType::get(B->mInt64Ty, mask1->getType()->getVectorNumElements()));
                    Value* gather1 =
                        B->CALL(pX86IntrinFunc, {src1, pBase, indices1, mask1, i8Scale});

                    v32Gather = B->VSHUFFLE(gather0, gather1, B->C({0, 1, 2, 3, 4, 5, 6, 7}));
                    v32Gather = B->BITCAST(v32Gather, vSrc->getType());
                }
                else
                {
                    // Double pump 8-wide for 32bit elements
                    auto v32Mask = pThis->VectorMask(vi1Mask);
                    v32Mask      = B->BITCAST(v32Mask, vSrc->getType());
                    Value* src0  = B->EXTRACT_16(vSrc, 0);
                    Value* src1  = B->EXTRACT_16(vSrc, 1);

                    Value* indices0 = B->EXTRACT_16(vi32Indices, 0);
                    Value* indices1 = B->EXTRACT_16(vi32Indices, 1);

                    Value* mask0 = B->EXTRACT_16(v32Mask, 0);
                    Value* mask1 = B->EXTRACT_16(v32Mask, 1);

                    Value* gather0 =
                        B->CALL(pX86IntrinFunc, {src0, pBase, indices0, mask0, i8Scale});
                    Value* gather1 =
                        B->CALL(pX86IntrinFunc, {src1, pBase, indices1, mask1, i8Scale});

                    v32Gather = B->JOIN_16(gather0, gather1);
                }
            }
        }
        else if (arch == AVX512)
        {
            Value*    iMask;
            Function* pX86IntrinFunc;
            if (srcTy == B->mFP32Ty)
            {
                pX86IntrinFunc = Intrinsic::getDeclaration(B->JM()->mpCurrentModule,
                                                           Intrinsic::x86_avx512_gather_dps_512);
                iMask          = B->BITCAST(vi1Mask, B->mInt16Ty);
            }
            else if (srcTy == B->mInt32Ty)
            {
                pX86IntrinFunc = Intrinsic::getDeclaration(B->JM()->mpCurrentModule,
                                                           Intrinsic::x86_avx512_gather_dpi_512);
                iMask          = B->BITCAST(vi1Mask, B->mInt16Ty);
            }
            else if (srcTy == B->mDoubleTy)
            {
                pX86IntrinFunc = Intrinsic::getDeclaration(B->JM()->mpCurrentModule,
                                                           Intrinsic::x86_avx512_gather_dpd_512);
                iMask          = B->BITCAST(vi1Mask, B->mInt8Ty);
            }
            else
            {
                SWR_ASSERT(false, "Unsupported vector element type for gather.");
            }

            auto i32Scale = B->Z_EXT(i8Scale, B->mInt32Ty);
            v32Gather     = B->CALL(pX86IntrinFunc, {vSrc, pBase, vi32Indices, iMask, i32Scale});
        }

        return cast<Instruction>(v32Gather);
    }

    // No support for vroundps in avx512 (it is available in kncni), so emulate with avx
    // instructions
    Instruction*
    VROUND_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst)
    {
        SWR_ASSERT(arch == AVX512);

        auto B       = pThis->B;
        auto vf32Src = pCallInst->getOperand(0);
        auto i8Round = pCallInst->getOperand(1);
        auto pfnFunc =
            Intrinsic::getDeclaration(B->JM()->mpCurrentModule, Intrinsic::x86_avx_round_ps_256);

        if (width == W256)
        {
            return cast<Instruction>(B->CALL2(pfnFunc, vf32Src, i8Round));
        }
        else if (width == W512)
        {
            auto v8f32SrcLo = B->EXTRACT_16(vf32Src, 0);
            auto v8f32SrcHi = B->EXTRACT_16(vf32Src, 1);

            auto v8f32ResLo = B->CALL2(pfnFunc, v8f32SrcLo, i8Round);
            auto v8f32ResHi = B->CALL2(pfnFunc, v8f32SrcHi, i8Round);

            return cast<Instruction>(B->JOIN_16(v8f32ResLo, v8f32ResHi));
        }
        else
        {
            SWR_ASSERT(false, "Unimplemented vector width.");
        }

        return nullptr;
    }

    Instruction*
    VCONVERT_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst)
    {
        SWR_ASSERT(arch == AVX512);

        auto B       = pThis->B;
        auto vf32Src = pCallInst->getOperand(0);

        if (width == W256)
        {
            auto vf32SrcRound = Intrinsic::getDeclaration(B->JM()->mpCurrentModule,
                                                          Intrinsic::x86_avx_round_ps_256);
            return cast<Instruction>(B->FP_TRUNC(vf32SrcRound, B->mFP32Ty));
        }
        else if (width == W512)
        {
            // 512 can use intrinsic
            auto pfnFunc = Intrinsic::getDeclaration(B->JM()->mpCurrentModule,
                                                     Intrinsic::x86_avx512_mask_cvtpd2ps_512);
            return cast<Instruction>(B->CALL(pfnFunc, vf32Src));
        }
        else
        {
            SWR_ASSERT(false, "Unimplemented vector width.");
        }

        return nullptr;
    }

    // No support for hsub in AVX512
    Instruction* VHSUB_EMU(LowerX86* pThis, TargetArch arch, TargetWidth width, CallInst* pCallInst)
    {
        SWR_ASSERT(arch == AVX512);

        auto B    = pThis->B;
        auto src0 = pCallInst->getOperand(0);
        auto src1 = pCallInst->getOperand(1);

        // 256b hsub can just use avx intrinsic
        if (width == W256)
        {
            auto pX86IntrinFunc =
                Intrinsic::getDeclaration(B->JM()->mpCurrentModule, Intrinsic::x86_avx_hsub_ps_256);
            return cast<Instruction>(B->CALL2(pX86IntrinFunc, src0, src1));
        }
        else if (width == W512)
        {
            // 512b hsub can be accomplished with shuf/sub combo
            auto minuend    = B->VSHUFFLE(src0, src1, B->C({0, 2, 8, 10, 4, 6, 12, 14}));
            auto subtrahend = B->VSHUFFLE(src0, src1, B->C({1, 3, 9, 11, 5, 7, 13, 15}));
            return cast<Instruction>(B->SUB(minuend, subtrahend));
        }
        else
        {
            SWR_ASSERT(false, "Unimplemented vector width.");
            return nullptr;
        }
    }

    // Double pump input using Intrin template arg. This blindly extracts lower and upper 256 from
    // each vector argument and calls the 256 wide intrinsic, then merges the results to 512 wide
    Instruction* DOUBLE_EMU(LowerX86*     pThis,
                            TargetArch    arch,
                            TargetWidth   width,
                            CallInst*     pCallInst,
                            Intrinsic::ID intrin)
    {
        auto B = pThis->B;
        SWR_ASSERT(width == W512);
        Value*    result[2];
        Function* pX86IntrinFunc = Intrinsic::getDeclaration(B->JM()->mpCurrentModule, intrin);
        for (uint32_t i = 0; i < 2; ++i)
        {
            SmallVector<Value*, 8> args;
            for (auto& arg : pCallInst->arg_operands())
            {
                auto argType = arg.get()->getType();
                if (argType->isVectorTy())
                {
                    uint32_t vecWidth  = argType->getVectorNumElements();
                    Value*   lanes     = B->CInc<int>(i * vecWidth / 2, vecWidth / 2);
                    Value*   argToPush = B->VSHUFFLE(
                        arg.get(), B->VUNDEF(argType->getVectorElementType(), vecWidth), lanes);
                    args.push_back(argToPush);
                }
                else
                {
                    args.push_back(arg.get());
                }
            }
            result[i] = B->CALLA(pX86IntrinFunc, args);
        }
        uint32_t vecWidth;
        if (result[0]->getType()->isVectorTy())
        {
            assert(result[1]->getType()->isVectorTy());
            vecWidth = result[0]->getType()->getVectorNumElements() +
                       result[1]->getType()->getVectorNumElements();
        }
        else
        {
            vecWidth = 2;
        }
        Value* lanes = B->CInc<int>(0, vecWidth);
        return cast<Instruction>(B->VSHUFFLE(result[0], result[1], lanes));
    }

} // namespace SwrJit

using namespace SwrJit;

INITIALIZE_PASS_BEGIN(LowerX86, "LowerX86", "LowerX86", false, false)
INITIALIZE_PASS_END(LowerX86, "LowerX86", "LowerX86", false, false)
