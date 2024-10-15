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
 * @file JitManager.h
 *
 * @brief JitManager contains the LLVM data structures used for JIT generation
 *
 * Notes:
 *
 ******************************************************************************/
#pragma once

#include "jit_pch.hpp"
#include "common/isa.hpp"


//////////////////////////////////////////////////////////////////////////
/// JitInstructionSet
/// @brief Subclass of InstructionSet that allows users to override
/// the reporting of support for certain ISA features.  This allows capping
/// the jitted code to a certain feature level, e.g. jit AVX level code on
/// a platform that supports AVX2.
//////////////////////////////////////////////////////////////////////////
class JitInstructionSet : public InstructionSet
{
public:
    JitInstructionSet(const char* requestedIsa) : isaRequest(requestedIsa)
    {
        std::transform(isaRequest.begin(), isaRequest.end(), isaRequest.begin(), ::tolower);

        if (isaRequest == "avx")
        {
            bForceAVX    = true;
            bForceAVX2   = false;
            bForceAVX512 = false;
        }
        else if (isaRequest == "avx2")
        {
            bForceAVX    = false;
            bForceAVX2   = true;
            bForceAVX512 = false;
        }
        else if (isaRequest == "avx512")
        {
            bForceAVX    = false;
            bForceAVX2   = false;
            bForceAVX512 = true;
        }
    };

    bool AVX2(void) { return bForceAVX ? 0 : InstructionSet::AVX2(); }
    bool AVX512F(void) { return (bForceAVX | bForceAVX2) ? 0 : InstructionSet::AVX512F(); }
    bool AVX512ER(void) { return (bForceAVX | bForceAVX2) ? 0 : InstructionSet::AVX512ER(); }
    bool BMI2(void) { return bForceAVX ? 0 : InstructionSet::BMI2(); }

private:
    bool        bForceAVX    = false;
    bool        bForceAVX2   = false;
    bool        bForceAVX512 = false;
    std::string isaRequest;
};

struct JitLLVMContext : llvm::LLVMContext
{
};

//////////////////////////////////////////////////////////////////////////
/// JitCache
//////////////////////////////////////////////////////////////////////////
struct JitManager; // Forward Decl
class JitCache : public llvm::ObjectCache
{
public:
    /// constructor
    JitCache();
    virtual ~JitCache() {}

    void Init(JitManager* pJitMgr, const llvm::StringRef& cpu, llvm::CodeGenOpt::Level level)
    {
        mCpu      = cpu.str();
        mpJitMgr  = pJitMgr;
        mOptLevel = level;
    }

    /// notifyObjectCompiled - Provides a pointer to compiled code for Module M.
    void notifyObjectCompiled(const llvm::Module* M, llvm::MemoryBufferRef Obj) override;

    /// Returns a pointer to a newly allocated MemoryBuffer that contains the
    /// object which corresponds with Module M, or 0 if an object is not
    /// available.
    std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module* M) override;

private:
    std::string                 mCpu;
    llvm::SmallString<MAX_PATH> mCacheDir;
    uint32_t                    mCurrentModuleCRC = 0;
    JitManager*                 mpJitMgr          = nullptr;
    llvm::CodeGenOpt::Level     mOptLevel         = llvm::CodeGenOpt::None;
};

//////////////////////////////////////////////////////////////////////////
/// JitManager
//////////////////////////////////////////////////////////////////////////
struct JitManager
{
    JitManager(uint32_t w, const char* arch, const char* core);
    ~JitManager(){};

    JitLLVMContext         mContext; ///< LLVM compiler
    llvm::IRBuilder<>      mBuilder; ///< LLVM IR Builder
    llvm::ExecutionEngine* mpExec;
    JitCache               mCache;

    // Need to be rebuilt after a JIT and before building new IR
    llvm::Module* mpCurrentModule;
    bool          mIsModuleFinalized;
    uint32_t      mJitNumber;

    uint32_t mVWidth;

    bool mUsingAVX512 = false;

    // fetch shader types
    llvm::FunctionType* mFetchShaderTy;

    JitInstructionSet mArch;

    // Debugging support
    std::unordered_map<llvm::StructType*, llvm::DIType*> mDebugStructMap;

    void SetupNewModule();

    void               DumpAsm(llvm::Function* pFunction, const char* fileName);
    static void        DumpToFile(llvm::Function* f, const char* fileName);
    static void        DumpToFile(llvm::Module* M, const char* fileName);
    static std::string GetOutputDir();

    // Debugging support methods
    llvm::DIType* GetDebugType(llvm::Type* pTy);
    llvm::DIType* GetDebugIntegerType(llvm::Type* pTy);
    llvm::DIType* GetDebugArrayType(llvm::Type* pTy);
    llvm::DIType* GetDebugVectorType(llvm::Type* pTy);
    llvm::DIType* GetDebugFunctionType(llvm::Type* pTy);

    llvm::DIType* GetDebugStructType(llvm::Type* pType)
    {
        llvm::StructType* pStructTy = llvm::cast<llvm::StructType>(pType);
        if (mDebugStructMap.find(pStructTy) == mDebugStructMap.end())
        {
            return nullptr;
        }
        return mDebugStructMap[pStructTy];
    }

    llvm::DIType*
    CreateDebugStructType(llvm::StructType*                                    pType,
                          const std::string&                                   name,
                          llvm::DIFile*                                        pFile,
                          uint32_t                                             lineNum,
                          const std::vector<std::pair<std::string, uint32_t>>& members);
};
