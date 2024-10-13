//
// Copyright 2016 Francisco Jerez
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//

///
/// \file
/// Some thin wrappers around the Clang/LLVM API used to preserve
/// compatibility with older API versions while keeping the ifdef clutter low
/// in the rest of the clover::llvm subtree.  In case of an API break please
/// consider whether it's possible to preserve backwards compatibility by
/// introducing a new one-liner inline function or typedef here under the
/// compat namespace in order to keep the running code free from preprocessor
/// conditionals.
///

#ifndef CLOVER_LLVM_COMPAT_HPP
#define CLOVER_LLVM_COMPAT_HPP

#include "util/algorithm.hpp"

#if HAVE_LLVM < 0x0400
#include <llvm/Bitcode/ReaderWriter.h>
#else
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#endif

#include <llvm/IR/LLVMContext.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Target/TargetMachine.h>
#if HAVE_LLVM >= 0x0400
#include <llvm/Support/Error.h>
#else
#include <llvm/Support/ErrorOr.h>
#endif

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Analysis/TargetLibraryInfo.h>

#include <clang/Basic/TargetInfo.h>
#include <clang/Frontend/CompilerInstance.h>

#if HAVE_LLVM >= 0x0800
#include <clang/Basic/CodeGenOptions.h>
#else
#include <clang/Frontend/CodeGenOptions.h>
#endif

namespace clover {
   namespace llvm {
      namespace compat {
         template<typename T, typename AS>
         unsigned target_address_space(const T &target, const AS lang_as) {
            const auto &map = target.getAddressSpaceMap();
#if HAVE_LLVM >= 0x0500
            return map[static_cast<unsigned>(lang_as)];
#else
            return map[lang_as - clang::LangAS::Offset];
#endif
         }

#if HAVE_LLVM >= 0x0500
         const clang::InputKind ik_opencl = clang::InputKind::OpenCL;
         const clang::LangStandard::Kind lang_opencl10 = clang::LangStandard::lang_opencl10;
#else
         const clang::InputKind ik_opencl = clang::IK_OpenCL;
         const clang::LangStandard::Kind lang_opencl10 = clang::LangStandard::lang_opencl;
#endif

         inline void
         add_link_bitcode_file(clang::CodeGenOptions &opts,
                               const std::string &path) {
#if HAVE_LLVM >= 0x0500
            clang::CodeGenOptions::BitcodeFileToLink F;

            F.Filename = path;
            F.PropagateAttrs = true;
            F.LinkFlags = ::llvm::Linker::Flags::None;
            opts.LinkBitcodeFiles.emplace_back(F);
#else
            opts.LinkBitcodeFiles.emplace_back(::llvm::Linker::Flags::None, path);
#endif
         }

#if HAVE_LLVM >= 0x0600
         const auto default_code_model = ::llvm::None;
#else
         const auto default_code_model = ::llvm::CodeModel::Default;
#endif

         template<typename M, typename F> void
         handle_module_error(M &mod, const F &f) {
#if HAVE_LLVM >= 0x0400
            if (::llvm::Error err = mod.takeError())
               ::llvm::handleAllErrors(std::move(err), [&](::llvm::ErrorInfoBase &eib) {
                     f(eib.message());
                  });
#else
            if (!mod)
               f(mod.getError().message());
#endif
         }

        template<typename T> void
        set_diagnostic_handler(::llvm::LLVMContext &ctx,
                               T *diagnostic_handler, void *data) {
#if HAVE_LLVM >= 0x0600
           ctx.setDiagnosticHandlerCallBack(diagnostic_handler, data);
#else
           ctx.setDiagnosticHandler(diagnostic_handler, data);
#endif
        }

	inline std::unique_ptr< ::llvm::Module>
	clone_module(const ::llvm::Module &mod)
	{
#if HAVE_LLVM >= 0x0700
		return ::llvm::CloneModule(mod);
#else
		return ::llvm::CloneModule(&mod);
#endif
	}

	template<typename T> void
	write_bitcode_to_file(const ::llvm::Module &mod, T &os)
	{
#if HAVE_LLVM >= 0x0700
		::llvm::WriteBitcodeToFile(mod, os);
#else
		::llvm::WriteBitcodeToFile(&mod, os);
#endif
	}

	template<typename TM, typename PM, typename OS, typename FT>
	bool add_passes_to_emit_file(TM &tm, PM &pm, OS &os, FT &ft)
	{
#if HAVE_LLVM >= 0x0700
		return tm.addPassesToEmitFile(pm, os, nullptr, ft);
#else
		return tm.addPassesToEmitFile(pm, os, ft);
#endif
	}

	template<typename T, typename M>
	T get_abi_type(const T &arg_type, const M &mod) {
#if HAVE_LLVM >= 0x0700
          return arg_type;
#else
          ::llvm::DataLayout dl(&mod);
          const unsigned arg_store_size = dl.getTypeStoreSize(arg_type);
          return !arg_type->isIntegerTy() ? arg_type :
            dl.getSmallestLegalIntType(mod.getContext(), arg_store_size * 8);
#endif
	}
      }
   }
}

#endif
