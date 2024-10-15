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
/// Utility functions for LLVM IR metadata introspection.
///

#ifndef CLOVER_LLVM_METADATA_HPP
#define CLOVER_LLVM_METADATA_HPP

#include "llvm/compat.hpp"
#include "util/algorithm.hpp"

#include <vector>
#include <llvm/IR/Module.h>
#include <llvm/IR/Metadata.h>

namespace clover {
   namespace llvm {
      namespace detail {
         inline std::vector<const ::llvm::MDNode *>
         get_kernel_nodes(const ::llvm::Module &mod) {
            if (const ::llvm::NamedMDNode *n =
                   mod.getNamedMetadata("opencl.kernels"))
               return { n->op_begin(), n->op_end() };
            else
               return {};
         }

         inline std::function<bool (const ::llvm::MDNode *n)>
         is_kernel_node_for(const ::llvm::Function &f) {
            return [&](const ::llvm::MDNode *n) {
               using ::llvm::mdconst::dyn_extract;
               return &f == dyn_extract< ::llvm::Function>(n->getOperand(0));
            };
         }

         inline bool
         is_kernel(const ::llvm::Function &f) {
#if HAVE_LLVM >= 0x0309
            return f.getMetadata("kernel_arg_type");
#else
            return clover::any_of(is_kernel_node_for(f),
                                  get_kernel_nodes(*f.getParent()));
#endif
         }

         inline iterator_range< ::llvm::MDNode::op_iterator>
         get_kernel_metadata_operands(const ::llvm::Function &f,
                                      const std::string &name) {
#if HAVE_LLVM >= 0x0309
            // On LLVM v3.9+ kernel argument attributes are stored as
            // function metadata.
            const auto data_node = f.getMetadata(name);
            return range(data_node->op_begin(), data_node->op_end());
#else
            using ::llvm::cast;
            using ::llvm::dyn_cast;
            const auto kernel_node = find(is_kernel_node_for(f),
                                          get_kernel_nodes(*f.getParent()));

            const auto data_node = cast< ::llvm::MDNode>(
               find([&](const ::llvm::MDOperand &op) {
                     if (auto m = dyn_cast< ::llvm::MDNode>(op))
                        if (m->getNumOperands())
                           if (auto m_name = dyn_cast< ::llvm::MDString>(
                                  m->getOperand(0).get()))
                              return m_name->getString() == name;

                     return false;
                  },
                  kernel_node->operands()));

            // Skip the first operand node which is just the metadata
            // attribute name.
            return range(data_node->op_begin() + 1, data_node->op_end());
#endif
         }
      }

      ///
      /// Extract the string metadata node \p name corresponding to the kernel
      /// argument given by \p arg.
      ///
      inline std::string
      get_argument_metadata(const ::llvm::Function &f,
                            const ::llvm::Argument &arg,
                            const std::string &name) {
         return ::llvm::cast< ::llvm::MDString>(
               detail::get_kernel_metadata_operands(f, name)[arg.getArgNo()])
            ->getString();
      }

      ///
      /// Return a vector with all CL kernel functions found in the LLVM
      /// module \p mod.
      ///
      inline std::vector<const ::llvm::Function *>
      get_kernels(const ::llvm::Module &mod) {
         std::vector<const ::llvm::Function *> fs;

         for (auto &f : mod.getFunctionList()) {
            if (detail::is_kernel(f))
               fs.push_back(&f);
         }

         return fs;
      }
   }
}

#endif
