//============================================================================
// Copyright (C) 2014-2017 Intel Corporation.   All Rights Reserved.
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
// @file gen_builder_intrin.hpp
//
// @brief auto-generated file
//
// DO NOT EDIT
//
// Generation Command Line:
//  ./rasterizer/codegen/gen_llvm_ir_macros.py
//    --output
//    rasterizer/jitter
//    --gen_intrin_h
//
//============================================================================
// clang-format off
#pragma once

//============================================================================
// Auto-generated llvm intrinsics
//============================================================================
Value* CTTZ(Value* a, Value* flag, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::cttz, args);
    return CALL(pFunc, std::initializer_list<Value*>{a, flag}, name);
}

Value* CTLZ(Value* a, Value* flag, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::ctlz, args);
    return CALL(pFunc, std::initializer_list<Value*>{a, flag}, name);
}

Value* VSQRTPS(Value* a, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::sqrt, args);
    return CALL(pFunc, std::initializer_list<Value*>{a}, name);
}

Value* STACKSAVE(const llvm::Twine& name = "")
{
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::stacksave);
    return CALL(pFunc, std::initializer_list<Value*>{}, name);
}

Value* STACKRESTORE(Value* a, const llvm::Twine& name = "")
{
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::stackrestore);
    return CALL(pFunc, std::initializer_list<Value*>{a}, name);
}

Value* VMINPS(Value* a, Value* b, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::minnum, args);
    return CALL(pFunc, std::initializer_list<Value*>{a, b}, name);
}

Value* VMAXPS(Value* a, Value* b, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::maxnum, args);
    return CALL(pFunc, std::initializer_list<Value*>{a, b}, name);
}

Value* VFMADDPS(Value* a, Value* b, Value* c, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::fmuladd, args);
    return CALL(pFunc, std::initializer_list<Value*>{a, b, c}, name);
}

Value* DEBUGTRAP(const llvm::Twine& name = "")
{
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::debugtrap);
    return CALL(pFunc, std::initializer_list<Value*>{}, name);
}

Value* POPCNT(Value* a, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::ctpop, args);
    return CALL(pFunc, std::initializer_list<Value*>{a}, name);
}

Value* LOG2(Value* a, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::log2, args);
    return CALL(pFunc, std::initializer_list<Value*>{a}, name);
}

Value* FABS(Value* a, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::fabs, args);
    return CALL(pFunc, std::initializer_list<Value*>{a}, name);
}

Value* EXP2(Value* a, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::exp2, args);
    return CALL(pFunc, std::initializer_list<Value*>{a}, name);
}

Value* POW(Value* a, Value* b, const llvm::Twine& name = "")
{
    SmallVector<Type*, 1> args;
    args.push_back(a->getType());
    Function* pFunc = Intrinsic::getDeclaration(JM()->mpCurrentModule, Intrinsic::pow, args);
    return CALL(pFunc, std::initializer_list<Value*>{a, b}, name);
}

    // clang-format on
