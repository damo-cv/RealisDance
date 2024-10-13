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
 * @file gen_swr_context_llvm.h
 *
 * @brief auto-generated file
 *
 * DO NOT EDIT
 *
 * Generation Command Line:
 *   ./rasterizer/codegen/gen_llvm_types.py
 *     --input
 *     ./swr_context.h
 *     --output
 *     ./gen_swr_context_llvm.h
 *
 ******************************************************************************/
// clang-format off

#pragma once

namespace SwrJit
{
    using namespace llvm;

    INLINE static StructType* Gen_swr_jit_texture(JitManager* pJitMgr)
    {
        LLVMContext& ctx = pJitMgr->mContext;

        StructType* pRetType = pJitMgr->mpCurrentModule->getTypeByName("swr_jit_texture");
        if (pRetType == nullptr)
        {
            std::vector<Type*> members =
            {
                /* width       */ Type::getInt32Ty(ctx),
                /* height      */ Type::getInt32Ty(ctx),
                /* depth       */ Type::getInt32Ty(ctx),
                /* first_level */ Type::getInt32Ty(ctx),
                /* last_level  */ Type::getInt32Ty(ctx),
                /* base_ptr    */ PointerType::get(Type::getInt8Ty(ctx), 0),
                /* row_stride  */ ArrayType::get(Type::getInt32Ty(ctx), PIPE_MAX_TEXTURE_LEVELS),
                /* img_stride  */ ArrayType::get(Type::getInt32Ty(ctx), PIPE_MAX_TEXTURE_LEVELS),
                /* mip_offsets */ ArrayType::get(Type::getInt32Ty(ctx), PIPE_MAX_TEXTURE_LEVELS),
            };

            pRetType = StructType::create(members, "swr_jit_texture", false);

            // Compute debug metadata
            llvm::DIBuilder builder(*pJitMgr->mpCurrentModule);
            llvm::DIFile* pFile = builder.createFile("swr_context.h", ".");

            std::vector<std::pair<std::string, uint32_t>> dbgMembers =
            {
                std::make_pair("width",       67),
                std::make_pair("height",      68),
                std::make_pair("depth",       69),
                std::make_pair("first_level", 70),
                std::make_pair("last_level",  71),
                std::make_pair("base_ptr",    72),
                std::make_pair("row_stride",  73),
                std::make_pair("img_stride",  74),
                std::make_pair("mip_offsets", 75),
            };
            pJitMgr->CreateDebugStructType(pRetType, "swr_jit_texture", pFile, 66, dbgMembers);
        }

        return pRetType;
    }

    static const uint32_t swr_jit_texture_width       = 0;
    static const uint32_t swr_jit_texture_height      = 1;
    static const uint32_t swr_jit_texture_depth       = 2;
    static const uint32_t swr_jit_texture_first_level = 3;
    static const uint32_t swr_jit_texture_last_level  = 4;
    static const uint32_t swr_jit_texture_base_ptr    = 5;
    static const uint32_t swr_jit_texture_row_stride  = 6;
    static const uint32_t swr_jit_texture_img_stride  = 7;
    static const uint32_t swr_jit_texture_mip_offsets = 8;

    INLINE static StructType* Gen_swr_jit_sampler(JitManager* pJitMgr)
    {
        LLVMContext& ctx = pJitMgr->mContext;

        StructType* pRetType = pJitMgr->mpCurrentModule->getTypeByName("swr_jit_sampler");
        if (pRetType == nullptr)
        {
            std::vector<Type*> members =
            {
                /* min_lod      */ Type::getFloatTy(ctx),
                /* max_lod      */ Type::getFloatTy(ctx),
                /* lod_bias     */ Type::getFloatTy(ctx),
                /* border_color */ ArrayType::get(Type::getFloatTy(ctx), 4),
            };

            pRetType = StructType::create(members, "swr_jit_sampler", false);

            // Compute debug metadata
            llvm::DIBuilder builder(*pJitMgr->mpCurrentModule);
            llvm::DIFile* pFile = builder.createFile("swr_context.h", ".");

            std::vector<std::pair<std::string, uint32_t>> dbgMembers =
            {
                std::make_pair("min_lod",      79),
                std::make_pair("max_lod",      80),
                std::make_pair("lod_bias",     81),
                std::make_pair("border_color", 82),
            };
            pJitMgr->CreateDebugStructType(pRetType, "swr_jit_sampler", pFile, 78, dbgMembers);
        }

        return pRetType;
    }

    static const uint32_t swr_jit_sampler_min_lod      = 0;
    static const uint32_t swr_jit_sampler_max_lod      = 1;
    static const uint32_t swr_jit_sampler_lod_bias     = 2;
    static const uint32_t swr_jit_sampler_border_color = 3;

    INLINE static StructType* Gen_swr_draw_context(JitManager* pJitMgr)
    {
        LLVMContext& ctx = pJitMgr->mContext;

        StructType* pRetType = pJitMgr->mpCurrentModule->getTypeByName("swr_draw_context");
        if (pRetType == nullptr)
        {
            std::vector<Type*> members =
            {
                /* constantVS       */ ArrayType::get(PointerType::get(Type::getFloatTy(ctx), 0), PIPE_MAX_CONSTANT_BUFFERS),
                /* num_constantsVS  */ ArrayType::get(Type::getInt32Ty(ctx), PIPE_MAX_CONSTANT_BUFFERS),
                /* constantFS       */ ArrayType::get(PointerType::get(Type::getFloatTy(ctx), 0), PIPE_MAX_CONSTANT_BUFFERS),
                /* num_constantsFS  */ ArrayType::get(Type::getInt32Ty(ctx), PIPE_MAX_CONSTANT_BUFFERS),
                /* constantGS       */ ArrayType::get(PointerType::get(Type::getFloatTy(ctx), 0), PIPE_MAX_CONSTANT_BUFFERS),
                /* num_constantsGS  */ ArrayType::get(Type::getInt32Ty(ctx), PIPE_MAX_CONSTANT_BUFFERS),
                /* texturesVS       */ ArrayType::get(Gen_swr_jit_texture(pJitMgr), PIPE_MAX_SHADER_SAMPLER_VIEWS),
                /* samplersVS       */ ArrayType::get(Gen_swr_jit_sampler(pJitMgr), PIPE_MAX_SAMPLERS),
                /* texturesFS       */ ArrayType::get(Gen_swr_jit_texture(pJitMgr), PIPE_MAX_SHADER_SAMPLER_VIEWS),
                /* samplersFS       */ ArrayType::get(Gen_swr_jit_sampler(pJitMgr), PIPE_MAX_SAMPLERS),
                /* texturesGS       */ ArrayType::get(Gen_swr_jit_texture(pJitMgr), PIPE_MAX_SHADER_SAMPLER_VIEWS),
                /* samplersGS       */ ArrayType::get(Gen_swr_jit_sampler(pJitMgr), PIPE_MAX_SAMPLERS),
                /* userClipPlanes   */ ArrayType::get(ArrayType::get(Type::getFloatTy(ctx), 4), PIPE_MAX_CLIP_PLANES),
                /* polyStipple      */ ArrayType::get(Type::getInt32Ty(ctx), 32),
                /* renderTargets    */ ArrayType::get(Gen_SWR_SURFACE_STATE(pJitMgr), SWR_NUM_ATTACHMENTS),
                /* swr_query_result */ PointerType::get(Type::getInt32Ty(ctx), 0),
                /* pAPI             */ PointerType::get(Type::getInt32Ty(ctx), 0),
            };

            pRetType = StructType::create(members, "swr_draw_context", false);

            // Compute debug metadata
            llvm::DIBuilder builder(*pJitMgr->mpCurrentModule);
            llvm::DIFile* pFile = builder.createFile("swr_context.h", ".");

            std::vector<std::pair<std::string, uint32_t>> dbgMembers =
            {
                std::make_pair("constantVS",       86),
                std::make_pair("num_constantsVS",  87),
                std::make_pair("constantFS",       88),
                std::make_pair("num_constantsFS",  89),
                std::make_pair("constantGS",       90),
                std::make_pair("num_constantsGS",  91),
                std::make_pair("texturesVS",       93),
                std::make_pair("samplersVS",       94),
                std::make_pair("texturesFS",       95),
                std::make_pair("samplersFS",       96),
                std::make_pair("texturesGS",       97),
                std::make_pair("samplersGS",       98),
                std::make_pair("userClipPlanes",   100),
                std::make_pair("polyStipple",      102),
                std::make_pair("renderTargets",    104),
                std::make_pair("swr_query_result", 105),
                std::make_pair("pAPI",             106),
            };
            pJitMgr->CreateDebugStructType(pRetType, "swr_draw_context", pFile, 85, dbgMembers);
        }

        return pRetType;
    }

    static const uint32_t swr_draw_context_constantVS       = 0;
    static const uint32_t swr_draw_context_num_constantsVS  = 1;
    static const uint32_t swr_draw_context_constantFS       = 2;
    static const uint32_t swr_draw_context_num_constantsFS  = 3;
    static const uint32_t swr_draw_context_constantGS       = 4;
    static const uint32_t swr_draw_context_num_constantsGS  = 5;
    static const uint32_t swr_draw_context_texturesVS       = 6;
    static const uint32_t swr_draw_context_samplersVS       = 7;
    static const uint32_t swr_draw_context_texturesFS       = 8;
    static const uint32_t swr_draw_context_samplersFS       = 9;
    static const uint32_t swr_draw_context_texturesGS       = 10;
    static const uint32_t swr_draw_context_samplersGS       = 11;
    static const uint32_t swr_draw_context_userClipPlanes   = 12;
    static const uint32_t swr_draw_context_polyStipple      = 13;
    static const uint32_t swr_draw_context_renderTargets    = 14;
    static const uint32_t swr_draw_context_swr_query_result = 15;
    static const uint32_t swr_draw_context_pAPI             = 16;

} // namespace SwrJit


// clang-format on
