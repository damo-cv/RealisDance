/**************************************************************************
 *
 * Copyright 2015 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include "lp_bld_format.h"
#include "lp_bld_type.h"
#include "lp_bld_struct.h"
#include "lp_bld_const.h"
#include "lp_bld_flow.h"
#include "lp_bld_swizzle.h"

#include "util/u_math.h"


/**
 * @file
 * Complex block-compression based formats are handled here by using a cache,
 * so re-decoding of every pixel is not required.
 * Especially for bilinear filtering, texel reuse is very high hence even
 * a small cache helps.
 * The elements in the cache are the decoded blocks - currently things
 * are restricted to formats which are 4x4 block based, and the decoded
 * texels must fit into 4x8 bits.
 * The cache is direct mapped so hitrates aren't all that great and cache
 * thrashing could happen.
 *
 * @author Roland Scheidegger <sroland@vmware.com>
 */


#if LP_BUILD_FORMAT_CACHE_DEBUG
static void
update_cache_access(struct gallivm_state *gallivm,
                    LLVMValueRef ptr,
                    unsigned count,
                    unsigned index)
{
   LLVMBuilderRef builder = gallivm->builder;
   LLVMValueRef member_ptr, cache_access;

   assert(index == LP_BUILD_FORMAT_CACHE_MEMBER_ACCESS_TOTAL ||
          index == LP_BUILD_FORMAT_CACHE_MEMBER_ACCESS_MISS);

   member_ptr = lp_build_struct_get_ptr(gallivm, ptr, index, "");
   cache_access = LLVMBuildLoad(builder, member_ptr, "cache_access");
   cache_access = LLVMBuildAdd(builder, cache_access,
                               LLVMConstInt(LLVMInt64TypeInContext(gallivm->context),
                                                                   count, 0), "");
   LLVMBuildStore(builder, cache_access, member_ptr);
}
#endif


static void
store_cached_block(struct gallivm_state *gallivm,
                   LLVMValueRef *col,
                   LLVMValueRef tag_value,
                   LLVMValueRef hash_index,
                   LLVMValueRef cache)
{
   LLVMBuilderRef builder = gallivm->builder;
   LLVMValueRef ptr, indices[3];
   LLVMTypeRef type_ptr4x32;
   unsigned count;

   type_ptr4x32 = LLVMPointerType(LLVMVectorType(LLVMInt32TypeInContext(gallivm->context), 4), 0);
   indices[0] = lp_build_const_int32(gallivm, 0);
   indices[1] = lp_build_const_int32(gallivm, LP_BUILD_FORMAT_CACHE_MEMBER_TAGS);
   indices[2] = hash_index;
   ptr = LLVMBuildGEP(builder, cache, indices, ARRAY_SIZE(indices), "");
   LLVMBuildStore(builder, tag_value, ptr);

   indices[1] = lp_build_const_int32(gallivm, LP_BUILD_FORMAT_CACHE_MEMBER_DATA);
   hash_index = LLVMBuildMul(builder, hash_index,
                             lp_build_const_int32(gallivm, 16), "");
   for (count = 0; count < 4; count++) {
      indices[2] = hash_index;
      ptr = LLVMBuildGEP(builder, cache, indices, ARRAY_SIZE(indices), "");
      ptr = LLVMBuildBitCast(builder, ptr, type_ptr4x32, "");
      LLVMBuildStore(builder, col[count], ptr);
      hash_index = LLVMBuildAdd(builder, hash_index,
                                lp_build_const_int32(gallivm, 4), "");
   }
}


static LLVMValueRef
lookup_cached_pixel(struct gallivm_state *gallivm,
                    LLVMValueRef ptr,
                    LLVMValueRef index)
{
   LLVMBuilderRef builder = gallivm->builder;
   LLVMValueRef member_ptr, indices[3];

   indices[0] = lp_build_const_int32(gallivm, 0);
   indices[1] = lp_build_const_int32(gallivm, LP_BUILD_FORMAT_CACHE_MEMBER_DATA);
   indices[2] = index;
   member_ptr = LLVMBuildGEP(builder, ptr, indices, ARRAY_SIZE(indices), "");
   return LLVMBuildLoad(builder, member_ptr, "cache_data");
}


static LLVMValueRef
lookup_tag_data(struct gallivm_state *gallivm,
                LLVMValueRef ptr,
                LLVMValueRef index)
{
   LLVMBuilderRef builder = gallivm->builder;
   LLVMValueRef member_ptr, indices[3];

   indices[0] = lp_build_const_int32(gallivm, 0);
   indices[1] = lp_build_const_int32(gallivm, LP_BUILD_FORMAT_CACHE_MEMBER_TAGS);
   indices[2] = index;
   member_ptr = LLVMBuildGEP(builder, ptr, indices, ARRAY_SIZE(indices), "");
   return LLVMBuildLoad(builder, member_ptr, "tag_data");
}


static void
update_cached_block(struct gallivm_state *gallivm,
                    const struct util_format_description *format_desc,
                    LLVMValueRef ptr_addr,
                    LLVMValueRef hash_index,
                    LLVMValueRef cache)

{
   LLVMBuilderRef builder = gallivm->builder;
   LLVMTypeRef i8t = LLVMInt8TypeInContext(gallivm->context);
   LLVMTypeRef pi8t = LLVMPointerType(i8t, 0);
   LLVMTypeRef i32t = LLVMInt32TypeInContext(gallivm->context);
   LLVMTypeRef i32x4 = LLVMVectorType(LLVMInt32TypeInContext(gallivm->context), 4);
   LLVMValueRef function;
   LLVMValueRef tag_value, tmp_ptr;
   LLVMValueRef col[4];
   unsigned i, j;

   /*
    * Use format_desc->fetch_rgba_8unorm() for each pixel in the block.
    * This doesn't actually make any sense whatsoever, someone would need
    * to write a function doing this for all pixels in a block (either as
    * an external c function or with generated code). Don't ask.
    */

   {
      /*
       * Function to call looks like:
       *   fetch(uint8_t *dst, const uint8_t *src, unsigned i, unsigned j)
       */
      LLVMTypeRef ret_type;
      LLVMTypeRef arg_types[4];
      LLVMTypeRef function_type;

      assert(format_desc->fetch_rgba_8unorm);

      ret_type = LLVMVoidTypeInContext(gallivm->context);
      arg_types[0] = pi8t;
      arg_types[1] = pi8t;
      arg_types[2] = i32t;
      arg_types[3] = i32t;
      function_type = LLVMFunctionType(ret_type, arg_types,
                                       ARRAY_SIZE(arg_types), 0);

      /* make const pointer for the C fetch_rgba_8unorm function */
      function = lp_build_const_int_pointer(gallivm,
         func_to_pointer((func_pointer) format_desc->fetch_rgba_8unorm));

      /* cast the callee pointer to the function's type */
      function = LLVMBuildBitCast(builder, function,
                                  LLVMPointerType(function_type, 0),
                                  "cast callee");
   }

   tmp_ptr = lp_build_array_alloca(gallivm, i32x4,
                                   lp_build_const_int32(gallivm, 16),
                                   "tmp_decode_store");
   tmp_ptr = LLVMBuildBitCast(builder, tmp_ptr, pi8t, "");

   /*
    * Invoke format_desc->fetch_rgba_8unorm() for each pixel.
    * This is going to be really really slow.
    * Note: the block store format is actually
    * x0y0x0y1x0y2x0y3 x1y0x1y1x1y2x1y3 ...
    */
   for (i = 0; i < 4; ++i) {
      for (j = 0; j < 4; ++j) {
         LLVMValueRef args[4];
         LLVMValueRef dst_offset = lp_build_const_int32(gallivm, (i * 4 + j) * 4);

         /*
          * Note we actually supply a pointer to the start of the block,
          * not the start of the texture.
          */
         args[0] = LLVMBuildGEP(gallivm->builder, tmp_ptr, &dst_offset, 1, "");
         args[1] = ptr_addr;
         args[2] = LLVMConstInt(i32t, i, 0);
         args[3] = LLVMConstInt(i32t, j, 0);
         LLVMBuildCall(builder, function, args, ARRAY_SIZE(args), "");
      }
   }

   /* Finally store the block - pointless mem copy + update tag. */
   tmp_ptr = LLVMBuildBitCast(builder, tmp_ptr, LLVMPointerType(i32x4, 0), "");
   for (i = 0; i < 4; ++i) {
      LLVMValueRef tmp_offset = lp_build_const_int32(gallivm, i);
      LLVMValueRef ptr = LLVMBuildGEP(gallivm->builder, tmp_ptr, &tmp_offset, 1, "");
      col[i] = LLVMBuildLoad(builder, ptr, "");
   }

   tag_value = LLVMBuildPtrToInt(gallivm->builder, ptr_addr,
                                 LLVMInt64TypeInContext(gallivm->context), "");
   store_cached_block(gallivm, col, tag_value, hash_index, cache);
}


/*
 * Do a cached lookup.
 *
 * Returns (vectors of) 4x8 rgba aos value
 */
LLVMValueRef
lp_build_fetch_cached_texels(struct gallivm_state *gallivm,
                             const struct util_format_description *format_desc,
                             unsigned n,
                             LLVMValueRef base_ptr,
                             LLVMValueRef offset,
                             LLVMValueRef i,
                             LLVMValueRef j,
                             LLVMValueRef cache)

{
   LLVMBuilderRef builder = gallivm->builder;
   unsigned count, low_bit, log2size;
   LLVMValueRef color, offset_stored, addr, ptr_addrtrunc, tmp;
   LLVMValueRef ij_index, hash_index, hash_mask, block_index;
   LLVMTypeRef i8t = LLVMInt8TypeInContext(gallivm->context);
   LLVMTypeRef i32t = LLVMInt32TypeInContext(gallivm->context);
   LLVMTypeRef i64t = LLVMInt64TypeInContext(gallivm->context);
   struct lp_type type;
   struct lp_build_context bld32;
   memset(&type, 0, sizeof type);
   type.width = 32;
   type.length = n;

   assert(format_desc->block.width == 4);
   assert(format_desc->block.height == 4);

   lp_build_context_init(&bld32, gallivm, type);

   /*
    * compute hash - we use direct mapped cache, the hash function could
    *                be better but it needs to be simple
    * per-element:
    *    compare offset with offset stored at tag (hash)
    *    if not equal decode/store block, update tag
    *    extract color from cache
    *    assemble result vector
    */

   /* TODO: not ideal with 32bit pointers... */

   low_bit = util_logbase2(format_desc->block.bits / 8);
   log2size = util_logbase2(LP_BUILD_FORMAT_CACHE_SIZE);
   addr = LLVMBuildPtrToInt(builder, base_ptr, i64t, "");
   ptr_addrtrunc = LLVMBuildPtrToInt(builder, base_ptr, i32t, "");
   ptr_addrtrunc = lp_build_broadcast_scalar(&bld32, ptr_addrtrunc);
   /* For the hash function, first mask off the unused lowest bits. Then just
      do some xor with address bits - only use lower 32bits */
   ptr_addrtrunc = LLVMBuildAdd(builder, offset, ptr_addrtrunc, "");
   ptr_addrtrunc = LLVMBuildLShr(builder, ptr_addrtrunc,
                                 lp_build_const_int_vec(gallivm, type, low_bit), "");
   /* This only really makes sense for size 64,128,256 */
   hash_index = ptr_addrtrunc;
   ptr_addrtrunc = LLVMBuildLShr(builder, ptr_addrtrunc,
                                 lp_build_const_int_vec(gallivm, type, 2*log2size), "");
   hash_index = LLVMBuildXor(builder, ptr_addrtrunc, hash_index, "");
   tmp = LLVMBuildLShr(builder, hash_index,
                       lp_build_const_int_vec(gallivm, type, log2size), "");
   hash_index = LLVMBuildXor(builder, hash_index, tmp, "");

   hash_mask = lp_build_const_int_vec(gallivm, type, LP_BUILD_FORMAT_CACHE_SIZE - 1);
   hash_index = LLVMBuildAnd(builder, hash_index, hash_mask, "");
   ij_index = LLVMBuildShl(builder, i, lp_build_const_int_vec(gallivm, type, 2), "");
   ij_index = LLVMBuildAdd(builder, ij_index, j, "");
   block_index = LLVMBuildShl(builder, hash_index,
                              lp_build_const_int_vec(gallivm, type, 4), "");
   block_index = LLVMBuildAdd(builder, ij_index, block_index, "");

   if (n > 1) {
      color = LLVMGetUndef(LLVMVectorType(i32t, n));
      for (count = 0; count < n; count++) {
         LLVMValueRef index, cond, colorx;
         LLVMValueRef block_indexx, hash_indexx, addrx, offsetx, ptr_addrx;
         struct lp_build_if_state if_ctx;

         index = lp_build_const_int32(gallivm, count);
         offsetx = LLVMBuildExtractElement(builder, offset, index, "");
         addrx = LLVMBuildZExt(builder, offsetx, i64t, "");
         addrx = LLVMBuildAdd(builder, addrx, addr, "");
         block_indexx = LLVMBuildExtractElement(builder, block_index, index, "");
         hash_indexx = LLVMBuildLShr(builder, block_indexx,
                                     lp_build_const_int32(gallivm, 4), "");
         offset_stored = lookup_tag_data(gallivm, cache, hash_indexx);
         cond = LLVMBuildICmp(builder, LLVMIntNE, offset_stored, addrx, "");

         lp_build_if(&if_ctx, gallivm, cond);
         {
            ptr_addrx = LLVMBuildIntToPtr(builder, addrx,
                                          LLVMPointerType(i8t, 0), "");
            update_cached_block(gallivm, format_desc, ptr_addrx, hash_indexx, cache);
#if LP_BUILD_FORMAT_CACHE_DEBUG
            update_cache_access(gallivm, cache, 1,
                                LP_BUILD_FORMAT_CACHE_MEMBER_ACCESS_MISS);
#endif
         }
         lp_build_endif(&if_ctx);

         colorx = lookup_cached_pixel(gallivm, cache, block_indexx);

         color = LLVMBuildInsertElement(builder, color, colorx,
                                        lp_build_const_int32(gallivm, count), "");
      }
   }
   else {
      LLVMValueRef cond;
      struct lp_build_if_state if_ctx;

      tmp = LLVMBuildZExt(builder, offset, i64t, "");
      addr = LLVMBuildAdd(builder, tmp, addr, "");
      offset_stored = lookup_tag_data(gallivm, cache, hash_index);
      cond = LLVMBuildICmp(builder, LLVMIntNE, offset_stored, addr, "");

      lp_build_if(&if_ctx, gallivm, cond);
      {
         tmp = LLVMBuildIntToPtr(builder, addr, LLVMPointerType(i8t, 0), "");
         update_cached_block(gallivm, format_desc, tmp, hash_index, cache);
#if LP_BUILD_FORMAT_CACHE_DEBUG
         update_cache_access(gallivm, cache, 1,
                             LP_BUILD_FORMAT_CACHE_MEMBER_ACCESS_MISS);
#endif
      }
      lp_build_endif(&if_ctx);

      color = lookup_cached_pixel(gallivm, cache, block_index);
   }
#if LP_BUILD_FORMAT_CACHE_DEBUG
   update_cache_access(gallivm, cache, n,
                       LP_BUILD_FORMAT_CACHE_MEMBER_ACCESS_TOTAL);
#endif
   return LLVMBuildBitCast(builder, color, LLVMVectorType(i8t, n * 4), "");
}

