/* -*- c++ -*- */
/*
 * Copyright © 2013-2015 Intel Corporation
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
 */

#ifndef BRW_FS_SURFACE_BUILDER_H
#define BRW_FS_SURFACE_BUILDER_H

#include "brw_fs_builder.h"

namespace brw {
   namespace surface_access {
      fs_reg
      emit_untyped_read(const fs_builder &bld,
                        const fs_reg &surface, const fs_reg &addr,
                        unsigned dims, unsigned size,
                        brw_predicate pred = BRW_PREDICATE_NONE);

      void
      emit_untyped_write(const fs_builder &bld, const fs_reg &surface,
                         const fs_reg &addr, const fs_reg &src,
                         unsigned dims, unsigned size,
                         brw_predicate pred = BRW_PREDICATE_NONE);

      fs_reg
      emit_untyped_atomic(const fs_builder &bld,
                          const fs_reg &surface, const fs_reg &addr,
                          const fs_reg &src0, const fs_reg &src1,
                          unsigned dims, unsigned rsize, unsigned op,
                          brw_predicate pred = BRW_PREDICATE_NONE);

      fs_reg
      emit_untyped_atomic_float(const fs_builder &bld,
                                const fs_reg &surface, const fs_reg &addr,
                                const fs_reg &src0, const fs_reg &src1,
                                unsigned dims, unsigned rsize, unsigned op,
                                brw_predicate pred);

      fs_reg
      emit_typed_read(const fs_builder &bld, const fs_reg &surface,
                      const fs_reg &addr, unsigned dims, unsigned size);

      void
      emit_typed_write(const fs_builder &bld, const fs_reg &surface,
                       const fs_reg &addr, const fs_reg &src,
                       unsigned dims, unsigned size);

      fs_reg
      emit_typed_atomic(const fs_builder &bld, const fs_reg &surface,
                        const fs_reg &addr,
                        const fs_reg &src0, const fs_reg &src1,
                        unsigned dims, unsigned rsize, unsigned op,
                        brw_predicate pred = BRW_PREDICATE_NONE);

      fs_reg
      emit_byte_scattered_read(const fs_builder &bld,
                               const fs_reg &surface, const fs_reg &addr,
                               unsigned dims, unsigned size,
                               unsigned bit_size,
                               brw_predicate pred = BRW_PREDICATE_NONE);

      void
      emit_byte_scattered_write(const fs_builder &bld, const fs_reg &surface,
                                const fs_reg &addr, const fs_reg &src,
                                unsigned dims,
                                unsigned bit_size,
                                brw_predicate pred = BRW_PREDICATE_NONE);
   }
}
#endif
