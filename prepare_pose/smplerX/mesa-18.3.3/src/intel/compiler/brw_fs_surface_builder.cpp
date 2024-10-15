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

#include "isl/isl.h"
#include "brw_fs_surface_builder.h"
#include "brw_fs.h"

using namespace brw;

namespace brw {
   namespace surface_access {
      namespace {
         /**
          * Generate a logical send opcode for a surface message and return
          * the result.
          */
         fs_reg
         emit_send(const fs_builder &bld, enum opcode opcode,
                   const fs_reg &addr, const fs_reg &src, const fs_reg &surface,
                   unsigned dims, unsigned arg, unsigned rsize,
                   brw_predicate pred = BRW_PREDICATE_NONE)
         {
            /* Reduce the dynamically uniform surface index to a single
             * scalar.
             */
            const fs_reg usurface = bld.emit_uniformize(surface);
            const fs_reg srcs[] = {
               addr, src, usurface, brw_imm_ud(dims), brw_imm_ud(arg)
            };
            const fs_reg dst = bld.vgrf(BRW_REGISTER_TYPE_UD, rsize);
            fs_inst *inst = bld.emit(opcode, dst, srcs, ARRAY_SIZE(srcs));

            inst->size_written = rsize * dst.component_size(inst->exec_size);
            inst->predicate = pred;
            return dst;
         }
      }

      /**
       * Emit an untyped surface read opcode.  \p dims determines the number
       * of components of the address and \p size the number of components of
       * the returned value.
       */
      fs_reg
      emit_untyped_read(const fs_builder &bld,
                        const fs_reg &surface, const fs_reg &addr,
                        unsigned dims, unsigned size,
                        brw_predicate pred)
      {
         return emit_send(bld, SHADER_OPCODE_UNTYPED_SURFACE_READ_LOGICAL,
                          addr, fs_reg(), surface, dims, size, size, pred);
      }

      /**
       * Emit an untyped surface write opcode.  \p dims determines the number
       * of components of the address and \p size the number of components of
       * the argument.
       */
      void
      emit_untyped_write(const fs_builder &bld, const fs_reg &surface,
                         const fs_reg &addr, const fs_reg &src,
                         unsigned dims, unsigned size,
                         brw_predicate pred)
      {
         emit_send(bld, SHADER_OPCODE_UNTYPED_SURFACE_WRITE_LOGICAL,
                   addr, src, surface, dims, size, 0, pred);
      }

      /**
       * Emit an untyped surface atomic opcode.  \p dims determines the number
       * of components of the address and \p rsize the number of components of
       * the returned value (either zero or one).
       */
      fs_reg
      emit_untyped_atomic(const fs_builder &bld,
                          const fs_reg &surface, const fs_reg &addr,
                          const fs_reg &src0, const fs_reg &src1,
                          unsigned dims, unsigned rsize, unsigned op,
                          brw_predicate pred)
      {
         /* FINISHME: Factor out this frequently recurring pattern into a
          * helper function.
          */
         const unsigned n = (src0.file != BAD_FILE) + (src1.file != BAD_FILE);
         const fs_reg srcs[] = { src0, src1 };
         const fs_reg tmp = bld.vgrf(BRW_REGISTER_TYPE_UD, n);
         bld.LOAD_PAYLOAD(tmp, srcs, n, 0);

         return emit_send(bld, SHADER_OPCODE_UNTYPED_ATOMIC_LOGICAL,
                          addr, tmp, surface, dims, op, rsize, pred);
      }

      /**
       * Emit an untyped surface atomic float opcode.  \p dims determines the
       * number of components of the address and \p rsize the number of
       * components of the returned value (either zero or one).
       */
      fs_reg
      emit_untyped_atomic_float(const fs_builder &bld,
                                const fs_reg &surface, const fs_reg &addr,
                                const fs_reg &src0, const fs_reg &src1,
                                unsigned dims, unsigned rsize, unsigned op,
                                brw_predicate pred)
      {
         /* FINISHME: Factor out this frequently recurring pattern into a
          * helper function.
          */
         const unsigned n = (src0.file != BAD_FILE) + (src1.file != BAD_FILE);
         const fs_reg srcs[] = { src0, src1 };
         const fs_reg tmp = bld.vgrf(src0.type, n);
         bld.LOAD_PAYLOAD(tmp, srcs, n, 0);

         return emit_send(bld, SHADER_OPCODE_UNTYPED_ATOMIC_FLOAT_LOGICAL,
                          addr, tmp, surface, dims, op, rsize, pred);
      }

      /**
       * Emit a typed surface read opcode.  \p dims determines the number of
       * components of the address and \p size the number of components of the
       * returned value.
       */
      fs_reg
      emit_typed_read(const fs_builder &bld, const fs_reg &surface,
                      const fs_reg &addr, unsigned dims, unsigned size)
      {
         return emit_send(bld, SHADER_OPCODE_TYPED_SURFACE_READ_LOGICAL,
                          addr, fs_reg(), surface, dims, size, size);
      }

      /**
       * Emit a typed surface write opcode.  \p dims determines the number of
       * components of the address and \p size the number of components of the
       * argument.
       */
      void
      emit_typed_write(const fs_builder &bld, const fs_reg &surface,
                       const fs_reg &addr, const fs_reg &src,
                       unsigned dims, unsigned size)
      {
         emit_send(bld, SHADER_OPCODE_TYPED_SURFACE_WRITE_LOGICAL,
                   addr, src, surface, dims, size, 0);
      }

      /**
       * Emit a typed surface atomic opcode.  \p dims determines the number of
       * components of the address and \p rsize the number of components of
       * the returned value (either zero or one).
       */
      fs_reg
      emit_typed_atomic(const fs_builder &bld, const fs_reg &surface,
                        const fs_reg &addr,
                        const fs_reg &src0, const fs_reg &src1,
                        unsigned dims, unsigned rsize, unsigned op,
                        brw_predicate pred)
      {
         /* FINISHME: Factor out this frequently recurring pattern into a
          * helper function.
          */
         const unsigned n = (src0.file != BAD_FILE) + (src1.file != BAD_FILE);
         const fs_reg srcs[] = { src0, src1 };
         const fs_reg tmp = bld.vgrf(BRW_REGISTER_TYPE_UD, n);
         bld.LOAD_PAYLOAD(tmp, srcs, n, 0);

         return emit_send(bld, SHADER_OPCODE_TYPED_ATOMIC_LOGICAL,
                          addr, tmp, surface, dims, op, rsize);
      }

      fs_reg
      emit_byte_scattered_read(const fs_builder &bld,
                               const fs_reg &surface, const fs_reg &addr,
                               unsigned dims, unsigned size,
                               unsigned bit_size, brw_predicate pred)
      {
         return emit_send(bld, SHADER_OPCODE_BYTE_SCATTERED_READ_LOGICAL,
                          addr, fs_reg(), surface, dims, bit_size, size, pred);
      }

      void
      emit_byte_scattered_write(const fs_builder &bld, const fs_reg &surface,
                                const fs_reg &addr, const fs_reg &src,
                                unsigned dims,
                                unsigned bit_size, brw_predicate pred)
      {
         emit_send(bld, SHADER_OPCODE_BYTE_SCATTERED_WRITE_LOGICAL,
                   addr, src, surface, dims, bit_size, 0, pred);
      }
   }
}
