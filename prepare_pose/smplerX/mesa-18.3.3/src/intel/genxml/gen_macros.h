/*
 * Copyright © 2015 Intel Corporation
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

#ifndef GEN_MACROS_H
#define GEN_MACROS_H

/* Macros for handling per-gen compilation.
 *
 * The prefixing macros GENX() and genX() automatically prefix whatever you
 * give them by GENX_ or genX_  where X is the gen number.
 *
 * You can do pseudo-runtime checks in your function such as
 *
 * if (GEN_GEN > 8 || GEN_IS_HASWELL) {
 *    // Do something
 * }
 *
 * The contents of the if statement must be valid regardless of gen, but
 * the if will get compiled away on everything except haswell.
 *
 * For places where you really do have a compile-time conflict, you can
 * use preprocessor logic:
 *
 * #if (GEN_GEN > 8 || GEN_IS_HASWELL)
 *    // Do something
 * #endif
 *
 * However, it is strongly recommended that the former be used whenever
 * possible.
 */

/* Base macro defined on the command line.  If we don't have this, we can't
 * do anything.
 */
#ifndef GEN_VERSIONx10
#  error "The GEN_VERSIONx10 macro must be defined"
#endif

#define GEN_GEN ((GEN_VERSIONx10) / 10)
#define GEN_IS_HASWELL ((GEN_VERSIONx10) == 75)
#define GEN_IS_G4X ((GEN_VERSIONx10) == 45)

/* Prefixing macros */
#if (GEN_VERSIONx10 == 40)
#  define GENX(X) GEN4_##X
#  define genX(x) gen4_##x
#elif (GEN_VERSIONx10 == 45)
#  define GENX(X) GEN45_##X
#  define genX(x) gen45_##x
#elif (GEN_VERSIONx10 == 50)
#  define GENX(X) GEN5_##X
#  define genX(x) gen5_##x
#elif (GEN_VERSIONx10 == 60)
#  define GENX(X) GEN6_##X
#  define genX(x) gen6_##x
#elif (GEN_VERSIONx10 == 70)
#  define GENX(X) GEN7_##X
#  define genX(x) gen7_##x
#elif (GEN_VERSIONx10 == 75)
#  define GENX(X) GEN75_##X
#  define genX(x) gen75_##x
#elif (GEN_VERSIONx10 == 80)
#  define GENX(X) GEN8_##X
#  define genX(x) gen8_##x
#elif (GEN_VERSIONx10 == 90)
#  define GENX(X) GEN9_##X
#  define genX(x) gen9_##x
#elif (GEN_VERSIONx10 == 100)
#  define GENX(X) GEN10_##X
#  define genX(x) gen10_##x
#elif (GEN_VERSIONx10 == 110)
#  define GENX(X) GEN11_##X
#  define genX(x) gen11_##x
#else
#  error "Need to add prefixing macros for this gen"
#endif

#endif /* GEN_MACROS_H */
