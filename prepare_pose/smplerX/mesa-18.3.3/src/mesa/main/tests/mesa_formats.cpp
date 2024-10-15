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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \name mesa_formats.cpp
 *
 * Verify that all mesa formats are handled in certain functions and that
 * the format info table is sane.
 *
 */

#include <gtest/gtest.h>

#include "main/formats.h"
#include "main/glformats.h"

/**
 * Debug/test: check that all uncompressed formats are handled in the
 * _mesa_uncompressed_format_to_type_and_comps() function. When new pixel
 * formats are added to Mesa, that function needs to be updated.
 */
TEST(MesaFormatsTest, FormatTypeAndComps)
{
   for (int fi = MESA_FORMAT_NONE + 1; fi < MESA_FORMAT_COUNT; ++fi) {
      mesa_format f = (mesa_format) fi;
      SCOPED_TRACE(_mesa_get_format_name(f));

      /* This function will emit a problem/warning if the format is
       * not handled.
       */
      if (!_mesa_is_format_compressed(f)) {
         GLenum datatype = 0;
         GLenum error = 0;
         GLuint comps = 0;

         /* If the datatype is zero, the format was not handled */
         _mesa_uncompressed_format_to_type_and_comps(f, &datatype, &comps);
         EXPECT_NE(datatype, (GLenum)0);

         /* If the error isn't NO_ERROR, the format was not handled.
          * Use an arbitrary GLenum format. */
         _mesa_format_matches_format_and_type(f, GL_RG, datatype,
                                              GL_FALSE, &error);
         EXPECT_EQ((GLenum)GL_NO_ERROR, error);
      }

   }
}

/**
 * Do sanity checking of the format info table.
 */
TEST(MesaFormatsTest, FormatSanity)
{
   for (int fi = 0; fi < MESA_FORMAT_COUNT; ++fi) {
      mesa_format f = (mesa_format) fi;
      SCOPED_TRACE(_mesa_get_format_name(f));
      GLenum datatype = _mesa_get_format_datatype(f);
      GLint r = _mesa_get_format_bits(f, GL_RED_BITS);
      GLint g = _mesa_get_format_bits(f, GL_GREEN_BITS);
      GLint b = _mesa_get_format_bits(f, GL_BLUE_BITS);
      GLint a = _mesa_get_format_bits(f, GL_ALPHA_BITS);
      GLint l = _mesa_get_format_bits(f, GL_TEXTURE_LUMINANCE_SIZE);
      GLint i = _mesa_get_format_bits(f, GL_TEXTURE_INTENSITY_SIZE);

      /* Note: Z32_FLOAT_X24S8 has datatype of GL_NONE */
      EXPECT_TRUE(datatype == GL_NONE ||
                  datatype == GL_UNSIGNED_NORMALIZED ||
                  datatype == GL_SIGNED_NORMALIZED ||
                  datatype == GL_UNSIGNED_INT ||
                  datatype == GL_INT ||
                  datatype == GL_FLOAT);

      if (r > 0 && !_mesa_is_format_compressed(f)) {
         GLint bytes = _mesa_get_format_bytes(f);
         EXPECT_LE((r+g+b+a) / 8, bytes);
      }

      /* Determines if the base format has a channel [rgba] or property [li].
      * >  indicates existance
      * == indicates non-existance
      */
      #define HAS_PROP(rop,gop,bop,aop,lop,iop) \
         do { \
            EXPECT_TRUE(r rop 0); \
            EXPECT_TRUE(g gop 0); \
            EXPECT_TRUE(b bop 0); \
            EXPECT_TRUE(a aop 0); \
            EXPECT_TRUE(l lop 0); \
            EXPECT_TRUE(i iop 0); \
         } while(0)

      switch (_mesa_get_format_base_format(f)) {
      case GL_RGBA:
         HAS_PROP(>,>,>,>,==,==);
         break;
      case GL_RGB:
         HAS_PROP(>,>,>,==,==,==);
         break;
      case GL_RG:
         HAS_PROP(>,>,==,==,==,==);
         break;
      case GL_RED:
         HAS_PROP(>,==,==,==,==,==);
         break;
      case GL_LUMINANCE:
         HAS_PROP(==,==,==,==,>,==);
         break;
      case GL_INTENSITY:
         HAS_PROP(==,==,==,==,==,>);
         break;
      default:
         break;
      }

      #undef HAS_PROP

   }
}
