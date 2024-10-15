/*
 * Copyright 2016 Red Hat.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "sp_context.h"
#include "sp_buffer.h"
#include "sp_texture.h"

#include "util/u_format.h"

static bool
get_dimensions(const struct pipe_shader_buffer *bview,
               const struct softpipe_resource *spr,
               unsigned *width)
{
   *width = bview->buffer_size;
   /*
    * Bounds check the buffer size from the view
    * and the buffer size from the underlying buffer.
    */
   if (*width > spr->base.width0)
      return false;
   return true;
}

/*
 * Implement the image LOAD operation.
 */
static void
sp_tgsi_load(const struct tgsi_buffer *buffer,
             const struct tgsi_buffer_params *params,
             const int s[TGSI_QUAD_SIZE],
             float rgba[TGSI_NUM_CHANNELS][TGSI_QUAD_SIZE])
{
   struct sp_tgsi_buffer *sp_buf = (struct sp_tgsi_buffer *)buffer;
   struct pipe_shader_buffer *bview;
   struct softpipe_resource *spr;
   unsigned width;
   int c, j;
   unsigned char *data_ptr;
   const struct util_format_description *format_desc = util_format_description(PIPE_FORMAT_R32_UINT);

   if (params->unit >= PIPE_MAX_SHADER_BUFFERS)
      goto fail_write_all_zero;

   bview = &sp_buf->sp_bview[params->unit];
   spr = softpipe_resource(bview->buffer);
   if (!spr)
      goto fail_write_all_zero;

   if (!get_dimensions(bview, spr, &width))
      return;

   for (j = 0; j < TGSI_QUAD_SIZE; j++) {
      int s_coord;
      bool fill_zero = false;
      uint32_t sdata[4];

      if (!(params->execmask & (1 << j)))
         fill_zero = true;

      s_coord = s[j];
      if (s_coord >= width)
         fill_zero = true;

      if (fill_zero) {
         for (c = 0; c < 4; c++)
            rgba[c][j] = 0;
         continue;
      }
      data_ptr = (unsigned char *)spr->data + bview->buffer_offset + s_coord;
      for (c = 0; c < 4; c++) {
         format_desc->fetch_rgba_uint(sdata, data_ptr, 0, 0);
         ((uint32_t *)rgba[c])[j] = sdata[0];
         data_ptr += 4;
      }
   }
   return;
fail_write_all_zero:
   memset(rgba, 0, TGSI_NUM_CHANNELS * TGSI_QUAD_SIZE * 4);
   return;
}

/*
 * Implement the buffer STORE operation.
 */
static void
sp_tgsi_store(const struct tgsi_buffer *buffer,
              const struct tgsi_buffer_params *params,
              const int s[TGSI_QUAD_SIZE],
              float rgba[TGSI_NUM_CHANNELS][TGSI_QUAD_SIZE])
{
   struct sp_tgsi_buffer *sp_buf = (struct sp_tgsi_buffer *)buffer;
   struct pipe_shader_buffer *bview;
   struct softpipe_resource *spr;
   unsigned width;
   unsigned char *data_ptr;
   int j, c;
   const struct util_format_description *format_desc = util_format_description(PIPE_FORMAT_R32_UINT);

   if (params->unit >= PIPE_MAX_SHADER_BUFFERS)
      return;

   bview = &sp_buf->sp_bview[params->unit];
   spr = softpipe_resource(bview->buffer);
   if (!spr)
      return;

   if (!get_dimensions(bview, spr, &width))
      return;

   for (j = 0; j < TGSI_QUAD_SIZE; j++) {
      int s_coord;

      if (!(params->execmask & (1 << j)))
         continue;

      s_coord = s[j];
      if (s_coord >= width)
         continue;

      data_ptr = (unsigned char *)spr->data + bview->buffer_offset + s_coord;

      for (c = 0; c < 4; c++) {
         if (params->writemask & (1 << c)) {
            unsigned temp[4];
            unsigned char *dptr = data_ptr + (c * 4);
            temp[0] = ((uint32_t *)rgba[c])[j];
            format_desc->pack_rgba_uint(dptr, 0, temp, 0, 1, 1);
         }
      }
   }
}

/*
 * Implement atomic operations on unsigned integers.
 */
static void
handle_op_uint(const struct pipe_shader_buffer *bview,
               bool just_read,
               unsigned char *data_ptr,
               uint qi,
               enum tgsi_opcode opcode,
               unsigned writemask,
               float rgba[TGSI_NUM_CHANNELS][TGSI_QUAD_SIZE],
               float rgba2[TGSI_NUM_CHANNELS][TGSI_QUAD_SIZE])
{
   uint c;
   const struct util_format_description *format_desc = util_format_description(PIPE_FORMAT_R32_UINT);
   unsigned sdata[4];

   for (c = 0; c < 4; c++) {
      unsigned temp[4];
      unsigned char *dptr = data_ptr + (c * 4);
      format_desc->fetch_rgba_uint(temp, dptr, 0, 0);
      sdata[c] = temp[0];
   }

   if (just_read) {
      for (c = 0; c < 4; c++) {
         ((uint32_t *)rgba[c])[qi] = sdata[c];
      }
      return;
   }

   switch (opcode) {
   case TGSI_OPCODE_ATOMUADD:
      for (c = 0; c < 4; c++) {
         unsigned temp = sdata[c];
         sdata[c] += ((uint32_t *)rgba[c])[qi];
         ((uint32_t *)rgba[c])[qi] = temp;
      }
      break;
   case TGSI_OPCODE_ATOMXCHG:
      for (c = 0; c < 4; c++) {
         unsigned temp = sdata[c];
         sdata[c] = ((uint32_t *)rgba[c])[qi];
         ((uint32_t *)rgba[c])[qi] = temp;
      }
      break;
   case TGSI_OPCODE_ATOMCAS:
      for (c = 0; c < 4; c++) {
         unsigned dst_x = sdata[c];
         unsigned cmp_x = ((uint32_t *)rgba[c])[qi];
         unsigned src_x = ((uint32_t *)rgba2[c])[qi];
         unsigned temp = sdata[c];
         sdata[c] = (dst_x == cmp_x) ? src_x : dst_x;
         ((uint32_t *)rgba[c])[qi] = temp;
      }
      break;
   case TGSI_OPCODE_ATOMAND:
      for (c = 0; c < 4; c++) {
         unsigned temp = sdata[c];
         sdata[c] &= ((uint32_t *)rgba[c])[qi];
         ((uint32_t *)rgba[c])[qi] = temp;
      }
      break;
   case TGSI_OPCODE_ATOMOR:
      for (c = 0; c < 4; c++) {
         unsigned temp = sdata[c];
         sdata[c] |= ((uint32_t *)rgba[c])[qi];
         ((uint32_t *)rgba[c])[qi] = temp;
      }
      break;
   case TGSI_OPCODE_ATOMXOR:
      for (c = 0; c < 4; c++) {
         unsigned temp = sdata[c];
         sdata[c] ^= ((uint32_t *)rgba[c])[qi];
         ((uint32_t *)rgba[c])[qi] = temp;
      }
      break;
   case TGSI_OPCODE_ATOMUMIN:
      for (c = 0; c < 4; c++) {
         unsigned dst_x = sdata[c];
         unsigned src_x = ((uint32_t *)rgba[c])[qi];
         sdata[c] = MIN2(dst_x, src_x);
         ((uint32_t *)rgba[c])[qi] = dst_x;
      }
      break;
   case TGSI_OPCODE_ATOMUMAX:
      for (c = 0; c < 4; c++) {
         unsigned dst_x = sdata[c];
         unsigned src_x = ((uint32_t *)rgba[c])[qi];
         sdata[c] = MAX2(dst_x, src_x);
         ((uint32_t *)rgba[c])[qi] = dst_x;
      }
      break;
   case TGSI_OPCODE_ATOMIMIN:
      for (c = 0; c < 4; c++) {
         int dst_x = sdata[c];
         int src_x = ((uint32_t *)rgba[c])[qi];
         sdata[c] = MIN2(dst_x, src_x);
         ((uint32_t *)rgba[c])[qi] = dst_x;
      }
      break;
   case TGSI_OPCODE_ATOMIMAX:
      for (c = 0; c < 4; c++) {
         int dst_x = sdata[c];
         int src_x = ((uint32_t *)rgba[c])[qi];
         sdata[c] = MAX2(dst_x, src_x);
         ((uint32_t *)rgba[c])[qi] = dst_x;
      }
      break;
   default:
      assert(!"Unexpected TGSI opcode in sp_tgsi_op");
      break;
   }

   for (c = 0; c < 4; c++) {
      if (writemask & (1 << c)) {
         unsigned temp[4];
         unsigned char *dptr = data_ptr + (c * 4);
         temp[0] = sdata[c];
         format_desc->pack_rgba_uint(dptr, 0, temp, 0, 1, 1);
      }
   }
}

/*
 * Implement atomic buffer operations.
 */
static void
sp_tgsi_op(const struct tgsi_buffer *buffer,
           const struct tgsi_buffer_params *params,
           enum tgsi_opcode opcode,
           const int s[TGSI_QUAD_SIZE],
           float rgba[TGSI_NUM_CHANNELS][TGSI_QUAD_SIZE],
           float rgba2[TGSI_NUM_CHANNELS][TGSI_QUAD_SIZE])
{
   struct sp_tgsi_buffer *sp_buf = (struct sp_tgsi_buffer *)buffer;
   struct pipe_shader_buffer *bview;
   struct softpipe_resource *spr;
   unsigned width;
   int j, c;
   unsigned char *data_ptr;

   if (params->unit >= PIPE_MAX_SHADER_BUFFERS)
      return;

   bview = &sp_buf->sp_bview[params->unit];
   spr = softpipe_resource(bview->buffer);
   if (!spr)
      goto fail_write_all_zero;

   if (!get_dimensions(bview, spr, &width))
      goto fail_write_all_zero;

   for (j = 0; j < TGSI_QUAD_SIZE; j++) {
      int s_coord;
      bool just_read = false;

      s_coord = s[j];
      if (s_coord >= width) {
         for (c = 0; c < 4; c++) {
            rgba[c][j] = 0;
         }
         continue;
      }

      /* just readback the value for atomic if execmask isn't set */
      if (!(params->execmask & (1 << j))) {
         just_read = true;
      }

      data_ptr = (unsigned char *)spr->data + bview->buffer_offset + s_coord;
      /* we should see atomic operations on r32 formats */

      handle_op_uint(bview, just_read, data_ptr, j,
                     opcode, params->writemask, rgba, rgba2);
   }
   return;
fail_write_all_zero:
   memset(rgba, 0, TGSI_NUM_CHANNELS * TGSI_QUAD_SIZE * 4);
   return;
}

/*
 * return size of the attached buffer for RESQ opcode.
 */
static void
sp_tgsi_get_dims(const struct tgsi_buffer *buffer,
                 const struct tgsi_buffer_params *params,
                 int *dim)
{
   struct sp_tgsi_buffer *sp_buf = (struct sp_tgsi_buffer *)buffer;
   struct pipe_shader_buffer *bview;
   struct softpipe_resource *spr;

   if (params->unit >= PIPE_MAX_SHADER_BUFFERS)
      return;

   bview = &sp_buf->sp_bview[params->unit];
   spr = softpipe_resource(bview->buffer);
   if (!spr)
      return;

   *dim = bview->buffer_size;
}

struct sp_tgsi_buffer *
sp_create_tgsi_buffer(void)
{
   struct sp_tgsi_buffer *buf = CALLOC_STRUCT(sp_tgsi_buffer);
   if (!buf)
      return NULL;

   buf->base.load = sp_tgsi_load;
   buf->base.store = sp_tgsi_store;
   buf->base.op = sp_tgsi_op;
   buf->base.get_dims = sp_tgsi_get_dims;
   return buf;
};
