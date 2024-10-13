/**********************************************************
 * Copyright 2008-2009 VMware, Inc.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **********************************************************/

#include "util/u_inlines.h"
#include "util/u_prim.h"
#include "util/u_upload_mgr.h"
#include "indices/u_indices.h"

#include "svga_cmd.h"
#include "svga_draw.h"
#include "svga_draw_private.h"
#include "svga_resource_buffer.h"
#include "svga_winsys.h"
#include "svga_context.h"
#include "svga_hw_reg.h"


/**
 * Return a new index buffer which contains a translation of the original
 * index buffer.  An example of a translation is converting from QUAD
 * primitives to TRIANGLE primitives.  Each set of four indexes for a quad
 * will be converted to six indices for two triangles.
 *
 * Before generating the new index buffer we'll check if the incoming
 * buffer already has a translated buffer that can be re-used.
 * This benefits demos like Cinebench R15 which has many
 * glDrawElements(GL_QUADS) commands (we can't draw quads natively).
 *
 * \param offset  offset in bytes to first index to translate in src buffer
 * \param orig_prim  original primitive type (like PIPE_PRIM_QUADS)
 * \param gen_prim  new/generated primitive type (like PIPE_PRIM_TRIANGLES)
 * \param orig_nr  number of indexes to translate in source buffer
 * \param gen_nr  number of indexes to write into new/dest buffer
 * \param index_size  bytes per index (2 or 4)
 * \param translate  the translation function from the u_translate module
 * \param out_buf  returns the new/translated index buffer
 * \return error code to indicate success failure
 */
static enum pipe_error
translate_indices(struct svga_hwtnl *hwtnl, struct pipe_resource *src,
                  unsigned offset,
                  enum pipe_prim_type orig_prim, enum pipe_prim_type gen_prim,
                  unsigned orig_nr, unsigned gen_nr,
                  unsigned index_size,
                  u_translate_func translate, struct pipe_resource **out_buf)
{
   struct pipe_context *pipe = &hwtnl->svga->pipe;
   struct svga_screen *screen = svga_screen(pipe->screen);
   struct svga_buffer *src_sbuf = svga_buffer(src);
   struct pipe_transfer *src_transfer = NULL;
   struct pipe_transfer *dst_transfer = NULL;
   unsigned size = index_size * gen_nr;
   const void *src_map = NULL;
   struct pipe_resource *dst = NULL;
   void *dst_map = NULL;

   assert(index_size == 2 || index_size == 4);

   if (!screen->debug.no_cache_index_buffers) {
      /* Check if we already have a translated index buffer */
      if (src_sbuf->translated_indices.buffer &&
          src_sbuf->translated_indices.orig_prim == orig_prim &&
          src_sbuf->translated_indices.new_prim == gen_prim &&
          src_sbuf->translated_indices.offset == offset &&
          src_sbuf->translated_indices.count == orig_nr &&
          src_sbuf->translated_indices.index_size == index_size) {
         pipe_resource_reference(out_buf, src_sbuf->translated_indices.buffer);
         return PIPE_OK;
      }
   }

   /* Need to trim vertex count to make sure we don't write too much data
    * to the dst buffer in the translate() call.
    */
   u_trim_pipe_prim(gen_prim, &gen_nr);

   size = index_size * gen_nr;

   dst = pipe_buffer_create(pipe->screen,
                            PIPE_BIND_INDEX_BUFFER, PIPE_USAGE_DEFAULT, size);
   if (!dst)
      goto fail;

   src_map = pipe_buffer_map(pipe, src, PIPE_TRANSFER_READ, &src_transfer);
   if (!src_map)
      goto fail;

   dst_map = pipe_buffer_map(pipe, dst, PIPE_TRANSFER_WRITE, &dst_transfer);
   if (!dst_map)
      goto fail;

   translate((const char *) src_map + offset, 0, 0, gen_nr, 0, dst_map);

   pipe_buffer_unmap(pipe, src_transfer);
   pipe_buffer_unmap(pipe, dst_transfer);

   *out_buf = dst;

   if (!screen->debug.no_cache_index_buffers) {
      /* Save the new, translated index buffer in the hope we can use it
       * again in the future.
       */
      pipe_resource_reference(&src_sbuf->translated_indices.buffer, dst);
      src_sbuf->translated_indices.orig_prim = orig_prim;
      src_sbuf->translated_indices.new_prim = gen_prim;
      src_sbuf->translated_indices.offset = offset;
      src_sbuf->translated_indices.count = orig_nr;
      src_sbuf->translated_indices.index_size = index_size;
   }

   return PIPE_OK;

 fail:
   if (src_map)
      pipe_buffer_unmap(pipe, src_transfer);

   if (dst_map)
      pipe_buffer_unmap(pipe, dst_transfer);

   if (dst)
      pipe->screen->resource_destroy(pipe->screen, dst);

   return PIPE_ERROR_OUT_OF_MEMORY;
}


enum pipe_error
svga_hwtnl_simple_draw_range_elements(struct svga_hwtnl *hwtnl,
                                      struct pipe_resource *index_buffer,
                                      unsigned index_size, int index_bias,
                                      unsigned min_index, unsigned max_index,
                                      enum pipe_prim_type prim, unsigned start,
                                      unsigned count,
                                      unsigned start_instance,
                                      unsigned instance_count)
{
   SVGA3dPrimitiveRange range;
   unsigned hw_prim;
   unsigned hw_count;
   unsigned index_offset = start * index_size;

   hw_prim = svga_translate_prim(prim, count, &hw_count);
   if (hw_count == 0)
      return PIPE_OK; /* nothing to draw */

   range.primType = hw_prim;
   range.primitiveCount = hw_count;
   range.indexArray.offset = index_offset;
   range.indexArray.stride = index_size;
   range.indexWidth = index_size;
   range.indexBias = index_bias;

   return svga_hwtnl_prim(hwtnl, &range, count,
                          min_index, max_index, index_buffer,
                          start_instance, instance_count);
}


enum pipe_error
svga_hwtnl_draw_range_elements(struct svga_hwtnl *hwtnl,
                               struct pipe_resource *index_buffer,
                               unsigned index_size, int index_bias,
                               unsigned min_index, unsigned max_index,
                               enum pipe_prim_type prim, unsigned start, unsigned count,
                               unsigned start_instance, unsigned instance_count)
{
   enum pipe_prim_type gen_prim;
   unsigned gen_size, gen_nr;
   enum indices_mode gen_type;
   u_translate_func gen_func;
   enum pipe_error ret = PIPE_OK;

   SVGA_STATS_TIME_PUSH(svga_sws(hwtnl->svga),
                        SVGA_STATS_TIME_HWTNLDRAWELEMENTS);

   if (svga_need_unfilled_fallback(hwtnl, prim)) {
      gen_type = u_unfilled_translator(prim,
                                       index_size,
                                       count,
                                       hwtnl->api_fillmode,
                                       &gen_prim,
                                       &gen_size, &gen_nr, &gen_func);
   }
   else {
      gen_type = u_index_translator(svga_hw_prims,
                                    prim,
                                    index_size,
                                    count,
                                    hwtnl->api_pv,
                                    hwtnl->hw_pv,
                                    PR_DISABLE,
                                    &gen_prim, &gen_size, &gen_nr, &gen_func);
   }

   if (gen_type == U_TRANSLATE_MEMCPY) {
      /* No need for translation, just pass through to hardware:
       */
      ret = svga_hwtnl_simple_draw_range_elements(hwtnl, index_buffer,
                                                   index_size,
                                                   index_bias,
                                                   min_index,
                                                   max_index,
                                                   gen_prim, start, count,
                                                   start_instance,
                                                   instance_count);
   }
   else {
      struct pipe_resource *gen_buf = NULL;

      /* Need to allocate a new index buffer and run the translate
       * func to populate it.  Could potentially cache this translated
       * index buffer with the original to avoid future
       * re-translations.  Not much point if we're just accelerating
       * GL though, as index buffers are typically used only once
       * there.
       */
      ret = translate_indices(hwtnl,
                              index_buffer,
                              start * index_size,
                              prim, gen_prim,
                              count, gen_nr, gen_size,
                              gen_func, &gen_buf);
      if (ret == PIPE_OK) {
         ret = svga_hwtnl_simple_draw_range_elements(hwtnl,
                                                     gen_buf,
                                                     gen_size,
                                                     index_bias,
                                                     min_index,
                                                     max_index,
                                                     gen_prim, 0, gen_nr,
                                                     start_instance,
                                                     instance_count);
      }

      if (gen_buf) {
         pipe_resource_reference(&gen_buf, NULL);
      }
   }

   SVGA_STATS_TIME_POP(svga_sws(hwtnl->svga));
   return ret;
}
