
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

#include "util/u_format.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "pipe/p_defines.h"
#include "util/u_upload_mgr.h"

#include "svga_screen.h"
#include "svga_context.h"
#include "svga_state.h"
#include "svga_cmd.h"
#include "svga_tgsi.h"
#include "svga_debug.h"
#include "svga_resource_buffer.h"
#include "svga_shader.h"

#include "svga_hw_reg.h"


/*
 * Don't try to send more than 4kb of successive constants.
 */
#define MAX_CONST_REG_COUNT 256  /**< number of float[4] constants */

/**
 * Extra space for svga-specific VS/PS constants (such as texcoord
 * scale factors, vertex transformation scale/translation).
 */
#define MAX_EXTRA_CONSTS 32

/** Guest-backed surface constant buffers must be this size */
#define GB_CONSTBUF_SIZE (SVGA3D_CONSTREG_MAX)


/**
 * Emit any extra shader-type-independent shader constants into the buffer
 * pointed to by 'dest'.
 * \return number of float[4] constants put into the 'dest' buffer
 */
static unsigned
svga_get_extra_constants_common(const struct svga_context *svga,
                                const struct svga_shader_variant *variant,
                                enum pipe_shader_type shader, float *dest)
{
   uint32_t *dest_u = (uint32_t *) dest;  // uint version of dest
   unsigned i;
   unsigned count = 0;

   for (i = 0; i < variant->key.num_textures; i++) {
      const struct pipe_sampler_view *sv = svga->curr.sampler_views[shader][i];
      if (sv) {
         const struct pipe_resource *tex = sv->texture;
         /* Scaling factors needed for handling unnormalized texture coordinates
          * for texture rectangles.
          */
         if (variant->key.tex[i].unnormalized) {
            /* debug/sanity check */
            assert(variant->key.tex[i].width_height_idx == count);

            *dest++ = 1.0f / (float) tex->width0;
            *dest++ = 1.0f / (float) tex->height0;
            *dest++ = 1.0f;
            *dest++ = 1.0f;

            count++;
         }

         /* Store the sizes for texture buffers.
         */
         if (tex->target == PIPE_BUFFER) {
            unsigned bytes_per_element = util_format_get_blocksize(sv->format);
            *dest_u++ = tex->width0 / bytes_per_element;
            *dest_u++ = 1;
            *dest_u++ = 1;
            *dest_u++ = 1;

            count++;
         }
      }
   }

   return count;
}


/**
 * Emit any extra fragment shader constants into the buffer pointed
 * to by 'dest'.
 * \return number of float[4] constants put into the dest buffer
 */
static unsigned
svga_get_extra_fs_constants(const struct svga_context *svga, float *dest)
{
   const struct svga_shader_variant *variant = svga->state.hw_draw.fs;
   unsigned count = 0;

   count += svga_get_extra_constants_common(svga, variant,
                                            PIPE_SHADER_FRAGMENT, dest);

   assert(count <= MAX_EXTRA_CONSTS);

   return count;
}

/**
 * Emit extra constants needed for prescale computation into the
 * the buffer pointed to by '*dest'. The updated buffer pointer
 * will be returned in 'dest'.
 */
static unsigned
svga_get_prescale_constants(const struct svga_context *svga, float **dest)
{
   memcpy(*dest, svga->state.hw_clear.prescale.scale, 4 * sizeof(float));
   *dest += 4;

   memcpy(*dest, svga->state.hw_clear.prescale.translate, 4 * sizeof(float));
   *dest += 4;

   return 2;
}

/**
 * Emit extra constants needed for point sprite emulation.
 */
static unsigned
svga_get_pt_sprite_constants(const struct svga_context *svga, float **dest)
{
   const struct svga_screen *screen = svga_screen(svga->pipe.screen);
   float *dst = *dest;

   dst[0] = 1.0 / (svga->curr.viewport.scale[0] * 2);
   dst[1] = 1.0 / (svga->curr.viewport.scale[1] * 2);
   dst[2] = svga->curr.rast->pointsize;
   dst[3] = screen->maxPointSize;
   *dest = *dest + 4;
   return 1;
}

/**
 * Emit user-defined clip plane coefficients into the buffer pointed to
 * by '*dest'. The updated buffer pointer will be returned in 'dest'.
 */
static unsigned
svga_get_clip_plane_constants(const struct svga_context *svga,
                              const struct svga_shader_variant *variant,
                              float **dest)
{
   unsigned count = 0;

   /* SVGA_NEW_CLIP */
   if (svga_have_vgpu10(svga)) {
      /* append user-defined clip plane coefficients onto constant buffer */
      unsigned clip_planes = variant->key.clip_plane_enable;
      while (clip_planes) {
         int i = u_bit_scan(&clip_planes);
         COPY_4V(*dest, svga->curr.clip.ucp[i]);
         *dest += 4;
         count += 1;
      }
   }
   return count;
}

/**
 * Emit any extra vertex shader constants into the buffer pointed
 * to by 'dest'.
 * In particular, these would be the scale and bias factors computed
 * from the framebuffer size which are used to copy with differences in
 * GL vs D3D coordinate spaces.  See svga_tgsi_insn.c for more info.
 * \return number of float[4] constants put into the dest buffer
 */
static unsigned
svga_get_extra_vs_constants(const struct svga_context *svga, float *dest)
{
   const struct svga_shader_variant *variant = svga->state.hw_draw.vs;
   unsigned count = 0;

   /* SVGA_NEW_VS_VARIANT
    */
   if (variant->key.vs.need_prescale) {
      count += svga_get_prescale_constants(svga, &dest);
   }

   if (variant->key.vs.undo_viewport) {
      /* Used to convert window coords back to NDC coords */
      dest[0] = 1.0f / svga->curr.viewport.scale[0];
      dest[1] = 1.0f / svga->curr.viewport.scale[1];
      dest[2] = -svga->curr.viewport.translate[0];
      dest[3] = -svga->curr.viewport.translate[1];
      dest += 4;
      count += 1;
   }

   /* SVGA_NEW_CLIP */
   count += svga_get_clip_plane_constants(svga, variant, &dest);

   /* common constants */
   count += svga_get_extra_constants_common(svga, variant,
                                            PIPE_SHADER_VERTEX, dest);

   assert(count <= MAX_EXTRA_CONSTS);

   return count;
}

/**
 * Emit any extra geometry shader constants into the buffer pointed
 * to by 'dest'.
 */
static unsigned
svga_get_extra_gs_constants(const struct svga_context *svga, float *dest)
{
   const struct svga_shader_variant *variant = svga->state.hw_draw.gs;
   unsigned count = 0;

   /* SVGA_NEW_GS_VARIANT
    */

   /* Constants for point sprite
    * These are used in the transformed gs that supports point sprite.
    * They need to be added before the prescale constants.
    */
   if (variant->key.gs.wide_point) {
      count += svga_get_pt_sprite_constants(svga, &dest);
   }

   if (variant->key.gs.need_prescale) {
      count += svga_get_prescale_constants(svga, &dest);
   }

   /* SVGA_NEW_CLIP */
   count += svga_get_clip_plane_constants(svga, variant, &dest);

   /* common constants */
   count += svga_get_extra_constants_common(svga, variant,
                                            PIPE_SHADER_GEOMETRY, dest);

   assert(count <= MAX_EXTRA_CONSTS);
   return count;
}


/*
 * Check and emit a range of shader constant registers, trying to coalesce
 * successive shader constant updates in a single command in order to save
 * space on the command buffer.  This is a HWv8 feature.
 */
static enum pipe_error
emit_const_range(struct svga_context *svga,
                 enum pipe_shader_type shader,
                 unsigned offset,
                 unsigned count,
                 const float (*values)[4])
{
   unsigned i, j;
   enum pipe_error ret;

   assert(shader == PIPE_SHADER_VERTEX ||
          shader == PIPE_SHADER_FRAGMENT);
   assert(!svga_have_vgpu10(svga));

#ifdef DEBUG
   if (offset + count > SVGA3D_CONSTREG_MAX) {
      debug_printf("svga: too many constants (offset %u + count %u = %u (max = %u))\n",
                   offset, count, offset + count, SVGA3D_CONSTREG_MAX);
   }
#endif

   if (offset > SVGA3D_CONSTREG_MAX) {
      /* This isn't OK, but if we propagate an error all the way up we'll
       * just get into more trouble.
       * XXX note that offset is always zero at this time so this is moot.
       */
      return PIPE_OK;
   }

   if (offset + count > SVGA3D_CONSTREG_MAX) {
      /* Just drop the extra constants for now.
       * Ideally we should not have allowed the app to create a shader
       * that exceeds our constant buffer size but there's no way to
       * express that in gallium at this time.
       */
      count = SVGA3D_CONSTREG_MAX - offset;
   }

   i = 0;
   while (i < count) {
      if (memcmp(svga->state.hw_draw.cb[shader][offset + i],
                 values[i],
                 4 * sizeof(float)) != 0) {
         /* Found one dirty constant
          */
         if (SVGA_DEBUG & DEBUG_CONSTS)
            debug_printf("%s %s %d: %f %f %f %f\n",
                         __FUNCTION__,
                         shader == PIPE_SHADER_VERTEX ? "VERT" : "FRAG",
                         offset + i,
                         values[i][0],
                         values[i][1],
                         values[i][2],
                         values[i][3]);

         /* Look for more consecutive dirty constants.
          */
         j = i + 1;
         while (j < count &&
                j < i + MAX_CONST_REG_COUNT &&
                memcmp(svga->state.hw_draw.cb[shader][offset + j],
                       values[j],
                       4 * sizeof(float)) != 0) {

            if (SVGA_DEBUG & DEBUG_CONSTS)
               debug_printf("%s %s %d: %f %f %f %f\n",
                            __FUNCTION__,
                            shader == PIPE_SHADER_VERTEX ? "VERT" : "FRAG",
                            offset + j,
                            values[j][0],
                            values[j][1],
                            values[j][2],
                            values[j][3]);

            ++j;
         }

         assert(j >= i + 1);

         /* Send them all together.
          */
         if (svga_have_gb_objects(svga)) {
            ret = SVGA3D_SetGBShaderConstsInline(svga->swc,
                                                 offset + i, /* start */
                                                 j - i,  /* count */
                                                 svga_shader_type(shader),
                                                 SVGA3D_CONST_TYPE_FLOAT,
                                                 values + i);
         }
         else {
            ret = SVGA3D_SetShaderConsts(svga->swc,
                                         offset + i, j - i,
                                         svga_shader_type(shader),
                                         SVGA3D_CONST_TYPE_FLOAT,
                                         values + i);
         }
         if (ret != PIPE_OK) {
            return ret;
         }

         /*
          * Local copy of the hardware state.
          */
         memcpy(svga->state.hw_draw.cb[shader][offset + i],
                values[i],
                (j - i) * 4 * sizeof(float));

         i = j + 1;

         svga->hud.num_const_updates++;

      } else {
         ++i;
      }
   }

   return PIPE_OK;
}


/**
 * Emit all the constants in a constant buffer for a shader stage.
 * On VGPU10, emit_consts_vgpu10 is used instead.
 */
static enum pipe_error
emit_consts_vgpu9(struct svga_context *svga, enum pipe_shader_type shader)
{
   const struct pipe_constant_buffer *cbuf;
   struct pipe_transfer *transfer = NULL;
   unsigned count;
   const float (*data)[4] = NULL;
   enum pipe_error ret = PIPE_OK;
   const unsigned offset = 0;

   assert(shader < PIPE_SHADER_TYPES);
   assert(!svga_have_vgpu10(svga));
   /* Only one constant buffer per shader is supported before VGPU10.
    * This is only an approximate check against that.
    */
   assert(svga->curr.constbufs[shader][1].buffer == NULL);

   cbuf = &svga->curr.constbufs[shader][0];

   if (svga->curr.constbufs[shader][0].buffer) {
      /* emit user-provided constants */
      data = (const float (*)[4])
         pipe_buffer_map(&svga->pipe, svga->curr.constbufs[shader][0].buffer,
                         PIPE_TRANSFER_READ, &transfer);
      if (!data) {
         return PIPE_ERROR_OUT_OF_MEMORY;
      }

      /* sanity check */
      assert(cbuf->buffer->width0 >= cbuf->buffer_size);

      /* Use/apply the constant buffer size and offsets here */
      count = cbuf->buffer_size / (4 * sizeof(float));
      data += cbuf->buffer_offset / (4 * sizeof(float));

      ret = emit_const_range( svga, shader, offset, count, data );

      pipe_buffer_unmap(&svga->pipe, transfer);

      if (ret != PIPE_OK) {
         return ret;
      }
   }

   /* emit extra shader constants */
   {
      const struct svga_shader_variant *variant = NULL;
      unsigned offset;
      float extras[MAX_EXTRA_CONSTS][4];
      unsigned count;

      switch (shader) {
      case PIPE_SHADER_VERTEX:
         variant = svga->state.hw_draw.vs;
         count = svga_get_extra_vs_constants(svga, (float *) extras);
         break;
      case PIPE_SHADER_FRAGMENT:
         variant = svga->state.hw_draw.fs;
         count = svga_get_extra_fs_constants(svga, (float *) extras);
         break;
      default:
         assert(!"Unexpected shader type");
         count = 0;
      }

      assert(variant);
      offset = variant->shader->info.file_max[TGSI_FILE_CONSTANT] + 1;
      assert(count <= ARRAY_SIZE(extras));

      if (count > 0) {
         ret = emit_const_range(svga, shader, offset, count,
                                (const float (*) [4])extras);
      }
   }

   return ret;
}



static enum pipe_error
emit_constbuf_vgpu10(struct svga_context *svga, enum pipe_shader_type shader)
{
   const struct pipe_constant_buffer *cbuf;
   struct pipe_resource *dst_buffer = NULL;
   enum pipe_error ret = PIPE_OK;
   struct pipe_transfer *src_transfer;
   struct svga_winsys_surface *dst_handle;
   float extras[MAX_EXTRA_CONSTS][4];
   unsigned extra_count, extra_size, extra_offset;
   unsigned new_buf_size;
   void *src_map = NULL, *dst_map;
   unsigned offset;
   const struct svga_shader_variant *variant;
   unsigned alloc_buf_size;

   switch (shader) {
   case PIPE_SHADER_VERTEX:
      variant = svga->state.hw_draw.vs;
      extra_count = svga_get_extra_vs_constants(svga, (float *) extras);
      break;
   case PIPE_SHADER_FRAGMENT:
      variant = svga->state.hw_draw.fs;
      extra_count = svga_get_extra_fs_constants(svga, (float *) extras);
      break;
   case PIPE_SHADER_GEOMETRY:
      variant = svga->state.hw_draw.gs;
      extra_count = svga_get_extra_gs_constants(svga, (float *) extras);
      break;
   default:
      assert(!"Unexpected shader type");
      /* Don't return an error code since we don't want to keep re-trying
       * this function and getting stuck in an infinite loop.
       */
      return PIPE_OK;
   }

   assert(variant);

   cbuf = &svga->curr.constbufs[shader][0];

   /* Compute extra constants size and offset in bytes */
   extra_size = extra_count * 4 * sizeof(float);
   extra_offset = 4 * sizeof(float) * variant->extra_const_start;

   if (cbuf->buffer_size + extra_size == 0)
      return PIPE_OK;  /* nothing to do */

   /* Typically, the cbuf->buffer here is a user-space buffer so mapping
    * it is really cheap.  If we ever get real HW buffers for constants
    * we should void mapping and instead use a ResourceCopy command.
    */
   if (cbuf->buffer_size > 0) {
      src_map = pipe_buffer_map_range(&svga->pipe, cbuf->buffer,
                                      cbuf->buffer_offset, cbuf->buffer_size,
                                      PIPE_TRANSFER_READ, &src_transfer);
      assert(src_map);
      if (!src_map) {
         return PIPE_ERROR_OUT_OF_MEMORY;
      }
   }

   /* The new/dest buffer's size must be large enough to hold the original,
    * user-specified constants, plus the extra constants.
    * The size of the original constant buffer _should_ agree with what the
    * shader is expecting, but it might not (it's not enforced anywhere by
    * gallium).
    */
   new_buf_size = MAX2(cbuf->buffer_size, extra_offset) + extra_size;

   /* According to the DX10 spec, the constant buffer size must be
    * in multiples of 16.
    */
   new_buf_size = align(new_buf_size, 16);

   /* Constant buffer size in the upload buffer must be in multiples of 256.
    * In order to maximize the chance of merging the upload buffer chunks
    * when svga_buffer_add_range() is called,
    * the allocate buffer size needs to be in multiples of 256 as well.
    * Otherwise, since there is gap between each dirty range of the upload buffer,
    * each dirty range will end up in its own UPDATE_GB_IMAGE command.
    */
   alloc_buf_size = align(new_buf_size, CONST0_UPLOAD_ALIGNMENT);

   u_upload_alloc(svga->const0_upload, 0, alloc_buf_size,
                  CONST0_UPLOAD_ALIGNMENT, &offset,
                  &dst_buffer, &dst_map);
   if (!dst_map) {
      if (src_map)
         pipe_buffer_unmap(&svga->pipe, src_transfer);
      return PIPE_ERROR_OUT_OF_MEMORY;
   }

   if (src_map) {
      memcpy(dst_map, src_map, cbuf->buffer_size);
      pipe_buffer_unmap(&svga->pipe, src_transfer);
   }

   if (extra_size) {
      assert(extra_offset + extra_size <= new_buf_size);
      memcpy((char *) dst_map + extra_offset, extras, extra_size);
   }

   /* Get winsys handle for the constant buffer */
   if (svga->state.hw_draw.const0_buffer == dst_buffer &&
       svga->state.hw_draw.const0_handle) {
      /* re-reference already mapped buffer */
      dst_handle = svga->state.hw_draw.const0_handle;
   }
   else {
      /* we must unmap the buffer before getting the winsys handle */
      u_upload_unmap(svga->const0_upload);

      dst_handle = svga_buffer_handle(svga, dst_buffer,
                                      PIPE_BIND_CONSTANT_BUFFER);
      if (!dst_handle) {
         pipe_resource_reference(&dst_buffer, NULL);
         return PIPE_ERROR_OUT_OF_MEMORY;
      }

      /* save the buffer / handle for next time */
      pipe_resource_reference(&svga->state.hw_draw.const0_buffer, dst_buffer);
      svga->state.hw_draw.const0_handle = dst_handle;
   }

   /* Issue the SetSingleConstantBuffer command */
   assert(new_buf_size % 16 == 0);
   ret = SVGA3D_vgpu10_SetSingleConstantBuffer(svga->swc,
                                               0, /* index */
                                               svga_shader_type(shader),
                                               dst_handle,
                                               offset,
                                               new_buf_size);

   if (ret != PIPE_OK) {
      pipe_resource_reference(&dst_buffer, NULL);
      return ret;
   }

   /* Save this const buffer until it's replaced in the future.
    * Otherwise, all references to the buffer will go away after the
    * command buffer is submitted, it'll get recycled and we will have
    * incorrect constant buffer bindings.
    */
   pipe_resource_reference(&svga->state.hw_draw.constbuf[shader], dst_buffer);

   svga->state.hw_draw.default_constbuf_size[shader] = new_buf_size;

   pipe_resource_reference(&dst_buffer, NULL);

   svga->hud.num_const_buf_updates++;

   return ret;
}


static enum pipe_error
emit_consts_vgpu10(struct svga_context *svga, enum pipe_shader_type shader)
{
   enum pipe_error ret;
   unsigned dirty_constbufs;
   unsigned enabled_constbufs;

   /* Emit 0th constant buffer (with extra constants) */
   ret = emit_constbuf_vgpu10(svga, shader);
   if (ret != PIPE_OK) {
      return ret;
   }

   enabled_constbufs = svga->state.hw_draw.enabled_constbufs[shader] | 1u;

   /* Emit other constant buffers (UBOs) */
   dirty_constbufs = svga->state.dirty_constbufs[shader] & ~1u;

   while (dirty_constbufs) {
      unsigned index = u_bit_scan(&dirty_constbufs);
      unsigned offset = svga->curr.constbufs[shader][index].buffer_offset;
      unsigned size = svga->curr.constbufs[shader][index].buffer_size;
      struct svga_buffer *buffer =
         svga_buffer(svga->curr.constbufs[shader][index].buffer);
      struct svga_winsys_surface *handle;

      if (buffer) {
         handle = svga_buffer_handle(svga, &buffer->b.b,
                                     PIPE_BIND_CONSTANT_BUFFER);
         enabled_constbufs |= 1 << index;
      }
      else {
         handle = NULL;
         enabled_constbufs &= ~(1 << index);
         assert(offset == 0);
         assert(size == 0);
      }

      if (size % 16 != 0) {
         /* GL's buffer range sizes can be any number of bytes but the
          * SVGA3D device requires a multiple of 16 bytes.
          */
         const unsigned total_size = buffer->b.b.width0;

         if (offset + align(size, 16) <= total_size) {
            /* round up size to multiple of 16 */
            size = align(size, 16);
         }
         else {
            /* round down to mulitple of 16 (this may cause rendering problems
             * but should avoid a device error).
             */
            size &= ~15;
         }
      }

      assert(size % 16 == 0);
      ret = SVGA3D_vgpu10_SetSingleConstantBuffer(svga->swc,
                                                  index,
                                                  svga_shader_type(shader),
                                                  handle,
                                                  offset,
                                                  size);
      if (ret != PIPE_OK)
         return ret;

      svga->hud.num_const_buf_updates++;
   }

   svga->state.hw_draw.enabled_constbufs[shader] = enabled_constbufs;
   svga->state.dirty_constbufs[shader] = 0;

   return ret;
}

static enum pipe_error
emit_fs_consts(struct svga_context *svga, unsigned dirty)
{
   const struct svga_shader_variant *variant = svga->state.hw_draw.fs;
   enum pipe_error ret = PIPE_OK;

   /* SVGA_NEW_FS_VARIANT
    */
   if (!variant)
      return PIPE_OK;

   /* SVGA_NEW_FS_CONST_BUFFER
    */
   if (svga_have_vgpu10(svga)) {
      ret = emit_consts_vgpu10(svga, PIPE_SHADER_FRAGMENT);
   }
   else {
      ret = emit_consts_vgpu9(svga, PIPE_SHADER_FRAGMENT);
   }

   return ret;
}


struct svga_tracked_state svga_hw_fs_constants =
{
   "hw fs params",
   (SVGA_NEW_FS_CONST_BUFFER |
    SVGA_NEW_FS_VARIANT |
    SVGA_NEW_TEXTURE_CONSTS),
   emit_fs_consts
};



static enum pipe_error
emit_vs_consts(struct svga_context *svga, unsigned dirty)
{
   const struct svga_shader_variant *variant = svga->state.hw_draw.vs;
   enum pipe_error ret = PIPE_OK;

   /* SVGA_NEW_VS_VARIANT
    */
   if (!variant)
      return PIPE_OK;

   /* SVGA_NEW_VS_CONST_BUFFER
    */
   if (svga_have_vgpu10(svga)) {
      ret = emit_consts_vgpu10(svga, PIPE_SHADER_VERTEX);
   }
   else {
      ret = emit_consts_vgpu9(svga, PIPE_SHADER_VERTEX);
   }

   return ret;
}


struct svga_tracked_state svga_hw_vs_constants =
{
   "hw vs params",
   (SVGA_NEW_PRESCALE |
    SVGA_NEW_VS_CONST_BUFFER |
    SVGA_NEW_VS_VARIANT |
    SVGA_NEW_TEXTURE_CONSTS),
   emit_vs_consts
};


static enum pipe_error
emit_gs_consts(struct svga_context *svga, unsigned dirty)
{
   const struct svga_shader_variant *variant = svga->state.hw_draw.gs;
   enum pipe_error ret = PIPE_OK;

   /* SVGA_NEW_GS_VARIANT
    */
   if (!variant)
      return PIPE_OK;

   /* SVGA_NEW_GS_CONST_BUFFER
    */
   if (svga_have_vgpu10(svga)) {
      /**
       * If only the rasterizer state has changed and the current geometry
       * shader does not emit wide points, then there is no reason to
       * re-emit the GS constants, so skip it.
       */
      if (dirty == SVGA_NEW_RAST && !variant->key.gs.wide_point)
         return PIPE_OK;

      ret = emit_consts_vgpu10(svga, PIPE_SHADER_GEOMETRY);
   }

   return ret;
}


struct svga_tracked_state svga_hw_gs_constants =
{
   "hw gs params",
   (SVGA_NEW_PRESCALE |
    SVGA_NEW_GS_CONST_BUFFER |
    SVGA_NEW_RAST |
    SVGA_NEW_GS_VARIANT |
    SVGA_NEW_TEXTURE_CONSTS),
   emit_gs_consts
};
