/**********************************************************
 * Copyright 2014 VMware, Inc.  All rights reserved.
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

#include "util/u_memory.h"
#include "util/u_bitmask.h"

#include "svga_cmd.h"
#include "svga_context.h"
#include "svga_resource_buffer.h"
#include "svga_shader.h"
#include "svga_debug.h"
#include "svga_streamout.h"

struct svga_stream_output_target {
   struct pipe_stream_output_target base;
};

/** cast wrapper */
static inline struct svga_stream_output_target *
svga_stream_output_target(struct pipe_stream_output_target *s)
{
   return (struct svga_stream_output_target *)s;
}

struct svga_stream_output *
svga_create_stream_output(struct svga_context *svga,
                          struct svga_shader *shader,
                          const struct pipe_stream_output_info *info)
{
   struct svga_stream_output *streamout;
   SVGA3dStreamOutputDeclarationEntry decls[SVGA3D_MAX_STREAMOUT_DECLS];
   unsigned strides[SVGA3D_DX_MAX_SOTARGETS];
   unsigned i;
   enum pipe_error ret;
   unsigned id;

   assert(info->num_outputs <= PIPE_MAX_SO_OUTPUTS);

   /* Gallium utility creates shaders with stream output.
    * For non-DX10, just return NULL.
    */
   if (!svga_have_vgpu10(svga))
      return NULL;

   assert(info->num_outputs <= SVGA3D_MAX_STREAMOUT_DECLS);

   /* Allocate an integer ID for the stream output */
   id = util_bitmask_add(svga->stream_output_id_bm);
   if (id == UTIL_BITMASK_INVALID_INDEX) {
      return NULL;
   }

   /* Allocate the streamout data structure */
   streamout = CALLOC_STRUCT(svga_stream_output);

   if (!streamout)
      return NULL;

   streamout->info = *info;
   streamout->id = id;
   streamout->pos_out_index = -1;

   SVGA_DBG(DEBUG_STREAMOUT, "%s, num_outputs=%d id=%d\n", __FUNCTION__,
            info->num_outputs, id);

   /* init whole decls and stride arrays to zero to avoid garbage values */
   memset(decls, 0, sizeof(decls));
   memset(strides, 0, sizeof(strides));

   for (i = 0; i < info->num_outputs; i++) {
      unsigned reg_idx = info->output[i].register_index;
      unsigned buf_idx = info->output[i].output_buffer;
      const enum tgsi_semantic sem_name =
         shader->info.output_semantic_name[reg_idx];

      assert(buf_idx <= PIPE_MAX_SO_BUFFERS);

      if (sem_name == TGSI_SEMANTIC_POSITION) {
         /**
          * Check if streaming out POSITION. If so, replace the
          * register index with the index for NON_ADJUSTED POSITION.
          */
         decls[i].registerIndex = shader->info.num_outputs;

         /* Save this output index, so we can tell later if this stream output
          * includes an output of a vertex position
          */
         streamout->pos_out_index = i;
      }
      else if (sem_name == TGSI_SEMANTIC_CLIPDIST) {
         /**
          * Use the shadow copy for clip distance because
          * CLIPDIST instruction is only emitted for enabled clip planes.
          * It's valid to write to ClipDistance variable for non-enabled
          * clip planes.
          */
         decls[i].registerIndex = shader->info.num_outputs + 1 +
                                  shader->info.output_semantic_index[reg_idx];
      }
      else {
         decls[i].registerIndex = reg_idx;
      }

      decls[i].outputSlot = buf_idx;
      decls[i].registerMask =
         ((1 << info->output[i].num_components) - 1)
            << info->output[i].start_component;

      SVGA_DBG(DEBUG_STREAMOUT, "%d slot=%d regIdx=%d regMask=0x%x\n",
               i, decls[i].outputSlot, decls[i].registerIndex,
               decls[i].registerMask);

      strides[buf_idx] = info->stride[buf_idx] * sizeof(float);
   }

   ret = SVGA3D_vgpu10_DefineStreamOutput(svga->swc, id,
                                          info->num_outputs,
                                          strides,
                                          decls);
   if (ret != PIPE_OK) {
      svga_context_flush(svga, NULL);
      ret = SVGA3D_vgpu10_DefineStreamOutput(svga->swc, id,
                                             info->num_outputs,
                                             strides,
                                             decls);
      if (ret != PIPE_OK) {
         util_bitmask_clear(svga->stream_output_id_bm, id);
         FREE(streamout);
         streamout = NULL;
      }
   }
   return streamout;
}

enum pipe_error
svga_set_stream_output(struct svga_context *svga,
                       struct svga_stream_output *streamout)
{
   unsigned id = streamout ? streamout->id : SVGA3D_INVALID_ID;

   if (!svga_have_vgpu10(svga)) {
      return PIPE_OK;
   }

   SVGA_DBG(DEBUG_STREAMOUT, "%s streamout=0x%x id=%d\n", __FUNCTION__,
            streamout, id);

   if (svga->current_so != streamout) {
      enum pipe_error ret = SVGA3D_vgpu10_SetStreamOutput(svga->swc, id);
      if (ret != PIPE_OK) {
         return ret;
      }

      svga->current_so = streamout;
   }

   return PIPE_OK;
}

void
svga_delete_stream_output(struct svga_context *svga,
                          struct svga_stream_output *streamout)
{
   enum pipe_error ret;

   SVGA_DBG(DEBUG_STREAMOUT, "%s streamout=0x%x\n", __FUNCTION__, streamout);

   assert(svga_have_vgpu10(svga));
   assert(streamout != NULL);

   ret = SVGA3D_vgpu10_DestroyStreamOutput(svga->swc, streamout->id);
   if (ret != PIPE_OK) {
      svga_context_flush(svga, NULL);
      ret = SVGA3D_vgpu10_DestroyStreamOutput(svga->swc, streamout->id);
   }

   /* Release the ID */
   util_bitmask_clear(svga->stream_output_id_bm, streamout->id);

   /* Free streamout structure */
   FREE(streamout);
}

static struct pipe_stream_output_target *
svga_create_stream_output_target(struct pipe_context *pipe,
                                 struct pipe_resource *buffer,
                                 unsigned buffer_offset,
                                 unsigned buffer_size)
{
   struct svga_context *svga = svga_context(pipe);
   struct svga_stream_output_target *sot;

   SVGA_DBG(DEBUG_STREAMOUT, "%s offset=%d size=%d\n", __FUNCTION__,
            buffer_offset, buffer_size);

   assert(svga_have_vgpu10(svga));
   (void) svga;

   sot = CALLOC_STRUCT(svga_stream_output_target);
   if (!sot)
      return NULL;

   pipe_reference_init(&sot->base.reference, 1);
   pipe_resource_reference(&sot->base.buffer, buffer);
   sot->base.context = pipe;
   sot->base.buffer = buffer;
   sot->base.buffer_offset = buffer_offset;
   sot->base.buffer_size = buffer_size;

   return &sot->base;
}

static void
svga_destroy_stream_output_target(struct pipe_context *pipe,
                                  struct pipe_stream_output_target *target)
{
   struct svga_stream_output_target *sot = svga_stream_output_target(target);

   SVGA_DBG(DEBUG_STREAMOUT, "%s\n", __FUNCTION__);

   pipe_resource_reference(&sot->base.buffer, NULL);
   FREE(sot);
}

static void
svga_set_stream_output_targets(struct pipe_context *pipe,
                               unsigned num_targets,
                               struct pipe_stream_output_target **targets,
                               const unsigned *offsets)
{
   struct svga_context *svga = svga_context(pipe);
   struct SVGA3dSoTarget soBindings[SVGA3D_DX_MAX_SOTARGETS];
   enum pipe_error ret;
   unsigned i;
   unsigned num_so_targets;

   SVGA_DBG(DEBUG_STREAMOUT, "%s num_targets=%d\n", __FUNCTION__,
            num_targets);

   assert(svga_have_vgpu10(svga));

   /* Mark the streamout buffers as dirty so that we'll issue readbacks
    * before mapping.
    */
   for (i = 0; i < svga->num_so_targets; i++) {
      struct svga_buffer *sbuf = svga_buffer(svga->so_targets[i]->buffer);
      sbuf->dirty = TRUE;
   }

   assert(num_targets <= SVGA3D_DX_MAX_SOTARGETS);

   for (i = 0; i < num_targets; i++) {
      struct svga_stream_output_target *sot
         = svga_stream_output_target(targets[i]);
      unsigned size;

      svga->so_surfaces[i] = svga_buffer_handle(svga, sot->base.buffer,
                                                PIPE_BIND_STREAM_OUTPUT);

      assert(svga_buffer(sot->base.buffer)->key.flags
             & SVGA3D_SURFACE_BIND_STREAM_OUTPUT);

      svga->so_targets[i] = &sot->base;
      soBindings[i].offset = sot->base.buffer_offset;

      /* The size cannot extend beyond the end of the buffer.  Clamp it. */
      size = MIN2(sot->base.buffer_size,
                  sot->base.buffer->width0 - sot->base.buffer_offset);

      soBindings[i].sizeInBytes = size;
   }

   /* unbind any previously bound stream output buffers */
   for (; i < svga->num_so_targets; i++) {
      svga->so_surfaces[i] = NULL;
      svga->so_targets[i] = NULL;
   }

   num_so_targets = MAX2(svga->num_so_targets, num_targets);
   ret = SVGA3D_vgpu10_SetSOTargets(svga->swc, num_so_targets,
                                    soBindings, svga->so_surfaces);
   if (ret != PIPE_OK) {
      svga_context_flush(svga, NULL);
      ret = SVGA3D_vgpu10_SetSOTargets(svga->swc, num_so_targets,
                                       soBindings, svga->so_surfaces);
   }

   svga->num_so_targets = num_targets;
}

/**
 * Rebind stream output target surfaces
 */
enum pipe_error
svga_rebind_stream_output_targets(struct svga_context *svga)
{
   struct svga_winsys_context *swc = svga->swc;
   enum pipe_error ret;
   unsigned i;

   for (i = 0; i < svga->num_so_targets; i++) {
      ret = swc->resource_rebind(swc, svga->so_surfaces[i], NULL, SVGA_RELOC_WRITE);
      if (ret != PIPE_OK)
         return ret;
   }

   return PIPE_OK;
}

void
svga_init_stream_output_functions(struct svga_context *svga)
{
   svga->pipe.create_stream_output_target = svga_create_stream_output_target;
   svga->pipe.stream_output_target_destroy = svga_destroy_stream_output_target;
   svga->pipe.set_stream_output_targets = svga_set_stream_output_targets;
}
