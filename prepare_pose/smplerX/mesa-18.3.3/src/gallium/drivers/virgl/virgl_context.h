/*
 * Copyright 2014, 2015 Red Hat.
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
#ifndef VIRGL_CONTEXT_H
#define VIRGL_CONTEXT_H

#include "pipe/p_state.h"
#include "pipe/p_context.h"
#include "util/slab.h"
#include "util/list.h"

struct pipe_screen;
struct tgsi_token;
struct u_upload_mgr;
struct virgl_cmd_buf;
struct virgl_vertex_elements_state;

struct virgl_sampler_view {
   struct pipe_sampler_view base;
   uint32_t handle;
};

struct virgl_so_target {
   struct pipe_stream_output_target base;
   uint32_t handle;
};

struct virgl_textures_info {
   struct virgl_sampler_view *views[16];
   uint32_t enabled_mask;
};

struct virgl_context {
   struct pipe_context base;
   struct virgl_cmd_buf *cbuf;

   struct virgl_textures_info samplers[PIPE_SHADER_TYPES];
   struct virgl_vertex_elements_state *vertex_elements;

   struct pipe_framebuffer_state framebuffer;

   struct slab_child_pool texture_transfer_pool;

   struct u_upload_mgr *uploader;

   struct pipe_vertex_buffer vertex_buffer[PIPE_MAX_ATTRIBS];
   unsigned num_vertex_buffers;
   boolean vertex_array_dirty;

   struct virgl_so_target so_targets[PIPE_MAX_SO_BUFFERS];
   unsigned num_so_targets;

   struct pipe_resource *ubos[PIPE_SHADER_TYPES][PIPE_MAX_CONSTANT_BUFFERS];

   struct pipe_resource *ssbos[PIPE_SHADER_TYPES][PIPE_MAX_SHADER_BUFFERS];
   struct pipe_resource *images[PIPE_SHADER_TYPES][PIPE_MAX_SHADER_BUFFERS];
   int num_transfers;
   int num_draws;
   struct list_head to_flush_bufs;

   struct pipe_resource *atomic_buffers[PIPE_MAX_HW_ATOMIC_BUFFERS];

   struct primconvert_context *primconvert;
   uint32_t hw_sub_ctx_id;
};

static inline struct virgl_sampler_view *
virgl_sampler_view(struct pipe_sampler_view *view)
{
   return (struct virgl_sampler_view *)view;
};

static inline struct virgl_so_target *
virgl_so_target(struct pipe_stream_output_target *target)
{
   return (struct virgl_so_target *)target;
}

static inline struct virgl_context *virgl_context(struct pipe_context *ctx)
{
   return (struct virgl_context *)ctx;
}

struct pipe_context *virgl_context_create(struct pipe_screen *pscreen,
                                          void *priv, unsigned flags);

void virgl_init_blit_functions(struct virgl_context *vctx);
void virgl_init_query_functions(struct virgl_context *vctx);
void virgl_init_so_functions(struct virgl_context *vctx);

void virgl_transfer_inline_write(struct pipe_context *ctx,
                                struct pipe_resource *res,
                                unsigned level,
                                unsigned usage,
                                const struct pipe_box *box,
                                const void *data,
                                unsigned stride,
                                unsigned layer_stride);

struct tgsi_token *virgl_tgsi_transform(struct virgl_context *vctx, const struct tgsi_token *tokens_in);

#endif
