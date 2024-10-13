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
#include <stdint.h>
#include <assert.h>
#include <string.h>

#include "util/u_format.h"
#include "util/u_memory.h"
#include "util/u_math.h"
#include "pipe/p_state.h"
#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_parse.h"

#include "virgl_context.h"
#include "virgl_encode.h"
#include "virgl_protocol.h"
#include "virgl_resource.h"
#include "virgl_screen.h"

static int virgl_encoder_write_cmd_dword(struct virgl_context *ctx,
                                        uint32_t dword)
{
   int len = (dword >> 16);

   if ((ctx->cbuf->cdw + len + 1) > VIRGL_MAX_CMDBUF_DWORDS)
      ctx->base.flush(&ctx->base, NULL, 0);

   virgl_encoder_write_dword(ctx->cbuf, dword);
   return 0;
}

static void virgl_encoder_write_res(struct virgl_context *ctx,
                                    struct virgl_resource *res)
{
   struct virgl_winsys *vws = virgl_screen(ctx->base.screen)->vws;

   if (res && res->hw_res)
      vws->emit_res(vws, ctx->cbuf, res->hw_res, TRUE);
   else {
      virgl_encoder_write_dword(ctx->cbuf, 0);
   }
}

static void virgl_dirty_res(struct virgl_resource *res)
{
   if (res)
      res->clean = FALSE;
}

int virgl_encode_bind_object(struct virgl_context *ctx,
                            uint32_t handle, uint32_t object)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_BIND_OBJECT, object, 1));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   return 0;
}

int virgl_encode_delete_object(struct virgl_context *ctx,
                              uint32_t handle, uint32_t object)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_DESTROY_OBJECT, object, 1));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   return 0;
}

int virgl_encode_blend_state(struct virgl_context *ctx,
                            uint32_t handle,
                            const struct pipe_blend_state *blend_state)
{
   uint32_t tmp;
   int i;

   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_BLEND, VIRGL_OBJ_BLEND_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, handle);

   tmp =
      VIRGL_OBJ_BLEND_S0_INDEPENDENT_BLEND_ENABLE(blend_state->independent_blend_enable) |
      VIRGL_OBJ_BLEND_S0_LOGICOP_ENABLE(blend_state->logicop_enable) |
      VIRGL_OBJ_BLEND_S0_DITHER(blend_state->dither) |
      VIRGL_OBJ_BLEND_S0_ALPHA_TO_COVERAGE(blend_state->alpha_to_coverage) |
      VIRGL_OBJ_BLEND_S0_ALPHA_TO_ONE(blend_state->alpha_to_one);

   virgl_encoder_write_dword(ctx->cbuf, tmp);

   tmp = VIRGL_OBJ_BLEND_S1_LOGICOP_FUNC(blend_state->logicop_func);
   virgl_encoder_write_dword(ctx->cbuf, tmp);

   for (i = 0; i < VIRGL_MAX_COLOR_BUFS; i++) {
      tmp =
         VIRGL_OBJ_BLEND_S2_RT_BLEND_ENABLE(blend_state->rt[i].blend_enable) |
         VIRGL_OBJ_BLEND_S2_RT_RGB_FUNC(blend_state->rt[i].rgb_func) |
         VIRGL_OBJ_BLEND_S2_RT_RGB_SRC_FACTOR(blend_state->rt[i].rgb_src_factor) |
         VIRGL_OBJ_BLEND_S2_RT_RGB_DST_FACTOR(blend_state->rt[i].rgb_dst_factor)|
         VIRGL_OBJ_BLEND_S2_RT_ALPHA_FUNC(blend_state->rt[i].alpha_func) |
         VIRGL_OBJ_BLEND_S2_RT_ALPHA_SRC_FACTOR(blend_state->rt[i].alpha_src_factor) |
         VIRGL_OBJ_BLEND_S2_RT_ALPHA_DST_FACTOR(blend_state->rt[i].alpha_dst_factor) |
         VIRGL_OBJ_BLEND_S2_RT_COLORMASK(blend_state->rt[i].colormask);
      virgl_encoder_write_dword(ctx->cbuf, tmp);
   }
   return 0;
}

int virgl_encode_dsa_state(struct virgl_context *ctx,
                          uint32_t handle,
                          const struct pipe_depth_stencil_alpha_state *dsa_state)
{
   uint32_t tmp;
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_DSA, VIRGL_OBJ_DSA_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, handle);

   tmp = VIRGL_OBJ_DSA_S0_DEPTH_ENABLE(dsa_state->depth.enabled) |
      VIRGL_OBJ_DSA_S0_DEPTH_WRITEMASK(dsa_state->depth.writemask) |
      VIRGL_OBJ_DSA_S0_DEPTH_FUNC(dsa_state->depth.func) |
      VIRGL_OBJ_DSA_S0_ALPHA_ENABLED(dsa_state->alpha.enabled) |
      VIRGL_OBJ_DSA_S0_ALPHA_FUNC(dsa_state->alpha.func);
   virgl_encoder_write_dword(ctx->cbuf, tmp);

   for (i = 0; i < 2; i++) {
      tmp = VIRGL_OBJ_DSA_S1_STENCIL_ENABLED(dsa_state->stencil[i].enabled) |
         VIRGL_OBJ_DSA_S1_STENCIL_FUNC(dsa_state->stencil[i].func) |
         VIRGL_OBJ_DSA_S1_STENCIL_FAIL_OP(dsa_state->stencil[i].fail_op) |
         VIRGL_OBJ_DSA_S1_STENCIL_ZPASS_OP(dsa_state->stencil[i].zpass_op) |
         VIRGL_OBJ_DSA_S1_STENCIL_ZFAIL_OP(dsa_state->stencil[i].zfail_op) |
         VIRGL_OBJ_DSA_S1_STENCIL_VALUEMASK(dsa_state->stencil[i].valuemask) |
         VIRGL_OBJ_DSA_S1_STENCIL_WRITEMASK(dsa_state->stencil[i].writemask);
      virgl_encoder_write_dword(ctx->cbuf, tmp);
   }

   virgl_encoder_write_dword(ctx->cbuf, fui(dsa_state->alpha.ref_value));
   return 0;
}
int virgl_encode_rasterizer_state(struct virgl_context *ctx,
                                  uint32_t handle,
                                  const struct pipe_rasterizer_state *state)
{
   uint32_t tmp;

   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_RASTERIZER, VIRGL_OBJ_RS_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, handle);

   tmp = VIRGL_OBJ_RS_S0_FLATSHADE(state->flatshade) |
      VIRGL_OBJ_RS_S0_DEPTH_CLIP(state->depth_clip_near) |
      VIRGL_OBJ_RS_S0_CLIP_HALFZ(state->clip_halfz) |
      VIRGL_OBJ_RS_S0_RASTERIZER_DISCARD(state->rasterizer_discard) |
      VIRGL_OBJ_RS_S0_FLATSHADE_FIRST(state->flatshade_first) |
      VIRGL_OBJ_RS_S0_LIGHT_TWOSIZE(state->light_twoside) |
      VIRGL_OBJ_RS_S0_SPRITE_COORD_MODE(state->sprite_coord_mode) |
      VIRGL_OBJ_RS_S0_POINT_QUAD_RASTERIZATION(state->point_quad_rasterization) |
      VIRGL_OBJ_RS_S0_CULL_FACE(state->cull_face) |
      VIRGL_OBJ_RS_S0_FILL_FRONT(state->fill_front) |
      VIRGL_OBJ_RS_S0_FILL_BACK(state->fill_back) |
      VIRGL_OBJ_RS_S0_SCISSOR(state->scissor) |
      VIRGL_OBJ_RS_S0_FRONT_CCW(state->front_ccw) |
      VIRGL_OBJ_RS_S0_CLAMP_VERTEX_COLOR(state->clamp_vertex_color) |
      VIRGL_OBJ_RS_S0_CLAMP_FRAGMENT_COLOR(state->clamp_fragment_color) |
      VIRGL_OBJ_RS_S0_OFFSET_LINE(state->offset_line) |
      VIRGL_OBJ_RS_S0_OFFSET_POINT(state->offset_point) |
      VIRGL_OBJ_RS_S0_OFFSET_TRI(state->offset_tri) |
      VIRGL_OBJ_RS_S0_POLY_SMOOTH(state->poly_smooth) |
      VIRGL_OBJ_RS_S0_POLY_STIPPLE_ENABLE(state->poly_stipple_enable) |
      VIRGL_OBJ_RS_S0_POINT_SMOOTH(state->point_smooth) |
      VIRGL_OBJ_RS_S0_POINT_SIZE_PER_VERTEX(state->point_size_per_vertex) |
      VIRGL_OBJ_RS_S0_MULTISAMPLE(state->multisample) |
      VIRGL_OBJ_RS_S0_LINE_SMOOTH(state->line_smooth) |
      VIRGL_OBJ_RS_S0_LINE_STIPPLE_ENABLE(state->line_stipple_enable) |
      VIRGL_OBJ_RS_S0_LINE_LAST_PIXEL(state->line_last_pixel) |
      VIRGL_OBJ_RS_S0_HALF_PIXEL_CENTER(state->half_pixel_center) |
      VIRGL_OBJ_RS_S0_BOTTOM_EDGE_RULE(state->bottom_edge_rule) |
      VIRGL_OBJ_RS_S0_FORCE_PERSAMPLE_INTERP(state->force_persample_interp);

   virgl_encoder_write_dword(ctx->cbuf, tmp); /* S0 */
   virgl_encoder_write_dword(ctx->cbuf, fui(state->point_size)); /* S1 */
   virgl_encoder_write_dword(ctx->cbuf, state->sprite_coord_enable); /* S2 */
   tmp = VIRGL_OBJ_RS_S3_LINE_STIPPLE_PATTERN(state->line_stipple_pattern) |
      VIRGL_OBJ_RS_S3_LINE_STIPPLE_FACTOR(state->line_stipple_factor) |
      VIRGL_OBJ_RS_S3_CLIP_PLANE_ENABLE(state->clip_plane_enable);
   virgl_encoder_write_dword(ctx->cbuf, tmp); /* S3 */
   virgl_encoder_write_dword(ctx->cbuf, fui(state->line_width)); /* S4 */
   virgl_encoder_write_dword(ctx->cbuf, fui(state->offset_units)); /* S5 */
   virgl_encoder_write_dword(ctx->cbuf, fui(state->offset_scale)); /* S6 */
   virgl_encoder_write_dword(ctx->cbuf, fui(state->offset_clamp)); /* S7 */
   return 0;
}

static void virgl_emit_shader_header(struct virgl_context *ctx,
                                     uint32_t handle, uint32_t len,
                                     uint32_t type, uint32_t offlen,
                                     uint32_t num_tokens)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_SHADER, len));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   virgl_encoder_write_dword(ctx->cbuf, type);
   virgl_encoder_write_dword(ctx->cbuf, offlen);
   virgl_encoder_write_dword(ctx->cbuf, num_tokens);
}

static void virgl_emit_shader_streamout(struct virgl_context *ctx,
                                        const struct pipe_stream_output_info *so_info)
{
   int num_outputs = 0;
   int i;
   uint32_t tmp;

   if (so_info)
      num_outputs = so_info->num_outputs;

   virgl_encoder_write_dword(ctx->cbuf, num_outputs);
   if (num_outputs) {
      for (i = 0; i < 4; i++)
         virgl_encoder_write_dword(ctx->cbuf, so_info->stride[i]);

      for (i = 0; i < so_info->num_outputs; i++) {
         tmp =
           VIRGL_OBJ_SHADER_SO_OUTPUT_REGISTER_INDEX(so_info->output[i].register_index) |
           VIRGL_OBJ_SHADER_SO_OUTPUT_START_COMPONENT(so_info->output[i].start_component) |
           VIRGL_OBJ_SHADER_SO_OUTPUT_NUM_COMPONENTS(so_info->output[i].num_components) |
           VIRGL_OBJ_SHADER_SO_OUTPUT_BUFFER(so_info->output[i].output_buffer) |
           VIRGL_OBJ_SHADER_SO_OUTPUT_DST_OFFSET(so_info->output[i].dst_offset);
         virgl_encoder_write_dword(ctx->cbuf, tmp);
         virgl_encoder_write_dword(ctx->cbuf, so_info->output[i].stream);
      }
   }
}

int virgl_encode_shader_state(struct virgl_context *ctx,
                              uint32_t handle,
                              uint32_t type,
                              const struct pipe_stream_output_info *so_info,
                              uint32_t cs_req_local_mem,
                              const struct tgsi_token *tokens)
{
   char *str, *sptr;
   uint32_t shader_len, len;
   bool bret;
   int num_tokens = tgsi_num_tokens(tokens);
   int str_total_size = 65536;
   int retry_size = 1;
   uint32_t left_bytes, base_hdr_size, strm_hdr_size, thispass;
   bool first_pass;
   str = CALLOC(1, str_total_size);
   if (!str)
      return -1;

   do {
      int old_size;

      bret = tgsi_dump_str(tokens, TGSI_DUMP_FLOAT_AS_HEX, str, str_total_size);
      if (bret == false) {
         if (virgl_debug & VIRGL_DEBUG_VERBOSE)
            debug_printf("Failed to translate shader in available space - trying again\n");
         old_size = str_total_size;
         str_total_size = 65536 * ++retry_size;
         str = REALLOC(str, old_size, str_total_size);
         if (!str)
            return -1;
      }
   } while (bret == false && retry_size < 10);

   if (bret == false)
      return -1;

   if (virgl_debug & VIRGL_DEBUG_TGSI)
      debug_printf("TGSI:\n---8<---\n%s\n---8<---\n", str);

   shader_len = strlen(str) + 1;

   left_bytes = shader_len;

   base_hdr_size = 5;
   strm_hdr_size = so_info->num_outputs ? so_info->num_outputs * 2 + 4 : 0;
   first_pass = true;
   sptr = str;
   while (left_bytes) {
      uint32_t length, offlen;
      int hdr_len = base_hdr_size + (first_pass ? strm_hdr_size : 0);
      if (ctx->cbuf->cdw + hdr_len + 1 >= VIRGL_MAX_CMDBUF_DWORDS)
         ctx->base.flush(&ctx->base, NULL, 0);

      thispass = (VIRGL_MAX_CMDBUF_DWORDS - ctx->cbuf->cdw - hdr_len - 1) * 4;

      length = MIN2(thispass, left_bytes);
      len = ((length + 3) / 4) + hdr_len;

      if (first_pass)
         offlen = VIRGL_OBJ_SHADER_OFFSET_VAL(shader_len);
      else
         offlen = VIRGL_OBJ_SHADER_OFFSET_VAL((uintptr_t)sptr - (uintptr_t)str) | VIRGL_OBJ_SHADER_OFFSET_CONT;

      virgl_emit_shader_header(ctx, handle, len, type, offlen, num_tokens);

      if (type == PIPE_SHADER_COMPUTE)
         virgl_encoder_write_dword(ctx->cbuf, cs_req_local_mem);
      else
         virgl_emit_shader_streamout(ctx, first_pass ? so_info : NULL);

      virgl_encoder_write_block(ctx->cbuf, (uint8_t *)sptr, length);

      sptr += length;
      first_pass = false;
      left_bytes -= length;
   }

   FREE(str);
   return 0;
}


int virgl_encode_clear(struct virgl_context *ctx,
                      unsigned buffers,
                      const union pipe_color_union *color,
                      double depth, unsigned stencil)
{
   int i;
   uint64_t qword;

   STATIC_ASSERT(sizeof(qword) == sizeof(depth));
   memcpy(&qword, &depth, sizeof(qword));

   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CLEAR, 0, VIRGL_OBJ_CLEAR_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, buffers);
   for (i = 0; i < 4; i++)
      virgl_encoder_write_dword(ctx->cbuf, color->ui[i]);
   virgl_encoder_write_qword(ctx->cbuf, qword);
   virgl_encoder_write_dword(ctx->cbuf, stencil);
   return 0;
}

int virgl_encoder_set_framebuffer_state(struct virgl_context *ctx,
                                       const struct pipe_framebuffer_state *state)
{
   struct virgl_surface *zsurf = virgl_surface(state->zsbuf);
   int i;

   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_FRAMEBUFFER_STATE, 0, VIRGL_SET_FRAMEBUFFER_STATE_SIZE(state->nr_cbufs)));
   virgl_encoder_write_dword(ctx->cbuf, state->nr_cbufs);
   virgl_encoder_write_dword(ctx->cbuf, zsurf ? zsurf->handle : 0);
   for (i = 0; i < state->nr_cbufs; i++) {
      struct virgl_surface *surf = virgl_surface(state->cbufs[i]);
      virgl_encoder_write_dword(ctx->cbuf, surf ? surf->handle : 0);
   }

   struct virgl_screen *rs = virgl_screen(ctx->base.screen);
   if (rs->caps.caps.v2.capability_bits & VIRGL_CAP_FB_NO_ATTACH) {
      virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_FRAMEBUFFER_STATE_NO_ATTACH, 0, VIRGL_SET_FRAMEBUFFER_STATE_NO_ATTACH_SIZE));
      virgl_encoder_write_dword(ctx->cbuf, state->width | (state->height << 16));
      virgl_encoder_write_dword(ctx->cbuf, state->layers | (state->samples << 16));
   }
   return 0;
}

int virgl_encoder_set_viewport_states(struct virgl_context *ctx,
                                      int start_slot,
                                      int num_viewports,
                                      const struct pipe_viewport_state *states)
{
   int i,v;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_VIEWPORT_STATE, 0, VIRGL_SET_VIEWPORT_STATE_SIZE(num_viewports)));
   virgl_encoder_write_dword(ctx->cbuf, start_slot);
   for (v = 0; v < num_viewports; v++) {
      for (i = 0; i < 3; i++)
         virgl_encoder_write_dword(ctx->cbuf, fui(states[v].scale[i]));
      for (i = 0; i < 3; i++)
         virgl_encoder_write_dword(ctx->cbuf, fui(states[v].translate[i]));
   }
   return 0;
}

int virgl_encoder_create_vertex_elements(struct virgl_context *ctx,
                                        uint32_t handle,
                                        unsigned num_elements,
                                        const struct pipe_vertex_element *element)
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_VERTEX_ELEMENTS, VIRGL_OBJ_VERTEX_ELEMENTS_SIZE(num_elements)));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   for (i = 0; i < num_elements; i++) {
      virgl_encoder_write_dword(ctx->cbuf, element[i].src_offset);
      virgl_encoder_write_dword(ctx->cbuf, element[i].instance_divisor);
      virgl_encoder_write_dword(ctx->cbuf, element[i].vertex_buffer_index);
      virgl_encoder_write_dword(ctx->cbuf, element[i].src_format);
   }
   return 0;
}

int virgl_encoder_set_vertex_buffers(struct virgl_context *ctx,
                                    unsigned num_buffers,
                                    const struct pipe_vertex_buffer *buffers)
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_VERTEX_BUFFERS, 0, VIRGL_SET_VERTEX_BUFFERS_SIZE(num_buffers)));
   for (i = 0; i < num_buffers; i++) {
      struct virgl_resource *res = virgl_resource(buffers[i].buffer.resource);
      virgl_encoder_write_dword(ctx->cbuf, buffers[i].stride);
      virgl_encoder_write_dword(ctx->cbuf, buffers[i].buffer_offset);
      virgl_encoder_write_res(ctx, res);
   }
   return 0;
}

int virgl_encoder_set_index_buffer(struct virgl_context *ctx,
                                  const struct virgl_indexbuf *ib)
{
   int length = VIRGL_SET_INDEX_BUFFER_SIZE(ib);
   struct virgl_resource *res = NULL;
   if (ib)
      res = virgl_resource(ib->buffer);

   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_INDEX_BUFFER, 0, length));
   virgl_encoder_write_res(ctx, res);
   if (ib) {
      virgl_encoder_write_dword(ctx->cbuf, ib->index_size);
      virgl_encoder_write_dword(ctx->cbuf, ib->offset);
   }
   return 0;
}

int virgl_encoder_draw_vbo(struct virgl_context *ctx,
                          const struct pipe_draw_info *info)
{
   uint32_t length = VIRGL_DRAW_VBO_SIZE;
   if (info->mode == PIPE_PRIM_PATCHES)
      length = VIRGL_DRAW_VBO_SIZE_TESS;
   if (info->indirect)
      length = VIRGL_DRAW_VBO_SIZE_INDIRECT;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_DRAW_VBO, 0, length));
   virgl_encoder_write_dword(ctx->cbuf, info->start);
   virgl_encoder_write_dword(ctx->cbuf, info->count);
   virgl_encoder_write_dword(ctx->cbuf, info->mode);
   virgl_encoder_write_dword(ctx->cbuf, !!info->index_size);
   virgl_encoder_write_dword(ctx->cbuf, info->instance_count);
   virgl_encoder_write_dword(ctx->cbuf, info->index_bias);
   virgl_encoder_write_dword(ctx->cbuf, info->start_instance);
   virgl_encoder_write_dword(ctx->cbuf, info->primitive_restart);
   virgl_encoder_write_dword(ctx->cbuf, info->restart_index);
   virgl_encoder_write_dword(ctx->cbuf, info->min_index);
   virgl_encoder_write_dword(ctx->cbuf, info->max_index);
   if (info->count_from_stream_output)
      virgl_encoder_write_dword(ctx->cbuf, info->count_from_stream_output->buffer_size);
   else
      virgl_encoder_write_dword(ctx->cbuf, 0);
   if (length >= VIRGL_DRAW_VBO_SIZE_TESS) {
      virgl_encoder_write_dword(ctx->cbuf, info->vertices_per_patch); /* vertices per patch */
      virgl_encoder_write_dword(ctx->cbuf, info->drawid); /* drawid */
   }
   if (length == VIRGL_DRAW_VBO_SIZE_INDIRECT) {
      virgl_encoder_write_res(ctx, virgl_resource(info->indirect->buffer));
      virgl_encoder_write_dword(ctx->cbuf, info->indirect->offset);
      virgl_encoder_write_dword(ctx->cbuf, 0); /* indirect stride */
      virgl_encoder_write_dword(ctx->cbuf, 0); /* indirect draw count */
      virgl_encoder_write_dword(ctx->cbuf, 0); /* indirect draw count offset */
      virgl_encoder_write_dword(ctx->cbuf, 0); /* indirect draw count handle */
   }
   return 0;
}

int virgl_encoder_create_surface(struct virgl_context *ctx,
                                uint32_t handle,
                                struct virgl_resource *res,
                                const struct pipe_surface *templat)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_SURFACE, VIRGL_OBJ_SURFACE_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   virgl_encoder_write_res(ctx, res);
   virgl_encoder_write_dword(ctx->cbuf, templat->format);
   if (templat->texture->target == PIPE_BUFFER) {
      virgl_encoder_write_dword(ctx->cbuf, templat->u.buf.first_element);
      virgl_encoder_write_dword(ctx->cbuf, templat->u.buf.last_element);

   } else {
      virgl_encoder_write_dword(ctx->cbuf, templat->u.tex.level);
      virgl_encoder_write_dword(ctx->cbuf, templat->u.tex.first_layer | (templat->u.tex.last_layer << 16));
   }
   return 0;
}

int virgl_encoder_create_so_target(struct virgl_context *ctx,
                                  uint32_t handle,
                                  struct virgl_resource *res,
                                  unsigned buffer_offset,
                                  unsigned buffer_size)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_STREAMOUT_TARGET, VIRGL_OBJ_STREAMOUT_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   virgl_encoder_write_res(ctx, res);
   virgl_encoder_write_dword(ctx->cbuf, buffer_offset);
   virgl_encoder_write_dword(ctx->cbuf, buffer_size);
   return 0;
}

static void virgl_encoder_iw_emit_header_1d(struct virgl_context *ctx,
                                           struct virgl_resource *res,
                                           unsigned level, unsigned usage,
                                           const struct pipe_box *box,
                                           unsigned stride, unsigned layer_stride)
{
   virgl_encoder_write_res(ctx, res);
   virgl_encoder_write_dword(ctx->cbuf, level);
   virgl_encoder_write_dword(ctx->cbuf, usage);
   virgl_encoder_write_dword(ctx->cbuf, stride);
   virgl_encoder_write_dword(ctx->cbuf, layer_stride);
   virgl_encoder_write_dword(ctx->cbuf, box->x);
   virgl_encoder_write_dword(ctx->cbuf, box->y);
   virgl_encoder_write_dword(ctx->cbuf, box->z);
   virgl_encoder_write_dword(ctx->cbuf, box->width);
   virgl_encoder_write_dword(ctx->cbuf, box->height);
   virgl_encoder_write_dword(ctx->cbuf, box->depth);
}

int virgl_encoder_inline_write(struct virgl_context *ctx,
                              struct virgl_resource *res,
                              unsigned level, unsigned usage,
                              const struct pipe_box *box,
                              const void *data, unsigned stride,
                              unsigned layer_stride)
{
   uint32_t size = (stride ? stride : box->width) * box->height;
   uint32_t length, thispass, left_bytes;
   struct pipe_box mybox = *box;

   length = 11 + (size + 3) / 4;
   if ((ctx->cbuf->cdw + length + 1) > VIRGL_MAX_CMDBUF_DWORDS) {
      if (box->height > 1 || box->depth > 1) {
         debug_printf("inline transfer failed due to multi dimensions and too large\n");
         assert(0);
      }
   }

   left_bytes = size;
   while (left_bytes) {
      if (ctx->cbuf->cdw + 12 >= VIRGL_MAX_CMDBUF_DWORDS)
         ctx->base.flush(&ctx->base, NULL, 0);

      thispass = (VIRGL_MAX_CMDBUF_DWORDS - ctx->cbuf->cdw - 12) * 4;

      length = MIN2(thispass, left_bytes);

      mybox.width = length;
      virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_RESOURCE_INLINE_WRITE, 0, ((length + 3) / 4) + 11));
      virgl_encoder_iw_emit_header_1d(ctx, res, level, usage, &mybox, stride, layer_stride);
      virgl_encoder_write_block(ctx->cbuf, data, length);
      left_bytes -= length;
      mybox.x += length;
      data += length;
   }
   return 0;
}

int virgl_encoder_flush_frontbuffer(struct virgl_context *ctx,
                                   struct virgl_resource *res)
{
//   virgl_encoder_write_dword(ctx->cbuf, VIRGL_CMD0(VIRGL_CCMD_FLUSH_FRONTUBFFER, 0, 1));
//   virgl_encoder_write_dword(ctx->cbuf, res_handle);
   return 0;
}

int virgl_encode_sampler_state(struct virgl_context *ctx,
                              uint32_t handle,
                              const struct pipe_sampler_state *state)
{
   uint32_t tmp;
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_SAMPLER_STATE, VIRGL_OBJ_SAMPLER_STATE_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, handle);

   tmp = VIRGL_OBJ_SAMPLE_STATE_S0_WRAP_S(state->wrap_s) |
      VIRGL_OBJ_SAMPLE_STATE_S0_WRAP_T(state->wrap_t) |
      VIRGL_OBJ_SAMPLE_STATE_S0_WRAP_R(state->wrap_r) |
      VIRGL_OBJ_SAMPLE_STATE_S0_MIN_IMG_FILTER(state->min_img_filter) |
      VIRGL_OBJ_SAMPLE_STATE_S0_MIN_MIP_FILTER(state->min_mip_filter) |
      VIRGL_OBJ_SAMPLE_STATE_S0_MAG_IMG_FILTER(state->mag_img_filter) |
      VIRGL_OBJ_SAMPLE_STATE_S0_COMPARE_MODE(state->compare_mode) |
      VIRGL_OBJ_SAMPLE_STATE_S0_COMPARE_FUNC(state->compare_func) |
      VIRGL_OBJ_SAMPLE_STATE_S0_SEAMLESS_CUBE_MAP(state->seamless_cube_map);

   virgl_encoder_write_dword(ctx->cbuf, tmp);
   virgl_encoder_write_dword(ctx->cbuf, fui(state->lod_bias));
   virgl_encoder_write_dword(ctx->cbuf, fui(state->min_lod));
   virgl_encoder_write_dword(ctx->cbuf, fui(state->max_lod));
   for (i = 0; i <  4; i++)
      virgl_encoder_write_dword(ctx->cbuf, state->border_color.ui[i]);
   return 0;
}


int virgl_encode_sampler_view(struct virgl_context *ctx,
                             uint32_t handle,
                             struct virgl_resource *res,
                             const struct pipe_sampler_view *state)
{
   unsigned elem_size = util_format_get_blocksize(state->format);
   struct virgl_screen *rs = virgl_screen(ctx->base.screen);
   uint32_t tmp;
   uint32_t dword_fmt_target = state->format;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_SAMPLER_VIEW, VIRGL_OBJ_SAMPLER_VIEW_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   virgl_encoder_write_res(ctx, res);
   if (rs->caps.caps.v2.capability_bits & VIRGL_CAP_TEXTURE_VIEW)
     dword_fmt_target |= (state->target << 24);
   virgl_encoder_write_dword(ctx->cbuf, dword_fmt_target);
   if (res->u.b.target == PIPE_BUFFER) {
      virgl_encoder_write_dword(ctx->cbuf, state->u.buf.offset / elem_size);
      virgl_encoder_write_dword(ctx->cbuf, (state->u.buf.offset + state->u.buf.size) / elem_size - 1);
      virgl_dirty_res(res);
   } else {
      virgl_encoder_write_dword(ctx->cbuf, state->u.tex.first_layer | state->u.tex.last_layer << 16);
      virgl_encoder_write_dword(ctx->cbuf, state->u.tex.first_level | state->u.tex.last_level << 8);
   }
   tmp = VIRGL_OBJ_SAMPLER_VIEW_SWIZZLE_R(state->swizzle_r) |
      VIRGL_OBJ_SAMPLER_VIEW_SWIZZLE_G(state->swizzle_g) |
      VIRGL_OBJ_SAMPLER_VIEW_SWIZZLE_B(state->swizzle_b) |
      VIRGL_OBJ_SAMPLER_VIEW_SWIZZLE_A(state->swizzle_a);
   virgl_encoder_write_dword(ctx->cbuf, tmp);
   return 0;
}

int virgl_encode_set_sampler_views(struct virgl_context *ctx,
                                  uint32_t shader_type,
                                  uint32_t start_slot,
                                  uint32_t num_views,
                                  struct virgl_sampler_view **views)
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_SAMPLER_VIEWS, 0, VIRGL_SET_SAMPLER_VIEWS_SIZE(num_views)));
   virgl_encoder_write_dword(ctx->cbuf, shader_type);
   virgl_encoder_write_dword(ctx->cbuf, start_slot);
   for (i = 0; i < num_views; i++) {
      uint32_t handle = views[i] ? views[i]->handle : 0;
      virgl_encoder_write_dword(ctx->cbuf, handle);
   }
   return 0;
}

int virgl_encode_bind_sampler_states(struct virgl_context *ctx,
                                    uint32_t shader_type,
                                    uint32_t start_slot,
                                    uint32_t num_handles,
                                    uint32_t *handles)
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_BIND_SAMPLER_STATES, 0, VIRGL_BIND_SAMPLER_STATES(num_handles)));
   virgl_encoder_write_dword(ctx->cbuf, shader_type);
   virgl_encoder_write_dword(ctx->cbuf, start_slot);
   for (i = 0; i < num_handles; i++)
      virgl_encoder_write_dword(ctx->cbuf, handles[i]);
   return 0;
}

int virgl_encoder_write_constant_buffer(struct virgl_context *ctx,
                                       uint32_t shader,
                                       uint32_t index,
                                       uint32_t size,
                                       const void *data)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_CONSTANT_BUFFER, 0, size + 2));
   virgl_encoder_write_dword(ctx->cbuf, shader);
   virgl_encoder_write_dword(ctx->cbuf, index);
   if (data)
      virgl_encoder_write_block(ctx->cbuf, data, size * 4);
   return 0;
}

int virgl_encoder_set_uniform_buffer(struct virgl_context *ctx,
                                     uint32_t shader,
                                     uint32_t index,
                                     uint32_t offset,
                                     uint32_t length,
                                     struct virgl_resource *res)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_UNIFORM_BUFFER, 0, VIRGL_SET_UNIFORM_BUFFER_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, shader);
   virgl_encoder_write_dword(ctx->cbuf, index);
   virgl_encoder_write_dword(ctx->cbuf, offset);
   virgl_encoder_write_dword(ctx->cbuf, length);
   virgl_encoder_write_res(ctx, res);
   return 0;
}


int virgl_encoder_set_stencil_ref(struct virgl_context *ctx,
                                 const struct pipe_stencil_ref *ref)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_STENCIL_REF, 0, VIRGL_SET_STENCIL_REF_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, VIRGL_STENCIL_REF_VAL(ref->ref_value[0] , (ref->ref_value[1])));
   return 0;
}

int virgl_encoder_set_blend_color(struct virgl_context *ctx,
                                 const struct pipe_blend_color *color)
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_BLEND_COLOR, 0, VIRGL_SET_BLEND_COLOR_SIZE));
   for (i = 0; i < 4; i++)
      virgl_encoder_write_dword(ctx->cbuf, fui(color->color[i]));
   return 0;
}

int virgl_encoder_set_scissor_state(struct virgl_context *ctx,
                                    unsigned start_slot,
                                    int num_scissors,
                                    const struct pipe_scissor_state *ss)
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_SCISSOR_STATE, 0, VIRGL_SET_SCISSOR_STATE_SIZE(num_scissors)));
   virgl_encoder_write_dword(ctx->cbuf, start_slot);
   for (i = 0; i < num_scissors; i++) {
      virgl_encoder_write_dword(ctx->cbuf, (ss[i].minx | ss[i].miny << 16));
      virgl_encoder_write_dword(ctx->cbuf, (ss[i].maxx | ss[i].maxy << 16));
   }
   return 0;
}

void virgl_encoder_set_polygon_stipple(struct virgl_context *ctx,
                                      const struct pipe_poly_stipple *ps)
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_POLYGON_STIPPLE, 0, VIRGL_POLYGON_STIPPLE_SIZE));
   for (i = 0; i < VIRGL_POLYGON_STIPPLE_SIZE; i++) {
      virgl_encoder_write_dword(ctx->cbuf, ps->stipple[i]);
   }
}

void virgl_encoder_set_sample_mask(struct virgl_context *ctx,
                                  unsigned sample_mask)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_SAMPLE_MASK, 0, VIRGL_SET_SAMPLE_MASK_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, sample_mask);
}

void virgl_encoder_set_min_samples(struct virgl_context *ctx,
                                  unsigned min_samples)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_MIN_SAMPLES, 0, VIRGL_SET_MIN_SAMPLES_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, min_samples);
}

void virgl_encoder_set_clip_state(struct virgl_context *ctx,
                                 const struct pipe_clip_state *clip)
{
   int i, j;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_CLIP_STATE, 0, VIRGL_SET_CLIP_STATE_SIZE));
   for (i = 0; i < VIRGL_MAX_CLIP_PLANES; i++) {
      for (j = 0; j < 4; j++) {
         virgl_encoder_write_dword(ctx->cbuf, fui(clip->ucp[i][j]));
      }
   }
}

int virgl_encode_resource_copy_region(struct virgl_context *ctx,
                                     struct virgl_resource *dst_res,
                                     unsigned dst_level,
                                     unsigned dstx, unsigned dsty, unsigned dstz,
                                     struct virgl_resource *src_res,
                                     unsigned src_level,
                                     const struct pipe_box *src_box)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_RESOURCE_COPY_REGION, 0, VIRGL_CMD_RESOURCE_COPY_REGION_SIZE));
   virgl_encoder_write_res(ctx, dst_res);
   virgl_encoder_write_dword(ctx->cbuf, dst_level);
   virgl_encoder_write_dword(ctx->cbuf, dstx);
   virgl_encoder_write_dword(ctx->cbuf, dsty);
   virgl_encoder_write_dword(ctx->cbuf, dstz);
   virgl_encoder_write_res(ctx, src_res);
   virgl_encoder_write_dword(ctx->cbuf, src_level);
   virgl_encoder_write_dword(ctx->cbuf, src_box->x);
   virgl_encoder_write_dword(ctx->cbuf, src_box->y);
   virgl_encoder_write_dword(ctx->cbuf, src_box->z);
   virgl_encoder_write_dword(ctx->cbuf, src_box->width);
   virgl_encoder_write_dword(ctx->cbuf, src_box->height);
   virgl_encoder_write_dword(ctx->cbuf, src_box->depth);
   return 0;
}

int virgl_encode_blit(struct virgl_context *ctx,
                     struct virgl_resource *dst_res,
                     struct virgl_resource *src_res,
                     const struct pipe_blit_info *blit)
{
   uint32_t tmp;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_BLIT, 0, VIRGL_CMD_BLIT_SIZE));
   tmp = VIRGL_CMD_BLIT_S0_MASK(blit->mask) |
      VIRGL_CMD_BLIT_S0_FILTER(blit->filter) |
      VIRGL_CMD_BLIT_S0_SCISSOR_ENABLE(blit->scissor_enable) |
      VIRGL_CMD_BLIT_S0_RENDER_CONDITION_ENABLE(blit->render_condition_enable) |
      VIRGL_CMD_BLIT_S0_ALPHA_BLEND(blit->alpha_blend);
   virgl_encoder_write_dword(ctx->cbuf, tmp);
   virgl_encoder_write_dword(ctx->cbuf, (blit->scissor.minx | blit->scissor.miny << 16));
   virgl_encoder_write_dword(ctx->cbuf, (blit->scissor.maxx | blit->scissor.maxy << 16));

   virgl_encoder_write_res(ctx, dst_res);
   virgl_encoder_write_dword(ctx->cbuf, blit->dst.level);
   virgl_encoder_write_dword(ctx->cbuf, blit->dst.format);
   virgl_encoder_write_dword(ctx->cbuf, blit->dst.box.x);
   virgl_encoder_write_dword(ctx->cbuf, blit->dst.box.y);
   virgl_encoder_write_dword(ctx->cbuf, blit->dst.box.z);
   virgl_encoder_write_dword(ctx->cbuf, blit->dst.box.width);
   virgl_encoder_write_dword(ctx->cbuf, blit->dst.box.height);
   virgl_encoder_write_dword(ctx->cbuf, blit->dst.box.depth);

   virgl_encoder_write_res(ctx, src_res);
   virgl_encoder_write_dword(ctx->cbuf, blit->src.level);
   virgl_encoder_write_dword(ctx->cbuf, blit->src.format);
   virgl_encoder_write_dword(ctx->cbuf, blit->src.box.x);
   virgl_encoder_write_dword(ctx->cbuf, blit->src.box.y);
   virgl_encoder_write_dword(ctx->cbuf, blit->src.box.z);
   virgl_encoder_write_dword(ctx->cbuf, blit->src.box.width);
   virgl_encoder_write_dword(ctx->cbuf, blit->src.box.height);
   virgl_encoder_write_dword(ctx->cbuf, blit->src.box.depth);
   return 0;
}

int virgl_encoder_create_query(struct virgl_context *ctx,
                              uint32_t handle,
                              uint query_type,
                              uint query_index,
                              struct virgl_resource *res,
                              uint32_t offset)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_OBJECT, VIRGL_OBJECT_QUERY, VIRGL_OBJ_QUERY_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   virgl_encoder_write_dword(ctx->cbuf, ((query_type & 0xffff) | (query_index << 16)));
   virgl_encoder_write_dword(ctx->cbuf, offset);
   virgl_encoder_write_res(ctx, res);
   return 0;
}

int virgl_encoder_begin_query(struct virgl_context *ctx,
                             uint32_t handle)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_BEGIN_QUERY, 0, 1));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   return 0;
}

int virgl_encoder_end_query(struct virgl_context *ctx,
                           uint32_t handle)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_END_QUERY, 0, 1));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   return 0;
}

int virgl_encoder_get_query_result(struct virgl_context *ctx,
                                  uint32_t handle, boolean wait)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_GET_QUERY_RESULT, 0, 2));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   virgl_encoder_write_dword(ctx->cbuf, wait ? 1 : 0);
   return 0;
}

int virgl_encoder_render_condition(struct virgl_context *ctx,
                                  uint32_t handle, boolean condition,
                                  enum pipe_render_cond_flag mode)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_RENDER_CONDITION, 0, VIRGL_RENDER_CONDITION_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   virgl_encoder_write_dword(ctx->cbuf, condition);
   virgl_encoder_write_dword(ctx->cbuf, mode);
   return 0;
}

int virgl_encoder_set_so_targets(struct virgl_context *ctx,
                                unsigned num_targets,
                                struct pipe_stream_output_target **targets,
                                unsigned append_bitmask)
{
   int i;

   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_STREAMOUT_TARGETS, 0, num_targets + 1));
   virgl_encoder_write_dword(ctx->cbuf, append_bitmask);
   for (i = 0; i < num_targets; i++) {
      struct virgl_so_target *tg = virgl_so_target(targets[i]);
      virgl_encoder_write_dword(ctx->cbuf, tg ? tg->handle : 0);
   }
   return 0;
}


int virgl_encoder_set_sub_ctx(struct virgl_context *ctx, uint32_t sub_ctx_id)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_SUB_CTX, 0, 1));
   virgl_encoder_write_dword(ctx->cbuf, sub_ctx_id);
   return 0;
}

int virgl_encoder_create_sub_ctx(struct virgl_context *ctx, uint32_t sub_ctx_id)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_CREATE_SUB_CTX, 0, 1));
   virgl_encoder_write_dword(ctx->cbuf, sub_ctx_id);
   return 0;
}

int virgl_encoder_destroy_sub_ctx(struct virgl_context *ctx, uint32_t sub_ctx_id)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_DESTROY_SUB_CTX, 0, 1));
   virgl_encoder_write_dword(ctx->cbuf, sub_ctx_id);
   return 0;
}

int virgl_encode_bind_shader(struct virgl_context *ctx,
                             uint32_t handle, uint32_t type)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_BIND_SHADER, 0, 2));
   virgl_encoder_write_dword(ctx->cbuf, handle);
   virgl_encoder_write_dword(ctx->cbuf, type);
   return 0;
}

int virgl_encode_set_tess_state(struct virgl_context *ctx,
                                const float outer[4],
                                const float inner[2])
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_TESS_STATE, 0, 6));
   for (i = 0; i < 4; i++)
      virgl_encoder_write_dword(ctx->cbuf, fui(outer[i]));
   for (i = 0; i < 2; i++)
      virgl_encoder_write_dword(ctx->cbuf, fui(inner[i]));
   return 0;
}

int virgl_encode_set_shader_buffers(struct virgl_context *ctx,
                                    enum pipe_shader_type shader,
                                    unsigned start_slot, unsigned count,
                                    const struct pipe_shader_buffer *buffers)
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_SHADER_BUFFERS, 0, VIRGL_SET_SHADER_BUFFER_SIZE(count)));

   virgl_encoder_write_dword(ctx->cbuf, shader);
   virgl_encoder_write_dword(ctx->cbuf, start_slot);
   for (i = 0; i < count; i++) {
      if (buffers) {
         struct virgl_resource *res = virgl_resource(buffers[i].buffer);
         virgl_encoder_write_dword(ctx->cbuf, buffers[i].buffer_offset);
         virgl_encoder_write_dword(ctx->cbuf, buffers[i].buffer_size);
         virgl_encoder_write_res(ctx, res);
         virgl_dirty_res(res);
      } else {
         virgl_encoder_write_dword(ctx->cbuf, 0);
         virgl_encoder_write_dword(ctx->cbuf, 0);
         virgl_encoder_write_dword(ctx->cbuf, 0);
      }
   }
   return 0;
}

int virgl_encode_set_hw_atomic_buffers(struct virgl_context *ctx,
                                       unsigned start_slot, unsigned count,
                                       const struct pipe_shader_buffer *buffers)
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_ATOMIC_BUFFERS, 0, VIRGL_SET_ATOMIC_BUFFER_SIZE(count)));

   virgl_encoder_write_dword(ctx->cbuf, start_slot);
   for (i = 0; i < count; i++) {
      if (buffers) {
         struct virgl_resource *res = virgl_resource(buffers[i].buffer);
         virgl_encoder_write_dword(ctx->cbuf, buffers[i].buffer_offset);
         virgl_encoder_write_dword(ctx->cbuf, buffers[i].buffer_size);
         virgl_encoder_write_res(ctx, res);
         virgl_dirty_res(res);
      } else {
         virgl_encoder_write_dword(ctx->cbuf, 0);
         virgl_encoder_write_dword(ctx->cbuf, 0);
         virgl_encoder_write_dword(ctx->cbuf, 0);
      }
   }
   return 0;
}

int virgl_encode_set_shader_images(struct virgl_context *ctx,
                                   enum pipe_shader_type shader,
                                   unsigned start_slot, unsigned count,
                                   const struct pipe_image_view *images)
{
   int i;
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_SET_SHADER_IMAGES, 0, VIRGL_SET_SHADER_IMAGE_SIZE(count)));

   virgl_encoder_write_dword(ctx->cbuf, shader);
   virgl_encoder_write_dword(ctx->cbuf, start_slot);
   for (i = 0; i < count; i++) {
      if (images) {
         struct virgl_resource *res = virgl_resource(images[i].resource);
         virgl_encoder_write_dword(ctx->cbuf, images[i].format);
         virgl_encoder_write_dword(ctx->cbuf, images[i].access);
         virgl_encoder_write_dword(ctx->cbuf, images[i].u.buf.offset);
         virgl_encoder_write_dword(ctx->cbuf, images[i].u.buf.size);
         virgl_encoder_write_res(ctx, res);
         virgl_dirty_res(res);
      } else {
         virgl_encoder_write_dword(ctx->cbuf, 0);
         virgl_encoder_write_dword(ctx->cbuf, 0);
         virgl_encoder_write_dword(ctx->cbuf, 0);
         virgl_encoder_write_dword(ctx->cbuf, 0);
         virgl_encoder_write_dword(ctx->cbuf, 0);
      }
   }
   return 0;
}

int virgl_encode_memory_barrier(struct virgl_context *ctx,
                                unsigned flags)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_MEMORY_BARRIER, 0, 1));
   virgl_encoder_write_dword(ctx->cbuf, flags);
   return 0;
}

int virgl_encode_launch_grid(struct virgl_context *ctx,
                             const struct pipe_grid_info *grid_info)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_LAUNCH_GRID, 0, VIRGL_LAUNCH_GRID_SIZE));
   virgl_encoder_write_dword(ctx->cbuf, grid_info->block[0]);
   virgl_encoder_write_dword(ctx->cbuf, grid_info->block[1]);
   virgl_encoder_write_dword(ctx->cbuf, grid_info->block[2]);
   virgl_encoder_write_dword(ctx->cbuf, grid_info->grid[0]);
   virgl_encoder_write_dword(ctx->cbuf, grid_info->grid[1]);
   virgl_encoder_write_dword(ctx->cbuf, grid_info->grid[2]);
   if (grid_info->indirect) {
      struct virgl_resource *res = virgl_resource(grid_info->indirect);
      virgl_encoder_write_res(ctx, res);
   } else
      virgl_encoder_write_dword(ctx->cbuf, 0);
   virgl_encoder_write_dword(ctx->cbuf, grid_info->indirect_offset);
   return 0;
}

int virgl_encode_texture_barrier(struct virgl_context *ctx,
                                 unsigned flags)
{
   virgl_encoder_write_cmd_dword(ctx, VIRGL_CMD0(VIRGL_CCMD_TEXTURE_BARRIER, 0, 1));
   virgl_encoder_write_dword(ctx->cbuf, flags);
   return 0;
}
