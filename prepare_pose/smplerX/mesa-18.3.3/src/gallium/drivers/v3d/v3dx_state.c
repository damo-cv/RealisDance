/*
 * Copyright © 2014-2017 Broadcom
 * Copyright (C) 2012 Rob Clark <robclark@freedesktop.org>
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

#include "pipe/p_state.h"
#include "util/u_format.h"
#include "util/u_framebuffer.h"
#include "util/u_inlines.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_half.h"
#include "util/u_helpers.h"

#include "v3d_context.h"
#include "v3d_tiling.h"
#include "broadcom/common/v3d_macros.h"
#include "broadcom/cle/v3dx_pack.h"

static void
v3d_generic_cso_state_delete(struct pipe_context *pctx, void *hwcso)
{
        free(hwcso);
}

static void
v3d_set_blend_color(struct pipe_context *pctx,
                    const struct pipe_blend_color *blend_color)
{
        struct v3d_context *v3d = v3d_context(pctx);
        v3d->blend_color.f = *blend_color;
        for (int i = 0; i < 4; i++) {
                v3d->blend_color.hf[i] =
                        util_float_to_half(blend_color->color[i]);
        }
        v3d->dirty |= VC5_DIRTY_BLEND_COLOR;
}

static void
v3d_set_stencil_ref(struct pipe_context *pctx,
                    const struct pipe_stencil_ref *stencil_ref)
{
        struct v3d_context *v3d = v3d_context(pctx);
        v3d->stencil_ref = *stencil_ref;
        v3d->dirty |= VC5_DIRTY_STENCIL_REF;
}

static void
v3d_set_clip_state(struct pipe_context *pctx,
                   const struct pipe_clip_state *clip)
{
        struct v3d_context *v3d = v3d_context(pctx);
        v3d->clip = *clip;
        v3d->dirty |= VC5_DIRTY_CLIP;
}

static void
v3d_set_sample_mask(struct pipe_context *pctx, unsigned sample_mask)
{
        struct v3d_context *v3d = v3d_context(pctx);
        v3d->sample_mask = sample_mask & ((1 << VC5_MAX_SAMPLES) - 1);
        v3d->dirty |= VC5_DIRTY_SAMPLE_STATE;
}

static void *
v3d_create_rasterizer_state(struct pipe_context *pctx,
                            const struct pipe_rasterizer_state *cso)
{
        struct v3d_rasterizer_state *so;

        so = CALLOC_STRUCT(v3d_rasterizer_state);
        if (!so)
                return NULL;

        so->base = *cso;

        /* Workaround: HW-2726 PTB does not handle zero-size points (BCM2835,
         * BCM21553).
         */
        so->point_size = MAX2(cso->point_size, .125f);

        STATIC_ASSERT(sizeof(so->depth_offset) >=
                      cl_packet_length(DEPTH_OFFSET));
        v3dx_pack(&so->depth_offset, DEPTH_OFFSET, depth) {
                depth.depth_offset_factor = cso->offset_scale;
                depth.depth_offset_units = cso->offset_units;
        }

        /* The HW treats polygon offset units based on a Z24 buffer, so we
         * need to scale up offset_units if we're only Z16.
         */
        v3dx_pack(&so->depth_offset_z16, DEPTH_OFFSET, depth) {
                depth.depth_offset_factor = cso->offset_scale;
                depth.depth_offset_units = cso->offset_units * 256.0;
        }

        return so;
}

/* Blend state is baked into shaders. */
static void *
v3d_create_blend_state(struct pipe_context *pctx,
                       const struct pipe_blend_state *cso)
{
        struct v3d_blend_state *so;

        so = CALLOC_STRUCT(v3d_blend_state);
        if (!so)
                return NULL;

        so->base = *cso;

        if (cso->independent_blend_enable) {
                for (int i = 0; i < VC5_MAX_DRAW_BUFFERS; i++) {
                        so->blend_enables |= cso->rt[i].blend_enable << i;

                        /* V3D 4.x is when we got independent blend enables. */
                        assert(V3D_VERSION >= 40 ||
                               cso->rt[i].blend_enable == cso->rt[0].blend_enable);
                }
        } else {
                if (cso->rt[0].blend_enable)
                        so->blend_enables = (1 << VC5_MAX_DRAW_BUFFERS) - 1;
        }

        return so;
}

static uint32_t
translate_stencil_op(enum pipe_stencil_op op)
{
        switch (op) {
        case PIPE_STENCIL_OP_KEEP:      return V3D_STENCIL_OP_KEEP;
        case PIPE_STENCIL_OP_ZERO:      return V3D_STENCIL_OP_ZERO;
        case PIPE_STENCIL_OP_REPLACE:   return V3D_STENCIL_OP_REPLACE;
        case PIPE_STENCIL_OP_INCR:      return V3D_STENCIL_OP_INCR;
        case PIPE_STENCIL_OP_DECR:      return V3D_STENCIL_OP_DECR;
        case PIPE_STENCIL_OP_INCR_WRAP: return V3D_STENCIL_OP_INCWRAP;
        case PIPE_STENCIL_OP_DECR_WRAP: return V3D_STENCIL_OP_DECWRAP;
        case PIPE_STENCIL_OP_INVERT:    return V3D_STENCIL_OP_INVERT;
        }
        unreachable("bad stencil op");
}

static void *
v3d_create_depth_stencil_alpha_state(struct pipe_context *pctx,
                                     const struct pipe_depth_stencil_alpha_state *cso)
{
        struct v3d_depth_stencil_alpha_state *so;

        so = CALLOC_STRUCT(v3d_depth_stencil_alpha_state);
        if (!so)
                return NULL;

        so->base = *cso;

        if (cso->depth.enabled) {
                switch (cso->depth.func) {
                case PIPE_FUNC_LESS:
                case PIPE_FUNC_LEQUAL:
                        so->ez_state = VC5_EZ_LT_LE;
                        break;
                case PIPE_FUNC_GREATER:
                case PIPE_FUNC_GEQUAL:
                        so->ez_state = VC5_EZ_GT_GE;
                        break;
                case PIPE_FUNC_NEVER:
                case PIPE_FUNC_EQUAL:
                        so->ez_state = VC5_EZ_UNDECIDED;
                        break;
                default:
                        so->ez_state = VC5_EZ_DISABLED;
                        break;
                }

                /* If stencil is enabled and it's not a no-op, then it would
                 * break EZ updates.
                 */
                if (cso->stencil[0].enabled &&
                    (cso->stencil[0].zfail_op != PIPE_STENCIL_OP_KEEP ||
                     cso->stencil[0].func != PIPE_FUNC_ALWAYS ||
                     (cso->stencil[1].enabled &&
                      (cso->stencil[1].zfail_op != PIPE_STENCIL_OP_KEEP &&
                       cso->stencil[1].func != PIPE_FUNC_ALWAYS)))) {
                        so->ez_state = VC5_EZ_DISABLED;
                }
        }

        const struct pipe_stencil_state *front = &cso->stencil[0];
        const struct pipe_stencil_state *back = &cso->stencil[1];

        if (front->enabled) {
                STATIC_ASSERT(sizeof(so->stencil_front) >=
                              cl_packet_length(STENCIL_CFG));
                v3dx_pack(&so->stencil_front, STENCIL_CFG, config) {
                        config.front_config = true;
                        /* If !back->enabled, then the front values should be
                         * used for both front and back-facing primitives.
                         */
                        config.back_config = !back->enabled;

                        config.stencil_write_mask = front->writemask;
                        config.stencil_test_mask = front->valuemask;

                        config.stencil_test_function = front->func;
                        config.stencil_pass_op =
                                translate_stencil_op(front->zpass_op);
                        config.depth_test_fail_op =
                                translate_stencil_op(front->zfail_op);
                        config.stencil_test_fail_op =
                                translate_stencil_op(front->fail_op);
                }
        }
        if (back->enabled) {
                STATIC_ASSERT(sizeof(so->stencil_back) >=
                              cl_packet_length(STENCIL_CFG));
                v3dx_pack(&so->stencil_back, STENCIL_CFG, config) {
                        config.front_config = false;
                        config.back_config = true;

                        config.stencil_write_mask = back->writemask;
                        config.stencil_test_mask = back->valuemask;

                        config.stencil_test_function = back->func;
                        config.stencil_pass_op =
                                translate_stencil_op(back->zpass_op);
                        config.depth_test_fail_op =
                                translate_stencil_op(back->zfail_op);
                        config.stencil_test_fail_op =
                                translate_stencil_op(back->fail_op);
                }
        }

        return so;
}

static void
v3d_set_polygon_stipple(struct pipe_context *pctx,
                        const struct pipe_poly_stipple *stipple)
{
        struct v3d_context *v3d = v3d_context(pctx);
        v3d->stipple = *stipple;
        v3d->dirty |= VC5_DIRTY_STIPPLE;
}

static void
v3d_set_scissor_states(struct pipe_context *pctx,
                       unsigned start_slot,
                       unsigned num_scissors,
                       const struct pipe_scissor_state *scissor)
{
        struct v3d_context *v3d = v3d_context(pctx);

        v3d->scissor = *scissor;
        v3d->dirty |= VC5_DIRTY_SCISSOR;
}

static void
v3d_set_viewport_states(struct pipe_context *pctx,
                        unsigned start_slot,
                        unsigned num_viewports,
                        const struct pipe_viewport_state *viewport)
{
        struct v3d_context *v3d = v3d_context(pctx);
        v3d->viewport = *viewport;
        v3d->dirty |= VC5_DIRTY_VIEWPORT;
}

static void
v3d_set_vertex_buffers(struct pipe_context *pctx,
                       unsigned start_slot, unsigned count,
                       const struct pipe_vertex_buffer *vb)
{
        struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_vertexbuf_stateobj *so = &v3d->vertexbuf;

        util_set_vertex_buffers_mask(so->vb, &so->enabled_mask, vb,
                                     start_slot, count);
        so->count = util_last_bit(so->enabled_mask);

        v3d->dirty |= VC5_DIRTY_VTXBUF;
}

static void
v3d_blend_state_bind(struct pipe_context *pctx, void *hwcso)
{
        struct v3d_context *v3d = v3d_context(pctx);
        v3d->blend = hwcso;
        v3d->dirty |= VC5_DIRTY_BLEND;
}

static void
v3d_rasterizer_state_bind(struct pipe_context *pctx, void *hwcso)
{
        struct v3d_context *v3d = v3d_context(pctx);
        v3d->rasterizer = hwcso;
        v3d->dirty |= VC5_DIRTY_RASTERIZER;
}

static void
v3d_zsa_state_bind(struct pipe_context *pctx, void *hwcso)
{
        struct v3d_context *v3d = v3d_context(pctx);
        v3d->zsa = hwcso;
        v3d->dirty |= VC5_DIRTY_ZSA;
}

static void *
v3d_vertex_state_create(struct pipe_context *pctx, unsigned num_elements,
                        const struct pipe_vertex_element *elements)
{
        struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_vertex_stateobj *so = CALLOC_STRUCT(v3d_vertex_stateobj);

        if (!so)
                return NULL;

        memcpy(so->pipe, elements, sizeof(*elements) * num_elements);
        so->num_elements = num_elements;

        for (int i = 0; i < so->num_elements; i++) {
                const struct pipe_vertex_element *elem = &elements[i];
                const struct util_format_description *desc =
                        util_format_description(elem->src_format);
                uint32_t r_size = desc->channel[0].size;

                const uint32_t size =
                        cl_packet_length(GL_SHADER_STATE_ATTRIBUTE_RECORD);

                v3dx_pack(&so->attrs[i * size],
                          GL_SHADER_STATE_ATTRIBUTE_RECORD, attr) {
                        /* vec_size == 0 means 4 */
                        attr.vec_size = desc->nr_channels & 3;
                        attr.signed_int_type = (desc->channel[0].type ==
                                                UTIL_FORMAT_TYPE_SIGNED);

                        attr.normalized_int_type = desc->channel[0].normalized;
                        attr.read_as_int_uint = desc->channel[0].pure_integer;
                        attr.instance_divisor = MIN2(elem->instance_divisor,
                                                     0xffff);

                        switch (desc->channel[0].type) {
                        case UTIL_FORMAT_TYPE_FLOAT:
                                if (r_size == 32) {
                                        attr.type = ATTRIBUTE_FLOAT;
                                } else {
                                        assert(r_size == 16);
                                        attr.type = ATTRIBUTE_HALF_FLOAT;
                                }
                                break;

                        case UTIL_FORMAT_TYPE_SIGNED:
                        case UTIL_FORMAT_TYPE_UNSIGNED:
                                switch (r_size) {
                                case 32:
                                        attr.type = ATTRIBUTE_INT;
                                        break;
                                case 16:
                                        attr.type = ATTRIBUTE_SHORT;
                                        break;
                                case 10:
                                        attr.type = ATTRIBUTE_INT2_10_10_10;
                                        break;
                                case 8:
                                        attr.type = ATTRIBUTE_BYTE;
                                        break;
                                default:
                                        fprintf(stderr,
                                                "format %s unsupported\n",
                                                desc->name);
                                        attr.type = ATTRIBUTE_BYTE;
                                        abort();
                                }
                                break;

                        default:
                                fprintf(stderr,
                                        "format %s unsupported\n",
                                        desc->name);
                                abort();
                        }
                }
        }

        /* Set up the default attribute values in case any of the vertex
         * elements use them.
         */
        so->default_attribute_values = v3d_bo_alloc(v3d->screen,
                                                    VC5_MAX_ATTRIBUTES *
                                                    4 * sizeof(float),
                                                    "default_attributes");
        uint32_t *attrs = v3d_bo_map(so->default_attribute_values);
        for (int i = 0; i < VC5_MAX_ATTRIBUTES; i++) {
                attrs[i * 4 + 0] = 0;
                attrs[i * 4 + 1] = 0;
                attrs[i * 4 + 2] = 0;
                if (i < so->num_elements &&
                    util_format_is_pure_integer(so->pipe[i].src_format)) {
                        attrs[i * 4 + 3] = 1;
                } else {
                        attrs[i * 4 + 3] = fui(1.0);
                }
        }

        return so;
}

static void
v3d_vertex_state_delete(struct pipe_context *pctx, void *hwcso)
{
        struct v3d_vertex_stateobj *so = hwcso;

        v3d_bo_unreference(&so->default_attribute_values);
        free(so);
}

static void
v3d_vertex_state_bind(struct pipe_context *pctx, void *hwcso)
{
        struct v3d_context *v3d = v3d_context(pctx);
        v3d->vtx = hwcso;
        v3d->dirty |= VC5_DIRTY_VTXSTATE;
}

static void
v3d_set_constant_buffer(struct pipe_context *pctx, uint shader, uint index,
                        const struct pipe_constant_buffer *cb)
{
        struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_constbuf_stateobj *so = &v3d->constbuf[shader];

        util_copy_constant_buffer(&so->cb[index], cb);

        /* Note that the state tracker can unbind constant buffers by
         * passing NULL here.
         */
        if (unlikely(!cb)) {
                so->enabled_mask &= ~(1 << index);
                so->dirty_mask &= ~(1 << index);
                return;
        }

        so->enabled_mask |= 1 << index;
        so->dirty_mask |= 1 << index;
        v3d->dirty |= VC5_DIRTY_CONSTBUF;
}

static void
v3d_set_framebuffer_state(struct pipe_context *pctx,
                          const struct pipe_framebuffer_state *framebuffer)
{
        struct v3d_context *v3d = v3d_context(pctx);
        struct pipe_framebuffer_state *cso = &v3d->framebuffer;

        v3d->job = NULL;

        util_copy_framebuffer_state(cso, framebuffer);

        v3d->swap_color_rb = 0;
        v3d->blend_dst_alpha_one = 0;
        for (int i = 0; i < v3d->framebuffer.nr_cbufs; i++) {
                struct pipe_surface *cbuf = v3d->framebuffer.cbufs[i];
                if (!cbuf)
                        continue;

                const struct util_format_description *desc =
                        util_format_description(cbuf->format);

                /* For BGRA8 formats (DRI window system default format), we
                 * need to swap R and B, since the HW's format is RGBA8.
                 */
                if (desc->swizzle[0] == PIPE_SWIZZLE_Z &&
                    cbuf->format != PIPE_FORMAT_B5G6R5_UNORM) {
                        v3d->swap_color_rb |= 1 << i;
                }

                if (desc->swizzle[3] == PIPE_SWIZZLE_1)
                        v3d->blend_dst_alpha_one |= 1 << i;
        }

        v3d->dirty |= VC5_DIRTY_FRAMEBUFFER;
}

static struct v3d_texture_stateobj *
v3d_get_stage_tex(struct v3d_context *v3d, enum pipe_shader_type shader)
{
        switch (shader) {
        case PIPE_SHADER_FRAGMENT:
                v3d->dirty |= VC5_DIRTY_FRAGTEX;
                return &v3d->fragtex;
                break;
        case PIPE_SHADER_VERTEX:
                v3d->dirty |= VC5_DIRTY_VERTTEX;
                return &v3d->verttex;
                break;
        default:
                fprintf(stderr, "Unknown shader target %d\n", shader);
                abort();
        }
}

static uint32_t translate_wrap(uint32_t pipe_wrap, bool using_nearest)
{
        switch (pipe_wrap) {
        case PIPE_TEX_WRAP_REPEAT:
                return 0;
        case PIPE_TEX_WRAP_CLAMP_TO_EDGE:
                return 1;
        case PIPE_TEX_WRAP_MIRROR_REPEAT:
                return 2;
        case PIPE_TEX_WRAP_CLAMP_TO_BORDER:
                return 3;
        case PIPE_TEX_WRAP_CLAMP:
                return (using_nearest ? 1 : 3);
        default:
                unreachable("Unknown wrap mode");
        }
}


static void *
v3d_create_sampler_state(struct pipe_context *pctx,
                         const struct pipe_sampler_state *cso)
{
        MAYBE_UNUSED struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_sampler_state *so = CALLOC_STRUCT(v3d_sampler_state);

        if (!so)
                return NULL;

        memcpy(so, cso, sizeof(*cso));

        bool either_nearest =
                (cso->mag_img_filter == PIPE_TEX_MIPFILTER_NEAREST ||
                 cso->min_img_filter == PIPE_TEX_MIPFILTER_NEAREST);

#if V3D_VERSION >= 40
        so->bo = v3d_bo_alloc(v3d->screen, cl_packet_length(SAMPLER_STATE),
                              "sampler");
        void *map = v3d_bo_map(so->bo);

        v3dx_pack(map, SAMPLER_STATE, sampler) {
                sampler.wrap_i_border = false;

                sampler.wrap_s = translate_wrap(cso->wrap_s, either_nearest);
                sampler.wrap_t = translate_wrap(cso->wrap_t, either_nearest);
                sampler.wrap_r = translate_wrap(cso->wrap_r, either_nearest);

                sampler.fixed_bias = cso->lod_bias;
                sampler.depth_compare_function = cso->compare_func;

                sampler.min_filter_nearest =
                        cso->min_img_filter == PIPE_TEX_FILTER_NEAREST;
                sampler.mag_filter_nearest =
                        cso->mag_img_filter == PIPE_TEX_FILTER_NEAREST;
                sampler.mip_filter_nearest =
                        cso->min_mip_filter != PIPE_TEX_MIPFILTER_LINEAR;

                sampler.min_level_of_detail = MIN2(MAX2(0, cso->min_lod),
                                                   15);
                sampler.max_level_of_detail = MIN2(cso->max_lod, 15);

                /* If we're not doing inter-miplevel filtering, we need to
                 * clamp the LOD so that we only sample from baselevel.
                 * However, we need to still allow the calculated LOD to be
                 * fractionally over the baselevel, so that the HW can decide
                 * between the min and mag filters.
                 */
                if (cso->min_mip_filter == PIPE_TEX_MIPFILTER_NONE) {
                        sampler.min_level_of_detail =
                                MIN2(sampler.min_level_of_detail, 1.0 / 256.0);
                        sampler.max_level_of_detail =
                                MIN2(sampler.max_level_of_detail, 1.0 / 256.0);
                }

                if (cso->max_anisotropy) {
                        sampler.anisotropy_enable = true;

                        if (cso->max_anisotropy > 8)
                                sampler.maximum_anisotropy = 3;
                        else if (cso->max_anisotropy > 4)
                                sampler.maximum_anisotropy = 2;
                        else if (cso->max_anisotropy > 2)
                                sampler.maximum_anisotropy = 1;
                }

                sampler.border_color_mode = V3D_BORDER_COLOR_FOLLOWS;
                /* XXX: The border color field is in the TMU blending format
                 * (32, f16, or i16), and we need to customize it based on
                 * that.
                 *
                 * XXX: for compat alpha formats, we need the alpha field to
                 * be in the red channel.
                 */
                sampler.border_color_red =
                        util_float_to_half(cso->border_color.f[0]);
                sampler.border_color_green =
                        util_float_to_half(cso->border_color.f[1]);
                sampler.border_color_blue =
                        util_float_to_half(cso->border_color.f[2]);
                sampler.border_color_alpha =
                        util_float_to_half(cso->border_color.f[3]);
        }

#else /* V3D_VERSION < 40 */
        v3dx_pack(&so->p0, TEXTURE_UNIFORM_PARAMETER_0_CFG_MODE1, p0) {
                p0.s_wrap_mode = translate_wrap(cso->wrap_s, either_nearest);
                p0.t_wrap_mode = translate_wrap(cso->wrap_t, either_nearest);
                p0.r_wrap_mode = translate_wrap(cso->wrap_r, either_nearest);
        }

        v3dx_pack(&so->texture_shader_state, TEXTURE_SHADER_STATE, tex) {
                tex.depth_compare_function = cso->compare_func;
                tex.fixed_bias = cso->lod_bias;
        }
#endif /* V3D_VERSION < 40 */
        return so;
}

static void
v3d_sampler_states_bind(struct pipe_context *pctx,
                        enum pipe_shader_type shader, unsigned start,
                        unsigned nr, void **hwcso)
{
        struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_texture_stateobj *stage_tex = v3d_get_stage_tex(v3d, shader);

        assert(start == 0);
        unsigned i;
        unsigned new_nr = 0;

        for (i = 0; i < nr; i++) {
                if (hwcso[i])
                        new_nr = i + 1;
                stage_tex->samplers[i] = hwcso[i];
        }

        for (; i < stage_tex->num_samplers; i++) {
                stage_tex->samplers[i] = NULL;
        }

        stage_tex->num_samplers = new_nr;
}

static void
v3d_sampler_state_delete(struct pipe_context *pctx,
                         void *hwcso)
{
        struct pipe_sampler_state *psampler = hwcso;
        struct v3d_sampler_state *sampler = v3d_sampler_state(psampler);

        v3d_bo_unreference(&sampler->bo);
        free(psampler);
}

#if V3D_VERSION >= 40
static uint32_t
translate_swizzle(unsigned char pipe_swizzle)
{
        switch (pipe_swizzle) {
        case PIPE_SWIZZLE_0:
                return 0;
        case PIPE_SWIZZLE_1:
                return 1;
        case PIPE_SWIZZLE_X:
        case PIPE_SWIZZLE_Y:
        case PIPE_SWIZZLE_Z:
        case PIPE_SWIZZLE_W:
                return 2 + pipe_swizzle;
        default:
                unreachable("unknown swizzle");
        }
}
#endif

static struct pipe_sampler_view *
v3d_create_sampler_view(struct pipe_context *pctx, struct pipe_resource *prsc,
                        const struct pipe_sampler_view *cso)
{
        struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_screen *screen = v3d->screen;
        struct v3d_sampler_view *so = CALLOC_STRUCT(v3d_sampler_view);
        struct v3d_resource *rsc = v3d_resource(prsc);

        if (!so)
                return NULL;

        so->base = *cso;

        pipe_reference(NULL, &prsc->reference);

        /* Compute the sampler view's swizzle up front. This will be plugged
         * into either the sampler (for 16-bit returns) or the shader's
         * texture key (for 32)
         */
        uint8_t view_swizzle[4] = {
                cso->swizzle_r,
                cso->swizzle_g,
                cso->swizzle_b,
                cso->swizzle_a
        };
        const uint8_t *fmt_swizzle =
                v3d_get_format_swizzle(&screen->devinfo, so->base.format);
        util_format_compose_swizzles(fmt_swizzle, view_swizzle, so->swizzle);

        so->base.texture = prsc;
        so->base.reference.count = 1;
        so->base.context = pctx;

        int msaa_scale = prsc->nr_samples > 1 ? 2 : 1;

#if V3D_VERSION >= 40
        so->bo = v3d_bo_alloc(v3d->screen,
                              cl_packet_length(TEXTURE_SHADER_STATE), "sampler");
        void *map = v3d_bo_map(so->bo);

        v3dx_pack(map, TEXTURE_SHADER_STATE, tex) {
#else /* V3D_VERSION < 40 */
        STATIC_ASSERT(sizeof(so->texture_shader_state) >=
                      cl_packet_length(TEXTURE_SHADER_STATE));
        v3dx_pack(&so->texture_shader_state, TEXTURE_SHADER_STATE, tex) {
#endif

                tex.image_width = prsc->width0 * msaa_scale;
                tex.image_height = prsc->height0 * msaa_scale;

#if V3D_VERSION >= 40
                /* On 4.x, the height of a 1D texture is redefined to be the
                 * upper 14 bits of the width (which is only usable with txf).
                 */
                if (prsc->target == PIPE_TEXTURE_1D ||
                    prsc->target == PIPE_TEXTURE_1D_ARRAY) {
                        tex.image_height = tex.image_width >> 14;
                }
#endif

                if (prsc->target == PIPE_TEXTURE_3D) {
                        tex.image_depth = prsc->depth0;
                } else {
                        tex.image_depth = (cso->u.tex.last_layer -
                                           cso->u.tex.first_layer) + 1;
                }

                tex.srgb = util_format_is_srgb(cso->format);

                tex.base_level = cso->u.tex.first_level;
#if V3D_VERSION >= 40
                tex.max_level = cso->u.tex.last_level;
                /* Note that we don't have a job to reference the texture's sBO
                 * at state create time, so any time this sampler view is used
                 * we need to add the texture to the job.
                 */
                tex.texture_base_pointer = cl_address(NULL,
                                                      rsc->bo->offset +
                                                      rsc->slices[0].offset +
                                                      cso->u.tex.first_layer *
                                                      rsc->cube_map_stride),

                tex.swizzle_r = translate_swizzle(so->swizzle[0]);
                tex.swizzle_g = translate_swizzle(so->swizzle[1]);
                tex.swizzle_b = translate_swizzle(so->swizzle[2]);
                tex.swizzle_a = translate_swizzle(so->swizzle[3]);
#endif
                tex.array_stride_64_byte_aligned = rsc->cube_map_stride / 64;

                if (prsc->nr_samples > 1 && V3D_VERSION < 40) {
                        /* Using texture views to reinterpret formats on our
                         * MSAA textures won't work, because we don't lay out
                         * the bits in memory as it's expected -- for example,
                         * RGBA8 and RGB10_A2 are compatible in the
                         * ARB_texture_view spec, but in HW we lay them out as
                         * 32bpp RGBA8 and 64bpp RGBA16F.  Just assert for now
                         * to catch failures.
                         *
                         * We explicitly allow remapping S8Z24 to RGBA8888 for
                         * v3d_blit.c's stencil blits.
                         */
                        assert((util_format_linear(cso->format) ==
                                util_format_linear(prsc->format)) ||
                               (prsc->format == PIPE_FORMAT_S8_UINT_Z24_UNORM &&
                                cso->format == PIPE_FORMAT_R8G8B8A8_UNORM));
                        uint32_t output_image_format =
                                v3d_get_rt_format(&screen->devinfo, cso->format);
                        uint32_t internal_type;
                        uint32_t internal_bpp;
                        v3d_get_internal_type_bpp_for_output_format(&screen->devinfo,
                                                                    output_image_format,
                                                                    &internal_type,
                                                                    &internal_bpp);

                        switch (internal_type) {
                        case V3D_INTERNAL_TYPE_8:
                                tex.texture_type = TEXTURE_DATA_FORMAT_RGBA8;
                                break;
                        case V3D_INTERNAL_TYPE_16F:
                                tex.texture_type = TEXTURE_DATA_FORMAT_RGBA16F;
                                break;
                        default:
                                unreachable("Bad MSAA texture type");
                        }

                        /* sRGB was stored in the tile buffer as linear and
                         * would have been encoded to sRGB on resolved tile
                         * buffer store.  Note that this means we would need
                         * shader code if we wanted to read an MSAA sRGB
                         * texture without sRGB decode.
                         */
                        tex.srgb = false;
                } else {
                        tex.texture_type = v3d_get_tex_format(&screen->devinfo,
                                                              cso->format);
                }

                /* Since other platform devices may produce UIF images even
                 * when they're not big enough for V3D to assume they're UIF,
                 * we force images with level 0 as UIF to be always treated
                 * that way.
                 */
                tex.level_0_is_strictly_uif = (rsc->slices[0].tiling ==
                                               VC5_TILING_UIF_XOR ||
                                               rsc->slices[0].tiling ==
                                               VC5_TILING_UIF_NO_XOR);
                tex.level_0_xor_enable = (rsc->slices[0].tiling ==
                                          VC5_TILING_UIF_XOR);

                if (tex.level_0_is_strictly_uif)
                        tex.level_0_ub_pad = rsc->slices[0].ub_pad;

#if V3D_VERSION >= 40
                if (tex.uif_xor_disable ||
                    tex.level_0_is_strictly_uif) {
                        tex.extended = true;
                }
#endif /* V3D_VERSION >= 40 */
        };

        return &so->base;
}

static void
v3d_sampler_view_destroy(struct pipe_context *pctx,
                         struct pipe_sampler_view *psview)
{
        struct v3d_sampler_view *sview = v3d_sampler_view(psview);

        v3d_bo_unreference(&sview->bo);
        pipe_resource_reference(&psview->texture, NULL);
        free(psview);
}

static void
v3d_set_sampler_views(struct pipe_context *pctx,
                      enum pipe_shader_type shader,
                      unsigned start, unsigned nr,
                      struct pipe_sampler_view **views)
{
        struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_texture_stateobj *stage_tex = v3d_get_stage_tex(v3d, shader);
        unsigned i;
        unsigned new_nr = 0;

        assert(start == 0);

        for (i = 0; i < nr; i++) {
                if (views[i])
                        new_nr = i + 1;
                pipe_sampler_view_reference(&stage_tex->textures[i], views[i]);
        }

        for (; i < stage_tex->num_textures; i++) {
                pipe_sampler_view_reference(&stage_tex->textures[i], NULL);
        }

        stage_tex->num_textures = new_nr;
}

static struct pipe_stream_output_target *
v3d_create_stream_output_target(struct pipe_context *pctx,
                                struct pipe_resource *prsc,
                                unsigned buffer_offset,
                                unsigned buffer_size)
{
        struct pipe_stream_output_target *target;

        target = CALLOC_STRUCT(pipe_stream_output_target);
        if (!target)
                return NULL;

        pipe_reference_init(&target->reference, 1);
        pipe_resource_reference(&target->buffer, prsc);

        target->context = pctx;
        target->buffer_offset = buffer_offset;
        target->buffer_size = buffer_size;

        return target;
}

static void
v3d_stream_output_target_destroy(struct pipe_context *pctx,
                                 struct pipe_stream_output_target *target)
{
        pipe_resource_reference(&target->buffer, NULL);
        free(target);
}

static void
v3d_set_stream_output_targets(struct pipe_context *pctx,
                              unsigned num_targets,
                              struct pipe_stream_output_target **targets,
                              const unsigned *offsets)
{
        struct v3d_context *ctx = v3d_context(pctx);
        struct v3d_streamout_stateobj *so = &ctx->streamout;
        unsigned i;

        assert(num_targets <= ARRAY_SIZE(so->targets));

        for (i = 0; i < num_targets; i++) {
                if (offsets[i] != -1)
                        so->offsets[i] = offsets[i];

                pipe_so_target_reference(&so->targets[i], targets[i]);
        }

        for (; i < so->num_targets; i++)
                pipe_so_target_reference(&so->targets[i], NULL);

        so->num_targets = num_targets;

        ctx->dirty |= VC5_DIRTY_STREAMOUT;
}

void
v3dX(state_init)(struct pipe_context *pctx)
{
        pctx->set_blend_color = v3d_set_blend_color;
        pctx->set_stencil_ref = v3d_set_stencil_ref;
        pctx->set_clip_state = v3d_set_clip_state;
        pctx->set_sample_mask = v3d_set_sample_mask;
        pctx->set_constant_buffer = v3d_set_constant_buffer;
        pctx->set_framebuffer_state = v3d_set_framebuffer_state;
        pctx->set_polygon_stipple = v3d_set_polygon_stipple;
        pctx->set_scissor_states = v3d_set_scissor_states;
        pctx->set_viewport_states = v3d_set_viewport_states;

        pctx->set_vertex_buffers = v3d_set_vertex_buffers;

        pctx->create_blend_state = v3d_create_blend_state;
        pctx->bind_blend_state = v3d_blend_state_bind;
        pctx->delete_blend_state = v3d_generic_cso_state_delete;

        pctx->create_rasterizer_state = v3d_create_rasterizer_state;
        pctx->bind_rasterizer_state = v3d_rasterizer_state_bind;
        pctx->delete_rasterizer_state = v3d_generic_cso_state_delete;

        pctx->create_depth_stencil_alpha_state = v3d_create_depth_stencil_alpha_state;
        pctx->bind_depth_stencil_alpha_state = v3d_zsa_state_bind;
        pctx->delete_depth_stencil_alpha_state = v3d_generic_cso_state_delete;

        pctx->create_vertex_elements_state = v3d_vertex_state_create;
        pctx->delete_vertex_elements_state = v3d_vertex_state_delete;
        pctx->bind_vertex_elements_state = v3d_vertex_state_bind;

        pctx->create_sampler_state = v3d_create_sampler_state;
        pctx->delete_sampler_state = v3d_sampler_state_delete;
        pctx->bind_sampler_states = v3d_sampler_states_bind;

        pctx->create_sampler_view = v3d_create_sampler_view;
        pctx->sampler_view_destroy = v3d_sampler_view_destroy;
        pctx->set_sampler_views = v3d_set_sampler_views;

        pctx->create_stream_output_target = v3d_create_stream_output_target;
        pctx->stream_output_target_destroy = v3d_stream_output_target_destroy;
        pctx->set_stream_output_targets = v3d_set_stream_output_targets;
}
