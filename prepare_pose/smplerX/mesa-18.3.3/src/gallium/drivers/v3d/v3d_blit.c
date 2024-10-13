/*
 * Copyright © 2015-2017 Broadcom
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

#include "util/u_format.h"
#include "util/u_surface.h"
#include "util/u_blitter.h"
#include "v3d_context.h"

#if 0
static struct pipe_surface *
v3d_get_blit_surface(struct pipe_context *pctx,
                     struct pipe_resource *prsc, unsigned level)
{
        struct pipe_surface tmpl;

        memset(&tmpl, 0, sizeof(tmpl));
        tmpl.format = prsc->format;
        tmpl.u.tex.level = level;
        tmpl.u.tex.first_layer = 0;
        tmpl.u.tex.last_layer = 0;

        return pctx->create_surface(pctx, prsc, &tmpl);
}

static bool
is_tile_unaligned(unsigned size, unsigned tile_size)
{
        return size & (tile_size - 1);
}

static bool
v3d_tile_blit(struct pipe_context *pctx, const struct pipe_blit_info *info)
{
        struct v3d_context *v3d = v3d_context(pctx);
        bool msaa = (info->src.resource->nr_samples > 1 ||
                     info->dst.resource->nr_samples > 1);
        int tile_width = msaa ? 32 : 64;
        int tile_height = msaa ? 32 : 64;

        if (util_format_is_depth_or_stencil(info->dst.resource->format))
                return false;

        if (info->scissor_enable)
                return false;

        if ((info->mask & PIPE_MASK_RGBA) == 0)
                return false;

        if (info->dst.box.x != info->src.box.x ||
            info->dst.box.y != info->src.box.y ||
            info->dst.box.width != info->src.box.width ||
            info->dst.box.height != info->src.box.height) {
                return false;
        }

        int dst_surface_width = u_minify(info->dst.resource->width0,
                                         info->dst.level);
        int dst_surface_height = u_minify(info->dst.resource->height0,
                                         info->dst.level);
        if (is_tile_unaligned(info->dst.box.x, tile_width) ||
            is_tile_unaligned(info->dst.box.y, tile_height) ||
            (is_tile_unaligned(info->dst.box.width, tile_width) &&
             info->dst.box.x + info->dst.box.width != dst_surface_width) ||
            (is_tile_unaligned(info->dst.box.height, tile_height) &&
             info->dst.box.y + info->dst.box.height != dst_surface_height)) {
                return false;
        }

        /* VC5_PACKET_LOAD_TILE_BUFFER_GENERAL uses the
         * VC5_PACKET_TILE_RENDERING_MODE_CONFIG's width (determined by our
         * destination surface) to determine the stride.  This may be wrong
         * when reading from texture miplevels > 0, which are stored in
         * POT-sized areas.  For MSAA, the tile addresses are computed
         * explicitly by the RCL, but still use the destination width to
         * determine the stride (which could be fixed by explicitly supplying
         * it in the ABI).
         */
        struct v3d_resource *rsc = v3d_resource(info->src.resource);

        uint32_t stride;

        if (info->src.resource->nr_samples > 1)
                stride = align(dst_surface_width, 32) * 4 * rsc->cpp;
        /* XXX else if (rsc->slices[info->src.level].tiling == VC5_TILING_FORMAT_T)
           stride = align(dst_surface_width * rsc->cpp, 128); */
        else
                stride = align(dst_surface_width * rsc->cpp, 16);

        if (stride != rsc->slices[info->src.level].stride)
                return false;

        if (info->dst.resource->format != info->src.resource->format)
                return false;

        if (false) {
                fprintf(stderr, "RCL blit from %d,%d to %d,%d (%d,%d)\n",
                        info->src.box.x,
                        info->src.box.y,
                        info->dst.box.x,
                        info->dst.box.y,
                        info->dst.box.width,
                        info->dst.box.height);
        }

        struct pipe_surface *dst_surf =
                v3d_get_blit_surface(pctx, info->dst.resource, info->dst.level);
        struct pipe_surface *src_surf =
                v3d_get_blit_surface(pctx, info->src.resource, info->src.level);

        v3d_flush_jobs_reading_resource(v3d, info->src.resource);

        struct v3d_job *job = v3d_get_job(v3d, dst_surf, NULL);
        pipe_surface_reference(&job->color_read, src_surf);

        /* If we're resolving from MSAA to single sample, we still need to run
         * the engine in MSAA mode for the load.
         */
        if (!job->msaa && info->src.resource->nr_samples > 1) {
                job->msaa = true;
                job->tile_width = 32;
                job->tile_height = 32;
        }

        job->draw_min_x = info->dst.box.x;
        job->draw_min_y = info->dst.box.y;
        job->draw_max_x = info->dst.box.x + info->dst.box.width;
        job->draw_max_y = info->dst.box.y + info->dst.box.height;
        job->draw_width = dst_surf->width;
        job->draw_height = dst_surf->height;

        job->tile_width = tile_width;
        job->tile_height = tile_height;
        job->msaa = msaa;
        job->needs_flush = true;
        job->resolve |= PIPE_CLEAR_COLOR;

        v3d_job_submit(v3d, job);

        pipe_surface_reference(&dst_surf, NULL);
        pipe_surface_reference(&src_surf, NULL);

        return true;
}
#endif

void
v3d_blitter_save(struct v3d_context *v3d)
{
        util_blitter_save_fragment_constant_buffer_slot(v3d->blitter,
                                                        v3d->constbuf[PIPE_SHADER_FRAGMENT].cb);
        util_blitter_save_vertex_buffer_slot(v3d->blitter, v3d->vertexbuf.vb);
        util_blitter_save_vertex_elements(v3d->blitter, v3d->vtx);
        util_blitter_save_vertex_shader(v3d->blitter, v3d->prog.bind_vs);
        util_blitter_save_so_targets(v3d->blitter, v3d->streamout.num_targets,
                                     v3d->streamout.targets);
        util_blitter_save_rasterizer(v3d->blitter, v3d->rasterizer);
        util_blitter_save_viewport(v3d->blitter, &v3d->viewport);
        util_blitter_save_scissor(v3d->blitter, &v3d->scissor);
        util_blitter_save_fragment_shader(v3d->blitter, v3d->prog.bind_fs);
        util_blitter_save_blend(v3d->blitter, v3d->blend);
        util_blitter_save_depth_stencil_alpha(v3d->blitter, v3d->zsa);
        util_blitter_save_stencil_ref(v3d->blitter, &v3d->stencil_ref);
        util_blitter_save_sample_mask(v3d->blitter, v3d->sample_mask);
        util_blitter_save_framebuffer(v3d->blitter, &v3d->framebuffer);
        util_blitter_save_fragment_sampler_states(v3d->blitter,
                        v3d->fragtex.num_samplers,
                        (void **)v3d->fragtex.samplers);
        util_blitter_save_fragment_sampler_views(v3d->blitter,
                        v3d->fragtex.num_textures, v3d->fragtex.textures);
        util_blitter_save_so_targets(v3d->blitter, v3d->streamout.num_targets,
                                     v3d->streamout.targets);
}

static bool
v3d_render_blit(struct pipe_context *ctx, struct pipe_blit_info *info)
{
        struct v3d_context *v3d = v3d_context(ctx);
        struct v3d_resource *src = v3d_resource(info->src.resource);
        struct pipe_resource *tiled = NULL;

        if (!src->tiled) {
                struct pipe_box box = {
                        .x = 0,
                        .y = 0,
                        .width = u_minify(info->src.resource->width0,
                                           info->src.level),
                        .height = u_minify(info->src.resource->height0,
                                           info->src.level),
                        .depth = 1,
                };
                struct pipe_resource tmpl = {
                        .target = info->src.resource->target,
                        .format = info->src.resource->format,
                        .width0 = box.width,
                        .height0 = box.height,
                        .depth0 = 1,
                        .array_size = 1,
                };
                tiled = ctx->screen->resource_create(ctx->screen, &tmpl);
                if (!tiled) {
                        fprintf(stderr, "Failed to create tiled blit temp\n");
                        return false;
                }
                ctx->resource_copy_region(ctx,
                                          tiled, 0,
                                          0, 0, 0,
                                          info->src.resource, info->src.level,
                                          &box);
                info->src.level = 0;
                info->src.resource = tiled;
        }

        if (!util_blitter_is_blit_supported(v3d->blitter, info)) {
                fprintf(stderr, "blit unsupported %s -> %s\n",
                    util_format_short_name(info->src.resource->format),
                    util_format_short_name(info->dst.resource->format));
                return false;
        }

        v3d_blitter_save(v3d);
        util_blitter_blit(v3d->blitter, info);

        pipe_resource_reference(&tiled, NULL);

        return true;
}

/* Implement stencil blits by reinterpreting the stencil data as an RGBA8888
 * or R8 texture.
 */
static void
v3d_stencil_blit(struct pipe_context *ctx, const struct pipe_blit_info *info)
{
        struct v3d_context *v3d = v3d_context(ctx);
        struct v3d_resource *src = v3d_resource(info->src.resource);
        struct v3d_resource *dst = v3d_resource(info->dst.resource);
        enum pipe_format src_format, dst_format;

        if (src->separate_stencil) {
                src = src->separate_stencil;
                src_format = PIPE_FORMAT_R8_UNORM;
        } else {
                src_format = PIPE_FORMAT_RGBA8888_UNORM;
        }

        if (dst->separate_stencil) {
                dst = dst->separate_stencil;
                dst_format = PIPE_FORMAT_R8_UNORM;
        } else {
                dst_format = PIPE_FORMAT_RGBA8888_UNORM;
        }

        /* Initialize the surface. */
        struct pipe_surface dst_tmpl = {
                .u.tex = {
                        .level = info->dst.level,
                        .first_layer = info->dst.box.z,
                        .last_layer = info->dst.box.z,
                },
                .format = dst_format,
        };
        struct pipe_surface *dst_surf =
                ctx->create_surface(ctx, &dst->base, &dst_tmpl);

        /* Initialize the sampler view. */
        struct pipe_sampler_view src_tmpl = {
                .target = src->base.target,
                .format = src_format,
                .u.tex = {
                        .first_level = info->src.level,
                        .last_level = info->src.level,
                        .first_layer = 0,
                        .last_layer = (PIPE_TEXTURE_3D ?
                                       u_minify(src->base.depth0,
                                                info->src.level) - 1 :
                                       src->base.array_size - 1),
                },
                .swizzle_r = PIPE_SWIZZLE_X,
                .swizzle_g = PIPE_SWIZZLE_Y,
                .swizzle_b = PIPE_SWIZZLE_Z,
                .swizzle_a = PIPE_SWIZZLE_W,
        };
        struct pipe_sampler_view *src_view =
                ctx->create_sampler_view(ctx, &src->base, &src_tmpl);

        v3d_blitter_save(v3d);
        util_blitter_blit_generic(v3d->blitter, dst_surf, &info->dst.box,
                                  src_view, &info->src.box,
                                  src->base.width0, src->base.height0,
                                  PIPE_MASK_R,
                                  PIPE_TEX_FILTER_NEAREST,
                                  info->scissor_enable ? &info->scissor : NULL,
                                  info->alpha_blend);

        pipe_surface_reference(&dst_surf, NULL);
        pipe_sampler_view_reference(&src_view, NULL);
}

/* Optimal hardware path for blitting pixels.
 * Scaling, format conversion, up- and downsampling (resolve) are allowed.
 */
void
v3d_blit(struct pipe_context *pctx, const struct pipe_blit_info *blit_info)
{
        struct pipe_blit_info info = *blit_info;

        if (info.mask & PIPE_MASK_S) {
                v3d_stencil_blit(pctx, blit_info);
                info.mask &= ~PIPE_MASK_S;
        }

#if 0
        if (v3d_tile_blit(pctx, blit_info))
                return;
#endif

        v3d_render_blit(pctx, &info);
}
