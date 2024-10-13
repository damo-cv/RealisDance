/*
 * Copyright (c) 2012-2015 Etnaviv Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Wladimir J. van der Laan <laanwj@gmail.com>
 */

#include "etnaviv_texture_state.h"

#include "hw/common.xml.h"

#include "etnaviv_clear_blit.h"
#include "etnaviv_context.h"
#include "etnaviv_emit.h"
#include "etnaviv_format.h"
#include "etnaviv_texture.h"
#include "etnaviv_translate.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"

#include <drm_fourcc.h>

static void *
etna_create_sampler_state_state(struct pipe_context *pipe,
                          const struct pipe_sampler_state *ss)
{
   struct etna_sampler_state *cs = CALLOC_STRUCT(etna_sampler_state);

   if (!cs)
      return NULL;

   cs->TE_SAMPLER_CONFIG0 =
      VIVS_TE_SAMPLER_CONFIG0_UWRAP(translate_texture_wrapmode(ss->wrap_s)) |
      VIVS_TE_SAMPLER_CONFIG0_VWRAP(translate_texture_wrapmode(ss->wrap_t)) |
      VIVS_TE_SAMPLER_CONFIG0_MIN(translate_texture_filter(ss->min_img_filter)) |
      VIVS_TE_SAMPLER_CONFIG0_MIP(translate_texture_mipfilter(ss->min_mip_filter)) |
      VIVS_TE_SAMPLER_CONFIG0_MAG(translate_texture_filter(ss->mag_img_filter)) |
      COND(ss->normalized_coords, VIVS_TE_SAMPLER_CONFIG0_ROUND_UV);
   cs->TE_SAMPLER_CONFIG1 = 0; /* VIVS_TE_SAMPLER_CONFIG1 (swizzle, extended
                                  format) fully determined by sampler view */
   cs->TE_SAMPLER_LOD_CONFIG =
      COND(ss->lod_bias != 0.0, VIVS_TE_SAMPLER_LOD_CONFIG_BIAS_ENABLE) |
      VIVS_TE_SAMPLER_LOD_CONFIG_BIAS(etna_float_to_fixp55(ss->lod_bias));

   if (ss->min_mip_filter != PIPE_TEX_MIPFILTER_NONE) {
      cs->min_lod = etna_float_to_fixp55(ss->min_lod);
      cs->max_lod = etna_float_to_fixp55(ss->max_lod);
   } else {
      /* when not mipmapping, we need to set max/min lod so that always
       * lowest LOD is selected */
      cs->min_lod = cs->max_lod = etna_float_to_fixp55(ss->min_lod);
   }

   return cs;
}

static void
etna_delete_sampler_state_state(struct pipe_context *pctx, void *ss)
{
   FREE(ss);
}

static struct pipe_sampler_view *
etna_create_sampler_view_state(struct pipe_context *pctx, struct pipe_resource *prsc,
                         const struct pipe_sampler_view *so)
{
   struct etna_sampler_view *sv = CALLOC_STRUCT(etna_sampler_view);
   struct etna_context *ctx = etna_context(pctx);
   const uint32_t format = translate_texture_format(so->format);
   const bool ext = !!(format & EXT_FORMAT);
   const bool astc = !!(format & ASTC_FORMAT);
   const uint32_t swiz = get_texture_swiz(so->format, so->swizzle_r,
                                          so->swizzle_g, so->swizzle_b,
                                          so->swizzle_a);

   if (!sv)
      return NULL;

   struct etna_resource *res = etna_texture_handle_incompatible(pctx, prsc);
   if (!res) {
      free(sv);
      return NULL;
   }

   sv->base = *so;
   pipe_reference_init(&sv->base.reference, 1);
   sv->base.texture = NULL;
   pipe_resource_reference(&sv->base.texture, prsc);
   sv->base.context = pctx;

   /* merged with sampler state */
   sv->TE_SAMPLER_CONFIG0 = COND(!ext && !astc, VIVS_TE_SAMPLER_CONFIG0_FORMAT(format));
   sv->TE_SAMPLER_CONFIG0_MASK = 0xffffffff;

   switch (sv->base.target) {
   case PIPE_TEXTURE_1D:
      /* For 1D textures, we will have a height of 1, so we can use 2D
       * but set T wrap to repeat */
      sv->TE_SAMPLER_CONFIG0_MASK = ~VIVS_TE_SAMPLER_CONFIG0_VWRAP__MASK;
      sv->TE_SAMPLER_CONFIG0 |= VIVS_TE_SAMPLER_CONFIG0_VWRAP(TEXTURE_WRAPMODE_REPEAT);
      /* fallthrough */
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_RECT:
      sv->TE_SAMPLER_CONFIG0 |= VIVS_TE_SAMPLER_CONFIG0_TYPE(TEXTURE_TYPE_2D);
      break;
   case PIPE_TEXTURE_CUBE:
      sv->TE_SAMPLER_CONFIG0 |= VIVS_TE_SAMPLER_CONFIG0_TYPE(TEXTURE_TYPE_CUBE_MAP);
      break;
   default:
      BUG("Unhandled texture target");
      free(sv);
      return NULL;
   }

   sv->TE_SAMPLER_CONFIG1 = COND(ext, VIVS_TE_SAMPLER_CONFIG1_FORMAT_EXT(format)) |
                            COND(astc, VIVS_TE_SAMPLER_CONFIG1_FORMAT_EXT(TEXTURE_FORMAT_EXT_ASTC)) |
                            VIVS_TE_SAMPLER_CONFIG1_HALIGN(res->halign) | swiz;
   sv->TE_SAMPLER_ASTC0 = COND(astc, VIVS_NTE_SAMPLER_ASTC0_ASTC_FORMAT(format)) |
                          VIVS_NTE_SAMPLER_ASTC0_UNK8(0xc) |
                          VIVS_NTE_SAMPLER_ASTC0_UNK16(0xc) |
                          VIVS_NTE_SAMPLER_ASTC0_UNK24(0xc);
   sv->TE_SAMPLER_SIZE = VIVS_TE_SAMPLER_SIZE_WIDTH(res->base.width0) |
                         VIVS_TE_SAMPLER_SIZE_HEIGHT(res->base.height0);
   sv->TE_SAMPLER_LOG_SIZE =
      VIVS_TE_SAMPLER_LOG_SIZE_WIDTH(etna_log2_fixp55(res->base.width0)) |
      VIVS_TE_SAMPLER_LOG_SIZE_HEIGHT(etna_log2_fixp55(res->base.height0)) |
      COND(util_format_is_srgb(so->format) && !astc, VIVS_TE_SAMPLER_LOG_SIZE_SRGB) |
      COND(astc, VIVS_TE_SAMPLER_LOG_SIZE_ASTC);

   /* Set up levels-of-detail */
   for (int lod = 0; lod <= res->base.last_level; ++lod) {
      sv->TE_SAMPLER_LOD_ADDR[lod].bo = res->bo;
      sv->TE_SAMPLER_LOD_ADDR[lod].offset = res->levels[lod].offset;
      sv->TE_SAMPLER_LOD_ADDR[lod].flags = ETNA_RELOC_READ;
   }
   sv->min_lod = sv->base.u.tex.first_level << 5;
   sv->max_lod = MIN2(sv->base.u.tex.last_level, res->base.last_level) << 5;

   /* Workaround for npot textures -- it appears that only CLAMP_TO_EDGE is
    * supported when the appropriate capability is not set. */
   if (!ctx->specs.npot_tex_any_wrap &&
       (!util_is_power_of_two_or_zero(res->base.width0) ||
        !util_is_power_of_two_or_zero(res->base.height0))) {
      sv->TE_SAMPLER_CONFIG0_MASK = ~(VIVS_TE_SAMPLER_CONFIG0_UWRAP__MASK |
                                      VIVS_TE_SAMPLER_CONFIG0_VWRAP__MASK);
      sv->TE_SAMPLER_CONFIG0 |=
         VIVS_TE_SAMPLER_CONFIG0_UWRAP(TEXTURE_WRAPMODE_CLAMP_TO_EDGE) |
         VIVS_TE_SAMPLER_CONFIG0_VWRAP(TEXTURE_WRAPMODE_CLAMP_TO_EDGE);
   }

   return &sv->base;
}

static void
etna_sampler_view_state_destroy(struct pipe_context *pctx,
                          struct pipe_sampler_view *view)
{
   pipe_resource_reference(&view->texture, NULL);
   FREE(view);
}

#define EMIT_STATE(state_name, src_value) \
   etna_coalsence_emit(stream, &coalesce, VIVS_##state_name, src_value)

#define EMIT_STATE_FIXP(state_name, src_value) \
   etna_coalsence_emit_fixp(stream, &coalesce, VIVS_##state_name, src_value)

#define EMIT_STATE_RELOC(state_name, src_value) \
   etna_coalsence_emit_reloc(stream, &coalesce, VIVS_##state_name, src_value)

/* Emit plain (non-descriptor) texture state */
static void
etna_emit_texture_state(struct etna_context *ctx)
{
   struct etna_cmd_stream *stream = ctx->stream;
   uint32_t active_samplers = active_samplers_bits(ctx);
   uint32_t dirty = ctx->dirty;
   struct etna_coalesce coalesce;

   etna_coalesce_start(stream, &coalesce);

   if (unlikely(dirty & ETNA_DIRTY_SAMPLER_VIEWS)) {
      for (int x = 0; x < VIVS_TS_SAMPLER__LEN; ++x) {
         if ((1 << x) & active_samplers) {
            struct etna_sampler_view *sv = etna_sampler_view(ctx->sampler_view[x]);
            /*01720*/ EMIT_STATE(TS_SAMPLER_CONFIG(x), sv->ts.TS_SAMPLER_CONFIG);
         }
      }
      for (int x = 0; x < VIVS_TS_SAMPLER__LEN; ++x) {
         if ((1 << x) & active_samplers) {
            struct etna_sampler_view *sv = etna_sampler_view(ctx->sampler_view[x]);
            /*01740*/ EMIT_STATE_RELOC(TS_SAMPLER_STATUS_BASE(x), &sv->ts.TS_SAMPLER_STATUS_BASE);
         }
      }
      for (int x = 0; x < VIVS_TS_SAMPLER__LEN; ++x) {
         if ((1 << x) & active_samplers) {
            struct etna_sampler_view *sv = etna_sampler_view(ctx->sampler_view[x]);
            /*01760*/ EMIT_STATE(TS_SAMPLER_CLEAR_VALUE(x), sv->ts.TS_SAMPLER_CLEAR_VALUE);
         }
      }
      for (int x = 0; x < VIVS_TS_SAMPLER__LEN; ++x) {
         if ((1 << x) & active_samplers) {
            struct etna_sampler_view *sv = etna_sampler_view(ctx->sampler_view[x]);
            /*01780*/ EMIT_STATE(TS_SAMPLER_CLEAR_VALUE2(x), sv->ts.TS_SAMPLER_CLEAR_VALUE2);
         }
      }
   }
   if (unlikely(dirty & (ETNA_DIRTY_SAMPLER_VIEWS | ETNA_DIRTY_SAMPLERS))) {
      for (int x = 0; x < VIVS_TE_SAMPLER__LEN; ++x) {
         uint32_t val = 0; /* 0 == sampler inactive */

         /* set active samplers to their configuration value (determined by both
          * the sampler state and sampler view) */
         if ((1 << x) & active_samplers) {
            struct etna_sampler_state *ss = etna_sampler_state(ctx->sampler[x]);
            struct etna_sampler_view *sv = etna_sampler_view(ctx->sampler_view[x]);

            val = (ss->TE_SAMPLER_CONFIG0 & sv->TE_SAMPLER_CONFIG0_MASK) |
                  sv->TE_SAMPLER_CONFIG0;
         }

         /*02000*/ EMIT_STATE(TE_SAMPLER_CONFIG0(x), val);
      }
   }
   if (unlikely(dirty & (ETNA_DIRTY_SAMPLER_VIEWS))) {
      struct etna_sampler_view *sv;

      for (int x = 0; x < VIVS_TE_SAMPLER__LEN; ++x) {
         if ((1 << x) & active_samplers) {
            sv = etna_sampler_view(ctx->sampler_view[x]);
            /*02040*/ EMIT_STATE(TE_SAMPLER_SIZE(x), sv->TE_SAMPLER_SIZE);
         }
      }
      for (int x = 0; x < VIVS_TE_SAMPLER__LEN; ++x) {
         if ((1 << x) & active_samplers) {
            sv = etna_sampler_view(ctx->sampler_view[x]);
            /*02080*/ EMIT_STATE(TE_SAMPLER_LOG_SIZE(x), sv->TE_SAMPLER_LOG_SIZE);
         }
      }
   }
   if (unlikely(dirty & (ETNA_DIRTY_SAMPLER_VIEWS | ETNA_DIRTY_SAMPLERS))) {
      struct etna_sampler_state *ss;
      struct etna_sampler_view *sv;

      for (int x = 0; x < VIVS_TE_SAMPLER__LEN; ++x) {
         if ((1 << x) & active_samplers) {
            ss = etna_sampler_state(ctx->sampler[x]);
            sv = etna_sampler_view(ctx->sampler_view[x]);

            /* min and max lod is determined both by the sampler and the view */
            /*020C0*/ EMIT_STATE(TE_SAMPLER_LOD_CONFIG(x),
                                 ss->TE_SAMPLER_LOD_CONFIG |
                                 VIVS_TE_SAMPLER_LOD_CONFIG_MAX(MIN2(ss->max_lod, sv->max_lod)) |
                                 VIVS_TE_SAMPLER_LOD_CONFIG_MIN(MAX2(ss->min_lod, sv->min_lod)));
         }
      }
      for (int x = 0; x < VIVS_TE_SAMPLER__LEN; ++x) {
         if ((1 << x) & active_samplers) {
            ss = etna_sampler_state(ctx->sampler[x]);
            sv = etna_sampler_view(ctx->sampler_view[x]);

            /*021C0*/ EMIT_STATE(TE_SAMPLER_CONFIG1(x), ss->TE_SAMPLER_CONFIG1 |
                                                        sv->TE_SAMPLER_CONFIG1 |
                                                        COND(sv->ts.enable, VIVS_TE_SAMPLER_CONFIG1_USE_TS));
         }
      }
   }
   if (unlikely(dirty & (ETNA_DIRTY_SAMPLER_VIEWS))) {
      for (int y = 0; y < VIVS_TE_SAMPLER_LOD_ADDR__LEN; ++y) {
         for (int x = 0; x < VIVS_TE_SAMPLER__LEN; ++x) {
            if ((1 << x) & active_samplers) {
               struct etna_sampler_view *sv = etna_sampler_view(ctx->sampler_view[x]);
               /*02400*/ EMIT_STATE_RELOC(TE_SAMPLER_LOD_ADDR(x, y),&sv->TE_SAMPLER_LOD_ADDR[y]);
            }
         }
      }
   }
   if (unlikely(ctx->specs.tex_astc && (dirty & (ETNA_DIRTY_SAMPLER_VIEWS)))) {
      for (int x = 0; x < VIVS_TE_SAMPLER__LEN; ++x) {
         if ((1 << x) & active_samplers) {
            struct etna_sampler_view *sv = etna_sampler_view(ctx->sampler_view[x]);
            /*10500*/ EMIT_STATE(NTE_SAMPLER_ASTC0(x), sv->TE_SAMPLER_ASTC0);
         }
      }
   }
   etna_coalesce_end(stream, &coalesce);
}

#undef EMIT_STATE
#undef EMIT_STATE_FIXP
#undef EMIT_STATE_RELOC

static struct etna_sampler_ts*
etna_ts_for_sampler_view_state(struct pipe_sampler_view *pview)
{
   struct etna_sampler_view *sv = etna_sampler_view(pview);
   return &sv->ts;
}

void
etna_texture_state_init(struct pipe_context *pctx)
{
   struct etna_context *ctx = etna_context(pctx);
   DBG("etnaviv: Using state-based texturing");
   ctx->base.create_sampler_state = etna_create_sampler_state_state;
   ctx->base.delete_sampler_state = etna_delete_sampler_state_state;
   ctx->base.create_sampler_view = etna_create_sampler_view_state;
   ctx->base.sampler_view_destroy = etna_sampler_view_state_destroy;
   ctx->emit_texture_state = etna_emit_texture_state;
   ctx->ts_for_sampler_view = etna_ts_for_sampler_view_state;
}
