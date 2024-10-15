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

#include "etnaviv_blend.h"

#include "etnaviv_context.h"
#include "etnaviv_screen.h"
#include "etnaviv_translate.h"
#include "hw/common.xml.h"
#include "pipe/p_defines.h"
#include "util/u_memory.h"

void *
etna_blend_state_create(struct pipe_context *pctx,
                        const struct pipe_blend_state *so)
{
   struct etna_context *ctx = etna_context(pctx);
   const struct pipe_rt_blend_state *rt0 = &so->rt[0];
   struct etna_blend_state *co = CALLOC_STRUCT(etna_blend_state);
   bool alpha_enable, logicop_enable;

   if (!co)
      return NULL;

   co->base = *so;

   /* Enable blending if
    * - blend enabled in blend state
    * - NOT source factor is ONE and destination factor ZERO for both rgb and
    *   alpha (which would mean that blending is effectively disabled)
    */
   alpha_enable = rt0->blend_enable &&
                 !(rt0->rgb_src_factor == PIPE_BLENDFACTOR_ONE &&
                   rt0->rgb_dst_factor == PIPE_BLENDFACTOR_ZERO &&
                   rt0->alpha_src_factor == PIPE_BLENDFACTOR_ONE &&
                   rt0->alpha_dst_factor == PIPE_BLENDFACTOR_ZERO);

   /* Enable separate alpha if
    * - Blending enabled (see above)
    * - NOT source factor is equal to destination factor for both rgb abd
    *   alpha (which would effectively that mean alpha is not separate)
    */
   bool separate_alpha = alpha_enable &&
                         !(rt0->rgb_src_factor == rt0->alpha_src_factor &&
                           rt0->rgb_dst_factor == rt0->alpha_dst_factor);

   if (alpha_enable) {
      co->PE_ALPHA_CONFIG =
         VIVS_PE_ALPHA_CONFIG_BLEND_ENABLE_COLOR |
         COND(separate_alpha, VIVS_PE_ALPHA_CONFIG_BLEND_SEPARATE_ALPHA) |
         VIVS_PE_ALPHA_CONFIG_SRC_FUNC_COLOR(translate_blend_factor(rt0->rgb_src_factor)) |
         VIVS_PE_ALPHA_CONFIG_SRC_FUNC_ALPHA(translate_blend_factor(rt0->alpha_src_factor)) |
         VIVS_PE_ALPHA_CONFIG_DST_FUNC_COLOR(translate_blend_factor(rt0->rgb_dst_factor)) |
         VIVS_PE_ALPHA_CONFIG_DST_FUNC_ALPHA(translate_blend_factor(rt0->alpha_dst_factor)) |
         VIVS_PE_ALPHA_CONFIG_EQ_COLOR(translate_blend(rt0->rgb_func)) |
         VIVS_PE_ALPHA_CONFIG_EQ_ALPHA(translate_blend(rt0->alpha_func));
   } else {
      co->PE_ALPHA_CONFIG = 0;
   }

   logicop_enable = so->logicop_enable &&
                    VIV_FEATURE(ctx->screen, chipMinorFeatures2, LOGIC_OP);

   co->PE_LOGIC_OP =
         VIVS_PE_LOGIC_OP_OP(logicop_enable ? so->logicop_func : LOGIC_OP_COPY) |
         0x000E4000 /* ??? */;

   co->fo_allowed = !alpha_enable && !logicop_enable;

   /* independent_blend_enable not needed: only one rt supported */
   /* XXX alpha_to_coverage / alpha_to_one? */
   /* Set dither registers based on dither status. These registers set the
    * dither pattern,
    * for now, set the same values as the blob.
    */
   if (so->dither) {
      co->PE_DITHER[0] = 0x6e4ca280;
      co->PE_DITHER[1] = 0x5d7f91b3;
   } else {
      co->PE_DITHER[0] = 0xffffffff;
      co->PE_DITHER[1] = 0xffffffff;
   }

   return co;
}

bool
etna_update_blend(struct etna_context *ctx)
{
   struct pipe_framebuffer_state *pfb = &ctx->framebuffer_s;
   struct pipe_blend_state *pblend = ctx->blend;
   struct etna_blend_state *blend = etna_blend_state(pblend);
   const struct pipe_rt_blend_state *rt0 = &pblend->rt[0];
   uint32_t colormask;

   if (pfb->cbufs[0] &&
       translate_rs_format_rb_swap(pfb->cbufs[0]->texture->format)) {
      colormask = rt0->colormask & (PIPE_MASK_A | PIPE_MASK_G);
      if (rt0->colormask & PIPE_MASK_R)
         colormask |= PIPE_MASK_B;
      if (rt0->colormask & PIPE_MASK_B)
         colormask |= PIPE_MASK_R;
   } else {
      colormask = rt0->colormask;
   }

   /* If the complete render target is written, set full_overwrite:
    * - The color mask is 1111
    * - No blending is used
    */
   bool full_overwrite = ((rt0->colormask == 0xf) && blend->fo_allowed) ||
                         !pfb->cbufs[0];
   blend->PE_COLOR_FORMAT =
            VIVS_PE_COLOR_FORMAT_COMPONENTS(colormask) |
            COND(full_overwrite, VIVS_PE_COLOR_FORMAT_OVERWRITE);

   return true;
}

void
etna_set_blend_color(struct pipe_context *pctx, const struct pipe_blend_color *bc)
{
   struct etna_context *ctx = etna_context(pctx);
   struct compiled_blend_color *cs = &ctx->blend_color;

   memcpy(cs->color, bc->color, sizeof(float) * 4);

   ctx->dirty |= ETNA_DIRTY_BLEND_COLOR;
}

bool
etna_update_blend_color(struct etna_context *ctx)
{
   struct pipe_framebuffer_state *pfb = &ctx->framebuffer_s;
   struct compiled_blend_color *cs = &ctx->blend_color;

   if (pfb->cbufs[0] &&
       translate_rs_format_rb_swap(pfb->cbufs[0]->texture->format)) {
      cs->PE_ALPHA_BLEND_COLOR =
         VIVS_PE_ALPHA_BLEND_COLOR_R(etna_cfloat_to_uint8(cs->color[2])) |
         VIVS_PE_ALPHA_BLEND_COLOR_G(etna_cfloat_to_uint8(cs->color[1])) |
         VIVS_PE_ALPHA_BLEND_COLOR_B(etna_cfloat_to_uint8(cs->color[0])) |
         VIVS_PE_ALPHA_BLEND_COLOR_A(etna_cfloat_to_uint8(cs->color[3]));
   } else {
      cs->PE_ALPHA_BLEND_COLOR =
         VIVS_PE_ALPHA_BLEND_COLOR_R(etna_cfloat_to_uint8(cs->color[0])) |
         VIVS_PE_ALPHA_BLEND_COLOR_G(etna_cfloat_to_uint8(cs->color[1])) |
         VIVS_PE_ALPHA_BLEND_COLOR_B(etna_cfloat_to_uint8(cs->color[2])) |
         VIVS_PE_ALPHA_BLEND_COLOR_A(etna_cfloat_to_uint8(cs->color[3]));
	}

	return true;
}
