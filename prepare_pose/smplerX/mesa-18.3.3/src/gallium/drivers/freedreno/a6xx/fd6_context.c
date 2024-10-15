/*
 * Copyright (C) 2016 Rob Clark <robclark@freedesktop.org>
 * Copyright © 2018 Google, Inc.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "freedreno_query_acc.h"

#include "fd6_context.h"
#include "fd6_blend.h"
#include "fd6_blitter.h"
#include "fd6_draw.h"
#include "fd6_emit.h"
#include "fd6_gmem.h"
#include "fd6_program.h"
#include "fd6_query.h"
#include "fd6_rasterizer.h"
#include "fd6_texture.h"
#include "fd6_zsa.h"

static void
fd6_context_destroy(struct pipe_context *pctx)
{
	struct fd6_context *fd6_ctx = fd6_context(fd_context(pctx));

	u_upload_destroy(fd6_ctx->border_color_uploader);

	fd_context_destroy(pctx);

	fd_bo_del(fd6_ctx->vs_pvt_mem);
	fd_bo_del(fd6_ctx->fs_pvt_mem);
	fd_bo_del(fd6_ctx->vsc_data);
	fd_bo_del(fd6_ctx->vsc_data2);
	fd_bo_del(fd6_ctx->blit_mem);

	fd_context_cleanup_common_vbos(&fd6_ctx->base);

	ir3_cache_destroy(fd6_ctx->shader_cache);

	fd6_texture_fini(pctx);

	free(fd6_ctx);
}

static const uint8_t primtypes[] = {
		[PIPE_PRIM_POINTS]         = DI_PT_POINTLIST,
		[PIPE_PRIM_LINES]          = DI_PT_LINELIST,
		[PIPE_PRIM_LINE_STRIP]     = DI_PT_LINESTRIP,
		[PIPE_PRIM_LINE_LOOP]      = DI_PT_LINELOOP,
		[PIPE_PRIM_TRIANGLES]      = DI_PT_TRILIST,
		[PIPE_PRIM_TRIANGLE_STRIP] = DI_PT_TRISTRIP,
		[PIPE_PRIM_TRIANGLE_FAN]   = DI_PT_TRIFAN,
		[PIPE_PRIM_MAX]            = DI_PT_RECTLIST,  /* internal clear blits */
};

struct pipe_context *
fd6_context_create(struct pipe_screen *pscreen, void *priv, unsigned flags)
{
	struct fd_screen *screen = fd_screen(pscreen);
	struct fd6_context *fd6_ctx = CALLOC_STRUCT(fd6_context);
	struct pipe_context *pctx;

	if (!fd6_ctx)
		return NULL;

	pctx = &fd6_ctx->base.base;

	fd6_ctx->base.dev = fd_device_ref(screen->dev);
	fd6_ctx->base.screen = fd_screen(pscreen);

	pctx->destroy = fd6_context_destroy;
	pctx->create_blend_state = fd6_blend_state_create;
	pctx->create_rasterizer_state = fd6_rasterizer_state_create;
	pctx->create_depth_stencil_alpha_state = fd6_zsa_state_create;

	fd6_draw_init(pctx);
	fd6_gmem_init(pctx);
	fd6_texture_init(pctx);
	fd6_prog_init(pctx);
	fd6_emit_init(pctx);

	pctx = fd_context_init(&fd6_ctx->base, pscreen, primtypes, priv, flags);
	if (!pctx)
		return NULL;

	/* fd_context_init overwrites delete_rasterizer_state, so set this
	 * here. */
	pctx->delete_rasterizer_state = fd6_rasterizer_state_delete;
	pctx->delete_depth_stencil_alpha_state = fd6_depth_stencil_alpha_state_delete;

	fd6_ctx->vs_pvt_mem = fd_bo_new(screen->dev, 0x2000,
			DRM_FREEDRENO_GEM_TYPE_KMEM);

	fd6_ctx->fs_pvt_mem = fd_bo_new(screen->dev, 0x2000,
			DRM_FREEDRENO_GEM_TYPE_KMEM);

	fd6_ctx->vsc_data = fd_bo_new(screen->dev,
			(A6XX_VSC_DATA_PITCH * 32) + 0x100,
			DRM_FREEDRENO_GEM_TYPE_KMEM);

	fd6_ctx->vsc_data2 = fd_bo_new(screen->dev,
			A6XX_VSC_DATA2_PITCH * 32,
			DRM_FREEDRENO_GEM_TYPE_KMEM);

	fd6_ctx->blit_mem = fd_bo_new(screen->dev, 0x1000,
			DRM_FREEDRENO_GEM_TYPE_KMEM);

	fd_context_setup_common_vbos(&fd6_ctx->base);

	fd6_query_context_init(pctx);
	fd6_blitter_init(pctx);

	fd6_ctx->border_color_uploader = u_upload_create(pctx, 4096, 0,
                                                         PIPE_USAGE_STREAM, 0);

	return pctx;
}
