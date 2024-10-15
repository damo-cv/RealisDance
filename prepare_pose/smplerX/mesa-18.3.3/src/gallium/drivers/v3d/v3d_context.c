/*
 * Copyright © 2014-2017 Broadcom
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

#include <xf86drm.h>
#include <err.h>

#include "pipe/p_defines.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_blitter.h"
#include "util/u_upload_mgr.h"
#include "indices/u_primconvert.h"
#include "pipe/p_screen.h"

#include "v3d_screen.h"
#include "v3d_context.h"
#include "v3d_resource.h"

void
v3d_flush(struct pipe_context *pctx)
{
        struct v3d_context *v3d = v3d_context(pctx);

        hash_table_foreach(v3d->jobs, entry) {
                struct v3d_job *job = entry->data;
                v3d_job_submit(v3d, job);
        }
}

static void
v3d_pipe_flush(struct pipe_context *pctx, struct pipe_fence_handle **fence,
               unsigned flags)
{
        struct v3d_context *v3d = v3d_context(pctx);

        v3d_flush(pctx);

        if (fence) {
                struct pipe_screen *screen = pctx->screen;
                struct v3d_fence *f = v3d_fence_create(v3d);
                screen->fence_reference(screen, fence, NULL);
                *fence = (struct pipe_fence_handle *)f;
        }
}

static void
v3d_invalidate_resource(struct pipe_context *pctx, struct pipe_resource *prsc)
{
        struct v3d_context *v3d = v3d_context(pctx);
        struct v3d_resource *rsc = v3d_resource(prsc);

        rsc->initialized_buffers = 0;

        struct hash_entry *entry = _mesa_hash_table_search(v3d->write_jobs,
                                                           prsc);
        if (!entry)
                return;

        struct v3d_job *job = entry->data;
        if (job->key.zsbuf && job->key.zsbuf->texture == prsc)
                job->store &= ~(PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL);
}

static void
v3d_context_destroy(struct pipe_context *pctx)
{
        struct v3d_context *v3d = v3d_context(pctx);

        v3d_flush(pctx);

        if (v3d->blitter)
                util_blitter_destroy(v3d->blitter);

        if (v3d->primconvert)
                util_primconvert_destroy(v3d->primconvert);

        if (v3d->uploader)
                u_upload_destroy(v3d->uploader);

        slab_destroy_child(&v3d->transfer_pool);

        pipe_surface_reference(&v3d->framebuffer.cbufs[0], NULL);
        pipe_surface_reference(&v3d->framebuffer.zsbuf, NULL);

        v3d_program_fini(pctx);

        ralloc_free(v3d);
}

struct pipe_context *
v3d_context_create(struct pipe_screen *pscreen, void *priv, unsigned flags)
{
        struct v3d_screen *screen = v3d_screen(pscreen);
        struct v3d_context *v3d;

        /* Prevent dumping of the shaders built during context setup. */
        uint32_t saved_shaderdb_flag = V3D_DEBUG & V3D_DEBUG_SHADERDB;
        V3D_DEBUG &= ~V3D_DEBUG_SHADERDB;

        v3d = rzalloc(NULL, struct v3d_context);
        if (!v3d)
                return NULL;
        struct pipe_context *pctx = &v3d->base;

        v3d->screen = screen;

        int ret = drmSyncobjCreate(screen->fd, DRM_SYNCOBJ_CREATE_SIGNALED,
                                   &v3d->out_sync);
        if (ret) {
                ralloc_free(v3d);
                return NULL;
        }

        pctx->screen = pscreen;
        pctx->priv = priv;
        pctx->destroy = v3d_context_destroy;
        pctx->flush = v3d_pipe_flush;
        pctx->invalidate_resource = v3d_invalidate_resource;

        if (screen->devinfo.ver >= 41) {
                v3d41_draw_init(pctx);
                v3d41_state_init(pctx);
        } else {
                v3d33_draw_init(pctx);
                v3d33_state_init(pctx);
        }
        v3d_program_init(pctx);
        v3d_query_init(pctx);
        v3d_resource_context_init(pctx);

        v3d_job_init(v3d);

        v3d->fd = screen->fd;

        slab_create_child(&v3d->transfer_pool, &screen->transfer_pool);

        v3d->uploader = u_upload_create_default(&v3d->base);
        v3d->base.stream_uploader = v3d->uploader;
        v3d->base.const_uploader = v3d->uploader;

        v3d->blitter = util_blitter_create(pctx);
        if (!v3d->blitter)
                goto fail;
        v3d->blitter->use_index_buffer = true;

        v3d->primconvert = util_primconvert_create(pctx,
                                                   (1 << PIPE_PRIM_QUADS) - 1);
        if (!v3d->primconvert)
                goto fail;

        V3D_DEBUG |= saved_shaderdb_flag;

        v3d->sample_mask = (1 << VC5_MAX_SAMPLES) - 1;
        v3d->active_queries = true;

        return &v3d->base;

fail:
        pctx->destroy(pctx);
        return NULL;
}
