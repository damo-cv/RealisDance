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

/** @file v3d_job.c
 *
 * Functions for submitting VC5 render jobs to the kernel.
 */

#include <xf86drm.h>
#include "v3d_context.h"
/* The OQ/semaphore packets are the same across V3D versions. */
#define V3D_VERSION 33
#include "broadcom/cle/v3dx_pack.h"
#include "broadcom/common/v3d_macros.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/set.h"
#include "broadcom/clif/clif_dump.h"

static void
remove_from_ht(struct hash_table *ht, void *key)
{
        struct hash_entry *entry = _mesa_hash_table_search(ht, key);
        _mesa_hash_table_remove(ht, entry);
}

static void
v3d_job_free(struct v3d_context *v3d, struct v3d_job *job)
{
        set_foreach(job->bos, entry) {
                struct v3d_bo *bo = (struct v3d_bo *)entry->key;
                v3d_bo_unreference(&bo);
        }

        remove_from_ht(v3d->jobs, &job->key);

        if (job->write_prscs) {
                set_foreach(job->write_prscs, entry) {
                        const struct pipe_resource *prsc = entry->key;

                        remove_from_ht(v3d->write_jobs, (void *)prsc);
                }
        }

        for (int i = 0; i < VC5_MAX_DRAW_BUFFERS; i++) {
                if (job->cbufs[i]) {
                        remove_from_ht(v3d->write_jobs, job->cbufs[i]->texture);
                        pipe_surface_reference(&job->cbufs[i], NULL);
                }
        }
        if (job->zsbuf) {
                struct v3d_resource *rsc = v3d_resource(job->zsbuf->texture);
                if (rsc->separate_stencil)
                        remove_from_ht(v3d->write_jobs,
                                       &rsc->separate_stencil->base);

                remove_from_ht(v3d->write_jobs, job->zsbuf->texture);
                pipe_surface_reference(&job->zsbuf, NULL);
        }

        if (v3d->job == job)
                v3d->job = NULL;

        v3d_destroy_cl(&job->bcl);
        v3d_destroy_cl(&job->rcl);
        v3d_destroy_cl(&job->indirect);
        v3d_bo_unreference(&job->tile_alloc);
        v3d_bo_unreference(&job->tile_state);

        ralloc_free(job);
}

static struct v3d_job *
v3d_job_create(struct v3d_context *v3d)
{
        struct v3d_job *job = rzalloc(v3d, struct v3d_job);

        job->v3d = v3d;

        v3d_init_cl(job, &job->bcl);
        v3d_init_cl(job, &job->rcl);
        v3d_init_cl(job, &job->indirect);

        job->draw_min_x = ~0;
        job->draw_min_y = ~0;
        job->draw_max_x = 0;
        job->draw_max_y = 0;

        job->bos = _mesa_set_create(job,
                                    _mesa_hash_pointer,
                                    _mesa_key_pointer_equal);
        return job;
}

void
v3d_job_add_bo(struct v3d_job *job, struct v3d_bo *bo)
{
        if (!bo)
                return;

        if (_mesa_set_search(job->bos, bo))
                return;

        v3d_bo_reference(bo);
        _mesa_set_add(job->bos, bo);
        job->referenced_size += bo->size;

        uint32_t *bo_handles = (void *)(uintptr_t)job->submit.bo_handles;

        if (job->submit.bo_handle_count >= job->bo_handles_size) {
                job->bo_handles_size = MAX2(4, job->bo_handles_size * 2);
                bo_handles = reralloc(job, bo_handles,
                                      uint32_t, job->bo_handles_size);
                job->submit.bo_handles = (uintptr_t)(void *)bo_handles;
        }
        bo_handles[job->submit.bo_handle_count++] = bo->handle;
}

void
v3d_job_add_write_resource(struct v3d_job *job, struct pipe_resource *prsc)
{
        struct v3d_context *v3d = job->v3d;

        if (!job->write_prscs) {
                job->write_prscs = _mesa_set_create(job,
                                                    _mesa_hash_pointer,
                                                    _mesa_key_pointer_equal);
        }

        _mesa_set_add(job->write_prscs, prsc);
        _mesa_hash_table_insert(v3d->write_jobs, prsc, job);
}

void
v3d_flush_jobs_writing_resource(struct v3d_context *v3d,
                                struct pipe_resource *prsc)
{
        struct hash_entry *entry = _mesa_hash_table_search(v3d->write_jobs,
                                                           prsc);
        if (entry) {
                struct v3d_job *job = entry->data;
                v3d_job_submit(v3d, job);
        }
}

void
v3d_flush_jobs_reading_resource(struct v3d_context *v3d,
                                struct pipe_resource *prsc)
{
        struct v3d_resource *rsc = v3d_resource(prsc);

        v3d_flush_jobs_writing_resource(v3d, prsc);

        hash_table_foreach(v3d->jobs, entry) {
                struct v3d_job *job = entry->data;

                if (_mesa_set_search(job->bos, rsc->bo)) {
                        v3d_job_submit(v3d, job);
                        /* Reminder: v3d->jobs is safe to keep iterating even
                         * after deletion of an entry.
                         */
                        continue;
                }
        }
}

static void
v3d_job_set_tile_buffer_size(struct v3d_job *job)
{
        static const uint8_t tile_sizes[] = {
                64, 64,
                64, 32,
                32, 32,
                32, 16,
                16, 16,
        };
        int tile_size_index = 0;
        if (job->msaa)
                tile_size_index += 2;

        if (job->cbufs[3] || job->cbufs[2])
                tile_size_index += 2;
        else if (job->cbufs[1])
                tile_size_index++;

        int max_bpp = RENDER_TARGET_MAXIMUM_32BPP;
        for (int i = 0; i < VC5_MAX_DRAW_BUFFERS; i++) {
                if (job->cbufs[i]) {
                        struct v3d_surface *surf = v3d_surface(job->cbufs[i]);
                        max_bpp = MAX2(max_bpp, surf->internal_bpp);
                }
        }
        job->internal_bpp = max_bpp;
        STATIC_ASSERT(RENDER_TARGET_MAXIMUM_32BPP == 0);
        tile_size_index += max_bpp;

        assert(tile_size_index < ARRAY_SIZE(tile_sizes));
        job->tile_width = tile_sizes[tile_size_index * 2 + 0];
        job->tile_height = tile_sizes[tile_size_index * 2 + 1];
}

/**
 * Returns a v3d_job struture for tracking V3D rendering to a particular FBO.
 *
 * If we've already started rendering to this FBO, then return old same job,
 * otherwise make a new one.  If we're beginning rendering to an FBO, make
 * sure that any previous reads of the FBO (or writes to its color/Z surfaces)
 * have been flushed.
 */
struct v3d_job *
v3d_get_job(struct v3d_context *v3d,
            struct pipe_surface **cbufs, struct pipe_surface *zsbuf)
{
        /* Return the existing job for this FBO if we have one */
        struct v3d_job_key local_key = {
                .cbufs = {
                        cbufs[0],
                        cbufs[1],
                        cbufs[2],
                        cbufs[3],
                },
                .zsbuf = zsbuf,
        };
        struct hash_entry *entry = _mesa_hash_table_search(v3d->jobs,
                                                           &local_key);
        if (entry)
                return entry->data;

        /* Creating a new job.  Make sure that any previous jobs reading or
         * writing these buffers are flushed.
         */
        struct v3d_job *job = v3d_job_create(v3d);

        for (int i = 0; i < VC5_MAX_DRAW_BUFFERS; i++) {
                if (cbufs[i]) {
                        v3d_flush_jobs_reading_resource(v3d, cbufs[i]->texture);
                        pipe_surface_reference(&job->cbufs[i], cbufs[i]);

                        if (cbufs[i]->texture->nr_samples > 1)
                                job->msaa = true;
                }
        }
        if (zsbuf) {
                v3d_flush_jobs_reading_resource(v3d, zsbuf->texture);
                pipe_surface_reference(&job->zsbuf, zsbuf);
                if (zsbuf->texture->nr_samples > 1)
                        job->msaa = true;
        }

        v3d_job_set_tile_buffer_size(job);

        for (int i = 0; i < VC5_MAX_DRAW_BUFFERS; i++) {
                if (cbufs[i])
                        _mesa_hash_table_insert(v3d->write_jobs,
                                                cbufs[i]->texture, job);
        }
        if (zsbuf) {
                _mesa_hash_table_insert(v3d->write_jobs, zsbuf->texture, job);

                struct v3d_resource *rsc = v3d_resource(zsbuf->texture);
                if (rsc->separate_stencil) {
                        v3d_flush_jobs_reading_resource(v3d,
                                                        &rsc->separate_stencil->base);
                        _mesa_hash_table_insert(v3d->write_jobs,
                                                &rsc->separate_stencil->base,
                                                job);
                }
        }

        memcpy(&job->key, &local_key, sizeof(local_key));
        _mesa_hash_table_insert(v3d->jobs, &job->key, job);

        return job;
}

struct v3d_job *
v3d_get_job_for_fbo(struct v3d_context *v3d)
{
        if (v3d->job)
                return v3d->job;

        struct pipe_surface **cbufs = v3d->framebuffer.cbufs;
        struct pipe_surface *zsbuf = v3d->framebuffer.zsbuf;
        struct v3d_job *job = v3d_get_job(v3d, cbufs, zsbuf);

        /* The dirty flags are tracking what's been updated while v3d->job has
         * been bound, so set them all to ~0 when switching between jobs.  We
         * also need to reset all state at the start of rendering.
         */
        v3d->dirty = ~0;

        /* If we're binding to uninitialized buffers, no need to load their
         * contents before drawing.
         */
        for (int i = 0; i < 4; i++) {
                if (cbufs[i]) {
                        struct v3d_resource *rsc = v3d_resource(cbufs[i]->texture);
                        if (!rsc->writes)
                                job->clear |= PIPE_CLEAR_COLOR0 << i;
                }
        }

        if (zsbuf) {
                struct v3d_resource *rsc = v3d_resource(zsbuf->texture);
                if (!rsc->writes)
                        job->clear |= PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL;
        }

        job->draw_tiles_x = DIV_ROUND_UP(v3d->framebuffer.width,
                                         job->tile_width);
        job->draw_tiles_y = DIV_ROUND_UP(v3d->framebuffer.height,
                                         job->tile_height);

        v3d->job = job;

        return job;
}

static void
v3d_clif_dump(struct v3d_context *v3d, struct v3d_job *job)
{
        if (!(V3D_DEBUG & (V3D_DEBUG_CL | V3D_DEBUG_CLIF)))
                return;

        struct clif_dump *clif = clif_dump_init(&v3d->screen->devinfo,
                                                stderr,
                                                V3D_DEBUG & V3D_DEBUG_CL);

        set_foreach(job->bos, entry) {
                struct v3d_bo *bo = (void *)entry->key;
                char *name = ralloc_asprintf(NULL, "%s_0x%x",
                                             bo->name, bo->offset);

                v3d_bo_map(bo);
                clif_dump_add_bo(clif, name, bo->offset, bo->size, bo->map);

                ralloc_free(name);
        }

        clif_dump(clif, &job->submit);

        clif_dump_destroy(clif);
}

/**
 * Submits the job to the kernel and then reinitializes it.
 */
void
v3d_job_submit(struct v3d_context *v3d, struct v3d_job *job)
{
        MAYBE_UNUSED struct v3d_screen *screen = v3d->screen;

        if (!job->needs_flush)
                goto done;

        if (v3d->screen->devinfo.ver >= 41)
                v3d41_emit_rcl(job);
        else
                v3d33_emit_rcl(job);

        if (cl_offset(&job->bcl) > 0) {
                if (screen->devinfo.ver >= 41)
                        v3d41_bcl_epilogue(v3d, job);
                else
                        v3d33_bcl_epilogue(v3d, job);
        }

        job->submit.out_sync = v3d->out_sync;
        job->submit.bcl_end = job->bcl.bo->offset + cl_offset(&job->bcl);
        job->submit.rcl_end = job->rcl.bo->offset + cl_offset(&job->rcl);

        /* On V3D 4.1, the tile alloc/state setup moved to register writes
         * instead of binner packets.
         */
        if (screen->devinfo.ver >= 41) {
                v3d_job_add_bo(job, job->tile_alloc);
                job->submit.qma = job->tile_alloc->offset;
                job->submit.qms = job->tile_alloc->size;

                v3d_job_add_bo(job, job->tile_state);
                job->submit.qts = job->tile_state->offset;
        }

        v3d_clif_dump(v3d, job);

        if (!(V3D_DEBUG & V3D_DEBUG_NORAST)) {
                int ret;

#ifndef USE_V3D_SIMULATOR
                ret = drmIoctl(v3d->fd, DRM_IOCTL_V3D_SUBMIT_CL, &job->submit);
#else
                ret = v3d_simulator_flush(v3d, &job->submit, job);
#endif
                static bool warned = false;
                if (ret && !warned) {
                        fprintf(stderr, "Draw call returned %s.  "
                                        "Expect corruption.\n", strerror(errno));
                        warned = true;
                }
        }

done:
        v3d_job_free(v3d, job);
}

static bool
v3d_job_compare(const void *a, const void *b)
{
        return memcmp(a, b, sizeof(struct v3d_job_key)) == 0;
}

static uint32_t
v3d_job_hash(const void *key)
{
        return _mesa_hash_data(key, sizeof(struct v3d_job_key));
}

void
v3d_job_init(struct v3d_context *v3d)
{
        v3d->jobs = _mesa_hash_table_create(v3d,
                                            v3d_job_hash,
                                            v3d_job_compare);
        v3d->write_jobs = _mesa_hash_table_create(v3d,
                                                  _mesa_hash_pointer,
                                                  _mesa_key_pointer_equal);
}

