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

/**
 * @file v3d_simulator.c
 *
 * Implements VC5 simulation on top of a non-VC5 GEM fd.
 *
 * This file's goal is to emulate the VC5 ioctls' behavior in the kernel on
 * top of the simpenrose software simulator.  Generally, VC5 driver BOs have a
 * GEM-side copy of their contents and a simulator-side memory area that the
 * GEM contents get copied into during simulation.  Once simulation is done,
 * the simulator's data is copied back out to the GEM BOs, so that rendering
 * appears on the screen as if actual hardware rendering had been done.
 *
 * One of the limitations of this code is that we shouldn't really need a
 * GEM-side BO for non-window-system BOs.  However, do we need unique BO
 * handles for each of our GEM bos so that this file can look up its state
 * from the handle passed in at submit ioctl time (also, a couple of places
 * outside of this file still call ioctls directly on the fd).
 *
 * Another limitation is that BO import doesn't work unless the underlying
 * window system's BO size matches what VC5 is going to use, which of course
 * doesn't work out in practice.  This means that for now, only DRI3 (VC5
 * makes the winsys BOs) is supported, not DRI2 (window system makes the winys
 * BOs).
 */

#ifdef USE_V3D_SIMULATOR

#include <sys/mman.h>
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/set.h"
#include "util/u_memory.h"
#include "util/u_mm.h"
#include "v3d_simulator_wrapper.h"

#include "v3d_screen.h"
#include "v3d_context.h"

/** Global (across GEM fds) state for the simulator */
static struct v3d_simulator_state {
        mtx_t mutex;

        struct v3d_hw *v3d;
        int ver;

        /* Base virtual address of the heap. */
        void *mem;
        /* Base hardware address of the heap. */
        uint32_t mem_base;
        /* Size of the heap. */
        size_t mem_size;

        struct mem_block *heap;
        struct mem_block *overflow;

        /** Mapping from GEM handle to struct v3d_simulator_bo * */
        struct hash_table *fd_map;

        int refcount;
} sim_state = {
        .mutex = _MTX_INITIALIZER_NP,
};

/** Per-GEM-fd state for the simulator. */
struct v3d_simulator_file {
        int fd;

        /** Mapping from GEM handle to struct v3d_simulator_bo * */
        struct hash_table *bo_map;

        struct mem_block *gmp;
        void *gmp_vaddr;
};

/** Wrapper for drm_v3d_bo tracking the simulator-specific state. */
struct v3d_simulator_bo {
        struct v3d_simulator_file *file;

        /** Area for this BO within sim_state->mem */
        struct mem_block *block;
        uint32_t size;
        void *vaddr;

        void *winsys_map;
        uint32_t winsys_stride;

        int handle;
};

static void *
int_to_key(int key)
{
        return (void *)(uintptr_t)key;
}

static struct v3d_simulator_file *
v3d_get_simulator_file_for_fd(int fd)
{
        struct hash_entry *entry = _mesa_hash_table_search(sim_state.fd_map,
                                                           int_to_key(fd + 1));
        return entry ? entry->data : NULL;
}

/* A marker placed just after each BO, then checked after rendering to make
 * sure it's still there.
 */
#define BO_SENTINEL		0xfedcba98

/* 128kb */
#define GMP_ALIGN2		17

/**
 * Sets the range of GPU virtual address space to have the given GMP
 * permissions (bit 0 = read, bit 1 = write, write-only forbidden).
 */
static void
set_gmp_flags(struct v3d_simulator_file *file,
              uint32_t offset, uint32_t size, uint32_t flag)
{
        assert((offset & ((1 << GMP_ALIGN2) - 1)) == 0);
        int gmp_offset = offset >> GMP_ALIGN2;
        int gmp_count = align(size, 1 << GMP_ALIGN2) >> GMP_ALIGN2;
        uint32_t *gmp = file->gmp_vaddr;

        assert(flag <= 0x3);

        for (int i = gmp_offset; i < gmp_offset + gmp_count; i++) {
                int32_t bitshift = (i % 16) * 2;
                gmp[i / 16] &= ~(0x3 << bitshift);
                gmp[i / 16] |= flag << bitshift;
        }
}

/**
 * Allocates space in simulator memory and returns a tracking struct for it
 * that also contains the drm_gem_cma_object struct.
 */
static struct v3d_simulator_bo *
v3d_create_simulator_bo(int fd, int handle, unsigned size)
{
        struct v3d_simulator_file *file = v3d_get_simulator_file_for_fd(fd);
        struct v3d_simulator_bo *sim_bo = rzalloc(file,
                                                  struct v3d_simulator_bo);
        size = align(size, 4096);

        sim_bo->file = file;
        sim_bo->handle = handle;

        mtx_lock(&sim_state.mutex);
        sim_bo->block = u_mmAllocMem(sim_state.heap, size + 4, GMP_ALIGN2, 0);
        mtx_unlock(&sim_state.mutex);
        assert(sim_bo->block);

        set_gmp_flags(file, sim_bo->block->ofs, size, 0x3);

        sim_bo->size = size;
        sim_bo->vaddr = sim_state.mem + sim_bo->block->ofs - sim_state.mem_base;
        memset(sim_bo->vaddr, 0xd0, size);

        *(uint32_t *)(sim_bo->vaddr + sim_bo->size) = BO_SENTINEL;

        /* A handle of 0 is used for v3d_gem.c internal allocations that
         * don't need to go in the lookup table.
         */
        if (handle != 0) {
                mtx_lock(&sim_state.mutex);
                _mesa_hash_table_insert(file->bo_map, int_to_key(handle),
                                        sim_bo);
                mtx_unlock(&sim_state.mutex);
        }

        return sim_bo;
}

static void
v3d_free_simulator_bo(struct v3d_simulator_bo *sim_bo)
{
        struct v3d_simulator_file *sim_file = sim_bo->file;

        if (sim_bo->winsys_map)
                munmap(sim_bo->winsys_map, sim_bo->size);

        set_gmp_flags(sim_file, sim_bo->block->ofs, sim_bo->size, 0x0);

        mtx_lock(&sim_state.mutex);
        u_mmFreeMem(sim_bo->block);
        if (sim_bo->handle) {
                struct hash_entry *entry =
                        _mesa_hash_table_search(sim_file->bo_map,
                                                int_to_key(sim_bo->handle));
                _mesa_hash_table_remove(sim_file->bo_map, entry);
        }
        mtx_unlock(&sim_state.mutex);
        ralloc_free(sim_bo);
}

static struct v3d_simulator_bo *
v3d_get_simulator_bo(struct v3d_simulator_file *file, int gem_handle)
{
        mtx_lock(&sim_state.mutex);
        struct hash_entry *entry =
                _mesa_hash_table_search(file->bo_map, int_to_key(gem_handle));
        mtx_unlock(&sim_state.mutex);

        return entry ? entry->data : NULL;
}

static int
v3d_simulator_pin_bos(int fd, struct v3d_job *job)
{
        struct v3d_simulator_file *file = v3d_get_simulator_file_for_fd(fd);

        set_foreach(job->bos, entry) {
                struct v3d_bo *bo = (struct v3d_bo *)entry->key;
                struct v3d_simulator_bo *sim_bo =
                        v3d_get_simulator_bo(file, bo->handle);

                v3d_bo_map(bo);
                memcpy(sim_bo->vaddr, bo->map, bo->size);
        }

        return 0;
}

static int
v3d_simulator_unpin_bos(int fd, struct v3d_job *job)
{
        struct v3d_simulator_file *file = v3d_get_simulator_file_for_fd(fd);

        set_foreach(job->bos, entry) {
                struct v3d_bo *bo = (struct v3d_bo *)entry->key;
                struct v3d_simulator_bo *sim_bo =
                        v3d_get_simulator_bo(file, bo->handle);

                if (*(uint32_t *)(sim_bo->vaddr +
                                  sim_bo->size) != BO_SENTINEL) {
                        fprintf(stderr, "Buffer overflow in %s\n", bo->name);
                }

                v3d_bo_map(bo);
                memcpy(bo->map, sim_bo->vaddr, bo->size);
        }

        return 0;
}

#if 0
static void
v3d_dump_to_file(struct v3d_exec_info *exec)
{
        static int dumpno = 0;
        struct drm_v3d_get_hang_state *state;
        struct drm_v3d_get_hang_state_bo *bo_state;
        unsigned int dump_version = 0;

        if (!(v3d_debug & VC5_DEBUG_DUMP))
                return;

        state = calloc(1, sizeof(*state));

        int unref_count = 0;
        list_for_each_entry_safe(struct drm_v3d_bo, bo, &exec->unref_list,
                                 unref_head) {
                unref_count++;
        }

        /* Add one more for the overflow area that isn't wrapped in a BO. */
        state->bo_count = exec->bo_count + unref_count + 1;
        bo_state = calloc(state->bo_count, sizeof(*bo_state));

        char *filename = NULL;
        asprintf(&filename, "v3d-dri-%d.dump", dumpno++);
        FILE *f = fopen(filename, "w+");
        if (!f) {
                fprintf(stderr, "Couldn't open %s: %s", filename,
                        strerror(errno));
                return;
        }

        fwrite(&dump_version, sizeof(dump_version), 1, f);

        state->ct0ca = exec->ct0ca;
        state->ct0ea = exec->ct0ea;
        state->ct1ca = exec->ct1ca;
        state->ct1ea = exec->ct1ea;
        state->start_bin = exec->ct0ca;
        state->start_render = exec->ct1ca;
        fwrite(state, sizeof(*state), 1, f);

        int i;
        for (i = 0; i < exec->bo_count; i++) {
                struct drm_gem_cma_object *cma_bo = exec->bo[i];
                bo_state[i].handle = i; /* Not used by the parser. */
                bo_state[i].paddr = cma_bo->paddr;
                bo_state[i].size = cma_bo->base.size;
        }

        list_for_each_entry_safe(struct drm_v3d_bo, bo, &exec->unref_list,
                                 unref_head) {
                struct drm_gem_cma_object *cma_bo = &bo->base;
                bo_state[i].handle = 0;
                bo_state[i].paddr = cma_bo->paddr;
                bo_state[i].size = cma_bo->base.size;
                i++;
        }

        /* Add the static overflow memory area. */
        bo_state[i].handle = exec->bo_count;
        bo_state[i].paddr = sim_state.overflow->ofs;
        bo_state[i].size = sim_state.overflow->size;
        i++;

        fwrite(bo_state, sizeof(*bo_state), state->bo_count, f);

        for (int i = 0; i < exec->bo_count; i++) {
                struct drm_gem_cma_object *cma_bo = exec->bo[i];
                fwrite(cma_bo->vaddr, cma_bo->base.size, 1, f);
        }

        list_for_each_entry_safe(struct drm_v3d_bo, bo, &exec->unref_list,
                                 unref_head) {
                struct drm_gem_cma_object *cma_bo = &bo->base;
                fwrite(cma_bo->vaddr, cma_bo->base.size, 1, f);
        }

        void *overflow = calloc(1, sim_state.overflow->size);
        fwrite(overflow, 1, sim_state.overflow->size, f);
        free(overflow);

        free(state);
        free(bo_state);
        fclose(f);
}
#endif

int
v3d_simulator_flush(struct v3d_context *v3d,
                    struct drm_v3d_submit_cl *submit, struct v3d_job *job)
{
        struct v3d_screen *screen = v3d->screen;
        int fd = screen->fd;
        struct v3d_simulator_file *file = v3d_get_simulator_file_for_fd(fd);
        struct v3d_surface *csurf = v3d_surface(v3d->framebuffer.cbufs[0]);
        struct v3d_resource *ctex = csurf ? v3d_resource(csurf->base.texture) : NULL;
        struct v3d_simulator_bo *csim_bo = ctex ? v3d_get_simulator_bo(file, ctex->bo->handle) : NULL;
        uint32_t winsys_stride = ctex ? csim_bo->winsys_stride : 0;
        uint32_t sim_stride = ctex ? ctex->slices[0].stride : 0;
        uint32_t row_len = MIN2(sim_stride, winsys_stride);
        int ret;

        if (ctex && csim_bo->winsys_map) {
#if 0
                fprintf(stderr, "%dx%d %d %d %d\n",
                        ctex->base.b.width0, ctex->base.b.height0,
                        winsys_stride,
                        sim_stride,
                        ctex->bo->size);
#endif

                for (int y = 0; y < ctex->base.height0; y++) {
                        memcpy(ctex->bo->map + y * sim_stride,
                               csim_bo->winsys_map + y * winsys_stride,
                               row_len);
                }
        }

        ret = v3d_simulator_pin_bos(fd, job);
        if (ret)
                return ret;

        //v3d_dump_to_file(&exec);

        if (sim_state.ver >= 41)
                v3d41_simulator_flush(sim_state.v3d, submit, file->gmp->ofs);
        else
                v3d33_simulator_flush(sim_state.v3d, submit, file->gmp->ofs);

        ret = v3d_simulator_unpin_bos(fd, job);
        if (ret)
                return ret;

        if (ctex && csim_bo->winsys_map) {
                for (int y = 0; y < ctex->base.height0; y++) {
                        memcpy(csim_bo->winsys_map + y * winsys_stride,
                               ctex->bo->map + y * sim_stride,
                               row_len);
                }
        }

        return 0;
}

/**
 * Map the underlying GEM object from the real hardware GEM handle.
 */
static void *
v3d_simulator_map_winsys_bo(int fd, struct v3d_simulator_bo *sim_bo)
{
        int ret;
        void *map;

        struct drm_mode_map_dumb map_dumb = {
                .handle = sim_bo->handle,
        };
        ret = drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &map_dumb);
        if (ret != 0) {
                fprintf(stderr, "map ioctl failure\n");
                abort();
        }

        map = mmap(NULL, sim_bo->size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, map_dumb.offset);
        if (map == MAP_FAILED) {
                fprintf(stderr,
                        "mmap of bo %d (offset 0x%016llx, size %d) failed\n",
                        sim_bo->handle, (long long)map_dumb.offset,
                        (int)sim_bo->size);
                abort();
        }

        return map;
}

/**
 * Do fixups after a BO has been opened from a handle.
 *
 * This could be done at DRM_IOCTL_GEM_OPEN/DRM_IOCTL_GEM_PRIME_FD_TO_HANDLE
 * time, but we're still using drmPrimeFDToHandle() so we have this helper to
 * be called afterward instead.
 */
void v3d_simulator_open_from_handle(int fd, uint32_t winsys_stride,
                                    int handle, uint32_t size)
{
        struct v3d_simulator_bo *sim_bo =
                v3d_create_simulator_bo(fd, handle, size);

        sim_bo->winsys_stride = winsys_stride;
        sim_bo->winsys_map = v3d_simulator_map_winsys_bo(fd, sim_bo);
}

/**
 * Simulated ioctl(fd, DRM_VC5_CREATE_BO) implementation.
 *
 * Making a VC5 BO is just a matter of making a corresponding BO on the host.
 */
static int
v3d_simulator_create_bo_ioctl(int fd, struct drm_v3d_create_bo *args)
{
        int ret;
        struct drm_mode_create_dumb create = {
                .width = 128,
                .bpp = 8,
                .height = (args->size + 127) / 128,
        };

        ret = drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create);
        assert(create.size >= args->size);

        args->handle = create.handle;

        struct v3d_simulator_bo *sim_bo =
                v3d_create_simulator_bo(fd, create.handle, args->size);

        args->offset = sim_bo->block->ofs;

        return ret;
}

/**
 * Simulated ioctl(fd, DRM_VC5_MMAP_BO) implementation.
 *
 * We just pass this straight through to dumb mmap.
 */
static int
v3d_simulator_mmap_bo_ioctl(int fd, struct drm_v3d_mmap_bo *args)
{
        int ret;
        struct drm_mode_map_dumb map = {
                .handle = args->handle,
        };

        ret = drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &map);
        args->offset = map.offset;

        return ret;
}

static int
v3d_simulator_get_bo_offset_ioctl(int fd, struct drm_v3d_get_bo_offset *args)
{
        struct v3d_simulator_file *file = v3d_get_simulator_file_for_fd(fd);
        struct v3d_simulator_bo *sim_bo = v3d_get_simulator_bo(file,
                                                               args->handle);

        args->offset = sim_bo->block->ofs;

        return 0;
}

static int
v3d_simulator_gem_close_ioctl(int fd, struct drm_gem_close *args)
{
        /* Free the simulator's internal tracking. */
        struct v3d_simulator_file *file = v3d_get_simulator_file_for_fd(fd);
        struct v3d_simulator_bo *sim_bo = v3d_get_simulator_bo(file,
                                                               args->handle);

        v3d_free_simulator_bo(sim_bo);

        /* Pass the call on down. */
        return drmIoctl(fd, DRM_IOCTL_GEM_CLOSE, args);
}

static int
v3d_simulator_get_param_ioctl(int fd, struct drm_v3d_get_param *args)
{
        if (sim_state.ver >= 41)
                return v3d41_simulator_get_param_ioctl(sim_state.v3d, args);
        else
                return v3d33_simulator_get_param_ioctl(sim_state.v3d, args);
}

int
v3d_simulator_ioctl(int fd, unsigned long request, void *args)
{
        switch (request) {
        case DRM_IOCTL_V3D_CREATE_BO:
                return v3d_simulator_create_bo_ioctl(fd, args);
        case DRM_IOCTL_V3D_MMAP_BO:
                return v3d_simulator_mmap_bo_ioctl(fd, args);
        case DRM_IOCTL_V3D_GET_BO_OFFSET:
                return v3d_simulator_get_bo_offset_ioctl(fd, args);

        case DRM_IOCTL_V3D_WAIT_BO:
                /* We do all of the v3d rendering synchronously, so we just
                 * return immediately on the wait ioctls.  This ignores any
                 * native rendering to the host BO, so it does mean we race on
                 * front buffer rendering.
                 */
                return 0;

        case DRM_IOCTL_V3D_GET_PARAM:
                return v3d_simulator_get_param_ioctl(fd, args);

        case DRM_IOCTL_GEM_CLOSE:
                return v3d_simulator_gem_close_ioctl(fd, args);

        case DRM_IOCTL_GEM_OPEN:
        case DRM_IOCTL_GEM_FLINK:
                return drmIoctl(fd, request, args);
        default:
                fprintf(stderr, "Unknown ioctl 0x%08x\n", (int)request);
                abort();
        }
}

static void
v3d_simulator_init_global(const struct v3d_device_info *devinfo)
{
        mtx_lock(&sim_state.mutex);
        if (sim_state.refcount++) {
                mtx_unlock(&sim_state.mutex);
                return;
        }

        sim_state.v3d = v3d_hw_auto_new(NULL);
        v3d_hw_alloc_mem(sim_state.v3d, 1024 * 1024 * 1024);
        sim_state.mem_base =
                v3d_hw_get_mem(sim_state.v3d, &sim_state.mem_size,
                               &sim_state.mem);

        /* Allocate from anywhere from 4096 up.  We don't allocate at 0,
         * because for OQs and some other addresses in the HW, 0 means
         * disabled.
         */
        sim_state.heap = u_mmInit(4096, sim_state.mem_size - 4096);

        /* Make a block of 0xd0 at address 0 to make sure we don't screw up
         * and land there.
         */
        struct mem_block *b = u_mmAllocMem(sim_state.heap, 4096, GMP_ALIGN2, 0);
        memset(sim_state.mem + b->ofs - sim_state.mem_base, 0xd0, 4096);

        sim_state.ver = v3d_hw_get_version(sim_state.v3d);

        mtx_unlock(&sim_state.mutex);

        sim_state.fd_map =
                _mesa_hash_table_create(NULL,
                                        _mesa_hash_pointer,
                                        _mesa_key_pointer_equal);

        if (sim_state.ver >= 41)
                v3d41_simulator_init_regs(sim_state.v3d);
        else
                v3d33_simulator_init_regs(sim_state.v3d);
}

void
v3d_simulator_init(struct v3d_screen *screen)
{
        v3d_simulator_init_global(&screen->devinfo);

        screen->sim_file = rzalloc(screen, struct v3d_simulator_file);
        struct v3d_simulator_file *sim_file = screen->sim_file;

        screen->sim_file->bo_map =
                _mesa_hash_table_create(screen->sim_file,
                                        _mesa_hash_pointer,
                                        _mesa_key_pointer_equal);

        mtx_lock(&sim_state.mutex);
        _mesa_hash_table_insert(sim_state.fd_map, int_to_key(screen->fd + 1),
                                screen->sim_file);
        mtx_unlock(&sim_state.mutex);

        sim_file->gmp = u_mmAllocMem(sim_state.heap, 8096, GMP_ALIGN2, 0);
        sim_file->gmp_vaddr = (sim_state.mem + sim_file->gmp->ofs -
                               sim_state.mem_base);
}

void
v3d_simulator_destroy(struct v3d_screen *screen)
{
        mtx_lock(&sim_state.mutex);
        if (!--sim_state.refcount) {
                _mesa_hash_table_destroy(sim_state.fd_map, NULL);
                u_mmDestroy(sim_state.heap);
                /* No memsetting the struct, because it contains the mutex. */
                sim_state.mem = NULL;
        }
        mtx_unlock(&sim_state.mutex);
}

#endif /* USE_V3D_SIMULATOR */
