/*
 * Copyright © 2009 Corbin Simpson <MostAwesomeDude@gmail.com>
 * Copyright © 2009 Joakim Sindholt <opensource@zhasha.com>
 * Copyright © 2011 Marek Olšák <maraeo@gmail.com>
 * Copyright © 2015 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS, AUTHORS
 * AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 */

#include "amdgpu_cs.h"
#include "amdgpu_public.h"

#include "util/u_cpu_detect.h"
#include "util/u_hash_table.h"
#include "util/hash_table.h"
#include "util/xmlconfig.h"
#include <amdgpu_drm.h>
#include <xf86drm.h>
#include <stdio.h>
#include <sys/stat.h>
#include "amd/common/sid.h"
#include "amd/common/gfx9d.h"

#ifndef AMDGPU_INFO_NUM_VRAM_CPU_PAGE_FAULTS
#define AMDGPU_INFO_NUM_VRAM_CPU_PAGE_FAULTS	0x1E
#endif

static struct util_hash_table *dev_tab = NULL;
static simple_mtx_t dev_tab_mutex = _SIMPLE_MTX_INITIALIZER_NP;

DEBUG_GET_ONCE_BOOL_OPTION(all_bos, "RADEON_ALL_BOS", false)

/* Helper function to do the ioctls needed for setup and init. */
static bool do_winsys_init(struct amdgpu_winsys *ws,
                           const struct pipe_screen_config *config,
                           int fd)
{
   if (!ac_query_gpu_info(fd, ws->dev, &ws->info, &ws->amdinfo))
      goto fail;

   ws->addrlib = amdgpu_addr_create(&ws->info, &ws->amdinfo, &ws->info.max_alignment);
   if (!ws->addrlib) {
      fprintf(stderr, "amdgpu: Cannot create addrlib.\n");
      goto fail;
   }

   ws->check_vm = strstr(debug_get_option("R600_DEBUG", ""), "check_vm") != NULL;
   ws->debug_all_bos = debug_get_option_all_bos();
   ws->reserve_vmid = strstr(debug_get_option("R600_DEBUG", ""), "reserve_vmid") != NULL;
   ws->zero_all_vram_allocs = strstr(debug_get_option("R600_DEBUG", ""), "zerovram") != NULL ||
      driQueryOptionb(config->options, "radeonsi_zerovram");

   return true;

fail:
   amdgpu_device_deinitialize(ws->dev);
   ws->dev = NULL;
   return false;
}

static void do_winsys_deinit(struct amdgpu_winsys *ws)
{
   AddrDestroy(ws->addrlib);
   amdgpu_device_deinitialize(ws->dev);
}

static void amdgpu_winsys_destroy(struct radeon_winsys *rws)
{
   struct amdgpu_winsys *ws = (struct amdgpu_winsys*)rws;

   if (ws->reserve_vmid)
      amdgpu_vm_unreserve_vmid(ws->dev, 0);

   if (util_queue_is_initialized(&ws->cs_queue))
      util_queue_destroy(&ws->cs_queue);

   simple_mtx_destroy(&ws->bo_fence_lock);
   pb_slabs_deinit(&ws->bo_slabs);
   pb_cache_deinit(&ws->bo_cache);
   util_hash_table_destroy(ws->bo_export_table);
   simple_mtx_destroy(&ws->global_bo_list_lock);
   simple_mtx_destroy(&ws->bo_export_table_lock);
   do_winsys_deinit(ws);
   FREE(rws);
}

static void amdgpu_winsys_query_info(struct radeon_winsys *rws,
                                     struct radeon_info *info)
{
   *info = ((struct amdgpu_winsys *)rws)->info;
}

static bool amdgpu_cs_request_feature(struct radeon_cmdbuf *rcs,
                                      enum radeon_feature_id fid,
                                      bool enable)
{
   return false;
}

static uint64_t amdgpu_query_value(struct radeon_winsys *rws,
                                   enum radeon_value_id value)
{
   struct amdgpu_winsys *ws = (struct amdgpu_winsys*)rws;
   struct amdgpu_heap_info heap;
   uint64_t retval = 0;

   switch (value) {
   case RADEON_REQUESTED_VRAM_MEMORY:
      return ws->allocated_vram;
   case RADEON_REQUESTED_GTT_MEMORY:
      return ws->allocated_gtt;
   case RADEON_MAPPED_VRAM:
      return ws->mapped_vram;
   case RADEON_MAPPED_GTT:
      return ws->mapped_gtt;
   case RADEON_BUFFER_WAIT_TIME_NS:
      return ws->buffer_wait_time;
   case RADEON_NUM_MAPPED_BUFFERS:
      return ws->num_mapped_buffers;
   case RADEON_TIMESTAMP:
      amdgpu_query_info(ws->dev, AMDGPU_INFO_TIMESTAMP, 8, &retval);
      return retval;
   case RADEON_NUM_GFX_IBS:
      return ws->num_gfx_IBs;
   case RADEON_NUM_SDMA_IBS:
      return ws->num_sdma_IBs;
   case RADEON_GFX_BO_LIST_COUNTER:
      return ws->gfx_bo_list_counter;
   case RADEON_GFX_IB_SIZE_COUNTER:
      return ws->gfx_ib_size_counter;
   case RADEON_NUM_BYTES_MOVED:
      amdgpu_query_info(ws->dev, AMDGPU_INFO_NUM_BYTES_MOVED, 8, &retval);
      return retval;
   case RADEON_NUM_EVICTIONS:
      amdgpu_query_info(ws->dev, AMDGPU_INFO_NUM_EVICTIONS, 8, &retval);
      return retval;
   case RADEON_NUM_VRAM_CPU_PAGE_FAULTS:
      amdgpu_query_info(ws->dev, AMDGPU_INFO_NUM_VRAM_CPU_PAGE_FAULTS, 8, &retval);
      return retval;
   case RADEON_VRAM_USAGE:
      amdgpu_query_heap_info(ws->dev, AMDGPU_GEM_DOMAIN_VRAM, 0, &heap);
      return heap.heap_usage;
   case RADEON_VRAM_VIS_USAGE:
      amdgpu_query_heap_info(ws->dev, AMDGPU_GEM_DOMAIN_VRAM,
                             AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED, &heap);
      return heap.heap_usage;
   case RADEON_GTT_USAGE:
      amdgpu_query_heap_info(ws->dev, AMDGPU_GEM_DOMAIN_GTT, 0, &heap);
      return heap.heap_usage;
   case RADEON_GPU_TEMPERATURE:
      amdgpu_query_sensor_info(ws->dev, AMDGPU_INFO_SENSOR_GPU_TEMP, 4, &retval);
      return retval;
   case RADEON_CURRENT_SCLK:
      amdgpu_query_sensor_info(ws->dev, AMDGPU_INFO_SENSOR_GFX_SCLK, 4, &retval);
      return retval;
   case RADEON_CURRENT_MCLK:
      amdgpu_query_sensor_info(ws->dev, AMDGPU_INFO_SENSOR_GFX_MCLK, 4, &retval);
      return retval;
   case RADEON_GPU_RESET_COUNTER:
      assert(0);
      return 0;
   case RADEON_CS_THREAD_TIME:
      return util_queue_get_thread_time_nano(&ws->cs_queue, 0);
   }
   return 0;
}

static bool amdgpu_read_registers(struct radeon_winsys *rws,
                                  unsigned reg_offset,
                                  unsigned num_registers, uint32_t *out)
{
   struct amdgpu_winsys *ws = (struct amdgpu_winsys*)rws;

   return amdgpu_read_mm_registers(ws->dev, reg_offset / 4, num_registers,
                                   0xffffffff, 0, out) == 0;
}

static unsigned hash_pointer(void *key)
{
   return _mesa_hash_pointer(key);
}

static int compare_pointers(void *key1, void *key2)
{
   return key1 != key2;
}

static bool amdgpu_winsys_unref(struct radeon_winsys *rws)
{
   struct amdgpu_winsys *ws = (struct amdgpu_winsys*)rws;
   bool destroy;

   /* When the reference counter drops to zero, remove the device pointer
    * from the table.
    * This must happen while the mutex is locked, so that
    * amdgpu_winsys_create in another thread doesn't get the winsys
    * from the table when the counter drops to 0. */
   simple_mtx_lock(&dev_tab_mutex);

   destroy = pipe_reference(&ws->reference, NULL);
   if (destroy && dev_tab) {
      util_hash_table_remove(dev_tab, ws->dev);
      if (util_hash_table_count(dev_tab) == 0) {
         util_hash_table_destroy(dev_tab);
         dev_tab = NULL;
      }
   }

   simple_mtx_unlock(&dev_tab_mutex);
   return destroy;
}

static const char* amdgpu_get_chip_name(struct radeon_winsys *ws)
{
   amdgpu_device_handle dev = ((struct amdgpu_winsys *)ws)->dev;
   return amdgpu_get_marketing_name(dev);
}

static void amdgpu_pin_threads_to_L3_cache(struct radeon_winsys *rws,
                                           unsigned cache)
{
   struct amdgpu_winsys *ws = (struct amdgpu_winsys*)rws;

   util_pin_thread_to_L3(ws->cs_queue.threads[0], cache,
                         util_cpu_caps.cores_per_L3);
}

PUBLIC struct radeon_winsys *
amdgpu_winsys_create(int fd, const struct pipe_screen_config *config,
		     radeon_screen_create_t screen_create)
{
   struct amdgpu_winsys *ws;
   drmVersionPtr version = drmGetVersion(fd);
   amdgpu_device_handle dev;
   uint32_t drm_major, drm_minor, r;

   /* The DRM driver version of amdgpu is 3.x.x. */
   if (version->version_major != 3) {
      drmFreeVersion(version);
      return NULL;
   }
   drmFreeVersion(version);

   /* Look up the winsys from the dev table. */
   simple_mtx_lock(&dev_tab_mutex);
   if (!dev_tab)
      dev_tab = util_hash_table_create(hash_pointer, compare_pointers);

   /* Initialize the amdgpu device. This should always return the same pointer
    * for the same fd. */
   r = amdgpu_device_initialize(fd, &drm_major, &drm_minor, &dev);
   if (r) {
      simple_mtx_unlock(&dev_tab_mutex);
      fprintf(stderr, "amdgpu: amdgpu_device_initialize failed.\n");
      return NULL;
   }

   /* Lookup a winsys if we have already created one for this device. */
   ws = util_hash_table_get(dev_tab, dev);
   if (ws) {
      pipe_reference(NULL, &ws->reference);
      simple_mtx_unlock(&dev_tab_mutex);

      /* Release the device handle, because we don't need it anymore.
       * This function is returning an existing winsys instance, which
       * has its own device handle.
       */
      amdgpu_device_deinitialize(dev);
      return &ws->base;
   }

   /* Create a new winsys. */
   ws = CALLOC_STRUCT(amdgpu_winsys);
   if (!ws)
      goto fail;

   ws->dev = dev;
   ws->info.drm_major = drm_major;
   ws->info.drm_minor = drm_minor;

   if (!do_winsys_init(ws, config, fd))
      goto fail_alloc;

   /* Create managers. */
   pb_cache_init(&ws->bo_cache, RADEON_MAX_CACHED_HEAPS,
                 500000, ws->check_vm ? 1.0f : 2.0f, 0,
                 (ws->info.vram_size + ws->info.gart_size) / 8,
                 amdgpu_bo_destroy, amdgpu_bo_can_reclaim);

   if (!pb_slabs_init(&ws->bo_slabs,
                      AMDGPU_SLAB_MIN_SIZE_LOG2, AMDGPU_SLAB_MAX_SIZE_LOG2,
                      RADEON_MAX_SLAB_HEAPS,
                      ws,
                      amdgpu_bo_can_reclaim_slab,
                      amdgpu_bo_slab_alloc,
                      amdgpu_bo_slab_free))
      goto fail_cache;

   ws->info.min_alloc_size = 1 << AMDGPU_SLAB_MIN_SIZE_LOG2;

   /* init reference */
   pipe_reference_init(&ws->reference, 1);

   /* Set functions. */
   ws->base.unref = amdgpu_winsys_unref;
   ws->base.destroy = amdgpu_winsys_destroy;
   ws->base.query_info = amdgpu_winsys_query_info;
   ws->base.cs_request_feature = amdgpu_cs_request_feature;
   ws->base.query_value = amdgpu_query_value;
   ws->base.read_registers = amdgpu_read_registers;
   ws->base.get_chip_name = amdgpu_get_chip_name;
   ws->base.pin_threads_to_L3_cache = amdgpu_pin_threads_to_L3_cache;

   amdgpu_bo_init_functions(ws);
   amdgpu_cs_init_functions(ws);
   amdgpu_surface_init_functions(ws);

   LIST_INITHEAD(&ws->global_bo_list);
   ws->bo_export_table = util_hash_table_create(hash_pointer, compare_pointers);

   (void) simple_mtx_init(&ws->global_bo_list_lock, mtx_plain);
   (void) simple_mtx_init(&ws->bo_fence_lock, mtx_plain);
   (void) simple_mtx_init(&ws->bo_export_table_lock, mtx_plain);

   if (!util_queue_init(&ws->cs_queue, "cs", 8, 1,
                        UTIL_QUEUE_INIT_RESIZE_IF_FULL)) {
      amdgpu_winsys_destroy(&ws->base);
      simple_mtx_unlock(&dev_tab_mutex);
      return NULL;
   }

   /* Create the screen at the end. The winsys must be initialized
    * completely.
    *
    * Alternatively, we could create the screen based on "ws->gen"
    * and link all drivers into one binary blob. */
   ws->base.screen = screen_create(&ws->base, config);
   if (!ws->base.screen) {
      amdgpu_winsys_destroy(&ws->base);
      simple_mtx_unlock(&dev_tab_mutex);
      return NULL;
   }

   util_hash_table_set(dev_tab, dev, ws);

   if (ws->reserve_vmid) {
	   r = amdgpu_vm_reserve_vmid(dev, 0);
	   if (r) {
		fprintf(stderr, "amdgpu: amdgpu_vm_reserve_vmid failed. (%i)\n", r);
		goto fail_cache;
	   }
   }

   /* We must unlock the mutex once the winsys is fully initialized, so that
    * other threads attempting to create the winsys from the same fd will
    * get a fully initialized winsys and not just half-way initialized. */
   simple_mtx_unlock(&dev_tab_mutex);

   return &ws->base;

fail_cache:
   pb_cache_deinit(&ws->bo_cache);
   do_winsys_deinit(ws);
fail_alloc:
   FREE(ws);
fail:
   simple_mtx_unlock(&dev_tab_mutex);
   return NULL;
}
