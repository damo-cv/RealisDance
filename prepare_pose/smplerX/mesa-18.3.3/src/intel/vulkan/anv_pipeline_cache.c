/*
 * Copyright © 2015 Intel Corporation
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

#include "compiler/blob.h"
#include "util/hash_table.h"
#include "util/debug.h"
#include "util/disk_cache.h"
#include "util/mesa-sha1.h"
#include "anv_private.h"

struct anv_shader_bin *
anv_shader_bin_create(struct anv_device *device,
                      const void *key_data, uint32_t key_size,
                      const void *kernel_data, uint32_t kernel_size,
                      const void *constant_data, uint32_t constant_data_size,
                      const struct brw_stage_prog_data *prog_data_in,
                      uint32_t prog_data_size, const void *prog_data_param_in,
                      const struct anv_pipeline_bind_map *bind_map)
{
   struct anv_shader_bin *shader;
   struct anv_shader_bin_key *key;
   struct brw_stage_prog_data *prog_data;
   uint32_t *prog_data_param;
   struct anv_pipeline_binding *surface_to_descriptor, *sampler_to_descriptor;

   ANV_MULTIALLOC(ma);
   anv_multialloc_add(&ma, &shader, 1);
   anv_multialloc_add_size(&ma, &key, sizeof(*key) + key_size);
   anv_multialloc_add_size(&ma, &prog_data, prog_data_size);
   anv_multialloc_add(&ma, &prog_data_param, prog_data_in->nr_params);
   anv_multialloc_add(&ma, &surface_to_descriptor,
                           bind_map->surface_count);
   anv_multialloc_add(&ma, &sampler_to_descriptor,
                           bind_map->sampler_count);

   if (!anv_multialloc_alloc(&ma, &device->alloc,
                             VK_SYSTEM_ALLOCATION_SCOPE_DEVICE))
      return NULL;

   shader->ref_cnt = 1;

   key->size = key_size;
   memcpy(key->data, key_data, key_size);
   shader->key = key;

   shader->kernel =
      anv_state_pool_alloc(&device->instruction_state_pool, kernel_size, 64);
   memcpy(shader->kernel.map, kernel_data, kernel_size);
   shader->kernel_size = kernel_size;

   if (constant_data_size) {
      shader->constant_data =
         anv_state_pool_alloc(&device->dynamic_state_pool,
                              constant_data_size, 32);
      memcpy(shader->constant_data.map, constant_data, constant_data_size);
   } else {
      shader->constant_data = ANV_STATE_NULL;
   }
   shader->constant_data_size = constant_data_size;

   memcpy(prog_data, prog_data_in, prog_data_size);
   memcpy(prog_data_param, prog_data_param_in,
          prog_data->nr_params * sizeof(*prog_data_param));
   prog_data->param = prog_data_param;
   shader->prog_data = prog_data;
   shader->prog_data_size = prog_data_size;

   shader->bind_map = *bind_map;
   typed_memcpy(surface_to_descriptor, bind_map->surface_to_descriptor,
                bind_map->surface_count);
   shader->bind_map.surface_to_descriptor = surface_to_descriptor;
   typed_memcpy(sampler_to_descriptor, bind_map->sampler_to_descriptor,
                bind_map->sampler_count);
   shader->bind_map.sampler_to_descriptor = sampler_to_descriptor;

   return shader;
}

void
anv_shader_bin_destroy(struct anv_device *device,
                       struct anv_shader_bin *shader)
{
   assert(shader->ref_cnt == 0);
   anv_state_pool_free(&device->instruction_state_pool, shader->kernel);
   anv_state_pool_free(&device->dynamic_state_pool, shader->constant_data);
   vk_free(&device->alloc, shader);
}

static bool
anv_shader_bin_write_to_blob(const struct anv_shader_bin *shader,
                             struct blob *blob)
{
   bool ok;

   ok = blob_write_uint32(blob, shader->key->size);
   ok = blob_write_bytes(blob, shader->key->data, shader->key->size);

   ok = blob_write_uint32(blob, shader->kernel_size);
   ok = blob_write_bytes(blob, shader->kernel.map, shader->kernel_size);

   ok = blob_write_uint32(blob, shader->constant_data_size);
   ok = blob_write_bytes(blob, shader->constant_data.map,
                         shader->constant_data_size);

   ok = blob_write_uint32(blob, shader->prog_data_size);
   ok = blob_write_bytes(blob, shader->prog_data, shader->prog_data_size);
   ok = blob_write_bytes(blob, shader->prog_data->param,
                               shader->prog_data->nr_params *
                               sizeof(*shader->prog_data->param));

   ok = blob_write_uint32(blob, shader->bind_map.surface_count);
   ok = blob_write_uint32(blob, shader->bind_map.sampler_count);
   ok = blob_write_uint32(blob, shader->bind_map.image_count);
   ok = blob_write_bytes(blob, shader->bind_map.surface_to_descriptor,
                               shader->bind_map.surface_count *
                               sizeof(*shader->bind_map.surface_to_descriptor));
   ok = blob_write_bytes(blob, shader->bind_map.sampler_to_descriptor,
                               shader->bind_map.sampler_count *
                               sizeof(*shader->bind_map.sampler_to_descriptor));

   return ok;
}

static struct anv_shader_bin *
anv_shader_bin_create_from_blob(struct anv_device *device,
                                struct blob_reader *blob)
{
   uint32_t key_size = blob_read_uint32(blob);
   const void *key_data = blob_read_bytes(blob, key_size);

   uint32_t kernel_size = blob_read_uint32(blob);
   const void *kernel_data = blob_read_bytes(blob, kernel_size);

   uint32_t constant_data_size = blob_read_uint32(blob);
   const void *constant_data = blob_read_bytes(blob, constant_data_size);

   uint32_t prog_data_size = blob_read_uint32(blob);
   const struct brw_stage_prog_data *prog_data =
      blob_read_bytes(blob, prog_data_size);
   if (blob->overrun)
      return NULL;
   const void *prog_data_param =
      blob_read_bytes(blob, prog_data->nr_params * sizeof(*prog_data->param));

   struct anv_pipeline_bind_map bind_map;
   bind_map.surface_count = blob_read_uint32(blob);
   bind_map.sampler_count = blob_read_uint32(blob);
   bind_map.image_count = blob_read_uint32(blob);
   bind_map.surface_to_descriptor = (void *)
      blob_read_bytes(blob, bind_map.surface_count *
                            sizeof(*bind_map.surface_to_descriptor));
   bind_map.sampler_to_descriptor = (void *)
      blob_read_bytes(blob, bind_map.sampler_count *
                            sizeof(*bind_map.sampler_to_descriptor));

   if (blob->overrun)
      return NULL;

   return anv_shader_bin_create(device,
                                key_data, key_size,
                                kernel_data, kernel_size,
                                constant_data, constant_data_size,
                                prog_data, prog_data_size, prog_data_param,
                                &bind_map);
}

/* Remaining work:
 *
 * - Compact binding table layout so it's tight and not dependent on
 *   descriptor set layout.
 *
 * - Review prog_data struct for size and cacheability: struct
 *   brw_stage_prog_data has binding_table which uses a lot of uint32_t for 8
 *   bit quantities etc; use bit fields for all bools, eg dual_src_blend.
 */

static uint32_t
shader_bin_key_hash_func(const void *void_key)
{
   const struct anv_shader_bin_key *key = void_key;
   return _mesa_hash_data(key->data, key->size);
}

static bool
shader_bin_key_compare_func(const void *void_a, const void *void_b)
{
   const struct anv_shader_bin_key *a = void_a, *b = void_b;
   if (a->size != b->size)
      return false;

   return memcmp(a->data, b->data, a->size) == 0;
}

void
anv_pipeline_cache_init(struct anv_pipeline_cache *cache,
                        struct anv_device *device,
                        bool cache_enabled)
{
   cache->device = device;
   pthread_mutex_init(&cache->mutex, NULL);

   if (cache_enabled) {
      cache->cache = _mesa_hash_table_create(NULL, shader_bin_key_hash_func,
                                             shader_bin_key_compare_func);
   } else {
      cache->cache = NULL;
   }
}

void
anv_pipeline_cache_finish(struct anv_pipeline_cache *cache)
{
   pthread_mutex_destroy(&cache->mutex);

   if (cache->cache) {
      /* This is a bit unfortunate.  In order to keep things from randomly
       * going away, the shader cache has to hold a reference to all shader
       * binaries it contains.  We unref them when we destroy the cache.
       */
      hash_table_foreach(cache->cache, entry)
         anv_shader_bin_unref(cache->device, entry->data);

      _mesa_hash_table_destroy(cache->cache, NULL);
   }
}

static struct anv_shader_bin *
anv_pipeline_cache_search_locked(struct anv_pipeline_cache *cache,
                                 const void *key_data, uint32_t key_size)
{
   uint32_t vla[1 + DIV_ROUND_UP(key_size, sizeof(uint32_t))];
   struct anv_shader_bin_key *key = (void *)vla;
   key->size = key_size;
   memcpy(key->data, key_data, key_size);

   struct hash_entry *entry = _mesa_hash_table_search(cache->cache, key);
   if (entry)
      return entry->data;
   else
      return NULL;
}

struct anv_shader_bin *
anv_pipeline_cache_search(struct anv_pipeline_cache *cache,
                          const void *key_data, uint32_t key_size)
{
   if (!cache->cache)
      return NULL;

   pthread_mutex_lock(&cache->mutex);

   struct anv_shader_bin *shader =
      anv_pipeline_cache_search_locked(cache, key_data, key_size);

   pthread_mutex_unlock(&cache->mutex);

   /* We increment refcount before handing it to the caller */
   if (shader)
      anv_shader_bin_ref(shader);

   return shader;
}

static void
anv_pipeline_cache_add_shader_bin(struct anv_pipeline_cache *cache,
                                  struct anv_shader_bin *bin)
{
   if (!cache->cache)
      return;

   pthread_mutex_lock(&cache->mutex);

   struct hash_entry *entry = _mesa_hash_table_search(cache->cache, bin->key);
   if (entry == NULL) {
      /* Take a reference for the cache */
      anv_shader_bin_ref(bin);
      _mesa_hash_table_insert(cache->cache, bin->key, bin);
   }

   pthread_mutex_unlock(&cache->mutex);
}

static struct anv_shader_bin *
anv_pipeline_cache_add_shader_locked(struct anv_pipeline_cache *cache,
                                     const void *key_data, uint32_t key_size,
                                     const void *kernel_data,
                                     uint32_t kernel_size,
                                     const void *constant_data,
                                     uint32_t constant_data_size,
                                     const struct brw_stage_prog_data *prog_data,
                                     uint32_t prog_data_size,
                                     const void *prog_data_param,
                                     const struct anv_pipeline_bind_map *bind_map)
{
   struct anv_shader_bin *shader =
      anv_pipeline_cache_search_locked(cache, key_data, key_size);
   if (shader)
      return shader;

   struct anv_shader_bin *bin =
      anv_shader_bin_create(cache->device, key_data, key_size,
                            kernel_data, kernel_size,
                            constant_data, constant_data_size,
                            prog_data, prog_data_size, prog_data_param,
                            bind_map);
   if (!bin)
      return NULL;

   _mesa_hash_table_insert(cache->cache, bin->key, bin);

   return bin;
}

struct anv_shader_bin *
anv_pipeline_cache_upload_kernel(struct anv_pipeline_cache *cache,
                                 const void *key_data, uint32_t key_size,
                                 const void *kernel_data, uint32_t kernel_size,
                                 const void *constant_data,
                                 uint32_t constant_data_size,
                                 const struct brw_stage_prog_data *prog_data,
                                 uint32_t prog_data_size,
                                 const struct anv_pipeline_bind_map *bind_map)
{
   if (cache->cache) {
      pthread_mutex_lock(&cache->mutex);

      struct anv_shader_bin *bin =
         anv_pipeline_cache_add_shader_locked(cache, key_data, key_size,
                                              kernel_data, kernel_size,
                                              constant_data, constant_data_size,
                                              prog_data, prog_data_size,
                                              prog_data->param, bind_map);

      pthread_mutex_unlock(&cache->mutex);

      /* We increment refcount before handing it to the caller */
      if (bin)
         anv_shader_bin_ref(bin);

      return bin;
   } else {
      /* In this case, we're not caching it so the caller owns it entirely */
      return anv_shader_bin_create(cache->device, key_data, key_size,
                                   kernel_data, kernel_size,
                                   constant_data, constant_data_size,
                                   prog_data, prog_data_size,
                                   prog_data->param, bind_map);
   }
}

struct cache_header {
   uint32_t header_size;
   uint32_t header_version;
   uint32_t vendor_id;
   uint32_t device_id;
   uint8_t  uuid[VK_UUID_SIZE];
};

static void
anv_pipeline_cache_load(struct anv_pipeline_cache *cache,
                        const void *data, size_t size)
{
   struct anv_device *device = cache->device;
   struct anv_physical_device *pdevice = &device->instance->physicalDevice;

   if (cache->cache == NULL)
      return;

   struct blob_reader blob;
   blob_reader_init(&blob, data, size);

   struct cache_header header;
   blob_copy_bytes(&blob, &header, sizeof(header));
   uint32_t count = blob_read_uint32(&blob);
   if (blob.overrun)
      return;

   if (header.header_size < sizeof(header))
      return;
   if (header.header_version != VK_PIPELINE_CACHE_HEADER_VERSION_ONE)
      return;
   if (header.vendor_id != 0x8086)
      return;
   if (header.device_id != device->chipset_id)
      return;
   if (memcmp(header.uuid, pdevice->pipeline_cache_uuid, VK_UUID_SIZE) != 0)
      return;

   for (uint32_t i = 0; i < count; i++) {
      struct anv_shader_bin *bin =
         anv_shader_bin_create_from_blob(device, &blob);
      if (!bin)
         break;
      _mesa_hash_table_insert(cache->cache, bin->key, bin);
   }
}

VkResult anv_CreatePipelineCache(
    VkDevice                                    _device,
    const VkPipelineCacheCreateInfo*            pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPipelineCache*                            pPipelineCache)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   struct anv_pipeline_cache *cache;

   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO);
   assert(pCreateInfo->flags == 0);

   cache = vk_alloc2(&device->alloc, pAllocator,
                       sizeof(*cache), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (cache == NULL)
      return vk_error(VK_ERROR_OUT_OF_HOST_MEMORY);

   anv_pipeline_cache_init(cache, device,
                           device->instance->pipeline_cache_enabled);

   if (pCreateInfo->initialDataSize > 0)
      anv_pipeline_cache_load(cache,
                              pCreateInfo->pInitialData,
                              pCreateInfo->initialDataSize);

   *pPipelineCache = anv_pipeline_cache_to_handle(cache);

   return VK_SUCCESS;
}

void anv_DestroyPipelineCache(
    VkDevice                                    _device,
    VkPipelineCache                             _cache,
    const VkAllocationCallbacks*                pAllocator)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_pipeline_cache, cache, _cache);

   if (!cache)
      return;

   anv_pipeline_cache_finish(cache);

   vk_free2(&device->alloc, pAllocator, cache);
}

VkResult anv_GetPipelineCacheData(
    VkDevice                                    _device,
    VkPipelineCache                             _cache,
    size_t*                                     pDataSize,
    void*                                       pData)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_pipeline_cache, cache, _cache);
   struct anv_physical_device *pdevice = &device->instance->physicalDevice;

   struct blob blob;
   if (pData) {
      blob_init_fixed(&blob, pData, *pDataSize);
   } else {
      blob_init_fixed(&blob, NULL, SIZE_MAX);
   }

   struct cache_header header = {
      .header_size = sizeof(struct cache_header),
      .header_version = VK_PIPELINE_CACHE_HEADER_VERSION_ONE,
      .vendor_id = 0x8086,
      .device_id = device->chipset_id,
   };
   memcpy(header.uuid, pdevice->pipeline_cache_uuid, VK_UUID_SIZE);
   blob_write_bytes(&blob, &header, sizeof(header));

   uint32_t count = 0;
   intptr_t count_offset = blob_reserve_uint32(&blob);
   if (count_offset < 0) {
      *pDataSize = 0;
      blob_finish(&blob);
      return VK_INCOMPLETE;
   }

   VkResult result = VK_SUCCESS;
   if (cache->cache) {
      hash_table_foreach(cache->cache, entry) {
         struct anv_shader_bin *shader = entry->data;

         size_t save_size = blob.size;
         if (!anv_shader_bin_write_to_blob(shader, &blob)) {
            /* If it fails reset to the previous size and bail */
            blob.size = save_size;
            result = VK_INCOMPLETE;
            break;
         }

         count++;
      }
   }

   blob_overwrite_uint32(&blob, count_offset, count);

   *pDataSize = blob.size;

   blob_finish(&blob);

   return result;
}

VkResult anv_MergePipelineCaches(
    VkDevice                                    _device,
    VkPipelineCache                             destCache,
    uint32_t                                    srcCacheCount,
    const VkPipelineCache*                      pSrcCaches)
{
   ANV_FROM_HANDLE(anv_pipeline_cache, dst, destCache);

   if (!dst->cache)
      return VK_SUCCESS;

   for (uint32_t i = 0; i < srcCacheCount; i++) {
      ANV_FROM_HANDLE(anv_pipeline_cache, src, pSrcCaches[i]);
      if (!src->cache)
         continue;

      hash_table_foreach(src->cache, entry) {
         struct anv_shader_bin *bin = entry->data;
         assert(bin);

         if (_mesa_hash_table_search(dst->cache, bin->key))
            continue;

         anv_shader_bin_ref(bin);
         _mesa_hash_table_insert(dst->cache, bin->key, bin);
      }
   }

   return VK_SUCCESS;
}

struct anv_shader_bin *
anv_device_search_for_kernel(struct anv_device *device,
                             struct anv_pipeline_cache *cache,
                             const void *key_data, uint32_t key_size)
{
   struct anv_shader_bin *bin;

   if (cache) {
      bin = anv_pipeline_cache_search(cache, key_data, key_size);
      if (bin)
         return bin;
   }

#ifdef ENABLE_SHADER_CACHE
   struct disk_cache *disk_cache = device->instance->physicalDevice.disk_cache;
   if (disk_cache && device->instance->pipeline_cache_enabled) {
      cache_key cache_key;
      disk_cache_compute_key(disk_cache, key_data, key_size, cache_key);

      size_t buffer_size;
      uint8_t *buffer = disk_cache_get(disk_cache, cache_key, &buffer_size);
      if (buffer) {
         struct blob_reader blob;
         blob_reader_init(&blob, buffer, buffer_size);
         bin = anv_shader_bin_create_from_blob(device, &blob);
         free(buffer);

         if (bin) {
            if (cache)
               anv_pipeline_cache_add_shader_bin(cache, bin);
            return bin;
         }
      }
   }
#endif

   return NULL;
}

struct anv_shader_bin *
anv_device_upload_kernel(struct anv_device *device,
                         struct anv_pipeline_cache *cache,
                         const void *key_data, uint32_t key_size,
                         const void *kernel_data, uint32_t kernel_size,
                         const void *constant_data,
                         uint32_t constant_data_size,
                         const struct brw_stage_prog_data *prog_data,
                         uint32_t prog_data_size,
                         const struct anv_pipeline_bind_map *bind_map)
{
   struct anv_shader_bin *bin;
   if (cache) {
      bin = anv_pipeline_cache_upload_kernel(cache, key_data, key_size,
                                             kernel_data, kernel_size,
                                             constant_data, constant_data_size,
                                             prog_data, prog_data_size,
                                             bind_map);
   } else {
      bin = anv_shader_bin_create(device, key_data, key_size,
                                  kernel_data, kernel_size,
                                  constant_data, constant_data_size,
                                  prog_data, prog_data_size,
                                  prog_data->param, bind_map);
   }

   if (bin == NULL)
      return NULL;

#ifdef ENABLE_SHADER_CACHE
   struct disk_cache *disk_cache = device->instance->physicalDevice.disk_cache;
   if (disk_cache) {
      struct blob binary;
      blob_init(&binary);
      anv_shader_bin_write_to_blob(bin, &binary);

      if (!binary.out_of_memory) {
         cache_key cache_key;
         disk_cache_compute_key(disk_cache, key_data, key_size, cache_key);

         disk_cache_put(disk_cache, cache_key, binary.data, binary.size, NULL);
      }

      blob_finish(&binary);
   }
#endif

   return bin;
}
