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

#ifndef ANV_PRIVATE_H
#define ANV_PRIVATE_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <assert.h>
#include <stdint.h>
#include <i915_drm.h>

#ifdef HAVE_VALGRIND
#include <valgrind.h>
#include <memcheck.h>
#define VG(x) x
#ifndef NDEBUG
#define __gen_validate_value(x) VALGRIND_CHECK_MEM_IS_DEFINED(&(x), sizeof(x))
#endif
#else
#define VG(x)
#endif

#include "common/gen_clflush.h"
#include "common/gen_gem.h"
#include "dev/gen_device_info.h"
#include "blorp/blorp.h"
#include "compiler/brw_compiler.h"
#include "util/macros.h"
#include "util/hash_table.h"
#include "util/list.h"
#include "util/set.h"
#include "util/u_atomic.h"
#include "util/u_vector.h"
#include "util/u_math.h"
#include "util/vma.h"
#include "vk_alloc.h"
#include "vk_debug_report.h"

/* Pre-declarations needed for WSI entrypoints */
struct wl_surface;
struct wl_display;
typedef struct xcb_connection_t xcb_connection_t;
typedef uint32_t xcb_visualid_t;
typedef uint32_t xcb_window_t;

struct anv_buffer;
struct anv_buffer_view;
struct anv_image_view;
struct anv_instance;

struct gen_l3_config;

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_intel.h>
#include <vulkan/vk_icd.h>
#include <vulkan/vk_android_native_buffer.h>

#include "anv_entrypoints.h"
#include "anv_extensions.h"
#include "isl/isl.h"

#include "common/gen_debug.h"
#include "common/intel_log.h"
#include "wsi_common.h"

/* anv Virtual Memory Layout
 * =========================
 *
 * When the anv driver is determining the virtual graphics addresses of memory
 * objects itself using the softpin mechanism, the following memory ranges
 * will be used.
 *
 * Three special considerations to notice:
 *
 * (1) the dynamic state pool is located within the same 4 GiB as the low
 * heap. This is to work around a VF cache issue described in a comment in
 * anv_physical_device_init_heaps.
 *
 * (2) the binding table pool is located at lower addresses than the surface
 * state pool, within a 4 GiB range. This allows surface state base addresses
 * to cover both binding tables (16 bit offsets) and surface states (32 bit
 * offsets).
 *
 * (3) the last 4 GiB of the address space is withheld from the high
 * heap. Various hardware units will read past the end of an object for
 * various reasons. This healthy margin prevents reads from wrapping around
 * 48-bit addresses.
 */
#define LOW_HEAP_MIN_ADDRESS               0x000000001000ULL /* 4 KiB */
#define LOW_HEAP_MAX_ADDRESS               0x0000bfffffffULL
#define DYNAMIC_STATE_POOL_MIN_ADDRESS     0x0000c0000000ULL /* 3 GiB */
#define DYNAMIC_STATE_POOL_MAX_ADDRESS     0x0000ffffffffULL
#define BINDING_TABLE_POOL_MIN_ADDRESS     0x000100000000ULL /* 4 GiB */
#define BINDING_TABLE_POOL_MAX_ADDRESS     0x00013fffffffULL
#define SURFACE_STATE_POOL_MIN_ADDRESS     0x000140000000ULL /* 5 GiB */
#define SURFACE_STATE_POOL_MAX_ADDRESS     0x00017fffffffULL
#define INSTRUCTION_STATE_POOL_MIN_ADDRESS 0x000180000000ULL /* 6 GiB */
#define INSTRUCTION_STATE_POOL_MAX_ADDRESS 0x0001bfffffffULL
#define HIGH_HEAP_MIN_ADDRESS              0x0001c0000000ULL /* 7 GiB */
#define HIGH_HEAP_MAX_ADDRESS              0xfffeffffffffULL

#define LOW_HEAP_SIZE               \
   (LOW_HEAP_MAX_ADDRESS - LOW_HEAP_MIN_ADDRESS + 1)
#define HIGH_HEAP_SIZE              \
   (HIGH_HEAP_MAX_ADDRESS - HIGH_HEAP_MIN_ADDRESS + 1)
#define DYNAMIC_STATE_POOL_SIZE     \
   (DYNAMIC_STATE_POOL_MAX_ADDRESS - DYNAMIC_STATE_POOL_MIN_ADDRESS + 1)
#define BINDING_TABLE_POOL_SIZE     \
   (BINDING_TABLE_POOL_MAX_ADDRESS - BINDING_TABLE_POOL_MIN_ADDRESS + 1)
#define SURFACE_STATE_POOL_SIZE     \
   (SURFACE_STATE_POOL_MAX_ADDRESS - SURFACE_STATE_POOL_MIN_ADDRESS + 1)
#define INSTRUCTION_STATE_POOL_SIZE \
   (INSTRUCTION_STATE_POOL_MAX_ADDRESS - INSTRUCTION_STATE_POOL_MIN_ADDRESS + 1)

/* Allowing different clear colors requires us to perform a depth resolve at
 * the end of certain render passes. This is because while slow clears store
 * the clear color in the HiZ buffer, fast clears (without a resolve) don't.
 * See the PRMs for examples describing when additional resolves would be
 * necessary. To enable fast clears without requiring extra resolves, we set
 * the clear value to a globally-defined one. We could allow different values
 * if the user doesn't expect coherent data during or after a render passes
 * (VK_ATTACHMENT_STORE_OP_DONT_CARE), but such users (aside from the CTS)
 * don't seem to exist yet. In almost all Vulkan applications tested thus far,
 * 1.0f seems to be the only value used. The only application that doesn't set
 * this value does so through the usage of an seemingly uninitialized clear
 * value.
 */
#define ANV_HZ_FC_VAL 1.0f

#define MAX_VBS         28
#define MAX_SETS         8
#define MAX_RTS          8
#define MAX_VIEWPORTS   16
#define MAX_SCISSORS    16
#define MAX_PUSH_CONSTANTS_SIZE 128
#define MAX_DYNAMIC_BUFFERS 16
#define MAX_IMAGES 64
#define MAX_GEN8_IMAGES 8
#define MAX_PUSH_DESCRIPTORS 32 /* Minimum requirement */

/* The kernel relocation API has a limitation of a 32-bit delta value
 * applied to the address before it is written which, in spite of it being
 * unsigned, is treated as signed .  Because of the way that this maps to
 * the Vulkan API, we cannot handle an offset into a buffer that does not
 * fit into a signed 32 bits.  The only mechanism we have for dealing with
 * this at the moment is to limit all VkDeviceMemory objects to a maximum
 * of 2GB each.  The Vulkan spec allows us to do this:
 *
 *    "Some platforms may have a limit on the maximum size of a single
 *    allocation. For example, certain systems may fail to create
 *    allocations with a size greater than or equal to 4GB. Such a limit is
 *    implementation-dependent, and if such a failure occurs then the error
 *    VK_ERROR_OUT_OF_DEVICE_MEMORY should be returned."
 *
 * We don't use vk_error here because it's not an error so much as an
 * indication to the application that the allocation is too large.
 */
#define MAX_MEMORY_ALLOCATION_SIZE (1ull << 31)

#define ANV_SVGS_VB_INDEX    MAX_VBS
#define ANV_DRAWID_VB_INDEX (MAX_VBS + 1)

#define anv_printflike(a, b) __attribute__((__format__(__printf__, a, b)))

static inline uint32_t
align_down_npot_u32(uint32_t v, uint32_t a)
{
   return v - (v % a);
}

static inline uint32_t
align_u32(uint32_t v, uint32_t a)
{
   assert(a != 0 && a == (a & -a));
   return (v + a - 1) & ~(a - 1);
}

static inline uint64_t
align_u64(uint64_t v, uint64_t a)
{
   assert(a != 0 && a == (a & -a));
   return (v + a - 1) & ~(a - 1);
}

static inline int32_t
align_i32(int32_t v, int32_t a)
{
   assert(a != 0 && a == (a & -a));
   return (v + a - 1) & ~(a - 1);
}

/** Alignment must be a power of 2. */
static inline bool
anv_is_aligned(uintmax_t n, uintmax_t a)
{
   assert(a == (a & -a));
   return (n & (a - 1)) == 0;
}

static inline uint32_t
anv_minify(uint32_t n, uint32_t levels)
{
   if (unlikely(n == 0))
      return 0;
   else
      return MAX2(n >> levels, 1);
}

static inline float
anv_clamp_f(float f, float min, float max)
{
   assert(min < max);

   if (f > max)
      return max;
   else if (f < min)
      return min;
   else
      return f;
}

static inline bool
anv_clear_mask(uint32_t *inout_mask, uint32_t clear_mask)
{
   if (*inout_mask & clear_mask) {
      *inout_mask &= ~clear_mask;
      return true;
   } else {
      return false;
   }
}

static inline union isl_color_value
vk_to_isl_color(VkClearColorValue color)
{
   return (union isl_color_value) {
      .u32 = {
         color.uint32[0],
         color.uint32[1],
         color.uint32[2],
         color.uint32[3],
      },
   };
}

#define for_each_bit(b, dword)                          \
   for (uint32_t __dword = (dword);                     \
        (b) = __builtin_ffs(__dword) - 1, __dword;      \
        __dword &= ~(1 << (b)))

#define typed_memcpy(dest, src, count) ({ \
   STATIC_ASSERT(sizeof(*src) == sizeof(*dest)); \
   memcpy((dest), (src), (count) * sizeof(*(src))); \
})

/* Mapping from anv object to VkDebugReportObjectTypeEXT. New types need
 * to be added here in order to utilize mapping in debug/error/perf macros.
 */
#define REPORT_OBJECT_TYPE(o)                                                      \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_instance*),              \
   VK_DEBUG_REPORT_OBJECT_TYPE_INSTANCE_EXT,                                       \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_physical_device*),       \
   VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT,                                \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_device*),                \
   VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT,                                         \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), const struct anv_device*),          \
   VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT,                                         \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_queue*),                 \
   VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT,                                          \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_semaphore*),             \
   VK_DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT,                                      \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_cmd_buffer*),            \
   VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,                                 \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_fence*),                 \
   VK_DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT,                                          \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_device_memory*),         \
   VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT,                                  \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_buffer*),                \
   VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT,                                         \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_image*),                 \
   VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,                                          \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), const struct anv_image*),           \
   VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,                                          \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_event*),                 \
   VK_DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT,                                          \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_query_pool*),            \
   VK_DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT,                                     \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_buffer_view*),           \
   VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT,                                    \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_image_view*),            \
   VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT,                                     \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_shader_module*),         \
   VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT,                                  \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_pipeline_cache*),        \
   VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT,                                 \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_pipeline_layout*),       \
   VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT,                                \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_render_pass*),           \
   VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT,                                    \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_pipeline*),              \
   VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT,                                       \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_descriptor_set_layout*), \
   VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT,                          \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_sampler*),               \
   VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT,                                        \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_descriptor_pool*),       \
   VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT,                                \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_descriptor_set*),        \
   VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT,                                 \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_framebuffer*),           \
   VK_DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT,                                    \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_cmd_pool*),              \
   VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT,                                   \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct anv_surface*),               \
   VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT,                                    \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct wsi_swapchain*),             \
   VK_DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT,                                  \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), struct vk_debug_callback*),         \
   VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT_EXT,                      \
   __builtin_choose_expr (                                                         \
   __builtin_types_compatible_p (__typeof (o), void*),                             \
   VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,                                        \
   /* The void expression results in a compile-time error                          \
      when assigning the result to something.  */                                  \
   (void)0)))))))))))))))))))))))))))))))

/* Whenever we generate an error, pass it through this function. Useful for
 * debugging, where we can break on it. Only call at error site, not when
 * propagating errors. Might be useful to plug in a stack trace here.
 */

VkResult __vk_errorv(struct anv_instance *instance, const void *object,
                     VkDebugReportObjectTypeEXT type, VkResult error,
                     const char *file, int line, const char *format,
                     va_list args);

VkResult __vk_errorf(struct anv_instance *instance, const void *object,
                     VkDebugReportObjectTypeEXT type, VkResult error,
                     const char *file, int line, const char *format, ...);

#ifdef DEBUG
#define vk_error(error) __vk_errorf(NULL, NULL,\
                                    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,\
                                    error, __FILE__, __LINE__, NULL)
#define vk_errorv(instance, obj, error, format, args)\
    __vk_errorv(instance, obj, REPORT_OBJECT_TYPE(obj), error,\
                __FILE__, __LINE__, format, args)
#define vk_errorf(instance, obj, error, format, ...)\
    __vk_errorf(instance, obj, REPORT_OBJECT_TYPE(obj), error,\
                __FILE__, __LINE__, format, ## __VA_ARGS__)
#else
#define vk_error(error) error
#define vk_errorf(instance, obj, error, format, ...) error
#endif

/**
 * Warn on ignored extension structs.
 *
 * The Vulkan spec requires us to ignore unsupported or unknown structs in
 * a pNext chain.  In debug mode, emitting warnings for ignored structs may
 * help us discover structs that we should not have ignored.
 *
 *
 * From the Vulkan 1.0.38 spec:
 *
 *    Any component of the implementation (the loader, any enabled layers,
 *    and drivers) must skip over, without processing (other than reading the
 *    sType and pNext members) any chained structures with sType values not
 *    defined by extensions supported by that component.
 */
#define anv_debug_ignored_stype(sType) \
   intel_logd("%s: ignored VkStructureType %u\n", __func__, (sType))

void __anv_perf_warn(struct anv_instance *instance, const void *object,
                     VkDebugReportObjectTypeEXT type, const char *file,
                     int line, const char *format, ...)
   anv_printflike(6, 7);
void anv_loge(const char *format, ...) anv_printflike(1, 2);
void anv_loge_v(const char *format, va_list va);

/**
 * Print a FINISHME message, including its source location.
 */
#define anv_finishme(format, ...) \
   do { \
      static bool reported = false; \
      if (!reported) { \
         intel_logw("%s:%d: FINISHME: " format, __FILE__, __LINE__, \
                    ##__VA_ARGS__); \
         reported = true; \
      } \
   } while (0)

/**
 * Print a perf warning message.  Set INTEL_DEBUG=perf to see these.
 */
#define anv_perf_warn(instance, obj, format, ...) \
   do { \
      static bool reported = false; \
      if (!reported && unlikely(INTEL_DEBUG & DEBUG_PERF)) { \
         __anv_perf_warn(instance, obj, REPORT_OBJECT_TYPE(obj), __FILE__, __LINE__,\
                         format, ##__VA_ARGS__); \
         reported = true; \
      } \
   } while (0)

/* A non-fatal assert.  Useful for debugging. */
#ifdef DEBUG
#define anv_assert(x) ({ \
   if (unlikely(!(x))) \
      intel_loge("%s:%d ASSERT: %s", __FILE__, __LINE__, #x); \
})
#else
#define anv_assert(x)
#endif

/* A multi-pointer allocator
 *
 * When copying data structures from the user (such as a render pass), it's
 * common to need to allocate data for a bunch of different things.  Instead
 * of doing several allocations and having to handle all of the error checking
 * that entails, it can be easier to do a single allocation.  This struct
 * helps facilitate that.  The intended usage looks like this:
 *
 *    ANV_MULTIALLOC(ma)
 *    anv_multialloc_add(&ma, &main_ptr, 1);
 *    anv_multialloc_add(&ma, &substruct1, substruct1Count);
 *    anv_multialloc_add(&ma, &substruct2, substruct2Count);
 *
 *    if (!anv_multialloc_alloc(&ma, pAllocator, VK_ALLOCATION_SCOPE_FOO))
 *       return vk_error(VK_ERROR_OUT_OF_HOST_MEORY);
 */
struct anv_multialloc {
    size_t size;
    size_t align;

    uint32_t ptr_count;
    void **ptrs[8];
};

#define ANV_MULTIALLOC_INIT \
   ((struct anv_multialloc) { 0, })

#define ANV_MULTIALLOC(_name) \
   struct anv_multialloc _name = ANV_MULTIALLOC_INIT

__attribute__((always_inline))
static inline void
_anv_multialloc_add(struct anv_multialloc *ma,
                    void **ptr, size_t size, size_t align)
{
   size_t offset = align_u64(ma->size, align);
   ma->size = offset + size;
   ma->align = MAX2(ma->align, align);

   /* Store the offset in the pointer. */
   *ptr = (void *)(uintptr_t)offset;

   assert(ma->ptr_count < ARRAY_SIZE(ma->ptrs));
   ma->ptrs[ma->ptr_count++] = ptr;
}

#define anv_multialloc_add_size(_ma, _ptr, _size) \
   _anv_multialloc_add((_ma), (void **)(_ptr), (_size), __alignof__(**(_ptr)))

#define anv_multialloc_add(_ma, _ptr, _count) \
   anv_multialloc_add_size(_ma, _ptr, (_count) * sizeof(**(_ptr)));

__attribute__((always_inline))
static inline void *
anv_multialloc_alloc(struct anv_multialloc *ma,
                     const VkAllocationCallbacks *alloc,
                     VkSystemAllocationScope scope)
{
   void *ptr = vk_alloc(alloc, ma->size, ma->align, scope);
   if (!ptr)
      return NULL;

   /* Fill out each of the pointers with their final value.
    *
    *   for (uint32_t i = 0; i < ma->ptr_count; i++)
    *      *ma->ptrs[i] = ptr + (uintptr_t)*ma->ptrs[i];
    *
    * Unfortunately, even though ma->ptr_count is basically guaranteed to be a
    * constant, GCC is incapable of figuring this out and unrolling the loop
    * so we have to give it a little help.
    */
   STATIC_ASSERT(ARRAY_SIZE(ma->ptrs) == 8);
#define _ANV_MULTIALLOC_UPDATE_POINTER(_i) \
   if ((_i) < ma->ptr_count) \
      *ma->ptrs[_i] = ptr + (uintptr_t)*ma->ptrs[_i]
   _ANV_MULTIALLOC_UPDATE_POINTER(0);
   _ANV_MULTIALLOC_UPDATE_POINTER(1);
   _ANV_MULTIALLOC_UPDATE_POINTER(2);
   _ANV_MULTIALLOC_UPDATE_POINTER(3);
   _ANV_MULTIALLOC_UPDATE_POINTER(4);
   _ANV_MULTIALLOC_UPDATE_POINTER(5);
   _ANV_MULTIALLOC_UPDATE_POINTER(6);
   _ANV_MULTIALLOC_UPDATE_POINTER(7);
#undef _ANV_MULTIALLOC_UPDATE_POINTER

   return ptr;
}

__attribute__((always_inline))
static inline void *
anv_multialloc_alloc2(struct anv_multialloc *ma,
                      const VkAllocationCallbacks *parent_alloc,
                      const VkAllocationCallbacks *alloc,
                      VkSystemAllocationScope scope)
{
   return anv_multialloc_alloc(ma, alloc ? alloc : parent_alloc, scope);
}

/* Extra ANV-defined BO flags which won't be passed to the kernel */
#define ANV_BO_EXTERNAL    (1ull << 31)
#define ANV_BO_FLAG_MASK   (1ull << 31)

struct anv_bo {
   uint32_t gem_handle;

   /* Index into the current validation list.  This is used by the
    * validation list building alrogithm to track which buffers are already
    * in the validation list so that we can ensure uniqueness.
    */
   uint32_t index;

   /* Last known offset.  This value is provided by the kernel when we
    * execbuf and is used as the presumed offset for the next bunch of
    * relocations.
    */
   uint64_t offset;

   uint64_t size;
   void *map;

   /** Flags to pass to the kernel through drm_i915_exec_object2::flags */
   uint32_t flags;
};

static inline void
anv_bo_init(struct anv_bo *bo, uint32_t gem_handle, uint64_t size)
{
   bo->gem_handle = gem_handle;
   bo->index = 0;
   bo->offset = -1;
   bo->size = size;
   bo->map = NULL;
   bo->flags = 0;
}

/* Represents a lock-free linked list of "free" things.  This is used by
 * both the block pool and the state pools.  Unfortunately, in order to
 * solve the ABA problem, we can't use a single uint32_t head.
 */
union anv_free_list {
   struct {
      int32_t offset;

      /* A simple count that is incremented every time the head changes. */
      uint32_t count;
   };
   uint64_t u64;
};

#define ANV_FREE_LIST_EMPTY ((union anv_free_list) { { 1, 0 } })

struct anv_block_state {
   union {
      struct {
         uint32_t next;
         uint32_t end;
      };
      uint64_t u64;
   };
};

struct anv_block_pool {
   struct anv_device *device;

   uint64_t bo_flags;

   struct anv_bo bo;

   /* The address where the start of the pool is pinned. The various bos that
    * are created as the pool grows will have addresses in the range
    * [start_address, start_address + BLOCK_POOL_MEMFD_SIZE).
    */
   uint64_t start_address;

   /* The offset from the start of the bo to the "center" of the block
    * pool.  Pointers to allocated blocks are given by
    * bo.map + center_bo_offset + offsets.
    */
   uint32_t center_bo_offset;

   /* Current memory map of the block pool.  This pointer may or may not
    * point to the actual beginning of the block pool memory.  If
    * anv_block_pool_alloc_back has ever been called, then this pointer
    * will point to the "center" position of the buffer and all offsets
    * (negative or positive) given out by the block pool alloc functions
    * will be valid relative to this pointer.
    *
    * In particular, map == bo.map + center_offset
    */
   void *map;
   int fd;

   /**
    * Array of mmaps and gem handles owned by the block pool, reclaimed when
    * the block pool is destroyed.
    */
   struct u_vector mmap_cleanups;

   struct anv_block_state state;

   struct anv_block_state back_state;
};

/* Block pools are backed by a fixed-size 1GB memfd */
#define BLOCK_POOL_MEMFD_SIZE (1ul << 30)

/* The center of the block pool is also the middle of the memfd.  This may
 * change in the future if we decide differently for some reason.
 */
#define BLOCK_POOL_MEMFD_CENTER (BLOCK_POOL_MEMFD_SIZE / 2)

static inline uint32_t
anv_block_pool_size(struct anv_block_pool *pool)
{
   return pool->state.end + pool->back_state.end;
}

struct anv_state {
   int32_t offset;
   uint32_t alloc_size;
   void *map;
};

#define ANV_STATE_NULL ((struct anv_state) { .alloc_size = 0 })

struct anv_fixed_size_state_pool {
   union anv_free_list free_list;
   struct anv_block_state block;
};

#define ANV_MIN_STATE_SIZE_LOG2 6
#define ANV_MAX_STATE_SIZE_LOG2 20

#define ANV_STATE_BUCKETS (ANV_MAX_STATE_SIZE_LOG2 - ANV_MIN_STATE_SIZE_LOG2 + 1)

struct anv_state_pool {
   struct anv_block_pool block_pool;

   /* The size of blocks which will be allocated from the block pool */
   uint32_t block_size;

   /** Free list for "back" allocations */
   union anv_free_list back_alloc_free_list;

   struct anv_fixed_size_state_pool buckets[ANV_STATE_BUCKETS];
};

struct anv_state_stream_block;

struct anv_state_stream {
   struct anv_state_pool *state_pool;

   /* The size of blocks to allocate from the state pool */
   uint32_t block_size;

   /* Current block we're allocating from */
   struct anv_state block;

   /* Offset into the current block at which to allocate the next state */
   uint32_t next;

   /* List of all blocks allocated from this pool */
   struct anv_state_stream_block *block_list;
};

/* The block_pool functions exported for testing only.  The block pool should
 * only be used via a state pool (see below).
 */
VkResult anv_block_pool_init(struct anv_block_pool *pool,
                             struct anv_device *device,
                             uint64_t start_address,
                             uint32_t initial_size,
                             uint64_t bo_flags);
void anv_block_pool_finish(struct anv_block_pool *pool);
int32_t anv_block_pool_alloc(struct anv_block_pool *pool,
                             uint32_t block_size);
int32_t anv_block_pool_alloc_back(struct anv_block_pool *pool,
                                  uint32_t block_size);

VkResult anv_state_pool_init(struct anv_state_pool *pool,
                             struct anv_device *device,
                             uint64_t start_address,
                             uint32_t block_size,
                             uint64_t bo_flags);
void anv_state_pool_finish(struct anv_state_pool *pool);
struct anv_state anv_state_pool_alloc(struct anv_state_pool *pool,
                                      uint32_t state_size, uint32_t alignment);
struct anv_state anv_state_pool_alloc_back(struct anv_state_pool *pool);
void anv_state_pool_free(struct anv_state_pool *pool, struct anv_state state);
void anv_state_stream_init(struct anv_state_stream *stream,
                           struct anv_state_pool *state_pool,
                           uint32_t block_size);
void anv_state_stream_finish(struct anv_state_stream *stream);
struct anv_state anv_state_stream_alloc(struct anv_state_stream *stream,
                                        uint32_t size, uint32_t alignment);

/**
 * Implements a pool of re-usable BOs.  The interface is identical to that
 * of block_pool except that each block is its own BO.
 */
struct anv_bo_pool {
   struct anv_device *device;

   uint64_t bo_flags;

   void *free_list[16];
};

void anv_bo_pool_init(struct anv_bo_pool *pool, struct anv_device *device,
                      uint64_t bo_flags);
void anv_bo_pool_finish(struct anv_bo_pool *pool);
VkResult anv_bo_pool_alloc(struct anv_bo_pool *pool, struct anv_bo *bo,
                           uint32_t size);
void anv_bo_pool_free(struct anv_bo_pool *pool, const struct anv_bo *bo);

struct anv_scratch_bo {
   bool exists;
   struct anv_bo bo;
};

struct anv_scratch_pool {
   /* Indexed by Per-Thread Scratch Space number (the hardware value) and stage */
   struct anv_scratch_bo bos[16][MESA_SHADER_STAGES];
};

void anv_scratch_pool_init(struct anv_device *device,
                           struct anv_scratch_pool *pool);
void anv_scratch_pool_finish(struct anv_device *device,
                             struct anv_scratch_pool *pool);
struct anv_bo *anv_scratch_pool_alloc(struct anv_device *device,
                                      struct anv_scratch_pool *pool,
                                      gl_shader_stage stage,
                                      unsigned per_thread_scratch);

/** Implements a BO cache that ensures a 1-1 mapping of GEM BOs to anv_bos */
struct anv_bo_cache {
   struct hash_table *bo_map;
   pthread_mutex_t mutex;
};

VkResult anv_bo_cache_init(struct anv_bo_cache *cache);
void anv_bo_cache_finish(struct anv_bo_cache *cache);
VkResult anv_bo_cache_alloc(struct anv_device *device,
                            struct anv_bo_cache *cache,
                            uint64_t size, uint64_t bo_flags,
                            struct anv_bo **bo);
VkResult anv_bo_cache_import(struct anv_device *device,
                             struct anv_bo_cache *cache,
                             int fd, uint64_t bo_flags,
                             struct anv_bo **bo);
VkResult anv_bo_cache_export(struct anv_device *device,
                             struct anv_bo_cache *cache,
                             struct anv_bo *bo_in, int *fd_out);
void anv_bo_cache_release(struct anv_device *device,
                          struct anv_bo_cache *cache,
                          struct anv_bo *bo);

struct anv_memory_type {
   /* Standard bits passed on to the client */
   VkMemoryPropertyFlags   propertyFlags;
   uint32_t                heapIndex;

   /* Driver-internal book-keeping */
   VkBufferUsageFlags      valid_buffer_usage;
};

struct anv_memory_heap {
   /* Standard bits passed on to the client */
   VkDeviceSize      size;
   VkMemoryHeapFlags flags;

   /* Driver-internal book-keeping */
   bool              supports_48bit_addresses;
};

struct anv_physical_device {
    VK_LOADER_DATA                              _loader_data;

    struct anv_instance *                       instance;
    uint32_t                                    chipset_id;
    bool                                        no_hw;
    char                                        path[20];
    const char *                                name;
    struct {
       uint16_t                                 domain;
       uint8_t                                  bus;
       uint8_t                                  device;
       uint8_t                                  function;
    }                                           pci_info;
    struct gen_device_info                      info;
    /** Amount of "GPU memory" we want to advertise
     *
     * Clearly, this value is bogus since Intel is a UMA architecture.  On
     * gen7 platforms, we are limited by GTT size unless we want to implement
     * fine-grained tracking and GTT splitting.  On Broadwell and above we are
     * practically unlimited.  However, we will never report more than 3/4 of
     * the total system ram to try and avoid running out of RAM.
     */
    bool                                        supports_48bit_addresses;
    struct brw_compiler *                       compiler;
    struct isl_device                           isl_dev;
    int                                         cmd_parser_version;
    bool                                        has_exec_async;
    bool                                        has_exec_capture;
    bool                                        has_exec_fence;
    bool                                        has_syncobj;
    bool                                        has_syncobj_wait;
    bool                                        has_context_priority;
    bool                                        use_softpin;
    bool                                        has_context_isolation;

    struct anv_device_extension_table           supported_extensions;

    uint32_t                                    eu_total;
    uint32_t                                    subslice_total;

    struct {
      uint32_t                                  type_count;
      struct anv_memory_type                    types[VK_MAX_MEMORY_TYPES];
      uint32_t                                  heap_count;
      struct anv_memory_heap                    heaps[VK_MAX_MEMORY_HEAPS];
    } memory;

    uint8_t                                     driver_build_sha1[20];
    uint8_t                                     pipeline_cache_uuid[VK_UUID_SIZE];
    uint8_t                                     driver_uuid[VK_UUID_SIZE];
    uint8_t                                     device_uuid[VK_UUID_SIZE];

    struct disk_cache *                         disk_cache;

    struct wsi_device                       wsi_device;
    int                                         local_fd;
    int                                         master_fd;
};

struct anv_app_info {
   const char*        app_name;
   uint32_t           app_version;
   const char*        engine_name;
   uint32_t           engine_version;
   uint32_t           api_version;
};

struct anv_instance {
    VK_LOADER_DATA                              _loader_data;

    VkAllocationCallbacks                       alloc;

    struct anv_app_info                         app_info;

    struct anv_instance_extension_table         enabled_extensions;
    struct anv_instance_dispatch_table          dispatch;
    struct anv_device_dispatch_table            device_dispatch;

    int                                         physicalDeviceCount;
    struct anv_physical_device                  physicalDevice;

    bool                                        pipeline_cache_enabled;

    struct vk_debug_report_instance             debug_report_callbacks;
};

VkResult anv_init_wsi(struct anv_physical_device *physical_device);
void anv_finish_wsi(struct anv_physical_device *physical_device);

uint32_t anv_physical_device_api_version(struct anv_physical_device *dev);
bool anv_physical_device_extension_supported(struct anv_physical_device *dev,
                                             const char *name);

struct anv_queue {
    VK_LOADER_DATA                              _loader_data;

    struct anv_device *                         device;

    VkDeviceQueueCreateFlags                    flags;
};

struct anv_pipeline_cache {
   struct anv_device *                          device;
   pthread_mutex_t                              mutex;

   struct hash_table *                          cache;
};

struct anv_pipeline_bind_map;

void anv_pipeline_cache_init(struct anv_pipeline_cache *cache,
                             struct anv_device *device,
                             bool cache_enabled);
void anv_pipeline_cache_finish(struct anv_pipeline_cache *cache);

struct anv_shader_bin *
anv_pipeline_cache_search(struct anv_pipeline_cache *cache,
                          const void *key, uint32_t key_size);
struct anv_shader_bin *
anv_pipeline_cache_upload_kernel(struct anv_pipeline_cache *cache,
                                 const void *key_data, uint32_t key_size,
                                 const void *kernel_data, uint32_t kernel_size,
                                 const void *constant_data,
                                 uint32_t constant_data_size,
                                 const struct brw_stage_prog_data *prog_data,
                                 uint32_t prog_data_size,
                                 const struct anv_pipeline_bind_map *bind_map);

struct anv_shader_bin *
anv_device_search_for_kernel(struct anv_device *device,
                             struct anv_pipeline_cache *cache,
                             const void *key_data, uint32_t key_size);

struct anv_shader_bin *
anv_device_upload_kernel(struct anv_device *device,
                         struct anv_pipeline_cache *cache,
                         const void *key_data, uint32_t key_size,
                         const void *kernel_data, uint32_t kernel_size,
                         const void *constant_data,
                         uint32_t constant_data_size,
                         const struct brw_stage_prog_data *prog_data,
                         uint32_t prog_data_size,
                         const struct anv_pipeline_bind_map *bind_map);

struct anv_device {
    VK_LOADER_DATA                              _loader_data;

    VkAllocationCallbacks                       alloc;

    struct anv_instance *                       instance;
    uint32_t                                    chipset_id;
    bool                                        no_hw;
    struct gen_device_info                      info;
    struct isl_device                           isl_dev;
    int                                         context_id;
    int                                         fd;
    bool                                        can_chain_batches;
    bool                                        robust_buffer_access;
    struct anv_device_extension_table           enabled_extensions;
    struct anv_device_dispatch_table            dispatch;

    pthread_mutex_t                             vma_mutex;
    struct util_vma_heap                        vma_lo;
    struct util_vma_heap                        vma_hi;
    uint64_t                                    vma_lo_available;
    uint64_t                                    vma_hi_available;

    struct anv_bo_pool                          batch_bo_pool;

    struct anv_bo_cache                         bo_cache;

    struct anv_state_pool                       dynamic_state_pool;
    struct anv_state_pool                       instruction_state_pool;
    struct anv_state_pool                       binding_table_pool;
    struct anv_state_pool                       surface_state_pool;

    struct anv_bo                               workaround_bo;
    struct anv_bo                               trivial_batch_bo;
    struct anv_bo                               hiz_clear_bo;

    struct anv_pipeline_cache                   default_pipeline_cache;
    struct blorp_context                        blorp;

    struct anv_state                            border_colors;

    struct anv_queue                            queue;

    struct anv_scratch_pool                     scratch_pool;

    uint32_t                                    default_mocs;
    uint32_t                                    external_mocs;

    pthread_mutex_t                             mutex;
    pthread_cond_t                              queue_submit;
    bool                                        _lost;
};

static inline struct anv_state_pool *
anv_binding_table_pool(struct anv_device *device)
{
   if (device->instance->physicalDevice.use_softpin)
      return &device->binding_table_pool;
   else
      return &device->surface_state_pool;
}

static inline struct anv_state
anv_binding_table_pool_alloc(struct anv_device *device) {
   if (device->instance->physicalDevice.use_softpin)
      return anv_state_pool_alloc(&device->binding_table_pool,
                                  device->binding_table_pool.block_size, 0);
   else
      return anv_state_pool_alloc_back(&device->surface_state_pool);
}

static inline void
anv_binding_table_pool_free(struct anv_device *device, struct anv_state state) {
   anv_state_pool_free(anv_binding_table_pool(device), state);
}

static inline uint32_t
anv_mocs_for_bo(const struct anv_device *device, const struct anv_bo *bo)
{
   if (bo->flags & ANV_BO_EXTERNAL)
      return device->external_mocs;
   else
      return device->default_mocs;
}

static void inline
anv_state_flush(struct anv_device *device, struct anv_state state)
{
   if (device->info.has_llc)
      return;

   gen_flush_range(state.map, state.alloc_size);
}

void anv_device_init_blorp(struct anv_device *device);
void anv_device_finish_blorp(struct anv_device *device);

VkResult _anv_device_set_lost(struct anv_device *device,
                              const char *file, int line,
                              const char *msg, ...);
#define anv_device_set_lost(dev, ...) \
   _anv_device_set_lost(dev, __FILE__, __LINE__, __VA_ARGS__)

static inline bool
anv_device_is_lost(struct anv_device *device)
{
   return unlikely(device->_lost);
}

VkResult anv_device_execbuf(struct anv_device *device,
                            struct drm_i915_gem_execbuffer2 *execbuf,
                            struct anv_bo **execbuf_bos);
VkResult anv_device_query_status(struct anv_device *device);
VkResult anv_device_bo_busy(struct anv_device *device, struct anv_bo *bo);
VkResult anv_device_wait(struct anv_device *device, struct anv_bo *bo,
                         int64_t timeout);

void* anv_gem_mmap(struct anv_device *device,
                   uint32_t gem_handle, uint64_t offset, uint64_t size, uint32_t flags);
void anv_gem_munmap(void *p, uint64_t size);
uint32_t anv_gem_create(struct anv_device *device, uint64_t size);
void anv_gem_close(struct anv_device *device, uint32_t gem_handle);
uint32_t anv_gem_userptr(struct anv_device *device, void *mem, size_t size);
int anv_gem_busy(struct anv_device *device, uint32_t gem_handle);
int anv_gem_wait(struct anv_device *device, uint32_t gem_handle, int64_t *timeout_ns);
int anv_gem_execbuffer(struct anv_device *device,
                       struct drm_i915_gem_execbuffer2 *execbuf);
int anv_gem_set_tiling(struct anv_device *device, uint32_t gem_handle,
                       uint32_t stride, uint32_t tiling);
int anv_gem_create_context(struct anv_device *device);
bool anv_gem_has_context_priority(int fd);
int anv_gem_destroy_context(struct anv_device *device, int context);
int anv_gem_set_context_param(int fd, int context, uint32_t param,
                              uint64_t value);
int anv_gem_get_context_param(int fd, int context, uint32_t param,
                              uint64_t *value);
int anv_gem_get_param(int fd, uint32_t param);
int anv_gem_get_tiling(struct anv_device *device, uint32_t gem_handle);
bool anv_gem_get_bit6_swizzle(int fd, uint32_t tiling);
int anv_gem_get_aperture(int fd, uint64_t *size);
int anv_gem_gpu_get_reset_stats(struct anv_device *device,
                                uint32_t *active, uint32_t *pending);
int anv_gem_handle_to_fd(struct anv_device *device, uint32_t gem_handle);
int anv_gem_reg_read(struct anv_device *device,
                     uint32_t offset, uint64_t *result);
uint32_t anv_gem_fd_to_handle(struct anv_device *device, int fd);
int anv_gem_set_caching(struct anv_device *device, uint32_t gem_handle, uint32_t caching);
int anv_gem_set_domain(struct anv_device *device, uint32_t gem_handle,
                       uint32_t read_domains, uint32_t write_domain);
int anv_gem_sync_file_merge(struct anv_device *device, int fd1, int fd2);
uint32_t anv_gem_syncobj_create(struct anv_device *device, uint32_t flags);
void anv_gem_syncobj_destroy(struct anv_device *device, uint32_t handle);
int anv_gem_syncobj_handle_to_fd(struct anv_device *device, uint32_t handle);
uint32_t anv_gem_syncobj_fd_to_handle(struct anv_device *device, int fd);
int anv_gem_syncobj_export_sync_file(struct anv_device *device,
                                     uint32_t handle);
int anv_gem_syncobj_import_sync_file(struct anv_device *device,
                                     uint32_t handle, int fd);
void anv_gem_syncobj_reset(struct anv_device *device, uint32_t handle);
bool anv_gem_supports_syncobj_wait(int fd);
int anv_gem_syncobj_wait(struct anv_device *device,
                         uint32_t *handles, uint32_t num_handles,
                         int64_t abs_timeout_ns, bool wait_all);

bool anv_vma_alloc(struct anv_device *device, struct anv_bo *bo);
void anv_vma_free(struct anv_device *device, struct anv_bo *bo);

VkResult anv_bo_init_new(struct anv_bo *bo, struct anv_device *device, uint64_t size);

struct anv_reloc_list {
   uint32_t                                     num_relocs;
   uint32_t                                     array_length;
   struct drm_i915_gem_relocation_entry *       relocs;
   struct anv_bo **                             reloc_bos;
   struct set *                                 deps;
};

VkResult anv_reloc_list_init(struct anv_reloc_list *list,
                             const VkAllocationCallbacks *alloc);
void anv_reloc_list_finish(struct anv_reloc_list *list,
                           const VkAllocationCallbacks *alloc);

VkResult anv_reloc_list_add(struct anv_reloc_list *list,
                            const VkAllocationCallbacks *alloc,
                            uint32_t offset, struct anv_bo *target_bo,
                            uint32_t delta);

struct anv_batch_bo {
   /* Link in the anv_cmd_buffer.owned_batch_bos list */
   struct list_head                             link;

   struct anv_bo                                bo;

   /* Bytes actually consumed in this batch BO */
   uint32_t                                     length;

   struct anv_reloc_list                        relocs;
};

struct anv_batch {
   const VkAllocationCallbacks *                alloc;

   void *                                       start;
   void *                                       end;
   void *                                       next;

   struct anv_reloc_list *                      relocs;

   /* This callback is called (with the associated user data) in the event
    * that the batch runs out of space.
    */
   VkResult (*extend_cb)(struct anv_batch *, void *);
   void *                                       user_data;

   /**
    * Current error status of the command buffer. Used to track inconsistent
    * or incomplete command buffer states that are the consequence of run-time
    * errors such as out of memory scenarios. We want to track this in the
    * batch because the command buffer object is not visible to some parts
    * of the driver.
    */
   VkResult                                     status;
};

void *anv_batch_emit_dwords(struct anv_batch *batch, int num_dwords);
void anv_batch_emit_batch(struct anv_batch *batch, struct anv_batch *other);
uint64_t anv_batch_emit_reloc(struct anv_batch *batch,
                              void *location, struct anv_bo *bo, uint32_t offset);
VkResult anv_device_submit_simple_batch(struct anv_device *device,
                                        struct anv_batch *batch);

static inline VkResult
anv_batch_set_error(struct anv_batch *batch, VkResult error)
{
   assert(error != VK_SUCCESS);
   if (batch->status == VK_SUCCESS)
      batch->status = error;
   return batch->status;
}

static inline bool
anv_batch_has_error(struct anv_batch *batch)
{
   return batch->status != VK_SUCCESS;
}

struct anv_address {
   struct anv_bo *bo;
   uint32_t offset;
};

#define ANV_NULL_ADDRESS ((struct anv_address) { NULL, 0 })

static inline bool
anv_address_is_null(struct anv_address addr)
{
   return addr.bo == NULL && addr.offset == 0;
}

static inline uint64_t
anv_address_physical(struct anv_address addr)
{
   if (addr.bo && (addr.bo->flags & EXEC_OBJECT_PINNED))
      return gen_canonical_address(addr.bo->offset + addr.offset);
   else
      return gen_canonical_address(addr.offset);
}

static inline struct anv_address
anv_address_add(struct anv_address addr, uint64_t offset)
{
   addr.offset += offset;
   return addr;
}

static inline void
write_reloc(const struct anv_device *device, void *p, uint64_t v, bool flush)
{
   unsigned reloc_size = 0;
   if (device->info.gen >= 8) {
      reloc_size = sizeof(uint64_t);
      *(uint64_t *)p = gen_canonical_address(v);
   } else {
      reloc_size = sizeof(uint32_t);
      *(uint32_t *)p = v;
   }

   if (flush && !device->info.has_llc)
      gen_flush_range(p, reloc_size);
}

static inline uint64_t
_anv_combine_address(struct anv_batch *batch, void *location,
                     const struct anv_address address, uint32_t delta)
{
   if (address.bo == NULL) {
      return address.offset + delta;
   } else {
      assert(batch->start <= location && location < batch->end);

      return anv_batch_emit_reloc(batch, location, address.bo, address.offset + delta);
   }
}

#define __gen_address_type struct anv_address
#define __gen_user_data struct anv_batch
#define __gen_combine_address _anv_combine_address

/* Wrapper macros needed to work around preprocessor argument issues.  In
 * particular, arguments don't get pre-evaluated if they are concatenated.
 * This means that, if you pass GENX(3DSTATE_PS) into the emit macro, the
 * GENX macro won't get evaluated if the emit macro contains "cmd ## foo".
 * We can work around this easily enough with these helpers.
 */
#define __anv_cmd_length(cmd) cmd ## _length
#define __anv_cmd_length_bias(cmd) cmd ## _length_bias
#define __anv_cmd_header(cmd) cmd ## _header
#define __anv_cmd_pack(cmd) cmd ## _pack
#define __anv_reg_num(reg) reg ## _num

#define anv_pack_struct(dst, struc, ...) do {                              \
      struct struc __template = {                                          \
         __VA_ARGS__                                                       \
      };                                                                   \
      __anv_cmd_pack(struc)(NULL, dst, &__template);                       \
      VG(VALGRIND_CHECK_MEM_IS_DEFINED(dst, __anv_cmd_length(struc) * 4)); \
   } while (0)

#define anv_batch_emitn(batch, n, cmd, ...) ({             \
      void *__dst = anv_batch_emit_dwords(batch, n);       \
      if (__dst) {                                         \
         struct cmd __template = {                         \
            __anv_cmd_header(cmd),                         \
           .DWordLength = n - __anv_cmd_length_bias(cmd),  \
            __VA_ARGS__                                    \
         };                                                \
         __anv_cmd_pack(cmd)(batch, __dst, &__template);   \
      }                                                    \
      __dst;                                               \
   })

#define anv_batch_emit_merge(batch, dwords0, dwords1)                   \
   do {                                                                 \
      uint32_t *dw;                                                     \
                                                                        \
      STATIC_ASSERT(ARRAY_SIZE(dwords0) == ARRAY_SIZE(dwords1));        \
      dw = anv_batch_emit_dwords((batch), ARRAY_SIZE(dwords0));         \
      if (!dw)                                                          \
         break;                                                         \
      for (uint32_t i = 0; i < ARRAY_SIZE(dwords0); i++)                \
         dw[i] = (dwords0)[i] | (dwords1)[i];                           \
      VG(VALGRIND_CHECK_MEM_IS_DEFINED(dw, ARRAY_SIZE(dwords0) * 4));\
   } while (0)

#define anv_batch_emit(batch, cmd, name)                            \
   for (struct cmd name = { __anv_cmd_header(cmd) },                    \
        *_dst = anv_batch_emit_dwords(batch, __anv_cmd_length(cmd));    \
        __builtin_expect(_dst != NULL, 1);                              \
        ({ __anv_cmd_pack(cmd)(batch, _dst, &name);                     \
           VG(VALGRIND_CHECK_MEM_IS_DEFINED(_dst, __anv_cmd_length(cmd) * 4)); \
           _dst = NULL;                                                 \
         }))

#define GEN7_MOCS (struct GEN7_MEMORY_OBJECT_CONTROL_STATE) {  \
   .GraphicsDataTypeGFDT                        = 0,           \
   .LLCCacheabilityControlLLCCC                 = 0,           \
   .L3CacheabilityControlL3CC                   = 1,           \
}

#define GEN75_MOCS (struct GEN75_MEMORY_OBJECT_CONTROL_STATE) {  \
   .LLCeLLCCacheabilityControlLLCCC             = 0,           \
   .L3CacheabilityControlL3CC                   = 1,           \
}

#define GEN8_MOCS (struct GEN8_MEMORY_OBJECT_CONTROL_STATE) {  \
      .MemoryTypeLLCeLLCCacheabilityControl = WB,              \
      .TargetCache = L3DefertoPATforLLCeLLCselection,          \
      .AgeforQUADLRU = 0                                       \
   }

#define GEN8_EXTERNAL_MOCS (struct GEN8_MEMORY_OBJECT_CONTROL_STATE) {     \
      .MemoryTypeLLCeLLCCacheabilityControl = UCwithFenceifcoherentcycle,  \
      .TargetCache = L3DefertoPATforLLCeLLCselection,                      \
      .AgeforQUADLRU = 0                                                   \
   }

/* Skylake: MOCS is now an index into an array of 62 different caching
 * configurations programmed by the kernel.
 */

#define GEN9_MOCS (struct GEN9_MEMORY_OBJECT_CONTROL_STATE) {  \
      /* TC=LLC/eLLC, LeCC=WB, LRUM=3, L3CC=WB */              \
      .IndextoMOCSTables                           = 2         \
   }

#define GEN9_EXTERNAL_MOCS (struct GEN9_MEMORY_OBJECT_CONTROL_STATE) {  \
      /* TC=LLC/eLLC, LeCC=WB, LRUM=3, L3CC=WB */                       \
      .IndextoMOCSTables                           = 1                  \
   }

/* Cannonlake MOCS defines are duplicates of Skylake MOCS defines. */
#define GEN10_MOCS (struct GEN10_MEMORY_OBJECT_CONTROL_STATE) {  \
      /* TC=LLC/eLLC, LeCC=WB, LRUM=3, L3CC=WB */              \
      .IndextoMOCSTables                           = 2         \
   }

#define GEN10_EXTERNAL_MOCS (struct GEN10_MEMORY_OBJECT_CONTROL_STATE) {   \
      /* TC=LLC/eLLC, LeCC=WB, LRUM=3, L3CC=WB */                          \
      .IndextoMOCSTables                           = 1                     \
   }

/* Ice Lake MOCS defines are duplicates of Skylake MOCS defines. */
#define GEN11_MOCS (struct GEN11_MEMORY_OBJECT_CONTROL_STATE) {  \
      /* TC=LLC/eLLC, LeCC=WB, LRUM=3, L3CC=WB */              \
      .IndextoMOCSTables                           = 2         \
   }

#define GEN11_EXTERNAL_MOCS (struct GEN11_MEMORY_OBJECT_CONTROL_STATE) {   \
      /* TC=LLC/eLLC, LeCC=WB, LRUM=3, L3CC=WB */                          \
      .IndextoMOCSTables                           = 1                     \
   }

struct anv_device_memory {
   struct anv_bo *                              bo;
   struct anv_memory_type *                     type;
   VkDeviceSize                                 map_size;
   void *                                       map;
};

/**
 * Header for Vertex URB Entry (VUE)
 */
struct anv_vue_header {
   uint32_t Reserved;
   uint32_t RTAIndex; /* RenderTargetArrayIndex */
   uint32_t ViewportIndex;
   float PointWidth;
};

struct anv_descriptor_set_binding_layout {
#ifndef NDEBUG
   /* The type of the descriptors in this binding */
   VkDescriptorType type;
#endif

   /* Number of array elements in this binding */
   uint16_t array_size;

   /* Index into the flattend descriptor set */
   uint16_t descriptor_index;

   /* Index into the dynamic state array for a dynamic buffer */
   int16_t dynamic_offset_index;

   /* Index into the descriptor set buffer views */
   int16_t buffer_index;

   struct {
      /* Index into the binding table for the associated surface */
      int16_t surface_index;

      /* Index into the sampler table for the associated sampler */
      int16_t sampler_index;

      /* Index into the image table for the associated image */
      int16_t image_index;
   } stage[MESA_SHADER_STAGES];

   /* Immutable samplers (or NULL if no immutable samplers) */
   struct anv_sampler **immutable_samplers;
};

struct anv_descriptor_set_layout {
   /* Descriptor set layouts can be destroyed at almost any time */
   uint32_t ref_cnt;

   /* Number of bindings in this descriptor set */
   uint16_t binding_count;

   /* Total size of the descriptor set with room for all array entries */
   uint16_t size;

   /* Shader stages affected by this descriptor set */
   uint16_t shader_stages;

   /* Number of buffers in this descriptor set */
   uint16_t buffer_count;

   /* Number of dynamic offsets used by this descriptor set */
   uint16_t dynamic_offset_count;

   /* Bindings in this descriptor set */
   struct anv_descriptor_set_binding_layout binding[0];
};

static inline void
anv_descriptor_set_layout_ref(struct anv_descriptor_set_layout *layout)
{
   assert(layout && layout->ref_cnt >= 1);
   p_atomic_inc(&layout->ref_cnt);
}

static inline void
anv_descriptor_set_layout_unref(struct anv_device *device,
                                struct anv_descriptor_set_layout *layout)
{
   assert(layout && layout->ref_cnt >= 1);
   if (p_atomic_dec_zero(&layout->ref_cnt))
      vk_free(&device->alloc, layout);
}

struct anv_descriptor {
   VkDescriptorType type;

   union {
      struct {
         VkImageLayout layout;
         struct anv_image_view *image_view;
         struct anv_sampler *sampler;
      };

      struct {
         struct anv_buffer *buffer;
         uint64_t offset;
         uint64_t range;
      };

      struct anv_buffer_view *buffer_view;
   };
};

struct anv_descriptor_set {
   struct anv_descriptor_set_layout *layout;
   uint32_t size;
   uint32_t buffer_count;
   struct anv_buffer_view *buffer_views;
   struct anv_descriptor descriptors[0];
};

struct anv_buffer_view {
   enum isl_format format; /**< VkBufferViewCreateInfo::format */
   uint64_t range; /**< VkBufferViewCreateInfo::range */

   struct anv_address address;

   struct anv_state surface_state;
   struct anv_state storage_surface_state;
   struct anv_state writeonly_storage_surface_state;

   struct brw_image_param storage_image_param;
};

struct anv_push_descriptor_set {
   struct anv_descriptor_set set;

   /* Put this field right behind anv_descriptor_set so it fills up the
    * descriptors[0] field. */
   struct anv_descriptor descriptors[MAX_PUSH_DESCRIPTORS];
   struct anv_buffer_view buffer_views[MAX_PUSH_DESCRIPTORS];
};

struct anv_descriptor_pool {
   uint32_t size;
   uint32_t next;
   uint32_t free_list;

   struct anv_state_stream surface_state_stream;
   void *surface_state_free_list;

   char data[0];
};

enum anv_descriptor_template_entry_type {
   ANV_DESCRIPTOR_TEMPLATE_ENTRY_TYPE_IMAGE,
   ANV_DESCRIPTOR_TEMPLATE_ENTRY_TYPE_BUFFER,
   ANV_DESCRIPTOR_TEMPLATE_ENTRY_TYPE_BUFFER_VIEW
};

struct anv_descriptor_template_entry {
   /* The type of descriptor in this entry */
   VkDescriptorType type;

   /* Binding in the descriptor set */
   uint32_t binding;

   /* Offset at which to write into the descriptor set binding */
   uint32_t array_element;

   /* Number of elements to write into the descriptor set binding */
   uint32_t array_count;

   /* Offset into the user provided data */
   size_t offset;

   /* Stride between elements into the user provided data */
   size_t stride;
};

struct anv_descriptor_update_template {
    VkPipelineBindPoint bind_point;

   /* The descriptor set this template corresponds to. This value is only
    * valid if the template was created with the templateType
    * VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET_KHR.
    */
   uint8_t set;

   /* Number of entries in this template */
   uint32_t entry_count;

   /* Entries of the template */
   struct anv_descriptor_template_entry entries[0];
};

size_t
anv_descriptor_set_layout_size(const struct anv_descriptor_set_layout *layout);

void
anv_descriptor_set_write_image_view(struct anv_descriptor_set *set,
                                    const struct gen_device_info * const devinfo,
                                    const VkDescriptorImageInfo * const info,
                                    VkDescriptorType type,
                                    uint32_t binding,
                                    uint32_t element);

void
anv_descriptor_set_write_buffer_view(struct anv_descriptor_set *set,
                                     VkDescriptorType type,
                                     struct anv_buffer_view *buffer_view,
                                     uint32_t binding,
                                     uint32_t element);

void
anv_descriptor_set_write_buffer(struct anv_descriptor_set *set,
                                struct anv_device *device,
                                struct anv_state_stream *alloc_stream,
                                VkDescriptorType type,
                                struct anv_buffer *buffer,
                                uint32_t binding,
                                uint32_t element,
                                VkDeviceSize offset,
                                VkDeviceSize range);

void
anv_descriptor_set_write_template(struct anv_descriptor_set *set,
                                  struct anv_device *device,
                                  struct anv_state_stream *alloc_stream,
                                  const struct anv_descriptor_update_template *template,
                                  const void *data);

VkResult
anv_descriptor_set_create(struct anv_device *device,
                          struct anv_descriptor_pool *pool,
                          struct anv_descriptor_set_layout *layout,
                          struct anv_descriptor_set **out_set);

void
anv_descriptor_set_destroy(struct anv_device *device,
                           struct anv_descriptor_pool *pool,
                           struct anv_descriptor_set *set);

#define ANV_DESCRIPTOR_SET_SHADER_CONSTANTS (UINT8_MAX - 1)
#define ANV_DESCRIPTOR_SET_COLOR_ATTACHMENTS UINT8_MAX

struct anv_pipeline_binding {
   /* The descriptor set this surface corresponds to.  The special value of
    * ANV_DESCRIPTOR_SET_COLOR_ATTACHMENTS indicates that the offset refers
    * to a color attachment and not a regular descriptor.
    */
   uint8_t set;

   /* Binding in the descriptor set */
   uint32_t binding;

   /* Index in the binding */
   uint32_t index;

   /* Plane in the binding index */
   uint8_t plane;

   /* Input attachment index (relative to the subpass) */
   uint8_t input_attachment_index;

   /* For a storage image, whether it is write-only */
   bool write_only;
};

struct anv_pipeline_layout {
   struct {
      struct anv_descriptor_set_layout *layout;
      uint32_t dynamic_offset_start;
   } set[MAX_SETS];

   uint32_t num_sets;

   struct {
      bool has_dynamic_offsets;
   } stage[MESA_SHADER_STAGES];

   unsigned char sha1[20];
};

struct anv_buffer {
   struct anv_device *                          device;
   VkDeviceSize                                 size;

   VkBufferUsageFlags                           usage;

   /* Set when bound */
   struct anv_address                           address;
};

static inline uint64_t
anv_buffer_get_range(struct anv_buffer *buffer, uint64_t offset, uint64_t range)
{
   assert(offset <= buffer->size);
   if (range == VK_WHOLE_SIZE) {
      return buffer->size - offset;
   } else {
      assert(range + offset >= range);
      assert(range + offset <= buffer->size);
      return range;
   }
}

enum anv_cmd_dirty_bits {
   ANV_CMD_DIRTY_DYNAMIC_VIEWPORT                  = 1 << 0, /* VK_DYNAMIC_STATE_VIEWPORT */
   ANV_CMD_DIRTY_DYNAMIC_SCISSOR                   = 1 << 1, /* VK_DYNAMIC_STATE_SCISSOR */
   ANV_CMD_DIRTY_DYNAMIC_LINE_WIDTH                = 1 << 2, /* VK_DYNAMIC_STATE_LINE_WIDTH */
   ANV_CMD_DIRTY_DYNAMIC_DEPTH_BIAS                = 1 << 3, /* VK_DYNAMIC_STATE_DEPTH_BIAS */
   ANV_CMD_DIRTY_DYNAMIC_BLEND_CONSTANTS           = 1 << 4, /* VK_DYNAMIC_STATE_BLEND_CONSTANTS */
   ANV_CMD_DIRTY_DYNAMIC_DEPTH_BOUNDS              = 1 << 5, /* VK_DYNAMIC_STATE_DEPTH_BOUNDS */
   ANV_CMD_DIRTY_DYNAMIC_STENCIL_COMPARE_MASK      = 1 << 6, /* VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK */
   ANV_CMD_DIRTY_DYNAMIC_STENCIL_WRITE_MASK        = 1 << 7, /* VK_DYNAMIC_STATE_STENCIL_WRITE_MASK */
   ANV_CMD_DIRTY_DYNAMIC_STENCIL_REFERENCE         = 1 << 8, /* VK_DYNAMIC_STATE_STENCIL_REFERENCE */
   ANV_CMD_DIRTY_DYNAMIC_ALL                       = (1 << 9) - 1,
   ANV_CMD_DIRTY_PIPELINE                          = 1 << 9,
   ANV_CMD_DIRTY_INDEX_BUFFER                      = 1 << 10,
   ANV_CMD_DIRTY_RENDER_TARGETS                    = 1 << 11,
};
typedef uint32_t anv_cmd_dirty_mask_t;

enum anv_pipe_bits {
   ANV_PIPE_DEPTH_CACHE_FLUSH_BIT            = (1 << 0),
   ANV_PIPE_STALL_AT_SCOREBOARD_BIT          = (1 << 1),
   ANV_PIPE_STATE_CACHE_INVALIDATE_BIT       = (1 << 2),
   ANV_PIPE_CONSTANT_CACHE_INVALIDATE_BIT    = (1 << 3),
   ANV_PIPE_VF_CACHE_INVALIDATE_BIT          = (1 << 4),
   ANV_PIPE_DATA_CACHE_FLUSH_BIT             = (1 << 5),
   ANV_PIPE_TEXTURE_CACHE_INVALIDATE_BIT     = (1 << 10),
   ANV_PIPE_INSTRUCTION_CACHE_INVALIDATE_BIT = (1 << 11),
   ANV_PIPE_RENDER_TARGET_CACHE_FLUSH_BIT    = (1 << 12),
   ANV_PIPE_DEPTH_STALL_BIT                  = (1 << 13),
   ANV_PIPE_CS_STALL_BIT                     = (1 << 20),

   /* This bit does not exist directly in PIPE_CONTROL.  Instead it means that
    * a flush has happened but not a CS stall.  The next time we do any sort
    * of invalidation we need to insert a CS stall at that time.  Otherwise,
    * we would have to CS stall on every flush which could be bad.
    */
   ANV_PIPE_NEEDS_CS_STALL_BIT               = (1 << 21),

   /* This bit does not exist directly in PIPE_CONTROL. It means that render
    * target operations are ongoing. Some operations like copies on the
    * command streamer might need to be aware of this to trigger the
    * appropriate stall before they can proceed with the copy.
    */
   ANV_PIPE_RENDER_TARGET_WRITES              = (1 << 22),
};

#define ANV_PIPE_FLUSH_BITS ( \
   ANV_PIPE_DEPTH_CACHE_FLUSH_BIT | \
   ANV_PIPE_DATA_CACHE_FLUSH_BIT | \
   ANV_PIPE_RENDER_TARGET_CACHE_FLUSH_BIT)

#define ANV_PIPE_STALL_BITS ( \
   ANV_PIPE_STALL_AT_SCOREBOARD_BIT | \
   ANV_PIPE_DEPTH_STALL_BIT | \
   ANV_PIPE_CS_STALL_BIT)

#define ANV_PIPE_INVALIDATE_BITS ( \
   ANV_PIPE_STATE_CACHE_INVALIDATE_BIT | \
   ANV_PIPE_CONSTANT_CACHE_INVALIDATE_BIT | \
   ANV_PIPE_VF_CACHE_INVALIDATE_BIT | \
   ANV_PIPE_DATA_CACHE_FLUSH_BIT | \
   ANV_PIPE_TEXTURE_CACHE_INVALIDATE_BIT | \
   ANV_PIPE_INSTRUCTION_CACHE_INVALIDATE_BIT)

static inline enum anv_pipe_bits
anv_pipe_flush_bits_for_access_flags(VkAccessFlags flags)
{
   enum anv_pipe_bits pipe_bits = 0;

   unsigned b;
   for_each_bit(b, flags) {
      switch ((VkAccessFlagBits)(1 << b)) {
      case VK_ACCESS_SHADER_WRITE_BIT:
         pipe_bits |= ANV_PIPE_DATA_CACHE_FLUSH_BIT;
         break;
      case VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT:
         pipe_bits |= ANV_PIPE_RENDER_TARGET_CACHE_FLUSH_BIT;
         break;
      case VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT:
         pipe_bits |= ANV_PIPE_DEPTH_CACHE_FLUSH_BIT;
         break;
      case VK_ACCESS_TRANSFER_WRITE_BIT:
         pipe_bits |= ANV_PIPE_RENDER_TARGET_CACHE_FLUSH_BIT;
         pipe_bits |= ANV_PIPE_DEPTH_CACHE_FLUSH_BIT;
         break;
      case VK_ACCESS_MEMORY_WRITE_BIT:
         pipe_bits |= ANV_PIPE_FLUSH_BITS;
         break;
      default:
         break; /* Nothing to do */
      }
   }

   return pipe_bits;
}

static inline enum anv_pipe_bits
anv_pipe_invalidate_bits_for_access_flags(VkAccessFlags flags)
{
   enum anv_pipe_bits pipe_bits = 0;

   unsigned b;
   for_each_bit(b, flags) {
      switch ((VkAccessFlagBits)(1 << b)) {
      case VK_ACCESS_INDIRECT_COMMAND_READ_BIT:
      case VK_ACCESS_INDEX_READ_BIT:
      case VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT:
         pipe_bits |= ANV_PIPE_VF_CACHE_INVALIDATE_BIT;
         break;
      case VK_ACCESS_UNIFORM_READ_BIT:
         pipe_bits |= ANV_PIPE_CONSTANT_CACHE_INVALIDATE_BIT;
         pipe_bits |= ANV_PIPE_TEXTURE_CACHE_INVALIDATE_BIT;
         break;
      case VK_ACCESS_SHADER_READ_BIT:
      case VK_ACCESS_INPUT_ATTACHMENT_READ_BIT:
      case VK_ACCESS_TRANSFER_READ_BIT:
         pipe_bits |= ANV_PIPE_TEXTURE_CACHE_INVALIDATE_BIT;
         break;
      case VK_ACCESS_MEMORY_READ_BIT:
         pipe_bits |= ANV_PIPE_INVALIDATE_BITS;
         break;
      case VK_ACCESS_MEMORY_WRITE_BIT:
         pipe_bits |= ANV_PIPE_FLUSH_BITS;
         break;
      default:
         break; /* Nothing to do */
      }
   }

   return pipe_bits;
}

#define VK_IMAGE_ASPECT_ANY_COLOR_BIT_ANV (         \
   VK_IMAGE_ASPECT_COLOR_BIT | \
   VK_IMAGE_ASPECT_PLANE_0_BIT | \
   VK_IMAGE_ASPECT_PLANE_1_BIT | \
   VK_IMAGE_ASPECT_PLANE_2_BIT)
#define VK_IMAGE_ASPECT_PLANES_BITS_ANV ( \
   VK_IMAGE_ASPECT_PLANE_0_BIT | \
   VK_IMAGE_ASPECT_PLANE_1_BIT | \
   VK_IMAGE_ASPECT_PLANE_2_BIT)

struct anv_vertex_binding {
   struct anv_buffer *                          buffer;
   VkDeviceSize                                 offset;
};

#define ANV_PARAM_PUSH(offset)         ((1 << 16) | (uint32_t)(offset))
#define ANV_PARAM_PUSH_OFFSET(param)   ((param) & 0xffff)

struct anv_push_constants {
   /* Current allocated size of this push constants data structure.
    * Because a decent chunk of it may not be used (images on SKL, for
    * instance), we won't actually allocate the entire structure up-front.
    */
   uint32_t size;

   /* Push constant data provided by the client through vkPushConstants */
   uint8_t client_data[MAX_PUSH_CONSTANTS_SIZE];

   /* Used for vkCmdDispatchBase */
   uint32_t base_work_group_id[3];

   /* Image data for image_load_store on pre-SKL */
   struct brw_image_param images[MAX_GEN8_IMAGES];
};

struct anv_dynamic_state {
   struct {
      uint32_t                                  count;
      VkViewport                                viewports[MAX_VIEWPORTS];
   } viewport;

   struct {
      uint32_t                                  count;
      VkRect2D                                  scissors[MAX_SCISSORS];
   } scissor;

   float                                        line_width;

   struct {
      float                                     bias;
      float                                     clamp;
      float                                     slope;
   } depth_bias;

   float                                        blend_constants[4];

   struct {
      float                                     min;
      float                                     max;
   } depth_bounds;

   struct {
      uint32_t                                  front;
      uint32_t                                  back;
   } stencil_compare_mask;

   struct {
      uint32_t                                  front;
      uint32_t                                  back;
   } stencil_write_mask;

   struct {
      uint32_t                                  front;
      uint32_t                                  back;
   } stencil_reference;
};

extern const struct anv_dynamic_state default_dynamic_state;

void anv_dynamic_state_copy(struct anv_dynamic_state *dest,
                            const struct anv_dynamic_state *src,
                            uint32_t copy_mask);

struct anv_surface_state {
   struct anv_state state;
   /** Address of the surface referred to by this state
    *
    * This address is relative to the start of the BO.
    */
   struct anv_address address;
   /* Address of the aux surface, if any
    *
    * This field is ANV_NULL_ADDRESS if and only if no aux surface exists.
    *
    * With the exception of gen8, the bottom 12 bits of this address' offset
    * include extra aux information.
    */
   struct anv_address aux_address;
   /* Address of the clear color, if any
    *
    * This address is relative to the start of the BO.
    */
   struct anv_address clear_address;
};

/**
 * Attachment state when recording a renderpass instance.
 *
 * The clear value is valid only if there exists a pending clear.
 */
struct anv_attachment_state {
   enum isl_aux_usage                           aux_usage;
   enum isl_aux_usage                           input_aux_usage;
   struct anv_surface_state                     color;
   struct anv_surface_state                     input;

   VkImageLayout                                current_layout;
   VkImageAspectFlags                           pending_clear_aspects;
   VkImageAspectFlags                           pending_load_aspects;
   bool                                         fast_clear;
   VkClearValue                                 clear_value;
   bool                                         clear_color_is_zero_one;
   bool                                         clear_color_is_zero;

   /* When multiview is active, attachments with a renderpass clear
    * operation have their respective layers cleared on the first
    * subpass that uses them, and only in that subpass. We keep track
    * of this using a bitfield to indicate which layers of an attachment
    * have not been cleared yet when multiview is active.
    */
   uint32_t                                     pending_clear_views;
};

/** State tracking for particular pipeline bind point
 *
 * This struct is the base struct for anv_cmd_graphics_state and
 * anv_cmd_compute_state.  These are used to track state which is bound to a
 * particular type of pipeline.  Generic state that applies per-stage such as
 * binding table offsets and push constants is tracked generically with a
 * per-stage array in anv_cmd_state.
 */
struct anv_cmd_pipeline_state {
   struct anv_pipeline *pipeline;
   struct anv_pipeline_layout *layout;

   struct anv_descriptor_set *descriptors[MAX_SETS];
   uint32_t dynamic_offsets[MAX_DYNAMIC_BUFFERS];

   struct anv_push_descriptor_set *push_descriptors[MAX_SETS];
};

/** State tracking for graphics pipeline
 *
 * This has anv_cmd_pipeline_state as a base struct to track things which get
 * bound to a graphics pipeline.  Along with general pipeline bind point state
 * which is in the anv_cmd_pipeline_state base struct, it also contains other
 * state which is graphics-specific.
 */
struct anv_cmd_graphics_state {
   struct anv_cmd_pipeline_state base;

   anv_cmd_dirty_mask_t dirty;
   uint32_t vb_dirty;

   struct anv_dynamic_state dynamic;

   struct {
      struct anv_buffer *index_buffer;
      uint32_t index_type; /**< 3DSTATE_INDEX_BUFFER.IndexFormat */
      uint32_t index_offset;
   } gen7;
};

/** State tracking for compute pipeline
 *
 * This has anv_cmd_pipeline_state as a base struct to track things which get
 * bound to a compute pipeline.  Along with general pipeline bind point state
 * which is in the anv_cmd_pipeline_state base struct, it also contains other
 * state which is compute-specific.
 */
struct anv_cmd_compute_state {
   struct anv_cmd_pipeline_state base;

   bool pipeline_dirty;

   struct anv_address num_workgroups;
};

/** State required while building cmd buffer */
struct anv_cmd_state {
   /* PIPELINE_SELECT.PipelineSelection */
   uint32_t                                     current_pipeline;
   const struct gen_l3_config *                 current_l3_config;

   struct anv_cmd_graphics_state                gfx;
   struct anv_cmd_compute_state                 compute;

   enum anv_pipe_bits                           pending_pipe_bits;
   VkShaderStageFlags                           descriptors_dirty;
   VkShaderStageFlags                           push_constants_dirty;

   struct anv_framebuffer *                     framebuffer;
   struct anv_render_pass *                     pass;
   struct anv_subpass *                         subpass;
   VkRect2D                                     render_area;
   uint32_t                                     restart_index;
   struct anv_vertex_binding                    vertex_bindings[MAX_VBS];
   VkShaderStageFlags                           push_constant_stages;
   struct anv_push_constants *                  push_constants[MESA_SHADER_STAGES];
   struct anv_state                             binding_tables[MESA_SHADER_STAGES];
   struct anv_state                             samplers[MESA_SHADER_STAGES];

   /**
    * Whether or not the gen8 PMA fix is enabled.  We ensure that, at the top
    * of any command buffer it is disabled by disabling it in EndCommandBuffer
    * and before invoking the secondary in ExecuteCommands.
    */
   bool                                         pma_fix_enabled;

   /**
    * Whether or not we know for certain that HiZ is enabled for the current
    * subpass.  If, for whatever reason, we are unsure as to whether HiZ is
    * enabled or not, this will be false.
    */
   bool                                         hiz_enabled;

   /**
    * Array length is anv_cmd_state::pass::attachment_count. Array content is
    * valid only when recording a render pass instance.
    */
   struct anv_attachment_state *                attachments;

   /**
    * Surface states for color render targets.  These are stored in a single
    * flat array.  For depth-stencil attachments, the surface state is simply
    * left blank.
    */
   struct anv_state                             render_pass_states;

   /**
    * A null surface state of the right size to match the framebuffer.  This
    * is one of the states in render_pass_states.
    */
   struct anv_state                             null_surface_state;
};

struct anv_cmd_pool {
   VkAllocationCallbacks                        alloc;
   struct list_head                             cmd_buffers;
};

#define ANV_CMD_BUFFER_BATCH_SIZE 8192

enum anv_cmd_buffer_exec_mode {
   ANV_CMD_BUFFER_EXEC_MODE_PRIMARY,
   ANV_CMD_BUFFER_EXEC_MODE_EMIT,
   ANV_CMD_BUFFER_EXEC_MODE_GROW_AND_EMIT,
   ANV_CMD_BUFFER_EXEC_MODE_CHAIN,
   ANV_CMD_BUFFER_EXEC_MODE_COPY_AND_CHAIN,
};

struct anv_cmd_buffer {
   VK_LOADER_DATA                               _loader_data;

   struct anv_device *                          device;

   struct anv_cmd_pool *                        pool;
   struct list_head                             pool_link;

   struct anv_batch                             batch;

   /* Fields required for the actual chain of anv_batch_bo's.
    *
    * These fields are initialized by anv_cmd_buffer_init_batch_bo_chain().
    */
   struct list_head                             batch_bos;
   enum anv_cmd_buffer_exec_mode                exec_mode;

   /* A vector of anv_batch_bo pointers for every batch or surface buffer
    * referenced by this command buffer
    *
    * initialized by anv_cmd_buffer_init_batch_bo_chain()
    */
   struct u_vector                            seen_bbos;

   /* A vector of int32_t's for every block of binding tables.
    *
    * initialized by anv_cmd_buffer_init_batch_bo_chain()
    */
   struct u_vector                              bt_block_states;
   uint32_t                                     bt_next;

   struct anv_reloc_list                        surface_relocs;
   /** Last seen surface state block pool center bo offset */
   uint32_t                                     last_ss_pool_center;

   /* Serial for tracking buffer completion */
   uint32_t                                     serial;

   /* Stream objects for storing temporary data */
   struct anv_state_stream                      surface_state_stream;
   struct anv_state_stream                      dynamic_state_stream;

   VkCommandBufferUsageFlags                    usage_flags;
   VkCommandBufferLevel                         level;

   struct anv_cmd_state                         state;
};

VkResult anv_cmd_buffer_init_batch_bo_chain(struct anv_cmd_buffer *cmd_buffer);
void anv_cmd_buffer_fini_batch_bo_chain(struct anv_cmd_buffer *cmd_buffer);
void anv_cmd_buffer_reset_batch_bo_chain(struct anv_cmd_buffer *cmd_buffer);
void anv_cmd_buffer_end_batch_buffer(struct anv_cmd_buffer *cmd_buffer);
void anv_cmd_buffer_add_secondary(struct anv_cmd_buffer *primary,
                                  struct anv_cmd_buffer *secondary);
void anv_cmd_buffer_prepare_execbuf(struct anv_cmd_buffer *cmd_buffer);
VkResult anv_cmd_buffer_execbuf(struct anv_device *device,
                                struct anv_cmd_buffer *cmd_buffer,
                                const VkSemaphore *in_semaphores,
                                uint32_t num_in_semaphores,
                                const VkSemaphore *out_semaphores,
                                uint32_t num_out_semaphores,
                                VkFence fence);

VkResult anv_cmd_buffer_reset(struct anv_cmd_buffer *cmd_buffer);

VkResult
anv_cmd_buffer_ensure_push_constants_size(struct anv_cmd_buffer *cmd_buffer,
                                          gl_shader_stage stage, uint32_t size);
#define anv_cmd_buffer_ensure_push_constant_field(cmd_buffer, stage, field) \
   anv_cmd_buffer_ensure_push_constants_size(cmd_buffer, stage, \
      (offsetof(struct anv_push_constants, field) + \
       sizeof(cmd_buffer->state.push_constants[0]->field)))

struct anv_state anv_cmd_buffer_emit_dynamic(struct anv_cmd_buffer *cmd_buffer,
                                             const void *data, uint32_t size, uint32_t alignment);
struct anv_state anv_cmd_buffer_merge_dynamic(struct anv_cmd_buffer *cmd_buffer,
                                              uint32_t *a, uint32_t *b,
                                              uint32_t dwords, uint32_t alignment);

struct anv_address
anv_cmd_buffer_surface_base_address(struct anv_cmd_buffer *cmd_buffer);
struct anv_state
anv_cmd_buffer_alloc_binding_table(struct anv_cmd_buffer *cmd_buffer,
                                   uint32_t entries, uint32_t *state_offset);
struct anv_state
anv_cmd_buffer_alloc_surface_state(struct anv_cmd_buffer *cmd_buffer);
struct anv_state
anv_cmd_buffer_alloc_dynamic_state(struct anv_cmd_buffer *cmd_buffer,
                                   uint32_t size, uint32_t alignment);

VkResult
anv_cmd_buffer_new_binding_table_block(struct anv_cmd_buffer *cmd_buffer);

void gen8_cmd_buffer_emit_viewport(struct anv_cmd_buffer *cmd_buffer);
void gen8_cmd_buffer_emit_depth_viewport(struct anv_cmd_buffer *cmd_buffer,
                                         bool depth_clamp_enable);
void gen7_cmd_buffer_emit_scissor(struct anv_cmd_buffer *cmd_buffer);

void anv_cmd_buffer_setup_attachments(struct anv_cmd_buffer *cmd_buffer,
                                      struct anv_render_pass *pass,
                                      struct anv_framebuffer *framebuffer,
                                      const VkClearValue *clear_values);

void anv_cmd_buffer_emit_state_base_address(struct anv_cmd_buffer *cmd_buffer);

struct anv_state
anv_cmd_buffer_push_constants(struct anv_cmd_buffer *cmd_buffer,
                              gl_shader_stage stage);
struct anv_state
anv_cmd_buffer_cs_push_constants(struct anv_cmd_buffer *cmd_buffer);

void anv_cmd_buffer_resolve_subpass(struct anv_cmd_buffer *cmd_buffer);

const struct anv_image_view *
anv_cmd_buffer_get_depth_stencil_view(const struct anv_cmd_buffer *cmd_buffer);

VkResult
anv_cmd_buffer_alloc_blorp_binding_table(struct anv_cmd_buffer *cmd_buffer,
                                         uint32_t num_entries,
                                         uint32_t *state_offset,
                                         struct anv_state *bt_state);

void anv_cmd_buffer_dump(struct anv_cmd_buffer *cmd_buffer);

enum anv_fence_type {
   ANV_FENCE_TYPE_NONE = 0,
   ANV_FENCE_TYPE_BO,
   ANV_FENCE_TYPE_SYNCOBJ,
   ANV_FENCE_TYPE_WSI,
};

enum anv_bo_fence_state {
   /** Indicates that this is a new (or newly reset fence) */
   ANV_BO_FENCE_STATE_RESET,

   /** Indicates that this fence has been submitted to the GPU but is still
    * (as far as we know) in use by the GPU.
    */
   ANV_BO_FENCE_STATE_SUBMITTED,

   ANV_BO_FENCE_STATE_SIGNALED,
};

struct anv_fence_impl {
   enum anv_fence_type type;

   union {
      /** Fence implementation for BO fences
       *
       * These fences use a BO and a set of CPU-tracked state flags.  The BO
       * is added to the object list of the last execbuf call in a QueueSubmit
       * and is marked EXEC_WRITE.  The state flags track when the BO has been
       * submitted to the kernel.  We need to do this because Vulkan lets you
       * wait on a fence that has not yet been submitted and I915_GEM_BUSY
       * will say it's idle in this case.
       */
      struct {
         struct anv_bo bo;
         enum anv_bo_fence_state state;
      } bo;

      /** DRM syncobj handle for syncobj-based fences */
      uint32_t syncobj;

      /** WSI fence */
      struct wsi_fence *fence_wsi;
   };
};

struct anv_fence {
   /* Permanent fence state.  Every fence has some form of permanent state
    * (type != ANV_SEMAPHORE_TYPE_NONE).  This may be a BO to fence on (for
    * cross-process fences) or it could just be a dummy for use internally.
    */
   struct anv_fence_impl permanent;

   /* Temporary fence state.  A fence *may* have temporary state.  That state
    * is added to the fence by an import operation and is reset back to
    * ANV_SEMAPHORE_TYPE_NONE when the fence is reset.  A fence with temporary
    * state cannot be signaled because the fence must already be signaled
    * before the temporary state can be exported from the fence in the other
    * process and imported here.
    */
   struct anv_fence_impl temporary;
};

struct anv_event {
   uint64_t                                     semaphore;
   struct anv_state                             state;
};

enum anv_semaphore_type {
   ANV_SEMAPHORE_TYPE_NONE = 0,
   ANV_SEMAPHORE_TYPE_DUMMY,
   ANV_SEMAPHORE_TYPE_BO,
   ANV_SEMAPHORE_TYPE_SYNC_FILE,
   ANV_SEMAPHORE_TYPE_DRM_SYNCOBJ,
};

struct anv_semaphore_impl {
   enum anv_semaphore_type type;

   union {
      /* A BO representing this semaphore when type == ANV_SEMAPHORE_TYPE_BO.
       * This BO will be added to the object list on any execbuf2 calls for
       * which this semaphore is used as a wait or signal fence.  When used as
       * a signal fence, the EXEC_OBJECT_WRITE flag will be set.
       */
      struct anv_bo *bo;

      /* The sync file descriptor when type == ANV_SEMAPHORE_TYPE_SYNC_FILE.
       * If the semaphore is in the unsignaled state due to either just being
       * created or because it has been used for a wait, fd will be -1.
       */
      int fd;

      /* Sync object handle when type == ANV_SEMAPHORE_TYPE_DRM_SYNCOBJ.
       * Unlike GEM BOs, DRM sync objects aren't deduplicated by the kernel on
       * import so we don't need to bother with a userspace cache.
       */
      uint32_t syncobj;
   };
};

struct anv_semaphore {
   /* Permanent semaphore state.  Every semaphore has some form of permanent
    * state (type != ANV_SEMAPHORE_TYPE_NONE).  This may be a BO to fence on
    * (for cross-process semaphores0 or it could just be a dummy for use
    * internally.
    */
   struct anv_semaphore_impl permanent;

   /* Temporary semaphore state.  A semaphore *may* have temporary state.
    * That state is added to the semaphore by an import operation and is reset
    * back to ANV_SEMAPHORE_TYPE_NONE when the semaphore is waited on.  A
    * semaphore with temporary state cannot be signaled because the semaphore
    * must already be signaled before the temporary state can be exported from
    * the semaphore in the other process and imported here.
    */
   struct anv_semaphore_impl temporary;
};

void anv_semaphore_reset_temporary(struct anv_device *device,
                                   struct anv_semaphore *semaphore);

struct anv_shader_module {
   unsigned char                                sha1[20];
   uint32_t                                     size;
   char                                         data[0];
};

static inline gl_shader_stage
vk_to_mesa_shader_stage(VkShaderStageFlagBits vk_stage)
{
   assert(__builtin_popcount(vk_stage) == 1);
   return ffs(vk_stage) - 1;
}

static inline VkShaderStageFlagBits
mesa_to_vk_shader_stage(gl_shader_stage mesa_stage)
{
   return (1 << mesa_stage);
}

#define ANV_STAGE_MASK ((1 << MESA_SHADER_STAGES) - 1)

#define anv_foreach_stage(stage, stage_bits)                         \
   for (gl_shader_stage stage,                                       \
        __tmp = (gl_shader_stage)((stage_bits) & ANV_STAGE_MASK);    \
        stage = __builtin_ffs(__tmp) - 1, __tmp;                     \
        __tmp &= ~(1 << (stage)))

struct anv_pipeline_bind_map {
   uint32_t surface_count;
   uint32_t sampler_count;
   uint32_t image_count;

   struct anv_pipeline_binding *                surface_to_descriptor;
   struct anv_pipeline_binding *                sampler_to_descriptor;
};

struct anv_shader_bin_key {
   uint32_t size;
   uint8_t data[0];
};

struct anv_shader_bin {
   uint32_t ref_cnt;

   const struct anv_shader_bin_key *key;

   struct anv_state kernel;
   uint32_t kernel_size;

   struct anv_state constant_data;
   uint32_t constant_data_size;

   const struct brw_stage_prog_data *prog_data;
   uint32_t prog_data_size;

   struct anv_pipeline_bind_map bind_map;
};

struct anv_shader_bin *
anv_shader_bin_create(struct anv_device *device,
                      const void *key, uint32_t key_size,
                      const void *kernel, uint32_t kernel_size,
                      const void *constant_data, uint32_t constant_data_size,
                      const struct brw_stage_prog_data *prog_data,
                      uint32_t prog_data_size, const void *prog_data_param,
                      const struct anv_pipeline_bind_map *bind_map);

void
anv_shader_bin_destroy(struct anv_device *device, struct anv_shader_bin *shader);

static inline void
anv_shader_bin_ref(struct anv_shader_bin *shader)
{
   assert(shader && shader->ref_cnt >= 1);
   p_atomic_inc(&shader->ref_cnt);
}

static inline void
anv_shader_bin_unref(struct anv_device *device, struct anv_shader_bin *shader)
{
   assert(shader && shader->ref_cnt >= 1);
   if (p_atomic_dec_zero(&shader->ref_cnt))
      anv_shader_bin_destroy(device, shader);
}

struct anv_pipeline {
   struct anv_device *                          device;
   struct anv_batch                             batch;
   uint32_t                                     batch_data[512];
   struct anv_reloc_list                        batch_relocs;
   uint32_t                                     dynamic_state_mask;
   struct anv_dynamic_state                     dynamic_state;

   struct anv_subpass *                         subpass;

   bool                                         needs_data_cache;

   struct anv_shader_bin *                      shaders[MESA_SHADER_STAGES];

   struct {
      const struct gen_l3_config *              l3_config;
      uint32_t                                  total_size;
   } urb;

   VkShaderStageFlags                           active_stages;
   struct anv_state                             blend_state;

   uint32_t                                     vb_used;
   struct anv_pipeline_vertex_binding {
      uint32_t                                  stride;
      bool                                      instanced;
      uint32_t                                  instance_divisor;
   } vb[MAX_VBS];

   bool                                         primitive_restart;
   uint32_t                                     topology;

   uint32_t                                     cs_right_mask;

   bool                                         writes_depth;
   bool                                         depth_test_enable;
   bool                                         writes_stencil;
   bool                                         stencil_test_enable;
   bool                                         depth_clamp_enable;
   bool                                         sample_shading_enable;
   bool                                         kill_pixel;

   struct {
      uint32_t                                  sf[7];
      uint32_t                                  depth_stencil_state[3];
   } gen7;

   struct {
      uint32_t                                  sf[4];
      uint32_t                                  raster[5];
      uint32_t                                  wm_depth_stencil[3];
   } gen8;

   struct {
      uint32_t                                  wm_depth_stencil[4];
   } gen9;

   uint32_t                                     interface_descriptor_data[8];
};

static inline bool
anv_pipeline_has_stage(const struct anv_pipeline *pipeline,
                       gl_shader_stage stage)
{
   return (pipeline->active_stages & mesa_to_vk_shader_stage(stage)) != 0;
}

#define ANV_DECL_GET_PROG_DATA_FUNC(prefix, stage)                   \
static inline const struct brw_##prefix##_prog_data *                \
get_##prefix##_prog_data(const struct anv_pipeline *pipeline)        \
{                                                                    \
   if (anv_pipeline_has_stage(pipeline, stage)) {                    \
      return (const struct brw_##prefix##_prog_data *)               \
             pipeline->shaders[stage]->prog_data;                    \
   } else {                                                          \
      return NULL;                                                   \
   }                                                                 \
}

ANV_DECL_GET_PROG_DATA_FUNC(vs, MESA_SHADER_VERTEX)
ANV_DECL_GET_PROG_DATA_FUNC(tcs, MESA_SHADER_TESS_CTRL)
ANV_DECL_GET_PROG_DATA_FUNC(tes, MESA_SHADER_TESS_EVAL)
ANV_DECL_GET_PROG_DATA_FUNC(gs, MESA_SHADER_GEOMETRY)
ANV_DECL_GET_PROG_DATA_FUNC(wm, MESA_SHADER_FRAGMENT)
ANV_DECL_GET_PROG_DATA_FUNC(cs, MESA_SHADER_COMPUTE)

static inline const struct brw_vue_prog_data *
anv_pipeline_get_last_vue_prog_data(const struct anv_pipeline *pipeline)
{
   if (anv_pipeline_has_stage(pipeline, MESA_SHADER_GEOMETRY))
      return &get_gs_prog_data(pipeline)->base;
   else if (anv_pipeline_has_stage(pipeline, MESA_SHADER_TESS_EVAL))
      return &get_tes_prog_data(pipeline)->base;
   else
      return &get_vs_prog_data(pipeline)->base;
}

VkResult
anv_pipeline_init(struct anv_pipeline *pipeline, struct anv_device *device,
                  struct anv_pipeline_cache *cache,
                  const VkGraphicsPipelineCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *alloc);

VkResult
anv_pipeline_compile_cs(struct anv_pipeline *pipeline,
                        struct anv_pipeline_cache *cache,
                        const VkComputePipelineCreateInfo *info,
                        const struct anv_shader_module *module,
                        const char *entrypoint,
                        const VkSpecializationInfo *spec_info);

struct anv_format_plane {
   enum isl_format isl_format:16;
   struct isl_swizzle swizzle;

   /* Whether this plane contains chroma channels */
   bool has_chroma;

   /* For downscaling of YUV planes */
   uint8_t denominator_scales[2];

   /* How to map sampled ycbcr planes to a single 4 component element. */
   struct isl_swizzle ycbcr_swizzle;
};


struct anv_format {
   struct anv_format_plane planes[3];
   uint8_t n_planes;
   bool can_ycbcr;
};

static inline uint32_t
anv_image_aspect_to_plane(VkImageAspectFlags image_aspects,
                          VkImageAspectFlags aspect_mask)
{
   switch (aspect_mask) {
   case VK_IMAGE_ASPECT_COLOR_BIT:
   case VK_IMAGE_ASPECT_DEPTH_BIT:
   case VK_IMAGE_ASPECT_PLANE_0_BIT:
      return 0;
   case VK_IMAGE_ASPECT_STENCIL_BIT:
      if ((image_aspects & VK_IMAGE_ASPECT_DEPTH_BIT) == 0)
         return 0;
      /* Fall-through */
   case VK_IMAGE_ASPECT_PLANE_1_BIT:
      return 1;
   case VK_IMAGE_ASPECT_PLANE_2_BIT:
      return 2;
   default:
      /* Purposefully assert with depth/stencil aspects. */
      unreachable("invalid image aspect");
   }
}

static inline uint32_t
anv_image_aspect_get_planes(VkImageAspectFlags aspect_mask)
{
   uint32_t planes = 0;

   if (aspect_mask & (VK_IMAGE_ASPECT_COLOR_BIT |
                      VK_IMAGE_ASPECT_DEPTH_BIT |
                      VK_IMAGE_ASPECT_STENCIL_BIT |
                      VK_IMAGE_ASPECT_PLANE_0_BIT))
      planes++;
   if (aspect_mask & VK_IMAGE_ASPECT_PLANE_1_BIT)
      planes++;
   if (aspect_mask & VK_IMAGE_ASPECT_PLANE_2_BIT)
      planes++;

   if ((aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 &&
       (aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0)
      planes++;

   return planes;
}

static inline VkImageAspectFlags
anv_plane_to_aspect(VkImageAspectFlags image_aspects,
                    uint32_t plane)
{
   if (image_aspects & VK_IMAGE_ASPECT_ANY_COLOR_BIT_ANV) {
      if (util_bitcount(image_aspects) > 1)
         return VK_IMAGE_ASPECT_PLANE_0_BIT << plane;
      return VK_IMAGE_ASPECT_COLOR_BIT;
   }
   if (image_aspects & VK_IMAGE_ASPECT_DEPTH_BIT)
      return VK_IMAGE_ASPECT_DEPTH_BIT << plane;
   assert(image_aspects == VK_IMAGE_ASPECT_STENCIL_BIT);
   return VK_IMAGE_ASPECT_STENCIL_BIT;
}

#define anv_foreach_image_aspect_bit(b, image, aspects) \
   for_each_bit(b, anv_image_expand_aspects(image, aspects))

const struct anv_format *
anv_get_format(VkFormat format);

static inline uint32_t
anv_get_format_planes(VkFormat vk_format)
{
   const struct anv_format *format = anv_get_format(vk_format);

   return format != NULL ? format->n_planes : 0;
}

struct anv_format_plane
anv_get_format_plane(const struct gen_device_info *devinfo, VkFormat vk_format,
                     VkImageAspectFlagBits aspect, VkImageTiling tiling);

static inline enum isl_format
anv_get_isl_format(const struct gen_device_info *devinfo, VkFormat vk_format,
                   VkImageAspectFlags aspect, VkImageTiling tiling)
{
   return anv_get_format_plane(devinfo, vk_format, aspect, tiling).isl_format;
}

static inline struct isl_swizzle
anv_swizzle_for_render(struct isl_swizzle swizzle)
{
   /* Sometimes the swizzle will have alpha map to one.  We do this to fake
    * RGB as RGBA for texturing
    */
   assert(swizzle.a == ISL_CHANNEL_SELECT_ONE ||
          swizzle.a == ISL_CHANNEL_SELECT_ALPHA);

   /* But it doesn't matter what we render to that channel */
   swizzle.a = ISL_CHANNEL_SELECT_ALPHA;

   return swizzle;
}

void
anv_pipeline_setup_l3_config(struct anv_pipeline *pipeline, bool needs_slm);

/**
 * Subsurface of an anv_image.
 */
struct anv_surface {
   /** Valid only if isl_surf::size_B > 0. */
   struct isl_surf isl;

   /**
    * Offset from VkImage's base address, as bound by vkBindImageMemory().
    */
   uint32_t offset;
};

struct anv_image {
   VkImageType type;
   /* The original VkFormat provided by the client.  This may not match any
    * of the actual surface formats.
    */
   VkFormat vk_format;
   const struct anv_format *format;

   VkImageAspectFlags aspects;
   VkExtent3D extent;
   uint32_t levels;
   uint32_t array_size;
   uint32_t samples; /**< VkImageCreateInfo::samples */
   uint32_t n_planes;
   VkImageUsageFlags usage; /**< Superset of VkImageCreateInfo::usage. */
   VkImageTiling tiling; /** VkImageCreateInfo::tiling */

   /** True if this is needs to be bound to an appropriately tiled BO.
    *
    * When not using modifiers, consumers such as X11, Wayland, and KMS need
    * the tiling passed via I915_GEM_SET_TILING.  When exporting these buffers
    * we require a dedicated allocation so that we can know to allocate a
    * tiled buffer.
    */
   bool needs_set_tiling;

   /**
    * Must be DRM_FORMAT_MOD_INVALID unless tiling is
    * VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT.
    */
   uint64_t drm_format_mod;

   VkDeviceSize size;
   uint32_t alignment;

   /* Whether the image is made of several underlying buffer objects rather a
    * single one with different offsets.
    */
   bool disjoint;

   /**
    * Image subsurfaces
    *
    * For each foo, anv_image::planes[x].surface is valid if and only if
    * anv_image::aspects has a x aspect. Refer to anv_image_aspect_to_plane()
    * to figure the number associated with a given aspect.
    *
    * The hardware requires that the depth buffer and stencil buffer be
    * separate surfaces.  From Vulkan's perspective, though, depth and stencil
    * reside in the same VkImage.  To satisfy both the hardware and Vulkan, we
    * allocate the depth and stencil buffers as separate surfaces in the same
    * bo.
    *
    * Memory layout :
    *
    * -----------------------
    * |     surface0        |   /|\
    * -----------------------    |
    * |   shadow surface0   |    |
    * -----------------------    | Plane 0
    * |    aux surface0     |    |
    * -----------------------    |
    * | fast clear colors0  |   \|/
    * -----------------------
    * |     surface1        |   /|\
    * -----------------------    |
    * |   shadow surface1   |    |
    * -----------------------    | Plane 1
    * |    aux surface1     |    |
    * -----------------------    |
    * | fast clear colors1  |   \|/
    * -----------------------
    * |        ...          |
    * |                     |
    * -----------------------
    */
   struct {
      /**
       * Offset of the entire plane (whenever the image is disjoint this is
       * set to 0).
       */
      uint32_t offset;

      VkDeviceSize size;
      uint32_t alignment;

      struct anv_surface surface;

      /**
       * A surface which shadows the main surface and may have different
       * tiling. This is used for sampling using a tiling that isn't supported
       * for other operations.
       */
      struct anv_surface shadow_surface;

      /**
       * For color images, this is the aux usage for this image when not used
       * as a color attachment.
       *
       * For depth/stencil images, this is set to ISL_AUX_USAGE_HIZ if the
       * image has a HiZ buffer.
       */
      enum isl_aux_usage aux_usage;

      struct anv_surface aux_surface;

      /**
       * Offset of the fast clear state (used to compute the
       * fast_clear_state_offset of the following planes).
       */
      uint32_t fast_clear_state_offset;

      /**
       * BO associated with this plane, set when bound.
       */
      struct anv_address address;

      /**
       * When destroying the image, also free the bo.
       * */
      bool bo_is_owned;
   } planes[3];
};

/* The ordering of this enum is important */
enum anv_fast_clear_type {
   /** Image does not have/support any fast-clear blocks */
   ANV_FAST_CLEAR_NONE = 0,
   /** Image has/supports fast-clear but only to the default value */
   ANV_FAST_CLEAR_DEFAULT_VALUE = 1,
   /** Image has/supports fast-clear with an arbitrary fast-clear value */
   ANV_FAST_CLEAR_ANY = 2,
};

/* Returns the number of auxiliary buffer levels attached to an image. */
static inline uint8_t
anv_image_aux_levels(const struct anv_image * const image,
                     VkImageAspectFlagBits aspect)
{
   uint32_t plane = anv_image_aspect_to_plane(image->aspects, aspect);
   return image->planes[plane].aux_surface.isl.size_B > 0 ?
          image->planes[plane].aux_surface.isl.levels : 0;
}

/* Returns the number of auxiliary buffer layers attached to an image. */
static inline uint32_t
anv_image_aux_layers(const struct anv_image * const image,
                     VkImageAspectFlagBits aspect,
                     const uint8_t miplevel)
{
   assert(image);

   /* The miplevel must exist in the main buffer. */
   assert(miplevel < image->levels);

   if (miplevel >= anv_image_aux_levels(image, aspect)) {
      /* There are no layers with auxiliary data because the miplevel has no
       * auxiliary data.
       */
      return 0;
   } else {
      uint32_t plane = anv_image_aspect_to_plane(image->aspects, aspect);
      return MAX2(image->planes[plane].aux_surface.isl.logical_level0_px.array_len,
                  image->planes[plane].aux_surface.isl.logical_level0_px.depth >> miplevel);
   }
}

static inline struct anv_address
anv_image_get_clear_color_addr(const struct anv_device *device,
                               const struct anv_image *image,
                               VkImageAspectFlagBits aspect)
{
   assert(image->aspects & VK_IMAGE_ASPECT_ANY_COLOR_BIT_ANV);

   uint32_t plane = anv_image_aspect_to_plane(image->aspects, aspect);
   return anv_address_add(image->planes[plane].address,
                          image->planes[plane].fast_clear_state_offset);
}

static inline struct anv_address
anv_image_get_fast_clear_type_addr(const struct anv_device *device,
                                   const struct anv_image *image,
                                   VkImageAspectFlagBits aspect)
{
   struct anv_address addr =
      anv_image_get_clear_color_addr(device, image, aspect);

   const unsigned clear_color_state_size = device->info.gen >= 10 ?
      device->isl_dev.ss.clear_color_state_size :
      device->isl_dev.ss.clear_value_size;
   addr.offset += clear_color_state_size;
   return addr;
}

static inline struct anv_address
anv_image_get_compression_state_addr(const struct anv_device *device,
                                     const struct anv_image *image,
                                     VkImageAspectFlagBits aspect,
                                     uint32_t level, uint32_t array_layer)
{
   assert(level < anv_image_aux_levels(image, aspect));
   assert(array_layer < anv_image_aux_layers(image, aspect, level));
   UNUSED uint32_t plane = anv_image_aspect_to_plane(image->aspects, aspect);
   assert(image->planes[plane].aux_usage == ISL_AUX_USAGE_CCS_E);

   struct anv_address addr =
      anv_image_get_fast_clear_type_addr(device, image, aspect);
   addr.offset += 4; /* Go past the fast clear type */

   if (image->type == VK_IMAGE_TYPE_3D) {
      for (uint32_t l = 0; l < level; l++)
         addr.offset += anv_minify(image->extent.depth, l) * 4;
   } else {
      addr.offset += level * image->array_size * 4;
   }
   addr.offset += array_layer * 4;

   return addr;
}

/* Returns true if a HiZ-enabled depth buffer can be sampled from. */
static inline bool
anv_can_sample_with_hiz(const struct gen_device_info * const devinfo,
                        const struct anv_image *image)
{
   if (!(image->aspects & VK_IMAGE_ASPECT_DEPTH_BIT))
      return false;

   if (devinfo->gen < 8)
      return false;

   return image->samples == 1;
}

void
anv_cmd_buffer_mark_image_written(struct anv_cmd_buffer *cmd_buffer,
                                  const struct anv_image *image,
                                  VkImageAspectFlagBits aspect,
                                  enum isl_aux_usage aux_usage,
                                  uint32_t level,
                                  uint32_t base_layer,
                                  uint32_t layer_count);

void
anv_image_clear_color(struct anv_cmd_buffer *cmd_buffer,
                      const struct anv_image *image,
                      VkImageAspectFlagBits aspect,
                      enum isl_aux_usage aux_usage,
                      enum isl_format format, struct isl_swizzle swizzle,
                      uint32_t level, uint32_t base_layer, uint32_t layer_count,
                      VkRect2D area, union isl_color_value clear_color);
void
anv_image_clear_depth_stencil(struct anv_cmd_buffer *cmd_buffer,
                              const struct anv_image *image,
                              VkImageAspectFlags aspects,
                              enum isl_aux_usage depth_aux_usage,
                              uint32_t level,
                              uint32_t base_layer, uint32_t layer_count,
                              VkRect2D area,
                              float depth_value, uint8_t stencil_value);
void
anv_image_hiz_op(struct anv_cmd_buffer *cmd_buffer,
                 const struct anv_image *image,
                 VkImageAspectFlagBits aspect, uint32_t level,
                 uint32_t base_layer, uint32_t layer_count,
                 enum isl_aux_op hiz_op);
void
anv_image_hiz_clear(struct anv_cmd_buffer *cmd_buffer,
                    const struct anv_image *image,
                    VkImageAspectFlags aspects,
                    uint32_t level,
                    uint32_t base_layer, uint32_t layer_count,
                    VkRect2D area, uint8_t stencil_value);
void
anv_image_mcs_op(struct anv_cmd_buffer *cmd_buffer,
                 const struct anv_image *image,
                 enum isl_format format,
                 VkImageAspectFlagBits aspect,
                 uint32_t base_layer, uint32_t layer_count,
                 enum isl_aux_op mcs_op, union isl_color_value *clear_value,
                 bool predicate);
void
anv_image_ccs_op(struct anv_cmd_buffer *cmd_buffer,
                 const struct anv_image *image,
                 enum isl_format format,
                 VkImageAspectFlagBits aspect, uint32_t level,
                 uint32_t base_layer, uint32_t layer_count,
                 enum isl_aux_op ccs_op, union isl_color_value *clear_value,
                 bool predicate);

void
anv_image_copy_to_shadow(struct anv_cmd_buffer *cmd_buffer,
                         const struct anv_image *image,
                         uint32_t base_level, uint32_t level_count,
                         uint32_t base_layer, uint32_t layer_count);

enum isl_aux_usage
anv_layout_to_aux_usage(const struct gen_device_info * const devinfo,
                        const struct anv_image *image,
                        const VkImageAspectFlagBits aspect,
                        const VkImageLayout layout);

enum anv_fast_clear_type
anv_layout_to_fast_clear_type(const struct gen_device_info * const devinfo,
                              const struct anv_image * const image,
                              const VkImageAspectFlagBits aspect,
                              const VkImageLayout layout);

/* This is defined as a macro so that it works for both
 * VkImageSubresourceRange and VkImageSubresourceLayers
 */
#define anv_get_layerCount(_image, _range) \
   ((_range)->layerCount == VK_REMAINING_ARRAY_LAYERS ? \
    (_image)->array_size - (_range)->baseArrayLayer : (_range)->layerCount)

static inline uint32_t
anv_get_levelCount(const struct anv_image *image,
                   const VkImageSubresourceRange *range)
{
   return range->levelCount == VK_REMAINING_MIP_LEVELS ?
          image->levels - range->baseMipLevel : range->levelCount;
}

static inline VkImageAspectFlags
anv_image_expand_aspects(const struct anv_image *image,
                         VkImageAspectFlags aspects)
{
   /* If the underlying image has color plane aspects and
    * VK_IMAGE_ASPECT_COLOR_BIT has been requested, then return the aspects of
    * the underlying image. */
   if ((image->aspects & VK_IMAGE_ASPECT_PLANES_BITS_ANV) != 0 &&
       aspects == VK_IMAGE_ASPECT_COLOR_BIT)
      return image->aspects;

   return aspects;
}

static inline bool
anv_image_aspects_compatible(VkImageAspectFlags aspects1,
                             VkImageAspectFlags aspects2)
{
   if (aspects1 == aspects2)
      return true;

   /* Only 1 color aspects are compatibles. */
   if ((aspects1 & VK_IMAGE_ASPECT_ANY_COLOR_BIT_ANV) != 0 &&
       (aspects2 & VK_IMAGE_ASPECT_ANY_COLOR_BIT_ANV) != 0 &&
       util_bitcount(aspects1) == util_bitcount(aspects2))
      return true;

   return false;
}

struct anv_image_view {
   const struct anv_image *image; /**< VkImageViewCreateInfo::image */

   VkImageAspectFlags aspect_mask;
   VkFormat vk_format;
   VkExtent3D extent; /**< Extent of VkImageViewCreateInfo::baseMipLevel. */

   unsigned n_planes;
   struct {
      uint32_t image_plane;

      struct isl_view isl;

      /**
       * RENDER_SURFACE_STATE when using image as a sampler surface with an
       * image layout of SHADER_READ_ONLY_OPTIMAL or
       * DEPTH_STENCIL_READ_ONLY_OPTIMAL.
       */
      struct anv_surface_state optimal_sampler_surface_state;

      /**
       * RENDER_SURFACE_STATE when using image as a sampler surface with an
       * image layout of GENERAL.
       */
      struct anv_surface_state general_sampler_surface_state;

      /**
       * RENDER_SURFACE_STATE when using image as a storage image. Separate
       * states for write-only and readable, using the real format for
       * write-only and the lowered format for readable.
       */
      struct anv_surface_state storage_surface_state;
      struct anv_surface_state writeonly_storage_surface_state;

      struct brw_image_param storage_image_param;
   } planes[3];
};

enum anv_image_view_state_flags {
   ANV_IMAGE_VIEW_STATE_STORAGE_WRITE_ONLY   = (1 << 0),
   ANV_IMAGE_VIEW_STATE_TEXTURE_OPTIMAL      = (1 << 1),
};

void anv_image_fill_surface_state(struct anv_device *device,
                                  const struct anv_image *image,
                                  VkImageAspectFlagBits aspect,
                                  const struct isl_view *view,
                                  isl_surf_usage_flags_t view_usage,
                                  enum isl_aux_usage aux_usage,
                                  const union isl_color_value *clear_color,
                                  enum anv_image_view_state_flags flags,
                                  struct anv_surface_state *state_inout,
                                  struct brw_image_param *image_param_out);

struct anv_image_create_info {
   const VkImageCreateInfo *vk_info;

   /** An opt-in bitmask which filters an ISL-mapping of the Vulkan tiling. */
   isl_tiling_flags_t isl_tiling_flags;

   /** These flags will be added to any derived from VkImageCreateInfo. */
   isl_surf_usage_flags_t isl_extra_usage_flags;

   uint32_t stride;
};

VkResult anv_image_create(VkDevice _device,
                          const struct anv_image_create_info *info,
                          const VkAllocationCallbacks* alloc,
                          VkImage *pImage);

#ifdef ANDROID
VkResult anv_image_from_gralloc(VkDevice device_h,
                                const VkImageCreateInfo *base_info,
                                const VkNativeBufferANDROID *gralloc_info,
                                const VkAllocationCallbacks *alloc,
                                VkImage *pImage);
#endif

const struct anv_surface *
anv_image_get_surface_for_aspect_mask(const struct anv_image *image,
                                      VkImageAspectFlags aspect_mask);

enum isl_format
anv_isl_format_for_descriptor_type(VkDescriptorType type);

static inline struct VkExtent3D
anv_sanitize_image_extent(const VkImageType imageType,
                          const struct VkExtent3D imageExtent)
{
   switch (imageType) {
   case VK_IMAGE_TYPE_1D:
      return (VkExtent3D) { imageExtent.width, 1, 1 };
   case VK_IMAGE_TYPE_2D:
      return (VkExtent3D) { imageExtent.width, imageExtent.height, 1 };
   case VK_IMAGE_TYPE_3D:
      return imageExtent;
   default:
      unreachable("invalid image type");
   }
}

static inline struct VkOffset3D
anv_sanitize_image_offset(const VkImageType imageType,
                          const struct VkOffset3D imageOffset)
{
   switch (imageType) {
   case VK_IMAGE_TYPE_1D:
      return (VkOffset3D) { imageOffset.x, 0, 0 };
   case VK_IMAGE_TYPE_2D:
      return (VkOffset3D) { imageOffset.x, imageOffset.y, 0 };
   case VK_IMAGE_TYPE_3D:
      return imageOffset;
   default:
      unreachable("invalid image type");
   }
}


void anv_fill_buffer_surface_state(struct anv_device *device,
                                   struct anv_state state,
                                   enum isl_format format,
                                   struct anv_address address,
                                   uint32_t range, uint32_t stride);

static inline void
anv_clear_color_from_att_state(union isl_color_value *clear_color,
                               const struct anv_attachment_state *att_state,
                               const struct anv_image_view *iview)
{
   const struct isl_format_layout *view_fmtl =
      isl_format_get_layout(iview->planes[0].isl.format);

#define COPY_CLEAR_COLOR_CHANNEL(c, i) \
   if (view_fmtl->channels.c.bits) \
      clear_color->u32[i] = att_state->clear_value.color.uint32[i]

   COPY_CLEAR_COLOR_CHANNEL(r, 0);
   COPY_CLEAR_COLOR_CHANNEL(g, 1);
   COPY_CLEAR_COLOR_CHANNEL(b, 2);
   COPY_CLEAR_COLOR_CHANNEL(a, 3);

#undef COPY_CLEAR_COLOR_CHANNEL
}


struct anv_ycbcr_conversion {
   const struct anv_format *        format;
   VkSamplerYcbcrModelConversion    ycbcr_model;
   VkSamplerYcbcrRange              ycbcr_range;
   VkComponentSwizzle               mapping[4];
   VkChromaLocation                 chroma_offsets[2];
   VkFilter                         chroma_filter;
   bool                             chroma_reconstruction;
};

struct anv_sampler {
   uint32_t                     state[3][4];
   uint32_t                     n_planes;
   struct anv_ycbcr_conversion *conversion;
};

struct anv_framebuffer {
   uint32_t                                     width;
   uint32_t                                     height;
   uint32_t                                     layers;

   uint32_t                                     attachment_count;
   struct anv_image_view *                      attachments[0];
};

struct anv_subpass_attachment {
   VkImageUsageFlagBits usage;
   uint32_t attachment;
   VkImageLayout layout;
};

struct anv_subpass {
   uint32_t                                     attachment_count;

   /**
    * A pointer to all attachment references used in this subpass.
    * Only valid if ::attachment_count > 0.
    */
   struct anv_subpass_attachment *              attachments;
   uint32_t                                     input_count;
   struct anv_subpass_attachment *              input_attachments;
   uint32_t                                     color_count;
   struct anv_subpass_attachment *              color_attachments;
   struct anv_subpass_attachment *              resolve_attachments;

   struct anv_subpass_attachment *              depth_stencil_attachment;

   uint32_t                                     view_mask;

   /** Subpass has a depth/stencil self-dependency */
   bool                                         has_ds_self_dep;

   /** Subpass has at least one resolve attachment */
   bool                                         has_resolve;
};

static inline unsigned
anv_subpass_view_count(const struct anv_subpass *subpass)
{
   return MAX2(1, util_bitcount(subpass->view_mask));
}

struct anv_render_pass_attachment {
   /* TODO: Consider using VkAttachmentDescription instead of storing each of
    * its members individually.
    */
   VkFormat                                     format;
   uint32_t                                     samples;
   VkImageUsageFlags                            usage;
   VkAttachmentLoadOp                           load_op;
   VkAttachmentStoreOp                          store_op;
   VkAttachmentLoadOp                           stencil_load_op;
   VkImageLayout                                initial_layout;
   VkImageLayout                                final_layout;
   VkImageLayout                                first_subpass_layout;

   /* The subpass id in which the attachment will be used last. */
   uint32_t                                     last_subpass_idx;
};

struct anv_render_pass {
   uint32_t                                     attachment_count;
   uint32_t                                     subpass_count;
   /* An array of subpass_count+1 flushes, one per subpass boundary */
   enum anv_pipe_bits *                         subpass_flushes;
   struct anv_render_pass_attachment *          attachments;
   struct anv_subpass                           subpasses[0];
};

#define ANV_PIPELINE_STATISTICS_MASK 0x000007ff

struct anv_query_pool {
   VkQueryType                                  type;
   VkQueryPipelineStatisticFlags                pipeline_statistics;
   /** Stride between slots, in bytes */
   uint32_t                                     stride;
   /** Number of slots in this query pool */
   uint32_t                                     slots;
   struct anv_bo                                bo;
};

int anv_get_instance_entrypoint_index(const char *name);
int anv_get_device_entrypoint_index(const char *name);

bool
anv_instance_entrypoint_is_enabled(int index, uint32_t core_version,
                                   const struct anv_instance_extension_table *instance);

bool
anv_device_entrypoint_is_enabled(int index, uint32_t core_version,
                                 const struct anv_instance_extension_table *instance,
                                 const struct anv_device_extension_table *device);

void *anv_lookup_entrypoint(const struct gen_device_info *devinfo,
                            const char *name);

void anv_dump_image_to_ppm(struct anv_device *device,
                           struct anv_image *image, unsigned miplevel,
                           unsigned array_layer, VkImageAspectFlagBits aspect,
                           const char *filename);

enum anv_dump_action {
   ANV_DUMP_FRAMEBUFFERS_BIT = 0x1,
};

void anv_dump_start(struct anv_device *device, enum anv_dump_action actions);
void anv_dump_finish(void);

void anv_dump_add_framebuffer(struct anv_cmd_buffer *cmd_buffer,
                              struct anv_framebuffer *fb);

static inline uint32_t
anv_get_subpass_id(const struct anv_cmd_state * const cmd_state)
{
   /* This function must be called from within a subpass. */
   assert(cmd_state->pass && cmd_state->subpass);

   const uint32_t subpass_id = cmd_state->subpass - cmd_state->pass->subpasses;

   /* The id of this subpass shouldn't exceed the number of subpasses in this
    * render pass minus 1.
    */
   assert(subpass_id < cmd_state->pass->subpass_count);
   return subpass_id;
}

#define ANV_DEFINE_HANDLE_CASTS(__anv_type, __VkType)                      \
                                                                           \
   static inline struct __anv_type *                                       \
   __anv_type ## _from_handle(__VkType _handle)                            \
   {                                                                       \
      return (struct __anv_type *) _handle;                                \
   }                                                                       \
                                                                           \
   static inline __VkType                                                  \
   __anv_type ## _to_handle(struct __anv_type *_obj)                       \
   {                                                                       \
      return (__VkType) _obj;                                              \
   }

#define ANV_DEFINE_NONDISP_HANDLE_CASTS(__anv_type, __VkType)              \
                                                                           \
   static inline struct __anv_type *                                       \
   __anv_type ## _from_handle(__VkType _handle)                            \
   {                                                                       \
      return (struct __anv_type *)(uintptr_t) _handle;                     \
   }                                                                       \
                                                                           \
   static inline __VkType                                                  \
   __anv_type ## _to_handle(struct __anv_type *_obj)                       \
   {                                                                       \
      return (__VkType)(uintptr_t) _obj;                                   \
   }

#define ANV_FROM_HANDLE(__anv_type, __name, __handle) \
   struct __anv_type *__name = __anv_type ## _from_handle(__handle)

ANV_DEFINE_HANDLE_CASTS(anv_cmd_buffer, VkCommandBuffer)
ANV_DEFINE_HANDLE_CASTS(anv_device, VkDevice)
ANV_DEFINE_HANDLE_CASTS(anv_instance, VkInstance)
ANV_DEFINE_HANDLE_CASTS(anv_physical_device, VkPhysicalDevice)
ANV_DEFINE_HANDLE_CASTS(anv_queue, VkQueue)

ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_cmd_pool, VkCommandPool)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_buffer, VkBuffer)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_buffer_view, VkBufferView)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_descriptor_pool, VkDescriptorPool)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_descriptor_set, VkDescriptorSet)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_descriptor_set_layout, VkDescriptorSetLayout)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_descriptor_update_template, VkDescriptorUpdateTemplateKHR)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_device_memory, VkDeviceMemory)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_fence, VkFence)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_event, VkEvent)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_framebuffer, VkFramebuffer)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_image, VkImage)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_image_view, VkImageView);
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_pipeline_cache, VkPipelineCache)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_pipeline, VkPipeline)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_pipeline_layout, VkPipelineLayout)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_query_pool, VkQueryPool)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_render_pass, VkRenderPass)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_sampler, VkSampler)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_semaphore, VkSemaphore)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_shader_module, VkShaderModule)
ANV_DEFINE_NONDISP_HANDLE_CASTS(vk_debug_report_callback, VkDebugReportCallbackEXT)
ANV_DEFINE_NONDISP_HANDLE_CASTS(anv_ycbcr_conversion, VkSamplerYcbcrConversion)

/* Gen-specific function declarations */
#ifdef genX
#  include "anv_genX.h"
#else
#  define genX(x) gen7_##x
#  include "anv_genX.h"
#  undef genX
#  define genX(x) gen75_##x
#  include "anv_genX.h"
#  undef genX
#  define genX(x) gen8_##x
#  include "anv_genX.h"
#  undef genX
#  define genX(x) gen9_##x
#  include "anv_genX.h"
#  undef genX
#  define genX(x) gen10_##x
#  include "anv_genX.h"
#  undef genX
#  define genX(x) gen11_##x
#  include "anv_genX.h"
#  undef genX
#endif

#endif /* ANV_PRIVATE_H */
