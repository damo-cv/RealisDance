/*
 * Copyright © 2011 Intel Corporation
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Benjamin Franzke <benjaminfranzke@googlemail.com>
 */

#ifndef INTERNAL_H_
#define INTERNAL_H_

#include "gbm.h"
#include <sys/stat.h>

/* GCC visibility */
#if defined(__GNUC__)
#define GBM_EXPORT __attribute__ ((visibility("default")))
#else
#define GBM_EXPORT
#endif

/**
 * \file gbmint.h
 * \brief Internal implementation details of gbm
 */

/**
 * The device used for the memory allocation.
 *
 * The members of this structure should be not accessed directly
 */
struct gbm_device {
   /* Hack to make a gbm_device detectable by its first element. */
   struct gbm_device *(*dummy)(int);

   int fd;
   const char *name;
   unsigned int refcount;
   struct stat stat;

   void (*destroy)(struct gbm_device *gbm);
   int (*is_format_supported)(struct gbm_device *gbm,
                              uint32_t format,
                              uint32_t usage);
   int (*get_format_modifier_plane_count)(struct gbm_device *device,
                                          uint32_t format,
                                          uint64_t modifier);

   struct gbm_bo *(*bo_create)(struct gbm_device *gbm,
                               uint32_t width, uint32_t height,
                               uint32_t format,
                               uint32_t usage,
                               const uint64_t *modifiers,
                               const unsigned int count);
   struct gbm_bo *(*bo_import)(struct gbm_device *gbm, uint32_t type,
                               void *buffer, uint32_t usage);
   void *(*bo_map)(struct gbm_bo *bo,
                               uint32_t x, uint32_t y,
                               uint32_t width, uint32_t height,
                               uint32_t flags, uint32_t *stride,
                               void **map_data);
   void (*bo_unmap)(struct gbm_bo *bo, void *map_data);
   int (*bo_write)(struct gbm_bo *bo, const void *buf, size_t data);
   int (*bo_get_fd)(struct gbm_bo *bo);
   int (*bo_get_planes)(struct gbm_bo *bo);
   union gbm_bo_handle (*bo_get_handle)(struct gbm_bo *bo, int plane);
   uint32_t (*bo_get_stride)(struct gbm_bo *bo, int plane);
   uint32_t (*bo_get_offset)(struct gbm_bo *bo, int plane);
   uint64_t (*bo_get_modifier)(struct gbm_bo *bo);
   void (*bo_destroy)(struct gbm_bo *bo);

   struct gbm_surface *(*surface_create)(struct gbm_device *gbm,
                                         uint32_t width, uint32_t height,
                                         uint32_t format, uint32_t flags,
                                         const uint64_t *modifiers,
                                         const unsigned count);
   struct gbm_bo *(*surface_lock_front_buffer)(struct gbm_surface *surface);
   void (*surface_release_buffer)(struct gbm_surface *surface,
                                  struct gbm_bo *bo);
   int (*surface_has_free_buffers)(struct gbm_surface *surface);
   void (*surface_destroy)(struct gbm_surface *surface);
};

/**
 * The allocated buffer object.
 *
 * The members in this structure should not be accessed directly.
 */
struct gbm_bo {
   struct gbm_device *gbm;
   uint32_t width;
   uint32_t height;
   uint32_t stride;
   uint32_t format;
   union gbm_bo_handle  handle;
   void *user_data;
   void (*destroy_user_data)(struct gbm_bo *, void *);
};

struct gbm_surface {
   struct gbm_device *gbm;
   uint32_t width;
   uint32_t height;
   uint32_t format;
   uint32_t flags;
   struct {
      uint64_t *modifiers;
      unsigned count;
   };
};

struct gbm_backend {
   const char *backend_name;
   struct gbm_device *(*create_device)(int fd);
};

#endif
