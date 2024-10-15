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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef MAJOR_IN_MKDEV
#include <sys/mkdev.h>
#endif
#ifdef MAJOR_IN_SYSMACROS
#include <sys/sysmacros.h>
#endif
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include "gbm.h"
#include "gbmint.h"
#include "backend.h"

/** Returns the file description for the gbm device
 *
 * \return The fd that the struct gbm_device was created with
 */
GBM_EXPORT int
gbm_device_get_fd(struct gbm_device *gbm)
{
   return gbm->fd;
}

/** Get the backend name for the given gbm device
 *
 * \return The backend name string - this belongs to the device and must not
 * be freed
 */
GBM_EXPORT const char *
gbm_device_get_backend_name(struct gbm_device *gbm)
{
   return gbm->name;
}

/** Test if a format is supported for a given set of usage flags.
 *
 * \param gbm The created buffer manager
 * \param format The format to test
 * \param usage A bitmask of the usages to test the format against
 * \return 1 if the format is supported otherwise 0
 *
 * \sa enum gbm_bo_flags for the list of flags that the format can be
 * tested against
 *
 * \sa enum gbm_bo_format for the list of formats
 */
GBM_EXPORT int
gbm_device_is_format_supported(struct gbm_device *gbm,
                               uint32_t format, uint32_t usage)
{
   return gbm->is_format_supported(gbm, format, usage);
}

/** Get the number of planes that are required for a given format+modifier
 *
 * \param gbm The gbm device returned from gbm_create_device()
 * \param format The format to query
 * \param modifier The modifier to query
 */
GBM_EXPORT int
gbm_device_get_format_modifier_plane_count(struct gbm_device *gbm,
                                           uint32_t format,
                                           uint64_t modifier)
{
   return gbm->get_format_modifier_plane_count(gbm, format, modifier);
}

/** Destroy the gbm device and free all resources associated with it.
 *
 * \param gbm The device created using gbm_create_device()
 */
GBM_EXPORT void
gbm_device_destroy(struct gbm_device *gbm)
{
   gbm->refcount--;
   if (gbm->refcount == 0)
      gbm->destroy(gbm);
}

/** Create a gbm device for allocating buffers
 *
 * The file descriptor passed in is used by the backend to communicate with
 * platform for allocating the memory. For allocations using DRI this would be
 * the file descriptor returned when opening a device such as \c
 * /dev/dri/card0
 *
 * \param fd The file descriptor for a backend specific device
 * \return The newly created struct gbm_device. The resources associated with
 * the device should be freed with gbm_device_destroy() when it is no longer
 * needed. If the creation of the device failed NULL will be returned.
 */
GBM_EXPORT struct gbm_device *
gbm_create_device(int fd)
{
   struct gbm_device *gbm = NULL;
   struct stat buf;

   if (fd < 0 || fstat(fd, &buf) < 0 || !S_ISCHR(buf.st_mode)) {
      errno = EINVAL;
      return NULL;
   }

   gbm = _gbm_create_device(fd);
   if (gbm == NULL)
      return NULL;

   gbm->dummy = gbm_create_device;
   gbm->stat = buf;
   gbm->refcount = 1;

   return gbm;
}

/** Get the width of the buffer object
 *
 * \param bo The buffer object
 * \return The width of the allocated buffer object
 *
 */
GBM_EXPORT uint32_t
gbm_bo_get_width(struct gbm_bo *bo)
{
   return bo->width;
}

/** Get the height of the buffer object
 *
 * \param bo The buffer object
 * \return The height of the allocated buffer object
 */
GBM_EXPORT uint32_t
gbm_bo_get_height(struct gbm_bo *bo)
{
   return bo->height;
}

/** Get the stride of the buffer object
 *
 * This is calculated by the backend when it does the allocation in
 * gbm_bo_create()
 *
 * \param bo The buffer object
 * \return The stride of the allocated buffer object in bytes
 */
GBM_EXPORT uint32_t
gbm_bo_get_stride(struct gbm_bo *bo)
{
   return gbm_bo_get_stride_for_plane(bo, 0);
}

/** Get the stride for the given plane
 *
 * \param bo The buffer object
 * \param plane for which you want the stride
 *
 * \sa gbm_bo_get_stride()
 */
GBM_EXPORT uint32_t
gbm_bo_get_stride_for_plane(struct gbm_bo *bo, int plane)
{
   return bo->gbm->bo_get_stride(bo, plane);
}

/** Get the format of the buffer object
 *
 * The format of the pixels in the buffer.
 *
 * \param bo The buffer object
 * \return The format of buffer object, one of the GBM_FORMAT_* codes
 */
GBM_EXPORT uint32_t
gbm_bo_get_format(struct gbm_bo *bo)
{
   return bo->format;
}

/** Get the bit-per-pixel of the buffer object's format
 *
 * The bits-per-pixel of the buffer object's format.
 *
 * Note; The 'in-memory pixel' concept makes no sense for YUV formats
 * (pixels are the result of the combination of multiple memory sources:
 * Y, Cb & Cr; usually these are even in separate buffers), so YUV
 * formats are not supported by this function.
 *
 * \param bo The buffer object
 * \return The number of bits0per-pixel of the buffer object's format.
 */
GBM_EXPORT uint32_t
gbm_bo_get_bpp(struct gbm_bo *bo)
{
   switch (bo->format) {
      default:
         return 0;
      case GBM_FORMAT_C8:
      case GBM_FORMAT_R8:
      case GBM_FORMAT_RGB332:
      case GBM_FORMAT_BGR233:
         return 8;
      case GBM_FORMAT_GR88:
      case GBM_FORMAT_XRGB4444:
      case GBM_FORMAT_XBGR4444:
      case GBM_FORMAT_RGBX4444:
      case GBM_FORMAT_BGRX4444:
      case GBM_FORMAT_ARGB4444:
      case GBM_FORMAT_ABGR4444:
      case GBM_FORMAT_RGBA4444:
      case GBM_FORMAT_BGRA4444:
      case GBM_FORMAT_XRGB1555:
      case GBM_FORMAT_XBGR1555:
      case GBM_FORMAT_RGBX5551:
      case GBM_FORMAT_BGRX5551:
      case GBM_FORMAT_ARGB1555:
      case GBM_FORMAT_ABGR1555:
      case GBM_FORMAT_RGBA5551:
      case GBM_FORMAT_BGRA5551:
      case GBM_FORMAT_RGB565:
      case GBM_FORMAT_BGR565:
         return 16;
      case GBM_FORMAT_RGB888:
      case GBM_FORMAT_BGR888:
         return 24;
      case GBM_FORMAT_XRGB8888:
      case GBM_FORMAT_XBGR8888:
      case GBM_FORMAT_RGBX8888:
      case GBM_FORMAT_BGRX8888:
      case GBM_FORMAT_ARGB8888:
      case GBM_FORMAT_ABGR8888:
      case GBM_FORMAT_RGBA8888:
      case GBM_FORMAT_BGRA8888:
      case GBM_FORMAT_XRGB2101010:
      case GBM_FORMAT_XBGR2101010:
      case GBM_FORMAT_RGBX1010102:
      case GBM_FORMAT_BGRX1010102:
      case GBM_FORMAT_ARGB2101010:
      case GBM_FORMAT_ABGR2101010:
      case GBM_FORMAT_RGBA1010102:
      case GBM_FORMAT_BGRA1010102:
         return 32;
   }
}

/** Get the offset for the data of the specified plane
 *
 * Extra planes, and even the first plane, may have an offset from the start of
 * the buffer object. This function will provide the offset for the given plane
 * to be used in various KMS APIs.
 *
 * \param bo The buffer object
 * \return The offset
 */
GBM_EXPORT uint32_t
gbm_bo_get_offset(struct gbm_bo *bo, int plane)
{
   return bo->gbm->bo_get_offset(bo, plane);
}

/** Get the gbm device used to create the buffer object
 *
 * \param bo The buffer object
 * \return Returns the gbm device with which the buffer object was created
 */
GBM_EXPORT struct gbm_device *
gbm_bo_get_device(struct gbm_bo *bo)
{
	return bo->gbm;
}

/** Get the handle of the buffer object
 *
 * This is stored in the platform generic union gbm_bo_handle type. However
 * the format of this handle is platform specific.
 *
 * \param bo The buffer object
 * \return Returns the handle of the allocated buffer object
 */
GBM_EXPORT union gbm_bo_handle
gbm_bo_get_handle(struct gbm_bo *bo)
{
   return bo->handle;
}

/** Get a DMA-BUF file descriptor for the buffer object
 *
 * This function creates a DMA-BUF (also known as PRIME) file descriptor
 * handle for the buffer object.  Each call to gbm_bo_get_fd() returns a new
 * file descriptor and the caller is responsible for closing the file
 * descriptor.

 * \param bo The buffer object
 * \return Returns a file descriptor referring to the underlying buffer or -1
 * if an error occurs.
 */
GBM_EXPORT int
gbm_bo_get_fd(struct gbm_bo *bo)
{
   return bo->gbm->bo_get_fd(bo);
}

/** Get the number of planes for the given bo.
 *
 * \param bo The buffer object
 * \return The number of planes
 */
GBM_EXPORT int
gbm_bo_get_plane_count(struct gbm_bo *bo)
{
   return bo->gbm->bo_get_planes(bo);
}

/** Get the handle for the specified plane of the buffer object
 *
 * This function gets the handle for any plane associated with the BO. When
 * dealing with multi-planar formats, or formats which might have implicit
 * planes based on different underlying hardware it is necessary for the client
 * to be able to get this information to pass to the DRM.
 *
 * \param bo The buffer object
 * \param plane the plane to get a handle for
 *
 * \sa gbm_bo_get_handle()
 */
GBM_EXPORT union gbm_bo_handle
gbm_bo_get_handle_for_plane(struct gbm_bo *bo, int plane)
{
   return bo->gbm->bo_get_handle(bo, plane);
}

/**
 * Get the chosen modifier for the buffer object
 *
 * This function returns the modifier that was chosen for the object. These
 * properties may be generic, or platform/implementation dependent.
 *
 * \param bo The buffer object
 * \return Returns the selected modifier (chosen by the implementation) for the
 * BO.
 * \sa gbm_bo_create_with_modifiers() where possible modifiers are set
 * \sa gbm_surface_create_with_modifiers() where possible modifiers are set
 * \sa define DRM_FORMAT_MOD_* in drm_fourcc.h for possible modifiers
 */
GBM_EXPORT uint64_t
gbm_bo_get_modifier(struct gbm_bo *bo)
{
   return bo->gbm->bo_get_modifier(bo);
}

/** Write data into the buffer object
 *
 * If the buffer object was created with the GBM_BO_USE_WRITE flag,
 * this function can be used to write data into the buffer object.  The
 * data is copied directly into the object and it's the responsibility
 * of the caller to make sure the data represents valid pixel data,
 * according to the width, height, stride and format of the buffer object.
 *
 * \param bo The buffer object
 * \param buf The data to write
 * \param count The number of bytes to write
 * \return Returns 0 on success, otherwise -1 is returned an errno set
 */
GBM_EXPORT int
gbm_bo_write(struct gbm_bo *bo, const void *buf, size_t count)
{
   return bo->gbm->bo_write(bo, buf, count);
}

/** Set the user data associated with a buffer object
 *
 * \param bo The buffer object
 * \param data The data to associate to the buffer object
 * \param destroy_user_data A callback (which may be %NULL) that will be
 * called prior to the buffer destruction
 */
GBM_EXPORT void
gbm_bo_set_user_data(struct gbm_bo *bo, void *data,
		     void (*destroy_user_data)(struct gbm_bo *, void *))
{
   bo->user_data = data;
   bo->destroy_user_data = destroy_user_data;
}

/** Get the user data associated with a buffer object
 *
 * \param bo The buffer object
 * \return Returns the user data associated with the buffer object or %NULL
 * if no data was associated with it
 *
 * \sa gbm_bo_set_user_data()
 */
GBM_EXPORT void *
gbm_bo_get_user_data(struct gbm_bo *bo)
{
   return bo->user_data;
}

/**
 * Destroys the given buffer object and frees all resources associated with
 * it.
 *
 * \param bo The buffer object
 */
GBM_EXPORT void
gbm_bo_destroy(struct gbm_bo *bo)
{
   if (bo->destroy_user_data)
      bo->destroy_user_data(bo, bo->user_data);

   bo->gbm->bo_destroy(bo);
}

/**
 * Allocate a buffer object for the given dimensions
 *
 * \param gbm The gbm device returned from gbm_create_device()
 * \param width The width for the buffer
 * \param height The height for the buffer
 * \param format The format to use for the buffer
 * \param usage The union of the usage flags for this buffer
 *
 * \return A newly allocated buffer that should be freed with gbm_bo_destroy()
 * when no longer needed. If an error occurs during allocation %NULL will be
 * returned and errno set.
 *
 * \sa enum gbm_bo_format for the list of formats
 * \sa enum gbm_bo_flags for the list of usage flags
 */
GBM_EXPORT struct gbm_bo *
gbm_bo_create(struct gbm_device *gbm,
              uint32_t width, uint32_t height,
              uint32_t format, uint32_t usage)
{
   if (width == 0 || height == 0) {
      errno = EINVAL;
      return NULL;
   }

   return gbm->bo_create(gbm, width, height, format, usage, NULL, 0);
}

GBM_EXPORT struct gbm_bo *
gbm_bo_create_with_modifiers(struct gbm_device *gbm,
                             uint32_t width, uint32_t height,
                             uint32_t format,
                             const uint64_t *modifiers,
                             const unsigned int count)
{
   if (width == 0 || height == 0) {
      errno = EINVAL;
      return NULL;
   }

   if ((count && !modifiers) || (modifiers && !count)) {
      errno = EINVAL;
      return NULL;
   }

   return gbm->bo_create(gbm, width, height, format, 0, modifiers, count);
}
/**
 * Create a gbm buffer object from an foreign object
 *
 * This function imports a foreign object and creates a new gbm bo for it.
 * This enabled using the foreign object with a display API such as KMS.
 * Currently three types of foreign objects are supported, indicated by the type
 * argument:
 *
 *   GBM_BO_IMPORT_WL_BUFFER
 *   GBM_BO_IMPORT_EGL_IMAGE
 *   GBM_BO_IMPORT_FD
 *
 * The gbm bo shares the underlying pixels but its life-time is
 * independent of the foreign object.
 *
 * \param gbm The gbm device returned from gbm_create_device()
 * \param type The type of object we're importing
 * \param buffer Pointer to the external object
 * \param usage The union of the usage flags for this buffer
 *
 * \return A newly allocated buffer object that should be freed with
 * gbm_bo_destroy() when no longer needed. On error, %NULL is returned
 * and errno is set.
 *
 * \sa enum gbm_bo_flags for the list of usage flags
 */
GBM_EXPORT struct gbm_bo *
gbm_bo_import(struct gbm_device *gbm,
              uint32_t type, void *buffer, uint32_t usage)
{
   return gbm->bo_import(gbm, type, buffer, usage);
}

/**
 * Map a region of a gbm buffer object for cpu access
 *
 * This function maps a region of a gbm bo for cpu read and/or write
 * access.
 *
 * \param bo The buffer object
 * \param x The X (top left origin) starting position of the mapped region for
 * the buffer
 * \param y The Y (top left origin) starting position of the mapped region for
 * the buffer
 * \param width The width of the mapped region for the buffer
 * \param height The height of the mapped region for the buffer
 * \param flags The union of the GBM_BO_TRANSFER_* flags for this buffer
 * \param stride Ptr for returned stride in bytes of the mapped region
 * \param map_data Returned opaque ptr for the mapped region
 *
 * \return Address of the mapped buffer that should be unmapped with
 * gbm_bo_unmap() when no longer needed. On error, %NULL is returned
 * and errno is set.
 *
 * \sa enum gbm_bo_transfer_flags for the list of flags
 */
GBM_EXPORT void *
gbm_bo_map(struct gbm_bo *bo,
              uint32_t x, uint32_t y,
              uint32_t width, uint32_t height,
              uint32_t flags, uint32_t *stride, void **map_data)
{
   if (!bo || width == 0 || height == 0 || !stride || !map_data) {
      errno = EINVAL;
      return NULL;
   }

   return bo->gbm->bo_map(bo, x, y, width, height,
                          flags, stride, map_data);
}

/**
 * Unmap a previously mapped region of a gbm buffer object
 *
 * This function unmaps a region of a gbm bo for cpu read and/or write
 * access.
 *
 * \param bo The buffer object
 * \param map_data opaque ptr returned from prior gbm_bo_map
 */
GBM_EXPORT void
gbm_bo_unmap(struct gbm_bo *bo, void *map_data)
{
   bo->gbm->bo_unmap(bo, map_data);
}

/**
 * Allocate a surface object
 *
 * \param gbm The gbm device returned from gbm_create_device()
 * \param width The width for the surface
 * \param height The height for the surface
 * \param format The format to use for the surface
 *
 * \return A newly allocated surface that should be freed with
 * gbm_surface_destroy() when no longer needed. If an error occurs
 * during allocation %NULL will be returned.
 *
 * \sa enum gbm_bo_format for the list of formats
 */
GBM_EXPORT struct gbm_surface *
gbm_surface_create(struct gbm_device *gbm,
                   uint32_t width, uint32_t height,
		   uint32_t format, uint32_t flags)
{
   return gbm->surface_create(gbm, width, height, format, flags, NULL, 0);
}

GBM_EXPORT struct gbm_surface *
gbm_surface_create_with_modifiers(struct gbm_device *gbm,
                                  uint32_t width, uint32_t height,
                                  uint32_t format,
                                  const uint64_t *modifiers,
                                  const unsigned int count)
{
   if ((count && !modifiers) || (modifiers && !count)) {
      errno = EINVAL;
      return NULL;
   }

   return gbm->surface_create(gbm, width, height, format, 0,
                              modifiers, count);
}

/**
 * Destroys the given surface and frees all resources associated with
 * it.
 *
 * All buffers locked with gbm_surface_lock_front_buffer() should be
 * released prior to calling this function.
 *
 * \param surf The surface
 */
GBM_EXPORT void
gbm_surface_destroy(struct gbm_surface *surf)
{
   surf->gbm->surface_destroy(surf);
}

/**
 * Lock the surface's current front buffer
 *
 * Lock rendering to the surface's current front buffer until it is
 * released with gbm_surface_release_buffer().
 *
 * This function must be called exactly once after calling
 * eglSwapBuffers.  Calling it before any eglSwapBuffer has happened
 * on the surface or two or more times after eglSwapBuffers is an
 * error.  A new bo representing the new front buffer is returned.  On
 * multiple invocations, all the returned bos must be released in
 * order to release the actual surface buffer.
 *
 * \param surf The surface
 *
 * \return A buffer object that should be released with
 * gbm_surface_release_buffer() when no longer needed.  The implementation
 * is free to reuse buffers released with gbm_surface_release_buffer() so
 * this bo should not be destroyed using gbm_bo_destroy().  If an error
 * occurs this function returns %NULL.
 */
GBM_EXPORT struct gbm_bo *
gbm_surface_lock_front_buffer(struct gbm_surface *surf)
{
   return surf->gbm->surface_lock_front_buffer(surf);
}

/**
 * Release a locked buffer obtained with gbm_surface_lock_front_buffer()
 *
 * Returns the underlying buffer to the gbm surface.  Releasing a bo
 * will typically make gbm_surface_has_free_buffer() return 1 and thus
 * allow rendering the next frame, but not always. The implementation
 * may choose to destroy the bo immediately or reuse it, in which case
 * the user data associated with it is unchanged.
 *
 * \param surf The surface
 * \param bo The buffer object
 */
GBM_EXPORT void
gbm_surface_release_buffer(struct gbm_surface *surf, struct gbm_bo *bo)
{
   surf->gbm->surface_release_buffer(surf, bo);
}

/**
 * Return whether or not a surface has free (non-locked) buffers
 *
 * Before starting a new frame, the surface must have a buffer
 * available for rendering.  Initially, a gbm surface will have a free
 * buffer, but after one or more buffers have been locked (\sa
 * gbm_surface_lock_front_buffer()), the application must check for a
 * free buffer before rendering.
 *
 * If a surface doesn't have a free buffer, the application must
 * return a buffer to the surface using gbm_surface_release_buffer()
 * and after that, the application can query for free buffers again.
 *
 * \param surf The surface
 * \return 1 if the surface has free buffers, 0 otherwise
 */
GBM_EXPORT int
gbm_surface_has_free_buffers(struct gbm_surface *surf)
{
   return surf->gbm->surface_has_free_buffers(surf);
}
