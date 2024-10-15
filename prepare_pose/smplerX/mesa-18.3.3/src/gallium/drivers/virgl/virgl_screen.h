/*
 * Copyright 2014, 2015 Red Hat.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef VIRGL_H
#define VIRGL_H

#include "pipe/p_screen.h"
#include "util/slab.h"
#include "virgl_winsys.h"

#define VIRGL_DEBUG_VERBOSE 1
#define VIRGL_DEBUG_TGSI    2
extern int virgl_debug;

struct virgl_screen {
   struct pipe_screen base;

   int refcnt;

   /* place for winsys to stash it's own stuff: */
   void *winsys_priv;

   struct virgl_winsys *vws;

   struct virgl_drm_caps caps;

   struct slab_parent_pool texture_transfer_pool;

   uint32_t sub_ctx_id;
};


static inline struct virgl_screen *
virgl_screen(struct pipe_screen *pipe)
{
   return (struct virgl_screen *)pipe;
}

#define VIRGL_MAP_BUFFER_ALIGNMENT 64

#endif
