/*
 * Copyright © 2017 Google, Inc.
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

#include <stdarg.h>

#ifdef ANDROID
#include <android/log.h>
#else
#include <stdio.h>
#endif

#include "intel_log.h"

#ifdef ANDROID
static inline android_LogPriority
level_to_android(enum intel_log_level l)
{
   switch (l) {
   case INTEL_LOG_ERROR: return ANDROID_LOG_ERROR;
   case INTEL_LOG_WARN: return ANDROID_LOG_WARN;
   case INTEL_LOG_INFO: return ANDROID_LOG_INFO;
   case INTEL_LOG_DEBUG: return ANDROID_LOG_DEBUG;
   }

   unreachable("bad intel_log_level");
}
#endif

#ifndef ANDROID
static inline const char *
level_to_str(enum intel_log_level l)
{
   switch (l) {
   case INTEL_LOG_ERROR: return "error";
   case INTEL_LOG_WARN: return "warning";
   case INTEL_LOG_INFO: return "info";
   case INTEL_LOG_DEBUG: return "debug";
   }

   unreachable("bad intel_log_level");
}
#endif

void
intel_log(enum intel_log_level level, const char *tag, const char *format, ...)
{
   va_list va;

   va_start(va, format);
   intel_log_v(level, tag, format, va);
   va_end(va);
}

void
intel_log_v(enum intel_log_level level, const char *tag, const char *format,
            va_list va)
{
#ifdef ANDROID
   __android_log_vprint(level_to_android(level), tag, format, va);
#else
   flockfile(stderr);
   fprintf(stderr, "%s: %s: ", tag, level_to_str(level));
   vfprintf(stderr, format, va);
   fprintf(stderr, "\n");
   funlockfile(stderr);
#endif
}
