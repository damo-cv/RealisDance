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

#include <pthread.h>

#include "anv_private.h"

#define NUM_THREADS 8
#define STATES_PER_THREAD_LOG2 10
#define STATES_PER_THREAD (1 << STATES_PER_THREAD_LOG2)
#define NUM_RUNS 64

#include "state_pool_test_helper.h"

int main(int argc, char **argv)
{
   struct anv_instance instance;
   struct anv_device device = {
      .instance = &instance,
   };
   struct anv_state_pool state_pool;

   pthread_mutex_init(&device.mutex, NULL);

   for (unsigned i = 0; i < NUM_RUNS; i++) {
      anv_state_pool_init(&state_pool, &device, 4096, 256, 0);

      /* Grab one so a zero offset is impossible */
      anv_state_pool_alloc(&state_pool, 16, 16);

      run_state_pool_test(&state_pool);

      anv_state_pool_finish(&state_pool);
   }

   pthread_mutex_destroy(&device.mutex);
}
