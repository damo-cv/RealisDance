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

#define NUM_THREADS 16
#define BLOCKS_PER_THREAD 1024
#define NUM_RUNS 64

struct job {
   pthread_t thread;
   unsigned id;
   struct anv_block_pool *pool;
   uint32_t blocks[BLOCKS_PER_THREAD];
   uint32_t back_blocks[BLOCKS_PER_THREAD];
} jobs[NUM_THREADS];


static void *alloc_blocks(void *_job)
{
   struct job *job = _job;
   uint32_t job_id = job - jobs;
   uint32_t block_size = 16 * ((job_id % 4) + 1);
   int32_t block, *data;

   for (unsigned i = 0; i < BLOCKS_PER_THREAD; i++) {
      block = anv_block_pool_alloc(job->pool, block_size);
      data = job->pool->map + block;
      *data = block;
      assert(block >= 0);
      job->blocks[i] = block;

      block = anv_block_pool_alloc_back(job->pool, block_size);
      data = job->pool->map + block;
      *data = block;
      assert(block < 0);
      job->back_blocks[i] = -block;
   }

   for (unsigned i = 0; i < BLOCKS_PER_THREAD; i++) {
      block = job->blocks[i];
      data = job->pool->map + block;
      assert(*data == block);

      block = -job->back_blocks[i];
      data = job->pool->map + block;
      assert(*data == block);
   }

   return NULL;
}

static void validate_monotonic(uint32_t **blocks)
{
   /* A list of indices, one per thread */
   unsigned next[NUM_THREADS];
   memset(next, 0, sizeof(next));

   int highest = -1;
   while (true) {
      /* First, we find which thread has the highest next element */
      int thread_max = -1;
      int max_thread_idx = -1;
      for (unsigned i = 0; i < NUM_THREADS; i++) {
         if (next[i] >= BLOCKS_PER_THREAD)
            continue;

         if (thread_max < blocks[i][next[i]]) {
            thread_max = blocks[i][next[i]];
            max_thread_idx = i;
         }
      }

      /* The only way this can happen is if all of the next[] values are at
       * BLOCKS_PER_THREAD, in which case, we're done.
       */
      if (thread_max == -1)
         break;

      /* That next element had better be higher than the previous highest */
      assert(blocks[max_thread_idx][next[max_thread_idx]] > highest);

      highest = blocks[max_thread_idx][next[max_thread_idx]];
      next[max_thread_idx]++;
   }
}

static void run_test()
{
   struct anv_instance instance;
   struct anv_device device = {
      .instance = &instance,
   };
   struct anv_block_pool pool;

   pthread_mutex_init(&device.mutex, NULL);
   anv_block_pool_init(&pool, &device, 4096, 4096, 0);

   for (unsigned i = 0; i < NUM_THREADS; i++) {
      jobs[i].pool = &pool;
      jobs[i].id = i;
      pthread_create(&jobs[i].thread, NULL, alloc_blocks, &jobs[i]);
   }

   for (unsigned i = 0; i < NUM_THREADS; i++)
      pthread_join(jobs[i].thread, NULL);

   /* Validate that the block allocations were monotonic */
   uint32_t *block_ptrs[NUM_THREADS];
   for (unsigned i = 0; i < NUM_THREADS; i++)
      block_ptrs[i] = jobs[i].blocks;
   validate_monotonic(block_ptrs);

   /* Validate that the back block allocations were monotonic */
   for (unsigned i = 0; i < NUM_THREADS; i++)
      block_ptrs[i] = jobs[i].back_blocks;
   validate_monotonic(block_ptrs);

   anv_block_pool_finish(&pool);
   pthread_mutex_destroy(&device.mutex);
}

int main(int argc, char **argv)
{
   for (unsigned i = 0; i < NUM_RUNS; i++)
      run_test();
}
