/*
 * Copyright © 2013 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \file brw_vec4_gs.c
 *
 * State atom for client-programmable geometry shaders, and support code.
 */

#include "brw_gs.h"
#include "brw_context.h"
#include "brw_state.h"
#include "brw_ff_gs.h"
#include "compiler/brw_nir.h"
#include "brw_program.h"
#include "compiler/glsl/ir_uniform.h"

static void
brw_gs_debug_recompile(struct brw_context *brw, struct gl_program *prog,
                       const struct brw_gs_prog_key *key)
{
   perf_debug("Recompiling geometry shader for program %d\n", prog->Id);

   bool found = false;
   const struct brw_gs_prog_key *old_key =
      brw_find_previous_compile(&brw->cache, BRW_CACHE_GS_PROG,
                                key->program_string_id);

   if (!old_key) {
      perf_debug("  Didn't find previous compile in the shader cache for "
                 "debug\n");
      return;
   }

   found |= brw_debug_recompile_sampler_key(brw, &old_key->tex, &key->tex);

   if (!found) {
      perf_debug("  Something else\n");
   }
}

static void
assign_gs_binding_table_offsets(const struct gen_device_info *devinfo,
                                const struct gl_program *prog,
                                struct brw_gs_prog_data *prog_data)
{
   /* In gen6 we reserve the first BRW_MAX_SOL_BINDINGS entries for transform
    * feedback surfaces.
    */
   uint32_t reserved = devinfo->gen == 6 ? BRW_MAX_SOL_BINDINGS : 0;

   brw_assign_common_binding_table_offsets(devinfo, prog,
                                           &prog_data->base.base, reserved);
}

static bool
brw_codegen_gs_prog(struct brw_context *brw,
                    struct brw_program *gp,
                    struct brw_gs_prog_key *key)
{
   struct brw_compiler *compiler = brw->screen->compiler;
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   struct brw_stage_state *stage_state = &brw->gs.base;
   struct brw_gs_prog_data prog_data;
   bool start_busy = false;
   double start_time = 0;

   memset(&prog_data, 0, sizeof(prog_data));

   void *mem_ctx = ralloc_context(NULL);

   assign_gs_binding_table_offsets(devinfo, &gp->program, &prog_data);

   brw_nir_setup_glsl_uniforms(mem_ctx, gp->program.nir, &gp->program,
                               &prog_data.base.base,
                               compiler->scalar_stage[MESA_SHADER_GEOMETRY]);
   brw_nir_analyze_ubo_ranges(compiler, gp->program.nir, NULL,
                              prog_data.base.base.ubo_ranges);

   uint64_t outputs_written = gp->program.nir->info.outputs_written;

   brw_compute_vue_map(devinfo,
                       &prog_data.base.vue_map, outputs_written,
                       gp->program.info.separate_shader);

   int st_index = -1;
   if (INTEL_DEBUG & DEBUG_SHADER_TIME)
      st_index = brw_get_shader_time_index(brw, &gp->program, ST_GS, true);

   if (unlikely(brw->perf_debug)) {
      start_busy = brw->batch.last_bo && brw_bo_busy(brw->batch.last_bo);
      start_time = get_time();
   }

   char *error_str;
   const unsigned *program =
      brw_compile_gs(brw->screen->compiler, brw, mem_ctx, key,
                     &prog_data, gp->program.nir, &gp->program,
                     st_index, &error_str);
   if (program == NULL) {
      ralloc_strcat(&gp->program.sh.data->InfoLog, error_str);
      _mesa_problem(NULL, "Failed to compile geometry shader: %s\n", error_str);

      ralloc_free(mem_ctx);
      return false;
   }

   if (unlikely(brw->perf_debug)) {
      if (gp->compiled_once) {
         brw_gs_debug_recompile(brw, &gp->program, key);
      }
      if (start_busy && !brw_bo_busy(brw->batch.last_bo)) {
         perf_debug("GS compile took %.03f ms and stalled the GPU\n",
                    (get_time() - start_time) * 1000);
      }
      gp->compiled_once = true;
   }

   /* Scratch space is used for register spilling */
   brw_alloc_stage_scratch(brw, stage_state,
                           prog_data.base.base.total_scratch);

   /* The param and pull_param arrays will be freed by the shader cache. */
   ralloc_steal(NULL, prog_data.base.base.param);
   ralloc_steal(NULL, prog_data.base.base.pull_param);
   brw_upload_cache(&brw->cache, BRW_CACHE_GS_PROG,
                    key, sizeof(*key),
                    program, prog_data.base.base.program_size,
                    &prog_data, sizeof(prog_data),
                    &stage_state->prog_offset, &brw->gs.base.prog_data);
   ralloc_free(mem_ctx);

   return true;
}

static bool
brw_gs_state_dirty(const struct brw_context *brw)
{
   return brw_state_dirty(brw,
                          _NEW_TEXTURE,
                          BRW_NEW_GEOMETRY_PROGRAM |
                          BRW_NEW_TRANSFORM_FEEDBACK);
}

void
brw_gs_populate_key(struct brw_context *brw,
                    struct brw_gs_prog_key *key)
{
   struct gl_context *ctx = &brw->ctx;
   struct brw_program *gp =
      (struct brw_program *) brw->programs[MESA_SHADER_GEOMETRY];

   memset(key, 0, sizeof(*key));

   key->program_string_id = gp->id;

   /* _NEW_TEXTURE */
   brw_populate_sampler_prog_key_data(ctx, &gp->program, &key->tex);
}

void
brw_upload_gs_prog(struct brw_context *brw)
{
   struct brw_stage_state *stage_state = &brw->gs.base;
   struct brw_gs_prog_key key;
   /* BRW_NEW_GEOMETRY_PROGRAM */
   struct brw_program *gp =
      (struct brw_program *) brw->programs[MESA_SHADER_GEOMETRY];

   if (!brw_gs_state_dirty(brw))
      return;

   brw_gs_populate_key(brw, &key);

   if (brw_search_cache(&brw->cache, BRW_CACHE_GS_PROG, &key, sizeof(key),
                        &stage_state->prog_offset, &brw->gs.base.prog_data,
                        true))
      return;

   if (brw_disk_cache_upload_program(brw, MESA_SHADER_GEOMETRY))
      return;

   gp = (struct brw_program *) brw->programs[MESA_SHADER_GEOMETRY];
   gp->id = key.program_string_id;

   MAYBE_UNUSED bool success = brw_codegen_gs_prog(brw, gp, &key);
   assert(success);
}

void
brw_gs_populate_default_key(const struct gen_device_info *devinfo,
                            struct brw_gs_prog_key *key,
                            struct gl_program *prog)
{
   memset(key, 0, sizeof(*key));

   brw_setup_tex_for_precompile(devinfo, &key->tex, prog);
   key->program_string_id = brw_program(prog)->id;
}

bool
brw_gs_precompile(struct gl_context *ctx, struct gl_program *prog)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_gs_prog_key key;
   uint32_t old_prog_offset = brw->gs.base.prog_offset;
   struct brw_stage_prog_data *old_prog_data = brw->gs.base.prog_data;
   bool success;

   struct brw_program *bgp = brw_program(prog);

   brw_gs_populate_default_key(&brw->screen->devinfo, &key, prog);

   success = brw_codegen_gs_prog(brw, bgp, &key);

   brw->gs.base.prog_offset = old_prog_offset;
   brw->gs.base.prog_data = old_prog_data;

   return success;
}
