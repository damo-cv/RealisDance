/*
 * Copyright © 2016 Intel Corporation
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

/*
 * NOTE: The header can be included multiple times, from the same file.
 */

/*
 * Gen-specific function declarations.  This header must *not* be included
 * directly.  Instead, it is included multiple times by anv_private.h.
 * 
 * In this header file, the usual genx() macro is available.
 */

#ifndef ANV_PRIVATE_H
#error This file is included by means other than anv_private.h
#endif

VkResult genX(init_device_state)(struct anv_device *device);

void genX(cmd_buffer_emit_state_base_address)(struct anv_cmd_buffer *cmd_buffer);

void genX(cmd_buffer_apply_pipe_flushes)(struct anv_cmd_buffer *cmd_buffer);

void genX(cmd_buffer_emit_gen7_depth_flush)(struct anv_cmd_buffer *cmd_buffer);

void genX(flush_pipeline_select_3d)(struct anv_cmd_buffer *cmd_buffer);
void genX(flush_pipeline_select_gpgpu)(struct anv_cmd_buffer *cmd_buffer);

void genX(cmd_buffer_config_l3)(struct anv_cmd_buffer *cmd_buffer,
                                const struct gen_l3_config *cfg);

void genX(cmd_buffer_flush_state)(struct anv_cmd_buffer *cmd_buffer);
void genX(cmd_buffer_flush_dynamic_state)(struct anv_cmd_buffer *cmd_buffer);

void genX(cmd_buffer_flush_compute_state)(struct anv_cmd_buffer *cmd_buffer);

void genX(cmd_buffer_enable_pma_fix)(struct anv_cmd_buffer *cmd_buffer,
                                     bool enable);

void genX(cmd_buffer_mark_image_written)(struct anv_cmd_buffer *cmd_buffer,
                                         const struct anv_image *image,
                                         VkImageAspectFlagBits aspect,
                                         enum isl_aux_usage aux_usage,
                                         uint32_t level,
                                         uint32_t base_layer,
                                         uint32_t layer_count);

void
genX(emit_urb_setup)(struct anv_device *device, struct anv_batch *batch,
                     const struct gen_l3_config *l3_config,
                     VkShaderStageFlags active_stages,
                     const unsigned entry_size[4]);

void genX(cmd_buffer_so_memcpy)(struct anv_cmd_buffer *cmd_buffer,
                                struct anv_address dst, struct anv_address src,
                                uint32_t size);

void genX(cmd_buffer_mi_memcpy)(struct anv_cmd_buffer *cmd_buffer,
                                struct anv_address dst, struct anv_address src,
                                uint32_t size);

void genX(cmd_buffer_mi_memset)(struct anv_cmd_buffer *cmd_buffer,
                                struct anv_address dst, uint32_t value,
                                uint32_t size);

void genX(blorp_exec)(struct blorp_batch *batch,
                      const struct blorp_params *params);
