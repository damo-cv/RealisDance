/*
 * Copyright 2013 Advanced Micro Devices, Inc.
 * All Rights Reserved.
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

/* Resource binding slots and sampler states (each described with 8 or
 * 4 dwords) are stored in lists in memory which is accessed by shaders
 * using scalar load instructions.
 *
 * This file is responsible for managing such lists. It keeps a copy of all
 * descriptors in CPU memory and re-uploads a whole list if some slots have
 * been changed.
 *
 * This code is also reponsible for updating shader pointers to those lists.
 *
 * Note that CP DMA can't be used for updating the lists, because a GPU hang
 * could leave the list in a mid-IB state and the next IB would get wrong
 * descriptors and the whole context would be unusable at that point.
 * (Note: The register shadowing can't be used due to the same reason)
 *
 * Also, uploading descriptors to newly allocated memory doesn't require
 * a KCACHE flush.
 *
 *
 * Possible scenarios for one 16 dword image+sampler slot:
 *
 *       | Image        | w/ FMASK   | Buffer       | NULL
 * [ 0: 3] Image[0:3]   | Image[0:3] | Null[0:3]    | Null[0:3]
 * [ 4: 7] Image[4:7]   | Image[4:7] | Buffer[0:3]  | 0
 * [ 8:11] Null[0:3]    | Fmask[0:3] | Null[0:3]    | Null[0:3]
 * [12:15] Sampler[0:3] | Fmask[4:7] | Sampler[0:3] | Sampler[0:3]
 *
 * FMASK implies MSAA, therefore no sampler state.
 * Sampler states are never unbound except when FMASK is bound.
 */

#include "si_pipe.h"
#include "sid.h"
#include "gfx9d.h"

#include "util/hash_table.h"
#include "util/u_idalloc.h"
#include "util/u_format.h"
#include "util/u_memory.h"
#include "util/u_upload_mgr.h"


/* NULL image and buffer descriptor for textures (alpha = 1) and images
 * (alpha = 0).
 *
 * For images, all fields must be zero except for the swizzle, which
 * supports arbitrary combinations of 0s and 1s. The texture type must be
 * any valid type (e.g. 1D). If the texture type isn't set, the hw hangs.
 *
 * For buffers, all fields must be zero. If they are not, the hw hangs.
 *
 * This is the only reason why the buffer descriptor must be in words [4:7].
 */
static uint32_t null_texture_descriptor[8] = {
	0,
	0,
	0,
	S_008F1C_DST_SEL_W(V_008F1C_SQ_SEL_1) |
	S_008F1C_TYPE(V_008F1C_SQ_RSRC_IMG_1D)
	/* the rest must contain zeros, which is also used by the buffer
	 * descriptor */
};

static uint32_t null_image_descriptor[8] = {
	0,
	0,
	0,
	S_008F1C_TYPE(V_008F1C_SQ_RSRC_IMG_1D)
	/* the rest must contain zeros, which is also used by the buffer
	 * descriptor */
};

static uint64_t si_desc_extract_buffer_address(const uint32_t *desc)
{
	uint64_t va = desc[0] |
		      ((uint64_t)G_008F04_BASE_ADDRESS_HI(desc[1]) << 32);

	/* Sign-extend the 48-bit address. */
	va <<= 16;
	va = (int64_t)va >> 16;
	return va;
}

static void si_init_descriptor_list(uint32_t *desc_list,
				    unsigned element_dw_size,
				    unsigned num_elements,
				    const uint32_t *null_descriptor)
{
	int i;

	/* Initialize the array to NULL descriptors if the element size is 8. */
	if (null_descriptor) {
		assert(element_dw_size % 8 == 0);
		for (i = 0; i < num_elements * element_dw_size / 8; i++)
			memcpy(desc_list + i * 8, null_descriptor, 8 * 4);
	}
}

static void si_init_descriptors(struct si_descriptors *desc,
				short shader_userdata_rel_index,
				unsigned element_dw_size,
				unsigned num_elements)
{
	desc->list = CALLOC(num_elements, element_dw_size * 4);
	desc->element_dw_size = element_dw_size;
	desc->num_elements = num_elements;
	desc->shader_userdata_offset = shader_userdata_rel_index * 4;
	desc->slot_index_to_bind_directly = -1;
}

static void si_release_descriptors(struct si_descriptors *desc)
{
	r600_resource_reference(&desc->buffer, NULL);
	FREE(desc->list);
}

static bool si_upload_descriptors(struct si_context *sctx,
				  struct si_descriptors *desc)
{
	unsigned slot_size = desc->element_dw_size * 4;
	unsigned first_slot_offset = desc->first_active_slot * slot_size;
	unsigned upload_size = desc->num_active_slots * slot_size;

	/* Skip the upload if no shader is using the descriptors. dirty_mask
	 * will stay dirty and the descriptors will be uploaded when there is
	 * a shader using them.
	 */
	if (!upload_size)
		return true;

	/* If there is just one active descriptor, bind it directly. */
	if ((int)desc->first_active_slot == desc->slot_index_to_bind_directly &&
	    desc->num_active_slots == 1) {
		uint32_t *descriptor = &desc->list[desc->slot_index_to_bind_directly *
						   desc->element_dw_size];

		/* The buffer is already in the buffer list. */
		r600_resource_reference(&desc->buffer, NULL);
		desc->gpu_list = NULL;
		desc->gpu_address = si_desc_extract_buffer_address(descriptor);
		si_mark_atom_dirty(sctx, &sctx->atoms.s.shader_pointers);
		return true;
	}

	uint32_t *ptr;
	unsigned buffer_offset;
	u_upload_alloc(sctx->b.const_uploader, first_slot_offset, upload_size,
		       si_optimal_tcc_alignment(sctx, upload_size),
		       &buffer_offset, (struct pipe_resource**)&desc->buffer,
		       (void**)&ptr);
	if (!desc->buffer) {
		desc->gpu_address = 0;
		return false; /* skip the draw call */
	}

	util_memcpy_cpu_to_le32(ptr, (char*)desc->list + first_slot_offset,
				upload_size);
	desc->gpu_list = ptr - first_slot_offset / 4;

	radeon_add_to_buffer_list(sctx, sctx->gfx_cs, desc->buffer,
                            RADEON_USAGE_READ, RADEON_PRIO_DESCRIPTORS);

	/* The shader pointer should point to slot 0. */
	buffer_offset -= first_slot_offset;
	desc->gpu_address = desc->buffer->gpu_address + buffer_offset;

	assert(desc->buffer->flags & RADEON_FLAG_32BIT);
	assert((desc->buffer->gpu_address >> 32) == sctx->screen->info.address32_hi);
	assert((desc->gpu_address >> 32) == sctx->screen->info.address32_hi);

	si_mark_atom_dirty(sctx, &sctx->atoms.s.shader_pointers);
	return true;
}

static void
si_descriptors_begin_new_cs(struct si_context *sctx, struct si_descriptors *desc)
{
	if (!desc->buffer)
		return;

	radeon_add_to_buffer_list(sctx, sctx->gfx_cs, desc->buffer,
				  RADEON_USAGE_READ, RADEON_PRIO_DESCRIPTORS);
}

/* SAMPLER VIEWS */

static inline enum radeon_bo_priority
si_get_sampler_view_priority(struct r600_resource *res)
{
	if (res->b.b.target == PIPE_BUFFER)
		return RADEON_PRIO_SAMPLER_BUFFER;

	if (res->b.b.nr_samples > 1)
		return RADEON_PRIO_SAMPLER_TEXTURE_MSAA;

	return RADEON_PRIO_SAMPLER_TEXTURE;
}

static unsigned
si_sampler_and_image_descriptors_idx(unsigned shader)
{
	return SI_DESCS_FIRST_SHADER + shader * SI_NUM_SHADER_DESCS +
	       SI_SHADER_DESCS_SAMPLERS_AND_IMAGES;
}

static struct si_descriptors *
si_sampler_and_image_descriptors(struct si_context *sctx, unsigned shader)
{
	return &sctx->descriptors[si_sampler_and_image_descriptors_idx(shader)];
}

static void si_release_sampler_views(struct si_samplers *samplers)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(samplers->views); i++) {
		pipe_sampler_view_reference(&samplers->views[i], NULL);
	}
}

static void si_sampler_view_add_buffer(struct si_context *sctx,
				       struct pipe_resource *resource,
				       enum radeon_bo_usage usage,
				       bool is_stencil_sampler,
				       bool check_mem)
{
	struct si_texture *tex = (struct si_texture*)resource;
	enum radeon_bo_priority priority;

	if (!resource)
		return;

	/* Use the flushed depth texture if direct sampling is unsupported. */
	if (resource->target != PIPE_BUFFER &&
	    tex->is_depth && !si_can_sample_zs(tex, is_stencil_sampler))
		tex = tex->flushed_depth_texture;

	priority = si_get_sampler_view_priority(&tex->buffer);
	radeon_add_to_gfx_buffer_list_check_mem(sctx, &tex->buffer, usage, priority,
						check_mem);

	if (resource->target == PIPE_BUFFER)
		return;

	/* Add separate DCC. */
	if (tex->dcc_separate_buffer) {
		radeon_add_to_gfx_buffer_list_check_mem(sctx, tex->dcc_separate_buffer,
							usage, RADEON_PRIO_SEPARATE_META, check_mem);
	}
}

static void si_sampler_views_begin_new_cs(struct si_context *sctx,
					  struct si_samplers *samplers)
{
	unsigned mask = samplers->enabled_mask;

	/* Add buffers to the CS. */
	while (mask) {
		int i = u_bit_scan(&mask);
		struct si_sampler_view *sview = (struct si_sampler_view *)samplers->views[i];

		si_sampler_view_add_buffer(sctx, sview->base.texture,
					   RADEON_USAGE_READ,
					   sview->is_stencil_sampler, false);
	}
}

/* Set buffer descriptor fields that can be changed by reallocations. */
static void si_set_buf_desc_address(struct r600_resource *buf,
				    uint64_t offset, uint32_t *state)
{
	uint64_t va = buf->gpu_address + offset;

	state[0] = va;
	state[1] &= C_008F04_BASE_ADDRESS_HI;
	state[1] |= S_008F04_BASE_ADDRESS_HI(va >> 32);
}

/* Set texture descriptor fields that can be changed by reallocations.
 *
 * \param tex			texture
 * \param base_level_info	information of the level of BASE_ADDRESS
 * \param base_level		the level of BASE_ADDRESS
 * \param first_level		pipe_sampler_view.u.tex.first_level
 * \param block_width		util_format_get_blockwidth()
 * \param is_stencil		select between separate Z & Stencil
 * \param state			descriptor to update
 */
void si_set_mutable_tex_desc_fields(struct si_screen *sscreen,
				    struct si_texture *tex,
				    const struct legacy_surf_level *base_level_info,
				    unsigned base_level, unsigned first_level,
				    unsigned block_width, bool is_stencil,
				    uint32_t *state)
{
	uint64_t va, meta_va = 0;

	if (tex->is_depth && !si_can_sample_zs(tex, is_stencil)) {
		tex = tex->flushed_depth_texture;
		is_stencil = false;
	}

	va = tex->buffer.gpu_address;

	if (sscreen->info.chip_class >= GFX9) {
		/* Only stencil_offset needs to be added here. */
		if (is_stencil)
			va += tex->surface.u.gfx9.stencil_offset;
		else
			va += tex->surface.u.gfx9.surf_offset;
	} else {
		va += base_level_info->offset;
	}

	state[0] = va >> 8;
	state[1] &= C_008F14_BASE_ADDRESS_HI;
	state[1] |= S_008F14_BASE_ADDRESS_HI(va >> 40);

	/* Only macrotiled modes can set tile swizzle.
	 * GFX9 doesn't use (legacy) base_level_info.
	 */
	if (sscreen->info.chip_class >= GFX9 ||
	    base_level_info->mode == RADEON_SURF_MODE_2D)
		state[0] |= tex->surface.tile_swizzle;

	if (sscreen->info.chip_class >= VI) {
		state[6] &= C_008F28_COMPRESSION_EN;
		state[7] = 0;

		if (vi_dcc_enabled(tex, first_level)) {
			meta_va = (!tex->dcc_separate_buffer ? tex->buffer.gpu_address : 0) +
				  tex->dcc_offset;

			if (sscreen->info.chip_class == VI) {
				meta_va += base_level_info->dcc_offset;
				assert(base_level_info->mode == RADEON_SURF_MODE_2D);
			}

			meta_va |= (uint32_t)tex->surface.tile_swizzle << 8;
		} else if (vi_tc_compat_htile_enabled(tex, first_level)) {
			meta_va = tex->buffer.gpu_address + tex->htile_offset;
		}

		if (meta_va) {
			state[6] |= S_008F28_COMPRESSION_EN(1);
			state[7] = meta_va >> 8;
		}
	}

	if (sscreen->info.chip_class >= GFX9) {
		state[3] &= C_008F1C_SW_MODE;
		state[4] &= C_008F20_PITCH_GFX9;

		if (is_stencil) {
			state[3] |= S_008F1C_SW_MODE(tex->surface.u.gfx9.stencil.swizzle_mode);
			state[4] |= S_008F20_PITCH_GFX9(tex->surface.u.gfx9.stencil.epitch);
		} else {
			state[3] |= S_008F1C_SW_MODE(tex->surface.u.gfx9.surf.swizzle_mode);
			state[4] |= S_008F20_PITCH_GFX9(tex->surface.u.gfx9.surf.epitch);
		}

		state[5] &= C_008F24_META_DATA_ADDRESS &
			    C_008F24_META_PIPE_ALIGNED &
			    C_008F24_META_RB_ALIGNED;
		if (meta_va) {
			struct gfx9_surf_meta_flags meta;

			if (tex->dcc_offset)
				meta = tex->surface.u.gfx9.dcc;
			else
				meta = tex->surface.u.gfx9.htile;

			state[5] |= S_008F24_META_DATA_ADDRESS(meta_va >> 40) |
				    S_008F24_META_PIPE_ALIGNED(meta.pipe_aligned) |
				    S_008F24_META_RB_ALIGNED(meta.rb_aligned);
		}
	} else {
		/* SI-CI-VI */
		unsigned pitch = base_level_info->nblk_x * block_width;
		unsigned index = si_tile_mode_index(tex, base_level, is_stencil);

		state[3] &= C_008F1C_TILING_INDEX;
		state[3] |= S_008F1C_TILING_INDEX(index);
		state[4] &= C_008F20_PITCH_GFX6;
		state[4] |= S_008F20_PITCH_GFX6(pitch - 1);
	}
}

static void si_set_sampler_state_desc(struct si_sampler_state *sstate,
				      struct si_sampler_view *sview,
				      struct si_texture *tex,
				      uint32_t *desc)
{
	if (sview && sview->is_integer)
		memcpy(desc, sstate->integer_val, 4*4);
	else if (tex && tex->upgraded_depth &&
		 (!sview || !sview->is_stencil_sampler))
		memcpy(desc, sstate->upgraded_depth_val, 4*4);
	else
		memcpy(desc, sstate->val, 4*4);
}

static void si_set_sampler_view_desc(struct si_context *sctx,
				     struct si_sampler_view *sview,
				     struct si_sampler_state *sstate,
				     uint32_t *desc)
{
	struct pipe_sampler_view *view = &sview->base;
	struct si_texture *tex = (struct si_texture *)view->texture;
	bool is_buffer = tex->buffer.b.b.target == PIPE_BUFFER;

	if (unlikely(!is_buffer && sview->dcc_incompatible)) {
		if (vi_dcc_enabled(tex, view->u.tex.first_level))
			if (!si_texture_disable_dcc(sctx, tex))
				si_decompress_dcc(sctx, tex);

		sview->dcc_incompatible = false;
	}

	assert(tex); /* views with texture == NULL aren't supported */
	memcpy(desc, sview->state, 8*4);

	if (is_buffer) {
		si_set_buf_desc_address(&tex->buffer,
					sview->base.u.buf.offset,
					desc + 4);
	} else {
		bool is_separate_stencil = tex->db_compatible &&
					   sview->is_stencil_sampler;

		si_set_mutable_tex_desc_fields(sctx->screen, tex,
					       sview->base_level_info,
					       sview->base_level,
					       sview->base.u.tex.first_level,
					       sview->block_width,
					       is_separate_stencil,
					       desc);
	}

	if (!is_buffer && tex->surface.fmask_size) {
		memcpy(desc + 8, sview->fmask_state, 8*4);
	} else {
		/* Disable FMASK and bind sampler state in [12:15]. */
		memcpy(desc + 8, null_texture_descriptor, 4*4);

		if (sstate)
			si_set_sampler_state_desc(sstate, sview,
						  is_buffer ? NULL : tex,
						  desc + 12);
	}
}

static bool color_needs_decompression(struct si_texture *tex)
{
	return tex->surface.fmask_size ||
	       (tex->dirty_level_mask &&
		(tex->cmask_buffer || tex->dcc_offset));
}

static bool depth_needs_decompression(struct si_texture *tex)
{
	/* If the depth/stencil texture is TC-compatible, no decompression
	 * will be done. The decompression function will only flush DB caches
	 * to make it coherent with shaders. That's necessary because the driver
	 * doesn't flush DB caches in any other case.
	 */
	return tex->db_compatible;
}

static void si_set_sampler_view(struct si_context *sctx,
				unsigned shader,
				unsigned slot, struct pipe_sampler_view *view,
				bool disallow_early_out)
{
	struct si_samplers *samplers = &sctx->samplers[shader];
	struct si_sampler_view *rview = (struct si_sampler_view*)view;
	struct si_descriptors *descs = si_sampler_and_image_descriptors(sctx, shader);
	unsigned desc_slot = si_get_sampler_slot(slot);
	uint32_t *desc = descs->list + desc_slot * 16;

	if (samplers->views[slot] == view && !disallow_early_out)
		return;

	if (view) {
		struct si_texture *tex = (struct si_texture *)view->texture;

		si_set_sampler_view_desc(sctx, rview,
					 samplers->sampler_states[slot], desc);

		if (tex->buffer.b.b.target == PIPE_BUFFER) {
			tex->buffer.bind_history |= PIPE_BIND_SAMPLER_VIEW;
			samplers->needs_depth_decompress_mask &= ~(1u << slot);
			samplers->needs_color_decompress_mask &= ~(1u << slot);
		} else {
			if (depth_needs_decompression(tex)) {
				samplers->needs_depth_decompress_mask |= 1u << slot;
			} else {
				samplers->needs_depth_decompress_mask &= ~(1u << slot);
			}
			if (color_needs_decompression(tex)) {
				samplers->needs_color_decompress_mask |= 1u << slot;
			} else {
				samplers->needs_color_decompress_mask &= ~(1u << slot);
			}

			if (tex->dcc_offset &&
			    p_atomic_read(&tex->framebuffers_bound))
				sctx->need_check_render_feedback = true;
		}

		pipe_sampler_view_reference(&samplers->views[slot], view);
		samplers->enabled_mask |= 1u << slot;

		/* Since this can flush, it must be done after enabled_mask is
		 * updated. */
		si_sampler_view_add_buffer(sctx, view->texture,
					   RADEON_USAGE_READ,
					   rview->is_stencil_sampler, true);
	} else {
		pipe_sampler_view_reference(&samplers->views[slot], NULL);
		memcpy(desc, null_texture_descriptor, 8*4);
		/* Only clear the lower dwords of FMASK. */
		memcpy(desc + 8, null_texture_descriptor, 4*4);
		/* Re-set the sampler state if we are transitioning from FMASK. */
		if (samplers->sampler_states[slot])
			si_set_sampler_state_desc(samplers->sampler_states[slot], NULL, NULL,
						  desc + 12);

		samplers->enabled_mask &= ~(1u << slot);
		samplers->needs_depth_decompress_mask &= ~(1u << slot);
		samplers->needs_color_decompress_mask &= ~(1u << slot);
	}

	sctx->descriptors_dirty |= 1u << si_sampler_and_image_descriptors_idx(shader);
}

static void si_update_shader_needs_decompress_mask(struct si_context *sctx,
						   unsigned shader)
{
	struct si_samplers *samplers = &sctx->samplers[shader];
	unsigned shader_bit = 1 << shader;

	if (samplers->needs_depth_decompress_mask ||
	    samplers->needs_color_decompress_mask ||
	    sctx->images[shader].needs_color_decompress_mask)
		sctx->shader_needs_decompress_mask |= shader_bit;
	else
		sctx->shader_needs_decompress_mask &= ~shader_bit;
}

static void si_set_sampler_views(struct pipe_context *ctx,
				 enum pipe_shader_type shader, unsigned start,
                                 unsigned count,
				 struct pipe_sampler_view **views)
{
	struct si_context *sctx = (struct si_context *)ctx;
	int i;

	if (!count || shader >= SI_NUM_SHADERS)
		return;

	if (views) {
		for (i = 0; i < count; i++)
			si_set_sampler_view(sctx, shader, start + i, views[i], false);
	} else {
		for (i = 0; i < count; i++)
			si_set_sampler_view(sctx, shader, start + i, NULL, false);
	}

	si_update_shader_needs_decompress_mask(sctx, shader);
}

static void
si_samplers_update_needs_color_decompress_mask(struct si_samplers *samplers)
{
	unsigned mask = samplers->enabled_mask;

	while (mask) {
		int i = u_bit_scan(&mask);
		struct pipe_resource *res = samplers->views[i]->texture;

		if (res && res->target != PIPE_BUFFER) {
			struct si_texture *tex = (struct si_texture *)res;

			if (color_needs_decompression(tex)) {
				samplers->needs_color_decompress_mask |= 1u << i;
			} else {
				samplers->needs_color_decompress_mask &= ~(1u << i);
			}
		}
	}
}

/* IMAGE VIEWS */

static void
si_release_image_views(struct si_images *images)
{
	unsigned i;

	for (i = 0; i < SI_NUM_IMAGES; ++i) {
		struct pipe_image_view *view = &images->views[i];

		pipe_resource_reference(&view->resource, NULL);
	}
}

static void
si_image_views_begin_new_cs(struct si_context *sctx, struct si_images *images)
{
	uint mask = images->enabled_mask;

	/* Add buffers to the CS. */
	while (mask) {
		int i = u_bit_scan(&mask);
		struct pipe_image_view *view = &images->views[i];

		assert(view->resource);

		si_sampler_view_add_buffer(sctx, view->resource,
					   RADEON_USAGE_READWRITE, false, false);
	}
}

static void
si_disable_shader_image(struct si_context *ctx, unsigned shader, unsigned slot)
{
	struct si_images *images = &ctx->images[shader];

	if (images->enabled_mask & (1u << slot)) {
		struct si_descriptors *descs = si_sampler_and_image_descriptors(ctx, shader);
		unsigned desc_slot = si_get_image_slot(slot);

		pipe_resource_reference(&images->views[slot].resource, NULL);
		images->needs_color_decompress_mask &= ~(1 << slot);

		memcpy(descs->list + desc_slot*8, null_image_descriptor, 8*4);
		images->enabled_mask &= ~(1u << slot);
		ctx->descriptors_dirty |= 1u << si_sampler_and_image_descriptors_idx(shader);
	}
}

static void
si_mark_image_range_valid(const struct pipe_image_view *view)
{
	struct r600_resource *res = r600_resource(view->resource);

	assert(res && res->b.b.target == PIPE_BUFFER);

	util_range_add(&res->valid_buffer_range,
		       view->u.buf.offset,
		       view->u.buf.offset + view->u.buf.size);
}

static void si_set_shader_image_desc(struct si_context *ctx,
				     const struct pipe_image_view *view,
				     bool skip_decompress,
				     uint32_t *desc, uint32_t *fmask_desc)
{
	struct si_screen *screen = ctx->screen;
	struct r600_resource *res;

	res = r600_resource(view->resource);

	if (res->b.b.target == PIPE_BUFFER) {
		if (view->access & PIPE_IMAGE_ACCESS_WRITE)
			si_mark_image_range_valid(view);

		si_make_buffer_descriptor(screen, res,
					  view->format,
					  view->u.buf.offset,
					  view->u.buf.size, desc);
		si_set_buf_desc_address(res, view->u.buf.offset, desc + 4);
	} else {
		static const unsigned char swizzle[4] = { 0, 1, 2, 3 };
		struct si_texture *tex = (struct si_texture *)res;
		unsigned level = view->u.tex.level;
		unsigned width, height, depth, hw_level;
		bool uses_dcc = vi_dcc_enabled(tex, level);
		unsigned access = view->access;

		/* Clear the write flag when writes can't occur.
		 * Note that DCC_DECOMPRESS for MSAA doesn't work in some cases,
		 * so we don't wanna trigger it.
		 */
		if (tex->is_depth ||
		    (!fmask_desc && tex->surface.fmask_size != 0)) {
			assert(!"Z/S and MSAA image stores are not supported");
			access &= ~PIPE_IMAGE_ACCESS_WRITE;
		}

		assert(!tex->is_depth);
		assert(fmask_desc || tex->surface.fmask_size == 0);

		if (uses_dcc && !skip_decompress &&
		    (view->access & PIPE_IMAGE_ACCESS_WRITE ||
		     !vi_dcc_formats_compatible(res->b.b.format, view->format))) {
			/* If DCC can't be disabled, at least decompress it.
			 * The decompression is relatively cheap if the surface
			 * has been decompressed already.
			 */
			if (!si_texture_disable_dcc(ctx, tex))
				si_decompress_dcc(ctx, tex);
		}

		if (ctx->chip_class >= GFX9) {
			/* Always set the base address. The swizzle modes don't
			 * allow setting mipmap level offsets as the base.
			 */
			width = res->b.b.width0;
			height = res->b.b.height0;
			depth = res->b.b.depth0;
			hw_level = level;
		} else {
			/* Always force the base level to the selected level.
			 *
			 * This is required for 3D textures, where otherwise
			 * selecting a single slice for non-layered bindings
			 * fails. It doesn't hurt the other targets.
			 */
			width = u_minify(res->b.b.width0, level);
			height = u_minify(res->b.b.height0, level);
			depth = u_minify(res->b.b.depth0, level);
			hw_level = 0;
		}

		si_make_texture_descriptor(screen, tex,
					   false, res->b.b.target,
					   view->format, swizzle,
					   hw_level, hw_level,
					   view->u.tex.first_layer,
					   view->u.tex.last_layer,
					   width, height, depth,
					   desc, fmask_desc);
		si_set_mutable_tex_desc_fields(screen, tex,
					       &tex->surface.u.legacy.level[level],
					       level, level,
					       util_format_get_blockwidth(view->format),
					       false, desc);
	}
}

static void si_set_shader_image(struct si_context *ctx,
				unsigned shader,
				unsigned slot, const struct pipe_image_view *view,
				bool skip_decompress)
{
	struct si_images *images = &ctx->images[shader];
	struct si_descriptors *descs = si_sampler_and_image_descriptors(ctx, shader);
	struct r600_resource *res;
	unsigned desc_slot = si_get_image_slot(slot);
	uint32_t *desc = descs->list + desc_slot * 8;

	if (!view || !view->resource) {
		si_disable_shader_image(ctx, shader, slot);
		return;
	}

	res = r600_resource(view->resource);

	if (&images->views[slot] != view)
		util_copy_image_view(&images->views[slot], view);

	si_set_shader_image_desc(ctx, view, skip_decompress, desc, NULL);

	if (res->b.b.target == PIPE_BUFFER) {
		images->needs_color_decompress_mask &= ~(1 << slot);
		res->bind_history |= PIPE_BIND_SHADER_IMAGE;
	} else {
		struct si_texture *tex = (struct si_texture *)res;
		unsigned level = view->u.tex.level;

		if (color_needs_decompression(tex)) {
			images->needs_color_decompress_mask |= 1 << slot;
		} else {
			images->needs_color_decompress_mask &= ~(1 << slot);
		}

		if (vi_dcc_enabled(tex, level) &&
		    p_atomic_read(&tex->framebuffers_bound))
			ctx->need_check_render_feedback = true;
	}

	images->enabled_mask |= 1u << slot;
	ctx->descriptors_dirty |= 1u << si_sampler_and_image_descriptors_idx(shader);

	/* Since this can flush, it must be done after enabled_mask is updated. */
	si_sampler_view_add_buffer(ctx, &res->b.b,
				   (view->access & PIPE_IMAGE_ACCESS_WRITE) ?
				   RADEON_USAGE_READWRITE : RADEON_USAGE_READ,
				   false, true);
}

static void
si_set_shader_images(struct pipe_context *pipe,
		     enum pipe_shader_type shader,
		     unsigned start_slot, unsigned count,
		     const struct pipe_image_view *views)
{
	struct si_context *ctx = (struct si_context *)pipe;
	unsigned i, slot;

	assert(shader < SI_NUM_SHADERS);

	if (!count)
		return;

	assert(start_slot + count <= SI_NUM_IMAGES);

	if (views) {
		for (i = 0, slot = start_slot; i < count; ++i, ++slot)
			si_set_shader_image(ctx, shader, slot, &views[i], false);
	} else {
		for (i = 0, slot = start_slot; i < count; ++i, ++slot)
			si_set_shader_image(ctx, shader, slot, NULL, false);
	}

	si_update_shader_needs_decompress_mask(ctx, shader);
}

static void
si_images_update_needs_color_decompress_mask(struct si_images *images)
{
	unsigned mask = images->enabled_mask;

	while (mask) {
		int i = u_bit_scan(&mask);
		struct pipe_resource *res = images->views[i].resource;

		if (res && res->target != PIPE_BUFFER) {
			struct si_texture *tex = (struct si_texture *)res;

			if (color_needs_decompression(tex)) {
				images->needs_color_decompress_mask |= 1 << i;
			} else {
				images->needs_color_decompress_mask &= ~(1 << i);
			}
		}
	}
}

void si_update_ps_colorbuf0_slot(struct si_context *sctx)
{
	struct si_buffer_resources *buffers = &sctx->rw_buffers;
	struct si_descriptors *descs = &sctx->descriptors[SI_DESCS_RW_BUFFERS];
	unsigned slot = SI_PS_IMAGE_COLORBUF0;
	struct pipe_surface *surf = NULL;

	/* si_texture_disable_dcc can get us here again. */
	if (sctx->blitter->running)
		return;

	/* See whether FBFETCH is used and color buffer 0 is set. */
	if (sctx->ps_shader.cso &&
	    sctx->ps_shader.cso->info.opcode_count[TGSI_OPCODE_FBFETCH] &&
	    sctx->framebuffer.state.nr_cbufs &&
	    sctx->framebuffer.state.cbufs[0])
		surf = sctx->framebuffer.state.cbufs[0];

	/* Return if FBFETCH transitions from disabled to disabled. */
	if (!buffers->buffers[slot] && !surf)
		return;

	sctx->ps_uses_fbfetch = surf != NULL;
	si_update_ps_iter_samples(sctx);

	if (surf) {
		struct si_texture *tex = (struct si_texture*)surf->texture;
		struct pipe_image_view view;

		assert(tex);
		assert(!tex->is_depth);

		/* Disable DCC, because the texture is used as both a sampler
		 * and color buffer.
		 */
		si_texture_disable_dcc(sctx, tex);

		if (tex->buffer.b.b.nr_samples <= 1 && tex->cmask_buffer) {
			/* Disable CMASK. */
			assert(tex->cmask_buffer != &tex->buffer);
			si_eliminate_fast_color_clear(sctx, tex);
			si_texture_discard_cmask(sctx->screen, tex);
		}

		view.resource = surf->texture;
		view.format = surf->format;
		view.access = PIPE_IMAGE_ACCESS_READ;
		view.u.tex.first_layer = surf->u.tex.first_layer;
		view.u.tex.last_layer = surf->u.tex.last_layer;
		view.u.tex.level = surf->u.tex.level;

		/* Set the descriptor. */
		uint32_t *desc = descs->list + slot*4;
		memset(desc, 0, 16 * 4);
		si_set_shader_image_desc(sctx, &view, true, desc, desc + 8);

		pipe_resource_reference(&buffers->buffers[slot], &tex->buffer.b.b);
		radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
					  &tex->buffer, RADEON_USAGE_READ,
					  RADEON_PRIO_SHADER_RW_IMAGE);
		buffers->enabled_mask |= 1u << slot;
	} else {
		/* Clear the descriptor. */
		memset(descs->list + slot*4, 0, 8*4);
		pipe_resource_reference(&buffers->buffers[slot], NULL);
		buffers->enabled_mask &= ~(1u << slot);
	}

	sctx->descriptors_dirty |= 1u << SI_DESCS_RW_BUFFERS;
}

/* SAMPLER STATES */

static void si_bind_sampler_states(struct pipe_context *ctx,
                                   enum pipe_shader_type shader,
                                   unsigned start, unsigned count, void **states)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_samplers *samplers = &sctx->samplers[shader];
	struct si_descriptors *desc = si_sampler_and_image_descriptors(sctx, shader);
	struct si_sampler_state **sstates = (struct si_sampler_state**)states;
	int i;

	if (!count || shader >= SI_NUM_SHADERS)
		return;

	for (i = 0; i < count; i++) {
		unsigned slot = start + i;
		unsigned desc_slot = si_get_sampler_slot(slot);

		if (!sstates[i] ||
		    sstates[i] == samplers->sampler_states[slot])
			continue;

#ifdef DEBUG
		assert(sstates[i]->magic == SI_SAMPLER_STATE_MAGIC);
#endif
		samplers->sampler_states[slot] = sstates[i];

		/* If FMASK is bound, don't overwrite it.
		 * The sampler state will be set after FMASK is unbound.
		 */
		struct si_sampler_view *sview =
			(struct si_sampler_view *)samplers->views[slot];

		struct si_texture *tex = NULL;

		if (sview && sview->base.texture &&
		    sview->base.texture->target != PIPE_BUFFER)
			tex = (struct si_texture *)sview->base.texture;

		if (tex && tex->surface.fmask_size)
			continue;

		si_set_sampler_state_desc(sstates[i], sview, tex,
					  desc->list + desc_slot * 16 + 12);

		sctx->descriptors_dirty |= 1u << si_sampler_and_image_descriptors_idx(shader);
	}
}

/* BUFFER RESOURCES */

static void si_init_buffer_resources(struct si_buffer_resources *buffers,
				     struct si_descriptors *descs,
				     unsigned num_buffers,
				     short shader_userdata_rel_index,
				     enum radeon_bo_usage shader_usage,
				     enum radeon_bo_usage shader_usage_constbuf,
				     enum radeon_bo_priority priority,
				     enum radeon_bo_priority priority_constbuf)
{
	buffers->shader_usage = shader_usage;
	buffers->shader_usage_constbuf = shader_usage_constbuf;
	buffers->priority = priority;
	buffers->priority_constbuf = priority_constbuf;
	buffers->buffers = CALLOC(num_buffers, sizeof(struct pipe_resource*));

	si_init_descriptors(descs, shader_userdata_rel_index, 4, num_buffers);
}

static void si_release_buffer_resources(struct si_buffer_resources *buffers,
					struct si_descriptors *descs)
{
	int i;

	for (i = 0; i < descs->num_elements; i++) {
		pipe_resource_reference(&buffers->buffers[i], NULL);
	}

	FREE(buffers->buffers);
}

static void si_buffer_resources_begin_new_cs(struct si_context *sctx,
					     struct si_buffer_resources *buffers)
{
	unsigned mask = buffers->enabled_mask;

	/* Add buffers to the CS. */
	while (mask) {
		int i = u_bit_scan(&mask);

		radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
			r600_resource(buffers->buffers[i]),
			i < SI_NUM_SHADER_BUFFERS ? buffers->shader_usage :
						    buffers->shader_usage_constbuf,
			i < SI_NUM_SHADER_BUFFERS ? buffers->priority :
						    buffers->priority_constbuf);
	}
}

static void si_get_buffer_from_descriptors(struct si_buffer_resources *buffers,
					   struct si_descriptors *descs,
					   unsigned idx, struct pipe_resource **buf,
					   unsigned *offset, unsigned *size)
{
	pipe_resource_reference(buf, buffers->buffers[idx]);
	if (*buf) {
		struct r600_resource *res = r600_resource(*buf);
		const uint32_t *desc = descs->list + idx * 4;
		uint64_t va;

		*size = desc[2];

		assert(G_008F04_STRIDE(desc[1]) == 0);
		va = si_desc_extract_buffer_address(desc);

		assert(va >= res->gpu_address && va + *size <= res->gpu_address + res->bo_size);
		*offset = va - res->gpu_address;
	}
}

/* VERTEX BUFFERS */

static void si_vertex_buffers_begin_new_cs(struct si_context *sctx)
{
	int count = sctx->vertex_elements ? sctx->vertex_elements->count : 0;
	int i;

	for (i = 0; i < count; i++) {
		int vb = sctx->vertex_elements->vertex_buffer_index[i];

		if (vb >= ARRAY_SIZE(sctx->vertex_buffer))
			continue;
		if (!sctx->vertex_buffer[vb].buffer.resource)
			continue;

		radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
				      r600_resource(sctx->vertex_buffer[vb].buffer.resource),
				      RADEON_USAGE_READ, RADEON_PRIO_VERTEX_BUFFER);
	}

	if (!sctx->vb_descriptors_buffer)
		return;
	radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
				  sctx->vb_descriptors_buffer, RADEON_USAGE_READ,
				  RADEON_PRIO_DESCRIPTORS);
}

bool si_upload_vertex_buffer_descriptors(struct si_context *sctx)
{
	struct si_vertex_elements *velems = sctx->vertex_elements;
	unsigned i, count;
	unsigned desc_list_byte_size;
	unsigned first_vb_use_mask;
	uint32_t *ptr;

	if (!sctx->vertex_buffers_dirty || !velems)
		return true;

	count = velems->count;

	if (!count)
		return true;

	desc_list_byte_size = velems->desc_list_byte_size;
	first_vb_use_mask = velems->first_vb_use_mask;

	/* Vertex buffer descriptors are the only ones which are uploaded
	 * directly through a staging buffer and don't go through
	 * the fine-grained upload path.
	 */
	u_upload_alloc(sctx->b.const_uploader, 0,
		       desc_list_byte_size,
		       si_optimal_tcc_alignment(sctx, desc_list_byte_size),
		       &sctx->vb_descriptors_offset,
		       (struct pipe_resource**)&sctx->vb_descriptors_buffer,
		       (void**)&ptr);
	if (!sctx->vb_descriptors_buffer) {
		sctx->vb_descriptors_offset = 0;
		sctx->vb_descriptors_gpu_list = NULL;
		return false;
	}

	sctx->vb_descriptors_gpu_list = ptr;
	radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
				  sctx->vb_descriptors_buffer, RADEON_USAGE_READ,
				  RADEON_PRIO_DESCRIPTORS);

	assert(count <= SI_MAX_ATTRIBS);

	for (i = 0; i < count; i++) {
		struct pipe_vertex_buffer *vb;
		struct r600_resource *rbuffer;
		unsigned vbo_index = velems->vertex_buffer_index[i];
		uint32_t *desc = &ptr[i*4];

		vb = &sctx->vertex_buffer[vbo_index];
		rbuffer = r600_resource(vb->buffer.resource);
		if (!rbuffer) {
			memset(desc, 0, 16);
			continue;
		}

		int64_t offset = (int64_t)((int)vb->buffer_offset) +
				 velems->src_offset[i];
		uint64_t va = rbuffer->gpu_address + offset;

		int64_t num_records = (int64_t)rbuffer->b.b.width0 - offset;
		if (sctx->chip_class != VI && vb->stride) {
			/* Round up by rounding down and adding 1 */
			num_records = (num_records - velems->format_size[i]) /
				      vb->stride + 1;
		}
		assert(num_records >= 0 && num_records <= UINT_MAX);

		desc[0] = va;
		desc[1] = S_008F04_BASE_ADDRESS_HI(va >> 32) |
			  S_008F04_STRIDE(vb->stride);
		desc[2] = num_records;
		desc[3] = velems->rsrc_word3[i];

		if (first_vb_use_mask & (1 << i)) {
			radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
					      r600_resource(vb->buffer.resource),
					      RADEON_USAGE_READ, RADEON_PRIO_VERTEX_BUFFER);
		}
	}

	/* Don't flush the const cache. It would have a very negative effect
	 * on performance (confirmed by testing). New descriptors are always
	 * uploaded to a fresh new buffer, so I don't think flushing the const
	 * cache is needed. */
	si_mark_atom_dirty(sctx, &sctx->atoms.s.shader_pointers);
	sctx->vertex_buffers_dirty = false;
	sctx->vertex_buffer_pointer_dirty = true;
	sctx->prefetch_L2_mask |= SI_PREFETCH_VBO_DESCRIPTORS;
	return true;
}


/* CONSTANT BUFFERS */

static unsigned
si_const_and_shader_buffer_descriptors_idx(unsigned shader)
{
	return SI_DESCS_FIRST_SHADER + shader * SI_NUM_SHADER_DESCS +
	       SI_SHADER_DESCS_CONST_AND_SHADER_BUFFERS;
}

static struct si_descriptors *
si_const_and_shader_buffer_descriptors(struct si_context *sctx, unsigned shader)
{
	return &sctx->descriptors[si_const_and_shader_buffer_descriptors_idx(shader)];
}

void si_upload_const_buffer(struct si_context *sctx, struct r600_resource **rbuffer,
			    const uint8_t *ptr, unsigned size, uint32_t *const_offset)
{
	void *tmp;

	u_upload_alloc(sctx->b.const_uploader, 0, size,
		       si_optimal_tcc_alignment(sctx, size),
		       const_offset,
		       (struct pipe_resource**)rbuffer, &tmp);
	if (*rbuffer)
		util_memcpy_cpu_to_le32(tmp, ptr, size);
}

static void si_set_constant_buffer(struct si_context *sctx,
				   struct si_buffer_resources *buffers,
				   unsigned descriptors_idx,
				   uint slot, const struct pipe_constant_buffer *input)
{
	struct si_descriptors *descs = &sctx->descriptors[descriptors_idx];
	assert(slot < descs->num_elements);
	pipe_resource_reference(&buffers->buffers[slot], NULL);

	/* CIK cannot unbind a constant buffer (S_BUFFER_LOAD is buggy
	 * with a NULL buffer). We need to use a dummy buffer instead. */
	if (sctx->chip_class == CIK &&
	    (!input || (!input->buffer && !input->user_buffer)))
		input = &sctx->null_const_buf;

	if (input && (input->buffer || input->user_buffer)) {
		struct pipe_resource *buffer = NULL;
		uint64_t va;

		/* Upload the user buffer if needed. */
		if (input->user_buffer) {
			unsigned buffer_offset;

			si_upload_const_buffer(sctx,
					       (struct r600_resource**)&buffer, input->user_buffer,
					       input->buffer_size, &buffer_offset);
			if (!buffer) {
				/* Just unbind on failure. */
				si_set_constant_buffer(sctx, buffers, descriptors_idx, slot, NULL);
				return;
			}
			va = r600_resource(buffer)->gpu_address + buffer_offset;
		} else {
			pipe_resource_reference(&buffer, input->buffer);
			va = r600_resource(buffer)->gpu_address + input->buffer_offset;
			/* Only track usage for non-user buffers. */
			r600_resource(buffer)->bind_history |= PIPE_BIND_CONSTANT_BUFFER;
		}

		/* Set the descriptor. */
		uint32_t *desc = descs->list + slot*4;
		desc[0] = va;
		desc[1] = S_008F04_BASE_ADDRESS_HI(va >> 32) |
			  S_008F04_STRIDE(0);
		desc[2] = input->buffer_size;
		desc[3] = S_008F0C_DST_SEL_X(V_008F0C_SQ_SEL_X) |
			  S_008F0C_DST_SEL_Y(V_008F0C_SQ_SEL_Y) |
			  S_008F0C_DST_SEL_Z(V_008F0C_SQ_SEL_Z) |
			  S_008F0C_DST_SEL_W(V_008F0C_SQ_SEL_W) |
			  S_008F0C_NUM_FORMAT(V_008F0C_BUF_NUM_FORMAT_FLOAT) |
			  S_008F0C_DATA_FORMAT(V_008F0C_BUF_DATA_FORMAT_32);

		buffers->buffers[slot] = buffer;
		radeon_add_to_gfx_buffer_list_check_mem(sctx,
							r600_resource(buffer),
							buffers->shader_usage_constbuf,
							buffers->priority_constbuf, true);
		buffers->enabled_mask |= 1u << slot;
	} else {
		/* Clear the descriptor. */
		memset(descs->list + slot*4, 0, sizeof(uint32_t) * 4);
		buffers->enabled_mask &= ~(1u << slot);
	}

	sctx->descriptors_dirty |= 1u << descriptors_idx;
}

void si_set_rw_buffer(struct si_context *sctx,
		      uint slot, const struct pipe_constant_buffer *input)
{
	si_set_constant_buffer(sctx, &sctx->rw_buffers,
			                        SI_DESCS_RW_BUFFERS, slot, input);
}

static void si_pipe_set_constant_buffer(struct pipe_context *ctx,
					enum pipe_shader_type shader, uint slot,
					const struct pipe_constant_buffer *input)
{
	struct si_context *sctx = (struct si_context *)ctx;

	if (shader >= SI_NUM_SHADERS)
		return;

	if (slot == 0 && input && input->buffer &&
	    !(r600_resource(input->buffer)->flags & RADEON_FLAG_32BIT)) {
		assert(!"constant buffer 0 must have a 32-bit VM address, use const_uploader");
		return;
	}

	slot = si_get_constbuf_slot(slot);
	si_set_constant_buffer(sctx, &sctx->const_and_shader_buffers[shader],
			       si_const_and_shader_buffer_descriptors_idx(shader),
			       slot, input);
}

void si_get_pipe_constant_buffer(struct si_context *sctx, uint shader,
				 uint slot, struct pipe_constant_buffer *cbuf)
{
	cbuf->user_buffer = NULL;
	si_get_buffer_from_descriptors(
		&sctx->const_and_shader_buffers[shader],
		si_const_and_shader_buffer_descriptors(sctx, shader),
		si_get_constbuf_slot(slot),
		&cbuf->buffer, &cbuf->buffer_offset, &cbuf->buffer_size);
}

/* SHADER BUFFERS */

static void si_set_shader_buffers(struct pipe_context *ctx,
				  enum pipe_shader_type shader,
				  unsigned start_slot, unsigned count,
				  const struct pipe_shader_buffer *sbuffers)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_buffer_resources *buffers = &sctx->const_and_shader_buffers[shader];
	struct si_descriptors *descs = si_const_and_shader_buffer_descriptors(sctx, shader);
	unsigned i;

	assert(start_slot + count <= SI_NUM_SHADER_BUFFERS);

	for (i = 0; i < count; ++i) {
		const struct pipe_shader_buffer *sbuffer = sbuffers ? &sbuffers[i] : NULL;
		struct r600_resource *buf;
		unsigned slot = si_get_shaderbuf_slot(start_slot + i);
		uint32_t *desc = descs->list + slot * 4;
		uint64_t va;

		if (!sbuffer || !sbuffer->buffer) {
			pipe_resource_reference(&buffers->buffers[slot], NULL);
			memset(desc, 0, sizeof(uint32_t) * 4);
			buffers->enabled_mask &= ~(1u << slot);
			sctx->descriptors_dirty |=
				1u << si_const_and_shader_buffer_descriptors_idx(shader);
			continue;
		}

		buf = r600_resource(sbuffer->buffer);
		va = buf->gpu_address + sbuffer->buffer_offset;

		desc[0] = va;
		desc[1] = S_008F04_BASE_ADDRESS_HI(va >> 32) |
			  S_008F04_STRIDE(0);
		desc[2] = sbuffer->buffer_size;
		desc[3] = S_008F0C_DST_SEL_X(V_008F0C_SQ_SEL_X) |
			  S_008F0C_DST_SEL_Y(V_008F0C_SQ_SEL_Y) |
			  S_008F0C_DST_SEL_Z(V_008F0C_SQ_SEL_Z) |
			  S_008F0C_DST_SEL_W(V_008F0C_SQ_SEL_W) |
			  S_008F0C_NUM_FORMAT(V_008F0C_BUF_NUM_FORMAT_FLOAT) |
			  S_008F0C_DATA_FORMAT(V_008F0C_BUF_DATA_FORMAT_32);

		pipe_resource_reference(&buffers->buffers[slot], &buf->b.b);
		radeon_add_to_gfx_buffer_list_check_mem(sctx, buf,
							buffers->shader_usage,
							buffers->priority, true);
		buf->bind_history |= PIPE_BIND_SHADER_BUFFER;

		buffers->enabled_mask |= 1u << slot;
		sctx->descriptors_dirty |=
			1u << si_const_and_shader_buffer_descriptors_idx(shader);

		util_range_add(&buf->valid_buffer_range, sbuffer->buffer_offset,
			       sbuffer->buffer_offset + sbuffer->buffer_size);
	}
}

void si_get_shader_buffers(struct si_context *sctx,
			   enum pipe_shader_type shader,
			   uint start_slot, uint count,
			   struct pipe_shader_buffer *sbuf)
{
	struct si_buffer_resources *buffers = &sctx->const_and_shader_buffers[shader];
	struct si_descriptors *descs = si_const_and_shader_buffer_descriptors(sctx, shader);

	for (unsigned i = 0; i < count; ++i) {
		si_get_buffer_from_descriptors(
			buffers, descs,
			si_get_shaderbuf_slot(start_slot + i),
			&sbuf[i].buffer, &sbuf[i].buffer_offset,
			&sbuf[i].buffer_size);
	}
}

/* RING BUFFERS */

void si_set_ring_buffer(struct si_context *sctx, uint slot,
			struct pipe_resource *buffer,
			unsigned stride, unsigned num_records,
			bool add_tid, bool swizzle,
			unsigned element_size, unsigned index_stride, uint64_t offset)
{
	struct si_buffer_resources *buffers = &sctx->rw_buffers;
	struct si_descriptors *descs = &sctx->descriptors[SI_DESCS_RW_BUFFERS];

	/* The stride field in the resource descriptor has 14 bits */
	assert(stride < (1 << 14));

	assert(slot < descs->num_elements);
	pipe_resource_reference(&buffers->buffers[slot], NULL);

	if (buffer) {
		uint64_t va;

		va = r600_resource(buffer)->gpu_address + offset;

		switch (element_size) {
		default:
			assert(!"Unsupported ring buffer element size");
		case 0:
		case 2:
			element_size = 0;
			break;
		case 4:
			element_size = 1;
			break;
		case 8:
			element_size = 2;
			break;
		case 16:
			element_size = 3;
			break;
		}

		switch (index_stride) {
		default:
			assert(!"Unsupported ring buffer index stride");
		case 0:
		case 8:
			index_stride = 0;
			break;
		case 16:
			index_stride = 1;
			break;
		case 32:
			index_stride = 2;
			break;
		case 64:
			index_stride = 3;
			break;
		}

		if (sctx->chip_class >= VI && stride)
			num_records *= stride;

		/* Set the descriptor. */
		uint32_t *desc = descs->list + slot*4;
		desc[0] = va;
		desc[1] = S_008F04_BASE_ADDRESS_HI(va >> 32) |
			  S_008F04_STRIDE(stride) |
			  S_008F04_SWIZZLE_ENABLE(swizzle);
		desc[2] = num_records;
		desc[3] = S_008F0C_DST_SEL_X(V_008F0C_SQ_SEL_X) |
			  S_008F0C_DST_SEL_Y(V_008F0C_SQ_SEL_Y) |
			  S_008F0C_DST_SEL_Z(V_008F0C_SQ_SEL_Z) |
			  S_008F0C_DST_SEL_W(V_008F0C_SQ_SEL_W) |
			  S_008F0C_NUM_FORMAT(V_008F0C_BUF_NUM_FORMAT_FLOAT) |
			  S_008F0C_DATA_FORMAT(V_008F0C_BUF_DATA_FORMAT_32) |
			  S_008F0C_INDEX_STRIDE(index_stride) |
			  S_008F0C_ADD_TID_ENABLE(add_tid);

		if (sctx->chip_class >= GFX9)
			assert(!swizzle || element_size == 1); /* always 4 bytes on GFX9 */
		else
			desc[3] |= S_008F0C_ELEMENT_SIZE(element_size);

		pipe_resource_reference(&buffers->buffers[slot], buffer);
		radeon_add_to_buffer_list(sctx, sctx->gfx_cs,
				      r600_resource(buffer),
				      buffers->shader_usage, buffers->priority);
		buffers->enabled_mask |= 1u << slot;
	} else {
		/* Clear the descriptor. */
		memset(descs->list + slot*4, 0, sizeof(uint32_t) * 4);
		buffers->enabled_mask &= ~(1u << slot);
	}

	sctx->descriptors_dirty |= 1u << SI_DESCS_RW_BUFFERS;
}

static void si_desc_reset_buffer_offset(uint32_t *desc, uint64_t old_buf_va,
					struct pipe_resource *new_buf)
{
	/* Retrieve the buffer offset from the descriptor. */
	uint64_t old_desc_va = si_desc_extract_buffer_address(desc);

	assert(old_buf_va <= old_desc_va);
	uint64_t offset_within_buffer = old_desc_va - old_buf_va;

	/* Update the descriptor. */
	si_set_buf_desc_address(r600_resource(new_buf), offset_within_buffer,
				desc);
}

/* INTERNAL CONST BUFFERS */

static void si_set_polygon_stipple(struct pipe_context *ctx,
				   const struct pipe_poly_stipple *state)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct pipe_constant_buffer cb = {};
	unsigned stipple[32];
	int i;

	for (i = 0; i < 32; i++)
		stipple[i] = util_bitreverse(state->stipple[i]);

	cb.user_buffer = stipple;
	cb.buffer_size = sizeof(stipple);

	si_set_rw_buffer(sctx, SI_PS_CONST_POLY_STIPPLE, &cb);
}

/* TEXTURE METADATA ENABLE/DISABLE */

static void
si_resident_handles_update_needs_color_decompress(struct si_context *sctx)
{
	util_dynarray_clear(&sctx->resident_tex_needs_color_decompress);
	util_dynarray_clear(&sctx->resident_img_needs_color_decompress);

	util_dynarray_foreach(&sctx->resident_tex_handles,
			      struct si_texture_handle *, tex_handle) {
		struct pipe_resource *res = (*tex_handle)->view->texture;
		struct si_texture *tex;

		if (!res || res->target == PIPE_BUFFER)
			continue;

		tex = (struct si_texture *)res;
		if (!color_needs_decompression(tex))
			continue;

		util_dynarray_append(&sctx->resident_tex_needs_color_decompress,
				     struct si_texture_handle *, *tex_handle);
	}

	util_dynarray_foreach(&sctx->resident_img_handles,
			      struct si_image_handle *, img_handle) {
		struct pipe_image_view *view = &(*img_handle)->view;
		struct pipe_resource *res = view->resource;
		struct si_texture *tex;

		if (!res || res->target == PIPE_BUFFER)
			continue;

		tex = (struct si_texture *)res;
		if (!color_needs_decompression(tex))
			continue;

		util_dynarray_append(&sctx->resident_img_needs_color_decompress,
				     struct si_image_handle *, *img_handle);
	}
}

/* CMASK can be enabled (for fast clear) and disabled (for texture export)
 * while the texture is bound, possibly by a different context. In that case,
 * call this function to update needs_*_decompress_masks.
 */
void si_update_needs_color_decompress_masks(struct si_context *sctx)
{
	for (int i = 0; i < SI_NUM_SHADERS; ++i) {
		si_samplers_update_needs_color_decompress_mask(&sctx->samplers[i]);
		si_images_update_needs_color_decompress_mask(&sctx->images[i]);
		si_update_shader_needs_decompress_mask(sctx, i);
	}

	si_resident_handles_update_needs_color_decompress(sctx);
}

/* BUFFER DISCARD/INVALIDATION */

/** Reset descriptors of buffer resources after \p buf has been invalidated. */
static void si_reset_buffer_resources(struct si_context *sctx,
				      struct si_buffer_resources *buffers,
				      unsigned descriptors_idx,
				      unsigned slot_mask,
				      struct pipe_resource *buf,
				      uint64_t old_va,
				      enum radeon_bo_usage usage,
				      enum radeon_bo_priority priority)
{
	struct si_descriptors *descs = &sctx->descriptors[descriptors_idx];
	unsigned mask = buffers->enabled_mask & slot_mask;

	while (mask) {
		unsigned i = u_bit_scan(&mask);
		if (buffers->buffers[i] == buf) {
			si_desc_reset_buffer_offset(descs->list + i*4,
						    old_va, buf);
			sctx->descriptors_dirty |= 1u << descriptors_idx;

			radeon_add_to_gfx_buffer_list_check_mem(sctx,
								r600_resource(buf),
								usage, priority, true);
		}
	}
}

/* Update all resource bindings where the buffer is bound, including
 * all resource descriptors. This is invalidate_buffer without
 * the invalidation. */
void si_rebind_buffer(struct si_context *sctx, struct pipe_resource *buf,
		      uint64_t old_va)
{
	struct r600_resource *rbuffer = r600_resource(buf);
	unsigned i, shader;
	unsigned num_elems = sctx->vertex_elements ?
				       sctx->vertex_elements->count : 0;

	/* We changed the buffer, now we need to bind it where the old one
	 * was bound. This consists of 2 things:
	 *   1) Updating the resource descriptor and dirtying it.
	 *   2) Adding a relocation to the CS, so that it's usable.
	 */

	/* Vertex buffers. */
	if (rbuffer->bind_history & PIPE_BIND_VERTEX_BUFFER) {
		for (i = 0; i < num_elems; i++) {
			int vb = sctx->vertex_elements->vertex_buffer_index[i];

			if (vb >= ARRAY_SIZE(sctx->vertex_buffer))
				continue;
			if (!sctx->vertex_buffer[vb].buffer.resource)
				continue;

			if (sctx->vertex_buffer[vb].buffer.resource == buf) {
				sctx->vertex_buffers_dirty = true;
				break;
			}
		}
	}

	/* Streamout buffers. (other internal buffers can't be invalidated) */
	if (rbuffer->bind_history & PIPE_BIND_STREAM_OUTPUT) {
		for (i = SI_VS_STREAMOUT_BUF0; i <= SI_VS_STREAMOUT_BUF3; i++) {
			struct si_buffer_resources *buffers = &sctx->rw_buffers;
			struct si_descriptors *descs =
				&sctx->descriptors[SI_DESCS_RW_BUFFERS];

			if (buffers->buffers[i] != buf)
				continue;

			si_desc_reset_buffer_offset(descs->list + i*4,
						    old_va, buf);
			sctx->descriptors_dirty |= 1u << SI_DESCS_RW_BUFFERS;

			radeon_add_to_gfx_buffer_list_check_mem(sctx,
								rbuffer, buffers->shader_usage,
								RADEON_PRIO_SHADER_RW_BUFFER,
								true);

			/* Update the streamout state. */
			if (sctx->streamout.begin_emitted)
				si_emit_streamout_end(sctx);
			sctx->streamout.append_bitmask =
					sctx->streamout.enabled_mask;
			si_streamout_buffers_dirty(sctx);
		}
	}

	/* Constant and shader buffers. */
	if (rbuffer->bind_history & PIPE_BIND_CONSTANT_BUFFER) {
		for (shader = 0; shader < SI_NUM_SHADERS; shader++)
			si_reset_buffer_resources(sctx, &sctx->const_and_shader_buffers[shader],
						  si_const_and_shader_buffer_descriptors_idx(shader),
						  u_bit_consecutive(SI_NUM_SHADER_BUFFERS, SI_NUM_CONST_BUFFERS),
						  buf, old_va,
						  sctx->const_and_shader_buffers[shader].shader_usage_constbuf,
						  sctx->const_and_shader_buffers[shader].priority_constbuf);
	}

	if (rbuffer->bind_history & PIPE_BIND_SHADER_BUFFER) {
		for (shader = 0; shader < SI_NUM_SHADERS; shader++)
			si_reset_buffer_resources(sctx, &sctx->const_and_shader_buffers[shader],
						  si_const_and_shader_buffer_descriptors_idx(shader),
						  u_bit_consecutive(0, SI_NUM_SHADER_BUFFERS),
						  buf, old_va,
						  sctx->const_and_shader_buffers[shader].shader_usage,
						  sctx->const_and_shader_buffers[shader].priority);
	}

	if (rbuffer->bind_history & PIPE_BIND_SAMPLER_VIEW) {
		/* Texture buffers - update bindings. */
		for (shader = 0; shader < SI_NUM_SHADERS; shader++) {
			struct si_samplers *samplers = &sctx->samplers[shader];
			struct si_descriptors *descs =
				si_sampler_and_image_descriptors(sctx, shader);
			unsigned mask = samplers->enabled_mask;

			while (mask) {
				unsigned i = u_bit_scan(&mask);
				if (samplers->views[i]->texture == buf) {
					unsigned desc_slot = si_get_sampler_slot(i);

					si_desc_reset_buffer_offset(descs->list +
								    desc_slot * 16 + 4,
								    old_va, buf);
					sctx->descriptors_dirty |=
						1u << si_sampler_and_image_descriptors_idx(shader);

					radeon_add_to_gfx_buffer_list_check_mem(sctx,
									    rbuffer, RADEON_USAGE_READ,
									    RADEON_PRIO_SAMPLER_BUFFER,
									    true);
				}
			}
		}
	}

	/* Shader images */
	if (rbuffer->bind_history & PIPE_BIND_SHADER_IMAGE) {
		for (shader = 0; shader < SI_NUM_SHADERS; ++shader) {
			struct si_images *images = &sctx->images[shader];
			struct si_descriptors *descs =
				si_sampler_and_image_descriptors(sctx, shader);
			unsigned mask = images->enabled_mask;

			while (mask) {
				unsigned i = u_bit_scan(&mask);

				if (images->views[i].resource == buf) {
					unsigned desc_slot = si_get_image_slot(i);

					if (images->views[i].access & PIPE_IMAGE_ACCESS_WRITE)
						si_mark_image_range_valid(&images->views[i]);

					si_desc_reset_buffer_offset(
						descs->list + desc_slot * 8 + 4,
						old_va, buf);
					sctx->descriptors_dirty |=
						1u << si_sampler_and_image_descriptors_idx(shader);

					radeon_add_to_gfx_buffer_list_check_mem(
						sctx, rbuffer,
						RADEON_USAGE_READWRITE,
						RADEON_PRIO_SAMPLER_BUFFER, true);
				}
			}
		}
	}

	/* Bindless texture handles */
	if (rbuffer->texture_handle_allocated) {
		struct si_descriptors *descs = &sctx->bindless_descriptors;

		util_dynarray_foreach(&sctx->resident_tex_handles,
				      struct si_texture_handle *, tex_handle) {
			struct pipe_sampler_view *view = (*tex_handle)->view;
			unsigned desc_slot = (*tex_handle)->desc_slot;

			if (view->texture == buf) {
				si_set_buf_desc_address(rbuffer,
							view->u.buf.offset,
							descs->list +
							desc_slot * 16 + 4);

				(*tex_handle)->desc_dirty = true;
				sctx->bindless_descriptors_dirty = true;

				radeon_add_to_gfx_buffer_list_check_mem(
					sctx, rbuffer,
					RADEON_USAGE_READ,
					RADEON_PRIO_SAMPLER_BUFFER, true);
			}
		}
	}

	/* Bindless image handles */
	if (rbuffer->image_handle_allocated) {
		struct si_descriptors *descs = &sctx->bindless_descriptors;

		util_dynarray_foreach(&sctx->resident_img_handles,
				      struct si_image_handle *, img_handle) {
			struct pipe_image_view *view = &(*img_handle)->view;
			unsigned desc_slot = (*img_handle)->desc_slot;

			if (view->resource == buf) {
				if (view->access & PIPE_IMAGE_ACCESS_WRITE)
					si_mark_image_range_valid(view);

				si_set_buf_desc_address(rbuffer,
							view->u.buf.offset,
							descs->list +
							desc_slot * 16 + 4);

				(*img_handle)->desc_dirty = true;
				sctx->bindless_descriptors_dirty = true;

				radeon_add_to_gfx_buffer_list_check_mem(
					sctx, rbuffer,
					RADEON_USAGE_READWRITE,
					RADEON_PRIO_SAMPLER_BUFFER, true);
			}
		}
	}
}

static void si_upload_bindless_descriptor(struct si_context *sctx,
					  unsigned desc_slot,
					  unsigned num_dwords)
{
	struct si_descriptors *desc = &sctx->bindless_descriptors;
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	unsigned desc_slot_offset = desc_slot * 16;
	uint32_t *data;
	uint64_t va;

	data = desc->list + desc_slot_offset;
	va = desc->gpu_address + desc_slot_offset * 4;

	radeon_emit(cs, PKT3(PKT3_WRITE_DATA, 2 + num_dwords, 0));
	radeon_emit(cs, S_370_DST_SEL(V_370_TC_L2) |
		    S_370_WR_CONFIRM(1) |
		    S_370_ENGINE_SEL(V_370_ME));
	radeon_emit(cs, va);
	radeon_emit(cs, va >> 32);
	radeon_emit_array(cs, data, num_dwords);
}

static void si_upload_bindless_descriptors(struct si_context *sctx)
{
	if (!sctx->bindless_descriptors_dirty)
		return;

	/* Wait for graphics/compute to be idle before updating the resident
	 * descriptors directly in memory, in case the GPU is using them.
	 */
	sctx->flags |= SI_CONTEXT_PS_PARTIAL_FLUSH |
			 SI_CONTEXT_CS_PARTIAL_FLUSH;
	si_emit_cache_flush(sctx);

	util_dynarray_foreach(&sctx->resident_tex_handles,
			      struct si_texture_handle *, tex_handle) {
		unsigned desc_slot = (*tex_handle)->desc_slot;

		if (!(*tex_handle)->desc_dirty)
			continue;

		si_upload_bindless_descriptor(sctx, desc_slot, 16);
		(*tex_handle)->desc_dirty = false;
	}

	util_dynarray_foreach(&sctx->resident_img_handles,
			      struct si_image_handle *, img_handle) {
		unsigned desc_slot = (*img_handle)->desc_slot;

		if (!(*img_handle)->desc_dirty)
			continue;

		si_upload_bindless_descriptor(sctx, desc_slot, 8);
		(*img_handle)->desc_dirty = false;
	}

	/* Invalidate L1 because it doesn't know that L2 changed. */
	sctx->flags |= SI_CONTEXT_INV_SMEM_L1;
	si_emit_cache_flush(sctx);

	sctx->bindless_descriptors_dirty = false;
}

/* Update mutable image descriptor fields of all resident textures. */
static void si_update_bindless_texture_descriptor(struct si_context *sctx,
						  struct si_texture_handle *tex_handle)
{
	struct si_sampler_view *sview = (struct si_sampler_view *)tex_handle->view;
	struct si_descriptors *desc = &sctx->bindless_descriptors;
	unsigned desc_slot_offset = tex_handle->desc_slot * 16;
	uint32_t desc_list[16];

	if (sview->base.texture->target == PIPE_BUFFER)
		return;

	memcpy(desc_list, desc->list + desc_slot_offset, sizeof(desc_list));
	si_set_sampler_view_desc(sctx, sview, &tex_handle->sstate,
				 desc->list + desc_slot_offset);

	if (memcmp(desc_list, desc->list + desc_slot_offset,
		   sizeof(desc_list))) {
		tex_handle->desc_dirty = true;
		sctx->bindless_descriptors_dirty = true;
	}
}

static void si_update_bindless_image_descriptor(struct si_context *sctx,
						struct si_image_handle *img_handle)
{
	struct si_descriptors *desc = &sctx->bindless_descriptors;
	unsigned desc_slot_offset = img_handle->desc_slot * 16;
	struct pipe_image_view *view = &img_handle->view;
	uint32_t desc_list[8];

	if (view->resource->target == PIPE_BUFFER)
		return;

	memcpy(desc_list, desc->list + desc_slot_offset,
	       sizeof(desc_list));
	si_set_shader_image_desc(sctx, view, true,
				 desc->list + desc_slot_offset, NULL);

	if (memcmp(desc_list, desc->list + desc_slot_offset,
		   sizeof(desc_list))) {
		img_handle->desc_dirty = true;
		sctx->bindless_descriptors_dirty = true;
	}
}

static void si_update_all_resident_texture_descriptors(struct si_context *sctx)
{
	util_dynarray_foreach(&sctx->resident_tex_handles,
			      struct si_texture_handle *, tex_handle) {
		si_update_bindless_texture_descriptor(sctx, *tex_handle);
	}

	util_dynarray_foreach(&sctx->resident_img_handles,
			      struct si_image_handle *, img_handle) {
		si_update_bindless_image_descriptor(sctx, *img_handle);
	}

	si_upload_bindless_descriptors(sctx);
}

/* Update mutable image descriptor fields of all bound textures. */
void si_update_all_texture_descriptors(struct si_context *sctx)
{
	unsigned shader;

	for (shader = 0; shader < SI_NUM_SHADERS; shader++) {
		struct si_samplers *samplers = &sctx->samplers[shader];
		struct si_images *images = &sctx->images[shader];
		unsigned mask;

		/* Images. */
		mask = images->enabled_mask;
		while (mask) {
			unsigned i = u_bit_scan(&mask);
			struct pipe_image_view *view = &images->views[i];

			if (!view->resource ||
			    view->resource->target == PIPE_BUFFER)
				continue;

			si_set_shader_image(sctx, shader, i, view, true);
		}

		/* Sampler views. */
		mask = samplers->enabled_mask;
		while (mask) {
			unsigned i = u_bit_scan(&mask);
			struct pipe_sampler_view *view = samplers->views[i];

			if (!view ||
			    !view->texture ||
			    view->texture->target == PIPE_BUFFER)
				continue;

			si_set_sampler_view(sctx, shader, i,
					    samplers->views[i], true);
		}

		si_update_shader_needs_decompress_mask(sctx, shader);
	}

	si_update_all_resident_texture_descriptors(sctx);
	si_update_ps_colorbuf0_slot(sctx);
}

/* SHADER USER DATA */

static void si_mark_shader_pointers_dirty(struct si_context *sctx,
					  unsigned shader)
{
	sctx->shader_pointers_dirty |=
		u_bit_consecutive(SI_DESCS_FIRST_SHADER + shader * SI_NUM_SHADER_DESCS,
				  SI_NUM_SHADER_DESCS);

	if (shader == PIPE_SHADER_VERTEX)
		sctx->vertex_buffer_pointer_dirty = sctx->vb_descriptors_buffer != NULL;

	si_mark_atom_dirty(sctx, &sctx->atoms.s.shader_pointers);
}

static void si_shader_pointers_begin_new_cs(struct si_context *sctx)
{
	sctx->shader_pointers_dirty = u_bit_consecutive(0, SI_NUM_DESCS);
	sctx->vertex_buffer_pointer_dirty = sctx->vb_descriptors_buffer != NULL;
	si_mark_atom_dirty(sctx, &sctx->atoms.s.shader_pointers);
	sctx->graphics_bindless_pointer_dirty = sctx->bindless_descriptors.buffer != NULL;
	sctx->compute_bindless_pointer_dirty = sctx->bindless_descriptors.buffer != NULL;
}

/* Set a base register address for user data constants in the given shader.
 * This assigns a mapping from PIPE_SHADER_* to SPI_SHADER_USER_DATA_*.
 */
static void si_set_user_data_base(struct si_context *sctx,
				  unsigned shader, uint32_t new_base)
{
	uint32_t *base = &sctx->shader_pointers.sh_base[shader];

	if (*base != new_base) {
		*base = new_base;

		if (new_base)
			si_mark_shader_pointers_dirty(sctx, shader);

		/* Any change in enabled shader stages requires re-emitting
		 * the VS state SGPR, because it contains the clamp_vertex_color
		 * state, which can be done in VS, TES, and GS.
		 */
		sctx->last_vs_state = ~0;
	}
}

/* This must be called when these shaders are changed from non-NULL to NULL
 * and vice versa:
 * - geometry shader
 * - tessellation control shader
 * - tessellation evaluation shader
 */
void si_shader_change_notify(struct si_context *sctx)
{
	/* VS can be bound as VS, ES, or LS. */
	if (sctx->tes_shader.cso) {
		if (sctx->chip_class >= GFX9) {
			si_set_user_data_base(sctx, PIPE_SHADER_VERTEX,
					      R_00B430_SPI_SHADER_USER_DATA_LS_0);
		} else {
			si_set_user_data_base(sctx, PIPE_SHADER_VERTEX,
					      R_00B530_SPI_SHADER_USER_DATA_LS_0);
		}
	} else if (sctx->gs_shader.cso) {
		si_set_user_data_base(sctx, PIPE_SHADER_VERTEX,
				      R_00B330_SPI_SHADER_USER_DATA_ES_0);
	} else {
		si_set_user_data_base(sctx, PIPE_SHADER_VERTEX,
				      R_00B130_SPI_SHADER_USER_DATA_VS_0);
	}

	/* TES can be bound as ES, VS, or not bound. */
	if (sctx->tes_shader.cso) {
		if (sctx->gs_shader.cso)
			si_set_user_data_base(sctx, PIPE_SHADER_TESS_EVAL,
					      R_00B330_SPI_SHADER_USER_DATA_ES_0);
		else
			si_set_user_data_base(sctx, PIPE_SHADER_TESS_EVAL,
					      R_00B130_SPI_SHADER_USER_DATA_VS_0);
	} else {
		si_set_user_data_base(sctx, PIPE_SHADER_TESS_EVAL, 0);
	}
}

static void si_emit_shader_pointer_head(struct radeon_cmdbuf *cs,
					unsigned sh_offset,
					unsigned pointer_count)
{
	radeon_emit(cs, PKT3(PKT3_SET_SH_REG, pointer_count * (HAVE_32BIT_POINTERS ? 1 : 2), 0));
	radeon_emit(cs, (sh_offset - SI_SH_REG_OFFSET) >> 2);
}

static void si_emit_shader_pointer_body(struct si_screen *sscreen,
					struct radeon_cmdbuf *cs,
					uint64_t va)
{
	radeon_emit(cs, va);

	if (HAVE_32BIT_POINTERS)
		assert(va == 0 || (va >> 32) == sscreen->info.address32_hi);
	else
		radeon_emit(cs, va >> 32);
}

static void si_emit_shader_pointer(struct si_context *sctx,
				   struct si_descriptors *desc,
				   unsigned sh_base)
{
	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	unsigned sh_offset = sh_base + desc->shader_userdata_offset;

	si_emit_shader_pointer_head(cs, sh_offset, 1);
	si_emit_shader_pointer_body(sctx->screen, cs, desc->gpu_address);
}

static void si_emit_consecutive_shader_pointers(struct si_context *sctx,
						unsigned pointer_mask,
						unsigned sh_base)
{
	if (!sh_base)
		return;

	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	unsigned mask = sctx->shader_pointers_dirty & pointer_mask;

	while (mask) {
		int start, count;
		u_bit_scan_consecutive_range(&mask, &start, &count);

		struct si_descriptors *descs = &sctx->descriptors[start];
		unsigned sh_offset = sh_base + descs->shader_userdata_offset;

		si_emit_shader_pointer_head(cs, sh_offset, count);
		for (int i = 0; i < count; i++)
			si_emit_shader_pointer_body(sctx->screen, cs,
						    descs[i].gpu_address);
	}
}

static void si_emit_disjoint_shader_pointers(struct si_context *sctx,
					     unsigned pointer_mask,
					     unsigned sh_base)
{
	if (!sh_base)
		return;

	struct radeon_cmdbuf *cs = sctx->gfx_cs;
	unsigned mask = sctx->shader_pointers_dirty & pointer_mask;

	while (mask) {
		struct si_descriptors *descs = &sctx->descriptors[u_bit_scan(&mask)];
		unsigned sh_offset = sh_base + descs->shader_userdata_offset;

		si_emit_shader_pointer_head(cs, sh_offset, 1);
		si_emit_shader_pointer_body(sctx->screen, cs, descs->gpu_address);
	}
}

static void si_emit_global_shader_pointers(struct si_context *sctx,
					   struct si_descriptors *descs)
{
	if (sctx->chip_class == GFX9) {
		/* Broadcast it to all shader stages. */
		si_emit_shader_pointer(sctx, descs,
				       R_00B530_SPI_SHADER_USER_DATA_COMMON_0);
		return;
	}

	si_emit_shader_pointer(sctx, descs,
			       R_00B030_SPI_SHADER_USER_DATA_PS_0);
	si_emit_shader_pointer(sctx, descs,
			       R_00B130_SPI_SHADER_USER_DATA_VS_0);
	si_emit_shader_pointer(sctx, descs,
			       R_00B330_SPI_SHADER_USER_DATA_ES_0);
	si_emit_shader_pointer(sctx, descs,
			       R_00B230_SPI_SHADER_USER_DATA_GS_0);
	si_emit_shader_pointer(sctx, descs,
			       R_00B430_SPI_SHADER_USER_DATA_HS_0);
	si_emit_shader_pointer(sctx, descs,
			       R_00B530_SPI_SHADER_USER_DATA_LS_0);
}

void si_emit_graphics_shader_pointers(struct si_context *sctx)
{
	uint32_t *sh_base = sctx->shader_pointers.sh_base;

	if (sctx->shader_pointers_dirty & (1 << SI_DESCS_RW_BUFFERS)) {
		si_emit_global_shader_pointers(sctx,
					       &sctx->descriptors[SI_DESCS_RW_BUFFERS]);
	}

	si_emit_consecutive_shader_pointers(sctx, SI_DESCS_SHADER_MASK(VERTEX),
					    sh_base[PIPE_SHADER_VERTEX]);
	si_emit_consecutive_shader_pointers(sctx, SI_DESCS_SHADER_MASK(TESS_EVAL),
					    sh_base[PIPE_SHADER_TESS_EVAL]);
	si_emit_consecutive_shader_pointers(sctx, SI_DESCS_SHADER_MASK(FRAGMENT),
					    sh_base[PIPE_SHADER_FRAGMENT]);
	if (HAVE_32BIT_POINTERS || sctx->chip_class <= VI) {
		si_emit_consecutive_shader_pointers(sctx, SI_DESCS_SHADER_MASK(TESS_CTRL),
						    sh_base[PIPE_SHADER_TESS_CTRL]);
		si_emit_consecutive_shader_pointers(sctx, SI_DESCS_SHADER_MASK(GEOMETRY),
						    sh_base[PIPE_SHADER_GEOMETRY]);
	} else {
		si_emit_disjoint_shader_pointers(sctx, SI_DESCS_SHADER_MASK(TESS_CTRL),
						 sh_base[PIPE_SHADER_TESS_CTRL]);
		si_emit_disjoint_shader_pointers(sctx, SI_DESCS_SHADER_MASK(GEOMETRY),
						 sh_base[PIPE_SHADER_GEOMETRY]);
	}

	sctx->shader_pointers_dirty &=
		~u_bit_consecutive(SI_DESCS_RW_BUFFERS, SI_DESCS_FIRST_COMPUTE);

	if (sctx->vertex_buffer_pointer_dirty) {
		struct radeon_cmdbuf *cs = sctx->gfx_cs;

		/* Find the location of the VB descriptor pointer. */
		/* TODO: In the future, the pointer will be packed in unused
		 *       bits of the first 2 VB descriptors. */
		unsigned sh_dw_offset = SI_VS_NUM_USER_SGPR;
		if (sctx->chip_class >= GFX9) {
			if (sctx->tes_shader.cso)
				sh_dw_offset = GFX9_TCS_NUM_USER_SGPR;
			else if (sctx->gs_shader.cso)
				sh_dw_offset = GFX9_VSGS_NUM_USER_SGPR;
		}

		unsigned sh_offset = sh_base[PIPE_SHADER_VERTEX] + sh_dw_offset * 4;
		si_emit_shader_pointer_head(cs, sh_offset, 1);
		si_emit_shader_pointer_body(sctx->screen, cs,
					    sctx->vb_descriptors_buffer->gpu_address +
					    sctx->vb_descriptors_offset);
		sctx->vertex_buffer_pointer_dirty = false;
	}

	if (sctx->graphics_bindless_pointer_dirty) {
		si_emit_global_shader_pointers(sctx,
					       &sctx->bindless_descriptors);
		sctx->graphics_bindless_pointer_dirty = false;
	}
}

void si_emit_compute_shader_pointers(struct si_context *sctx)
{
	unsigned base = R_00B900_COMPUTE_USER_DATA_0;

	si_emit_consecutive_shader_pointers(sctx, SI_DESCS_SHADER_MASK(COMPUTE),
					    R_00B900_COMPUTE_USER_DATA_0);
	sctx->shader_pointers_dirty &= ~SI_DESCS_SHADER_MASK(COMPUTE);

	if (sctx->compute_bindless_pointer_dirty) {
		si_emit_shader_pointer(sctx, &sctx->bindless_descriptors, base);
		sctx->compute_bindless_pointer_dirty = false;
	}
}

/* BINDLESS */

static void si_init_bindless_descriptors(struct si_context *sctx,
					 struct si_descriptors *desc,
					 short shader_userdata_rel_index,
					 unsigned num_elements)
{
	MAYBE_UNUSED unsigned desc_slot;

	si_init_descriptors(desc, shader_userdata_rel_index, 16, num_elements);
	sctx->bindless_descriptors.num_active_slots = num_elements;

	/* The first bindless descriptor is stored at slot 1, because 0 is not
	 * considered to be a valid handle.
	 */
	sctx->num_bindless_descriptors = 1;

	/* Track which bindless slots are used (or not). */
	util_idalloc_init(&sctx->bindless_used_slots);
	util_idalloc_resize(&sctx->bindless_used_slots, num_elements);

	/* Reserve slot 0 because it's an invalid handle for bindless. */
	desc_slot = util_idalloc_alloc(&sctx->bindless_used_slots);
	assert(desc_slot == 0);
}

static void si_release_bindless_descriptors(struct si_context *sctx)
{
	si_release_descriptors(&sctx->bindless_descriptors);
	util_idalloc_fini(&sctx->bindless_used_slots);
}

static unsigned si_get_first_free_bindless_slot(struct si_context *sctx)
{
	struct si_descriptors *desc = &sctx->bindless_descriptors;
	unsigned desc_slot;

	desc_slot = util_idalloc_alloc(&sctx->bindless_used_slots);
	if (desc_slot >= desc->num_elements) {
		/* The array of bindless descriptors is full, resize it. */
		unsigned slot_size = desc->element_dw_size * 4;
		unsigned new_num_elements = desc->num_elements * 2;

		desc->list = REALLOC(desc->list, desc->num_elements * slot_size,
				     new_num_elements * slot_size);
		desc->num_elements = new_num_elements;
		desc->num_active_slots = new_num_elements;
	}

	assert(desc_slot);
	return desc_slot;
}

static unsigned
si_create_bindless_descriptor(struct si_context *sctx, uint32_t *desc_list,
			      unsigned size)
{
	struct si_descriptors *desc = &sctx->bindless_descriptors;
	unsigned desc_slot, desc_slot_offset;

	/* Find a free slot. */
	desc_slot = si_get_first_free_bindless_slot(sctx);

	/* For simplicity, sampler and image bindless descriptors use fixed
	 * 16-dword slots for now. Image descriptors only need 8-dword but this
	 * doesn't really matter because no real apps use image handles.
	 */
	desc_slot_offset = desc_slot * 16;

	/* Copy the descriptor into the array. */
	memcpy(desc->list + desc_slot_offset, desc_list, size);

	/* Re-upload the whole array of bindless descriptors into a new buffer.
	 */
	if (!si_upload_descriptors(sctx, desc))
		return 0;

	/* Make sure to re-emit the shader pointers for all stages. */
	sctx->graphics_bindless_pointer_dirty = true;
	sctx->compute_bindless_pointer_dirty = true;

	return desc_slot;
}

static void si_update_bindless_buffer_descriptor(struct si_context *sctx,
						 unsigned desc_slot,
						 struct pipe_resource *resource,
						 uint64_t offset,
						 bool *desc_dirty)
{
	struct si_descriptors *desc = &sctx->bindless_descriptors;
	struct r600_resource *buf = r600_resource(resource);
	unsigned desc_slot_offset = desc_slot * 16;
	uint32_t *desc_list = desc->list + desc_slot_offset + 4;
	uint64_t old_desc_va;

	assert(resource->target == PIPE_BUFFER);

	/* Retrieve the old buffer addr from the descriptor. */
	old_desc_va = si_desc_extract_buffer_address(desc_list);

	if (old_desc_va != buf->gpu_address + offset) {
		/* The buffer has been invalidated when the handle wasn't
		 * resident, update the descriptor and the dirty flag.
		 */
		si_set_buf_desc_address(buf, offset, &desc_list[0]);

		*desc_dirty = true;
	}
}

static uint64_t si_create_texture_handle(struct pipe_context *ctx,
					 struct pipe_sampler_view *view,
					 const struct pipe_sampler_state *state)
{
	struct si_sampler_view *sview = (struct si_sampler_view *)view;
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_texture_handle *tex_handle;
	struct si_sampler_state *sstate;
	uint32_t desc_list[16];
	uint64_t handle;

	tex_handle = CALLOC_STRUCT(si_texture_handle);
	if (!tex_handle)
		return 0;

	memset(desc_list, 0, sizeof(desc_list));
	si_init_descriptor_list(&desc_list[0], 16, 1, null_texture_descriptor);

	sstate = ctx->create_sampler_state(ctx, state);
	if (!sstate) {
		FREE(tex_handle);
		return 0;
	}

	si_set_sampler_view_desc(sctx, sview, sstate, &desc_list[0]);
	memcpy(&tex_handle->sstate, sstate, sizeof(*sstate));
	ctx->delete_sampler_state(ctx, sstate);

	tex_handle->desc_slot = si_create_bindless_descriptor(sctx, desc_list,
							      sizeof(desc_list));
	if (!tex_handle->desc_slot) {
		FREE(tex_handle);
		return 0;
	}

	handle = tex_handle->desc_slot;

	if (!_mesa_hash_table_insert(sctx->tex_handles,
				     (void *)(uintptr_t)handle,
				     tex_handle)) {
		FREE(tex_handle);
		return 0;
	}

	pipe_sampler_view_reference(&tex_handle->view, view);

	r600_resource(sview->base.texture)->texture_handle_allocated = true;

	return handle;
}

static void si_delete_texture_handle(struct pipe_context *ctx, uint64_t handle)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_texture_handle *tex_handle;
	struct hash_entry *entry;

	entry = _mesa_hash_table_search(sctx->tex_handles,
					(void *)(uintptr_t)handle);
	if (!entry)
		return;

	tex_handle = (struct si_texture_handle *)entry->data;

	/* Allow this descriptor slot to be re-used. */
	util_idalloc_free(&sctx->bindless_used_slots, tex_handle->desc_slot);

	pipe_sampler_view_reference(&tex_handle->view, NULL);
	_mesa_hash_table_remove(sctx->tex_handles, entry);
	FREE(tex_handle);
}

static void si_make_texture_handle_resident(struct pipe_context *ctx,
					    uint64_t handle, bool resident)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_texture_handle *tex_handle;
	struct si_sampler_view *sview;
	struct hash_entry *entry;

	entry = _mesa_hash_table_search(sctx->tex_handles,
					(void *)(uintptr_t)handle);
	if (!entry)
		return;

	tex_handle = (struct si_texture_handle *)entry->data;
	sview = (struct si_sampler_view *)tex_handle->view;

	if (resident) {
		if (sview->base.texture->target != PIPE_BUFFER) {
			struct si_texture *tex =
				(struct si_texture *)sview->base.texture;

			if (depth_needs_decompression(tex)) {
				util_dynarray_append(
					&sctx->resident_tex_needs_depth_decompress,
					struct si_texture_handle *,
					tex_handle);
			}

			if (color_needs_decompression(tex)) {
				util_dynarray_append(
					&sctx->resident_tex_needs_color_decompress,
					struct si_texture_handle *,
					tex_handle);
			}

			if (tex->dcc_offset &&
			    p_atomic_read(&tex->framebuffers_bound))
				sctx->need_check_render_feedback = true;

			si_update_bindless_texture_descriptor(sctx, tex_handle);
		} else {
			si_update_bindless_buffer_descriptor(sctx,
							     tex_handle->desc_slot,
							     sview->base.texture,
							     sview->base.u.buf.offset,
							     &tex_handle->desc_dirty);
		}

		/* Re-upload the descriptor if it has been updated while it
		 * wasn't resident.
		 */
		if (tex_handle->desc_dirty)
			sctx->bindless_descriptors_dirty = true;

		/* Add the texture handle to the per-context list. */
		util_dynarray_append(&sctx->resident_tex_handles,
				     struct si_texture_handle *, tex_handle);

		/* Add the buffers to the current CS in case si_begin_new_cs()
		 * is not going to be called.
		 */
		si_sampler_view_add_buffer(sctx, sview->base.texture,
					   RADEON_USAGE_READ,
					   sview->is_stencil_sampler, false);
	} else {
		/* Remove the texture handle from the per-context list. */
		util_dynarray_delete_unordered(&sctx->resident_tex_handles,
					       struct si_texture_handle *,
					       tex_handle);

		if (sview->base.texture->target != PIPE_BUFFER) {
			util_dynarray_delete_unordered(
				&sctx->resident_tex_needs_depth_decompress,
				struct si_texture_handle *, tex_handle);

			util_dynarray_delete_unordered(
				&sctx->resident_tex_needs_color_decompress,
				struct si_texture_handle *, tex_handle);
		}
	}
}

static uint64_t si_create_image_handle(struct pipe_context *ctx,
				       const struct pipe_image_view *view)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_image_handle *img_handle;
	uint32_t desc_list[8];
	uint64_t handle;

	if (!view || !view->resource)
		return 0;

	img_handle = CALLOC_STRUCT(si_image_handle);
	if (!img_handle)
		return 0;

	memset(desc_list, 0, sizeof(desc_list));
	si_init_descriptor_list(&desc_list[0], 8, 1, null_image_descriptor);

	si_set_shader_image_desc(sctx, view, false, &desc_list[0], NULL);

	img_handle->desc_slot = si_create_bindless_descriptor(sctx, desc_list,
							      sizeof(desc_list));
	if (!img_handle->desc_slot) {
		FREE(img_handle);
		return 0;
	}

	handle = img_handle->desc_slot;

	if (!_mesa_hash_table_insert(sctx->img_handles,
				     (void *)(uintptr_t)handle,
				     img_handle)) {
		FREE(img_handle);
		return 0;
	}

	util_copy_image_view(&img_handle->view, view);

	r600_resource(view->resource)->image_handle_allocated = true;

	return handle;
}

static void si_delete_image_handle(struct pipe_context *ctx, uint64_t handle)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_image_handle *img_handle;
	struct hash_entry *entry;

	entry = _mesa_hash_table_search(sctx->img_handles,
					(void *)(uintptr_t)handle);
	if (!entry)
		return;

	img_handle = (struct si_image_handle *)entry->data;

	util_copy_image_view(&img_handle->view, NULL);
	_mesa_hash_table_remove(sctx->img_handles, entry);
	FREE(img_handle);
}

static void si_make_image_handle_resident(struct pipe_context *ctx,
					  uint64_t handle, unsigned access,
					  bool resident)
{
	struct si_context *sctx = (struct si_context *)ctx;
	struct si_image_handle *img_handle;
	struct pipe_image_view *view;
	struct r600_resource *res;
	struct hash_entry *entry;

	entry = _mesa_hash_table_search(sctx->img_handles,
					(void *)(uintptr_t)handle);
	if (!entry)
		return;

	img_handle = (struct si_image_handle *)entry->data;
	view = &img_handle->view;
	res = r600_resource(view->resource);

	if (resident) {
		if (res->b.b.target != PIPE_BUFFER) {
			struct si_texture *tex = (struct si_texture *)res;
			unsigned level = view->u.tex.level;

			if (color_needs_decompression(tex)) {
				util_dynarray_append(
					&sctx->resident_img_needs_color_decompress,
					struct si_image_handle *,
					img_handle);
			}

			if (vi_dcc_enabled(tex, level) &&
			    p_atomic_read(&tex->framebuffers_bound))
				sctx->need_check_render_feedback = true;

			si_update_bindless_image_descriptor(sctx, img_handle);
		} else {
			si_update_bindless_buffer_descriptor(sctx,
							     img_handle->desc_slot,
							     view->resource,
							     view->u.buf.offset,
							     &img_handle->desc_dirty);
		}

		/* Re-upload the descriptor if it has been updated while it
		 * wasn't resident.
		 */
		if (img_handle->desc_dirty)
			sctx->bindless_descriptors_dirty = true;

		/* Add the image handle to the per-context list. */
		util_dynarray_append(&sctx->resident_img_handles,
				     struct si_image_handle *, img_handle);

		/* Add the buffers to the current CS in case si_begin_new_cs()
		 * is not going to be called.
		 */
		si_sampler_view_add_buffer(sctx, view->resource,
					   (access & PIPE_IMAGE_ACCESS_WRITE) ?
					   RADEON_USAGE_READWRITE :
					   RADEON_USAGE_READ, false, false);
	} else {
		/* Remove the image handle from the per-context list. */
		util_dynarray_delete_unordered(&sctx->resident_img_handles,
					       struct si_image_handle *,
					       img_handle);

		if (res->b.b.target != PIPE_BUFFER) {
			util_dynarray_delete_unordered(
				&sctx->resident_img_needs_color_decompress,
				struct si_image_handle *,
				img_handle);
		}
	}
}


void si_all_resident_buffers_begin_new_cs(struct si_context *sctx)
{
	unsigned num_resident_tex_handles, num_resident_img_handles;

	num_resident_tex_handles = sctx->resident_tex_handles.size /
				   sizeof(struct si_texture_handle *);
	num_resident_img_handles = sctx->resident_img_handles.size /
				   sizeof(struct si_image_handle *);

	/* Add all resident texture handles. */
	util_dynarray_foreach(&sctx->resident_tex_handles,
			      struct si_texture_handle *, tex_handle) {
		struct si_sampler_view *sview =
			(struct si_sampler_view *)(*tex_handle)->view;

		si_sampler_view_add_buffer(sctx, sview->base.texture,
					   RADEON_USAGE_READ,
					   sview->is_stencil_sampler, false);
	}

	/* Add all resident image handles. */
	util_dynarray_foreach(&sctx->resident_img_handles,
			      struct si_image_handle *, img_handle) {
		struct pipe_image_view *view = &(*img_handle)->view;

		si_sampler_view_add_buffer(sctx, view->resource,
					   RADEON_USAGE_READWRITE,
					   false, false);
	}

	sctx->num_resident_handles += num_resident_tex_handles +
					num_resident_img_handles;
}

/* INIT/DEINIT/UPLOAD */

void si_init_all_descriptors(struct si_context *sctx)
{
	int i;

#if !HAVE_32BIT_POINTERS
	STATIC_ASSERT(GFX9_SGPR_2ND_SAMPLERS_AND_IMAGES % 2 == 0);
#endif

	for (i = 0; i < SI_NUM_SHADERS; i++) {
		bool is_2nd = sctx->chip_class >= GFX9 &&
				     (i == PIPE_SHADER_TESS_CTRL ||
				      i == PIPE_SHADER_GEOMETRY);
		unsigned num_sampler_slots = SI_NUM_IMAGES / 2 + SI_NUM_SAMPLERS;
		unsigned num_buffer_slots = SI_NUM_SHADER_BUFFERS + SI_NUM_CONST_BUFFERS;
		int rel_dw_offset;
		struct si_descriptors *desc;

		if (is_2nd) {
			if (i == PIPE_SHADER_TESS_CTRL) {
				rel_dw_offset = (R_00B408_SPI_SHADER_USER_DATA_ADDR_LO_HS -
						 R_00B430_SPI_SHADER_USER_DATA_LS_0) / 4;
			} else { /* PIPE_SHADER_GEOMETRY */
				rel_dw_offset = (R_00B208_SPI_SHADER_USER_DATA_ADDR_LO_GS -
						 R_00B330_SPI_SHADER_USER_DATA_ES_0) / 4;
			}
		} else {
			rel_dw_offset = SI_SGPR_CONST_AND_SHADER_BUFFERS;
		}
		desc = si_const_and_shader_buffer_descriptors(sctx, i);
		si_init_buffer_resources(&sctx->const_and_shader_buffers[i], desc,
					 num_buffer_slots, rel_dw_offset,
					 RADEON_USAGE_READWRITE,
					 RADEON_USAGE_READ,
					 RADEON_PRIO_SHADER_RW_BUFFER,
					 RADEON_PRIO_CONST_BUFFER);
		desc->slot_index_to_bind_directly = si_get_constbuf_slot(0);

		if (is_2nd) {
#if HAVE_32BIT_POINTERS
			if (i == PIPE_SHADER_TESS_CTRL) {
				rel_dw_offset = (R_00B40C_SPI_SHADER_USER_DATA_ADDR_HI_HS -
						 R_00B430_SPI_SHADER_USER_DATA_LS_0) / 4;
			} else { /* PIPE_SHADER_GEOMETRY */
				rel_dw_offset = (R_00B20C_SPI_SHADER_USER_DATA_ADDR_HI_GS -
						 R_00B330_SPI_SHADER_USER_DATA_ES_0) / 4;
			}
#else
			rel_dw_offset = GFX9_SGPR_2ND_SAMPLERS_AND_IMAGES;
#endif
		} else {
			rel_dw_offset = SI_SGPR_SAMPLERS_AND_IMAGES;
		}

		desc = si_sampler_and_image_descriptors(sctx, i);
		si_init_descriptors(desc, rel_dw_offset, 16, num_sampler_slots);

		int j;
		for (j = 0; j < SI_NUM_IMAGES; j++)
			memcpy(desc->list + j * 8, null_image_descriptor, 8 * 4);
		for (; j < SI_NUM_IMAGES + SI_NUM_SAMPLERS * 2; j++)
			memcpy(desc->list + j * 8, null_texture_descriptor, 8 * 4);
	}

	si_init_buffer_resources(&sctx->rw_buffers,
				 &sctx->descriptors[SI_DESCS_RW_BUFFERS],
				 SI_NUM_RW_BUFFERS, SI_SGPR_RW_BUFFERS,
				 /* The second set of usage/priority is used by
				  * const buffers in RW buffer slots. */
				 RADEON_USAGE_READWRITE, RADEON_USAGE_READ,
				 RADEON_PRIO_SHADER_RINGS, RADEON_PRIO_CONST_BUFFER);
	sctx->descriptors[SI_DESCS_RW_BUFFERS].num_active_slots = SI_NUM_RW_BUFFERS;

	/* Initialize an array of 1024 bindless descriptors, when the limit is
	 * reached, just make it larger and re-upload the whole array.
	 */
	si_init_bindless_descriptors(sctx, &sctx->bindless_descriptors,
				     SI_SGPR_BINDLESS_SAMPLERS_AND_IMAGES,
				     1024);

	sctx->descriptors_dirty = u_bit_consecutive(0, SI_NUM_DESCS);

	/* Set pipe_context functions. */
	sctx->b.bind_sampler_states = si_bind_sampler_states;
	sctx->b.set_shader_images = si_set_shader_images;
	sctx->b.set_constant_buffer = si_pipe_set_constant_buffer;
	sctx->b.set_polygon_stipple = si_set_polygon_stipple;
	sctx->b.set_shader_buffers = si_set_shader_buffers;
	sctx->b.set_sampler_views = si_set_sampler_views;
	sctx->b.create_texture_handle = si_create_texture_handle;
	sctx->b.delete_texture_handle = si_delete_texture_handle;
	sctx->b.make_texture_handle_resident = si_make_texture_handle_resident;
	sctx->b.create_image_handle = si_create_image_handle;
	sctx->b.delete_image_handle = si_delete_image_handle;
	sctx->b.make_image_handle_resident = si_make_image_handle_resident;

	/* Shader user data. */
	sctx->atoms.s.shader_pointers.emit = si_emit_graphics_shader_pointers;

	/* Set default and immutable mappings. */
	si_set_user_data_base(sctx, PIPE_SHADER_VERTEX, R_00B130_SPI_SHADER_USER_DATA_VS_0);

	if (sctx->chip_class >= GFX9) {
		si_set_user_data_base(sctx, PIPE_SHADER_TESS_CTRL,
				      R_00B430_SPI_SHADER_USER_DATA_LS_0);
		si_set_user_data_base(sctx, PIPE_SHADER_GEOMETRY,
				      R_00B330_SPI_SHADER_USER_DATA_ES_0);
	} else {
		si_set_user_data_base(sctx, PIPE_SHADER_TESS_CTRL,
				      R_00B430_SPI_SHADER_USER_DATA_HS_0);
		si_set_user_data_base(sctx, PIPE_SHADER_GEOMETRY,
				      R_00B230_SPI_SHADER_USER_DATA_GS_0);
	}
	si_set_user_data_base(sctx, PIPE_SHADER_FRAGMENT, R_00B030_SPI_SHADER_USER_DATA_PS_0);
}

static bool si_upload_shader_descriptors(struct si_context *sctx, unsigned mask)
{
	unsigned dirty = sctx->descriptors_dirty & mask;

	/* Assume nothing will go wrong: */
	sctx->shader_pointers_dirty |= dirty;

	while (dirty) {
		unsigned i = u_bit_scan(&dirty);

		if (!si_upload_descriptors(sctx, &sctx->descriptors[i]))
			return false;
	}

	sctx->descriptors_dirty &= ~mask;

	si_upload_bindless_descriptors(sctx);

	return true;
}

bool si_upload_graphics_shader_descriptors(struct si_context *sctx)
{
	const unsigned mask = u_bit_consecutive(0, SI_DESCS_FIRST_COMPUTE);
	return si_upload_shader_descriptors(sctx, mask);
}

bool si_upload_compute_shader_descriptors(struct si_context *sctx)
{
	/* Does not update rw_buffers as that is not needed for compute shaders
	 * and the input buffer is using the same SGPR's anyway.
	 */
	const unsigned mask = u_bit_consecutive(SI_DESCS_FIRST_COMPUTE,
						SI_NUM_DESCS - SI_DESCS_FIRST_COMPUTE);
	return si_upload_shader_descriptors(sctx, mask);
}

void si_release_all_descriptors(struct si_context *sctx)
{
	int i;

	for (i = 0; i < SI_NUM_SHADERS; i++) {
		si_release_buffer_resources(&sctx->const_and_shader_buffers[i],
					    si_const_and_shader_buffer_descriptors(sctx, i));
		si_release_sampler_views(&sctx->samplers[i]);
		si_release_image_views(&sctx->images[i]);
	}
	si_release_buffer_resources(&sctx->rw_buffers,
				    &sctx->descriptors[SI_DESCS_RW_BUFFERS]);
	for (i = 0; i < SI_NUM_VERTEX_BUFFERS; i++)
		pipe_vertex_buffer_unreference(&sctx->vertex_buffer[i]);

	for (i = 0; i < SI_NUM_DESCS; ++i)
		si_release_descriptors(&sctx->descriptors[i]);

	r600_resource_reference(&sctx->vb_descriptors_buffer, NULL);
	sctx->vb_descriptors_gpu_list = NULL; /* points into a mapped buffer */

	si_release_bindless_descriptors(sctx);
}

void si_all_descriptors_begin_new_cs(struct si_context *sctx)
{
	int i;

	for (i = 0; i < SI_NUM_SHADERS; i++) {
		si_buffer_resources_begin_new_cs(sctx, &sctx->const_and_shader_buffers[i]);
		si_sampler_views_begin_new_cs(sctx, &sctx->samplers[i]);
		si_image_views_begin_new_cs(sctx, &sctx->images[i]);
	}
	si_buffer_resources_begin_new_cs(sctx, &sctx->rw_buffers);
	si_vertex_buffers_begin_new_cs(sctx);

	for (i = 0; i < SI_NUM_DESCS; ++i)
		si_descriptors_begin_new_cs(sctx, &sctx->descriptors[i]);
	si_descriptors_begin_new_cs(sctx, &sctx->bindless_descriptors);

	si_shader_pointers_begin_new_cs(sctx);
}

void si_set_active_descriptors(struct si_context *sctx, unsigned desc_idx,
			       uint64_t new_active_mask)
{
	struct si_descriptors *desc = &sctx->descriptors[desc_idx];

	/* Ignore no-op updates and updates that disable all slots. */
	if (!new_active_mask ||
	    new_active_mask == u_bit_consecutive64(desc->first_active_slot,
						   desc->num_active_slots))
		return;

	int first, count;
	u_bit_scan_consecutive_range64(&new_active_mask, &first, &count);
	assert(new_active_mask == 0);

	/* Upload/dump descriptors if slots are being enabled. */
	if (first < desc->first_active_slot ||
	    first + count > desc->first_active_slot + desc->num_active_slots)
		sctx->descriptors_dirty |= 1u << desc_idx;

	desc->first_active_slot = first;
	desc->num_active_slots = count;
}

void si_set_active_descriptors_for_shader(struct si_context *sctx,
					  struct si_shader_selector *sel)
{
	if (!sel)
		return;

	si_set_active_descriptors(sctx,
		si_const_and_shader_buffer_descriptors_idx(sel->type),
		sel->active_const_and_shader_buffers);
	si_set_active_descriptors(sctx,
		si_sampler_and_image_descriptors_idx(sel->type),
		sel->active_samplers_and_images);
}
