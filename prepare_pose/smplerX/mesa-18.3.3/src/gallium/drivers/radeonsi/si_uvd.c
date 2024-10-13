/**************************************************************************
 *
 * Copyright 2011 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include "si_pipe.h"
#include "radeon/radeon_video.h"
#include "radeon/radeon_uvd.h"
#include "radeon/radeon_vce.h"
#include "radeon/radeon_vcn_dec.h"
#include "radeon/radeon_vcn_enc.h"
#include "radeon/radeon_uvd_enc.h"
#include "util/u_video.h"

/**
 * creates an video buffer with an UVD compatible memory layout
 */
struct pipe_video_buffer *si_video_buffer_create(struct pipe_context *pipe,
						 const struct pipe_video_buffer *tmpl)
{
	struct si_context *ctx = (struct si_context *)pipe;
	struct si_texture *resources[VL_NUM_COMPONENTS] = {};
	struct radeon_surf *surfaces[VL_NUM_COMPONENTS] = {};
	struct pb_buffer **pbs[VL_NUM_COMPONENTS] = {};
	const enum pipe_format *resource_formats;
	struct pipe_video_buffer vidtemplate;
	struct pipe_resource templ;
	unsigned i, array_size;

	assert(pipe);

	/* first create the needed resources as "normal" textures */
	resource_formats = vl_video_buffer_formats(pipe->screen, tmpl->buffer_format);
	if (!resource_formats)
		return NULL;

	array_size = tmpl->interlaced ? 2 : 1;
	vidtemplate = *tmpl;
	vidtemplate.width = align(tmpl->width, VL_MACROBLOCK_WIDTH);
	vidtemplate.height = align(tmpl->height / array_size, VL_MACROBLOCK_HEIGHT);

	assert(resource_formats[0] != PIPE_FORMAT_NONE);

	for (i = 0; i < VL_NUM_COMPONENTS; ++i) {
		if (resource_formats[i] != PIPE_FORMAT_NONE) {
			vl_video_buffer_template(&templ, &vidtemplate,
			                         resource_formats[i], 1,
			                         array_size, PIPE_USAGE_DEFAULT, i);
			/* Set PIPE_BIND_SHARED to avoid reallocation in si_texture_get_handle,
			 * which can't handle joined surfaces. */
			/* TODO: get tiling working */
			templ.bind = PIPE_BIND_LINEAR | PIPE_BIND_SHARED;
			resources[i] = (struct si_texture *)
			                pipe->screen->resource_create(pipe->screen, &templ);
			if (!resources[i])
				goto error;
		}
	}

	for (i = 0; i < VL_NUM_COMPONENTS; ++i) {
		if (!resources[i])
			continue;

		surfaces[i] = & resources[i]->surface;
		pbs[i] = &resources[i]->buffer.buf;
	}

	si_vid_join_surfaces(ctx, pbs, surfaces);

	for (i = 0; i < VL_NUM_COMPONENTS; ++i) {
		if (!resources[i])
			continue;

		/* reset the address */
		resources[i]->buffer.gpu_address = ctx->ws->buffer_get_virtual_address(
			resources[i]->buffer.buf);
	}

	vidtemplate.height *= array_size;
	return vl_video_buffer_create_ex2(pipe, &vidtemplate, (struct pipe_resource **)resources);

error:
	for (i = 0; i < VL_NUM_COMPONENTS; ++i)
		si_texture_reference(&resources[i], NULL);

	return NULL;
}

/* set the decoding target buffer offsets */
static struct pb_buffer* si_uvd_set_dtb(struct ruvd_msg *msg, struct vl_video_buffer *buf)
{
	struct si_screen *sscreen = (struct si_screen*)buf->base.context->screen;
	struct si_texture *luma = (struct si_texture *)buf->resources[0];
	struct si_texture *chroma = (struct si_texture *)buf->resources[1];
	enum ruvd_surface_type type =  (sscreen->info.chip_class >= GFX9) ?
					RUVD_SURFACE_TYPE_GFX9 :
					RUVD_SURFACE_TYPE_LEGACY;

	msg->body.decode.dt_field_mode = buf->base.interlaced;

	si_uvd_set_dt_surfaces(msg, &luma->surface, (chroma) ? &chroma->surface : NULL, type);

	return luma->buffer.buf;
}

/* get the radeon resources for VCE */
static void si_vce_get_buffer(struct pipe_resource *resource,
			      struct pb_buffer **handle,
			      struct radeon_surf **surface)
{
	struct si_texture *res = (struct si_texture *)resource;

	if (handle)
		*handle = res->buffer.buf;

	if (surface)
		*surface = &res->surface;
}

/**
 * creates an UVD compatible decoder
 */
struct pipe_video_codec *si_uvd_create_decoder(struct pipe_context *context,
					       const struct pipe_video_codec *templ)
{
	struct si_context *ctx = (struct si_context *)context;
	bool vcn = ctx->family == CHIP_RAVEN ||
		   ctx->family == CHIP_RAVEN2;

	if (templ->entrypoint == PIPE_VIDEO_ENTRYPOINT_ENCODE) {
		if (vcn) {
			return radeon_create_encoder(context, templ, ctx->ws, si_vce_get_buffer);
		} else {
			if (u_reduce_video_profile(templ->profile) == PIPE_VIDEO_FORMAT_HEVC)
				return radeon_uvd_create_encoder(context, templ, ctx->ws, si_vce_get_buffer);
			else
				return si_vce_create_encoder(context, templ, ctx->ws, si_vce_get_buffer);
		}
	}

	return (vcn) ? 	radeon_create_decoder(context, templ) :
		si_common_uvd_create_decoder(context, templ, si_uvd_set_dtb);
}
