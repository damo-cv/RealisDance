/**************************************************************************
 *
 * Copyright 2017 Advanced Micro Devices, Inc.
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

#include <stdio.h>

#include "pipe/p_video_codec.h"

#include "util/u_video.h"
#include "util/u_memory.h"

#include "vl/vl_video_buffer.h"

#include "radeonsi/si_pipe.h"
#include "radeon_video.h"
#include "radeon_vcn_enc.h"

static void radeon_vcn_enc_get_param(struct radeon_encoder *enc, struct pipe_picture_desc *picture)
{
   if (u_reduce_video_profile(picture->profile) == PIPE_VIDEO_FORMAT_MPEG4_AVC) {
      struct pipe_h264_enc_picture_desc *pic = (struct pipe_h264_enc_picture_desc *)picture;
      enc->enc_pic.picture_type = pic->picture_type;
      enc->enc_pic.frame_num = pic->frame_num;
      enc->enc_pic.pic_order_cnt = pic->pic_order_cnt;
      enc->enc_pic.pic_order_cnt_type = pic->pic_order_cnt_type;
      enc->enc_pic.ref_idx_l0 = pic->ref_idx_l0;
      enc->enc_pic.ref_idx_l1 = pic->ref_idx_l1;
      enc->enc_pic.not_referenced = pic->not_referenced;
      enc->enc_pic.is_idr = (pic->picture_type == PIPE_H264_ENC_PICTURE_TYPE_IDR);
      enc->enc_pic.crop_left = 0;
      enc->enc_pic.crop_right = (align(enc->base.width, 16) - enc->base.width) / 2;
      enc->enc_pic.crop_top = 0;
      enc->enc_pic.crop_bottom = (align(enc->base.height, 16) - enc->base.height) / 2;
   } else if (u_reduce_video_profile(picture->profile) == PIPE_VIDEO_FORMAT_HEVC) {
      struct pipe_h265_enc_picture_desc *pic = (struct pipe_h265_enc_picture_desc *)picture;
      enc->enc_pic.picture_type = pic->picture_type;
      enc->enc_pic.frame_num = pic->frame_num;
      enc->enc_pic.pic_order_cnt = pic->pic_order_cnt;
      enc->enc_pic.pic_order_cnt_type = pic->pic_order_cnt_type;
      enc->enc_pic.ref_idx_l0 = pic->ref_idx_l0;
      enc->enc_pic.ref_idx_l1 = pic->ref_idx_l1;
      enc->enc_pic.not_referenced = pic->not_referenced;
      enc->enc_pic.is_idr = (pic->picture_type == PIPE_H265_ENC_PICTURE_TYPE_IDR) ||
                            (pic->picture_type == PIPE_H265_ENC_PICTURE_TYPE_I);
      enc->enc_pic.crop_left = 0;
      enc->enc_pic.crop_right = (align(enc->base.width, 16) - enc->base.width) / 2;
      enc->enc_pic.crop_top = 0;
      enc->enc_pic.crop_bottom = (align(enc->base.height, 16) - enc->base.height) / 2;
      enc->enc_pic.general_tier_flag = pic->seq.general_tier_flag;
      enc->enc_pic.general_profile_idc = pic->seq.general_profile_idc;
      enc->enc_pic.general_level_idc = pic->seq.general_level_idc;
      enc->enc_pic.max_poc = pic->seq.intra_period;
      enc->enc_pic.log2_max_poc = 0;
      for (int i = enc->enc_pic.max_poc; i != 0; enc->enc_pic.log2_max_poc++)
         i = (i >> 1);
      enc->enc_pic.chroma_format_idc = pic->seq.chroma_format_idc;
      enc->enc_pic.pic_width_in_luma_samples = pic->seq.pic_width_in_luma_samples;
      enc->enc_pic.pic_height_in_luma_samples = pic->seq.pic_height_in_luma_samples;
      enc->enc_pic.log2_diff_max_min_luma_coding_block_size = pic->seq.log2_diff_max_min_luma_coding_block_size;
      enc->enc_pic.log2_min_transform_block_size_minus2 = pic->seq.log2_min_transform_block_size_minus2;
      enc->enc_pic.log2_diff_max_min_transform_block_size = pic->seq.log2_diff_max_min_transform_block_size;
      enc->enc_pic.max_transform_hierarchy_depth_inter = pic->seq.max_transform_hierarchy_depth_inter;
      enc->enc_pic.max_transform_hierarchy_depth_intra = pic->seq.max_transform_hierarchy_depth_intra;
      enc->enc_pic.log2_parallel_merge_level_minus2 = pic->pic.log2_parallel_merge_level_minus2;
      enc->enc_pic.bit_depth_luma_minus8 = pic->seq.bit_depth_luma_minus8;
      enc->enc_pic.bit_depth_chroma_minus8 = pic->seq.bit_depth_chroma_minus8;
      enc->enc_pic.nal_unit_type = pic->pic.nal_unit_type;
      enc->enc_pic.max_num_merge_cand = pic->slice.max_num_merge_cand;
      enc->enc_pic.sample_adaptive_offset_enabled_flag = pic->seq.sample_adaptive_offset_enabled_flag;
      enc->enc_pic.pcm_enabled_flag = pic->seq.pcm_enabled_flag;
      enc->enc_pic.sps_temporal_mvp_enabled_flag = pic->seq.sps_temporal_mvp_enabled_flag;
   }
}

static void flush(struct radeon_encoder *enc)
{
	enc->ws->cs_flush(enc->cs, PIPE_FLUSH_ASYNC, NULL);
}

static void radeon_enc_flush(struct pipe_video_codec *encoder)
{
	struct radeon_encoder *enc = (struct radeon_encoder*)encoder;
	flush(enc);
}

static void radeon_enc_cs_flush(void *ctx, unsigned flags,
								struct pipe_fence_handle **fence)
{
	// just ignored
}

static unsigned get_cpb_num(struct radeon_encoder *enc)
{
	unsigned w = align(enc->base.width, 16) / 16;
	unsigned h = align(enc->base.height, 16) / 16;
	unsigned dpb;

	switch (enc->base.level) {
	case 10:
		dpb = 396;
		break;
	case 11:
		dpb = 900;
		break;
	case 12:
	case 13:
	case 20:
		dpb = 2376;
		break;
	case 21:
		dpb = 4752;
		break;
	case 22:
	case 30:
		dpb = 8100;
		break;
	case 31:
		dpb = 18000;
		break;
	case 32:
		dpb = 20480;
		break;
	case 40:
	case 41:
		dpb = 32768;
		break;
	case 42:
		dpb = 34816;
		break;
	case 50:
		dpb = 110400;
		break;
	default:
	case 51:
	case 52:
		dpb = 184320;
		break;
	}

	return MIN2(dpb / (w * h), 16);
}

static void radeon_enc_begin_frame(struct pipe_video_codec *encoder,
							 struct pipe_video_buffer *source,
							 struct pipe_picture_desc *picture)
{
	struct radeon_encoder *enc = (struct radeon_encoder*)encoder;
	struct vl_video_buffer *vid_buf = (struct vl_video_buffer *)source;

	radeon_vcn_enc_get_param(enc, picture);

	enc->get_buffer(vid_buf->resources[0], &enc->handle, &enc->luma);
	enc->get_buffer(vid_buf->resources[1], NULL, &enc->chroma);

	enc->need_feedback = false;

	if (!enc->stream_handle) {
		struct rvid_buffer fb;
		enc->stream_handle = si_vid_alloc_stream_handle();
		enc->si = CALLOC_STRUCT(rvid_buffer);
		si_vid_create_buffer(enc->screen, enc->si, 128 * 1024, PIPE_USAGE_STAGING);
		si_vid_create_buffer(enc->screen, &fb, 4096, PIPE_USAGE_STAGING);
		enc->fb = &fb;
		enc->begin(enc, picture);
		flush(enc);
		si_vid_destroy_buffer(&fb);
	}
}

static void radeon_enc_encode_bitstream(struct pipe_video_codec *encoder,
								  struct pipe_video_buffer *source,
								  struct pipe_resource *destination,
								  void **fb)
{
	struct radeon_encoder *enc = (struct radeon_encoder*)encoder;
	enc->get_buffer(destination, &enc->bs_handle, NULL);
	enc->bs_size = destination->width0;

	*fb = enc->fb = CALLOC_STRUCT(rvid_buffer);

	if (!si_vid_create_buffer(enc->screen, enc->fb, 4096, PIPE_USAGE_STAGING)) {
		RVID_ERR("Can't create feedback buffer.\n");
		return;
	}

	enc->need_feedback = true;
	enc->encode(enc);
}

static void radeon_enc_end_frame(struct pipe_video_codec *encoder,
						   struct pipe_video_buffer *source,
						   struct pipe_picture_desc *picture)
{
	struct radeon_encoder *enc = (struct radeon_encoder*)encoder;
	flush(enc);
}

static void radeon_enc_destroy(struct pipe_video_codec *encoder)
{
	struct radeon_encoder *enc = (struct radeon_encoder*)encoder;

	if (enc->stream_handle) {
		struct rvid_buffer fb;
		enc->need_feedback = false;
		si_vid_create_buffer(enc->screen, &fb, 512, PIPE_USAGE_STAGING);
		enc->fb = &fb;
		enc->destroy(enc);
		flush(enc);
		si_vid_destroy_buffer(&fb);
	}

	si_vid_destroy_buffer(&enc->cpb);
	enc->ws->cs_destroy(enc->cs);
	FREE(enc);
}

static void radeon_enc_get_feedback(struct pipe_video_codec *encoder,
							  void *feedback, unsigned *size)
{
	struct radeon_encoder *enc = (struct radeon_encoder*)encoder;
	struct rvid_buffer *fb = feedback;

	if (size) {
		uint32_t *ptr = enc->ws->buffer_map(fb->res->buf, enc->cs, PIPE_TRANSFER_READ_WRITE);
		if (ptr[1])
			*size = ptr[6];
		else
			*size = 0;
		enc->ws->buffer_unmap(fb->res->buf);
	}

	si_vid_destroy_buffer(fb);
	FREE(fb);
}

struct pipe_video_codec *radeon_create_encoder(struct pipe_context *context,
		const struct pipe_video_codec *templ,
		struct radeon_winsys* ws,
		radeon_enc_get_buffer get_buffer)
{
	struct si_screen *sscreen = (struct si_screen *)context->screen;
	struct si_context *sctx = (struct si_context*)context;
	struct radeon_encoder *enc;
	struct pipe_video_buffer *tmp_buf, templat = {};
	struct radeon_surf *tmp_surf;
	unsigned cpb_size;

	enc = CALLOC_STRUCT(radeon_encoder);

	if (!enc)
		return NULL;

	enc->alignment = 256;
	enc->base = *templ;
	enc->base.context = context;
	enc->base.destroy = radeon_enc_destroy;
	enc->base.begin_frame = radeon_enc_begin_frame;
	enc->base.encode_bitstream = radeon_enc_encode_bitstream;
	enc->base.end_frame = radeon_enc_end_frame;
	enc->base.flush = radeon_enc_flush;
	enc->base.get_feedback = radeon_enc_get_feedback;
	enc->get_buffer = get_buffer;
	enc->bits_in_shifter = 0;
	enc->screen = context->screen;
	enc->ws = ws;
	enc->cs = ws->cs_create(sctx->ctx, RING_VCN_ENC, radeon_enc_cs_flush, enc);

	if (!enc->cs) {
		RVID_ERR("Can't get command submission context.\n");
		goto error;
	}

	struct rvid_buffer si;
	si_vid_create_buffer(enc->screen, &si, 128 * 1024, PIPE_USAGE_STAGING);
	enc->si = &si;

	templat.buffer_format = PIPE_FORMAT_NV12;
	templat.chroma_format = PIPE_VIDEO_CHROMA_FORMAT_420;
	templat.width = enc->base.width;
	templat.height = enc->base.height;
	templat.interlaced = false;

	if (!(tmp_buf = context->create_video_buffer(context, &templat))) {
		RVID_ERR("Can't create video buffer.\n");
		goto error;
	}

	enc->cpb_num = get_cpb_num(enc);

	if (!enc->cpb_num)
		goto error;

	get_buffer(((struct vl_video_buffer *)tmp_buf)->resources[0], NULL, &tmp_surf);

	cpb_size = (sscreen->info.chip_class < GFX9) ?
			   align(tmp_surf->u.legacy.level[0].nblk_x * tmp_surf->bpe, 128) *
			   align(tmp_surf->u.legacy.level[0].nblk_y, 32) :
			   align(tmp_surf->u.gfx9.surf_pitch * tmp_surf->bpe, 256) *
			   align(tmp_surf->u.gfx9.surf_height, 32);

	cpb_size = cpb_size * 3 / 2;
	cpb_size = cpb_size * enc->cpb_num;
	tmp_buf->destroy(tmp_buf);

	if (!si_vid_create_buffer(enc->screen, &enc->cpb, cpb_size, PIPE_USAGE_DEFAULT)) {
		RVID_ERR("Can't create CPB buffer.\n");
		goto error;
	}

	radeon_enc_1_2_init(enc);

	return &enc->base;

error:
	if (enc->cs)
		enc->ws->cs_destroy(enc->cs);

	si_vid_destroy_buffer(&enc->cpb);

	FREE(enc);
	return NULL;
}
