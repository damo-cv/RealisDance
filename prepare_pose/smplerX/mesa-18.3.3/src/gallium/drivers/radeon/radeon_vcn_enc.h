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

#ifndef _RADEON_VCN_ENC_H
#define _RADEON_VCN_ENC_H

#define RENCODE_FW_INTERFACE_MAJOR_VERSION		1
#define RENCODE_FW_INTERFACE_MINOR_VERSION		2

#define RENCODE_IB_PARAM_SESSION_INFO				0x00000001
#define RENCODE_IB_PARAM_TASK_INFO  				0x00000002
#define RENCODE_IB_PARAM_SESSION_INIT				0x00000003
#define RENCODE_IB_PARAM_LAYER_CONTROL				0x00000004
#define RENCODE_IB_PARAM_LAYER_SELECT				0x00000005
#define RENCODE_IB_PARAM_RATE_CONTROL_SESSION_INIT 		0x00000006
#define RENCODE_IB_PARAM_RATE_CONTROL_LAYER_INIT	 	0x00000007
#define RENCODE_IB_PARAM_RATE_CONTROL_PER_PICTURE 		0x00000008
#define RENCODE_IB_PARAM_QUALITY_PARAMS 			0x00000009
#define RENCODE_IB_PARAM_SLICE_HEADER				0x0000000a
#define RENCODE_IB_PARAM_ENCODE_PARAMS				0x0000000b
#define RENCODE_IB_PARAM_INTRA_REFRESH				0x0000000c
#define RENCODE_IB_PARAM_ENCODE_CONTEXT_BUFFER  		0x0000000d
#define RENCODE_IB_PARAM_VIDEO_BITSTREAM_BUFFER 		0x0000000e
#define RENCODE_IB_PARAM_FEEDBACK_BUFFER			0x00000010
#define RENCODE_IB_PARAM_DIRECT_OUTPUT_NALU			0x00000020

#define RENCODE_HEVC_IB_PARAM_SLICE_CONTROL			0x00100001
#define RENCODE_HEVC_IB_PARAM_SPEC_MISC 			0x00100002
#define RENCODE_HEVC_IB_PARAM_DEBLOCKING_FILTER 		0x00100003

#define RENCODE_H264_IB_PARAM_SLICE_CONTROL			0x00200001
#define RENCODE_H264_IB_PARAM_SPEC_MISC 			0x00200002
#define RENCODE_H264_IB_PARAM_ENCODE_PARAMS			0x00200003
#define RENCODE_H264_IB_PARAM_DEBLOCKING_FILTER 		0x00200004

#define RENCODE_IB_OP_INITIALIZE    				0x01000001
#define RENCODE_IB_OP_CLOSE_SESSION 				0x01000002
#define RENCODE_IB_OP_ENCODE        				0x01000003
#define RENCODE_IB_OP_INIT_RC       				0x01000004
#define RENCODE_IB_OP_INIT_RC_VBV_BUFFER_LEVEL  		0x01000005
#define RENCODE_IB_OP_SET_SPEED_ENCODING_MODE   		0x01000006
#define RENCODE_IB_OP_SET_BALANCE_ENCODING_MODE 		0x01000007
#define RENCODE_IB_OP_SET_QUALITY_ENCODING_MODE 		0x01000008

#define RENCODE_IF_MAJOR_VERSION_MASK				0xFFFF0000
#define RENCODE_IF_MAJOR_VERSION_SHIFT				16
#define RENCODE_IF_MINOR_VERSION_MASK				0x0000FFFF
#define RENCODE_IF_MINOR_VERSION_SHIFT				0

#define RENCODE_ENCODE_STANDARD_HEVC				0
#define RENCODE_ENCODE_STANDARD_H264				1

#define RENCODE_PREENCODE_MODE_NONE 				0x00000000
#define RENCODE_PREENCODE_MODE_1X   				0x00000001
#define RENCODE_PREENCODE_MODE_2X   				0x00000002
#define RENCODE_PREENCODE_MODE_4X   				0x00000004

#define RENCODE_H264_SLICE_CONTROL_MODE_FIXED_MBS       	0x00000000
#define RENCODE_H264_SLICE_CONTROL_MODE_FIXED_BITS      	0x00000001

#define RENCODE_HEVC_SLICE_CONTROL_MODE_FIXED_CTBS       	0x00000000
#define RENCODE_HEVC_SLICE_CONTROL_MODE_FIXED_BITS      	0x00000001

#define RENCODE_RATE_CONTROL_METHOD_NONE        		0x00000000
#define RENCODE_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR  	0x00000001
#define RENCODE_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR 	0x00000002
#define RENCODE_RATE_CONTROL_METHOD_CBR         		0x00000003

#define RENCODE_DIRECT_OUTPUT_NALU_TYPE_AUD 			0x00000000
#define RENCODE_DIRECT_OUTPUT_NALU_TYPE_VPS 			0x00000001
#define RENCODE_DIRECT_OUTPUT_NALU_TYPE_SPS 			0x00000002
#define RENCODE_DIRECT_OUTPUT_NALU_TYPE_PPS 			0x00000003
#define RENCODE_DIRECT_OUTPUT_NALU_TYPE_PREFIX			0x00000004
#define RENCODE_DIRECT_OUTPUT_NALU_TYPE_END_OF_SEQUENCE 	0x00000005

#define RENCODE_SLICE_HEADER_TEMPLATE_MAX_TEMPLATE_SIZE_IN_DWORDS  	16
#define RENCODE_SLICE_HEADER_TEMPLATE_MAX_NUM_INSTRUCTIONS  		16

#define RENCODE_HEADER_INSTRUCTION_END  			0x00000000
#define RENCODE_HEADER_INSTRUCTION_COPY 			0x00000001

#define RENCODE_HEVC_HEADER_INSTRUCTION_DEPENDENT_SLICE_END	0x00010000
#define RENCODE_HEVC_HEADER_INSTRUCTION_FIRST_SLICE		0x00010001
#define RENCODE_HEVC_HEADER_INSTRUCTION_SLICE_SEGMENT		0x00010002
#define RENCODE_HEVC_HEADER_INSTRUCTION_SLICE_QP_DELTA		0x00010003

#define RENCODE_H264_HEADER_INSTRUCTION_FIRST_MB		0x00020000
#define RENCODE_H264_HEADER_INSTRUCTION_SLICE_QP_DELTA 	0x00020001

#define RENCODE_PICTURE_TYPE_B					0
#define RENCODE_PICTURE_TYPE_P					1
#define RENCODE_PICTURE_TYPE_I					2
#define RENCODE_PICTURE_TYPE_P_SKIP				3

#define RENCODE_INPUT_SWIZZLE_MODE_LINEAR			0
#define RENCODE_INPUT_SWIZZLE_MODE_256B_S			1
#define RENCODE_INPUT_SWIZZLE_MODE_4kB_S			5
#define RENCODE_INPUT_SWIZZLE_MODE_64kB_S			9

#define RENCODE_H264_PICTURE_STRUCTURE_FRAME			0
#define RENCODE_H264_PICTURE_STRUCTURE_TOP_FIELD		1
#define RENCODE_H264_PICTURE_STRUCTURE_BOTTOM_FIELD		2

#define RENCODE_H264_INTERLACING_MODE_PROGRESSIVE           	0
#define RENCODE_H264_INTERLACING_MODE_INTERLACED_STACKED    	1
#define RENCODE_H264_INTERLACING_MODE_INTERLACED_INTERLEAVED 	2

#define RENCODE_H264_DISABLE_DEBLOCKING_FILTER_IDC_ENABLE                       	0
#define RENCODE_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLE                      	1
#define RENCODE_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISALBE_ACROSS_SLICE_BOUNDARY 	2

#define RENCODE_INTRA_REFRESH_MODE_NONE     			0
#define RENCODE_INTRA_REFRESH_MODE_CTB_MB_ROWS			1
#define RENCODE_INTRA_REFRESH_MODE_CTB_MB_COLUMNS		2

#define RENCODE_MAX_NUM_RECONSTRUCTED_PICTURES			34

#define RENCODE_REC_SWIZZLE_MODE_LINEAR     			0
#define RENCODE_REC_SWIZZLE_MODE_256B_S     			1

#define RENCODE_VIDEO_BITSTREAM_BUFFER_MODE_LINEAR		0
#define RENCODE_VIDEO_BITSTREAM_BUFFER_MODE_CIRCULAR   	1

#define RENCODE_FEEDBACK_BUFFER_MODE_LINEAR 			0
#define RENCODE_FEEDBACK_BUFFER_MODE_CIRCULAR			1

typedef struct rvcn_enc_session_info_s
{
    uint32_t	interface_version;
    uint32_t	sw_context_address_hi;
    uint32_t	sw_context_address_lo;
} rvcn_enc_session_info_t;

typedef struct rvcn_enc_task_info_s
{
    uint32_t	total_size_of_all_packages;
    uint32_t	task_id;
    uint32_t	allowed_max_num_feedbacks;
} rvcn_enc_task_info_t;

typedef struct rvcn_enc_session_init_s
{
    uint32_t	encode_standard;
    uint32_t	aligned_picture_width;
    uint32_t	aligned_picture_height;
    uint32_t	padding_width;
    uint32_t	padding_height;
    uint32_t	pre_encode_mode;
    uint32_t	pre_encode_chroma_enabled;
} rvcn_enc_session_init_t;

typedef struct rvcn_enc_layer_control_s
{
    uint32_t	max_num_temporal_layers;
    uint32_t	num_temporal_layers;
} rvcn_enc_layer_control_t;

typedef struct rvcn_enc_layer_select_s
{
    uint32_t	temporal_layer_index;
} rvcn_enc_layer_select_t;

typedef struct rvcn_enc_h264_slice_control_s
{
    uint32_t	slice_control_mode;
    union
    {
        uint32_t	num_mbs_per_slice;
        uint32_t	num_bits_per_slice;
    };
} rvcn_enc_h264_slice_control_t;

typedef struct rvcn_enc_hevc_slice_control_s
{
    uint32_t	slice_control_mode;
    union
    {
        struct
        {
            uint32_t	num_ctbs_per_slice;
            uint32_t	num_ctbs_per_slice_segment;
        } fixed_ctbs_per_slice;

        struct
        {
            uint32_t	num_bits_per_slice;
            uint32_t	num_bits_per_slice_segment;
        } fixed_bits_per_slice;
    };
} rvcn_enc_hevc_slice_control_t;

typedef struct rvcn_enc_h264_spec_misc_s
{
    uint32_t	constrained_intra_pred_flag;
    uint32_t	cabac_enable;
    uint32_t	cabac_init_idc;
    uint32_t	half_pel_enabled;
    uint32_t	quarter_pel_enabled;
    uint32_t	profile_idc;
    uint32_t	level_idc;
} rvcn_enc_h264_spec_misc_t;

typedef struct rvcn_enc_hevc_spec_misc_s
{
    uint32_t	log2_min_luma_coding_block_size_minus3;
    uint32_t	amp_disabled;
    uint32_t	strong_intra_smoothing_enabled;
    uint32_t	constrained_intra_pred_flag;
    uint32_t	cabac_init_flag;
    uint32_t	half_pel_enabled;
    uint32_t	quarter_pel_enabled;
} rvcn_enc_hevc_spec_misc_t;

typedef struct rvcn_enc_rate_ctl_session_init_s
{
    uint32_t	rate_control_method;
    uint32_t	vbv_buffer_level;
} rvcn_enc_rate_ctl_session_init_t;

typedef struct rvcn_enc_rate_ctl_layer_init_s
{
    uint32_t	target_bit_rate;
    uint32_t	peak_bit_rate;
    uint32_t	frame_rate_num;
    uint32_t	frame_rate_den;
    uint32_t	vbv_buffer_size;
    uint32_t	avg_target_bits_per_picture;
    uint32_t	peak_bits_per_picture_integer;
    uint32_t	peak_bits_per_picture_fractional;
} rvcn_enc_rate_ctl_layer_init_t;

typedef struct rvcn_enc_rate_ctl_per_picture_s
{
    uint32_t	qp;
    uint32_t	min_qp_app;
    uint32_t	max_qp_app;
    uint32_t	max_au_size;
    uint32_t	enabled_filler_data;
    uint32_t	skip_frame_enable;
    uint32_t	enforce_hrd;
} rvcn_enc_rate_ctl_per_picture_t;

typedef struct rvcn_enc_quality_params_s
{
    uint32_t	vbaq_mode;
    uint32_t	scene_change_sensitivity;
    uint32_t	scene_change_min_idr_interval;
} rvcn_enc_quality_params_t;

typedef struct rvcn_enc_direct_output_nalu_s
{
    uint32_t	type;
    uint32_t	size;
    uint32_t	data[1];
} rvcn_enc_direct_output_nalu_t;

typedef struct rvcn_enc_slice_header_s
{
    uint32_t	bitstream_template[RENCODE_SLICE_HEADER_TEMPLATE_MAX_TEMPLATE_SIZE_IN_DWORDS];
    struct {
        uint32_t	instruction;
        uint32_t	num_bits;
    } instructions[RENCODE_SLICE_HEADER_TEMPLATE_MAX_NUM_INSTRUCTIONS];
} rvcn_enc_slice_header_t;

typedef struct rvcn_enc_encode_params_s
{
    uint32_t	pic_type;
    uint32_t	allowed_max_bitstream_size;
    uint32_t	input_picture_luma_address_hi;
    uint32_t	input_picture_luma_address_lo;
    uint32_t	input_picture_chroma_address_hi;
    uint32_t	input_picture_chroma_address_lo;
    uint32_t	input_pic_luma_pitch;
    uint32_t	input_pic_chroma_pitch;
    uint8_t	input_pic_swizzle_mode;
    uint32_t	reference_picture_index;
    uint32_t	reconstructed_picture_index;
} rvcn_enc_encode_params_t;

typedef struct rvcn_enc_h264_encode_params_s
{
    uint32_t	input_picture_structure;
    uint32_t	interlaced_mode;
    uint32_t	reference_picture_structure;
    uint32_t	reference_picture1_index;
} rvcn_enc_h264_encode_params_t;

typedef struct rvcn_enc_h264_deblocking_filter_s
{
    uint32_t	disable_deblocking_filter_idc;
    int32_t 	alpha_c0_offset_div2;
    int32_t 	beta_offset_div2;
    int32_t 	cb_qp_offset;
    int32_t 	cr_qp_offset;
} rvcn_enc_h264_deblocking_filter_t;

typedef struct rvcn_enc_hevc_deblocking_filter_s
{
    uint32_t	loop_filter_across_slices_enabled;
    int32_t 	deblocking_filter_disabled;
    int32_t 	beta_offset_div2;
    int32_t 	tc_offset_div2;
    int32_t 	cb_qp_offset;
    int32_t 	cr_qp_offset;
} rvcn_enc_hevc_deblocking_filter_t;

typedef struct rvcn_enc_intra_refresh_s
{
    uint32_t	intra_refresh_mode;
    uint32_t	offset;
    uint32_t	region_size;
} rvcn_enc_intra_refresh_t;

typedef struct rvcn_enc_reconstructed_picture_s
{
    uint32_t	luma_offset;
    uint32_t	chroma_offset;
} rvcn_enc_reconstructed_picture_t;

typedef struct rvcn_enc_encode_context_buffer_s
{
    uint32_t	encode_context_address_hi;
    uint32_t	encode_context_address_lo;
    uint32_t	swizzle_mode;
    uint32_t	rec_luma_pitch;
    uint32_t	rec_chroma_pitch;
    uint32_t	num_reconstructed_pictures;
    rvcn_enc_reconstructed_picture_t	reconstructed_pictures[RENCODE_MAX_NUM_RECONSTRUCTED_PICTURES];
    uint32_t	pre_encode_picture_luma_pitch;
    uint32_t	pre_encode_picture_chroma_pitch;
    rvcn_enc_reconstructed_picture_t	pre_encode_reconstructed_pictures[RENCODE_MAX_NUM_RECONSTRUCTED_PICTURES];
    rvcn_enc_reconstructed_picture_t	pre_encode_input_picture;
} rvcn_enc_encode_context_buffer_t;

typedef struct rvcn_enc_video_bitstream_buffer_s
{
    uint32_t	mode;
    uint32_t	video_bitstream_buffer_address_hi;
    uint32_t	video_bitstream_buffer_address_lo;
    uint32_t	video_bitstream_buffer_size;
    uint32_t	video_bitstream_data_offset;
} rvcn_enc_video_bitstream_buffer_t;

typedef struct rvcn_enc_feedback_buffer_s
{
    uint32_t	mode;
    uint32_t	feedback_buffer_address_hi;
    uint32_t	feedback_buffer_address_lo;
    uint32_t	feedback_buffer_size;
    uint32_t	feedback_data_size;
} rvcn_enc_feedback_buffer_t;

typedef void (*radeon_enc_get_buffer)(struct pipe_resource *resource,
		struct pb_buffer **handle,
		struct radeon_surf **surface);

struct pipe_video_codec *radeon_create_encoder(struct pipe_context *context,
		const struct pipe_video_codec *templat,
		struct radeon_winsys* ws,
		radeon_enc_get_buffer get_buffer);

struct radeon_enc_pic {
	enum	pipe_h264_enc_picture_type picture_type;

	unsigned	frame_num;
	unsigned	pic_order_cnt;
	unsigned	pic_order_cnt_type;
	unsigned	ref_idx_l0;
	unsigned	ref_idx_l1;
	unsigned	crop_left;
	unsigned	crop_right;
	unsigned	crop_top;
	unsigned	crop_bottom;
	unsigned	general_tier_flag;
	unsigned	general_profile_idc;
	unsigned	general_level_idc;
	unsigned	max_poc;
	unsigned	log2_max_poc;
	unsigned	chroma_format_idc;
	unsigned	pic_width_in_luma_samples;
	unsigned	pic_height_in_luma_samples;
	unsigned	log2_diff_max_min_luma_coding_block_size;
	unsigned	log2_min_transform_block_size_minus2;
	unsigned	log2_diff_max_min_transform_block_size;
	unsigned	max_transform_hierarchy_depth_inter;
	unsigned	max_transform_hierarchy_depth_intra;
	unsigned	log2_parallel_merge_level_minus2;
	unsigned	bit_depth_luma_minus8;
	unsigned	bit_depth_chroma_minus8;
	unsigned	nal_unit_type;
	unsigned	max_num_merge_cand;

	bool	not_referenced;
	bool	is_idr;
	bool	is_even_frame;
	bool	sample_adaptive_offset_enabled_flag;
	bool	pcm_enabled_flag;
	bool	sps_temporal_mvp_enabled_flag;

	rvcn_enc_task_info_t	task_info;
	rvcn_enc_session_init_t	session_init;
	rvcn_enc_layer_control_t	layer_ctrl;
	rvcn_enc_layer_select_t	layer_sel;
	rvcn_enc_h264_slice_control_t	slice_ctrl;
	rvcn_enc_hevc_slice_control_t	hevc_slice_ctrl;
	rvcn_enc_h264_spec_misc_t	spec_misc;
	rvcn_enc_hevc_spec_misc_t	hevc_spec_misc;
	rvcn_enc_rate_ctl_session_init_t	rc_session_init;
	rvcn_enc_rate_ctl_layer_init_t	rc_layer_init;
	rvcn_enc_h264_encode_params_t	h264_enc_params;
	rvcn_enc_h264_deblocking_filter_t	h264_deblock;
	rvcn_enc_hevc_deblocking_filter_t	hevc_deblock;
	rvcn_enc_rate_ctl_per_picture_t	rc_per_pic;
	rvcn_enc_quality_params_t	quality_params;
	rvcn_enc_encode_context_buffer_t	ctx_buf;
	rvcn_enc_video_bitstream_buffer_t	bit_buf;
	rvcn_enc_feedback_buffer_t	fb_buf;
	rvcn_enc_intra_refresh_t	intra_ref;
	rvcn_enc_encode_params_t	enc_params;
};

struct radeon_encoder {
	struct pipe_video_codec		base;

	void (*begin)(struct radeon_encoder *enc, struct pipe_picture_desc *pic);
	void (*encode)(struct radeon_encoder *enc);
	void (*destroy)(struct radeon_encoder *enc);

	unsigned			stream_handle;

	struct pipe_screen		*screen;
	struct radeon_winsys*		ws;
	struct radeon_cmdbuf*	cs;

	radeon_enc_get_buffer			get_buffer;

	struct pb_buffer*	handle;
	struct radeon_surf*		luma;
	struct radeon_surf*		chroma;

	struct pb_buffer*	bs_handle;
	unsigned			bs_size;

	unsigned			cpb_num;

	struct rvid_buffer		*si;
	struct rvid_buffer		*fb;
	struct rvid_buffer		cpb;
	struct radeon_enc_pic	enc_pic;

	unsigned			alignment;
	unsigned			shifter;
	unsigned			bits_in_shifter;
	unsigned			num_zeros;
	unsigned			byte_index;
	unsigned			bits_output;
	uint32_t			total_task_size;
	uint32_t*			p_task_size;

	bool				emulation_prevention;
	bool				need_feedback;
};

void radeon_enc_1_2_init(struct radeon_encoder *enc);

#endif  // _RADEON_VCN_ENC_H
