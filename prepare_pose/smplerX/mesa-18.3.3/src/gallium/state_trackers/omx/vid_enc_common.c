/**************************************************************************
 *
 * Copyright 2013 Advanced Micro Devices, Inc.
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

#include "vid_enc_common.h"

#include "vl/vl_video_buffer.h"

void enc_ReleaseTasks(struct list_head *head)
{
   struct encode_task *i, *next;

   if (!head || !head->next)
      return;

   LIST_FOR_EACH_ENTRY_SAFE(i, next, head, list) {
      pipe_resource_reference(&i->bitstream, NULL);
      i->buf->destroy(i->buf);
      FREE(i);
   }
}

void enc_MoveTasks(struct list_head *from, struct list_head *to)
{
   to->prev->next = from->next;
   from->next->prev = to->prev;
   from->prev->next = to;
   to->prev = from->prev;
   LIST_INITHEAD(from);
}

static void enc_GetPictureParamPreset(struct pipe_h264_enc_picture_desc *picture)
{
   picture->motion_est.enc_disable_sub_mode = 0x000000fe;
   picture->motion_est.enc_ime2_search_range_x = 0x00000001;
   picture->motion_est.enc_ime2_search_range_y = 0x00000001;
   picture->pic_ctrl.enc_constraint_set_flags = 0x00000040;
}

enum pipe_video_profile enc_TranslateOMXProfileToPipe(unsigned omx_profile)
{
   switch (omx_profile) {
   case OMX_VIDEO_AVCProfileBaseline:
      return PIPE_VIDEO_PROFILE_MPEG4_AVC_BASELINE;
   case OMX_VIDEO_AVCProfileMain:
      return PIPE_VIDEO_PROFILE_MPEG4_AVC_MAIN;
   case OMX_VIDEO_AVCProfileExtended:
      return PIPE_VIDEO_PROFILE_MPEG4_AVC_EXTENDED;
   case OMX_VIDEO_AVCProfileHigh:
      return PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH;
   case OMX_VIDEO_AVCProfileHigh10:
      return PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH10;
   case OMX_VIDEO_AVCProfileHigh422:
      return PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH422;
   case OMX_VIDEO_AVCProfileHigh444:
      return PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH444;
   default:
      return PIPE_VIDEO_PROFILE_UNKNOWN;
   }
}

unsigned enc_TranslateOMXLevelToPipe(unsigned omx_level)
{
   switch (omx_level) {
   case OMX_VIDEO_AVCLevel1:
   case OMX_VIDEO_AVCLevel1b:
      return 10;
   case OMX_VIDEO_AVCLevel11:
      return 11;
   case OMX_VIDEO_AVCLevel12:
      return 12;
   case OMX_VIDEO_AVCLevel13:
      return 13;
   case OMX_VIDEO_AVCLevel2:
      return 20;
   case OMX_VIDEO_AVCLevel21:
      return 21;
   case OMX_VIDEO_AVCLevel22:
      return 22;
   case OMX_VIDEO_AVCLevel3:
      return 30;
   case OMX_VIDEO_AVCLevel31:
      return 31;
   case OMX_VIDEO_AVCLevel32:
      return 32;
   case OMX_VIDEO_AVCLevel4:
      return 40;
   case OMX_VIDEO_AVCLevel41:
      return 41;
   default:
   case OMX_VIDEO_AVCLevel42:
      return 42;
   case OMX_VIDEO_AVCLevel5:
      return 50;
   case OMX_VIDEO_AVCLevel51:
      return 51;
   }
}

void vid_enc_BufferEncoded_common(vid_enc_PrivateType * priv, OMX_BUFFERHEADERTYPE* input, OMX_BUFFERHEADERTYPE* output)
{
   struct output_buf_private *outp = output->pOutputPortPrivate;
   struct input_buf_private *inp = input->pInputPortPrivate;
   struct encode_task *task;
   struct pipe_box box = {};
   unsigned size;

#if ENABLE_ST_OMX_BELLAGIO
   if (!inp || LIST_IS_EMPTY(&inp->tasks)) {
      input->nFilledLen = 0; /* mark buffer as empty */
      enc_MoveTasks(&priv->used_tasks, &inp->tasks);
      return;
   }
#endif

   task = LIST_ENTRY(struct encode_task, inp->tasks.next, list);
   LIST_DEL(&task->list);
   LIST_ADDTAIL(&task->list, &priv->used_tasks);

   if (!task->bitstream)
      return;

   /* ------------- map result buffer ----------------- */

   if (outp->transfer)
      pipe_transfer_unmap(priv->t_pipe, outp->transfer);

   pipe_resource_reference(&outp->bitstream, task->bitstream);
   pipe_resource_reference(&task->bitstream, NULL);

   box.width = outp->bitstream->width0;
   box.height = outp->bitstream->height0;
   box.depth = outp->bitstream->depth0;

   output->pBuffer = priv->t_pipe->transfer_map(priv->t_pipe, outp->bitstream, 0,
                                                PIPE_TRANSFER_READ_WRITE,
                                                &box, &outp->transfer);

   /* ------------- get size of result ----------------- */

   priv->codec->get_feedback(priv->codec, task->feedback, &size);

   output->nOffset = 0;
   output->nFilledLen = size; /* mark buffer as full */

   /* all output buffers contain exactly one frame */
   output->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;

#if ENABLE_ST_OMX_TIZONIA
   input->nFilledLen = 0; /* mark buffer as empty */
   enc_MoveTasks(&priv->used_tasks, &inp->tasks);
#endif
}


struct encode_task *enc_NeedTask_common(vid_enc_PrivateType * priv, OMX_VIDEO_PORTDEFINITIONTYPE *def)
{
   struct pipe_video_buffer templat = {};
   struct encode_task *task;

   if (!LIST_IS_EMPTY(&priv->free_tasks)) {
      task = LIST_ENTRY(struct encode_task, priv->free_tasks.next, list);
      LIST_DEL(&task->list);
      return task;
   }

   /* allocate a new one */
   task = CALLOC_STRUCT(encode_task);
   if (!task)
      return NULL;

   templat.buffer_format = PIPE_FORMAT_NV12;
   templat.chroma_format = PIPE_VIDEO_CHROMA_FORMAT_420;
   templat.width = def->nFrameWidth;
   templat.height = def->nFrameHeight;
   templat.interlaced = false;

   task->buf = priv->s_pipe->create_video_buffer(priv->s_pipe, &templat);
   if (!task->buf) {
      FREE(task);
      return NULL;
   }

   return task;
}

void enc_ScaleInput_common(vid_enc_PrivateType * priv, OMX_VIDEO_PORTDEFINITIONTYPE *def,
                                  struct pipe_video_buffer **vbuf, unsigned *size)
{
   struct pipe_video_buffer *src_buf = *vbuf;
   struct vl_compositor *compositor = &priv->compositor;
   struct vl_compositor_state *s = &priv->cstate;
   struct pipe_sampler_view **views;
   struct pipe_surface **dst_surface;
   unsigned i;

   if (!priv->scale_buffer[priv->current_scale_buffer])
      return;

   views = src_buf->get_sampler_view_planes(src_buf);
   dst_surface = priv->scale_buffer[priv->current_scale_buffer]->get_surfaces
                 (priv->scale_buffer[priv->current_scale_buffer]);
   vl_compositor_clear_layers(s);

   for (i = 0; i < VL_MAX_SURFACES; ++i) {
      struct u_rect src_rect;
      if (!views[i] || !dst_surface[i])
         continue;
      src_rect.x0 = 0;
      src_rect.y0 = 0;
      src_rect.x1 = def->nFrameWidth;
      src_rect.y1 = def->nFrameHeight;
      if (i > 0) {
         src_rect.x1 /= 2;
         src_rect.y1 /= 2;
      }
      vl_compositor_set_rgba_layer(s, compositor, 0, views[i], &src_rect, NULL, NULL);
      vl_compositor_render(s, compositor, dst_surface[i], NULL, false);
   }
   *size  = priv->scale.xWidth * priv->scale.xHeight * 2;
   *vbuf = priv->scale_buffer[priv->current_scale_buffer++];
   priv->current_scale_buffer %= OMX_VID_ENC_NUM_SCALING_BUFFERS;
}

void enc_ControlPicture_common(vid_enc_PrivateType * priv, struct pipe_h264_enc_picture_desc *picture)
{
   struct pipe_h264_enc_rate_control *rate_ctrl = &picture->rate_ctrl;

   /* Get bitrate from port */
   switch (priv->bitrate.eControlRate) {
   case OMX_Video_ControlRateVariable:
      rate_ctrl->rate_ctrl_method = PIPE_H264_ENC_RATE_CONTROL_METHOD_VARIABLE;
      break;
   case OMX_Video_ControlRateConstant:
      rate_ctrl->rate_ctrl_method = PIPE_H264_ENC_RATE_CONTROL_METHOD_CONSTANT;
      break;
   case OMX_Video_ControlRateVariableSkipFrames:
      rate_ctrl->rate_ctrl_method = PIPE_H264_ENC_RATE_CONTROL_METHOD_VARIABLE_SKIP;
      break;
   case OMX_Video_ControlRateConstantSkipFrames:
      rate_ctrl->rate_ctrl_method = PIPE_H264_ENC_RATE_CONTROL_METHOD_CONSTANT_SKIP;
      break;
   default:
      rate_ctrl->rate_ctrl_method = PIPE_H264_ENC_RATE_CONTROL_METHOD_DISABLE;
      break;
   }

   rate_ctrl->frame_rate_den = OMX_VID_ENC_CONTROL_FRAME_RATE_DEN_DEFAULT;
   rate_ctrl->frame_rate_num = ((priv->frame_rate) >> 16) * rate_ctrl->frame_rate_den;

   if (rate_ctrl->rate_ctrl_method != PIPE_H264_ENC_RATE_CONTROL_METHOD_DISABLE) {
      if (priv->bitrate.nTargetBitrate < OMX_VID_ENC_BITRATE_MIN)
         rate_ctrl->target_bitrate = OMX_VID_ENC_BITRATE_MIN;
      else if (priv->bitrate.nTargetBitrate < OMX_VID_ENC_BITRATE_MAX)
         rate_ctrl->target_bitrate = priv->bitrate.nTargetBitrate;
      else
         rate_ctrl->target_bitrate = OMX_VID_ENC_BITRATE_MAX;
      rate_ctrl->peak_bitrate = rate_ctrl->target_bitrate;
      if (rate_ctrl->target_bitrate < OMX_VID_ENC_BITRATE_MEDIAN)
         rate_ctrl->vbv_buffer_size = MIN2((rate_ctrl->target_bitrate * 2.75), OMX_VID_ENC_BITRATE_MEDIAN);
      else
         rate_ctrl->vbv_buffer_size = rate_ctrl->target_bitrate;

      if (rate_ctrl->frame_rate_num) {
         unsigned long long t = rate_ctrl->target_bitrate;
         t *= rate_ctrl->frame_rate_den;
         rate_ctrl->target_bits_picture = t / rate_ctrl->frame_rate_num;
      } else {
         rate_ctrl->target_bits_picture = rate_ctrl->target_bitrate;
      }
      rate_ctrl->peak_bits_picture_integer = rate_ctrl->target_bits_picture;
      rate_ctrl->peak_bits_picture_fraction = 0;
   }

   picture->quant_i_frames = priv->quant.nQpI;
   picture->quant_p_frames = priv->quant.nQpP;
   picture->quant_b_frames = priv->quant.nQpB;

   picture->frame_num = priv->frame_num;
   picture->ref_idx_l0 = priv->ref_idx_l0;
   picture->ref_idx_l1 = priv->ref_idx_l1;
   picture->enable_vui = (picture->rate_ctrl.frame_rate_num != 0);
   enc_GetPictureParamPreset(picture);
}

OMX_ERRORTYPE enc_LoadImage_common(vid_enc_PrivateType * priv, OMX_VIDEO_PORTDEFINITIONTYPE *def,
                                   OMX_BUFFERHEADERTYPE *buf,
                                   struct pipe_video_buffer *vbuf)
{
   struct pipe_box box = {};
   struct input_buf_private *inp = buf->pInputPortPrivate;

   if (!inp->resource) {
      struct pipe_sampler_view **views;
      void *ptr;

      views = vbuf->get_sampler_view_planes(vbuf);
      if (!views)
         return OMX_ErrorInsufficientResources;

      ptr = buf->pBuffer;
      box.width = def->nFrameWidth;
      box.height = def->nFrameHeight;
      box.depth = 1;
      priv->s_pipe->texture_subdata(priv->s_pipe, views[0]->texture, 0,
                                    PIPE_TRANSFER_WRITE, &box,
                                    ptr, def->nStride, 0);
      ptr = ((uint8_t*)buf->pBuffer) + (def->nStride * box.height);
      box.width = def->nFrameWidth / 2;
      box.height = def->nFrameHeight / 2;
      box.depth = 1;
      priv->s_pipe->texture_subdata(priv->s_pipe, views[1]->texture, 0,
                                    PIPE_TRANSFER_WRITE, &box,
                                    ptr, def->nStride, 0);
   } else {
      struct pipe_blit_info blit;
      struct vl_video_buffer *dst_buf = (struct vl_video_buffer *)vbuf;

      pipe_transfer_unmap(priv->s_pipe, inp->transfer);

      box.width = def->nFrameWidth;
      box.height = def->nFrameHeight;
      box.depth = 1;

      priv->s_pipe->resource_copy_region(priv->s_pipe,
                                         dst_buf->resources[0],
                                         0, 0, 0, 0, inp->resource, 0, &box);

      memset(&blit, 0, sizeof(blit));
      blit.src.resource = inp->resource;
      blit.src.format = inp->resource->format;

      blit.src.box.x = -1;
      blit.src.box.y = def->nFrameHeight;
      blit.src.box.width = def->nFrameWidth;
      blit.src.box.height = def->nFrameHeight / 2 ;
      blit.src.box.depth = 1;

      blit.dst.resource = dst_buf->resources[1];
      blit.dst.format = blit.dst.resource->format;

      blit.dst.box.width = def->nFrameWidth / 2;
      blit.dst.box.height = def->nFrameHeight / 2;
      blit.dst.box.depth = 1;
      blit.filter = PIPE_TEX_FILTER_NEAREST;

      blit.mask = PIPE_MASK_R;
      priv->s_pipe->blit(priv->s_pipe, &blit);

      blit.src.box.x = 0;
      blit.mask = PIPE_MASK_G;
      priv->s_pipe->blit(priv->s_pipe, &blit);
      priv->s_pipe->flush(priv->s_pipe, NULL, 0);

      box.width = inp->resource->width0;
      box.height = inp->resource->height0;
      box.depth = inp->resource->depth0;
      buf->pBuffer = priv->s_pipe->transfer_map(priv->s_pipe, inp->resource, 0,
                                                PIPE_TRANSFER_WRITE, &box,
                                                &inp->transfer);
   }

   return OMX_ErrorNone;
}
