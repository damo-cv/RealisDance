/**************************************************************************
 *
 * Copyright 2014 Advanced Micro Devices, Inc.
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

#include "va_private.h"

void vlVaHandlePictureParameterBufferMPEG12(vlVaDriver *drv, vlVaContext *context, vlVaBuffer *buf)
{
   VAPictureParameterBufferMPEG2 *mpeg2 = buf->data;

   assert(buf->size >= sizeof(VAPictureParameterBufferMPEG2) && buf->num_elements == 1);
   context->desc.mpeg12.num_slices = 0;
   /*horizontal_size;*/
   /*vertical_size;*/
   vlVaGetReferenceFrame(drv, mpeg2->forward_reference_picture, &context->desc.mpeg12.ref[0]);
   vlVaGetReferenceFrame(drv, mpeg2->backward_reference_picture, &context->desc.mpeg12.ref[1]);
   context->desc.mpeg12.picture_coding_type = mpeg2->picture_coding_type;
   context->desc.mpeg12.f_code[0][0] = ((mpeg2->f_code >> 12) & 0xf) - 1;
   context->desc.mpeg12.f_code[0][1] = ((mpeg2->f_code >> 8) & 0xf) - 1;
   context->desc.mpeg12.f_code[1][0] = ((mpeg2->f_code >> 4) & 0xf) - 1;
   context->desc.mpeg12.f_code[1][1] = (mpeg2->f_code & 0xf) - 1;
   context->desc.mpeg12.intra_dc_precision =
      mpeg2->picture_coding_extension.bits.intra_dc_precision;
   context->desc.mpeg12.picture_structure =
      mpeg2->picture_coding_extension.bits.picture_structure;
   context->desc.mpeg12.top_field_first =
      mpeg2->picture_coding_extension.bits.top_field_first;
   context->desc.mpeg12.frame_pred_frame_dct =
      mpeg2->picture_coding_extension.bits.frame_pred_frame_dct;
   context->desc.mpeg12.concealment_motion_vectors =
      mpeg2->picture_coding_extension.bits.concealment_motion_vectors;
   context->desc.mpeg12.q_scale_type =
      mpeg2->picture_coding_extension.bits.q_scale_type;
   context->desc.mpeg12.intra_vlc_format =
      mpeg2->picture_coding_extension.bits.intra_vlc_format;
   context->desc.mpeg12.alternate_scan =
      mpeg2->picture_coding_extension.bits.alternate_scan;
   /*repeat_first_field*/
   /*progressive_frame*/
   /*is_first_field*/
}

void vlVaHandleIQMatrixBufferMPEG12(vlVaContext *context, vlVaBuffer *buf)
{
   VAIQMatrixBufferMPEG2 *mpeg2 = buf->data;

   assert(buf->size >= sizeof(VAIQMatrixBufferMPEG2) && buf->num_elements == 1);
   if (mpeg2->load_intra_quantiser_matrix)
      context->desc.mpeg12.intra_matrix = mpeg2->intra_quantiser_matrix;
   else
      context->desc.mpeg12.intra_matrix = NULL;

   if (mpeg2->load_non_intra_quantiser_matrix)
      context->desc.mpeg12.non_intra_matrix = mpeg2->non_intra_quantiser_matrix;
   else
      context->desc.mpeg12.non_intra_matrix = NULL;
}

void vlVaHandleSliceParameterBufferMPEG12(vlVaContext *context, vlVaBuffer *buf)
{
   assert(buf->size >= sizeof(VASliceParameterBufferMPEG2));
   context->desc.mpeg12.num_slices += buf->num_elements;
}
