
#ifndef __NVC0_PROGRAM_H__
#define __NVC0_PROGRAM_H__

#include "pipe/p_state.h"

#define NVC0_CAP_MAX_PROGRAM_TEMPS 128


struct nvc0_transform_feedback_state {
   uint32_t stride[4];
   uint8_t stream[4];
   uint8_t varying_count[4];
   uint8_t varying_index[4][128];
};


#define NVC0_SHADER_HEADER_SIZE (20 * 4)

struct nvc0_program {
   struct pipe_shader_state pipe;

   ubyte type;
   bool translated;
   bool need_tls;
   uint8_t num_gprs;

   uint32_t *code;
   unsigned code_base;
   unsigned code_size;
   unsigned parm_size; /* size of non-bindable uniforms (c0[]) */

   uint32_t hdr[20];
   uint32_t flags[2];

   struct {
      uint32_t clip_mode; /* clip/cull selection */
      uint8_t clip_enable; /* mask of defined clip planes */
      uint8_t cull_enable; /* mask of defined cull distances */
      uint8_t num_ucps; /* also set to max if ClipDistance is used */
      uint8_t edgeflag; /* attribute index of edgeflag input */
      bool need_vertex_id;
      bool need_draw_parameters;
   } vp;
   struct {
      uint8_t early_z;
      uint8_t colors;
      uint8_t color_interp[2];
      bool sample_mask_in;
      bool force_persample_interp;
      bool flatshade;
      bool reads_framebuffer;
      bool post_depth_coverage;
   } fp;
   struct {
      uint32_t tess_mode; /* ~0 if defined by the other stage */
      uint32_t input_patch_size;
   } tp;
   struct {
      uint32_t lmem_size; /* local memory (TGSI PRIVATE resource) size */
      uint32_t smem_size; /* shared memory (TGSI LOCAL resource) size */
      void *syms;
      unsigned num_syms;
   } cp;
   uint8_t num_barriers;

   void *relocs;
   void *fixups;

   struct nvc0_transform_feedback_state *tfb;

   struct nouveau_heap *mem;
};

#endif
