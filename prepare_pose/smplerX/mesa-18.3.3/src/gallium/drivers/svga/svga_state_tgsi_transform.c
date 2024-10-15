/**********************************************************
 * Copyright 2014 VMware, Inc.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **********************************************************/

#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_bitmask.h"
#include "util/u_simple_shaders.h"
#include "tgsi/tgsi_ureg.h"
#include "tgsi/tgsi_point_sprite.h"
#include "tgsi/tgsi_dump.h"

#include "svga_context.h"
#include "svga_shader.h"
#include "svga_tgsi.h"


/**
 * Bind a new GS.  This updates the derived current gs state, not the
 * user-specified GS state.
 */
static void
bind_gs_state(struct svga_context *svga,
              struct svga_geometry_shader *gs)
{
   svga->curr.gs = gs;
   svga->dirty |= SVGA_NEW_GS;
}


/**
 * emulate_point_sprite searches the shader variants list to see it there is
 * a shader variant with a token string that matches the emulation
 * requirement. It there isn't, then it will use a tgsi utility
 * tgsi_add_point_sprite to transform the original token string to support
 * point sprite. A new geometry shader state will be created with the
 * transformed token string and added to the shader variants list of the
 * original geometry shader. The new geometry shader state will then be
 * bound as the current geometry shader.
 */
static struct svga_shader *
emulate_point_sprite(struct svga_context *svga,
                     struct svga_shader *shader,
                     const struct tgsi_token *tokens)
{
   struct svga_token_key key;
   struct tgsi_token *new_tokens;
   const struct tgsi_token *orig_tokens;
   struct svga_geometry_shader *orig_gs = (struct svga_geometry_shader *)shader;
   struct svga_geometry_shader *gs = NULL;
   struct pipe_shader_state templ;
   struct svga_stream_output *streamout = NULL;
   int pos_out_index = -1;
   int aa_point_coord_index = -1;

   assert(tokens != NULL);

   orig_tokens = tokens;

   /* Create a token key */
   memset(&key, 0, sizeof key);
   key.gs.writes_psize = 1;
   key.gs.sprite_coord_enable = svga->curr.rast->templ.sprite_coord_enable;

   key.gs.sprite_origin_upper_left =
      !(svga->curr.rast->templ.sprite_coord_mode == PIPE_SPRITE_COORD_LOWER_LEFT);

   key.gs.aa_point = svga->curr.rast->templ.point_smooth;

   if (orig_gs) {

      /* Check if the original geometry shader has stream output and
       * if position is one of the outputs.
       */
      streamout = orig_gs->base.stream_output;
      if (streamout) {
         pos_out_index = streamout->pos_out_index;
         key.gs.point_pos_stream_out = pos_out_index != -1;
      }

      /* Search the shader lists to see if there is a variant that matches
       * this token key.
       */
      gs = (struct svga_geometry_shader *)
              svga_search_shader_token_key(&orig_gs->base, &key);
   }

   /* If there isn't, then call the tgsi utility tgsi_add_point_sprite
    * to transform the original tokens to support point sprite.
    * Flip the sprite origin as SVGA3D device only supports an
    * upper-left origin.
    */
   if (!gs) {
      new_tokens = tgsi_add_point_sprite(orig_tokens,
                                         key.gs.sprite_coord_enable,
                                         key.gs.sprite_origin_upper_left,
                                         key.gs.point_pos_stream_out,
                                         key.gs.aa_point ?
                                            &aa_point_coord_index : NULL);

      if (!new_tokens) {
         /* if no new tokens are generated for whatever reason, just return */
         return NULL;
      }

      if (0) {
         debug_printf("Before tgsi_add_point_sprite ---------------\n");
         tgsi_dump(orig_tokens, 0);
         debug_printf("After tgsi_add_point_sprite --------------\n");
         tgsi_dump(new_tokens, 0);
      }

      templ.tokens = new_tokens;
      templ.stream_output.num_outputs = 0;

      if (streamout) {
         templ.stream_output = streamout->info;
         /* The tgsi_add_point_sprite utility adds an extra output
          * for the original point position for stream output purpose.
          * We need to replace the position output register index in the
          * stream output declaration with the new register index.
          */
         if (pos_out_index != -1) {
            assert(orig_gs != NULL);
            templ.stream_output.output[pos_out_index].register_index =
               orig_gs->base.info.num_outputs;
         }
      }

      /* Create a new geometry shader state with the new tokens */
      gs = svga->pipe.create_gs_state(&svga->pipe, &templ);

      /* Don't need the token string anymore. There is a local copy
       * in the shader state.
       */
      FREE(new_tokens);

      if (!gs) {
         return NULL;
      }

      gs->wide_point = TRUE;
      gs->aa_point_coord_index = aa_point_coord_index;
      gs->base.token_key = key;
      gs->base.parent = &orig_gs->base;
      gs->base.next = NULL;

      /* Add the new geometry shader to the head of the shader list
       * pointed to by the original geometry shader.
       */
      if (orig_gs) {
         gs->base.next = orig_gs->base.next;
         orig_gs->base.next = &gs->base;
      }
   }

   /* Bind the new geometry shader state */
   bind_gs_state(svga, gs);

   return &gs->base;
}

/**
 * Generate a geometry shader that emits a wide point by drawing a quad.
 * This function first creates a passthrough geometry shader and then
 * calls emulate_point_sprite() to transform the geometry shader to
 * support point sprite.
 */
static struct svga_shader *
add_point_sprite_shader(struct svga_context *svga)
{
   struct svga_vertex_shader *vs = svga->curr.vs;
   struct svga_geometry_shader *orig_gs = vs->gs;
   struct svga_geometry_shader *new_gs;
   const struct tgsi_token *tokens;

   if (orig_gs == NULL) {

      /* If this is the first time adding a geometry shader to this
       * vertex shader to support point sprite, then create
       * a passthrough geometry shader first.
       */
      orig_gs = (struct svga_geometry_shader *)
                   util_make_geometry_passthrough_shader(
                      &svga->pipe, vs->base.info.num_outputs,
                      vs->base.info.output_semantic_name,
                      vs->base.info.output_semantic_index);

      if (!orig_gs)
         return NULL;
   }
   else {
      if (orig_gs->base.parent)
         orig_gs = (struct svga_geometry_shader *)orig_gs->base.parent;
   }
   tokens = orig_gs->base.tokens;

   /* Call emulate_point_sprite to find or create a transformed
    * geometry shader for supporting point sprite.
    */
   new_gs = (struct svga_geometry_shader *)
               emulate_point_sprite(svga, &orig_gs->base, tokens);

   /* If this is the first time creating a geometry shader to
    * support vertex point size, then add the new geometry shader
    * to the vertex shader.
    */
   if (vs->gs == NULL) {
      vs->gs = new_gs;
   }

   return &new_gs->base;
}

/* update_tgsi_transform provides a hook to transform a shader if needed.
 */
static enum pipe_error
update_tgsi_transform(struct svga_context *svga, unsigned dirty)
{
   struct svga_geometry_shader *gs = svga->curr.user_gs;   /* current gs */
   struct svga_vertex_shader *vs = svga->curr.vs;     /* currently bound vs */
   struct svga_shader *orig_gs;                       /* original gs */
   struct svga_shader *new_gs;                        /* new gs */

   if (!svga_have_vgpu10(svga))
      return PIPE_OK;

   if (svga->curr.reduced_prim == PIPE_PRIM_POINTS) {
      /* If the current prim type is POINTS and the current geometry shader
       * emits wide points, transform the shader to emulate wide points using
       * quads. NOTE: we don't do emulation of wide points in GS when
       * transform feedback is enabled.
       */
      if (gs != NULL && !gs->base.stream_output &&
          (gs->base.info.writes_psize || gs->wide_point)) {
         orig_gs = gs->base.parent ? gs->base.parent : &gs->base;
         new_gs = emulate_point_sprite(svga, orig_gs, orig_gs->tokens);
      }

      /* If there is not an active geometry shader and the current vertex
       * shader emits wide point then create a new geometry shader to emulate
       * wide point.
       */
      else if (gs == NULL && !vs->base.stream_output &&
               (svga->curr.rast->pointsize > 1.0 ||
                vs->base.info.writes_psize)) {
         new_gs = add_point_sprite_shader(svga);
      }
      else {
         /* use the user's GS */
         bind_gs_state(svga, svga->curr.user_gs);
      }
   }
   else if (svga->curr.gs != svga->curr.user_gs) {
      /* If current primitive type is not POINTS, then make sure
       * we don't bind to any of the generated geometry shader
       */
      bind_gs_state(svga, svga->curr.user_gs);
   }
   (void) new_gs;    /* silence the unused var warning */

   return PIPE_OK;
}

struct svga_tracked_state svga_need_tgsi_transform =
{
   "transform shader for optimization",
   (SVGA_NEW_VS |
    SVGA_NEW_FS |
    SVGA_NEW_GS |
    SVGA_NEW_REDUCED_PRIMITIVE |
    SVGA_NEW_RAST),
   update_tgsi_transform
};
