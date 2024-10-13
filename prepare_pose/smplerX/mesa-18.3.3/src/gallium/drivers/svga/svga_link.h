
#ifndef SVGA_LINK_H
#define SVGA_LINK_H

#include "pipe/p_defines.h"

struct svga_context;

struct shader_linkage
{
   unsigned num_inputs;
   unsigned input_map_max;  /* highest index of mapped inputs */
   ubyte input_map[PIPE_MAX_SHADER_INPUTS];
};

void
svga_link_shaders(const struct tgsi_shader_info *outshader_info,
                  const struct tgsi_shader_info *inshader_info,
                  struct shader_linkage *linkage);

#endif /* SVGA_LINK_H */
