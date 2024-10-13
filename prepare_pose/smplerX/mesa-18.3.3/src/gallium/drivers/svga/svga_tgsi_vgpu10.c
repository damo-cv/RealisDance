/**********************************************************
 * Copyright 1998-2013 VMware, Inc.  All rights reserved.
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

/**
 * @file svga_tgsi_vgpu10.c
 *
 * TGSI -> VGPU10 shader translation.
 *
 * \author Mingcheng Chen
 * \author Brian Paul
 */

#include "pipe/p_compiler.h"
#include "pipe/p_shader_tokens.h"
#include "pipe/p_defines.h"
#include "tgsi/tgsi_build.h"
#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_info.h"
#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_scan.h"
#include "tgsi/tgsi_two_side.h"
#include "tgsi/tgsi_aa_point.h"
#include "tgsi/tgsi_util.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_bitmask.h"
#include "util/u_debug.h"
#include "util/u_pstipple.h"

#include "svga_context.h"
#include "svga_debug.h"
#include "svga_link.h"
#include "svga_shader.h"
#include "svga_tgsi.h"

#include "VGPU10ShaderTokens.h"


#define INVALID_INDEX 99999
#define MAX_INTERNAL_TEMPS 3
#define MAX_SYSTEM_VALUES 4
#define MAX_IMMEDIATE_COUNT \
        (VGPU10_MAX_IMMEDIATE_CONSTANT_BUFFER_ELEMENT_COUNT/4)
#define MAX_TEMP_ARRAYS 64  /* Enough? */


/**
 * Clipping is complicated.  There's four different cases which we
 * handle during VS/GS shader translation:
 */
enum clipping_mode
{
   CLIP_NONE,     /**< No clipping enabled */
   CLIP_LEGACY,   /**< The shader has no clipping declarations or code but
                   * one or more user-defined clip planes are enabled.  We
                   * generate extra code to emit clip distances.
                   */
   CLIP_DISTANCE, /**< The shader already declares clip distance output
                   * registers and has code to write to them.
                   */
   CLIP_VERTEX    /**< The shader declares a clip vertex output register and
                  * has code that writes to the register.  We convert the
                  * clipvertex position into one or more clip distances.
                  */
};


struct svga_shader_emitter_v10
{
   /* The token output buffer */
   unsigned size;
   char *buf;
   char *ptr;

   /* Information about the shader and state (does not change) */
   struct svga_compile_key key;
   struct tgsi_shader_info info;
   unsigned unit;
   unsigned version; /**< Either 40 or 41 at this time */

   unsigned inst_start_token;
   boolean discard_instruction; /**< throw away current instruction? */

   union tgsi_immediate_data immediates[MAX_IMMEDIATE_COUNT][4];
   unsigned num_immediates;      /**< Number of immediates emitted */
   unsigned common_immediate_pos[8];  /**< literals for common immediates */
   unsigned num_common_immediates;
   boolean immediates_emitted;

   unsigned num_outputs;      /**< include any extra outputs */
                              /**  The first extra output is reserved for
                               *   non-adjusted vertex position for
                               *   stream output purpose
                               */

   /* Temporary Registers */
   unsigned num_shader_temps; /**< num of temps used by original shader */
   unsigned internal_temp_count;  /**< currently allocated internal temps */
   struct {
      unsigned start, size;
   } temp_arrays[MAX_TEMP_ARRAYS];
   unsigned num_temp_arrays;

   /** Map TGSI temp registers to VGPU10 temp array IDs and indexes */
   struct {
      unsigned arrayId, index;
   } temp_map[VGPU10_MAX_TEMPS]; /**< arrayId, element */

   /** Number of constants used by original shader for each constant buffer.
    * The size should probably always match with that of svga_state.constbufs.
    */
   unsigned num_shader_consts[SVGA_MAX_CONST_BUFS];

   /* Samplers */
   unsigned num_samplers;
   boolean sampler_view[PIPE_MAX_SAMPLERS];  /**< True if sampler view exists*/
   ubyte sampler_target[PIPE_MAX_SAMPLERS];  /**< TGSI_TEXTURE_x */
   ubyte sampler_return_type[PIPE_MAX_SAMPLERS];  /**< TGSI_RETURN_TYPE_x */

   /* Address regs (really implemented with temps) */
   unsigned num_address_regs;
   unsigned address_reg_index[MAX_VGPU10_ADDR_REGS];

   /* Output register usage masks */
   ubyte output_usage_mask[PIPE_MAX_SHADER_OUTPUTS];

   /* To map TGSI system value index to VGPU shader input indexes */
   ubyte system_value_indexes[MAX_SYSTEM_VALUES];

   struct {
      /* vertex position scale/translation */
      unsigned out_index;  /**< the real position output reg */
      unsigned tmp_index;  /**< the fake/temp position output reg */
      unsigned so_index;   /**< the non-adjusted position output reg */
      unsigned prescale_scale_index, prescale_trans_index;
      boolean  need_prescale;
   } vposition;

   /* For vertex shaders only */
   struct {
      /* viewport constant */
      unsigned viewport_index;

      /* temp index of adjusted vertex attributes */
      unsigned adjusted_input[PIPE_MAX_SHADER_INPUTS];
   } vs;

   /* For fragment shaders only */
   struct {
      unsigned color_out_index[PIPE_MAX_COLOR_BUFS];  /**< the real color output regs */
      unsigned num_color_outputs;
      unsigned color_tmp_index;  /**< fake/temp color output reg */
      unsigned alpha_ref_index;  /**< immediate constant for alpha ref */

      /* front-face */
      unsigned face_input_index; /**< real fragment shader face reg (bool) */
      unsigned face_tmp_index;   /**< temp face reg converted to -1 / +1 */

      unsigned pstipple_sampler_unit;

      unsigned fragcoord_input_index;  /**< real fragment position input reg */
      unsigned fragcoord_tmp_index;    /**< 1/w modified position temp reg */

      /** Which texture units are doing shadow comparison in the FS code */
      unsigned shadow_compare_units;

      unsigned sample_id_sys_index;  /**< TGSI index of sample id sys value */

      unsigned sample_pos_sys_index; /**< TGSI index of sample pos sys value */
      unsigned sample_pos_tmp_index; /**< which temp reg has the sample pos */
   } fs;

   /* For geometry shaders only */
   struct {
      VGPU10_PRIMITIVE prim_type;/**< VGPU10 primitive type */
      VGPU10_PRIMITIVE_TOPOLOGY prim_topology; /**< VGPU10 primitive topology */
      unsigned input_size;       /**< size of input arrays */
      unsigned prim_id_index;    /**< primitive id register index */
      unsigned max_out_vertices; /**< maximum number of output vertices */
   } gs;

   /* For vertex or geometry shaders */
   enum clipping_mode clip_mode;
   unsigned clip_dist_out_index; /**< clip distance output register index */
   unsigned clip_dist_tmp_index; /**< clip distance temporary register */
   unsigned clip_dist_so_index;  /**< clip distance shadow copy */

   /** Index of temporary holding the clipvertex coordinate */
   unsigned clip_vertex_out_index; /**< clip vertex output register index */
   unsigned clip_vertex_tmp_index; /**< clip vertex temporary index */

   /* user clip plane constant slot indexes */
   unsigned clip_plane_const[PIPE_MAX_CLIP_PLANES];

   unsigned num_output_writes;
   boolean constant_color_output;

   boolean uses_flat_interp;

   /* For all shaders: const reg index for RECT coord scaling */
   unsigned texcoord_scale_index[PIPE_MAX_SAMPLERS];

   /* For all shaders: const reg index for texture buffer size */
   unsigned texture_buffer_size_index[PIPE_MAX_SAMPLERS];

   /* VS/GS/FS Linkage info */
   struct shader_linkage linkage;

   bool register_overflow;  /**< Set if we exceed a VGPU10 register limit */
};


static boolean
emit_post_helpers(struct svga_shader_emitter_v10 *emit);

static boolean
emit_vertex(struct svga_shader_emitter_v10 *emit,
            const struct tgsi_full_instruction *inst);

static char err_buf[128];

static boolean
expand(struct svga_shader_emitter_v10 *emit)
{
   char *new_buf;
   unsigned newsize = emit->size * 2;

   if (emit->buf != err_buf)
      new_buf = REALLOC(emit->buf, emit->size, newsize);
   else
      new_buf = NULL;

   if (!new_buf) {
      emit->ptr = err_buf;
      emit->buf = err_buf;
      emit->size = sizeof(err_buf);
      return FALSE;
   }

   emit->size = newsize;
   emit->ptr = new_buf + (emit->ptr - emit->buf);
   emit->buf = new_buf;
   return TRUE;
}

/**
 * Create and initialize a new svga_shader_emitter_v10 object.
 */
static struct svga_shader_emitter_v10 *
alloc_emitter(void)
{
   struct svga_shader_emitter_v10 *emit = CALLOC(1, sizeof(*emit));

   if (!emit)
      return NULL;

   /* to initialize the output buffer */
   emit->size = 512;
   if (!expand(emit)) {
      FREE(emit);
      return NULL;
   }
   return emit;
}

/**
 * Free an svga_shader_emitter_v10 object.
 */
static void
free_emitter(struct svga_shader_emitter_v10 *emit)
{
   assert(emit);
   FREE(emit->buf);    /* will be NULL if translation succeeded */
   FREE(emit);
}

static inline boolean
reserve(struct svga_shader_emitter_v10 *emit,
        unsigned nr_dwords)
{
   while (emit->ptr - emit->buf + nr_dwords * sizeof(uint32) >= emit->size) {
      if (!expand(emit))
         return FALSE;
   }

   return TRUE;
}

static boolean
emit_dword(struct svga_shader_emitter_v10 *emit, uint32 dword)
{
   if (!reserve(emit, 1))
      return FALSE;

   *(uint32 *)emit->ptr = dword;
   emit->ptr += sizeof dword;
   return TRUE;
}

static boolean
emit_dwords(struct svga_shader_emitter_v10 *emit,
            const uint32 *dwords,
            unsigned nr)
{
   if (!reserve(emit, nr))
      return FALSE;

   memcpy(emit->ptr, dwords, nr * sizeof *dwords);
   emit->ptr += nr * sizeof *dwords;
   return TRUE;
}

/** Return the number of tokens in the emitter's buffer */
static unsigned
emit_get_num_tokens(const struct svga_shader_emitter_v10 *emit)
{
   return (emit->ptr - emit->buf) / sizeof(unsigned);
}


/**
 * Check for register overflow.  If we overflow we'll set an
 * error flag.  This function can be called for register declarations
 * or use as src/dst instruction operands.
 * \param type  register type.  One of VGPU10_OPERAND_TYPE_x
                or VGPU10_OPCODE_DCL_x
 * \param index  the register index
 */
static void
check_register_index(struct svga_shader_emitter_v10 *emit,
                     unsigned operandType, unsigned index)
{
   bool overflow_before = emit->register_overflow;

   switch (operandType) {
   case VGPU10_OPERAND_TYPE_TEMP:
   case VGPU10_OPERAND_TYPE_INDEXABLE_TEMP:
   case VGPU10_OPCODE_DCL_TEMPS:
      if (index >= VGPU10_MAX_TEMPS) {
         emit->register_overflow = TRUE;
      }
      break;
   case VGPU10_OPERAND_TYPE_CONSTANT_BUFFER:
   case VGPU10_OPCODE_DCL_CONSTANT_BUFFER:
      if (index >= VGPU10_MAX_CONSTANT_BUFFER_ELEMENT_COUNT) {
         emit->register_overflow = TRUE;
      }
      break;
   case VGPU10_OPERAND_TYPE_INPUT:
   case VGPU10_OPERAND_TYPE_INPUT_PRIMITIVEID:
   case VGPU10_OPCODE_DCL_INPUT:
   case VGPU10_OPCODE_DCL_INPUT_SGV:
   case VGPU10_OPCODE_DCL_INPUT_SIV:
   case VGPU10_OPCODE_DCL_INPUT_PS:
   case VGPU10_OPCODE_DCL_INPUT_PS_SGV:
   case VGPU10_OPCODE_DCL_INPUT_PS_SIV:
      if ((emit->unit == PIPE_SHADER_VERTEX &&
           index >= VGPU10_MAX_VS_INPUTS) ||
          (emit->unit == PIPE_SHADER_GEOMETRY &&
           index >= VGPU10_MAX_GS_INPUTS) ||
          (emit->unit == PIPE_SHADER_FRAGMENT &&
           index >= VGPU10_MAX_FS_INPUTS)) {
         emit->register_overflow = TRUE;
      }
      break;
   case VGPU10_OPERAND_TYPE_OUTPUT:
   case VGPU10_OPCODE_DCL_OUTPUT:
   case VGPU10_OPCODE_DCL_OUTPUT_SGV:
   case VGPU10_OPCODE_DCL_OUTPUT_SIV:
      if ((emit->unit == PIPE_SHADER_VERTEX &&
           index >= VGPU10_MAX_VS_OUTPUTS) ||
          (emit->unit == PIPE_SHADER_GEOMETRY &&
           index >= VGPU10_MAX_GS_OUTPUTS) ||
          (emit->unit == PIPE_SHADER_FRAGMENT &&
           index >= VGPU10_MAX_FS_OUTPUTS)) {
         emit->register_overflow = TRUE;
      }
      break;
   case VGPU10_OPERAND_TYPE_SAMPLER:
   case VGPU10_OPCODE_DCL_SAMPLER:
      if (index >= VGPU10_MAX_SAMPLERS) {
         emit->register_overflow = TRUE;
      }
      break;
   case VGPU10_OPERAND_TYPE_RESOURCE:
   case VGPU10_OPCODE_DCL_RESOURCE:
      if (index >= VGPU10_MAX_RESOURCES) {
         emit->register_overflow = TRUE;
      }
      break;
   case VGPU10_OPERAND_TYPE_IMMEDIATE_CONSTANT_BUFFER:
      if (index >= MAX_IMMEDIATE_COUNT) {
         emit->register_overflow = TRUE;
      }
      break;
   case VGPU10_OPERAND_TYPE_OUTPUT_COVERAGE_MASK:
      /* nothing */
      break;
   default:
      assert(0);
      ; /* nothing */
   }

   if (emit->register_overflow && !overflow_before) {
      debug_printf("svga: vgpu10 register overflow (reg %u, index %u)\n",
                   operandType, index);
   }
}


/**
 * Examine misc state to determine the clipping mode.
 */
static void
determine_clipping_mode(struct svga_shader_emitter_v10 *emit)
{
   if (emit->info.num_written_clipdistance > 0) {
      emit->clip_mode = CLIP_DISTANCE;
   }
   else if (emit->info.writes_clipvertex) {
      emit->clip_mode = CLIP_VERTEX;
   }
   else if (emit->key.clip_plane_enable) {
      emit->clip_mode = CLIP_LEGACY;
   }
   else {
      emit->clip_mode = CLIP_NONE;
   }
}


/**
 * For clip distance register declarations and clip distance register
 * writes we need to mask the declaration usage or instruction writemask
 * (respectively) against the set of the really-enabled clipping planes.
 *
 * The piglit test spec/glsl-1.30/execution/clipping/vs-clip-distance-enables
 * has a VS that writes to all 8 clip distance registers, but the plane enable
 * flags are a subset of that.
 *
 * This function is used to apply the plane enable flags to the register
 * declaration or instruction writemask.
 *
 * \param writemask  the declaration usage mask or instruction writemask
 * \param clip_reg_index  which clip plane register is being declared/written.
 *                        The legal values are 0 and 1 (two clip planes per
 *                        register, for a total of 8 clip planes)
 */
static unsigned
apply_clip_plane_mask(struct svga_shader_emitter_v10 *emit,
                      unsigned writemask, unsigned clip_reg_index)
{
   unsigned shift;

   assert(clip_reg_index < 2);

   /* four clip planes per clip register: */
   shift = clip_reg_index * 4;
   writemask &= ((emit->key.clip_plane_enable >> shift) & 0xf);

   return writemask;
}


/**
 * Translate gallium shader type into VGPU10 type.
 */
static VGPU10_PROGRAM_TYPE
translate_shader_type(unsigned type)
{
   switch (type) {
   case PIPE_SHADER_VERTEX:
      return VGPU10_VERTEX_SHADER;
   case PIPE_SHADER_GEOMETRY:
      return VGPU10_GEOMETRY_SHADER;
   case PIPE_SHADER_FRAGMENT:
      return VGPU10_PIXEL_SHADER;
   default:
      assert(!"Unexpected shader type");
      return VGPU10_VERTEX_SHADER;
   }
}


/**
 * Translate a TGSI_OPCODE_x into a VGPU10_OPCODE_x
 * Note: we only need to translate the opcodes for "simple" instructions,
 * as seen below.  All other opcodes are handled/translated specially.
 */
static VGPU10_OPCODE_TYPE
translate_opcode(enum tgsi_opcode opcode)
{
   switch (opcode) {
   case TGSI_OPCODE_MOV:
      return VGPU10_OPCODE_MOV;
   case TGSI_OPCODE_MUL:
      return VGPU10_OPCODE_MUL;
   case TGSI_OPCODE_ADD:
      return VGPU10_OPCODE_ADD;
   case TGSI_OPCODE_DP3:
      return VGPU10_OPCODE_DP3;
   case TGSI_OPCODE_DP4:
      return VGPU10_OPCODE_DP4;
   case TGSI_OPCODE_MIN:
      return VGPU10_OPCODE_MIN;
   case TGSI_OPCODE_MAX:
      return VGPU10_OPCODE_MAX;
   case TGSI_OPCODE_MAD:
      return VGPU10_OPCODE_MAD;
   case TGSI_OPCODE_SQRT:
      return VGPU10_OPCODE_SQRT;
   case TGSI_OPCODE_FRC:
      return VGPU10_OPCODE_FRC;
   case TGSI_OPCODE_FLR:
      return VGPU10_OPCODE_ROUND_NI;
   case TGSI_OPCODE_FSEQ:
      return VGPU10_OPCODE_EQ;
   case TGSI_OPCODE_FSGE:
      return VGPU10_OPCODE_GE;
   case TGSI_OPCODE_FSNE:
      return VGPU10_OPCODE_NE;
   case TGSI_OPCODE_DDX:
      return VGPU10_OPCODE_DERIV_RTX;
   case TGSI_OPCODE_DDY:
      return VGPU10_OPCODE_DERIV_RTY;
   case TGSI_OPCODE_RET:
      return VGPU10_OPCODE_RET;
   case TGSI_OPCODE_DIV:
      return VGPU10_OPCODE_DIV;
   case TGSI_OPCODE_IDIV:
      return VGPU10_OPCODE_IDIV;
   case TGSI_OPCODE_DP2:
      return VGPU10_OPCODE_DP2;
   case TGSI_OPCODE_BRK:
      return VGPU10_OPCODE_BREAK;
   case TGSI_OPCODE_IF:
      return VGPU10_OPCODE_IF;
   case TGSI_OPCODE_ELSE:
      return VGPU10_OPCODE_ELSE;
   case TGSI_OPCODE_ENDIF:
      return VGPU10_OPCODE_ENDIF;
   case TGSI_OPCODE_CEIL:
      return VGPU10_OPCODE_ROUND_PI;
   case TGSI_OPCODE_I2F:
      return VGPU10_OPCODE_ITOF;
   case TGSI_OPCODE_NOT:
      return VGPU10_OPCODE_NOT;
   case TGSI_OPCODE_TRUNC:
      return VGPU10_OPCODE_ROUND_Z;
   case TGSI_OPCODE_SHL:
      return VGPU10_OPCODE_ISHL;
   case TGSI_OPCODE_AND:
      return VGPU10_OPCODE_AND;
   case TGSI_OPCODE_OR:
      return VGPU10_OPCODE_OR;
   case TGSI_OPCODE_XOR:
      return VGPU10_OPCODE_XOR;
   case TGSI_OPCODE_CONT:
      return VGPU10_OPCODE_CONTINUE;
   case TGSI_OPCODE_EMIT:
      return VGPU10_OPCODE_EMIT;
   case TGSI_OPCODE_ENDPRIM:
      return VGPU10_OPCODE_CUT;
   case TGSI_OPCODE_BGNLOOP:
      return VGPU10_OPCODE_LOOP;
   case TGSI_OPCODE_ENDLOOP:
      return VGPU10_OPCODE_ENDLOOP;
   case TGSI_OPCODE_ENDSUB:
      return VGPU10_OPCODE_RET;
   case TGSI_OPCODE_NOP:
      return VGPU10_OPCODE_NOP;
   case TGSI_OPCODE_END:
      return VGPU10_OPCODE_RET;
   case TGSI_OPCODE_F2I:
      return VGPU10_OPCODE_FTOI;
   case TGSI_OPCODE_IMAX:
      return VGPU10_OPCODE_IMAX;
   case TGSI_OPCODE_IMIN:
      return VGPU10_OPCODE_IMIN;
   case TGSI_OPCODE_UDIV:
   case TGSI_OPCODE_UMOD:
   case TGSI_OPCODE_MOD:
      return VGPU10_OPCODE_UDIV;
   case TGSI_OPCODE_IMUL_HI:
      return VGPU10_OPCODE_IMUL;
   case TGSI_OPCODE_INEG:
      return VGPU10_OPCODE_INEG;
   case TGSI_OPCODE_ISHR:
      return VGPU10_OPCODE_ISHR;
   case TGSI_OPCODE_ISGE:
      return VGPU10_OPCODE_IGE;
   case TGSI_OPCODE_ISLT:
      return VGPU10_OPCODE_ILT;
   case TGSI_OPCODE_F2U:
      return VGPU10_OPCODE_FTOU;
   case TGSI_OPCODE_UADD:
      return VGPU10_OPCODE_IADD;
   case TGSI_OPCODE_U2F:
      return VGPU10_OPCODE_UTOF;
   case TGSI_OPCODE_UCMP:
      return VGPU10_OPCODE_MOVC;
   case TGSI_OPCODE_UMAD:
      return VGPU10_OPCODE_UMAD;
   case TGSI_OPCODE_UMAX:
      return VGPU10_OPCODE_UMAX;
   case TGSI_OPCODE_UMIN:
      return VGPU10_OPCODE_UMIN;
   case TGSI_OPCODE_UMUL:
   case TGSI_OPCODE_UMUL_HI:
      return VGPU10_OPCODE_UMUL;
   case TGSI_OPCODE_USEQ:
      return VGPU10_OPCODE_IEQ;
   case TGSI_OPCODE_USGE:
      return VGPU10_OPCODE_UGE;
   case TGSI_OPCODE_USHR:
      return VGPU10_OPCODE_USHR;
   case TGSI_OPCODE_USLT:
      return VGPU10_OPCODE_ULT;
   case TGSI_OPCODE_USNE:
      return VGPU10_OPCODE_INE;
   case TGSI_OPCODE_SWITCH:
      return VGPU10_OPCODE_SWITCH;
   case TGSI_OPCODE_CASE:
      return VGPU10_OPCODE_CASE;
   case TGSI_OPCODE_DEFAULT:
      return VGPU10_OPCODE_DEFAULT;
   case TGSI_OPCODE_ENDSWITCH:
      return VGPU10_OPCODE_ENDSWITCH;
   case TGSI_OPCODE_FSLT:
      return VGPU10_OPCODE_LT;
   case TGSI_OPCODE_ROUND:
      return VGPU10_OPCODE_ROUND_NE;
   case TGSI_OPCODE_SAMPLE_POS:
      /* Note: we never actually get this opcode because there's no GLSL
       * function to query multisample resource sample positions.  There's
       * only the TGSI_SEMANTIC_SAMPLEPOS system value which contains the
       * position of the current sample in the render target.
       */
      /* FALL-THROUGH */
   case TGSI_OPCODE_SAMPLE_INFO:
      /* NOTE: we never actually get this opcode because the GLSL compiler
       * implements the gl_NumSamples variable with a simple constant in the
       * constant buffer.
       */
      /* FALL-THROUGH */
   default:
      assert(!"Unexpected TGSI opcode in translate_opcode()");
      return VGPU10_OPCODE_NOP;
   }
}


/**
 * Translate a TGSI register file type into a VGPU10 operand type.
 * \param array  is the TGSI_FILE_TEMPORARY register an array?
 */
static VGPU10_OPERAND_TYPE
translate_register_file(enum tgsi_file_type file, boolean array)
{
   switch (file) {
   case TGSI_FILE_CONSTANT:
      return VGPU10_OPERAND_TYPE_CONSTANT_BUFFER;
   case TGSI_FILE_INPUT:
      return VGPU10_OPERAND_TYPE_INPUT;
   case TGSI_FILE_OUTPUT:
      return VGPU10_OPERAND_TYPE_OUTPUT;
   case TGSI_FILE_TEMPORARY:
      return array ? VGPU10_OPERAND_TYPE_INDEXABLE_TEMP
                   : VGPU10_OPERAND_TYPE_TEMP;
   case TGSI_FILE_IMMEDIATE:
      /* all immediates are 32-bit values at this time so
       * VGPU10_OPERAND_TYPE_IMMEDIATE64 is not possible at this time.
       */
      return VGPU10_OPERAND_TYPE_IMMEDIATE_CONSTANT_BUFFER;
   case TGSI_FILE_SAMPLER:
      return VGPU10_OPERAND_TYPE_SAMPLER;
   case TGSI_FILE_SYSTEM_VALUE:
      return VGPU10_OPERAND_TYPE_INPUT;

   /* XXX TODO more cases to finish */

   default:
      assert(!"Bad tgsi register file!");
      return VGPU10_OPERAND_TYPE_NULL;
   }
}


/**
 * Emit a null dst register
 */
static void
emit_null_dst_register(struct svga_shader_emitter_v10 *emit)
{
   VGPU10OperandToken0 operand;

   operand.value = 0;
   operand.operandType = VGPU10_OPERAND_TYPE_NULL;
   operand.numComponents = VGPU10_OPERAND_0_COMPONENT;

   emit_dword(emit, operand.value);
}


/**
 * If the given register is a temporary, return the array ID.
 * Else return zero.
 */
static unsigned
get_temp_array_id(const struct svga_shader_emitter_v10 *emit,
                  enum tgsi_file_type file, unsigned index)
{
   if (file == TGSI_FILE_TEMPORARY) {
      return emit->temp_map[index].arrayId;
   }
   else {
      return 0;
   }
}


/**
 * If the given register is a temporary, convert the index from a TGSI
 * TEMPORARY index to a VGPU10 temp index.
 */
static unsigned
remap_temp_index(const struct svga_shader_emitter_v10 *emit,
                 enum tgsi_file_type file, unsigned index)
{
   if (file == TGSI_FILE_TEMPORARY) {
      return emit->temp_map[index].index;
   }
   else {
      return index;
   }
}


/**
 * Setup the operand0 fields related to indexing (1D, 2D, relative, etc).
 * Note: the operandType field must already be initialized.
 */
static VGPU10OperandToken0
setup_operand0_indexing(struct svga_shader_emitter_v10 *emit,
                        VGPU10OperandToken0 operand0,
                        enum tgsi_file_type file,
                        boolean indirect, boolean index2D,
                        unsigned tempArrayID)
{
   unsigned indexDim, index0Rep, index1Rep = VGPU10_OPERAND_INDEX_IMMEDIATE32;

   /*
    * Compute index dimensions
    */
   if (operand0.operandType == VGPU10_OPERAND_TYPE_IMMEDIATE32 ||
       operand0.operandType == VGPU10_OPERAND_TYPE_INPUT_PRIMITIVEID) {
      /* there's no swizzle for in-line immediates */
      indexDim = VGPU10_OPERAND_INDEX_0D;
      assert(operand0.selectionMode == 0);
   }
   else {
      if (index2D ||
          tempArrayID > 0 ||
          operand0.operandType == VGPU10_OPERAND_TYPE_CONSTANT_BUFFER) {
         indexDim = VGPU10_OPERAND_INDEX_2D;
      }
      else {
         indexDim = VGPU10_OPERAND_INDEX_1D;
      }
   }

   /*
    * Compute index representations (immediate, relative, etc).
    */
   if (tempArrayID > 0) {
      assert(file == TGSI_FILE_TEMPORARY);
      /* First index is the array ID, second index is the array element */
      index0Rep = VGPU10_OPERAND_INDEX_IMMEDIATE32;
      if (indirect) {
         index1Rep = VGPU10_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE;
      }
      else {
         index1Rep = VGPU10_OPERAND_INDEX_IMMEDIATE32;
      }
   }
   else if (indirect) {
      if (file == TGSI_FILE_CONSTANT) {
         /* index[0] indicates which constant buffer while index[1] indicates
          * the position in the constant buffer.
          */
         index0Rep = VGPU10_OPERAND_INDEX_IMMEDIATE32;
         index1Rep = VGPU10_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE;
      }
      else {
         /* All other register files are 1-dimensional */
         index0Rep = VGPU10_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE;
      }
   }
   else {
      index0Rep = VGPU10_OPERAND_INDEX_IMMEDIATE32;
      index1Rep = VGPU10_OPERAND_INDEX_IMMEDIATE32;
   }

   operand0.indexDimension = indexDim;
   operand0.index0Representation = index0Rep;
   operand0.index1Representation = index1Rep;

   return operand0;
}


/**
 * Emit the operand for expressing an address register for indirect indexing.
 * Note that the address register is really just a temp register.
 * \param addr_reg_index  which address register to use
 */
static void
emit_indirect_register(struct svga_shader_emitter_v10 *emit,
                       unsigned addr_reg_index)
{
   unsigned tmp_reg_index;
   VGPU10OperandToken0 operand0;

   assert(addr_reg_index < MAX_VGPU10_ADDR_REGS);

   tmp_reg_index = emit->address_reg_index[addr_reg_index];

   /* operand0 is a simple temporary register, selecting one component */
   operand0.value = 0;
   operand0.operandType = VGPU10_OPERAND_TYPE_TEMP;
   operand0.numComponents = VGPU10_OPERAND_4_COMPONENT;
   operand0.indexDimension = VGPU10_OPERAND_INDEX_1D;
   operand0.index0Representation = VGPU10_OPERAND_INDEX_IMMEDIATE32;
   operand0.selectionMode = VGPU10_OPERAND_4_COMPONENT_SELECT_1_MODE;
   operand0.swizzleX = 0;
   operand0.swizzleY = 1;
   operand0.swizzleZ = 2;
   operand0.swizzleW = 3;

   emit_dword(emit, operand0.value);
   emit_dword(emit, remap_temp_index(emit, TGSI_FILE_TEMPORARY, tmp_reg_index));
}


/**
 * Translate the dst register of a TGSI instruction and emit VGPU10 tokens.
 * \param emit  the emitter context
 * \param reg  the TGSI dst register to translate
 */
static void
emit_dst_register(struct svga_shader_emitter_v10 *emit,
                  const struct tgsi_full_dst_register *reg)
{
   enum tgsi_file_type file = reg->Register.File;
   unsigned index = reg->Register.Index;
   const enum tgsi_semantic sem_name = emit->info.output_semantic_name[index];
   const unsigned sem_index = emit->info.output_semantic_index[index];
   unsigned writemask = reg->Register.WriteMask;
   const boolean indirect = reg->Register.Indirect;
   const unsigned tempArrayId = get_temp_array_id(emit, file, index);
   const boolean index2d = reg->Register.Dimension;
   VGPU10OperandToken0 operand0;

   if (file == TGSI_FILE_OUTPUT) {
      if (emit->unit == PIPE_SHADER_VERTEX ||
          emit->unit == PIPE_SHADER_GEOMETRY) {
         if (index == emit->vposition.out_index &&
             emit->vposition.tmp_index != INVALID_INDEX) {
            /* replace OUTPUT[POS] with TEMP[POS].  We need to store the
             * vertex position result in a temporary so that we can modify
             * it in the post_helper() code.
             */
            file = TGSI_FILE_TEMPORARY;
            index = emit->vposition.tmp_index;
         }
         else if (sem_name == TGSI_SEMANTIC_CLIPDIST &&
                  emit->clip_dist_tmp_index != INVALID_INDEX) {
            /* replace OUTPUT[CLIPDIST] with TEMP[CLIPDIST].
             * We store the clip distance in a temporary first, then
             * we'll copy it to the shadow copy and to CLIPDIST with the
             * enabled planes mask in emit_clip_distance_instructions().
             */
            file = TGSI_FILE_TEMPORARY;
            index = emit->clip_dist_tmp_index + sem_index;
         }
         else if (sem_name == TGSI_SEMANTIC_CLIPVERTEX &&
                  emit->clip_vertex_tmp_index != INVALID_INDEX) {
            /* replace the CLIPVERTEX output register with a temporary */
            assert(emit->clip_mode == CLIP_VERTEX);
            assert(sem_index == 0);
            file = TGSI_FILE_TEMPORARY;
            index = emit->clip_vertex_tmp_index;
         }
      }
      else if (emit->unit == PIPE_SHADER_FRAGMENT) {
         if (sem_name == TGSI_SEMANTIC_POSITION) {
            /* Fragment depth output register */
            operand0.value = 0;
            operand0.operandType = VGPU10_OPERAND_TYPE_OUTPUT_DEPTH;
            operand0.indexDimension = VGPU10_OPERAND_INDEX_0D;
            operand0.numComponents = VGPU10_OPERAND_1_COMPONENT;
            emit_dword(emit, operand0.value);
            return;
         }
         else if (sem_name == TGSI_SEMANTIC_SAMPLEMASK) {
            /* Fragment sample mask output */
            operand0.value = 0;
            operand0.operandType = VGPU10_OPERAND_TYPE_OUTPUT_COVERAGE_MASK;
            operand0.indexDimension = VGPU10_OPERAND_INDEX_0D;
            operand0.numComponents = VGPU10_OPERAND_1_COMPONENT;
            emit_dword(emit, operand0.value);
            return;
         }
         else if (index == emit->fs.color_out_index[0] &&
             emit->fs.color_tmp_index != INVALID_INDEX) {
            /* replace OUTPUT[COLOR] with TEMP[COLOR].  We need to store the
             * fragment color result in a temporary so that we can read it
             * it in the post_helper() code.
             */
            file = TGSI_FILE_TEMPORARY;
            index = emit->fs.color_tmp_index;
         }
         else {
            /* Typically, for fragment shaders, the output register index
             * matches the color semantic index.  But not when we write to
             * the fragment depth register.  In that case, OUT[0] will be
             * fragdepth and OUT[1] will be the 0th color output.  We need
             * to use the semantic index for color outputs.
             */
            assert(sem_name == TGSI_SEMANTIC_COLOR);
            index = emit->info.output_semantic_index[index];

            emit->num_output_writes++;
         }
      }
   }

   /* init operand tokens to all zero */
   operand0.value = 0;

   operand0.numComponents = VGPU10_OPERAND_4_COMPONENT;

   /* the operand has a writemask */
   operand0.selectionMode = VGPU10_OPERAND_4_COMPONENT_MASK_MODE;

   /* Which of the four dest components to write to. Note that we can use a
    * simple assignment here since TGSI writemasks match VGPU10 writemasks.
    */
   STATIC_ASSERT(TGSI_WRITEMASK_X == VGPU10_OPERAND_4_COMPONENT_MASK_X);
   operand0.mask = writemask;

   /* translate TGSI register file type to VGPU10 operand type */
   operand0.operandType = translate_register_file(file, tempArrayId > 0);

   check_register_index(emit, operand0.operandType, index);

   operand0 = setup_operand0_indexing(emit, operand0, file, indirect,
                                      index2d, tempArrayId);

   /* Emit tokens */
   emit_dword(emit, operand0.value);
   if (tempArrayId > 0) {
      emit_dword(emit, tempArrayId);
   }

   emit_dword(emit, remap_temp_index(emit, file, index));

   if (indirect) {
      emit_indirect_register(emit, reg->Indirect.Index);
   }
}


/**
 * Translate a src register of a TGSI instruction and emit VGPU10 tokens.
 * In quite a few cases, we do register substitution.  For example, if
 * the TGSI register is the front/back-face register, we replace that with
 * a temp register containing a value we computed earlier.
 */
static void
emit_src_register(struct svga_shader_emitter_v10 *emit,
                  const struct tgsi_full_src_register *reg)
{
   enum tgsi_file_type file = reg->Register.File;
   unsigned index = reg->Register.Index;
   const boolean indirect = reg->Register.Indirect;
   const unsigned tempArrayId = get_temp_array_id(emit, file, index);
   const boolean index2d = reg->Register.Dimension;
   const unsigned swizzleX = reg->Register.SwizzleX;
   const unsigned swizzleY = reg->Register.SwizzleY;
   const unsigned swizzleZ = reg->Register.SwizzleZ;
   const unsigned swizzleW = reg->Register.SwizzleW;
   const boolean absolute = reg->Register.Absolute;
   const boolean negate = reg->Register.Negate;
   bool is_prim_id = FALSE;

   VGPU10OperandToken0 operand0;
   VGPU10OperandToken1 operand1;

   if (emit->unit == PIPE_SHADER_FRAGMENT){
      if (file == TGSI_FILE_INPUT) {
         if (index == emit->fs.face_input_index) {
            /* Replace INPUT[FACE] with TEMP[FACE] */
            file = TGSI_FILE_TEMPORARY;
            index = emit->fs.face_tmp_index;
         }
         else if (index == emit->fs.fragcoord_input_index) {
            /* Replace INPUT[POSITION] with TEMP[POSITION] */
            file = TGSI_FILE_TEMPORARY;
            index = emit->fs.fragcoord_tmp_index;
         }
         else {
            /* We remap fragment shader inputs to that FS input indexes
             * match up with VS/GS output indexes.
             */
            index = emit->linkage.input_map[index];
         }
      }
      else if (file == TGSI_FILE_SYSTEM_VALUE) {
         if (index == emit->fs.sample_pos_sys_index) {
            assert(emit->version >= 41);
            /* Current sample position is in a temp register */
            file = TGSI_FILE_TEMPORARY;
            index = emit->fs.sample_pos_tmp_index;
         }
         else {
            /* Map the TGSI system value to a VGPU10 input register */
            assert(index < ARRAY_SIZE(emit->system_value_indexes));
            file = TGSI_FILE_INPUT;
            index = emit->system_value_indexes[index];
         }
      }
   }
   else if (emit->unit == PIPE_SHADER_GEOMETRY) {
      if (file == TGSI_FILE_INPUT) {
         is_prim_id = (index == emit->gs.prim_id_index);
         index = emit->linkage.input_map[index];
      }
   }
   else if (emit->unit == PIPE_SHADER_VERTEX) {
      if (file == TGSI_FILE_INPUT) {
         /* if input is adjusted... */
         if ((emit->key.vs.adjust_attrib_w_1 |
              emit->key.vs.adjust_attrib_itof |
              emit->key.vs.adjust_attrib_utof |
              emit->key.vs.attrib_is_bgra |
              emit->key.vs.attrib_puint_to_snorm |
              emit->key.vs.attrib_puint_to_uscaled |
              emit->key.vs.attrib_puint_to_sscaled) & (1 << index)) {
            file = TGSI_FILE_TEMPORARY;
            index = emit->vs.adjusted_input[index];
         }
      }
      else if (file == TGSI_FILE_SYSTEM_VALUE) {
         /* Map the TGSI system value to a VGPU10 input register */
         assert(index < ARRAY_SIZE(emit->system_value_indexes));
         file = TGSI_FILE_INPUT;
         index = emit->system_value_indexes[index];
      }
   }

   operand0.value = operand1.value = 0;

   if (is_prim_id) {
      /* NOTE: we should be using VGPU10_OPERAND_1_COMPONENT here, but
       * our virtual GPU accepts this as-is.
       */
      operand0.numComponents = VGPU10_OPERAND_0_COMPONENT;
      operand0.operandType = VGPU10_OPERAND_TYPE_INPUT_PRIMITIVEID;
   }
   else {
      operand0.numComponents = VGPU10_OPERAND_4_COMPONENT;
      operand0.operandType = translate_register_file(file, tempArrayId > 0);
   }

   operand0 = setup_operand0_indexing(emit, operand0, file, indirect,
                                      index2d, tempArrayId);

   if (operand0.operandType != VGPU10_OPERAND_TYPE_IMMEDIATE32 &&
       operand0.operandType != VGPU10_OPERAND_TYPE_INPUT_PRIMITIVEID) {
      /* there's no swizzle for in-line immediates */
      if (swizzleX == swizzleY &&
          swizzleX == swizzleZ &&
          swizzleX == swizzleW) {
         operand0.selectionMode = VGPU10_OPERAND_4_COMPONENT_SELECT_1_MODE;
      }
      else {
         operand0.selectionMode = VGPU10_OPERAND_4_COMPONENT_SWIZZLE_MODE;
      }

      operand0.swizzleX = swizzleX;
      operand0.swizzleY = swizzleY;
      operand0.swizzleZ = swizzleZ;
      operand0.swizzleW = swizzleW;

      if (absolute || negate) {
         operand0.extended = 1;
         operand1.extendedOperandType = VGPU10_EXTENDED_OPERAND_MODIFIER;
         if (absolute && !negate)
            operand1.operandModifier = VGPU10_OPERAND_MODIFIER_ABS;
         if (!absolute && negate)
            operand1.operandModifier = VGPU10_OPERAND_MODIFIER_NEG;
         if (absolute && negate)
            operand1.operandModifier = VGPU10_OPERAND_MODIFIER_ABSNEG;
      }
   }

   /* Emit the operand tokens */
   emit_dword(emit, operand0.value);
   if (operand0.extended)
      emit_dword(emit, operand1.value);

   if (operand0.operandType == VGPU10_OPERAND_TYPE_IMMEDIATE32) {
      /* Emit the four float/int in-line immediate values */
      unsigned *c;
      assert(index < ARRAY_SIZE(emit->immediates));
      assert(file == TGSI_FILE_IMMEDIATE);
      assert(swizzleX < 4);
      assert(swizzleY < 4);
      assert(swizzleZ < 4);
      assert(swizzleW < 4);
      c = (unsigned *) emit->immediates[index];
      emit_dword(emit, c[swizzleX]);
      emit_dword(emit, c[swizzleY]);
      emit_dword(emit, c[swizzleZ]);
      emit_dword(emit, c[swizzleW]);
   }
   else if (operand0.indexDimension >= VGPU10_OPERAND_INDEX_1D) {
      /* Emit the register index(es) */
      if (index2d ||
          operand0.operandType == VGPU10_OPERAND_TYPE_CONSTANT_BUFFER) {
         emit_dword(emit, reg->Dimension.Index);
      }

      if (tempArrayId > 0) {
         emit_dword(emit, tempArrayId);
      }

      emit_dword(emit, remap_temp_index(emit, file, index));

      if (indirect) {
         emit_indirect_register(emit, reg->Indirect.Index);
      }
   }
}


/**
 * Emit a resource operand (for use with a SAMPLE instruction).
 */
static void
emit_resource_register(struct svga_shader_emitter_v10 *emit,
                       unsigned resource_number)
{
   VGPU10OperandToken0 operand0;

   check_register_index(emit, VGPU10_OPERAND_TYPE_RESOURCE, resource_number);

   /* init */
   operand0.value = 0;

   operand0.operandType = VGPU10_OPERAND_TYPE_RESOURCE;
   operand0.indexDimension = VGPU10_OPERAND_INDEX_1D;
   operand0.numComponents = VGPU10_OPERAND_4_COMPONENT;
   operand0.selectionMode = VGPU10_OPERAND_4_COMPONENT_SWIZZLE_MODE;
   operand0.swizzleX = VGPU10_COMPONENT_X;
   operand0.swizzleY = VGPU10_COMPONENT_Y;
   operand0.swizzleZ = VGPU10_COMPONENT_Z;
   operand0.swizzleW = VGPU10_COMPONENT_W;

   emit_dword(emit, operand0.value);
   emit_dword(emit, resource_number);
}


/**
 * Emit a sampler operand (for use with a SAMPLE instruction).
 */
static void
emit_sampler_register(struct svga_shader_emitter_v10 *emit,
                      unsigned sampler_number)
{
   VGPU10OperandToken0 operand0;

   check_register_index(emit, VGPU10_OPERAND_TYPE_SAMPLER, sampler_number);

   /* init */
   operand0.value = 0;

   operand0.operandType = VGPU10_OPERAND_TYPE_SAMPLER;
   operand0.indexDimension = VGPU10_OPERAND_INDEX_1D;

   emit_dword(emit, operand0.value);
   emit_dword(emit, sampler_number);
}


/**
 * Emit an operand which reads the IS_FRONT_FACING register.
 */
static void
emit_face_register(struct svga_shader_emitter_v10 *emit)
{
   VGPU10OperandToken0 operand0;
   unsigned index = emit->linkage.input_map[emit->fs.face_input_index];

   /* init */
   operand0.value = 0;

   operand0.operandType = VGPU10_OPERAND_TYPE_INPUT;
   operand0.indexDimension = VGPU10_OPERAND_INDEX_1D;
   operand0.selectionMode = VGPU10_OPERAND_4_COMPONENT_SELECT_1_MODE;
   operand0.numComponents = VGPU10_OPERAND_4_COMPONENT;

   operand0.swizzleX = VGPU10_COMPONENT_X;
   operand0.swizzleY = VGPU10_COMPONENT_X;
   operand0.swizzleZ = VGPU10_COMPONENT_X;
   operand0.swizzleW = VGPU10_COMPONENT_X;

   emit_dword(emit, operand0.value);
   emit_dword(emit, index);
}


/**
 * Emit tokens for the "rasterizer" register used by the SAMPLE_POS
 * instruction.
 */
static void
emit_rasterizer_register(struct svga_shader_emitter_v10 *emit)
{
   VGPU10OperandToken0 operand0;

   /* init */
   operand0.value = 0;

   /* No register index for rasterizer index (there's only one) */
   operand0.operandType = VGPU10_OPERAND_TYPE_RASTERIZER;
   operand0.indexDimension = VGPU10_OPERAND_INDEX_0D;
   operand0.numComponents = VGPU10_OPERAND_4_COMPONENT;
   operand0.selectionMode = VGPU10_OPERAND_4_COMPONENT_SWIZZLE_MODE;
   operand0.swizzleX = VGPU10_COMPONENT_X;
   operand0.swizzleY = VGPU10_COMPONENT_Y;
   operand0.swizzleZ = VGPU10_COMPONENT_Z;
   operand0.swizzleW = VGPU10_COMPONENT_W;

   emit_dword(emit, operand0.value);
}


/**
 * Emit the token for a VGPU10 opcode.
 * \param saturate   clamp result to [0,1]?
 */
static void
emit_opcode(struct svga_shader_emitter_v10 *emit,
            VGPU10_OPCODE_TYPE vgpu10_opcode, boolean saturate)
{
   VGPU10OpcodeToken0 token0;

   token0.value = 0;  /* init all fields to zero */
   token0.opcodeType = vgpu10_opcode;
   token0.instructionLength = 0; /* Filled in by end_emit_instruction() */
   token0.saturate = saturate;

   emit_dword(emit, token0.value);
}


/**
 * Emit the token for a VGPU10 resinfo instruction.
 * \param modifier   return type modifier, _uint or _rcpFloat.
 *                   TODO: We may want to remove this parameter if it will
 *                   only ever be used as _uint.
 */
static void
emit_opcode_resinfo(struct svga_shader_emitter_v10 *emit,
                    VGPU10_RESINFO_RETURN_TYPE modifier)
{
   VGPU10OpcodeToken0 token0;

   token0.value = 0;  /* init all fields to zero */
   token0.opcodeType = VGPU10_OPCODE_RESINFO;
   token0.instructionLength = 0; /* Filled in by end_emit_instruction() */
   token0.resinfoReturnType = modifier;

   emit_dword(emit, token0.value);
}


/**
 * Emit opcode tokens for a texture sample instruction.  Texture instructions
 * can be rather complicated (texel offsets, etc) so we have this specialized
 * function.
 */
static void
emit_sample_opcode(struct svga_shader_emitter_v10 *emit,
                   unsigned vgpu10_opcode, boolean saturate,
                   const int offsets[3])
{
   VGPU10OpcodeToken0 token0;
   VGPU10OpcodeToken1 token1;

   token0.value = 0;  /* init all fields to zero */
   token0.opcodeType = vgpu10_opcode;
   token0.instructionLength = 0; /* Filled in by end_emit_instruction() */
   token0.saturate = saturate;

   if (offsets[0] || offsets[1] || offsets[2]) {
      assert(offsets[0] >= VGPU10_MIN_TEXEL_FETCH_OFFSET);
      assert(offsets[1] >= VGPU10_MIN_TEXEL_FETCH_OFFSET);
      assert(offsets[2] >= VGPU10_MIN_TEXEL_FETCH_OFFSET);
      assert(offsets[0] <= VGPU10_MAX_TEXEL_FETCH_OFFSET);
      assert(offsets[1] <= VGPU10_MAX_TEXEL_FETCH_OFFSET);
      assert(offsets[2] <= VGPU10_MAX_TEXEL_FETCH_OFFSET);

      token0.extended = 1;
      token1.value = 0;
      token1.opcodeType = VGPU10_EXTENDED_OPCODE_SAMPLE_CONTROLS;
      token1.offsetU = offsets[0];
      token1.offsetV = offsets[1];
      token1.offsetW = offsets[2];
   }

   emit_dword(emit, token0.value);
   if (token0.extended) {
      emit_dword(emit, token1.value);
   }
}


/**
 * Emit a DISCARD opcode token.
 * If nonzero is set, we'll discard the fragment if the X component is not 0.
 * Otherwise, we'll discard the fragment if the X component is 0.
 */
static void
emit_discard_opcode(struct svga_shader_emitter_v10 *emit, boolean nonzero)
{
   VGPU10OpcodeToken0 opcode0;

   opcode0.value = 0;
   opcode0.opcodeType = VGPU10_OPCODE_DISCARD;
   if (nonzero)
      opcode0.testBoolean = VGPU10_INSTRUCTION_TEST_NONZERO;

   emit_dword(emit, opcode0.value);
}


/**
 * We need to call this before we begin emitting a VGPU10 instruction.
 */
static void
begin_emit_instruction(struct svga_shader_emitter_v10 *emit)
{
   assert(emit->inst_start_token == 0);
   /* Save location of the instruction's VGPU10OpcodeToken0 token.
    * Note, we can't save a pointer because it would become invalid if
    * we have to realloc the output buffer.
    */
   emit->inst_start_token = emit_get_num_tokens(emit);
}


/**
 * We need to call this after we emit the last token of a VGPU10 instruction.
 * This function patches in the opcode token's instructionLength field.
 */
static void
end_emit_instruction(struct svga_shader_emitter_v10 *emit)
{
   VGPU10OpcodeToken0 *tokens = (VGPU10OpcodeToken0 *) emit->buf;
   unsigned inst_length;

   assert(emit->inst_start_token > 0);

   if (emit->discard_instruction) {
      /* Back up the emit->ptr to where this instruction started so
       * that we discard the current instruction.
       */
      emit->ptr = (char *) (tokens + emit->inst_start_token);
   }
   else {
      /* Compute instruction length and patch that into the start of
       * the instruction.
       */
      inst_length = emit_get_num_tokens(emit) - emit->inst_start_token;

      assert(inst_length > 0);

      tokens[emit->inst_start_token].instructionLength = inst_length;
   }

   emit->inst_start_token = 0; /* reset to zero for error checking */
   emit->discard_instruction = FALSE;
}


/**
 * Return index for a free temporary register.
 */
static unsigned
get_temp_index(struct svga_shader_emitter_v10 *emit)
{
   assert(emit->internal_temp_count < MAX_INTERNAL_TEMPS);
   return emit->num_shader_temps + emit->internal_temp_count++;
}


/**
 * Release the temporaries which were generated by get_temp_index().
 */
static void
free_temp_indexes(struct svga_shader_emitter_v10 *emit)
{
   emit->internal_temp_count = 0;
}


/**
 * Create a tgsi_full_src_register.
 */
static struct tgsi_full_src_register
make_src_reg(enum tgsi_file_type file, unsigned index)
{
   struct tgsi_full_src_register reg;

   memset(&reg, 0, sizeof(reg));
   reg.Register.File = file;
   reg.Register.Index = index;
   reg.Register.SwizzleX = TGSI_SWIZZLE_X;
   reg.Register.SwizzleY = TGSI_SWIZZLE_Y;
   reg.Register.SwizzleZ = TGSI_SWIZZLE_Z;
   reg.Register.SwizzleW = TGSI_SWIZZLE_W;
   return reg;
}


/**
 * Create a tgsi_full_src_register with a swizzle such that all four
 * vector components have the same scalar value.
 */
static struct tgsi_full_src_register
make_src_scalar_reg(enum tgsi_file_type file, unsigned index, unsigned component)
{
   struct tgsi_full_src_register reg;

   assert(component >= TGSI_SWIZZLE_X);
   assert(component <= TGSI_SWIZZLE_W);

   memset(&reg, 0, sizeof(reg));
   reg.Register.File = file;
   reg.Register.Index = index;
   reg.Register.SwizzleX =
   reg.Register.SwizzleY =
   reg.Register.SwizzleZ =
   reg.Register.SwizzleW = component;
   return reg;
}


/**
 * Create a tgsi_full_src_register for a temporary.
 */
static struct tgsi_full_src_register
make_src_temp_reg(unsigned index)
{
   return make_src_reg(TGSI_FILE_TEMPORARY, index);
}


/**
 * Create a tgsi_full_src_register for a constant.
 */
static struct tgsi_full_src_register
make_src_const_reg(unsigned index)
{
   return make_src_reg(TGSI_FILE_CONSTANT, index);
}


/**
 * Create a tgsi_full_src_register for an immediate constant.
 */
static struct tgsi_full_src_register
make_src_immediate_reg(unsigned index)
{
   return make_src_reg(TGSI_FILE_IMMEDIATE, index);
}


/**
 * Create a tgsi_full_dst_register.
 */
static struct tgsi_full_dst_register
make_dst_reg(enum tgsi_file_type file, unsigned index)
{
   struct tgsi_full_dst_register reg;

   memset(&reg, 0, sizeof(reg));
   reg.Register.File = file;
   reg.Register.Index = index;
   reg.Register.WriteMask = TGSI_WRITEMASK_XYZW;
   return reg;
}


/**
 * Create a tgsi_full_dst_register for a temporary.
 */
static struct tgsi_full_dst_register
make_dst_temp_reg(unsigned index)
{
   return make_dst_reg(TGSI_FILE_TEMPORARY, index);
}


/**
 * Create a tgsi_full_dst_register for an output.
 */
static struct tgsi_full_dst_register
make_dst_output_reg(unsigned index)
{
   return make_dst_reg(TGSI_FILE_OUTPUT, index);
}


/**
 * Create negated tgsi_full_src_register.
 */
static struct tgsi_full_src_register
negate_src(const struct tgsi_full_src_register *reg)
{
   struct tgsi_full_src_register neg = *reg;
   neg.Register.Negate = !reg->Register.Negate;
   return neg;
}

/**
 * Create absolute value of a tgsi_full_src_register.
 */
static struct tgsi_full_src_register
absolute_src(const struct tgsi_full_src_register *reg)
{
   struct tgsi_full_src_register absolute = *reg;
   absolute.Register.Absolute = 1;
   return absolute;
}


/** Return the named swizzle term from the src register */
static inline unsigned
get_swizzle(const struct tgsi_full_src_register *reg, enum tgsi_swizzle term)
{
   switch (term) {
   case TGSI_SWIZZLE_X:
      return reg->Register.SwizzleX;
   case TGSI_SWIZZLE_Y:
      return reg->Register.SwizzleY;
   case TGSI_SWIZZLE_Z:
      return reg->Register.SwizzleZ;
   case TGSI_SWIZZLE_W:
      return reg->Register.SwizzleW;
   default:
      assert(!"Bad swizzle");
      return TGSI_SWIZZLE_X;
   }
}


/**
 * Create swizzled tgsi_full_src_register.
 */
static struct tgsi_full_src_register
swizzle_src(const struct tgsi_full_src_register *reg,
            enum tgsi_swizzle swizzleX, enum tgsi_swizzle swizzleY,
            enum tgsi_swizzle swizzleZ, enum tgsi_swizzle swizzleW)
{
   struct tgsi_full_src_register swizzled = *reg;
   /* Note: we swizzle the current swizzle */
   swizzled.Register.SwizzleX = get_swizzle(reg, swizzleX);
   swizzled.Register.SwizzleY = get_swizzle(reg, swizzleY);
   swizzled.Register.SwizzleZ = get_swizzle(reg, swizzleZ);
   swizzled.Register.SwizzleW = get_swizzle(reg, swizzleW);
   return swizzled;
}


/**
 * Create swizzled tgsi_full_src_register where all the swizzle
 * terms are the same.
 */
static struct tgsi_full_src_register
scalar_src(const struct tgsi_full_src_register *reg, enum tgsi_swizzle swizzle)
{
   struct tgsi_full_src_register swizzled = *reg;
   /* Note: we swizzle the current swizzle */
   swizzled.Register.SwizzleX =
   swizzled.Register.SwizzleY =
   swizzled.Register.SwizzleZ =
   swizzled.Register.SwizzleW = get_swizzle(reg, swizzle);
   return swizzled;
}


/**
 * Create new tgsi_full_dst_register with writemask.
 * \param mask  bitmask of TGSI_WRITEMASK_[XYZW]
 */
static struct tgsi_full_dst_register
writemask_dst(const struct tgsi_full_dst_register *reg, unsigned mask)
{
   struct tgsi_full_dst_register masked = *reg;
   masked.Register.WriteMask = mask;
   return masked;
}


/**
 * Check if the register's swizzle is XXXX, YYYY, ZZZZ, or WWWW.
 */
static boolean
same_swizzle_terms(const struct tgsi_full_src_register *reg)
{
   return (reg->Register.SwizzleX == reg->Register.SwizzleY &&
           reg->Register.SwizzleY == reg->Register.SwizzleZ &&
           reg->Register.SwizzleZ == reg->Register.SwizzleW);
}


/**
 * Search the vector for the value 'x' and return its position.
 */
static int
find_imm_in_vec4(const union tgsi_immediate_data vec[4],
                 union tgsi_immediate_data x)
{
   unsigned i;
   for (i = 0; i < 4; i++) {
      if (vec[i].Int == x.Int)
         return i;
   }
   return -1;
}


/**
 * Helper used by make_immediate_reg(), make_immediate_reg_4().
 */
static int
find_immediate(struct svga_shader_emitter_v10 *emit,
               union tgsi_immediate_data x, unsigned startIndex)
{
   const unsigned endIndex = emit->num_immediates;
   unsigned i;

   assert(emit->immediates_emitted);

   /* Search immediates for x, y, z, w */
   for (i = startIndex; i < endIndex; i++) {
      if (x.Int == emit->immediates[i][0].Int ||
          x.Int == emit->immediates[i][1].Int ||
          x.Int == emit->immediates[i][2].Int ||
          x.Int == emit->immediates[i][3].Int) {
         return i;
      }
   }
   /* Should never try to use an immediate value that wasn't pre-declared */
   assert(!"find_immediate() failed!");
   return -1;
}


/**
 * Return a tgsi_full_src_register for an immediate/literal
 * union tgsi_immediate_data[4] value.
 * Note: the values must have been previously declared/allocated in
 * emit_pre_helpers().  And, all of x,y,z,w must be located in the same
 * vec4 immediate.
 */
static struct tgsi_full_src_register
make_immediate_reg_4(struct svga_shader_emitter_v10 *emit,
                     const union tgsi_immediate_data imm[4])
{
   struct tgsi_full_src_register reg;
   unsigned i;

   for (i = 0; i < emit->num_common_immediates; i++) {
      /* search for first component value */
      int immpos = find_immediate(emit, imm[0], i);
      int x, y, z, w;

      assert(immpos >= 0);

      /* find remaining components within the immediate vector */
      x = find_imm_in_vec4(emit->immediates[immpos], imm[0]);
      y = find_imm_in_vec4(emit->immediates[immpos], imm[1]);
      z = find_imm_in_vec4(emit->immediates[immpos], imm[2]);
      w = find_imm_in_vec4(emit->immediates[immpos], imm[3]);

      if (x >=0 &&  y >= 0 && z >= 0 && w >= 0) {
         /* found them all */
         memset(&reg, 0, sizeof(reg));
         reg.Register.File = TGSI_FILE_IMMEDIATE;
         reg.Register.Index = immpos;
         reg.Register.SwizzleX = x;
         reg.Register.SwizzleY = y;
         reg.Register.SwizzleZ = z;
         reg.Register.SwizzleW = w;
         return reg;
      }
      /* else, keep searching */
   }

   assert(!"Failed to find immediate register!");

   /* Just return IMM[0].xxxx */
   memset(&reg, 0, sizeof(reg));
   reg.Register.File = TGSI_FILE_IMMEDIATE;
   return reg;
}


/**
 * Return a tgsi_full_src_register for an immediate/literal
 * union tgsi_immediate_data value of the form {value, value, value, value}.
 * \sa make_immediate_reg_4() regarding allowed values.
 */
static struct tgsi_full_src_register
make_immediate_reg(struct svga_shader_emitter_v10 *emit,
                   union tgsi_immediate_data value)
{
   struct tgsi_full_src_register reg;
   int immpos = find_immediate(emit, value, 0);

   assert(immpos >= 0);

   memset(&reg, 0, sizeof(reg));
   reg.Register.File = TGSI_FILE_IMMEDIATE;
   reg.Register.Index = immpos;
   reg.Register.SwizzleX =
   reg.Register.SwizzleY =
   reg.Register.SwizzleZ =
   reg.Register.SwizzleW = find_imm_in_vec4(emit->immediates[immpos], value);

   return reg;
}


/**
 * Return a tgsi_full_src_register for an immediate/literal float[4] value.
 * \sa make_immediate_reg_4() regarding allowed values.
 */
static struct tgsi_full_src_register
make_immediate_reg_float4(struct svga_shader_emitter_v10 *emit,
                          float x, float y, float z, float w)
{
   union tgsi_immediate_data imm[4];
   imm[0].Float = x;
   imm[1].Float = y;
   imm[2].Float = z;
   imm[3].Float = w;
   return make_immediate_reg_4(emit, imm);
}


/**
 * Return a tgsi_full_src_register for an immediate/literal float value
 * of the form {value, value, value, value}.
 * \sa make_immediate_reg_4() regarding allowed values.
 */
static struct tgsi_full_src_register
make_immediate_reg_float(struct svga_shader_emitter_v10 *emit, float value)
{
   union tgsi_immediate_data imm;
   imm.Float = value;
   return make_immediate_reg(emit, imm);
}


/**
 * Return a tgsi_full_src_register for an immediate/literal int[4] vector.
 */
static struct tgsi_full_src_register
make_immediate_reg_int4(struct svga_shader_emitter_v10 *emit,
                        int x, int y, int z, int w)
{
   union tgsi_immediate_data imm[4];
   imm[0].Int = x;
   imm[1].Int = y;
   imm[2].Int = z;
   imm[3].Int = w;
   return make_immediate_reg_4(emit, imm);
}


/**
 * Return a tgsi_full_src_register for an immediate/literal int value
 * of the form {value, value, value, value}.
 * \sa make_immediate_reg_4() regarding allowed values.
 */
static struct tgsi_full_src_register
make_immediate_reg_int(struct svga_shader_emitter_v10 *emit, int value)
{
   union tgsi_immediate_data imm;
   imm.Int = value;
   return make_immediate_reg(emit, imm);
}


/**
 * Allocate space for a union tgsi_immediate_data[4] immediate.
 * \return  the index/position of the immediate.
 */
static unsigned
alloc_immediate_4(struct svga_shader_emitter_v10 *emit,
                  const union tgsi_immediate_data imm[4])
{
   unsigned n = emit->num_immediates++;
   assert(!emit->immediates_emitted);
   assert(n < ARRAY_SIZE(emit->immediates));
   emit->immediates[n][0] = imm[0];
   emit->immediates[n][1] = imm[1];
   emit->immediates[n][2] = imm[2];
   emit->immediates[n][3] = imm[3];
   return n;
}


/**
 * Allocate space for a float[4] immediate.
 * \return  the index/position of the immediate.
 */
static unsigned
alloc_immediate_float4(struct svga_shader_emitter_v10 *emit,
                       float x, float y, float z, float w)
{
   union tgsi_immediate_data imm[4];
   imm[0].Float = x;
   imm[1].Float = y;
   imm[2].Float = z;
   imm[3].Float = w;
   return alloc_immediate_4(emit, imm);
}


/**
 * Allocate space for an int[4] immediate.
 * \return  the index/position of the immediate.
 */
static unsigned
alloc_immediate_int4(struct svga_shader_emitter_v10 *emit,
                       int x, int y, int z, int w)
{
   union tgsi_immediate_data imm[4];
   imm[0].Int = x;
   imm[1].Int = y;
   imm[2].Int = z;
   imm[3].Int = w;
   return alloc_immediate_4(emit, imm);
}


/**
 * Allocate a shader input to store a system value.
 */
static unsigned
alloc_system_value_index(struct svga_shader_emitter_v10 *emit, unsigned index)
{
   const unsigned n = emit->linkage.input_map_max + 1 + index;
   assert(index < ARRAY_SIZE(emit->system_value_indexes));
   emit->system_value_indexes[index] = n;
   return n;
}


/**
 * Translate a TGSI immediate value (union tgsi_immediate_data[4]) to VGPU10.
 */
static boolean
emit_vgpu10_immediate(struct svga_shader_emitter_v10 *emit,
                      const struct tgsi_full_immediate *imm)
{
   /* We don't actually emit any code here.  We just save the
    * immediate values and emit them later.
    */
   alloc_immediate_4(emit, imm->u);
   return TRUE;
}


/**
 * Emit a VGPU10_CUSTOMDATA_DCL_IMMEDIATE_CONSTANT_BUFFER block
 * containing all the immediate values previously allocated
 * with alloc_immediate_4().
 */
static boolean
emit_vgpu10_immediates_block(struct svga_shader_emitter_v10 *emit)
{
   VGPU10OpcodeToken0 token;

   assert(!emit->immediates_emitted);

   token.value = 0;
   token.opcodeType = VGPU10_OPCODE_CUSTOMDATA;
   token.customDataClass = VGPU10_CUSTOMDATA_DCL_IMMEDIATE_CONSTANT_BUFFER;

   /* Note: no begin/end_emit_instruction() calls */
   emit_dword(emit, token.value);
   emit_dword(emit, 2 + 4 * emit->num_immediates);
   emit_dwords(emit, (unsigned *) emit->immediates, 4 * emit->num_immediates);

   emit->immediates_emitted = TRUE;

   return TRUE;
}


/**
 * Translate a fragment shader's TGSI_INTERPOLATE_x mode to a vgpu10
 * interpolation mode.
 * \return a VGPU10_INTERPOLATION_x value
 */
static unsigned
translate_interpolation(const struct svga_shader_emitter_v10 *emit,
                        enum tgsi_interpolate_mode interp,
                        enum tgsi_interpolate_loc interpolate_loc)
{
   if (interp == TGSI_INTERPOLATE_COLOR) {
      interp = emit->key.fs.flatshade ?
         TGSI_INTERPOLATE_CONSTANT : TGSI_INTERPOLATE_PERSPECTIVE;
   }

   switch (interp) {
   case TGSI_INTERPOLATE_CONSTANT:
      return VGPU10_INTERPOLATION_CONSTANT;
   case TGSI_INTERPOLATE_LINEAR:
      if (interpolate_loc == TGSI_INTERPOLATE_LOC_CENTROID) {
         return VGPU10_INTERPOLATION_LINEAR_NOPERSPECTIVE_CENTROID;
      } else if (interpolate_loc == TGSI_INTERPOLATE_LOC_SAMPLE &&
                 emit->version >= 41) {
         return VGPU10_INTERPOLATION_LINEAR_NOPERSPECTIVE_SAMPLE;
      } else {
         return VGPU10_INTERPOLATION_LINEAR_NOPERSPECTIVE;
      }
      break;
   case TGSI_INTERPOLATE_PERSPECTIVE:
      if (interpolate_loc == TGSI_INTERPOLATE_LOC_CENTROID) {
         return VGPU10_INTERPOLATION_LINEAR_CENTROID;
      } else if (interpolate_loc == TGSI_INTERPOLATE_LOC_SAMPLE &&
                 emit->version >= 41) {
         return VGPU10_INTERPOLATION_LINEAR_SAMPLE;
      } else {
         return VGPU10_INTERPOLATION_LINEAR;
      }
      break;
   default:
      assert(!"Unexpected interpolation mode");
      return VGPU10_INTERPOLATION_CONSTANT;
   }
}


/**
 * Translate a TGSI property to VGPU10.
 * Don't emit any instructions yet, only need to gather the primitive property
 * information.  The output primitive topology might be changed later. The
 * final property instructions will be emitted as part of the pre-helper code.
 */
static boolean
emit_vgpu10_property(struct svga_shader_emitter_v10 *emit,
                     const struct tgsi_full_property *prop)
{
   static const VGPU10_PRIMITIVE primType[] = {
      VGPU10_PRIMITIVE_POINT,           /* PIPE_PRIM_POINTS */
      VGPU10_PRIMITIVE_LINE,            /* PIPE_PRIM_LINES */
      VGPU10_PRIMITIVE_LINE,            /* PIPE_PRIM_LINE_LOOP */
      VGPU10_PRIMITIVE_LINE,            /* PIPE_PRIM_LINE_STRIP */
      VGPU10_PRIMITIVE_TRIANGLE,        /* PIPE_PRIM_TRIANGLES */
      VGPU10_PRIMITIVE_TRIANGLE,        /* PIPE_PRIM_TRIANGLE_STRIP */
      VGPU10_PRIMITIVE_TRIANGLE,        /* PIPE_PRIM_TRIANGLE_FAN */
      VGPU10_PRIMITIVE_UNDEFINED,       /* PIPE_PRIM_QUADS */
      VGPU10_PRIMITIVE_UNDEFINED,       /* PIPE_PRIM_QUAD_STRIP */
      VGPU10_PRIMITIVE_UNDEFINED,       /* PIPE_PRIM_POLYGON */
      VGPU10_PRIMITIVE_LINE_ADJ,        /* PIPE_PRIM_LINES_ADJACENCY */
      VGPU10_PRIMITIVE_LINE_ADJ,        /* PIPE_PRIM_LINE_STRIP_ADJACENCY */
      VGPU10_PRIMITIVE_TRIANGLE_ADJ,    /* PIPE_PRIM_TRIANGLES_ADJACENCY */
      VGPU10_PRIMITIVE_TRIANGLE_ADJ     /* PIPE_PRIM_TRIANGLE_STRIP_ADJACENCY */
   };

   static const VGPU10_PRIMITIVE_TOPOLOGY primTopology[] = {
      VGPU10_PRIMITIVE_TOPOLOGY_POINTLIST,     /* PIPE_PRIM_POINTS */
      VGPU10_PRIMITIVE_TOPOLOGY_LINELIST,      /* PIPE_PRIM_LINES */
      VGPU10_PRIMITIVE_TOPOLOGY_LINELIST,      /* PIPE_PRIM_LINE_LOOP */
      VGPU10_PRIMITIVE_TOPOLOGY_LINESTRIP,     /* PIPE_PRIM_LINE_STRIP */
      VGPU10_PRIMITIVE_TOPOLOGY_TRIANGLELIST,  /* PIPE_PRIM_TRIANGLES */
      VGPU10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, /* PIPE_PRIM_TRIANGLE_STRIP */
      VGPU10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, /* PIPE_PRIM_TRIANGLE_FAN */
      VGPU10_PRIMITIVE_TOPOLOGY_UNDEFINED,     /* PIPE_PRIM_QUADS */
      VGPU10_PRIMITIVE_TOPOLOGY_UNDEFINED,     /* PIPE_PRIM_QUAD_STRIP */
      VGPU10_PRIMITIVE_TOPOLOGY_UNDEFINED,     /* PIPE_PRIM_POLYGON */
      VGPU10_PRIMITIVE_TOPOLOGY_LINELIST_ADJ,  /* PIPE_PRIM_LINES_ADJACENCY */
      VGPU10_PRIMITIVE_TOPOLOGY_LINELIST_ADJ,  /* PIPE_PRIM_LINE_STRIP_ADJACENCY */
      VGPU10_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ, /* PIPE_PRIM_TRIANGLES_ADJACENCY */
      VGPU10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ /* PIPE_PRIM_TRIANGLE_STRIP_ADJACENCY */
   };

   static const unsigned inputArraySize[] = {
      0,       /* VGPU10_PRIMITIVE_UNDEFINED */
      1,       /* VGPU10_PRIMITIVE_POINT */
      2,       /* VGPU10_PRIMITIVE_LINE */
      3,       /* VGPU10_PRIMITIVE_TRIANGLE */
      0,
      0,
      4,       /* VGPU10_PRIMITIVE_LINE_ADJ */
      6        /* VGPU10_PRIMITIVE_TRIANGLE_ADJ */
   };

   switch (prop->Property.PropertyName) {
   case TGSI_PROPERTY_GS_INPUT_PRIM:
      assert(prop->u[0].Data < ARRAY_SIZE(primType));
      emit->gs.prim_type = primType[prop->u[0].Data];
      assert(emit->gs.prim_type != VGPU10_PRIMITIVE_UNDEFINED);
      emit->gs.input_size = inputArraySize[emit->gs.prim_type];
      break;

   case TGSI_PROPERTY_GS_OUTPUT_PRIM:
      assert(prop->u[0].Data < ARRAY_SIZE(primTopology));
      emit->gs.prim_topology = primTopology[prop->u[0].Data];
      assert(emit->gs.prim_topology != VGPU10_PRIMITIVE_TOPOLOGY_UNDEFINED);
      break;

   case TGSI_PROPERTY_GS_MAX_OUTPUT_VERTICES:
      emit->gs.max_out_vertices = prop->u[0].Data;
      break;

   default:
      break;
   }

   return TRUE;
}


static void
emit_property_instruction(struct svga_shader_emitter_v10 *emit,
                          VGPU10OpcodeToken0 opcode0, unsigned nData,
                          unsigned data)
{
   begin_emit_instruction(emit);
   emit_dword(emit, opcode0.value);
   if (nData)
      emit_dword(emit, data);
   end_emit_instruction(emit);
}


/**
 * Emit property instructions
 */
static void
emit_property_instructions(struct svga_shader_emitter_v10 *emit)
{
   VGPU10OpcodeToken0 opcode0;

   assert(emit->unit == PIPE_SHADER_GEOMETRY);

   /* emit input primitive type declaration */
   opcode0.value = 0;
   opcode0.opcodeType = VGPU10_OPCODE_DCL_GS_INPUT_PRIMITIVE;
   opcode0.primitive = emit->gs.prim_type;
   emit_property_instruction(emit, opcode0, 0, 0);

   /* emit output primitive topology declaration */
   opcode0.value = 0;
   opcode0.opcodeType = VGPU10_OPCODE_DCL_GS_OUTPUT_PRIMITIVE_TOPOLOGY;
   opcode0.primitiveTopology = emit->gs.prim_topology;
   emit_property_instruction(emit, opcode0, 0, 0);

   /* emit max output vertices */
   opcode0.value = 0;
   opcode0.opcodeType = VGPU10_OPCODE_DCL_MAX_OUTPUT_VERTEX_COUNT;
   emit_property_instruction(emit, opcode0, 1, emit->gs.max_out_vertices);
}


/**
 * Emit a vgpu10 declaration "instruction".
 * \param index  the register index
 * \param size   array size of the operand. In most cases, it is 1,
 *               but for inputs to geometry shader, the array size varies
 *               depending on the primitive type.
 */
static void
emit_decl_instruction(struct svga_shader_emitter_v10 *emit,
                      VGPU10OpcodeToken0 opcode0,
                      VGPU10OperandToken0 operand0,
                      VGPU10NameToken name_token,
                      unsigned index, unsigned size)
{
   assert(opcode0.opcodeType);
   assert(operand0.mask ||
          (operand0.operandType == VGPU10_OPERAND_TYPE_OUTPUT_DEPTH) ||
          (operand0.operandType == VGPU10_OPERAND_TYPE_OUTPUT_COVERAGE_MASK));

   begin_emit_instruction(emit);
   emit_dword(emit, opcode0.value);

   emit_dword(emit, operand0.value);

   if (operand0.indexDimension == VGPU10_OPERAND_INDEX_1D) {
      /* Next token is the index of the register to declare */
      emit_dword(emit, index);
   }
   else if (operand0.indexDimension >= VGPU10_OPERAND_INDEX_2D) {
      /* Next token is the size of the register */
      emit_dword(emit, size);

      /* Followed by the index of the register */
      emit_dword(emit, index);
   }

   if (name_token.value) {
      emit_dword(emit, name_token.value);
   }

   end_emit_instruction(emit);
}


/**
 * Emit the declaration for a shader input.
 * \param opcodeType  opcode type, one of VGPU10_OPCODE_DCL_INPUTx
 * \param operandType operand type, one of VGPU10_OPERAND_TYPE_INPUT_x
 * \param dim         index dimension
 * \param index       the input register index
 * \param size        array size of the operand. In most cases, it is 1,
 *                    but for inputs to geometry shader, the array size varies
 *                    depending on the primitive type.
 * \param name        one of VGPU10_NAME_x
 * \parma numComp     number of components
 * \param selMode     component selection mode
 * \param usageMask   bitfield of VGPU10_OPERAND_4_COMPONENT_MASK_x values
 * \param interpMode  interpolation mode
 */
static void
emit_input_declaration(struct svga_shader_emitter_v10 *emit,
                       VGPU10_OPCODE_TYPE opcodeType,
                       VGPU10_OPERAND_TYPE operandType,
                       VGPU10_OPERAND_INDEX_DIMENSION dim,
                       unsigned index, unsigned size,
                       VGPU10_SYSTEM_NAME name,
                       VGPU10_OPERAND_NUM_COMPONENTS numComp,
                       VGPU10_OPERAND_4_COMPONENT_SELECTION_MODE selMode,
                       unsigned usageMask,
                       VGPU10_INTERPOLATION_MODE interpMode)
{
   VGPU10OpcodeToken0 opcode0;
   VGPU10OperandToken0 operand0;
   VGPU10NameToken name_token;

   assert(usageMask <= VGPU10_OPERAND_4_COMPONENT_MASK_ALL);
   assert(opcodeType == VGPU10_OPCODE_DCL_INPUT ||
          opcodeType == VGPU10_OPCODE_DCL_INPUT_SIV ||
          opcodeType == VGPU10_OPCODE_DCL_INPUT_PS ||
          opcodeType == VGPU10_OPCODE_DCL_INPUT_PS_SIV ||
          opcodeType == VGPU10_OPCODE_DCL_INPUT_PS_SGV);
   assert(operandType == VGPU10_OPERAND_TYPE_INPUT ||
          operandType == VGPU10_OPERAND_TYPE_INPUT_PRIMITIVEID);
   assert(numComp <= VGPU10_OPERAND_4_COMPONENT);
   assert(selMode <= VGPU10_OPERAND_4_COMPONENT_MASK_MODE);
   assert(dim <= VGPU10_OPERAND_INDEX_3D);
   assert(name == VGPU10_NAME_UNDEFINED ||
          name == VGPU10_NAME_POSITION ||
          name == VGPU10_NAME_INSTANCE_ID ||
          name == VGPU10_NAME_VERTEX_ID ||
          name == VGPU10_NAME_PRIMITIVE_ID ||
          name == VGPU10_NAME_IS_FRONT_FACE ||
          name == VGPU10_NAME_SAMPLE_INDEX);

   assert(interpMode == VGPU10_INTERPOLATION_UNDEFINED ||
          interpMode == VGPU10_INTERPOLATION_CONSTANT ||
          interpMode == VGPU10_INTERPOLATION_LINEAR ||
          interpMode == VGPU10_INTERPOLATION_LINEAR_CENTROID ||
          interpMode == VGPU10_INTERPOLATION_LINEAR_NOPERSPECTIVE ||
          interpMode == VGPU10_INTERPOLATION_LINEAR_NOPERSPECTIVE_CENTROID ||
          interpMode == VGPU10_INTERPOLATION_LINEAR_SAMPLE ||
          interpMode == VGPU10_INTERPOLATION_LINEAR_NOPERSPECTIVE_SAMPLE);

   check_register_index(emit, opcodeType, index);

   opcode0.value = operand0.value = name_token.value = 0;

   opcode0.opcodeType = opcodeType;
   opcode0.interpolationMode = interpMode;

   operand0.operandType = operandType;
   operand0.numComponents = numComp;
   operand0.selectionMode = selMode;
   operand0.mask = usageMask;
   operand0.indexDimension = dim;
   operand0.index0Representation = VGPU10_OPERAND_INDEX_IMMEDIATE32;
   if (dim == VGPU10_OPERAND_INDEX_2D)
      operand0.index1Representation = VGPU10_OPERAND_INDEX_IMMEDIATE32;

   name_token.name = name;

   emit_decl_instruction(emit, opcode0, operand0, name_token, index, size);
}


/**
 * Emit the declaration for a shader output.
 * \param type  one of VGPU10_OPCODE_DCL_OUTPUTx
 * \param index  the output register index
 * \param name  one of VGPU10_NAME_x
 * \param usageMask  bitfield of VGPU10_OPERAND_4_COMPONENT_MASK_x values
 */
static void
emit_output_declaration(struct svga_shader_emitter_v10 *emit,
                        VGPU10_OPCODE_TYPE type, unsigned index,
                        VGPU10_SYSTEM_NAME name,
                        unsigned usageMask)
{
   VGPU10OpcodeToken0 opcode0;
   VGPU10OperandToken0 operand0;
   VGPU10NameToken name_token;

   assert(usageMask <= VGPU10_OPERAND_4_COMPONENT_MASK_ALL);
   assert(type == VGPU10_OPCODE_DCL_OUTPUT ||
          type == VGPU10_OPCODE_DCL_OUTPUT_SGV ||
          type == VGPU10_OPCODE_DCL_OUTPUT_SIV);
   assert(name == VGPU10_NAME_UNDEFINED ||
          name == VGPU10_NAME_POSITION ||
          name == VGPU10_NAME_PRIMITIVE_ID ||
          name == VGPU10_NAME_RENDER_TARGET_ARRAY_INDEX ||
          name == VGPU10_NAME_CLIP_DISTANCE);

   check_register_index(emit, type, index);

   opcode0.value = operand0.value = name_token.value = 0;

   opcode0.opcodeType = type;
   operand0.operandType = VGPU10_OPERAND_TYPE_OUTPUT;
   operand0.numComponents = VGPU10_OPERAND_4_COMPONENT;
   operand0.selectionMode = VGPU10_OPERAND_4_COMPONENT_MASK_MODE;
   operand0.mask = usageMask;
   operand0.indexDimension = VGPU10_OPERAND_INDEX_1D;
   operand0.index0Representation = VGPU10_OPERAND_INDEX_IMMEDIATE32;

   name_token.name = name;

   emit_decl_instruction(emit, opcode0, operand0, name_token, index, 1);
}


/**
 * Emit the declaration for the fragment depth output.
 */
static void
emit_fragdepth_output_declaration(struct svga_shader_emitter_v10 *emit)
{
   VGPU10OpcodeToken0 opcode0;
   VGPU10OperandToken0 operand0;
   VGPU10NameToken name_token;

   assert(emit->unit == PIPE_SHADER_FRAGMENT);

   opcode0.value = operand0.value = name_token.value = 0;

   opcode0.opcodeType = VGPU10_OPCODE_DCL_OUTPUT;
   operand0.operandType = VGPU10_OPERAND_TYPE_OUTPUT_DEPTH;
   operand0.numComponents = VGPU10_OPERAND_1_COMPONENT;
   operand0.indexDimension = VGPU10_OPERAND_INDEX_0D;
   operand0.mask = 0;

   emit_decl_instruction(emit, opcode0, operand0, name_token, 0, 1);
}


/**
 * Emit the declaration for the fragment sample mask/coverage output.
 */
static void
emit_samplemask_output_declaration(struct svga_shader_emitter_v10 *emit)
{
   VGPU10OpcodeToken0 opcode0;
   VGPU10OperandToken0 operand0;
   VGPU10NameToken name_token;

   assert(emit->unit == PIPE_SHADER_FRAGMENT);
   assert(emit->version >= 41);

   opcode0.value = operand0.value = name_token.value = 0;

   opcode0.opcodeType = VGPU10_OPCODE_DCL_OUTPUT;
   operand0.operandType = VGPU10_OPERAND_TYPE_OUTPUT_COVERAGE_MASK;
   operand0.numComponents = VGPU10_OPERAND_0_COMPONENT;
   operand0.indexDimension = VGPU10_OPERAND_INDEX_0D;
   operand0.mask = 0;

   emit_decl_instruction(emit, opcode0, operand0, name_token, 0, 1);
}


/**
 * Emit the declaration for a system value input/output.
 */
static void
emit_system_value_declaration(struct svga_shader_emitter_v10 *emit,
                              enum tgsi_semantic semantic_name, unsigned index)
{
   switch (semantic_name) {
   case TGSI_SEMANTIC_INSTANCEID:
      index = alloc_system_value_index(emit, index);
      emit_input_declaration(emit, VGPU10_OPCODE_DCL_INPUT_SIV,
                             VGPU10_OPERAND_TYPE_INPUT,
                             VGPU10_OPERAND_INDEX_1D,
                             index, 1,
                             VGPU10_NAME_INSTANCE_ID,
                             VGPU10_OPERAND_4_COMPONENT,
                             VGPU10_OPERAND_4_COMPONENT_MASK_MODE,
                             VGPU10_OPERAND_4_COMPONENT_MASK_X,
                             VGPU10_INTERPOLATION_UNDEFINED);
      break;
   case TGSI_SEMANTIC_VERTEXID:
      index = alloc_system_value_index(emit, index);
      emit_input_declaration(emit, VGPU10_OPCODE_DCL_INPUT_SIV,
                             VGPU10_OPERAND_TYPE_INPUT,
                             VGPU10_OPERAND_INDEX_1D,
                             index, 1,
                             VGPU10_NAME_VERTEX_ID,
                             VGPU10_OPERAND_4_COMPONENT,
                             VGPU10_OPERAND_4_COMPONENT_MASK_MODE,
                             VGPU10_OPERAND_4_COMPONENT_MASK_X,
                             VGPU10_INTERPOLATION_UNDEFINED);
      break;
   case TGSI_SEMANTIC_SAMPLEID:
      assert(emit->unit == PIPE_SHADER_FRAGMENT);
      emit->fs.sample_id_sys_index = index;
      index = alloc_system_value_index(emit, index);
      emit_input_declaration(emit, VGPU10_OPCODE_DCL_INPUT_PS_SIV,
                             VGPU10_OPERAND_TYPE_INPUT,
                             VGPU10_OPERAND_INDEX_1D,
                             index, 1,
                             VGPU10_NAME_SAMPLE_INDEX,
                             VGPU10_OPERAND_4_COMPONENT,
                             VGPU10_OPERAND_4_COMPONENT_MASK_MODE,
                             VGPU10_OPERAND_4_COMPONENT_MASK_X,
                             VGPU10_INTERPOLATION_CONSTANT);
      break;
   case TGSI_SEMANTIC_SAMPLEPOS:
      /* This system value contains the position of the current sample
       * when using per-sample shading.  We implement this by calling
       * the VGPU10_OPCODE_SAMPLE_POS instruction with the current sample
       * index as the argument.  See emit_sample_position_instructions().
       */
      assert(emit->version >= 41);
      emit->fs.sample_pos_sys_index = index;
      index = alloc_system_value_index(emit, index);
      break;
   default:
      debug_printf("unexpected sytem value semantic index %u\n",
         semantic_name);
   }
}

/**
 * Translate a TGSI declaration to VGPU10.
 */
static boolean
emit_vgpu10_declaration(struct svga_shader_emitter_v10 *emit,
                        const struct tgsi_full_declaration *decl)
{
   switch (decl->Declaration.File) {
   case TGSI_FILE_INPUT:
      /* do nothing - see emit_input_declarations() */
      return TRUE;

   case TGSI_FILE_OUTPUT:
      assert(decl->Range.First == decl->Range.Last);
      emit->output_usage_mask[decl->Range.First] = decl->Declaration.UsageMask;
      return TRUE;

   case TGSI_FILE_TEMPORARY:
      /* Don't declare the temps here.  Just keep track of how many
       * and emit the declaration later.
       */
      if (decl->Declaration.Array) {
         /* Indexed temporary array.  Save the start index of the array
          * and the size of the array.
          */
         const unsigned arrayID = MIN2(decl->Array.ArrayID, MAX_TEMP_ARRAYS);
         unsigned i;

         assert(arrayID < ARRAY_SIZE(emit->temp_arrays));

         /* Save this array so we can emit the declaration for it later */
         emit->temp_arrays[arrayID].start = decl->Range.First;
         emit->temp_arrays[arrayID].size =
            decl->Range.Last - decl->Range.First + 1;

         emit->num_temp_arrays = MAX2(emit->num_temp_arrays, arrayID + 1);
         assert(emit->num_temp_arrays <= MAX_TEMP_ARRAYS);
         emit->num_temp_arrays = MIN2(emit->num_temp_arrays, MAX_TEMP_ARRAYS);

         /* Fill in the temp_map entries for this array */
         for (i = decl->Range.First; i <= decl->Range.Last; i++) {
            emit->temp_map[i].arrayId = arrayID;
            emit->temp_map[i].index = i - decl->Range.First;
         }
      }

      /* for all temps, indexed or not, keep track of highest index */
      emit->num_shader_temps = MAX2(emit->num_shader_temps,
                                    decl->Range.Last + 1);
      return TRUE;

   case TGSI_FILE_CONSTANT:
      /* Don't declare constants here.  Just keep track and emit later. */
      {
         unsigned constbuf = 0, num_consts;
         if (decl->Declaration.Dimension) {
            constbuf = decl->Dim.Index2D;
         }
         /* We throw an assertion here when, in fact, the shader should never
          * have linked due to constbuf index out of bounds, so we shouldn't
          * have reached here.
          */
         assert(constbuf < ARRAY_SIZE(emit->num_shader_consts));

         num_consts = MAX2(emit->num_shader_consts[constbuf],
                           decl->Range.Last + 1);

         if (num_consts > VGPU10_MAX_CONSTANT_BUFFER_ELEMENT_COUNT) {
            debug_printf("Warning: constant buffer is declared to size [%u]"
                         " but [%u] is the limit.\n",
                         num_consts,
                         VGPU10_MAX_CONSTANT_BUFFER_ELEMENT_COUNT);
         }
         /* The linker doesn't enforce the max UBO size so we clamp here */
         emit->num_shader_consts[constbuf] =
            MIN2(num_consts, VGPU10_MAX_CONSTANT_BUFFER_ELEMENT_COUNT);
      }
      return TRUE;

   case TGSI_FILE_IMMEDIATE:
      assert(!"TGSI_FILE_IMMEDIATE not handled yet!");
      return FALSE;

   case TGSI_FILE_SYSTEM_VALUE:
      emit_system_value_declaration(emit, decl->Semantic.Name,
                                    decl->Range.First);
      return TRUE;

   case TGSI_FILE_SAMPLER:
      /* Don't declare samplers here.  Just keep track and emit later. */
      emit->num_samplers = MAX2(emit->num_samplers, decl->Range.Last + 1);
      return TRUE;

#if 0
   case TGSI_FILE_RESOURCE:
      /*opcode0.opcodeType = VGPU10_OPCODE_DCL_RESOURCE;*/
      /* XXX more, VGPU10_RETURN_TYPE_FLOAT */
      assert(!"TGSI_FILE_RESOURCE not handled yet");
      return FALSE;
#endif

   case TGSI_FILE_ADDRESS:
      emit->num_address_regs = MAX2(emit->num_address_regs,
                                    decl->Range.Last + 1);
      return TRUE;

   case TGSI_FILE_SAMPLER_VIEW:
      {
         unsigned unit = decl->Range.First;
         assert(decl->Range.First == decl->Range.Last);
         emit->sampler_target[unit] = decl->SamplerView.Resource;
         /* Note: we can ignore YZW return types for now */
         emit->sampler_return_type[unit] = decl->SamplerView.ReturnTypeX;
         emit->sampler_view[unit] = TRUE;
      }
      return TRUE;

   default:
      assert(!"Unexpected type of declaration");
      return FALSE;
   }
}



/**
 * Emit all input declarations.
 */
static boolean
emit_input_declarations(struct svga_shader_emitter_v10 *emit)
{
   unsigned i;

   if (emit->unit == PIPE_SHADER_FRAGMENT) {

      for (i = 0; i < emit->linkage.num_inputs; i++) {
         enum tgsi_semantic semantic_name = emit->info.input_semantic_name[i];
         unsigned usage_mask = emit->info.input_usage_mask[i];
         unsigned index = emit->linkage.input_map[i];
         VGPU10_OPCODE_TYPE type;
         VGPU10_INTERPOLATION_MODE interpolationMode;
         VGPU10_SYSTEM_NAME name;

         if (usage_mask == 0)
            continue;  /* register is not actually used */

         if (semantic_name == TGSI_SEMANTIC_POSITION) {
            /* fragment position input */
            type = VGPU10_OPCODE_DCL_INPUT_PS_SGV;
            interpolationMode = VGPU10_INTERPOLATION_LINEAR;
            name = VGPU10_NAME_POSITION;
            if (usage_mask & TGSI_WRITEMASK_W) {
               /* we need to replace use of 'w' with '1/w' */
               emit->fs.fragcoord_input_index = i;
            }
         }
         else if (semantic_name == TGSI_SEMANTIC_FACE) {
            /* fragment front-facing input */
            type = VGPU10_OPCODE_DCL_INPUT_PS_SGV;
            interpolationMode = VGPU10_INTERPOLATION_CONSTANT;
            name = VGPU10_NAME_IS_FRONT_FACE;
            emit->fs.face_input_index = i;
         }
         else if (semantic_name == TGSI_SEMANTIC_PRIMID) {
            /* primitive ID */
            type = VGPU10_OPCODE_DCL_INPUT_PS_SGV;
            interpolationMode = VGPU10_INTERPOLATION_CONSTANT;
            name = VGPU10_NAME_PRIMITIVE_ID;
         }
         else if (semantic_name == TGSI_SEMANTIC_SAMPLEID) {
            /* sample index / ID */
            type = VGPU10_OPCODE_DCL_INPUT_PS_SGV;
            interpolationMode = VGPU10_INTERPOLATION_CONSTANT;
            name = VGPU10_NAME_SAMPLE_INDEX;
         }
         else {
            /* general fragment input */
            type = VGPU10_OPCODE_DCL_INPUT_PS;
            interpolationMode =
               translate_interpolation(emit,
                                       emit->info.input_interpolate[i],
                                       emit->info.input_interpolate_loc[i]);

            /* keeps track if flat interpolation mode is being used */
            emit->uses_flat_interp |=
               (interpolationMode == VGPU10_INTERPOLATION_CONSTANT);

            name = VGPU10_NAME_UNDEFINED;
         }

         emit_input_declaration(emit, type,
                                VGPU10_OPERAND_TYPE_INPUT,
                                VGPU10_OPERAND_INDEX_1D, index, 1,
                                name,
                                VGPU10_OPERAND_4_COMPONENT,
                                VGPU10_OPERAND_4_COMPONENT_MASK_MODE,
                                VGPU10_OPERAND_4_COMPONENT_MASK_ALL,
                                interpolationMode);
      }
   }
   else if (emit->unit == PIPE_SHADER_GEOMETRY) {

      for (i = 0; i < emit->info.num_inputs; i++) {
         enum tgsi_semantic semantic_name = emit->info.input_semantic_name[i];
         unsigned usage_mask = emit->info.input_usage_mask[i];
         unsigned index = emit->linkage.input_map[i];
         VGPU10_OPCODE_TYPE opcodeType, operandType;
         VGPU10_OPERAND_NUM_COMPONENTS numComp;
         VGPU10_OPERAND_4_COMPONENT_SELECTION_MODE selMode;
         VGPU10_SYSTEM_NAME name;
         VGPU10_OPERAND_INDEX_DIMENSION dim;

         if (usage_mask == 0)
            continue;  /* register is not actually used */

         opcodeType = VGPU10_OPCODE_DCL_INPUT;
         operandType = VGPU10_OPERAND_TYPE_INPUT;
         numComp = VGPU10_OPERAND_4_COMPONENT;
         selMode = VGPU10_OPERAND_4_COMPONENT_MASK_MODE;
         name = VGPU10_NAME_UNDEFINED;

         /* all geometry shader inputs are two dimensional except
          * gl_PrimitiveID
          */
         dim = VGPU10_OPERAND_INDEX_2D;

         if (semantic_name == TGSI_SEMANTIC_PRIMID) {
            /* Primitive ID */
            operandType = VGPU10_OPERAND_TYPE_INPUT_PRIMITIVEID;
            dim = VGPU10_OPERAND_INDEX_0D;
            numComp = VGPU10_OPERAND_0_COMPONENT;
            selMode = 0;

            /* also save the register index so we can check for
             * primitive id when emit src register. We need to modify the
             * operand type, index dimension when emit primitive id src reg.
             */
            emit->gs.prim_id_index = i;
         }
         else if (semantic_name == TGSI_SEMANTIC_POSITION) {
            /* vertex position input */
            opcodeType = VGPU10_OPCODE_DCL_INPUT_SIV;
            name = VGPU10_NAME_POSITION;
         }

         emit_input_declaration(emit, opcodeType, operandType,
                                dim, index,
                                emit->gs.input_size,
                                name,
                                numComp, selMode,
                                VGPU10_OPERAND_4_COMPONENT_MASK_ALL,
                                VGPU10_INTERPOLATION_UNDEFINED);
      }
   }
   else {
      assert(emit->unit == PIPE_SHADER_VERTEX);

      for (i = 0; i < emit->info.file_max[TGSI_FILE_INPUT] + 1; i++) {
         unsigned usage_mask = emit->info.input_usage_mask[i];
         unsigned index = i;

         if (usage_mask == 0)
            continue;  /* register is not actually used */

         emit_input_declaration(emit, VGPU10_OPCODE_DCL_INPUT,
                                VGPU10_OPERAND_TYPE_INPUT,
                                VGPU10_OPERAND_INDEX_1D, index, 1,
                                VGPU10_NAME_UNDEFINED,
                                VGPU10_OPERAND_4_COMPONENT,
                                VGPU10_OPERAND_4_COMPONENT_MASK_MODE,
                                VGPU10_OPERAND_4_COMPONENT_MASK_ALL,
                                VGPU10_INTERPOLATION_UNDEFINED);
      }
   }

   return TRUE;
}


/**
 * Emit all output declarations.
 */
static boolean
emit_output_declarations(struct svga_shader_emitter_v10 *emit)
{
   unsigned i;

   for (i = 0; i < emit->info.num_outputs; i++) {
      /*const unsigned usage_mask = emit->info.output_usage_mask[i];*/
      const enum tgsi_semantic semantic_name =
         emit->info.output_semantic_name[i];
      const unsigned semantic_index = emit->info.output_semantic_index[i];
      unsigned index = i;

      if (emit->unit == PIPE_SHADER_FRAGMENT) {
         if (semantic_name == TGSI_SEMANTIC_COLOR) {
            assert(semantic_index < ARRAY_SIZE(emit->fs.color_out_index));

            emit->fs.color_out_index[semantic_index] = index;

            emit->fs.num_color_outputs = MAX2(emit->fs.num_color_outputs,
                                              index + 1);

            /* The semantic index is the shader's color output/buffer index */
            emit_output_declaration(emit,
                                    VGPU10_OPCODE_DCL_OUTPUT, semantic_index,
                                    VGPU10_NAME_UNDEFINED,
                                    VGPU10_OPERAND_4_COMPONENT_MASK_ALL);

            if (semantic_index == 0) {
               if (emit->key.fs.write_color0_to_n_cbufs > 1) {
                  /* Emit declarations for the additional color outputs
                   * for broadcasting.
                   */
                  unsigned j;
                  for (j = 1; j < emit->key.fs.write_color0_to_n_cbufs; j++) {
                     /* Allocate a new output index */
                     unsigned idx = emit->info.num_outputs + j - 1;
                     emit->fs.color_out_index[j] = idx;
                     emit_output_declaration(emit,
                                        VGPU10_OPCODE_DCL_OUTPUT, idx,
                                        VGPU10_NAME_UNDEFINED,
                                        VGPU10_OPERAND_4_COMPONENT_MASK_ALL);
                     emit->info.output_semantic_index[idx] = j;
                  }

                  emit->fs.num_color_outputs =
                     emit->key.fs.write_color0_to_n_cbufs;
               }
            }
            else {
               assert(!emit->key.fs.write_color0_to_n_cbufs);
            }
         }
         else if (semantic_name == TGSI_SEMANTIC_POSITION) {
            /* Fragment depth output */
            emit_fragdepth_output_declaration(emit);
         }
         else if (semantic_name == TGSI_SEMANTIC_SAMPLEMASK) {
            /* Fragment depth output */
            emit_samplemask_output_declaration(emit);
         }
         else {
            assert(!"Bad output semantic name");
         }
      }
      else {
         /* VS or GS */
         VGPU10_COMPONENT_NAME name;
         VGPU10_OPCODE_TYPE type;
         unsigned writemask = VGPU10_OPERAND_4_COMPONENT_MASK_ALL;

         switch (semantic_name) {
         case TGSI_SEMANTIC_POSITION:
            assert(emit->unit != PIPE_SHADER_FRAGMENT);
            type = VGPU10_OPCODE_DCL_OUTPUT_SIV;
            name = VGPU10_NAME_POSITION;
            /* Save the index of the vertex position output register */
            emit->vposition.out_index = index;
            break;
         case TGSI_SEMANTIC_CLIPDIST:
            type = VGPU10_OPCODE_DCL_OUTPUT_SIV;
            name = VGPU10_NAME_CLIP_DISTANCE;
            /* save the starting index of the clip distance output register */
            if (semantic_index == 0)
               emit->clip_dist_out_index = index;
            writemask = emit->output_usage_mask[index];
            writemask = apply_clip_plane_mask(emit, writemask, semantic_index);
            if (writemask == 0x0) {
               continue; /* discard this do-nothing declaration */
            }
            break;
         case TGSI_SEMANTIC_PRIMID:
            assert(emit->unit == PIPE_SHADER_GEOMETRY);
            type = VGPU10_OPCODE_DCL_OUTPUT_SGV;
            name = VGPU10_NAME_PRIMITIVE_ID;
            break;
         case TGSI_SEMANTIC_LAYER:
            assert(emit->unit == PIPE_SHADER_GEOMETRY);
            type = VGPU10_OPCODE_DCL_OUTPUT_SGV;
            name = VGPU10_NAME_RENDER_TARGET_ARRAY_INDEX;
            break;
         case TGSI_SEMANTIC_CLIPVERTEX:
            type = VGPU10_OPCODE_DCL_OUTPUT;
            name = VGPU10_NAME_UNDEFINED;
            emit->clip_vertex_out_index = index;
            break;
         default:
            /* generic output */
            type = VGPU10_OPCODE_DCL_OUTPUT;
            name = VGPU10_NAME_UNDEFINED;
         }

         emit_output_declaration(emit, type, index, name, writemask);
      }
   }

   if (emit->vposition.so_index != INVALID_INDEX &&
       emit->vposition.out_index != INVALID_INDEX) {

      assert(emit->unit != PIPE_SHADER_FRAGMENT);

      /* Emit the declaration for the non-adjusted vertex position
       * for stream output purpose
       */
      emit_output_declaration(emit, VGPU10_OPCODE_DCL_OUTPUT,
                              emit->vposition.so_index,
                              VGPU10_NAME_UNDEFINED,
                              VGPU10_OPERAND_4_COMPONENT_MASK_ALL);
   }

   if (emit->clip_dist_so_index != INVALID_INDEX &&
       emit->clip_dist_out_index != INVALID_INDEX) {

      assert(emit->unit != PIPE_SHADER_FRAGMENT);

      /* Emit the declaration for the clip distance shadow copy which
       * will be used for stream output purpose and for clip distance
       * varying variable
       */
      emit_output_declaration(emit, VGPU10_OPCODE_DCL_OUTPUT,
                              emit->clip_dist_so_index,
                              VGPU10_NAME_UNDEFINED,
                              emit->output_usage_mask[emit->clip_dist_out_index]);

      if (emit->info.num_written_clipdistance > 4) {
         /* for the second clip distance register, each handles 4 planes */
         emit_output_declaration(emit, VGPU10_OPCODE_DCL_OUTPUT,
                                 emit->clip_dist_so_index + 1,
                                 VGPU10_NAME_UNDEFINED,
                                 emit->output_usage_mask[emit->clip_dist_out_index+1]);
      }
   }

   return TRUE;
}


/**
 * Emit the declaration for the temporary registers.
 */
static boolean
emit_temporaries_declaration(struct svga_shader_emitter_v10 *emit)
{
   unsigned total_temps, reg, i;

   total_temps = emit->num_shader_temps;

   /* If there is indirect access to non-indexable temps in the shader,
    * convert those temps to indexable temps. This works around a bug
    * in the GLSL->TGSI translator exposed in piglit test
    * glsl-1.20/execution/fs-const-array-of-struct-of-array.shader_test.
    * Internal temps added by the driver remain as non-indexable temps.
    */
   if ((emit->info.indirect_files & (1 << TGSI_FILE_TEMPORARY)) &&
       emit->num_temp_arrays == 0) {
      unsigned arrayID;

      arrayID = 1;
      emit->num_temp_arrays = arrayID + 1; 
      emit->temp_arrays[arrayID].start = 0;
      emit->temp_arrays[arrayID].size = total_temps;

      /* Fill in the temp_map entries for this temp array */
      for (i = 0; i < total_temps; i++) {
         emit->temp_map[i].arrayId = arrayID;
         emit->temp_map[i].index = i;
      }
   }

   /* Allocate extra temps for specially-implemented instructions,
    * such as LIT.
    */
   total_temps += MAX_INTERNAL_TEMPS;

   if (emit->unit == PIPE_SHADER_VERTEX || emit->unit == PIPE_SHADER_GEOMETRY) {
      if (emit->vposition.need_prescale || emit->key.vs.undo_viewport ||
          emit->key.clip_plane_enable ||
          emit->vposition.so_index != INVALID_INDEX) {
         emit->vposition.tmp_index = total_temps;
         total_temps += 1;
      }

      if (emit->unit == PIPE_SHADER_VERTEX) {
         unsigned attrib_mask = (emit->key.vs.adjust_attrib_w_1 |
                                 emit->key.vs.adjust_attrib_itof |
                                 emit->key.vs.adjust_attrib_utof |
                                 emit->key.vs.attrib_is_bgra |
                                 emit->key.vs.attrib_puint_to_snorm |
                                 emit->key.vs.attrib_puint_to_uscaled |
                                 emit->key.vs.attrib_puint_to_sscaled);
         while (attrib_mask) {
            unsigned index = u_bit_scan(&attrib_mask);
            emit->vs.adjusted_input[index] = total_temps++;
         }
      }

      if (emit->clip_mode == CLIP_DISTANCE) {
         /* We need to write the clip distance to a temporary register
          * first. Then it will be copied to the shadow copy for
          * the clip distance varying variable and stream output purpose.
          * It will also be copied to the actual CLIPDIST register
          * according to the enabled clip planes
          */
         emit->clip_dist_tmp_index = total_temps++;
         if (emit->info.num_written_clipdistance > 4)
            total_temps++; /* second clip register */
      }
      else if (emit->clip_mode == CLIP_VERTEX) {
         /* We need to convert the TGSI CLIPVERTEX output to one or more
          * clip distances.  Allocate a temp reg for the clipvertex here.
          */
         assert(emit->info.writes_clipvertex > 0);
         emit->clip_vertex_tmp_index = total_temps;
         total_temps++;
      }
   }
   else if (emit->unit == PIPE_SHADER_FRAGMENT) {
      if (emit->key.fs.alpha_func != SVGA3D_CMP_ALWAYS ||
          emit->key.fs.write_color0_to_n_cbufs > 1) {
         /* Allocate a temp to hold the output color */
         emit->fs.color_tmp_index = total_temps;
         total_temps += 1;
      }

      if (emit->fs.face_input_index != INVALID_INDEX) {
         /* Allocate a temp for the +/-1 face register */
         emit->fs.face_tmp_index = total_temps;
         total_temps += 1;
      }

      if (emit->fs.fragcoord_input_index != INVALID_INDEX) {
         /* Allocate a temp for modified fragment position register */
         emit->fs.fragcoord_tmp_index = total_temps;
         total_temps += 1;
      }

      if (emit->fs.sample_pos_sys_index != INVALID_INDEX) {
         /* Allocate a temp for the sample position */
         emit->fs.sample_pos_tmp_index = total_temps++;
      }
   }

   for (i = 0; i < emit->num_address_regs; i++) {
      emit->address_reg_index[i] = total_temps++;
   }

   /* Initialize the temp_map array which maps TGSI temp indexes to VGPU10
    * temp indexes.  Basically, we compact all the non-array temp register
    * indexes into a consecutive series.
    *
    * Before, we may have some TGSI declarations like:
    *   DCL TEMP[0..1], LOCAL
    *   DCL TEMP[2..4], ARRAY(1), LOCAL
    *   DCL TEMP[5..7], ARRAY(2), LOCAL
    *   plus, some extra temps, like TEMP[8], TEMP[9] for misc things
    *
    * After, we'll have a map like this:
    *   temp_map[0] = { array 0, index 0 }
    *   temp_map[1] = { array 0, index 1 }
    *   temp_map[2] = { array 1, index 0 }
    *   temp_map[3] = { array 1, index 1 }
    *   temp_map[4] = { array 1, index 2 }
    *   temp_map[5] = { array 2, index 0 }
    *   temp_map[6] = { array 2, index 1 }
    *   temp_map[7] = { array 2, index 2 }
    *   temp_map[8] = { array 0, index 2 }
    *   temp_map[9] = { array 0, index 3 }
    *
    * We'll declare two arrays of 3 elements, plus a set of four non-indexed
    * temps numbered 0..3
    *
    * Any time we emit a temporary register index, we'll have to use the
    * temp_map[] table to convert the TGSI index to the VGPU10 index.
    *
    * Finally, we recompute the total_temps value here.
    */
   reg = 0;
   for (i = 0; i < total_temps; i++) {
      if (emit->temp_map[i].arrayId == 0) {
         emit->temp_map[i].index = reg++;
      }
   }

   if (0) {
      debug_printf("total_temps %u\n", total_temps);
      for (i = 0; i < total_temps; i++) {
         debug_printf("temp %u ->  array %u  index %u\n",
                      i, emit->temp_map[i].arrayId, emit->temp_map[i].index);
      }
   }

   total_temps = reg;

   /* Emit declaration of ordinary temp registers */
   if (total_temps > 0) {
      VGPU10OpcodeToken0 opcode0;

      opcode0.value = 0;
      opcode0.opcodeType = VGPU10_OPCODE_DCL_TEMPS;

      begin_emit_instruction(emit);
      emit_dword(emit, opcode0.value);
      emit_dword(emit, total_temps);
      end_emit_instruction(emit);
   }

   /* Emit declarations for indexable temp arrays.  Skip 0th entry since
    * it's unused.
    */
   for (i = 1; i < emit->num_temp_arrays; i++) {
      unsigned num_temps = emit->temp_arrays[i].size;

      if (num_temps > 0) {
         VGPU10OpcodeToken0 opcode0;

         opcode0.value = 0;
         opcode0.opcodeType = VGPU10_OPCODE_DCL_INDEXABLE_TEMP;

         begin_emit_instruction(emit);
         emit_dword(emit, opcode0.value);
         emit_dword(emit, i); /* which array */
         emit_dword(emit, num_temps);
         emit_dword(emit, 4); /* num components */
         end_emit_instruction(emit);

         total_temps += num_temps;
      }
   }

   /* Check that the grand total of all regular and indexed temps is
    * under the limit.
    */
   check_register_index(emit, VGPU10_OPCODE_DCL_TEMPS, total_temps - 1);

   return TRUE;
}


static boolean
emit_constant_declaration(struct svga_shader_emitter_v10 *emit)
{
   VGPU10OpcodeToken0 opcode0;
   VGPU10OperandToken0 operand0;
   unsigned total_consts, i;

   opcode0.value = 0;
   opcode0.opcodeType = VGPU10_OPCODE_DCL_CONSTANT_BUFFER;
   opcode0.accessPattern = VGPU10_CB_IMMEDIATE_INDEXED;
   /* XXX or, access pattern = VGPU10_CB_DYNAMIC_INDEXED */

   operand0.value = 0;
   operand0.numComponents = VGPU10_OPERAND_4_COMPONENT;
   operand0.indexDimension = VGPU10_OPERAND_INDEX_2D;
   operand0.index0Representation = VGPU10_OPERAND_INDEX_IMMEDIATE32;
   operand0.index1Representation = VGPU10_OPERAND_INDEX_IMMEDIATE32;
   operand0.operandType = VGPU10_OPERAND_TYPE_CONSTANT_BUFFER;
   operand0.selectionMode = VGPU10_OPERAND_4_COMPONENT_SWIZZLE_MODE;
   operand0.swizzleX = 0;
   operand0.swizzleY = 1;
   operand0.swizzleZ = 2;
   operand0.swizzleW = 3;

   /**
    * Emit declaration for constant buffer [0].  We also allocate
    * room for the extra constants here.
    */
   total_consts = emit->num_shader_consts[0];

   /* Now, allocate constant slots for the "extra" constants.
    * Note: it's critical that these extra constant locations
    * exactly match what's emitted by the "extra" constants code
    * in svga_state_constants.c
    */

   /* Vertex position scale/translation */
   if (emit->vposition.need_prescale) {
      emit->vposition.prescale_scale_index = total_consts++;
      emit->vposition.prescale_trans_index = total_consts++;
   }

   if (emit->unit == PIPE_SHADER_VERTEX) {
      if (emit->key.vs.undo_viewport) {
         emit->vs.viewport_index = total_consts++;
      }
   }

   /* user-defined clip planes */
   if (emit->key.clip_plane_enable) {
      unsigned n = util_bitcount(emit->key.clip_plane_enable);
      assert(emit->unit == PIPE_SHADER_VERTEX ||
             emit->unit == PIPE_SHADER_GEOMETRY);
      for (i = 0; i < n; i++) {
         emit->clip_plane_const[i] = total_consts++;
      }
   }

   for (i = 0; i < emit->num_samplers; i++) {

      if (emit->sampler_view[i]) {

         /* Texcoord scale factors for RECT textures */
         if (emit->key.tex[i].unnormalized) {
            emit->texcoord_scale_index[i] = total_consts++;
         }

         /* Texture buffer sizes */
         if (emit->sampler_target[i] == TGSI_TEXTURE_BUFFER) {
            emit->texture_buffer_size_index[i] = total_consts++;
         }
      }
   }

   if (total_consts > 0) {
      begin_emit_instruction(emit);
      emit_dword(emit, opcode0.value);
      emit_dword(emit, operand0.value);
      emit_dword(emit, 0);  /* which const buffer slot */
      emit_dword(emit, total_consts);
      end_emit_instruction(emit);
   }

   /* Declare remaining constant buffers (UBOs) */
   for (i = 1; i < ARRAY_SIZE(emit->num_shader_consts); i++) {
      if (emit->num_shader_consts[i] > 0) {
         begin_emit_instruction(emit);
         emit_dword(emit, opcode0.value);
         emit_dword(emit, operand0.value);
         emit_dword(emit, i);  /* which const buffer slot */
         emit_dword(emit, emit->num_shader_consts[i]);
         end_emit_instruction(emit);
      }
   }

   return TRUE;
}


/**
 * Emit declarations for samplers.
 */
static boolean
emit_sampler_declarations(struct svga_shader_emitter_v10 *emit)
{
   unsigned i;

   for (i = 0; i < emit->num_samplers; i++) {
      VGPU10OpcodeToken0 opcode0;
      VGPU10OperandToken0 operand0;

      opcode0.value = 0;
      opcode0.opcodeType = VGPU10_OPCODE_DCL_SAMPLER;
      opcode0.samplerMode = VGPU10_SAMPLER_MODE_DEFAULT;

      operand0.value = 0;
      operand0.numComponents = VGPU10_OPERAND_0_COMPONENT;
      operand0.operandType = VGPU10_OPERAND_TYPE_SAMPLER;
      operand0.indexDimension = VGPU10_OPERAND_INDEX_1D;
      operand0.index0Representation = VGPU10_OPERAND_INDEX_IMMEDIATE32;

      begin_emit_instruction(emit);
      emit_dword(emit, opcode0.value);
      emit_dword(emit, operand0.value);
      emit_dword(emit, i);
      end_emit_instruction(emit);
   }

   return TRUE;
}


/**
 * Translate TGSI_TEXTURE_x to VGPU10_RESOURCE_DIMENSION_x.
 */
static unsigned
tgsi_texture_to_resource_dimension(enum tgsi_texture_type target,
                                   unsigned num_samples,
                                   boolean is_array)
{
   if (target == TGSI_TEXTURE_2D_MSAA && num_samples < 2) {
      target = TGSI_TEXTURE_2D;
   }
   else if (target == TGSI_TEXTURE_2D_ARRAY_MSAA && num_samples < 2) {
      target = TGSI_TEXTURE_2D_ARRAY;
   }

   switch (target) {
   case TGSI_TEXTURE_BUFFER:
      return VGPU10_RESOURCE_DIMENSION_BUFFER;
   case TGSI_TEXTURE_1D:
      return VGPU10_RESOURCE_DIMENSION_TEXTURE1D;
   case TGSI_TEXTURE_2D:
   case TGSI_TEXTURE_RECT:
      return VGPU10_RESOURCE_DIMENSION_TEXTURE2D;
   case TGSI_TEXTURE_3D:
      return VGPU10_RESOURCE_DIMENSION_TEXTURE3D;
   case TGSI_TEXTURE_CUBE:
   case TGSI_TEXTURE_SHADOWCUBE:
      return VGPU10_RESOURCE_DIMENSION_TEXTURECUBE;
   case TGSI_TEXTURE_SHADOW1D:
      return VGPU10_RESOURCE_DIMENSION_TEXTURE1D;
   case TGSI_TEXTURE_SHADOW2D:
   case TGSI_TEXTURE_SHADOWRECT:
      return VGPU10_RESOURCE_DIMENSION_TEXTURE2D;
   case TGSI_TEXTURE_1D_ARRAY:
   case TGSI_TEXTURE_SHADOW1D_ARRAY:
      return is_array ? VGPU10_RESOURCE_DIMENSION_TEXTURE1DARRAY
         : VGPU10_RESOURCE_DIMENSION_TEXTURE1D;
   case TGSI_TEXTURE_2D_ARRAY:
   case TGSI_TEXTURE_SHADOW2D_ARRAY:
      return is_array ? VGPU10_RESOURCE_DIMENSION_TEXTURE2DARRAY
         : VGPU10_RESOURCE_DIMENSION_TEXTURE2D;
   case TGSI_TEXTURE_2D_MSAA:
      return VGPU10_RESOURCE_DIMENSION_TEXTURE2DMS;
   case TGSI_TEXTURE_2D_ARRAY_MSAA:
      return is_array ? VGPU10_RESOURCE_DIMENSION_TEXTURE2DMSARRAY
         : VGPU10_RESOURCE_DIMENSION_TEXTURE2DMS;
   case TGSI_TEXTURE_CUBE_ARRAY:
   case TGSI_TEXTURE_SHADOWCUBE_ARRAY:
      return is_array ? VGPU10_RESOURCE_DIMENSION_TEXTURECUBEARRAY
         : VGPU10_RESOURCE_DIMENSION_TEXTURECUBE;
   default:
      assert(!"Unexpected resource type");
      return VGPU10_RESOURCE_DIMENSION_TEXTURE2D;
   }
}


/**
 * Given a tgsi_return_type, return true iff it is an integer type.
 */
static boolean
is_integer_type(enum tgsi_return_type type)
{
   switch (type) {
      case TGSI_RETURN_TYPE_SINT:
      case TGSI_RETURN_TYPE_UINT:
         return TRUE;
      case TGSI_RETURN_TYPE_FLOAT:
      case TGSI_RETURN_TYPE_UNORM:
      case TGSI_RETURN_TYPE_SNORM:
         return FALSE;
      case TGSI_RETURN_TYPE_COUNT:
      default:
         assert(!"is_integer_type: Unknown tgsi_return_type");
         return FALSE;
   }
}


/**
 * Emit declarations for resources.
 * XXX When we're sure that all TGSI shaders will be generated with
 * sampler view declarations (Ex: DCL SVIEW[n], 2D, UINT) we may
 * rework this code.
 */
static boolean
emit_resource_declarations(struct svga_shader_emitter_v10 *emit)
{
   unsigned i;

   /* Emit resource decl for each sampler */
   for (i = 0; i < emit->num_samplers; i++) {
      VGPU10OpcodeToken0 opcode0;
      VGPU10OperandToken0 operand0;
      VGPU10ResourceReturnTypeToken return_type;
      VGPU10_RESOURCE_RETURN_TYPE rt;

      opcode0.value = 0;
      opcode0.opcodeType = VGPU10_OPCODE_DCL_RESOURCE;
      opcode0.resourceDimension =
         tgsi_texture_to_resource_dimension(emit->sampler_target[i],
                                            emit->key.tex[i].num_samples,
                                            emit->key.tex[i].is_array);
      opcode0.sampleCount = emit->key.tex[i].num_samples;
      operand0.value = 0;
      operand0.numComponents = VGPU10_OPERAND_0_COMPONENT;
      operand0.operandType = VGPU10_OPERAND_TYPE_RESOURCE;
      operand0.indexDimension = VGPU10_OPERAND_INDEX_1D;
      operand0.index0Representation = VGPU10_OPERAND_INDEX_IMMEDIATE32;

#if 1
      /* convert TGSI_RETURN_TYPE_x to VGPU10_RETURN_TYPE_x */
      STATIC_ASSERT(VGPU10_RETURN_TYPE_UNORM == TGSI_RETURN_TYPE_UNORM + 1);
      STATIC_ASSERT(VGPU10_RETURN_TYPE_SNORM == TGSI_RETURN_TYPE_SNORM + 1);
      STATIC_ASSERT(VGPU10_RETURN_TYPE_SINT == TGSI_RETURN_TYPE_SINT + 1);
      STATIC_ASSERT(VGPU10_RETURN_TYPE_UINT == TGSI_RETURN_TYPE_UINT + 1);
      STATIC_ASSERT(VGPU10_RETURN_TYPE_FLOAT == TGSI_RETURN_TYPE_FLOAT + 1);
      assert(emit->sampler_return_type[i] <= TGSI_RETURN_TYPE_FLOAT);
      rt = emit->sampler_return_type[i] + 1;
#else
      switch (emit->sampler_return_type[i]) {
         case TGSI_RETURN_TYPE_UNORM: rt = VGPU10_RETURN_TYPE_UNORM; break;
         case TGSI_RETURN_TYPE_SNORM: rt = VGPU10_RETURN_TYPE_SNORM; break;
         case TGSI_RETURN_TYPE_SINT:  rt = VGPU10_RETURN_TYPE_SINT;  break;
         case TGSI_RETURN_TYPE_UINT:  rt = VGPU10_RETURN_TYPE_UINT;  break;
         case TGSI_RETURN_TYPE_FLOAT: rt = VGPU10_RETURN_TYPE_FLOAT; break;
         case TGSI_RETURN_TYPE_COUNT:
         default:
            rt = VGPU10_RETURN_TYPE_FLOAT;
            assert(!"emit_resource_declarations: Unknown tgsi_return_type");
      }
#endif

      return_type.value = 0;
      return_type.component0 = rt;
      return_type.component1 = rt;
      return_type.component2 = rt;
      return_type.component3 = rt;

      begin_emit_instruction(emit);
      emit_dword(emit, opcode0.value);
      emit_dword(emit, operand0.value);
      emit_dword(emit, i);
      emit_dword(emit, return_type.value);
      end_emit_instruction(emit);
   }

   return TRUE;
}

static void
emit_instruction_op1(struct svga_shader_emitter_v10 *emit,
                     VGPU10_OPCODE_TYPE opcode,
                     const struct tgsi_full_dst_register *dst,
                     const struct tgsi_full_src_register *src,
                     boolean saturate)
{
   begin_emit_instruction(emit);
   emit_opcode(emit, opcode, saturate);
   emit_dst_register(emit, dst);
   emit_src_register(emit, src);
   end_emit_instruction(emit);
}

static void
emit_instruction_op2(struct svga_shader_emitter_v10 *emit,
                     VGPU10_OPCODE_TYPE opcode,
                     const struct tgsi_full_dst_register *dst,
                     const struct tgsi_full_src_register *src1,
                     const struct tgsi_full_src_register *src2,
                     boolean saturate)
{
   begin_emit_instruction(emit);
   emit_opcode(emit, opcode, saturate);
   emit_dst_register(emit, dst);
   emit_src_register(emit, src1);
   emit_src_register(emit, src2);
   end_emit_instruction(emit);
}

static void
emit_instruction_op3(struct svga_shader_emitter_v10 *emit,
                     VGPU10_OPCODE_TYPE opcode,
                     const struct tgsi_full_dst_register *dst,
                     const struct tgsi_full_src_register *src1,
                     const struct tgsi_full_src_register *src2,
                     const struct tgsi_full_src_register *src3,
                     boolean saturate)
{
   begin_emit_instruction(emit);
   emit_opcode(emit, opcode, saturate);
   emit_dst_register(emit, dst);
   emit_src_register(emit, src1);
   emit_src_register(emit, src2);
   emit_src_register(emit, src3);
   end_emit_instruction(emit);
}

/**
 * Emit the actual clip distance instructions to be used for clipping
 * by copying the clip distance from the temporary registers to the
 * CLIPDIST registers written with the enabled planes mask.
 * Also copy the clip distance from the temporary to the clip distance
 * shadow copy register which will be referenced by the input shader
 */
static void
emit_clip_distance_instructions(struct svga_shader_emitter_v10 *emit)
{
   struct tgsi_full_src_register tmp_clip_dist_src;
   struct tgsi_full_dst_register clip_dist_dst;

   unsigned i;
   unsigned clip_plane_enable = emit->key.clip_plane_enable;
   unsigned clip_dist_tmp_index = emit->clip_dist_tmp_index;
   int num_written_clipdist = emit->info.num_written_clipdistance;

   assert(emit->clip_dist_out_index != INVALID_INDEX);
   assert(emit->clip_dist_tmp_index != INVALID_INDEX);

   /**
    * Temporary reset the temporary clip dist register index so
    * that the copy to the real clip dist register will not
    * attempt to copy to the temporary register again
    */
   emit->clip_dist_tmp_index = INVALID_INDEX;

   for (i = 0; i < 2 && num_written_clipdist > 0; i++, num_written_clipdist-=4) {

      tmp_clip_dist_src = make_src_temp_reg(clip_dist_tmp_index + i);

      /**
       * copy to the shadow copy for use by varying variable and
       * stream output. All clip distances
       * will be written regardless of the enabled clipping planes.
       */
      clip_dist_dst = make_dst_reg(TGSI_FILE_OUTPUT,
                                   emit->clip_dist_so_index + i);

      /* MOV clip_dist_so, tmp_clip_dist */
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &clip_dist_dst,
                           &tmp_clip_dist_src, FALSE);

      /**
       * copy those clip distances to enabled clipping planes
       * to CLIPDIST registers for clipping
       */
      if (clip_plane_enable & 0xf) {
         clip_dist_dst = make_dst_reg(TGSI_FILE_OUTPUT,
                                      emit->clip_dist_out_index + i);
         clip_dist_dst = writemask_dst(&clip_dist_dst, clip_plane_enable & 0xf);

         /* MOV CLIPDIST, tmp_clip_dist */
         emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &clip_dist_dst,
                              &tmp_clip_dist_src, FALSE);
      }
      /* four clip planes per clip register */
      clip_plane_enable >>= 4;
   }
   /**
    * set the temporary clip dist register index back to the
    * temporary index for the next vertex
    */
   emit->clip_dist_tmp_index = clip_dist_tmp_index;
}

/* Declare clip distance output registers for user-defined clip planes
 * or the TGSI_CLIPVERTEX output.
 */
static void
emit_clip_distance_declarations(struct svga_shader_emitter_v10 *emit)
{
   unsigned num_clip_planes = util_bitcount(emit->key.clip_plane_enable);
   unsigned index = emit->num_outputs;
   unsigned plane_mask;

   assert(emit->unit == PIPE_SHADER_VERTEX ||
          emit->unit == PIPE_SHADER_GEOMETRY);
   assert(num_clip_planes <= 8);

   if (emit->clip_mode != CLIP_LEGACY &&
       emit->clip_mode != CLIP_VERTEX) {
      return;
   }

   if (num_clip_planes == 0)
      return;

   /* Declare one or two clip output registers.  The number of components
    * in the mask reflects the number of clip planes.  For example, if 5
    * clip planes are needed, we'll declare outputs similar to:
    * dcl_output_siv o2.xyzw, clip_distance
    * dcl_output_siv o3.x, clip_distance
    */
   emit->clip_dist_out_index = index; /* save the starting clip dist reg index */

   plane_mask = (1 << num_clip_planes) - 1;
   if (plane_mask & 0xf) {
      unsigned cmask = plane_mask & VGPU10_OPERAND_4_COMPONENT_MASK_ALL;
      emit_output_declaration(emit, VGPU10_OPCODE_DCL_OUTPUT_SIV, index,
                              VGPU10_NAME_CLIP_DISTANCE, cmask);
      emit->num_outputs++;
   }
   if (plane_mask & 0xf0) {
      unsigned cmask = (plane_mask >> 4) & VGPU10_OPERAND_4_COMPONENT_MASK_ALL;
      emit_output_declaration(emit, VGPU10_OPCODE_DCL_OUTPUT_SIV, index + 1,
                              VGPU10_NAME_CLIP_DISTANCE, cmask);
      emit->num_outputs++;
   }
}


/**
 * Emit the instructions for writing to the clip distance registers
 * to handle legacy/automatic clip planes.
 * For each clip plane, the distance is the dot product of the vertex
 * position (found in TEMP[vpos_tmp_index]) and the clip plane coefficients.
 * This is not used when the shader has an explicit CLIPVERTEX or CLIPDISTANCE
 * output registers already declared.
 */
static void
emit_clip_distance_from_vpos(struct svga_shader_emitter_v10 *emit,
                             unsigned vpos_tmp_index)
{
   unsigned i, num_clip_planes = util_bitcount(emit->key.clip_plane_enable);

   assert(emit->clip_mode == CLIP_LEGACY);
   assert(num_clip_planes <= 8);

   assert(emit->unit == PIPE_SHADER_VERTEX ||
          emit->unit == PIPE_SHADER_GEOMETRY);

   for (i = 0; i < num_clip_planes; i++) {
      struct tgsi_full_dst_register dst;
      struct tgsi_full_src_register plane_src, vpos_src;
      unsigned reg_index = emit->clip_dist_out_index + i / 4;
      unsigned comp = i % 4;
      unsigned writemask = VGPU10_OPERAND_4_COMPONENT_MASK_X << comp;

      /* create dst, src regs */
      dst = make_dst_reg(TGSI_FILE_OUTPUT, reg_index);
      dst = writemask_dst(&dst, writemask);

      plane_src = make_src_const_reg(emit->clip_plane_const[i]);
      vpos_src = make_src_temp_reg(vpos_tmp_index);

      /* DP4 clip_dist, plane, vpos */
      emit_instruction_op2(emit, VGPU10_OPCODE_DP4, &dst,
                           &plane_src, &vpos_src, FALSE);
   }
}


/**
 * Emit the instructions for computing the clip distance results from
 * the clip vertex temporary.
 * For each clip plane, the distance is the dot product of the clip vertex
 * position (found in a temp reg) and the clip plane coefficients.
 */
static void
emit_clip_vertex_instructions(struct svga_shader_emitter_v10 *emit)
{
   const unsigned num_clip = util_bitcount(emit->key.clip_plane_enable);
   unsigned i;
   struct tgsi_full_dst_register dst;
   struct tgsi_full_src_register clipvert_src;
   const unsigned clip_vertex_tmp = emit->clip_vertex_tmp_index;

   assert(emit->unit == PIPE_SHADER_VERTEX ||
          emit->unit == PIPE_SHADER_GEOMETRY);

   assert(emit->clip_mode == CLIP_VERTEX);

   clipvert_src = make_src_temp_reg(clip_vertex_tmp);

   for (i = 0; i < num_clip; i++) {
      struct tgsi_full_src_register plane_src;
      unsigned reg_index = emit->clip_dist_out_index + i / 4;
      unsigned comp = i % 4;
      unsigned writemask = VGPU10_OPERAND_4_COMPONENT_MASK_X << comp;

      /* create dst, src regs */
      dst = make_dst_reg(TGSI_FILE_OUTPUT, reg_index);
      dst = writemask_dst(&dst, writemask);

      plane_src = make_src_const_reg(emit->clip_plane_const[i]);

      /* DP4 clip_dist, plane, vpos */
      emit_instruction_op2(emit, VGPU10_OPCODE_DP4, &dst,
                           &plane_src, &clipvert_src, FALSE);
   }

   /* copy temporary clip vertex register to the clip vertex register */

   assert(emit->clip_vertex_out_index != INVALID_INDEX);

   /**
    * temporary reset the temporary clip vertex register index so
    * that copy to the clip vertex register will not attempt
    * to copy to the temporary register again
    */
   emit->clip_vertex_tmp_index = INVALID_INDEX;

   /* MOV clip_vertex, clip_vertex_tmp */
   dst = make_dst_reg(TGSI_FILE_OUTPUT, emit->clip_vertex_out_index);
   emit_instruction_op1(emit, VGPU10_OPCODE_MOV,
                        &dst, &clipvert_src, FALSE);

   /**
    * set the temporary clip vertex register index back to the
    * temporary index for the next vertex
    */
   emit->clip_vertex_tmp_index = clip_vertex_tmp;
}

/**
 * Emit code to convert RGBA to BGRA
 */
static void
emit_swap_r_b(struct svga_shader_emitter_v10 *emit,
                     const struct tgsi_full_dst_register *dst,
                     const struct tgsi_full_src_register *src)
{
   struct tgsi_full_src_register bgra_src =
      swizzle_src(src, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_X, TGSI_SWIZZLE_W);

   begin_emit_instruction(emit);
   emit_opcode(emit, VGPU10_OPCODE_MOV, FALSE);
   emit_dst_register(emit, dst);
   emit_src_register(emit, &bgra_src);
   end_emit_instruction(emit);
}


/** Convert from 10_10_10_2 normalized to 10_10_10_2_snorm */
static void
emit_puint_to_snorm(struct svga_shader_emitter_v10 *emit,
                    const struct tgsi_full_dst_register *dst,
                    const struct tgsi_full_src_register *src)
{
   struct tgsi_full_src_register half = make_immediate_reg_float(emit, 0.5f);
   struct tgsi_full_src_register two =
      make_immediate_reg_float4(emit, 2.0f, 2.0f, 2.0f, 3.0f);
   struct tgsi_full_src_register neg_two =
      make_immediate_reg_float4(emit, -2.0f, -2.0f, -2.0f, -1.66666f);

   unsigned val_tmp = get_temp_index(emit);
   struct tgsi_full_dst_register val_dst = make_dst_temp_reg(val_tmp);
   struct tgsi_full_src_register val_src = make_src_temp_reg(val_tmp);

   unsigned bias_tmp = get_temp_index(emit);
   struct tgsi_full_dst_register bias_dst = make_dst_temp_reg(bias_tmp);
   struct tgsi_full_src_register bias_src = make_src_temp_reg(bias_tmp);

   /* val = src * 2.0 */
   emit_instruction_op2(emit, VGPU10_OPCODE_MUL, &val_dst,
                        src, &two, FALSE);

   /* bias = src > 0.5 */
   emit_instruction_op2(emit, VGPU10_OPCODE_GE, &bias_dst,
                        src, &half, FALSE);

   /* bias = bias & -2.0 */
   emit_instruction_op2(emit, VGPU10_OPCODE_AND, &bias_dst,
                        &bias_src, &neg_two, FALSE);

   /* dst = val + bias */
   emit_instruction_op2(emit, VGPU10_OPCODE_ADD, dst,
                        &val_src, &bias_src, FALSE);

   free_temp_indexes(emit);
}


/** Convert from 10_10_10_2_unorm to 10_10_10_2_uscaled */
static void
emit_puint_to_uscaled(struct svga_shader_emitter_v10 *emit,
                      const struct tgsi_full_dst_register *dst,
                      const struct tgsi_full_src_register *src)
{
   struct tgsi_full_src_register scale =
      make_immediate_reg_float4(emit, 1023.0f, 1023.0f, 1023.0f, 3.0f);

   /* dst = src * scale */
   emit_instruction_op2(emit, VGPU10_OPCODE_MUL, dst, src, &scale, FALSE);
}


/** Convert from R32_UINT to 10_10_10_2_sscaled */
static void
emit_puint_to_sscaled(struct svga_shader_emitter_v10 *emit,
                      const struct tgsi_full_dst_register *dst,
                      const struct tgsi_full_src_register *src)
{
   struct tgsi_full_src_register lshift =
      make_immediate_reg_int4(emit, 22, 12, 2, 0);
   struct tgsi_full_src_register rshift =
      make_immediate_reg_int4(emit, 22, 22, 22, 30);

   struct tgsi_full_src_register src_xxxx = scalar_src(src, TGSI_SWIZZLE_X);

   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);

   /*
    * r = (pixel << 22) >> 22;   # signed int in [511, -512]
    * g = (pixel << 12) >> 22;   # signed int in [511, -512]
    * b = (pixel <<  2) >> 22;   # signed int in [511, -512]
    * a = (pixel <<  0) >> 30;   # signed int in [1, -2]
    * dst = i_to_f(r,g,b,a);     # convert to float
    */
   emit_instruction_op2(emit, VGPU10_OPCODE_ISHL, &tmp_dst,
                        &src_xxxx, &lshift, FALSE);
   emit_instruction_op2(emit, VGPU10_OPCODE_ISHR, &tmp_dst,
                        &tmp_src, &rshift, FALSE);
   emit_instruction_op1(emit, VGPU10_OPCODE_ITOF, dst, &tmp_src, FALSE);

   free_temp_indexes(emit);
}


/**
 * Emit code for TGSI_OPCODE_ARL or TGSI_OPCODE_UARL instruction.
 */
static boolean
emit_arl_uarl(struct svga_shader_emitter_v10 *emit,
              const struct tgsi_full_instruction *inst)
{
   unsigned index = inst->Dst[0].Register.Index;
   struct tgsi_full_dst_register dst;
   VGPU10_OPCODE_TYPE opcode;

   assert(index < MAX_VGPU10_ADDR_REGS);
   dst = make_dst_temp_reg(emit->address_reg_index[index]);

   /* ARL dst, s0
    * Translates into:
    * FTOI address_tmp, s0
    *
    * UARL dst, s0
    * Translates into:
    * MOV address_tmp, s0
    */
   if (inst->Instruction.Opcode == TGSI_OPCODE_ARL)
      opcode = VGPU10_OPCODE_FTOI;
   else
      opcode = VGPU10_OPCODE_MOV;

   emit_instruction_op1(emit, opcode, &dst, &inst->Src[0], FALSE);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_CAL instruction.
 */
static boolean
emit_cal(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   unsigned label = inst->Label.Label;
   VGPU10OperandToken0 operand;
   operand.value = 0;
   operand.operandType = VGPU10_OPERAND_TYPE_LABEL;

   begin_emit_instruction(emit);
   emit_dword(emit, operand.value);
   emit_dword(emit, label);
   end_emit_instruction(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_IABS instruction.
 */
static boolean
emit_iabs(struct svga_shader_emitter_v10 *emit,
          const struct tgsi_full_instruction *inst)
{
   /* dst.x = (src0.x < 0) ? -src0.x : src0.x
    * dst.y = (src0.y < 0) ? -src0.y : src0.y
    * dst.z = (src0.z < 0) ? -src0.z : src0.z
    * dst.w = (src0.w < 0) ? -src0.w : src0.w
    *
    * Translates into
    *   IMAX dst, src, neg(src)
    */
   struct tgsi_full_src_register neg_src = negate_src(&inst->Src[0]);
   emit_instruction_op2(emit, VGPU10_OPCODE_IMAX, &inst->Dst[0],
                        &inst->Src[0], &neg_src, FALSE);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_CMP instruction.
 */
static boolean
emit_cmp(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* dst.x = (src0.x < 0) ? src1.x : src2.x
    * dst.y = (src0.y < 0) ? src1.y : src2.y
    * dst.z = (src0.z < 0) ? src1.z : src2.z
    * dst.w = (src0.w < 0) ? src1.w : src2.w
    *
    * Translates into
    *   LT tmp, src0, 0.0
    *   MOVC dst, tmp, src1, src2
    */
   struct tgsi_full_src_register zero = make_immediate_reg_float(emit, 0.0f);
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);

   emit_instruction_op2(emit, VGPU10_OPCODE_LT, &tmp_dst,
                        &inst->Src[0], &zero, FALSE);
   emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &inst->Dst[0],
                        &tmp_src, &inst->Src[1], &inst->Src[2],
                        inst->Instruction.Saturate);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_DST instruction.
 */
static boolean
emit_dst(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /*
    * dst.x = 1
    * dst.y = src0.y * src1.y
    * dst.z = src0.z
    * dst.w = src1.w
    */

   struct tgsi_full_src_register s0_yyyy =
      scalar_src(&inst->Src[0], TGSI_SWIZZLE_Y);
   struct tgsi_full_src_register s0_zzzz =
      scalar_src(&inst->Src[0], TGSI_SWIZZLE_Z);
   struct tgsi_full_src_register s1_yyyy =
      scalar_src(&inst->Src[1], TGSI_SWIZZLE_Y);
   struct tgsi_full_src_register s1_wwww =
      scalar_src(&inst->Src[1], TGSI_SWIZZLE_W);

   /*
    * If dst and either src0 and src1 are the same we need
    * to create a temporary for it and insert a extra move.
    */
   unsigned tmp_move = get_temp_index(emit);
   struct tgsi_full_src_register move_src = make_src_temp_reg(tmp_move);
   struct tgsi_full_dst_register move_dst = make_dst_temp_reg(tmp_move);

   /* MOV dst.x, 1.0 */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_X) {
      struct tgsi_full_dst_register dst_x =
         writemask_dst(&move_dst, TGSI_WRITEMASK_X);
      struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);

      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &dst_x, &one, FALSE);
   }

   /* MUL dst.y, s0.y, s1.y */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_Y) {
      struct tgsi_full_dst_register dst_y =
         writemask_dst(&move_dst, TGSI_WRITEMASK_Y);

      emit_instruction_op2(emit, VGPU10_OPCODE_MUL, &dst_y, &s0_yyyy,
                           &s1_yyyy, inst->Instruction.Saturate);
   }

   /* MOV dst.z, s0.z */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_Z) {
      struct tgsi_full_dst_register dst_z =
         writemask_dst(&move_dst, TGSI_WRITEMASK_Z);

      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &dst_z, &s0_zzzz,
                           inst->Instruction.Saturate);
  }

   /* MOV dst.w, s1.w */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_W) {
      struct tgsi_full_dst_register dst_w =
         writemask_dst(&move_dst, TGSI_WRITEMASK_W);

      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &dst_w, &s1_wwww,
                           inst->Instruction.Saturate);
   }

   emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &inst->Dst[0], &move_src,
                        FALSE);
   free_temp_indexes(emit);

   return TRUE;
}



/**
 * Emit code for TGSI_OPCODE_ENDPRIM (GS only)
 */
static boolean
emit_endprim(struct svga_shader_emitter_v10 *emit,
             const struct tgsi_full_instruction *inst)
{
   assert(emit->unit == PIPE_SHADER_GEOMETRY);

   /* We can't use emit_simple() because the TGSI instruction has one
    * operand (vertex stream number) which we must ignore for VGPU10.
    */
   begin_emit_instruction(emit);
   emit_opcode(emit, VGPU10_OPCODE_CUT, FALSE);
   end_emit_instruction(emit);
   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_EX2 (2^x) instruction.
 */
static boolean
emit_ex2(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* Note that TGSI_OPCODE_EX2 computes only one value from src.x
    * while VGPU10 computes four values.
    *
    * dst = EX2(src):
    *   dst.xyzw = 2.0 ^ src.x
    */

   struct tgsi_full_src_register src_xxxx =
      swizzle_src(&inst->Src[0], TGSI_SWIZZLE_X, TGSI_SWIZZLE_X,
                  TGSI_SWIZZLE_X, TGSI_SWIZZLE_X);

   /* EXP tmp, s0.xxxx */
   emit_instruction_op1(emit, VGPU10_OPCODE_EXP, &inst->Dst[0], &src_xxxx,
                        inst->Instruction.Saturate);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_EXP instruction.
 */
static boolean
emit_exp(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /*
    * dst.x = 2 ^ floor(s0.x)
    * dst.y = s0.x - floor(s0.x)
    * dst.z = 2 ^ s0.x
    * dst.w = 1.0
    */

   struct tgsi_full_src_register src_xxxx =
      scalar_src(&inst->Src[0], TGSI_SWIZZLE_X);
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);

   /*
    * If dst and src are the same we need to create
    * a temporary for it and insert a extra move.
    */
   unsigned tmp_move = get_temp_index(emit);
   struct tgsi_full_src_register move_src = make_src_temp_reg(tmp_move);
   struct tgsi_full_dst_register move_dst = make_dst_temp_reg(tmp_move);

   /* only use X component of temp reg */
   tmp_dst = writemask_dst(&tmp_dst, TGSI_WRITEMASK_X);
   tmp_src = scalar_src(&tmp_src, TGSI_SWIZZLE_X);

   /* ROUND_NI tmp.x, s0.x */
   emit_instruction_op1(emit, VGPU10_OPCODE_ROUND_NI, &tmp_dst,
                        &src_xxxx, FALSE); /* round to -infinity */

   /* EXP dst.x, tmp.x */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_X) {
      struct tgsi_full_dst_register dst_x =
         writemask_dst(&move_dst, TGSI_WRITEMASK_X);

      emit_instruction_op1(emit, VGPU10_OPCODE_EXP, &dst_x, &tmp_src,
                           inst->Instruction.Saturate);
   }

   /* ADD dst.y, s0.x, -tmp */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_Y) {
      struct tgsi_full_dst_register dst_y =
         writemask_dst(&move_dst, TGSI_WRITEMASK_Y);
      struct tgsi_full_src_register neg_tmp_src = negate_src(&tmp_src);

      emit_instruction_op2(emit, VGPU10_OPCODE_ADD, &dst_y, &src_xxxx,
                           &neg_tmp_src, inst->Instruction.Saturate);
   }

   /* EXP dst.z, s0.x */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_Z) {
      struct tgsi_full_dst_register dst_z =
         writemask_dst(&move_dst, TGSI_WRITEMASK_Z);

      emit_instruction_op1(emit, VGPU10_OPCODE_EXP, &dst_z, &src_xxxx,
                           inst->Instruction.Saturate);
   }

   /* MOV dst.w, 1.0 */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_W) {
      struct tgsi_full_dst_register dst_w =
         writemask_dst(&move_dst, TGSI_WRITEMASK_W);
      struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);

      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &dst_w, &one,
                           FALSE);
   }

   emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &inst->Dst[0], &move_src,
                        FALSE);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_IF instruction.
 */
static boolean
emit_if(struct svga_shader_emitter_v10 *emit,
        const struct tgsi_full_instruction *inst)
{
   VGPU10OpcodeToken0 opcode0;

   /* The src register should be a scalar */
   assert(inst->Src[0].Register.SwizzleX == inst->Src[0].Register.SwizzleY &&
          inst->Src[0].Register.SwizzleX == inst->Src[0].Register.SwizzleZ &&
          inst->Src[0].Register.SwizzleX == inst->Src[0].Register.SwizzleW);

   /* The only special thing here is that we need to set the
    * VGPU10_INSTRUCTION_TEST_NONZERO flag since we want to test if
    * src.x is non-zero.
    */
   opcode0.value = 0;
   opcode0.opcodeType = VGPU10_OPCODE_IF;
   opcode0.testBoolean = VGPU10_INSTRUCTION_TEST_NONZERO;

   begin_emit_instruction(emit);
   emit_dword(emit, opcode0.value);
   emit_src_register(emit, &inst->Src[0]);
   end_emit_instruction(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_KILL_IF instruction (kill fragment if any of
 * the register components are negative).
 */
static boolean
emit_kill_if(struct svga_shader_emitter_v10 *emit,
             const struct tgsi_full_instruction *inst)
{
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);

   struct tgsi_full_src_register zero = make_immediate_reg_float(emit, 0.0f);

   struct tgsi_full_dst_register tmp_dst_x =
      writemask_dst(&tmp_dst, TGSI_WRITEMASK_X);
   struct tgsi_full_src_register tmp_src_xxxx =
      scalar_src(&tmp_src, TGSI_SWIZZLE_X);

   /* tmp = src[0] < 0.0 */
   emit_instruction_op2(emit, VGPU10_OPCODE_LT, &tmp_dst, &inst->Src[0],
                        &zero, FALSE);

   if (!same_swizzle_terms(&inst->Src[0])) {
      /* If the swizzle is not XXXX, YYYY, ZZZZ or WWWW we need to
       * logically OR the swizzle terms.  Most uses of KILL_IF only
       * test one channel so it's good to avoid these extra steps.
       */
      struct tgsi_full_src_register tmp_src_yyyy =
         scalar_src(&tmp_src, TGSI_SWIZZLE_Y);
      struct tgsi_full_src_register tmp_src_zzzz =
         scalar_src(&tmp_src, TGSI_SWIZZLE_Z);
      struct tgsi_full_src_register tmp_src_wwww =
         scalar_src(&tmp_src, TGSI_SWIZZLE_W);

      emit_instruction_op2(emit, VGPU10_OPCODE_OR, &tmp_dst_x, &tmp_src_xxxx,
                           &tmp_src_yyyy, FALSE);
      emit_instruction_op2(emit, VGPU10_OPCODE_OR, &tmp_dst_x, &tmp_src_xxxx,
                           &tmp_src_zzzz, FALSE);
      emit_instruction_op2(emit, VGPU10_OPCODE_OR, &tmp_dst_x, &tmp_src_xxxx,
                           &tmp_src_wwww, FALSE);
   }

   begin_emit_instruction(emit);
   emit_discard_opcode(emit, TRUE); /* discard if src0.x is non-zero */
   emit_src_register(emit, &tmp_src_xxxx);
   end_emit_instruction(emit);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_KILL instruction (unconditional discard).
 */
static boolean
emit_kill(struct svga_shader_emitter_v10 *emit,
          const struct tgsi_full_instruction *inst)
{
   struct tgsi_full_src_register zero = make_immediate_reg_float(emit, 0.0f);

   /* DISCARD if 0.0 is zero */
   begin_emit_instruction(emit);
   emit_discard_opcode(emit, FALSE);
   emit_src_register(emit, &zero);
   end_emit_instruction(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_LG2 instruction.
 */
static boolean
emit_lg2(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* Note that TGSI_OPCODE_LG2 computes only one value from src.x
    * while VGPU10 computes four values.
    *
    * dst = LG2(src):
    *   dst.xyzw = log2(src.x)
    */

   struct tgsi_full_src_register src_xxxx =
      swizzle_src(&inst->Src[0], TGSI_SWIZZLE_X, TGSI_SWIZZLE_X,
                  TGSI_SWIZZLE_X, TGSI_SWIZZLE_X);

   /* LOG tmp, s0.xxxx */
   emit_instruction_op1(emit, VGPU10_OPCODE_LOG, &inst->Dst[0], &src_xxxx,
                        inst->Instruction.Saturate);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_LIT instruction.
 */
static boolean
emit_lit(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);

   /*
    * If dst and src are the same we need to create
    * a temporary for it and insert a extra move.
    */
   unsigned tmp_move = get_temp_index(emit);
   struct tgsi_full_src_register move_src = make_src_temp_reg(tmp_move);
   struct tgsi_full_dst_register move_dst = make_dst_temp_reg(tmp_move);

   /*
    * dst.x = 1
    * dst.y = max(src.x, 0)
    * dst.z = (src.x > 0) ? max(src.y, 0)^{clamp(src.w, -128, 128))} : 0
    * dst.w = 1
    */

   /* MOV dst.x, 1.0 */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_X) {
      struct tgsi_full_dst_register dst_x =
         writemask_dst(&move_dst, TGSI_WRITEMASK_X);
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &dst_x, &one, FALSE);
   }

   /* MOV dst.w, 1.0 */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_W) {
      struct tgsi_full_dst_register dst_w =
         writemask_dst(&move_dst, TGSI_WRITEMASK_W);
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &dst_w, &one, FALSE);
   }

   /* MAX dst.y, src.x, 0.0 */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_Y) {
      struct tgsi_full_dst_register dst_y =
         writemask_dst(&move_dst, TGSI_WRITEMASK_Y);
      struct tgsi_full_src_register zero =
         make_immediate_reg_float(emit, 0.0f);
      struct tgsi_full_src_register src_xxxx =
         swizzle_src(&inst->Src[0], TGSI_SWIZZLE_X, TGSI_SWIZZLE_X,
                     TGSI_SWIZZLE_X, TGSI_SWIZZLE_X);

      emit_instruction_op2(emit, VGPU10_OPCODE_MAX, &dst_y, &src_xxxx,
                           &zero, inst->Instruction.Saturate);
   }

   /*
    * tmp1 = clamp(src.w, -128, 128);
    *   MAX tmp1, src.w, -128
    *   MIN tmp1, tmp1, 128
    *
    * tmp2 = max(tmp2, 0);
    *   MAX tmp2, src.y, 0
    *
    * tmp1 = pow(tmp2, tmp1);
    *   LOG tmp2, tmp2
    *   MUL tmp1, tmp2, tmp1
    *   EXP tmp1, tmp1
    *
    * tmp1 = (src.w == 0) ? 1 : tmp1;
    *   EQ tmp2, 0, src.w
    *   MOVC tmp1, tmp2, 1.0, tmp1
    *
    * dst.z = (0 < src.x) ? tmp1 : 0;
    *   LT tmp2, 0, src.x
    *   MOVC dst.z, tmp2, tmp1, 0.0
    */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_Z) {
      struct tgsi_full_dst_register dst_z =
         writemask_dst(&move_dst, TGSI_WRITEMASK_Z);

      unsigned tmp1 = get_temp_index(emit);
      struct tgsi_full_src_register tmp1_src = make_src_temp_reg(tmp1);
      struct tgsi_full_dst_register tmp1_dst = make_dst_temp_reg(tmp1);
      unsigned tmp2 = get_temp_index(emit);
      struct tgsi_full_src_register tmp2_src = make_src_temp_reg(tmp2);
      struct tgsi_full_dst_register tmp2_dst = make_dst_temp_reg(tmp2);

      struct tgsi_full_src_register src_xxxx =
         scalar_src(&inst->Src[0], TGSI_SWIZZLE_X);
      struct tgsi_full_src_register src_yyyy =
         scalar_src(&inst->Src[0], TGSI_SWIZZLE_Y);
      struct tgsi_full_src_register src_wwww =
         scalar_src(&inst->Src[0], TGSI_SWIZZLE_W);

      struct tgsi_full_src_register zero =
         make_immediate_reg_float(emit, 0.0f);
      struct tgsi_full_src_register lowerbound =
         make_immediate_reg_float(emit, -128.0f);
      struct tgsi_full_src_register upperbound =
         make_immediate_reg_float(emit, 128.0f);

      emit_instruction_op2(emit, VGPU10_OPCODE_MAX, &tmp1_dst, &src_wwww,
                           &lowerbound, FALSE);
      emit_instruction_op2(emit, VGPU10_OPCODE_MIN, &tmp1_dst, &tmp1_src,
                           &upperbound, FALSE);
      emit_instruction_op2(emit, VGPU10_OPCODE_MAX, &tmp2_dst, &src_yyyy,
                           &zero, FALSE);

      /* POW tmp1, tmp2, tmp1 */
      /* LOG tmp2, tmp2 */
      emit_instruction_op1(emit, VGPU10_OPCODE_LOG, &tmp2_dst, &tmp2_src,
                           FALSE);

      /* MUL tmp1, tmp2, tmp1 */
      emit_instruction_op2(emit, VGPU10_OPCODE_MUL, &tmp1_dst, &tmp2_src,
                           &tmp1_src, FALSE);

      /* EXP tmp1, tmp1 */
      emit_instruction_op1(emit, VGPU10_OPCODE_EXP, &tmp1_dst, &tmp1_src,
                           FALSE);

      /* EQ tmp2, 0, src.w */
      emit_instruction_op2(emit, VGPU10_OPCODE_EQ, &tmp2_dst, &zero,
                           &src_wwww, FALSE);
      /* MOVC tmp1.z, tmp2, tmp1, 1.0 */
      emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &tmp1_dst,
                           &tmp2_src, &one, &tmp1_src, FALSE);

      /* LT tmp2, 0, src.x */
      emit_instruction_op2(emit, VGPU10_OPCODE_LT, &tmp2_dst, &zero,
                           &src_xxxx, FALSE);
      /* MOVC dst.z, tmp2, tmp1, 0.0 */
      emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &dst_z,
                           &tmp2_src, &tmp1_src, &zero, FALSE);
   }

   emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &inst->Dst[0], &move_src,
                        FALSE);
   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit Level Of Detail Query (LODQ) instruction.
 */
static boolean
emit_lodq(struct svga_shader_emitter_v10 *emit,
          const struct tgsi_full_instruction *inst)
{
   const uint unit = inst->Src[1].Register.Index;

   assert(emit->version >= 41);

   /* LOD dst, coord, resource, sampler */
   begin_emit_instruction(emit);
   emit_opcode(emit, VGPU10_OPCODE_LOD, FALSE);
   emit_dst_register(emit, &inst->Dst[0]);
   emit_src_register(emit, &inst->Src[0]); /* coord */
   emit_resource_register(emit, unit);
   emit_sampler_register(emit, unit);
   end_emit_instruction(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_LOG instruction.
 */
static boolean
emit_log(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /*
    * dst.x = floor(lg2(abs(s0.x)))
    * dst.y = abs(s0.x) / (2 ^ floor(lg2(abs(s0.x))))
    * dst.z = lg2(abs(s0.x))
    * dst.w = 1.0
    */

   struct tgsi_full_src_register src_xxxx =
      scalar_src(&inst->Src[0], TGSI_SWIZZLE_X);
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register abs_src_xxxx = absolute_src(&src_xxxx);

   /* only use X component of temp reg */
   tmp_dst = writemask_dst(&tmp_dst, TGSI_WRITEMASK_X);
   tmp_src = scalar_src(&tmp_src, TGSI_SWIZZLE_X);

   /* LOG tmp.x, abs(s0.x) */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_XYZ) {
      emit_instruction_op1(emit, VGPU10_OPCODE_LOG, &tmp_dst,
                          &abs_src_xxxx, FALSE);
   }

   /* MOV dst.z, tmp.x */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_Z) {
      struct tgsi_full_dst_register dst_z =
         writemask_dst(&inst->Dst[0], TGSI_WRITEMASK_Z);

      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &dst_z,
                           &tmp_src, inst->Instruction.Saturate);
   }

   /* FLR tmp.x, tmp.x */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_XY) {
      emit_instruction_op1(emit, VGPU10_OPCODE_ROUND_NI, &tmp_dst,
                           &tmp_src, FALSE);
   }

   /* MOV dst.x, tmp.x */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_X) {
      struct tgsi_full_dst_register dst_x =
         writemask_dst(&inst->Dst[0], TGSI_WRITEMASK_X);

      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &dst_x, &tmp_src,
                           inst->Instruction.Saturate);
   }

   /* EXP tmp.x, tmp.x */
   /* DIV dst.y, abs(s0.x), tmp.x */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_Y) {
      struct tgsi_full_dst_register dst_y =
         writemask_dst(&inst->Dst[0], TGSI_WRITEMASK_Y);

      emit_instruction_op1(emit, VGPU10_OPCODE_EXP, &tmp_dst, &tmp_src,
                           FALSE);
      emit_instruction_op2(emit, VGPU10_OPCODE_DIV, &dst_y, &abs_src_xxxx,
                           &tmp_src, inst->Instruction.Saturate);
   }

   /* MOV dst.w, 1.0 */
   if (inst->Dst[0].Register.WriteMask & TGSI_WRITEMASK_W) {
      struct tgsi_full_dst_register dst_w =
         writemask_dst(&inst->Dst[0], TGSI_WRITEMASK_W);
      struct tgsi_full_src_register one =
         make_immediate_reg_float(emit, 1.0f);

      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &dst_w, &one, FALSE);
   }

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_LRP instruction.
 */
static boolean
emit_lrp(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* dst = LRP(s0, s1, s2):
    *   dst = s0 * (s1 - s2) + s2
    * Translates into:
    *   SUB tmp, s1, s2;        tmp = s1 - s2
    *   MAD dst, s0, tmp, s2;   dst = s0 * t1 + s2
    */
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register src_tmp = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register dst_tmp = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register neg_src2 = negate_src(&inst->Src[2]);

   /* ADD tmp, s1, -s2 */
   emit_instruction_op2(emit, VGPU10_OPCODE_ADD, &dst_tmp,
                        &inst->Src[1], &neg_src2, FALSE);

   /* MAD dst, s1, tmp, s3 */
   emit_instruction_op3(emit, VGPU10_OPCODE_MAD, &inst->Dst[0],
                        &inst->Src[0], &src_tmp, &inst->Src[2],
                        inst->Instruction.Saturate);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_POW instruction.
 */
static boolean
emit_pow(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* Note that TGSI_OPCODE_POW computes only one value from src0.x and
    * src1.x while VGPU10 computes four values.
    *
    * dst = POW(src0, src1):
    *   dst.xyzw = src0.x ^ src1.x
    */
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register src0_xxxx =
      swizzle_src(&inst->Src[0], TGSI_SWIZZLE_X, TGSI_SWIZZLE_X,
                  TGSI_SWIZZLE_X, TGSI_SWIZZLE_X);
   struct tgsi_full_src_register src1_xxxx =
      swizzle_src(&inst->Src[1], TGSI_SWIZZLE_X, TGSI_SWIZZLE_X,
                  TGSI_SWIZZLE_X, TGSI_SWIZZLE_X);

   /* LOG tmp, s0.xxxx */
   emit_instruction_op1(emit, VGPU10_OPCODE_LOG, &tmp_dst, &src0_xxxx,
                        FALSE);

   /* MUL tmp, tmp, s1.xxxx */
   emit_instruction_op2(emit, VGPU10_OPCODE_MUL, &tmp_dst, &tmp_src,
                        &src1_xxxx, FALSE);

   /* EXP tmp, s0.xxxx */
   emit_instruction_op1(emit, VGPU10_OPCODE_EXP, &inst->Dst[0],
                        &tmp_src, inst->Instruction.Saturate);

   /* free tmp */
   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_RCP (reciprocal) instruction.
 */
static boolean
emit_rcp(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);

   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);

   struct tgsi_full_dst_register tmp_dst_x =
      writemask_dst(&tmp_dst, TGSI_WRITEMASK_X);
   struct tgsi_full_src_register tmp_src_xxxx =
      scalar_src(&tmp_src, TGSI_SWIZZLE_X);

   /* DIV tmp.x, 1.0, s0 */
   emit_instruction_op2(emit, VGPU10_OPCODE_DIV, &tmp_dst_x, &one,
                        &inst->Src[0], FALSE);

   /* MOV dst, tmp.xxxx */
   emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &inst->Dst[0],
                        &tmp_src_xxxx, inst->Instruction.Saturate);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_RSQ instruction.
 */
static boolean
emit_rsq(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* dst = RSQ(src):
    *   dst.xyzw = 1 / sqrt(src.x)
    * Translates into:
    *   RSQ tmp, src.x
    *   MOV dst, tmp.xxxx
    */

   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);

   struct tgsi_full_dst_register tmp_dst_x =
      writemask_dst(&tmp_dst, TGSI_WRITEMASK_X);
   struct tgsi_full_src_register tmp_src_xxxx =
      scalar_src(&tmp_src, TGSI_SWIZZLE_X);

   /* RSQ tmp, src.x */
   emit_instruction_op1(emit, VGPU10_OPCODE_RSQ, &tmp_dst_x,
                        &inst->Src[0], FALSE);

   /* MOV dst, tmp.xxxx */
   emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &inst->Dst[0],
                        &tmp_src_xxxx, inst->Instruction.Saturate);

   /* free tmp */
   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_SEQ (Set Equal) instruction.
 */
static boolean
emit_seq(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* dst = SEQ(s0, s1):
    *   dst = s0 == s1 ? 1.0 : 0.0  (per component)
    * Translates into:
    *   EQ tmp, s0, s1;           tmp = s0 == s1 : 0xffffffff : 0 (per comp)
    *   MOVC dst, tmp, 1.0, 0.0;  dst = tmp ? 1.0 : 0.0 (per component)
    */
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register zero = make_immediate_reg_float(emit, 0.0f);
   struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);

   /* EQ tmp, s0, s1 */
   emit_instruction_op2(emit, VGPU10_OPCODE_EQ, &tmp_dst, &inst->Src[0],
                        &inst->Src[1], FALSE);

   /* MOVC dst, tmp, one, zero */
   emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &inst->Dst[0], &tmp_src,
                        &one, &zero, FALSE);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_SGE (Set Greater than or Equal) instruction.
 */
static boolean
emit_sge(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* dst = SGE(s0, s1):
    *   dst = s0 >= s1 ? 1.0 : 0.0  (per component)
    * Translates into:
    *   GE tmp, s0, s1;           tmp = s0 >= s1 : 0xffffffff : 0 (per comp)
    *   MOVC dst, tmp, 1.0, 0.0;  dst = tmp ? 1.0 : 0.0 (per component)
    */
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register zero = make_immediate_reg_float(emit, 0.0f);
   struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);

   /* GE tmp, s0, s1 */
   emit_instruction_op2(emit, VGPU10_OPCODE_GE, &tmp_dst, &inst->Src[0],
                        &inst->Src[1], FALSE);

   /* MOVC dst, tmp, one, zero */
   emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &inst->Dst[0], &tmp_src,
                        &one, &zero, FALSE);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_SGT (Set Greater than) instruction.
 */
static boolean
emit_sgt(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* dst = SGT(s0, s1):
    *   dst = s0 > s1 ? 1.0 : 0.0  (per component)
    * Translates into:
    *   LT tmp, s1, s0;           tmp = s1 < s0 ? 0xffffffff : 0 (per comp)
    *   MOVC dst, tmp, 1.0, 0.0;  dst = tmp ? 1.0 : 0.0 (per component)
    */
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register zero = make_immediate_reg_float(emit, 0.0f);
   struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);

   /* LT tmp, s1, s0 */
   emit_instruction_op2(emit, VGPU10_OPCODE_LT, &tmp_dst, &inst->Src[1],
                        &inst->Src[0], FALSE);

   /* MOVC dst, tmp, one, zero */
   emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &inst->Dst[0], &tmp_src,
                        &one, &zero, FALSE);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_SIN and TGSI_OPCODE_COS instructions.
 */
static boolean
emit_sincos(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);

   struct tgsi_full_src_register tmp_src_xxxx =
      scalar_src(&tmp_src, TGSI_SWIZZLE_X);
   struct tgsi_full_dst_register tmp_dst_x =
      writemask_dst(&tmp_dst, TGSI_WRITEMASK_X);

   begin_emit_instruction(emit);
   emit_opcode(emit, VGPU10_OPCODE_SINCOS, FALSE);

   if(inst->Instruction.Opcode == TGSI_OPCODE_SIN)
   {
      emit_dst_register(emit, &tmp_dst_x);  /* first destination register */
      emit_null_dst_register(emit);  /* second destination register */
   }
   else {
      emit_null_dst_register(emit);
      emit_dst_register(emit, &tmp_dst_x);
   }

   emit_src_register(emit, &inst->Src[0]);
   end_emit_instruction(emit);

   emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &inst->Dst[0],
                        &tmp_src_xxxx, inst->Instruction.Saturate);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_SLE (Set Less than or Equal) instruction.
 */
static boolean
emit_sle(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* dst = SLE(s0, s1):
    *   dst = s0 <= s1 ? 1.0 : 0.0  (per component)
    * Translates into:
    *   GE tmp, s1, s0;           tmp = s1 >= s0 : 0xffffffff : 0 (per comp)
    *   MOVC dst, tmp, 1.0, 0.0;  dst = tmp ? 1.0 : 0.0 (per component)
    */
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register zero = make_immediate_reg_float(emit, 0.0f);
   struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);

   /* GE tmp, s1, s0 */
   emit_instruction_op2(emit, VGPU10_OPCODE_GE, &tmp_dst, &inst->Src[1],
                        &inst->Src[0], FALSE);

   /* MOVC dst, tmp, one, zero */
   emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &inst->Dst[0], &tmp_src,
                        &one, &zero, FALSE);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_SLT (Set Less than) instruction.
 */
static boolean
emit_slt(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* dst = SLT(s0, s1):
    *   dst = s0 < s1 ? 1.0 : 0.0  (per component)
    * Translates into:
    *   LT tmp, s0, s1;           tmp = s0 < s1 ? 0xffffffff : 0 (per comp)
    *   MOVC dst, tmp, 1.0, 0.0;  dst = tmp ? 1.0 : 0.0 (per component)
    */
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register zero = make_immediate_reg_float(emit, 0.0f);
   struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);

   /* LT tmp, s0, s1 */
   emit_instruction_op2(emit, VGPU10_OPCODE_LT, &tmp_dst, &inst->Src[0],
                        &inst->Src[1], FALSE);

   /* MOVC dst, tmp, one, zero */
   emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &inst->Dst[0], &tmp_src,
                        &one, &zero, FALSE);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_SNE (Set Not Equal) instruction.
 */
static boolean
emit_sne(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* dst = SNE(s0, s1):
    *   dst = s0 != s1 ? 1.0 : 0.0  (per component)
    * Translates into:
    *   EQ tmp, s0, s1;           tmp = s0 == s1 : 0xffffffff : 0 (per comp)
    *   MOVC dst, tmp, 1.0, 0.0;  dst = tmp ? 1.0 : 0.0 (per component)
    */
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register zero = make_immediate_reg_float(emit, 0.0f);
   struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);

   /* NE tmp, s0, s1 */
   emit_instruction_op2(emit, VGPU10_OPCODE_NE, &tmp_dst, &inst->Src[0],
                        &inst->Src[1], FALSE);

   /* MOVC dst, tmp, one, zero */
   emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &inst->Dst[0], &tmp_src,
                        &one, &zero, FALSE);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_SSG (Set Sign) instruction.
 */
static boolean
emit_ssg(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   /* dst.x = (src.x > 0.0) ? 1.0 : (src.x < 0.0) ? -1.0 : 0.0
    * dst.y = (src.y > 0.0) ? 1.0 : (src.y < 0.0) ? -1.0 : 0.0
    * dst.z = (src.z > 0.0) ? 1.0 : (src.z < 0.0) ? -1.0 : 0.0
    * dst.w = (src.w > 0.0) ? 1.0 : (src.w < 0.0) ? -1.0 : 0.0
    * Translates into:
    *   LT tmp1, src, zero;           tmp1 = src < zero ? 0xffffffff : 0 (per comp)
    *   MOVC tmp2, tmp1, -1.0, 0.0;   tmp2 = tmp1 ? -1.0 : 0.0 (per component)
    *   LT tmp1, zero, src;           tmp1 = zero < src ? 0xffffffff : 0 (per comp)
    *   MOVC dst, tmp1, 1.0, tmp2;    dst = tmp1 ? 1.0 : tmp2 (per component)
    */
   struct tgsi_full_src_register zero =
      make_immediate_reg_float(emit, 0.0f);
   struct tgsi_full_src_register one =
      make_immediate_reg_float(emit, 1.0f);
   struct tgsi_full_src_register neg_one =
      make_immediate_reg_float(emit, -1.0f);

   unsigned tmp1 = get_temp_index(emit);
   struct tgsi_full_src_register tmp1_src = make_src_temp_reg(tmp1);
   struct tgsi_full_dst_register tmp1_dst = make_dst_temp_reg(tmp1);

   unsigned tmp2 = get_temp_index(emit);
   struct tgsi_full_src_register tmp2_src = make_src_temp_reg(tmp2);
   struct tgsi_full_dst_register tmp2_dst = make_dst_temp_reg(tmp2);

   emit_instruction_op2(emit, VGPU10_OPCODE_LT, &tmp1_dst, &inst->Src[0],
                        &zero, FALSE);
   emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &tmp2_dst, &tmp1_src,
                        &neg_one, &zero, FALSE);
   emit_instruction_op2(emit, VGPU10_OPCODE_LT, &tmp1_dst, &zero,
                        &inst->Src[0], FALSE);
   emit_instruction_op3(emit, VGPU10_OPCODE_MOVC, &inst->Dst[0], &tmp1_src,
                        &one, &tmp2_src, FALSE);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_ISSG (Integer Set Sign) instruction.
 */
static boolean
emit_issg(struct svga_shader_emitter_v10 *emit,
          const struct tgsi_full_instruction *inst)
{
   /* dst.x = (src.x > 0) ? 1 : (src.x < 0) ? -1 : 0
    * dst.y = (src.y > 0) ? 1 : (src.y < 0) ? -1 : 0
    * dst.z = (src.z > 0) ? 1 : (src.z < 0) ? -1 : 0
    * dst.w = (src.w > 0) ? 1 : (src.w < 0) ? -1 : 0
    * Translates into:
    *   ILT tmp1, src, 0              tmp1 = src < 0 ? -1 : 0 (per component)
    *   ILT tmp2, 0, src              tmp2 = 0 < src ? -1 : 0 (per component)
    *   IADD dst, tmp1, neg(tmp2)     dst  = tmp1 - tmp2      (per component)
    */
   struct tgsi_full_src_register zero = make_immediate_reg_float(emit, 0.0f);

   unsigned tmp1 = get_temp_index(emit);
   struct tgsi_full_src_register tmp1_src = make_src_temp_reg(tmp1);
   struct tgsi_full_dst_register tmp1_dst = make_dst_temp_reg(tmp1);

   unsigned tmp2 = get_temp_index(emit);
   struct tgsi_full_src_register tmp2_src = make_src_temp_reg(tmp2);
   struct tgsi_full_dst_register tmp2_dst = make_dst_temp_reg(tmp2);

   struct tgsi_full_src_register neg_tmp2 = negate_src(&tmp2_src);

   emit_instruction_op2(emit, VGPU10_OPCODE_ILT, &tmp1_dst,
                        &inst->Src[0], &zero, FALSE);
   emit_instruction_op2(emit, VGPU10_OPCODE_ILT, &tmp2_dst,
                        &zero, &inst->Src[0], FALSE);
   emit_instruction_op2(emit, VGPU10_OPCODE_IADD, &inst->Dst[0],
                        &tmp1_src, &neg_tmp2, FALSE);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit a comparison instruction.  The dest register will get
 * 0 or ~0 values depending on the outcome of comparing src0 to src1.
 */
static void
emit_comparison(struct svga_shader_emitter_v10 *emit,
                SVGA3dCmpFunc func,
                const struct tgsi_full_dst_register *dst,
                const struct tgsi_full_src_register *src0,
                const struct tgsi_full_src_register *src1)
{
   struct tgsi_full_src_register immediate;
   VGPU10OpcodeToken0 opcode0;
   boolean swapSrc = FALSE;

   /* Sanity checks for svga vs. gallium enums */
   STATIC_ASSERT(SVGA3D_CMP_LESS == (PIPE_FUNC_LESS + 1));
   STATIC_ASSERT(SVGA3D_CMP_GREATEREQUAL == (PIPE_FUNC_GEQUAL + 1));

   opcode0.value = 0;

   switch (func) {
   case SVGA3D_CMP_NEVER:
      immediate = make_immediate_reg_int(emit, 0);
      /* MOV dst, {0} */
      begin_emit_instruction(emit);
      emit_dword(emit, VGPU10_OPCODE_MOV);
      emit_dst_register(emit, dst);
      emit_src_register(emit, &immediate);
      end_emit_instruction(emit);
      return;
   case SVGA3D_CMP_ALWAYS:
      immediate = make_immediate_reg_int(emit, -1);
      /* MOV dst, {-1} */
      begin_emit_instruction(emit);
      emit_dword(emit, VGPU10_OPCODE_MOV);
      emit_dst_register(emit, dst);
      emit_src_register(emit, &immediate);
      end_emit_instruction(emit);
      return;
   case SVGA3D_CMP_LESS:
      opcode0.opcodeType = VGPU10_OPCODE_LT;
      break;
   case SVGA3D_CMP_EQUAL:
      opcode0.opcodeType = VGPU10_OPCODE_EQ;
      break;
   case SVGA3D_CMP_LESSEQUAL:
      opcode0.opcodeType = VGPU10_OPCODE_GE;
      swapSrc = TRUE;
      break;
   case SVGA3D_CMP_GREATER:
      opcode0.opcodeType = VGPU10_OPCODE_LT;
      swapSrc = TRUE;
      break;
   case SVGA3D_CMP_NOTEQUAL:
      opcode0.opcodeType = VGPU10_OPCODE_NE;
      break;
   case SVGA3D_CMP_GREATEREQUAL:
      opcode0.opcodeType = VGPU10_OPCODE_GE;
      break;
   default:
      assert(!"Unexpected comparison mode");
      opcode0.opcodeType = VGPU10_OPCODE_EQ;
   }

   begin_emit_instruction(emit);
   emit_dword(emit, opcode0.value);
   emit_dst_register(emit, dst);
   if (swapSrc) {
      emit_src_register(emit, src1);
      emit_src_register(emit, src0);
   }
   else {
      emit_src_register(emit, src0);
      emit_src_register(emit, src1);
   }
   end_emit_instruction(emit);
}


/**
 * Get texel/address offsets for a texture instruction.
 */
static void
get_texel_offsets(const struct svga_shader_emitter_v10 *emit,
                  const struct tgsi_full_instruction *inst, int offsets[3])
{
   if (inst->Texture.NumOffsets == 1) {
      /* According to OpenGL Shader Language spec the offsets are only
       * fetched from a previously-declared immediate/literal.
       */
      const struct tgsi_texture_offset *off = inst->TexOffsets;
      const unsigned index = off[0].Index;
      const unsigned swizzleX = off[0].SwizzleX;
      const unsigned swizzleY = off[0].SwizzleY;
      const unsigned swizzleZ = off[0].SwizzleZ;
      const union tgsi_immediate_data *imm = emit->immediates[index];

      assert(inst->TexOffsets[0].File == TGSI_FILE_IMMEDIATE);

      offsets[0] = imm[swizzleX].Int;
      offsets[1] = imm[swizzleY].Int;
      offsets[2] = imm[swizzleZ].Int;
   }
   else {
      offsets[0] = offsets[1] = offsets[2] = 0;
   }
}


/**
 * Set up the coordinate register for texture sampling.
 * When we're sampling from a RECT texture we have to scale the
 * unnormalized coordinate to a normalized coordinate.
 * We do that by multiplying the coordinate by an "extra" constant.
 * An alternative would be to use the RESINFO instruction to query the
 * texture's size.
 */
static struct tgsi_full_src_register
setup_texcoord(struct svga_shader_emitter_v10 *emit,
               unsigned unit,
               const struct tgsi_full_src_register *coord)
{
   if (emit->sampler_view[unit] && emit->key.tex[unit].unnormalized) {
      unsigned scale_index = emit->texcoord_scale_index[unit];
      unsigned tmp = get_temp_index(emit);
      struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
      struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
      struct tgsi_full_src_register scale_src = make_src_const_reg(scale_index);

      if (emit->key.tex[unit].texel_bias) {
         /* to fix texture coordinate rounding issue, 0.0001 offset is
          * been added. This fixes piglit test fbo-blit-scaled-linear. */
         struct tgsi_full_src_register offset =
            make_immediate_reg_float(emit, 0.0001f);

         /* ADD tmp, coord, offset */
         emit_instruction_op2(emit, VGPU10_OPCODE_ADD, &tmp_dst,
                              coord, &offset, FALSE);
         /* MUL tmp, tmp, scale */
         emit_instruction_op2(emit, VGPU10_OPCODE_MUL, &tmp_dst,
                              &tmp_src, &scale_src, FALSE);
      }
      else {
         /* MUL tmp, coord, const[] */
         emit_instruction_op2(emit, VGPU10_OPCODE_MUL, &tmp_dst,
                              coord, &scale_src, FALSE);
      }
      return tmp_src;
   }
   else {
      /* use texcoord as-is */
      return *coord;
   }
}


/**
 * For SAMPLE_C instructions, emit the extra src register which indicates
 * the reference/comparision value.
 */
static void
emit_tex_compare_refcoord(struct svga_shader_emitter_v10 *emit,
                          enum tgsi_texture_type target,
                          const struct tgsi_full_src_register *coord)
{
   struct tgsi_full_src_register coord_src_ref;
   int component;

   assert(tgsi_is_shadow_target(target));

   component = tgsi_util_get_shadow_ref_src_index(target) % 4;
   assert(component >= 0);

   coord_src_ref = scalar_src(coord, component);

   emit_src_register(emit, &coord_src_ref);
}


/**
 * Info for implementing texture swizzles.
 * The begin_tex_swizzle(), get_tex_swizzle_dst() and end_tex_swizzle()
 * functions use this to encapsulate the extra steps needed to perform
 * a texture swizzle, or shadow/depth comparisons.
 * The shadow/depth comparison is only done here if for the cases where
 * there's no VGPU10 opcode (like texture bias lookup w/ shadow compare).
 */
struct tex_swizzle_info
{
   boolean swizzled;
   boolean shadow_compare;
   unsigned unit;
   enum tgsi_texture_type texture_target;  /**< TGSI_TEXTURE_x */
   struct tgsi_full_src_register tmp_src;
   struct tgsi_full_dst_register tmp_dst;
   const struct tgsi_full_dst_register *inst_dst;
   const struct tgsi_full_src_register *coord_src;
};


/**
 * Do setup for handling texture swizzles or shadow compares.
 * \param unit  the texture unit
 * \param inst  the TGSI texture instruction
 * \param shadow_compare  do shadow/depth comparison?
 * \param swz  returns the swizzle info
 */
static void
begin_tex_swizzle(struct svga_shader_emitter_v10 *emit,
                  unsigned unit,
                  const struct tgsi_full_instruction *inst,
                  boolean shadow_compare,
                  struct tex_swizzle_info *swz)
{
   swz->swizzled = (emit->key.tex[unit].swizzle_r != TGSI_SWIZZLE_X ||
                    emit->key.tex[unit].swizzle_g != TGSI_SWIZZLE_Y ||
                    emit->key.tex[unit].swizzle_b != TGSI_SWIZZLE_Z ||
                    emit->key.tex[unit].swizzle_a != TGSI_SWIZZLE_W);

   swz->shadow_compare = shadow_compare;
   swz->texture_target = inst->Texture.Texture;

   if (swz->swizzled || shadow_compare) {
      /* Allocate temp register for the result of the SAMPLE instruction
       * and the source of the MOV/compare/swizzle instructions.
       */
      unsigned tmp = get_temp_index(emit);
      swz->tmp_src = make_src_temp_reg(tmp);
      swz->tmp_dst = make_dst_temp_reg(tmp);

      swz->unit = unit;
   }
   swz->inst_dst = &inst->Dst[0];
   swz->coord_src = &inst->Src[0];

   emit->fs.shadow_compare_units |= shadow_compare << unit;
}


/**
 * Returns the register to put the SAMPLE instruction results into.
 * This will either be the original instruction dst reg (if no swizzle
 * and no shadow comparison) or a temporary reg if there is a swizzle.
 */
static const struct tgsi_full_dst_register *
get_tex_swizzle_dst(const struct tex_swizzle_info *swz)
{
   return (swz->swizzled || swz->shadow_compare)
      ? &swz->tmp_dst : swz->inst_dst;
}


/**
 * This emits the MOV instruction that actually implements a texture swizzle
 * and/or shadow comparison.
 */
static void
end_tex_swizzle(struct svga_shader_emitter_v10 *emit,
                const struct tex_swizzle_info *swz)
{
   if (swz->shadow_compare) {
      /* Emit extra instructions to compare the fetched texel value against
       * a texture coordinate component.  The result of the comparison
       * is 0.0 or 1.0.
       */
      struct tgsi_full_src_register coord_src;
      struct tgsi_full_src_register texel_src =
         scalar_src(&swz->tmp_src, TGSI_SWIZZLE_X);
      struct tgsi_full_src_register one =
         make_immediate_reg_float(emit, 1.0f);
      /* convert gallium comparison func to SVGA comparison func */
      SVGA3dCmpFunc compare_func = emit->key.tex[swz->unit].compare_func + 1;

      assert(emit->unit == PIPE_SHADER_FRAGMENT);

      int component =
         tgsi_util_get_shadow_ref_src_index(swz->texture_target) % 4;
      assert(component >= 0);
      coord_src = scalar_src(swz->coord_src, component);

      /* COMPARE tmp, coord, texel */
      emit_comparison(emit, compare_func,
                      &swz->tmp_dst, &coord_src, &texel_src);

      /* AND dest, tmp, {1.0} */
      begin_emit_instruction(emit);
      emit_opcode(emit, VGPU10_OPCODE_AND, FALSE);
      if (swz->swizzled) {
         emit_dst_register(emit, &swz->tmp_dst);
      }
      else {
         emit_dst_register(emit, swz->inst_dst);
      }
      emit_src_register(emit, &swz->tmp_src);
      emit_src_register(emit, &one);
      end_emit_instruction(emit);
   }

   if (swz->swizzled) {
      unsigned swz_r = emit->key.tex[swz->unit].swizzle_r;
      unsigned swz_g = emit->key.tex[swz->unit].swizzle_g;
      unsigned swz_b = emit->key.tex[swz->unit].swizzle_b;
      unsigned swz_a = emit->key.tex[swz->unit].swizzle_a;
      unsigned writemask_0 = 0, writemask_1 = 0;
      boolean int_tex = is_integer_type(emit->sampler_return_type[swz->unit]);

      /* Swizzle w/out zero/one terms */
      struct tgsi_full_src_register src_swizzled =
         swizzle_src(&swz->tmp_src,
                     swz_r < PIPE_SWIZZLE_0 ? swz_r : PIPE_SWIZZLE_X,
                     swz_g < PIPE_SWIZZLE_0 ? swz_g : PIPE_SWIZZLE_Y,
                     swz_b < PIPE_SWIZZLE_0 ? swz_b : PIPE_SWIZZLE_Z,
                     swz_a < PIPE_SWIZZLE_0 ? swz_a : PIPE_SWIZZLE_W);

      /* MOV dst, color(tmp).<swizzle> */
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV,
                           swz->inst_dst, &src_swizzled, FALSE);

      /* handle swizzle zero terms */
      writemask_0 = (((swz_r == PIPE_SWIZZLE_0) << 0) |
                     ((swz_g == PIPE_SWIZZLE_0) << 1) |
                     ((swz_b == PIPE_SWIZZLE_0) << 2) |
                     ((swz_a == PIPE_SWIZZLE_0) << 3));
      writemask_0 &= swz->inst_dst->Register.WriteMask;

      if (writemask_0) {
         struct tgsi_full_src_register zero = int_tex ?
            make_immediate_reg_int(emit, 0) :
            make_immediate_reg_float(emit, 0.0f);
         struct tgsi_full_dst_register dst =
            writemask_dst(swz->inst_dst, writemask_0);

         /* MOV dst.writemask_0, {0,0,0,0} */
         emit_instruction_op1(emit, VGPU10_OPCODE_MOV,
                              &dst, &zero, FALSE);
      }

      /* handle swizzle one terms */
      writemask_1 = (((swz_r == PIPE_SWIZZLE_1) << 0) |
                     ((swz_g == PIPE_SWIZZLE_1) << 1) |
                     ((swz_b == PIPE_SWIZZLE_1) << 2) |
                     ((swz_a == PIPE_SWIZZLE_1) << 3));
      writemask_1 &= swz->inst_dst->Register.WriteMask;

      if (writemask_1) {
         struct tgsi_full_src_register one = int_tex ?
            make_immediate_reg_int(emit, 1) :
            make_immediate_reg_float(emit, 1.0f);
         struct tgsi_full_dst_register dst =
            writemask_dst(swz->inst_dst, writemask_1);

         /* MOV dst.writemask_1, {1,1,1,1} */
         emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &dst, &one, FALSE);
      }
   }
}


/**
 * Emit code for TGSI_OPCODE_SAMPLE instruction.
 */
static boolean
emit_sample(struct svga_shader_emitter_v10 *emit,
            const struct tgsi_full_instruction *inst)
{
   const unsigned resource_unit = inst->Src[1].Register.Index;
   const unsigned sampler_unit = inst->Src[2].Register.Index;
   struct tgsi_full_src_register coord;
   int offsets[3];
   struct tex_swizzle_info swz_info;

   begin_tex_swizzle(emit, sampler_unit, inst, FALSE, &swz_info);

   get_texel_offsets(emit, inst, offsets);

   coord = setup_texcoord(emit, resource_unit, &inst->Src[0]);

   /* SAMPLE dst, coord(s0), resource, sampler */
   begin_emit_instruction(emit);

   /* NOTE: for non-fragment shaders, we should use VGPU10_OPCODE_SAMPLE_L
    * with LOD=0.  But our virtual GPU accepts this as-is.
    */
   emit_sample_opcode(emit, VGPU10_OPCODE_SAMPLE,
                      inst->Instruction.Saturate, offsets);
   emit_dst_register(emit, get_tex_swizzle_dst(&swz_info));
   emit_src_register(emit, &coord);
   emit_resource_register(emit, resource_unit);
   emit_sampler_register(emit, sampler_unit);
   end_emit_instruction(emit);

   end_tex_swizzle(emit, &swz_info);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Check if a texture instruction is valid.
 * An example of an invalid texture instruction is doing shadow comparison
 * with an integer-valued texture.
 * If we detect an invalid texture instruction, we replace it with:
 *   MOV dst, {1,1,1,1};
 * \return TRUE if valid, FALSE if invalid.
 */
static boolean
is_valid_tex_instruction(struct svga_shader_emitter_v10 *emit,
                         const struct tgsi_full_instruction *inst)
{
   const unsigned unit = inst->Src[1].Register.Index;
   const enum tgsi_texture_type target = inst->Texture.Texture;
   boolean valid = TRUE;

   if (tgsi_is_shadow_target(target) &&
       is_integer_type(emit->sampler_return_type[unit])) {
      debug_printf("Invalid SAMPLE_C with an integer texture!\n");
      valid = FALSE;
   }
   /* XXX might check for other conditions in the future here */

   if (!valid) {
      /* emit a MOV dst, {1,1,1,1} instruction. */
      struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);
      begin_emit_instruction(emit);
      emit_opcode(emit, VGPU10_OPCODE_MOV, FALSE);
      emit_dst_register(emit, &inst->Dst[0]);
      emit_src_register(emit, &one);
      end_emit_instruction(emit);
   }

   return valid;
}


/**
 * Emit code for TGSI_OPCODE_TEX (simple texture lookup)
 */
static boolean
emit_tex(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   const uint unit = inst->Src[1].Register.Index;
   const enum tgsi_texture_type target = inst->Texture.Texture;
   VGPU10_OPCODE_TYPE opcode;
   struct tgsi_full_src_register coord;
   int offsets[3];
   struct tex_swizzle_info swz_info;

   /* check that the sampler returns a float */
   if (!is_valid_tex_instruction(emit, inst))
      return TRUE;

   begin_tex_swizzle(emit, unit, inst, FALSE, &swz_info);

   get_texel_offsets(emit, inst, offsets);

   coord = setup_texcoord(emit, unit, &inst->Src[0]);

   /* SAMPLE dst, coord(s0), resource, sampler */
   begin_emit_instruction(emit);

   if (tgsi_is_shadow_target(target))
      opcode = VGPU10_OPCODE_SAMPLE_C;
   else
      opcode = VGPU10_OPCODE_SAMPLE;

   emit_sample_opcode(emit, opcode, inst->Instruction.Saturate, offsets);
   emit_dst_register(emit, get_tex_swizzle_dst(&swz_info));
   emit_src_register(emit, &coord);
   emit_resource_register(emit, unit);
   emit_sampler_register(emit, unit);
   if (opcode == VGPU10_OPCODE_SAMPLE_C) {
      emit_tex_compare_refcoord(emit, target, &coord);
   }
   end_emit_instruction(emit);

   end_tex_swizzle(emit, &swz_info);

   free_temp_indexes(emit);

   return TRUE;
}

/**
 * Emit code for TGSI_OPCODE_TG4 (texture lookup for texture gather)
 */
static boolean
emit_tg4(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   const uint unit = inst->Src[2].Register.Index;
   struct tgsi_full_src_register src;
   int offsets[3];

   /* check that the sampler returns a float */
   if (!is_valid_tex_instruction(emit, inst))
      return TRUE;

   /* Only a single channel is supported in SM4_1 and we report
    * PIPE_CAP_MAX_TEXTURE_GATHER_COMPONENTS = 1.
    * Only the 0th component will be gathered.
    */
   switch (emit->key.tex[unit].swizzle_r) {
   case PIPE_SWIZZLE_X:
      get_texel_offsets(emit, inst, offsets);
      src = setup_texcoord(emit, unit, &inst->Src[0]);

      /* Gather dst, coord, resource, sampler */
      begin_emit_instruction(emit);
      emit_sample_opcode(emit, VGPU10_OPCODE_GATHER4,
                         inst->Instruction.Saturate, offsets);
      emit_dst_register(emit, &inst->Dst[0]);
      emit_src_register(emit, &src);
      emit_resource_register(emit, unit);
      emit_sampler_register(emit, unit);
      end_emit_instruction(emit);
      break;
   case PIPE_SWIZZLE_W:
   case PIPE_SWIZZLE_1:
      src = make_immediate_reg_float(emit, 1.0);
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV,
                           &inst->Dst[0], &src, FALSE);
      break;
   case PIPE_SWIZZLE_Y:
   case PIPE_SWIZZLE_Z:
   case PIPE_SWIZZLE_0:
   default:
      src = make_immediate_reg_float(emit, 0.0);
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV,
                           &inst->Dst[0], &src, FALSE);
      break;
   }

   return TRUE;
}



/**
 * Emit code for TGSI_OPCODE_TEX2 (texture lookup for shadow cube map arrays)
 */
static boolean
emit_tex2(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   const uint unit = inst->Src[2].Register.Index;
   unsigned target = inst->Texture.Texture;
   struct tgsi_full_src_register coord, ref;
   int offsets[3];
   struct tex_swizzle_info swz_info;

   /* check that the sampler returns a float */
   if (!is_valid_tex_instruction(emit, inst))
      return TRUE;

   begin_tex_swizzle(emit, unit, inst, FALSE, &swz_info);

   get_texel_offsets(emit, inst, offsets);

   coord = setup_texcoord(emit, unit, &inst->Src[0]);
   ref = scalar_src(&inst->Src[1], TGSI_SWIZZLE_X);

   /* SAMPLE_C dst, coord, resource, sampler, ref */
   begin_emit_instruction(emit);
   emit_sample_opcode(emit, VGPU10_OPCODE_SAMPLE_C,
                      inst->Instruction.Saturate, offsets);
   emit_dst_register(emit, get_tex_swizzle_dst(&swz_info));
   emit_src_register(emit, &coord);
   emit_resource_register(emit, unit);
   emit_sampler_register(emit, unit);
   emit_tex_compare_refcoord(emit, target, &ref);
   end_emit_instruction(emit);

   end_tex_swizzle(emit, &swz_info);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_TXP (projective texture)
 */
static boolean
emit_txp(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   const uint unit = inst->Src[1].Register.Index;
   const enum tgsi_texture_type target = inst->Texture.Texture;
   VGPU10_OPCODE_TYPE opcode;
   int offsets[3];
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register src0_wwww =
      scalar_src(&inst->Src[0], TGSI_SWIZZLE_W);
   struct tgsi_full_src_register coord;
   struct tex_swizzle_info swz_info;

   /* check that the sampler returns a float */
   if (!is_valid_tex_instruction(emit, inst))
      return TRUE;

   begin_tex_swizzle(emit, unit, inst, FALSE, &swz_info);

   get_texel_offsets(emit, inst, offsets);

   coord = setup_texcoord(emit, unit, &inst->Src[0]);

   /* DIV tmp, coord, coord.wwww */
   emit_instruction_op2(emit, VGPU10_OPCODE_DIV, &tmp_dst,
                        &coord, &src0_wwww, FALSE);

   /* SAMPLE dst, coord(tmp), resource, sampler */
   begin_emit_instruction(emit);

   if (tgsi_is_shadow_target(target))
      /* NOTE: for non-fragment shaders, we should use
       * VGPU10_OPCODE_SAMPLE_C_LZ, but our virtual GPU accepts this as-is.
       */
      opcode = VGPU10_OPCODE_SAMPLE_C;
   else
      opcode = VGPU10_OPCODE_SAMPLE;

   emit_sample_opcode(emit, opcode, inst->Instruction.Saturate, offsets);
   emit_dst_register(emit, get_tex_swizzle_dst(&swz_info));
   emit_src_register(emit, &tmp_src);  /* projected coord */
   emit_resource_register(emit, unit);
   emit_sampler_register(emit, unit);
   if (opcode == VGPU10_OPCODE_SAMPLE_C) {
      emit_tex_compare_refcoord(emit, target, &tmp_src);
   }
   end_emit_instruction(emit);

   end_tex_swizzle(emit, &swz_info);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_TXD (explicit derivatives)
 */
static boolean
emit_txd(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   const uint unit = inst->Src[3].Register.Index;
   const enum tgsi_texture_type target = inst->Texture.Texture;
   int offsets[3];
   struct tgsi_full_src_register coord;
   struct tex_swizzle_info swz_info;

   begin_tex_swizzle(emit, unit, inst, tgsi_is_shadow_target(target),
                     &swz_info);

   get_texel_offsets(emit, inst, offsets);

   coord = setup_texcoord(emit, unit, &inst->Src[0]);

   /* SAMPLE_D dst, coord(s0), resource, sampler, Xderiv(s1), Yderiv(s2) */
   begin_emit_instruction(emit);
   emit_sample_opcode(emit, VGPU10_OPCODE_SAMPLE_D,
                      inst->Instruction.Saturate, offsets);
   emit_dst_register(emit, get_tex_swizzle_dst(&swz_info));
   emit_src_register(emit, &coord);
   emit_resource_register(emit, unit);
   emit_sampler_register(emit, unit);
   emit_src_register(emit, &inst->Src[1]);  /* Xderiv */
   emit_src_register(emit, &inst->Src[2]);  /* Yderiv */
   end_emit_instruction(emit);

   end_tex_swizzle(emit, &swz_info);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_TXF (texel fetch)
 */
static boolean
emit_txf(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   const uint unit = inst->Src[1].Register.Index;
   const boolean msaa = tgsi_is_msaa_target(inst->Texture.Texture)
      && emit->key.tex[unit].num_samples > 1;
   int offsets[3];
   struct tex_swizzle_info swz_info;

   begin_tex_swizzle(emit, unit, inst, FALSE, &swz_info);

   get_texel_offsets(emit, inst, offsets);

   if (msaa) {
      assert(emit->key.tex[unit].num_samples > 1);

      /* Fetch one sample from an MSAA texture */
      struct tgsi_full_src_register sampleIndex =
         scalar_src(&inst->Src[0], TGSI_SWIZZLE_W);
      /* LD_MS dst, coord(s0), resource, sampleIndex */
      begin_emit_instruction(emit);
      emit_sample_opcode(emit, VGPU10_OPCODE_LD_MS,
                         inst->Instruction.Saturate, offsets);
      emit_dst_register(emit, get_tex_swizzle_dst(&swz_info));
      emit_src_register(emit, &inst->Src[0]);
      emit_resource_register(emit, unit);
      emit_src_register(emit, &sampleIndex);
      end_emit_instruction(emit);
   }
   else {
      /* Fetch one texel specified by integer coordinate */
      /* LD dst, coord(s0), resource */
      begin_emit_instruction(emit);
      emit_sample_opcode(emit, VGPU10_OPCODE_LD,
                         inst->Instruction.Saturate, offsets);
      emit_dst_register(emit, get_tex_swizzle_dst(&swz_info));
      emit_src_register(emit, &inst->Src[0]);
      emit_resource_register(emit, unit);
      end_emit_instruction(emit);
   }

   end_tex_swizzle(emit, &swz_info);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_TXL (explicit LOD) or TGSI_OPCODE_TXB (LOD bias)
 * or TGSI_OPCODE_TXB2 (for cube shadow maps).
 */
static boolean
emit_txl_txb(struct svga_shader_emitter_v10 *emit,
             const struct tgsi_full_instruction *inst)
{
   const enum tgsi_texture_type target = inst->Texture.Texture;
   VGPU10_OPCODE_TYPE opcode;
   unsigned unit;
   int offsets[3];
   struct tgsi_full_src_register coord, lod_bias;
   struct tex_swizzle_info swz_info;

   assert(inst->Instruction.Opcode == TGSI_OPCODE_TXL ||
          inst->Instruction.Opcode == TGSI_OPCODE_TXB ||
          inst->Instruction.Opcode == TGSI_OPCODE_TXB2);

   if (inst->Instruction.Opcode == TGSI_OPCODE_TXB2) {
      lod_bias = scalar_src(&inst->Src[1], TGSI_SWIZZLE_X);
      unit = inst->Src[2].Register.Index;
   }
   else {
      lod_bias = scalar_src(&inst->Src[0], TGSI_SWIZZLE_W);
      unit = inst->Src[1].Register.Index;
   }

   begin_tex_swizzle(emit, unit, inst, tgsi_is_shadow_target(target),
                     &swz_info);

   get_texel_offsets(emit, inst, offsets);

   coord = setup_texcoord(emit, unit, &inst->Src[0]);

   /* SAMPLE_L/B dst, coord(s0), resource, sampler, lod(s3) */
   begin_emit_instruction(emit);
   if (inst->Instruction.Opcode == TGSI_OPCODE_TXL) {
      opcode = VGPU10_OPCODE_SAMPLE_L;
   }
   else {
      opcode = VGPU10_OPCODE_SAMPLE_B;
   }
   emit_sample_opcode(emit, opcode, inst->Instruction.Saturate, offsets);
   emit_dst_register(emit, get_tex_swizzle_dst(&swz_info));
   emit_src_register(emit, &coord);
   emit_resource_register(emit, unit);
   emit_sampler_register(emit, unit);
   emit_src_register(emit, &lod_bias);
   end_emit_instruction(emit);

   end_tex_swizzle(emit, &swz_info);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_TXL2 (explicit LOD) for cubemap array.
 */
static boolean
emit_txl2(struct svga_shader_emitter_v10 *emit,
          const struct tgsi_full_instruction *inst)
{
   unsigned target = inst->Texture.Texture;
   unsigned opcode, unit;
   int offsets[3];
   struct tgsi_full_src_register coord, lod;
   struct tex_swizzle_info swz_info;

   assert(inst->Instruction.Opcode == TGSI_OPCODE_TXL2);

   lod = scalar_src(&inst->Src[1], TGSI_SWIZZLE_X);
   unit = inst->Src[2].Register.Index;

   begin_tex_swizzle(emit, unit, inst, tgsi_is_shadow_target(target),
                     &swz_info);

   get_texel_offsets(emit, inst, offsets);

   coord = setup_texcoord(emit, unit, &inst->Src[0]);

   /* SAMPLE_L dst, coord(s0), resource, sampler, lod(s3) */
   begin_emit_instruction(emit);
   opcode = VGPU10_OPCODE_SAMPLE_L;
   emit_sample_opcode(emit, opcode, inst->Instruction.Saturate, offsets);
   emit_dst_register(emit, get_tex_swizzle_dst(&swz_info));
   emit_src_register(emit, &coord);
   emit_resource_register(emit, unit);
   emit_sampler_register(emit, unit);
   emit_src_register(emit, &lod);
   end_emit_instruction(emit);

   end_tex_swizzle(emit, &swz_info);

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit code for TGSI_OPCODE_TXQ (texture query) instruction.
 */
static boolean
emit_txq(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   const uint unit = inst->Src[1].Register.Index;

   if (emit->sampler_target[unit] == TGSI_TEXTURE_BUFFER) {
      /* RESINFO does not support querying texture buffers, so we instead
       * store texture buffer sizes in shader constants, then copy them to
       * implement TXQ instead of emitting RESINFO.
       * MOV dst, const[texture_buffer_size_index[unit]]
       */
      struct tgsi_full_src_register size_src =
         make_src_const_reg(emit->texture_buffer_size_index[unit]);
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &inst->Dst[0], &size_src,
                           FALSE);
   } else {
      /* RESINFO dst, srcMipLevel, resource */
      begin_emit_instruction(emit);
      emit_opcode_resinfo(emit, VGPU10_RESINFO_RETURN_UINT);
      emit_dst_register(emit, &inst->Dst[0]);
      emit_src_register(emit, &inst->Src[0]);
      emit_resource_register(emit, unit);
      end_emit_instruction(emit);
   }

   free_temp_indexes(emit);

   return TRUE;
}


/**
 * Emit a simple instruction (like ADD, MUL, MIN, etc).
 */
static boolean
emit_simple(struct svga_shader_emitter_v10 *emit,
            const struct tgsi_full_instruction *inst)
{
   const enum tgsi_opcode opcode = inst->Instruction.Opcode;
   const struct tgsi_opcode_info *op = tgsi_get_opcode_info(opcode);
   unsigned i;

   begin_emit_instruction(emit);
   emit_opcode(emit, translate_opcode(opcode), inst->Instruction.Saturate);
   for (i = 0; i < op->num_dst; i++) {
      emit_dst_register(emit, &inst->Dst[i]);
   }
   for (i = 0; i < op->num_src; i++) {
      emit_src_register(emit, &inst->Src[i]);
   }
   end_emit_instruction(emit);

   return TRUE;
}


/**
 * We only special case the MOV instruction to try to detect constant
 * color writes in the fragment shader.
 */
static boolean
emit_mov(struct svga_shader_emitter_v10 *emit,
         const struct tgsi_full_instruction *inst)
{
   const struct tgsi_full_src_register *src = &inst->Src[0];
   const struct tgsi_full_dst_register *dst = &inst->Dst[0];

   if (emit->unit == PIPE_SHADER_FRAGMENT &&
       dst->Register.File == TGSI_FILE_OUTPUT &&
       dst->Register.Index == 0 &&
       src->Register.File == TGSI_FILE_CONSTANT &&
       !src->Register.Indirect) {
      emit->constant_color_output = TRUE;
   }

   return emit_simple(emit, inst);
}


/**
 * Emit a simple VGPU10 instruction which writes to multiple dest registers,
 * where TGSI only uses one dest register.
 */
static boolean
emit_simple_1dst(struct svga_shader_emitter_v10 *emit,
                 const struct tgsi_full_instruction *inst,
                 unsigned dst_count,
                 unsigned dst_index)
{
   const enum tgsi_opcode opcode = inst->Instruction.Opcode;
   const struct tgsi_opcode_info *op = tgsi_get_opcode_info(opcode);
   unsigned i;

   begin_emit_instruction(emit);
   emit_opcode(emit, translate_opcode(opcode), inst->Instruction.Saturate);

   for (i = 0; i < dst_count; i++) {
      if (i == dst_index) {
         emit_dst_register(emit, &inst->Dst[0]);
      } else {
         emit_null_dst_register(emit);
      }
   }

   for (i = 0; i < op->num_src; i++) {
      emit_src_register(emit, &inst->Src[i]);
   }
   end_emit_instruction(emit);

   return TRUE;
}


/**
 * Translate a single TGSI instruction to VGPU10.
 */
static boolean
emit_vgpu10_instruction(struct svga_shader_emitter_v10 *emit,
                        unsigned inst_number,
                        const struct tgsi_full_instruction *inst)
{
   const enum tgsi_opcode opcode = inst->Instruction.Opcode;

   switch (opcode) {
   case TGSI_OPCODE_ADD:
   case TGSI_OPCODE_AND:
   case TGSI_OPCODE_BGNLOOP:
   case TGSI_OPCODE_BRK:
   case TGSI_OPCODE_CEIL:
   case TGSI_OPCODE_CONT:
   case TGSI_OPCODE_DDX:
   case TGSI_OPCODE_DDY:
   case TGSI_OPCODE_DIV:
   case TGSI_OPCODE_DP2:
   case TGSI_OPCODE_DP3:
   case TGSI_OPCODE_DP4:
   case TGSI_OPCODE_ELSE:
   case TGSI_OPCODE_ENDIF:
   case TGSI_OPCODE_ENDLOOP:
   case TGSI_OPCODE_ENDSUB:
   case TGSI_OPCODE_F2I:
   case TGSI_OPCODE_F2U:
   case TGSI_OPCODE_FLR:
   case TGSI_OPCODE_FRC:
   case TGSI_OPCODE_FSEQ:
   case TGSI_OPCODE_FSGE:
   case TGSI_OPCODE_FSLT:
   case TGSI_OPCODE_FSNE:
   case TGSI_OPCODE_I2F:
   case TGSI_OPCODE_IMAX:
   case TGSI_OPCODE_IMIN:
   case TGSI_OPCODE_INEG:
   case TGSI_OPCODE_ISGE:
   case TGSI_OPCODE_ISHR:
   case TGSI_OPCODE_ISLT:
   case TGSI_OPCODE_MAD:
   case TGSI_OPCODE_MAX:
   case TGSI_OPCODE_MIN:
   case TGSI_OPCODE_MUL:
   case TGSI_OPCODE_NOP:
   case TGSI_OPCODE_NOT:
   case TGSI_OPCODE_OR:
   case TGSI_OPCODE_RET:
   case TGSI_OPCODE_UADD:
   case TGSI_OPCODE_USEQ:
   case TGSI_OPCODE_USGE:
   case TGSI_OPCODE_USLT:
   case TGSI_OPCODE_UMIN:
   case TGSI_OPCODE_UMAD:
   case TGSI_OPCODE_UMAX:
   case TGSI_OPCODE_ROUND:
   case TGSI_OPCODE_SQRT:
   case TGSI_OPCODE_SHL:
   case TGSI_OPCODE_TRUNC:
   case TGSI_OPCODE_U2F:
   case TGSI_OPCODE_UCMP:
   case TGSI_OPCODE_USHR:
   case TGSI_OPCODE_USNE:
   case TGSI_OPCODE_XOR:
      /* simple instructions */
      return emit_simple(emit, inst);

   case TGSI_OPCODE_MOV:
      return emit_mov(emit, inst);
   case TGSI_OPCODE_EMIT:
      return emit_vertex(emit, inst);
   case TGSI_OPCODE_ENDPRIM:
      return emit_endprim(emit, inst);
   case TGSI_OPCODE_IABS:
      return emit_iabs(emit, inst);
   case TGSI_OPCODE_ARL:
      /* fall-through */
   case TGSI_OPCODE_UARL:
      return emit_arl_uarl(emit, inst);
   case TGSI_OPCODE_BGNSUB:
      /* no-op */
      return TRUE;
   case TGSI_OPCODE_CAL:
      return emit_cal(emit, inst);
   case TGSI_OPCODE_CMP:
      return emit_cmp(emit, inst);
   case TGSI_OPCODE_COS:
      return emit_sincos(emit, inst);
   case TGSI_OPCODE_DST:
      return emit_dst(emit, inst);
   case TGSI_OPCODE_EX2:
      return emit_ex2(emit, inst);
   case TGSI_OPCODE_EXP:
      return emit_exp(emit, inst);
   case TGSI_OPCODE_IF:
      return emit_if(emit, inst);
   case TGSI_OPCODE_KILL:
      return emit_kill(emit, inst);
   case TGSI_OPCODE_KILL_IF:
      return emit_kill_if(emit, inst);
   case TGSI_OPCODE_LG2:
      return emit_lg2(emit, inst);
   case TGSI_OPCODE_LIT:
      return emit_lit(emit, inst);
   case TGSI_OPCODE_LODQ:
      return emit_lodq(emit, inst);
   case TGSI_OPCODE_LOG:
      return emit_log(emit, inst);
   case TGSI_OPCODE_LRP:
      return emit_lrp(emit, inst);
   case TGSI_OPCODE_POW:
      return emit_pow(emit, inst);
   case TGSI_OPCODE_RCP:
      return emit_rcp(emit, inst);
   case TGSI_OPCODE_RSQ:
      return emit_rsq(emit, inst);
   case TGSI_OPCODE_SAMPLE:
      return emit_sample(emit, inst);
   case TGSI_OPCODE_SEQ:
      return emit_seq(emit, inst);
   case TGSI_OPCODE_SGE:
      return emit_sge(emit, inst);
   case TGSI_OPCODE_SGT:
      return emit_sgt(emit, inst);
   case TGSI_OPCODE_SIN:
      return emit_sincos(emit, inst);
   case TGSI_OPCODE_SLE:
      return emit_sle(emit, inst);
   case TGSI_OPCODE_SLT:
      return emit_slt(emit, inst);
   case TGSI_OPCODE_SNE:
      return emit_sne(emit, inst);
   case TGSI_OPCODE_SSG:
      return emit_ssg(emit, inst);
   case TGSI_OPCODE_ISSG:
      return emit_issg(emit, inst);
   case TGSI_OPCODE_TEX:
      return emit_tex(emit, inst);
   case TGSI_OPCODE_TG4:
      return emit_tg4(emit, inst);
   case TGSI_OPCODE_TEX2:
      return emit_tex2(emit, inst);
   case TGSI_OPCODE_TXP:
      return emit_txp(emit, inst);
   case TGSI_OPCODE_TXB:
   case TGSI_OPCODE_TXB2:
   case TGSI_OPCODE_TXL:
      return emit_txl_txb(emit, inst);
   case TGSI_OPCODE_TXD:
      return emit_txd(emit, inst);
   case TGSI_OPCODE_TXF:
      return emit_txf(emit, inst);
   case TGSI_OPCODE_TXL2:
      return emit_txl2(emit, inst);
   case TGSI_OPCODE_TXQ:
      return emit_txq(emit, inst);
   case TGSI_OPCODE_UIF:
      return emit_if(emit, inst);
   case TGSI_OPCODE_UMUL_HI:
   case TGSI_OPCODE_IMUL_HI:
   case TGSI_OPCODE_UDIV:
   case TGSI_OPCODE_IDIV:
      /* These cases use only the FIRST of two destination registers */
      return emit_simple_1dst(emit, inst, 2, 0);
   case TGSI_OPCODE_UMUL:
   case TGSI_OPCODE_UMOD:
   case TGSI_OPCODE_MOD:
      /* These cases use only the SECOND of two destination registers */
      return emit_simple_1dst(emit, inst, 2, 1);
   case TGSI_OPCODE_END:
      if (!emit_post_helpers(emit))
         return FALSE;
      return emit_simple(emit, inst);

   default:
      debug_printf("Unimplemented tgsi instruction %s\n",
                   tgsi_get_opcode_name(opcode));
      return FALSE;
   }

   return TRUE;
}


/**
 * Emit the extra instructions to adjust the vertex position.
 * There are two possible adjustments:
 * 1. Converting from Gallium to VGPU10 coordinate space by applying the
 *    "prescale" and "pretranslate" values.
 * 2. Undoing the viewport transformation when we use the swtnl/draw path.
 * \param vs_pos_tmp_index  which temporary register contains the vertex pos.
 */
static void
emit_vpos_instructions(struct svga_shader_emitter_v10 *emit,
                       unsigned vs_pos_tmp_index)
{
   struct tgsi_full_src_register tmp_pos_src;
   struct tgsi_full_dst_register pos_dst;

   /* Don't bother to emit any extra vertex instructions if vertex position is
    * not written out
    */
   if (emit->vposition.out_index == INVALID_INDEX)
      return;

   tmp_pos_src = make_src_temp_reg(vs_pos_tmp_index);
   pos_dst = make_dst_output_reg(emit->vposition.out_index);

   /* If non-adjusted vertex position register index
    * is valid, copy the vertex position from the temporary
    * vertex position register before it is modified by the
    * prescale computation.
    */
   if (emit->vposition.so_index != INVALID_INDEX) {
      struct tgsi_full_dst_register pos_so_dst =
         make_dst_output_reg(emit->vposition.so_index);

      /* MOV pos_so, tmp_pos */
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &pos_so_dst,
                           &tmp_pos_src, FALSE);
   }

   if (emit->vposition.need_prescale) {
      /* This code adjusts the vertex position to match the VGPU10 convention.
       * If p is the position computed by the shader (usually by applying the
       * modelview and projection matrices), the new position q is computed by:
       *
       * q.x = p.w * trans.x + p.x * scale.x
       * q.y = p.w * trans.y + p.y * scale.y
       * q.z = p.w * trans.z + p.z * scale.z;
       * q.w = p.w * trans.w + p.w;
       */
      struct tgsi_full_src_register tmp_pos_src_w =
         scalar_src(&tmp_pos_src, TGSI_SWIZZLE_W);
      struct tgsi_full_dst_register tmp_pos_dst =
         make_dst_temp_reg(vs_pos_tmp_index);
      struct tgsi_full_dst_register tmp_pos_dst_xyz =
         writemask_dst(&tmp_pos_dst, TGSI_WRITEMASK_XYZ);

      struct tgsi_full_src_register prescale_scale =
         make_src_const_reg(emit->vposition.prescale_scale_index);
      struct tgsi_full_src_register prescale_trans =
         make_src_const_reg(emit->vposition.prescale_trans_index);

      /* MUL tmp_pos.xyz, tmp_pos, prescale.scale */
      emit_instruction_op2(emit, VGPU10_OPCODE_MUL, &tmp_pos_dst_xyz,
                           &tmp_pos_src, &prescale_scale, FALSE);

      /* MAD pos, tmp_pos.wwww, prescale.trans, tmp_pos */
      emit_instruction_op3(emit, VGPU10_OPCODE_MAD, &pos_dst, &tmp_pos_src_w,
                           &prescale_trans, &tmp_pos_src, FALSE);
   }
   else if (emit->key.vs.undo_viewport) {
      /* This code computes the final vertex position from the temporary
       * vertex position by undoing the viewport transformation and the
       * divide-by-W operation (we convert window coords back to clip coords).
       * This is needed when we use the 'draw' module for fallbacks.
       * If p is the temp pos in window coords, then the NDC coord q is:
       *   q.x = (p.x - vp.x_trans) / vp.x_scale * p.w
       *   q.y = (p.y - vp.y_trans) / vp.y_scale * p.w
       *   q.z = p.z * p.w
       *   q.w = p.w
       * CONST[vs_viewport_index] contains:
       *   { 1/vp.x_scale, 1/vp.y_scale, -vp.x_trans, -vp.y_trans }
       */
      struct tgsi_full_dst_register tmp_pos_dst =
         make_dst_temp_reg(vs_pos_tmp_index);
      struct tgsi_full_dst_register tmp_pos_dst_xy =
         writemask_dst(&tmp_pos_dst, TGSI_WRITEMASK_XY);
      struct tgsi_full_src_register tmp_pos_src_wwww =
         scalar_src(&tmp_pos_src, TGSI_SWIZZLE_W);

      struct tgsi_full_dst_register pos_dst_xyz =
         writemask_dst(&pos_dst, TGSI_WRITEMASK_XYZ);
      struct tgsi_full_dst_register pos_dst_w =
         writemask_dst(&pos_dst, TGSI_WRITEMASK_W);

      struct tgsi_full_src_register vp_xyzw =
         make_src_const_reg(emit->vs.viewport_index);
      struct tgsi_full_src_register vp_zwww =
         swizzle_src(&vp_xyzw, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W,
                     TGSI_SWIZZLE_W, TGSI_SWIZZLE_W);

      /* ADD tmp_pos.xy, tmp_pos.xy, viewport.zwww */
      emit_instruction_op2(emit, VGPU10_OPCODE_ADD, &tmp_pos_dst_xy,
                           &tmp_pos_src, &vp_zwww, FALSE);

      /* MUL tmp_pos.xy, tmp_pos.xyzw, viewport.xyzy */
      emit_instruction_op2(emit, VGPU10_OPCODE_MUL, &tmp_pos_dst_xy,
                           &tmp_pos_src, &vp_xyzw, FALSE);

      /* MUL pos.xyz, tmp_pos.xyz, tmp_pos.www */
      emit_instruction_op2(emit, VGPU10_OPCODE_MUL, &pos_dst_xyz,
                           &tmp_pos_src, &tmp_pos_src_wwww, FALSE);

      /* MOV pos.w, tmp_pos.w */
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &pos_dst_w,
                           &tmp_pos_src, FALSE);
   }
   else if (vs_pos_tmp_index != INVALID_INDEX) {
      /* This code is to handle the case where the temporary vertex
       * position register is created when the vertex shader has stream
       * output and prescale is disabled because rasterization is to be
       * discarded.
       */
      struct tgsi_full_dst_register pos_dst =
         make_dst_output_reg(emit->vposition.out_index);

      /* MOV pos, tmp_pos */
      begin_emit_instruction(emit);
      emit_opcode(emit, VGPU10_OPCODE_MOV, FALSE);
      emit_dst_register(emit, &pos_dst);
      emit_src_register(emit, &tmp_pos_src);
      end_emit_instruction(emit);
   }
}

static void
emit_clipping_instructions(struct svga_shader_emitter_v10 *emit)
{
   if (emit->clip_mode == CLIP_DISTANCE) {
      /* Copy from copy distance temporary to CLIPDIST & the shadow copy */
      emit_clip_distance_instructions(emit);

   } else if (emit->clip_mode == CLIP_VERTEX) {
      /* Convert TGSI CLIPVERTEX to CLIPDIST */
      emit_clip_vertex_instructions(emit);
   }

   /**
    * Emit vertex position and take care of legacy user planes only if
    * there is a valid vertex position register index.
    * This is to take care of the case
    * where the shader doesn't output vertex position. Then in
    * this case, don't bother to emit more vertex instructions.
    */
   if (emit->vposition.out_index == INVALID_INDEX)
      return;

   /**
    * Emit per-vertex clipping instructions for legacy user defined clip planes.
    * NOTE: we must emit the clip distance instructions before the
    * emit_vpos_instructions() call since the later function will change
    * the TEMP[vs_pos_tmp_index] value.
    */
   if (emit->clip_mode == CLIP_LEGACY) {
      /* Emit CLIPDIST for legacy user defined clip planes */
      emit_clip_distance_from_vpos(emit, emit->vposition.tmp_index);
   }
}


/**
 * Emit extra per-vertex instructions.  This includes clip-coordinate
 * space conversion and computing clip distances.  This is called for
 * each GS emit-vertex instruction and at the end of VS translation.
 */
static void
emit_vertex_instructions(struct svga_shader_emitter_v10 *emit)
{
   const unsigned vs_pos_tmp_index = emit->vposition.tmp_index;

   /* Emit clipping instructions based on clipping mode */
   emit_clipping_instructions(emit);

   /**
    * Reset the temporary vertex position register index
    * so that emit_dst_register() will use the real vertex position output
    */
   emit->vposition.tmp_index = INVALID_INDEX;

   /* Emit vertex position instructions */
   emit_vpos_instructions(emit, vs_pos_tmp_index);

   /* Restore original vposition.tmp_index value for the next GS vertex.
    * It doesn't matter for VS.
    */
   emit->vposition.tmp_index = vs_pos_tmp_index;
}

/**
 * Translate the TGSI_OPCODE_EMIT GS instruction.
 */
static boolean
emit_vertex(struct svga_shader_emitter_v10 *emit,
            const struct tgsi_full_instruction *inst)
{
   unsigned ret = TRUE;

   assert(emit->unit == PIPE_SHADER_GEOMETRY);

   emit_vertex_instructions(emit);

   /* We can't use emit_simple() because the TGSI instruction has one
    * operand (vertex stream number) which we must ignore for VGPU10.
    */
   begin_emit_instruction(emit);
   emit_opcode(emit, VGPU10_OPCODE_EMIT, FALSE);
   end_emit_instruction(emit);

   return ret;
}


/**
 * Emit the extra code to convert from VGPU10's boolean front-face
 * register to TGSI's signed front-face register.
 *
 * TODO: Make temporary front-face register a scalar.
 */
static void
emit_frontface_instructions(struct svga_shader_emitter_v10 *emit)
{
   assert(emit->unit == PIPE_SHADER_FRAGMENT);

   if (emit->fs.face_input_index != INVALID_INDEX) {
      /* convert vgpu10 boolean face register to gallium +/-1 value */
      struct tgsi_full_dst_register tmp_dst =
         make_dst_temp_reg(emit->fs.face_tmp_index);
      struct tgsi_full_src_register one =
         make_immediate_reg_float(emit, 1.0f);
      struct tgsi_full_src_register neg_one =
         make_immediate_reg_float(emit, -1.0f);

      /* MOVC face_tmp, IS_FRONT_FACE.x, 1.0, -1.0 */
      begin_emit_instruction(emit);
      emit_opcode(emit, VGPU10_OPCODE_MOVC, FALSE);
      emit_dst_register(emit, &tmp_dst);
      emit_face_register(emit);
      emit_src_register(emit, &one);
      emit_src_register(emit, &neg_one);
      end_emit_instruction(emit);
   }
}


/**
 * Emit the extra code to convert from VGPU10's fragcoord.w value to 1/w.
 */
static void
emit_fragcoord_instructions(struct svga_shader_emitter_v10 *emit)
{
   assert(emit->unit == PIPE_SHADER_FRAGMENT);

   if (emit->fs.fragcoord_input_index != INVALID_INDEX) {
      struct tgsi_full_dst_register tmp_dst =
         make_dst_temp_reg(emit->fs.fragcoord_tmp_index);
      struct tgsi_full_dst_register tmp_dst_xyz =
         writemask_dst(&tmp_dst, TGSI_WRITEMASK_XYZ);
      struct tgsi_full_dst_register tmp_dst_w =
         writemask_dst(&tmp_dst, TGSI_WRITEMASK_W);
      struct tgsi_full_src_register one =
         make_immediate_reg_float(emit, 1.0f);
      struct tgsi_full_src_register fragcoord =
         make_src_reg(TGSI_FILE_INPUT, emit->fs.fragcoord_input_index);

      /* save the input index */
      unsigned fragcoord_input_index = emit->fs.fragcoord_input_index;
      /* set to invalid to prevent substitution in emit_src_register() */
      emit->fs.fragcoord_input_index = INVALID_INDEX;

      /* MOV fragcoord_tmp.xyz, fragcoord.xyz */
      begin_emit_instruction(emit);
      emit_opcode(emit, VGPU10_OPCODE_MOV, FALSE);
      emit_dst_register(emit, &tmp_dst_xyz);
      emit_src_register(emit, &fragcoord);
      end_emit_instruction(emit);

      /* DIV fragcoord_tmp.w, 1.0, fragcoord.w */
      begin_emit_instruction(emit);
      emit_opcode(emit, VGPU10_OPCODE_DIV, FALSE);
      emit_dst_register(emit, &tmp_dst_w);
      emit_src_register(emit, &one);
      emit_src_register(emit, &fragcoord);
      end_emit_instruction(emit);

      /* restore saved value */
      emit->fs.fragcoord_input_index = fragcoord_input_index;
   }
}


/**
 * Emit the extra code to get the current sample position value and
 * put it into a temp register.
 */
static void
emit_sample_position_instructions(struct svga_shader_emitter_v10 *emit)
{
   assert(emit->unit == PIPE_SHADER_FRAGMENT);

   if (emit->fs.sample_pos_sys_index != INVALID_INDEX) {
      assert(emit->version >= 41);

      struct tgsi_full_dst_register tmp_dst =
         make_dst_temp_reg(emit->fs.sample_pos_tmp_index);
      struct tgsi_full_src_register half =
         make_immediate_reg_float4(emit, 0.5, 0.5, 0.0, 0.0);

      struct tgsi_full_src_register tmp_src =
         make_src_temp_reg(emit->fs.sample_pos_tmp_index);
      struct tgsi_full_src_register sample_index_reg =
         make_src_scalar_reg(TGSI_FILE_SYSTEM_VALUE,
                             emit->fs.sample_id_sys_index, TGSI_SWIZZLE_X);

      /* The first src register is a shader resource (if we want a
       * multisampled resource sample position) or the rasterizer register
       * (if we want the current sample position in the color buffer).  We
       * want the later.
       */

      /* SAMPLE_POS dst, RASTERIZER, sampleIndex */
      begin_emit_instruction(emit);
      emit_opcode(emit, VGPU10_OPCODE_SAMPLE_POS, FALSE);
      emit_dst_register(emit, &tmp_dst);
      emit_rasterizer_register(emit);
      emit_src_register(emit, &sample_index_reg);
      end_emit_instruction(emit);

      /* Convert from D3D coords to GL coords by adding 0.5 bias */
      /* ADD dst, dst, half */
      begin_emit_instruction(emit);
      emit_opcode(emit, VGPU10_OPCODE_ADD, FALSE);
      emit_dst_register(emit, &tmp_dst);
      emit_src_register(emit, &tmp_src);
      emit_src_register(emit, &half);
      end_emit_instruction(emit);
   }
}


/**
 * Emit extra instructions to adjust VS inputs/attributes.  This can
 * mean casting a vertex attribute from int to float or setting the
 * W component to 1, or both.
 */
static void
emit_vertex_attrib_instructions(struct svga_shader_emitter_v10 *emit)
{
   const unsigned save_w_1_mask = emit->key.vs.adjust_attrib_w_1;
   const unsigned save_itof_mask = emit->key.vs.adjust_attrib_itof;
   const unsigned save_utof_mask = emit->key.vs.adjust_attrib_utof;
   const unsigned save_is_bgra_mask = emit->key.vs.attrib_is_bgra;
   const unsigned save_puint_to_snorm_mask = emit->key.vs.attrib_puint_to_snorm;
   const unsigned save_puint_to_uscaled_mask = emit->key.vs.attrib_puint_to_uscaled;
   const unsigned save_puint_to_sscaled_mask = emit->key.vs.attrib_puint_to_sscaled;

   unsigned adjust_mask = (save_w_1_mask |
                           save_itof_mask |
                           save_utof_mask |
                           save_is_bgra_mask |
                           save_puint_to_snorm_mask |
                           save_puint_to_uscaled_mask |
                           save_puint_to_sscaled_mask);

   assert(emit->unit == PIPE_SHADER_VERTEX);

   if (adjust_mask) {
      struct tgsi_full_src_register one =
         make_immediate_reg_float(emit, 1.0f);

      struct tgsi_full_src_register one_int =
         make_immediate_reg_int(emit, 1);

      /* We need to turn off these bitmasks while emitting the
       * instructions below, then restore them afterward.
       */
      emit->key.vs.adjust_attrib_w_1 = 0;
      emit->key.vs.adjust_attrib_itof = 0;
      emit->key.vs.adjust_attrib_utof = 0;
      emit->key.vs.attrib_is_bgra = 0;
      emit->key.vs.attrib_puint_to_snorm = 0;
      emit->key.vs.attrib_puint_to_uscaled = 0;
      emit->key.vs.attrib_puint_to_sscaled = 0;

      while (adjust_mask) {
         unsigned index = u_bit_scan(&adjust_mask);

         /* skip the instruction if this vertex attribute is not being used */
         if (emit->info.input_usage_mask[index] == 0)
            continue;

         unsigned tmp = emit->vs.adjusted_input[index];
         struct tgsi_full_src_register input_src =
            make_src_reg(TGSI_FILE_INPUT, index);

         struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
         struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
         struct tgsi_full_dst_register tmp_dst_w =
            writemask_dst(&tmp_dst, TGSI_WRITEMASK_W);

         /* ITOF/UTOF/MOV tmp, input[index] */
         if (save_itof_mask & (1 << index)) {
            emit_instruction_op1(emit, VGPU10_OPCODE_ITOF,
                                 &tmp_dst, &input_src, FALSE);
         }
         else if (save_utof_mask & (1 << index)) {
            emit_instruction_op1(emit, VGPU10_OPCODE_UTOF,
                                 &tmp_dst, &input_src, FALSE);
         }
         else if (save_puint_to_snorm_mask & (1 << index)) {
            emit_puint_to_snorm(emit, &tmp_dst, &input_src);
         }
         else if (save_puint_to_uscaled_mask & (1 << index)) {
            emit_puint_to_uscaled(emit, &tmp_dst, &input_src);
         }
         else if (save_puint_to_sscaled_mask & (1 << index)) {
            emit_puint_to_sscaled(emit, &tmp_dst, &input_src);
         }
         else {
            assert((save_w_1_mask | save_is_bgra_mask) & (1 << index));
            emit_instruction_op1(emit, VGPU10_OPCODE_MOV,
                                 &tmp_dst, &input_src, FALSE);
         }

         if (save_is_bgra_mask & (1 << index)) {
            emit_swap_r_b(emit, &tmp_dst, &tmp_src);
         }

         if (save_w_1_mask & (1 << index)) {
            /* MOV tmp.w, 1.0 */
            if (emit->key.vs.attrib_is_pure_int & (1 << index)) {
               emit_instruction_op1(emit, VGPU10_OPCODE_MOV,
                                    &tmp_dst_w, &one_int, FALSE);
            }
            else {
               emit_instruction_op1(emit, VGPU10_OPCODE_MOV,
                                    &tmp_dst_w, &one, FALSE);
            }
         }
      }

      emit->key.vs.adjust_attrib_w_1 = save_w_1_mask;
      emit->key.vs.adjust_attrib_itof = save_itof_mask;
      emit->key.vs.adjust_attrib_utof = save_utof_mask;
      emit->key.vs.attrib_is_bgra = save_is_bgra_mask;
      emit->key.vs.attrib_puint_to_snorm = save_puint_to_snorm_mask;
      emit->key.vs.attrib_puint_to_uscaled = save_puint_to_uscaled_mask;
      emit->key.vs.attrib_puint_to_sscaled = save_puint_to_sscaled_mask;
   }
}


/**
 * Some common values like 0.0, 1.0, 0.5, etc. are frequently needed
 * to implement some instructions.  We pre-allocate those values here
 * in the immediate constant buffer.
 */
static void
alloc_common_immediates(struct svga_shader_emitter_v10 *emit)
{
   unsigned n = 0;

   emit->common_immediate_pos[n++] =
      alloc_immediate_float4(emit, 0.0f, 1.0f, 0.5f, -1.0f);

   if (emit->info.opcode_count[TGSI_OPCODE_LIT] > 0) {
      emit->common_immediate_pos[n++] =
         alloc_immediate_float4(emit, 128.0f, -128.0f, 0.0f, 0.0f);
   }

   emit->common_immediate_pos[n++] =
      alloc_immediate_int4(emit, 0, 1, 0, -1);

   if (emit->key.vs.attrib_puint_to_snorm) {
      emit->common_immediate_pos[n++] =
         alloc_immediate_float4(emit, -2.0f, 2.0f, 3.0f, -1.66666f);
   }

   if (emit->key.vs.attrib_puint_to_uscaled) {
      emit->common_immediate_pos[n++] =
         alloc_immediate_float4(emit, 1023.0f, 3.0f, 0.0f, 0.0f);
   }

   if (emit->key.vs.attrib_puint_to_sscaled) {
      emit->common_immediate_pos[n++] =
         alloc_immediate_int4(emit, 22, 12, 2, 0);

      emit->common_immediate_pos[n++] =
         alloc_immediate_int4(emit, 22, 30, 0, 0);
   }

   unsigned i;

   for (i = 0; i < PIPE_MAX_SAMPLERS; i++) {
      if (emit->key.tex[i].texel_bias) {
         /* Replace 0.0f if more immediate float value is needed */
         emit->common_immediate_pos[n++] =
            alloc_immediate_float4(emit, 0.0001f, 0.0f, 0.0f, 0.0f);
         break;
      }
   }

   assert(n <= ARRAY_SIZE(emit->common_immediate_pos));
   emit->num_common_immediates = n;
}


/**
 * Emit any extra/helper declarations/code that we might need between
 * the declaration section and code section.
 */
static boolean
emit_pre_helpers(struct svga_shader_emitter_v10 *emit)
{
   /* Properties */
   if (emit->unit == PIPE_SHADER_GEOMETRY)
      emit_property_instructions(emit);

   /* Declare inputs */
   if (!emit_input_declarations(emit))
      return FALSE;

   /* Declare outputs */
   if (!emit_output_declarations(emit))
      return FALSE;

   /* Declare temporary registers */
   emit_temporaries_declaration(emit);

   /* Declare constant registers */
   emit_constant_declaration(emit);

   /* Declare samplers and resources */
   emit_sampler_declarations(emit);
   emit_resource_declarations(emit);

   /* Declare clip distance output registers */
   if (emit->unit == PIPE_SHADER_VERTEX ||
       emit->unit == PIPE_SHADER_GEOMETRY) {
      emit_clip_distance_declarations(emit);
   }

   alloc_common_immediates(emit);

   if (emit->unit == PIPE_SHADER_FRAGMENT &&
       emit->key.fs.alpha_func != SVGA3D_CMP_ALWAYS) {
      float alpha = emit->key.fs.alpha_ref;
      emit->fs.alpha_ref_index =
         alloc_immediate_float4(emit, alpha, alpha, alpha, alpha);
   }

   /* Now, emit the constant block containing all the immediates
    * declared by shader, as well as the extra ones seen above.
    */
   emit_vgpu10_immediates_block(emit);

   if (emit->unit == PIPE_SHADER_FRAGMENT) {
      emit_frontface_instructions(emit);
      emit_fragcoord_instructions(emit);
      emit_sample_position_instructions(emit);
   }
   else if (emit->unit == PIPE_SHADER_VERTEX) {
      emit_vertex_attrib_instructions(emit);
   }

   return TRUE;
}


/**
 * The device has no direct support for the pipe_blend_state::alpha_to_one
 * option so we implement it here with shader code.
 *
 * Note that this is kind of pointless, actually.  Here we're clobbering
 * the alpha value with 1.0.  So if alpha-to-coverage is enabled, we'll wind
 * up with 100% coverage.  That's almost certainly not what the user wants.
 * The work-around is to add extra shader code to compute coverage from alpha
 * and write it to the coverage output register (if the user's shader doesn't
 * do so already).  We'll probably do that in the future.
 */
static void
emit_alpha_to_one_instructions(struct svga_shader_emitter_v10 *emit,
                               unsigned fs_color_tmp_index)
{
   struct tgsi_full_src_register one = make_immediate_reg_float(emit, 1.0f);
   unsigned i;

   /* Note: it's not 100% clear from the spec if we're supposed to clobber
    * the alpha for all render targets.  But that's what NVIDIA does and
    * that's what Piglit tests.
    */
   for (i = 0; i < emit->fs.num_color_outputs; i++) {
      struct tgsi_full_dst_register color_dst;

      if (fs_color_tmp_index != INVALID_INDEX && i == 0) {
         /* write to the temp color register */
         color_dst = make_dst_temp_reg(fs_color_tmp_index);
      }
      else {
         /* write directly to the color[i] output */
         color_dst = make_dst_output_reg(emit->fs.color_out_index[i]);
      }

      color_dst = writemask_dst(&color_dst, TGSI_WRITEMASK_W);

      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &color_dst, &one, FALSE);
   }
}


/**
 * Emit alpha test code.  This compares TEMP[fs_color_tmp_index].w
 * against the alpha reference value and discards the fragment if the
 * comparison fails.
 */
static void
emit_alpha_test_instructions(struct svga_shader_emitter_v10 *emit,
                             unsigned fs_color_tmp_index)
{
   /* compare output color's alpha to alpha ref and kill */
   unsigned tmp = get_temp_index(emit);
   struct tgsi_full_src_register tmp_src = make_src_temp_reg(tmp);
   struct tgsi_full_src_register tmp_src_x =
      scalar_src(&tmp_src, TGSI_SWIZZLE_X);
   struct tgsi_full_dst_register tmp_dst = make_dst_temp_reg(tmp);
   struct tgsi_full_src_register color_src =
      make_src_temp_reg(fs_color_tmp_index);
   struct tgsi_full_src_register color_src_w =
      scalar_src(&color_src, TGSI_SWIZZLE_W);
   struct tgsi_full_src_register ref_src =
      make_src_immediate_reg(emit->fs.alpha_ref_index);
   struct tgsi_full_dst_register color_dst =
      make_dst_output_reg(emit->fs.color_out_index[0]);

   assert(emit->unit == PIPE_SHADER_FRAGMENT);

   /* dst = src0 'alpha_func' src1 */
   emit_comparison(emit, emit->key.fs.alpha_func, &tmp_dst,
                   &color_src_w, &ref_src);

   /* DISCARD if dst.x == 0 */
   begin_emit_instruction(emit);
   emit_discard_opcode(emit, FALSE);  /* discard if src0.x is zero */
   emit_src_register(emit, &tmp_src_x);
   end_emit_instruction(emit);

   /* If we don't need to broadcast the color below, emit the final color here.
    */
   if (emit->key.fs.write_color0_to_n_cbufs <= 1) {
      /* MOV output.color, tempcolor */
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &color_dst,
                           &color_src, FALSE);     /* XXX saturate? */
   }

   free_temp_indexes(emit);
}


/**
 * Emit instructions for writing a single color output to multiple
 * color buffers.
 * This is used when the TGSI_PROPERTY_FS_COLOR0_WRITES_ALL_CBUFS (or
 * when key.fs.white_fragments is true).
 * property is set and the number of render targets is greater than one.
 * \param fs_color_tmp_index  index of the temp register that holds the
 *                            color to broadcast.
 */
static void
emit_broadcast_color_instructions(struct svga_shader_emitter_v10 *emit,
                                 unsigned fs_color_tmp_index)
{
   const unsigned n = emit->key.fs.write_color0_to_n_cbufs;
   unsigned i;
   struct tgsi_full_src_register color_src;

   if (emit->key.fs.white_fragments) {
      /* set all color outputs to white */
      color_src = make_immediate_reg_float(emit, 1.0f);
   }
   else {
      /* set all color outputs to TEMP[fs_color_tmp_index] */
      assert(fs_color_tmp_index != INVALID_INDEX);
      color_src = make_src_temp_reg(fs_color_tmp_index);
   }

   assert(emit->unit == PIPE_SHADER_FRAGMENT);

   for (i = 0; i < n; i++) {
      unsigned output_reg = emit->fs.color_out_index[i];
      struct tgsi_full_dst_register color_dst =
         make_dst_output_reg(output_reg);

      /* Fill in this semantic here since we'll use it later in
       * emit_dst_register().
       */
      emit->info.output_semantic_name[output_reg] = TGSI_SEMANTIC_COLOR;

      /* MOV output.color[i], tempcolor */
      emit_instruction_op1(emit, VGPU10_OPCODE_MOV, &color_dst,
                           &color_src, FALSE);     /* XXX saturate? */
   }
}


/**
 * Emit extra helper code after the original shader code, but before the
 * last END/RET instruction.
 * For vertex shaders this means emitting the extra code to apply the
 * prescale scale/translation.
 */
static boolean
emit_post_helpers(struct svga_shader_emitter_v10 *emit)
{
   if (emit->unit == PIPE_SHADER_VERTEX) {
      emit_vertex_instructions(emit);
   }
   else if (emit->unit == PIPE_SHADER_FRAGMENT) {
      const unsigned fs_color_tmp_index = emit->fs.color_tmp_index;

      assert(!(emit->key.fs.white_fragments &&
               emit->key.fs.write_color0_to_n_cbufs == 0));

      /* We no longer want emit_dst_register() to substitute the
       * temporary fragment color register for the real color output.
       */
      emit->fs.color_tmp_index = INVALID_INDEX;

      if (emit->key.fs.alpha_to_one) {
         emit_alpha_to_one_instructions(emit, fs_color_tmp_index);
      }
      if (emit->key.fs.alpha_func != SVGA3D_CMP_ALWAYS) {
         emit_alpha_test_instructions(emit, fs_color_tmp_index);
      }
      if (emit->key.fs.write_color0_to_n_cbufs > 1 ||
          emit->key.fs.white_fragments) {
         emit_broadcast_color_instructions(emit, fs_color_tmp_index);
      }
   }

   return TRUE;
}


/**
 * Translate the TGSI tokens into VGPU10 tokens.
 */
static boolean
emit_vgpu10_instructions(struct svga_shader_emitter_v10 *emit,
                         const struct tgsi_token *tokens)
{
   struct tgsi_parse_context parse;
   boolean ret = TRUE;
   boolean pre_helpers_emitted = FALSE;
   unsigned inst_number = 0;

   tgsi_parse_init(&parse, tokens);

   while (!tgsi_parse_end_of_tokens(&parse)) {
      tgsi_parse_token(&parse);

      switch (parse.FullToken.Token.Type) {
      case TGSI_TOKEN_TYPE_IMMEDIATE:
         ret = emit_vgpu10_immediate(emit, &parse.FullToken.FullImmediate);
         if (!ret)
            goto done;
         break;

      case TGSI_TOKEN_TYPE_DECLARATION:
         ret = emit_vgpu10_declaration(emit, &parse.FullToken.FullDeclaration);
         if (!ret)
            goto done;
         break;

      case TGSI_TOKEN_TYPE_INSTRUCTION:
         if (!pre_helpers_emitted) {
            ret = emit_pre_helpers(emit);
            if (!ret)
               goto done;
            pre_helpers_emitted = TRUE;
         }
         ret = emit_vgpu10_instruction(emit, inst_number++,
                                       &parse.FullToken.FullInstruction);
         if (!ret)
            goto done;
         break;

      case TGSI_TOKEN_TYPE_PROPERTY:
         ret = emit_vgpu10_property(emit, &parse.FullToken.FullProperty);
         if (!ret)
            goto done;
         break;

      default:
         break;
      }
   }

done:
   tgsi_parse_free(&parse);
   return ret;
}


/**
 * Emit the first VGPU10 shader tokens.
 */
static boolean
emit_vgpu10_header(struct svga_shader_emitter_v10 *emit)
{
   VGPU10ProgramToken ptoken;

   /* First token: VGPU10ProgramToken  (version info, program type (VS,GS,PS)) */
   ptoken.majorVersion = emit->version / 10;
   ptoken.minorVersion = emit->version % 10;
   ptoken.programType = translate_shader_type(emit->unit);
   if (!emit_dword(emit, ptoken.value))
      return FALSE;

   /* Second token: total length of shader, in tokens.  We can't fill this
    * in until we're all done.  Emit zero for now.
    */
   return emit_dword(emit, 0);
}


static boolean
emit_vgpu10_tail(struct svga_shader_emitter_v10 *emit)
{
   VGPU10ProgramToken *tokens;

   /* Replace the second token with total shader length */
   tokens = (VGPU10ProgramToken *) emit->buf;
   tokens[1].value = emit_get_num_tokens(emit);

   return TRUE;
}


/**
 * Modify the FS to read the BCOLORs and use the FACE register
 * to choose between the front/back colors.
 */
static const struct tgsi_token *
transform_fs_twoside(const struct tgsi_token *tokens)
{
   if (0) {
      debug_printf("Before tgsi_add_two_side ------------------\n");
      tgsi_dump(tokens,0);
   }
   tokens = tgsi_add_two_side(tokens);
   if (0) {
      debug_printf("After tgsi_add_two_side ------------------\n");
      tgsi_dump(tokens, 0);
   }
   return tokens;
}


/**
 * Modify the FS to do polygon stipple.
 */
static const struct tgsi_token *
transform_fs_pstipple(struct svga_shader_emitter_v10 *emit,
                      const struct tgsi_token *tokens)
{
   const struct tgsi_token *new_tokens;
   unsigned unit;

   if (0) {
      debug_printf("Before pstipple ------------------\n");
      tgsi_dump(tokens,0);
   }

   new_tokens = util_pstipple_create_fragment_shader(tokens, &unit, 0,
                                                     TGSI_FILE_INPUT);

   emit->fs.pstipple_sampler_unit = unit;

   /* Setup texture state for stipple */
   emit->sampler_target[unit] = TGSI_TEXTURE_2D;
   emit->key.tex[unit].swizzle_r = TGSI_SWIZZLE_X;
   emit->key.tex[unit].swizzle_g = TGSI_SWIZZLE_Y;
   emit->key.tex[unit].swizzle_b = TGSI_SWIZZLE_Z;
   emit->key.tex[unit].swizzle_a = TGSI_SWIZZLE_W;

   if (0) {
      debug_printf("After pstipple ------------------\n");
      tgsi_dump(new_tokens, 0);
   }

   return new_tokens;
}

/**
 * Modify the FS to support anti-aliasing point.
 */
static const struct tgsi_token *
transform_fs_aapoint(const struct tgsi_token *tokens,
                     int aa_coord_index)
{
   if (0) {
      debug_printf("Before tgsi_add_aa_point ------------------\n");
      tgsi_dump(tokens,0);
   }
   tokens = tgsi_add_aa_point(tokens, aa_coord_index);
   if (0) {
      debug_printf("After tgsi_add_aa_point ------------------\n");
      tgsi_dump(tokens, 0);
   }
   return tokens;
}

/**
 * This is the main entrypoint for the TGSI -> VPGU10 translator.
 */
struct svga_shader_variant *
svga_tgsi_vgpu10_translate(struct svga_context *svga,
                           const struct svga_shader *shader,
                           const struct svga_compile_key *key,
                           enum pipe_shader_type unit)
{
   struct svga_shader_variant *variant = NULL;
   struct svga_shader_emitter_v10 *emit;
   const struct tgsi_token *tokens = shader->tokens;
   struct svga_vertex_shader *vs = svga->curr.vs;
   struct svga_geometry_shader *gs = svga->curr.gs;

   assert(unit == PIPE_SHADER_VERTEX ||
          unit == PIPE_SHADER_GEOMETRY ||
          unit == PIPE_SHADER_FRAGMENT);

   /* These two flags cannot be used together */
   assert(key->vs.need_prescale + key->vs.undo_viewport <= 1);

   SVGA_STATS_TIME_PUSH(svga_sws(svga), SVGA_STATS_TIME_TGSIVGPU10TRANSLATE);
   /*
    * Setup the code emitter
    */
   emit = alloc_emitter();
   if (!emit)
      goto done;

   emit->unit = unit;
   emit->version = svga_have_sm4_1(svga) ? 41 : 40;

   emit->key = *key;

   emit->vposition.need_prescale = (emit->key.vs.need_prescale ||
                                   emit->key.gs.need_prescale);
   emit->vposition.tmp_index = INVALID_INDEX;
   emit->vposition.so_index = INVALID_INDEX;
   emit->vposition.out_index = INVALID_INDEX;

   emit->fs.color_tmp_index = INVALID_INDEX;
   emit->fs.face_input_index = INVALID_INDEX;
   emit->fs.fragcoord_input_index = INVALID_INDEX;
   emit->fs.sample_id_sys_index = INVALID_INDEX;
   emit->fs.sample_pos_sys_index = INVALID_INDEX;

   emit->gs.prim_id_index = INVALID_INDEX;

   emit->clip_dist_out_index = INVALID_INDEX;
   emit->clip_dist_tmp_index = INVALID_INDEX;
   emit->clip_dist_so_index = INVALID_INDEX;
   emit->clip_vertex_out_index = INVALID_INDEX;

   if (emit->key.fs.alpha_func == SVGA3D_CMP_INVALID) {
      emit->key.fs.alpha_func = SVGA3D_CMP_ALWAYS;
   }

   if (unit == PIPE_SHADER_FRAGMENT) {
      if (key->fs.light_twoside) {
         tokens = transform_fs_twoside(tokens);
      }
      if (key->fs.pstipple) {
         const struct tgsi_token *new_tokens =
            transform_fs_pstipple(emit, tokens);
         if (tokens != shader->tokens) {
            /* free the two-sided shader tokens */
            tgsi_free_tokens(tokens);
         }
         tokens = new_tokens;
      }
      if (key->fs.aa_point) {
         tokens = transform_fs_aapoint(tokens, key->fs.aa_point_coord_index);
      }
   }

   if (SVGA_DEBUG & DEBUG_TGSI) {
      debug_printf("#####################################\n");
      debug_printf("### TGSI Shader %u\n", shader->id);
      tgsi_dump(tokens, 0);
   }

   /**
    * Rescan the header if the token string is different from the one
    * included in the shader; otherwise, the header info is already up-to-date
    */
   if (tokens != shader->tokens) {
      tgsi_scan_shader(tokens, &emit->info);
   } else {
      emit->info = shader->info;
   }

   emit->num_outputs = emit->info.num_outputs;

   if (unit == PIPE_SHADER_FRAGMENT) {
      /* Compute FS input remapping to match the output from VS/GS */
      if (gs) {
         svga_link_shaders(&gs->base.info, &emit->info, &emit->linkage);
      } else {
         assert(vs);
         svga_link_shaders(&vs->base.info, &emit->info, &emit->linkage);
      }
   } else if (unit == PIPE_SHADER_GEOMETRY) {
      assert(vs);
      svga_link_shaders(&vs->base.info, &emit->info, &emit->linkage);
   }

   /* Since vertex shader does not need to go through the linker to
    * establish the input map, we need to make sure the highest index
    * of input registers is set properly here.
    */
   emit->linkage.input_map_max = MAX2((int)emit->linkage.input_map_max,
                                      emit->info.file_max[TGSI_FILE_INPUT]);

   determine_clipping_mode(emit);

   if (unit == PIPE_SHADER_GEOMETRY || unit == PIPE_SHADER_VERTEX) {
      if (shader->stream_output != NULL || emit->clip_mode == CLIP_DISTANCE) {
         /* if there is stream output declarations associated
          * with this shader or the shader writes to ClipDistance
          * then reserve extra registers for the non-adjusted vertex position
          * and the ClipDistance shadow copy
          */
         emit->vposition.so_index = emit->num_outputs++;

         if (emit->clip_mode == CLIP_DISTANCE) {
            emit->clip_dist_so_index = emit->num_outputs++;
            if (emit->info.num_written_clipdistance > 4)
               emit->num_outputs++;
         }
      }
   }

   /*
    * Do actual shader translation.
    */
   if (!emit_vgpu10_header(emit)) {
      debug_printf("svga: emit VGPU10 header failed\n");
      goto cleanup;
   }

   if (!emit_vgpu10_instructions(emit, tokens)) {
      debug_printf("svga: emit VGPU10 instructions failed\n");
      goto cleanup;
   }

   if (!emit_vgpu10_tail(emit)) {
      debug_printf("svga: emit VGPU10 tail failed\n");
      goto cleanup;
   }

   if (emit->register_overflow) {
      goto cleanup;
   }

   /*
    * Create, initialize the 'variant' object.
    */
   variant = svga_new_shader_variant(svga);
   if (!variant)
      goto cleanup;

   variant->shader = shader;
   variant->nr_tokens = emit_get_num_tokens(emit);
   variant->tokens = (const unsigned *)emit->buf;
   emit->buf = NULL;  /* buffer is no longer owed by emitter context */
   memcpy(&variant->key, key, sizeof(*key));
   variant->id = UTIL_BITMASK_INVALID_INDEX;

   /* The extra constant starting offset starts with the number of
    * shader constants declared in the shader.
    */
   variant->extra_const_start = emit->num_shader_consts[0];
   if (key->gs.wide_point) {
      /**
       * The extra constant added in the transformed shader
       * for inverse viewport scale is to be supplied by the driver.
       * So the extra constant starting offset needs to be reduced by 1.
       */
      assert(variant->extra_const_start > 0);
      variant->extra_const_start--;
   }

   variant->pstipple_sampler_unit = emit->fs.pstipple_sampler_unit;

   /* If there was exactly one write to a fragment shader output register
    * and it came from a constant buffer, we know all fragments will have
    * the same color (except for blending).
    */
   variant->constant_color_output =
      emit->constant_color_output && emit->num_output_writes == 1;

   /** keep track in the variant if flat interpolation is used
    *  for any of the varyings.
    */
   variant->uses_flat_interp = emit->uses_flat_interp;

   variant->fs_shadow_compare_units = emit->fs.shadow_compare_units;

   variant->fs_shadow_compare_units = emit->fs.shadow_compare_units;

   if (tokens != shader->tokens) {
      tgsi_free_tokens(tokens);
   }

cleanup:
   free_emitter(emit);

done:
   SVGA_STATS_TIME_POP(svga_sws(svga));
   return variant;
}
