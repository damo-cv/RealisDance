/*
 * Copyright © 2016 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef GEN_DECODER_H
#define GEN_DECODER_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include "dev/gen_device_info.h"
#include "util/hash_table.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gen_spec;
struct gen_group;
struct gen_field;
union gen_field_value;

static inline uint32_t gen_make_gen(uint32_t major, uint32_t minor)
{
   return (major << 8) | minor;
}

struct gen_group *gen_spec_find_struct(struct gen_spec *spec, const char *name);
struct gen_spec *gen_spec_load(const struct gen_device_info *devinfo);
struct gen_spec *gen_spec_load_from_path(const struct gen_device_info *devinfo,
                                         const char *path);
void gen_spec_destroy(struct gen_spec *spec);
uint32_t gen_spec_get_gen(struct gen_spec *spec);
struct gen_group *gen_spec_find_instruction(struct gen_spec *spec, const uint32_t *p);
struct gen_group *gen_spec_find_register(struct gen_spec *spec, uint32_t offset);
struct gen_group *gen_spec_find_register_by_name(struct gen_spec *spec, const char *name);
struct gen_enum *gen_spec_find_enum(struct gen_spec *spec, const char *name);

int gen_group_get_length(struct gen_group *group, const uint32_t *p);
const char *gen_group_get_name(struct gen_group *group);
uint32_t gen_group_get_opcode(struct gen_group *group);
struct gen_field *gen_group_find_field(struct gen_group *group, const char *name);
struct gen_enum *gen_spec_find_enum(struct gen_spec *spec, const char *name);

bool gen_field_is_header(struct gen_field *field);

struct gen_field_iterator {
   struct gen_group *group;
   char name[128];
   char value[128];
   uint64_t raw_value;
   struct gen_group *struct_desc;
   const uint32_t *p;
   int p_bit; /**< bit offset into p */
   const uint32_t *p_end;
   int start_bit; /**< current field starts at this bit offset into p */
   int end_bit; /**< current field ends at this bit offset into p */

   int group_iter;

   struct gen_field *field;
   bool print_colors;
};

struct gen_spec {
   uint32_t gen;

   struct hash_table *commands;
   struct hash_table *structs;
   struct hash_table *registers_by_name;
   struct hash_table *registers_by_offset;
   struct hash_table *enums;

   struct hash_table *access_cache;
};

struct gen_group {
   struct gen_spec *spec;
   char *name;

   struct gen_field *fields; /* linked list of fields */
   struct gen_field *dword_length_field; /* <instruction> specific */

   uint32_t dw_length;
   uint32_t bias; /* <instruction> specific */
   uint32_t group_offset, group_count;
   uint32_t group_size;
   bool variable; /* <group> specific */
   bool fixed_length; /* True for <struct> & <register> */

   struct gen_group *parent;
   struct gen_group *next;

   uint32_t opcode_mask;
   uint32_t opcode;

   uint32_t register_offset; /* <register> specific */
};

struct gen_value {
   char *name;
   uint64_t value;
};

struct gen_enum {
   char *name;
   int nvalues;
   struct gen_value **values;
};

struct gen_type {
   enum {
      GEN_TYPE_UNKNOWN,
      GEN_TYPE_INT,
      GEN_TYPE_UINT,
      GEN_TYPE_BOOL,
      GEN_TYPE_FLOAT,
      GEN_TYPE_ADDRESS,
      GEN_TYPE_OFFSET,
      GEN_TYPE_STRUCT,
      GEN_TYPE_UFIXED,
      GEN_TYPE_SFIXED,
      GEN_TYPE_MBO,
      GEN_TYPE_ENUM
   } kind;

   /* Struct definition for  GEN_TYPE_STRUCT */
   union {
      struct gen_group *gen_struct;
      struct gen_enum *gen_enum;
      struct {
         /* Integer and fractional sizes for GEN_TYPE_UFIXED and GEN_TYPE_SFIXED */
         int i, f;
      };
   };
};

union gen_field_value {
   bool b32;
   float f32;
   uint64_t u64;
   int64_t i64;
};

struct gen_field {
   struct gen_group *parent;
   struct gen_field *next;

   char *name;
   int start, end;
   struct gen_type type;
   bool has_default;
   uint32_t default_value;

   struct gen_enum inline_enum;
};

void gen_field_iterator_init(struct gen_field_iterator *iter,
                             struct gen_group *group,
                             const uint32_t *p, int p_bit,
                             bool print_colors);

bool gen_field_iterator_next(struct gen_field_iterator *iter);

void gen_print_group(FILE *out,
                     struct gen_group *group,
                     uint64_t offset, const uint32_t *p, int p_bit,
                     bool color);

enum gen_batch_decode_flags {
   /** Print in color! */
   GEN_BATCH_DECODE_IN_COLOR  = (1 << 0),
   /** Print everything, not just headers */
   GEN_BATCH_DECODE_FULL      = (1 << 1),
   /** Print offsets along with the batch */
   GEN_BATCH_DECODE_OFFSETS   = (1 << 2),
   /** Guess when a value is a float and print it as such */
   GEN_BATCH_DECODE_FLOATS    = (1 << 3),
};

struct gen_batch_decode_bo {
   uint64_t addr;
   uint32_t size;
   const void *map;
};

struct gen_batch_decode_ctx {
   /**
    * Return information about the buffer containing the given address.
    *
    * If the given address is inside a buffer, the map pointer should be
    * offset accordingly so it points at the data corresponding to address.
    */
   struct gen_batch_decode_bo (*get_bo)(void *user_data, uint64_t address);
   unsigned (*get_state_size)(void *user_data,
                              uint32_t offset_from_dynamic_state_base_addr);
   void *user_data;

   FILE *fp;
   struct gen_spec *spec;
   enum gen_batch_decode_flags flags;

   struct gen_disasm *disasm;

   uint64_t surface_base;
   uint64_t dynamic_base;
   uint64_t instruction_base;

   int max_vbo_decoded_lines;
};

void gen_batch_decode_ctx_init(struct gen_batch_decode_ctx *ctx,
                               const struct gen_device_info *devinfo,
                               FILE *fp, enum gen_batch_decode_flags flags,
                               const char *xml_path,
                               struct gen_batch_decode_bo (*get_bo)(void *,
                                                                    uint64_t),

                               unsigned (*get_state_size)(void *, uint32_t),
                               void *user_data);
void gen_batch_decode_ctx_finish(struct gen_batch_decode_ctx *ctx);


void gen_print_batch(struct gen_batch_decode_ctx *ctx,
                     const uint32_t *batch, uint32_t batch_size,
                     uint64_t batch_addr);

#ifdef __cplusplus
}
#endif


#endif /* GEN_DECODER_H */
