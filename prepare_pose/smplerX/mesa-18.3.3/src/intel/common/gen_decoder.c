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

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <expat.h>
#include <inttypes.h>
#include <zlib.h>

#include <util/macros.h>
#include <util/ralloc.h>

#include "gen_decoder.h"

#include "isl/isl.h"
#include "genxml/genX_xml.h"

#define XML_BUFFER_SIZE 4096
#define MAX_VALUE_ITEMS 128

struct location {
   const char *filename;
   int line_number;
};

struct parser_context {
   XML_Parser parser;
   int foo;
   struct location loc;

   struct gen_group *group;
   struct gen_enum *enoom;

   int n_values, n_allocated_values;
   struct gen_value **values;

   struct gen_field *last_field;

   struct gen_spec *spec;
};

const char *
gen_group_get_name(struct gen_group *group)
{
   return group->name;
}

uint32_t
gen_group_get_opcode(struct gen_group *group)
{
   return group->opcode;
}

struct gen_group *
gen_spec_find_struct(struct gen_spec *spec, const char *name)
{
   struct hash_entry *entry = _mesa_hash_table_search(spec->structs,
                                                      name);
   return entry ? entry->data : NULL;
}

struct gen_group *
gen_spec_find_register(struct gen_spec *spec, uint32_t offset)
{
   struct hash_entry *entry =
      _mesa_hash_table_search(spec->registers_by_offset,
                              (void *) (uintptr_t) offset);
   return entry ? entry->data : NULL;
}

struct gen_group *
gen_spec_find_register_by_name(struct gen_spec *spec, const char *name)
{
   struct hash_entry *entry =
      _mesa_hash_table_search(spec->registers_by_name, name);
   return entry ? entry->data : NULL;
}

struct gen_enum *
gen_spec_find_enum(struct gen_spec *spec, const char *name)
{
   struct hash_entry *entry = _mesa_hash_table_search(spec->enums,
                                                      name);
   return entry ? entry->data : NULL;
}

uint32_t
gen_spec_get_gen(struct gen_spec *spec)
{
   return spec->gen;
}

static void __attribute__((noreturn))
fail(struct location *loc, const char *msg, ...)
{
   va_list ap;

   va_start(ap, msg);
   fprintf(stderr, "%s:%d: error: ",
           loc->filename, loc->line_number);
   vfprintf(stderr, msg, ap);
   fprintf(stderr, "\n");
   va_end(ap);
   exit(EXIT_FAILURE);
}

static void
get_group_offset_count(const char **atts, uint32_t *offset, uint32_t *count,
                       uint32_t *size, bool *variable)
{
   for (int i = 0; atts[i]; i += 2) {
      char *p;

      if (strcmp(atts[i], "count") == 0) {
         *count = strtoul(atts[i + 1], &p, 0);
         if (*count == 0)
            *variable = true;
      } else if (strcmp(atts[i], "start") == 0) {
         *offset = strtoul(atts[i + 1], &p, 0);
      } else if (strcmp(atts[i], "size") == 0) {
         *size = strtoul(atts[i + 1], &p, 0);
      }
   }
   return;
}

static struct gen_group *
create_group(struct parser_context *ctx,
             const char *name,
             const char **atts,
             struct gen_group *parent,
             bool fixed_length)
{
   struct gen_group *group;

   group = rzalloc(ctx->spec, struct gen_group);
   if (name)
      group->name = ralloc_strdup(group, name);

   group->spec = ctx->spec;
   group->variable = false;
   group->fixed_length = fixed_length;
   group->dword_length_field = NULL;
   group->dw_length = 0;
   group->bias = 1;

   for (int i = 0; atts[i]; i += 2) {
      char *p;
      if (strcmp(atts[i], "length") == 0) {
         group->dw_length = strtoul(atts[i + 1], &p, 0);
      } else if (strcmp(atts[i], "bias") == 0) {
         group->bias = strtoul(atts[i + 1], &p, 0);
      }
   }

   if (parent) {
      group->parent = parent;
      get_group_offset_count(atts,
                             &group->group_offset,
                             &group->group_count,
                             &group->group_size,
                             &group->variable);
   }

   return group;
}

static struct gen_enum *
create_enum(struct parser_context *ctx, const char *name, const char **atts)
{
   struct gen_enum *e;

   e = rzalloc(ctx->spec, struct gen_enum);
   if (name)
      e->name = ralloc_strdup(e, name);

   return e;
}

static void
get_register_offset(const char **atts, uint32_t *offset)
{
   for (int i = 0; atts[i]; i += 2) {
      char *p;

      if (strcmp(atts[i], "num") == 0)
         *offset = strtoul(atts[i + 1], &p, 0);
   }
   return;
}

static void
get_start_end_pos(int *start, int *end)
{
   /* start value has to be mod with 32 as we need the relative
    * start position in the first DWord. For the end position, add
    * the length of the field to the start position to get the
    * relative postion in the 64 bit address.
    */
   if (*end - *start > 32) {
      int len = *end - *start;
      *start = *start % 32;
      *end = *start + len;
   } else {
      *start = *start % 32;
      *end = *end % 32;
   }

   return;
}

static inline uint64_t
mask(int start, int end)
{
   uint64_t v;

   v = ~0ULL >> (63 - end + start);

   return v << start;
}

static inline uint64_t
field_value(uint64_t value, int start, int end)
{
   get_start_end_pos(&start, &end);
   return (value & mask(start, end)) >> (start);
}

static struct gen_type
string_to_type(struct parser_context *ctx, const char *s)
{
   int i, f;
   struct gen_group *g;
   struct gen_enum *e;

   if (strcmp(s, "int") == 0)
      return (struct gen_type) { .kind = GEN_TYPE_INT };
   else if (strcmp(s, "uint") == 0)
      return (struct gen_type) { .kind = GEN_TYPE_UINT };
   else if (strcmp(s, "bool") == 0)
      return (struct gen_type) { .kind = GEN_TYPE_BOOL };
   else if (strcmp(s, "float") == 0)
      return (struct gen_type) { .kind = GEN_TYPE_FLOAT };
   else if (strcmp(s, "address") == 0)
      return (struct gen_type) { .kind = GEN_TYPE_ADDRESS };
   else if (strcmp(s, "offset") == 0)
      return (struct gen_type) { .kind = GEN_TYPE_OFFSET };
   else if (sscanf(s, "u%d.%d", &i, &f) == 2)
      return (struct gen_type) { .kind = GEN_TYPE_UFIXED, .i = i, .f = f };
   else if (sscanf(s, "s%d.%d", &i, &f) == 2)
      return (struct gen_type) { .kind = GEN_TYPE_SFIXED, .i = i, .f = f };
   else if (g = gen_spec_find_struct(ctx->spec, s), g != NULL)
      return (struct gen_type) { .kind = GEN_TYPE_STRUCT, .gen_struct = g };
   else if (e = gen_spec_find_enum(ctx->spec, s), e != NULL)
      return (struct gen_type) { .kind = GEN_TYPE_ENUM, .gen_enum = e };
   else if (strcmp(s, "mbo") == 0)
      return (struct gen_type) { .kind = GEN_TYPE_MBO };
   else
      fail(&ctx->loc, "invalid type: %s", s);
}

static struct gen_field *
create_field(struct parser_context *ctx, const char **atts)
{
   struct gen_field *field;

   field = rzalloc(ctx->group, struct gen_field);
   field->parent = ctx->group;

   for (int i = 0; atts[i]; i += 2) {
      char *p;

      if (strcmp(atts[i], "name") == 0) {
         field->name = ralloc_strdup(field, atts[i + 1]);
         if (strcmp(field->name, "DWord Length") == 0) {
            field->parent->dword_length_field = field;
         }
      } else if (strcmp(atts[i], "start") == 0) {
         field->start = strtoul(atts[i + 1], &p, 0);
      } else if (strcmp(atts[i], "end") == 0) {
         field->end = strtoul(atts[i + 1], &p, 0);
      } else if (strcmp(atts[i], "type") == 0) {
         field->type = string_to_type(ctx, atts[i + 1]);
      } else if (strcmp(atts[i], "default") == 0 &&
               field->start >= 16 && field->end <= 31) {
         field->has_default = true;
         field->default_value = strtoul(atts[i + 1], &p, 0);
      }
   }

   return field;
}

static struct gen_value *
create_value(struct parser_context *ctx, const char **atts)
{
   struct gen_value *value = rzalloc(ctx->values, struct gen_value);

   for (int i = 0; atts[i]; i += 2) {
      if (strcmp(atts[i], "name") == 0)
         value->name = ralloc_strdup(value, atts[i + 1]);
      else if (strcmp(atts[i], "value") == 0)
         value->value = strtoul(atts[i + 1], NULL, 0);
   }

   return value;
}

static struct gen_field *
create_and_append_field(struct parser_context *ctx,
                        const char **atts)
{
   struct gen_field *field = create_field(ctx, atts);
   struct gen_field *prev = NULL, *list = ctx->group->fields;

   while (list && field->start > list->start) {
      prev = list;
      list = list->next;
   }

   field->next = list;
   if (prev == NULL)
      ctx->group->fields = field;
   else
      prev->next = field;

   return field;
}

static void
start_element(void *data, const char *element_name, const char **atts)
{
   struct parser_context *ctx = data;
   const char *name = NULL;
   const char *gen = NULL;

   ctx->loc.line_number = XML_GetCurrentLineNumber(ctx->parser);

   for (int i = 0; atts[i]; i += 2) {
      if (strcmp(atts[i], "name") == 0)
         name = atts[i + 1];
      else if (strcmp(atts[i], "gen") == 0)
         gen = atts[i + 1];
   }

   if (strcmp(element_name, "genxml") == 0) {
      if (name == NULL)
         fail(&ctx->loc, "no platform name given");
      if (gen == NULL)
         fail(&ctx->loc, "no gen given");

      int major, minor;
      int n = sscanf(gen, "%d.%d", &major, &minor);
      if (n == 0)
         fail(&ctx->loc, "invalid gen given: %s", gen);
      if (n == 1)
         minor = 0;

      ctx->spec->gen = gen_make_gen(major, minor);
   } else if (strcmp(element_name, "instruction") == 0) {
      ctx->group = create_group(ctx, name, atts, NULL, false);
   } else if (strcmp(element_name, "struct") == 0) {
      ctx->group = create_group(ctx, name, atts, NULL, true);
   } else if (strcmp(element_name, "register") == 0) {
      ctx->group = create_group(ctx, name, atts, NULL, true);
      get_register_offset(atts, &ctx->group->register_offset);
   } else if (strcmp(element_name, "group") == 0) {
      struct gen_group *previous_group = ctx->group;
      while (previous_group->next)
         previous_group = previous_group->next;

      struct gen_group *group = create_group(ctx, "", atts, ctx->group, false);
      previous_group->next = group;
      ctx->group = group;
   } else if (strcmp(element_name, "field") == 0) {
      ctx->last_field = create_and_append_field(ctx, atts);
   } else if (strcmp(element_name, "enum") == 0) {
      ctx->enoom = create_enum(ctx, name, atts);
   } else if (strcmp(element_name, "value") == 0) {
      if (ctx->n_values >= ctx->n_allocated_values) {
         ctx->n_allocated_values = MAX2(2, ctx->n_allocated_values * 2);
         ctx->values = reralloc_array_size(ctx->spec, ctx->values,
                                           sizeof(struct gen_value *),
                                           ctx->n_allocated_values);
      }
      assert(ctx->n_values < ctx->n_allocated_values);
      ctx->values[ctx->n_values++] = create_value(ctx, atts);
   }

}

static void
end_element(void *data, const char *name)
{
   struct parser_context *ctx = data;
   struct gen_spec *spec = ctx->spec;

   if (strcmp(name, "instruction") == 0 ||
       strcmp(name, "struct") == 0 ||
       strcmp(name, "register") == 0) {
      struct gen_group *group = ctx->group;
      struct gen_field *list = group->fields;

      ctx->group = ctx->group->parent;

      while (list && list->end <= 31) {
         if (list->start >= 16 && list->has_default) {
            group->opcode_mask |=
               mask(list->start % 32, list->end % 32);
            group->opcode |= list->default_value << list->start;
         }
         list = list->next;
      }

      if (strcmp(name, "instruction") == 0)
         _mesa_hash_table_insert(spec->commands, group->name, group);
      else if (strcmp(name, "struct") == 0)
         _mesa_hash_table_insert(spec->structs, group->name, group);
      else if (strcmp(name, "register") == 0) {
         _mesa_hash_table_insert(spec->registers_by_name, group->name, group);
         _mesa_hash_table_insert(spec->registers_by_offset,
                                 (void *) (uintptr_t) group->register_offset,
                                 group);
      }
   } else if (strcmp(name, "group") == 0) {
      ctx->group = ctx->group->parent;
   } else if (strcmp(name, "field") == 0) {
      struct gen_field *field = ctx->last_field;
      ctx->last_field = NULL;
      field->inline_enum.values = ctx->values;
      field->inline_enum.nvalues = ctx->n_values;
      ctx->values = ralloc_array(ctx->spec, struct gen_value*, ctx->n_allocated_values = 2);
      ctx->n_values = 0;
   } else if (strcmp(name, "enum") == 0) {
      struct gen_enum *e = ctx->enoom;
      e->values = ctx->values;
      e->nvalues = ctx->n_values;
      ctx->values = ralloc_array(ctx->spec, struct gen_value*, ctx->n_allocated_values = 2);
      ctx->n_values = 0;
      ctx->enoom = NULL;
      _mesa_hash_table_insert(spec->enums, e->name, e);
   }
}

static void
character_data(void *data, const XML_Char *s, int len)
{
}

static int
devinfo_to_gen(const struct gen_device_info *devinfo, bool x10)
{
   if (devinfo->is_baytrail || devinfo->is_haswell) {
      return devinfo->gen * 10 + 5;
   }

   return x10 ? devinfo->gen * 10 : devinfo->gen;
}

static uint32_t zlib_inflate(const void *compressed_data,
                             uint32_t compressed_len,
                             void **out_ptr)
{
   struct z_stream_s zstream;
   void *out;

   memset(&zstream, 0, sizeof(zstream));

   zstream.next_in = (unsigned char *)compressed_data;
   zstream.avail_in = compressed_len;

   if (inflateInit(&zstream) != Z_OK)
      return 0;

   out = malloc(4096);
   zstream.next_out = out;
   zstream.avail_out = 4096;

   do {
      switch (inflate(&zstream, Z_SYNC_FLUSH)) {
      case Z_STREAM_END:
         goto end;
      case Z_OK:
         break;
      default:
         inflateEnd(&zstream);
         return 0;
      }

      if (zstream.avail_out)
         break;

      out = realloc(out, 2*zstream.total_out);
      if (out == NULL) {
         inflateEnd(&zstream);
         return 0;
      }

      zstream.next_out = (unsigned char *)out + zstream.total_out;
      zstream.avail_out = zstream.total_out;
   } while (1);
 end:
   inflateEnd(&zstream);
   *out_ptr = out;
   return zstream.total_out;
}

static uint32_t _hash_uint32(const void *key)
{
   return (uint32_t) (uintptr_t) key;
}

static struct gen_spec *
gen_spec_init(void)
{
   struct gen_spec *spec;
   spec = rzalloc(NULL, struct gen_spec);
   if (spec == NULL)
      return NULL;

   spec->commands =
      _mesa_hash_table_create(spec, _mesa_hash_string, _mesa_key_string_equal);
   spec->structs =
      _mesa_hash_table_create(spec, _mesa_hash_string, _mesa_key_string_equal);
   spec->registers_by_name =
      _mesa_hash_table_create(spec, _mesa_hash_string, _mesa_key_string_equal);
   spec->registers_by_offset =
      _mesa_hash_table_create(spec, _hash_uint32, _mesa_key_pointer_equal);
   spec->enums =
      _mesa_hash_table_create(spec, _mesa_hash_string, _mesa_key_string_equal);
   spec->access_cache =
      _mesa_hash_table_create(spec, _mesa_hash_string, _mesa_key_string_equal);

   return spec;
}

struct gen_spec *
gen_spec_load(const struct gen_device_info *devinfo)
{
   struct parser_context ctx;
   void *buf;
   uint8_t *text_data = NULL;
   uint32_t text_offset = 0, text_length = 0;
   MAYBE_UNUSED uint32_t total_length;
   uint32_t gen_10 = devinfo_to_gen(devinfo, true);

   for (int i = 0; i < ARRAY_SIZE(genxml_files_table); i++) {
      if (genxml_files_table[i].gen_10 == gen_10) {
         text_offset = genxml_files_table[i].offset;
         text_length = genxml_files_table[i].length;
         break;
      }
   }

   if (text_length == 0) {
      fprintf(stderr, "unable to find gen (%u) data\n", gen_10);
      return NULL;
   }

   memset(&ctx, 0, sizeof ctx);
   ctx.parser = XML_ParserCreate(NULL);
   XML_SetUserData(ctx.parser, &ctx);
   if (ctx.parser == NULL) {
      fprintf(stderr, "failed to create parser\n");
      return NULL;
   }

   XML_SetElementHandler(ctx.parser, start_element, end_element);
   XML_SetCharacterDataHandler(ctx.parser, character_data);

   ctx.spec = gen_spec_init();
   if (ctx.spec == NULL) {
      fprintf(stderr, "Failed to create gen_spec\n");
      return NULL;
   }

   total_length = zlib_inflate(compress_genxmls,
                               sizeof(compress_genxmls),
                               (void **) &text_data);
   assert(text_offset + text_length <= total_length);

   buf = XML_GetBuffer(ctx.parser, text_length);
   memcpy(buf, &text_data[text_offset], text_length);

   if (XML_ParseBuffer(ctx.parser, text_length, true) == 0) {
      fprintf(stderr,
              "Error parsing XML at line %ld col %ld byte %ld/%u: %s\n",
              XML_GetCurrentLineNumber(ctx.parser),
              XML_GetCurrentColumnNumber(ctx.parser),
              XML_GetCurrentByteIndex(ctx.parser), text_length,
              XML_ErrorString(XML_GetErrorCode(ctx.parser)));
      XML_ParserFree(ctx.parser);
      free(text_data);
      return NULL;
   }

   XML_ParserFree(ctx.parser);
   free(text_data);

   return ctx.spec;
}

struct gen_spec *
gen_spec_load_from_path(const struct gen_device_info *devinfo,
                        const char *path)
{
   struct parser_context ctx;
   size_t len, filename_len = strlen(path) + 20;
   char *filename = malloc(filename_len);
   void *buf;
   FILE *input;

   len = snprintf(filename, filename_len, "%s/gen%i.xml",
                  path, devinfo_to_gen(devinfo, false));
   assert(len < filename_len);

   input = fopen(filename, "r");
   if (input == NULL) {
      fprintf(stderr, "failed to open xml description\n");
      free(filename);
      return NULL;
   }

   memset(&ctx, 0, sizeof ctx);
   ctx.parser = XML_ParserCreate(NULL);
   XML_SetUserData(ctx.parser, &ctx);
   if (ctx.parser == NULL) {
      fprintf(stderr, "failed to create parser\n");
      fclose(input);
      free(filename);
      return NULL;
   }

   XML_SetElementHandler(ctx.parser, start_element, end_element);
   XML_SetCharacterDataHandler(ctx.parser, character_data);
   ctx.loc.filename = filename;

   ctx.spec = gen_spec_init();
   if (ctx.spec == NULL) {
      fprintf(stderr, "Failed to create gen_spec\n");
      goto end;
   }

   do {
      buf = XML_GetBuffer(ctx.parser, XML_BUFFER_SIZE);
      len = fread(buf, 1, XML_BUFFER_SIZE, input);
      if (ferror(input)) {
         fprintf(stderr, "fread: %m\n");
         gen_spec_destroy(ctx.spec);
         ctx.spec = NULL;
         goto end;
      } else if (feof(input))
         goto end;

      if (XML_ParseBuffer(ctx.parser, len, len == 0) == 0) {
         fprintf(stderr,
                 "Error parsing XML at line %ld col %ld: %s\n",
                 XML_GetCurrentLineNumber(ctx.parser),
                 XML_GetCurrentColumnNumber(ctx.parser),
                 XML_ErrorString(XML_GetErrorCode(ctx.parser)));
         gen_spec_destroy(ctx.spec);
         ctx.spec = NULL;
         goto end;
      }
   } while (len > 0);

 end:
   XML_ParserFree(ctx.parser);

   fclose(input);
   free(filename);

   /* free ctx.spec if genxml is empty */
   if (ctx.spec && _mesa_hash_table_num_entries(ctx.spec->commands) == 0) {
      gen_spec_destroy(ctx.spec);
      return NULL;
   }

   return ctx.spec;
}

void gen_spec_destroy(struct gen_spec *spec)
{
   ralloc_free(spec);
}

struct gen_group *
gen_spec_find_instruction(struct gen_spec *spec, const uint32_t *p)
{
   hash_table_foreach(spec->commands, entry) {
      struct gen_group *command = entry->data;
      uint32_t opcode = *p & command->opcode_mask;
      if (opcode == command->opcode)
         return command;
   }

   return NULL;
}

struct gen_field *
gen_group_find_field(struct gen_group *group, const char *name)
{
   char path[256];
   snprintf(path, sizeof(path), "%s/%s", group->name, name);

   struct gen_spec *spec = group->spec;
   struct hash_entry *entry = _mesa_hash_table_search(spec->access_cache,
                                                      path);
   if (entry)
      return entry->data;

   struct gen_field *field = group->fields;
   while (field) {
      if (strcmp(field->name, name) == 0) {
         _mesa_hash_table_insert(spec->access_cache,
                                 ralloc_strdup(spec, path),
                                 field);
         return field;
      }
      field = field->next;
   }

   return NULL;
}

int
gen_group_get_length(struct gen_group *group, const uint32_t *p)
{
   if (group) {
      if (group->fixed_length)
         return group->dw_length;
      else {
         struct gen_field *field = group->dword_length_field;
         if (field) {
            return field_value(p[0], field->start, field->end) + group->bias;
         }
      }
   }

   uint32_t h = p[0];
   uint32_t type = field_value(h, 29, 31);

   switch (type) {
   case 0: /* MI */ {
      uint32_t opcode = field_value(h, 23, 28);
      if (opcode < 16)
         return 1;
      else
         return field_value(h, 0, 7) + 2;
      break;
   }

   case 2: /* BLT */ {
      return field_value(h, 0, 7) + 2;
   }

   case 3: /* Render */ {
      uint32_t subtype = field_value(h, 27, 28);
      uint32_t opcode = field_value(h, 24, 26);
      uint16_t whole_opcode = field_value(h, 16, 31);
      switch (subtype) {
      case 0:
         if (whole_opcode == 0x6104 /* PIPELINE_SELECT_965 */)
            return 1;
         else if (opcode < 2)
            return field_value(h, 0, 7) + 2;
         else
            return -1;
      case 1:
         if (opcode < 2)
            return 1;
         else
            return -1;
      case 2: {
         if (opcode == 0)
            return field_value(h, 0, 7) + 2;
         else if (opcode < 3)
            return field_value(h, 0, 15) + 2;
         else
            return -1;
      }
      case 3:
         if (whole_opcode == 0x780b)
            return 1;
         else if (opcode < 4)
            return field_value(h, 0, 7) + 2;
         else
            return -1;
      }
   }
   }

   return -1;
}

static const char *
gen_get_enum_name(struct gen_enum *e, uint64_t value)
{
   for (int i = 0; i < e->nvalues; i++) {
      if (e->values[i]->value == value) {
         return e->values[i]->name;
      }
   }
   return NULL;
}

static bool
iter_more_fields(const struct gen_field_iterator *iter)
{
   return iter->field != NULL && iter->field->next != NULL;
}

static uint32_t
iter_group_offset_bits(const struct gen_field_iterator *iter,
                       uint32_t group_iter)
{
   return iter->group->group_offset + (group_iter * iter->group->group_size);
}

static bool
iter_more_groups(const struct gen_field_iterator *iter)
{
   if (iter->group->variable) {
      int length = gen_group_get_length(iter->group, iter->p);
      assert(length >= 0 && "error the length is unknown!");
      return iter_group_offset_bits(iter, iter->group_iter + 1) <
              (length * 32);
   } else {
      return (iter->group_iter + 1) < iter->group->group_count ||
         iter->group->next != NULL;
   }
}

static void
iter_start_field(struct gen_field_iterator *iter, struct gen_field *field)
{
   iter->field = field;

   int group_member_offset = iter_group_offset_bits(iter, iter->group_iter);

   iter->start_bit = group_member_offset + iter->field->start;
   iter->end_bit = group_member_offset + iter->field->end;
   iter->struct_desc = NULL;
}

static void
iter_advance_group(struct gen_field_iterator *iter)
{
   if (iter->group->variable)
      iter->group_iter++;
   else {
      if ((iter->group_iter + 1) < iter->group->group_count) {
         iter->group_iter++;
      } else {
         iter->group = iter->group->next;
         iter->group_iter = 0;
      }
   }

   iter_start_field(iter, iter->group->fields);
}

static bool
iter_advance_field(struct gen_field_iterator *iter)
{
   if (iter_more_fields(iter)) {
      iter_start_field(iter, iter->field->next);
   } else {
      if (!iter_more_groups(iter))
         return false;

      iter_advance_group(iter);
   }
   return true;
}

static bool
iter_decode_field_raw(struct gen_field_iterator *iter, uint64_t *qw)
{
   *qw = 0;

   int field_start = iter->p_bit + iter->start_bit;
   int field_end = iter->p_bit + iter->end_bit;

   const uint32_t *p = iter->p + (iter->start_bit / 32);
   if (iter->p_end && p >= iter->p_end)
      return false;

   if ((field_end - field_start) > 32) {
      if (!iter->p_end || (p + 1) < iter->p_end)
         *qw = ((uint64_t) p[1]) << 32;
      *qw |= p[0];
   } else
      *qw = p[0];

   *qw = field_value(*qw, field_start, field_end);

   /* Address & offset types have to be aligned to dwords, their start bit is
    * a reminder of the alignment requirement.
    */
   if (iter->field->type.kind == GEN_TYPE_ADDRESS ||
       iter->field->type.kind == GEN_TYPE_OFFSET)
      *qw <<= field_start % 32;

   return true;
}

static bool
iter_decode_field(struct gen_field_iterator *iter)
{
   union {
      uint64_t qw;
      float f;
   } v;

   if (iter->field->name)
      snprintf(iter->name, sizeof(iter->name), "%s", iter->field->name);
   else
      memset(iter->name, 0, sizeof(iter->name));

   memset(&v, 0, sizeof(v));

   if (!iter_decode_field_raw(iter, &iter->raw_value))
      return false;

   const char *enum_name = NULL;

   v.qw = iter->raw_value;
   switch (iter->field->type.kind) {
   case GEN_TYPE_UNKNOWN:
   case GEN_TYPE_INT: {
      snprintf(iter->value, sizeof(iter->value), "%"PRId64, v.qw);
      enum_name = gen_get_enum_name(&iter->field->inline_enum, v.qw);
      break;
   }
   case GEN_TYPE_UINT: {
      snprintf(iter->value, sizeof(iter->value), "%"PRIu64, v.qw);
      enum_name = gen_get_enum_name(&iter->field->inline_enum, v.qw);
      break;
   }
   case GEN_TYPE_BOOL: {
      const char *true_string =
         iter->print_colors ? "\e[0;35mtrue\e[0m" : "true";
      snprintf(iter->value, sizeof(iter->value), "%s",
               v.qw ? true_string : "false");
      break;
   }
   case GEN_TYPE_FLOAT:
      snprintf(iter->value, sizeof(iter->value), "%f", v.f);
      break;
   case GEN_TYPE_ADDRESS:
   case GEN_TYPE_OFFSET:
      snprintf(iter->value, sizeof(iter->value), "0x%08"PRIx64, v.qw);
      break;
   case GEN_TYPE_STRUCT:
      snprintf(iter->value, sizeof(iter->value), "<struct %s>",
               iter->field->type.gen_struct->name);
      iter->struct_desc =
         gen_spec_find_struct(iter->group->spec,
                              iter->field->type.gen_struct->name);
      break;
   case GEN_TYPE_UFIXED:
      snprintf(iter->value, sizeof(iter->value), "%f",
               (float) v.qw / (1 << iter->field->type.f));
      break;
   case GEN_TYPE_SFIXED: {
      /* Sign extend before converting */
      int bits = iter->field->type.i + iter->field->type.f + 1;
      int64_t v_sign_extend = ((int64_t)(v.qw << (64 - bits))) >> (64 - bits);
      snprintf(iter->value, sizeof(iter->value), "%f",
               (float) v_sign_extend / (1 << iter->field->type.f));
      break;
   }
   case GEN_TYPE_MBO:
       break;
   case GEN_TYPE_ENUM: {
      snprintf(iter->value, sizeof(iter->value), "%"PRId64, v.qw);
      enum_name = gen_get_enum_name(iter->field->type.gen_enum, v.qw);
      break;
   }
   }

   if (strlen(iter->group->name) == 0) {
      int length = strlen(iter->name);
      snprintf(iter->name + length, sizeof(iter->name) - length,
               "[%i]", iter->group_iter);
   }

   if (enum_name) {
      int length = strlen(iter->value);
      snprintf(iter->value + length, sizeof(iter->value) - length,
               " (%s)", enum_name);
   } else if (strcmp(iter->name, "Surface Format") == 0 ||
              strcmp(iter->name, "Source Element Format") == 0) {
      if (isl_format_is_valid((enum isl_format)v.qw)) {
         const char *fmt_name = isl_format_get_name((enum isl_format)v.qw);
         int length = strlen(iter->value);
         snprintf(iter->value + length, sizeof(iter->value) - length,
                  " (%s)", fmt_name);
      }
   }

   return true;
}

void
gen_field_iterator_init(struct gen_field_iterator *iter,
                        struct gen_group *group,
                        const uint32_t *p, int p_bit,
                        bool print_colors)
{
   memset(iter, 0, sizeof(*iter));

   iter->group = group;
   iter->p = p;
   iter->p_bit = p_bit;

   int length = gen_group_get_length(iter->group, iter->p);
   assert(length >= 0 && "error the length is unknown!");
   iter->p_end = length >= 0 ? &p[length] : NULL;
   iter->print_colors = print_colors;
}

bool
gen_field_iterator_next(struct gen_field_iterator *iter)
{
   /* Initial condition */
   if (!iter->field) {
      if (iter->group->fields)
         iter_start_field(iter, iter->group->fields);
      else
         iter_start_field(iter, iter->group->next->fields);

      bool result = iter_decode_field(iter);
      if (!result && iter->p_end) {
         /* We're dealing with a non empty struct of length=0 (BLEND_STATE on
          * Gen 7.5)
          */
         assert(iter->group->dw_length == 0);
      }

      return result;
   }

   if (!iter_advance_field(iter))
      return false;

   if (!iter_decode_field(iter))
      return false;

   return true;
}

static void
print_dword_header(FILE *outfile,
                   struct gen_field_iterator *iter,
                   uint64_t offset, uint32_t dword)
{
   fprintf(outfile, "0x%08"PRIx64":  0x%08x : Dword %d\n",
           offset + 4 * dword, iter->p[dword], dword);
}

bool
gen_field_is_header(struct gen_field *field)
{
   uint32_t bits;

   if (field->start >= 32)
      return false;

   bits = (1U << (field->end - field->start + 1)) - 1;
   bits <<= field->start;

   return (field->parent->opcode_mask & bits) != 0;
}

void
gen_print_group(FILE *outfile, struct gen_group *group, uint64_t offset,
                const uint32_t *p, int p_bit, bool color)
{
   struct gen_field_iterator iter;
   int last_dword = -1;

   gen_field_iterator_init(&iter, group, p, p_bit, color);
   while (gen_field_iterator_next(&iter)) {
      int iter_dword = iter.end_bit / 32;
      if (last_dword != iter_dword) {
         for (int i = last_dword + 1; i <= iter_dword; i++)
            print_dword_header(outfile, &iter, offset, i);
         last_dword = iter_dword;
      }
      if (!gen_field_is_header(iter.field)) {
         fprintf(outfile, "    %s: %s\n", iter.name, iter.value);
         if (iter.struct_desc) {
            int struct_dword = iter.start_bit / 32;
            uint64_t struct_offset = offset + 4 * struct_dword;
            gen_print_group(outfile, iter.struct_desc, struct_offset,
                            &p[struct_dword], iter.start_bit % 32, color);
         }
      }
   }
}
