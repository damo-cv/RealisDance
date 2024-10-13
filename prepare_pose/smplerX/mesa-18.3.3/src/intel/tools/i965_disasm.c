/*
 * Copyright © 2018 Intel Corporation
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
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "compiler/brw_eu.h"
#include "dev/gen_device_info.h"

uint64_t INTEL_DEBUG;

/* Return size of file in bytes pointed by fp */
static size_t
i965_disasm_get_file_size(FILE *fp)
{
   size_t size;

   fseek(fp, 0L, SEEK_END);
   size = ftell(fp);
   fseek(fp, 0L, SEEK_SET);

   return size;
}

static void *
i965_disasm_read_binary(FILE *fp, size_t *end)
{
   void *assembly;

   *end = i965_disasm_get_file_size(fp);

   assembly = malloc(*end + 1);
   if (assembly == NULL)
      return NULL;

   fread(assembly, *end, 1, fp);
   fclose(fp);

   return assembly;
}

static struct gen_device_info *
i965_disasm_init(uint16_t pci_id)
{
   struct gen_device_info *devinfo;

   devinfo = malloc(sizeof *devinfo);
   if (devinfo == NULL)
      return NULL;

   if (!gen_get_device_info(pci_id, devinfo)) {
      fprintf(stderr, "can't find device information: pci_id=0x%x\n",
              pci_id);
      exit(EXIT_FAILURE);
   }

   /* initialize compaction table in order to handle compacted instructions */
   brw_init_compaction_tables(devinfo);

   return devinfo;
}

static void
print_help(const char *progname, FILE *file)
{
   fprintf(file,
           "Usage: %s [OPTION]...\n"
           "Disassemble i965 instructions from binary file.\n\n"
           "      --help             display this help and exit\n"
           "      --binary-path=PATH read binary file from binary file PATH\n"
           "      --gen=platform     disassemble instructions for given \n"
           "                         platform (3 letter platform name)\n",
           progname);
}

int main(int argc, char *argv[])
{
   FILE *fp = NULL;
   void *assembly = NULL;
   char *binary_path = NULL;
   size_t start = 0, end = 0;
   uint16_t pci_id = 0;
   int c, i;
   struct gen_device_info *devinfo;

   bool help = false;
   const struct option i965_disasm_opts[] = {
      { "help",          no_argument,       (int *) &help,      true },
      { "binary-path",   required_argument, NULL,               'b' },
      { "gen",           required_argument, NULL,               'g'},
      { NULL,            0,                 NULL,                0 }
   };

   i = 0;
   while ((c = getopt_long(argc, argv, "", i965_disasm_opts, &i)) != -1) {
      switch (c) {
      case 'g': {
         const int id = gen_device_name_to_pci_device_id(optarg);
         if (id < 0) {
            fprintf(stderr, "can't parse gen: '%s', expected 3 letter "
                            "platform name\n", optarg);
            /* Clean up binary path if given pci id is wrong */
            if (binary_path) {
               free(binary_path);
               fclose(fp);
            }
            exit(EXIT_FAILURE);
         } else {
            pci_id = id;
         }
         break;
      }
      case 'b':
         binary_path = strdup(optarg);
         fp = fopen(binary_path, "rb");
         if (!fp) {
            fprintf(stderr, "Unable to read input binary file : %s\n",
                    binary_path);
            /* free binary_path if path is wrong */
            free(binary_path);
            exit(EXIT_FAILURE);
         }
         break;
      default:
         /* Clean up binary path if given option is wrong */
         if (binary_path) {
            free(binary_path);
            fclose(fp);
         }
         break;
      }
   }

   if (help || !binary_path || !pci_id) {
      print_help(argv[0], stderr);
      exit(0);
   }

   devinfo = i965_disasm_init(pci_id);
   if (!devinfo) {
      fprintf(stderr, "Unable to allocate memory for "
                      "gen_device_info struct instance.\n");
      exit(EXIT_FAILURE);
   }

   assembly = i965_disasm_read_binary(fp, &end);
   if (!assembly) {
      fprintf(stderr, "Unable to allocate buffer to read binary file\n");
      exit(EXIT_FAILURE);
   }

   /* Disassemble i965 instructions from buffer assembly */
   brw_disassemble(devinfo, assembly, start, end, stdout);

   free(binary_path);
   free(assembly);
   free(devinfo);

   return EXIT_SUCCESS;
}
