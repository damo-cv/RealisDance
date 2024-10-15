/*
 * Copyright (C) 2014 Rob Clark <robclark@freedesktop.org>
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <err.h>

#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_text.h"
#include "tgsi/tgsi_dump.h"

#include "freedreno_util.h"

#include "ir3_compiler.h"
#include "ir3_nir.h"
#include "instr-a3xx.h"
#include "ir3.h"

#include "compiler/glsl/standalone.h"
#include "compiler/glsl/glsl_to_nir.h"
#include "compiler/glsl/gl_nir.h"
#include "compiler/nir_types.h"
#include "compiler/spirv/nir_spirv.h"

static void dump_info(struct ir3_shader_variant *so, const char *str)
{
	uint32_t *bin;
	const char *type = ir3_shader_stage(so->shader);
	bin = ir3_shader_assemble(so, so->shader->compiler->gpu_id);
	debug_printf("; %s: %s\n", type, str);
	ir3_shader_disasm(so, bin, stdout);
	free(bin);
}

static void
insert_sorted(struct exec_list *var_list, nir_variable *new_var)
{
	nir_foreach_variable(var, var_list) {
		if (var->data.location > new_var->data.location) {
			exec_node_insert_node_before(&var->node, &new_var->node);
			return;
		}
	}
	exec_list_push_tail(var_list, &new_var->node);
}

static void
sort_varyings(struct exec_list *var_list)
{
	struct exec_list new_list;
	exec_list_make_empty(&new_list);
	nir_foreach_variable_safe(var, var_list) {
		exec_node_remove(&var->node);
		insert_sorted(&new_list, var);
	}
	exec_list_move_nodes_to(&new_list, var_list);
}

static void
fixup_varying_slots(struct exec_list *var_list)
{
	nir_foreach_variable(var, var_list) {
		if (var->data.location >= VARYING_SLOT_VAR0) {
			var->data.location += 9;
		} else if ((var->data.location >= VARYING_SLOT_TEX0) &&
				(var->data.location <= VARYING_SLOT_TEX7)) {
			var->data.location += VARYING_SLOT_VAR0 - VARYING_SLOT_TEX0;
		}
	}
}

static struct ir3_compiler *compiler;

static nir_shader *
load_glsl(unsigned num_files, char* const* files, gl_shader_stage stage)
{
	static const struct standalone_options options = {
			.glsl_version = 140,
			.do_link = true,
	};
	struct gl_shader_program *prog;

	prog = standalone_compile_shader(&options, num_files, files);
	if (!prog)
		errx(1, "couldn't parse `%s'", files[0]);

	nir_shader *nir = glsl_to_nir(prog, stage, ir3_get_compiler_options(compiler));

	/* required NIR passes: */
	/* TODO cmdline args for some of the conditional lowering passes? */

	NIR_PASS_V(nir, nir_lower_io_to_temporaries,
			nir_shader_get_entrypoint(nir),
			true, true);
	NIR_PASS_V(nir, nir_lower_global_vars_to_local);
	NIR_PASS_V(nir, nir_split_var_copies);
	NIR_PASS_V(nir, nir_lower_var_copies);

	NIR_PASS_V(nir, nir_split_var_copies);
	NIR_PASS_V(nir, nir_lower_var_copies);

	switch (stage) {
	case MESA_SHADER_VERTEX:
		nir_assign_var_locations(&nir->inputs,
				&nir->num_inputs,
				ir3_glsl_type_size);

		/* Re-lower global vars, to deal with any dead VS inputs. */
		NIR_PASS_V(nir, nir_lower_global_vars_to_local);

		sort_varyings(&nir->outputs);
		nir_assign_var_locations(&nir->outputs,
				&nir->num_outputs,
				ir3_glsl_type_size);
		fixup_varying_slots(&nir->outputs);
		break;
	case MESA_SHADER_FRAGMENT:
		sort_varyings(&nir->inputs);
		nir_assign_var_locations(&nir->inputs,
				&nir->num_inputs,
				ir3_glsl_type_size);
		fixup_varying_slots(&nir->inputs);
		nir_assign_var_locations(&nir->outputs,
				&nir->num_outputs,
				ir3_glsl_type_size);
		break;
	default:
		errx(1, "unhandled shader stage: %d", stage);
	}

	nir_assign_var_locations(&nir->uniforms,
			&nir->num_uniforms,
			ir3_glsl_type_size);

	NIR_PASS_V(nir, nir_lower_system_values);
	NIR_PASS_V(nir, nir_lower_io, nir_var_all, ir3_glsl_type_size, 0);
	NIR_PASS_V(nir, gl_nir_lower_samplers, prog);

	return nir;
}

static int
read_file(const char *filename, void **ptr, size_t *size)
{
	int fd, ret;
	struct stat st;

	*ptr = MAP_FAILED;

	fd = open(filename, O_RDONLY);
	if (fd == -1) {
		warnx("couldn't open `%s'", filename);
		return 1;
	}

	ret = fstat(fd, &st);
	if (ret)
		errx(1, "couldn't stat `%s'", filename);

	*size = st.st_size;
	*ptr = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if (*ptr == MAP_FAILED)
		errx(1, "couldn't map `%s'", filename);

	close(fd);

	return 0;
}

static void debug_func(void *priv, enum nir_spirv_debug_level level,
		size_t spirv_offset, const char *message)
{
//	printf("%s\n", message);
}

static nir_shader *
load_spirv(const char *filename, const char *entry, gl_shader_stage stage)
{
	const struct spirv_to_nir_options spirv_options = {
		/* these caps are just make-believe */
		.caps = {
			.draw_parameters = true,
			.float64 = true,
			.image_read_without_format = true,
			.image_write_without_format = true,
			.int64 = true,
			.variable_pointers = true,
		},
		.lower_workgroup_access_to_offsets = true,
		.debug = {
			.func = debug_func,
		}
	};
	nir_function *entry_point;
	void *buf;
	size_t size;

	read_file(filename, &buf, &size);

	entry_point = spirv_to_nir(buf, size / 4,
			NULL, 0, /* spec_entries */
			stage, entry,
			&spirv_options,
			ir3_get_compiler_options(compiler));

	nir_print_shader(entry_point->shader, stdout);

	return entry_point->shader;
}

static void print_usage(void)
{
	printf("Usage: ir3_compiler [OPTIONS]... <file.tgsi | file.spv entry_point | (file.vert | file.frag)*>\n");
	printf("    --verbose         - verbose compiler/debug messages\n");
	printf("    --binning-pass    - generate binning pass shader (VERT)\n");
	printf("    --color-two-side  - emulate two-sided color (FRAG)\n");
	printf("    --half-precision  - use half-precision\n");
	printf("    --saturate-s MASK - bitmask of samplers to saturate S coord\n");
	printf("    --saturate-t MASK - bitmask of samplers to saturate T coord\n");
	printf("    --saturate-r MASK - bitmask of samplers to saturate R coord\n");
	printf("    --astc-srgb MASK  - bitmask of samplers to enable astc-srgb workaround\n");
	printf("    --stream-out      - enable stream-out (aka transform feedback)\n");
	printf("    --ucp MASK        - bitmask of enabled user-clip-planes\n");
	printf("    --gpu GPU_ID      - specify gpu-id (default 320)\n");
	printf("    --help            - show this message\n");
}

int main(int argc, char **argv)
{
	int ret = 0, n = 1;
	char *filenames[2];
	int num_files = 0;
	unsigned stage = 0;
	struct ir3_shader_variant v;
	struct ir3_shader s;
	struct ir3_shader_key key = {};
	/* TODO cmdline option to target different gpus: */
	unsigned gpu_id = 320;
	const char *info;
	const char *entry;
	void *ptr;
	bool from_spirv = false;
	size_t size;

	memset(&s, 0, sizeof(s));
	memset(&v, 0, sizeof(v));

	/* cmdline args which impact shader variant get spit out in a
	 * comment on the first line..  a quick/dirty way to preserve
	 * that info so when ir3test recompiles the shader with a new
	 * compiler version, we use the same shader-key settings:
	 */
	debug_printf("; options:");

	while (n < argc) {
		if (!strcmp(argv[n], "--verbose")) {
			fd_mesa_debug |= FD_DBG_MSGS | FD_DBG_OPTMSGS | FD_DBG_DISASM;
			n++;
			continue;
		}

		if (!strcmp(argv[n], "--binning-pass")) {
			debug_printf(" %s", argv[n]);
			v.binning_pass = true;
			n++;
			continue;
		}

		if (!strcmp(argv[n], "--color-two-side")) {
			debug_printf(" %s", argv[n]);
			key.color_two_side = true;
			n++;
			continue;
		}

		if (!strcmp(argv[n], "--half-precision")) {
			debug_printf(" %s", argv[n]);
			key.half_precision = true;
			n++;
			continue;
		}

		if (!strcmp(argv[n], "--saturate-s")) {
			debug_printf(" %s %s", argv[n], argv[n+1]);
			key.vsaturate_s = key.fsaturate_s = strtol(argv[n+1], NULL, 0);
			n += 2;
			continue;
		}

		if (!strcmp(argv[n], "--saturate-t")) {
			debug_printf(" %s %s", argv[n], argv[n+1]);
			key.vsaturate_t = key.fsaturate_t = strtol(argv[n+1], NULL, 0);
			n += 2;
			continue;
		}

		if (!strcmp(argv[n], "--saturate-r")) {
			debug_printf(" %s %s", argv[n], argv[n+1]);
			key.vsaturate_r = key.fsaturate_r = strtol(argv[n+1], NULL, 0);
			n += 2;
			continue;
		}

		if (!strcmp(argv[n], "--astc-srgb")) {
			debug_printf(" %s %s", argv[n], argv[n+1]);
			key.vastc_srgb = key.fastc_srgb = strtol(argv[n+1], NULL, 0);
			n += 2;
			continue;
		}

		if (!strcmp(argv[n], "--stream-out")) {
			struct pipe_stream_output_info *so = &s.stream_output;
			debug_printf(" %s", argv[n]);
			/* TODO more dynamic config based on number of outputs, etc
			 * rather than just hard-code for first output:
			 */
			so->num_outputs = 1;
			so->stride[0] = 4;
			so->output[0].register_index = 0;
			so->output[0].start_component = 0;
			so->output[0].num_components = 4;
			so->output[0].output_buffer = 0;
			so->output[0].dst_offset = 2;
			so->output[0].stream = 0;
			n++;
			continue;
		}

		if (!strcmp(argv[n], "--ucp")) {
			debug_printf(" %s %s", argv[n], argv[n+1]);
			key.ucp_enables = strtol(argv[n+1], NULL, 0);
			n += 2;
			continue;
		}

		if (!strcmp(argv[n], "--gpu")) {
			debug_printf(" %s %s", argv[n], argv[n+1]);
			gpu_id = strtol(argv[n+1], NULL, 0);
			n += 2;
			continue;
		}

		if (!strcmp(argv[n], "--help")) {
			print_usage();
			return 0;
		}

		break;
	}
	debug_printf("\n");

	while (n < argc) {
		char *filename = argv[n];
		char *ext = strrchr(filename, '.');

		if (strcmp(ext, ".tgsi") == 0) {
			if (num_files != 0)
				errx(1, "in TGSI mode, only a single file may be specified");
			s.from_tgsi = true;
		} else if (strcmp(ext, ".spv") == 0) {
			if (num_files != 0)
				errx(1, "in SPIR-V mode, only a single file may be specified");
			stage = MESA_SHADER_COMPUTE;
			from_spirv = true;
			filenames[num_files++] = filename;
			n++;
			if (n == argc)
				errx(1, "in SPIR-V mode, an entry point must be specified");
			entry = argv[n];
			n++;
		} else if (strcmp(ext, ".frag") == 0) {
			if (s.from_tgsi || from_spirv)
				errx(1, "cannot mix GLSL/TGSI/SPIRV");
			if (num_files >= ARRAY_SIZE(filenames))
				errx(1, "too many GLSL files");
			stage = MESA_SHADER_FRAGMENT;
		} else if (strcmp(ext, ".vert") == 0) {
			if (s.from_tgsi)
				errx(1, "cannot mix GLSL and TGSI");
			if (num_files >= ARRAY_SIZE(filenames))
				errx(1, "too many GLSL files");
			stage = MESA_SHADER_VERTEX;
		} else {
			print_usage();
			return -1;
		}

		filenames[num_files++] = filename;

		n++;
	}

	nir_shader *nir;

	compiler = ir3_compiler_create(NULL, gpu_id);

	if (s.from_tgsi) {
		struct tgsi_token toks[65536];

		ret = read_file(filenames[0], &ptr, &size);
		if (ret) {
			print_usage();
			return ret;
		}

		if (fd_mesa_debug & FD_DBG_OPTMSGS)
			debug_printf("%s\n", (char *)ptr);

		if (!tgsi_text_translate(ptr, toks, ARRAY_SIZE(toks)))
			errx(1, "could not parse `%s'", filenames[0]);

		if (fd_mesa_debug & FD_DBG_OPTMSGS)
			tgsi_dump(toks, 0);

		nir = ir3_tgsi_to_nir(toks);
		NIR_PASS_V(nir, nir_lower_global_vars_to_local);
	} else if (from_spirv) {
		nir = load_spirv(filenames[0], entry, stage);

		NIR_PASS_V(nir, nir_lower_io, nir_var_all, ir3_glsl_type_size,
				(nir_lower_io_options)0);

		/* TODO do this somewhere else */
		nir_lower_int64(nir, ~0);
		nir_lower_system_values(nir);
	} else if (num_files > 0) {
		nir = load_glsl(num_files, filenames, stage);
	} else {
		print_usage();
		return -1;
	}

	s.compiler = compiler;
	s.nir = ir3_optimize_nir(&s, nir, NULL);

	v.key = key;
	v.shader = &s;

	switch (nir->info.stage) {
	case MESA_SHADER_FRAGMENT:
		s.type = v.type = SHADER_FRAGMENT;
		break;
	case MESA_SHADER_VERTEX:
		s.type = v.type = SHADER_VERTEX;
		break;
	case MESA_SHADER_COMPUTE:
		s.type = v.type = SHADER_COMPUTE;
		break;
	default:
		errx(1, "unhandled shader stage: %d", nir->info.stage);
	}

	info = "NIR compiler";
	ret = ir3_compile_shader_nir(s.compiler, &v);
	if (ret) {
		fprintf(stderr, "compiler failed!\n");
		return ret;
	}
	dump_info(&v, info);
}
