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

#include "pipe/p_state.h"
#include "util/u_string.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "util/u_format.h"
#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_parse.h"

#include "freedreno_context.h"
#include "freedreno_util.h"

#include "ir3_shader.h"
#include "ir3_compiler.h"
#include "ir3_nir.h"

int
ir3_glsl_type_size(const struct glsl_type *type)
{
	return glsl_count_attribute_slots(type, false);
}

static void
delete_variant(struct ir3_shader_variant *v)
{
	if (v->ir)
		ir3_destroy(v->ir);
	if (v->bo)
		fd_bo_del(v->bo);
	if (v->immediates)
		free(v->immediates);
	free(v);
}

/* for vertex shader, the inputs are loaded into registers before the shader
 * is executed, so max_regs from the shader instructions might not properly
 * reflect the # of registers actually used, especially in case passthrough
 * varyings.
 *
 * Likewise, for fragment shader, we can have some regs which are passed
 * input values but never touched by the resulting shader (ie. as result
 * of dead code elimination or simply because we don't know how to turn
 * the reg off.
 */
static void
fixup_regfootprint(struct ir3_shader_variant *v)
{
	unsigned i;

	for (i = 0; i < v->inputs_count; i++) {
		/* skip frag inputs fetch via bary.f since their reg's are
		 * not written by gpu before shader starts (and in fact the
		 * regid's might not even be valid)
		 */
		if (v->inputs[i].bary)
			continue;

		/* ignore high regs that are global to all threads in a warp
		 * (they exist by default) (a5xx+)
		 */
		if (v->inputs[i].regid >= regid(48,0))
			continue;

		if (v->inputs[i].compmask) {
			unsigned n = util_last_bit(v->inputs[i].compmask) - 1;
			int32_t regid = (v->inputs[i].regid + n) >> 2;
			v->info.max_reg = MAX2(v->info.max_reg, regid);
		}
	}

	for (i = 0; i < v->outputs_count; i++) {
		int32_t regid = (v->outputs[i].regid + 3) >> 2;
		v->info.max_reg = MAX2(v->info.max_reg, regid);
	}
}

/* wrapper for ir3_assemble() which does some info fixup based on
 * shader state.  Non-static since used by ir3_cmdline too.
 */
void * ir3_shader_assemble(struct ir3_shader_variant *v, uint32_t gpu_id)
{
	void *bin;

	bin = ir3_assemble(v->ir, &v->info, gpu_id);
	if (!bin)
		return NULL;

	if (gpu_id >= 400) {
		v->instrlen = v->info.sizedwords / (2 * 16);
	} else {
		v->instrlen = v->info.sizedwords / (2 * 4);
	}

	/* NOTE: if relative addressing is used, we set constlen in
	 * the compiler (to worst-case value) since we don't know in
	 * the assembler what the max addr reg value can be:
	 */
	v->constlen = MIN2(255, MAX2(v->constlen, v->info.max_const + 1));

	fixup_regfootprint(v);

	return bin;
}

static void
assemble_variant(struct ir3_shader_variant *v)
{
	struct ir3_compiler *compiler = v->shader->compiler;
	uint32_t gpu_id = compiler->gpu_id;
	uint32_t sz, *bin;

	bin = ir3_shader_assemble(v, gpu_id);
	sz = v->info.sizedwords * 4;

	v->bo = fd_bo_new(compiler->dev, sz,
			DRM_FREEDRENO_GEM_CACHE_WCOMBINE |
			DRM_FREEDRENO_GEM_TYPE_KMEM);

	memcpy(fd_bo_map(v->bo), bin, sz);

	if (fd_mesa_debug & FD_DBG_DISASM) {
		struct ir3_shader_key key = v->key;
		printf("disassemble: type=%d, k={bp=%u,cts=%u,hp=%u}", v->type,
			v->binning_pass, key.color_two_side, key.half_precision);
		ir3_shader_disasm(v, bin, stdout);
	}

	if (shader_debug_enabled(v->shader->type)) {
		fprintf(stderr, "Native code for unnamed %s shader %s:\n",
			shader_stage_name(v->shader->type), v->shader->nir->info.name);
		if (v->shader->type == SHADER_FRAGMENT)
			fprintf(stderr, "SIMD0\n");
		ir3_shader_disasm(v, bin, stderr);
	}

	free(bin);

	/* no need to keep the ir around beyond this point: */
	ir3_destroy(v->ir);
	v->ir = NULL;
}

static void
dump_shader_info(struct ir3_shader_variant *v, struct pipe_debug_callback *debug)
{
	if (!unlikely(fd_mesa_debug & FD_DBG_SHADERDB))
		return;

	pipe_debug_message(debug, SHADER_INFO, "\n"
			"SHADER-DB: %s prog %d/%d: %u instructions, %u dwords\n"
			"SHADER-DB: %s prog %d/%d: %u half, %u full\n"
			"SHADER-DB: %s prog %d/%d: %u const, %u constlen\n"
			"SHADER-DB: %s prog %d/%d: %u (ss), %u (sy)\n",
			ir3_shader_stage(v->shader),
			v->shader->id, v->id,
			v->info.instrs_count,
			v->info.sizedwords,
			ir3_shader_stage(v->shader),
			v->shader->id, v->id,
			v->info.max_half_reg + 1,
			v->info.max_reg + 1,
			ir3_shader_stage(v->shader),
			v->shader->id, v->id,
			v->info.max_const + 1,
			v->constlen,
			ir3_shader_stage(v->shader),
			v->shader->id, v->id,
			v->info.ss, v->info.sy);
}

static struct ir3_shader_variant *
create_variant(struct ir3_shader *shader, struct ir3_shader_key key,
		bool binning_pass)
{
	struct ir3_shader_variant *v = CALLOC_STRUCT(ir3_shader_variant);
	int ret;

	if (!v)
		return NULL;

	v->id = ++shader->variant_count;
	v->shader = shader;
	v->binning_pass = binning_pass;
	v->key = key;
	v->type = shader->type;

	ret = ir3_compile_shader_nir(shader->compiler, v);
	if (ret) {
		debug_error("compile failed!");
		goto fail;
	}

	assemble_variant(v);
	if (!v->bo) {
		debug_error("assemble failed!");
		goto fail;
	}

	return v;

fail:
	delete_variant(v);
	return NULL;
}

static inline struct ir3_shader_variant *
shader_variant(struct ir3_shader *shader, struct ir3_shader_key key,
		struct pipe_debug_callback *debug)
{
	struct ir3_shader_variant *v;

	/* some shader key values only apply to vertex or frag shader,
	 * so normalize the key to avoid constructing multiple identical
	 * variants:
	 */
	switch (shader->type) {
	case SHADER_FRAGMENT:
		if (key.has_per_samp) {
			key.vsaturate_s = 0;
			key.vsaturate_t = 0;
			key.vsaturate_r = 0;
			key.vastc_srgb = 0;
			key.vsamples = 0;
		}
		break;
	case SHADER_VERTEX:
		key.color_two_side = false;
		key.half_precision = false;
		key.rasterflat = false;
		if (key.has_per_samp) {
			key.fsaturate_s = 0;
			key.fsaturate_t = 0;
			key.fsaturate_r = 0;
			key.fastc_srgb = 0;
			key.fsamples = 0;
		}
		break;
	default:
		/* TODO */
		break;
	}

	for (v = shader->variants; v; v = v->next)
		if (ir3_shader_key_equal(&key, &v->key))
			return v;

	/* compile new variant if it doesn't exist already: */
	v = create_variant(shader, key, false);
	if (v) {
		v->next = shader->variants;
		shader->variants = v;
		dump_shader_info(v, debug);
	}

	return v;
}


struct ir3_shader_variant *
ir3_shader_variant(struct ir3_shader *shader, struct ir3_shader_key key,
		bool binning_pass, struct pipe_debug_callback *debug)
{
	struct ir3_shader_variant *v =
			shader_variant(shader, key, debug);

	if (binning_pass) {
		if (!v->binning)
			v->binning = create_variant(shader, key, true);
		return v->binning;
	}

	return v;
}

void
ir3_shader_destroy(struct ir3_shader *shader)
{
	struct ir3_shader_variant *v, *t;
	for (v = shader->variants; v; ) {
		t = v;
		v = v->next;
		delete_variant(t);
	}
	ralloc_free(shader->nir);
	free(shader);
}

struct ir3_shader *
ir3_shader_create(struct ir3_compiler *compiler,
		const struct pipe_shader_state *cso, enum shader_t type,
		struct pipe_debug_callback *debug)
{
	struct ir3_shader *shader = CALLOC_STRUCT(ir3_shader);
	shader->compiler = compiler;
	shader->id = ++shader->compiler->shader_count;
	shader->type = type;

	nir_shader *nir;
	if (cso->type == PIPE_SHADER_IR_NIR) {
		/* we take ownership of the reference: */
		nir = cso->ir.nir;
	} else {
		debug_assert(cso->type == PIPE_SHADER_IR_TGSI);
		if (fd_mesa_debug & FD_DBG_DISASM) {
			DBG("dump tgsi: type=%d", shader->type);
			tgsi_dump(cso->tokens, 0);
		}
		nir = ir3_tgsi_to_nir(cso->tokens);
	}
	NIR_PASS_V(nir, nir_lower_io, nir_var_all, ir3_glsl_type_size,
			   (nir_lower_io_options)0);
	/* do first pass optimization, ignoring the key: */
	shader->nir = ir3_optimize_nir(shader, nir, NULL);
	if (fd_mesa_debug & FD_DBG_DISASM) {
		DBG("dump nir%d: type=%d", shader->id, shader->type);
		nir_print_shader(shader->nir, stdout);
	}

	shader->stream_output = cso->stream_output;
	if (fd_mesa_debug & FD_DBG_SHADERDB) {
		/* if shader-db run, create a standard variant immediately
		 * (as otherwise nothing will trigger the shader to be
		 * actually compiled)
		 */
		static struct ir3_shader_key key;
		memset(&key, 0, sizeof(key));
		ir3_shader_variant(shader, key, false, debug);
	}
	return shader;
}

/* a bit annoying that compute-shader and normal shader state objects
 * aren't a bit more aligned.
 */
struct ir3_shader *
ir3_shader_create_compute(struct ir3_compiler *compiler,
		const struct pipe_compute_state *cso,
		struct pipe_debug_callback *debug)
{
	struct ir3_shader *shader = CALLOC_STRUCT(ir3_shader);

	shader->compiler = compiler;
	shader->id = ++shader->compiler->shader_count;
	shader->type = SHADER_COMPUTE;

	nir_shader *nir;
	if (cso->ir_type == PIPE_SHADER_IR_NIR) {
		/* we take ownership of the reference: */
		nir = (nir_shader *)cso->prog;

		NIR_PASS_V(nir, nir_lower_io, nir_var_all, ir3_glsl_type_size,
			   (nir_lower_io_options)0);
	} else {
		debug_assert(cso->ir_type == PIPE_SHADER_IR_TGSI);
		if (fd_mesa_debug & FD_DBG_DISASM) {
			DBG("dump tgsi: type=%d", shader->type);
			tgsi_dump(cso->prog, 0);
		}
		nir = ir3_tgsi_to_nir(cso->prog);
	}

	/* do first pass optimization, ignoring the key: */
	shader->nir = ir3_optimize_nir(shader, nir, NULL);
	if (fd_mesa_debug & FD_DBG_DISASM) {
		printf("dump nir%d: type=%d\n", shader->id, shader->type);
		nir_print_shader(shader->nir, stdout);
	}

	return shader;
}

static void dump_reg(FILE *out, const char *name, uint32_t r)
{
	if (r != regid(63,0))
		fprintf(out, "; %s: r%d.%c\n", name, r >> 2, "xyzw"[r & 0x3]);
}

static void dump_output(FILE *out, struct ir3_shader_variant *so,
		unsigned slot, const char *name)
{
	uint32_t regid;
	regid = ir3_find_output_regid(so, slot);
	dump_reg(out, name, regid);
}

void
ir3_shader_disasm(struct ir3_shader_variant *so, uint32_t *bin, FILE *out)
{
	struct ir3 *ir = so->ir;
	struct ir3_register *reg;
	const char *type = ir3_shader_stage(so->shader);
	uint8_t regid;
	unsigned i;

	for (i = 0; i < ir->ninputs; i++) {
		if (!ir->inputs[i]) {
			fprintf(out, "; in%d unused\n", i);
			continue;
		}
		reg = ir->inputs[i]->regs[0];
		regid = reg->num;
		fprintf(out, "@in(%sr%d.%c)\tin%d\n",
				(reg->flags & IR3_REG_HALF) ? "h" : "",
				(regid >> 2), "xyzw"[regid & 0x3], i);
	}

	for (i = 0; i < ir->noutputs; i++) {
		if (!ir->outputs[i]) {
			fprintf(out, "; out%d unused\n", i);
			continue;
		}
		/* kill shows up as a virtual output.. skip it! */
		if (is_kill(ir->outputs[i]))
			continue;
		reg = ir->outputs[i]->regs[0];
		regid = reg->num;
		fprintf(out, "@out(%sr%d.%c)\tout%d\n",
				(reg->flags & IR3_REG_HALF) ? "h" : "",
				(regid >> 2), "xyzw"[regid & 0x3], i);
	}

	for (i = 0; i < so->immediates_count; i++) {
		fprintf(out, "@const(c%d.x)\t", so->constbase.immediate + i);
		fprintf(out, "0x%08x, 0x%08x, 0x%08x, 0x%08x\n",
				so->immediates[i].val[0],
				so->immediates[i].val[1],
				so->immediates[i].val[2],
				so->immediates[i].val[3]);
	}

	disasm_a3xx(bin, so->info.sizedwords, 0, out);

	switch (so->type) {
	case SHADER_VERTEX:
		fprintf(out, "; %s: outputs:", type);
		for (i = 0; i < so->outputs_count; i++) {
			uint8_t regid = so->outputs[i].regid;
			fprintf(out, " r%d.%c (%s)",
					(regid >> 2), "xyzw"[regid & 0x3],
					gl_varying_slot_name(so->outputs[i].slot));
		}
		fprintf(out, "\n");
		fprintf(out, "; %s: inputs:", type);
		for (i = 0; i < so->inputs_count; i++) {
			uint8_t regid = so->inputs[i].regid;
			fprintf(out, " r%d.%c (cm=%x,il=%u,b=%u)",
					(regid >> 2), "xyzw"[regid & 0x3],
					so->inputs[i].compmask,
					so->inputs[i].inloc,
					so->inputs[i].bary);
		}
		fprintf(out, "\n");
		break;
	case SHADER_FRAGMENT:
		fprintf(out, "; %s: outputs:", type);
		for (i = 0; i < so->outputs_count; i++) {
			uint8_t regid = so->outputs[i].regid;
			fprintf(out, " r%d.%c (%s)",
					(regid >> 2), "xyzw"[regid & 0x3],
					gl_frag_result_name(so->outputs[i].slot));
		}
		fprintf(out, "\n");
		fprintf(out, "; %s: inputs:", type);
		for (i = 0; i < so->inputs_count; i++) {
			uint8_t regid = so->inputs[i].regid;
			fprintf(out, " r%d.%c (%s,cm=%x,il=%u,b=%u)",
					(regid >> 2), "xyzw"[regid & 0x3],
					gl_varying_slot_name(so->inputs[i].slot),
					so->inputs[i].compmask,
					so->inputs[i].inloc,
					so->inputs[i].bary);
		}
		fprintf(out, "\n");
		break;
	default:
		/* TODO */
		break;
	}

	/* print generic shader info: */
	fprintf(out, "; %s prog %d/%d: %u instructions, %d half, %d full\n",
			type, so->shader->id, so->id,
			so->info.instrs_count,
			so->info.max_half_reg + 1,
			so->info.max_reg + 1);

	fprintf(out, "; %d const, %u constlen\n",
			so->info.max_const + 1,
			so->constlen);

	fprintf(out, "; %u (ss), %u (sy)\n", so->info.ss, so->info.sy);

	/* print shader type specific info: */
	switch (so->type) {
	case SHADER_VERTEX:
		dump_output(out, so, VARYING_SLOT_POS, "pos");
		dump_output(out, so, VARYING_SLOT_PSIZ, "psize");
		break;
	case SHADER_FRAGMENT:
		dump_reg(out, "pos (bary)",
			ir3_find_sysval_regid(so, SYSTEM_VALUE_VARYING_COORD));
		dump_output(out, so, FRAG_RESULT_DEPTH, "posz");
		if (so->color0_mrt) {
			dump_output(out, so, FRAG_RESULT_COLOR, "color");
		} else {
			dump_output(out, so, FRAG_RESULT_DATA0, "data0");
			dump_output(out, so, FRAG_RESULT_DATA1, "data1");
			dump_output(out, so, FRAG_RESULT_DATA2, "data2");
			dump_output(out, so, FRAG_RESULT_DATA3, "data3");
			dump_output(out, so, FRAG_RESULT_DATA4, "data4");
			dump_output(out, so, FRAG_RESULT_DATA5, "data5");
			dump_output(out, so, FRAG_RESULT_DATA6, "data6");
			dump_output(out, so, FRAG_RESULT_DATA7, "data7");
		}
		/* these two are hard-coded since we don't know how to
		 * program them to anything but all 0's...
		 */
		if (so->frag_coord)
			fprintf(out, "; fragcoord: r0.x\n");
		if (so->frag_face)
			fprintf(out, "; fragface: hr0.x\n");
		break;
	default:
		/* TODO */
		break;
	}

	fprintf(out, "\n");
}

uint64_t
ir3_shader_outputs(const struct ir3_shader *so)
{
	return so->nir->info.outputs_written;
}

/* This has to reach into the fd_context a bit more than the rest of
 * ir3, but it needs to be aligned with the compiler, so both agree
 * on which const regs hold what.  And the logic is identical between
 * a3xx/a4xx, the only difference is small details in the actual
 * CP_LOAD_STATE packets (which is handled inside the generation
 * specific ctx->emit_const(_bo)() fxns)
 */

#include "freedreno_resource.h"

static inline bool
is_stateobj(struct fd_ringbuffer *ring)
{
	/* XXX this is an ugly way to differentiate.. */
	return !!(ring->flags & FD_RINGBUFFER_STREAMING);
}

static inline void
ring_wfi(struct fd_batch *batch, struct fd_ringbuffer *ring)
{
	/* when we emit const state via ring (IB2) we need a WFI, but when
	 * it is emit'd via stateobj, we don't
	 */
	if (is_stateobj(ring))
		return;

	fd_wfi(batch, ring);
}

static void
emit_user_consts(struct fd_context *ctx, const struct ir3_shader_variant *v,
		struct fd_ringbuffer *ring, struct fd_constbuf_stateobj *constbuf)
{
	const unsigned index = 0;     /* user consts are index 0 */

	if (constbuf->enabled_mask & (1 << index)) {
		struct pipe_constant_buffer *cb = &constbuf->cb[index];
		unsigned size = align(cb->buffer_size, 4) / 4; /* size in dwords */

		/* in particular, with binning shader we may end up with
		 * unused consts, ie. we could end up w/ constlen that is
		 * smaller than first_driver_param.  In that case truncate
		 * the user consts early to avoid HLSQ lockup caused by
		 * writing too many consts
		 */
		uint32_t max_const = MIN2(v->num_uniforms, v->constlen);

		// I expect that size should be a multiple of vec4's:
		assert(size == align(size, 4));

		/* and even if the start of the const buffer is before
		 * first_immediate, the end may not be:
		 */
		size = MIN2(size, 4 * max_const);

		if (size > 0) {
			ring_wfi(ctx->batch, ring);
			ctx->emit_const(ring, v->type, 0,
					cb->buffer_offset, size,
					cb->user_buffer, cb->buffer);
		}
	}
}

static void
emit_ubos(struct fd_context *ctx, const struct ir3_shader_variant *v,
		struct fd_ringbuffer *ring, struct fd_constbuf_stateobj *constbuf)
{
	uint32_t offset = v->constbase.ubo;
	if (v->constlen > offset) {
		uint32_t params = v->num_ubos;
		uint32_t offsets[params];
		struct pipe_resource *prscs[params];

		for (uint32_t i = 0; i < params; i++) {
			const uint32_t index = i + 1;   /* UBOs start at index 1 */
			struct pipe_constant_buffer *cb = &constbuf->cb[index];
			assert(!cb->user_buffer);

			if ((constbuf->enabled_mask & (1 << index)) && cb->buffer) {
				offsets[i] = cb->buffer_offset;
				prscs[i] = cb->buffer;
			} else {
				offsets[i] = 0;
				prscs[i] = NULL;
			}
		}

		ring_wfi(ctx->batch, ring);
		ctx->emit_const_bo(ring, v->type, false, offset * 4, params, prscs, offsets);
	}
}

static void
emit_ssbo_sizes(struct fd_context *ctx, const struct ir3_shader_variant *v,
		struct fd_ringbuffer *ring, struct fd_shaderbuf_stateobj *sb)
{
	uint32_t offset = v->constbase.ssbo_sizes;
	if (v->constlen > offset) {
		uint32_t sizes[align(v->const_layout.ssbo_size.count, 4)];
		unsigned mask = v->const_layout.ssbo_size.mask;

		while (mask) {
			unsigned index = u_bit_scan(&mask);
			unsigned off = v->const_layout.ssbo_size.off[index];
			sizes[off] = sb->sb[index].buffer_size;
		}

		ring_wfi(ctx->batch, ring);
		ctx->emit_const(ring, v->type, offset * 4,
			0, ARRAY_SIZE(sizes), sizes, NULL);
	}
}

static void
emit_image_dims(struct fd_context *ctx, const struct ir3_shader_variant *v,
		struct fd_ringbuffer *ring, struct fd_shaderimg_stateobj *si)
{
	uint32_t offset = v->constbase.image_dims;
	if (v->constlen > offset) {
		uint32_t dims[align(v->const_layout.image_dims.count, 4)];
		unsigned mask = v->const_layout.image_dims.mask;

		while (mask) {
			struct pipe_image_view *img;
			struct fd_resource *rsc;
			unsigned index = u_bit_scan(&mask);
			unsigned off = v->const_layout.image_dims.off[index];

			img = &si->si[index];
			rsc = fd_resource(img->resource);

			dims[off + 0] = util_format_get_blocksize(img->format);
			if (img->resource->target != PIPE_BUFFER) {
				unsigned lvl = img->u.tex.level;
				/* note for 2d/cube/etc images, even if re-interpreted
				 * as a different color format, the pixel size should
				 * be the same, so use original dimensions for y and z
				 * stride:
				 */
				dims[off + 1] = rsc->slices[lvl].pitch * rsc->cpp;
				/* see corresponding logic in fd_resource_offset(): */
				if (rsc->layer_first) {
					dims[off + 2] = rsc->layer_size;
				} else {
					dims[off + 2] = rsc->slices[lvl].size0;
				}
			} else {
				/* For buffer-backed images, the log2 of the format's
				 * bytes-per-pixel is placed on the 2nd slot. This is useful
				 * when emitting image_size instructions, for which we need
				 * to divide by bpp for image buffers. Since the bpp
				 * can only be power-of-two, the division is implemented
				 * as a SHR, and for that it is handy to have the log2 of
				 * bpp as a constant. (log2 = first-set-bit - 1)
				 */
				dims[off + 1] = ffs(dims[off + 0]) - 1;
			}
		}

		ring_wfi(ctx->batch, ring);
		ctx->emit_const(ring, v->type, offset * 4,
			0, ARRAY_SIZE(dims), dims, NULL);
	}
}

static void
emit_immediates(struct fd_context *ctx, const struct ir3_shader_variant *v,
		struct fd_ringbuffer *ring)
{
	int size = v->immediates_count;
	uint32_t base = v->constbase.immediate;

	/* truncate size to avoid writing constants that shader
	 * does not use:
	 */
	size = MIN2(size + base, v->constlen) - base;

	/* convert out of vec4: */
	base *= 4;
	size *= 4;

	if (size > 0) {
		ring_wfi(ctx->batch, ring);
		ctx->emit_const(ring, v->type, base,
			0, size, v->immediates[0].val, NULL);
	}
}

/* emit stream-out buffers: */
static void
emit_tfbos(struct fd_context *ctx, const struct ir3_shader_variant *v,
		struct fd_ringbuffer *ring)
{
	/* streamout addresses after driver-params: */
	uint32_t offset = v->constbase.tfbo;
	if (v->constlen > offset) {
		struct fd_streamout_stateobj *so = &ctx->streamout;
		struct pipe_stream_output_info *info = &v->shader->stream_output;
		uint32_t params = 4;
		uint32_t offsets[params];
		struct pipe_resource *prscs[params];

		for (uint32_t i = 0; i < params; i++) {
			struct pipe_stream_output_target *target = so->targets[i];

			if (target) {
				offsets[i] = (so->offsets[i] * info->stride[i] * 4) +
						target->buffer_offset;
				prscs[i] = target->buffer;
			} else {
				offsets[i] = 0;
				prscs[i] = NULL;
			}
		}

		ring_wfi(ctx->batch, ring);
		ctx->emit_const_bo(ring, v->type, true, offset * 4, params, prscs, offsets);
	}
}

static uint32_t
max_tf_vtx(struct fd_context *ctx, const struct ir3_shader_variant *v)
{
	struct fd_streamout_stateobj *so = &ctx->streamout;
	struct pipe_stream_output_info *info = &v->shader->stream_output;
	uint32_t maxvtxcnt = 0x7fffffff;

	if (ctx->screen->gpu_id >= 500)
		return 0;
	if (v->binning_pass)
		return 0;
	if (v->shader->stream_output.num_outputs == 0)
		return 0;
	if (so->num_targets == 0)
		return 0;

	/* offset to write to is:
	 *
	 *   total_vtxcnt = vtxcnt + offsets[i]
	 *   offset = total_vtxcnt * stride[i]
	 *
	 *   offset =   vtxcnt * stride[i]       ; calculated in shader
	 *            + offsets[i] * stride[i]   ; calculated at emit_tfbos()
	 *
	 * assuming for each vtx, each target buffer will have data written
	 * up to 'offset + stride[i]', that leaves maxvtxcnt as:
	 *
	 *   buffer_size = (maxvtxcnt * stride[i]) + stride[i]
	 *   maxvtxcnt   = (buffer_size - stride[i]) / stride[i]
	 *
	 * but shader is actually doing a less-than (rather than less-than-
	 * equal) check, so we can drop the -stride[i].
	 *
	 * TODO is assumption about `offset + stride[i]` legit?
	 */
	for (unsigned i = 0; i < so->num_targets; i++) {
		struct pipe_stream_output_target *target = so->targets[i];
		unsigned stride = info->stride[i] * 4;   /* convert dwords->bytes */
		if (target) {
			uint32_t max = target->buffer_size / stride;
			maxvtxcnt = MIN2(maxvtxcnt, max);
		}
	}

	return maxvtxcnt;
}

static void
emit_common_consts(const struct ir3_shader_variant *v, struct fd_ringbuffer *ring,
		struct fd_context *ctx, enum pipe_shader_type t)
{
	enum fd_dirty_shader_state dirty = ctx->dirty_shader[t];

	/* When we use CP_SET_DRAW_STATE objects to emit constant state,
	 * if we emit any of it we need to emit all.  This is because
	 * we are using the same state-group-id each time for uniform
	 * state, and if previous update is never evaluated (due to no
	 * visible primitives in the current tile) then the new stateobj
	 * completely replaces the old one.
	 *
	 * Possibly if we split up different parts of the const state to
	 * different state-objects we could avoid this.
	 */
	if (dirty && is_stateobj(ring))
		dirty = ~0;

	if (dirty & (FD_DIRTY_SHADER_PROG | FD_DIRTY_SHADER_CONST)) {
		struct fd_constbuf_stateobj *constbuf;
		bool shader_dirty;

		constbuf = &ctx->constbuf[t];
		shader_dirty = !!(dirty & FD_DIRTY_SHADER_PROG);

		emit_user_consts(ctx, v, ring, constbuf);
		emit_ubos(ctx, v, ring, constbuf);
		if (shader_dirty)
			emit_immediates(ctx, v, ring);
	}

	if (dirty & (FD_DIRTY_SHADER_PROG | FD_DIRTY_SHADER_SSBO)) {
		struct fd_shaderbuf_stateobj *sb = &ctx->shaderbuf[t];
		emit_ssbo_sizes(ctx, v, ring, sb);
	}

	if (dirty & (FD_DIRTY_SHADER_PROG | FD_DIRTY_SHADER_IMAGE)) {
		struct fd_shaderimg_stateobj *si = &ctx->shaderimg[t];
		emit_image_dims(ctx, v, ring, si);
	}
}

void
ir3_emit_vs_consts(const struct ir3_shader_variant *v, struct fd_ringbuffer *ring,
		struct fd_context *ctx, const struct pipe_draw_info *info)
{
	debug_assert(v->type == SHADER_VERTEX);

	emit_common_consts(v, ring, ctx, PIPE_SHADER_VERTEX);

	/* emit driver params every time: */
	/* TODO skip emit if shader doesn't use driver params to avoid WFI.. */
	if (info) {
		uint32_t offset = v->constbase.driver_param;
		if (v->constlen > offset) {
			uint32_t vertex_params[IR3_DP_VS_COUNT] = {
				[IR3_DP_VTXID_BASE] = info->index_size ?
						info->index_bias : info->start,
				[IR3_DP_VTXCNT_MAX] = max_tf_vtx(ctx, v),
			};
			/* if no user-clip-planes, we don't need to emit the
			 * entire thing:
			 */
			uint32_t vertex_params_size = 4;

			if (v->key.ucp_enables) {
				struct pipe_clip_state *ucp = &ctx->ucp;
				unsigned pos = IR3_DP_UCP0_X;
				for (unsigned i = 0; pos <= IR3_DP_UCP7_W; i++) {
					for (unsigned j = 0; j < 4; j++) {
						vertex_params[pos] = fui(ucp->ucp[i][j]);
						pos++;
					}
				}
				vertex_params_size = ARRAY_SIZE(vertex_params);
			}

			ring_wfi(ctx->batch, ring);

			bool needs_vtxid_base =
				ir3_find_sysval_regid(v, SYSTEM_VALUE_VERTEX_ID_ZERO_BASE) != regid(63, 0);

			/* for indirect draw, we need to copy VTXID_BASE from
			 * indirect-draw parameters buffer.. which is annoying
			 * and means we can't easily emit these consts in cmd
			 * stream so need to copy them to bo.
			 */
			if (info->indirect && needs_vtxid_base) {
				struct pipe_draw_indirect_info *indirect = info->indirect;
				struct pipe_resource *vertex_params_rsc =
					pipe_buffer_create(&ctx->screen->base,
						PIPE_BIND_CONSTANT_BUFFER, PIPE_USAGE_STREAM,
						vertex_params_size * 4);
				unsigned src_off = info->indirect->offset;;
				void *ptr;

				ptr = fd_bo_map(fd_resource(vertex_params_rsc)->bo);
				memcpy(ptr, vertex_params, vertex_params_size * 4);

				if (info->index_size) {
					/* indexed draw, index_bias is 4th field: */
					src_off += 3 * 4;
				} else {
					/* non-indexed draw, start is 3rd field: */
					src_off += 2 * 4;
				}

				/* copy index_bias or start from draw params: */
				ctx->mem_to_mem(ring, vertex_params_rsc, 0,
						indirect->buffer, src_off, 1);

				ctx->emit_const(ring, SHADER_VERTEX, offset * 4, 0,
						vertex_params_size, NULL, vertex_params_rsc);

				pipe_resource_reference(&vertex_params_rsc, NULL);
			} else {
				ctx->emit_const(ring, SHADER_VERTEX, offset * 4, 0,
						vertex_params_size, vertex_params, NULL);
			}

			/* if needed, emit stream-out buffer addresses: */
			if (vertex_params[IR3_DP_VTXCNT_MAX] > 0) {
				emit_tfbos(ctx, v, ring);
			}
		}
	}
}

void
ir3_emit_fs_consts(const struct ir3_shader_variant *v, struct fd_ringbuffer *ring,
		struct fd_context *ctx)
{
	debug_assert(v->type == SHADER_FRAGMENT);

	emit_common_consts(v, ring, ctx, PIPE_SHADER_FRAGMENT);
}

/* emit compute-shader consts: */
void
ir3_emit_cs_consts(const struct ir3_shader_variant *v, struct fd_ringbuffer *ring,
		struct fd_context *ctx, const struct pipe_grid_info *info)
{
	debug_assert(v->type == SHADER_COMPUTE);

	emit_common_consts(v, ring, ctx, PIPE_SHADER_COMPUTE);

	/* emit compute-shader driver-params: */
	uint32_t offset = v->constbase.driver_param;
	if (v->constlen > offset) {
		ring_wfi(ctx->batch, ring);

		if (info->indirect) {
			struct pipe_resource *indirect = NULL;
			unsigned indirect_offset;

			/* This is a bit awkward, but CP_LOAD_STATE.EXT_SRC_ADDR needs
			 * to be aligned more strongly than 4 bytes.  So in this case
			 * we need a temporary buffer to copy NumWorkGroups.xyz to.
			 *
			 * TODO if previous compute job is writing to info->indirect,
			 * we might need a WFI.. but since we currently flush for each
			 * compute job, we are probably ok for now.
			 */
			if (info->indirect_offset & 0xf) {
				indirect = pipe_buffer_create(&ctx->screen->base,
					PIPE_BIND_COMMAND_ARGS_BUFFER, PIPE_USAGE_STREAM,
					0x1000);
				indirect_offset = 0;

				ctx->mem_to_mem(ring, indirect, 0, info->indirect,
						info->indirect_offset, 3);
			} else {
				pipe_resource_reference(&indirect, info->indirect);
				indirect_offset = info->indirect_offset;
			}

			ctx->emit_const(ring, SHADER_COMPUTE, offset * 4,
					indirect_offset, 4, NULL, indirect);

			pipe_resource_reference(&indirect, NULL);
		} else {
			uint32_t compute_params[IR3_DP_CS_COUNT] = {
				[IR3_DP_NUM_WORK_GROUPS_X] = info->grid[0],
				[IR3_DP_NUM_WORK_GROUPS_Y] = info->grid[1],
				[IR3_DP_NUM_WORK_GROUPS_Z] = info->grid[2],
				[IR3_DP_LOCAL_GROUP_SIZE_X] = info->block[0],
				[IR3_DP_LOCAL_GROUP_SIZE_Y] = info->block[1],
				[IR3_DP_LOCAL_GROUP_SIZE_Z] = info->block[2],
			};

			ctx->emit_const(ring, SHADER_COMPUTE, offset * 4, 0,
					ARRAY_SIZE(compute_params), compute_params, NULL);
		}
	}
}
