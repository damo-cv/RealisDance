/*
 * Copyright (C) 2012 Rob Clark <robclark@freedesktop.org>
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

#ifndef FREEDRENO_UTIL_H_
#define FREEDRENO_UTIL_H_

#include "drm/freedreno_drmif.h"
#include "drm/freedreno_ringbuffer.h"

#include "pipe/p_format.h"
#include "pipe/p_state.h"
#include "util/u_debug.h"
#include "util/u_math.h"
#include "util/u_half.h"
#include "util/u_dynarray.h"
#include "util/u_pack_color.h"

#include "disasm.h"
#include "adreno_common.xml.h"
#include "adreno_pm4.xml.h"

enum adreno_rb_depth_format fd_pipe2depth(enum pipe_format format);
enum pc_di_index_size fd_pipe2index(enum pipe_format format);
enum pipe_format fd_gmem_restore_format(enum pipe_format format);
enum adreno_rb_blend_factor fd_blend_factor(unsigned factor);
enum adreno_pa_su_sc_draw fd_polygon_mode(unsigned mode);
enum adreno_stencil_op fd_stencil_op(unsigned op);

#define A3XX_MAX_MIP_LEVELS 14
/* TBD if it is same on a2xx, but for now: */
#define MAX_MIP_LEVELS A3XX_MAX_MIP_LEVELS

#define A2XX_MAX_RENDER_TARGETS 1
#define A3XX_MAX_RENDER_TARGETS 4
#define A4XX_MAX_RENDER_TARGETS 8
#define A5XX_MAX_RENDER_TARGETS 8
#define A6XX_MAX_RENDER_TARGETS 8

#define MAX_RENDER_TARGETS A6XX_MAX_RENDER_TARGETS

#define FD_DBG_MSGS     0x0001
#define FD_DBG_DISASM   0x0002
#define FD_DBG_DCLEAR   0x0004
#define FD_DBG_DDRAW    0x0008
#define FD_DBG_NOSCIS   0x0010
#define FD_DBG_DIRECT   0x0020
#define FD_DBG_NOBYPASS 0x0040
#define FD_DBG_FRAGHALF 0x0080
#define FD_DBG_NOBIN    0x0100
#define FD_DBG_OPTMSGS  0x0200
#define FD_DBG_GLSL120  0x0400
#define FD_DBG_SHADERDB 0x0800
#define FD_DBG_FLUSH    0x1000
#define FD_DBG_DEQP     0x2000
#define FD_DBG_INORDER  0x4000
#define FD_DBG_BSTAT    0x8000
#define FD_DBG_NOGROW  0x10000
#define FD_DBG_LRZ     0x20000
#define FD_DBG_NOINDR  0x40000
#define FD_DBG_NOBLIT  0x80000
#define FD_DBG_HIPRIO 0x100000
#define FD_DBG_TTILE  0x200000
#define FD_DBG_PERFC  0x400000
#define FD_DBG_SOFTPIN 0x800000

extern int fd_mesa_debug;
extern bool fd_binning_enabled;

#define DBG(fmt, ...) \
		do { if (fd_mesa_debug & FD_DBG_MSGS) \
			debug_printf("%s:%d: "fmt "\n", \
				__FUNCTION__, __LINE__, ##__VA_ARGS__); } while (0)

/* for conditionally setting boolean flag(s): */
#define COND(bool, val) ((bool) ? (val) : 0)

#define CP_REG(reg) ((0x4 << 16) | ((unsigned int)((reg) - (0x2000))))

static inline uint32_t DRAW(enum pc_di_primtype prim_type,
		enum pc_di_src_sel source_select, enum pc_di_index_size index_size,
		enum pc_di_vis_cull_mode vis_cull_mode,
		uint8_t instances)
{
	return (prim_type         << 0) |
			(source_select     << 6) |
			((index_size & 1)  << 11) |
			((index_size >> 1) << 13) |
			(vis_cull_mode     << 9) |
			(1                 << 14) |
			(instances         << 24);
}

static inline uint32_t DRAW_A20X(enum pc_di_primtype prim_type,
		enum pc_di_src_sel source_select, enum pc_di_index_size index_size,
		enum pc_di_vis_cull_mode vis_cull_mode,
		uint16_t count)
{
	return (prim_type         << 0) |
			(source_select     << 6) |
			((index_size & 1)  << 11) |
			((index_size >> 1) << 13) |
			(vis_cull_mode     << 9) |
			(count         << 16);
}

/* for tracking cmdstream positions that need to be patched: */
struct fd_cs_patch {
	uint32_t *cs;
	uint32_t val;
};
#define fd_patch_num_elements(buf) ((buf)->size / sizeof(struct fd_cs_patch))
#define fd_patch_element(buf, i)   util_dynarray_element(buf, struct fd_cs_patch, i)

static inline enum pipe_format
pipe_surface_format(struct pipe_surface *psurf)
{
	if (!psurf)
		return PIPE_FORMAT_NONE;
	return psurf->format;
}

static inline bool
fd_surface_half_precision(const struct pipe_surface *psurf)
{
	enum pipe_format format;

	if (!psurf)
		return true;

	format = psurf->format;

	/* colors are provided in consts, which go through cov.f32f16, which will
	 * break these values
	 */
	if (util_format_is_pure_integer(format))
		return false;

	/* avoid losing precision on 32-bit float formats */
	if (util_format_is_float(format) &&
		util_format_get_component_bits(format, UTIL_FORMAT_COLORSPACE_RGB, 0) == 32)
		return false;

	return true;
}

static inline unsigned
fd_sampler_first_level(const struct pipe_sampler_view *view)
{
	if (view->target == PIPE_BUFFER)
		return 0;
	return view->u.tex.first_level;
}

static inline unsigned
fd_sampler_last_level(const struct pipe_sampler_view *view)
{
	if (view->target == PIPE_BUFFER)
		return 0;
	return view->u.tex.last_level;
}

static inline bool
fd_half_precision(struct pipe_framebuffer_state *pfb)
{
	unsigned i;

	for (i = 0; i < pfb->nr_cbufs; i++)
		if (!fd_surface_half_precision(pfb->cbufs[i]))
			return false;

	return true;
}

#define LOG_DWORDS 0

static inline void emit_marker(struct fd_ringbuffer *ring, int scratch_idx);

static inline void
OUT_RING(struct fd_ringbuffer *ring, uint32_t data)
{
	if (LOG_DWORDS) {
		DBG("ring[%p]: OUT_RING   %04x:  %08x", ring,
				(uint32_t)(ring->cur - ring->start), data);
	}
	fd_ringbuffer_emit(ring, data);
}

/* like OUT_RING() but appends a cmdstream patch point to 'buf' */
static inline void
OUT_RINGP(struct fd_ringbuffer *ring, uint32_t data,
		struct util_dynarray *buf)
{
	if (LOG_DWORDS) {
		DBG("ring[%p]: OUT_RINGP  %04x:  %08x", ring,
				(uint32_t)(ring->cur - ring->start), data);
	}
	util_dynarray_append(buf, struct fd_cs_patch, ((struct fd_cs_patch){
		.cs  = ring->cur++,
		.val = data,
	}));
}

/*
 * NOTE: OUT_RELOC*() is 2 dwords (64b) on a5xx+
 */

static inline void
OUT_RELOC(struct fd_ringbuffer *ring, struct fd_bo *bo,
		uint32_t offset, uint64_t or, int32_t shift)
{
	if (LOG_DWORDS) {
		DBG("ring[%p]: OUT_RELOC   %04x:  %p+%u << %d", ring,
				(uint32_t)(ring->cur - ring->start), bo, offset, shift);
	}
	debug_assert(offset < fd_bo_size(bo));
	fd_ringbuffer_reloc(ring, &(struct fd_reloc){
		.bo = bo,
		.flags = FD_RELOC_READ,
		.offset = offset,
		.or = or,
		.shift = shift,
		.orhi = or >> 32,
	});
}

static inline void
OUT_RELOCW(struct fd_ringbuffer *ring, struct fd_bo *bo,
		uint32_t offset, uint64_t or, int32_t shift)
{
	if (LOG_DWORDS) {
		DBG("ring[%p]: OUT_RELOCW  %04x:  %p+%u << %d", ring,
				(uint32_t)(ring->cur - ring->start), bo, offset, shift);
	}
	debug_assert(offset < fd_bo_size(bo));
	fd_ringbuffer_reloc(ring, &(struct fd_reloc){
		.bo = bo,
		.flags = FD_RELOC_READ | FD_RELOC_WRITE,
		.offset = offset,
		.or = or,
		.shift = shift,
		.orhi = or >> 32,
	});
}

static inline void
OUT_RB(struct fd_ringbuffer *ring, struct fd_ringbuffer *target)
{
	fd_ringbuffer_emit_reloc_ring_full(ring, target, 0);
}

static inline void BEGIN_RING(struct fd_ringbuffer *ring, uint32_t ndwords)
{
	if (ring->cur + ndwords > ring->end)
		fd_ringbuffer_grow(ring, ndwords);
}

static inline void
OUT_PKT0(struct fd_ringbuffer *ring, uint16_t regindx, uint16_t cnt)
{
	BEGIN_RING(ring, cnt+1);
	OUT_RING(ring, CP_TYPE0_PKT | ((cnt-1) << 16) | (regindx & 0x7FFF));
}

static inline void
OUT_PKT2(struct fd_ringbuffer *ring)
{
	BEGIN_RING(ring, 1);
	OUT_RING(ring, CP_TYPE2_PKT);
}

static inline void
OUT_PKT3(struct fd_ringbuffer *ring, uint8_t opcode, uint16_t cnt)
{
	BEGIN_RING(ring, cnt+1);
	OUT_RING(ring, CP_TYPE3_PKT | ((cnt-1) << 16) | ((opcode & 0xFF) << 8));
}

/*
 * Starting with a5xx, pkt4/pkt7 are used instead of pkt0/pkt3
 */

static inline unsigned
_odd_parity_bit(unsigned val)
{
	/* See: http://graphics.stanford.edu/~seander/bithacks.html#ParityParallel
	 * note that we want odd parity so 0x6996 is inverted.
	 */
	val ^= val >> 16;
	val ^= val >> 8;
	val ^= val >> 4;
	val &= 0xf;
	return (~0x6996 >> val) & 1;
}

static inline void
OUT_PKT4(struct fd_ringbuffer *ring, uint16_t regindx, uint16_t cnt)
{
	BEGIN_RING(ring, cnt+1);
	OUT_RING(ring, CP_TYPE4_PKT | cnt |
			(_odd_parity_bit(cnt) << 7) |
			((regindx & 0x3ffff) << 8) |
			((_odd_parity_bit(regindx) << 27)));
}

static inline void
OUT_PKT7(struct fd_ringbuffer *ring, uint8_t opcode, uint16_t cnt)
{
	BEGIN_RING(ring, cnt+1);
	OUT_RING(ring, CP_TYPE7_PKT | cnt |
			(_odd_parity_bit(cnt) << 15) |
			((opcode & 0x7f) << 16) |
			((_odd_parity_bit(opcode) << 23)));
}

static inline void
OUT_WFI(struct fd_ringbuffer *ring)
{
	OUT_PKT3(ring, CP_WAIT_FOR_IDLE, 1);
	OUT_RING(ring, 0x00000000);
}

static inline void
OUT_WFI5(struct fd_ringbuffer *ring)
{
	OUT_PKT7(ring, CP_WAIT_FOR_IDLE, 0);
}

static inline void
__OUT_IB(struct fd_ringbuffer *ring, bool prefetch, struct fd_ringbuffer *target)
{
	if (target->cur == target->start)
		return;

	unsigned count = fd_ringbuffer_cmd_count(target);

	/* for debug after a lock up, write a unique counter value
	 * to scratch6 for each IB, to make it easier to match up
	 * register dumps to cmdstream.  The combination of IB and
	 * DRAW (scratch7) is enough to "triangulate" the particular
	 * draw that caused lockup.
	 */
	emit_marker(ring, 6);

	for (unsigned i = 0; i < count; i++) {
		uint32_t dwords;
		OUT_PKT3(ring, prefetch ? CP_INDIRECT_BUFFER_PFE : CP_INDIRECT_BUFFER_PFD, 2);
		dwords = fd_ringbuffer_emit_reloc_ring_full(ring, target, i) / 4;
		assert(dwords > 0);
		OUT_RING(ring, dwords);
		OUT_PKT2(ring);
	}

	emit_marker(ring, 6);
}

static inline void
__OUT_IB5(struct fd_ringbuffer *ring, struct fd_ringbuffer *target)
{
	if (target->cur == target->start)
		return;

	unsigned count = fd_ringbuffer_cmd_count(target);

	for (unsigned i = 0; i < count; i++) {
		uint32_t dwords;
		OUT_PKT7(ring, CP_INDIRECT_BUFFER, 3);
		dwords = fd_ringbuffer_emit_reloc_ring_full(ring, target, i) / 4;
		assert(dwords > 0);
		OUT_RING(ring, dwords);
	}
}

/* CP_SCRATCH_REG4 is used to hold base address for query results: */
// XXX annoyingly scratch regs move on a5xx.. and additionally different
// packet types.. so freedreno_query_hw is going to need a bit of
// rework..
#define HW_QUERY_BASE_REG REG_AXXX_CP_SCRATCH_REG4

static inline void
emit_marker(struct fd_ringbuffer *ring, int scratch_idx)
{
	extern unsigned marker_cnt;
	unsigned reg = REG_AXXX_CP_SCRATCH_REG0 + scratch_idx;
	assert(reg != HW_QUERY_BASE_REG);
	if (reg == HW_QUERY_BASE_REG)
		return;
	OUT_PKT0(ring, reg, 1);
	OUT_RING(ring, ++marker_cnt);
}

/* helper to get numeric value from environment variable..  mostly
 * just leaving this here because it is helpful to brute-force figure
 * out unknown formats, etc, which blob driver does not support:
 */
static inline uint32_t env2u(const char *envvar)
{
	char *str = getenv(envvar);
	if (str)
		return strtoul(str, NULL, 0);
	return 0;
}

static inline uint32_t
pack_rgba(enum pipe_format format, const float *rgba)
{
	union util_color uc;
	util_pack_color(rgba, format, &uc);
	return uc.ui[0];
}

/*
 * swap - swap value of @a and @b
 */
#define swap(a, b) \
	do { __typeof(a) __tmp = (a); (a) = (b); (b) = __tmp; } while (0)

#define foreach_bit(b, mask) \
	for (uint32_t _m = (mask); _m && ({(b) = u_bit_scan(&_m); 1;});)


#define BIT(bit) (1u << bit)

/*
 * a3xx+ helpers:
 */

static inline enum a3xx_msaa_samples
fd_msaa_samples(unsigned samples)
{
	switch (samples) {
	default:
		debug_assert(0);
	case 1: return MSAA_ONE;
	case 2: return MSAA_TWO;
	case 4: return MSAA_FOUR;
	}
}

/*
 * a4xx+ helpers:
 */

static inline enum a4xx_state_block
fd4_stage2shadersb(enum shader_t type)
{
	switch (type) {
	case SHADER_VERTEX:
		return SB4_VS_SHADER;
	case SHADER_FRAGMENT:
		return SB4_FS_SHADER;
	case SHADER_COMPUTE:
		return SB4_CS_SHADER;
	default:
		unreachable("bad shader type");
		return ~0;
	}
}

#endif /* FREEDRENO_UTIL_H_ */
