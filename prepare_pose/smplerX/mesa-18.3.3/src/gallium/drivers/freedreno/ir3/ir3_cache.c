/* -*- mode: C; c-file-style: "k&r"; tab-width 4; indent-tabs-mode: t; -*- */

/*
 * Copyright (C) 2015 Rob Clark <robclark@freedesktop.org>
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

#include "util/ralloc.h"
#include "util/hash_table.h"

#include "ir3_cache.h"
#include "ir3_shader.h"


static uint32_t
key_hash(const void *_key)
{
	const struct ir3_cache_key *key = _key;
	uint32_t hash = _mesa_fnv32_1a_offset_bias;
	hash = _mesa_fnv32_1a_accumulate_block(hash, key, sizeof(*key));
	return hash;
}

static bool
key_equals(const void *_a, const void *_b)
{
	const struct ir3_cache_key *a = _a;
	const struct ir3_cache_key *b = _b;
	// TODO we could optimize the key shader-variant key comparison by not
	// ignoring has_per_samp.. not really sure if that helps..
	return memcmp(a, b, sizeof(struct ir3_cache_key)) == 0;
}

struct ir3_cache {
	/* cache mapping gallium/etc shader state-objs + shader-key to backend
	 * specific state-object
	 */
	struct hash_table *ht;

	const struct ir3_cache_funcs *funcs;
	void *data;
};

struct ir3_cache * ir3_cache_create(const struct ir3_cache_funcs *funcs, void *data)
{
	struct ir3_cache *cache = rzalloc(NULL, struct ir3_cache);

	cache->ht = _mesa_hash_table_create(cache, key_hash, key_equals);
	cache->funcs = funcs;
	cache->data = data;

	return cache;
}

void ir3_cache_destroy(struct ir3_cache *cache)
{
	/* _mesa_hash_table_destroy is so *almost* useful.. */
	hash_table_foreach(cache->ht, entry) {
		cache->funcs->destroy_state(cache->data, entry->data);
	}

	ralloc_free(cache);
}

struct ir3_program_state *
ir3_cache_lookup(struct ir3_cache *cache, const struct ir3_cache_key *key,
		struct pipe_debug_callback *debug)
{
	uint32_t hash = key_hash(key);
	struct hash_entry *entry =
		_mesa_hash_table_search_pre_hashed(cache->ht, hash, key);

	if (entry) {
		return entry->data;
	}

	struct ir3_shader_variant *bs = ir3_shader_variant(key->vs, key->key, true, debug);
	struct ir3_shader_variant *vs = ir3_shader_variant(key->vs, key->key, false, debug);
	struct ir3_shader_variant *fs = ir3_shader_variant(key->fs, key->key, false, debug);

	struct ir3_program_state *state =
		cache->funcs->create_state(cache->data, bs, vs, fs, &key->key);
	state->key = *key;

	/* NOTE: uses copy of key in state obj, because pointer passed by caller
	 * is probably on the stack
	 */
	_mesa_hash_table_insert_pre_hashed(cache->ht, hash, &state->key, state);

	return state;
}

/* call when an API level state object is destroyed, to invalidate
 * cache entries which reference that state object.
 */
void ir3_cache_invalidate(struct ir3_cache *cache, void *stobj)
{
	hash_table_foreach(cache->ht, entry) {
		const struct ir3_cache_key *key = entry->key;
		if ((key->fs == stobj) || (key->vs == stobj)) {
			cache->funcs->destroy_state(cache->data, entry->data);
			_mesa_hash_table_remove(cache->ht, entry);
			return;
		}
	}
}
