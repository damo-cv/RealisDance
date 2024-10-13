/******************************************************************************
 * Copyright (C) 2015-2018 Intel Corporation.   All Rights Reserved.
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
 *
 * @file gen_knobs.h
 *
 * @brief Dynamic Knobs for Core.
 *
 * ======================= AUTO GENERATED: DO NOT EDIT !!! ====================
 *
 * Generation Command Line:
 *  ./rasterizer/codegen/gen_knobs.py
 *    --output
 *    rasterizer/codegen/gen_knobs.h
 *    --gen_h
 *
 ******************************************************************************/
// clang-format off

#pragma once
#include <string>

struct KnobBase
{
private:
    // Update the input string.
    static void autoExpandEnvironmentVariables(std::string& text);

protected:
    // Leave input alone and return new string.
    static std::string expandEnvironmentVariables(std::string const& input)
    {
        std::string text = input;
        autoExpandEnvironmentVariables(text);
        return text;
    }

    template <typename T>
    static T expandEnvironmentVariables(T const& input)
    {
        return input;
    }
};

template <typename T>
struct Knob : KnobBase
{
public:
    const T& Value() const { return m_Value; }
    const T& Value(T const& newValue)
    {
        m_Value = expandEnvironmentVariables(newValue);
        return Value();
    }

private:
    T m_Value;
};

#define DEFINE_KNOB(_name, _type, _default)                     \
    struct Knob_##_name : Knob<_type>                           \
    {                                                           \
        static const char* Name() { return "KNOB_" #_name; }    \
        static _type DefaultValue() { return (_default); }      \
    } _name;

#define GET_KNOB(_name)             g_GlobalKnobs._name.Value()
#define SET_KNOB(_name, _newValue)  g_GlobalKnobs._name.Value(_newValue)

struct GlobalKnobs
{
    //-----------------------------------------------------------
    // KNOB_ENABLE_ASSERT_DIALOGS
    //
    // Use dialogs when asserts fire.
    // Asserts are only enabled in debug builds
    //
    DEFINE_KNOB(ENABLE_ASSERT_DIALOGS, bool, true);

    //-----------------------------------------------------------
    // KNOB_SINGLE_THREADED
    //
    // If enabled will perform all rendering on the API thread.
    // This is useful mainly for debugging purposes.
    //
    DEFINE_KNOB(SINGLE_THREADED, bool, false);

    //-----------------------------------------------------------
    // KNOB_DUMP_SHADER_IR
    //
    // Dumps shader LLVM IR at various stages of jit compilation.
    //
    DEFINE_KNOB(DUMP_SHADER_IR, bool, false);

    //-----------------------------------------------------------
    // KNOB_USE_GENERIC_STORETILE
    //
    // Always use generic function for performing StoreTile.
    // Will be slightly slower than using optimized (jitted) path
    //
    DEFINE_KNOB(USE_GENERIC_STORETILE, bool, false);

    //-----------------------------------------------------------
    // KNOB_FAST_CLEAR
    //
    // Replace 3D primitive execute with a SWRClearRT operation and
    // defer clear execution to first backend op on hottile, or hottile store
    //
    DEFINE_KNOB(FAST_CLEAR, bool, true);

    //-----------------------------------------------------------
    // KNOB_MAX_NUMA_NODES
    //
    // Maximum # of NUMA-nodes per system used for worker threads
    //   0 == ALL NUMA-nodes in the system
    //   N == Use at most N NUMA-nodes for rendering
    //
    DEFINE_KNOB(MAX_NUMA_NODES, uint32_t, 0);

    //-----------------------------------------------------------
    // KNOB_MAX_CORES_PER_NUMA_NODE
    //
    // Maximum # of cores per NUMA-node used for worker threads.
    //   0 == ALL non-API thread cores per NUMA-node
    //   N == Use at most N cores per NUMA-node
    //
    DEFINE_KNOB(MAX_CORES_PER_NUMA_NODE, uint32_t, 0);

    //-----------------------------------------------------------
    // KNOB_MAX_THREADS_PER_CORE
    //
    // Maximum # of (hyper)threads per physical core used for worker threads.
    //   0 == ALL hyper-threads per core
    //   N == Use at most N hyper-threads per physical core
    //
    DEFINE_KNOB(MAX_THREADS_PER_CORE, uint32_t, 1);

    //-----------------------------------------------------------
    // KNOB_MAX_WORKER_THREADS
    //
    // Maximum worker threads to spawn.
    // 
    // IMPORTANT: If this is non-zero, no worker threads will be bound to
    // specific HW threads.  They will all be "floating" SW threads.
    // In this case, the above 3 KNOBS will be ignored.
    //
    DEFINE_KNOB(MAX_WORKER_THREADS, uint32_t, 0);

    //-----------------------------------------------------------
    // KNOB_BASE_NUMA_NODE
    //
    // Starting NUMA node index to use when allocating compute resources.
    // Setting this to a non-zero value will reduce the maximum # of NUMA nodes used.
    //
    DEFINE_KNOB(BASE_NUMA_NODE, uint32_t, 0);

    //-----------------------------------------------------------
    // KNOB_BASE_CORE
    //
    // Starting core index to use when allocating compute resources.
    // Setting this to a non-zero value will reduce the maximum # of cores used.
    //
    DEFINE_KNOB(BASE_CORE, uint32_t, 0);

    //-----------------------------------------------------------
    // KNOB_BASE_THREAD
    //
    // Starting thread index to use when allocating compute resources.
    // Setting this to a non-zero value will reduce the maximum # of threads used.
    //
    DEFINE_KNOB(BASE_THREAD, uint32_t, 0);

    //-----------------------------------------------------------
    // KNOB_BUCKETS_START_FRAME
    //
    // Frame from when to start saving buckets data.
    // 
    // NOTE: KNOB_ENABLE_RDTSC must be enabled in core/knobs.h
    // for this to have an effect.
    //
    DEFINE_KNOB(BUCKETS_START_FRAME, uint32_t, 1200);

    //-----------------------------------------------------------
    // KNOB_BUCKETS_END_FRAME
    //
    // Frame at which to stop saving buckets data.
    // 
    // NOTE: KNOB_ENABLE_RDTSC must be enabled in core/knobs.h
    // for this to have an effect.
    //
    DEFINE_KNOB(BUCKETS_END_FRAME, uint32_t, 1400);

    //-----------------------------------------------------------
    // KNOB_WORKER_SPIN_LOOP_COUNT
    //
    // Number of spin-loop iterations worker threads will perform
    // before going to sleep when waiting for work
    //
    DEFINE_KNOB(WORKER_SPIN_LOOP_COUNT, uint32_t, 5000);

    //-----------------------------------------------------------
    // KNOB_MAX_DRAWS_IN_FLIGHT
    //
    // Maximum number of draws outstanding before API thread blocks.
    // This value MUST be evenly divisible into 2^32
    //
    DEFINE_KNOB(MAX_DRAWS_IN_FLIGHT, uint32_t, 256);

    //-----------------------------------------------------------
    // KNOB_MAX_PRIMS_PER_DRAW
    //
    // Maximum primitives in a single Draw().
    // Larger primitives are split into smaller Draw calls.
    // Should be a multiple of (3 * vectorWidth).
    //
    DEFINE_KNOB(MAX_PRIMS_PER_DRAW, uint32_t, 49152);

    //-----------------------------------------------------------
    // KNOB_MAX_TESS_PRIMS_PER_DRAW
    //
    // Maximum primitives in a single Draw() with tessellation enabled.
    // Larger primitives are split into smaller Draw calls.
    // Should be a multiple of (vectorWidth).
    //
    DEFINE_KNOB(MAX_TESS_PRIMS_PER_DRAW, uint32_t, 16);

    //-----------------------------------------------------------
    // KNOB_DEBUG_OUTPUT_DIR
    //
    // Output directory for debug data.
    //
    DEFINE_KNOB(DEBUG_OUTPUT_DIR, std::string, "/tmp/Rast/DebugOutput");

    //-----------------------------------------------------------
    // KNOB_JIT_ENABLE_CACHE
    //
    // Enables caching of compiled shaders
    //
    DEFINE_KNOB(JIT_ENABLE_CACHE, bool, false);

    //-----------------------------------------------------------
    // KNOB_JIT_OPTIMIZATION_LEVEL
    //
    // JIT compile optimization level:
    //
    //     Automatic    = -0x0000001
    //     Debug        = 0x00000000
    //     Less         = 0x00000001
    //     Optimize     = 0x00000002
    //     Aggressive   = 0x00000003
    //
    DEFINE_KNOB(JIT_OPTIMIZATION_LEVEL, int, -1);

    //-----------------------------------------------------------
    // KNOB_JIT_CACHE_DIR
    //
    // Cache directory for compiled shaders.
    //
    DEFINE_KNOB(JIT_CACHE_DIR, std::string, "${HOME}/.swr/jitcache");

    //-----------------------------------------------------------
    // KNOB_TOSS_DRAW
    //
    // Disable per-draw/dispatch execution
    //
    DEFINE_KNOB(TOSS_DRAW, bool, false);

    //-----------------------------------------------------------
    // KNOB_TOSS_QUEUE_FE
    //
    // Stop per-draw execution at worker FE
    // 
    // NOTE: Requires KNOB_ENABLE_TOSS_POINTS to be enabled in core/knobs.h
    //
    DEFINE_KNOB(TOSS_QUEUE_FE, bool, false);

    //-----------------------------------------------------------
    // KNOB_TOSS_FETCH
    //
    // Stop per-draw execution at vertex fetch
    // 
    // NOTE: Requires KNOB_ENABLE_TOSS_POINTS to be enabled in core/knobs.h
    //
    DEFINE_KNOB(TOSS_FETCH, bool, false);

    //-----------------------------------------------------------
    // KNOB_TOSS_IA
    //
    // Stop per-draw execution at input assembler
    // 
    // NOTE: Requires KNOB_ENABLE_TOSS_POINTS to be enabled in core/knobs.h
    //
    DEFINE_KNOB(TOSS_IA, bool, false);

    //-----------------------------------------------------------
    // KNOB_TOSS_VS
    //
    // Stop per-draw execution at vertex shader
    // 
    // NOTE: Requires KNOB_ENABLE_TOSS_POINTS to be enabled in core/knobs.h
    //
    DEFINE_KNOB(TOSS_VS, bool, false);

    //-----------------------------------------------------------
    // KNOB_TOSS_SETUP_TRIS
    //
    // Stop per-draw execution at primitive setup
    // 
    // NOTE: Requires KNOB_ENABLE_TOSS_POINTS to be enabled in core/knobs.h
    //
    DEFINE_KNOB(TOSS_SETUP_TRIS, bool, false);

    //-----------------------------------------------------------
    // KNOB_TOSS_BIN_TRIS
    //
    // Stop per-draw execution at primitive binning
    // 
    // NOTE: Requires KNOB_ENABLE_TOSS_POINTS to be enabled in core/knobs.h
    //
    DEFINE_KNOB(TOSS_BIN_TRIS, bool, false);

    //-----------------------------------------------------------
    // KNOB_TOSS_RS
    //
    // Stop per-draw execution at rasterizer
    // 
    // NOTE: Requires KNOB_ENABLE_TOSS_POINTS to be enabled in core/knobs.h
    //
    DEFINE_KNOB(TOSS_RS, bool, false);

    //-----------------------------------------------------------
    // KNOB_DISABLE_SPLIT_DRAW
    //
    // Don't split large draws into smaller draws.,
    // MAX_PRIMS_PER_DRAW and MAX_TESS_PRIMS_PER_DRAW can be used to control split size.
    // 
    // Useful to disable split draws for gathering archrast stats.
    //
    DEFINE_KNOB(DISABLE_SPLIT_DRAW, bool, false);


    std::string ToString(const char* optPerLinePrefix="");
    GlobalKnobs();
};
extern GlobalKnobs g_GlobalKnobs;

#undef DEFINE_KNOB

#define KNOB_ENABLE_ASSERT_DIALOGS       GET_KNOB(ENABLE_ASSERT_DIALOGS)
#define KNOB_SINGLE_THREADED             GET_KNOB(SINGLE_THREADED)
#define KNOB_DUMP_SHADER_IR              GET_KNOB(DUMP_SHADER_IR)
#define KNOB_USE_GENERIC_STORETILE       GET_KNOB(USE_GENERIC_STORETILE)
#define KNOB_FAST_CLEAR                  GET_KNOB(FAST_CLEAR)
#define KNOB_MAX_NUMA_NODES              GET_KNOB(MAX_NUMA_NODES)
#define KNOB_MAX_CORES_PER_NUMA_NODE     GET_KNOB(MAX_CORES_PER_NUMA_NODE)
#define KNOB_MAX_THREADS_PER_CORE        GET_KNOB(MAX_THREADS_PER_CORE)
#define KNOB_MAX_WORKER_THREADS          GET_KNOB(MAX_WORKER_THREADS)
#define KNOB_BASE_NUMA_NODE              GET_KNOB(BASE_NUMA_NODE)
#define KNOB_BASE_CORE                   GET_KNOB(BASE_CORE)
#define KNOB_BASE_THREAD                 GET_KNOB(BASE_THREAD)
#define KNOB_BUCKETS_START_FRAME         GET_KNOB(BUCKETS_START_FRAME)
#define KNOB_BUCKETS_END_FRAME           GET_KNOB(BUCKETS_END_FRAME)
#define KNOB_WORKER_SPIN_LOOP_COUNT      GET_KNOB(WORKER_SPIN_LOOP_COUNT)
#define KNOB_MAX_DRAWS_IN_FLIGHT         GET_KNOB(MAX_DRAWS_IN_FLIGHT)
#define KNOB_MAX_PRIMS_PER_DRAW          GET_KNOB(MAX_PRIMS_PER_DRAW)
#define KNOB_MAX_TESS_PRIMS_PER_DRAW     GET_KNOB(MAX_TESS_PRIMS_PER_DRAW)
#define KNOB_DEBUG_OUTPUT_DIR            GET_KNOB(DEBUG_OUTPUT_DIR)
#define KNOB_JIT_ENABLE_CACHE            GET_KNOB(JIT_ENABLE_CACHE)
#define KNOB_JIT_OPTIMIZATION_LEVEL      GET_KNOB(JIT_OPTIMIZATION_LEVEL)
#define KNOB_JIT_CACHE_DIR               GET_KNOB(JIT_CACHE_DIR)
#define KNOB_TOSS_DRAW                   GET_KNOB(TOSS_DRAW)
#define KNOB_TOSS_QUEUE_FE               GET_KNOB(TOSS_QUEUE_FE)
#define KNOB_TOSS_FETCH                  GET_KNOB(TOSS_FETCH)
#define KNOB_TOSS_IA                     GET_KNOB(TOSS_IA)
#define KNOB_TOSS_VS                     GET_KNOB(TOSS_VS)
#define KNOB_TOSS_SETUP_TRIS             GET_KNOB(TOSS_SETUP_TRIS)
#define KNOB_TOSS_BIN_TRIS               GET_KNOB(TOSS_BIN_TRIS)
#define KNOB_TOSS_RS                     GET_KNOB(TOSS_RS)
#define KNOB_DISABLE_SPLIT_DRAW          GET_KNOB(DISABLE_SPLIT_DRAW)


// clang-format on
