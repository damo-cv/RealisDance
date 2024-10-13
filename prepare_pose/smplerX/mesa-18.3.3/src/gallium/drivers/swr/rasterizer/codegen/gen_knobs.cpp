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
 * @file gen_knobs.cpp
 *
 * @brief Dynamic Knobs for Core.
 *
 * ======================= AUTO GENERATED: DO NOT EDIT !!! ====================
 *
 * Generation Command Line:
 *  ./rasterizer/codegen/gen_knobs.py
 *    --output
 *    rasterizer/codegen/gen_knobs.cpp
 *    --gen_cpp
 *
 ******************************************************************************/
// clang-format off

#include <core/knobs_init.h>
#include <common/os.h>
#include <sstream>
#include <iomanip>
#include <regex>
#include <core/utils.h>

//========================================================
// Implementation
//========================================================
void KnobBase::autoExpandEnvironmentVariables(std::string& text)
{
#if (__GNUC__) && (GCC_VERSION < 409000)
    // <regex> isn't implemented prior to gcc-4.9.0
    // unix style variable replacement
    size_t start;
    while ((start = text.find("${")) != std::string::npos)
    {
        size_t end = text.find("}");
        if (end == std::string::npos)
            break;
        const std::string var = GetEnv(text.substr(start + 2, end - start - 2));
        text.replace(start, end - start + 1, var);
    }
    // win32 style variable replacement
    while ((start = text.find("%")) != std::string::npos)
    {
        size_t end = text.find("%", start + 1);
        if (end == std::string::npos)
            break;
        const std::string var = GetEnv(text.substr(start + 1, end - start - 1));
        text.replace(start, end - start + 1, var);
    }
#else
    {
        // unix style variable replacement
        static std::regex env("\\$\\{([^}]+)\\}");
        std::smatch       match;
        while (std::regex_search(text, match, env))
        {
            const std::string var = GetEnv(match[1].str());
            // certain combinations of gcc/libstd++ have problems with this
            // text.replace(match[0].first, match[0].second, var);
            text.replace(match.prefix().length(), match[0].length(), var);
        }
    }
    {
        // win32 style variable replacement
        static std::regex env("\\%([^}]+)\\%");
        std::smatch       match;
        while (std::regex_search(text, match, env))
        {
            const std::string var = GetEnv(match[1].str());
            // certain combinations of gcc/libstd++ have problems with this
            // text.replace(match[0].first, match[0].second, var);
            text.replace(match.prefix().length(), match[0].length(), var);
        }
    }
#endif
}

//========================================================
// Static Data Members
//========================================================
GlobalKnobs g_GlobalKnobs;

//========================================================
// Knob Initialization
//========================================================
GlobalKnobs::GlobalKnobs()
{
    InitKnob(ENABLE_ASSERT_DIALOGS);
    InitKnob(SINGLE_THREADED);
    InitKnob(DUMP_SHADER_IR);
    InitKnob(USE_GENERIC_STORETILE);
    InitKnob(FAST_CLEAR);
    InitKnob(MAX_NUMA_NODES);
    InitKnob(MAX_CORES_PER_NUMA_NODE);
    InitKnob(MAX_THREADS_PER_CORE);
    InitKnob(MAX_WORKER_THREADS);
    InitKnob(BASE_NUMA_NODE);
    InitKnob(BASE_CORE);
    InitKnob(BASE_THREAD);
    InitKnob(BUCKETS_START_FRAME);
    InitKnob(BUCKETS_END_FRAME);
    InitKnob(WORKER_SPIN_LOOP_COUNT);
    InitKnob(MAX_DRAWS_IN_FLIGHT);
    InitKnob(MAX_PRIMS_PER_DRAW);
    InitKnob(MAX_TESS_PRIMS_PER_DRAW);
    InitKnob(DEBUG_OUTPUT_DIR);
    InitKnob(JIT_ENABLE_CACHE);
    InitKnob(JIT_OPTIMIZATION_LEVEL);
    InitKnob(JIT_CACHE_DIR);
    InitKnob(TOSS_DRAW);
    InitKnob(TOSS_QUEUE_FE);
    InitKnob(TOSS_FETCH);
    InitKnob(TOSS_IA);
    InitKnob(TOSS_VS);
    InitKnob(TOSS_SETUP_TRIS);
    InitKnob(TOSS_BIN_TRIS);
    InitKnob(TOSS_RS);
    InitKnob(DISABLE_SPLIT_DRAW);
}

//========================================================
// Knob Display (Convert to String)
//========================================================
std::string GlobalKnobs::ToString(const char* optPerLinePrefix)
{
    std::basic_stringstream<char> str;
    str << std::showbase << std::setprecision(1) << std::fixed;

    if (optPerLinePrefix == nullptr)
    {
        optPerLinePrefix = "";
    }

    str << optPerLinePrefix << "KNOB_ENABLE_ASSERT_DIALOGS:      ";
    str << (KNOB_ENABLE_ASSERT_DIALOGS ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_SINGLE_THREADED:            ";
    str << (KNOB_SINGLE_THREADED ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_DUMP_SHADER_IR:             ";
    str << (KNOB_DUMP_SHADER_IR ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_USE_GENERIC_STORETILE:      ";
    str << (KNOB_USE_GENERIC_STORETILE ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_FAST_CLEAR:                 ";
    str << (KNOB_FAST_CLEAR ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_MAX_NUMA_NODES:             ";
    str << std::hex << std::setw(11) << std::left << KNOB_MAX_NUMA_NODES;
    str << std::dec << KNOB_MAX_NUMA_NODES << "\n";
    str << optPerLinePrefix << "KNOB_MAX_CORES_PER_NUMA_NODE:    ";
    str << std::hex << std::setw(11) << std::left << KNOB_MAX_CORES_PER_NUMA_NODE;
    str << std::dec << KNOB_MAX_CORES_PER_NUMA_NODE << "\n";
    str << optPerLinePrefix << "KNOB_MAX_THREADS_PER_CORE:       ";
    str << std::hex << std::setw(11) << std::left << KNOB_MAX_THREADS_PER_CORE;
    str << std::dec << KNOB_MAX_THREADS_PER_CORE << "\n";
    str << optPerLinePrefix << "KNOB_MAX_WORKER_THREADS:         ";
    str << std::hex << std::setw(11) << std::left << KNOB_MAX_WORKER_THREADS;
    str << std::dec << KNOB_MAX_WORKER_THREADS << "\n";
    str << optPerLinePrefix << "KNOB_BASE_NUMA_NODE:             ";
    str << std::hex << std::setw(11) << std::left << KNOB_BASE_NUMA_NODE;
    str << std::dec << KNOB_BASE_NUMA_NODE << "\n";
    str << optPerLinePrefix << "KNOB_BASE_CORE:                  ";
    str << std::hex << std::setw(11) << std::left << KNOB_BASE_CORE;
    str << std::dec << KNOB_BASE_CORE << "\n";
    str << optPerLinePrefix << "KNOB_BASE_THREAD:                ";
    str << std::hex << std::setw(11) << std::left << KNOB_BASE_THREAD;
    str << std::dec << KNOB_BASE_THREAD << "\n";
    str << optPerLinePrefix << "KNOB_BUCKETS_START_FRAME:        ";
    str << std::hex << std::setw(11) << std::left << KNOB_BUCKETS_START_FRAME;
    str << std::dec << KNOB_BUCKETS_START_FRAME << "\n";
    str << optPerLinePrefix << "KNOB_BUCKETS_END_FRAME:          ";
    str << std::hex << std::setw(11) << std::left << KNOB_BUCKETS_END_FRAME;
    str << std::dec << KNOB_BUCKETS_END_FRAME << "\n";
    str << optPerLinePrefix << "KNOB_WORKER_SPIN_LOOP_COUNT:     ";
    str << std::hex << std::setw(11) << std::left << KNOB_WORKER_SPIN_LOOP_COUNT;
    str << std::dec << KNOB_WORKER_SPIN_LOOP_COUNT << "\n";
    str << optPerLinePrefix << "KNOB_MAX_DRAWS_IN_FLIGHT:        ";
    str << std::hex << std::setw(11) << std::left << KNOB_MAX_DRAWS_IN_FLIGHT;
    str << std::dec << KNOB_MAX_DRAWS_IN_FLIGHT << "\n";
    str << optPerLinePrefix << "KNOB_MAX_PRIMS_PER_DRAW:         ";
    str << std::hex << std::setw(11) << std::left << KNOB_MAX_PRIMS_PER_DRAW;
    str << std::dec << KNOB_MAX_PRIMS_PER_DRAW << "\n";
    str << optPerLinePrefix << "KNOB_MAX_TESS_PRIMS_PER_DRAW:    ";
    str << std::hex << std::setw(11) << std::left << KNOB_MAX_TESS_PRIMS_PER_DRAW;
    str << std::dec << KNOB_MAX_TESS_PRIMS_PER_DRAW << "\n";
    str << optPerLinePrefix << "KNOB_DEBUG_OUTPUT_DIR:           ";
    str << KNOB_DEBUG_OUTPUT_DIR << "\n";
    str << optPerLinePrefix << "KNOB_JIT_ENABLE_CACHE:           ";
    str << (KNOB_JIT_ENABLE_CACHE ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_JIT_OPTIMIZATION_LEVEL:     ";
    str << std::hex << std::setw(11) << std::left << KNOB_JIT_OPTIMIZATION_LEVEL;
    str << std::dec << KNOB_JIT_OPTIMIZATION_LEVEL << "\n";
    str << optPerLinePrefix << "KNOB_JIT_CACHE_DIR:              ";
    str << KNOB_JIT_CACHE_DIR << "\n";
    str << optPerLinePrefix << "KNOB_TOSS_DRAW:                  ";
    str << (KNOB_TOSS_DRAW ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_TOSS_QUEUE_FE:              ";
    str << (KNOB_TOSS_QUEUE_FE ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_TOSS_FETCH:                 ";
    str << (KNOB_TOSS_FETCH ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_TOSS_IA:                    ";
    str << (KNOB_TOSS_IA ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_TOSS_VS:                    ";
    str << (KNOB_TOSS_VS ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_TOSS_SETUP_TRIS:            ";
    str << (KNOB_TOSS_SETUP_TRIS ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_TOSS_BIN_TRIS:              ";
    str << (KNOB_TOSS_BIN_TRIS ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_TOSS_RS:                    ";
    str << (KNOB_TOSS_RS ? "+\n" : "-\n");
    str << optPerLinePrefix << "KNOB_DISABLE_SPLIT_DRAW:         ";
    str << (KNOB_DISABLE_SPLIT_DRAW ? "+\n" : "-\n");
    str << std::ends;

    return str.str();
}

// clang-format on
