/****************************************************************************
 * Copyright (C) 2014-2015 Intel Corporation.   All Rights Reserved.
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
 ****************************************************************************/

#pragma once
#include "knobs.h"

#include "common/os.h"
#include "common/rdtsc_buckets.h"

#include <vector>

///////////////////////////////////////////////////////////////////////////////
// NOTE:  This enum MUST be kept in sync with gCoreBuckets in rdtsc_core.cpp
///////////////////////////////////////////////////////////////////////////////
enum CORE_BUCKETS
{
    APIClearRenderTarget,
    APIDraw,
    APIDrawWakeAllThreads,
    APIDrawIndexed,
    APIDispatch,
    APIStoreTiles,
    APIGetDrawContext,
    APISync,
    APIWaitForIdle,
    FEProcessDraw,
    FEProcessDrawIndexed,
    FEFetchShader,
    FEVertexShader,
    FEHullShader,
    FETessellation,
    FEDomainShader,
    FEGeometryShader,
    FEStreamout,
    FEPAAssemble,
    FEBinPoints,
    FEBinLines,
    FEBinTriangles,
    FETriangleSetup,
    FEViewportCull,
    FEGuardbandClip,
    FEClipPoints,
    FEClipLines,
    FEClipTriangles,
    FEClipRectangles,
    FECullZeroAreaAndBackface,
    FECullBetweenCenters,
    FEEarlyRastEnter,
    FEEarlyRastExit,
    FEProcessStoreTiles,
    FEProcessInvalidateTiles,
    WorkerWorkOnFifoBE,
    WorkerFoundWork,
    BELoadTiles,
    BEDispatch,
    BEClear,
    BERasterizeLine,
    BERasterizeTriangle,
    BETriangleSetup,
    BEStepSetup,
    BECullZeroArea,
    BEEmptyTriangle,
    BETrivialAccept,
    BETrivialReject,
    BERasterizePartial,
    BEPixelBackend,
    BESetup,
    BEBarycentric,
    BEEarlyDepthTest,
    BEPixelShader,
    BESingleSampleBackend,
    BEPixelRateBackend,
    BESampleRateBackend,
    BENullBackend,
    BELateDepthTest,
    BEOutputMerger,
    BEStoreTiles,
    BEEndTile,

    NumBuckets
};

void rdtscReset();
void rdtscInit(int threadId);
void rdtscStart(uint32_t bucketId);
void rdtscStop(uint32_t bucketId, uint32_t count, uint64_t drawId);
void rdtscEvent(uint32_t bucketId, uint32_t count1, uint32_t count2);
void rdtscEndFrame();

#ifdef KNOB_ENABLE_RDTSC
#define RDTSC_RESET() rdtscReset()
#define RDTSC_INIT(threadId) rdtscInit(threadId)
#define RDTSC_START(bucket) rdtscStart(bucket)
#define RDTSC_STOP(bucket, count, draw) rdtscStop(bucket, count, draw)
#define RDTSC_EVENT(bucket, count1, count2) rdtscEvent(bucket, count1, count2)
#define RDTSC_ENDFRAME() rdtscEndFrame()
#else
#define RDTSC_RESET()
#define RDTSC_INIT(threadId)
#define RDTSC_START(bucket)
#define RDTSC_STOP(bucket, count, draw)
#define RDTSC_EVENT(bucket, count1, count2)
#define RDTSC_ENDFRAME()
#endif

extern std::vector<uint32_t> gBucketMap;
extern BucketManager         gBucketMgr;
extern BUCKET_DESC           gCoreBuckets[];
extern uint32_t              gCurrentFrame;
extern bool                  gBucketsInitialized;

INLINE void rdtscReset()
{
    gCurrentFrame = 0;
    gBucketMgr.ClearThreads();
}

INLINE void rdtscInit(int threadId)
{
    // register all the buckets once
    if (!gBucketsInitialized && (threadId == 0))
    {
        gBucketMap.resize(NumBuckets);
        for (uint32_t i = 0; i < NumBuckets; ++i)
        {
            gBucketMap[i] = gBucketMgr.RegisterBucket(gCoreBuckets[i]);
        }
        gBucketsInitialized = true;
    }

    std::string name = threadId == 0 ? "API" : "WORKER";
    gBucketMgr.RegisterThread(name);
}

INLINE void rdtscStart(uint32_t bucketId)
{
    uint32_t id = gBucketMap[bucketId];
    gBucketMgr.StartBucket(id);
}

INLINE void rdtscStop(uint32_t bucketId, uint32_t count, uint64_t drawId)
{
    uint32_t id = gBucketMap[bucketId];
    gBucketMgr.StopBucket(id);
}

INLINE void rdtscEvent(uint32_t bucketId, uint32_t count1, uint32_t count2)
{
    uint32_t id = gBucketMap[bucketId];
    gBucketMgr.AddEvent(id, count1);
}

INLINE void rdtscEndFrame()
{
    gCurrentFrame++;

    if (gCurrentFrame == KNOB_BUCKETS_START_FRAME &&
        KNOB_BUCKETS_START_FRAME < KNOB_BUCKETS_END_FRAME)
    {
        gBucketMgr.StartCapture();
    }

    if (gCurrentFrame == KNOB_BUCKETS_END_FRAME &&
        KNOB_BUCKETS_START_FRAME < KNOB_BUCKETS_END_FRAME)
    {
        gBucketMgr.StopCapture();
        gBucketMgr.PrintReport("rdtsc.txt");
    }
}
