/****************************************************************************
 * Copyright (C) 2016 Intel Corporation.   All Rights Reserved.
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
 * @file gen_ar_eventhandler.hpp
 *
 * @brief Event handler interface.  auto-generated file
 *
 * DO NOT EDIT
 *
 * Generation Command Line:
 *  ./rasterizer/codegen/gen_archrast.py
 *    --proto
 *    ./rasterizer/archrast/events.proto
 *    --proto_private
 *    ./rasterizer/archrast/events_private.proto
 *    --output
 *    rasterizer/archrast/gen_ar_eventhandler.hpp
 *    --gen_eventhandler_hpp
 *
 ******************************************************************************/
// clang-format on
#pragma once

#include "gen_ar_event.hpp"

namespace ArchRast
{
    //////////////////////////////////////////////////////////////////////////
    /// EventHandler - interface for handling events.
    //////////////////////////////////////////////////////////////////////////
    class EventHandler
    {
    public:
        EventHandler() {}
        virtual ~EventHandler() {}

        virtual void FlushDraw(uint32_t drawId) {}

        virtual void Handle(const ThreadStartApiEvent& event) {}
        virtual void Handle(const ThreadStartWorkerEvent& event) {}
        virtual void Handle(const DrawInfoEvent& event) {}
        virtual void Handle(const DispatchEvent& event) {}
        virtual void Handle(const FrameEndEvent& event) {}
        virtual void Handle(const SwrSyncEvent& event) {}
        virtual void Handle(const SwrInvalidateTilesEvent& event) {}
        virtual void Handle(const SwrDiscardRectEvent& event) {}
        virtual void Handle(const SwrStoreTilesEvent& event) {}
        virtual void Handle(const FrontendStatsEvent& event) {}
        virtual void Handle(const BackendStatsEvent& event) {}
        virtual void Handle(const EarlyZSingleSample& event) {}
        virtual void Handle(const LateZSingleSample& event) {}
        virtual void Handle(const EarlyStencilSingleSample& event) {}
        virtual void Handle(const LateStencilSingleSample& event) {}
        virtual void Handle(const EarlyZSampleRate& event) {}
        virtual void Handle(const LateZSampleRate& event) {}
        virtual void Handle(const EarlyStencilSampleRate& event) {}
        virtual void Handle(const LateStencilSampleRate& event) {}
        virtual void Handle(const EarlyZ& event) {}
        virtual void Handle(const LateZ& event) {}
        virtual void Handle(const EarlyStencil& event) {}
        virtual void Handle(const LateStencil& event) {}
        virtual void Handle(const EarlyZNullPS& event) {}
        virtual void Handle(const EarlyStencilNullPS& event) {}
        virtual void Handle(const EarlyZPixelRate& event) {}
        virtual void Handle(const LateZPixelRate& event) {}
        virtual void Handle(const EarlyOmZ& event) {}
        virtual void Handle(const EarlyOmStencil& event) {}
        virtual void Handle(const LateOmZ& event) {}
        virtual void Handle(const LateOmStencil& event) {}
        virtual void Handle(const GSInputPrims& event) {}
        virtual void Handle(const GSPrimsGen& event) {}
        virtual void Handle(const GSVertsInput& event) {}
        virtual void Handle(const TessPrims& event) {}
        virtual void Handle(const RasterTiles& event) {}
        virtual void Handle(const ClipperEvent& event) {}
        virtual void Handle(const CullEvent& event) {}
        virtual void Handle(const AlphaEvent& event) {}
        virtual void Handle(const VSInfo& event) {}
        virtual void Handle(const HSInfo& event) {}
        virtual void Handle(const DSInfo& event) {}
        virtual void Handle(const GSInfo& event) {}
        virtual void Handle(const PSInfo& event) {}
        virtual void Handle(const CSInfo& event) {}
        virtual void Handle(const EarlyDepthStencilInfoSingleSample& event) {}
        virtual void Handle(const EarlyDepthStencilInfoSampleRate& event) {}
        virtual void Handle(const EarlyDepthStencilInfoNullPS& event) {}
        virtual void Handle(const LateDepthStencilInfoSingleSample& event) {}
        virtual void Handle(const LateDepthStencilInfoSampleRate& event) {}
        virtual void Handle(const LateDepthStencilInfoNullPS& event) {}
        virtual void Handle(const EarlyDepthInfoPixelRate& event) {}
        virtual void Handle(const LateDepthInfoPixelRate& event) {}
        virtual void Handle(const BackendDrawEndEvent& event) {}
        virtual void Handle(const FrontendDrawEndEvent& event) {}
        virtual void Handle(const TessPrimCount& event) {}
        virtual void Handle(const RasterTileCount& event) {}
        virtual void Handle(const GSPrimInfo& event) {}
        virtual void Handle(const ClipInfoEvent& event) {}
        virtual void Handle(const CullInfoEvent& event) {}
        virtual void Handle(const AlphaInfoEvent& event) {}
        virtual void Handle(const DrawInstancedEvent& event) {}
        virtual void Handle(const DrawIndexedInstancedEvent& event) {}
        virtual void Handle(const VSStats& event) {}
        virtual void Handle(const HSStats& event) {}
        virtual void Handle(const DSStats& event) {}
        virtual void Handle(const GSStats& event) {}
        virtual void Handle(const PSStats& event) {}
        virtual void Handle(const CSStats& event) {}
    };
} // namespace ArchRast
// clan-format off
