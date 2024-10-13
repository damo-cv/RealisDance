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
 * @file gen_ar_event.hpp
 *
 * @brief Definitions for events.  auto-generated file
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
 *    rasterizer/archrast/gen_ar_event.hpp
 *    --gen_event_hpp
 *
 ******************************************************************************/
// clang-format off
#pragma once

#include "common/os.h"
#include "core/state.h"

namespace ArchRast
{
    enum AR_DRAW_TYPE
    {
        Instanced = 0,
        IndexedInstanced = 1,
        InstancedSplit = 2,
        IndexedInstancedSplit = 3
    };

    // Forward decl
    class EventHandler;

    //////////////////////////////////////////////////////////////////////////
    /// Event - interface for handling events.
    //////////////////////////////////////////////////////////////////////////
    struct Event
    {
        Event() {}
        virtual ~Event() {}

        virtual void Accept(EventHandler* pHandler) const = 0;
    };

    //////////////////////////////////////////////////////////////////////////
    /// ThreadStartApiEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct ThreadStartApiEventData
    {
        // Fields
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// ThreadStartApiEvent
    //////////////////////////////////////////////////////////////////////////
    struct ThreadStartApiEvent : Event
    {
        ThreadStartApiEventData data;

        // Constructor
        ThreadStartApiEvent(
        )
        {
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// ThreadStartWorkerEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct ThreadStartWorkerEventData
    {
        // Fields
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// ThreadStartWorkerEvent
    //////////////////////////////////////////////////////////////////////////
    struct ThreadStartWorkerEvent : Event
    {
        ThreadStartWorkerEventData data;

        // Constructor
        ThreadStartWorkerEvent(
        )
        {
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// DrawInfoEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct DrawInfoEventData
    {
        // Fields
        uint32_t drawId;
        AR_DRAW_TYPE type;
        uint32_t topology;
        uint32_t numVertices;
        uint32_t numIndices;
        int32_t indexOffset;
        int32_t baseVertex;
        uint32_t numInstances;
        uint32_t startInstance;
        uint32_t tsEnable;
        uint32_t gsEnable;
        uint32_t soEnable;
        uint32_t soTopology;
        uint32_t splitId;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// DrawInfoEvent
    //////////////////////////////////////////////////////////////////////////
    struct DrawInfoEvent : Event
    {
        DrawInfoEventData data;

        // Constructor
        DrawInfoEvent(
            uint32_t drawId,
            AR_DRAW_TYPE type,
            uint32_t topology,
            uint32_t numVertices,
            uint32_t numIndices,
            int32_t indexOffset,
            int32_t baseVertex,
            uint32_t numInstances,
            uint32_t startInstance,
            uint32_t tsEnable,
            uint32_t gsEnable,
            uint32_t soEnable,
            uint32_t soTopology,
            uint32_t splitId
        )
        {
            data.drawId = drawId;
            data.type = type;
            data.topology = topology;
            data.numVertices = numVertices;
            data.numIndices = numIndices;
            data.indexOffset = indexOffset;
            data.baseVertex = baseVertex;
            data.numInstances = numInstances;
            data.startInstance = startInstance;
            data.tsEnable = tsEnable;
            data.gsEnable = gsEnable;
            data.soEnable = soEnable;
            data.soTopology = soTopology;
            data.splitId = splitId;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// DispatchEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct DispatchEventData
    {
        // Fields
        uint32_t drawId;
        uint32_t threadGroupCountX;
        uint32_t threadGroupCountY;
        uint32_t threadGroupCountZ;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// DispatchEvent
    //////////////////////////////////////////////////////////////////////////
    struct DispatchEvent : Event
    {
        DispatchEventData data;

        // Constructor
        DispatchEvent(
            uint32_t drawId,
            uint32_t threadGroupCountX,
            uint32_t threadGroupCountY,
            uint32_t threadGroupCountZ
        )
        {
            data.drawId = drawId;
            data.threadGroupCountX = threadGroupCountX;
            data.threadGroupCountY = threadGroupCountY;
            data.threadGroupCountZ = threadGroupCountZ;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// FrameEndEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct FrameEndEventData
    {
        // Fields
        uint32_t frameId;
        uint32_t nextDrawId;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// FrameEndEvent
    //////////////////////////////////////////////////////////////////////////
    struct FrameEndEvent : Event
    {
        FrameEndEventData data;

        // Constructor
        FrameEndEvent(
            uint32_t frameId,
            uint32_t nextDrawId
        )
        {
            data.frameId = frameId;
            data.nextDrawId = nextDrawId;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// SwrSyncEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct SwrSyncEventData
    {
        // Fields
        uint32_t drawId;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// SwrSyncEvent
    //////////////////////////////////////////////////////////////////////////
    struct SwrSyncEvent : Event
    {
        SwrSyncEventData data;

        // Constructor
        SwrSyncEvent(
            uint32_t drawId
        )
        {
            data.drawId = drawId;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// SwrInvalidateTilesEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct SwrInvalidateTilesEventData
    {
        // Fields
        uint32_t drawId;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// SwrInvalidateTilesEvent
    //////////////////////////////////////////////////////////////////////////
    struct SwrInvalidateTilesEvent : Event
    {
        SwrInvalidateTilesEventData data;

        // Constructor
        SwrInvalidateTilesEvent(
            uint32_t drawId
        )
        {
            data.drawId = drawId;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// SwrDiscardRectEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct SwrDiscardRectEventData
    {
        // Fields
        uint32_t drawId;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// SwrDiscardRectEvent
    //////////////////////////////////////////////////////////////////////////
    struct SwrDiscardRectEvent : Event
    {
        SwrDiscardRectEventData data;

        // Constructor
        SwrDiscardRectEvent(
            uint32_t drawId
        )
        {
            data.drawId = drawId;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// SwrStoreTilesEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct SwrStoreTilesEventData
    {
        // Fields
        uint32_t drawId;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// SwrStoreTilesEvent
    //////////////////////////////////////////////////////////////////////////
    struct SwrStoreTilesEvent : Event
    {
        SwrStoreTilesEventData data;

        // Constructor
        SwrStoreTilesEvent(
            uint32_t drawId
        )
        {
            data.drawId = drawId;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// FrontendStatsEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct FrontendStatsEventData
    {
        // Fields
        uint32_t drawId;
        uint64_t IaVertices;
        uint64_t IaPrimitives;
        uint64_t VsInvocations;
        uint64_t HsInvocations;
        uint64_t DsInvocations;
        uint64_t GsInvocations;
        uint64_t GsPrimitives;
        uint64_t CInvocations;
        uint64_t CPrimitives;
        uint64_t SoPrimStorageNeeded0;
        uint64_t SoPrimStorageNeeded1;
        uint64_t SoPrimStorageNeeded2;
        uint64_t SoPrimStorageNeeded3;
        uint64_t SoNumPrimsWritten0;
        uint64_t SoNumPrimsWritten1;
        uint64_t SoNumPrimsWritten2;
        uint64_t SoNumPrimsWritten3;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// FrontendStatsEvent
    //////////////////////////////////////////////////////////////////////////
    struct FrontendStatsEvent : Event
    {
        FrontendStatsEventData data;

        // Constructor
        FrontendStatsEvent(
            uint32_t drawId,
            uint64_t IaVertices,
            uint64_t IaPrimitives,
            uint64_t VsInvocations,
            uint64_t HsInvocations,
            uint64_t DsInvocations,
            uint64_t GsInvocations,
            uint64_t GsPrimitives,
            uint64_t CInvocations,
            uint64_t CPrimitives,
            uint64_t SoPrimStorageNeeded0,
            uint64_t SoPrimStorageNeeded1,
            uint64_t SoPrimStorageNeeded2,
            uint64_t SoPrimStorageNeeded3,
            uint64_t SoNumPrimsWritten0,
            uint64_t SoNumPrimsWritten1,
            uint64_t SoNumPrimsWritten2,
            uint64_t SoNumPrimsWritten3
        )
        {
            data.drawId = drawId;
            data.IaVertices = IaVertices;
            data.IaPrimitives = IaPrimitives;
            data.VsInvocations = VsInvocations;
            data.HsInvocations = HsInvocations;
            data.DsInvocations = DsInvocations;
            data.GsInvocations = GsInvocations;
            data.GsPrimitives = GsPrimitives;
            data.CInvocations = CInvocations;
            data.CPrimitives = CPrimitives;
            data.SoPrimStorageNeeded0 = SoPrimStorageNeeded0;
            data.SoPrimStorageNeeded1 = SoPrimStorageNeeded1;
            data.SoPrimStorageNeeded2 = SoPrimStorageNeeded2;
            data.SoPrimStorageNeeded3 = SoPrimStorageNeeded3;
            data.SoNumPrimsWritten0 = SoNumPrimsWritten0;
            data.SoNumPrimsWritten1 = SoNumPrimsWritten1;
            data.SoNumPrimsWritten2 = SoNumPrimsWritten2;
            data.SoNumPrimsWritten3 = SoNumPrimsWritten3;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// BackendStatsEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct BackendStatsEventData
    {
        // Fields
        uint32_t drawId;
        uint64_t DepthPassCount;
        uint64_t PsInvocations;
        uint64_t CsInvocations;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// BackendStatsEvent
    //////////////////////////////////////////////////////////////////////////
    struct BackendStatsEvent : Event
    {
        BackendStatsEventData data;

        // Constructor
        BackendStatsEvent(
            uint32_t drawId,
            uint64_t DepthPassCount,
            uint64_t PsInvocations,
            uint64_t CsInvocations
        )
        {
            data.drawId = drawId;
            data.DepthPassCount = DepthPassCount;
            data.PsInvocations = PsInvocations;
            data.CsInvocations = CsInvocations;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyZSingleSampleData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyZSingleSampleData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyZSingleSample
    //////////////////////////////////////////////////////////////////////////
    struct EarlyZSingleSample : Event
    {
        EarlyZSingleSampleData data;

        // Constructor
        EarlyZSingleSample(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateZSingleSampleData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateZSingleSampleData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateZSingleSample
    //////////////////////////////////////////////////////////////////////////
    struct LateZSingleSample : Event
    {
        LateZSingleSampleData data;

        // Constructor
        LateZSingleSample(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyStencilSingleSampleData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyStencilSingleSampleData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyStencilSingleSample
    //////////////////////////////////////////////////////////////////////////
    struct EarlyStencilSingleSample : Event
    {
        EarlyStencilSingleSampleData data;

        // Constructor
        EarlyStencilSingleSample(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateStencilSingleSampleData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateStencilSingleSampleData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateStencilSingleSample
    //////////////////////////////////////////////////////////////////////////
    struct LateStencilSingleSample : Event
    {
        LateStencilSingleSampleData data;

        // Constructor
        LateStencilSingleSample(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyZSampleRateData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyZSampleRateData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyZSampleRate
    //////////////////////////////////////////////////////////////////////////
    struct EarlyZSampleRate : Event
    {
        EarlyZSampleRateData data;

        // Constructor
        EarlyZSampleRate(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateZSampleRateData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateZSampleRateData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateZSampleRate
    //////////////////////////////////////////////////////////////////////////
    struct LateZSampleRate : Event
    {
        LateZSampleRateData data;

        // Constructor
        LateZSampleRate(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyStencilSampleRateData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyStencilSampleRateData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyStencilSampleRate
    //////////////////////////////////////////////////////////////////////////
    struct EarlyStencilSampleRate : Event
    {
        EarlyStencilSampleRateData data;

        // Constructor
        EarlyStencilSampleRate(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateStencilSampleRateData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateStencilSampleRateData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateStencilSampleRate
    //////////////////////////////////////////////////////////////////////////
    struct LateStencilSampleRate : Event
    {
        LateStencilSampleRateData data;

        // Constructor
        LateStencilSampleRate(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyZData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyZData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyZ
    //////////////////////////////////////////////////////////////////////////
    struct EarlyZ : Event
    {
        EarlyZData data;

        // Constructor
        EarlyZ(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateZData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateZData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateZ
    //////////////////////////////////////////////////////////////////////////
    struct LateZ : Event
    {
        LateZData data;

        // Constructor
        LateZ(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyStencilData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyStencilData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyStencil
    //////////////////////////////////////////////////////////////////////////
    struct EarlyStencil : Event
    {
        EarlyStencilData data;

        // Constructor
        EarlyStencil(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateStencilData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateStencilData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateStencil
    //////////////////////////////////////////////////////////////////////////
    struct LateStencil : Event
    {
        LateStencilData data;

        // Constructor
        LateStencil(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyZNullPSData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyZNullPSData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyZNullPS
    //////////////////////////////////////////////////////////////////////////
    struct EarlyZNullPS : Event
    {
        EarlyZNullPSData data;

        // Constructor
        EarlyZNullPS(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyStencilNullPSData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyStencilNullPSData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyStencilNullPS
    //////////////////////////////////////////////////////////////////////////
    struct EarlyStencilNullPS : Event
    {
        EarlyStencilNullPSData data;

        // Constructor
        EarlyStencilNullPS(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyZPixelRateData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyZPixelRateData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyZPixelRate
    //////////////////////////////////////////////////////////////////////////
    struct EarlyZPixelRate : Event
    {
        EarlyZPixelRateData data;

        // Constructor
        EarlyZPixelRate(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateZPixelRateData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateZPixelRateData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateZPixelRate
    //////////////////////////////////////////////////////////////////////////
    struct LateZPixelRate : Event
    {
        LateZPixelRateData data;

        // Constructor
        LateZPixelRate(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyOmZData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyOmZData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyOmZ
    //////////////////////////////////////////////////////////////////////////
    struct EarlyOmZ : Event
    {
        EarlyOmZData data;

        // Constructor
        EarlyOmZ(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyOmStencilData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyOmStencilData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyOmStencil
    //////////////////////////////////////////////////////////////////////////
    struct EarlyOmStencil : Event
    {
        EarlyOmStencilData data;

        // Constructor
        EarlyOmStencil(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateOmZData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateOmZData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateOmZ
    //////////////////////////////////////////////////////////////////////////
    struct LateOmZ : Event
    {
        LateOmZData data;

        // Constructor
        LateOmZ(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateOmStencilData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateOmStencilData
    {
        // Fields
        uint32_t drawId;
        uint64_t passCount;
        uint64_t failCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateOmStencil
    //////////////////////////////////////////////////////////////////////////
    struct LateOmStencil : Event
    {
        LateOmStencilData data;

        // Constructor
        LateOmStencil(
            uint32_t drawId,
            uint64_t passCount,
            uint64_t failCount
        )
        {
            data.drawId = drawId;
            data.passCount = passCount;
            data.failCount = failCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// GSInputPrimsData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct GSInputPrimsData
    {
        // Fields
        uint32_t drawId;
        uint64_t inputPrimCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// GSInputPrims
    //////////////////////////////////////////////////////////////////////////
    struct GSInputPrims : Event
    {
        GSInputPrimsData data;

        // Constructor
        GSInputPrims(
            uint32_t drawId,
            uint64_t inputPrimCount
        )
        {
            data.drawId = drawId;
            data.inputPrimCount = inputPrimCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// GSPrimsGenData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct GSPrimsGenData
    {
        // Fields
        uint32_t drawId;
        uint64_t primGeneratedCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// GSPrimsGen
    //////////////////////////////////////////////////////////////////////////
    struct GSPrimsGen : Event
    {
        GSPrimsGenData data;

        // Constructor
        GSPrimsGen(
            uint32_t drawId,
            uint64_t primGeneratedCount
        )
        {
            data.drawId = drawId;
            data.primGeneratedCount = primGeneratedCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// GSVertsInputData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct GSVertsInputData
    {
        // Fields
        uint32_t drawId;
        uint64_t vertsInput;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// GSVertsInput
    //////////////////////////////////////////////////////////////////////////
    struct GSVertsInput : Event
    {
        GSVertsInputData data;

        // Constructor
        GSVertsInput(
            uint32_t drawId,
            uint64_t vertsInput
        )
        {
            data.drawId = drawId;
            data.vertsInput = vertsInput;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// TessPrimsData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct TessPrimsData
    {
        // Fields
        uint32_t drawId;
        uint64_t primCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// TessPrims
    //////////////////////////////////////////////////////////////////////////
    struct TessPrims : Event
    {
        TessPrimsData data;

        // Constructor
        TessPrims(
            uint32_t drawId,
            uint64_t primCount
        )
        {
            data.drawId = drawId;
            data.primCount = primCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// RasterTilesData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct RasterTilesData
    {
        // Fields
        uint32_t drawId;
        uint32_t rastTileCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// RasterTiles
    //////////////////////////////////////////////////////////////////////////
    struct RasterTiles : Event
    {
        RasterTilesData data;

        // Constructor
        RasterTiles(
            uint32_t drawId,
            uint32_t rastTileCount
        )
        {
            data.drawId = drawId;
            data.rastTileCount = rastTileCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// ClipperEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct ClipperEventData
    {
        // Fields
        uint32_t drawId;
        uint32_t trivialRejectCount;
        uint32_t trivialAcceptCount;
        uint32_t mustClipCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// ClipperEvent
    //////////////////////////////////////////////////////////////////////////
    struct ClipperEvent : Event
    {
        ClipperEventData data;

        // Constructor
        ClipperEvent(
            uint32_t drawId,
            uint32_t trivialRejectCount,
            uint32_t trivialAcceptCount,
            uint32_t mustClipCount
        )
        {
            data.drawId = drawId;
            data.trivialRejectCount = trivialRejectCount;
            data.trivialAcceptCount = trivialAcceptCount;
            data.mustClipCount = mustClipCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// CullEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct CullEventData
    {
        // Fields
        uint32_t drawId;
        uint64_t backfacePrimCount;
        uint64_t degeneratePrimCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// CullEvent
    //////////////////////////////////////////////////////////////////////////
    struct CullEvent : Event
    {
        CullEventData data;

        // Constructor
        CullEvent(
            uint32_t drawId,
            uint64_t backfacePrimCount,
            uint64_t degeneratePrimCount
        )
        {
            data.drawId = drawId;
            data.backfacePrimCount = backfacePrimCount;
            data.degeneratePrimCount = degeneratePrimCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// AlphaEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct AlphaEventData
    {
        // Fields
        uint32_t drawId;
        uint32_t alphaTestCount;
        uint32_t alphaBlendCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// AlphaEvent
    //////////////////////////////////////////////////////////////////////////
    struct AlphaEvent : Event
    {
        AlphaEventData data;

        // Constructor
        AlphaEvent(
            uint32_t drawId,
            uint32_t alphaTestCount,
            uint32_t alphaBlendCount
        )
        {
            data.drawId = drawId;
            data.alphaTestCount = alphaTestCount;
            data.alphaBlendCount = alphaBlendCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// VSInfoData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct VSInfoData
    {
        // Fields
        uint32_t drawId;
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// VSInfo
    //////////////////////////////////////////////////////////////////////////
    struct VSInfo : Event
    {
        VSInfoData data;

        // Constructor
        VSInfo(
            uint32_t drawId,
            uint32_t numInstExecuted
        )
        {
            data.drawId = drawId;
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// HSInfoData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct HSInfoData
    {
        // Fields
        uint32_t drawId;
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// HSInfo
    //////////////////////////////////////////////////////////////////////////
    struct HSInfo : Event
    {
        HSInfoData data;

        // Constructor
        HSInfo(
            uint32_t drawId,
            uint32_t numInstExecuted
        )
        {
            data.drawId = drawId;
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// DSInfoData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct DSInfoData
    {
        // Fields
        uint32_t drawId;
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// DSInfo
    //////////////////////////////////////////////////////////////////////////
    struct DSInfo : Event
    {
        DSInfoData data;

        // Constructor
        DSInfo(
            uint32_t drawId,
            uint32_t numInstExecuted
        )
        {
            data.drawId = drawId;
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// GSInfoData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct GSInfoData
    {
        // Fields
        uint32_t drawId;
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// GSInfo
    //////////////////////////////////////////////////////////////////////////
    struct GSInfo : Event
    {
        GSInfoData data;

        // Constructor
        GSInfo(
            uint32_t drawId,
            uint32_t numInstExecuted
        )
        {
            data.drawId = drawId;
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// PSInfoData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct PSInfoData
    {
        // Fields
        uint32_t drawId;
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// PSInfo
    //////////////////////////////////////////////////////////////////////////
    struct PSInfo : Event
    {
        PSInfoData data;

        // Constructor
        PSInfo(
            uint32_t drawId,
            uint32_t numInstExecuted
        )
        {
            data.drawId = drawId;
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// CSInfoData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct CSInfoData
    {
        // Fields
        uint32_t drawId;
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// CSInfo
    //////////////////////////////////////////////////////////////////////////
    struct CSInfo : Event
    {
        CSInfoData data;

        // Constructor
        CSInfo(
            uint32_t drawId,
            uint32_t numInstExecuted
        )
        {
            data.drawId = drawId;
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyDepthStencilInfoSingleSampleData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyDepthStencilInfoSingleSampleData
    {
        // Fields
        uint64_t depthPassMask;
        uint64_t stencilPassMask;
        uint64_t coverageMask;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyDepthStencilInfoSingleSample
    //////////////////////////////////////////////////////////////////////////
    struct EarlyDepthStencilInfoSingleSample : Event
    {
        EarlyDepthStencilInfoSingleSampleData data;

        // Constructor
        EarlyDepthStencilInfoSingleSample(
            uint64_t depthPassMask,
            uint64_t stencilPassMask,
            uint64_t coverageMask
        )
        {
            data.depthPassMask = depthPassMask;
            data.stencilPassMask = stencilPassMask;
            data.coverageMask = coverageMask;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyDepthStencilInfoSampleRateData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyDepthStencilInfoSampleRateData
    {
        // Fields
        uint64_t depthPassMask;
        uint64_t stencilPassMask;
        uint64_t coverageMask;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyDepthStencilInfoSampleRate
    //////////////////////////////////////////////////////////////////////////
    struct EarlyDepthStencilInfoSampleRate : Event
    {
        EarlyDepthStencilInfoSampleRateData data;

        // Constructor
        EarlyDepthStencilInfoSampleRate(
            uint64_t depthPassMask,
            uint64_t stencilPassMask,
            uint64_t coverageMask
        )
        {
            data.depthPassMask = depthPassMask;
            data.stencilPassMask = stencilPassMask;
            data.coverageMask = coverageMask;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyDepthStencilInfoNullPSData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyDepthStencilInfoNullPSData
    {
        // Fields
        uint64_t depthPassMask;
        uint64_t stencilPassMask;
        uint64_t coverageMask;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyDepthStencilInfoNullPS
    //////////////////////////////////////////////////////////////////////////
    struct EarlyDepthStencilInfoNullPS : Event
    {
        EarlyDepthStencilInfoNullPSData data;

        // Constructor
        EarlyDepthStencilInfoNullPS(
            uint64_t depthPassMask,
            uint64_t stencilPassMask,
            uint64_t coverageMask
        )
        {
            data.depthPassMask = depthPassMask;
            data.stencilPassMask = stencilPassMask;
            data.coverageMask = coverageMask;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateDepthStencilInfoSingleSampleData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateDepthStencilInfoSingleSampleData
    {
        // Fields
        uint64_t depthPassMask;
        uint64_t stencilPassMask;
        uint64_t coverageMask;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateDepthStencilInfoSingleSample
    //////////////////////////////////////////////////////////////////////////
    struct LateDepthStencilInfoSingleSample : Event
    {
        LateDepthStencilInfoSingleSampleData data;

        // Constructor
        LateDepthStencilInfoSingleSample(
            uint64_t depthPassMask,
            uint64_t stencilPassMask,
            uint64_t coverageMask
        )
        {
            data.depthPassMask = depthPassMask;
            data.stencilPassMask = stencilPassMask;
            data.coverageMask = coverageMask;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateDepthStencilInfoSampleRateData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateDepthStencilInfoSampleRateData
    {
        // Fields
        uint64_t depthPassMask;
        uint64_t stencilPassMask;
        uint64_t coverageMask;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateDepthStencilInfoSampleRate
    //////////////////////////////////////////////////////////////////////////
    struct LateDepthStencilInfoSampleRate : Event
    {
        LateDepthStencilInfoSampleRateData data;

        // Constructor
        LateDepthStencilInfoSampleRate(
            uint64_t depthPassMask,
            uint64_t stencilPassMask,
            uint64_t coverageMask
        )
        {
            data.depthPassMask = depthPassMask;
            data.stencilPassMask = stencilPassMask;
            data.coverageMask = coverageMask;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateDepthStencilInfoNullPSData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateDepthStencilInfoNullPSData
    {
        // Fields
        uint64_t depthPassMask;
        uint64_t stencilPassMask;
        uint64_t coverageMask;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateDepthStencilInfoNullPS
    //////////////////////////////////////////////////////////////////////////
    struct LateDepthStencilInfoNullPS : Event
    {
        LateDepthStencilInfoNullPSData data;

        // Constructor
        LateDepthStencilInfoNullPS(
            uint64_t depthPassMask,
            uint64_t stencilPassMask,
            uint64_t coverageMask
        )
        {
            data.depthPassMask = depthPassMask;
            data.stencilPassMask = stencilPassMask;
            data.coverageMask = coverageMask;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// EarlyDepthInfoPixelRateData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct EarlyDepthInfoPixelRateData
    {
        // Fields
        uint64_t depthPassCount;
        uint64_t activeLanes;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// EarlyDepthInfoPixelRate
    //////////////////////////////////////////////////////////////////////////
    struct EarlyDepthInfoPixelRate : Event
    {
        EarlyDepthInfoPixelRateData data;

        // Constructor
        EarlyDepthInfoPixelRate(
            uint64_t depthPassCount,
            uint64_t activeLanes
        )
        {
            data.depthPassCount = depthPassCount;
            data.activeLanes = activeLanes;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// LateDepthInfoPixelRateData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct LateDepthInfoPixelRateData
    {
        // Fields
        uint64_t depthPassCount;
        uint64_t activeLanes;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// LateDepthInfoPixelRate
    //////////////////////////////////////////////////////////////////////////
    struct LateDepthInfoPixelRate : Event
    {
        LateDepthInfoPixelRateData data;

        // Constructor
        LateDepthInfoPixelRate(
            uint64_t depthPassCount,
            uint64_t activeLanes
        )
        {
            data.depthPassCount = depthPassCount;
            data.activeLanes = activeLanes;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// BackendDrawEndEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct BackendDrawEndEventData
    {
        // Fields
        uint32_t drawId;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// BackendDrawEndEvent
    //////////////////////////////////////////////////////////////////////////
    struct BackendDrawEndEvent : Event
    {
        BackendDrawEndEventData data;

        // Constructor
        BackendDrawEndEvent(
            uint32_t drawId
        )
        {
            data.drawId = drawId;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// FrontendDrawEndEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct FrontendDrawEndEventData
    {
        // Fields
        uint32_t drawId;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// FrontendDrawEndEvent
    //////////////////////////////////////////////////////////////////////////
    struct FrontendDrawEndEvent : Event
    {
        FrontendDrawEndEventData data;

        // Constructor
        FrontendDrawEndEvent(
            uint32_t drawId
        )
        {
            data.drawId = drawId;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// TessPrimCountData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct TessPrimCountData
    {
        // Fields
        uint64_t primCount;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// TessPrimCount
    //////////////////////////////////////////////////////////////////////////
    struct TessPrimCount : Event
    {
        TessPrimCountData data;

        // Constructor
        TessPrimCount(
            uint64_t primCount
        )
        {
            data.primCount = primCount;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// RasterTileCountData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct RasterTileCountData
    {
        // Fields
        uint32_t drawId;
        uint64_t rasterTiles;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// RasterTileCount
    //////////////////////////////////////////////////////////////////////////
    struct RasterTileCount : Event
    {
        RasterTileCountData data;

        // Constructor
        RasterTileCount(
            uint32_t drawId,
            uint64_t rasterTiles
        )
        {
            data.drawId = drawId;
            data.rasterTiles = rasterTiles;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// GSPrimInfoData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct GSPrimInfoData
    {
        // Fields
        uint64_t inputPrimCount;
        uint64_t primGeneratedCount;
        uint64_t vertsInput;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// GSPrimInfo
    //////////////////////////////////////////////////////////////////////////
    struct GSPrimInfo : Event
    {
        GSPrimInfoData data;

        // Constructor
        GSPrimInfo(
            uint64_t inputPrimCount,
            uint64_t primGeneratedCount,
            uint64_t vertsInput
        )
        {
            data.inputPrimCount = inputPrimCount;
            data.primGeneratedCount = primGeneratedCount;
            data.vertsInput = vertsInput;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// ClipInfoEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct ClipInfoEventData
    {
        // Fields
        uint32_t numInvocations;
        uint32_t validMask;
        uint32_t clipMask;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// ClipInfoEvent
    //////////////////////////////////////////////////////////////////////////
    struct ClipInfoEvent : Event
    {
        ClipInfoEventData data;

        // Constructor
        ClipInfoEvent(
            uint32_t numInvocations,
            uint32_t validMask,
            uint32_t clipMask
        )
        {
            data.numInvocations = numInvocations;
            data.validMask = validMask;
            data.clipMask = clipMask;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// CullInfoEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct CullInfoEventData
    {
        // Fields
        uint32_t drawId;
        uint64_t degeneratePrimMask;
        uint64_t backfacePrimMask;
        uint32_t validMask;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// CullInfoEvent
    //////////////////////////////////////////////////////////////////////////
    struct CullInfoEvent : Event
    {
        CullInfoEventData data;

        // Constructor
        CullInfoEvent(
            uint32_t drawId,
            uint64_t degeneratePrimMask,
            uint64_t backfacePrimMask,
            uint32_t validMask
        )
        {
            data.drawId = drawId;
            data.degeneratePrimMask = degeneratePrimMask;
            data.backfacePrimMask = backfacePrimMask;
            data.validMask = validMask;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// AlphaInfoEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct AlphaInfoEventData
    {
        // Fields
        uint32_t drawId;
        uint32_t alphaTestEnable;
        uint32_t alphaBlendEnable;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// AlphaInfoEvent
    //////////////////////////////////////////////////////////////////////////
    struct AlphaInfoEvent : Event
    {
        AlphaInfoEventData data;

        // Constructor
        AlphaInfoEvent(
            uint32_t drawId,
            uint32_t alphaTestEnable,
            uint32_t alphaBlendEnable
        )
        {
            data.drawId = drawId;
            data.alphaTestEnable = alphaTestEnable;
            data.alphaBlendEnable = alphaBlendEnable;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// DrawInstancedEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct DrawInstancedEventData
    {
        // Fields
        uint32_t drawId;
        uint32_t topology;
        uint32_t numVertices;
        int32_t startVertex;
        uint32_t numInstances;
        uint32_t startInstance;
        uint32_t tsEnable;
        uint32_t gsEnable;
        uint32_t soEnable;
        uint32_t soTopology;
        uint32_t splitId;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// DrawInstancedEvent
    //////////////////////////////////////////////////////////////////////////
    struct DrawInstancedEvent : Event
    {
        DrawInstancedEventData data;

        // Constructor
        DrawInstancedEvent(
            uint32_t drawId,
            uint32_t topology,
            uint32_t numVertices,
            int32_t startVertex,
            uint32_t numInstances,
            uint32_t startInstance,
            uint32_t tsEnable,
            uint32_t gsEnable,
            uint32_t soEnable,
            uint32_t soTopology,
            uint32_t splitId
        )
        {
            data.drawId = drawId;
            data.topology = topology;
            data.numVertices = numVertices;
            data.startVertex = startVertex;
            data.numInstances = numInstances;
            data.startInstance = startInstance;
            data.tsEnable = tsEnable;
            data.gsEnable = gsEnable;
            data.soEnable = soEnable;
            data.soTopology = soTopology;
            data.splitId = splitId;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// DrawIndexedInstancedEventData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct DrawIndexedInstancedEventData
    {
        // Fields
        uint32_t drawId;
        uint32_t topology;
        uint32_t numIndices;
        int32_t indexOffset;
        int32_t baseVertex;
        uint32_t numInstances;
        uint32_t startInstance;
        uint32_t tsEnable;
        uint32_t gsEnable;
        uint32_t soEnable;
        uint32_t soTopology;
        uint32_t splitId;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// DrawIndexedInstancedEvent
    //////////////////////////////////////////////////////////////////////////
    struct DrawIndexedInstancedEvent : Event
    {
        DrawIndexedInstancedEventData data;

        // Constructor
        DrawIndexedInstancedEvent(
            uint32_t drawId,
            uint32_t topology,
            uint32_t numIndices,
            int32_t indexOffset,
            int32_t baseVertex,
            uint32_t numInstances,
            uint32_t startInstance,
            uint32_t tsEnable,
            uint32_t gsEnable,
            uint32_t soEnable,
            uint32_t soTopology,
            uint32_t splitId
        )
        {
            data.drawId = drawId;
            data.topology = topology;
            data.numIndices = numIndices;
            data.indexOffset = indexOffset;
            data.baseVertex = baseVertex;
            data.numInstances = numInstances;
            data.startInstance = startInstance;
            data.tsEnable = tsEnable;
            data.gsEnable = gsEnable;
            data.soEnable = soEnable;
            data.soTopology = soTopology;
            data.splitId = splitId;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// VSStatsData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct VSStatsData
    {
        // Fields
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// VSStats
    //////////////////////////////////////////////////////////////////////////
    struct VSStats : Event
    {
        VSStatsData data;

        // Constructor
        VSStats(
            uint32_t numInstExecuted
        )
        {
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// HSStatsData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct HSStatsData
    {
        // Fields
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// HSStats
    //////////////////////////////////////////////////////////////////////////
    struct HSStats : Event
    {
        HSStatsData data;

        // Constructor
        HSStats(
            uint32_t numInstExecuted
        )
        {
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// DSStatsData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct DSStatsData
    {
        // Fields
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// DSStats
    //////////////////////////////////////////////////////////////////////////
    struct DSStats : Event
    {
        DSStatsData data;

        // Constructor
        DSStats(
            uint32_t numInstExecuted
        )
        {
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// GSStatsData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct GSStatsData
    {
        // Fields
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// GSStats
    //////////////////////////////////////////////////////////////////////////
    struct GSStats : Event
    {
        GSStatsData data;

        // Constructor
        GSStats(
            uint32_t numInstExecuted
        )
        {
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// PSStatsData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct PSStatsData
    {
        // Fields
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// PSStats
    //////////////////////////////////////////////////////////////////////////
    struct PSStats : Event
    {
        PSStatsData data;

        // Constructor
        PSStats(
            uint32_t numInstExecuted
        )
        {
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };

    //////////////////////////////////////////////////////////////////////////
    /// CSStatsData
    //////////////////////////////////////////////////////////////////////////
#pragma pack(push, 1)
    struct CSStatsData
    {
        // Fields
        uint32_t numInstExecuted;
    };
#pragma pack(pop)

    //////////////////////////////////////////////////////////////////////////
    /// CSStats
    //////////////////////////////////////////////////////////////////////////
    struct CSStats : Event
    {
        CSStatsData data;

        // Constructor
        CSStats(
            uint32_t numInstExecuted
        )
        {
            data.numInstExecuted = numInstExecuted;
        }

        virtual void Accept(EventHandler* pHandler) const;
    };
} // namespace ArchRast
// clang-format on
