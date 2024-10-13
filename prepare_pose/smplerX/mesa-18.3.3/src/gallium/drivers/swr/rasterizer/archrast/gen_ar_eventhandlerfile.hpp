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
 * @file gen_ar_eventhandlerfile.hpp
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
 *    rasterizer/archrast/gen_ar_eventhandlerfile.hpp
 *    --gen_eventhandlerfile_hpp
 *
 ******************************************************************************/
// clang-format off
#pragma once

#include "common/os.h"
#include "gen_ar_eventhandler.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>

namespace ArchRast
{
    //////////////////////////////////////////////////////////////////////////
    /// EventHandlerFile - interface for handling events.
    //////////////////////////////////////////////////////////////////////////
    class EventHandlerFile : public EventHandler
    {
    public:
        EventHandlerFile(uint32_t id) : mBufOffset(0)
        {
#if defined(_WIN32)
            DWORD pid = GetCurrentProcessId();
            TCHAR procname[MAX_PATH];
            GetModuleFileName(NULL, procname, MAX_PATH);
            const char*       pBaseName = strrchr(procname, '\\');
            std::stringstream outDir;
            outDir << KNOB_DEBUG_OUTPUT_DIR << pBaseName << "_" << pid << std::ends;
            mOutputDir = outDir.str();
            if (CreateDirectory(mOutputDir.c_str(), NULL))
            {
                std::cout << std::endl
                          << "ArchRast Dir:       " << mOutputDir << std::endl
                          << std::endl
                          << std::flush;
            }

            // There could be multiple threads creating thread pools. We
            // want to make sure they are uniquly identified by adding in
            // the creator's thread id into the filename.
            std::stringstream fstr;
            fstr << outDir.str().c_str() << "\\ar_event" << std::this_thread::get_id();
            fstr << "_" << id << ".bin" << std::ends;
            mFilename = fstr.str();
#else
            // There could be multiple threads creating thread pools. We
            // want to make sure they are uniquly identified by adding in
            // the creator's thread id into the filename.
            std::stringstream fstr;
            fstr << "/tmp/ar_event" << std::this_thread::get_id();
            fstr << "_" << id << ".bin" << std::ends;
            mFilename = fstr.str();
#endif
        }

        virtual ~EventHandlerFile() { FlushBuffer(); }

        //////////////////////////////////////////////////////////////////////////
        /// @brief Flush buffer to file.
        bool FlushBuffer()
        {
            if (mBufOffset > 0)
            {
                if (mBufOffset == mHeaderBufOffset)
                {
                    // Nothing to flush. Only header has been generated.
                    return false;
                }

                std::ofstream file;
                file.open(mFilename, std::ios::out | std::ios::app | std::ios::binary);

                if (!file.is_open())
                {
                    SWR_INVALID("ArchRast: Could not open event file!");
                    return false;
                }

                file.write((char*)mBuffer, mBufOffset);
                file.close();

                mBufOffset       = 0;
                mHeaderBufOffset = 0; // Reset header offset so its no longer considered.
            }
            return true;
        }

        //////////////////////////////////////////////////////////////////////////
        /// @brief Write event and its payload to the memory buffer.
        void Write(uint32_t eventId, const char* pBlock, uint32_t size)
        {
            if ((mBufOffset + size + sizeof(eventId)) > mBufferSize)
            {
                if (!FlushBuffer())
                {
                    // Don't corrupt what's already in the buffer?
                    /// @todo Maybe add corrupt marker to buffer here in case we can open file in
                    /// future?
                    return;
                }
            }

            memcpy(&mBuffer[mBufOffset], (char*)&eventId, sizeof(eventId));
            mBufOffset += sizeof(eventId);
            memcpy(&mBuffer[mBufOffset], pBlock, size);
            mBufOffset += size;
        }

        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle ThreadStartApiEvent event
        virtual void Handle(const ThreadStartApiEvent& event)
        {
            Write(1, (char*)&event.data, 0);
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle ThreadStartWorkerEvent event
        virtual void Handle(const ThreadStartWorkerEvent& event)
        {
            Write(2, (char*)&event.data, 0);
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle DrawInfoEvent event
        virtual void Handle(const DrawInfoEvent& event)
        {
            Write(3, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle DispatchEvent event
        virtual void Handle(const DispatchEvent& event)
        {
            Write(4, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle FrameEndEvent event
        virtual void Handle(const FrameEndEvent& event)
        {
            Write(5, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle SwrSyncEvent event
        virtual void Handle(const SwrSyncEvent& event)
        {
            Write(6, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle SwrInvalidateTilesEvent event
        virtual void Handle(const SwrInvalidateTilesEvent& event)
        {
            Write(7, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle SwrDiscardRectEvent event
        virtual void Handle(const SwrDiscardRectEvent& event)
        {
            Write(8, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle SwrStoreTilesEvent event
        virtual void Handle(const SwrStoreTilesEvent& event)
        {
            Write(9, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle FrontendStatsEvent event
        virtual void Handle(const FrontendStatsEvent& event)
        {
            Write(10, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle BackendStatsEvent event
        virtual void Handle(const BackendStatsEvent& event)
        {
            Write(11, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyZSingleSample event
        virtual void Handle(const EarlyZSingleSample& event)
        {
            Write(12, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateZSingleSample event
        virtual void Handle(const LateZSingleSample& event)
        {
            Write(13, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyStencilSingleSample event
        virtual void Handle(const EarlyStencilSingleSample& event)
        {
            Write(14, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateStencilSingleSample event
        virtual void Handle(const LateStencilSingleSample& event)
        {
            Write(15, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyZSampleRate event
        virtual void Handle(const EarlyZSampleRate& event)
        {
            Write(16, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateZSampleRate event
        virtual void Handle(const LateZSampleRate& event)
        {
            Write(17, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyStencilSampleRate event
        virtual void Handle(const EarlyStencilSampleRate& event)
        {
            Write(18, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateStencilSampleRate event
        virtual void Handle(const LateStencilSampleRate& event)
        {
            Write(19, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyZ event
        virtual void Handle(const EarlyZ& event)
        {
            Write(20, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateZ event
        virtual void Handle(const LateZ& event)
        {
            Write(21, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyStencil event
        virtual void Handle(const EarlyStencil& event)
        {
            Write(22, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateStencil event
        virtual void Handle(const LateStencil& event)
        {
            Write(23, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyZNullPS event
        virtual void Handle(const EarlyZNullPS& event)
        {
            Write(24, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyStencilNullPS event
        virtual void Handle(const EarlyStencilNullPS& event)
        {
            Write(25, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyZPixelRate event
        virtual void Handle(const EarlyZPixelRate& event)
        {
            Write(26, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateZPixelRate event
        virtual void Handle(const LateZPixelRate& event)
        {
            Write(27, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyOmZ event
        virtual void Handle(const EarlyOmZ& event)
        {
            Write(28, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyOmStencil event
        virtual void Handle(const EarlyOmStencil& event)
        {
            Write(29, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateOmZ event
        virtual void Handle(const LateOmZ& event)
        {
            Write(30, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateOmStencil event
        virtual void Handle(const LateOmStencil& event)
        {
            Write(31, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle GSInputPrims event
        virtual void Handle(const GSInputPrims& event)
        {
            Write(32, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle GSPrimsGen event
        virtual void Handle(const GSPrimsGen& event)
        {
            Write(33, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle GSVertsInput event
        virtual void Handle(const GSVertsInput& event)
        {
            Write(34, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle TessPrims event
        virtual void Handle(const TessPrims& event)
        {
            Write(35, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle RasterTiles event
        virtual void Handle(const RasterTiles& event)
        {
            Write(36, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle ClipperEvent event
        virtual void Handle(const ClipperEvent& event)
        {
            Write(37, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle CullEvent event
        virtual void Handle(const CullEvent& event)
        {
            Write(38, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle AlphaEvent event
        virtual void Handle(const AlphaEvent& event)
        {
            Write(39, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle VSInfo event
        virtual void Handle(const VSInfo& event)
        {
            Write(40, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle HSInfo event
        virtual void Handle(const HSInfo& event)
        {
            Write(41, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle DSInfo event
        virtual void Handle(const DSInfo& event)
        {
            Write(42, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle GSInfo event
        virtual void Handle(const GSInfo& event)
        {
            Write(43, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle PSInfo event
        virtual void Handle(const PSInfo& event)
        {
            Write(44, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle CSInfo event
        virtual void Handle(const CSInfo& event)
        {
            Write(45, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyDepthStencilInfoSingleSample event
        virtual void Handle(const EarlyDepthStencilInfoSingleSample& event)
        {
            Write(1, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyDepthStencilInfoSampleRate event
        virtual void Handle(const EarlyDepthStencilInfoSampleRate& event)
        {
            Write(2, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyDepthStencilInfoNullPS event
        virtual void Handle(const EarlyDepthStencilInfoNullPS& event)
        {
            Write(3, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateDepthStencilInfoSingleSample event
        virtual void Handle(const LateDepthStencilInfoSingleSample& event)
        {
            Write(4, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateDepthStencilInfoSampleRate event
        virtual void Handle(const LateDepthStencilInfoSampleRate& event)
        {
            Write(5, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateDepthStencilInfoNullPS event
        virtual void Handle(const LateDepthStencilInfoNullPS& event)
        {
            Write(6, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle EarlyDepthInfoPixelRate event
        virtual void Handle(const EarlyDepthInfoPixelRate& event)
        {
            Write(7, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle LateDepthInfoPixelRate event
        virtual void Handle(const LateDepthInfoPixelRate& event)
        {
            Write(8, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle BackendDrawEndEvent event
        virtual void Handle(const BackendDrawEndEvent& event)
        {
            Write(9, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle FrontendDrawEndEvent event
        virtual void Handle(const FrontendDrawEndEvent& event)
        {
            Write(10, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle TessPrimCount event
        virtual void Handle(const TessPrimCount& event)
        {
            Write(11, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle RasterTileCount event
        virtual void Handle(const RasterTileCount& event)
        {
            Write(12, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle GSPrimInfo event
        virtual void Handle(const GSPrimInfo& event)
        {
            Write(13, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle ClipInfoEvent event
        virtual void Handle(const ClipInfoEvent& event)
        {
            Write(14, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle CullInfoEvent event
        virtual void Handle(const CullInfoEvent& event)
        {
            Write(15, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle AlphaInfoEvent event
        virtual void Handle(const AlphaInfoEvent& event)
        {
            Write(16, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle DrawInstancedEvent event
        virtual void Handle(const DrawInstancedEvent& event)
        {
            Write(17, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle DrawIndexedInstancedEvent event
        virtual void Handle(const DrawIndexedInstancedEvent& event)
        {
            Write(18, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle VSStats event
        virtual void Handle(const VSStats& event)
        {
            Write(19, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle HSStats event
        virtual void Handle(const HSStats& event)
        {
            Write(20, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle DSStats event
        virtual void Handle(const DSStats& event)
        {
            Write(21, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle GSStats event
        virtual void Handle(const GSStats& event)
        {
            Write(22, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle PSStats event
        virtual void Handle(const PSStats& event)
        {
            Write(23, (char*)&event.data, sizeof(event.data));
        }
        //////////////////////////////////////////////////////////////////////////
        /// @brief Handle CSStats event
        virtual void Handle(const CSStats& event)
        {
            Write(24, (char*)&event.data, sizeof(event.data));
        }

        //////////////////////////////////////////////////////////////////////////
        /// @brief Everything written to buffer this point is the header.
        virtual void MarkHeader()
        {
            mHeaderBufOffset = mBufOffset;
        }

        std::string mFilename;
        std::string mOutputDir;

        static const uint32_t mBufferSize = 1024;
        uint8_t               mBuffer[mBufferSize];
        uint32_t mBufOffset{0};
        uint32_t mHeaderBufOffset{0};
    };
} // namespace ArchRast
// clang-format on
