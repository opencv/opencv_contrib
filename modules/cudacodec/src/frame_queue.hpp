/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __FRAME_QUEUE_HPP__
#define __FRAME_QUEUE_HPP__
#include <queue>

#include "opencv2/core/utility.hpp"

class RawPacket {
public:
    RawPacket(const unsigned char* _data, const size_t _size = 0, const bool _containsKeyFrame = false);
    const unsigned char* Data() const noexcept { return data.data(); }
    size_t Size() const noexcept  { return data.size(); }
    bool ContainsKeyFrame() const noexcept  { return containsKeyFrame; }
private:
    std::vector<unsigned char> data;
    bool containsKeyFrame = false;
};

namespace cv { namespace cudacodec { namespace detail {

class FrameQueue
{
public:
    ~FrameQueue();
    void init(const int _maxSz);

    void endDecode() { endOfDecode_ = true; }
    bool isEndOfDecode() const { return endOfDecode_ != 0;}

    // Spins until frame becomes available or decoding gets canceled.
    // If the requested frame is available the method returns true.
    // If decoding was interrupted before the requested frame becomes
    // available, the method returns false.
    // If allowFrameDrop == true, spin is disabled and n > 0 frames are discarded
    // to ensure a frame is available.
    bool waitUntilFrameAvailable(int pictureIndex, const bool allowFrameDrop = false);

    void enqueue(const CUVIDPARSERDISPINFO* picParams, const std::vector<RawPacket> rawPackets);

    // Deque the next frame.
    // Parameters:
    //      displayInfo - New frame info gets placed into this object.
    // Returns:
    //      true, if a new frame was returned,
    //      false, if the queue was empty and no new frame could be returned.
    bool dequeue(CUVIDPARSERDISPINFO& displayInfo, std::vector<RawPacket>& rawPackets);

    // Deque all frames up to and including the frame with index pictureIndex - must only
    // be called in the same thread as enqueue.
    // Parameters:
    //      pictureIndex - Display index of the frame.
    // Returns:
    //      true, if successful,
    //      false, if no frames are dequed.
    bool dequeueUntil(const int pictureIndex);

    void releaseFrame(const CUVIDPARSERDISPINFO& picParams) { isFrameInUse_[picParams.picture_index] = 0; }
private:
    bool isInUse(int pictureIndex) const { return isFrameInUse_[pictureIndex] != 0; }

    Mutex mtx_;
    volatile int* isFrameInUse_ = 0;
    volatile int endOfDecode_ = 0;
    int framesInQueue_ = 0;
    int readPosition_ = 0;
    std::vector< CUVIDPARSERDISPINFO> displayQueue_;
    int maxSz = 0;
    std::queue<RawPacket> rawPacketQueue;
};

}}}

#endif // __FRAME_QUEUE_HPP__
