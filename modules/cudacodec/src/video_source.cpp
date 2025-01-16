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

#include "precomp.hpp"

#ifdef HAVE_NVCUVID

using namespace cv;
using namespace cv::cudacodec;
using namespace cv::cudacodec::detail;

bool cv::cudacodec::detail::VideoSource::parseVideoData(const unsigned char* data, size_t size, const bool rawMode, const bool containsKeyFrame, bool endOfStream)
{
    return videoParser_->parseVideoData(data, size, rawMode, containsKeyFrame, endOfStream);
}

cv::cudacodec::detail::RawVideoSourceWrapper::RawVideoSourceWrapper(const Ptr<RawVideoSource>& source, const bool rawMode) :
    source_(source)
{
    SetRawMode(rawMode);
    CV_Assert( !source_.empty() );
}

cv::cudacodec::FormatInfo cv::cudacodec::detail::RawVideoSourceWrapper::format() const
{
    return source_->format();
}

void cv::cudacodec::detail::RawVideoSourceWrapper::updateFormat(const FormatInfo& videoFormat)
{
    source_->updateFormat(videoFormat);
}

bool cv::cudacodec::detail::RawVideoSourceWrapper::get(const int propertyId, double& propertyVal) const
{
    return source_->get(propertyId, propertyVal);
}

int cv::cudacodec::detail::RawVideoSourceWrapper::getFirstFrameIdx() const {
    return source_->getFirstFrameIdx();
}

void cv::cudacodec::detail::RawVideoSourceWrapper::start()
{
    stop_ = false;
    hasError_ = false;
    thread_.reset(new Thread(readLoop, this));
}

void cv::cudacodec::detail::RawVideoSourceWrapper::stop()
{
    stop_ = true;
    thread_->wait();
    thread_.release();
}

bool cv::cudacodec::detail::RawVideoSourceWrapper::isStarted() const
{
    return !stop_;
}

bool cv::cudacodec::detail::RawVideoSourceWrapper::hasError() const
{
    return hasError_;
}

void cv::cudacodec::detail::RawVideoSourceWrapper::readLoop(void* userData)
{
    RawVideoSourceWrapper* thiz = static_cast<RawVideoSourceWrapper*>(userData);

    for (;;)
    {
        unsigned char* data;
        size_t size;

        if (!thiz->source_->getNextPacket(&data, &size))
        {
            thiz->hasError_ = false;
            break;
        }

        bool containsKeyFrame = false;
        if (thiz->RawModeEnabled()) {
            containsKeyFrame = thiz->source_->lastPacketContainsKeyFrame();
            if (!thiz->extraDataQueried) {
                thiz->extraDataQueried = true;
                Mat extraData;
                thiz->source_->getExtraData(extraData);
                if(!extraData.empty())
                    thiz->setExtraData(extraData);
            }
        }

        if (!thiz->parseVideoData(data, size, thiz->RawModeEnabled(), containsKeyFrame))
        {
            thiz->hasError_ = true;
            break;
        }

        if (thiz->stop_)
            break;
    }

    if(!thiz->hasError_)
        thiz->parseVideoData(0, 0, false, false, true);
}

#endif // HAVE_NVCUVID
