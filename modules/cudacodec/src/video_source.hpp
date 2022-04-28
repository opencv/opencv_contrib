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

#ifndef __CUDACODEC_VIDEO_SOURCE_H__
#define __CUDACODEC_VIDEO_SOURCE_H__

#include "thread.hpp"

namespace cv { namespace cudacodec { namespace detail {

class VideoParser;

class VideoSource
{
public:
    virtual ~VideoSource() {}

    virtual FormatInfo format() const = 0;
    virtual void updateFormat(const FormatInfo& videoFormat) = 0;
    virtual bool get(const int propertyId, double& propertyVal) const { return false; }
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual bool isStarted() const = 0;
    virtual bool hasError() const = 0;
    void setVideoParser(detail::VideoParser* videoParser) { videoParser_ = videoParser; }
    void setExtraData(const cv::Mat _extraData) {
        AutoLock autoLock(mtx_);
        extraData = _extraData.clone();
    }
    void getExtraData(cv::Mat& _extraData) {
        AutoLock autoLock(mtx_);
        _extraData = extraData.clone();
    }
    void SetRawMode(const bool enabled) { rawMode_ = enabled; }
    bool RawModeEnabled() const { return rawMode_; }
protected:
    bool parseVideoData(const uchar* data, size_t size, const bool rawMode, const bool containsKeyFrame,  bool endOfStream = false);
    bool extraDataQueried = false;
private:
    detail::VideoParser* videoParser_ = 0;
    cv::Mat extraData;
    bool rawMode_ = false;
    Mutex mtx_;
};

class RawVideoSourceWrapper : public VideoSource
{
public:
    RawVideoSourceWrapper(const Ptr<RawVideoSource>& source, const bool rawMode);

    FormatInfo format() const CV_OVERRIDE;
    void updateFormat(const FormatInfo& videoFormat) CV_OVERRIDE;
    bool get(const int propertyId, double& propertyVal) const CV_OVERRIDE;
    void start() CV_OVERRIDE;
    void stop() CV_OVERRIDE;
    bool isStarted() const CV_OVERRIDE;
    bool hasError() const CV_OVERRIDE;
private:
    static void readLoop(void* userData);
    Ptr<RawVideoSource> source_ = 0;
    Ptr<Thread> thread_ = 0;
    volatile bool stop_;
    volatile bool hasError_;
};

}}}

#endif // __CUDACODEC_VIDEO_SOURCE_H__
