/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
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
#include "opencv2/tracking/tracking_legacy.hpp"

namespace cv {
namespace legacy {
inline namespace tracking {
namespace impl {

class TrackerMILImpl CV_FINAL : public legacy::TrackerMIL
{
public:
    Ptr<cv::TrackerMIL> impl;
    legacy::TrackerMIL::Params params;

    TrackerMILImpl(const legacy::TrackerMIL::Params &parameters)
        : impl(cv::TrackerMIL::create(parameters))
        , params(parameters)
    {
        isInit = false;
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        params.read(fn);
        CV_Error(Error::StsNotImplemented, "Can't update legacy tracker wrapper");
    }
    void write(FileStorage& fs) const CV_OVERRIDE
    {
        params.write(fs);
    }

    bool initImpl(const Mat& image, const Rect2d& boundingBox2d) CV_OVERRIDE
    {
        int x1 = cvRound(boundingBox2d.x);
        int y1 = cvRound(boundingBox2d.y);
        int x2 = cvRound(boundingBox2d.x + boundingBox2d.width);
        int y2 = cvRound(boundingBox2d.y + boundingBox2d.height);
        Rect boundingBox = Rect(x1, y1, x2 - x1, y2 - y1) & Rect(Point(0, 0), image.size());
        impl->init(image, boundingBox);
        isInit = true;
        return true;
    }
    bool updateImpl(const Mat& image, Rect2d& boundingBox) CV_OVERRIDE
    {
        Rect bb;
        bool res = impl->update(image, bb);
        boundingBox = bb;
        return res;
    }
};

}  // namespace

void legacy::TrackerMIL::Params::read(const cv::FileNode& fn)
{
  samplerInitInRadius = fn["samplerInitInRadius"];
  samplerSearchWinSize = fn["samplerSearchWinSize"];
  samplerInitMaxNegNum = fn["samplerInitMaxNegNum"];
  samplerTrackInRadius = fn["samplerTrackInRadius"];
  samplerTrackMaxPosNum = fn["samplerTrackMaxPosNum"];
  samplerTrackMaxNegNum = fn["samplerTrackMaxNegNum"];
  featureSetNumFeatures = fn["featureSetNumFeatures"];
}

void legacy::TrackerMIL::Params::write(cv::FileStorage& fs) const
{
  fs << "samplerInitInRadius" << samplerInitInRadius;
  fs << "samplerSearchWinSize" << samplerSearchWinSize;
  fs << "samplerInitMaxNegNum" << samplerInitMaxNegNum;
  fs << "samplerTrackInRadius" << samplerTrackInRadius;
  fs << "samplerTrackMaxPosNum" << samplerTrackMaxPosNum;
  fs << "samplerTrackMaxNegNum" << samplerTrackMaxNegNum;
  fs << "featureSetNumFeatures" << featureSetNumFeatures;
}

}}  // namespace

Ptr<legacy::TrackerMIL> legacy::TrackerMIL::create(const legacy::TrackerMIL::Params &parameters)
{
    return makePtr<legacy::tracking::impl::TrackerMILImpl>(parameters);
}
Ptr<legacy::TrackerMIL> legacy::TrackerMIL::create()
{
    return create(legacy::TrackerMIL::Params());
}

}  // namespace
