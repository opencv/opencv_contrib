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

#include "opencv2/tracking/tracking_legacy.hpp"

namespace cv {
namespace legacy {
inline namespace tracking {
namespace impl {

class TrackerMILImpl CV_FINAL : public legacy::TrackerMIL
{
public:
    cv::tracking::impl::TrackerMILImpl impl;

    TrackerMILImpl(const legacy::TrackerMIL::Params &parameters)
        : impl(parameters)
    {
        isInit = false;
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        static_cast<legacy::TrackerMIL::Params&>(impl.params).read(fn);
    }
    void write(FileStorage& fs) const CV_OVERRIDE
    {
        static_cast<const legacy::TrackerMIL::Params&>(impl.params).write(fs);
    }

    bool initImpl(const Mat& image, const Rect2d& boundingBox) CV_OVERRIDE
    {
        impl.init(image, boundingBox);
        model = impl.model;
        featureSet = impl.featureSet;
        sampler = impl.sampler;
        isInit = true;
        return true;
    }
    bool updateImpl(const Mat& image, Rect2d& boundingBox) CV_OVERRIDE
    {
        Rect bb;
        bool res = impl.update(image, bb);
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
