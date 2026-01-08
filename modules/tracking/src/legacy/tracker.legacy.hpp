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

Tracker::Tracker()
{
    isInit = false;
}

Tracker::~Tracker()
{
}

bool Tracker::init( InputArray image, const Rect2d& boundingBox )
{

  if( isInit )
  {
    return false;
  }

  if( image.empty() )
    return false;

  sampler = Ptr<TrackerContribSampler>( new TrackerContribSampler() );
  featureSet = Ptr<TrackerContribFeatureSet>( new TrackerContribFeatureSet() );
  model = Ptr<TrackerModel>();

  bool initTracker = initImpl( image.getMat(), boundingBox );

  if (initTracker)
  {
    isInit = true;
  }

  return initTracker;
}

bool Tracker::update( InputArray image, Rect2d& boundingBox )
{

  if( !isInit )
  {
    return false;
  }

  if( image.empty() )
    return false;

  return updateImpl( image.getMat(), boundingBox );
}



class LegacyTrackerWrapper : public cv::Tracker
{
    const Ptr<legacy::Tracker> legacy_tracker_;
public:
    LegacyTrackerWrapper(const Ptr<legacy::Tracker>& legacy_tracker) : legacy_tracker_(legacy_tracker)
    {
        CV_Assert(legacy_tracker_);
    }
    virtual ~LegacyTrackerWrapper() CV_OVERRIDE {};

    void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE
    {
        CV_DbgAssert(legacy_tracker_);
        legacy_tracker_->init(image, (Rect2d)boundingBox);
    }

    bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE
    {
        CV_DbgAssert(legacy_tracker_);
        Rect2d boundingBox2d;
        bool res = legacy_tracker_->update(image, boundingBox2d);
        int x1 = cvRound(boundingBox2d.x);
        int y1 = cvRound(boundingBox2d.y);
        int x2 = cvRound(boundingBox2d.x + boundingBox2d.width);
        int y2 = cvRound(boundingBox2d.y + boundingBox2d.height);
        boundingBox = Rect(x1, y1, x2 - x1, y2 - y1) & Rect(Point(0, 0), image.size());
        return res;
    }
};


CV_EXPORTS_W Ptr<cv::Tracker> upgradeTrackingAPI(const Ptr<legacy::Tracker>& legacy_tracker)
{
    return makePtr<LegacyTrackerWrapper>(legacy_tracker);
}

}}}  // namespace
