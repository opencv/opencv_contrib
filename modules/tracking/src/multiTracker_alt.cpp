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

namespace cv {

  // constructor
  MultiTracker::MultiTracker(){};

  // destructor
  MultiTracker::~MultiTracker(){};

  // add a new tracked object
  bool MultiTracker::add( Ptr<Tracker> newTracker, InputArray image, const Rect2d& boundingBox )
  {
    // add the tracker algorithm to the trackers list
    trackerList.push_back(newTracker);

    // add the ROI to the bounding box list
    objects.push_back(boundingBox);

    // initialize the created tracker
    return trackerList.back()->init(image, boundingBox);
  };

  // add a set of objects to be tracked
  bool MultiTracker::add(std::vector<Ptr<Tracker> > newTrackers, InputArray image, std::vector<Rect2d> boundingBox){
    // status of the tracker addition
    bool stat=false;

    // add tracker for all input objects
    for(unsigned i =0;i<boundingBox.size();i++){
      stat=add(newTrackers[i],image,boundingBox[i]);
      if(!stat)break;
    }

    // return the status
    return stat;
  };

  // update position of the tracked objects, the result is stored in internal storage
  bool MultiTracker::update(InputArray image)
  {
    bool status = true;
    for(unsigned i=0;i< trackerList.size(); i++){
      status &= trackerList[i]->update(image, objects[i]);
    }
    return status;
  };

  // update position of the tracked objects, the result is copied to external variable
  bool MultiTracker::update(InputArray image, std::vector<Rect2d> & boundingBox )
  {
    bool status = update(image);
    boundingBox=objects;
    return status;
  };

  const std::vector<Rect2d>& MultiTracker::getObjects() const
  {
      return objects;
  }

  Ptr<MultiTracker> MultiTracker::create()
  {
      return makePtr<MultiTracker>();
  }

} /* namespace cv */
