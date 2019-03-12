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

#include "trackerBoostingModel.hpp"

/**
 * TrackerBoostingModel
 */

namespace cv
{

TrackerBoostingModel::TrackerBoostingModel( const Rect& boundingBox )
{

  mode = MODE_POSITIVE;

  Ptr<TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState> initState =
      Ptr<TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState>(
          new TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState( Point2f( (float)boundingBox.x, (float)boundingBox.y ), boundingBox.width,
                                                                               boundingBox.height, true, Mat() ) );
  trajectory.push_back( initState );
  maxCMLength = 10;
}

void TrackerBoostingModel::modelEstimationImpl( const std::vector<Mat>& responses )
{
  responseToConfidenceMap( responses, currentConfidenceMap );
}

void TrackerBoostingModel::modelUpdateImpl()
{

}

void TrackerBoostingModel::setMode( int trainingMode, const std::vector<Mat>& samples )
{
  currentSample.clear();
  currentSample = samples;

  mode = trainingMode;
}

std::vector<int> TrackerBoostingModel::getSelectedWeakClassifier()
{
  return stateEstimator.staticCast<TrackerStateEstimatorAdaBoosting>()->computeSelectedWeakClassifier();
}

void TrackerBoostingModel::responseToConfidenceMap( const std::vector<Mat>& responses, ConfidenceMap& confidenceMap )
{
  if( currentSample.empty() )
  {
    CV_Error( -1, "The samples in Model estimation are empty" );
  }

  for ( size_t i = 0; i < currentSample.size(); i++ )
  {

    Size currentSize;
    Point currentOfs;
    currentSample.at( i ).locateROI( currentSize, currentOfs );
    bool foreground = false;
    if( mode == MODE_POSITIVE || mode == MODE_CLASSIFY )
    {
      foreground = true;
    }
    else if( mode == MODE_NEGATIVE )
    {
      foreground = false;
    }
    const Mat resp = responses[0].col( (int)i );

    //create the state
    Ptr<TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState> currentState = Ptr<
        TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState>(
        new TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState( currentOfs, currentSample.at( i ).cols, currentSample.at( i ).rows,
                                                                             foreground, resp ) );

    confidenceMap.push_back( std::make_pair( currentState, 0.0f ) );

  }
}

}
