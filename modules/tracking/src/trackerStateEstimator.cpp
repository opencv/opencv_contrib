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
namespace detail {
inline namespace tracking {

Ptr<TrackerStateEstimator> TrackerStateEstimator::create( const String& trackeStateEstimatorType )
{

  if( trackeStateEstimatorType.find( "SVM" ) == 0 )
  {
    return Ptr<TrackerStateEstimatorSVM>( new TrackerStateEstimatorSVM() );
  }

  if( trackeStateEstimatorType.find( "BOOSTING" ) == 0 )
  {
    CV_Error(Error::StsNotImplemented, "TrackerStateEstimatorMILBoosting API is not available");
    //return Ptr<TrackerStateEstimatorMILBoosting>( new TrackerStateEstimatorMILBoosting() );
  }

  CV_Error( -1, "Tracker state estimator type not supported" );
}



/**
 * TrackerStateEstimatorAdaBoosting
 */
TrackerStateEstimatorAdaBoosting::TrackerStateEstimatorAdaBoosting( int numClassifer, int initIterations, int nFeatures, Size patchSize, const Rect& ROI )
{
  className = "ADABOOSTING";
  numBaseClassifier = numClassifer;
  numFeatures = nFeatures;
  iterationInit = initIterations;
  initPatchSize = patchSize;
  trained = false;
  sampleROI = ROI;

}

Rect TrackerStateEstimatorAdaBoosting::getSampleROI() const
{
  return sampleROI;
}

void TrackerStateEstimatorAdaBoosting::setSampleROI( const Rect& ROI )
{
  sampleROI = ROI;
}

/**
 * TrackerAdaBoostingTargetState::TrackerAdaBoostingTargetState
 */
TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::TrackerAdaBoostingTargetState( const Point2f& position, int width, int height,
                                                                                                bool foreground, const Mat& responses )
{
  setTargetPosition( position );
  setTargetWidth( width );
  setTargetHeight( height );

  setTargetFg( foreground );
  setTargetResponses( responses );
}

void TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::setTargetFg( bool foreground )
{
  isTarget = foreground;
}

bool TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::isTargetFg() const
{
  return isTarget;
}

void TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::setTargetResponses( const Mat& responses )
{
  targetResponses = responses;
}

Mat TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::getTargetResponses() const
{
  return targetResponses;
}

TrackerStateEstimatorAdaBoosting::~TrackerStateEstimatorAdaBoosting()
{

}
void TrackerStateEstimatorAdaBoosting::setCurrentConfidenceMap( ConfidenceMap& confidenceMap )
{
  currentConfidenceMap.clear();
  currentConfidenceMap = confidenceMap;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeReplacedClassifier()
{
  return replacedClassifier;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeSwappedClassifier()
{
  return swappedClassifier;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeSelectedWeakClassifier()
{
  return boostClassifier->getSelectedWeakClassifier();
}

Ptr<TrackerTargetState> TrackerStateEstimatorAdaBoosting::estimateImpl( const std::vector<ConfidenceMap>& /*confidenceMaps*/ )
{
  //run classify in order to compute next location
  if( currentConfidenceMap.empty() )
    return Ptr<TrackerTargetState>();

  std::vector<Mat> images;

  for ( size_t i = 0; i < currentConfidenceMap.size(); i++ )
  {
    Ptr<TrackerAdaBoostingTargetState> currentTargetState = currentConfidenceMap.at( i ).first.staticCast<TrackerAdaBoostingTargetState>();
    images.push_back( currentTargetState->getTargetResponses() );
  }

  int bestIndex;
  boostClassifier->classifySmooth( images, sampleROI, bestIndex );

  // get bestIndex from classifySmooth
  return currentConfidenceMap.at( bestIndex ).first;

}

void TrackerStateEstimatorAdaBoosting::updateImpl( std::vector<ConfidenceMap>& confidenceMaps )
{
  if( !trained )
  {
    //this is the first time that the classifier is built
    int numWeakClassifier = numBaseClassifier * 10;

    bool useFeatureExchange = true;
    boostClassifier = Ptr<StrongClassifierDirectSelection>(
        new StrongClassifierDirectSelection( numBaseClassifier, numWeakClassifier, initPatchSize, sampleROI, useFeatureExchange, iterationInit ) );
    //init base classifiers
    boostClassifier->initBaseClassifier();

    trained = true;
  }

  ConfidenceMap lastConfidenceMap = confidenceMaps.back();
  bool featureEx = boostClassifier->getUseFeatureExchange();

  replacedClassifier.clear();
  replacedClassifier.resize( lastConfidenceMap.size(), -1 );
  swappedClassifier.clear();
  swappedClassifier.resize( lastConfidenceMap.size(), -1 );

  for ( size_t i = 0; i < lastConfidenceMap.size() / 2; i++ )
  {
    Ptr<TrackerAdaBoostingTargetState> currentTargetState = lastConfidenceMap.at( i ).first.staticCast<TrackerAdaBoostingTargetState>();

    int currentFg = 1;
    if( !currentTargetState->isTargetFg() )
      currentFg = -1;
    Mat res = currentTargetState->getTargetResponses();

    boostClassifier->update( res, currentFg );
    if( featureEx )
    {
      replacedClassifier[i] = boostClassifier->getReplacedClassifier();
      swappedClassifier[i] = boostClassifier->getSwappedClassifier();
      if( replacedClassifier[i] >= 0 && swappedClassifier[i] >= 0 )
        boostClassifier->replaceWeakClassifier( replacedClassifier[i] );
    }
    else
    {
      replacedClassifier[i] = -1;
      swappedClassifier[i] = -1;
    }

    int mapPosition = (int)(i + lastConfidenceMap.size() / 2);
    Ptr<TrackerAdaBoostingTargetState> currentTargetState2 = lastConfidenceMap.at( mapPosition ).first.staticCast<TrackerAdaBoostingTargetState>();

    currentFg = 1;
    if( !currentTargetState2->isTargetFg() )
      currentFg = -1;
    const Mat res2 = currentTargetState2->getTargetResponses();

    boostClassifier->update( res2, currentFg );
    if( featureEx )
    {
      replacedClassifier[mapPosition] = boostClassifier->getReplacedClassifier();
      swappedClassifier[mapPosition] = boostClassifier->getSwappedClassifier();
      if( replacedClassifier[mapPosition] >= 0 && swappedClassifier[mapPosition] >= 0 )
        boostClassifier->replaceWeakClassifier( replacedClassifier[mapPosition] );
    }
    else
    {
      replacedClassifier[mapPosition] = -1;
      swappedClassifier[mapPosition] = -1;
    }
  }

}

/**
 * TrackerStateEstimatorSVM
 */
TrackerStateEstimatorSVM::TrackerStateEstimatorSVM()
{
  className = "SVM";
}

TrackerStateEstimatorSVM::~TrackerStateEstimatorSVM()
{

}

Ptr<TrackerTargetState> TrackerStateEstimatorSVM::estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps )
{
  return confidenceMaps.back().back().first;
}

void TrackerStateEstimatorSVM::updateImpl( std::vector<ConfidenceMap>& /*confidenceMaps*/)
{

}

}}}  // namespace
