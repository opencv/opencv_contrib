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
#include "trackerMILModel.hpp"

namespace cv
{

class TrackerMILImpl : public TrackerMIL
{
 public:
  TrackerMILImpl( const TrackerMIL::Params &parameters = TrackerMIL::Params() );
  void read( const FileNode& fn );
  void write( FileStorage& fs ) const;

 protected:

  bool initImpl( const Mat& image, const Rect2d& boundingBox );
  bool updateImpl( const Mat& image, Rect2d& boundingBox );
  void compute_integral( const Mat & img, Mat & ii_img );

  TrackerMIL::Params params;
};

/*
 *  TrackerMIL
 */

/*
 * Parameters
 */
TrackerMIL::Params::Params()
{
  samplerInitInRadius = 3;
  samplerSearchWinSize = 25;
  samplerInitMaxNegNum = 65;
  samplerTrackInRadius = 4;
  samplerTrackMaxPosNum = 100000;
  samplerTrackMaxNegNum = 65;
  featureSetNumFeatures = 250;
}

void TrackerMIL::Params::read( const cv::FileNode& fn )
{
  samplerInitInRadius = fn["samplerInitInRadius"];
  samplerSearchWinSize = fn["samplerSearchWinSize"];
  samplerInitMaxNegNum = fn["samplerInitMaxNegNum"];
  samplerTrackInRadius = fn["samplerTrackInRadius"];
  samplerTrackMaxPosNum = fn["samplerTrackMaxPosNum"];
  samplerTrackMaxNegNum = fn["samplerTrackMaxNegNum"];
  featureSetNumFeatures = fn["featureSetNumFeatures"];
}

void TrackerMIL::Params::write( cv::FileStorage& fs ) const
{
  fs << "samplerInitInRadius" << samplerInitInRadius;
  fs << "samplerSearchWinSize" << samplerSearchWinSize;
  fs << "samplerInitMaxNegNum" << samplerInitMaxNegNum;
  fs << "samplerTrackInRadius" << samplerTrackInRadius;
  fs << "samplerTrackMaxPosNum" << samplerTrackMaxPosNum;
  fs << "samplerTrackMaxNegNum" << samplerTrackMaxNegNum;
  fs << "featureSetNumFeatures" << featureSetNumFeatures;

}

/*
 * Constructor
 */
Ptr<TrackerMIL> TrackerMIL::createTracker(const TrackerMIL::Params &parameters){
    return Ptr<TrackerMILImpl>(new TrackerMILImpl(parameters));
}
TrackerMILImpl::TrackerMILImpl( const TrackerMIL::Params &parameters ) :
    params( parameters )
{
  isInit = false;
}

void TrackerMILImpl::read( const cv::FileNode& fn )
{
  params.read( fn );
}

void TrackerMILImpl::write( cv::FileStorage& fs ) const
{
  params.write( fs );
}

void TrackerMILImpl::compute_integral( const Mat & img, Mat & ii_img )
{
  Mat ii;
  std::vector<Mat> ii_imgs;
  integral( img, ii, CV_32F );
  split( ii, ii_imgs );
  ii_img = ii_imgs[0];
}

bool TrackerMILImpl::initImpl( const Mat& image, const Rect2d& boundingBox )
{
  srand (1);
  Mat intImage;
  compute_integral( image, intImage );
  TrackerSamplerCSC::Params CSCparameters;
  CSCparameters.initInRad = params.samplerInitInRadius;
  CSCparameters.searchWinSize = params.samplerSearchWinSize;
  CSCparameters.initMaxNegNum = params.samplerInitMaxNegNum;
  CSCparameters.trackInPosRad = params.samplerTrackInRadius;
  CSCparameters.trackMaxPosNum = params.samplerTrackMaxPosNum;
  CSCparameters.trackMaxNegNum = params.samplerTrackMaxNegNum;

  Ptr<TrackerSamplerAlgorithm> CSCSampler = Ptr<TrackerSamplerCSC>( new TrackerSamplerCSC( CSCparameters ) );
  if( !sampler->addTrackerSamplerAlgorithm( CSCSampler ) )
    return false;

  //or add CSC sampler with default parameters
  //sampler->addTrackerSamplerAlgorithm( "CSC" );

  //Positive sampling
  CSCSampler.staticCast<TrackerSamplerCSC>()->setMode( TrackerSamplerCSC::MODE_INIT_POS );
  sampler->sampling( intImage, boundingBox );
  std::vector<Mat> posSamples = sampler->getSamples();

  //Negative sampling
  CSCSampler.staticCast<TrackerSamplerCSC>()->setMode( TrackerSamplerCSC::MODE_INIT_NEG );
  sampler->sampling( intImage, boundingBox );
  std::vector<Mat> negSamples = sampler->getSamples();

  if( posSamples.empty() || negSamples.empty() )
    return false;

  //compute HAAR features
  TrackerFeatureHAAR::Params HAARparameters;
  HAARparameters.numFeatures = params.featureSetNumFeatures;
  HAARparameters.rectSize = Size( (int)boundingBox.width, (int)boundingBox.height );
  HAARparameters.isIntegral = true;
  Ptr<TrackerFeature> trackerFeature = Ptr<TrackerFeatureHAAR>( new TrackerFeatureHAAR( HAARparameters ) );
  featureSet->addTrackerFeature( trackerFeature );

  featureSet->extraction( posSamples );
  const std::vector<Mat> posResponse = featureSet->getResponses();

  featureSet->extraction( negSamples );
  const std::vector<Mat> negResponse = featureSet->getResponses();

  model = Ptr<TrackerMILModel>( new TrackerMILModel( boundingBox ) );
  Ptr<TrackerStateEstimatorMILBoosting> stateEstimator = Ptr<TrackerStateEstimatorMILBoosting>(
      new TrackerStateEstimatorMILBoosting( params.featureSetNumFeatures ) );
  model->setTrackerStateEstimator( stateEstimator );

  //Run model estimation and update
  model.staticCast<TrackerMILModel>()->setMode( TrackerMILModel::MODE_POSITIVE, posSamples );
  model->modelEstimation( posResponse );
  model.staticCast<TrackerMILModel>()->setMode( TrackerMILModel::MODE_NEGATIVE, negSamples );
  model->modelEstimation( negResponse );
  model->modelUpdate();

  return true;
}

bool TrackerMILImpl::updateImpl( const Mat& image, Rect2d& boundingBox )
{
  Mat intImage;
  compute_integral( image, intImage );

  //get the last location [AAM] X(k-1)
  Ptr<TrackerTargetState> lastLocation = model->getLastTargetState();
  Rect lastBoundingBox( (int)lastLocation->getTargetPosition().x, (int)lastLocation->getTargetPosition().y, lastLocation->getTargetWidth(),
                        lastLocation->getTargetHeight() );

  //sampling new frame based on last location
  ( sampler->getSamplers().at( 0 ).second ).staticCast<TrackerSamplerCSC>()->setMode( TrackerSamplerCSC::MODE_DETECT );
  sampler->sampling( intImage, lastBoundingBox );
  std::vector<Mat> detectSamples = sampler->getSamples();
  if( detectSamples.empty() )
    return false;

  /*//TODO debug samples
   Mat f;
   image.copyTo(f);

   for( size_t i = 0; i < detectSamples.size(); i=i+10 )
   {
   Size sz;
   Point off;
   detectSamples.at(i).locateROI(sz, off);
   rectangle(f, Rect(off.x,off.y,detectSamples.at(i).cols,detectSamples.at(i).rows), Scalar(255,0,0), 1);
   }*/

  //extract features from new samples
  featureSet->extraction( detectSamples );
  std::vector<Mat> response = featureSet->getResponses();

  //predict new location
  ConfidenceMap cmap;
  model.staticCast<TrackerMILModel>()->setMode( TrackerMILModel::MODE_ESTIMATON, detectSamples );
  model.staticCast<TrackerMILModel>()->responseToConfidenceMap( response, cmap );
  model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorMILBoosting>()->setCurrentConfidenceMap( cmap );

  if( !model->runStateEstimator() )
  {
    return false;
  }

  Ptr<TrackerTargetState> currentState = model->getLastTargetState();
  boundingBox = Rect( (int)currentState->getTargetPosition().x, (int)currentState->getTargetPosition().y, currentState->getTargetWidth(),
                      currentState->getTargetHeight() );

  /*//TODO debug
   rectangle(f, lastBoundingBox, Scalar(0,255,0), 1);
   rectangle(f, boundingBox, Scalar(0,0,255), 1);
   imshow("f", f);
   //waitKey( 0 );*/

  //sampling new frame based on new location
  //Positive sampling
  ( sampler->getSamplers().at( 0 ).second ).staticCast<TrackerSamplerCSC>()->setMode( TrackerSamplerCSC::MODE_INIT_POS );
  sampler->sampling( intImage, boundingBox );
  std::vector<Mat> posSamples = sampler->getSamples();

  //Negative sampling
  ( sampler->getSamplers().at( 0 ).second ).staticCast<TrackerSamplerCSC>()->setMode( TrackerSamplerCSC::MODE_INIT_NEG );
  sampler->sampling( intImage, boundingBox );
  std::vector<Mat> negSamples = sampler->getSamples();

  if( posSamples.empty() || negSamples.empty() )
    return false;

  //extract features
  featureSet->extraction( posSamples );
  std::vector<Mat> posResponse = featureSet->getResponses();

  featureSet->extraction( negSamples );
  std::vector<Mat> negResponse = featureSet->getResponses();

  //model estimate
  model.staticCast<TrackerMILModel>()->setMode( TrackerMILModel::MODE_POSITIVE, posSamples );
  model->modelEstimation( posResponse );
  model.staticCast<TrackerMILModel>()->setMode( TrackerMILModel::MODE_NEGATIVE, negSamples );
  model->modelEstimation( negResponse );

  //model update
  model->modelUpdate();

  return true;
}

} /* namespace cv */
