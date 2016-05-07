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
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

namespace cv
{
namespace optflow
{

class OpticalFlowPCAFlow : public DenseOpticalFlow
{
protected:
  float sparseRate;

public:
  OpticalFlowPCAFlow() : sparseRate( 0.02 ){};

  void calc( InputArray I0, InputArray I1, InputOutputArray flow );
  void collectGarbage();

private:
  void findSparseFeatures( Mat &from, Mat &to, std::vector<Point2f> &features,
                           std::vector<Point2f> &predictedFeatures );
};

void OpticalFlowPCAFlow::findSparseFeatures( Mat &from, Mat &to, std::vector<Point2f> &features,
                                             std::vector<Point2f> &predictedFeatures )
{
  std::vector<uchar> predictedStatus;
  std::vector<float> predictedError;
  calcOpticalFlowPyrLK( from, to, features, predictedFeatures, predictedStatus, predictedError );

  size_t j = 0;
  for ( size_t i = 0; i < features.size(); ++i )
  {
    if ( predictedStatus[i] )
    {
      features[j] = features[i];
      predictedFeatures[j] = predictedFeatures[i];
      ++j;
    }
  }
  features.resize( j );
  predictedFeatures.resize( j );
}

void OpticalFlowPCAFlow::calc( InputArray I0, InputArray I1, InputOutputArray flow_out )
{
  Size size = I0.size();
  CV_Assert( size == I1.size() );
  CV_Assert( sparseRate > 0 && sparseRate < 0.1 );

  Mat from, to;
  if ( I0.channels() == 3 )
  {
    cvtColor( I0, from, COLOR_BGR2GRAY );
    from.convertTo( from, CV_8U );
  }
  else
  {
    I0.getMat().convertTo( from, CV_8U );
  }
  if ( I1.channels() == 3 )
  {
    cvtColor( I1, to, COLOR_BGR2GRAY );
    to.convertTo( to, CV_8U );
  }
  else
  {
    I1.getMat().convertTo( to, CV_8U );
  }

  CV_Assert( from.channels() == 1 );
  CV_Assert( to.channels() == 1 );

  std::vector<Point2f> features, predictedFeatures;
  const unsigned maxFeatures = size.area() * sparseRate;
  goodFeaturesToTrack( from, features, maxFeatures, 0.005, 3 );

  // Add points along the grid if not enough features
  {
    const unsigned missingPoints = maxFeatures - features.size();
    const unsigned blockSize = sqrt( (float)size.area() / missingPoints );
    for ( int x = blockSize / 2; x < size.width; x += blockSize )
      for ( int y = blockSize / 2; y < size.height; y += blockSize )
        features.push_back( Point2f( x, y ) );
  }

  findSparseFeatures( from, to, features, predictedFeatures );

  // TODO: Remove occlusions

  flow_out.create( size, CV_32FC2 );
  Mat flow = flow_out.getMat();
  for ( size_t i = 0; i < features.size(); ++i )
  {
    flow.at<Point2f>( features[i].y, features[i].x ) = predictedFeatures[i] - features[i];
  }

  from.convertTo( from, CV_32F );
  to.convertTo( to, CV_32F );
}

void OpticalFlowPCAFlow::collectGarbage() {}

Ptr<DenseOpticalFlow> createOptFlow_PCAFlow() { return makePtr<OpticalFlowPCAFlow>(); }
}
}
