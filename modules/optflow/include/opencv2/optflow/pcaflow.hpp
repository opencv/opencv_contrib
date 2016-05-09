/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef __OPENCV_OPTFLOW_PCAFLOW_HPP__
#define __OPENCV_OPTFLOW_PCAFLOW_HPP__

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"

namespace cv
{
namespace optflow
{

class OpticalFlowPCAFlow : public DenseOpticalFlow
{
protected:
  const Size basisSize;
  const float sparseRate;              // (0 .. 0.1)
  const float retainedCornersFraction; // [0 .. 1]
  const float occlusionsThreshold;

public:
  OpticalFlowPCAFlow( Size _basisSize = Size( 18, 14 ), float _sparseRate = 0.02, float _retainedCornersFraction = 1.0,
                      float _occlusionsThreshold = 0.00002 );

  void calc( InputArray I0, InputArray I1, InputOutputArray flow );
  void collectGarbage();

private:
  void findSparseFeatures( Mat &from, Mat &to, std::vector<Point2f> &features,
                           std::vector<Point2f> &predictedFeatures ) const;

  void removeOcclusions( Mat &from, Mat &to, std::vector<Point2f> &features,
                         std::vector<Point2f> &predictedFeatures ) const;

  void getSystem( OutputArray AOut, OutputArray b1Out, OutputArray b2Out, const std::vector<Point2f> &features,
                  const std::vector<Point2f> &predictedFeatures, const Size size );
};

CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_PCAFlow();
}
}

#endif
