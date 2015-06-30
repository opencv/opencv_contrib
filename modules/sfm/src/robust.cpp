/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "opencv2/sfm/eigen.hpp"
#include "opencv2/sfm/robust.hpp"
#include "opencv2/sfm/numeric.hpp"

#include "libmv/multiview/robust_fundamental.h"
#include <opencv2/core/eigen.hpp>

using namespace std;

namespace cv
{

void
fundamentalFromCorrespondences8PointRobust( const Mat_<double> &_x1,
                                            const Mat_<double> &_x2,
                                            double max_error,
                                            Matx33d &_F,
                                            vector<int> &_inliers,
                                            double outliers_probability )
{
    libmv::Mat x1, x2;
    libmv::Mat3 F;
    libmv::vector<int> inliers;

    cv2eigen( _x1, x1 );
    cv2eigen( _x2, x2 );

    libmv::FundamentalFromCorrespondences8PointRobust( x1, x2, max_error, &F, &inliers, outliers_probability );

    eigen2cv( F, _F );

    // transform from libmv::vector to std::vector
    int n = inliers.size();
    _inliers.resize(n);
    for( int i=0; i < n; ++i )
    {
        _inliers[i] = inliers.at(i);
    }
}

} /* namespace cv */
