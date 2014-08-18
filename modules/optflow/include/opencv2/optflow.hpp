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

#ifndef __OPENCV_OPTFLOW_HPP__
#define __OPENCV_OPTFLOW_HPP__

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"

namespace cv
{
namespace optflow
{
    
//! computes dense optical flow using Simple Flow algorithm
CV_EXPORTS_W void calcOpticalFlowSF( InputArray from, InputArray to, OutputArray flow,
                                     int layers, int averaging_block_size, int max_flow);

CV_EXPORTS_W void calcOpticalFlowSF( InputArray from, InputArray to, OutputArray flow, int layers,
                                     int averaging_block_size, int max_flow,
                                     double sigma_dist, double sigma_color, int postprocess_window,
                                     double sigma_dist_fix, double sigma_color_fix, double occ_thr,
                                     int upscale_averaging_radius, double upscale_sigma_dist,
                                     double upscale_sigma_color, double speed_up_thr );    

//! reads optical flow from a file, Middlebury format:
// http://vision.middlebury.edu/flow/code/flow-code/README.txt
CV_EXPORTS_W Mat readOpticalFlow( const String& path );
//! writes optical flow to a file, Middlebury format
CV_EXPORTS_W bool writeOpticalFlow( const String& path, InputArray flow );




// DeepFlow implementation, based on:
//   P. Weinzaepfel, J. Revaud, Z. Harchaoui, and C. Schmid, “DeepFlow: Large Displacement Optical Flow with Deep Matching,”
CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_DeepFlow();

// Additional interface to the SimpleFlow algorithm - calcOpticalFlowSF()
CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_SimpleFlow();

// Additional interface to the Farneback's algorithm - calcOpticalFlowFarneback()
CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_Farneback();
} //optflow
}

#include "opencv2/optflow/motempl.hpp"

#endif
