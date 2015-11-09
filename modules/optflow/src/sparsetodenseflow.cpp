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
#include "opencv2/ximgproc/sparse_match_interpolator.hpp"
using namespace std;

namespace cv {
namespace optflow {

CV_EXPORTS_W void calcOpticalFlowSparseToDense(InputArray from, InputArray to, OutputArray flow,
                                               int grid_step, int k,
                                               float sigma, bool use_post_proc,
                                               float fgs_lambda, float fgs_sigma)
{
    CV_Assert( grid_step>1 && k>3 && sigma>0.0001f && fgs_lambda>1.0f && fgs_sigma>0.01f );
    CV_Assert( !from.empty() && from.depth() == CV_8U && (from.channels() == 3 || from.channels() == 1) );
    CV_Assert( !to  .empty() && to  .depth() == CV_8U && (to  .channels() == 3 || to  .channels() == 1) );
    CV_Assert( from.sameSize(to) );

    Mat prev = from.getMat();
    Mat cur  = to.getMat();
    Mat prev_grayscale, cur_grayscale;

    while( (prev.cols/grid_step)*(prev.rows/grid_step) > SHRT_MAX ) //ensure that the number matches is not too big
        grid_step*=2;

    if(prev.channels()==3)
    {
        cvtColor(prev,prev_grayscale,COLOR_BGR2GRAY);
        cvtColor(cur, cur_grayscale, COLOR_BGR2GRAY);
    }
    else
    {
        prev.copyTo(prev_grayscale);
        cur .copyTo(cur_grayscale);
    }

    vector<Point2f> points;
    vector<Point2f> dst_points;
    vector<unsigned char> status;
    vector<float> err;
    vector<Point2f> points_filtered, dst_points_filtered;

    for(int i=0;i<prev.rows;i+=grid_step)
        for(int j=0;j<prev.cols;j+=grid_step)
            points.push_back(Point2f((float)j,(float)i));

    calcOpticalFlowPyrLK(prev_grayscale,cur_grayscale,points,dst_points,status,err,Size(21,21));

    for(unsigned int i=0;i<points.size();i++)
    {
        if(status[i]!=0)
        {
            points_filtered.push_back(points[i]);
            dst_points_filtered.push_back(dst_points[i]);
        }
    }

    flow.create(from.size(),CV_32FC2);
    Mat dense_flow = flow.getMat();

    Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
    gd->setK(k);
    gd->setSigma(sigma);
    gd->setUsePostProcessing(use_post_proc);
    gd->setFGSLambda(fgs_lambda);
    gd->setFGSSigma (fgs_sigma);
    gd->interpolate(prev,points_filtered,cur,dst_points_filtered,dense_flow);
}

}
}