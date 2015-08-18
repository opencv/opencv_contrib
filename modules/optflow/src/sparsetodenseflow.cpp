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
                                                int grid_step, float lambda, int k, 
                                                float sigma, float inlier_eps, 
                                                float fgs_lambda, float fgs_sigma)
{
    Mat prev = from.getMat();
    Mat cur = to.getMat();
    Mat prev_grayscale, cur_grayscale;

    vector<Point2f> points;
    vector<Point2f> dst_points;
    vector<unsigned char> status;
    vector<float> err;

    for(int i=0;i<prev.rows;i+=grid_step)
        for(int j=0;j<prev.cols;j+=grid_step)
            points.push_back(Point2f(j,i));

    cvtColor(prev,prev_grayscale,COLOR_BGR2GRAY);
    cvtColor(cur, cur_grayscale, COLOR_BGR2GRAY);
    calcOpticalFlowPyrLK(prev_grayscale,cur_grayscale,points,dst_points,status,err,Size(21,21));

    vector<ximgproc::SparseMatch> matches;
    flow.create(from.size(),CV_32FC2);
    Mat dense_flow = flow.getMat();
        
    for(int i=0;i<points.size();i++)
    {
        if(status[i]!=0)
            matches.push_back(ximgproc::SparseMatch(points[i],dst_points[i]));
    }

    Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
    gd->interpolate(prev,cur,matches,dense_flow);
}

}
}