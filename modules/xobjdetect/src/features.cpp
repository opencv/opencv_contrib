/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "precomp.hpp"

namespace cv {
namespace xobjdetect {

float calcNormFactor( const Mat& sum, const Mat& sqSum )
{
    CV_DbgAssert( sum.cols > 3 && sqSum.rows > 3 );
    Rect normrect( 1, 1, sum.cols - 3, sum.rows - 3 );
    size_t p0, p1, p2, p3;
    CV_SUM_OFFSETS( p0, p1, p2, p3, normrect, sum.step1() )
    double area = normrect.width * normrect.height;
    const int *sp = sum.ptr<int>();
    int valSum = sp[p0] - sp[p1] - sp[p2] + sp[p3];
    const double *sqp = sqSum.ptr<double>();
    double valSqSum = sqp[p0] - sqp[p1] - sqp[p2] + sqp[p3];
    return (float) sqrt( (double) (area * valSqSum - (double)valSum * valSum) );
}

CvParams::CvParams() : name( "params" ) {}
void CvParams::printDefaults() const
{ std::cout << "--" << name << "--" << std::endl; }
void CvParams::printAttrs() const {}
bool CvParams::scanAttr( const std::string, const std::string ) { return false; }


//---------------------------- FeatureParams --------------------------------------

CvFeatureParams::CvFeatureParams() : maxCatCount( 0 ), featSize( 1 )
{
    name = CC_FEATURE_PARAMS;
}

void CvFeatureParams::init( const CvFeatureParams& fp )
{
    maxCatCount = fp.maxCatCount;
    featSize = fp.featSize;
}

void CvFeatureParams::write( FileStorage &fs ) const
{
    fs << CC_MAX_CAT_COUNT << maxCatCount;
    fs << CC_FEATURE_SIZE << featSize;
}

bool CvFeatureParams::read( const FileNode &node )
{
    if ( node.empty() )
        return false;
    maxCatCount = node[CC_MAX_CAT_COUNT];
    featSize = node[CC_FEATURE_SIZE];
    return ( maxCatCount >= 0 && featSize >= 1 );
}

Ptr<CvFeatureParams> CvFeatureParams::create()
{
    return Ptr<CvFeatureParams>(new CvLBPFeatureParams);
}

//------------------------------------- FeatureEvaluator ---------------------------------------

void CvFeatureEvaluator::init(const CvFeatureParams *_featureParams,
                              int _maxSampleCount, Size _winSize )
{
    CV_Assert(_maxSampleCount > 0);
    featureParams = (CvFeatureParams *)_featureParams;
    winSize = _winSize;
    numFeatures = 0;
    cls.create( (int)_maxSampleCount, 1, CV_32FC1 );
    generateFeatures();
}

void CvFeatureEvaluator::setImage(const Mat &, uchar clsLabel, int idx, const std::vector<int>&)
{
    //CV_Assert(img.cols == winSize.width);
    //CV_Assert(img.rows == winSize.height);
    CV_Assert(idx < cls.rows);
    cls.ptr<float>(idx)[0] = clsLabel;
}

Ptr<CvFeatureEvaluator> CvFeatureEvaluator::create()
{
    return Ptr<CvFeatureEvaluator>(new CvLBPEvaluator);
}

}
}
