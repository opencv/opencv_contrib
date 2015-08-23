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

#ifndef _OPENCV_LBPFEATURES_H_
#define _OPENCV_LBPFEATURES_H_

#include "precomp.hpp"


#define LBPF_NAME "lbpFeatureParams"

namespace cv {
namespace xobjdetect {

struct CvLBPFeatureParams : CvFeatureParams
{
    CvLBPFeatureParams();

};

class CvLBPEvaluator : public CvFeatureEvaluator
{
public:
    virtual ~CvLBPEvaluator() {}
    virtual void init(const CvFeatureParams *_featureParams,
        int _maxSampleCount, cv::Size _winSize );
    virtual void setImage(const cv::Mat& img, uchar clsLabel, int idx, const std::vector<int> &feature_ind);
    virtual void setWindow(const cv::Point& p)
    { cur_sum = sum.rowRange(p.y, p.y + winSize.height).colRange(p.x, p.x + winSize.width); }
    virtual float operator()(int featureIdx)
    { return (float)features[featureIdx].calc( cur_sum ); }
    virtual void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
protected:
    virtual void generateFeatures();

    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int _block_w, int _block_h  );
        uchar calc( const cv::Mat& _sum );
        void write( cv::FileStorage &fs ) const;

        cv::Rect rect;
        int p[16];

        int x_, y_, block_w_, block_h_, offset_;
        void calcPoints(int offset);
    };
    std::vector<Feature> features;

    cv::Mat sum, cur_sum;
    int offset_;
};

inline uchar CvLBPEvaluator::Feature::calc(const cv::Mat &_sum)
{
    const int* psum = _sum.ptr<int>();

    int cval = psum[p[5]] - psum[p[6]] - psum[p[9]] + psum[p[10]];

    return (uchar)((psum[p[0]] - psum[p[1]] - psum[p[4]] + psum[p[5]] >= cval ? 128 : 0) |   // 0
        (psum[p[1]] - psum[p[2]] - psum[p[5]] + psum[p[6]] >= cval ? 64 : 0) |    // 1
        (psum[p[2]] - psum[p[3]] - psum[p[6]] + psum[p[7]] >= cval ? 32 : 0) |    // 2
        (psum[p[6]] - psum[p[7]] - psum[p[10]] + psum[p[11]] >= cval ? 16 : 0) |  // 5
        (psum[p[10]] - psum[p[11]] - psum[p[14]] + psum[p[15]] >= cval ? 8 : 0) | // 8
        (psum[p[9]] - psum[p[10]] - psum[p[13]] + psum[p[14]] >= cval ? 4 : 0) |  // 7
        (psum[p[8]] - psum[p[9]] - psum[p[12]] + psum[p[13]] >= cval ? 2 : 0) |   // 6
        (psum[p[4]] - psum[p[5]] - psum[p[8]] + psum[p[9]] >= cval ? 1 : 0));     // 3
}

}
}

#endif
