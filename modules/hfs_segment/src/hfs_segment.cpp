/*M///////////////////////////////////////////////////////////////////////////////////////
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                              License Agreement
//                    For Open Source Computer Vision Library
//                           (3 - clause BSD License)
//
// Copyright(C) 2000 - 2016, Intel Corporation, all rights reserved.
// Copyright(C) 2009 - 2011, Willow Garage Inc., all rights reserved.
// Copyright(C) 2009 - 2016, NVIDIA Corporation, all rights reserved.
// Copyright(C) 2010 - 2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright(C) 2015 - 2016, OpenCV Foundation, all rights reserved.
// Copyright(C) 2015 - 2016, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met :
//
//      * Redistributions of source code must retain the above copyright notice,
//        this list of conditions and the following disclaimer.
//
//      * Redistributions in binary form must reproduce the above copyright notice,
//        this list of conditions and the following disclaimer in the documentation
//        and / or other materials provided with the distribution.
//
//      * Neither the names of the copyright holders nor the names of the contributors
//        may be used to endorse or promote products derived from this software
//        without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


#include "precomp.hpp"
#include "opencv2/hfs_segment.hpp"
#include "hfs_core.hpp"

namespace cv{ namespace hfs{

class HfsSegmentImpl : public cv::hfs::HfsSegment{
public:

    void setSegEgbThresholdI(float c) 
    { 
        core->hfsSettings.egbThresholdI = c; 
    }
    float getSegEgbThresholdI() { 
        return core->hfsSettings.egbThresholdI; 
    }

    
    void setMinRegionSizeI(int n)
    {
        core->hfsSettings.minRegionSizeI = n;
    }
    int getMinRegionSizeI()
    {
        return core->hfsSettings.minRegionSizeI;
    }
    
    void setSegEgbThresholdII(float c)
    {
        core->hfsSettings.egbThresholdII = c;
    }
    float getSegEgbThresholdII() {
        return core->hfsSettings.egbThresholdII;
    }


    void setMinRegionSizeII(int n)
    {
        core->hfsSettings.minRegionSizeII = n;
    }
    int getMinRegionSizeII()
    {
        return core->hfsSettings.minRegionSizeII;
    }

    void setSpatialWeight(float w)
    { 
        core->hfsSettings.slicSettings.coh_weight = w;
        core->reconstructEngine();
    }
    float getSpatialWeight() 
    { 
        return core->hfsSettings.slicSettings.coh_weight; 
    }
    
    
    void setSlicSpixelSize(int n) 
    { 
        core->hfsSettings.slicSettings.spixel_size = n;
        core->reconstructEngine();
    }
    int getSlicSpixelSize() 
    { 
        return core->hfsSettings.slicSettings.spixel_size; 
    }
    
    
    void setNumSlicIter(int n) 
    { 
        core->hfsSettings.slicSettings.num_iters = n;
        core->reconstructEngine();
    }
    int getNumSlicIter() 
    { 
        return core->hfsSettings.slicSettings.num_iters; 
    }

    
    HfsSegmentImpl(int height, int width, 
        float segEgbThresholdI, int minRegionSizeI, float segEgbThresholdII, int minRegionSizeII,
        float spatialWeight, int spixelSize, int numIter) 
    {
        core = Ptr<HfsCore>(new HfsCore(height, width, 
            segEgbThresholdI, minRegionSizeI, segEgbThresholdII, minRegionSizeII,
            spatialWeight, spixelSize, numIter));
    }
    
    Mat performSegment(const Mat& src, bool ifDraw = true);

private:
    Ptr<HfsCore> core;
};

Mat HfsSegmentImpl::performSegment(const Mat& src, bool ifDraw) {
    CV_Assert(src.rows == core->hfsSettings.slicSettings.img_size.y);
    CV_Assert(src.cols == core->hfsSettings.slicSettings.img_size.x);

    Mat seg;
    int num_css = core->processImage(src, seg);
    if(ifDraw){
        Mat res;
        core->drawSegmentationRes( seg, src, num_css, res );
        return res;
    }else{
        return seg;
    }
}

Ptr<HfsSegment> HfsSegment::create(int height, int width, float segEgbThresholdI, int minRegionSizeI,
                                   float segEgbThresholdII, int minRegionSizeII,
                                   float spatialWeight, int spixelSize, int numIter) 
{
    return Ptr<HfsSegmentImpl>(new HfsSegmentImpl(height, width, 
        segEgbThresholdI, minRegionSizeI, segEgbThresholdII, minRegionSizeII,
        spatialWeight, spixelSize, numIter));
}

}}
