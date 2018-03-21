// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "precomp.hpp"
#include "opencv2/hfs.hpp"
#include "hfs_core.hpp"

namespace cv{ namespace hfs{

class HfsSegmentImpl CV_FINAL : public cv::hfs::HfsSegment{
public:

    void setSegEgbThresholdI(float c) CV_OVERRIDE
    {
        core->hfsSettings.egbThresholdI = c;
    }
    float getSegEgbThresholdI() CV_OVERRIDE {
        return core->hfsSettings.egbThresholdI;
    }


    void setMinRegionSizeI(int n) CV_OVERRIDE
    {
        core->hfsSettings.minRegionSizeI = n;
    }
    int getMinRegionSizeI() CV_OVERRIDE
    {
        return core->hfsSettings.minRegionSizeI;
    }

    void setSegEgbThresholdII(float c) CV_OVERRIDE
    {
        core->hfsSettings.egbThresholdII = c;
    }
    float getSegEgbThresholdII() CV_OVERRIDE {
        return core->hfsSettings.egbThresholdII;
    }


    void setMinRegionSizeII(int n) CV_OVERRIDE
    {
        core->hfsSettings.minRegionSizeII = n;
    }
    int getMinRegionSizeII() CV_OVERRIDE
    {
        return core->hfsSettings.minRegionSizeII;
    }

    void setSpatialWeight(float w) CV_OVERRIDE
    {
        core->hfsSettings.slicSettings.coh_weight = w;
        core->reconstructEngine();
    }
    float getSpatialWeight() CV_OVERRIDE
    {
        return core->hfsSettings.slicSettings.coh_weight;
    }


    void setSlicSpixelSize(int n) CV_OVERRIDE
    {
        core->hfsSettings.slicSettings.spixel_size = n;
        core->reconstructEngine();
    }
    int getSlicSpixelSize() CV_OVERRIDE
    {
        return core->hfsSettings.slicSettings.spixel_size;
    }


    void setNumSlicIter(int n) CV_OVERRIDE
    {
        core->hfsSettings.slicSettings.num_iters = n;
        core->reconstructEngine();
    }
    int getNumSlicIter() CV_OVERRIDE
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

    Mat performSegmentGpu(InputArray src, bool ifDraw = true) CV_OVERRIDE;
    Mat performSegmentCpu(InputArray src, bool ifDraw = true) CV_OVERRIDE;
private:
    Ptr<HfsCore> core;
};

Mat HfsSegmentImpl::performSegmentGpu(InputArray src, bool ifDraw) {
    Mat src_ = src.getMat();

    CV_Assert(src_.rows == core->hfsSettings.slicSettings.img_size.y);
    CV_Assert(src_.cols == core->hfsSettings.slicSettings.img_size.x);

    Mat seg;
    int num_css = core->processImageGpu(src_, seg);
    if(ifDraw){
        Mat res;
        core->drawSegmentationRes( seg, src_, num_css, res );
        return res;
    }else{
        return seg;
    }
}

Mat HfsSegmentImpl::performSegmentCpu(InputArray src, bool ifDraw) {
    Mat src_ = src.getMat();

    CV_Assert(src_.rows == core->hfsSettings.slicSettings.img_size.y);
    CV_Assert(src_.cols == core->hfsSettings.slicSettings.img_size.x);

    Mat seg;
    int num_css = core->processImageCpu(src_, seg);
    if (ifDraw) {
        Mat res;
        core->drawSegmentationRes(seg, src_, num_css, res);
        return res;
    }
    else {
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
