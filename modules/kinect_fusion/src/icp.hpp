//TODO: add license

#ifndef __OPENCV_KINFU_ICP_H__
#define __OPENCV_KINFU_ICP_H__

#include "precomp.hpp"
#include "frame.hpp"

class ICP
{
public:
    ICP(const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold);

    virtual bool estimateTransform(cv::Affine3f& transform, cv::Ptr<Frame> oldFrame, cv::Ptr<Frame> newFrame) const = 0;

    virtual ~ICP() { }

protected:

    std::vector<int> iterations;
    float angleThreshold;
    float distanceThreshold;
    cv::kinfu::Intr intrinsics;
};

cv::Ptr<ICP> makeICP(cv::kinfu::KinFu::KinFuParams::PlatformType t,
                     const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations,
                     float _angleThreshold, float _distanceThreshold);

#endif
