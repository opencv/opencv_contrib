//TODO: add license

#ifndef __OPENCV_KINFU_FRAME_H__
#define __OPENCV_KINFU_FRAME_H__

#include "opencv2/kinect_fusion/utils.hpp"

Distance depthToDistance(Depth depth, Intr intr, float depthFactor);
void computePointsNormals(Intr intr, const Depth &depth, Points &points, Normals &normals, float depthFactor);

struct Frame
{
public:
    Frame(int levels, cv::Size frameSize);
    void computePointsNormals(const std::vector<Depth> &pyramid, Intr intrinsics, float depthFactor);

    cv::Affine3f pose;
    std::vector<Points> points;
    std::vector<Normals> normals;
};

#endif


