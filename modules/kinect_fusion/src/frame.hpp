//TODO: add license

#ifndef __OPENCV_KINFU_FRAME_H__
#define __OPENCV_KINFU_FRAME_H__

#include "opencv2/kinect_fusion/utils.hpp"

struct Frame
{
public:
    Frame();
    Frame(const Depth, const Intr, int levels, float depthFactor, float sigmaDepth, float sigmaSpatial, int kernelSize);
    Frame(const Points, const Normals, int levels);

    Image render(int level, cv::Affine3f lightPose) const;

    std::vector<Points> points;
    std::vector<Normals> normals;
};

void computePointsNormals(const Intr, float depthFactor, const Depth, Points, Normals );
Depth pyrDownBilateral(const Depth depth, float sigma);
void pyrDownPointsNormals(const Points p, const Normals n, Points& pdown, Normals& ndown);

#endif


