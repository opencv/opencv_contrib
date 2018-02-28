//TODO: add license

#ifndef __OPENCV_KINFU_ICP_H__
#define __OPENCV_KINFU_ICP_H__

#include "opencv2/kinect_fusion/utils.hpp"

class ICP
{
public:
    ICP(int nLevels);

    bool estimateTransform(cv::Affine3f& transform,
                           const std::vector<Points>& oldPoints, const std::vector<Normals>& oldNormals,
                           const std::vector<Points>& newPoints, const std::vector<Normals>& newNormals);
private:
    int levels;

};

#endif
