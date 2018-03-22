//TODO: add license

#ifndef __OPENCV_KINFU_ICP_H__
#define __OPENCV_KINFU_ICP_H__

#include "precomp.hpp"

class ICP
{
public:
    ICP(const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold);

    bool estimateTransform(cv::Affine3f& transform,
                           const std::vector<Points>& oldPoints, const std::vector<Normals>& oldNormals,
                           const std::vector<Points>& newPoints, const std::vector<Normals>& newNormals);
private:
    void getAb(const Points oldPts, const Normals oldNrm, const Points newPts, const Normals newNrm,
               cv::Affine3f pose, int level, cv::Matx66f& A, cv::Vec6f& b);

    std::vector<int> iterations;
    float angleThreshold;
    float distanceThreshold;
    cv::kinfu::Intr intrinsics;
};

#endif
