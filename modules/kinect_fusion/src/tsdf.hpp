//TODO: add license

#ifndef __OPENCV_KINFU_TSDF_H__
#define __OPENCV_KINFU_TSDF_H__

#include "opencv2/kinect_fusion/utils.hpp"

class TSDFVolume
{
public:
    // dimension in voxels, size in meters
    TSDFVolume(int _dims, float _size);

    void integrate(Distance distance, cv::Affine3f pose, Intr intrinsics);
    //TODO: inside should be like this:
    //    volume_->raycast(poses_.back(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]);
    //    for (int i = 1; i < LEVELS; ++i)
    //        resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
    void raycast(cv::Affine3f pose, Intr intrinsics,
                 std::vector<Points>& pointsPyr, std::vector<Normals>& normalsPyr);

    Points fetchCloud();

private:

};

#endif


