//TODO: add license

#include "tsdf.hpp"

using namespace cv::kinfu;

// dimension in voxels, size in meters
TSDFVolume::TSDFVolume(int _dims, float _size)
{
    //TODO: allocate memory, etc
}

void TSDFVolume::integrate(Distance distance, cv::Affine3f pose, Intr intrinsics)
{
    //TODO: really integrate somehow
}

void TSDFVolume::raycast(cv::Affine3f pose, Intr intrinsics,
                         std::vector<Points>& pointsPyr, std::vector<Normals>& normalsPyr)
{
    //TODO: make it like this:
    //    volume_->raycast(poses_.back(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]);
    //    for (int i = 1; i < LEVELS; ++i)
    //        resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);

}

Points TSDFVolume::fetchCloud()
{
    //TODO: implement this
}
