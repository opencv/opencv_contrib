// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_DYNAFU_NONRIGID_ICP_H__
#define __OPENCV_DYNAFU_NONRIGID_ICP_H__

#include "precomp.hpp"
#include "kinfu_frame.hpp"
#include "warpfield.hpp"
#include "dynafu_tsdf.hpp"

namespace cv {
namespace dynafu {

class NonRigidICP
{
public:
    NonRigidICP(const cv::kinfu::Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume, int _iterations);

    virtual bool estimateWarpNodes(WarpField& currentWarp, const Affine3f& pose,
                                   InputArray vertImage, InputArray oldPoints,
                                   InputArray oldNormals,
                                   InputArray newPoints, InputArray newNormals) const = 0;

    virtual ~NonRigidICP() { }

protected:

    int iterations;
    const cv::Ptr<TSDFVolume>& volume;
    cv::kinfu::Intr intrinsics;
};

cv::Ptr<NonRigidICP> makeNonRigidICP(const cv::kinfu::Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume,
                                     int _iterations);

} // namespace dynafu
} // namespace cv
#endif
