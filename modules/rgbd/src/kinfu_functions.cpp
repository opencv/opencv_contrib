// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "kinfu_functions.hpp"

namespace cv {

//template< typename MatType >

bool kinfuCommonUpdateT(Odometry& odometry, Volume& volume, InputArray _depth, OdometryFrame& prevFrame, OdometryFrame& renderFrame, Matx44f& pose, int& frameCounter)
{
    CV_TRACE_FUNCTION();

    //MatType depth;
    Mat depth = _depth.getMat();
    //if (_depth.type() != DEPTH_TYPE)
    //    _depth.convertTo(depth, DEPTH_TYPE);
    //else
    //    depth = _depth;

    OdometryFrame newFrame = odometry.createOdometryFrame();
    newFrame.setDepth(depth);

    if (frameCounter == 0)
    {
        odometry.prepareFrame(newFrame);
        // use depth instead of distance
        volume.integrate(depth, pose);
    }
    else
    {
        Affine3f affine;
        Matx44d mrt;
        Mat Rt;
        odometry.prepareFrames(newFrame, prevFrame);
        bool success = odometry.compute(newFrame, prevFrame, Rt);
        if (!success)
            return false;
        affine.matrix = Matx44f(Rt);
        pose = (Affine3f(pose) * affine).matrix;

        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());
        // We do not integrate volume if camera does not move
        if ((rnorm + tnorm) / 2 >= 0.f/*params.tsdf_min_camera_movement*/)
        {
            // use depth instead of distance
            volume.integrate(depth, pose);
        }

        //MatType points, normals;
        Mat points, normals;
        newFrame.getPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
        newFrame.getPyramidAt(normals, OdometryFramePyramidType::PYR_NORM, 0);
        volume.raycast(pose, points, normals);

        newFrame.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
        newFrame.setPyramidAt(normals, OdometryFramePyramidType::PYR_NORM, 0);
    }

    renderFrame = newFrame;
    prevFrame = newFrame;

    frameCounter++;
    return true;
}


} // namespace cv
