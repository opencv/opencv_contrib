/*M///////////////////////////////////////////////////////////////////////////////////////
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//                        (3-clause BSD License)
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * Neither the names of the copyright holders nor the names of the contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// Author : Awabot SAS
// Copyright (C) 2015, Awabot SAS, all rights reserved.
//
//M*/

#ifndef __OPENCV_SKELETON_FRAME_HPP__
#define __OPENCV_SKELETON_FRAME_HPP__

#ifdef __cplusplus

#include <vector>

#include <opencv2/core.hpp>

/** @defgroup gestures Gestures Recognition module
 */

namespace cv
{
    namespace gestures
    {
        //! @addtogroup gestures
        //! @{
        /* @brief Class to describe a skeleton frame coming from a MoCap stream, and compute its descriptor.
         */
        class CV_EXPORTS SkeletonFrame
        {
            public:
                /**
                 List of joints used in the descriptor.
                 */
                CV_WRAP enum Joint
                {
                    JOINT_HIP_CENTER = 0,
                    JOINT_SHOULDER_CENTER,
                    JOINT_HEAD,
                    JOINT_SHOULDER_LEFT,
                    JOINT_ELBOW_LEFT,
                    JOINT_HAND_LEFT,
                    JOINT_SHOULDER_RIGHT,
                    JOINT_ELBOW_RIGHT,
                    JOINT_HAND_RIGHT,
                    JOINT_HIP_LEFT,
                    JOINT_HIP_RIGHT,
                    JOINTS_COUNT
                };


                /**
                 Default constructor
                 */
                CV_WRAP SkeletonFrame();
                /**
                 Full constructor.
                 @sa fromData
                 */
                CV_WRAP SkeletonFrame(InputArray data);

                /**
                 Read mocap data and initialize object
                 @param data Array containing 11 columns, one for each joint
                 and 5 rows, first 3 for the world's coordinates, in meter, in the sensor frame, last 2 for the image's coordinates, in pixels.
                 */
                CV_WRAP void fromData(InputArray data);

                /**
                 Normalize all joints world position to get rid of variation in people's height and shape.
                 */
                CV_WRAP void normalize();

                /**
                 Returns true if mocap data is valid (ie not all zeros)
                 */
                CV_WRAP bool isValid() const;


                /**
                 Create a descriptor to be accumulated over multiple frames and fed to the neural network.
                 @param previous SkeletonFrame at time t-1 used to compute accelerations and velocities of each joint.
                 @param previous2 SkeletonFrame at time t-2 used to compute accelerations of each joint.
                 @param descriptor Output array of size 172x1 containing the complete descriptor for this frame.
                 First 28 values are various angles, next 54 values are distances between each possible pair of joints,
                 last 90 values are normalized positions, velocities and accelerations of each joint.
                 */
                CV_WRAP void createDescriptor(const SkeletonFrame& previous, const SkeletonFrame& previous2, OutputArray descriptor);

                /**
                 Returns the position of a given joint in the world, expressed in meters, in the sensor frame.
                 @param joint The joint whose position should be returned.
                 */
                CV_WRAP inline const cv::Vec3f getJointWorldCoords(const Joint& joint) const
                {
                    return mJointsWorldCoords[joint];
                }

                /**
                 Returns the position of a given joint in the image, expressed in pixels, from the top-left corner.
                 @param joint The joint whose position should be returned.
                 */
                CV_WRAP inline const cv::Vec2i getJointPixelCoords(const Joint& joint) const
                {
                    return mJointsPixelCoords[joint];
                }

            private:
                static const Joint ORDERED_JOINTS[JOINTS_COUNT];
                static const Joint PARENT_JOINTS[JOINTS_COUNT];
                static const float BONES_LENGTH[JOINTS_COUNT];
                static const Joint ANGLES_JOINTS[9][3];
                static const Joint BASIS_JOINTS[6];

                void subtractOrigin();
                void normalizeByHeight();
                void normalizeByBonelength();

                void copyPositions(cv::Mat positions);
                void computeVelocities(const SkeletonFrame& previous, cv::Mat velocities);
                void computeAccelerations(const SkeletonFrame& previous, const SkeletonFrame& previous2, cv::Mat accelerations);

                void computeAngles(cv::Mat angles);
                void computeDistances(cv::Mat distances);


                cv::Vec3f mJointsWorldCoords[JOINTS_COUNT];
                cv::Vec2i mJointsPixelCoords[JOINTS_COUNT];

                bool mValid;
        };
    } // namespace gestures
} // namespace cv

#endif // __cplusplus
#endif // __OPENCV_SKELETON_FRAME_HPP__
