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

#include <opencv2/gestures/skeleton_frame.hpp>

#include <cmath>

#include <opencv2/core/core_c.h>
#include <opencv2/imgproc.hpp>

namespace cv
{
    namespace gestures
    {
        const SkeletonFrame::Joint SkeletonFrame::ORDERED_JOINTS[JOINTS_COUNT] =
        {
            JOINT_HIP_CENTER,
            JOINT_SHOULDER_CENTER,
            JOINT_HIP_RIGHT,
            JOINT_HIP_LEFT,
            JOINT_SHOULDER_LEFT,
            JOINT_HEAD,
            JOINT_SHOULDER_RIGHT,
            JOINT_ELBOW_LEFT,
            JOINT_ELBOW_RIGHT,
            JOINT_HAND_LEFT,
            JOINT_HAND_RIGHT
        };

        const SkeletonFrame::Joint SkeletonFrame::PARENT_JOINTS[JOINTS_COUNT] =
        {
            JOINT_HIP_CENTER,
            JOINT_HIP_CENTER,
            JOINT_SHOULDER_CENTER,
            JOINT_SHOULDER_CENTER,
            JOINT_SHOULDER_LEFT,
            JOINT_ELBOW_LEFT,
            JOINT_SHOULDER_CENTER,
            JOINT_SHOULDER_RIGHT,
            JOINT_ELBOW_RIGHT,
            JOINT_HIP_CENTER,
            JOINT_HIP_CENTER
        };

        const float SkeletonFrame::BONES_LENGTH[JOINTS_COUNT] =
        {
            0.0,
            0.429,
            0.188,
            0.201,
            0.261,
            0.325,
            0.202,
            0.259,
            0.332,
            0.104,
            0.104
        };

        const SkeletonFrame::Joint SkeletonFrame::ANGLES_JOINTS[9][3] =
        {
            {JOINT_HEAD, JOINT_SHOULDER_CENTER, JOINT_HIP_CENTER},
            {JOINT_SHOULDER_LEFT, JOINT_SHOULDER_CENTER, JOINT_SHOULDER_RIGHT},
            {JOINT_ELBOW_LEFT, JOINT_SHOULDER_LEFT, JOINT_SHOULDER_CENTER},
            {JOINT_ELBOW_RIGHT, JOINT_SHOULDER_RIGHT, JOINT_SHOULDER_CENTER},
            {JOINT_HAND_RIGHT, JOINT_ELBOW_RIGHT, JOINT_SHOULDER_RIGHT},
            {JOINT_HAND_LEFT, JOINT_ELBOW_LEFT, JOINT_SHOULDER_LEFT},
            {JOINT_HAND_LEFT, JOINT_HIP_CENTER, JOINT_HAND_RIGHT},
            {JOINT_HAND_LEFT, JOINT_HIP_CENTER, JOINT_SHOULDER_CENTER},
            {JOINT_HAND_RIGHT, JOINT_HIP_CENTER, JOINT_SHOULDER_CENTER}
        };

        const SkeletonFrame::Joint SkeletonFrame::BASIS_JOINTS[6] =
        {
            JOINT_SHOULDER_CENTER,
            JOINT_SHOULDER_LEFT,
            JOINT_SHOULDER_RIGHT,
            JOINT_HIP_CENTER,
            JOINT_HIP_LEFT,
            JOINT_HIP_RIGHT
        };


        SkeletonFrame::SkeletonFrame():
            mValid(false)
        {
        }

        SkeletonFrame::SkeletonFrame(InputArray data)
        {
            fromData(data);
        }

        void SkeletonFrame::fromData(InputArray data)
        {
            Mat data_mat = data.getMat();

            CV_Assert(data_mat.type() == CV_32F && data_mat.cols == JOINTS_COUNT && data_mat.rows == 5);

            double max;
            minMaxLoc(abs(data_mat), NULL, &max, NULL, NULL);
            if(max <= 0)
            {
                mValid = false;
                return;
            }

            for(int j = 0; j < JOINTS_COUNT; ++j)
            {
                data_mat(Range(0,3), Range(j,j+1)).copyTo(mJointsWorldCoords[j]);
                data_mat(Range(3,5), Range(j,j+1)).convertTo(mJointsPixelCoords[j], CV_32S);
            }

            mValid = true;
        }

        void SkeletonFrame::normalize()
        {
            subtractOrigin();
            normalizeByHeight();

            normalizeByBonelength();
        }

        bool SkeletonFrame::isValid() const
        {
            return mValid;
        }

        void SkeletonFrame::createDescriptor(const SkeletonFrame& previous, const SkeletonFrame& previous2, OutputArray descriptor)
        {
            descriptor.create(cv::Size(1, JOINTS_COUNT - 1 + 2*9 + JOINTS_COUNT * (JOINTS_COUNT - 1) / 2 - 1 + (JOINTS_COUNT - 1) * 9), CV_32F);
            Mat dest = descriptor.getMat();
            dest.setTo(cv::Scalar(0.0));

            subtractOrigin();
            normalizeByHeight();

            int dest_idx = 0;
            computeAngles(dest(Range(0, JOINTS_COUNT - 1 + 2*9), Range(0,1)));
            dest_idx += JOINTS_COUNT - 1 + 2*9;
            computeDistances(dest(Range(dest_idx, dest_idx + JOINTS_COUNT * (JOINTS_COUNT - 1) / 2 - 1), Range(0,1)));
            dest_idx += JOINTS_COUNT * (JOINTS_COUNT - 1) / 2 - 1;

            normalizeByBonelength();

            copyPositions(dest(Range(dest_idx, dest_idx + (JOINTS_COUNT - 1) * 9), Range(0,1)));
            computeVelocities(previous, dest(Range(dest_idx, dest_idx + (JOINTS_COUNT - 1) * 9), Range(0,1)));
            computeAccelerations(previous, previous2, dest(Range(dest_idx, dest_idx + (JOINTS_COUNT - 1) * 9), Range(0,1)));
        }


        void SkeletonFrame::subtractOrigin()
        {
            for(int j = 1; j < JOINTS_COUNT; ++j)
            {
                mJointsWorldCoords[j] -= mJointsWorldCoords[JOINT_HIP_CENTER];
            }

            mJointsWorldCoords[JOINT_HIP_CENTER] = Vec3f(0.0, 0.0, 0.0);
        }

        void SkeletonFrame::normalizeByHeight()
        {
            double height = norm(mJointsWorldCoords[JOINT_SHOULDER_CENTER], mJointsWorldCoords[JOINT_HIP_CENTER]);

            for(int j = 1; j < JOINTS_COUNT; ++j)
            {
                mJointsWorldCoords[j] *= 1.0/height;
            }
        }

        void SkeletonFrame::normalizeByBonelength()
        {
            Vec3f temp[JOINTS_COUNT] = {Vec3f(0.0, 0.0, 0.0)};

            for(int j = 1; j < JOINTS_COUNT; ++j)
            {
                Vec3f bone = mJointsWorldCoords[j] - mJointsWorldCoords[PARENT_JOINTS[j]];
                temp[j] = temp[PARENT_JOINTS[j]] + BONES_LENGTH[j] / norm(bone) * bone;
            }

            for(int j = 1; j < JOINTS_COUNT; ++j)
            {
                mJointsWorldCoords[j] = temp[j];
            }
        }


        void SkeletonFrame::copyPositions(Mat positions)
        {
            for(int j = 0; j < JOINTS_COUNT-1; ++j)
            {
                Joint joint = ORDERED_JOINTS[j+1];
                Mat(mJointsWorldCoords[joint]).copyTo(positions(Range(9*j, 9*j+3), Range(0,1)));
            }
        }

        void SkeletonFrame::computeVelocities(const SkeletonFrame& previous, Mat velocities)
        {
            for(int j = 0; j < JOINTS_COUNT-1; ++j)
            {
                Joint joint = ORDERED_JOINTS[j+1];
                Mat(mJointsWorldCoords[joint] - previous.getJointWorldCoords(joint)).copyTo(
                        velocities(Range(9*j+3, 9*j+6), Range(0,1)));
            }
        }

        void SkeletonFrame::computeAccelerations(const SkeletonFrame& previous, const SkeletonFrame& previous2, Mat accelerations)
        {
            for(int j = 0; j < JOINTS_COUNT-1; ++j)
            {
                Joint joint = ORDERED_JOINTS[j+1];
                Mat(mJointsWorldCoords[joint] - 2 * previous.getJointWorldCoords(joint) + previous2.getJointWorldCoords(joint)).copyTo(
                        accelerations(Range(9*j+6, 9*j+9), Range(0,1)));
            }
        }


        void SkeletonFrame::computeAngles(Mat angles)
        {
            Mat basisJoints(3, 6, CV_32F, Scalar(0.0));

            for(int j = 0; j < 6; ++j)
            {
                Mat(mJointsWorldCoords[BASIS_JOINTS[j]]).copyTo(basisJoints.col(j));
            }

            Mat covar, mean;
            calcCovarMatrix(basisJoints, covar, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);

            Mat eigval, eigvec;
            eigen(covar, eigval, eigvec);

            Vec3f basis_x(eigvec.row(0));
            basis_x *= -1;
            Vec3f basis_z(eigvec.row(2));

            int i = 0;
            for(int j = 1; j < JOINTS_COUNT; ++j)
            {
                Joint joint = ORDERED_JOINTS[j];
                angles.at<float>(i++) = acos(std::max(-1.0, std::min(1.0,
                                mJointsWorldCoords[joint].dot(basis_z) / norm(mJointsWorldCoords[joint]))));
            }

            for(int a = 0; a < 9; ++a)
            {
                Vec3f v1 = mJointsWorldCoords[ANGLES_JOINTS[a][0]] - mJointsWorldCoords[ANGLES_JOINTS[a][1]];
                Vec3f v2 = mJointsWorldCoords[ANGLES_JOINTS[a][2]] - mJointsWorldCoords[ANGLES_JOINTS[a][1]];

                angles.at<float>(i++) = acos(std::max(-1.0, std::min(1.0,
                                v1.dot(v2) / norm(v1) / norm(v2) )));

                Vec3f ux = basis_x - v1 * v1.dot(basis_x) / norm(v1) / norm(v1);
                Vec3f pe = v2 - v1 * v1.dot(v2) / norm(v1) / norm(v1);

                angles.at<float>(i++) = acos(std::max(-1.0, std::min(1.0,
                                ux.dot(pe) / norm(ux) / norm(pe))));
            }
        }

        void SkeletonFrame::computeDistances(Mat distances)
        {
            int i = 0;
            for(int j = 0; j < JOINTS_COUNT - 1; ++j)
            {
                for(int j2 = j+1; j2 < JOINTS_COUNT; ++j2)
                {
                    if(j2 == 1)
                    {
                        continue;
                    }

                    Joint joint1 = ORDERED_JOINTS[j];
                    Joint joint2 = ORDERED_JOINTS[j2];

                    distances.at<float>(i++) = norm(mJointsWorldCoords[joint1], mJointsWorldCoords[joint2]);
                }
            }
        }
    } // namespace gestures
} // namespace cv
