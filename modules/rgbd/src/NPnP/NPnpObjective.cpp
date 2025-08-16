/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifdef HAVE_EIGEN
#include "NPnpObjective.h"
#include "../Utils_Npnp/Definitions.h"
#include "../Utils_Npnp/MatrixUtils.h"
#include <iostream>

namespace NPnP
{
        void PnpObjective::set_C(ColMatrix<3, 9> &C, Eigen::Vector3d point)
        {
                C(0, 0) = C(1, 3) = C(2, 6) = point.x();
                C(0, 1) = C(1, 4) = C(2, 7) = point.y();
                C(0, 2) = C(1, 5) = C(2, 8) = point.z();
        }

        std::shared_ptr<PnpObjective>
        PnpObjective::init(std::shared_ptr<PnpInput> input)
        {
                PnpObjective objective;
                objective.M.setZero();
                objective.T.setZero();
                objective.b.setZero();
                objective.sum_weights = 0.0;

                ColMatrix<3, 9> C = ColMatrix<3, 9>::Zero();
                ColMatrix<3, 9> temp_sum = ColMatrix<3, 9>::Zero();
                RowMatrix<3, 3> eye3 = RowMatrix<3, 3>::Identity();
                RowMatrix<3, 3> Q_sum = RowMatrix<3, 3>::Zero();
                RowMatrix<3, 3> q;
                std::vector<RowMatrix<3, 3>> Q;

                for (int i = 0; i < input->indices_amount; i++)
                {
                        auto index = input->indices[i];
                        auto weight = input->weights[i];
                        auto point = input->points[index];
                        auto line = input->lines[index];
                        objective.sum_weights += weight;

                        PnpObjective::set_C(C, point);
                        q.noalias() = line * line.transpose() / line.squaredNorm();
                        q -= eye3;
                        auto temp = q * q;
                        q = weight * temp;
                        Q_sum += q;
                        temp_sum.noalias() += q * C;
                        Q.emplace_back(q);
                }
                symmetrize(Q_sum);
                RowMatrix<3, 3> Q_sum_inv = Q_sum.inverse();
                objective.T.noalias() = -Q_sum_inv * temp_sum;

                ColMatrix<3, 9> current;
                for (int i = 0; i < input->indices_amount; i++)
                {
                        auto index = input->indices[i];
                        auto point = input->points[index];
                        PnpObjective::set_C(C, point);
                        current = C + objective.T;
                        auto temp = current.transpose() * Q[i];
                        objective.M.noalias() += temp * current;
                }
                symmetrize(objective.M);
                auto &M = objective.M;
                auto &b = objective.b;
                b(34) = M(0, 0) + M(0, 4) + M(0, 8) + M(4, 0) + M(4, 4) + M(4, 8) + M(8, 0) +
                        M(8, 4) + M(8, 8); // q_1 q_1 q_1 q_1
                b(35) = 2 * M(0, 5) - 2 * M(0, 7) + 2 * M(4, 5) - 2 * M(4, 7) + 2 * M(5, 0) +
                        2 * M(5, 4) + 2 * M(5, 8) - 2 * M(7, 0) - 2 * M(7, 4) - 2 * M(7, 8) +
                        2 * M(8, 5) - 2 * M(8, 7); // q_1 q_1 q_1 q_2
                b(36) = -2 * M(0, 2) + 2 * M(0, 6) - 2 * M(2, 0) - 2 * M(2, 4) - 2 * M(2, 8) -
                        2 * M(4, 2) + 2 * M(4, 6) + 2 * M(6, 0) + 2 * M(6, 4) + 2 * M(6, 8) -
                        2 * M(8, 2) + 2 * M(8, 6); // q_1 q_1 q_1 q_3
                b(37) = 2 * M(0, 1) - 2 * M(0, 3) + 2 * M(1, 0) + 2 * M(1, 4) + 2 * M(1, 8) -
                        2 * M(3, 0) - 2 * M(3, 4) - 2 * M(3, 8) + 2 * M(4, 1) - 2 * M(4, 3) +
                        2 * M(8, 1) - 2 * M(8, 3); // q_1 q_1 q_1 q_4
                b(38) = 2 * M(0, 0) - 2 * M(4, 4) - 2 * M(4, 8) + 4 * M(5, 5) - 4 * M(5, 7) -
                        4 * M(7, 5) + 4 * M(7, 7) - 2 * M(8, 4) -
                        2 * M(8, 8); // q_1 q_1 q_2 q_2
                b(39) = 2 * M(0, 1) + 2 * M(0, 3) + 2 * M(1, 0) + 2 * M(1, 4) + 2 * M(1, 8) -
                        4 * M(2, 5) + 4 * M(2, 7) + 2 * M(3, 0) + 2 * M(3, 4) + 2 * M(3, 8) +
                        2 * M(4, 1) + 2 * M(4, 3) - 4 * M(5, 2) + 4 * M(5, 6) + 4 * M(6, 5) -
                        4 * M(6, 7) + 4 * M(7, 2) - 4 * M(7, 6) + 2 * M(8, 1) +
                        2 * M(8, 3); // q_1 q_1 q_2 q_3
                b(40) = +2 * M(0, 2) + 2 * M(0, 6) + 4 * M(1, 5) - 4 * M(1, 7) + 2 * M(2, 0) +
                        2 * M(2, 4) + 2 * M(2, 8) - 4 * M(3, 5) + 4 * M(3, 7) + 2 * M(4, 2) +
                        2 * M(4, 6) + 4 * M(5, 1) - 4 * M(5, 3) + 2 * M(6, 0) + 2 * M(6, 4) +
                        2 * M(6, 8) - 4 * M(7, 1) + 4 * M(7, 3) + 2 * M(8, 2) +
                        2 * M(8, 6); // q_1 q_1 q_2 q_4
                b(41) = -2 * M(0, 0) - 2 * M(0, 8) + 4 * M(2, 2) - 4 * M(2, 6) + 2 * M(4, 4) -
                        4 * M(6, 2) + 4 * M(6, 6) - 2 * M(8, 0) -
                        2 * M(8, 8); // q_1 q_1 q_3 q_3
                b(42) = 2 * M(0, 5) + 2 * M(0, 7) - 4 * M(1, 2) + 4 * M(1, 6) - 4 * M(2, 1) +
                        4 * M(2, 3) + 4 * M(3, 2) - 4 * M(3, 6) + 2 * M(4, 5) + 2 * M(4, 7) +
                        2 * M(5, 0) + 2 * M(5, 4) + 2 * M(5, 8) + 4 * M(6, 1) - 4 * M(6, 3) +
                        2 * M(7, 0) + 2 * M(7, 4) + 2 * M(7, 8) + 2 * M(8, 5) +
                        2 * M(8, 7); // q_1 q_1 q_3 q_4
                b(43) = -2 * M(0, 0) - 2 * M(0, 4) + 4 * M(1, 1) - 4 * M(1, 3) - 4 * M(3, 1) +
                        4 * M(3, 3) - 2 * M(4, 0) - 2 * M(4, 4) +
                        2 * M(8, 8); // q_1 q_1 q_4 q_4
                b(44) = 2 * M(0, 5) - 2 * M(0, 7) - 2 * M(4, 5) + 2 * M(4, 7) + 2 * M(5, 0) -
                        2 * M(5, 4) - 2 * M(5, 8) - 2 * M(7, 0) + 2 * M(7, 4) + 2 * M(7, 8) -
                        2 * M(8, 5) + 2 * M(8, 7); // q_1 q_2 q_2 q_2
                b(45) = -2 * M(0, 2) + 2 * M(0, 6) + 4 * M(1, 5) - 4 * M(1, 7) - 2 * M(2, 0) +
                        2 * M(2, 4) + 2 * M(2, 8) + 4 * M(3, 5) - 4 * M(3, 7) + 2 * M(4, 2) -
                        2 * M(4, 6) + 4 * M(5, 1) + 4 * M(5, 3) + 2 * M(6, 0) - 2 * M(6, 4) -
                        2 * M(6, 8) - 4 * M(7, 1) - 4 * M(7, 3) + 2 * M(8, 2) -
                        2 * M(8, 6); // q_1 q_2 q_2 q_3
                b(46) = 2 * M(0, 1) - 2 * M(0, 3) + 2 * M(1, 0) - 2 * M(1, 4) - 2 * M(1, 8) +
                        4 * M(2, 5) - 4 * M(2, 7) - 2 * M(3, 0) + 2 * M(3, 4) + 2 * M(3, 8) -
                        2 * M(4, 1) + 2 * M(4, 3) + 4 * M(5, 2) + 4 * M(5, 6) + 4 * M(6, 5) -
                        4 * M(6, 7) - 4 * M(7, 2) - 4 * M(7, 6) - 2 * M(8, 1) +
                        2 * M(8, 3); // q_1 q_2 q_2 q_4
                b(47) = -2 * M(0, 5) + 2 * M(0, 7) - 4 * M(1, 2) + 4 * M(1, 6) - 4 * M(2, 1) -
                        4 * M(2, 3) - 4 * M(3, 2) + 4 * M(3, 6) + 2 * M(4, 5) - 2 * M(4, 7) -
                        2 * M(5, 0) + 2 * M(5, 4) - 2 * M(5, 8) + 4 * M(6, 1) + 4 * M(6, 3) +
                        2 * M(7, 0) - 2 * M(7, 4) + 2 * M(7, 8) - 2 * M(8, 5) +
                        2 * M(8, 7); // q_1 q_2 q_3 q_3
                b(48) = 8 * M(1, 1) - 8 * M(2, 2) - 8 * M(3, 3) + 8 * M(5, 5) + 8 * M(6, 6) -
                        8 * M(7, 7); // q_1 q_2 q_3 q_4
                b(49) = -2 * M(0, 5) + 2 * M(0, 7) + 4 * M(1, 2) + 4 * M(1, 6) + 4 * M(2, 1) -
                        4 * M(2, 3) - 4 * M(3, 2) - 4 * M(3, 6) - 2 * M(4, 5) + 2 * M(4, 7) -
                        2 * M(5, 0) - 2 * M(5, 4) + 2 * M(5, 8) + 4 * M(6, 1) - 4 * M(6, 3) +
                        2 * M(7, 0) + 2 * M(7, 4) - 2 * M(7, 8) + 2 * M(8, 5) -
                        2 * M(8, 7); // q_1 q_2 q_4 q_4
                b(50) = 2 * M(0, 2) - 2 * M(0, 6) + 2 * M(2, 0) - 2 * M(2, 4) + 2 * M(2, 8) -
                        2 * M(4, 2) + 2 * M(4, 6) - 2 * M(6, 0) + 2 * M(6, 4) - 2 * M(6, 8) +
                        2 * M(8, 2) - 2 * M(8, 6); // q_1 q_3 q_3 q_3
                b(51) = -2 * M(0, 1) + 2 * M(0, 3) - 2 * M(1, 0) + 2 * M(1, 4) - 2 * M(1, 8) -
                        4 * M(2, 5) - 4 * M(2, 7) + 2 * M(3, 0) - 2 * M(3, 4) + 2 * M(3, 8) +
                        2 * M(4, 1) - 2 * M(4, 3) - 4 * M(5, 2) + 4 * M(5, 6) + 4 * M(6, 5) +
                        4 * M(6, 7) - 4 * M(7, 2) + 4 * M(7, 6) - 2 * M(8, 1) +
                        2 * M(8, 3); // q_1 q_3 q_3 q_4
                b(52) = 2 * M(0, 2) - 2 * M(0, 6) + 4 * M(1, 5) + 4 * M(1, 7) + 2 * M(2, 0) +
                        2 * M(2, 4) - 2 * M(2, 8) - 4 * M(3, 5) - 4 * M(3, 7) + 2 * M(4, 2) -
                        2 * M(4, 6) + 4 * M(5, 1) - 4 * M(5, 3) - 2 * M(6, 0) - 2 * M(6, 4) +
                        2 * M(6, 8) + 4 * M(7, 1) - 4 * M(7, 3) - 2 * M(8, 2) +
                        2 * M(8, 6); // q_1 q_3 q_4 q_4
                b(53) = -2 * M(0, 1) + 2 * M(0, 3) - 2 * M(1, 0) - 2 * M(1, 4) + 2 * M(1, 8) +
                        2 * M(3, 0) + 2 * M(3, 4) - 2 * M(3, 8) - 2 * M(4, 1) + 2 * M(4, 3) +
                        2 * M(8, 1) - 2 * M(8, 3); // q_1 q_4 q_4 q_4
                b(54) = M(0, 0) - M(0, 4) - M(0, 8) - M(4, 0) + M(4, 4) + M(4, 8) - M(8, 0) +
                        M(8, 4) + M(8, 8); // q_2 q_2 q_2 q_2
                b(55) = 2 * M(0, 1) + 2 * M(0, 3) + 2 * M(1, 0) - 2 * M(1, 4) - 2 * M(1, 8) +
                        2 * M(3, 0) - 2 * M(3, 4) - 2 * M(3, 8) - 2 * M(4, 1) - 2 * M(4, 3) -
                        2 * M(8, 1) - 2 * M(8, 3); // q_2 q_2 q_2 q_3
                b(56) = 2 * M(0, 2) + 2 * M(0, 6) + 2 * M(2, 0) - 2 * M(2, 4) - 2 * M(2, 8) -
                        2 * M(4, 2) - 2 * M(4, 6) + 2 * M(6, 0) - 2 * M(6, 4) - 2 * M(6, 8) -
                        2 * M(8, 2) - 2 * M(8, 6); // q_2 q_2 q_2 q_4
                b(57) = -2 * M(0, 0) + 2 * M(0, 4) + 4 * M(1, 1) + 4 * M(1, 3) + 4 * M(3, 1) +
                        4 * M(3, 3) + 2 * M(4, 0) - 2 * M(4, 4) +
                        2 * M(8, 8); // q_2 q_2 q_3 q_3
                b(58) = 2 * M(0, 5) + 2 * M(0, 7) + 4 * M(1, 2) + 4 * M(1, 6) + 4 * M(2, 1) +
                        4 * M(2, 3) + 4 * M(3, 2) + 4 * M(3, 6) - 2 * M(4, 5) - 2 * M(4, 7) +
                        2 * M(5, 0) - 2 * M(5, 4) - 2 * M(5, 8) + 4 * M(6, 1) + 4 * M(6, 3) +
                        2 * M(7, 0) - 2 * M(7, 4) - 2 * M(7, 8) - 2 * M(8, 5) -
                        2 * M(8, 7); // q_2 q_2 q_3 q_4
                b(59) = -2 * M(0, 0) + 2 * M(0, 8) + 4 * M(2, 2) + 4 * M(2, 6) + 2 * M(4, 4) +
                        4 * M(6, 2) + 4 * M(6, 6) + 2 * M(8, 0) -
                        2 * M(8, 8); // q_2 q_2 q_4 q_4
                b(60) = -2 * M(0, 1) - 2 * M(0, 3) - 2 * M(1, 0) + 2 * M(1, 4) - 2 * M(1, 8) -
                        2 * M(3, 0) + 2 * M(3, 4) - 2 * M(3, 8) + 2 * M(4, 1) + 2 * M(4, 3) -
                        2 * M(8, 1) - 2 * M(8, 3); // q_2 q_3 q_3 q_3
                b(61) = -2 * M(0, 2) - 2 * M(0, 6) + 4 * M(1, 5) + 4 * M(1, 7) - 2 * M(2, 0) +
                        2 * M(2, 4) - 2 * M(2, 8) + 4 * M(3, 5) + 4 * M(3, 7) + 2 * M(4, 2) +
                        2 * M(4, 6) + 4 * M(5, 1) + 4 * M(5, 3) - 2 * M(6, 0) + 2 * M(6, 4) -
                        2 * M(6, 8) + 4 * M(7, 1) + 4 * M(7, 3) - 2 * M(8, 2) -
                        2 * M(8, 6); // q_2 q_3 q_3 q_4
                b(62) = -2 * M(0, 1) - 2 * M(0, 3) - 2 * M(1, 0) - 2 * M(1, 4) + 2 * M(1, 8) +
                        4 * M(2, 5) + 4 * M(2, 7) - 2 * M(3, 0) - 2 * M(3, 4) + 2 * M(3, 8) -
                        2 * M(4, 1) - 2 * M(4, 3) + 4 * M(5, 2) + 4 * M(5, 6) + 4 * M(6, 5) +
                        4 * M(6, 7) + 4 * M(7, 2) + 4 * M(7, 6) + 2 * M(8, 1) +
                        2 * M(8, 3); // q_2 q_3 q_4 q_4
                b(63) = -2 * M(0, 2) - 2 * M(0, 6) - 2 * M(2, 0) - 2 * M(2, 4) + 2 * M(2, 8) -
                        2 * M(4, 2) - 2 * M(4, 6) - 2 * M(6, 0) - 2 * M(6, 4) + 2 * M(6, 8) +
                        2 * M(8, 2) + 2 * M(8, 6); // q_2 q_4 q_4 q_4
                b(64) = M(0, 0) - M(0, 4) + M(0, 8) - M(4, 0) + M(4, 4) - M(4, 8) + M(8, 0) -
                        M(8, 4) + M(8, 8); // q_3 q_3 q_3 q_3
                b(65) = -2 * M(0, 5) - 2 * M(0, 7) + 2 * M(4, 5) + 2 * M(4, 7) - 2 * M(5, 0) +
                        2 * M(5, 4) - 2 * M(5, 8) - 2 * M(7, 0) + 2 * M(7, 4) - 2 * M(7, 8) -
                        2 * M(8, 5) - 2 * M(8, 7); // q_3 q_3 q_3 q_4
                b(66) = 2 * M(0, 0) - 2 * M(4, 4) + 2 * M(4, 8) + 4 * M(5, 5) + 4 * M(5, 7) +
                        4 * M(7, 5) + 4 * M(7, 7) + 2 * M(8, 4) -
                        2 * M(8, 8); // q_3 q_3 q_4 q_4
                b(67) = -2 * M(0, 5) - 2 * M(0, 7) - 2 * M(4, 5) - 2 * M(4, 7) - 2 * M(5, 0) -
                        2 * M(5, 4) + 2 * M(5, 8) - 2 * M(7, 0) - 2 * M(7, 4) + 2 * M(7, 8) +
                        2 * M(8, 5) + 2 * M(8, 7); // q_3 q_4 q_4 q_4
                b(68) = M(0, 0) + M(0, 4) - M(0, 8) + M(4, 0) + M(4, 4) - M(4, 8) - M(8, 0) -
                        M(8, 4) + M(8, 8); // q_4 q_4 q_4 q_4
                b *= -1;
                b /= b.norm();
                return std::make_shared<PnpObjective>(objective);
        }
} // namespace NPnP
#endif
