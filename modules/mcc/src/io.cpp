// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                       License Agreement
//              For Open Source Computer Vision Library
//
// Copyright(C) 2020, Huawei Technologies Co.,Ltd. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//             http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "opencv2/mcc/io.hpp"
namespace cv
{
namespace ccm
{
IO::IO(std::string illuminant_, std::string observer_) :illuminant(illuminant_), observer(observer_) {};

bool IO::operator<(const IO& other) const
{
    return (illuminant < other.illuminant || ((illuminant == other.illuminant) && (observer < other.observer)));
}

bool IO::operator==(const IO& other) const
{
    return illuminant == other.illuminant && observer == other.observer;
};

// data from https://en.wikipedia.org/wiki/Standard_illuminant.
std::vector<double> xyY2XYZ(const std::vector<double>& xyY)
{
    double Y = xyY.size() >= 3 ? xyY[2] : 1;
    return { Y * xyY[0] / xyY[1], Y, Y / xyY[1] * (1 - xyY[0] - xyY[1]) };
}

} // namespace ccm
} // namespace cv
