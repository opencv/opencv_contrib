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

#ifndef __OPENCV_MCC_IO_HPP__
#define __OPENCV_MCC_IO_HPP__

#include <string>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>

namespace cv
{
namespace ccm
{

/* *\ brief Io is the meaning of illuminant and observer. See notes of ccm.hpp
   *          for supported list for illuminant and observer*/
class IO
{
public:

    std::string illuminant;
    std::string observer;

    IO() {};

    IO(std::string illuminant_, std::string observer_) :illuminant(illuminant_), observer(observer_) {};

    virtual ~IO() {};

    bool operator<(const IO& other) const
    {
        return (illuminant < other.illuminant || ((illuminant == other.illuminant) && (observer < other.observer)));
    }

    bool operator==(const IO& other) const
    {
        return illuminant == other.illuminant && observer == other.observer;
    };
};

const IO A_2("A", "2"), A_10("A", "10"),
    D50_2("D50", "2"), D50_10("D50", "10"),
    D55_2("D55", "2"), D55_10("D55", "10"),
    D65_2("D65", "2"), D65_10("D65", "10"),
    D75_2("D75", "2"), D75_10("D75", "10"),
    E_2("E", "2"), E_10("E", "10");

// data from https://en.wikipedia.org/wiki/Standard_illuminant.
const static std::map<IO, std::vector<double>> illuminants_xy =
{
    {A_2, { 0.44757, 0.40745 }}, {A_10, { 0.45117, 0.40594 }},
    {D50_2, { 0.34567, 0.35850 }}, {D50_10, { 0.34773, 0.35952 }},
    {D55_2, { 0.33242, 0.34743 }}, {D55_10, { 0.33411, 0.34877 }},
    {D65_2, { 0.31271, 0.32902 }}, {D65_10, { 0.31382, 0.33100 }},
    {D75_2, { 0.29902, 0.31485 }}, {D75_10, { 0.45117, 0.40594 }},
    {E_2, { 1 / 3, 1 / 3 }}, {E_10, { 1 / 3, 1 / 3 }},
};

std::vector<double> xyY2XYZ(const std::vector<double>& xyY);
std::vector<double> xyY2XYZ(const std::vector<double>& xyY)
{
    double Y = xyY.size() >= 3 ? xyY[2] : 1;
    return { Y * xyY[0] / xyY[1], Y, Y / xyY[1] * (1 - xyY[0] - xyY[1]) };
}

/* *\ brief function to get illuminants*/
static std::map <IO, std::vector<double>> getIlluminant();
static std::map <IO, std::vector<double>> getIlluminant()
{
    std::map <IO, std::vector<double>>  illuminants;
    for (auto it = illuminants_xy.begin(); it != illuminants_xy.end(); ++it)
    {
        illuminants[it->first] = xyY2XYZ(it->second);
    }
    return illuminants;
}

const std::map<IO, std::vector<double> >  illuminants = getIlluminant();

} // namespace ccm
} // namespace cv


#endif