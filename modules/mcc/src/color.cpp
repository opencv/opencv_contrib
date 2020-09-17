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

#include "precomp.hpp"

namespace cv
{
namespace ccm
{

Color::Color(Mat colors_, const ColorSpace& cs_, Mat colored_) : colors(colors_), cs(cs_), colored(colored_)
{
    grays= ~colored;
}

Color::Color(Mat colors_, const ColorSpace& cs_) : colors(colors_), cs(cs_) {};

Color Color::to(const ColorSpace& other, CAM method, bool save)
{
    if (history.count(other) == 1)
    {
        return *history[other];
    }
    if (cs.relate(other))
    {
        return Color(cs.relation(other).run(colors), other);
    }
    Operations ops;
    ops.add(cs.to).add(XYZ(cs.io).cam(other.io, method)).add(other.from);
    std::shared_ptr<Color> color(new Color(ops.run(colors), other));
    if (save)
    {
        history[other] = color;
    }
    return *color;
}

Mat Color::channel(Mat m, int i)
{
    Mat dchannels[3];
    split(m, dchannels);
    return dchannels[i];
}

Mat Color::toGray(IO io, CAM method, bool save)
{
    XYZ xyz(io);
    return channel(this->to(xyz, method, save).colors, 1);
}

Mat Color::toLuminant(IO io, CAM method, bool save)
{
    Lab lab(io);
    return channel(this->to(lab, method, save).colors, 0);
}

Mat Color::diff(Color& other, DISTANCE_TYPE method)
{
    return diff(other, cs.io, method);
}

Mat Color::diff(Color& other, IO io, DISTANCE_TYPE method)
{
    Lab lab(io);
    switch (method)
    {
    case cv::ccm::CIE76:
    case cv::ccm::CIE94_GRAPHIC_ARTS:
    case cv::ccm::CIE94_TEXTILES:
    case cv::ccm::CIE2000:
    case cv::ccm::CMC_1TO1:
    case cv::ccm::CMC_2TO1:
        return distance(to(lab).colors, other.to(lab).colors, method);
    case cv::ccm::RGB:
        return distance(to(*cs.nl).colors, other.to(*cs.nl).colors, method);
    case cv::ccm::RGBL:
        return distance(to(*cs.l).colors, other.to(*cs.l).colors, method);
    default:
        throw std::invalid_argument{ "Wrong method!" };
        break;
    }
}

void Color::getGray(double JDN)
{
    if (!grays.empty())
    {
        return;
    }
    Mat lab = to(Lab_D65_2).colors;
    Mat gray(colors.size(), colors.type());
    int fromto[] = { 0,0, -1,1, -1,2 };
    mixChannels(&lab, 1, &gray, 1, fromto, 3);
    Mat d = distance(lab, gray, CIE2000);
    this->grays = d < JDN;
    this->colored = ~grays;
}

Color Color::operator[](Mat mask)
{
    return Color(maskCopyTo(colors, mask), cs);
}

Color Macbeth_D50_2(ColorChecker2005_LAB_D50_2, Lab_D50_2, ColorChecker2005_COLORED_MASK);
Color Macbeth_D65_2(ColorChecker2005_LAB_D65_2, Lab_D65_2, ColorChecker2005_COLORED_MASK);
Color Vinyl_D50_2(Vinyl_LAB_D50_2, Lab_D50_2, Vinyl_COLORED_MASK);

} // namespace ccm
} // namespace cv