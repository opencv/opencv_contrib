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

#ifndef __OPENCV_MCC_COLOR_HPP__
#define __OPENCV_MCC_COLOR_HPP__

#include <map>
#include "opencv2/mcc/colorspace.hpp"
#include "opencv2/mcc/distance.hpp"

namespace cv
{
namespace ccm
{

/** @brief Color defined by color_values and color space
*/

class CV_EXPORTS_W Color
{
public:

    /** @param grays mask of grayscale color
        @param colored mask of colored color
        @param history storage of historical conversion
    */
    Mat colors;
    const ColorSpace& cs;
    Mat grays;
    Mat colored;
    std::map<ColorSpace, std::shared_ptr<Color>> history;
    Color(Mat colors_, enum COLOR_SPACE cs_);
    Color(Mat colors_, enum COLOR_SPACE cs_, Mat colored_);
   // Color(double *colors_,int row,int col, enum COLOR_SPACE cs_);
    Color(Mat colors_, const ColorSpace& cs_, Mat colored_);
    Color(Mat colors_, const ColorSpace& cs_);

    virtual ~Color() {};

    /** @brief Change to other color space.
                 The conversion process incorporates linear transformations to speed up.
        @param other type of ColorSpace.
        @param  method the chromatic adapation method.
        @param save when save if True, get data from history first.
        @return Color.
    */
    Color to(COLOR_SPACE other, CAM method = BRADFORD, bool save = true);
    Color to(const ColorSpace& other, CAM method = BRADFORD, bool save = true);
    /** @brief Channels split.
       @return each channel.
    */
    Mat channel(Mat m, int i);

    /** @brief To Gray.
    */
    Mat toGray(IO io, CAM method = BRADFORD, bool save = true);

    /** @brief To Luminant.
    */
    Mat toLuminant(IO io, CAM method = BRADFORD, bool save = true);

    /** @brief Diff without IO.
        @param other type of Color.
        @param method type of distance.
        @return distance between self and other
    */
    Mat diff(Color& other, DISTANCE_TYPE method = CIE2000);

    /** @brief Diff with IO.
        @param other type of Color.
        @param io type of IO.
        @param method type of distance.
        @return distance between self and other
    */
    Mat diff(Color& other, IO io, DISTANCE_TYPE method = CIE2000);

    /** @brief Calculate gray mask.
    */
    void getGray(double JDN = 2.0);

    /** @brief Operator for mask copy.
    */
    Color operator[](Mat mask);

};


/** @brief  Macbeth and Vinyl ColorChecker with 2deg D50 .
*/
enum CONST_COLOR {
    Macbeth,
    Vinyl,
    DigitalSG
};
class CV_EXPORTS_W GetColor {
public:
    static Color get_color(CONST_COLOR const_color);
    static double create();
    static Mat get_ColorChecker(const double *checker,int row);
    static Mat get_ColorChecker_MASK(const uchar *checker,int row);
};



} // namespace ccm
} // namespace cv


#endif