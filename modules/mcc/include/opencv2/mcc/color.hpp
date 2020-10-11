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


/** @brief Data is from https://www.imatest.com/wp-content/uploads/2011/11/Lab-data-Iluminate-D65-D50-spectro.xls
           see Miscellaneous.md for details.
*/
const double ColorChecker2005_LAB_D50_2 [24][3] =
   { {37.986, 13.555, 14.059},
    {65.711, 18.13, 17.81},
    {49.927, -4.88, -21.925},
    {43.139, -13.095, 21.905},
    {55.112, 8.844, -25.399},
    {70.719, -33.397, -0.199},
    {62.661, 36.067, 57.096},
    {40.02, 10.41, -45.964},
    {51.124, 48.239, 16.248},
    {30.325, 22.976, -21.587},
    {72.532, -23.709, 57.255},
    {71.941, 19.363, 67.857},
    {28.778, 14.179, -50.297},
    {55.261, -38.342, 31.37},
    {42.101, 53.378, 28.19},
    {81.733, 4.039, 79.819},
    {51.935, 49.986, -14.574},
    {51.038, -28.631, -28.638},
    {96.539, -0.425, 1.186},
    {81.257, -0.638, -0.335},
    {66.766, -0.734, -0.504},
    {50.867, -0.153, -0.27},
    {35.656, -0.421, -1.231},
    {20.461, -0.079, -0.973}};

const double ColorChecker2005_LAB_D65_2[24][3]=
    {{37.542, 12.018, 13.33},
    {65.2, 14.821, 17.545},
    {50.366, -1.573, -21.431},
    {43.125, -14.63, 22.12},
    {55.343, 11.449, -25.289},
    {71.36, -32.718, 1.636},
    {61.365, 32.885, 55.155},
    {40.712, 16.908, -45.085},
    {49.86, 45.934, 13.876},
    {30.15, 24.915, -22.606},
    {72.438, -27.464, 58.469},
    {70.916, 15.583, 66.543},
    {29.624, 21.425, -49.031},
    {55.643, -40.76, 33.274},
    {40.554, 49.972, 25.46},
    {80.982, -1.037, 80.03},
    {51.006, 49.876, -16.93},
    {52.121, -24.61, -26.176},
    {96.536, -0.694, 1.354},
    {81.274, -0.61, -0.24},
    {66.787, -0.647, -0.429},
    {50.872, -0.059, -0.247},
    {35.68, -0.22, -1.205},
    {20.475, 0.049, -0.972}};

const uchar ColorChecker2005_COLORED_MASK[24] =
    {1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0};
const double Vinyl_LAB_D50_2[18][3] =
   { {100, 0.00520000001, -0.0104},
    {73.0833969, -0.819999993, -2.02099991},
    {62.493, 0.425999999, -2.23099995},
    {50.4640007, 0.446999997, -2.32399988},
    {37.7970009, 0.0359999985, -1.29700005},
    {0, 0, 0},
    {51.5880013, 73.5179977, 51.5690002},
    {93.6989975, -15.7340002, 91.9420013},
    {69.4079971, -46.5940018, 50.4869995},
    {66.61000060000001, -13.6789999, -43.1720009},
    {11.7110004, 16.9799995, -37.1759987},
    {51.973999, 81.9440002, -8.40699959},
    {40.5489998, 50.4399986, 24.8490009},
    {60.8160019, 26.0690002, 49.4420013},
    {52.2529984, -19.9500008, -23.9960003},
    {51.2859993, 48.4700012, -15.0579996},
    {68.70700069999999, 12.2959995, 16.2129993},
    {63.6839981, 10.2930002, 16.7639999}};
const uchar Vinyl_COLORED_MASK[18]=
   { 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1};
const double DigitalSG_LAB_D50_2[140][3] =
   { {96.55,-0.91,0.57},
    {6.43,-0.06,-0.41},
    {49.7,-0.18,0.03},
    {96.5,-0.89,0.59},
    {6.5,-0.06,-0.44},
    {49.66,-0.2,0.01},
    {96.52,-0.91,0.58},
    {6.49,-0.02,-0.28},
    {49.72,-0.2,0.04},
    {96.43,-0.91,0.67},
    {49.72,-0.19,0},
    {32.6,51.58,-10.85},
    {60.75,26.22,-18.6},
    {28.69,48.28,-39},
    {49.38,-15.43,-48.48},
    {60.63,-30.77,-26.23},
    {19.29,-26.37,-6.15},
    {60.15,-41.77,-12.6},
    {21.42,1.67,8.79},
    {49.69,-0.2,0.01},
    {6.5,-0.03,-0.67},
    {21.82,17.33,-18.35},
    {41.53,18.48,-37.26},
    {19.99,-0.16,-36.29},
    {60.16,-18.45,-31.42},
    {19.94,-17.92,-20.96},
    {60.68,-6.05,-32.81},
    {50.81,-49.8,-9.63},
    {60.65,-39.77,20.76},
    {6.53,-0.03,-0.43},
    {96.56,-0.91,0.59},
    {84.19,-1.95,-8.23},
    {84.75,14.55,0.23},
    {84.87,-19.07,-0.82},
    {85.15,13.48,6.82},
    {84.17,-10.45,26.78},
    {61.74,31.06,36.42},
    {64.37,20.82,18.92},
    {50.4,-53.22,14.62},
    {96.51,-0.89,0.65},
    {49.74,-0.19,0.03},
    {31.91,18.62,21.99},
    {60.74,38.66,70.97},
    {19.35,22.23,-58.86},
    {96.52,-0.91,0.62},
    {6.66,0,-0.3},
    {76.51,20.81,22.72},
    {72.79,29.15,24.18},
    {22.33,-20.7,5.75},
    {49.7,-0.19,0.01},
    {6.53,-0.05,-0.61},
    {63.42,20.19,19.22},
    {34.94,11.64,-50.7},
    {52.03,-44.15,39.04},
    {79.43,0.29,-0.17},
    {30.67,-0.14,-0.53},
    {63.6,14.44,26.07},
    {64.37,14.5,17.05},
    {60.01,-44.33,8.49},
    {6.63,-0.01,-0.47},
    {96.56,-0.93,0.59},
    {46.37,-5.09,-24.46},
    {47.08,52.97,20.49},
    {36.04,64.92,38.51},
    {65.05,0,-0.32},
    {40.14,-0.19,-0.38},
    {43.77,16.46,27.12},
    {64.39,17,16.59},
    {60.79,-29.74,41.5},
    {96.48,-0.89,0.64},
    {49.75,-0.21,0.01},
    {38.18,-16.99,30.87},
    {21.31,29.14,-27.51},
    {80.57,3.85,89.61},
    {49.71,-0.2,0.03},
    {60.27,0.08,-0.41},
    {67.34,14.45,16.9},
    {64.69,16.95,18.57},
    {51.12,-49.31,44.41},
    {49.7,-0.2,0.02},
    {6.67,-0.05,-0.64},
    {51.56,9.16,-26.88},
    {70.83,-24.26,64.77},
    {48.06,55.33,-15.61},
    {35.26,-0.09,-0.24},
    {75.16,0.25,-0.2},
    {44.54,26.27,38.93},
    {35.91,16.59,26.46},
    {61.49,-52.73,47.3},
    {6.59,-0.05,-0.5},
    {96.58,-0.9,0.61},
    {68.93,-34.58,-0.34},
    {69.65,20.09,78.57},
    {47.79,-33.18,-30.21},
    {15.94,-0.42,-1.2},
    {89.02,-0.36,-0.48},
    {63.43,25.44,26.25},
    {65.75,22.06,27.82},
    {61.47,17.1,50.72},
    {96.53,-0.89,0.66},
    {49.79,-0.2,0.03},
    {85.17,10.89,17.26},
    {89.74,-16.52,6.19},
    {84.55,5.07,-6.12},
    {84.02,-13.87,-8.72},
    {70.76,0.07,-0.35},
    {45.59,-0.05,0.23},
    {20.3,0.07,-0.32},
    {61.79,-13.41,55.42},
    {49.72,-0.19,0.02},
    {6.77,-0.05,-0.44},
    {21.85,34.37,7.83},
    {42.66,67.43,48.42},
    {60.33,36.56,3.56},
    {61.22,36.61,17.32},
    {62.07,52.8,77.14},
    {72.42,-9.82,89.66},
    {62.03,3.53,57.01},
    {71.95,-27.34,73.69},
    {6.59,-0.04,-0.45},
    {49.77,-0.19,0.04},
    {41.84,62.05,10.01},
    {19.78,29.16,-7.85},
    {39.56,65.98,33.71},
    {52.39,68.33,47.84},
    {81.23,24.12,87.51},
    {81.8,6.78,95.75},
    {71.72,-16.23,76.28},
    {20.31,14.45,16.74},
    {49.68,-0.19,0.05},
    {96.48,-0.88,0.68},
    {49.69,-0.18,0.03},
    {6.39,-0.04,-0.33},
    {96.54,-0.9,0.67},
    {49.72,-0.18,0.05},
    {6.49,-0.03,-0.41},
    {96.51,-0.9,0.69},
    {49.7,-0.19,0.07},
    {6.47,0,-0.38},
    {96.46,-0.89,0.7}};
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
    static Mat get_ColorChecker(const double *checker,int row);
    static Mat get_ColorChecker_MASK(const uchar *checker,int row);
};

} // namespace ccm
} // namespace cv


#endif