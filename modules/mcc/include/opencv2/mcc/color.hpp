// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __OPENCV_MCC_COLOR_HPP__
#define __OPENCV_MCC_COLOR_HPP__

#include <map>
#include "opencv2/mcc/colorspace.hpp"
#include "opencv2/mcc/distance.hpp"
#include "opencv2/mcc/utils.hpp"

namespace cv
{
namespace ccm
{

/* *\ brief Color defined by color_values and color space
*/

class Color
{
public:

    /* *\ param grays mask of grayscale color
       *\ param colored mask of colored color
       *\ param history storage of historical conversion
    */
    cv::Mat colors;
    const ColorSpace& cs;
    cv::Mat grays;
    cv::Mat colored;
    std::map<ColorSpace, std::shared_ptr<Color>> history;

    Color(cv::Mat colors_, const ColorSpace& cs_) :colors(colors_), cs(cs_) {};

    virtual ~Color() {};

    /* *\ brief Change to other color space.
       *        The conversion process incorporates linear transformations to speed up.
       *        method is chromatic adapation method.
       *        when save if True, get data from history first.
       *\ param other type of ColorSpace.
       *\ return Color.
    */
    Color to(const ColorSpace& other, CAM method = BRADFORD, bool save = true)
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

    /* *\ brief Channels split.
       *\ return each channel.
    */
    cv::Mat channel(cv::Mat m, int i)
    {
        cv::Mat dchannels[3];
        split(m, dchannels);
        return dchannels[i];
    }

    /* *\ brief To Gray.
    */
    cv::Mat toGray(IO io, CAM method = BRADFORD, bool save = true)
    {
        XYZ xyz(io);
        return channel(this->to(xyz, method, save).colors, 1);
    }

    /* *\ brief To Luminant.
    */
    cv::Mat toLuminant(IO io, CAM method = BRADFORD, bool save = true)
    {
        Lab lab(io);
        return channel(this->to(lab, method, save).colors, 0);
    }

    /* *\ brief Diff without IO.
       *\ param other type of Color.
       *\ param method type of distance.
       *\ return distance between self and other
    */
    cv::Mat diff(Color& other, DISTANCE_TYPE method = CIE2000)
    {
        return diff(other, cs.io, method);
    }

    /* *\ brief Diff with IO.
       *\ param other type of Color.
       *\ param io type of IO.
       *\ param method type of distance.
       *\ return distance between self and other
    */
    cv::Mat diff(Color& other, IO io, DISTANCE_TYPE method = CIE2000)
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
            throw std::invalid_argument { "Wrong method!" };
            break;
        }
    }

    /* *\ brief Calculate gray mask.
    */
    void getGray(double JDN = 2.0)
    {
        cv::Mat lab = to(Lab_D65_2).colors;
        cv::Mat gray(colors.size(), colors.type());
        int fromto[] = { 0,0, -1,1, -1,2 };
        mixChannels(&lab, 1, &gray, 1, fromto, 3);
        cv::Mat d = distance(lab, gray, CIE2000);
        this->grays = d < JDN;
        this->colored = ~grays;
    }

    /* *\ brief Operator for mask copy.
    */
    Color operator[](cv::Mat mask)
    {
        return Color(maskCopyTo(colors, mask), cs);
    }

    Color operator=(Color inp)
    {
        return inp;
    }
};


/* *\ brief Data is from https://www.imatest.com/wp-content/uploads/2011/11/Lab-data-Iluminate-D65-D50-spectro.xls
   *        see Miscellaneous.md for details.
*/
const cv::Mat ColorChecker2005_LAB_D50_2 = (cv::Mat_<cv::Vec3d>(24, 1) <<
    cv::Vec3d(37.986, 13.555, 14.059),
    cv::Vec3d(65.711, 18.13, 17.81),
    cv::Vec3d(49.927, -4.88, -21.925),
    cv::Vec3d(43.139, -13.095, 21.905),
    cv::Vec3d(55.112, 8.844, -25.399),
    cv::Vec3d(70.719, -33.397, -0.199),
    cv::Vec3d(62.661, 36.067, 57.096),
    cv::Vec3d(40.02, 10.41, -45.964),
    cv::Vec3d(51.124, 48.239, 16.248),
    cv::Vec3d(30.325, 22.976, -21.587),
    cv::Vec3d(72.532, -23.709, 57.255),
    cv::Vec3d(71.941, 19.363, 67.857),
    cv::Vec3d(28.778, 14.179, -50.297),
    cv::Vec3d(55.261, -38.342, 31.37),
    cv::Vec3d(42.101, 53.378, 28.19),
    cv::Vec3d(81.733, 4.039, 79.819),
    cv::Vec3d(51.935, 49.986, -14.574),
    cv::Vec3d(51.038, -28.631, -28.638),
    cv::Vec3d(96.539, -0.425, 1.186),
    cv::Vec3d(81.257, -0.638, -0.335),
    cv::Vec3d(66.766, -0.734, -0.504),
    cv::Vec3d(50.867, -0.153, -0.27),
    cv::Vec3d(35.656, -0.421, -1.231),
    cv::Vec3d(20.461, -0.079, -0.973));

const cv::Mat ColorChecker2005_LAB_D65_2 = (cv::Mat_<cv::Vec3d>(24, 1) <<
    cv::Vec3d(37.542, 12.018, 13.33),
    cv::Vec3d(65.2, 14.821, 17.545),
    cv::Vec3d(50.366, -1.573, -21.431),
    cv::Vec3d(43.125, -14.63, 22.12),
    cv::Vec3d(55.343, 11.449, -25.289),
    cv::Vec3d(71.36, -32.718, 1.636),
    cv::Vec3d(61.365, 32.885, 55.155),
    cv::Vec3d(40.712, 16.908, -45.085),
    cv::Vec3d(49.86, 45.934, 13.876),
    cv::Vec3d(30.15, 24.915, -22.606),
    cv::Vec3d(72.438, -27.464, 58.469),
    cv::Vec3d(70.916, 15.583, 66.543),
    cv::Vec3d(29.624, 21.425, -49.031),
    cv::Vec3d(55.643, -40.76, 33.274),
    cv::Vec3d(40.554, 49.972, 25.46),
    cv::Vec3d(80.982, -1.037, 80.03),
    cv::Vec3d(51.006, 49.876, -16.93),
    cv::Vec3d(52.121, -24.61, -26.176),
    cv::Vec3d(96.536, -0.694, 1.354),
    cv::Vec3d(81.274, -0.61, -0.24),
    cv::Vec3d(66.787, -0.647, -0.429),
    cv::Vec3d(50.872, -0.059, -0.247),
    cv::Vec3d(35.68, -0.22, -1.205),
    cv::Vec3d(20.475, 0.049, -0.972));

/* *\ brief  Macbeth ColorChecker with 2deg D50.
*/
const Color Macbeth_D50_2(ColorChecker2005_LAB_D50_2, Lab_D50_2);

/* *\ brief  Macbeth ColorChecker with 2deg D65.
*/
const Color Macbeth_D65_2(ColorChecker2005_LAB_D65_2, Lab_D65_2);

} // namespace ccm
} // namespace cv


#endif