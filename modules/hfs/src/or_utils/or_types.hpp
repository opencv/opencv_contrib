// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_OR_TYPES_HPP_
#define _OPENCV_OR_TYPES_HPP_

#include "or_vector.hpp"
#include "or_image.hpp"

//------------------------------------------------------
//
// math defines
//
//------------------------------------------------------

typedef cv::hfs::orutils::Vector2<int> Vector2i;
typedef cv::hfs::orutils::Vector2<float> Vector2f;

typedef cv::hfs::orutils::Vector4<float> Vector4f;
typedef cv::hfs::orutils::Vector4<int> Vector4i;
typedef cv::hfs::orutils::Vector4<unsigned char> Vector4u;

//------------------------------------------------------
//
// image defines
//
//------------------------------------------------------

typedef  cv::hfs::orutils::Image<int> IntImage;
typedef  cv::hfs::orutils::Image<unsigned char> UCharImage;
typedef  cv::hfs::orutils::Image<float> FloatImage;
typedef  cv::hfs::orutils::Image<Vector4f> Float4Image;
typedef  cv::hfs::orutils::Image<Vector4u> UChar4Image;

#endif
