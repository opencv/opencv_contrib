// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_COMMON_UTIL_HPP_
#define SRC_COMMON_UTIL_HPP_

#include "source.hpp"
#include "sink.hpp"

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#  include <emscripten/bind.h>
#  include <fstream>
#endif

namespace cv {
namespace viz {
using std::string;
class Viz2D;
std::string get_gl_info();
std::string get_cl_info();
bool is_intel_va_supported();
bool is_cl_gl_sharing_supported();
bool keep_running();
void print_system_info();
void update_fps(cv::Ptr<Viz2D> viz2d, bool graphical);

#ifndef __EMSCRIPTEN__
Sink make_va_sink(const string &outputFilename, const int fourcc, const float fps, const cv::Size &frameSize, const int vaDeviceIndex);
Source make_va_source(const string &inputFilename, const int vaDeviceIndex);
Sink make_writer_sink(const string &outputFilename, const int fourcc, const float fps, const cv::Size &frameSize);
Source make_capture_source(const string &inputFilename);
#else
Source make_capture_source(int width, int height);
#endif

}
}

#endif /* SRC_COMMON_UTIL_HPP_ */
