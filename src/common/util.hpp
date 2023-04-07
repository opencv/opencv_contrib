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
/*!
 * Returns the OpenGL Version information.
 * @return a string object with the OpenGL version information
 */
std::string get_gl_info();
/*!
 * Returns the OpenCL Version information.
 * @return a string object with the OpenCL version information
 */
std::string get_cl_info();
/*!
 * Determines if Intel VAAPI is supported
 * @return true if it is supported
 */
bool is_intel_va_supported();
/*!
 * Determines if cl_khr_gl_sharing is supported
 * @return true if it is supported
 */
bool is_cl_gl_sharing_supported();
/*!
 * Pretty prints system information.
 */
void print_system_info();
/*!
 * Tells the application if it's alright to keep on running.
 * Note: If you use this mechanism signal handlers are installed
 * using #install_signal_handlers()
 * @return true if the program should keep on running
 */
bool keep_running();
/*!
 * Little helper program to keep track of FPS and optionally display it using NanoVG
 * @param v2d The Viz2D object to operate on
 * @param graphically if this parameter is true the FPS drawn on display
 */
void update_fps(cv::Ptr<Viz2D> viz2d, bool graphical);

#ifndef __EMSCRIPTEN__
/*!
 * Creates an Intel VAAPI enabled VideoWriter sink object to use in conjunction with #Viz2D::setSink().
 * Usually you would call #make_writer_sink() and let it automatically decide if VAAPI is available.
 * @param outputFilename The filename to write the video to.
 * @param fourcc    The fourcc code of the codec to use.
 * @param fps       The fps of the target video.
 * @param frameSize The frame size of the target video.
 * @param vaDeviceIndex The VAAPI device index to use.
 * @return A VAAPI enabled sink object.
 */
Sink make_va_sink(const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize, const int vaDeviceIndex);
/*!
 * Creates an Intel VAAPI enabled VideoCapture source object to use in conjunction with #Viz2D::setSource().
 * Usually you would call #make_capture_source() and let it automatically decide if VAAPI is available.
 * @param inputFilename The file to read from.
 * @param vaDeviceIndex The VAAPI device index to use.
 * @return A VAAPI enabled source object.
 */
Source make_va_source(const string& inputFilename, const int vaDeviceIndex);
/*!
 * Creates a VideoWriter sink object to use in conjunction with #Viz2D::setSink().
 * This function automatically determines if Intel VAAPI is available and enables it if so.
 * @param outputFilename The filename to write the video to.
 * @param fourcc    The fourcc code of the codec to use.
 * @param fps       The fps of the target video.
 * @param frameSize The frame size of the target video.
  * @return A (optionally VAAPI enabled) VideoWriter sink object.
 */
Sink make_writer_sink(const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize);
/*!
 * Creates a VideoCapture source object to use in conjunction with #Viz2D::setSource().
 * This function automatically determines if Intel VAAPI is available and enables it if so.
 * @param inputFilename The file to read from.
 * @return A (optionally VAAPI enabled) VideoCapture enabled source object.
 */
Source make_capture_source(const string& inputFilename);
#else
Source make_capture_source(int width, int height);
#endif

}
}

#endif /* SRC_COMMON_UTIL_HPP_ */
