// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_UTIL_HPP_
#define SRC_OPENCV_V4D_UTIL_HPP_

#include "source.hpp"
#include "sink.hpp"

#include <string>
#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#  include <emscripten/bind.h>
#  include <fstream>
#endif

namespace cv {
namespace viz {
using std::string;
class V4D;
/*!
 * Returns the OpenGL Version information.
 * @return a string object with the OpenGL version information
 */
CV_EXPORTS std::string getGlInfo();
/*!
 * Returns the OpenCL Version information.
 * @return a string object with the OpenCL version information
 */
CV_EXPORTS std::string getClInfo();
/*!
 * Determines if Intel VAAPI is supported
 * @return true if it is supported
 */
CV_EXPORTS bool isIntelVaSupported();
/*!
 * Determines if cl_khr_gl_sharing is supported
 * @return true if it is supported
 */
CV_EXPORTS bool isClGlSharingSupported();
/*!
 * Pretty prints system information.
 */
CV_EXPORTS void printSystemInfo();
/*!
 * Tells the application if it's alright to keep on running.
 * Note: If you use this mechanism signal handlers are installed
 * @return true if the program should keep on running
 */
CV_EXPORTS bool keepRunning();

/*!
 * Little helper function to keep track of FPS and optionally display it using NanoVG
 * @param v4d The V4D object to operate on
 * @param graphical if this parameter is true the FPS drawn on display
 */
CV_EXPORTS void updateFps(cv::Ptr<V4D> v4d, bool graphical);

#ifndef __EMSCRIPTEN__
/*!
 * Creates an Intel VAAPI enabled VideoWriter sink object to use in conjunction with #V4D::setSink().
 * Usually you would call #makeWriterSink() and let it automatically decide if VAAPI is available.
 * @param outputFilename The filename to write the video to.
 * @param fourcc    The fourcc code of the codec to use.
 * @param fps       The fps of the target video.
 * @param frameSize The frame size of the target video.
 * @param vaDeviceIndex The VAAPI device index to use.
 * @return A VAAPI enabled sink object.
 */
CV_EXPORTS Sink makeVaSink(const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize, const int vaDeviceIndex);
/*!
 * Creates an Intel VAAPI enabled VideoCapture source object to use in conjunction with #V4D::setSource().
 * Usually you would call #makeCaptureSource() and let it automatically decide if VAAPI is available.
 * @param inputFilename The file to read from.
 * @param vaDeviceIndex The VAAPI device index to use.
 * @return A VAAPI enabled source object.
 */
CV_EXPORTS Source makeVaSource(const string& inputFilename, const int vaDeviceIndex);
/*!
 * Creates a VideoWriter sink object to use in conjunction with #V4D::setSink().
 * This function automatically determines if Intel VAAPI is available and enables it if so.
 * @param outputFilename The filename to write the video to.
 * @param fourcc    The fourcc code of the codec to use.
 * @param fps       The fps of the target video.
 * @param frameSize The frame size of the target video.
  * @return A (optionally VAAPI enabled) VideoWriter sink object.
 */
CV_EXPORTS Sink makeWriterSink(const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize);
/*!
 * Creates a VideoCapture source object to use in conjunction with #V4D::setSource().
 * This function automatically determines if Intel VAAPI is available and enables it if so.
 * @param inputFilename The file to read from.
 * @return A (optionally VAAPI enabled) VideoCapture enabled source object.
 */
CV_EXPORTS Source makeCaptureSource(const string& inputFilename);
#else
/*!
 * Creates a WebCam source object to use in conjunction with #V4D::setSource().
 * In the background it uses emscripten's file system implementation to transfer frames from the camera to the source object
 * @param width The frame width to capture (usually the initial width of the V4D object)
 * @param height The frame height to capture (usually the initial height of the V4D object)
 * @return A WebCam source object.
 */
CV_EXPORTS Source makeCaptureSource(int width, int height);
#endif

}
}

#endif /* SRC_OPENCV_V4D_UTIL_HPP_ */
