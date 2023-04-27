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
#  include <emscripten/threading.h>
#  include <fstream>
#endif
#include <unistd.h>
#include <thread>
#include <chrono>
#include <mutex>
#include <functional>
#include <iostream>
#include <cmath>

namespace cv {
namespace v4d {
namespace detail {


template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
struct fun_ptr_helper
{
public:
    typedef std::function<_Res(_ArgTypes...)> function_type;

    static void bind(function_type&& f)
    { instance().fn_.swap(f); }

    static void bind(const function_type& f)
    { instance().fn_=f; }

    static _Res invoke(_ArgTypes... args)
    { return instance().fn_(args...); }

    typedef decltype(&fun_ptr_helper::invoke) pointer_type;
    static pointer_type ptr()
    { return &invoke; }

private:
    static fun_ptr_helper& instance()
    {
        static fun_ptr_helper inst_;
        return inst_;
    }

    fun_ptr_helper() {}

    function_type fn_;
};

template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
typename fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::pointer_type
get_fn_ptr(const std::function<_Res(_ArgTypes...)>& f)
{
    fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::bind(f);
    return fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::ptr();
}

template<typename T>
std::function<typename std::enable_if<std::is_function<T>::value, T>::type>
make_function(T *t)
{
    return {t};
}

long proxy_to_mainl(std::function<long()> fn);
void proxy_to_mainv(std::function<void()> fn);
bool proxy_to_mainb(std::function<bool()> fn);
}
using std::string;
class V4D;
#ifdef __EMSCRIPTEN__
CV_EXPORTS Mat read_image(const string &path);
#endif

CV_EXPORTS unsigned int init_shader(const char* vShader, const char* fShader, const char* outputAttributeName);
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
