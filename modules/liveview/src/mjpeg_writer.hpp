#ifndef OPENCV_LIVEVIEW_MJPEG_WRITER_HPP
#define OPENCV_LIVEVIEW_MJPEG_WRITER_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace liveview {

String httpStatusText(int status);
String httpHeader(int status, const String& contentType, size_t contentLength);
String mjpegStreamHeader(const String& boundary);
String mjpegPartHeader(const String& boundary, size_t jpegSize);

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_MJPEG_WRITER_HPP
