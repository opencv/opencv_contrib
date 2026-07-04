#include "frame_encoder.hpp"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace cv {
namespace liveview {

FrameEncoder::FrameEncoder(int jpegQuality)
    : jpegQuality_(jpegQuality)
{
    if (jpegQuality_ < 1 || jpegQuality_ > 100)
        CV_Error(Error::StsBadArg, "JPEG quality must be in [1, 100]");
}

void FrameEncoder::validate(InputArray frame) const
{
    Mat src = frame.getMat();
    if (src.empty())
        CV_Error(Error::StsBadArg, "LiveView cannot publish an empty frame");

    const int type = src.type();
    if (type != CV_8UC1 && type != CV_8UC3 && type != CV_8UC4)
    {
        CV_Error(Error::StsUnsupportedFormat,
                 "LiveView Step 1 supports only CV_8UC1, CV_8UC3, and CV_8UC4 frames");
    }
}

std::vector<uchar> FrameEncoder::encodeJpeg(InputArray frame) const
{
    Mat src = frame.getMat();
    validate(src);

    Mat encodedInput;
    const int type = src.type();
    if (type == CV_8UC1 || type == CV_8UC3)
    {
        encodedInput = src;
    }
    else if (type == CV_8UC4)
    {
        cvtColor(src, encodedInput, COLOR_BGRA2BGR);
    }

    std::vector<uchar> out;
    std::vector<int> params;
    params.push_back(IMWRITE_JPEG_QUALITY);
    params.push_back(jpegQuality_);
    if (!imencode(".jpg", encodedInput, out, params) || out.empty())
        CV_Error(Error::StsError, "LiveView failed to encode frame as JPEG");

    return out;
}

} // namespace liveview
} // namespace cv
