// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds
#include "ean13_decoder.hpp"
#include "bardecode.hpp"

namespace cv {
namespace barcode {
void BarDecode::init(const cv::Mat &src, const std::vector<cv::Point2f> &points)
{
    //CV_TRACE_FUNCTION();
    original = src.clone();
    CV_Assert(!points.empty());
    CV_Assert((points.size() % 4) == 0);
    src_points.clear();
    for (size_t i = 0; i < points.size(); i += 4)
    {
        vector<Point2f> tempMat{points.cbegin() + i, points.cbegin() + i + 4};
        if (contourArea(tempMat) > 0.0)
        {
            src_points.push_back(tempMat);
        }
    }
    CV_Assert(!src_points.empty());
}

bool BarDecode::decodingProcess()
{
    std::unique_ptr<AbsDecoder> decoder(new Ean13Decoder);

    result_info = decoder->decodeImg(original, src_points);
    return !result_info.empty();
}

bool BarDecode::decodeMultiplyProcess()
{
    class ParallelBarCodeDecodeProcess : public ParallelLoopBody
    {
    public:
        ParallelBarCodeDecodeProcess(Mat &inarr_, vector<vector<Point2f>> &src_points_, vector<Result> &decoded_info_)
                : inarr(inarr_), decoded_info(decoded_info_), src_points(src_points_)
        {
            for (size_t i = 0; i < src_points.size(); ++i)
            {
                decoder.push_back(std::unique_ptr<AbsDecoder>(new Ean13Decoder()));
            }
        }

        void operator()(const Range &range) const CV_OVERRIDE
        {
            CV_Assert(inarr.channels() == 1);
            Mat gray = inarr.clone();
            for (int i = range.start; i < range.end; i++)
            {
                Mat bar_img;
                cutImage(gray, bar_img, src_points[i]);
                decoded_info[i] = decoder[i]->decodeImg(bar_img, src_points[i]);
            }
        }

    private:
        Mat &inarr;
        vector<Result> &decoded_info;
        vector<vector<Point2f> > &src_points;
        vector<std::unique_ptr<AbsDecoder>> decoder;
    };
    result_info.clear();
    result_info.resize(src_points.size());
    ParallelBarCodeDecodeProcess parallelDecodeProcess{original, src_points, result_info};
    parallel_for_(Range(0, int(src_points.size())), parallelDecodeProcess);
    return !result_info.empty();
}
}
}