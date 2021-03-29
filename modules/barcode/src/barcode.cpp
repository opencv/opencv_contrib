// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds


#include "precomp.hpp"
#include <opencv2/barcode.hpp>
#include "decoder/ean13_decoder.hpp"
#include "detector/bardetect.hpp"

namespace cv {
namespace barcode {

static bool checkBarInputImage(InputArray img, Mat &gray)
{
    CV_Assert(!img.empty());
    CV_CheckDepthEQ(img.depth(), CV_8U, "");
    if (img.cols() <= 40 || img.rows() <= 40)
    {
        return false; // image data is not enough for providing reliable results
    }
    int incn = img.channels();
    CV_Check(incn, incn == 1 || incn == 3 || incn == 4, "");
    if (incn == 3 || incn == 4)
    {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = img.getMat();
    }
    return true;
}

static void updatePointsResult(OutputArray points_, const vector<Point2f> &points)
{
    if (points_.needed())
    {
        int N = int(points.size() / 4);
        if (N > 0)
        {
            Mat m_p(N, 4, CV_32FC2, (void *) &points[0]);
            int points_type = points_.fixedType() ? points_.type() : CV_32FC2;
            m_p.reshape(2, points_.rows()).convertTo(points_, points_type);  // Mat layout: N x 4 x 2cn
        }
        else
        {
            points_.release();
        }
    }
}

class BarDecode
{
public:
    void init(const Mat &src, const vector<Point2f> &points);

    const vector<Result> &getDecodeInformation()
    { return result_info; }

    bool decodeMultiplyProcess();

private:
    vector<vector<Point2f>> src_points;
    Mat original;
    vector<Result> result_info;
};

void BarDecode::init(const cv::Mat &src, const std::vector <cv::Point2f> &points) {
    //CV_TRACE_FUNCTION();
    original = src.clone();
    CV_Assert(!points.empty());
    CV_Assert((points.size() % 4) == 0);
    src_points.clear();
    for (size_t i = 0; i < points.size(); i += 4) {
        vector <Point2f> tempMat{points.cbegin() + i, points.cbegin() + i + 4};
        if (contourArea(tempMat) > 0.0) {
            src_points.push_back(tempMat);
        }
    }
    CV_Assert(!src_points.empty());
}

bool BarDecode::decodeMultiplyProcess() {
    class ParallelBarCodeDecodeProcess : public ParallelLoopBody {
    public:
        ParallelBarCodeDecodeProcess(Mat &inarr_, vector <vector<Point2f>> &src_points_,
                                     vector <Result> &decoded_info_)
                : inarr(inarr_), decoded_info(decoded_info_), src_points(src_points_) {
            for (size_t i = 0; i < src_points.size(); ++i) {
                decoder.push_back(std::unique_ptr<AbsDecoder>(new Ean13Decoder()));
            }
        }

        void operator()(const Range &range) const

        CV_OVERRIDE
        {
            CV_Assert(inarr.channels() == 1);
            Mat gray = inarr.clone();
            for (int i = range.start; i < range.end; i++) {
                Mat bar_img;
                cutImage(gray, bar_img, src_points[i]);
                decoded_info[i] = decoder[i]->decodeImg(bar_img, src_points[i]);
            }
        }

    private:
        Mat &inarr;
        vector <Result> &decoded_info;
        vector <vector<Point2f>> &src_points;
        vector <std::unique_ptr<AbsDecoder>> decoder;
    };
    result_info.clear();
    result_info.resize(src_points.size());
    ParallelBarCodeDecodeProcess parallelDecodeProcess{original, src_points, result_info};
    parallel_for_(Range(0, int(src_points.size())), parallelDecodeProcess);
    return !result_info.empty();
}

//struct BarcodeDetector::Impl
//{
//public:
//    Impl() {};
//
//    ~Impl() {};
//};

BarcodeDetector::BarcodeDetector() {};

BarcodeDetector::~BarcodeDetector() {};

bool BarcodeDetector::detect(InputArray img, OutputArray points) const
{
    Mat inarr;
    if (!checkBarInputImage(img, inarr))
    {
        points.release();
        return false;
    }

    Detect bardet;
    bardet.init(inarr);
    bardet.localization();
    if (!bardet.computeTransformationPoints())
    { return false; }
    vector<vector<Point2f>> pnts2f = bardet.getTransformationPoints();
    vector<Point2f> trans_points;
    for (auto &i : pnts2f)
    {
        for (const auto &j : i)
        {
            trans_points.push_back(j);
        }
    }

    updatePointsResult(points, trans_points);
    return true;
}

bool BarcodeDetector::decode(InputArray img, InputArray points, vector<std::string> &decoded_info,
                             vector<BarcodeType> &decoded_type) const
{
    Mat inarr;
    if (!checkBarInputImage(img, inarr))
    {
        return false;
    }
    CV_Assert(points.size().width > 0);
    CV_Assert((points.size().width % 4) == 0);
    vector<Point2f> src_points;
    points.copyTo(src_points);
    BarDecode bardec;
    bardec.init(img.getMat(), src_points);
    bool ok = bardec.decodeMultiplyProcess();
    const vector<Result> &_decoded_info = bardec.getDecodeInformation();
    decoded_info.clear();
    decoded_type.clear();
    for (const auto &info : _decoded_info)
    {
        decoded_info.emplace_back(info.result);
        decoded_type.emplace_back(info.format);
    }
    return ok;
}

bool BarcodeDetector::detectAndDecode(InputArray img, vector<std::string> &decoded_info,
                                      vector<BarcodeType> &decoded_type, OutputArray points_) const
{
    Mat inarr;
    if (!checkBarInputImage(img, inarr))
    {
        points_.release();
        return false;
    }
    vector<Point2f> points;
    bool ok = this->detect(img, points);
    if (!ok)
    {
        points_.release();
        return false;
    }
    updatePointsResult(points_, points);
    decoded_info.clear();
    decoded_type.clear();
    ok = this->decode(inarr, points, decoded_info, decoded_type);
    return ok;
}

}// namespace barcode
} // namespace cv
