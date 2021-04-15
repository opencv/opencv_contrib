// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#include "precomp.hpp"
#include <opencv2/barcode.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "decoder/ean13_decoder.hpp"
#include "decoder/ean8_decoder.hpp"
#include "detector/bardetect.hpp"
#include "decoder/common/super_scale.hpp"
#include "decoder/common/utils.hpp"

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

inline const std::array<std::shared_ptr<AbsDecoder>, 2> &getDecoders()
{
    //indicate Decoder
    static const std::array<std::shared_ptr<AbsDecoder>, 2> decoders{
            std::shared_ptr<AbsDecoder>(new Ean13Decoder()), std::shared_ptr<AbsDecoder>(new Ean8Decoder())};
    return decoders;
}

class BarDecode
{
public:
    void init(const vector<Mat> &bar_imgs_);

    const vector<Result> &getDecodeInformation()
    { return result_info; }

    bool decodeMultiplyProcess();

private:
    vector<Mat> bar_imgs;
    vector<Result> result_info;
};

void BarDecode::init(const vector<Mat> &bar_imgs_)
{
    bar_imgs = bar_imgs_;
}

bool BarDecode::decodeMultiplyProcess()
{
    class ParallelBarCodeDecodeProcess : public ParallelLoopBody
    {
    public:
        ParallelBarCodeDecodeProcess(vector<Mat> &bar_imgs_, vector<Result> &decoded_info_) : bar_imgs(bar_imgs_),
                                                                                              decoded_info(
                                                                                                      decoded_info_)
        {}

        void operator()(const Range &range) const CV_OVERRIDE
        {
            for (int i = range.start; i < range.end; i++)
            {
                Mat bin_bar;
                Result max_res;
                float max_conf = -1.f;
                bool decoded = false;
                for (const auto &decoder:getDecoders())
                {
                    if (decoded)
                    { break; }
                    for (const auto binary_type : binary_types)
                    {
                        binarize(bar_imgs[i], bin_bar, binary_type);
                        auto cur_res = decoder->decodeROI(bin_bar);
                        if (cur_res.second > max_conf)
                        {
                            max_res = cur_res.first;
                            max_conf = cur_res.second;
                            if (max_conf > 0.6)
                            {
                                // code decoded
                                decoded = true;
                                break;
                            }
                        }

                    }
                }

                decoded_info[i] = max_res;
            }
        }

    private:
        vector<Mat> bar_imgs;
        vector<Result> &decoded_info;
    };
    result_info.clear();
    result_info.resize(bar_imgs.size());

    ParallelBarCodeDecodeProcess parallelDecodeProcess{bar_imgs, result_info};
    parallel_for_(Range(0, int(bar_imgs.size())), parallelDecodeProcess);
    return !result_info.empty();
}


struct BarcodeDetector::Impl
{
public:
    Impl() = default;;

    ~Impl() = default;;

    vector<Mat> initDecode(const Mat &src, const vector<Point2f> &points) const;

    std::shared_ptr<SuperScale> sr;
    bool use_nn_sr = false;
};

// return cropped and scaled bar img
vector<Mat> BarcodeDetector::Impl::initDecode(const Mat &src, const vector<Point2f> &points) const
{
    vector<vector<Point2f>> src_points;
    //CV_TRACE_FUNCTION();
    CV_Assert(!points.empty());
    CV_Assert((points.size() % 4) == 0);
    src_points.clear();
    for (int i = 0; (uint) i < points.size(); i += 4)
    {
        vector<Point2f> tempMat{points.cbegin() + i, points.cbegin() + i + 4};
        if (contourArea(tempMat) > 0.0)
        {
            src_points.push_back(tempMat);
        }
    }
    CV_Assert(!src_points.empty());
    vector<Mat> bar_imgs;
    for (auto &corners : src_points)
    {
        Mat bar_img;
        cropROI(src, bar_img, corners);
//        sharpen(bar_img, bar_img);
        // empirical settings
        if (bar_img.cols < 320 || bar_img.cols > 640)
        {
            float scale = 560.0f / static_cast<float>(bar_img.cols);
            sr->processImageScale(bar_img, bar_img, scale, use_nn_sr);
        }
        bar_imgs.emplace_back(bar_img);
    }
    return bar_imgs;

}

BarcodeDetector::BarcodeDetector(const string &prototxt_path, const string &model_path) : p(new Impl)
{
    if (!prototxt_path.empty() && !model_path.empty())
    {
        CV_Assert(utils::fs::exists(prototxt_path));
        CV_Assert(utils::fs::exists(model_path));
        p->sr = std::make_shared<SuperScale>();
        int res = p->sr->init(prototxt_path, model_path);
        CV_Assert(res == 0);
        p->use_nn_sr = true;
    }
}


BarcodeDetector::~BarcodeDetector() = default;

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
    vector<Mat> bar_imgs = p->initDecode(inarr, src_points);
    BarDecode bardec;
    bardec.init(bar_imgs);
    bardec.decodeMultiplyProcess();
    const vector<Result> info = bardec.getDecodeInformation();
    decoded_info.clear();
    decoded_type.clear();
    bool ok = false;
    for (const auto &res : info)
    {
        if (res.format != NONE)
        {
            ok = true;
        }

        decoded_info.emplace_back(res.result);
        decoded_type.emplace_back(res.format);
    }
    return ok;
}

bool
BarcodeDetector::detectAndDecode(InputArray img, vector<std::string> &decoded_info, vector<BarcodeType> &decoded_type,
                                 OutputArray points_) const
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
