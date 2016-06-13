/*
  By downloading, copying, installing or using the software you agree to this license.
  If you do not agree to this license, do not download, install,
  copy or use the software.


                            License Agreement
                 For Open Source Computer Vision Library
                         (3-clause BSD License)

  Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
  Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
  Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
  Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
  Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
  Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
  Third party copyrights are property of their respective owners.

  Redistribution and use in source and binary forms, with or without modification,
  are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of the copyright holders nor the names of the contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

  This software is provided by the copyright holders and contributors "as is" and
  any express or implied warranties, including, but not limited to, the implied
  warranties of merchantability and fitness for a particular purpose are disclaimed.
  In no event shall copyright holders or contributors be liable for any direct,
  indirect, incidental, special, exemplary, or consequential damages
  (including, but not limited to, procurement of substitute goods or services;
  loss of use, data, or profits; or business interruption) however caused
  and on any theory of liability, whether in contract, strict liability,
  or tort (including negligence or otherwise) arising in any way out of
  the use of this software, even if advised of the possibility of such damage.
 */

#include "precomp.hpp"

#define CNN_USE_CAFFE_CONVERTER
#include <tiny_cnn/tiny_cnn.h>

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;

namespace cv {
namespace dnn2 {

/*
 !CaffeConverter Implementation
 */
class CaffeConverter_Impl : public CaffeConverter {
 public:
    explicit CaffeConverter_Impl(const cv::String& model_file,
                                 const cv::String& trained_file,
                                 const cv::String& mean_file) {
        net_ = create_net_from_caffe_prototxt(model_file);
        reload_weight_from_caffe_protobinary(trained_file, net_.get());

        // int channels = (*net_)[0]->in_data_shape()[0].depth_;
        int width    = (*net_)[0]->in_data_shape()[0].width_;
        int height   = (*net_)[0]->in_data_shape()[0].height_;

        mean_ = compute_mean(mean_file, width, height);
    }

    ~CaffeConverter_Impl() {}

    virtual void eval(const cv::InputArray image, std::vector<float>* results);

 private:
    cv::Mat compute_mean(const string& mean_file, int width, int height);
    cv::ColorConversionCodes get_cvt_codes(int src_channels, int dst_channels);

    void preprocess(const cv::Mat& img, const cv::Mat& mean, int num_channels,
                    cv::Size geometry, vector<cv::Mat>* input_channels);

    cv::Mat mean_;
    std::shared_ptr<network<sequential>> net_;
};

cv::Mat
CaffeConverter_Impl::compute_mean(const string& mean_file,
                                  int width, int height) {
    caffe::BlobProto blob;
    ::detail::read_proto_from_binary(mean_file, &blob);

    vector<cv::Mat> channels;
    auto data = blob.mutable_data()->mutable_data();

    for (int i = 0; i < blob.channels(); i++, data += blob.height() * blob.width())
        channels.emplace_back(blob.height(), blob.width(), CV_32FC1, data);

    cv::Mat mean;
    cv::merge(channels, mean);

    return cv::Mat(cv::Size(width, height), mean.type(), cv::mean(mean));
}

cv::ColorConversionCodes
CaffeConverter_Impl::get_cvt_codes(int src_channels, int dst_channels) {
    assert(src_channels != dst_channels);

    if (dst_channels == 3) {
        return src_channels == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR;
    } else if (dst_channels == 1) {
        return src_channels == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY;
    } else {
        throw runtime_error("unsupported color code");
    }
}

void CaffeConverter_Impl::preprocess(const cv::Mat& img,
                                     const cv::Mat& mean,
                                     int num_channels,
                                     cv::Size geometry,
                                     vector<cv::Mat>* input_channels) {
    cv::Mat sample;

    // convert color
    if (img.channels() != num_channels) {
        cv::cvtColor(img, sample,
                     get_cvt_codes(img.channels(), num_channels));
    } else {
        sample = img;
    }

    // resize
    cv::Mat sample_resized;
    cv::resize(sample, sample_resized, geometry);

    cv::Mat sample_float;
    sample_resized.convertTo(sample_float, num_channels == 3 ? CV_32FC3 : CV_32FC1);

    // subtract mean
    if (mean.size().width > 0) {
        cv::Mat sample_normalized;
        cv::subtract(sample_float, mean, sample_normalized);
        cv::split(sample_normalized, *input_channels);
    }
    else {
        cv::split(sample_float, *input_channels);
    }
}

void CaffeConverter_Impl::eval(const cv::InputArray image,
                               std::vector<float>* results) {
    const cv::Mat img = image.getMat();

    // TODO: refactor this
    int channels = (*net_)[0]->in_data_shape()[0].depth_;
    int width    = (*net_)[0]->in_data_shape()[0].width_;
    int height   = (*net_)[0]->in_data_shape()[0].height_;

    vector<float> inputvec(width*height*channels);
    vector<cv::Mat> input_channels;

    for (int i = 0; i < channels; i++) {
        input_channels.emplace_back(height, width, CV_32FC1,
                                    &inputvec[width*height*i]);
    }

    preprocess(img, mean_, 3, cv::Size(width, height), &input_channels);

    vector<tiny_cnn::float_t> vec(inputvec.begin(), inputvec.end());

    auto result = net_->predict(vec);

    // allocate outpute
    results->clear();
    results->reserve(result.size());

    for (cnn_size_t i = 0; i < result.size(); i++) {
        results->push_back(result[i]);
    }
}

Ptr<CaffeConverter> CaffeConverter::create(const cv::String& model_file,
                                           const cv::String& trained_file,
                                           const cv::String& mean_file) {
    return makePtr<CaffeConverter_Impl>(
        model_file, trained_file, mean_file);
}

} // namespace dnn2
} // namespace cv
