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
#include <opencv2/imgproc.hpp>

#include <tiny_dnn/tiny_dnn.h>
#include <tiny_dnn/io/caffe/caffe.pb.cc>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

namespace cv {
namespace dnn2 {

/*
 !CaffeConverter Implementation
 */
class CaffeConverter_Impl : public CaffeConverter {
 public:
    explicit CaffeConverter_Impl(const String& model_file,
                                 const String& trained_file,
                                 const String& mean_file) {
        net_ = create_net_from_caffe_prototxt(model_file);
        reload_weight_from_caffe_protobinary(trained_file, net_.get());

        const size_t width  = (*net_)[0]->in_data_shape()[0].width_;
        const size_t height = (*net_)[0]->in_data_shape()[0].height_;

        mean_ = compute_mean(mean_file, width, height);
    }

    ~CaffeConverter_Impl() {}

    virtual void eval(InputArray image, std::vector<float>& results);

 private:
    Mat compute_mean(const string& mean_file, const size_t width,
		         const size_t height);

    ColorConversionCodes get_cvt_codes(const int src_channels,
                                           const int dst_channels);

    void preprocess(const Mat& img, const Mat& mean,
            const int num_channels, const Size& geometry,
            vector<Mat>* input_channels);

    Mat mean_;
    std::shared_ptr<network<sequential>> net_;
};

Mat
CaffeConverter_Impl::compute_mean(const string& mean_file,
                                  const size_t width,
				  const size_t height) {
    caffe::BlobProto blob;
    ::detail::read_proto_from_binary(mean_file, &blob);

    vector<Mat> channels;
    auto data = blob.mutable_data()->mutable_data();

    const size_t offset = blob.height() * blob.width();

    for (int i = 0; i < blob.channels(); i++, data += offset) {
        channels.emplace_back(blob.height(), blob.width(), CV_32FC1, data);
    }

    Mat meanChannel;
    merge(channels, meanChannel);

    return Mat(Size(width, height), meanChannel.type(), mean(meanChannel));
}

ColorConversionCodes
CaffeConverter_Impl::get_cvt_codes(const int src_channels,
                                   const int dst_channels) {
    assert(src_channels != dst_channels);

    if (dst_channels == 3) {
        return src_channels == 1 ? COLOR_GRAY2BGR : COLOR_BGRA2BGR;
    } else if (dst_channels == 1) {
        return src_channels == 3 ? COLOR_BGR2GRAY : COLOR_BGRA2GRAY;
    } else {
        throw runtime_error("unsupported color code");
    }
}

void CaffeConverter_Impl::preprocess(const Mat& img,
                                     const Mat& mean,
                                     const int num_channels,
                                     const Size& geometry,
                                     vector<Mat>* input_channels) {
    Mat sample;

    // convert color
    if (img.channels() != num_channels) {
        cvtColor(img, sample,
                     get_cvt_codes(img.channels(), num_channels));
    } else {
        sample = img;
    }

    // resize
    Mat sample_resized;
    resize(sample, sample_resized, geometry);

    Mat sample_float;
    sample_resized.convertTo(sample_float,
                             num_channels == 3 ? CV_32FC3 : CV_32FC1);

    // subtract mean
    if (mean.size().width > 0) {
        Mat sample_normalized;
        subtract(sample_float, mean, sample_normalized);
        split(sample_normalized, *input_channels);
    }
    else {
        split(sample_float, *input_channels);
    }
}

void CaffeConverter_Impl::eval(InputArray image,
                               std::vector<float>& results) {
    const Mat img = image.getMat();

    const size_t channels = (*net_)[0]->in_data_shape()[0].depth_;
    const size_t width    = (*net_)[0]->in_data_shape()[0].width_;
    const size_t height   = (*net_)[0]->in_data_shape()[0].height_;

    vector<Mat> input_channels;
    vector<float> inputvec(width*height*channels);

    for (size_t i = 0; i < channels; i++) {
        input_channels.emplace_back(height, width, CV_32FC1,
                                    &inputvec[width*height*i]);
    }

    // subtract mean from input
    preprocess(img, mean_, 3, Size(width, height), &input_channels);

    const vector<tiny_dnn::float_t> vec(inputvec.begin(), inputvec.end());

    // perform inderence
    auto result = net_->predict(vec);

    // allocate output
    results.clear();
    results.reserve(result.size());

    for (size_t i = 0; i < result.size(); i++) {
        results.push_back(result[i]);
    }
}

Ptr<CaffeConverter> CaffeConverter::create(const String& model_file,
                                           const String& trained_file,
                                           const String& mean_file) {
    return makePtr<CaffeConverter_Impl>(model_file, trained_file, mean_file);
}

} // namespace dnn2
} // namespace cv
