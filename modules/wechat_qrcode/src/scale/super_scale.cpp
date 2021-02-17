// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#include "../precomp.hpp"
#include "super_scale.hpp"


#define CLIP(x, x1, x2) max(x1, min(x, x2))
namespace cv {
namespace wechat_qrcode {
int SuperScale::init(const std::string &proto_path, const std::string &model_path) {
    srnet_ = dnn::readNetFromCaffe(proto_path, model_path);
    net_loaded_ = true;
    return 0;
}

Mat SuperScale::processImageScale(const Mat &src, float scale, const bool &use_sr,
                                  int sr_max_size) {
    Mat dst = src;
    if (scale == 1.0) {  // src
        return dst;
    }

    int width = src.cols;
    int height = src.rows;
    if (scale == 2.0) {  // upsample
        int SR_TH = sr_max_size;
        if (use_sr && (int)sqrt(width * height * 1.0) < SR_TH && net_loaded_) {
            int ret = superResoutionScale(src, dst);
            if (ret == 0) return dst;
        }

        { resize(src, dst, Size(), scale, scale, INTER_CUBIC); }
    } else if (scale < 1.0) {  // downsample
        resize(src, dst, Size(), scale, scale, INTER_AREA);
    }

    return dst;
}

int SuperScale::superResoutionScale(const Mat &src, Mat &dst) {
    Mat blob;
    dnn::blobFromImage(src, blob, 1.0 / 255, Size(src.cols, src.rows), {0.0f}, false, false);

    srnet_.setInput(blob);
    auto prob = srnet_.forward();

    dst = Mat(prob.size[2], prob.size[3], CV_8UC1);

    for (int row = 0; row < prob.size[2]; row++) {
        const float *prob_score = prob.ptr<float>(0, 0, row);
        for (int col = 0; col < prob.size[3]; col++) {
            float pixel = prob_score[col] * 255.0;
            dst.at<uint8_t>(row, col) = static_cast<uint8_t>(CLIP(pixel, 0.0f, 255.0f));
        }
    }
    return 0;
}
}  // namespace wechat_qrcode
}  // namespace cv