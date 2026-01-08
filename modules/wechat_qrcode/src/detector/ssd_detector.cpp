// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#include "../precomp.hpp"
#include "ssd_detector.hpp"
#define CLIP(x, x1, x2) max(x1, min(x, x2))
namespace cv {
namespace wechat_qrcode {
int SSDDetector::init(const string& proto_path, const string& model_path) {
    net_ = dnn::readNetFromCaffe(proto_path, model_path);
    return 0;
}

vector<Mat> SSDDetector::forward(Mat img, const int target_width, const int target_height) {
    int img_w = img.cols;
    int img_h = img.rows;
    Mat input;
    resize(img, input, Size(target_width, target_height), 0, 0, INTER_CUBIC);

    dnn::blobFromImage(input, input, 1.0 / 255, Size(input.cols, input.rows), {0.0f, 0.0f, 0.0f},
                       false, false);
    net_.setInput(input, "data");

    auto prob = net_.forward("detection_output");
    vector<Mat> point_list;
    // the shape is (1,1,100,7)=>(batch,channel,count,dim)
    for (int row = 0; row < prob.size[2]; row++) {
        const float* prob_score = prob.ptr<float>(0, 0, row);
        // prob_score[0] is not used.
        // prob_score[1]==1 stands for qrcode
        if (prob_score[1] == 1 && prob_score[2] > 1E-5) {
            // add a safe score threshold due to https://github.com/opencv/opencv_contrib/issues/2877
            // prob_score[2] is the probability of the qrcode, which is not used.
            auto point = Mat(4, 2, CV_32FC1);
            float x0 = CLIP(prob_score[3] * img_w, 0.0f, img_w - 1.0f);
            float y0 = CLIP(prob_score[4] * img_h, 0.0f, img_h - 1.0f);
            float x1 = CLIP(prob_score[5] * img_w, 0.0f, img_w - 1.0f);
            float y1 = CLIP(prob_score[6] * img_h, 0.0f, img_h - 1.0f);

            point.at<float>(0, 0) = x0;
            point.at<float>(0, 1) = y0;
            point.at<float>(1, 0) = x1;
            point.at<float>(1, 1) = y0;
            point.at<float>(2, 0) = x1;
            point.at<float>(2, 1) = y1;
            point.at<float>(3, 0) = x0;
            point.at<float>(3, 1) = y1;
            point_list.push_back(point);
        }
    }
    return point_list;
}
}  // namespace wechat_qrcode
}  // namespace cv