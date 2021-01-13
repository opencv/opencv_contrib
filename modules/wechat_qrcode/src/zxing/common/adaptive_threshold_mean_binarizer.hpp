// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __ZXING_COMMON_ADAPTIVE_THRESHOLD_MEAN_BINARIZER_HPP__
#define __ZXING_COMMON_ADAPTIVE_THRESHOLD_MEAN_BINARIZER_HPP__
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "zxing/binarizer.hpp"
#include "zxing/common/bitarray.hpp"
#include "zxing/common/bitmatrix.hpp"
#include "zxing/common/bytematrix.hpp"
#include "zxing/common/global_histogram_binarizer.hpp"
#include "zxing/errorhandler.hpp"

// Macro to use max-min in function calculateBlackPoints
#ifndef USE_MAX_MIN
#define USE_MAX_MIN 0
#endif

#ifndef USE_GOOGLE_CODE
#define USE_GOOGLE_CODE 0
#endif

namespace zxing {

class AdaptiveThresholdMeanBinarizer : public GlobalHistogramBinarizer {
private:
    int width_;
    int height_;

public:
    explicit AdaptiveThresholdMeanBinarizer(Ref<LuminanceSource> source);
    virtual ~AdaptiveThresholdMeanBinarizer();

    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler& err_handler) override;
    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler& err_handler) override;
    Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source) override;

private:
    int binarizeImage(ErrorHandler& err_handler);
    int TransBufferToMat(unsigned char* pBuffer, cv::Mat& mDst, int nWidth, int nHeight);
    int TransMatToBuffer(cv::Mat mSrc, unsigned char* ppBuffer, int& nWidth, int& nHeight);
};

}  // namespace zxing
#endif  // __ZXING_COMMON_ADAPTIVE_THRESHOLD_MEAN_BINARIZER_HPP__
