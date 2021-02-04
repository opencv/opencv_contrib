// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#include "../../../precomp.hpp"
#include "adaptive_threshold_mean_binarizer.hpp"
using zxing::AdaptiveThresholdMeanBinarizer;

namespace {
const int BLOCK_SIZE = 25;
const int Bias = 10;
}  // namespace

AdaptiveThresholdMeanBinarizer::AdaptiveThresholdMeanBinarizer(Ref<LuminanceSource> source)
    : GlobalHistogramBinarizer(source) {}

AdaptiveThresholdMeanBinarizer::~AdaptiveThresholdMeanBinarizer() {}

Ref<Binarizer> AdaptiveThresholdMeanBinarizer::createBinarizer(Ref<LuminanceSource> source) {
    return Ref<Binarizer>(new AdaptiveThresholdMeanBinarizer(source));
}

Ref<BitArray> AdaptiveThresholdMeanBinarizer::getBlackRow(int y, Ref<BitArray> row,
                                                          ErrorHandler& err_handler) {
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_) {
        binarizeImage(err_handler);
        if (err_handler.ErrCode()) return Ref<BitArray>();
    }

    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackRow(y, row, err_handler);
}

Ref<BitMatrix> AdaptiveThresholdMeanBinarizer::getBlackMatrix(ErrorHandler& err_handler) {
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_) {
        binarizeImage(err_handler);
        if (err_handler.ErrCode()) return Ref<BitMatrix>();
    }
    return Binarizer::getBlackMatrix(err_handler);
}

int AdaptiveThresholdMeanBinarizer::binarizeImage(ErrorHandler& err_handler) {
    if (width >= BLOCK_SIZE && height >= BLOCK_SIZE) {
        LuminanceSource& source = *getLuminanceSource();
        Ref<BitMatrix> matrix(new BitMatrix(width, height, err_handler));
        if (err_handler.ErrCode()) return -1;
        auto src = (unsigned char*)source.getMatrix()->data();
        auto dst = matrix->getPtr();
        cv::Mat mDst;
        mDst = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
        TransBufferToMat(src, mDst, width, height);
        cv::Mat result;
        int bs = width / 10;
        bs = bs + bs % 2 - 1;
        if (!(bs % 2 == 1 && bs > 1)) return -1;
        cv::adaptiveThreshold(mDst, result, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                              bs, Bias);
        TransMatToBuffer(result, dst, width, height);
        if (err_handler.ErrCode()) return -1;
        matrix0_ = matrix;
    } else {
        matrix0_ = GlobalHistogramBinarizer::getBlackMatrix(err_handler);
        if (err_handler.ErrCode()) return 1;
    }
    return 0;
}

int AdaptiveThresholdMeanBinarizer::TransBufferToMat(unsigned char* pBuffer, cv::Mat& mDst,
                                                     int nWidth, int nHeight) {
    for (int j = 0; j < nHeight; ++j) {
        unsigned char* data = mDst.ptr<unsigned char>(j);
        unsigned char* pSubBuffer = pBuffer + (nHeight - 1 - j) * nWidth;
        memcpy(data, pSubBuffer, nWidth);
    }
    return 0;
}

int AdaptiveThresholdMeanBinarizer::TransMatToBuffer(cv::Mat mSrc, unsigned char* ppBuffer,
                                                     int& nWidth, int& nHeight) {
    nWidth = mSrc.cols;
    // nWidth = ((nWidth + 3) / 4) * 4;
    nHeight = mSrc.rows;
    for (int j = 0; j < nHeight; ++j) {
        unsigned char* pdi = ppBuffer + j * nWidth;
        for (int z = 0; z < nWidth; ++z) {
            int nj = nHeight - j - 1;
            int value = *(uchar*)(mSrc.ptr<uchar>(nj) + z);
            if (value > 120)
                pdi[z] = 0;
            else
                pdi[z] = 1;
        }
    }
    return 0;
}