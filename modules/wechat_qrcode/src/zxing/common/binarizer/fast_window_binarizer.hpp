// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_DETECTOR_RESULT_HPP__
#define __ZXING_COMMON_DETECTOR_RESULT_HPP__

#include "../../binarizer.hpp"
#include "../../errorhandler.hpp"
#include "../bitarray.hpp"
#include "../bitmatrix.hpp"
#include "global_histogram_binarizer.hpp"

#include <vector>

namespace zxing {

class FastWindowBinarizer : public GlobalHistogramBinarizer {
private:
    Ref<BitMatrix> matrix_;
    Ref<BitArray> cached_row_;

    int* _luminancesInt;
    int* _blockTotals;
    int* _totals;
    int* _rowTotals;

    unsigned int* _internal;

public:
    explicit FastWindowBinarizer(Ref<LuminanceSource> source);
    virtual ~FastWindowBinarizer();

    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler& err_handler) override;
    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler& err_handler) override;

    Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source) override;

private:
    void calcBlockTotals(int* luminancesInt, int* output, int aw, int ah);
    void cumulative(int* data, int* output, int _width, int _height);
    int binarizeImage0(ErrorHandler& err_handler);
    void fastIntegral(const unsigned char* inputMatrix, unsigned int* outputMatrix);
    int binarizeImage1(ErrorHandler& err_handler);
    void fastWindow(const unsigned char* src, unsigned char* dst, ErrorHandler& err_handler);
};

}  // namespace zxing

#endif  // __ZXING_COMMON_DETECTOR_RESULT_HPP__
