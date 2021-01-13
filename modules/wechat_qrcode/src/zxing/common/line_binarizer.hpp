// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_LINE_BINARIZER_HPP__
#define __ZXING_COMMON_LINE_BINARIZER_HPP__

#include "zxing/binarizer.hpp"
#include "zxing/common/bitarray.hpp"
#include "zxing/common/bitmatrix.hpp"
#include "zxing/common/global_histogram_binarizer.hpp"
#include "zxing/errorhandler.hpp"

#include <vector>

namespace zxing {

class LineBinarizer : public GlobalHistogramBinarizer {
private:
public:
    explicit LineBinarizer(Ref<LuminanceSource> source);
    virtual ~LineBinarizer();

    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler &err_handler) override;
    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler &err_handler) override;
    Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source) override;

private:
    int binarizeImage(ErrorHandler &err_handler);
    void binarizeImage(const unsigned char *src, unsigned char *dst, int _width, int _height);
    bool binarizeLine(const unsigned char *src, unsigned char *dst, int _width);
    void scanLine(const unsigned char *line, int _width, std::vector<short> &maxiam_index,
                  std::vector<short> &miniam_index);
};

}  // namespace zxing

#endif  // __ZXING_COMMON_LINE_BINARIZER_HPP__
