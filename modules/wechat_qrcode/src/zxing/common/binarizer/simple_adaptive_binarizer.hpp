// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_SIMPLEADAPTIVEBINARIZER_HPP__
#define __ZXING_COMMON_SIMPLEADAPTIVEBINARIZER_HPP__

#include "../../binarizer.hpp"
#include "../array.hpp"
#include "../bitarray.hpp"
#include "../bitmatrix.hpp"
#include "global_histogram_binarizer.hpp"


namespace zxing {

class SimpleAdaptiveBinarizer : public GlobalHistogramBinarizer {
public:
    explicit SimpleAdaptiveBinarizer(Ref<LuminanceSource> source);
    virtual ~SimpleAdaptiveBinarizer();

    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler &err_handler) override;
    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler &err_handler) override;
    Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source) override;

private:
    int binarizeImage0(ErrorHandler &err_handler);
    int qrBinarize(const unsigned char *src, unsigned char *dst);
    bool filtered;
};

}  // namespace zxing

#endif  // __ZXING_COMMON_SIMPLEADAPTIVEBINARIZER_HPP__
