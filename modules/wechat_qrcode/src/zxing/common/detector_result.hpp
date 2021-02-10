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

#include "../resultpoint.hpp"
#include "array.hpp"
#include "bitmatrix.hpp"
#include "bytematrix.hpp"
#include "counted.hpp"

namespace zxing {

class DetectorResult : public Counted {
private:
    Ref<BitMatrix> bits_;
    Ref<ByteMatrix> gray_;
    ArrayRef<Ref<ResultPoint> > points_;

public:
    DetectorResult(Ref<BitMatrix> bits, ArrayRef<Ref<ResultPoint> > points, int dimension = 0,
                   float modulesize = 0);
    DetectorResult(Ref<ByteMatrix> gray, ArrayRef<Ref<ResultPoint> > points, int dimension = 0,
                   float modulesize = 0);
    Ref<BitMatrix> getBits();
    Ref<ByteMatrix> getGray();
    void SetGray(Ref<ByteMatrix> gray);
    ArrayRef<Ref<ResultPoint> > getPoints();
    int dimension_;
    float modulesize_;
};
}  // namespace zxing

#endif  // __ZXING_COMMON_DETECTOR_RESULT_HPP__
