// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../../precomp.hpp"
#include "detector_result.hpp"

namespace zxing {

DetectorResult::DetectorResult(Ref<BitMatrix> bits, ArrayRef<Ref<ResultPoint> > points,
                               int dimension, float modulesize)
    : bits_(bits), points_(points), dimension_(dimension), modulesize_(modulesize) {}

void DetectorResult::SetGray(Ref<ByteMatrix> gray) { gray_ = gray; }

Ref<BitMatrix> DetectorResult::getBits() { return bits_; }

Ref<ByteMatrix> DetectorResult::getGray() { return gray_; }

ArrayRef<Ref<ResultPoint> > DetectorResult::getPoints() { return points_; }

}  // namespace zxing
