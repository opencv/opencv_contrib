// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_GRID_SAMPLER_HPP__
#define __ZXING_COMMON_GRID_SAMPLER_HPP__

#include "bitmatrix.hpp"
#include "bytematrix.hpp"
#include "counted.hpp"
#include "perspective_transform.hpp"

namespace zxing {
class GridSampler {
private:
    static GridSampler gridSampler;
    GridSampler();

public:
    Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimension,
                              Ref<PerspectiveTransform> transform, ErrorHandler &err_handler);
    static int checkAndNudgePoints(int width, int height, vector<float> &points,
                                   ErrorHandler &err_handler);
    static GridSampler &getInstance();
};
}  // namespace zxing

#endif  // __ZXING_COMMON_GRID_SAMPLER_HPP__
