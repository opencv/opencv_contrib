// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_PERSPECTIVETRANSFORM_HPP__
#define __ZXING_COMMON_PERSPECTIVETRANSFORM_HPP__

#include "counted.hpp"

#include <vector>

namespace zxing {
class PerspectiveTransform : public Counted {
private:
    float a11, a12, a13, a21, a22, a23, a31, a32, a33;
    PerspectiveTransform(float a11, float a21, float a31, float a12, float a22, float a32,
                         float a13, float a23, float a33);

public:
    static Ref<PerspectiveTransform> quadrilateralToQuadrilateral(
        float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3, float x0p,
        float y0p, float x1p, float y1p, float x2p, float y2p, float x3p, float y3p);
    static Ref<PerspectiveTransform> squareToQuadrilateral(float x0, float y0, float x1, float y1,
                                                           float x2, float y2, float x3, float y3);
    static Ref<PerspectiveTransform> quadrilateralToSquare(float x0, float y0, float x1, float y1,
                                                           float x2, float y2, float x3, float y3);
    Ref<PerspectiveTransform> buildAdjoint();
    Ref<PerspectiveTransform> times(Ref<PerspectiveTransform> other);
    void transformPoints(std::vector<float>& points);
};
}  // namespace zxing

#endif  // __ZXING_COMMON_PERSPECTIVETRANSFORM_HPP__
