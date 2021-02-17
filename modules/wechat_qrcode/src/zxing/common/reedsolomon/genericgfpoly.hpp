// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_REEDSOLOMON_GENERICGFPOLY_HPP__
#define __ZXING_COMMON_REEDSOLOMON_GENERICGFPOLY_HPP__

#include "../../errorhandler.hpp"
#include "../array.hpp"
#include "../counted.hpp"

namespace zxing {

class GenericGF;

class GenericGFPoly : public Counted {
private:
    GenericGF &field_;
    ArrayRef<int> coefficients_;

public:
    GenericGFPoly(GenericGF &field, ArrayRef<int> coefficients, ErrorHandler &err_handler);
    ArrayRef<int> getCoefficients();
    int getDegree();
    bool isZero();
    int getCoefficient(int degree);
    int evaluateAt(int a);
    Ref<GenericGFPoly> addOrSubtract(Ref<GenericGFPoly> other, ErrorHandler &err_handler);
    Ref<GenericGFPoly> multiply(Ref<GenericGFPoly> other, ErrorHandler &err_handler);
    Ref<GenericGFPoly> multiply(int scalar, ErrorHandler &err_handler);
    Ref<GenericGFPoly> multiplyByMonomial(int degree, int coefficient, ErrorHandler &err_handler);
    std::vector<Ref<GenericGFPoly>> divide(Ref<GenericGFPoly> other, ErrorHandler &err_handler);
};

}  // namespace zxing

#endif  // __ZXING_COMMON_REEDSOLOMON_GENERICGFPOLY_HPP__
