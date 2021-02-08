// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_REEDSOLOMON_REEDSOLOMONDECODER_HPP__
#define __ZXING_COMMON_REEDSOLOMON_REEDSOLOMONDECODER_HPP__

#include "../../errorhandler.hpp"
#include "../array.hpp"
#include "../counted.hpp"
#include "genericgf.hpp"
#include "genericgfpoly.hpp"

namespace zxing {
class GenericGFPoly;
class GenericGF;

class ReedSolomonDecoder {
private:
    Ref<GenericGF> field;

public:
    explicit ReedSolomonDecoder(Ref<GenericGF> fld);
    ~ReedSolomonDecoder();
    void decode(ArrayRef<int> received, int twoS, ErrorHandler &err_handler);
    std::vector<Ref<GenericGFPoly>> runEuclideanAlgorithm(Ref<GenericGFPoly> a,
                                                          Ref<GenericGFPoly> b, int R,
                                                          ErrorHandler &err_handler);

private:
    ArrayRef<int> findErrorLocations(Ref<GenericGFPoly> errorLocator, ErrorHandler &err_handler);
    ArrayRef<int> findErrorMagnitudes(Ref<GenericGFPoly> errorEvaluator,
                                      ArrayRef<int> errorLocations, ErrorHandler &err_handler);
};
}  // namespace zxing

#endif  // __ZXING_COMMON_REEDSOLOMON_REEDSOLOMONDECODER_HPP__
