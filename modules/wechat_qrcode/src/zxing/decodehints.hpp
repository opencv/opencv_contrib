// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_DECODEHINTS_HPP__
#define __ZXING_DECODEHINTS_HPP__

#include "errorhandler.hpp"

namespace zxing {
class DecodeHints {
private:
    bool use_nn_detector_;

public:
    explicit DecodeHints(bool use_nn_detector = false) : use_nn_detector_(use_nn_detector){};

    bool getUseNNDetector() const { return use_nn_detector_; }
    void setUseNNDetector(bool use_nn_detector) { use_nn_detector_ = use_nn_detector; }
};

}  // namespace zxing

#endif  // __ZXING_DECODEHINTS_HPP__
