// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DETECTOR_ALIGNMENT_PATTERN_HPP_
#define __ZXING_QRCODE_DETECTOR_ALIGNMENT_PATTERN_HPP_

#include "../../common/bitmatrix.hpp"
#include "../../resultpoint.hpp"
namespace zxing {
namespace qrcode {

class AlignmentPattern : public ResultPoint {
private:
    float estimatedModuleSize_;

public:
    AlignmentPattern(float posX, float posY, float estimatedModuleSize);
    bool aboutEquals(float moduleSize, float i, float j) const;
    float getModuleSize() { return estimatedModuleSize_; };

    Ref<AlignmentPattern> combineEstimate(float i, float j, float newModuleSize) const;
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DETECTOR_ALIGNMENT_PATTERN_HPP_
