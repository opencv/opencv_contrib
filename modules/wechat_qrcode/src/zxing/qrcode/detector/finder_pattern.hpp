// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_HPP_
#define __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_HPP_

#include "../../common/bitmatrix.hpp"
#include "../../resultpoint.hpp"

namespace zxing {
namespace qrcode {

class FinderPattern : public ResultPoint {
public:
    enum CheckState {
        HORIZONTAL_STATE_NORMAL = 0,
        HORIZONTAL_STATE_LEFT_SPILL = 1,
        HORIZONTAL_STATE_RIGHT_SPILL = 2,
        VERTICAL_STATE_NORMAL = 3,
        VERTICAL_STATE_UP_SPILL = 4,
        VERTICAL_STATE_DOWN_SPILL = 5
    };

private:
    float estimatedModuleSize_;
    int count_;

    FinderPattern(float posX, float posY, float estimatedModuleSize, int count);

public:
    FinderPattern(float posX, float posY, float estimatedModuleSize);
    int getCount() const;
    float getEstimatedModuleSize() const;
    void incrementCount();
    bool aboutEquals(float moduleSize, float i, float j) const;
    Ref<FinderPattern> combineEstimate(float i, float j, float newModuleSize) const;

    void setHorizontalCheckState(int state);
    void setVerticalCheckState(int state);

    int getHorizontalCheckState() { return horizontalState_; }
    int getVerticalCheckState() { return verticalState_; }

private:
    float fix_;
    CheckState horizontalState_;
    CheckState verticalState_;
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_HPP_
