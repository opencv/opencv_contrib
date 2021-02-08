// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../../../precomp.hpp"
#include "finder_pattern.hpp"

using zxing::Ref;

namespace zxing {
namespace qrcode {

FinderPattern::FinderPattern(float posX, float posY, float estimatedModuleSize)
    : ResultPoint(posX, posY),
      estimatedModuleSize_(estimatedModuleSize),
      count_(1),
      horizontalState_(FinderPattern::HORIZONTAL_STATE_NORMAL),
      verticalState_(FinderPattern::VERTICAL_STATE_NORMAL) {
    fix_ = -1.0f;
}

FinderPattern::FinderPattern(float posX, float posY, float estimatedModuleSize, int count)
    : ResultPoint(posX, posY),
      estimatedModuleSize_(estimatedModuleSize),
      count_(count),
      horizontalState_(FinderPattern::HORIZONTAL_STATE_NORMAL),
      verticalState_(FinderPattern::VERTICAL_STATE_NORMAL) {
    fix_ = -1.0f;
}
int FinderPattern::getCount() const { return count_; }
void FinderPattern::incrementCount() { count_++; }

bool FinderPattern::aboutEquals(float moduleSize, float i, float j) const {
    if (abs(i - getY()) <= moduleSize && abs(j - getX()) <= moduleSize) {
        float moduleSizeDiff = abs(moduleSize - estimatedModuleSize_);
        return moduleSizeDiff <= 1.0f || moduleSizeDiff <= estimatedModuleSize_;
    }
    return false;
}

float FinderPattern::getEstimatedModuleSize() const { return estimatedModuleSize_; }

Ref<FinderPattern> FinderPattern::combineEstimate(float i, float j, float newModuleSize) const {
    int combinedCount = count_ + 1;
    float combinedX = getX();
    float combinedY = getY();
    float combinedModuleSize = getEstimatedModuleSize();
    if (combinedCount <= 3) {
        combinedX = (count_ * getX() + j) / combinedCount;
        combinedY = (count_ * getY() + i) / combinedCount;
        combinedModuleSize = (count_ * getEstimatedModuleSize() + newModuleSize) / combinedCount;
    }
    return Ref<FinderPattern>(
        new FinderPattern(combinedX, combinedY, combinedModuleSize, combinedCount));
}

void FinderPattern::setHorizontalCheckState(int state) {
    switch (state) {
        case 0:
            horizontalState_ = FinderPattern::HORIZONTAL_STATE_NORMAL;
            break;
        case 1:
            horizontalState_ = FinderPattern::HORIZONTAL_STATE_LEFT_SPILL;
            break;
        case 2:
            horizontalState_ = FinderPattern::HORIZONTAL_STATE_RIGHT_SPILL;
            break;
    }
    return;
}
void FinderPattern::setVerticalCheckState(int state) {
    switch (state) {
        case 0:
            verticalState_ = FinderPattern::VERTICAL_STATE_NORMAL;
            break;
        case 1:
            verticalState_ = FinderPattern::VERTICAL_STATE_UP_SPILL;
            break;
        case 2:
            verticalState_ = FinderPattern::VERTICAL_STATE_DOWN_SPILL;
            break;
    }
    return;
}
}  // namespace qrcode
}  // namespace zxing
