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
#include "alignment_pattern.hpp"
using zxing::Ref;
using zxing::qrcode::AlignmentPattern;
namespace zxing {
namespace qrcode {
AlignmentPattern::AlignmentPattern(float posX, float posY, float estimatedModuleSize)
    : ResultPoint(posX, posY), estimatedModuleSize_(estimatedModuleSize) {}

// Determines if this alignment pattern "about equals" an alignment pattern at
// the stated position and size -- meaning, it is at nearly the same center with
// nearly the same size.
bool AlignmentPattern::aboutEquals(float moduleSize, float i, float j) const {
    if (abs(i - getY()) <= moduleSize && abs(j - getX()) <= moduleSize) {
        float moduleSizeDiff = abs(moduleSize - estimatedModuleSize_);
        return moduleSizeDiff <= 1.0f || moduleSizeDiff <= estimatedModuleSize_;
    }
    return false;
}

// Combines this object's current estimate of a finder pattern position and
// module size with a new estimate. It returns a new {@code FinderPattern}
// containing an average of the two.
Ref<AlignmentPattern> AlignmentPattern::combineEstimate(float i, float j,
                                                        float newModuleSize) const {
    float combinedX = (getX() + j) / 2.0f;
    float combinedY = (getY() + i) / 2.0f;
    float combinedModuleSize = (estimatedModuleSize_ + newModuleSize) / 2.0f;
    Ref<AlignmentPattern> result(new AlignmentPattern(combinedX, combinedY, combinedModuleSize));
    return result;
}
}  // namespace qrcode
}  // namespace zxing