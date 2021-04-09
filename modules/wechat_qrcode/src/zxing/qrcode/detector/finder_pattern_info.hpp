// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_INFO_HPP_
#define __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_INFO_HPP_

#include "../../common/array.hpp"
#include "../../common/counted.hpp"
#include "finder_pattern.hpp"

namespace zxing {
namespace qrcode {

class FinderPatternInfo : public Counted {
private:
    Ref<FinderPattern> bottomLeft_;
    Ref<FinderPattern> topLeft_;
    Ref<FinderPattern> topRight_;
    float possibleFix_;
    float anglePossibleFix_;

public:
    explicit FinderPatternInfo(std::vector<Ref<FinderPattern> > patternCenters);

    Ref<FinderPattern> getBottomLeft();
    Ref<FinderPattern> getTopLeft();
    Ref<FinderPattern> getTopRight();
    void estimateFinderPatternInfo();
    float getPossibleFix();
    float getAnglePossibleFix();
    // to void code duplicated
    static void calculateSides(Ref<FinderPattern> centerA, Ref<FinderPattern> centerB,
                               Ref<FinderPattern> centerC, float &longSide, float &shortSide1,
                               float &shortSide2);
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_INFO_HPP_
