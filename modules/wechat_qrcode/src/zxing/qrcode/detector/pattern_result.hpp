// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DETECTOR_PATTERN_RESULT_HPP_
#define __ZXING_QRCODE_DETECTOR_PATTERN_RESULT_HPP_

#include "../../common/array.hpp"
#include "../../common/bitmatrix.hpp"
#include "../../common/counted.hpp"
#include "../../common/detector_result.hpp"
#include "../../resultpoint.hpp"
#include "alignment_pattern.hpp"
#include "finder_pattern.hpp"
#include "finder_pattern_info.hpp"
namespace zxing {
namespace qrcode {
class PatternResult : public Counted {
public:
    Ref<FinderPatternInfo> finderPatternInfo;
    vector<Ref<AlignmentPattern> > possibleAlignmentPatterns;
    Ref<AlignmentPattern> confirmedAlignmentPattern;
    int possibleDimension;
    // vector<int> possibleDimensions;
    unsigned int possibleVersion;
    float possibleFix;
    float possibleModuleSize;

    explicit PatternResult(Ref<FinderPatternInfo> info);
    void setConfirmedAlignmentPattern(int index);
    int getPossibleAlignmentCount() { return possibleAlignmentPatterns.size(); }
    // int getPossibleDimensionCount();
public:
    unsigned int getPossibleVersion() { return possibleVersion; }
    float getPossibleFix() { return possibleFix; }
    float getPossibleModuleSize() { return possibleModuleSize; }
    int getDimension() { return possibleDimension; }
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DETECTOR_PATTERN_RESULT_HPP_
