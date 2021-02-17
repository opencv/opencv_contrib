// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DETECTOR_ALIGNMENT_PATTERN_FINDER_HPP_
#define __ZXING_QRCODE_DETECTOR_ALIGNMENT_PATTERN_FINDER_HPP_

#include "../../common/bitmatrix.hpp"
#include "../../common/counted.hpp"
#include "../../errorhandler.hpp"
#include "alignment_pattern.hpp"
#include "finder_pattern.hpp"

namespace zxing {
namespace qrcode {

class AlignmentPatternFinder : public Counted {
private:
    static int CENTER_QUORUM;
    static int MIN_SKIP;
    static int MAX_MODULES;

    Ref<BitMatrix> image_;
    std::vector<AlignmentPattern *> *possibleCenters_;

    int startX_;
    int startY_;
    int width_;
    int height_;
    float moduleSize_;
    static float centerFromEnd(std::vector<int> &stateCount, int end);
    float crossCheckVertical(int startI, int centerJ, int maxCount, int originalStateCountTotal);


public:
    AlignmentPatternFinder(Ref<BitMatrix> image, int startX, int startY, int width, int height,
                           float moduleSize);
    AlignmentPatternFinder(Ref<BitMatrix> image, float moduleSize);
    ~AlignmentPatternFinder();

    Ref<AlignmentPattern> find(ErrorHandler &err_handler);
    bool foundPatternCross(std::vector<int> &stateCount);
    Ref<AlignmentPattern> handlePossibleCenter(std::vector<int> &stateCount, int i, int j);


private:
    AlignmentPatternFinder(const AlignmentPatternFinder &);
    AlignmentPatternFinder &operator=(const AlignmentPatternFinder &);

};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DETECTOR_ALIGNMENT_PATTERN_FINDER_HPP_
