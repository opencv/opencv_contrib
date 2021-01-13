// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DETECTOR_DETECTOR_HPP_
#define __ZXING_QRCODE_DETECTOR_DETECTOR_HPP_

#include "zxing/common/bitmatrix.hpp"
#include "zxing/common/counted.hpp"
#include "zxing/common/detector_result.hpp"
#include "zxing/common/perspective_transform.hpp"
#include "zxing/common/unicomblock.hpp"
#include "zxing/errorhandler.hpp"
#include "zxing/qrcode/detector/alignment_pattern.hpp"
#include "zxing/qrcode/detector/finder_pattern.hpp"
#include "zxing/qrcode/detector/finder_pattern_info.hpp"
#include "zxing/qrcode/detector/pattern_result.hpp"

#include <vector>

namespace zxing {
class DecodeHints;

namespace qrcode {

// Possible Detect Result

class Detector : public Counted {
public:
    enum DetectorState {
        START = 10,
        FINDFINDERPATTERN = 11,
        FINDALIGNPATTERN = 12,
    };

    // Fix module size error when LEFT_SPILL or RIGHT_SPILL
    enum FinderPatternMode {
        NORMAL = 0,
        LEFT_SPILL = 1,
        RIGHT_SPILL = 2,
        UP_SPILL = 3,
        DOWN_SPILL = 4,
    };

    typedef struct Rect_ {
        int x;
        int y;
        int width;
        int height;
    } Rect;

private:
    Ref<BitMatrix> image_;
    Ref<UnicomBlock> block_;

    vector<Ref<PatternResult> > possiblePatternResults_;

#ifdef FIND_WHITE_ALIGNMENTPATTERN
    Ref<AlignmentPattern> whiteAlignmentPattern_;
    vector<Ref<AlignmentPattern> > possibleWhiteAlignmentPatterns_;
#endif

    DetectorState detectorState_;
    bool finderConditionLoose_;

protected:
    Ref<BitMatrix> getImage() const;
    static int computeDimension(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                                Ref<ResultPoint> bottomLeft, float moduleSizeX, float moduleSizeY);
    float calculateModuleSize(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                              Ref<ResultPoint> bottomLeft);
    float calculateModuleSizeOneWay(Ref<ResultPoint> pattern, Ref<ResultPoint> otherPattern,
                                    int patternState, int otherPatternState);
    float sizeOfBlackWhiteBlackRunBothWays(int fromX, int fromY, int toX, int toY, int patternState,
                                           bool isReverse);
    float sizeOfBlackWhiteBlackRunBothWays(int fromX, int fromY, int toX, int toY);
    float sizeOfBlackWhiteBlackRun(int fromX, int fromY, int toX, int toY);
    Ref<AlignmentPattern> findAlignmentInRegion(float overallEstModuleSize, int estAlignmentX,
                                                int estAlignmentY, float allowanceFactor,
                                                ErrorHandler &err_handler);
    Ref<AlignmentPattern> findAlignmentWithFitLine(Ref<ResultPoint> topLeft,
                                                   Ref<ResultPoint> topRight,
                                                   Ref<ResultPoint> bottomLeft, float moduleSize,
                                                   ErrorHandler &err_handler);
    int fitLine(vector<Ref<ResultPoint> > &oldPoints, float &k, float &b, int &a);
    bool checkTolerance(Ref<ResultPoint> &topLeft, Ref<ResultPoint> &topRight, Rect &topRightRect,
                        double modelSize, Ref<ResultPoint> &p, int flag);
    void findPointsForLine(Ref<ResultPoint> &topLeft, Ref<ResultPoint> &topRight,
                           Ref<ResultPoint> &bottomLeft, Rect topRightRect, Rect bottomLeftRect,
                           vector<Ref<ResultPoint> > &topRightPoints,
                           vector<Ref<ResultPoint> > &bottomLeftPoints, float modelSize);
    bool checkConvexQuadrilateral(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                                  Ref<ResultPoint> bottomLeft, Ref<ResultPoint> bottomRight);

public:
    virtual Ref<PerspectiveTransform> createTransform(Ref<ResultPoint> topLeft,
                                                      Ref<ResultPoint> topRight,
                                                      Ref<ResultPoint> bottomLeft,
                                                      Ref<ResultPoint> alignmentPattern,
                                                      int dimension);
    Ref<PerspectiveTransform> createTransform(Ref<FinderPatternInfo> finderPatternInfo,
                                              Ref<ResultPoint> alignmentPattern, int dimension);

    static Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimension, Ref<PerspectiveTransform>,
                                     ErrorHandler &err_handler);

    Detector(Ref<BitMatrix> image, Ref<UnicomBlock> block);
    void detect(DecodeHints const &hints, ErrorHandler &err_handler);
    Ref<DetectorResult> getResultViaAlignment(int patternIdx, int alignmentIndex,
                                              int possibleDimension, ErrorHandler &err_handler);

    int getPossiblePatternCount() { return possiblePatternResults_.size(); }
    int getPossibleAlignmentCount(int idx);

    Ref<AlignmentPattern> getNearestAlignmentPattern(int tryFindRange, float moduleSize,
                                                     int estAlignmentX, int estAlignmentY);
    bool hasSameResult(vector<Ref<AlignmentPattern> > possibleAlignmentPatterns,
                       Ref<AlignmentPattern> alignmentPattern);
    void fixAlignmentPattern(float &alignmentX, float &alignmentY, float moduleSize);

    // void processFinderPatternInfo(Ref<FinderPatternInfo> info);
    Ref<PatternResult> processFinderPatternInfo(Ref<FinderPatternInfo> info,
                                                ErrorHandler &err_handler);
    // void setConfirmedAlignmentPattern(int index);

#ifdef FIND_WHITE_ALIGNMENTPATTERN
    Ref<AlignmentPattern> getWhiteAlignmentPattern() { return whiteAlignmentPattern_; }
#endif

public:
    // size_t getPossibleResultCount();

    Ref<FinderPatternInfo> getFinderPatternInfo(int idx) {
        return possiblePatternResults_[idx]->finderPatternInfo;
    }
    Ref<AlignmentPattern> getAlignmentPattern(int patternIdx, int alignmentIdx) {
        return possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIdx];
    }

    DetectorState getState() { return detectorState_; }
    void setFinderLoose(bool loose) { finderConditionLoose_ = loose; }
    bool getFinderLoose() { return finderConditionLoose_; }

    unsigned int getPossibleVersion(int idx) {
        return possiblePatternResults_[idx]->possibleVersion;
    }
    float getPossibleFix(int idx) { return possiblePatternResults_[idx]->possibleFix; }
    float getPossibleModuleSize(int idx) {
        return possiblePatternResults_[idx]->possibleModuleSize;
    }
    int getDimension(int idx) { return possiblePatternResults_[idx]->possibleDimension; }
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DETECTOR_DETECTOR_HPP_
