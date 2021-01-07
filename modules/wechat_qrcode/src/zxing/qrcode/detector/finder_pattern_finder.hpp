// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

/*
 *  finder_pattern_finder.hpp
 *  zxing
 *
 *  Copyright 2010 ZXing authors All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_FINDER_HPP_
#define __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_FINDER_HPP_

#include "zxing/common/bitmatrix.hpp"
#include "zxing/common/counted.hpp"
#include "zxing/common/unicomblock.hpp"
#include "zxing/errorhandler.hpp"
#include "zxing/qrcode/detector/finder_pattern.hpp"
#include "zxing/qrcode/detector/finder_pattern_info.hpp"

#include <vector>

using zxing::ErrorHandler;
using zxing::ReaderErrorHandler;
// Donot handle to small finder pattern, as it may usually caused by noise
//#define MIN_MODULESIZE 3.0

namespace zxing {

class DecodeHints;

namespace qrcode {

class FinderPatternFinder {
public:
    enum CrossCheckState {
        NORMAL = 0,
        LEFT_SPILL = 1,
        RIHGT_SPILL = 2,
        LEFT_RIGHT_SPILL = 3,
        NOT_PATTERN = 4,
        // UP_SPILL = 3,
        // DOWN_SPILL = 4,
    };

private:
    static int CENTER_QUORUM;
    static int MIN_SKIP;
    static int MAX_MODULES;
    static int INTEGER_MATH_SHIFT;
    static int FP_INPUT_CNN_MAX_NUM;
    static int FP_IS_SELECT_BEST;
    static int FP_IS_SELECT_FILE_BEST;
    static int FP_INPUT_MAX_NUM;
    static int FP_FILTER_SIZE;
    static int FPS_CLUSTER_MAX;
    static int FPS_RESULT_MAX;
    static int K_FACTOR;

    static float FPS_MS_VAL;
    static float FP_COUNT_MIN;
    static float FP_MS_MIN;
    static float FP_RIGHT_ANGLE;
    static float FP_SMALL_ANGLE1;
    static float FP_SMALL_ANGLE2;
    static float QR_MIN_FP_AREA_ERR;
    static float QR_MIN_FP_MS_ERR;
    static int QR_MIN_FP_ACCEPT;

    int finder_time;
    CrossCheckState CURRENT_CHECK_STATE;
    int compared_finder_counts;

    // ArrayRef<float> _isVerticalChecked;
    struct HorizontalCheckedResult {
        size_t centerI;
        float centerJ;
    };

    vector<vector<HorizontalCheckedResult> > _horizontalCheckedResult;
    // ArrayRef<float> _checkedResult;

    // INI CONFIG

protected:
    Ref<BitMatrix> image_;
    std::vector<Ref<FinderPattern> > possibleCenters_;

    bool hasSkipped_;
    Ref<UnicomBlock> block_;

    /** stateCount must be int[5] */
    float centerFromEnd(int* stateCount, int end);
    // 检查是否满足回字形
    bool foundPatternCross(int* stateCount);
    // Used only for checkDiagonal
    bool foundPatternCrossLoose(int* stateCount);
    bool finderConditionLoose_;
    // 尝试插入到possibleCenters_
    // 返回true表示成功插入
    // 返回false表示列表中已经有了
    int getStateCountTotal(int* stateCount, const CrossCheckState& check_state);
    bool tryToPushToCenters(float posX, float posY, float estimatedModuleSize,
                            CrossCheckState horizontalState = FinderPatternFinder::NORMAL,
                            CrossCheckState verticalState = FinderPatternFinder::NORMAL);
    bool crossCheckDiagonal(int startI, int centerJ, int maxCount, int originalStateCountTotal);
    float crossCheckVertical(size_t startI, size_t centerJ, int maxCount,
                             int originalStateCountTota, float& estimatedVerticalModuleSize);
    float crossCheckHorizontal(size_t startJ, size_t centerI, int maxCount,
                               int originalStateCountTotal, float& estimatedHorizontalModuleSize);

    float hasHorizontalCheckedResult(size_t startJ, size_t centerI);
    int addHorizontalCheckedResult(size_t startJ, size_t centerI, float centerJ);
    int getMinModuleSize();

    bool isEqualResult(Ref<FinderPatternInfo> src, Ref<FinderPatternInfo> dst);

    // float getEstimatedVerticalModuleSize(int* stateCount_, CrossCheckState
    // checkState_);

    /** stateCount must be int[5] */
    bool handlePossibleCenter(int* stateCount, size_t i, size_t j);
    int getMaxMinModuleSize(float& minModuleSize, float& maxModuleSize);
    int findRowSkip();

    // std::vector<Ref<FinderPattern> > selectBestPatterns();
    std::vector<Ref<FinderPattern> > selectBestPatterns(ErrorHandler& err_handler);
    std::vector<Ref<FinderPattern> > selectFileBestPatterns(ErrorHandler& err_handler);
    std::vector<Ref<FinderPattern> > orderBestPatterns(std::vector<Ref<FinderPattern> > patterns);

    vector<Ref<FinderPatternInfo> > getPatternInfosFileMode(DecodeHints const& hints,
                                                            ErrorHandler& err_handler);

    bool IsPossibleFindPatterInfo(Ref<FinderPattern> a, Ref<FinderPattern> b, Ref<FinderPattern> c);
    void PushToResult(Ref<FinderPattern> a, Ref<FinderPattern> b, Ref<FinderPattern> c,
                      vector<Ref<FinderPatternInfo> >& patternInfos);

    Ref<BitMatrix> getImage();
    std::vector<Ref<FinderPattern> >& getPossibleCenters();

public:
    void InitConfig();
    float distance(Ref<ResultPoint> p1, Ref<ResultPoint> p2);
    FinderPatternFinder(Ref<BitMatrix> image, Ref<UnicomBlock> block);

    std::vector<Ref<FinderPatternInfo> > find(DecodeHints const& hints, ErrorHandler& err_handler);
    void setFinderLoose(bool loose) { finderConditionLoose_ = loose; }
    bool getFinderLoose() { return finderConditionLoose_; }

    bool checkIsoscelesRightTriangle(Ref<FinderPattern> centerA, Ref<FinderPattern> centerB,
                                     Ref<FinderPattern> centerC, float& longSide);
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DETECTOR_FINDER_PATTERN_FINDER_HPP_
