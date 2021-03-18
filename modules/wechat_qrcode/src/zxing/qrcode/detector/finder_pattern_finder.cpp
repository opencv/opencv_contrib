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
#include "finder_pattern_finder.hpp"
#include "../../common/kmeans.hpp"
#include "../../common/mathutils.hpp"
#include "../../decodehints.hpp"
#include "../../errorhandler.hpp"

using zxing::Ref;
using zxing::qrcode::FinderPattern;
using zxing::qrcode::FinderPatternFinder;
using zxing::qrcode::FinderPatternInfo;

// VC++

using zxing::BitMatrix;
using zxing::DecodeHints;
using zxing::ResultPoint;

namespace zxing {

namespace qrcode {


namespace {
class FurthestFromAverageComparator {
private:
    const float averageModuleSize_;

public:
    explicit FurthestFromAverageComparator(float averageModuleSize)
        : averageModuleSize_(averageModuleSize) {}
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b) {
        float dA = abs(a->getEstimatedModuleSize() - averageModuleSize_);
        float dB = abs(b->getEstimatedModuleSize() - averageModuleSize_);
        return dA > dB;
    }
};

// Orders by furthes from average
class CenterComparator {
    const float averageModuleSize_;

public:
    explicit CenterComparator(float averageModuleSize) : averageModuleSize_(averageModuleSize) {}
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b) {
        // N.B.: we want the result in descending order ...
        if (a->getCount() != b->getCount()) {
            return a->getCount() > b->getCount();
        } else {
            float dA = abs(a->getEstimatedModuleSize() - averageModuleSize_);
            float dB = abs(b->getEstimatedModuleSize() - averageModuleSize_);
            return dA < dB;
        }
    }
};

class CountComparator {
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b) {
        return a->getCount() > b->getCount();
    }
};

class ModuleSizeComparator {
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b) {
        return a->getEstimatedModuleSize() > b->getEstimatedModuleSize();
    }
};

class BestComparator {
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b) {
        if (a->getCount() != b->getCount()) {
            return a->getCount() > b->getCount();
        } else {
            return a->getEstimatedModuleSize() > b->getEstimatedModuleSize();
        }
    }
};
class BestComparator2 {
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b) {
        if (a->getCount() != b->getCount()) {
            return a->getCount() > b->getCount();
        } else {
            int aErr = 0, bErr = 0;
            if (a->getHorizontalCheckState() != FinderPattern::HORIZONTAL_STATE_NORMAL) aErr++;
            if (a->getVerticalCheckState() != FinderPattern::VERTICAL_STATE_NORMAL) aErr++;
            if (b->getHorizontalCheckState() != FinderPattern::HORIZONTAL_STATE_NORMAL) bErr++;
            if (b->getVerticalCheckState() != FinderPattern::VERTICAL_STATE_NORMAL) bErr++;

            if (aErr != bErr) {
                return aErr < bErr;
            } else {
                return a->getEstimatedModuleSize() > b->getEstimatedModuleSize();
            }
        }
    }
};

class XComparator {
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b) { return a->getX() < b->getX(); }
};

class YComparator {
public:
    bool operator()(Ref<FinderPattern> a, Ref<FinderPattern> b) { return a->getY() < b->getY(); }
};

}  // namespace

int FinderPatternFinder::CENTER_QUORUM = 2;
int FinderPatternFinder::MIN_SKIP = 1;       // 1 pixel/module times MIN_SKIP modules/center
int FinderPatternFinder::MAX_MODULES = 177;  // support up to version 40  which has 177 modules
int FinderPatternFinder::INTEGER_MATH_SHIFT = 8;
int FinderPatternFinder::FP_INPUT_CNN_MAX_NUM = 10;
int FinderPatternFinder::FP_IS_SELECT_BEST = 1;
int FinderPatternFinder::FP_IS_SELECT_FILE_BEST = 1;
int FinderPatternFinder::FP_INPUT_MAX_NUM = 100;
int FinderPatternFinder::FP_FILTER_SIZE = 100;
int FinderPatternFinder::FPS_CLUSTER_MAX = 4;
int FinderPatternFinder::FPS_RESULT_MAX = 12;
int FinderPatternFinder::K_FACTOR = 2;

float FinderPatternFinder::FPS_MS_VAL = 1.0f;
float FinderPatternFinder::FP_COUNT_MIN = 2.0f;
float FinderPatternFinder::FP_MS_MIN = 1.0f;
float FinderPatternFinder::FP_RIGHT_ANGLE = 0.342f;
float FinderPatternFinder::FP_SMALL_ANGLE1 = 0.8191f;
float FinderPatternFinder::FP_SMALL_ANGLE2 = 0.5736f;
float FinderPatternFinder::QR_MIN_FP_AREA_ERR = 3;
float FinderPatternFinder::QR_MIN_FP_MS_ERR = 1;
int FinderPatternFinder::QR_MIN_FP_ACCEPT = 4;

std::vector<Ref<FinderPatternInfo>> FinderPatternFinder::find(DecodeHints const& hints,
                                                              ErrorHandler& err_handler) {
    bool tryHarder = true;

    size_t maxI = image_->getHeight();
    size_t maxJ = image_->getWidth();
    // Init pre check result
    _horizontalCheckedResult.clear();
    _horizontalCheckedResult.resize(maxJ);
    // As this is used often, we use an integer array instead of vector
    int stateCount[5];

    // Let's assume that the maximum version QR Code we support
    // (Version 40, 177modules, and finder pattern start at: 0~7) takes up 1/4
    // the height of the image, and then account for the center being 3
    // modules in size. This gives the smallest number of pixels the center
    // could be, so skip this often. When trying harder, look for all
    // QR versions regardless of how dense they are.
    int iSkip = (3 * maxI) / (4 * MAX_MODULES);
    if (iSkip < MIN_SKIP || tryHarder) {
        iSkip = MIN_SKIP;
    }

    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix& matrix = *image_;

    // If we need to use getRowRecords or getRowCounterOffsetEnd, we should call
    // initRowCounters first
    matrix.initRowCounters();

    // scan line algorithm
    for (size_t i = iSkip - 1; i < maxI; i += iSkip) {
        COUNTER_TYPE* irow_states = matrix.getRowRecords(i);
        COUNTER_TYPE* irow_offsets = matrix.getRowRecordsOffset(i);

        size_t rj = matrix.getRowFirstIsWhite(i) ? 1 : 0;
        COUNTER_TYPE row_counter_width = matrix.getRowCounterOffsetEnd(i);
        // because the rj is black, rj+1 must be white, so we can skip it by +2
        for (; (rj + 4) < size_t(row_counter_width) && (rj + 4) < maxJ; rj += 2) {
            stateCount[0] = irow_states[rj];
            stateCount[1] = irow_states[rj + 1];
            stateCount[2] = irow_states[rj + 2];
            stateCount[3] = irow_states[rj + 3];
            stateCount[4] = irow_states[rj + 4];

            size_t j = irow_offsets[rj + 4] + stateCount[4];
            if (j > maxJ) {
                rj = row_counter_width - 1;
                continue;
            }
            if (foundPatternCross(stateCount)) {
                if (j == maxJ) {
                    // check whether it is the "true" central
                    bool confirmed = handlePossibleCenter(stateCount, i, maxJ);
                    if (confirmed) {
                        iSkip = int(possibleCenters_.back()->getEstimatedModuleSize());
                        if (iSkip < 1) iSkip = 1;
                    }
                    rj = row_counter_width - 1;
                    continue;
                } else {
                    bool confirmed = handlePossibleCenter(stateCount, i, j);
                    if (confirmed) {
                        // Start examining every other line. Checking each line
                        // turned out to be too expensive and didn't improve
                        // performance.
                        iSkip = 2;
                        if (!hasSkipped_) {
                            int rowSkip = findRowSkip();
                            if (rowSkip > stateCount[2]) {
                                // Skip rows between row of lower confirmed
                                // center and top of presumed third confirmed
                                // center but back up a bit to get a full chance
                                // of detecting it, entire width of center of
                                // finder pattern Skip by rowSkip, but back off
                                // by stateCount[2] (size of last center of
                                // pattern we saw) to be conservative, and also
                                // back off by iSkip which is about to be
                                // re-added
                                i += rowSkip - stateCount[2] - iSkip;
                                rj = row_counter_width - 1;
                                j = maxJ - 1;
                            }
                        }
                    } else {
                        continue;
                    }
                    rj += 4;
                }
            }
        }
    }
    // use connected cells algorithm
    {
        for (size_t i = iSkip - 1; i < maxI; i += iSkip) {
            COUNTER_TYPE* irow_states = matrix.getRowRecords(i);
            COUNTER_TYPE* irow_offsets = matrix.getRowRecordsOffset(i);
            COUNTER_TYPE row_counter_width = matrix.getRowCounterOffsetEnd(i);

            for (size_t rj = matrix.getRowFirstIsWhite(i) ? 1 : 0;
                 (rj + 4) < size_t(row_counter_width); rj += 2) {
                if (block_->GetUnicomBlockIndex(i, irow_offsets[rj]) ==
                        block_->GetUnicomBlockIndex(i, irow_offsets[rj + 4]) &&
                    block_->GetUnicomBlockIndex(i, irow_offsets[rj + 1]) ==
                        block_->GetUnicomBlockIndex(i, irow_offsets[rj + 3]) &&
                    block_->GetUnicomBlockIndex(i, irow_offsets[rj]) !=
                        block_->GetUnicomBlockIndex(i, irow_offsets[rj + 2])) {
                    const int iBlackCir = block_->GetUnicomBlockSize(i, irow_offsets[rj]);
                    const int iWhiteCir = block_->GetUnicomBlockSize(i, irow_offsets[rj + 1]);
                    const int iBlackPnt = block_->GetUnicomBlockSize(i, irow_offsets[rj + 2]);

                    if (-1 == iBlackCir || -1 == iWhiteCir) continue;

                    const float fBlackCir = sqrt(iBlackCir / 24.0);
                    const float fWhiteCir = sqrt(iWhiteCir / 16.0);
                    const float fBlackPnt = sqrt(iBlackPnt / 9.0);

                    // use center 1:3:1, because the border may be padded.
                    // a plan for MS
                    const float fRealMS = sqrt((iWhiteCir + iBlackPnt) / 25.0);

                    // b plan for MS
                    int iTotalCount = 0;
                    for (int j = 1; j < 4; ++j) iTotalCount += irow_states[rj + j];
                    const float fEstRowMS = iTotalCount / 5.0;

                    if (fabs(fBlackCir - fWhiteCir) <= QR_MIN_FP_AREA_ERR &&
                        fabs(fWhiteCir - fBlackPnt) <= QR_MIN_FP_AREA_ERR &&
                        fabs(fRealMS - fEstRowMS) < QR_MIN_FP_MS_ERR) {
                        int centerI = 0;
                        int centerJ = 0;
                        if (fRealMS < QR_MIN_FP_ACCEPT) {
                            centerI = i;
                            centerJ = irow_offsets[rj + 2] + irow_states[rj + 2] / 2;
                        } else {
                            int iMinX = 0, iMinY = 0, iMaxX = 0, iMaxY = 0;
                            block_->GetMinPoint(i, irow_offsets[rj + 1], iMinY, iMinX);
                            block_->GetMaxPoint(i, irow_offsets[rj + 3], iMaxY, iMaxX);
                            centerI = (iMaxY + iMinY) / 2.0;  // y
                            centerJ = (iMaxX + iMinX) / 2.0;  // x
                        }
                        tryToPushToCenters(centerI, centerJ, fRealMS);
                        int rowSkip = findRowSkip();
                        if (rowSkip > irow_states[rj + 2]) {
                            // Skip rows between row of lower confirmed center
                            // and top of presumed third confirmed center but
                            // back up a bit to get a full chance of detecting
                            // it, entire width of center of finder pattern Skip
                            // by rowSkip, but back off by stateCount[2] (size
                            // of last center of pattern we saw) to be
                            // conservative, and also back off by iSkip which is
                            // about to be re-added
                            i += rowSkip - irow_states[rj + 2] - iSkip;
                            rj = row_counter_width - 1;
                        }
                        rj += 4;
                    }
                }
            }
        }
    }

    vector<Ref<FinderPatternInfo>> patternInfos = getPatternInfosFileMode(hints, err_handler);
    if (err_handler.ErrCode()) {
        return std::vector<Ref<FinderPatternInfo>>();
    }
    // sort with score
    sort(patternInfos.begin(), patternInfos.end(),
         [](Ref<FinderPatternInfo> a, Ref<FinderPatternInfo> b) {
             return a->getPossibleFix() > b->getPossibleFix();
         });

    return patternInfos;
}

bool FinderPatternFinder::tryToPushToCenters(float centerI, float centerJ,
                                             float estimatedModuleSize,
                                             CrossCheckState horizontalState,
                                             CrossCheckState verticalState) {
    for (size_t index = 0; index < possibleCenters_.size(); index++) {
        Ref<FinderPattern> center = possibleCenters_[index];
        // Look for about the same center and module size:
        if (center->aboutEquals(estimatedModuleSize, centerI, centerJ)) {
            possibleCenters_[index] =
                center->combineEstimate(centerI, centerJ, estimatedModuleSize);
            possibleCenters_[index]->setHorizontalCheckState(
                horizontalState == FinderPatternFinder::NORMAL ? center->getHorizontalCheckState()
                                                               : horizontalState);
            possibleCenters_[index]->setVerticalCheckState(
                verticalState == FinderPatternFinder::NORMAL ? center->getVerticalCheckState()
                                                             : verticalState);
            return false;
        }
    }
    Ref<FinderPattern> newPattern(new FinderPattern(centerJ, centerI, estimatedModuleSize));
    newPattern->setHorizontalCheckState(horizontalState);
    newPattern->setVerticalCheckState(verticalState);
    possibleCenters_.push_back(newPattern);
    return true;
}

bool FinderPatternFinder::isEqualResult(Ref<FinderPatternInfo> src, Ref<FinderPatternInfo> dst) {
    if (src == NULL) {
        return false;
    }

    if (dst == NULL) {
        return true;
    }

    auto topLeft = src->getTopLeft();
    auto bottomLeft = src->getBottomLeft();
    auto topRight = src->getTopRight();

    return topLeft->aboutEquals(1.0, dst->getTopLeft()->getY(), dst->getTopLeft()->getX()) &&
           bottomLeft->aboutEquals(1.0, dst->getBottomLeft()->getY(),
                                   dst->getBottomLeft()->getX()) &&
           topRight->aboutEquals(1.0, dst->getTopRight()->getY(), dst->getTopRight()->getX());
}

bool FinderPatternFinder::IsPossibleFindPatterInfo(Ref<FinderPattern> a, Ref<FinderPattern> b,
                                                   Ref<FinderPattern> c) {
    // check variance
    float aMs = a->getEstimatedModuleSize();
    float bMs = b->getEstimatedModuleSize();
    float cMs = c->getEstimatedModuleSize();

    float avg = (aMs + bMs + cMs) / 3.0;
    float val =
        sqrt((aMs - avg) * (aMs - avg) + (bMs - avg) * (bMs - avg) + (cMs - avg) * (cMs - avg));

    if (val >= FPS_MS_VAL) return false;

    float longSize = 0.0;

    return checkIsoscelesRightTriangle(a, b, c, longSize);
}

void FinderPatternFinder::PushToResult(Ref<FinderPattern> a, Ref<FinderPattern> b,
                                       Ref<FinderPattern> c,
                                       vector<Ref<FinderPatternInfo>>& patternInfos) {
    vector<Ref<FinderPattern>> finderPatterns;
    finderPatterns.push_back(a);
    finderPatterns.push_back(b);
    finderPatterns.push_back(c);
    vector<Ref<FinderPattern>> finderPattern = orderBestPatterns(finderPatterns);

    Ref<FinderPatternInfo> patternInfo(new FinderPatternInfo(finderPattern));

    for (size_t j = 0; j < patternInfos.size(); j++) {
        if (isEqualResult(patternInfos[j], patternInfo)) {
            return;
        }
    }
    patternInfos.push_back(patternInfo);
}

vector<Ref<FinderPatternInfo>> FinderPatternFinder::getPatternInfosFileMode(
    DecodeHints const& hints, ErrorHandler& err_handler) {
    size_t startSize = possibleCenters_.size();

    if (startSize < 3) {
        // Couldn't find enough finder patterns
        err_handler = ReaderErrorHandler("Could not find three finder patterns");
        return vector<Ref<FinderPatternInfo>>();
    }

    std::vector<Ref<FinderPatternInfo>> patternInfos;

    if (startSize == 3) {
        PushToResult(possibleCenters_[0], possibleCenters_[1], possibleCenters_[2], patternInfos);
        return patternInfos;
    }

    vector<Ref<FinderPattern>> finderPatterns;
    Ref<FinderPatternInfo> resultBest;

    // select best
    if (FP_IS_SELECT_BEST) {
        finderPatterns = selectBestPatterns(err_handler);
        if (err_handler.ErrCode() == 0)
            PushToResult(finderPatterns[0], finderPatterns[1], finderPatterns[2], patternInfos);
    }

    if (FP_IS_SELECT_FILE_BEST) {
        finderPatterns = selectFileBestPatterns(err_handler);
        if (err_handler.ErrCode() == 0)
            PushToResult(finderPatterns[0], finderPatterns[1], finderPatterns[2], patternInfos);
    }

    // sort and filter
    sort(possibleCenters_.begin(), possibleCenters_.end(), ModuleSizeComparator());
    std::vector<Ref<FinderPattern>> standardCenters;

    for (size_t i = 0; i < possibleCenters_.size(); i++) {
        if (possibleCenters_[i]->getEstimatedModuleSize() >= FP_MS_MIN &&
            possibleCenters_[i]->getCount() >= FP_COUNT_MIN) {
            standardCenters.push_back(possibleCenters_[i]);
            if (standardCenters.size() >= size_t(FP_INPUT_MAX_NUM)) break;
            if (hints.getUseNNDetector() && standardCenters.size() >= size_t(FP_INPUT_CNN_MAX_NUM))
                break;
        }
    }

    if (standardCenters.size() < 3) {
        err_handler = ReaderErrorHandler("Could not find three finder patterns");
        return vector<Ref<FinderPatternInfo>>();
    }

    if (standardCenters.size() <= size_t(FP_INPUT_CNN_MAX_NUM)) {
        for (size_t x = 0; x < standardCenters.size(); x++) {
            for (size_t y = x + 1; y < standardCenters.size(); y++) {
                for (size_t z = y + 1; z < standardCenters.size(); z++) {
                    bool check_result = IsPossibleFindPatterInfo(
                        standardCenters[x], standardCenters[y], standardCenters[z]);
                    if (check_result) {
                        PushToResult(standardCenters[x], standardCenters[y], standardCenters[z],
                                     patternInfos);
                    }
                }
            }
        }
        return patternInfos;
    }

    // Kmeans
    const int maxepoches = 100;
    const int minchanged = 0;
    // calculate K
    int k = log(float(standardCenters.size())) * K_FACTOR - 1;
    if (k < 1) k = 1;

    vector<vector<double>> trainX;
    for (size_t i = 0; i < standardCenters.size(); i++) {
        vector<double> tmp;
        tmp.push_back(standardCenters[i]->getCount());
        tmp.push_back(standardCenters[i]->getEstimatedModuleSize());
        trainX.push_back(tmp);
    }

    vector<Cluster> clusters_out = k_means(trainX, k, maxepoches, minchanged);

    for (size_t i = 0; i < clusters_out.size(); i++) {
        int cluster_select = 0;

        if (clusters_out[i].samples.size() < 3) {
            if (i < clusters_out.size() - 1 && clusters_out[i + 1].samples.size() < 3) {
                for (size_t j = 0; j < clusters_out[i].samples.size(); j++)
                    clusters_out[i + 1].samples.push_back(clusters_out[i].samples[j]);
            }
            continue;
        }

        vector<Ref<FinderPattern>> clusterPatterns;
        for (size_t j = 0; j < clusters_out[i].samples.size(); j++) {
            clusterPatterns.push_back(standardCenters[clusters_out[i].samples[j]]);
        }

        sort(clusterPatterns.begin(), clusterPatterns.end(), BestComparator2());

        for (size_t x = 0;
             x < clusters_out[i].samples.size() && cluster_select <= FPS_CLUSTER_MAX &&
             patternInfos.size() <= size_t(FPS_RESULT_MAX);
             x++) {
            for (size_t y = x + 1;
                 y < clusters_out[i].samples.size() && cluster_select <= FPS_CLUSTER_MAX &&
                 patternInfos.size() <= size_t(FPS_RESULT_MAX);
                 y++) {
                for (size_t z = y + 1;
                     z < clusters_out[i].samples.size() && cluster_select <= FPS_CLUSTER_MAX &&
                     patternInfos.size() <= size_t(FPS_RESULT_MAX);
                     z++) {
                    bool check_result = IsPossibleFindPatterInfo(
                        clusterPatterns[x], clusterPatterns[y], clusterPatterns[z]);
                    if (check_result) {
                        PushToResult(clusterPatterns[x], clusterPatterns[y], clusterPatterns[z],
                                     patternInfos);
                        cluster_select++;
                    }
                }
            }
        }
    }
    return patternInfos;
}

// Given a count of black/white/black/white/black pixels just seen and an end
// position, figures the location of the center of this run.
float FinderPatternFinder::centerFromEnd(int* stateCount, int end) {
    // calculate the center by pattern 1:3:1 is better than pattern 3
    // because the finder pattern is irregular in some case
    return (float)(end - stateCount[4]) - (stateCount[3] + stateCount[2] + stateCount[1]) / 2.0f;
}

// return if the proportions of the counts is close enough to 1/1/3/1/1 ratios
// used by finder patterns to be considered a match
bool FinderPatternFinder::foundPatternCross(int* stateCount) {
    int totalModuleSize = 0;

    int stateCountINT[5];

    int minModuleSizeINT = 3;
    minModuleSizeINT <<= INTEGER_MATH_SHIFT;

    for (int i = 0; i < 5; i++) {
        if (stateCount[i] <= 0) {
            return false;
        }
        stateCountINT[i] = stateCount[i] << INTEGER_MATH_SHIFT;
        totalModuleSize += stateCount[i];
    }
    if (totalModuleSize < 7) {
        return false;
    }

    CURRENT_CHECK_STATE = FinderPatternFinder::NOT_PATTERN;

    totalModuleSize = totalModuleSize << INTEGER_MATH_SHIFT;

    // Newer version to check 1 time, use 3 points
    int moduleSize = ((totalModuleSize - stateCountINT[0] - stateCountINT[4])) / 5;

    int maxVariance = moduleSize;

    if (moduleSize > minModuleSizeINT) maxVariance = moduleSize / 2;

    int startCountINT = stateCountINT[0];
    int endCountINT = stateCountINT[4];

    bool leftFit = (abs(moduleSize - startCountINT) <= maxVariance);
    bool rightFit = (abs(moduleSize - endCountINT) <= maxVariance);

    if (leftFit) {
        if (rightFit) {
            moduleSize = totalModuleSize / 7;
            CURRENT_CHECK_STATE = FinderPatternFinder::NORMAL;
        } else {
            moduleSize = (totalModuleSize - stateCountINT[4]) / 6;
            CURRENT_CHECK_STATE = FinderPatternFinder::RIHGT_SPILL;
        }
    } else {
        if (rightFit) {
            moduleSize = (totalModuleSize - stateCountINT[0]) / 6;
            CURRENT_CHECK_STATE = FinderPatternFinder::LEFT_SPILL;
        } else {
            // return false;
            CURRENT_CHECK_STATE = FinderPatternFinder::LEFT_RIGHT_SPILL;
        }
    }

    // 1:1:3:1:1 || n:1:3:1:1 || 1:1:3:1:n
    if (abs(moduleSize - stateCountINT[1]) <= maxVariance &&
        abs(3 * moduleSize - stateCountINT[2]) <= 3 * maxVariance &&
        abs(moduleSize - stateCountINT[3]) <= maxVariance) {
        return true;
    }
    return false;
}

int FinderPatternFinder::getMinModuleSize() {
    int minModuleSize = (3 * min(image_->getWidth(), image_->getHeight())) / (4 * MAX_MODULES);

    if (minModuleSize < MIN_SKIP) {
        minModuleSize = MIN_SKIP;
    }

    return minModuleSize;
}

/**
 * After a vertical and horizontal scan finds a potential finder pattern, this
 * method "cross-cross-cross-checks" by scanning down diagonally through the
 * center of the possible finder pattern to see if the same proportion is
 * detected.
 *
 * @param startI row where a finder pattern was detected
 * @param centerJ center of the section that appears to cross a finder pattern
 * @param maxCount maximum reasonable number of modules that should be
 *  observed in any reading state, based on the results of the horizontal scan
 * @param originalStateCountTotal The original state count total.
 * @return true if proportions are withing expected limits
 */
bool FinderPatternFinder::crossCheckDiagonal(int startI, int centerJ, int maxCount,
                                             int originalStateCountTotal) {
    int maxI = image_->getHeight();
    int maxJ = image_->getWidth();

    if ((startI < 0) || (startI > maxI - 1) || (centerJ < 0) || (centerJ > maxJ - 1)) {
        return false;
    }

    int stateCount[5];
    stateCount[0] = 0;
    stateCount[1] = 0;
    stateCount[2] = 0;
    stateCount[3] = 0;
    stateCount[4] = 0;

    if (!image_->get(centerJ, startI)) {
        if (startI + 1 < maxI && image_->get(centerJ, startI + 1))
            startI = startI + 1;
        else if (0 < startI - 1 && image_->get(centerJ, startI - 1))
            startI = startI - 1;
        else
            return false;
    }

    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix& matrix = *image_;

    // Start counting up, left from center finding black center mass
    int i = 0;
    // Fix possible crash 20140418
    // while (startI - i >= 0 && image.get(centerJ - i, startI - i)) {
    while ((startI - i >= 0) && (centerJ - i >= 0) && matrix.get(centerJ - i, startI - i)) {
        stateCount[2]++;
        i++;
    }

    if ((startI - i < 0) || (centerJ - i < 0)) {
        return false;
    }

    // Continue up, left finding white space
    while ((startI - i >= 0) && (centerJ - i >= 0) && !matrix.get(centerJ - i, startI - i) &&
           stateCount[1] <= maxCount) {
        stateCount[1]++;
        i++;
    }

    // If already too many modules in this state or ran off the edge:
    if ((startI - i < 0) || (centerJ - i < 0) || stateCount[1] > maxCount) {
        return false;
    }

    CrossCheckState tmpCheckState = FinderPatternFinder::NORMAL;

    // Continue up, left finding black border
    while ((startI - i >= 0) && (centerJ - i >= 0) && matrix.get(centerJ - i, startI - i) &&
           stateCount[0] <= maxCount) {
        stateCount[0]++;
        i++;
    }

    if (stateCount[0] >= maxCount) {
        tmpCheckState = FinderPatternFinder::LEFT_SPILL;
    }

    // Now also count down, right from center
    i = 1;
    while ((startI + i < maxI) && (centerJ + i < maxJ) && matrix.get(centerJ + i, startI + i)) {
        stateCount[2]++;
        i++;
    }

    // Ran off the edge?
    if ((startI + i >= maxI) || (centerJ + i >= maxJ)) {
        return false;
    }

    while ((startI + i < maxI) && (centerJ + i < maxJ) && !matrix.get(centerJ + i, startI + i) &&
           stateCount[3] < maxCount) {
        stateCount[3]++;
        i++;
    }

    if ((startI + i >= maxI) || (centerJ + i >= maxJ) || stateCount[3] >= maxCount) {
        return false;
    }

    while ((startI + i < maxI) && (centerJ + i < maxJ) && matrix.get(centerJ + i, startI + i) &&
           stateCount[4] < maxCount) {
        stateCount[4]++;
        i++;
    }

    if (stateCount[4] >= maxCount) {
        tmpCheckState = tmpCheckState == FinderPatternFinder::LEFT_SPILL
                            ? FinderPatternFinder::LEFT_RIGHT_SPILL
                            : FinderPatternFinder::RIHGT_SPILL;
    }

    bool diagonal_check = foundPatternCross(stateCount);
    if (!diagonal_check) return false;

    if (CURRENT_CHECK_STATE == FinderPatternFinder::LEFT_SPILL &&
        tmpCheckState == FinderPatternFinder::RIHGT_SPILL)
        return false;

    if (CURRENT_CHECK_STATE == FinderPatternFinder::RIHGT_SPILL &&
        tmpCheckState == FinderPatternFinder::LEFT_SPILL)
        return false;

    int stateCountTotal = getStateCountTotal(stateCount, CURRENT_CHECK_STATE);

    if (abs(stateCountTotal - originalStateCountTotal) < 2 * originalStateCountTotal) {
        return true;
    } else {
        return false;
    }
}

int FinderPatternFinder::getStateCountTotal(int* stateCount, const CrossCheckState& check_state) {
    int stateCountTotal = stateCount[1] + stateCount[2] + stateCount[3];
    if (check_state == FinderPatternFinder::NORMAL) {
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[4];
    } else if (check_state == FinderPatternFinder::LEFT_SPILL) {
        stateCountTotal = stateCountTotal + stateCount[1] + stateCount[4];
    } else if (check_state == FinderPatternFinder::RIHGT_SPILL) {
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[3];
    } else if (check_state == FinderPatternFinder::LEFT_RIGHT_SPILL) {
        stateCountTotal = stateCountTotal + stateCount[1] + stateCount[3];
    }
    return stateCountTotal;
}
// After a horizontal scan finds a potential finder pattern, this method
// "cross-checks" by scanning down vertically through the center of the possible
// finder pattern to see if the same proportion is detected.
float FinderPatternFinder::crossCheckVertical(size_t startI, size_t centerJ, int maxCount,
                                              int originalStateCountTotal,
                                              float& estimatedVerticalModuleSize) {
    int maxI = image_->getHeight();

    int stateCount[5];
    for (int i = 0; i < 5; i++) stateCount[i] = 0;

    if (!image_->get(centerJ, startI)) {
        if ((int)startI + 1 < maxI && image_->get(centerJ, startI + 1))
            startI = startI + 1;
        else if (0 < (int)startI - 1 && image_->get(centerJ, startI - 1))
            startI = startI - 1;
        else
            return nan();
    }

    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix& matrix = *image_;

    bool* imageRow0 = matrix.getRowBoolPtr(0);
    bool* p = imageRow0;
    int imgWidth = matrix.getWidth();

    // Start counting up from center
    int ii = startI;

    p = imageRow0 + ii * imgWidth + centerJ;

    while (ii >= 0 && *p) {
        stateCount[2]++;
        ii--;
        p -= imgWidth;
    }
    if (ii < 0) {
        return nan();
    }
    while (ii >= 0 && !*p && stateCount[1] <= maxCount) {
        stateCount[1]++;
        ii--;
        p -= imgWidth;
    }
    // If already too many modules in this state or ran off the edge:
    if (ii < 0 || stateCount[1] > maxCount) {
        return nan();
    }

    CrossCheckState tmpCheckState = FinderPatternFinder::NORMAL;

    while (ii >= 0 && *p /*&& stateCount[0] <= maxCount*/) {  // n:1:3:1:1
        stateCount[0]++;
        ii--;
        p -= imgWidth;
    }

    if (stateCount[0] >= maxCount) {
        tmpCheckState = FinderPatternFinder::LEFT_SPILL;
    }

    // Now also count down from center
    ii = startI + 1;

    p = imageRow0 + ii * imgWidth + centerJ;

    while (ii < maxI && *p) {  // 1:1:"3":1:1
                               // while (ii < maxI && matrix.get(centerJ, ii)) {
        stateCount[2]++;
        ii++;

        p += imgWidth;
    }
    if (ii == maxI) {
        return nan();
    }
    while (ii < maxI && !*p && stateCount[3] < maxCount) {  // 1:1:3:"1":1
        stateCount[3]++;
        ii++;

        p += imgWidth;
    }
    if (ii == maxI || stateCount[3] >= maxCount) {
        return nan();
    }

    if (tmpCheckState == FinderPatternFinder::LEFT_SPILL) {
        while (ii < maxI && *p && stateCount[4] < maxCount) {  // 1:1:3:1:"1"
            stateCount[4]++;
            ii++;

            p += imgWidth;
        }
        if (stateCount[4] >= maxCount) {
            return nan();
        }
    } else {  // 1:1:3:1:"n"
        while (ii < maxI && *p) {
            stateCount[4]++;
            ii++;

            p += imgWidth;
        }
        if (stateCount[4] >= maxCount) {
            tmpCheckState = FinderPatternFinder::RIHGT_SPILL;
        }
    }

    bool vertical_check = foundPatternCross(stateCount);
    if (!vertical_check) return nan();

    if ((CURRENT_CHECK_STATE == FinderPatternFinder::LEFT_SPILL &&
         tmpCheckState == FinderPatternFinder::RIHGT_SPILL) ||
        (CURRENT_CHECK_STATE == FinderPatternFinder::RIHGT_SPILL &&
         tmpCheckState == FinderPatternFinder::LEFT_SPILL)) {
        return nan();
    }

    int stateCountTotal = getStateCountTotal(stateCount, CURRENT_CHECK_STATE);

    // If we found a finder-pattern-like section, but its size is more than 40%
    // different than the original, assume it's a false positive
    if (5 * abs(stateCountTotal - originalStateCountTotal) >= 2 * originalStateCountTotal) {
        return nan();
    }

    estimatedVerticalModuleSize = (float)stateCountTotal / 7.0f;

    return centerFromEnd(stateCount, ii);
}

// Like #crossCheckVertical(), and in fact is basically identical,
// except it reads horizontally instead of vertically. This is used to
// cross-cross check a vertical cross check and locate the real center of the
// alignment pattern.
float FinderPatternFinder::crossCheckHorizontal(size_t startJ, size_t centerI, int maxCount,
                                                int originalStateCountTotal,
                                                float& estimatedHorizontalModuleSize) {
    int maxJ = image_->getWidth();

    int stateCount[5];
    for (int i = 0; i < 5; i++) stateCount[i] = 0;

    if (!image_->get(startJ, centerI)) {
        if ((int)startJ + 1 < maxJ && image_->get(startJ + 1, centerI))
            startJ = startJ + 1;
        else if (0 < (int)startJ - 1 && image_->get(startJ - 1, centerI))
            startJ = startJ - 1;
        else
            return nan();
    }

    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix& matrix = *image_;
    int j = startJ;

    bool* centerIrow = NULL;

    centerIrow = matrix.getRowBoolPtr(centerI);

    //	while (j >= 0 &&matrix.get(j, centerI)) {
    while (j >= 0 && centerIrow[j]) {
        stateCount[2]++;
        j--;
    }
    if (j < 0) {
        return nan();
    }
    while (j >= 0 && !centerIrow[j] && stateCount[1] <= maxCount) {
        stateCount[1]++;
        j--;
    }
    if (j < 0 || stateCount[1] > maxCount) {
        return nan();
    }
    CrossCheckState tmpCheckState = FinderPatternFinder::NORMAL;

    while (j >= 0 && centerIrow[j] /* && stateCount[0] <= maxCount*/) {
        stateCount[0]++;
        j--;
    }
    if (stateCount[0] >= maxCount) {
        tmpCheckState = FinderPatternFinder::LEFT_SPILL;
    }

    j = startJ + 1;
    while (j < maxJ && centerIrow[j]) {
        stateCount[2]++;
        j++;
    }
    if (j == maxJ) {
        return nan();
    }
    while (j < maxJ && !centerIrow[j] && stateCount[3] < maxCount) {
        stateCount[3]++;
        j++;
    }
    if (j == maxJ || stateCount[3] >= maxCount) {
        return nan();
    }

    if (tmpCheckState == LEFT_SPILL) {
        while (j < maxJ && centerIrow[j] && stateCount[4] <= maxCount) {
            stateCount[4]++;
            j++;
        }
        if (stateCount[4] >= maxCount) {
            return nan();
        }
    } else {
        while (j < maxJ && centerIrow[j]) {
            stateCount[4]++;
            j++;
        }
        if (stateCount[4] >= maxCount) {
            tmpCheckState = RIHGT_SPILL;
        }
    }

    while (j < maxJ && centerIrow[j] /*&& stateCount[4] < maxCount*/) {
        stateCount[4]++;
        j++;
    }

    // If we found a finder-pattern-like section, but its size is significantly
    // different than the original, assume it's a false positive
    // int stateCountTotal = stateCount[0] + stateCount[1] + stateCount[2] +
    // stateCount[3] + stateCount[4];
    bool horizontal_check = foundPatternCross(stateCount);
    if (!horizontal_check) return nan();

    /*
    if(tmpCheckState!=CURRENT_CHECK_STATE)
        return nan();*/

    // Cannot be a LEFT-RIGHT center
    if ((CURRENT_CHECK_STATE == FinderPatternFinder::LEFT_SPILL &&
         tmpCheckState == FinderPatternFinder::RIHGT_SPILL) ||
        (CURRENT_CHECK_STATE == FinderPatternFinder::RIHGT_SPILL &&
         tmpCheckState == FinderPatternFinder::LEFT_SPILL)) {
        return nan();
    }

    int stateCountTotal = getStateCountTotal(stateCount, CURRENT_CHECK_STATE);
    if (5 * abs(stateCountTotal - originalStateCountTotal) >= originalStateCountTotal) {
        return nan();
    }

    estimatedHorizontalModuleSize = (float)stateCountTotal / 7.0f;
    return centerFromEnd(stateCount, j);
}

float FinderPatternFinder::hasHorizontalCheckedResult(size_t startJ, size_t centerI) {
    for (size_t i = 0; i < _horizontalCheckedResult[startJ].size(); i++) {
        if (_horizontalCheckedResult[startJ][i].centerI == centerI) {
            return _horizontalCheckedResult[startJ][i].centerJ;
        }
    }

    return -1.0;
}

int FinderPatternFinder::addHorizontalCheckedResult(size_t startJ, size_t centerI, float centerJ) {
    HorizontalCheckedResult result;
    result.centerI = centerI;
    result.centerJ = centerJ;

    _horizontalCheckedResult[startJ].push_back(result);

    return 1;
}

#define CENTER_CHECK_TIME 3

/**
 * <p>This is called when a horizontal scan finds a possible alignment pattern.
 * It will cross check with a vertical scan, and if successful, will, ah,
 * cross-cross-check with another horizontal scan. This is needed primarily to
 * locate the real horizontal center of the pattern in cases of extreme skew.
 * And then we cross-cross-cross check with another diagonal scan.</p>
 *
 * <p>If that succeeds the finder pattern location is added to a list that
 * tracks the number of times each location has been nearly-matched as a finder
 * pattern. Each additional find is more evidence that the location is in fact a
 * finder pattern center
 *
 * @param stateCount reading state module counts from horizontal scan
 * @param i row where finder pattern may be found
 * @param j end of possible finder pattern in row
 * @return true if a finder pattern candidate was found this time
 */
bool FinderPatternFinder::handlePossibleCenter(int* stateCount, size_t i, size_t j) {
    CrossCheckState tmpHorizontalState = CURRENT_CHECK_STATE;
    float centerJ = centerFromEnd(stateCount, j);
    int stateCountTotal = stateCount[1] + stateCount[2] + stateCount[3];
    if (tmpHorizontalState == FinderPatternFinder::NORMAL) {
        // 1:1:3:1:1
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[4];
    } else if (tmpHorizontalState == FinderPatternFinder::LEFT_SPILL) {
        // n:1:3:1:1
        stateCountTotal = stateCountTotal + stateCount[1] + stateCount[4];
    } else if (tmpHorizontalState == FinderPatternFinder::RIHGT_SPILL) {
        // 1:1:3:1:n
        stateCountTotal = stateCountTotal + stateCount[0] + stateCount[3];
    }
    float estimatedHorizontalModuleSize = (float)stateCountTotal / 7.0f;

    float estimatedVerticalModuleSize;

    // try different size according to the estimatedHorizontalModuleSize
    float tolerateModuleSize =
        estimatedHorizontalModuleSize > 4.0 ? estimatedHorizontalModuleSize / 2.0f : 1.0f;
    float possbileCenterJs[7] = {centerJ,
                                 centerJ - tolerateModuleSize,
                                 centerJ + tolerateModuleSize,
                                 centerJ - 2 * tolerateModuleSize,
                                 centerJ + 2 * tolerateModuleSize,
                                 centerJ - 3 * tolerateModuleSize,
                                 centerJ + 3 * tolerateModuleSize};
    int image_height = image_->getHeight();
    int image_width = image_->getWidth();
    for (int k = 0; k < CENTER_CHECK_TIME; k++) {
        float possibleCenterJ = possbileCenterJs[k];
        if (possibleCenterJ < 0 || possibleCenterJ >= image_width) continue;
        float centerI = crossCheckVertical(i, (size_t)possibleCenterJ, stateCount[2],
                                           stateCountTotal, estimatedVerticalModuleSize);

        if (!isnan(centerI) && centerI >= 0.0) {
            CrossCheckState tmpVerticalState = CURRENT_CHECK_STATE;

            float moduleSizeDiff = abs(estimatedHorizontalModuleSize - estimatedVerticalModuleSize);

            if (moduleSizeDiff > estimatedHorizontalModuleSize ||
                moduleSizeDiff > estimatedVerticalModuleSize)
                return false;

            tolerateModuleSize =
                estimatedVerticalModuleSize > 4.0 ? estimatedVerticalModuleSize / 2.0f : 1.0f;

            float possbileCenterIs[7] = {centerI,
                                         centerI - tolerateModuleSize,
                                         centerI + tolerateModuleSize,
                                         centerI - 2 * tolerateModuleSize,
                                         centerI + 2 * tolerateModuleSize,
                                         centerI - 3 * tolerateModuleSize,
                                         centerI + 3 * tolerateModuleSize};

            for (int l = 0; l < CENTER_CHECK_TIME; l++) {
                float possibleCenterI = possbileCenterIs[l];
                if (possibleCenterI < 0 || possibleCenterI >= image_height) continue;
                // Re-cross check
                float reEstimatedHorizontalModuleSize;
                float cJ = hasHorizontalCheckedResult(centerJ, possibleCenterI);

                if (!isnan(cJ) && cJ >= 0.0) {
                    centerJ = cJ;
                } else {
                    cJ = centerJ;

                    float ccj =
                        crossCheckHorizontal((size_t)cJ, (size_t)possibleCenterI, stateCount[2],
                                             stateCountTotal, reEstimatedHorizontalModuleSize);

                    if (!isnan(ccj)) {
                        centerJ = ccj;
                        addHorizontalCheckedResult(cJ, possibleCenterI, ccj);
                    }
                }
                if (!isnan(centerJ)) {
                    tryToPushToCenters(
                        centerI, centerJ,
                        (estimatedHorizontalModuleSize + estimatedVerticalModuleSize) / 2.0,
                        tmpHorizontalState, tmpVerticalState);
                    return true;
                }
            }
        }
    }

    return false;
}

// return the number of rows we could safely skip during scanning, based on the
// first two finder patterns that have been located. In some cases their
// position will allow us to infer that the third pattern must lie below a
// certain point farther down the image.
int FinderPatternFinder::findRowSkip() {
    int max = possibleCenters_.size();
    if (max <= 1) {
        return 0;
    }

    if (max <= compared_finder_counts) return 0;

    Ref<FinderPattern> firstConfirmedCenter, secondConfirmedCenter;

    for (int i = 0; i < max - 1; i++) {
        firstConfirmedCenter = possibleCenters_[i];
        if (firstConfirmedCenter->getCount() >= CENTER_QUORUM) {
            float firstModuleSize = firstConfirmedCenter->getEstimatedModuleSize();
            int j_start = (i < compared_finder_counts) ? compared_finder_counts : i + 1;
            for (int j = j_start; j < max; j++) {
                secondConfirmedCenter = possibleCenters_[j];
                if (secondConfirmedCenter->getCount() >= CENTER_QUORUM) {
                    float secondModuleSize = secondConfirmedCenter->getEstimatedModuleSize();
                    float moduleSizeDiff = abs(firstModuleSize - secondModuleSize);
                    if (moduleSizeDiff < 1.0f) {
                        hasSkipped_ = true;
                        return (int)(abs(firstConfirmedCenter->getX() -
                                         secondConfirmedCenter->getX()) -
                                     abs(firstConfirmedCenter->getY() -
                                         secondConfirmedCenter->getY())) /
                               2;
                    }
                }
            }
        }
    }

    compared_finder_counts = max;

    return 0;
}

// return the 3 finder patterns from our list of candidates. The "best" are
// those that have been detected at least #CENTER_QUORUM times, and whose module
// size differs from the average among those patterns the least. //
vector<Ref<FinderPattern>> FinderPatternFinder::selectBestPatterns(ErrorHandler& err_handler) {
    size_t startSize = possibleCenters_.size();

    if (startSize < 3) {
        // Couldn't find enough finder patterns
        err_handler = ReaderErrorHandler("Could not find three finder patterns");
        return vector<Ref<FinderPattern>>();
    }

    vector<Ref<FinderPattern>> result(3);

    if (startSize == 3) {
        result[0] = possibleCenters_[0];
        result[1] = possibleCenters_[1];
        result[2] = possibleCenters_[2];
        return result;
    }

    sort(possibleCenters_.begin(), possibleCenters_.end(), CountComparator());
    if ((possibleCenters_[2]->getCount() - possibleCenters_[3]->getCount()) > 1 &&
        possibleCenters_[2]->getCount() > 1) {
        result[0] = possibleCenters_[0];
        result[1] = possibleCenters_[1];
        result[2] = possibleCenters_[2];
        return result;
    } else if (possibleCenters_[3]->getCount() > 1) {
        float totalModuleSize = 0.0f;
        for (int i = 0; i < 4; i++) {
            totalModuleSize += possibleCenters_[i]->getEstimatedModuleSize();
        }
        float everageModuleSize = totalModuleSize / 4.0f;
        float maxDiffModuleSize = 0.0f;
        int maxID = 0;
        for (int i = 0; i < 4; i++) {
            float diff = abs(possibleCenters_[i]->getEstimatedModuleSize() - everageModuleSize);
            if (diff > maxDiffModuleSize) {
                maxDiffModuleSize = diff;
                maxID = i;
            }
        }
        switch (maxID) {
            case 0:
                result[0] = possibleCenters_[1];
                result[1] = possibleCenters_[2];
                result[2] = possibleCenters_[3];
                break;
            case 1:
                result[0] = possibleCenters_[0];
                result[1] = possibleCenters_[2];
                result[2] = possibleCenters_[3];
                break;
            case 2:
                result[0] = possibleCenters_[0];
                result[1] = possibleCenters_[1];
                result[2] = possibleCenters_[3];
                break;
            default:
                result[0] = possibleCenters_[0];
                result[1] = possibleCenters_[1];
                result[2] = possibleCenters_[2];
                break;
        }

        return result;
    } else if (possibleCenters_[1]->getCount() > 1 && possibleCenters_[2]->getCount() == 1) {
        vector<Ref<FinderPattern>> possibleThirdCenter;
        float possibleModuleSize = (possibleCenters_[0]->getEstimatedModuleSize() +
                                    possibleCenters_[1]->getEstimatedModuleSize()) /
                                   2.0f;
        for (size_t i = 2; i < startSize; i++) {
            if (abs(possibleCenters_[i]->getEstimatedModuleSize() - possibleModuleSize) <
                0.5 * possibleModuleSize)
                possibleThirdCenter.push_back(possibleCenters_[i]);
        }
        float longestSide = 0.0f;
        size_t longestId = 0;
        for (size_t i = 0; i < possibleThirdCenter.size(); i++) {
            float tmpLongSide = 0.0f;
            if (checkIsoscelesRightTriangle(possibleCenters_[0], possibleCenters_[1],
                                            possibleThirdCenter[i], tmpLongSide)) {
                if (tmpLongSide >= longestSide) {
                    longestSide = tmpLongSide;
                    longestId = i;
                }
            }
        }
        result[0] = possibleCenters_[0];
        result[1] = possibleCenters_[1];

        // Error with decoding
        if (longestId >= possibleThirdCenter.size()) {
            err_handler = ReaderErrorHandler("Not find any available possibleThirdCenter");
            return vector<Ref<FinderPattern>>();
        } else {
            result[2] = possibleThirdCenter[longestId];
        }

        return result;
    }

    // Filter outlier possibilities whose module size is too different
    if (startSize > 3) {
        // But we can only afford to do so if we have at least 4 possibilities
        // to choose from
        float totalModuleSize = 0.0f;
        float square = 0.0f;
        for (size_t i = 0; i < startSize; i++) {
            float size = possibleCenters_[i]->getEstimatedModuleSize();
            totalModuleSize += size;
            square += size * size;
        }
        float average = totalModuleSize / (float)startSize;
        float stdDev = (float)sqrt(square / startSize - average * average);

        sort(possibleCenters_.begin(), possibleCenters_.end(),
             FurthestFromAverageComparator(average));

        // float limit = max(0.2f * average, stdDev);
        float limit = max(0.5f * average, stdDev);

        for (size_t i = 0; i < possibleCenters_.size() && possibleCenters_.size() > 3; i++) {
            if (abs(possibleCenters_[i]->getEstimatedModuleSize() - average) > limit) {
                possibleCenters_.erase(possibleCenters_.begin() + i);
                i--;
            }
        }
    }

    int tryHardPossibleCenterSize = 15;
    int possibleCenterSize = 12;

    if (possibleCenters_.size() > size_t(tryHardPossibleCenterSize)) {
        sort(possibleCenters_.begin(), possibleCenters_.end(), CountComparator());
        possibleCenters_.erase(possibleCenters_.begin() + tryHardPossibleCenterSize,
                               possibleCenters_.end());
    } else if (possibleCenters_.size() > size_t(possibleCenterSize)) {
        sort(possibleCenters_.begin(), possibleCenters_.end(), CountComparator());
        possibleCenters_.erase(possibleCenters_.begin() + possibleCenterSize,
                               possibleCenters_.end());
    }

    if (possibleCenters_.size() >= 6) {
        sort(possibleCenters_.begin(), possibleCenters_.end(), XComparator());
        possibleCenters_.erase(possibleCenters_.begin() + 4, possibleCenters_.end() - 2);
        sort(possibleCenters_.begin(), possibleCenters_.begin() + 4, YComparator());
        possibleCenters_.erase(possibleCenters_.begin() + 1, possibleCenters_.begin() + 3);
        sort(possibleCenters_.end() - 2, possibleCenters_.end(), YComparator());
        possibleCenters_.erase(possibleCenters_.end() - 1, possibleCenters_.end());
    } else if (possibleCenters_.size() > 3) {
        // Throw away all but those first size candidate points we found.
        float totalModuleSize = 0.0f;
        for (size_t i = 0; i < possibleCenters_.size(); i++) {
            float size = possibleCenters_[i]->getEstimatedModuleSize();
            totalModuleSize += size;
        }
        float average = totalModuleSize / (float)possibleCenters_.size();
        sort(possibleCenters_.begin(), possibleCenters_.end(), CenterComparator(average));
        possibleCenters_.erase(possibleCenters_.begin() + 3, possibleCenters_.end());
    }

    result[0] = possibleCenters_[0];
    result[1] = possibleCenters_[1];
    result[2] = possibleCenters_[2];

    return result;
}

vector<Ref<FinderPattern>> FinderPatternFinder::selectFileBestPatterns(ErrorHandler& err_handler) {
    size_t startSize = possibleCenters_.size();

    if (startSize < 3) {
        // Couldn't find enough finder patterns
        err_handler = ReaderErrorHandler("Could not find three finder patterns");
        return vector<Ref<FinderPattern>>();
    }

    vector<Ref<FinderPattern>> result(3);

    if (startSize == 3) {
        result[0] = possibleCenters_[0];
        result[1] = possibleCenters_[1];
        result[2] = possibleCenters_[2];
        return result;
    }

    sort(possibleCenters_.begin(), possibleCenters_.end(), BestComparator());

    result[0] = possibleCenters_[0];
    result[1] = possibleCenters_[1];
    result[2] = possibleCenters_[2];

    for (size_t i = 0; i < possibleCenters_.size() - 2; ++i) {
        float tmpLongSide = 0;

        int iCountDiff = 0;
        float fModuleSizeDiff = 0;
        for (size_t j = 0; j < 3; ++j) {
            iCountDiff += abs(possibleCenters_[i + j]->getCount() -
                              possibleCenters_[i + ((j + 1) % 3)]->getCount());
            fModuleSizeDiff += fabs(possibleCenters_[i + j]->getEstimatedModuleSize() -
                                    possibleCenters_[i + ((j + 1) % 3)]->getEstimatedModuleSize());
        }

        if (iCountDiff > 2) continue;
        if (fModuleSizeDiff > 5) continue;

        if (checkIsoscelesRightTriangle(possibleCenters_[i], possibleCenters_[i + 1],
                                        possibleCenters_[i + 2], tmpLongSide)) {
            result[0] = possibleCenters_[i];
            result[1] = possibleCenters_[i + 1];
            result[2] = possibleCenters_[i + 2];

            break;
        }
    }

    return result;
}

// Orders an array of three patterns in an order [A,B,C] such that
// AB<AC and BC<AC and the angle between BC and BA is less than 180 degrees.
vector<Ref<FinderPattern>> FinderPatternFinder::orderBestPatterns(
    vector<Ref<FinderPattern>> patterns) {
    // Find distances between pattern centers
    float abDistance = distance(patterns[0], patterns[1]);
    float bcDistance = distance(patterns[1], patterns[2]);
    float acDistance = distance(patterns[0], patterns[2]);

    Ref<FinderPattern> topLeft;
    Ref<FinderPattern> topRight;
    Ref<FinderPattern> bottomLeft;
    // Assume one closest to other two is top left;
    // topRight and bottomLeft will just be guesses below at first
    if (bcDistance >= abDistance && bcDistance >= acDistance) {
        topLeft = patterns[0];
        topRight = patterns[1];
        bottomLeft = patterns[2];
    } else if (acDistance >= bcDistance && acDistance >= abDistance) {
        topLeft = patterns[1];
        topRight = patterns[0];
        bottomLeft = patterns[2];
    } else {
        topLeft = patterns[2];
        topRight = patterns[0];
        bottomLeft = patterns[1];
    }

    // Use cross product to figure out which of other1/2 is the bottom left
    // pattern. The vector "top_left -> bottom_left" x "top_left -> top_right"
    // should yield a vector with positive z component
    if ((bottomLeft->getY() - topLeft->getY()) * (topRight->getX() - topLeft->getX()) <
        (bottomLeft->getX() - topLeft->getX()) * (topRight->getY() - topLeft->getY())) {
        Ref<FinderPattern> temp = topRight;
        topRight = bottomLeft;
        bottomLeft = temp;
    }

    vector<Ref<FinderPattern>> results(3);
    results[0] = bottomLeft;
    results[1] = topLeft;
    results[2] = topRight;

    return results;
}

bool FinderPatternFinder::checkIsoscelesRightTriangle(Ref<FinderPattern> centerA,
                                                      Ref<FinderPattern> centerB,
                                                      Ref<FinderPattern> centerC, float& longSide) {
    float shortSide1, shortSide2;
    FinderPatternInfo::calculateSides(centerA, centerB, centerC, longSide, shortSide1, shortSide2);

    auto minAmongThree = [](float a, float b, float c) { return min(min(a, b), c); };
    auto maxAmongThree = [](float a, float b, float c) { return max(max(a, b), c); };

    float shortSideSqrt1 = sqrt(shortSide1);
    float shortSideSqrt2 = sqrt(shortSide2);
    float longSideSqrt = sqrt(longSide);
    auto minSide = minAmongThree(shortSideSqrt1, shortSideSqrt2, longSideSqrt);
    auto maxModuleSize =
        maxAmongThree(centerA->getEstimatedModuleSize(), centerB->getEstimatedModuleSize(),
                      centerC->getEstimatedModuleSize());
    // if edge length smaller than 14 * module size
    if (minSide <= maxModuleSize * 14) return false;

    float CosLong = (shortSide1 + shortSide2 - longSide) / (2 * shortSideSqrt1 * shortSideSqrt2);
    float CosShort1 = (longSide + shortSide1 - shortSide2) / (2 * longSideSqrt * shortSideSqrt1);
    float CosShort2 = (longSide + shortSide2 - shortSide1) / (2 * longSideSqrt * shortSideSqrt2);

    if (abs(CosLong) > FP_RIGHT_ANGLE ||
        (CosShort1 < FP_SMALL_ANGLE2 || CosShort1 > FP_SMALL_ANGLE1) ||
        (CosShort2 < FP_SMALL_ANGLE2 || CosShort2 > FP_SMALL_ANGLE1)) {
        return false;
    }

    return true;
}

// return distance between two points
float FinderPatternFinder::distance(Ref<ResultPoint> p1, Ref<ResultPoint> p2) {
    float dx = p1->getX() - p2->getX();
    float dy = p1->getY() - p2->getY();
    return (float)sqrt(dx * dx + dy * dy);
}

FinderPatternFinder::FinderPatternFinder(Ref<BitMatrix> image, Ref<UnicomBlock> block)
    : finder_time(0),
      compared_finder_counts(0),
      image_(image),
      possibleCenters_(),
      hasSkipped_(false),
      block_(block) {
    CURRENT_CHECK_STATE = FinderPatternFinder::NORMAL;
}

Ref<BitMatrix> FinderPatternFinder::getImage() { return image_; }

vector<Ref<FinderPattern>>& FinderPatternFinder::getPossibleCenters() { return possibleCenters_; }

}  // namespace qrcode
}  // namespace zxing