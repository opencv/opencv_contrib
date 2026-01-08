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
#include "alignment_pattern_finder.hpp"

using zxing::ErrorHandler;
using zxing::ReaderErrorHandler;
using zxing::Ref;
using zxing::qrcode::AlignmentPattern;
using zxing::qrcode::AlignmentPatternFinder;
using zxing::qrcode::FinderPattern;

// VC++
// This class attempts to find alignment patterns in a QR Code. Alignment
// patterns look like finder patterns but are smaller and appear at regular
// intervals throughout the image. At the moment this only looks for the
// bottom-right alignment pattern. This is mostly a simplified copy of {@link
// FinderPatternFinder}. It is copied, pasted and stripped down here for maximum
// performance but does unfortunately duplicat some code. This class is
// thread-safe but not reentrant. Each thread must allocate its own object.
using zxing::BitMatrix;

// Creates a finder that will look in a portion of the whole image.
AlignmentPatternFinder::AlignmentPatternFinder(Ref<BitMatrix> image, int startX, int startY,
                                               int width, int height, float moduleSize)
    : image_(image),
      possibleCenters_(new vector<AlignmentPattern *>()),
      startX_(startX),
      startY_(startY),
      width_(width),
      height_(height),
      moduleSize_(moduleSize) {}

AlignmentPatternFinder::AlignmentPatternFinder(Ref<BitMatrix> image, float moduleSize)
    : image_(image),
      moduleSize_(moduleSize) {}

// This method attempts to find the bottom-right alignment pattern in the image.
// It is a bit messy since it's pretty performance-critical and so is written to
// be fast foremost.
Ref<AlignmentPattern> AlignmentPatternFinder::find(ErrorHandler &err_handler) {
    int maxJ = startX_ + width_;
    int middleI = startY_ + (height_ >> 1);
    // We are looking for black/white/black modules in 1:1:1 ratio;
    // this tracks the number of black/white/black modules seen so far
    vector<int> stateCount(3, 0);
    for (int iGen = 0; iGen < height_; iGen++) {
        // Search from middle outwards
        int i = middleI + ((iGen & 0x01) == 0 ? ((iGen + 1) >> 1) : -((iGen + 1) >> 1));
        stateCount[0] = 0;
        stateCount[1] = 0;
        stateCount[2] = 0;
        int j = startX_;
        // Burn off leading white pixels before anything else; if we start in
        // the middle of a white run, it doesn't make sense to count its length,
        // since we don't know if the white run continued to the left of the
        // start point
        while (j < maxJ && !image_->get(j, i)) {
            j++;
        }
        int currentState = 0;
        while (j < maxJ) {
            if (image_->get(j, i)) {
                // Black pixel
                if (currentState == 1) {  // Counting black pixels
                    stateCount[currentState]++;
                } else {                                      // Counting white pixels
                    if (currentState == 2) {                  // A winner?
                        if (foundPatternCross(stateCount)) {  // Yes
                            Ref<AlignmentPattern> confirmed(handlePossibleCenter(stateCount, i, j));
                            if (confirmed != 0) {
                                return confirmed;
                            }
                        }
                        stateCount[0] = stateCount[2];
                        stateCount[1] = 1;
                        stateCount[2] = 0;
                        currentState = 1;
                    } else {
                        stateCount[++currentState]++;
                    }
                }
            } else {                      // White pixel
                if (currentState == 1) {  // Counting black pixels
                    currentState++;
                }
                stateCount[currentState]++;
            }
            j++;
        }
        if (foundPatternCross(stateCount)) {
            Ref<AlignmentPattern> confirmed(handlePossibleCenter(stateCount, i, maxJ));
            if (confirmed != 0) {
                return confirmed;
            }
        }
    }
    // Nothing we saw was observed and confirmed twice. If we had any guess at
    // all, return it.
    if (possibleCenters_->size() > 0) {
        Ref<AlignmentPattern> center((*possibleCenters_)[0]);
        return center;
    }
    err_handler = ReaderErrorHandler("Could not find alignment pattern");
    return Ref<AlignmentPattern>();
}


// Given a count of black/white/black pixels just seen and an end position,
// figures the location of the center of this black/white/black run.
float AlignmentPatternFinder::centerFromEnd(vector<int> &stateCount, int end) {
    return (float)(end - stateCount[2]) - stateCount[1] / 2.0f;
}



bool AlignmentPatternFinder::foundPatternCross(vector<int> &stateCount) {
    float maxVariance = moduleSize_ / 2.0f;
    for (int i = 0; i < 3; i++) {
        if (abs(moduleSize_ - stateCount[i]) >= maxVariance) {
            return false;
        }
    }
    return true;
}

// After a horizontal scan finds a potential alignment pattern, this method
// "cross-checks" by scanning down vertically through the center of the possible
// alignment pattern to see if the same proportion is detected. return vertical
// center of alignment pattern, or nan() if not found startI: row where an
// alignment pattern was detected centerJ: center of the section that appears to
// cross an alignment pattern
// maxCount: maximum reasonable number of modules that should be observed in any
// reading state,
// based on the results of the horizontal scan
float AlignmentPatternFinder::crossCheckVertical(int startI, int centerJ, int maxCount,
                                                 int originalStateCountTotal) {
    // This is slightly faster than using the Ref. Efficiency is important here
    BitMatrix &matrix = *image_;

    int maxI = matrix.getHeight();
    vector<int> stateCount(3, 0);
    // Start counting up from center
    int i = startI;
    while (i >= 0 && matrix.get(centerJ, i) && stateCount[1] <= maxCount) {
        stateCount[1]++;
        i--;
    }
    // If already too many modules in this state or ran off the edge:
    if (i < 0 || stateCount[1] > maxCount) {
        return nan();
    }
    while (i >= 0 && !matrix.get(centerJ, i) && stateCount[0] <= maxCount) {
        stateCount[0]++;
        i--;
    }
    if (stateCount[0] > maxCount) {
        return nan();
    }

    // Now also count down from center
    i = startI + 1;
    while (i < maxI && matrix.get(centerJ, i) && stateCount[1] <= maxCount) {
        stateCount[1]++;
        i++;
    }
    if (i == maxI || stateCount[1] > maxCount) {
        return nan();
    }
    while (i < maxI && !matrix.get(centerJ, i) && stateCount[2] <= maxCount) {
        stateCount[2]++;
        i++;
    }
    if (stateCount[2] > maxCount) {
        return nan();
    }

    int stateCountTotal = stateCount[0] + stateCount[1] + stateCount[2];
    if (5 * abs(stateCountTotal - originalStateCountTotal) >= 2 * originalStateCountTotal) {
        return nan();
    }
    return foundPatternCross(stateCount) ? centerFromEnd(stateCount, i) : nan();
}


// This is called when a horizontal scan finds a possible alignment pattern. It
// will cross check with a vertical scan, and if successful, will see if this
// pattern had been found on a previous horizontal scan. If so, we consider it
// confirmed and conclude we have found the alignment pattern. return {@link
// AlignmentPattern} if we have found the same pattern twice, or null if not i:
// row where alignment pattern may be found j: end of possible alignment pattern
// in row
Ref<AlignmentPattern> AlignmentPatternFinder::handlePossibleCenter(vector<int> &stateCount, int i,
                                                                   int j) {
    int stateCountTotal = stateCount[0] + stateCount[1] + stateCount[2];
    float centerJ = centerFromEnd(stateCount, j);
    float centerI = crossCheckVertical(i, (int)centerJ, 2 * stateCount[1], stateCountTotal);
    if (!isnan(centerI)) {
        float estimatedModuleSize = (float)(stateCount[0] + stateCount[1] + stateCount[2]) / 3.0f;
        int max = possibleCenters_->size();
        for (int index = 0; index < max; index++) {
            Ref<AlignmentPattern> center((*possibleCenters_)[index]);
            // Look for about the same center and module size:
            if (center->aboutEquals(estimatedModuleSize, centerI, centerJ)) {
                return center->combineEstimate(centerI, centerJ, estimatedModuleSize);
            }
        }
        // Hadn't found this before; save it
        AlignmentPattern *tmp = new AlignmentPattern(centerJ, centerI, estimatedModuleSize);
        tmp->retain();
        possibleCenters_->push_back(tmp);
    }
    Ref<AlignmentPattern> result;
    return result;
}


AlignmentPatternFinder::~AlignmentPatternFinder() {
    for (int i = 0; i < int(possibleCenters_->size()); i++) {
        (*possibleCenters_)[i]->release();
        (*possibleCenters_)[i] = 0;
    }
    delete possibleCenters_;
}
