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
#include "detector.hpp"
#include <sstream>
#include "../../common/grid_sampler.hpp"
#include "../../common/mathutils.hpp"
#include "../../decodehints.hpp"
#include "../version.hpp"
#include "alignment_pattern.hpp"
#include "alignment_pattern_finder.hpp"
#include "finder_pattern.hpp"
#include "finder_pattern_finder.hpp"
#include "opencv2/core.hpp"

using zxing::BitMatrix;
using zxing::DetectorResult;
using zxing::ErrorHandler;
using zxing::PerspectiveTransform;
using zxing::Ref;
using zxing::common::MathUtils;
using zxing::qrcode::AlignmentPattern;
using zxing::qrcode::Detector;
using zxing::qrcode::FinderPattern;

// VC++
using zxing::DecodeHints;
using zxing::ResultPoint;
using zxing::UnicomBlock;
using zxing::qrcode::FinderPatternFinder;
using zxing::qrcode::FinderPatternInfo;
using zxing::qrcode::PatternResult;

// Encapsulates logic that can detect a QR Code in an image,
// even if the QR Code is rotated or skewed, or partially obscured.
Detector::Detector(Ref<BitMatrix> image, Ref<UnicomBlock> block) : image_(image), block_(block) {
    detectorState_ = START;
    possiblePatternResults_.clear();
}

Ref<BitMatrix> Detector::getImage() const { return image_; }

// Detects a QR Code in an image
void Detector::detect(DecodeHints const &hints, ErrorHandler &err_handler) {
    FinderPatternFinder finder(image_, block_);
    std::vector<Ref<FinderPatternInfo> > finderInfos = finder.find(hints, err_handler);
    if (err_handler.ErrCode()) return;

    // Get all possible results
    possiblePatternResults_.clear();

    for (size_t i = 0; i < finderInfos.size(); i++) {
        Ref<PatternResult> result(new PatternResult(finderInfos[i]));
        result->possibleVersion = 0;
        result->possibleFix = 0.0f;
        result->possibleModuleSize = 0.0f;

        possiblePatternResults_.push_back(result);
    }
    detectorState_ = FINDFINDERPATTERN;
}

int Detector::getPossibleAlignmentCount(int idx) {
    if (idx >= int(possiblePatternResults_.size())) {
        return -1;
    }

    ErrorHandler err_handler;
    // If it is first time to get, process it now
    if (possiblePatternResults_[idx]->possibleAlignmentPatterns.size() == 0) {
        Ref<PatternResult> result =
            processFinderPatternInfo(possiblePatternResults_[idx]->finderPatternInfo, err_handler);
        if (err_handler.ErrCode()) return -1;

        possiblePatternResults_[idx] = result;
    }

    return possiblePatternResults_[idx]->possibleAlignmentPatterns.size();
}

Ref<DetectorResult> Detector::getResultViaAlignment(int patternIdx, int alignmentIdx,
                                                    int possibleDimension,
                                                    ErrorHandler &err_handler) {
    if (patternIdx >= int(possiblePatternResults_.size()) || patternIdx < 0) {
        return Ref<DetectorResult>(NULL);
    }

    if (alignmentIdx >=
            int(possiblePatternResults_[patternIdx]->possibleAlignmentPatterns.size()) ||
        alignmentIdx < 0) {
        return Ref<DetectorResult>(NULL);
    }

    // Default is the dimension
    if (possibleDimension <= 0) {
        possibleDimension = possiblePatternResults_[patternIdx]->getDimension();
    }

    Ref<FinderPattern> topLeft(
        possiblePatternResults_[patternIdx]->finderPatternInfo->getTopLeft());
    Ref<FinderPattern> topRight(
        possiblePatternResults_[patternIdx]->finderPatternInfo->getTopRight());
    Ref<FinderPattern> bottomLeft(
        possiblePatternResults_[patternIdx]->finderPatternInfo->getBottomLeft());

    Ref<AlignmentPattern> alignment(
        possiblePatternResults_[patternIdx]->possibleAlignmentPatterns[alignmentIdx]);
    Ref<PerspectiveTransform> transform =
        createTransform(topLeft, topRight, bottomLeft, alignment, possibleDimension);
    Ref<BitMatrix> bits(sampleGrid(image_, possibleDimension, transform, err_handler));
    if (err_handler.ErrCode()) return Ref<DetectorResult>();

    ArrayRef<Ref<ResultPoint> > corrners(new Array<Ref<ResultPoint> >(4));
    vector<float> points(8, 0.0f);
    points[0] = 0.0f;
    points[1] = possibleDimension;  // bottomLeft
    points[2] = 0.0f;
    points[3] = 0.0f;  // topLeft
    points[4] = possibleDimension;
    points[5] = 0.0f;  // topRight
    points[6] = possibleDimension;
    points[7] = possibleDimension;  // bottomRight
    transform->transformPoints(points);
    corrners[0].reset(Ref<FinderPattern>(new FinderPattern(points[0], points[1], 0)));
    corrners[1].reset(Ref<FinderPattern>(new FinderPattern(points[2], points[3], 0)));
    corrners[2].reset(Ref<FinderPattern>(new FinderPattern(points[4], points[5], 0)));
    corrners[3].reset(Ref<FinderPattern>(new FinderPattern(points[6], points[7], 0)));

    Ref<DetectorResult> result(new DetectorResult(bits, corrners, possibleDimension));
    return result;
}

bool Detector::hasSameResult(vector<Ref<AlignmentPattern> > possibleAlignmentPatterns,
                             Ref<AlignmentPattern> alignmentPattern) {
    float moduleSize = alignmentPattern->getModuleSize() / 5.0;

    if (moduleSize < 1.0) {
        moduleSize = 1.0;
    }

    for (size_t i = 0; i < possibleAlignmentPatterns.size(); i++) {
        if (possibleAlignmentPatterns[i]->aboutEquals(moduleSize, alignmentPattern->getY(),
                                                      alignmentPattern->getX())) {
            return true;
        }
    }
    return false;
}

Ref<AlignmentPattern> Detector::getNearestAlignmentPattern(int tryFindRange, float moduleSize,
                                                           int estAlignmentX, int estAlignmentY) {
    Ref<AlignmentPattern> alignmentPattern;

    ErrorHandler err_handler;
    for (int i = 2; i <= tryFindRange; i <<= 1) {
        err_handler.Reset();
        alignmentPattern =
            findAlignmentInRegion(moduleSize, estAlignmentX, estAlignmentY, (float)i, err_handler);
        if (err_handler.ErrCode() == 0) break;
    }

    return alignmentPattern;
}

Ref<PatternResult> Detector::processFinderPatternInfo(Ref<FinderPatternInfo> info,
                                                      ErrorHandler &err_handler) {
    Ref<FinderPattern> topLeft(info->getTopLeft());
    Ref<FinderPattern> topRight(info->getTopRight());
    Ref<FinderPattern> bottomLeft(info->getBottomLeft());

    Ref<PatternResult> result(new PatternResult(info));
    result->finderPatternInfo = info;
    result->possibleAlignmentPatterns.clear();

    float moduleSizeX_ = calculateModuleSizeOneWay(
        topLeft, topRight, topLeft->getHorizontalCheckState(), topRight->getHorizontalCheckState());
    float moduleSizeY_ = calculateModuleSizeOneWay(
        topLeft, bottomLeft, topLeft->getVerticalCheckState(), bottomLeft->getVerticalCheckState());

    if (moduleSizeX_ < 1.0f || moduleSizeY_ < 1.0f) {
        err_handler = ReaderErrorHandler("bad midule size");
        return Ref<PatternResult>();
    }

    float moduleSize = (moduleSizeX_ + moduleSizeY_) / 2.0f;

    if (moduleSize > topLeft->getEstimatedModuleSize() * 1.05 &&
        moduleSize > topRight->getEstimatedModuleSize() * 1.05 &&
        moduleSize > bottomLeft->getEstimatedModuleSize() * 1.05) {
        moduleSize = (topLeft->getEstimatedModuleSize() + topRight->getEstimatedModuleSize() +
                      bottomLeft->getEstimatedModuleSize()) /
                     3;
        moduleSizeX_ = moduleSize;
        moduleSizeY_ = moduleSize;
    }
    result->possibleModuleSize = moduleSize;

    if (moduleSize < 1.0f) {
        err_handler = ReaderErrorHandler("bad midule size");
        return Ref<PatternResult>();
    }
    int dimension = computeDimension(topLeft, topRight, bottomLeft, moduleSizeX_, moduleSizeY_);
    Version *provisionalVersion = NULL;

    // Try demension around if it cannot get a version
    int dimensionDiff[5] = {0, 1, -1, 2, -2};

    int oriDimension = dimension;

    for (int i = 0; i < 5; i++) {
        err_handler.Reset();
        dimension = oriDimension + dimensionDiff[i];

        provisionalVersion = Version::getProvisionalVersionForDimension(dimension, err_handler);
        if (err_handler.ErrCode() == 0) break;
    }
    if (provisionalVersion == NULL) {
        err_handler = zxing::ReaderErrorHandler("Cannot get version number");
        return Ref<PatternResult>();
    }

    result->possibleDimension = dimension;

    result->possibleVersion = provisionalVersion->getVersionNumber();

    int modulesBetweenFPCenters = provisionalVersion->getDimensionForVersion(err_handler) - 7;
    if (err_handler.ErrCode()) return Ref<PatternResult>();

    Ref<AlignmentPattern> alignmentPattern;

    // Guess where a "bottom right" finder pattern would have been
    float bottomRightX = topRight->getX() - topLeft->getX() + bottomLeft->getX();
    float bottomRightY = topRight->getY() - topLeft->getY() + bottomLeft->getY();
    // Estimate that alignment pattern is closer by 3 modules from "bottom
    // right" to known top left location
    float correctionToTopLeft = 1.0f - 3.0f / (float)modulesBetweenFPCenters;
    int estAlignmentX =
        (int)(topLeft->getX() + correctionToTopLeft * (bottomRightX - topLeft->getX()));
    int estAlignmentY =
        (int)(topLeft->getY() + correctionToTopLeft * (bottomRightY - topLeft->getY()));

    Ref<AlignmentPattern> estimateCenter(
        new AlignmentPattern(estAlignmentX, estAlignmentY, moduleSize));

    bool foundFitLine = false;
    Ref<AlignmentPattern> fitLineCenter;

    fitLineCenter =
        findAlignmentWithFitLine(topLeft, topRight, bottomLeft, moduleSize, err_handler);
    if (err_handler.ErrCode() == 0) {
        if (fitLineCenter != NULL &&
            MathUtils::isInRange(fitLineCenter->getX(), fitLineCenter->getY(), image_->getWidth(),
                                 image_->getHeight())) {
            foundFitLine = true;
        }
    }
    err_handler.Reset();

    Ref<AlignmentPattern> fitAP, estAP;

    // Anything above version 1 has an alignment pattern
    if (provisionalVersion->getAlignmentPatternCenters().size()) {
        // if(alignmentPattern!=NULL&&alignmentPattern->getX()>0&&alignmentPattern->getY()>0){
        int tryFindRange = provisionalVersion->getDimensionForVersion(err_handler) / 2;
        if (err_handler.ErrCode()) return Ref<PatternResult>();

        if (foundFitLine == true) {
            fitAP = getNearestAlignmentPattern(tryFindRange, moduleSize, fitLineCenter->getX(),
                                               fitLineCenter->getY());

            if (fitAP != NULL && !hasSameResult(result->possibleAlignmentPatterns, fitAP))
            // if (fitAP != NULL &&
            // !hasSameResult(result->possibleAlignmentPatterns, fitAP) &&
            // checkConvexQuadrilateral(topLeft, topRight, bottomLeft, fitAP))
            {
                result->possibleAlignmentPatterns.push_back(fitAP);
            }
        }

        estAP = getNearestAlignmentPattern(tryFindRange, moduleSize, estimateCenter->getX(),
                                           estimateCenter->getY());

        if (estAP != NULL && !hasSameResult(result->possibleAlignmentPatterns, estAP))
        // if (estAP != NULL &&
        // !hasSameResult(result->possibleAlignmentPatterns, estAP) &&
        // checkConvexQuadrilateral(topLeft, topRight, bottomLeft, estAP))
        {
            result->possibleAlignmentPatterns.push_back(estAP);
        }
    }

    // Any way use the fit line result
    if (foundFitLine == true && !hasSameResult(result->possibleAlignmentPatterns, fitLineCenter)) {
        float alignmentX = fitLineCenter->getX();
        float alignmentY = fitLineCenter->getY();
        fixAlignmentPattern(alignmentX, alignmentY, moduleSize);
        Ref<AlignmentPattern> fitLineCenterFixed =
            Ref<AlignmentPattern>(new AlignmentPattern(alignmentX, alignmentY, moduleSize));
        if (!hasSameResult(result->possibleAlignmentPatterns, fitLineCenterFixed)) {
            result->possibleAlignmentPatterns.push_back(fitLineCenterFixed);
        }

        if (!hasSameResult(result->possibleAlignmentPatterns, fitLineCenter)) {
            result->possibleAlignmentPatterns.push_back(fitLineCenter);
        }
    }

    if (!hasSameResult(result->possibleAlignmentPatterns, estimateCenter)) {
        float alignmentX = estimateCenter->getX();
        float alignmentY = estimateCenter->getY();
        fixAlignmentPattern(alignmentX, alignmentY, moduleSize);
        Ref<AlignmentPattern> estimateCenterFixed =
            Ref<AlignmentPattern>(new AlignmentPattern(alignmentX, alignmentY, moduleSize));
        if (!hasSameResult(result->possibleAlignmentPatterns, estimateCenterFixed)) {
            result->possibleAlignmentPatterns.push_back(estimateCenterFixed);
        }

        if (!hasSameResult(result->possibleAlignmentPatterns, estimateCenter)) {
            result->possibleAlignmentPatterns.push_back(estimateCenter);
        }
    }
    Ref<AlignmentPattern> NoneEstimateCenter =
        Ref<AlignmentPattern>(new AlignmentPattern(0, 0, moduleSize));
    result->possibleAlignmentPatterns.push_back(NoneEstimateCenter);

    if (result->possibleAlignmentPatterns.size() > 0) {
        result->confirmedAlignmentPattern = result->possibleAlignmentPatterns[0];
    }
    detectorState_ = FINDALIGNPATTERN;

    return result;
}

// Computes an average estimated module size based on estimated derived from the
// positions of the three finder patterns.
float Detector::calculateModuleSize(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                                    Ref<ResultPoint> bottomLeft) {
    // Take the average
    return (calculateModuleSizeOneWay(topLeft, topRight, NORMAL, NORMAL) +
            calculateModuleSizeOneWay(topLeft, bottomLeft, NORMAL, NORMAL)) /
           2.0f;
}

// Estimates module size based on two finder patterns
// it uses sizeOfBlackWhiteBlackRunBothWays() to figure the width of each,
// measuring along the axis between their centers.
float Detector::calculateModuleSizeOneWay(Ref<ResultPoint> pattern, Ref<ResultPoint> otherPattern,
                                          int patternState, int otherPatternState) {
    float moduleSizeEst1 = sizeOfBlackWhiteBlackRunBothWays(
        (int)pattern->getX(), (int)pattern->getY(), (int)otherPattern->getX(),
        (int)otherPattern->getY(), patternState, false);
    float moduleSizeEst2 = sizeOfBlackWhiteBlackRunBothWays(
        (int)otherPattern->getX(), (int)otherPattern->getY(), (int)pattern->getX(),
        (int)pattern->getY(), otherPatternState, true);
    if (zxing::isnan(moduleSizeEst1)) {
        return moduleSizeEst2 / 7.0f;
    }
    if (zxing::isnan(moduleSizeEst2)) {
        return moduleSizeEst1 / 7.0f;
    }
    // Average them, and divide by 7 since we've counted the width of 3 black
    // modules, and 1 white and 1 black module on either side. Ergo, divide sum
    // by 14.
    return (moduleSizeEst1 + moduleSizeEst2) / 14.0f;
}

// Computes the total width of a finder pattern by looking for a
// black-white-black run from the center in the direction of another point
// (another finder pattern center), and in the opposite direction too.
float Detector::sizeOfBlackWhiteBlackRunBothWays(int fromX, int fromY, int toX, int toY,
                                                 int patternState, bool isReverse) {
    float result1 = sizeOfBlackWhiteBlackRun(fromX, fromY, toX, toY);
    float result = 0.0;
    // Now count other way -- don't run off image though of course
    float scale = 1.0f;
    int otherToX = fromX - (toX - fromX);
    if (otherToX < 0) {
        scale = (float)fromX / (float)(fromX - otherToX);
        otherToX = 0;
    } else if (otherToX >= (int)image_->getWidth()) {
        scale = (float)(image_->getWidth() - 1 - fromX) / (float)(otherToX - fromX);
        otherToX = image_->getWidth() - 1;
    }
    int otherToY = (int)(fromY - (toY - fromY) * scale);

    scale = 1.0f;
    if (otherToY < 0) {
        scale = (float)fromY / (float)(fromY - otherToY);
        otherToY = 0;
    } else if (otherToY >= (int)image_->getHeight()) {
        scale = (float)(image_->getHeight() - 1 - fromY) / (float)(otherToY - fromY);
        otherToY = image_->getHeight() - 1;
    }
    otherToX = (int)(fromX + (otherToX - fromX) * scale);

    float result2 = sizeOfBlackWhiteBlackRun(fromX, fromY, otherToX, otherToY);

    if (patternState == FinderPattern::HORIZONTAL_STATE_LEFT_SPILL ||
        patternState == FinderPattern::VERTICAL_STATE_UP_SPILL) {
        if (!isReverse)
            result = result1 * 2;
        else
            result = result2 * 2;
    } else if (patternState == FinderPattern::HORIZONTAL_STATE_RIGHT_SPILL ||
               patternState == FinderPattern::VERTICAL_STATE_DOWN_SPILL) {
        if (!isReverse)
            result = result2 * 2;
        else
            result = result1 * 2;
    } else {
        result = result1 + result2;
    }
    // Middle pixel is double-counted this way; subtract 1
    return result - 1.0f;
}

Ref<BitMatrix> Detector::sampleGrid(Ref<BitMatrix> image, int dimension,
                                    Ref<PerspectiveTransform> transform,
                                    ErrorHandler &err_handler) {
    GridSampler &sampler = GridSampler::getInstance();
    // return sampler.sampleGrid(image, dimension, transform);
    Ref<BitMatrix> bits = sampler.sampleGrid(image, dimension, transform, err_handler);
    if (err_handler.ErrCode()) return Ref<BitMatrix>();
    return bits;
}

// This method traces a line from a point in the image, in the direction towards
// another point. It begins in a black region, and keeps going until it finds
// white, then black, then white again. It reports the distance from the start
// to this point.
float Detector::sizeOfBlackWhiteBlackRun(int fromX, int fromY, int toX, int toY) {
    // Mild variant of Bresenham's algorithm;
    // see http://en.wikipedia.org/wiki/Bresenham's_line_algorithm
    bool steep = abs(toY - fromY) > abs(toX - fromX);
    if (steep) {
        // swap(fromX,fromY)
        int temp = fromX;
        fromX = fromY;
        fromY = temp;
        // swap(toX,toY)
        temp = toX;
        toX = toY;
        toY = temp;
    }

    int dx = abs(toX - fromX);
    int dy = abs(toY - fromY);
    int error = -dx >> 1;
    int xstep = fromX < toX ? 1 : -1;
    int ystep = fromY < toY ? 1 : -1;
    // In black pixels, looking for white, first or second time.
    int state = 0;
    // Loop up until x == toX, but not beyond
    int xLimit = toX + xstep;
    for (int x = fromX, y = fromY; x != xLimit; x += xstep) {
        int realX = steep ? y : x;
        int realY = steep ? x : y;

        // Does current pixel mean we have moved white to black or vice versa?
        // Scanning black in state 0,2 and white in state 1, so if we find the
        // wrong color, advance to next state or end if we are in state 2
        // already
        if (!((state == 1) ^ image_->get(realX, realY))) {
            if (state == 2) {
                return MathUtils::distance(x, y, fromX, fromY);
            }
            state++;
        }

        error += dy;
        if (error > 0) {
            if (y == toY) {
                break;
            }
            y += ystep;
            error -= dx;
        }
    }
    // Found black-white-black; give the benefit of the doubt that the next
    // pixel outside the image is "white" so this last point at (toX+xStep,toY)
    // is the right ending. This is really a small approximation;
    // (toX+xStep,toY+yStep) might be really correct. Ignore this.
    if (state == 2) {
        return MathUtils::distance(toX + xstep, toY, fromX, fromY);
    }
    // else we didn't find even black-white-black; no estimate is really
    // possible
    return nan();
}

// Attempts to locate an alignment pattern in a limited region of the image,
// which is guessed to contain it.
Ref<AlignmentPattern> Detector::findAlignmentInRegion(float overallEstModuleSize, int estAlignmentX,
                                                      int estAlignmentY, float allowanceFactor,
                                                      ErrorHandler &err_handler) {
    // Look for an alignment pattern (3 modules in size) around where it should
    // be
    int allowance = (int)(allowanceFactor * overallEstModuleSize);
    int alignmentAreaLeftX = max(0, estAlignmentX - allowance);
    int alignmentAreaRightX = min((int)(image_->getWidth() - 1), estAlignmentX + allowance);
    if (alignmentAreaRightX - alignmentAreaLeftX < overallEstModuleSize * 3) {
        err_handler = ReaderErrorHandler("region too small to hold alignment pattern");
        return Ref<AlignmentPattern>();
    }
    int alignmentAreaTopY = max(0, estAlignmentY - allowance);
    int alignmentAreaBottomY = min((int)(image_->getHeight() - 1), estAlignmentY + allowance);
    if (alignmentAreaBottomY - alignmentAreaTopY < overallEstModuleSize * 3) {
        err_handler = ReaderErrorHandler("region too small to hold alignment pattern");
        return Ref<AlignmentPattern>();
    }

    AlignmentPatternFinder alignmentFinder(
        image_, alignmentAreaLeftX, alignmentAreaTopY, alignmentAreaRightX - alignmentAreaLeftX,
        alignmentAreaBottomY - alignmentAreaTopY, overallEstModuleSize);

    Ref<AlignmentPattern> ap = alignmentFinder.find(err_handler);
    if (err_handler.ErrCode()) return Ref<AlignmentPattern>();
    return ap;

}

Ref<AlignmentPattern> Detector::findAlignmentWithFitLine(Ref<ResultPoint> topLeft,
                                                         Ref<ResultPoint> topRight,
                                                         Ref<ResultPoint> bottomLeft,
                                                         float moduleSize,
                                                         ErrorHandler &err_handler) {
    float alignmentX = 0.0f, alignmentY = 0.0f;
    int imgWidth = image_->getWidth();
    int imgHeight = image_->getHeight();
    Rect bottomLeftRect, topRightRect;
    double rectSize = moduleSize * 7;
    bottomLeftRect.x =
        (bottomLeft->getX() - rectSize / 2.0f) > 0 ? (bottomLeft->getX() - rectSize / 2.0f) : 0;
    bottomLeftRect.y =
        (bottomLeft->getY() - rectSize / 2.0f) > 0 ? (bottomLeft->getY() - rectSize / 2.0f) : 0;
    bottomLeftRect.width = (bottomLeft->getX() - bottomLeftRect.x) * 2;
    if (bottomLeftRect.x + bottomLeftRect.width > imgWidth)
        bottomLeftRect.width = imgWidth - bottomLeftRect.x;
    bottomLeftRect.height = (bottomLeft->getY() - bottomLeftRect.y) * 2;
    if (bottomLeftRect.y + bottomLeftRect.height > imgHeight)
        bottomLeftRect.height = imgHeight - bottomLeftRect.y;

    topRightRect.x =
        (topRight->getX() - rectSize / 2.0f) > 0 ? (topRight->getX() - rectSize / 2.0f) : 0;
    topRightRect.y =
        (topRight->getY() - rectSize / 2.0f) > 0 ? (topRight->getY() - rectSize / 2.0f) : 0;
    topRightRect.width = (topRight->getX() - topRightRect.x) * 2;
    if (topRightRect.x + topRightRect.width > imgWidth)
        topRightRect.width = imgWidth - topRightRect.x;
    topRightRect.height = (topRight->getY() - topRightRect.y) * 2;
    if (topRightRect.y + topRightRect.height > imgHeight)
        topRightRect.height = imgHeight - topRightRect.y;

    vector<Ref<ResultPoint> > topRightPoints;
    vector<Ref<ResultPoint> > bottomLeftPoints;

    findPointsForLine(topLeft, topRight, bottomLeft, topRightRect, bottomLeftRect, topRightPoints,
                      bottomLeftPoints, moduleSize);

    int a1;
    float k1, b1;
    int fitResult = fitLine(topRightPoints, k1, b1, a1);
    if (fitResult < 0) {
        err_handler = ReaderErrorHandler("Cannot find a valid divide for line fit");
        return Ref<AlignmentPattern>();
    }

    int a2;
    float k2, b2;
    int fitResult2 = fitLine(bottomLeftPoints, k2, b2, a2);
    if (fitResult2 < 0) {
        err_handler = ReaderErrorHandler("Cannot find a valid divide for line fit");
        return Ref<AlignmentPattern>();
    }

    int hasResult = 1;
    if (a1 == 0) {
        if (a2 == 0) {
            hasResult = 0;
        } else {
            alignmentX = -b1;
            alignmentY = b2 - b1 * k2;
        }
    } else {
        if (a2 == 0) {
            alignmentX = -b2;
            alignmentY = b1 - b2 * k1;
        } else {
            if (k1 == k2) {
                hasResult = 0;
            } else {
                alignmentX = (b2 - b1) / (k1 - k2);
                alignmentY = k1 * alignmentX + b1;
            }
        }
    }

    // Donot have a valid divide
    if (hasResult == 0) {
        err_handler = ReaderErrorHandler("Cannot find a valid divide for line fit");
        return Ref<AlignmentPattern>();
    }
    Ref<AlignmentPattern> result(new AlignmentPattern(alignmentX, alignmentY, moduleSize));
    return result;
}

void Detector::fixAlignmentPattern(float &alignmentX, float &alignmentY, float moduleSize) {
    int imgWidth = image_->getWidth();
    int imgHeight = image_->getHeight();
    int maxFixStep = moduleSize * 2;
    int fixStep = 0;
    while (alignmentX < imgWidth && alignmentY < imgHeight && alignmentX > 0 && alignmentY > 0 &&
           !image_->get(alignmentX, alignmentY) && fixStep < maxFixStep) {
        ++fixStep;
        // Newest Version:  The fix process is like this:
        // 1  2  3
        // 4  0  5
        // 6  7  8
        for (int y = alignmentY - fixStep; y <= alignmentY + fixStep; y++) {
            if (y == alignmentY - fixStep || y == alignmentY + fixStep) {
                for (int x = alignmentX - fixStep; x <= alignmentX + fixStep; x++) {
                    if (x < imgWidth && y < imgHeight && x > 0 && y > 0 && image_->get(x, y)) {
                        alignmentX = x;
                        alignmentY = y;
                        return;
                    }
                }
            } else {
                int x = alignmentX - fixStep;
                if (x < imgWidth && y < imgHeight && x > 0 && y > 0 && image_->get(x, y)) {
                    alignmentX = x;
                    alignmentY = y;
                    return;
                }
                x = alignmentX + fixStep;
                if (x < imgWidth && y < imgHeight && x > 0 && y > 0 && image_->get(x, y)) {
                    alignmentX = x;
                    alignmentY = y;
                    return;
                }
            }
        }
    }

    return;
}

int Detector::fitLine(vector<Ref<ResultPoint> > &oldPoints, float &k, float &b, int &a) {
    a = 1;
    k = 0.0f;
    b = 0.0f;
    int old_num = oldPoints.size();
    if (old_num < 2) {
        return -1;
    }
    float tolerance = 2.0f;
    vector<Ref<ResultPoint> > fitPoints;
    float pre_diff = -1;
    for (vector<Ref<ResultPoint> >::iterator it = oldPoints.begin() + 1; it != oldPoints.end() - 1;
         it++) {
        float diff_x = 0.0f, diff_y = 0.0f, diff = 0.0f;
        if (pre_diff < 0) {
            diff_x = (*(it - 1))->getX() - (*it)->getX();
            diff_y = (*(it - 1))->getY() - (*it)->getY();
            diff = (diff_x * diff_x + diff_y * diff_y);
            pre_diff = diff;
        }
        diff_x = (*(it + 1))->getX() - (*it)->getX();
        diff_y = (*(it + 1))->getY() - (*it)->getY();
        diff = (diff_x * diff_x + diff_y * diff_y);
        if (pre_diff <= tolerance && diff <= tolerance) {
            fitPoints.push_back(*(it));
        }
        pre_diff = diff;
    }

    int num = fitPoints.size();
    if (num < 2) return -1;

    double x = 0, y = 0, xx = 0, xy = 0, tem = 0;
    for (int i = 0; i < num; i++) {
        int point_x = fitPoints[i]->getX();
        int point_y = fitPoints[i]->getY();
        x += point_x;
        y += point_y;
        xx += point_x * point_x;
        xy += point_x * point_y;
    }

    tem = xx * num - x * x;
    if (abs(tem) < 0.0000001) {
        // Set b as average x
        b = -x / num;
        a = 0;
        k = 1;

        return 1;
    }

    k = (num * xy - x * y) / tem;
    b = (y - k * x) / num;
    a = 1;
    if (abs(k) < 0.01) k = 0;
    return 1;
}

bool Detector::checkTolerance(Ref<ResultPoint> &topLeft, Ref<ResultPoint> &topRight,
                              Rect &topRightRect, double modelSize, Ref<ResultPoint> &p, int flag) {
    int topLeftX = topLeft->getX(), topLeftY = topLeft->getY(), topRightX = topRight->getX(),
        topRightY = topRight->getY();
    double left_right_k = 0.0f, left_right_b = 0.0f, left_right_b_tolerance, tolerance_b1 = 0.0f,
           tolerance_b2 = 0.0f;
    if (flag < 2) {
        double tolerance_y1 = 0.0f, tolerance_y2 = 0.0f;
        double tolerance_x = topRightRect.x;
        if (flag == 1) tolerance_x = topRightRect.x + topRightRect.width;
        if (topRightX != topLeftX) {
            left_right_k = (topRightY - topLeftY) / (double)(topRightX - topLeftX);
            left_right_b = (topRightY - left_right_k * topRightX);
            double tmp_1 = modelSize * 2.5f;
            double tmp_2 = tmp_1 * left_right_k;

            left_right_b_tolerance = sqrt(tmp_1 * tmp_1 + tmp_2 * tmp_2);
            tolerance_b1 = left_right_b - left_right_b_tolerance;
            tolerance_b2 = left_right_b + left_right_b_tolerance;
            tolerance_y1 = left_right_k * tolerance_x + tolerance_b1;
            tolerance_y2 = left_right_k * tolerance_x + tolerance_b2;
        } else {
            return false;
        }
        if (p->getY() < tolerance_y1 || p->getY() > tolerance_y2) return false;
        return true;
    } else {
        double tolerance_x1 = 0.0f, tolerance_x2 = 0.0f;
        if (topRightY != topLeftY) {
            double tolerance_y = topRightRect.y;
            if (flag == 3) tolerance_y = topRightRect.y + topRightRect.height;
            left_right_k = (topRightX - topLeftX) / (double)(topRightY - topLeftY);
            left_right_b = (topRightX - left_right_k * topRightY);
            double tmp_1 = modelSize * 2.5f;
            double tmp_2 = tmp_1 / left_right_k;
            left_right_b_tolerance = sqrt(tmp_1 * tmp_1 + tmp_2 * tmp_2);
            tolerance_b1 = left_right_b - left_right_b_tolerance;
            tolerance_b2 = left_right_b + left_right_b_tolerance;
            tolerance_x1 = left_right_k * tolerance_y + tolerance_b1;
            tolerance_x2 = left_right_k * tolerance_y + tolerance_b2;
            if (p->getX() < tolerance_x1 || p->getX() > tolerance_x2) return false;
            return true;
        } else {
            return false;
        }
    }
}

void Detector::findPointsForLine(Ref<ResultPoint> &topLeft, Ref<ResultPoint> &topRight,
                                 Ref<ResultPoint> &bottomLeft, Rect topRightRect,
                                 Rect bottomLeftRect, vector<Ref<ResultPoint> > &topRightPoints,
                                 vector<Ref<ResultPoint> > &bottomLeftPoints, float modelSize) {
    int topLeftX = topLeft->getX(), topLeftY = topLeft->getY(), topRightX = topRight->getX(),
        topRightY = topRight->getY();
    if (!topRightPoints.empty()) topRightPoints.clear();
    if (!bottomLeftPoints.empty()) bottomLeftPoints.clear();

    int xMin = 0;
    int xMax = 0;
    int yMin = 0;
    int yMax = 0;

    int imgWidth = image_->getWidth();
    int imgHeight = image_->getHeight();

    // [-45, 45] or [135, 180) or [-180, -45)
    if (topLeftY == topRightY || abs((topRightX - topLeftX) / (topRightY - topLeftY)) >= 1) {
        if (topLeftX < topRightX) {
            xMin = topRightRect.x;
            xMax = topRightRect.x + modelSize * 2;
            yMin = topRightRect.y + modelSize;
            yMax = topRightRect.y - modelSize + topRightRect.height;
            // [-45, 45] TopRight: left, black->white points; BottomLeft: top, black->white points
            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);

            for (int i = yMin; i < yMax; i++) {
                for (int j = xMin; j < xMax; j++) {
                    // left->right, black->white
                    if (image_->get(j, i) && !image_->get(j + 1, i)) {
                        Ref<ResultPoint> topRightPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, topRight, topRightRect, modelSize,
                                           topRightPoint, 0)) {
                            topRightPoints.push_back(topRightPoint);
                            break;
                        }
                    }
                }
            }

            xMin = bottomLeftRect.x + modelSize;
            xMax = bottomLeftRect.x - modelSize + bottomLeftRect.width;
            yMin = bottomLeftRect.y;
            yMax = bottomLeftRect.y + 2 * modelSize;

            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);

            for (int j = xMin; j < xMax; j++) {
                for (int i = yMin; i < yMax; i++) {
                    // top to down, black->white
                    if (image_->get(j, i) && !image_->get(j, i + 1)) {
                        Ref<ResultPoint> bottomLeftPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, bottomLeft, bottomLeftRect, modelSize,
                                           bottomLeftPoint, 2)) {
                            bottomLeftPoints.push_back(bottomLeftPoint);
                            break;
                        }
                    }
                }
            }
        } else {
            // white->black points
            xMin = topRightRect.x + topRightRect.width - 2 * modelSize;
            xMax = topRightRect.x + topRightRect.width;
            yMin = topRightRect.y + modelSize;
            yMax = topRightRect.y - modelSize + topRightRect.height;
            // [135, 180) or [-180, -45)  TopRight: right, white->black points; BottomLeft: bottom,
            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);

            for (int i = yMin; i < yMax; i++) {
                for (int j = xMin; j < xMax; j++) {
                    // left->right, white->black
                    if (!image_->get(j, i) && image_->get(j + 1, i)) {
                        Ref<ResultPoint> topRightPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, topRight, topRightRect, modelSize,
                                           topRightPoint, 1)) {
                            topRightPoints.push_back(topRightPoint);
                            break;
                        }
                    }
                }
            }

            xMin = bottomLeftRect.x + modelSize;
            xMax = bottomLeftRect.x - modelSize + bottomLeftRect.width;
            yMin = bottomLeftRect.y + bottomLeftRect.height - 2 * modelSize;
            yMax = bottomLeftRect.y + bottomLeftRect.height;

            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);

            for (int j = xMin; j < xMax; j++) {
                for (int i = yMin; i < yMax; i++) {
                    // top to down, white->black
                    if (!image_->get(j, i) && image_->get(j, i + 1)) {
                        Ref<ResultPoint> bottomLeftPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, bottomLeft, bottomLeftRect, modelSize,
                                           bottomLeftPoint, 3)) {
                            bottomLeftPoints.push_back(bottomLeftPoint);
                            break;
                        }
                    }
                }
            }
        }
    } else {
        // (45, 135) or (-45, -135)
        // (45, 135) TopRight: top, black->white; BottomRight: right, black->white
        if (topLeftY < topRightY) {
            xMin = topRightRect.x + modelSize;
            xMax = topRightRect.x - modelSize + topRightRect.width;
            yMin = topRightRect.y;
            yMax = topRightRect.y + 2 * modelSize;

            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);

            for (int j = xMin; j < xMax; j++) {
                for (int i = yMin; i < yMax; i++) {
                    // top to down, black->white
                    if (image_->get(j, i) && !image_->get(j, i + 1)) {
                        Ref<ResultPoint> topRightPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, topRight, topRightRect, modelSize,
                                           topRightPoint, 2)) {
                            topRightPoints.push_back(topRightPoint);
                            break;
                        }
                    }
                }
            }

            xMin = topRightRect.x + topRightRect.width - 2 * modelSize;
            xMax = topRightRect.x + topRightRect.width;
            yMin = topRightRect.y + modelSize;
            yMax = topRightRect.y - modelSize + topRightRect.height;

            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);

            for (int i = yMin; i < yMax; i++) {
                for (int j = xMin; j < xMax; j++) {
                    // left to right, white-> black
                    if (!image_->get(j, i) && image_->get(j + 1, i)) {
                        Ref<ResultPoint> bottomLeftPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, bottomLeft, bottomLeftRect, modelSize,
                                           bottomLeftPoint, 1)) {
                            bottomLeftPoints.push_back(bottomLeftPoint);
                            break;
                        }
                    }
                }
            }
        } else {
            // (-45, -135) TopRight: bottom, white->black; BottomRight: left, black->white
            xMin = topRightRect.x + modelSize;
            xMax = topRightRect.x - modelSize + topRightRect.width;
            yMin = topRightRect.y + topRightRect.height - 2 * modelSize;
            yMax = topRightRect.y + topRightRect.height;

            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);

            for (int j = xMin; j < xMax; j++) {
                for (int i = yMin; i < yMax; i++) {
                    // top to down, white->balck
                    if (!image_->get(j, i) && image_->get(j, i + 1)) {
                        Ref<ResultPoint> topRightPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, topRight, topRightRect, modelSize,
                                           topRightPoint, 3)) {
                            topRightPoints.push_back(topRightPoint);
                            break;
                        }
                    }
                }
            }

            xMin = bottomLeftRect.x;
            xMax = bottomLeftRect.x + 2 * modelSize;
            yMin = bottomLeftRect.y + modelSize;
            yMax = bottomLeftRect.y + bottomLeftRect.height - modelSize;

            MathUtils::getRangeValues(xMin, xMax, 0, imgWidth - 1);
            MathUtils::getRangeValues(yMin, yMax, 0, imgHeight - 1);

            for (int i = yMin; i < yMax; i++) {
                for (int j = xMin; j < xMax; j++) {
                    // left to right, black->white
                    if (image_->get(j, i) && !image_->get(j + 1, i)) {
                        Ref<ResultPoint> bottomLeftPoint(new ResultPoint(j, i));
                        if (checkTolerance(topLeft, bottomLeft, bottomLeftRect, modelSize,
                                           bottomLeftPoint, 0)) {
                            bottomLeftPoints.push_back(bottomLeftPoint);
                            break;
                        }
                    }
                }
            }
        }
    }
}

Ref<PerspectiveTransform> Detector::createTransform(Ref<FinderPatternInfo> info,
                                                    Ref<ResultPoint> alignmentPattern,
                                                    int dimension) {
    Ref<FinderPattern> topLeft(info->getTopLeft());
    Ref<FinderPattern> topRight(info->getTopRight());
    Ref<FinderPattern> bottomLeft(info->getBottomLeft());
    Ref<PerspectiveTransform> transform =
        createTransform(topLeft, topRight, bottomLeft, alignmentPattern, dimension);
    return transform;
}

Ref<PerspectiveTransform> Detector::createTransform(Ref<ResultPoint> topLeft,
                                                    Ref<ResultPoint> topRight,
                                                    Ref<ResultPoint> bottomLeft,
                                                    Ref<ResultPoint> alignmentPattern,
                                                    int dimension) {
    float dimMinusThree = (float)dimension - 3.5f;
    float bottomRightX;
    float bottomRightY;
    float sourceBottomRightX;
    float sourceBottomRightY;
    if (alignmentPattern && alignmentPattern->getX()) {
        bottomRightX = alignmentPattern->getX();
        bottomRightY = alignmentPattern->getY();
        sourceBottomRightX = dimMinusThree - 3.0f;
        sourceBottomRightY = sourceBottomRightX;
    } else {
        // Don't have an alignment pattern, just make up the bottom-right point
        bottomRightX = (topRight->getX() - topLeft->getX()) + bottomLeft->getX();
        bottomRightY = (topRight->getY() - topLeft->getY()) + bottomLeft->getY();
        float deltaX = topLeft->getX() - bottomLeft->getX();
        float deltaY = topLeft->getY() - bottomLeft->getY();
        if (fabs(deltaX) < fabs(deltaY))
            deltaY = topLeft->getY() - topRight->getY();
        else
            deltaX = topLeft->getX() - topRight->getX();
        bottomRightX += 2 * deltaX;
        bottomRightY += 2 * deltaY;
        sourceBottomRightX = dimMinusThree;
        sourceBottomRightY = dimMinusThree;
    }
    Ref<PerspectiveTransform> transform(PerspectiveTransform::quadrilateralToQuadrilateral(
        3.5f, 3.5f, dimMinusThree, 3.5f, sourceBottomRightX, sourceBottomRightY, 3.5f,
        dimMinusThree, topLeft->getX(), topLeft->getY(), topRight->getX(), topRight->getY(),
        bottomRightX, bottomRightY, bottomLeft->getX(), bottomLeft->getY()));
    return transform;
}

// Computes the dimension (number of modules on a size) of the QR code based on
// the position of the finder patterns and estimated module size.
int Detector::computeDimension(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                               Ref<ResultPoint> bottomLeft, float moduleSizeX, float moduleSizeY) {
    int tltrCentersDimension = ResultPoint::distance(topLeft, topRight) / moduleSizeX;
    int tlblCentersDimension = ResultPoint::distance(topLeft, bottomLeft) / moduleSizeY;

    float tmp_dimension = ((tltrCentersDimension + tlblCentersDimension) / 2.0) + 7.0;
    int dimension = cvRound(tmp_dimension);
    int mod = dimension & 0x03;  // mod 4

    switch (mod) {  // mod 4
        case 0:
            dimension++;
            break;
            // 1? do nothing
        case 2:
            dimension--;
            break;
    }
    return dimension;
}

bool Detector::checkConvexQuadrilateral(Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                                        Ref<ResultPoint> bottomLeft, Ref<ResultPoint> bottomRight) {
    float v1[2];
    float v2[2];
    float v3[2];
    float v4[2];

    v1[0] = topLeft->getX() - topRight->getX();
    v1[1] = topLeft->getY() - topRight->getY();
    v2[0] = topRight->getX() - bottomRight->getX();
    v2[1] = topRight->getY() - bottomRight->getY();
    v3[0] = bottomRight->getX() - bottomLeft->getX();
    v3[1] = bottomRight->getY() - bottomLeft->getY();
    v4[0] = bottomLeft->getX() - topLeft->getX();
    v4[1] = bottomLeft->getY() - topLeft->getY();

    float c1 = MathUtils::VecCross(v1, v2);
    float c2 = MathUtils::VecCross(v2, v3);
    float c3 = MathUtils::VecCross(v3, v4);
    float c4 = MathUtils::VecCross(v4, v1);

    if ((c1 < 0.0 && c2 < 0.0 && c3 < 0.0 && c4 < 0.0) ||
        (c1 > 0.0 && c2 > 0.0 && c3 > 0.0 && c4 > 0.0))
        return true;
    else
        return false;
}
