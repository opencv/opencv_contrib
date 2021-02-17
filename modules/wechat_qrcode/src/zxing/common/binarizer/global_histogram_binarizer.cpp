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
#include "global_histogram_binarizer.hpp"
using zxing::GlobalHistogramBinarizer;

namespace {
const int LUMINANCE_BITS = 5;
const int LUMINANCE_SHIFT = 8 - LUMINANCE_BITS;
const int LUMINANCE_BUCKETS = 1 << LUMINANCE_BITS;
const ArrayRef<char> EMPTY(0);
}  // namespace

GlobalHistogramBinarizer::GlobalHistogramBinarizer(Ref<LuminanceSource> source)
    : Binarizer(source), luminances(EMPTY), buckets(LUMINANCE_BUCKETS) {
    filtered = false;
}

GlobalHistogramBinarizer::~GlobalHistogramBinarizer() {}

void GlobalHistogramBinarizer::initArrays(int luminanceSize) {
    if (luminances->size() < luminanceSize) {
        luminances = ArrayRef<char>(luminanceSize);
    }
    for (int x = 0; x < LUMINANCE_BUCKETS; x++) {
        buckets[x] = 0;
    }
}

// Applies simple sharpening to the row data to improve performance of the 1D
// readers.
Ref<BitArray> GlobalHistogramBinarizer::getBlackRow(int y, Ref<BitArray> row,
                                                    ErrorHandler& err_handler) {
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_) {
        binarizeImage0(err_handler);
        if (err_handler.ErrCode()) return Ref<BitArray>();
    }
    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackRow(y, row, err_handler);
}

// Does not sharpen the data, as this call is intended to only be used by 2D
// readers.
Ref<BitMatrix> GlobalHistogramBinarizer::getBlackMatrix(ErrorHandler& err_handler) {
    binarizeImage0(err_handler);
    if (err_handler.ErrCode()) return Ref<BitMatrix>();
    // First call binarize image in child class to get matrix0_ and binCache
    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackMatrix(err_handler);
}

using namespace std;

int GlobalHistogramBinarizer::estimateBlackPoint(ArrayRef<int> const& _buckets,
                                                 ErrorHandler& err_handler) {
    // Find tallest peak in histogram
    int numBuckets = _buckets->size();
    int maxBucketCount = 0;
    int firstPeak = 0;
    int firstPeakSize = 0;
    for (int x = 0; x < numBuckets; x++) {
        if (_buckets[x] > firstPeakSize) {
            firstPeak = x;
            firstPeakSize = _buckets[x];
        }
        if (_buckets[x] > maxBucketCount) {
            maxBucketCount = _buckets[x];
        }
    }

    // Find second-tallest peak -- well, another peak that is tall and not
    // so close to the first one
    int secondPeak = 0;
    int secondPeakScore = 0;
    for (int x = 0; x < numBuckets; x++) {
        int distanceToBiggest = x - firstPeak;
        // Encourage more distant second peaks by multiplying by square of
        // distance
        int score = _buckets[x] * distanceToBiggest * distanceToBiggest;
        if (score > secondPeakScore) {
            secondPeak = x;
            secondPeakScore = score;
        }
    }
    // Make sure firstPeak corresponds to the black peak.
    if (firstPeak > secondPeak) {
        int temp = firstPeak;
        firstPeak = secondPeak;
        secondPeak = temp;
    }

    // Kind of arbitrary; if the two peaks are very close, then we figure there
    // is so little dynamic range in the image, that discriminating black and
    // white is too error-prone. Decoding the image/line is either pointless, or
    // may in some cases lead to a false positive for 1D formats, which are
    // relatively lenient. We arbitrarily say "close" is  "<= 1/16 of the total
    // histogram buckets apart" std::cerr << "! " << secondPeak << " " <<
    // firstPeak << " " << numBuckets << std::endl;
    if (secondPeak - firstPeak <= numBuckets >> 4) {
        err_handler = NotFoundErrorHandler("NotFound GlobalHistogramBinarizer");
        return -1;
    }

    // Find a valley between them that is low and closer to the white peak
    int bestValley = secondPeak - 1;
    int bestValleyScore = -1;
    for (int x = secondPeak - 1; x > firstPeak; x--) {
        int fromFirst = x - firstPeak;
        // Favor a "valley" that is not too close to either peak -- especially
        // not the black peak -- and that has a low value of course
        int score = fromFirst * fromFirst * (secondPeak - x) * (maxBucketCount - buckets[x]);
        if (score > bestValleyScore) {
            bestValley = x;
            bestValleyScore = score;
        }
    }

    // std::cerr << "bps " << (bestValley << LUMINANCE_SHIFT) << std::endl;
    return bestValley << LUMINANCE_SHIFT;
}

// codes from sagazhou, only works well on one dataset
int GlobalHistogramBinarizer::estimateBlackPoint2(ArrayRef<int> const& _buckets) {
    int midValue = LUMINANCE_BUCKETS / 2 + 1;
    // Find tallest and lowest peaks in histogram
    // const int numBuckets = buckets->size();
    int maxPointArray[LUMINANCE_BUCKETS] = {0};
    int maxCrusor = 0;
    int maxValue = 0, maxIndex = 0;
    int minPointArray[LUMINANCE_BUCKETS] = {0};
    int minCrusor = 0;

    for (int i = 2; i < LUMINANCE_BUCKETS - 3; i++) {
        if (_buckets[i] < _buckets[i + 1] && _buckets[i] < _buckets[i + 2] &&
            _buckets[i] < _buckets[i - 1] && _buckets[i] < _buckets[i - 2]) {
            minPointArray[minCrusor++] = i;
        } else if (_buckets[i] > _buckets[i + 1] && _buckets[i] > _buckets[i + 2] &&
                   _buckets[i] > _buckets[i - 1] && _buckets[i] > _buckets[i - 2]) {
            maxPointArray[maxCrusor++] = i;
            if (_buckets[i] > maxValue) {
                maxValue = _buckets[i];
                maxIndex = i;
            }
        }
    }
    bool bSlantBlack = true;
    // most pixels are black
    for (int i = 0; i < maxCrusor; ++i) {
        if (maxPointArray[i] > midValue) {
            bSlantBlack = false;
            break;
        }
    }

    bool bSlantWhite = true;
    // most pixels are white
    for (int i = 0; i < maxCrusor; ++i) {
        if (maxPointArray[i] < midValue) {
            bSlantWhite = false;
            break;
        }
    }

    if (bSlantBlack) {
        int start = maxIndex + 30;
        int end = midValue;

        if (minCrusor == 0)  // unimodal
        {
            return 255;
        } else {
            int mostLeftIndex = 0;
            bool bFind = false;

            for (int i = 0; i < minCrusor; ++i)  // wave motion
            {
                if (minPointArray[i] > start && minPointArray[i] < end) {
                    mostLeftIndex = minPointArray[i];
                    bFind = true;
                    break;
                }
            }

            if (bFind) {
                return mostLeftIndex;
            } else {
                return 255;
            }
        }
    }

    if (bSlantWhite) {
        int start = midValue;
        int end = maxIndex - 30;

        if (minCrusor == 0)  // unimodal
        {
            return 0;
        } else {
            int mostRightIndex = 0;
            bool bFind = false;

            for (int i = 0; i < minCrusor; ++i)  // wave motion
            {
                if (minPointArray[i] > start && minPointArray[i] < end) {
                    mostRightIndex = i;
                    bFind = true;
                }
            }

            if (bFind) {
                return mostRightIndex;
            } else {
                return 0;
            }
        }
    }

    // balanced distribution
    if (maxIndex < midValue) {
        // the minest min value
        if (minCrusor == 0) {
            return 255;  // all black
        } else {
            int start = maxIndex + 30;
            int end = 253;

            for (int i = 0; i < minCrusor; ++i)  // wave motion
            {
                if (minPointArray[i] > start && minPointArray[i] < end) {
                    return minPointArray[i];
                }
            }
        }
    } else {
        // maxest min value
        if (minCrusor == 0) {
            return 0;  // white
        } else {
            int start = 0;
            int end = maxIndex - 30;
            int mostRightIndex = 0;

            for (int i = 0; i < minCrusor; ++i)  // wave motion
            {
                if (minPointArray[i] > start && minPointArray[i] < end) {
                    mostRightIndex = minPointArray[i];
                }
            }

            return mostRightIndex;
        }
    }
    return 0;
}

int GlobalHistogramBinarizer::binarizeImage0(ErrorHandler& err_handler) {
    LuminanceSource& source = *getLuminanceSource();
    Ref<BitMatrix> matrix(new BitMatrix(width, height, err_handler));
    if (err_handler.ErrCode()) return -1;
    // Quickly calculates the histogram by sampling four rows from the image.
    // This proved to be more robust on the blackbox tests than sampling a
    // diagonal as we used to do.
    initArrays(width);
    ArrayRef<int> localBuckets = buckets;

    for (int y = 1; y < 5; y++) {
        int row = height * y / 5;
        ArrayRef<char> localLuminances = source.getRow(row, luminances, err_handler);
        if (err_handler.ErrCode()) return -1;
        int right = (width << 2) / 5;
        for (int x = width / 5; x < right; x++) {
            int pixel = localLuminances[x] & 0xff;
            localBuckets[pixel >> LUMINANCE_SHIFT]++;
        }
    }


    int blackPoint = estimateBlackPoint(localBuckets, err_handler);
    if (err_handler.ErrCode()) return -1;

    ArrayRef<char> localLuminances = source.getMatrix();
    for (int y = 0; y < height; y++) {
        int offset = y * width;
        for (int x = 0; x < width; x++) {
            int pixel = localLuminances[offset + x] & 0xff;
            if (pixel < blackPoint) {
                matrix->set(x, y);
            }
        }
    }

    matrix0_ = matrix;

    return 0;
}

Ref<Binarizer> GlobalHistogramBinarizer::createBinarizer(Ref<LuminanceSource> source) {
    return Ref<Binarizer>(new GlobalHistogramBinarizer(source));
}
