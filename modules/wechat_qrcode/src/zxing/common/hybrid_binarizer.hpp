// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

/*
 *  hybrid_binarizer.hpp
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
#ifndef __ZXING_COMMON_HYBRID_BINARIZER_HPP__
#define __ZXING_COMMON_HYBRID_BINARIZER_HPP__

#include "zxing/binarizer.hpp"
#include "zxing/common/bitarray.hpp"
#include "zxing/common/bitmatrix.hpp"
#include "zxing/common/bytematrix.hpp"
#include "zxing/common/global_histogram_binarizer.hpp"
#include "zxing/errorhandler.hpp"

#include <vector>

// Macro to use max-min in function calculateBlackPoints
#ifndef USE_MAX_MIN
#define USE_MAX_MIN 0
#endif

#ifndef USE_GOOGLE_CODE
#define USE_GOOGLE_CODE 0
#endif

#define USE_LEVEL_BINARIZER 1

// Macro to set the 8 bits one time in function calculateThresholdForBlock
// and BitMatrix::setEightOneTime
//#ifndef USE_SET_EIGHT
//#define USE_SET_EIGHT 0
//#endif

// Macro to set the entire integer one time in function
// calculateThresholdForBlock and BitMatrix::setIntOneTime
//#ifndef USE_SET_INT
//#ifndef USE_GOOGLE_CODE
//#define USE_SET_INT 0
//#endif
//#endif

namespace zxing {

class HybridBinarizer : public GlobalHistogramBinarizer {
private:
    // Ref<BitMatrix> matrix_;
    // Ref<BitArray> cached_row_;

    // bool onedFirstGetBlackRow;

    Ref<ByteMatrix> grayByte_;
    // ArrayRef<int> integral_;
    ArrayRef<int> blockIntegral_;
    ArrayRef<BINARIZER_BLOCK> blocks_;

    ArrayRef<int> blackPoints_;
    int level_;

    int width_;
    int height_;
    int subWidth_;
    int subHeight_;

public:
    explicit HybridBinarizer(Ref<LuminanceSource> source);
    virtual ~HybridBinarizer();

    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler& err_handler);
    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler& err_handler);

    Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source);

private:
#ifdef USE_LEVEL_BINARIZER
    int initIntegral();
    int initBlockIntegral();
    int initBlocks();

    // int calculateBlackPoints();
    ArrayRef<int> getBlackPoints(int level);
    int getBlockThreshold(int x, int y, int subWidth, int sum, int min, int max,
                          int minDynamicRange, int SIZE_POWER);
#else
    ArrayRef<int> calculateBlackPoints(Ref<ByteMatrix>& luminances, int subWidth, int subHeight,
                                       int width, int height);
#endif

#ifdef USE_LEVEL_BINARIZER
    void calculateThresholdForBlock(Ref<ByteMatrix>& luminances, int subWidth, int subHeight,
                                    int width, int height, int SIZE_POWER,
                                    // ArrayRef<int> &blackPoints,
                                    Ref<BitMatrix> const& matrix, ErrorHandler& err_handler);
#else
    void calculateThresholdForBlock(Ref<ByteMatrix>& luminances, int subWidth, int subHeight,
                                    int width, int height, ArrayRef<int>& blackPoints,
                                    Ref<BitMatrix> const& matrix, ErrorHandler& err_handler);
#endif

    void thresholdBlock(Ref<ByteMatrix>& luminances, int xoffset, int yoffset, int threshold,
                        int stride, Ref<BitMatrix> const& matrix, ErrorHandler& err_handler);

    void thresholdIrregularBlock(Ref<ByteMatrix>& luminances, int xoffset, int yoffset,
                                 int blockWidth, int blockHeight, int threshold, int stride,
                                 Ref<BitMatrix> const& matrix, ErrorHandler& err_handler);

#ifdef USE_SET_INT
    void thresholdFourBlocks(Ref<ByteMatrix>& luminances, int xoffset, int yoffset, int* thresholds,
                             int stride, Ref<BitMatrix> const& matrix);
#endif

    // Add for binarize image when call getBlackMatrix
    // By Skylook
    int binarizeByBlock(int blockLevel, ErrorHandler& err_handler);
};

}  // namespace zxing

#endif  // __ZXING_COMMON_HYBRID_BINARIZER_HPP__
