// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_HYBRID_BINARIZER_HPP__
#define __ZXING_COMMON_HYBRID_BINARIZER_HPP__

#include "../../binarizer.hpp"
#include "../../errorhandler.hpp"
#include "../bitarray.hpp"
#include "../bitmatrix.hpp"
#include "../bytematrix.hpp"
#include "global_histogram_binarizer.hpp"

#include <vector>


namespace zxing {

class HybridBinarizer : public GlobalHistogramBinarizer {
private:
    Ref<ByteMatrix> grayByte_;
    // ArrayRef<int> integral_;
    ArrayRef<int> blockIntegral_;
    ArrayRef<BINARIZER_BLOCK> blocks_;

    ArrayRef<int> blackPoints_;
    int level_;

    int subWidth_;
    int subHeight_;
    int blockIntegralWidth;
    int blockIntegralHeight;

public:
    explicit HybridBinarizer(Ref<LuminanceSource> source);
    virtual ~HybridBinarizer();

    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler& err_handler) override;
    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler& err_handler) override;

    Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source) override;

private:
    int initIntegral();
    int initBlockIntegral();
    int initBlocks();

    // int calculateBlackPoints();
    ArrayRef<int> getBlackPoints();
    int getBlockThreshold(int x, int y, int subWidth, int sum, int min, int max,
                          int minDynamicRange, int SIZE_POWER);


    void calculateThresholdForBlock(Ref<ByteMatrix>& luminances, int subWidth, int subHeight,
                                    int SIZE_POWER, Ref<BitMatrix> const& matrix,
                                    ErrorHandler& err_handler);


    void thresholdBlock(Ref<ByteMatrix>& luminances, int xoffset, int yoffset, int threshold,
                        Ref<BitMatrix> const& matrix, ErrorHandler& err_handler);

    void thresholdIrregularBlock(Ref<ByteMatrix>& luminances, int xoffset, int yoffset,
                                 int blockWidth, int blockHeight, int threshold,
                                 Ref<BitMatrix> const& matrix, ErrorHandler& err_handler);

#ifdef USE_SET_INT
    void thresholdFourBlocks(Ref<ByteMatrix>& luminances, int xoffset, int yoffset, int* thresholds,
                             int stride, Ref<BitMatrix> const& matrix);
#endif

    // Add for binarize image when call getBlackMatrix
    // By Skylook
    int binarizeByBlock(ErrorHandler& err_handler);
};

}  // namespace zxing

#endif  // __ZXING_COMMON_HYBRID_BINARIZER_HPP__
