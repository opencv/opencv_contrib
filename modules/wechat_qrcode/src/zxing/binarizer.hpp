// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

/*
 *  binarizer.hpp
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
#ifndef __ZXING_BINARIZER_HPP__
#define __ZXING_BINARIZER_HPP__

#include "zxing/common/bitarray.hpp"
#include "zxing/common/bitmatrix.hpp"
#include "zxing/common/counted.hpp"
#include "zxing/errorhandler.hpp"
#include "zxing/luminance_source.hpp"

#define ONED_ENABLE_LINE_BINARIZER

namespace zxing {

// typedef unsigned char uint8_t;

struct BINARIZER_BLOCK {
    int sum;
    int min;
    int max;
    int threshold;
    // int average;
};

#ifdef ONED_ENABLE_LINE_BINARIZER
struct DecodeTipInfo {
    int class_id;
};
#endif

class Binarizer : public Counted {
private:
    Ref<LuminanceSource> source_;
    bool histogramBinarized;
    bool usingHistogram;

public:
    explicit Binarizer(Ref<LuminanceSource> source);
    virtual ~Binarizer();

    // Added for store binarized result

    int dataWidth;
    int dataHeight;
    int width;
    int height;

    // Store dynamicalli choice of which matrix is currently used
    Ref<BitMatrix> matrix_;

    // Restore 0 degree result
    Ref<BitMatrix> matrix0_;

    Ref<BitMatrix> matrixInverted_;

    bool isRotateSupported() const { return false; }

    // rotate counter clockwise 45 & 90 degree from binarized cache
    int rotateCounterClockwise();
    int rotateCounterClockwise45();

    // bool isHistogramBinarized() const{
    //	return histogramBinarized;
    //}
    // void setUsingHistogram(bool use){
    //	usingHistogram=use;
    //	if(!usingHistogram)
    //		histogramBinarized=false;
    //}
    // bool getUsingHisogram(){
    //	return usingHistogram;
    //}

    // void binarizeByHistogram();

    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler& err_handler);
    virtual Ref<BitMatrix> getInvertedMatrix(ErrorHandler& err_handler);
    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler& err_handler);

    Ref<LuminanceSource> getLuminanceSource() const;
    // virtual Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source) = 0;
    virtual Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source) {
        return Ref<Binarizer>(new Binarizer(source));
    };

    int getWidth() const;
    int getHeight() const;

    ArrayRef<BINARIZER_BLOCK> getBlockArray(int size);
};

}  // namespace zxing
#endif  // __ZXING_BINARIZER_HPP__
