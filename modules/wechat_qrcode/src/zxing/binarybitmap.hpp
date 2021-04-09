// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_BINARYBITMAP_HPP__
#define __ZXING_BINARYBITMAP_HPP__

#include "binarizer.hpp"
#include "common/bitarray.hpp"
#include "common/bitmatrix.hpp"
#include "common/counted.hpp"
#include "common/unicomblock.hpp"
#include "errorhandler.hpp"

namespace zxing {

class BinaryBitmap : public Counted {
private:
    Ref<Binarizer> binarizer_;

public:
    explicit BinaryBitmap(Ref<Binarizer> binarizer);
    virtual ~BinaryBitmap();

    Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler& err_handler);
    Ref<BitMatrix> getBlackMatrix(ErrorHandler& err_handler);
    Ref<BitMatrix> getInvertedMatrix(ErrorHandler& err_handler);

    Ref<LuminanceSource> getLuminanceSource() const;
    Ref<UnicomBlock> m_poUnicomBlock;

    int getWidth() const;
    int getHeight() const;

    bool isRotateSupported() const;
    Ref<BinaryBitmap> rotateCounterClockwise();

    bool isCropSupported() const;
    Ref<BinaryBitmap> crop(int left, int top, int width, int height, ErrorHandler& err_handler);

    bool isHistogramBinarized() const;
    bool ifUseHistogramBinarize() const;
};

}  // namespace zxing

#endif  // __ZXING_BINARYBITMAP_HPP__
