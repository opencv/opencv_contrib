// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

/*
 *  line_binarizer.hpp
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
#ifndef __ZXING_COMMON_LINE_BINARIZER_HPP__
#define __ZXING_COMMON_LINE_BINARIZER_HPP__

#include "zxing/binarizer.hpp"
#include "zxing/common/bitarray.hpp"
#include "zxing/common/bitmatrix.hpp"
#include "zxing/common/global_histogram_binarizer.hpp"
#include "zxing/errorhandler.hpp"

#include <vector>

namespace zxing {

class LineBinarizer : public GlobalHistogramBinarizer {
private:
public:
    explicit LineBinarizer(Ref<LuminanceSource> source);
    virtual ~LineBinarizer();

    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler &err_handler) override;
    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler &err_handler) override;
    Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source) override;

private:
    int binarizeImage(ErrorHandler &err_handler);
    void binarizeImage(const unsigned char *src, unsigned char *dst, int width, int height);
    bool binarizeLine(const unsigned char *src, unsigned char *dst, int width);
    void scanLine(const unsigned char *line, int width, std::vector<short> &maxiam_index,
                  std::vector<short> &miniam_index);
};

}  // namespace zxing

#endif  // __ZXING_COMMON_LINE_BINARIZER_HPP__
