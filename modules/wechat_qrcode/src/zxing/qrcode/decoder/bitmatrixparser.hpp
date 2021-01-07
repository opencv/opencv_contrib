// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

/*
 *  bitmatrixparser.hpp
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
#ifndef __ZXING_QRCODE_DECODER_BITMATRIXPARSER_HPP__
#define __ZXING_QRCODE_DECODER_BITMATRIXPARSER_HPP__

#include "zxing/common/array.hpp"
#include "zxing/common/bitmatrix.hpp"
#include "zxing/common/counted.hpp"
#include "zxing/errorhandler.hpp"
#include "zxing/qrcode/format_information.hpp"
#include "zxing/qrcode/version.hpp"
#include "zxing/reader_exception.hpp"

namespace zxing {
namespace qrcode {

class BitMatrixParser : public Counted {
private:
    Ref<BitMatrix> bitMatrix_;
    Version *parsedVersion_;
    Ref<FormatInformation> parsedFormatInfo_;
    bool mirror_;

    int copyBit(size_t x, size_t y, int versionBits);

public:
    BitMatrixParser(Ref<BitMatrix> bitMatrix, ErrorHandler &err_handler);
    Ref<FormatInformation> readFormatInformation(ErrorHandler &err_handler);
    Version *readVersion(ErrorHandler &err_handler);
    ArrayRef<char> readCodewords(ErrorHandler &err_handler);

public:
    void remask();
    void setMirror(bool mirror);
    void mirror();
    void mirrorH();

private:
    BitMatrixParser(const BitMatrixParser &);
    BitMatrixParser &operator=(const BitMatrixParser &);
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DECODER_BITMATRIXPARSER_HPP__
