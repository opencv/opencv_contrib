// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DECODER_DATABLOCK_HPP__
#define __ZXING_QRCODE_DECODER_DATABLOCK_HPP__

#include "zxing/common/array.hpp"
#include "zxing/common/counted.hpp"
#include "zxing/errorhandler.hpp"
#include "zxing/qrcode/error_correction_level.hpp"
#include "zxing/qrcode/version.hpp"

#include <vector>

namespace zxing {
namespace qrcode {

class DataBlock : public Counted {
private:
    int numDataCodewords_;
    ArrayRef<char> codewords_;

    DataBlock(int numDataCodewords, ArrayRef<char> codewords);

public:
    static std::vector<Ref<DataBlock> > getDataBlocks(ArrayRef<char> rawCodewords, Version *version,
                                                      ErrorCorrectionLevel &ecLevel,
                                                      ErrorHandler &err_handler);

    int getNumDataCodewords();
    ArrayRef<char> getCodewords();
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DECODER_DATABLOCK_HPP__
