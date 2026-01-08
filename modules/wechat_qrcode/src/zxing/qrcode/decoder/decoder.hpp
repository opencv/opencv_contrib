// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DECODER_DECODER_HPP__
#define __ZXING_QRCODE_DECODER_DECODER_HPP__

#include "../../common/array.hpp"
#include "../../common/bitmatrix.hpp"
#include "../../common/counted.hpp"
#include "../../common/decoder_result.hpp"
#include "../../common/detector_result.hpp"
#include "../../common/reedsolomon/reed_solomon_decoder.hpp"
#include "../../errorhandler.hpp"
#include "../version.hpp"
#include "bitmatrixparser.hpp"

namespace zxing {
namespace qrcode {

class Decoder {
public:
    enum DecoderState {
        NOTSTART = 19,
        START = 20,
        READVERSION = 21,
        READERRORCORRECTIONLEVEL = 22,
        READCODEWORDSORRECTIONLEVEL = 23,
        FINISH = 24
    };

private:
    DecoderState decoderState_;
    float possibleFix_;
    ReedSolomonDecoder rsDecoder_;
    void correctErrors(ArrayRef<char> bytes, int numDataCodewords, ErrorHandler& err_handler);

public:
    Decoder();
    Ref<DecoderResult> decode(Ref<BitMatrix> bits, ErrorHandler& err_handler);

private:
    Ref<DecoderResult> decode(Ref<BitMatrix> bits, bool isMirror, ErrorHandler& err_handler);

    float estimateFixedPattern(Ref<BitMatrix> bits, Version* version, ErrorHandler& err_handler);

private:
    unsigned int possibleVersion_;

public:
    unsigned int getPossibleVersion();
    DecoderState getState() { return decoderState_; }
    float getPossibleFix() { return possibleFix_; }
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DECODER_DECODER_HPP__
