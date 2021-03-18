// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DECODER_DECODEDBITSTREAMPARSER_HPP__
#define __ZXING_QRCODE_DECODER_DECODEDBITSTREAMPARSER_HPP__

#include "../../common/array.hpp"
#include "../../common/bitsource.hpp"
#include "../../common/characterseteci.hpp"
#include "../../common/counted.hpp"
#include "../../common/decoder_result.hpp"
#include "../../decodehints.hpp"
#include "../../errorhandler.hpp"
#include "mode.hpp"


namespace zxing {
namespace qrcode {

class DecodedBitStreamParser {
public:
    DecodedBitStreamParser() : outputCharset("UTF-8") {}

private:
    static char const ALPHANUMERIC_CHARS[];

    string outputCharset;
    // string outputCharset;

    char toAlphaNumericChar(size_t value, ErrorHandler& err_handler);

    void decodeHanziSegment(Ref<BitSource> bits, std::string& result, int count,
                            ErrorHandler& err_handler);
    void decodeKanjiSegment(Ref<BitSource> bits, std::string& result, int count,
                            ErrorHandler& err_handler);
    void decodeByteSegment(Ref<BitSource> bits, std::string& result, int count);
    void decodeByteSegment(Ref<BitSource> bits_, std::string& result, int count,
                           zxing::common::CharacterSetECI* currentCharacterSetECI,
                           ArrayRef<ArrayRef<char> >& byteSegments, ErrorHandler& err_handler);
    void decodeAlphanumericSegment(Ref<BitSource> bits, std::string& result, int count,
                                   bool fc1InEffect, ErrorHandler& err_handler);
    void decodeNumericSegment(Ref<BitSource> bits, std::string& result, int count,
                              ErrorHandler& err_handler);

    void append(std::string& ost, const char* bufIn, size_t nIn, ErrorHandler& err_handler);
    void append(std::string& ost, std::string const& in, ErrorHandler& err_handler);

public:
    Ref<DecoderResult> decode(ArrayRef<char> bytes, Version* version,
                              ErrorCorrectionLevel const& ecLevel, ErrorHandler& err_handler,
                              int iVersion = -1);

    // string getResultCharset();
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DECODER_DECODEDBITSTREAMPARSER_HPP__
