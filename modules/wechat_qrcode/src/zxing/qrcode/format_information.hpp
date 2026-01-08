// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_FORMAT_INFORMATION_HPP__
#define __ZXING_QRCODE_FORMAT_INFORMATION_HPP__

#include "../common/counted.hpp"
#include "../errorhandler.hpp"
#include "error_correction_level.hpp"

namespace zxing {
namespace qrcode {

class FormatInformation : public Counted {
private:
    static int FORMAT_INFO_MASK_QR;
    static int FORMAT_INFO_DECODE_LOOKUP[][2];
    static int N_FORMAT_INFO_DECODE_LOOKUPS;
    static int BITS_SET_IN_HALF_BYTE[];

    ErrorCorrectionLevel &errorCorrectionLevel_;
    char dataMask_;
    float possiableFix_;

    FormatInformation(int formatInfo, float possiableFix, ErrorHandler &err_handler);

public:
    static int numBitsDiffering(int a, int b);
    static Ref<FormatInformation> decodeFormatInformation(int maskedFormatInfo1,
                                                          int maskedFormatInfo2);
    static Ref<FormatInformation> doDecodeFormatInformation(int maskedFormatInfo1,
                                                            int maskedFormatInfo2);
    ErrorCorrectionLevel &getErrorCorrectionLevel();
    char getDataMask();
    float getPossiableFix();
    friend bool operator==(const FormatInformation &a, const FormatInformation &b);
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_FORMAT_INFORMATION_HPP__
