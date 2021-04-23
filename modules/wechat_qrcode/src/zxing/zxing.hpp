// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_ZXING_HPP__
#define __ZXING_ZXING_HPP__


#define COUNTER_TYPE short


#define ZXING_ARRAY_LEN(v) ((int)(sizeof(v) / sizeof(v[0])))
#define ZX_LOG_DIGITS(digits) \
    ((digits == 8)            \
         ? 3                  \
         : ((digits == 16)    \
                ? 4           \
                : ((digits == 32) ? 5 : ((digits == 64) ? 6 : ((digits == 128) ? 7 : (-1))))))

#ifndef USE_QRCODE_ONLY
#define USE_ONED_WRITER 1
#endif

#if defined(__ANDROID_API__) || defined(_MSC_VER)

#ifndef NO_ICONV
#define NO_ICONV
#endif

#endif



#ifndef NO_ICONV_INSIDE
#define NO_ICONV_INSIDE
#endif

#define ZXING_MAX_WIDTH 2048
#define ZXING_MAX_HEIGHT 2048

namespace zxing {
typedef char byte;
typedef unsigned char boolean;
// typedef unsigned short ushort;
}  // namespace zxing

#include <limits>
#include <cmath>

namespace zxing {
inline bool isnan(float v) { return std::isnan(v); }
inline bool isnan(double v) { return std::isnan(v); }
inline float nan() { return std::numeric_limits<float>::quiet_NaN(); }
}  // namespace zxing

#ifndef ZXING_TIME
#define ZXING_TIME(string) (void)0
#endif
#ifndef ZXING_TIME_MARK
#define ZXING_TIME_MARK(string) (void)0
#endif

#endif  // __ZXING_ZXING_HPP__
