// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

/*
 *  Copyright 2013 ZXing authors All rights reserved.
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
#ifndef __ZXING_ZXING_HPP__
#define __ZXING_ZXING_HPP__

//>>>>>>>> type define

#define COUNTER_TYPE short

//<<<<<<<

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

#if defined(__ANDROID_API__)

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

#if defined(_WIN32) || defined(_WIN64)

#include <float.h>

namespace zxing {
inline bool isnan(float v) { return _isnan(v) != 0; }
inline bool isnan(double v) { return _isnan(v) != 0; }
inline float nan() { return std::numeric_limits<float>::quiet_NaN(); }
}  // namespace zxing

#else

#include <cmath>

#undef isnan
namespace zxing {
inline bool isnan(float v) { return std::isnan(v); }
inline bool isnan(double v) { return std::isnan(v); }
inline float nan() { return std::numeric_limits<float>::quiet_NaN(); }
}  // namespace zxing

//#endif

#endif

#ifndef ZXING_TIME
#define ZXING_TIME(string) (void)0
#endif
#ifndef ZXING_TIME_MARK
#define ZXING_TIME_MARK(string) (void)0
#endif

#endif  // __ZXING_ZXING_HPP__
