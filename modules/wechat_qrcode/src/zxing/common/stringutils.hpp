// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_STRINGUTILS_HPP__
#define __ZXING_COMMON_STRINGUTILS_HPP__

#include "../decodehints.hpp"
#include "../zxing.hpp"

#include <map>
#include <string>

namespace zxing {
namespace common {
class StringUtils;
}
}  // namespace zxing
using namespace std;

class zxing::common::StringUtils {
private:
    static char const* const PLATFORM_DEFAULT_ENCODING;

public:
    static char const* const ASCII;
    static char const* const SHIFT_JIS;
    static char const* const GB2312;
    static char const* const EUC_JP;
    static char const* const UTF8;
    static char const* const ISO88591;
    static char const* const GBK;
    static char const* const GB18030;
    static char const* const BIG5;

    static const bool ASSUME_SHIFT_JIS;

    static std::string guessEncoding(char* bytes, int length);
    static std::string guessEncodingZXing(char* bytes, int length);

#ifdef USE_UCHARDET
    static std::string guessEncodingUCharDet(char* bytes, int length);
#endif

    static int is_utf8_special_byte(unsigned char c);
    // static int is_utf8_code(const string& str);
    static int is_utf8_code(char* str, int length);
    static int is_gb2312_code(char* str, int length);
    static int is_big5_code(char* str, int length);
    static int is_gbk_code(char* str, int length);
    static int is_ascii_code(char* str, int length);
    static int shift_jis_to_jis(const unsigned char* may_be_shift_jis, int* jis_first_ptr,
                                int* jis_second_ptr);

    static std::string convertString(const char* rawData, int length, const char* fromCharset,
                                     const char* toCharset);
};

#endif  // __ZXING_COMMON_STRINGUTILS_HPP__
