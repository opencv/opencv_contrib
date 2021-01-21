// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __ZXING_COMMON_CHARACTER_HPP___
#define __ZXING_COMMON_CHARACTER_HPP___

#include <ctype.h>
#include <stdio.h>

#include <iostream>

using namespace std;

namespace zxing {

class Character {
public:
    static char toUpperCase(char c) { return toupper(c); };

    static bool isDigit(char c) {
        if (c < '0' || c > '9') {
            return false;
        }

        return true;

        // return isdigit(c);
    };

    static int digit(char c, int radix) {
        // return digit(c, radix);

        if (c >= '0' && c <= '9') {
            return (int)(c - '0');
        }

        if (c >= 'a' && c <= 'z' && c < (radix + 'a' - 10)) {
            return (int)(c - 'a' + 10);
        }

        if (c >= 'A' && c <= 'Z' && c < (radix + 'A' - 10)) {
            return (int)(c - 'A' + 10);
        }

        return -1;
    }
};
}  // namespace zxing

#endif  // __ZXING_COMMON_CHARACTER_HPP___
