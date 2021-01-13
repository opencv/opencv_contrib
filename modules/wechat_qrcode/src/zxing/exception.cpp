// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#include "zxing/exception.hpp"
#include <string.h>
#include "zxing/zxing.hpp"

using zxing::Exception;

void Exception::deleteMessage() { delete[] message; }

char const* Exception::copy(char const* msg) {
    char* message = 0;
    if (msg) {
        int l = strlen(msg) + 1;
        if (l) {
            message = new char[l];
            strcpy(message, msg);
        }
    }
    return message;
}
