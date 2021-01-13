// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_ILLEGALARGUMENTEXCEPTION_HPP__
#define __ZXING_COMMON_ILLEGALARGUMENTEXCEPTION_HPP__

#include "zxing/exception.hpp"

using namespace std;

namespace zxing {

class IllegalArgumentException : public Exception {
public:
    IllegalArgumentException();
    explicit IllegalArgumentException(const char *msg);
    explicit IllegalArgumentException(string msg);
    ~IllegalArgumentException() throw();
};

}  // namespace zxing

#endif  // __ZXING_COMMON_ILLEGALARGUMENTEXCEPTION_HPP__
