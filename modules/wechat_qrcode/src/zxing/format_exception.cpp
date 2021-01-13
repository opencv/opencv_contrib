// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#include "zxing/format_exception.hpp"

namespace zxing {

FormatException::FormatException() {}

FormatException::FormatException(const char* msg) : ReaderException(msg) {}

FormatException::~FormatException() throw() {}

FormatException const& FormatException::getFormatInstance() {
    static FormatException instance;
    return instance;
}

}  // namespace zxing
