// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#include "zxing/common/reedsolomon/reed_solomon_exception.hpp"

namespace zxing {
ReedSolomonException::ReedSolomonException(const char *msg) throw() : Exception(msg) {}
ReedSolomonException::~ReedSolomonException() throw() {}

}  // namespace zxing
