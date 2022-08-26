// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../precomp.hpp"
#include "reader.hpp"

namespace zxing {

Reader::~Reader() {}

vector<Ref<Result>> Reader::decode(Ref<BinaryBitmap> image) { return decode(image, DecodeHints()); }

unsigned int Reader::getDecodeID() { return 0; }

void Reader::setDecodeID(unsigned int) {}

float Reader::getPossibleFix() { return 0.0; }


string Reader::name() { return "unknow"; }

}  // namespace zxing
