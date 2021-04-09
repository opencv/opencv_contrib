// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_READER_HPP__
#define __ZXING_READER_HPP__

#include "binarybitmap.hpp"
#include "decodehints.hpp"
#include "errorhandler.hpp"
#include "result.hpp"

namespace zxing {

class Reader : public Counted {
protected:
    Reader() {}

public:
    virtual Ref<Result> decode(Ref<BinaryBitmap> image);
    virtual Ref<Result> decode(Ref<BinaryBitmap> image, DecodeHints hints) = 0;

    virtual ~Reader();
    virtual string name();
    virtual unsigned int getDecodeID();
    virtual void setDecodeID(unsigned int id);

    virtual float getPossibleFix();
};

}  // namespace zxing

#endif  // __ZXING_READER_HPP__
