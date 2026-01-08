// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DECODER_DATAMASK_HPP__
#define __ZXING_QRCODE_DECODER_DATAMASK_HPP__

#include "../../common/array.hpp"
#include "../../common/bitmatrix.hpp"
#include "../../common/counted.hpp"
#include "../../errorhandler.hpp"
namespace zxing {
namespace qrcode {

class DataMask : public Counted {
private:
    static std::vector<Ref<DataMask> > DATA_MASKS;

protected:
public:
    DataMask();
    virtual ~DataMask();
    void unmaskBitMatrix(BitMatrix& matrix, size_t dimension);
    virtual bool isMasked(size_t x, size_t y) = 0;
    static DataMask& forReference(int reference, ErrorHandler& err_handler);
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DECODER_DATAMASK_HPP__
