// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __ZXING_COMMON_IMAGECUT_HPP__
#define __ZXING_COMMON_IMAGECUT_HPP__
#include "bytematrix.hpp"
#include "counted.hpp"

namespace zxing {

typedef struct _ImageCutResult {
    ArrayRef<uint8_t> arrImage;
    int iWidth;
    int iHeight;
} ImageCutResult;

class ImageCut {
public:
    ImageCut();
    ~ImageCut();

    static int Cut(uint8_t* poImageData, int iWidth, int iHeight, int iTopLeftX, int iTopLeftY,
                   int iBottomRightX, int iBottomRightY, ImageCutResult& result);
    static int Cut(Ref<ByteMatrix> matrix, float fRatio, ImageCutResult& result);
};

}  // namespace zxing
#endif  // __ZXING_COMMON_IMAGECUT_HPP__
