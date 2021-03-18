// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_QRCODE_DECODER_QRCODEDECODERMETADATA_HPP__
#define __ZXING_QRCODE_DECODER_QRCODEDECODERMETADATA_HPP__

#include "../../common/array.hpp"
#include "../../common/counted.hpp"
#include "../../resultpoint.hpp"

// VC++
// The main class which implements QR Code decoding -- as opposed to locating
// and extracting the QR Code from an image.

namespace zxing {
namespace qrcode {

/**
 * Meta-data container for QR Code decoding. Instances of this class may be used
 * to convey information back to the decoding caller. Callers are expected to
 * process this.
 *
 * @see com.google.zxing.common.DecoderResult#getOther()
 */
class QRCodeDecoderMetaData : public Counted {
private:
    bool mirrored_;

public:
    explicit QRCodeDecoderMetaData(bool mirrored) : mirrored_(mirrored) {}

public:
    /**
     * @return true if the QR Code was mirrored.
     */
    bool isMirrored() { return mirrored_; };

    /**
     * Apply the result points' order correction due to mirroring.
     *
     * @param points Array of points to apply mirror correction to.
     */
    void applyMirroredCorrection(ArrayRef<Ref<ResultPoint> >& points) {
        if (!mirrored_ || points->size() < 3) {
            return;
        }
        Ref<ResultPoint> bottomLeft = points[0];
        points[0] = points[2];
        points[2] = bottomLeft;
        // No need to 'fix' top-left and alignment pattern.
    };
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DECODER_QRCODEDECODERMETADATA_HPP__
