// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __OPENCV_WECHAT_QRCODE_DECODERMGR_HPP__
#define __OPENCV_WECHAT_QRCODE_DECODERMGR_HPP__

// zxing
#include "zxing/binarizer.hpp"
#include "zxing/binarybitmap.hpp"
#include "zxing/decodehints.hpp"
#include "zxing/qrcode/qrcode_reader.hpp"
#include "zxing/result.hpp"

// qbar
#include "binarizermgr.hpp"
#include "imgsource.hpp"
namespace cv {
namespace wechat_qrcode {

class DecoderMgr {
public:
    DecoderMgr() { reader_ = new zxing::qrcode::QRCodeReader(); };
    ~DecoderMgr(){};

    int decodeImage(cv::Mat src, bool use_nn_detector, vector<string>& result, vector<vector<Point2f>>& zxing_points);

private:
    zxing::Ref<zxing::UnicomBlock> qbarUicomBlock_;
    zxing::DecodeHints decode_hints_;

    zxing::Ref<zxing::qrcode::QRCodeReader> reader_;
    BinarizerMgr binarizer_mgr_;

    vector<zxing::Ref<zxing::Result>> Decode(zxing::Ref<zxing::BinaryBitmap> image,
                                     zxing::DecodeHints hints);

    int TryDecode(zxing::Ref<zxing::LuminanceSource> source, vector<zxing::Ref<zxing::Result>>& result);
};

}  // namespace wechat_qrcode
}  // namespace cv
#endif  // __OPENCV_WECHAT_QRCODE_DECODERMGR_HPP__
