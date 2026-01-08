// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_RESULT_HPP__
#define __ZXING_RESULT_HPP__

#include <stdint.h>
#include "common/array.hpp"
#include "common/counted.hpp"
#include "common/str.hpp"
#include "resultpoint.hpp"

#include <string>

namespace zxing {

class Result : public Counted {
private:
    Ref<String> text_;
    ArrayRef<char> rawBytes_;
    ArrayRef<Ref<ResultPoint> > resultPoints_;
    std::string charset_;
    int qrcodeVersion_;
    int pyramidLv_;
    int binaryMethod_;
    string ecLevel_;
    string charsetMode_;
    string scale_list_;
    float decode_scale_;
    uint32_t detect_time_;
    uint32_t sr_time_;

public:
    Result(Ref<String> text, ArrayRef<char> rawBytes, ArrayRef<Ref<ResultPoint> > resultPoints);

    Result(Ref<String> text, ArrayRef<char> rawBytes, ArrayRef<Ref<ResultPoint> > resultPoints,
           std::string charset);

    Result(Ref<String> text, ArrayRef<char> rawBytes, ArrayRef<Ref<ResultPoint> > resultPoints,
           std::string charset, int QRCodeVersion, string ecLevel, string charsetMode);

    ~Result();

    Ref<String> getText();
    ArrayRef<char> getRawBytes();
    ArrayRef<Ref<ResultPoint> > const& getResultPoints() const;
    ArrayRef<Ref<ResultPoint> >& getResultPoints();
    std::string getCharset() const;
    std::string getChartsetMode() const;
    void enlargeResultPoints(int scale);

    int getQRCodeVersion() const { return qrcodeVersion_; };
    void setQRCodeVersion(int QRCodeVersion) { qrcodeVersion_ = QRCodeVersion; };
    int getPyramidLv() const { return pyramidLv_; };
    void setPyramidLv(int pyramidLv) { pyramidLv_ = pyramidLv; };
    int getBinaryMethod() const { return binaryMethod_; };
    void setBinaryMethod(int binaryMethod) { binaryMethod_ = binaryMethod; };
    string getEcLevel() const { return ecLevel_; }
    void setEcLevel(char ecLevel) { ecLevel_ = ecLevel; }
    std::string getScaleList() { return scale_list_; };
    void setScaleList(const std::string& scale_list) { scale_list_ = scale_list; };
    float getDecodeScale() { return decode_scale_; };
    void setDecodeScale(float decode_scale) { decode_scale_ = decode_scale; };
    uint32_t getDetectTime() { return detect_time_; };
    void setDetectTime(uint32_t detect_time) { detect_time_ = detect_time; };
    uint32_t getSrTime() { return sr_time_; };
    void setSrTime(uint32_t sr_time) { sr_time_ = sr_time; };
};

}  // namespace zxing
#endif  // __ZXING_RESULT_HPP__
