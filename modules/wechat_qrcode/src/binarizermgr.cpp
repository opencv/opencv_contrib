// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#include "precomp.hpp"
#include "binarizermgr.hpp"
#include "imgsource.hpp"


using zxing::Binarizer;
using zxing::LuminanceSource;
namespace cv {
namespace wechat_qrcode {
BinarizerMgr::BinarizerMgr() : m_iNowRotateIndex(0), m_iNextOnceBinarizer(-1) {
    m_vecRotateBinarizer.push_back(Hybrid);
    m_vecRotateBinarizer.push_back(FastWindow);
    m_vecRotateBinarizer.push_back(SimpleAdaptive);
    m_vecRotateBinarizer.push_back(AdaptiveThreshold);
}

BinarizerMgr::~BinarizerMgr() {}

zxing::Ref<Binarizer> BinarizerMgr::Binarize(zxing::Ref<LuminanceSource> source) {
    BINARIZER binarizerIdx = m_vecRotateBinarizer[m_iNowRotateIndex];
    if (m_iNextOnceBinarizer >= 0) {
        binarizerIdx = (BINARIZER)m_iNextOnceBinarizer;
    }

    zxing::Ref<Binarizer> binarizer;
    switch (binarizerIdx) {
        case Hybrid:
            binarizer = new zxing::HybridBinarizer(source);
            break;
        case FastWindow:
            binarizer = new zxing::FastWindowBinarizer(source);
            break;
        case SimpleAdaptive:
            binarizer = new zxing::SimpleAdaptiveBinarizer(source);
            break;
        case AdaptiveThreshold:
            binarizer = new zxing::AdaptiveThresholdMeanBinarizer(source);
            break;
        default:
            binarizer = new zxing::HybridBinarizer(source);
            break;
    }

    return binarizer;
}

void BinarizerMgr::SwitchBinarizer() {
    m_iNowRotateIndex = (m_iNowRotateIndex + 1) % m_vecRotateBinarizer.size();
}

int BinarizerMgr::GetCurBinarizer() {
    if (m_iNextOnceBinarizer != -1) return m_iNextOnceBinarizer;
    return m_vecRotateBinarizer[m_iNowRotateIndex];
}

void BinarizerMgr::SetNextOnceBinarizer(int iBinarizerIndex) {
    m_iNextOnceBinarizer = iBinarizerIndex;
}

void BinarizerMgr::SetBinarizer(vector<BINARIZER> vecRotateBinarizer) {
    m_vecRotateBinarizer = vecRotateBinarizer;
}
}  // namespace wechat_qrcode
}  // namespace cv