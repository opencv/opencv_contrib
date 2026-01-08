// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_RESULTPOINT_HPP__
#define __ZXING_RESULTPOINT_HPP__

#include "common/counted.hpp"

namespace zxing {

class ResultPoint : public Counted {
protected:
    float posX_;
    float posY_;

public:
    ResultPoint();
    ResultPoint(float x, float y);
    ResultPoint(int x, int y);
    virtual ~ResultPoint();

    virtual float getX() const;
    virtual float getY() const;
    virtual void SetX(float fX);
    virtual void SetY(float fY);

    bool equals(Ref<ResultPoint> other);

    static void orderBestPatterns(std::vector<Ref<ResultPoint> > &patterns);
    static float distance(Ref<ResultPoint> point1, Ref<ResultPoint> point2);
    static float distance(float x1, float x2, float y1, float y2);

private:
    static float crossProductZ(Ref<ResultPoint> pointA, Ref<ResultPoint> pointB,
                               Ref<ResultPoint> pointC);
};

}  // namespace zxing

#endif  // __ZXING_RESULTPOINT_HPP__
