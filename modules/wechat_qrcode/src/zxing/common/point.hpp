// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_POINT_HPP__
#define __ZXING_COMMON_POINT_HPP__

namespace zxing {
class PointI {
public:
    int x;
    int y;
};

class Point {
public:
    Point() : x(0.0f), y(0.0f){};
    Point(float x_, float y_) : x(x_), y(y_){};

    float x;
    float y;
};

class Line {
public:
    Line(Point start_, Point end_) : start(start_), end(end_){};

    Point start;
    Point end;
};
}  // namespace zxing
#endif  // __ZXING_COMMON_POINT_HPP__
