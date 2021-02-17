// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_MATHUTILS_HPP__
#define __ZXING_COMMON_MATHUTILS_HPP__

#include <cmath>
#if (defined __GNUC__ && defined __x86_64__ && defined __SSE2__ && !defined __APPLE__ && \
     !defined __GXX_WEAK__)
#include <ammintrin.h>
#elif defined _MSC_VER && (defined _M_X64 || defined _M_IX86)
#include <emmintrin.h>
#endif

#include <algorithm>
#include <numeric>
#include <vector>

namespace zxing {
namespace common {

class MathUtils {
private:
    MathUtils();
    ~MathUtils();

public:
    static inline float distance(float aX, float aY, float bX, float bY) {
        float xDiff = aX - bX;
        float yDiff = aY - bY;
        return sqrt(float(xDiff * xDiff + yDiff * yDiff));
    }

    static inline float distance_4_int(int aX, int aY, int bX, int bY) {
        return sqrt(float((aX - bX) * (aX - bX) + (aY - bY) * (aY - bY)));
    }

    static inline void getRangeValues(int& minValue, int& maxValue, int min, int max) {
        int finalMinValue, finalMaxValue;

        if (minValue < maxValue) {
            finalMinValue = minValue;
            finalMaxValue = maxValue;
        } else {
            finalMinValue = maxValue;
            finalMaxValue = minValue;
        }

        finalMinValue = finalMinValue > min ? finalMinValue : min;
        finalMaxValue = finalMaxValue < max ? finalMaxValue : max;

        minValue = finalMinValue;
        maxValue = finalMaxValue;
    }

    static inline bool isInRange(float x, float y, float width, float height) {
        if ((x >= 0.0 && x <= (width - 1.0)) && (y >= 0.0 && y <= (height - 1.0))) {
            return true;
        } else {
            return false;
        }
    }

    static inline float distance(int aX, int aY, int bX, int bY) {
        int xDiff = aX - bX;
        int yDiff = aY - bY;
        return sqrt(float(xDiff * xDiff + yDiff * yDiff));
    }

    static inline float VecCross(float* v1, float* v2) { return v1[0] * v2[1] - v1[1] * v2[0]; }

    static inline void Stddev(std::vector<float>& resultSet, float& avg, float& stddev) {
        double sum = std::accumulate(resultSet.begin(), resultSet.end(), 0.0);
        avg = sum / resultSet.size();

        double accum = 0.0;
        for (std::size_t i = 0; i < resultSet.size(); i++) {
            accum += (resultSet[i] - avg) * (resultSet[i] - avg);
        }

        stddev = sqrt(accum / (resultSet.size()));
    }
};

}  // namespace common
}  // namespace zxing

#endif  // __ZXING_COMMON_MATHUTILS_HPP__
