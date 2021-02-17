// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../../precomp.hpp"
#include "grid_sampler.hpp"
#include "perspective_transform.hpp"
#include <sstream>

namespace zxing {

GridSampler GridSampler::gridSampler;

GridSampler::GridSampler() {}

// Samples an image for a rectangular matrix of bits of the given dimension.
Ref<BitMatrix> GridSampler::sampleGrid(Ref<BitMatrix> image, int dimension,
                                       Ref<PerspectiveTransform> transform,
                                       ErrorHandler &err_handler) {
    Ref<BitMatrix> bits(new BitMatrix(dimension, err_handler));
    if (err_handler.ErrCode()) return Ref<BitMatrix>();

    vector<float> points(dimension << 1, 0.0f);

    int outlier = 0;
    int maxOutlier = dimension * dimension * 3 / 10 - 1;

    for (int y = 0; y < dimension; y++) {
        int max = points.size();
        float yValue = (float)y + 0.5f;
        for (int x = 0; x < max; x += 2) {
            points[x] = (float)(x >> 1) + 0.5f;
            points[x + 1] = yValue;
        }
        transform->transformPoints(points);
        // Quick check to see if points transformed to something inside the
        // image; sufficient to check the endpoings
        outlier += checkAndNudgePoints(image->getWidth(), image->getHeight(), points, err_handler);
        if (err_handler.ErrCode()) return Ref<BitMatrix>();

        if (outlier >= maxOutlier) {
            ostringstream s;
            s << "Over 30% points out of bounds.";
            err_handler = ReaderErrorHandler(s.str().c_str());
            return Ref<BitMatrix>();
        }

        for (int x = 0; x < max; x += 2) {
            if (image->get((int)points[x], (int)points[x + 1])) {
                // Black (-ish) pixel
                bits->set(x >> 1, y);
            }
        }
    }
    return bits;
}

int GridSampler::checkAndNudgePoints(int width, int height, vector<float> &points,
                                     ErrorHandler &err_handler) {
    // Modified to support stlport
    float *pts = NULL;

    if (points.size() > 0) {
        pts = &points[0];
    } else {
        err_handler = ReaderErrorHandler("checkAndNudgePoints:: no points!");
        return -1;
    }

    int size = (int)points.size() / 2;

    // The Java code assumes that if the start and end points are in bounds, the
    // rest will also be. However, in some unusual cases points in the middle
    // may also be out of bounds. Since we can't rely on an
    // ArrayIndexOutOfBoundsException like Java, we check every point.

    int outCount = 0;
    // int maxError = (int)(size/2/3 - 1);

    float maxborder = width / size * 3;

    for (size_t offset = 0; offset < points.size(); offset += 2) {
        int x = (int)pts[offset];
        int y = (int)pts[offset + 1];
        // if((int)offset==0)
        //	cout<<"checkAndNudgePoints "<<(int)offset<<": ("<<x<<",
        //"<<y<<")"<<endl;

        if (x < -1 || x > width || y < -1 || y > height) {
            outCount++;
            if (x > width + maxborder || y > height + maxborder || x < -maxborder ||
                y < -maxborder) {
                err_handler = ReaderErrorHandler("checkAndNudgePoints::Out of bounds!");
                return -1;
            }
        }

        if (x <= -1) {
            points[offset] = 0.0f;
        } else if (x >= width) {
            points[offset] = float(width - 1);
        }
        if (y <= -1) {
            points[offset + 1] = 0.0f;
        } else if (y >= height) {
            points[offset + 1] = float(height - 1);
        }
    }

    return outCount;
}

GridSampler &GridSampler::getInstance() { return gridSampler; }
}  // namespace zxing
