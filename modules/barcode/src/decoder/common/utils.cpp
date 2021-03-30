// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#include "../../precomp.hpp"
#include "utils.hpp"
#include "hybrid_binarizer.hpp"

namespace cv {
namespace barcode {


void preprocess(Mat &src, Mat &dst)
{
    Mat blur;
    GaussianBlur(src, blur, Size(0, 0), 25);
    addWeighted(src, 2, blur, -1, 0, dst);
    dst.convertTo(dst, CV_8UC1, 1, -20);
}

Mat binarize(const Mat &src, int mode)
{
    Mat dst;
    switch (mode)
    {
        case OTSU:
            threshold(src, dst, 155, 255, THRESH_OTSU + THRESH_BINARY);
            break;
        case HYBRID:
            hybridBinarization(src, dst);
            break;
        default:
            break;
    }
    return dst;
}
}
}
