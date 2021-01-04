// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds
#ifndef __OPENCV_BARCODE_BARDECODE_HPP__
#define __OPENCV_BARCODE_BARDECODE_HPP__

#include "abs_decoder.hpp"

namespace cv {
namespace barcode {
using std::vector;
using std::string;

class BarDecode
{
public:
    void init(const Mat &src, const vector<Point2f> &points);

    const vector<Result> &getDecodeInformation()
    { return result_info; }

    bool decodingProcess();

    bool decodeMultiplyProcess();

private:
    vector<vector<Point2f>> src_points;
    Mat original;
    vector<Result> result_info;
};
}
}
#endif //! __OPENCV_BARCODE_BARDECODE_HPP__