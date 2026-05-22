// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv {namespace ximgproc {
void RadonTransform(InputArray src,
                    OutputArray dst,
                    double theta,
                    double start_angle,
                    double end_angle,
                    bool crop,
                    bool norm)
{
    CV_Assert(src.dims() == 2);
    CV_Assert(src.channels() == 1);
    CV_Assert((end_angle - start_angle) * theta > 0);

    int col_num = cvRound((end_angle - start_angle) / theta);
    int row_num, out_mat_type;
    Point center;
    Mat srcMat, masked_src;

    transpose(src, srcMat);

    if (srcMat.type() == CV_32FC1 || srcMat.type() == CV_64FC1) {
        out_mat_type = CV_64FC1;
    }
    else {
        out_mat_type = CV_32SC1;
    }

    if (crop) {
        // Crop the source into square
        row_num = min(srcMat.rows, srcMat.cols);
        Rect crop_ROI(
            srcMat.cols / 2 - row_num / 2,
            srcMat.rows / 2 - row_num / 2,
            row_num, row_num);
        srcMat = srcMat(crop_ROI);

        // Crop the source into circle
        Mat mask(srcMat.size(), CV_8UC1, Scalar(0));
        center = Point(srcMat.cols / 2, srcMat.rows / 2);
        circle(mask, center, srcMat.cols / 2, Scalar(255), FILLED);
        srcMat.copyTo(masked_src, mask);
    }
    else {
        // Avoid cropping corner when rotating
        row_num = cvCeil(sqrt(srcMat.rows * srcMat.rows + srcMat.cols * srcMat.cols));
        masked_src = Mat(Size(row_num, row_num), srcMat.type(), Scalar(0));
        center = Point(masked_src.cols / 2, masked_src.rows / 2);
        srcMat.copyTo(masked_src(Rect(
            (row_num - srcMat.cols) / 2,
            (row_num - srcMat.rows) / 2,
            srcMat.cols, srcMat.rows)));
    }

    dst.create(row_num, col_num, out_mat_type);
    Mat radon = dst.getMat();

    // Define the parallel loop as a lambda function
    parallel_for_(Range(0, col_num), [&](const Range& range) {
        for (int col = range.start; col < range.end; col++) {
            // Rotate the source by t
            double t = (start_angle + col * theta);
            Mat r_matrix = getRotationMatrix2D(center, t, 1);

            Mat rotated_src;
            warpAffine(masked_src, rotated_src, r_matrix, masked_src.size());

            Mat col_mat = radon.col(col);
            // Make projection
            reduce(rotated_src, col_mat, 1, REDUCE_SUM, out_mat_type);
        }
    });

    if (norm) {
        normalize(radon, dst.getMatRef(), 0, 255, NORM_MINMAX, CV_8UC1);
    }
}
} }
