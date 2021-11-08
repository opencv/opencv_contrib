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
        CV_Assert(src.type() == CV_8U);
        CV_Assert(src.dims() == 2);
        CV_Assert(src.channels() == 1);
        CV_Assert((end_angle - start_angle) * theta > 0);

        int _row_num = cvRound((end_angle - start_angle) / theta);
        Mat _srcMat = src.getMat();
        Mat _masked_src;
        cv::Point _center(_srcMat.cols / 2, _srcMat.rows / 2);

        if (crop) {
            CV_Assert(src.rows() == src.cols());
            Mat _mask(_srcMat.size(), CV_8U, Scalar(0));
            circle(_mask, _center, _srcMat.cols / 2, Scalar(255), FILLED);
            _srcMat.copyTo(_masked_src, _mask);
        }
        else {
            _masked_src = _srcMat;
        }

        double _t;
        Mat _rotated_src;
        Mat _hough(_row_num, _masked_src.cols, CV_32SC1);

        for (int _row = 0; _row < _row_num; _row++) {
            _t = start_angle + _row * theta;
            cv::Mat r = cv::getRotationMatrix2D(_center, _t, 1);
            cv::warpAffine(_masked_src, _rotated_src, r, _masked_src.size());
            Mat _row_mat = _hough.row(_row);
            cv::reduce(_rotated_src, _row_mat, 0, REDUCE_SUM, CV_32SC1);
        }

        if (norm) {
            normalize(_hough, _hough, 0, 255, NORM_MINMAX, CV_8U);
        }

        _hough.copyTo(dst);
        return;
    }
} }
