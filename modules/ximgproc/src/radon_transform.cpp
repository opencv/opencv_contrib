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

        Mat _srcMat = src.getMat();

        int _row_num, _col_num, _out_mat_type;
        _col_num = cvRound((end_angle - start_angle) / theta);
        transpose(_srcMat, _srcMat);
        Mat _masked_src;
        cv::Point _center;

        if (_srcMat.type() == CV_32FC1 || _srcMat.type() == CV_64FC1) {
            _out_mat_type = CV_64FC1;
        }
        else {
            _out_mat_type = CV_32SC1;
        }

        if (crop) {
            // crop the source into square
            _row_num = min(_srcMat.rows, _srcMat.cols);
            cv::Rect _crop_ROI(
                _srcMat.cols / 2 - _row_num / 2,
                _srcMat.rows / 2 - _row_num / 2,
                _row_num, _row_num);
            _srcMat = _srcMat(_crop_ROI);
            // crop the source into circle
            Mat _mask(_srcMat.size(), CV_8UC1, Scalar(0));
            _center = Point(_srcMat.cols / 2, _srcMat.rows / 2);
            circle(_mask, _center, _srcMat.cols / 2, Scalar(255), FILLED);
            _srcMat.copyTo(_masked_src, _mask);
        }
        else {
            // avoid cropping corner when rotating
            _row_num = cvCeil(sqrt(_srcMat.rows * _srcMat.rows + _srcMat.cols * _srcMat.cols));
            _masked_src = Mat(Size(_row_num, _row_num), _srcMat.type(), Scalar(0));
            _center = Point(_masked_src.cols / 2, _masked_src.rows / 2);
            _srcMat.copyTo(_masked_src(Rect(
                (_row_num - _srcMat.cols) / 2,
                (_row_num - _srcMat.rows) / 2,
                _srcMat.cols, _srcMat.rows)));
        }

        double _t;
        Mat _rotated_src;
        Mat _radon(_row_num, _col_num, _out_mat_type);

        for (int _col = 0; _col < _col_num; _col++) {
            // rotate the source by _t
            _t = (start_angle + _col * theta);
            cv::Mat _r_matrix = cv::getRotationMatrix2D(_center, _t, 1);
            cv::warpAffine(_masked_src, _rotated_src, _r_matrix, _masked_src.size());
            Mat _col_mat = _radon.col(_col);
            // make projection
            cv::reduce(_rotated_src, _col_mat, 1, REDUCE_SUM, _out_mat_type);
        }

        if (norm) {
            normalize(_radon, _radon, 0, 255, NORM_MINMAX, CV_8UC1);
        }

        _radon.copyTo(dst);
        return;
    }
} }
