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
        Point2f _center;

        if (_srcMat.type() == CV_32FC1 || _srcMat.type() == CV_64FC1) {
            _out_mat_type = CV_64FC1;
        }
        else {
            _out_mat_type = CV_32SC1;
        }

        if (crop) {
            // crop the source into square
            _row_num = min(_srcMat.rows, _srcMat.cols);
            Point2f _srcCenter = Point2f(_srcMat.cols / 2.f - 0.5f, _srcMat.rows / 2.f - 0.5f);
            Mat _squared_src = Mat(Size(_row_num, _row_num), _srcMat.type(), Scalar(0));
            _center = Point2f(_squared_src.cols / 2.f - 0.5f, _squared_src.rows / 2.f - 0.5f);
            Point2f _offset = _center - _srcCenter;
            Mat _t_matrix = (Mat1f(2, 3) << 1, 0, _offset.x, 0, 1, _offset.y);
            warpAffine(_srcMat, _squared_src, _t_matrix, _squared_src.size());
            // crop the source into circle
            Mat _mask(_squared_src.size(), CV_8UC1, Scalar(0));
            circle(_mask, _center * 2, _srcMat.cols, Scalar(255), FILLED, LINE_8, 1);
            _squared_src.copyTo(_masked_src, _mask);
        }
        else {
            // avoid cropping corner when rotating
            _row_num = cvCeil(sqrt(_srcMat.rows * _srcMat.rows + _srcMat.cols * _srcMat.cols));
            _masked_src = Mat(Size(_row_num, _row_num), _srcMat.type(), Scalar(0));
            _center = Point2f(_masked_src.cols / 2.f - 0.5f, _masked_src.rows / 2.f - 0.5f);
            Point2f _srcCenter = Point2f(_srcMat.cols / 2.f - 0.5f, _srcMat.rows / 2.f - 0.5f);
            Point2f _offset = _center - _srcCenter;
            Mat _t_matrix = (Mat1f(2, 3) << 1, 0, _offset.x, 0, 1, _offset.y);
            warpAffine(_srcMat, _masked_src, _t_matrix, _masked_src.size());
        }

        double _t;
        Mat _rotated_src;
        Mat _radon(_row_num, _col_num, _out_mat_type);

        for (int _col = 0; _col < _col_num; _col++) {
            // rotate the source by _t
            _t = (start_angle + _col * theta);
            Mat _r_matrix = getRotationMatrix2D(_center, _t, 1);
            warpAffine(_masked_src, _rotated_src, _r_matrix, _masked_src.size());
            Mat _col_mat = _radon.col(_col);
            // make projection
            reduce(_rotated_src, _col_mat, 1, REDUCE_SUM, _out_mat_type);
        }

        if (norm) {
            normalize(_radon, _radon, 0, 255, NORM_MINMAX, CV_8UC1);
        }

        _radon.copyTo(dst);
        return;
    }
} }
