// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_QUALITY_QUALITY_UTILS_HPP
#define OPENCV_QUALITY_QUALITY_UTILS_HPP

#include "qualitybase.hpp"

namespace cv
{
namespace quality
{
namespace quality_utils
{

// default type of matrix to expand to
static CV_CONSTEXPR const int EXPANDED_MAT_DEFAULT_TYPE = CV_32F;

// convert inputarray to specified mat type.  set type == -1 to preserve existing type
template <typename R>
inline R extract_mat(InputArray in, const int type = -1)
{
    R result = {};
    if ( in.isMat() )
        in.getMat().convertTo( result, (type != -1) ? type : in.getMat().type());
    else if ( in.isUMat() )
        in.getUMat().convertTo( result, (type != -1) ? type : in.getUMat().type());
    else
        CV_Error(Error::StsNotImplemented, "Unsupported input type");

    return result;
}

// extract and expand matrix to target type
template <typename R>
inline R expand_mat( InputArray src, int TYPE_DEFAULT = EXPANDED_MAT_DEFAULT_TYPE)
{
    auto result = extract_mat<R>(src, -1);

    // by default, expand to 32F unless we already have >= 32 bits, then go to 64
    //  if/when we can detect OpenCL CV_16F support, opt for that when input depth == 8
    //  note that this may impact the precision of the algorithms and would need testing
    int type = TYPE_DEFAULT;

    switch (result.depth())
    {
    case CV_32F:
    case CV_32S:
    case CV_64F:
        type = CV_64F;
    };  // switch

    result.convertTo(result, type);
    return result;
}

// return mat of observed min/max pair per column
//  row 0:  min per column
//  row 1:  max per column
// template <typename T>
inline cv::Mat get_column_range( const cv::Mat& data )
{
    CV_Assert(data.channels() == 1);
    CV_Assert(data.rows > 0);

    cv::Mat result( cv::Size( data.cols, 2 ), data.type() );

    auto
        row_min = result.row(0)
        , row_max = result.row(1)
        ;

    // set initial min/max
    data.row(0).copyTo(row_min);
    data.row(0).copyTo(row_max);

    for (int y = 1; y < data.rows; ++y)
    {
        auto row = data.row(y);
        cv::min(row,row_min, row_min);
        cv::max(row, row_max, row_max);
    }
    return result;
}   // get_column_range

// linear scale of each column from min to max
//  range is column-wise pair of observed min/max.  See get_column_range
template <typename T>
inline void scale( cv::Mat& mat, const cv::Mat& range, const T min, const T max )
{
    // value = lower + (upper - lower) * (value - feature_min[index]) / (feature_max[index] - feature_min[index]);
    // where [lower] = lower bound, [upper] = upper bound

    for (int y = 0; y < mat.rows; ++y)
    {
        auto row = mat.row(y);
        auto row_min = range.row(0);
        auto row_max = range.row(1);

        for (int x = 0; x < mat.cols; ++x)
            row.at<T>(x) = min + (max - min) * (row.at<T>(x) - row_min.at<T>(x) ) / (row_max.at<T>(x) - row_min.at<T>(x));
    }
}

}   // quality_utils
}   // quality
}   // cv
#endif