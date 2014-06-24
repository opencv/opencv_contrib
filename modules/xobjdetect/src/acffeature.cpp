#include <opencv2/xobjdetect.hpp>

using std::vector;

#include <algorithm>
using std::min;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

namespace cv
{
namespace adas
{

ACFFeatureEvaluator::ACFFeatureEvaluator(const vector<Point3i>& features):
    features_(features), channels_(), position_()
{
}
int ACFFeatureEvaluator::evaluate(size_t feature_ind) const
{
    /* Assume there are 10 channels */
    CV_Assert(channels_.size() == 10);
    CV_Assert(feature_ind < features_.size());

    Point3i feature = features_.at(feature_ind);
    int x = feature.x;
    int y = feature.y;
    int n = feature.z;
    return channels_[n].at<int>(y, x);
}

void ACFFeatureEvaluator::setChannels(cv::InputArrayOfArrays channels)
{
    channels_.clear();
    vector<Mat> ch;
    channels.getMatVector(ch);
    for( size_t i = 0; i < ch.size(); ++i )
    {
        const Mat &channel = ch[i];
        Mat_<int> acf_channel(channel.rows / 4, channel.cols / 4);
        for( int row = 0; row < channel.rows; row += 4 )
        {
            for( int col = 0; col < channel.cols; col += 4 )
            {
                int sum = 0;
                for( int cell_row = row; cell_row < row + 4; ++cell_row )
                    for( int cell_col = col; cell_col < col + 4; ++cell_col )
                        sum += channel.at<float>(cell_row, cell_col);

                acf_channel(row / 4, col / 4) = sum;
            }
        }
        channels_.push_back(acf_channel);
    }
}

void ACFFeatureEvaluator::setPosition(Size position)
{
    position_ = position;
}

void ACFFeatureEvaluator::evaluateAll(OutputArray feature_values) const
{
    Mat_<int> feature_vals(1, features_.size());
    for( size_t i = 0; i < features_.size(); ++i )
    {
        feature_vals(0, i) = evaluate(i);
    }
    feature_values.setTo(feature_vals);
}

vector<Point3i> generateFeatures(Size window_size, int count)
{
    CV_Assert(count > 0);
    int cur_count = 0;
    int max_count = window_size.width * window_size.height / 16;
    count = min(count, max_count);
    vector<Point3i> features;
    for( int x = 0; x < window_size.width / 4; ++x )
    {
        for( int y = 0; y < window_size.height / 4; ++y )
        {
            /* Assume there are 10 channel types */
            for( int n = 0; n < 10; ++n )
            {
                features.push_back(Point3i(x, y, n));
                if( (cur_count += 1) == count )
                    break;
            }
        }
    }
    return features;
}

void computeChannels(cv::InputArray image, cv::OutputArrayOfArrays channels_)
{
    Mat src(image.getMat().rows, image.getMat().cols, CV_32FC3);
    image.getMat().convertTo(src, CV_32FC3, 1./255);

    Mat_<float> grad;
    Mat gray;
    cvtColor(src, gray, CV_RGB2GRAY);

    Mat_<float> row_der, col_der;
    Sobel(gray, row_der, CV_32F, 0, 1);
    Sobel(gray, col_der, CV_32F, 1, 0);

    magnitude(row_der, col_der, grad);

    Mat_<Vec6f> hist(grad.rows, grad.cols);
    const float to_deg = 180 / 3.1415926;
    for (int row = 0; row < grad.rows; ++row) {
        for (int col = 0; col < grad.cols; ++col) {
            float angle = atan2(row_der(row, col), col_der(row, col)) * to_deg;
            if (angle < 0)
                angle += 180;
            int ind = angle / 30;
            hist(row, col)[ind] = grad(row, col);
        }
    }

    vector<Mat> channels;
    channels.push_back(gray);
    channels.push_back(grad);

    vector<Mat> hist_channels;
    split(hist, hist_channels);

    for( size_t i = 0; i < hist_channels.size(); ++i )
        channels.push_back(hist_channels[i]);

    channels_.setTo(channels);
}

} /* namespace adas */
} /* namespace cv */
