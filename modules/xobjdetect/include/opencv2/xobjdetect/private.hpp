#ifndef __OPENCV_XOBJDETECT_PRIVATE_HPP__
#define __OPENCV_XOBJDETECT_PRIVATE_HPP__

#ifndef __OPENCV_BUILD
#  error this is a private header, do not include it outside OpenCV
#endif

#include <opencv2/core.hpp>

namespace cv
{
namespace xobjdetect
{

class CV_EXPORTS Stump
{
public:

    /* Initialize zero stump */
    Stump(): threshold_(0), polarity_(1), pos_value_(1), neg_value_(-1) {}

    /* Initialize stump with given threshold, polarity
        and classification values */
    Stump(int threshold, int polarity, float pos_value, float neg_value):
        threshold_(threshold), polarity_(polarity),
        pos_value_(pos_value), neg_value_(neg_value) {}

    /* Train stump for given data

        data — matrix of feature values, size M x N, one feature per row

        labels — matrix of sample class labels, size 1 x N. Labels can be from
            {-1, +1}

        weights — matrix of sample weights, size 1 x N

    Returns chosen feature index. Feature enumeration starts from 0
    */
    int train(const Mat& data, const Mat& labels, const Mat& weights);

    /* Predict object class given

        value — feature value. Feature must be the same as was chosen
        during training stump

    Returns real value, sign(value) means class
    */
    float predict(int value) const;

    /* Write stump in FileStorage */
    void write(FileStorage& fs) const
    {
        fs << "{"
            << "threshold" << threshold_
            << "polarity" << polarity_
            << "pos_value" << pos_value_
            << "neg_value" << neg_value_
            << "}";
    }

    /* Read stump */
    void read(const FileNode& node)
    {
        threshold_ = (int)node["threshold"];
        polarity_ = (int)node["polarity"];
        pos_value_ = (float)node["pos_value"];
        neg_value_ = (float)node["neg_value"];
    }

private:
    /* Stump decision threshold */
    int threshold_;
    /* Stump polarity, can be from {-1, +1} */
    int polarity_;
    /* Classification values for positive and negative classes  */
    float pos_value_, neg_value_;
};

void read(const FileNode& node, Stump& s, const Stump& default_value=Stump());

void write(FileStorage& fs, String&, const Stump& s);

} /* namespace xobjdetect */
} /* namespace cv */

#endif // __OPENCV_XOBJDETECT_PRIVATE_HPP__
