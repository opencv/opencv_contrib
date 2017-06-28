// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DATASETS_SALIENCY_MIT1003_HPP
#define OPENCV_DATASETS_SALIENCY_MIT1003_HPP

#include <string>
#include <vector>

#include "opencv2/datasets/dataset.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace cv
{
namespace datasets
{

//! @addtogroup datasets_saliency
//! @{

struct SALIENCY_mit1003obj : public Object
{
    int id;
    std::string name;
    Mat img;
    Mat fixMap;
    Mat fixPts;
};

class CV_EXPORTS SALIENCY_mit1003 : public Dataset
{
public:
    virtual void load(const std::string &path) = 0;
    virtual std::vector<std::vector<Mat> > getDataset() = 0;
    static Ptr<SALIENCY_mit1003> create();
};

//! @}

}
}

#endif
