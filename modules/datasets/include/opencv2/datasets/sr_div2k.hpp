// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DATASETS_SR_DIV2K_HPP
#define OPENCV_DATASETS_SR_DIV2K_HPP

#include <string>
#include <vector>

#include "opencv2/datasets/dataset.hpp"

#include <opencv2/core.hpp>

namespace cv
{
namespace datasets
{

//! @addtogroup datasets_sr
//! @{

struct SR_div2kObj : public Object
{
    std::string imageName;
};

class CV_EXPORTS SR_div2k : public Dataset
{
public:
    virtual void load(const std::string &path) CV_OVERRIDE = 0;

    static Ptr<SR_div2k> create();
};

//! @}

}
}

#endif