// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_READER_HPP
#define _OPENCV_READER_HPP

#include "opencv2/photometric_calib.hpp"

namespace cv { namespace photometric_calib{

//! @addtogroup photometric_calib
//! @{

class Reader
{
public:
    Reader(const std::string &folderPath, const std::string &timesPath);

    unsigned long getNumImages() const;

    double getTimestamp(unsigned long id) const;

    float getExposureTime(unsigned long id) const;


private:
    inline void loadTimestamps(const std::string &timesFile);

    std::vector<String> files;
    std::vector<double> timeStamps;
    std::vector<float> exposureTimes;

    int width, height;

    String path;
};

//! @}

}} // namespace cv photometric_calib
#endif //_OPENCV_READER_HPP