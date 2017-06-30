// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_READER_HPP
#define _OPENCV_READER_HPP

#include "opencv2/photometric_calib.hpp"

#include <vector>
#include <string>

namespace cv { namespace photometric_calib{

//! @addtogroup photometric_calib
//! @{

class Reader
{
public:
    Reader(const std::string &folderPath, const std::string &timesPath);

    unsigned long getNumImages() const;

    double getTimestamp(unsigned long id) const;

    float getExposureDuration(unsigned long id) const;

    int getWidth() const;
    int getHeight() const;
    const std::string &getFolderPath() const;

private:
    inline void loadTimestamps(const std::string &timesFile);

    std::vector<String> images; //All the names/paths of images
    std::vector<double> timeStamps; //All the Unix Time Stamps of images
    std::vector<float> exposureDurations;//All the exposure duration for images

    int _width, _height;//The image width and height. All the images should be of the same size.

    std::string _folderPath;
};

//! @}

}} // namespace cv photometric_calib
#endif //_OPENCV_READER_HPP