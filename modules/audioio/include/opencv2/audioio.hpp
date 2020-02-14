// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#ifndef OPENCV_AUDIOIO_HPP
#define OPENCV_AUDIOIO_HPP

#include <opencv2/core.hpp>

namespace cv {

class IAudioReader {
public:
    IAudioReader(const String& filename);
    virtual bool open();
    virtual Mat read();
    virtual ~IAudioReader();
};

class CV_EXPORTS_W AudioCapture
{
public:
    AudioCapture();
    CV_WRAP explicit AudioCapture(const std::string& filename);
    CV_WRAP virtual bool open(const std::string& filename);
    virtual ~AudioCapture();
protected:
    IAudioReader ireader;
};

}
#endif //OPENCV_AUDIOIO_HPP
