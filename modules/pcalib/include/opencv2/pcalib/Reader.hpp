#ifndef _OPENCV_READER_HPP
#define _OPENCV_READER_HPP

#include "opencv2/pcalib.hpp"

namespace cv { namespace pcalib{

class Reader
{
public:
    Reader(std::string folderPath, std::string timesPath);

    unsigned long getNumImages();

    double getTimestamp(unsigned long id);

    float getExposureTime(unsigned long id);


private:
    inline void loadTimestamps(std::string timesFile);

    std::vector<String> files;
    std::vector<double> timeStamps;
    std::vector<float> exposureTimes;

    int width, height;

    String path;
};

}} // namespace cv pcalib
#endif //_OPENCV_READER_HPP
