//
// Created by 杨楠 on 17/6/14.
//

#include "precomp.hpp"
#include "opencv2/pcalib/Reader.hpp"

namespace cv { namespace pcalib{

unsigned long Reader::getNumImages()
{
    return (unsigned long)files.size();
}

void Reader::loadTimestamps(std::string timesFile)
{
    std::ifstream timesStream;
    timesStream.open(timesFile.c_str());
    timeStamps.clear();
    exposureTimes.clear();
    while (!timesStream.eof() && timesStream.good())
    {
        char buf[1000];
        timesStream.getline(buf, 1000);

        int id = 0;
        double timeStamp = 0.0;
        float exposureTime = 0.0;

        CV_Assert(3 == scanf(buf, "%d %lf %f", &id, &timeStamp, &exposureTime));

        timeStamps.push_back(timeStamp);
        exposureTimes.push_back(exposureTime);
    }
    timesStream.close();

    CV_Assert(timeStamps.size() == getNumImages() && exposureTimes.size() == getNumImages());
}

Reader::Reader(std::string folderPath, std::string timesPath)
{
    String cvFolderPath(folderPath);
    glob(cvFolderPath, files);
    CV_Assert(files.size() > 0);
    std::sort(files.begin(), files.end());
    loadTimestamps(timesPath);

    width = 0;
    height = 0;

    for(unsigned long i = 0; i < files.size(); ++i)
    {
        Mat img = imread(files[i]);
        CV_Assert(img.type() == CV_8U);
        if(0 == i)
        {
            width = img.cols;
            height = img.rows;
        }
        else
        {
            CV_Assert(width == img.cols && height == img.rows);
        }
    }

    std::cout<<getNumImages()<<" imgases from"<<folderPath<<" loaded successfully!"<<std::endl;
}

double Reader::getTimestamp(unsigned long id)
{
    CV_Assert(id < timeStamps.size());
    return timeStamps[id];
}

float Reader::getExposureTime(unsigned long id)
{
    CV_Assert(id < exposureTimes.size());
    return exposureTimes[id];
}

}}