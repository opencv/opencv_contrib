// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/photometric_calib/Reader.hpp"

namespace cv { namespace photometric_calib{

unsigned long Reader::getNumImages() const
{
    return (unsigned long)files.size();
}

void Reader::loadTimestamps(const std::string &timesFile)
{
    CV_Assert(timesFile.substr(timesFile.find_last_of(".") + 1) == "yaml" || timesFile.substr(timesFile.find_last_of(".") + 1) == "yml");

    FileStorage timeFile;
    timeFile.open(timesFile, FileStorage::READ);
    timeStamps.clear();
    exposureTimes.clear();

    CV_Assert(timeFile.isOpened());

    FileNode timeStampNode = timeFile["times"];
    FileNode exposureTimeNode = timeFile["exposures"];

    CV_Assert(timeStampNode.type() == FileNode::SEQ && exposureTimeNode.type() == FileNode::SEQ);

    FileNodeIterator itTs = timeStampNode.begin(), itTsEnd = timeStampNode.end();
    FileNodeIterator itEt = exposureTimeNode.begin(), itEtEnd = exposureTimeNode.end();

    for (; itTs != itTsEnd; ++itTs)
        timeStamps.push_back((double)*itTs);
    for (; itEt != itEtEnd; ++itEt)
        exposureTimes.push_back((float)*itEt);

    timeFile.release();

    CV_Assert(timeStamps.size() == getNumImages() && exposureTimes.size() == getNumImages());
}

Reader::Reader(const std::string &folderPath, const std::string &timesPath)
{
    String cvFolderPath(folderPath);
    glob(cvFolderPath, files);
    CV_Assert(files.size() > 0);
    std::sort(files.begin(), files.end());
    loadTimestamps(timesPath);

    width = 0;
    height = 0;

    for(size_t i = 0; i < files.size(); ++i)
    {
        Mat img = imread(files[i]);
        CV_Assert(img.type() == CV_8U);
        if(i == 0)
        {
            width = img.cols;
            height = img.rows;
        }
        else
        {
            CV_Assert(width == img.cols && height == img.rows);
        }
    }
}

double Reader::getTimestamp(unsigned long id) const
{
    CV_Assert(id < timeStamps.size());
    return timeStamps[id];
}

float Reader::getExposureTime(unsigned long id) const
{
    CV_Assert(id < exposureTimes.size());
    return exposureTimes[id];
}

}} // namespace photometric_calib, cv