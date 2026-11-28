// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/photometric_calib/Reader.hpp"

namespace cv {
namespace photometric_calib {

unsigned long Reader::getNumImages() const
{
    return (unsigned long) images.size();
}

void Reader::loadTimestamps(const std::string &timesFile)
{
    // check the extension of the time file.
    CV_Assert(timesFile.substr(timesFile.find_last_of(".") + 1) == "yaml" ||
              timesFile.substr(timesFile.find_last_of(".") + 1) == "yml");

    FileStorage timeFile;
    timeFile.open(timesFile, FileStorage::READ);
    timeStamps.clear();
    exposureDurations.clear();

    CV_Assert(timeFile.isOpened());

    FileNode timeStampNode = timeFile["times"];
    FileNode exposureTimeNode = timeFile["exposures"];

    CV_Assert(timeStampNode.type() == FileNode::SEQ && exposureTimeNode.type() == FileNode::SEQ);

    FileNodeIterator itTs = timeStampNode.begin(), itTsEnd = timeStampNode.end();
    FileNodeIterator itEt = exposureTimeNode.begin(), itEtEnd = exposureTimeNode.end();

    for (; itTs != itTsEnd; ++itTs)
    {
        timeStamps.push_back((double) *itTs);
    }
    for (; itEt != itEtEnd; ++itEt)
    {
        exposureDurations.push_back((float) *itEt);
    }

    timeFile.release();

    CV_Assert(timeStamps.size() == getNumImages() && exposureDurations.size() == getNumImages());
    _timeFilePath = timesFile;
}

Reader::Reader(const std::string &folderPath, const std::string &imageExt, const std::string &timesPath)
{
    String cvFolderPath(folderPath);

#if defined WIN32 || defined _WIN32 || defined WINCE
    *cvFolderPath.end() == '\\' ? cvFolderPath = cvFolderPath : cvFolderPath += '\\';
#else
    *cvFolderPath.end() == '/' ? cvFolderPath = cvFolderPath : cvFolderPath += '/';
#endif

    cvFolderPath += ("*." + imageExt);
    glob(cvFolderPath, images);
    CV_Assert(images.size() > 0);
    std::sort(images.begin(), images.end());
    loadTimestamps(timesPath);

    _width = 0;
    _height = 0;

    // images should be of CV_8U and same size
    for (size_t i = 0; i < images.size(); ++i)
    {
        Mat img = imread(images[i], IMREAD_GRAYSCALE);
        CV_Assert(img.type() == CV_8U);
        if (i == 0)
        {
            _width = img.cols;
            _height = img.rows;
        }
        else
        {
            CV_Assert(_width == img.cols && _height == img.rows);
        }
    }

    _folderPath = folderPath;
}

Mat Reader::getImage(unsigned long id) const
{
    CV_Assert(id < images.size());
    return imread(images[id], IMREAD_GRAYSCALE);
}

double Reader::getTimestamp(unsigned long id) const
{
    CV_Assert(id < timeStamps.size());
    return timeStamps[id];
}

float Reader::getExposureDuration(unsigned long id) const
{
    CV_Assert(id < exposureDurations.size());
    return exposureDurations[id];
}

int Reader::getWidth() const
{
    return _width;
}

int Reader::getHeight() const
{
    return _height;
}

const std::string &Reader::getFolderPath() const
{
    return _folderPath;
}

const std::string &Reader::getTimeFilePath() const
{
    return _timeFilePath;
}

} // namespace photometric_calib
} // namespace cv