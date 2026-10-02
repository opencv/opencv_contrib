// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_READER_HPP
#define _OPENCV_READER_HPP

#include "opencv2/core.hpp"

#include <vector>
#include <string>

namespace cv {
namespace photometric_calib {

//! @addtogroup photometric_calib
//! @{

/*!
 * @brief Class for reading the sequence used for photometric calibration. Both the folder path of the sequence
 * and the path of time file should be provided. The images of the sequence should be of format CV_8U. The time
 * file should be .yaml or .yml. In the time file, the timestamps and exposure duration of the corresponding images
 * of the sequence should be provided.
 *
 * The image paths are stored in std::vector<String> images, timestamps are stored in std::vector<double> timeStamps,
 * exposure duration is stored in std::vector<float> exposureTimes
 */

class CV_EXPORTS Reader
{
public:
    /*!
     * @brief Constructor
     * @param folderPath the path of folder which contains the images
     * @param imageExt the format of the input images, e.g., jpg or png.
     * @param timesPath the path of time file
     */
    Reader(const std::string &folderPath, const std::string &imageExt, const std::string &timesPath);

    /*!
     * @return the amount of images loaded
     */
    unsigned long getNumImages() const;


    /*!
     * @brief Given the id of the image and return the image. id is in fact just the index of the image in the
     * vector contains all the images.
     * @param id
     * @return Mat of the id^th image.
     */
    Mat getImage(unsigned long id) const;

    /*!
     * @brief Given the id of the image and return its timestamp value. id is in fact just the index of the image in the
     * vector contains all the images.
     * @param id
     * @return timestamp of the id^th image.
     */
    double getTimestamp(unsigned long id) const;

    /*!
     * @brief Given the id of the image and return its exposure duration when is was taken.
     * @param id
     * @return exposure duration of the image.
     */
    float getExposureDuration(unsigned long id) const;

    int getWidth() const;

    int getHeight() const;

    const std::string &getFolderPath() const;

    const std::string &getTimeFilePath() const;

private:
    /*!
     * @brief Load timestamps and exposure duration.
     * @param timesFile
     */
    inline void loadTimestamps(const std::string &timesFile);

    std::vector<String> images; //All the names/paths of images
    std::vector<double> timeStamps; //All the Unix Time Stamps of images
    std::vector<float> exposureDurations;//All the exposure duration for images

    int _width, _height;//The image width and height. All the images should be of the same size.

    std::string _folderPath;
    std::string _timeFilePath;
};

//! @}

} // namespace photometric_calib
} // namespace cv

#endif //_OPENCV_READER_HPP