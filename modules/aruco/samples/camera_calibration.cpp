/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Use: board_detector video" << std::endl;
        return 0;
    }

    cv::VideoCapture input;
    input.open(argv[1]);

    cv::aruco::Board b = cv::aruco::createPlanarBoard(4, 6, 0.04, 0.008, cv::aruco::DICT_ARUCO);
    b.ids.clear();
    b.ids.push_back(985);
    b.ids.push_back(838);
    b.ids.push_back(908);
    b.ids.push_back(299);
    b.ids.push_back(428);
    b.ids.push_back(177);

    b.ids.push_back(64);
    b.ids.push_back(341);
    b.ids.push_back(760);
    b.ids.push_back(882);
    b.ids.push_back(982);
    b.ids.push_back(977);

    b.ids.push_back(477);
    b.ids.push_back(125);
    b.ids.push_back(717);
    b.ids.push_back(791);
    b.ids.push_back(618);
    b.ids.push_back(76);

    b.ids.push_back(181);
    b.ids.push_back(1005);
    b.ids.push_back(175);
    b.ids.push_back(684);
    b.ids.push_back(233);
    b.ids.push_back(461);

    std::vector<std::vector<std::vector<cv::Point2f> > > allImgPoints;
    std::vector<std::vector<int> > allIds;
    cv::Size imgSize;

    while (input.grab()) {
        cv::Mat image, imageCopy;
        input.retrieve(image);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > imgPoints;
        cv::Mat rvec, tvec;

        // detect markers and estimate pose
        cv::aruco::detectMarkers(image, cv::aruco::DICT_ARUCO, imgPoints, ids);

        // draw results
        if (ids.size() > 0)
            cv::aruco::drawDetectedMarkers(image, imageCopy, imgPoints, ids);
        else
            image.copyTo(imageCopy);

        cv::imshow("out", imageCopy);
        char key = cv::waitKey(0);
        if (key == 27)
            break;
        if (key == 'y') {
            std::cout << "Frame captured" << std::endl;
            allImgPoints.push_back(imgPoints);
            allIds.push_back(ids);
            imgSize = image.size();
        }
    }

    cv::Mat cameraMatrix, distCoeffs;
    double repError =
        cv::aruco::calibrateCameraAruco(allImgPoints, allIds, b, imgSize, cameraMatrix, distCoeffs);

    std::cout << "Rep Error: " << repError << std::endl;
    std::cout << "cameraMatrix:" << std::endl;
    std::cout << cameraMatrix << std::endl;
    std::cout << "distCoeffs:" << std::endl;
    std::cout << distCoeffs << std::endl;

    return 0;
}
