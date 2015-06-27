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

#include "test_precomp.hpp"
#include <opencv2/highgui.hpp>
#include <string>

using namespace std;

/* ///////////////////// aruco_detection_simple ///////////////////////// */

class CV_ArucoDetectionSimple : public cvtest::BaseTest
{
public:
    CV_ArucoDetectionSimple();
protected:
    void run(int);
};


CV_ArucoDetectionSimple::CV_ArucoDetectionSimple() {}


void CV_ArucoDetectionSimple::run(int) {

    for(int i=0; i<20; i++) {

        std::vector< std::vector<cv::Point2f> > groundTruthCorners;
        std::vector< int > groundTruthIds;

        const int markerSidePixels = 100;
        int imageSize = markerSidePixels*2 + 3*(markerSidePixels/2);

        cv::Mat img = cv::Mat(imageSize, imageSize, CV_8UC1, cv::Scalar::all(255));
        for(int y=0; y<2; y++) {
            for(int x=0; x<2; x++) {
                cv::Mat marker;
                int id = i*4 + y*2 + x;
                cv::aruco::drawMarker(cv::aruco::DICT_6X6_250, id, markerSidePixels, marker);
                cv::Point2f firstCorner = cv::Point2f(markerSidePixels/2 + x*(1.5*markerSidePixels),
                                                      markerSidePixels/2 + y*(1.5*markerSidePixels)
                                                      );
                cv::Mat aux = img.colRange(firstCorner.x, firstCorner.x+markerSidePixels)
                                 .rowRange(firstCorner.y, firstCorner.y+markerSidePixels);
                marker.copyTo(aux);
                groundTruthIds.push_back(id);
                groundTruthCorners.push_back( std::vector<cv::Point2f>() );
                groundTruthCorners.back().push_back(firstCorner);
                groundTruthCorners.back().push_back(firstCorner +
                                                    cv::Point2f(markerSidePixels-1,0));
                groundTruthCorners.back().push_back(firstCorner +
                                                    cv::Point2f(markerSidePixels-1,
                                                                markerSidePixels-1));
                groundTruthCorners.back().push_back(firstCorner +
                                                    cv::Point2f(0,markerSidePixels-1));
            }
        }
        if(i%2==1) img.convertTo(img, CV_8UC3);

        std::vector< std::vector<cv::Point2f> > corners;
        std::vector< int > ids;
        cv::aruco::DetectorParameters params;
        params.doCornerRefinement = false;
        cv::aruco::detectMarkers(img, cv::aruco::DICT_6X6_250, corners, ids, params);
        for (int m=0; m<groundTruthIds.size(); m++) {
            int idx = -1;
            for(int k=0; k<ids.size(); k++) {
                if(groundTruthIds[m] == ids[k]) {
                    idx = k;
                    break;
                }
            }
            if(idx == -1) {
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                return;
            }

            for(int c=0; c<4; c++) {
                double dist = cv::norm( groundTruthCorners[m][c] - corners[idx][c] );
                if(dist > 0.001) {
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    return;
                }
            }

        }

    }

}


TEST(Aruco_SimpleMarkerDetection, algorithmic) { CV_ArucoDetectionSimple test; test.safe_run(); }

/* End of file. */
