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


/**
 */
static void help() {
    std::cout << "Create an ArUco marker image" << std::endl;
    std::cout << "Parameters: " << std::endl;
    std::cout << "-o <image> # Output image" << std::endl;
    std::cout << "-d <dictionary> # 0: ARUCO, ..." << std::endl;
    std::cout << "-id <int> # Marker id in the dictionary" << std::endl;
    std::cout << "[-ms <int>] # Marker size in pixels. Default is 200" << std::endl;
    std::cout << "[-bb <int>] # Number of bits in marker borders. Default is 1" << std::endl;
    std::cout << "[-si] # show generated image" << std::endl;
}


/**
 */
static bool isParam(string param, int argc, char **argv ) {
    for (int i=0; i<argc; i++)
        if (string(argv[i]) == param )
            return true;
    return false;

}


/**
 */
static string getParam(string param, int argc, char **argv, string defvalue = "") {
    int idx=-1;
    for (int i=0; i<argc && idx==-1; i++)
        if (string(argv[i]) == param)
            idx = i;
    if (idx == -1 || (idx + 1) >= argc)
        return defvalue;
    else
        return argv[idx+1];
}


/**
 */
int main(int argc, char *argv[]) {

    if (!isParam("-d", argc, argv) || !isParam("-o", argc, argv) ) {
        help();
        return 0;
    }

    int dictionaryId = atoi( getParam("-d", argc, argv).c_str() );
    cv::aruco::DICTIONARY dictionary = cv::aruco::DICTIONARY(dictionaryId);

    int markerId = atoi( getParam("-id", argc, argv).c_str() );

    int borderBits = 1;
    if (isParam("-bb", argc, argv)) {
      borderBits = atoi( getParam("-bb", argc, argv).c_str() );
    }

    int markerSize = 200;
    if (isParam("-ms", argc, argv)) {
      markerSize = atoi( getParam("-ms", argc, argv).c_str() );
    }

    bool showImage = false;
    if (isParam("-si", argc, argv))
      showImage = true;

    cv::Mat markerImg;
    cv::aruco::drawMarker(dictionary, markerId, markerSize, markerImg, borderBits);

    if (showImage) {
      cv::imshow("marker", markerImg);
      cv::waitKey(0);
    }

    cv::imwrite( getParam("-o", argc, argv), markerImg);

    return 0;
}
