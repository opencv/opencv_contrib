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
#include <opencv2/aruco/charuco.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;


/**
 */
static void help() {
    cout << "Create a ChArUco marker image" << endl;
    cout << "Parameters: " << endl;
    cout << "-o <image> # Output image" << endl;
    cout << "-sl <squareLength> # Square side lenght (in pixels)" << endl;
    cout << "-ml <markerLength> # Marker side lenght (in pixels)" << endl;
    cout << "-d <dictionary> # DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, "
         << "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
         << "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
         << "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16" << endl;
    cout << "-ids <id1,id2,id3,id4> # Four ids for the ChArUco marker" << endl;
    cout << "[-m <marginSize>] # Margins size (in pixels). Default is 0" << endl;
    cout << "[-bb <int>] # Number of bits in marker borders. Default is 1" << endl;
    cout << "[-si] # show generated image" << endl;
}


/**
 */
static bool isParam(string param, int argc, char **argv) {
    for(int i = 0; i < argc; i++)
        if(string(argv[i]) == param) return true;
    return false;
}


/**
 */
static string getParam(string param, int argc, char **argv, string defvalue = "") {
    int idx = -1;
    for(int i = 0; i < argc && idx == -1; i++)
        if(string(argv[i]) == param) idx = i;
    if(idx == -1 || (idx + 1) >= argc)
        return defvalue;
    else
        return argv[idx + 1];
}


/**
 */
int main(int argc, char *argv[]) {

    if(!isParam("-sl", argc, argv) || !isParam("-ml", argc, argv) || !isParam("-d", argc, argv) ||
       !isParam("-ids", argc, argv)) {
        help();
        return 0;
    }

    int squareLength = atoi(getParam("-sl", argc, argv).c_str());
    int markerLength = atoi(getParam("-ml", argc, argv).c_str());
    int dictionaryId = atoi(getParam("-d", argc, argv).c_str());
    aruco::Dictionary dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    string idsString = getParam("-ids", argc, argv);
    istringstream ss(idsString);
    vector< string > splittedIds;
    string token;
    while(getline(ss, token, ','))
        splittedIds.push_back(token);
    if(splittedIds.size() < 4) {
        cerr << "Incorrect ids format" << endl;
        help();
        return 0;
    }
    Vec4i ids;
    for(int i = 0; i < 4; i++)
        ids[i] = atoi(splittedIds[i].c_str());

    int margins = 0;
    if(isParam("-m", argc, argv)) {
        margins = atoi(getParam("-m", argc, argv).c_str());
    }

    int borderBits = 1;
    if(isParam("-bb", argc, argv)) {
        borderBits = atoi(getParam("-bb", argc, argv).c_str());
    }

    bool showImage = false;
    if(isParam("-si", argc, argv)) showImage = true;

    Mat markerImg;
    aruco::drawCharucoDiamond(dictionary, ids, squareLength, markerLength, markerImg, margins,
                              borderBits);

    if(showImage) {
        imshow("board", markerImg);
        waitKey(0);
    }

    imwrite(getParam("-o", argc, argv), markerImg);

    return 0;
}
