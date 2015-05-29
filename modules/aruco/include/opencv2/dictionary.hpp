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


#ifndef __OPENCV_ARUCO_DICTIONARY_HPP__
#define __OPENCV_ARUCO_DICTIONARY_HPP__

#include <opencv2/core.hpp>
#include <vector>


namespace cv { namespace aruco {


    /**
     * @brief Dictionary/Set of markers. It contains the inner codification
     * 
     */
class CV_EXPORTS Dictionary {

public:  
    
    Dictionary() { };
    Dictionary(const char *quartets, int _markerSize, int _dictsize, int _maxcorr)  {
        markerSize = _markerSize;
        maxCorrectionBits = _maxcorr;
        int nquartets = (markerSize*markerSize)/4 + (markerSize*markerSize)%4;
        codes = cv::Mat(_dictsize, nquartets, CV_8UC1);
        for(int i=0; i<_dictsize; i++) {
            for(int j=0; j<nquartets; j++) codes.at<unsigned char>(i,j) = quartets[i*nquartets+j];
        }
    };

    cv::Mat codes;
    int markerSize;
    int maxCorrectionBits; // maximum number of bits that can be corrected
    // float borderSize; // black border size respect to inner bits size

    /**
     * @brief Given an image and four corners positions, identify the marker
     * 
     * @param image
     * @param imgPoints corner positions
     * @param idx marker idx if valid
     * @return true if identification is correct, else false
     */
    bool identify(InputArray image, InputOutputArray imgPoints, int &idx);

    
    /**
     * @brief Draw a canonical marker image
     *
     * @param img
     * @param id
     * @return void
     */
    void drawMarker(int id, int sidePixels, OutputArray img);



private:

    cv::Mat _getQuartet(cv::Mat bits);
    cv::Mat _getBits(cv::Mat quartets);
    cv::Mat _getDistances(cv::Mat quartets);
    cv::Mat _extractBits(InputArray image, InputOutputArray imgPoints);
    bool _isBorderValid(cv::Mat bits);

};


enum PREDEFINED_DICTIONARIES { DICT_ARUCO=0 };



}} 


#endif
