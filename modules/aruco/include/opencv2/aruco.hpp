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


#ifndef __OPENCV_ARUCO_HPP__
#define __OPENCV_ARUCO_HPP__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <vector>


#include <iostream>


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
    bool identify(InputArray image, InputArray imgPoints, int &idx);

    
    /**
     * @brief Draw a canonical marker image
     * 
     * @param img
     * @param id
     * @return void
     */
    void drawMarker(InputOutputArray img, int id);



private:

    cv::Mat _getQuartet(cv::Mat bits);
    cv::Mat _getDistances(cv::Mat quartets);

};


extern const char quartets_distances[16][16][4];
extern const char dict_hrm_4x4_quartets[][4] = {{ 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 }};
extern const Dictionary DICT_HRM_4X4 (&(dict_hrm_4x4_quartets[0][0]), 4, 5, 4);
//const Dictionary DICT_HRM_4X4;

// predefined marker dictionaries
//const Dictionary DICT_HRM_4X4, DICT_HRM_5X5, DICT_HRM_6X6, DICT_ARUCO, DICT_ARTAG; // ...;





/**
 * @brief Board of markers containing the layout of the markers
 * 
 */
class CV_EXPORTS Board {

public:
    
    Dictionary dictionary;
    std::vector< std::vector<cv::Point3f> > objPoints; // each marker include its 4 corners, i.e. for M marker, Mx4

    
    /**
     * @brief Draw board image considering (x,y) coordinates (ignoring Z)
     * 
     * @param image 
     * @return void
     */
    void drawBoard(InputOutputArray image);

    
    
    /**
     * @brief Fast creation of a planar Board object
     * 
     * @param width number of markers in X
     * @param height number of markers in Y
     * @param markerSize marker side distance (in meters)
     * @param markerSeparation consecutive marker separation (in meters)
     * @param dictionary
     * @return cv::aruco::Board
     */
    static Board createPlanarBoard(int width, int height, float markerSize, 
				   float markerSeparation, Dictionary dictionary);
    
};





/**
 * @brief Detect single markers of the specific dictionary in an image
 * 
 * @param image input image
 * @param cameraMatrix 
 * @param distCoeffs
 * @param markersize size of marker side in meters
 * @param dictionary markers are identified based on this dictionary
 * @param imgPoints returns detected markers corner positions
 * @param ids returns detected markers ids
 * @param Rvec returns rvec for each marker. If NoArray, then pose is not calculated
 * @param Tvec same as Tvec
 * @param threshParam window size for adaptative thresholding
 * @param minLenght minimum size of candidates contour lenght. It is indicated as a ratio 
 *                  respect to the largest image dimension
 * @return void
 */
CV_EXPORTS void detectSingleMarkers(InputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                       float markersize, Dictionary dictionary, OutputArrayOfArrays imgPoints,
                       OutputArray ids, OutputArray rvec=noArray(), OutputArray tvec=noArray(),
                       int threshParam=21,float minLenght=0.03);




/**
 * @brief Detect an Aruco Board
 * 
 * @param image input image
 * @param cameraMatrix
 * @param distCoeffs
 * @param board board configuration including the layout and dictionary
 * @param imgPoints returns detected markers corner positions
 * @param ids returns detected markers ids
 * @param rvec ...
 * @param tvec ...
 * @param threshParam window size for adaptative thresholding
 * @param minLenght minimum size of candidates contour lenght. It is indicated as a ratio 
 *                  respect to the largest image dimension
 * @return void
 */
CV_EXPORTS void detectBoardMarkers(InputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                      Board board, OutputArrayOfArrays imgPoints, OutputArray ids,
                      OutputArray rvec, OutputArray tvec, int threshParam=21, float minLenght=0.03);





/**
 * @brief Draw detected markers in image
 * 
 * @param image 
 * @param markers maker corner positions
 * @param ids marker ids
 * @return void
 */
CV_EXPORTS void drawDetectedMarkers(InputArray image, InputArrayOfArrays markers, InputArray ids);


/**
 * @brief Draw colored axis in image
 *
 * @param image
 * @param cameraMatrix
 * @param distCoeffs
 * @param rvec
 * @param tvec
 * @param lenght axis lenght in meters
 * @return void
 */
CV_EXPORTS void drawAxis(InputArray image, InputArray cameraMatrix, InputArray distCoeffs, InputArray rvec, InputArray tvec, float lenght);




//CV_EXPORTS void calibrateCamera(InputArrayOfArrays images, Board board, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, OutputArrayOfArrays imgPoints, OutputArrayOfArrays ids, OutputArray tvecs, OutputArray tvecs, int threshParam, int minLenght)



}} 


#endif
