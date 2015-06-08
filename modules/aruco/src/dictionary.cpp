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

#ifndef __OPENCV_ARUCO_DICTIONARY_CPP__
#define __OPENCV_ARUCO_DICTIONARY_CPP__
#ifdef __cplusplus

#include "precomp.hpp"
#include "predefined_dictionaries.cpp"

#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>



namespace cv{ namespace aruco{

using namespace std;




const char quartets_distances[16][16][4] =
{
    { {0,0,0,0},{1,1,1,1},{1,1,1,1},{2,2,2,2},{1,1,1,1},{2,2,2,2},{2,2,2,2},{3,3,3,3},{1,1,1,1},{2,2,2,2},{2,2,2,2},{3,3,3,3},{2,2,2,2},{3,3,3,3},{3,3,3,3},{4,4,4,4}, },
    { {1,1,1,1},{0,2,2,2},{2,0,2,2},{1,1,3,3},{2,2,0,2},{1,3,1,3},{3,1,1,3},{2,2,2,4},{2,2,2,0},{1,3,3,1},{3,1,3,1},{2,2,4,2},{3,3,1,1},{2,4,2,2},{4,2,2,2},{3,3,3,3}, },
    { {1,1,1,1},{2,2,2,0},{0,2,2,2},{1,3,3,1},{2,0,2,2},{3,1,3,1},{1,1,3,3},{2,2,4,2},{2,2,0,2},{3,3,1,1},{1,3,1,3},{2,4,2,2},{3,1,1,3},{4,2,2,2},{2,2,2,4},{3,3,3,3}, },
    { {2,2,2,2},{1,3,3,1},{1,1,3,3},{0,2,4,2},{3,1,1,3},{2,2,2,2},{2,0,2,4},{1,1,3,3},{3,3,1,1},{2,4,2,0},{2,2,2,2},{1,3,3,1},{4,2,0,2},{3,3,1,1},{3,1,1,3},{2,2,2,2}, },
    { {1,1,1,1},{2,2,0,2},{2,2,2,0},{3,3,1,1},{0,2,2,2},{1,3,1,3},{1,3,3,1},{2,4,2,2},{2,0,2,2},{3,1,1,3},{3,1,3,1},{4,2,2,2},{1,1,3,3},{2,2,2,4},{2,2,4,2},{3,3,3,3}, },
    { {2,2,2,2},{1,3,1,3},{3,1,3,1},{2,2,2,2},{1,3,1,3},{0,4,0,4},{2,2,2,2},{1,3,1,3},{3,1,3,1},{2,2,2,2},{4,0,4,0},{3,1,3,1},{2,2,2,2},{1,3,1,3},{3,1,3,1},{2,2,2,2}, },
    { {2,2,2,2},{3,3,1,1},{1,3,3,1},{2,4,2,0},{1,1,3,3},{2,2,2,2},{0,2,4,2},{1,3,3,1},{3,1,1,3},{4,2,0,2},{2,2,2,2},{3,3,1,1},{2,0,2,4},{3,1,1,3},{1,1,3,3},{2,2,2,2}, },
    { {3,3,3,3},{2,4,2,2},{2,2,4,2},{1,3,3,1},{2,2,2,4},{1,3,1,3},{1,1,3,3},{0,2,2,2},{4,2,2,2},{3,3,1,1},{3,1,3,1},{2,2,2,0},{3,1,1,3},{2,2,0,2},{2,0,2,2},{1,1,1,1}, },
    { {1,1,1,1},{2,0,2,2},{2,2,0,2},{3,1,1,3},{2,2,2,0},{3,1,3,1},{3,3,1,1},{4,2,2,2},{0,2,2,2},{1,1,3,3},{1,3,1,3},{2,2,2,4},{1,3,3,1},{2,2,4,2},{2,4,2,2},{3,3,3,3}, },
    { {2,2,2,2},{1,1,3,3},{3,1,1,3},{2,0,2,4},{3,3,1,1},{2,2,2,2},{4,2,0,2},{3,1,1,3},{1,3,3,1},{0,2,4,2},{2,2,2,2},{1,1,3,3},{2,4,2,0},{1,3,3,1},{3,3,1,1},{2,2,2,2}, },
    { {2,2,2,2},{3,1,3,1},{1,3,1,3},{2,2,2,2},{3,1,3,1},{4,0,4,0},{2,2,2,2},{3,1,3,1},{1,3,1,3},{2,2,2,2},{0,4,0,4},{1,3,1,3},{2,2,2,2},{3,1,3,1},{1,3,1,3},{2,2,2,2}, },
    { {3,3,3,3},{2,2,4,2},{2,2,2,4},{1,1,3,3},{4,2,2,2},{3,1,3,1},{3,1,1,3},{2,0,2,2},{2,4,2,2},{1,3,3,1},{1,3,1,3},{0,2,2,2},{3,3,1,1},{2,2,2,0},{2,2,0,2},{1,1,1,1}, },
    { {2,2,2,2},{3,1,1,3},{3,3,1,1},{4,2,0,2},{1,3,3,1},{2,2,2,2},{2,4,2,0},{3,3,1,1},{1,1,3,3},{2,0,2,4},{2,2,2,2},{3,1,1,3},{0,2,4,2},{1,1,3,3},{1,3,3,1},{2,2,2,2}, },
    { {3,3,3,3},{2,2,2,4},{4,2,2,2},{3,1,1,3},{2,4,2,2},{1,3,1,3},{3,3,1,1},{2,2,0,2},{2,2,4,2},{1,1,3,3},{3,1,3,1},{2,0,2,2},{1,3,3,1},{0,2,2,2},{2,2,2,0},{1,1,1,1}, },
    { {3,3,3,3},{4,2,2,2},{2,4,2,2},{3,3,1,1},{2,2,4,2},{3,1,3,1},{1,3,3,1},{2,2,2,0},{2,2,2,4},{3,1,1,3},{1,3,1,3},{2,2,0,2},{1,1,3,3},{2,0,2,2},{0,2,2,2},{1,1,1,1}, },
    { {4,4,4,4},{3,3,3,3},{3,3,3,3},{2,2,2,2},{3,3,3,3},{2,2,2,2},{2,2,2,2},{1,1,1,1},{3,3,3,3},{2,2,2,2},{2,2,2,2},{1,1,1,1},{2,2,2,2},{1,1,1,1},{1,1,1,1},{0,0,0,0}, },
};





/**
 * @brief Dictionary/Set of markers. It contains the inner codification
 *
 */
class DictionaryData {

public:

DictionaryData() { };
DictionaryData(const char *quartets, int _markerSize, int _dictsize, int _maxcorr)  {
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
bool identify(InputArray image, InputOutputArray imgPoints, int &idx) {
    // get bits
    cv::Mat candidateBits = _extractBits(image, imgPoints);
    if(!_isBorderValid(candidateBits)) return false; // not really necessary
    cv::Mat onlyBits = candidateBits.rowRange(1,candidateBits.rows-1).colRange(1,candidateBits.rows-1);

    // get quartets
    cv::Mat candidateQuartets = _getQuartet(onlyBits);

    // search closest marker in dict
    int closestId=-1;
    unsigned int rotation=0;
    unsigned int closestDistance=markerSize*markerSize+1;
    cv::Mat candidateDistances = _getDistances(candidateQuartets);

    for(int i=0; i<codes.rows; i++) {
        if(candidateDistances.ptr<int>(i)[0] < closestDistance) {
            closestDistance = candidateDistances.ptr<int>(i)[0];
            closestId = i;
            rotation = candidateDistances.ptr<int>(i)[1];
        }
    }

    // return closest id
    if(closestId!=-1 && closestDistance<=maxCorrectionBits) {
        idx = closestId;
        // correct imgPoints positions
        if(rotation!=0) {
            cv::Mat copyPoints = imgPoints.getMat().clone();
            for(int j=0; j<4; j++) imgPoints.getMat().ptr<cv::Point2f>(0)[j] = copyPoints.ptr<cv::Point2f>(0)[(j+4-rotation)%4];
        }
        return true;
    }
    else {
        idx = -1;
        return false;
    }
}


/**
 * @brief Draw a canonical marker image
 *
 * @param img
 * @param id
 * @return void
 */
void drawMarker(int id, int sidePixels, OutputArray img) {
    img.create(sidePixels, sidePixels, CV_8UC1);
    cv::Mat tinyMarker(markerSize+2, markerSize+2, CV_8UC1, cv::Scalar::all(0));
    cv::Mat innerRegion = tinyMarker.rowRange(1,tinyMarker.rows-1).colRange(1,tinyMarker.cols-1);
    cv::Mat bits = 255*_getBits( codes.rowRange(id, id+1) );
    bits.copyTo(innerRegion);
    cv::resize(tinyMarker, img.getMat(), img.getMat().size(),0,0,cv::INTER_NEAREST);
}


private:

cv::Mat _getQuartet(cv::Mat bits) {

    int nquartets = (markerSize*markerSize)/4 + (markerSize*markerSize)%4;
    cv::Mat candidateQuartets(1, nquartets, CV_8UC1);
    int currentQuartet=0;
    for(int row=0; row<markerSize/2; row++)
    {
        for(int col=row; col<markerSize-row-1; col++) {
            unsigned char bit3 = bits.at<unsigned char>(row,col);
            unsigned char bit2 = bits.at<unsigned char>(col,markerSize-1-row);
            unsigned char bit1 = bits.at<unsigned char>(markerSize-1-row,markerSize-1-col);
            unsigned char bit0 = bits.at<unsigned char>(markerSize-1-col,row);
            unsigned char quartet = 8*bit3 + 4*bit2 + 2*bit1 + bit0;
            candidateQuartets.ptr<unsigned char>()[currentQuartet] = quartet;
            currentQuartet++;
        }
    }
    if((markerSize*markerSize)%4 == 1) { // middle bit
        unsigned char middleBit = bits.at<unsigned char>(markerSize/2,markerSize/2);
        candidateQuartets.ptr<unsigned char>()[currentQuartet] = middleBit;
    }
    return candidateQuartets;
}

cv::Mat _getBits(cv::Mat quartets) {
    cv::Mat bits(markerSize, markerSize, CV_8UC1);
    int currentQuartetIdx=0;
    for(int row=0; row<markerSize/2; row++)
    {
        for(int col=row; col<markerSize-row-1; col++) {
            unsigned char currentQuartet = quartets.ptr<unsigned char>(0)[currentQuartetIdx];
            unsigned char bit3 = 0;
            if(currentQuartet>=8) {
                bit3 = 1;
                currentQuartet-=8;
            }
            bits.at<unsigned char>(row,col) = bit3;
            unsigned char bit2 = 0;
            if(currentQuartet>=4) {
                bit2 = 1;
                currentQuartet-=4;
            }
            bits.at<unsigned char>(col,markerSize-1-row) = bit2;
            unsigned char bit1 = 0;
            if(currentQuartet>=2) {
                bit1 = 1;
                currentQuartet-=2;
            }
            bits.at<unsigned char>(markerSize-1-row,markerSize-1-col)= bit1;
            unsigned char bit0 = currentQuartet;
            bits.at<unsigned char>(markerSize-1-col,row) = bit0;
            currentQuartetIdx++;
        }
    }
    if((markerSize*markerSize)%4 == 1) { // middle bit
        if(quartets.ptr<unsigned char>()[currentQuartetIdx]==0) bits.at<unsigned char>(markerSize/2,markerSize/2) = 0;
        else bits.at<unsigned char>(markerSize/2,markerSize/2) = 1;
    }
    return bits;
}

cv::Mat _getDistances(cv::Mat quartets) {

    bool middleBit = (markerSize%2==1);
    int ncompleteQuartets = quartets.total() - (middleBit?1:0);

    cv::Mat res(codes.rows, 2, CV_32SC1);
    for(unsigned int m=0; m<codes.rows; m++) {
        res.ptr<int>(m)[0]=10e8;
        for(unsigned int r=0; r<4; r++) {
            int currentHamming=0;
            for(unsigned int q=0; q<ncompleteQuartets; q++) {
                currentHamming += (int)quartets_distances[ (codes.ptr<unsigned char>(m)[q]) ][ (quartets.ptr<unsigned char>(0)[q]) ][r];
            }
            if(middleBit && ((codes.ptr<unsigned char>(m)[ncompleteQuartets])!=quartets.ptr<unsigned char>(0)[ncompleteQuartets]))
                currentHamming++;
            if(currentHamming < res.ptr<int>(m)[0]) {
                res.ptr<int>(m)[0]=currentHamming;
                res.ptr<int>(m)[1]=r;
            }
        }
    }
    return res;
}

cv::Mat _extractBits(InputArray image, InputOutputArray imgPoints) {

    CV_Assert(image.getMat().channels()==1);

    cv::Mat resultImg; // marker image after removing perspective
    int squareSizePixels = 8;
    int resultImgSize = (markerSize+2)*squareSizePixels;
    cv::Mat resultImgCorners(4,1,CV_32FC2);
    resultImgCorners.ptr<cv::Point2f>(0)[0]= Point2f ( 0,0 );
    resultImgCorners.ptr<cv::Point2f>(0)[1]= Point2f ( resultImgSize-1,0 );
    resultImgCorners.ptr<cv::Point2f>(0)[2]= Point2f ( resultImgSize-1,resultImgSize-1 );
    resultImgCorners.ptr<cv::Point2f>(0)[3]= Point2f ( 0,resultImgSize-1 );

    // remove perspective
    cv::Mat transformation = cv::getPerspectiveTransform(imgPoints, resultImgCorners);
    cv::warpPerspective(image, resultImg, transformation, cv::Size(resultImgSize, resultImgSize), cv::INTER_NEAREST);

    // now extract code
    cv::Mat bits(markerSize+2, markerSize+2, CV_8UC1, cv::Scalar::all(0));
    cv::threshold(resultImg, resultImg,125, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
    for (unsigned int y=0; y<markerSize+2; y++)  {
        for (unsigned int x=0; x<markerSize+2;x++) {
            int Xstart=x*(squareSizePixels)+1;
            int Ystart=y*(squareSizePixels)+1;
            cv::Mat square=resultImg(cv::Rect(Xstart,Ystart,squareSizePixels-2,squareSizePixels-2));
            int nZ=countNonZero(square);
            if (nZ> square.total()/2)  bits.at<unsigned char>(y,x)=1;
        }
     }

    return bits;
}

bool _isBorderValid(cv::Mat bits) {
    int sizeWithBorders = markerSize+2;
    int totalErrors = 0;
    for(int y=0; y<sizeWithBorders; y++) {
        if(bits.ptr<unsigned char>(y)[0]!=0) totalErrors++;
        if(bits.ptr<unsigned char>(y)[sizeWithBorders-1]!=0) totalErrors++;
    }
    for(int x=1; x<sizeWithBorders-1; x++) {
        if(bits.ptr<unsigned char>(0)[x]!=0) totalErrors++;
        if(bits.ptr<unsigned char>(sizeWithBorders-1)[x]!=0) totalErrors++;
    }
    if(totalErrors > 1) return false; // markersize is a good value for check border errors
    else return true;
}

};



// PREDEFINED DICTIONARIES
const DictionaryData _dict_aruco_data = DictionaryData (&(_dict_aruco_quartets[0][0]), 5, 1024, 1);


}}

#endif // cplusplus
#endif // __OPENCV_ARUCO_DICTIONARY_CPP__

