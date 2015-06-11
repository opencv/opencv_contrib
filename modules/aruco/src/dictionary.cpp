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

#include <iostream>

namespace cv {
namespace aruco {

using namespace std;



/**
  * Hamming weight look up table from 0 to 255
  */
const unsigned char hammingWeightLUT[] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};



/**
 * @brief Dictionary/Set of markers. It contains the inner codification
 *
 */
class DictionaryData {

  public:
    cv::Mat bytesList;
    int markerSize;
    int maxCorrectionBits; // maximum number of bits that can be corrected
    // float borderSize; // black border size respect to inner bits size


    /**
      */
    DictionaryData(const char * bytes = 0, int _markerSize = 0, int dictsize = 0,
                   int _maxcorr = 0) {
        markerSize = _markerSize;
        maxCorrectionBits = _maxcorr;
        int nbytes = (markerSize * markerSize) / 8;
        if ((markerSize * markerSize) % 8 != 0)
            nbytes++;

        // save bytes in internal format
        // bytesList.at<cv::Vec4b>(i, j)[k] is j-th byte of i-th marker, in its k-th rotation
        bytesList = cv::Mat(dictsize, nbytes, CV_8UC4);
        for (int i = 0; i < dictsize; i++) {
            for (int j = 0; j < nbytes; j++) {
                for (int k = 0; k < 4; k++)
                    bytesList.at<cv::Vec4b>(i, j)[k] = bytes[i * (4 * nbytes) + k * nbytes + j];
            }
        }
    }



    /**
     * @brief Given a matrix of bits. Returns whether if marker is identified or not.
     * It returns by reference the correct id (if any) and the correct rotation
     */
    bool identify(const cv::Mat &onlyBits, int &idx, int &rotation) const {

        CV_Assert(onlyBits.rows == markerSize && onlyBits.cols == markerSize);

        // get as a byte list
        cv::Mat candidateBytes = _getByteListFromBits(onlyBits);

        // search closest marker in dict
        int closestId = -1;
        rotation = 0;
        unsigned int closestDistance = markerSize * markerSize + 1;
        cv::Mat candidateDistances = _getDistances(candidateBytes);

        for (int i = 0; i < bytesList.rows; i++) {
            if (candidateDistances.ptr<int>(i)[0] < closestDistance) {
                closestDistance = candidateDistances.ptr<int>(i)[0];
                closestId = i;
                rotation = candidateDistances.ptr<int>(i)[1];
            }
        }

        // return closest id
        if (closestId != -1 && closestDistance <= maxCorrectionBits) {
            idx = closestId;
            return true;
        } else {
            idx = -1;
            return false;
        }
    }



    /**
     * @brief Draw a canonical marker image
     */
    void drawMarker(int id, int sidePixels, OutputArray _img) const {

        CV_Assert(sidePixels > markerSize);
        CV_Assert(id < bytesList.rows);

        _img.create(sidePixels, sidePixels, CV_8UC1);

        // create small marker with 1 pixel per bin
        cv::Mat tinyMarker(markerSize + 2, markerSize + 2, CV_8UC1, cv::Scalar::all(0));
        cv::Mat innerRegion =
            tinyMarker.rowRange(1, tinyMarker.rows - 1).colRange(1, tinyMarker.cols - 1);
        // put inner bits
        cv::Mat bits = 255 * _getBitsFromByteList(bytesList.rowRange(id, id + 1));
        bits.copyTo(innerRegion);

        // resize tiny marker to output size
        cv::resize(tinyMarker, _img.getMat(), _img.getMat().size(), 0, 0, cv::INTER_NEAREST);
    }



  private:


    /**
      * @brief Transform matrix of bits to list of bytes in the 4 rotations
      */
    cv::Mat _getByteListFromBits(const cv::Mat &bits) const {

        int nbytes = (bits.cols * bits.rows) / 8;
        if ((bits.cols * bits.rows) % 8 != 0)
            nbytes++;
        cv::Mat candidateByteList(1, nbytes, CV_8UC4, cv::Scalar::all(0));
        unsigned char currentBit = 0;
        int currentByte = 0;
        for (int row = 0; row < bits.rows; row++) {
            for (int col = 0; col < bits.cols; col++) {
                candidateByteList.ptr<cv::Vec4b>(0)[currentByte][0] =
                    candidateByteList.ptr<cv::Vec4b>(0)[currentByte][0] << 1;
                candidateByteList.ptr<cv::Vec4b>(0)[currentByte][1] =
                    candidateByteList.ptr<cv::Vec4b>(0)[currentByte][1] << 1;
                candidateByteList.ptr<cv::Vec4b>(0)[currentByte][2] =
                    candidateByteList.ptr<cv::Vec4b>(0)[currentByte][2] << 1;
                candidateByteList.ptr<cv::Vec4b>(0)[currentByte][3] =
                    candidateByteList.ptr<cv::Vec4b>(0)[currentByte][3] << 1;
                if (bits.at<unsigned char>(row, col))
                    candidateByteList.ptr<cv::Vec4b>(0)[currentByte][0]++;
                if (bits.at<unsigned char>(col, bits.cols - 1 - row))
                    candidateByteList.ptr<cv::Vec4b>(0)[currentByte][1]++;
                if (bits.at<unsigned char>(bits.rows - 1 - row, bits.cols - 1 - col))
                    candidateByteList.ptr<cv::Vec4b>(0)[currentByte][2]++;
                if (bits.at<unsigned char>(bits.rows - 1 - col, row))
                    candidateByteList.ptr<cv::Vec4b>(0)[currentByte][3]++;
                currentBit++;
                if (currentBit == 8) {
                    currentBit = 0;
                    currentByte++;
                }
            }
        }
        return candidateByteList;
    }



    /**
      * @brief Transform list of bytes to matrix of bits
      */
    cv::Mat _getBitsFromByteList(const cv::Mat &byteList) const {
        CV_Assert(byteList.total() >= markerSize*markerSize/8 &&
                  byteList.total() <= markerSize*markerSize/8+1);

        cv::Mat bits(markerSize, markerSize, CV_8UC1, cv::Scalar::all(0));

        unsigned char base2List[] = {128, 64, 32, 16, 8, 4, 2, 1};
        int currentByteIdx = 0;
        // we only need the bytes in normal rotation
        unsigned char currentByte = byteList.ptr<cv::Vec4b>(0)[0][0];
        int currentBit = 0;
        for (int row = 0; row < bits.rows; row++) {
            for (int col = 0; col < bits.cols; col++) {
                if (currentByte >= base2List[currentBit]) {
                    bits.at<unsigned char>(row, col) = 1;
                    currentByte -= base2List[currentBit];
                }
                currentBit++;
                if (currentBit == 8) {
                    currentBit = 0;
                    currentByteIdx++;
                }
            }
        }
        return bits;
    }



    /**
      * @brief Calculate all distances of input byteList to markers in dictionary
      * Returned matrix has one row per dictionary marker and two columns
      * Column 0 is the distance to the candidate, Column 1 is the rotation with minimum distance
      */
    cv::Mat _getDistances(const cv::Mat &byteList) const {

        CV_Assert(bytesList.cols == byteList.total() && byteList.type()== CV_8UC4);

        cv::Mat res(bytesList.rows, 2, CV_32SC1);
        for (unsigned int m = 0; m < bytesList.rows; m++) {
            res.ptr<int>(m)[0] = 10e8;
            for (unsigned int r = 0; r < 4; r++) {
                int currentHamming = 0;
                // for each byte, calculate XOR result and then sum the Hamming weight from the LUT
                for (int b = 0; b < byteList.total(); b++) {
                    unsigned char xorRes =
                        bytesList.ptr<cv::Vec4b>(m)[b][r] ^ byteList.ptr<cv::Vec4b>(0)[b][0];
                    currentHamming += hammingWeightLUT[xorRes];
                }

                if (currentHamming < res.ptr<int>(m)[0]) {
                    res.ptr<int>(m)[0] = currentHamming;
                    res.ptr<int>(m)[1] = r;
                }
            }
        }
        return res;
    }



};



// DictionaryData constructors calls
const DictionaryData DICT_ARUCO_DATA = DictionaryData(&(DICT_ARUCO_BYTES[0][0][0]), 5, 1024, 1);



}
}

#endif // cplusplus
#endif // __OPENCV_ARUCO_DICTIONARY_CPP__
