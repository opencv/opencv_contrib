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

#include "precomp.hpp"
#include "opencv2/aruco/dictionary.hpp"
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
      */
    Dictionary::Dictionary(const unsigned char * bytes, int _markerSize, int dictsize,
                           int _maxcorr) {
        markerSize = _markerSize;
        maxCorrectionBits = _maxcorr;
        int nbytes = (markerSize * markerSize) / 8;
        if ((markerSize * markerSize) % 8 != 0)
            nbytes++;

        // save bytes in internal format
        // bytesList.at<Vec4b>(i, j)[k] is j-th byte of i-th marker, in its k-th rotation
        bytesList = Mat(dictsize, nbytes, CV_8UC4);
        for (int i = 0; i < dictsize; i++) {
            for (int j = 0; j < nbytes; j++) {
                for (int k = 0; k < 4; k++)
                    bytesList.at<Vec4b>(i, j)[k] = bytes[i * (4 * nbytes) + k * nbytes + j];
            }
        }
    }



    /**
     * @brief Given a matrix of bits. Returns whether if marker is identified or not.
     * It returns by reference the correct id (if any) and the correct rotation
     */
    bool Dictionary::identify(const Mat &onlyBits, int &idx, int &rotation,
                              double maxCorrectionRate) const {

        CV_Assert(onlyBits.rows == markerSize && onlyBits.cols == markerSize);

        int maxCorrectionRecalculed = int( double(maxCorrectionBits) * maxCorrectionRate );

        // get as a byte list
        Mat candidateBytes = _getByteListFromBits(onlyBits);

        idx = -1; // by default, not found

        // search closest marker in dict
        for (int m = 0; m < bytesList.rows; m++) {
            int currentMinDistance = markerSize * markerSize + 1;
            int currentRotation = -1;
            for (unsigned int r = 0; r < 4; r++) {
                int currentHamming = 0;
                // for each byte, calculate XOR result and then sum the Hamming weight from the LUT
                for (unsigned int b = 0; b < candidateBytes.total(); b++) {
                    unsigned char xorRes =
                        bytesList.ptr<Vec4b>(m)[b][r] ^ candidateBytes.ptr<Vec4b>(0)[b][0];
                    currentHamming += hammingWeightLUT[xorRes];
                }

                if (currentHamming < currentMinDistance) {
                    currentMinDistance = currentHamming;
                    currentRotation = r;
                }
            }

            // if maxCorrection is fullfilled, return this one
            if (currentMinDistance <= maxCorrectionRecalculed ) {
                idx = m;
                rotation = currentRotation;
                break;
            }
        }

        if (idx != -1)
            return true;
        else
            return false;

    }


    /**
      * Returns the distance of the input bits to the specific id.
      */
    int Dictionary::getDistanceToId(InputArray bits, int id, bool allRotations) const {
        CV_Assert(id >= 0 && id < bytesList.rows);

        Mat candidateBytes = _getByteListFromBits(bits.getMat());
        unsigned int nRotations = 4;
        if(!allRotations) nRotations = 1;
        int currentMinDistance = int(bits.total() * bits.total());
        for (unsigned int r = 0; r < nRotations; r++) {
            int currentHamming = 0;
            for (unsigned int b = 0; b < candidateBytes.total(); b++) {
                unsigned char xorRes =
                    bytesList.ptr<Vec4b>(id)[b][r] ^ candidateBytes.ptr<Vec4b>(0)[b][0];
                currentHamming += hammingWeightLUT[xorRes];
            }

            if (currentHamming < currentMinDistance) {
                currentMinDistance = currentHamming;
            }
        }
        return currentMinDistance;
    }



    /**
     * @brief Draw a canonical marker image
     */
    void Dictionary::drawMarker(int id, int sidePixels, OutputArray _img, int borderBits) const {

        CV_Assert(sidePixels > markerSize);
        CV_Assert(id < bytesList.rows);
        CV_Assert(borderBits > 0);

        _img.create(sidePixels, sidePixels, CV_8UC1);

        // create small marker with 1 pixel per bin
        Mat tinyMarker(markerSize + 2*borderBits, markerSize + 2*borderBits, CV_8UC1,
                       Scalar::all(0));
        Mat innerRegion =
            tinyMarker.rowRange(borderBits, tinyMarker.rows - borderBits).
                       colRange(borderBits, tinyMarker.cols - borderBits);
        // put inner bits
        Mat bits = 255 * _getBitsFromByteList(bytesList.rowRange(id, id + 1));
        CV_Assert(innerRegion.total() == bits.total());
        bits.copyTo(innerRegion);

        // resize tiny marker to output size
        cv::resize(tinyMarker, _img.getMat(), _img.getMat().size(), 0, 0, INTER_NEAREST);
    }




    /**
      * @brief Transform matrix of bits to list of bytes in the 4 rotations
      */
    Mat Dictionary::_getByteListFromBits(const Mat &bits) const {

        int nbytes = (bits.cols * bits.rows) / 8;
        if ((bits.cols * bits.rows) % 8 != 0)
            nbytes++;
        Mat candidateByteList(1, nbytes, CV_8UC4, Scalar::all(0));
        unsigned char currentBit = 0;
        int currentByte = 0;
        for (int row = 0; row < bits.rows; row++) {
            for (int col = 0; col < bits.cols; col++) {
                candidateByteList.ptr<Vec4b>(0)[currentByte][0] =
                    candidateByteList.ptr<Vec4b>(0)[currentByte][0] << 1;
                candidateByteList.ptr<Vec4b>(0)[currentByte][1] =
                    candidateByteList.ptr<Vec4b>(0)[currentByte][1] << 1;
                candidateByteList.ptr<Vec4b>(0)[currentByte][2] =
                    candidateByteList.ptr<Vec4b>(0)[currentByte][2] << 1;
                candidateByteList.ptr<Vec4b>(0)[currentByte][3] =
                    candidateByteList.ptr<Vec4b>(0)[currentByte][3] << 1;
                if (bits.at<unsigned char>(row, col))
                    candidateByteList.ptr<Vec4b>(0)[currentByte][0]++;
                if (bits.at<unsigned char>(col, bits.cols - 1 - row))
                    candidateByteList.ptr<Vec4b>(0)[currentByte][1]++;
                if (bits.at<unsigned char>(bits.rows - 1 - row, bits.cols - 1 - col))
                    candidateByteList.ptr<Vec4b>(0)[currentByte][2]++;
                if (bits.at<unsigned char>(bits.rows - 1 - col, row))
                    candidateByteList.ptr<Vec4b>(0)[currentByte][3]++;
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
    Mat Dictionary::_getBitsFromByteList(const Mat &byteList) const {
        CV_Assert(byteList.total() > 0 && byteList.total() >= (unsigned int)markerSize*markerSize/8
                  && byteList.total() <= (unsigned int)markerSize*markerSize/8+1);
        Mat bits(markerSize, markerSize, CV_8UC1, Scalar::all(0));

        unsigned char base2List[] = {128, 64, 32, 16, 8, 4, 2, 1};
        int currentByteIdx = 0;
        // we only need the bytes in normal rotation
        unsigned char currentByte = byteList.ptr<Vec4b>(0)[0][0];
        int currentBit = 0;
        for (int row = 0; row < bits.rows; row++) {
            for (int col = 0; col < bits.cols; col++) {
                if (currentByte >= base2List[currentBit]) {
                    bits.at<unsigned char>(row, col) = 1;
                    currentByte -= base2List[currentBit];
                }
                currentBit++;
                if (currentBit == 8) {
                    currentByteIdx++;
                    currentByte = byteList.ptr<Vec4b>(0)[currentByteIdx][0];
                    // if not enough bits for one more byte, we are in the end
                    // update bit position accordingly
                    if (8 * (currentByteIdx + 1) > (int)bits.total())
                        currentBit = 8 * (currentByteIdx + 1) - (int)bits.total();
                    else
                        currentBit = 0; // ok, bits enough for next byte
                }
            }
        }
        return bits;
    }





// DictionaryData constructors calls
const Dictionary DICT_ARUCO_DATA = Dictionary(&(DICT_ARUCO_BYTES[0][0][0]), 5, 1024, 1);
const Dictionary DICT_6X6_250_DATA = Dictionary(&(DICT_6X6_250_BYTES[0][0][0]), 6, 250, 5);


const Dictionary & getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME name) {
    switch (name) {
    case DICT_ARUCO:
        return DICT_ARUCO_DATA;
    case DICT_6X6_250:
        return DICT_6X6_250_DATA;
    }
    return DICT_ARUCO_DATA;
}



}
}
