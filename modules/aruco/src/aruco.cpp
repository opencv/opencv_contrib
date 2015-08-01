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
#include "opencv2/aruco.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>


namespace cv {
namespace aruco {

using namespace std;





/**
  *
  */
DetectorParameters::DetectorParameters() : adaptiveThreshWinSize(21),
                                           adaptiveThreshConstant(7),
                                           minMarkerPerimeterRate(0.03),
                                           maxMarkerPerimeterRate(4.),
                                           polygonalApproxAccuracyRate(0.05),
                                           minCornerDistance(10),
                                           minDistanceToBorder(3),
                                           minMarkerDistance(10),
                                           doCornerRefinement(false),
                                           cornerRefinementWinSize(5),
                                           cornerRefinementMaxIterations(30),
                                           cornerRefinementMinAccuracy(0.1),
                                           markerBorderBits(1),
                                           perspectiveRemoveDistortion(false),
                                           perspectiveRemovePixelPerCell(4),
                                           perspectiveRemoveIgnoredMarginPerCell(0.13),
                                           maxErroneousBitsInBorderRate(0.5),
                                           minOtsuStdDev(5.0),
                                           errorCorrectionRate(0.6) {

}


/**
  * @brief Convert input image to gray if it is a 3 channels image
  */
static void _convertToGrey(InputArray _in, OutputArray _out) {

    CV_Assert(_in.getMat().channels() == 1 || _in.getMat().channels() == 3);

    _out.create(_in.getMat().size(), CV_8UC1);
    if (_in.getMat().type() == CV_8UC3)
        cvtColor(_in.getMat(), _out.getMat(), COLOR_BGR2GRAY);
    else
       _in.getMat().copyTo(_out);
}


/**
  * @brief Threshold input image using adaptive thresholding
  */
static void _threshold(InputArray _in, OutputArray _out, int winSize, double constant) {

    CV_Assert(winSize >= 3);

    adaptiveThreshold(_in, _out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, winSize, constant);
}


/**
  * @brief Given a tresholded image, find the contours, calculate their polygonal approximation
  * and take those that accomplish some conditions
  */
static void
_findMarkerContours(InputArray _in, vector<vector<Point2f> > &candidates,
                    vector<vector<Point> > &contoursOut,
                    double minPerimeterRate, double maxPerimeterRate, double accuracyRate,
                    double minCornerDistance, int minDistanceToBorder) {

    CV_Assert(minPerimeterRate > 0 && maxPerimeterRate > 0 && accuracyRate > 0 &&
              minCornerDistance > 0 && minDistanceToBorder >= 0);

    unsigned int minPerimeterPixels = (unsigned int)(minPerimeterRate * max(_in.getMat().cols,
                                                                            _in.getMat().rows));
    unsigned int maxPerimeterPixels = (unsigned int)(maxPerimeterRate * max(_in.getMat().cols,
                                                                            _in.getMat().rows));
    Mat contoursImg;
    _in.getMat().copyTo(contoursImg);
    vector<vector<Point> > contours;
    findContours(contoursImg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // now filter list of contours
    for (unsigned int i = 0; i < contours.size(); i++) {
        // check perimeter
        if (contours[i].size() < minPerimeterPixels || contours[i].size() > maxPerimeterPixels)
            continue;

        // check is square and is convex
        vector<Point> approxCurve;
        approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * accuracyRate, true);
        if (approxCurve.size() != 4 || !isContourConvex(approxCurve))
            continue;

        // check min distance between corners (minimum distance is 10,
        // so minimun square distance is 100)
        double minDistSq = max(contoursImg.cols, contoursImg.rows) *
                           max(contoursImg.cols, contoursImg.rows);
        for (int j = 0; j < 4; j++) {
            double d = (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) *
                       (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                       (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y) *
                       (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y);
            minDistSq = min(minDistSq, d);
        }
        if (minDistSq < minCornerDistance * minCornerDistance)
            continue;

        // check if it is too near to the image border
        bool tooNearBorder = false;
        for (int j = 0; j < 4; j++) {
            if (approxCurve[j].x < minDistanceToBorder || approxCurve[j].y < minDistanceToBorder ||
                approxCurve[j].x > contoursImg.cols - 1 - minDistanceToBorder ||
                approxCurve[j].y > contoursImg.rows - 1 - minDistanceToBorder)
                tooNearBorder = true;
        }
        if (tooNearBorder)
            continue;

        // if it passes all the test, add to candidates vector
        vector<Point2f> currentCandidate;
        currentCandidate.resize(4);
        for (int j = 0; j < 4; j++) {
            currentCandidate[j] = Point2f((float)approxCurve[j].x, (float)approxCurve[j].y);
        }
        candidates.push_back(currentCandidate);
        contoursOut.push_back(contours[i]);
    }
}


/**
  * @brief Assure order of candidate corners is clockwise direction
  */
static void _reorderCandidatesCorners(vector<vector<Point2f> > &candidates,
                                      vector<vector<Point> > &contours) {

    for (unsigned int i = 0; i < candidates.size(); i++) {
        double dx1 = candidates[i][1].x - candidates[i][0].x;
        double dy1 = candidates[i][1].y - candidates[i][0].y;
        double dx2 = candidates[i][2].x - candidates[i][0].x;
        double dy2 = candidates[i][2].y - candidates[i][0].y;
        double crossProduct = (dx1 * dy2) - (dy1 * dx2); // clockwise direction

        if (crossProduct < 0.0) {
            swap(candidates[i][1], candidates[i][3]);
//            reverse(contours.begin(), contours.end());
        }
    }
}


/**
  * @brief Check candidates that are too close to each other and remove the smaller one
  */
static void
_filterTooCloseCandidates(const vector<vector<Point2f> > &candidatesIn,
                          vector<vector<Point2f> > &candidatesOut,
                          const vector<vector<Point> > &contoursIn,
                          vector<vector<Point> > &contoursOut,
                          double minMarkerDistance) {

    CV_Assert(minMarkerDistance > 0);

    vector<pair<int, int> > nearCandidates;
    for (unsigned int i = 0; i < candidatesIn.size(); i++) {
        for (unsigned int j = i + 1; j < candidatesIn.size(); j++) {
            double distSq = 0;
            for (int c = 0; c < 4; c++)
                distSq += (candidatesIn[i][c].x - candidatesIn[j][c].x) *
                              (candidatesIn[i][c].x - candidatesIn[j][c].x) +
                          (candidatesIn[i][c].y - candidatesIn[j][c].y) *
                              (candidatesIn[i][c].y - candidatesIn[j][c].y);
            distSq /= 4.;
            // if mean square distance is lower than 100, remove the smaller one of the two markers
            // (minimum mean distance = 10)
            if (distSq < minMarkerDistance * minMarkerDistance)
                nearCandidates.push_back(pair<int, int>(i, j));
        }
    }

    // mark smaller one in pairs to remove
    vector<bool> toRemove(candidatesIn.size(), false);
    for (unsigned int i = 0; i < nearCandidates.size(); i++) {
        double perimeterSq1 = 0, perimeterSq2 = 0;
        for (unsigned int c = 0; c < 4; c++) {
            // check which one is the smaller and remove it
            perimeterSq1 += (candidatesIn[nearCandidates[i].first][c].x -
                             candidatesIn[nearCandidates[i].first][(c + 1) % 4].x) *
                                (candidatesIn[nearCandidates[i].first][c].x -
                                 candidatesIn[nearCandidates[i].first][(c + 1) % 4].x) +
                            (candidatesIn[nearCandidates[i].first][c].y -
                             candidatesIn[nearCandidates[i].first][(c + 1) % 4].y) *
                                (candidatesIn[nearCandidates[i].first][c].y -
                                 candidatesIn[nearCandidates[i].first][(c + 1) % 4].y);
            perimeterSq2 += (candidatesIn[nearCandidates[i].second][c].x -
                             candidatesIn[nearCandidates[i].second][(c + 1) % 4].x) *
                                (candidatesIn[nearCandidates[i].second][c].x -
                                 candidatesIn[nearCandidates[i].second][(c + 1) % 4].x) +
                            (candidatesIn[nearCandidates[i].second][c].y -
                             candidatesIn[nearCandidates[i].second][(c + 1) % 4].y) *
                                (candidatesIn[nearCandidates[i].second][c].y -
                                 candidatesIn[nearCandidates[i].second][(c + 1) % 4].y);
            if (perimeterSq1 > perimeterSq2)
                toRemove[nearCandidates[i].second] = true;
            else
                toRemove[nearCandidates[i].first] = true;
        }
    }

    // remove extra candidates
    candidatesOut.clear();
    int totalRemaining = 0;
    for (unsigned int i = 0; i < toRemove.size(); i++)
        if (!toRemove[i])
            totalRemaining++;
    candidatesOut.resize(totalRemaining);
    contoursOut.resize(totalRemaining);
    for (unsigned int i = 0, currIdx = 0; i < candidatesIn.size(); i++) {
        if (toRemove[i])
            continue;
        candidatesOut[currIdx] = candidatesIn[i];
        contoursOut[currIdx] = contoursIn[i];
        currIdx++;
    }
}


/**
 * @brief Detect square candidates in the input image
 */
static void
_detectCandidates(InputArray _image, OutputArrayOfArrays _candidates, OutputArrayOfArrays _contours,
                  DetectorParameters params, OutputArray _thresholdedImage = noArray()) {

    Mat image = _image.getMat();
    CV_Assert(image.total() != 0);

    /// 1. CONVERT TO GRAY
    Mat grey;
    _convertToGrey(image, grey);

    /// 2. THRESHOLD
    CV_Assert(params.adaptiveThreshWinSize >= 3);
    Mat thresh;
    _threshold(grey, thresh, params.adaptiveThreshWinSize, params.adaptiveThreshConstant);
    if (_thresholdedImage.needed())
        thresh.copyTo(_thresholdedImage);

    /// 3. DETECT RECTANGLES
    vector<vector<Point2f> > candidates;
    vector<vector<Point> > contours;
    _findMarkerContours(thresh, candidates, contours, params.minMarkerPerimeterRate,
                        params.maxMarkerPerimeterRate, params.polygonalApproxAccuracyRate,
                        params.minCornerDistance, params.minDistanceToBorder);

    /// 4. SORT CORNERS
    _reorderCandidatesCorners(candidates, contours);

    /// 5. FILTER OUT NEAR CANDIDATE PAIRS
    vector<vector<Point2f> > candidatesOut;
    vector<vector<Point> > contoursOut;
    _filterTooCloseCandidates(candidates, candidatesOut, contours, contoursOut,
                              params.minMarkerDistance);


    // parse output
    _candidates.create((int)candidatesOut.size(), 1, CV_32FC2);
    _contours.create((int)contoursOut.size(), 1, CV_32SC2);
    for (int i = 0; i < (int)candidatesOut.size(); i++) {
        _candidates.create(4, 1, CV_32FC2, i, true);
        Mat m = _candidates.getMat(i);
        for (int j = 0; j < 4; j++)
            m.ptr<Vec2f>(0)[j] = candidatesOut[i][j];

        _contours.create(contoursOut[i].size(), 1, CV_32SC2, i, true);
        Mat c = _contours.getMat(i);
        for (int j = 0; j < contoursOut[i].size(); j++)
            c.ptr<Point2i>()[j] = contoursOut[i][j];

    }

}


/**
  * @brief Given an input image and a candidate corners, extract the bits of the candidate, including
  * the border
  */
static Mat
_extractBits(InputArray _image, InputArray _corners, int markerSize, int markerBorderBits,
             int cellSize, double cellMarginRate, double minStdDevOtsu) {

    CV_Assert(_image.getMat().channels() == 1);
    CV_Assert(_corners.total() == 4);
    CV_Assert(markerBorderBits > 0 && cellSize > 0 && cellMarginRate > 0);

    int cellMarginPixels = int(cellMarginRate*cellSize);
    int markerSizeWithBorders = markerSize + 2*markerBorderBits;
    Mat resultImg; // marker image after removing perspective
    int resultImgSize = markerSizeWithBorders * cellSize;
    Mat resultImgCorners(4, 1, CV_32FC2);
    resultImgCorners.ptr<Point2f>(0)[0] = Point2f(0, 0);
    resultImgCorners.ptr<Point2f>(0)[1] = Point2f((float)resultImgSize - 1, 0);
    resultImgCorners.ptr<Point2f>(0)[2] = Point2f((float)resultImgSize - 1,
                                                      (float)resultImgSize - 1);
    resultImgCorners.ptr<Point2f>(0)[3] = Point2f(0, (float)resultImgSize - 1);

    // remove perspective
    Mat transformation = getPerspectiveTransform(_corners, resultImgCorners);
    warpPerspective(_image, resultImg, transformation, Size(resultImgSize, resultImgSize),
                    INTER_NEAREST);

    Mat bits(markerSizeWithBorders, markerSizeWithBorders, CV_8UC1, Scalar::all(0));

    // check if standard deviation is enough to apply Otsu
    // if not enough, it probably means all bits are the same color (black or white)
    Mat mean, stddev;
    // Remove some border just to avoid border noise from perspective transformation
    Mat innerRegion = resultImg.colRange(cellSize/2, resultImg.cols-cellSize/2).
                                rowRange(cellSize/2, resultImg.rows-cellSize/2);
    meanStdDev(innerRegion, mean, stddev);
    if (stddev.ptr<double>(0)[0] < minStdDevOtsu) {
        // all black or all white, anyway it is invalid, return all 0
        if (mean.ptr<double>(0)[0] > 127)
            bits.setTo(1);
        else
            bits.setTo(0);
        return bits;
    }

    // now extract code
    threshold(resultImg, resultImg, 125, 255, THRESH_BINARY | THRESH_OTSU);

    for (int y = 0; y < markerSizeWithBorders; y++) {
        for (int x = 0; x < markerSizeWithBorders; x++) {
            int Xstart = x * (cellSize)+cellMarginPixels;
            int Ystart = y * (cellSize)+cellMarginPixels;
            Mat square = resultImg(Rect(Xstart, Ystart,
                                        cellSize - 2*cellMarginPixels,
                                        cellSize - 2*cellMarginPixels));
            unsigned int nZ = countNonZero(square);
            if (nZ > square.total() / 2)
                bits.at<unsigned char>(y, x) = 1;
        }
    }

    return bits;
}



/**
  * @brief Given an input image and a candidate corners, extract the bits of the candidate, including
  * the border
  * This function version consider marker contour to try a better bits extraction in the presence
  * of distortion.
  * @note TO BE TESTED
  */
static Mat
_extractBitsDistortion(InputArray _image, InputArray _corners, InputArray _contour, int markerSize,
                       int markerBorderBits, int cellSize, double cellMarginRate,
                       double minStdDevOtsu) {

    CV_Assert(_image.getMat().channels() == 1);
    CV_Assert(_corners.total() == 4);
    CV_Assert(markerBorderBits > 0 && cellSize > 0 && cellMarginRate > 0);



    // Find marker corners in contour
    // Transform corners to Point2i to find them easily
    Point cornersInt[4];
    for(int c=0; c<4; c++)
        cornersInt[c] = Point(_corners.getMat().ptr<Point2f>()[c].x,
                              _corners.getMat().ptr<Point2f>()[c].y);

    // This stores the indexes of each corner in the contour vector
    int cornerInContourIdxs[4];
    for(int c=0; c<4; c++)
        cornerInContourIdxs[c] = -1; // default -1

    // look for each corner and store position in cornerInContourIdxs
    for(int i=0; i<_contour.total(); i++) {
        for(int c=0; c<4; c++) {
            if(cornerInContourIdxs[c]==-1 && _contour.getMat().ptr<Point>()[i] == cornersInt[c]) {
                cornerInContourIdxs[c] = i;
                break;
            }

        }
    }

    // Are contour in clockwise or anticlockwise direction ?
    // corners are in clockwise for sure, so check direction of each step of found corners
    int dirFreq[2];
    dirFreq[0] = dirFreq[1] = 0;
    for(int i=0; i<4; i++) {
        if(cornerInContourIdxs[i] < cornerInContourIdxs[(i+1)%4])
            dirFreq[0]++;
        else
            dirFreq[1]++;
    }

    // if more steps in clockwise, direction is clockwise (+1), elsewere is anticlockwise (-1)
    int incr = +1;
    if(dirFreq[1] > dirFreq[0]) incr = -1;

    // calculate perspective transformation from four corners
    int markerSizeWithBorders = markerSize + 2*markerBorderBits;
    Mat resultImgCorners(4, 1, CV_32FC2);
    resultImgCorners.ptr<Point2f>(0)[0] = Point2f(0, 0);
    resultImgCorners.ptr<Point2f>(0)[1] = Point2f((float)markerSizeWithBorders, 0);
    resultImgCorners.ptr<Point2f>(0)[2] = Point2f((float)markerSizeWithBorders,
                                                  (float)markerSizeWithBorders);
    resultImgCorners.ptr<Point2f>(0)[3] = Point2f(0, (float)markerSizeWithBorders);
    Mat transformation = getPerspectiveTransform(_corners, resultImgCorners);

    // Transform contour
    Mat contour2f;
    _contour.getMat().convertTo(contour2f, CV_32F);
    vector<Point2f> tContour;
    perspectiveTransform(contour2f, tContour, transformation);

    // calculate error of each side (top, right, bottom and left) for each bit transition
    // error is separation of contour from ideal rect line of perspective transformation
    vector<double> fErr[4];
    for(int k=0; k<4; k++) {
        fErr[k].resize(markerSizeWithBorders+1, 0); // error on each bit transition
        // on the corners, the error is 0
        fErr[k][0] = 0.;
        fErr[k][markerSizeWithBorders] = 0;
        int currBit = 1; // current bit transition
        // indexes of corners in contours vector
        int startIdx = cornerInContourIdxs[k];
        int endIdx = cornerInContourIdxs[(k+1)%4];
        float lastValueErr = 0;
        // while doing this side of contour
        for(int i=startIdx; i!=endIdx; i=(i+tContour.size()+incr)%tContour.size()) {
            // valueError is the error respect to the rectLine
            // valueInterval is the other corrdinate
            float valueInterval, valueErr;
            if(k==0 || k==2) { // top and bottom
                valueInterval = tContour[i].x;
                if(k==0)
                    valueErr = tContour[i].y;
                else
                    valueErr = tContour[i].y - markerSizeWithBorders;
            }
            else { // right and left
                valueInterval = tContour[i].y;
                if(k==3)
                    valueErr = tContour[i].x;
                else
                    valueErr = tContour[i].x - markerSizeWithBorders;
            }
            // if there is a bit transition jump, we have a new value to store
            if(valueInterval > currBit) {
                // error is mean between this and last error
                fErr[k][currBit] = valueErr*0.5 + lastValueErr*0.5;
                currBit++; // increase, now look for next bit transition
                if(currBit == (markerSizeWithBorders+1)) // if no more bits, finish this side
                    break;
            }
            lastValueErr = valueErr; // store last error

        }
    }

   // corrPnts is a vector of each of the grid points corrected with the values of fErr
   vector<Point2f> corrPnts;
   corrPnts.resize((markerSizeWithBorders+1)*(markerSizeWithBorders+1));
   for(int i=0; i<markerSizeWithBorders+1; i++) {
       // this is the increment on each direction
       float incrX = (fErr[1][i] - fErr[3][i])/float(markerSizeWithBorders);
       float incrY = (fErr[2][i] - fErr[0][i])/float(markerSizeWithBorders);

       // correct x values
       for(int x=0; x<markerSizeWithBorders+1; x++) {
           int idx = i*(markerSizeWithBorders+1) + x;
           corrPnts[idx].x = float(x) + fErr[3][i] + float(x)*(incrX);
       }

       // correct y values
       for(int y=0; y<markerSizeWithBorders+1; y++) {
           int idx = y*(markerSizeWithBorders+1) + i;
           corrPnts[idx].y = float(y) + fErr[0][i] + float(y)*(incrY);
       }

   }

   // convert corrected points to original image in corrPntsOrig
   vector<Point2f> corrPntsOrig;
   perspectiveTransform(corrPnts, corrPntsOrig, transformation.inv());


//   // show grid in original image
//   Mat bb = _image.getMat().clone();
//   Mat aa;
//   cvtColor(bb,aa,COLOR_GRAY2BGR);
//   for(int y=0; y<markerSizeWithBorders+1; y++) {
//       for(int x=0; x<markerSizeWithBorders; x++) {
//           int idx = (markerSizeWithBorders+1)*y + x;
//           line(aa, corrPntsOrig[idx], corrPntsOrig[idx+1], Scalar(255,x*40,0), 1);
//       }
//   }
//   for(int x=0; x<markerSizeWithBorders+1; x++) {
//        for(int y=0; y<markerSizeWithBorders; y++) {
//           int idx1 = (markerSizeWithBorders+1)*y + x;
//           int idx2 = (markerSizeWithBorders+1)*(y+1) + x;
//           line(aa, corrPntsOrig[idx1], corrPntsOrig[idx2], Scalar(255,0,y*40), 1);
//       }
//   }
//   imshow("aaa", aa); waitKey(0);


   // create final image with corrected grid
   Mat resultImg(markerSizeWithBorders*cellSize, markerSizeWithBorders*cellSize, CV_8UC1,
                 Scalar::all(0));

   // for each cell, calculate transformation and include it in resultImg
    vector<Point2f> pointsOut(4);
    pointsOut[0] = Point2f(0,0);
    pointsOut[1] = Point2f(cellSize,0);
    pointsOut[2] = Point2f(cellSize,cellSize);
    pointsOut[3] = Point2f(0,cellSize);
    for(int y=0; y<markerSizeWithBorders; y++) {
        for(int x=0; x<markerSizeWithBorders; x++) {
            vector<Point2f> pointsIn(4);
            pointsIn[0] = corrPntsOrig[y*(markerSizeWithBorders+1) + x];
            pointsIn[1] = corrPntsOrig[y*(markerSizeWithBorders+1) + x+1];
            pointsIn[2] = corrPntsOrig[(y+1)*(markerSizeWithBorders+1) + x+1];
            pointsIn[3] = corrPntsOrig[(y+1)*(markerSizeWithBorders+1) + x];
            Mat t = getPerspectiveTransform(pointsIn, pointsOut);
            Mat outImg;
            warpPerspective(_image, outImg, t, Size(cellSize, cellSize), INTER_NEAREST);
            Mat crop = resultImg.rowRange(y*cellSize, (y+1)*cellSize).
                                     colRange(x*cellSize, (x+1)*cellSize);

            outImg.copyTo(crop);

        }
    }

    // now, calculate bits
    Mat bits(markerSizeWithBorders, markerSizeWithBorders, CV_8UC1, Scalar::all(0));
    int cellMarginPixels = int(cellMarginRate*cellSize);

        // check if standard deviation is enough to apply Otsu
        // if not enough, it probably means all bits are the same color (black or white)
        Mat mean, stddev;
        // Remove some border just to avoid border noise from perspective transformation
        Mat innerRegion = resultImg.colRange(cellSize/2, resultImg.cols-cellSize/2).
                                         rowRange(cellSize/2, resultImg.rows-cellSize/2);
        meanStdDev(innerRegion, mean, stddev);
        if (stddev.ptr<double>(0)[0] < minStdDevOtsu) {
            // all black or all white, anyway it is invalid, return all 0
            if (mean.ptr<double>(0)[0] > 127)
                bits.setTo(1);
            else
                bits.setTo(0);
            return bits;
        }

        // now extract code
        threshold(resultImg, resultImg, 125, 255, THRESH_BINARY | THRESH_OTSU);

        for (int y = 0; y < markerSizeWithBorders; y++) {
            for (int x = 0; x < markerSizeWithBorders; x++) {
                int Xstart = x * (cellSize)+cellMarginPixels;
                int Ystart = y * (cellSize)+cellMarginPixels;
                Mat square =
                    resultImg(Rect(Xstart, Ystart,
                                   cellSize - 2*cellMarginPixels,
                                   cellSize - 2*cellMarginPixels));
                unsigned int nZ = countNonZero(square);
                if (nZ > square.total() / 2)
                    bits.at<unsigned char>(y, x) = 1;
            }
        }


    return bits;

}



/**
  * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
  */
static int _getBorderErrors(const Mat &bits, int markerSize, int borderSize) {

    int sizeWithBorders = markerSize + 2*borderSize;

    CV_Assert(markerSize > 0 && bits.cols == sizeWithBorders && bits.rows == sizeWithBorders);

    int totalErrors = 0;
    for (int y = 0; y < sizeWithBorders; y++) {
        for (int k = 0; k < borderSize; k++) {
            if (bits.ptr<unsigned char>(y)[k] != 0)
                totalErrors++;
            if (bits.ptr<unsigned char>(y)[sizeWithBorders - 1 - k] != 0)
                totalErrors++;
        }
    }
    for (int x = borderSize; x < sizeWithBorders - borderSize; x++) {
        for (int k = 0; k < borderSize; k++) {
            if (bits.ptr<unsigned char>(k)[x] != 0)
                totalErrors++;
            if (bits.ptr<unsigned char>(sizeWithBorders - 1 - k)[x] != 0)
                totalErrors++;
        }
    }
    return totalErrors;
}


/**
 * @brief Tries to identify one candidate given the dictionary
 */
static bool
_identifyOneCandidate(Dictionary dictionary, InputArray _image, InputOutputArray _corners,
                      InputArray _contour, int &idx, DetectorParameters params) {

    CV_Assert(_corners.total() == 4);
    CV_Assert(_image.getMat().cols != 0 && _image.getMat().rows);
    CV_Assert(params.markerBorderBits > 0);

    // get bits
    Mat candidateBits;
    if(!params.perspectiveRemoveDistortion) {
        candidateBits = _extractBits(_image, _corners, dictionary.markerSize,
                                     params.markerBorderBits,
                                     params.perspectiveRemovePixelPerCell,
                                     params.perspectiveRemoveIgnoredMarginPerCell,
                                     params.minOtsuStdDev);
    }
    else {
        candidateBits = _extractBitsDistortion(_image, _corners, _contour,
                                               dictionary.markerSize,
                                               params.markerBorderBits,
                                               params.perspectiveRemovePixelPerCell,
                                               params.perspectiveRemoveIgnoredMarginPerCell,
                                               params.minOtsuStdDev);
    }

    int maximumErrorsInBorder = int( dictionary.markerSize * dictionary.markerSize *
                                params.maxErroneousBitsInBorderRate );
    int borderErrors = _getBorderErrors(candidateBits, dictionary.markerSize,
                                        params.markerBorderBits);
    if (borderErrors > maximumErrorsInBorder)
        return false; // border is wrong
    Mat onlyBits = candidateBits.rowRange(params.markerBorderBits,
                                          candidateBits.rows - params.markerBorderBits)
                                .colRange(params.markerBorderBits,
                                          candidateBits.rows - params.markerBorderBits);

    int rotation;
    if (!dictionary.identify(onlyBits, idx, rotation, params.errorCorrectionRate))
        return false;
    else {
        if (rotation != 0) {
            Mat copyPoints = _corners.getMat().clone();
            for (int j = 0; j < 4; j++)
                _corners.getMat().ptr<Point2f>(0)[j] =
                    copyPoints.ptr<Point2f>(0)[(j + 4 - rotation) % 4];
        }
        return true;
    }
}



/**
 * @brief Identify square candidates according to a marker dictionary
 */
static void
_identifyCandidates(InputArray _image, InputArrayOfArrays _candidates, InputArrayOfArrays _contours,
                    const Dictionary &dictionary, OutputArrayOfArrays _accepted,
                    OutputArray _ids, DetectorParameters params,
                    OutputArrayOfArrays _rejected = noArray()) {

    int ncandidates = (int)_candidates.total();

    vector<Mat> accepted;
    vector<Mat> rejected;
    vector<int> ids;

    CV_Assert(_image.getMat().total() != 0);

    Mat grey;
    _convertToGrey(_image.getMat(), grey);

    // try to identify each candidate
    for (int i = 0; i < ncandidates; i++) {
        int currId;
        Mat currentCandidate = _candidates.getMat(i);
        Mat currentContour = _contours.getMat(i);
        if (_identifyOneCandidate(dictionary, grey, currentCandidate, currentContour, currId,
                                  params)) {
            accepted.push_back(currentCandidate);
            ids.push_back(currId);
        } else
            rejected.push_back(_candidates.getMat(i));
    }

    // parse output
    _accepted.create((int)accepted.size(), 1, CV_32FC2);
    for (unsigned int i = 0; i < accepted.size(); i++) {
        _accepted.create(4, 1, CV_32FC2, i, true);
        Mat m = _accepted.getMat(i);
        accepted[i].copyTo(m);
    }

    _ids.create((int)ids.size(), 1, CV_32SC1);
    for (unsigned int i = 0; i < ids.size(); i++)
        _ids.getMat().ptr<int>(0)[i] = ids[i];

    if (_rejected.needed()) {
        _rejected.create((int)rejected.size(), 1, CV_32FC2);
        for (unsigned int i = 0; i < rejected.size(); i++) {
            _rejected.create(4, 1, CV_32FC2, i, true);
            Mat m = _rejected.getMat(i);
            rejected[i].copyTo(m);
        }
    }
}


/**
  * @brief Final filter of markers after its identification
  */
static void _filterDetectedMarkers(InputArrayOfArrays _inCorners, InputArray _inIds,
                                   OutputArrayOfArrays _outCorners, OutputArray _outIds) {

    CV_Assert(_inCorners.total() == _inIds.total());
    if(_inCorners.total() == 0)
        return;

    vector<bool> toRemove(_inCorners.total(), false);
    bool atLeastOneRemove = false;

    for (unsigned int i=0; i<_inCorners.total()-1; i++) {
        for(unsigned int j=i+1; j<_inCorners.total(); j++) {
            if ( _inIds.getMat().ptr<int>(0)[i] != _inIds.getMat().ptr<int>(0)[j] )
                continue;

            // check if first marker is inside second
            bool inside = true;
            for(unsigned int p=0; p<4; p++) {
                Point2f point = _inCorners.getMat(j).ptr<Point2f>(0)[p];
                if (pointPolygonTest(_inCorners.getMat(i), point, false) < 0 ) {
                    inside = false;
                    break;
                }
            }
            if(inside) {
                toRemove[j] = true;
                atLeastOneRemove = true;
                continue;
            }

            // check the second marker
            inside = true;
            for(unsigned int p=0; p<4; p++) {
                Point2f point = _inCorners.getMat(i).ptr<Point2f>(0)[p];
                if (pointPolygonTest(_inCorners.getMat(j), point, false) < 0 ) {
                    inside = false;
                    break;
                }
            }
            if(inside) {
                toRemove[i] = true;
                atLeastOneRemove = true;
                continue;
            }


        }
    }

    // parse output
    if (atLeastOneRemove) {
        vector<Mat> filteredCorners;
        vector<int> filteredIds;

        for(unsigned int i=0; i<toRemove.size(); i++) {
            if(!toRemove[i]) {
                filteredCorners.push_back( _inCorners.getMat(i).clone() );
                filteredIds.push_back( _inIds.getMat().ptr<int>(0)[i] );
            }
        }

        _outIds.create((int)filteredIds.size(), 1, CV_32SC1);
        for (unsigned int i = 0; i < filteredIds.size(); i++)
            _outIds.getMat().ptr<int>(0)[i] = filteredIds[i];

        _outCorners.create((int)filteredCorners.size(), 1, CV_32FC2);
        for (unsigned int i = 0; i < filteredCorners.size(); i++) {
            _outCorners.create(4, 1, CV_32FC2, i, true);
            filteredCorners[i].copyTo(_outCorners.getMat(i));
        }

    }




}



/**
  * @brief Return object points for the system centered in a single marker, given the marker length
  */
static void _getSingleMarkerObjectPoints(float markerLength, OutputArray _objPoints) {

    CV_Assert(markerLength > 0);

    _objPoints.create(4, 1, CV_32FC3);
    Mat objPoints = _objPoints.getMat();
    objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
    objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);
}



/**
  */
//static const DictionaryData &_getDictionaryData(DICTIONARY name) {
//    switch (name) {
//    case DICT_ARUCO:
//        return DICT_ARUCO_DATA;
//    case DICT_6X6_250:
//        return DICT_6X6_250_DATA;
//    }
//    return DICT_ARUCO_DATA;
//}



/**
  */
void detectMarkers(InputArray _image, Dictionary dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, DetectorParameters params,
                   OutputArrayOfArrays _rejectedImgPoints) {

    CV_Assert(_image.getMat().total() != 0);

    Mat grey;
    _convertToGrey(_image.getMat(), grey);

    /// STEP 1: Detect marker candidates
    vector<vector<Point2f> > candidates;
    vector<vector<Point> > contours;
    _detectCandidates(grey, candidates, contours, params);

    /// STEP 2: Check candidate codification (identify markers)
    _identifyCandidates(grey, candidates, contours, dictionary, _corners, _ids, params,
                        _rejectedImgPoints);

    /// STEP 3: Filter detected markers;
    _filterDetectedMarkers(_corners, _ids, _corners, _ids);

    /// STEP 4: Corner refinement
    if(params.doCornerRefinement) {
        CV_Assert(params.cornerRefinementWinSize > 0 && params.cornerRefinementMaxIterations > 0 &&
                  params.cornerRefinementMinAccuracy > 0);
        for (unsigned int i = 0; i < _corners.total(); i++) {
            cornerSubPix(grey, _corners.getMat(i),
                         cvSize(params.cornerRefinementWinSize, params.cornerRefinementWinSize),
                         cvSize(-1, -1),
                         cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                        params.cornerRefinementMaxIterations,
                                        params.cornerRefinementMinAccuracy));
        }
    }
}



/**
  */
void estimatePoseSingleMarkers(InputArrayOfArrays _corners, float markerLength,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs) {

    CV_Assert(markerLength > 0);

    Mat markerObjPoints;
    _getSingleMarkerObjectPoints(markerLength, markerObjPoints);
    int nMarkers = (int)_corners.total();
    _rvecs.create(nMarkers, 1, CV_32FC1);
    _tvecs.create(nMarkers, 1, CV_32FC1);

    // for each marker, calculate its pose
    for (int i = 0; i < nMarkers; i++) {
        _rvecs.create(3, 1, CV_64FC1, i, true);
        _tvecs.create(3, 1, CV_64FC1, i, true);
        solvePnP(markerObjPoints, _corners.getMat(i), _cameraMatrix, _distCoeffs,
                 _rvecs.getMat(i), _tvecs.getMat(i));
    }
}



/**
  * @brief Given a board configuration and a set of detected markers, returns the corresponding
  * image points and object points to call solvePnP
  */
static void
_getBoardObjectAndImagePoints(const Board &board, InputArray _detectedIds,
                              InputArrayOfArrays _detectedCorners,
                              OutputArray _imgPoints, OutputArray _objPoints) {

    CV_Assert(board.ids.size()==board.objPoints.size());
    CV_Assert(_detectedIds.total() == _detectedCorners.total() );

    int nDetectedMarkets = (int)_detectedIds.total();

    vector<Point3f> objPnts;
    objPnts.reserve(nDetectedMarkets);

    vector<Point2f> imgPnts;
    imgPnts.reserve(nDetectedMarkets);

    // look for detected markers that belong to the board and get their information
    for (int i = 0; i < nDetectedMarkets; i++) {
        int currentId = _detectedIds.getMat().ptr<int>(0)[i];
        for (unsigned int j = 0; j < board.ids.size(); j++) {
            if (currentId == board.ids[j]) {
                for (int p = 0; p < 4; p++) {
                    objPnts.push_back(board.objPoints[j][p]);
                    imgPnts.push_back(_detectedCorners.getMat(i).ptr<Point2f>(0)[p]);
                }
            }
        }
    }

    // create output
    _objPoints.create((int)objPnts.size(), 1, CV_32FC3);
    for (unsigned int i = 0; i < objPnts.size(); i++)
        _objPoints.getMat().ptr<Point3f>(0)[i] = objPnts[i];

    _imgPoints.create((int)objPnts.size(), 1, CV_32FC2);
    for (unsigned int i = 0; i < imgPnts.size(); i++)
        _imgPoints.getMat().ptr<Point2f>(0)[i] = imgPnts[i];
}



/**
  * Estimate pose and project board markers that are not included in the list of detected markers
  */
static void
_projectUndetectedMarkers(const Board &board, InputOutputArrayOfArrays _detectedCorners,
                          InputOutputArray _detectedIds,
                          InputArray _cameraMatrix, InputArray _distCoeffs,
                          OutputArrayOfArrays _undetectedMarkersProjectedCorners,
                          OutputArray _undetectedMarkersIds) {

    Mat rvec, tvec;
    int boardDetectedMarkers;
    boardDetectedMarkers = aruco::estimatePoseBoard(_detectedCorners, _detectedIds, board,
                                                        _cameraMatrix, _distCoeffs, rvec, tvec);

    // at least one marker from board so rvec and tvec are valids
    if (boardDetectedMarkers == 0)
        return;

    vector< vector<Point2f> > undetectedCorners;
    vector< int > undetectedIds;
    for(unsigned int i=0; i<board.ids.size(); i++) {
        int foundIdx = -1;
        for(unsigned int j=0; j<_detectedIds.total(); j++) {
            if(board.ids[i] == _detectedIds.getMat().ptr<int>()[j]) {
                foundIdx = j;
                break;
            }
        }

        // not detected
        if(foundIdx == -1) {
            undetectedCorners.push_back( vector<Point2f>() );
            undetectedIds.push_back(board.ids[i]);
            projectPoints(board.objPoints[i], rvec, tvec, _cameraMatrix, _distCoeffs,
                          undetectedCorners.back());
        }

    }


    // parse output
    _undetectedMarkersIds.create((int)undetectedIds.size(), 1, CV_32SC1);
    for (unsigned int i = 0; i < undetectedIds.size(); i++)
        _undetectedMarkersIds.getMat().ptr<int>(0)[i] = undetectedIds[i];

    _undetectedMarkersProjectedCorners.create((int)undetectedCorners.size(), 1, CV_32FC2);
    for (unsigned int i = 0; i < undetectedCorners.size(); i++) {
        _undetectedMarkersProjectedCorners.create(4, 1, CV_32FC2, i, true);
        for(int j=0; j<4; j++) {
            _undetectedMarkersProjectedCorners.getMat(i).ptr<Point2f>()[j] =
                                                         undetectedCorners[i][j];

        }

    }

}



/**
  * Interpolate board markers that are not included in the list of detected markers using
  * global homography
  */
static void
_projectUndetectedMarkers(const Board &board, InputOutputArrayOfArrays _detectedCorners,
                          InputOutputArray _detectedIds,
                          OutputArrayOfArrays _undetectedMarkersProjectedCorners,
                          OutputArray _undetectedMarkersIds) {


    // check board points are in the same plane, if not, global homography cannot be applied
    CV_Assert(board.objPoints.size() > 0);
    CV_Assert(board.objPoints[0].size() > 0);
    float boardZ = board.objPoints[0][0].z;
    for(unsigned int i=0; i<board.objPoints.size(); i++) {
        for(unsigned int j=0; j<board.objPoints[i].size(); j++) {
            CV_Assert(boardZ == board.objPoints[i][j].z);
        }
    }

    // find markers included in board, and missing markers from board
    vector<Point2f> markerCornersAllObj2D, markerCornersAll;
    vector< vector<Point2f> > undetectedMarkersObj2D;
    vector< int > undetectedMarkersIds;
    for(unsigned int j=0; j<board.ids.size(); j++) {
        bool found = false;
        for(unsigned int i=0; i<_detectedIds.total(); i++) {
            if(_detectedIds.getMat().ptr<int>()[i] == board.ids[j]) {
                for(int c=0; c<4; c++) {
                    markerCornersAll.push_back(_detectedCorners.getMat(i).ptr<Point2f>()[c]);
                    markerCornersAllObj2D.push_back( Point2f(board.objPoints[j][c].x,
                                                             board.objPoints[j][c].y));
                }
                found = true;
                break;
            }
        }
        if(!found) {
            undetectedMarkersObj2D.push_back( vector<Point2f>() );
            for(int c=0; c<4; c++) {
                undetectedMarkersObj2D.back().push_back( Point2f(board.objPoints[j][c].x,
                                                                 board.objPoints[j][c].y));
            }
            undetectedMarkersIds.push_back(board.ids[j]);
        }
    }
    if(markerCornersAll.size() == 0)
        return;

    Mat transformation = findHomography(markerCornersAllObj2D, markerCornersAll);

    _undetectedMarkersProjectedCorners.create((int)undetectedMarkersIds.size(), 1, CV_32FC2);

    // for each undetected marker, apply transformation
    for(unsigned int i=0; i<undetectedMarkersObj2D.size(); i++) {
        Mat projectedMarker;
        perspectiveTransform(undetectedMarkersObj2D[i], projectedMarker, transformation);

        _undetectedMarkersProjectedCorners.create(4, 1, CV_32FC2, i, true);
        projectedMarker.copyTo(_undetectedMarkersProjectedCorners.getMat(i));

    }

    _undetectedMarkersIds.create((int)undetectedMarkersIds.size(), 1, CV_32SC1);
    for (unsigned int i = 0; i < undetectedMarkersIds.size(); i++)
        _undetectedMarkersIds.getMat().ptr<int>(0)[i] = undetectedMarkersIds[i];

}



/**
  */
void refineDetectedMarkers(InputArray _image, const Board &board,
                           InputOutputArrayOfArrays _detectedCorners, InputOutputArray _detectedIds,
                           InputOutputArray _rejectedCorners, InputArray _cameraMatrix,
                           InputArray _distCoeffs, float minRepDistance, float errorCorrectionRate,
                           OutputArray _recoveredIdxs, DetectorParameters params) {

    if(_detectedIds.total() == 0)
        return;

    vector< vector<Point2f> > undetectedMarkersCorners;
    vector<int> undetectedMarkersIds;
    if(_cameraMatrix.total() != 0) {
        // reproject based on camera projection model
        _projectUndetectedMarkers(board, _detectedCorners, _detectedIds, _cameraMatrix, _distCoeffs,
                                  undetectedMarkersCorners, undetectedMarkersIds);

    }
    else {
        // reproject based on global homography
        _projectUndetectedMarkers(board, _detectedCorners, _detectedIds,
                                  undetectedMarkersCorners, undetectedMarkersIds);

    }

    vector<bool> alreadyIdentified(_rejectedCorners.total(), false);

    int maxCorrectionRecalculed = int( double(board.dictionary.maxCorrectionBits) *
                                       errorCorrectionRate );
    Mat grey;
    _convertToGrey(_image, grey);

    vector<Mat> finalAcceptedCorners;
    vector<int> finalAcceptedIds;
    finalAcceptedCorners.resize(_detectedCorners.total());
    finalAcceptedIds.resize(_detectedIds.total());
    for(unsigned int i=0; i<_detectedIds.total(); i++) {
        finalAcceptedCorners[i] = _detectedCorners.getMat(i).clone();
        finalAcceptedIds[i] = _detectedIds.getMat().ptr<int>()[i];
    }

    vector<int> recoveredIdxs; // origin ids of accepted markers in _rejectedCorners

    for (unsigned int i=0; i < undetectedMarkersIds.size(); i++) {
        for(unsigned int j=0; j<_rejectedCorners.total(); j++) {
            if (alreadyIdentified[j])
                continue;

            // check distance
            double minDistance = minRepDistance + 1;
            bool valid = false;
            int validRot = 0;
            for(int c=0; c<4; c++) { // first corner in rejected candidate
                double currentMaxDistance = 0;
                for(int k=0; k<4; k++) {
                    Point2f rejCorner = _rejectedCorners.getMat(j).ptr<Point2f>()[(c+k)%4];
                    double cornerDist = norm(undetectedMarkersCorners[i][k] - rejCorner);
                    currentMaxDistance = max(currentMaxDistance, cornerDist);
                }
                if(currentMaxDistance < minRepDistance && currentMaxDistance < minDistance ) {
                    valid = true;
                    validRot = c;
                    minDistance = currentMaxDistance;
                }
            }

            if(!valid)
                continue;

            // apply rotation before extract bits
            Mat rotatedMarker = Mat(4, 1, CV_32FC2);
            for (int c=0; c < 4; c++)
                rotatedMarker.ptr<Point2f>()[c] =
                        _rejectedCorners.getMat(j).ptr<Point2f>()[(c + 4 + validRot) % 4];



            int codeDistance=0;
            if(errorCorrectionRate >=0 ) {

                // extract bits
                Mat bits = _extractBits(grey, rotatedMarker, board.dictionary.markerSize,
                                        params.markerBorderBits,
                                        params.perspectiveRemovePixelPerCell,
                                        params.perspectiveRemoveIgnoredMarginPerCell,
                                        params.minOtsuStdDev);

                Mat onlyBits =
                    bits.rowRange(params.markerBorderBits, bits.rows - params.markerBorderBits)
                        .colRange(params.markerBorderBits, bits.rows - params.markerBorderBits);

                codeDistance = board.dictionary.getDistanceToId(onlyBits, undetectedMarkersIds[i],
                                                                false);
            }

            if(errorCorrectionRate<0 || codeDistance <= maxCorrectionRecalculed) {
                // subpixel refinement
                if(params.doCornerRefinement) {
                    CV_Assert(params.cornerRefinementWinSize > 0 &&
                              params.cornerRefinementMaxIterations > 0 &&
                              params.cornerRefinementMinAccuracy > 0);
                    cornerSubPix(grey, rotatedMarker,
                                 cvSize(params.cornerRefinementWinSize,
                                        params.cornerRefinementWinSize),
                                 cvSize(-1, -1),
                                 cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                                params.cornerRefinementMaxIterations,
                                                params.cornerRefinementMinAccuracy));
                }

                // remove from rejected
                alreadyIdentified[j] = true;

                // add to detected
                finalAcceptedCorners.push_back(rotatedMarker);
                finalAcceptedIds.push_back(undetectedMarkersIds[i]);
                recoveredIdxs.push_back(j);

            }

        }
    }



    // parse output
    if(finalAcceptedIds.size() != _detectedIds.total()) {
        _detectedCorners.clear();
        _detectedIds.clear();

        // parse output
        _detectedIds.create((int)finalAcceptedIds.size(), 1, CV_32SC1);
        for (unsigned int i = 0; i < finalAcceptedIds.size(); i++)
            _detectedIds.getMat().ptr<int>(0)[i] = finalAcceptedIds[i];

        _detectedCorners.create((int)finalAcceptedCorners.size(), 1, CV_32FC2);
        for (unsigned int i = 0; i < finalAcceptedCorners.size(); i++) {
            _detectedCorners.create(4, 1, CV_32FC2, i, true);
            for(int j=0; j<4; j++) {
                _detectedCorners.getMat(i).ptr<Point2f>()[j] =
                        finalAcceptedCorners[i].ptr<Point2f>()[j];

            }
        }

        // recalculate _rejectedCorners based on alreadyIdentified
        vector<Mat> finalRejected;
        for(unsigned int i=0; i < alreadyIdentified.size(); i++) {
            if(!alreadyIdentified[i]) {
                finalRejected.push_back(_rejectedCorners.getMat(i).clone());
            }
        }

        _rejectedCorners.clear();
        _rejectedCorners.create((int)finalRejected.size(), 1, CV_32FC2);
        for (unsigned int i = 0; i < finalRejected.size(); i++) {
            _rejectedCorners.create(4, 1, CV_32FC2, i, true);
            for(int j=0; j<4; j++) {
                _rejectedCorners.getMat(i).ptr<Point2f>()[j] =
                        finalRejected[i].ptr<Point2f>()[j];

            }
        }

        if (_recoveredIdxs.needed()) {
            _recoveredIdxs.create((int)recoveredIdxs.size(), 1, CV_32SC1 );
            for (unsigned int i=0; i< recoveredIdxs.size(); i++) {
                _recoveredIdxs.getMat().ptr<int>()[i] = recoveredIdxs[i];
            }
        }

    }

}





/**
  */
int estimatePoseBoard(InputArrayOfArrays _corners, InputArray _ids, const Board &board,
                      InputArray _cameraMatrix, InputArray _distCoeffs, OutputArray _rvec,
                      OutputArray _tvec) {

    CV_Assert(_corners.total() == _ids.total() );

    Mat objPoints, imgPoints;
    _getBoardObjectAndImagePoints(board, _ids, _corners, imgPoints, objPoints);

    CV_Assert(imgPoints.total() == objPoints.total());

    if (objPoints.total() == 0) // 0 of the detected markers in board
        return 0;

    _rvec.create(3, 1, CV_64FC1);
    _tvec.create(3, 1, CV_64FC1);
    solvePnP(objPoints, imgPoints, _cameraMatrix, _distCoeffs, _rvec, _tvec);

    return (int)objPoints.total()/4;
}




/**
 */
void GridBoard::draw(Size outSize, OutputArray _img, int marginSize, int borderBits) {
    aruco::drawPlanarBoard((*this), outSize, _img, marginSize, borderBits);
}



/**
 */
GridBoard GridBoard::create(int markersX, int markersY, float markerLength,
                            float markerSeparation, Dictionary _dictionary) {

    GridBoard res;

    CV_Assert(markersX>0 && markersY>0 && markerLength>0 && markerSeparation>0);

    res._markersX = markersX;
    res._markersY = markersY;
    res._markerLength = markerLength;
    res._markerSeparation = markerSeparation;
    res.dictionary = _dictionary;

    int totalMarkers = markersX * markersY;
    res.ids.resize(totalMarkers);
    res.objPoints.reserve(totalMarkers);

    // fill ids with first identifiers
    for (int i = 0; i < totalMarkers; i++)
        res.ids[i] = i;

    // calculate Board objPoints
    float maxY = (float)markersY * markerLength + (markersY - 1) * markerSeparation;
    for (int y = 0; y < markersY; y++) {
        for (int x = 0; x < markersX; x++) {
            vector<Point3f> corners;
            corners.resize(4);
            corners[0] = Point3f(x * (markerLength + markerSeparation),
                                     maxY - y * (markerLength + markerSeparation), 0);
            corners[1] = corners[0] + Point3f(markerLength, 0, 0);
            corners[2] = corners[0] + Point3f(markerLength, -markerLength, 0);
            corners[3] = corners[0] + Point3f(0, -markerLength, 0);
            res.objPoints.push_back(corners);
        }
    }

    return res;
}



/**
 */
void drawDetectedMarkers(InputArray _in, OutputArray _out, InputArrayOfArrays _corners,
                         InputArray _ids, Scalar borderColor) {


    CV_Assert(_in.getMat().cols != 0 && _in.getMat().rows != 0 &&
              (_in.getMat().channels() == 1 || _in.getMat().channels() == 3));
    CV_Assert((_corners.total() == _ids.total()) || _ids.total()==0 );

    // calculate colors
    Scalar textColor, cornerColor;
    textColor = cornerColor = borderColor;
    swap(textColor.val[0], textColor.val[1]);     // text color just sawp G and R
    swap(cornerColor.val[1], cornerColor.val[2]); // corner color just sawp G and B

    _out.create(_in.size(), CV_8UC3);
    Mat outImg = _out.getMat();
    if (_in.getMat().channels()==3)
        _in.getMat().copyTo(outImg);
    else
        cvtColor(_in.getMat(), outImg, COLOR_GRAY2BGR);

    int nMarkers = (int)_corners.total();
    for (int i = 0; i < nMarkers; i++) {
        Mat currentMarker = _corners.getMat(i);
        CV_Assert(currentMarker.total()==4 && currentMarker.type()==CV_32FC2);

        // draw marker sides
        for (int j = 0; j < 4; j++) {
            Point2f p0, p1;
            p0 = currentMarker.ptr<Point2f>(0)[j];
            p1 = currentMarker.ptr<Point2f>(0)[(j + 1) % 4];
            line(outImg, p0, p1, borderColor, 1);
        }
        // draw first corner mark
        rectangle(outImg, currentMarker.ptr<Point2f>(0)[0] - Point2f(3, 3),
                 currentMarker.ptr<Point2f>(0)[0] + Point2f(3, 3), cornerColor, 1, LINE_AA);

        // draw ID
        if (_ids.total() != 0) {
            Point2f cent(0, 0);
            for (int p = 0; p < 4; p++)
                cent += currentMarker.ptr<Point2f>(0)[p];
            cent = cent / 4.;
            stringstream s;
            s << "id=" << _ids.getMat().ptr<int>(0)[i];
            putText(outImg, s.str(), cent, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);
        }
    }

}



/**
 */
void drawAxis(InputArray _in, OutputArray _out, InputArray _cameraMatrix, InputArray _distCoeffs,
              InputArray _rvec, InputArray _tvec, float length) {

    CV_Assert(_in.getMat().cols != 0 && _in.getMat().rows != 0 &&
              (_in.getMat().channels() == 1 || _in.getMat().channels() == 3));
    CV_Assert(length > 0);

    _out.create(_in.size(), CV_8UC3);
    Mat outImg = _out.getMat();
    if (_in.getMat().channels()==3)
        _in.getMat().copyTo(outImg);
    else
        cvtColor(_in.getMat(), outImg, COLOR_GRAY2BGR);

    vector<Point3f> axisPoints;
    axisPoints.push_back(Point3f(0, 0, 0));
    axisPoints.push_back(Point3f(length, 0, 0));
    axisPoints.push_back(Point3f(0, length, 0));
    axisPoints.push_back(Point3f(0, 0, length));
    vector<Point2f> imagePoints;
    projectPoints(axisPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

    line(outImg, imagePoints[0], imagePoints[1], Scalar(0, 0, 255), 3);
    line(outImg, imagePoints[0], imagePoints[2], Scalar(0, 255, 0), 3);
    line(outImg, imagePoints[0], imagePoints[3], Scalar(255, 0, 0), 3);
}



/**
 */
void drawMarker(Dictionary dictionary, int id, int sidePixels, OutputArray _img, int borderBits) {
    dictionary.drawMarker(id, sidePixels, _img, borderBits);
}



/**
 */
void drawPlanarBoard(const Board &board, Size outSize, OutputArray _img, int marginSize,
                     int borderBits) {

    CV_Assert(outSize.area() > 0);
    CV_Assert(marginSize >= 0);

    _img.create(outSize, CV_8UC1);
    Mat out = _img.getMat();
    out.setTo(Scalar::all(255));
    Mat outNoMargins = out.colRange(marginSize, out.cols - marginSize).
                               rowRange(marginSize, out.rows - marginSize);

    // calculate max and min values in XY plane
    CV_Assert(board.objPoints.size() > 0);
    float minX, maxX, minY, maxY;
    minX = maxX = board.objPoints[0][0].x;
    minY = maxY = board.objPoints[0][0].y;

    for (unsigned int i = 0; i < board.objPoints.size(); i++) {
        for (int j = 0; j < 4; j++) {
            minX = min(minX, board.objPoints[i][j].x);
            maxX = max(maxX, board.objPoints[i][j].x);
            minY = min(minY, board.objPoints[i][j].y);
            maxY = max(maxY, board.objPoints[i][j].y);
        }
    }

    float sizeX, sizeY;
    sizeX = maxX - minX;
    sizeY = maxY - minY;

    float xReduction = sizeX / float(outNoMargins.cols);
    float yReduction = sizeY / float(outNoMargins.rows);

    // determine the zone where the markers are placed
    Mat markerZone;
    if (xReduction > yReduction) {
        int nRows = int( sizeY / xReduction);
        int rowsMargins = (outNoMargins.rows - nRows) / 2;
        markerZone = outNoMargins.rowRange(rowsMargins, outNoMargins.rows - rowsMargins);
    } else {
        int nCols = int( sizeX / yReduction);
        int colsMargins = (outNoMargins.cols - nCols) / 2;
        markerZone = outNoMargins.colRange(colsMargins, outNoMargins.cols - colsMargins);
    }

    // now paint each marker
    for (unsigned int m = 0; m < board.objPoints.size(); m++) {

        // transform corners to markerZone coordinates
        vector<Point2f> outCorners;
        outCorners.resize(4);
        for (int j = 0; j < 4; j++) {
            Point2f p0, p1, pf;
            p0 = Point2f(board.objPoints[m][j].x, board.objPoints[m][j].y);
            // remove negativity
            p1.x = p0.x - minX;
            p1.y = p0.y - minY;
            pf.x = p1.x * float(markerZone.cols - 1) / sizeX;
            pf.y = float(markerZone.rows - 1) - p1.y * float(markerZone.rows - 1) / sizeY;
            outCorners[j] = pf;
        }

        // get tiny marker
        int tinyMarkerSize = 10 * board.dictionary.markerSize + 2;
        Mat tinyMarker;
        board.dictionary.drawMarker(board.ids[m], tinyMarkerSize, tinyMarker, borderBits);

        // interpolate tiny marker to marker position in markerZone
        Mat inCorners(4, 1, CV_32FC2);
        inCorners.ptr<Point2f>(0)[0] = Point2f(0, 0);
        inCorners.ptr<Point2f>(0)[1] = Point2f((float)tinyMarker.cols, 0);
        inCorners.ptr<Point2f>(0)[2] = Point2f((float)tinyMarker.cols, (float)tinyMarker.rows);
        inCorners.ptr<Point2f>(0)[3] = Point2f(0, (float)tinyMarker.rows);

        // remove perspective
        Mat transformation = getPerspectiveTransform(inCorners, outCorners);
        Mat aux;
        const char borderValue = 127;
        warpPerspective(tinyMarker, aux, transformation, markerZone.size(), INTER_NEAREST,
                        BORDER_CONSTANT, Scalar::all(borderValue));

        // copy only not-border pixels
        for (int y = 0; y < aux.rows; y++) {
            for (int x = 0; x < aux.cols; x++) {
                if (aux.at<unsigned char>(y, x) == borderValue)
                    continue;
                markerZone.at<unsigned char>(y, x) = aux.at<unsigned char>(y, x);
            }
        }
    }

}



/**
  */
double calibrateCameraAruco(InputArrayOfArrays _corners, InputArray _ids, InputArray _counter,
                            const Board &board, Size imageSize, InputOutputArray _cameraMatrix,
                            InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs,
                            OutputArrayOfArrays _tvecs, int flags, TermCriteria criteria) {

    // for each frame, get properly processed imagePoints and objectPoints for the calibrateCamera
    // function
    vector<Mat> processedObjectPoints, processedImagePoints;
    int nFrames = _counter.total();
    int markerCounter = 0;
    for (int frame = 0; frame < nFrames; frame++) {
        int nMarkersInThisFrame = _counter.getMat().ptr<int>()[frame];
        vector<Mat> thisFrameCorners;
        vector<int> thisFrameIds;
        thisFrameCorners.reserve(nMarkersInThisFrame);
        thisFrameIds.reserve(nMarkersInThisFrame);
        for(int j=markerCounter; j<markerCounter+nMarkersInThisFrame; j++) {
            thisFrameCorners.push_back(_corners.getMat(j));
            thisFrameIds.push_back( _ids.getMat().ptr<int>()[j] );
        }
        markerCounter += nMarkersInThisFrame;
        Mat currentImgPoints, currentObjPoints;
        _getBoardObjectAndImagePoints(board, thisFrameIds, thisFrameCorners, currentImgPoints,
                                      currentObjPoints);
        if (currentImgPoints.total() > 0 && currentObjPoints.total() > 0) {
            processedImagePoints.push_back(currentImgPoints);
            processedObjectPoints.push_back(currentObjPoints);
        }
    }

    return calibrateCamera(processedObjectPoints, processedImagePoints, imageSize,
                           _cameraMatrix, _distCoeffs, _rvecs, _tvecs, flags, criteria);
}


}
}
