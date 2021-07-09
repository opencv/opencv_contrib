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
#include <opencv2/imgproc.hpp>

#include "apriltag_quad_thresh.hpp"
#include "zarray.hpp"

//#define APRIL_DEBUG
#ifdef APRIL_DEBUG
#include "opencv2/imgcodecs.hpp"
#endif

namespace cv {
namespace aruco {

using namespace std;


/**
  *
  */
DetectorParameters::DetectorParameters()
    : adaptiveThreshWinSizeMin(3),
      adaptiveThreshWinSizeMax(23),
      adaptiveThreshWinSizeStep(10),
      adaptiveThreshConstant(7),
      minMarkerPerimeterRate(0.03),
      maxMarkerPerimeterRate(4.),
      polygonalApproxAccuracyRate(0.03),
      minCornerDistanceRate(0.05),
      minDistanceToBorder(3),
      minMarkerDistanceRate(0.05),
      cornerRefinementMethod(CORNER_REFINE_NONE),
      cornerRefinementWinSize(5),
      cornerRefinementMaxIterations(30),
      cornerRefinementMinAccuracy(0.1),
      markerBorderBits(1),
      perspectiveRemovePixelPerCell(4),
      perspectiveRemoveIgnoredMarginPerCell(0.13),
      maxErroneousBitsInBorderRate(0.35),
      minOtsuStdDev(5.0),
      errorCorrectionRate(0.6),
      aprilTagQuadDecimate(0.0),
      aprilTagQuadSigma(0.0),
      aprilTagMinClusterPixels(5),
      aprilTagMaxNmaxima(10),
      aprilTagCriticalRad( (float)(10* CV_PI /180) ),
      aprilTagMaxLineFitMse(10.0),
      aprilTagMinWhiteBlackDiff(5),
      aprilTagDeglitch(0),
      detectInvertedMarker(false){}


/**
  * @brief Create a new set of DetectorParameters with default values.
  */
Ptr<DetectorParameters> DetectorParameters::create() {
    Ptr<DetectorParameters> params = makePtr<DetectorParameters>();
    return params;
}


/**
  * @brief Convert input image to gray if it is a 3-channels image
  */
static void _convertToGrey(InputArray _in, OutputArray _out) {

    CV_Assert(_in.type() == CV_8UC1 || _in.type() == CV_8UC3);

    if(_in.type() == CV_8UC3)
        cvtColor(_in, _out, COLOR_BGR2GRAY);
    else
        _in.copyTo(_out);
}


/**
  * @brief Threshold input image using adaptive thresholding
  */
static void _threshold(InputArray _in, OutputArray _out, int winSize, double constant) {

    CV_Assert(winSize >= 3);
    if(winSize % 2 == 0) winSize++; // win size must be odd
    adaptiveThreshold(_in, _out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, winSize, constant);
}


/**
  * @brief Given a tresholded image, find the contours, calculate their polygonal approximation
  * and take those that accomplish some conditions
  */
static void _findMarkerContours(InputArray _in, vector< vector< Point2f > > &candidates,
                                vector< vector< Point > > &contoursOut, double minPerimeterRate,
                                double maxPerimeterRate, double accuracyRate,
                                double minCornerDistanceRate, int minDistanceToBorder) {

    CV_Assert(minPerimeterRate > 0 && maxPerimeterRate > 0 && accuracyRate > 0 &&
              minCornerDistanceRate >= 0 && minDistanceToBorder >= 0);

    // calculate maximum and minimum sizes in pixels
    unsigned int minPerimeterPixels =
        (unsigned int)(minPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));
    unsigned int maxPerimeterPixels =
        (unsigned int)(maxPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));

    Mat contoursImg;
    _in.getMat().copyTo(contoursImg);
    vector< vector< Point > > contours;
    findContours(contoursImg, contours, RETR_LIST, CHAIN_APPROX_NONE);
    // now filter list of contours
    for(unsigned int i = 0; i < contours.size(); i++) {
        // check perimeter
        if(contours[i].size() < minPerimeterPixels || contours[i].size() > maxPerimeterPixels)
            continue;

        // check is square and is convex
        vector< Point > approxCurve;
        approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * accuracyRate, true);
        if(approxCurve.size() != 4 || !isContourConvex(approxCurve)) continue;

        // check min distance between corners
        double minDistSq =
            max(contoursImg.cols, contoursImg.rows) * max(contoursImg.cols, contoursImg.rows);
        for(int j = 0; j < 4; j++) {
            double d = (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) *
                           (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                       (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y) *
                           (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y);
            minDistSq = min(minDistSq, d);
        }
        double minCornerDistancePixels = double(contours[i].size()) * minCornerDistanceRate;
        if(minDistSq < minCornerDistancePixels * minCornerDistancePixels) continue;

        // check if it is too near to the image border
        bool tooNearBorder = false;
        for(int j = 0; j < 4; j++) {
            if(approxCurve[j].x < minDistanceToBorder || approxCurve[j].y < minDistanceToBorder ||
               approxCurve[j].x > contoursImg.cols - 1 - minDistanceToBorder ||
               approxCurve[j].y > contoursImg.rows - 1 - minDistanceToBorder)
                tooNearBorder = true;
        }
        if(tooNearBorder) continue;

        // if it passes all the test, add to candidates vector
        vector< Point2f > currentCandidate;
        currentCandidate.resize(4);
        for(int j = 0; j < 4; j++) {
            currentCandidate[j] = Point2f((float)approxCurve[j].x, (float)approxCurve[j].y);
        }
        candidates.push_back(currentCandidate);
        contoursOut.push_back(contours[i]);
    }
}


/**
  * @brief Assure order of candidate corners is clockwise direction
  */
static void _reorderCandidatesCorners(vector< vector< Point2f > > &candidates) {

    for(unsigned int i = 0; i < candidates.size(); i++) {
        double dx1 = candidates[i][1].x - candidates[i][0].x;
        double dy1 = candidates[i][1].y - candidates[i][0].y;
        double dx2 = candidates[i][2].x - candidates[i][0].x;
        double dy2 = candidates[i][2].y - candidates[i][0].y;
        double crossProduct = (dx1 * dy2) - (dy1 * dx2);

        if(crossProduct < 0.0) { // not clockwise direction
            swap(candidates[i][1], candidates[i][3]);
        }
    }
}

/**
  * @brief to make sure that the corner's order of both candidates (default/white) is the same
  */
static vector< Point2f > alignContourOrder( Point2f corner, vector< Point2f > candidate){
    uint8_t r=0;
    double min = cv::norm( Vec2f( corner - candidate[0] ), NORM_L2SQR);
    for(uint8_t pos=1; pos < 4; pos++) {
        double nDiff = cv::norm( Vec2f( corner - candidate[pos] ), NORM_L2SQR);
        if(nDiff < min){
            r = pos;
            min =nDiff;
        }
    }
    std::rotate(candidate.begin(), candidate.begin() + r, candidate.end());
    return candidate;
}

/**
  * @brief Check candidates that are too close to each other, save the potential candidates
  *        (i.e. biggest/smallest contour) and remove the rest
  */
static void _filterTooCloseCandidates(const vector< vector< Point2f > > &candidatesIn,
                                      vector< vector< vector< Point2f > > > &candidatesSetOut,
                                      const vector< vector< Point > > &contoursIn,
                                      vector< vector< vector< Point > > > &contoursSetOut,
                                      double minMarkerDistanceRate, bool detectInvertedMarker) {

    CV_Assert(minMarkerDistanceRate >= 0);

    vector<int> candGroup;
    candGroup.resize(candidatesIn.size(), -1);
    vector< vector<unsigned int> > groupedCandidates;
    for(unsigned int i = 0; i < candidatesIn.size(); i++) {
        for(unsigned int j = i + 1; j < candidatesIn.size(); j++) {

            int minimumPerimeter = min((int)contoursIn[i].size(), (int)contoursIn[j].size() );

            // fc is the first corner considered on one of the markers, 4 combinations are possible
            for(int fc = 0; fc < 4; fc++) {
                double distSq = 0;
                for(int c = 0; c < 4; c++) {
                    // modC is the corner considering first corner is fc
                    int modC = (c + fc) % 4;
                    distSq += (candidatesIn[i][modC].x - candidatesIn[j][c].x) *
                                  (candidatesIn[i][modC].x - candidatesIn[j][c].x) +
                              (candidatesIn[i][modC].y - candidatesIn[j][c].y) *
                                  (candidatesIn[i][modC].y - candidatesIn[j][c].y);
                }
                distSq /= 4.;

                // if mean square distance is too low, remove the smaller one of the two markers
                double minMarkerDistancePixels = double(minimumPerimeter) * minMarkerDistanceRate;
                if(distSq < minMarkerDistancePixels * minMarkerDistancePixels) {

                    // i and j are not related to a group
                    if(candGroup[i]<0 && candGroup[j]<0){
                        // mark candidates with their corresponding group number
                        candGroup[i] = candGroup[j] = (int)groupedCandidates.size();

                        // create group
                        vector<unsigned int> grouped;
                        grouped.push_back(i);
                        grouped.push_back(j);
                        groupedCandidates.push_back( grouped );
                    }
                    // i is related to a group
                    else if(candGroup[i] > -1 && candGroup[j] == -1){
                        int group = candGroup[i];
                        candGroup[j] = group;

                        // add to group
                        groupedCandidates[group].push_back( j );
                    }
                    // j is related to a group
                    else if(candGroup[j] > -1 && candGroup[i] == -1){
                        int group = candGroup[j];
                        candGroup[i] = group;

                        // add to group
                        groupedCandidates[group].push_back( i );
                    }
                }
            }
        }
    }

    // save possible candidates
    candidatesSetOut.clear();
    contoursSetOut.clear();

    vector< vector< Point2f > > biggerCandidates;
    vector< vector< Point > > biggerContours;
    vector< vector< Point2f > > smallerCandidates;
    vector< vector< Point > > smallerContours;

    // save possible candidates
    for(unsigned int i = 0; i < groupedCandidates.size(); i++) {
        unsigned int smallerIdx = groupedCandidates[i][0];
        unsigned int biggerIdx = smallerIdx;
        double smallerArea = contourArea(candidatesIn[smallerIdx]);
        double biggerArea = smallerArea;

        // evaluate group elements
        for(unsigned int j = 1; j < groupedCandidates[i].size(); j++) {
            unsigned int currIdx = groupedCandidates[i][j];
            double currArea = contourArea(candidatesIn[currIdx]);

            // check if current contour is bigger
            if(currArea >= biggerArea) {
                biggerIdx = currIdx;
                biggerArea = currArea;
            }

            // check if current contour is smaller
            if(currArea < smallerArea && detectInvertedMarker) {
                smallerIdx = currIdx;
                smallerArea = currArea;
            }
        }

        // add contours and candidates
        biggerCandidates.push_back(candidatesIn[biggerIdx]);
        biggerContours.push_back(contoursIn[biggerIdx]);
        if(detectInvertedMarker) {
            smallerCandidates.push_back(alignContourOrder(candidatesIn[biggerIdx][0], candidatesIn[smallerIdx]));
            smallerContours.push_back(contoursIn[smallerIdx]);
        }
    }
    // to preserve the structure :: candidateSet< defaultCandidates, whiteCandidates >
    // default candidates
    candidatesSetOut.push_back(biggerCandidates);
    contoursSetOut.push_back(biggerContours);
    // white candidates
    candidatesSetOut.push_back(smallerCandidates);
    contoursSetOut.push_back(smallerContours);
}

/**
 * @brief Initial steps on finding square candidates
 */
static void _detectInitialCandidates(const Mat &grey, vector< vector< Point2f > > &candidates,
                                     vector< vector< Point > > &contours,
                                     const Ptr<DetectorParameters> &params) {

    CV_Assert(params->adaptiveThreshWinSizeMin >= 3 && params->adaptiveThreshWinSizeMax >= 3);
    CV_Assert(params->adaptiveThreshWinSizeMax >= params->adaptiveThreshWinSizeMin);
    CV_Assert(params->adaptiveThreshWinSizeStep > 0);

    // number of window sizes (scales) to apply adaptive thresholding
    int nScales =  (params->adaptiveThreshWinSizeMax - params->adaptiveThreshWinSizeMin) /
                      params->adaptiveThreshWinSizeStep + 1;

    vector< vector< vector< Point2f > > > candidatesArrays((size_t) nScales);
    vector< vector< vector< Point > > > contoursArrays((size_t) nScales);

    ////for each value in the interval of thresholding window sizes
    parallel_for_(Range(0, nScales), [&](const Range& range) {
        const int begin = range.start;
        const int end = range.end;

        for (int i = begin; i < end; i++) {
            int currScale = params->adaptiveThreshWinSizeMin + i * params->adaptiveThreshWinSizeStep;
            // threshold
            Mat thresh;
            _threshold(grey, thresh, currScale, params->adaptiveThreshConstant);

            // detect rectangles
            _findMarkerContours(thresh, candidatesArrays[i], contoursArrays[i],
                                params->minMarkerPerimeterRate, params->maxMarkerPerimeterRate,
                                params->polygonalApproxAccuracyRate, params->minCornerDistanceRate,
                                params->minDistanceToBorder);
        }
    });

    // join candidates
    for(int i = 0; i < nScales; i++) {
        for(unsigned int j = 0; j < candidatesArrays[i].size(); j++) {
            candidates.push_back(candidatesArrays[i][j]);
            contours.push_back(contoursArrays[i][j]);
        }
    }
}


/**
 * @brief Detect square candidates in the input image
 */
static void _detectCandidates(InputArray _image, vector< vector< vector< Point2f > > >& candidatesSetOut,
                              vector< vector< vector< Point > > >& contoursSetOut, const Ptr<DetectorParameters> &_params) {

    Mat image = _image.getMat();
    CV_Assert(image.total() != 0);

    /// 1. CONVERT TO GRAY
    Mat grey;
    _convertToGrey(image, grey);

    vector< vector< Point2f > > candidates;
    vector< vector< Point > > contours;
    /// 2. DETECT FIRST SET OF CANDIDATES
    _detectInitialCandidates(grey, candidates, contours, _params);

    /// 3. SORT CORNERS
    _reorderCandidatesCorners(candidates);

    /// 4. FILTER OUT NEAR CANDIDATE PAIRS
    // save the outter/inner border (i.e. potential candidates)
    _filterTooCloseCandidates(candidates, candidatesSetOut, contours, contoursSetOut,
                              _params->minMarkerDistanceRate, _params->detectInvertedMarker);
}


/**
  * @brief Given an input image and candidate corners, extract the bits of the candidate, including
  * the border bits
  */
static Mat _extractBits(InputArray _image, InputArray _corners, int markerSize,
                        int markerBorderBits, int cellSize, double cellMarginRate,
                        double minStdDevOtsu) {

    CV_Assert(_image.getMat().channels() == 1);
    CV_Assert(_corners.total() == 4);
    CV_Assert(markerBorderBits > 0 && cellSize > 0 && cellMarginRate >= 0 && cellMarginRate <= 1);
    CV_Assert(minStdDevOtsu >= 0);

    // number of bits in the marker
    int markerSizeWithBorders = markerSize + 2 * markerBorderBits;
    int cellMarginPixels = int(cellMarginRate * cellSize);

    Mat resultImg; // marker image after removing perspective
    int resultImgSize = markerSizeWithBorders * cellSize;
    Mat resultImgCorners(4, 1, CV_32FC2);
    resultImgCorners.ptr< Point2f >(0)[0] = Point2f(0, 0);
    resultImgCorners.ptr< Point2f >(0)[1] = Point2f((float)resultImgSize - 1, 0);
    resultImgCorners.ptr< Point2f >(0)[2] =
        Point2f((float)resultImgSize - 1, (float)resultImgSize - 1);
    resultImgCorners.ptr< Point2f >(0)[3] = Point2f(0, (float)resultImgSize - 1);

    // remove perspective
    Mat transformation = getPerspectiveTransform(_corners, resultImgCorners);
    warpPerspective(_image, resultImg, transformation, Size(resultImgSize, resultImgSize),
                    INTER_NEAREST);

    // output image containing the bits
    Mat bits(markerSizeWithBorders, markerSizeWithBorders, CV_8UC1, Scalar::all(0));

    // check if standard deviation is enough to apply Otsu
    // if not enough, it probably means all bits are the same color (black or white)
    Mat mean, stddev;
    // Remove some border just to avoid border noise from perspective transformation
    Mat innerRegion = resultImg.colRange(cellSize / 2, resultImg.cols - cellSize / 2)
                          .rowRange(cellSize / 2, resultImg.rows - cellSize / 2);
    meanStdDev(innerRegion, mean, stddev);
    if(stddev.ptr< double >(0)[0] < minStdDevOtsu) {
        // all black or all white, depending on mean value
        if(mean.ptr< double >(0)[0] > 127)
            bits.setTo(1);
        else
            bits.setTo(0);
        return bits;
    }

    // now extract code, first threshold using Otsu
    threshold(resultImg, resultImg, 125, 255, THRESH_BINARY | THRESH_OTSU);

    // for each cell
    for(int y = 0; y < markerSizeWithBorders; y++) {
        for(int x = 0; x < markerSizeWithBorders; x++) {
            int Xstart = x * (cellSize) + cellMarginPixels;
            int Ystart = y * (cellSize) + cellMarginPixels;
            Mat square = resultImg(Rect(Xstart, Ystart, cellSize - 2 * cellMarginPixels,
                                        cellSize - 2 * cellMarginPixels));
            // count white pixels on each cell to assign its value
            size_t nZ = (size_t) countNonZero(square);
            if(nZ > square.total() / 2) bits.at< unsigned char >(y, x) = 1;
        }
    }

    return bits;
}



/**
  * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
  */
static int _getBorderErrors(const Mat &bits, int markerSize, int borderSize) {

    int sizeWithBorders = markerSize + 2 * borderSize;

    CV_Assert(markerSize > 0 && bits.cols == sizeWithBorders && bits.rows == sizeWithBorders);

    int totalErrors = 0;
    for(int y = 0; y < sizeWithBorders; y++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr< unsigned char >(y)[k] != 0) totalErrors++;
            if(bits.ptr< unsigned char >(y)[sizeWithBorders - 1 - k] != 0) totalErrors++;
        }
    }
    for(int x = borderSize; x < sizeWithBorders - borderSize; x++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr< unsigned char >(k)[x] != 0) totalErrors++;
            if(bits.ptr< unsigned char >(sizeWithBorders - 1 - k)[x] != 0) totalErrors++;
        }
    }
    return totalErrors;
}


/**
 * @brief Tries to identify one candidate given the dictionary
 * @return candidate typ. zero if the candidate is not valid,
 *                           1 if the candidate is a black candidate (default candidate)
 *                           2 if the candidate is a white candidate
 */
static uint8_t _identifyOneCandidate(const Ptr<Dictionary>& dictionary, InputArray _image,
                                  vector<Point2f>& _corners, int& idx,
                                  const Ptr<DetectorParameters>& params, int& rotation)
{
    CV_Assert(_corners.size() == 4);
    CV_Assert(_image.getMat().total() != 0);
    CV_Assert(params->markerBorderBits > 0);

    uint8_t typ=1;
    // get bits
    Mat candidateBits =
        _extractBits(_image, _corners, dictionary->markerSize, params->markerBorderBits,
                     params->perspectiveRemovePixelPerCell,
                     params->perspectiveRemoveIgnoredMarginPerCell, params->minOtsuStdDev);

    // analyze border bits
    int maximumErrorsInBorder =
        int(dictionary->markerSize * dictionary->markerSize * params->maxErroneousBitsInBorderRate);
    int borderErrors =
        _getBorderErrors(candidateBits, dictionary->markerSize, params->markerBorderBits);

    // check if it is a white marker
    if(params->detectInvertedMarker){
        // to get from 255 to 1
        Mat invertedImg = ~candidateBits-254;
        int invBError = _getBorderErrors(invertedImg, dictionary->markerSize, params->markerBorderBits);
        // white marker
        if(invBError<borderErrors){
            borderErrors = invBError;
            invertedImg.copyTo(candidateBits);
            typ=2;
        }
    }
    if(borderErrors > maximumErrorsInBorder) return 0; // border is wrong

    // take only inner bits
    Mat onlyBits =
        candidateBits.rowRange(params->markerBorderBits,
                               candidateBits.rows - params->markerBorderBits)
            .colRange(params->markerBorderBits, candidateBits.cols - params->markerBorderBits);

    // try to indentify the marker
    if(!dictionary->identify(onlyBits, idx, rotation, params->errorCorrectionRate))
        return 0;

    return typ;
}

/**
 * @brief Copy the contents of a corners vector to an OutputArray, settings its size.
 */
static void _copyVector2Output(vector< vector< Point2f > > &vec, OutputArrayOfArrays out) {
    out.create((int)vec.size(), 1, CV_32FC2);

    if(out.isMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            Mat &m = out.getMatRef(i);
            Mat(Mat(vec[i]).t()).copyTo(m);
        }
    }
    else if(out.isUMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            UMat &m = out.getUMatRef(i);
            Mat(Mat(vec[i]).t()).copyTo(m);
        }
    }
    else if(out.kind() == _OutputArray::STD_VECTOR_VECTOR){
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            Mat m = out.getMat(i);
            Mat(Mat(vec[i]).t()).copyTo(m);
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> OutputArrays are currently supported.");
    }
}

/**
 * @brief rotate the initial corner to get to the right position
 */
static void correctCornerPosition( vector< Point2f >& _candidate, int rotate){
    std::rotate(_candidate.begin(), _candidate.begin() + 4 - rotate, _candidate.end());
}

/**
 * @brief Identify square candidates according to a marker dictionary
 */
static void _identifyCandidates(InputArray _image, vector< vector< vector< Point2f > > >& _candidatesSet,
                                vector< vector< vector<Point> > >& _contoursSet, const Ptr<Dictionary> &_dictionary,
                                vector< vector< Point2f > >& _accepted, vector< vector<Point> >& _contours, vector< int >& ids,
                                const Ptr<DetectorParameters> &params,
                                OutputArrayOfArrays _rejected = noArray()) {

    int ncandidates = (int)_candidatesSet[0].size();
    vector< vector< Point2f > > accepted;
    vector< vector< Point2f > > rejected;

    vector< vector< Point > > contours;

    CV_Assert(_image.getMat().total() != 0);

    Mat grey;
    _convertToGrey(_image.getMat(), grey);

    vector< int > idsTmp(ncandidates, -1);
    vector< int > rotated(ncandidates, 0);
    vector< uint8_t > validCandidates(ncandidates, 0);

    //// Analyze each of the candidates
    parallel_for_(Range(0, ncandidates), [&](const Range &range) {
        const int begin = range.start;
        const int end = range.end;

        vector< vector< Point2f > >& candidates = params->detectInvertedMarker ? _candidatesSet[1] : _candidatesSet[0];

        for(int i = begin; i < end; i++) {
            int currId;
            validCandidates[i] = _identifyOneCandidate(_dictionary, grey, candidates[i], currId, params, rotated[i]);

            if(validCandidates[i] > 0)
                idsTmp[i] = currId;
        }
    });

    for(int i = 0; i < ncandidates; i++) {
        if(validCandidates[i] > 0) {
            // to choose the right set of candidates :: 0 for default, 1 for white markers
            uint8_t set = validCandidates[i]-1;

            // shift corner positions to the correct rotation
            correctCornerPosition(_candidatesSet[set][i], rotated[i]);

            if( !params->detectInvertedMarker && validCandidates[i] == 2 )
                continue;

            // add valid candidate
            accepted.push_back(_candidatesSet[set][i]);
            ids.push_back(idsTmp[i]);

            contours.push_back(_contoursSet[set][i]);

        } else {
            rejected.push_back(_candidatesSet[0][i]);
        }
    }

    // parse output
    _accepted = accepted;

    _contours= contours;

    if(_rejected.needed()) {
        _copyVector2Output(rejected, _rejected);
    }
}


/**
  * @brief Return object points for the system centered in a single marker, given the marker length
  */
static void _getSingleMarkerObjectPoints(float markerLength, OutputArray _objPoints) {

    CV_Assert(markerLength > 0);

    _objPoints.create(4, 1, CV_32FC3);
    Mat objPoints = _objPoints.getMat();
    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.ptr< Vec3f >(0)[0] = Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[1] = Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[2] = Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[3] = Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);
}

/**
 * Line fitting  A * B = C :: Called from function refineCandidateLines
 * @param nContours, contour-container
 */
static Point3f _interpolate2Dline(const std::vector<cv::Point2f>& nContours){
	float minX, minY, maxX, maxY;
	minX = maxX = nContours[0].x;
	minY = maxY = nContours[0].y;

	for(unsigned int i = 0; i< nContours.size(); i++){
		minX = nContours[i].x < minX ? nContours[i].x : minX;
		minY = nContours[i].y < minY ? nContours[i].y : minY;
		maxX = nContours[i].x > maxX ? nContours[i].x : maxX;
		maxY = nContours[i].y > maxY ? nContours[i].y : maxY;
	}

	Mat A = Mat::ones((int)nContours.size(), 2, CV_32F); // Coefficient Matrix (N x 2)
	Mat B((int)nContours.size(), 1, CV_32F);				// Variables   Matrix (N x 1)
	Mat C;											// Constant

	if(maxX - minX > maxY - minY){
		for(unsigned int i =0; i < nContours.size(); i++){
            A.at<float>(i,0)= nContours[i].x;
            B.at<float>(i,0)= nContours[i].y;
		}

		solve(A, B, C, DECOMP_NORMAL);

		return Point3f(C.at<float>(0, 0), -1., C.at<float>(1, 0));
	}
	else{
		for(unsigned int i =0; i < nContours.size(); i++){
			A.at<float>(i,0)= nContours[i].y;
			B.at<float>(i,0)= nContours[i].x;
		}

		solve(A, B, C, DECOMP_NORMAL);

		return Point3f(-1., C.at<float>(0, 0), C.at<float>(1, 0));
	}

}

/**
 * Find the Point where the lines crosses :: Called from function refineCandidateLines
 * @param nLine1
 * @param nLine2
 * @return Crossed Point
 */
static Point2f _getCrossPoint(Point3f nLine1, Point3f nLine2){
	Matx22f A(nLine1.x, nLine1.y, nLine2.x, nLine2.y);
	Vec2f B(-nLine1.z, -nLine2.z);
	return Vec2f(A.solve(B).val);
}

static void _distortPoints(vector<cv::Point2f>& in, const Mat& camMatrix, const Mat& distCoeff) {
    // trivial extrinsics
    Matx31f Rvec(0,0,0);
    Matx31f Tvec(0,0,0);

    // calculate 3d points and then reproject, so opencv makes the distortion internally
    vector<cv::Point3f> cornersPoints3d;
    for (unsigned int i = 0; i < in.size(); i++){
        float x= (in[i].x - float(camMatrix.at<double>(0, 2))) / float(camMatrix.at<double>(0, 0));
        float y= (in[i].y - float(camMatrix.at<double>(1, 2))) / float(camMatrix.at<double>(1, 1));
        cornersPoints3d.push_back(Point3f(x,y,1));
    }
    cv::projectPoints(cornersPoints3d, Rvec, Tvec, camMatrix, distCoeff, in);
}

/**
 * Refine Corners using the contour vector :: Called from function detectMarkers
 * @param nContours, contour-container
 * @param nCorners, candidate Corners
 * @param camMatrix, cameraMatrix input 3x3 floating-point camera matrix
 * @param distCoeff, distCoeffs vector of distortion coefficient
 */
static void _refineCandidateLines(std::vector<Point>& nContours, std::vector<Point2f>& nCorners, const Mat& camMatrix, const Mat& distCoeff){
	vector<Point2f> contour2f(nContours.begin(), nContours.end());

	if(!camMatrix.empty() && !distCoeff.empty()){
		undistortPoints(contour2f, contour2f, camMatrix, distCoeff);
	}

	/* 5 groups :: to group the edges
	 * 4 - classified by its corner
	 * extra group - (temporary) if contours do not begin with a corner
	 */
	vector<Point2f> cntPts[5];
	int cornerIndex[4]={-1};
	int group=4;

	for ( unsigned int i =0; i < nContours.size(); i++ ) {
		for(unsigned int j=0; j<4; j++){
			if ( nCorners[j] == contour2f[i] ){
				cornerIndex[j] = i;
				group=j;
			}
		}
		cntPts[group].push_back(contour2f[i]);
	}

	// saves extra group into corresponding
	if( !cntPts[4].empty() ){
            CV_CheckLT(group, 4, "FIXIT: avoiding infinite loop: implementation should be revised: https://github.com/opencv/opencv_contrib/issues/2738");
		for( unsigned int i=0; i < cntPts[4].size() ; i++ )
			cntPts[group].push_back(cntPts[4].at(i));
		cntPts[4].clear();
	}

	//Evaluate contour direction :: using the position of the detected corners
	int inc=1;

        inc = ( (cornerIndex[0] > cornerIndex[1]) &&  (cornerIndex[3] > cornerIndex[0]) ) ? -1:inc;
	inc = ( (cornerIndex[2] > cornerIndex[3]) &&  (cornerIndex[1] > cornerIndex[2]) ) ? -1:inc;

	// calculate the line :: who passes through the grouped points
	Point3f lines[4];
	for(int i=0; i<4; i++){
		lines[i]=_interpolate2Dline(cntPts[i]);
	}

	/*
	 * calculate the corner :: where the lines crosses to each other
	 * clockwise direction		no clockwise direction
	 *      0                           1
	 *      .---. 1                     .---. 2
	 *      |   |                       |   |
	 *    3 .___.                     0 .___.
	 *          2                           3
	 */
	for(int i=0; i < 4; i++){
		if(inc<0)
			nCorners[i] = _getCrossPoint(lines[ i ], lines[ (i+1)%4 ]);	// 01 12 23 30
		else
			nCorners[i] = _getCrossPoint(lines[ i ], lines[ (i+3)%4 ]);	// 30 01 12 23
	}

	if(!camMatrix.empty() && !distCoeff.empty()){
		_distortPoints(nCorners, camMatrix, distCoeff);
	}
}

#ifdef APRIL_DEBUG
static void _darken(const Mat &im){
    for (int y = 0; y < im.rows; y++) {
        for (int x = 0; x < im.cols; x++) {
            im.data[im.cols*y+x] /= 2;
        }
    }
}
#endif

/**
 *
 * @param im_orig
 * @param _params
 * @param candidates
 * @param contours
 */
static void _apriltag(Mat im_orig, const Ptr<DetectorParameters> & _params, std::vector< std::vector< Point2f > > &candidates,
        std::vector< std::vector< Point > > &contours){

    ///////////////////////////////////////////////////////////
    /// Step 1. Detect quads according to requested image decimation
    /// and blurring parameters.
    Mat quad_im;
    im_orig.copyTo(quad_im);

    if (_params->aprilTagQuadDecimate > 1){
        resize(im_orig, quad_im, Size(), 1/_params->aprilTagQuadDecimate, 1/_params->aprilTagQuadDecimate, INTER_AREA );
    }

    // Apply a Blur
    if (_params->aprilTagQuadSigma != 0) {
        // compute a reasonable kernel width by figuring that the
        // kernel should go out 2 std devs.
        //
        // max sigma          ksz
        // 0.499              1  (disabled)
        // 0.999              3
        // 1.499              5
        // 1.999              7

        float sigma = fabsf((float) _params->aprilTagQuadSigma);

        int ksz = cvFloor(4 * sigma); // 2 std devs in each direction
        ksz |= 1; // make odd number

        if (ksz > 1) {
            if (_params->aprilTagQuadSigma > 0)
                GaussianBlur(quad_im, quad_im, Size(ksz, ksz), sigma, sigma, BORDER_REPLICATE);
            else {
                Mat orig;
                quad_im.copyTo(orig);
                GaussianBlur(quad_im, quad_im, Size(ksz, ksz), sigma, sigma, BORDER_REPLICATE);

                // SHARPEN the image by subtracting the low frequency components.
                for (int y = 0; y < orig.rows; y++) {
                    for (int x = 0; x < orig.cols; x++) {
                        int vorig = orig.data[y*orig.step + x];
                        int vblur = quad_im.data[y*quad_im.step + x];

                        int v = 2*vorig - vblur;
                        if (v < 0)
                            v = 0;
                        if (v > 255)
                            v = 255;

                        quad_im.data[y*quad_im.step + x] = (uint8_t) v;
                    }
                }
            }
        }
    }

#ifdef APRIL_DEBUG
    imwrite("1.1 debug_preprocess.pnm", quad_im);
#endif

    ///////////////////////////////////////////////////////////
    /// Step 2. do the Threshold :: get the set of candidate quads
    zarray_t *quads = apriltag_quad_thresh(_params, quad_im, contours);

    CV_Assert(quads != NULL);

    // adjust centers of pixels so that they correspond to the
    // original full-resolution image.
    if (_params->aprilTagQuadDecimate > 1) {
        for (int i = 0; i < _zarray_size(quads); i++) {
            struct sQuad *q;
            _zarray_get_volatile(quads, i, &q);
            for (int j = 0; j < 4; j++) {
                q->p[j][0] *= _params->aprilTagQuadDecimate;
                q->p[j][1] *= _params->aprilTagQuadDecimate;
            }
        }
    }

#ifdef APRIL_DEBUG
    Mat im_quads = im_orig.clone();
    im_quads = im_quads*0.5;
    srandom(0);

    for (int i = 0; i < _zarray_size(quads); i++) {
        struct sQuad *quad;
        _zarray_get_volatile(quads, i, &quad);

        const int bias = 100;
        int color = bias + (random() % (255-bias));

        line(im_quads, Point(quad->p[0][0], quad->p[0][1]), Point(quad->p[1][0], quad->p[1][1]), color, 1);
        line(im_quads, Point(quad->p[1][0], quad->p[1][1]), Point(quad->p[2][0], quad->p[2][1]), color, 1);
        line(im_quads, Point(quad->p[2][0], quad->p[2][1]), Point(quad->p[3][0], quad->p[3][1]), color, 1);
        line(im_quads, Point(quad->p[3][0], quad->p[3][1]), Point(quad->p[0][0], quad->p[0][1]), color, 1);
    }
    imwrite("1.2 debug_quads_raw.pnm", im_quads);
#endif

    ////////////////////////////////////////////////////////////////
    /// Step 3. Save the output :: candidate corners
    for (int i = 0; i < _zarray_size(quads); i++) {
        struct sQuad *quad;
        _zarray_get_volatile(quads, i, &quad);

        std::vector< Point2f > corners;
        corners.push_back(Point2f(quad->p[3][0], quad->p[3][1]));   //pA
        corners.push_back(Point2f(quad->p[0][0], quad->p[0][1]));   //pB
        corners.push_back(Point2f(quad->p[1][0], quad->p[1][1]));   //pC
        corners.push_back(Point2f(quad->p[2][0], quad->p[2][1]));   //pD

        candidates.push_back(corners);
    }

    _zarray_destroy(quads);
}


/**
  */
void detectMarkers(InputArray _image, const Ptr<Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints, InputArrayOfArrays camMatrix, InputArrayOfArrays distCoeff) {

    CV_Assert(!_image.empty());

    Mat grey;
    _convertToGrey(_image.getMat(), grey);

    /// STEP 1: Detect marker candidates
    vector< vector< Point2f > > candidates;
    vector< vector< Point > > contours;
    vector< int > ids;

    vector< vector< vector< Point2f > > > candidatesSet;
    vector< vector< vector< Point > > > contoursSet;
    /// STEP 1.a Detect marker candidates :: using AprilTag
    if(_params->cornerRefinementMethod == CORNER_REFINE_APRILTAG){
        _apriltag(grey, _params, candidates, contours);

        candidatesSet.push_back(candidates);
        contoursSet.push_back(contours);
    }

    /// STEP 1.b Detect marker candidates :: traditional way
    else
        _detectCandidates(grey, candidatesSet, contoursSet, _params);

    /// STEP 2: Check candidate codification (identify markers)
    _identifyCandidates(grey, candidatesSet, contoursSet, _dictionary, candidates, contours, ids, _params,
                        _rejectedImgPoints);

    // copy to output arrays
    _copyVector2Output(candidates, _corners);
    Mat(ids).copyTo(_ids);

    /// STEP 3: Corner refinement :: use corner subpix
    if( _params->cornerRefinementMethod == CORNER_REFINE_SUBPIX ) {
        CV_Assert(_params->cornerRefinementWinSize > 0 && _params->cornerRefinementMaxIterations > 0 &&
                  _params->cornerRefinementMinAccuracy > 0);

        //// do corner refinement for each of the detected markers
        parallel_for_(Range(0, _corners.cols()), [&](const Range& range) {
            const int begin = range.start;
            const int end = range.end;

            for (int i = begin; i < end; i++) {
                cornerSubPix(grey, _corners.getMat(i),
                             Size(_params->cornerRefinementWinSize, _params->cornerRefinementWinSize),
                             Size(-1, -1),
                             TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                          _params->cornerRefinementMaxIterations,
                                          _params->cornerRefinementMinAccuracy));
            }
        });
    }

    /// STEP 3, Optional : Corner refinement :: use contour container
    if( _params->cornerRefinementMethod == CORNER_REFINE_CONTOUR){

        if(! _ids.empty()){

            // do corner refinement using the contours for each detected markers
            parallel_for_(Range(0, _corners.cols()), [&](const Range& range) {
                for (int i = range.start; i < range.end; i++) {
                    _refineCandidateLines(contours[i], candidates[i], camMatrix.getMat(),
                                          distCoeff.getMat());
                }
            });

            // copy the corners to the output array
            _copyVector2Output(candidates, _corners);
        }
    }
}

/**
  */
void estimatePoseSingleMarkers(InputArrayOfArrays _corners, float markerLength,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArray _rvecs, OutputArray _tvecs, OutputArray _objPoints) {

    CV_Assert(markerLength > 0);

    Mat markerObjPoints;
    _getSingleMarkerObjectPoints(markerLength, markerObjPoints);
    int nMarkers = (int)_corners.total();
    _rvecs.create(nMarkers, 1, CV_64FC3);
    _tvecs.create(nMarkers, 1, CV_64FC3);

    Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();

    //// for each marker, calculate its pose
    parallel_for_(Range(0, nMarkers), [&](const Range& range) {
        const int begin = range.start;
        const int end = range.end;

        for (int i = begin; i < end; i++) {
            solvePnP(markerObjPoints, _corners.getMat(i), _cameraMatrix, _distCoeffs, rvecs.at<Vec3d>(i),
                     tvecs.at<Vec3d>(i));
        }
    });

    if(_objPoints.needed()){
        markerObjPoints.convertTo(_objPoints, -1);
    }
}



void getBoardObjectAndImagePoints(const Ptr<Board> &board, InputArrayOfArrays detectedCorners,
    InputArray detectedIds, OutputArray objPoints, OutputArray imgPoints) {

    CV_Assert(board->ids.size() == board->objPoints.size());
    CV_Assert(detectedIds.total() == detectedCorners.total());

    size_t nDetectedMarkers = detectedIds.total();

    vector< Point3f > objPnts;
    objPnts.reserve(nDetectedMarkers);

    vector< Point2f > imgPnts;
    imgPnts.reserve(nDetectedMarkers);

    // look for detected markers that belong to the board and get their information
    for(unsigned int i = 0; i < nDetectedMarkers; i++) {
        int currentId = detectedIds.getMat().ptr< int >(0)[i];
        for(unsigned int j = 0; j < board->ids.size(); j++) {
            if(currentId == board->ids[j]) {
                for(int p = 0; p < 4; p++) {
                    objPnts.push_back(board->objPoints[j][p]);
                    imgPnts.push_back(detectedCorners.getMat(i).ptr< Point2f >(0)[p]);
                }
            }
        }
    }

    // create output
    Mat(objPnts).copyTo(objPoints);
    Mat(imgPnts).copyTo(imgPoints);
}



/**
  * Project board markers that are not included in the list of detected markers
  */
static void _projectUndetectedMarkers(const Ptr<Board> &_board, InputOutputArrayOfArrays _detectedCorners,
                                      InputOutputArray _detectedIds, InputArray _cameraMatrix,
                                      InputArray _distCoeffs,
                                      vector< vector< Point2f > >& _undetectedMarkersProjectedCorners,
                                      OutputArray _undetectedMarkersIds) {

    // first estimate board pose with the current avaible markers
    Mat rvec, tvec;
    int boardDetectedMarkers;
    boardDetectedMarkers = aruco::estimatePoseBoard(_detectedCorners, _detectedIds, _board,
                                                    _cameraMatrix, _distCoeffs, rvec, tvec);

    // at least one marker from board so rvec and tvec are valid
    if(boardDetectedMarkers == 0) return;

    // search undetected markers and project them using the previous pose
    vector< vector< Point2f > > undetectedCorners;
    vector< int > undetectedIds;
    for(unsigned int i = 0; i < _board->ids.size(); i++) {
        int foundIdx = -1;
        for(unsigned int j = 0; j < _detectedIds.total(); j++) {
            if(_board->ids[i] == _detectedIds.getMat().ptr< int >()[j]) {
                foundIdx = j;
                break;
            }
        }

        // not detected
        if(foundIdx == -1) {
            undetectedCorners.push_back(vector< Point2f >());
            undetectedIds.push_back(_board->ids[i]);
            projectPoints(_board->objPoints[i], rvec, tvec, _cameraMatrix, _distCoeffs,
                          undetectedCorners.back());
        }
    }


    // parse output
    Mat(undetectedIds).copyTo(_undetectedMarkersIds);
    _undetectedMarkersProjectedCorners = undetectedCorners;
}



/**
  * Interpolate board markers that are not included in the list of detected markers using
  * global homography
  */
static void _projectUndetectedMarkers(const Ptr<Board> &_board, InputOutputArrayOfArrays _detectedCorners,
                                      InputOutputArray _detectedIds,
                                      vector< vector< Point2f > >& _undetectedMarkersProjectedCorners,
                                      OutputArray _undetectedMarkersIds) {


    // check board points are in the same plane, if not, global homography cannot be applied
    CV_Assert(_board->objPoints.size() > 0);
    CV_Assert(_board->objPoints[0].size() > 0);
    float boardZ = _board->objPoints[0][0].z;
    for(unsigned int i = 0; i < _board->objPoints.size(); i++) {
        for(unsigned int j = 0; j < _board->objPoints[i].size(); j++) {
            CV_Assert(boardZ == _board->objPoints[i][j].z);
        }
    }

    vector< Point2f > detectedMarkersObj2DAll; // Object coordinates (without Z) of all the detected
                                               // marker corners in a single vector
    vector< Point2f > imageCornersAll; // Image corners of all detected markers in a single vector
    vector< vector< Point2f > > undetectedMarkersObj2D; // Object coordinates (without Z) of all
                                                        // missing markers in different vectors
    vector< int > undetectedMarkersIds; // ids of missing markers
    // find markers included in board, and missing markers from board. Fill the previous vectors
    for(unsigned int j = 0; j < _board->ids.size(); j++) {
        bool found = false;
        for(unsigned int i = 0; i < _detectedIds.total(); i++) {
            if(_detectedIds.getMat().ptr< int >()[i] == _board->ids[j]) {
                for(int c = 0; c < 4; c++) {
                    imageCornersAll.push_back(_detectedCorners.getMat(i).ptr< Point2f >()[c]);
                    detectedMarkersObj2DAll.push_back(
                        Point2f(_board->objPoints[j][c].x, _board->objPoints[j][c].y));
                }
                found = true;
                break;
            }
        }
        if(!found) {
            undetectedMarkersObj2D.push_back(vector< Point2f >());
            for(int c = 0; c < 4; c++) {
                undetectedMarkersObj2D.back().push_back(
                    Point2f(_board->objPoints[j][c].x, _board->objPoints[j][c].y));
            }
            undetectedMarkersIds.push_back(_board->ids[j]);
        }
    }
    if(imageCornersAll.size() == 0) return;

    // get homography from detected markers
    Mat transformation = findHomography(detectedMarkersObj2DAll, imageCornersAll);

    _undetectedMarkersProjectedCorners.resize(undetectedMarkersIds.size());

    // for each undetected marker, apply transformation
    for(unsigned int i = 0; i < undetectedMarkersObj2D.size(); i++) {
        perspectiveTransform(undetectedMarkersObj2D[i], _undetectedMarkersProjectedCorners[i], transformation);
    }

    Mat(undetectedMarkersIds).copyTo(_undetectedMarkersIds);
}



/**
  */
void refineDetectedMarkers(InputArray _image, const Ptr<Board> &_board,
                           InputOutputArrayOfArrays _detectedCorners, InputOutputArray _detectedIds,
                           InputOutputArrayOfArrays _rejectedCorners, InputArray _cameraMatrix,
                           InputArray _distCoeffs, float minRepDistance, float errorCorrectionRate,
                           bool checkAllOrders, OutputArray _recoveredIdxs,
                           const Ptr<DetectorParameters> &_params) {

    CV_Assert(minRepDistance > 0);

    if(_detectedIds.total() == 0 || _rejectedCorners.total() == 0) return;

    DetectorParameters &params = *_params;

    // get projections of missing markers in the board
    vector< vector< Point2f > > undetectedMarkersCorners;
    vector< int > undetectedMarkersIds;
    if(_cameraMatrix.total() != 0) {
        // reproject based on camera projection model
        _projectUndetectedMarkers(_board, _detectedCorners, _detectedIds, _cameraMatrix, _distCoeffs,
                                  undetectedMarkersCorners, undetectedMarkersIds);

    } else {
        // reproject based on global homography
        _projectUndetectedMarkers(_board, _detectedCorners, _detectedIds, undetectedMarkersCorners,
                                  undetectedMarkersIds);
    }

    // list of missing markers indicating if they have been assigned to a candidate
    vector< bool > alreadyIdentified(_rejectedCorners.total(), false);

    // maximum bits that can be corrected
    Dictionary &dictionary = *(_board->dictionary);
    int maxCorrectionRecalculated =
        int(double(dictionary.maxCorrectionBits) * errorCorrectionRate);

    Mat grey;
    _convertToGrey(_image, grey);

    // vector of final detected marker corners and ids
    vector< Mat > finalAcceptedCorners;
    vector< int > finalAcceptedIds;
    // fill with the current markers
    finalAcceptedCorners.resize(_detectedCorners.total());
    finalAcceptedIds.resize(_detectedIds.total());
    for(unsigned int i = 0; i < _detectedIds.total(); i++) {
        finalAcceptedCorners[i] = _detectedCorners.getMat(i).clone();
        finalAcceptedIds[i] = _detectedIds.getMat().ptr< int >()[i];
    }
    vector< int > recoveredIdxs; // original indexes of accepted markers in _rejectedCorners

    // for each missing marker, try to find a correspondence
    for(unsigned int i = 0; i < undetectedMarkersIds.size(); i++) {

        // best match at the moment
        int closestCandidateIdx = -1;
        double closestCandidateDistance = minRepDistance * minRepDistance + 1;
        Mat closestRotatedMarker;

        for(unsigned int j = 0; j < _rejectedCorners.total(); j++) {
            if(alreadyIdentified[j]) continue;

            // check distance
            double minDistance = closestCandidateDistance + 1;
            bool valid = false;
            int validRot = 0;
            for(int c = 0; c < 4; c++) { // first corner in rejected candidate
                double currentMaxDistance = 0;
                for(int k = 0; k < 4; k++) {
                    Point2f rejCorner = _rejectedCorners.getMat(j).ptr< Point2f >()[(c + k) % 4];
                    Point2f distVector = undetectedMarkersCorners[i][k] - rejCorner;
                    double cornerDist = distVector.x * distVector.x + distVector.y * distVector.y;
                    currentMaxDistance = max(currentMaxDistance, cornerDist);
                }
                // if distance is better than current best distance
                if(currentMaxDistance < closestCandidateDistance) {
                    valid = true;
                    validRot = c;
                    minDistance = currentMaxDistance;
                }
                if(!checkAllOrders) break;
            }

            if(!valid) continue;

            // apply rotation
            Mat rotatedMarker;
            if(checkAllOrders) {
                rotatedMarker = Mat(4, 1, CV_32FC2);
                for(int c = 0; c < 4; c++)
                    rotatedMarker.ptr< Point2f >()[c] =
                        _rejectedCorners.getMat(j).ptr< Point2f >()[(c + 4 + validRot) % 4];
            }
            else rotatedMarker = _rejectedCorners.getMat(j);

            // last filter, check if inner code is close enough to the assigned marker code
            int codeDistance = 0;
            // if errorCorrectionRate, dont check code
            if(errorCorrectionRate >= 0) {

                // extract bits
                Mat bits = _extractBits(
                    grey, rotatedMarker, dictionary.markerSize, params.markerBorderBits,
                    params.perspectiveRemovePixelPerCell,
                    params.perspectiveRemoveIgnoredMarginPerCell, params.minOtsuStdDev);

                Mat onlyBits =
                    bits.rowRange(params.markerBorderBits, bits.rows - params.markerBorderBits)
                        .colRange(params.markerBorderBits, bits.rows - params.markerBorderBits);

                codeDistance =
                    dictionary.getDistanceToId(onlyBits, undetectedMarkersIds[i], false);
            }

            // if everythin is ok, assign values to current best match
            if(errorCorrectionRate < 0 || codeDistance < maxCorrectionRecalculated) {
                closestCandidateIdx = j;
                closestCandidateDistance = minDistance;
                closestRotatedMarker = rotatedMarker;
            }
        }

        // if at least one good match, we have rescue the missing marker
        if(closestCandidateIdx >= 0) {

            // subpixel refinement
            if(_params->cornerRefinementMethod == CORNER_REFINE_SUBPIX) {
                CV_Assert(params.cornerRefinementWinSize > 0 &&
                          params.cornerRefinementMaxIterations > 0 &&
                          params.cornerRefinementMinAccuracy > 0);
                cornerSubPix(grey, closestRotatedMarker,
                             Size(params.cornerRefinementWinSize, params.cornerRefinementWinSize),
                             Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                                        params.cornerRefinementMaxIterations,
                                                        params.cornerRefinementMinAccuracy));
            }

            // remove from rejected
            alreadyIdentified[closestCandidateIdx] = true;

            // add to detected
            finalAcceptedCorners.push_back(closestRotatedMarker);
            finalAcceptedIds.push_back(undetectedMarkersIds[i]);

            // add the original index of the candidate
            recoveredIdxs.push_back(closestCandidateIdx);
        }
    }

    // parse output
    if(finalAcceptedIds.size() != _detectedIds.total()) {
        _detectedCorners.clear();
        _detectedIds.clear();

        // parse output
        Mat(finalAcceptedIds).copyTo(_detectedIds);

        _detectedCorners.create((int)finalAcceptedCorners.size(), 1, CV_32FC2);
        for(unsigned int i = 0; i < finalAcceptedCorners.size(); i++) {
            _detectedCorners.create(4, 1, CV_32FC2, i, true);
            for(int j = 0; j < 4; j++) {
                _detectedCorners.getMat(i).ptr< Point2f >()[j] =
                    finalAcceptedCorners[i].ptr< Point2f >()[j];
            }
        }

        // recalculate _rejectedCorners based on alreadyIdentified
        vector< Mat > finalRejected;
        for(unsigned int i = 0; i < alreadyIdentified.size(); i++) {
            if(!alreadyIdentified[i]) {
                finalRejected.push_back(_rejectedCorners.getMat(i).clone());
            }
        }

        _rejectedCorners.clear();
        _rejectedCorners.create((int)finalRejected.size(), 1, CV_32FC2);
        for(unsigned int i = 0; i < finalRejected.size(); i++) {
            _rejectedCorners.create(4, 1, CV_32FC2, i, true);
            for(int j = 0; j < 4; j++) {
                _rejectedCorners.getMat(i).ptr< Point2f >()[j] =
                    finalRejected[i].ptr< Point2f >()[j];
            }
        }

        if(_recoveredIdxs.needed()) {
            Mat(recoveredIdxs).copyTo(_recoveredIdxs);
        }
    }
}




/**
  */
int estimatePoseBoard(InputArrayOfArrays _corners, InputArray _ids, const Ptr<Board> &board,
                      InputArray _cameraMatrix, InputArray _distCoeffs, InputOutputArray _rvec,
                      InputOutputArray _tvec, bool useExtrinsicGuess) {

    CV_Assert(_corners.total() == _ids.total());

    // get object and image points for the solvePnP function
    Mat objPoints, imgPoints;
    getBoardObjectAndImagePoints(board, _corners, _ids, objPoints, imgPoints);

    CV_Assert(imgPoints.total() == objPoints.total());

    if(objPoints.total() == 0) // 0 of the detected markers in board
        return 0;

    solvePnP(objPoints, imgPoints, _cameraMatrix, _distCoeffs, _rvec, _tvec, useExtrinsicGuess);

    // divide by four since all the four corners are concatenated in the array for each marker
    return (int)objPoints.total() / 4;
}




/**
 */
void GridBoard::draw(Size outSize, OutputArray _img, int marginSize, int borderBits) {
    _drawPlanarBoardImpl(this, outSize, _img, marginSize, borderBits);
}


/**
*/
Ptr<Board> Board::create(InputArrayOfArrays objPoints, const Ptr<Dictionary> &dictionary, InputArray ids) {

    CV_Assert(objPoints.total() == ids.total());
    CV_Assert(objPoints.type() == CV_32FC3 || objPoints.type() == CV_32FC1);

    std::vector< std::vector< Point3f > > obj_points_vector;
    for (unsigned int i = 0; i < objPoints.total(); i++) {
        std::vector<Point3f> corners;
        Mat corners_mat = objPoints.getMat(i);

        if(corners_mat.type() == CV_32FC1)
            corners_mat = corners_mat.reshape(3);
        CV_Assert(corners_mat.total() == 4);

        for (int j = 0; j < 4; j++) {
            corners.push_back(corners_mat.at<Point3f>(j));
        }
        obj_points_vector.push_back(corners);
    }

    Ptr<Board> res = makePtr<Board>();
    ids.copyTo(res->ids);
    res->objPoints = obj_points_vector;
    res->dictionary = cv::makePtr<Dictionary>(dictionary);
    return res;
}

/**
 */
void Board::setIds(InputArray ids_) {
    CV_Assert(objPoints.size() == ids_.total());
    ids_.copyTo(this->ids);
}

/**
 */
Ptr<GridBoard> GridBoard::create(int markersX, int markersY, float markerLength, float markerSeparation,
                            const Ptr<Dictionary> &dictionary, int firstMarker) {

    CV_Assert(markersX > 0 && markersY > 0 && markerLength > 0 && markerSeparation > 0);

    Ptr<GridBoard> res = makePtr<GridBoard>();

    res->_markersX = markersX;
    res->_markersY = markersY;
    res->_markerLength = markerLength;
    res->_markerSeparation = markerSeparation;
    res->dictionary = dictionary;

    size_t totalMarkers = (size_t) markersX * markersY;
    res->ids.resize(totalMarkers);
    res->objPoints.reserve(totalMarkers);

    // fill ids with first identifiers
    for(unsigned int i = 0; i < totalMarkers; i++) {
        res->ids[i] = i + firstMarker;
    }

    // calculate Board objPoints
    float maxY = (float)markersY * markerLength + (markersY - 1) * markerSeparation;
    for(int y = 0; y < markersY; y++) {
        for(int x = 0; x < markersX; x++) {
            vector< Point3f > corners;
            corners.resize(4);
            corners[0] = Point3f(x * (markerLength + markerSeparation),
                                 maxY - y * (markerLength + markerSeparation), 0);
            corners[1] = corners[0] + Point3f(markerLength, 0, 0);
            corners[2] = corners[0] + Point3f(markerLength, -markerLength, 0);
            corners[3] = corners[0] + Point3f(0, -markerLength, 0);
            res->objPoints.push_back(corners);
        }
    }

    return res;
}



/**
 */
void drawDetectedMarkers(InputOutputArray _image, InputArrayOfArrays _corners,
                         InputArray _ids, Scalar borderColor) {


    CV_Assert(_image.getMat().total() != 0 &&
              (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
    CV_Assert((_corners.total() == _ids.total()) || _ids.total() == 0);

    // calculate colors
    Scalar textColor, cornerColor;
    textColor = cornerColor = borderColor;
    swap(textColor.val[0], textColor.val[1]);     // text color just sawp G and R
    swap(cornerColor.val[1], cornerColor.val[2]); // corner color just sawp G and B

    int nMarkers = (int)_corners.total();
    for(int i = 0; i < nMarkers; i++) {
        Mat currentMarker = _corners.getMat(i);
        CV_Assert(currentMarker.total() == 4 && currentMarker.type() == CV_32FC2);

        // draw marker sides
        for(int j = 0; j < 4; j++) {
            Point2f p0, p1;
            p0 = currentMarker.ptr< Point2f >(0)[j];
            p1 = currentMarker.ptr< Point2f >(0)[(j + 1) % 4];
            line(_image, p0, p1, borderColor, 1);
        }
        // draw first corner mark
        rectangle(_image, currentMarker.ptr< Point2f >(0)[0] - Point2f(3, 3),
                  currentMarker.ptr< Point2f >(0)[0] + Point2f(3, 3), cornerColor, 1, LINE_AA);

        // draw ID
        if(_ids.total() != 0) {
            Point2f cent(0, 0);
            for(int p = 0; p < 4; p++)
                cent += currentMarker.ptr< Point2f >(0)[p];
            cent = cent / 4.;
            stringstream s;
            s << "id=" << _ids.getMat().ptr< int >(0)[i];
            putText(_image, s.str(), cent, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);
        }
    }
}



/**
 */
void drawAxis(InputOutputArray _image, InputArray _cameraMatrix, InputArray _distCoeffs, InputArray _rvec,
              InputArray _tvec, float length)
{
    drawFrameAxes(_image, _cameraMatrix, _distCoeffs, _rvec, _tvec, length, 3);
}

/**
 */
void drawMarker(const Ptr<Dictionary> &dictionary, int id, int sidePixels, OutputArray _img, int borderBits) {
    dictionary->drawMarker(id, sidePixels, _img, borderBits);
}



void _drawPlanarBoardImpl(Board *_board, Size outSize, OutputArray _img, int marginSize,
                     int borderBits) {

    CV_Assert(!outSize.empty());
    CV_Assert(marginSize >= 0);

    _img.create(outSize, CV_8UC1);
    Mat out = _img.getMat();
    out.setTo(Scalar::all(255));
    out.adjustROI(-marginSize, -marginSize, -marginSize, -marginSize);

    // calculate max and min values in XY plane
    CV_Assert(_board->objPoints.size() > 0);
    float minX, maxX, minY, maxY;
    minX = maxX = _board->objPoints[0][0].x;
    minY = maxY = _board->objPoints[0][0].y;

    for(unsigned int i = 0; i < _board->objPoints.size(); i++) {
        for(int j = 0; j < 4; j++) {
            minX = min(minX, _board->objPoints[i][j].x);
            maxX = max(maxX, _board->objPoints[i][j].x);
            minY = min(minY, _board->objPoints[i][j].y);
            maxY = max(maxY, _board->objPoints[i][j].y);
        }
    }

    float sizeX = maxX - minX;
    float sizeY = maxY - minY;

    // proportion transformations
    float xReduction = sizeX / float(out.cols);
    float yReduction = sizeY / float(out.rows);

    // determine the zone where the markers are placed
    if(xReduction > yReduction) {
        int nRows = int(sizeY / xReduction);
        int rowsMargins = (out.rows - nRows) / 2;
        out.adjustROI(-rowsMargins, -rowsMargins, 0, 0);
    } else {
        int nCols = int(sizeX / yReduction);
        int colsMargins = (out.cols - nCols) / 2;
        out.adjustROI(0, 0, -colsMargins, -colsMargins);
    }

    // now paint each marker
    Dictionary &dictionary = *(_board->dictionary);
    Mat marker;
    Point2f outCorners[3];
    Point2f inCorners[3];
    for(unsigned int m = 0; m < _board->objPoints.size(); m++) {
        // transform corners to markerZone coordinates
        for(int j = 0; j < 3; j++) {
            Point2f pf = Point2f(_board->objPoints[m][j].x, _board->objPoints[m][j].y);
            // move top left to 0, 0
            pf -= Point2f(minX, minY);
            pf.x = pf.x / sizeX * float(out.cols);
            pf.y = (1.0f - pf.y / sizeY) * float(out.rows);
            outCorners[j] = pf;
        }

        // get marker
        Size dst_sz(outCorners[2] - outCorners[0]); // assuming CCW order
        dst_sz.width = dst_sz.height = std::min(dst_sz.width, dst_sz.height); //marker should be square
        dictionary.drawMarker(_board->ids[m], dst_sz.width, marker, borderBits);

        if((outCorners[0].y == outCorners[1].y) && (outCorners[1].x == outCorners[2].x)) {
            // marker is aligned to image axes
            marker.copyTo(out(Rect(outCorners[0], dst_sz)));
            continue;
        }

        // interpolate tiny marker to marker position in markerZone
        inCorners[0] = Point2f(-0.5f, -0.5f);
        inCorners[1] = Point2f(marker.cols - 0.5f, -0.5f);
        inCorners[2] = Point2f(marker.cols - 0.5f, marker.rows - 0.5f);

        // remove perspective
        Mat transformation = getAffineTransform(inCorners, outCorners);
        warpAffine(marker, out, transformation, out.size(), INTER_LINEAR,
                        BORDER_TRANSPARENT);
    }
}



/**
 */
void drawPlanarBoard(const Ptr<Board> &_board, Size outSize, OutputArray _img, int marginSize,
                     int borderBits) {
    _drawPlanarBoardImpl(_board, outSize, _img, marginSize, borderBits);
}



/**
 */
double calibrateCameraAruco(InputArrayOfArrays _corners, InputArray _ids, InputArray _counter,
                            const Ptr<Board> &board, Size imageSize, InputOutputArray _cameraMatrix,
                            InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs,
                            OutputArrayOfArrays _tvecs,
                            OutputArray _stdDeviationsIntrinsics,
                            OutputArray _stdDeviationsExtrinsics,
                            OutputArray _perViewErrors,
                            int flags, TermCriteria criteria) {

    // for each frame, get properly processed imagePoints and objectPoints for the calibrateCamera
    // function
    vector< Mat > processedObjectPoints, processedImagePoints;
    size_t nFrames = _counter.total();
    int markerCounter = 0;
    for(size_t frame = 0; frame < nFrames; frame++) {
        int nMarkersInThisFrame =  _counter.getMat().ptr< int >()[frame];
        vector< Mat > thisFrameCorners;
        vector< int > thisFrameIds;

        CV_Assert(nMarkersInThisFrame > 0);

        thisFrameCorners.reserve((size_t) nMarkersInThisFrame);
        thisFrameIds.reserve((size_t) nMarkersInThisFrame);
        for(int j = markerCounter; j < markerCounter + nMarkersInThisFrame; j++) {
            thisFrameCorners.push_back(_corners.getMat(j));
            thisFrameIds.push_back(_ids.getMat().ptr< int >()[j]);
        }
        markerCounter += nMarkersInThisFrame;
        Mat currentImgPoints, currentObjPoints;
        getBoardObjectAndImagePoints(board, thisFrameCorners, thisFrameIds, currentObjPoints,
            currentImgPoints);
        if(currentImgPoints.total() > 0 && currentObjPoints.total() > 0) {
            processedImagePoints.push_back(currentImgPoints);
            processedObjectPoints.push_back(currentObjPoints);
        }
    }

    return calibrateCamera(processedObjectPoints, processedImagePoints, imageSize, _cameraMatrix,
                           _distCoeffs, _rvecs, _tvecs, _stdDeviationsIntrinsics, _stdDeviationsExtrinsics,
                           _perViewErrors, flags, criteria);
}



/**
 */
double calibrateCameraAruco(InputArrayOfArrays _corners, InputArray _ids, InputArray _counter,
  const Ptr<Board> &board, Size imageSize, InputOutputArray _cameraMatrix,
  InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs,
  OutputArrayOfArrays _tvecs, int flags, TermCriteria criteria) {
    return calibrateCameraAruco(_corners, _ids, _counter, board, imageSize, _cameraMatrix, _distCoeffs, _rvecs, _tvecs,
      noArray(), noArray(), noArray(), flags, criteria);
}



}
}
