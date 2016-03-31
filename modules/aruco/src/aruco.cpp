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
      doCornerRefinement(false),
      cornerRefinementWinSize(5),
      cornerRefinementMaxIterations(30),
      cornerRefinementMinAccuracy(0.1),
      markerBorderBits(1),
      perspectiveRemovePixelPerCell(4),
      perspectiveRemoveIgnoredMarginPerCell(0.13),
      maxErroneousBitsInBorderRate(0.35),
      minOtsuStdDev(5.0),
      errorCorrectionRate(0.6) {}


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

    CV_Assert(_in.getMat().channels() == 1 || _in.getMat().channels() == 3);

    _out.create(_in.getMat().size(), CV_8UC1);
    if(_in.getMat().type() == CV_8UC3)
        cvtColor(_in.getMat(), _out.getMat(), COLOR_BGR2GRAY);
    else
        _in.getMat().copyTo(_out);
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
  * @brief Check candidates that are too close to each other and remove the smaller one
  */
static void _filterTooCloseCandidates(const vector< vector< Point2f > > &candidatesIn,
                                      vector< vector< Point2f > > &candidatesOut,
                                      const vector< vector< Point > > &contoursIn,
                                      vector< vector< Point > > &contoursOut,
                                      double minMarkerDistanceRate) {

    CV_Assert(minMarkerDistanceRate >= 0);

    vector< pair< int, int > > nearCandidates;
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
                    nearCandidates.push_back(pair< int, int >(i, j));
                    break;
                }
            }
        }
    }

    // mark smaller one in pairs to remove
    vector< bool > toRemove(candidatesIn.size(), false);
    for(unsigned int i = 0; i < nearCandidates.size(); i++) {
        // if one of the marker has been already markerd to removed, dont need to do anything
        if(toRemove[nearCandidates[i].first] || toRemove[nearCandidates[i].second]) continue;
        size_t perimeter1 = contoursIn[nearCandidates[i].first].size();
        size_t perimeter2 = contoursIn[nearCandidates[i].second].size();
        if(perimeter1 > perimeter2)
            toRemove[nearCandidates[i].second] = true;
        else
            toRemove[nearCandidates[i].first] = true;
    }

    // remove extra candidates
    candidatesOut.clear();
    unsigned long totalRemaining = 0;
    for(unsigned int i = 0; i < toRemove.size(); i++)
        if(!toRemove[i]) totalRemaining++;
    candidatesOut.resize(totalRemaining);
    contoursOut.resize(totalRemaining);
    for(unsigned int i = 0, currIdx = 0; i < candidatesIn.size(); i++) {
        if(toRemove[i]) continue;
        candidatesOut[currIdx] = candidatesIn[i];
        contoursOut[currIdx] = contoursIn[i];
        currIdx++;
    }
}


/**
  * ParallelLoopBody class for the parallelization of the basic candidate detections using
  * different threhold window sizes. Called from function _detectInitialCandidates()
  */
class DetectInitialCandidatesParallel : public ParallelLoopBody {
    public:
    DetectInitialCandidatesParallel(const Mat *_grey,
                                    vector< vector< vector< Point2f > > > *_candidatesArrays,
                                    vector< vector< vector< Point > > > *_contoursArrays,
                                    const Ptr<DetectorParameters> &_params)
        : grey(_grey), candidatesArrays(_candidatesArrays), contoursArrays(_contoursArrays),
          params(_params) {}

    void operator()(const Range &range) const {
        const int begin = range.start;
        const int end = range.end;

        for(int i = begin; i < end; i++) {
            int currScale =
                params->adaptiveThreshWinSizeMin + i * params->adaptiveThreshWinSizeStep;
            // threshold
            Mat thresh;
            _threshold(*grey, thresh, currScale, params->adaptiveThreshConstant);

            // detect rectangles
            _findMarkerContours(thresh, (*candidatesArrays)[i], (*contoursArrays)[i],
                                params->minMarkerPerimeterRate, params->maxMarkerPerimeterRate,
                                params->polygonalApproxAccuracyRate, params->minCornerDistanceRate,
                                params->minDistanceToBorder);
        }
    }

    private:
    DetectInitialCandidatesParallel &operator=(const DetectInitialCandidatesParallel &);

    const Mat *grey;
    vector< vector< vector< Point2f > > > *candidatesArrays;
    vector< vector< vector< Point > > > *contoursArrays;
    const Ptr<DetectorParameters> &params;
};


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
    // for(int i = 0; i < nScales; i++) {
    //    int currScale = params.adaptiveThreshWinSizeMin + i*params.adaptiveThreshWinSizeStep;
    //    // treshold
    //    Mat thresh;
    //    _threshold(grey, thresh, currScale, params.adaptiveThreshConstant);
    //    // detect rectangles
    //    _findMarkerContours(thresh, candidatesArrays[i], contoursArrays[i],
    // params.minMarkerPerimeterRate,
    //                        params.maxMarkerPerimeterRate, params.polygonalApproxAccuracyRate,
    //                        params.minCornerDistance, params.minDistanceToBorder);
    //}

    // this is the parallel call for the previous commented loop (result is equivalent)
    parallel_for_(Range(0, nScales), DetectInitialCandidatesParallel(&grey, &candidatesArrays,
                                                                     &contoursArrays, params));

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
static void _detectCandidates(InputArray _image, OutputArrayOfArrays _candidates,
                              OutputArrayOfArrays _contours, const Ptr<DetectorParameters> &_params) {

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
    vector< vector< Point2f > > candidatesOut;
    vector< vector< Point > > contoursOut;
    _filterTooCloseCandidates(candidates, candidatesOut, contours, contoursOut,
                              _params->minMarkerDistanceRate);

    // parse output
    _candidates.create((int)candidatesOut.size(), 1, CV_32FC2);
    _contours.create((int)contoursOut.size(), 1, CV_32SC2);
    for(int i = 0; i < (int)candidatesOut.size(); i++) {
        _candidates.create(4, 1, CV_32FC2, i, true);
        Mat m = _candidates.getMat(i);
        for(int j = 0; j < 4; j++)
            m.ptr< Vec2f >(0)[j] = candidatesOut[i][j];

        _contours.create((int)contoursOut[i].size(), 1, CV_32SC2, i, true);
        Mat c = _contours.getMat(i);
        for(unsigned int j = 0; j < contoursOut[i].size(); j++)
            c.ptr< Point2i >()[j] = contoursOut[i][j];
    }
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
 */
static bool _identifyOneCandidate(Ptr<Dictionary> &dictionary, InputArray _image,
                                  InputOutputArray _corners, int &idx, const Ptr<DetectorParameters> &params) {

    CV_Assert(_corners.total() == 4);
    CV_Assert(_image.getMat().total() != 0);
    CV_Assert(params->markerBorderBits > 0);

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
    if(borderErrors > maximumErrorsInBorder) return false; // border is wrong

    // take only inner bits
    Mat onlyBits =
        candidateBits.rowRange(params->markerBorderBits,
                               candidateBits.rows - params->markerBorderBits)
            .colRange(params->markerBorderBits, candidateBits.rows - params->markerBorderBits);

    // try to indentify the marker
    int rotation;
    if(!dictionary->identify(onlyBits, idx, rotation, params->errorCorrectionRate))
        return false;
    else {
        // shift corner positions to the correct rotation
        if(rotation != 0) {
            Mat copyPoints = _corners.getMat().clone();
            for(int j = 0; j < 4; j++)
                _corners.getMat().ptr< Point2f >(0)[j] =
                    copyPoints.ptr< Point2f >(0)[(j + 4 - rotation) % 4];
        }
        return true;
    }
}


/**
  * ParallelLoopBody class for the parallelization of the marker identification step
  * Called from function _identifyCandidates()
  */
class IdentifyCandidatesParallel : public ParallelLoopBody {
    public:
    IdentifyCandidatesParallel(const Mat *_grey, InputArrayOfArrays _candidates,
                               InputArrayOfArrays _contours, Ptr<Dictionary> &_dictionary,
                               vector< int > *_idsTmp, vector< char > *_validCandidates,
                               const Ptr<DetectorParameters> &_params)
        : grey(_grey), candidates(_candidates), contours(_contours), dictionary(_dictionary),
          idsTmp(_idsTmp), validCandidates(_validCandidates), params(_params) {}

    void operator()(const Range &range) const {
        const int begin = range.start;
        const int end = range.end;

        for(int i = begin; i < end; i++) {
            int currId;
            Mat currentCandidate = candidates.getMat(i);
            if(_identifyOneCandidate(dictionary, *grey, currentCandidate, currId, params)) {
                (*validCandidates)[i] = 1;
                (*idsTmp)[i] = currId;
            }
        }
    }

    private:
    IdentifyCandidatesParallel &operator=(const IdentifyCandidatesParallel &); // to quiet MSVC

    const Mat *grey;
    InputArrayOfArrays candidates, contours;
    Ptr<Dictionary> &dictionary;
    vector< int > *idsTmp;
    vector< char > *validCandidates;
    const Ptr<DetectorParameters> &params;
};



/**
 * @brief Copy the contents of a Mat vector to an OutputArray, settings its size.
 */
void _copyVector2Output(vector< Mat > &vec, OutputArrayOfArrays out);



/**
 *
 */
void _copyVector2Output(vector< Mat > &vec, OutputArrayOfArrays out) {

    out.release();
    out.create((int)vec.size(), 1, CV_32FC2);

    if(out.isMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i, true);
            Mat &m = out.getMatRef(i);
            vec[i].copyTo(m);
        }
    }
    else if(out.isUMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i, true);
            UMat &m = out.getUMatRef(i);
            vec[i].copyTo(m);
        }
    }
    else if(out.kind() == _OutputArray::STD_VECTOR_VECTOR){
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i, true);
            Mat m = out.getMat(i);
            vec[i].copyTo(m);
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> OutputArrays are currently supported.");
    }
}



/**
 * @brief Identify square candidates according to a marker dictionary
 */
static void _identifyCandidates(InputArray _image, InputArrayOfArrays _candidates,
                                InputArrayOfArrays _contours, Ptr<Dictionary> &_dictionary,
                                OutputArrayOfArrays _accepted, OutputArray _ids,
                                const Ptr<DetectorParameters> &params,
                                OutputArrayOfArrays _rejected = noArray()) {

    int ncandidates = (int)_candidates.total();

    vector< Mat > accepted;
    vector< Mat > rejected;
    vector< int > ids;

    CV_Assert(_image.getMat().total() != 0);

    Mat grey;
    _convertToGrey(_image.getMat(), grey);

    vector< int > idsTmp(ncandidates, -1);
    vector< char > validCandidates(ncandidates, 0);

    //// Analyze each of the candidates
    // for (int i = 0; i < ncandidates; i++) {
    //    int currId = i;
    //    Mat currentCandidate = _candidates.getMat(i);
    //    if (_identifyOneCandidate(dictionary, grey, currentCandidate, currId, params)) {
    //        validCandidates[i] = 1;
    //        idsTmp[i] = currId;
    //    }
    //}

    // this is the parallel call for the previous commented loop (result is equivalent)
    parallel_for_(Range(0, ncandidates),
                  IdentifyCandidatesParallel(&grey, _candidates, _contours, _dictionary, &idsTmp,
                                             &validCandidates, params));

    for(int i = 0; i < ncandidates; i++) {
        if(validCandidates[i] == 1) {
            accepted.push_back(_candidates.getMat(i));
            ids.push_back(idsTmp[i]);
        } else {
            rejected.push_back(_candidates.getMat(i));
        }
    }

    // parse output
    _copyVector2Output(accepted, _accepted);

    _ids.create((int)ids.size(), 1, CV_32SC1);
    for(unsigned int i = 0; i < ids.size(); i++)
        _ids.getMat().ptr< int >(0)[i] = ids[i];

    if(_rejected.needed()) {
        _copyVector2Output(rejected, _rejected);
    }
}


/**
  * @brief Final filter of markers after its identification
  */
static void _filterDetectedMarkers(InputArrayOfArrays _inCorners, InputArray _inIds,
                                   OutputArrayOfArrays _outCorners, OutputArray _outIds) {

    CV_Assert(_inCorners.total() == _inIds.total());
    if(_inCorners.total() == 0) return;

    // mark markers that will be removed
    vector< bool > toRemove(_inCorners.total(), false);
    bool atLeastOneRemove = false;

    // remove repeated markers with same id, if one contains the other (doble border bug)
    for(unsigned int i = 0; i < _inCorners.total() - 1; i++) {
        for(unsigned int j = i + 1; j < _inCorners.total(); j++) {
            if(_inIds.getMat().ptr< int >(0)[i] != _inIds.getMat().ptr< int >(0)[j]) continue;

            // check if first marker is inside second
            bool inside = true;
            for(unsigned int p = 0; p < 4; p++) {
                Point2f point = _inCorners.getMat(j).ptr< Point2f >(0)[p];
                if(pointPolygonTest(_inCorners.getMat(i), point, false) < 0) {
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
            for(unsigned int p = 0; p < 4; p++) {
                Point2f point = _inCorners.getMat(i).ptr< Point2f >(0)[p];
                if(pointPolygonTest(_inCorners.getMat(j), point, false) < 0) {
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
    if(atLeastOneRemove) {
        vector< Mat > filteredCorners;
        vector< int > filteredIds;

        for(unsigned int i = 0; i < toRemove.size(); i++) {
            if(!toRemove[i]) {
                filteredCorners.push_back(_inCorners.getMat(i).clone());
                filteredIds.push_back(_inIds.getMat().ptr< int >(0)[i]);
            }
        }

        _outIds.create((int)filteredIds.size(), 1, CV_32SC1);
        for(unsigned int i = 0; i < filteredIds.size(); i++)
            _outIds.getMat().ptr< int >(0)[i] = filteredIds[i];

        _outCorners.create((int)filteredCorners.size(), 1, CV_32FC2);
        for(unsigned int i = 0; i < filteredCorners.size(); i++) {
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
    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.ptr< Vec3f >(0)[0] = Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[1] = Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[2] = Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[3] = Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);
}




/**
  * ParallelLoopBody class for the parallelization of the marker corner subpixel refinement
  * Called from function detectMarkers()
  */
class MarkerSubpixelParallel : public ParallelLoopBody {
    public:
    MarkerSubpixelParallel(const Mat *_grey, OutputArrayOfArrays _corners,
                           const Ptr<DetectorParameters> &_params)
        : grey(_grey), corners(_corners), params(_params) {}

    void operator()(const Range &range) const {
        const int begin = range.start;
        const int end = range.end;

        for(int i = begin; i < end; i++) {
            cornerSubPix(*grey, corners.getMat(i),
                         Size(params->cornerRefinementWinSize, params->cornerRefinementWinSize),
                         Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                                    params->cornerRefinementMaxIterations,
                                                    params->cornerRefinementMinAccuracy));
        }
    }

    private:
    MarkerSubpixelParallel &operator=(const MarkerSubpixelParallel &); // to quiet MSVC

    const Mat *grey;
    OutputArrayOfArrays corners;
    const Ptr<DetectorParameters> &params;
};



/**
  */
void detectMarkers(InputArray _image, Ptr<Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints) {

    CV_Assert(_image.getMat().total() != 0);

    Mat grey;
    _convertToGrey(_image.getMat(), grey);

    /// STEP 1: Detect marker candidates
    vector< vector< Point2f > > candidates;
    vector< vector< Point > > contours;
    _detectCandidates(grey, candidates, contours, _params);

    /// STEP 2: Check candidate codification (identify markers)
    _identifyCandidates(grey, candidates, contours, _dictionary, _corners, _ids, _params,
                        _rejectedImgPoints);

    /// STEP 3: Filter detected markers;
    _filterDetectedMarkers(_corners, _ids, _corners, _ids);

    /// STEP 4: Corner refinement
    if(_params->doCornerRefinement) {
        CV_Assert(_params->cornerRefinementWinSize > 0 && _params->cornerRefinementMaxIterations > 0 &&
                  _params->cornerRefinementMinAccuracy > 0);

        //// do corner refinement for each of the detected markers
        // for (unsigned int i = 0; i < _corners.total(); i++) {
        //    cornerSubPix(grey, _corners.getMat(i),
        //                 Size(params.cornerRefinementWinSize, params.cornerRefinementWinSize),
        //                 Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
        //                                            params.cornerRefinementMaxIterations,
        //                                            params.cornerRefinementMinAccuracy));
        //}

        // this is the parallel call for the previous commented loop (result is equivalent)
        parallel_for_(Range(0, (int)_corners.total()),
                      MarkerSubpixelParallel(&grey, _corners, _params));
    }
}



/**
  * ParallelLoopBody class for the parallelization of the single markers pose estimation
  * Called from function estimatePoseSingleMarkers()
  */
class SinglePoseEstimationParallel : public ParallelLoopBody {
    public:
    SinglePoseEstimationParallel(Mat& _markerObjPoints, InputArrayOfArrays _corners,
                                 InputArray _cameraMatrix, InputArray _distCoeffs,
                                 Mat& _rvecs, Mat& _tvecs)
        : markerObjPoints(_markerObjPoints), corners(_corners), cameraMatrix(_cameraMatrix),
          distCoeffs(_distCoeffs), rvecs(_rvecs), tvecs(_tvecs) {}

    void operator()(const Range &range) const {
        const int begin = range.start;
        const int end = range.end;

        for(int i = begin; i < end; i++) {
            solvePnP(markerObjPoints, corners.getMat(i), cameraMatrix, distCoeffs,
                    rvecs.at<Vec3d>(i), tvecs.at<Vec3d>(i));
        }
    }

    private:
    SinglePoseEstimationParallel &operator=(const SinglePoseEstimationParallel &); // to quiet MSVC

    Mat& markerObjPoints;
    InputArrayOfArrays corners;
    InputArray cameraMatrix, distCoeffs;
    Mat& rvecs, tvecs;
};




/**
  */
void estimatePoseSingleMarkers(InputArrayOfArrays _corners, float markerLength,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArray _rvecs, OutputArray _tvecs) {

    CV_Assert(markerLength > 0);

    Mat markerObjPoints;
    _getSingleMarkerObjectPoints(markerLength, markerObjPoints);
    int nMarkers = (int)_corners.total();
    _rvecs.create(nMarkers, 1, CV_64FC3);
    _tvecs.create(nMarkers, 1, CV_64FC3);

    Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();

    //// for each marker, calculate its pose
    // for (int i = 0; i < nMarkers; i++) {
    //    solvePnP(markerObjPoints, _corners.getMat(i), _cameraMatrix, _distCoeffs,
    //             _rvecs.getMat(i), _tvecs.getMat(i));
    //}

    // this is the parallel call for the previous commented loop (result is equivalent)
    parallel_for_(Range(0, nMarkers),
                  SinglePoseEstimationParallel(markerObjPoints, _corners, _cameraMatrix,
                                               _distCoeffs, rvecs, tvecs));
}



/**
  * @brief Given a board configuration and a set of detected markers, returns the corresponding
  * image points and object points to call solvePnP
  */
static void _getBoardObjectAndImagePoints(Ptr<Board> &_board, InputArray _detectedIds,
                                          InputArrayOfArrays _detectedCorners,
                                          OutputArray _imgPoints, OutputArray _objPoints) {

    CV_Assert(_board->ids.size() == _board->objPoints.size());
    CV_Assert(_detectedIds.total() == _detectedCorners.total());

    size_t nDetectedMarkers = _detectedIds.total();

    vector< Point3f > objPnts;
    objPnts.reserve(nDetectedMarkers);

    vector< Point2f > imgPnts;
    imgPnts.reserve(nDetectedMarkers);

    // look for detected markers that belong to the board and get their information
    for(unsigned int i = 0; i < nDetectedMarkers; i++) {
        int currentId = _detectedIds.getMat().ptr< int >(0)[i];
        for(unsigned int j = 0; j < _board->ids.size(); j++) {
            if(currentId == _board->ids[j]) {
                for(int p = 0; p < 4; p++) {
                    objPnts.push_back(_board->objPoints[j][p]);
                    imgPnts.push_back(_detectedCorners.getMat(i).ptr< Point2f >(0)[p]);
                }
            }
        }
    }

    // create output
    _objPoints.create((int)objPnts.size(), 1, CV_32FC3);
    for(unsigned int i = 0; i < objPnts.size(); i++)
        _objPoints.getMat().ptr< Point3f >(0)[i] = objPnts[i];

    _imgPoints.create((int)objPnts.size(), 1, CV_32FC2);
    for(unsigned int i = 0; i < imgPnts.size(); i++)
        _imgPoints.getMat().ptr< Point2f >(0)[i] = imgPnts[i];
}



/**
  * Project board markers that are not included in the list of detected markers
  */
static void _projectUndetectedMarkers(Ptr<Board> &_board, InputOutputArrayOfArrays _detectedCorners,
                                      InputOutputArray _detectedIds, InputArray _cameraMatrix,
                                      InputArray _distCoeffs,
                                      OutputArrayOfArrays _undetectedMarkersProjectedCorners,
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
    _undetectedMarkersIds.create((int)undetectedIds.size(), 1, CV_32SC1);
    for(unsigned int i = 0; i < undetectedIds.size(); i++)
        _undetectedMarkersIds.getMat().ptr< int >(0)[i] = undetectedIds[i];

    _undetectedMarkersProjectedCorners.create((int)undetectedCorners.size(), 1, CV_32FC2);
    for(unsigned int i = 0; i < undetectedCorners.size(); i++) {
        _undetectedMarkersProjectedCorners.create(4, 1, CV_32FC2, i, true);
        for(int j = 0; j < 4; j++) {
            _undetectedMarkersProjectedCorners.getMat(i).ptr< Point2f >()[j] =
                undetectedCorners[i][j];
        }
    }
}



/**
  * Interpolate board markers that are not included in the list of detected markers using
  * global homography
  */
static void _projectUndetectedMarkers(Ptr<Board> &_board, InputOutputArrayOfArrays _detectedCorners,
                                      InputOutputArray _detectedIds,
                                      OutputArrayOfArrays _undetectedMarkersProjectedCorners,
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

    _undetectedMarkersProjectedCorners.create((int)undetectedMarkersIds.size(), 1, CV_32FC2);

    // for each undetected marker, apply transformation
    for(unsigned int i = 0; i < undetectedMarkersObj2D.size(); i++) {
        Mat projectedMarker;
        perspectiveTransform(undetectedMarkersObj2D[i], projectedMarker, transformation);

        _undetectedMarkersProjectedCorners.create(4, 1, CV_32FC2, i, true);
        projectedMarker.copyTo(_undetectedMarkersProjectedCorners.getMat(i));
    }

    _undetectedMarkersIds.create((int)undetectedMarkersIds.size(), 1, CV_32SC1);
    for(unsigned int i = 0; i < undetectedMarkersIds.size(); i++)
        _undetectedMarkersIds.getMat().ptr< int >(0)[i] = undetectedMarkersIds[i];
}



/**
  */
void refineDetectedMarkers(InputArray _image, Ptr<Board> &_board,
                           InputOutputArrayOfArrays _detectedCorners, InputOutputArray _detectedIds,
                           InputOutputArray _rejectedCorners, InputArray _cameraMatrix,
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
            if(params.doCornerRefinement) {
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
        _detectedIds.create((int)finalAcceptedIds.size(), 1, CV_32SC1);
        for(unsigned int i = 0; i < finalAcceptedIds.size(); i++)
            _detectedIds.getMat().ptr< int >(0)[i] = finalAcceptedIds[i];

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
            _recoveredIdxs.create((int)recoveredIdxs.size(), 1, CV_32SC1);
            for(unsigned int i = 0; i < recoveredIdxs.size(); i++) {
                _recoveredIdxs.getMat().ptr< int >()[i] = recoveredIdxs[i];
            }
        }
    }
}




/**
  */
int estimatePoseBoard(InputArrayOfArrays _corners, InputArray _ids, Ptr<Board> &board,
                      InputArray _cameraMatrix, InputArray _distCoeffs, OutputArray _rvec,
                      OutputArray _tvec) {

    CV_Assert(_corners.total() == _ids.total());

    // get object and image points for the solvePnP function
    Mat objPoints, imgPoints;
    _getBoardObjectAndImagePoints(board, _ids, _corners, imgPoints, objPoints);

    CV_Assert(imgPoints.total() == objPoints.total());

    if(objPoints.total() == 0) // 0 of the detected markers in board
        return 0;

    _rvec.create(3, 1, CV_64FC1);
    _tvec.create(3, 1, CV_64FC1);
    solvePnP(objPoints, imgPoints, _cameraMatrix, _distCoeffs, _rvec, _tvec);

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
Ptr<GridBoard> GridBoard::create(int markersX, int markersY, float markerLength, float markerSeparation,
                            Ptr<Dictionary> &dictionary, int firstMarker) {

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
void drawAxis(InputOutputArray _image, InputArray _cameraMatrix, InputArray _distCoeffs,
              InputArray _rvec, InputArray _tvec, float length) {

    CV_Assert(_image.getMat().total() != 0 &&
              (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
    CV_Assert(length > 0);

    // project axis points
    vector< Point3f > axisPoints;
    axisPoints.push_back(Point3f(0, 0, 0));
    axisPoints.push_back(Point3f(length, 0, 0));
    axisPoints.push_back(Point3f(0, length, 0));
    axisPoints.push_back(Point3f(0, 0, length));
    vector< Point2f > imagePoints;
    projectPoints(axisPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

    // draw axis lines
    line(_image, imagePoints[0], imagePoints[1], Scalar(0, 0, 255), 3);
    line(_image, imagePoints[0], imagePoints[2], Scalar(0, 255, 0), 3);
    line(_image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0), 3);
}



/**
 */
void drawMarker(Ptr<Dictionary> &dictionary, int id, int sidePixels, OutputArray _img, int borderBits) {
    dictionary->drawMarker(id, sidePixels, _img, borderBits);
}



void _drawPlanarBoardImpl(Board *_board, Size outSize, OutputArray _img, int marginSize,
                     int borderBits) {

    CV_Assert(outSize.area() > 0);
    CV_Assert(marginSize >= 0);

    _img.create(outSize, CV_8UC1);
    Mat out = _img.getMat();
    out.setTo(Scalar::all(255));
    Mat outNoMargins =
        out.colRange(marginSize, out.cols - marginSize).rowRange(marginSize, out.rows - marginSize);

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

    float sizeX, sizeY;
    sizeX = maxX - minX;
    sizeY = maxY - minY;

    // proportion transformations
    float xReduction = sizeX / float(outNoMargins.cols);
    float yReduction = sizeY / float(outNoMargins.rows);

    // determine the zone where the markers are placed
    Mat markerZone;
    if(xReduction > yReduction) {
        int nRows = int(sizeY / xReduction);
        int rowsMargins = (outNoMargins.rows - nRows) / 2;
        markerZone = outNoMargins.rowRange(rowsMargins, outNoMargins.rows - rowsMargins);
    } else {
        int nCols = int(sizeX / yReduction);
        int colsMargins = (outNoMargins.cols - nCols) / 2;
        markerZone = outNoMargins.colRange(colsMargins, outNoMargins.cols - colsMargins);
    }

    // now paint each marker
    Dictionary &dictionary = *(_board->dictionary);
    for(unsigned int m = 0; m < _board->objPoints.size(); m++) {

        // transform corners to markerZone coordinates
        vector< Point2f > outCorners;
        outCorners.resize(4);
        for(int j = 0; j < 4; j++) {
            Point2f p0, p1, pf;
            p0 = Point2f(_board->objPoints[m][j].x, _board->objPoints[m][j].y);
            // remove negativity
            p1.x = p0.x - minX;
            p1.y = p0.y - minY;
            pf.x = p1.x * float(markerZone.cols - 1) / sizeX;
            pf.y = float(markerZone.rows - 1) - p1.y * float(markerZone.rows - 1) / sizeY;
            outCorners[j] = pf;
        }

        // get tiny marker
        int tinyMarkerSize = 10 * dictionary.markerSize + 2;
        Mat tinyMarker;
        dictionary.drawMarker(_board->ids[m], tinyMarkerSize, tinyMarker, borderBits);

        // interpolate tiny marker to marker position in markerZone
        Mat inCorners(4, 1, CV_32FC2);
        inCorners.ptr< Point2f >(0)[0] = Point2f(0, 0);
        inCorners.ptr< Point2f >(0)[1] = Point2f((float)tinyMarker.cols, 0);
        inCorners.ptr< Point2f >(0)[2] = Point2f((float)tinyMarker.cols, (float)tinyMarker.rows);
        inCorners.ptr< Point2f >(0)[3] = Point2f(0, (float)tinyMarker.rows);

        // remove perspective
        Mat transformation = getPerspectiveTransform(inCorners, outCorners);
        Mat aux;
        const char borderValue = 127;
        warpPerspective(tinyMarker, aux, transformation, markerZone.size(), INTER_NEAREST,
                        BORDER_CONSTANT, Scalar::all(borderValue));

        // copy only not-border pixels
        for(int y = 0; y < aux.rows; y++) {
            for(int x = 0; x < aux.cols; x++) {
                if(aux.at< unsigned char >(y, x) == borderValue) continue;
                markerZone.at< unsigned char >(y, x) = aux.at< unsigned char >(y, x);
            }
        }
    }
}



/**
 */
void drawPlanarBoard(Ptr<Board> &_board, Size outSize, OutputArray _img, int marginSize,
                     int borderBits) {
    _drawPlanarBoardImpl(_board, outSize, _img, marginSize, borderBits);
}



/**
  */
double calibrateCameraAruco(InputArrayOfArrays _corners, InputArray _ids, InputArray _counter,
                            Ptr<Board> &board, Size imageSize, InputOutputArray _cameraMatrix,
                            InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs,
                            OutputArrayOfArrays _tvecs, int flags, TermCriteria criteria) {

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
        _getBoardObjectAndImagePoints(board, thisFrameIds, thisFrameCorners, currentImgPoints,
                                      currentObjPoints);
        if(currentImgPoints.total() > 0 && currentObjPoints.total() > 0) {
            processedImagePoints.push_back(currentImgPoints);
            processedObjectPoints.push_back(currentObjPoints);
        }
    }

    return calibrateCamera(processedObjectPoints, processedImagePoints, imageSize, _cameraMatrix,
                           _distCoeffs, _rvecs, _tvecs, flags, criteria);
}



}
}
