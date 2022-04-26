// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include <opencv2/aruco_detector.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "apriltag_quad_thresh.hpp"
#include "zarray.hpp"

namespace cv {
namespace aruco {

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
      detectInvertedMarker(false),
      useAruco3Detection(false),
      minSideLengthCanonicalImg(32),
      minMarkerLengthRatioOriginalImg(0.0)
{}

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

static inline void findCornerInPyrImage(const float scale_init, const int closest_pyr_image_idx,
                                        const std::vector<cv::Mat>& grey_pyramid, Mat corners,
                                        const Ptr<DetectorParameters>& params) {
    // scale them to the closest pyramid level
    if (scale_init != 1.f)
        corners *= scale_init; // scale_init * scale_pyr
    for (int idx = closest_pyr_image_idx - 1; idx >= 0; --idx) {
        // scale them to new pyramid level
        corners *= 2.f; // *= scale_pyr;
        // use larger win size for larger images
        const int subpix_win_size = std::max(grey_pyramid[idx].cols, grey_pyramid[idx].rows) > 1080 ? 5 : 3;
        cornerSubPix(grey_pyramid[idx], corners,
                     Size(subpix_win_size, subpix_win_size),
                     Size(-1, -1),
                     TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                  params->cornerRefinementMaxIterations,
                                  params->cornerRefinementMinAccuracy));
    }
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
                                double minCornerDistanceRate, int minDistanceToBorder, int minSize) {

    CV_Assert(minPerimeterRate > 0 && maxPerimeterRate > 0 && accuracyRate > 0 &&
              minCornerDistanceRate >= 0 && minDistanceToBorder >= 0);

    // calculate maximum and minimum sizes in pixels
    unsigned int minPerimeterPixels =
        (unsigned int)(minPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));
    unsigned int maxPerimeterPixels =
        (unsigned int)(maxPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));

    // for aruco3 functionality
    if (minSize != 0) {
        minPerimeterPixels = 4*minSize;
    }

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
        bool isSingleContour = true;
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
                    isSingleContour = false;
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
        if (isSingleContour && candGroup[i] < 0)
        {
            candGroup[i] = (int)groupedCandidates.size();
            vector<unsigned int> grouped;
            grouped.push_back(i);
            grouped.push_back(i); // step "save possible candidates" require minimum 2 elements
            groupedCandidates.push_back(grouped);
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
                                params->minDistanceToBorder, params->minSideLengthCanonicalImg);
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
static void _detectCandidates(InputArray _grayImage, vector< vector< vector< Point2f > > >& candidatesSetOut,
                              vector< vector< vector< Point > > >& contoursSetOut, const Ptr<DetectorParameters> &_params) {
    Mat grey = _grayImage.getMat();
    CV_DbgAssert(grey.total() != 0);
    CV_DbgAssert(grey.type() == CV_8UC1);

    /// 1. DETECT FIRST SET OF CANDIDATES
    vector< vector< Point2f > > candidates;
    vector< vector< Point > > contours;
    _detectInitialCandidates(grey, candidates, contours, _params);
    /// 2. SORT CORNERS
    _reorderCandidatesCorners(candidates);

    /// 3. FILTER OUT NEAR CANDIDATE PAIRS
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
                                  const vector<Point2f>& _corners, int& idx,
                                  const Ptr<DetectorParameters>& params, int& rotation,
                                  const float scale = 1.f)
{
    CV_DbgAssert(_corners.size() == 4);
    CV_DbgAssert(_image.getMat().total() != 0);
    CV_DbgAssert(params->markerBorderBits > 0);
    uint8_t typ=1;
    // get bits
    // scale corners to the correct size to search on the corresponding image pyramid
    vector<Point2f> scaled_corners(4);
    for (int i = 0; i < 4; ++i) {
        scaled_corners[i].x = _corners[i].x * scale;
        scaled_corners[i].y = _corners[i].y * scale;
    }

    Mat candidateBits =
        _extractBits(_image, scaled_corners, dictionary->markerSize, params->markerBorderBits,
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
static void _copyVector2Output(vector< vector< Point2f > > &vec, OutputArrayOfArrays out, const float scale = 1.f) {
    out.create((int)vec.size(), 1, CV_32FC2);

    if(out.isMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            Mat &m = out.getMatRef(i);
            Mat(Mat(vec[i]).t()*scale).copyTo(m);
        }
    }
    else if(out.isUMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            UMat &m = out.getUMatRef(i);
            Mat(Mat(vec[i]).t()*scale).copyTo(m);
        }
    }
    else if(out.kind() == _OutputArray::STD_VECTOR_VECTOR){
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            Mat m = out.getMat(i);
            Mat(Mat(vec[i]).t()*scale).copyTo(m);
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

static size_t _findOptPyrImageForCanonicalImg(
        const std::vector<Mat>& img_pyr,
        const int scaled_width,
        const int cur_perimeter,
        const int min_perimeter) {
    CV_Assert(scaled_width > 0);
    size_t optLevel = 0;
    float dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < img_pyr.size(); ++i) {
        const float scale = img_pyr[i].cols / static_cast<float>(scaled_width);
        const float perimeter_scaled = cur_perimeter * scale;
        // instead of std::abs() favor the larger pyramid level by checking if the distance is postive
        // will slow down the algorithm but find more corners in the end
        const float new_dist = perimeter_scaled - min_perimeter;
        if (new_dist < dist && new_dist > 0.f) {
            dist = new_dist;
            optLevel = i;
        }
    }
    return optLevel;
}

/**
 * @brief Identify square candidates according to a marker dictionary
 */
static void _identifyCandidates(InputArray grey,
                                const std::vector<cv::Mat>& image_pyr,
                                vector< vector< vector< Point2f > > >& _candidatesSet,
                                vector< vector< vector<Point> > >& _contoursSet, const Ptr<Dictionary> &_dictionary,
                                vector< vector< Point2f > >& _accepted, vector< vector<Point> >& _contours, vector< int >& ids,
                                const Ptr<DetectorParameters> &params,
                                vector<vector<Point2f> > &_rejected) {
    CV_DbgAssert(grey.getMat().total() != 0);
    CV_DbgAssert(grey.getMat().type() == CV_8UC1);
    int ncandidates = (int)_candidatesSet[0].size();
    vector< vector< Point2f > > accepted;
    vector< vector< Point > > contours;

    vector< int > idsTmp(ncandidates, -1);
    vector< int > rotated(ncandidates, 0);
    vector< uint8_t > validCandidates(ncandidates, 0);

    //// Analyze each of the candidates
    parallel_for_(Range(0, ncandidates), [&](const Range &range) {
        const int begin = range.start;
        const int end = range.end;

        vector< vector< Point2f > >& candidates = params->detectInvertedMarker ? _candidatesSet[1] : _candidatesSet[0];
        vector< vector< Point > >& contourS = params->detectInvertedMarker ? _contoursSet[1] : _contoursSet[0];

        for(int i = begin; i < end; i++) {
            int currId = -1;
            // implements equation (4)
            if (params->useAruco3Detection) {
                const int perimeterOfContour = static_cast<int>(contourS[i].size());
                const int min_perimeter = params->minSideLengthCanonicalImg * 4;
                const size_t nearestImgId = _findOptPyrImageForCanonicalImg(image_pyr, grey.cols(), perimeterOfContour, min_perimeter);
                const float scale = image_pyr[nearestImgId].cols / static_cast<float>(grey.cols());

                validCandidates[i] = _identifyOneCandidate(_dictionary, image_pyr[nearestImgId], candidates[i], currId, params, rotated[i], scale);
            }
            else {
                validCandidates[i] = _identifyOneCandidate(_dictionary, grey, candidates[i], currId, params, rotated[i]);
            }

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
            _rejected.push_back(_candidatesSet[0][i]);
        }
    }

    // parse output
    _accepted = accepted;

    _contours= contours;
}

/**
 * Line fitting  A * B = C :: Called from function refineCandidateLines
 * @param nContours, contour-container
 */
static Point3f _interpolate2Dline(const std::vector<cv::Point2f>& nContours){
    CV_Assert(nContours.size() >= 2);
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

/**
 * Refine Corners using the contour vector :: Called from function detectMarkers
 * @param nContours, contour-container
 * @param nCorners, candidate Corners
 * @param camMatrix, cameraMatrix input 3x3 floating-point camera matrix
 * @param distCoeff, distCoeffs vector of distortion coefficient
 */
static void _refineCandidateLines(std::vector<Point>& nContours, std::vector<Point2f>& nCorners){
	vector<Point2f> contour2f(nContours.begin(), nContours.end());
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
    for (int i = 0; i < 4; i++)
        CV_Assert(cornerIndex[i] != -1);
	// saves extra group into corresponding
	if( !cntPts[4].empty() ){
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

Ptr<ArucoDetector> ArucoDetector::create(const Ptr<Dictionary> &_dictionary, const Ptr<DetectorParameters> &_params) {
    return makePtr<ArucoDetector>(_dictionary, _params);
}

void ArucoDetector::detectMarkers(InputArray _image, CV_OUT vector<vector<Point2f> > &_corners, vector<int> &_ids,
                                  CV_OUT vector<vector<Point2f> > &_rejectedImgPoints)
{
    CV_Assert(!_image.empty());
    CV_Assert(params->markerBorderBits > 0);
    // check that the parameters are set correctly if Aruco3 is used
    CV_Assert(!(params->useAruco3Detection == true &&
                params->minSideLengthCanonicalImg == 0 &&
                params->minMarkerLengthRatioOriginalImg == 0.0));

    Mat grey;
    _convertToGrey(_image.getMat(), grey);

    // Aruco3 functionality is the extension of Aruco.
    // The description can be found in:
    // [1] Speeded up detection of squared fiducial markers, 2018, FJ Romera-Ramirez et al.
    // if Aruco3 functionality if not wanted
    // change some parameters to be sure to turn it off
    if (!params->useAruco3Detection) {
        params->minMarkerLengthRatioOriginalImg = 0.0;
        params->minSideLengthCanonicalImg = 0;
    }
    else {
        // always turn on corner refinement in case of Aruco3, due to upsampling
        params->cornerRefinementMethod = CORNER_REFINE_SUBPIX;
        // only CORNER_REFINE_SUBPIX implement correctly for useAruco3Detection
        // TODO: update other CORNER_REFINE methods
    }

    /// Step 0: equation (2) from paper [1]
    const float fxfy = (!params->useAruco3Detection ? 1.f : params->minSideLengthCanonicalImg /
        (params->minSideLengthCanonicalImg + std::max(grey.cols, grey.rows)*params->minMarkerLengthRatioOriginalImg));

    /// Step 1: create image pyramid. Section 3.4. in [1]
    std::vector<cv::Mat> grey_pyramid;
    int closest_pyr_image_idx = 0, num_levels = 0;
    //// Step 1.1: resize image with equation (1) from paper [1]
    if (params->useAruco3Detection) {
        const float scale_pyr = 2.f;
        const float img_area = static_cast<float>(grey.rows*grey.cols);
        const float min_area_marker = static_cast<float>(params->minSideLengthCanonicalImg*params->minSideLengthCanonicalImg);
        // find max level
        num_levels = static_cast<int>(log2(img_area / min_area_marker)/scale_pyr);
        // the closest pyramid image to the downsampled segmentation image
        // will later be used as start index for corner upsampling
        const float scale_img_area = img_area * fxfy * fxfy;
        closest_pyr_image_idx = cvRound(log2(img_area / scale_img_area)/scale_pyr);
    }
    cv::buildPyramid(grey, grey_pyramid, num_levels);

    // resize to segmentation image
    // in this reduces size the contours will be detected
    if (fxfy != 1.f)
        cv::resize(grey, grey, cv::Size(cvRound(fxfy * grey.cols), cvRound(fxfy * grey.rows)));

    /// STEP 2: Detect marker candidates
    vector< vector< Point > > contours;

    vector< vector< vector< Point2f > > > candidatesSet;
    vector< vector< vector< Point > > > contoursSet;

    /// STEP 2.a Detect marker candidates :: using AprilTag
    if(params->cornerRefinementMethod == CORNER_REFINE_APRILTAG){
        _apriltag(grey, params, _corners, contours);

        candidatesSet.push_back(_corners);
        contoursSet.push_back(contours);
    }
    /// STEP 2.b Detect marker candidates :: traditional way
    else
        _detectCandidates(grey, candidatesSet, contoursSet, params);

    /// STEP 2: Check candidate codification (identify markers)
    _identifyCandidates(grey, grey_pyramid, candidatesSet, contoursSet, dictionary,
                        _corners, contours, _ids, params, _rejectedImgPoints);

    /// STEP 3: Corner refinement :: use corner subpix
    if( params->cornerRefinementMethod == CORNER_REFINE_SUBPIX ) {
        CV_Assert(params->cornerRefinementWinSize > 0 && params->cornerRefinementMaxIterations > 0 &&
                  params->cornerRefinementMinAccuracy > 0);
        // Do subpixel estimation. In Aruco3 start on the lowest pyramid level and upscale the corners
        parallel_for_(Range(0, (int)_corners.size()), [&](const Range& range) {
            const int begin = range.start;
            const int end = range.end;

            for (int i = begin; i < end; i++) {
                if (params->useAruco3Detection) {
                    const float scale_init = (float) grey_pyramid[closest_pyr_image_idx].cols / grey.cols;
                    findCornerInPyrImage(scale_init, closest_pyr_image_idx, grey_pyramid,
                                         Mat(4, 1, CV_32FC2, _corners[i].data()), params);
                }
                else
                cornerSubPix(grey, _corners[i],
                             Size(params->cornerRefinementWinSize, params->cornerRefinementWinSize),
                             Size(-1, -1),
                             TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                          params->cornerRefinementMaxIterations,
                                          params->cornerRefinementMinAccuracy));
            }
        });
    }

    /// STEP 3, Optional : Corner refinement :: use contour container
    if( params->cornerRefinementMethod == CORNER_REFINE_CONTOUR){
        if(! _ids.empty()){
            // do corner refinement using the contours for each detected markers
            parallel_for_(Range(0, (int)_corners.size()), [&](const Range& range) {
                for (int i = range.start; i < range.end; i++) {
                    _refineCandidateLines(contours[i], _corners[i]);
                }
            });
        }
    }
    if (params->cornerRefinementMethod != CORNER_REFINE_SUBPIX && fxfy != 1.f) {
        // only CORNER_REFINE_SUBPIX implement correctly for useAruco3Detection
        // TODO: update other CORNER_REFINE methods

        // scale to orignal size, this however will lead to inaccurate detections!
        for (auto &vecPoints : _corners)
            for (auto &point : vecPoints)
                point *= 1.f/fxfy;
    }
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

void ArucoDetector::refineDetectedMarkers(const _InputArray &_image, const Ptr<Board> &_board,
                                          const _InputOutputArray &_detectedCorners,
                                          const _InputOutputArray &_detectedIds,
                                          const _InputOutputArray &_rejectedCorners,
                                          const _InputArray &_cameraMatrix, const _InputArray &_distCoeffs,
                                          float minRepDistance, float errorCorrectionRate, bool checkAllOrders,
                                          const _OutputArray &_recoveredIdxs) {
    CV_Assert(minRepDistance > 0);

    if(_detectedIds.total() == 0 || _rejectedCorners.total() == 0) return;

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
    int maxCorrectionRecalculated =
        int(double(dictionary->maxCorrectionBits) * errorCorrectionRate);

    Mat grey;
    _convertToGrey(_image, grey);

    // vector of final detected marker corners and ids
    vector<vector<Point2f> > finalAcceptedCorners;
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
                    grey, rotatedMarker, dictionary->markerSize, params->markerBorderBits,
                    params->perspectiveRemovePixelPerCell,
                    params->perspectiveRemoveIgnoredMarginPerCell, params->minOtsuStdDev);

                Mat onlyBits =
                    bits.rowRange(params->markerBorderBits, bits.rows - params->markerBorderBits)
                        .colRange(params->markerBorderBits, bits.rows - params->markerBorderBits);

                codeDistance =
                    dictionary->getDistanceToId(onlyBits, undetectedMarkersIds[i], false);
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
            if(params->cornerRefinementMethod == CORNER_REFINE_SUBPIX) {
                CV_Assert(params->cornerRefinementWinSize > 0 &&
                          params->cornerRefinementMaxIterations > 0 &&
                          params->cornerRefinementMinAccuracy > 0);
                cornerSubPix(grey, closestRotatedMarker,
                             Size(params->cornerRefinementWinSize, params->cornerRefinementWinSize),
                             Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                                        params->cornerRefinementMaxIterations,
                                                        params->cornerRefinementMinAccuracy));
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
        // parse output
        Mat(finalAcceptedIds).copyTo(_detectedIds);
        _copyVector2Output(finalAcceptedCorners, _detectedCorners);

        // recalculate _rejectedCorners based on alreadyIdentified
        vector<vector<Point2f> > finalRejected;
        for(unsigned int i = 0; i < alreadyIdentified.size(); i++) {
            if(!alreadyIdentified[i]) {
                finalRejected.push_back(_rejectedCorners.getMat(i).clone());
            }
        }
        _copyVector2Output(finalRejected, _rejectedCorners);

        if(_recoveredIdxs.needed()) {
            Mat(recoveredIdxs).copyTo(_recoveredIdxs);
        }
    }
}

    }
}
