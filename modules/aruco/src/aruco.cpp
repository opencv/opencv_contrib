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

#include "zarray.hpp"

namespace cv {
namespace aruco {

using namespace std;

/**
  * @brief Create a new set of DetectorParameters with default values.
  */
Ptr<DetectorParameters> DetectorParameters::create() {
    Ptr<DetectorParameters> params = makePtr<DetectorParameters>();
    return params;
}

template<typename T>
static inline bool readParameter(const FileNode& node, T& parameter)
{
    if (!node.empty()) {
        node >> parameter;
        return true;
    }
    return false;
}

/**
  * @brief Read a new set of DetectorParameters from FileStorage.
  */
bool DetectorParameters::readDetectorParameters(const FileNode& fn)
{
    if(fn.empty())
        return true;
    bool checkRead = false;
    checkRead |= readParameter(fn["adaptiveThreshWinSizeMin"], this->adaptiveThreshWinSizeMin);
    checkRead |= readParameter(fn["adaptiveThreshWinSizeMax"], this->adaptiveThreshWinSizeMax);
    checkRead |= readParameter(fn["adaptiveThreshWinSizeStep"], this->adaptiveThreshWinSizeStep);
    checkRead |= readParameter(fn["adaptiveThreshConstant"], this->adaptiveThreshConstant);
    checkRead |= readParameter(fn["minMarkerPerimeterRate"], this->minMarkerPerimeterRate);
    checkRead |= readParameter(fn["maxMarkerPerimeterRate"], this->maxMarkerPerimeterRate);
    checkRead |= readParameter(fn["polygonalApproxAccuracyRate"], this->polygonalApproxAccuracyRate);
    checkRead |= readParameter(fn["minCornerDistanceRate"], this->minCornerDistanceRate);
    checkRead |= readParameter(fn["minDistanceToBorder"], this->minDistanceToBorder);
    checkRead |= readParameter(fn["minMarkerDistanceRate"], this->minMarkerDistanceRate);
    checkRead |= readParameter(fn["cornerRefinementMethod"], this->cornerRefinementMethod);
    checkRead |= readParameter(fn["cornerRefinementWinSize"], this->cornerRefinementWinSize);
    checkRead |= readParameter(fn["cornerRefinementMaxIterations"], this->cornerRefinementMaxIterations);
    checkRead |= readParameter(fn["cornerRefinementMinAccuracy"], this->cornerRefinementMinAccuracy);
    checkRead |= readParameter(fn["markerBorderBits"], this->markerBorderBits);
    checkRead |= readParameter(fn["perspectiveRemovePixelPerCell"], this->perspectiveRemovePixelPerCell);
    checkRead |= readParameter(fn["perspectiveRemoveIgnoredMarginPerCell"], this->perspectiveRemoveIgnoredMarginPerCell);
    checkRead |= readParameter(fn["maxErroneousBitsInBorderRate"], this->maxErroneousBitsInBorderRate);
    checkRead |= readParameter(fn["minOtsuStdDev"], this->minOtsuStdDev);
    checkRead |= readParameter(fn["errorCorrectionRate"], this->errorCorrectionRate);
    // new aruco 3 functionality
    checkRead |= readParameter(fn["useAruco3Detection"], this->useAruco3Detection);
    checkRead |= readParameter(fn["minSideLengthCanonicalImg"], this->minSideLengthCanonicalImg);
    checkRead |= readParameter(fn["minMarkerLengthRatioOriginalImg"], this->minMarkerLengthRatioOriginalImg);
    return checkRead;
}

/**
  * @brief Return object points for the system centered in a single marker, given the marker length
  */
static void _getSingleMarkerObjectPoints(float markerLength, OutputArray _objPoints) {

    CV_Assert(markerLength > 0);

    _objPoints.create(4, 1, CV_32FC3);
    Mat objPoints = _objPoints.getMat();
    // set coordinate system in the top-left corner of the marker, with Z pointing out
    objPoints.ptr< Vec3f >(0)[0] = Vec3f(0.f, 0.f, 0);
    objPoints.ptr< Vec3f >(0)[1] = Vec3f(markerLength, 0.f, 0);
    objPoints.ptr< Vec3f >(0)[2] = Vec3f(markerLength, markerLength, 0);
    objPoints.ptr< Vec3f >(0)[3] = Vec3f(0.f, markerLength, 0);
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

static vector<vector<Point2f>> getVectors(InputArrayOfArrays& in)
{
    vector<vector<Point2f>> v;
    if (in.isMatVector() || in.kind())
    {
        for (size_t i = 0; i < in.total(); i++)
        {
            Mat m = in.getMat(i);
            vector<Point2f> tmp;
            m.copyTo(tmp);
            v.push_back(tmp);
        }
    }
    else if (in.isUMatVector())
    {
        for (size_t i = 0; i < in.total(); i++) {
            UMat m = in.getUMat(i);
            vector<Point2f> tmp;
            m.copyTo(tmp);
            v.push_back(tmp);
        }
    }
    else if (in.kind() == _InputArray::STD_VECTOR_MAT) {
        for (size_t i = 0; i < in.total(); i++) {
            Mat m = in.getMat(i).reshape(2);
            vector<Point2f> tmp;
            m.copyTo(tmp);
            v.push_back(tmp);
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> InputArrays are currently supported.");
    }
    return v;
}

/**
  */
void detectMarkers(InputArray _image, const Ptr<Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints) {
    ArucoDetector detector(_dictionary, _params);
    vector<vector<Point2f>> corners;
    vector<vector<Point2f>> rejectedImgPoints;
    vector<int> ids;

    detector.detectMarkers(_image, corners, ids, rejectedImgPoints);

    _copyVector2Output(corners, _corners);
    if (_rejectedImgPoints.needed())
        _copyVector2Output(rejectedImgPoints, _rejectedImgPoints);
    Mat(ids).copyTo(_ids);
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
  */
void refineDetectedMarkers(InputArray _image, const Ptr<Board> &_board,
                           InputOutputArrayOfArrays _detectedCorners, InputOutputArray _detectedIds,
                           InputOutputArrayOfArrays _rejectedCorners, InputArray _cameraMatrix,
                           InputArray _distCoeffs, float minRepDistance, float errorCorrectionRate,
                           bool checkAllOrders, OutputArray _recoveredIdxs,
                           const Ptr<DetectorParameters> &_params){
    Ptr<RefineParameters> refineParams = RefineParameters::create(minRepDistance, errorCorrectionRate, checkAllOrders);
    ArucoDetector detector(_board->dictionary, _params, refineParams);

    vector<vector<Point2f> > detectedCorners = move(getVectors(_detectedCorners));
    vector<vector<Point2f> > rejectedCorners = move(getVectors(_rejectedCorners));

    vector<int> detectedIds;
    _detectedIds.copyTo(detectedIds);

    vector<int> recoveredIdxs;
    detector.refineDetectedMarkers(_image, _board, detectedCorners, detectedIds, rejectedCorners,
                                   recoveredIdxs, _cameraMatrix, _distCoeffs);

    _copyVector2Output(detectedCorners, _detectedCorners);
    Mat(detectedIds).copyTo(_detectedIds);
    _copyVector2Output(rejectedCorners, _rejectedCorners);
    if (_recoveredIdxs.needed())
        Mat(recoveredIdxs).copyTo(_recoveredIdxs);
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
    Point3f rightBottomBorder = Point3f(0.f, 0.f, 0.f);
    for (unsigned int i = 0; i < objPoints.total(); i++) {
        std::vector<Point3f> corners;
        Mat corners_mat = objPoints.getMat(i);

        if(corners_mat.type() == CV_32FC1)
            corners_mat = corners_mat.reshape(3);
        CV_Assert(corners_mat.total() == 4);

        for (int j = 0; j < 4; j++) {
            const Point3f& corner = corners_mat.at<Point3f>(j);
            corners.push_back(corner);
            rightBottomBorder.x = std::max(rightBottomBorder.x, corner.x);
            rightBottomBorder.y = std::max(rightBottomBorder.y, corner.y);
            rightBottomBorder.z = std::max(rightBottomBorder.z, corner.z);
        }
        obj_points_vector.push_back(corners);
    }

    Ptr<Board> res = makePtr<Board>();
    ids.copyTo(res->ids);
    res->objPoints = obj_points_vector;
    res->dictionary = cv::makePtr<Dictionary>(dictionary);
    res->rightBottomBorder = rightBottomBorder;
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
    for(int y = 0; y < markersY; y++) {
        for(int x = 0; x < markersX; x++) {
            vector<Point3f> corners(4);
            corners[0] = Point3f(x * (markerLength + markerSeparation),
                                 y * (markerLength + markerSeparation), 0);
            corners[1] = corners[0] + Point3f(markerLength, 0, 0);
            corners[2] = corners[0] + Point3f(markerLength, markerLength, 0);
            corners[3] = corners[0] + Point3f(0, markerLength, 0);
            res->objPoints.push_back(corners);
        }
    }
    res->rightBottomBorder = Point3f(markersX * markerLength + markerSeparation * (markersX - 1),
                                     markersY * markerLength + markerSeparation * (markersY - 1), 0.f);
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
            pf.y = pf.y / sizeY * float(out.rows);
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
