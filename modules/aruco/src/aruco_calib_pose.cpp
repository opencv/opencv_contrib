// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/aruco/aruco_calib_pose.hpp>

namespace cv {
namespace aruco {
using namespace std;

void getBoardObjectAndImagePoints(const Ptr<Board> &board, InputArrayOfArrays detectedCorners, InputArray detectedIds,
                                  OutputArray objPoints, OutputArray imgPoints) {
    CV_Assert(board->ids.size() == board->objPoints.size());
    CV_Assert(detectedIds.total() == detectedCorners.total());

    size_t nDetectedMarkers = detectedIds.total();

    vector<Point3f > objPnts;
    objPnts.reserve(nDetectedMarkers);

    vector<Point2f > imgPnts;
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
  * @brief Return object points for the system centered in a middle (by default) or in a top left corner of single
  * marker, given the marker length
  */
static Mat _getSingleMarkerObjectPoints(float markerLength, const EstimateParameters& estimateParameters) {
    CV_Assert(markerLength > 0);
    Mat objPoints(4, 1, CV_32FC3);
    // set coordinate system in the top-left corner of the marker, with Z pointing out
    if (estimateParameters.pattern == CW_top_left_corner) {
        objPoints.ptr<Vec3f>(0)[0] = Vec3f(0.f, 0.f, 0);
        objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength, 0.f, 0);
        objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength, markerLength, 0);
        objPoints.ptr<Vec3f>(0)[3] = Vec3f(0.f, markerLength, 0);
    }
    else if (estimateParameters.pattern == CCW_center) {
        objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength/2.f, markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength/2.f, markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength/2.f, -markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength/2.f, -markerLength/2.f, 0);
    }
    else
        CV_Error(Error::StsBadArg, "Unknown estimateParameters pattern");
    return objPoints;
}

void estimatePoseSingleMarkers(InputArrayOfArrays _corners, float markerLength,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArray _rvecs, OutputArray _tvecs, OutputArray _objPoints,
                               Ptr<EstimateParameters> estimateParameters) {
    CV_Assert(markerLength > 0);

    Mat markerObjPoints = _getSingleMarkerObjectPoints(markerLength, *estimateParameters);
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
                     tvecs.at<Vec3d>(i), estimateParameters->useExtrinsicGuess, estimateParameters->solvePnPMethod);
        }
    });

    if(_objPoints.needed()){
        markerObjPoints.convertTo(_objPoints, -1);
    }
}

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
  * Check if a set of 3d points are enough for calibration. Z coordinate is ignored.
  * Only axis parallel lines are considered
  */
static bool _arePointsEnoughForPoseEstimation(const vector<Point3f> &points) {
    if(points.size() < 4) return false;

    vector<double> sameXValue; // different x values in points
    vector<int> sameXCounter;  // number of points with the x value in sameXValue
    for(unsigned int i = 0; i < points.size(); i++) {
        bool found = false;
        for(unsigned int j = 0; j < sameXValue.size(); j++) {
            if(sameXValue[j] == points[i].x) {
                found = true;
                sameXCounter[j]++;
            }
        }
        if(!found) {
            sameXValue.push_back(points[i].x);
            sameXCounter.push_back(1);
        }
    }

    // count how many x values has more than 2 points
    int moreThan2 = 0;
    for(unsigned int i = 0; i < sameXCounter.size(); i++) {
        if(sameXCounter[i] >= 2) moreThan2++;
    }

    // if we have more than 1 two xvalues with more than 2 points, calibration is ok
    if(moreThan2 > 1)
        return true;
    return false;
}

bool estimatePoseCharucoBoard(InputArray _charucoCorners, InputArray _charucoIds,
                              const Ptr<CharucoBoard> &_board, InputArray _cameraMatrix, InputArray _distCoeffs,
                              InputOutputArray _rvec, InputOutputArray _tvec, bool useExtrinsicGuess) {
    CV_Assert((_charucoCorners.getMat().total() == _charucoIds.getMat().total()));

    // need, at least, 4 corners
    if(_charucoIds.getMat().total() < 4) return false;

    vector<Point3f> objPoints;
    objPoints.reserve(_charucoIds.getMat().total());
    for(unsigned int i = 0; i < _charucoIds.getMat().total(); i++) {
        int currId = _charucoIds.getMat().at< int >(i);
        CV_Assert(currId >= 0 && currId < (int)_board->chessboardCorners.size());
        objPoints.push_back(_board->chessboardCorners[currId]);
    }

    // points need to be in different lines, check if detected points are enough
    if(!_arePointsEnoughForPoseEstimation(objPoints)) return false;

    solvePnP(objPoints, _charucoCorners, _cameraMatrix, _distCoeffs, _rvec, _tvec, useExtrinsicGuess);
    return true;
}

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
    vector<Mat> processedObjectPoints, processedImagePoints;
    size_t nFrames = _counter.total();
    int markerCounter = 0;
    for(size_t frame = 0; frame < nFrames; frame++) {
        int nMarkersInThisFrame =  _counter.getMat().ptr< int >()[frame];
        vector<Mat> thisFrameCorners;
        vector<int> thisFrameIds;

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
    return calibrateCamera(processedObjectPoints, processedImagePoints, imageSize, _cameraMatrix, _distCoeffs, _rvecs,
                           _tvecs, _stdDeviationsIntrinsics, _stdDeviationsExtrinsics, _perViewErrors, flags, criteria);
}

double calibrateCameraAruco(InputArrayOfArrays _corners, InputArray _ids, InputArray _counter, const Ptr<Board> &board,
                            Size imageSize, InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                            OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags, TermCriteria criteria) {
    return calibrateCameraAruco(_corners, _ids, _counter, board, imageSize, _cameraMatrix, _distCoeffs,
                                _rvecs, _tvecs, noArray(), noArray(), noArray(), flags, criteria);
}

double calibrateCameraCharuco(InputArrayOfArrays _charucoCorners, InputArrayOfArrays _charucoIds,
                              const Ptr<CharucoBoard> &_board, Size imageSize,
                              InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                              OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                              OutputArray _stdDeviationsIntrinsics,
                              OutputArray _stdDeviationsExtrinsics,
                              OutputArray _perViewErrors,
                              int flags, TermCriteria criteria) {
    CV_Assert(_charucoIds.total() > 0 && (_charucoIds.total() == _charucoCorners.total()));

    // Join object points of charuco corners in a single vector for calibrateCamera() function
    vector<vector<Point3f> > allObjPoints;
    allObjPoints.resize(_charucoIds.total());
    for(unsigned int i = 0; i < _charucoIds.total(); i++) {
        unsigned int nCorners = (unsigned int)_charucoIds.getMat(i).total();
        CV_Assert(nCorners > 0 && nCorners == _charucoCorners.getMat(i).total());
        allObjPoints[i].reserve(nCorners);

        for(unsigned int j = 0; j < nCorners; j++) {
            int pointId = _charucoIds.getMat(i).at< int >(j);
            CV_Assert(pointId >= 0 && pointId < (int)_board->chessboardCorners.size());
            allObjPoints[i].push_back(_board->chessboardCorners[pointId]);
        }
    }
    return calibrateCamera(allObjPoints, _charucoCorners, imageSize, _cameraMatrix, _distCoeffs, _rvecs, _tvecs,
                           _stdDeviationsIntrinsics, _stdDeviationsExtrinsics, _perViewErrors, flags, criteria);
}

double calibrateCameraCharuco(InputArrayOfArrays _charucoCorners, InputArrayOfArrays _charucoIds,
                              const Ptr<CharucoBoard> &_board, Size imageSize, InputOutputArray _cameraMatrix,
                              InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                              int flags, TermCriteria criteria) {
return calibrateCameraCharuco(_charucoCorners, _charucoIds, _board, imageSize, _cameraMatrix, _distCoeffs, _rvecs,
                              _tvecs, noArray(), noArray(), noArray(), flags, criteria);
}

}
}
