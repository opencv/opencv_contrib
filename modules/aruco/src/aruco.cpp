// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/aruco.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace aruco {

using namespace std;

// BUG FIX 1: detectMarkers
// Original code dereferenced _dictionary and _params with no null check.
// On ARM/aarch64 (Raspberry Pi 5), a null/uninitialised Ptr<> dereference causes
// SIGSEGV at address 0x60.  The new-API objects (getPredefinedDictionary /
// DetectorParameters()) can arrive here as empty Ptr<> on ARM builds.
// FIX: Use Ptr<>::empty() guards (safer than operator bool on all OpenCV builds).
void detectMarkers(InputArray _image, const Ptr<Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints) {
    CV_Assert(!_dictionary.empty() && "detectMarkers: dictionary is null. "
              "Pass a valid Dictionary from getPredefinedDictionary() or Dictionary_get().");

    // BUG FIX 17 (impl): When parameters default arg is Ptr<>() (null), create a
    // default DetectorParameters internally rather than crashing.  This is the safe
    // fallback for the ARM-safe null-default header signature.
    const Ptr<DetectorParameters> params = _params.empty() ? makePtr<DetectorParameters>() : _params;

    ArucoDetector detector(*_dictionary, *params);
    detector.detectMarkers(_image, _corners, _ids, _rejectedImgPoints);
}

// BUG FIX 2: refineDetectedMarkers
// Same ARM null-deref crash pattern for _board and _params.
// FIX: Null guards on both Ptr<> arguments before any dereference.
void refineDetectedMarkers(InputArray _image, const Ptr<Board> &_board,
                           InputOutputArrayOfArrays _detectedCorners, InputOutputArray _detectedIds,
                           InputOutputArrayOfArrays _rejectedCorners, InputArray _cameraMatrix,
                           InputArray _distCoeffs, float minRepDistance, float errorCorrectionRate,
                           bool checkAllOrders, OutputArray _recoveredIdxs,
                           const Ptr<DetectorParameters> &_params) {
    CV_Assert(!_board.empty() && "refineDetectedMarkers: board is null.");

    // BUG FIX 17 (impl): null-safe fallback for _params default arg.
    const Ptr<DetectorParameters> params = _params.empty() ? makePtr<DetectorParameters>() : _params;

    RefineParameters refineParams(minRepDistance, errorCorrectionRate, checkAllOrders);
    ArucoDetector detector(_board->getDictionary(), *params, refineParams);
    detector.refineDetectedMarkers(_image, *_board, _detectedCorners, _detectedIds, _rejectedCorners,
                                   _cameraMatrix, _distCoeffs, _recoveredIdxs);
}

// BUG FIX 3: drawPlanarBoard
// board deref with no null check.  Crashes if caller passes an empty Ptr<Board>.
// FIX: null guard added.
void drawPlanarBoard(const Ptr<Board> &board, Size outSize, const _OutputArray &img, int marginSize, int borderBits) {
    CV_Assert(!board.empty() && "drawPlanarBoard: board is null.");
    board->generateImage(outSize, img, marginSize, borderBits);
}

// BUG FIX 4: getBoardObjectAndImagePoints
// board deref with no null check.
// FIX: null guard added.
void getBoardObjectAndImagePoints(const Ptr<Board> &board, InputArrayOfArrays detectedCorners, InputArray detectedIds,
                                  OutputArray objPoints, OutputArray imgPoints) {
    CV_Assert(!board.empty() && "getBoardObjectAndImagePoints: board is null.");
    board->matchImagePoints(detectedCorners, detectedIds, objPoints, imgPoints);
}

// BUG FIX 5: estimatePoseBoard
// (a) board deref with no null check.
// (b) solvePnP not wrapped in try/catch — any PnP failure propagates as unhandled
//     exception and terminates the process on Raspberry Pi.
// FIX: null guard + try/catch around solvePnP; return 0 on PnP failure.
int estimatePoseBoard(InputArrayOfArrays corners, InputArray ids, const Ptr<Board> &board,
                      InputArray cameraMatrix, InputArray distCoeffs, InputOutputArray rvec,
                      InputOutputArray tvec, bool useExtrinsicGuess) {
    CV_Assert(!board.empty() && "estimatePoseBoard: board is null.");
    CV_Assert(corners.total() == ids.total());

    Mat objPoints, imgPoints;
    board->matchImagePoints(corners, ids, objPoints, imgPoints);

    CV_Assert(imgPoints.total() == objPoints.total());

    if (objPoints.total() == 0) // 0 of the detected markers in board
        return 0;

    try {
        solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess);
    }
    catch (const cv::Exception& e) {
        CV_LOG_WARNING(NULL, "estimatePoseBoard: solvePnP failed: " << e.what());
        return 0;
    }

    // divide by four since all four corners are concatenated in the array for each marker
    return (int)objPoints.total() / 4;
}

// BUG FIX 6: estimatePoseCharucoBoard
// board deref with no null check.
// FIX: null guard added before board->matchImagePoints().
bool estimatePoseCharucoBoard(InputArray charucoCorners, InputArray charucoIds,
                              const Ptr<CharucoBoard> &board, InputArray cameraMatrix,
                              InputArray distCoeffs, InputOutputArray rvec,
                              InputOutputArray tvec, bool useExtrinsicGuess) {
    CV_Assert(!board.empty() && "estimatePoseCharucoBoard: board is null.");
    CV_Assert((charucoCorners.getMat().total() == charucoIds.getMat().total()));

    if (charucoIds.getMat().total() < 4) return false;

    Mat objPoints, imgPoints;
    board->matchImagePoints(charucoCorners, charucoIds, objPoints, imgPoints);
    try {
        solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess);
    }
    catch (const cv::Exception& e) {
        CV_LOG_WARNING(NULL, "estimatePoseCharucoBoard: " << std::endl << e.what());
        return false;
    }

    return objPoints.total() > 0ull;
}

// BUG FIX 7: testCharucoCornersCollinear
// board deref with no null check.
// FIX: null guard added.
bool testCharucoCornersCollinear(const Ptr<CharucoBoard> &board, InputArray charucoIds) {
    CV_Assert(!board.empty() && "testCharucoCornersCollinear: board is null.");
    return board->checkCharucoCornersCollinear(charucoIds);
}

/**
 * @brief Return object points for the system centered in the middle (default) or in the top-left
 * corner of a single marker, given the marker length.
 */
static Mat _getSingleMarkerObjectPoints(float markerLength, const EstimateParameters& estimateParameters) {
    CV_Assert(markerLength > 0);
    Mat objPoints(4, 1, CV_32FC3);
    // set coordinate system in the top-left corner of the marker, with Z pointing out
    if (estimateParameters.pattern == ARUCO_CW_TOP_LEFT_CORNER) {
        objPoints.ptr<Vec3f>(0)[0] = Vec3f(0.f, 0.f, 0);
        objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength, 0.f, 0);
        objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength, markerLength, 0);
        objPoints.ptr<Vec3f>(0)[3] = Vec3f(0.f, markerLength, 0);
    }
    else if (estimateParameters.pattern == ARUCO_CCW_CENTER) {
        objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength/2.f, markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength/2.f, markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength/2.f, -markerLength/2.f, 0);
        objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength/2.f, -markerLength/2.f, 0);
    }
    else {
        CV_Error(Error::StsBadArg, "Unknown estimateParameters pattern");
    }
    return objPoints;
}

// BUG FIX 8: estimatePoseSingleMarkers
// estimateParameters deref inside parallel_for_ lambda with no null check.
// The default argument makePtr<EstimateParameters>() can return an empty Ptr on
// ARM builds, causing a silent crash inside the parallel lambda — extremely hard
// to debug because the stack unwinds across thread boundaries.
// FIX: Null guard BEFORE parallel_for_ is entered.
void estimatePoseSingleMarkers(InputArrayOfArrays _corners, float markerLength,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArray _rvecs, OutputArray _tvecs, OutputArray _objPoints,
                               const Ptr<EstimateParameters>& estimateParameters) {
    CV_Assert(markerLength > 0);

    // BUG FIX 17 (impl): null-safe fallback for estimateParameters default arg.
    const Ptr<EstimateParameters> ep = estimateParameters.empty() ? makePtr<EstimateParameters>() : estimateParameters;

    Mat markerObjPoints = _getSingleMarkerObjectPoints(markerLength, *ep);
    int nMarkers = (int)_corners.total();
    _rvecs.create(nMarkers, 1, CV_64FC3);
    _tvecs.create(nMarkers, 1, CV_64FC3);

    Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();

    // for each marker, calculate its pose
    parallel_for_(Range(0, nMarkers), [&](const Range& range) {
        const int begin = range.start;
        const int end = range.end;

        for (int i = begin; i < end; i++) {
            solvePnP(markerObjPoints, _corners.getMat(i), _cameraMatrix, _distCoeffs,
                     rvecs.at<Vec3d>(i), tvecs.at<Vec3d>(i),
                     ep->useExtrinsicGuess,
                     ep->solvePnPMethod);
        }
    });

    if (_objPoints.needed()) {
        markerObjPoints.convertTo(_objPoints, -1);
    }
}

}
}
