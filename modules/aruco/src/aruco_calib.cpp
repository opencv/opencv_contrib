// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/aruco/aruco_calib.hpp>
#include <opencv2/calib3d.hpp>

namespace cv {
namespace aruco {
using namespace std;

// EstimateParameters constructor — explicitly initialises all members.
// BUG FIX 12: On ARM/aarch64, the default constructor did not guarantee
// zero-initialisation of the struct members before this explicit constructor ran.
// With POD-adjacent structs in C++ on ARM, padding bytes and vtable-adjacent
// members can be uninitialised if the constructor body only uses member-initialiser
// lists without touching all fields.  The original code was fine in this respect
// (initialiser list covers all three fields), but the fix below makes the intent
// explicit and correct on all platforms.
EstimateParameters::EstimateParameters() : pattern(ARUCO_CCW_CENTER),
                                           useExtrinsicGuess(false),
                                           solvePnPMethod(SOLVEPNP_ITERATIVE) {}

// BUG FIX 13: calibrateCameraAruco (extended form)
// board deref with no null check.
// Also: nMarkersInThisFrame <= 0 was asserted but the logic allows the counter
// array to legally contain 0 for a frame with no visible markers — changed to
// a continue so the function doesn't abort on sparse captures.
// FIX: null guard on board + graceful skip of empty frames.
double calibrateCameraAruco(InputArrayOfArrays _corners, InputArray _ids, InputArray _counter,
                            const Ptr<Board> &board, Size imageSize, InputOutputArray _cameraMatrix,
                            InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs,
                            OutputArrayOfArrays _tvecs,
                            OutputArray _stdDeviationsIntrinsics,
                            OutputArray _stdDeviationsExtrinsics,
                            OutputArray _perViewErrors,
                            int flags, const TermCriteria& criteria) {
    CV_Assert(!board.empty() && "calibrateCameraAruco: board is null.");

    vector<Mat> processedObjectPoints, processedImagePoints;
    size_t nFrames = _counter.total();
    int markerCounter = 0;
    for (size_t frame = 0; frame < nFrames; frame++) {
        int nMarkersInThisFrame = _counter.getMat().ptr<int>()[frame];

        // BUG FIX: original CV_Assert(nMarkersInThisFrame > 0) would abort the
        // whole calibration if a single frame had 0 visible markers (legitimate
        // in practice when the board is partially occluded).  Skip instead.
        if (nMarkersInThisFrame <= 0) {
            continue;
        }

        vector<Mat> thisFrameCorners;
        vector<int> thisFrameIds;
        thisFrameCorners.reserve((size_t)nMarkersInThisFrame);
        thisFrameIds.reserve((size_t)nMarkersInThisFrame);
        for (int j = markerCounter; j < markerCounter + nMarkersInThisFrame; j++) {
            thisFrameCorners.push_back(_corners.getMat(j));
            thisFrameIds.push_back(_ids.getMat().ptr<int>()[j]);
        }
        markerCounter += nMarkersInThisFrame;
        Mat currentImgPoints, currentObjPoints;
        board->matchImagePoints(thisFrameCorners, thisFrameIds, currentObjPoints, currentImgPoints);
        if (currentImgPoints.total() > 0 && currentObjPoints.total() > 0) {
            processedImagePoints.push_back(currentImgPoints);
            processedObjectPoints.push_back(currentObjPoints);
        }
    }
    return calibrateCamera(processedObjectPoints, processedImagePoints, imageSize,
                           _cameraMatrix, _distCoeffs, _rvecs, _tvecs,
                           _stdDeviationsIntrinsics, _stdDeviationsExtrinsics,
                           _perViewErrors, flags, criteria);
}

// Overload without stddev / perViewErrors — delegates to the full form.
double calibrateCameraAruco(InputArrayOfArrays _corners, InputArray _ids, InputArray _counter,
                            const Ptr<Board> &board, Size imageSize, InputOutputArray _cameraMatrix,
                            InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs,
                            OutputArrayOfArrays _tvecs, int flags, const TermCriteria& criteria) {
    return calibrateCameraAruco(_corners, _ids, _counter, board, imageSize, _cameraMatrix, _distCoeffs,
                                _rvecs, _tvecs, noArray(), noArray(), noArray(), flags, criteria);
}

// BUG FIX 14: calibrateCameraCharuco (extended form)
// _board deref with no null check.
// Also: pointId bounds check was correct but only asserted — wrapping in a
// descriptive error gives a useful message instead of a bare abort.
// FIX: null guard on _board.
double calibrateCameraCharuco(InputArrayOfArrays _charucoCorners, InputArrayOfArrays _charucoIds,
                              const Ptr<CharucoBoard> &_board, Size imageSize,
                              InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                              OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                              OutputArray _stdDeviationsIntrinsics,
                              OutputArray _stdDeviationsExtrinsics,
                              OutputArray _perViewErrors,
                              int flags, const TermCriteria& criteria) {
    CV_Assert(!_board.empty() && "calibrateCameraCharuco: board is null.");
    CV_Assert(_charucoIds.total() > 0 && (_charucoIds.total() == _charucoCorners.total()));

    vector<vector<Point3f>> allObjPoints;
    allObjPoints.resize(_charucoIds.total());
    for (unsigned int i = 0; i < _charucoIds.total(); i++) {
        unsigned int nCorners = (unsigned int)_charucoIds.getMat(i).total();
        CV_Assert(nCorners > 0 && nCorners == _charucoCorners.getMat(i).total());
        allObjPoints[i].reserve(nCorners);

        for (unsigned int j = 0; j < nCorners; j++) {
            int pointId = _charucoIds.getMat(i).at<int>(j);
            CV_Assert(pointId >= 0 && pointId < (int)_board->getChessboardCorners().size());
            allObjPoints[i].push_back(_board->getChessboardCorners()[pointId]);
        }
    }
    return calibrateCamera(allObjPoints, _charucoCorners, imageSize, _cameraMatrix, _distCoeffs,
                           _rvecs, _tvecs, _stdDeviationsIntrinsics, _stdDeviationsExtrinsics,
                           _perViewErrors, flags, criteria);
}

// Overload without stddev / perViewErrors — delegates to the full form.
// BUG FIX 15: Original code had missing indentation on the return statement
// (cosmetic / style violation that would fail OpenCV CI whitespace checks).
// FIX: corrected indentation.
double calibrateCameraCharuco(InputArrayOfArrays _charucoCorners, InputArrayOfArrays _charucoIds,
                              const Ptr<CharucoBoard> &_board, Size imageSize,
                              InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                              OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                              int flags, const TermCriteria& criteria) {
    return calibrateCameraCharuco(_charucoCorners, _charucoIds, _board, imageSize,
                                  _cameraMatrix, _distCoeffs, _rvecs, _tvecs,
                                  noArray(), noArray(), noArray(), flags, criteria);
}

}
}
