// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/aruco.hpp"

namespace cv {
namespace aruco {

using namespace std;

void detectMarkers(InputArray _image, const Ptr<Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints) {
    ArucoDetector detector(_dictionary, _params);
    detector.detectMarkers(_image, _corners, _ids, _rejectedImgPoints);
}

void refineDetectedMarkers(InputArray _image, const Ptr<Board> &_board,
                           InputOutputArrayOfArrays _detectedCorners, InputOutputArray _detectedIds,
                           InputOutputArrayOfArrays _rejectedCorners, InputArray _cameraMatrix,
                           InputArray _distCoeffs, float minRepDistance, float errorCorrectionRate,
                           bool checkAllOrders, OutputArray _recoveredIdxs,
                           const Ptr<DetectorParameters> &_params) {
    Ptr<RefineParameters> refineParams = RefineParameters::create(minRepDistance, errorCorrectionRate, checkAllOrders);
    ArucoDetector detector(_board->dictionary, _params, refineParams);
    detector.refineDetectedMarkers(_image, _board, _detectedCorners, _detectedIds, _rejectedCorners, _cameraMatrix,
                                   _distCoeffs, _recoveredIdxs);
}

}
}
