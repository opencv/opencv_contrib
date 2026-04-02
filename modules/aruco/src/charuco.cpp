// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/aruco/charuco.hpp"
#include <opencv2/imgproc.hpp>

namespace cv {
namespace aruco {

using namespace std;

// BUG FIX 9: interpolateCornersCharuco
// _board deref with no null check before CharucoDetector construction.
// If caller passes an empty Ptr<CharucoBoard>, crashes on ARM.
// FIX: null guard added.  Also moved params block after null check for clarity.
int interpolateCornersCharuco(InputArrayOfArrays _markerCorners, InputArray _markerIds,
                              InputArray _image, const Ptr<CharucoBoard> &_board,
                              OutputArray _charucoCorners, OutputArray _charucoIds,
                              InputArray _cameraMatrix, InputArray _distCoeffs, int minMarkers) {
    CV_Assert(!_board.empty() && "interpolateCornersCharuco: board is null.");

    CharucoParameters params;
    params.minMarkers = minMarkers;
    params.cameraMatrix = _cameraMatrix.getMat();
    params.distCoeffs = _distCoeffs.getMat();
    CharucoDetector detector(*_board, params);
    vector<Mat> markerCorners;
    _markerCorners.getMatVector(markerCorners);
    detector.detectBoard(_image, _charucoCorners, _charucoIds, markerCorners, _markerIds.getMat());
    return (int)_charucoIds.total();
}

// BUG FIX 10: detectCharucoDiamond
// dictionary deref with no null check; crash if empty Ptr<Dictionary> is passed.
// FIX: null guard before *dictionary dereference in CharucoBoard constructor.
// Also: squareMarkerLengthRate <= 0 would produce a degenerate board — added validation.
void detectCharucoDiamond(InputArray _image, InputArrayOfArrays _markerCorners, InputArray _markerIds,
                          float squareMarkerLengthRate, OutputArrayOfArrays _diamondCorners, OutputArray _diamondIds,
                          InputArray _cameraMatrix, InputArray _distCoeffs, Ptr<Dictionary> dictionary) {
    CV_Assert(squareMarkerLengthRate > 0.f && "detectCharucoDiamond: squareMarkerLengthRate must be > 0.");

    // BUG FIX 17 (impl): null-safe fallback for dictionary default arg.
    const Ptr<Dictionary> dict = dictionary.empty() ?
        makePtr<Dictionary>(getPredefinedDictionary(PredefinedDictionaryType::DICT_4X4_50)) : dictionary;

    CharucoParameters params;
    params.cameraMatrix = _cameraMatrix.getMat();
    params.distCoeffs = _distCoeffs.getMat();
    CharucoBoard board({3, 3}, squareMarkerLengthRate, 1.f, *dict);
    CharucoDetector detector(board, params);
    vector<Mat> markerCorners;
    _markerCorners.getMatVector(markerCorners);
    detector.detectDiamonds(_image, _diamondCorners, _diamondIds, markerCorners, _markerIds.getMat());
}

// BUG FIX 11: drawCharucoDiamond
// dictionary deref with no null check.
// FIX: null guard added before *dictionary dereference in CharucoBoard constructor.
// The existing squareLength/markerLength/marginSize/borderBits assertions were correct —
// kept and placed consistently at the top.
void drawCharucoDiamond(const Ptr<Dictionary> &dictionary, Vec4i ids, int squareLength, int markerLength,
                        OutputArray _img, int marginSize, int borderBits) {
    CV_Assert(!dictionary.empty() && "drawCharucoDiamond: dictionary is null.");
    CV_Assert(squareLength > 0 && markerLength > 0 && squareLength > markerLength);
    CV_Assert(marginSize >= 0 && borderBits > 0);

    // assign the charuco marker ids
    vector<int> tmpIds(4);
    for (int i = 0; i < 4; i++)
        tmpIds[i] = ids[i];
    // create a charuco board similar to a charuco marker and print it
    CharucoBoard board(Size(3, 3), (float)squareLength, (float)markerLength, *dictionary, tmpIds);
    Size outSize(3 * squareLength + 2 * marginSize, 3 * squareLength + 2 * marginSize);
    board.generateImage(outSize, _img, marginSize, borderBits);
}

}
}
