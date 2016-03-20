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
#include "opencv2/aruco/charuco.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


namespace cv {
namespace aruco {

using namespace std;



/**
 */
void CharucoBoard::draw(Size outSize, OutputArray _img, int marginSize, int borderBits) {

    CV_Assert(outSize.area() > 0);
    CV_Assert(marginSize >= 0);

    _img.create(outSize, CV_8UC1);
    _img.setTo(255);
    Mat out = _img.getMat();
    Mat noMarginsImg =
        out.colRange(marginSize, out.cols - marginSize).rowRange(marginSize, out.rows - marginSize);

    double totalLengthX, totalLengthY;
    totalLengthX = _squareLength * _squaresX;
    totalLengthY = _squareLength * _squaresY;

    // proportional transformation
    double xReduction = totalLengthX / double(noMarginsImg.cols);
    double yReduction = totalLengthY / double(noMarginsImg.rows);

    // determine the zone where the chessboard is placed
    Mat chessboardZoneImg;
    if(xReduction > yReduction) {
        int nRows = int(totalLengthY / xReduction);
        int rowsMargins = (noMarginsImg.rows - nRows) / 2;
        chessboardZoneImg = noMarginsImg.rowRange(rowsMargins, noMarginsImg.rows - rowsMargins);
    } else {
        int nCols = int(totalLengthX / yReduction);
        int colsMargins = (noMarginsImg.cols - nCols) / 2;
        chessboardZoneImg = noMarginsImg.colRange(colsMargins, noMarginsImg.cols - colsMargins);
    }

    // determine the margins to draw only the markers
    // take the minimum just to be sure
    double squareSizePixels = min(double(chessboardZoneImg.cols) / double(_squaresX),
                                  double(chessboardZoneImg.rows) / double(_squaresY));

    double diffSquareMarkerLength = (_squareLength - _markerLength) / 2;
    int diffSquareMarkerLengthPixels =
        int(diffSquareMarkerLength * squareSizePixels / _squareLength);

    // draw markers
    Mat markersImg;
    aruco::_drawPlanarBoardImpl(this, chessboardZoneImg.size(), markersImg,
                                diffSquareMarkerLengthPixels, borderBits);

    markersImg.copyTo(chessboardZoneImg);

    // now draw black squares
    for(int y = 0; y < _squaresY; y++) {
        for(int x = 0; x < _squaresX; x++) {

            if(y % 2 != x % 2) continue; // white corner, dont do anything

            double startX, startY;
            startX = squareSizePixels * double(x);
            startY = double(chessboardZoneImg.rows) - squareSizePixels * double(y + 1);

            Mat squareZone = chessboardZoneImg.rowRange(int(startY), int(startY + squareSizePixels))
                                 .colRange(int(startX), int(startX + squareSizePixels));

            squareZone.setTo(0);
        }
    }
}



/**
 */
Ptr<CharucoBoard> CharucoBoard::create(int squaresX, int squaresY, float squareLength,
                                  float markerLength, Ptr<Dictionary> &dictionary) {

    CV_Assert(squaresX > 1 && squaresY > 1 && markerLength > 0 && squareLength > markerLength);
    Ptr<CharucoBoard> res = makePtr<CharucoBoard>();

    res->_squaresX = squaresX;
    res->_squaresY = squaresY;
    res->_squareLength = squareLength;
    res->_markerLength = markerLength;
    res->dictionary = dictionary;

    float diffSquareMarkerLength = (squareLength - markerLength) / 2;

    // calculate Board objPoints
    for(int y = squaresY - 1; y >= 0; y--) {
        for(int x = 0; x < squaresX; x++) {

            if(y % 2 == x % 2) continue; // black corner, no marker here

            vector< Point3f > corners;
            corners.resize(4);
            corners[0] = Point3f(x * squareLength + diffSquareMarkerLength,
                                 y * squareLength + diffSquareMarkerLength + markerLength, 0);
            corners[1] = corners[0] + Point3f(markerLength, 0, 0);
            corners[2] = corners[0] + Point3f(markerLength, -markerLength, 0);
            corners[3] = corners[0] + Point3f(0, -markerLength, 0);
            res->objPoints.push_back(corners);
            // first ids in dictionary
            int nextId = (int)res->ids.size();
            res->ids.push_back(nextId);
        }
    }

    // now fill chessboardCorners
    for(int y = 0; y < squaresY - 1; y++) {
        for(int x = 0; x < squaresX - 1; x++) {
            Point3f corner;
            corner.x = (x + 1) * squareLength;
            corner.y = (y + 1) * squareLength;
            corner.z = 0;
            res->chessboardCorners.push_back(corner);
        }
    }

    res->_getNearestMarkerCorners();

    return res;
}



/**
  * Fill nearestMarkerIdx and nearestMarkerCorners arrays
  */
void CharucoBoard::_getNearestMarkerCorners() {

    nearestMarkerIdx.resize(chessboardCorners.size());
    nearestMarkerCorners.resize(chessboardCorners.size());

    unsigned int nMarkers = (unsigned int)ids.size();
    unsigned int nCharucoCorners = (unsigned int)chessboardCorners.size();
    for(unsigned int i = 0; i < nCharucoCorners; i++) {
        double minDist = -1; // distance of closest markers
        Point3f charucoCorner = chessboardCorners[i];
        for(unsigned int j = 0; j < nMarkers; j++) {
            // calculate distance from marker center to charuco corner
            Point3f center = Point3f(0, 0, 0);
            for(unsigned int k = 0; k < 4; k++)
                center += objPoints[j][k];
            center /= 4.;
            double sqDistance;
            Point3f distVector = charucoCorner - center;
            sqDistance = distVector.x * distVector.x + distVector.y * distVector.y;
            if(j == 0 || fabs(sqDistance - minDist) < 0.0001) {
                // if same minimum distance (or first iteration), add to nearestMarkerIdx vector
                nearestMarkerIdx[i].push_back(j);
                minDist = sqDistance;
            } else if(sqDistance < minDist) {
                // if finding a closest marker to the charuco corner
                nearestMarkerIdx[i].clear(); // remove any previous added marker
                nearestMarkerIdx[i].push_back(j); // add the new closest marker index
                minDist = sqDistance;
            }
        }

        // for each of the closest markers, search the marker corner index closer
        // to the charuco corner
        for(unsigned int j = 0; j < nearestMarkerIdx[i].size(); j++) {
            nearestMarkerCorners[i].resize(nearestMarkerIdx[i].size());
            double minDistCorner = -1;
            for(unsigned int k = 0; k < 4; k++) {
                double sqDistance;
                Point3f distVector = charucoCorner - objPoints[nearestMarkerIdx[i][j]][k];
                sqDistance = distVector.x * distVector.x + distVector.y * distVector.y;
                if(k == 0 || sqDistance < minDistCorner) {
                    // if this corner is closer to the charuco corner, assing its index
                    // to nearestMarkerCorners
                    minDistCorner = sqDistance;
                    nearestMarkerCorners[i][j] = k;
                }
            }
        }
    }
}


/**
  * Remove charuco corners if any of their minMarkers closest markers has not been detected
  */
static unsigned int _filterCornersWithoutMinMarkers(Ptr<CharucoBoard> &_board,
                                                    InputArray _allCharucoCorners,
                                                    InputArray _allCharucoIds,
                                                    InputArray _allArucoIds, int minMarkers,
                                                    OutputArray _filteredCharucoCorners,
                                                    OutputArray _filteredCharucoIds) {

    CV_Assert(minMarkers >= 0 && minMarkers <= 2);

    vector< Point2f > filteredCharucoCorners;
    vector< int > filteredCharucoIds;
    // for each charuco corner
    for(unsigned int i = 0; i < _allCharucoIds.getMat().total(); i++) {
        int currentCharucoId = _allCharucoIds.getMat().ptr< int >(0)[i];
        int totalMarkers = 0; // nomber of closest marker detected
        // look for closest markers
        for(unsigned int m = 0; m < _board->nearestMarkerIdx[currentCharucoId].size(); m++) {
            int markerId = _board->ids[_board->nearestMarkerIdx[currentCharucoId][m]];
            bool found = false;
            for(unsigned int k = 0; k < _allArucoIds.getMat().total(); k++) {
                if(_allArucoIds.getMat().ptr< int >(0)[k] == markerId) {
                    found = true;
                    break;
                }
            }
            if(found) totalMarkers++;
        }
        // if enough markers detected, add the charuco corner to the final list
        if(totalMarkers >= minMarkers) {
            filteredCharucoIds.push_back(currentCharucoId);
            filteredCharucoCorners.push_back(_allCharucoCorners.getMat().ptr< Point2f >(0)[i]);
        }
    }

    // parse output
    _filteredCharucoCorners.create((int)filteredCharucoCorners.size(), 1, CV_32FC2);
    for(unsigned int i = 0; i < filteredCharucoCorners.size(); i++) {
        _filteredCharucoCorners.getMat().ptr< Point2f >(0)[i] = filteredCharucoCorners[i];
    }

    _filteredCharucoIds.create((int)filteredCharucoIds.size(), 1, CV_32SC1);
    for(unsigned int i = 0; i < filteredCharucoIds.size(); i++) {
        _filteredCharucoIds.getMat().ptr< int >(0)[i] = filteredCharucoIds[i];
    }

    return (unsigned int)filteredCharucoCorners.size();
}


/**
  * ParallelLoopBody class for the parallelization of the charuco corners subpixel refinement
  * Called from function _selectAndRefineChessboardCorners()
  */
class CharucoSubpixelParallel : public ParallelLoopBody {
    public:
    CharucoSubpixelParallel(const Mat *_grey, vector< Point2f > *_filteredChessboardImgPoints,
                            vector< Size > *_filteredWinSizes, const Ptr<DetectorParameters> &_params)
        : grey(_grey), filteredChessboardImgPoints(_filteredChessboardImgPoints),
          filteredWinSizes(_filteredWinSizes), params(_params) {}

    void operator()(const Range &range) const {
        const int begin = range.start;
        const int end = range.end;

        for(int i = begin; i < end; i++) {
            vector< Point2f > in;
            in.push_back((*filteredChessboardImgPoints)[i]);
            Size winSize = (*filteredWinSizes)[i];
            if(winSize.height == -1 || winSize.width == -1)
                winSize = Size(params->cornerRefinementWinSize, params->cornerRefinementWinSize);

            cornerSubPix(*grey, in, winSize, Size(),
                         TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                      params->cornerRefinementMaxIterations,
                                      params->cornerRefinementMinAccuracy));

            (*filteredChessboardImgPoints)[i] = in[0];
        }
    }

    private:
    CharucoSubpixelParallel &operator=(const CharucoSubpixelParallel &); // to quiet MSVC

    const Mat *grey;
    vector< Point2f > *filteredChessboardImgPoints;
    vector< Size > *filteredWinSizes;
    const Ptr<DetectorParameters> &params;
};




/**
  * @brief From all projected chessboard corners, select those inside the image and apply subpixel
  * refinement. Returns number of valid corners.
  */
static unsigned int _selectAndRefineChessboardCorners(InputArray _allCorners, InputArray _image,
                                                      OutputArray _selectedCorners,
                                                      OutputArray _selectedIds,
                                                      const vector< Size > &winSizes) {

    const int minDistToBorder = 2; // minimum distance of the corner to the image border
    // remaining corners, ids and window refinement sizes after removing corners outside the image
    vector< Point2f > filteredChessboardImgPoints;
    vector< Size > filteredWinSizes;
    vector< int > filteredIds;

    // filter corners outside the image
    Rect innerRect(minDistToBorder, minDistToBorder, _image.getMat().cols - 2 * minDistToBorder,
                   _image.getMat().rows - 2 * minDistToBorder);
    for(unsigned int i = 0; i < _allCorners.getMat().total(); i++) {
        if(innerRect.contains(_allCorners.getMat().ptr< Point2f >(0)[i])) {
            filteredChessboardImgPoints.push_back(_allCorners.getMat().ptr< Point2f >(0)[i]);
            filteredIds.push_back(i);
            filteredWinSizes.push_back(winSizes[i]);
        }
    }

    // if none valid, return 0
    if(filteredChessboardImgPoints.size() == 0) return 0;

    // corner refinement, first convert input image to grey
    Mat grey;
    if(_image.getMat().type() == CV_8UC3)
        cvtColor(_image.getMat(), grey, COLOR_BGR2GRAY);
    else
        _image.getMat().copyTo(grey);

    const Ptr<DetectorParameters> params = DetectorParameters::create(); // use default params for corner refinement

    //// For each of the charuco corners, apply subpixel refinement using its correspondind winSize
    // for(unsigned int i=0; i<filteredChessboardImgPoints.size(); i++) {
    //    vector<Point2f> in;
    //    in.push_back(filteredChessboardImgPoints[i]);
    //    Size winSize = filteredWinSizes[i];
    //    if(winSize.height == -1 || winSize.width == -1)
    //        winSize = Size(params.cornerRefinementWinSize, params.cornerRefinementWinSize);
    //    cornerSubPix(grey, in, winSize, Size(),
    //                 TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
    //                              params->cornerRefinementMaxIterations,
    //                              params->cornerRefinementMinAccuracy));
    //    filteredChessboardImgPoints[i] = in[0];
    //}

    // this is the parallel call for the previous commented loop (result is equivalent)
    parallel_for_(
        Range(0, (int)filteredChessboardImgPoints.size()),
        CharucoSubpixelParallel(&grey, &filteredChessboardImgPoints, &filteredWinSizes, params));

    // parse output
    _selectedCorners.create((int)filteredChessboardImgPoints.size(), 1, CV_32FC2);
    for(unsigned int i = 0; i < filteredChessboardImgPoints.size(); i++) {
        _selectedCorners.getMat().ptr< Point2f >(0)[i] = filteredChessboardImgPoints[i];
    }

    _selectedIds.create((int)filteredIds.size(), 1, CV_32SC1);
    for(unsigned int i = 0; i < filteredIds.size(); i++) {
        _selectedIds.getMat().ptr< int >(0)[i] = filteredIds[i];
    }

    return (unsigned int)filteredChessboardImgPoints.size();
}


/**
  * Calculate the maximum window sizes for corner refinement for each charuco corner based on the
  * distance to their closest markers
  */
static void _getMaximumSubPixWindowSizes(InputArrayOfArrays markerCorners, InputArray markerIds,
                                         InputArray charucoCorners, Ptr<CharucoBoard> &board,
                                         vector< Size > &sizes) {

    unsigned int nCharucoCorners = (unsigned int)charucoCorners.getMat().total();
    sizes.resize(nCharucoCorners, Size(-1, -1));

    for(unsigned int i = 0; i < nCharucoCorners; i++) {
        if(charucoCorners.getMat().ptr< Point2f >(0)[i] == Point2f(-1, -1)) continue;
        if(board->nearestMarkerIdx[i].size() == 0) continue;

        double minDist = -1;
        int counter = 0;

        // calculate the distance to each of the closest corner of each closest marker
        for(unsigned int j = 0; j < board->nearestMarkerIdx[i].size(); j++) {
            // find marker
            int markerId = board->ids[board->nearestMarkerIdx[i][j]];
            int markerIdx = -1;
            for(unsigned int k = 0; k < markerIds.getMat().total(); k++) {
                if(markerIds.getMat().ptr< int >(0)[k] == markerId) {
                    markerIdx = k;
                    break;
                }
            }
            if(markerIdx == -1) continue;
            Point2f markerCorner =
                markerCorners.getMat(markerIdx).ptr< Point2f >(0)[board->nearestMarkerCorners[i][j]];
            Point2f charucoCorner = charucoCorners.getMat().ptr< Point2f >(0)[i];
            double dist = norm(markerCorner - charucoCorner);
            if(minDist == -1) minDist = dist; // if first distance, just assign it
            minDist = min(dist, minDist);
            counter++;
        }

        // if this is the first closest marker, dont do anything
        if(counter == 0)
            continue;
        else {
            // else, calculate the maximum window size
            int winSizeInt = int(minDist - 2); // remove 2 pixels for safety
            if(winSizeInt < 1) winSizeInt = 1; // minimum size is 1
            if(winSizeInt > 10) winSizeInt = 10; // maximum size is 10
            sizes[i] = Size(winSizeInt, winSizeInt);
        }
    }
}



/**
  * Interpolate charuco corners using approximated pose estimation
  */
static int _interpolateCornersCharucoApproxCalib(InputArrayOfArrays _markerCorners,
                                                 InputArray _markerIds, InputArray _image,
                                                 Ptr<CharucoBoard> &_board,
                                                 InputArray _cameraMatrix, InputArray _distCoeffs,
                                                 OutputArray _charucoCorners,
                                                 OutputArray _charucoIds) {

    CV_Assert(_image.getMat().channels() == 1 || _image.getMat().channels() == 3);
    CV_Assert(_markerCorners.total() == _markerIds.getMat().total() &&
              _markerIds.getMat().total() > 0);

    // approximated pose estimation using marker corners
    Mat approximatedRvec, approximatedTvec;
    int detectedBoardMarkers;
    Ptr<Board> _b = _board.staticCast<Board>();
    detectedBoardMarkers =
        aruco::estimatePoseBoard(_markerCorners, _markerIds, _b,
                                 _cameraMatrix, _distCoeffs, approximatedRvec, approximatedTvec);

    if(detectedBoardMarkers == 0) return 0;

    // project chessboard corners
    vector< Point2f > allChessboardImgPoints;

    projectPoints(_board->chessboardCorners, approximatedRvec, approximatedTvec, _cameraMatrix,
                  _distCoeffs, allChessboardImgPoints);


    // calculate maximum window sizes for subpixel refinement. The size is limited by the distance
    // to the closes marker corner to avoid erroneous displacements to marker corners
    vector< Size > subPixWinSizes;
    _getMaximumSubPixWindowSizes(_markerCorners, _markerIds, allChessboardImgPoints, _board,
                                 subPixWinSizes);

    // filter corners outside the image and subpixel-refine charuco corners
    unsigned int nRefinedCorners;
    nRefinedCorners = _selectAndRefineChessboardCorners(
        allChessboardImgPoints, _image, _charucoCorners, _charucoIds, subPixWinSizes);

    // to return a charuco corner, its two closes aruco markers should have been detected
    nRefinedCorners = _filterCornersWithoutMinMarkers(_board, _charucoCorners, _charucoIds,
                                                      _markerIds, 2, _charucoCorners, _charucoIds);

    return nRefinedCorners;
}



/**
  * Interpolate charuco corners using local homography
  */
static int _interpolateCornersCharucoLocalHom(InputArrayOfArrays _markerCorners,
                                              InputArray _markerIds, InputArray _image,
                                              Ptr<CharucoBoard> &_board,
                                              OutputArray _charucoCorners,
                                              OutputArray _charucoIds) {

    CV_Assert(_image.getMat().channels() == 1 || _image.getMat().channels() == 3);
    CV_Assert(_markerCorners.total() == _markerIds.getMat().total() &&
              _markerIds.getMat().total() > 0);

    unsigned int nMarkers = (unsigned int)_markerIds.getMat().total();

    // calculate local homographies for each marker
    vector< Mat > transformations;
    transformations.resize(nMarkers);
    for(unsigned int i = 0; i < nMarkers; i++) {
        vector< Point2f > markerObjPoints2D;
        int markerId = _markerIds.getMat().ptr< int >(0)[i];
        vector< int >::const_iterator it = find(_board->ids.begin(), _board->ids.end(), markerId);
        if(it == _board->ids.end()) continue;
        int boardIdx = (int)std::distance<std::vector<int>::const_iterator>(_board->ids.begin(), it);
        markerObjPoints2D.resize(4);
        for(unsigned int j = 0; j < 4; j++)
            markerObjPoints2D[j] =
                Point2f(_board->objPoints[boardIdx][j].x, _board->objPoints[boardIdx][j].y);

        transformations[i] = getPerspectiveTransform(markerObjPoints2D, _markerCorners.getMat(i));
    }

    unsigned int nCharucoCorners = (unsigned int)_board->chessboardCorners.size();
    vector< Point2f > allChessboardImgPoints(nCharucoCorners, Point2f(-1, -1));

    // for each charuco corner, calculate its interpolation position based on the closest markers
    // homographies
    for(unsigned int i = 0; i < nCharucoCorners; i++) {
        Point2f objPoint2D = Point2f(_board->chessboardCorners[i].x, _board->chessboardCorners[i].y);

        vector< Point2f > interpolatedPositions;
        for(unsigned int j = 0; j < _board->nearestMarkerIdx[i].size(); j++) {
            int markerId = _board->ids[_board->nearestMarkerIdx[i][j]];
            int markerIdx = -1;
            for(unsigned int k = 0; k < _markerIds.getMat().total(); k++) {
                if(_markerIds.getMat().ptr< int >(0)[k] == markerId) {
                    markerIdx = k;
                    break;
                }
            }
            if(markerIdx != -1) {
                vector< Point2f > in, out;
                in.push_back(objPoint2D);
                perspectiveTransform(in, out, transformations[markerIdx]);
                interpolatedPositions.push_back(out[0]);
            }
        }

        // none of the closest markers detected
        if(interpolatedPositions.size() == 0) continue;

        // more than one closest marker detected, take middle point
        if(interpolatedPositions.size() > 1) {
            allChessboardImgPoints[i] = (interpolatedPositions[0] + interpolatedPositions[1]) / 2.;
        }
        // a single closest marker detected
        else allChessboardImgPoints[i] = interpolatedPositions[0];
    }

    // calculate maximum window sizes for subpixel refinement. The size is limited by the distance
    // to the closes marker corner to avoid erroneous displacements to marker corners
    vector< Size > subPixWinSizes;
    _getMaximumSubPixWindowSizes(_markerCorners, _markerIds, allChessboardImgPoints, _board,
                                 subPixWinSizes);


    // filter corners outside the image and subpixel-refine charuco corners
    unsigned int nRefinedCorners;
    nRefinedCorners = _selectAndRefineChessboardCorners(
        allChessboardImgPoints, _image, _charucoCorners, _charucoIds, subPixWinSizes);

    // to return a charuco corner, its two closes aruco markers should have been detected
    nRefinedCorners = _filterCornersWithoutMinMarkers(_board, _charucoCorners, _charucoIds,
                                                      _markerIds, 2, _charucoCorners, _charucoIds);

    return nRefinedCorners;
}



/**
  */
int interpolateCornersCharuco(InputArrayOfArrays _markerCorners, InputArray _markerIds,
                              InputArray _image, Ptr<CharucoBoard> &_board,
                              OutputArray _charucoCorners, OutputArray _charucoIds,
                              InputArray _cameraMatrix, InputArray _distCoeffs) {

    // if camera parameters are avaible, use approximated calibration
    if(_cameraMatrix.total() != 0) {
        return _interpolateCornersCharucoApproxCalib(_markerCorners, _markerIds, _image, _board,
                                                     _cameraMatrix, _distCoeffs, _charucoCorners,
                                                     _charucoIds);
    }
    // else use local homography
    else {
        return _interpolateCornersCharucoLocalHom(_markerCorners, _markerIds, _image, _board,
                                                  _charucoCorners, _charucoIds);
    }
}



/**
  */
void drawDetectedCornersCharuco(InputOutputArray _image, InputArray _charucoCorners,
                                InputArray _charucoIds, Scalar cornerColor) {

    CV_Assert(_image.getMat().total() != 0 &&
              (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
    CV_Assert((_charucoCorners.getMat().total() == _charucoIds.getMat().total()) ||
              _charucoIds.getMat().total() == 0);

    unsigned int nCorners = (unsigned int)_charucoCorners.getMat().total();
    for(unsigned int i = 0; i < nCorners; i++) {
        Point2f corner = _charucoCorners.getMat().ptr< Point2f >(0)[i];

        // draw first corner mark
        rectangle(_image, corner - Point2f(3, 3), corner + Point2f(3, 3), cornerColor, 1, LINE_AA);

        // draw ID
        if(_charucoIds.total() != 0) {
            int id = _charucoIds.getMat().ptr< int >(0)[i];
            stringstream s;
            s << "id=" << id;
            putText(_image, s.str(), corner + Point2f(5, -5), FONT_HERSHEY_SIMPLEX, 0.5,
                    cornerColor, 2);
        }
    }
}


/**
  * Check if a set of 3d points are enough for calibration. Z coordinate is ignored.
  * Only axis paralel lines are considered
  */
static bool _arePointsEnoughForPoseEstimation(const vector< Point3f > &points) {

    if(points.size() < 4) return false;

    vector< double > sameXValue; // different x values in points
    vector< int > sameXCounter;  // number of points with the x value in sameXValue
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
    else
        return false;
}


/**
  */
bool estimatePoseCharucoBoard(InputArray _charucoCorners, InputArray _charucoIds,
                              Ptr<CharucoBoard> &_board, InputArray _cameraMatrix, InputArray _distCoeffs,
                              OutputArray _rvec, OutputArray _tvec) {

    CV_Assert((_charucoCorners.getMat().total() == _charucoIds.getMat().total()));

    // need, at least, 4 corners
    if(_charucoIds.getMat().total() < 4) return false;

    vector< Point3f > objPoints;
    objPoints.reserve(_charucoIds.getMat().total());
    for(unsigned int i = 0; i < _charucoIds.getMat().total(); i++) {
        int currId = _charucoIds.getMat().ptr< int >(0)[i];
        CV_Assert(currId >= 0 && currId < (int)_board->chessboardCorners.size());
        objPoints.push_back(_board->chessboardCorners[currId]);
    }

    // points need to be in different lines, check if detected points are enough
    if(!_arePointsEnoughForPoseEstimation(objPoints)) return false;

    solvePnP(objPoints, _charucoCorners, _cameraMatrix, _distCoeffs, _rvec, _tvec);

    return true;
}




/**
  */
double calibrateCameraCharuco(InputArrayOfArrays _charucoCorners, InputArrayOfArrays _charucoIds,
                              Ptr<CharucoBoard> &_board, Size imageSize,
                              InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                              OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags,
                              TermCriteria criteria) {

    CV_Assert(_charucoIds.total() > 0 && (_charucoIds.total() == _charucoCorners.total()));

    // Join object points of charuco corners in a single vector for calibrateCamera() function
    vector< vector< Point3f > > allObjPoints;
    allObjPoints.resize(_charucoIds.total());
    for(unsigned int i = 0; i < _charucoIds.total(); i++) {
        unsigned int nCorners = (unsigned int)_charucoIds.getMat(i).total();
        CV_Assert(nCorners > 0 && nCorners == _charucoCorners.getMat(i).total());
        allObjPoints[i].reserve(nCorners);

        for(unsigned int j = 0; j < nCorners; j++) {
            int pointId = _charucoIds.getMat(i).ptr< int >(0)[j];
            CV_Assert(pointId >= 0 && pointId < (int)_board->chessboardCorners.size());
            allObjPoints[i].push_back(_board->chessboardCorners[pointId]);
        }
    }

    return calibrateCamera(allObjPoints, _charucoCorners, imageSize, _cameraMatrix, _distCoeffs,
                           _rvecs, _tvecs, flags, criteria);
}



/**
 */
void detectCharucoDiamond(InputArray _image, InputArrayOfArrays _markerCorners,
                          InputArray _markerIds, float squareMarkerLengthRate,
                          OutputArrayOfArrays _diamondCorners, OutputArray _diamondIds,
                          InputArray _cameraMatrix, InputArray _distCoeffs) {

    CV_Assert(_markerIds.total() > 0 && _markerIds.total() == _markerCorners.total());

    const float minRepDistanceRate = 0.12f;

    // create Charuco board layout for diamond (3x3 layout)
    Ptr<Dictionary> dict = getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME(0));
    Ptr<CharucoBoard> _charucoDiamondLayout = CharucoBoard::create(3, 3, squareMarkerLengthRate, 1., dict);


    vector< vector< Point2f > > diamondCorners;
    vector< Vec4i > diamondIds;

    // stores if the detected markers have been assigned or not to a diamond
    vector< bool > assigned(_markerIds.total(), false);
    if(_markerIds.total() < 4) return; // a diamond need at least 4 markers

    // convert input image to grey
    Mat grey;
    if(_image.getMat().type() == CV_8UC3)
        cvtColor(_image.getMat(), grey, COLOR_BGR2GRAY);
    else
        _image.getMat().copyTo(grey);

    // for each of the detected markers, try to find a diamond
    for(unsigned int i = 0; i < _markerIds.total(); i++) {
        if(assigned[i]) continue;

        // calculate marker perimeter
        float perimeterSq = 0;
        Mat corners = _markerCorners.getMat(i);
        for(int c = 0; c < 4; c++) {
            perimeterSq +=
                (corners.ptr< Point2f >()[c].x - corners.ptr< Point2f >()[(c + 1) % 4].x) *
                    (corners.ptr< Point2f >()[c].x - corners.ptr< Point2f >()[(c + 1) % 4].x) +
                (corners.ptr< Point2f >()[c].y - corners.ptr< Point2f >()[(c + 1) % 4].y) *
                    (corners.ptr< Point2f >()[c].y - corners.ptr< Point2f >()[(c + 1) % 4].y);
        }
        // maximum reprojection error relative to perimeter
        float minRepDistance = perimeterSq * minRepDistanceRate * minRepDistanceRate;

        int currentId = _markerIds.getMat().ptr< int >()[i];

        // prepare data to call refineDetectedMarkers()
        // detected markers (only the current one)
        vector< Mat > currentMarker;
        vector< int > currentMarkerId;
        currentMarker.push_back(_markerCorners.getMat(i));
        currentMarkerId.push_back(currentId);

        // marker candidates (the rest of markers if they have not been assigned)
        vector< Mat > candidates;
        vector< int > candidatesIdxs;
        for(unsigned int k = 0; k < assigned.size(); k++) {
            if(k == i) continue;
            if(!assigned[k]) {
                candidates.push_back(_markerCorners.getMat(k));
                candidatesIdxs.push_back(k);
            }
        }
        if(candidates.size() < 3) break; // we need at least 3 free markers

        // modify charuco layout id to make sure all the ids are different than current id
        for(int k = 1; k < 4; k++)
            _charucoDiamondLayout->ids[k] = currentId + 1 + k;
        // current id is assigned to [0], so it is the marker on the top
        _charucoDiamondLayout->ids[0] = currentId;

        // try to find the rest of markers in the diamond
        vector< int > acceptedIdxs;
        Ptr<Board> _b = _charucoDiamondLayout.staticCast<Board>();
        aruco::refineDetectedMarkers(grey, _b,
                                     currentMarker, currentMarkerId,
                                     candidates, noArray(), noArray(), minRepDistance, -1, false,
                                     acceptedIdxs);

        // if found, we have a diamond
        if(currentMarker.size() == 4) {

            assigned[i] = true;

            // calculate diamond id, acceptedIdxs array indicates the markers taken from candidates
            // array
            Vec4i markerId;
            markerId[0] = currentId;
            for(int k = 1; k < 4; k++) {
                int currentMarkerIdx = candidatesIdxs[acceptedIdxs[k - 1]];
                markerId[k] = _markerIds.getMat().ptr< int >()[currentMarkerIdx];
                assigned[currentMarkerIdx] = true;
            }

            // interpolate the charuco corners of the diamond
            vector< Point2f > currentMarkerCorners;
            Mat aux;
            interpolateCornersCharuco(currentMarker, currentMarkerId, grey, _charucoDiamondLayout,
                                      currentMarkerCorners, aux, _cameraMatrix, _distCoeffs);

            // if everything is ok, save the diamond
            if(currentMarkerCorners.size() > 0) {
                // reorder corners
                vector< Point2f > currentMarkerCornersReorder;
                currentMarkerCornersReorder.resize(4);
                currentMarkerCornersReorder[0] = currentMarkerCorners[2];
                currentMarkerCornersReorder[1] = currentMarkerCorners[3];
                currentMarkerCornersReorder[2] = currentMarkerCorners[1];
                currentMarkerCornersReorder[3] = currentMarkerCorners[0];

                diamondCorners.push_back(currentMarkerCornersReorder);
                diamondIds.push_back(markerId);
            }
        }
    }


    if(diamondIds.size() > 0) {

        // parse output
        _diamondIds.create((int)diamondIds.size(), 1, CV_32SC4);
        for(unsigned int i = 0; i < diamondIds.size(); i++)
            _diamondIds.getMat().ptr< Vec4i >(0)[i] = diamondIds[i];

        _diamondCorners.create((int)diamondCorners.size(), 1, CV_32FC2);
        for(unsigned int i = 0; i < diamondCorners.size(); i++) {
            _diamondCorners.create(4, 1, CV_32FC2, i, true);
            for(int j = 0; j < 4; j++) {
                _diamondCorners.getMat(i).ptr< Point2f >()[j] = diamondCorners[i][j];
            }
        }
    }
}




/**
  */
void drawCharucoDiamond(Ptr<Dictionary> &dictionary, Vec4i ids, int squareLength, int markerLength,
                        OutputArray _img, int marginSize, int borderBits) {

    CV_Assert(squareLength > 0 && markerLength > 0 && squareLength > markerLength);
    CV_Assert(marginSize >= 0 && borderBits > 0);

    // create a charuco board similar to a charuco marker and print it
    Ptr<CharucoBoard> board =
        CharucoBoard::create(3, 3, (float)squareLength, (float)markerLength, dictionary);

    // assign the charuco marker ids
    for(int i = 0; i < 4; i++)
        board->ids[i] = ids[i];

    Size outSize(3 * squareLength + 2 * marginSize, 3 * squareLength + 2 * marginSize);
    board->draw(outSize, _img, marginSize, borderBits);
}


/**
 */
void drawDetectedDiamonds(InputOutputArray _image, InputArrayOfArrays _corners,
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

        // draw id composed by four numbers
        if(_ids.total() != 0) {
            Point2f cent(0, 0);
            for(int p = 0; p < 4; p++)
                cent += currentMarker.ptr< Point2f >(0)[p];
            cent = cent / 4.;
            stringstream s;
            s << "id=" << _ids.getMat().ptr< Vec4i >(0)[i];
            putText(_image, s.str(), cent, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);
        }
    }
}
}
}
