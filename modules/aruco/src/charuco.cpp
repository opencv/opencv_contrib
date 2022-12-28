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
/**
  * Remove charuco corners if any of their minMarkers closest markers has not been detected
  */
static int _filterCornersWithoutMinMarkers(const Ptr<CharucoBoard> &_board, InputArray _allCharucoCorners,
                                           InputArray _allCharucoIds, InputArray _allArucoIds, int minMarkers,
                                           OutputArray _filteredCharucoCorners, OutputArray _filteredCharucoIds) {

    CV_Assert(minMarkers >= 0 && minMarkers <= 2);

    vector< Point2f > filteredCharucoCorners;
    vector< int > filteredCharucoIds;
    // for each charuco corner
    for(unsigned int i = 0; i < _allCharucoIds.getMat().total(); i++) {
        int currentCharucoId = _allCharucoIds.getMat().at< int >(i);
        int totalMarkers = 0; // nomber of closest marker detected
        // look for closest markers
        for(unsigned int m = 0; m < _board->getNearestMarkerIdx()[currentCharucoId].size(); m++) {
            int markerId = _board->getIds()[_board->getNearestMarkerIdx()[currentCharucoId][m]];
            bool found = false;
            for(unsigned int k = 0; k < _allArucoIds.getMat().total(); k++) {
                if(_allArucoIds.getMat().at< int >(k) == markerId) {
                    found = true;
                    break;
                }
            }
            if(found) totalMarkers++;
        }
        // if enough markers detected, add the charuco corner to the final list
        if(totalMarkers >= minMarkers) {
            filteredCharucoIds.push_back(currentCharucoId);
            filteredCharucoCorners.push_back(_allCharucoCorners.getMat().at< Point2f >(i));
        }
    }

    // parse output
    Mat(filteredCharucoCorners).copyTo(_filteredCharucoCorners);
    Mat(filteredCharucoIds).copyTo(_filteredCharucoIds);
    return (int)_filteredCharucoIds.total();
}

/**
  * @brief From all projected chessboard corners, select those inside the image and apply subpixel
  * refinement. Returns number of valid corners.
  */
static int _selectAndRefineChessboardCorners(InputArray _allCorners, InputArray _image, OutputArray _selectedCorners,
                                             OutputArray _selectedIds, const vector< Size > &winSizes) {

    const int minDistToBorder = 2; // minimum distance of the corner to the image border
    // remaining corners, ids and window refinement sizes after removing corners outside the image
    vector< Point2f > filteredChessboardImgPoints;
    vector< Size > filteredWinSizes;
    vector< int > filteredIds;

    // filter corners outside the image
    Rect innerRect(minDistToBorder, minDistToBorder, _image.getMat().cols - 2 * minDistToBorder,
                   _image.getMat().rows - 2 * minDistToBorder);
    for(unsigned int i = 0; i < _allCorners.getMat().total(); i++) {
        if(innerRect.contains(_allCorners.getMat().at< Point2f >(i))) {
            filteredChessboardImgPoints.push_back(_allCorners.getMat().at< Point2f >(i));
            filteredIds.push_back(i);
            filteredWinSizes.push_back(winSizes[i]);
        }
    }

    // if none valid, return 0
    if(filteredChessboardImgPoints.size() == 0) return 0;

    // corner refinement, first convert input image to grey
    Mat grey;
    if(_image.type() == CV_8UC3)
        cvtColor(_image, grey, COLOR_BGR2GRAY);
    else
        grey = _image.getMat();

    DetectorParameters params; // use default params for corner refinement

    //// For each of the charuco corners, apply subpixel refinement using its correspondind winSize
    parallel_for_(Range(0, (int)filteredChessboardImgPoints.size()), [&](const Range& range) {
        const int begin = range.start;
        const int end = range.end;

        for (int i = begin; i < end; i++) {
            vector<Point2f> in;
            in.push_back(filteredChessboardImgPoints[i] - Point2f(0.5, 0.5)); // adjust sub-pixel coordinates for cornerSubPix
            Size winSize = filteredWinSizes[i];
            if (winSize.height == -1 || winSize.width == -1)
                winSize = Size(params.cornerRefinementWinSize, params.cornerRefinementWinSize);

            cornerSubPix(grey, in, winSize, Size(),
                         TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                      params.cornerRefinementMaxIterations,
                                      params.cornerRefinementMinAccuracy));

            filteredChessboardImgPoints[i] = in[0] + Point2f(0.5, 0.5);
        }
    });

    // parse output
    Mat(filteredChessboardImgPoints).copyTo(_selectedCorners);
    Mat(filteredIds).copyTo(_selectedIds);
    return (int)filteredChessboardImgPoints.size();
}


/**
  * Calculate the maximum window sizes for corner refinement for each charuco corner based on the
  * distance to their closest markers
  */
static void _getMaximumSubPixWindowSizes(InputArrayOfArrays markerCorners, InputArray markerIds,
                                         InputArray charucoCorners, const Ptr<CharucoBoard> &board,
                                         vector< Size > &sizes) {

    unsigned int nCharucoCorners = (unsigned int)charucoCorners.getMat().total();
    sizes.resize(nCharucoCorners, Size(-1, -1));

    for(unsigned int i = 0; i < nCharucoCorners; i++) {
        if(charucoCorners.getMat().at< Point2f >(i) == Point2f(-1, -1)) continue;
        if(board->getNearestMarkerIdx()[i].empty()) continue;

        double minDist = -1;
        int counter = 0;

        // calculate the distance to each of the closest corner of each closest marker
        for(unsigned int j = 0; j < board->getNearestMarkerIdx()[i].size(); j++) {
            // find marker
            int markerId = board->getIds()[board->getNearestMarkerIdx()[i][j]];
            int markerIdx = -1;
            for(unsigned int k = 0; k < markerIds.getMat().total(); k++) {
                if(markerIds.getMat().at< int >(k) == markerId) {
                    markerIdx = k;
                    break;
                }
            }
            if(markerIdx == -1) continue;
            Point2f markerCorner =
                markerCorners.getMat(markerIdx).at< Point2f >(board->getNearestMarkerCorners()[i][j]);
            Point2f charucoCorner = charucoCorners.getMat().at< Point2f >(i);
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
static int _interpolateCornersCharucoApproxCalib(InputArrayOfArrays _markerCorners, InputArray _markerIds,
                                                 InputArray _image, const Ptr<CharucoBoard> &_board,
                                                 InputArray _cameraMatrix, InputArray _distCoeffs,
                                                 OutputArray _charucoCorners, OutputArray _charucoIds) {

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

    projectPoints(_board->getChessboardCorners(), approximatedRvec, approximatedTvec, _cameraMatrix,
                  _distCoeffs, allChessboardImgPoints);


    // calculate maximum window sizes for subpixel refinement. The size is limited by the distance
    // to the closes marker corner to avoid erroneous displacements to marker corners
    vector< Size > subPixWinSizes;
    _getMaximumSubPixWindowSizes(_markerCorners, _markerIds, allChessboardImgPoints, _board,
                                 subPixWinSizes);

    // filter corners outside the image and subpixel-refine charuco corners
    return _selectAndRefineChessboardCorners(allChessboardImgPoints, _image, _charucoCorners,
                                             _charucoIds, subPixWinSizes);
}


/**
  * Interpolate charuco corners using local homography
  */
static int _interpolateCornersCharucoLocalHom(InputArrayOfArrays _markerCorners, InputArray _markerIds,
                                              InputArray _image, const Ptr<CharucoBoard> &_board,
                                              OutputArray _charucoCorners, OutputArray _charucoIds) {

    CV_Assert(_image.getMat().channels() == 1 || _image.getMat().channels() == 3);
    CV_Assert(_markerCorners.total() == _markerIds.getMat().total() &&
              _markerIds.getMat().total() > 0);

    unsigned int nMarkers = (unsigned int)_markerIds.getMat().total();

    // calculate local homographies for each marker
    vector< Mat > transformations;
    transformations.resize(nMarkers);

    vector< bool > validTransform(nMarkers, false);

    const auto& ids = _board->getIds();
    for(unsigned int i = 0; i < nMarkers; i++) {
        vector<Point2f> markerObjPoints2D;
        int markerId = _markerIds.getMat().at<int>(i);

        auto it = find(ids.begin(), ids.end(), markerId);
        if(it == ids.end()) continue;
        auto boardIdx = it - ids.begin();
        markerObjPoints2D.resize(4);
        for(unsigned int j = 0; j < 4; j++)
            markerObjPoints2D[j] =
                Point2f(_board->getObjPoints()[boardIdx][j].x, _board->getObjPoints()[boardIdx][j].y);

        transformations[i] = getPerspectiveTransform(markerObjPoints2D, _markerCorners.getMat(i));

        // set transform as valid if transformation is non-singular
        double det = determinant(transformations[i]);
        validTransform[i] = std::abs(det) > 1e-6;
    }

    unsigned int nCharucoCorners = (unsigned int)_board->getChessboardCorners().size();
    vector< Point2f > allChessboardImgPoints(nCharucoCorners, Point2f(-1, -1));

    // for each charuco corner, calculate its interpolation position based on the closest markers
    // homographies
    for(unsigned int i = 0; i < nCharucoCorners; i++) {
        Point2f objPoint2D = Point2f(_board->getChessboardCorners()[i].x, _board->getChessboardCorners()[i].y);

        vector< Point2f > interpolatedPositions;
        for(unsigned int j = 0; j < _board->getNearestMarkerIdx()[i].size(); j++) {
            int markerId = _board->getIds()[_board->getNearestMarkerIdx()[i][j]];
            int markerIdx = -1;
            for(unsigned int k = 0; k < _markerIds.getMat().total(); k++) {
                if(_markerIds.getMat().at< int >(k) == markerId) {
                    markerIdx = k;
                    break;
                }
            }
            if (markerIdx != -1 &&
                validTransform[markerIdx])
            {
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
    return _selectAndRefineChessboardCorners(allChessboardImgPoints, _image, _charucoCorners,
                                             _charucoIds, subPixWinSizes);
}


int interpolateCornersCharuco(InputArrayOfArrays _markerCorners, InputArray _markerIds,
                              InputArray _image, const Ptr<CharucoBoard> &_board,
                              OutputArray _charucoCorners, OutputArray _charucoIds,
                              InputArray _cameraMatrix, InputArray _distCoeffs, int minMarkers) {

    // if camera parameters are avaible, use approximated calibration
    if(_cameraMatrix.total() != 0) {
        _interpolateCornersCharucoApproxCalib(_markerCorners, _markerIds, _image, _board, _cameraMatrix, _distCoeffs,
                                              _charucoCorners, _charucoIds);
    }
    // else use local homography
    else {
        _interpolateCornersCharucoLocalHom(_markerCorners, _markerIds, _image, _board, _charucoCorners, _charucoIds);
    }

    // to return a charuco corner, its closest aruco markers should have been detected
    return _filterCornersWithoutMinMarkers(_board, _charucoCorners, _charucoIds, _markerIds,
                                           minMarkers, _charucoCorners, _charucoIds);
}


void detectCharucoDiamond(InputArray _image, InputArrayOfArrays _markerCorners, InputArray _markerIds,
                          float squareMarkerLengthRate, OutputArrayOfArrays _diamondCorners, OutputArray _diamondIds,
                          InputArray _cameraMatrix, InputArray _distCoeffs, Ptr<Dictionary> dictionary) {
    CV_Assert(_markerIds.total() > 0 && _markerIds.total() == _markerCorners.total());

    const float minRepDistanceRate = 1.302455f;

    vector< vector< Point2f > > diamondCorners;
    vector< Vec4i > diamondIds;

    // stores if the detected markers have been assigned or not to a diamond
    vector< bool > assigned(_markerIds.total(), false);
    if(_markerIds.total() < 4) return; // a diamond need at least 4 markers

    // convert input image to grey
    Mat grey;
    if(_image.type() == CV_8UC3)
        cvtColor(_image, grey, COLOR_BGR2GRAY);
    else
        grey = _image.getMat();

    // for each of the detected markers, try to find a diamond
    for(unsigned int i = 0; i < _markerIds.total(); i++) {
        if(assigned[i]) continue;

        // calculate marker perimeter
        float perimeterSq = 0;
        Mat corners = _markerCorners.getMat(i);
        for(int c = 0; c < 4; c++) {
          Point2f edge = corners.at< Point2f >(c) - corners.at< Point2f >((c + 1) % 4);
          perimeterSq += edge.x*edge.x + edge.y*edge.y;
        }
        // maximum reprojection error relative to perimeter
        float minRepDistance = sqrt(perimeterSq) * minRepDistanceRate;

        int currentId = _markerIds.getMat().at< int >(i);

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
        vector<int> tmpIds(4);
        for(int k = 1; k < 4; k++)
            tmpIds[k] = currentId + 1 + k;
        // current id is assigned to [0], so it is the marker on the top
        tmpIds[0] = currentId;
        // create Charuco board layout for diamond (3x3 layout)
        Ptr<CharucoBoard> _charucoDiamondLayout = new CharucoBoard(Size(3, 3), squareMarkerLengthRate, 1., *dictionary, tmpIds);

        // try to find the rest of markers in the diamond
        vector< int > acceptedIdxs;
        RefineParameters refineParameters(minRepDistance, -1.f, false);
        ArucoDetector detector(*dictionary, DetectorParameters(), refineParameters);
        detector.refineDetectedMarkers(grey, *_charucoDiamondLayout, currentMarker, currentMarkerId, candidates,
                                       noArray(), noArray(), acceptedIdxs);

        // if found, we have a diamond
        if(currentMarker.size() == 4) {

            assigned[i] = true;

            // calculate diamond id, acceptedIdxs array indicates the markers taken from candidates
            // array
            Vec4i markerId;
            markerId[0] = currentId;
            for(int k = 1; k < 4; k++) {
                int currentMarkerIdx = candidatesIdxs[acceptedIdxs[k - 1]];
                markerId[k] = _markerIds.getMat().at< int >(currentMarkerIdx);
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
                currentMarkerCornersReorder[0] = currentMarkerCorners[0];
                currentMarkerCornersReorder[1] = currentMarkerCorners[1];
                currentMarkerCornersReorder[2] = currentMarkerCorners[3];
                currentMarkerCornersReorder[3] = currentMarkerCorners[2];

                diamondCorners.push_back(currentMarkerCornersReorder);
                diamondIds.push_back(markerId);
            }
        }
    }


    if(diamondIds.size() > 0) {
        // parse output
        Mat(diamondIds).copyTo(_diamondIds);

        _diamondCorners.create((int)diamondCorners.size(), 1, CV_32FC2);
        for(unsigned int i = 0; i < diamondCorners.size(); i++) {
            _diamondCorners.create(4, 1, CV_32FC2, i, true);
            for(int j = 0; j < 4; j++) {
                _diamondCorners.getMat(i).at< Point2f >(j) = diamondCorners[i][j];
            }
        }
    }
}


void drawCharucoDiamond(const Ptr<Dictionary> &dictionary, Vec4i ids, int squareLength, int markerLength,
                        OutputArray _img, int marginSize, int borderBits) {
    CV_Assert(squareLength > 0 && markerLength > 0 && squareLength > markerLength);
    CV_Assert(marginSize >= 0 && borderBits > 0);

    // assign the charuco marker ids
    vector<int> tmpIds(4);
    for(int i = 0; i < 4; i++)
       tmpIds[i] = ids[i];
    // create a charuco board similar to a charuco marker and print it
    CharucoBoard board(Size(3, 3), (float)squareLength, (float)markerLength, *dictionary, tmpIds);
    Size outSize(3 * squareLength + 2 * marginSize, 3 * squareLength + 2 * marginSize);
    board.generateImage(outSize, _img, marginSize, borderBits);
}

}
}
