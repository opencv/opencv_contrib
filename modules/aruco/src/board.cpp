// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/imgproc.hpp>
#include <opencv2/aruco/board.hpp>
#include <opencv2/aruco/dictionary.hpp>

namespace cv {
namespace aruco {
using namespace std;

/**
 * @brief Implementation of drawPlanarBoard that accepts a raw Board pointer.
 */
static void _drawPlanarBoardImpl(Board *_board, Size outSize, OutputArray _img, int marginSize, int borderBits) {
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

void drawPlanarBoard(const Ptr<Board> &_board, Size outSize, OutputArray _img, int marginSize,
                     int borderBits) {
    _drawPlanarBoardImpl(_board, outSize, _img, marginSize, borderBits);
}

Ptr<Board> Board::create(InputArrayOfArrays objPoints, const Ptr<Dictionary> &dictionary, InputArray ids) {
    CV_Assert(objPoints.total() == ids.total());
    CV_Assert(objPoints.type() == CV_32FC3 || objPoints.type() == CV_32FC1);

    std::vector<std::vector<Point3f> > obj_points_vector;
    Point3f rightBottomBorder = Point3f(0.f, 0.f, 0.f);
    for (unsigned int i = 0; i < objPoints.total(); i++) {
        std::vector<Point3f> corners;
        Mat corners_mat = objPoints.getMat(i);

        if (corners_mat.type() == CV_32FC1)
            corners_mat = corners_mat.reshape(3);
        CV_Assert(corners_mat.total() == 4);

        for (int j = 0; j < 4; j++) {
            const Point3f &corner = corners_mat.at<Point3f>(j);
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

void Board::setIds(InputArray ids_) {
    CV_Assert(objPoints.size() == ids_.total());
    ids_.copyTo(this->ids);
}

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
    for (unsigned int i = 0; i < totalMarkers; i++) {
        res->ids[i] = i + firstMarker;
    }

    // calculate Board objPoints
    for (int y = 0; y < markersY; y++) {
        for (int x = 0; x < markersX; x++) {
            vector <Point3f> corners(4);
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

void GridBoard::draw(Size outSize, OutputArray _img, int marginSize, int borderBits) {
    _drawPlanarBoardImpl((Board*)this, outSize, _img, marginSize, borderBits);
}

void CharucoBoard::draw(Size outSize, OutputArray _img, int marginSize, int borderBits) {
    CV_Assert(!outSize.empty());
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
    _drawPlanarBoardImpl(this, chessboardZoneImg.size(), markersImg, diffSquareMarkerLengthPixels, borderBits);
    markersImg.copyTo(chessboardZoneImg);

    // now draw black squares
    for(int y = 0; y < _squaresY; y++) {
        for(int x = 0; x < _squaresX; x++) {

            if(y % 2 != x % 2) continue; // white corner, dont do anything

            double startX, startY;
            startX = squareSizePixels * double(x);
            startY = squareSizePixels * double(y);

            Mat squareZone = chessboardZoneImg.rowRange(int(startY), int(startY + squareSizePixels))
                                 .colRange(int(startX), int(startX + squareSizePixels));

            squareZone.setTo(0);
        }
    }
}

Ptr<CharucoBoard> CharucoBoard::create(int squaresX, int squaresY, float squareLength,
                                  float markerLength, const Ptr<Dictionary> &dictionary) {
    CV_Assert(squaresX > 1 && squaresY > 1 && markerLength > 0 && squareLength > markerLength);
    Ptr<CharucoBoard> res = makePtr<CharucoBoard>();

    res->_squaresX = squaresX;
    res->_squaresY = squaresY;
    res->_squareLength = squareLength;
    res->_markerLength = markerLength;
    res->dictionary = dictionary;

    float diffSquareMarkerLength = (squareLength - markerLength) / 2;
    // calculate Board objPoints
    for(int y = 0; y < squaresY; y++) {
        for(int x = 0; x < squaresX; x++) {

            if(y % 2 == x % 2) continue; // black corner, no marker here

            vector<Point3f> corners(4);
            corners[0] = Point3f(x * squareLength + diffSquareMarkerLength,
                                 y * squareLength + diffSquareMarkerLength, 0);
            corners[1] = corners[0] + Point3f(markerLength, 0, 0);
            corners[2] = corners[0] + Point3f(markerLength, markerLength, 0);
            corners[3] = corners[0] + Point3f(0, markerLength, 0);
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
    res->rightBottomBorder = Point3f(squaresX * squareLength,
                                     squaresY * squareLength, 0.f);
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
            if(j == 0 || fabs(sqDistance - minDist) < cv::pow(0.01 * _squareLength, 2)) {
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

bool testCharucoCornersCollinear(const Ptr<CharucoBoard> &_board, InputArray _charucoIds) {
    unsigned int nCharucoCorners = (unsigned int)_charucoIds.getMat().total();
    if (nCharucoCorners <= 2)
        return true;

    // only test if there are 3 or more corners
    CV_Assert( _board->chessboardCorners.size() >= _charucoIds.getMat().total());

    Vec<double, 3> point0( _board->chessboardCorners[_charucoIds.getMat().at< int >(0)].x,
            _board->chessboardCorners[_charucoIds.getMat().at< int >(0)].y, 1);

    Vec<double, 3> point1( _board->chessboardCorners[_charucoIds.getMat().at< int >(1)].x,
            _board->chessboardCorners[_charucoIds.getMat().at< int >(1)].y, 1);

    // create a line from the first two points.
    Vec<double, 3> testLine = point0.cross(point1);
    Vec<double, 3> testPoint(0, 0, 1);

    double divisor = sqrt(testLine[0]*testLine[0] + testLine[1]*testLine[1]);
    CV_Assert(divisor != 0.0);

    // normalize the line with normal
    testLine /= divisor;

    double dotProduct;
    for (unsigned int i = 2; i < nCharucoCorners; i++){
        testPoint(0) = _board->chessboardCorners[_charucoIds.getMat().at< int >(i)].x;
        testPoint(1) = _board->chessboardCorners[_charucoIds.getMat().at< int >(i)].y;

        // if testPoint is on testLine, dotProduct will be zero (or very, very close)
        dotProduct = testPoint.dot(testLine);

        if (std::abs(dotProduct) > 1e-6){
            return false;
        }
    }
    // no points found that were off of testLine, return true that all points collinear.
    return true;
}

}
}
