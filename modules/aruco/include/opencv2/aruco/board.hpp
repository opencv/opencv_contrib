// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef __OPENCV_ARUCO_BOARD_HPP__
#define __OPENCV_ARUCO_BOARD_HPP__

#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace aruco {
//! @addtogroup aruco
//! @{

class Dictionary;

/**
 * @brief Board of markers
 *
 * A board is a set of markers in the 3D space with a common coordinate system.
 * The common form of a board of marker is a planar (2D) board, however any 3D layout can be used.
 * A Board object is composed by:
 * - The object points of the marker corners, i.e. their coordinates respect to the board system.
 * - The dictionary which indicates the type of markers of the board
 * - The identifier of all the markers in the board.
 */
class CV_EXPORTS_W Board {
    public:
    /**
    * @brief Provide way to create Board by passing necessary data. Specially needed in Python.
    *
    * @param objPoints array of object points of all the marker corners in the board
    * @param dictionary the dictionary of markers employed for this board
    * @param ids vector of the identifiers of the markers in the board
    *
    */
    CV_WRAP static Ptr<Board> create(InputArrayOfArrays objPoints, const Ptr<Dictionary> &dictionary, InputArray ids);

    /**
    * @brief Set ids vector
    *
    * @param ids vector of the identifiers of the markers in the board (should be the same size
    * as objPoints)
    *
    * Recommended way to set ids vector, which will fail if the size of ids does not match size
     * of objPoints.
    */
    CV_WRAP void setIds(InputArray ids);

    /// array of object points of all the marker corners in the board
    /// each marker include its 4 corners in this order:
    ///-   objPoints[i][0] - left-top point of i-th marker
    ///-   objPoints[i][1] - right-top point of i-th marker
    ///-   objPoints[i][2] - right-bottom point of i-th marker
    ///-   objPoints[i][3] - left-bottom point of i-th marker
    ///
    /// Markers are placed in a certain order - row by row, left to right in every row.
    /// For M markers, the size is Mx4.
    CV_PROP std::vector< std::vector< Point3f > > objPoints;

    /// the dictionary of markers employed for this board
    CV_PROP Ptr<Dictionary> dictionary;

    /// vector of the identifiers of the markers in the board (same size than objPoints)
    /// The identifiers refers to the board dictionary
    CV_PROP_RW std::vector< int > ids;

    /// coordinate of the bottom right corner of the board, is set when calling the function create()
    CV_PROP Point3f rightBottomBorder;
};

/**
 * @brief Draw a planar board
 * @sa drawPlanarBoard
 *
 * @param board layout of the board that will be drawn. The board should be planar,
 * z coordinate is ignored
 * @param outSize size of the output image in pixels.
 * @param img output image with the board. The size of this image will be outSize
 * and the board will be on the center, keeping the board proportions.
 * @param marginSize minimum margins (in pixels) of the board in the output image
 * @param borderBits width of the marker borders.
 *
 * This function return the image of a planar board, ready to be printed. It assumes
 * the Board layout specified is planar by ignoring the z coordinates of the object points.
 */
CV_EXPORTS_W void drawPlanarBoard(const Ptr<Board> &board, Size outSize, OutputArray img,
                                  int marginSize = 0, int borderBits = 1);

/**
 * @brief Planar board with grid arrangement of markers
 * More common type of board. All markers are placed in the same plane in a grid arrangement.
 * The board can be drawn using drawPlanarBoard() function (@sa drawPlanarBoard)
 */

class CV_EXPORTS_W GridBoard : public Board {
    public:
    /**
     * @brief Draw a GridBoard
     *
     * @param outSize size of the output image in pixels.
     * @param img output image with the board. The size of this image will be outSize
     * and the board will be on the center, keeping the board proportions.
     * @param marginSize minimum margins (in pixels) of the board in the output image
     * @param borderBits width of the marker borders.
     *
     * This function return the image of the GridBoard, ready to be printed.
     */
    CV_WRAP void draw(Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1);

    /**
     * @brief Create a GridBoard object
     *
     * @param markersX number of markers in X direction
     * @param markersY number of markers in Y direction
     * @param markerLength marker side length (normally in meters)
     * @param markerSeparation separation between two markers (same unit as markerLength)
     * @param dictionary dictionary of markers indicating the type of markers
     * @param firstMarker id of first marker in dictionary to use on board.
     * @return the output GridBoard object
     *
     * This functions creates a GridBoard object given the number of markers in each direction and
     * the marker size and marker separation.
     */
    CV_WRAP static Ptr<GridBoard> create(int markersX, int markersY, float markerLength, float markerSeparation,
                                         const Ptr<Dictionary> &dictionary, int firstMarker = 0);

    CV_WRAP Size getGridSize() const { return Size(_markersX, _markersY); }

    CV_WRAP float getMarkerLength() const { return _markerLength; }

    CV_WRAP float getMarkerSeparation() const { return _markerSeparation; }

    private:
    // number of markers in X and Y directions
    int _markersX, _markersY;

    // marker side length (normally in meters)
    float _markerLength;

    // separation between markers in the grid
    float _markerSeparation;
};

/**
 * @brief ChArUco board
 * Specific class for ChArUco boards. A ChArUco board is a planar board where the markers are placed
 * inside the white squares of a chessboard. The benefits of ChArUco boards is that they provide
 * both, ArUco markers versatility and chessboard corner precision, which is important for
 * calibration and pose estimation.
 * This class also allows the easy creation and drawing of ChArUco boards.
 */
class CV_EXPORTS_W CharucoBoard : public Board {
    public:
    // vector of chessboard 3D corners precalculated
    CV_PROP std::vector< Point3f > chessboardCorners;

    // for each charuco corner, nearest marker id and nearest marker corner id of each marker
    CV_PROP std::vector< std::vector< int > > nearestMarkerIdx;
    CV_PROP std::vector< std::vector< int > > nearestMarkerCorners;

    /**
     * @brief Draw a ChArUco board
     *
     * @param outSize size of the output image in pixels.
     * @param img output image with the board. The size of this image will be outSize
     * and the board will be on the center, keeping the board proportions.
     * @param marginSize minimum margins (in pixels) of the board in the output image
     * @param borderBits width of the marker borders.
     *
     * This function return the image of the ChArUco board, ready to be printed.
     */
    CV_WRAP void draw(Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1);


    /**
     * @brief Create a CharucoBoard object
     *
     * @param squaresX number of chessboard squares in X direction
     * @param squaresY number of chessboard squares in Y direction
     * @param squareLength chessboard square side length (normally in meters)
     * @param markerLength marker side length (same unit than squareLength)
     * @param dictionary dictionary of markers indicating the type of markers.
     * The first markers in the dictionary are used to fill the white chessboard squares.
     * @return the output CharucoBoard object
     *
     * This functions creates a CharucoBoard object given the number of squares in each direction
     * and the size of the markers and chessboard squares.
     */
    CV_WRAP static Ptr<CharucoBoard> create(int squaresX, int squaresY, float squareLength,
                                            float markerLength, const Ptr<Dictionary> &dictionary);

    CV_WRAP Size getChessboardSize() const { return Size(_squaresX, _squaresY); }

    CV_WRAP float getSquareLength() const { return _squareLength; }

    CV_WRAP float getMarkerLength() const { return _markerLength; }

    private:
    void _getNearestMarkerCorners();

    // number of markers in X and Y directions
    int _squaresX, _squaresY;

    // size of chessboard squares side (normally in meters)
    float _squareLength;

    // marker side length (normally in meters)
    float _markerLength;
};

/**
 * @brief test whether the ChArUco markers are collinear
 *
 * @param board layout of ChArUco board.
 * @param charucoIds list of identifiers for each corner in charucoCorners per frame.
 * @return bool value, 1 (true) if detected corners form a line, 0 (false) if they do not.
      solvePnP, calibration functions will fail if the corners are collinear (true).
 *
 * The number of ids in charucoIDs should be <= the number of chessboard corners in the board.  This functions checks whether the charuco corners are on a straight line (returns true, if so), or not (false).  Axis parallel, as well as diagonal and other straight lines detected.  Degenerate cases: for number of charucoIDs <= 2, the function returns true.
 */
CV_EXPORTS_W bool testCharucoCornersCollinear(const Ptr<CharucoBoard> &board, InputArray charucoIds);

//! @}

}
}

#endif
