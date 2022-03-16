// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
namespace opencv_test {
namespace {

static inline vector<Point2f> getAxis(InputArray _cameraMatrix, InputArray _distCoeffs, InputArray _rvec,
                                      InputArray _tvec, float length, const float offset = 0.f)
{
    vector<Point3f> axis;
    axis.push_back(Point3f(offset, offset, 0.f));
    axis.push_back(Point3f(length+offset, offset, 0.f));
    axis.push_back(Point3f(offset, length+offset, 0.f));
    axis.push_back(Point3f(offset, offset, length));
    vector<Point2f> axis_to_img;
    projectPoints(axis, _rvec, _tvec, _cameraMatrix, _distCoeffs, axis_to_img);
    return axis_to_img;
}

static inline vector<Point2f> getMarkerById(int id, const vector<vector<Point2f> >& corners, const vector<int>& ids)
{
    for (size_t i = 0ull; i < ids.size(); i++)
        if (ids[i] == id)
            return corners[i];
    return vector<Point2f>();
}

static inline double deg2rad(double deg) { return deg * CV_PI / 180.; }

/**
 * @brief Get rvec and tvec from yaw, pitch and distance
 */
static inline void getSyntheticRT(double yaw, double pitch, double distance, Mat& rvec, Mat& tvec) {
    rvec = Mat::zeros(3, 1, CV_64FC1);
    tvec = Mat::zeros(3, 1, CV_64FC1);

    // rotate "scene" in pitch axis (x-axis)
    Mat rotPitch(3, 1, CV_64FC1);
    rotPitch.at<double>(0) = -pitch;
    rotPitch.at<double>(1) = 0;
    rotPitch.at<double>(2) = 0;

    // rotate "scene" in yaw (y-axis)
    Mat rotYaw(3, 1, CV_64FC1);
    rotYaw.at<double>(0) = 0;
    rotYaw.at<double>(1) = yaw;
    rotYaw.at<double>(2) = 0;

    // compose both rotations
    composeRT(rotPitch, Mat(3, 1, CV_64FC1, Scalar::all(0)), rotYaw,
        Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec, tvec);

    // Tvec, just move in z (camera) direction the specific distance
    tvec.at<double>(0) = 0.;
    tvec.at<double>(1) = 0.;
    tvec.at<double>(2) = distance;
}

/**
 * @brief Project a synthetic marker
 */
static inline void projectMarker(Mat& img, Ptr<aruco::Board> board, int markerIndex, Mat cameraMatrix, Mat rvec, Mat tvec,
    int markerBorder) {
    // canonical image
    Mat markerImg;
    const int markerSizePixels = 100;
    aruco::drawMarker(board->dictionary, board->ids[markerIndex], markerSizePixels, markerImg, markerBorder);

    // projected corners
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    vector< Point2f > corners;

    // get max coordinate of board
    Point3f maxCoord = board->rightBottomBorder;
    // copy objPoints
    vector<Point3f> objPoints = board->objPoints[markerIndex];
    // move the marker to the origin
    for (size_t i = 0; i < objPoints.size(); i++)
        objPoints[i] -= (maxCoord / 2.f);

    projectPoints(objPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

    // get perspective transform
    vector< Point2f > originalCorners;
    originalCorners.push_back(Point2f(0, 0));
    originalCorners.push_back(Point2f((float)markerSizePixels, 0));
    originalCorners.push_back(Point2f((float)markerSizePixels, (float)markerSizePixels));
    originalCorners.push_back(Point2f(0, (float)markerSizePixels));
    Mat transformation = getPerspectiveTransform(originalCorners, corners);

    // apply transformation
    Mat aux;
    const char borderValue = 127;
    warpPerspective(markerImg, aux, transformation, img.size(), INTER_NEAREST, BORDER_CONSTANT,
        Scalar::all(borderValue));

    // copy only not-border pixels
    for (int y = 0; y < aux.rows; y++) {
        for (int x = 0; x < aux.cols; x++) {
            if (aux.at< unsigned char >(y, x) == borderValue) continue;
            img.at< unsigned char >(y, x) = aux.at< unsigned char >(y, x);
        }
    }
}


/**
 * @brief Get a synthetic image of GridBoard in perspective
 */
static inline Mat projectBoard(Ptr<aruco::GridBoard>& board, Mat cameraMatrix, double yaw, double pitch,
    double distance, Size imageSize, int markerBorder) {

    Mat rvec, tvec;
    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    Mat img = Mat(imageSize, CV_8UC1, Scalar::all(255));
    for (unsigned int index = 0; index < board->ids.size(); index++) {
        projectMarker(img, board.staticCast<aruco::Board>(), index, cameraMatrix, rvec, tvec, markerBorder);
    }

    return img;
}

}

}
