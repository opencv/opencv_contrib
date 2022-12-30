#include <iostream>
#include <string>
//! [charuco_header]
#include <opencv2/objdetect.hpp>
//! [charuco_header]
#include <opencv2/highgui.hpp>
#include "aruco_samples_utility.hpp"

namespace {
const char* about = "A tutorial code on charuco board creation and detection of charuco board with and without camera caliberation";
const char* keys =
    "{c        |       | Put value of c=1 to create charuco board; c=2 to detect charuco board without camera calibration; c=3 to detect charuco board with camera calibration and Pose Estimation}";
}

static inline void createBoard()
{
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    //! [charuco_create_board]
    cv::aruco::CharucoBoard board(cv::Size(5, 7), 0.04f, 0.02f, dictionary);
    cv::Mat boardImage;
    board.generateImage(cv::Size(600, 500), boardImage, 10, 1);
    //! [charuco_create_board]

    cv::imwrite("BoardImage.jpg", boardImage);
}

//! [charuco_with_calib]
static inline void detectCharucoBoardWithCalibrationPose()
{
    cv::VideoCapture inputVideo;
    inputVideo.open(0);

    //! [charuco_read_cam_params]
    cv::Mat cameraMatrix, distCoeffs;
    std::string filename = "../samples/tutorial_camera_params.yml";
    bool readOk = readCameraParameters(filename, cameraMatrix, distCoeffs);
    //! [charuco_read_cam_params]

    if (!readOk) {
        std::cerr << "Invalid camera file" << std::endl;
        return;
    }

    //! [charuco_board_params]
    cv::aruco::CharucoParameters charucoParams;
    charucoParams.cameraMatrix = cameraMatrix;
    charucoParams.distCoeffs = distCoeffs;

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    cv::aruco::CharucoBoard board(cv::Size(5, 7), 0.04f, 0.02f, dictionary);
    cv::aruco::CharucoDetector detector(board, charucoParams, detectorParams);
    //! [charuco_board_params]

    while (inputVideo.grab()) {
        cv::Mat image;

        cv::Mat imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);

        //! [charuco_detect_board]
        std::vector<int> markerIds;
        std::vector<int> charucoIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        std::vector<cv::Point2f> charucoCorners;

        detector.detectBoard(image, charucoCorners, charucoIds, markerCorners, markerIds);
        //! [charuco_detect_board]

        // If at least one marker detected
        if (!charucoIds.empty()) {
            cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
        }

        // If at least one charuco corner detected
        if (!charucoIds.empty()) {
            //! [charuco_draw_corners]
            cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds);
            //! [charuco_draw_corners]

            //! [charuco_pose_estimation]
            cv::Vec3d rvec, tvec;
            cv::Mat objPoints, imgPoints;
            int markersOfBoardDetected = 0;

            // Get object and image points for the solvePnP function
            board.matchImagePoints(charucoCorners, charucoIds, objPoints, imgPoints);
            // Find pose
            cv::solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
            //! [charuco_pose_estimation]

            // If charuco pose is valid
            markersOfBoardDetected = (int)objPoints.total() / 4;
            if (markersOfBoardDetected != 0)
                cv::drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
        }

        cv::imshow("out", imageCopy);
        char key = (char)cv::waitKey(30);
        if (key == 27)
            break;
    }
}
//! [charuco_with_calib]

//! [charuco_wo_calib]
static inline void detectCharucoBoardWithoutCalibration()
{
    cv::VideoCapture inputVideo;

    inputVideo.open(0);

    std::vector<int> markerIds;
    std::vector<int> charucoIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    std::vector<cv::Point2f> charucoCorners;

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::CharucoBoard board(cv::Size(5, 7), 0.04f, 0.02f, dictionary);
    cv::aruco::CharucoDetector detector(board);

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);

        // Detect markers and interpolate corners
        detector.detectBoard(image, charucoCorners, charucoIds, markerCorners, markerIds);

        // If at least one marker detected
        if (!charucoIds.empty()) {
            cv::aruco::drawDetectedMarkers(imageCopy, markerCorners); //, markerIds);
        }

        // If at least one charuco corner detected
        if (!charucoCorners.empty()) {
            cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds);
        }

        cv::imshow("out", imageCopy);
        char key = (char)cv::waitKey(30);
        if (key == 27)
            break;
    }
}
//! [charuco_wo_calib]

int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (argc < 2) {
        parser.printMessage();
        return 0;
    }
    int choose = parser.get<int>("c");
    switch (choose) {
    case 1:
        createBoard();
        std::cout << "An image named BoardImg.jpg is generated in folder containing this file" << std::endl;
        break;
    case 2:
        detectCharucoBoardWithoutCalibration();
        break;
    case 3:
        detectCharucoBoardWithCalibrationPose();
        break;
    default:
        break;
    }
    return 0;
}
