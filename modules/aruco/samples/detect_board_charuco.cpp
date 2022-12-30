#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include "aruco_samples_utility.hpp"

using namespace std;
using namespace cv;


namespace {
const char* about = "Pose estimation using a ChArUco board";
const char* keys  =
        "{w        |       | Number of squares in X direction }"
        "{h        |       | Number of squares in Y direction }"
        "{sl       |       | Square side length (in meters) }"
        "{ml       |       | Marker side length (in meters) }"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{cd       |       | Input file with custom dictionary }"
        "{c        |       | Output file with calibrated camera parameters }"
        "{v        |       | Input from video or image file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{dp       |       | File of marker detector parameters }"
        "{rs       |       | Apply refind strategy }"
        "{r        |       | show rejected candidates too }";
}


int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 6) {
        parser.printMessage();
        return 0;
    }

    int squaresX = parser.get<int>("w");
    int squaresY = parser.get<int>("h");
    float squareLength = parser.get<float>("sl");
    float markerLength = parser.get<float>("ml");
    bool showRejected = parser.has("r");
    bool refindStrategy = parser.has("rs");
    int camId = parser.get<int>("ci");

    String video;
    if(parser.has("v")) {
        video = parser.get<String>("v");
    }

    Mat cameraMatrix, distCoeffs;
    if(parser.has("c")) {
        bool readOk = readCameraParameters(parser.get<string>("c"), cameraMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }

    aruco::DetectorParameters detectorParams = aruco::DetectorParameters();
    if(parser.has("dp")) {
        FileStorage fs(parser.get<string>("dp"), FileStorage::READ);
        bool readOk = detectorParams.readDetectorParameters(fs.root());
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(0);
    if (parser.has("d")) {
        int dictionaryId = parser.get<int>("d");
        dictionary = aruco::getPredefinedDictionary(aruco::PredefinedDictionaryType(dictionaryId));
    }
    else if (parser.has("cd")) {
        FileStorage fs(parser.get<std::string>("cd"), FileStorage::READ);
        bool readOk = dictionary.aruco::Dictionary::readDictionary(fs.root());
        if(!readOk) {
            cerr << "Invalid dictionary file" << endl;
            return 0;
        }
    }
    else {
        cerr << "Dictionary not specified" << endl;
        return 0;
    }

    VideoCapture inputVideo;
    int waitTime;
    if(!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
    } else {
        inputVideo.open(camId);
        waitTime = 10;
    }

    float axisLength = 0.5f * ((float)min(squaresX, squaresY) * (squareLength));

    // Create charuco board object
    aruco::CharucoBoard board(Size(squaresX, squaresY), squareLength, markerLength, dictionary);

    aruco::CharucoParameters charucoParams;
    charucoParams.cameraMatrix = cameraMatrix;
    charucoParams.distCoeffs = distCoeffs;

    if(refindStrategy) {
        charucoParams.tryRefineMarkers = true;
    }

    aruco::CharucoDetector detector(board, charucoParams, detectorParams);  // , refineParams);

    double totalTime = 0;
    int totalIterations = 0;

    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)getTickCount();

        vector<int> markerIds, charucoIds;
        vector<vector<Point2f>> markerCorners, rejectedMarkers;
        vector<Point2f> charucoCorners;
        Vec3d rvec, tvec;

        detector.detectBoard(image, charucoCorners, charucoIds, markerCorners, markerIds);

        // Estimate charuco board pose
        Mat objPoints, imgPoints;
        int markersOfBoardDetected = 0;

        // Get object and image points for the solvePnP function
        board.matchImagePoints(charucoCorners, charucoIds, objPoints, imgPoints);

        // Find pose
        solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
        markersOfBoardDetected = (int)objPoints.total() / 4;

        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }

        // Draw results
        image.copyTo(imageCopy);
        if(!markerIds.empty()) {
            aruco::drawDetectedMarkers(imageCopy, markerCorners);  // , markerIds);
        }

        if(showRejected && !rejectedMarkers.empty()) {
            aruco::drawDetectedMarkers(imageCopy, rejectedMarkers, noArray(), Scalar(100, 0, 255));
        }

        if(!charucoIds.empty()) {
            Scalar color;
            color = Scalar(255, 0, 0);
            aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, color);
        }

        if(markersOfBoardDetected > 0) {
            drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvec, tvec, axisLength);
        }

        imshow("out", imageCopy);
        char key = (char)waitKey(waitTime);
        if(key == 27) break;
    }

    return 0;
}
