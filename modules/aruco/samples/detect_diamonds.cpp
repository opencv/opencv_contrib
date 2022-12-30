#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include "aruco_samples_utility.hpp"

using namespace std;
using namespace cv;


namespace {
const char* about = "Detect ChArUco markers";
const char* keys  =
        "{sl       |       | Square side length (in meters) }"
        "{ml       |       | Marker side length (in meters) }"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{cd       |       | Input file with custom dictionary }"
        "{c        |       | Output file with calibrated camera parameters }"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{dp       |       | File of marker detector parameters }"
        "{refine   |       | Corner refinement: CORNER_REFINE_NONE=0, CORNER_REFINE_SUBPIX=1,"
        "CORNER_REFINE_CONTOUR=2, CORNER_REFINE_APRILTAG=3}"
        "{r        |       | show rejected candidates too }";
}


int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 4) {
        parser.printMessage();
        return 0;
    }

    float squareLength = parser.get<float>("sl");
    float markerLength = parser.get<float>("ml");
    bool showRejected = parser.has("r");
    bool estimatePose = parser.has("c");

    aruco::DetectorParameters detectorParams = aruco::DetectorParameters();
    if(parser.has("dp")) {
        FileStorage fs(parser.get<string>("dp"), FileStorage::READ);
        bool readOk = detectorParams.readDetectorParameters(fs.root());
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }

    if (parser.has("refine")) {
        //override cornerRefinementMethod read from config file
        detectorParams.cornerRefinementMethod = parser.get<aruco::CornerRefineMethod>("refine");
    }
    std::cout << "Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag 2): "
              << (int)detectorParams.cornerRefinementMethod << std::endl;

    int camId = parser.get<int>("ci");
    String video;

    if(parser.has("v")) {
        video = parser.get<String>("v");
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

    Mat cameraMatrix, distCoeffs;
    if(estimatePose) {
        bool readOk = readCameraParameters(parser.get<string>("c"), cameraMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
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

    aruco::CharucoParameters charucoParams;
    charucoParams.cameraMatrix = cameraMatrix;
    charucoParams.distCoeffs = distCoeffs;

    aruco::CharucoBoard board(Size(3, 3), squareLength, markerLength, dictionary);
    aruco::CharucoDetector detector(board, charucoParams, detectorParams);

    // Set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

    double totalTime = 0;
    int totalIterations = 0;

    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)getTickCount();

        vector<vector<Point2f>> markerCorners, rejectedMarkers;
        vector<int> markerIds;
        vector<vector<Point2f>> diamondCorners;
        vector<Vec4i> diamondIds;

        // Detect diamonds and markers
        detector.detectDiamonds(image, diamondCorners, diamondIds,
            markerCorners, markerIds);

        size_t nDiamonds = diamondCorners.size();
        vector<Vec3d> rvecs(nDiamonds), tvecs(nDiamonds);

        // Estimate diamond poses
        if(estimatePose && !markerCorners.empty()) {
            for(size_t i = 0; i < nDiamonds; i++) {
                solvePnP(objPoints, diamondCorners[i], cameraMatrix,
                    distCoeffs, rvecs[i], tvecs[i]);
            }
        }

        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations)
                 << " ms)" << endl;
        }

        // Draw results
        image.copyTo(imageCopy);

        if(!markerIds.empty()) {
            aruco::drawDetectedMarkers(imageCopy, markerCorners);
        }

        if(showRejected && !rejectedMarkers.empty()) {
            aruco::drawDetectedMarkers(imageCopy, rejectedMarkers, noArray(), Scalar(100, 0, 255));
        }

        if(!diamondIds.empty()) {
            aruco::drawDetectedDiamonds(imageCopy, diamondCorners, diamondIds);
        }

        if(estimatePose) {
            for(unsigned int i = 0; i < diamondIds.size(); i++) {
                drawFrameAxes(imageCopy, cameraMatrix, distCoeffs,
                              rvecs[i], tvecs[i], squareLength * 1.1f);
            }
        }

        imshow("out", imageCopy);
        char key = (char)waitKey(waitTime);
        if(key == 27) break;
    }

    return 0;
}
