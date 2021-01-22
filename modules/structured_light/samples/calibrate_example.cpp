#include <iostream>
#include <opencv2/structured_light.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

static const char* keys =
{
    "{@camSettingsPath | | Path of camera calibration file}"
    "{@projSettingsPath | | Path of projector settings}"
    "{@patternPath | | Path to checkerboard pattern}"
    "{@outputName | | Base name for the calibration data}"
};

enum calibrationPattern{CHESSBOARD, CIRCLES_GRID, ASYMETRIC_CIRCLES_GRID};

int main( int argc, char **argv )
{
    VideoCapture cap(CAP_PVAPI);
    Mat frame;

    int nbrOfValidFrames = 0;

    vector<vector<Point2f>> imagePointsCam, imagePointsProj, PointsInProj, imagePointsProjN, pointsInProjN;
    vector<vector<Point3f>> objectPointsCam, worldPointsProj;
    vector<Point3f> tempCam;
    vector<Point2f> tempProj;
    vector<Mat> T1, T2;
    vector<Mat> projInProj, projInCam;
    vector<Mat> projInProjN, projInCamN;

    vector<Mat> rVecs, tVecs, projectorRVecs, projectorTVecs;
    Mat cameraMatrix, distCoeffs, projectorMatrix, projectorDistCoeffs;
    Mat pattern;
    vector<Mat> images;

    structured_light::Settings camSettings, projSettings;

    CommandLineParser parser(argc, argv, keys);

    String camSettingsPath = parser.get<String>(0);
    String projSettingsPath = parser.get<String>(1);
    String patternPath = parser.get<String>(2);
    String outputName = parser.get<String>(3);

    if( camSettingsPath.empty() || projSettingsPath.empty() || patternPath.empty() || outputName.empty() ){
        //structured_light::help();
        return -1;
    }

    pattern = imread(patternPath);

    loadSettings(camSettingsPath, camSettings);
    loadSettings(projSettingsPath, projSettings);

    projSettings.imageSize = Size(pattern.rows, pattern.cols);

    structured_light::createObjectPoints(tempCam, camSettings.patternSize,
                       camSettings.squareSize, camSettings.patternType);

    structured_light::createProjectorObjectPoints(tempProj, projSettings.patternSize,
                                projSettings.squareSize, projSettings.patternType);

    if(!cap.isOpened())
    {
        std::cout << "Camera could not be opened" << std::endl;
        return -1;
    }
    cap.set(CAP_PROP_PVAPI_PIXELFORMAT, CAP_PVAPI_PIXELFORMAT_BAYER8);

    namedWindow("pattern", WINDOW_NORMAL);
    setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    namedWindow("camera view", WINDOW_NORMAL);

    imshow("pattern", pattern);
    std::cout << "Press any key when ready" << std::endl;
    waitKey(0);

    while( nbrOfValidFrames < camSettings.nbrOfFrames )
    {
        cap >> frame;
        if( frame.data )
        {
            Mat color;
            cvtColor(frame, color, COLOR_BayerBG2BGR);
            if( camSettings.imageSize.height == 0 || camSettings.imageSize.width == 0 )
            {
                camSettings.imageSize = Size(frame.rows, frame.cols);
            }

            bool foundProj, foundCam;

            vector<Point2f> projPointBuf;
            vector<Point2f> camPointBuf;

            imshow("camera view", color);
            if( camSettings.patternType == CHESSBOARD && projSettings.patternType == CHESSBOARD )
            {
                int calibFlags = CALIB_CB_ADAPTIVE_THRESH;

                foundCam = findChessboardCorners(color, camSettings.patternSize,
                                                 camPointBuf, calibFlags);

                foundProj = findChessboardCorners(color, projSettings.patternSize,
                                                  projPointBuf, calibFlags);

                if( foundCam && foundProj )
                {
                    Mat gray;
                    cvtColor(color, gray, COLOR_BGR2GRAY);
                    cout << "found pattern" << endl;
                    Mat projCorners, camCorners;
                    cornerSubPix(gray, camPointBuf, camSettings.subpixelSize, Size(-1, -1),
                            TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));

                    cornerSubPix(gray, projPointBuf, projSettings.subpixelSize, Size(-1, -1),
                            TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));

                    drawChessboardCorners(gray, camSettings.patternSize, camPointBuf, foundCam);
                    drawChessboardCorners(gray, projSettings.patternSize, projPointBuf, foundProj);

                    imshow("camera view", gray);
                    char c = (char)waitKey(0);
                    if( c == 10 )
                    {
                        std::cout << "saving pattern #" << nbrOfValidFrames << " for calibration" << std::endl;
                        ostringstream name;
                        name << nbrOfValidFrames;
                        nbrOfValidFrames += 1;

                        imagePointsCam.push_back(camPointBuf);
                        imagePointsProj.push_back(projPointBuf);
                        objectPointsCam.push_back(tempCam);
                        PointsInProj.push_back(tempProj);
                        images.push_back(frame);

                        Mat ptsProjProj, ptsProjCam;
                        Mat ptsProjProjN, ptsProjCamN;
                        Mat TProjProj, TProjCam;
                        vector<Point2f> ptsProjProjVec;
                        vector<Point2f> ptsProjCamVec;

                        structured_light::fromVectorToMat(tempProj, ptsProjProj);
                        structured_light::normalize(ptsProjProj, 2, ptsProjProjN, TProjProj);
                        structured_light::fromMatToVector(ptsProjProjN, ptsProjProjVec);
                        pointsInProjN.push_back(ptsProjProjVec);
                        T2.push_back(TProjProj);
                        projInProj.push_back(ptsProjProj);
                        projInProjN.push_back(ptsProjProjN);

                        structured_light::fromVectorToMat(projPointBuf, ptsProjCam);
                        structured_light::normalize(ptsProjCam, 2, ptsProjCamN, TProjCam);
                        structured_light::fromMatToVector(ptsProjCamN, ptsProjCamVec);
                        imagePointsProjN.push_back(ptsProjCamVec);
                        T1.push_back(TProjCam);
                        projInCam.push_back(ptsProjCam);
                        projInCamN.push_back(ptsProjCamN);

                    }
                    else if( c == 32 )
                    {
                       std::cout << "capture discarded" << std::endl;
                    }
                    else if( c == 27 )
                    {
                        std::cout << "closing program" << std::endl;
                        return -1;
                    }
                }
                else
                {
                    cout << "no pattern found, move board and press any key" << endl;
                    imshow("camera view", frame);
                    waitKey(0);
                }
            }
        }
    }

    structured_light::saveCalibrationData(outputName + "_points.yml", T1, T2, projInCam, projInProj, projInCamN, projInProjN);

    double rms = structured_light::calibrate(objectPointsCam, imagePointsCam, cameraMatrix, distCoeffs,
                          rVecs, tVecs, camSettings.imageSize);
    cout << "rms = " << rms << endl;
    cout << "camera matrix = \n" << cameraMatrix << endl;
    cout << "dist coeffs = \n" << distCoeffs << endl;

    structured_light::fromCamToWorld(cameraMatrix, rVecs, tVecs, imagePointsProj, worldPointsProj);

    rms = structured_light::calibrate(worldPointsProj, PointsInProj, projectorMatrix, projectorDistCoeffs,
                    projectorRVecs, projectorTVecs, projSettings.imageSize);

    cout << "rms = " << rms << endl;
    cout << "projector matrix = \n" << projectorMatrix << endl;
    cout << "projector dist coeffs = \n" << distCoeffs << endl;

    Mat stereoR, stereoT, essential, fundamental;
    Mat RCam, RProj, PCam, PProj, Q;
    rms = stereoCalibrate(worldPointsProj, imagePointsProj, PointsInProj, cameraMatrix, distCoeffs,
                projectorMatrix, projectorDistCoeffs, camSettings.imageSize, stereoR, stereoT,
                essential, fundamental);

    cout << "stereo calibrate: \n" << fundamental << endl;

    structured_light::saveCalibrationResults(outputName, cameraMatrix, distCoeffs, projectorMatrix, projectorDistCoeffs, fundamental );
    return 0;
}
