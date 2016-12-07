/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

static const char* keys =
{
    "{@camSettingsPath | | Path of camera calibration file}"
    "{@projSettingsPath | | Path of projector settings}"
    "{@patternPath | | Path to checkerboard pattern}"
    "{@outputName | | Base name for the calibration data}"
};

static void help()
{
    cout << "\nThis example calibrates a camera and a projector" << endl;
    cout << "To call: ./example_structured_light_projectorcalibration <cam_settings_path> "
            " <proj_settings_path> <chessboard_path> <calibration_basename>"
            " cam settings are parameters about the chessboard that needs to be detected to"
            " calibrate the camera and proj setting are the same kind of parameters about the chessboard"
            " that needs to be detected to calibrate the projector" << endl;
}
enum calibrationPattern{ CHESSBOARD, CIRCLES_GRID, ASYMETRIC_CIRCLES_GRID };

struct Settings
{
    Settings();
    int patternType;
    Size patternSize;
    Size subpixelSize;
    Size imageSize;
    float squareSize;
    int nbrOfFrames;
};

void loadSettings( String path, Settings &sttngs );

void createObjectPoints( vector<Point3f> &patternCorners, Size patternSize, float squareSize,
                        int patternType );

void createProjectorObjectPoints( vector<Point2f> &patternCorners, Size patternSize, float squareSize,
                        int patternType );

double calibrate( vector< vector<Point3f> > objPoints, vector< vector<Point2f> > imgPoints,
               Mat &cameraMatrix, Mat &distCoeffs, vector<Mat> &r, vector<Mat> &t, Size imgSize );

void fromCamToWorld( Mat cameraMatrix, vector<Mat> rV, vector<Mat> tV,
                    vector< vector<Point2f> > imgPoints, vector< vector<Point3f> > &worldPoints );

void saveCalibrationResults( String path, Mat camK, Mat camDistCoeffs, Mat projK, Mat projDistCoeffs,
                      Mat fundamental );

void saveCalibrationData( String path, vector<Mat> T1, vector<Mat> T2, vector<Mat> ptsProjCam, vector<Mat> ptsProjProj, vector<Mat> ptsProjCamN, vector<Mat> ptsProjProjN);

void normalize(const Mat &pts, const int& dim, Mat& normpts, Mat &T);

void fromVectorToMat( vector<Point2f> v, Mat &pts);

void fromMatToVector( Mat pts, vector<Point2f> &v );

int main( int argc, char **argv )
{
    VideoCapture cap(CAP_PVAPI);
    Mat frame;

    int nbrOfValidFrames = 0;

    vector< vector<Point2f> > imagePointsCam, imagePointsProj, PointsInProj, imagePointsProjN, pointsInProjN;
    vector< vector<Point3f> > objectPointsCam, worldPointsProj;
    vector<Point3f> tempCam;
    vector<Point2f> tempProj;
    vector<Mat> T1, T2;
    vector<Mat> projInProj, projInCam;
    vector<Mat> projInProjN, projInCamN;

    vector<Mat> rVecs, tVecs, projectorRVecs, projectorTVecs;
    Mat cameraMatrix, distCoeffs, projectorMatrix, projectorDistCoeffs;
    Mat pattern;
    vector<Mat> images;

    Settings camSettings, projSettings;

    CommandLineParser parser(argc, argv, keys);

    String camSettingsPath = parser.get<String>(0);
    String projSettingsPath = parser.get<String>(1);
    String patternPath = parser.get<String>(2);
    String outputName = parser.get<String>(3);

    if( camSettingsPath.empty() || projSettingsPath.empty() || patternPath.empty() || outputName.empty() ){
        help();
        return -1;
    }

    pattern = imread(patternPath);

    loadSettings(camSettingsPath, camSettings);
    loadSettings(projSettingsPath, projSettings);

    projSettings.imageSize = Size(pattern.rows, pattern.cols);

    createObjectPoints(tempCam, camSettings.patternSize,
                       camSettings.squareSize, camSettings.patternType);
    createProjectorObjectPoints(tempProj, projSettings.patternSize,
                                projSettings.squareSize, projSettings.patternType);

    if(!cap.isOpened())
    {
        cout << "Camera could not be opened" << endl;
        return -1;
    }
    cap.set(CAP_PROP_PVAPI_PIXELFORMAT, CAP_PVAPI_PIXELFORMAT_BAYER8);

    namedWindow("pattern", WINDOW_NORMAL);
    setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    namedWindow("camera view", WINDOW_NORMAL);

    imshow("pattern", pattern);
    cout << "Press any key when ready" << endl;
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
                        cout << "saving pattern #" << nbrOfValidFrames << " for calibration" << endl;
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

                        fromVectorToMat(tempProj, ptsProjProj);
                        normalize(ptsProjProj, 2, ptsProjProjN, TProjProj);
                        fromMatToVector(ptsProjProjN, ptsProjProjVec);
                        pointsInProjN.push_back(ptsProjProjVec);
                        T2.push_back(TProjProj);
                        projInProj.push_back(ptsProjProj);
                        projInProjN.push_back(ptsProjProjN);

                        fromVectorToMat(projPointBuf, ptsProjCam);
                        normalize(ptsProjCam, 2, ptsProjCamN, TProjCam);
                        fromMatToVector(ptsProjCamN, ptsProjCamVec);
                        imagePointsProjN.push_back(ptsProjCamVec);
                        T1.push_back(TProjCam);
                        projInCam.push_back(ptsProjCam);
                        projInCamN.push_back(ptsProjCamN);

                    }
                    else if( c == 32 )
                    {
                       cout << "capture discarded" << endl;
                    }
                    else if( c == 27 )
                    {
                        cout << "closing program" << endl;
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

    saveCalibrationData(outputName + "_points.yml", T1, T2, projInCam, projInProj, projInCamN, projInProjN);

    double rms = calibrate(objectPointsCam, imagePointsCam, cameraMatrix, distCoeffs,
                          rVecs, tVecs, camSettings.imageSize);
    cout << "rms = " << rms << endl;
    cout << "camera matrix = \n" << cameraMatrix << endl;
    cout << "dist coeffs = \n" << distCoeffs << endl;

    fromCamToWorld(cameraMatrix, rVecs, tVecs, imagePointsProj, worldPointsProj);

    rms = calibrate(worldPointsProj, PointsInProj, projectorMatrix, projectorDistCoeffs,
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

    saveCalibrationResults(outputName, cameraMatrix, distCoeffs, projectorMatrix, projectorDistCoeffs, fundamental );
    return 0;
}

Settings::Settings(){
    patternType = CHESSBOARD;
    patternSize = Size(13, 9);
    subpixelSize = Size(11, 11);
    squareSize = 50;
    nbrOfFrames = 25;
}

void loadSettings( String path, Settings &sttngs )
{
    FileStorage fsInput(path, FileStorage::READ);

    fsInput["PatternWidth"] >> sttngs.patternSize.width;
    fsInput["PatternHeight"] >> sttngs.patternSize.height;
    fsInput["SubPixelWidth"] >> sttngs.subpixelSize.width;
    fsInput["SubPixelHeight"] >> sttngs.subpixelSize.height;
    fsInput["SquareSize"] >> sttngs.squareSize;
    fsInput["NbrOfFrames"] >> sttngs.nbrOfFrames;
    fsInput["PatternType"] >> sttngs.patternType;
    fsInput.release();
}

double calibrate( vector< vector<Point3f> > objPoints, vector< vector<Point2f> > imgPoints,
               Mat &cameraMatrix, Mat &distCoeffs, vector<Mat> &r, vector<Mat> &t, Size imgSize )
{
    int calibFlags = 0;

    double rms = calibrateCamera(objPoints, imgPoints, imgSize, cameraMatrix,
                                distCoeffs, r, t, calibFlags);

    return rms;
}

void createObjectPoints( vector<Point3f> &patternCorners, Size patternSize, float squareSize,
                         int patternType )
{
    switch( patternType )
    {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for( int i = 0; i < patternSize.height; ++i )
            {
                for( int j = 0; j < patternSize.width; ++j )
                {
                    patternCorners.push_back(Point3f(float(i*squareSize), float(j*squareSize), 0));
                }
            }
            break;
        case ASYMETRIC_CIRCLES_GRID:
            break;
    }
}

void createProjectorObjectPoints( vector<Point2f> &patternCorners, Size patternSize, float squareSize,
                        int patternType )
{
    switch( patternType )
    {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for( int i = 1; i <= patternSize.height; ++i )
            {
                for( int j = 1; j <= patternSize.width; ++j )
                {
                    patternCorners.push_back(Point2f(float(j*squareSize), float(i*squareSize)));
                }
            }
            break;
        case ASYMETRIC_CIRCLES_GRID:
            break;
    }
}

void fromCamToWorld( Mat cameraMatrix, vector<Mat> rV, vector<Mat> tV,
                    vector< vector<Point2f> > imgPoints, vector< vector<Point3f> > &worldPoints )
{
    int s = (int) rV.size();
    Mat invK64, invK;
    invK64 = cameraMatrix.inv();
    invK64.convertTo(invK, CV_32F);

    for(int i = 0; i < s; ++i)
    {
        Mat r, t, rMat;
        rV[i].convertTo(r, CV_32F);
        tV[i].convertTo(t, CV_32F);

        Rodrigues(r, rMat);
        Mat transPlaneToCam = rMat.inv()*t;

        vector<Point3f> wpTemp;
        int s2 = (int) imgPoints[i].size();
        for(int j = 0; j < s2; ++j){
            Mat coords(3, 1, CV_32F);
            coords.at<float>(0, 0) = imgPoints[i][j].x;
            coords.at<float>(1, 0) = imgPoints[i][j].y;
            coords.at<float>(2, 0) = 1.0f;

            Mat worldPtCam = invK*coords;
            Mat worldPtPlane = rMat.inv()*worldPtCam;

            float scale = transPlaneToCam.at<float>(2)/worldPtPlane.at<float>(2);
            Mat worldPtPlaneReproject = scale*worldPtPlane - transPlaneToCam;

            Point3f pt;
            pt.x = worldPtPlaneReproject.at<float>(0);
            pt.y = worldPtPlaneReproject.at<float>(1);
            pt.z = 0;
            wpTemp.push_back(pt);
        }
        worldPoints.push_back(wpTemp);
    }
}

void saveCalibrationResults( String path, Mat camK, Mat camDistCoeffs, Mat projK, Mat projDistCoeffs,
                      Mat fundamental )
{
    FileStorage fs(path + ".yml", FileStorage::WRITE);
    fs << "camIntrinsics" << camK;
    fs << "camDistCoeffs" << camDistCoeffs;
    fs << "projIntrinsics" << projK;
    fs << "projDistCoeffs" << projDistCoeffs;
    fs << "fundamental" << fundamental;
    fs.release();
}

void saveCalibrationData( String path, vector<Mat> T1, vector<Mat> T2, vector<Mat> ptsProjCam, vector<Mat> ptsProjProj, vector<Mat> ptsProjCamN, vector<Mat> ptsProjProjN )
{
    FileStorage fs(path + ".yml", FileStorage::WRITE);

    int size = (int) T1.size();
    fs << "size" << size;
    for( int i = 0; i < (int)T1.size(); ++i )
    {
        ostringstream nbr;
        nbr << i;
        fs << "TprojCam" + nbr.str() << T1[i];
        fs << "TProjProj" + nbr.str() << T2[i];
        fs << "ptsProjCam" + nbr.str() << ptsProjCam[i];
        fs << "ptsProjProj" + nbr.str() << ptsProjProj[i];
        fs << "ptsProjCamN" + nbr.str() << ptsProjCamN[i];
        fs << "ptsProjProjN" + nbr.str() << ptsProjProjN[i];
    }
    fs.release();

}

void normalize( const Mat &pts, const int& dim, Mat& normpts, Mat &T )
{
    float averagedist = 0;
    float scale = 0;

    //centroid

    Mat centroid(dim,1,CV_32F);
    Scalar tmp;

    if( normpts.empty() )
    {
        normpts= Mat(pts.rows,pts.cols,CV_32F);
    }

    for( int i = 0 ; i < dim ; ++i )
    {
        tmp = mean(pts.row(i));
        centroid.at<float>(i,0) = (float)tmp[0];
        subtract(pts.row(i), centroid.at<float>(i, 0), normpts.row(i));
    }

    //average distance

    Mat ptstmp;
    for( int i = 0 ; i < normpts.cols; ++i )
    {
        ptstmp = normpts.col(i);
        averagedist = averagedist+(float)norm(ptstmp);
    }
    averagedist = averagedist / normpts.cols;
    scale = (float)(sqrt(dim) / averagedist);

    normpts = normpts * scale;

    T=cv::Mat::eye(dim+1,dim+1,CV_32F);
    for( int i = 0; i < dim; ++i )
    {
        T.at<float>(i, i) = scale;
        T.at<float>(i, dim) = -scale*centroid.at<float>(i, 0);
    }
}

void fromVectorToMat( vector<Point2f> v, Mat &pts )
{
    int nbrOfPoints = (int) v.size();

    if( pts.empty() )
        pts.create(2, nbrOfPoints, CV_32F);

    for( int i = 0; i < nbrOfPoints; ++i )
    {
        pts.at<float>(0, i) = v[i].x;
        pts.at<float>(1, i) = v[i].y;
    }
}

void fromMatToVector( Mat pts, vector<Point2f> &v )
{
    int nbrOfPoints = pts.cols;

    for( int i = 0; i < nbrOfPoints; ++i )
    {
        Point2f temp;
        temp.x = pts.at<float>(0, i);
        temp.y = pts.at<float>(1, i);
        v.push_back(temp);
    }
}