// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_structured_light_mono_calibration_HPP
#define OPENCV_structured_light_mono_calibration_HPP

#include <opencv2/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/utility.hpp"
#include <opencv2/calib3d.hpp>

using namespace std;

namespace cv{
namespace structured_light{

enum calibrationPattern{CHESSBOARD, CIRCLES_GRID, ASYMETRIC_CIRCLES_GRID};

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

void loadSettings(String path, Settings &sttngs);

void createObjectPoints( InputArrayOfArrays patternCorners, Size patternSize, float squareSize, int patternType );

void createProjectorObjectPoints(InputArrayOfArrays patternCorners, Size patternSize, float squareSize, int patternType );

double calibrate(InputArrayOfArrays objPoints, InputArrayOfArrays imgPoints, InputOutputArray cameraMatrix, InputOutputArray distCoeffs, OutputArrayOfArrays r, OutputArrayOfArrays t, Size imgSize );

void fromCamToWorld(InputArray cameraMatrix, InputArrayOfArrays rV, InputArrayOfArrays tV, InputArrayOfArrays imgPoints, OutputArrayOfArrays worldPoints );

void saveCalibrationResults( String path, InputArray camK, InputArray camDistCoeffs, InputArray projK, InputArray projDistCoeffs, InputArray fundamental);

void saveCalibrationData( String path, InputArrayOfArrays T1, InputArrayOfArrays T2,InputArrayOfArrays ptsProjCam, InputArrayOfArrays ptsProjProj, InputArrayOfArrays ptsProjCamN, InputArrayOfArrays ptsProjProjN);

void normalize(InputArray pts, const int& dim, InputOutputArray normpts, OutputArray T);

void fromVectorToMat(InputArrayOfArrays v, OutputArray pts);

void fromMatToVector(InputArray pts, OutputArrayOfArrays v);

void loadCalibrationData(string filename, OutputArray cameraIntrinsic, OutputArray projectorIntrinsic, OutputArray cameraDistortion, OutputArray projectorDistortion, OutputArray rotation, OutputArray translation);

void distortImage(InputArray input, InputArray camMat, InputArray dist, OutputArray output);

}
}

#endif
