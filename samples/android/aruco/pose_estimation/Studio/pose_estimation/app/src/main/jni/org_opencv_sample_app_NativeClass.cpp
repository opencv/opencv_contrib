#include <org_opencv_sample_app_NativeClass.h>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>
#include <android/log.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::aruco;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

static vector<Mat> calibrationImages;

static String cameraParams = "/sdcard/pose_estimation/calib.txt";
static String modelParams = "/sdcard/pose_estimation/detector_params.yml";



/**
 * To be replaced
 */
static bool readCameraParameters_old(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

/**
 * To be replaced
 */
static bool readDetectorParameters_old(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["doCornerRefinement"] >> params->doCornerRefinement;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}


/**
 * To be replaced
 */
/**
 */
static bool saveCameraParams(const string &filename, Size imageSize, float aspectRatio, int flags,
                             const Mat &cameraMatrix, const Mat &distCoeffs, double totalAvgErr) {
    FileStorage fs(filename, FileStorage::WRITE);
    if(!fs.isOpened())
        return false;

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    if(flags & CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

    if(flags != 0) {
        sprintf(buf, "flags: %s%s%s%s",
                flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;

    return true;
}

/**
 * To be made useful
 */
static bool readCameraParameters(JNIEnv * env, jobject cameraMatrixObj, Mat &camMatrix, Mat &distCoeffs) {

    jclass CameraMatrixClass = env->GetObjectClass(cameraMatrixObj);

    jfieldID fid_camera_matrix = env->GetStaticFieldID(CameraMatrixClass, "MainMatrix_ADDRESS", "L");
    camMatrix =*(Mat*)env->GetStaticLongField(CameraMatrixClass,fid_camera_matrix);

    jfieldID fid_dist_coeff = env->GetStaticFieldID(CameraMatrixClass, "DistortionCoefficients_ADDRESS", "L");
    distCoeffs =*(Mat*)env->GetStaticLongField(CameraMatrixClass,fid_dist_coeff);

    return true;
}

/**
 * To be made useful
 */
static bool readDetectorParameters(JNIEnv * env, jobject parameter, Ptr<aruco::DetectorParameters> &params) {

    jclass ParameterClass = env->GetObjectClass(parameter);

    //public static final int DEFAULT_ADAPTIVE_THRESH_WIN_SIZE_MIN= 3;
    jfieldID fid_adaptiveThreshWinSizeMin = env->GetStaticFieldID(ParameterClass, "ADAPTIVE_THRESH_WIN_SIZE_MIN", "I");
    params->adaptiveThreshWinSizeMin=env->GetStaticIntField(ParameterClass,fid_adaptiveThreshWinSizeMin);


    //public static final int DEFAULT_ADAPTIVE_THRESH_WIN_SIZE_MAX= 23;
    jfieldID fid_adaptiveThreshWinSizeMax = env->GetStaticFieldID(ParameterClass, "ADAPTIVE_THRESH_WIN_SIZE_MAX", "I");
    params->adaptiveThreshWinSizeMax=env->GetStaticIntField(ParameterClass,fid_adaptiveThreshWinSizeMax);

    //public static final int DEFAULT_ADAPTIVE_THRESH_WIN_SIZE_STEP= 10;
    jfieldID fid_adaptiveThreshWinSizeStep = env->GetStaticFieldID(ParameterClass, "ADAPTIVE_THRESH_WIN_SIZE_STEP", "I");
    params->adaptiveThreshWinSizeStep=env->GetStaticIntField(ParameterClass,fid_adaptiveThreshWinSizeStep);

    //public static final int DEFAULT_ADAPTIVE_THRESH_CONSTANT= 7;
    jfieldID fid_adaptiveThreshConstant = env->GetStaticFieldID(ParameterClass, "ADAPTIVE_THRESH_CONSTANT", "F");
    params->adaptiveThreshConstant=env->GetStaticFloatField(ParameterClass,fid_adaptiveThreshConstant);

    //public static final float DEFAULT_MIN_MARKER_PERIMETER_RATE= (float) 0.03;
    jfieldID fid_minMarkerPerimeterRate = env->GetStaticFieldID(ParameterClass, "MIN_MARKER_PERIMETER_RATE", "F");
    params->minMarkerPerimeterRate=env->GetStaticFloatField(ParameterClass,fid_minMarkerPerimeterRate);


    //public static final float DEFAULT_MAX_MARKER_PERIMETER_RATE= (float) 4.0;
    jfieldID fid_maxMarkerPerimeterRate = env->GetStaticFieldID(ParameterClass, "MAX_MARKER_PERIMETER_RATE", "F");
    params->maxMarkerPerimeterRate=env->GetStaticFloatField(ParameterClass,fid_maxMarkerPerimeterRate);


    //public static final float DEFAULT_POLYGONAL_APPROX_ACCURACY_RATE= (float)  0.05;
    jfieldID fid_polygonalApproxAccuracyRate = env->GetStaticFieldID(ParameterClass, "POLYGONAL_APPROX_ACCURACY_RATE", "F");
    params->polygonalApproxAccuracyRate=env->GetStaticFloatField(ParameterClass,fid_polygonalApproxAccuracyRate);


    //public static final float DEFAULT_MIN_CORNER_DISTANCE= (float)  10.0;
    jfieldID fid_minCornerDistanceRate = env->GetStaticFieldID(ParameterClass, "MIN_CORNER_DISTANCE", "F");
    params->minCornerDistanceRate=env->GetStaticFloatField(ParameterClass,fid_minCornerDistanceRate);


    //public static final int DEFAULT_MIN_DISTANCE_TO_BORDER= 3;
    jfieldID fid_minDistanceToBorder = env->GetStaticFieldID(ParameterClass, "MIN_DISTANCE_TO_BORDER", "I");
    params->minDistanceToBorder=env->GetStaticIntField(ParameterClass,fid_minDistanceToBorder);

    //public static final float DEFAULT_MIN_MARKER_DISTANCE_RATE= (float)  0.05;
    jfieldID fid_minMarkerDistanceRate = env->GetStaticFieldID(ParameterClass, "MIN_MARKER_DISTANCE_RATE", "F");
    params->minMarkerDistanceRate=env->GetStaticFloatField(ParameterClass,fid_minMarkerDistanceRate);

    //public static final boolean DEFAULT_DO_CORNER_REFINEMENT= false;
    jfieldID fid_doCornerRefinement = env->GetStaticFieldID(ParameterClass, "DO_CORNER_REFINEMENT", "Z");
    params->doCornerRefinement=env->GetStaticBooleanField(ParameterClass,fid_doCornerRefinement);


    //public static final int DEFAULT_CORNER_REFINEMENT_MAX_ITERATIONS= 30;
    jfieldID fid_cornerRefinementMaxIterations = env->GetStaticFieldID(ParameterClass, "CORNER_REFINEMENT_MAX_ITERATIONS", "I");
    params->cornerRefinementMaxIterations=env->GetStaticIntField(ParameterClass,fid_cornerRefinementMaxIterations);

    //public static final float DEFAULT_CORNER_REFINEMENT_MIN_ACCURACY= (float)  0.1;
    jfieldID fid_cornerRefinementMinAccuracy = env->GetStaticFieldID(ParameterClass, "CORNER_REFINEMENT_MIN_ACCURACY", "F");
    params->cornerRefinementMinAccuracy=env->GetStaticFloatField(ParameterClass,fid_cornerRefinementMinAccuracy);

    //public static final int DEFAULT_MARKER_BORDER_BITS= 1;
    jfieldID fid_markerBorderBits = env->GetStaticFieldID(ParameterClass, "MARKER_BORDER_BITS", "I");
    params->markerBorderBits=env->GetStaticIntField(ParameterClass,fid_markerBorderBits);

    //public static final int  DEFAULT_PERSPECTIVE_REMOVE_PIXEL_PER_CELL= 8;
    jfieldID fid_perspectiveRemovePixelPerCell = env->GetStaticFieldID(ParameterClass, "PERSPECTIVE_REMOVE_PIXEL_PER_CELL", "I");
    params->perspectiveRemovePixelPerCell=env->GetStaticIntField(ParameterClass,fid_perspectiveRemovePixelPerCell);

    //public static final float DEFAULT_PERSEPCTIVE_REMOVE_IGNORE_MARGIN_PER_CELL= (float)  0.13;
    jfieldID fid_perspectiveRemoveIgnoredMarginPerCell = env->GetStaticFieldID(ParameterClass, "PERSEPCTIVE_REMOVE_IGNORE_MARGIN_PER_CELL", "F");
    params->perspectiveRemoveIgnoredMarginPerCell=env->GetStaticFloatField(ParameterClass,fid_perspectiveRemoveIgnoredMarginPerCell);

    //public static final float DEFAULT_MAX_ERRONEOUS_BITS_IN_BORDER_RATE= (float)  0.04;
    jfieldID fid_maxErroneousBitsInBorderRate = env->GetStaticFieldID(ParameterClass, "MAX_ERRONEOUS_BITS_IN_BORDER_RATE", "F");
    params->maxErroneousBitsInBorderRate=env->GetStaticFloatField(ParameterClass,fid_maxErroneousBitsInBorderRate);

    //public static final float DEFAULT_MIN_OTSU_STD_DEV= (float)  5.0;
    jfieldID fid_minOtsuStdDev = env->GetStaticFieldID(ParameterClass, "MIN_OTSU_STD_DEV", "F");
    params->minOtsuStdDev=env->GetStaticFloatField(ParameterClass,fid_minOtsuStdDev);

    //public static final float DEFAULT_ERROR_CORRECTION_RATE= (float)  0.6;
    jfieldID fid_errorCorrectionRate = env->GetStaticFieldID(ParameterClass, "ERROR_CORRECTION_RATE", "F");
    params->errorCorrectionRate=env->GetStaticFloatField(ParameterClass,fid_errorCorrectionRate);

   //public static final int DEFAULT_CORNER_REFINEMENT_WINDOW_SIZE= 5;
    jfieldID fid_cornerRefinementWinSize = env->GetStaticFieldID(ParameterClass, "CORNER_REFINEMENT_WIN_SIZE", "I");
    params->cornerRefinementWinSize=env->GetStaticIntField(ParameterClass,fid_cornerRefinementWinSize);

    return true;
}

JNIEXPORT jstring JNICALL Java_org_opencv_sample_app_NativeClass_SendCalibrationImage
  (JNIEnv *env, jobject obj, jlong inputImage){
    Mat& image = *(Mat*)inputImage;
    calibrationImages.push_back(image);
    return env->NewStringUTF("Image Added!");
  }

JNIEXPORT jstring JNICALL Java_org_opencv_sample_app_NativeClass_ResetCalibration
  (JNIEnv *env, jobject obj){
    calibrationImages.clear();
    return env->NewStringUTF("Calibration Finished!");
  }

JNIEXPORT jstring JNICALL Java_org_opencv_sample_app_NativeClass_CalibrateCameraCharucoBoard
  (JNIEnv *env, jobject obj, jobject parameter, jobject cameraMatrixObj){
    env->ExceptionClear();

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    bool readOk;

    readOk=readDetectorParameters_old(modelParams, detectorParams);

    if(!readOk) {
          return env->NewStringUTF("Invalid detector parameters file");
      }
    //bool readOk = readDetectorParameters(env, parameter, detectorParams);

    jclass ParameterClass = env->GetObjectClass(parameter);

    //public static final int DEFAULT_CHARUCO_DIAMOND_D= 10;
    jfieldID fid_dictionaryId = env->GetStaticFieldID(ParameterClass, "CHARUCO_BOARD_D", "I");
    int dictionaryId = env->GetStaticIntField(ParameterClass,fid_dictionaryId);
    __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The dictionary id is %d", dictionaryId);
    //public static final float DEFAULT_CHARUCO_DIAMOND_SL= (float)  0.04;
    jfieldID fid_squaresX = env->GetStaticFieldID(ParameterClass, "CHARUCO_BOARD_W", "I");
    int squaresX =env->GetStaticIntField(ParameterClass,fid_squaresX);
    __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The square X is %d", squaresX);
    //public static final float DEFAULT_CHARUCO_DIAMOND_ML= (float)  0.02;
    jfieldID fid_squaresY = env->GetStaticFieldID(ParameterClass, "CHARUCO_BOARD_H", "I");
    int squaresY = env->GetStaticIntField(ParameterClass,fid_squaresY);
    __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The square Y is %d", squaresY);
    //public static final float DEFAULT_CHARUCO_DIAMOND_SL= (float)  0.04;
    jfieldID fid_squareLength = env->GetStaticFieldID(ParameterClass, "CHARUCO_BOARD_SL", "F");
    float squareLength =env->GetStaticFloatField(ParameterClass,fid_squareLength);
    __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The SquareLength is %f", squareLength);
    //public static final float DEFAULT_CHARUCO_DIAMOND_ML= (float)  0.02;
    jfieldID fid_markerLength = env->GetStaticFieldID(ParameterClass, "CHARUCO_BOARD_ML", "F");
    float markerLength = env->GetStaticFloatField(ParameterClass,fid_markerLength);
    __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The MarkerLength is %f", markerLength);
    bool estimatePose = true;

    int calibrationFlags = 0;
    float aspectRatio = 1;

    Ptr<aruco::Dictionary> dictionary =
            aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    // create charuco board object
    Ptr<aruco::CharucoBoard> charucoboard =
            aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
    Ptr<aruco::Board> board = charucoboard.staticCast<aruco::Board>();

    // collect data from each frame
    vector< vector< vector< Point2f > > > allCorners;
    vector< vector< int > > allIds;
    vector< Mat > allImgs;
    Size imgSize;

    __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "Calibrating on %d images", calibrationImages.size());

    for(int i=0;i<calibrationImages.size();i++){
        Mat image=calibrationImages[i];

        vector< int > ids;
        vector< vector< Point2f > > corners, rejected;

        // detect markers
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

        // interpolate charuco corners
        Mat currentCharucoCorners, currentCharucoIds;
        if(ids.size() > 0)
            aruco::interpolateCornersCharuco(corners, ids, image, charucoboard, currentCharucoCorners,
                                             currentCharucoIds);

        if(ids.size() > 0)
            aruco::drawDetectedMarkers(image, corners);

        if(currentCharucoCorners.total() > 0){
            aruco::drawDetectedCornersCharuco(image, currentCharucoCorners, currentCharucoIds);
        }
            allCorners.push_back(corners);
            allIds.push_back(ids);
            allImgs.push_back(image);
            imgSize = image.size();

    }

    if(allIds.size() < 1) {
        return env->NewStringUTF("Not enough captures for calibration");
    }

    Mat cameraMatrix, distCoeffs;
    vector< Mat > rvecs, tvecs;
    double repError;

// prepare data for charuco calibration
    int nFrames = (int)allCorners.size();
    vector< Mat > allCharucoCorners;
    vector< Mat > allCharucoIds;
    vector< Mat > filteredImages;
    allCharucoCorners.reserve(nFrames);
    allCharucoIds.reserve(nFrames);

    for(int i = 0; i < nFrames; i++) {
        // interpolate using camera parameters
        Mat currentCharucoCorners, currentCharucoIds;
        aruco::interpolateCornersCharuco(allCorners[i], allIds[i], allImgs[i], charucoboard,
                                         currentCharucoCorners, currentCharucoIds, cameraMatrix,
                                         distCoeffs);

        allCharucoCorners.push_back(currentCharucoCorners);
        allCharucoIds.push_back(currentCharucoIds);
        filteredImages.push_back(allImgs[i]);
    }

    if(allCharucoCorners.size() < 4) {
        return env->NewStringUTF("Not enough frames to calibrate!");
    }

    // calibrate camera using charuco
    repError =
        aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charucoboard, imgSize,
                                      cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);

    bool saveOk =  saveCameraParams(cameraParams, imgSize, aspectRatio, calibrationFlags,
                                        cameraMatrix, distCoeffs, repError);
    if(!saveOk) {
        return env->NewStringUTF("Unable to save file!");
    }

    return env->NewStringUTF("Calibration Successful!");
  }


JNIEXPORT jstring JNICALL Java_org_opencv_sample_app_NativeClass_EstimatePoseCharucoBoard
  (JNIEnv *env, jobject obj, jlong inputImage, jobject parameter, jobject cameraMatrix){
      env->ExceptionClear();

      Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
      Ptr<aruco::DetectorParameters> detectorParams2 = aruco::DetectorParameters::create();

      readDetectorParameters(env, parameter, detectorParams);
      bool readOk;

      jclass ParameterClass = env->GetObjectClass(parameter);

      readOk=readDetectorParameters_old(modelParams, detectorParams2);
      if(!readOk) {
        return env->NewStringUTF("Invalid detector parameters file");
      }

        //DEBUGGING

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "ADAPTIVE_THRESH_WIN_SIZE_MIN %d %d", detectorParams->adaptiveThreshWinSizeMin, detectorParams2->adaptiveThreshWinSizeMin);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_ADAPTIVE_THRESH_WIN_SIZE_MAX %d %d", detectorParams->adaptiveThreshWinSizeMax, detectorParams2->adaptiveThreshWinSizeMax);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_ADAPTIVE_THRESH_WIN_SIZE_STEP %d %d", detectorParams->adaptiveThreshWinSizeStep,detectorParams2->adaptiveThreshWinSizeStep);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_ADAPTIVE_THRESH_CONSTANT %f %f", detectorParams->adaptiveThreshConstant,detectorParams2->adaptiveThreshConstant);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_MIN_MARKER_PERIMETER_RATE %f  %f", detectorParams->minMarkerPerimeterRate,detectorParams2->minMarkerPerimeterRate);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_MIN_MARKER_PERIMETER_RATE %f %f", detectorParams->maxMarkerPerimeterRate,detectorParams2->maxMarkerPerimeterRate);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_POLYGONAL_APPROX_ACCURACY_RATE %f %f", detectorParams->polygonalApproxAccuracyRate,detectorParams2->polygonalApproxAccuracyRate);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_MIN_DISTANCE_TO_BORDER %d %d", detectorParams->minDistanceToBorder, detectorParams2->minDistanceToBorder);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_MIN_MARKER_DISTANCE_RATE %f %f", detectorParams->minMarkerDistanceRate, detectorParams2->minMarkerDistanceRate);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_CORNER_REFINEMENT_WIN_SIZE %d %d", detectorParams->cornerRefinementWinSize, detectorParams2->cornerRefinementWinSize);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_DO_CORNER_REFINEMENT %d %d", detectorParams->doCornerRefinement, detectorParams2->doCornerRefinement);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_CORNER_REFINEMENT_MAX_ITERATIONS %d %d", detectorParams->cornerRefinementMaxIterations, detectorParams2->cornerRefinementMaxIterations);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_CORNER_REFINEMENT_MIN_ACCURACY %f %f", detectorParams->cornerRefinementMinAccuracy, detectorParams2->cornerRefinementMinAccuracy);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_MARKER_BORDER_BITS %d %d", detectorParams->markerBorderBits, detectorParams2->markerBorderBits);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_PERSPECTIVE_REMOVE_PIXEL_PER_CELL %d %d", detectorParams->perspectiveRemovePixelPerCell, detectorParams2->perspectiveRemovePixelPerCell);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_PERSEPCTIVE_REMOVE_IGNORE_MARGIN_PER_CELL %f %f", detectorParams->perspectiveRemoveIgnoredMarginPerCell, detectorParams2->perspectiveRemoveIgnoredMarginPerCell);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_MAX_ERRONEOUS_BITS_IN_BORDER_RATE %f %f", detectorParams->maxErroneousBitsInBorderRate,detectorParams2->maxErroneousBitsInBorderRate);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_MIN_OTSU_STD_DEV %f %f", detectorParams->minOtsuStdDev,detectorParams2->minOtsuStdDev);

         __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "DEFAULT_ERROR_CORRECTION_RATE %f %f", detectorParams->errorCorrectionRate,detectorParams2->errorCorrectionRate);


      //public static final int DEFAULT_CHARUCO_DIAMOND_D= 10;
      jfieldID fid_dictionaryId = env->GetStaticFieldID(ParameterClass, "CHARUCO_BOARD_D", "I");
      int dictionaryId = env->GetStaticIntField(ParameterClass,fid_dictionaryId);
      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The dictionary id is %d", dictionaryId);
      //public static final float DEFAULT_CHARUCO_DIAMOND_SL= (float)  0.04;
      jfieldID fid_squaresX = env->GetStaticFieldID(ParameterClass, "CHARUCO_BOARD_W", "I");
      int squaresX =env->GetStaticIntField(ParameterClass,fid_squaresX);
      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The square X is %d", squaresX);
      //public static final float DEFAULT_CHARUCO_DIAMOND_ML= (float)  0.02;
      jfieldID fid_squaresY = env->GetStaticFieldID(ParameterClass, "CHARUCO_BOARD_H", "I");
      int squaresY = env->GetStaticIntField(ParameterClass,fid_squaresY);
      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The square Y is %d", squaresY);
      //public static final float DEFAULT_CHARUCO_DIAMOND_SL= (float)  0.04;
      jfieldID fid_squareLength = env->GetStaticFieldID(ParameterClass, "CHARUCO_BOARD_SL", "F");
      float squareLength =env->GetStaticFloatField(ParameterClass,fid_squareLength);
      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The SquareLength is %f", squareLength);
      //public static final float DEFAULT_CHARUCO_DIAMOND_ML= (float)  0.02;
      jfieldID fid_markerLength = env->GetStaticFieldID(ParameterClass, "CHARUCO_BOARD_ML", "F");
      float markerLength = env->GetStaticFloatField(ParameterClass,fid_markerLength);
      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The MarkerLength is %f", markerLength);
      bool estimatePose = true;

      Ptr<aruco::Dictionary> dictionary =
          aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

      Mat camMatrix, distCoeffs;
      if(estimatePose) {
            readOk = readCameraParameters_old(cameraParams, camMatrix, distCoeffs);
              if(!readOk) {
                  return env->NewStringUTF("Invalid camera parameters file");
              }
      }

      Mat& image = *(Mat*)inputImage;

      if (image.empty())
      {
          return env->NewStringUTF("Can't read image from the file");
      }


      float axisLength = 0.5f * ((float)min(squaresX, squaresY) * (squareLength));

      // create charuco board object
      Ptr<aruco::CharucoBoard> charucoboard =
            aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
      Ptr<aruco::Board> board = charucoboard.staticCast<aruco::Board>();

      vector< int > markerIds, charucoIds;
      vector< vector< Point2f > > markerCorners, rejectedMarkers;
      vector< Point2f > charucoCorners;
      Vec3d rvec, tvec;

      // detect markers
      aruco::detectMarkers(image, dictionary, markerCorners, markerIds, detectorParams2,
                           rejectedMarkers);


      // interpolate charuco corners
      int interpolatedCorners = 0;
      if(markerIds.size() > 0)
          interpolatedCorners =
              aruco::interpolateCornersCharuco(markerCorners, markerIds, image, charucoboard,
                                               charucoCorners, charucoIds, camMatrix, distCoeffs);

      // estimate charuco board pose
      bool validPose = false;

      if(camMatrix.total() != 0)
          validPose = aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, charucoboard,
                                                      camMatrix, distCoeffs, rvec, tvec);

      if(markerIds.size() > 0) {
          aruco::drawDetectedMarkers(image, markerCorners);
      }
      else{
          return env->NewStringUTF("Could Not Find Any Marker IDs!");
      }

      if(interpolatedCorners > 0) {
          Scalar color;
          color = Scalar(255, 0, 0);
          aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds, color);
      }
      else{
          return env->NewStringUTF("Could Not Find Interpolated Corners!");
      }

      if(validPose)
          aruco::drawAxis(image, camMatrix, distCoeffs, rvec, tvec, axisLength);
      else{
          return env->NewStringUTF("Could Not Find Valid Pose!");
      }
      return env->NewStringUTF("Awesome!");
  }


JNIEXPORT jstring JNICALL Java_org_opencv_sample_app_NativeClass_EstimatePoseCharucoDiamond
        (JNIEnv * env, jobject obj, jlong inputImage, jobject parameter, jobject cameraMatrix){

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

      bool readOk;

      readOk=readDetectorParameters_old(modelParams, detectorParams);
      if(!readOk) {
        return env->NewStringUTF("Invalid detector parameters file");
      }
    jclass ParameterClass = env->GetObjectClass(parameter);

    //public static final int DEFAULT_CHARUCO_DIAMOND_D= 10;
    jfieldID fid_dictionaryId = env->GetStaticFieldID(ParameterClass, "CHARUCO_DIAMOND_D", "I");
    int dictionaryId = env->GetStaticIntField(ParameterClass,fid_dictionaryId);

    //public static final float DEFAULT_CHARUCO_DIAMOND_SL= (float)  0.04;
    jfieldID fid_squareLength = env->GetStaticFieldID(ParameterClass, "CHARUCO_DIAMOND_SL", "F");
    float squareLength =env->GetStaticFloatField(ParameterClass,fid_squareLength);

    //public static final float DEFAULT_CHARUCO_DIAMOND_ML= (float)  0.02;
    jfieldID fid_markerLength = env->GetStaticFieldID(ParameterClass, "CHARUCO_DIAMOND_ML", "F");
    float markerLength = env->GetStaticFloatField(ParameterClass,fid_markerLength);

    bool estimatePose = true;

    if(!readOk) {
        return env->NewStringUTF("Invalid detector parameters file");
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Mat camMatrix, distCoeffs;
      if(estimatePose) {
            readOk = readCameraParameters_old(cameraParams, camMatrix, distCoeffs);
              if(!readOk) {
                  return env->NewStringUTF("Invalid camera parameters file");
              }
      }

    Mat& image = *(Mat*)inputImage;

    if (image.empty())
    {
        return env->NewStringUTF("Can't read image from the file");
        exit(-1);
    }

    vector< int > markerIds;
    vector< Vec4i > diamondIds;
    vector< vector< Point2f > > markerCorners, rejectedMarkers, diamondCorners;
    vector< Vec3d > rvecs, tvecs;

    // detect markers
    aruco::detectMarkers(image, dictionary, markerCorners, markerIds, detectorParams,
                         rejectedMarkers);

    // detect diamonds
    if(markerIds.size() > 0)
        aruco::detectCharucoDiamond(image, markerCorners, markerIds,
                                    squareLength / markerLength, diamondCorners, diamondIds,
                                    camMatrix, distCoeffs);

    // estimate diamond pose
    if(estimatePose && diamondIds.size() > 0) {
        aruco::estimatePoseSingleMarkers(diamondCorners, squareLength, camMatrix,
                                                         distCoeffs, rvecs, tvecs);
    }

    if(markerIds.size() > 0){
            aruco::drawDetectedMarkers(image, markerCorners);
    }else{
        return env->NewStringUTF("No markers found!");
    }


    if(diamondIds.size() > 0) {
        aruco::drawDetectedDiamonds(image, diamondCorners, diamondIds);
        if(estimatePose) {
            for(unsigned int i = 0; i < diamondIds.size(); i++)
                aruco::drawAxis(image, camMatrix, distCoeffs, rvecs[i], tvecs[i],
                                squareLength * 0.5f);
        }
    }
    else{
        return env->NewStringUTF("No diamonds found!");
    }

    return env->NewStringUTF("Awesome!");

}

JNIEXPORT jstring JNICALL Java_org_opencv_sample_app_NativeClass_CalibrateCameraArucoBoard
  (JNIEnv *env, jobject obj, jobject parameter, jobject cameraMatrixObj){
      env->ExceptionClear();

      Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
      bool readOk;

      readOk=readDetectorParameters_old(modelParams, detectorParams);

      if(!readOk) {
            return env->NewStringUTF("Invalid detector parameters file");
        }
      //bool readOk = readDetectorParameters(env, parameter, detectorParams);

      jclass ParameterClass = env->GetObjectClass(parameter);

      //public static final int DEFAULT_CHARUCO_DIAMOND_D= 10;
      jfieldID fid_dictionaryId = env->GetStaticFieldID(ParameterClass, "ARUCO_BOARD_D", "I");
      int dictionaryId = env->GetStaticIntField(ParameterClass,fid_dictionaryId);
      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The dictionary id is %d", dictionaryId);
      //public static final float DEFAULT_CHARUCO_DIAMOND_SL= (float)  0.04;
      jfieldID fid_squaresX = env->GetStaticFieldID(ParameterClass, "ARUCO_BOARD_W", "I");
      int markersX =env->GetStaticIntField(ParameterClass,fid_squaresX);
      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The square X is %d", markersX);
      //public static final float DEFAULT_CHARUCO_DIAMOND_ML= (float)  0.02;
      jfieldID fid_squaresY = env->GetStaticFieldID(ParameterClass, "ARUCO_BOARD_H", "I");
      int markersY = env->GetStaticIntField(ParameterClass,fid_squaresY);
      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The square Y is %d", markersY);
      //public static final float DEFAULT_CHARUCO_DIAMOND_SL= (float)  0.04;
      jfieldID fid_squareLength = env->GetStaticFieldID(ParameterClass, "ARUCO_BOARD_S", "F");
      float markerSeparation =env->GetStaticFloatField(ParameterClass,fid_squareLength);
      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The SquareLength is %f", markerSeparation);
      //public static final float DEFAULT_CHARUCO_DIAMOND_ML= (float)  0.02;
      jfieldID fid_markerLength = env->GetStaticFieldID(ParameterClass, "ARUCO_BOARD_L", "F");
      float markerLength = env->GetStaticFloatField(ParameterClass,fid_markerLength);
      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The MarkerLength is %f", markerLength);
      bool estimatePose = true;

      int calibrationFlags = 0;
      float aspectRatio = 1;

      Ptr<aruco::Dictionary> dictionary =
              aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

      // create board object
      Ptr<aruco::GridBoard> gridboard =
              aruco::GridBoard::create(markersX, markersY, markerLength, markerSeparation, dictionary);
      Ptr<aruco::Board> board = gridboard.staticCast<aruco::Board>();

      // collect data from each frame
      vector< vector< vector< Point2f > > > allCorners;
      vector< vector< int > > allIds;
      vector< Mat > allImgs;
      Size imgSize;

      __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "Calibrating on %d images", calibrationImages.size());

      for(int i=0;i<calibrationImages.size();i++){
          Mat image=calibrationImages[i];

          vector< int > ids;
          vector< vector< Point2f > > corners, rejected;

          // detect markers
          aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);


              if(ids.size() > 0) aruco::drawDetectedMarkers(image, corners, ids);

              allCorners.push_back(corners);
              allIds.push_back(ids);
              allImgs.push_back(image);
              imgSize = image.size();

      }

      if(allIds.size() < 1) {
          return env->NewStringUTF("Not enough captures for calibration");
      }

    Mat cameraMatrix, distCoeffs;
    vector< Mat > rvecs, tvecs;
    double repError;

    if(calibrationFlags & CALIB_FIX_ASPECT_RATIO) {
        cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at< double >(0, 0) = aspectRatio;
    }

    // prepare data for calibration
    vector< vector< Point2f > > allCornersConcatenated;
    vector< int > allIdsConcatenated;
    vector< int > markerCounterPerFrame;
    markerCounterPerFrame.reserve(allCorners.size());
    for(unsigned int i = 0; i < allCorners.size(); i++) {
        markerCounterPerFrame.push_back((int)allCorners[i].size());
        for(unsigned int j = 0; j < allCorners[i].size(); j++) {
            allCornersConcatenated.push_back(allCorners[i][j]);
            allIdsConcatenated.push_back(allIds[i][j]);
        }
    }
    // calibrate camera
    repError = aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated,
                                           markerCounterPerFrame, board, imgSize, cameraMatrix,
                                           distCoeffs, rvecs, tvecs, calibrationFlags);

    bool saveOk = saveCameraParams(cameraParams, imgSize, aspectRatio, calibrationFlags, cameraMatrix,
                                   distCoeffs, repError);

      if(!saveOk) {
          return env->NewStringUTF("Unable to save file!");
      }

      return env->NewStringUTF("Calibration Successful!");
  }

JNIEXPORT jstring JNICALL Java_org_opencv_sample_app_NativeClass_EstimatePoseArucoBoard
 (JNIEnv *env, jobject obj, jlong inputImage, jobject parameter, jobject cameraMatrix){
       env->ExceptionClear();

       Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

       readDetectorParameters(env, parameter, detectorParams);
       bool readOk;

       jclass ParameterClass = env->GetObjectClass(parameter);

       readOk=readDetectorParameters_old(modelParams, detectorParams);

       if(!readOk) {
         return env->NewStringUTF("Invalid detector parameters file");
       }

       //public static final int DEFAULT_CHARUCO_DIAMOND_D= 10;
       jfieldID fid_dictionaryId = env->GetStaticFieldID(ParameterClass, "ARUCO_BOARD_D", "I");
       int dictionaryId = env->GetStaticIntField(ParameterClass,fid_dictionaryId);
       __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The dictionary id is %d", dictionaryId);
       //public static final float DEFAULT_CHARUCO_DIAMOND_SL= (float)  0.04;
       jfieldID fid_markersX = env->GetStaticFieldID(ParameterClass, "ARUCO_BOARD_W", "I");
       int markersX =env->GetStaticIntField(ParameterClass,fid_markersX);
       __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The square X is %d", markersX);
       //public static final float DEFAULT_CHARUCO_DIAMOND_ML= (float)  0.02;
       jfieldID fid_markersY = env->GetStaticFieldID(ParameterClass, "ARUCO_BOARD_H", "I");
       int markersY = env->GetStaticIntField(ParameterClass,fid_markersY);
       __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The square Y is %d", markersY);
       //public static final float DEFAULT_CHARUCO_DIAMOND_SL= (float)  0.04;
       jfieldID fid_markerSeparation = env->GetStaticFieldID(ParameterClass, "ARUCO_BOARD_S", "F");
       float markerSeparation =env->GetStaticFloatField(ParameterClass,fid_markerSeparation);
       __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The SquareLength is %f", markerSeparation);
       //public static final float DEFAULT_CHARUCO_DIAMOND_ML= (float)  0.02;
       jfieldID fid_markerLength = env->GetStaticFieldID(ParameterClass, "ARUCO_BOARD_L", "F");
       float markerLength = env->GetStaticFloatField(ParameterClass,fid_markerLength);
       __android_log_print(ANDROID_LOG_VERBOSE, "MyApp", "The MarkerLength is %f", markerLength);
       bool estimatePose = true;

       Ptr<aruco::Dictionary> dictionary =
           aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

       Mat camMatrix, distCoeffs;
       if(estimatePose) {
             readOk = readCameraParameters_old(cameraParams, camMatrix, distCoeffs);
               if(!readOk) {
                   return env->NewStringUTF("Invalid camera parameters file");
               }
       }

       Mat& image = *(Mat*)inputImage;

       if (image.empty())
       {
           return env->NewStringUTF("Can't read image from the file");
       }


       float axisLength = 0.5f * ((float)min(markersX, markersY) * (markerLength));

       // create charuco board object
       Ptr<aruco::GridBoard> gridboard =
           aruco::GridBoard::create(markersX, markersY, markerLength, markerSeparation, dictionary);
       Ptr<aruco::Board> board = gridboard.staticCast<aruco::Board>();

       vector< int > ids;
       vector< vector< Point2f > > corners, rejected;
       Vec3d rvec, tvec;

       // detect markers
       aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

       // estimate board pose
       int markersOfBoardDetected = 0;
       if(ids.size() > 0)
           markersOfBoardDetected =
               aruco::estimatePoseBoard(corners, ids, board, camMatrix, distCoeffs, rvec, tvec);

       if(ids.size() > 0) {
           aruco::drawDetectedMarkers(image, corners, ids);
       }
       else{
            return env->NewStringUTF("No markers detected!");
       }

       if(markersOfBoardDetected > 0)
           aruco::drawAxis(image, camMatrix, distCoeffs, rvec, tvec, axisLength);
       else
          return env->NewStringUTF("No Boards!");
       return env->NewStringUTF("Awesome!");
   }