
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.calib3d;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.utils.Converters;

public class Calib3d {

    public static final int
            CALIB_USE_INTRINSIC_GUESS = 1,
            CALIB_RECOMPUTE_EXTRINSIC = 2,
            CALIB_CHECK_COND = 4,
            CALIB_FIX_SKEW = 8,
            CALIB_FIX_K1 = 16,
            CALIB_FIX_K2 = 32,
            CALIB_FIX_K3 = 64,
            CALIB_FIX_K4 = 128,
            CALIB_FIX_INTRINSIC = 256,
            CV_ITERATIVE = 0,
            CV_EPNP = 1,
            CV_P3P = 2,
            CV_DLS = 3,
            LMEDS = 4,
            RANSAC = 8,
            RHO = 16,
            SOLVEPNP_ITERATIVE = 0,
            SOLVEPNP_EPNP = 1,
            SOLVEPNP_P3P = 2,
            SOLVEPNP_DLS = 3,
            SOLVEPNP_UPNP = 4,
            CALIB_CB_ADAPTIVE_THRESH = 1,
            CALIB_CB_NORMALIZE_IMAGE = 2,
            CALIB_CB_FILTER_QUADS = 4,
            CALIB_CB_FAST_CHECK = 8,
            CALIB_CB_SYMMETRIC_GRID = 1,
            CALIB_CB_ASYMMETRIC_GRID = 2,
            CALIB_CB_CLUSTERING = 4,
            CALIB_FIX_ASPECT_RATIO = 0x00002,
            CALIB_FIX_PRINCIPAL_POINT = 0x00004,
            CALIB_ZERO_TANGENT_DIST = 0x00008,
            CALIB_FIX_FOCAL_LENGTH = 0x00010,
            CALIB_FIX_K5 = 0x01000,
            CALIB_FIX_K6 = 0x02000,
            CALIB_RATIONAL_MODEL = 0x04000,
            CALIB_THIN_PRISM_MODEL = 0x08000,
            CALIB_FIX_S1_S2_S3_S4 = 0x10000,
            CALIB_SAME_FOCAL_LENGTH = 0x00200,
            CALIB_ZERO_DISPARITY = 0x00400,
            FM_7POINT = 1,
            FM_8POINT = 2,
            FM_LMEDS = 4,
            FM_RANSAC = 8;


    //
    // C++:  Mat findEssentialMat(Mat points1, Mat points2, double focal = 1.0, Point2d pp = Point2d(0, 0), int method = RANSAC, double prob = 0.999, double threshold = 1.0, Mat& mask = Mat())
    //

    //javadoc: findEssentialMat(points1, points2, focal, pp, method, prob, threshold, mask)
    public static Mat findEssentialMat(Mat points1, Mat points2, double focal, Point pp, int method, double prob, double threshold, Mat mask)
    {
        
        Mat retVal = new Mat(findEssentialMat_0(points1.nativeObj, points2.nativeObj, focal, pp.x, pp.y, method, prob, threshold, mask.nativeObj));
        
        return retVal;
    }

    //javadoc: findEssentialMat(points1, points2, focal, pp, method, prob, threshold)
    public static Mat findEssentialMat(Mat points1, Mat points2, double focal, Point pp, int method, double prob, double threshold)
    {
        
        Mat retVal = new Mat(findEssentialMat_1(points1.nativeObj, points2.nativeObj, focal, pp.x, pp.y, method, prob, threshold));
        
        return retVal;
    }

    //javadoc: findEssentialMat(points1, points2)
    public static Mat findEssentialMat(Mat points1, Mat points2)
    {
        
        Mat retVal = new Mat(findEssentialMat_2(points1.nativeObj, points2.nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat findFundamentalMat(vector_Point2f points1, vector_Point2f points2, int method = FM_RANSAC, double param1 = 3., double param2 = 0.99, Mat& mask = Mat())
    //

    //javadoc: findFundamentalMat(points1, points2, method, param1, param2, mask)
    public static Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2, int method, double param1, double param2, Mat mask)
    {
        Mat points1_mat = points1;
        Mat points2_mat = points2;
        Mat retVal = new Mat(findFundamentalMat_0(points1_mat.nativeObj, points2_mat.nativeObj, method, param1, param2, mask.nativeObj));
        
        return retVal;
    }

    //javadoc: findFundamentalMat(points1, points2, method, param1, param2)
    public static Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2, int method, double param1, double param2)
    {
        Mat points1_mat = points1;
        Mat points2_mat = points2;
        Mat retVal = new Mat(findFundamentalMat_1(points1_mat.nativeObj, points2_mat.nativeObj, method, param1, param2));
        
        return retVal;
    }

    //javadoc: findFundamentalMat(points1, points2)
    public static Mat findFundamentalMat(MatOfPoint2f points1, MatOfPoint2f points2)
    {
        Mat points1_mat = points1;
        Mat points2_mat = points2;
        Mat retVal = new Mat(findFundamentalMat_2(points1_mat.nativeObj, points2_mat.nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat findHomography(vector_Point2f srcPoints, vector_Point2f dstPoints, int method = 0, double ransacReprojThreshold = 3, Mat& mask = Mat(), int maxIters = 2000, double confidence = 0.995)
    //

    //javadoc: findHomography(srcPoints, dstPoints, method, ransacReprojThreshold, mask, maxIters, confidence)
    public static Mat findHomography(MatOfPoint2f srcPoints, MatOfPoint2f dstPoints, int method, double ransacReprojThreshold, Mat mask, int maxIters, double confidence)
    {
        Mat srcPoints_mat = srcPoints;
        Mat dstPoints_mat = dstPoints;
        Mat retVal = new Mat(findHomography_0(srcPoints_mat.nativeObj, dstPoints_mat.nativeObj, method, ransacReprojThreshold, mask.nativeObj, maxIters, confidence));
        
        return retVal;
    }

    //javadoc: findHomography(srcPoints, dstPoints, method, ransacReprojThreshold)
    public static Mat findHomography(MatOfPoint2f srcPoints, MatOfPoint2f dstPoints, int method, double ransacReprojThreshold)
    {
        Mat srcPoints_mat = srcPoints;
        Mat dstPoints_mat = dstPoints;
        Mat retVal = new Mat(findHomography_1(srcPoints_mat.nativeObj, dstPoints_mat.nativeObj, method, ransacReprojThreshold));
        
        return retVal;
    }

    //javadoc: findHomography(srcPoints, dstPoints)
    public static Mat findHomography(MatOfPoint2f srcPoints, MatOfPoint2f dstPoints)
    {
        Mat srcPoints_mat = srcPoints;
        Mat dstPoints_mat = dstPoints;
        Mat retVal = new Mat(findHomography_2(srcPoints_mat.nativeObj, dstPoints_mat.nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getOptimalNewCameraMatrix(Mat cameraMatrix, Mat distCoeffs, Size imageSize, double alpha, Size newImgSize = Size(), Rect* validPixROI = 0, bool centerPrincipalPoint = false)
    //

    //javadoc: getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha, newImgSize, validPixROI, centerPrincipalPoint)
    public static Mat getOptimalNewCameraMatrix(Mat cameraMatrix, Mat distCoeffs, Size imageSize, double alpha, Size newImgSize, Rect validPixROI, boolean centerPrincipalPoint)
    {
        double[] validPixROI_out = new double[4];
        Mat retVal = new Mat(getOptimalNewCameraMatrix_0(cameraMatrix.nativeObj, distCoeffs.nativeObj, imageSize.width, imageSize.height, alpha, newImgSize.width, newImgSize.height, validPixROI_out, centerPrincipalPoint));
        if(validPixROI!=null){ validPixROI.x = (int)validPixROI_out[0]; validPixROI.y = (int)validPixROI_out[1]; validPixROI.width = (int)validPixROI_out[2]; validPixROI.height = (int)validPixROI_out[3]; } 
        return retVal;
    }

    //javadoc: getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha)
    public static Mat getOptimalNewCameraMatrix(Mat cameraMatrix, Mat distCoeffs, Size imageSize, double alpha)
    {
        
        Mat retVal = new Mat(getOptimalNewCameraMatrix_1(cameraMatrix.nativeObj, distCoeffs.nativeObj, imageSize.width, imageSize.height, alpha));
        
        return retVal;
    }


    //
    // C++:  Mat initCameraMatrix2D(vector_vector_Point3f objectPoints, vector_vector_Point2f imagePoints, Size imageSize, double aspectRatio = 1.0)
    //

    //javadoc: initCameraMatrix2D(objectPoints, imagePoints, imageSize, aspectRatio)
    public static Mat initCameraMatrix2D(List<MatOfPoint3f> objectPoints, List<MatOfPoint2f> imagePoints, Size imageSize, double aspectRatio)
    {
        List<Mat> objectPoints_tmplm = new ArrayList<Mat>((objectPoints != null) ? objectPoints.size() : 0);
        Mat objectPoints_mat = Converters.vector_vector_Point3f_to_Mat(objectPoints, objectPoints_tmplm);
        List<Mat> imagePoints_tmplm = new ArrayList<Mat>((imagePoints != null) ? imagePoints.size() : 0);
        Mat imagePoints_mat = Converters.vector_vector_Point2f_to_Mat(imagePoints, imagePoints_tmplm);
        Mat retVal = new Mat(initCameraMatrix2D_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, aspectRatio));
        
        return retVal;
    }

    //javadoc: initCameraMatrix2D(objectPoints, imagePoints, imageSize)
    public static Mat initCameraMatrix2D(List<MatOfPoint3f> objectPoints, List<MatOfPoint2f> imagePoints, Size imageSize)
    {
        List<Mat> objectPoints_tmplm = new ArrayList<Mat>((objectPoints != null) ? objectPoints.size() : 0);
        Mat objectPoints_mat = Converters.vector_vector_Point3f_to_Mat(objectPoints, objectPoints_tmplm);
        List<Mat> imagePoints_tmplm = new ArrayList<Mat>((imagePoints != null) ? imagePoints.size() : 0);
        Mat imagePoints_mat = Converters.vector_vector_Point2f_to_Mat(imagePoints, imagePoints_tmplm);
        Mat retVal = new Mat(initCameraMatrix2D_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height));
        
        return retVal;
    }


    //
    // C++:  Rect getValidDisparityROI(Rect roi1, Rect roi2, int minDisparity, int numberOfDisparities, int SADWindowSize)
    //

    //javadoc: getValidDisparityROI(roi1, roi2, minDisparity, numberOfDisparities, SADWindowSize)
    public static Rect getValidDisparityROI(Rect roi1, Rect roi2, int minDisparity, int numberOfDisparities, int SADWindowSize)
    {
        
        Rect retVal = new Rect(getValidDisparityROI_0(roi1.x, roi1.y, roi1.width, roi1.height, roi2.x, roi2.y, roi2.width, roi2.height, minDisparity, numberOfDisparities, SADWindowSize));
        
        return retVal;
    }


    //
    // C++:  Vec3d RQDecomp3x3(Mat src, Mat& mtxR, Mat& mtxQ, Mat& Qx = Mat(), Mat& Qy = Mat(), Mat& Qz = Mat())
    //

    //javadoc: RQDecomp3x3(src, mtxR, mtxQ, Qx, Qy, Qz)
    public static double[] RQDecomp3x3(Mat src, Mat mtxR, Mat mtxQ, Mat Qx, Mat Qy, Mat Qz)
    {
        
        double[] retVal = RQDecomp3x3_0(src.nativeObj, mtxR.nativeObj, mtxQ.nativeObj, Qx.nativeObj, Qy.nativeObj, Qz.nativeObj);
        
        return retVal;
    }

    //javadoc: RQDecomp3x3(src, mtxR, mtxQ)
    public static double[] RQDecomp3x3(Mat src, Mat mtxR, Mat mtxQ)
    {
        
        double[] retVal = RQDecomp3x3_1(src.nativeObj, mtxR.nativeObj, mtxQ.nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool findChessboardCorners(Mat image, Size patternSize, vector_Point2f& corners, int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE)
    //

    //javadoc: findChessboardCorners(image, patternSize, corners, flags)
    public static boolean findChessboardCorners(Mat image, Size patternSize, MatOfPoint2f corners, int flags)
    {
        Mat corners_mat = corners;
        boolean retVal = findChessboardCorners_0(image.nativeObj, patternSize.width, patternSize.height, corners_mat.nativeObj, flags);
        
        return retVal;
    }

    //javadoc: findChessboardCorners(image, patternSize, corners)
    public static boolean findChessboardCorners(Mat image, Size patternSize, MatOfPoint2f corners)
    {
        Mat corners_mat = corners;
        boolean retVal = findChessboardCorners_1(image.nativeObj, patternSize.width, patternSize.height, corners_mat.nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool findCirclesGrid(Mat image, Size patternSize, Mat& centers, int flags = CALIB_CB_SYMMETRIC_GRID, Ptr_FeatureDetector blobDetector = SimpleBlobDetector::create())
    //

    //javadoc: findCirclesGrid(image, patternSize, centers, flags)
    public static boolean findCirclesGrid(Mat image, Size patternSize, Mat centers, int flags)
    {
        
        boolean retVal = findCirclesGrid_0(image.nativeObj, patternSize.width, patternSize.height, centers.nativeObj, flags);
        
        return retVal;
    }

    //javadoc: findCirclesGrid(image, patternSize, centers)
    public static boolean findCirclesGrid(Mat image, Size patternSize, Mat centers)
    {
        
        boolean retVal = findCirclesGrid_1(image.nativeObj, patternSize.width, patternSize.height, centers.nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool solvePnP(vector_Point3f objectPoints, vector_Point2f imagePoints, Mat cameraMatrix, vector_double distCoeffs, Mat& rvec, Mat& tvec, bool useExtrinsicGuess = false, int flags = SOLVEPNP_ITERATIVE)
    //

    //javadoc: solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, flags)
    public static boolean solvePnP(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, boolean useExtrinsicGuess, int flags)
    {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        boolean retVal = solvePnP_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, useExtrinsicGuess, flags);
        
        return retVal;
    }

    //javadoc: solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec)
    public static boolean solvePnP(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec)
    {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        boolean retVal = solvePnP_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool solvePnPRansac(vector_Point3f objectPoints, vector_Point2f imagePoints, Mat cameraMatrix, vector_double distCoeffs, Mat& rvec, Mat& tvec, bool useExtrinsicGuess = false, int iterationsCount = 100, float reprojectionError = 8.0, double confidence = 0.99, Mat& inliers = Mat(), int flags = SOLVEPNP_ITERATIVE)
    //

    //javadoc: solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers, flags)
    public static boolean solvePnPRansac(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec, boolean useExtrinsicGuess, int iterationsCount, float reprojectionError, double confidence, Mat inliers, int flags)
    {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        boolean retVal = solvePnPRansac_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers.nativeObj, flags);
        
        return retVal;
    }

    //javadoc: solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec)
    public static boolean solvePnPRansac(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs, Mat rvec, Mat tvec)
    {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        Mat distCoeffs_mat = distCoeffs;
        boolean retVal = solvePnPRansac_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, rvec.nativeObj, tvec.nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool stereoRectifyUncalibrated(Mat points1, Mat points2, Mat F, Size imgSize, Mat& H1, Mat& H2, double threshold = 5)
    //

    //javadoc: stereoRectifyUncalibrated(points1, points2, F, imgSize, H1, H2, threshold)
    public static boolean stereoRectifyUncalibrated(Mat points1, Mat points2, Mat F, Size imgSize, Mat H1, Mat H2, double threshold)
    {
        
        boolean retVal = stereoRectifyUncalibrated_0(points1.nativeObj, points2.nativeObj, F.nativeObj, imgSize.width, imgSize.height, H1.nativeObj, H2.nativeObj, threshold);
        
        return retVal;
    }

    //javadoc: stereoRectifyUncalibrated(points1, points2, F, imgSize, H1, H2)
    public static boolean stereoRectifyUncalibrated(Mat points1, Mat points2, Mat F, Size imgSize, Mat H1, Mat H2)
    {
        
        boolean retVal = stereoRectifyUncalibrated_1(points1.nativeObj, points2.nativeObj, F.nativeObj, imgSize.width, imgSize.height, H1.nativeObj, H2.nativeObj);
        
        return retVal;
    }


    //
    // C++:  double calibrateCamera(vector_Mat objectPoints, vector_Mat imagePoints, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, int flags = 0, TermCriteria criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))
    //

    //javadoc: calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria)
    public static double calibrateCamera(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, int flags, TermCriteria criteria)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCamera_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    //javadoc: calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags)
    public static double calibrateCamera(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs, int flags)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCamera_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, flags);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    //javadoc: calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs)
    public static double calibrateCamera(List<Mat> objectPoints, List<Mat> imagePoints, Size imageSize, Mat cameraMatrix, Mat distCoeffs, List<Mat> rvecs, List<Mat> tvecs)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrateCamera_2(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, imageSize.width, imageSize.height, cameraMatrix.nativeObj, distCoeffs.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }


    //
    // C++:  double stereoCalibrate(vector_Mat objectPoints, vector_Mat imagePoints1, vector_Mat imagePoints2, Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2, Size imageSize, Mat& R, Mat& T, Mat& E, Mat& F, int flags = CALIB_FIX_INTRINSIC, TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6))
    //

    //javadoc: stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, E, F, flags, criteria)
    public static double stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat E, Mat F, int flags, TermCriteria criteria)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        double retVal = stereoCalibrate_0(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, E.nativeObj, F.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
        
        return retVal;
    }

    //javadoc: stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, E, F, flags)
    public static double stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat E, Mat F, int flags)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        double retVal = stereoCalibrate_1(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, E.nativeObj, F.nativeObj, flags);
        
        return retVal;
    }

    //javadoc: stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, E, F)
    public static double stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat E, Mat F)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        double retVal = stereoCalibrate_2(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, E.nativeObj, F.nativeObj);
        
        return retVal;
    }


    //
    // C++:  double calibrate(vector_Mat objectPoints, vector_Mat imagePoints, Size image_size, Mat& K, Mat& D, vector_Mat& rvecs, vector_Mat& tvecs, int flags = 0, TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON))
    //

    //javadoc: calibrate(objectPoints, imagePoints, image_size, K, D, rvecs, tvecs, flags, criteria)
    public static double calibrate(List<Mat> objectPoints, List<Mat> imagePoints, Size image_size, Mat K, Mat D, List<Mat> rvecs, List<Mat> tvecs, int flags, TermCriteria criteria)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrate_0(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, image_size.width, image_size.height, K.nativeObj, D.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    //javadoc: calibrate(objectPoints, imagePoints, image_size, K, D, rvecs, tvecs, flags)
    public static double calibrate(List<Mat> objectPoints, List<Mat> imagePoints, Size image_size, Mat K, Mat D, List<Mat> rvecs, List<Mat> tvecs, int flags)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrate_1(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, image_size.width, image_size.height, K.nativeObj, D.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj, flags);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }

    //javadoc: calibrate(objectPoints, imagePoints, image_size, K, D, rvecs, tvecs)
    public static double calibrate(List<Mat> objectPoints, List<Mat> imagePoints, Size image_size, Mat K, Mat D, List<Mat> rvecs, List<Mat> tvecs)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints_mat = Converters.vector_Mat_to_Mat(imagePoints);
        Mat rvecs_mat = new Mat();
        Mat tvecs_mat = new Mat();
        double retVal = calibrate_2(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, image_size.width, image_size.height, K.nativeObj, D.nativeObj, rvecs_mat.nativeObj, tvecs_mat.nativeObj);
        Converters.Mat_to_vector_Mat(rvecs_mat, rvecs);
        rvecs_mat.release();
        Converters.Mat_to_vector_Mat(tvecs_mat, tvecs);
        tvecs_mat.release();
        return retVal;
    }


    //
    // C++:  double stereoCalibrate(vector_Mat objectPoints, vector_Mat imagePoints1, vector_Mat imagePoints2, Mat& K1, Mat& D1, Mat& K2, Mat& D2, Size imageSize, Mat& R, Mat& T, int flags = fisheye::CALIB_FIX_INTRINSIC, TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON))
    //

    //javadoc: stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R, T, flags, criteria)
    public static double stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat T, int flags, TermCriteria criteria)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        double retVal = stereoCalibrate_3(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, flags, criteria.type, criteria.maxCount, criteria.epsilon);
        
        return retVal;
    }

    //javadoc: stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R, T, flags)
    public static double stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat T, int flags)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        double retVal = stereoCalibrate_4(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, flags);
        
        return retVal;
    }

    //javadoc: stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R, T)
    public static double stereoCalibrate(List<Mat> objectPoints, List<Mat> imagePoints1, List<Mat> imagePoints2, Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat T)
    {
        Mat objectPoints_mat = Converters.vector_Mat_to_Mat(objectPoints);
        Mat imagePoints1_mat = Converters.vector_Mat_to_Mat(imagePoints1);
        Mat imagePoints2_mat = Converters.vector_Mat_to_Mat(imagePoints2);
        double retVal = stereoCalibrate_5(objectPoints_mat.nativeObj, imagePoints1_mat.nativeObj, imagePoints2_mat.nativeObj, K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj);
        
        return retVal;
    }


    //
    // C++:  float rectify3Collinear(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Mat cameraMatrix3, Mat distCoeffs3, vector_Mat imgpt1, vector_Mat imgpt3, Size imageSize, Mat R12, Mat T12, Mat R13, Mat T13, Mat& R1, Mat& R2, Mat& R3, Mat& P1, Mat& P2, Mat& P3, Mat& Q, double alpha, Size newImgSize, Rect* roi1, Rect* roi2, int flags)
    //

    //javadoc: rectify3Collinear(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, cameraMatrix3, distCoeffs3, imgpt1, imgpt3, imageSize, R12, T12, R13, T13, R1, R2, R3, P1, P2, P3, Q, alpha, newImgSize, roi1, roi2, flags)
    public static float rectify3Collinear(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Mat cameraMatrix3, Mat distCoeffs3, List<Mat> imgpt1, List<Mat> imgpt3, Size imageSize, Mat R12, Mat T12, Mat R13, Mat T13, Mat R1, Mat R2, Mat R3, Mat P1, Mat P2, Mat P3, Mat Q, double alpha, Size newImgSize, Rect roi1, Rect roi2, int flags)
    {
        Mat imgpt1_mat = Converters.vector_Mat_to_Mat(imgpt1);
        Mat imgpt3_mat = Converters.vector_Mat_to_Mat(imgpt3);
        double[] roi1_out = new double[4];
        double[] roi2_out = new double[4];
        float retVal = rectify3Collinear_0(cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, cameraMatrix3.nativeObj, distCoeffs3.nativeObj, imgpt1_mat.nativeObj, imgpt3_mat.nativeObj, imageSize.width, imageSize.height, R12.nativeObj, T12.nativeObj, R13.nativeObj, T13.nativeObj, R1.nativeObj, R2.nativeObj, R3.nativeObj, P1.nativeObj, P2.nativeObj, P3.nativeObj, Q.nativeObj, alpha, newImgSize.width, newImgSize.height, roi1_out, roi2_out, flags);
        if(roi1!=null){ roi1.x = (int)roi1_out[0]; roi1.y = (int)roi1_out[1]; roi1.width = (int)roi1_out[2]; roi1.height = (int)roi1_out[3]; } 
        if(roi2!=null){ roi2.x = (int)roi2_out[0]; roi2.y = (int)roi2_out[1]; roi2.width = (int)roi2_out[2]; roi2.height = (int)roi2_out[3]; } 
        return retVal;
    }


    //
    // C++:  int decomposeHomographyMat(Mat H, Mat K, vector_Mat& rotations, vector_Mat& translations, vector_Mat& normals)
    //

    //javadoc: decomposeHomographyMat(H, K, rotations, translations, normals)
    public static int decomposeHomographyMat(Mat H, Mat K, List<Mat> rotations, List<Mat> translations, List<Mat> normals)
    {
        Mat rotations_mat = new Mat();
        Mat translations_mat = new Mat();
        Mat normals_mat = new Mat();
        int retVal = decomposeHomographyMat_0(H.nativeObj, K.nativeObj, rotations_mat.nativeObj, translations_mat.nativeObj, normals_mat.nativeObj);
        Converters.Mat_to_vector_Mat(rotations_mat, rotations);
        rotations_mat.release();
        Converters.Mat_to_vector_Mat(translations_mat, translations);
        translations_mat.release();
        Converters.Mat_to_vector_Mat(normals_mat, normals);
        normals_mat.release();
        return retVal;
    }


    //
    // C++:  int estimateAffine3D(Mat src, Mat dst, Mat& out, Mat& inliers, double ransacThreshold = 3, double confidence = 0.99)
    //

    //javadoc: estimateAffine3D(src, dst, out, inliers, ransacThreshold, confidence)
    public static int estimateAffine3D(Mat src, Mat dst, Mat out, Mat inliers, double ransacThreshold, double confidence)
    {
        
        int retVal = estimateAffine3D_0(src.nativeObj, dst.nativeObj, out.nativeObj, inliers.nativeObj, ransacThreshold, confidence);
        
        return retVal;
    }

    //javadoc: estimateAffine3D(src, dst, out, inliers)
    public static int estimateAffine3D(Mat src, Mat dst, Mat out, Mat inliers)
    {
        
        int retVal = estimateAffine3D_1(src.nativeObj, dst.nativeObj, out.nativeObj, inliers.nativeObj);
        
        return retVal;
    }


    //
    // C++:  int recoverPose(Mat E, Mat points1, Mat points2, Mat& R, Mat& t, double focal = 1.0, Point2d pp = Point2d(0, 0), Mat& mask = Mat())
    //

    //javadoc: recoverPose(E, points1, points2, R, t, focal, pp, mask)
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat R, Mat t, double focal, Point pp, Mat mask)
    {
        
        int retVal = recoverPose_0(E.nativeObj, points1.nativeObj, points2.nativeObj, R.nativeObj, t.nativeObj, focal, pp.x, pp.y, mask.nativeObj);
        
        return retVal;
    }

    //javadoc: recoverPose(E, points1, points2, R, t, focal, pp)
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat R, Mat t, double focal, Point pp)
    {
        
        int retVal = recoverPose_1(E.nativeObj, points1.nativeObj, points2.nativeObj, R.nativeObj, t.nativeObj, focal, pp.x, pp.y);
        
        return retVal;
    }

    //javadoc: recoverPose(E, points1, points2, R, t)
    public static int recoverPose(Mat E, Mat points1, Mat points2, Mat R, Mat t)
    {
        
        int retVal = recoverPose_2(E.nativeObj, points1.nativeObj, points2.nativeObj, R.nativeObj, t.nativeObj);
        
        return retVal;
    }


    //
    // C++:  void Rodrigues(Mat src, Mat& dst, Mat& jacobian = Mat())
    //

    //javadoc: Rodrigues(src, dst, jacobian)
    public static void Rodrigues(Mat src, Mat dst, Mat jacobian)
    {
        
        Rodrigues_0(src.nativeObj, dst.nativeObj, jacobian.nativeObj);
        
        return;
    }

    //javadoc: Rodrigues(src, dst)
    public static void Rodrigues(Mat src, Mat dst)
    {
        
        Rodrigues_1(src.nativeObj, dst.nativeObj);
        
        return;
    }


    //
    // C++:  void calibrationMatrixValues(Mat cameraMatrix, Size imageSize, double apertureWidth, double apertureHeight, double& fovx, double& fovy, double& focalLength, Point2d& principalPoint, double& aspectRatio)
    //

    //javadoc: calibrationMatrixValues(cameraMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectRatio)
    public static void calibrationMatrixValues(Mat cameraMatrix, Size imageSize, double apertureWidth, double apertureHeight, double[] fovx, double[] fovy, double[] focalLength, Point principalPoint, double[] aspectRatio)
    {
        double[] fovx_out = new double[1];
        double[] fovy_out = new double[1];
        double[] focalLength_out = new double[1];
        double[] principalPoint_out = new double[2];
        double[] aspectRatio_out = new double[1];
        calibrationMatrixValues_0(cameraMatrix.nativeObj, imageSize.width, imageSize.height, apertureWidth, apertureHeight, fovx_out, fovy_out, focalLength_out, principalPoint_out, aspectRatio_out);
        if(fovx!=null) fovx[0] = fovx_out[0];
        if(fovy!=null) fovy[0] = fovy_out[0];
        if(focalLength!=null) focalLength[0] = focalLength_out[0];
        if(principalPoint!=null){ principalPoint.x = principalPoint_out[0]; principalPoint.y = principalPoint_out[1]; } 
        if(aspectRatio!=null) aspectRatio[0] = aspectRatio_out[0];
        return;
    }


    //
    // C++:  void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat& rvec3, Mat& tvec3, Mat& dr3dr1 = Mat(), Mat& dr3dt1 = Mat(), Mat& dr3dr2 = Mat(), Mat& dr3dt2 = Mat(), Mat& dt3dr1 = Mat(), Mat& dt3dt1 = Mat(), Mat& dt3dr2 = Mat(), Mat& dt3dt2 = Mat())
    //

    //javadoc: composeRT(rvec1, tvec1, rvec2, tvec2, rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2)
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3, Mat dr3dr1, Mat dr3dt1, Mat dr3dr2, Mat dr3dt2, Mat dt3dr1, Mat dt3dt1, Mat dt3dr2, Mat dt3dt2)
    {
        
        composeRT_0(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj, dr3dr1.nativeObj, dr3dt1.nativeObj, dr3dr2.nativeObj, dr3dt2.nativeObj, dt3dr1.nativeObj, dt3dt1.nativeObj, dt3dr2.nativeObj, dt3dt2.nativeObj);
        
        return;
    }

    //javadoc: composeRT(rvec1, tvec1, rvec2, tvec2, rvec3, tvec3)
    public static void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat rvec3, Mat tvec3)
    {
        
        composeRT_1(rvec1.nativeObj, tvec1.nativeObj, rvec2.nativeObj, tvec2.nativeObj, rvec3.nativeObj, tvec3.nativeObj);
        
        return;
    }


    //
    // C++:  void computeCorrespondEpilines(Mat points, int whichImage, Mat F, Mat& lines)
    //

    //javadoc: computeCorrespondEpilines(points, whichImage, F, lines)
    public static void computeCorrespondEpilines(Mat points, int whichImage, Mat F, Mat lines)
    {
        
        computeCorrespondEpilines_0(points.nativeObj, whichImage, F.nativeObj, lines.nativeObj);
        
        return;
    }


    //
    // C++:  void convertPointsFromHomogeneous(Mat src, Mat& dst)
    //

    //javadoc: convertPointsFromHomogeneous(src, dst)
    public static void convertPointsFromHomogeneous(Mat src, Mat dst)
    {
        
        convertPointsFromHomogeneous_0(src.nativeObj, dst.nativeObj);
        
        return;
    }


    //
    // C++:  void convertPointsToHomogeneous(Mat src, Mat& dst)
    //

    //javadoc: convertPointsToHomogeneous(src, dst)
    public static void convertPointsToHomogeneous(Mat src, Mat dst)
    {
        
        convertPointsToHomogeneous_0(src.nativeObj, dst.nativeObj);
        
        return;
    }


    //
    // C++:  void correctMatches(Mat F, Mat points1, Mat points2, Mat& newPoints1, Mat& newPoints2)
    //

    //javadoc: correctMatches(F, points1, points2, newPoints1, newPoints2)
    public static void correctMatches(Mat F, Mat points1, Mat points2, Mat newPoints1, Mat newPoints2)
    {
        
        correctMatches_0(F.nativeObj, points1.nativeObj, points2.nativeObj, newPoints1.nativeObj, newPoints2.nativeObj);
        
        return;
    }


    //
    // C++:  void decomposeEssentialMat(Mat E, Mat& R1, Mat& R2, Mat& t)
    //

    //javadoc: decomposeEssentialMat(E, R1, R2, t)
    public static void decomposeEssentialMat(Mat E, Mat R1, Mat R2, Mat t)
    {
        
        decomposeEssentialMat_0(E.nativeObj, R1.nativeObj, R2.nativeObj, t.nativeObj);
        
        return;
    }


    //
    // C++:  void decomposeProjectionMatrix(Mat projMatrix, Mat& cameraMatrix, Mat& rotMatrix, Mat& transVect, Mat& rotMatrixX = Mat(), Mat& rotMatrixY = Mat(), Mat& rotMatrixZ = Mat(), Mat& eulerAngles = Mat())
    //

    //javadoc: decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles)
    public static void decomposeProjectionMatrix(Mat projMatrix, Mat cameraMatrix, Mat rotMatrix, Mat transVect, Mat rotMatrixX, Mat rotMatrixY, Mat rotMatrixZ, Mat eulerAngles)
    {
        
        decomposeProjectionMatrix_0(projMatrix.nativeObj, cameraMatrix.nativeObj, rotMatrix.nativeObj, transVect.nativeObj, rotMatrixX.nativeObj, rotMatrixY.nativeObj, rotMatrixZ.nativeObj, eulerAngles.nativeObj);
        
        return;
    }

    //javadoc: decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect)
    public static void decomposeProjectionMatrix(Mat projMatrix, Mat cameraMatrix, Mat rotMatrix, Mat transVect)
    {
        
        decomposeProjectionMatrix_1(projMatrix.nativeObj, cameraMatrix.nativeObj, rotMatrix.nativeObj, transVect.nativeObj);
        
        return;
    }


    //
    // C++:  void drawChessboardCorners(Mat& image, Size patternSize, vector_Point2f corners, bool patternWasFound)
    //

    //javadoc: drawChessboardCorners(image, patternSize, corners, patternWasFound)
    public static void drawChessboardCorners(Mat image, Size patternSize, MatOfPoint2f corners, boolean patternWasFound)
    {
        Mat corners_mat = corners;
        drawChessboardCorners_0(image.nativeObj, patternSize.width, patternSize.height, corners_mat.nativeObj, patternWasFound);
        
        return;
    }


    //
    // C++:  void filterSpeckles(Mat& img, double newVal, int maxSpeckleSize, double maxDiff, Mat& buf = Mat())
    //

    //javadoc: filterSpeckles(img, newVal, maxSpeckleSize, maxDiff, buf)
    public static void filterSpeckles(Mat img, double newVal, int maxSpeckleSize, double maxDiff, Mat buf)
    {
        
        filterSpeckles_0(img.nativeObj, newVal, maxSpeckleSize, maxDiff, buf.nativeObj);
        
        return;
    }

    //javadoc: filterSpeckles(img, newVal, maxSpeckleSize, maxDiff)
    public static void filterSpeckles(Mat img, double newVal, int maxSpeckleSize, double maxDiff)
    {
        
        filterSpeckles_1(img.nativeObj, newVal, maxSpeckleSize, maxDiff);
        
        return;
    }


    //
    // C++:  void matMulDeriv(Mat A, Mat B, Mat& dABdA, Mat& dABdB)
    //

    //javadoc: matMulDeriv(A, B, dABdA, dABdB)
    public static void matMulDeriv(Mat A, Mat B, Mat dABdA, Mat dABdB)
    {
        
        matMulDeriv_0(A.nativeObj, B.nativeObj, dABdA.nativeObj, dABdB.nativeObj);
        
        return;
    }


    //
    // C++:  void projectPoints(vector_Point3f objectPoints, Mat rvec, Mat tvec, Mat cameraMatrix, vector_double distCoeffs, vector_Point2f& imagePoints, Mat& jacobian = Mat(), double aspectRatio = 0)
    //

    //javadoc: projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, jacobian, aspectRatio)
    public static void projectPoints(MatOfPoint3f objectPoints, Mat rvec, Mat tvec, Mat cameraMatrix, MatOfDouble distCoeffs, MatOfPoint2f imagePoints, Mat jacobian, double aspectRatio)
    {
        Mat objectPoints_mat = objectPoints;
        Mat distCoeffs_mat = distCoeffs;
        Mat imagePoints_mat = imagePoints;
        projectPoints_0(objectPoints_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, imagePoints_mat.nativeObj, jacobian.nativeObj, aspectRatio);
        
        return;
    }

    //javadoc: projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints)
    public static void projectPoints(MatOfPoint3f objectPoints, Mat rvec, Mat tvec, Mat cameraMatrix, MatOfDouble distCoeffs, MatOfPoint2f imagePoints)
    {
        Mat objectPoints_mat = objectPoints;
        Mat distCoeffs_mat = distCoeffs;
        Mat imagePoints_mat = imagePoints;
        projectPoints_1(objectPoints_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, cameraMatrix.nativeObj, distCoeffs_mat.nativeObj, imagePoints_mat.nativeObj);
        
        return;
    }


    //
    // C++:  void reprojectImageTo3D(Mat disparity, Mat& _3dImage, Mat Q, bool handleMissingValues = false, int ddepth = -1)
    //

    //javadoc: reprojectImageTo3D(disparity, _3dImage, Q, handleMissingValues, ddepth)
    public static void reprojectImageTo3D(Mat disparity, Mat _3dImage, Mat Q, boolean handleMissingValues, int ddepth)
    {
        
        reprojectImageTo3D_0(disparity.nativeObj, _3dImage.nativeObj, Q.nativeObj, handleMissingValues, ddepth);
        
        return;
    }

    //javadoc: reprojectImageTo3D(disparity, _3dImage, Q, handleMissingValues)
    public static void reprojectImageTo3D(Mat disparity, Mat _3dImage, Mat Q, boolean handleMissingValues)
    {
        
        reprojectImageTo3D_1(disparity.nativeObj, _3dImage.nativeObj, Q.nativeObj, handleMissingValues);
        
        return;
    }

    //javadoc: reprojectImageTo3D(disparity, _3dImage, Q)
    public static void reprojectImageTo3D(Mat disparity, Mat _3dImage, Mat Q)
    {
        
        reprojectImageTo3D_2(disparity.nativeObj, _3dImage.nativeObj, Q.nativeObj);
        
        return;
    }


    //
    // C++:  void stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, int flags = CALIB_ZERO_DISPARITY, double alpha = -1, Size newImageSize = Size(), Rect* validPixROI1 = 0, Rect* validPixROI2 = 0)
    //

    //javadoc: stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha, newImageSize, validPixROI1, validPixROI2)
    public static void stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags, double alpha, Size newImageSize, Rect validPixROI1, Rect validPixROI2)
    {
        double[] validPixROI1_out = new double[4];
        double[] validPixROI2_out = new double[4];
        stereoRectify_0(cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags, alpha, newImageSize.width, newImageSize.height, validPixROI1_out, validPixROI2_out);
        if(validPixROI1!=null){ validPixROI1.x = (int)validPixROI1_out[0]; validPixROI1.y = (int)validPixROI1_out[1]; validPixROI1.width = (int)validPixROI1_out[2]; validPixROI1.height = (int)validPixROI1_out[3]; } 
        if(validPixROI2!=null){ validPixROI2.x = (int)validPixROI2_out[0]; validPixROI2.y = (int)validPixROI2_out[1]; validPixROI2.width = (int)validPixROI2_out[2]; validPixROI2.height = (int)validPixROI2_out[3]; } 
        return;
    }

    //javadoc: stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q)
    public static void stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q)
    {
        
        stereoRectify_1(cameraMatrix1.nativeObj, distCoeffs1.nativeObj, cameraMatrix2.nativeObj, distCoeffs2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, T.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj);
        
        return;
    }


    //
    // C++:  void triangulatePoints(Mat projMatr1, Mat projMatr2, Mat projPoints1, Mat projPoints2, Mat& points4D)
    //

    //javadoc: triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2, points4D)
    public static void triangulatePoints(Mat projMatr1, Mat projMatr2, Mat projPoints1, Mat projPoints2, Mat points4D)
    {
        
        triangulatePoints_0(projMatr1.nativeObj, projMatr2.nativeObj, projPoints1.nativeObj, projPoints2.nativeObj, points4D.nativeObj);
        
        return;
    }


    //
    // C++:  void validateDisparity(Mat& disparity, Mat cost, int minDisparity, int numberOfDisparities, int disp12MaxDisp = 1)
    //

    //javadoc: validateDisparity(disparity, cost, minDisparity, numberOfDisparities, disp12MaxDisp)
    public static void validateDisparity(Mat disparity, Mat cost, int minDisparity, int numberOfDisparities, int disp12MaxDisp)
    {
        
        validateDisparity_0(disparity.nativeObj, cost.nativeObj, minDisparity, numberOfDisparities, disp12MaxDisp);
        
        return;
    }

    //javadoc: validateDisparity(disparity, cost, minDisparity, numberOfDisparities)
    public static void validateDisparity(Mat disparity, Mat cost, int minDisparity, int numberOfDisparities)
    {
        
        validateDisparity_1(disparity.nativeObj, cost.nativeObj, minDisparity, numberOfDisparities);
        
        return;
    }


    //
    // C++:  void distortPoints(Mat undistorted, Mat& distorted, Mat K, Mat D, double alpha = 0)
    //

    //javadoc: distortPoints(undistorted, distorted, K, D, alpha)
    public static void distortPoints(Mat undistorted, Mat distorted, Mat K, Mat D, double alpha)
    {
        
        distortPoints_0(undistorted.nativeObj, distorted.nativeObj, K.nativeObj, D.nativeObj, alpha);
        
        return;
    }

    //javadoc: distortPoints(undistorted, distorted, K, D)
    public static void distortPoints(Mat undistorted, Mat distorted, Mat K, Mat D)
    {
        
        distortPoints_1(undistorted.nativeObj, distorted.nativeObj, K.nativeObj, D.nativeObj);
        
        return;
    }


    //
    // C++:  void estimateNewCameraMatrixForUndistortRectify(Mat K, Mat D, Size image_size, Mat R, Mat& P, double balance = 0.0, Size new_size = Size(), double fov_scale = 1.0)
    //

    //javadoc: estimateNewCameraMatrixForUndistortRectify(K, D, image_size, R, P, balance, new_size, fov_scale)
    public static void estimateNewCameraMatrixForUndistortRectify(Mat K, Mat D, Size image_size, Mat R, Mat P, double balance, Size new_size, double fov_scale)
    {
        
        estimateNewCameraMatrixForUndistortRectify_0(K.nativeObj, D.nativeObj, image_size.width, image_size.height, R.nativeObj, P.nativeObj, balance, new_size.width, new_size.height, fov_scale);
        
        return;
    }

    //javadoc: estimateNewCameraMatrixForUndistortRectify(K, D, image_size, R, P)
    public static void estimateNewCameraMatrixForUndistortRectify(Mat K, Mat D, Size image_size, Mat R, Mat P)
    {
        
        estimateNewCameraMatrixForUndistortRectify_1(K.nativeObj, D.nativeObj, image_size.width, image_size.height, R.nativeObj, P.nativeObj);
        
        return;
    }


    //
    // C++:  void initUndistortRectifyMap(Mat K, Mat D, Mat R, Mat P, Size size, int m1type, Mat& map1, Mat& map2)
    //

    //javadoc: initUndistortRectifyMap(K, D, R, P, size, m1type, map1, map2)
    public static void initUndistortRectifyMap(Mat K, Mat D, Mat R, Mat P, Size size, int m1type, Mat map1, Mat map2)
    {
        
        initUndistortRectifyMap_0(K.nativeObj, D.nativeObj, R.nativeObj, P.nativeObj, size.width, size.height, m1type, map1.nativeObj, map2.nativeObj);
        
        return;
    }


    //
    // C++:  void projectPoints(vector_Point3f objectPoints, vector_Point2f& imagePoints, Mat rvec, Mat tvec, Mat K, Mat D, double alpha = 0, Mat& jacobian = Mat())
    //

    //javadoc: projectPoints(objectPoints, imagePoints, rvec, tvec, K, D, alpha, jacobian)
    public static void projectPoints(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat rvec, Mat tvec, Mat K, Mat D, double alpha, Mat jacobian)
    {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        projectPoints_2(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, K.nativeObj, D.nativeObj, alpha, jacobian.nativeObj);
        
        return;
    }

    //javadoc: projectPoints(objectPoints, imagePoints, rvec, tvec, K, D)
    public static void projectPoints(MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat rvec, Mat tvec, Mat K, Mat D)
    {
        Mat objectPoints_mat = objectPoints;
        Mat imagePoints_mat = imagePoints;
        projectPoints_3(objectPoints_mat.nativeObj, imagePoints_mat.nativeObj, rvec.nativeObj, tvec.nativeObj, K.nativeObj, D.nativeObj);
        
        return;
    }


    //
    // C++:  void stereoRectify(Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat tvec, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, int flags, Size newImageSize = Size(), double balance = 0.0, double fov_scale = 1.0)
    //

    //javadoc: stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, R1, R2, P1, P2, Q, flags, newImageSize, balance, fov_scale)
    public static void stereoRectify(Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat tvec, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags, Size newImageSize, double balance, double fov_scale)
    {
        
        stereoRectify_2(K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, tvec.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags, newImageSize.width, newImageSize.height, balance, fov_scale);
        
        return;
    }

    //javadoc: stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, R1, R2, P1, P2, Q, flags)
    public static void stereoRectify(Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat tvec, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, int flags)
    {
        
        stereoRectify_3(K1.nativeObj, D1.nativeObj, K2.nativeObj, D2.nativeObj, imageSize.width, imageSize.height, R.nativeObj, tvec.nativeObj, R1.nativeObj, R2.nativeObj, P1.nativeObj, P2.nativeObj, Q.nativeObj, flags);
        
        return;
    }


    //
    // C++:  void undistortImage(Mat distorted, Mat& undistorted, Mat K, Mat D, Mat Knew = cv::Mat(), Size new_size = Size())
    //

    //javadoc: undistortImage(distorted, undistorted, K, D, Knew, new_size)
    public static void undistortImage(Mat distorted, Mat undistorted, Mat K, Mat D, Mat Knew, Size new_size)
    {
        
        undistortImage_0(distorted.nativeObj, undistorted.nativeObj, K.nativeObj, D.nativeObj, Knew.nativeObj, new_size.width, new_size.height);
        
        return;
    }

    //javadoc: undistortImage(distorted, undistorted, K, D)
    public static void undistortImage(Mat distorted, Mat undistorted, Mat K, Mat D)
    {
        
        undistortImage_1(distorted.nativeObj, undistorted.nativeObj, K.nativeObj, D.nativeObj);
        
        return;
    }


    //
    // C++:  void undistortPoints(Mat distorted, Mat& undistorted, Mat K, Mat D, Mat R = Mat(), Mat P = Mat())
    //

    //javadoc: undistortPoints(distorted, undistorted, K, D, R, P)
    public static void undistortPoints(Mat distorted, Mat undistorted, Mat K, Mat D, Mat R, Mat P)
    {
        
        undistortPoints_0(distorted.nativeObj, undistorted.nativeObj, K.nativeObj, D.nativeObj, R.nativeObj, P.nativeObj);
        
        return;
    }

    //javadoc: undistortPoints(distorted, undistorted, K, D)
    public static void undistortPoints(Mat distorted, Mat undistorted, Mat K, Mat D)
    {
        
        undistortPoints_1(distorted.nativeObj, undistorted.nativeObj, K.nativeObj, D.nativeObj);
        
        return;
    }




    // C++:  Mat findEssentialMat(Mat points1, Mat points2, double focal = 1.0, Point2d pp = Point2d(0, 0), int method = RANSAC, double prob = 0.999, double threshold = 1.0, Mat& mask = Mat())
    private static native long findEssentialMat_0(long points1_nativeObj, long points2_nativeObj, double focal, double pp_x, double pp_y, int method, double prob, double threshold, long mask_nativeObj);
    private static native long findEssentialMat_1(long points1_nativeObj, long points2_nativeObj, double focal, double pp_x, double pp_y, int method, double prob, double threshold);
    private static native long findEssentialMat_2(long points1_nativeObj, long points2_nativeObj);

    // C++:  Mat findFundamentalMat(vector_Point2f points1, vector_Point2f points2, int method = FM_RANSAC, double param1 = 3., double param2 = 0.99, Mat& mask = Mat())
    private static native long findFundamentalMat_0(long points1_mat_nativeObj, long points2_mat_nativeObj, int method, double param1, double param2, long mask_nativeObj);
    private static native long findFundamentalMat_1(long points1_mat_nativeObj, long points2_mat_nativeObj, int method, double param1, double param2);
    private static native long findFundamentalMat_2(long points1_mat_nativeObj, long points2_mat_nativeObj);

    // C++:  Mat findHomography(vector_Point2f srcPoints, vector_Point2f dstPoints, int method = 0, double ransacReprojThreshold = 3, Mat& mask = Mat(), int maxIters = 2000, double confidence = 0.995)
    private static native long findHomography_0(long srcPoints_mat_nativeObj, long dstPoints_mat_nativeObj, int method, double ransacReprojThreshold, long mask_nativeObj, int maxIters, double confidence);
    private static native long findHomography_1(long srcPoints_mat_nativeObj, long dstPoints_mat_nativeObj, int method, double ransacReprojThreshold);
    private static native long findHomography_2(long srcPoints_mat_nativeObj, long dstPoints_mat_nativeObj);

    // C++:  Mat getOptimalNewCameraMatrix(Mat cameraMatrix, Mat distCoeffs, Size imageSize, double alpha, Size newImgSize = Size(), Rect* validPixROI = 0, bool centerPrincipalPoint = false)
    private static native long getOptimalNewCameraMatrix_0(long cameraMatrix_nativeObj, long distCoeffs_nativeObj, double imageSize_width, double imageSize_height, double alpha, double newImgSize_width, double newImgSize_height, double[] validPixROI_out, boolean centerPrincipalPoint);
    private static native long getOptimalNewCameraMatrix_1(long cameraMatrix_nativeObj, long distCoeffs_nativeObj, double imageSize_width, double imageSize_height, double alpha);

    // C++:  Mat initCameraMatrix2D(vector_vector_Point3f objectPoints, vector_vector_Point2f imagePoints, Size imageSize, double aspectRatio = 1.0)
    private static native long initCameraMatrix2D_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, double aspectRatio);
    private static native long initCameraMatrix2D_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height);

    // C++:  Rect getValidDisparityROI(Rect roi1, Rect roi2, int minDisparity, int numberOfDisparities, int SADWindowSize)
    private static native double[] getValidDisparityROI_0(int roi1_x, int roi1_y, int roi1_width, int roi1_height, int roi2_x, int roi2_y, int roi2_width, int roi2_height, int minDisparity, int numberOfDisparities, int SADWindowSize);

    // C++:  Vec3d RQDecomp3x3(Mat src, Mat& mtxR, Mat& mtxQ, Mat& Qx = Mat(), Mat& Qy = Mat(), Mat& Qz = Mat())
    private static native double[] RQDecomp3x3_0(long src_nativeObj, long mtxR_nativeObj, long mtxQ_nativeObj, long Qx_nativeObj, long Qy_nativeObj, long Qz_nativeObj);
    private static native double[] RQDecomp3x3_1(long src_nativeObj, long mtxR_nativeObj, long mtxQ_nativeObj);

    // C++:  bool findChessboardCorners(Mat image, Size patternSize, vector_Point2f& corners, int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE)
    private static native boolean findChessboardCorners_0(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_mat_nativeObj, int flags);
    private static native boolean findChessboardCorners_1(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_mat_nativeObj);

    // C++:  bool findCirclesGrid(Mat image, Size patternSize, Mat& centers, int flags = CALIB_CB_SYMMETRIC_GRID, Ptr_FeatureDetector blobDetector = SimpleBlobDetector::create())
    private static native boolean findCirclesGrid_0(long image_nativeObj, double patternSize_width, double patternSize_height, long centers_nativeObj, int flags);
    private static native boolean findCirclesGrid_1(long image_nativeObj, double patternSize_width, double patternSize_height, long centers_nativeObj);

    // C++:  bool solvePnP(vector_Point3f objectPoints, vector_Point2f imagePoints, Mat cameraMatrix, vector_double distCoeffs, Mat& rvec, Mat& tvec, bool useExtrinsicGuess = false, int flags = SOLVEPNP_ITERATIVE)
    private static native boolean solvePnP_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, boolean useExtrinsicGuess, int flags);
    private static native boolean solvePnP_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj);

    // C++:  bool solvePnPRansac(vector_Point3f objectPoints, vector_Point2f imagePoints, Mat cameraMatrix, vector_double distCoeffs, Mat& rvec, Mat& tvec, bool useExtrinsicGuess = false, int iterationsCount = 100, float reprojectionError = 8.0, double confidence = 0.99, Mat& inliers = Mat(), int flags = SOLVEPNP_ITERATIVE)
    private static native boolean solvePnPRansac_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, boolean useExtrinsicGuess, int iterationsCount, float reprojectionError, double confidence, long inliers_nativeObj, int flags);
    private static native boolean solvePnPRansac_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj);

    // C++:  bool stereoRectifyUncalibrated(Mat points1, Mat points2, Mat F, Size imgSize, Mat& H1, Mat& H2, double threshold = 5)
    private static native boolean stereoRectifyUncalibrated_0(long points1_nativeObj, long points2_nativeObj, long F_nativeObj, double imgSize_width, double imgSize_height, long H1_nativeObj, long H2_nativeObj, double threshold);
    private static native boolean stereoRectifyUncalibrated_1(long points1_nativeObj, long points2_nativeObj, long F_nativeObj, double imgSize_width, double imgSize_height, long H1_nativeObj, long H2_nativeObj);

    // C++:  double calibrateCamera(vector_Mat objectPoints, vector_Mat imagePoints, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs, vector_Mat& rvecs, vector_Mat& tvecs, int flags = 0, TermCriteria criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON))
    private static native double calibrateCamera_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double calibrateCamera_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, int flags);
    private static native double calibrateCamera_2(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double imageSize_width, double imageSize_height, long cameraMatrix_nativeObj, long distCoeffs_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj);

    // C++:  double stereoCalibrate(vector_Mat objectPoints, vector_Mat imagePoints1, vector_Mat imagePoints2, Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2, Size imageSize, Mat& R, Mat& T, Mat& E, Mat& F, int flags = CALIB_FIX_INTRINSIC, TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6))
    private static native double stereoCalibrate_0(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long E_nativeObj, long F_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double stereoCalibrate_1(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long E_nativeObj, long F_nativeObj, int flags);
    private static native double stereoCalibrate_2(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long E_nativeObj, long F_nativeObj);

    // C++:  double calibrate(vector_Mat objectPoints, vector_Mat imagePoints, Size image_size, Mat& K, Mat& D, vector_Mat& rvecs, vector_Mat& tvecs, int flags = 0, TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON))
    private static native double calibrate_0(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double image_size_width, double image_size_height, long K_nativeObj, long D_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double calibrate_1(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double image_size_width, double image_size_height, long K_nativeObj, long D_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj, int flags);
    private static native double calibrate_2(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, double image_size_width, double image_size_height, long K_nativeObj, long D_nativeObj, long rvecs_mat_nativeObj, long tvecs_mat_nativeObj);

    // C++:  double stereoCalibrate(vector_Mat objectPoints, vector_Mat imagePoints1, vector_Mat imagePoints2, Mat& K1, Mat& D1, Mat& K2, Mat& D2, Size imageSize, Mat& R, Mat& T, int flags = fisheye::CALIB_FIX_INTRINSIC, TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON))
    private static native double stereoCalibrate_3(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, int flags, int criteria_type, int criteria_maxCount, double criteria_epsilon);
    private static native double stereoCalibrate_4(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, int flags);
    private static native double stereoCalibrate_5(long objectPoints_mat_nativeObj, long imagePoints1_mat_nativeObj, long imagePoints2_mat_nativeObj, long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj);

    // C++:  float rectify3Collinear(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Mat cameraMatrix3, Mat distCoeffs3, vector_Mat imgpt1, vector_Mat imgpt3, Size imageSize, Mat R12, Mat T12, Mat R13, Mat T13, Mat& R1, Mat& R2, Mat& R3, Mat& P1, Mat& P2, Mat& P3, Mat& Q, double alpha, Size newImgSize, Rect* roi1, Rect* roi2, int flags)
    private static native float rectify3Collinear_0(long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, long cameraMatrix3_nativeObj, long distCoeffs3_nativeObj, long imgpt1_mat_nativeObj, long imgpt3_mat_nativeObj, double imageSize_width, double imageSize_height, long R12_nativeObj, long T12_nativeObj, long R13_nativeObj, long T13_nativeObj, long R1_nativeObj, long R2_nativeObj, long R3_nativeObj, long P1_nativeObj, long P2_nativeObj, long P3_nativeObj, long Q_nativeObj, double alpha, double newImgSize_width, double newImgSize_height, double[] roi1_out, double[] roi2_out, int flags);

    // C++:  int decomposeHomographyMat(Mat H, Mat K, vector_Mat& rotations, vector_Mat& translations, vector_Mat& normals)
    private static native int decomposeHomographyMat_0(long H_nativeObj, long K_nativeObj, long rotations_mat_nativeObj, long translations_mat_nativeObj, long normals_mat_nativeObj);

    // C++:  int estimateAffine3D(Mat src, Mat dst, Mat& out, Mat& inliers, double ransacThreshold = 3, double confidence = 0.99)
    private static native int estimateAffine3D_0(long src_nativeObj, long dst_nativeObj, long out_nativeObj, long inliers_nativeObj, double ransacThreshold, double confidence);
    private static native int estimateAffine3D_1(long src_nativeObj, long dst_nativeObj, long out_nativeObj, long inliers_nativeObj);

    // C++:  int recoverPose(Mat E, Mat points1, Mat points2, Mat& R, Mat& t, double focal = 1.0, Point2d pp = Point2d(0, 0), Mat& mask = Mat())
    private static native int recoverPose_0(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long R_nativeObj, long t_nativeObj, double focal, double pp_x, double pp_y, long mask_nativeObj);
    private static native int recoverPose_1(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long R_nativeObj, long t_nativeObj, double focal, double pp_x, double pp_y);
    private static native int recoverPose_2(long E_nativeObj, long points1_nativeObj, long points2_nativeObj, long R_nativeObj, long t_nativeObj);

    // C++:  void Rodrigues(Mat src, Mat& dst, Mat& jacobian = Mat())
    private static native void Rodrigues_0(long src_nativeObj, long dst_nativeObj, long jacobian_nativeObj);
    private static native void Rodrigues_1(long src_nativeObj, long dst_nativeObj);

    // C++:  void calibrationMatrixValues(Mat cameraMatrix, Size imageSize, double apertureWidth, double apertureHeight, double& fovx, double& fovy, double& focalLength, Point2d& principalPoint, double& aspectRatio)
    private static native void calibrationMatrixValues_0(long cameraMatrix_nativeObj, double imageSize_width, double imageSize_height, double apertureWidth, double apertureHeight, double[] fovx_out, double[] fovy_out, double[] focalLength_out, double[] principalPoint_out, double[] aspectRatio_out);

    // C++:  void composeRT(Mat rvec1, Mat tvec1, Mat rvec2, Mat tvec2, Mat& rvec3, Mat& tvec3, Mat& dr3dr1 = Mat(), Mat& dr3dt1 = Mat(), Mat& dr3dr2 = Mat(), Mat& dr3dt2 = Mat(), Mat& dt3dr1 = Mat(), Mat& dt3dt1 = Mat(), Mat& dt3dr2 = Mat(), Mat& dt3dt2 = Mat())
    private static native void composeRT_0(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj, long dr3dr1_nativeObj, long dr3dt1_nativeObj, long dr3dr2_nativeObj, long dr3dt2_nativeObj, long dt3dr1_nativeObj, long dt3dt1_nativeObj, long dt3dr2_nativeObj, long dt3dt2_nativeObj);
    private static native void composeRT_1(long rvec1_nativeObj, long tvec1_nativeObj, long rvec2_nativeObj, long tvec2_nativeObj, long rvec3_nativeObj, long tvec3_nativeObj);

    // C++:  void computeCorrespondEpilines(Mat points, int whichImage, Mat F, Mat& lines)
    private static native void computeCorrespondEpilines_0(long points_nativeObj, int whichImage, long F_nativeObj, long lines_nativeObj);

    // C++:  void convertPointsFromHomogeneous(Mat src, Mat& dst)
    private static native void convertPointsFromHomogeneous_0(long src_nativeObj, long dst_nativeObj);

    // C++:  void convertPointsToHomogeneous(Mat src, Mat& dst)
    private static native void convertPointsToHomogeneous_0(long src_nativeObj, long dst_nativeObj);

    // C++:  void correctMatches(Mat F, Mat points1, Mat points2, Mat& newPoints1, Mat& newPoints2)
    private static native void correctMatches_0(long F_nativeObj, long points1_nativeObj, long points2_nativeObj, long newPoints1_nativeObj, long newPoints2_nativeObj);

    // C++:  void decomposeEssentialMat(Mat E, Mat& R1, Mat& R2, Mat& t)
    private static native void decomposeEssentialMat_0(long E_nativeObj, long R1_nativeObj, long R2_nativeObj, long t_nativeObj);

    // C++:  void decomposeProjectionMatrix(Mat projMatrix, Mat& cameraMatrix, Mat& rotMatrix, Mat& transVect, Mat& rotMatrixX = Mat(), Mat& rotMatrixY = Mat(), Mat& rotMatrixZ = Mat(), Mat& eulerAngles = Mat())
    private static native void decomposeProjectionMatrix_0(long projMatrix_nativeObj, long cameraMatrix_nativeObj, long rotMatrix_nativeObj, long transVect_nativeObj, long rotMatrixX_nativeObj, long rotMatrixY_nativeObj, long rotMatrixZ_nativeObj, long eulerAngles_nativeObj);
    private static native void decomposeProjectionMatrix_1(long projMatrix_nativeObj, long cameraMatrix_nativeObj, long rotMatrix_nativeObj, long transVect_nativeObj);

    // C++:  void drawChessboardCorners(Mat& image, Size patternSize, vector_Point2f corners, bool patternWasFound)
    private static native void drawChessboardCorners_0(long image_nativeObj, double patternSize_width, double patternSize_height, long corners_mat_nativeObj, boolean patternWasFound);

    // C++:  void filterSpeckles(Mat& img, double newVal, int maxSpeckleSize, double maxDiff, Mat& buf = Mat())
    private static native void filterSpeckles_0(long img_nativeObj, double newVal, int maxSpeckleSize, double maxDiff, long buf_nativeObj);
    private static native void filterSpeckles_1(long img_nativeObj, double newVal, int maxSpeckleSize, double maxDiff);

    // C++:  void matMulDeriv(Mat A, Mat B, Mat& dABdA, Mat& dABdB)
    private static native void matMulDeriv_0(long A_nativeObj, long B_nativeObj, long dABdA_nativeObj, long dABdB_nativeObj);

    // C++:  void projectPoints(vector_Point3f objectPoints, Mat rvec, Mat tvec, Mat cameraMatrix, vector_double distCoeffs, vector_Point2f& imagePoints, Mat& jacobian = Mat(), double aspectRatio = 0)
    private static native void projectPoints_0(long objectPoints_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long imagePoints_mat_nativeObj, long jacobian_nativeObj, double aspectRatio);
    private static native void projectPoints_1(long objectPoints_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, long cameraMatrix_nativeObj, long distCoeffs_mat_nativeObj, long imagePoints_mat_nativeObj);

    // C++:  void reprojectImageTo3D(Mat disparity, Mat& _3dImage, Mat Q, bool handleMissingValues = false, int ddepth = -1)
    private static native void reprojectImageTo3D_0(long disparity_nativeObj, long _3dImage_nativeObj, long Q_nativeObj, boolean handleMissingValues, int ddepth);
    private static native void reprojectImageTo3D_1(long disparity_nativeObj, long _3dImage_nativeObj, long Q_nativeObj, boolean handleMissingValues);
    private static native void reprojectImageTo3D_2(long disparity_nativeObj, long _3dImage_nativeObj, long Q_nativeObj);

    // C++:  void stereoRectify(Mat cameraMatrix1, Mat distCoeffs1, Mat cameraMatrix2, Mat distCoeffs2, Size imageSize, Mat R, Mat T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, int flags = CALIB_ZERO_DISPARITY, double alpha = -1, Size newImageSize = Size(), Rect* validPixROI1 = 0, Rect* validPixROI2 = 0)
    private static native void stereoRectify_0(long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags, double alpha, double newImageSize_width, double newImageSize_height, double[] validPixROI1_out, double[] validPixROI2_out);
    private static native void stereoRectify_1(long cameraMatrix1_nativeObj, long distCoeffs1_nativeObj, long cameraMatrix2_nativeObj, long distCoeffs2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long T_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj);

    // C++:  void triangulatePoints(Mat projMatr1, Mat projMatr2, Mat projPoints1, Mat projPoints2, Mat& points4D)
    private static native void triangulatePoints_0(long projMatr1_nativeObj, long projMatr2_nativeObj, long projPoints1_nativeObj, long projPoints2_nativeObj, long points4D_nativeObj);

    // C++:  void validateDisparity(Mat& disparity, Mat cost, int minDisparity, int numberOfDisparities, int disp12MaxDisp = 1)
    private static native void validateDisparity_0(long disparity_nativeObj, long cost_nativeObj, int minDisparity, int numberOfDisparities, int disp12MaxDisp);
    private static native void validateDisparity_1(long disparity_nativeObj, long cost_nativeObj, int minDisparity, int numberOfDisparities);

    // C++:  void distortPoints(Mat undistorted, Mat& distorted, Mat K, Mat D, double alpha = 0)
    private static native void distortPoints_0(long undistorted_nativeObj, long distorted_nativeObj, long K_nativeObj, long D_nativeObj, double alpha);
    private static native void distortPoints_1(long undistorted_nativeObj, long distorted_nativeObj, long K_nativeObj, long D_nativeObj);

    // C++:  void estimateNewCameraMatrixForUndistortRectify(Mat K, Mat D, Size image_size, Mat R, Mat& P, double balance = 0.0, Size new_size = Size(), double fov_scale = 1.0)
    private static native void estimateNewCameraMatrixForUndistortRectify_0(long K_nativeObj, long D_nativeObj, double image_size_width, double image_size_height, long R_nativeObj, long P_nativeObj, double balance, double new_size_width, double new_size_height, double fov_scale);
    private static native void estimateNewCameraMatrixForUndistortRectify_1(long K_nativeObj, long D_nativeObj, double image_size_width, double image_size_height, long R_nativeObj, long P_nativeObj);

    // C++:  void initUndistortRectifyMap(Mat K, Mat D, Mat R, Mat P, Size size, int m1type, Mat& map1, Mat& map2)
    private static native void initUndistortRectifyMap_0(long K_nativeObj, long D_nativeObj, long R_nativeObj, long P_nativeObj, double size_width, double size_height, int m1type, long map1_nativeObj, long map2_nativeObj);

    // C++:  void projectPoints(vector_Point3f objectPoints, vector_Point2f& imagePoints, Mat rvec, Mat tvec, Mat K, Mat D, double alpha = 0, Mat& jacobian = Mat())
    private static native void projectPoints_2(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, long K_nativeObj, long D_nativeObj, double alpha, long jacobian_nativeObj);
    private static native void projectPoints_3(long objectPoints_mat_nativeObj, long imagePoints_mat_nativeObj, long rvec_nativeObj, long tvec_nativeObj, long K_nativeObj, long D_nativeObj);

    // C++:  void stereoRectify(Mat K1, Mat D1, Mat K2, Mat D2, Size imageSize, Mat R, Mat tvec, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, int flags, Size newImageSize = Size(), double balance = 0.0, double fov_scale = 1.0)
    private static native void stereoRectify_2(long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long tvec_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags, double newImageSize_width, double newImageSize_height, double balance, double fov_scale);
    private static native void stereoRectify_3(long K1_nativeObj, long D1_nativeObj, long K2_nativeObj, long D2_nativeObj, double imageSize_width, double imageSize_height, long R_nativeObj, long tvec_nativeObj, long R1_nativeObj, long R2_nativeObj, long P1_nativeObj, long P2_nativeObj, long Q_nativeObj, int flags);

    // C++:  void undistortImage(Mat distorted, Mat& undistorted, Mat K, Mat D, Mat Knew = cv::Mat(), Size new_size = Size())
    private static native void undistortImage_0(long distorted_nativeObj, long undistorted_nativeObj, long K_nativeObj, long D_nativeObj, long Knew_nativeObj, double new_size_width, double new_size_height);
    private static native void undistortImage_1(long distorted_nativeObj, long undistorted_nativeObj, long K_nativeObj, long D_nativeObj);

    // C++:  void undistortPoints(Mat distorted, Mat& undistorted, Mat K, Mat D, Mat R = Mat(), Mat P = Mat())
    private static native void undistortPoints_0(long distorted_nativeObj, long undistorted_nativeObj, long K_nativeObj, long D_nativeObj, long R_nativeObj, long P_nativeObj);
    private static native void undistortPoints_1(long distorted_nativeObj, long undistorted_nativeObj, long K_nativeObj, long D_nativeObj);

}
