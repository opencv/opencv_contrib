
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import java.lang.String;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.utils.Converters;

// C++: class javaFeatureDetector
//javadoc: javaFeatureDetector
public class FeatureDetector {

    protected final long nativeObj;
    protected FeatureDetector(long addr) { nativeObj = addr; }


    private static final int
            GRIDDETECTOR = 1000,
            PYRAMIDDETECTOR = 2000,
            DYNAMICDETECTOR = 3000;


    public static final int
            FAST = 1,
            STAR = 2,
            SIFT = 3,
            SURF = 4,
            ORB = 5,
            MSER = 6,
            GFTT = 7,
            HARRIS = 8,
            SIMPLEBLOB = 9,
            DENSE = 10,
            BRISK = 11,
            AKAZE = 12,
            GRID_FAST = GRIDDETECTOR + FAST,
            GRID_STAR = GRIDDETECTOR + STAR,
            GRID_SIFT = GRIDDETECTOR + SIFT,
            GRID_SURF = GRIDDETECTOR + SURF,
            GRID_ORB = GRIDDETECTOR + ORB,
            GRID_MSER = GRIDDETECTOR + MSER,
            GRID_GFTT = GRIDDETECTOR + GFTT,
            GRID_HARRIS = GRIDDETECTOR + HARRIS,
            GRID_SIMPLEBLOB = GRIDDETECTOR + SIMPLEBLOB,
            GRID_DENSE = GRIDDETECTOR + DENSE,
            GRID_BRISK = GRIDDETECTOR + BRISK,
            GRID_AKAZE = GRIDDETECTOR + AKAZE,
            PYRAMID_FAST = PYRAMIDDETECTOR + FAST,
            PYRAMID_STAR = PYRAMIDDETECTOR + STAR,
            PYRAMID_SIFT = PYRAMIDDETECTOR + SIFT,
            PYRAMID_SURF = PYRAMIDDETECTOR + SURF,
            PYRAMID_ORB = PYRAMIDDETECTOR + ORB,
            PYRAMID_MSER = PYRAMIDDETECTOR + MSER,
            PYRAMID_GFTT = PYRAMIDDETECTOR + GFTT,
            PYRAMID_HARRIS = PYRAMIDDETECTOR + HARRIS,
            PYRAMID_SIMPLEBLOB = PYRAMIDDETECTOR + SIMPLEBLOB,
            PYRAMID_DENSE = PYRAMIDDETECTOR + DENSE,
            PYRAMID_BRISK = PYRAMIDDETECTOR + BRISK,
            PYRAMID_AKAZE = PYRAMIDDETECTOR + AKAZE,
            DYNAMIC_FAST = DYNAMICDETECTOR + FAST,
            DYNAMIC_STAR = DYNAMICDETECTOR + STAR,
            DYNAMIC_SIFT = DYNAMICDETECTOR + SIFT,
            DYNAMIC_SURF = DYNAMICDETECTOR + SURF,
            DYNAMIC_ORB = DYNAMICDETECTOR + ORB,
            DYNAMIC_MSER = DYNAMICDETECTOR + MSER,
            DYNAMIC_GFTT = DYNAMICDETECTOR + GFTT,
            DYNAMIC_HARRIS = DYNAMICDETECTOR + HARRIS,
            DYNAMIC_SIMPLEBLOB = DYNAMICDETECTOR + SIMPLEBLOB,
            DYNAMIC_DENSE = DYNAMICDETECTOR + DENSE,
            DYNAMIC_BRISK = DYNAMICDETECTOR + BRISK,
            DYNAMIC_AKAZE = DYNAMICDETECTOR + AKAZE;


    //
    // C++:  bool empty()
    //

    //javadoc: javaFeatureDetector::empty()
    public  boolean empty()
    {
        
        boolean retVal = empty_0(nativeObj);
        
        return retVal;
    }


    //
    // C++: static javaFeatureDetector* create(int detectorType)
    //

    //javadoc: javaFeatureDetector::create(detectorType)
    public static FeatureDetector create(int detectorType)
    {
        
        FeatureDetector retVal = new FeatureDetector(create_0(detectorType));
        
        return retVal;
    }


    //
    // C++:  void detect(Mat image, vector_KeyPoint& keypoints, Mat mask = Mat())
    //

    //javadoc: javaFeatureDetector::detect(image, keypoints, mask)
    public  void detect(Mat image, MatOfKeyPoint keypoints, Mat mask)
    {
        Mat keypoints_mat = keypoints;
        detect_0(nativeObj, image.nativeObj, keypoints_mat.nativeObj, mask.nativeObj);
        
        return;
    }

    //javadoc: javaFeatureDetector::detect(image, keypoints)
    public  void detect(Mat image, MatOfKeyPoint keypoints)
    {
        Mat keypoints_mat = keypoints;
        detect_1(nativeObj, image.nativeObj, keypoints_mat.nativeObj);
        
        return;
    }


    //
    // C++:  void detect(vector_Mat images, vector_vector_KeyPoint& keypoints, vector_Mat masks = std::vector<Mat>())
    //

    //javadoc: javaFeatureDetector::detect(images, keypoints, masks)
    public  void detect(List<Mat> images, List<MatOfKeyPoint> keypoints, List<Mat> masks)
    {
        Mat images_mat = Converters.vector_Mat_to_Mat(images);
        Mat keypoints_mat = new Mat();
        Mat masks_mat = Converters.vector_Mat_to_Mat(masks);
        detect_2(nativeObj, images_mat.nativeObj, keypoints_mat.nativeObj, masks_mat.nativeObj);
        Converters.Mat_to_vector_vector_KeyPoint(keypoints_mat, keypoints);
        keypoints_mat.release();
        return;
    }

    //javadoc: javaFeatureDetector::detect(images, keypoints)
    public  void detect(List<Mat> images, List<MatOfKeyPoint> keypoints)
    {
        Mat images_mat = Converters.vector_Mat_to_Mat(images);
        Mat keypoints_mat = new Mat();
        detect_3(nativeObj, images_mat.nativeObj, keypoints_mat.nativeObj);
        Converters.Mat_to_vector_vector_KeyPoint(keypoints_mat, keypoints);
        keypoints_mat.release();
        return;
    }


    //
    // C++:  void read(String fileName)
    //

    //javadoc: javaFeatureDetector::read(fileName)
    public  void read(String fileName)
    {
        
        read_0(nativeObj, fileName);
        
        return;
    }


    //
    // C++:  void write(String fileName)
    //

    //javadoc: javaFeatureDetector::write(fileName)
    public  void write(String fileName)
    {
        
        write_0(nativeObj, fileName);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  bool empty()
    private static native boolean empty_0(long nativeObj);

    // C++: static javaFeatureDetector* create(int detectorType)
    private static native long create_0(int detectorType);

    // C++:  void detect(Mat image, vector_KeyPoint& keypoints, Mat mask = Mat())
    private static native void detect_0(long nativeObj, long image_nativeObj, long keypoints_mat_nativeObj, long mask_nativeObj);
    private static native void detect_1(long nativeObj, long image_nativeObj, long keypoints_mat_nativeObj);

    // C++:  void detect(vector_Mat images, vector_vector_KeyPoint& keypoints, vector_Mat masks = std::vector<Mat>())
    private static native void detect_2(long nativeObj, long images_mat_nativeObj, long keypoints_mat_nativeObj, long masks_mat_nativeObj);
    private static native void detect_3(long nativeObj, long images_mat_nativeObj, long keypoints_mat_nativeObj);

    // C++:  void read(String fileName)
    private static native void read_0(long nativeObj, String fileName);

    // C++:  void write(String fileName)
    private static native void write_0(long nativeObj, String fileName);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
