
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.features2d;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.utils.Converters;

public class Features2d {

    public static final int
            DRAW_OVER_OUTIMG = 1,
            NOT_DRAW_SINGLE_POINTS = 2,
            DRAW_RICH_KEYPOINTS = 4;


    //
    // C++:  void drawKeypoints(Mat image, vector_KeyPoint keypoints, Mat outImage, Scalar color = Scalar::all(-1), int flags = 0)
    //

    //javadoc: drawKeypoints(image, keypoints, outImage, color, flags)
    public static void drawKeypoints(Mat image, MatOfKeyPoint keypoints, Mat outImage, Scalar color, int flags)
    {
        Mat keypoints_mat = keypoints;
        drawKeypoints_0(image.nativeObj, keypoints_mat.nativeObj, outImage.nativeObj, color.val[0], color.val[1], color.val[2], color.val[3], flags);
        
        return;
    }

    //javadoc: drawKeypoints(image, keypoints, outImage)
    public static void drawKeypoints(Mat image, MatOfKeyPoint keypoints, Mat outImage)
    {
        Mat keypoints_mat = keypoints;
        drawKeypoints_1(image.nativeObj, keypoints_mat.nativeObj, outImage.nativeObj);
        
        return;
    }


    //
    // C++:  void drawMatches(Mat img1, vector_KeyPoint keypoints1, Mat img2, vector_KeyPoint keypoints2, vector_DMatch matches1to2, Mat outImg, Scalar matchColor = Scalar::all(-1), Scalar singlePointColor = Scalar::all(-1), vector_char matchesMask = std::vector<char>(), int flags = 0)
    //

    //javadoc: drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor, singlePointColor, matchesMask, flags)
    public static void drawMatches(Mat img1, MatOfKeyPoint keypoints1, Mat img2, MatOfKeyPoint keypoints2, MatOfDMatch matches1to2, Mat outImg, Scalar matchColor, Scalar singlePointColor, MatOfByte matchesMask, int flags)
    {
        Mat keypoints1_mat = keypoints1;
        Mat keypoints2_mat = keypoints2;
        Mat matches1to2_mat = matches1to2;
        Mat matchesMask_mat = matchesMask;
        drawMatches_0(img1.nativeObj, keypoints1_mat.nativeObj, img2.nativeObj, keypoints2_mat.nativeObj, matches1to2_mat.nativeObj, outImg.nativeObj, matchColor.val[0], matchColor.val[1], matchColor.val[2], matchColor.val[3], singlePointColor.val[0], singlePointColor.val[1], singlePointColor.val[2], singlePointColor.val[3], matchesMask_mat.nativeObj, flags);
        
        return;
    }

    //javadoc: drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg)
    public static void drawMatches(Mat img1, MatOfKeyPoint keypoints1, Mat img2, MatOfKeyPoint keypoints2, MatOfDMatch matches1to2, Mat outImg)
    {
        Mat keypoints1_mat = keypoints1;
        Mat keypoints2_mat = keypoints2;
        Mat matches1to2_mat = matches1to2;
        drawMatches_1(img1.nativeObj, keypoints1_mat.nativeObj, img2.nativeObj, keypoints2_mat.nativeObj, matches1to2_mat.nativeObj, outImg.nativeObj);
        
        return;
    }


    //
    // C++:  void drawMatches(Mat img1, vector_KeyPoint keypoints1, Mat img2, vector_KeyPoint keypoints2, vector_vector_DMatch matches1to2, Mat outImg, Scalar matchColor = Scalar::all(-1), Scalar singlePointColor = Scalar::all(-1), vector_vector_char matchesMask = std::vector<std::vector<char> >(), int flags = 0)
    //

    //javadoc: drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor, singlePointColor, matchesMask, flags)
    public static void drawMatches2(Mat img1, MatOfKeyPoint keypoints1, Mat img2, MatOfKeyPoint keypoints2, List<MatOfDMatch> matches1to2, Mat outImg, Scalar matchColor, Scalar singlePointColor, List<MatOfByte> matchesMask, int flags)
    {
        Mat keypoints1_mat = keypoints1;
        Mat keypoints2_mat = keypoints2;
        List<Mat> matches1to2_tmplm = new ArrayList<Mat>((matches1to2 != null) ? matches1to2.size() : 0);
        Mat matches1to2_mat = Converters.vector_vector_DMatch_to_Mat(matches1to2, matches1to2_tmplm);
        List<Mat> matchesMask_tmplm = new ArrayList<Mat>((matchesMask != null) ? matchesMask.size() : 0);
        Mat matchesMask_mat = Converters.vector_vector_char_to_Mat(matchesMask, matchesMask_tmplm);
        drawMatches2_0(img1.nativeObj, keypoints1_mat.nativeObj, img2.nativeObj, keypoints2_mat.nativeObj, matches1to2_mat.nativeObj, outImg.nativeObj, matchColor.val[0], matchColor.val[1], matchColor.val[2], matchColor.val[3], singlePointColor.val[0], singlePointColor.val[1], singlePointColor.val[2], singlePointColor.val[3], matchesMask_mat.nativeObj, flags);
        
        return;
    }

    //javadoc: drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg)
    public static void drawMatches2(Mat img1, MatOfKeyPoint keypoints1, Mat img2, MatOfKeyPoint keypoints2, List<MatOfDMatch> matches1to2, Mat outImg)
    {
        Mat keypoints1_mat = keypoints1;
        Mat keypoints2_mat = keypoints2;
        List<Mat> matches1to2_tmplm = new ArrayList<Mat>((matches1to2 != null) ? matches1to2.size() : 0);
        Mat matches1to2_mat = Converters.vector_vector_DMatch_to_Mat(matches1to2, matches1to2_tmplm);
        drawMatches2_1(img1.nativeObj, keypoints1_mat.nativeObj, img2.nativeObj, keypoints2_mat.nativeObj, matches1to2_mat.nativeObj, outImg.nativeObj);
        
        return;
    }




    // C++:  void drawKeypoints(Mat image, vector_KeyPoint keypoints, Mat outImage, Scalar color = Scalar::all(-1), int flags = 0)
    private static native void drawKeypoints_0(long image_nativeObj, long keypoints_mat_nativeObj, long outImage_nativeObj, double color_val0, double color_val1, double color_val2, double color_val3, int flags);
    private static native void drawKeypoints_1(long image_nativeObj, long keypoints_mat_nativeObj, long outImage_nativeObj);

    // C++:  void drawMatches(Mat img1, vector_KeyPoint keypoints1, Mat img2, vector_KeyPoint keypoints2, vector_DMatch matches1to2, Mat outImg, Scalar matchColor = Scalar::all(-1), Scalar singlePointColor = Scalar::all(-1), vector_char matchesMask = std::vector<char>(), int flags = 0)
    private static native void drawMatches_0(long img1_nativeObj, long keypoints1_mat_nativeObj, long img2_nativeObj, long keypoints2_mat_nativeObj, long matches1to2_mat_nativeObj, long outImg_nativeObj, double matchColor_val0, double matchColor_val1, double matchColor_val2, double matchColor_val3, double singlePointColor_val0, double singlePointColor_val1, double singlePointColor_val2, double singlePointColor_val3, long matchesMask_mat_nativeObj, int flags);
    private static native void drawMatches_1(long img1_nativeObj, long keypoints1_mat_nativeObj, long img2_nativeObj, long keypoints2_mat_nativeObj, long matches1to2_mat_nativeObj, long outImg_nativeObj);

    // C++:  void drawMatches(Mat img1, vector_KeyPoint keypoints1, Mat img2, vector_KeyPoint keypoints2, vector_vector_DMatch matches1to2, Mat outImg, Scalar matchColor = Scalar::all(-1), Scalar singlePointColor = Scalar::all(-1), vector_vector_char matchesMask = std::vector<std::vector<char> >(), int flags = 0)
    private static native void drawMatches2_0(long img1_nativeObj, long keypoints1_mat_nativeObj, long img2_nativeObj, long keypoints2_mat_nativeObj, long matches1to2_mat_nativeObj, long outImg_nativeObj, double matchColor_val0, double matchColor_val1, double matchColor_val2, double matchColor_val3, double singlePointColor_val0, double singlePointColor_val1, double singlePointColor_val2, double singlePointColor_val3, long matchesMask_mat_nativeObj, int flags);
    private static native void drawMatches2_1(long img1_nativeObj, long keypoints1_mat_nativeObj, long img2_nativeObj, long keypoints2_mat_nativeObj, long matches1to2_mat_nativeObj, long outImg_nativeObj);

}
