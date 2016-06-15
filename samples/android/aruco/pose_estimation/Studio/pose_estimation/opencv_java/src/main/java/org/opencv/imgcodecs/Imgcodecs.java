
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.imgcodecs;

import java.lang.String;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.utils.Converters;

public class Imgcodecs {

    public static final int
            CV_LOAD_IMAGE_UNCHANGED = -1,
            CV_LOAD_IMAGE_GRAYSCALE = 0,
            CV_LOAD_IMAGE_COLOR = 1,
            CV_LOAD_IMAGE_ANYDEPTH = 2,
            CV_LOAD_IMAGE_ANYCOLOR = 4,
            CV_IMWRITE_JPEG_QUALITY = 1,
            CV_IMWRITE_JPEG_PROGRESSIVE = 2,
            CV_IMWRITE_JPEG_OPTIMIZE = 3,
            CV_IMWRITE_JPEG_RST_INTERVAL = 4,
            CV_IMWRITE_JPEG_LUMA_QUALITY = 5,
            CV_IMWRITE_JPEG_CHROMA_QUALITY = 6,
            CV_IMWRITE_PNG_COMPRESSION = 16,
            CV_IMWRITE_PNG_STRATEGY = 17,
            CV_IMWRITE_PNG_BILEVEL = 18,
            CV_IMWRITE_PNG_STRATEGY_DEFAULT = 0,
            CV_IMWRITE_PNG_STRATEGY_FILTERED = 1,
            CV_IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,
            CV_IMWRITE_PNG_STRATEGY_RLE = 3,
            CV_IMWRITE_PNG_STRATEGY_FIXED = 4,
            CV_IMWRITE_PXM_BINARY = 32,
            CV_IMWRITE_WEBP_QUALITY = 64,
            CV_CVTIMG_FLIP = 1,
            CV_CVTIMG_SWAP_RB = 2,
            IMREAD_UNCHANGED = -1,
            IMREAD_GRAYSCALE = 0,
            IMREAD_COLOR = 1,
            IMREAD_ANYDEPTH = 2,
            IMREAD_ANYCOLOR = 4,
            IMREAD_LOAD_GDAL = 8,
            IMREAD_REDUCED_GRAYSCALE_2 = 16,
            IMREAD_REDUCED_COLOR_2 = 17,
            IMREAD_REDUCED_GRAYSCALE_4 = 32,
            IMREAD_REDUCED_COLOR_4 = 33,
            IMREAD_REDUCED_GRAYSCALE_8 = 64,
            IMREAD_REDUCED_COLOR_8 = 65,
            IMWRITE_JPEG_QUALITY = 1,
            IMWRITE_JPEG_PROGRESSIVE = 2,
            IMWRITE_JPEG_OPTIMIZE = 3,
            IMWRITE_JPEG_RST_INTERVAL = 4,
            IMWRITE_JPEG_LUMA_QUALITY = 5,
            IMWRITE_JPEG_CHROMA_QUALITY = 6,
            IMWRITE_PNG_COMPRESSION = 16,
            IMWRITE_PNG_STRATEGY = 17,
            IMWRITE_PNG_BILEVEL = 18,
            IMWRITE_PXM_BINARY = 32,
            IMWRITE_WEBP_QUALITY = 64,
            IMWRITE_PNG_STRATEGY_DEFAULT = 0,
            IMWRITE_PNG_STRATEGY_FILTERED = 1,
            IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,
            IMWRITE_PNG_STRATEGY_RLE = 3,
            IMWRITE_PNG_STRATEGY_FIXED = 4;


    //
    // C++:  Mat imdecode(Mat buf, int flags)
    //

    //javadoc: imdecode(buf, flags)
    public static Mat imdecode(Mat buf, int flags)
    {
        
        Mat retVal = new Mat(imdecode_0(buf.nativeObj, flags));
        
        return retVal;
    }


    //
    // C++:  Mat imread(String filename, int flags = IMREAD_COLOR)
    //

    //javadoc: imread(filename, flags)
    public static Mat imread(String filename, int flags)
    {
        
        Mat retVal = new Mat(imread_0(filename, flags));
        
        return retVal;
    }

    //javadoc: imread(filename)
    public static Mat imread(String filename)
    {
        
        Mat retVal = new Mat(imread_1(filename));
        
        return retVal;
    }


    //
    // C++:  bool imencode(String ext, Mat img, vector_uchar& buf, vector_int params = std::vector<int>())
    //

    //javadoc: imencode(ext, img, buf, params)
    public static boolean imencode(String ext, Mat img, MatOfByte buf, MatOfInt params)
    {
        Mat buf_mat = buf;
        Mat params_mat = params;
        boolean retVal = imencode_0(ext, img.nativeObj, buf_mat.nativeObj, params_mat.nativeObj);
        
        return retVal;
    }

    //javadoc: imencode(ext, img, buf)
    public static boolean imencode(String ext, Mat img, MatOfByte buf)
    {
        Mat buf_mat = buf;
        boolean retVal = imencode_1(ext, img.nativeObj, buf_mat.nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool imreadmulti(String filename, vector_Mat mats, int flags = IMREAD_ANYCOLOR)
    //

    //javadoc: imreadmulti(filename, mats, flags)
    public static boolean imreadmulti(String filename, List<Mat> mats, int flags)
    {
        Mat mats_mat = Converters.vector_Mat_to_Mat(mats);
        boolean retVal = imreadmulti_0(filename, mats_mat.nativeObj, flags);
        
        return retVal;
    }

    //javadoc: imreadmulti(filename, mats)
    public static boolean imreadmulti(String filename, List<Mat> mats)
    {
        Mat mats_mat = Converters.vector_Mat_to_Mat(mats);
        boolean retVal = imreadmulti_1(filename, mats_mat.nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool imwrite(String filename, Mat img, vector_int params = std::vector<int>())
    //

    //javadoc: imwrite(filename, img, params)
    public static boolean imwrite(String filename, Mat img, MatOfInt params)
    {
        Mat params_mat = params;
        boolean retVal = imwrite_0(filename, img.nativeObj, params_mat.nativeObj);
        
        return retVal;
    }

    //javadoc: imwrite(filename, img)
    public static boolean imwrite(String filename, Mat img)
    {
        
        boolean retVal = imwrite_1(filename, img.nativeObj);
        
        return retVal;
    }




    // C++:  Mat imdecode(Mat buf, int flags)
    private static native long imdecode_0(long buf_nativeObj, int flags);

    // C++:  Mat imread(String filename, int flags = IMREAD_COLOR)
    private static native long imread_0(String filename, int flags);
    private static native long imread_1(String filename);

    // C++:  bool imencode(String ext, Mat img, vector_uchar& buf, vector_int params = std::vector<int>())
    private static native boolean imencode_0(String ext, long img_nativeObj, long buf_mat_nativeObj, long params_mat_nativeObj);
    private static native boolean imencode_1(String ext, long img_nativeObj, long buf_mat_nativeObj);

    // C++:  bool imreadmulti(String filename, vector_Mat mats, int flags = IMREAD_ANYCOLOR)
    private static native boolean imreadmulti_0(String filename, long mats_mat_nativeObj, int flags);
    private static native boolean imreadmulti_1(String filename, long mats_mat_nativeObj);

    // C++:  bool imwrite(String filename, Mat img, vector_int params = std::vector<int>())
    private static native boolean imwrite_0(String filename, long img_nativeObj, long params_mat_nativeObj);
    private static native boolean imwrite_1(String filename, long img_nativeObj);

}
