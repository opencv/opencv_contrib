
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.objdetect;

import java.lang.String;
import java.util.ArrayList;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Size;

// C++: class CascadeClassifier
//javadoc: CascadeClassifier
public class CascadeClassifier {

    protected final long nativeObj;
    protected CascadeClassifier(long addr) { nativeObj = addr; }


    //
    // C++:   CascadeClassifier(String filename)
    //

    //javadoc: CascadeClassifier::CascadeClassifier(filename)
    public   CascadeClassifier(String filename)
    {
        
        nativeObj = CascadeClassifier_0(filename);
        
        return;
    }


    //
    // C++:   CascadeClassifier()
    //

    //javadoc: CascadeClassifier::CascadeClassifier()
    public   CascadeClassifier()
    {
        
        nativeObj = CascadeClassifier_1();
        
        return;
    }


    //
    // C++:  Size getOriginalWindowSize()
    //

    //javadoc: CascadeClassifier::getOriginalWindowSize()
    public  Size getOriginalWindowSize()
    {
        
        Size retVal = new Size(getOriginalWindowSize_0(nativeObj));
        
        return retVal;
    }


    //
    // C++: static bool convert(String oldcascade, String newcascade)
    //

    //javadoc: CascadeClassifier::convert(oldcascade, newcascade)
    public static boolean convert(String oldcascade, String newcascade)
    {
        
        boolean retVal = convert_0(oldcascade, newcascade);
        
        return retVal;
    }


    //
    // C++:  bool empty()
    //

    //javadoc: CascadeClassifier::empty()
    public  boolean empty()
    {
        
        boolean retVal = empty_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool isOldFormatCascade()
    //

    //javadoc: CascadeClassifier::isOldFormatCascade()
    public  boolean isOldFormatCascade()
    {
        
        boolean retVal = isOldFormatCascade_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool load(String filename)
    //

    //javadoc: CascadeClassifier::load(filename)
    public  boolean load(String filename)
    {
        
        boolean retVal = load_0(nativeObj, filename);
        
        return retVal;
    }


    //
    // C++:  bool read(FileNode node)
    //

    // Unknown type 'FileNode' (I), skipping the function


    //
    // C++:  int getFeatureType()
    //

    //javadoc: CascadeClassifier::getFeatureType()
    public  int getFeatureType()
    {
        
        int retVal = getFeatureType_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void detectMultiScale(Mat image, vector_Rect& objects, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size())
    //

    //javadoc: CascadeClassifier::detectMultiScale(image, objects, scaleFactor, minNeighbors, flags, minSize, maxSize)
    public  void detectMultiScale(Mat image, MatOfRect objects, double scaleFactor, int minNeighbors, int flags, Size minSize, Size maxSize)
    {
        Mat objects_mat = objects;
        detectMultiScale_0(nativeObj, image.nativeObj, objects_mat.nativeObj, scaleFactor, minNeighbors, flags, minSize.width, minSize.height, maxSize.width, maxSize.height);
        
        return;
    }

    //javadoc: CascadeClassifier::detectMultiScale(image, objects)
    public  void detectMultiScale(Mat image, MatOfRect objects)
    {
        Mat objects_mat = objects;
        detectMultiScale_1(nativeObj, image.nativeObj, objects_mat.nativeObj);
        
        return;
    }


    //
    // C++:  void detectMultiScale(Mat image, vector_Rect& objects, vector_int& numDetections, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size())
    //

    //javadoc: CascadeClassifier::detectMultiScale(image, objects, numDetections, scaleFactor, minNeighbors, flags, minSize, maxSize)
    public  void detectMultiScale2(Mat image, MatOfRect objects, MatOfInt numDetections, double scaleFactor, int minNeighbors, int flags, Size minSize, Size maxSize)
    {
        Mat objects_mat = objects;
        Mat numDetections_mat = numDetections;
        detectMultiScale2_0(nativeObj, image.nativeObj, objects_mat.nativeObj, numDetections_mat.nativeObj, scaleFactor, minNeighbors, flags, minSize.width, minSize.height, maxSize.width, maxSize.height);
        
        return;
    }

    //javadoc: CascadeClassifier::detectMultiScale(image, objects, numDetections)
    public  void detectMultiScale2(Mat image, MatOfRect objects, MatOfInt numDetections)
    {
        Mat objects_mat = objects;
        Mat numDetections_mat = numDetections;
        detectMultiScale2_1(nativeObj, image.nativeObj, objects_mat.nativeObj, numDetections_mat.nativeObj);
        
        return;
    }


    //
    // C++:  void detectMultiScale(Mat image, vector_Rect& objects, vector_int& rejectLevels, vector_double& levelWeights, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size(), bool outputRejectLevels = false)
    //

    //javadoc: CascadeClassifier::detectMultiScale(image, objects, rejectLevels, levelWeights, scaleFactor, minNeighbors, flags, minSize, maxSize, outputRejectLevels)
    public  void detectMultiScale3(Mat image, MatOfRect objects, MatOfInt rejectLevels, MatOfDouble levelWeights, double scaleFactor, int minNeighbors, int flags, Size minSize, Size maxSize, boolean outputRejectLevels)
    {
        Mat objects_mat = objects;
        Mat rejectLevels_mat = rejectLevels;
        Mat levelWeights_mat = levelWeights;
        detectMultiScale3_0(nativeObj, image.nativeObj, objects_mat.nativeObj, rejectLevels_mat.nativeObj, levelWeights_mat.nativeObj, scaleFactor, minNeighbors, flags, minSize.width, minSize.height, maxSize.width, maxSize.height, outputRejectLevels);
        
        return;
    }

    //javadoc: CascadeClassifier::detectMultiScale(image, objects, rejectLevels, levelWeights)
    public  void detectMultiScale3(Mat image, MatOfRect objects, MatOfInt rejectLevels, MatOfDouble levelWeights)
    {
        Mat objects_mat = objects;
        Mat rejectLevels_mat = rejectLevels;
        Mat levelWeights_mat = levelWeights;
        detectMultiScale3_1(nativeObj, image.nativeObj, objects_mat.nativeObj, rejectLevels_mat.nativeObj, levelWeights_mat.nativeObj);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   CascadeClassifier(String filename)
    private static native long CascadeClassifier_0(String filename);

    // C++:   CascadeClassifier()
    private static native long CascadeClassifier_1();

    // C++:  Size getOriginalWindowSize()
    private static native double[] getOriginalWindowSize_0(long nativeObj);

    // C++: static bool convert(String oldcascade, String newcascade)
    private static native boolean convert_0(String oldcascade, String newcascade);

    // C++:  bool empty()
    private static native boolean empty_0(long nativeObj);

    // C++:  bool isOldFormatCascade()
    private static native boolean isOldFormatCascade_0(long nativeObj);

    // C++:  bool load(String filename)
    private static native boolean load_0(long nativeObj, String filename);

    // C++:  int getFeatureType()
    private static native int getFeatureType_0(long nativeObj);

    // C++:  void detectMultiScale(Mat image, vector_Rect& objects, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size())
    private static native void detectMultiScale_0(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, double scaleFactor, int minNeighbors, int flags, double minSize_width, double minSize_height, double maxSize_width, double maxSize_height);
    private static native void detectMultiScale_1(long nativeObj, long image_nativeObj, long objects_mat_nativeObj);

    // C++:  void detectMultiScale(Mat image, vector_Rect& objects, vector_int& numDetections, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size())
    private static native void detectMultiScale2_0(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long numDetections_mat_nativeObj, double scaleFactor, int minNeighbors, int flags, double minSize_width, double minSize_height, double maxSize_width, double maxSize_height);
    private static native void detectMultiScale2_1(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long numDetections_mat_nativeObj);

    // C++:  void detectMultiScale(Mat image, vector_Rect& objects, vector_int& rejectLevels, vector_double& levelWeights, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size(), bool outputRejectLevels = false)
    private static native void detectMultiScale3_0(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long rejectLevels_mat_nativeObj, long levelWeights_mat_nativeObj, double scaleFactor, int minNeighbors, int flags, double minSize_width, double minSize_height, double maxSize_width, double maxSize_height, boolean outputRejectLevels);
    private static native void detectMultiScale3_1(long nativeObj, long image_nativeObj, long objects_mat_nativeObj, long rejectLevels_mat_nativeObj, long levelWeights_mat_nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
