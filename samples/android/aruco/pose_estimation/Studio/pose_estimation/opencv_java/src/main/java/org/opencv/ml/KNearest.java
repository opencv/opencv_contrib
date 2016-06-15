
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;

// C++: class KNearest
//javadoc: KNearest
public class KNearest extends StatModel {

    protected KNearest(long addr) { super(addr); }


    public static final int
            BRUTE_FORCE = 1,
            KDTREE = 2;


    //
    // C++: static Ptr_KNearest create()
    //

    //javadoc: KNearest::create()
    public static KNearest create()
    {
        
        KNearest retVal = new KNearest(create_0());
        
        return retVal;
    }


    //
    // C++:  bool getIsClassifier()
    //

    //javadoc: KNearest::getIsClassifier()
    public  boolean getIsClassifier()
    {
        
        boolean retVal = getIsClassifier_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  float findNearest(Mat samples, int k, Mat& results, Mat& neighborResponses = Mat(), Mat& dist = Mat())
    //

    //javadoc: KNearest::findNearest(samples, k, results, neighborResponses, dist)
    public  float findNearest(Mat samples, int k, Mat results, Mat neighborResponses, Mat dist)
    {
        
        float retVal = findNearest_0(nativeObj, samples.nativeObj, k, results.nativeObj, neighborResponses.nativeObj, dist.nativeObj);
        
        return retVal;
    }

    //javadoc: KNearest::findNearest(samples, k, results)
    public  float findNearest(Mat samples, int k, Mat results)
    {
        
        float retVal = findNearest_1(nativeObj, samples.nativeObj, k, results.nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getAlgorithmType()
    //

    //javadoc: KNearest::getAlgorithmType()
    public  int getAlgorithmType()
    {
        
        int retVal = getAlgorithmType_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getDefaultK()
    //

    //javadoc: KNearest::getDefaultK()
    public  int getDefaultK()
    {
        
        int retVal = getDefaultK_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getEmax()
    //

    //javadoc: KNearest::getEmax()
    public  int getEmax()
    {
        
        int retVal = getEmax_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void setAlgorithmType(int val)
    //

    //javadoc: KNearest::setAlgorithmType(val)
    public  void setAlgorithmType(int val)
    {
        
        setAlgorithmType_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setDefaultK(int val)
    //

    //javadoc: KNearest::setDefaultK(val)
    public  void setDefaultK(int val)
    {
        
        setDefaultK_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setEmax(int val)
    //

    //javadoc: KNearest::setEmax(val)
    public  void setEmax(int val)
    {
        
        setEmax_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setIsClassifier(bool val)
    //

    //javadoc: KNearest::setIsClassifier(val)
    public  void setIsClassifier(boolean val)
    {
        
        setIsClassifier_0(nativeObj, val);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++: static Ptr_KNearest create()
    private static native long create_0();

    // C++:  bool getIsClassifier()
    private static native boolean getIsClassifier_0(long nativeObj);

    // C++:  float findNearest(Mat samples, int k, Mat& results, Mat& neighborResponses = Mat(), Mat& dist = Mat())
    private static native float findNearest_0(long nativeObj, long samples_nativeObj, int k, long results_nativeObj, long neighborResponses_nativeObj, long dist_nativeObj);
    private static native float findNearest_1(long nativeObj, long samples_nativeObj, int k, long results_nativeObj);

    // C++:  int getAlgorithmType()
    private static native int getAlgorithmType_0(long nativeObj);

    // C++:  int getDefaultK()
    private static native int getDefaultK_0(long nativeObj);

    // C++:  int getEmax()
    private static native int getEmax_0(long nativeObj);

    // C++:  void setAlgorithmType(int val)
    private static native void setAlgorithmType_0(long nativeObj, int val);

    // C++:  void setDefaultK(int val)
    private static native void setDefaultK_0(long nativeObj, int val);

    // C++:  void setEmax(int val)
    private static native void setEmax_0(long nativeObj, int val);

    // C++:  void setIsClassifier(bool val)
    private static native void setIsClassifier_0(long nativeObj, boolean val);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
