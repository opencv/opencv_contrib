
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;

// C++: class StatModel
//javadoc: StatModel
public class StatModel extends Algorithm {

    protected StatModel(long addr) { super(addr); }


    public static final int
            UPDATE_MODEL = 1,
            RAW_OUTPUT = 1,
            COMPRESSED_INPUT = 2,
            PREPROCESSED_INPUT = 4;


    //
    // C++:  bool empty()
    //

    //javadoc: StatModel::empty()
    public  boolean empty()
    {
        
        boolean retVal = empty_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool isClassifier()
    //

    //javadoc: StatModel::isClassifier()
    public  boolean isClassifier()
    {
        
        boolean retVal = isClassifier_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool isTrained()
    //

    //javadoc: StatModel::isTrained()
    public  boolean isTrained()
    {
        
        boolean retVal = isTrained_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool train(Mat samples, int layout, Mat responses)
    //

    //javadoc: StatModel::train(samples, layout, responses)
    public  boolean train(Mat samples, int layout, Mat responses)
    {
        
        boolean retVal = train_0(nativeObj, samples.nativeObj, layout, responses.nativeObj);
        
        return retVal;
    }


    //
    // C++:  bool train(Ptr_TrainData trainData, int flags = 0)
    //

    // Unknown type 'Ptr_TrainData' (I), skipping the function


    //
    // C++:  float calcError(Ptr_TrainData data, bool test, Mat& resp)
    //

    // Unknown type 'Ptr_TrainData' (I), skipping the function


    //
    // C++:  float predict(Mat samples, Mat& results = Mat(), int flags = 0)
    //

    //javadoc: StatModel::predict(samples, results, flags)
    public  float predict(Mat samples, Mat results, int flags)
    {
        
        float retVal = predict_0(nativeObj, samples.nativeObj, results.nativeObj, flags);
        
        return retVal;
    }

    //javadoc: StatModel::predict(samples)
    public  float predict(Mat samples)
    {
        
        float retVal = predict_1(nativeObj, samples.nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getVarCount()
    //

    //javadoc: StatModel::getVarCount()
    public  int getVarCount()
    {
        
        int retVal = getVarCount_0(nativeObj);
        
        return retVal;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  bool empty()
    private static native boolean empty_0(long nativeObj);

    // C++:  bool isClassifier()
    private static native boolean isClassifier_0(long nativeObj);

    // C++:  bool isTrained()
    private static native boolean isTrained_0(long nativeObj);

    // C++:  bool train(Mat samples, int layout, Mat responses)
    private static native boolean train_0(long nativeObj, long samples_nativeObj, int layout, long responses_nativeObj);

    // C++:  float predict(Mat samples, Mat& results = Mat(), int flags = 0)
    private static native float predict_0(long nativeObj, long samples_nativeObj, long results_nativeObj, int flags);
    private static native float predict_1(long nativeObj, long samples_nativeObj);

    // C++:  int getVarCount()
    private static native int getVarCount_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
