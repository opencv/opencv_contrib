
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;

// C++: class TrainData
//javadoc: TrainData
public class TrainData {

    protected final long nativeObj;
    protected TrainData(long addr) { nativeObj = addr; }


    //
    // C++:  Mat getCatMap()
    //

    //javadoc: TrainData::getCatMap()
    public  Mat getCatMap()
    {
        
        Mat retVal = new Mat(getCatMap_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getCatOfs()
    //

    //javadoc: TrainData::getCatOfs()
    public  Mat getCatOfs()
    {
        
        Mat retVal = new Mat(getCatOfs_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getClassLabels()
    //

    //javadoc: TrainData::getClassLabels()
    public  Mat getClassLabels()
    {
        
        Mat retVal = new Mat(getClassLabels_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getDefaultSubstValues()
    //

    //javadoc: TrainData::getDefaultSubstValues()
    public  Mat getDefaultSubstValues()
    {
        
        Mat retVal = new Mat(getDefaultSubstValues_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getMissing()
    //

    //javadoc: TrainData::getMissing()
    public  Mat getMissing()
    {
        
        Mat retVal = new Mat(getMissing_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getNormCatResponses()
    //

    //javadoc: TrainData::getNormCatResponses()
    public  Mat getNormCatResponses()
    {
        
        Mat retVal = new Mat(getNormCatResponses_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getResponses()
    //

    //javadoc: TrainData::getResponses()
    public  Mat getResponses()
    {
        
        Mat retVal = new Mat(getResponses_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getSampleWeights()
    //

    //javadoc: TrainData::getSampleWeights()
    public  Mat getSampleWeights()
    {
        
        Mat retVal = new Mat(getSampleWeights_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getSamples()
    //

    //javadoc: TrainData::getSamples()
    public  Mat getSamples()
    {
        
        Mat retVal = new Mat(getSamples_0(nativeObj));
        
        return retVal;
    }


    //
    // C++: static Mat getSubVector(Mat vec, Mat idx)
    //

    //javadoc: TrainData::getSubVector(vec, idx)
    public static Mat getSubVector(Mat vec, Mat idx)
    {
        
        Mat retVal = new Mat(getSubVector_0(vec.nativeObj, idx.nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getTestNormCatResponses()
    //

    //javadoc: TrainData::getTestNormCatResponses()
    public  Mat getTestNormCatResponses()
    {
        
        Mat retVal = new Mat(getTestNormCatResponses_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getTestResponses()
    //

    //javadoc: TrainData::getTestResponses()
    public  Mat getTestResponses()
    {
        
        Mat retVal = new Mat(getTestResponses_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getTestSampleIdx()
    //

    //javadoc: TrainData::getTestSampleIdx()
    public  Mat getTestSampleIdx()
    {
        
        Mat retVal = new Mat(getTestSampleIdx_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getTestSampleWeights()
    //

    //javadoc: TrainData::getTestSampleWeights()
    public  Mat getTestSampleWeights()
    {
        
        Mat retVal = new Mat(getTestSampleWeights_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getTrainNormCatResponses()
    //

    //javadoc: TrainData::getTrainNormCatResponses()
    public  Mat getTrainNormCatResponses()
    {
        
        Mat retVal = new Mat(getTrainNormCatResponses_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getTrainResponses()
    //

    //javadoc: TrainData::getTrainResponses()
    public  Mat getTrainResponses()
    {
        
        Mat retVal = new Mat(getTrainResponses_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getTrainSampleIdx()
    //

    //javadoc: TrainData::getTrainSampleIdx()
    public  Mat getTrainSampleIdx()
    {
        
        Mat retVal = new Mat(getTrainSampleIdx_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getTrainSampleWeights()
    //

    //javadoc: TrainData::getTrainSampleWeights()
    public  Mat getTrainSampleWeights()
    {
        
        Mat retVal = new Mat(getTrainSampleWeights_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getTrainSamples(int layout = ROW_SAMPLE, bool compressSamples = true, bool compressVars = true)
    //

    //javadoc: TrainData::getTrainSamples(layout, compressSamples, compressVars)
    public  Mat getTrainSamples(int layout, boolean compressSamples, boolean compressVars)
    {
        
        Mat retVal = new Mat(getTrainSamples_0(nativeObj, layout, compressSamples, compressVars));
        
        return retVal;
    }

    //javadoc: TrainData::getTrainSamples()
    public  Mat getTrainSamples()
    {
        
        Mat retVal = new Mat(getTrainSamples_1(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getVarIdx()
    //

    //javadoc: TrainData::getVarIdx()
    public  Mat getVarIdx()
    {
        
        Mat retVal = new Mat(getVarIdx_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getVarType()
    //

    //javadoc: TrainData::getVarType()
    public  Mat getVarType()
    {
        
        Mat retVal = new Mat(getVarType_0(nativeObj));
        
        return retVal;
    }


    //
    // C++: static Ptr_TrainData create(Mat samples, int layout, Mat responses, Mat varIdx = Mat(), Mat sampleIdx = Mat(), Mat sampleWeights = Mat(), Mat varType = Mat())
    //

    // Return type 'Ptr_TrainData' is not supported, skipping the function


    //
    // C++:  int getCatCount(int vi)
    //

    //javadoc: TrainData::getCatCount(vi)
    public  int getCatCount(int vi)
    {
        
        int retVal = getCatCount_0(nativeObj, vi);
        
        return retVal;
    }


    //
    // C++:  int getLayout()
    //

    //javadoc: TrainData::getLayout()
    public  int getLayout()
    {
        
        int retVal = getLayout_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getNAllVars()
    //

    //javadoc: TrainData::getNAllVars()
    public  int getNAllVars()
    {
        
        int retVal = getNAllVars_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getNSamples()
    //

    //javadoc: TrainData::getNSamples()
    public  int getNSamples()
    {
        
        int retVal = getNSamples_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getNTestSamples()
    //

    //javadoc: TrainData::getNTestSamples()
    public  int getNTestSamples()
    {
        
        int retVal = getNTestSamples_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getNTrainSamples()
    //

    //javadoc: TrainData::getNTrainSamples()
    public  int getNTrainSamples()
    {
        
        int retVal = getNTrainSamples_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getNVars()
    //

    //javadoc: TrainData::getNVars()
    public  int getNVars()
    {
        
        int retVal = getNVars_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getResponseType()
    //

    //javadoc: TrainData::getResponseType()
    public  int getResponseType()
    {
        
        int retVal = getResponseType_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void getSample(Mat varIdx, int sidx, float* buf)
    //

    //javadoc: TrainData::getSample(varIdx, sidx, buf)
    public  void getSample(Mat varIdx, int sidx, float buf)
    {
        
        getSample_0(nativeObj, varIdx.nativeObj, sidx, buf);
        
        return;
    }


    //
    // C++:  void getValues(int vi, Mat sidx, float* values)
    //

    //javadoc: TrainData::getValues(vi, sidx, values)
    public  void getValues(int vi, Mat sidx, float values)
    {
        
        getValues_0(nativeObj, vi, sidx.nativeObj, values);
        
        return;
    }


    //
    // C++:  void setTrainTestSplit(int count, bool shuffle = true)
    //

    //javadoc: TrainData::setTrainTestSplit(count, shuffle)
    public  void setTrainTestSplit(int count, boolean shuffle)
    {
        
        setTrainTestSplit_0(nativeObj, count, shuffle);
        
        return;
    }

    //javadoc: TrainData::setTrainTestSplit(count)
    public  void setTrainTestSplit(int count)
    {
        
        setTrainTestSplit_1(nativeObj, count);
        
        return;
    }


    //
    // C++:  void setTrainTestSplitRatio(double ratio, bool shuffle = true)
    //

    //javadoc: TrainData::setTrainTestSplitRatio(ratio, shuffle)
    public  void setTrainTestSplitRatio(double ratio, boolean shuffle)
    {
        
        setTrainTestSplitRatio_0(nativeObj, ratio, shuffle);
        
        return;
    }

    //javadoc: TrainData::setTrainTestSplitRatio(ratio)
    public  void setTrainTestSplitRatio(double ratio)
    {
        
        setTrainTestSplitRatio_1(nativeObj, ratio);
        
        return;
    }


    //
    // C++:  void shuffleTrainTest()
    //

    //javadoc: TrainData::shuffleTrainTest()
    public  void shuffleTrainTest()
    {
        
        shuffleTrainTest_0(nativeObj);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat getCatMap()
    private static native long getCatMap_0(long nativeObj);

    // C++:  Mat getCatOfs()
    private static native long getCatOfs_0(long nativeObj);

    // C++:  Mat getClassLabels()
    private static native long getClassLabels_0(long nativeObj);

    // C++:  Mat getDefaultSubstValues()
    private static native long getDefaultSubstValues_0(long nativeObj);

    // C++:  Mat getMissing()
    private static native long getMissing_0(long nativeObj);

    // C++:  Mat getNormCatResponses()
    private static native long getNormCatResponses_0(long nativeObj);

    // C++:  Mat getResponses()
    private static native long getResponses_0(long nativeObj);

    // C++:  Mat getSampleWeights()
    private static native long getSampleWeights_0(long nativeObj);

    // C++:  Mat getSamples()
    private static native long getSamples_0(long nativeObj);

    // C++: static Mat getSubVector(Mat vec, Mat idx)
    private static native long getSubVector_0(long vec_nativeObj, long idx_nativeObj);

    // C++:  Mat getTestNormCatResponses()
    private static native long getTestNormCatResponses_0(long nativeObj);

    // C++:  Mat getTestResponses()
    private static native long getTestResponses_0(long nativeObj);

    // C++:  Mat getTestSampleIdx()
    private static native long getTestSampleIdx_0(long nativeObj);

    // C++:  Mat getTestSampleWeights()
    private static native long getTestSampleWeights_0(long nativeObj);

    // C++:  Mat getTrainNormCatResponses()
    private static native long getTrainNormCatResponses_0(long nativeObj);

    // C++:  Mat getTrainResponses()
    private static native long getTrainResponses_0(long nativeObj);

    // C++:  Mat getTrainSampleIdx()
    private static native long getTrainSampleIdx_0(long nativeObj);

    // C++:  Mat getTrainSampleWeights()
    private static native long getTrainSampleWeights_0(long nativeObj);

    // C++:  Mat getTrainSamples(int layout = ROW_SAMPLE, bool compressSamples = true, bool compressVars = true)
    private static native long getTrainSamples_0(long nativeObj, int layout, boolean compressSamples, boolean compressVars);
    private static native long getTrainSamples_1(long nativeObj);

    // C++:  Mat getVarIdx()
    private static native long getVarIdx_0(long nativeObj);

    // C++:  Mat getVarType()
    private static native long getVarType_0(long nativeObj);

    // C++:  int getCatCount(int vi)
    private static native int getCatCount_0(long nativeObj, int vi);

    // C++:  int getLayout()
    private static native int getLayout_0(long nativeObj);

    // C++:  int getNAllVars()
    private static native int getNAllVars_0(long nativeObj);

    // C++:  int getNSamples()
    private static native int getNSamples_0(long nativeObj);

    // C++:  int getNTestSamples()
    private static native int getNTestSamples_0(long nativeObj);

    // C++:  int getNTrainSamples()
    private static native int getNTrainSamples_0(long nativeObj);

    // C++:  int getNVars()
    private static native int getNVars_0(long nativeObj);

    // C++:  int getResponseType()
    private static native int getResponseType_0(long nativeObj);

    // C++:  void getSample(Mat varIdx, int sidx, float* buf)
    private static native void getSample_0(long nativeObj, long varIdx_nativeObj, int sidx, float buf);

    // C++:  void getValues(int vi, Mat sidx, float* values)
    private static native void getValues_0(long nativeObj, int vi, long sidx_nativeObj, float values);

    // C++:  void setTrainTestSplit(int count, bool shuffle = true)
    private static native void setTrainTestSplit_0(long nativeObj, int count, boolean shuffle);
    private static native void setTrainTestSplit_1(long nativeObj, int count);

    // C++:  void setTrainTestSplitRatio(double ratio, bool shuffle = true)
    private static native void setTrainTestSplitRatio_0(long nativeObj, double ratio, boolean shuffle);
    private static native void setTrainTestSplitRatio_1(long nativeObj, double ratio);

    // C++:  void shuffleTrainTest()
    private static native void shuffleTrainTest_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
