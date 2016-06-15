
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;

// C++: class ANN_MLP
//javadoc: ANN_MLP
public class ANN_MLP extends StatModel {

    protected ANN_MLP(long addr) { super(addr); }


    public static final int
            BACKPROP = 0,
            RPROP = 1,
            IDENTITY = 0,
            SIGMOID_SYM = 1,
            GAUSSIAN = 2,
            UPDATE_WEIGHTS = 1,
            NO_INPUT_SCALE = 2,
            NO_OUTPUT_SCALE = 4;


    //
    // C++:  Mat getLayerSizes()
    //

    //javadoc: ANN_MLP::getLayerSizes()
    public  Mat getLayerSizes()
    {
        
        Mat retVal = new Mat(getLayerSizes_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getWeights(int layerIdx)
    //

    //javadoc: ANN_MLP::getWeights(layerIdx)
    public  Mat getWeights(int layerIdx)
    {
        
        Mat retVal = new Mat(getWeights_0(nativeObj, layerIdx));
        
        return retVal;
    }


    //
    // C++: static Ptr_ANN_MLP create()
    //

    //javadoc: ANN_MLP::create()
    public static ANN_MLP create()
    {
        
        ANN_MLP retVal = new ANN_MLP(create_0());
        
        return retVal;
    }


    //
    // C++:  TermCriteria getTermCriteria()
    //

    //javadoc: ANN_MLP::getTermCriteria()
    public  TermCriteria getTermCriteria()
    {
        
        TermCriteria retVal = new TermCriteria(getTermCriteria_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  double getBackpropMomentumScale()
    //

    //javadoc: ANN_MLP::getBackpropMomentumScale()
    public  double getBackpropMomentumScale()
    {
        
        double retVal = getBackpropMomentumScale_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getBackpropWeightScale()
    //

    //javadoc: ANN_MLP::getBackpropWeightScale()
    public  double getBackpropWeightScale()
    {
        
        double retVal = getBackpropWeightScale_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getRpropDW0()
    //

    //javadoc: ANN_MLP::getRpropDW0()
    public  double getRpropDW0()
    {
        
        double retVal = getRpropDW0_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getRpropDWMax()
    //

    //javadoc: ANN_MLP::getRpropDWMax()
    public  double getRpropDWMax()
    {
        
        double retVal = getRpropDWMax_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getRpropDWMin()
    //

    //javadoc: ANN_MLP::getRpropDWMin()
    public  double getRpropDWMin()
    {
        
        double retVal = getRpropDWMin_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getRpropDWMinus()
    //

    //javadoc: ANN_MLP::getRpropDWMinus()
    public  double getRpropDWMinus()
    {
        
        double retVal = getRpropDWMinus_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getRpropDWPlus()
    //

    //javadoc: ANN_MLP::getRpropDWPlus()
    public  double getRpropDWPlus()
    {
        
        double retVal = getRpropDWPlus_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getTrainMethod()
    //

    //javadoc: ANN_MLP::getTrainMethod()
    public  int getTrainMethod()
    {
        
        int retVal = getTrainMethod_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void setActivationFunction(int type, double param1 = 0, double param2 = 0)
    //

    //javadoc: ANN_MLP::setActivationFunction(type, param1, param2)
    public  void setActivationFunction(int type, double param1, double param2)
    {
        
        setActivationFunction_0(nativeObj, type, param1, param2);
        
        return;
    }

    //javadoc: ANN_MLP::setActivationFunction(type)
    public  void setActivationFunction(int type)
    {
        
        setActivationFunction_1(nativeObj, type);
        
        return;
    }


    //
    // C++:  void setBackpropMomentumScale(double val)
    //

    //javadoc: ANN_MLP::setBackpropMomentumScale(val)
    public  void setBackpropMomentumScale(double val)
    {
        
        setBackpropMomentumScale_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setBackpropWeightScale(double val)
    //

    //javadoc: ANN_MLP::setBackpropWeightScale(val)
    public  void setBackpropWeightScale(double val)
    {
        
        setBackpropWeightScale_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setLayerSizes(Mat _layer_sizes)
    //

    //javadoc: ANN_MLP::setLayerSizes(_layer_sizes)
    public  void setLayerSizes(Mat _layer_sizes)
    {
        
        setLayerSizes_0(nativeObj, _layer_sizes.nativeObj);
        
        return;
    }


    //
    // C++:  void setRpropDW0(double val)
    //

    //javadoc: ANN_MLP::setRpropDW0(val)
    public  void setRpropDW0(double val)
    {
        
        setRpropDW0_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setRpropDWMax(double val)
    //

    //javadoc: ANN_MLP::setRpropDWMax(val)
    public  void setRpropDWMax(double val)
    {
        
        setRpropDWMax_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setRpropDWMin(double val)
    //

    //javadoc: ANN_MLP::setRpropDWMin(val)
    public  void setRpropDWMin(double val)
    {
        
        setRpropDWMin_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setRpropDWMinus(double val)
    //

    //javadoc: ANN_MLP::setRpropDWMinus(val)
    public  void setRpropDWMinus(double val)
    {
        
        setRpropDWMinus_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setRpropDWPlus(double val)
    //

    //javadoc: ANN_MLP::setRpropDWPlus(val)
    public  void setRpropDWPlus(double val)
    {
        
        setRpropDWPlus_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setTermCriteria(TermCriteria val)
    //

    //javadoc: ANN_MLP::setTermCriteria(val)
    public  void setTermCriteria(TermCriteria val)
    {
        
        setTermCriteria_0(nativeObj, val.type, val.maxCount, val.epsilon);
        
        return;
    }


    //
    // C++:  void setTrainMethod(int method, double param1 = 0, double param2 = 0)
    //

    //javadoc: ANN_MLP::setTrainMethod(method, param1, param2)
    public  void setTrainMethod(int method, double param1, double param2)
    {
        
        setTrainMethod_0(nativeObj, method, param1, param2);
        
        return;
    }

    //javadoc: ANN_MLP::setTrainMethod(method)
    public  void setTrainMethod(int method)
    {
        
        setTrainMethod_1(nativeObj, method);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat getLayerSizes()
    private static native long getLayerSizes_0(long nativeObj);

    // C++:  Mat getWeights(int layerIdx)
    private static native long getWeights_0(long nativeObj, int layerIdx);

    // C++: static Ptr_ANN_MLP create()
    private static native long create_0();

    // C++:  TermCriteria getTermCriteria()
    private static native double[] getTermCriteria_0(long nativeObj);

    // C++:  double getBackpropMomentumScale()
    private static native double getBackpropMomentumScale_0(long nativeObj);

    // C++:  double getBackpropWeightScale()
    private static native double getBackpropWeightScale_0(long nativeObj);

    // C++:  double getRpropDW0()
    private static native double getRpropDW0_0(long nativeObj);

    // C++:  double getRpropDWMax()
    private static native double getRpropDWMax_0(long nativeObj);

    // C++:  double getRpropDWMin()
    private static native double getRpropDWMin_0(long nativeObj);

    // C++:  double getRpropDWMinus()
    private static native double getRpropDWMinus_0(long nativeObj);

    // C++:  double getRpropDWPlus()
    private static native double getRpropDWPlus_0(long nativeObj);

    // C++:  int getTrainMethod()
    private static native int getTrainMethod_0(long nativeObj);

    // C++:  void setActivationFunction(int type, double param1 = 0, double param2 = 0)
    private static native void setActivationFunction_0(long nativeObj, int type, double param1, double param2);
    private static native void setActivationFunction_1(long nativeObj, int type);

    // C++:  void setBackpropMomentumScale(double val)
    private static native void setBackpropMomentumScale_0(long nativeObj, double val);

    // C++:  void setBackpropWeightScale(double val)
    private static native void setBackpropWeightScale_0(long nativeObj, double val);

    // C++:  void setLayerSizes(Mat _layer_sizes)
    private static native void setLayerSizes_0(long nativeObj, long _layer_sizes_nativeObj);

    // C++:  void setRpropDW0(double val)
    private static native void setRpropDW0_0(long nativeObj, double val);

    // C++:  void setRpropDWMax(double val)
    private static native void setRpropDWMax_0(long nativeObj, double val);

    // C++:  void setRpropDWMin(double val)
    private static native void setRpropDWMin_0(long nativeObj, double val);

    // C++:  void setRpropDWMinus(double val)
    private static native void setRpropDWMinus_0(long nativeObj, double val);

    // C++:  void setRpropDWPlus(double val)
    private static native void setRpropDWPlus_0(long nativeObj, double val);

    // C++:  void setTermCriteria(TermCriteria val)
    private static native void setTermCriteria_0(long nativeObj, int val_type, int val_maxCount, double val_epsilon);

    // C++:  void setTrainMethod(int method, double param1 = 0, double param2 = 0)
    private static native void setTrainMethod_0(long nativeObj, int method, double param1, double param2);
    private static native void setTrainMethod_1(long nativeObj, int method);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
