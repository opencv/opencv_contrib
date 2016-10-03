
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.ml;

import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;

// C++: class SVM
//javadoc: SVM
public class SVM extends StatModel {

    protected SVM(long addr) { super(addr); }


    public static final int
            C_SVC = 100,
            NU_SVC = 101,
            ONE_CLASS = 102,
            EPS_SVR = 103,
            NU_SVR = 104,
            CUSTOM = -1,
            LINEAR = 0,
            POLY = 1,
            RBF = 2,
            SIGMOID = 3,
            CHI2 = 4,
            INTER = 5,
            C = 0,
            GAMMA = 1,
            P = 2,
            NU = 3,
            COEF = 4,
            DEGREE = 5;


    //
    // C++:  Mat getClassWeights()
    //

    //javadoc: SVM::getClassWeights()
    public  Mat getClassWeights()
    {
        
        Mat retVal = new Mat(getClassWeights_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getSupportVectors()
    //

    //javadoc: SVM::getSupportVectors()
    public  Mat getSupportVectors()
    {
        
        Mat retVal = new Mat(getSupportVectors_0(nativeObj));
        
        return retVal;
    }


    //
    // C++: static Ptr_SVM create()
    //

    //javadoc: SVM::create()
    public static SVM create()
    {
        
        SVM retVal = new SVM(create_0());
        
        return retVal;
    }


    //
    // C++:  TermCriteria getTermCriteria()
    //

    //javadoc: SVM::getTermCriteria()
    public  TermCriteria getTermCriteria()
    {
        
        TermCriteria retVal = new TermCriteria(getTermCriteria_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  double getC()
    //

    //javadoc: SVM::getC()
    public  double getC()
    {
        
        double retVal = getC_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getCoef0()
    //

    //javadoc: SVM::getCoef0()
    public  double getCoef0()
    {
        
        double retVal = getCoef0_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getDecisionFunction(int i, Mat& alpha, Mat& svidx)
    //

    //javadoc: SVM::getDecisionFunction(i, alpha, svidx)
    public  double getDecisionFunction(int i, Mat alpha, Mat svidx)
    {
        
        double retVal = getDecisionFunction_0(nativeObj, i, alpha.nativeObj, svidx.nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getDegree()
    //

    //javadoc: SVM::getDegree()
    public  double getDegree()
    {
        
        double retVal = getDegree_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getGamma()
    //

    //javadoc: SVM::getGamma()
    public  double getGamma()
    {
        
        double retVal = getGamma_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getNu()
    //

    //javadoc: SVM::getNu()
    public  double getNu()
    {
        
        double retVal = getNu_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  double getP()
    //

    //javadoc: SVM::getP()
    public  double getP()
    {
        
        double retVal = getP_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getKernelType()
    //

    //javadoc: SVM::getKernelType()
    public  int getKernelType()
    {
        
        int retVal = getKernelType_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  int getType()
    //

    //javadoc: SVM::getType()
    public  int getType()
    {
        
        int retVal = getType_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void setC(double val)
    //

    //javadoc: SVM::setC(val)
    public  void setC(double val)
    {
        
        setC_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setClassWeights(Mat val)
    //

    //javadoc: SVM::setClassWeights(val)
    public  void setClassWeights(Mat val)
    {
        
        setClassWeights_0(nativeObj, val.nativeObj);
        
        return;
    }


    //
    // C++:  void setCoef0(double val)
    //

    //javadoc: SVM::setCoef0(val)
    public  void setCoef0(double val)
    {
        
        setCoef0_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setDegree(double val)
    //

    //javadoc: SVM::setDegree(val)
    public  void setDegree(double val)
    {
        
        setDegree_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setGamma(double val)
    //

    //javadoc: SVM::setGamma(val)
    public  void setGamma(double val)
    {
        
        setGamma_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setKernel(int kernelType)
    //

    //javadoc: SVM::setKernel(kernelType)
    public  void setKernel(int kernelType)
    {
        
        setKernel_0(nativeObj, kernelType);
        
        return;
    }


    //
    // C++:  void setNu(double val)
    //

    //javadoc: SVM::setNu(val)
    public  void setNu(double val)
    {
        
        setNu_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setP(double val)
    //

    //javadoc: SVM::setP(val)
    public  void setP(double val)
    {
        
        setP_0(nativeObj, val);
        
        return;
    }


    //
    // C++:  void setTermCriteria(TermCriteria val)
    //

    //javadoc: SVM::setTermCriteria(val)
    public  void setTermCriteria(TermCriteria val)
    {
        
        setTermCriteria_0(nativeObj, val.type, val.maxCount, val.epsilon);
        
        return;
    }


    //
    // C++:  void setType(int val)
    //

    //javadoc: SVM::setType(val)
    public  void setType(int val)
    {
        
        setType_0(nativeObj, val);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat getClassWeights()
    private static native long getClassWeights_0(long nativeObj);

    // C++:  Mat getSupportVectors()
    private static native long getSupportVectors_0(long nativeObj);

    // C++: static Ptr_SVM create()
    private static native long create_0();

    // C++:  TermCriteria getTermCriteria()
    private static native double[] getTermCriteria_0(long nativeObj);

    // C++:  double getC()
    private static native double getC_0(long nativeObj);

    // C++:  double getCoef0()
    private static native double getCoef0_0(long nativeObj);

    // C++:  double getDecisionFunction(int i, Mat& alpha, Mat& svidx)
    private static native double getDecisionFunction_0(long nativeObj, int i, long alpha_nativeObj, long svidx_nativeObj);

    // C++:  double getDegree()
    private static native double getDegree_0(long nativeObj);

    // C++:  double getGamma()
    private static native double getGamma_0(long nativeObj);

    // C++:  double getNu()
    private static native double getNu_0(long nativeObj);

    // C++:  double getP()
    private static native double getP_0(long nativeObj);

    // C++:  int getKernelType()
    private static native int getKernelType_0(long nativeObj);

    // C++:  int getType()
    private static native int getType_0(long nativeObj);

    // C++:  void setC(double val)
    private static native void setC_0(long nativeObj, double val);

    // C++:  void setClassWeights(Mat val)
    private static native void setClassWeights_0(long nativeObj, long val_nativeObj);

    // C++:  void setCoef0(double val)
    private static native void setCoef0_0(long nativeObj, double val);

    // C++:  void setDegree(double val)
    private static native void setDegree_0(long nativeObj, double val);

    // C++:  void setGamma(double val)
    private static native void setGamma_0(long nativeObj, double val);

    // C++:  void setKernel(int kernelType)
    private static native void setKernel_0(long nativeObj, int kernelType);

    // C++:  void setNu(double val)
    private static native void setNu_0(long nativeObj, double val);

    // C++:  void setP(double val)
    private static native void setP_0(long nativeObj, double val);

    // C++:  void setTermCriteria(TermCriteria val)
    private static native void setTermCriteria_0(long nativeObj, int val_type, int val_maxCount, double val_epsilon);

    // C++:  void setType(int val)
    private static native void setType_0(long nativeObj, int val);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
