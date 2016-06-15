
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.video;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;

// C++: class DenseOpticalFlow
//javadoc: DenseOpticalFlow
public class DenseOpticalFlow extends Algorithm {

    protected DenseOpticalFlow(long addr) { super(addr); }


    //
    // C++:  void calc(Mat I0, Mat I1, Mat& flow)
    //

    //javadoc: DenseOpticalFlow::calc(I0, I1, flow)
    public  void calc(Mat I0, Mat I1, Mat flow)
    {
        
        calc_0(nativeObj, I0.nativeObj, I1.nativeObj, flow.nativeObj);
        
        return;
    }


    //
    // C++:  void collectGarbage()
    //

    //javadoc: DenseOpticalFlow::collectGarbage()
    public  void collectGarbage()
    {
        
        collectGarbage_0(nativeObj);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  void calc(Mat I0, Mat I1, Mat& flow)
    private static native void calc_0(long nativeObj, long I0_nativeObj, long I1_nativeObj, long flow_nativeObj);

    // C++:  void collectGarbage()
    private static native void collectGarbage_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
