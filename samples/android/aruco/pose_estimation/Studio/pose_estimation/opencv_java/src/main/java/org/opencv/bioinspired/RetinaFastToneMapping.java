
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.bioinspired;

import org.opencv.core.Algorithm;
import org.opencv.core.Mat;

// C++: class RetinaFastToneMapping
//javadoc: RetinaFastToneMapping
public class RetinaFastToneMapping extends Algorithm {

    protected RetinaFastToneMapping(long addr) { super(addr); }


    //
    // C++:  void applyFastToneMapping(Mat inputImage, Mat& outputToneMappedImage)
    //

    //javadoc: RetinaFastToneMapping::applyFastToneMapping(inputImage, outputToneMappedImage)
    public  void applyFastToneMapping(Mat inputImage, Mat outputToneMappedImage)
    {
        
        applyFastToneMapping_0(nativeObj, inputImage.nativeObj, outputToneMappedImage.nativeObj);
        
        return;
    }


    //
    // C++:  void setup(float photoreceptorsNeighborhoodRadius = 3.f, float ganglioncellsNeighborhoodRadius = 1.f, float meanLuminanceModulatorK = 1.f)
    //

    //javadoc: RetinaFastToneMapping::setup(photoreceptorsNeighborhoodRadius, ganglioncellsNeighborhoodRadius, meanLuminanceModulatorK)
    public  void setup(float photoreceptorsNeighborhoodRadius, float ganglioncellsNeighborhoodRadius, float meanLuminanceModulatorK)
    {
        
        setup_0(nativeObj, photoreceptorsNeighborhoodRadius, ganglioncellsNeighborhoodRadius, meanLuminanceModulatorK);
        
        return;
    }

    //javadoc: RetinaFastToneMapping::setup()
    public  void setup()
    {
        
        setup_1(nativeObj);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  void applyFastToneMapping(Mat inputImage, Mat& outputToneMappedImage)
    private static native void applyFastToneMapping_0(long nativeObj, long inputImage_nativeObj, long outputToneMappedImage_nativeObj);

    // C++:  void setup(float photoreceptorsNeighborhoodRadius = 3.f, float ganglioncellsNeighborhoodRadius = 1.f, float meanLuminanceModulatorK = 1.f)
    private static native void setup_0(long nativeObj, float photoreceptorsNeighborhoodRadius, float ganglioncellsNeighborhoodRadius, float meanLuminanceModulatorK);
    private static native void setup_1(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
