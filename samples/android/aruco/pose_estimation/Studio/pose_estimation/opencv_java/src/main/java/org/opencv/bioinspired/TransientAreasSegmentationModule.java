
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.bioinspired;

import java.lang.String;
import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.core.Size;

// C++: class TransientAreasSegmentationModule
//javadoc: TransientAreasSegmentationModule
public class TransientAreasSegmentationModule extends Algorithm {

    protected TransientAreasSegmentationModule(long addr) { super(addr); }


    //
    // C++:  Size getSize()
    //

    //javadoc: TransientAreasSegmentationModule::getSize()
    public  Size getSize()
    {
        
        Size retVal = new Size(getSize_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  String printSetup()
    //

    //javadoc: TransientAreasSegmentationModule::printSetup()
    public  String printSetup()
    {
        
        String retVal = printSetup_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void clearAllBuffers()
    //

    //javadoc: TransientAreasSegmentationModule::clearAllBuffers()
    public  void clearAllBuffers()
    {
        
        clearAllBuffers_0(nativeObj);
        
        return;
    }


    //
    // C++:  void getSegmentationPicture(Mat& transientAreas)
    //

    //javadoc: TransientAreasSegmentationModule::getSegmentationPicture(transientAreas)
    public  void getSegmentationPicture(Mat transientAreas)
    {
        
        getSegmentationPicture_0(nativeObj, transientAreas.nativeObj);
        
        return;
    }


    //
    // C++:  void run(Mat inputToSegment, int channelIndex = 0)
    //

    //javadoc: TransientAreasSegmentationModule::run(inputToSegment, channelIndex)
    public  void run(Mat inputToSegment, int channelIndex)
    {
        
        run_0(nativeObj, inputToSegment.nativeObj, channelIndex);
        
        return;
    }

    //javadoc: TransientAreasSegmentationModule::run(inputToSegment)
    public  void run(Mat inputToSegment)
    {
        
        run_1(nativeObj, inputToSegment.nativeObj);
        
        return;
    }


    //
    // C++:  void setup(String segmentationParameterFile = "", bool applyDefaultSetupOnFailure = true)
    //

    //javadoc: TransientAreasSegmentationModule::setup(segmentationParameterFile, applyDefaultSetupOnFailure)
    public  void setup(String segmentationParameterFile, boolean applyDefaultSetupOnFailure)
    {
        
        setup_0(nativeObj, segmentationParameterFile, applyDefaultSetupOnFailure);
        
        return;
    }

    //javadoc: TransientAreasSegmentationModule::setup()
    public  void setup()
    {
        
        setup_1(nativeObj);
        
        return;
    }


    //
    // C++:  void write(String fs)
    //

    //javadoc: TransientAreasSegmentationModule::write(fs)
    public  void write(String fs)
    {
        
        write_0(nativeObj, fs);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Size getSize()
    private static native double[] getSize_0(long nativeObj);

    // C++:  String printSetup()
    private static native String printSetup_0(long nativeObj);

    // C++:  void clearAllBuffers()
    private static native void clearAllBuffers_0(long nativeObj);

    // C++:  void getSegmentationPicture(Mat& transientAreas)
    private static native void getSegmentationPicture_0(long nativeObj, long transientAreas_nativeObj);

    // C++:  void run(Mat inputToSegment, int channelIndex = 0)
    private static native void run_0(long nativeObj, long inputToSegment_nativeObj, int channelIndex);
    private static native void run_1(long nativeObj, long inputToSegment_nativeObj);

    // C++:  void setup(String segmentationParameterFile = "", bool applyDefaultSetupOnFailure = true)
    private static native void setup_0(long nativeObj, String segmentationParameterFile, boolean applyDefaultSetupOnFailure);
    private static native void setup_1(long nativeObj);

    // C++:  void write(String fs)
    private static native void write_0(long nativeObj, String fs);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
