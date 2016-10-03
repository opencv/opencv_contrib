
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.bioinspired;

import java.lang.String;
import org.opencv.core.Algorithm;
import org.opencv.core.Mat;
import org.opencv.core.Size;

// C++: class Retina
//javadoc: Retina
public class Retina extends Algorithm {

    protected Retina(long addr) { super(addr); }


    //
    // C++:  Mat getMagnoRAW()
    //

    //javadoc: Retina::getMagnoRAW()
    public  Mat getMagnoRAW()
    {
        
        Mat retVal = new Mat(getMagnoRAW_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Mat getParvoRAW()
    //

    //javadoc: Retina::getParvoRAW()
    public  Mat getParvoRAW()
    {
        
        Mat retVal = new Mat(getParvoRAW_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Size getInputSize()
    //

    //javadoc: Retina::getInputSize()
    public  Size getInputSize()
    {
        
        Size retVal = new Size(getInputSize_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  Size getOutputSize()
    //

    //javadoc: Retina::getOutputSize()
    public  Size getOutputSize()
    {
        
        Size retVal = new Size(getOutputSize_0(nativeObj));
        
        return retVal;
    }


    //
    // C++:  String printSetup()
    //

    //javadoc: Retina::printSetup()
    public  String printSetup()
    {
        
        String retVal = printSetup_0(nativeObj);
        
        return retVal;
    }


    //
    // C++:  void activateContoursProcessing(bool activate)
    //

    //javadoc: Retina::activateContoursProcessing(activate)
    public  void activateContoursProcessing(boolean activate)
    {
        
        activateContoursProcessing_0(nativeObj, activate);
        
        return;
    }


    //
    // C++:  void activateMovingContoursProcessing(bool activate)
    //

    //javadoc: Retina::activateMovingContoursProcessing(activate)
    public  void activateMovingContoursProcessing(boolean activate)
    {
        
        activateMovingContoursProcessing_0(nativeObj, activate);
        
        return;
    }


    //
    // C++:  void applyFastToneMapping(Mat inputImage, Mat& outputToneMappedImage)
    //

    //javadoc: Retina::applyFastToneMapping(inputImage, outputToneMappedImage)
    public  void applyFastToneMapping(Mat inputImage, Mat outputToneMappedImage)
    {
        
        applyFastToneMapping_0(nativeObj, inputImage.nativeObj, outputToneMappedImage.nativeObj);
        
        return;
    }


    //
    // C++:  void clearBuffers()
    //

    //javadoc: Retina::clearBuffers()
    public  void clearBuffers()
    {
        
        clearBuffers_0(nativeObj);
        
        return;
    }


    //
    // C++:  void getMagno(Mat& retinaOutput_magno)
    //

    //javadoc: Retina::getMagno(retinaOutput_magno)
    public  void getMagno(Mat retinaOutput_magno)
    {
        
        getMagno_0(nativeObj, retinaOutput_magno.nativeObj);
        
        return;
    }


    //
    // C++:  void getMagnoRAW(Mat& retinaOutput_magno)
    //

    //javadoc: Retina::getMagnoRAW(retinaOutput_magno)
    public  void getMagnoRAW(Mat retinaOutput_magno)
    {
        
        getMagnoRAW_1(nativeObj, retinaOutput_magno.nativeObj);
        
        return;
    }


    //
    // C++:  void getParvo(Mat& retinaOutput_parvo)
    //

    //javadoc: Retina::getParvo(retinaOutput_parvo)
    public  void getParvo(Mat retinaOutput_parvo)
    {
        
        getParvo_0(nativeObj, retinaOutput_parvo.nativeObj);
        
        return;
    }


    //
    // C++:  void getParvoRAW(Mat& retinaOutput_parvo)
    //

    //javadoc: Retina::getParvoRAW(retinaOutput_parvo)
    public  void getParvoRAW(Mat retinaOutput_parvo)
    {
        
        getParvoRAW_1(nativeObj, retinaOutput_parvo.nativeObj);
        
        return;
    }


    //
    // C++:  void run(Mat inputImage)
    //

    //javadoc: Retina::run(inputImage)
    public  void run(Mat inputImage)
    {
        
        run_0(nativeObj, inputImage.nativeObj);
        
        return;
    }


    //
    // C++:  void setColorSaturation(bool saturateColors = true, float colorSaturationValue = 4.0f)
    //

    //javadoc: Retina::setColorSaturation(saturateColors, colorSaturationValue)
    public  void setColorSaturation(boolean saturateColors, float colorSaturationValue)
    {
        
        setColorSaturation_0(nativeObj, saturateColors, colorSaturationValue);
        
        return;
    }

    //javadoc: Retina::setColorSaturation()
    public  void setColorSaturation()
    {
        
        setColorSaturation_1(nativeObj);
        
        return;
    }


    //
    // C++:  void setup(String retinaParameterFile = "", bool applyDefaultSetupOnFailure = true)
    //

    //javadoc: Retina::setup(retinaParameterFile, applyDefaultSetupOnFailure)
    public  void setup(String retinaParameterFile, boolean applyDefaultSetupOnFailure)
    {
        
        setup_0(nativeObj, retinaParameterFile, applyDefaultSetupOnFailure);
        
        return;
    }

    //javadoc: Retina::setup()
    public  void setup()
    {
        
        setup_1(nativeObj);
        
        return;
    }


    //
    // C++:  void setupIPLMagnoChannel(bool normaliseOutput = true, float parasolCells_beta = 0.f, float parasolCells_tau = 0.f, float parasolCells_k = 7.f, float amacrinCellsTemporalCutFrequency = 1.2f, float V0CompressionParameter = 0.95f, float localAdaptintegration_tau = 0.f, float localAdaptintegration_k = 7.f)
    //

    //javadoc: Retina::setupIPLMagnoChannel(normaliseOutput, parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency, V0CompressionParameter, localAdaptintegration_tau, localAdaptintegration_k)
    public  void setupIPLMagnoChannel(boolean normaliseOutput, float parasolCells_beta, float parasolCells_tau, float parasolCells_k, float amacrinCellsTemporalCutFrequency, float V0CompressionParameter, float localAdaptintegration_tau, float localAdaptintegration_k)
    {
        
        setupIPLMagnoChannel_0(nativeObj, normaliseOutput, parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency, V0CompressionParameter, localAdaptintegration_tau, localAdaptintegration_k);
        
        return;
    }

    //javadoc: Retina::setupIPLMagnoChannel()
    public  void setupIPLMagnoChannel()
    {
        
        setupIPLMagnoChannel_1(nativeObj);
        
        return;
    }


    //
    // C++:  void setupOPLandIPLParvoChannel(bool colorMode = true, bool normaliseOutput = true, float photoreceptorsLocalAdaptationSensitivity = 0.7f, float photoreceptorsTemporalConstant = 0.5f, float photoreceptorsSpatialConstant = 0.53f, float horizontalCellsGain = 0.f, float HcellsTemporalConstant = 1.f, float HcellsSpatialConstant = 7.f, float ganglionCellsSensitivity = 0.7f)
    //

    //javadoc: Retina::setupOPLandIPLParvoChannel(colorMode, normaliseOutput, photoreceptorsLocalAdaptationSensitivity, photoreceptorsTemporalConstant, photoreceptorsSpatialConstant, horizontalCellsGain, HcellsTemporalConstant, HcellsSpatialConstant, ganglionCellsSensitivity)
    public  void setupOPLandIPLParvoChannel(boolean colorMode, boolean normaliseOutput, float photoreceptorsLocalAdaptationSensitivity, float photoreceptorsTemporalConstant, float photoreceptorsSpatialConstant, float horizontalCellsGain, float HcellsTemporalConstant, float HcellsSpatialConstant, float ganglionCellsSensitivity)
    {
        
        setupOPLandIPLParvoChannel_0(nativeObj, colorMode, normaliseOutput, photoreceptorsLocalAdaptationSensitivity, photoreceptorsTemporalConstant, photoreceptorsSpatialConstant, horizontalCellsGain, HcellsTemporalConstant, HcellsSpatialConstant, ganglionCellsSensitivity);
        
        return;
    }

    //javadoc: Retina::setupOPLandIPLParvoChannel()
    public  void setupOPLandIPLParvoChannel()
    {
        
        setupOPLandIPLParvoChannel_1(nativeObj);
        
        return;
    }


    //
    // C++:  void write(String fs)
    //

    //javadoc: Retina::write(fs)
    public  void write(String fs)
    {
        
        write_0(nativeObj, fs);
        
        return;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:  Mat getMagnoRAW()
    private static native long getMagnoRAW_0(long nativeObj);

    // C++:  Mat getParvoRAW()
    private static native long getParvoRAW_0(long nativeObj);

    // C++:  Size getInputSize()
    private static native double[] getInputSize_0(long nativeObj);

    // C++:  Size getOutputSize()
    private static native double[] getOutputSize_0(long nativeObj);

    // C++:  String printSetup()
    private static native String printSetup_0(long nativeObj);

    // C++:  void activateContoursProcessing(bool activate)
    private static native void activateContoursProcessing_0(long nativeObj, boolean activate);

    // C++:  void activateMovingContoursProcessing(bool activate)
    private static native void activateMovingContoursProcessing_0(long nativeObj, boolean activate);

    // C++:  void applyFastToneMapping(Mat inputImage, Mat& outputToneMappedImage)
    private static native void applyFastToneMapping_0(long nativeObj, long inputImage_nativeObj, long outputToneMappedImage_nativeObj);

    // C++:  void clearBuffers()
    private static native void clearBuffers_0(long nativeObj);

    // C++:  void getMagno(Mat& retinaOutput_magno)
    private static native void getMagno_0(long nativeObj, long retinaOutput_magno_nativeObj);

    // C++:  void getMagnoRAW(Mat& retinaOutput_magno)
    private static native void getMagnoRAW_1(long nativeObj, long retinaOutput_magno_nativeObj);

    // C++:  void getParvo(Mat& retinaOutput_parvo)
    private static native void getParvo_0(long nativeObj, long retinaOutput_parvo_nativeObj);

    // C++:  void getParvoRAW(Mat& retinaOutput_parvo)
    private static native void getParvoRAW_1(long nativeObj, long retinaOutput_parvo_nativeObj);

    // C++:  void run(Mat inputImage)
    private static native void run_0(long nativeObj, long inputImage_nativeObj);

    // C++:  void setColorSaturation(bool saturateColors = true, float colorSaturationValue = 4.0f)
    private static native void setColorSaturation_0(long nativeObj, boolean saturateColors, float colorSaturationValue);
    private static native void setColorSaturation_1(long nativeObj);

    // C++:  void setup(String retinaParameterFile = "", bool applyDefaultSetupOnFailure = true)
    private static native void setup_0(long nativeObj, String retinaParameterFile, boolean applyDefaultSetupOnFailure);
    private static native void setup_1(long nativeObj);

    // C++:  void setupIPLMagnoChannel(bool normaliseOutput = true, float parasolCells_beta = 0.f, float parasolCells_tau = 0.f, float parasolCells_k = 7.f, float amacrinCellsTemporalCutFrequency = 1.2f, float V0CompressionParameter = 0.95f, float localAdaptintegration_tau = 0.f, float localAdaptintegration_k = 7.f)
    private static native void setupIPLMagnoChannel_0(long nativeObj, boolean normaliseOutput, float parasolCells_beta, float parasolCells_tau, float parasolCells_k, float amacrinCellsTemporalCutFrequency, float V0CompressionParameter, float localAdaptintegration_tau, float localAdaptintegration_k);
    private static native void setupIPLMagnoChannel_1(long nativeObj);

    // C++:  void setupOPLandIPLParvoChannel(bool colorMode = true, bool normaliseOutput = true, float photoreceptorsLocalAdaptationSensitivity = 0.7f, float photoreceptorsTemporalConstant = 0.5f, float photoreceptorsSpatialConstant = 0.53f, float horizontalCellsGain = 0.f, float HcellsTemporalConstant = 1.f, float HcellsSpatialConstant = 7.f, float ganglionCellsSensitivity = 0.7f)
    private static native void setupOPLandIPLParvoChannel_0(long nativeObj, boolean colorMode, boolean normaliseOutput, float photoreceptorsLocalAdaptationSensitivity, float photoreceptorsTemporalConstant, float photoreceptorsSpatialConstant, float horizontalCellsGain, float HcellsTemporalConstant, float HcellsSpatialConstant, float ganglionCellsSensitivity);
    private static native void setupOPLandIPLParvoChannel_1(long nativeObj);

    // C++:  void write(String fs)
    private static native void write_0(long nativeObj, String fs);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
