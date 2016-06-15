
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.bioinspired;

import org.opencv.core.Size;

public class Bioinspired {

    public static final int
            RETINA_COLOR_RANDOM = 0,
            RETINA_COLOR_DIAGONAL = 1,
            RETINA_COLOR_BAYER = 2;


    //
    // C++:  Ptr_Retina createRetina(Size inputSize, bool colorMode, int colorSamplingMethod = RETINA_COLOR_BAYER, bool useRetinaLogSampling = false, float reductionFactor = 1.0f, float samplingStrenght = 10.0f)
    //

    //javadoc: createRetina(inputSize, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght)
    public static Retina createRetina(Size inputSize, boolean colorMode, int colorSamplingMethod, boolean useRetinaLogSampling, float reductionFactor, float samplingStrenght)
    {
        
        Retina retVal = new Retina(createRetina_0(inputSize.width, inputSize.height, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght));
        
        return retVal;
    }

    //javadoc: createRetina(inputSize, colorMode)
    public static Retina createRetina(Size inputSize, boolean colorMode)
    {
        
        Retina retVal = new Retina(createRetina_1(inputSize.width, inputSize.height, colorMode));
        
        return retVal;
    }


    //
    // C++:  Ptr_Retina createRetina(Size inputSize)
    //

    //javadoc: createRetina(inputSize)
    public static Retina createRetina(Size inputSize)
    {
        
        Retina retVal = new Retina(createRetina_2(inputSize.width, inputSize.height));
        
        return retVal;
    }


    //
    // C++:  Ptr_RetinaFastToneMapping createRetinaFastToneMapping(Size inputSize)
    //

    //javadoc: createRetinaFastToneMapping(inputSize)
    public static RetinaFastToneMapping createRetinaFastToneMapping(Size inputSize)
    {
        
        RetinaFastToneMapping retVal = new RetinaFastToneMapping(createRetinaFastToneMapping_0(inputSize.width, inputSize.height));
        
        return retVal;
    }


    //
    // C++:  Ptr_TransientAreasSegmentationModule createTransientAreasSegmentationModule(Size inputSize)
    //

    //javadoc: createTransientAreasSegmentationModule(inputSize)
    public static TransientAreasSegmentationModule createTransientAreasSegmentationModule(Size inputSize)
    {
        
        TransientAreasSegmentationModule retVal = new TransientAreasSegmentationModule(createTransientAreasSegmentationModule_0(inputSize.width, inputSize.height));
        
        return retVal;
    }




    // C++:  Ptr_Retina createRetina(Size inputSize, bool colorMode, int colorSamplingMethod = RETINA_COLOR_BAYER, bool useRetinaLogSampling = false, float reductionFactor = 1.0f, float samplingStrenght = 10.0f)
    private static native long createRetina_0(double inputSize_width, double inputSize_height, boolean colorMode, int colorSamplingMethod, boolean useRetinaLogSampling, float reductionFactor, float samplingStrenght);
    private static native long createRetina_1(double inputSize_width, double inputSize_height, boolean colorMode);

    // C++:  Ptr_Retina createRetina(Size inputSize)
    private static native long createRetina_2(double inputSize_width, double inputSize_height);

    // C++:  Ptr_RetinaFastToneMapping createRetinaFastToneMapping(Size inputSize)
    private static native long createRetinaFastToneMapping_0(double inputSize_width, double inputSize_height);

    // C++:  Ptr_TransientAreasSegmentationModule createTransientAreasSegmentationModule(Size inputSize)
    private static native long createTransientAreasSegmentationModule_0(double inputSize_width, double inputSize_height);

}
