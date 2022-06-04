/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2013, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OCL_RETINA_HPP__
#define __OCL_RETINA_HPP__

#include "precomp.hpp"
#include "opencv2/bioinspired/retina.hpp"

#ifdef HAVE_OPENCL

// please refer to c++ headers for API comments
namespace cv
{
namespace bioinspired
{
namespace ocl
{
void normalizeGrayOutputCentredSigmoide(const float meanValue, const float sensitivity, UMat &in, UMat &out, const float maxValue = 255.f);
void normalizeGrayOutput_0_maxOutputValue(UMat &inputOutputBuffer, const float maxOutputValue = 255.0);
void normalizeGrayOutputNearZeroCentreredSigmoide(UMat &inputPicture, UMat &outputBuffer, const float sensitivity = 40, const float maxOutputValue = 255.0f);
void centerReductImageLuminance(UMat &inputOutputBuffer);

class BasicRetinaFilter
{
public:
    BasicRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns, const unsigned int parametersListSize = 1, const bool useProgressiveFilter = false);
    ~BasicRetinaFilter();
    inline void clearOutputBuffer()
    {
        _filterOutput = 0;
    }
    inline void clearSecondaryBuffer()
    {
        _localBuffer = 0;
    }
    inline void clearAllBuffers()
    {
        clearOutputBuffer();
        clearSecondaryBuffer();
    }
    void  resize(const unsigned int NBrows, const unsigned int NBcolumns);
    const UMat &runFilter_LPfilter(const UMat &inputFrame, const unsigned int filterIndex = 0);
    void  runFilter_LPfilter(const UMat &inputFrame, UMat &outputFrame, const unsigned int filterIndex = 0);
    void  runFilter_LPfilter_Autonomous(UMat &inputOutputFrame, const unsigned int filterIndex = 0);
    const UMat &runFilter_LocalAdapdation(const UMat &inputOutputFrame, const UMat &localLuminance);
    void  runFilter_LocalAdapdation(const UMat &inputFrame, const UMat &localLuminance, UMat &outputFrame);
    const UMat &runFilter_LocalAdapdation_autonomous(const UMat &inputFrame);
    void  runFilter_LocalAdapdation_autonomous(const UMat &inputFrame, UMat &outputFrame);
    void  setLPfilterParameters(const float beta, const float tau, const float k, const unsigned int filterIndex = 0);
    inline void setV0CompressionParameter(const float v0, const float maxInputValue, const float)
    {
        _v0 = v0 * maxInputValue;
        _localLuminanceFactor = v0;
        _localLuminanceAddon = maxInputValue * (1.0f - v0);
        _maxInputValue = maxInputValue;
    }
    inline void setV0CompressionParameter(const float v0, const float meanLuminance)
    {
        this->setV0CompressionParameter(v0, _maxInputValue, meanLuminance);
    }
    inline void setV0CompressionParameter(const float v0)
    {
        _v0 = v0 * _maxInputValue;
        _localLuminanceFactor = v0;
        _localLuminanceAddon = _maxInputValue * (1.0f - v0);
    }
    inline void setV0CompressionParameterToneMapping(const float v0, const float maxInputValue, const float meanLuminance = 128.0f)
    {
        _v0 = v0 * maxInputValue;
        _localLuminanceFactor = 1.0f;
        _localLuminanceAddon = meanLuminance * _v0;
        _maxInputValue = maxInputValue;
    }
    inline void updateCompressionParameter(const float meanLuminance)
    {
        _localLuminanceFactor = 1;
        _localLuminanceAddon = meanLuminance * _v0;
    }
    inline float getV0CompressionParameter()
    {
        return _v0 / _maxInputValue;
    }
    inline const UMat &getOutput() const
    {
        return _filterOutput;
    }
    inline unsigned int getNBrows()
    {
        return _filterOutput.rows;
    }
    inline unsigned int getNBcolumns()
    {
        return _filterOutput.cols;
    }
    inline unsigned int getNBpixels()
    {
        return _filterOutput.size().area();
    }
    inline void normalizeGrayOutput_0_maxOutputValue(const float maxValue)
    {
        ocl::normalizeGrayOutput_0_maxOutputValue(_filterOutput, maxValue);
    }
    inline void normalizeGrayOutputCentredSigmoide()
    {
        ocl::normalizeGrayOutputCentredSigmoide(0.0, 2.0, _filterOutput, _filterOutput);
    }
    inline void centerReductImageLuminance()
    {
        ocl::centerReductImageLuminance(_filterOutput);
    }
    inline float getMaxInputValue()
    {
        return this->_maxInputValue;
    }
    inline void setMaxInputValue(const float newMaxInputValue)
    {
        this->_maxInputValue = newMaxInputValue;
    }

protected:

    int _NBrows;
    int _NBcols;
    unsigned int _halfNBrows;
    unsigned int _halfNBcolumns;

    UMat _filterOutput;
    UMat _localBuffer;

    std::valarray <float>_filteringCoeficientsTable;
    float _v0;
    float _maxInputValue;
    float _meanInputValue;
    float _localLuminanceFactor;
    float _localLuminanceAddon;

    float _a;
    float _tau;
    float _gain;

    void _spatiotemporalLPfilter(const UMat &inputFrame, UMat &LPfilterOutput, const unsigned int coefTableOffset = 0);
    void _spatiotemporalLPfilter_h(const UMat &inputFrame, UMat &LPfilterOutput, const unsigned int coefTableOffset = 0);
    void _spatiotemporalLPfilter_v(UMat &LPfilterOutput, const unsigned int multichannel = 0);
    float _squaringSpatiotemporalLPfilter(const UMat &inputFrame, UMat &outputFrame, const unsigned int filterIndex = 0);
    void _spatiotemporalLPfilter_Irregular(const UMat &inputFrame, UMat &outputFrame, const unsigned int filterIndex = 0);
    void _localSquaringSpatioTemporalLPfilter(const UMat &inputFrame, UMat &LPfilterOutput, const unsigned int *integrationAreas, const unsigned int filterIndex = 0);
    void _localLuminanceAdaptation(const UMat &inputFrame, const UMat &localLuminance, UMat &outputFrame, const bool updateLuminanceMean = true);
    void _localLuminanceAdaptation(UMat &inputOutputFrame, const UMat &localLuminance);
    void _localLuminanceAdaptationPosNegValues(const UMat &inputFrame, const UMat &localLuminance, float *outputFrame);
    void _horizontalCausalFilter_addInput(const UMat &inputFrame, UMat &outputFrame);
    void _verticalCausalFilter(UMat &outputFrame);
    void _verticalCausalFilter_multichannel(UMat &outputFrame);
    void _verticalCausalFilter_Irregular(UMat &outputFrame, const UMat &spatialConstantBuffer);
};

class MagnoRetinaFilter: public BasicRetinaFilter
{
public:
    MagnoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns);
    virtual ~MagnoRetinaFilter();
    void clearAllBuffers();
    void resize(const unsigned int NBrows, const unsigned int NBcolumns);
    void setCoefficientsTable(const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float localAdaptIntegration_tau, const float localAdaptIntegration_k);

    const UMat &runFilter(const UMat &OPL_ON, const UMat &OPL_OFF);

    inline const UMat &getMagnoON() const
    {
        return _magnoXOutputON;
    }
    inline const UMat &getMagnoOFF() const
    {
        return _magnoXOutputOFF;
    }
    inline const UMat &getMagnoYsaturated() const
    {
        return _magnoYsaturated;
    }
    inline void normalizeGrayOutputNearZeroCentreredSigmoide()
    {
        ocl::normalizeGrayOutputNearZeroCentreredSigmoide(_magnoYOutput, _magnoYsaturated);
    }
    inline float getTemporalConstant()
    {
        return this->_filteringCoeficientsTable[2];
    }
private:
    UMat _previousInput_ON;
    UMat _previousInput_OFF;
    UMat _amacrinCellsTempOutput_ON;
    UMat _amacrinCellsTempOutput_OFF;
    UMat _magnoXOutputON;
    UMat _magnoXOutputOFF;
    UMat _localProcessBufferON;
    UMat _localProcessBufferOFF;
    UMat _magnoYOutput;
    UMat _magnoYsaturated;

    float _temporalCoefficient;
    void _amacrineCellsComputing(const UMat &OPL_ON,  const UMat &OPL_OFF);
};

class ParvoRetinaFilter: public BasicRetinaFilter
{
public:
    ParvoRetinaFilter(const unsigned int NBrows = 480, const unsigned int NBcolumns = 640);
    virtual ~ParvoRetinaFilter();
    void resize(const unsigned int NBrows, const unsigned int NBcolumns);
    void clearAllBuffers();
    void setOPLandParvoFiltersParameters(const float beta1, const float tau1, const float k1, const float beta2, const float tau2, const float k2);

    inline void setGanglionCellsLocalAdaptationLPfilterParameters(const float tau, const float k)
    {
        BasicRetinaFilter::setLPfilterParameters(0, tau, k, 2);
    }
    const UMat &runFilter(const UMat &inputFrame, const bool useParvoOutput = true);

    inline const UMat &getPhotoreceptorsLPfilteringOutput() const
    {
        return _photoreceptorsOutput;
    }

    inline const UMat &getHorizontalCellsOutput() const
    {
        return _horizontalCellsOutput;
    }

    inline const UMat &getParvoON() const
    {
        return _parvocellularOutputON;
    }

    inline const UMat &getParvoOFF() const
    {
        return _parvocellularOutputOFF;
    }

    inline const UMat &getBipolarCellsON() const
    {
        return _bipolarCellsOutputON;
    }

    inline const UMat &getBipolarCellsOFF() const
    {
        return _bipolarCellsOutputOFF;
    }

    inline float getPhotoreceptorsTemporalConstant()
    {
        return this->_filteringCoeficientsTable[2];
    }

    inline float getHcellsTemporalConstant()
    {
        return this->_filteringCoeficientsTable[5];
    }
private:
    UMat _photoreceptorsOutput;
    UMat _horizontalCellsOutput;
    UMat _parvocellularOutputON;
    UMat _parvocellularOutputOFF;
    UMat _bipolarCellsOutputON;
    UMat _bipolarCellsOutputOFF;
    UMat _localAdaptationOFF;
    UMat _localAdaptationON;
    UMat _parvocellularOutputONminusOFF;
    void _OPL_OnOffWaysComputing();
};
class RetinaColor: public BasicRetinaFilter
{
public:
    RetinaColor(const unsigned int NBrows, const unsigned int NBcolumns, const int samplingMethod = RETINA_COLOR_DIAGONAL);
    virtual ~RetinaColor();

    void clearAllBuffers();
    void resize(const unsigned int NBrows, const unsigned int NBcolumns);
    inline void runColorMultiplexing(const UMat &inputRGBFrame)
    {
        runColorMultiplexing(inputRGBFrame, _multiplexedFrame);
    }
    void runColorMultiplexing(const UMat &demultiplexedInputFrame, UMat &multiplexedFrame);
    void runColorDemultiplexing(const UMat &multiplexedColorFrame, const bool adaptiveFiltering = false, const float maxInputValue = 255.0);

    void setColorSaturation(const bool saturateColors = true, const float colorSaturationValue = 4.0)
    {
        _saturateColors = saturateColors;
        _colorSaturationValue = colorSaturationValue;
    }

    void setChrominanceLPfilterParameters(const float beta, const float tau, const float k)
    {
        setLPfilterParameters(beta, tau, k);
    }

    bool applyKrauskopfLMS2Acr1cr2Transform(UMat &result);
    bool applyLMS2LabTransform(UMat &result);
    inline const UMat &getMultiplexedFrame() const
    {
        return _multiplexedFrame;
    }

    inline const UMat &getDemultiplexedColorFrame() const
    {
        return _demultiplexedColorFrame;
    }

    inline const UMat &getLuminance() const
    {
        return _luminance;
    }
    inline const UMat &getChrominance() const
    {
        return _chrominance;
    }
    void clipRGBOutput_0_maxInputValue(UMat &inputOutputBuffer, const float maxOutputValue = 255.0);
    void normalizeRGBOutput_0_maxOutputValue(const float maxOutputValue = 255.0);
    inline void setDemultiplexedColorFrame(const UMat &demultiplexedImage)
    {
        _demultiplexedColorFrame = demultiplexedImage;
    }
protected:
    inline unsigned int bayerSampleOffset(unsigned int index)
    {
        return index + ((index / getNBcolumns()) % 2) * getNBpixels() + ((index % getNBcolumns()) % 2) * getNBpixels();
    }
    inline Rect getROI(int idx)
    {
        return Rect(0, idx * _NBrows, _NBcols, _NBrows);
    }
    int _samplingMethod;
    bool _saturateColors;
    float _colorSaturationValue;
    UMat _luminance;
    UMat _multiplexedFrame;
    UMat _RGBmosaic;
    UMat _tempMultiplexedFrame;
    UMat _demultiplexedTempBuffer;
    UMat _demultiplexedColorFrame;
    UMat _chrominance;
    UMat _colorLocalDensity;
    UMat _imageGradient;

    float _pR, _pG, _pB;
    bool _objectInit;

    void _initColorSampling();
    void _adaptiveSpatialLPfilter_h(const UMat &inputFrame, const UMat &gradient, UMat &outputFrame);
    void _adaptiveSpatialLPfilter_v(const UMat &gradient, UMat &outputFrame);
    void _adaptiveHorizontalCausalFilter_addInput(const UMat &inputFrame, const UMat &gradient, UMat &outputFrame);
    void _computeGradient(const UMat &luminance, UMat &gradient);
    void _normalizeOutputs_0_maxOutputValue(void);
    void _applyImageColorSpaceConversion(const UMat &inputFrame, UMat &outputFrame, const float *transformTable);
};
class RetinaFilter
{
public:
    RetinaFilter(const unsigned int sizeRows, const unsigned int sizeColumns, const bool colorMode = false, const int samplingMethod = RETINA_COLOR_BAYER, const bool useRetinaLogSampling = false, const double reductionFactor = 1.0, const double samplingStrength = 10.0);
    ~RetinaFilter();

    void clearAllBuffers();
    void resize(const unsigned int NBrows, const unsigned int NBcolumns);
    bool checkInput(const UMat &input, const bool colorMode);
    bool runFilter(const UMat &imageInput, const bool useAdaptiveFiltering = true, const bool processRetinaParvoMagnoMapping = false, const bool useColorMode = false, const bool inputIsColorMultiplexed = false);

    void setGlobalParameters(const float OPLspatialResponse1 = 0.7, const float OPLtemporalresponse1 = 1, const float OPLassymetryGain = 0, const float OPLspatialResponse2 = 5, const float OPLtemporalresponse2 = 1, const float LPfilterSpatialResponse = 5, const float LPfilterGain = 0, const float LPfilterTemporalresponse = 0, const float MovingContoursExtractorCoefficient = 5, const bool normalizeParvoOutput_0_maxOutputValue = false, const bool normalizeMagnoOutput_0_maxOutputValue = false, const float maxOutputValue = 255.0, const float maxInputValue = 255.0, const float meanValue = 128.0);

    inline void setPhotoreceptorsLocalAdaptationSensitivity(const float V0CompressionParameter)
    {
        _photoreceptorsPrefilter.setV0CompressionParameter(1 - V0CompressionParameter);
        _setInitPeriodCount();
    }

    inline void setParvoGanglionCellsLocalAdaptationSensitivity(const float V0CompressionParameter)
    {
        _ParvoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);
        _setInitPeriodCount();
    }

    inline void setGanglionCellsLocalAdaptationLPfilterParameters(const float spatialResponse, const float temporalResponse)
    {
        _ParvoRetinaFilter.setGanglionCellsLocalAdaptationLPfilterParameters(temporalResponse, spatialResponse);
        _setInitPeriodCount();
    };

    inline void setMagnoGanglionCellsLocalAdaptationSensitivity(const float V0CompressionParameter)
    {
        _MagnoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);
        _setInitPeriodCount();
    }

    void setOPLandParvoParameters(const float beta1, const float tau1, const float k1, const float beta2, const float tau2, const float k2, const float V0CompressionParameter)
    {
        _ParvoRetinaFilter.setOPLandParvoFiltersParameters(beta1, tau1, k1, beta2, tau2, k2);
        _ParvoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);
        _setInitPeriodCount();
    }

    void setMagnoCoefficientsTable(const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float V0CompressionParameter, const float localAdaptintegration_tau, const float localAdaptintegration_k)
    {
        _MagnoRetinaFilter.setCoefficientsTable(parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency, localAdaptintegration_tau, localAdaptintegration_k);
        _MagnoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);
        _setInitPeriodCount();
    }

    inline void activateNormalizeParvoOutput_0_maxOutputValue(const bool normalizeParvoOutput_0_maxOutputValue)
    {
        _normalizeParvoOutput_0_maxOutputValue = normalizeParvoOutput_0_maxOutputValue;
    }

    inline void activateNormalizeMagnoOutput_0_maxOutputValue(const bool normalizeMagnoOutput_0_maxOutputValue)
    {
        _normalizeMagnoOutput_0_maxOutputValue = normalizeMagnoOutput_0_maxOutputValue;
    }

    inline void setMaxOutputValue(const float maxOutputValue)
    {
        _maxOutputValue = maxOutputValue;
    }

    void setColorMode(const bool desiredColorMode)
    {
        _useColorMode = desiredColorMode;
    }
    inline void setColorSaturation(const bool saturateColors = true, const float colorSaturationValue = 4.0)
    {
        _colorEngine.setColorSaturation(saturateColors, colorSaturationValue);
    }
    inline const UMat &getLocalAdaptation() const
    {
        return _photoreceptorsPrefilter.getOutput();
    }
    inline const UMat &getPhotoreceptors() const
    {
        return _ParvoRetinaFilter.getPhotoreceptorsLPfilteringOutput();
    }

    inline const UMat &getHorizontalCells() const
    {
        return _ParvoRetinaFilter.getHorizontalCellsOutput();
    }
    inline bool areContoursProcessed()
    {
        return _useParvoOutput;
    }
    bool getParvoFoveaResponse(UMat &parvoFovealResponse);
    inline void activateContoursProcessing(const bool useParvoOutput)
    {
        _useParvoOutput = useParvoOutput;
    }

    const UMat &getContours();

    inline const UMat &getContoursON() const
    {
        return _ParvoRetinaFilter.getParvoON();
    }

    inline const UMat &getContoursOFF() const
    {
        return _ParvoRetinaFilter.getParvoOFF();
    }

    inline bool areMovingContoursProcessed()
    {
        return _useMagnoOutput;
    }

    inline void activateMovingContoursProcessing(const bool useMagnoOutput)
    {
        _useMagnoOutput = useMagnoOutput;
    }

    inline const UMat &getMovingContours() const
    {
        return _MagnoRetinaFilter.getOutput();
    }

    inline const UMat &getMovingContoursSaturated() const
    {
        return _MagnoRetinaFilter.getMagnoYsaturated();
    }

    inline const UMat &getMovingContoursON() const
    {
        return _MagnoRetinaFilter.getMagnoON();
    }

    inline const UMat &getMovingContoursOFF() const
    {
        return _MagnoRetinaFilter.getMagnoOFF();
    }

    inline const UMat &getRetinaParvoMagnoMappedOutput() const
    {
        return _retinaParvoMagnoMappedFrame;
    }

    inline const UMat &getParvoContoursChannel() const
    {
        return _colorEngine.getLuminance();
    }

    inline const UMat &getParvoChrominance() const
    {
        return _colorEngine.getChrominance();
    }
    inline const UMat &getColorOutput() const
    {
        return _colorEngine.getDemultiplexedColorFrame();
    }

    inline bool isColorMode()
    {
        return _useColorMode;
    }
    bool getColorMode()
    {
        return _useColorMode;
    }

    inline bool isInitTransitionDone()
    {
        if (_ellapsedFramesSinceLastReset < _globalTemporalConstant)
        {
            return false;
        }
        return true;
    }
    inline float getRetinaSamplingBackProjection(const float projectedRadiusLength)
    {
        return projectedRadiusLength;
    }

    inline unsigned int getInputNBrows()
    {
        return _photoreceptorsPrefilter.getNBrows();
    }

    inline unsigned int getInputNBcolumns()
    {
        return _photoreceptorsPrefilter.getNBcolumns();
    }

    inline unsigned int getInputNBpixels()
    {
        return _photoreceptorsPrefilter.getNBpixels();
    }

    inline unsigned int getOutputNBrows()
    {
        return _photoreceptorsPrefilter.getNBrows();
    }

    inline unsigned int getOutputNBcolumns()
    {
        return _photoreceptorsPrefilter.getNBcolumns();
    }

    inline unsigned int getOutputNBpixels()
    {
        return _photoreceptorsPrefilter.getNBpixels();
    }
private:
    bool _useParvoOutput;
    bool _useMagnoOutput;

    unsigned int _ellapsedFramesSinceLastReset;
    unsigned int _globalTemporalConstant;

    UMat _retinaParvoMagnoMappedFrame;
    BasicRetinaFilter _photoreceptorsPrefilter;
    ParvoRetinaFilter _ParvoRetinaFilter;
    MagnoRetinaFilter _MagnoRetinaFilter;
    RetinaColor       _colorEngine;

    bool _useMinimalMemoryForToneMappingONLY;
    bool _normalizeParvoOutput_0_maxOutputValue;
    bool _normalizeMagnoOutput_0_maxOutputValue;
    float _maxOutputValue;
    bool _useColorMode;

    void _setInitPeriodCount();
    void _processRetinaParvoMagnoMapping();
    void _runGrayToneMapping(const UMat &grayImageInput, UMat &grayImageOutput , const float PhotoreceptorsCompression = 0.6, const float ganglionCellsCompression = 0.6);
};

class RetinaOCLImpl CV_FINAL : public Retina
{
public:
    RetinaOCLImpl(Size getInputSize);
    RetinaOCLImpl(Size getInputSize, const bool colorMode, int colorSamplingMethod = RETINA_COLOR_BAYER, const bool useRetinaLogSampling = false, const double reductionFactor = 1.0, const double samplingStrength = 10.0);
    virtual ~RetinaOCLImpl() CV_OVERRIDE;

    Size getInputSize() CV_OVERRIDE;
    Size getOutputSize() CV_OVERRIDE;

    void setup(String retinaParameterFile = "", const bool applyDefaultSetupOnFailure = true) CV_OVERRIDE;
    void setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure = true) CV_OVERRIDE;
    void setup(RetinaParameters newParameters) CV_OVERRIDE;

    RetinaParameters getParameters() CV_OVERRIDE;

    String printSetup() CV_OVERRIDE;
    virtual void write(String fs) const CV_OVERRIDE;
    virtual void write(FileStorage& fs) const CV_OVERRIDE;

    void setupOPLandIPLParvoChannel(const bool colorMode = true, const bool normaliseOutput = true, const float photoreceptorsLocalAdaptationSensitivity = 0.7, const float photoreceptorsTemporalConstant = 0.5, const float photoreceptorsSpatialConstant = 0.53, const float horizontalCellsGain = 0, const float HcellsTemporalConstant = 1, const float HcellsSpatialConstant = 7, const float ganglionCellsSensitivity = 0.7) CV_OVERRIDE;
    void setupIPLMagnoChannel(const bool normaliseOutput = true, const float parasolCells_beta = 0, const float parasolCells_tau = 0, const float parasolCells_k = 7, const float amacrinCellsTemporalCutFrequency = 1.2, const float V0CompressionParameter = 0.95, const float localAdaptintegration_tau = 0, const float localAdaptintegration_k = 7) CV_OVERRIDE;

    void run(InputArray inputImage) CV_OVERRIDE;
    void getParvo(OutputArray retinaOutput_parvo) CV_OVERRIDE;
    void getMagno(OutputArray retinaOutput_magno) CV_OVERRIDE;

    void setColorSaturation(const bool saturateColors = true, const float colorSaturationValue = 4.0) CV_OVERRIDE;
    void clearBuffers() CV_OVERRIDE;
    void activateMovingContoursProcessing(const bool activate) CV_OVERRIDE;
    void activateContoursProcessing(const bool activate) CV_OVERRIDE;

    // unimplemented interfaces:
    void applyFastToneMapping(InputArray /*inputImage*/, OutputArray /*outputToneMappedImage*/) CV_OVERRIDE;
    void getParvoRAW(OutputArray /*retinaOutput_parvo*/) CV_OVERRIDE;
    void getMagnoRAW(OutputArray /*retinaOutput_magno*/) CV_OVERRIDE;
    Mat getMagnoRAW() const CV_OVERRIDE;
    Mat getParvoRAW() const CV_OVERRIDE;

protected:
    RetinaParameters _retinaParameters;
    UMat _inputBuffer;
    cv::Ptr<RetinaFilter> _retinaFilter;
    bool convertToColorPlanes(const UMat& input, UMat &output);
    void convertToInterleaved(const UMat& input, bool colorMode, UMat &output);
    void _init(const Size getInputSize, const bool colorMode, int colorSamplingMethod = RETINA_COLOR_BAYER, const bool useRetinaLogSampling = false, const double reductionFactor = 1.0, const double samplingStrength = 10.0);
};

}  /* namespace ocl */
}  /* namespace bioinspired */
}  /* namespace cv */

#endif  /* HAVE_OPENCL */
#endif  /* __OCL_RETINA_HPP__ */
