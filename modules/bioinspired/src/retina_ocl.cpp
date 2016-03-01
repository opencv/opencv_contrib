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
//     and/or other oclMaterials provided with the distribution.
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

#include "precomp.hpp"
#include "retina_ocl.hpp"
#include <iostream>
#include <sstream>

#ifdef HAVE_OPENCL

#include "opencl_kernels_bioinspired.hpp"

#define NOT_IMPLEMENTED CV_Error(cv::Error::StsNotImplemented, "Not implemented")

namespace
{
    template <typename T, size_t N>
    inline size_t sizeOfArray(const T(&)[N])
    {
        return N;
    }

    inline void ensureSizeIsEnough(int rows, int cols, int type, cv::UMat &m)
    {
        m.create(rows, cols, type, m.usageFlags);
    }
}

namespace cv
{
static ocl::ProgramEntry retina_kernel = ocl::bioinspired::retina_kernel;
static ocl::ProgramSource retina_kernel_source(retina_kernel.programStr);

namespace bioinspired
{
namespace ocl
{
using namespace cv::ocl;

RetinaOCLImpl::RetinaOCLImpl(const cv::Size inputSz)
{
    _retinaFilter = 0;
    _init(inputSz, true, RETINA_COLOR_BAYER, false);
}

RetinaOCLImpl::RetinaOCLImpl(const cv::Size inputSz, const bool colorMode, int colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    _retinaFilter = 0;
    _init(inputSz, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);
}

RetinaOCLImpl::~RetinaOCLImpl()
{
    if (_retinaFilter)
    {
        delete _retinaFilter;
    }
}

/**
* retrieve retina input buffer size
*/
Size RetinaOCLImpl::getInputSize()
{
    return cv::Size(_retinaFilter->getInputNBcolumns(), _retinaFilter->getInputNBrows());
}

/**
* retrieve retina output buffer size
*/
Size RetinaOCLImpl::getOutputSize()
{
    return cv::Size(_retinaFilter->getOutputNBcolumns(), _retinaFilter->getOutputNBrows());
}


void RetinaOCLImpl::setColorSaturation(const bool saturateColors, const float colorSaturationValue)
{
    _retinaFilter->setColorSaturation(saturateColors, colorSaturationValue);
}

struct RetinaParameters RetinaOCLImpl::getParameters()
{
    return _retinaParameters;
}


void RetinaOCLImpl::setup(String retinaParameterFile, const bool applyDefaultSetupOnFailure)
{
    try
    {
        // opening retinaParameterFile in read mode
        cv::FileStorage fs(retinaParameterFile, cv::FileStorage::READ);
        setup(fs, applyDefaultSetupOnFailure);
    }
    catch(Exception &e)
    {
        std::cout << "RetinaOCLImpl::setup: wrong/inappropriate xml parameter file : error report :`n=>" << e.what() << std::endl;
        if (applyDefaultSetupOnFailure)
        {
            std::cout << "RetinaOCLImpl::setup: resetting retina with default parameters" << std::endl;
            setupOPLandIPLParvoChannel();
            setupIPLMagnoChannel();
        }
        else
        {
            std::cout << "=> keeping current parameters" << std::endl;
        }
    }
}

void RetinaOCLImpl::setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure)
{
    try
    {
        // read parameters file if it exists or apply default setup if asked for
        if (!fs.isOpened())
        {
            std::cout << "RetinaOCLImpl::setup: provided parameters file could not be open... skipping configuration" << std::endl;
            return;
            // implicit else case : retinaParameterFile could be open (it exists at least)
        }
        // OPL and Parvo init first... update at the same time the parameters structure and the retina core
        cv::FileNode rootFn = fs.root(), currFn = rootFn["OPLandIPLparvo"];
        currFn["colorMode"] >> _retinaParameters.OPLandIplParvo.colorMode;
        currFn["normaliseOutput"] >> _retinaParameters.OPLandIplParvo.normaliseOutput;
        currFn["photoreceptorsLocalAdaptationSensitivity"] >> _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity;
        currFn["photoreceptorsTemporalConstant"] >> _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant;
        currFn["photoreceptorsSpatialConstant"] >> _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant;
        currFn["horizontalCellsGain"] >> _retinaParameters.OPLandIplParvo.horizontalCellsGain;
        currFn["hcellsTemporalConstant"] >> _retinaParameters.OPLandIplParvo.hcellsTemporalConstant;
        currFn["hcellsSpatialConstant"] >> _retinaParameters.OPLandIplParvo.hcellsSpatialConstant;
        currFn["ganglionCellsSensitivity"] >> _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity;
        setupOPLandIPLParvoChannel(_retinaParameters.OPLandIplParvo.colorMode, _retinaParameters.OPLandIplParvo.normaliseOutput, _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity, _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant, _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant, _retinaParameters.OPLandIplParvo.horizontalCellsGain, _retinaParameters.OPLandIplParvo.hcellsTemporalConstant, _retinaParameters.OPLandIplParvo.hcellsSpatialConstant, _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity);

        // init retina IPL magno setup... update at the same time the parameters structure and the retina core
        currFn = rootFn["IPLmagno"];
        currFn["normaliseOutput"] >> _retinaParameters.IplMagno.normaliseOutput;
        currFn["parasolCells_beta"] >> _retinaParameters.IplMagno.parasolCells_beta;
        currFn["parasolCells_tau"] >> _retinaParameters.IplMagno.parasolCells_tau;
        currFn["parasolCells_k"] >> _retinaParameters.IplMagno.parasolCells_k;
        currFn["amacrinCellsTemporalCutFrequency"] >> _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency;
        currFn["V0CompressionParameter"] >> _retinaParameters.IplMagno.V0CompressionParameter;
        currFn["localAdaptintegration_tau"] >> _retinaParameters.IplMagno.localAdaptintegration_tau;
        currFn["localAdaptintegration_k"] >> _retinaParameters.IplMagno.localAdaptintegration_k;

        setupIPLMagnoChannel(_retinaParameters.IplMagno.normaliseOutput, _retinaParameters.IplMagno.parasolCells_beta, _retinaParameters.IplMagno.parasolCells_tau, _retinaParameters.IplMagno.parasolCells_k, _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency, _retinaParameters.IplMagno.V0CompressionParameter, _retinaParameters.IplMagno.localAdaptintegration_tau, _retinaParameters.IplMagno.localAdaptintegration_k);

    }
    catch(Exception &e)
    {
        std::cout << "RetinaOCLImpl::setup: resetting retina with default parameters" << std::endl;
        if (applyDefaultSetupOnFailure)
        {
            setupOPLandIPLParvoChannel();
            setupIPLMagnoChannel();
        }
        std::cout << "RetinaOCLImpl::setup: wrong/inappropriate xml parameter file : error report :`n=>" << e.what() << std::endl;
        std::cout << "=> keeping current parameters" << std::endl;
    }
}

void RetinaOCLImpl::setup(cv::bioinspired::RetinaParameters newConfiguration)
{
    // simply copy structures
    memcpy(&_retinaParameters, &newConfiguration, sizeof(cv::bioinspired::RetinaParameters));
    // apply setup
    setupOPLandIPLParvoChannel(_retinaParameters.OPLandIplParvo.colorMode, _retinaParameters.OPLandIplParvo.normaliseOutput, _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity, _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant, _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant, _retinaParameters.OPLandIplParvo.horizontalCellsGain, _retinaParameters.OPLandIplParvo.hcellsTemporalConstant, _retinaParameters.OPLandIplParvo.hcellsSpatialConstant, _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity);
    setupIPLMagnoChannel(_retinaParameters.IplMagno.normaliseOutput, _retinaParameters.IplMagno.parasolCells_beta, _retinaParameters.IplMagno.parasolCells_tau, _retinaParameters.IplMagno.parasolCells_k, _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency, _retinaParameters.IplMagno.V0CompressionParameter, _retinaParameters.IplMagno.localAdaptintegration_tau, _retinaParameters.IplMagno.localAdaptintegration_k);
}

const String RetinaOCLImpl::printSetup()
{
    std::stringstream outmessage;

    // displaying OPL and IPL parvo setup
    outmessage << "Current Retina instance setup :"
               << "\nOPLandIPLparvo" << "{"
               << "\n==> colorMode : " << _retinaParameters.OPLandIplParvo.colorMode
               << "\n==> normalizeParvoOutput :" << _retinaParameters.OPLandIplParvo.normaliseOutput
               << "\n==> photoreceptorsLocalAdaptationSensitivity : " << _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity
               << "\n==> photoreceptorsTemporalConstant : " << _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant
               << "\n==> photoreceptorsSpatialConstant : " << _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant
               << "\n==> horizontalCellsGain : " << _retinaParameters.OPLandIplParvo.horizontalCellsGain
               << "\n==> hcellsTemporalConstant : " << _retinaParameters.OPLandIplParvo.hcellsTemporalConstant
               << "\n==> hcellsSpatialConstant : " << _retinaParameters.OPLandIplParvo.hcellsSpatialConstant
               << "\n==> parvoGanglionCellsSensitivity : " << _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity
               << "}\n";

    // displaying IPL magno setup
    outmessage << "Current Retina instance setup :"
               << "\nIPLmagno" << "{"
               << "\n==> normaliseOutput : " << _retinaParameters.IplMagno.normaliseOutput
               << "\n==> parasolCells_beta : " << _retinaParameters.IplMagno.parasolCells_beta
               << "\n==> parasolCells_tau : " << _retinaParameters.IplMagno.parasolCells_tau
               << "\n==> parasolCells_k : " << _retinaParameters.IplMagno.parasolCells_k
               << "\n==> amacrinCellsTemporalCutFrequency : " << _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency
               << "\n==> V0CompressionParameter : " << _retinaParameters.IplMagno.V0CompressionParameter
               << "\n==> localAdaptintegration_tau : " << _retinaParameters.IplMagno.localAdaptintegration_tau
               << "\n==> localAdaptintegration_k : " << _retinaParameters.IplMagno.localAdaptintegration_k
               << "}";
    return outmessage.str().c_str();
}

void RetinaOCLImpl::write( String fs ) const
{
    FileStorage parametersSaveFile(fs, cv::FileStorage::WRITE );
    write(parametersSaveFile);
}

void RetinaOCLImpl::write( FileStorage& fs ) const
{
    if (!fs.isOpened())
    {
        return;    // basic error case
    }
    fs << "OPLandIPLparvo" << "{";
    fs << "colorMode" << _retinaParameters.OPLandIplParvo.colorMode;
    fs << "normaliseOutput" << _retinaParameters.OPLandIplParvo.normaliseOutput;
    fs << "photoreceptorsLocalAdaptationSensitivity" << _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity;
    fs << "photoreceptorsTemporalConstant" << _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant;
    fs << "photoreceptorsSpatialConstant" << _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant;
    fs << "horizontalCellsGain" << _retinaParameters.OPLandIplParvo.horizontalCellsGain;
    fs << "hcellsTemporalConstant" << _retinaParameters.OPLandIplParvo.hcellsTemporalConstant;
    fs << "hcellsSpatialConstant" << _retinaParameters.OPLandIplParvo.hcellsSpatialConstant;
    fs << "ganglionCellsSensitivity" << _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity;
    fs << "}";
    fs << "IPLmagno" << "{";
    fs << "normaliseOutput" << _retinaParameters.IplMagno.normaliseOutput;
    fs << "parasolCells_beta" << _retinaParameters.IplMagno.parasolCells_beta;
    fs << "parasolCells_tau" << _retinaParameters.IplMagno.parasolCells_tau;
    fs << "parasolCells_k" << _retinaParameters.IplMagno.parasolCells_k;
    fs << "amacrinCellsTemporalCutFrequency" << _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency;
    fs << "V0CompressionParameter" << _retinaParameters.IplMagno.V0CompressionParameter;
    fs << "localAdaptintegration_tau" << _retinaParameters.IplMagno.localAdaptintegration_tau;
    fs << "localAdaptintegration_k" << _retinaParameters.IplMagno.localAdaptintegration_k;
    fs << "}";
}

void RetinaOCLImpl::setupOPLandIPLParvoChannel(const bool colorMode, const bool normaliseOutput, const float photoreceptorsLocalAdaptationSensitivity, const float photoreceptorsTemporalConstant, const float photoreceptorsSpatialConstant, const float horizontalCellsGain, const float HcellsTemporalConstant, const float HcellsSpatialConstant, const float ganglionCellsSensitivity)
{
    // retina core parameters setup
    _retinaFilter->setColorMode(colorMode);
    _retinaFilter->setPhotoreceptorsLocalAdaptationSensitivity(photoreceptorsLocalAdaptationSensitivity);
    _retinaFilter->setOPLandParvoParameters(0, photoreceptorsTemporalConstant, photoreceptorsSpatialConstant, horizontalCellsGain, HcellsTemporalConstant, HcellsSpatialConstant, ganglionCellsSensitivity);
    _retinaFilter->setParvoGanglionCellsLocalAdaptationSensitivity(ganglionCellsSensitivity);
    _retinaFilter->activateNormalizeParvoOutput_0_maxOutputValue(normaliseOutput);

    // update parameters structure

    _retinaParameters.OPLandIplParvo.colorMode = colorMode;
    _retinaParameters.OPLandIplParvo.normaliseOutput = normaliseOutput;
    _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity = photoreceptorsLocalAdaptationSensitivity;
    _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant = photoreceptorsTemporalConstant;
    _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant = photoreceptorsSpatialConstant;
    _retinaParameters.OPLandIplParvo.horizontalCellsGain = horizontalCellsGain;
    _retinaParameters.OPLandIplParvo.hcellsTemporalConstant = HcellsTemporalConstant;
    _retinaParameters.OPLandIplParvo.hcellsSpatialConstant = HcellsSpatialConstant;
    _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity = ganglionCellsSensitivity;
}

void RetinaOCLImpl::setupIPLMagnoChannel(const bool normaliseOutput, const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float V0CompressionParameter, const float localAdaptintegration_tau, const float localAdaptintegration_k)
{

    _retinaFilter->setMagnoCoefficientsTable(parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency, V0CompressionParameter, localAdaptintegration_tau, localAdaptintegration_k);
    _retinaFilter->activateNormalizeMagnoOutput_0_maxOutputValue(normaliseOutput);

    // update parameters structure
    _retinaParameters.IplMagno.normaliseOutput = normaliseOutput;
    _retinaParameters.IplMagno.parasolCells_beta = parasolCells_beta;
    _retinaParameters.IplMagno.parasolCells_tau = parasolCells_tau;
    _retinaParameters.IplMagno.parasolCells_k = parasolCells_k;
    _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency = amacrinCellsTemporalCutFrequency;
    _retinaParameters.IplMagno.V0CompressionParameter = V0CompressionParameter;
    _retinaParameters.IplMagno.localAdaptintegration_tau = localAdaptintegration_tau;
    _retinaParameters.IplMagno.localAdaptintegration_k = localAdaptintegration_k;
}

void RetinaOCLImpl::run(InputArray input)
{
    UMat inputMatToConvert = input.getUMat();
    bool colorMode = convertToColorPlanes(inputMatToConvert, _inputBuffer);
    // first convert input image to the compatible format : std::valarray<float>
    // process the retina
    if (!_retinaFilter->runFilter(_inputBuffer, colorMode, false, _retinaParameters.OPLandIplParvo.colorMode && colorMode, false))
    {
        throw cv::Exception(-1, "Retina cannot be applied, wrong input buffer size", "RetinaOCLImpl::run", "Retina.h", 0);
    }
}

void RetinaOCLImpl::getParvo(OutputArray output)
{
    UMat &retinaOutput_parvo = output.getUMatRef();
    if (_retinaFilter->getColorMode())
    {
        // reallocate output buffer (if necessary)
        convertToInterleaved(_retinaFilter->getColorOutput(), true, retinaOutput_parvo);
    }
    else
    {
        // reallocate output buffer (if necessary)
        convertToInterleaved(_retinaFilter->getContours(), false, retinaOutput_parvo);
    }
    //retinaOutput_parvo/=255.0;
}
void RetinaOCLImpl::getMagno(OutputArray output)
{
    UMat &retinaOutput_magno = output.getUMatRef();
    // reallocate output buffer (if necessary)
    convertToInterleaved(_retinaFilter->getMovingContours(), false, retinaOutput_magno);
    //retinaOutput_magno/=255.0;
}
// private method called by constructors
void RetinaOCLImpl::_init(const cv::Size inputSz, const bool colorMode, int colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    // basic error check
    if (inputSz.height*inputSz.width <= 0)
    {
        throw cv::Exception(-1, "Bad retina size setup : size height and with must be superior to zero", "RetinaOCLImpl::setup", "Retina.h", 0);
    }

    // allocate the retina model
    if (_retinaFilter)
    {
        delete _retinaFilter;
    }
    _retinaFilter = new RetinaFilter(inputSz.height, inputSz.width, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);

    // prepare the default parameter XML file with default setup
    setup(_retinaParameters);

    // init retina
    _retinaFilter->clearAllBuffers();
}

bool RetinaOCLImpl::convertToColorPlanes(const UMat& input, UMat &output)
{
    UMat convert_input;
    input.convertTo(convert_input, CV_32F);
    if(convert_input.channels() == 3 || convert_input.channels() == 4)
    {
        ensureSizeIsEnough(int(_retinaFilter->getInputNBrows() * 4),
                           int(_retinaFilter->getInputNBcolumns()), CV_32FC1, output);
        std::vector<UMat> channel_splits;
        channel_splits.reserve(4);
        channel_splits.push_back(output(Rect(Point(0, _retinaFilter->getInputNBrows() * 2), getInputSize())));
        channel_splits.push_back(output(Rect(Point(0, _retinaFilter->getInputNBrows()), getInputSize())));
        channel_splits.push_back(output(Rect(Point(0, 0), getInputSize())));
        channel_splits.push_back(output(Rect(Point(0, _retinaFilter->getInputNBrows() * 3), getInputSize())));

        cv::split(convert_input, channel_splits);
        return true;
    }
    else if(convert_input.channels() == 1)
    {
        convert_input.copyTo(output);
        return false;
    }
    else
    {
        CV_Error(-1, "Retina ocl only support 1, 3, 4 channel input");
        return false;
    }
}
void RetinaOCLImpl::convertToInterleaved(const UMat& input, bool colorMode, UMat &output)
{
    input.convertTo(output, CV_8U);
    if(colorMode)
    {
        int numOfSplits = input.rows / getInputSize().height;
        std::vector<UMat> channel_splits(numOfSplits);
        for(int i = 0; i < static_cast<int>(channel_splits.size()); i ++)
        {
            channel_splits[i] =
                output(Rect(Point(0, _retinaFilter->getInputNBrows() * (numOfSplits - i - 1)), getInputSize()));
        }
        merge(channel_splits, output);
    }
    else
    {
        //...
    }
}

void RetinaOCLImpl::clearBuffers()
{
    _retinaFilter->clearAllBuffers();
}

void RetinaOCLImpl::activateMovingContoursProcessing(const bool activate)
{
    _retinaFilter->activateMovingContoursProcessing(activate);
}

void RetinaOCLImpl::activateContoursProcessing(const bool activate)
{
    _retinaFilter->activateContoursProcessing(activate);
}

// unimplemented interfaces:
void RetinaOCLImpl::applyFastToneMapping(InputArray /*inputImage*/, OutputArray /*outputToneMappedImage*/) { NOT_IMPLEMENTED; }
void RetinaOCLImpl::getParvoRAW(OutputArray /*retinaOutput_parvo*/) { NOT_IMPLEMENTED; }
void RetinaOCLImpl::getMagnoRAW(OutputArray /*retinaOutput_magno*/) { NOT_IMPLEMENTED; }
const Mat RetinaOCLImpl::getMagnoRAW() const { NOT_IMPLEMENTED; return Mat(); }
const Mat RetinaOCLImpl::getParvoRAW() const { NOT_IMPLEMENTED; return Mat(); }

///////////////////////////////////////
///////// BasicRetinaFilter ///////////
///////////////////////////////////////
BasicRetinaFilter::BasicRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns, const unsigned int parametersListSize, const bool)
    : _NBrows(NBrows), _NBcols(NBcolumns),
      _filterOutput(NBrows, NBcolumns, CV_32FC1),
      _localBuffer(NBrows, NBcolumns, CV_32FC1),
      _filteringCoeficientsTable(3 * parametersListSize)
{
    _halfNBrows = _filterOutput.rows / 2;
    _halfNBcolumns = _filterOutput.cols / 2;

    // set default values
    _maxInputValue = 256.0;

    // reset all buffers
    clearAllBuffers();
}

BasicRetinaFilter::~BasicRetinaFilter()
{
}

void BasicRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
    // resizing buffers
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _filterOutput);

    // updating variables
    _halfNBrows = _filterOutput.rows / 2;
    _halfNBcolumns = _filterOutput.cols / 2;

    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _localBuffer);
    // reset buffers
    clearAllBuffers();
}

void BasicRetinaFilter::setLPfilterParameters(const float beta, const float tau, const float desired_k, const unsigned int filterIndex)
{
    float _beta = beta + tau;
    float k = desired_k;
    // check if the spatial constant is correct (avoid 0 value to avoid division by 0)
    if (desired_k <= 0)
    {
        k = 0.001f;
        std::cerr << "BasicRetinaFilter::spatial constant of the low pass filter must be superior to zero !!! correcting parameter setting to 0,001" << std::endl;
    }

    float _alpha = k * k;
    float _mu = 0.8f;
    unsigned int tableOffset = filterIndex * 3;
    if (k <= 0)
    {
        std::cerr << "BasicRetinaFilter::spatial filtering coefficient must be superior to zero, correcting value to 0.01" << std::endl;
        _alpha = 0.0001f;
    }

    float _temp =  (1.0f + _beta) / (2.0f * _mu * _alpha);
    float a = _filteringCoeficientsTable[tableOffset] = 1.0f + _temp - (float)sqrt( (1.0f + _temp) * (1.0f + _temp) - 1.0f);
    _filteringCoeficientsTable[1 + tableOffset] = (1.0f - a) * (1.0f - a) * (1.0f - a) * (1.0f - a) / (1.0f + _beta);
    _filteringCoeficientsTable[2 + tableOffset] = tau;
}
const UMat &BasicRetinaFilter::runFilter_LocalAdapdation(const UMat &inputFrame, const UMat &localLuminance)
{
    _localLuminanceAdaptation(inputFrame, localLuminance, _filterOutput);
    return _filterOutput;
}


void BasicRetinaFilter::runFilter_LocalAdapdation(const UMat &inputFrame, const UMat &localLuminance, UMat &outputFrame)
{
    _localLuminanceAdaptation(inputFrame, localLuminance, outputFrame);
}

const UMat &BasicRetinaFilter::runFilter_LocalAdapdation_autonomous(const UMat &inputFrame)
{
    _spatiotemporalLPfilter(inputFrame, _filterOutput);
    _localLuminanceAdaptation(inputFrame, _filterOutput, _filterOutput);
    return _filterOutput;
}
void BasicRetinaFilter::runFilter_LocalAdapdation_autonomous(const UMat &inputFrame, UMat &outputFrame)
{
    _spatiotemporalLPfilter(inputFrame, _filterOutput);
    _localLuminanceAdaptation(inputFrame, _filterOutput, outputFrame);
}

void BasicRetinaFilter::_localLuminanceAdaptation(UMat &inputOutputFrame, const UMat &localLuminance)
{
    _localLuminanceAdaptation(inputOutputFrame, localLuminance, inputOutputFrame, false);
}

void BasicRetinaFilter::_localLuminanceAdaptation(const UMat &inputFrame, const UMat &localLuminance, UMat &outputFrame, const bool updateLuminanceMean)
{
    if (updateLuminanceMean)
    {
        float meanLuminance = saturate_cast<float>(cv::sum(inputFrame)[0]) / getNBpixels();
        updateCompressionParameter(meanLuminance);
    }
    int elements_per_row = static_cast<int>(inputFrame.step / inputFrame.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, _NBrows, 1};
    size_t localSize[]  = {16, 16, 1};

    Kernel kernel("localLuminanceAdaptation", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(localLuminance), KernelArg::PtrReadOnly(inputFrame), KernelArg::PtrWriteOnly(outputFrame),
                _NBcols, _NBrows, elements_per_row, _localLuminanceAddon, _localLuminanceFactor, _maxInputValue);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

const UMat &BasicRetinaFilter::runFilter_LPfilter(const UMat &inputFrame, const unsigned int filterIndex)
{
    _spatiotemporalLPfilter(inputFrame, _filterOutput, filterIndex);
    return _filterOutput;
}
void BasicRetinaFilter::runFilter_LPfilter(const UMat &inputFrame, UMat &outputFrame, const unsigned int filterIndex)
{
    _spatiotemporalLPfilter(inputFrame, outputFrame, filterIndex);
}

void BasicRetinaFilter::_spatiotemporalLPfilter(const UMat &inputFrame, UMat &LPfilterOutput, const unsigned int filterIndex)
{
    unsigned int coefTableOffset = filterIndex * 3;

    _a = _filteringCoeficientsTable[coefTableOffset];
    _gain = _filteringCoeficientsTable[1 + coefTableOffset];
    _tau = _filteringCoeficientsTable[2 + coefTableOffset];

    _horizontalCausalFilter_addInput(inputFrame, LPfilterOutput);
    _horizontalAnticausalFilter(LPfilterOutput);
    _verticalCausalFilter(LPfilterOutput);
    _verticalAnticausalFilter_multGain(LPfilterOutput);
}

void BasicRetinaFilter::_horizontalCausalFilter_addInput(const UMat &inputFrame, UMat &outputFrame)
{
    int elements_per_row = static_cast<int>(inputFrame.step / inputFrame.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBrows, 1, 1};
    size_t localSize[] = { 256, 1, 1 };

    Kernel kernel("horizontalCausalFilter_addInput", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(inputFrame), KernelArg::PtrWriteOnly(outputFrame), _NBcols, _NBrows, elements_per_row, inputFrame.offset, inputFrame.offset, _tau, _a);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void BasicRetinaFilter::_horizontalAnticausalFilter(UMat &outputFrame)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBrows, 1, 1};
    size_t localSize[] = { 256, 1, 1 };

    Kernel kernel("horizontalAnticausalFilter", retina_kernel_source);
    kernel.args(KernelArg::PtrReadWrite(outputFrame), _NBcols, _NBrows, elements_per_row, outputFrame.offset, _a);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void BasicRetinaFilter::_verticalCausalFilter(UMat &outputFrame)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, 1, 1};
    size_t localSize[] = { 256, 1, 1 };

    Kernel kernel("verticalCausalFilter", retina_kernel_source);
    kernel.args(KernelArg::PtrReadWrite(outputFrame), _NBcols, _NBrows, elements_per_row, outputFrame.offset, _a);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void BasicRetinaFilter::_verticalAnticausalFilter_multGain(UMat &outputFrame)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, 1, 1};
    size_t localSize[] = { 256, 1, 1 };

    Kernel kernel("verticalAnticausalFilter_multGain", retina_kernel_source);
    kernel.args(KernelArg::PtrReadWrite(outputFrame), _NBcols, _NBrows, elements_per_row, outputFrame.offset, _a, _gain);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void BasicRetinaFilter::_horizontalAnticausalFilter_Irregular(UMat &outputFrame, const UMat &spatialConstantBuffer)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {outputFrame.rows, 1, 1};
    size_t localSize[] = { 256, 1, 1 };

    Kernel kernel("horizontalAnticausalFilter_Irregular", retina_kernel_source);
    kernel.args(KernelArg::PtrReadWrite(outputFrame), KernelArg::PtrReadWrite(spatialConstantBuffer), outputFrame.cols, outputFrame.rows, elements_per_row, outputFrame.offset, spatialConstantBuffer.offset);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

//  vertical anticausal filter
void BasicRetinaFilter::_verticalCausalFilter_Irregular(UMat &outputFrame, const UMat &spatialConstantBuffer)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {outputFrame.cols, 1, 1};
    size_t localSize[] = { 256, 1, 1 };

    Kernel kernel("verticalCausalFilter_Irregular", retina_kernel_source);
    kernel.args(KernelArg::PtrReadWrite(outputFrame), KernelArg::PtrReadWrite(spatialConstantBuffer), outputFrame.cols, outputFrame.rows, elements_per_row, outputFrame.offset, spatialConstantBuffer.offset);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void normalizeGrayOutput_0_maxOutputValue(UMat &inputOutputBuffer, const float maxOutputValue)
{
    double min_val, max_val;
    cv::minMaxLoc(inputOutputBuffer, &min_val, &max_val);
    float factor = maxOutputValue / static_cast<float>(max_val - min_val);
    float offset = - static_cast<float>(min_val) * factor;
    cv::multiply(factor, inputOutputBuffer, inputOutputBuffer);
    cv::add(inputOutputBuffer, offset, inputOutputBuffer);
}

void normalizeGrayOutputCentredSigmoide(const float meanValue, const float sensitivity, UMat &in, UMat &out, const float maxValue)
{
    if (sensitivity == 1.0f)
    {
        std::cerr << "TemplateBuffer::TemplateBuffer<type>::normalizeGrayOutputCentredSigmoide error: 2nd parameter (sensitivity) must not equal 0, copying original data..." << std::endl;
        in.copyTo(out);
        return;
    }

    float X0 = maxValue / (sensitivity - 1.0f);

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {in.cols, out.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    int elements_per_row = static_cast<int>(out.step / out.elemSize());

    Kernel kernel("normalizeGrayOutputCentredSigmoide", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(in), KernelArg::PtrWriteOnly(out), in.cols, in.rows, elements_per_row, meanValue, X0);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void normalizeGrayOutputNearZeroCentreredSigmoide(UMat &inputPicture, UMat &outputBuffer, const float sensitivity, const float maxOutputValue)
{
    float X0cube = sensitivity * sensitivity * sensitivity;

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {inputPicture.cols, inputPicture.rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    int elements_per_row = static_cast<int>(inputPicture.step / inputPicture.elemSize());

    Kernel kernel("normalizeGrayOutputNearZeroCentreredSigmoide", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(inputPicture), KernelArg::PtrWriteOnly(outputBuffer), inputPicture.cols, inputPicture.rows, elements_per_row, maxOutputValue, X0cube);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void centerReductImageLuminance(UMat &inputoutput)
{
    Scalar mean, stddev;
    cv::meanStdDev(inputoutput.getMat(ACCESS_READ), mean, stddev);

    Context ctx = Context::getDefault();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {inputoutput.cols, inputoutput.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    float f_mean = static_cast<float>(mean[0]);
    float f_stddev = static_cast<float>(stddev[0]);
    int elements_per_row = static_cast<int>(inputoutput.step / inputoutput.elemSize());

    Kernel kernel("centerReductImageLuminance", retina_kernel_source);
    kernel.args(KernelArg::PtrReadWrite(inputoutput), inputoutput.cols, inputoutput.rows, elements_per_row, f_mean, f_stddev);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

///////////////////////////////////////
///////// ParvoRetinaFilter ///////////
///////////////////////////////////////
ParvoRetinaFilter::ParvoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns)
    : BasicRetinaFilter(NBrows, NBcolumns, 3),
      _photoreceptorsOutput(NBrows, NBcolumns, CV_32FC1),
      _horizontalCellsOutput(NBrows, NBcolumns, CV_32FC1),
      _parvocellularOutputON(NBrows, NBcolumns, CV_32FC1),
      _parvocellularOutputOFF(NBrows, NBcolumns, CV_32FC1),
      _bipolarCellsOutputON(NBrows, NBcolumns, CV_32FC1),
      _bipolarCellsOutputOFF(NBrows, NBcolumns, CV_32FC1),
      _localAdaptationOFF(NBrows, NBcolumns, CV_32FC1)
{
    // link to the required local parent adaptation buffers
    _localAdaptationON = _localBuffer;
    _parvocellularOutputONminusOFF = _filterOutput;

    // init: set all the values to 0
    clearAllBuffers();
}

ParvoRetinaFilter::~ParvoRetinaFilter()
{
}

void ParvoRetinaFilter::clearAllBuffers()
{
    BasicRetinaFilter::clearAllBuffers();
    _photoreceptorsOutput = 0;
    _horizontalCellsOutput = 0;
    _parvocellularOutputON = 0;
    _parvocellularOutputOFF = 0;
    _bipolarCellsOutputON = 0;
    _bipolarCellsOutputOFF = 0;
    _localAdaptationOFF = 0;
}
void ParvoRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
    BasicRetinaFilter::resize(NBrows, NBcolumns);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _photoreceptorsOutput);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _horizontalCellsOutput);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _parvocellularOutputON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _parvocellularOutputOFF);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _bipolarCellsOutputON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _bipolarCellsOutputOFF);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _localAdaptationOFF);

    // link to the required local parent adaptation buffers
    _localAdaptationON = _localBuffer;
    _parvocellularOutputONminusOFF = _filterOutput;

    // clean buffers
    clearAllBuffers();
}

void ParvoRetinaFilter::setOPLandParvoFiltersParameters(const float beta1, const float tau1, const float k1, const float beta2, const float tau2, const float k2)
{
    // init photoreceptors low pass filter
    setLPfilterParameters(beta1, tau1, k1);
    // init horizontal cells low pass filter
    setLPfilterParameters(beta2, tau2, k2, 1);
    // init parasol ganglion cells low pass filter (default parameters)
    setLPfilterParameters(0, tau1, k1, 2);

}
const UMat &ParvoRetinaFilter::runFilter(const UMat &inputFrame, const bool useParvoOutput)
{
    _spatiotemporalLPfilter(inputFrame, _photoreceptorsOutput);
    _spatiotemporalLPfilter(_photoreceptorsOutput, _horizontalCellsOutput, 1);
    _OPL_OnOffWaysComputing();

    if (useParvoOutput)
    {
        // local adaptation processes on ON and OFF ways
        _spatiotemporalLPfilter(_bipolarCellsOutputON, _localAdaptationON, 2);
        _localLuminanceAdaptation(_parvocellularOutputON, _localAdaptationON);
        _spatiotemporalLPfilter(_bipolarCellsOutputOFF, _localAdaptationOFF, 2);
        _localLuminanceAdaptation(_parvocellularOutputOFF, _localAdaptationOFF);
        cv::subtract(_parvocellularOutputON, _parvocellularOutputOFF, _parvocellularOutputONminusOFF);
    }

    return _parvocellularOutputONminusOFF;
}
void ParvoRetinaFilter::_OPL_OnOffWaysComputing()
{
    int elements_per_row = static_cast<int>(_photoreceptorsOutput.step / _photoreceptorsOutput.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {(_photoreceptorsOutput.cols + 3) / 4, _photoreceptorsOutput.rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("OPL_OnOffWaysComputing", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(_photoreceptorsOutput), KernelArg::PtrReadOnly(_horizontalCellsOutput),
                KernelArg::PtrWriteOnly(_bipolarCellsOutputON), KernelArg::PtrWriteOnly(_bipolarCellsOutputOFF),
                KernelArg::PtrWriteOnly(_parvocellularOutputON), KernelArg::PtrWriteOnly(_parvocellularOutputOFF),
                _photoreceptorsOutput.cols, _photoreceptorsOutput.rows, elements_per_row);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

///////////////////////////////////////
//////////// MagnoFilter //////////////
///////////////////////////////////////
MagnoRetinaFilter::MagnoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns)
    : BasicRetinaFilter(NBrows, NBcolumns, 2),
      _previousInput_ON(NBrows, NBcolumns, CV_32FC1),
      _previousInput_OFF(NBrows, NBcolumns, CV_32FC1),
      _amacrinCellsTempOutput_ON(NBrows, NBcolumns, CV_32FC1),
      _amacrinCellsTempOutput_OFF(NBrows, NBcolumns, CV_32FC1),
      _magnoXOutputON(NBrows, NBcolumns, CV_32FC1),
      _magnoXOutputOFF(NBrows, NBcolumns, CV_32FC1),
      _localProcessBufferON(NBrows, NBcolumns, CV_32FC1),
      _localProcessBufferOFF(NBrows, NBcolumns, CV_32FC1)
{
    _magnoYOutput = _filterOutput;
    _magnoYsaturated = _localBuffer;

    clearAllBuffers();
}

MagnoRetinaFilter::~MagnoRetinaFilter()
{
}
void MagnoRetinaFilter::clearAllBuffers()
{
    BasicRetinaFilter::clearAllBuffers();
    _previousInput_ON = 0;
    _previousInput_OFF = 0;
    _amacrinCellsTempOutput_ON = 0;
    _amacrinCellsTempOutput_OFF = 0;
    _magnoXOutputON = 0;
    _magnoXOutputOFF = 0;
    _localProcessBufferON = 0;
    _localProcessBufferOFF = 0;

}
void MagnoRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
    BasicRetinaFilter::resize(NBrows, NBcolumns);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _previousInput_ON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _previousInput_OFF);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _amacrinCellsTempOutput_ON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _amacrinCellsTempOutput_OFF);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _magnoXOutputON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _magnoXOutputOFF);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _localProcessBufferON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _localProcessBufferOFF);

    // to be sure, relink buffers
    _magnoYOutput = _filterOutput;
    _magnoYsaturated = _localBuffer;

    // reset all buffers
    clearAllBuffers();
}

void MagnoRetinaFilter::setCoefficientsTable(const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float localAdaptIntegration_tau, const float localAdaptIntegration_k )
{
    _temporalCoefficient = (float)std::exp(-1.0f / amacrinCellsTemporalCutFrequency);
    // the first set of parameters is dedicated to the low pass filtering property of the ganglion cells
    BasicRetinaFilter::setLPfilterParameters(parasolCells_beta, parasolCells_tau, parasolCells_k, 0);
    // the second set of parameters is dedicated to the ganglion cells output intergartion for their local adaptation property
    BasicRetinaFilter::setLPfilterParameters(0, localAdaptIntegration_tau, localAdaptIntegration_k, 1);
}

void MagnoRetinaFilter::_amacrineCellsComputing(
    const UMat &OPL_ON,
    const UMat &OPL_OFF
)
{
    int elements_per_row = static_cast<int>(OPL_ON.step / OPL_ON.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {OPL_ON.cols, OPL_ON.rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("amacrineCellsComputing", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(OPL_ON), KernelArg::PtrReadOnly(OPL_OFF),
                KernelArg::PtrReadWrite(_previousInput_ON), KernelArg::PtrReadWrite(_previousInput_OFF),
                KernelArg::PtrReadWrite(_amacrinCellsTempOutput_ON), KernelArg::PtrReadWrite(_amacrinCellsTempOutput_OFF),
                OPL_ON.cols, OPL_ON.rows, elements_per_row, _temporalCoefficient);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

const UMat &MagnoRetinaFilter::runFilter(const UMat &OPL_ON, const UMat &OPL_OFF)
{
    // Compute the high pass temporal filter
    _amacrineCellsComputing(OPL_ON, OPL_OFF);

    // apply low pass filtering on ON and OFF ways after temporal high pass filtering
    _spatiotemporalLPfilter(_amacrinCellsTempOutput_ON, _magnoXOutputON, 0);
    _spatiotemporalLPfilter(_amacrinCellsTempOutput_OFF, _magnoXOutputOFF, 0);

    // local adaptation of the ganglion cells to the local contrast of the moving contours
    _spatiotemporalLPfilter(_magnoXOutputON, _localProcessBufferON, 1);
    _localLuminanceAdaptation(_magnoXOutputON, _localProcessBufferON);

    _spatiotemporalLPfilter(_magnoXOutputOFF, _localProcessBufferOFF, 1);
    _localLuminanceAdaptation(_magnoXOutputOFF, _localProcessBufferOFF);

    add(_magnoXOutputON, _magnoXOutputOFF, _magnoYOutput);

    return _magnoYOutput;
}

///////////////////////////////////////
//////////// RetinaColor //////////////
///////////////////////////////////////

// define an array of ROI headers of input x
#define MAKE_OCLMAT_SLICES(x, n) \
    UMat x##_slices[n];\
    for(int _SLICE_INDEX_ = 0; _SLICE_INDEX_ < n; _SLICE_INDEX_ ++)\
    {\
        x##_slices[_SLICE_INDEX_] = x(getROI(_SLICE_INDEX_));\
    }

RetinaColor::RetinaColor(const unsigned int NBrows, const unsigned int NBcolumns, const int samplingMethod)
    : BasicRetinaFilter(NBrows, NBcolumns, 3),
      _RGBmosaic(NBrows * 3, NBcolumns, CV_32FC1),
      _tempMultiplexedFrame(NBrows, NBcolumns, CV_32FC1),
      _demultiplexedTempBuffer(NBrows * 3, NBcolumns, CV_32FC1),
      _demultiplexedColorFrame(NBrows * 3, NBcolumns, CV_32FC1),
      _chrominance(NBrows * 3, NBcolumns, CV_32FC1),
      _colorLocalDensity(NBrows * 3, NBcolumns, CV_32FC1),
      _imageGradient(NBrows * 3, NBcolumns, CV_32FC1)
{
    // link to parent buffers (let's recycle !)
    _luminance = _filterOutput;
    _multiplexedFrame = _localBuffer;

    _objectInit = false;
    _samplingMethod = samplingMethod;
    _saturateColors = false;
    _colorSaturationValue = 4.0;

    // set default spatio-temporal filter parameters
    setLPfilterParameters(0.0, 0.0, 1.5);
    setLPfilterParameters(0.0, 0.0, 10.5, 1);// for the low pass filter dedicated to contours energy extraction (demultiplexing process)
    setLPfilterParameters(0.f, 0.f, 0.9f, 2);

    // init default value on image Gradient
    _imageGradient = 0.57f;

    // init color sampling map
    _initColorSampling();

    // flush all buffers
    clearAllBuffers();
}

RetinaColor::~RetinaColor()
{

}

void RetinaColor::clearAllBuffers()
{
    BasicRetinaFilter::clearAllBuffers();
    _tempMultiplexedFrame = 0.f;
    _demultiplexedTempBuffer = 0.f;

    _demultiplexedColorFrame = 0.f;
    _chrominance = 0.f;
    _imageGradient = 0.57f;
}

void RetinaColor::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
    BasicRetinaFilter::clearAllBuffers();
    ensureSizeIsEnough(NBrows,     NBcolumns, CV_32FC1, _tempMultiplexedFrame);
    ensureSizeIsEnough(NBrows * 2, NBcolumns, CV_32FC1, _imageGradient);
    ensureSizeIsEnough(NBrows * 3, NBcolumns, CV_32FC1, _RGBmosaic);
    ensureSizeIsEnough(NBrows * 3, NBcolumns, CV_32FC1, _demultiplexedTempBuffer);
    ensureSizeIsEnough(NBrows * 3, NBcolumns, CV_32FC1, _demultiplexedColorFrame);
    ensureSizeIsEnough(NBrows * 3, NBcolumns, CV_32FC1, _chrominance);
    ensureSizeIsEnough(NBrows * 3, NBcolumns, CV_32FC1, _colorLocalDensity);

    // link to parent buffers (let's recycle !)
    _luminance = _filterOutput;
    _multiplexedFrame = _localBuffer;

    // init color sampling map
    _initColorSampling();

    // clean buffers
    clearAllBuffers();
}

static void inverseValue(UMat &input)
{
    int elements_per_row = static_cast<int>(input.step / input.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {input.cols, input.rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("inverseValue", retina_kernel_source);
    kernel.args(KernelArg::PtrReadWrite(input), input.cols, input.rows, elements_per_row);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void RetinaColor::_initColorSampling()
{
    CV_Assert(_samplingMethod == RETINA_COLOR_BAYER);
    _pR = _pB = 0.25;
    _pG = 0.5;
    // filling the mosaic buffer:
    _RGBmosaic = 0;
    Mat tmp_mat(_NBrows * 3, _NBcols, CV_32FC1);
    float * tmp_mat_ptr = tmp_mat.ptr<float>();
    tmp_mat.setTo(0);
    for (unsigned int index = 0 ; index < getNBpixels(); ++index)
    {
        tmp_mat_ptr[bayerSampleOffset(index)] = 1.0;
    }
    _RGBmosaic = tmp_mat.getUMat(ACCESS_RW);
    // computing photoreceptors local density
    MAKE_OCLMAT_SLICES(_RGBmosaic, 3);
    MAKE_OCLMAT_SLICES(_colorLocalDensity, 3);
    _colorLocalDensity.setTo(0);
    _spatiotemporalLPfilter(_RGBmosaic_slices[0], _colorLocalDensity_slices[0]);
    _spatiotemporalLPfilter(_RGBmosaic_slices[1], _colorLocalDensity_slices[1]);
    _spatiotemporalLPfilter(_RGBmosaic_slices[2], _colorLocalDensity_slices[2]);

    //_colorLocalDensity = UMat(_colorLocalDensity.size(), _colorLocalDensity.type(), 1.f) / _colorLocalDensity;
    inverseValue(_colorLocalDensity);

    _objectInit = true;
}

static void demultiplex(const UMat &input, UMat &ouput)
{
    int elements_per_row = static_cast<int>(input.step / input.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {input.cols, input.rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("runColorDemultiplexingBayer", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(input), KernelArg::PtrWriteOnly(ouput), input.cols, input.rows, elements_per_row);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

static void normalizePhotoDensity(
    const UMat &chroma,
    const UMat &colorDensity,
    const UMat &multiplex,
    UMat &ocl_luma,
    UMat &demultiplex,
    const float pG
)
{
    int elements_per_row = static_cast<int>(ocl_luma.step / ocl_luma.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {ocl_luma.cols, ocl_luma.rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("normalizePhotoDensity", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(chroma), KernelArg::PtrReadOnly(colorDensity), KernelArg::PtrReadOnly(multiplex),
                KernelArg::PtrWriteOnly(ocl_luma), KernelArg::PtrWriteOnly(demultiplex), ocl_luma.cols, ocl_luma.rows, elements_per_row, pG);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

static void substractResidual(
    UMat &colorDemultiplex,
    float pR,
    float pG,
    float pB
)
{
    int elements_per_row = static_cast<int>(colorDemultiplex.step / colorDemultiplex.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    int rows = colorDemultiplex.rows / 3, cols = colorDemultiplex.cols;
    size_t globalSize[] = {cols, rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("substractResidual", retina_kernel_source);
    kernel.args(KernelArg::PtrReadWrite(colorDemultiplex), cols, rows, elements_per_row, pR, pG, pB);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

static void demultiplexAssign(const UMat& input, const UMat& output)
{
    // only supports bayer
    int elements_per_row = static_cast<int>(input.step / input.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    int rows = input.rows / 3, cols = input.cols;
    size_t globalSize[] = {cols, rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("demultiplexAssign", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(input), KernelArg::PtrWriteOnly(output), cols, rows, elements_per_row);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void RetinaColor::runColorDemultiplexing(
    const UMat &ocl_multiplexed_input,
    const bool adaptiveFiltering,
    const float maxInputValue
)
{
    MAKE_OCLMAT_SLICES(_demultiplexedTempBuffer, 3);
    MAKE_OCLMAT_SLICES(_chrominance, 3);
    MAKE_OCLMAT_SLICES(_RGBmosaic, 3);
    MAKE_OCLMAT_SLICES(_demultiplexedColorFrame, 3);
    MAKE_OCLMAT_SLICES(_colorLocalDensity, 3);

    _demultiplexedTempBuffer.setTo(0);
    demultiplex(ocl_multiplexed_input, _demultiplexedTempBuffer);

    // interpolate the demultiplexed frame depending on the color sampling method
    if (!adaptiveFiltering)
    {
        CV_Assert(adaptiveFiltering == false);
    }

    _spatiotemporalLPfilter(_demultiplexedTempBuffer_slices[0], _chrominance_slices[0]);
    _spatiotemporalLPfilter(_demultiplexedTempBuffer_slices[1], _chrominance_slices[1]);
    _spatiotemporalLPfilter(_demultiplexedTempBuffer_slices[2], _chrominance_slices[2]);

    if (!adaptiveFiltering)// compute the gradient on the luminance
    {
        // TODO: implement me!
        CV_Assert(adaptiveFiltering == false);
    }
    else
    {
        normalizePhotoDensity(_chrominance, _colorLocalDensity, ocl_multiplexed_input, _luminance, _demultiplexedTempBuffer, _pG);
        // compute the gradient of the luminance
        _computeGradient(_luminance, _imageGradient);

        _adaptiveSpatialLPfilter(_RGBmosaic_slices[0], _imageGradient, _chrominance_slices[0]);
        _adaptiveSpatialLPfilter(_RGBmosaic_slices[1], _imageGradient, _chrominance_slices[1]);
        _adaptiveSpatialLPfilter(_RGBmosaic_slices[2], _imageGradient, _chrominance_slices[2]);

        _adaptiveSpatialLPfilter(_demultiplexedTempBuffer_slices[0], _imageGradient, _demultiplexedColorFrame_slices[0]);
        _adaptiveSpatialLPfilter(_demultiplexedTempBuffer_slices[1], _imageGradient, _demultiplexedColorFrame_slices[1]);
        _adaptiveSpatialLPfilter(_demultiplexedTempBuffer_slices[2], _imageGradient, _demultiplexedColorFrame_slices[2]);

        divide(_demultiplexedColorFrame, _chrominance, _demultiplexedColorFrame);
        substractResidual(_demultiplexedColorFrame, _pR, _pG, _pB);
        runColorMultiplexing(_demultiplexedColorFrame, _tempMultiplexedFrame);

        _demultiplexedTempBuffer.setTo(0);
        subtract(ocl_multiplexed_input, _tempMultiplexedFrame, _luminance);
        demultiplexAssign(_demultiplexedColorFrame, _demultiplexedTempBuffer);

        for(int i = 0; i < 3; i ++)
        {
            _spatiotemporalLPfilter(_demultiplexedTempBuffer_slices[i], _demultiplexedTempBuffer_slices[i]);
            //_demultiplexedColorFrame_slices[i] = _demultiplexedTempBuffer_slices[i] * _colorLocalDensity_slices[i] + _luminance;
            multiply(_demultiplexedTempBuffer_slices[i], _colorLocalDensity_slices[i], _demultiplexedColorFrame_slices[i]);
            add(_demultiplexedColorFrame_slices[i], _luminance, _demultiplexedColorFrame_slices[i]);
        }
    }
    // eliminate saturated colors by simple clipping values to the input range
    clipRGBOutput_0_maxInputValue(_demultiplexedColorFrame, maxInputValue);

    if (_saturateColors)
    {
        ocl::normalizeGrayOutputCentredSigmoide(128, maxInputValue, _demultiplexedColorFrame, _demultiplexedColorFrame);
    }
}
void RetinaColor::runColorMultiplexing(const UMat &demultiplexedInputFrame, UMat &multiplexedFrame)
{
    int elements_per_row = static_cast<int>(multiplexedFrame.step / multiplexedFrame.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {multiplexedFrame.cols, multiplexedFrame.rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("runColorMultiplexingBayer", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(demultiplexedInputFrame), KernelArg::PtrWriteOnly(multiplexedFrame), multiplexedFrame.cols, multiplexedFrame.rows, elements_per_row);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void RetinaColor::clipRGBOutput_0_maxInputValue(UMat &inputOutputBuffer, const float maxInputValue)
{
    // the kernel is equivalent to:
    //ocl::threshold(inputOutputBuffer, inputOutputBuffer, maxInputValue, maxInputValue, THRESH_TRUNC);
    //ocl::threshold(inputOutputBuffer, inputOutputBuffer, 0, 0, THRESH_TOZERO);
    int elements_per_row = static_cast<int>(inputOutputBuffer.step / inputOutputBuffer.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, inputOutputBuffer.rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("clipRGBOutput_0_maxInputValue", retina_kernel_source);
    kernel.args(KernelArg::PtrReadWrite(inputOutputBuffer), _NBcols, inputOutputBuffer.rows, elements_per_row, maxInputValue);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void RetinaColor::_adaptiveSpatialLPfilter(const UMat &inputFrame, const UMat &gradient, UMat &outputFrame)
{
    /**********/
    _gain = (1 - 0.57f) * (1 - 0.57f) * (1 - 0.06f) * (1 - 0.06f);

    // launch the serie of 1D directional filters in order to compute the 2D low pass filter
    // -> horizontal filters work with the first layer of imageGradient
    _adaptiveHorizontalCausalFilter_addInput(inputFrame, gradient, outputFrame);
    _horizontalAnticausalFilter_Irregular(outputFrame, gradient);
    // -> horizontal filters work with the second layer of imageGradient
    _verticalCausalFilter_Irregular(outputFrame, gradient(getROI(1)));
    _adaptiveVerticalAnticausalFilter_multGain(gradient, outputFrame);
}

void RetinaColor::_adaptiveHorizontalCausalFilter_addInput(const UMat &inputFrame, const UMat &gradient, UMat &outputFrame)
{
    int elements_per_row = static_cast<int>(inputFrame.step / inputFrame.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBrows, 1, 1};
    size_t localSize[] = { 256, 1, 1 };

    Kernel kernel("adaptiveHorizontalCausalFilter_addInput", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(inputFrame), KernelArg::PtrReadOnly(gradient), KernelArg::PtrWriteOnly(outputFrame),
                _NBcols, _NBrows, elements_per_row, inputFrame.offset, gradient.offset, outputFrame.offset);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

void RetinaColor::_adaptiveVerticalAnticausalFilter_multGain(const UMat &gradient, UMat &outputFrame)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, 1, 1};
    size_t localSize[]  = {256, 1, 1};

    int gradOffset = gradient.offset + static_cast<int>(gradient.step * _NBrows);

    Kernel kernel("adaptiveVerticalAnticausalFilter_multGain", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(gradient), KernelArg::PtrReadWrite(outputFrame), _NBcols, _NBrows, elements_per_row, gradOffset, outputFrame.offset, _gain);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}
void RetinaColor::_computeGradient(const UMat &luminance, UMat &gradient)
{
    int elements_per_row = static_cast<int>(luminance.step / luminance.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, _NBrows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("computeGradient", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(luminance), KernelArg::PtrWriteOnly(gradient), _NBcols, _NBrows, elements_per_row);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}

///////////////////////////////////////
//////////// RetinaFilter /////////////
///////////////////////////////////////
RetinaFilter::RetinaFilter(const unsigned int sizeRows, const unsigned int sizeColumns, const bool colorMode, const int samplingMethod, const bool useRetinaLogSampling, const double, const double)
    :
    _photoreceptorsPrefilter(sizeRows, sizeColumns, 4),
    _ParvoRetinaFilter(sizeRows, sizeColumns),
    _MagnoRetinaFilter(sizeRows, sizeColumns),
    _colorEngine(sizeRows, sizeColumns, samplingMethod)
{
    CV_Assert(!useRetinaLogSampling);

    // set default processing activities
    _useParvoOutput = true;
    _useMagnoOutput = true;

    _useColorMode = colorMode;

    // set default parameters
    setGlobalParameters();

    // stability controls values init
    _setInitPeriodCount();
    _globalTemporalConstant = 25;

    // reset all buffers
    clearAllBuffers();
}

RetinaFilter::~RetinaFilter()
{
}

void RetinaFilter::clearAllBuffers()
{
    _photoreceptorsPrefilter.clearAllBuffers();
    _ParvoRetinaFilter.clearAllBuffers();
    _MagnoRetinaFilter.clearAllBuffers();
    _colorEngine.clearAllBuffers();
    // stability controls value init
    _setInitPeriodCount();
}

void RetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
    unsigned int rows = NBrows, cols = NBcolumns;

    // resize optionnal member and adjust other modules size if required
    _photoreceptorsPrefilter.resize(rows, cols);
    _ParvoRetinaFilter.resize(rows, cols);
    _MagnoRetinaFilter.resize(rows, cols);
    _colorEngine.resize(rows, cols);

    // clean buffers
    clearAllBuffers();

}

void RetinaFilter::_setInitPeriodCount()
{
    // find out the maximum temporal constant value and apply a security factor
    // false value (obviously too long) but appropriate for simple use
    _globalTemporalConstant = (unsigned int)(_ParvoRetinaFilter.getPhotoreceptorsTemporalConstant() + _ParvoRetinaFilter.getHcellsTemporalConstant() + _MagnoRetinaFilter.getTemporalConstant());
    // reset frame counter
    _ellapsedFramesSinceLastReset = 0;
}

void RetinaFilter::setGlobalParameters(const float OPLspatialResponse1, const float OPLtemporalresponse1, const float OPLassymetryGain, const float OPLspatialResponse2, const float OPLtemporalresponse2, const float LPfilterSpatialResponse, const float LPfilterGain, const float LPfilterTemporalresponse, const float MovingContoursExtractorCoefficient, const bool normalizeParvoOutput_0_maxOutputValue, const bool normalizeMagnoOutput_0_maxOutputValue, const float maxOutputValue, const float maxInputValue, const float meanValue)
{
    _normalizeParvoOutput_0_maxOutputValue = normalizeParvoOutput_0_maxOutputValue;
    _normalizeMagnoOutput_0_maxOutputValue = normalizeMagnoOutput_0_maxOutputValue;
    _maxOutputValue = maxOutputValue;
    _photoreceptorsPrefilter.setV0CompressionParameter(0.9f, maxInputValue, meanValue);
    _photoreceptorsPrefilter.setLPfilterParameters(0, 0, 10, 3); // keeps low pass filter with low cut frequency in memory (usefull for the tone mapping function)
    _ParvoRetinaFilter.setOPLandParvoFiltersParameters(0, OPLtemporalresponse1, OPLspatialResponse1, OPLassymetryGain, OPLtemporalresponse2, OPLspatialResponse2);
    _ParvoRetinaFilter.setV0CompressionParameter(0.9f, maxInputValue, meanValue);
    _MagnoRetinaFilter.setCoefficientsTable(LPfilterGain, LPfilterTemporalresponse, LPfilterSpatialResponse, MovingContoursExtractorCoefficient, 0, 2.0f * LPfilterSpatialResponse);
    _MagnoRetinaFilter.setV0CompressionParameter(0.7f, maxInputValue, meanValue);

    // stability controls value init
    _setInitPeriodCount();
}

bool RetinaFilter::checkInput(const UMat &input, const bool)
{
    BasicRetinaFilter *inputTarget = &_photoreceptorsPrefilter;

    bool test = (input.rows == static_cast<int>(inputTarget->getNBrows())
                 || input.rows == static_cast<int>(inputTarget->getNBrows()) * 3
                 || input.rows == static_cast<int>(inputTarget->getNBrows()) * 4)
                && input.cols == static_cast<int>(inputTarget->getNBcolumns());
    if (!test)
    {
        std::cerr << "RetinaFilter::checkInput: input buffer does not match retina buffer size, conversion aborted" << std::endl;
        return false;
    }

    return true;
}

// main function that runs the filter for a given input frame
bool RetinaFilter::runFilter(const UMat &imageInput, const bool useAdaptiveFiltering, const bool processRetinaParvoMagnoMapping, const bool useColorMode, const bool inputIsColorMultiplexed)
{
    // preliminary check
    bool processSuccess = true;
    if (!checkInput(imageInput, useColorMode))
    {
        return false;
    }

    // run the color multiplexing if needed and compute each suub filter of the retina:
    // -> local adaptation
    // -> contours OPL extraction
    // -> moving contours extraction

    // stability controls value update
    ++_ellapsedFramesSinceLastReset;

    _useColorMode = useColorMode;

    UMat selectedPhotoreceptorsLocalAdaptationInput = imageInput;
    UMat selectedPhotoreceptorsColorInput = imageInput;

    //********** Following is input data specific photoreceptors processing
    if (useColorMode && (!inputIsColorMultiplexed)) // not multiplexed color input case
    {
        _colorEngine.runColorMultiplexing(selectedPhotoreceptorsColorInput);
        selectedPhotoreceptorsLocalAdaptationInput = _colorEngine.getMultiplexedFrame();
    }
    //********** Following is generic Retina processing

    // photoreceptors local adaptation
    _photoreceptorsPrefilter.runFilter_LocalAdapdation(selectedPhotoreceptorsLocalAdaptationInput, _ParvoRetinaFilter.getHorizontalCellsOutput());

    // run parvo filter
    _ParvoRetinaFilter.runFilter(_photoreceptorsPrefilter.getOutput(), _useParvoOutput);

    if (_useParvoOutput)
    {
        _ParvoRetinaFilter.normalizeGrayOutputCentredSigmoide(); // models the saturation of the cells, usefull for visualisation of the ON-OFF Parvo Output, Bipolar cells outputs do not change !!!
        _ParvoRetinaFilter.centerReductImageLuminance(); // best for further spectrum analysis

        if (_normalizeParvoOutput_0_maxOutputValue)
        {
            _ParvoRetinaFilter.normalizeGrayOutput_0_maxOutputValue(_maxOutputValue);
        }
    }

    if (_useParvoOutput && _useMagnoOutput)
    {
        _MagnoRetinaFilter.runFilter(_ParvoRetinaFilter.getBipolarCellsON(), _ParvoRetinaFilter.getBipolarCellsOFF());
        if (_normalizeMagnoOutput_0_maxOutputValue)
        {
            _MagnoRetinaFilter.normalizeGrayOutput_0_maxOutputValue(_maxOutputValue);
        }
        _MagnoRetinaFilter.normalizeGrayOutputNearZeroCentreredSigmoide();
    }

    if (_useParvoOutput && _useMagnoOutput && processRetinaParvoMagnoMapping)
    {
        _processRetinaParvoMagnoMapping();
        if (_useColorMode)
        {
            _colorEngine.runColorDemultiplexing(_retinaParvoMagnoMappedFrame, useAdaptiveFiltering, _maxOutputValue);
        }
        return processSuccess;
    }

    if (_useParvoOutput && _useColorMode)
    {
        _colorEngine.runColorDemultiplexing(_ParvoRetinaFilter.getOutput(), useAdaptiveFiltering, _maxOutputValue);
    }
    return processSuccess;
}

const UMat &RetinaFilter::getContours()
{
    if (_useColorMode)
    {
        return _colorEngine.getLuminance();
    }
    else
    {
        return _ParvoRetinaFilter.getOutput();
    }
}
void RetinaFilter::_processRetinaParvoMagnoMapping()
{
    UMat parvo = _ParvoRetinaFilter.getOutput();
    UMat magno = _MagnoRetinaFilter.getOutput();

    int halfRows = parvo.rows / 2;
    int halfCols = parvo.cols / 2;
    float minDistance = MIN(halfRows, halfCols) * 0.7f;

    int elements_per_row = static_cast<int>(parvo.step / parvo.elemSize());

    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {parvo.cols, parvo.rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    Kernel kernel("processRetinaParvoMagnoMapping", retina_kernel_source);
    kernel.args(KernelArg::PtrReadOnly(parvo), KernelArg::PtrReadOnly(magno), parvo.cols, parvo.rows, halfCols, halfRows, elements_per_row, minDistance);
    kernel.run(sizeOfArray(globalSize), globalSize, localSize, false);
}
}  /* namespace ocl */

Ptr<Retina> createRetina_OCL(Size getInputSize){ return makePtr<ocl::RetinaOCLImpl>(getInputSize); }
Ptr<Retina> createRetina_OCL(Size inputSize, const bool colorMode, int colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    return makePtr<ocl::RetinaOCLImpl>(inputSize, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);
}

}  /* namespace bioinspired */
}  /* namespace cv */

#endif /* #ifdef HAVE_OPENCL */
