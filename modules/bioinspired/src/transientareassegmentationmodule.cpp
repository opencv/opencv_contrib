/*#******************************************************************************
 ** IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 **
 ** By downloading, copying, installing or using the software you agree to this license.
 ** If you do not agree to this license, do not download, install,
 ** copy or use the software.
 **
 **
 ** bioinspired : interfaces allowing OpenCV users to integrate Human Vision System models.
 ** TransientAreasSegmentationModule Use: extract areas that present spatio-temporal changes.
 ** => It should be used at the output of the cv::bioinspired::Retina::getMagnoRAW() output that enhances spatio-temporal changes
 **
 ** Maintainers : Listic lab (code author current affiliation & applications)
 **
 **  Creation - enhancement process 2007-2015
 **      Author: Alexandre Benoit (benoit.alexandre.vision@gmail.com), LISTIC lab, Annecy le vieux, France
 **
 ** Theses algorithm have been developped by Alexandre BENOIT since his thesis with Alice Caplier at Gipsa-Lab (www.gipsa-lab.inpg.fr) and the research he pursues at LISTIC Lab (www.listic.univ-savoie.fr).
 ** Refer to the following research paper for more information:
 ** Strat, S.T.; Benoit, A.; Lambert, P., "Retina enhanced bag of words descriptors for video classification," Signal Processing Conference (EUSIPCO), 2014 Proceedings of the 22nd European , vol., no., pp.1307,1311, 1-5 Sept. 2014 (http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6952461&isnumber=6951911)
 ** Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
 ** This work have been carried out thanks to Jeanny Herault who's research and great discussions are the basis of all this work, please take a look at his book:
 ** Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.
 **
 **
 **                          License Agreement
 **               For Open Source Computer Vision Library
 **
 ** Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 ** Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
 **
 **               For Human Visual System tools (bioinspired)
 ** Copyright (C) 2007-2015, LISTIC Lab, Annecy le Vieux and GIPSA Lab, Grenoble, France, all rights reserved.
 **
 ** Third party copyrights are property of their respective owners.
 **
 ** Redistribution and use in source and binary forms, with or without modification,
 ** are permitted provided that the following conditions are met:
 **
 ** * Redistributions of source code must retain the above copyright notice,
 **    this list of conditions and the following disclaimer.
 **
 ** * Redistributions in binary form must reproduce the above copyright notice,
 **    this list of conditions and the following disclaimer in the documentation
 **    and/or other materials provided with the distribution.
 **
 ** * The name of the copyright holders may not be used to endorse or promote products
 **    derived from this software without specific prior written permission.
 **
 ** This software is provided by the copyright holders and contributors "as is" and
 ** any express or implied warranties, including, but not limited to, the implied
 ** warranties of merchantability and fitness for a particular purpose are disclaimed.
 ** In no event shall the Intel Corporation or contributors be liable for any direct,
 ** indirect, incidental, special, exemplary, or consequential damages
 ** (including, but not limited to, procurement of substitute goods or services;
 ** loss of use, data, or profits; or business interruption) however caused
 ** and on any theory of liability, whether in contract, strict liability,
 ** or tort (including negligence or otherwise) arising in any way out of
 ** the use of this software, even if advised of the possibility of such damage.
 *******************************************************************************/


/**
 * @class TransientAreasSegmentationModule
 * @brief class which provides a transient/moving areas segmentation module
 * -> perform a locally adapted segmentation by using the retina magno output data
 * @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com
 * Release date 2007-2014
 * Based on Alexandre BENOIT thesis: "Le système visuel humain au secours de la vision par ordinateur"
 * -> 3 spatio temporal filters are used:
 * a first one which filters the noise and local variations of the input motion energy
 * a second (more powerfull low pass spatial filter) which gives the neighborhood motion energy
 * a third that measures the global motion context on a wide area
 * -> the segmentation consists in the comparison of these filter outputs, if the local motion energy is higher to the neighborhood motion energy, then the area is considered as moving and is segmented
 */

#include "precomp.hpp"
#include "basicretinafilter.hpp"

#include <sstream>

#define _SEGMENTATIONDEBUG //define SEGMENTATIONDEBUG to access more data/methods

namespace cv
{
namespace bioinspired
{

class TransientAreasSegmentationModuleImpl : protected BasicRetinaFilter
{
public:

    /**
     * constructor
     * @param Size : size of the images input to segment (output will be the same size)
     */
    TransientAreasSegmentationModuleImpl(const Size inputSize);

    /**
     * standard destructor
     */
    virtual ~TransientAreasSegmentationModuleImpl();

    /**
     * @return the size of the manage input and output images
     */
    Size getSize(){return cv::Size(getNBcolumns(), getNBrows());}

    /**
     * try to open an XML segmentation parameters file to adjust current segmentation instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param retinaParameterFile : the parameters filename
     * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
    void setup(String segmentationParameterFile="", const bool applyDefaultSetupOnFailure=true);

    /**
     * try to open an XML segmentation parameters file to adjust current segmentation instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param fs : the open Filestorage which contains segmentation parameters
     * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
    void setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure=true);

    /**
     * try to open an XML segmentation parameters file to adjust current segmentation instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param newParameters : a parameters structures updated with the new target configuration
     * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
    void setup(SegmentationParameters newParameters);

    /**
     * @return the current parameters setup
     */
    SegmentationParameters getParameters();

    /**
     * parameters setup display method
     * @return a string which contains formatted parameters information
     */
    String printSetup();

    /**
     * write xml/yml formated parameters information
     * @rparam fs : the filename of the xml file that will be open and writen with formatted parameters information
     */
    virtual void write( String fs ) const;

    /**
     * write xml/yml formated parameters information
     * @param fs : a cv::Filestorage object ready to be filled
         */
    virtual void write( cv::FileStorage& fs ) const;


    /**
     * main processing method, get result using methods getSegmentationPicture()
     * @param inputToSegment : the image to process, it must match the instance buffer size !
     * @param channelIndex : the channel to process in case of multichannel images
     */
    void run(InputArray inputToSegment, const int channelIndex=0);

    /**
     * access function
     * return the last segmentation result: a boolean picture which is resampled between 0 and 255 for a display purpose
     */
    void getSegmentationPicture(OutputArray transientAreas);

    /**
     * cleans all the buffers of the instance
     */
    void clearAllBuffers();

protected:

    /**
     * main processing method
     * @param inputToSegment : the image to process as a valarray buffer it must match the instance buffer size !
     * @param channelIndex : the channel to process in case of multichannel images
     */
    void _run(const std::valarray<float> &inputToSegment, const int channelIndex=0);

    /**
     * access function
     * @return the local motion energy level picture (experimental, not usefull)
     */
    inline const std::valarray<float> &getLocalMotionPicture() const {return _localMotion;}

    /**
     * access function
     * @return the neighborhood motion energy level picture (experimental, not usefull)
     */
    inline const std::valarray<float> &getNeighborhoodMotionPicture() const {return _neighborhoodMotion;}

    /**
     * access function
     * @return the motion energy context level picture (experimental, not usefull)
     */
    inline const std::valarray<float> &getMotionContextPicture() const {return _contextMotionEnergy;}

    cv::bioinspired::SegmentationParameters _segmentationParameters;
    // template buffers and related acess pointers
    std::valarray<float> _inputToSegment;
    std::valarray<float> _contextMotionEnergy;
    std::valarray<bool> _segmentedAreas;

    // pointers to base class buffers
    std::valarray<float> &_localMotion;
    std::valarray<float> &_neighborhoodMotion;
    unsigned int _numberOfSegmentedObjects;

    cv::Mat _conversionBuffer;
    cv::Mat _segmentedPicture;

    // Buffer conversion utilities
    void _convertValarrayBuffer2cvMat(const std::valarray<bool> &grayMatrixToConvert, const unsigned int nbRows, const unsigned int nbColumns, OutputArray outBuffer);
    bool _convertCvMat2ValarrayBuffer(InputArray inputMat, std::valarray<float> &outputValarrayMatrix);

    const TransientAreasSegmentationModuleImpl & operator = (const TransientAreasSegmentationModuleImpl &);
};

class TransientAreasSegmentationModuleImpl_: public  TransientAreasSegmentationModule
{
public:
    TransientAreasSegmentationModuleImpl_(const Size size):_segmTool(size){}
    inline virtual Size getSize() CV_OVERRIDE { return _segmTool.getSize(); }
    inline virtual void write( cv::FileStorage& fs ) const CV_OVERRIDE { _segmTool.write(fs); }
    inline virtual void setup(String segmentationParameterFile, const bool applyDefaultSetupOnFailure) CV_OVERRIDE { _segmTool.setup(segmentationParameterFile, applyDefaultSetupOnFailure); }
    inline virtual void setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure) CV_OVERRIDE { _segmTool.setup(fs, applyDefaultSetupOnFailure); }
    inline virtual void setup(SegmentationParameters newParameters) CV_OVERRIDE { _segmTool.setup(newParameters); }
    inline virtual String printSetup() CV_OVERRIDE { return _segmTool.printSetup(); }
    inline virtual SegmentationParameters getParameters() CV_OVERRIDE { return _segmTool.getParameters(); }
    inline virtual void write( String fs ) const CV_OVERRIDE { _segmTool.write(fs); }
    inline virtual void run(InputArray inputToSegment, const int channelIndex) CV_OVERRIDE { _segmTool.run(inputToSegment, channelIndex); }
    inline virtual void getSegmentationPicture(OutputArray transientAreas) CV_OVERRIDE { return _segmTool.getSegmentationPicture(transientAreas); }
    inline virtual void clearAllBuffers() CV_OVERRIDE { _segmTool.clearAllBuffers(); }

private:
    TransientAreasSegmentationModuleImpl _segmTool;
};

/**
* allocator
* @param Size : size of the images input to segment (output will be the same size)
*/
Ptr<TransientAreasSegmentationModule> TransientAreasSegmentationModule::create(Size inputSize){
    return makePtr<TransientAreasSegmentationModuleImpl_>(inputSize);
}

// Constructor and destructors
TransientAreasSegmentationModuleImpl::TransientAreasSegmentationModuleImpl(const Size size)
:BasicRetinaFilter(size.height, size.width, 3),
 // allocate the output of the class
 _inputToSegment(size.height*size.width),
 _contextMotionEnergy(size.height*size.width),
 _segmentedAreas(size.height*size.width),
 // set the pointer to the 2 frame buffer to the correct adress:
 // -> the first low pass filter buffer will be _localBuffer
 // -> the second will be _filterOutput;
 _localMotion(_localBuffer),
 _neighborhoodMotion(_filterOutput)
{

    // default parameters setup
    setup(_segmentationParameters);
    //clean before running
    clearAllBuffers();
}

TransientAreasSegmentationModuleImpl::~TransientAreasSegmentationModuleImpl()
{
}


void TransientAreasSegmentationModuleImpl::clearAllBuffers()
{
    // flush parent buffers
    bioinspired::BasicRetinaFilter::clearAllBuffers();
    // flush instance buffers
    _contextMotionEnergy=0;
    _segmentedAreas=0;
}

SegmentationParameters TransientAreasSegmentationModuleImpl::getParameters()
{
    return _segmentationParameters;
}

// setup from XML file
void TransientAreasSegmentationModuleImpl::setup(String segmentationParameterFile, const bool applyDefaultSetupOnFailure)
{
    try
    {
        // opening retinaParameterFile in read mode
        cv::FileStorage fs(segmentationParameterFile, cv::FileStorage::READ);
        setup(fs, applyDefaultSetupOnFailure);
    }catch(const cv::Exception &e)
    {
        printf("Retina::setup: wrong/unappropriate xml parameter file : error report :`n=>%s\n", e.what());
        if (applyDefaultSetupOnFailure)
        {
           printf("Retina::setup: resetting retina with default parameters\n");
            cv::bioinspired::SegmentationParameters defaults;
            setup(defaults);
        }
        else
        {
            printf("=> keeping current parameters");
        }
    }
}

// setup from cv::Filestorage object
void TransientAreasSegmentationModuleImpl::setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure)
{
    try
    {
        // read parameters file if it exists or apply default setup if asked for
        if (!fs.isOpened())
        {
            std::cout<<"Retina::setup: provided parameters file could not be open... skeeping configuration"<<std::endl;
            return;
            // implicit else case : retinaParameterFile could be open (it exists at least)
        }
        // OPL and Parvo init first... update at the same time the parameters structure and the retina core
        cv::FileNode rootFn = fs.root(), currFn=rootFn["SegmentationModuleSetup"];
        currFn["thresholdON"]>>_segmentationParameters.thresholdON;
        currFn["thresholdOFF"]>>_segmentationParameters.thresholdOFF;
        currFn["localEnergy_temporalConstant"]>>_segmentationParameters.localEnergy_temporalConstant;
        currFn["localEnergy_spatialConstant"]>>_segmentationParameters.localEnergy_spatialConstant;
        currFn["neighborhoodEnergy_temporalConstant"]>>_segmentationParameters.neighborhoodEnergy_temporalConstant;
        currFn["neighborhoodEnergy_spatialConstant"]>>_segmentationParameters.neighborhoodEnergy_spatialConstant;
        currFn["contextEnergy_temporalConstant"]>>_segmentationParameters.contextEnergy_temporalConstant;
        currFn["contextEnergy_spatialConstant"]>>_segmentationParameters.contextEnergy_spatialConstant;
        setup(_segmentationParameters);

    }catch(const cv::Exception &e)
    {
        std::cout<<"Retina::setup: resetting retina with default parameters"<<std::endl;
        if (applyDefaultSetupOnFailure)
        {
            cv::bioinspired::SegmentationParameters defaults;
            setup(defaults);
        }
        std::cout<<"SegmentationModule::setup: wrong/unappropriate xml parameter file : error report :`n=>"<<e.what()<<std::endl;
        std::cout<<"=> keeping current parameters"<<std::endl;
    }
}

// setup parameters for the 2 filters that allow the segmentation
void TransientAreasSegmentationModuleImpl::setup(cv::bioinspired::SegmentationParameters newParameters)
{

    // copy structure contents
    _segmentationParameters = newParameters;
    // apply setup
    // init local motion energy extraction low pass filter
    BasicRetinaFilter::setLPfilterParameters(0, newParameters.localEnergy_temporalConstant, newParameters.localEnergy_spatialConstant);
    // init neighbohood motion energy extraction low pass filter
    BasicRetinaFilter::setLPfilterParameters(0, newParameters.neighborhoodEnergy_temporalConstant, newParameters.neighborhoodEnergy_spatialConstant, 1);

    // init large area low pass filter
    BasicRetinaFilter::setLPfilterParameters(0, newParameters.contextEnergy_temporalConstant, newParameters.contextEnergy_spatialConstant, 2);

}

String TransientAreasSegmentationModuleImpl::printSetup()
{
    std::stringstream outmessage;

    outmessage<<"Current segmentation instance setup :"
            <<"\n\t thresholdON : " << _segmentationParameters.thresholdON
            <<"\n\t thresholdOFF : " << _segmentationParameters.thresholdOFF
            <<"\n\t localEnergy_temporalConstant : " << _segmentationParameters.localEnergy_temporalConstant
            <<"\n\t localEnergy_spatialConstant : " << _segmentationParameters.localEnergy_spatialConstant
            <<"\n\t neighborhoodEnergy_temporalConstant : " << _segmentationParameters.neighborhoodEnergy_temporalConstant
            <<"\n\t neighborhoodEnergy_spatialConstant : " << _segmentationParameters.neighborhoodEnergy_spatialConstant
            <<"\n\t contextEnergy_temporalConstant : " << _segmentationParameters.contextEnergy_temporalConstant
            <<"\n\t contextEnergy_spatialConstant : " << _segmentationParameters.contextEnergy_spatialConstant;

    return outmessage.str().c_str();
}

void TransientAreasSegmentationModuleImpl::write( String fs ) const
{
    cv::FileStorage parametersSaveFile(fs, cv::FileStorage::WRITE );
    write(parametersSaveFile);
}

void TransientAreasSegmentationModuleImpl::write( cv::FileStorage& fs ) const
{
    if (!fs.isOpened())
        return; // basic error case
    fs <<"SegmentationModuleSetup"<<"{";
    fs <<"thresholdON" << _segmentationParameters.thresholdON;
    fs <<"thresholdOFF" << _segmentationParameters.thresholdOFF;
    fs <<"localEnergy_temporalConstant" << _segmentationParameters.localEnergy_temporalConstant;
    fs <<"localEnergy_spatialConstant" << _segmentationParameters.localEnergy_spatialConstant;
    fs <<"neighborhoodEnergy_temporalConstant" << _segmentationParameters.neighborhoodEnergy_temporalConstant;
    fs <<"neighborhoodEnergy_spatialConstant" << _segmentationParameters.neighborhoodEnergy_spatialConstant;
    fs <<"contextEnergy_temporalConstant" << _segmentationParameters.contextEnergy_temporalConstant;
    fs <<"contextEnergy_spatialConstant" << _segmentationParameters.contextEnergy_spatialConstant;
    fs <<"}";
}

void TransientAreasSegmentationModuleImpl::run(InputArray inputToProcess, const int channelIndex)
{
    cv::Mat inputToSegment=inputToProcess.getMat();
    // preliminary basic error check
    if ( (inputToSegment.rows*inputToSegment.cols) != (int)_inputToSegment.size())
    {
    	std::stringstream errorMsg;
    	errorMsg<<"Input matrix size does not match instance buffers setup !"
    			<<"\n\t Input size is : "<<inputToSegment.rows*inputToSegment.cols
    			<<"\n\t v.s. internalBuffer size is : "<<  _inputToSegment.size();
    	throw cv::Exception(-1, errorMsg.str().c_str(), "SegmentationModule::run", "SegmentationModule.cpp", 0);
    }
    if (channelIndex >= inputToSegment.channels())
    {
    	std::stringstream errorMsg;
    	errorMsg<<"Cannot access channel index "<<channelIndex<<" on the input matrix with channels quantity = "<<inputToSegment.channels();
    	throw cv::Exception(-1, errorMsg.str().c_str(), "SegmentationModule::run", "SegmentationModule.cpp", 0);
    }

    // create a cv::Mat header for the input valarray
    // convert to float AND fill the valarray buffer
    typedef float T; // define here the target pixel format, here, float
    const int dsttype = cv::DataType<T>::depth; // output buffer is float format
    cv::Mat dst(inputToSegment.size(), dsttype, &_inputToSegment[0]);
    inputToSegment.convertTo(dst, dsttype);
    //cv::imshow("Mask",dst);
    //cv::waitKey();
    // call the low level method
    _run(_inputToSegment, channelIndex);
}

void TransientAreasSegmentationModuleImpl::_run(const std::valarray<float> &inputToSegment, const int channelIndex)
{
#ifdef SEGMENTATIONDEBUG
    std::cout<<"Input length vs internal buffers length = "<<inputToSegment.size()<<", "<<_localMotion.size()<<std::endl;
#endif
    // preliminary basic error check
    // FIXME validate basic tests
    //if (inputToSegment.size() != _localMotion.size())
    //    throw cv::Exception(-1, "Input matrix size does not match instance buffers setup !", "SegmentationModule::run", "SegmentationModule.cpp", 0);

    // first square the input in order to increase the signal to noise ratio
    // get motion local energy
    _squaringSpatiotemporalLPfilter(&const_cast<std::valarray<float>&>(inputToSegment)[channelIndex*getNBpixels()], &_localMotion[0]);

    // second low pass filter: access to the neighborhood motion energy
    _spatiotemporalLPfilter(&_localMotion[0], &_neighborhoodMotion[0], 1);

    // third low pass filter: access to the background motion energy
    _spatiotemporalLPfilter(&_localMotion[0], &_contextMotionEnergy[0], 2);

    // compute the ON and OFF ways (positive and negative values of the difference of the two filterings)
    float*localMotionPTR=&_localMotion[0], *neighborhoodMotionPTR=&_neighborhoodMotion[0], *contextMotionPTR=&_contextMotionEnergy[0];

    // float meanEnergy=LPfilter2.sum()/(float)_LPfilter2.size();
    bool *segmentationPicturePTR= &_segmentedAreas[0];
    for (unsigned int index=0; index<_filterOutput.getNBpixels() ; ++index, ++segmentationPicturePTR, ++localMotionPTR, ++neighborhoodMotionPTR, contextMotionPTR++)
    {
        float generalMotionContextDecision=*neighborhoodMotionPTR-*contextMotionPTR;

        if (generalMotionContextDecision>0) // local maximum should be detected in this case
        {
            /* apply segmentation on local motion superior to its neighborhood
             * => to segment objects moving faster than their neighborhood
             */
            if (generalMotionContextDecision>_segmentationParameters.thresholdON)// && meanEnergy*1.1<*neighborhoodMotionPTR)
            {
                *segmentationPicturePTR=((*localMotionPTR-*neighborhoodMotionPTR)>_segmentationParameters.thresholdON);
            }
            else
                *segmentationPicturePTR=false;
        }
#ifdef USE_LOCALMINIMUMS
        else  // local minimum should be detected in this case
        {
            /* apply segmentation for non moving objects
             * only if the wide area around motion energy is high
             * => to segment object moving slower than the neighborhood
             */
            if (-1.0*generalMotionContextDecision>_segmentationParameters.thresholdOFF && meanEnergy*0.9>*neighborhoodMotionPTR)
            {
                /* in order to segment non moving objects in a camera motion case
                 * we focus on local energy which is much lower than the wide neighborhood
                 */
                *segmentationPicturePTR+=(*neighborhoodMotionPTR-*localMotionPTR>_segmentationParameters.thresholdOFF)*127;
            }
        }
#else
        else
            *segmentationPicturePTR=false;
#endif
    }
    /*
#ifdef SEGMENTATIONDEBUG
    std::cout<<"ON: max, min="<<_localMotionON.min()<<", "<<_localMotionON.max();
    std::cout<<"/// \\\ OFF: max, min="<<_localMotionOFF.min()<<", "<<_localMotionOFF.max()<<std::endl;
    std::cout<<"/// \\\ motion: max, min="<<_globalMotionEnergy.min()<<", "<<_globalMotionEnergy.max()<<std::endl;
    std::cout<<"/// \\\ thresholds: ON, OFF="<<_thresholdON<<", "<<_thresholdOFF<<", meanEnergy= "<<meanEnergy<<std::endl;
#endif
     */

}

void TransientAreasSegmentationModuleImpl::getSegmentationPicture(OutputArray transientAreas)
{
    _convertValarrayBuffer2cvMat(_segmentedAreas, getNBrows(), getNBcolumns(), transientAreas);
}


void TransientAreasSegmentationModuleImpl::_convertValarrayBuffer2cvMat(const std::valarray<bool> &grayMatrixToConvert, const unsigned int nbRows, const unsigned int nbColumns, OutputArray outBuffer)
{
    // fill output buffer with the valarray buffer
    const bool *valarrayPTR=get_data(grayMatrixToConvert);

    outBuffer.create(cv::Size(nbColumns, nbRows), CV_8U);
    Mat outMat = outBuffer.getMat();
    for (unsigned int i=0;i<nbRows;++i)
    {
        for (unsigned int j=0;j<nbColumns;++j)
        {
            cv::Point2d pixel(j,i);
            outMat.at<unsigned char>(pixel)=(unsigned char)*(valarrayPTR++);
        }
    }
}

bool TransientAreasSegmentationModuleImpl::_convertCvMat2ValarrayBuffer(InputArray inputMat, std::valarray<float> &outputValarrayMatrix)
{
    const Mat inputMatToConvert=inputMat.getMat();
    // first check input consistency
    if (inputMatToConvert.empty())
        throw cv::Exception(-1, "RetinaImpl cannot be applied, input buffer is empty", "RetinaImpl::run", "RetinaImpl.h", 0);

    // retreive color mode from image input
    int imageNumberOfChannels = inputMatToConvert.channels();

        // convert to float AND fill the valarray buffer
    typedef float T; // define here the target pixel format, here, float
    const int dsttype = DataType<T>::depth; // output buffer is float format

    const unsigned int nbPixels=inputMat.getMat().rows*inputMat.getMat().cols;
    const unsigned int doubleNBpixels=inputMat.getMat().rows*inputMat.getMat().cols*2;

    if(imageNumberOfChannels==4)
    {
    // create a cv::Mat table (for RGBA planes)
        cv::Mat planes[4] =
        {
            cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[doubleNBpixels]),
            cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[nbPixels]),
            cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[0])
        };
        planes[3] = cv::Mat(inputMatToConvert.size(), dsttype);     // last channel (alpha) does not point on the valarray (not usefull in our case)
        // split color cv::Mat in 4 planes... it fills valarray directely
        cv::split(Mat_<Vec<T, 4> >(inputMatToConvert), planes);
    }
    else if (imageNumberOfChannels==3)
    {
        // create a cv::Mat table (for RGB planes)
        cv::Mat planes[] =
        {
        cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[doubleNBpixels]),
        cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[nbPixels]),
        cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[0])
        };
        // split color cv::Mat in 3 planes... it fills valarray directely
        cv::split(cv::Mat_<Vec<T, 3> >(inputMatToConvert), planes);
    }
    else if(imageNumberOfChannels==1)
    {
        // create a cv::Mat header for the valarray
        cv::Mat dst(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[0]);
        inputMatToConvert.convertTo(dst, dsttype);
    }
        else
            CV_Error(Error::StsUnsupportedFormat, "input image must be single channel (gray levels), bgr format (color) or bgra (color with transparency which won't be considered");

    return imageNumberOfChannels>1; // return bool : false for gray level image processing, true for color mode
}

}} //namespaces end : cv and bioinspired
