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
 **  Creation - enhancement process 2007-2013
 **      Author: Alexandre Benoit (benoit.alexandre.vision@gmail.com), LISTIC lab, Annecy le vieux, France
 **
 ** Theses algorithm have been developped by Alexandre BENOIT since his thesis with Alice Caplier at Gipsa-Lab (www.gipsa-lab.inpg.fr) and the research he pursues at LISTIC Lab (www.listic.univ-savoie.fr).
 ** Refer to the following research paper for more information:
 ** Strat S. T. , Benoit A.Lambert P. , Caplier A., "Retina Enhanced SURF Descriptors for Spatio-Temporal Concept Detection", Multimedia Tools and Applications, 2012 (DOI: 10.1007/s11042-012-1280-0)
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
 ** Copyright (C) 2007-2011, LISTIC Lab, Annecy le Vieux and GIPSA Lab, Grenoble, France, all rights reserved.
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

#ifndef SEGMENTATIONMODULE_HPP_
#define SEGMENTATIONMODULE_HPP_

/**
 * @class TransientAreasSegmentationModule
 * @brief class which provides a transient/moving areas segmentation module
 * -> perform a locally adapted segmentation by using the retina magno input data
 * @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com
 * Release date 2007-2013
 * Based on Alexandre BENOIT thesis: "Le système visuel humain au secours de la vision par ordinateur"
 * -> 3 spatio temporal filters are used:
 * a first one which filters the noise and local variations of the input motion energy
 * a second (more powerfull low pass spatial filter) which gives the neighborhood motion energy
 * -> the segmentation consists in the comparison of these both outputs, if the local motion energy is higher to the neighborhood otion energy, then the area is considered as moving and is segmented
 * => a stronger third low pass filter helps decision by providing a smooth information about the "motion context" in a wider area
 */

#include "opencv2/core.hpp" // for all OpenCV core functionalities access, including cv::Exception support

namespace cv
{
namespace bioinspired
{

class CV_EXPORTS TransientAreasSegmentationModule: public Algorithm
{
public:

    // parameters structure
    struct SegmentationParameters{
        SegmentationParameters():
            thresholdON(100),
            thresholdOFF(100),
            // local energy filtering parameters : the aim is to smooth local textures and residual noise
            localEnergy_temporalConstant(0.5), // the time constant of the first order low pass filter, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 0.5 frame
            localEnergy_spatialConstant(5), // the spatial constant of the first order low pass filter, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 5 pixel
            // local neighborhood energy filtering parameters : the aim is to get information about the energy neighborhood to perform a center surround energy analysis
            neighborhoodEnergy_temporalConstant(1),
            neighborhoodEnergy_spatialConstant(15),
            // context neighborhood energy filtering parameters : the aim is to get information about the energy on a wide neighborhood area to filtered out local effects
            contextEnergy_temporalConstant(1),
            contextEnergy_spatialConstant(75){};// default setup
        float thresholdON, thresholdOFF;
        float localEnergy_temporalConstant, localEnergy_spatialConstant;
        float neighborhoodEnergy_temporalConstant, neighborhoodEnergy_spatialConstant;
        float contextEnergy_temporalConstant, contextEnergy_spatialConstant;
    };

    /**
     * @return the sze of the manage input and output images
     */
    virtual Size getSize()=0;

    /**
     * try to open an XML segmentation parameters file to adjust current segmentation instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param retinaParameterFile : the parameters filename
     * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
    virtual void setup(String segmentationParameterFile="", const bool applyDefaultSetupOnFailure=true)=0;

    /**
     * try to open an XML segmentation parameters file to adjust current segmentation instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param fs : the open Filestorage which contains segmentation parameters
     * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
    virtual void setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure=true)=0;

    /**
     * try to open an XML segmentation parameters file to adjust current segmentation instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param newParameters : a parameters structures updated with the new target configuration
     * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
    virtual void setup(SegmentationParameters newParameters)=0;

    /**
     * @return the current parameters setup
     */
    virtual SegmentationParameters getParameters()=0;

    /**
     * parameters setup display method
     * @return a string which contains formatted parameters information
     */
    virtual const String printSetup()=0;

    /**
     * write xml/yml formated parameters information
     * @rparam fs : the filename of the xml file that will be open and writen with formatted parameters information
     */
    virtual void write( String fs ) const=0;

    /**
     * write xml/yml formated parameters information
     * @param fs : a cv::Filestorage object ready to be filled
         */
    virtual void write( cv::FileStorage& fs ) const=0;

    /**
     * main processing method, get result using methods getSegmentationPicture()
     * @param inputToSegment : the image to process, it must match the instance buffer size !
     * @param channelIndex : the channel to process in case of multichannel images
     */
    virtual void run(InputArray inputToSegment, const int channelIndex=0)=0;

    /**
     * access function
     * @return the last segmentation result: a boolean picture which is resampled between 0 and 255 for a display purpose
     */
    virtual void getSegmentationPicture(OutputArray transientAreas)=0;

    /**
     * cleans all the buffers of the instance
     */
    virtual void clearAllBuffers()=0;
};

    /**
     * allocator
     * @param Size : size of the images input to segment (output will be the same size)
     */
CV_EXPORTS Ptr<TransientAreasSegmentationModule> createTransientAreasSegmentationModule(Size inputSize);

}} // namespaces end : cv and bioinspired


#endif


