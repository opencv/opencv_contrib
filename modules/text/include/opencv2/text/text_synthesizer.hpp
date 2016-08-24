/*M//////////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
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
// This software is provided by the copyright holders and contributors "as is" and
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

#ifndef TEXT_SYNTHESIZER_HPP
#define TEXT_SYNTHESIZER_HPP



namespace cv
{
namespace text
{

enum{
    CV_TEXT_SYNTHESIZER_SCRIPT_ANY=1,
    CV_TEXT_SYNTHESIZER_SCRIPT_LATIN=2,
    CV_TEXT_SYNTHESIZER_SCRIPT_GREEK=3,
    CV_TEXT_SYNTHESIZER_SCRIPT_CYRILLIC=4,
    CV_TEXT_SYNTHESIZER_SCRIPT_ARABIC=5,
    CV_TEXT_SYNTHESIZER_SCRIPT_HEBREW=6
};

//TextSynthesizer::blendRandom depends upon these
//enums and should be updated if the change
enum {
    CV_TEXT_SYNTHESIZER_BLND_NORMAL =  100,
    CV_TEXT_SYNTHESIZER_BLND_OVERLAY = 200
};

enum {
    CV_TEXT_SYNTHESIZER_BLND_A_MAX=0,
    CV_TEXT_SYNTHESIZER_BLND_A_MULT=1,
    CV_TEXT_SYNTHESIZER_BLND_A_SUM=2,
    CV_TEXT_SYNTHESIZER_BLND_A_MIN=3,
    CV_TEXT_SYNTHESIZER_BLND_A_MEAN=4
};

/** @brief class that renders synthetic text images for training a CNN on
 * word spotting
 *
 * This functionallity is based on "Synthetic Data and Artificial Neural
 * Networks for Natural Scene Text Recognition" by Max Jaderberg.
 * available at <http://arxiv.org/pdf/1406.2227.pdf>
 *
 * @note
 * - (Python) a demo generating some samples in Greek can be found in:
 * <https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/text_synthesiser.py>
 */
class CV_EXPORTS_W  TextSynthesizer{
protected:
    int resHeight_;
    int maxResWidth_;

    double underlineProbabillity_;
    double italicProbabillity_;
    double boldProbabillity_;
    double maxPerspectiveDistortion_;

    double shadowProbabillity_;
    double maxShadowOpacity_;
    int maxShadowSize_;
    int maxShadowHoffset_;
    int maxShadowVoffset_;

    double borderProbabillity_;
    int maxBorderSize_;

    double curvingProbabillity_;
    double maxHeightDistortionPercentage_;
    double maxCurveArch_;

    double finalBlendAlpha_;
    double finalBlendProb_;

    double compressionNoiseProb_;
    TextSynthesizer(int maxSampleWidth,int sampleHeight);
public:
    CV_WRAP int  getMaxSampleWidth(){return maxResWidth_;}
    CV_WRAP int  getSampleHeight(){return resHeight_;}

    CV_WRAP double getUnderlineProbabillity(){return underlineProbabillity_;}
    CV_WRAP double getItalicProballity(){return italicProbabillity_;}
    CV_WRAP double getBoldProbabillity(){return boldProbabillity_;}
    CV_WRAP double getMaxPerspectiveDistortion(){return maxPerspectiveDistortion_;}

    CV_WRAP double getShadowProbabillity(){return shadowProbabillity_;}
    CV_WRAP double getMaxShadowOpacity(){return maxShadowOpacity_;}
    CV_WRAP int getMaxShadowSize(){return maxShadowSize_;}
    CV_WRAP int getMaxShadowHoffset(){return maxShadowHoffset_;}
    CV_WRAP int getMaxShadowVoffset(){return maxShadowVoffset_;}

    CV_WRAP double getBorderProbabillity(){return borderProbabillity_;}
    CV_WRAP int getMaxBorderSize(){return maxBorderSize_;}

    CV_WRAP double getCurvingProbabillity(){return curvingProbabillity_;}
    CV_WRAP double getMaxHeightDistortionPercentage(){return maxHeightDistortionPercentage_;}
    CV_WRAP double getMaxCurveArch(){return maxCurveArch_;}
    CV_WRAP double getBlendAlpha(){return finalBlendAlpha_;}
    CV_WRAP double getBlendProb(){return finalBlendProb_;}
    CV_WRAP double getCompressionNoiseProb(){return compressionNoiseProb_;}

    /**
     * @param v the probabillity the text will be generated with an underlined font
     */
    CV_WRAP void setUnderlineProbabillity(double v){CV_Assert(v>=0 && v<=1);underlineProbabillity_=v;}

    /**
     * @param v the probabillity the text will be generated with italic font instead of regular
     */
    CV_WRAP void setItalicProballity(double v){CV_Assert(v>=0 && v<=1);italicProbabillity_=v;}

    /**
     * @param v the probabillity the text will be generated with italic font instead of regular
     */
    CV_WRAP void setBoldProbabillity(double v){CV_Assert(v>=0 && v<=1);boldProbabillity_=v;}

    /** Perspective deformation is performed by calculating a homgraphy on a square whose edges
     * have moved randomly inside it.

     * @param v the percentage of the side of a ractangle each point is allowed moving
     */
    CV_WRAP void setMaxPerspectiveDistortion(double v){CV_Assert(v>=0 && v<50);maxPerspectiveDistortion_=v;}

    /**
     * @param v the probabillity a shadow will apear under the text.
     */
    CV_WRAP void setShadowProbabillity(double v){CV_Assert(v>=0 && v<=1);shadowProbabillity_=v;}

    /**
     * @param v the alpha value of the text shadow will be sampled uniformly between 0 and v
     */
    CV_WRAP void setMaxShadowOpacity(double v){CV_Assert(v>=0 && v<=1);maxShadowOpacity_=v;}

    /**
     * @param v the maximum size of the shadow in pixels.
     */
    CV_WRAP void setMaxShadowSize(int v){maxShadowSize_=v;}

    /**
     * @param v the maximum number of pixels the shadow can be horizontaly off-center.
     */
    CV_WRAP void setMaxShadowHoffset(int v){maxShadowHoffset_=v;}

    /**
     * @param v the maximum number of pixels the shadow can be vertically off-center.
     */
    CV_WRAP void setMaxShadowVoffset(int v){maxShadowVoffset_=v;}

    /**
     * @param v the probabillity of a border apearing around the text as oposed to shadows,
     * borders are always opaque and centered.
     */
    CV_WRAP void setBorderProbabillity(double v){CV_Assert(v>=0 && v<=1);borderProbabillity_=v;}

    /**
     * @param v the size in pixels used for border before geometric distortions.
     */
    CV_WRAP void setMaxBorderSize(int v){maxBorderSize_=v;}

    /**
     * @param v the probabillity the text will be curved.
     */
    CV_WRAP void setCurvingProbabillity(double v){CV_Assert(v>=0 && v<=1);curvingProbabillity_=v;}

    /**
     * @param v the maximum effect curving will have as a percentage of the samples height
     */
    CV_WRAP void setMaxHeightDistortionPercentage(double v){CV_Assert(v>=0 && v<=100);maxHeightDistortionPercentage_=v;}

    /**
     * @param v the arch in radians whose cosine will curve the text
     */
    CV_WRAP void setMaxCurveArch(double v){maxCurveArch_=v;}

    /**
     * @param v the maximum alpha used when blending text to the background with opacity
     */
    CV_WRAP void setBlendAlpha(double v){CV_Assert(v>=0 && v<=1);finalBlendAlpha_=v;}

    /**
     * @param v the probability the text will be blended with the background with alpha blending.
     */
    CV_WRAP void setBlendProb(double v){CV_Assert(v>=0 && v<=1);finalBlendProb_=v;}

    /**
     * @param v the probability the sample will be distorted by compression artifacts
     */
    CV_WRAP void getCompressionNoiseProb(double v){CV_Assert(v>=0 && v<=1);compressionNoiseProb_=v;}


    /** @brief adds ttf fonts to the Font Database system
     *
     * Note for the moment adding non system fonts in X11 systems is not an option.
     * <http://doc.qt.io/qt-5/qfontdatabase.html#addApplicationFont>
     *
     * @param v a list of TTF files to be incorporated in to the system.
     */
    CV_WRAP virtual void addFontFiles(const std::vector<String>& fntList)=0;

    CV_WRAP virtual std::vector<String> listAvailableFonts()=0;
    CV_WRAP virtual void modifyAvailableFonts(std::vector<String>& fntList)=0;

    CV_WRAP virtual void addBgSampleImage(const Mat& image)=0;


    CV_WRAP virtual void getColorClusters(CV_OUT Mat& clusters)=0;
    CV_WRAP virtual void setColorClusters(Mat clusters)=0;

    CV_WRAP virtual void generateBgSample(CV_OUT Mat& sample)=0;

    CV_WRAP virtual void generateTxtSample(String caption,CV_OUT Mat& sample,CV_OUT Mat& sampleMask)=0;

    CV_WRAP virtual void generateSample(String caption,CV_OUT Mat& sample)=0;

    CV_WRAP static Ptr<TextSynthesizer> create(int script=CV_TEXT_SYNTHESIZER_SCRIPT_LATIN);
    virtual ~TextSynthesizer(){}
};



}}//text //cv

#endif // TEXT_SYNTHESIZER_HPP
