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
    //based on QFontDatabase::WritingSystem
    //Qt is the default backend
    CV_TEXT_SYNTHESIZER_SCRIPT_ANY,
    CV_TEXT_SYNTHESIZER_SCRIPT_LATIN,
    CV_TEXT_SYNTHESIZER_SCRIPT_GREEK,
    CV_TEXT_SYNTHESIZER_SCRIPT_CYRILLIC,
    CV_TEXT_SYNTHESIZER_SCRIPT_ARMENIAN,
    CV_TEXT_SYNTHESIZER_SCRIPT_ARABIC,
    CV_TEXT_SYNTHESIZER_SCRIPT_HEBREW,
    CV_TEXT_SYNTHESIZER_SCRIPT_SYRIAC,
    CV_TEXT_SYNTHESIZER_SCRIPT_THAANA,
    CV_TEXT_SYNTHESIZER_SCRIPT_DEVANAGARI,
    CV_TEXT_SYNTHESIZER_SCRIPT_BENGALI,
    CV_TEXT_SYNTHESIZER_SCRIPT_GURMUKHI,
    CV_TEXT_SYNTHESIZER_SCRIPT_GUJARATI,
    CV_TEXT_SYNTHESIZER_SCRIPT_ORIYA,
    CV_TEXT_SYNTHESIZER_SCRIPT_TAMIL,
    CV_TEXT_SYNTHESIZER_SCRIPT_TELUGU,
    CV_TEXT_SYNTHESIZER_SCRIPT_KANNADA,
    CV_TEXT_SYNTHESIZER_SCRIPT_MALAYALAM,
    CV_TEXT_SYNTHESIZER_SCRIPT_SINHALA,
    CV_TEXT_SYNTHESIZER_SCRIPT_THAI,
    CV_TEXT_SYNTHESIZER_SCRIPT_LAO,
    CV_TEXT_SYNTHESIZER_SCRIPT_TIBETAN,
    CV_TEXT_SYNTHESIZER_SCRIPT_MYANMAR,
    CV_TEXT_SYNTHESIZER_SCRIPT_GEORGIAN,
    CV_TEXT_SYNTHESIZER_SCRIPT_KHMER,
    CV_TEXT_SYNTHESIZER_SCRIPT_CHINESE_SIMPLIFIED,
    CV_TEXT_SYNTHESIZER_SCRIPT_CHINESE_TRADITIONAL,
    CV_TEXT_SYNTHESIZER_SCRIPT_JAPANESE,
    CV_TEXT_SYNTHESIZER_SCRIPT_KOREAM,
    CV_TEXT_SYNTHESIZER_SCRIPT_VIETNAMESE
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
    CV_WRAP void setCompressionNoiseProb(double v){CV_Assert(v>=0 && v<=1);compressionNoiseProb_=v;}


    /** @brief adds ttf fonts to the Font Database system
     *
     * Note: for the moment adding non system fonts in X11 systems is not an option.
     * <http://doc.qt.io/qt-5/qfontdatabase.html#addApplicationFont>
     * Fonts should be added to the system if the are to be used with the syntheciser
     *
     * @param fntList a list of TTF files to be incorporated in to the system.
     */
    CV_WRAP virtual void addFontFiles(const std::vector<String>& fntList)=0;

    /** @brief retrieves the font family names that are beeing used by the text
     * synthesizer
     *
     * @return a list of strings with the names from which fonts are sampled.
     */
    CV_WRAP virtual std::vector<String> listAvailableFonts()=0;

    /** @brief updates retrieves the font family names that are randomly sampled
     *
     * This function indirectly allows you to define arbitrary font occurence
     * probabilities. Since fonts are uniformly sampled from this list if a font
     * is repeated, its occurence probabillity doubles.
     *
     * @param fntList a list of strings with the family names from which fonts
     * are sampled. Only font families available in the system can be added.
     */
    CV_WRAP virtual void modifyAvailableFonts(std::vector<String>& fntList)=0;

    /** @brief appends an image in to the collection of images from which
     * backgrounds are sampled.
     *
     * This function indirectly allows you to define arbitrary occurence
     * probabilities. Since background images are uniformly sampled from this
     * list if an image is repeated, its occurence probabillity doubles.
     *
     * @param image an image to be inserted. It should be an 8UC3 matrix which
     * must be least bigger than the generated samples.
     */
    CV_WRAP virtual void addBgSampleImage(const Mat& image)=0;

    /** @brief provides the data from which text colors are sampled
     *
     * @param clusters a 8UC3 Matrix whith three columns and N rows
     */
    CV_WRAP virtual void getColorClusters(CV_OUT Mat& clusters)=0;

    /** @brief defines the data from which text colors are sampled.
     *
     * Text has three color parameters and in order to be able to sample a joined
     * distribution instead of independently sampled, colors are uniformly sampled
     * as color triplets from a fixed collection.
     * This function indirectly allows you to define arbitrary occurence
     * probabilities for every triplet by repeating it samples or polulating with
     * samples.
     *
     * @param clusters a matrix that must be 8UC3, must have 3 columns and any
     * number of rows. Text color is the first matrix color, border color is the
     * second  column and shadow color is the third color.
     */
    CV_WRAP virtual void setColorClusters(Mat clusters)=0;

    /** @brief provides a randomly selected patch exactly as they are provided to text
     * syntheciser
     *
     * @param sample a result variable containing a 8UC3 matrix.
     */
    CV_WRAP virtual void generateBgSample(CV_OUT Mat& sample)=0;

    /** @brief provides the randomly rendered text with border and shadow.
     *
     * @param caption the string which will be rendered. Multilingual strings in
     * UTF8 are suported but some fonts might not support it. The syntheciser should
     * be created with a specific script for fonts guarantiing rendering of the script.
     *
     * @param sample an out variable containing a 32FC3 matrix with the rendered text
     * including border and shadow.
     *
     * @param sampleMask a result parameter which contains the alpha value which is usefull
     * for overlaying the text sample on other images.
     */
    CV_WRAP virtual void generateTxtSample(String caption,CV_OUT Mat& sample,CV_OUT Mat& sampleMask)=0;


    /** @brief generates a random text sample given a string
     *
     * This is the principal function of the text synthciser
     *
     * @param caption the transcription to be written.
     *
     * @param sample the resulting text sample.
     */
    CV_WRAP virtual void generateSample(String caption,CV_OUT Mat& sample)=0;

    /** @brief returns the name of the script beeing used
     *
     * @return a string with the name of the script
     */
    CV_WRAP virtual String getScriptName()=0;

    /** @brief returns the random seed used by the synthesizer
     *
     * @return an unsigned long integer with the random seed.
     */
    CV_WRAP virtual uint64 getRandomSeed()=0;

    /** @brief stets the random seed used by the synthesizer
     *
     * @param an unsigned long integer with the random seed to be set.
     */
    CV_WRAP virtual void setRandomSeed(uint64 s)=0;

    /** @brief public constructor for a syntheciser
     *
     * This constructor assigns only imutable properties of the syntheciser.
     *
     * @param sampleHeight the height of final samples in pixels
     *
     * @param maxWidth the maximum width of a sample. Any text requiring more
     * width to be rendered will be ignored.
     *
     * @param script an enumaration which is used to constrain the available fonts
     * to the ones beeing able to render strings in that script.
     */
    CV_WRAP static Ptr<TextSynthesizer> create(int sampleHeight=50, int maxWidth=600, int script=CV_TEXT_SYNTHESIZER_SCRIPT_ANY);
    virtual ~TextSynthesizer(){}
};



}}//text //cv

#endif // TEXT_SYNTHESIZER_HPP
