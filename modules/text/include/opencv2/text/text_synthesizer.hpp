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

    CV_WRAP void setUnderlineProbabillity(double v){underlineProbabillity_=v;}
    CV_WRAP void setItalicProballity(double v){italicProbabillity_=v;}
    CV_WRAP void setBoldProbabillity(double v){boldProbabillity_=v;}
    CV_WRAP void setMaxPerspectiveDistortion(double v){maxPerspectiveDistortion_=v;}

    CV_WRAP void setShadowProbabillity(double v){shadowProbabillity_=v;}
    CV_WRAP void setMaxShadowOpacity(double v){maxShadowOpacity_=v;}
    CV_WRAP void setMaxShadowSize(int v){maxShadowSize_=v;}
    CV_WRAP void setMaxShadowHoffset(int v){maxShadowHoffset_=v;}
    CV_WRAP void setMaxShadowVoffset(int v){maxShadowVoffset_=v;}

    CV_WRAP void setBorderProbabillity(double v){borderProbabillity_=v;}
    CV_WRAP void setMaxBorderSize(int v){maxBorderSize_=v;}

    CV_WRAP void setCurvingProbabillity(double v){curvingProbabillity_=v;}
    CV_WRAP void setMaxHeightDistortionPercentage(double v){maxHeightDistortionPercentage_=v;}
    CV_WRAP void setMaxCurveArch(double v){maxCurveArch_=v;}
    CV_WRAP void setBlendAlpha(double v){finalBlendAlpha_=v;}
    CV_WRAP void setBlendProb(double v){finalBlendProb_=v;}

    CV_WRAP virtual void addFontFiles(const std::vector<String>& fntList)=0;
    CV_WRAP virtual std::vector<String> listAvailableFonts()=0;

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
