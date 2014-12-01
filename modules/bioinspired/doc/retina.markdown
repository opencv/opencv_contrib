Bioinspired Module Retina Introduction {#bioinspired_retina}
======================================

Retina
------

@note do not forget that the retina model is included in the following namespace : cv::bioinspired

### Introduction

Class which provides the main controls to the Gipsa/Listic labs human retina model. This is a non
separable spatio-temporal filter modelling the two main retina information channels :

-   foveal vision for detailled color vision : the parvocellular pathway.
-   peripheral vision for sensitive transient signals detection (motion and events) : the
    magnocellular pathway.

From a general point of view, this filter whitens the image spectrum and corrects luminance thanks
to local adaptation. An other important property is its hability to filter out spatio-temporal noise
while enhancing details. This model originates from Jeanny Herault work @cite Herault2010 . It has been
involved in Alexandre Benoit phd and his current research @cite Benoit2010, @cite Strat2013 (he
currently maintains this module within OpenCV). It includes the work of other Jeanny's phd student
such as @cite Chaix2007 and the log polar transformations of Barthelemy Durette described in Jeanny's
book.

@note
-   For ease of use in computer vision applications, the two retina channels are applied
    homogeneously on all the input images. This does not follow the real retina topology but this
    can still be done using the log sampling capabilities proposed within the class.
-   Extend the retina description and code use in the tutorial/contrib section for complementary
    explanations.

### Preliminary illustration

As a preliminary presentation, let's start with a visual example. We propose to apply the filter on
a low quality color jpeg image with backlight problems. Here is the considered input... *"Well, my
eyes were able to see more that this strange black shadow..."*

![a low quality color jpeg image with backlight problems.](images/retinaInput.jpg)

Below, the retina foveal model applied on the entire image with default parameters. Here contours
are enforced, halo effects are voluntary visible with this configuration. See parameters discussion
below and increase horizontalCellsGain near 1 to remove them.

![the retina foveal model applied on the entire image with default parameters. Here contours are enforced, luminance is corrected and halo effects are voluntary visible with this configuration, increase horizontalCellsGain near 1 to remove them.](images/retinaOutput_default.jpg)

Below, a second retina foveal model output applied on the entire image with a parameters setup
focused on naturalness perception. *"Hey, i now recognize my cat, looking at the mountains at the
end of the day !"*. Here contours are enforced, luminance is corrected but halos are avoided with
this configuration. The backlight effect is corrected and highlight details are still preserved.
Then, even on a low quality jpeg image, if some luminance information remains, the retina is able to
reconstruct a proper visual signal. Such configuration is also usefull for High Dynamic Range
(*HDR*) images compression to 8bit images as discussed in @cite Benoit2010 and in the demonstration
codes discussed below. As shown at the end of the page, parameters change from defaults are :

-   horizontalCellsGain=0.3
-   photoreceptorsLocalAdaptationSensitivity=ganglioncellsSensitivity=0.89.

![the retina foveal model applied on the entire image with 'naturalness' parameters. Here contours are enforced but are avoided with this configuration, horizontalCellsGain is 0.3 and photoreceptorsLocalAdaptationSensitivity=ganglioncellsSensitivity=0.89.](images/retinaOutput_realistic.jpg)

As observed in this preliminary demo, the retina can be settled up with various parameters, by
default, as shown on the figure above, the retina strongly reduces mean luminance energy and
enforces all details of the visual scene. Luminance energy and halo effects can be modulated
(exagerated to cancelled as shown on the two examples). In order to use your own parameters, you can
use at least one time the *write(String fs)* method which will write a proper XML file with all
default parameters. Then, tweak it on your own and reload them at any time using method
*setup(String fs)*. These methods update a *Retina::RetinaParameters* member structure that is
described hereafter. XML parameters file samples are shown at the end of the page.

Here is an overview of the abstract Retina interface, allocate one instance with the *createRetina*
functions.:
@code{.cpp}
    namespace cv{namespace bioinspired{

    class Retina : public Algorithm
    {
    public:
      // parameters setup instance
      struct RetinaParameters; // this class is detailled later

      // main method for input frame processing (all use method, can also perform High Dynamic Range tone mapping)
      void run (InputArray inputImage);

      // specific method aiming at correcting luminance only (faster High Dynamic Range tone mapping)
      void applyFastToneMapping(InputArray inputImage, OutputArray outputToneMappedImage)

      // output buffers retreival methods
      // -> foveal color vision details channel with luminance and noise correction
      void getParvo (OutputArray retinaOutput_parvo);
      void getParvoRAW (OutputArray retinaOutput_parvo);// retreive original output buffers without any normalisation
      const Mat getParvoRAW () const;// retreive original output buffers without any normalisation
      // -> peripheral monochrome motion and events (transient information) channel
      void getMagno (OutputArray retinaOutput_magno);
      void getMagnoRAW (OutputArray retinaOutput_magno); // retreive original output buffers without any normalisation
      const Mat getMagnoRAW () const;// retreive original output buffers without any normalisation

      // reset retina buffers... equivalent to closing your eyes for some seconds
      void clearBuffers ();

      // retreive input and output buffers sizes
      Size getInputSize ();
      Size getOutputSize ();

      // setup methods with specific parameters specification of global xml config file loading/write
      void setup (String retinaParameterFile="", const bool applyDefaultSetupOnFailure=true);
      void setup (FileStorage &fs, const bool applyDefaultSetupOnFailure=true);
      void setup (RetinaParameters newParameters);
      struct Retina::RetinaParameters getParameters ();
      const String printSetup ();
      virtual void write (String fs) const;
      virtual void write (FileStorage &fs) const;
      void setupOPLandIPLParvoChannel (const bool colorMode=true, const bool normaliseOutput=true, const float photoreceptorsLocalAdaptationSensitivity=0.7, const float photoreceptorsTemporalConstant=0.5, const float photoreceptorsSpatialConstant=0.53, const float horizontalCellsGain=0, const float HcellsTemporalConstant=1, const float HcellsSpatialConstant=7, const float ganglionCellsSensitivity=0.7);
      void setupIPLMagnoChannel (const bool normaliseOutput=true, const float parasolCells_beta=0, const float parasolCells_tau=0, const float parasolCells_k=7, const float amacrinCellsTemporalCutFrequency=1.2, const float V0CompressionParameter=0.95, const float localAdaptintegration_tau=0, const float localAdaptintegration_k=7);
      void setColorSaturation (const bool saturateColors=true, const float colorSaturationValue=4.0);
      void activateMovingContoursProcessing (const bool activate);
      void activateContoursProcessing (const bool activate);
    };

      // Allocators
      cv::Ptr<Retina> createRetina (Size inputSize);
      cv::Ptr<Retina> createRetina (Size inputSize, const bool colorMode, RETINA_COLORSAMPLINGMETHOD colorSamplingMethod=RETINA_COLOR_BAYER, const bool useRetinaLogSampling=false, const double reductionFactor=1.0, const double samplingStrenght=10.0);
      }} // cv and bioinspired namespaces end
@endcode

### Description

Class which allows the [Gipsa](http://www.gipsa-lab.inpg.fr) (preliminary work) /
[Listic](http://www.listic.univ-savoie.fr) (code maintainer and user) labs retina model to be used.
This class allows human retina spatio-temporal image processing to be applied on still images,
images sequences and video sequences. Briefly, here are the main human retina model properties:

-   spectral whithening (mid-frequency details enhancement)
-   high frequency spatio-temporal noise reduction (temporal noise and high frequency spatial noise
    are minimized)
-   low frequency luminance reduction (luminance range compression) : high luminance regions do not
    hide details in darker regions anymore
-   local logarithmic luminance compression allows details to be enhanced even in low light
    conditions

Use : this model can be used basically for spatio-temporal video effects but also in the aim of :

-   performing texture analysis with enhanced signal to noise ratio and enhanced details robust
    against input images luminance ranges (check out the parvocellular retina channel output, by
    using the provided **getParvo** methods)
-   performing motion analysis also taking benefit of the previously cited properties (check out the
    magnocellular retina channel output, by using the provided **getMagno** methods)
-   general image/video sequence description using either one or both channels. An example of the
    use of Retina in a Bag of Words approach is given in @cite Strat2013 .

Literature
----------

For more information, refer to the following papers :

-   Model description : @cite Benoit2010

-   Model use in a Bag of Words approach : @cite Strat2013

-   Please have a look at the reference work of Jeanny Herault that you can read in his book : @cite Herault2010

This retina filter code includes the research contributions of phd/research collegues from which
code has been redrawn by the author :

-   take a look at the *retinacolor.hpp* module to discover Brice Chaix de Lavarene phD color
    mosaicing/demosaicing and his reference paper: @cite Chaix2007

-   take a look at *imagelogpolprojection.hpp* to discover retina spatial log sampling which
    originates from Barthelemy Durette phd with Jeanny Herault. A Retina / V1 cortex projection is
    also proposed and originates from Jeanny's discussions. More informations in the above cited
    Jeanny Heraults's book.

-   Meylan&al work on HDR tone mapping that is implemented as a specific method within the model : @cite Meylan2007

Demos and experiments !
-----------------------

@note Complementary to the following examples, have a look at the Retina tutorial in the
tutorial/contrib section for complementary explanations.**

Take a look at the provided C++ examples provided with OpenCV :

-   **samples/cpp/retinademo.cpp** shows how to use the retina module for details enhancement (Parvo channel output) and transient maps observation (Magno channel output). You can play with images, video sequences and webcam video.
    Typical uses are (provided your OpenCV installation is situated in folder *OpenCVReleaseFolder*)
    -   image processing : **OpenCVReleaseFolder/bin/retinademo -image myPicture.jpg**
    -   video processing : **OpenCVReleaseFolder/bin/retinademo -video myMovie.avi**
    -   webcam processing: **OpenCVReleaseFolder/bin/retinademo -video**

    @note This demo generates the file *RetinaDefaultParameters.xml* which contains the
    default parameters of the retina. Then, rename this as *RetinaSpecificParameters.xml*, adjust
    the parameters the way you want and reload the program to check the effect.

-   **samples/cpp/OpenEXRimages\_HDR\_Retina\_toneMapping.cpp** shows how to use the retina to
    perform High Dynamic Range (HDR) luminance compression

    Then, take a HDR image using bracketing with your camera and generate an OpenEXR image and
    then process it using the demo.

    Typical use, supposing that you have the OpenEXR image such as *memorial.exr* (present in the
    samples/cpp/ folder)

-   **OpenCVReleaseFolder/bin/OpenEXRimages\_HDR\_Retina\_toneMapping memorial.exr [optional:
    'fast']**

    Note that some sliders are made available to allow you to play with luminance compression.

    If not using the 'fast' option, then, tone mapping is performed using the full retina model
    @cite Benoit2010 . It includes spectral whitening that allows luminance energy to be reduced.
    When using the 'fast' option, then, a simpler method is used, it is an adaptation of the
    algorithm presented in @cite Meylan2007 . This method gives also good results and is faster to
    process but it sometimes requires some more parameters adjustement.
