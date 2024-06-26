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
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_SALIENCY_SPECIALIZED_CLASSES_HPP__
#define __OPENCV_SALIENCY_SPECIALIZED_CLASSES_HPP__

#include <cstdio>
#include <string>
#include <iostream>
#include <stdint.h>
#include <vector>
#include "saliencyBaseClasses.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

namespace cv
{
namespace saliency
{

//! @addtogroup saliency
//! @{

/************************************ Specific Static Saliency Specialized Classes ************************************/

/** @brief the Spectral Residual approach from  @cite SR

Starting from the principle of natural image statistics, this method simulate the behavior of
pre-attentive visual search. The algorithm analyze the log spectrum of each image and obtain the
spectral residual. Then transform the spectral residual to spatial domain to obtain the saliency
map, which suggests the positions of proto-objects.
 */
class CV_EXPORTS_W StaticSaliencySpectralResidual : public StaticSaliency
{
public:

  StaticSaliencySpectralResidual();
  virtual ~StaticSaliencySpectralResidual();

  CV_WRAP static Ptr<StaticSaliencySpectralResidual> create()
  {
    return makePtr<StaticSaliencySpectralResidual>();
  }

  CV_WRAP bool computeSaliency( InputArray image, OutputArray saliencyMap )
  {
    if( image.empty() )
      return false;

    return computeSaliencyImpl( image, saliencyMap );
  }

  CV_WRAP void read( const FileNode& fn ) CV_OVERRIDE;
  void write( FileStorage& fs ) const CV_OVERRIDE;

  CV_WRAP int getImageWidth() const
  {
    return resImWidth;
  }
  CV_WRAP inline void setImageWidth(int val)
  {
    resImWidth = val;
  }
  CV_WRAP int getImageHeight() const
  {
    return resImHeight;
  }
  CV_WRAP void setImageHeight(int val)
  {
    resImHeight = val;
  }

protected:
  bool computeSaliencyImpl( InputArray image, OutputArray saliencyMap ) CV_OVERRIDE;
  CV_PROP_RW int resImWidth;
  CV_PROP_RW int resImHeight;

};


/** @brief the Fine Grained Saliency approach from @cite FGS

This method calculates saliency based on center-surround differences.
High resolution saliency maps are generated in real time by using integral images.
 */
class CV_EXPORTS_W StaticSaliencyFineGrained : public StaticSaliency
{
public:

  StaticSaliencyFineGrained();

  CV_WRAP static Ptr<StaticSaliencyFineGrained> create()
  {
    return makePtr<StaticSaliencyFineGrained>();
  }

  CV_WRAP bool computeSaliency( InputArray image, OutputArray saliencyMap )
  {
    if( image.empty() )
      return false;

    return computeSaliencyImpl( image, saliencyMap );
  }
  virtual ~StaticSaliencyFineGrained();

protected:
  bool computeSaliencyImpl( InputArray image, OutputArray saliencyMap ) CV_OVERRIDE;

private:
  void calcIntensityChannel(Mat src, Mat dst);
  void copyImage(Mat src, Mat dst);
  void getIntensityScaled(Mat integralImage, Mat gray, Mat saliencyOn, Mat saliencyOff, int neighborhood);
  float getMean(Mat srcArg, Point2i PixArg, int neighbourhood, int centerVal);
  void mixScales(Mat *saliencyOn, Mat intensityOn, Mat *saliencyOff, Mat intensityOff, const int numScales);
  void mixOnOff(Mat intensityOn, Mat intensityOff, Mat intensity);
  void getIntensity(Mat srcArg, Mat dstArg,  Mat dstOnArg,  Mat dstOffArg, bool generateOnOff);
};

/** @brief the Deep Gaze 1 Saliency approach from @cite kummerer2014deep

This method uses the linear combination of the fifth convolution layers of the pretrained AlexNet, center bias and softmax to generate saliency map.
You can passes your own DNN layers into the DeepGaze1 object and retrain it with "training" method to take advantage of the cutting-edge DNN in future.
*/
class CV_EXPORTS_W DeepGaze1 : public StaticSaliency
{
private:
    /** @brief This is the field to store the input DNN
     * @sa dnn::Net
     */
    dnn::Net net;
    /** @brief This is the field to store the name of selected layer
     */
    std::vector<std::string> layers_names;
    /** @brief This is the field to store the weight of selected layer
    */
    std::vector<double> weights;

public:
    /** @brief This is the simplest constructor. It works with AlexNet caffe implementation.
     * You can put the prototxt file and caffemodel file in the project default path or pass the path of required files into the constructor.
     * The weights for AlexNet have been tuned and validated.
     * @param net_proto The path of AlexNet prototxt file
     * @param net_caffemodel The path of AlexNet caffemodel file
     * @sa dnn::Net , dnn:readNetFromCaffe
    */
    DeepGaze1( std::string net_proto = "deploy.prototxt", std::string net_caffemodel = "bvlc_alexnet.caffemodel" );
    /** @brief This is the constructor works with your own DNN caffe implementation.
     * You can put the prototxt file and caffemodel file in the project default path or pass the path of required files into the constructor.
     * The weights of the selected layers are randomly initialized. Need to call training method to tune them.
     * To download the caffemodel of Alexnet: http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
     * The prototxt we use is slightly different from the one in the bvlc github: https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt
     * To find the modified "deploy.prototxt" specific for DeepGaze1 module: https://github.com/opencv/opencv_contrib/tree/master/modules/cnn_3dobj/testdata/cv/deploy.prototxt
     * @param net_proto The path of DNN prototxt file. The default is "deploy.prototxt"
     * @param net_caffemodel The path of DNN caffemodel file. The default is "bvlc_alexnet.caffemodel"
     * @param selected_layers The name of the layers you selected
     * @param n_weights The number of weights you want to initialized
     * @sa dnn::Net , dnn:readNetFromCaffe
    */
    DeepGaze1( std::string net_proto, std::string net_caffemodel, std::vector<std::string> selected_layers, unsigned n_weights);
    /** @brief This is the constructor works with your own DNN caffe implementation with your own tuned weights.
     * You can put the prototxt file and caffemodel file in the project default path or pass the path of required files into the constructor.
     * @param net_proto The path of DNN prototxt file
     * @param net_caffemodel The path of DNN caffemodel file
     * @param selected_layers The name of the layers you selected
     * @param i_weights The weights you want to initializedcv
     * @sa dnn::Net , dnn:readNetFromCaffe
    */
    DeepGaze1( std::string net_proto, std::string net_caffemodel, std::vector<std::string> selected_layers, std::vector<double> i_weights);
    virtual ~DeepGaze1();
    CV_WRAP static Ptr<DeepGaze1> create()
    {
        return makePtr<DeepGaze1>();
    }
    CV_WRAP bool computeSaliency( InputArray image, OutputArray saliencyMap )
    {
        if( image.empty() )
            return false;
        return computeSaliencyImpl( image, saliencyMap );
    }
    /** @brief This is the method you can use to generate saliency map with your own input image size.
     * The input image size you set must fit your own DNN structure.
     * @param input_image The Mat you want to be processed
     * @param input_size The image size that fit your DNN structure. The default setting is 227 * 227 which fits AlexNet
    */
    Mat saliencyMapGenerator( Mat input_image, Size input_size = Size(227, 227) );
    /** @brief This is the method you can use to train untuned linear combination weights of DNN selected layers.
     * Be careful about decayed momentum SGD hyperparameter you set like the batch size, the learning rate, the momentum, the number iteration, the decay rate.
     * The default hyperparameter is not for pretrained AlexNet weights.
     * Do not be shock by the saliency map you generated. It needs to be thresholded before revealing saliency objects. You can use Otsu's method to threshold the saliency map and get the foreground object.
     * @param images The training set images
     * @param fixMaps The saliency map of training set
     * @param iteration The number of times you iterate on the same batch. The default is 1
     * @param batch_size The size of the batch. The default is 100
     * @param momentum The value of momentum. The default is 0.9
     * @param alpha The value of SGD learnin rate. The default is 0.01. Inproper setting may make performance worse
     * @param decay The value of decay. The default is 0.01
     * @param input_size The image size that fit your DNN structure. The default setting is 227 * 227 which fits AlexNet
    */
    void training( std::vector<Mat>& images, std::vector<Mat>& fixMaps, int iteration = 1, unsigned batch_size = 100, double momentum = 0.9, double alpha = 0.01, double decay = 0.01, Size input_size = Size(227, 227) );
    /** @brief This is the method you can use to calculate the AUC.
     * @param _saliencyMap The saliency map to be tested
     * @param _fixtionMap The ground truth map
    */
    double computeAUC( InputArray _saliencyMap, InputArray _fixtionMap );
    /** @brief This is the method you can use to visualize saliency map.
     * @param _saliencyMap The saliency map to be visualized
    */
    void saliencyMapVisualize( InputArray _saliencyMap );
protected:
    bool computeSaliencyImpl( InputArray image, OutputArray saliencyMap );
    /** @brief This is the method to retrieve layers from the DNN net which are normalized to have unit standard deviation on each elements across all layers.
     * @param img The img to be processed by DNN
     * @param input_size The rescaled image size that fits DNN
    */
    std::vector<Mat> featureMapGenerator( Mat img, Size input_size );
    /** @brief This is the method to do linear combination of selected normalized layers and Gaussian blur it.
     * @param featureMaps The selected normalized layers generated by method "featureMapGenerator"
     * @param wei The weights of selected normalized layers
    */
    static Mat comb( std::vector<Mat>& featureMaps, std::vector<double> wei );
    /** @brief This is the method to do softmax
     * @param res The result of method "comb"
    */
    static Mat softmax( Mat res );
    /** @brief This is the method to calculate grandients of loss function
     * @param featureMaps The selected normalized layers generated by method "featureMapGenerator"
     * @param randIndex The index of dimensions that you want to used to accumulate the loss
     * @param wei The weights of selected normalized layers to be updated
     * @param input_size The input image size which needs to fit the DNN you choose
    */
    static std::vector<double> evalGrad( std::vector<Mat>& featureMaps, std::vector<unsigned>& randIndex, std::vector<double> wei, Size input_size );
    /** @brief This is the method to randomly draw index of images in training dataset
     * @param total The size of the training dataset
     * @param batchSize The size of the batch you want to draw
    */
    std::vector<unsigned> batchIndex( unsigned total, unsigned batchSize );
    /** @brief This is the method to calculate the loss in order to update weights
     * @param saliency_sample The value of sampled elements on saliency map which are used to accumulate the loss
     * @param wei The weights of selected normalized layers in current iteration
    */
    static double loss( std::vector<double> saliency_sample, std::vector<double> wei );
    /** @brief This is the method to sample the elements on saliency map in current iteration
     * @param saliency Saliency map to be sampled
     * @param randIndex The index of elements to be sampled
    */
    static std::vector<double> mapSampler( Mat saliency, std::vector<unsigned> randIndex );
    /** @brief This is the method to determine which element on the fixation map in the training datasets belongs to saliency objects and generate the sampled elements index.
     * @param img The fixation map in the training datasets
     * @param input_size The input image size that fits DNN you choose
    */
    std::vector<unsigned> fixationLoc( Mat img, Size input_size );
};

/** @brief the Background Contrast Saliency approach from @cite zhu2014saliency

This method uses seed superpixel algorithm to partition the image and calculate the probability of belonging to background with CIE-Lab color contrast.
This method also provides an optimization framework may help foreground based saliency method perform better.
*/
class CV_EXPORTS_W BackgroundContrast : public StaticSaliency
{
private:
    /** @brief
     * limitOfSp is the maximum number of superpixel
     * nOfLevel, usePrior, histBin is same as the seed superpixel method
     * bgWei determine the weight of background when optimized with foreground method
    */
    int limitOfSP;
    int nOfLevel;
    int usePrior;
    int histBin;
    double bgWei;
public:
    /** @brief This is the default constructor.
     * It will initialize with bgWei: 5, limitOfSp: 600, nOfLevel: 4, usePrior: 2, histBin: 5
    */
    BackgroundContrast();
    /** @brief This is the constructor user can custumized those parameters
    */
    BackgroundContrast(  double, int, int, int, int );
    virtual ~BackgroundContrast();
    CV_WRAP static Ptr<BackgroundContrast> create()
    {
        return makePtr<BackgroundContrast>();
    }
    CV_WRAP bool computeSaliency( InputArray image, OutputArray saliencyMap )
    {
        if( image.empty() )
            return false;
        return computeSaliencyImpl( image, saliencyMap );
    }
    /** @brief This is the method you can use to generate saliency map with or without outside saliency map
     * @param img The image to be processed
     * @param fgImg The outside saliency map to be optimized with aggregated optimization framework
     * @param option The flag parameter for user to use outside saliency map or not. The default is 0 which represents do not use outside saliency map. 1 represents outside saliency map will be optimized
    */
    Mat saliencyMapGenerator( const Mat img, const Mat fgImg = Mat(), int option = 0 );
    /** @brief This is the method you can use to visualize saliency map
     * This method offers 3 different threshold options to retrieve saliency object from saliency map.
     * @param _saliencyMap The saliency map to be visualized
     * @param option 0(default): no threshold, 1: threshold with Otsu's method and binary threshold, 2: threshold with Otsu's method and tozero threshold
    */
    Mat saliencyMapVisualize( InputArray _saliencyMap, int option = 0 );
protected:
    /** @brief This is the method to use background probability to optimize saliency map
     * This method essentially use least-square to solve an optimization problem.
     * @param adjcMatrix superpixel adjacency matrix calculated by method superpixelSplit
     * @param colDistM superpixel CIE-Lab color distance matrix calcualted by method getColorPosDis
     * @param bgWeight The value of background probability
     * @param fgWeight The value of foreground probability(saliency)
     * @param saliencyOptimized The output optimized saliency map
     * @param bgLambda The weight of background probability compared with foreground probability, default 5
     * @param neiSigma The variance that used to transform smoothness weight into [0, 1] range, default 14
    */
    void saliencyOptimize( const Mat adjcMatrix, const Mat colDistM, const Mat bgWeight, const Mat fgWeight, Mat& saliencyOptimized, double bgLambda = 5, double neiSigma = 14 );
    bool computeSaliencyImpl( InputArray image, OutputArray saliencyMap );
    /** @brief This is the method to use opencv seed superpixel method to split the image and return label of each pixel and adjacency matrix
     * @param img The image to be processed
     * @param idxImg The output label map of each pixel
     * @param adjcMatrix The output adjacency matrix
    */
    void superpixelSplit( const Mat img, Mat& idxImg, Mat& adjcMatrix );
    /** @brief This is the method to find background superpixel
     * @param idxImg The label of each pixel in the original image
     * @param thickness The width of background boundary, default is 8
    */
    std::vector<unsigned> getBndPatchIds( const Mat idxImg, int thickness = 8);
    /** @brief this is the method to calculate superpixel CIE-Lab color distance matrix and geographic distance matrix
     * @param img The image to be processed
     * @param idxImg The label of each pixel
     * @param colDistM The output superpixel color distance matrix
     * @param posDistM The output superpixel geographic distance matrix
     * @param nOfSP The number of superpixels that the image is splited
    */
    void getColorPosDis( const Mat img, const Mat idxImg, Mat& colDistM, Mat& posDistM, int nOfSP);
    /** @brief This is the method to calculate the background probability of each superpixel
     * This method use floyd algorithm to find all shorted path between superpixels
     * @param adjcMatrix The adjacency matrix of superpixels
     * @param colDistM The superpixel color distance matrix
     * @param bdProb The output superpixel background probability
     * @param bdIds The index of background superpixels
     * @param clipVal The bias of distance between adjacent superpixels, default 3
     * @param geoSigma The variance that used to transform geographic distance into [0, 1] range, default 7
    */
    void boundaryConnectivity( const Mat adjcMatrix, const Mat colDistM, Mat& bdProb, std::vector<unsigned> bdIds, double clipVal = 3.0, double geoSigma = 7.0 );
    /** @brief This is the method to calcualte weighted background contrast
     * @param colDistM The superpixel color distance matrix
     * @param posDistM The superpixel geographic distance matrix
     * @param bgProb The background probability of superpixels
     * @param wCtr The background weighted contrast which can be treated as the saliency map generated by the background probability
    */
    void getWeightedContrast( const Mat colDistM, const Mat posDistM, const Mat bgProb, Mat& wCtr );
    /** @brief This is the method to transform the input value into [0,1] range with variance
    */
    void dist2WeightMatrix( Mat&, Mat&, double );
    /** @brief This is the method to transform OpenCV BGR color into CIE-Lab color space
    */
    void rgb2lab( Mat&, Mat& );
};





/************************************ Specific Motion Saliency Specialized Classes ************************************/

/*!
 * A Fast Self-tuning Background Subtraction Algorithm.
 *
 * This background subtraction algorithm is inspired to the work of B. Wang and P. Dudek [2]
 * [2]  B. Wang and P. Dudek "A Fast Self-tuning Background Subtraction Algorithm", in proc of IEEE Workshop on Change Detection, 2014
 *
 */
/** @brief the Fast Self-tuning Background Subtraction Algorithm from @cite BinWangApr2014
 */
class CV_EXPORTS_W MotionSaliencyBinWangApr2014 : public MotionSaliency
{
public:
  MotionSaliencyBinWangApr2014();
  virtual ~MotionSaliencyBinWangApr2014();

  CV_WRAP static Ptr<MotionSaliencyBinWangApr2014> create()
  {
    return makePtr<MotionSaliencyBinWangApr2014>();
  }

  CV_WRAP bool computeSaliency( InputArray image, OutputArray saliencyMap )
  {
    if( image.empty() )
      return false;

    return computeSaliencyImpl( image, saliencyMap );
  }

  /** @brief This is a utility function that allows to set the correct size (taken from the input image) in the
    corresponding variables that will be used to size the data structures of the algorithm.
    @param W width of input image
    @param H height of input image
  */
  CV_WRAP void setImagesize( int W, int H );
  /** @brief This function allows the correct initialization of all data structures that will be used by the
    algorithm.
  */
  CV_WRAP bool init();

  CV_WRAP int getImageWidth() const
  {
    return imageWidth;
  }
  CV_WRAP inline void setImageWidth(int val)
  {
    imageWidth = val;
  }
  CV_WRAP int getImageHeight() const
  {
    return imageHeight;
  }
  CV_WRAP void setImageHeight(int val)
  {
    imageHeight = val;
  }

protected:
  /** @brief Performs all the operations and calls all internal functions necessary for the accomplishment of the
    Fast Self-tuning Background Subtraction Algorithm algorithm.
    @param image input image. According to the needs of this specialized algorithm, the param image is a
    single *Mat*.
    @param saliencyMap Saliency Map. Is a binarized map that, in accordance with the nature of the algorithm, highlights the moving objects or areas of change in the scene.
       The saliency map is given by a single *Mat* (one for each frame of an hypothetical video
        stream).
  */
  bool computeSaliencyImpl( InputArray image, OutputArray saliencyMap ) CV_OVERRIDE;

private:

  // classification (and adaptation) functions
  bool fullResolutionDetection( const Mat& image, Mat& highResBFMask );
  bool lowResolutionDetection( const Mat& image, Mat& lowResBFMask );

  // Background model maintenance functions
  bool templateOrdering();
  bool templateReplacement( const Mat& finalBFMask, const Mat& image );

  // Decision threshold adaptation and Activity control function
  bool activityControl(const Mat& current_noisePixelsMask);
  bool decisionThresholdAdaptation();

  // changing structure
  std::vector<Ptr<Mat> > backgroundModel;// The vector represents the background template T0---TK of reference paper.
  // Matrices are two-channel matrix. In the first layer there are the B (background value)
  // for each pixel. In the second layer, there are the C (efficacy) value for each pixel
  Mat potentialBackground;// Two channel Matrix. For each pixel, in the first level there are the Ba value (potential background value)
                          // and in the secon level there are the Ca value, the counter for each potential value.
  Mat epslonPixelsValue;// epslon threshold

  Mat activityPixelsValue;// Activity level of each pixel

  //vector<Mat> noisePixelMask; // We define a ‘noise-pixel’ as a pixel that has been classified as a foreground pixel during the full resolution
  Mat noisePixelMask;// We define a ‘noise-pixel’ as a pixel that has been classified as a foreground pixel during the full resolution
  //detection process,however, after the low resolution detection, it has become a
  // background pixel. The matrix is  two-channel matrix. In the first layer there is the mask ( the identified noise-pixels are set to 1 while other pixels are 0)
  // for each pixel. In the second layer, there is the value of activity level A for each pixel.

  //fixed parameter
  bool activityControlFlag;
  bool neighborhoodCheck;
  int N_DS;// Number of template to be downsampled and used in lowResolutionDetection function
  CV_PROP_RW int imageWidth;// Width of input image
  CV_PROP_RW int imageHeight;//Height of input image
  int K;// Number of background model template
  int N;// NxN is the size of the block for downsampling in the lowlowResolutionDetection
  float alpha;// Learning rate
  int L0, L1;// Upper-bound values for C0 and C1 (efficacy of the first two template (matrices) of backgroundModel
  int thetaL;// T0, T1 swap threshold
  int thetaA;// Potential background value threshold
  int gamma;// Parameter that controls the time that the newly updated long-term background value will remain in the
            // long-term template, regardless of any subsequent background changes. A relatively large (eg gamma=3) will
            //restrain the generation of ghosts.

  uchar Ainc;// Activity Incrementation;
  int Bmax;// Upper-bound value for pixel activity
  int Bth;// Max activity threshold
  int Binc, Bdec;// Threshold for pixel-level decision threshold (epslon) adaptation
  float deltaINC, deltaDEC;// Increment-decrement value for epslon adaptation
  int epslonMIN, epslonMAX;// Range values for epslon threshold

};

/** @brief the Discriminant Saliency approach from @cite mahadevan2010spatiotemporal

This method uses dynamic texture model to capture the essence of dynamic background and seperate it from non background object.
*/
class CV_EXPORTS_W DiscriminantSaliency : public MotionSaliency
{
private:
    Size imgProcessingSize;
    unsigned hiddenSpaceDimension;
    unsigned centerSize;
    unsigned windowSize;
    unsigned patchSize;
    unsigned temporalSize;
    unsigned stride;
public:
    /** @brief The structure to store dynamic texture model parameters
    */
    struct DT
    {
        Mat A;
        Mat C;
        Mat Q;
        Mat R;
        Mat S;
        Mat MU;
        double VAR;
    };
//    DiscriminantSaliency();
    /** @brief Constructor
     * @param _stride The parameter to adjust saliency map resolution level. The higher the parameter is, the lower the resolution is, the faster the processing is, default 1
     * @param _imgProcessingSize The image size to be processed. You can rescale the image with this parameter. The larger the image is, the slower the processing is, default 127 * 127
     * @param _hidden The hidden space dimension. The default is 10.
     * @param _center The center window size. The default is 8 * 8
     * @param _window The sliding window size. The default is 96 * 96
     * @param _patch This parameter determines how many pixels are drawed when estimate dynamic texture parameters, default is 400
     * @param _temporal This parameter determines how many frames are considered when estimate dynamic texture parameters, default is 11
    */
    DiscriminantSaliency(unsigned _stride = 1, Size _imgProcessingSize = Size(127, 127), unsigned _hidden = 10, unsigned _center = 8, unsigned _window = 96, unsigned _patch = 400, unsigned _temporal = 11);
    virtual ~DiscriminantSaliency();
    CV_WRAP static Ptr<DiscriminantSaliency> create()
    {
        return makePtr<DiscriminantSaliency>();
    }
    CV_WRAP bool computeSaliency( InputArray image, OutputArray saliencyMap )
    {
        if( image.empty() )
            return false;
        return computeSaliencyImpl( image, saliencyMap );
    }
    /** @brief This is the method to estimate dynamic texture parameters
     * @param img_sq The sequence of pixels of randomly sampled patches in each temporal patch. The 2d images are transformed into 1d vertical vectors
     * @param para The output dynamic texture parameters
    */
    void dynamicTextureEstimator( const Mat img_sq , DT& para );
    /** @brief This method is used to randomly sample patches from each sliding window and center area
     * @param img_sq frames sequence
     * @param index the start index of the currently processed frames
     * @param r the currently processed pixel vertical index
     * @param c the currently processed pixel horizontal index
     * @param center the center window
     * @param surround the sliding window pixels surround center window
     * @param all sliding window
    */
    void patchGenerator( const std::vector<Mat>& img_sq, unsigned index, unsigned r, unsigned c, Mat& center, Mat& surround, Mat& all );
    std::vector<Mat> saliencyMapGenerator( std::vector<Mat>, std::vector<Mat>& );
    /** @brief This is the method you can use to visualize saliency map.
     * @param _saliencyMap The saliency map to be visualized
    */
    void saliencyMapVisualize( InputArray _saliencyMap );
protected:
    bool computeSaliencyImpl( InputArray image, OutputArray saliencyMap );
    /** @brief This is the core function to calculate the KL divergence between center window and sliding window with dynamic texture parameters
     * @param para_c The center window dynamic texture parameters
     * @param para_w The sliding window dynamic texture parameters
    */
    double KLdivDT( const DT& para_c, const DT& para_w );
};

/************************************ Specific Objectness Specialized Classes ************************************/

/**
 * \brief Objectness algorithms based on [3]
 * [3] Cheng, Ming-Ming, et al. "BING: Binarized normed gradients for objectness estimation at 300fps." IEEE CVPR. 2014.
 */

/** @brief the Binarized normed gradients algorithm from @cite BING
 */
class CV_EXPORTS_W ObjectnessBING : public Objectness
{
public:

  ObjectnessBING();
  virtual ~ObjectnessBING();

  CV_WRAP static Ptr<ObjectnessBING> create()
  {
    return makePtr<ObjectnessBING>();
  }

  CV_WRAP bool computeSaliency( InputArray image, OutputArray saliencyMap )
  {
    if( image.empty() )
      return false;

    return computeSaliencyImpl( image, saliencyMap );
  }

  CV_WRAP void read();
  CV_WRAP void write() const;

  /** @brief Return the list of the rectangles' objectness value,

    in the same order as the *vector\<Vec4i\> objectnessBoundingBox* returned by the algorithm (in
    computeSaliencyImpl function). The bigger value these scores are, it is more likely to be an
    object window.
     */
  CV_WRAP std::vector<float> getobjectnessValues();

  /** @brief This is a utility function that allows to set the correct path from which the algorithm will load
    the trained model.
    @param trainingPath trained model path
     */
  CV_WRAP void setTrainingPath( const String& trainingPath );

  /** @brief This is a utility function that allows to set an arbitrary path in which the algorithm will save the
    optional results

    (ie writing on file the total number and the list of rectangles returned by objectess, one for
    each row).
    @param resultsDir results' folder path
     */
  CV_WRAP void setBBResDir( const String& resultsDir );

  CV_WRAP double getBase() const
  {
    return _base;
  }
  CV_WRAP inline void setBase(double val)
  {
    _base = val;
  }
  CV_WRAP int getNSS() const
  {
    return _NSS;
  }
  CV_WRAP void setNSS(int val)
  {
    _NSS = val;
  }
  CV_WRAP int getW() const
  {
    return _W;
  }
  CV_WRAP void setW(int val)
  {
    _W = val;
  }

protected:
  /** @brief Performs all the operations and calls all internal functions necessary for the
  accomplishment of the Binarized normed gradients algorithm.

    @param image input image. According to the needs of this specialized algorithm, the param image is a
    single *Mat*
    @param objectnessBoundingBox objectness Bounding Box vector. According to the result given by this
    specialized algorithm, the objectnessBoundingBox is a *vector\<Vec4i\>*. Each bounding box is
    represented by a *Vec4i* for (minX, minY, maxX, maxY).
     */
  bool computeSaliencyImpl( InputArray image, OutputArray objectnessBoundingBox ) CV_OVERRIDE;

private:

  class FilterTIG
  {
  public:
    void update( Mat &w );

    // For a W by H gradient magnitude map, find a W-7 by H-7 CV_32F matching score map
    Mat matchTemplate( const Mat &mag1u );

    float dot( int64_t tig1, int64_t tig2, int64_t tig4, int64_t tig8 );
    void reconstruct( Mat &w );// For illustration purpose

  private:
    static const int NUM_COMP = 2;// Number of components
    static const int D = 64;// Dimension of TIG
    int64_t _bTIGs[NUM_COMP];// Binary TIG features
    float _coeffs1[NUM_COMP];// Coefficients of binary TIG features

    // For efficiently deals with different bits in CV_8U gradient map
    float _coeffs2[NUM_COMP], _coeffs4[NUM_COMP], _coeffs8[NUM_COMP];
  };

  template<typename VT, typename ST>
  struct ValStructVec
  {
    ValStructVec();
    int size() const;
    void clear();
    void reserve( int resSz );
    void pushBack( const VT& val, const ST& structVal );
    const VT& operator ()( int i ) const;
    const ST& operator []( int i ) const;
    VT& operator ()( int i );
    ST& operator []( int i );

    void sort( bool descendOrder = true );
    const std::vector<ST> &getSortedStructVal();
    std::vector<std::pair<VT, int> > getvalIdxes();
    void append( const ValStructVec<VT, ST> &newVals, int startV = 0 );

    std::vector<ST> structVals;  // struct values
    int sz;// size of the value struct vector
    std::vector<std::pair<VT, int> > valIdxes;// Indexes after sort
    bool smaller()
    {
      return true;
    }
    std::vector<ST> sortedStructVals;
  };

  enum
  {
    MAXBGR,
    HSV,
    G
  };

  double _base, _logBase;  // base for window size quantization
  int _W;// As described in the paper: #Size, Size(_W, _H) of feature window.
  int _NSS;// Size for non-maximal suppress
  int _maxT, _minT, _numT;// The minimal and maximal dimensions of the template

  int _Clr;//
  static const char* _clrName[3];

// Names and paths to read model and to store results
  std::string _modelName, _bbResDir, _trainingPath, _resultsDir;

  std::vector<int> _svmSzIdxs;// Indexes of active size. It's equal to _svmFilters.size() and _svmReW1f.rows
  Mat _svmFilter;// Filters learned at stage I, each is a _H by _W CV_32F matrix
  FilterTIG _tigF;// TIG filter
  Mat _svmReW1f;// Re-weight parameters learned at stage II.

// List of the rectangles' objectness value, in the same order as
// the  vector<Vec4i> objectnessBoundingBox returned by the algorithm (in computeSaliencyImpl function)
  std::vector<float> objectnessValues;

private:
// functions

  inline static float LoG( float x, float y, float delta )
  {
    float d = - ( x * x + y * y ) / ( 2 * delta * delta );
    return -1.0f / ( (float) ( CV_PI ) * (delta*delta*delta*delta) ) * ( 1 + d ) * exp( d );
  }  // Laplacian of Gaussian

// Read matrix from binary file
  static bool matRead( const std::string& filename, Mat& M );

  void setColorSpace( int clr = MAXBGR );

// Load trained model.
  int loadTrainedModel();// Return -1, 0, or 1 if partial, none, or all loaded

// Get potential bounding boxes, each of which is represented by a Vec4i for (minX, minY, maxX, maxY).
// The trained model should be prepared before calling this function: loadTrainedModel() or trainStageI() + trainStageII().
// Use numDet to control the final number of proposed bounding boxes, and number of per size (scale and aspect ratio)
  void getObjBndBoxes( Mat &img3u, ValStructVec<float, Vec4i> &valBoxes, int numDetPerSize = 120 );
  void getObjBndBoxesForSingleImage( Mat img, ValStructVec<float, Vec4i> &boxes, int numDetPerSize );

  bool filtersLoaded()
  {
    int n = (int) _svmSzIdxs.size();
    return n > 0 && _svmReW1f.size() == Size( 2, n ) && _svmFilter.size() == Size( _W, _W );
  }
  void predictBBoxSI( Mat &mag3u, ValStructVec<float, Vec4i> &valBoxes, std::vector<int> &sz, int NUM_WIN_PSZ = 100, bool fast = true );
  void predictBBoxSII( ValStructVec<float, Vec4i> &valBoxes, const std::vector<int> &sz );

// Calculate the image gradient: center option as in VLFeat
  void gradientMag( Mat &imgBGR3u, Mat &mag1u );

  static void gradientRGB( Mat &bgr3u, Mat &mag1u );
  static void gradientGray( Mat &bgr3u, Mat &mag1u );
  static void gradientHSV( Mat &bgr3u, Mat &mag1u );
  static void gradientXY( Mat &x1i, Mat &y1i, Mat &mag1u );

  static inline int bgrMaxDist( const Vec3b &u, const Vec3b &v )
  {
    int b = abs( u[0] - v[0] ), g = abs( u[1] - v[1] ), r = abs( u[2] - v[2] );
    b = max( b, g );
    return max( b, r );
  }
  static inline int vecDist3b( const Vec3b &u, const Vec3b &v )
  {
    return abs( u[0] - v[0] ) + abs( u[1] - v[1] ) + abs( u[2] - v[2] );
  }

//Non-maximal suppress
  static void nonMaxSup( Mat &matchCost1f, ValStructVec<float, Point> &matchCost, int NSS = 1, int maxPoint = 50, bool fast = true );

};

//! @}

}
/* namespace saliency */
} /* namespace cv */

#endif
