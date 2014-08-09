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
 // Copyright (C) 2013, Biagio Montesano, all rights reserved.
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

#include "precomp.hpp"

#define NUM_OF_BANDS 9

//using namespace cv;
namespace cv
{

/* combinations of internal indeces for binary descriptor extractor */
static const int combinations[32][2] =
{
{ 0, 1 },
{ 0, 2 },
{ 0, 3 },
{ 0, 4 },
{ 0, 5 },
{ 0, 6 },
{ 1, 2 },
{ 1, 3 },
{ 1, 4 },
{ 1, 5 },
{ 1, 6 },
{ 2, 3 },
{ 2, 4 },
{ 2, 5 },
{ 2, 6 },
{ 2, 7 },
{ 2, 8 },
{ 3, 4 },
{ 3, 5 },
{ 3, 6 },
{ 3, 7 },
{ 3, 8 },
{ 4, 5 },
{ 4, 6 },
{ 4, 7 },
{ 4, 8 },
{ 5, 6 },
{ 5, 7 },
{ 5, 8 },
{ 6, 7 },
{ 6, 8 },
{ 7, 8 } };

/* return default parameters */
BinaryDescriptor::Params::Params()
{
  numOfOctave_ = 1;
  widthOfBand_ = 7;
  reductionRatio = 2;
  ksize_ = 5;
}

/* setters and getters */
int BinaryDescriptor::getNumOfOctaves()
{
  return params.numOfOctave_;
}

void BinaryDescriptor::setNumOfOctaves( int octaves )
{
  params.numOfOctave_ = octaves;
}

int BinaryDescriptor::getWidthOfBand()
{
  return params.widthOfBand_;
}

void BinaryDescriptor::setWidthOfBand( int width )
{
  params.widthOfBand_ = width;

  /* reserve enough space for EDLine objects and images in Gaussian pyramid */
  edLineVec_.resize( params.numOfOctave_ );
  images_sizes.resize( params.numOfOctave_ );

  for ( int i = 0; i < params.numOfOctave_; i++ )
    edLineVec_[i] = Ptr<EDLineDetector>( new EDLineDetector() );

  /* prepare a vector to host local weights F_l*/
  gaussCoefL_.resize( params.widthOfBand_ * 3 );

  /* compute center of central band (every computation involves 2-3 bands) */
  double u = ( params.widthOfBand_ * 3 - 1 ) / 2;

  /* compute exponential part of F_l */
  double sigma = ( params.widthOfBand_ * 2 + 1 ) / 2;  // (widthOfBand_*2+1)/2;
  double invsigma2 = -1 / ( 2 * sigma * sigma );

  /* compute all local weights */
  double dis;
  for ( int i = 0; i < params.widthOfBand_ * 3; i++ )
  {
    dis = i - u;
    gaussCoefL_[i] = exp( dis * dis * invsigma2 );
  }

  /* prepare a vector for global weights F_g*/
  gaussCoefG_.resize( NUM_OF_BANDS * params.widthOfBand_ );

  /* compute center of LSR */
  u = ( NUM_OF_BANDS * params.widthOfBand_ - 1 ) / 2;

  /* compute exponential part of F_g */
  sigma = u;
  invsigma2 = -1 / ( 2 * sigma * sigma );
  for ( int i = 0; i < NUM_OF_BANDS * params.widthOfBand_; i++ )
  {
    dis = i - u;
    gaussCoefG_[i] = exp( dis * dis * invsigma2 );
  }
}

int BinaryDescriptor::getReductionRatio()
{
  return params.reductionRatio;
}

void BinaryDescriptor::setReductionRatio( int rRatio )
{
  params.reductionRatio = rRatio;
}

/* read parameters from a FileNode object and store them (struct function) */
void BinaryDescriptor::Params::read( const cv::FileNode& fn )
{
  numOfOctave_ = fn["numOfOctave_"];
  widthOfBand_ = fn["widthOfBand_"];
  reductionRatio = fn["reductionRatio"];
}

/* store parameters to a FileStorage object (struct function) */
void BinaryDescriptor::Params::write( cv::FileStorage& fs ) const
{
  fs << "numOfOctave_" << numOfOctave_;
  fs << "numOfBand_" << NUM_OF_BANDS;
  fs << "widthOfBand_" << widthOfBand_;
  fs << "reductionRatio" << reductionRatio;
}

Ptr<BinaryDescriptor> BinaryDescriptor::createBinaryDescriptor()
{
  return Ptr<BinaryDescriptor>( new BinaryDescriptor() );
}

Ptr<BinaryDescriptor> BinaryDescriptor::createBinaryDescriptor( Params parameters )
{
  return Ptr<BinaryDescriptor>( new BinaryDescriptor( parameters ) );
}

/* construct a BinaryDescrptor object and compute external private parameters */
BinaryDescriptor::BinaryDescriptor( const BinaryDescriptor::Params &parameters ) :
    params( parameters )
{
  /* reserve enough space for EDLine objects and images in Gaussian pyramid */
  edLineVec_.resize( params.numOfOctave_ );
  images_sizes.resize( params.numOfOctave_ );

  for ( int i = 0; i < params.numOfOctave_; i++ )
    edLineVec_[i] = Ptr<EDLineDetector>( new EDLineDetector() );

  /* prepare a vector to host local weights F_l*/
  gaussCoefL_.resize( params.widthOfBand_ * 3 );

  /* compute center of central band (every computation involves 2-3 bands) */
  double u = ( params.widthOfBand_ * 3 - 1 ) / 2;

  /* compute exponential part of F_l */
  double sigma = ( params.widthOfBand_ * 2 + 1 ) / 2;  // (widthOfBand_*2+1)/2;
  double invsigma2 = -1 / ( 2 * sigma * sigma );

  /* compute all local weights */
  double dis;
  for ( int i = 0; i < params.widthOfBand_ * 3; i++ )
  {
    dis = i - u;
    gaussCoefL_[i] = exp( dis * dis * invsigma2 );
  }

  /* prepare a vector for global weights F_g*/
  gaussCoefG_.resize( NUM_OF_BANDS * params.widthOfBand_ );

  /* compute center of LSR */
  u = ( NUM_OF_BANDS * params.widthOfBand_ - 1 ) / 2;

  /* compute exponential part of F_g */
  sigma = u;
  invsigma2 = -1 / ( 2 * sigma * sigma );
  for ( int i = 0; i < NUM_OF_BANDS * params.widthOfBand_; i++ )
  {
    dis = i - u;
    gaussCoefG_[i] = exp( dis * dis * invsigma2 );
  }
}

/* definition of operator () */
void BinaryDescriptor::operator()( InputArray image, InputArray mask, CV_OUT std::vector<KeyLine>& keylines, OutputArray descriptors,
                                   bool useProvidedKeyLines, bool returnFloatDescr ) const
{

  /* create some matrix objects */
  cv::Mat imageMat, maskMat, descrMat;

  /* store reference to input matrices */
  imageMat = image.getMat();
  maskMat = mask.getMat();

  /* require drawing KeyLines detection if demanded */
  if( !useProvidedKeyLines )
  {
    keylines.clear();
    BinaryDescriptor *bn = const_cast<BinaryDescriptor*>( this );
    bn->edLineVec_.clear();
    bn->edLineVec_.resize( params.numOfOctave_ );

    for ( int i = 0; i < params.numOfOctave_; i++ )
      bn->edLineVec_[i] = Ptr<EDLineDetector>( new EDLineDetector() );

    detectImpl( imageMat, keylines, maskMat );

  }

  /* initialize output matrix */
  //descriptors.create( Size( 32, (int) keylines.size() ), CV_8UC1 );
  /* store reference to output matrix */
  //descrMat = descriptors.getMat();
  /* compute descriptors */
  if( !useProvidedKeyLines )
    computeImpl( imageMat, keylines, descrMat, returnFloatDescr, true );

  else
    computeImpl( imageMat, keylines, descrMat, returnFloatDescr, false );

  descrMat.copyTo( descriptors );
}

BinaryDescriptor::~BinaryDescriptor()
{

}

/* read parameters from a FileNode object and store them (class function ) */
void BinaryDescriptor::read( const cv::FileNode& fn )
{
  params.read( fn );
}

/* store parameters to a FileStorage object (class function) */
void BinaryDescriptor::write( cv::FileStorage& fs ) const
{
  params.write( fs );
}

/* return norm mode */
int BinaryDescriptor::defaultNorm() const
{
  return NORM_HAMMING;
}

/* return data type */
int BinaryDescriptor::descriptorType() const
{
  return CV_8U;
}

/*return descriptor size */
int BinaryDescriptor::descriptorSize() const
{
  return 32 * 8;
}

/* power function with error management */
static inline int get2Pow( int i )
{
  if( i >= 0 && i <= 7 )
    return (int) pow( 2, (double) i );

  else
  {
    throw std::runtime_error( "Invalid power argument" );
  }
}

/* compute Gaussian pyramids */
void BinaryDescriptor::computeGaussianPyramid( const Mat& image, const int numOctaves )
{
  /* clear class fields */
  images_sizes.clear();
  octaveImages.clear();

  /* insert input image into pyramid */
  cv::Mat currentMat = image.clone();
  cv::GaussianBlur( currentMat, currentMat, cv::Size( 5, 5 ), 1 );
  octaveImages.push_back( currentMat );
  images_sizes.push_back( currentMat.size() );

  /* fill Gaussian pyramid */
  for ( int pyrCounter = 1; pyrCounter < numOctaves; pyrCounter++ )
  {
    /* compute and store next image in pyramid and its size */
    pyrDown( currentMat, currentMat, Size( currentMat.cols / params.reductionRatio, currentMat.rows / params.reductionRatio ) );
    octaveImages.push_back( currentMat );
    images_sizes.push_back( currentMat.size() );
  }
}

/* compute Sobel's derivatives */
void BinaryDescriptor::computeSobel( const cv::Mat& image, const int numOctaves )
{

  /* compute Gaussian pyramids */
  computeGaussianPyramid( image, numOctaves );

  /* reinitialize class structures */
  dxImg_vector.clear();
  dyImg_vector.clear();

//  dxImg_vector.resize( params.numOfOctave_ );
//  dyImg_vector.resize( params.numOfOctave_ );

  dxImg_vector.resize( octaveImages.size() );
  dyImg_vector.resize( octaveImages.size() );

  /* compute derivatives */
  for ( size_t sobelCnt = 0; sobelCnt < octaveImages.size(); sobelCnt++ )
  {
    dxImg_vector[sobelCnt].create( images_sizes[sobelCnt].height, images_sizes[sobelCnt].width, CV_16SC1 );
    dyImg_vector[sobelCnt].create( images_sizes[sobelCnt].height, images_sizes[sobelCnt].width, CV_16SC1 );

    cv::Sobel( octaveImages[sobelCnt], dxImg_vector[sobelCnt], CV_16SC1, 1, 0, 3 );
    cv::Sobel( octaveImages[sobelCnt], dyImg_vector[sobelCnt], CV_16SC1, 0, 1, 3 );
  }
}

/* utility function for conversion of an LBD descriptor to its binary representation */
unsigned char BinaryDescriptor::binaryConversion( float* f1, float* f2 )
{
  uchar result = 0;
  for ( int i = 0; i < 8; i++ )
  {
    if( f1[i] > f2[i] )
      result += (uchar) get2Pow( i );
  }

  return result;

}

/* requires line detection (only one image) */
void BinaryDescriptor::detect( const Mat& image, CV_OUT std::vector<KeyLine>& keylines, const Mat& mask )
{
  if( image.data == NULL )
  {
    std::cout << "Error: input image for detection is empty" << std::endl;
    return;
  }

  if( mask.data != NULL && ( mask.size() != image.size() || mask.type() != CV_8UC1 ) )
    throw std::runtime_error( "Mask error while detecting lines: please check its dimensions and that data type is CV_8UC1" );

  else
    detectImpl( image, keylines, mask );
}

/* requires line detection (more than one image) */
void BinaryDescriptor::detect( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, const std::vector<Mat>& masks ) const
{

  if( images.size() == 0 )
  {
    std::cout << "Error: input image for detection is empty" << std::endl;
    return;
  }

  /* detect lines from each image */
  for ( size_t counter = 0; counter < images.size(); counter++ )
  {
    if( masks[counter].data != NULL && ( masks[counter].size() != images[counter].size() || masks[counter].type() != CV_8UC1 ) )
      throw std::runtime_error( "Masks error while detecting lines: please check their dimensions and that data types are CV_8UC1" );

    else
      detectImpl( images[counter], keylines[counter], masks[counter] );
  }
}

void BinaryDescriptor::detectImpl( const Mat& imageSrc, std::vector<KeyLine>& keylines, const Mat& mask ) const
{

  cv::Mat image;
  if( imageSrc.channels() != 1 )
  {
    cvtColor( imageSrc, image, COLOR_BGR2GRAY );
  }
  else
    image = imageSrc.clone();

  /*check whether image depth is different from 0 */
  if( image.depth() != 0 )
    throw std::runtime_error( "Warning, depth image!= 0" );

  /* create a pointer to self */
  BinaryDescriptor *bn = const_cast<BinaryDescriptor*>( this );

  /* detect and arrange lines across octaves */
  ScaleLines sl;
  bn->OctaveKeyLines( image, sl );

  /* fill KeyLines vector */
  for ( int i = 0; i < (int) sl.size(); i++ )
  {
    for ( size_t j = 0; j < sl[i].size(); j++ )
    {
      /* get current line */
      OctaveSingleLine osl = sl[i][j];

      /* create a KeyLine object */
      KeyLine kl;

      /* fill KeyLine's fields */
      kl.startPointX = osl.startPointX;  //extremes[0];
      kl.startPointY = osl.startPointY;  //extremes[1];
      kl.endPointX = osl.endPointX;  //extremes[2];
      kl.endPointY = osl.endPointY;  //extremes[3];
      kl.sPointInOctaveX = osl.sPointInOctaveX;
      kl.sPointInOctaveY = osl.sPointInOctaveY;
      kl.ePointInOctaveX = osl.ePointInOctaveX;
      kl.ePointInOctaveY = osl.ePointInOctaveY;
      kl.lineLength = osl.lineLength;
      kl.numOfPixels = osl.numOfPixels;

      kl.angle = osl.direction;
      kl.class_id = i;
      kl.octave = osl.octaveCount;
      kl.size = ( osl.endPointX - osl.startPointX ) * ( osl.endPointY - osl.startPointY );
      kl.response = osl.lineLength / max( images_sizes[osl.octaveCount].width, images_sizes[osl.octaveCount].height );
      kl.pt = Point2f( ( osl.endPointX + osl.startPointX ) / 2, ( osl.endPointY + osl.startPointY ) / 2 );

      /* store KeyLine */
      keylines.push_back( kl );
    }

  }

  /* delete undesired KeyLines, according to input mask */
  if( !mask.empty() )
  {
    for ( size_t keyCounter = 0; keyCounter < keylines.size(); keyCounter++ )
    {
      KeyLine kl = keylines[keyCounter];
      if( mask.at<uchar>( (int) kl.startPointY, (int) kl.startPointX ) == 0 && mask.at<uchar>( (int) kl.endPointY, (int) kl.endPointX ) == 0 )
        keylines.erase( keylines.begin() + keyCounter );
    }
  }

}

/* requires descriptors computation (only one image) */
void BinaryDescriptor::compute( const Mat& image, CV_OUT CV_IN_OUT std::vector<KeyLine>& keylines, CV_OUT Mat& descriptors,
                                bool returnFloatDescr ) const
{
  computeImpl( image, keylines, descriptors, returnFloatDescr, false );
}

/* requires descriptors computation (more than one image) */
void BinaryDescriptor::compute( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, std::vector<Mat>& descriptors,
                                bool returnFloatDescr ) const
{
  for ( size_t i = 0; i < images.size(); i++ )
    computeImpl( images[i], keylines[i], descriptors[i], returnFloatDescr, false );
}

/* implementation of descriptors computation */
void BinaryDescriptor::computeImpl( const Mat& imageSrc, std::vector<KeyLine>& keylines, Mat& descriptors, bool returnFloatDescr,
                                    bool useDetectionData ) const
{
  /* convert input image to gray scale */
  cv::Mat image;
  if( imageSrc.channels() != 1 )
    cvtColor( imageSrc, image, COLOR_BGR2GRAY );
  else
    image = imageSrc.clone();

  /*check whether image's depth is different from 0 */
  if( image.depth() != 0 )
    throw std::runtime_error( "Error, depth of image != 0" );

  /* keypoints list can't be empty */
  if( keylines.size() == 0 )
  {
    std::cout << "Error: keypoint list is empty" << std::endl;
    return;
  }

  BinaryDescriptor* bd = const_cast<BinaryDescriptor*>( this );

  /* get maximum class_id and octave*/
  int numLines = 0;
  int octaveIndex = -1;
  for ( size_t l = 0; l < keylines.size(); l++ )
  {
    if( keylines[l].class_id > numLines )
      numLines = keylines[l].class_id;

    if( keylines[l].octave > octaveIndex )
      octaveIndex = keylines[l].octave;
  }

  if( !useDetectionData )
    bd->computeSobel( image, octaveIndex + 1 );

  /* create a ScaleLines object */
  OctaveSingleLine fictiousOSL;
//  fictiousOSL.octaveCount = params.numOfOctave_ + 1;
//  LinesVec lv( params.numOfOctave_, fictiousOSL );
  fictiousOSL.octaveCount = octaveIndex + 1;
  LinesVec lv( octaveIndex + 1, fictiousOSL );
  ScaleLines sl( numLines + 1, lv );

  /* create a map to record association between KeyLines and their position
   in ScaleLines vector */
  std::map<std::pair<int, int>, int> correspondences;

  /* fill ScaleLines object */
  for ( size_t slCounter = 0; slCounter < keylines.size(); slCounter++ )
  {
    /* get a KeyLine object and create a new line */
    KeyLine kl = keylines[slCounter];
    OctaveSingleLine osl;

    /* insert data in newly created line */
    osl.startPointX = kl.startPointX;
    osl.startPointY = kl.startPointY;
    osl.endPointX = kl.endPointX;
    osl.endPointY = kl.endPointY;
    osl.sPointInOctaveX = kl.sPointInOctaveX;
    osl.sPointInOctaveY = kl.sPointInOctaveY;
    osl.ePointInOctaveX = kl.ePointInOctaveX;
    osl.ePointInOctaveY = kl.ePointInOctaveY;
    osl.lineLength = kl.lineLength;
    osl.numOfPixels = kl.numOfPixels;
    osl.salience = kl.response;

    osl.direction = kl.angle;
    osl.octaveCount = kl.octave;

    /* store new line */
    sl[kl.class_id][kl.octave] = osl;

    /* update map */
    int id = kl.class_id;
    int oct = kl.octave;
    correspondences.insert( std::pair<std::pair<int, int>, int>( std::pair<int, int>( id, oct ), slCounter ) );
  }

  /* delete useless OctaveSingleLines */
  for ( size_t i = 0; i < sl.size(); i++ )
  {
    for ( size_t j = 0; j < sl[i].size(); j++ )
    {
      //if( (int) ( sl[i][j] ).octaveCount > params.numOfOctave_ )
      if( (int) ( sl[i][j] ).octaveCount > octaveIndex )
        ( sl[i] ).erase( ( sl[i] ).begin() + j );
    }
  }

  /* compute LBD descriptors */
  bd->computeLBD( sl, useDetectionData );

  /* resize output matrix */
  if( !returnFloatDescr )
    descriptors = cv::Mat( (int) keylines.size(), 32, CV_8UC1 );

  else
    descriptors = cv::Mat( (int) keylines.size(), NUM_OF_BANDS * 8, CV_32FC1 );

  /* fill output matrix with descriptors */
  for ( int k = 0; k < (int) sl.size(); k++ )
  {
    for ( int lineC = 0; lineC < (int) sl[k].size(); lineC++ )
    {
      /* get original index of keypoint */
      int lineOctave = ( sl[k][lineC] ).octaveCount;
      int originalIndex = correspondences.find( std::pair<int, int>( k, lineOctave ) )->second;

      if( !returnFloatDescr )
      {
        /* get a pointer to correspondent row in output matrix */
        uchar* pointerToRow = descriptors.ptr( originalIndex );

        /* get LBD data */
        float* desVec = sl[k][lineC].descriptor.data();

        /* fill current row with binary descriptor */
        for ( int comb = 0; comb < 32; comb++ )
        {
          *pointerToRow = bd->binaryConversion( &desVec[8 * combinations[comb][0]], &desVec[8 * combinations[comb][1]] );
          pointerToRow++;
        }
      }

      else
      {
        /* get a pointer to correspondent row in output matrix */
        float* pointerToRow = descriptors.ptr<float>( originalIndex );

        /* get LBD data */
        std::vector<float> desVec = sl[k][lineC].descriptor;

        for ( int count = 0; count < (int) desVec.size(); count++ )
        {
          *pointerToRow = desVec[count];
          pointerToRow++;
        }
      }

    }
  }

}

int BinaryDescriptor::OctaveKeyLines( cv::Mat& image, ScaleLines &keyLines )
{

  /* final number of extracted lines */
  unsigned int numOfFinalLine = 0;

  /* sigma values and reduction factor used in Gaussian pyramids */
  float preSigma2 = 0;  //orignal image is not blurred, has zero sigma;
  float curSigma2 = 1.0;  //[sqrt(2)]^0=1;
  double factor = sqrt( 2 );  //the down sample factor between connective two octave images

  /* loop over number of octaves */
  for ( int octaveCount = 0; octaveCount < params.numOfOctave_; octaveCount++ )
  {
    /* matrix storing results from blurring processes */
    cv::Mat blur;

    /* apply Gaussian blur */
    float increaseSigma = sqrt( curSigma2 - preSigma2 );
    cv::GaussianBlur( image, blur, cv::Size( params.ksize_, params.ksize_ ), increaseSigma );
    images_sizes[octaveCount] = blur.size();

    /* for current octave, extract lines */
    if( ( edLineVec_[octaveCount]->EDline( blur ) ) != 1 )
    {
      return -1;
    }

    /* update number of total extracted lines */
    numOfFinalLine += edLineVec_[octaveCount]->lines_.numOfLines;

    /* resize image for next level of pyramid */
    cv::resize( blur, image, cv::Size(), ( 1.f / factor ), ( 1.f / factor ) );

    /* update sigma values */
    preSigma2 = curSigma2;
    curSigma2 = curSigma2 * 2;

  } /* end of loop over number of octaves */

  /* prepare a vector to store octave information associated to extracted lines */
  std::vector<OctaveLine> octaveLines( numOfFinalLine );

  /* set lines' counter to 0 for reuse */
  numOfFinalLine = 0;

  /* counter to give a unique ID to lines in LineVecs */
  unsigned int lineIDInScaleLineVec = 0;

  /* floats to compute lines' lengths */
  float dx, dy;

  /* loop over lines extracted from scale 0 (original image) */
  for ( unsigned int lineCurId = 0; lineCurId < edLineVec_[0]->lines_.numOfLines; lineCurId++ )
  {
    /* FOR CURRENT LINE: */

    /* set octave from which it was extracted */
    octaveLines[numOfFinalLine].octaveCount = 0;
    /* set ID within its octave */
    octaveLines[numOfFinalLine].lineIDInOctave = lineCurId;
    /* set a unique ID among all lines extracted in all octaves */
    octaveLines[numOfFinalLine].lineIDInScaleLineVec = lineIDInScaleLineVec;

    /* compute absolute value of difference between X coordinates of line's extreme points */
    dx = fabs( edLineVec_[0]->lineEndpoints_[lineCurId][0] - edLineVec_[0]->lineEndpoints_[lineCurId][2] );
    /* compute absolute value of difference between Y coordinates of line's extreme points */
    dy = fabs( edLineVec_[0]->lineEndpoints_[lineCurId][1] - edLineVec_[0]->lineEndpoints_[lineCurId][3] );
    /* compute line's length */
    octaveLines[numOfFinalLine].lineLength = sqrt( dx * dx + dy * dy );

    /* update counters */
    numOfFinalLine++;
    lineIDInScaleLineVec++;
  }

  /* create and fill an array to store scale factors */
  float *scale = new float[params.numOfOctave_];
  scale[0] = 1;
  for ( int octaveCount = 1; octaveCount < params.numOfOctave_; octaveCount++ )
  {
    scale[octaveCount] = (float) ( factor * scale[octaveCount - 1] );
  }

  /* some variables' declarations */
  float rho1, rho2, tempValue;
  float direction, near, length;
  unsigned int octaveID, lineIDInOctave;

  /*more than one octave image, organize lines in scale space.
   *lines corresponding to the same line in octave images should have the same index in the ScaleLineVec */
  if( params.numOfOctave_ > 1 )
  {
    /* some other variables' declarations */
    double twoPI = 2 * M_PI;
    unsigned int closeLineID = 0;
    float endPointDis, minEndPointDis, minLocalDis, maxLocalDis;
    float lp0, lp1, lp2, lp3, np0, np1, np2, np3;

    /* loop over list of octaves */
    for ( int octaveCount = 1; octaveCount < params.numOfOctave_; octaveCount++ )
    {
      /*for each line in current octave image, find their corresponding lines in the octaveLines,
       *give them the same value of lineIDInScaleLineVec*/

      /* loop over list of lines extracted from current octave */
      for ( unsigned int lineCurId = 0; lineCurId < edLineVec_[octaveCount]->lines_.numOfLines; lineCurId++ )
      {
        /* get (scaled) known term from equation of current line */
        rho1 = (float) ( scale[octaveCount] * fabs( edLineVec_[octaveCount]->lineEquations_[lineCurId][2] ) );

        /*nearThreshold depends on the distance of the image coordinate origin to current line.
         *so nearThreshold = rho1 * nearThresholdRatio, where nearThresholdRatio = 1-cos(10*pi/180) = 0.0152*/
        tempValue = (float) ( rho1 * 0.0152 );
        float nearThreshold = ( tempValue > 6 ) ? ( tempValue ) : 6;
        nearThreshold = ( nearThreshold < 12 ) ? nearThreshold : 12;

        /* compute scaled lenght of current line */
        dx = fabs( edLineVec_[octaveCount]->lineEndpoints_[lineCurId][0] - edLineVec_[octaveCount]->lineEndpoints_[lineCurId][2] );  //x1-x2
        dy = fabs( edLineVec_[octaveCount]->lineEndpoints_[lineCurId][1] - edLineVec_[octaveCount]->lineEndpoints_[lineCurId][3] );  //y1-y2
        length = scale[octaveCount] * sqrt( dx * dx + dy * dy );

        minEndPointDis = 12;
        /* loop over the octave representations of all lines */
        for ( unsigned int lineNextId = 0; lineNextId < numOfFinalLine; lineNextId++ )
        {
          /* if a line from same octave is encountered,
           a comparison with it shouldn't be considered */
          octaveID = octaveLines[lineNextId].octaveCount;
          if( (int) octaveID == octaveCount )
          {  //lines in the same layer of octave image should not be compared.
            break;
          }

          /* take ID in octave of line to be compared */
          lineIDInOctave = octaveLines[lineNextId].lineIDInOctave;

          /*first check whether current line and next line are parallel.
           *If line1:a1*x+b1*y+c1=0 and line2:a2*x+b2*y+c2=0 are parallel, then
           *-a1/b1=-a2/b2, i.e., a1b2=b1a2.
           *we define parallel=fabs(a1b2-b1a2)
           *note that, in EDLine class, we have normalized the line equations
           *to make a1^2+ b1^2 = a2^2+ b2^2 = 1*/
          direction = fabs( edLineVec_[octaveCount]->lineDirection_[lineCurId] - edLineVec_[octaveID]->lineDirection_[lineIDInOctave] );

          /* the angle between two lines are larger than 10degrees
           (i.e. 10*pi/180=0.1745), they are not close to parallel */
          if( direction > 0.1745 && ( twoPI - direction > 0.1745 ) )
          {
            continue;
          }
          /*now check whether current line and next line are near to each other.
           *If line1:a1*x+b1*y+c1=0 and line2:a2*x+b2*y+c2=0 are near in image, then
           *rho1 = |a1*0+b1*0+c1|/sqrt(a1^2+b1^2) and rho2 = |a2*0+b2*0+c2|/sqrt(a2^2+b2^2) should close.
           *In our case, rho1 = |c1| and rho2 = |c2|, because sqrt(a1^2+b1^2) = sqrt(a2^2+b2^2) = 1;
           *note that, lines are in different octave images, so we define near =  fabs(scale*rho1 - rho2) or
           *where scale is the scale factor between to octave images*/

          /* get known term from equation to be compared */
          rho2 = (float) ( scale[octaveID] * fabs( edLineVec_[octaveID]->lineEquations_[lineIDInOctave][2] ) );
          /* compute difference between known ters */
          near = fabs( rho1 - rho2 );

          /* two lines are not close in the image */
          if( near > nearThreshold )
          {
            continue;
          }

          /*now check the end points distance between two lines, the scale of  distance is in the original image size.
           * find the minimal and maximal end points distance*/

          /* get the extreme points of the two lines */
          lp0 = scale[octaveCount] * edLineVec_[octaveCount]->lineEndpoints_[lineCurId][0];
          lp1 = scale[octaveCount] * edLineVec_[octaveCount]->lineEndpoints_[lineCurId][1];
          lp2 = scale[octaveCount] * edLineVec_[octaveCount]->lineEndpoints_[lineCurId][2];
          lp3 = scale[octaveCount] * edLineVec_[octaveCount]->lineEndpoints_[lineCurId][3];
          np0 = scale[octaveID] * edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][0];
          np1 = scale[octaveID] * edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][1];
          np2 = scale[octaveID] * edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][2];
          np3 = scale[octaveID] * edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][3];

          /* get the distance between the two leftmost extremes of lines
           L1(0,1)<->L2(0,1) */
          dx = lp0 - np0;
          dy = lp1 - np1;
          endPointDis = sqrt( dx * dx + dy * dy );

          /* set momentaneously min and max distance between lines to
           the one between left extremes */
          minLocalDis = endPointDis;
          maxLocalDis = endPointDis;

          /* compute distance between right extremes
           L1(2,3)<->L2(2,3) */
          dx = lp2 - np2;
          dy = lp3 - np3;
          endPointDis = sqrt( dx * dx + dy * dy );

          /* update (if necessary) min and max distance between lines */
          minLocalDis = ( endPointDis < minLocalDis ) ? endPointDis : minLocalDis;
          maxLocalDis = ( endPointDis > maxLocalDis ) ? endPointDis : maxLocalDis;

          /* compute distance between left extreme of current line and
           right extreme of line to be compared
           L1(0,1)<->L2(2,3) */
          dx = lp0 - np2;
          dy = lp1 - np3;
          endPointDis = sqrt( dx * dx + dy * dy );

          /* update (if necessary) min and max distance between lines */
          minLocalDis = ( endPointDis < minLocalDis ) ? endPointDis : minLocalDis;
          maxLocalDis = ( endPointDis > maxLocalDis ) ? endPointDis : maxLocalDis;

          /* compute distance between right extreme of current line and
           left extreme of line to be compared
           L1(2,3)<->L2(0,1) */
          dx = lp2 - np0;
          dy = lp3 - np1;
          endPointDis = sqrt( dx * dx + dy * dy );

          /* update (if necessary) min and max distance between lines */
          minLocalDis = ( endPointDis < minLocalDis ) ? endPointDis : minLocalDis;
          maxLocalDis = ( endPointDis > maxLocalDis ) ? endPointDis : maxLocalDis;

          /* check whether conditions for considering line to be compared
           worth to be inserted in the same LineVec are satisfied */
          if( ( maxLocalDis < 0.8 * ( length + octaveLines[lineNextId].lineLength ) ) && ( minLocalDis < minEndPointDis ) )
          {  //keep the closest line
            minEndPointDis = minLocalDis;
            closeLineID = lineNextId;
          }
        }

        /* add current line into octaveLines */
        if( minEndPointDis < 12 )
        {
          octaveLines[numOfFinalLine].lineIDInScaleLineVec = octaveLines[closeLineID].lineIDInScaleLineVec;
        }
        else
        {
          octaveLines[numOfFinalLine].lineIDInScaleLineVec = lineIDInScaleLineVec;
          lineIDInScaleLineVec++;
        }
        octaveLines[numOfFinalLine].octaveCount = octaveCount;
        octaveLines[numOfFinalLine].lineIDInOctave = lineCurId;
        octaveLines[numOfFinalLine].lineLength = length;
        numOfFinalLine++;
      }
    }  //end for(unsigned int octaveCount = 1; octaveCount<numOfOctave_; octaveCount++)
  }  //end if(numOfOctave_>1)

  ////////////////////////////////////
  //Reorganize the detected lines into keyLines
  keyLines.clear();
  keyLines.resize( lineIDInScaleLineVec );
  unsigned int tempID;
  float s1, e1, s2, e2;
  bool shouldChange;
  OctaveSingleLine singleLine;
  for ( unsigned int lineID = 0; lineID < numOfFinalLine; lineID++ )
  {
    lineIDInOctave = octaveLines[lineID].lineIDInOctave;
    octaveID = octaveLines[lineID].octaveCount;
    direction = edLineVec_[octaveID]->lineDirection_[lineIDInOctave];
    singleLine.octaveCount = octaveID;
    singleLine.direction = direction;
    singleLine.lineLength = octaveLines[lineID].lineLength;
    singleLine.salience = edLineVec_[octaveID]->lineSalience_[lineIDInOctave];
    singleLine.numOfPixels = edLineVec_[octaveID]->lines_.sId[lineIDInOctave + 1] - edLineVec_[octaveID]->lines_.sId[lineIDInOctave];
    //decide the start point and end point
    shouldChange = false;
    s1 = edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][0];  //sx
    s2 = edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][1];  //sy
    e1 = edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][2];  //ex
    e2 = edLineVec_[octaveID]->lineEndpoints_[lineIDInOctave][3];  //ey
    dx = e1 - s1;  //ex-sx
    dy = e2 - s2;  //ey-sy
    if( direction >= -0.75 * M_PI && direction < -0.25 * M_PI )
    {
      if( dy > 0 )
      {
        shouldChange = true;
      }
    }
    if( direction >= -0.25 * M_PI && direction < 0.25 * M_PI )
    {
      if( dx < 0 )
      {
        shouldChange = true;
      }
    }
    if( direction >= 0.25 * M_PI && direction < 0.75 * M_PI )
    {
      if( dy < 0 )
      {
        shouldChange = true;
      }
    }
    if( ( direction >= 0.75 * M_PI && direction < M_PI ) || ( direction >= -M_PI && direction < -0.75 * M_PI ) )
    {
      if( dx > 0 )
      {
        shouldChange = true;
      }
    }
    tempValue = scale[octaveID];
    if( shouldChange )
    {
      singleLine.sPointInOctaveX = e1;
      singleLine.sPointInOctaveY = e2;
      singleLine.ePointInOctaveX = s1;
      singleLine.ePointInOctaveY = s2;
      singleLine.startPointX = tempValue * e1;
      singleLine.startPointY = tempValue * e2;
      singleLine.endPointX = tempValue * s1;
      singleLine.endPointY = tempValue * s2;
    }
    else
    {
      singleLine.sPointInOctaveX = s1;
      singleLine.sPointInOctaveY = s2;
      singleLine.ePointInOctaveX = e1;
      singleLine.ePointInOctaveY = e2;
      singleLine.startPointX = tempValue * s1;
      singleLine.startPointY = tempValue * s2;
      singleLine.endPointX = tempValue * e1;
      singleLine.endPointY = tempValue * e2;
    }
    tempID = octaveLines[lineID].lineIDInScaleLineVec;
    keyLines[tempID].push_back( singleLine );
  }

  delete[] scale;
  return 1;
}

int BinaryDescriptor::computeLBD( ScaleLines &keyLines, bool useDetectionData )
{
  //the default length of the band is the line length.
  short numOfFinalLine = (short) keyLines.size();
  float *dL = new float[2];  //line direction cos(dir), sin(dir)
  float *dO = new float[2];  //the clockwise orthogonal vector of line direction.
  short heightOfLSP = (short) ( params.widthOfBand_ * NUM_OF_BANDS );  //the height of line support region;
  short descriptor_size = NUM_OF_BANDS * 8;  //each band, we compute the m( pgdL, ngdL,  pgdO, ngdO) and std( pgdL, ngdL,  pgdO, ngdO);
  float pgdLRowSum;  //the summation of {g_dL |g_dL>0 } for each row of the region;
  float ngdLRowSum;  //the summation of {g_dL |g_dL<0 } for each row of the region;
  float pgdL2RowSum;  //the summation of {g_dL^2 |g_dL>0 } for each row of the region;
  float ngdL2RowSum;  //the summation of {g_dL^2 |g_dL<0 } for each row of the region;
  float pgdORowSum;  //the summation of {g_dO |g_dO>0 } for each row of the region;
  float ngdORowSum;  //the summation of {g_dO |g_dO<0 } for each row of the region;
  float pgdO2RowSum;  //the summation of {g_dO^2 |g_dO>0 } for each row of the region;
  float ngdO2RowSum;  //the summation of {g_dO^2 |g_dO<0 } for each row of the region;

  float *pgdLBandSum = new float[NUM_OF_BANDS];  //the summation of {g_dL |g_dL>0 } for each band of the region;
  float *ngdLBandSum = new float[NUM_OF_BANDS];  //the summation of {g_dL |g_dL<0 } for each band of the region;
  float *pgdL2BandSum = new float[NUM_OF_BANDS];  //the summation of {g_dL^2 |g_dL>0 } for each band of the region;
  float *ngdL2BandSum = new float[NUM_OF_BANDS];  //the summation of {g_dL^2 |g_dL<0 } for each band of the region;
  float *pgdOBandSum = new float[NUM_OF_BANDS];  //the summation of {g_dO |g_dO>0 } for each band of the region;
  float *ngdOBandSum = new float[NUM_OF_BANDS];  //the summation of {g_dO |g_dO<0 } for each band of the region;
  float *pgdO2BandSum = new float[NUM_OF_BANDS];  //the summation of {g_dO^2 |g_dO>0 } for each band of the region;
  float *ngdO2BandSum = new float[NUM_OF_BANDS];  //the summation of {g_dO^2 |g_dO<0 } for each band of the region;

  short numOfBitsBand = NUM_OF_BANDS * sizeof(float);
  short lengthOfLSP;  //the length of line support region, varies with lines
  short halfHeight = ( heightOfLSP - 1 ) / 2;
  short halfWidth;
  short bandID;
  float coefInGaussion;
  float lineMiddlePointX, lineMiddlePointY;
  float sCorX, sCorY, sCorX0, sCorY0;
  short tempCor, xCor, yCor;  //pixel coordinates in image plane
  short dx, dy;
  float gDL;  //store the gradient projection of pixels in support region along dL vector
  float gDO;  //store the gradient projection of pixels in support region along dO vector
  short imageWidth, imageHeight, realWidth;
  short *pdxImg, *pdyImg;
  float *desVec;

  short sameLineSize;
  short octaveCount;
  OctaveSingleLine *pSingleLine;
  /* loop over list of LineVec */
  for ( short lineIDInScaleVec = 0; lineIDInScaleVec < numOfFinalLine; lineIDInScaleVec++ )
  {
    sameLineSize = (short) ( keyLines[lineIDInScaleVec].size() );
    /* loop over current LineVec's lines */
    for ( short lineIDInSameLine = 0; lineIDInSameLine < sameLineSize; lineIDInSameLine++ )
    {
      /* get a line in current LineVec and its original ID in its octave */
      pSingleLine = & ( keyLines[lineIDInScaleVec][lineIDInSameLine] );
      octaveCount = (short) pSingleLine->octaveCount;

      if( useDetectionData )
      {
        /* retrieve associated dxImg and dyImg */
        pdxImg = edLineVec_[octaveCount]->dxImg_.ptr<short>();
        pdyImg = edLineVec_[octaveCount]->dyImg_.ptr<short>();

        /* get image size to work on from real one */
        realWidth = (short) edLineVec_[octaveCount]->imageWidth;
        imageWidth = realWidth - 1;
        imageHeight = (short) ( edLineVec_[octaveCount]->imageHeight - 1 );
      }

      else
      {
        /* retrieve associated dxImg and dyImg */
        pdxImg = dxImg_vector[octaveCount].ptr<short>();
        pdyImg = dyImg_vector[octaveCount].ptr<short>();

        /* get image size to work on from real one */
        realWidth = (short) images_sizes[octaveCount].width;
        imageWidth = realWidth - 1;
        imageHeight = (short) ( images_sizes[octaveCount].height - 1 );
      }

      /* initialize memory areas */
      memset( pgdLBandSum, 0, numOfBitsBand );
      memset( ngdLBandSum, 0, numOfBitsBand );
      memset( pgdL2BandSum, 0, numOfBitsBand );
      memset( ngdL2BandSum, 0, numOfBitsBand );
      memset( pgdOBandSum, 0, numOfBitsBand );
      memset( ngdOBandSum, 0, numOfBitsBand );
      memset( pgdO2BandSum, 0, numOfBitsBand );
      memset( ngdO2BandSum, 0, numOfBitsBand );

      /* get length of line and its half */
      lengthOfLSP = (short) keyLines[lineIDInScaleVec][lineIDInSameLine].numOfPixels;
      halfWidth = ( lengthOfLSP - 1 ) / 2;

      /* get middlepoint of line */
      lineMiddlePointX = (float) ( 0.5 * ( pSingleLine->sPointInOctaveX + pSingleLine->ePointInOctaveX ) );
      lineMiddlePointY = (float) ( 0.5 * ( pSingleLine->sPointInOctaveY + pSingleLine->ePointInOctaveY ) );

      /*1.rotate the local coordinate system to the line direction (direction is the angle
       between positive line direction and positive X axis)
       *2.compute the gradient projection of pixels in line support region*/

      /* get the vector representing original image reference system after rotation to aligh with
       line's direction */
      dL[0] = cos( pSingleLine->direction );
      dL[1] = sin( pSingleLine->direction );

      /* set the clockwise orthogonal vector of line direction */
      dO[0] = -dL[1];
      dO[1] = dL[0];

      /* get rotated reference frame */
      sCorX0 = -dL[0] * halfWidth + dL[1] * halfHeight + lineMiddlePointX;  //hID =0; wID = 0;
      sCorY0 = -dL[1] * halfWidth - dL[0] * halfHeight + lineMiddlePointY;

      /* BIAS::Matrix<float> gDLMat(heightOfLSP,lengthOfLSP) */
      for ( short hID = 0; hID < heightOfLSP; hID++ )
      {
        /*initialization */
        sCorX = sCorX0;
        sCorY = sCorY0;

        pgdLRowSum = 0;
        ngdLRowSum = 0;
        pgdORowSum = 0;
        ngdORowSum = 0;

        for ( short wID = 0; wID < lengthOfLSP; wID++ )
        {
          tempCor = (short) round( sCorX );
          xCor = ( tempCor < 0 ) ? 0 : ( tempCor > imageWidth ) ? imageWidth : tempCor;
          tempCor = (short) round( sCorY );
          yCor = ( tempCor < 0 ) ? 0 : ( tempCor > imageHeight ) ? imageHeight : tempCor;

          /* To achieve rotation invariance, each simple gradient is rotated aligned with
           * the line direction and clockwise orthogonal direction.*/
          dx = pdxImg[yCor * realWidth + xCor];
          dy = pdyImg[yCor * realWidth + xCor];
          gDL = dx * dL[0] + dy * dL[1];
          gDO = dx * dO[0] + dy * dO[1];
          if( gDL > 0 )
          {
            pgdLRowSum += gDL;
          }
          else
          {
            ngdLRowSum -= gDL;
          }
          if( gDO > 0 )
          {
            pgdORowSum += gDO;
          }
          else
          {
            ngdORowSum -= gDO;
          }
          sCorX += dL[0];
          sCorY += dL[1];
          /* gDLMat[hID][wID] = gDL; */
        }
        sCorX0 -= dL[1];
        sCorY0 += dL[0];
        coefInGaussion = (float) gaussCoefG_[hID];
        pgdLRowSum = coefInGaussion * pgdLRowSum;
        ngdLRowSum = coefInGaussion * ngdLRowSum;
        pgdL2RowSum = pgdLRowSum * pgdLRowSum;
        ngdL2RowSum = ngdLRowSum * ngdLRowSum;
        pgdORowSum = coefInGaussion * pgdORowSum;
        ngdORowSum = coefInGaussion * ngdORowSum;
        pgdO2RowSum = pgdORowSum * pgdORowSum;
        ngdO2RowSum = ngdORowSum * ngdORowSum;

        /* compute {g_dL |g_dL>0 }, {g_dL |g_dL<0 },
         {g_dO |g_dO>0 }, {g_dO |g_dO<0 } of each band in the line support region
         first, current row belong to current band */
        bandID = (short) ( hID / params.widthOfBand_ );
        coefInGaussion = (float) ( gaussCoefL_[hID % params.widthOfBand_ + params.widthOfBand_] );
        pgdLBandSum[bandID] += coefInGaussion * pgdLRowSum;
        ngdLBandSum[bandID] += coefInGaussion * ngdLRowSum;
        pgdL2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdL2RowSum;
        ngdL2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdL2RowSum;
        pgdOBandSum[bandID] += coefInGaussion * pgdORowSum;
        ngdOBandSum[bandID] += coefInGaussion * ngdORowSum;
        pgdO2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdO2RowSum;
        ngdO2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdO2RowSum;

        /* In order to reduce boundary effect along the line gradient direction,
         * a row's gradient will contribute not only to its current band, but also
         * to its nearest upper and down band with gaussCoefL_.*/
        bandID--;
        if( bandID >= 0 )
        {/* the band above the current band */
          coefInGaussion = (float) ( gaussCoefL_[hID % params.widthOfBand_ + 2 * params.widthOfBand_] );
          pgdLBandSum[bandID] += coefInGaussion * pgdLRowSum;
          ngdLBandSum[bandID] += coefInGaussion * ngdLRowSum;
          pgdL2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdL2RowSum;
          ngdL2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdL2RowSum;
          pgdOBandSum[bandID] += coefInGaussion * pgdORowSum;
          ngdOBandSum[bandID] += coefInGaussion * ngdORowSum;
          pgdO2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdO2RowSum;
          ngdO2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdO2RowSum;
        }
        bandID = bandID + 2;
        if( bandID < NUM_OF_BANDS )
        {/*the band below the current band */
          coefInGaussion = (float) ( gaussCoefL_[hID % params.widthOfBand_] );
          pgdLBandSum[bandID] += coefInGaussion * pgdLRowSum;
          ngdLBandSum[bandID] += coefInGaussion * ngdLRowSum;
          pgdL2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdL2RowSum;
          ngdL2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdL2RowSum;
          pgdOBandSum[bandID] += coefInGaussion * pgdORowSum;
          ngdOBandSum[bandID] += coefInGaussion * ngdORowSum;
          pgdO2BandSum[bandID] += coefInGaussion * coefInGaussion * pgdO2RowSum;
          ngdO2BandSum[bandID] += coefInGaussion * coefInGaussion * ngdO2RowSum;
        }
      }
      /* gDLMat.Save("gDLMat.txt");
       return 0; */

      /* construct line descriptor */
      pSingleLine->descriptor.resize( descriptor_size );
      desVec = pSingleLine->descriptor.data();

      short desID;

      /*Note that the first and last bands only have (lengthOfLSP * widthOfBand_ * 2.0) pixels
       * which are counted. */
      float invN2 = (float) ( 1.0 / ( params.widthOfBand_ * 2.0 ) );
      float invN3 = (float) ( 1.0 / ( params.widthOfBand_ * 3.0 ) );
      float invN, temp;
      for ( bandID = 0; bandID < NUM_OF_BANDS; bandID++ )
      {
        if( bandID == 0 || bandID == NUM_OF_BANDS - 1 )
        {
          invN = invN2;
        }
        else
        {
          invN = invN3;
        }
        desID = bandID * 8;
        temp = pgdLBandSum[bandID] * invN;
        desVec[desID] = temp;/* mean value of pgdL; */
        desVec[desID + 4] = sqrt( pgdL2BandSum[bandID] * invN - temp * temp );  //std value of pgdL;
        temp = ngdLBandSum[bandID] * invN;
        desVec[desID + 1] = temp;  //mean value of ngdL;
        desVec[desID + 5] = sqrt( ngdL2BandSum[bandID] * invN - temp * temp );  //std value of ngdL;

        temp = pgdOBandSum[bandID] * invN;
        desVec[desID + 2] = temp;  //mean value of pgdO;
        desVec[desID + 6] = sqrt( pgdO2BandSum[bandID] * invN - temp * temp );  //std value of pgdO;
        temp = ngdOBandSum[bandID] * invN;
        desVec[desID + 3] = temp;  //mean value of ngdO;
        desVec[desID + 7] = sqrt( ngdO2BandSum[bandID] * invN - temp * temp );  //std value of ngdO;
      }

      // normalize;
      float tempM, tempS;
      tempM = 0;
      tempS = 0;
      desVec = pSingleLine->descriptor.data();

      int base = 0;
      for ( short i = 0; i < (short) ( NUM_OF_BANDS * 8 ); ++base, i = (short) ( base * 8 ) )
      {
        tempM += * ( desVec + i ) * * ( desVec + i );  //desVec[8*i+0] * desVec[8*i+0];
        tempM += * ( desVec + i + 1 ) * * ( desVec + i + 1 );  //desVec[8*i+1] * desVec[8*i+1];
        tempM += * ( desVec + i + 2 ) * * ( desVec + i + 2 );  //desVec[8*i+2] * desVec[8*i+2];
        tempM += * ( desVec + i + 3 ) * * ( desVec + i + 3 );  //desVec[8*i+3] * desVec[8*i+3];
        tempS += * ( desVec + i + 4 ) * * ( desVec + i + 4 );  //desVec[8*i+4] * desVec[8*i+4];
        tempS += * ( desVec + i + 5 ) * * ( desVec + i + 5 );  //desVec[8*i+5] * desVec[8*i+5];
        tempS += * ( desVec + i + 6 ) * * ( desVec + i + 6 );  //desVec[8*i+6] * desVec[8*i+6];
        tempS += * ( desVec + i + 7 ) * * ( desVec + i + 7 );  //desVec[8*i+7] * desVec[8*i+7];
      }

      tempM = 1 / sqrt( tempM );
      tempS = 1 / sqrt( tempS );
      desVec = pSingleLine->descriptor.data();
      base = 0;
      for ( short i = 0; i < (short) ( NUM_OF_BANDS * 8 ); ++base, i = (short) ( base * 8 ) )
      {
        * ( desVec + i ) = * ( desVec + i ) * tempM;  //desVec[8*i] =  desVec[8*i] * tempM;
        * ( desVec + 1 + i ) = * ( desVec + 1 + i ) * tempM;  //desVec[8*i+1] =  desVec[8*i+1] * tempM;
        * ( desVec + 2 + i ) = * ( desVec + 2 + i ) * tempM;  //desVec[8*i+2] =  desVec[8*i+2] * tempM;
        * ( desVec + 3 + i ) = * ( desVec + 3 + i ) * tempM;  //desVec[8*i+3] =  desVec[8*i+3] * tempM;
        * ( desVec + 4 + i ) = * ( desVec + 4 + i ) * tempS;  //desVec[8*i+4] =  desVec[8*i+4] * tempS;
        * ( desVec + 5 + i ) = * ( desVec + 5 + i ) * tempS;  //desVec[8*i+5] =  desVec[8*i+5] * tempS;
        * ( desVec + 6 + i ) = * ( desVec + 6 + i ) * tempS;  //desVec[8*i+6] =  desVec[8*i+6] * tempS;
        * ( desVec + 7 + i ) = * ( desVec + 7 + i ) * tempS;  //desVec[8*i+7] =  desVec[8*i+7] * tempS;
      }

      /* In order to reduce the influence of non-linear illumination,
       * a threshold is used to limit the value of element in the unit feature
       * vector no larger than this threshold. In Z.Wang's work, a value of 0.4 is found
       * empirically to be a proper threshold.*/
      desVec = pSingleLine->descriptor.data();
      for ( short i = 0; i < descriptor_size; i++ )
      {
        if( desVec[i] > 0.4 )
        {
          desVec[i] = (float) 0.4;
        }
      }

      //re-normalize desVec;
      temp = 0;
      for ( short i = 0; i < descriptor_size; i++ )
      {
        temp += desVec[i] * desVec[i];
      }

      temp = 1 / sqrt( temp );
      for ( short i = 0; i < descriptor_size; i++ )
      {
        desVec[i] = desVec[i] * temp;
      }
    }/* end for(short lineIDInSameLine = 0; lineIDInSameLine<sameLineSize;
     lineIDInSameLine++) */

    cv::Mat appoggio = cv::Mat( 1, 32, CV_32FC1 );
    float* pointerToRow = appoggio.ptr<float>( 0 );
    for ( int g = 0; g < 32; g++ )
    {
      /* get LBD data */
      float* des_Vec = keyLines[lineIDInScaleVec][0].descriptor.data();
      *pointerToRow = des_Vec[g];
      pointerToRow++;

    }

  }/* end for(short lineIDInScaleVec = 0;
   lineIDInScaleVec<numOfFinalLine; lineIDInScaleVec++) */

  delete[] dL;
  delete[] dO;
  delete[] pgdLBandSum;
  delete[] ngdLBandSum;
  delete[] pgdL2BandSum;
  delete[] ngdL2BandSum;
  delete[] pgdOBandSum;
  delete[] ngdOBandSum;
  delete[] pgdO2BandSum;
  delete[] ngdO2BandSum;

  return 1;

}
}
