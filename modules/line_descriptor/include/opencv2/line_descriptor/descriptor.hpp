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
 // Copyright (C) 2014, Biagio Montesano, all rights reserved.
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

#ifndef __OPENCV_DESCRIPTOR_HPP__
#define __OPENCV_DESCRIPTOR_HPP__

#include "line_structure.hpp"
#include "array32.hpp"
#include "bitarray.hpp"
#include "bitops.hpp"
#include "bucket_group.hpp"
#include "mihasher.hpp"
#include "sparse_hashtable.hpp"
#include "types.hpp"
#include "ed_line_detector.hpp"
#include <map>

namespace cv
{

CV_EXPORTS bool initModule_line_descriptor();

struct CV_EXPORTS KeyLine
{
 public:
  /* orientation of the line */
  CV_PROP_RW
  float angle;

  /* object ID, that can be used to cluster keylines by the line they represent */
  CV_PROP_RW
  int class_id;

  /* octave (pyramid layer), from which the keyline has been extracted */
  CV_PROP_RW
  int octave;

  /* coordinates of the middlepoint */
  CV_PROP_RW
  Point2f pt;

  /* the response, by which the strongest keylines have been selected.
   It's represented by the ratio between line's length and maximum between
   image's width and height */
  CV_PROP_RW
  float response;

  /* minimum area containing line */
  CV_PROP_RW
  float size;

  /* lines's extremes in original image */
  CV_PROP_RW
  float startPointX;CV_PROP_RW
  float startPointY;CV_PROP_RW
  float endPointX;CV_PROP_RW
  float endPointY;

  /* line's extremes in image it was extracted from */
  CV_PROP_RW
  float sPointInOctaveX;CV_PROP_RW
  float sPointInOctaveY;CV_PROP_RW
  float ePointInOctaveX;CV_PROP_RW
  float ePointInOctaveY;

  /* the length of line */
  CV_PROP_RW
  float lineLength;

  /* number of pixels covered by the line */
  CV_PROP_RW
  int numOfPixels;

  /* constructor */
  CV_WRAP
  KeyLine()
  {
  }
};

class CV_EXPORTS_W BinaryDescriptor : public Algorithm
{

 public:
  struct CV_EXPORTS Params
  {
    CV_WRAP
    Params();

    /* the number of image octaves (default = 1) */
    CV_PROP_RW
    int numOfOctave_;

    /* the width of band; (default: 7) */
    CV_PROP_RW
    int widthOfBand_;

    /* image's reduction ratio in construction of Gaussian pyramids */
    CV_PROP_RW
    int reductionRatio;

    CV_PROP_RW
    int ksize_;

    /* read parameters from a FileNode object and store them (struct function) */
    CV_WRAP
    void read( const FileNode& fn );

    /* store parameters to a FileStorage object (struct function) */
    CV_WRAP
    void write( FileStorage& fs ) const;

  };

  /* constructor */
  CV_WRAP
  BinaryDescriptor( const BinaryDescriptor::Params &parameters = BinaryDescriptor::Params() );

  /* constructors with smart pointers */
  CV_WRAP
  static Ptr<BinaryDescriptor> createBinaryDescriptor();CV_WRAP
  static Ptr<BinaryDescriptor> createBinaryDescriptor( Params parameters );

  /* destructor */
  ~BinaryDescriptor();

  /* setters and getters */
  CV_WRAP
  int getNumOfOctaves();CV_WRAP
  void setNumOfOctaves( int octaves );CV_WRAP
  int getWidthOfBand();CV_WRAP
  void setWidthOfBand( int width );CV_WRAP
  int getReductionRatio();CV_WRAP
  void setReductionRatio( int rRatio );

  /* reads parameters from a FileNode object and store them (class function ) */
  CV_WRAP
  virtual void read( const cv::FileNode& fn );

  /* stores parameters to a FileStorage object (class function) */
  CV_WRAP
  virtual void write( cv::FileStorage& fs ) const;

  /* requires line detection (only one image) */
  CV_WRAP
  void detect( const Mat& image, CV_OUT std::vector<KeyLine>& keypoints, const Mat& mask = Mat() );

  /* requires line detection (more than one image) */
  CV_WRAP
  void detect( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, const std::vector<Mat>& masks =
                   std::vector<Mat>() ) const;

  /* requires descriptors computation (only one image) */
  CV_WRAP
  void compute( const Mat& image, CV_OUT CV_IN_OUT std::vector<KeyLine>& keylines, CV_OUT Mat& descriptors, bool returnFloatDescr = false ) const;

  /* requires descriptors computation (more than one image) */
  CV_WRAP
  void compute( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, std::vector<Mat>& descriptors, bool returnFloatDescr =
                    false ) const;

  /* returns descriptor size */
  CV_WRAP
  int descriptorSize() const;

  /* returns data type */
  CV_WRAP
  int descriptorType() const;

  /* returns norm mode */
  CV_WRAP
  int defaultNorm() const;

  /* definition of operator () */
  CV_WRAP_AS(detectAndCompute)
  virtual void operator()( InputArray image, InputArray mask, CV_OUT std::vector<KeyLine>& keylines, OutputArray descriptors,
                           bool useProvidedKeyLines = false, bool returnFloatDescr = false ) const;

 protected:
  /* implementation of line detection */
  virtual void detectImpl( const Mat& imageSrc, std::vector<KeyLine>& keylines, const Mat& mask = Mat() ) const;

  /* implementation of descriptors' computation */
  virtual void computeImpl( const Mat& imageSrc, std::vector<KeyLine>& keylines, Mat& descriptors, bool returnFloatDescr,
                            bool useDetectionData ) const;

  /* function inherited from Algorithm */
  AlgorithmInfo* info() const;

 private:
  /* compute Gaussian pyramids */
  void computeGaussianPyramid( const Mat& image, const int numOctaves );

  /* compute Sobel's derivatives */
  void computeSobel( const Mat& image, const int numOctaves );

  /* conversion of an LBD descriptor to its binary representation */
  unsigned char binaryConversion( float* f1, float* f2 );

  /* compute LBD descriptors using EDLine extractor */
  int computeLBD( ScaleLines &keyLines, bool useDetectionData = false );

  /* gathers lines in groups using EDLine extractor.
   Each group contains the same line, detected in different octaves */
  int OctaveKeyLines( cv::Mat& image, ScaleLines &keyLines );

  /* the local gaussian coefficient applied to the orthogonal line direction within each band */
  std::vector<double> gaussCoefL_;

  /* the global gaussian coefficient applied to each row within line support region */
  std::vector<double> gaussCoefG_;

  /* descriptor parameters */
  Params params;

  /* vector of sizes of downsampled and blurred images */
  std::vector<cv::Size> images_sizes;

  /*For each octave of image, we define an EDLineDetector, because we can get gradient images (dxImg, dyImg, gImg)
   *from the EDLineDetector class without extra computation cost. Another reason is that, if we use
   *a single EDLineDetector to detect lines in different octave of images, then we need to allocate and release
   *memory for gradient images (dxImg, dyImg, gImg) repeatedly for their varying size*/
  std::vector<Ptr<EDLineDetector> > edLineVec_;

  /* Sobel's derivatives */
  std::vector<cv::Mat> dxImg_vector, dyImg_vector;

  /* Gaussian pyramid */
  std::vector<cv::Mat> octaveImages;

};

class CV_EXPORTS_W LSDDetector : public Algorithm
{
 public:

  /* constructor */
  CV_WRAP
  LSDDetector()
  {
  }
  ;

  /* constructor with smart pointer */
  CV_WRAP
  static Ptr<LSDDetector> createLSDDetector();

  /* requires line detection (only one image) */
  CV_WRAP
  void detect( const Mat& image, CV_OUT std::vector<KeyLine>& keypoints, int scale, int numOctaves, const Mat& mask = Mat() );

  /* requires line detection (more than one image) */
  CV_WRAP
  void detect( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, int scale, int numOctaves,
               const std::vector<Mat>& masks = std::vector<Mat>() ) const;

 private:
  /* compute Gaussian pyramid of input image */
  void computeGaussianPyramid( const Mat& image, int numOctaves, int scale );

  /* implementation of line detection */
  void detectImpl( const Mat& imageSrc, std::vector<KeyLine>& keylines, int numOctaves, int scale, const Mat& mask ) const;

  /* matrices for Gaussian pyramids */
  std::vector<cv::Mat> gaussianPyrs;

 protected:
  /* function inherited from Algorithm */
  AlgorithmInfo* info() const;
};

class CV_EXPORTS_W BinaryDescriptorMatcher : public Algorithm
{

 public:
  /* for every input descriptor,
   find the best matching one (for a pair of images) */
  CV_WRAP
  void match( const Mat& queryDescriptors, const Mat& trainDescriptors, std::vector<DMatch>& matches, const Mat& mask = Mat() ) const;

  /* for every input descriptor,
   find the best matching one (from one image to a set) */
  CV_WRAP
  void match( const Mat& queryDescriptors, std::vector<DMatch>& matches, const std::vector<Mat>& masks = std::vector<Mat>() );

  /* for every input descriptor,
   find the best k matching descriptors (for a pair of images) */
  CV_WRAP
  void knnMatch( const Mat& queryDescriptors, const Mat& trainDescriptors, std::vector<std::vector<DMatch> >& matches, int k, const Mat& mask = Mat(),
                 bool compactResult = false ) const;

  /* for every input descriptor,
   find the best k matching descriptors (from one image to a set) */
  CV_WRAP
  void knnMatch( const Mat& queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k, const std::vector<Mat>& masks = std::vector<Mat>(),
                 bool compactResult = false );

  /* for every input descriptor, find all the ones falling in a
   certain matching radius (for a pair of images) */
  CV_WRAP
  void radiusMatch( const Mat& queryDescriptors, const Mat& trainDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
                    const Mat& mask = Mat(), bool compactResult = false ) const;

  /* for every input descriptor, find all the ones falling in a
   certain matching radius (from one image to a set) */
  CV_WRAP
  void radiusMatch( const Mat& queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance, const std::vector<Mat>& masks =
                        std::vector<Mat>(),
                    bool compactResult = false );

  /* store new descriptors to be inserted in dataset */
  CV_WRAP
  void add( const std::vector<Mat>& descriptors );

  /* store new descriptors into dataset */
  CV_WRAP
  void train();

  /* constructor with smart pointer */
  CV_WRAP
  static Ptr<BinaryDescriptorMatcher> createBinaryDescriptorMatcher();

  /* clear dataset and internal data */
  CV_WRAP
  void clear();

  /* constructor */
  CV_WRAP
  BinaryDescriptorMatcher();

  /* destructor */
  ~BinaryDescriptorMatcher()
  {
  }

 protected:
  /* function inherited from Algorithm */
  AlgorithmInfo* info() const;

 private:
  /* retrieve Hamming distances */
  void checkKDistances( UINT32 * numres, int k, std::vector<int>& k_distances, int row, int string_length ) const;

  /* matrix to store new descriptors */
  Mat descriptorsMat;

  /* map storing where each bunch of descriptors benins in DS */
  std::map<int, int> indexesMap;

  /* internal MiHaser representing dataset */
  Mihasher* dataset;

  /* index from which next added descriptors' bunch must begin */
  int nextAddedIndex;

  /* number of images whose descriptors are stored in DS */
  int numImages;

  /* number of descriptors in dataset */
  int descrInDS;

};

/* --------------------------------------------------------------------------------------------
 UTILITY FUNCTIONS
 -------------------------------------------------------------------------------------------- */

/* struct for drawing options */
struct CV_EXPORTS DrawLinesMatchesFlags
{
  enum
  {
    DEFAULT = 0,  // Output image matrix will be created (Mat::create),
                  // i.e. existing memory of output image may be reused.
                  // Two source images, matches, and single keylines
                  // will be drawn.
    DRAW_OVER_OUTIMG = 1,  // Output image matrix will not be
    // created (using Mat::create). Matches will be drawn
    // on existing content of output image.
    NOT_DRAW_SINGLE_LINES = 2  // Single keylines will not be drawn.
  };
};

/* draw matches between two images */
CV_EXPORTS_W void drawLineMatches( const Mat& img1, const std::vector<KeyLine>& keylines1, const Mat& img2, const std::vector<KeyLine>& keylines2,
                                   const std::vector<DMatch>& matches1to2, Mat& outImg, const Scalar& matchColor = Scalar::all( -1 ),
                                   const Scalar& singleLineColor = Scalar::all( -1 ), const std::vector<char>& matchesMask = std::vector<char>(),
                                   int flags = DrawLinesMatchesFlags::DEFAULT );

/* draw extracted lines on original image */
CV_EXPORTS_W void drawKeylines( const Mat& image, const std::vector<KeyLine>& keylines, Mat& outImage, const Scalar& color = Scalar::all( -1 ),
                                int flags = DrawLinesMatchesFlags::DEFAULT );

}

#endif
