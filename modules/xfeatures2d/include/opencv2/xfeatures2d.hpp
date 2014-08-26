/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef __OPENCV_XFEATURES2D_HPP__
#define __OPENCV_XFEATURES2D_HPP__

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

namespace cv
{
namespace xfeatures2d
{

CV_EXPORTS bool initModule_xfeatures2d(void);

/*!
 FREAK implementation
 */
class CV_EXPORTS FREAK : public DescriptorExtractor
{
public:
    /** Constructor
     * @param orientationNormalized enable orientation normalization
     * @param scaleNormalized enable scale normalization
     * @param patternScale scaling of the description pattern
     * @param nbOctave number of octaves covered by the detected keypoints
     * @param selectedPairs (optional) user defined selected pairs
     */
    explicit FREAK( bool orientationNormalized = true,
                   bool scaleNormalized = true,
                   float patternScale = 22.0f,
                   int nOctaves = 4,
                   const std::vector<int>& selectedPairs = std::vector<int>());
    FREAK( const FREAK& rhs );
    FREAK& operator=( const FREAK& );

    virtual ~FREAK();

    /** returns the descriptor length in bytes */
    virtual int descriptorSize() const;

    /** returns the descriptor type */
    virtual int descriptorType() const;

    /** returns the default norm type */
    virtual int defaultNorm() const;

    /** select the 512 "best description pairs"
     * @param images grayscale images set
     * @param keypoints set of detected keypoints
     * @param corrThresh correlation threshold
     * @param verbose print construction information
     * @return list of best pair indexes
     */
    std::vector<int> selectPairs( const std::vector<Mat>& images, std::vector<std::vector<KeyPoint> >& keypoints,
                                 const double corrThresh = 0.7, bool verbose = true );

    AlgorithmInfo* info() const;

    enum
    {
        NB_SCALES = 64, NB_PAIRS = 512, NB_ORIENPAIRS = 45
    };

protected:
    virtual void computeImpl( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) const;
    void buildPattern();

    template <typename imgType, typename iiType>
    imgType meanIntensity( InputArray image, InputArray integral, const float kp_x, const float kp_y,
                          const unsigned int scale, const unsigned int rot, const unsigned int point ) const;

    template <typename srcMatType, typename iiMatType>
    void computeDescriptors( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) const;

    template <typename srcMatType>
    void extractDescriptor(srcMatType *pointsValue, void ** ptr) const;

    bool orientationNormalized; //true if the orientation is normalized, false otherwise
    bool scaleNormalized; //true if the scale is normalized, false otherwise
    double patternScale; //scaling of the pattern
    int nOctaves; //number of octaves
    bool extAll; // true if all pairs need to be extracted for pairs selection

    double patternScale0;
    int nOctaves0;
    std::vector<int> selectedPairs0;

    struct PatternPoint
    {
        float x; // x coordinate relative to center
        float y; // x coordinate relative to center
        float sigma; // Gaussian smoothing sigma
    };

    struct DescriptionPair
    {
        uchar i; // index of the first point
        uchar j; // index of the second point
    };

    struct OrientationPair
    {
        uchar i; // index of the first point
        uchar j; // index of the second point
        int weight_dx; // dx/(norm_sq))*4096
        int weight_dy; // dy/(norm_sq))*4096
    };
    
    std::vector<PatternPoint> patternLookup; // look-up table for the pattern points (position+sigma of all points at all scales and orientation)
    int patternSizes[NB_SCALES]; // size of the pattern at a specific scale (used to check if a point is within image boundaries)
    DescriptionPair descriptionPairs[NB_PAIRS];
    OrientationPair orientationPairs[NB_ORIENPAIRS];
};


/*!
 The "Star" Detector.

 The class implements the keypoint detector introduced by K. Konolige.
 */
class CV_EXPORTS_W StarDetector : public FeatureDetector
{
public:
    //! the full constructor
    CV_WRAP StarDetector(int _maxSize=45, int _responseThreshold=30,
                         int _lineThresholdProjected=10,
                         int _lineThresholdBinarized=8,
                         int _suppressNonmaxSize=5);

    //! finds the keypoints in the image
    CV_WRAP_AS(detect) void operator()(const Mat& image,
                                       CV_OUT std::vector<KeyPoint>& keypoints) const;

    AlgorithmInfo* info() const;

protected:
    void detectImpl( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() ) const;
    
    int maxSize;
    int responseThreshold;
    int lineThresholdProjected;
    int lineThresholdBinarized;
    int suppressNonmaxSize;
};

typedef StarDetector StarFeatureDetector;

/*
 * BRIEF Descriptor
 */
class CV_EXPORTS BriefDescriptorExtractor : public DescriptorExtractor
{
public:
    static const int PATCH_SIZE = 48;
    static const int KERNEL_SIZE = 9;

    // bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes.
    BriefDescriptorExtractor( int bytes = 32 );

    virtual void read( const FileNode& );
    virtual void write( FileStorage& ) const;

    virtual int descriptorSize() const;
    virtual int descriptorType() const;
    virtual int defaultNorm() const;

    /// @todo read and write for brief

    AlgorithmInfo* info() const;

protected:
    virtual void computeImpl(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) const;

    typedef void(*PixelTestFn)(InputArray, const std::vector<KeyPoint>&, OutputArray);
    
    int bytes_;
    PixelTestFn test_fn_;
};
    
}
}

#endif
