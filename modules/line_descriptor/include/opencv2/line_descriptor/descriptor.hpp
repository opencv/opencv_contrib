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

//#ifdef __OPENCV_BUILD
//#error this is a compatibility header which should not be used inside the OpenCV library
//#endif

#ifndef __OPENCV_DESCRIPTOR_HPP__
#define __OPENCV_DESCRIPTOR_HPP__

#include "LineStructure.hpp"
#include "opencv2/core.hpp"


namespace cv
{

    class CV_EXPORTS_W KeyLine: public KeyPoint
    {
    public:
        /* lines's extremes in original image */
        float startPointX;
        float startPointY;
        float endPointX;
        float endPointY;

        /* line's extremes in image it was extracted from */
        float sPointInOctaveX;
        float sPointInOctaveY;
        float ePointInOctaveX;
        float ePointInOctaveY;

        /* the length of line */
        float lineLength;

        /* number of pixels covered by the line */
        unsigned int numOfPixels;

    };

    class CV_EXPORTS_W BinaryDescriptor : public Feature2D
    {

    public:
        struct CV_EXPORTS_W_SIMPLE Params{
            CV_WRAP Params();

            /* global threshold for line descriptor distance, default is 0.35 */
            CV_PROP_RW  float LowestThreshold;

            /* the NNDR threshold for line descriptor distance, default is 0.6 */
            CV_PROP_RW float NNDRThreshold;

            /* the size of Gaussian kernel: ksize X ksize, default value is 5 */
            CV_PROP_RW  int ksize_;

            /* the number of image octaves (default = 5) */
            CV_PROP_RW int  numOfOctave_;

            /* the number of bands used to compute line descriptor (default: 9) */
            CV_PROP_RW int  numOfBand_;

            /* the width of band; (default: 7) */
            CV_PROP_RW int  widthOfBand_;

            /* image's reduction ratio in construction of Gaussian pyramids */
            CV_PROP_RW int reductionRatio;

            /* read parameters from a FileNode object and store them (struct function) */
            void read( const FileNode& fn );

            /* store parameters to a FileStorage object (struct function) */
            void write( FileStorage& fs ) const;

        };

        /* constructor */
        CV_WRAP BinaryDescriptor(const BinaryDescriptor::Params &parameters =
                BinaryDescriptor::Params());

        /* constructors with smart pointers */
        CV_EXPORTS Ptr<BinaryDescriptor> createBinaryDescriptor();
        CV_EXPORTS Ptr<BinaryDescriptor> createBinaryDescriptor(Params parameters);

        /* read parameters from a FileNode object and store them (class function ) */
        virtual void read( const cv::FileNode& fn );

        /* store parameters to a FileStorage object (class function) */
        virtual void write( cv::FileStorage& fs ) const;

        /* requires line detection (only one image) */
        CV_WRAP void detect( const Mat& image,
                             CV_OUT std::vector<KeyPoint>& keypoints,
                             const Mat& mask=Mat() );

        /* requires line detection (more than one image) */
        void detect( const std::vector<Mat>& images,
                     std::vector<std::vector<KeyPoint> >& keypoints,
                     const std::vector<Mat>& masks=std::vector<Mat>() ) const;

        /* requires descriptors computation (only one image) */
        CV_WRAP void compute( const Mat& image,
                              CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints,
                              CV_OUT Mat& descriptors ) const;

        /* requires descriptors computation (more than one image) */
        void compute( const std::vector<Mat>& images,
                      std::vector<std::vector<KeyPoint> >& keypoints,
                      std::vector<Mat>& descriptors ) const;

        /*return descriptor size */
        int descriptorSize() const;

        /* return data type */
        int descriptorType() const;

        /* return norm mode */
        int defaultNorm() const;

        /* check whether Gaussian pyramids were created */
        bool empty() const;

        CV_WRAP_AS(detectAndCompute) virtual void operator()( InputArray image,
                                                              InputArray mask,
                                                              CV_OUT std::vector<KeyPoint>& keypoints,
                                                              OutputArray descriptors,
                                                              bool useProvidedKeypoints=false ) const;


    protected:
        virtual void detectImpl( const Mat& image,
                                 std::vector<KeyPoint>& keypoints,
                                 const Mat& mask=Mat() ) const;

        virtual void computeImpl( const Mat& image,
                                  std::vector<KeyPoint>& keypoints,
                                  Mat& descriptors ) const;

        AlgorithmInfo* info() const;

    private:
        /* conversion of an LBD descriptor to its binary representation */
        unsigned char binaryConversion(float* f1, float* f2);

        /* compute LBD descriptors */
        int computeLBD(ScaleLines &keyLines);

        /* compute Gaussian pyramid of input image */
        void computeGaussianPyramid(const Mat& image);

        /* gather lines in groups.
        Each group contains the same line, detected in different octaves */
        int OctaveKeyLines(ScaleLines &keyLines);

        /* get coefficients of line passing by two points (in line_extremes) */
        void getLineParameters(cv::Vec4i &line_extremes, cv::Vec3i &lineParams);

        /* compute the angle between line and X axis */
        float getLineDirection(cv::Vec3i &lineParams);

        /* the local gaussian coefficient applied to the orthogonal line direction within each band */
        std::vector<float> gaussCoefL_;

        /* the global gaussian coefficient applied to each Row within line support region */
        std::vector<float> gaussCoefG_;

        /* vector to store horizontal and vertical derivatives of octave images */
        std::vector<cv::Mat> dxImg_vector, dyImg_vector;

        /* vectot to store sizes of octave images */
        std::vector<cv::Size> images_sizes;

        /* structure to store lines extracted from each octave image */
        std::vector<std::vector<cv::Vec4i> > extractedLines;

        /* descriptor parameters */
        Params params;

        /* vector to store the Gaussian pyramid od an input image */
        std::vector<cv::Mat> octaveImages;

    };

}

#endif
