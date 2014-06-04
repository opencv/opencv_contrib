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

    class CV_EXPORTS_W LineDescriptor : public virtual Algorithm
    {
    public:
        virtual ~LineDescriptor();
        void getLineBinaryDescriptors(cv::Mat &oct_binaryDescMat);

    protected:
        virtual void getLineBinaryDescriptorsImpl(cv::Mat &oct_binaryDescMat);

    };

    class CV_EXPORTS_W BinaryDescriptor : public LineDescriptor
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

            void read( const FileNode& fn );
            void write( FileStorage& fs ) const;

        };

        CV_WRAP BinaryDescriptor(const BinaryDescriptor::Params &parameters =
                BinaryDescriptor::Params());

        virtual void read( const cv::FileNode& fn );
        virtual void write( cv::FileStorage& fs ) const;
        void getLineBinaryDescriptors(cv::Mat &oct_binaryDescMat);


    protected:
        virtual void getLineBinaryDescriptorsImpl(cv::Mat &oct_binaryDescMat);
        AlgorithmInfo* info() const;

        Params params;


    private:
        unsigned char binaryTest(float* f1, float* f2);
        int ComputeLBD_(ScaleLines &keyLines);
        int OctaveKeyLines(std::vector<cv::Mat> & octaveImages, ScaleLines &keyLines);
        void getLineParameters(cv::Vec4i &line_extremes, cv::Vec3i &lineParams);
        float getLineDirection(cv::Vec3i &lineParams);

        /* the local gaussian coefficient apply to the orthogonal line direction within each band */
        std::vector<float> gaussCoefL_;

        /* the global gaussian coefficient apply to each Row within line support region */
        std::vector<float> gaussCoefG_;

        /* vector to store horizontal and vertical derivatives of octave images */
        std::vector<cv::Mat> dxImg_vector, dyImg_vector;

        /* vectot to store sizes of octave images */
        std::vector<cv::Size> images_sizes;

        /* structure to store lines extracted from each octave image */
        std::vector<std::vector<cv::Vec4i> > extractedLines;

    };

}

#endif
