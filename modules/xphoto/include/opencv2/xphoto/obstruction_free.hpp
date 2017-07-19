// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef __OPENCV_OBSTRUCTION_FREE_HPP__
#define __OPENCV_OBSTRUCTION_FREE_HPP__

/** @file
@date June 18, 2017
@author Binbin Xu
*/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/ximgproc/sparse_match_interpolator.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/highgui.hpp>

namespace cv
{
namespace xphoto
{

//! @addtogroup xphoto
//! @{


    /** @brief The class implements a general obstruction free approach that can remove occlusions and reflections from input image sequences without manual masks.

    See the original paper @cite Xue2015ObstructionFree for more details.


    */
    class CV_EXPORTS obstructionFree
    {
    public:
    /*!
     * @brief Constructors
     */
        obstructionFree();

     /*!
     * @brief Constructors
     * @param srcImgs input image sequences
     */
        obstructionFree(const std::vector <Mat> &srcImgs);

    /*!
     * @brief core function to remove occlusions
     * @param srcImgs source image sequences, involving translation motions.
     * @param dst Obstruction-removed destination image, corresponding to the reference image, with the same size and type. In general, the reference image is the center frame of the input image.
     * @param mask estimated occlusion areas (CV_8UC1), where zero pixels indicate area to be estimated to be occlusions.
     * @param obstructionType: reflection (0) or opaque obstruction (1)
     */
        void removeOcc(const std::vector <Mat> &srcImgs, Mat &dst, Mat &mask, const int obstructionType);

    private:
    /*!
     * @brief Parameters
     */
        size_t frameNumber; //frame number of the input sequences
        size_t referenceNumber; //target frame
        int pyramidLevel; // Pyramid level
        int coarseIterations; // iteration number for the coarsest level
        int upperIterations; // iteration number for the upper level
        int fixedPointIterations; // during each level of the pyramid
        int sorIterations; // iterations of SOR
        float omega; // relaxation factor in SOR
        float lambda1; // weight for alpha map smoothness constraints
        float lambda2; // weight for image smoothness constraints
        float lambda3; // weight for independence between back/foreground component
        float lambda4; // weight for gradient sparsity


        std::vector <Mat> backFlowFields; //estimated optical flow fields in the background layer
        std::vector <Mat> foreFlowFields; //estimated optical flow fields in the foreground layer
        std::vector <Mat> warpedToReference; //warped images from input images through the estimated background flow fields

        //switch different methods. TODO
        int interpolationType; //interpolation type: 0(reflection) or 1(opaque occlusion)
        int edgeflowType; //edge flow type

        //private functions
        //build pyramid by stacking all input image sequences
        std::vector<Mat> buildPyramid( const std::vector <Mat>& srcImgs);
        //extract certain level of image sequences from the stacked image pyramid
        std::vector<Mat> extractLevelImgs(const std::vector<Mat>& pyramid, const int level);
        //initialization: decompose the motion fields
        void motionInitDirect(const std::vector<Mat>& video_input, std::vector<Mat>& back_Flowfields, std::vector<Mat>& fore_flowfields, std::vector<Mat>& warpedToReference);
        //initialization: decompose the image components
        Mat imgInitDecompose(const std::vector <Mat> &warpedImgs, const std::vector<Mat>& back_flowfields, const int obstructionType);

    };
//! @}

}
}

#endif // __OPENCV_OBSTRUCTION_FREE_HPP__
