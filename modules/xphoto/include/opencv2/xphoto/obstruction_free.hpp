// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef __OPENCV_OBSTRUCTION_FREE_HPP__
#define __OPENCV_OBSTRUCTION_FREE_HPP__

/** @file
@date June 18, 2017
@author Binbin Xu
*/

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/sparse_match_interpolator.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/calib3d.hpp"

#include "opencv2/highgui.hpp"

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
     * @param foreground estimated reflection or opaque obstruction layer
     * @param mask estimated occlusion areas (CV_8UC1), where zero pixels indicate area to be estimated to be occlusions.
     * @param obstructionType: reflection (0) or opaque obstruction (1)
     */
    void removeOcc(const std::vector <Mat> &srcImgs, Mat &dst, Mat& foreground, Mat &mask, const int obstructionType);

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
        //std::vector <Mat> warpedToReference; //warped images from input images through the estimated background flow fields

        //switch different methods. TODO
        //int interpolationType; //interpolation type: 0(reflection) or 1(opaque occlusion)
        //int edgeflowType; //edge flow type

        /** @brief private functions
        */
        /** @brief Build pyramid by stacking all input image sequences
        */
        std::vector<Mat> buildPyramid( const std::vector <Mat>& srcImgs);

        /** @brief Extract certain level of image sequences from the stacked image pyramid
        */
        std::vector<Mat> extractLevelImgs(const std::vector<Mat>& pyramid, const int level);

        /** @brief Initialization: decompose the motion fields
        */
        void motionInitDirect(const std::vector<Mat>& video_input, std::vector<Mat>& back_Flowfields, std::vector<Mat>& fore_flowfields, std::vector<Mat>& warpedToReference);

        /** @brief Initialization: decompose the image components in the case of reflection
        */
        Mat imgInitDecomRef(const std::vector <Mat> &warpedImgs);

        /** @brief Initialization: decompose the image components in the case of opaque reflection
        */
        Mat imgInitDecomOpaq(const std::vector <Mat> &warpedImgs, Mat& foreground, Mat& alphaMap);

        /** @brief Convert from sparse edge displacement to dense motion fields
         */
        Mat sparseToDense(const Mat& im1, const Mat& im2, const Mat& im1_edges, const Mat& sparseFlow);

        /** @brief Visualize the optical flow flow with the window named figName
         */
        void colorFlow(const Mat& flow, std::string figName);

        /** @brief Decompose motion between two images: target frame and source frame, using homography ransac
         */
        void initMotionDecompose(const Mat& im1 , const Mat& im2 , Mat& back_denseFlow, Mat& fore_denseFlow, int back_ransacThre, int fore_ransacThre);

        /** @brief Warp im1 to output through optical flow:
        */
        Mat imgWarpFlow(const Mat& im1, const Mat& flow);

        /** @brief Estimate homography matrix using edge flow fields
        */
        Mat flowHomography(const Mat& edges, const Mat& flow, const int ransacThre);

        /** @brief Convert from index to matrix
        */
        Mat indexToMask(const Mat& indexMat, const int rows, const int cols);

        /** @brief Calculate laplacian filters D_x^TD_x + D_y^TD_y
        */
        Mat Laplac(const Mat& input);

        /*!
         * @brief Calculate the weights in the alternative motion decomposition step
         * @param inputSequence Input sequences
         * @param background Estimated background component in the last iteration
         * @param foreground Estimated foreground/obstruction component in the last iteration
         * @param alphaMap Estimated alpha map for obstruction layer
         * @return omega_1 The weight omega_1 = \phi(||I^t - W_O^t\hat{I}_O - W_O^t\hat{A} \circ W_B^t\hat{I}_B||^2)^{-1}
         * @return omega_2 The weight omega_2 = \phi(||D_x\hat{I}_B||^2+||D_y\hat{I}_B||^2)^{-1}
         * @return omega_3 The weight omega_3 = \phi(||D_x\hat{I}_O||^2+||D_y\hat{I}_O||^2)^{-1}
         */
        void motDecomIrlsWeight(const std::vector<Mat>& inputSequence, const Mat& background, const Mat& foreground,
                            Mat& alphaMap, std::vector<float>& omega_1, std::vector<float>& omega_2, std::vector<float>& omega_3);

    };
//! @}


}
}

#endif // __OPENCV_OBSTRUCTION_FREE_HPP__
