/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2015-2018, OpenCV Foundation, all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */


//__OPENCV_QUASI_DENSE_STEREO_H__
#ifndef __OPENCV_QUASI_DENSE_STEREO_H__
#define __OPENCV_QUASI_DENSE_STEREO_H__



#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> //GFT
#include <opencv2/video/tracking.hpp> //LK
#include <math.h>
#include <iostream>
#include <algorithm> //max
#include <queue>
#include <stdint.h>
#include <opencv2/tracking.hpp>
#include <fstream>
#include <string.h>


#include <opencv2/qds/defaults.hpp>

namespace cv
{
namespace qds
{
/** \addtogroup qds
 *  @{
 */


// A basic match structure
struct CV_EXPORTS_W_SIMPLE Match
{
    CV_PROP_RW cv::Point2i p0;
    CV_PROP_RW cv::Point2i p1;
    CV_PROP_RW float	corr;

    CV_WRAP_AS(less) bool operator < (const Match & rhs) const
    {
        return corr < rhs.corr;
    }
};
struct CV_EXPORTS_W_SIMPLE PropagationParameters
{
    CV_PROP_RW int	corrWinSizeX;			// similarity window
    CV_PROP_RW int	corrWinSizeY;

    CV_PROP_RW int borderX;					// border to ignore
    CV_PROP_RW int borderY;

    //matching
    CV_PROP_RW float correlationThreshold;	// correlation threshold
    CV_PROP_RW float textrureThreshold;		// texture threshold

    CV_PROP_RW int	  neighborhoodSize;		// neighborhood size
    CV_PROP_RW int	  disparityGradient;	// disparity gradient threshold

    // Parameters for LK flow algorithm
    CV_PROP_RW int lkTemplateSize;
    CV_PROP_RW int lkPyrLvl;
    CV_PROP_RW int lkTermParam1;
    CV_PROP_RW float lkTermParam2;

    // Parameters for GFT algorithm.
    CV_PROP_RW float gftQualityThres;
    CV_PROP_RW int gftMinSeperationDist;
    CV_PROP_RW int gftMaxNumFeatures;

};

typedef std::priority_queue<Match, std::vector<Match>, std::less<Match> > t_matchPriorityQueue;


/**
 * @brief Class containing the methods needed for Quasi Dense Stereo computation.
 *
 */

class CV_EXPORTS_W QuasiDenseStereo
{


public:
    /**
     * @brief constructor
     * @param monoImgSize The size of the input images.
     * @note Left and right images must be of the same size.
     * @param paramFilepath Specifies the location of the file containing the values of all the
     * parameters used in this class.
     * @note Default value is an an empty string "". In this case the class default parameters,
     * found in the defaults.hpp file, are loaded.
     */
    QuasiDenseStereo(cv::Size monoImgSize, cv::String paramFilepath ="");


    /**
     * @brief destructor
     * Method to free all the memory allocated by matrices and vectors in this class.
     */
    virtual ~QuasiDenseStereo();


    /**
     * @brief Load a file containing the configuration parameters of the class.
     * @param[in] filepath The location of the .YAML file containing the configuration parameters.
     * @note default value is an empty string in which case the default
     * parameters, specified in the qds/defaults.h header-file, are loaded.
     * @retval 1: If the path is not empty and the program loaded the parameters successfully.
     * @retval 0: If the path is empty and the program loaded default parameters.
     * @retval -1: If the file location is not valid or the program could not open the file and
     * loaded default parameters from defaults.hpp.
     * @note The method is automatically called in the constructor and configures the class.
     * @note Loading different parameters will have an effect on the output. This is useful for tuning
     * in case of video processing.
     * @sa loadParameters
     */
    CV_WRAP int loadParameters(cv::String filepath="");


    /**
     * @brief Save a file containing all the configuration parameters the class is currently set to.
     * @param[in] filepath The location to store the parameters file.
     * @note Calling this method with no arguments will result in storing class parameters to a file
     * names "qds_parameters.yaml" in the root project folder.
     * @note This method can be used to generate a template file for tuning the class.
     * @sa loadParameters
     */
    CV_WRAP int saveParameters(cv::String filepath="./qds_parameters.yaml");


    /**
     * @brief Get The sparse corresponding points.
     * @param[out] sMatches A vector containing all sparse correspondences.
     * @note The method clears the sMatches vector.
     * @note The returned Match elements inside the sMatches vector, do not use corr member.
     */
     CV_WRAP void getSparseMatches(std::vector<qds::Match> &sMatches);


    /**
     * @brief Get The dense corresponding points.
     * @param[out] dMatches A vector containing all dense matches.
     * @note The method clears the dMatches vector.
     * @note The returned Match elements inside the sMatches vector, do not use corr member.
     */
    CV_WRAP void getDenseMatches(std::vector<qds::Match> &dMatches);



    /**
     * @brief Main process of the algorithm. This method computes the sparse seeds and then densifies them.
     *
     * Initially input images are converted to gray-scale and then the sparseMatching method
     * is called to obtain the sparse stereo. Finally quasiDenseMatching is called to densify the corresponding
     * points.
     * @param[in] imgLeft The left Channel of a stereo image pair.
     * @param[in] imgRight The right Channel of a stereo image pair.
     * @note If input images are in color, the method assumes that are BGR and converts them to grayscale.
     * @sa sparseMatching
     * @sa quasiDenseMatching
     */
    CV_WRAP void process(const cv::Mat &imgLeft ,const cv::Mat &imgRight);


    /**
     * @brief Specify pixel coordinates in the left image and get its corresponding location in the right image.
     * @param[in] x The x pixel coordinate in the left image channel.
     * @param[in] y The y pixel coordinate in the left image channel.
     * @retval cv::Point(x, y) The location of the corresponding pixel in the right image.
     * @retval cv::Point(0, 0) (NO_MATCH)  if no match is found in the right image for the specified pixel location in the left image.
     * @note This method should be always called after process, otherwise the matches will not be correct.
     */
    CV_WRAP cv::Point2f getMatch(const int x, const int y);


    /**
     * @brief Compute and return the disparity map based on the correspondences found in the "process" method.
     * @param[in] disparityLvls The level of detail in output disparity image.
     * @note Default level is 50
     * @return cv::Mat containing a the disparity image in grayscale.
     * @sa computeDisparity
     * @sa quantizeDisparity
     */
    CV_WRAP cv::Mat getDisparity(uint8_t disparityLvls=50);



    PropagationParameters	Param;

protected:
    /**
     * @brief Computes sparse stereo. The output is stores in refMap and mthMap.
     *
     * This method used the "goodFeaturesToTrack" function of OpenCV to extracts salient points
     * in the left image. Feature locations are used as inputs in the "calcOpticalFlowPyrLK"
     * function of OpenCV along with the left and right images. The optical flow algorithm estimates
     * tracks the locations of the features in the right image. The two set of locations constitute
     * the sparse set of matches. These are then used as seeds in the intensification stage of the algorithm.
     * @param[in] imgLeft The left Channel of a stereo image.
     * @param[in] imgRight The right Channel of a stereo image.
     * @param[out] featuresLeft (vector of points) The location of the features in the left image.
     * @param[out] featuresRight (vector of points) The location of the features in the right image.
     * @note featuresLeft and featuresRight must have the same length and corresponding features
     * must be indexed the same way in both vectors.
     */
    CV_WRAP virtual void sparseMatching(const cv::Mat &imgLeft ,const cv::Mat &imgRight,
                                        std::vector< cv::Point2f > &featuresLeft,
                                        std::vector< cv::Point2f > &featuresRight);


    /**
     * @brief Based on the seeds computed in sparse stereo, this method calculates the semi dense set of correspondences.
     *
     * The method initially discards low quality matches based on their zero-normalized cross correlation (zncc) value.
     * This is done by calling the "extractSparseSeeds" method. Remaining high quality Matches are stored in a t_matchPriorityQueue
     * sorted according to their zncc value. The priority queue allows for new matches to be added while keeping track
     * of the best Match. The algorithm then process the queue iteratively.
     * In every iteration a Match is popped from the queue. The algorithm then tries to find candidate
     * matches by matching every point in a small patch around the left Match feature, with a point
     * within a same sized patch around the corresponding right feature. For each candidate point match,
     * the zncc is computed and if it surpasses a threshold, the candidate pair is stored in a temporary
     * priority queue. After this process completed the candidate matches are popped from the Local
     * priority queue and if a match is not registered in refMap, it means that is the best match for
     * this point. The algorithm registers this point in refMap and also push it to the Seed queue.
     * if a candidate match is already registered, it means that is not the best and the algorithm
     * discards it.
     *
     * @note This method does not have input arguments, but uses the "leftFeatures" and "rightFeatures" vectors.
     * Also there is no output since the method used refMap and mtcMap to store the results.
     * @param[in] featuresLeft The location of the features in the left image.
     * @param[in] featuresRight The location of the features in the right image.
     */
    CV_WRAP void quasiDenseMatching(const std::vector< cv::Point2f > &featuresLeft,
                                    const std::vector< cv::Point2f > &featuresRight);


    /**
     * @brief Compute the disparity map based on the Euclidean distance of corresponding points.
     * @param[in] matchMap A matrix of points, the same size as the left channel. Each cell of this
     * matrix stores the location of the corresponding point in the right image.
     * @param[out] dispMat The disparity map.
     * @sa quantizeDisparity
     * @sa getDisparity
     */
    CV_WRAP void computeDisparity(const cv::Mat_<cv::Point2i> &matchMap,
                                    cv::Mat_<float> &dispMat);


    /**
     * @brief Disparity map normalization for display purposes. If needed specify the quantization level as input argument.
     * @param[in] dispMat The disparity Map.
     * @param[in] lvls The quantization level of the output disparity map.
     * @return Disparity image.
     * @note Stores the output in the disparityImage class variable.
     * @sa computeDisparity
     * @sa quantiseDisparity
     */
    CV_WRAP cv::Mat quantiseDisparity(const cv::Mat_<float> &dispMat, const int lvls);


    /**
     * @brief Compute the Zero-mean Normalized Cross-correlation.
     *
     * Compare a patch in the left image, centered in point p0 with a patch in the right image, centered in point p1.
     * Patches are defined by wy, wx and the patch size is (2*wx+1) by (2*wy+1).
     * @param [in] p0 The central point of the patch in the left image.
     * @param [in] p1 The central point of the patch in the right image.
     * @param [in] wx The distance from the center of the patch to the border in the x direction.
     * @param [in] wy The distance from the center of the patch to the border in the y direction.
     * @return The value of the the zero-mean normalized cross correlation.
     * @note Default value for wx, wy is 1. in this case the patch is 3x3.
     */
    CV_WRAP float iZNCC_c1(const cv::Point2i p0, const cv::Point2i p1, const int wx=1, const int wy=1);


    /**
     * @brief Compute the sum of values and the sum of squared values of a patch with dimensions
     * 2*xWindow+1 by 2*yWindow+1 and centered in point p, using the integral image and integral image of squared pixel values.
     * @param[in] p The center of the patch we want to calculate the sum and sum of squared values.
     * @param[in] s The integral image
     * @param[in] ss The integral image of squared values.
     * @param[out] sum The sum of pixels inside the patch.
     * @param[out] ssum The sum of squared values inside the patch.
     * @param [in] xWindow The distance from the central pixel of the patch to the border in x direction.
     * @param [in] yWindow The distance from the central pixel of the patch to the border in y direction.
     * @note Default value for xWindow, yWindow is 1. in this case the patch is 3x3.
     * @note integral images are very useful to sum values of patches in constant time independent of their
     * size. For more information refer to the cv::Integral function OpenCV page.
     */
    CV_WRAP void patchSumSum2(const cv::Point2i p, const cv::Mat &sum, const cv::Mat &ssum,
                               float &s, float &ss, const int xWindow=1, const int yWindow=1);


    /**
     * @brief Create a priority queue containing sparse Matches
     *
     * This method computes the zncc for each Match extracted in "sparseMatching". If the zncc is over
     * the correlation threshold then the Match is inserted in the output priority queue.
     * @param[in] featuresLeft The feature locations in the left image.
     * @param[in] featuresRight The features locations in the right image.
     * @param[out] leftMap A matrix of points, of the same size as the left image. Each cell of this
     * matrix stores the location of the corresponding point in the right image.
     * @param[out] rightMap A matrix of points, the same size as the right image. Each cell of this
     * matrix stores the location of the corresponding point in the left image.
     * @return Priority queue containing sparse matches.
     */
    CV_WRAP t_matchPriorityQueue extractSparseSeeds(const std::vector< cv::Point2f > &featuresLeft,
                                                    const std::vector< cv::Point2f >  &featuresRight,
                                                    cv::Mat_<cv::Point2i> &leftMap,
                                                    cv::Mat_<cv::Point2i> &rightMap);


    /**
     * @brief Check if a match is close to the boarder of an image.
     * @param[in] m The match containing points in both image.
     * @param[in] bx The offset of the image edge that defines the border in x direction.
     * @param[in] by The offset of the image edge that defines the border in y direction.
     * @param[in] w The width of the image.
     * @param[in] h The height of the image.
     * @retval true If the feature is in the border of the image.
     * @retval false If the feature is not in the border of image.
     */
    CV_WRAP bool CheckBorder(Match m, int bx, int by, int w, int h);


    /**
     * @brief Compare two matches based on their zncc correlation values.
     * @param[in] a First Match.
     * @param[in] b Second Match.
     * @retval true If b is equal or grater Match than a.
     * @retval false If b is less Match than a.
     */
    CV_WRAP bool MatchCompare(const Match a, const Match b);


    /**
     * @brief Build a texture descriptor
     * @param[in] img The image we need to compute the descriptor for.
     * @param[out] descriptor The texture descriptor of the image.
     */
    CV_WRAP void buildTextureDescriptor(cv::Mat &img,cv::Mat &descriptor);


    // Variables used at sparse feature extraction.
    // Container for left images' features, extracted with GFT algorithm.
    CV_PROP_RW std::vector< cv::Point2f > leftFeatures;
    // Container for right images' features, matching is done with LK flow algorithm.
    CV_PROP_RW std::vector< cv::Point2f > rightFeatures;

    // Width and height of a single image.
    CV_PROP_RW int width;
    CV_PROP_RW int height;
    CV_PROP_RW int dMatchesLen;
    // Containers to store input images.
    CV_PROP_RW cv::Mat grayLeft;
    CV_PROP_RW cv::Mat grayRight;
    // Containers to store the locations of each points pair.
    CV_PROP_RW cv::Mat_<cv::Point2i> refMap;
    CV_PROP_RW cv::Mat_<cv::Point2i> mtcMap;
    CV_PROP_RW cv::Mat_<int32_t> sum0;
    CV_PROP_RW cv::Mat_<int32_t> sum1;
    CV_PROP_RW cv::Mat_<double> ssum0;
    CV_PROP_RW cv::Mat_<double> ssum1;
    // Container to store the disparity un-normalized
    CV_PROP_RW cv::Mat_<float> disparity;
    // Container to store the disparity image.
    CV_PROP_RW cv::Mat_<uchar> disparityImg;
    // Containers to store textures descriptors.
    CV_PROP_RW cv::Mat_<int> textureDescLeft;
    CV_PROP_RW cv::Mat_<int> textureDescRight;

};

} //namespace cv
} //namespace qds

/** @}*/

#endif // __OPENCV_QUASI_DENSE_STEREO_H__
