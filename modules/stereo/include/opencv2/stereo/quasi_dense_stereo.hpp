// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//authors: Danail Stoyanov, Evangelos Mazomenos, Dimitrios Psychogyios


//__OPENCV_QUASI_DENSE_STEREO_H__
#ifndef __OPENCV_QUASI_DENSE_STEREO_H__
#define __OPENCV_QUASI_DENSE_STEREO_H__



#include <opencv2/core.hpp>


namespace cv
{
namespace stereo
{
/** \addtogroup stereo
 *  @{
 */


// A basic match structure
struct CV_EXPORTS_W_SIMPLE MatchQuasiDense
{
    CV_PROP_RW cv::Point2i p0;
    CV_PROP_RW cv::Point2i p1;
    CV_PROP_RW float	corr;

    CV_WRAP MatchQuasiDense() { corr = 0; }

    CV_WRAP_AS(apply) bool operator < (const MatchQuasiDense & rhs) const//fixme  may be used uninitialized in this function
    {
        return this->corr < rhs.corr;
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


/**
 * @brief Class containing the methods needed for Quasi Dense Stereo computation.
 *
 * This module contains the code to perform quasi dense stereo matching.
 * The method initially starts with a sparse 3D reconstruction based on feature matching across a
 * stereo image pair and subsequently propagates the structure into neighboring image regions.
 * To obtain initial seed correspondences, the algorithm locates Shi and Tomashi features in the
 * left image of the stereo pair and then tracks them using pyramidal Lucas-Kanade in the right image.
 * To densify the sparse correspondences, the algorithm computes the zero-mean normalized
 * cross-correlation (ZNCC) in small patches around every seed pair and uses it as a quality metric
 * for each match. In this code, we introduce a custom structure to store the location and ZNCC value
 * of correspondences called "Match". Seed Matches are stored in a priority queue sorted according to
 * their ZNCC value, allowing for the best quality Match to be readily available. The algorithm pops
 * Matches and uses them to extract new matches around them. This is done by considering a small
 * neighboring area around each Seed and retrieving correspondences above a certain texture threshold
 * that are not previously computed. New matches are stored in the seed priority queue and used as seeds.
 * The propagation process ends when no additional matches can be retrieved.
 *
 *
 * @sa This code represents the work presented in @cite Stoyanov2010.
 * If this code is useful for your work please cite @cite Stoyanov2010.
 *
 * Also the original growing scheme idea is described in @cite Lhuillier2000
 *
 */

class  CV_EXPORTS_W QuasiDenseStereo
{
public:
    /**
     * @brief destructor
     * Method to free all the memory allocated by matrices and vectors in this class.
     */
    CV_WRAP virtual ~QuasiDenseStereo() = 0;


    /**
     * @brief Load a file containing the configuration parameters of the class.
     * @param[in] filepath The location of the .YAML file containing the configuration parameters.
     * @note default value is an empty string in which case the default parameters will be loaded.
     * @retval 1: If the path is not empty and the program loaded the parameters successfully.
     * @retval 0: If the path is empty and the program loaded default parameters.
     * @retval -1: If the file location is not valid or the program could not open the file and
     * loaded default parameters from defaults.hpp.
     * @note The method is automatically called in the constructor and configures the class.
     * @note Loading different parameters will have an effect on the output. This is useful for tuning
     * in case of video processing.
     * @sa loadParameters
     */
    CV_WRAP virtual int loadParameters(cv::String filepath) = 0;


    /**
     * @brief Save a file containing all the configuration parameters the class is currently set to.
     * @param[in] filepath The location to store the parameters file.
     * @note Calling this method with no arguments will result in storing class parameters to a file
     * names "qds_parameters.yaml" in the root project folder.
     * @note This method can be used to generate a template file for tuning the class.
     * @sa loadParameters
     */
    CV_WRAP virtual int saveParameters(cv::String filepath) = 0;


    /**
     * @brief Get The sparse corresponding points.
     * @param[out] sMatches A vector containing all sparse correspondences.
     * @note The method clears the sMatches vector.
     * @note The returned Match elements inside the sMatches vector, do not use corr member.
     */
    CV_WRAP virtual void getSparseMatches(CV_OUT std::vector<MatchQuasiDense> &sMatches) = 0;


    /**
     * @brief Get The dense corresponding points.
     * @param[out] denseMatches A vector containing all dense matches.
     * @note The method clears the denseMatches vector.
     * @note The returned Match elements inside the sMatches vector, do not use corr member.
     */
    CV_WRAP virtual void getDenseMatches(CV_OUT  std::vector<MatchQuasiDense> &denseMatches) = 0;



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
    CV_WRAP virtual void process(const cv::Mat &imgLeft ,const cv::Mat &imgRight) = 0;


    /**
     * @brief Specify pixel coordinates in the left image and get its corresponding location in the right image.
     * @param[in] x The x pixel coordinate in the left image channel.
     * @param[in] y The y pixel coordinate in the left image channel.
     * @retval cv::Point(x, y) The location of the corresponding pixel in the right image.
     * @retval cv::Point(0, 0) (NO_MATCH)  if no match is found in the right image for the specified pixel location in the left image.
     * @note This method should be always called after process, otherwise the matches will not be correct.
     */
    CV_WRAP virtual cv::Point2f getMatch(const int x, const int y) = 0;


    /**
     * @brief Compute and return the disparity map based on the correspondences found in the "process" method.
     * @note Default level is 50
     * @return cv::Mat containing a the disparity image in grayscale.
     * @sa computeDisparity
     * @sa quantizeDisparity
     */
    CV_WRAP virtual cv::Mat getDisparity() = 0;


    CV_WRAP static cv::Ptr<QuasiDenseStereo> create(cv::Size monoImgSize, cv::String paramFilepath = cv::String());


    CV_PROP_RW PropagationParameters Param;
};

} //namespace cv
} //namespace stereo

/** @}*/

#endif // __OPENCV_QUASI_DENSE_STEREO_H__
