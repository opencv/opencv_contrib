/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#ifndef __OPENCV_SPARSEMATCHINTERPOLATOR_HPP__
#define __OPENCV_SPARSEMATCHINTERPOLATOR_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv {
namespace ximgproc {

//! @addtogroup ximgproc_filters
//! @{

/** @brief Main interface for all filters, that take sparse matches as an
input and produce a dense per-pixel matching (optical flow) as an output.
 */
class CV_EXPORTS_W SparseMatchInterpolator : public Algorithm
{
public:
    /** @brief Interpolate input sparse matches.

    @param from_image first of the two matched images, 8-bit single-channel or three-channel.

    @param from_points points of the from_image for which there are correspondences in the
    to_image (Point2f vector or Mat of depth CV_32F)

    @param to_image second of the two matched images, 8-bit single-channel or three-channel.

    @param to_points points in the to_image corresponding to from_points
    (Point2f vector or Mat of depth CV_32F)

    @param dense_flow output dense matching (two-channel CV_32F image)
     */
    CV_WRAP virtual void interpolate(InputArray from_image, InputArray from_points,
                                     InputArray to_image  , InputArray to_points,
                                     OutputArray dense_flow) = 0;
};

/** @brief Sparse match interpolation algorithm based on modified locally-weighted affine
estimator from @cite Revaud2015 and Fast Global Smoother as post-processing filter.
 */
class CV_EXPORTS_W EdgeAwareInterpolator : public SparseMatchInterpolator
{
public:
    /** @brief Interface to provide a more elaborated cost map, i.e. edge map, for the edge-aware term.
     *  This implementation is based on a rather simple gradient-based edge map estimation.
     *  To used more complex edge map estimator (e.g. StructuredEdgeDetection that has been
     *  used in the original publication) that may lead to improved accuracies, the internal
     *  edge map estimation can be bypassed here.
     *  @param _costMap a type CV_32FC1 Mat is required.
     *  @see cv::ximgproc::createSuperpixelSLIC
    */
    CV_WRAP virtual void setCostMap(const Mat & _costMap) = 0;
    /** @brief Parameter to tune the approximate size of the superpixel used for oversegmentation.
     *  @see cv::ximgproc::createSuperpixelSLIC
    */
    /** @brief K is a number of nearest-neighbor matches considered, when fitting a locally affine
    model. Usually it should be around 128. However, lower values would make the interpolation
    noticeably faster.
     */
    CV_WRAP virtual void setK(int _k) = 0;
    /** @see setK */
    CV_WRAP virtual int  getK() = 0;

    /** @brief Sigma is a parameter defining how fast the weights decrease in the locally-weighted affine
    fitting. Higher values can help preserve fine details, lower values can help to get rid of noise in the
    output flow.
     */
    CV_WRAP virtual void  setSigma(float _sigma) = 0;
    /** @see setSigma */
    CV_WRAP virtual float getSigma() = 0;

    /** @brief Lambda is a parameter defining the weight of the edge-aware term in geodesic distance,
    should be in the range of 0 to 1000.
     */
    CV_WRAP virtual void  setLambda(float _lambda) = 0;
    /** @see setLambda */
    CV_WRAP virtual float getLambda() = 0;

    /** @brief Sets whether the fastGlobalSmootherFilter() post-processing is employed. It is turned on by
    default.
     */
    CV_WRAP virtual void setUsePostProcessing(bool _use_post_proc) = 0;
    /** @see setUsePostProcessing */
    CV_WRAP virtual bool getUsePostProcessing() = 0;

    /** @brief Sets the respective fastGlobalSmootherFilter() parameter.
     */
    CV_WRAP virtual void  setFGSLambda(float _lambda) = 0;
    /** @see setFGSLambda */
    CV_WRAP virtual float getFGSLambda() = 0;

    /** @see setFGSLambda */
    CV_WRAP virtual void  setFGSSigma(float _sigma) = 0;
    /** @see setFGSLambda */
    CV_WRAP virtual float getFGSSigma() = 0;
};

/** @brief Factory method that creates an instance of the
EdgeAwareInterpolator.
*/
CV_EXPORTS_W
Ptr<EdgeAwareInterpolator> createEdgeAwareInterpolator();

/** @brief Sparse match interpolation algorithm based on modified piecewise locally-weighted affine
 * estimator called Robust Interpolation method of Correspondences or RIC from @cite Hu2017 and Variational
 * and Fast Global Smoother as post-processing filter. The RICInterpolator is a extension of the EdgeAwareInterpolator.
 * Main concept of this extension is an piece-wise affine model based on over-segmentation via SLIC superpixel estimation.
 * The method contains an efficient propagation mechanism to estimate among the pieces-wise models.
 */
class CV_EXPORTS_W RICInterpolator : public SparseMatchInterpolator
{
public:
    /** @brief K is a number of nearest-neighbor matches considered, when fitting a locally affine
     *model for a superpixel segment. However, lower values would make the interpolation
     *noticeably faster. The original implementation of @cite Hu2017 uses 32.
    */
    CV_WRAP virtual void setK(int k = 32) = 0;
    /** @copybrief setK
     *  @see setK
     */
    CV_WRAP virtual int getK() const = 0;
    /** @brief Interface to provide a more elaborated cost map, i.e. edge map, for the edge-aware term.
     *  This implementation is based on a rather simple gradient-based edge map estimation.
     *  To used more complex edge map estimator (e.g. StructuredEdgeDetection that has been
     *  used in the original publication) that may lead to improved accuracies, the internal
     *  edge map estimation can be bypassed here.
     *  @param costMap a type CV_32FC1 Mat is required.
     *  @see cv::ximgproc::createSuperpixelSLIC
    */
    CV_WRAP virtual void setCostMap(const Mat & costMap) = 0;
    /** @brief Get the internal cost, i.e. edge map, used for estimating the edge-aware term.
     *  @see setCostMap
     */
    CV_WRAP virtual void setSuperpixelSize(int spSize = 15) = 0;
    /** @copybrief setSuperpixelSize
     *  @see setSuperpixelSize
     */
    CV_WRAP virtual int getSuperpixelSize() const = 0;
    /** @brief Parameter defines the number of nearest-neighbor matches for each superpixel considered, when fitting a locally affine
     *model.
    */
    CV_WRAP virtual void setSuperpixelNNCnt(int spNN = 150) = 0;
    /** @copybrief setSuperpixelNNCnt
     *  @see setSuperpixelNNCnt
    */
    CV_WRAP virtual int getSuperpixelNNCnt() const = 0;
    /** @brief Parameter to tune enforcement of superpixel smoothness factor used for oversegmentation.
     *  @see cv::ximgproc::createSuperpixelSLIC
    */
    CV_WRAP virtual void setSuperpixelRuler(float ruler = 15.f) = 0;
    /** @copybrief setSuperpixelRuler
     *  @see setSuperpixelRuler
     */
    CV_WRAP virtual float  getSuperpixelRuler() const = 0;
    /** @brief Parameter to choose superpixel algorithm variant to use:
     * - cv::ximgproc::SLICType SLIC segments image using a desired region_size (value: 100)
     * - cv::ximgproc::SLICType SLICO will optimize using adaptive compactness factor (value: 101)
     * - cv::ximgproc::SLICType MSLIC will optimize using manifold methods resulting in more content-sensitive superpixels (value: 102).
     *  @see cv::ximgproc::createSuperpixelSLIC
    */
    CV_WRAP virtual void setSuperpixelMode(int mode = 100) = 0;
    /** @copybrief setSuperpixelMode
     *  @see setSuperpixelMode
     */
    CV_WRAP virtual int getSuperpixelMode() const = 0;
    /** @brief Alpha is a parameter defining a global weight for transforming geodesic distance into weight.
     */
    CV_WRAP virtual void setAlpha(float alpha = 0.7f) = 0;
    /** @copybrief setAlpha
     *  @see setAlpha
     */
    CV_WRAP virtual float getAlpha() const = 0;
    /** @brief Parameter defining the number of iterations for piece-wise affine model estimation.
     */
    CV_WRAP virtual void setModelIter(int modelIter = 4) = 0;
    /** @copybrief setModelIter
     *  @see setModelIter
     */
    CV_WRAP virtual int getModelIter() const = 0;
    /** @brief Parameter to choose wether additional refinement of the piece-wise affine models is employed.
    */
    CV_WRAP virtual void setRefineModels(bool refineModles = true) = 0;
    /** @copybrief setRefineModels
     *  @see setRefineModels
     */
    CV_WRAP virtual bool getRefineModels() const = 0;
    /** @brief MaxFlow is a threshold to validate the predictions using a certain piece-wise affine model.
     * If the prediction exceeds the treshold the translational model will be applied instead.
    */
    CV_WRAP virtual void setMaxFlow(float maxFlow = 250.f) = 0;
    /** @copybrief setMaxFlow
     *  @see setMaxFlow
     */
    CV_WRAP virtual float getMaxFlow() const = 0;
    /** @brief Parameter to choose wether the VariationalRefinement post-processing  is employed.
    */
    CV_WRAP virtual void setUseVariationalRefinement(bool use_variational_refinement = false) = 0;
    /** @copybrief setUseVariationalRefinement
     *  @see setUseVariationalRefinement
     */
    CV_WRAP virtual bool  getUseVariationalRefinement() const = 0;
    /** @brief Sets whether the fastGlobalSmootherFilter() post-processing is employed.
    */
    CV_WRAP virtual void setUseGlobalSmootherFilter(bool use_FGS = true) = 0;
    /** @copybrief setUseGlobalSmootherFilter
     *  @see setUseGlobalSmootherFilter
     */
    CV_WRAP virtual bool getUseGlobalSmootherFilter() const = 0;
    /** @brief Sets the respective fastGlobalSmootherFilter() parameter.
     */
    CV_WRAP virtual void  setFGSLambda(float lambda = 500.f) = 0;
    /** @copybrief setFGSLambda
     *  @see setFGSLambda
     */
    CV_WRAP virtual float getFGSLambda() const = 0;
    /** @brief Sets the respective fastGlobalSmootherFilter() parameter.
     */
    CV_WRAP virtual void  setFGSSigma(float sigma = 1.5f) = 0;
    /** @copybrief setFGSSigma
     *  @see setFGSSigma
     */
    CV_WRAP virtual float getFGSSigma() const = 0;
};

/** @brief Factory method that creates an instance of the
RICInterpolator.
*/
CV_EXPORTS_W
Ptr<RICInterpolator> createRICInterpolator();
//! @}
}
}
#endif
#endif
