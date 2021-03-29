// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_OPTFLOW_RLOFFLOW_HPP__
#define __OPENCV_OPTFLOW_RLOFFLOW_HPP__

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"

namespace cv
{
namespace optflow
{
//! @addtogroup optflow
//! @{

enum SupportRegionType {
    SR_FIXED = 0,           /**<  Apply a constant support region */
    SR_CROSS = 1            /**<  Apply a adaptive support region obtained by cross-based segmentation
                             *    as described in @cite Senst2014
                            */
};
enum SolverType {
    ST_STANDART = 0,        /**< Apply standard iterative refinement */
    ST_BILINEAR = 1         /**< Apply optimized iterative refinement based bilinear equation solutions
                             *   as described in @cite Senst2013
                            */
};

enum InterpolationType
{
    INTERP_GEO = 0,    /**<  Fast geodesic interpolation, see @cite Geistert2016 */
    INTERP_EPIC = 1,   /**<  Edge-preserving interpolation using ximgproc::EdgeAwareInterpolator, see @cite Revaud2015,Geistert2016. */
    INTERP_RIC = 2,    /**<  SLIC based robust interpolation using ximgproc::RICInterpolator, see @cite Hu2017. */
};

/** @brief This is used store and set up the parameters of the robust local optical flow (RLOF) algoritm.
 *
 * The RLOF is a fast local optical flow approach described in @cite Senst2012 @cite Senst2013 @cite Senst2014
 * and @cite Senst2016 similar to the pyramidal iterative Lucas-Kanade method as
 * proposed by @cite Bouguet00. More details and experiments can be found in the following thesis @cite Senst2019.
 * The implementation is derived from optflow::calcOpticalFlowPyrLK().
 * This RLOF implementation can be seen as an improved pyramidal iterative Lucas-Kanade and includes
 * a set of improving modules. The main improvements in respect to the pyramidal iterative Lucas-Kanade
 * are:
 *  - A more robust redecending M-estimator framework (see @cite Senst2012) to improve the accuracy at
 *     motion boundaries and appearing and disappearing pixels.
 *  - an adaptive support region strategies to improve the accuracy at motion boundaries to reduce the
 *     corona effect, i.e oversmoothing of the PLK at motion/object boundaries. The cross-based segementation
 *     strategy (SR_CROSS) proposed in @cite Senst2014 uses a simple segmenation approach to obtain the optimal
 *     shape of the support region.
 *  - To deal with illumination changes (outdoor sequences and shadow) the intensity constancy assumption
 *     based optical flow equation has been adopt with the Gennert and Negahdaripour illumination model
 *     (see @cite Senst2016). This model can be switched on/off with the useIlluminationModel variable.
 *  - By using a global motion prior initialization (see @cite Senst2016) of the iterative refinement
 *     the accuracy could be significantly improved for large displacements. This initialization can be
 *     switched on and of with useGlobalMotionPrior variable.
 *
 * The RLOF can be computed with the SparseOpticalFlow class or function interface to track a set of features
 * or with the DenseOpticalFlow class or function interface to compute dense optical flow.
 *
 * @see optflow::DenseRLOFOpticalFlow, optflow::calcOpticalFlowDenseRLOF(), optflow::SparseRLOFOpticalFlow, optflow::calcOpticalFlowSparseRLOF()
 */
class CV_EXPORTS_W RLOFOpticalFlowParameter{
public:
    RLOFOpticalFlowParameter()
        :solverType(ST_BILINEAR)
        ,supportRegionType(SR_CROSS)
        ,normSigma0(std::numeric_limits<float>::max())
        ,normSigma1(std::numeric_limits<float>::max())
        ,smallWinSize(9)
        ,largeWinSize(21)
        ,crossSegmentationThreshold(25)
        ,maxLevel(4)
        ,useInitialFlow(false)
        ,useIlluminationModel(true)
        ,useGlobalMotionPrior(true)
        ,maxIteration(30)
        ,minEigenValue(0.0001f)
        ,globalMotionRansacThreshold(10)
    {}

    SolverType solverType;
    /**< Variable specifies the iterative refinement strategy. Please consider citing  @cite Senst2013 when
     *  using ST_BILINEAR.
    */

    SupportRegionType supportRegionType;
    /**< Variable specifies the support region shape extraction or shrinking strategy.
    */

    float normSigma0;
    /**< &sigma parameter of the shrinked Hampel norm introduced in @cite Senst2012. If
     * &sigma = std::numeric_limist<float>::max() the least-square estimator will be used
     * instead of the M-estimator. Althoug M-estimator is more robust against outlier in the support
     * region the least-square can be fast in computation.
    */
    float normSigma1;
    /**< &sigma parameter of the shrinked Hampel norm introduced in @cite Senst2012. If
     * &sigma = std::numeric_limist<float>::max() the least-square estimator will be used
     * instead of the M-estimator. Althoug M-estimator is more robust against outlier in the support
     * region the least-square can be fast in computation.
    */
    int smallWinSize;
    /**< Minimal window size of the support region. This parameter is only used if supportRegionType is SR_CROSS.
    */
    int largeWinSize;
    /**< Maximal window size of the support region. If supportRegionType is SR_FIXED this gives the exact support
     * region size. The speed of the RLOF is related to the applied win sizes. The smaller the window size the lower is the runtime,
     * but the more sensitive to noise is the method.
    */
    int crossSegmentationThreshold;
    /**< Color similarity threshold used by cross-based segmentation following @cite Senst2014 .
     *   (Only used  if supportRegionType is SR_CROSS). With the cross-bassed segmentation
     *   motion boundaries can be computed more accurately.
    */
    int maxLevel;
    /**< Maximal number of pyramid level used. The large this value is the more likely it is
     *   to obtain accurate solutions for long-range motions. The runtime is linear related to
     *   this parameter.
    */
    bool useInitialFlow;
    /**< Use next point list as initial values. A good intialization can imporve the algortihm
     *   accuracy and reduce the runtime by a faster convergence of the iteration refinement.
    */
    bool useIlluminationModel;
    /**< Use the Gennert and Negahdaripour illumination model instead of the intensity brigthness
     *   constraint. (proposed in @cite Senst2016 ) This model is defined as follow:
     *   \f[ I(\mathbf{x},t) + m \cdot I(\mathbf{x},t) + c = I(\mathbf{x},t+1) \f]
     *   and contains with m and c a multiplicative and additive term which makes the estimate
     *   more robust against illumination changes. The computational complexity is increased by
     *   enabling the illumination model.
    */
    bool useGlobalMotionPrior;
    /**< Use global motion prior initialisation has been introduced in @cite Senst2016 . It
     *   allows to be more accurate for long-range motion. The computational complexity is
     *   slightly increased by enabling the global motion prior initialisation.
    */
    int maxIteration;
    /**< Number of maximal iterations used for the iterative refinement. Lower values can
     *   reduce the runtime but also the accuracy.
    */
    float minEigenValue;
    /**< Threshold for the minimal eigenvalue of the gradient matrix defines when to abort the
     *   iterative refinement.
    */
    float globalMotionRansacThreshold;
    /**< To apply the global motion prior motion vectors will be computed on a regulary sampled which
     *   are the basis for Homography estimation using RANSAC. The reprojection threshold is based on
     *   n-th percentil (given by this value [0 ... 100]) of the motion vectors magnitude.
     *   See @cite Senst2016 for more details.
    */

    //! @brief Enable M-estimator or disable and use least-square estimator.
    /** Enables M-estimator by setting sigma parameters to (3.2, 7.0). Disabling M-estimator can reduce
     *  runtime, while enabling can improve the accuracy.
     *  @param val If true M-estimator is used. If false least-square estimator is used.
     *    @see setNormSigma0, setNormSigma1
    */
    CV_WRAP void setUseMEstimator(bool val);

    CV_WRAP void setSolverType(SolverType val);
    CV_WRAP SolverType getSolverType() const;

    CV_WRAP void setSupportRegionType(SupportRegionType val);
    CV_WRAP SupportRegionType getSupportRegionType() const;

    CV_WRAP void setNormSigma0(float val);
    CV_WRAP float getNormSigma0() const;

    CV_WRAP void setNormSigma1(float val);
    CV_WRAP float getNormSigma1() const;

    CV_WRAP void setSmallWinSize(int val);
    CV_WRAP int getSmallWinSize() const;

    CV_WRAP void setLargeWinSize(int val);
    CV_WRAP int getLargeWinSize() const;

    CV_WRAP void setCrossSegmentationThreshold(int val);
    CV_WRAP int getCrossSegmentationThreshold() const;

    CV_WRAP void setMaxLevel(int val);
    CV_WRAP int getMaxLevel() const;

    CV_WRAP void setUseInitialFlow(bool val);
    CV_WRAP bool getUseInitialFlow() const;

    CV_WRAP void setUseIlluminationModel(bool val);
    CV_WRAP bool getUseIlluminationModel() const;

    CV_WRAP void setUseGlobalMotionPrior(bool val);
    CV_WRAP bool getUseGlobalMotionPrior() const;

    CV_WRAP void setMaxIteration(int val);
    CV_WRAP int getMaxIteration() const;

    CV_WRAP void setMinEigenValue(float val);
    CV_WRAP float getMinEigenValue() const;

    CV_WRAP void setGlobalMotionRansacThreshold(float val);
    CV_WRAP float getGlobalMotionRansacThreshold() const;

    //! @brief Creates instance of optflow::RLOFOpticalFlowParameter
    CV_WRAP static Ptr<RLOFOpticalFlowParameter> create();
};

/** @brief Fast dense optical flow computation based on robust local optical flow (RLOF) algorithms and sparse-to-dense interpolation
 * scheme.
 *
 * The RLOF is a fast local optical flow approach described in @cite Senst2012 @cite Senst2013 @cite Senst2014
 * and @cite Senst2016 similar to the pyramidal iterative Lucas-Kanade method as
 * proposed by @cite Bouguet00. More details and experiments can be found in the following thesis @cite Senst2019.
 * The implementation is derived from optflow::calcOpticalFlowPyrLK().
 *
 * The sparse-to-dense interpolation scheme allows for fast computation of dense optical flow using RLOF (see @cite Geistert2016).
 * For this scheme the following steps are applied:
 * -# motion vector seeded at a regular sampled grid are computed. The sparsity of this grid can be configured with setGridStep
 * -# (optinally) errornous motion vectors are filter based on the forward backward confidence. The threshold can be configured
 * with setForwardBackward. The filter is only applied if the threshold >0 but than the runtime is doubled due to the estimation
 * of the backward flow.
 * -# Vector field interpolation is applied to the motion vector set to obtain a dense vector field.
 *
 * For the RLOF configuration see optflow::RLOFOpticalFlowParameter for further details.
 * Parameters have been described in @cite Senst2012 @cite Senst2013 @cite Senst2014 and @cite Senst2016.
 *
 * @note If the grid size is set to (1,1) and the forward backward threshold <= 0 than pixelwise dense optical flow field is
 * computed by RLOF without using interpolation.
 *
 * @note Note that in output, if no correspondences are found between \a I0 and \a I1, the \a flow is set to 0.
 * @see optflow::calcOpticalFlowDenseRLOF(), optflow::RLOFOpticalFlowParameter
*/
class CV_EXPORTS_W DenseRLOFOpticalFlow : public DenseOpticalFlow
{
public:
    //! @brief Configuration of the RLOF alogrithm.
    /**
        @see optflow::RLOFOpticalFlowParameter, getRLOFOpticalFlowParameter
    */
    CV_WRAP virtual void setRLOFOpticalFlowParameter(Ptr<RLOFOpticalFlowParameter>  val) = 0;
    /** @copybrief setRLOFOpticalFlowParameter
        @see optflow::RLOFOpticalFlowParameter, setRLOFOpticalFlowParameter
    */
    CV_WRAP virtual Ptr<RLOFOpticalFlowParameter>  getRLOFOpticalFlowParameter() const = 0;
    //! @brief Threshold for the forward backward confidence check
    /**For each grid point \f$ \mathbf{x} \f$ a motion vector \f$ d_{I0,I1}(\mathbf{x}) \f$ is computed.
     *     If the forward backward error \f[ EP_{FB} = || d_{I0,I1} + d_{I1,I0} || \f]
     *     is larger than threshold given by this function then the motion vector will not be used by the following
     *    vector field interpolation. \f$ d_{I1,I0} \f$ denotes the backward flow. Note, the forward backward test
     *    will only be applied if the threshold > 0. This may results into a doubled runtime for the motion estimation.
     *    @see getForwardBackward, setGridStep
    */
    CV_WRAP virtual void setForwardBackward(float val) = 0;
    /** @copybrief setForwardBackward
        @see setForwardBackward
    */
    CV_WRAP virtual float getForwardBackward() const = 0;
    //! @brief Size of the grid to spawn the motion vectors.
    /** For each grid point a motion vector is computed. Some motion vectors will be removed due to the forwatd backward
     *  threshold (if set >0). The rest will be the base of the vector field interpolation.
     *    @see getForwardBackward, setGridStep
    */
    CV_WRAP virtual Size getGridStep() const = 0;
    /** @copybrief getGridStep
     *    @see getGridStep
     */
    CV_WRAP virtual void setGridStep(Size val) = 0;

    //! @brief Interpolation used to compute the dense optical flow.
    /** Two interpolation algorithms are supported
     * - **INTERP_GEO** applies the fast geodesic interpolation, see @cite Geistert2016.
     * - **INTERP_EPIC_RESIDUAL** applies the edge-preserving interpolation, see @cite Revaud2015,Geistert2016.
     * @see ximgproc::EdgeAwareInterpolator, getInterpolation
    */
    CV_WRAP virtual void setInterpolation(InterpolationType val) = 0;
    /** @copybrief setInterpolation
     *    @see ximgproc::EdgeAwareInterpolator, setInterpolation
     */
    CV_WRAP virtual InterpolationType getInterpolation() const = 0;
    //! @brief see ximgproc::EdgeAwareInterpolator() K value.
    /** K is a number of nearest-neighbor matches considered, when fitting a locally affine
     *    model. Usually it should be around 128. However, lower values would make the interpolation noticeably faster.
     *    @see ximgproc::EdgeAwareInterpolator,  setEPICK
    */
    CV_WRAP virtual int getEPICK() const = 0;
    /** @copybrief getEPICK
     *    @see ximgproc::EdgeAwareInterpolator, getEPICK
     */
    CV_WRAP virtual void setEPICK(int val) = 0;
    //! @brief see ximgproc::EdgeAwareInterpolator() sigma value.
    /** Sigma is a parameter defining how fast the weights decrease in the locally-weighted affine
     *  fitting. Higher values can help preserve fine details, lower values can help to get rid of noise in the
     *  output flow.
     *    @see ximgproc::EdgeAwareInterpolator, setEPICSigma
    */
    CV_WRAP virtual float getEPICSigma() const = 0;
    /** @copybrief getEPICSigma
     *  @see ximgproc::EdgeAwareInterpolator, getEPICSigma
     */
    CV_WRAP virtual void setEPICSigma(float val) = 0;
    //! @brief  see ximgproc::EdgeAwareInterpolator() lambda value.
    /** Lambda is a parameter defining the weight of the edge-aware term in geodesic distance,
     *    should be in the range of 0 to 1000.
     *    @see ximgproc::EdgeAwareInterpolator, setEPICSigma
    */
    CV_WRAP virtual float getEPICLambda() const = 0;
    /** @copybrief getEPICLambda
     *    @see ximgproc::EdgeAwareInterpolator, getEPICLambda
    */
    CV_WRAP virtual void setEPICLambda(float val) = 0;
    //! @brief see ximgproc::EdgeAwareInterpolator().
    /** Sets the respective fastGlobalSmootherFilter() parameter.
     *    @see ximgproc::EdgeAwareInterpolator, setFgsLambda
    */
    CV_WRAP virtual float getFgsLambda() const = 0;
    /** @copybrief getFgsLambda
     *    @see ximgproc::EdgeAwareInterpolator, ximgproc::fastGlobalSmootherFilter, getFgsLambda
    */
    CV_WRAP virtual void setFgsLambda(float val) = 0;
    //! @brief see ximgproc::EdgeAwareInterpolator().
    /** Sets the respective fastGlobalSmootherFilter() parameter.
     *    @see ximgproc::EdgeAwareInterpolator, ximgproc::fastGlobalSmootherFilter, setFgsSigma
    */
    CV_WRAP virtual float getFgsSigma() const = 0;
    /** @copybrief getFgsSigma
     *    @see ximgproc::EdgeAwareInterpolator, ximgproc::fastGlobalSmootherFilter, getFgsSigma
     */
    CV_WRAP virtual void setFgsSigma(float val) = 0;
    //! @brief enables ximgproc::fastGlobalSmootherFilter
    /**
     * @see getUsePostProc
     */
    CV_WRAP virtual void setUsePostProc(bool val) = 0;
    /** @copybrief setUsePostProc
     *    @see ximgproc::fastGlobalSmootherFilter, setUsePostProc
     */
    CV_WRAP virtual bool getUsePostProc() const = 0;
    //! @brief enables VariationalRefinement
    /**
     * @see getUseVariationalRefinement
     */
    CV_WRAP virtual void setUseVariationalRefinement(bool val) = 0;
    /** @copybrief setUseVariationalRefinement
     *    @see ximgproc::fastGlobalSmootherFilter, setUsePostProc
     */
    CV_WRAP virtual bool getUseVariationalRefinement() const = 0;
    //! @brief Parameter to tune the approximate size of the superpixel used for oversegmentation.
    /**
     * @see cv::ximgproc::createSuperpixelSLIC, cv::ximgproc::RICInterpolator
     */
    CV_WRAP virtual void setRICSPSize(int val) = 0;
    /** @copybrief setRICSPSize
    *    @see setRICSPSize
    */
    CV_WRAP virtual int  getRICSPSize() const = 0;
    /** @brief Parameter to choose superpixel algorithm variant to use:
     * - cv::ximgproc::SLICType SLIC segments image using a desired region_size (value: 100)
     * - cv::ximgproc::SLICType SLICO will optimize using adaptive compactness factor (value: 101)
     * - cv::ximgproc::SLICType MSLIC will optimize using manifold methods resulting in more content-sensitive superpixels (value: 102).
     *  @see cv::ximgproc::createSuperpixelSLIC, cv::ximgproc::RICInterpolator
    */
    CV_WRAP virtual void setRICSLICType(int val) = 0;
    /** @copybrief setRICSLICType
     *    @see setRICSLICType
     */
    CV_WRAP virtual int  getRICSLICType() const = 0;
    //! @brief Creates instance of optflow::DenseRLOFOpticalFlow
    /**
     *    @param rlofParam see optflow::RLOFOpticalFlowParameter
     *    @param forwardBackwardThreshold see setForwardBackward
     *    @param gridStep see setGridStep
     *    @param interp_type see setInterpolation
     *    @param epicK see setEPICK
     *    @param epicSigma see setEPICSigma
     *    @param epicLambda see setEPICLambda
     *    @param ricSPSize see setRICSPSize
     *    @param ricSLICType see setRICSLICType
     *    @param use_post_proc see setUsePostProc
     *    @param fgsLambda see setFgsLambda
     *    @param fgsSigma see setFgsSigma
     *    @param use_variational_refinement see setUseVariationalRefinement
    */
    CV_WRAP static Ptr<DenseRLOFOpticalFlow> create(
        Ptr<RLOFOpticalFlowParameter> rlofParam = Ptr<RLOFOpticalFlowParameter>(),
        float forwardBackwardThreshold = 1.f,
        Size gridStep = Size(6, 6),
        InterpolationType interp_type = InterpolationType::INTERP_EPIC,
        int epicK = 128,
        float epicSigma = 0.05f,
        float epicLambda = 999.0f,
        int ricSPSize = 15,
        int ricSLICType = 100,
        bool use_post_proc = true,
        float fgsLambda = 500.0f,
        float fgsSigma = 1.5f,
        bool use_variational_refinement = false);
};

/** @brief Class used for calculation sparse optical flow and feature tracking with robust local optical flow (RLOF) algorithms.
*
* The RLOF is a fast local optical flow approach described in @cite Senst2012 @cite Senst2013 @cite Senst2014
 * and @cite Senst2016 similar to the pyramidal iterative Lucas-Kanade method as
* proposed by @cite Bouguet00. More details and experiments can be found in the following thesis @cite Senst2019.
* The implementation is derived from optflow::calcOpticalFlowPyrLK().
*
* For the RLOF configuration see optflow::RLOFOpticalFlowParameter for further details.
* Parameters have been described in @cite Senst2012, @cite Senst2013, @cite Senst2014 and @cite Senst2016.
*
* @note SIMD parallelization is only available when compiling with SSE4.1.
* @see optflow::calcOpticalFlowSparseRLOF(), optflow::RLOFOpticalFlowParameter
*/
class CV_EXPORTS_W SparseRLOFOpticalFlow : public SparseOpticalFlow
{
public:
    /** @copydoc DenseRLOFOpticalFlow::setRLOFOpticalFlowParameter
    */
    CV_WRAP virtual void setRLOFOpticalFlowParameter(Ptr<RLOFOpticalFlowParameter> val) = 0;
    /** @copybrief setRLOFOpticalFlowParameter
     *    @see setRLOFOpticalFlowParameter
    */
    CV_WRAP virtual Ptr<RLOFOpticalFlowParameter>  getRLOFOpticalFlowParameter() const = 0;
    //! @brief Threshold for the forward backward confidence check
    /** For each feature point a motion vector \f$ d_{I0,I1}(\mathbf{x}) \f$ is computed.
     *     If the forward backward error \f[ EP_{FB} = || d_{I0,I1} + d_{I1,I0} || \f]
     *     is larger than threshold given by this function then the status  will not be used by the following
     *    vector field interpolation. \f$ d_{I1,I0} \f$ denotes the backward flow. Note, the forward backward test
     *    will only be applied if the threshold > 0. This may results into a doubled runtime for the motion estimation.
     *    @see setForwardBackward
    */
    CV_WRAP virtual void setForwardBackward(float val) = 0;
    /** @copybrief setForwardBackward
     *    @see setForwardBackward
    */
    CV_WRAP virtual float getForwardBackward() const = 0;

    //! @brief Creates instance of SparseRLOFOpticalFlow
    /**
     *    @param rlofParam see setRLOFOpticalFlowParameter
     *    @param forwardBackwardThreshold see setForwardBackward
    */
    CV_WRAP static Ptr<SparseRLOFOpticalFlow> create(
        Ptr<RLOFOpticalFlowParameter> rlofParam = Ptr<RLOFOpticalFlowParameter>(),
        float forwardBackwardThreshold = 1.f);

};

/** @brief Fast dense optical flow computation based on robust local optical flow (RLOF) algorithms and sparse-to-dense interpolation scheme.
 *
 * The RLOF is a fast local optical flow approach described in @cite Senst2012 @cite Senst2013 @cite Senst2014
 * and @cite Senst2016 similar to the pyramidal iterative Lucas-Kanade method as
 * proposed by @cite Bouguet00. More details and experiments can be found in the following thesis @cite Senst2019.
 * The implementation is derived from optflow::calcOpticalFlowPyrLK().
 *
 * The sparse-to-dense interpolation scheme allows for fast computation of dense optical flow using RLOF (see @cite Geistert2016).
 * For this scheme the following steps are applied:
 * -# motion vector seeded at a regular sampled grid are computed. The sparsity of this grid can be configured with setGridStep
 * -# (optinally) errornous motion vectors are filter based on the forward backward confidence. The threshold can be configured
 * with setForwardBackward. The filter is only applied if the threshold >0 but than the runtime is doubled due to the estimation
 * of the backward flow.
 * -# Vector field interpolation is applied to the motion vector set to obtain a dense vector field.
 *
 * @param I0 first 8-bit input image. If The cross-based RLOF is used (by selecting optflow::RLOFOpticalFlowParameter::supportRegionType
 * = SupportRegionType::SR_CROSS) image has to be a 8-bit 3 channel image.
 * @param I1 second 8-bit input image. If The cross-based RLOF is used (by selecting optflow::RLOFOpticalFlowParameter::supportRegionType
 * = SupportRegionType::SR_CROSS) image has to be a 8-bit 3 channel image.
 * @param flow computed flow image that has the same size as I0 and type CV_32FC2.
 * @param rlofParam see optflow::RLOFOpticalFlowParameter
 * @param forwardBackwardThreshold Threshold for the forward backward confidence check.
 * For each grid point \f$ \mathbf{x} \f$ a motion vector \f$ d_{I0,I1}(\mathbf{x}) \f$ is computed.
 * If the forward backward error \f[ EP_{FB} = || d_{I0,I1} + d_{I1,I0} || \f]
 * is larger than threshold given by this function then the motion vector will not be used by the following
 * vector field interpolation. \f$ d_{I1,I0} \f$ denotes the backward flow. Note, the forward backward test
 *    will only be applied if the threshold > 0. This may results into a doubled runtime for the motion estimation.
 * @param gridStep Size of the grid to spawn the motion vectors. For each grid point a motion vector is computed.
 * Some motion vectors will be removed due to the forwatd backward threshold (if set >0). The rest will be the
 * base of the vector field interpolation.
 * @param interp_type interpolation method used to compute the dense optical flow. Two interpolation algorithms are
 * supported:
 * - **INTERP_GEO** applies the fast geodesic interpolation, see @cite Geistert2016.
 * - **INTERP_EPIC_RESIDUAL** applies the edge-preserving interpolation, see @cite Revaud2015,Geistert2016.
 * @param epicK see ximgproc::EdgeAwareInterpolator sets the respective parameter.
 * @param epicSigma see ximgproc::EdgeAwareInterpolator sets the respective parameter.
 * @param epicLambda see ximgproc::EdgeAwareInterpolator sets the respective parameter.
 * @param ricSPSize  see ximgproc::RICInterpolator sets the respective parameter.
 * @param ricSLICType see ximgproc::RICInterpolator sets the respective parameter.
 * @param use_post_proc enables ximgproc::fastGlobalSmootherFilter() parameter.
 * @param fgsLambda sets the respective ximgproc::fastGlobalSmootherFilter() parameter.
 * @param fgsSigma sets the respective ximgproc::fastGlobalSmootherFilter() parameter.
 * @param use_variational_refinement enables VariationalRefinement
 *
 * Parameters have been described in @cite Senst2012, @cite Senst2013, @cite Senst2014, @cite Senst2016.
 * For the RLOF configuration see optflow::RLOFOpticalFlowParameter for further details.
 * @note If the grid size is set to (1,1) and the forward backward threshold <= 0 that the dense optical flow field is purely
 * computed with the RLOF.
 *
 * @note SIMD parallelization is only available when compiling with SSE4.1.
 * @note Note that in output, if no correspondences are found between \a I0 and \a I1, the \a flow is set to 0.
 *
 * @sa optflow::DenseRLOFOpticalFlow, optflow::RLOFOpticalFlowParameter
*/
CV_EXPORTS_W void calcOpticalFlowDenseRLOF(InputArray I0, InputArray I1, InputOutputArray flow,
    Ptr<RLOFOpticalFlowParameter> rlofParam = Ptr<RLOFOpticalFlowParameter>(),
    float forwardBackwardThreshold = 0, Size gridStep = Size(6, 6),
    InterpolationType interp_type = InterpolationType::INTERP_EPIC,
    int epicK = 128, float epicSigma = 0.05f, float epicLambda = 100.f,
    int ricSPSize = 15, int ricSLICType = 100,
    bool use_post_proc = true, float fgsLambda = 500.0f, float fgsSigma = 1.5f,
    bool use_variational_refinement = false);

/** @brief Calculates fast optical flow for a sparse feature set using the robust local optical flow (RLOF) similar
* to optflow::calcOpticalFlowPyrLK().
*
* The RLOF is a fast local optical flow approach described in @cite Senst2012 @cite Senst2013 @cite Senst2014
 * and @cite Senst2016 similar to the pyramidal iterative Lucas-Kanade method as
* proposed by @cite Bouguet00. More details and experiments can be found in the following thesis @cite Senst2019.
* The implementation is derived from optflow::calcOpticalFlowPyrLK().
*
* @param prevImg first 8-bit input image. If The cross-based RLOF is used (by selecting optflow::RLOFOpticalFlowParameter::supportRegionType
* = SupportRegionType::SR_CROSS) image has to be a 8-bit 3 channel image.
* @param nextImg second 8-bit input image. If The cross-based RLOF is used (by selecting optflow::RLOFOpticalFlowParameter::supportRegionType
* = SupportRegionType::SR_CROSS) image has to be a 8-bit 3 channel image.
* @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be single-precision
* floating-point numbers.
* @param nextPts output vector of 2D points (with single-precision floating-point coordinates) containing the calculated
* new positions of input features in the second image; when optflow::RLOFOpticalFlowParameter::useInitialFlow variable is true  the vector must
* have the same size as in the input and contain the initialization point correspondences.
* @param status output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the
* corresponding features has passed the forward backward check.
* @param err output vector of errors; each element of the vector is set to the forward backward error for the corresponding feature.
* @param rlofParam see optflow::RLOFOpticalFlowParameter
* @param forwardBackwardThreshold Threshold for the forward backward confidence check. If forewardBackwardThreshold <=0 the forward
*
* @note SIMD parallelization is only available when compiling with SSE4.1.
*
* Parameters have been described in @cite Senst2012, @cite Senst2013, @cite Senst2014 and @cite Senst2016.
* For the RLOF configuration see optflow::RLOFOpticalFlowParameter for further details.
*/
CV_EXPORTS_W void calcOpticalFlowSparseRLOF(InputArray prevImg, InputArray nextImg,
    InputArray prevPts, InputOutputArray nextPts,
    OutputArray status, OutputArray err,
    Ptr<RLOFOpticalFlowParameter> rlofParam = Ptr<RLOFOpticalFlowParameter>(),
    float forwardBackwardThreshold = 0);

//! Additional interface to the Dense RLOF algorithm - optflow::calcOpticalFlowDenseRLOF()
CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_DenseRLOF();

//! Additional interface to the Sparse RLOF algorithm - optflow::calcOpticalFlowSparseRLOF()
CV_EXPORTS_W Ptr<SparseOpticalFlow> createOptFlow_SparseRLOF();
//! @}

} // namespace
} // namespace
#endif
