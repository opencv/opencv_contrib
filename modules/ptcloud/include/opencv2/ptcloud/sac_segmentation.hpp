// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_PTCLOUD_SAC_SEGMENTATION
#define OPENCV_PTCLOUD_SAC_SEGMENTATION

#include <vector>
#include <opencv2/core.hpp>


namespace cv
{
namespace ptcloud
{
//! @addtogroup ptcloud
//! @{

enum SacModelType {
    PLANE_MODEL = 1,
    SPHERE_MODEL = 2,
    CYLINDER_MODEL = 3,
    BLOB_MODEL = 4
};

//
//! follows ScoreMethod from calib3d
//
enum SacMethodType {
    SAC_METHOD_RANSAC = 0,     //!< maximize inlier count
    SAC_METHOD_MSAC   = 1      //!< minimize inlier distance
};

typedef std::pair<double, double> SACScore;

/** @brief structure to hold the segmentation results
*/
class CV_EXPORTS_W SACModel {
public:
    CV_PROP std::vector<double> coefficients; //!< the "model"
    CV_PROP std::vector<int> indices; //!< into the original cloud
    CV_PROP std::vector<Point3f> points; //!< (output) copies (original cloud might no more exist)
    CV_PROP int type; //!< see SacModelType
    CV_PROP SACScore score; //!< first:inlier_count(ransac), second:accumulated distance(msac)

    CV_WRAP SACModel(int typ=PLANE_MODEL);
};


/**  @brief The SACModelFitting class to segment geometric primitives like planes,spheres,cylinders from 3d point clouds.

2 alternative ransac strategies are implemented here:

- ransac loop:
  + generate random minimal hypothesis
  + find inliers for the model,
    - bail out if at 1/4 of the data it does not have more than 1/8 inliers of the current best model
    - if sprt is enabled, bail out if the accumulated probability of a bad model crosses a threshold
  + if this model is the current best one
    - best model = current
    - update stopping criterion (using sprt if enabled)
    - apply local optimization (generate a non-minimal model from the inliers)
      + if it improved, take that instead

- preemptive ransac loop:
  + generate M minimal random models in advance
  + slice the data into blocks(of size B), for each block:
    - evaluate all M models in parallel
    - sort descending (by inliers or accumulated distance)
    - prune model list, M' = M * (2 ^ -(i/B))
    - stop if there is only one model left, or the last data block reached
  + polish/optimize the last remaining model
*/
class CV_EXPORTS_W SACModelFitting {
public:
    virtual ~SACModelFitting() {}

    /** @brief set a new point cloud to segment

    @param cloud either a 3channel or 1 channel / 3cols or rows Mat
    @param with_normals if enabled, the cloud should have either 6 rows or 6 cols, and a single channel
    */
    CV_WRAP virtual void set_cloud(InputArray cloud, bool with_normals=false) = 0;

    /** @brief Set the type of model to be fitted

    @param model_type see SacModelType enum.
    */
    CV_WRAP virtual void set_model_type(int model_type) = 0;

    /** @brief Set the type of ransac method

    @param method_type see SacMethodType enum.
    */
    CV_WRAP virtual void set_method_type(int method_type) = 0;

    /** @brief Use Wald's Sequential Probability Ratio Test with ransac fitting

    This will result in less iterations and less evaluated data points, and thus be
    much faster, but it might miss some inliers
    (not used in the preemptive ransac)
    @param sprt true or false.
    */
    CV_WRAP virtual void set_use_sprt(bool sprt) = 0;

    /** @brief Set the threshold for the fitting.
    The threshold is usually the distance from the boundary of model, but may vary from model to model.

    This may be helpful when multiple fitting operations are to be performed.
    @param threshold the threshold to set.
    */
    CV_WRAP virtual void set_threshold(double threshold) = 0;

    /** @brief Set the number of iterations for the (non preemptive) ransac fitting.

    This may be helpful when multiple fitting operations are to be performed.
    @param iterations the threshold to set.
    */
    CV_WRAP virtual void set_iterations(int iterations) = 0;

    /** @brief Set the weight given to normal alignment before comparing overall error with threshold.

    By default it is set to 0.
    @param weight the desired normal alignment weight (between 0 to 1).
    */
    CV_WRAP virtual void set_normal_distance_weight(double weight) = 0;

    /** @brief Set the maximal radius for the sphere hypothesis generation.

    A radius larger than this is considered degenerate.
    @param max_radius the maximum valid sphere radius.
    */
    CV_WRAP virtual void set_max_sphere_radius(double max_radius) = 0;

    /** @brief Set the maximal radius for the (optional) napsac hypothesis sampling.

    Assume good hypothesis inliers are locally close to each other
    (enforce spatial proximity for hypothesis inliers).

    @param max_radius the maximum valid sphere radius for napsac sampling. set it to 0 to disable this strategy, and sample uniformly instead.
    */
    CV_WRAP virtual void set_max_napsac_radius(double max_radius) = 0;

    /**
    @param preemptive the number of models generated with preemptive ransac.
        set to 0 to disable preemptive hypothesis generation and do plain ransac instead
    */
    CV_WRAP virtual void set_preemptive_count(int preemptive) = 0;

    /**
    @param min_inliers reject a model if it has less inliers than this.
    */
    CV_WRAP virtual void set_min_inliers(int min_inliers) = 0;

    /** @brief Segment multiple models of the same type, this function would get the best fitting models on the given set of points.

    This stores the models in the model_instances, and optionally, the remaining cloud points.

    @param model_instances a vector of SACModels.
    @param new_cloud (optional) the remaining non-segmented cloud points.
    */
    CV_WRAP virtual void segment(CV_OUT std::vector<SACModel> &model_instances, OutputArray new_cloud=noArray()) = 0;

    /** @brief Initializes a SACModelFitting instance.

    @param cloud input Point Cloud.
    @param model_type type of model fitting to attempt - values can be either PLANE_MODEL, SPHERE_MODEL, or CYLINDER_MODEL.
    @param method_type which method to use - (use value SAC_METHOD_RANSAC or SAC_METHOD_MSAC).
    @param threshold set the threshold while choosing inliers.
    @param max_iters number of iterations for Sampling.
    */
    CV_WRAP static Ptr<SACModelFitting> create(InputArray cloud, int model_type = PLANE_MODEL, int method_type = SAC_METHOD_RANSAC, double threshold = 20, int max_iters = 1000);
};


/** @brief Cluster (remaining) points into blobs

    This is using cv::partition() internally to seperate blobs.

    @param cloud the input point cloud
    @param distance max distance to the next inlier point in the blob
    @param min_inliers reject blobs with less inliers than this
    @param models a vector of SACModels to hold the resulting blobs
    @param new_cloud optionally return non segmented points
*/
CV_EXPORTS_W void cluster(InputArray cloud, double distance, int min_inliers, CV_OUT std::vector<SACModel> &models, OutputArray new_cloud=noArray());

} // ptcloud
} // cv


#endif