// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_PTCLOUD_SAC_SEGMENTATION
#define OPENCV_PTCLOUD_SAC_SEGMENTATION

#include <vector>
#include <utility>
#include "opencv2/viz.hpp"
#define PLANE_MODEL 1
#define SPHERE_MODEL 2
#define CYLINDER_MODEL 3
#define SAC_METHOD_RANSAC 1
using namespace std;

namespace cv
{
namespace ptcloud
{
    //! @addtogroup ptcloud
    //! @{
    class CV_EXPORTS_W SACModel: public Algorithm {
        public:
            std::vector<double> ModelCoefficients;

            SACModel() {

            }

            SACModel(std::vector<double> ModelCoefficients);

            virtual ~SACModel()
            {

            }

    };

    class CV_EXPORTS_W SACPlaneModel : public SACModel {
        private:
            Point3d center;
            Vec3d normal;
            Size2d size = Size2d(2.0, 2.0);
        public:
            ~ SACPlaneModel()
            {

            }

            SACPlaneModel() {

            }

            /** @brief Create a plane model based on the given coefficients and a center point.

            @param coefficients coefficients in the plane equations of type Ax + By + Cz + D = 0. Also obtained using SACModelFitting.
            @param center the center point of the plane.
            @param size the size of the plane.
            */

            SACPlaneModel(Vec4d coefficients, Point3d center, Size2d size=Size2d(2.0, 2.0));

            /** @brief Create a plane model based on the given coefficients and an arbitrary center point.

            @param coefficients coefficients in the plane equations Ax + By + Cz + D = 0.
            @param size the size of the plane.
            */
            SACPlaneModel(Vec4d coefficients, Size2d size=Size2d(2.0, 2.0));

            /** @brief Create a plane model based on the given coefficients and an arbitrary center point.

            @param coefficients coefficients in the plane equations Ax + By + Cz + D = 0.
            @param size the size of the plane.
            */
            SACPlaneModel(std::vector<double> coefficients, Size2d size=Size2d(2.0, 2.0));

            viz::WPlane WindowWidget ();

            std::pair<double, double> getInliers(Mat cloud, std::vector<unsigned> indices, const double threshold, std::vector<unsigned>& inliers);
    };


    class CV_EXPORTS_W SACSphereModel : public SACModel {
        public:
            Point3d center;
            double radius;

            ~ SACSphereModel()
            {

            }

            SACSphereModel() {

            }

            /** @brief Create a spherical model based on the given center and radius.

            @param center the center point of the sphere
            @param radius the radius of the sphere.
            */

            SACSphereModel(Point3d center, double radius);

            /** @brief Create a spherical model based on the parametric coefficients.

            This is very helpful for creating a model for the fit models using SACModelFitting class.

            @param Coefficients parametric coefficients for the Sphere model
            */

            SACSphereModel(const std::vector<double> Coefficients);

            SACSphereModel(Vec4d coefficients);

            viz::WSphere WindowWidget ();

            double euclideanDist(Point3d& p, Point3d& q);

            std::pair<double, double> getInliers(Mat cloud, std::vector<unsigned> indices, const double threshold, std::vector<unsigned>& inliers);
    };

    class CV_EXPORTS_W SACCylinderModel : public SACModel {
        public:
            Point3d pt_on_axis;
            Vec3d axis_dir;
            double radius;
            double size = 20;

            ~ SACCylinderModel()
            {

            }

            SACCylinderModel() {

            }

            // /** @brief Create a spherical model based on the given center and radius.

            // @param center the center point of the sphere
            // @param radius the radius of the sphere.
            // */

            // SACCylinderModel(const std::vector<double> Coefficients);

            /** @brief Create a spherical model based on the parametric coefficients.

            This is very helpful for creating a model for the fit models using SACModelFitting class.

            @param Coefficients parametric coefficients for the Sphere model
            */

            SACCylinderModel(const std::vector<double> Coefficients);

            viz::WCylinder WindowWidget ();

            std::pair<double, double> getInliers(Mat cloud, Mat normals, std::vector<unsigned> indices, const double threshold, std::vector<unsigned>& inliers, double normal_distance_weight_ = 0);
    };

    class CV_EXPORTS_W SACModelFitting {
        private:
            Mat cloud;
            Mat normals;
            bool normals_available = false;
            int model_type;
            int method_type;
            double threshold;
            long unsigned max_iters;
            double normal_distance_weight_ = 0;

        public:
            // cv::Mat remainingCloud; // will be used while segmentation

            // Inlier indices only, not the points themselves. It would work like a mask output for segmentation in 2d.
            vector<vector<unsigned>> inliers;
            vector<SACModel> model_instances;

            /** @brief Initializes SACModelFitting class.

            Threshold and Iterations may also be set separately.

            @param cloud input Point Cloud.
            @param model_type type of model fitting to attempt - values can be either PLANE_MODEL, SPHERE_MODEL, or CYLINDER_MODEL.
            @param method_type which method to use - currently, only RANSAC is supported (use value SAC_METHOD_RANSAC).
            @param threshold set the threshold while choosing inliers.
            @param max_iters number of iterations for Sampling.
            */

            SACModelFitting (Mat cloud, int model_type = PLANE_MODEL, int method_type = SAC_METHOD_RANSAC, double threshold = 20,int max_iters = 1000);
                // :cloud(cloud), model_type(model_type), method_type(method_type), threshold(threshold), max_iters(max_iters) {}

            /** @brief Initializes SACModelFitting class.

            Threshold and Iterations may also be set separately.

            @param model_type type of model fitting to attempt - values can be either PLANE_MODEL, SPHERE_MODEL, or CYLINDER_MODEL.
            @param method_type which method to use - currently, only RANSAC is supported (use value SAC_METHOD_RANSAC).
            @param threshold set the threshold while choosing inliers.
            @param max_iters number of iterations for Sampling.
            */
            SACModelFitting (int model_type = PLANE_MODEL, int method_type = SAC_METHOD_RANSAC, double threshold = 20,int max_iters = 1000);
                // :model_type(model_type), method_type(method_type), threshold(threshold), max_iters(max_iters) {}

            /** @brief Fit one model, this function would get the best fitting model on the given set of points.

            This stores the model in the public class member model_instances, and the mask for inliers in inliers.
            */
            bool fit_once(vector<int> remaining_indices = {});

            /** @brief Fit multiple models of the same type, this function would get the best fitting models on the given set of points.

            This stores the models in the public class member model_instances, and the corresponding masks for inliers in inliers.

            Returns False if no valid model could be fit.

            @param remaining_cloud_threshold set the threshold for the remaining cloud (from 0 to 1) until which the segmentation should continue.
            */
            void segment(float remaining_cloud_threshold = 0.3);

            void setCloud(Mat cloud);

            void setCloud(Mat cloud, bool with_normals=false);

            /** @brief Set the threshold for the fitting.
            The threshold is usually the distance from the boundary of model, but may vary from model to model.

            This may be helpful when multiple fitting operations are to be performed.
            @param threshold the threshold to set.
            */
            void set_threshold (double threshold);

            /** @brief Set the number of iterations for the fitting.

            This may be helpful when multiple fitting operations are to be performed.
            @param iterations the threshold to set.
            */
            void set_iterations (long unsigned iterations);

            /** @brief Set the weight given to normal alignment before comparing overall error with threshold.
             *  By default it is set to 0.
            @param weight the desired normal alignment weight (between 0 to 1).
            */
            void set_normal_distance_weight(double weight);
    };

    bool getSphereFromPoints(const Vec3f*&, const vector<unsigned int>&, Point3d&, double&);

    Vec4d getPlaneFromPoints(const Vec3f*&, const std::vector<unsigned int>&, cv::Point3d&);

    bool getCylinderFromPoints(Mat cloud, Mat normal,
                                    const std::vector<unsigned> &inliers, vector<double> & coefficients) ;

    double euclideanDist(Point3d& p, Point3d& q);

} // ptcloud
}   // cv
#endif