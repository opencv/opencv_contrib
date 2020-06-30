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
            // vector<Point3f> inliers;
            SACModel();
            SACModel(std::vector<double> ModelCoefficients);

            virtual ~SACModel()
            {

            }
            // virtual viz::Widget3D WindowWidget () = 0;

            virtual void getModelFromPoints(Mat inliers);
    };

    class CV_EXPORTS_W SACPlaneModel : public SACModel {
        private:
            Point3d center;
            Vec3d normal;
            Size2d size = Size2d(2.0, 2.0);
        public:
            SACPlaneModel();

            SACPlaneModel(const std::vector<double> Coefficients);

            SACPlaneModel(Vec4d coefficients, Point3d center, Size2d size=Size2d(2.0, 2.0));

            SACPlaneModel(Vec4d coefficients, Size2d size=Size2d(2.0, 2.0));

            SACPlaneModel(std::vector<double> coefficients, Size2d size=Size2d(2.0, 2.0));

            viz::WPlane WindowWidget ();

            std::pair<double, double> getInliers(Mat cloud, std::vector<unsigned> indices, const double threshold, std::vector<unsigned>& inliers);
    };


    class CV_EXPORTS_W SACSphereModel : public SACModel {
        public:
            Point3d center;
            double radius;

            SACSphereModel() {
            }

            SACSphereModel(Point3d center, double radius);

            SACSphereModel(const std::vector<double> Coefficients);

            SACSphereModel(Vec4d coefficients);
            // SACSphereModel(std::vector<double> coefficients);
            viz::WSphere WindowWidget ();

            double euclideanDist(Point3d& p, Point3d& q);

            std::pair<double, double> getInliers(Mat cloud, std::vector<unsigned> indices, const double threshold, std::vector<unsigned>& inliers);
    };

    class CV_EXPORTS_W SACModelFitting {
        private:
            Mat cloud;
            int model_type;
            int method_type;
            double threshold;
            long unsigned max_iters;

        public:
            cv::Mat remainingCloud;
            vector<vector<unsigned>> inliers;
            vector<SACModel> model_instances;

            // viz::Viz3d window;
            SACModelFitting (Mat cloud, int model_type = PLANE_MODEL, int method_type = SAC_METHOD_RANSAC, double threshold = 20,int max_iters = 1000);
                // :cloud(cloud), model_type(model_type), method_type(method_type), threshold(threshold), max_iters(max_iters) {}

            SACModelFitting (int model_type = PLANE_MODEL, int method_type = SAC_METHOD_RANSAC, double threshold = 20,int max_iters = 1000);
                // :model_type(model_type), method_type(method_type), threshold(threshold), max_iters(max_iters) {}

            // Get one model (plane), this function would call RANSAC on the given set of points and get the biggest model (plane).
            void fit_once();
            };

    bool getSphereFromPoints(const Vec3f*&, const vector<unsigned int>&, Point3d&, double&);

    Vec4d getPlaneFromPoints(const Vec3f*&, const std::vector<unsigned int>&, cv::Point3d&);

    double euclideanDist(Point3d& p, Point3d& q);

} // ptcloud
}   // cv
#endif