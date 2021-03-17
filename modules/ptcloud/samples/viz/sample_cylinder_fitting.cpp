#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/ptcloud.hpp>
#include "opencv2/surface_matching.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include <cassert>
#include <numeric>
#include <cmath>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {
    Mat cloud = cv::ppf_match_3d::loadPLYSimple("./data/cylinder-big.ply", false);

    Mat ptset;
    Mat(cloud.colRange(0,3)).copyTo(ptset);
    long unsigned num_points = ptset.rows;
    ptset = ptset.reshape(3, num_points);
    ptset = ptset.t();

    Ptr<cv::ptcloud::SACModelFitting> cylinder_segmentation = cv::ptcloud::SACModelFitting::create(cloud, CYLINDER_MODEL);

    // add original cloud to window
    viz::Viz3d window("original cloud");
    viz::WCloud original_cloud(ptset);
    window.showWidget("cloud", original_cloud);

    cylinder_segmentation->set_threshold(0.5);
    cylinder_segmentation->set_iterations(80000);
    cylinder_segmentation->set_normal_distance_weight(0.5);

    vector<cv::ptcloud::SACModel> models;
    cylinder_segmentation->segment(models);
    cout << models[0].points.size();

    vector<double> model_coefficients = models.at(0).coefficients;
    cout << model_coefficients.size();

    double size = 10;
    double radius = model_coefficients[6];
    Point3d pt_on_axis(model_coefficients[0], model_coefficients[1], model_coefficients[2]);
    Point3d axis_dir(model_coefficients[3], model_coefficients[4], model_coefficients[5]);
    Point3d first_point = Point3d(Vec3d(pt_on_axis) + size * (axis_dir));
    Point3d second_point = Point3d(Vec3d(pt_on_axis) - size * (axis_dir));
    viz::WCylinder model(first_point, second_point, radius, 40, viz::Color::green());
    window.showWidget("model", model);

    viz::Viz3d fitted("fitted cloud");
    viz::WCloud cloud_widget2(models[0].points, viz::Color::red());
    fitted.showWidget("fit_cloud", cloud_widget2);
    window.showWidget("fit_cloud", cloud_widget2);
    fitted.spin();

    window.spin();
    waitKey(1);

    return 0;
}
#else
int main() {
    return CV_ERROR(-215, "this sample needs to be build with opencv's viz module");
}
#endif