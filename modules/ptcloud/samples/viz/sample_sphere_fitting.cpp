#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/ptcloud.hpp>
#include <cassert>
#include <numeric>
#include <cmath>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {
    Mat cloud = cv::viz::readCloud("./data/sphere-big.obj");
    Ptr<cv::ptcloud::SACModelFitting> sphere_segmentation = cv::ptcloud::SACModelFitting::create(cloud, 2);

    /// Adds original cloud to window
    viz::Viz3d window("original cloud");
    viz::WCloud cloud_widget1(cloud);
    window.showWidget("cloud 1", cloud_widget1);

    sphere_segmentation->set_threshold(0.001);
    sphere_segmentation->set_iterations(10000);
    sphere_segmentation->set_max_radius(10000);

    viz::Viz3d fitted("fitted cloud");

    vector<cv::ptcloud::SACModel> models;
    sphere_segmentation->segment(models);
    CV_Assert(models.size()>0);

    viz::Viz3d fitted("fitted cloud");
    viz::WCloud cloud_widget2(Mat(models[0].points), viz::Color::green());
    fitted.showWidget("fit plane", cloud_widget2);

    vector<double> model_coefficients = models.at(0).coefficients;
    cout << model_coefficients.size();

    Point3d center(model_coefficients[0],model_coefficients[1],model_coefficients[2]);
    double radius(model_coefficients[3]);
        cout << center;
        cout << radius;
    radius *= 0.75;

    viz::WSphere sphere(center, radius, 10, viz::Color::green());;
    window.showWidget("model", sphere);

    window.spin();
    fitted.spin();

    waitKey(1);

    return 0;
}
#else
int main() {
    return CV_ERROR(-215, "this sample needs to be build with opencv's viz module");
}
#endif