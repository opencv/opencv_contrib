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
    cv::ptcloud::SACModelFitting sphere_segmentation(cloud, 2);

    /// Adds original cloud to window
    viz::Viz3d window("original cloud");
    viz::WCloud cloud_widget1(cloud);
    window.showWidget("cloud 1", cloud_widget1);

    sphere_segmentation.set_threshold(0.001);
    sphere_segmentation.set_iterations(10000);

    viz::Viz3d fitted("fitted cloud");

    sphere_segmentation.fit_once();
    vector<double> model_coefficients = sphere_segmentation.model_instances.at(0).ModelCoefficients;
    cout << sphere_segmentation.model_instances.at(0).ModelCoefficients.size();
    cv::ptcloud::SACSphereModel sphere (model_coefficients);
        cout << sphere.center;
        cout << sphere.radius;
    sphere.radius *= 0.75;

    viz::WSphere model = sphere.WindowWidget();
    window.showWidget("model", model);

    window.spin();
    waitKey(1);

}