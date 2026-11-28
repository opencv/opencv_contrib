#include <opencv2/viz.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/ptcloud.hpp>
#include <cassert>
#include <numeric>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

int main() {
     Mat cloud = cv::viz::readCloud("./data/CobbleStones.obj");
    cv::ptcloud::SACModelFitting planar_segmentation(cloud);

    // add original cloud to window
    viz::Viz3d window("original cloud");
    viz::WCloud original_cloud(cloud);
    window.showWidget("cloud", original_cloud);


    planar_segmentation.set_threshold(0.001);
    planar_segmentation.set_iterations(1000);
    planar_segmentation.fit_once();

    // Adds segmented (int this case fit, since only once) plane to window

    const Vec3f* points = cloud.ptr<Vec3f>(0);
    vector<unsigned> inlier_vec =  planar_segmentation.inliers.at(0);
    cv::Mat fit_cloud(1, inlier_vec.size(), CV_32FC3);
    for(int j=0; j<fit_cloud.cols; ++j)
        fit_cloud.at<Vec3f>(0, j) = points[inlier_vec.at(j)];

    viz::Viz3d fitted("fitted cloud");
    viz::WCloud cloud_widget2(fit_cloud, viz::Color::green());
    fitted.showWidget("fit plane", cloud_widget2);

    window.showWidget("fit plane", cloud_widget2);

    vector<double> model_coefficients = planar_segmentation.model_instances.at(0).ModelCoefficients;
    cv::ptcloud::SACPlaneModel SACplane (model_coefficients);

    window.spin();
    fitted.spin();
    waitKey(1);

}