#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_VIZ
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
    Ptr<cv::ptcloud::SACModelFitting> planar_segmentation = cv::ptcloud::SACModelFitting::create(cloud);

    // add original cloud to window
    viz::Viz3d window("original cloud");
    viz::WCloud original_cloud(cloud);
    window.showWidget("cloud", original_cloud);

    planar_segmentation->set_threshold(0.001);
    planar_segmentation->set_iterations(1000);

    vector<cv::ptcloud::SACModel> models;
    planar_segmentation->segment(models);
    CV_Assert(models.size()>0);

    viz::Viz3d fitted("fitted cloud");
    viz::WCloud cloud_widget2(Mat(models[0].points), viz::Color::green());
    fitted.showWidget("fit plane", cloud_widget2);
    window.showWidget("fit plane", cloud_widget2);

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