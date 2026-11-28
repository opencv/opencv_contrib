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
    planar_segmentation.segment();


    const Vec3f* points = cloud.ptr<Vec3f>(0);

    // Initialise a colors array. These colors will be used (in a cyclic order) to visualise all the segmented planes.
    const vector<viz::Color> colors({viz::Color::green(), viz::Color::blue(), viz::Color::red(), viz::Color::yellow(), viz::Color::orange(),viz::Color::olive()});

    // Adds segmented planes to window
    for (unsigned model_idx = 0; model_idx < planar_segmentation.inliers.size(); model_idx++) {
        vector<unsigned> inlier_vec =  planar_segmentation.inliers.at(model_idx);
        cv::Mat fit_cloud(1, inlier_vec.size(), CV_32FC3);
        for(int j=0; j<fit_cloud.cols; ++j)
            fit_cloud.at<Vec3f>(0, j) = points[inlier_vec.at(j)];

        viz::Viz3d fitted("fit cloud " + to_string(model_idx + 1));
        fitted.showWidget("cloud", original_cloud);

        // Assign a color to this cloud from the colors array in a cyclic order.
        viz::Color cloud_color = colors[model_idx % colors.size()];
        viz::WCloud cloud_widget2(fit_cloud, cloud_color);
        fitted.showWidget("fit plane", cloud_widget2);
        window.showWidget("fit plane " + to_string(model_idx + 1), cloud_widget2);

        vector<double> model_coefficients = planar_segmentation.model_instances.at(0).ModelCoefficients;
        cv::ptcloud::SACPlaneModel SACplane (model_coefficients);

        fitted.spin();
    }

    window.spin();

    // waitKey(1);

}