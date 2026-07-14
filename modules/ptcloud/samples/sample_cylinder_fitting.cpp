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
    // Mat cloud = cv::ppf_match_3d::loadPLYSimple("./data/semi-cylinder-with-normals-usingOpenCV2.ply", true);
    Mat cloud = cv::ppf_match_3d::loadPLYSimple("./data/cylinder-big.ply", false);
    Mat ptset;
    Mat(cloud.colRange(0,3)).copyTo(ptset);
    long unsigned num_points = ptset.rows;
    ptset = ptset.reshape(3, num_points);
    ptset = ptset.t();

    cv::ptcloud::SACModelFitting cylinder_segmentation(CYLINDER_MODEL);
    cylinder_segmentation.setCloud(cloud, false);

    // add original cloud to window
    viz::Viz3d window("original cloud");
    viz::WCloud original_cloud(ptset);
    window.showWidget("cloud", original_cloud);
    
    cylinder_segmentation.set_threshold(0.5);
    cylinder_segmentation.set_iterations(80000);
    cylinder_segmentation.set_normal_distance_weight(0.5);
    cylinder_segmentation.fit_once();

    cout << cylinder_segmentation.inliers.size();
    vector<unsigned> inlier_vec =  cylinder_segmentation.inliers.at(0);


    vector<double> model_coefficients = cylinder_segmentation.model_instances.at(0).ModelCoefficients;
    cout << cylinder_segmentation.model_instances.at(0).ModelCoefficients.size();
    cv::ptcloud::SACCylinderModel cylinder (model_coefficients);
        cout << cylinder.pt_on_axis << endl;
        cout << cylinder.axis_dir << endl;
        cout << cylinder.radius << endl;

    viz::WCylinder model = cylinder.WindowWidget();
    window.showWidget("model", model);

    const Vec3f* points = ptset.ptr<Vec3f>(0);
    cout << endl << endl << inlier_vec.size();
    cv::Mat fit_cloud(1, inlier_vec.size(), CV_32FC3);
    for(int j=0; j<fit_cloud.cols; ++j){
        fit_cloud.at<Vec3f>(0, j) = points[(j)];
    }
    viz::Viz3d fitted("fitted cloud");
    viz::WCloud cloud_widget2(fit_cloud, viz::Color::red());
    fitted.showWidget("fit_cloud", cloud_widget2);
    window.showWidget("fit_cloud", cloud_widget2);
    fitted.spin(); 


    window.spin();
    waitKey(1);

}