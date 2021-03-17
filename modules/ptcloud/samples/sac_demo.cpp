#include "opencv2/ptcloud/sac_segmentation.hpp"
#include "opencv2/ptcloud/sac_utils.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"

#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::ptcloud;


int main(int argc,char **argv) {
    CommandLineParser parser (argc,argv,
        "{help         h |        | print this help}"
        "{model_type   m | 1      | 1:plane 2:sphere 3:cylinder 4:cluster 0:all}"
        "{use_sprt     u | false  | use sprt evaluation/termination with non-preemptive ransac}"
        "{ply          P |        | load a .ply file}"
        "{threshold    t | 0.0015 | rejection threshold .001 for planes, 0.1 for spheres, 2.5 for cylinders}"
        "{iters        i | 10000  | ransac iterations}"
        "{cloud        c | 15     | or'ed synthetic model types to generate 0:none 1:planes 2:sphere 4:cylinder 8:random 16:noise distortion 32:from ply}"
        "{min_inliers  M | 60     | rejection inlier count}"
        "{min_distance d | 6      | distance for clustering (partition)}"
        "{sac_method   s | 0      | SacMethodType (0:RANSAC or 1:MSAC)}"
        "{preemptive   p | 0      | number of hypotheses for preemptive evaluation. set to 0 to disable}"
        "{napsac       n | 0      | radius for napsac sampling. set to 0 to disable}"
        "{max_sphere   S | 50     | (sphere only) reject larger spheres}"
        "{normal_weight  w | 0.5  | (cylinder only) interpolate between point and normal(dot) distance. setting it to 0 will not use (or generate) normals}"
        "");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    int typ = parser.get<int>("model_type");
    int iters = parser.get<int>("iters");
    int elems = parser.get<int>("cloud");
    int method = parser.get<int>("sac_method");
    int napsac = parser.get<int>("napsac");
    int min_inliers = parser.get<int>("min_inliers");
    int min_distance = parser.get<int>("min_distance");
    int preemptive = parser.get<int>("preemptive");
    int max_sphere = parser.get<int>("max_sphere");
    float thresh = parser.get<float>("threshold");
    float normal_weight = parser.get<float>("normal_weight");
    bool sprt = parser.get<bool>("use_sprt");
    string ply = parser.get<string>("ply");

    Mat_<Point3f> pts;
    if (! ply.empty()) {
        pts = cv::ppf_match_3d::loadPLYSimple(ply.c_str(), false);
    }
    if (elems & 1) {
        generatePlane(pts, vector<double>{0.1,.1,.8, .76}, 64);
        generatePlane(pts, vector<double>{-.2,-.7,.3, .16}, 64);
    }
    if (elems & 2) {
        generateSphere(pts, vector<double>{12,15,-12, 5}, 128);
        generateSphere(pts, vector<double>{-12,15,-12, 5}, 128);
    }
    if (elems & 4) {
        generateCylinder(pts, vector<double>{-12,-15,-12, 0,0,1, 5}, 256);
        generateCylinder(pts, vector<double>{20,5,12, 0,1,0, 5}, 256);
    }
    if (elems & 8) {
        generateRandom(pts, vector<double>{-12,31,-5, 5}, 32);
        generateRandom(pts, vector<double>{12,-21,15, 5}, 32);
        generateRandom(pts, vector<double>{1,-2,1, 25}, 64);
    }
    if (elems & 16) {
        Mat fuzz(pts.size(), pts.type());
        float F = 0.001f;
        randu(fuzz,Scalar(-F,-F,-F), Scalar(F,F,F));
        pts += fuzz;
    }
    cout << pts.size() << " points." << endl;
    cv::ppf_match_3d::writePLY(pts, "cloud.ply");

    auto segment = [napsac,normal_weight,max_sphere,preemptive,sprt](const Mat &cloud, std::vector<SACModel> &models, int model_type, float threshold, int max_iters, int min_inlier, int method_type) -> Mat {
        Ptr<SACModelFitting> fitting = SACModelFitting::create(cloud, model_type, method_type, threshold, max_iters);
        fitting->set_normal_distance_weight(normal_weight);
        fitting->set_max_napsac_radius(napsac);
        fitting->set_max_sphere_radius(max_sphere);
        fitting->set_preemptive_count(preemptive);
        fitting->set_min_inliers(min_inlier);
        fitting->set_use_sprt(sprt);

        Mat new_cloud;
        fitting->segment(models, new_cloud);

        return new_cloud;
    };

    std::vector<SACModel> models;
    if (typ==4) { // cluster only
        cv::ptcloud::cluster(pts, min_distance, min_inliers, models, pts);
    } else
    if (typ==0) { // end to end
        pts = segment(pts, models, 1, thresh, iters, min_inliers, method);
        cout << pts.total() << " points left." << endl;
        pts = segment(pts, models, 2, 0.145f, iters, min_inliers, method);
        cout << pts.total() << " points left." << endl;
        pts = segment(pts, models, 3, 5.0f, iters, min_inliers, method);
        cout << pts.total() << " points left." << endl;
        cv::ptcloud::cluster(pts, 7, 20, models, pts);
    } else // single model type
        pts = segment(pts, models, typ, thresh, iters, min_inliers, method);

    string names[] = {"", "plane","sphere","cylinder","blob"};
    for (size_t i=0; i<models.size(); i++) {
        SACModel &model = models.at(i);
        cout << model.type << " " << model.points.size() << " " << model.score.second << "\t";
        cout << Mat(model.coefficients).t() << endl;
        cv::ppf_match_3d::writePLY(Mat(model.points), format("cloud_%s_%d.ply",names[model.type].c_str(), int(i+1)).c_str());
    }

    cout << pts.total() << " points left." << endl;
    cv::ppf_match_3d::writePLY(pts, "cloud_left.ply");
    return 0;
}
