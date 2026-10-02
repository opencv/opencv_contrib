#include <cmath>
#include <cstdio>
#include <iostream>
#include <kfusion/warp_field.hpp>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <kfusion/optimisation.hpp>

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    kfusion::WarpField warpField;
    std::vector<cv::Vec3f> warp_init;
    std::vector<cv::Vec3f> warp_normals;
    for(int i=0; i < KNN_NEIGHBOURS; i++)
        warp_normals.push_back(cv::Vec3f(0,0,1));

    warp_init.push_back(cv::Vec3f(1,1,1));
    warp_init.push_back(cv::Vec3f(1,1,-1));
    warp_init.push_back(cv::Vec3f(1,-1,1));
    warp_init.push_back(cv::Vec3f(1,-1,-1));
    warp_init.push_back(cv::Vec3f(-1,1,1));
    warp_init.push_back(cv::Vec3f(-1,1,-1));
    warp_init.push_back(cv::Vec3f(-1,-1,1));
    warp_init.push_back(cv::Vec3f(-1,-1,-1));

    warpField.init(warp_init, warp_normals);
    float weights[KNN_NEIGHBOURS];
    warpField.getWeightsAndUpdateKNN(cv::Vec3f(0,0,0), weights);

    std::vector<cv::Vec3f> canonical_vertices;
    canonical_vertices.push_back(cv::Vec3f(0,0,0));
    canonical_vertices.push_back(cv::Vec3f(2,2,2));

    std::vector<cv::Vec3f> canonical_normals;
    canonical_normals.push_back(cv::Vec3f(0,0,1));
    canonical_normals.push_back(cv::Vec3f(0,0,1));

    std::vector<cv::Vec3f> live_vertices;
    live_vertices.push_back(cv::Vec3f(0.01,0.01,0.01));
    live_vertices.push_back(cv::Vec3f(2.01,2.01,2.01));

    std::vector<cv::Vec3f> live_normals;
    live_normals.push_back(cv::Vec3f(0,0,1));
    live_normals.push_back(cv::Vec3f(0,0,1));
    
    warpField.energy_data(canonical_vertices, canonical_normals,live_vertices, live_normals);
    return 0;
}
