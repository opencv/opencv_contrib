// This file is part of the OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"


namespace opencv_test { namespace {

using namespace cv::ptcloud;

struct BaseModel {
    std::vector<double> coefs;
    std::vector<int> indices;
    Mat_<Point3f> cloud;
    int NPOINTS = 128;
    double THRESHOLD = 0.001;

    void generateIndices() {
        indices.resize(cloud.total());
        std::iota(indices.begin(),indices.end(),0);
    }
};

struct PlaneModel : public BaseModel {
    PlaneModel() {
        NPOINTS = 64;
        coefs = std::vector<double>{0.98, 0.13, 0.13, -10.829};
        generatePlane(cloud, coefs, NPOINTS);
        generateIndices();
    }
};

struct SphereModel : public BaseModel {
    SphereModel() {
        coefs = std::vector<double>{-8, -4, 10, 5.0};
        generateSphere(cloud, coefs, NPOINTS);
        generateIndices();
    }
};

struct CylinderModel : public BaseModel {
    Point3f dir;
    CylinderModel() {
        coefs = std::vector<double>{-10,-20,3, 0,0,1, 5.0};
        dir = Point3f(float(coefs[3]),float(coefs[4]),float(coefs[5]));
        generateCylinder(cloud, coefs, NPOINTS);
        generateIndices();
    }
};

PlaneModel plane;
SphereModel sphere;
CylinderModel cylinder;

//
// Part 1, check the internal building blocks:
//  coefficient generation, inliers, optimization
//

TEST(SAC, plane)
{
    // 1. see if we get our coefficients back
    vector<double> new_coefs;
    std::vector<int> hypothesis_inliers {1,8,11};
    bool valid_model = generateHypothesis(PLANE_MODEL, plane.cloud, hypothesis_inliers, Mat(), 50, 0, new_coefs);
    EXPECT_TRUE(valid_model);
    Point3d dir1(new_coefs[0],new_coefs[1],new_coefs[2]);
    Point3d dir2(plane.coefs[0],plane.coefs[1],plane.coefs[2]);
    double d2 = dir1.dot(dir2); // check if they are parallel
    EXPECT_GE(abs(d2), 0.95);
    EXPECT_LE(abs(abs(plane.coefs[3]) - abs(new_coefs[3])), 0.1); // length

    // 2. all plane points should be inliers
    std::vector<int> new_indices;
    SACScore score = getInliers(PLANE_MODEL, plane.coefs, plane.cloud, Mat(), plane.indices, plane.THRESHOLD, 0, new_indices, 0);
    EXPECT_EQ((int)score.first, plane.NPOINTS);
    EXPECT_LE(score.second, double(plane.THRESHOLD * plane.NPOINTS));

    // 3. no sphere points should be inliers
    score = getInliers(PLANE_MODEL, plane.coefs, sphere.cloud, Mat(), sphere.indices, plane.THRESHOLD, 0, new_indices, 0);
    EXPECT_EQ(score.first, 0.0);

    // 4. no cylinder points should be inliers
    score = getInliers(PLANE_MODEL, plane.coefs, cylinder.cloud, Mat(), cylinder.indices, plane.THRESHOLD, 0, new_indices, 0);
    EXPECT_EQ(score.first, 0.0);

    // 5. check optimization
    new_coefs = optimizeModel(PLANE_MODEL, plane.cloud, plane.indices, plane.coefs);
    Point3d dir3(new_coefs[0],new_coefs[1],new_coefs[2]);
    double d3 = dir3.dot(dir2); // check if they are parallel
    EXPECT_GE(abs(d3), 0.95);
    EXPECT_LE(abs(abs(plane.coefs[3]) - abs(new_coefs[3])), 0.1); // length
}

TEST(SAC, sphere)
{
    // 1. see if we get our coefficients back
    vector<double> new_coefs;
    std::vector<int> hypothesis_inliers {1,8,11,22};
    bool valid_model = generateHypothesis(SPHERE_MODEL, sphere.cloud, hypothesis_inliers, Mat(), 50, 0, new_coefs);
    double d = cv::norm(Mat(sphere.coefs),Mat(new_coefs));
    EXPECT_TRUE(valid_model);
    EXPECT_LE(d, 0.05);

    // 2. all sphere points should be inliers
    std::vector<int> new_indices;
    SACScore score = getInliers(SPHERE_MODEL, sphere.coefs, sphere.cloud, Mat(), sphere.indices, sphere.THRESHOLD, 0, new_indices, 0);
    EXPECT_EQ((int)score.first, sphere.NPOINTS);
    EXPECT_LE(score.second, double(sphere.THRESHOLD * sphere.NPOINTS));

    // 3. no plane points should be inliers
    score = getInliers(SPHERE_MODEL, sphere.coefs, plane.cloud, Mat(), plane.indices, sphere.THRESHOLD, 0, new_indices, 0);
    EXPECT_EQ(score.first, 0.0);

    // 4. no cylinder points should be inliers
    score = getInliers(SPHERE_MODEL, sphere.coefs, cylinder.cloud, Mat(), cylinder.indices, sphere.THRESHOLD, 0, new_indices, 0);
    EXPECT_EQ(score.first, 0.0);

    // 5. check optimization
    new_coefs = optimizeModel(SPHERE_MODEL, sphere.cloud, sphere.indices, sphere.coefs);
    double d1 = cv::norm(Mat(sphere.coefs),Mat(new_coefs));
    EXPECT_LE(d1, 0.05);
}

TEST(SAC, cylinder)
{
    // 1. see if we get our coefficients back
    Mat normals = generateNormals(cylinder.cloud);
    vector<double> new_coefs;
    std::vector<int> hypothesis_inliers {1,22};
    bool valid_model = generateHypothesis(CYLINDER_MODEL, cylinder.cloud, hypothesis_inliers, normals, 50, 0.5, new_coefs);
    EXPECT_TRUE(valid_model);
    // any point on the axis is actually valid, so we can only check the dir and the radius
    Point3d dir(new_coefs[3],new_coefs[4],new_coefs[5]);
    double d = cylinder.dir.dot(dir); // check if they are parallel
    EXPECT_GE(abs(d), 0.95);
    EXPECT_LE(abs(abs(cylinder.coefs[6]) - abs(new_coefs[6])), 2.0); // radius

    // 2. all cylinder points should be inliers
    std::vector<int> new_indices;
    SACScore score = getInliers(CYLINDER_MODEL, cylinder.coefs, cylinder.cloud, Mat(), cylinder.indices, cylinder.THRESHOLD, 0, new_indices, 0);
    EXPECT_EQ((int)score.first, cylinder.NPOINTS);
    EXPECT_LE(score.second, double(cylinder.THRESHOLD * cylinder.NPOINTS));

    // 3. no plane points should be inliers
    score = getInliers(CYLINDER_MODEL, cylinder.coefs, plane.cloud, Mat(), plane.indices, cylinder.THRESHOLD, 0, new_indices, 0);
    EXPECT_EQ(score.first, 0);

    // 4. no sphere points should be inliers
    score = getInliers(CYLINDER_MODEL, cylinder.coefs, sphere.cloud, Mat(), sphere.indices, cylinder.THRESHOLD, 0, new_indices, 0);
    EXPECT_EQ(score.first, 0.0);

    // 5. check optimization
    new_coefs = optimizeModel(CYLINDER_MODEL, cylinder.cloud, cylinder.indices, cylinder.coefs);
    Point3d dir2(new_coefs[3],new_coefs[4],new_coefs[5]);
    double d2 = cylinder.dir.dot(dir2); // check if they are parallel
    EXPECT_GE(abs(d2), 0.95);
    EXPECT_LE(abs(abs(cylinder.coefs[6]) - abs(new_coefs[6])), 2.0); // radius
}


//
// Part 2, check the segmentation:
//

Mat generateScene()
{
    Mat_<Point3f> pts;
    pts.push_back(plane.cloud);
    pts.push_back(sphere.cloud);
    pts.push_back(cylinder.cloud);
    theRNG().state=99999999;
    generateRandom(pts, vector<double>{-12,31,-5, 5}, 32); // these 2 form distinct blobs
    generateRandom(pts, vector<double>{12,-21,15, 5}, 32); // that can be retrieved via clustering
    generateRandom(pts, vector<double>{1,-2,1, 25}, 64);   // general noisy outliers
    return Mat(pts);
}

TEST(SAC, segment_ransac)
{
    Mat cloud = generateScene();
    std::vector<SACModel> models;
    Ptr<SACModelFitting> fitting = SACModelFitting::create(cloud);
    fitting->set_threshold(plane.THRESHOLD);
    fitting->segment(models);
    EXPECT_EQ(models.size(), size_t(1));
    EXPECT_EQ(models[0].indices.size(), (size_t)plane.NPOINTS);
    double d1 = cv::norm(Mat(plane.coefs), Mat(models[0].coefficients));
    EXPECT_LE(d1, 0.05);

    fitting->set_model_type(SPHERE_MODEL);
    fitting->set_threshold(sphere.THRESHOLD);
    fitting->segment(models);
    EXPECT_EQ(models.size(), size_t(2));
    EXPECT_EQ(models[1].indices.size(), (size_t)sphere.NPOINTS);
    double d2 = cv::norm(Mat(sphere.coefs), Mat(models[1].coefficients));
    EXPECT_LE(d2, 0.05);

    fitting->set_model_type(CYLINDER_MODEL);
    fitting->set_threshold(2.62);
    fitting->set_normal_distance_weight(0.5);
    fitting->set_min_inliers(int(cylinder.NPOINTS/2));
    fitting->segment(models);
    EXPECT_EQ(models.size(), size_t(3));
    EXPECT_GE(models[2].indices.size(), size_t(cylinder.NPOINTS*0.5));
    Point3d dir(models[2].coefficients[3],models[2].coefficients[4],models[2].coefficients[5]);
    double d3 = cylinder.dir.dot(dir); // check if they are parallel
    EXPECT_GE(abs(d3), 0.65);
    EXPECT_LE(abs(cylinder.coefs[6] - models[2].coefficients[6]), 0.6); // radius
}

TEST(SAC, segment_preemptive)
{
    Mat cloud = generateScene();
    std::vector<SACModel> models;
    Ptr<SACModelFitting> fitting = SACModelFitting::create(cloud);
    fitting->set_threshold(plane.THRESHOLD);
    fitting->set_preemptive_count(500);
    fitting->segment(models);
    EXPECT_EQ(models.size(), size_t(1));
    EXPECT_EQ(models[0].indices.size(), (size_t)plane.NPOINTS);

    fitting->set_model_type(SPHERE_MODEL);
    fitting->set_threshold(sphere.THRESHOLD);
    fitting->segment(models);
    EXPECT_EQ(models.size(), size_t(2));
    EXPECT_EQ(models[1].indices.size(), (size_t)sphere.NPOINTS);
    double d2 = cv::norm(Mat(sphere.coefs), Mat(models[1].coefficients));
    EXPECT_LE(d2, 0.05);

    fitting->set_model_type(CYLINDER_MODEL);
    fitting->set_threshold(3.62);
    fitting->set_normal_distance_weight(0.5);
    fitting->set_min_inliers(90);
    fitting->segment(models);
    EXPECT_EQ(models.size(), size_t(3));
    EXPECT_GE(models[2].indices.size(), size_t(cylinder.NPOINTS*0.7));
    Point3d dir(models[2].coefficients[3],models[2].coefficients[4],models[2].coefficients[5]);
    double d3 = cylinder.dir.dot(dir); // check if they are parallel
    EXPECT_GE(abs(d3), 0.65);
    EXPECT_LE(abs(abs(cylinder.coefs[6]) - abs(models[2].coefficients[6])), 0.6); // radius
}

TEST(SAC, cluster)
{
    Mat cloud = generateScene();
    cv::ppf_match_3d::writePLY(Mat(cloud), "cloud.ply");
    std::vector<SACModel> models;
    Mat new_cloud;
    cluster(cloud, 5, 20, models, new_cloud);
    EXPECT_EQ(models.size(), size_t(4)); // sphere, cylinder, 2 blobs
    // the plane needs a larger distance, so do a 2nd pass on the leftover points
    cluster(new_cloud, 8, 20, models, new_cloud);
    EXPECT_EQ(models.size(), size_t(5));
    EXPECT_LE(new_cloud.total(), size_t(64)); // the last large random blob
}

}} // namespace
