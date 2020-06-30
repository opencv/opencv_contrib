// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <cassert>
#include <numeric>


using namespace std;
using namespace cv;

namespace cv
{
namespace ptcloud
{

        bool getSphereFromPoints(const Vec3f* &points, const std::vector<unsigned> &inliers, Point3d& center, double& radius) {
        // assert that size of points is 3.
        Mat temp(5,5,CV_32FC1);
        // Vec4f temp;
        for(int i = 0; i < 4; i++)
        {
            unsigned point_idx = inliers[i];
            float* tempi = temp.ptr<float>(i);
            for(int j = 0; j < 3; j++) {
                tempi[j] = (float) points[point_idx][j];
            }
            tempi[3] = 1;
        }
        double m11 = determinant(temp);
        if (m11 == 0) return false; // no sphere exists

        for(int i = 0; i < 4; i++)
        {
            unsigned point_idx = inliers[i];
            float* tempi = temp.ptr<float>(i);

            tempi[0] = (float) points[point_idx][0] * (float) points[point_idx][0]
                        + (float) points[point_idx][1] * (float) points[point_idx][1]
                        + (float) points[point_idx][2] * (float) points[point_idx][2];

        }
        double m12 = determinant(temp);

        for(int i = 0; i < 4; i++)
        {
            unsigned point_idx = inliers[i];
            float* tempi = temp.ptr<float>(i);

            tempi[1] = tempi[0];
            tempi[0] = (float) points[point_idx][0];

        }
        double m13 = determinant(temp);

        for(int i = 0; i < 4; i++)
        {
            unsigned point_idx = inliers[i];
            float* tempi = temp.ptr<float>(i);

            tempi[2] = tempi[1];
            tempi[1] = (float) points[point_idx][1];

        }
        double m14 = determinant(temp);

        for(int i = 0; i < 4; i++)
        {
            unsigned point_idx = inliers[i];
            float* tempi = temp.ptr<float>(i);

            tempi[0] = tempi[2];
            tempi[1] = (float) points[point_idx][0];
            tempi[2] = (float) points[point_idx][1];
            tempi[3] = (float) points[point_idx][2];
        }
        double m15 = determinant(temp);

        center.x = 0.5 * m12 / m11;
        center.y = 0.5 * m13 / m11;
        center.z = 0.5 * m14 / m11;
        // Radius
        radius = std::sqrt (center.x * center.x +
                                            center.y * center.y +
                                            center.z * center.z - m15 / m11);

    return (true);

    }

    Vec4d getPlaneFromPoints(const Vec3f* &points,
                                    const std::vector<unsigned> &inliers, Point3d& center) {
        // REF: https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        Vec3f centroid(0, 0, 0);
        for (unsigned idx : inliers) {
            centroid += points[idx];
        }
        centroid /= double(inliers.size());

        double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

        for (size_t idx : inliers) {
            Vec3f r = points[idx] - centroid;
            xx += r(0) * r(0);
            xy += r(0) * r(1);
            xz += r(0) * r(2);
            yy += r(1) * r(1);
            yz += r(1) * r(2);
            zz += r(2) * r(2);
        }

        double det_x = yy * zz - yz * yz;
        double det_y = xx * zz - xz * xz;
        double det_z = xx * yy - xy * xy;

        Vec3d abc;
        if (det_x > det_y && det_x > det_z) {
            abc = Vec3d(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
        } else if (det_y > det_z) {
            abc = Vec3d(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
        } else {
            abc = Vec3d(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
        }


        double magnitude_abc = sqrt(abc[0]*abc[0] + abc[1]* abc[1] + abc[2] * abc[2]);

        // Return invalid plane if the points don't span a plane.
        if (magnitude_abc == 0) {
            return Vec4d (0, 0, 0, 0);
        }
        abc /= magnitude_abc;
        double d = -abc.dot(centroid);

        Vec4d coefficients (abc[0], abc[1], abc[2], d);
        center = Point3d (centroid);
        return coefficients;
    }

    SACPlaneModel::SACPlaneModel(Vec4d coefficients, Point3d set_center, Size2d set_size) {
        this->ModelCoefficients.reserve(4);
        for (int i = 0; i < 4; i++) {
            this->ModelCoefficients.push_back(coefficients[i]);
        }
        this->size = set_size;

        this-> normal = Vec3d(coefficients[0], coefficients[1], coefficients[2]);
        this -> center = set_center;
        // Assign normal vector
        for (unsigned i = 0; i < 3; i++) normal[i] = coefficients[i];
    }

    SACPlaneModel::SACPlaneModel(Vec4d coefficients, Size2d set_size) {
        this->ModelCoefficients.reserve(4);
        for (int i = 0; i < 4; i++) {
            this->ModelCoefficients.push_back(coefficients[i]);
        }
        this->size = set_size;

        this-> normal = Vec3d(coefficients[0], coefficients[1], coefficients[2]);
        this -> center = Point3d(0, 0, - coefficients[3] / coefficients[2]);
        // Assign normal vector
        for (unsigned i = 0; i < 3; i++) normal[i] = coefficients[i];

        if (coefficients[2] != 0) {
            center.x = 0;
            center.y = 0;
            center.z = -coefficients[3] / coefficients[2];
        } else if (coefficients[1] != 0) {
            center.x = 0;
            center.y = -coefficients[3] / coefficients[1];
            center.z = 0;
        } else if (coefficients[0] != 0) {
            center.x = -coefficients[3] / coefficients[0];
            center.y = 0;
            center.z = 0;
        }
    }

    SACPlaneModel::SACPlaneModel(vector<double> coefficients, Size2d set_size) {
        assert(coefficients.size() == 4);
        this->ModelCoefficients = coefficients;
        this->size = set_size;

        // Assign normal vector
        for (unsigned i = 0; i < 3; i++) normal[i] = coefficients[i];

        // Since the plane viz widget would be finite, it must have a center, we give it an arbitrary center
        // from the model coefficients.
        if (coefficients[2] != 0) {
            center.x = 0;
            center.y = 0;
            center.z = -coefficients[3] / coefficients[2];
        } else if (coefficients[1] != 0) {
            center.x = 0;
            center.y = -coefficients[3] / coefficients[1];
            center.z = 0;
        } else if (coefficients[0] != 0) {
            center.x = -coefficients[3] / coefficients[0];
            center.y = 0;
            center.z = 0;
        }
    }
    // void SACPlaneModel::addToWindow(viz::Viz3d & window) {
    //     viz::WPlane plane(this->center, this->normal, Vec3d(1, 0, 0), this->size, viz::Color::green());
    //     window.showWidget("planeD", plane);
    // }
    viz::WPlane SACPlaneModel::WindowWidget () {
        return viz::WPlane (this->center, this->normal, Vec3d(1, 0, 0), this->size, viz::Color::green());
    }

    pair<double, double> SACPlaneModel::getInliers(Mat cloud, vector<unsigned> indices, const double threshold, vector<unsigned>& inliers) {
        pair<double, double> result;
        inliers.clear();
        const Vec3f* points = cloud.ptr<Vec3f>(0);
        const unsigned num_points = indices.size();

        double magnitude_abc = sqrt(ModelCoefficients[0]*ModelCoefficients[0] + ModelCoefficients[1]* ModelCoefficients[1] + ModelCoefficients[2] * ModelCoefficients[2]);

        if (magnitude_abc == 0) {
            //something is wrong
        }


        Vec4d NormalisedCoefficients (ModelCoefficients[0]/magnitude_abc, ModelCoefficients[1]/magnitude_abc, ModelCoefficients[2]/magnitude_abc, ModelCoefficients[3]/magnitude_abc);
        double fitness = 0;
        double rmse = 0;
        for (unsigned i = 0; i < num_points; i++) {
            unsigned ind = indices[i];
            Vec4d point4d (points[ind][0], points[ind][1], points[ind][2], 1);
            double distanceFromPlane = point4d.dot(NormalisedCoefficients);
            if (abs(distanceFromPlane) > threshold) continue;
            inliers.emplace_back(ind);

            fitness+=1;
            rmse += distanceFromPlane;
        }

        unsigned num_inliers = fitness;
        if (num_inliers == 0) {
            result.first = 0;
            result.second = 0;
        } else {
            rmse /= num_inliers;
            fitness /= num_points;

            result.first = fitness;
            result.second = rmse;
        }

        return result;
    }
    SACSphereModel::SACSphereModel(Point3d set_center, double set_radius) {
        this -> center = set_center;
        this -> radius = set_radius;

        this->ModelCoefficients.reserve(4);
        this -> ModelCoefficients.push_back(center.x);
        this -> ModelCoefficients.push_back(center.y);
        this -> ModelCoefficients.push_back(center.z);

        this -> ModelCoefficients.push_back(radius);
    }


    SACSphereModel::SACSphereModel(Vec4d coefficients) {
        this->ModelCoefficients.reserve(4);
        for (int i = 0; i < 4; i++) {
            this->ModelCoefficients.push_back(coefficients[i]);
        }
        this -> center = Point3d(coefficients[0], coefficients[1], coefficients[2]);
        this -> radius = coefficients[3];
    }

    SACSphereModel::SACSphereModel(vector<double> coefficients) {
        assert(coefficients.size() == 4);
        for (int i = 0; i < 4; i++) {
            this->ModelCoefficients.push_back(coefficients[i]);
        }
        this -> center = Point3d(coefficients[0], coefficients[1], coefficients[2]);
        this -> radius = coefficients[3];
    }


    viz::WSphere SACSphereModel::WindowWidget () {
        return viz::WSphere(this->center, this->radius, 10, viz::Color::green());;
    }

    double SACSphereModel::euclideanDist(Point3d& p, Point3d& q) {
        Point3d diff = p - q;
        return cv::sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
    }

    pair<double, double> SACSphereModel::getInliers(Mat cloud, vector<unsigned> indices, const double threshold, vector<unsigned>& inliers) {
        pair<double, double> result;
        inliers.clear();
        const Vec3f* points = cloud.ptr<Vec3f>(0);
        const unsigned num_points = indices.size();

        double fitness = 0;
        double rmse = 0;
        if(!isnan(radius)) { // radius may come out to be nan if selected points form a plane
            for (unsigned i = 0; i < num_points; i++) {
                unsigned ind = indices[i];
                Point3d pt (points[ind][0], points[ind][1], points[ind][2]);
                double distanceFromCenter = euclideanDist(pt, center);

                double distanceFromSurface = distanceFromCenter - radius;
                if (distanceFromSurface > threshold) continue;
                inliers.emplace_back(ind);

                fitness+=1;
                rmse += max(0., distanceFromSurface);
            }
        }


        unsigned num_inliers = fitness;
        if (num_inliers == 0) {
            result.first = 0;
            result.second = 0;
        } else {
            rmse /= num_inliers;
            fitness /= num_points;
            result.first = fitness;
            result.second = rmse;
        }

        return result;
    }

    SACModelFitting::SACModelFitting (Mat set_cloud, int set_model_type, int set_method_type, double set_threshold, int set_max_iters)
            :cloud(set_cloud), model_type(set_model_type), method_type(set_method_type), threshold(set_threshold), max_iters(set_max_iters) {}

    SACModelFitting::SACModelFitting (int set_model_type, int set_method_type, double set_threshold, int set_max_iters)
        :model_type(set_model_type), method_type(set_method_type), threshold(set_threshold), max_iters(set_max_iters) {}

    void SACModelFitting::fit_once() {

        if (method_type != SAC_METHOD_RANSAC) return; // Only RANSAC supported ATM, need to integrate with Maksym's framework.

        // creates an array of indices for the points in the point cloud which will be appended as masks to denote inliers and outliers.
        const Vec3f* points = cloud.ptr<Vec3f>(0);
        unsigned num_points = cloud.cols;
        std::vector<unsigned> indices(num_points);
        std::iota(std::begin(indices), std::end(indices), 0);


        vector<unsigned> inliers_indices;

        // Initialize the best plane model.
        SACModel bestModel;
        pair<double, double> bestResult(0, 0);

        if (model_type == PLANE_MODEL) {
            const unsigned num_rnd_model_points = 3;
            RNG rng((uint64)-1);
            for (unsigned i = 0; i < max_iters; ++i) {
                vector<unsigned> current_model_inliers;
                SACModel model;

                for (unsigned j = 0; j < num_rnd_model_points; ++j) {
                    std::swap(indices[i], indices[rng.uniform(0, num_points)]);
                }

                for (unsigned j = 0; j < num_rnd_model_points; j++) {
                    unsigned idx = indices[i];
                    current_model_inliers.emplace_back(idx);
                }

                Point3d center;
                Vec4d coefficients = getPlaneFromPoints(points, current_model_inliers, center);
                SACPlaneModel planeModel (coefficients, center);
                pair<double, double> result = planeModel.getInliers(cloud, indices, threshold, current_model_inliers);

                // Compare fitness first.
                if (bestResult.first < result.first || (bestResult.first == result.first && bestResult.second > result.second )) {
                    bestResult = result;
                    bestModel.ModelCoefficients = planeModel.ModelCoefficients;
                    inliers_indices = current_model_inliers;
                }

            }
            inliers.push_back(inliers_indices);
            model_instances.push_back(bestModel);
        }

        if (model_type == SPHERE_MODEL) {
            RNG rng((uint64)-1);
            const unsigned num_rnd_model_points = 4;
            double bestRadius = 10000000;
            for (unsigned i = 0; i < max_iters; ++i) {
                vector<unsigned> current_model_inliers;
                SACModel model;

                for (unsigned j = 0; j < num_rnd_model_points; ++j) {
                    std::swap(indices[i], indices[rng.uniform(0, num_points)]);
                }

                for (unsigned j = 0; j < num_rnd_model_points; j++) {
                    unsigned idx = indices[i];
                    current_model_inliers.emplace_back(idx);
                }

                Point3d center;
                double radius;

                getSphereFromPoints(points, current_model_inliers, center, radius);
                SACSphereModel sphereModel (center, radius);
                pair<double, double> result = sphereModel.getInliers(cloud, indices, threshold, current_model_inliers);
                // Compare fitness first.
                if (bestResult.first < result.first || (bestResult.first == result.first && bestResult.second > result.second)
                    || (bestResult.first == result.first)) {

                    if (bestResult.first == result.first && bestModel.ModelCoefficients.size() == 4 && sphereModel.radius > bestRadius) continue;
                    bestResult = result;
                    bestModel.ModelCoefficients = sphereModel.ModelCoefficients;
                    bestModel.ModelCoefficients = sphereModel.ModelCoefficients;
                    inliers_indices = current_model_inliers;
                }

            }
            inliers.push_back(inliers_indices);
            model_instances.push_back(bestModel);
        }
    }

    void SACModelFitting::set_threshold (double threshold_value) {
        threshold = threshold_value;
    }

    void SACModelFitting::set_iterations (long unsigned iterations) {
        max_iters = iterations;
    }

} // ptcloud
}   // cv
