// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <cassert>
#include "opencv2/surface_matching.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include <numeric>
#include <limits>


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

    bool getCylinderFromPoints(const Mat cloud, const Mat normals_cld,
                                    const std::vector<unsigned> &inliers, vector<double> & model_coefficients) {
        assert(inliers.size() == 2);

        Mat _pointsAndNormals;

        assert(normals_cld.cols == cloud.cols);

        const Point3d* points = cloud.ptr<Point3d>(0);
        const Vec3f* normals = normals_cld.ptr<Vec3f>(0);

        if (fabs (points[inliers[0]].x - points[inliers[1]].x) <= std::numeric_limits<float>::epsilon () &&
            fabs (points[inliers[0]].y - points[inliers[1]].y) <= std::numeric_limits<float>::epsilon () &&
            fabs (points[inliers[0]].z - points[inliers[1]].z) <= std::numeric_limits<float>::epsilon ())
        {
            return (false);
        }
        Vec3f p1 (points[inliers[0]].x, points[inliers[0]].y, points[inliers[0]].z);
        Vec3f p2 (points[inliers[1]].x, points[inliers[1]].y, points[inliers[1]].z);

        Vec3f n1 (normals[inliers[0]] [0], normals[inliers[0]] [1], normals[inliers[0]] [2]);
        Vec3f n2 (normals[inliers[1]] [0], normals[inliers[1]] [1], normals[inliers[1]] [2]);
        Vec3f w = n1 + p1 - p2;

        float a = n1.dot (n1);
        float b = n1.dot (n2);
        float c = n2.dot (n2);
        float d = n1.dot (w);
        float e = n2.dot (w);
        float denominator = a*c - b*b;
        float sc, tc;
        // Compute the line parameters of the two closest points
        if (denominator < 1e-8)          // The lines are almost parallel
        {
        sc = 0.0f;
        tc = (b > c ? d / b : e / c);  // Use the largest denominator
        }
        else
        {
        sc = (b*e - c*d) / denominator;
        tc = (a*e - b*d) / denominator;
        }

        // point_on_axis, axis_direction
        Vec3f line_pt  = p1 + n1 + sc * n1;
        Vec3f line_dir = p2 + tc * n2 - line_pt;

        model_coefficients.resize (7);
        // point on line
        model_coefficients[0] = line_pt[0];
        model_coefficients[1] = line_pt[1];
        model_coefficients[2] = line_pt[2];

        double divide_by = std::sqrt (line_dir[0] * line_dir[0] +
                                            line_dir[1] * line_dir[1] +
                                            line_dir[2] * line_dir[2]);
        // direction of line;
        model_coefficients[3] = line_dir[0] / divide_by;
        model_coefficients[4] = line_dir[1] / divide_by;
        model_coefficients[5] = line_dir[2] / divide_by;

        double radius_squared = fabs((line_dir.cross(line_pt - p1)).dot(line_dir.cross(line_pt - p1)) / line_dir.dot(line_dir));

        // radius of cylinder
        model_coefficients[6] = sqrt(radius_squared);

        if (radius_squared == 0) return false;

        return (true);
    }

    SACPlaneModel::SACPlaneModel(Vec4d coefficients, Point3d set_center, Size2d set_size) {
        this -> ModelCoefficients.reserve(4);
        for (int i = 0; i < 4; i++) {
            this -> ModelCoefficients.push_back(coefficients[i]);
        }
        this -> size = set_size;

        this -> normal = Vec3d(coefficients[0], coefficients[1], coefficients[2]);
        this -> center = set_center;

        // Assign normal vector
        for (unsigned i = 0; i < 3; i++) normal[i] = coefficients[i];
    }

    SACPlaneModel::SACPlaneModel(Vec4d coefficients, Size2d set_size) {
        this -> ModelCoefficients.reserve(4);
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

    viz::WPlane SACPlaneModel::WindowWidget () {
        return viz::WPlane (this->center, this->normal, Vec3d(1, 0, 0), this->size, viz::Color::green());
    }

    pair<double, double> SACPlaneModel::getInliers(Mat cloud, vector<unsigned> indices, const double threshold, vector<unsigned>& inliers) {
        pair<double, double> result;
        inliers.clear();
        const Vec3f* points = cloud.ptr<Vec3f>(0);
        const unsigned num_points = indices.size();

        double magnitude_abc = sqrt(ModelCoefficients[0]*ModelCoefficients[0] + ModelCoefficients[1]* ModelCoefficients[1] + ModelCoefficients[2] * ModelCoefficients[2]);

        assert (magnitude_abc == 0);

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

        this -> ModelCoefficients.reserve(4);
        this -> ModelCoefficients.push_back(center.x);
        this -> ModelCoefficients.push_back(center.y);
        this -> ModelCoefficients.push_back(center.z);

        this -> ModelCoefficients.push_back(radius);
    }


    SACSphereModel::SACSphereModel(Vec4d coefficients) {
        this->ModelCoefficients.reserve(4);
        for (int i = 0; i < 4; i++) {
            this -> ModelCoefficients.push_back(coefficients[i]);
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

    viz::WCylinder SACCylinderModel::WindowWidget () {
        Point3d first_point = Point3d( Vec3d(pt_on_axis) + size * (axis_dir));
        Point3d second_point = Point3d(Vec3d(pt_on_axis) - size * (axis_dir));

        return viz::WCylinder (first_point, second_point, radius, 40, viz::Color::green());
    }

    SACCylinderModel::SACCylinderModel(const vector<double> coefficients) {
        assert(coefficients.size() == 7);
        for (int i = 0; i < 7; i++) {
            this -> ModelCoefficients.push_back(coefficients[i]);
        }
        this -> pt_on_axis = Point3d(coefficients[0], coefficients[1], coefficients[2]);
        this -> axis_dir = Vec3d(coefficients[3], coefficients[4], coefficients[5]);
        this -> radius = coefficients[6];

    }

    std::pair<double, double> SACCylinderModel::getInliers(Mat cloud, Mat normal_cloud, std::vector<unsigned> indices, const double threshold, std::vector<unsigned>& inliers, double normal_distance_weight_) {
        pair<double, double> result;
        inliers.clear();
        const Vec3f* points = cloud.ptr<Vec3f>(0);
        const Vec3f* normals = normal_cloud.ptr<Vec3f>(0);
        const unsigned num_points = indices.size();

        double fitness = 0;
        double rmse = 0;
        axis_dir = (axis_dir);

        // for (int i = 0; i < num_points; i++) {
        //     cout << i << " " << points[i] << endl;
        // }
        if(!isnan(radius)) { // radius may come out to be nan if selected points form a plane
            for (unsigned i = 0; i < num_points; i++) {
                unsigned ind = indices[i];
                Point3d pt (points[ind][0], points[ind][1], points[ind][2]);
                Vec3d normal (normals[ind][0], normals[ind][1], normals[ind][2]);
                normal = normal / sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);

                double distanceFromAxis = fabs((axis_dir.cross(pt_on_axis - pt)).dot(axis_dir.cross(pt_on_axis - pt)) / axis_dir.dot(axis_dir));

                double distanceFromSurface = fabs(distanceFromAxis - radius*radius);
                if (distanceFromSurface > threshold) continue;

                // Calculate the point's projection on the cylinder axis
                float dist = (pt.dot (axis_dir) - pt_on_axis.dot(axis_dir));
                Vec3d pt_proj = Vec3d(pt_on_axis) + dist * axis_dir;
                Vec3f dir = Vec3d(pt) - pt_proj;
                dir = dir / sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);

                // Calculate the angular distance between the point normal and the (dir=pt_proj->pt) vector
                double rad = normalize(normal).dot(dir);
                if (rad < -1.0) rad = -1.0;
                if (rad > 1.0) rad = 1.0;
                double d_normal = fabs(acos (rad));

                // convert 0 to PI/2
                d_normal = (std::min) (d_normal, M_PI - d_normal);

                // calculate overall distance as weighted sum of the two distances.
                double distance = fabs (normal_distance_weight_ * d_normal + (1 - normal_distance_weight_) * distanceFromSurface);

                if (distance > threshold) continue;

                inliers.emplace_back(ind);

                fitness += 1;
                rmse += max(0., distance);
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

    void SACModelFitting::setCloud(Mat inp_cloud, bool with_normals) {
        if (! with_normals) {

            // normals are not required.
            // the cloud should have three channels.
            assert(inp_cloud.channels() == 3 || (inp_cloud.channels() == 1 && (inp_cloud.cols == 3 || inp_cloud.rows == 3)));
            if (inp_cloud.rows == 1 && inp_cloud.channels() == 3) {
                cloud = inp_cloud.clone();
                return;
            }

            if (inp_cloud.channels() != 3 && inp_cloud.rows == 3) {
                inp_cloud = inp_cloud.t();
            }

            const long unsigned num_points = inp_cloud.rows;
            cloud = inp_cloud.reshape(3, num_points);
            cloud = cloud.t();

        }
        else {
            assert(inp_cloud.channels() == 1 && (inp_cloud.cols == 6 || inp_cloud.rows == 6));
            if (inp_cloud.rows == 6) {
                inp_cloud = inp_cloud.t();
            }

            Mat _cld;
            inp_cloud.colRange(0, 3).copyTo(_cld);

            Mat _normals;
            inp_cloud.colRange(3, 6).copyTo(_normals);

            this -> cloud = Mat(_cld).reshape(3, 1);
            this -> normals = Mat(_normals).reshape(3, 1);
            this -> normals_available = true;
        }
    }

    SACModelFitting::SACModelFitting (Mat set_cloud, int set_model_type, int set_method_type, double set_threshold, int set_max_iters)
            :cloud(set_cloud.clone()), model_type(set_model_type), method_type(set_method_type), threshold(set_threshold), max_iters(set_max_iters) {}

    SACModelFitting::SACModelFitting (int set_model_type, int set_method_type, double set_threshold, int set_max_iters)
        :model_type(set_model_type), method_type(set_method_type), threshold(set_threshold), max_iters(set_max_iters) {}

    bool SACModelFitting::fit_once(vector<int> labels /* = {} */) {

        // Only RANSAC supported ATM, need to integrate with Maksym's framework.
        if (method_type != SAC_METHOD_RANSAC) return false;

        // creates an array of indices for the points in the point cloud which will be appended as masks to denote inliers and outliers.
        const Vec3f* points = cloud.ptr<Vec3f>(0);
        unsigned num_points = cloud.cols;

        std::vector<unsigned> indices;

        if (labels.size() != num_points) {
            indices = std::vector<unsigned> (num_points);
            std::iota(std::begin(indices), std::end(indices), 0);
        } else {
            for (unsigned i = 0; i < num_points; i++) {
                if (labels[i] == -1) indices.push_back(i);
            }
        }

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

                for (unsigned j = 0; j < num_rnd_model_points;) {
                    std::swap(indices[j], indices[rng.uniform(0, num_points)]);
                    j++;
                }

                for (unsigned j = 0; j < num_rnd_model_points; j++) {
                    unsigned idx = indices[j];
                    current_model_inliers.emplace_back(idx);
                }

                Point3d center;
                Vec4d coefficients = getPlaneFromPoints(points, current_model_inliers, center);
                if (coefficients == Vec4d(0, 0, 0, 0)) continue;
                SACPlaneModel planeModel (coefficients, center);
                pair<double, double> result = planeModel.getInliers(cloud, indices, threshold, current_model_inliers);

                // Compare fitness first.
                if (bestResult.first < result.first || (bestResult.first == result.first && bestResult.second > result.second )) {
                    bestResult = result;
                    bestModel.ModelCoefficients = planeModel.ModelCoefficients;
                    inliers_indices = current_model_inliers;
                }

            }
            if (bestModel.ModelCoefficients.size()) {
                inliers.push_back(inliers_indices);
                model_instances.push_back(bestModel);
                return true;
            }
        }

        if (model_type == SPHERE_MODEL) {
            RNG rng((uint64)-1);
            const unsigned num_rnd_model_points = 4;
            double bestRadius = 10000000;
            for (unsigned i = 0; i < max_iters; ++i) {
                vector<unsigned> current_model_inliers;
                SACModel model;

                for (unsigned j = 0; j < num_rnd_model_points;) {
                    std::swap(indices[j], indices[rng.uniform(0, num_points)]);
                    j++;
                }

                for (unsigned j = 0; j < num_rnd_model_points; j++) {
                    unsigned idx = indices[j];
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
            if (bestModel.ModelCoefficients.size()) {
                inliers.push_back(inliers_indices);
                model_instances.push_back(bestModel);
                return true;
            }
        }

        if (model_type == CYLINDER_MODEL) {
            assert(this->normals_available == true);
            RNG rng((uint64)-1);
            const unsigned num_rnd_model_points = 2;

            if (!normals_available) {
                // Reshape the cloud for Compute Normals Function
                Mat _pointsAndNormals;
                Vec3d viewpoint(0, 0, 0);
                Mat _cld_reshaped = Mat(cloud).t();
                _cld_reshaped = _cld_reshaped.reshape(1);
                ppf_match_3d::computeNormalsPC3d(_cld_reshaped, _pointsAndNormals, 12, false, viewpoint);

                Mat(_pointsAndNormals.colRange(3,6)).copyTo(normals);
                normals = normals.reshape(3, num_points);
                normals = normals.t();
            }

            for (unsigned i = 0; i < max_iters; ++i) {
                vector<unsigned> current_model_inliers;
                SACModel model;

                for (unsigned j = 0; j < num_rnd_model_points;) {
                    std::swap(indices[j], indices[rng.uniform(0, num_points)]);
                    j++;
                }

                for (unsigned j = 0; j < num_rnd_model_points; j++) {
                    unsigned idx = indices[j];
                    current_model_inliers.emplace_back(idx);
                }

                Point3d center;
                vector<double> coefficients;
                bool valid_model = getCylinderFromPoints(cloud, normals, current_model_inliers, coefficients);

                if (!valid_model) continue;

                SACCylinderModel cylinderModel (coefficients);

                pair<double, double> result = cylinderModel.getInliers(cloud, normals, indices, threshold, current_model_inliers, normal_distance_weight_);

                // Compare fitness first.
                if (bestResult.first < result.first || (bestResult.first == result.first && bestResult.second > result.second)) {
                    // if (bestResult.first == result.first && bestModel.ModelCoefficients.size() == 7) continue;
                    bestResult = result;
                    bestModel.ModelCoefficients = cylinderModel.ModelCoefficients;
                    bestModel.ModelCoefficients = cylinderModel.ModelCoefficients;
                    inliers_indices = current_model_inliers;
                    cout << bestResult.first << endl;
                }

            }

            if (bestModel.ModelCoefficients.size()) {
                inliers.push_back(inliers_indices);
                model_instances.push_back(bestModel);
                return true;
            }

        }
        return false;
    }

    void SACModelFitting::segment(float remaining_cloud_threshold /*=0.3*/) {
        unsigned num_points = cloud.cols;

        std::vector<unsigned> indices (num_points);
        std::iota(std::begin(indices), std::end(indices), 0);

        std::vector<int> point_labels (num_points, -1);

        long num_segmented_points = 0;

        int label = 0;
        while ( (float) num_segmented_points / num_points < (1 - remaining_cloud_threshold )) {
            label = label + 1;
            bool successful_fitting = fit_once(point_labels);

            if (!successful_fitting) {
                cout << "Could not fit the required model" << endl;
                break;
            }
            vector<unsigned> latest_model_inliers = inliers.back();
            num_segmented_points += latest_model_inliers.size();

            // This loop is for implementation purposes only, and maps each point to a label.
            // All the points still labelled with -1 are non-segmented.
            // This way, complexity of the finding non-segmented is decreased to O(n).
            for(unsigned long i = 0; i < latest_model_inliers.size(); i++)
            {
                point_labels[latest_model_inliers[i]] = label;
            }
            label++;
        }
    }

    void SACModelFitting::set_threshold (double threshold_value) {
        threshold = threshold_value;
    }

    void SACModelFitting::set_iterations (long unsigned iterations) {
        max_iters = iterations;
    }

    void SACModelFitting::set_normal_distance_weight(double weight) {
        if (weight > 1) {
            normal_distance_weight_ = 1;
        } else if (weight < 0) {
            normal_distance_weight_ = 0;
        } else {
            normal_distance_weight_ = weight;
        }
    }

} // ptcloud
}   // cv
