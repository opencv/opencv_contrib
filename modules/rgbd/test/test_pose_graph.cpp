// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;

TEST( PoseGraph, sphereG2O )
{
    //TODO: add code itself

    //DEBUG
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_INFO);

    // The dataset was taken from here: https://lucacarlone.mit.edu/datasets/
    // Connected paper:
    // L.Carlone, R.Tron, K.Daniilidis, and F.Dellaert.
    // Initialization Techniques for 3D SLAM : a Survey on Rotation Estimation and its Use in Pose Graph Optimization.
    // In IEEE Intl.Conf.on Robotics and Automation(ICRA), pages 4597 - 4604, 2015.

    std::string filename = cvtest::TS::ptr()->get_data_path() + "sphere_bignoise_vertex3.g2o";
    //std::string filename = "C:\\Users\\rvasilik\\Downloads\\torus3D.g2o";
    PoseGraph pg(filename);

    //writePg(pg, "C:\\Temp\\g2opt\\in.obj");

    double t0 = cv::getTickCount();

    pg.optimize();

    double t1 = cv::getTickCount();

    std::cout << "time: " << (t1 - t0) / cv::getTickFrequency() << std::endl;

    //writePg(pg, "C:\\Temp\\g2opt\\out.obj");

    viz::Viz3d debug("debug");
    std::vector<Point3d> sv, dv;
    for (const auto& e : pg.edges)
    {
        size_t sid = e.sourceNodeId, tid = e.targetNodeId;
        Point3d sp = pg.nodes.at(sid).getPose().translation();
        Point3d tp = pg.nodes.at(tid).getPose().translation();
        sv.push_back(sp);
        dv.push_back(tp - sp);
    }
    debug.showWidget("after", viz::WCloudNormals(sv, dv, 1, 1, viz::Color::green()));
    debug.spin();

}

}} // namespace