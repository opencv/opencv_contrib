// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/sfm/rpc.hpp>
#include <iostream>

int main()
{
    // A synthetic camera for a small geographic neighbourhood. Real coefficients
    // and normalization parameters come from the image provider's RPC metadata.
    cv::sfm::RPCModel camera;
    camera.worldOffset = cv::Vec3d(-71, 42, 100);
    camera.worldScale = cv::Vec3d(0.05, 0.04, 500);
    camera.imageOffset = cv::Vec2d(4000, 3000);
    camera.imageScale = cv::Vec2d(4000, 3000);
    camera.coefficients(0, 3) = 0.1; // Height contributes to the sample coordinate.
    camera.coefficients(2, 3) = -0.05;

    cv::sfm::RPCLocalFrame frame(cv::Point3d(-71, 42, 100), CV_PI / 6);
    std::vector<cv::Point3d> ground = {cv::Point3d(100, 200, 10), cv::Point3d(-50, 25, 10)};
    std::vector<cv::Point2d> image;
    cv::sfm::projectPointsRPC(ground, camera, image, &frame);

    std::vector<cv::Point3d> initial(ground.size(), cv::Point3d(0, 0, 10)), recovered;
    std::vector<uchar> success;
    cv::sfm::backProjectPointsRPC(image, camera, cv::Vec4d(0, 0, 1, -10), initial,
                                 recovered, success, &frame, 1e-7);
    for (size_t i = 0; i < ground.size(); ++i)
    {
        if (!success[i])
        {
            std::cerr << "Inverse search failed for point " << i << std::endl;
            return 1;
        }
        std::cout << ground[i] << " -> " << image[i] << " -> " << recovered[i] << std::endl;
    }
    return 0;
}
