// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static void loadKeypoints(const std::string& vP_path,
                          const std::string& oP_path,
                          const std::string& sP_path,
                          const std::string& w_path,
                          std::vector<cv::KeyPoint>& keypoints,
                          std::vector<int>& nn)
{
    {
        std::ifstream file(vP_path.c_str());
        if (file.is_open())
        {
            float x = 0, y = 0;
            while (file >> x >> y)
            {
                keypoints.push_back(cv::KeyPoint(x, y, 0));
            }
        }
    }
    {
        std::ifstream file(oP_path.c_str());
        if (file.is_open())
        {
            float orientation = 0;
            size_t idx = 0;
            while (file >> orientation)
            {
                keypoints[idx].angle = static_cast<float>(orientation * 180.0 / CV_PI);
                idx++;
            }
        }
    }
    {
        std::ifstream file(sP_path.c_str());
        if (file.is_open())
        {
            float scale = 0;
            size_t idx = 0;
            while (file >> scale)
            {
                keypoints[idx].size = scale;
                idx++;
            }
        }
    }
    {
        std::ifstream file(w_path.c_str());
        if (file.is_open())
        {
            int neighborIdx = 0;
            while (file >> neighborIdx)
            {
                nn.push_back(neighborIdx);
            }
        }
    }

    ASSERT_TRUE(!keypoints.empty());
}

static void loadGroundTruth(const std::string& d1_path,
                            const std::string& b1_path,
                            std::vector<cv::DMatch>& groundTruth)
{
    std::vector<int> d1_vec;
    {
        std::ifstream file(d1_path.c_str());
        if (file.is_open())
        {
            int idx = 0;
            while (file >> idx)
            {
                d1_vec.push_back(idx-1);
            }
        }
    }

    std::vector<int> b1_vec;
    {
        std::ifstream file(b1_path.c_str());
        if (file.is_open())
        {
            int idx = 0;
            while (file >> idx)
            {
                b1_vec.push_back(idx-1);
            }
        }
    }

    ASSERT_TRUE(!d1_vec.empty());
    ASSERT_EQ(d1_vec.size(), b1_vec.size());

    for (size_t i = 0; i < d1_vec.size(); i++)
    {
        groundTruth.push_back(cv::DMatch(d1_vec[i], b1_vec[i], 0));
    }
}

TEST(XFeatures2d_LogosMatcher, logos_matcher_regression)
{
    const std::string vP1_path = cvtest::findDataFile("detectors_descriptors_evaluation/matching/LOGOS/vP1.txt");
    const std::string oP1_path = cvtest::findDataFile("detectors_descriptors_evaluation/matching/LOGOS/oP1.txt");
    const std::string sP1_path = cvtest::findDataFile("detectors_descriptors_evaluation/matching/LOGOS/sP1.txt");
    const std::string w1_path = cvtest::findDataFile("detectors_descriptors_evaluation/matching/LOGOS/w1.txt");

    const std::string vP2_path = cvtest::findDataFile("detectors_descriptors_evaluation/matching/LOGOS/vP2.txt");
    const std::string oP2_path = cvtest::findDataFile("detectors_descriptors_evaluation/matching/LOGOS/oP2.txt");
    const std::string sP2_path = cvtest::findDataFile("detectors_descriptors_evaluation/matching/LOGOS/sP2.txt");
    const std::string w2_path = cvtest::findDataFile("detectors_descriptors_evaluation/matching/LOGOS/w2.txt");

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<int> nn1, nn2;
    loadKeypoints(vP1_path, oP1_path, sP1_path, w1_path, keypoints1, nn1);
    loadKeypoints(vP2_path, oP2_path, sP2_path, w2_path, keypoints2, nn2);

    std::vector<cv::DMatch> matchesLogos;
    matchLOGOS(keypoints1, keypoints2, nn1, nn2, matchesLogos);

    std::vector<cv::DMatch> groundTruth;
    const std::string d1_path = cvtest::findDataFile("detectors_descriptors_evaluation/matching/LOGOS/d1.txt");
    const std::string b1_path = cvtest::findDataFile("detectors_descriptors_evaluation/matching/LOGOS/b1.txt");
    loadGroundTruth(d1_path, b1_path, groundTruth);

    int correctMatches = 0;
    for (size_t i = 0; i < matchesLogos.size(); i++)
    {
        for (size_t j = 0; j < groundTruth.size(); j++)
        {
            if (groundTruth[j].queryIdx == matchesLogos[i].queryIdx &&
                groundTruth[j].trainIdx == matchesLogos[j].trainIdx)
            {
                correctMatches++;
                break;
            }
        }
    }

    ASSERT_EQ(static_cast<int>(groundTruth.size()), correctMatches)
            << "groundTruth: " << groundTruth.size()
            << " ; matchesLogos: " << matchesLogos.size()
            << " ; correctMatches: " << correctMatches;
}

}} // namespace
