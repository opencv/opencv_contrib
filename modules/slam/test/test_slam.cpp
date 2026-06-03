// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test {
namespace {

// ============================================================
// Compile-time / value tests (no VO instance needed)
// ============================================================

TEST(SLAM, VOConfig_Default)
{
    cv::vo::VOConfig config;
    EXPECT_TRUE(config.camera_config_file.empty());
    EXPECT_TRUE(config.vocab_file.empty());
}

TEST(SLAM, VOState_Values)
{
    EXPECT_EQ(static_cast<int>(cv::vo::VOState::NotInitialized), 0);
    EXPECT_EQ(static_cast<int>(cv::vo::VOState::Initializing), 1);
    EXPECT_EQ(static_cast<int>(cv::vo::VOState::Tracking), 2);
    EXPECT_EQ(static_cast<int>(cv::vo::VOState::Lost), 3);
}

TEST(SLAM, SLAMMode_Values)
{
    EXPECT_EQ(static_cast<int>(cv::vo::SLAMMode::SLAM), 0);
    EXPECT_EQ(static_cast<int>(cv::vo::SLAMMode::LOCALIZATION), 1);
}

// ============================================================
// VisualOdometry::create() — invalid config throws
// ============================================================

TEST(SLAM, CreateFromInvalidYAML_Throws)
{
    cv::vo::VOConfig config;
    config.camera_config_file = "/nonexistent/path/camera.yaml";
    config.vocab_file = "/nonexistent/path/orb_vocab.fbow";

    // VisualOdometry::create() reads the YAML during initialization
    // and throws on invalid file path.
    EXPECT_ANY_THROW({
        auto vo = cv::vo::VisualOdometry::create(config,
            cv::ORB::create(1000),
            cv::BFMatcher::create(cv::NORM_HAMMING));
    });
}

TEST(SLAM, CreateFromEmptyConfig_Throws)
{
    cv::vo::VOConfig config;
    EXPECT_ANY_THROW({
        auto vo = cv::vo::VisualOdometry::create(config,
            cv::ORB::create(1000),
            cv::BFMatcher::create(cv::NORM_HAMMING));
    });
}

// ============================================================
// Tests requiring valid config + vocabulary (skipped if unavailable)
//
// These tests use a helper that checks for test data availability.
// To run them, set OPENCV_TEST_DATA_PATH to a directory containing
// EuRoC.yaml and orb_vocab.fbow.
// ============================================================

static std::string findTestDataFile(const std::string& relativePath)
{
    std::vector<std::string> searchPaths = {
        "/root/opencv_contrib_slam/modules/slam/testdata/",
        "/root/opencv_contrib_slam_dev/modules/slam/testdata/",
    };

    const char* envPath = std::getenv("OPENCV_TEST_DATA_PATH");
    if (envPath) {
        searchPaths.insert(searchPaths.begin(), std::string(envPath) + "/");
        searchPaths.insert(searchPaths.begin() + 1, std::string(envPath) + "/vocab/");
        searchPaths.insert(searchPaths.begin() + 2, std::string(envPath) + "/config/");
    }

    for (const auto& dir : searchPaths) {
        std::string fullPath = dir + relativePath;
        std::ifstream f(fullPath);
        if (f.good()) {
            return fullPath;
        }
    }
    return "";
}

// Fixture for tests that need a valid VisualOdometry instance
class SLAMWithConfig : public ::testing::Test {
protected:
    void SetUp() override {
        configFile_ = findTestDataFile("config/euroc_mh01.yaml");
        vocabFile_ = findTestDataFile("vocab/orb_vocab.fbow");

        if (configFile_.empty() || vocabFile_.empty()) {
            hasData_ = false;
            return;
        }
        hasData_ = true;
    }

    void TearDown() override {
        if (vo_) {
            vo_->release();
            vo_.reset();
        }
    }

    cv::Ptr<cv::vo::VisualOdometry> createVO(
        const cv::Ptr<cv::Feature2D>& detector = cv::ORB::create(1000),
        const cv::Ptr<cv::DescriptorMatcher>& matcher = cv::BFMatcher::create(cv::NORM_HAMMING))
    {
        cv::vo::VOConfig config;
        config.camera_config_file = configFile_;
        config.vocab_file = vocabFile_;
        vo_ = cv::vo::VisualOdometry::create(config, detector, matcher);
        return vo_;
    }

    std::string configFile_;
    std::string vocabFile_;
    cv::Ptr<cv::vo::VisualOdometry> vo_;
    bool hasData_ = false;
};

TEST_F(SLAMWithConfig, CreateAndState)
{
    if (!hasData_) return;
    auto vo = createVO();
    ASSERT_TRUE(vo != nullptr);
    EXPECT_EQ(vo->getState(), cv::vo::VOState::NotInitialized);
    EXPECT_FALSE(vo->isInitialized());
}

TEST_F(SLAMWithConfig, SetFeatureDetector)
{
    if (!hasData_) return;
    auto vo = createVO();
    auto orb = cv::ORB::create(2000);
    vo->setFeatureDetector(orb);
    auto retrieved = vo->getFeatureDetector();
    EXPECT_FALSE(retrieved.empty());
}

TEST_F(SLAMWithConfig, SetMatcher)
{
    if (!hasData_) return;
    auto vo = createVO();
    auto new_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    vo->setMatcher(new_matcher);
    auto retrieved = vo->getMatcher();
    EXPECT_FALSE(retrieved.empty());
}

TEST_F(SLAMWithConfig, BackendControl)
{
    if (!hasData_) return;
    auto vo = createVO();
    EXPECT_TRUE(vo->isBackendEnabled());
    vo->setBackendEnabled(false);
    EXPECT_FALSE(vo->isBackendEnabled());
    vo->setBackendEnabled(true);
    EXPECT_TRUE(vo->isBackendEnabled());
}

TEST_F(SLAMWithConfig, LoopClosureControl)
{
    if (!hasData_) return;
    auto vo = createVO();
    vo->setLoopClosureEnabled(false);
    EXPECT_FALSE(vo->isLoopClosureEnabled());
    vo->setLoopClosureEnabled(true);
    EXPECT_TRUE(vo->isLoopClosureEnabled());
}

TEST_F(SLAMWithConfig, ModeControl)
{
    if (!hasData_) return;
    auto vo = createVO();
    vo->setMode(cv::vo::SLAMMode::SLAM);
    EXPECT_EQ(vo->getMode(), cv::vo::SLAMMode::SLAM);
    vo->setMode(cv::vo::SLAMMode::LOCALIZATION);
    EXPECT_EQ(vo->getMode(), cv::vo::SLAMMode::LOCALIZATION);
}

TEST_F(SLAMWithConfig, ResetDoesNotCrash)
{
    if (!hasData_) return;
    auto vo = createVO();
    vo->reset();
    EXPECT_EQ(vo->getState(), cv::vo::VOState::NotInitialized);
}

TEST_F(SLAMWithConfig, EmptyTrajectory)
{
    if (!hasData_) return;
    auto vo = createVO();
    auto traj = vo->getTrajectory();
    EXPECT_TRUE(traj.empty());
}

TEST_F(SLAMWithConfig, EmptyMapPoints)
{
    if (!hasData_) return;
    auto vo = createVO();
    auto points = vo->getMapPoints();
    // Map may contain initial keyframe even before processing frames
    EXPECT_TRUE(points.empty());
}

}} // namespace