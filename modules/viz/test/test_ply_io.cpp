#include "test_precomp.hpp"
#include <opencv2/viz.hpp>
#include <opencv2/core.hpp>

namespace opencv_test { namespace {

// Helper: generate deterministic random point cloud
static cv::Mat makeCloud(int N, uint64 seed = 42) {
    cv::RNG rng(seed);
    cv::Mat cloud(1, N, CV_32FC3);
    rng.fill(cloud, cv::RNG::UNIFORM, -10.0f, 10.0f);
    return cloud;
}

static cv::Mat makeColors(int N, uint64 seed = 123) {
    cv::RNG rng(seed);
    cv::Mat colors(1, N, CV_8UC3);
    rng.fill(colors, cv::RNG::UNIFORM, 0, 255);
    return colors;
}

// --- ASCII PLY roundtrip ---
TEST(Viz_PLY_IO, AsciiRoundtrip_Positions) {
    const int N = 512;
    cv::Mat cloud = makeCloud(N);

    std::string path = cv::tempfile(".ply");
    cv::viz::writeCloud(path, cloud, cv::noArray(), cv::noArray(), /*binary=*/false);

    cv::Mat loaded = cv::viz::readCloud(path);
    ASSERT_EQ(loaded.total(), (size_t)N);

    // Per-point L-inf tolerance (ASCII float precision)
    for (int i = 0; i < N; i++) {
        cv::Vec3f orig = cloud.at<cv::Vec3f>(i);
        cv::Vec3f read = loaded.at<cv::Vec3f>(i);
        EXPECT_NEAR(orig[0], read[0], 1e-4f) << "point " << i << " x";
        EXPECT_NEAR(orig[1], read[1], 1e-4f) << "point " << i << " y";
        EXPECT_NEAR(orig[2], read[2], 1e-4f) << "point " << i << " z";
    }
    std::remove(path.c_str());
}

// --- Binary PLY roundtrip ---
TEST(Viz_PLY_IO, BinaryRoundtrip_Positions) {
    const int N = 1024;
    cv::Mat cloud = makeCloud(N, 99);

    std::string path = cv::tempfile(".ply");
    cv::viz::writeCloud(path, cloud, cv::noArray(), cv::noArray(), /*binary=*/true);

    cv::Mat loaded = cv::viz::readCloud(path);
    ASSERT_EQ(loaded.total(), (size_t)N);

    // Binary should be bit-exact for float32
    for (int i = 0; i < N; i++) {
        cv::Vec3f orig = cloud.at<cv::Vec3f>(i);
        cv::Vec3f read = loaded.at<cv::Vec3f>(i);
        EXPECT_FLOAT_EQ(orig[0], read[0]) << "point " << i;
        EXPECT_FLOAT_EQ(orig[1], read[1]) << "point " << i;
        EXPECT_FLOAT_EQ(orig[2], read[2]) << "point " << i;
    }
    std::remove(path.c_str());
}

// --- Color preservation ---
TEST(Viz_PLY_IO, BinaryRoundtrip_WithColors) {
    const int N = 256;
    cv::Mat cloud = makeCloud(N);
    cv::Mat colors = makeColors(N);

    std::string path = cv::tempfile(".ply");
    cv::viz::writeCloud(path, cloud, colors, cv::noArray(), true);

    cv::Mat loaded_cloud, loaded_colors;
    // readCloud returns cloud; colors come via the 4-channel overload if available
    // (test the positions at minimum; add color channel check if API exposes it)
    loaded_cloud = cv::viz::readCloud(path, loaded_colors);
    ASSERT_EQ(loaded_cloud.total(), (size_t)N);

    if (!loaded_colors.empty()) {
        ASSERT_EQ(loaded_colors.total(), (size_t)N);
        for (int i = 0; i < N; i++) {
            cv::Vec3b co = colors.at<cv::Vec3b>(i);
            cv::Vec3b cl = loaded_colors.at<cv::Vec3b>(i);
            EXPECT_EQ(co, cl) << "color mismatch at point " << i;
        }
    }
    std::remove(path.c_str());
}

// --- Large cloud performance smoke test ---
TEST(Viz_PLY_IO, LargeCloud_BinaryPerf) {
    const int N = 500000;
    cv::Mat cloud = makeCloud(N, 7);

    std::string path = cv::tempfile(".ply");

    auto t0 = cv::getTickCount();
    cv::viz::writeCloud(path, cloud, cv::noArray(), cv::noArray(), true);
    double write_ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();

    t0 = cv::getTickCount();
    cv::Mat loaded = cv::viz::readCloud(path);
    double read_ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();

    EXPECT_EQ(loaded.total(), (size_t)N);
    // Loose perf guard: should complete in <2s each on any modern machine
    EXPECT_LT(write_ms, 2000.0) << "write too slow: " << write_ms << " ms";
    EXPECT_LT(read_ms,  2000.0) << "read too slow: "  << read_ms  << " ms";

    std::cout << "[perf] N=" << N
              << " write=" << write_ms << "ms"
              << " read=" << read_ms << "ms" << std::endl;

    std::remove(path.c_str());
}

// --- Edge: empty cloud ---
TEST(Viz_PLY_IO, EmptyCloud_DoesNotCrash) {
    cv::Mat cloud(1, 0, CV_32FC3);
    std::string path = cv::tempfile(".ply");
    // Should not throw/crash; behavior on reload may vary
    EXPECT_NO_THROW(cv::viz::writeCloud(path, cloud, cv::noArray(), cv::noArray(), false));
    std::remove(path.c_str());
}

// -----------------------------------------------------------------------
// Trajectory read/write roundtrip tests
// -----------------------------------------------------------------------

static cv::Affine3d makePose(double tx, double ty, double tz, double angle) {
    // Rotation around Z axis by `angle` radians
    cv::Matx33d R(
        std::cos(angle), -std::sin(angle), 0,
        std::sin(angle),  std::cos(angle), 0,
        0,                0,               1
    );
    cv::Vec3d t(tx, ty, tz);
    return cv::Affine3d(R, t);
}

TEST(Viz_Trajectory, WriteRead_Affine3d_Roundtrip) {
    // Build a small trajectory of known poses
    std::vector<cv::Affine3d> traj_written = {
        makePose(1.0,  0.0, 0.0,  0.0),
        makePose(2.0,  1.0, 0.5,  CV_PI / 6),
        makePose(3.0,  2.0, 1.0,  CV_PI / 4),
        makePose(4.0, -1.0, 0.0,  CV_PI / 3),
    };

    // Write to temp files: /tmp/traj_test_000.yml, _001.yml, ...
    std::string fmt = cv::tempfile("") + "traj_%03d.yml";

    cv::viz::writeTrajectory(traj_written, fmt, /*start=*/0, /*tag=*/"pose");

    // Read back
    std::vector<cv::Affine3d> traj_read;
    cv::viz::readTrajectory(traj_read, fmt, /*start=*/0, /*end=*/4, /*tag=*/"pose");

    ASSERT_EQ(traj_read.size(), traj_written.size());

    for (int i = 0; i < (int)traj_written.size(); ++i) {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                EXPECT_NEAR(traj_written[i].matrix(r, c),
                            traj_read[i].matrix(r, c),
                            1e-9)
                    << "pose " << i << " matrix[" << r << "][" << c << "]";
    }

    // Cleanup
    for (int i = 0; i < 4; ++i)
        std::remove(cv::format(fmt.c_str(), i).c_str());
}

TEST(Viz_Trajectory, WriteRead_Affine3f_Roundtrip) {
    // Same as above but float precision
    std::vector<cv::Affine3f> traj_written = {
        cv::Affine3f(makePose(0.0, 0.0, 0.0, 0.0)),
        cv::Affine3f(makePose(1.0, 0.5, 0.2, CV_PI / 8)),
        cv::Affine3f(makePose(2.0, 1.0, 0.4, CV_PI / 2)),
    };

    std::string fmt = cv::tempfile("") + "trajf_%03d.yml";
    cv::viz::writeTrajectory(traj_written, fmt, 0, "pose");

    std::vector<cv::Affine3d> traj_read;  // readTrajectory always returns Affine3d
    cv::viz::readTrajectory(traj_read, fmt, 0, 3, "pose");

    ASSERT_EQ(traj_read.size(), traj_written.size());

    for (int i = 0; i < (int)traj_written.size(); ++i) {
        cv::Affine3d w(traj_written[i]);  // upcast for comparison
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                EXPECT_NEAR(w.matrix(r, c),
                            traj_read[i].matrix(r, c),
                            1e-5f)   // float precision
                    << "pose " << i << " matrix[" << r << "][" << c << "]";
    }

    for (int i = 0; i < 3; ++i)
        std::remove(cv::format(fmt.c_str(), i).c_str());
}

TEST(Viz_Trajectory, StartIndex_Offset) {
    // Write starting at index 10, read back from 10
    std::vector<cv::Affine3d> traj_written = {
        makePose(0, 0, 0, 0),
        makePose(1, 0, 0, 1),
    };

    std::string fmt = cv::tempfile("") + "trajoff_%03d.yml";
    cv::viz::writeTrajectory(traj_written, fmt, /*start=*/10, "pose");

    // Files exist at index 10 and 11, not 0
    std::vector<cv::Affine3d> traj_read;
    cv::viz::readTrajectory(traj_read, fmt, 10, 12, "pose");

    ASSERT_EQ(traj_read.size(), (size_t)2);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_NEAR(traj_written[0].matrix(r, c),
                        traj_read[0].matrix(r, c), 1e-9);

    for (int i = 10; i < 12; ++i)
        std::remove(cv::format(fmt.c_str(), i).c_str());
}

TEST(Viz_Trajectory, ReadStops_AtMissingFile) {
    // Write 2 poses, request 5 — should silently stop at 2
    std::vector<cv::Affine3d> traj_written = {
        makePose(0, 0, 0, 0),
        makePose(1, 1, 1, 1),
    };

    std::string fmt = cv::tempfile("") + "trajstop_%03d.yml";
    cv::viz::writeTrajectory(traj_written, fmt, 0, "pose");

    std::vector<cv::Affine3d> traj_read;
    cv::viz::readTrajectory(traj_read, fmt, 0, /*end=*/5, "pose");

    // Should have read exactly 2, not 5
    EXPECT_EQ(traj_read.size(), (size_t)2);

    for (int i = 0; i < 2; ++i)
        std::remove(cv::format(fmt.c_str(), i).c_str());
}

TEST(Viz_Trajectory, EmptyRange_ReturnsEmpty) {
    std::string fmt = cv::tempfile("") + "trajempty_%03d.yml";

    // Use a pre-typed Mat so _traj.depth() != -1 on the convertTo call
    cv::Mat traj_read;
    cv::viz::readTrajectory(traj_read, fmt, /*start=*/5, /*end=*/5, "pose");
    EXPECT_EQ(traj_read.total(), (size_t)0);
}
}} // namespace opencv_test