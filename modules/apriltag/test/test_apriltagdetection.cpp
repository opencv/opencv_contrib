// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include <numeric>  //std::accumulate

#define DISPLAY_DEBUG 0
// slow to load the tag family data (too big), skip it
#define TEST_BIG_FAMILY 0

namespace opencv_test { namespace {

static double toRad(double deg)
{
    return  deg * CV_PI / 180;
}

Matx31d generatorNED(double lon_deg, double lat_deg, double radius)
{
    double lon = toRad(lon_deg);
    double lat = toRad(lat_deg);

    //https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
    Matx33d R(-sin(lat)*cos(lon), -sin(lon),  -cos(lat)*cos(lon),
              -sin(lat)*sin(lon), cos(lon),   -cos(lat)*sin(lon),
              cos(lat),           0,          -sin(lat));

    Matx31d radius_v(0, 0, -radius);
    Matx31d pos = R * radius_v;

    return pos;
}

Matx31d normalize(const Matx31d& v)
{
    return (1/norm(v)) * v;
}

Matx31d cross(const Matx31d& v1, const Matx31d& v2)
{
    return Matx31d(Vec3d(v1(0), v1(1), v1(2)).cross(Vec3d(v2(0), v2(1), v2(2))));
}

// see glm::lookAt()
Matx44d lookAt(const Matx31d& eye, const Matx31d& center, const Matx31d& up)
{
    Matx31d f = normalize(center - eye);
    Matx31d s = normalize(cross(f, up));
    Matx31d u = cross(s, f);

    Matx44d T = Matx44d::eye();
    T(0,0) = s(0);
    T(0,1) = s(1);
    T(0,2) = s(2);

    T(1,0) = u(0);
    T(1,1) = u(1);
    T(1,2) = u(2);

    T(2,0) = -f(0);
    T(2,1) = -f(1);
    T(2,2) = -f(2);

    T(0,3) = -s.dot(eye);
    T(1,3) = -u.dot(eye);
    T(2,3) = -f.dot(eye);

    return T;
}

void extractRt(const Matx44d& T, Matx33d& R, Matx31d& tvec)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R(i,j) = T(i,j);
        }
        tvec(i,0) = T(i,3);
    }
}

Matx44d getCanonicalTransformation(const Matx33d& cameraMatrix, double tagSize, double tagSizePx)
{
    // compute tz to project the tag with tagSizePx size
    double tz = tagSize * cameraMatrix(0,0) / tagSizePx;

    Matx44d T = Matx44d::eye();
    T(2,3) = tz;

    return T;
}

void computeC2TC1(const Matx33d &R1, const Matx31d &tvec1, const Matx33d &R2, const Matx31d &tvec2,
                  Matx33d &R_1to2, Matx31d &tvec_1to2)
{
    R_1to2 = R2 * R1.t();
    tvec_1to2 = R2 * (-R1.t()*tvec1) + tvec2;
}

Matx33d computeHomography(const Matx33d &R_1to2, const Matx31d &tvec_1to2, double d_inv, const Matx31d &normal)
{
    Matx33d homography = R_1to2 + d_inv * tvec_1to2*normal.t();
    return homography;
}

static double getMax(const vector<double>& v)
{
    return *max_element(v.begin(), v.end());
}

static double getMean(const vector<double>& v)
{
    if (v.empty())
    {
        return 0.0;
    }

    double sum = accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

static double getMedian(const vector<double>& v)
{
    if (v.empty())
    {
        return 0.0;
    }

    vector<double> v_copy = v;
    size_t size = v_copy.size();

    size_t n = size / 2;
    nth_element(v_copy.begin(), v_copy.begin() + n, v_copy.end());
    double val_n = v_copy[n];

    if (size % 2 == 1)
    {
        return val_n;
    } else
    {
        nth_element(v_copy.begin(), v_copy.begin() + n - 1, v_copy.end());
        return 0.5 * (val_n + v_copy[n - 1]);
    }
}

class CV_ApriltagDetectionPerspective : public cvtest::BaseTest
{
public:
    CV_ApriltagDetectionPerspective(const apriltag::AprilTagFamily& tagFamily, double factor,
                                    double threshold, int nbTags, double step=20.0,
                                    double rotationErrThresh=5e-1, double translationErrThresh=5e-3) :
         m_tagFamily(tagFamily), m_factor(factor), m_accuracyThreshold(threshold), m_nbTags(nbTags),
         m_step(step), m_rotationErrorThreshold(rotationErrThresh), m_translationErrorThreshold(translationErrThresh)
    {
    }

protected:
    virtual void run(int);

    apriltag::AprilTagFamily m_tagFamily;
    double m_factor;
    double m_accuracyThreshold;
    int m_nbTags;
    double m_step;
    double m_rotationErrorThreshold;
    double m_translationErrorThreshold;
};

void CV_ApriltagDetectionPerspective::run(int)
{
    const int width = 640, height = 480;
    const int width_2 = width/2, height_2 = height/2;

    Matx33d cameraMatrix(600, 0, width_2,
                         0, 600, height_2,
                         0, 0, 1);

    const double tagSize = 0.1f;
    const int tagSizePx = 200;
    const int tagSizePx_2 = tagSizePx/2;
    const double tagSize_2 = tagSize/2;
    vector<Point3d> objectPoints;
    objectPoints.push_back(Point3d(-tagSize_2*m_factor,  tagSize_2*m_factor, 0));
    objectPoints.push_back(Point3d( tagSize_2*m_factor,  tagSize_2*m_factor, 0));
    objectPoints.push_back(Point3d( tagSize_2*m_factor, -tagSize_2*m_factor, 0));
    objectPoints.push_back(Point3d(-tagSize_2*m_factor, -tagSize_2*m_factor, 0));

    apriltag::AprilTagDetector detector(m_tagFamily);
    detector.setQuadDecimate(1.0);

    // image buffer
    Mat tagImg, curTagImg;
    Mat canonicalTagImg = Mat::zeros(height, width, CV_8UC1);
#if DISPLAY_DEBUG
    Mat imgDisplay;
#endif

    RNG& rng = ts->get_rng();
    int nbTests = 0, nbGoodCornersAccuracy = 0, nbGoodPosesAccuracy = 0;
    int nbGoodFirstSolutions = 0, nbGoodSecondSolutions = 0;
    vector<double> rmse_vec;
    for (double radius = 0.5; radius < 1.0; radius += 0.1)
    {
        // Lambda
        for (double lon = 0; lon < 180; lon += m_step)
        {
            // Phi
            for (double lat = 30; lat < 150; lat += m_step, nbTests++)
            {
                const int tagId = nbTests % m_nbTags;
                detector.drawTag(tagImg, Size(tagSizePx,tagSizePx), tagId);
                canonicalTagImg = 0;
                Mat ref = canonicalTagImg(Rect(width_2 - tagSizePx_2, height_2 - tagSizePx_2, tagSizePx, tagSizePx));
                tagImg.copyTo(ref);

#if DISPLAY_DEBUG
                imshow("Canonical Tag", canonicalTagImg);
#endif

                // canonical pose
                // compute the transformation to project the tag centered in the image with size = tagSizePx px
                Matx44d c1To = getCanonicalTransformation(cameraMatrix, tagSize, tagSizePx);
                Matx33d R1;
                Matx31d tvec1;
                extractRt(c1To, R1, tvec1);

                // generate a position using the NED coordinates
                Matx31d pos = generatorNED(lon, lat, radius);

                // look at a particular point
                Matx31d center(rng.uniform(-tagSize, tagSize), rng.uniform(-tagSize, tagSize), 0);
                Matx31d up(0,0,1);
                Matx44d c2To = lookAt(pos, center, up);

                Matx33d R2;
                Matx31d rvec2, tvec2;
                extractRt(c2To, R2, tvec2);
                Rodrigues(R2, rvec2);

                // ground truth tag corners locations in the current pose
                vector<Point2d> imagePoints;
                projectPoints(objectPoints, rvec2, tvec2, cameraMatrix, noArray(), imagePoints);

                // compute transformation to go from the canonical pose to the current pose
                Matx33d R_1to2;
                Matx31d tvec_1to2;
                computeC2TC1(R1, tvec1, R2, tvec2, R_1to2, tvec_1to2);

                // compute induced homography from the camera displacement
                Matx31d normal(0,0,1);
                Matx31d normal1 = R1*normal;
                double d1_inv = 1 / normal1.dot(tvec1);

                Matx33d H = cameraMatrix * computeHomography(R_1to2, tvec_1to2, d1_inv, normal1) * cameraMatrix.inv();
                H *= 1/H(2,2);

                // warp the canonical tag image to the desired current pose
                warpPerspective(canonicalTagImg, curTagImg, H, Size(width,height));

                // detect tag in the image
                vector<vector<Point2d> > corners;
                vector<int> ids;
                detector.detectTags(curTagImg, corners, ids);

                EXPECT_EQ(corners.size(), ids.size());
                EXPECT_EQ(static_cast<int>(corners.size()), 1);

#if DISPLAY_DEBUG
                cvtColor(curTagImg, imgDisplay, COLOR_GRAY2BGR);
                detector.drawDetectedTags(imgDisplay, corners);

                for (size_t i = 0; i < imagePoints.size(); i++)
                {
                    line(imgDisplay, imagePoints[i], imagePoints[(i+1) % imagePoints.size()], Scalar(0,0,255), 1);
                }
#endif

                if (corners.size() == 1)
                {
#if DISPLAY_DEBUG
                    cout << endl;
#endif
                    double rmse = 0.0;
                    for (size_t i = 0; i < corners[0].size(); i++)
                    {
                        Point2d diff = corners[0][i] - imagePoints[i];
                        rmse += diff.x*diff.x + diff.y*diff.y;
#if DISPLAY_DEBUG
                        cout << "detected: " << corners[0][i] << " ; real: " << imagePoints[i] << endl;
#endif
                    }
                    rmse = sqrt(rmse / (2*corners[0].size()));
                    rmse_vec.push_back(rmse);
                    if (rmse < m_accuracyThreshold)
                    {
                        nbGoodCornersAccuracy++;
                    }
#if DISPLAY_DEBUG
                    cout << "RMSE: " << rmse << endl;
                    putText(imgDisplay, format("RMSE: %0.2f", rmse), Point(20,20), 0, 0.6, Scalar(0,0,255));
                    putText(imgDisplay, format("id: %d / true id: %d", ids[0], tagId), Point(20,40), 0, 0.6, Scalar(0,0,255));
#endif
                }

                vector<pair<Matx31d, Matx31d> > rvecs, tvecs;
                detector.estimateTagsPosePnP(corners, tagSize*m_factor, cameraMatrix, noArray(), rvecs, tvecs);

                EXPECT_EQ(rvecs.size(), tvecs.size());
                EXPECT_EQ(static_cast<int>(rvecs.size()), 1);

                if (rvecs.size() == 1)
                {
                    bool correctSolution = false;
                    // first solution
                    correctSolution = cvtest::norm(rvecs[0].first, rvec2, NORM_L2) < m_rotationErrorThreshold &&
                                      cvtest::norm(tvecs[0].first, tvec2, NORM_L2) < m_translationErrorThreshold;
                    if (correctSolution)
                    {
                        nbGoodFirstSolutions++;
                    }

                    if (!correctSolution)
                    {
                        // second solution
                        correctSolution = cvtest::norm(rvecs[0].second, rvec2, NORM_L2) < m_rotationErrorThreshold &&
                                          cvtest::norm(tvecs[0].second, tvec2, NORM_L2) < m_translationErrorThreshold;

                        if (correctSolution)
                        {
                            nbGoodSecondSolutions++;
                        }
                    }

                    if (correctSolution)
                    {
                        nbGoodPosesAccuracy++;
                    }

#if DISPLAY_DEBUG
                    drawFrameAxes(imgDisplay, cameraMatrix, noArray(), rvecs[0].first, tvecs[0].first, tagSize_2*m_factor);
#endif
                }

#if DISPLAY_DEBUG
                imshow("Current tag", imgDisplay);
                waitKey(30);
#endif
            }
        }
    }

    double cornersAccuracyPercentage = nbGoodCornersAccuracy / static_cast<double>(nbTests);
    cout << "Corners accuracy percentage: " << cornersAccuracyPercentage*100 << "% for RMSE threshold: " << m_accuracyThreshold
         << " for " << nbGoodCornersAccuracy << " / " << nbTests << " tests." << endl;
    cout << "Max / Mean / Median RMS error: " << getMax(rmse_vec) << " / "
         << getMean(rmse_vec) << " / " << getMedian(rmse_vec) << endl;

    if (cornersAccuracyPercentage < 0.95)
    {
        ts->printf(cvtest::TS::LOG, "Invalid corners accuracy: %0.2f, threshold is 0.95, failed %d tests.\n",
                   cornersAccuracyPercentage, nbTests);
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
    }

    double posesAccuracyPercentage = nbGoodPosesAccuracy / static_cast<double>(nbTests);
    cout << "Poses accuracy percentage: " << posesAccuracyPercentage << " for rvec threshold: " << m_rotationErrorThreshold
         << ", tvec threshold: " << m_translationErrorThreshold << " and "
         << nbGoodPosesAccuracy << " / " << nbTests << " tests." << endl;
    cout << "Good first solutions: " << nbGoodFirstSolutions << " / Good second solutions: " << nbGoodSecondSolutions << endl;

    if (posesAccuracyPercentage < 0.9)
    {
        ts->printf(cvtest::TS::LOG, "Invalid poses accuracy: %0.2f, threshold is 0.9, failed %d tests.\n",
                   posesAccuracyPercentage, nbTests);
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
    }
}

TEST(CV_ApriltagDetectionPerspective, accuracy)
{
    const double rmseThreshold = 1.0; // 1px
    {
        cout << "\n16h5" << endl;
        CV_ApriltagDetectionPerspective test(apriltag::TAG_16h5, 6.0/8.0, rmseThreshold, 30);
        test.safe_run();
    }
    {
        cout << "\n25h9" << endl;
        CV_ApriltagDetectionPerspective test(apriltag::TAG_25h9, 7.0/9.0, rmseThreshold, 35);
        test.safe_run();
    }
    {
        cout << "\n36h11" << endl;
        CV_ApriltagDetectionPerspective test(apriltag::TAG_36h11, 8.0/10.0, rmseThreshold, 587);
        test.safe_run();
    }
    {
        cout << "\nCIRCLE21h7" << endl;
        CV_ApriltagDetectionPerspective test(apriltag::TAG_CIRCLE21h7, 5.0/9.0, rmseThreshold, 38);
        test.safe_run();
    }
    {
#if TEST_BIG_FAMILY
        cout << "\nCIRCLE49h12" << endl;
        CV_ApriltagDetectionPerspective test(apriltag::TAG_CIRCLE49h12, 5.0/11.0, rmseThreshold, 65535);
        test.safe_run();
#endif
    }
    {
#if TEST_BIG_FAMILY
        cout << "\nCUSTOM48h12" << endl;
        CV_ApriltagDetectionPerspective test(apriltag::TAG_CUSTOM48h12, 6.0/10.0, rmseThreshold, 42211);
        test.safe_run();
#endif
    }
    {
        cout << "\nSTANDARD41h12" << endl;
        CV_ApriltagDetectionPerspective test(apriltag::TAG_STANDARD41h12, 5.0/9.0, rmseThreshold, 2115);
        test.safe_run();
    }
    {
#if TEST_BIG_FAMILY
        cout << "\nSTANDARD52h13" << endl;
        CV_ApriltagDetectionPerspective test(apriltag::TAG_STANDARD52h13, 6.0/10.0, rmseThreshold, 48714);
        test.safe_run();
#endif
    }
}

class CV_ApriltagDetectionId : public cvtest::BaseTest
{
public:
    CV_ApriltagDetectionId(const apriltag::AprilTagFamily& tagFamily, int id1, int id2) :
        m_detector(tagFamily), m_groundTruthId1(id1), m_groundTruthId2(id2)
    {
        m_detector.setQuadDecimate(2.0);
    }

protected:
    virtual void run(int);
    int runDetection(int tagId);

    apriltag::AprilTagDetector m_detector;
    int m_groundTruthId1;
    int m_groundTruthId2;
};

void CV_ApriltagDetectionId::run(int)
{
    {
        int id = runDetection(m_groundTruthId1);
        cout << "Decoded: " << id << " ; ground truth: " << m_groundTruthId1 << endl;
        if (id != m_groundTruthId1)
        {
            ts->printf(cvtest::TS::LOG, "Invalid decoded id: %d, should be: %d.\n", id, m_groundTruthId1);
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
    {
        int id = runDetection(m_groundTruthId2);
        cout << "Decoded: " << id << " ; ground truth: " << m_groundTruthId2 << endl;
        if (id != m_groundTruthId2)
        {
            ts->printf(cvtest::TS::LOG, "Invalid decoded id: %d, should be: %d.\n", id, m_groundTruthId2);
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
}

int CV_ApriltagDetectionId::runDetection(int tagId)
{
    Mat tagImg;
    const int tagSizePx = 200;
    const int tagSizePx_2 = tagSizePx/2;
    m_detector.drawTag(tagImg, Size(tagSizePx,tagSizePx), tagId);
    const int width = 640, height = 480;
    const int width_2 = width/2, height_2 = height/2;
    Mat canonicalTagImg = Mat::zeros(height, width, CV_8UC1);
    Mat ref = canonicalTagImg(Rect(width_2 - tagSizePx_2, height_2 - tagSizePx_2, tagSizePx, tagSizePx));
    tagImg.copyTo(ref);

    vector<vector<Point2d> > corners;
    vector<int> ids;
    m_detector.detectTags(canonicalTagImg, corners, ids);

#if DISPLAY_DEBUG
    Mat imgDisplay;
    cvtColor(canonicalTagImg, imgDisplay, COLOR_GRAY2BGR);
    m_detector.drawDetectedTags(imgDisplay, corners);

    imshow("Detection", imgDisplay);
    waitKey(0);
#endif

    int id = -1;
    if (ids.size() == 1)
    {
        id = ids[0];
    }

    return id;
}

TEST(CV_ApriltagDetection, accuracy_id)
{
    {
        cout << "\n16h5" << endl;
        CV_ApriltagDetectionId test(apriltag::TAG_16h5, 0, 29);
        test.safe_run();
    }
    {
        cout << "\n25h9" << endl;
        CV_ApriltagDetectionId test(apriltag::TAG_25h9, 0, 34);
        test.safe_run();
    }
    {
        cout << "\n36h11" << endl;
        CV_ApriltagDetectionId test(apriltag::TAG_36h11, 0, 586);
        test.safe_run();
    }
    {
        cout << "\nCIRCLE21h7" << endl;
        CV_ApriltagDetectionId test(apriltag::TAG_CIRCLE21h7, 0, 37);
        test.safe_run();
    }
    {
#if TEST_BIG_FAMILY
        cout << "\nCIRCLE49h12" << endl;
        CV_ApriltagDetectionId test(apriltag::TAG_CIRCLE49h12, 0, 65534);
        test.safe_run();
#endif
    }
    {
#if TEST_BIG_FAMILY
        cout << "\nCUSTOM48h12" << endl;
        CV_ApriltagDetectionId test(apriltag::TAG_CUSTOM48h12, 0, 42210);
        test.safe_run();
#endif
    }
    {
        cout << "\nSTANDARD41h12" << endl;
        CV_ApriltagDetectionId test(apriltag::TAG_STANDARD41h12, 0, 2114);
        test.safe_run();
    }
    {
#if TEST_BIG_FAMILY
        cout << "\nSTANDARD52h13" << endl;
        CV_ApriltagDetectionId test(apriltag::TAG_STANDARD52h13, 0, 48713);
        test.safe_run();
#endif
    }
}

TEST(CV_ApriltagDetection, corners_type)
{
    Mat tagImg;
    const int tagSizePx = 200;
    const int tagSizePx_2 = tagSizePx/2;
    apriltag::AprilTagDetector detector(apriltag::TAG_36h11);
    detector.drawTag(tagImg, Size(tagSizePx,tagSizePx), 0);
    const int width = 640, height = 480;
    const int width_2 = width/2, height_2 = height/2;
    Mat canonicalTagImg = Mat::zeros(height, width, CV_8UC1);
    Mat ref = canonicalTagImg(Rect(width_2 - tagSizePx_2, height_2 - tagSizePx_2, tagSizePx, tagSizePx));
    tagImg.copyTo(ref);

    vector<Point2d> cornersRef;
    const double innerSize_2 = tagSizePx_2 * 4.0/5.0; // ratio for 36h11
    cornersRef.push_back(Point2d(width_2 - innerSize_2, height_2 + innerSize_2));
    cornersRef.push_back(Point2d(width_2 + innerSize_2, height_2 + innerSize_2));
    cornersRef.push_back(Point2d(width_2 + innerSize_2, height_2 - innerSize_2));
    cornersRef.push_back(Point2d(width_2 - innerSize_2, height_2 - innerSize_2));

    {
        // Point2d
        vector<vector<Point2d> > corners;
        vector<int> ids;
        detector.detectTags(canonicalTagImg, corners, ids);

        ASSERT_EQ(corners.size(), ids.size());
        ASSERT_EQ(static_cast<int>(corners.size()), 1);
        if (corners.size() == 1)
        {
            ASSERT_EQ(corners[0].size(), cornersRef.size());
            if (corners[0].size() == cornersRef.size())
            {
                for (size_t i = 0; i < corners[0].size(); i++)
                {
                    ASSERT_NEAR(cv::norm(corners[0][i] - cornersRef[i]), 0.0, 0.5);
                }
            }
        }
    }

    {
        // Point2f
        vector<vector<Point2f> > corners;
        vector<int> ids;
        detector.detectTags(canonicalTagImg, corners, ids);

        ASSERT_EQ(corners.size(), ids.size());
        ASSERT_EQ(static_cast<int>(corners.size()), 1);
        if (corners.size() == 1)
        {
            ASSERT_EQ(corners[0].size(), cornersRef.size());
            if (corners[0].size() == cornersRef.size())
            {
                for (size_t i = 0; i < corners[0].size(); i++)
                {
                    ASSERT_NEAR(cv::norm(Point2d(corners[0][i].x, corners[0][i].y) - cornersRef[i]), 0.0, 0.5);
                }
            }
        }
    }
}

}} // namespace
