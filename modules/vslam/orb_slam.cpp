#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <stdio.h>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/camera.hpp>

using namespace cv;
#include<vector>


class OrbSLAM
{
public:
    OrbSLAM(Mat calib_mat_, Mat dist_coeffs_);
    OrbSLAM();
    void loadImage(Mat &img)
    {
        frames.push_back(img);
        alignImages(frames.back());
        if(frames.size() > 1)
            updateCovisibilityGraph();
    };
protected:
    void alignImages(Mat& im1);
    std::vector<KeyPoint> gridFeatureDetect(Mat& img, Mat& descriptors, int grid);
    std::vector<KeyPoint> featureDetectInit(Mat& img, Mat& descriptor);
    void getMatcheInit(Mat& descriptor1, Mat& descriptor2, std::vector<DMatch>& matches);
    void getMatcheBF(Mat& descriptor1, Mat& descriptor2, std::vector<DMatch>& matches);
    void getMatcheKNN(Mat& descriptor1, Mat& descriptor2, std::vector<DMatch>& matches);
    double computeScore(Mat &M, std::vector<Point2f>& points1, std::vector<Point2f>& points2,
                        const double T_M, size_t num_iter);
    bool reconstract(std::vector<Point2f>& points1, std::vector<Point2f>& points2,
                     std::vector<Mat>& rotations, std::vector<Mat>& translations);
    std::vector<std::vector<Mat> > getRotationsAndTranslationsFundamental(Mat& F, std::vector<Point2f>& points1,
                                                                          std::vector<Point2f>& points2);
    std::vector<std::vector<Mat> > getRotationsAndTranslationsHomography(Mat& F, std::vector<Point2f>& points1,
                                                                         std::vector<Point2f>& points2);
    bool doMapInitialization(std::vector<Point2f>& points1, std::vector<Point2f>& points2);
    bool doTracking(Mat r_matrix, Mat t_matrix);
    int motionChooseSolution(std::vector<Point2f>& points1, std::vector<Point2f>& points2,
                             std::vector<Mat>& rotations, std::vector<Mat>& translations);
    void performBA_(Mat rotation, Mat translation);
    void setCamera();
    void updateCovisibilityGraph();
    Mat performBA(Mat rotation, Mat translation);
    std::vector<Mat> gray_frames;
    std::vector<Mat> frames;
    std::vector<std::vector<KeyPoint> > keypoints;
    std::vector<Mat> descriptors;
    std::vector<detail::MatchesInfo> features_matches;
    std::vector<detail::ImageFeatures> features_images;

    std::vector<std::vector<size_t> > covisibility_graph;
private:
    std::vector<std::vector<Point3f> > cur_points3d;
    std::vector<std::vector<Point2f> > cur_points2d;
    std::vector<std::vector<Vec3b> >  colors;
    std::vector<Affine3d> cam_pose;
    Mat calib_mat;
    Mat dist_coeffs;
    Matx33d new_calib;
    int grid = 5;
    std::vector<Mat> homographies;
    std::vector<std::vector<uchar> > homographies_mask;
    std::vector<detail::MatchesInfo> pairwise_matches;
    bool match_two_image = false;
};

OrbSLAM::OrbSLAM()
{
    covisibility_graph.resize(covisibility_graph.size() + 1);
    for (int i = 0; i < covisibility_graph.size(); i++)
    {
        covisibility_graph[i].resize(covisibility_graph.size());
    }
    covisibility_graph[covisibility_graph.size() - 1][covisibility_graph.size() - 1] = 0;
    setCamera();
};

OrbSLAM::OrbSLAM(Mat calib_mat_, Mat dist_coeffs_)
{
    calib_mat = calib_mat_;
    dist_coeffs = dist_coeffs_;
    new_calib = Matx33d(calib_mat);
    covisibility_graph.resize(covisibility_graph.size() + 1);
    for (int i = 0; i < covisibility_graph.size(); i++)
    {
        covisibility_graph[i].resize(covisibility_graph.size());
    }
    covisibility_graph[covisibility_graph.size() - 1][covisibility_graph.size() - 1] = 0;
};

//for logi 1080p HD C920
void OrbSLAM::setCamera()
{
    Mat calib_mat_(3, 3, CV_64FC1);
    Mat dist_coeffs_ = Mat::zeros(5, 1, CV_64FC1);
    /*
    calib_mat_.at<double>(0, 0) = 987;
    calib_mat_.at<double>(0, 1) = 0.0;
    calib_mat_.at<double>(0, 2) = 630;
    calib_mat_.at<double>(1, 0) = 0.0;
    calib_mat_.at<double>(1, 1) = 987;
    calib_mat_.at<double>(1, 2) = 357;
    calib_mat_.at<double>(2, 0) = 0.0;
    calib_mat_.at<double>(2, 1) = 0.0;
    calib_mat_.at<double>(2, 2) = 1.0;
    */
    calib_mat_.at<double>(0, 0) = 612.03;
    calib_mat_.at<double>(0, 1) = 0.0;
    calib_mat_.at<double>(0, 2) = 320.15;
    calib_mat_.at<double>(1, 0) = 0.0;
    calib_mat_.at<double>(1, 1) = 661.6614;
    calib_mat_.at<double>(1, 2) = 117.5195;
    calib_mat_.at<double>(2, 0) = 0.0;
    calib_mat_.at<double>(2, 1) = 0.0;
    calib_mat_.at<double>(2, 2) = 1.0;
    calib_mat = calib_mat_;

    dist_coeffs_.at<double>(0, 0) = -0.128224;
    dist_coeffs_.at<double>(1, 0) = 0.023572;
    dist_coeffs_.at<double>(2, 0) = -0.0596;
    dist_coeffs_.at<double>(3, 0) = 0.040301;
    dist_coeffs_.at<double>(4, 0) = 0.0;
    dist_coeffs = dist_coeffs_;
    new_calib = Matx33d(612.03, 0.0, 320.15, 0.0, 661.6614, 117.5195, 0.0, 0.0, 1.0);
}

void OrbSLAM::updateCovisibilityGraph()
{
    covisibility_graph.resize(covisibility_graph.size() + 1);
    for (int i = 0; i < covisibility_graph.size(); i++)
    {
        covisibility_graph[i].resize(covisibility_graph.size());
    }
    covisibility_graph[covisibility_graph.size() - 1][covisibility_graph.size() - 1] = 0;
    colors.resize(colors.size() + 1);
    cur_points2d.resize(cur_points2d.size() + 1);
    std::vector<Point2f> points1, points2;
    std::vector<std::vector<KeyPoint> > tmp_keypoints = keypoints;
    std::vector<Mat> tmp_descriptors = descriptors;
    std::cout << "sizes: " << cur_points2d.size() << " " << colors.size() << std::endl;
    int count_matches = 0;
    for (int k = 0; k < 1; k++)
    {
        int i = covisibility_graph.size() - 2;
        size_t j = covisibility_graph.size() - 1;

        std::cout << "graph ver :" << i << " " << j << std::endl;
        if ((j - i) == 1)
        {
            match_two_image = true;
        }
        else
            match_two_image = false;
        if ((descriptors[i].cols == 0) || (descriptors[j].cols == 0))
            continue;
        std::vector<DMatch> matches;
        if(match_two_image)
            //getMatcheBF(descriptors[i], descriptors[j], matches);
            getMatcheInit(descriptors[i], descriptors[j], matches);
        else
            getMatcheBF(descriptors[i], descriptors[j], matches);

        covisibility_graph[i][j] += matches.size();
        count_matches += matches.size();
        //draw matches
        //Mat imMatches;
        //drawMatches(frames[i], keypoints[i], frames[j], keypoints[j], matches, imMatches);
        //imshow("matches", imMatches);
        //waitKey();

        std::vector<KeyPoint> tmp_keypoints_i;
        std::vector<KeyPoint> tmp_keypoints_j;
        Mat tmp_descriptors_i(Size(0, 32), CV_8U);
        Mat tmp_descriptors_j(Size(0, 32), CV_8U);
        std::vector<size_t> ind_keypoints_i;
        std::vector<size_t> ind_keypoints_j;

        for (size_t n = 0; n < matches.size(); n++)
        {
            points1.push_back(keypoints[i][matches[n].queryIdx].pt);
            points2.push_back(keypoints[j][matches[n].trainIdx].pt);
            colors.back().push_back(viz::Color(frames[j].at<Vec3b>(points2.back())));
            ind_keypoints_i.push_back(matches[n].queryIdx);
            ind_keypoints_j.push_back(matches[n].trainIdx);
        }

        //sort(ind_keypoints_i.begin(), ind_keypoints_i.end());
        //sort(ind_keypoints_j.begin(), ind_keypoints_j.end());
        for (size_t n = 0; n < keypoints[i].size(); n++)
        {
            for (size_t m = 0; m < ind_keypoints_i.size(); m++)
            {
                if (ind_keypoints_i[m] == n)
                {
                    tmp_keypoints_i.push_back(keypoints[i][n]);
                    tmp_descriptors_i.push_back(descriptors[i].row(n));
                    break;
                }
            }
        }
        for (size_t n = 0; n < keypoints[j].size(); n++)
        {
            for (size_t m = 0; m < ind_keypoints_j.size(); m++)
            {
                if (ind_keypoints_j[m] == n)
                {
                    tmp_keypoints_j.push_back(keypoints[j][n]);
                    tmp_descriptors_j.push_back(descriptors[j].row(n));
                    break;
                }
            }
        }
        /*
        descriptors[i].resize(tmp_descriptors_i.rows);
        descriptors[j].resize(tmp_descriptors_j.rows);
        descriptors[i] = tmp_descriptors_i;
        descriptors[j] = tmp_descriptors_j;
        keypoints[i] = tmp_keypoints_i;
        keypoints[j] = tmp_keypoints_j;
        */


    }

    cur_points2d[cur_points2d.size() - 1].insert(cur_points2d[cur_points2d.size() - 1].end(),
                                                 points2.cbegin(),
                                                 points2.cend());
    std::cout << "sizes: " << cur_points2d.size() << " " << colors.size() << std::endl;
    if (points1.size() != 0)
    {
        bool res = doMapInitialization(points1, points2);
        if (!res)
        {

            keypoints = tmp_keypoints;
            descriptors = tmp_descriptors;

        }
    }
    else
    {
        colors.resize(colors.size() - 1);
        cur_points2d.resize(cur_points2d.size() - 1);
    }
}

bool OrbSLAM::doMapInitialization(std::vector<Point2f>& points1, std::vector<Point2f>& points2)
{
    std::cout << points1.size() << " " << points2.size() << std::endl;
    std::vector<uchar> mask;
    Mat h = findHomography(points1, points2, mask, RANSAC, 5.99);
    if (h.empty())
    {
        colors.resize(colors.size() - 1);
        cur_points2d.resize(cur_points2d.size() - 1);
        return false;
    }

    Mat f(3, 3, CV_64FC1);
    f = findFundamentalMat(points1, points2, FM_RANSAC, 3.84);
    if (f.empty())
    {
        colors.resize(colors.size() - 1);
        cur_points2d.resize(cur_points2d.size() - 1);
        return false;
    }
    std::cout << f << std::endl;
    double score_f = computeScore(f, points1, points2, 3.84, 8);
    double score_h = computeScore(h, points1, points2, 5.99, 4);
    double R_h = score_h / (score_h + score_f);

    if (R_h > 0.45)
    {
        std::vector<std::vector<Mat> >sol = getRotationsAndTranslationsHomography(h, points1, points2);
        if (sol.size() != 2)
            return false;
        std::vector<Mat> rotations = sol[0];
        std::vector<Mat> translations = sol[1];
        return reconstract(points1, points2, rotations, translations);
    }
    else

    {
        std::vector<std::vector<Mat> >sol = getRotationsAndTranslationsFundamental(f, points1, points2);
        if (sol.size() != 2)
            return false;
        std::vector<Mat> rotations = sol[0];
        std::vector<Mat> translations = sol[1];
        return reconstract(points1, points2, rotations, translations);
    }
    return true;

}

void OrbSLAM::getMatcheKNN(Mat& descriptor1, Mat& descriptor2, std::vector<DMatch>& matches)
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);
    const float ratio_thresh = 0.6f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i].size() >= 2)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                matches.push_back(knn_matches[i][0]);
            }
        }
    }
}

void OrbSLAM::getMatcheBF(Mat& descriptor1, Mat& descriptor2, std::vector<DMatch>& matches)
{

    float good_match_percent = 0.08f;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptor1, descriptor2, matches, Mat());
    std::sort(matches.begin(), matches.end(),
              [](DMatch a, DMatch b) { return a.distance > b.distance; });
    const int numGoodMatches = matches.size() * good_match_percent;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

}
void OrbSLAM::getMatcheInit(Mat& descriptor1, Mat& descriptor2, std::vector<DMatch>& matches)
{
    Ptr<detail::FeaturesMatcher> matcher;

    matcher = makePtr<detail::BestOf2NearestMatcher>(1, 0.1);

    std::vector<detail::ImageFeatures> features;
    features.push_back(features_images[frames.size() - 2]);
    features.push_back(features_images[frames.size() - 1]);
    features[0].img_idx = 0;
    features[1].img_idx = 1;
    (*matcher)(features, features_matches);
    matcher->collectGarbage();
    matches = features_matches[1].matches;
    std::cout << matches.size();

}

void OrbSLAM::alignImages(Mat& im1)
{
    Mat new_gray_frame;
    cvtColor(im1, new_gray_frame, cv::COLOR_BGR2GRAY);
    gray_frames.push_back(new_gray_frame);
    //keypoints.resize(keypoints.size() + 1);
    descriptors.resize(descriptors.size() + 1);
    //colors.resize(colors.size() + 1)
    cvtColor(im1, gray_frames.back(), cv::COLOR_BGR2GRAY);
    std::vector<KeyPoint> new_keypoints;
    //if (frames.size() > 2)
    //new_keypoints = gridFeatureDetect(gray_frames.back(), descriptors.back(), grid);
    //else
    new_keypoints = featureDetectInit(gray_frames.back(), descriptors.back());
    //new_keypoints = gridFeatureDetect(gray_frames.back(), descriptors.back(), grid);
    std::cout << "keypoints size " << new_keypoints.size() << std::endl;
    keypoints.push_back(new_keypoints);
    std::cout << "keypoints size " << keypoints.size() << std::endl;
}

std::vector<KeyPoint> OrbSLAM::featureDetectInit(Mat& img, Mat& descriptor)
{
    Ptr<Feature2D> finder;
    finder = ORB::create(2000);
    detail::ImageFeatures features;

    computeImageFeatures(finder, img, features);
    features.img_idx = frames.size();
    descriptor = features.descriptors.getMat(ACCESS_RW).clone();
    Mat im;
    std::vector<KeyPoint> keypoints_ = features.keypoints;
    //drawKeypoints(frames.back(), keypoints_, im, Scalar(100, 200, 0));
    //imshow("a", im);
    //waitKey();
    std::cout << keypoints_.size() << std::endl;
    std::cout << descriptor.size() << std::endl;
    features_images.push_back(features);
    return keypoints_;
}
std::vector<KeyPoint> OrbSLAM::gridFeatureDetect(Mat& img, Mat& descriptor, int grid)
{
    std::vector<KeyPoint> keypoints_;
    int step_c = img.cols / grid;

    int step_r = img.rows / grid;

    int max_threshold = 500;
    int step_threshold = 20;
    int max_features = 2000;
    max_features = max_features / grid;
    std::vector<Mat> masks;
    for (int i = 0; i < img.cols; i += step_c)
    {
        for (int j = 0; j < img.rows; j += step_r)
        {
            Mat mask = Mat::zeros(img.size(), CV_8UC1);
            rectangle(mask, Point(i, j), Point(i + step_c, j + step_r), Scalar(255), -1);
            masks.push_back(mask);
        }
    }

    for (size_t i = 0; i < masks.size(); i++)
    {
        //imshow("a", masks[i]);
        //waitKey();

        //orb->setScoreType(ORB::FAST_SCORE);
        for (int threshold = 20; threshold < max_threshold; threshold += step_threshold)
        {
            Ptr<ORB> orb = ORB::create(max_features);
            std::vector<KeyPoint> keypoints_temp;
            Mat descriptors_temp;
            orb->setFastThreshold(threshold);
            //orb->setEdgeThreshold(threshold);
            orb->detectAndCompute(img, masks[i], keypoints_temp, descriptors_temp);
            if (keypoints_temp.size() >= max_features)
            {
                std::copy(begin(keypoints_temp), end(keypoints_temp), std::back_inserter(keypoints_));
                descriptor.push_back(descriptors_temp);
                break;
            }

        }
    }
    Mat im;
    drawKeypoints(frames.back(), keypoints_, im, Scalar(100, 200, 0));
    imshow("a", im);
    waitKey();
    return keypoints_;
}

//S = summ(p_M(d^2(m1, M^(-1) * m2) + p_M(d^2(m2, M * m1))))
//p_M(d^2) = 5.99 - d^2 if d^2 < 5.99
//else p_M(d^2) = 0
double OrbSLAM::computeScore(Mat &M, std::vector<Point2f>& points1, std::vector<Point2f>& points2, const double T_M, size_t num_iter)
{
    Mat M_inv = M.inv();
    Mat m2(3, num_iter, CV_64FC1);
    if (points2.size() < num_iter)
        num_iter = points2.size();
    for (int i = 0; i < num_iter; i++)
    {
        m2.at<double>(0, i) = points2[i].x;
        m2.at<double>(1, i) = points2[i].y;
        m2.at<double>(2, i) = 1;
    }
    Mat M_inv_m2_mat = M_inv * m2;
    std::vector<Point2f> M_inv_m2;
    std::vector<double> dist1;
    for (int i = 0; i < num_iter; i++)
    {
        M_inv_m2.push_back(Point2f(M_inv_m2_mat.at<double>(0, i) / M_inv_m2_mat.at<double>(2, i),
                                   M_inv_m2_mat.at<double>(1, i) / M_inv_m2_mat.at<double>(2, i)));
        dist1.push_back((M_inv_m2[i].x - points1[i].x) * (M_inv_m2[i].x - points1[i].x) +
                        (M_inv_m2[i].y - points1[i].y) * (M_inv_m2[i].y - points1[i].y));
    }
    //TODO: use convertPointsToHomogeneous() and convertPointsFromHomogeneous()
    Mat m1(3, num_iter, CV_64FC1);
    for (int i = 0; i < num_iter; i++)
    {
        m1.at<double>(0, i) = points1[i].x;
        m1.at<double>(1, i) = points1[i].y;
        m1.at<double>(2, i) = 0;
    }
    Mat M_m1_mat = M * m1;
    std::vector<Point2f> M_m1;
    std::vector<double> dist2;
    double S_M = 0;
    for (int i = 0; i < num_iter; i++)
    {
        M_m1.push_back(Point2f(M_m1_mat.at<double>(0, i) / M_m1_mat.at<double>(2, i),
                               M_m1_mat.at<double>(1, i) / M_m1_mat.at<double>(2, i)));
        dist2.push_back((M_m1[i].x - points2[i].x) * (M_m1[i].x - points2[i].x) +
                        (M_m1[i].y - points2[i].y) * (M_m1[i].y - points2[i].y));
    }
    double T_H = 5.99;
    for (int i = 0; i < num_iter; i++)
    {
        if (dist1[i] < T_M)
            S_M += T_H - dist1[i];
        if (dist2[i] < T_M)
            S_M += T_H - dist2[i];
    }
    std::cout << "S_M = " << S_M << std::endl;
    return S_M;
}

int OrbSLAM::motionChooseSolution(std::vector<Point2f>& points1, std::vector<Point2f>& points2,
                                  std::vector<Mat>& rotations, std::vector<Mat>& translations)
{
    std::vector<int> count_solution(rotations.size());
    std::vector<std::vector<Point2f>> points_2d(rotations.size());
    std::vector<std::vector<Point3f>> points_3d(rotations.size());

    Mat projection1, id_mat, z;

    id_mat = Mat::eye(rotations[0].size(), CV_64FC1);
    z = Mat::zeros(translations[0].size(), CV_64FC1);

    cv::sfm::projectionFromKRt(calib_mat, id_mat, z, projection1);
    for (size_t i = 0; i < rotations.size(); i++)
    {
        count_solution[i] = 0;
        Mat projection2;
        cv::sfm::projectionFromKRt(calib_mat, rotations[i], translations[i], projection2);
        std::cout << "rotation " << rotations[i] << std::endl;
        cv::Mat points1Mat(2, points1.size(), CV_64FC1);
        cv::Mat points2Mat(2, points1.size(), CV_64FC1);
        //TODO: use reshape()

        for (int j = 0; j < points1.size(); j++)
        {
            cv::Point2f myPoint1 = points1.at(j);
            cv::Point2f myPoint2 = points2.at(j);
            points1Mat.at<double>(0, j) = myPoint1.x;
            points1Mat.at<double>(1, j) = myPoint1.y;
            points2Mat.at<double>(0, j) = myPoint2.x;
            points2Mat.at<double>(1, j) = myPoint2.y;
        }

        std::vector<Mat> points2d;
        points2d.push_back(points1Mat);
        points2d.push_back(points2Mat);

        std::vector<Mat> projections;
        projections.push_back(projection1);
        projections.push_back(projection2);

        Mat points3d_mat(3, 1, CV_64FC1);
        Mat Rs;
        Mat Ts;
        cv::sfm::triangulatePoints(points2d, projections, points3d_mat);
        std::vector<Affine3d> cam_pose;

        cam_pose.push_back(Affine3d(rotations[i], translations[i]));
        cam_pose.push_back(Affine3d(id_mat, z));
        std::vector<Point3d> all_points;
        for (size_t k = 0; k < points3d_mat.cols; k++)
        {
            all_points.push_back(Point3d(points3d_mat.at<double>(0, k),
                                         points3d_mat.at<double>(1, k),
                                         points3d_mat.at<double>(2, k)));
        }
        /*
        viz::Viz3d window;


        viz::WCloud cloud_wid(all_points, viz::Color::green());
        cloud_wid.setRenderingProperty(cv::viz::POINT_SIZE, 2);
        window.showWidget("cameras_frames_and_lines", viz::WTrajectory(cam_pose, viz::WTrajectory::BOTH, 0.1, viz::Color::red()));
        window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(cam_pose, Matx33d(calib_mat), 0.1, viz::Color::red()));
        //window.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
        window.showWidget("points", cloud_wid);
        window.setViewerPose(cam_pose.back());
        window.spin();
        */
        for (size_t k = 0; k < points3d_mat.cols; k++)
        {
            Mat X(3, 1, CV_64FC1);
            X.at<double>(0, 0) = points3d_mat.at<double>(0, k);
            X.at<double>(1, 0) = points3d_mat.at<double>(1, k);
            X.at<double>(2, 0) = points3d_mat.at<double>(2, k);
            double d1 = sfm::depth(id_mat, z, X);
            double d2 = sfm::depth(rotations[i], translations[i], X);

            // Test if point is front to the two cameras.
            if (d1 > 0 && d2 > 0)
            {
                count_solution[i]++;
                points_2d[i].push_back(points2[k]);
                points_3d[i].push_back(Point3f(X.at<double>(0, 0),
                                               X.at<double>(1, 0),
                                               X.at<double>(2, 0)));
            }
        }
    }
    int ind_max, prev_max;
    if (count_solution[0] > count_solution[1])
    {
        ind_max = 0;
        prev_max = 1;
    }
    else
    {
        ind_max = 1;
        prev_max = 0;
    }
    std::cout << "num: " << count_solution[0] << std::endl;
    std::cout << "num: " << count_solution[1] << std::endl;
    for (size_t i = 2; i < count_solution.size(); i++)
    {
        std::cout << "num: " << count_solution[i] << std::endl;
        if (count_solution[i] >= count_solution[ind_max])
        {
            prev_max = ind_max;
            ind_max = i;
        }
        else if (count_solution[i] >= count_solution[prev_max])
            prev_max = i;

    }
    ind_max = prev_max;
    std::cout << "num: " << count_solution[ind_max] << std::endl;
    if (count_solution[ind_max] < 7)
        return -1;
    //ind_max = 0;
    cur_points2d.back() = points_2d[ind_max];
    cur_points3d.resize(cur_points3d.size() + 1);
    cur_points3d.back() = points_3d[ind_max];
    return ind_max;
}

bool OrbSLAM::reconstract(std::vector<Point2f>& points1, std::vector<Point2f>& points2,
                          std::vector<Mat>& rotations, std::vector<Mat>& translations)
{
    int solution_ind = motionChooseSolution(points1, points2, rotations, translations);


    if (solution_ind == -1)
    {
        cur_points2d.resize(cur_points2d.size() - 1);
        colors.resize(colors.size() - 1);
        return false;
    }

    if (match_two_image)
    {
        Mat r = performBA(rotations[solution_ind], translations[solution_ind]);
        Mat t;

        //translations[solution_ind].at<double>(0, 0) *= -1;
        //translations[solution_ind].at<double>(1, 0) *= -1;
        //translations[solution_ind].at<double>(2, 0) *= -1;
        doTracking(r, translations[solution_ind]);
        //
        //std::cout << new_rotation << std::endl;
        //doTracking(rotations[solution_ind], translations[solution_ind]);
        return true;

    }
    else
        doTracking(rotations[solution_ind], translations[solution_ind]);
    return true;
}

std::vector<std::vector<Mat> > OrbSLAM::getRotationsAndTranslationsHomography(Mat& H, std::vector<Point2f>& points1,
                                                                              std::vector<Point2f>& points2)
{
    std::vector<Mat> rotations, translations, normals;
    int solutions = decomposeHomographyMat(H, calib_mat, rotations, translations, normals);

    std::cout << "solution " << solutions << std::endl;
    if (solutions == 0)
    {
        cur_points2d.resize(cur_points2d.size() - 1);
        colors.resize(colors.size() - 1);
        return std::vector<std::vector<Mat> >();
    }
    std::cout << normals[0] << std::endl;
    std::vector<std::vector<Mat> >rotate_and_transl(2);
    rotate_and_transl[0] = rotations;
    rotate_and_transl[1] = translations;
    return rotate_and_transl;
}

std::vector<std::vector<Mat> > OrbSLAM::getRotationsAndTranslationsFundamental(Mat& F, std::vector<Point2f>& points1,
                                                                               std::vector<Point2f>& points2)
{
    Mat E = (F.t().mul(calib_mat)).mul(F);

    std::vector<Mat> rotations(2);
    Mat translations;

    //sfm::motionFromEssential(E, rotations, translations);
    decomposeEssentialMat(E, rotations[0], rotations[1], translations);
    std::vector<std::vector<Mat> > rotate_and_transl(2);
    rotate_and_transl[0].push_back(rotations[0]);
    rotate_and_transl[0].push_back(rotations[0]);
    rotate_and_transl[0].push_back(rotations[1]);
    rotate_and_transl[0].push_back(rotations[1]);
    Mat neg_translation(3, 1, CV_64FC1);
    neg_translation.at<double>(0, 0) = - translations.at<double>(0, 0);
    neg_translation.at<double>(1, 0) = -translations.at<double>(1, 0);
    neg_translation.at<double>(2, 0) = -translations.at<double>(2, 0);
    rotate_and_transl[1].push_back(translations);
    rotate_and_transl[1].push_back(neg_translation);
    rotate_and_transl[1].push_back(translations);
    rotate_and_transl[1].push_back(neg_translation);
    return rotate_and_transl;

}

Mat OrbSLAM::performBA(Mat rotation, Mat translation)
{
    std::cout << "In BA\n";
    std::cout << rotation << std::endl;
    std::vector<detail::CameraParams> cameras;
    std::vector<Mat> rotations;
    if (cam_pose.size() > 0)
        rotations.push_back(Mat(cam_pose.back().rotation()));
    else
        rotations.push_back(rotation);
    rotations.push_back(rotation);
    for (size_t i = 0; i < 2; i++)
    {
        detail::CameraParams cam;
        rotations[i].convertTo(rotations[i], CV_32F);
        cam.R = rotations[i];
        cam.t = (Mat_<double>(3, 1) << 0, 0, 0);
        //cam.t = translation;
        cam.ppx = calib_mat.at<double>(0, 2);
        cam.ppy = calib_mat.at<double>(1, 2);
        cam.focal = calib_mat.at<double>(0, 0);
        cam.aspect = calib_mat.at<double>(1, 1) / calib_mat.at<double>(0, 0);

        cameras.push_back(cam);
    }

    Ptr<detail::BundleAdjusterBase> adjuster;
    adjuster = makePtr<detail::BundleAdjusterReproj>();
    adjuster->setConfThresh(0.01);
    /*
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    refine_mask(0, 0) = 1;
    refine_mask(0, 1) = 1;
    refine_mask(0, 2) = 1;
    refine_mask(1, 1) = 1;
    refine_mask(1, 2) = 1;
    adjuster->setRefinementMask(refine_mask);
    */
    std::vector<detail::ImageFeatures> features;
    features.push_back(features_images[frames.size() - 2]);
    features.push_back(features_images[frames.size() - 1]);
    features[0].img_idx = 0;
    features[1].img_idx = 1;
    if (!(*adjuster)(features, features_matches, cameras))
        std::cout << "Camera parameters adjusting failed." << std::endl;
    std::cout << calib_mat.at<double>(0, 2) << " "
              << calib_mat.at<double>(1, 2) << " "
              << calib_mat.at<double>(0, 0) << " "
              << calib_mat.at<double>(1, 1) / calib_mat.at<double>(0, 0) << std::endl;
    std::cout << cameras[1].ppx << " "
              << cameras[1].ppy << " "
              << cameras[1].focal << " "
              << cameras[1].aspect << std::endl;

    features_matches.clear();
    if (frames.size() == 3)
    {
        calib_mat.at<double>(0, 2) = cameras[1].ppx;
        calib_mat.at<double>(1, 2) = cameras[1].ppy;
        calib_mat.at<double>(0, 0) = cameras[1].focal;
        calib_mat.at<double>(1, 1) = cameras[1].aspect * cameras[1].focal;
    }

    return cameras[1].R;
}
bool OrbSLAM::doTracking(Mat r_matrix, Mat t_matrix)
{

    std::cout << "in tracking\n";
    r_matrix.convertTo(r_matrix, CV_64F);
    std::cout << r_matrix.type() << std::endl;
    if (!match_two_image)
    {
        Mat rvec(Mat::zeros(3, 1, CV_64FC1));       // output rotation vector
        Mat tvec(Mat::zeros(3, 1, CV_64FC1));       // output translation vector
        Mat _R_matrix(Mat::zeros(3, 1, CV_64FC1)); // rotation matrix
        Mat _t_matrix(Mat::zeros(3, 1, CV_64FC1)); // translation matrix



        int iterationsCount = 500;        // number of Ransac iterations.
        float reprojectionError = 5.0;    // maximum allowed distance to consider it an inlier.
        double confidence = 0.95;          // RANSAC successful confidence.
        bool useExtrinsicGuess = false;

        int flags = 0;
        Mat inliers(Mat::zeros(1, 1, CV_64FC1));
        std::cout << "3d and 2d sizes:" << cur_points3d.back().size() << " "
                  << cur_points2d.back().size() << std::endl;
        size_t cur_ind = cur_points3d.size() - 1;
        try
        {
            solvePnPRansac(cur_points3d[cur_ind], cur_points2d[cur_ind], calib_mat, dist_coeffs, rvec, tvec,
                           useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers);
        }
        catch (cv::Exception& e)
        {
            const char* err_msg = e.what();
            std::cout << "exception caught: " << err_msg << std::endl;
            cur_points2d.resize(cur_points2d.size() - 1);
            cur_points3d.resize(cur_points3d.size() - 1);
            colors.resize(colors.size() - 1);
            return false;
        }
        std::cout << inliers.size() << std::endl;

        Rodrigues(rvec, _R_matrix);                   // converts Rotation Vector to Matrix
        _t_matrix = tvec;                            // set translation matrix

        cam_pose.push_back(Affine3d(_R_matrix, _t_matrix));
    }
    else
        cam_pose.push_back(Affine3d(r_matrix, t_matrix));
    std::cout << "translation : " << cam_pose.back().translation() << std::endl;
    //cam_pose.push_back(Affine3d(r_matrix, t_matrix));
    std::vector<Point3d> all_points;
    std::vector<Vec3b> all_colors;
    for (int i = 0; i < cur_points3d.size(); i++)
    {
        for (int j = 0; j < cur_points3d[i].size(); j++)
        {
            all_points.push_back(cur_points3d[i][j]);
            all_colors.push_back(colors[i][j]);
        }

    }
    std::cout << "start visualization" << std::endl;
    viz::Viz3d window;
    //viz::WCameraPosition cpw(0.25); // Coordinate axes

    //viz::WCameraPosition cpw_frustum(new_calib, 0.3, viz::Color::yellow());
    //window.showWidget("coordinate", viz::WCoordinateSystem());
//    window.showWidget("CPW", cpw, cam_pose);
//used for visualization

    viz::WCloud cloud_wid(all_points, all_colors);
    cloud_wid.setRenderingProperty( cv::viz::POINT_SIZE, 2 );
    window.showWidget("cameras_frames_and_lines", viz::WTrajectory(cam_pose, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
    window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(cam_pose, Matx33d(calib_mat), 0.1, viz::Color::green()));
    //window.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
    window.showWidget("points", cloud_wid);
    window.setViewerPose(cam_pose.back());
    window.spin();
    return true;
}