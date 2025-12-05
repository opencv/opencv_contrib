#include "opencv2/slam/localizer.hpp"
#include "opencv2/slam/matcher.hpp"
#include "opencv2/slam/data_loader.hpp"
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace cv {
namespace vo {

Localizer::Localizer(float ratio) : ratio_(ratio) {}

bool Localizer::tryPnP(const MapManager &map, const Mat &desc, const std::vector<KeyPoint> &kps,
                       double fx, double fy, double cx, double cy, int imgW, int imgH,
                       int min_inliers,
                       Mat &R_out, Mat &t_out, int &inliers_out,
                       int frame_id, const Mat *image, const std::string &outDir,
                       int *out_preMatches, int *out_postMatches, double *out_meanReproj) const {
    inliers_out = 0; R_out.release(); t_out.release();
    const auto &mappoints = map.mappoints();
    const auto &keyframes = map.keyframes();
    if(mappoints.empty() || keyframes.empty() || desc.empty()) return false;

    // Use last keyframe as prior
    const KeyFrame &last = keyframes.back();
    Mat lastR = last.R_w, lastT = last.t_w;

    // select visible candidates
    std::vector<int> candidates = map.findVisibleCandidates(lastR, lastT, fx, fy, cx, cy, imgW, imgH);
    if(candidates.empty()) return false;

    // gather descriptors from map (prefer mp.descriptor if available)
    Mat trainDesc;
    std::vector<Point3f> objPts; objPts.reserve(candidates.size());
    std::vector<int> trainMpIds; trainMpIds.reserve(candidates.size());
    for(int idx: candidates){
        const auto &mp = mappoints[idx];
        if(mp.observations.empty()) continue;
        // prefer representative descriptor on MapPoint
        if(!mp.descriptor.empty()){
            trainDesc.push_back(mp.descriptor.row(0));
        } else {
            auto ob = mp.observations.front();
            int kfid = ob.first; int kpidx = ob.second;
            int kfIdx = map.keyframeIndex(kfid);
            if(kfIdx < 0) continue;
            const KeyFrame &kf = keyframes[kfIdx];
            if(kf.desc.empty() || kpidx < 0 || kpidx >= kf.desc.rows) continue;
            trainDesc.push_back(kf.desc.row(kpidx));
        }
        objPts.emplace_back((float)mp.p.x, (float)mp.p.y, (float)mp.p.z);
        trainMpIds.push_back(mp.id);
    }
    if(trainDesc.empty()) return false;

    // Forward and backward knn to enable mutual cross-check
    BFMatcher bf(NORM_HAMMING);
    std::vector<std::vector<DMatch>> knnF, knnB;
    bf.knnMatch(desc, trainDesc, knnF, 2);
    bf.knnMatch(trainDesc, desc, knnB, 1);

    int preMatches = static_cast<int>(knnF.size());
    if(out_preMatches) *out_preMatches = preMatches;

    // Ratio + mutual
    const float RATIO = ratio_;
    std::vector<DMatch> goodMatches;
    goodMatches.reserve(knnF.size());
    for(size_t q=0;q<knnF.size();++q){
        if(knnF[q].empty()) continue;
        const DMatch &m0 = knnF[q][0];
        if(knnF[q].size() >= 2){
            const DMatch &m1 = knnF[q][1];
            if(m0.distance > RATIO * m1.distance) continue;
        }
        int trainIdx = m0.trainIdx;
        // mutual check: ensure best match of trainIdx points back to this query
        if(trainIdx < 0 || trainIdx >= static_cast<int>(knnB.size())) continue;
        if(knnB[trainIdx].empty()) continue;
        int backIdx = knnB[trainIdx][0].trainIdx; // index in desc
        if(backIdx != static_cast<int>(q)) continue;
        // passed ratio and mutual
        goodMatches.push_back(DMatch(static_cast<int>(q), trainIdx, m0.distance));
    }

    if(out_postMatches) *out_postMatches = static_cast<int>(goodMatches.size());

    if(goodMatches.size() < static_cast<size_t>(std::max(6, min_inliers))) return false;

    // build PnP inputs
    std::vector<Point3f> obj; std::vector<Point2f> imgpts; obj.reserve(goodMatches.size()); imgpts.reserve(goodMatches.size());
    std::vector<int> matchedMpIds; matchedMpIds.reserve(goodMatches.size());
    for(const auto &m: goodMatches){
        int q = m.queryIdx; int t = m.trainIdx;
        if(t < 0 || t >= static_cast<int>(objPts.size()) || q < 0 || q >= static_cast<int>(kps.size())) continue;
        obj.push_back(objPts[t]);
        imgpts.push_back(kps[q].pt);
        matchedMpIds.push_back(trainMpIds[t]);
    }

    if(obj.size() < static_cast<size_t>(std::max(6, min_inliers))) return false;

    Mat camera = (Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
    Mat dist = Mat::zeros(4,1,CV_64F);
    std::vector<int> inliersIdx;
    bool ok = solvePnPRansac(obj, imgpts, camera, dist, R_out, t_out, false,
                                 100, 8.0, 0.99, inliersIdx, SOLVEPNP_ITERATIVE);
    if(!ok) return false;
    inliers_out = static_cast<int>(inliersIdx.size());

    // compute mean reprojection error on inliers
    double meanReproj = 0.0;
    if(!inliersIdx.empty()){
        std::vector<Point2f> proj;
        projectPoints(obj, R_out, t_out, camera, dist, proj);
        double sum = 0.0;
        for(int idx: inliersIdx){
            double e = std::hypot(proj[idx].x - imgpts[idx].x, proj[idx].y - imgpts[idx].y);
            sum += e;
        }
        meanReproj = sum / inliersIdx.size();
    }
    if(out_meanReproj) *out_meanReproj = meanReproj;

    // Diagnostics: draw matches and inliers if requested
    if(!outDir.empty() && image){
        try{
            ensureDirectoryExists(outDir);
            Mat vis;
            if(image->channels() == 1) cvtColor(*image, vis, COLOR_GRAY2BGR);
            else vis = image->clone();
            // draw all good matches as small circles; inliers green
            for(size_t i=0;i<goodMatches.size();++i){
                Point2f p = imgpts[i];
                bool isInlier = std::find(inliersIdx.begin(), inliersIdx.end(), static_cast<int>(i)) != inliersIdx.end();
                Scalar col = isInlier ? Scalar(0,255,0) : Scalar(0,0,255);
                circle(vis, p, 3, col, 2, LINE_AA);
            }
            std::ostringstream name; name << outDir << "/pnp_frame_" << frame_id << ".png";
            putText(vis, "pre=" + std::to_string(preMatches) + " post=" + std::to_string(goodMatches.size()) + " inliers=" + std::to_string(inliers_out) + " mean_px=" + std::to_string(meanReproj),
                        Point(10,20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 2);
            imwrite(name.str(), vis);
            // append a small CSV-like log
            std::ofstream logf((outDir + "/pnp_stats.csv"), std::ios::app);
            if(logf){
                logf << frame_id << "," << preMatches << "," << goodMatches.size() << "," << inliers_out << "," << meanReproj << "\n";
                logf.close();
            }
        } catch(...) { /* don't crash on diagnostics */ }
    }

    return inliers_out >= min_inliers;
}

} // namespace vo
} // namespace cv