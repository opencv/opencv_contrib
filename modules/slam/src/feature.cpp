#include "opencv2/slam/feature.hpp"
#include <limits>
#include <cmath>
#include <opencv2/video/tracking.hpp>
#include <map>

namespace cv {
namespace vo {

FeatureExtractor::FeatureExtractor(int nfeatures)
    : nfeatures_(nfeatures)
{
    orb_ = ORB::create(nfeatures_);
}

// Adaptive Non-Maximal Suppression (ANMS)
static void adaptiveNonMaximalSuppression(const std::vector<KeyPoint> &in, std::vector<KeyPoint> &out, int maxFeatures)
{
    out.clear();
    if(in.empty()) return;
    int N = (int)in.size();
    if(maxFeatures <= 0 || N <= maxFeatures){ out = in; return; }

    // For each keypoint, find distance to the nearest keypoint with strictly higher response
    std::vector<float> radius(N, std::numeric_limits<float>::infinity());
    cv::parallel_for_(Range(0, N), [&](const Range &r){
        for(int i=r.start;i<r.end;++i){
            for(int j=0;j<N;++j){
                if(in[j].response > in[i].response){
                    float dx = in[i].pt.x - in[j].pt.x;
                    float dy = in[i].pt.y - in[j].pt.y;
                    float d2 = dx*dx + dy*dy;
                    if(d2 < radius[i]) radius[i] = d2;
                }
            }
            // if no stronger keypoint exists, radius[i] stays INF
        }
    });

    // Now pick top maxFeatures by radius (larger radius preferred). If radius==INF, treat as large.
    std::vector<int> idx(N);
    for(int i=0;i<N;++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        float ra = radius[a]; float rb = radius[b];
        if(std::isinf(ra) && std::isinf(rb)) return in[a].response > in[b].response; // tie-break by response
        if(std::isinf(ra)) return true;
        if(std::isinf(rb)) return false;
        if(ra == rb) return in[a].response > in[b].response;
        return ra > rb;
    });

    int take = std::min(maxFeatures, N);
    out.reserve(take);
    for(int i=0;i<take;++i) out.push_back(in[idx[i]]);
}

// Unified detectAndCompute: uses flow-aware allocation when prevGray/prevKp provided,
// otherwise falls back to ANMS selection.
void FeatureExtractor::detectAndCompute(const Mat &image, std::vector<KeyPoint> &kps, Mat &desc,
                                       const Mat &prevGray, const std::vector<KeyPoint> &prevKp,
                                       double flow_lambda)
{
    kps.clear(); desc.release();
    if(image.empty()) return;

    // detect candidates with ORB
    std::vector<KeyPoint> candidates;
    orb_->detect(image, candidates);
    if(candidates.empty()) return;

    // If no previous-frame info is provided, use simple ANMS + descriptor computation
    if(prevGray.empty() || prevKp.empty()){
        std::vector<KeyPoint> selected;
        adaptiveNonMaximalSuppression(candidates, selected, nfeatures_);
        if(selected.empty()) return;
        orb_->compute(image, selected, desc);
        kps = std::move(selected);
        return;
    }

    // Flow-aware scoring path -------------------------------------------------
    // 1) track previous keypoints into current frame to estimate flows
    std::vector<Point2f> trackedPts;
    std::vector<unsigned char> status;
    std::vector<float> err;
    std::vector<double> trackedFlows; // aligned with prevKp
    std::vector<Point2f> prevPts; prevPts.reserve(prevKp.size());
    for(const auto &kp: prevKp) prevPts.push_back(kp.pt);
    trackedPts.resize(prevPts.size()); status.resize(prevPts.size()); err.resize(prevPts.size());
    try{
        calcOpticalFlowPyrLK(prevGray, image, prevPts, trackedPts, status, err, Size(21,21), 3,
                                 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 0, 1e-4);
    } catch(const cv::Exception &e) {
        CV_Error(cv::Error::StsError, std::string("calcOpticalFlowPyrLK failed: ") + e.what());
    } catch(const std::exception &e) {
        CV_Error(cv::Error::StsError, std::string("calcOpticalFlowPyrLK failed: ") + e.what());
    } catch(...) {
        CV_Error(cv::Error::StsError, "calcOpticalFlowPyrLK failed with unknown error");
    }
    trackedFlows.resize(prevPts.size());
    for(size_t i=0;i<prevPts.size();++i){
        if(status.size() == prevPts.size() && status[i]){
            double dx = trackedPts[i].x - prevPts[i].x;
            double dy = trackedPts[i].y - prevPts[i].y;
            trackedFlows[i] = std::sqrt(dx*dx + dy*dy);
        } else trackedFlows[i] = 0.0;
    }

    // 2) assign a flow value to each candidate: find nearest tracked point within radius
    const double FLOW_NEIGHBOR_RADIUS = 8.0; // px
    double diag = std::sqrt(double(image.cols)*double(image.cols) + double(image.rows)*double(image.rows));
    struct CandScore { double score; double flow; int idx; };
    std::vector<CandScore> scored; scored.reserve(candidates.size());
    for(size_t i=0;i<candidates.size();++i){
        double flow = 0.0;
        if(!trackedFlows.empty()){
            double bestd = FLOW_NEIGHBOR_RADIUS*FLOW_NEIGHBOR_RADIUS; int besti = -1;
            for(size_t j=0;j<trackedFlows.size();++j){
                if(trackedFlows[j] <= 0.0) continue;
                double dx = candidates[i].pt.x - trackedPts[j].x;
                double dy = candidates[i].pt.y - trackedPts[j].y;
                double d2 = dx*dx + dy*dy;
                if(d2 < bestd){ bestd = d2; besti = (int)j; }
            }
            if(besti >= 0) flow = trackedFlows[besti];
        }
        double resp = candidates[i].response;
        double norm_flow = diag > 0.0 ? (flow / diag) : flow;
        double score = resp * (1.0 + flow_lambda * norm_flow);
        scored.push_back({score, flow, (int)i});
    }

    // 3) grid allocation (ORB-style): split into grid and take top per-cell
    const int GRID_ROWS = 8;
    const int GRID_COLS = 8;
    int cellW = std::max(1, image.cols / GRID_COLS);
    int cellH = std::max(1, image.rows / GRID_ROWS);
    int cellCap = (nfeatures_ + GRID_ROWS*GRID_COLS - 1) / (GRID_ROWS*GRID_COLS);
    std::vector<std::vector<CandScore>> buckets(GRID_ROWS * GRID_COLS);
    for(const auto &c: scored){
        const KeyPoint &kp = candidates[c.idx];
        int cx = std::min(GRID_COLS-1, std::max(0, int(kp.pt.x) / cellW));
        int cy = std::min(GRID_ROWS-1, std::max(0, int(kp.pt.y) / cellH));
        buckets[cy*GRID_COLS + cx].push_back(c);
    }

    std::vector<KeyPoint> selected; selected.reserve(nfeatures_);
    for(auto &bucket: buckets){
        if(bucket.empty()) continue;
        std::sort(bucket.begin(), bucket.end(), [](const CandScore &a, const CandScore &b){ return a.score > b.score; });
        int take = std::min((int)bucket.size(), cellCap);
        for(int i=0;i<take && (int)selected.size() < nfeatures_; ++i){
            selected.push_back(candidates[bucket[i].idx]);
        }
    }

    // if we still have space, fill from all candidates by global score
    if((int)selected.size() < nfeatures_){
        std::vector<CandScore> all = scored;
        std::sort(all.begin(), all.end(), [](const CandScore &a, const CandScore &b){ return a.score > b.score; });
        for(const auto &c: all){
            if((int)selected.size() >= nfeatures_) break;
            bool dup = false;
            for(const auto &s: selected){ if(norm(s.pt - candidates[c.idx].pt) < 1.0){ dup = true; break; } }
            if(!dup) selected.push_back(candidates[c.idx]);
        }
    }

    if(selected.empty()) return;
    orb_->compute(image, selected, desc);
    kps = std::move(selected);
}

} // namespace vo
} // namespace cv