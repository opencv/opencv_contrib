#include "opencv2/slam/matcher.hpp"

namespace cv {
namespace vo {

Matcher::Matcher(float ratio)
    : ratio_(ratio), bf_(NORM_HAMMING)
{
}

void Matcher::knnMatch(const Mat &desc1, const Mat &desc2, std::vector<DMatch> &goodMatches)
{
    goodMatches.clear();
    if(desc1.empty() || desc2.empty()) return;
    std::vector<std::vector<DMatch>> knn;
    bf_.knnMatch(desc1, desc2, knn, 2);
    for(size_t i=0;i<knn.size();++i){
        if(knn[i].size() < 2) continue;
        const DMatch &m1 = knn[i][0];
        const DMatch &m2 = knn[i][1];
        if(m1.distance < ratio_ * m2.distance) goodMatches.push_back(m1);
    }
}

void Matcher::match(const Mat &desc1, const Mat &desc2,
                    const std::vector<KeyPoint> &kps1, const std::vector<KeyPoint> &kps2,
                    std::vector<DMatch> &goodMatches,
                    int imgCols, int imgRows,
                    int bucketRows, int bucketCols, int topKPerBucket,
                    int maxTotal, bool enableMutual, bool enableBucketing)
{
    goodMatches.clear();
    if(desc1.empty() || desc2.empty()) return;
    // 1) knn match desc1 -> desc2 and apply ratio test
    std::vector<std::vector<DMatch>> knn12;
    bf_.knnMatch(desc1, desc2, knn12, 2);
    const int n1 = static_cast<int>(knn12.size());

    std::vector<int> best12(n1, -1); // best trainIdx for each query if ratio test passed
    for(int i=0;i<n1;++i){
        if(knn12[i].size() < 2) continue;
        const DMatch &m1 = knn12[i][0];
        const DMatch &m2 = knn12[i][1];
        if(m1.distance < ratio_ * m2.distance) best12[i] = m1.trainIdx;
    }

    // 2) optionally perform reverse matching for mutual check
    std::vector<int> best21; // only filled if enableMutual
    int n2 = 0;
    if(enableMutual){
        std::vector<std::vector<DMatch>> knn21;
        bf_.knnMatch(desc2, desc1, knn21, 1);
        n2 = static_cast<int>(knn21.size());
        best21.assign(n2, -1);
        for(int j=0;j<n2;++j){
            if(knn21[j].size() < 1) continue;
            best21[j] = knn21[j][0].trainIdx;
        }
    } else {
        // if no mutual check, set n2 conservatively by descriptor count of desc2
        n2 = desc2.rows > 0 ? desc2.rows : 0;
    }

    // 3) Collect candidate matches according to whether mutual check is enabled
    struct Cand { DMatch m; float x; float y; };
    std::vector<Cand> candidates;
    candidates.reserve(n1);
    for(int i=0;i<n1;++i){
        int j = best12[i];
        if(j < 0) continue;
        if(enableMutual){
            if(j >= n2) continue;
            if(best21[j] != i) continue; // mutual check
        }
        // safe distance value from knn12
        float dist = (knn12[i].size() ? knn12[i][0].distance : 0.0f);
        DMatch dm(i, j, dist);
        // keypoint location on current frame (kps2)
        float x = 0, y = 0;
        if(j >= 0 && j < (int)kps2.size()){ x = kps2[j].pt.x; y = kps2[j].pt.y; }
        candidates.push_back({dm, x, y});
    }

    if(candidates.empty()) return;

    // 4) Spatial bucketing on current-frame locations to promote spatially-distributed matches
    std::vector<DMatch> interimMatches;
    interimMatches.reserve(candidates.size());
    if(enableBucketing){
        int cols = std::max(1, bucketCols);
        int rows = std::max(1, bucketRows);
        float cellW = (imgCols > 0) ? (float)imgCols / cols : 1.0f;
        float cellH = (imgRows > 0) ? (float)imgRows / rows : 1.0f;

        // buckets: for each cell keep topKPerBucket smallest-distance matches
        std::vector<std::vector<Cand>> buckets(cols * rows);
        for(const auto &c: candidates){
            int bx = 0, by = 0;
            if(cellW > 0) bx = std::min(cols - 1, std::max(0, (int)std::floor(c.x / cellW)));
            if(cellH > 0) by = std::min(rows - 1, std::max(0, (int)std::floor(c.y / cellH)));
            int idx = by * cols + bx;
            buckets[idx].push_back(c);
        }

        // sort each bucket and take top K
        for(auto &vec: buckets){
            if(vec.empty()) continue;
            std::sort(vec.begin(), vec.end(), [](const Cand &a, const Cand &b){ return a.m.distance < b.m.distance; });
            int take = std::min((int)vec.size(), topKPerBucket);
            for(int i=0;i<take;++i) interimMatches.push_back(vec[i].m);
        }
    } else {
        // no bucketing: keep all candidates
        for(const auto &c: candidates) interimMatches.push_back(c.m);
    }

    if(interimMatches.empty()) return;

    // 5) apply global cap if requested (keep lowest-distance matches)
    if(maxTotal > 0 && (int)interimMatches.size() > maxTotal){
        std::nth_element(interimMatches.begin(), interimMatches.begin() + maxTotal, interimMatches.end(),
                         [](const DMatch &a, const DMatch &b){ return a.distance < b.distance; });
        interimMatches.resize(maxTotal);
    }

    // final sort by distance
    std::sort(interimMatches.begin(), interimMatches.end(), [](const DMatch &a, const DMatch &b){ return a.distance < b.distance; });
    goodMatches = std::move(interimMatches);
}

} // namespace vo
} // namespace cv