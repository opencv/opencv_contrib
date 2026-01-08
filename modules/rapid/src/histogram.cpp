// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv
{
namespace rapid
{
static void compute1DCanny(const cv::Mat& src, cv::Mat& dst, uchar threshold)
{
    compute1DSobel(src, dst);

    // step2: compute 1D non-maximum suppression + threshold
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 1; j < dst.cols - 1; j++)
        {
            if (dst.at<uchar>(i, j) <= dst.at<uchar>(i, j - 1) || dst.at<uchar>(i, j) <= dst.at<uchar>(i, j + 1))
                dst.at<uchar>(i, j) = 0;

            // threshold
            if(dst.at<uchar>(i, j) < threshold)
                dst.at<uchar>(i, j) = 0;
        }
    }
}

static void calcHueSatHist(const Mat_<Vec3b>& hsv, Mat_<float>& hist)
{
    for (int i = 0; i < hsv.rows; i++)
    {
        for (int j = 0; j < hsv.cols; j++)
        {
            const Vec3b& c = hsv(i, j);
            // thresholds as in sec. 4.1
            if (c[1] > 25 && c[2] > 50)
            {
                hist(c[0] * hist.rows / 256, c[1] * hist.cols / 256)++;
            }
        }
    }
}

static float sum(const Mat_<float>& hist)
{
    CV_DbgAssert(hist.isContinuous());
    float ret = 0;
    int N = int(hist.total());
    const float* ptr = hist.ptr<float>();
    for (int i = 0; i < N; i++)
        ret += ptr[i];
    return ret;
}

static double bhattacharyyaCoeff(const Mat& a, const Mat& b)
{
    CV_DbgAssert(a.isContinuous() && b.isContinuous());
    int N = int(a.total());
    double ret = 0;
    const float* aptr = a.ptr<float>();
    const float* bptr = b.ptr<float>();
    for (int i = 0; i < N; i++)
        ret += std::sqrt(aptr[i] * bptr[i]);
    return ret;
}

static void findCorrespondenciesOLS(const cv::Mat_<float>& scores, cv::Mat_<int>& cols)
{
    cols.resize(scores.rows);
    for (int i = 0; i < scores.rows; i++)
    {
        int pos = -1;
        for (int j = scores.cols - 1; j >= 0; j--)
        {
            if (scores(i, j) >= 0.35)
            {
                pos = j;
                break;
            }
        }

        cols(i) = pos;
    }
}

static float computeEdgeWeight(const cv::Vec2s& curCandiPoint, const cv::Vec2s& preCandiPoint)
{
    float spatial_dist = (float)cv::norm(curCandiPoint - preCandiPoint, cv::NORM_L2SQR);
    return std::exp(-spatial_dist/1000.0f);
}

static void findCorrespondenciesGOS(Mat& bundleGrad, Mat_<float>& fgScores, Mat_<float>& bgScores,
                             const Mat_<Vec2s>& imgLocations, Mat_<int>& cols)
{
    // combine scores
    Mat_<float> scores;
    exp((fgScores + bgScores)/10.0f, scores);

    Mat_<int> fromLocations(scores.size());
    fromLocations = 0;

    // source node
    bool hasCandidate = false;
    for(int j=0; j<bundleGrad.cols; j++)
    {
        if(bundleGrad.at<uchar>(0, j))
        {
            hasCandidate = true;
            fromLocations(0, j) = j;
        }
    }
    // fall back to using center as candidate
    if(!hasCandidate)
    {
        fromLocations(0, bundleGrad.cols/2) = bundleGrad.cols/2;
    }

    int index_max_location = 0; // index in preceding line for backtracking

    // the other layers
    for(int i=1; i<bundleGrad.rows; i++)
    {
        hasCandidate = false;
        for(int j=0; j<bundleGrad.cols; j++)
        {
            if(bundleGrad.at<uchar>(i, j))
                hasCandidate = true;
        }
        if(!hasCandidate)
        {
            bundleGrad.at<uchar>(i, bundleGrad.cols/2) = 255;
        }

        for(int j=0; j<bundleGrad.cols; j++)
        {
            // search for max combined score
            float max_energy = -INFINITY;
            int location = bundleGrad.cols/2;

            if(bundleGrad.at<uchar>(i, j))
            {
                for(int k=0; k<bundleGrad.cols; k++)
                {
                    if(bundleGrad.at<uchar>(i - 1, k))
                    {
                        float edge_weight = computeEdgeWeight(imgLocations(i, j), imgLocations(i - 1, k));
                        float energy = scores(i, j) + scores(i-1, k) + edge_weight;
                        if(max_energy < energy)
                        {
                            max_energy = energy;
                            location = k;
                        }
                    }
                }

                scores(i, j)  = max_energy;  // update the score
                fromLocations(i, j) = location;
                index_max_location = j;
            }
        }
    }

    cols.resize(scores.rows);

    // backtrack along best path
    for (int i = bundleGrad.rows - 1; i >= 0; i--)
    {
        cols(i) = index_max_location;
        index_max_location = fromLocations(i, index_max_location);
    }
}

struct HistTrackerImpl : public OLSTracker
{
    Mat vtx;
    Mat tris;

    Mat_<float> fgHist;
    Mat_<float> bgHist;
    double tau;
    uchar sobelThresh;

    bool useGOS;

    HistTrackerImpl(InputArray _pts3d, InputArray _tris, int histBins, uchar _sobelThesh, bool _useGOS)
    {
        CV_Assert(_tris.getMat().checkVector(3, CV_32S) > 0);
        CV_Assert(_pts3d.getMat().checkVector(3, CV_32F) > 0);
        vtx = _pts3d.getMat();
        tris = _tris.getMat();

        tau = 0.7; // this is 1 - tau compared to OLS paper
        sobelThresh = _sobelThesh;
        useGOS = _useGOS;

        bgHist.create(histBins, histBins);
    }

    void computeAppearanceScores(const Mat& bundleHSV, const Mat& bundleGrad, Mat_<float>& scores) const
    {
        scores.resize(bundleHSV.rows);
        scores = 0;
        Mat_<float> hist(fgHist.size());

        for (int i = 0; i < bundleHSV.rows; i++)
        {
            int start = 0;
            for (int j = 0; j < bundleHSV.cols; j++)
            {
                if (bundleGrad.at<uchar>(i, j))
                {
                    // compute the histogram between last candidate point to current candidate point
                    // as in eq. (4)
                    hist = 0;
                    calcHueSatHist(bundleHSV({i, i + 1}, {start, j}), hist);
                    hist /= std::max(sum(hist), 1.0f);

                    double s = bhattacharyyaCoeff(fgHist, hist);
                    // handle object clutter as in eq. (5)
                    if(s > tau)
                        s = 1.0 - bhattacharyyaCoeff(bgHist, hist);
                    scores(i, j) = float(s);
                    start = j;
                }
            }
        }
    }

    void computeBackgroundScores(const Mat& bundleHSV, const Mat& bundleGrad, Mat_<float>& scores)
    {
        scores.resize(bundleHSV.rows);
        scores = 0;

        Mat_<float> hist(fgHist.size());

        for (int i = 0; i < bundleHSV.rows; i++)
        {
            int end = bundleHSV.cols - 1;
            for (int j = bundleHSV.cols - 1; j >= 0; j--)
            {
                if (bundleGrad.at<uchar>(i, j))
                {
                    // compute the histogram between last candidate point to current candidate point
                    hist = 0;
                    calcHueSatHist(bundleHSV({i, i + 1}, {j, end}), hist);
                    hist /= std::max(sum(hist), 1.0f);

                    double s = 1 - bhattacharyyaCoeff(fgHist, hist);
                    if (s <= tau)
                        s = bhattacharyyaCoeff(bgHist, hist);

                    scores(i, j) = float(s);
                    end = j;
                }
            }
        }
    }

    void updateFgBgHist(const Mat_<Vec3b>& hsv, const Mat_<int>& cols)
    {
        fgHist = 0;
        bgHist = 0;

        for (int i = 0; i < hsv.rows; i++)
        {
            int col = cols(i) < 0 ? hsv.cols / 2 + 1 : cols(i);
            calcHueSatHist(hsv({i, i + 1}, {0, col}), fgHist);
            calcHueSatHist(hsv({i, i + 1}, {col + 1, hsv.cols}), bgHist);
        }

        fgHist /= sum(fgHist);
        bgHist /= sum(bgHist);
    }

    float compute(InputArray img, int num, int len, InputArray K, InputOutputArray rvec,
                  InputOutputArray tvec, const TermCriteria& termcrit) CV_OVERRIDE
    {
        CV_Assert(num >= 3);
        Mat pts2d, pts3d;

        float ret = 0;

        int niter = std::max(1, termcrit.maxCount);
        for(int i = 0; i < niter; i++)
        {
            extractControlPoints(num, len, vtx, rvec, tvec, K, img.size(), tris, pts2d, pts3d);
            if (pts2d.empty())
                return 0;

            Mat lineBundle, imgLoc;
            extractLineBundle(len, pts2d, img, lineBundle, imgLoc);

            Mat bundleHSV;
            cvtColor(lineBundle, bundleHSV, COLOR_BGR2HSV_FULL);

            Mat_<int> cols(num, 1);
            if(fgHist.empty())
            {
                cols = len + 1;

                fgHist.create(bgHist.size());
                updateFgBgHist(bundleHSV, cols);
            }

            Mat bundleGrad;
            compute1DCanny(lineBundle, bundleGrad, sobelThresh);

            Mat_<float> scores(lineBundle.size());
            computeAppearanceScores(bundleHSV, bundleGrad, scores);

            if(useGOS)
            {
                Mat_<float> bgScores(scores.size());
                computeBackgroundScores(bundleHSV, bundleGrad, bgScores);
                findCorrespondenciesGOS(bundleGrad, scores, bgScores, imgLoc, cols);
            }
            else
            {
                findCorrespondenciesOLS(scores, cols);
            }

            convertCorrespondencies(cols, imgLoc, pts2d, pts3d, cols > -1);

            if (pts2d.rows < 3)
                return 0;

            solvePnPRefineLM(pts3d, pts2d, K, cv::noArray(), rvec, tvec);

            updateFgBgHist(bundleHSV, cols);

            ret = float(pts2d.rows) / num;

            if(termcrit.type & TermCriteria::EPS)
            {
                Mat tmp;
                cols.copyTo(tmp, cols > 0);
                tmp -= len + 1;
                double rmsd = std::sqrt(norm(tmp, NORM_L2SQR) / tmp.rows);
                if(rmsd < termcrit.epsilon)
                    break;
            }
        }

        return ret;
    }

    void clearState() CV_OVERRIDE
    {
        fgHist.release();
    }
};

Ptr<OLSTracker> OLSTracker::create(InputArray pts3d, InputArray tris, int histBins, uchar sobelThesh)
{
    return makePtr<HistTrackerImpl>(pts3d, tris, histBins, sobelThesh, false);
}

Ptr<OLSTracker> GOSTracker::create(InputArray pts3d, InputArray tris, int histBins, uchar sobelThesh)
{
    return makePtr<HistTrackerImpl>(pts3d, tris, histBins, sobelThesh, true);
}

} // namespace rapid
} // namespace cv
