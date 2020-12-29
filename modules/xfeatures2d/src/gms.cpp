// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
/*********************************************************************
* This is the implementation of the paper
*    GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence.
*    JiaWang Bian, Wen-Yan Lin, Yasuyuki Matsushita, Sai-Kit Yeung, Tan Dat Nguyen, Ming-Ming Cheng
*    IEEE CVPR, 2017
*    ProjectPage: http://jwbian.net/gms
*********************************************************************/

#include "precomp.hpp"
#include <algorithm>

using namespace std;

namespace cv
{
namespace xfeatures2d
{
// 8 possible rotation and each one is 3 X 3
const int mRotationPatterns[8][9] = {
    {
        1,2,3,
        4,5,6,
        7,8,9
    },
    {
        4,1,2,
        7,5,3,
        8,9,6
    },
    {
        7,4,1,
        8,5,2,
        9,6,3
    },
    {
        8,7,4,
        9,5,1,
        6,3,2
    },
    {
        9,8,7,
        6,5,4,
        3,2,1
    },
    {
        6,9,8,
        3,5,7,
        2,1,4
    },
    {
        3,6,9,
        2,5,8,
        1,4,7
    },
    {
        2,3,6,
        1,5,9,
        4,7,8
    }
};

// 5 level scales
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / std::sqrt(2.0), std::sqrt(2.0), 2.0 };

class GMSMatcher
{
public:
    // OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches
    GMSMatcher(const vector<KeyPoint>& vkp1, const Size& size1, const vector<KeyPoint>& vkp2, const Size& size2,
               const vector<DMatch>& vDMatches, const double thresholdFactor) : mThresholdFactor(thresholdFactor)
    {
        // Input initialize
        normalizePoints(vkp1, size1, mvP1);
        normalizePoints(vkp2, size2, mvP2);
        mNumberMatches = vDMatches.size();
        convertMatches(vDMatches, mvMatches);

        // Grid initialize
        mGridSizeLeft = Size(20, 20);
        mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

        // Initialize the neighbor of left grid
        mGridNeighborLeft = Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
        initalizeNeighbors(mGridNeighborLeft, mGridSizeLeft);
    }

    ~GMSMatcher() {}

    // Get Inlier Mask
    // Return number of inliers
    int getInlierMask(vector<bool> &vbInliers, const bool withRotation = false, const bool withScale = false);


private:
    // Normalized Points
    vector<Point2f> mvP1, mvP2;

    // Matches
    vector<pair<int, int> > mvMatches;

    // Number of Matches
    size_t mNumberMatches;

    // Grid Size
    Size mGridSizeLeft, mGridSizeRight;
    int mGridNumberLeft;
    int mGridNumberRight;

    // x      : left grid idx
    // y      : right grid idx
    // value  : how many matches from idx_left to idx_right
    Mat mMotionStatistics;

    //
    vector<int> mNumberPointsInPerCellLeft;

    // Inldex  : grid_idx_left
    // Value   : grid_idx_right
    vector<int> mCellPairs;

    // Every Matches has a cell-pair
    // first  : grid_idx_left
    // second : grid_idx_right
    vector<pair<int, int> > mvMatchPairs;

    // Inlier Mask for output
    vector<bool> mvbInlierMask;

    //
    Mat mGridNeighborLeft;
    Mat mGridNeighborRight;

    double mThresholdFactor;


    // Assign Matches to Cell Pairs
    void assignMatchPairs(const int GridType);

    void convertMatches(const vector<DMatch> &vDMatches, vector<pair<int, int> > &vMatches);

    int getGridIndexLeft(const Point2f &pt, const int type);

    int getGridIndexRight(const Point2f &pt);

    vector<int> getNB9(const int idx, const Size& GridSize);

    void initalizeNeighbors(Mat &neighbor, const Size& GridSize);

    void normalizePoints(const vector<KeyPoint> &kp, const Size &size, vector<Point2f> &npts);

    // Run
    int run(const int rotationType);

    void setScale(const int scale);

    // Verify Cell Pairs
    void verifyCellPairs(const int rotationType);
};

void GMSMatcher::assignMatchPairs(const int gridType)
{
    for (size_t i = 0; i < mNumberMatches; i++)
    {
        Point2f &lp = mvP1[mvMatches[i].first];
        Point2f &rp = mvP2[mvMatches[i].second];

        int lgidx = mvMatchPairs[i].first = getGridIndexLeft(lp, gridType);
        int rgidx = -1;

        if (gridType == 1)
        {
            rgidx = mvMatchPairs[i].second = getGridIndexRight(rp);
        }
        else
        {
            rgidx = mvMatchPairs[i].second;
        }

        if (lgidx < 0 || rgidx < 0) continue;

        mMotionStatistics.at<int>(lgidx, rgidx)++;
        mNumberPointsInPerCellLeft[lgidx]++;
    }
}

// Convert OpenCV DMatch to Match (pair<int, int>)
void GMSMatcher::convertMatches(const vector<DMatch> &vDMatches, vector<pair<int, int> > &vMatches)
{
    vMatches.resize(mNumberMatches);
    for (size_t i = 0; i < mNumberMatches; i++)
        vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
}

int GMSMatcher::getGridIndexLeft(const Point2f &pt, const int type)
{
    int x = 0, y = 0;

    if (type == 1) {
        x = cvFloor(pt.x * mGridSizeLeft.width);
        y = cvFloor(pt.y * mGridSizeLeft.height);
    }

    if (type == 2) {
        x = cvFloor(pt.x * mGridSizeLeft.width + 0.5);
        y = cvFloor(pt.y * mGridSizeLeft.height);
    }

    if (type == 3) {
        x = cvFloor(pt.x * mGridSizeLeft.width);
        y = cvFloor(pt.y * mGridSizeLeft.height + 0.5);
    }

    if (type == 4) {
        x = cvFloor(pt.x * mGridSizeLeft.width + 0.5);
        y = cvFloor(pt.y * mGridSizeLeft.height + 0.5);
    }


    if (x >= mGridSizeLeft.width || y >= mGridSizeLeft.height)
        return -1;

    return x + y * mGridSizeLeft.width;
}

int GMSMatcher::getGridIndexRight(const Point2f &pt)
{
    int x = cvFloor(pt.x * mGridSizeRight.width);
    int y = cvFloor(pt.y * mGridSizeRight.height);

    return x + y * mGridSizeRight.width;
}

int GMSMatcher::getInlierMask(vector<bool> &vbInliers, const bool withRotation, const bool withScale)
{
    int max_inlier = 0;

    if (!withScale && !withRotation)
    {
        setScale(0);
        max_inlier = run(1);
        vbInliers = mvbInlierMask;
        return max_inlier;
    }

    if (withRotation && withScale)
    {
        for (int scale = 0; scale < 5; scale++)
        {
            setScale(scale);
            for (int rotationType = 1; rotationType <= 8; rotationType++)
            {
                int num_inlier = run(rotationType);

                if (num_inlier > max_inlier)
                {
                    vbInliers = mvbInlierMask;
                    max_inlier = num_inlier;
                }
            }
        }
        return max_inlier;
    }

    if (withRotation && !withScale)
    {
        setScale(0);
        for (int rotationType = 1; rotationType <= 8; rotationType++)
        {
            int num_inlier = run(rotationType);

            if (num_inlier > max_inlier)
            {
                vbInliers = mvbInlierMask;
                max_inlier = num_inlier;
            }
        }
        return max_inlier;
    }

    if (!withRotation && withScale)
    {
        for (int scale = 0; scale < 5; scale++)
        {
            setScale(scale);
            int num_inlier = run(1);

            if (num_inlier > max_inlier)
            {
                vbInliers = mvbInlierMask;
                max_inlier = num_inlier;
            }

        }
        return max_inlier;
    }

    return max_inlier;
}

// Get Neighbor 9
vector<int> GMSMatcher::getNB9(const int idx, const Size& gridSize)
{
    vector<int> NB9(9, -1);

    int idx_x = idx % gridSize.width;
    int idx_y = idx / gridSize.width;

    for (int yi = -1; yi <= 1; yi++)
    {
        for (int xi = -1; xi <= 1; xi++)
        {
            int idx_xx = idx_x + xi;
            int idx_yy = idx_y + yi;

            if (idx_xx < 0 || idx_xx >= gridSize.width || idx_yy < 0 || idx_yy >= gridSize.height)
                continue;

            NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * gridSize.width;
        }
    }
    return NB9;
}

void GMSMatcher::initalizeNeighbors(Mat &neighbor, const Size& gridSize)
{
    for (int i = 0; i < neighbor.rows; i++)
    {
        vector<int> NB9 = getNB9(i, gridSize);
        int *data = neighbor.ptr<int>(i);
        memcpy(data, &NB9[0], sizeof(int) * 9);
    }
}

// Normalize Key Points to Range(0 - 1)
void GMSMatcher::normalizePoints(const vector<KeyPoint> &kp, const Size &size, vector<Point2f> &npts)
{
    const size_t numP = kp.size();
    const int width   = size.width;
    const int height  = size.height;
    npts.resize(numP);

    for (size_t i = 0; i < numP; i++)
    {
        npts[i].x = kp[i].pt.x / width;
        npts[i].y = kp[i].pt.y / height;
    }
}

int GMSMatcher::run(const int rotationType)
{
    mvbInlierMask.assign(mNumberMatches, false);

    // Initialize Motion Statisctics
    mMotionStatistics = Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);
    mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

    for (int gridType = 1; gridType <= 4; gridType++)
    {
        // initialize
        mMotionStatistics.setTo(0);
        mCellPairs.assign(mGridNumberLeft, -1);
        mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);

        assignMatchPairs(gridType);
        verifyCellPairs(rotationType);

        // Mark inliers
        for (size_t i = 0; i < mNumberMatches; i++)
        {
            if (mvMatchPairs[i].first < 0)
                continue;
                
            if (mvMatchPairs[i].first >= 0 && mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
                mvbInlierMask[i] = true;
        }
    }

    return (int) count(mvbInlierMask.begin(), mvbInlierMask.end(), true); //number of inliers
}

void GMSMatcher::setScale(const int scale)
{
    // Set Scale
    mGridSizeRight.width = cvRound(mGridSizeLeft.width  * mScaleRatios[scale]);
    mGridSizeRight.height = cvRound(mGridSizeLeft.height * mScaleRatios[scale]);
    mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;

    // Initialize the neighbor of right grid
    mGridNeighborRight = Mat::zeros(mGridNumberRight, 9, CV_32SC1);
    initalizeNeighbors(mGridNeighborRight, mGridSizeRight);
}

void GMSMatcher::verifyCellPairs(const int rotationType)
{
    const int *CurrentRP = mRotationPatterns[rotationType - 1];

    for (int i = 0; i < mGridNumberLeft; i++)
    {
        if (sum(mMotionStatistics.row(i))[0] == 0)
        {
            mCellPairs[i] = -1;
            continue;
        }

        int max_number = 0;
        for (int j = 0; j < mGridNumberRight; j++)
        {
            int *value = mMotionStatistics.ptr<int>(i);
            if (value[j] > max_number)
            {
                mCellPairs[i] = j;
                max_number = value[j];
            }
        }

        int idx_grid_rt = mCellPairs[i];

        const int *NB9_lt = mGridNeighborLeft.ptr<int>(i);
        const int *NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt);

        int score = 0;
        double thresh = 0;
        int numpair = 0;

        for (size_t j = 0; j < 9; j++)
        {
            int ll = NB9_lt[j];
            int rr = NB9_rt[CurrentRP[j] - 1];
            if (ll == -1 || rr == -1)
                continue;

            score += mMotionStatistics.at<int>(ll, rr);
            thresh += mNumberPointsInPerCellLeft[ll];
            numpair++;
        }

        thresh = mThresholdFactor * std::sqrt(thresh / numpair);

        if (score < thresh)
            mCellPairs[i] = -2;
    }
}

void matchGMS( const Size& size1, const Size& size2, const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
               const vector<DMatch>& matches1to2, vector<DMatch>& matchesGMS, const bool withRotation, const bool withScale,
               const double thresholdFactor )
{
    GMSMatcher gms(keypoints1, size1, keypoints2, size2, matches1to2, thresholdFactor);
    vector<bool> inlierMask;
    gms.getInlierMask(inlierMask, withRotation, withScale);

    matchesGMS.clear();
    for (size_t i = 0; i < inlierMask.size(); i++) {
        if (inlierMask[i])
            matchesGMS.push_back(matches1to2[i]);
    }
}

} //namespace xfeatures2d
} //namespace cv
