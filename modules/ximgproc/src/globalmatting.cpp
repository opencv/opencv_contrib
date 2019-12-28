// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"

namespace cv { namespace ximgproc {

using namespace std;

template <typename T>
inline T sqr(T a)
{
    return a * a;
}

struct IntensityComp
{
    IntensityComp(const Mat_<Vec3b> &img_temp) : img(img_temp)
    {
        // nothing
    }

    bool operator()(const Point &p0, const Point &p1) const
    {
        const Vec3b &c0 = img(p0.y, p0.x);
        const Vec3b &c1 = img(p1.y, p1.x);

        return ((int)c0[0] + (int)c0[1] + (int)c0[2]) < ((int)c1[0] + (int)c1[1] + (int)c1[2]);
    }

    const Mat_<Vec3b> &img;
};



class GlobalMattingImpl final : public GlobalMatting
{
private:
    vector<Point> findBoundaryPixels(const Mat_<uchar> &trimap, int a, int b);

    // Eq. 2
    float calculateAlpha(const Vec3b &F, const Vec3b &B, const Vec3b &I);

    // Eq. 3
    float colorCost(const Vec3b &F, const Vec3b &B, const Vec3b &I, float alpha);

    // Eq. 4
    float distCost(const Point &p0, const Point &p1, float minDist);

    float colorDist(const Vec3b &I0, const Vec3b &I1);
    float nearestDistance(const vector<Point> &boundary, const Point &p);


    void expansionOfKnownRegions(const Mat_<Vec3b>& img, Mat_<uchar> trimap, int niter);

    void expansionOfKnownRegions(const Mat_<Vec3b> &image,
                                 Mat_<uchar> &trimap,
                                 int r, float c);

    // erode foreground and background regions to increase the size of unknown region
    void erodeFB(Mat_<uchar> &trimap, int r);



    struct Sample
    {
        int fi, bj;
        float df, db;
        float cost, alpha;
    };

    void calculateAlphaPatchMatch(const Mat_<Vec3b> &image,
                                  const Mat_<uchar> &trimap,
                                  const vector<Point> &foregroundBoundary,
                                  const vector<Point> &backgroundBoundary,
                                  vector<vector<Sample> > &samples);

    void expansionOfKnownRegionsHelper(const Mat_<Vec3b> &image,
                                       Mat_<uchar> &trimap,
                                       int r, float c);


    void globalMattingHelper(const Mat_<Vec3b>& image, const Mat_<uchar>& trimap, Mat& foreground, Mat& alpha, Mat& conf);
public:
    GlobalMattingImpl() {}
    ~GlobalMattingImpl() override {}

    void globalMatting(InputArray image, InputArray trimap, OutputArray foreground, OutputArray alpha, OutputArray conf) override;

    void getMat(InputArray image, InputArray trimap, OutputArray foreground, OutputArray alpha, int niter=9) override;
};



vector<Point> GlobalMattingImpl::findBoundaryPixels(const Mat_<uchar> &trimap, int a, int b)
{
    vector<Point> result;

    for (int x = 1; x < trimap.cols - 1; ++x)
    for (int y = 1; y < trimap.rows - 1; ++y)
    {
        if (trimap(y, x) == a)
        {
            if (trimap(y - 1, x) == b ||
                trimap(y + 1, x) == b ||
                trimap(y, x - 1) == b ||
                trimap(y, x + 1) == b)
            {
                result.push_back(Point(x, y));
            }
        }
    }

    return result;
}

// Eq. 2
float GlobalMattingImpl::calculateAlpha(const Vec3b &F, const Vec3b &B, const Vec3b &I)
{
    float result = 0;
    float div = 1e-6f;
    for (int c = 0; c < 3; ++c)
    {
        float f = F[c];
        float b = B[c];
        float i = I[c];

        result += (i - b) * (f - b);
        div += (f - b) * (f - b);
    }

    return min(max(result / div, 0.f), 1.f);
}

// Eq. 3
float GlobalMattingImpl::colorCost(const Vec3b &F, const Vec3b &B, const Vec3b &I, float alpha)
{
    float result = 0;
    for (int c = 0; c < 3; ++c)
    {
        float f = F[c];
        float b = B[c];
        float i = I[c];

        result += sqr(i - (alpha * f + (1 - alpha) * b));
    }

    return sqrt(result);
}

// Eq. 4
float GlobalMattingImpl::distCost(const Point &p0, const Point &p1, float minDist)
{
    int dist = normL2Sqr<int>(p0 - p1);
    return sqrt((float)dist) / minDist;
}

float GlobalMattingImpl::colorDist(const Vec3b &I0, const Vec3b &I1)
{
    int result = 0;

    for (int c = 0; c < 3; ++c)
        result += sqr((int)I0[c] - (int)I1[c]);

    return sqrt((float)result);
}

float GlobalMattingImpl::nearestDistance(const vector<Point> &boundary, const Point &p)
{
    int minDist2 = INT_MAX;
    for (size_t i = 0; i < boundary.size(); ++i)
    {
        int dist2 = sqr(boundary[i].x - p.x)  + sqr(boundary[i].y - p.y);
        minDist2 = min(minDist2, dist2);
    }

    return sqrt((float)minDist2);
}

void GlobalMattingImpl::expansionOfKnownRegions(const Mat_<Vec3b>& img, Mat_<uchar> trimap, int niter)
{
    for (int i = 0; i < niter; ++i)
	expansionOfKnownRegionsHelper(img, trimap, i + 1, float(niter - i));
    erodeFB(trimap, 2);
}

void GlobalMattingImpl::expansionOfKnownRegions(
        const Mat_<Vec3b> &image,
        Mat_<uchar> &trimap,
        int r, float c)
{
    int w = image.cols;
    int h = image.rows;

    for (int y = 0; y< w; ++y)
    for (int x = 0; x< h; ++x)
    {
        if (trimap(y, x) != 128)
            continue;

        const Vec3b &I = image(y, x);

        for (int j = y-r; j <= y+r; ++j)
        {
            for (int i = x-r; i <= x+r; ++i)
            {
                if (i < 0 || i >= w || j < 0 || j >= h)
                    continue;

                if (trimap(j, i) != 0 && trimap(j, i) != 255)
                    continue;

                const Vec3b &I2 = image(j, i);

                float pd = sqrt((float)(sqr(x - i) + sqr(y - j)));
                float cd = colorDist(I, I2);

                if (pd <= r && cd <= c)
                {
                    if (trimap(j, i) == 0)
                        trimap(y, x) = 1;
                    else if (trimap(j, i) == 255)
                        trimap(y, x) = 254;
                }
            }
        }
    }

    for (int x = 0; x < trimap.cols; ++x)
    {
        for (int y = 0; y < trimap.rows; ++y)
        {
            if (trimap(y, x) == 1)
                trimap(y, x) = 0;
            else if (trimap(y, x) == 254)
                trimap(y, x) = 255;
        }
    }
}

// erode foreground and background regions to increase the size of unknown region
void GlobalMattingImpl::erodeFB(Mat_<uchar> &trimap, int r)
{
    int w = trimap.cols;
    int h = trimap.rows;

    Mat_<uchar> foreground(trimap.size(), (uchar)0);
    Mat_<uchar> background(trimap.size(), (uchar)0);

    for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
    {
        if (trimap(y, x) == 0)
            background(y, x) = 1;
        else if (trimap(y, x) == 255)
            foreground(y, x) = 1;
    }


    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(r, r));

    // FIXIT "Inplace" filtering call is ineffective in general (involves input data copying)
    erode(background, background, kernel);
    erode(foreground, foreground, kernel);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            if (background(y, x) == 0 && foreground(y, x) == 0)
                trimap(y, x) = 128;
        }
    }
}


void GlobalMattingImpl::calculateAlphaPatchMatch(const Mat_<Vec3b> &image,
    const Mat_<uchar> &trimap,
    const vector<Point> &foregroundBoundary,
    const vector<Point> &backgroundBoundary,
    vector<vector<Sample> > &samples)
{
    int w = image.cols;
    int h = image.rows;

    samples.resize(h, vector<Sample>(w));

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            if (trimap(y, x) == 128)
            {
                Point p(x, y);

                samples[y][x].fi = rand() % foregroundBoundary.size();
                samples[y][x].bj = rand() % backgroundBoundary.size();
                samples[y][x].df = nearestDistance(foregroundBoundary, p);
                samples[y][x].db = nearestDistance(backgroundBoundary, p);
                samples[y][x].cost = FLT_MAX;
            }
        }
    }

    vector<Point> coords(w * h);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            coords[x + y * w] = Point(x, y);
        }
    }

    for (int iter = 0; iter < 10; ++iter)
    {
        // propagation
        //random_shuffle(coords.begin(), coords.end());
        randShuffle(coords);

        for (size_t i = 0; i < coords.size(); ++i)
        {
            const Point &p = coords[i];

            int x = p.x;
            int y = p.y;

            if (trimap(y, x) != 128)
                continue;

            const Vec3b &I = image(y, x);

            Sample &s = samples[y][x];

            for (int y2 = y - 1; y2 <= y + 1; ++y2)
            {
                for (int x2 = x - 1; x2 <= x + 1; ++x2)
                {
                    if (x2 < 0 || x2 >= w || y2 < 0 || y2 >= h)
                        continue;

                    if (trimap(y2, x2) != 128)
                        continue;

                    Sample &s2 = samples[y2][x2];

                    const Point &fp = foregroundBoundary[s2.fi];
                    const Point &bp = backgroundBoundary[s2.bj];

                    const Vec3b F = image(fp.y, fp.x);
                    const Vec3b B = image(bp.y, bp.x);

                    float alpha = calculateAlpha(F, B, I);

                    float cost = colorCost(F, B, I, alpha) + distCost(p, fp, s.df) + distCost(p, bp, s.db);

                    if (cost < s.cost)
                    {
                        s.fi = s2.fi;
                        s.bj = s2.bj;
                        s.cost = cost;
                        s.alpha = alpha;
                    }
                }
            }
        }

        // random walk
        int w2 = (int)max(foregroundBoundary.size(), backgroundBoundary.size());

        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                if (trimap(y, x) != 128)
                    continue;

                Point p(x, y);

                const Vec3b &I = image(y, x);

                Sample &s = samples[y][x];

                for (int k = 0; ; k++)
                {
                    float r = w2 * pow(0.5f, k);

                    if (r < 1)
                        break;

                    int di = int(r * (rand() / (RAND_MAX + 1.f)));
                    int dj = int(r * (rand() / (RAND_MAX + 1.f)));

                    int fi = s.fi + di;
                    int bj = s.bj + dj;

                    if (fi < 0 || (unsigned)fi >= foregroundBoundary.size() || bj < 0 || (unsigned)bj >= backgroundBoundary.size())
                        continue;

                    const Point &fp = foregroundBoundary[fi];
                    const Point &bp = backgroundBoundary[bj];

                    const Vec3b F = image(fp.y, fp.x);
                    const Vec3b B = image(bp.y, bp.x);

                    float alpha = calculateAlpha(F, B, I);

                    float cost = colorCost(F, B, I, alpha) + distCost(p, fp, s.df) + distCost(p, bp, s.db);

                    if (cost < s.cost)
                    {
                        s.fi = fi;
                        s.bj = bj;
                        s.cost = cost;
                        s.alpha = alpha;
                    }
                }
            }
        }
    }  // iteration
}

void GlobalMattingImpl::expansionOfKnownRegionsHelper(
        const Mat_<Vec3b> &image,
        Mat_<uchar> &trimap,
        int r, float c)
{
    int w = image.cols;
    int h = image.rows;

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            if (trimap(y, x) != 128)
                continue;

            const Vec3b &I = image(y, x);

            for (int j = y-r; j <= y+r; ++j)
            {
                for (int i = x-r; i <= x+r; ++i)
                {
                    if (i < 0 || i >= w || j < 0 || j >= h)
                        continue;

                    if (trimap(j, i) != 0 && trimap(j, i) != 255)
                        continue;

                    const Vec3b &I2 = image(j, i);

                    float pd = sqrt((float)(sqr(x - i) + sqr(y - j)));  // FIXIT sqrt is not needed, compare with r*r instead.
                    float cd = colorDist(I, I2);

                    if (pd <= r && cd <= c)
                    {
                        if (trimap(j, i) == 0)
                            trimap(y, x) = 1;
                        else if (trimap(j, i) == 255)
                            trimap(y, x) = 254;
                    }
                }
            }
        }
    }

    for (int y = 0; y < trimap.rows; ++y)
    {
        for (int x = 0; x < trimap.cols; ++x)
        {
            if (trimap(y, x) == 1)
                trimap(y, x) = 0;
            else if (trimap(y, x) == 254)
                trimap(y, x) = 255;
        }
    }
}

void GlobalMattingImpl::globalMattingHelper(
        const Mat_<Vec3b> &image, const Mat_<uchar>& trimap,
        Mat &_foreground, Mat &_alpha, Mat &_conf
)
{
    vector<Point> foregroundBoundary = findBoundaryPixels(trimap, 255, 128);
    vector<Point> backgroundBoundary = findBoundaryPixels(trimap, 0, 128);

    int n = (int)(foregroundBoundary.size() + backgroundBoundary.size());
    for (int i = 0; i < n; ++i)
    {
        int x = rand() % trimap.cols;
        int y = rand() % trimap.rows;

        if (trimap(y, x) == 0)
            backgroundBoundary.push_back(Point(x, y));
        else if (trimap(y, x) == 255)
            foregroundBoundary.push_back(Point(x, y));
    }

    sort(foregroundBoundary.begin(), foregroundBoundary.end(), IntensityComp(image));
    sort(backgroundBoundary.begin(), backgroundBoundary.end(), IntensityComp(image));

    vector<vector<Sample> > samples;
    calculateAlphaPatchMatch(image, trimap, foregroundBoundary, backgroundBoundary, samples);

    _foreground.create(image.size(), CV_8UC3);
    _alpha.create(image.size(), CV_8UC1);
    _conf.create(image.size(), CV_8UC1);

    Mat_<Vec3b> &foreground = (Mat_<Vec3b>&)_foreground;
    Mat_<uchar> &alpha = (Mat_<uchar>&)_alpha;
    Mat_<uchar> &conf = (Mat_<uchar>&)_conf;

    for (int y = 0; y < alpha.rows; ++y)
    {
        for (int x = 0; x < alpha.cols; ++x)
        {
            switch (trimap(y, x))
            {
                case 0:
                    alpha(y, x) = 0;
                    conf(y, x) = 255;
                    foreground(y, x) = 0;
                    break;
                case 128:
                {
                    alpha(y, x) = uchar(255 * samples[y][x].alpha);
                    conf(y, x) = uchar(255 * exp(-samples[y][x].cost / 6));
                    Point p = foregroundBoundary[samples[y][x].fi];
                    foreground(y, x) = image(p.y, p.x);
                    break;
                }
                case 255:
                    alpha(y, x) = 255;
                    conf(y, x) = 255;
                    foreground(y, x) = image(y, x);
                    break;
            }
        }
    }
}

void GlobalMattingImpl::globalMatting(InputArray _image, InputArray _trimap, OutputArray _foreground, OutputArray _alpha, OutputArray _conf)
{
    Mat image = _image.getMat();
    Mat trimap = _trimap.getMat();

    if (image.empty())
        CV_Error(CV_StsBadArg, "image is empty");

    CV_CheckTypeEQ(image.type(), CV_8UC3, "image mush have CV_8UC3 type");

    if (trimap.empty())
        CV_Error(CV_StsBadArg, "trimap is empty");

    CV_CheckTypeEQ(trimap.type(), CV_8UC1, "trimap mush have CV_8UC1 type");

    if (!_image.sameSize(_trimap))
        CV_Error(CV_StsBadArg, "image and trimap mush have same size");

    const Mat_<Vec3b> &image_ = (const Mat_<Vec3b>&)image;
    const Mat_<uchar> &trimap_ = (const Mat_<uchar>&)trimap;

    // FIXIT unsafe code, strong checks are required to call .getMatRef()
    Mat &foreground = _foreground.getMatRef();
    Mat &alpha = _alpha.getMatRef();
    Mat tempConf;

    globalMattingHelper(image_, trimap_, foreground, alpha, tempConf);

    ximgproc::guidedFilter(image, alpha, alpha, 10, 1e-5);

    if (_conf.needed())
        tempConf.copyTo(_conf);
}

void GlobalMattingImpl::getMat(InputArray image, InputArray trimap, OutputArray foreground, OutputArray alpha, int niter)
{
    Mat conf;
    globalMatting(image, trimap, foreground, alpha, conf);

    CV_CheckTypeEQ(image.type(), CV_8UC3, "image mush have CV_8UC3 type");
    CV_CheckTypeEQ(trimap.type(), CV_8UC1, "trimap mush have CV_8UC1 type");
    const Mat_<Vec3b> &image_ = (const Mat_<Vec3b>&)image.getMat();
    const Mat_<uchar> &trimap_ = (const Mat_<uchar>&)trimap.getMat();
    expansionOfKnownRegions(image_, trimap_, niter);

    CV_Assert(alpha.sameSize(trimap));
    CV_CheckTypeEQ(alpha.type(), CV_8UC1, "");
    Mat alpha_ = alpha.getMat();
    for (int y = 0; y < trimap_.rows; ++y)
    {
        for (int x = 0; x < trimap_.cols; ++x)
        {
            if (trimap_.at<uchar>(y, x) == 0)
                alpha_.at<uchar>(y, x) = 0;
            else if (trimap_.at<uchar>(y, x) == 255)
                alpha_.at<uchar>(y, x) = 255;
        }
    }
}

GlobalMatting::GlobalMatting()
{
    // nothing
}

GlobalMatting::~GlobalMatting()
{
    // nothing
}

CV_EXPORTS Ptr<GlobalMatting> createGlobalMatting()
{
    return makePtr<GlobalMattingImpl>();
}

}}  // namespace
