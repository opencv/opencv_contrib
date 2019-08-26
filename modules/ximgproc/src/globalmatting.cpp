
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc/edge_filter.hpp>

#include <string> 
#include <cstdlib>
#include "opencv2/ximgproc/globalmatting.hpp"


namespace cv
{
namespace ximgproc
{	
std::vector<cv::Point> GlobalMatting::findBoundaryPixels(const cv::Mat_<uchar> &trimap, int a, int b)
{
    std::vector<cv::Point> result;

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
	            result.push_back(cv::Point(x, y));
	        }
	    }
	}

    return result;
}

// Eq. 2
float GlobalMatting::calculateAlpha(const cv::Vec3b &F, const cv::Vec3b &B, const cv::Vec3b &I)
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

    return std::min(std::max(result / div, 0.f), 1.f);
}

// Eq. 3
float GlobalMatting::colorCost(const cv::Vec3b &F, const cv::Vec3b &B, const cv::Vec3b &I, float alpha)
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
float GlobalMatting::distCost(const cv::Point &p0, const cv::Point &p1, float minDist)
{
    int dist = sqr(p0.x - p1.x) + sqr(p0.y - p1.y);
    return sqrt((float)dist) / minDist;
}

float GlobalMatting::colorDist(const cv::Vec3b &I0, const cv::Vec3b &I1)
{
    int result = 0;

    for (int c = 0; c < 3; ++c)
	result += sqr((int)I0[c] - (int)I1[c]);

    return sqrt((float)result);
}

float GlobalMatting::nearestDistance(const std::vector<cv::Point> &boundary, const cv::Point &p)
{
    int minDist2 = INT_MAX;
    for (std::size_t i = 0; i < boundary.size(); ++i)
    {
	int dist2 = sqr(boundary[i].x - p.x)  + sqr(boundary[i].y - p.y);
	minDist2 = std::min(minDist2, dist2);
    }

    return sqrt((float)minDist2);
}




void GlobalMatting::expansionOfKnownRegions(const cv::Mat_<cv::Vec3b> &image,
	                            cv::Mat_<uchar> &trimap,
	                            int r, float c)
{
    int w = image.cols;
    int h = image.rows;

    for (int x = 0; x < w; ++x)
	for (int y = 0; y < h; ++y)
	{
	    if (trimap(y, x) != 128)
	        continue;

	    const cv::Vec3b &I = image(y, x);

	    for (int j = y-r; j <= y+r; ++j)
	        for (int i = x-r; i <= x+r; ++i)
	        {
	            if (i < 0 || i >= w || j < 0 || j >= h)
	                continue;

	            if (trimap(j, i) != 0 && trimap(j, i) != 255)
	                continue;

	            const cv::Vec3b &I2 = image(j, i);

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

    for (int x = 0; x < trimap.cols; ++x)
	for (int y = 0; y < trimap.rows; ++y)
	{
	    if (trimap(y, x) == 1)
	        trimap(y, x) = 0;
	    else if (trimap(y, x) == 254)
	        trimap(y, x) = 255;

	}
}

// erode foreground and background regions to increase the size of unknown region
void GlobalMatting::erodeFB(cv::Mat_<uchar> &trimap, int r)
{
    int w = trimap.cols;
    int h = trimap.rows;

    cv::Mat_<uchar> foreground(trimap.size(), (uchar)0);
    cv::Mat_<uchar> background(trimap.size(), (uchar)0);

    for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	{
	    if (trimap(y, x) == 0)
	        background(y, x) = 1;
	    else if (trimap(y, x) == 255)
	        foreground(y, x) = 1;
	}


    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(r, r));

    cv::erode(background, background, kernel);
    cv::erode(foreground, foreground, kernel);

    for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	{
	    if (background(y, x) == 0 && foreground(y, x) == 0)
	        trimap(y, x) = 128;
	}
}


void GlobalMatting::calculateAlphaPatchMatch(const cv::Mat_<cv::Vec3b> &image,
	const cv::Mat_<uchar> &trimap,
	const std::vector<cv::Point> &foregroundBoundary,
	const std::vector<cv::Point> &backgroundBoundary,
	std::vector<std::vector<Sample> > &samples)
{
    int w = image.cols;
    int h = image.rows;

    samples.resize(h, std::vector<Sample>(w));

    for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	{
	    if (trimap(y, x) == 128)
	    {
	        cv::Point p(x, y);

	        samples[y][x].fi = rand() % foregroundBoundary.size();
	        samples[y][x].bj = rand() % backgroundBoundary.size();
	        samples[y][x].df = nearestDistance(foregroundBoundary, p);
	        samples[y][x].db = nearestDistance(backgroundBoundary, p);
	        samples[y][x].cost = FLT_MAX;
	    }
	}

    std::vector<cv::Point> coords(w * h);
    for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	    coords[x + y * w] = cv::Point(x, y);

    for (int iter = 0; iter < 10; ++iter)
    {
	// propagation
	std::random_shuffle(coords.begin(), coords.end());

	for (std::size_t i = 0; i < coords.size(); ++i)
	{
	    const cv::Point &p = coords[i];

	    int x = p.x;
	    int y = p.y;

	    if (trimap(y, x) != 128)
	        continue;

	    const cv::Vec3b &I = image(y, x);

	    Sample &s = samples[y][x];

	    for (int y2 = y - 1; y2 <= y + 1; ++y2)
	        for (int x2 = x - 1; x2 <= x + 1; ++x2)
	        {
	            if (x2 < 0 || x2 >= w || y2 < 0 || y2 >= h)
	                continue;

	            if (trimap(y2, x2) != 128)
	                continue;

	            Sample &s2 = samples[y2][x2];

	            const cv::Point &fp = foregroundBoundary[s2.fi];
	            const cv::Point &bp = backgroundBoundary[s2.bj];

	            const cv::Vec3b F = image(fp.y, fp.x);
	            const cv::Vec3b B = image(bp.y, bp.x);

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

	// random walk
	int w2 = (int)std::max(foregroundBoundary.size(), backgroundBoundary.size());

	for (int y = 0; y < h; ++y)
	    for (int x = 0; x < w; ++x)
	    {
	        if (trimap(y, x) != 128)
	            continue;

	        cv::Point p(x, y);

	        const cv::Vec3b &I = image(y, x);

	        Sample &s = samples[y][x];

	        for (int k = 0; ; k++)
	        {
	            float r = w2 * pow(0.5f, k);

	            if (r < 1)
	                break;

	            int di = r * (rand() / (RAND_MAX + 1.f));
	            int dj = r * (rand() / (RAND_MAX + 1.f));

	            int fi = s.fi + di;
	            int bj = s.bj + dj;

	            if (fi < 0 || fi >= foregroundBoundary.size() || bj < 0 || bj >= backgroundBoundary.size())
	                continue;

	            const cv::Point &fp = foregroundBoundary[fi];
	            const cv::Point &bp = backgroundBoundary[bj];

	            const cv::Vec3b F = image(fp.y, fp.x);
	            const cv::Vec3b B = image(bp.y, bp.x);

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
}

void GlobalMatting::expansionOfKnownRegionsHelper(const cv::Mat &_image,
	                                  cv::Mat &_trimap,
	                                  int r, float c)
{
    const cv::Mat_<cv::Vec3b> &image = (const cv::Mat_<cv::Vec3b> &)_image;
    cv::Mat_<uchar> &trimap = (cv::Mat_<uchar>&)_trimap;

    int w = image.cols;
    int h = image.rows;

    for (int x = 0; x < w; ++x)
	for (int y = 0; y < h; ++y)
	{
	    if (trimap(y, x) != 128)
	        continue;

	    const cv::Vec3b &I = image(y, x);

	    for (int j = y-r; j <= y+r; ++j)
	        for (int i = x-r; i <= x+r; ++i)
	        {
	            if (i < 0 || i >= w || j < 0 || j >= h)
	                continue;

	            if (trimap(j, i) != 0 && trimap(j, i) != 255)
	                continue;

	            const cv::Vec3b &I2 = image(j, i);

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

    for (int x = 0; x < trimap.cols; ++x)
	for (int y = 0; y < trimap.rows; ++y)
	{
	    if (trimap(y, x) == 1)
	        trimap(y, x) = 0;
	    else if (trimap(y, x) == 254)
	        trimap(y, x) = 255;

	}
}

// erode foreground and background regions to increase the size of unknown region
void GlobalMatting::erodeFB(cv::Mat &_trimap, int r)
{
    cv::Mat_<uchar> &trimap = (cv::Mat_<uchar>&)_trimap;

    int w = trimap.cols;
    int h = trimap.rows;

    cv::Mat_<uchar> foreground(trimap.size(), (uchar)0);
    cv::Mat_<uchar> background(trimap.size(), (uchar)0);

    for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	{
	    if (trimap(y, x) == 0)
	        background(y, x) = 1;
	    else if (trimap(y, x) == 255)
	        foreground(y, x) = 1;
	}


    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(r, r));

    cv::erode(background, background, kernel);
    cv::erode(foreground, foreground, kernel);

    for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	{
	    if (background(y, x) == 0 && foreground(y, x) == 0)
	        trimap(y, x) = 128;
	}
}

void GlobalMatting::expansionOfKnownRegions(cv::InputArray _img, cv::InputOutputArray _trimap, int niter)
{
    cv::Mat img = _img.getMat();
    cv::Mat &trimap = _trimap.getMatRef();

    if (img.empty())
	CV_Error(CV_StsBadArg, "image is empty");
    if (img.type() != CV_8UC3)
	CV_Error(CV_StsBadArg, "image mush have CV_8UC3 type");

    if (trimap.empty())
	CV_Error(CV_StsBadArg, "trimap is empty");
    if (trimap.type() != CV_8UC1)
	CV_Error(CV_StsBadArg, "trimap mush have CV_8UC1 type");

    if (img.size() != trimap.size())
	CV_Error(CV_StsBadArg, "image and trimap mush have same size");

    for (int i = 0; i < niter; ++i)
	expansionOfKnownRegionsHelper(img, trimap, i + 1, niter - i);
    erodeFB(trimap, 2);
}


void GlobalMatting::globalMattingHelper(cv::Mat _image, cv::Mat _trimap, cv::Mat &_foreground, cv::Mat &_alpha, cv::Mat &_conf)
{
    const cv::Mat_<cv::Vec3b> &image = (const cv::Mat_<cv::Vec3b>&)_image;
    const cv::Mat_<uchar> &trimap = (const cv::Mat_<uchar>&)_trimap;

    std::vector<cv::Point> foregroundBoundary = findBoundaryPixels(trimap, 255, 128);
    std::vector<cv::Point> backgroundBoundary = findBoundaryPixels(trimap, 0, 128);

    int n = (int)(foregroundBoundary.size() + backgroundBoundary.size());
    for (int i = 0; i < n; ++i)
    {
	int x = rand() % trimap.cols;
	int y = rand() % trimap.rows;

	if (trimap(y, x) == 0)
	    backgroundBoundary.push_back(cv::Point(x, y));
	else if (trimap(y, x) == 255)
	    foregroundBoundary.push_back(cv::Point(x, y));
    }

    std::sort(foregroundBoundary.begin(), foregroundBoundary.end(), IntensityComp(image));
    std::sort(backgroundBoundary.begin(), backgroundBoundary.end(), IntensityComp(image));

    std::vector<std::vector<Sample> > samples;
    calculateAlphaPatchMatch(image, trimap, foregroundBoundary, backgroundBoundary, samples);

    _foreground.create(image.size(), CV_8UC3);
    _alpha.create(image.size(), CV_8UC1);
    _conf.create(image.size(), CV_8UC1);

    cv::Mat_<cv::Vec3b> &foreground = (cv::Mat_<cv::Vec3b>&)_foreground;
    cv::Mat_<uchar> &alpha = (cv::Mat_<uchar>&)_alpha;
    cv::Mat_<uchar> &conf = (cv::Mat_<uchar>&)_conf;

    for (int y = 0; y < alpha.rows; ++y)
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
	            alpha(y, x) = 255 * samples[y][x].alpha;
	            conf(y, x) = 255 * exp(-samples[y][x].cost / 6);
	            cv::Point p = foregroundBoundary[samples[y][x].fi];
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

void GlobalMatting::globalMatting(cv::InputArray _image, cv::InputArray _trimap, cv::OutputArray _foreground, cv::OutputArray _alpha, cv::OutputArray _conf)
{
    cv::Mat image = _image.getMat();
    cv::Mat trimap = _trimap.getMat();

    if (image.empty())
	CV_Error(CV_StsBadArg, "image is empty");
    if (image.type() != CV_8UC3)
	CV_Error(CV_StsBadArg, "image mush have CV_8UC3 type");

    if (trimap.empty())
	CV_Error(CV_StsBadArg, "trimap is empty");
    if (trimap.type() != CV_8UC1)
	CV_Error(CV_StsBadArg, "trimap mush have CV_8UC1 type");

    if (image.size() != trimap.size())
	CV_Error(CV_StsBadArg, "image and trimap mush have same size");

    cv::Mat &foreground = _foreground.getMatRef();
    cv::Mat &alpha = _alpha.getMatRef();
    cv::Mat tempConf;

    globalMattingHelper(image, trimap, foreground, alpha, tempConf);

    cv::ximgproc::guidedFilter(image,alpha,alpha,10,1e-5);

    if(_conf.needed())
	tempConf.copyTo(_conf);
}

void GlobalMatting::getMat(cv::Mat image,cv::Mat trimap,cv::Mat &foreground,cv:: Mat &alpha,int niter)
{
  cv::Mat conf;
  globalMatting(image,trimap,foreground,alpha,conf);
  expansionOfKnownRegions(image,trimap,niter);
  for (int x = 0; x < trimap.cols; ++x)
        {
            for (int y = 0; y < trimap.rows; ++y)
		{
		    if (trimap.at<uchar>(y, x) == 0)
		        alpha.at<uchar>(y, x) = 0;
		    else if (trimap.at<uchar>(y, x) == 255)
		        alpha.at<uchar>(y, x) = 255;
		}
         }
    
}

GlobalMatting::GlobalMatting()
{
   cout<<"The Global matting object has been instantiated"<<endl;
}

}
}
 




