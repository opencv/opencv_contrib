#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

struct ThParameters {
    int levelNoise;
    int angle;
    int scale10;
    int origin;
    int xg;
    int yg;
    bool update;
} ;

static vector<Point> NoisyPolygon(vector<Point> pRef, double n);
static void UpdateShape(int , void *r);
static void AddSlider(String sliderName, String windowName, int minSlider, int maxSlider, int valDefault, int *valSlider, void(*f)(int, void *), void *r);

int main(void)
{
    vector<Point> ctrRef;
    vector<Point> ctrRotate, ctrNoisy, ctrNoisyRotate, ctrNoisyRotateShift;
    // build a shape with 5 vertex
    ctrRef.push_back(Point(250,250)); ctrRef.push_back(Point(400, 250));
    ctrRef.push_back(Point(400, 300)); ctrRef.push_back(Point(250, 300));ctrRef.push_back(Point(180, 270));
    Point cg(0,0);
    for (int i=0;i<static_cast<int>(ctrRef.size());i++)
        cg+=ctrRef[i];
    cg.x /= static_cast<int>(ctrRef.size());
    cg.y /= static_cast<int>(ctrRef.size());
    ThParameters p;
    p.levelNoise=6;
    p.angle=45;
    p.scale10=5;
    p.origin=10;
    p.xg=150;
    p.yg=150;
    p.update=true;
    namedWindow("FD Curve matching");
    // A rotation with center at (150,150) of angle 45 degrees and a scaling of 5/10
    AddSlider("Noise", "FD Curve matching", 0, 20, p.levelNoise, &p.levelNoise, UpdateShape, &p);
    AddSlider("Angle", "FD Curve matching", 0, 359, p.angle, &p.angle, UpdateShape, &p);
    AddSlider("Scale", "FD Curve matching", 5, 100, p.scale10, &p.scale10, UpdateShape, &p);
    AddSlider("Origin%%", "FD Curve matching", 0, 100, p.origin, &p.origin, UpdateShape, &p);
    AddSlider("Xg", "FD Curve matching", 150, 450, p.xg, &p.xg, UpdateShape, &p);
    AddSlider("Yg", "FD Curve matching", 150, 450, p.yg, &p.yg, UpdateShape, &p);
    int code=0;
    double dist;
    vector<vector<Point> > c;
    Mat img;
    cout << "******************** PRESS g TO MATCH CURVES *************\n";
    do
    {
        code = waitKey(30);
        if (p.update)
        {
            Mat r = getRotationMatrix2D(Point(p.xg, p.yg), p.angle, 10.0/ p.scale10);
            ctrNoisy= NoisyPolygon(ctrRef,static_cast<double>(p.levelNoise));
            cv::transform(ctrNoisy, ctrNoisyRotate, r);
            ctrNoisyRotateShift.clear();
            for (int i=0;i<static_cast<int>(ctrNoisy.size());i++)
                ctrNoisyRotateShift.push_back(ctrNoisyRotate[(i+(p.origin*ctrNoisy.size())/100)% ctrNoisy.size()]);
            // To draw contour using drawcontours
            c.clear();
            c.push_back(ctrRef);
            c.push_back(ctrNoisyRotateShift);
            p.update = false;
            Rect rglobal;
            for (int i = 0; i < static_cast<int>(c.size()); i++)
            {
                rglobal = boundingRect(c[i]) | rglobal;
            }
            rglobal.width += 10;
            rglobal.height += 10;
            img = Mat::zeros(2 * rglobal.height, 2 * rglobal.width, CV_8UC(3));
            drawContours(img, c, 0, Scalar(255,0,0));
            drawContours(img, c, 1, Scalar(0, 255, 0));
            circle(img, c[0][0], 5, Scalar(255, 0, 0));
            circle(img, c[1][0], 5, Scalar(0, 255, 0));
            imshow("FD Curve matching", img);
        }
        if (code == 'd')
        {
            destroyWindow("FD Curve matching");
            namedWindow("FD Curve matching");
            // A rotation with center at (150,150) of angle 45 degrees and a scaling of 5/10
            AddSlider("Noise", "FD Curve matching", 0, 20, p.levelNoise, &p.levelNoise, UpdateShape, &p);
            AddSlider("Angle", "FD Curve matching", 0, 359, p.angle, &p.angle, UpdateShape, &p);
            AddSlider("Scale", "FD Curve matching", 5, 100, p.scale10, &p.scale10, UpdateShape, &p);
            AddSlider("Origin%%", "FD Curve matching", 0, 100, p.origin, &p.origin, UpdateShape, &p);
            AddSlider("Xg", "FD Curve matching", 150, 450, p.xg, &p.xg, UpdateShape, &p);
            AddSlider("Yg", "FD Curve matching", 150, 450, p.yg, &p.yg, UpdateShape, &p);

        }
        if (code == 'g')
        {
            ximgproc::ContourFitting fit;
            vector<Point2f> ctrRef2d, ctrRot2d;
            // sampling contour we want 256 points
            ximgproc::contourSampling(ctrRef, ctrRef2d, 256); // use a mat
            ximgproc::contourSampling(ctrNoisyRotateShift, ctrRot2d, 256); // use a vector og point
            fit.setFDSize(16);
            Mat t;
            fit.estimateTransformation(ctrRot2d, ctrRef2d, t, &dist, false);
            cout << "Transform *********\n "<<"Origin = "<< t.at<double>(0,0)*ctrNoisy.size() <<" expected "<< (p.origin*ctrNoisy.size()) / 100 <<" ("<< ctrNoisy.size()<<")\n";
            cout << "Angle = " << t.at<double>(0, 1) * 180 / M_PI << " expected " << p.angle  <<"\n";
            cout << "Scale = " << t.at<double>(0, 2) << " expected " << p.scale10 / 10.0 << "\n";
            Mat dst;
            ximgproc::transformFD(ctrRot2d, t, dst, false);
            c.push_back(dst);
            drawContours(img, c, 2, Scalar(0,255,255));
            circle(img, c[2][0], 5, Scalar(0, 255, 255));
            imshow("FD Curve matching", img);
        }
    }
    while (code!=27);

    return 0;
}

vector<Point> NoisyPolygon(vector<Point> pRef, double n)
{
    RNG rng;
    vector<Point> c;
    vector<Point> p = pRef;
    vector<vector<Point> > contour;
    for (int i = 0; i<static_cast<int>(p.size()); i++)
        p[i] += Point(Point2d(n*rng.uniform((double)-1, (double)1), n*rng.uniform((double)-1, (double)1)));
    if (n==0)
        return p;
    c.push_back(p[0]);
    int minX = p[0].x, maxX = p[0].x, minY = p[0].y, maxY = p[0].y;
    for (int i = 0; i <static_cast<int>(p.size()); i++)
    {
        int next = i + 1;
        if (next == static_cast<int>(p.size()))
            next = 0;
        Point2d u = p[next] - p[i];
        int d = static_cast<int>(norm(u));
        double a = atan2(u.y, u.x);
        int step = 1;
        if (n != 0)
            step = static_cast<int>(d / n);
        for (int j = 1; j<d; j += max(step, 1))
        {
            Point pNew;
            do
            {

                Point2d pAct = (u*j) / static_cast<double>(d);
                double r = n*rng.uniform((double)0, (double)1);
                double theta = a + rng.uniform(0., 2 * CV_PI);
                pNew = Point(Point2d(r*cos(theta) + pAct.x + p[i].x, r*sin(theta) + pAct.y + p[i].y));
            } while (pNew.x<0 || pNew.y<0);
            if (pNew.x<minX)
                minX = pNew.x;
            if (pNew.x>maxX)
                maxX = pNew.x;
            if (pNew.y<minY)
                minY = pNew.y;
            if (pNew.y>maxY)
                maxY = pNew.y;
            c.push_back(pNew);
        }
    }
    return c;
}

void UpdateShape(int , void *r)
{
    ((ThParameters *)r)->update = true;
}

void AddSlider(String sliderName, String windowName, int minSlider, int maxSlider, int valDefault, int *valSlider, void(*f)(int, void *), void *r)
{
    createTrackbar(sliderName, windowName, valSlider, 1, f, r);
    setTrackbarMin(sliderName, windowName, minSlider);
    setTrackbarMax(sliderName, windowName, maxSlider);
    setTrackbarPos(sliderName, windowName, valDefault);
}
