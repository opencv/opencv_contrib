#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class RangeHisto {
public :
    vector<int> minChannel;
    vector<int> maxChannel;
    RangeHisto(int n)
    {minChannel.resize(n); maxChannel.resize(n);}
};

Mat PlotHistogram(Mat x, RangeHisto r,int channel);
void AddSlider(String sliderName, String windowName, int minSlider, int maxSlider, int valDefault, int *valSlider, void(*f)(int, void *), void *r);

int main(void)
{
    VideoCapture v(0);
    if (!v.isOpened())
    {
        cout<<"Cannot open video\n";
        return 0;
    }
    Mat frame;
    v >> frame;
    int code=0;
    RangeHisto rangeH(frame.channels());

    String winH="histogram";
    for (int i = 0; i < frame.channels(); i++)
    {
        String s= format("%s_%d", winH.c_str(), i);
        namedWindow(s);
        rangeH.minChannel[i] = 1;
        rangeH.maxChannel[i] = 254;
        AddSlider(format("minChannel_%d", i), s, 0, 256, rangeH.minChannel[i], &rangeH.minChannel[i], NULL, NULL);
        AddSlider(format("maxChannel_%d", i), s, 0, 256, rangeH.maxChannel[i], &rangeH.maxChannel[i], NULL, NULL);
    }

    while (code != 27)
    {
        v>>frame;
        for (int i = 0; i < frame.channels(); i++)
        {
            String s = format("%s_%d", winH.c_str(), i);
            Mat plotHist= PlotHistogram(frame, rangeH,i);
            if (!plotHist.empty())
                imshow(s, plotHist);
        }
        imshow("Video", frame);
        code =  waitKey(30);
    }
    return 0;
}

Mat PlotHistogram(Mat x,RangeHisto r, int channel)
{
    if (x.channels()<=channel)
        return Mat();
    Mat histo;
    vector<Mat> y;
    split(x,y);
    int histSize[] = { r.maxChannel[channel]- r.minChannel[channel] };
    if (histSize[0] <= 0)
        return Mat();
    float sranges[] = { static_cast<float>(r.minChannel[channel]), static_cast<float>(r.maxChannel[channel]) };
    int channels[] = { 0 };
    const float* ranges[] = { sranges };
    calcHist(&y[channel], 1, channels, Mat(), histo, 1, histSize, ranges,
        true, false);

    Ptr<plot::Plot2d> plot;
    Mat display;
    Vec3b color[3]={ Vec3b(255,0,0), Vec3b(0,255,0), Vec3b(0,0,255)};
    double minVal, maxVal;
    Point p;
    minMaxLoc(histo, &minVal, &maxVal,NULL,&p);
    histo.convertTo(histo,CV_64F);
    plot = plot::createPlot2d(histo);
    plot->setPlotSize(512, 500);
    plot->setPlotAxisColor(Vec3b(255, 255, 255));
    plot->setPlotLineColor(color[channel%3]);
    plot->setMaxX(r.maxChannel[channel]- r.minChannel[channel]);
    plot->setMinX(0);
    plot->setMaxY(0);
    plot->setMinY(maxVal);
    plot->render(display);
    return display;
}

void AddSlider(String sliderName, String windowName, int minSlider, int maxSlider, int valDefault, int *valSlider, void(*f)(int, void *), void *r)
{
    createTrackbar(sliderName, windowName, valSlider, 1, f, r);
    setTrackbarMin(sliderName, windowName, minSlider);
    setTrackbarMax(sliderName, windowName, maxSlider);
    setTrackbarPos(sliderName, windowName, valDefault);
}