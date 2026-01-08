#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/plot.hpp"
#include "samples_utility.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace cv;

// TODO: do normalization ala Kalal's assessment protocol for TLD

static const Scalar gtColor = Scalar(0, 255, 0);

static Scalar getNextColor()
{
    const int num = 6;
    static Scalar colors[num] = {Scalar(160, 0, 0),   Scalar(0, 0, 160),   Scalar(0, 160, 160),
                                 Scalar(160, 160, 0), Scalar(160, 0, 160), Scalar(20, 50, 160)};
    static int id = 0;
    return colors[id < num ? id++ : num - 1];
}

inline vector<Rect2d> readGT(const string &filename, const string &omitname)
{
    vector<Rect2d> res;
    {
        ifstream input(filename.c_str());
        if (!input.is_open())
            CV_Error(Error::StsError, "Failed to open file");
        while (input)
        {
            Rect2d one;
            input >> one.x;
            input.ignore(numeric_limits<std::streamsize>::max(), ',');
            input >> one.y;
            input.ignore(numeric_limits<std::streamsize>::max(), ',');
            input >> one.width;
            input.ignore(numeric_limits<std::streamsize>::max(), ',');
            input >> one.height;
            input.ignore(numeric_limits<std::streamsize>::max(), '\n');
            if (input.good())
                res.push_back(one);
        }
    }
    if (!omitname.empty())
    {
        ifstream input(omitname.c_str());
        if (!input.is_open())
            CV_Error(Error::StsError, "Failed to open file");
        while (input)
        {
            unsigned int a = 0, b = 0;
            input >> a >> b;
            input.ignore(numeric_limits<std::streamsize>::max(), '\n');
            if (a > 0 && b > 0 && a < res.size() && b < res.size())
            {
                if (a > b)
                    swap(a, b);
                for (vector<Rect2d>::iterator i = res.begin() + a; i != res.begin() + b; ++i)
                {
                    *i = Rect2d();
                }
            }
        }
    }
    return res;
}

inline bool isGoodBox(const Rect2d &box) { return box.width > 0. && box.height > 0.; }
const int LTRC_COUNT = 100;

struct AlgoWrap
{
    AlgoWrap(const string &name_)
        : lastState(NotFound), name(name_), color(getNextColor()),
          numTotal(0), numResponse(0), numPresent(0), numCorrect_0(0), numCorrect_0_5(0),
          timeTotal(0), auc(LTRC_COUNT + 1, 0)
    {
        tracker = createTrackerByName(name);
    }

    enum State
    {
        NotFound,
        Overlap_None,
        Overlap_0,
        Overlap_0_5,
    };

    Ptr<Tracker> tracker;
    bool lastRes;
    Rect lastBox;
    State lastState;

    // visual
    string name;
    Scalar color;

    // results
    int numTotal;       // frames passed to tracker
    int numResponse;    // frames where tracker had response
    int numPresent;     // frames where ground truth result present
    int numCorrect_0;   // frames where overlap with GT > 0
    int numCorrect_0_5; // frames where overlap with GT > 0.5
    int64 timeTotal;    // ticks
    vector<int> auc;   // number of frames for each overlap percent

    void eval(const Mat &frame, const Rect2d &gtBox, bool isVerbose)
    {
        // RUN
        lastBox = Rect();
        int64 frameTime = getTickCount();
        lastRes = tracker->update(frame, lastBox);
        frameTime = getTickCount() - frameTime;

        // RESULTS
        double intersectArea = (gtBox & (Rect2d)lastBox).area();
        double unionArea = (gtBox | (Rect2d)lastBox).area();
        numTotal++;
        numResponse += (lastRes && isGoodBox(lastBox)) ? 1 : 0;
        numPresent += isGoodBox(gtBox) ? 1 : 0;
        double overlap = unionArea > 0. ? intersectArea / unionArea : 0.;
        numCorrect_0 += overlap > 0. ? 1 : 0;
        numCorrect_0_5 += overlap > 0.5 ? 1 : 0;
        auc[std::min(std::max((size_t)(overlap * LTRC_COUNT), (size_t)0), (size_t)LTRC_COUNT)]++;
        timeTotal += frameTime;

        if (isVerbose)
            cout << name << " - " << overlap << endl;

        if (isGoodBox(gtBox) != isGoodBox(lastBox)) lastState = NotFound;
        else if (overlap > 0.5) lastState = Overlap_0_5;
        else if (overlap > 0.0001) lastState = Overlap_0;
        else lastState = Overlap_None;
    }

    void draw(Mat &image, const Point &textPoint) const
    {
        if (lastRes)
            rectangle(image, lastBox, color, 2, LINE_8);
        string suf;
        switch (lastState)
        {
        case AlgoWrap::NotFound: suf = " X"; break;
        case AlgoWrap::Overlap_None: suf = " ~"; break;
        case AlgoWrap::Overlap_0: suf = " +"; break;
        case AlgoWrap::Overlap_0_5: suf = " ++"; break;
        }
        putText(image, name + suf, textPoint, FONT_HERSHEY_PLAIN, 1, color, 1, LINE_AA);
    }

    // calculates "lost track ratio" curve - row of values growing from 0 to 1
    // number of elements is LTRC_COUNT + 2
    Mat getLTRC() const
    {
        Mat t, res;
        Mat(auc).convertTo(t, CV_64F); // integral does not support CV_32S input
        integral(t.t(), res, CV_64F); // t is a column of values
        return res.row(1) / (double)numTotal;
    }

    void plotLTRC(Mat &img) const
    {
        Ptr<plot::Plot2d> p_ = plot::Plot2d::create(getLTRC());
        p_->render(img);
    }

    double calcAUC() const
    {
        return cv::sum(getLTRC())[0] / (double)LTRC_COUNT;
    }

    void stat(ostream &out) const
    {
        out << name << endl;
        out << setw(20) << "Overlap > 0  " << setw(20) << (double)numCorrect_0 / numTotal * 100
            << "%" << setw(20) << numCorrect_0 << endl;
        out << setw(20) << "Overlap > 0.5" << setw(20) << (double)numCorrect_0_5 / numTotal * 100
            << "%" << setw(20) << numCorrect_0_5 << endl;

        double p = (double)numCorrect_0_5 / numResponse;
        double r = (double)numCorrect_0_5 / numPresent;
        double f = 2 * p * r / (p + r);
        out << setw(20) << "Precision" << setw(20) << p * 100 << "%" << endl;
        out << setw(20) << "Recall   " << setw(20) << r * 100 << "%" << endl;
        out << setw(20) << "f-measure" << setw(20) << f * 100 << "%" << endl;
        out << setw(20) << "AUC" << setw(20) << calcAUC() << endl;

        double s = (timeTotal / getTickFrequency()) / numTotal;
        out << setw(20) << "Performance" << setw(20) << s * 1000 << " ms/frame" << setw(20) << 1 / s
            << " fps" << endl;
    }
};

inline ostream &operator<<(ostream &out, const AlgoWrap &w) { w.stat(out); return out; }

inline vector<AlgoWrap> initAlgorithms(const string &algList)
{
    vector<AlgoWrap> res;
    istringstream input(algList);
    for (;;)
    {
        char one[30];
        input.getline(one, 30, ',');
        if (!input)
            break;
        cout << "  " << one << " - ";
        AlgoWrap a(one);
        if (a.tracker)
        {
            res.push_back(a);
            cout << "OK";
        }
        else
        {
            cout << "FAILED";
        }
        cout << endl;
    }
    return res;
}

static const string &window = "Tracking API";

int main(int argc, char **argv)
{
    const string keys =
        "{help h||show help}"
        "{video||video file to process}"
        "{gt||ground truth file (each line describes rectangle in format: '<x>,<y>,<w>,<h>')}"
        "{start|0|starting frame}"
        "{num|0|frame number (0 for all)}"
        "{omit||file with omit ranges (each line describes occluded frames: '<start> <end>')}"
        "{plot|false|plot LTR curves at the end}"
        "{v|false|print each frame info}"
        "{@algos||comma-separated algorithm names}";
    CommandLineParser p(argc, argv, keys);
    if (p.has("help"))
    {
        p.printMessage();
        return 0;
    }
    int startFrame = p.get<int>("start");
    int frameCount = p.get<int>("num");
    string videoFile = p.get<string>("video");
    string gtFile = p.get<string>("gt");
    string omitFile = p.get<string>("omit");
    string algList = p.get<string>("@algos");
    bool doPlot = p.get<bool>("plot");
    bool isVerbose = p.get<bool>("v");
    if (!p.check())
    {
        p.printErrors();
        return 0;
    }

    cout << "Reading GT from " << gtFile << " ... ";
    vector<Rect2d> gt = readGT(gtFile, omitFile);
    if (gt.empty())
        CV_Error(Error::StsError, "Failed to read GT file");
    cout << gt.size() << " boxes" << endl;

    cout << "Opening video " << videoFile << " ... ";
    VideoCapture cap;
    cap.open(videoFile);
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Failed to open video file");
    cap.set(CAP_PROP_POS_FRAMES, startFrame);
    cout << "at frame " << startFrame << endl;

    // INIT
    vector<AlgoWrap> algos = initAlgorithms(algList);
    Mat frame, image;
    cap >> frame;
    for (vector<AlgoWrap>::iterator i = algos.begin(); i != algos.end(); ++i)
        i->tracker->init(frame, gt[0]);

    // DRAW
    {
        namedWindow(window, WINDOW_AUTOSIZE);
        frame.copyTo(image);
        rectangle(image, gt[0], gtColor, 2, LINE_8);
        imshow(window, image);
    }

    bool paused = false;
    int frameId = 0;
    cout << "Hot keys:" << endl << "  q - exit" << endl << "  p - pause" << endl;
    for (;;)
    {
        if (!paused)
        {
            cap >> frame;
            if (frame.empty())
            {
                cout << "Done - video end" << endl;
                break;
            }
            frameId++;
            if (isVerbose)
                cout << endl << "Frame " << frameId << endl;
            // EVAL
            for (vector<AlgoWrap>::iterator i = algos.begin(); i != algos.end(); ++i)
                i->eval(frame, gt[frameId], isVerbose);
            // DRAW
            {
                Point textPoint(1, 16);
                frame.copyTo(image);
                rectangle(image, gt[frameId], gtColor, 2, LINE_8);
                putText(image, "GROUND TRUTH", textPoint, FONT_HERSHEY_PLAIN, 1, gtColor, 1, LINE_AA);
                for (vector<AlgoWrap>::iterator i = algos.begin(); i != algos.end(); ++i)
                {
                    textPoint.y += 14;
                    i->draw(image, textPoint);
                }
                imshow(window, image);
            }
        }

        char c = (char)waitKey(1);
        if (c == 'q')
        {
            cout << "Done - manual exit" << endl;
            break;
        }
        else if (c == 'p')
        {
            paused = !paused;
        }
        if (frameCount && frameId >= frameCount)
        {
            cout << "Done - max frame count" << endl;
            break;
        }
    }

    // STAT
    for (vector<AlgoWrap>::iterator i = algos.begin(); i != algos.end(); ++i)
        cout << "==========" << endl << *i << endl;

    if (doPlot)
    {
        Mat img(300, 300, CV_8UC3);
        for (vector<AlgoWrap>::iterator i = algos.begin(); i != algos.end(); ++i)
        {
            i->plotLTRC(img);
            imshow("LTR curve for " + i->name, img);
        }
        waitKey(0);
    }

    return 0;
}
