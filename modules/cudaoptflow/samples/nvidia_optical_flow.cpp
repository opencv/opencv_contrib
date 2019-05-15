#include <unordered_map>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

//this function is taken from opencv/samples/gpu/optical_flow.cpp
inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

//this function is taken from opencv/samples/gpu/optical_flow.cpp
static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

//this function is taken from opencv/samples/gpu/optical_flow.cpp
static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy
    , Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

int main(int argc, char **argv)
{
    std::unordered_map<std::string, NvidiaOpticalFlow_1_0::NVIDIA_OF_PERF_LEVEL> presetMap = {
        { "slow", NvidiaOpticalFlow_1_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_SLOW },
        { "medium", NvidiaOpticalFlow_1_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_MEDIUM },
        { "fast", NvidiaOpticalFlow_1_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_FAST } };

    try
    {
        CommandLineParser cmd(argc, argv,
            "{ l left   | ../data/basketball1.png | specify left image }"
            "{ r right  | ../data/basketball2.png | specify right image }"
            "{ g gpuid  | 0 | cuda device index}"
            "{ p preset | slow | perf preset for OF algo [ options : slow, medium, fast ]}"
            "{ o output | OpenCVNvOF.flo | output flow vector file in middlebury format}"
            "{ th enableTemporalHints | false | Enable temporal hints}"
            "{ eh enableExternalHints | false | Enable external hints}"
            "{ cb enableCostBuffer | false | Enable output cost buffer}"
            "{ h help   | | print help message }");

        cmd.about("Nvidia's optical flow sample.");
        if (cmd.has("help") || !cmd.check())
        {
            cmd.printMessage();
            cmd.printErrors();
            return 0;
        }

        string pathL = cmd.get<string>("left");
        string pathR = cmd.get<string>("right");
        string preset = cmd.get<string>("preset");
        string output = cmd.get<string>("output");
        bool enableExternalHints = cmd.get<bool>("enableExternalHints");
        bool enableTemporalHints = cmd.get<bool>("enableTemporalHints");
        bool enableCostBuffer = cmd.get<bool>("enableCostBuffer");
        int gpuId = cmd.get<int>("gpuid");

        if (pathL.empty()) cout << "Specify left image path\n";
        if (pathR.empty()) cout << "Specify right image path\n";
        if (preset.empty()) cout << "Specify perf preset for OpticalFlow algo\n";
        if (pathL.empty() || pathR.empty()) return 0;

        auto search = presetMap.find(preset);
        if (search == presetMap.end())
        {
            std::cout << "Invalid preset level : " << preset << std::endl;
            return 0;
        }
        NvidiaOpticalFlow_1_0::NVIDIA_OF_PERF_LEVEL perfPreset = search->second;

        Mat frameL = imread(pathL, IMREAD_GRAYSCALE);
        Mat frameR = imread(pathR, IMREAD_GRAYSCALE);
        if (frameL.empty()) cout << "Can't open '" << pathL << "'\n";
        if (frameR.empty()) cout << "Can't open '" << pathR << "'\n";
        if (frameL.empty() || frameR.empty()) return -1;

        Ptr<NvidiaOpticalFlow_1_0> nvof = NvidiaOpticalFlow_1_0::create(
            frameL.size().width, frameL.size().height, perfPreset,
            enableTemporalHints, enableExternalHints, enableCostBuffer, gpuId);

        Mat flowx, flowy, flowxy, upsampledFlowXY, image;

        nvof->calc(frameL, frameR, flowxy);

        nvof->upSampler(flowxy, frameL.size().width, frameL.size().height,
            nvof->getGridSize(), upsampledFlowXY);

        if (output.size() != 0)
        {
            if (!writeOpticalFlow(output, upsampledFlowXY))
                cout << "Failed to save Flow Vector" << endl;
            else
                cout << "Flow vector saved as '" << output << "'\n";
        }

        Mat planes[] = { flowx, flowy };
        split(upsampledFlowXY, planes);
        flowx = planes[0]; flowy = planes[1];

        drawOpticalFlow(flowx, flowy, image, 10);

        imshow("Colorize image",image);
        waitKey(0);
        nvof->collectGarbage();
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}