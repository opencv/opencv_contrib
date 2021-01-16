#include <unordered_map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/video/tracking.hpp"

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

/*
ROI config file format.
numrois 3
roi0 640 96 1152 192
roi1 640 64 896 864
roi2 640 960 256 32
*/
bool parseROI(std::string ROIFileName, std::vector<Rect>& roiData)
{
    std::string str;
    uint32_t nRois = 0;
    std::ifstream hRoiFile;
    hRoiFile.open(ROIFileName, std::ios::in);

    if (hRoiFile.is_open())
    {
        while (std::getline(hRoiFile, str))
        {
            std::istringstream iss(str);
            std::vector<std::string> tokens{ std::istream_iterator<std::string>{iss},
                std::istream_iterator<std::string>{} };

            if (tokens.size() == 0) continue; // if empty line, coninue

            transform(tokens[0].begin(), tokens[0].end(), tokens[0].begin(), ::tolower);
            if (tokens[0] == "numrois")
            {
                nRois = atoi(tokens[1].data());
            }
            else if (tokens[0].rfind("roi", 0) == 0)
            {
                cv::Rect roi;
                roi.x = atoi(tokens[1].data());
                roi.y = atoi(tokens[2].data());
                roi.width = atoi(tokens[3].data());
                roi.height = atoi(tokens[4].data());
                roiData.push_back(roi);
            }
            else if (tokens[0].rfind("#", 0) == 0)
            {
                continue;
            }
            else
            {
                std::cout << "Unidentified keyword in roi config file " << tokens[0] << std::endl;
                hRoiFile.close();
                return false;
            }
        }
    }
    else
    {
        std::cout << "Unable to open ROI file " << std::endl;
        return false;
    }
    if (nRois != roiData.size())
    {
        std::cout << "NumRois(" << nRois << ")and specified roi rects (" << roiData.size() << ")are not matching " << std::endl;
        hRoiFile.close();
        return false;
    }
    hRoiFile.close();
    return true;
}

int main(int argc, char **argv)
{
    std::unordered_map<std::string, NvidiaOpticalFlow_2_0::NVIDIA_OF_PERF_LEVEL> presetMap = {
        { "slow", NvidiaOpticalFlow_2_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_SLOW },
        { "medium", NvidiaOpticalFlow_2_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_MEDIUM },
        { "fast", NvidiaOpticalFlow_2_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_FAST } };

    std::unordered_map<int, NvidiaOpticalFlow_2_0::NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE> outputGridSize = {
        { 1, NvidiaOpticalFlow_2_0::NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE::NV_OF_OUTPUT_VECTOR_GRID_SIZE_1 },
        { 2, NvidiaOpticalFlow_2_0::NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE::NV_OF_OUTPUT_VECTOR_GRID_SIZE_2 },
        { 4, NvidiaOpticalFlow_2_0::NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE::NV_OF_OUTPUT_VECTOR_GRID_SIZE_4 } };

    std::unordered_map<int, NvidiaOpticalFlow_2_0::NVIDIA_OF_HINT_VECTOR_GRID_SIZE> hintGridSize = {
        { 1, NvidiaOpticalFlow_2_0::NVIDIA_OF_HINT_VECTOR_GRID_SIZE::NV_OF_HINT_VECTOR_GRID_SIZE_1 },
        { 2, NvidiaOpticalFlow_2_0::NVIDIA_OF_HINT_VECTOR_GRID_SIZE::NV_OF_HINT_VECTOR_GRID_SIZE_2 },
        { 4, NvidiaOpticalFlow_2_0::NVIDIA_OF_HINT_VECTOR_GRID_SIZE::NV_OF_HINT_VECTOR_GRID_SIZE_4 },
        { 8, NvidiaOpticalFlow_2_0::NVIDIA_OF_HINT_VECTOR_GRID_SIZE::NV_OF_HINT_VECTOR_GRID_SIZE_8 } };

    try
    {
        CommandLineParser cmd(argc, argv,
            "{ l left   | ../data/basketball1.png | specify left image }"
            "{ r right  | ../data/basketball2.png | specify right image }"
            "{ g gpuid  | 0 | cuda device index}"
            "{ p preset | slow | perf preset for OF algo [ options : slow, medium, fast ]}"
            "{ og outputGridSize | 1 | Output grid size of OF vector [ options : 1, 2, 4 ]}"
            "{ hg hintGridSize | 1 | Hint grid size of OF vector [ options : 1, 2, 4, 8 ]}"
            "{ o output | OpenCVNvOF.flo | output flow vector file in middlebury format}"
            "{ rc roiConfigFile | | Region of Interest config file }"
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

        std::string pathL = cmd.get<std::string>("left");
        std::string pathR = cmd.get<std::string>("right");
        std::string preset = cmd.get<std::string>("preset");
        std::string output = cmd.get<std::string>("output");
        std::string roiConfiFile = cmd.get<std::string>("roiConfigFile");
        bool enableExternalHints = cmd.get<bool>("enableExternalHints");
        bool enableTemporalHints = cmd.get<bool>("enableTemporalHints");
        bool enableCostBuffer = cmd.get<bool>("enableCostBuffer");
        int gpuId = cmd.get<int>("gpuid");
        int outputBufferGridSize = cmd.get<int>("outputGridSize");
        int hintBufferGridSize = cmd.get<int>("hintGridSize");

        if (pathL.empty()) std::cout << "Specify left image path" << std::endl;
        if (pathR.empty()) std::cout << "Specify right image path" << std::endl;
        if (preset.empty()) std::cout << "Specify perf preset for OpticalFlow algo" << std::endl;
        if (pathL.empty() || pathR.empty()) return 0;

        auto p = presetMap.find(preset);
        if (p == presetMap.end())
        {
            std::cout << "Invalid preset level : " << preset << std::endl;
            return 0;
        }
        NvidiaOpticalFlow_2_0::NVIDIA_OF_PERF_LEVEL perfPreset = p->second;

        auto o = outputGridSize.find(outputBufferGridSize);
        if (o == outputGridSize.end())
        {
            std::cout << "Invalid output grid size: " << outputBufferGridSize << std::endl;
            return 0;
        }
        NvidiaOpticalFlow_2_0::NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE outBufGridSize = o->second;

        NvidiaOpticalFlow_2_0::NVIDIA_OF_HINT_VECTOR_GRID_SIZE hintBufGridSize =
            NvidiaOpticalFlow_2_0::NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
        if (enableExternalHints)
        {
            auto h = hintGridSize.find(hintBufferGridSize);
            if (h == hintGridSize.end())
            {
                std::cout << "Invalid hint grid size: " << hintBufferGridSize << std::endl;
                return 0;
            }
            hintBufGridSize = h->second;
        }

        std::vector<Rect> roiData;

        if (!roiConfiFile.empty())
        {
            if (!parseROI(roiConfiFile, roiData))
            {
                std::cout << "Wrong Region of Interest config file, proceeding without ROI" << std::endl;
            }
        }

        Mat frameL = imread(pathL, IMREAD_GRAYSCALE);
        Mat frameR = imread(pathR, IMREAD_GRAYSCALE);
        if (frameL.empty()) std::cout << "Can't open '" << pathL << "'" << std::endl;
        if (frameR.empty()) std::cout << "Can't open '" << pathR << "'" << std::endl;
        if (frameL.empty() || frameR.empty()) return -1;

        Ptr<NvidiaOpticalFlow_2_0> nvof = NvidiaOpticalFlow_2_0::create(
            frameL.size(), roiData, perfPreset, outBufGridSize, hintBufGridSize,
            enableTemporalHints, enableExternalHints, enableCostBuffer, gpuId);

        Mat flowx, flowy, flowxy, floatFlow, image;

        nvof->calc(frameL, frameR, flowxy);

        nvof->convertToFloat(flowxy, floatFlow);

        if (!output.empty())
        {
            if (!writeOpticalFlow(output, floatFlow))
                std::cout << "Failed to save Flow Vector" << std::endl;
            else
                std::cout << "Flow vector saved as '" << output << "'" << std::endl;
        }

        Mat planes[] = { flowx, flowy };
        split(floatFlow, planes);
        flowx = planes[0]; flowy = planes[1];

        drawOpticalFlow(flowx, flowy, image, 10);

        imshow("Colorize image", image);
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