// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/rgbd/colored_kinfu.hpp>

#include "io_utils.hpp"

using namespace cv;
using namespace cv::kinfu;
using namespace cv::colored_kinfu;
using namespace cv::io_utils;


static const char* keys =
{
    "{help h usage ? | | print this message   }"
    "{depth  | | Path to folder with depth.txt and rgb.txt files listing a set of depth and rgb images }"
    "{camera |0| Index of depth camera to be used as a depth source }"
    "{coarse | | Run on coarse settings (fast but ugly) or on default (slow but looks better),"
        " in coarse mode points and normals are displayed }"
    "{idle   | | Do not run KinFu, just display depth frames }"
    "{record | | Write depth frames to specified file list"
        " (the same format as for the 'depth' key) }"
};

static const std::string message =
 "\nThis demo uses live depth input or RGB-D dataset taken from"
 "\nhttps://vision.in.tum.de/data/datasets/rgbd-dataset"
 "\nto demonstrate KinectFusion implementation \n";


int main(int argc, char** argv)
{
    bool coarse = false;
    bool idle = false;
    std::string recordPath;

    CommandLineParser parser(argc, argv, keys);
    parser.about(message);

    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (parser.has("coarse"))
    {
        coarse = true;
    }
    if (parser.has("record"))
    {
        recordPath = parser.get<String>("record");
    }
    if (parser.has("idle"))
    {
        idle = true;
    }

    Ptr<DepthSource> ds;
    Ptr<RGBSource> rgbs;

    if (parser.has("depth"))
        ds = makePtr<DepthSource>(parser.get<String>("depth") + "/depth.txt");
    else
        ds = makePtr<DepthSource>(parser.get<int>("camera"));

    //TODO: intrinsics for camera
    rgbs = makePtr<RGBSource>(parser.get<String>("depth") + "/rgb.txt");

    if (ds->empty())
    {
        std::cerr << "Failed to open depth source" << std::endl;
        parser.printMessage();
        return -1;
    }

    Ptr<DepthWriter> depthWriter;
    Ptr<RGBWriter> rgbWriter;

    if (!recordPath.empty())
    {
        depthWriter = makePtr<DepthWriter>(recordPath);
        rgbWriter = makePtr<RGBWriter>(recordPath);
    }
    VolumeSettings vs(VolumeType::ColorTSDF);
    Ptr<ColoredKinFu> kf = ColoredKinFu::create();

    // Enables OpenCL explicitly (by default can be switched-off)
    cv::setUseOptimized(false);

    UMat rendered;
    UMat points;
    UMat normals;

    int64 prevTime = getTickCount();

    for (UMat frame = ds->getDepth(); !frame.empty(); frame = ds->getDepth())
    {
        if (depthWriter)
            depthWriter->append(frame);
        UMat rgb_frame = rgbs->getRGB();
        {
            UMat cvt8;
            float depthFactor = vs.getDepthFactor();
            convertScaleAbs(frame, cvt8, 0.25 * 256. / depthFactor);
            if (!idle)
            {
                imshow("depth", cvt8);
                imshow("rgb", rgb_frame);
                if (!kf->update(frame, rgb_frame))
                {
                    kf->reset();


                    kf->render(rendered);
                }
                else
                {
                    rendered = cvt8;
                }
            }

            int64 newTime = getTickCount();
            putText(rendered, cv::format("FPS: %2d press R to reset, P to pause, Q to quit",
                (int)(getTickFrequency() / (newTime - prevTime))),
                Point(0, rendered.rows - 1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255));
            prevTime = newTime;

            imshow("render", rendered);

            int c = waitKey(1);
            switch (c)
            {
            case 'r':
                if (!idle)
                    kf->reset();
                break;
            case 'q':
                return 0;
            default:
                break;
            }
        }

        return 0;
    }
}
