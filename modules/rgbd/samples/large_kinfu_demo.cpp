// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#include <fstream>
#include <iostream>
#include <opencv2/3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd/large_kinfu.hpp>

#include "io_utils.hpp"

using namespace cv;
using namespace cv::kinfu;
using namespace cv::large_kinfu;
using namespace cv::io_utils;


static const char* keys = {
    "{help h usage ? | | print this message   }"
    "{depth  | | Path to depth.txt file listing a set of depth images }"
    "{camera |0| Index of depth camera to be used as a depth source }"
    "{coarse | | Run on coarse settings (fast but ugly) or on default (slow but looks better),"
    " in coarse mode points and normals are displayed }"
    "{idle   | | Do not run LargeKinfu, just display depth frames }"
    "{record | | Write depth frames to specified file list"
    " (the same format as for the 'depth' key) }"
};

static const std::string message =
    "\nThis demo uses live depth input or RGB-D dataset taken from"
    "\nhttps://vision.in.tum.de/data/datasets/rgbd-dataset"
    "\nto demonstrate Submap based large environment reconstruction"
    "\nThis module uses the newer hashtable based TSDFVolume (relatively fast) for larger "
    "reconstructions by default\n";

int main(int argc, char** argv)
{
    bool coarse = false;
    bool idle   = false;
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
    if (parser.has("depth"))
        ds = makePtr<DepthSource>(parser.get<String>("depth"));
    else
        ds = makePtr<DepthSource>(parser.get<int>("camera"));

    if (ds->empty())
    {
        std::cerr << "Failed to open depth source" << std::endl;
        parser.printMessage();
        return -1;
    }

    Ptr<DepthWriter> depthWriter;
    if (!recordPath.empty())
        depthWriter = makePtr<DepthWriter>(recordPath);

    VolumeSettings vs = VolumeSettings(VolumeType::HashTSDF);
    Ptr<LargeKinfu> largeKinfu = LargeKinfu::create();

    cv::setUseOptimized(true);

    UMat rendered;
    UMat points;
    UMat normals;

    int64 prevTime = getTickCount();

    for (UMat frame = ds->getDepth(); !frame.empty(); frame = ds->getDepth())
    {
        if (depthWriter)
            depthWriter->append(frame);


        Vec3i volResolution;
        vs.getVolumeResolution(volResolution);
        Matx44f pose;
        vs.getVolumePose(pose);
        Affine3f volPose(pose);
        {
            UMat cvt8;
            float depthFactor = vs.getDepthFactor();
            convertScaleAbs(frame, cvt8, 0.25 * 256. / depthFactor);
            if (!idle)
            {
                imshow("depth", cvt8);

                if (!largeKinfu->update(frame))
                {
                    largeKinfu->reset();
                    std::cout << "reset" << std::endl;
                }

                largeKinfu->render(rendered);
            }
            else
            {
                rendered = cvt8;
            }
        }

        int64 newTime = getTickCount();
        putText(rendered,
                cv::format("FPS: %2d press R to reset, P to pause, Q to quit",
                           (int)(getTickFrequency() / (newTime - prevTime))),
                Point(0, rendered.rows - 1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255));
        prevTime = newTime;
        imshow("render", rendered);

        int c = waitKey(1);
        switch (c)
        {
            case 'r':
                if (!idle)
                    largeKinfu->reset();
                break;
            case 'q': return 0;
            default: break;
        }
    }

    return 0;
}
