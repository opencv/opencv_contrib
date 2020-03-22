// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/rgbd/kinfu.hpp>

using namespace cv;
using namespace cv::kinfu;
using namespace std;

#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#endif

static vector<string> readDepth(std::string fileList);

static vector<string> readDepth(std::string fileList)
{
    vector<string> v;

    fstream file(fileList);
    if(!file.is_open())
        throw std::runtime_error("Failed to read depth list");

    std::string dir;
    size_t slashIdx = fileList.rfind('/');
    slashIdx = slashIdx != std::string::npos ? slashIdx : fileList.rfind('\\');
    dir = fileList.substr(0, slashIdx);

    while(!file.eof())
    {
        std::string s, imgPath;
        std::getline(file, s);
        if(s.empty() || s[0] == '#') continue;
        std::stringstream ss;
        ss << s;
        double thumb;
        ss >> thumb >> imgPath;
        v.push_back(dir+'/'+imgPath);
    }

    return v;
}

struct DepthWriter
{
    DepthWriter(string fileList) :
        file(fileList, ios::out), count(0), dir()
    {
        size_t slashIdx = fileList.rfind('/');
        slashIdx = slashIdx != std::string::npos ? slashIdx : fileList.rfind('\\');
        dir = fileList.substr(0, slashIdx);

        if(!file.is_open())
            throw std::runtime_error("Failed to write depth list");

        file << "# depth maps saved from device" << endl;
        file << "# useless_number filename" << endl;
    }

    void append(InputArray _depth)
    {
        Mat depth = _depth.getMat();
        string depthFname = cv::format("%04d.png", count);
        string fullDepthFname = dir + '/' + depthFname;
        if(!imwrite(fullDepthFname, depth))
            throw std::runtime_error("Failed to write depth to file " + fullDepthFname);
        file << count++ << " " << depthFname << endl;
    }

    fstream file;
    int count;
    string dir;
};

namespace Kinect2Params
{
    static const Size frameSize = Size(512, 424);
    // approximate values, no guarantee to be correct
    static const float focal = 366.1f;
    static const float cx = 258.2f;
    static const float cy = 204.f;
    static const float k1 =  0.12f;
    static const float k2 = -0.34f;
    static const float k3 =  0.12f;
};

struct DepthSource
{
public:
    enum Type
    {
        DEPTH_LIST,
        DEPTH_KINECT2_LIST,
        DEPTH_KINECT2,
        DEPTH_REALSENSE
    };

    DepthSource(int cam) :
        DepthSource("", cam)
    { }

    DepthSource(String fileListName) :
        DepthSource(fileListName, -1)
    { }

    DepthSource(String fileListName, int cam) :
        depthFileList(fileListName.empty() ? vector<string>() : readDepth(fileListName)),
        frameIdx(0),
        undistortMap1(),
        undistortMap2()
    {
        if(cam >= 0)
        {
            vc = VideoCapture(VideoCaptureAPIs::CAP_OPENNI2 + cam);
            if(vc.isOpened())
            {
                sourceType = Type::DEPTH_KINECT2;
            }
            else
            {
                vc = VideoCapture(VideoCaptureAPIs::CAP_REALSENSE + cam);
                if(vc.isOpened())
                {
                    sourceType = Type::DEPTH_REALSENSE;
                }
            }
        }
        else
        {
            vc = VideoCapture();
            sourceType = Type::DEPTH_KINECT2_LIST;
        }
    }

    UMat getDepth()
    {
        UMat out;
        if (!vc.isOpened())
        {
            if (frameIdx < depthFileList.size())
            {
                Mat f = cv::imread(depthFileList[frameIdx++], IMREAD_ANYDEPTH);
                f.copyTo(out);
            }
            else
            {
                return UMat();
            }
        }
        else
        {
            vc.grab();
            switch (sourceType)
            {
            case Type::DEPTH_KINECT2:
                vc.retrieve(out, CAP_OPENNI_DEPTH_MAP);
                break;
            case Type::DEPTH_REALSENSE:
                vc.retrieve(out, CAP_INTELPERC_DEPTH_MAP);
                break;
            default:
                // unknown depth source
                vc.retrieve(out);
            }

            // workaround for Kinect 2
            if(sourceType == Type::DEPTH_KINECT2)
            {
                out = out(Rect(Point(), Kinect2Params::frameSize));

                UMat outCopy;
                // linear remap adds gradient between valid and invalid pixels
                // which causes garbage, use nearest instead
                remap(out, outCopy, undistortMap1, undistortMap2, cv::INTER_NEAREST);

                cv::flip(outCopy, out, 1);
            }
        }
        if (out.empty())
            throw std::runtime_error("Matrix is empty");
        return out;
    }

    bool empty()
    {
        return depthFileList.empty() && !(vc.isOpened());
    }

    void updateParams(Params& params)
    {
        if (vc.isOpened())
        {
            // this should be set in according to user's depth sensor
            int w = (int)vc.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
            int h = (int)vc.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);

            // it's recommended to calibrate sensor to obtain its intrinsics
            float fx, fy, cx, cy;
            float depthFactor = 1000.f;
            Size frameSize;
            if(sourceType == Type::DEPTH_KINECT2)
            {
                fx = fy = Kinect2Params::focal;
                cx = Kinect2Params::cx;
                cy = Kinect2Params::cy;

                frameSize = Kinect2Params::frameSize;
            }
            else
            {
                if(sourceType == Type::DEPTH_REALSENSE)
                {
                    fx = (float)vc.get(CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ);
                    fy = (float)vc.get(CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT);
                    depthFactor = 1.f/(float)vc.get(CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE);
                }
                else
                {
                    fx = fy = (float)vc.get(CAP_OPENNI_DEPTH_GENERATOR | CAP_PROP_OPENNI_FOCAL_LENGTH);
                }

                cx = w/2 - 0.5f;
                cy = h/2 - 0.5f;

                frameSize = Size(w, h);
            }

            Matx33f camMatrix = Matx33f(fx,  0, cx,
                                        0,  fy, cy,
                                        0,   0,  1);

            params.frameSize = frameSize;
            params.intr = camMatrix;
            params.depthFactor = depthFactor;

            // RealSense has shorter depth range, some params should be tuned
            if(sourceType == Type::DEPTH_REALSENSE)
            {
                // all sizes in meters
                float cubeSize = 1.f;
                params.voxelSize = cubeSize/params.volumeDims[0];
                params.tsdf_trunc_dist = 0.01f;
                params.icpDistThresh = 0.01f;
                params.volumePose = Affine3f().translate(Vec3f(-cubeSize/2.f,
                                                               -cubeSize/2.f,
                                                               0.05f));
                params.truncateThreshold = 2.5f;
                params.bilateral_sigma_depth = 0.01f;
            }

            if(sourceType == Type::DEPTH_KINECT2)
            {
                Matx<float, 1, 5> distCoeffs;
                distCoeffs(0) = Kinect2Params::k1;
                distCoeffs(1) = Kinect2Params::k2;
                distCoeffs(4) = Kinect2Params::k3;

                initUndistortRectifyMap(camMatrix, distCoeffs, cv::noArray(),
                                        camMatrix, frameSize, CV_16SC2,
                                        undistortMap1, undistortMap2);
            }
        }
    }

    vector<string> depthFileList;
    size_t frameIdx;
    VideoCapture vc;
    UMat undistortMap1, undistortMap2;
    Type sourceType;
};

#ifdef HAVE_OPENCV_VIZ
const std::string vizWindowName = "cloud";

struct PauseCallbackArgs
{
    PauseCallbackArgs(KinFu& _kf) : kf(_kf)
    { }

    KinFu& kf;
};

void pauseCallback(const viz::MouseEvent& me, void* args);
void pauseCallback(const viz::MouseEvent& me, void* args)
{
    if(me.type == viz::MouseEvent::Type::MouseMove       ||
       me.type == viz::MouseEvent::Type::MouseScrollDown ||
       me.type == viz::MouseEvent::Type::MouseScrollUp)
    {
        PauseCallbackArgs pca = *((PauseCallbackArgs*)(args));
        viz::Viz3d window(vizWindowName);
        UMat rendered;
        pca.kf.render(rendered, window.getViewerPose().matrix);
        imshow("render", rendered);
        waitKey(1);
    }
}
#endif

static const char* keys =
{
    "{help h usage ? | | print this message   }"
    "{depth  | | Path to depth.txt file listing a set of depth images }"
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


int main(int argc, char **argv)
{
    bool coarse = false;
    bool idle = false;
    string recordPath;

    CommandLineParser parser(argc, argv, keys);
    parser.about(message);

    if(!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }

    if(parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if(parser.has("coarse"))
    {
        coarse = true;
    }
    if(parser.has("record"))
    {
        recordPath = parser.get<String>("record");
    }
    if(parser.has("idle"))
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
    if(!recordPath.empty())
        depthWriter = makePtr<DepthWriter>(recordPath);

    Ptr<Params> params;
    Ptr<KinFu> kf;

    if(coarse)
        params = Params::coarseParams();
    else
        params = Params::defaultParams();

    // These params can be different for each depth sensor
    ds->updateParams(*params);

    // Enables OpenCL explicitly (by default can be switched-off)
    cv::setUseOptimized(true);

    // Scene-specific params should be tuned for each scene individually
    //float cubeSize = 1.f;
    //params->voxelSize = cubeSize/params->volumeDims[0]; //meters
    //params->tsdf_trunc_dist = 0.01f; //meters
    //params->icpDistThresh = 0.01f; //meters
    //params->volumePose = Affine3f().translate(Vec3f(-cubeSize/2.f, -cubeSize/2.f, 0.25f)); //meters
    //params->tsdf_max_weight = 16;

    if(!idle)
        kf = KinFu::create(params);

#ifdef HAVE_OPENCV_VIZ
    cv::viz::Viz3d window(vizWindowName);
    window.setViewerPose(Affine3f::Identity());
    bool pause = false;
#endif

    UMat rendered;
    UMat points;
    UMat normals;

    int64 prevTime = getTickCount();

    for(UMat frame = ds->getDepth(); !frame.empty(); frame = ds->getDepth())
    {
        if(depthWriter)
            depthWriter->append(frame);

#ifdef HAVE_OPENCV_VIZ
        if(pause)
        {
            // doesn't happen in idle mode
            kf->getCloud(points, normals);
            if(!points.empty() && !normals.empty())
            {
                viz::WCloud cloudWidget(points, viz::Color::white());
                viz::WCloudNormals cloudNormals(points, normals, /*level*/1, /*scale*/0.05, viz::Color::gray());
                window.showWidget("cloud", cloudWidget);
                window.showWidget("normals", cloudNormals);

                Vec3d volSize = kf->getParams().voxelSize*Vec3d(kf->getParams().volumeDims);
                window.showWidget("cube", viz::WCube(Vec3d::all(0),
                                                     volSize),
                                  kf->getParams().volumePose);
                PauseCallbackArgs pca(*kf);
                window.registerMouseCallback(pauseCallback, (void*)&pca);
                window.showWidget("text", viz::WText(cv::String("Move camera in this window. "
                                                                "Close the window or press Q to resume"), Point()));
                window.spin();
                window.removeWidget("text");
                window.removeWidget("cloud");
                window.removeWidget("normals");
                window.registerMouseCallback(0);
            }

            pause = false;
        }
        else
#endif
        {
            UMat cvt8;
            float depthFactor = params->depthFactor;
            convertScaleAbs(frame, cvt8, 0.25*256. / depthFactor);
            if(!idle)
            {
                imshow("depth", cvt8);

                if(!kf->update(frame))
                {
                    kf->reset();
                    std::cout << "reset" << std::endl;
                }
#ifdef HAVE_OPENCV_VIZ
                else
                {
                    if(coarse)
                    {
                        kf->getCloud(points, normals);
                        if(!points.empty() && !normals.empty())
                        {
                            viz::WCloud cloudWidget(points, viz::Color::white());
                            viz::WCloudNormals cloudNormals(points, normals, /*level*/1, /*scale*/0.05, viz::Color::gray());
                            window.showWidget("cloud", cloudWidget);
                            window.showWidget("normals", cloudNormals);
                        }
                    }

                    //window.showWidget("worldAxes", viz::WCoordinateSystem());
                    Vec3d volSize = kf->getParams().voxelSize*kf->getParams().volumeDims;
                    window.showWidget("cube", viz::WCube(Vec3d::all(0),
                                                         volSize),
                                      kf->getParams().volumePose);
                    window.setViewerPose(kf->getPose());
                    window.spinOnce(1, true);
                }
#endif

                kf->render(rendered);
            }
            else
            {
                rendered = cvt8;
            }
        }

        int64 newTime = getTickCount();
        putText(rendered, cv::format("FPS: %2d press R to reset, P to pause, Q to quit",
                                     (int)(getTickFrequency()/(newTime - prevTime))),
                Point(0, rendered.rows-1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255));
        prevTime = newTime;

        imshow("render", rendered);

        int c = waitKey(1);
        switch (c)
        {
        case 'r':
            if(!idle)
                kf->reset();
            break;
        case 'q':
            return 0;
#ifdef HAVE_OPENCV_VIZ
        case 'p':
            if(!idle)
                pause = true;
#endif
        default:
            break;
        }
    }

    return 0;
}
