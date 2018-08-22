// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
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
        throw std::runtime_error("Failed to open file");

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


const Size kinect2FrameSize(512, 424);
// approximate values, no guarantee to be correct
const float kinect2Focal = 366.1f;
const float kinect2Cx = 258.2f;
const float kinect2Cy = 204.f;

struct DepthSource
{
public:
    DepthSource() :
        depthFileList(),
        frameIdx(0),
        vc(),
        useKinect2Workarounds(true)
    { }

    DepthSource(int cam) :
        depthFileList(),
        frameIdx(),
        vc(VideoCaptureAPIs::CAP_OPENNI2 + cam),
        useKinect2Workarounds(true)
    { }

    DepthSource(String fileListName) :
        depthFileList(readDepth(fileListName)),
        frameIdx(0),
        vc(),
        useKinect2Workarounds(true)
    { }

    Mat getDepth()
    {
        Mat out;
        if (!vc.isOpened())
        {
            if (frameIdx < depthFileList.size())
                out = cv::imread(depthFileList[frameIdx++], IMREAD_ANYDEPTH);
            else
            {
                return Mat();
            }
        }
        else
        {
            vc.grab();
            vc.retrieve(out, CAP_OPENNI_DEPTH_MAP);

            // workaround for Kinect 2
            if(useKinect2Workarounds)
            {
                out = out(Rect(Point(), kinect2FrameSize));
                cv::flip(out, out, 1);
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

            float focal = (float)vc.get(CAP_OPENNI_DEPTH_GENERATOR | CAP_PROP_OPENNI_FOCAL_LENGTH);

            // it's recommended to calibrate sensor to obtain its intrinsics
            float fx, fy, cx, cy;
            fx = fy = useKinect2Workarounds ? kinect2Focal : focal;
            cx = useKinect2Workarounds ? kinect2Cx : w/2 - 0.5f;
            cy = useKinect2Workarounds ? kinect2Cy : h/2 - 0.5f;

            params.frameSize = useKinect2Workarounds ? kinect2FrameSize : Size(w, h);
            params.intr = Matx33f(fx,  0, cx,
                                   0, fy, cy,
                                   0,  0,  1);
            params.depthFactor = 1000.f;
        }
    }

    vector<string> depthFileList;
    size_t frameIdx;
    VideoCapture vc;
    bool useKinect2Workarounds;
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
        Mat rendered;
        pca.kf.render(rendered, window.getViewerPose().matrix);
        imshow("render", rendered);
        waitKey(1);
    }
}
#endif

static const char* keys =
{
    "{help h usage ? | | print this message   }"
    "{depth | | Path to depth.txt file listing a set of depth images }"
    "{camera | | Index of depth camera to be used as a depth source }"
    "{coarse | | Run on coarse settings (fast but ugly) or on default (slow but looks better),"
        " in coarse mode points and normals are displayed }"
};

static const std::string message =
 "\nThis demo uses live depth input or RGB-D dataset taken from"
 "\nhttps://vision.in.tum.de/data/datasets/rgbd-dataset"
 "\nto demonstrate KinectFusion implementation \n";


int main(int argc, char **argv)
{
    bool coarse = false;

    CommandLineParser parser(argc, argv, keys);
    parser.about(message);
    if(parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if(parser.has("coarse"))
    {
        coarse = true;
    }

    if(!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }

    DepthSource ds;
    if (parser.has("depth"))
        ds = DepthSource(parser.get<String>("depth"));
    if (parser.has("camera") && ds.empty())
        ds = DepthSource(parser.get<int>("camera"));

    if (ds.empty())
    {
        std::cerr << "Failed to open depth source" << std::endl;
        parser.printMessage();
        return -1;
    }

    Ptr<Params> params;
    if(coarse)
        params = Params::coarseParams();
    else
        params = Params::defaultParams();

    // These params can be different for each depth sensor
    ds.updateParams(*params);

    // Scene-specific params should be tuned for each scene individually
    //params.volumePose = params.volumePose.translate(Vec3f(0.f, 0.f, 0.5f));
    //params.tsdf_max_weight = 16;

    Ptr<KinFu> kf = KinFu::create(params);

#ifdef HAVE_OPENCV_VIZ
    cv::viz::Viz3d window(vizWindowName);
    window.setViewerPose(Affine3f::Identity());
    bool pause = false;
#endif

    // TODO: can we use UMats for that?
    Mat rendered;
    Mat points;
    Mat normals;

    int64 prevTime = getTickCount();

    for(Mat frame = ds.getDepth(); !frame.empty(); frame = ds.getDepth())
    {
#ifdef HAVE_OPENCV_VIZ
        if(pause)
        {
            kf->getCloud(points, normals);
            if(!points.empty() && !normals.empty())
            {
                viz::WCloud cloudWidget(points, viz::Color::white());
                viz::WCloudNormals cloudNormals(points, normals, /*level*/1, /*scale*/0.05, viz::Color::gray());
                window.showWidget("cloud", cloudWidget);
                window.showWidget("normals", cloudNormals);

                window.showWidget("cube", viz::WCube(Vec3d::all(0),
                                                     Vec3d::all(kf->getParams().volumeSize)),
                                  kf->getParams().volumePose);
                PauseCallbackArgs pca(*kf);
                window.registerMouseCallback(pauseCallback, (void*)&pca);
                window.showWidget("text", viz::WText(cv::String("Move camera in this window. "
                                                                "Close the window or press Q to resume"), Point()));
                window.spin();
                window.removeWidget("text");
                window.registerMouseCallback(0);
            }

            pause = false;
        }
        else
#endif
        {
            Mat cvt8;
            float depthFactor = kf->getParams().depthFactor;
            convertScaleAbs(frame, cvt8, 0.25*256. / depthFactor);
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
                window.showWidget("cube", viz::WCube(Vec3d::all(0),
                                                     Vec3d::all(kf->getParams().volumeSize)),
                                  kf->getParams().volumePose);
                window.setViewerPose(kf->getPose());
                window.spinOnce(1, true);
            }
#endif

            kf->render(rendered);
        }

        int64 newTime = getTickCount();
        putText(rendered, cv::format("FPS: %2d press R to reset, P to pause, Q to quit",
                                     (int)(getTickFrequency()/(newTime - prevTime))),
                Point(0, rendered.rows-1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255));
        prevTime = newTime;

        imshow("render", rendered);

        int c = waitKey(100);
        switch (c)
        {
        case 'r':
            kf->reset();
            break;
        case 'q':
            return 0;
#ifdef HAVE_OPENCV_VIZ
        case 'p':
            pause = true;
#endif
        default:
            break;
        }
    }

    return 0;
}
