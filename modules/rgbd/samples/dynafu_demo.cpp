// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE

#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/rgbd.hpp>

using namespace cv;
using namespace cv::dynafu;
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
    DepthSource(int cam) :
        DepthSource("", cam)
    { }

    DepthSource(String fileListName) :
        DepthSource(fileListName, -1)
    { }

    DepthSource(String fileListName, int cam) :
        depthFileList(fileListName.empty() ? vector<string>() : readDepth(fileListName)),
        frameIdx(0),
        vc( cam >= 0 ? VideoCapture(VideoCaptureAPIs::CAP_OPENNI2 + cam) : VideoCapture()),
        undistortMap1(),
        undistortMap2(),
        useKinect2Workarounds(true)
    {
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
            vc.retrieve(out, CAP_OPENNI_DEPTH_MAP);

            // workaround for Kinect 2
            if(useKinect2Workarounds)
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

            float focal = (float)vc.get(CAP_OPENNI_DEPTH_GENERATOR | CAP_PROP_OPENNI_FOCAL_LENGTH);

            // it's recommended to calibrate sensor to obtain its intrinsics
            float fx, fy, cx, cy;
            Size frameSize;
            if(useKinect2Workarounds)
            {
                fx = fy = Kinect2Params::focal;
                cx = Kinect2Params::cx;
                cy = Kinect2Params::cy;

                frameSize = Kinect2Params::frameSize;
            }
            else
            {
                fx = fy = focal;
                cx = w/2 - 0.5f;
                cy = h/2 - 0.5f;

                frameSize = Size(w, h);
            }

            Matx33f camMatrix = Matx33f(fx,  0, cx,
                                        0,  fy, cy,
                                        0,   0,  1);

            params.frameSize = frameSize;
            params.intr = camMatrix;
            params.depthFactor = 1000.f;

            Matx<float, 1, 5> distCoeffs;
            distCoeffs(0) = Kinect2Params::k1;
            distCoeffs(1) = Kinect2Params::k2;
            distCoeffs(4) = Kinect2Params::k3;
            if(useKinect2Workarounds)
                initUndistortRectifyMap(camMatrix, distCoeffs, cv::noArray(),
                                        camMatrix, frameSize, CV_16SC2,
                                        undistortMap1, undistortMap2);
        }
    }

    vector<string> depthFileList;
    size_t frameIdx;
    VideoCapture vc;
    UMat undistortMap1, undistortMap2;
    bool useKinect2Workarounds;
};

#ifdef HAVE_OPENCV_VIZ
const std::string vizWindowName = "cloud";

struct PauseCallbackArgs
{
    PauseCallbackArgs(DynaFu& _df) : df(_df)
    { }

    DynaFu& df;
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
        pca.df.render(rendered, window.getViewerPose().matrix);
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
    "{idle   | | Do not run DynaFu, just display depth frames }"
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
    Ptr<DynaFu> df;

    if(coarse)
        params = Params::coarseParams();
    else
        params = Params::defaultParams();

    // These params can be different for each depth sensor
    ds->updateParams(*params);

    // Enables OpenCL explicitly (by default can be switched-off)
    cv::setUseOptimized(false);

    // Scene-specific params should be tuned for each scene individually
    //params->volumePose = params->volumePose.translate(Vec3f(0.f, 0.f, 0.5f));
    //params->tsdf_max_weight = 16;

    namedWindow("OpenGL Window", WINDOW_OPENGL);
    resizeWindow("OpenGL Window", 1, 1);
    if(!idle)
        df = DynaFu::create(params);

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
        Mat depthImg, vertImg, normImg;
        setOpenGlContext("OpenGL Window");
        df->renderSurface(depthImg, vertImg, normImg);
        if(!depthImg.empty())
        {
            UMat depthCvt8, vertCvt8, normCvt8;
            convertScaleAbs(depthImg, depthCvt8, 0.33*255);
            vertImg.convertTo(vertCvt8, CV_8UC3, 255);
            normImg.convertTo(normCvt8, CV_8UC3, 255);

            imshow("Surface prediction", depthCvt8);
            imshow("vertex prediction", vertCvt8);
            imshow("normal prediction", normCvt8);
        }

        if(depthWriter)
            depthWriter->append(frame);

#ifdef HAVE_OPENCV_VIZ
        if(pause)
        {
            // doesn't happen in idle mode
            df->getCloud(points, normals);

            if(!points.empty() && !normals.empty())
            {
                viz::WCloud cloudWidget(points, viz::Color::white());
                viz::WCloudNormals cloudNormals(points, normals, /*level*/1, /*scale*/0.05, viz::Color::gray());

                Vec3d volSize = df->getParams().voxelSize*Vec3d(df->getParams().volumeDims);
                window.showWidget("cube", viz::WCube(Vec3d::all(0),
                                                     volSize),
                                  df->getParams().volumePose);
                PauseCallbackArgs pca(*df);
                window.registerMouseCallback(pauseCallback, (void*)&pca);
                window.showWidget("text", viz::WText(cv::String("Move camera in this window. "
                                                                "Close the window or press Q to resume"), Point()));
                window.spin();
                window.removeWidget("text");
                //window.removeWidget("cloud");
                //window.removeWidget("normals");
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

                if(!df->update(frame))
                {
                    df->reset();
                    std::cout << "reset" << std::endl;
                }
#ifdef HAVE_OPENCV_VIZ
                else
                {
                    Mat meshCloud, meshEdges, meshPoly;
                    df->marchCubes(meshCloud, meshEdges);
                    for(int i = 0; i < meshEdges.size().height; i += 3)
                    {
                        meshPoly.push_back<int>(3);
                        meshPoly.push_back<int>(meshEdges.at<int>(i, 0));
                        meshPoly.push_back<int>(meshEdges.at<int>(i+1, 0));
                        meshPoly.push_back<int>(meshEdges.at<int>(i+2, 0));
                    }

                    viz::WMesh mesh(meshCloud.t(), meshPoly);
                    window.showWidget("mesh", mesh);

                    if(coarse)
                    {
                        df->getCloud(points, normals);

                        if(!points.empty() && !normals.empty())
                        {
                            viz::WCloud cloudWidget(points, viz::Color::white());
                            viz::WCloudNormals cloudNormals(points, normals, /*level*/1, /*scale*/0.05, viz::Color::gray());
                            //window.showWidget("cloud", cloudWidget);
                            //window.showWidget("normals", cloudNormals);
                            if(!df->getNodesPos().empty())
                            {
                                viz::WCloud nodeCloud(df->getNodesPos(), viz::Color::red());
                                nodeCloud.setRenderingProperty(viz::POINT_SIZE, 4);
                                window.showWidget("nodes", nodeCloud);
                            }
                        }
                    }

                    //window.showWidget("worldAxes", viz::WCoordinateSystem());
                    Vec3d volSize = df->getParams().voxelSize*df->getParams().volumeDims;
                    window.showWidget("cube", viz::WCube(Vec3d::all(0),
                                                         volSize),
                                      df->getParams().volumePose);
                    window.setViewerPose(df->getPose());
                    window.spinOnce(1, true);
                }
#endif

                df->render(rendered);
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
                df->reset();
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
