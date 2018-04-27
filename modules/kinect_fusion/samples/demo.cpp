//TODO: license here

#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/kinect_fusion.hpp>

using namespace cv;
using namespace std;

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


static const char* keys =
{
    "{help h usage ? | | print this message   }"
    "{@depth |<none>| Path to depth.txt file listing a set of depth images }"
};

static const std::string message =
 "\nThis demo uses RGB-D dataset taken from"
 "\nhttps://vision.in.tum.de/data/datasets/rgbd-dataset"
 "\nto demonstrate KinectFusion implementation \n";


int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(message);
    if(parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String depthPath = parser.get<String>(0);

    if(!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }

    vector<string> depthFileList = readDepth(depthPath);

    kinfu::KinFu kf(kinfu::KinFu::KinFuParams::coarseParams());

    cv::viz::Viz3d window("debug");
    window.setViewerPose(Affine3f::Identity());

    // TODO: can we use UMats for that?
    Mat rendered;
    Mat points;
    Mat normals;

    for(size_t nFrame = 0; nFrame < depthFileList.size(); nFrame++)
    {
        Mat frame = cv::imread(depthFileList[nFrame], IMREAD_ANYDEPTH);
        if(frame.empty())
            throw std::runtime_error("Matrix is empty");

        Mat cvt8;
        convertScaleAbs(frame, cvt8, 0.25f/5000.f*256.f);
        imshow("depth", cvt8);

        if(!kf(frame))
            std::cout << "reset" << std::endl;
        else
        {
            kf.fetchCloud(points, normals);
            viz::WCloud cloudWidget(points, viz::Color::white());
            viz::WCloudNormals cloudNormals(points, normals, /*level*/1, /*scale*/0.05, viz::Color::gray());
            window.showWidget("cloud", cloudWidget);
            window.showWidget("normals", cloudNormals);
            window.showWidget("worldAxes", viz::WCoordinateSystem());
            window.showWidget("cube", viz::WCube(Vec3d::all(0),
                                                 Vec3d::all(kf.getParams().volumeSize)),
                              kf.getParams().volumePose);
            window.spinOnce(1, true);
        }

        kf.render(rendered);
        imshow("render", rendered);

        waitKey(10);
    }

    return 0;
}
