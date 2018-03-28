//TODO: license here

#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/kinect_fusion.hpp>

using namespace cv;
using namespace std;

/*
//DEBUG!
static vector<double> timestamps;
typedef std::pair<double, Affine3f> TimePose;
static vector<TimePose> poses;
static Affine3f currentPose = Affine3f::Identity();
//because the first call is done at 1st frame
static int frame = 1;
static void readGT()
{
    timestamps.clear();
    fstream depthFile("/home/savuor/datasets/rgbd_dataset_freiburg1_xyz/depth.txt");
    if(!depthFile.is_open())
        throw std::runtime_error("Failed to open file");

    while(!depthFile.eof())
    {
        std::string s, imgPath;
        std::getline(depthFile, s);
        if(s.empty() || s[0] == '#') continue;
        std::stringstream ss;
        ss << s;
        double timestamp;
        ss >> timestamp >> imgPath;
        timestamps.push_back(timestamp);
    }
    depthFile.close();

    poses.clear();
    std::string fileList = "/home/savuor/datasets/rgbd_dataset_freiburg1_xyz/groundtruth.txt";
    fstream posesFile(fileList);
    if(!posesFile.is_open())
        throw std::runtime_error("Failed to open file");

    while(!posesFile.eof())
    {
        std::string s;
        std::getline(posesFile, s);
        if(s.empty() || s[0] == '#') continue;
        std::stringstream ss;
        ss << s;
        double timestamp;
        float tx, ty, tz, qx, qy, qz, qw;
        ss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Matx33f m(1.f - 2.f*qy*qy - 2.f*qz*qz,       2.f*qx*qy - 2.f*qz*qw,       2.f*qx*qz + 2.f*qy*qw,
                        2.f*qx*qy + 2.f*qz*qw, 1.f - 2.f*qx*qx - 2.f*qz*qz,       2.f*qy*qz - 2.f*qx*qw,
                        2.f*qx*qz - 2.f*qy*qw,       2.f*qy*qz + 2.f*qx*qw, 1.f - 2.f*qx*qx - 2.f*qy*qy);

        Affine3f aff(m, Vec3f(tx, ty, tz));
        poses.push_back(TimePose(timestamp, aff));
    }
    posesFile.close();
}
static Affine3f debugPose()
{
    //DEBUG!
    Affine3f newPose;
    if(frame == 1)
    {
        double ts0 = timestamps[0];
        int elem0 = -1;
        for(size_t i = 0; i < poses.size()-1; i++)
        {
            if(poses[i  ].first < ts0 &&
               poses[i+1].first > ts0)
                elem0 = i;
        }
        if(elem0 < 0)
            throw std::runtime_error("You'd better write correct code");
        currentPose = poses[elem0].second;
    }

    double timestamp = timestamps[frame++];
    int elem = -1;
    for(size_t i = 0; i < poses.size()-1; i++)
    {
        if(poses[i  ].first < timestamp &&
           poses[i+1].first > timestamp)
            elem = i;
    }
    if(elem < 0)
        throw std::runtime_error("You'd better write correct code");
    Affine3f low = poses[elem].second, high = poses[elem+1].second;
    newPose = low; // can be interpolated

    Affine3f t = currentPose.inv() * newPose;

    currentPose = newPose;
    return t;
}
*/

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

    kinfu::KinFu kf(kinfu::KinFu::KinFuParams::defaultParams());

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
            window.spinOnce(1, true);

            //DEBUG!
    //        Image img = render();
    //        Mat cvt8u; convertScaleAbs(img, cvt8u, 255.);
    //        viz::Viz3d vimg("render");
    //        vimg.showWidget("showImage", viz::WImageOverlay(cvt8u, Rect(Point(), cvt8u.size())));
    //        vimg.spin();
    //        viz::Viz3d window = viz::getWindowByName("debug");
    //        window.showWidget("worldAxes", viz::WCoordinateSystem());
    //        window.showWidget("cube", viz::WCube(Vec3d::all(0), Vec3d::all(params.volumeSize)), params.volumePose);

    //        std::vector<Point3f> points;
    //        std::vector<uint8_t> colors;
    //        float voxelSize = (params.volumeSize/params.volumeDims);
    //        for(int x = 0; x < params.volumeDims; x++)
    //        {
    //            for(int y = 0; y < params.volumeDims; y++)
    //            {
    //                for(int z = 0; z < params.volumeDims; z++)
    //                {
    //                    Voxel& v = *(volume.volume + x*params.volumeDims*params.volumeDims + y*params.volumeDims + z);
    //                    Point3f pvox = params.volumePose * Point3f(x*voxelSize, y*voxelSize, z*voxelSize);
    //                    if(v.v != 1 && v.weight != 0)
    //                    //if(v.weight != 0)
    //                    {
    //                        points.push_back(pvox);
    //                        colors.push_back(255*(1-abs(v.v)));
    //                    }
    //                }
    //            }
    //        }
    //        viz::WCloud func(points, colors);
    //        window.showWidget("tsdf", func);

    //        int npyr = 0;
    //        viz::WCloud pts(frame.points[npyr], viz::Color::red());
    //        window.showWidget("ptsRaycast", pts);
    //        viz::WCloudNormals nrm(frame.points[npyr], frame.normals[npyr], 1, 0.02, viz::Color::gold());
    //        window.showWidget("nrmRaycast", nrm);

            //viz::WCloud ptsOrig(newFrame.points[npyr], viz::Color::green());
            //window.showWidget("ptsOrig", ptsOrig);
    //        window.setCamera(viz::Camera(params.intr.fx,
    //                                     params.intr.fy,
    //                                     params.intr.cx,
    //                                     params.intr.cy,
    //                                     params.frameSize));
    //        window.setViewerPose(pose);
    //        viz::WCameraPosition camPos(Vec2d(params.intr.fy/params.frameSize.height,
    //                                          params.intr.fx/params.frameSize.width));
    //        window.showWidget("camera", camPos, pose);

    //        std::cout << Mat(pose.matrix) << std::endl;
    //        window.spinOnce(1, true);
    //        window.spin();

        }

        kf.render(rendered);
        imshow("render", rendered);

        waitKey(10);
    }

    return 0;
}
