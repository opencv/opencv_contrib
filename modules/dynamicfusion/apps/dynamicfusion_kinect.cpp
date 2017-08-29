#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <io/capture.hpp>
using namespace kfusion;

struct KinFuApp
{
    static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
    {
        KinFuApp& kinfu = *static_cast<KinFuApp*>(pthis);

        if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
            return;

        if(event.code == 't' || event.code == 'T')
            kinfu.take_cloud(*kinfu.kinfu_);

        if(event.code == 'i' || event.code == 'I')
            kinfu.interactive_mode_ = !kinfu.interactive_mode_;
    }

    KinFuApp(OpenNISource& source) : exit_ (false), interactive_mode_(false), capture_ (source), pause_(false)
    {
        KinFuParams params = KinFuParams::default_params();
        kinfu_ = KinFu::Ptr( new KinFu(params) );

        capture_.setRegistration(true);

        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);
    }

    static void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        //cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0/4000);
        cv::imshow("Depth", display);
    }

    void show_raycasted(KinFu& kinfu)
    {
        const int mode = 3;
        if (interactive_mode_)
            kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
            kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        cv::imshow("Scene", view_host_);
    }

    void take_cloud(KinFu& kinfu)
    {
//        cv::Mat cloud_host = kinfu.tsdf().get_cloud_host();
        cv::Mat normal_host =  kinfu.tsdf().get_normal_host();
        cv::Mat warp_host =  kinfu.getWarp().getNodesAsMat();

//        viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
//        viz.showWidget("cloud_normals", cv::viz::WCloudNormals(cloud_host, normal_host, 64, 0.05, cv::viz::Color::blue()));
        viz1.showWidget("warp_field", cv::viz::WCloud(warp_host));
    }

    bool execute()
    {
        KinFu& kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;
            for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
            {
                bool has_frame = capture_.grab(depth, image);
                if (!has_frame)
                    return std::cout << "Can't grab" << std::endl, false;
                depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

                {
                    SampledScopeTime fps(time_ms); (void)fps;
                    has_image = kinfu(depth_device_);
                }

                if (has_image)
                    show_raycasted(kinfu);

                show_depth(depth);
                cv::imshow("Image", image);

                if (!interactive_mode_)
                {
                    viz.setViewerPose(kinfu.getCameraPose());
                    viz1.setViewerPose(kinfu.getCameraPose());
                }

                int key = cv::waitKey(pause_ ? 0 : 3);
                take_cloud(kinfu);
                switch(key)
                {
                    case 't': case 'T' : take_cloud(kinfu); break;
                    case 'i': case 'I' : interactive_mode_ = !interactive_mode_; break;
                    case 27: exit_ = true; break;
                    case 32: pause_ = !pause_; break;
                }

                //exit_ = exit_ || i > 100;
                viz.spinOnce(3, true);
                viz1.spinOnce(3, true);
            }
        return true;
    }

    bool pause_ /*= false*/;
    bool exit_, interactive_mode_;
    KinFu::Ptr kinfu_;
    cv::viz::Viz3d viz;
    cv::viz::Viz3d viz1;

    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    OpenNISource& capture_;

};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    int device = 0;
    cuda::setDevice (device);
    cuda::printShortCudaDeviceInfo (device);

    if(cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, -1;

    KinFuApp *app;

    OpenNISource capture;
    capture.open(argv[1]);
    app = new KinFuApp(capture);

    // executing
    try { app->execute (); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    delete app;
    return 0;
}
