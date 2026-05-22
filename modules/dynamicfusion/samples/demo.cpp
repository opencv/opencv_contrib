#include <iostream>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/dynamicfusion.hpp>
#include <opencv2/dynamicfusion/cuda/device_array.hpp>
using namespace cv;
struct DynamicFusionApp
{
    static void KeyboardCallback(const viz::KeyboardEvent& event, void* pthis)
    {
        DynamicFusionApp& kinfu = *static_cast<DynamicFusionApp*>(pthis);

        if(event.action != viz::KeyboardEvent::KEY_DOWN)
            return;

        if(event.code == 't' || event.code == 'T')
            kinfu.show_warp(*kinfu.kinfu_);

        if(event.code == 'i' || event.code == 'I')
            kinfu.interactive_mode_ = !kinfu.interactive_mode_;
    }

    DynamicFusionApp(std::string dir) : exit_ (false), interactive_mode_(false), pause_(false), directory(true), dir_name(dir)
    {
        kfusion::KinFuParams params = kfusion::KinFuParams::default_params_dynamicfusion();
        kinfu_ = kfusion::KinFu::Ptr( new kfusion::KinFu(params) );


        viz::WCube cube(Vec3d::all(0), Vec3d(params.volume_size), true, viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);

    }
    static void show_depth(const Mat& depth)
    {
        Mat display;
        depth.convertTo(display, CV_8U, 255.0/4000);
        viz::imshow("Depth", display);
    }

    void show_raycasted(kfusion::KinFu& kinfu)
    {
        const int mode = 3;
        if (interactive_mode_)
            kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
            kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        viz::imshow("Scene", view_host_);
    }

    void show_warp(kfusion::KinFu &kinfu)
    {
        Mat warp_host =  kinfu.getWarp().getNodesAsMat();
        viz1.showWidget("warp_field", viz::WCloud(warp_host));
    }

    bool execute()
    {
        kfusion::KinFu& dfusion = *kinfu_;
        Mat depth, image;
        double time_ms = 0;
        bool has_image = false;
        std::vector<String> depths;             // store paths,
        std::vector<String> images;             // store paths,
        glob(dir_name + "/depth", depths);
        glob(dir_name + "/color", images);

        std::sort(depths.begin(), depths.end());
        std::sort(images.begin(), images.end());

        for (int i = 0; i < depths.size() && !exit_ && !viz.wasStopped(); i++) {
            image = imread(images[i], CV_LOAD_IMAGE_COLOR);
            depth = imread(depths[i], CV_LOAD_IMAGE_ANYDEPTH);
            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

            {
                kfusion::SampledScopeTime fps(time_ms);
                (void) fps;
                has_image = dfusion(depth_device_);
            }

            if (has_image)
                show_raycasted(dfusion);

            show_depth(depth);
            viz::imshow("Image", image);

            if (!interactive_mode_) {
                viz.setViewerPose(dfusion.getCameraPose());
                viz1.setViewerPose(dfusion.getCameraPose());
            }

            int key = waitKey(pause_ ? 0 : 3);
            show_warp(dfusion);
            switch (key) {
                case 't':
                case 'T' :
                    show_warp(dfusion);
                    break;
                case 'i':
                case 'I' :
                    interactive_mode_ = !interactive_mode_;
                    break;
                case 27:
                    exit_ = true;
                    break;
                case 32:
                    pause_ = !pause_;
                    break;
            }

            //exit_ = exit_ || i > 100;
            viz.spinOnce(3, true);
            viz1.spinOnce(3, true);
        }
        return true;
    }

    bool pause_ /*= false*/;
    bool exit_, interactive_mode_, directory;
    std::string dir_name;
    kfusion::KinFu::Ptr kinfu_;
    viz::Viz3d viz;
    viz::Viz3d viz1;

    Mat view_host_;
    kfusion::cuda::Image view_device_;
    kfusion::cuda::Depth depth_device_;


};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    int device = 0;
    kfusion::cuda::setDevice (device);
    kfusion::cuda::printShortCudaDeviceInfo(device);

    if(kfusion::cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, -1;

    DynamicFusionApp *app;
    app = new DynamicFusionApp(argv[1]);

    // executing
    try { app->execute (); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    delete app;
    return 0;
}
