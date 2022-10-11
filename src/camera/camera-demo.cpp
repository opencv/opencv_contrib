#define CL_TARGET_OPENCL_VERSION 220

const long unsigned int WIDTH = 1920;
const long unsigned int HEIGHT = 1080;
double FPS;
constexpr double OFFSCREEN = false;

#include "../tetra/subsystems.hpp"

using std::cerr;
using std::endl;

cv::ocl::OpenCLExecutionContext VA_CONTEXT;

int main(int argc, char **argv) {
    using namespace kb;

    va::init_va();
    /*
     * The OpenCLExecutionContext for VAAPI needs to be copied right after init_va().
     * Now everytime you want to do VAAPI interop first bind the context.
     */
    VA_CONTEXT = cv::ocl::OpenCLExecutionContext::getCurrent();

    cv::VideoCapture cap("output.mp4", cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, 0,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    FPS = cap.get(cv::CAP_PROP_FPS);
    std::cerr << "FPS: " << FPS << std::endl;

    cerr << "VA Version: " << va::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    cv::UMat frameBuffer(HEIGHT, WIDTH, CV_8UC4, cv::Scalar::all(0));
    cv::UMat videoFrame;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();

    while (true) {
        cap >> videoFrame;
        if (videoFrame.empty()) {
            cerr << "End of stream. Exiting" << endl;
            break;
        }

        //Measure FPS
        if (cnt % uint64(FPS) == 0) {
            int64 tick = cv::getTickCount();
            cerr << "FPS : " << tickFreq / ((tick - start + 1) / cnt) << '\r';
            start = tick;
            cnt = 1;
        }

        ++cnt;
    }

    return 0;
}
