#define CL_TARGET_OPENCL_VERSION 120

constexpr unsigned long WIDTH = 1920;
constexpr unsigned long HEIGHT = 1080;
constexpr unsigned int DOWN_SCALE = 2;
constexpr bool OFFSCREEN = false;
constexpr int VA_HW_DEVICE_INDEX = 0;

#include "../common/subsystems.hpp"
#include <csignal>
#include <list>
#include <vector>
#include <cstdint>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

using std::cerr;
using std::endl;
using std::vector;
using std::list;
using std::string;

int current_max_points = 2000;
float current_stroke = 5.0f;

static bool done = false;
static void finish(int ignore) {
    std::cerr << endl;
    done = true;
}

void overdefine(const vector<cv::Point2f>& srcPts, vector<cv::Point2f>& dstPts, size_t minPoints) {
    assert(srcPts.size() > 1);
    list<cv::Point2f> copy;
    for(auto& pt : srcPts) {
        copy.push_back(pt);
    }

    off_t diff = minPoints - copy.size();
    while(diff > 0)  {
        auto it = copy.begin();
        for(size_t i = 0; i < copy.size() && diff > 0; i+=2) {
            const auto& first = *it;
            ++it;
            if(it == copy.end())
                it = copy.begin();
            const auto& second = *it;
            auto insertee = first;
            auto vector = second - first;
            vector.x /= 2.0;
            vector.y /= 2.0;
            insertee.x += vector.x;
            insertee.y += vector.y;
            --it;
            copy.insert(it, std::move(insertee));
            ++it;
            --diff;
        }
    }
    for(auto& pt : copy) {
        dstPts.push_back(pt);
    }
    assert(dstPts.size() >= minPoints);
}

void make_delaunay_mesh(const cv::Size &size, cv::Subdiv2D &subdiv, vector<cv::Point2f> &dstPoints) {
    vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<cv::Point2f> pt(3);
    cv::Rect rect(0, 0, size.width, size.height);

    for (size_t i = 0; i < triangleList.size(); i++) {
        cv::Vec6f t = triangleList[i];
        pt[0] = cv::Point2f(t[0], t[1]);
        pt[1] = cv::Point2f(t[2], t[3]);
        pt[2] = cv::Point2f(t[4], t[5]);

        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
            dstPoints.push_back(pt[0]);
            dstPoints.push_back(pt[1]);
            dstPoints.push_back(pt[2]);
        }
    }
}

void collect_delaunay_mesh_points(const int width, const int height, const std::vector<cv::Point2f> &inPoints, std::vector<cv::Point2f> &outPoints) {
    cv::Subdiv2D subdiv(cv::Rect(0, 0, width, height));
    subdiv.insert(inPoints);
    vector<cv::Point2f> triPoints;
    make_delaunay_mesh( { width, height }, subdiv, triPoints);

    for (size_t i = 0; i < triPoints.size(); i++) {
        outPoints.push_back(triPoints[i]);
    }
}

int main(int argc, char **argv) {
    signal(SIGINT, finish);
    using namespace kb;

    if (argc != 2) {
        std::cerr << "Usage: optflow <input-video-file>" << endl;
        exit(1);
    }

    va::init();
    cv::VideoCapture cap(argv[1], cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera" << endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter encoder("optflow.mkv", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(WIDTH, HEIGHT), {
            cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    if (!OFFSCREEN)
        x11::init();
    egl::init();
    gl::init();
    nvg::init();

    cerr << "VA Version: " << va::get_info() << endl;
    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();
    double lastFps = fps;

    cv::UMat frameBuffer, videoFrame, downScaled, background;
    cv::UMat foreground(HEIGHT, WIDTH, CV_8UC4, cv::Scalar::all(0));
    cv::UMat downPrevGrey, downNextGrey, downMaskGrey;
    vector<cv::Point2f> contourPoints, downNewPoints, downPrevPoints, downNextPoints, meshPoints, upPrevPoints, upNextPoints;
    cv::Ptr<cv::BackgroundSubtractor> bgSubtractor = cv::createBackgroundSubtractorMOG2(100, 32.0, false);
    std::vector<uchar> status;
    std::vector<float> err;
    vector<cv::Point2f> downHull, downOverdefined;
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    while (!done) {
        va::bind();
        cap >> videoFrame;
        if (videoFrame.empty())
            break;

        cv::resize(videoFrame, videoFrame, cv::Size(WIDTH, HEIGHT));
        cv::cvtColor(videoFrame, background, cv::COLOR_RGB2BGRA);
        cv::resize(videoFrame, downScaled, cv::Size(0, 0), 1.0 / DOWN_SCALE, 1.0 / DOWN_SCALE);
        cv::boxFilter(downScaled, downScaled, -1, cv::Size(5, 5), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
        cvtColor(downScaled, downNextGrey, cv::COLOR_RGB2GRAY);

        bgSubtractor->apply(downScaled, downMaskGrey);

        if (cv::countNonZero(downMaskGrey) < (WIDTH * HEIGHT * 0.1)) {
            int morph_size = 1;
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
            cv::morphologyEx(downMaskGrey, downMaskGrey, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
            findContours(downMaskGrey, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
            imshow("dmg", downMaskGrey);
            cv::waitKey(1);

            meshPoints.clear();
            for (const auto &c : contours) {
                contourPoints.clear();
                for (const auto &pt : c) {
                    contourPoints.push_back(pt);
                }
                collect_delaunay_mesh_points(downMaskGrey.cols, downMaskGrey.rows, contourPoints, meshPoints);
            }

            gl::bind();
            nvg::begin();
            nvg::clear();

            if (meshPoints.size() > 12) {
                cv::convexHull(meshPoints, downHull);
                double area = cv::contourArea(downHull);
                double density = (meshPoints.size() / area);
                current_max_points = density * 25000.0;
                int copyn = std::min(meshPoints.size(), (current_max_points - downPrevPoints.size()));
                std::random_shuffle(meshPoints.begin(), meshPoints.end());

                if (downPrevPoints.size() < current_max_points) {
                    std::copy(meshPoints.begin(), meshPoints.begin() + copyn, std::back_inserter(downPrevPoints));
                }

                if (downPrevGrey.empty()) {
                    downPrevGrey = downNextGrey.clone();
                }

                cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
                cv::calcOpticalFlowPyrLK(downPrevGrey, downNextGrey, downPrevPoints, downNextPoints, status, err, cv::Size(15, 15), 2, criteria, cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

                downNewPoints.clear();

                if (downPrevPoints.size() > 1 && downNextPoints.size() > 1) {
                    upNextPoints.clear();
                    upPrevPoints.clear();
                    for (cv::Point2f pt : downPrevPoints) {
                        upPrevPoints.push_back(pt *= float(DOWN_SCALE));
                    }

                    for (cv::Point2f pt : downNextPoints) {
                        upNextPoints.push_back(pt *= float(DOWN_SCALE));
                    }
                    overdefine(downHull, downOverdefined, 5);
                    cv::RotatedRect ellipse = cv::fitEllipse(downOverdefined);
                    double shortAxis = std::min(ellipse.size.width, ellipse.size.height) * float(DOWN_SCALE);
                    double wholeArea = WIDTH * HEIGHT;
                    current_stroke = 20.0 * sqrt(area / wholeArea);
                    std::cerr << current_stroke << ":" << sqrt(area / wholeArea) << endl;

                    using kb::nvg::vg;
                    nvgBeginPath(vg);
                    nvgStrokeWidth(vg, current_stroke);
                    nvgStrokeColor(vg, nvgHSLA(0.1, 1, 0.5, 48));

                    for (size_t i = 0; i < downPrevPoints.size(); i++) {
                        if (status[i] == 1 && err[i] < 0.005 && upNextPoints[i].y >= 0 && upNextPoints[i].x >= 0 && upNextPoints[i].y < HEIGHT && upNextPoints[i].x < WIDTH) {
                            downNewPoints.push_back(downNextPoints[i]);
                            double diffX = fabs(upNextPoints[i].x - upPrevPoints[i].x);
                            double diffY = fabs(upNextPoints[i].y - upPrevPoints[i].y);
                            double len = hypot(diffX, diffY);
                            if (len > 0 && len < shortAxis / 3.0) {
//                            std::cerr << err[i] << ":" << current_max_points << endl;
                                nvgMoveTo(vg, upNextPoints[i].x, upNextPoints[i].y);
                                nvgLineTo(vg, upPrevPoints[i].x, upPrevPoints[i].y);
                            }
                        }
                    }
                    nvgStroke(vg);

                }
                downPrevPoints = downNewPoints;
            }
            nvg::end();
        } else {
            cerr << "clear" << endl;
            gl::bind();
            nvg::begin();
            nvg::clear();
            nvg::end();
            foreground = cv::Scalar::all(0);
        }

        downPrevGrey = downNextGrey.clone();

        gl::acquire_from_gl(frameBuffer);

        cv::flip(frameBuffer, frameBuffer, 0);
        cv::addWeighted(foreground, 0.9, frameBuffer, 1.1, 0.0, foreground);
        cv::addWeighted(background, 1.0, foreground, 1.0, 0.0, frameBuffer);
        cv::flip(frameBuffer, frameBuffer, 0);
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);

        gl::release_to_gl(frameBuffer);

        va::bind();
        cv::flip(videoFrame, videoFrame, 0);
        encoder.write(videoFrame);

        if (x11::is_initialized()) {
            gl::bind();
            gl::blit_frame_buffer_to_screen();

            if (x11::window_closed()) {
                finish(0);
                break;
            }

            gl::swap_buffers();
        }

        //Measure FPS
        if (cnt % uint64(ceil(lastFps)) == 0) {
            int64 tick = cv::getTickCount();
            lastFps = tickFreq / ((tick - start + 1) / cnt);
            cerr << "FPS : " << lastFps << '\r';
            start = tick;
            cnt = 1;
        }

        ++cnt;
    }

    return 0;
}
