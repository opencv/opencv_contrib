

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include <time.h>
#include <vector>
#include <iostream>
#include <opencv2/ximgproc.hpp>



using namespace cv;

#define MARK_RADIUS 15

static int globalMouseX;
static int globalMouseY;
static bool globalMouseClick = false;
static bool glb_mouse_left = false;

static bool mouseClick;
static bool mouseLeft;
static int mouseX;
static int mouseY;
cv::Mat mat_input_reference;
cv::Mat mat_input_confidence;


static void mouseCallback(int event, int x, int y, int flags, void* param);
void drawTrajectoryByReference(cv::Mat* img);
cv::Mat copyGlayForRGB(cv::Mat gray, cv::Mat color);


const String keys =
    "{help h usage ?     |                | print this message                                                }"
    "{sigma_spatial      |8               | parameter of post-filtering                                       }"
    "{sigma_luma         |8               | parameter of post-filtering                                       }"
    "{sigma_chroma       |8               | parameter of post-filtering                                       }"
    ;



int main(int argc, char* argv[])
{

#ifdef HAVE_EIGEN
    // CommandLineParser parser(argc,argv,keys);
    // parser.about("Disparity Filtering Demo");
    // if (parser.has("help"))
    // {
    //     parser.printMessage();
    //     return 0;
    // }
    //
    // String img = parser.get<String>(0);
    // double sigma_spatial  = parser.get<double>("sigma_spatial");
    // double sigma_luma  = parser.get<double>("sigma_luma");
    // double sigma_chroma  = parser.get<double>("sigma_chroma");


    float filtering_time;

    cv::Mat reference = cv::imread(argv[1],1);
    cv::Mat input = cv::imread(argv[1],0);
    // cv::Mat target = cv::imread(argv[2],1);
    cv::Mat target;

    float sigma_spatial = float(atof(argv[2]));
    float sigma_luma = float(atof(argv[3]));
    float sigma_chroma = float(atof(argv[4]));

    std::cout << "reference:" << reference.cols<<"x"<< reference.rows<< std::endl;


    cv::Mat mat_gray;
    cv::cvtColor(reference, mat_gray, cv::COLOR_BGR2GRAY);
    target = copyGlayForRGB(mat_gray, reference);

    cv::namedWindow("draw", cv::WINDOW_AUTOSIZE);
    cv::imshow("draw", target);
    cv::setMouseCallback("draw", mouseCallback, (void *)&target);
    mat_input_reference = reference.clone();
    mat_input_confidence = 0*cv::Mat::ones(mat_gray.size(),mat_gray.type());
    // mat_input_confidence = mat_gray;
    int show_count = 0;
    while (1)
    {
            mouseX = globalMouseX;
            mouseY = globalMouseY;
            mouseClick = globalMouseClick;
            mouseLeft = glb_mouse_left;


        if (mouseClick)
        {
            drawTrajectoryByReference(&target);

            if(show_count%5==0)
            {
                cv::Mat target_temp;
                filtering_time = (double)getTickCount();
                cv::cvtColor(target, target_temp, cv::COLOR_BGR2YCrCb);

                std::vector<cv::Mat> src_channels;
                std::vector<cv::Mat> dst_channels;

                cv::split(target_temp,src_channels);

                cv::Mat result1 = cv::Mat(input.size(),input.type());
                cv::Mat result2 = cv::Mat(input.size(),input.type());

                dst_channels.push_back(src_channels[0]);
                cv::ximgproc::fastBilateralSolverFilter(input,src_channels[1],mat_input_confidence,result1,sigma_spatial,sigma_luma,sigma_chroma);
                dst_channels.push_back(result1);
                cv::ximgproc::fastBilateralSolverFilter(input,src_channels[2],mat_input_confidence,result2,sigma_spatial,sigma_luma,sigma_chroma);
                dst_channels.push_back(result2);

                cv::merge(dst_channels,target_temp);
                cv::cvtColor(target_temp, target_temp, cv::COLOR_YCrCb2BGR);
                filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
                std::cout << "solver time: " << filtering_time << "ms" << std::endl;

                cv::imshow("draw", target_temp);
            }
            show_count++;
        }
        if (cv::waitKey(2) == 27)
            break;
    }
    cv::cvtColor(target, target, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> src_channels;
    std::vector<cv::Mat> dst_channels;

    cv::split(target,src_channels);

    cv::Mat result1 = cv::Mat(input.size(),input.type());
    cv::Mat result2 = cv::Mat(input.size(),input.type());

    filtering_time = (double)getTickCount();

    dst_channels.push_back(src_channels[0]);
    cv::ximgproc::fastBilateralSolverFilter(input,src_channels[1],mat_input_confidence,result1,sigma_spatial,sigma_luma,sigma_chroma);
    dst_channels.push_back(result1);
    cv::ximgproc::fastBilateralSolverFilter(input,src_channels[2],mat_input_confidence,result2,sigma_spatial,sigma_luma,sigma_chroma);
    dst_channels.push_back(result2);

    cv::merge(dst_channels,target);
    cv::cvtColor(target, target, cv::COLOR_YCrCb2BGR);

    filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
    std::cout << "solver time: " << filtering_time << "ms" << std::endl;



    // cv::equalizeHist(result, result);
    cv::imshow("input",input);
    cv::imshow("output",target);

    cv::waitKey(0);
#else
    std::cout << "Do not find eigen, please build with eigen by set WITH_EIGEN=ON" << '\n';
#endif

    return 0;
}


static void mouseCallback(int event, int x, int y, int, void*)
{
    switch (event)
    {
        case cv::EVENT_MOUSEMOVE:
            if (globalMouseClick)
            {
                globalMouseX = x;
                globalMouseY = y;
            }
            break;

        case cv::EVENT_LBUTTONDOWN:
            globalMouseClick = true;
            globalMouseX = x;
            globalMouseY = y;
            break;

        case cv::EVENT_LBUTTONUP:
            glb_mouse_left = true;
            globalMouseClick = false;
            break;
    }
}

void drawTrajectoryByReference(cv::Mat* img)
{
    int i, j;
    uchar red, green, blue;
    int y, x;
    int r = MARK_RADIUS;
    int r2 = r * r;
    uchar* colorPix;

    y = mouseY - r;
    for(i=-r; i<r+1 ; i++, y++)
    {
        x = mouseX - r;
        colorPix = mat_input_reference.ptr<uchar>(y, x);
        for(j=-r; j<r+1; j++, x++)
        {
            if(i*i + j*j > r2)
            {
                colorPix += mat_input_reference.channels();
                continue;
            }

            if(y<0 || y>=mat_input_reference.rows || x<0 || x>=mat_input_reference.cols)
            {
                break;
            }

            blue = *colorPix;
            colorPix++;
            green = *colorPix;
            colorPix++;
            red = *colorPix;
            colorPix++;
            cv::circle(*img, cv::Point2d(x, y), 0.1, cv::Scalar(blue, green, red), -1);
            // mat_input_confidence.at<uchar>(x,y) = (blue + green + red)/3;
            mat_input_confidence.at<uchar>(y,x) = 255;
        }
    }
}

cv::Mat copyGlayForRGB(cv::Mat gray, cv::Mat color)
{
    int y, x, c;
    uchar* gray_pix;
    uchar* colorPix;
    cv::Mat ret = color.clone();
    gray_pix = gray.ptr<uchar>(0, 0);
    colorPix = ret.ptr<uchar>(0, 0);

    for(y=0; y<gray.rows; y++)
    {
        for(x=0; x<gray.cols; x++)
        {
            for(c=0; c<color.channels(); c++)
            {
                *colorPix = *gray_pix;
                colorPix++;
            }
            gray_pix++;
        }
    }
    return ret;
}
