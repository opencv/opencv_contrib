

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include <time.h>
#include <vector>
#include <iostream>
#include <opencv2/ximgproc.hpp>


#ifdef HAVE_EIGEN

using namespace cv;

#define MARK_RADIUS 15

static int glb_mouse_x;
static int glb_mouse_y;
static bool glb_mouse_click = false;
static bool glb_mouse_left = false;

static bool mouse_click;
static bool mouse_left;
static int mouse_x;
static int mouse_y;
cv::Mat mat_input_reference;
cv::Mat mat_input_confidence;


static void mouseCallback(int event, int x, int y, int flags, void* param)
{
    switch (event)
    {
        case cv::EVENT_MOUSEMOVE:
            if (glb_mouse_click)
            {
                glb_mouse_x = x;
                glb_mouse_y = y;
            }
            break;

        case cv::EVENT_LBUTTONDOWN:
            glb_mouse_click = true;
            glb_mouse_x = x;
            glb_mouse_y = y;
            break;

        case cv::EVENT_LBUTTONUP:
            glb_mouse_left = true;
            glb_mouse_click = false;
            break;
    }
}

void draw_Trajectory_Byreference(cv::Mat* img)
{
    int i, j;
    uchar red, green, blue;
    int y, x;
    int r = MARK_RADIUS;
    int r2 = r * r;
    uchar* color_pix;

    y = mouse_y - r;
    for(i=-r; i<r+1 ; i++, y++)
    {
        x = mouse_x - r;
        color_pix = mat_input_reference.ptr<uchar>(y, x);
        for(j=-r; j<r+1; j++, x++)
        {
            if(i*i + j*j > r2)
            {
                color_pix += mat_input_reference.channels();
                continue;
            }

            if(y<0 || y>=mat_input_reference.rows || x<0 || x>=mat_input_reference.cols)
            {
                break;
            }

            blue = *color_pix;
            color_pix++;
            green = *color_pix;
            color_pix++;
            red = *color_pix;
            color_pix++;
            cv::circle(*img, cv::Point2d(x, y), 0.1, cv::Scalar(blue, green, red), -1);
            // mat_input_confidence.at<uchar>(x,y) = (blue + green + red)/3;
            mat_input_confidence.at<uchar>(y,x) = 255;
        }
    }
}

cv::Mat copy_GlaychForRGBch(cv::Mat gray, cv::Mat color)
{
    int y, x, c;
    uchar* gray_pix;
    uchar* color_pix;
    cv::Mat ret = color.clone();
    gray_pix = gray.ptr<uchar>(0, 0);
    color_pix = ret.ptr<uchar>(0, 0);

    for(y=0; y<gray.rows; y++)
    {
        for(x=0; x<gray.cols; x++)
        {
            for(c=0; c<color.channels(); c++)
            {
                *color_pix = *gray_pix;
                color_pix++;
            }
            gray_pix++;
        }
    }
    return ret;
}




const String keys =
    "{help h usage ?     |                | print this message                                                }"
    "{sigma_spatial      |8               | parameter of post-filtering                                       }"
    "{sigma_luma         |8               | parameter of post-filtering                                       }"
    "{sigma_chroma       |8               | parameter of post-filtering                                       }"
    ;



int main(int argc, char** argv)
{
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
    target = copy_GlaychForRGBch(mat_gray, reference);

    cv::namedWindow("draw", cv::WINDOW_AUTOSIZE);
    cv::imshow("draw", target);
    cv::setMouseCallback("draw", mouseCallback, (void *)&target);
    mat_input_reference = reference.clone();
    mat_input_confidence = 0*cv::Mat::ones(mat_gray.size(),mat_gray.type());
    // mat_input_confidence = mat_gray;
    int show_count = 0;
    while (1)
    {
            mouse_x = glb_mouse_x;
            mouse_y = glb_mouse_y;
            mouse_click = glb_mouse_click;
            mouse_left = glb_mouse_left;


        if (mouse_click)
        {
            draw_Trajectory_Byreference(&target);

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


    return 0;
}

#endif
