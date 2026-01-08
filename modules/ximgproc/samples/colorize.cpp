
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include <time.h>
#include <vector>
#include <iostream>
#include <opencv2/ximgproc.hpp>



using namespace cv;

#ifdef HAVE_EIGEN

#define MARK_RADIUS 5
#define PALLET_RADIUS 100
int max_width = 1280;
int max_height = 720;

static int globalMouseX;
static int globalMouseY;
static int selected_r;
static int selected_g;
static int selected_b;
static bool globalMouseClick = false;
static bool glb_mouse_left = false;
static bool drawByReference = false;
static bool mouseDraw = false;
static bool mouseClick;
static bool mouseLeft;
static int mouseX;
static int mouseY;

cv::Mat mat_draw;
cv::Mat mat_input_gray;
cv::Mat mat_input_reference;
cv::Mat mat_input_confidence;
cv::Mat mat_pallet(PALLET_RADIUS*2,PALLET_RADIUS*2,CV_8UC3);


static void mouseCallback(int event, int x, int y, int flags, void* param);
void drawTrajectoryByReference(cv::Mat& img);
double module(Point pt);
double distance(Point pt1, Point pt2);
double cross(Point pt1, Point pt2);
double angle(Point pt1, Point pt2);
int inCircle(Point p, Point c, int r);
void createPlate(Mat &im1, int radius);


#endif

const String keys =
    "{help h usage ?     |                | print this message                                                }"
    "{@image             |                | input image                                                       }"
    "{sigma_spatial      |8               | parameter of post-filtering                                       }"
    "{sigma_luma         |8               | parameter of post-filtering                                       }"
    "{sigma_chroma       |8               | parameter of post-filtering                                       }"
    "{dst_path           |None            | optional path to save the resulting colorized image               }"
    "{dst_raw_path       |None            | optional path to save drawed image before filtering               }"
    "{draw_by_reference  |false           | optional flag to use color image as reference                     }"
    ;



int main(int argc, char* argv[])
{

    CommandLineParser parser(argc,argv,keys);
    parser.about("fastBilateralSolverFilter Demo");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

#ifdef HAVE_EIGEN

    String img = parser.get<String>(0);
    double sigma_spatial  = parser.get<double>("sigma_spatial");
    double sigma_luma  = parser.get<double>("sigma_luma");
    double sigma_chroma  = parser.get<double>("sigma_chroma");
    String dst_path = parser.get<String>("dst_path");
    String dst_raw_path = parser.get<String>("dst_raw_path");
    drawByReference = parser.get<bool>("draw_by_reference");

    mat_input_reference = cv::imread(img, IMREAD_COLOR);
    if (mat_input_reference.empty())
    {
        std::cerr << "input image '" << img << "' could not be read !" << std::endl << std::endl;
        parser.printMessage();
        return 1;
    }

    cvtColor(mat_input_reference, mat_input_gray, COLOR_BGR2GRAY);

    if(mat_input_gray.cols > max_width)
    {
        double scale = float(max_width) / float(mat_input_gray.cols);
        cv::resize(mat_input_reference, mat_input_reference, cv::Size(), scale, scale);
        cv::resize(mat_input_gray, mat_input_gray, cv::Size(), scale, scale);
    }

    if(mat_input_gray.rows > max_height)
    {
        double scale = float(max_height) / float(mat_input_gray.rows);
        cv::resize(mat_input_reference, mat_input_reference, cv::Size(), scale, scale);
        cv::resize(mat_input_gray, mat_input_gray, cv::Size(), scale, scale);
    }


    float filtering_time;
    std::cout << "mat_input_reference:" << mat_input_reference.cols<<"x"<< mat_input_reference.rows<< std::endl;
    std::cout << "please select a color from the palette, by clicking into that," << std::endl;
    std::cout << "  then select a coarse region in the image to be coloured." << std::endl;
    std::cout << "  press 'escape' to see the final coloured image." << std::endl;


    cv::Mat mat_gray;
    cv::cvtColor(mat_input_reference, mat_gray, cv::COLOR_BGR2GRAY);

    cv::Mat target = mat_input_reference.clone();
    cvtColor(mat_gray, mat_input_reference, COLOR_GRAY2BGR);

    cv::namedWindow("draw", cv::WINDOW_AUTOSIZE);

    // construct pallet
    createPlate(mat_pallet, PALLET_RADIUS);
    selected_b = 0;
    selected_g = 0;
    selected_r = 0;

    cv::Mat mat_show(target.rows,target.cols+PALLET_RADIUS*2,CV_8UC3);
    cv::Mat color_select(target.rows-mat_pallet.rows,PALLET_RADIUS*2,CV_8UC3,cv::Scalar(selected_b, selected_g, selected_r));
    target.copyTo(Mat(mat_show,Rect(0,0,target.cols,target.rows)));
    mat_pallet.copyTo(Mat(mat_show,Rect(target.cols,0,mat_pallet.cols,mat_pallet.rows)));
    color_select.copyTo(Mat(mat_show,Rect(target.cols,PALLET_RADIUS*2,color_select.cols,color_select.rows)));

    cv::imshow("draw", mat_show);
    cv::setMouseCallback("draw", mouseCallback, (void *)&mat_show);
    mat_input_confidence = 0*cv::Mat::ones(mat_gray.size(),mat_gray.type());

    int show_count = 0;
    while (1)
    {
            mouseX = globalMouseX;
            mouseY = globalMouseY;
            mouseClick = globalMouseClick;
            mouseLeft = glb_mouse_left;


        if (mouseClick)
        {
            drawTrajectoryByReference(target);

            if(show_count%5==0)
            {
                cv::Mat target_temp(target.size(),target.type());
                filtering_time = static_cast<float>(getTickCount());
                if(mouseDraw)
                {
                    cv::cvtColor(target, target_temp, cv::COLOR_BGR2YCrCb);
                    std::vector<cv::Mat> src_channels;
                    std::vector<cv::Mat> dst_channels;

                    cv::split(target_temp,src_channels);

                    cv::Mat result1 = cv::Mat(mat_input_gray.size(),mat_input_gray.type());
                    cv::Mat result2 = cv::Mat(mat_input_gray.size(),mat_input_gray.type());

                    dst_channels.push_back(mat_input_gray);
                    cv::ximgproc::fastBilateralSolverFilter(mat_input_gray,src_channels[1],mat_input_confidence,result1,sigma_spatial,sigma_luma,sigma_chroma);
                    dst_channels.push_back(result1);
                    cv::ximgproc::fastBilateralSolverFilter(mat_input_gray,src_channels[2],mat_input_confidence,result2,sigma_spatial,sigma_luma,sigma_chroma);
                    dst_channels.push_back(result2);

                    cv::merge(dst_channels,target_temp);
                    cv::cvtColor(target_temp, target_temp, cv::COLOR_YCrCb2BGR);
                }
                else
                {
                  target_temp = target.clone();
                }
                filtering_time = static_cast<float>(((double)getTickCount() - filtering_time)/getTickFrequency());
                std::cout << "solver time: " << filtering_time << "s" << std::endl;

                cv::Mat color_selected(target_temp.rows-mat_pallet.rows,PALLET_RADIUS*2,CV_8UC3,cv::Scalar(selected_b, selected_g, selected_r));
                target_temp.copyTo(Mat(mat_show,Rect(0,0,target_temp.cols,target_temp.rows)));
                mat_pallet.copyTo(Mat(mat_show,Rect(target_temp.cols,0,mat_pallet.cols,mat_pallet.rows)));
                color_selected.copyTo(Mat(mat_show,Rect(target_temp.cols,PALLET_RADIUS*2,color_selected.cols,color_selected.rows)));
                cv::imshow("draw", mat_show);
            }
            show_count++;
        }
        if (cv::waitKey(2) == 27)
            break;
    }
    mat_draw = target.clone();
    cv::cvtColor(target, target, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> src_channels;
    std::vector<cv::Mat> dst_channels;

    cv::split(target,src_channels);

    cv::Mat result1 = cv::Mat(mat_input_gray.size(),mat_input_gray.type());
    cv::Mat result2 = cv::Mat(mat_input_gray.size(),mat_input_gray.type());

    filtering_time = static_cast<float>(getTickCount());

    // dst_channels.push_back(src_channels[0]);
    dst_channels.push_back(mat_input_gray);
    cv::ximgproc::fastBilateralSolverFilter(mat_input_gray,src_channels[1],mat_input_confidence,result1,sigma_spatial,sigma_luma,sigma_chroma);
    dst_channels.push_back(result1);
    cv::ximgproc::fastBilateralSolverFilter(mat_input_gray,src_channels[2],mat_input_confidence,result2,sigma_spatial,sigma_luma,sigma_chroma);
    dst_channels.push_back(result2);

    cv::merge(dst_channels,target);
    cv::cvtColor(target, target, cv::COLOR_YCrCb2BGR);

    filtering_time = static_cast<float>(((double)getTickCount() - filtering_time)/getTickFrequency());
    std::cout << "solver time: " << filtering_time << "s" << std::endl;



    cv::imshow("mat_draw",mat_draw);
    cv::imshow("output",target);

    if(dst_path!="None")
    {
        imwrite(dst_path,target);
    }
    if(dst_raw_path!="None")
    {
        imwrite(dst_raw_path,mat_draw);
    }

    cv::waitKey(0);



#else
    std::cout << "Can not find eigen, please build with eigen by set WITH_EIGEN=ON" << '\n';
#endif

    return 0;
}


#ifdef HAVE_EIGEN
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

void drawTrajectoryByReference(cv::Mat& img)
{
    int i, j;
    uchar red, green, blue;
    float gray;
    int y, x;
    int r = MARK_RADIUS;
    int r2 = r * r;
    uchar* colorPix;
    uchar* grayPix;

    if(mouseY < PALLET_RADIUS*2 && img.cols <= mouseX && mouseX < img.cols+PALLET_RADIUS*2)
    {
        colorPix = mat_pallet.ptr<uchar>(mouseY, mouseX - img.cols);
        // colorPix = mat_pallet.ptr<uchar>(mouseY, mouseX);
        selected_b = *colorPix;
        colorPix++;
        selected_g = *colorPix;
        colorPix++;
        selected_r = *colorPix;
        colorPix++;
        std::cout << "x y:("<<mouseX<<"," <<mouseY<< " rgb_select:("<< selected_r<<","<<selected_g<<","<<selected_b<<")" << '\n';
    }
    else
    {
        mouseDraw = true;
        y = mouseY - r;
        for(i=-r; i<r+1 ; i++, y++)
        {
            x = mouseX - r;
            colorPix = mat_input_reference.ptr<uchar>(y, x);
            grayPix = mat_input_gray.ptr<uchar>(y, x);
            for(j=-r; j<r+1; j++, x++)
            {
                if(i*i + j*j > r2)
                {
                    colorPix += mat_input_reference.channels();
                    grayPix += mat_input_gray.channels();
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
                gray = *grayPix;
                grayPix++;
                mat_input_confidence.at<uchar>(y,x) = 255;
                float draw_y = 0.229f*(float(selected_r)) + 0.587f*(float(selected_g)) + 0.114f*(float(selected_b));
                int draw_b = int(float(selected_b)*(gray/draw_y));
                int draw_g = int(float(selected_g)*(gray/draw_y));
                int draw_r = int(float(selected_r)*(gray/draw_y));

                if(drawByReference)
                {
                    cv::circle(img, cv::Point2d(x, y), 1, cv::Scalar(blue, green, red), -1);
                }
                else
                {
                    cv::circle(img, cv::Point2d(x, y), 1, cv::Scalar(draw_b, draw_g, draw_r), -1);
                }
            }
        }
    }
}

double module(Point pt)
{
	return sqrt((double)pt.x*pt.x + pt.y*pt.y);
}

double distance(Point pt1, Point pt2)
{
	int dx = pt1.x - pt2.x;
	int dy = pt1.y - pt2.y;
	return sqrt((double)dx*dx + dy*dy);
}

double cross(Point pt1, Point pt2)
{
	return pt1.x*pt2.x + pt1.y*pt2.y;
}

double angle(Point pt1, Point pt2)
{
	return acos(cross(pt1, pt2) / (module(pt1)*module(pt2) + DBL_EPSILON));
}

// p or c is the center
int inCircle(Point p, Point c, int r)
{
	int dx = p.x - c.x;
	int dy = p.y - c.y;
	return dx*dx + dy*dy <= r*r ? 1 : 0;

}

//draw the hsv-plate
void createPlate(Mat &im1, int radius)
{
	Mat hsvImag(Size(radius << 1, radius << 1), CV_8UC3, Scalar(0, 0, 255));
	int w = hsvImag.cols;
	int h = hsvImag.rows;
	int cx = w >> 1;
	int cy = h >> 1;
	Point pt1(cx, 0);

	for (int j = 0; j < w; j++)
	{
		for (int i = 0; i < h; i++)
		{
			Point pt2(j - cx, i - cy);
			if (inCircle(Point(0, 0), pt2, radius))
			{
				int theta = static_cast<int>(angle(pt1, pt2) * 180 / CV_PI);
				if (i > cx)
				{
					theta = -theta + 360;
				}
				hsvImag.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(theta / 2);
				hsvImag.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(module(pt2) / cx * 255);
				hsvImag.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}


	cvtColor(hsvImag, im1, COLOR_HSV2BGR);
}


#endif
