#include <opencv2/highgui.hpp>
#include <opencv2/plot.hpp>
#include <iostream>

using namespace cv;

int main()
{
    Mat data_x(1, 50, CV_64F);
    Mat data_y(1, 50, CV_64F);

    for (int i = 0; i < 50; i++)
    {
        data_x.at<double>(0, i) = (i - 25);
        data_y.at<double>(0, i) = (i - 25)*(i - 25)*(i - 25);
    }

    std::cout << "data_x : " << data_x << std::endl;
    std::cout << "data_y : " << data_y << std::endl;

    Mat plot_result;

    Ptr<plot::Plot2d> plot = plot::Plot2d::create(data_x, data_y);
    plot->render(plot_result);

    imshow("default orientation", plot_result);

    plot = plot::Plot2d::create(data_x, data_y,true);
    plot->render(plot_result);

    imshow("inverted orientation", plot_result);
    waitKey();

    return 0;
}
