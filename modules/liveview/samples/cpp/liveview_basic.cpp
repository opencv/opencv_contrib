#include <opencv2/core.hpp>
#include <opencv2/liveview.hpp>

#include <iostream>

int main()
{
    cv::Ptr<cv::liveview::Server> view = cv::liveview::createServer("127.0.0.1", 0);
    view->start();
    view->publish("sample", cv::Mat(240, 320, CV_8UC3, cv::Scalar(40, 120, 220)));
    std::cout << "LiveView URL: " << view->url() << std::endl;
    std::cout << "Press Enter to stop." << std::endl;
    std::cin.get();
    return 0;
}
