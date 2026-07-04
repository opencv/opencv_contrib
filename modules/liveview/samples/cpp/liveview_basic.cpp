#include <opencv2/core.hpp>
#include <opencv2/liveview.hpp>

int main()
{
    cv::Ptr<cv::liveview::Server> view = cv::liveview::createServer();
    (void)view;
    return 0;
}
