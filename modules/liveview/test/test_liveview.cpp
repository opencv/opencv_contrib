#include "opencv2/liveview.hpp"
#include "opencv2/ts.hpp"

namespace opencv_test {
namespace {

TEST(LiveView, CreateServer)
{
    cv::Ptr<cv::liveview::Server> server = cv::liveview::createServer();
    ASSERT_FALSE(server.empty());
    EXPECT_FALSE(server->isRunning());
}

} // namespace
} // namespace opencv_test
