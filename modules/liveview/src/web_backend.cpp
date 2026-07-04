#include "web_backend.hpp"

namespace cv {
namespace liveview {

Ptr<WebBackend> createWebBackend()
{
#if defined(HAVE_LIVEVIEW_HTTP_CIVETWEB)
    return createCivetWebBackend();
#elif defined(HAVE_LIVEVIEW_HTTP_BOOST)
    return createBoostWebBackend();
#elif defined(HAVE_LIVEVIEW_HTTP_MONGOOSE)
    return createMongooseWebBackend();
#else
    CV_Error(Error::StsError, "No LiveView web backend was selected");
#endif
}

} // namespace liveview
} // namespace cv
