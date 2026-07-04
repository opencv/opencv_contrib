#ifndef OPENCV_LIVEVIEW_ROUTE_MATCHER_HPP
#define OPENCV_LIVEVIEW_ROUTE_MATCHER_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace liveview {

enum class RouteKind
{
    Index,
    Health,
    ChannelsJson,
    Snapshot,
    Mjpeg,
    Unknown
};

struct RouteMatch
{
    RouteKind kind;
    String channel;
};

RouteMatch matchRoute(const String& method, const String& path);
bool isValidChannelName(const String& name);
String typeToString(int type);
String jsonEscape(const String& value);
String htmlEscape(const String& value);

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_ROUTE_MATCHER_HPP
