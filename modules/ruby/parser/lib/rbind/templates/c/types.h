#ifndef <%= name %>
#define <%= name %>

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined CVAPI_EXPORTS
#  define RBIND_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define RBIND_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define RBIND_EXPORTS
#endif

#include <cstddef>

<%= wrap_includes %>

#ifdef __cplusplus
extern "C"
{
#endif

<%= wrap_types%>

#ifdef __cplusplus
}
#endif
#endif
