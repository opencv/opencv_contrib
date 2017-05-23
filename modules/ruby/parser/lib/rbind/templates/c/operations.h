#ifndef <%= name %>
#define <%= name %>

#include "types.h"

<%= wrap_includes %>

#ifdef __cplusplus
extern "C"
{
#endif

// general rbind functions 
RBIND_EXPORTS const char* rbindGetLastError();
RBIND_EXPORTS bool rbindHasError();
RBIND_EXPORTS void rbindClearError();

<%= wrap_operations %>

#ifdef __cplusplus
}
#endif
#endif
