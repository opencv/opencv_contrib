
#include "conversions.hpp"
#include "operations.h"
<%= wrap_includes %>

char last_error_message[255] = {0};
const char* rbindGetLastError()
{
    return &last_error_message[0];
}

bool rbindHasError()
{
    return (*rbindGetLastError() != '\0');
}

void rbindClearError()
{
    last_error_message[0] = '\0';
}


<%= wrap_operations%>
