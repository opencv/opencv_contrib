#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE) && defined KFUSION_API_EXPORTS
  #define KF_EXPORTS __declspec(dllexport)
#else
  #define KF_EXPORTS
#endif
