set(target "gtest")

FILE(GLOB hdrs "gtest/*.h")
FILE(GLOB srcs "gtest/*.cc")

ADD_LIBRARY(${target} STATIC ${srcs})

# FIXME: update this check (Kirill Kornyakov)
# if((CXX_GCC OR CXX_CLANG) AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    set(Pthread "-pthread")
# endif()
TARGET_LINK_LIBRARIES(${target} ${Pthread})
