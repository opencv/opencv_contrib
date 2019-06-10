include_directories(.)
aux_source_directory(common COMMON_SRC)
aux_source_directory(. APRILTAG_SRCS)

if(MSVC)
    ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4018 /wd4244)
else()
    ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-parameter -Wstrict-prototypes -Wmissing-prototypes)
endif()

add_library(apriltag STATIC ${APRILTAG_SRCS} ${COMMON_SRC})

target_include_directories(apriltag PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>/apriltag")
