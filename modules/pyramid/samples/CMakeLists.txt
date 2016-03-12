CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(name "PhaseBasedVideoMagnification")
project(pyramid_cpp_samples)

#SET(OpenCV_DIR /path/to/your/opencv/installation)

# packages
find_package(OpenCV REQUIRED) # http://opencv.org

# probably you should loop through the sample files here
add_executable(phase_based_video_magnify phase_based_video_magnify.cpp)
target_link_libraries(phase_based_video_magnify opencv_core opencv_pyramid opencv_imgproc opencv_highgui)

