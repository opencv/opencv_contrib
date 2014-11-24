CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(name "facerec")
project(facerec_cpp_samples)

# set OpenCV_DIR to opencv's subdirectory containing OpenCVConfig.cmake:
# SET(OpenCV_DIR </path/to/your/opencv/installation>/share/OpenCV)

# alternatively, you can specify it when running cmake:
# cmake . -DOpenCV_DIR=</path/to/your/opencv/installation>/share/OpenCV

# packages
find_package(OpenCV REQUIRED) # http://opencv.org

include_directories(${OpenCV_INCLUDE_DIRS})

# probably you should loop through the sample files here
add_executable(facerec_demo facerec_demo.cpp)
target_link_libraries(facerec_demo opencv_core opencv_face opencv_imgproc opencv_highgui opencv_imgcodecs)

add_executable(facerec_video facerec_video.cpp)
target_link_libraries(facerec_video opencv_face opencv_core opencv_imgproc opencv_highgui opencv_objdetect)

add_executable(facerec_eigenfaces facerec_eigenfaces.cpp)
target_link_libraries(facerec_eigenfaces opencv_face opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

add_executable(facerec_fisherfaces facerec_fisherfaces.cpp)
target_link_libraries(facerec_fisherfaces opencv_face opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

add_executable(facerec_lbph facerec_lbph.cpp)
target_link_libraries(facerec_lbph opencv_face opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
