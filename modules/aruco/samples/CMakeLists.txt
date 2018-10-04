# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(opencv_aruco_samples)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(calibrate_camera_charuco calibrate_camera_charuco.cpp)
add_executable(calibrate_camera calibrate_camera.cpp)
add_executable(create_board_charuco create_board_charuco.cpp)
add_executable(create_board create_board.cpp)
add_executable(create_diamond create_diamond.cpp)
add_executable(create_marker create_marker.cpp)
add_executable(detect_board_charuco detect_board_charuco.cpp)
add_executable(detect_board detect_board.cpp)
add_executable(detect_diamonds detect_diamonds.cpp)
add_executable(detect_markers detect_markers.cpp)

# Link your application with OpenCV libraries
target_link_libraries(calibrate_camera_charuco ${OpenCV_LIBS})
target_link_libraries(calibrate_camera ${OpenCV_LIBS})
target_link_libraries(create_board_charuco ${OpenCV_LIBS})
target_link_libraries(create_board ${OpenCV_LIBS})
target_link_libraries(create_diamond ${OpenCV_LIBS})
target_link_libraries(create_marker ${OpenCV_LIBS})
target_link_libraries(detect_board_charuco ${OpenCV_LIBS})
target_link_libraries(detect_board ${OpenCV_LIBS})
target_link_libraries(detect_diamonds ${OpenCV_LIBS})
target_link_libraries(detect_markers ${OpenCV_LIBS})
