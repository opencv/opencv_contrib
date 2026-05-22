include_directories(${CMAKE_SOURCE_DIR}/kfusion/include)
file(GLOB SourceFiles
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(utils_tests ${SourceFiles})
target_link_libraries(
        utils_tests
        ${GTEST_BOTH_LIBRARIES}
        ${OpenCV_LIBS}
        kfusion
)

target_include_directories(
        utils_tests PUBLIC
        ${CMAKE_SOURCE_DIR}/kfusion/include
        ${OpenCV_INCLUDE_DIRS}
)