set(JULIA_TEST_PROXY ${CMAKE_CURRENT_BINARY_DIR}/test.proxy)
file(REMOVE ${JULIA_TEST_PROXY})

# generate
# call the python executable to generate the julia gateways
add_custom_command(
    OUTPUT ${JULIA_TEST_PROXY}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/testsuite.jl ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/test_feature2d.jl ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/test_objdetect.jl ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/test_imgproc.jl ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/test_mat.jl ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/test_dnn.jl ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${JULIA_TEST_PROXY}
    COMMENT "Building Julia tests"
)

# targets# opencv_julia_sources --> opencv_julia

add_custom_target(opencv_test_julia ALL DEPENDS ${JULIA_TEST_PROXY})
add_dependencies(opencv_test_julia ${the_module})

message(STATUS "Placing Julia tests in ${CMAKE_CURRENT_BINARY_DIR}")
# run the julia test suite
add_test(NAME opencv_test_julia
    COMMAND ${Julia_EXECUTABLE} "testsuite.jl"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
