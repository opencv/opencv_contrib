 if(IOS OR WINRT OR ANDROID OR APPLE OR WIN32 OR (NOT HAVE_CANN))
   ocv_module_disable(cannops)
 endif()

set(the_description "Ascend-accelerated Operations on Matrices")

ocv_add_module(cannops opencv_core WRAP python)
ocv_module_include_directories(${CANN_INCLUDE_DIRS})
ocv_glob_module_sources()
ocv_install_used_external_targets(${CANN_LIBRARIES})
ocv_create_module(${CANN_LIBRARIES})

ocv_include_directories(${CMAKE_SOURCE_DIR}/modules/ts/include)

ocv_add_accuracy_tests(DEPENDS_ON opencv_cannops)
ocv_add_perf_tests(DEPENDS_ON opencv_cannops)
ocv_add_samples(opencv_cannops)

# compile ascnedc kernels.
add_subdirectory(ascendc_kernels)
ocv_include_directories(${CMAKE_BINARY_DIR}/include/ascendc_kernels)
ocv_target_link_libraries(opencv_cannops PRIVATE ascendc_kernels)
ocv_target_link_libraries(opencv_test_cannops PRIVATE ascendc_kernels)
