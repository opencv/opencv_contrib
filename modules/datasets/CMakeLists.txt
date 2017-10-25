set(the_description "datasets framework")

# find HDF5 library
find_package(HDF5)
if (HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    add_definitions(-DHAVE_HDF5=1)
    set(HDF5_LIBRARY "hdf5")
    message(STATUS "datasets: HDF5:   Yes")
    message(STATUS "datasets: HDF5 libs: " ${HDF5_LIBRARIES})
    message(STATUS "datasets: HDF5 include: " ${HDF5_INCLUDE_DIRS})
else()
    set(HDF5_LIBRARIES "")
    set(HDF5_LIBRARY "")
    message(STATUS "datasets: HDF5:   NO")
endif()

ocv_define_module(datasets opencv_core opencv_imgcodecs opencv_ml opencv_flann OPTIONAL opencv_text WRAP python)

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4267) # flann, Win64

ocv_target_link_libraries(opencv_datasets LINK_PRIVATE ${HDF5_LIBRARIES})
if(BUILD_EXAMPLES)
    ocv_target_link_libraries(example_datasets_fr_adience_toh5 LINK_PRIVATE ${HDF5_LIBRARIES})
endif()
