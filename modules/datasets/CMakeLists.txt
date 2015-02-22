set(the_description "datasets framework")

#ocv_define_module(datasets opencv_core opencv_face opencv_ml opencv_flann opencv_text)

#ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4267) # flann, Win64

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

ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4267) # flann, Win64

ocv_add_module(datasets opencv_core opencv_face opencv_ml opencv_flann opencv_text)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module(${HDF5_LIBRARIES})

ocv_add_samples(${HDF5_LIBRARY})

