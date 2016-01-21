set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(HDF5)
if(HDF5_FOUND)
    set(HAVE_HDF5 1)
    message(STATUS "HDF5:   YES")
else()
    ocv_module_disable(hdf)
    message(STATUS "HDF5:   NO")
endif()

if(${HDF5_FOUND})
  include_directories(${HDF5_INCLUDE_DIRS})
endif()

set(the_description "Hierarchical Data Format I/O")
ocv_define_module(hdf opencv_core WRAP python)

if(${HDF5_FOUND})
  target_link_libraries(opencv_hdf ${HDF5_LIBRARIES})
endif()
