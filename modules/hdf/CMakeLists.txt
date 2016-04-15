set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
  # windows cmake internal lookups are broken for now
  # will lookup for headers and shared libs given HDF_DIR env
  find_path(HDF5_INCLUDE_DIRS hdf5.h HINTS "$ENV{HDF5_DIR}\\..\\include")
  find_library(HDF5_C_LIBRARY NAMES hdf5 HINTS "$ENV{HDF5_DIR}\\..\\lib")
  if(HDF5_INCLUDE_DIRS AND HDF5_C_LIBRARY)
    set(HDF5_FOUND "YES")
    set(HDF5_LIBRARIES ${HDF5_C_LIBRARY})
    mark_as_advanced(HDF5_LIBRARIES)
    mark_as_advanced(HDF5_C_LIBRARY)
    mark_as_advanced(HDF5_INCLUDE_DIRS)
    add_definitions(-DH5_BUILT_AS_DYNAMIC_LIB -D_HDF5USEDLL_)
  else()
     set(HDF5_FOUND "NO")
  endif()
else()
  find_package(HDF5)
endif()

if(HDF5_FOUND)
    set(HAVE_HDF5 1)
    message(STATUS "HDF5:   YES")
else()
    ocv_module_disable(hdf)
    message(STATUS "HDF5:   NO")
endif()

set(the_description "Hierarchical Data Format I/O")
ocv_define_module(hdf opencv_core WRAP python)

if(HDF5_FOUND)
  include_directories(${HDF5_INCLUDE_DIRS})
endif()

if(HDF5_FOUND)
  target_link_libraries(opencv_hdf ${HDF5_LIBRARIES})
endif()
