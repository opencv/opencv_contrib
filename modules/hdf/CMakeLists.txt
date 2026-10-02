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
  if(NOT CMAKE_CROSSCOMPILING) # iOS build should not reuse OSX package
    find_package(HDF5)
  endif()
endif()

if(NOT HDF5_FOUND)
    ocv_module_disable(hdf) # no return
endif()

# See https://github.com/opencv/opencv_contrib/issues/3858
# If HDF5_PREFER_PARALLEL is enabled, find_package(HDF5) sets HDF5_IS_PARALLEL=ON.
if(HDF5_IS_PARALLEL)
  find_package(MPI)
  if(NOT MPI_FOUND)
    message(STATUS "Module opencv_hdf disabled because the following dependency is not found: MPI")
    ocv_module_disable(hdf) # no return
  endif()
else()
  unset(MPI_CXX_LIBRARIES)
  unset(MPI_CXX_INCLUDE_DIRS)
endif()

set(HAVE_HDF5 1)
ocv_warnings_disable(CMAKE_CXX_FLAGS -Winvalid-offsetof)

set(the_description "Hierarchical Data Format I/O")
ocv_define_module(hdf opencv_core WRAP python)
ocv_target_link_libraries(${the_module} ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES})
ocv_include_directories(${HDF5_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})
