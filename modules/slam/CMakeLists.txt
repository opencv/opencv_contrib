set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)
set(the_description "SLAM")

set(BOW_FRAMEWORK "FBoW" CACHE STRING "DBoW2 or FBoW")
set_property(CACHE BOW_FRAMEWORK PROPERTY STRINGS "DBoW2" "FBoW")
set(SLAM_REQUIRED_G2O_VERSION "20230223_git")

set(SLAM_DEPS_OK TRUE)
set(SLAM_DISABLE_MSG "Module opencv_slam disabled because the following dependencies are not found:")

if(NOT HAVE_EIGEN)
    set(SLAM_DEPS_OK FALSE)
    set(SLAM_DISABLE_MSG "${SLAM_DISABLE_MSG} Eigen3>=3.3")
elseif(DEFINED EIGEN_VERSION AND EIGEN_VERSION VERSION_LESS "3.3")
    set(SLAM_DEPS_OK FALSE)
    set(SLAM_DISABLE_MSG "${SLAM_DISABLE_MSG} Eigen3>=3.3")
endif()

find_package(yaml-cpp 0.6 QUIET)
if(NOT yaml-cpp_FOUND)
    set(SLAM_DEPS_OK FALSE)
    set(SLAM_DISABLE_MSG "${SLAM_DISABLE_MSG} yaml-cpp>=0.6")
endif()

find_package(nlohmann_json 3.2 QUIET)
set(SLAM_NLOHMANN_JSON_LIB "")
set(SLAM_NLOHMANN_JSON_INCLUDE_DIR "")
if(TARGET nlohmann_json::nlohmann_json)
    set(SLAM_NLOHMANN_JSON_LIB nlohmann_json::nlohmann_json)
elseif(TARGET nlohmann_json)
    set(SLAM_NLOHMANN_JSON_LIB nlohmann_json)
elseif(DEFINED nlohmann_json_INCLUDE_DIRS)
    set(SLAM_NLOHMANN_JSON_INCLUDE_DIR ${nlohmann_json_INCLUDE_DIRS})
elseif(DEFINED NLOHMANN_JSON_INCLUDE_DIRS)
    set(SLAM_NLOHMANN_JSON_INCLUDE_DIR ${NLOHMANN_JSON_INCLUDE_DIRS})
else()
    find_path(SLAM_NLOHMANN_JSON_INCLUDE_DIR NAMES nlohmann/json.hpp)
endif()

if(NOT SLAM_NLOHMANN_JSON_LIB AND NOT SLAM_NLOHMANN_JSON_INCLUDE_DIR)
    set(SLAM_DEPS_OK FALSE)
    set(SLAM_DISABLE_MSG "${SLAM_DISABLE_MSG} nlohmann_json")
endif()

if(SLAM_NLOHMANN_JSON_INCLUDE_DIR)
    set(SLAM_NLOHMANN_JSON_INCLUDE_DIR ${SLAM_NLOHMANN_JSON_INCLUDE_DIR} CACHE PATH "Path to nlohmann_json headers")
endif()

find_package(g2o QUIET
             COMPONENTS
             g2o::core
             g2o::stuff
             g2o::types_sba
             g2o::types_sim3
             g2o::solver_dense
             g2o::solver_eigen
             g2o::solver_csparse
             g2o::csparse_extension
             OPTIONAL_COMPONENTS
             g2o::csparse)
if(NOT g2o_FOUND)
    set(SLAM_DEPS_OK FALSE)
    set(SLAM_DISABLE_MSG "${SLAM_DISABLE_MSG} g2o")
elseif(DEFINED g2o_VERSION AND NOT g2o_VERSION STREQUAL SLAM_REQUIRED_G2O_VERSION)
    message(WARNING "opencv_slam: recommended g2o version is ${SLAM_REQUIRED_G2O_VERSION}, found ${g2o_VERSION}")
endif()

if(BOW_FRAMEWORK STREQUAL "DBoW2")
    find_package(DBoW2 QUIET)
    if(NOT DBoW2_FOUND)
        set(SLAM_DEPS_OK FALSE)
        set(SLAM_DISABLE_MSG "${SLAM_DISABLE_MSG} DBoW2")
    endif()
elseif(BOW_FRAMEWORK STREQUAL "FBoW")
    find_package(fbow QUIET)
    set(SLAM_FBOW_LIBRARY "")
    set(SLAM_FBOW_INCLUDE_DIRS "")

    if(TARGET fbow::fbow)
        set(SLAM_FBOW_LIBRARY fbow::fbow)
    elseif(TARGET fbow)
        set(SLAM_FBOW_LIBRARY fbow)
    elseif(DEFINED fbow_LIBS)
        set(SLAM_FBOW_LIBRARY ${fbow_LIBS})
    elseif(DEFINED FBOW_LIBRARIES)
        set(SLAM_FBOW_LIBRARY ${FBOW_LIBRARIES})
    endif()

    if(DEFINED fbow_INCLUDE_DIRS)
        set(SLAM_FBOW_INCLUDE_DIRS ${fbow_INCLUDE_DIRS})
    elseif(DEFINED FBOW_INCLUDE_DIRS)
        set(SLAM_FBOW_INCLUDE_DIRS ${FBOW_INCLUDE_DIRS})
    endif()

    if(NOT fbow_FOUND AND NOT SLAM_FBOW_LIBRARY)
        set(SLAM_DEPS_OK FALSE)
        set(SLAM_DISABLE_MSG "${SLAM_DISABLE_MSG} FBoW")
    endif()
else()
    message(FATAL_ERROR "opencv_slam: Invalid BoW framework: ${BOW_FRAMEWORK}")
endif()

find_package(Pangolin QUIET)
if(Pangolin_FOUND)
    message(STATUS "opencv_slam: Pangolin found (optional GUI enabled)")
else()
    message(STATUS "opencv_slam: Pangolin not found (optional GUI disabled)")
endif()

if(NOT SLAM_DEPS_OK)
    message(STATUS "${SLAM_DISABLE_MSG}")
    ocv_module_disable(slam)
else()
    ocv_add_module(slam
        opencv_core
        opencv_imgcodecs
        opencv_videoio
        opencv_features2d
        opencv_calib3d
        opencv_highgui
        OPTIONAL opencv_objdetect opencv_aruco)

    find_package(Threads REQUIRED)
    find_package(OpenMP QUIET)
    find_package(SQLite3 REQUIRED)

    set(USE_GTSAM OFF CACHE BOOL "Enable gtsam")
    if(USE_GTSAM)
        find_package(GTSAM REQUIRED)
    endif()

    if(TARGET g2o::csparse)
        set(CXSPARSE_LIBRARIES g2o::csparse)
    else()
        find_package(CXSparse QUIET)
        if(CXSPARSE_FOUND)
            include_directories(${CXSPARSE_INCLUDE_DIRS})
        endif()
        if(NOT SUITESPARSE_FOUND)
            if(MSVC)
                find_package(SuiteSparse CONFIG QUIET)
            else()
                find_package(SuiteSparse QUIET)
            endif()
            if(SUITESPARSE_FOUND)
                include_directories(${SUITESPARSE_INCLUDE_DIRS})
            endif()
        endif()
    endif()

    if(BOW_FRAMEWORK STREQUAL "DBoW2")
        set(BoW_LIBRARY ${DBoW2_LIBS})
    elseif(BOW_FRAMEWORK STREQUAL "FBoW")
        set(BoW_LIBRARY ${SLAM_FBOW_LIBRARY})
    endif()

    include(FindPackageHandleStandardArgs)
    set(USE_CUDA_EFFICIENT_DESCRIPTORS OFF CACHE BOOL "Enable cuda_efficient_features")
    find_path(cuda_efficient_features_INCLUDE_DIR NAMES cuda_efficient_features.hpp)
    find_library(cuda_efficient_features_LIBRARY NAMES cuda_efficient_features)
    find_package_handle_standard_args(cuda_efficient_features DEFAULT_MSG cuda_efficient_features_LIBRARY cuda_efficient_features_INCLUDE_DIR)

    if(USE_CUDA_EFFICIENT_DESCRIPTORS)
        if(cuda_efficient_features_FOUND)
            find_package(CUDAToolkit REQUIRED)
        else()
            message(WARNING "opencv_slam: USE_CUDA_EFFICIENT_DESCRIPTORS=ON but cuda_efficient_features was not found; disabling")
            set(USE_CUDA_EFFICIENT_DESCRIPTORS OFF)
        endif()
    endif()

    file(GLOB_RECURSE SLAM_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
    file(GLOB_RECURSE SLAM_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")

    if(NOT USE_GTSAM)
        list(FILTER SLAM_SOURCES EXCLUDE REGEX ".*_gtsam\\.cpp$")
        list(FILTER SLAM_HEADERS EXCLUDE REGEX ".*_gtsam\\.hpp$")
        list(FILTER SLAM_HEADERS EXCLUDE REGEX "/internal_gtsam/")
    endif()

    ocv_set_module_sources(
        HEADERS ${SLAM_HEADERS}
        SOURCES ${SLAM_SOURCES}
    )

    ocv_module_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(SLAM_NLOHMANN_JSON_INCLUDE_DIR)
        ocv_include_directories(${SLAM_NLOHMANN_JSON_INCLUDE_DIR})
    endif()
    if(SLAM_FBOW_INCLUDE_DIRS)
        ocv_include_directories(${SLAM_FBOW_INCLUDE_DIRS})
    endif()

    ocv_create_module()
    set_target_properties(${the_module} PROPERTIES OUTPUT_NAME cv_slam)
    target_compile_features(${the_module} PUBLIC cxx_std_17)

    if(USE_CUDA_EFFICIENT_DESCRIPTORS)
        ocv_target_compile_definitions(${the_module} PUBLIC USE_CUDA_EFFICIENT_DESCRIPTORS)
    endif()

    if(USE_GTSAM)
        ocv_target_compile_definitions(${the_module} PUBLIC USE_GTSAM)
    endif()

    set(USE_OPENMP OFF CACHE BOOL "Use OpenMP")
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_compile_options(${the_module} PRIVATE ${OpenMP_CXX_FLAGS})
        ocv_target_compile_definitions(${the_module} PUBLIC USE_OPENMP)
    elseif(USE_OPENMP)
        message(WARNING "opencv_slam: USE_OPENMP=ON but OpenMP was not found; disabling")
        set(USE_OPENMP OFF)
    endif()

    set(USE_SSE_ORB OFF CACHE BOOL "Enable SSE3 instruction for ORB extraction")
    if(USE_SSE_ORB)
        target_compile_options(${the_module} PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-msse3>)
        ocv_target_compile_definitions(${the_module} PUBLIC USE_SSE_ORB)
    endif()

    set(USE_SSE_FP_MATH OFF CACHE BOOL "Enable SSE instruction for floating-point operation")
    if(USE_SSE_FP_MATH)
        target_compile_options(${the_module} PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mfpmath=sse>)
    endif()

    if(BOW_FRAMEWORK STREQUAL "DBoW2")
        ocv_target_compile_definitions(${the_module} PUBLIC USE_DBOW2)
    endif()

    set(USE_ARUCO ON CACHE BOOL "Enable aruco")
    if(USE_ARUCO)
        ocv_target_compile_definitions(${the_module} PUBLIC USE_ARUCO)
    endif()

    option(USE_ARUCO_NANO "Compile with aruco nano support" OFF)
    if(USE_ARUCO_NANO)
        ocv_target_compile_definitions(${the_module} PUBLIC USE_ARUCO_NANO)
    endif()

    set(ENABLE_TRACE_LEVEL_LOG OFF CACHE BOOL "Enable trace level log")
    if(ENABLE_TRACE_LEVEL_LOG)
        ocv_target_compile_definitions(${the_module} PUBLIC CV_LOG_LEVEL=CV_LOG_LEVEL_VERBOSE)
    else()
        ocv_target_compile_definitions(${the_module} PUBLIC CV_LOG_LEVEL=CV_LOG_LEVEL_DEBUG)
    endif()

    set(DETERMINISTIC OFF CACHE BOOL "Use deterministic code")
    if(DETERMINISTIC)
        ocv_target_compile_definitions(${the_module} PUBLIC DETERMINISTIC)
    endif()

    set(USE_TRACY OFF CACHE BOOL "Enable tracy")
    if(USE_TRACY)
        find_package(Tracy REQUIRED)
        ocv_target_compile_definitions(${the_module} PRIVATE TRACY_ENABLE=1)
        ocv_target_link_libraries(${the_module} PRIVATE Tracy::TracyClient)
    endif()

    if(Pangolin_FOUND)
        ocv_target_compile_definitions(${the_module} PUBLIC USE_PANGOLIN)
    endif()

    ocv_target_link_libraries(${the_module}
        PUBLIC
        Threads::Threads
        Eigen3::Eigen
        yaml-cpp
        ${SQLite3_LIBRARIES}
        g2o::core
        g2o::stuff
        g2o::types_sba
        g2o::types_sim3
        g2o::solver_dense
        g2o::solver_eigen
        g2o::solver_csparse
        g2o::csparse_extension
        ${CXSPARSE_LIBRARIES}
        ${SUITESPARSE_LIBRARIES}
        ${BoW_LIBRARY}
        "$<$<BOOL:${USE_GTSAM}>:gtsam>"
        "$<$<BOOL:${USE_CUDA_EFFICIENT_DESCRIPTORS}>:${cuda_efficient_features_LIBRARY};opencv_cudafilters;opencv_cudawarping>"
        PRIVATE
        ${LAPACK_LIBRARIES}
        "$<$<BOOL:${USE_CUDA_EFFICIENT_DESCRIPTORS}>:CUDA::cudart;CUDA::cublas>")

    if(SLAM_NLOHMANN_JSON_LIB)
        ocv_target_link_libraries(${the_module} PUBLIC ${SLAM_NLOHMANN_JSON_LIB})
    endif()

    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        ocv_target_link_libraries(${the_module} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()
