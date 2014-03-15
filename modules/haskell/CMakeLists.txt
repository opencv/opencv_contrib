set(the_description "Haskell Bindings")

if(NOT ANDROID AND NOT IOS)

if(CMAKE_HOST_WIN32)
    set(GHC_NAME ghc.exe)
    set(CABAL_NAME cabal.exe)
else()
    set(GHC_NAME ghc)
    set(CABAL_NAME cabal)
endif()

find_host_program(GHC_EXECUTABLE NAMES ${GHC_NAME}
    PATHS "/usr/bin" #TODO: Figure out where ghc/cabal are installed on other systems.
    NO_DEFAULT_PATH
)

find_host_program(GHC_EXECUTABLE NAMES ${GHC_NAME})

if(GHC_EXECUTABLE)
    execute_process(COMMAND ${GHC_EXECUTABLE} --version
        RESULT_VARIABLE GHC_ERROR_LEVEL
        OUTPUT_VARIABLE GHC_VERSION_FULL
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(GHC_ERROR_LEVEL)
        unset(GHC_EXECUTABLE)
        unset(GHC_EXECUTABLE CACHE)
    else()
        string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" GHC_VERSION "${GHC_VERSION_FULL}")
        set(GHC_VERSION "${GHC_VERSION}" CACHE INTERNAL "Detected ghc version")
    endif()
endif()

find_host_program(CABAL_EXECUTABLE NAMES ${CABAL_NAME}
    PATHS "/usr/bin" 
    NO_DEFAULT_PATH
)

find_host_program(CABAL_EXECUTABLE NAMES ${CABAL_NAME})

if(CABAL_EXECUTABLE)
    execute_process(COMMAND ${CABAL_EXECUTABLE} --version
        RESULT_VARIABLE CABAL_ERROR_LEVEL
        OUTPUT_VARIABLE CABAL_VERSION_FULL
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(CABAL_ERROR_LEVEL)
        unset(CABAL_EXECUTABLE)
        unset(CABAL_EXECUTABLE CACHE)
    else()
        string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" CABAL_VERSION "${CABAL_VERSION_FULL}")
        set(CABAL_VERSION "${CABAL_VERSION}" CACHE INTERNAL "Detected cabal version")
    endif()
endif()

if(GHC_EXECUTABLE AND CABAL_EXECUTABLE)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/OpenCV")
set(opencv_hdrs
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/base.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/types.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/persistence.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/utility.hpp"
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/mat.hpp"
    "${OPENCV_MODULE_opencv_flann_LOCATION}/include/opencv2/flann/miniflann.hpp"
    "${OPENCV_MODULE_opencv_imgproc_LOCATION}/include/opencv2/imgproc.hpp"
    "${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/background_segm.hpp"
    "${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/tracking.hpp"
    "${OPENCV_MODULE_opencv_photo_LOCATION}/include/opencv2/photo.hpp"
    "${OPENCV_MODULE_opencv_highgui_LOCATION}/include/opencv2/highgui.hpp"
    "${OPENCV_MODULE_opencv_ml_LOCATION}/include/opencv2/ml.hpp"
    "${OPENCV_MODULE_opencv_features2d_LOCATION}/include/opencv2/features2d.hpp"
    "${OPENCV_MODULE_opencv_calib3d_LOCATION}/include/opencv2/calib3d.hpp"
    "${OPENCV_MODULE_opencv_objdetect_LOCATION}/include/opencv2/objdetect.hpp"
    "${OPENCV_MODULE_opencv_contrib_LOCATION}/include/opencv2/contrib.hpp"
    "${OPENCV_MODULE_opencv_superres_LOCATION}/include/opencv2/superres.hpp"
    "${OPENCV_MODULE_opencv_bioinspired_LOCATION}/include/opencv2/bioinspired.hpp"
)

set(opencv_hdr_dirs
    "${OPENCV_MODULE_opencv_core_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_flann_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_imgproc_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_video_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_photo_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_highgui_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_ml_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_features2d_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_calib3d_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_objdetect_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_contrib_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_stitching_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_superres_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_ts_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_bioinspired_LOCATION}/include/"
    "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/"
    )
if(HAVE_opencv_nonfree)
    list(APPEND opencv_hdrs     "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree/features2d.hpp"
                                "${OPENCV_MODULE_opencv_nonfree_LOCATION}/include/opencv2/nonfree.hpp")
endif()

set(haskell_generated_sources
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Types.hsc"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Funcs.hsc"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Consts.hsc"
)

set(haskell_sources
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Mat.hsc"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV/Point.hsc"
)

set(haskell_build_files
    "${CMAKE_CURRENT_BINARY_DIR}/OpenCV.hs"
    "${CMAKE_CURRENT_BINARY_DIR}/Setup.hs"
    "${CMAKE_CURRENT_BINARY_DIR}/opencv.cabal"
    "${CMAKE_CURRENT_BINARY_DIR}/LICENSE"
)    

# Copy necessary files over to the build directory.
foreach(file_i ${haskell_sources})
    get_filename_component(name_i ${file_i} NAME)
    if(NOT EXISTS ${file_i})
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/OpenCV/${name_i} ${CMAKE_CURRENT_BINARY_DIR}/OpenCV COPYONLY)
    endif()
endforeach()

foreach(file_i ${haskell_build_files})
    get_filename_component(name_i ${file_i} NAME)
    if(NOT EXISTS ${file_i})
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/${name_i} ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)
    endif()
endforeach()

# generate ffi wrappers
add_custom_command(
    OUTPUT ${haskell_generated_sources}
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/src/genhsc.py" "${CMAKE_CURRENT_BINARY_DIR}/OpenCV" ${opencv_hdrs}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/genhsc.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/hdr_parser.py
    DEPENDS ${opencv_hdrs}
)

add_custom_target(haskell ALL
    COMMAND ${CMAKE_COMMAND} -E echo Generating Haskell bindings if necessary.
    DEPENDS ${haskell_generated_sources}
    DEPENDS ${haskell_sources}
    DEPENDS ${haskell_build_files}
    DEPENDS opencv_c
)

set(includes)
foreach(hdr ${opencv_hdr_dirs})
    list(APPEND includes "--extra-include-dirs=${hdr} ")
endforeach()

install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR} ${CABAL_EXECUTABLE} install \"--global\" \"--only-dependencies\")")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR} ${CABAL_EXECUTABLE} install \"--global\" \"--with-gcc=${CMAKE_CXX_COMPILER}\" \"--extra-include-dirs=${CMAKE_BINARY_DIR}/include\" ${includes} \"--extra-lib-dirs=${CMAKE_BINARY_DIR}/lib\" \"--gcc-options=${CMAKE_C_FLAGS}${CMAKE_CXX_FLAGS}\")")
endif()
endif()
