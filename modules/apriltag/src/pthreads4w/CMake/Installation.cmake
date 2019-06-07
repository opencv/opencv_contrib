#Install macro for pthreads4w libraries
MACRO (PTHREADS4W_INSTALL_LIB name)

if(NOT BUILD_SHARED_LIBS)
    ocv_install_target(${name} EXPORT OpenCVModules
    ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

ENDMACRO (PTHREADS4W_INSTALL_LIB)
